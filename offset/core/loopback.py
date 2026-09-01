"""The loopback: code offset is running may call offset back.

The agent writes a script and runs it through the `bash` tool.  Inside that
subprocess it is outside everything — outside the workspace boundary, outside
the approval policy, outside the transcript — so a script that needs to read a
file the user has consented to, search the repository the agent has indexed, or
say what it is doing halfway through, has no way to ask.  It either
reimplements the capability badly or it goes without.  This module publishes a
socket for exactly the lifetime of one tool call so it can ask instead.

Four decisions carry the design.

*It is not a privilege escalation.*  A `call` arriving over the loopback goes
through the very same `Runtime` the model's own calls go through, holding the
very same `Approval` object, so a tool that would have prompted the human still
prompts, a tool that was denied earlier is still denied, and a tool above the
mode's threshold with nobody attached to ask is **refused**.  Failing open here
would turn "run this script" into "run anything", which is the whole reason the
approval policy exists.  The runtime is not bypassed even for the bookkeeping:
argument validation, the write snapshot hook and the per-call deadline are all
inherited by construction rather than reimplemented.

*The secret travels in the environment, never in argv.*  `/proc/<pid>/cmdline`
is world-readable on Linux, so a token on a command line is a token published
to every account on the machine for as long as the process lives.  The child is
handed `OFFSET_LOOPBACK_SOCKET` and `OFFSET_LOOPBACK_TOKEN` and nothing else,
and no token file is written: unlike the editor bridge there is no discovery
problem to solve, because the parent is the one spawning the child.

*Recursion is counted, and the count can only go up.*  A tool reached over the
loopback runs with `OFFSET_LOOPBACK_DEPTH` one higher than the server that
invoked it, and — this is the part that matters — with the socket and token
variables *blanked*.  A child inherits its parent's environment, so leaving the
grandparent's address visible would let a script three hops deep reconnect to
the depth-zero server and reset the counter, which is an unbounded loop that
looks like progress.  Past `MAX_DEPTH` the server refuses to bind at all, so
the capability is withdrawn while the tool itself still runs.

*No socket outlives its call.*  The socket lives in a `0o700` directory of its
own, created per server rather than at a fixed path so two concurrent tool
calls cannot collide, and `serve()` is a context manager whose `finally`
removes the directory.  A stale socket file with a live token in somebody's
memory is a capability nobody is watching.

The framing is deliberately identical to the editor bridge
(`offset/core/bridge.py`): newline-delimited JSON-RPC 2.0, requests carrying an
`id`, protocol faults in `error`, domain failures in the result.  A refused
tool call is `{"ok": false, ...}` with a `200`-shaped reply, because "you may
not do that" is an answer, not a malformed request.  The four one-line frame
helpers are repeated here instead of imported: the bridge pulls in the agent
loop, and dragging that in to reuse `json.dumps` would be a worse trade.

Unlike the bridge there is no writer thread and no outbound queue, because
nothing is ever pushed to a loopback client — every frame it receives is the
answer to a question it is blocked on.  One reader thread per connection,
handling requests in order, is the whole concurrency model.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Final, Iterator, Mapping

from offset.core import settings
from offset.core.entries import new_id
from offset.providers.base import ToolCall
from offset.tools.runtime import THRESHOLD, Runtime

#: JSON-RPC version string every frame carries.
PROTOCOL: Final = "2.0"

#: Bumped when the wire contract changes in a way an already-written script
#: would misread.  Handed out in the `hello` reply so a client can refuse a
#: version it does not know instead of guessing.
LOOPBACK_VERSION: Final = "1"

#: The address of the published socket: a filesystem path, or `tcp://host:port`
#: on a platform without unix sockets.  Empty means "nothing is listening", and
#: is written deliberately over an inherited value.
SOCKET_ENV: Final = "OFFSET_LOOPBACK_SOCKET"

#: The shared secret.  Environment only, never argv.
TOKEN_ENV: Final = "OFFSET_LOOPBACK_TOKEN"

#: How many loopback hops already separate this process from the human.
DEPTH_ENV: Final = "OFFSET_LOOPBACK_DEPTH"

#: Every variable this module publishes.  Kept as one tuple so masking an
#: inherited environment cannot forget one of them.
ENV_NAMES: Final = (SOCKET_ENV, TOKEN_ENV, DEPTH_ENV)

SOCKET_NAME: Final = "loopback.sock"

#: Module name the client helper is materialised under.  Prefixed because it
#: lands on the child's `sys.path`, where a name like `client` would shadow
#: somebody else's module.
CLIENT_NAME: Final = "offset_loopback.py"

#: Loopback hops allowed.  One is the point of the feature — a script calling a
#: tool.  Two covers the honest nested case, a script whose tool call runs
#: another script.  Beyond that every observed case has been a loop, and each
#: level costs a socket, a thread and possibly an approval prompt.
MAX_DEPTH: Final = 2

#: Concurrent connections.  A script with more open at once is leaking them;
#: the ceiling turns a leak into an error message instead of thread exhaustion.
MAX_CLIENTS: Final = 8

#: Tool calls one published loopback may run.  A script looping over a
#: directory legitimately makes hundreds; one making thousands has a bug, and
#: without a ceiling it can spend the whole turn's budget before anyone looks.
MAX_CALLS: Final = 512

#: `log` lines retained for the UI.  The rest are counted and dropped: a script
#: logging in a tight loop must not be able to grow the parent's heap.
MAX_LOGS: Final = 200

#: Longest single frame accepted.  Without a ceiling a client that never sends
#: a newline would grow the read buffer until the process died.
MAX_FRAME: Final = 4 * 1024 * 1024

#: Seconds a connection may stay silent before it must have authenticated.
AUTH_GRACE: Final = 10.0

#: Poll slice for the accept and read loops: the granularity at which
#: `shutdown()` is noticed.
_TICK: Final = 0.05

#: How long `shutdown()` waits for its own threads before giving up on them.
#: Bounded because tearing the loopback down must not be able to hang the turn
#: that is tearing it down.
_JOIN: Final = 2.0

#: Methods a client may call once authenticated.  `hello` is answered before
#: authentication and is therefore not in the dispatch table.
METHODS: Final = ("tools", "call", "log", "ask")

PARSE_ERROR: Final = -32700
INVALID_REQUEST: Final = -32600
METHOD_NOT_FOUND: Final = -32601
INVALID_PARAMS: Final = -32602
INTERNAL_ERROR: Final = -32603
#: Outside the JSON-RPC reserved range, as the spec requires for application
#: errors.  A client keys its "not authorised" state on exactly this.
UNAUTHENTICATED: Final = -32001


class _Invalid(Exception):
    """Bad params, raised by a handler and answered as INVALID_PARAMS."""


# -- wire -------------------------------------------------------------------


def _encode(frame: dict[str, Any]) -> bytes:
    """One frame, one line.  `default=str` because a tool result's `data` may
    carry a Path or a datetime, and a serialisation error here would look to
    the script like the agent hanging up."""
    return json.dumps(frame, ensure_ascii=False, default=str).encode("utf-8") + b"\n"


def _reply(ident: Any, result: Any) -> bytes:
    return _encode({"jsonrpc": PROTOCOL, "id": ident, "result": result})


def _failure(ident: Any, code: int, message: str) -> bytes:
    return _encode({"jsonrpc": PROTOCOL, "id": ident, "error": {"code": code, "message": message}})


def _hangup(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        # Already closed, or never connected. Nothing to report: the only goal
        # is that the peer's blocking recv returns.
        pass
    try:
        sock.close()
    except OSError:
        pass


def depth_of(env: Mapping[str, str] | None = None) -> int:
    """How many loopback hops already separate this process from the human.

    Read from the supplied environment first and from `os.environ` second, so
    the count survives both an in-process nested call and a whole second offset
    started as a child.  An unparseable counter answers `MAX_DEPTH + 1`: a
    corrupted guard must fail closed rather than restart the count at zero,
    which is precisely the loop the guard exists to stop.
    """
    for source in ((env or {}), os.environ):
        raw = source.get(DEPTH_ENV) or ""
        if raw:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                return MAX_DEPTH + 1
    return 0


# -- progress -----------------------------------------------------------------


@dataclass(slots=True)
class Record:
    """One `log` frame from a running script, as the UI wants to show it."""

    message: str
    fields: dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    at: float = field(default_factory=time.time)

    def line(self) -> str:
        shown = "  ".join(f"{k}={v}" for k, v in list(self.fields.items())[:6])
        head = "  " * self.depth
        return f"{head}{self.message}" + (f"  ({shown})" if shown else "")


# -- the server ---------------------------------------------------------------


class Loopback:
    """A token-authenticated JSON-RPC server exposing one runtime to one call.

    Constructed, started, used and shut down inside a single tool call.  The
    normal way to hold one is `serve()`, which guarantees the teardown.
    """

    __slots__ = (
        "_accept",
        "_conns",
        "_dir",
        "_lock",
        "_methods",
        "_runtime",
        "_server",
        "_stop",
        "asks",
        "ask_hook",
        "calls",
        "depth",
        "dropped_logs",
        "host",
        "log_hook",
        "max_calls",
        "port",
        "problems",
        "records",
        "refusals",
        "runtime",
        "socket_path",
        "started",
        "token",
        "transport",
    )

    def __init__(
        self,
        runtime: Runtime,
        *,
        depth: int = 0,
        ask: Callable[[str, bool], bool] | None = None,
        log: Callable[[Record], None] | None = None,
        transport: str = "",
        max_calls: int = MAX_CALLS,
    ) -> None:
        self.runtime = runtime
        self.depth = max(0, int(depth))
        #: Asked for `ask`.  `None` means nobody is attached, and every
        #: question is then answered "no" — a question the human never saw must
        #: not be able to return "yes".
        self.ask_hook = ask
        self.log_hook = log
        #: Empty means "a unix socket if this platform has one".  `tcp` forces
        #: the fallback, which is bound to loopback only and defended solely by
        #: the token, since a TCP port has no filesystem mode to hide behind.
        self.transport = transport.strip().lower()
        self.max_calls = max(1, int(max_calls))
        self.token = ""
        self.socket_path: Path | None = None
        self.host = ""
        self.port = 0
        self.started = 0.0
        self.calls = 0
        self.asks = 0
        #: Calls the approval policy turned down, counted so the turn can say
        #: "your script was refused eleven times" instead of the user wondering
        #: why the script did nothing.
        self.refusals = 0
        self.records: list[Record] = []
        self.dropped_logs = 0
        self.problems: list[str] = []
        self._dir: Path | None = None
        self._server: socket.socket | None = None
        self._accept: threading.Thread | None = None
        self._conns: dict[str, socket.socket] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._runtime = self._child_runtime()
        self._methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "tools": self._tools,
            "call": self._call,
            "log": self._log,
            "ask": self._ask,
        }

    # -- the runtime a loopback call runs on --------------------------------

    def _child_runtime(self) -> Runtime:
        """The same toolbox, the same approval object, a deeper environment.

        Sharing `Approval` by reference is the point: a remembered "always
        allow" made for the model applies to the script, and a decision made
        for the script is remembered for the model.  Two policies would mean
        the script had found a second, quieter door.

        The environment is where they differ.  The socket and token are blanked
        so a tool run over the loopback cannot hand its own child the address
        of *this* server — that would reset the depth counter to zero and make
        the recursion cap decorative — and the depth is incremented so whoever
        publishes a nested loopback for that tool inherits the right count.
        """
        context = replace(
            self.runtime.context,
            env={
                **self.runtime.context.env,
                SOCKET_ENV: "",
                TOKEN_ENV: "",
                DEPTH_ENV: str(self.depth + 1),
            },
        )
        child = Runtime(
            self.runtime.toolbox,
            context,
            self.runtime.approval,
            before_write=self.runtime.before_write,
        )
        # One abort event for the whole turn: a ctrl-c must reach a tool the
        # script started, not just the ones the model started.
        child.abort = self.runtime.abort
        return child

    # -- lifecycle ----------------------------------------------------------

    @property
    def unix(self) -> bool:
        return self.transport != "tcp" and hasattr(socket, "AF_UNIX")

    @property
    def listening(self) -> bool:
        return self._server is not None and not self._stop.is_set()

    @property
    def address(self) -> str:
        """What goes in `OFFSET_LOOPBACK_SOCKET`, or "" when nothing is up."""
        if not self.listening:
            return ""
        if self.socket_path is not None:
            return str(self.socket_path)
        return f"tcp://{self.host or '127.0.0.1'}:{self.port}"

    def start(self) -> list[str]:
        """Bind and accept in a daemon thread.  Returns the reasons it could
        not, empty on success.

        Never raises.  A loopback is a convenience: a tool call whose callback
        socket could not be published must still run, merely without the
        callback.
        """
        if self._server is not None:
            return []
        self.problems = []
        if self.depth > MAX_DEPTH:
            self.problems = [
                f"loopback nesting stopped at depth {self.depth}: the limit is {MAX_DEPTH}"
            ]
            return list(self.problems)

        server, problem = self._bind()
        if server is None:
            self._cleanup_dir()
            self.problems = [problem or "the loopback socket could not be bound"]
            return list(self.problems)

        self.token = secrets.token_urlsafe(32)
        self._server = server
        self._stop.clear()
        self.started = time.time()
        self._accept = threading.Thread(
            target=self._accept_loop, name=f"offset-loopback-{self.depth}", daemon=True
        )
        self._accept.start()
        return []

    def _bind(self) -> tuple[socket.socket | None, str | None]:
        if not self.unix:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # No SO_REUSEADDR: this port is ephemeral and private, and
                # reusing an address is how a second server silently inherits
                # the first one's clients.
                server.bind(("127.0.0.1", 0))
                server.listen(MAX_CLIENTS)
            except OSError as exc:
                server.close()
                return None, f"127.0.0.1: {type(exc).__name__}: {exc}"
            self.host, self.port = "127.0.0.1", int(server.getsockname()[1])
            server.settimeout(_TICK)
            return server, None

        try:
            # Its own directory, `0o700`, so the socket is unreachable by other
            # accounts even where the socket mode itself is ignored, and so two
            # concurrent tool calls cannot land on the same path.
            self._dir = Path(tempfile.mkdtemp(prefix="offset-loopback-"))
        except OSError as exc:
            return None, f"no directory for the loopback socket: {type(exc).__name__}: {exc}"
        path = self._dir / SOCKET_NAME
        # 108 on Linux, 104 on BSD; both are the size of sockaddr_un.sun_path.
        if len(str(path).encode("utf-8")) >= 104:
            return None, f"{path} is too long for a unix socket path"

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Created under a restrictive umask rather than chmod'ed afterwards:
        # between bind and chmod the socket is connectable by anyone.
        previous = os.umask(0o177)
        try:
            server.bind(str(path))
            server.listen(MAX_CLIENTS)
        except OSError as exc:
            server.close()
            return None, f"{path}: {type(exc).__name__}: {exc}"
        finally:
            os.umask(previous)
        self.socket_path = path
        server.settimeout(_TICK)
        return server, None

    def env(self) -> dict[str, str]:
        """The variables to merge into a child process's environment.

        Every name is always present, and blank when nothing is listening.
        Omitting them would leave a grandparent's socket and token visible
        through plain inheritance, which is the reconnect that defeats the
        depth cap; an empty value masks it.
        """
        if not self.listening:
            return {SOCKET_ENV: "", TOKEN_ENV: "", DEPTH_ENV: str(self.depth)}
        return {SOCKET_ENV: self.address, TOKEN_ENV: self.token, DEPTH_ENV: str(self.depth + 1)}

    def shutdown(self) -> None:
        """Stop accepting, hang up on everyone, remove the socket.  Idempotent.

        Runs in the `finally` of `serve()`, so it must survive being called
        after a failed `start()` and being called twice.
        """
        self._stop.set()
        server, self._server = self._server, None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        with self._lock:
            conns, self._conns = list(self._conns.values()), {}
        for conn in conns:
            _hangup(conn)
        accept, self._accept = self._accept, None
        if accept is not None and accept is not threading.current_thread():
            accept.join(timeout=_JOIN)
        self.token = ""
        self.socket_path = None
        self._cleanup_dir()

    def _cleanup_dir(self) -> None:
        target, self._dir = self._dir, None
        if target is None:
            return
        # `ignore_errors` because a socket file that has already gone is the
        # success case, and a temp directory that cannot be removed must not
        # turn into an exception raised from a `finally` block.
        shutil.rmtree(target, ignore_errors=True)

    def report(self) -> list[str]:
        """Human-facing state, for a `/loopback` command or a debug notice."""
        if self.problems:
            return ["loopback: not published"] + [f"  {p}" for p in self.problems]
        if not self.listening:
            return ["loopback: not started"]
        lines = [
            f"loopback: listening on {self.address}",
            f"  depth {self.depth}/{MAX_DEPTH}, {self.calls} calls, {self.refusals} refused, {self.asks} asked",
        ]
        for record in self.records[-5:]:
            lines.append(f"  {record.line()}")
        if self.dropped_logs:
            lines.append(f"  ({self.dropped_logs} earlier log lines dropped)")
        return lines

    # -- accepting ----------------------------------------------------------

    def _accept_loop(self) -> None:
        server = self._server
        while not self._stop.is_set() and server is not None:
            try:
                conn, _addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(_TICK)
                continue
            self._adopt(conn)

    def _adopt(self, conn: socket.socket) -> None:
        ident = new_id()
        conn.settimeout(_TICK)
        with self._lock:
            over = len(self._conns) >= MAX_CLIENTS
            if not over:
                self._conns[ident] = conn
        if over:
            # Answered rather than dropped silently, so a script leaking
            # connections reads the reason instead of seeing a bare EOF.
            try:
                conn.sendall(_failure(None, INTERNAL_ERROR, f"the loopback already has {MAX_CLIENTS} clients"))
            except OSError:
                pass
            _hangup(conn)
            return
        threading.Thread(
            target=self._read_loop, args=(ident, conn), name=f"loopback-rx-{ident[-6:]}", daemon=True
        ).start()

    def _forget(self, ident: str) -> None:
        with self._lock:
            conn = self._conns.pop(ident, None)
        if conn is not None:
            _hangup(conn)

    # -- reading ------------------------------------------------------------

    def _read_loop(self, ident: str, conn: socket.socket) -> None:
        buf = bytearray()
        authenticated = False
        deadline = time.monotonic() + AUTH_GRACE
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(65536)
                except TimeoutError:
                    if not authenticated and time.monotonic() > deadline:
                        self._reject(conn, None, "no hello frame arrived; the loopback requires a token")
                        return
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                if len(buf) > MAX_FRAME:
                    self._reject(conn, None, f"a single frame exceeded {MAX_FRAME} bytes")
                    return
                while (nl := buf.find(b"\n")) >= 0:
                    line = bytes(buf[:nl])
                    del buf[: nl + 1]
                    if not line.strip():
                        continue
                    alive, authenticated = self._handle(conn, line, authenticated)
                    if not alive:
                        return
        finally:
            self._forget(ident)

    def _handle(self, conn: socket.socket, line: bytes, authenticated: bool) -> tuple[bool, bool]:
        """One frame.  Returns `(keep the connection, authenticated)`."""
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # A bad frame is the script's bug, not grounds for a hangup: the
            # next frame may well be fine, and a closed socket mid-script is
            # far harder to diagnose than one error reply.
            return self._send(conn, _failure(None, PARSE_ERROR, f"frame is not valid JSON: {exc}")), authenticated
        if not isinstance(message, dict):
            return self._send(conn, _failure(None, INVALID_REQUEST, "every frame must be a JSON object")), authenticated

        ident = message.get("id")
        method = message.get("method")
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(method, str) or not method:
            return self._send(conn, _failure(ident, INVALID_REQUEST, "every request needs a string 'method'")), authenticated
        if not isinstance(params, dict):
            return self._send(conn, _failure(ident, INVALID_PARAMS, f"params for {method} must be a JSON object")), authenticated

        if not authenticated:
            if method != "hello":
                self._reject(conn, ident, f"{method} was sent before hello; authenticate first")
                return False, False
            offered = params.get("token")
            # `compare_digest` against a fresh random string when no token
            # exists, so "not started yet" takes the same time as "wrong
            # token" and answers the same way.
            if not isinstance(offered, str) or not secrets.compare_digest(offered, self.token or new_id()):
                self._reject(conn, ident, f"the token in {TOKEN_ENV} does not match this loopback")
                return False, False
            return self._send(conn, _reply(ident, self.greeting())), True

        if method == "hello":
            return self._send(conn, _reply(ident, self.greeting())), True

        handler = self._methods.get(method)
        if handler is None:
            return self._send(conn, _failure(
                ident, METHOD_NOT_FOUND,
                f"no method named {method!r}. available: " + ", ".join(sorted(self._methods)),
            )), True
        try:
            result = handler(params)
        except _Invalid as exc:
            return self._send(conn, _failure(ident, INVALID_PARAMS, str(exc))), True
        except Exception as exc:  # a broken handler must not kill the turn
            return self._send(conn, _failure(ident, INTERNAL_ERROR, f"{method} failed: {type(exc).__name__}: {exc}")), True
        if ident is None:
            return True, True
        return self._send(conn, _reply(ident, result)), True

    def _send(self, conn: socket.socket, frame: bytes) -> bool:
        try:
            conn.sendall(frame)
        except OSError:
            # The script exited without reading its answer. Common and
            # harmless: the work is already done, and there is nobody to tell.
            return False
        return True

    def _reject(self, conn: socket.socket, ident: Any, why: str) -> None:
        """One error frame, then the connection goes.  A socket that can edit
        files does not get to retry its way in."""
        self._send(conn, _failure(ident, UNAUTHENTICATED, why))
        _hangup(conn)

    def greeting(self) -> dict[str, Any]:
        return {
            "ok": True,
            "version": LOOPBACK_VERSION,
            "protocol": PROTOCOL,
            "pid": os.getpid(),
            "depth": self.depth,
            "max_depth": MAX_DEPTH,
            "cwd": str(self.runtime.context.cwd),
            "root": None if self.runtime.context.root is None else str(self.runtime.context.root),
            "methods": list(METHODS),
        }

    # -- methods ------------------------------------------------------------

    def _tools(self, _params: dict[str, Any]) -> dict[str, Any]:
        """What is callable, and which of it will stop to ask a human.

        `needs_approval` is advertised so a script can order its work: doing
        the free reads first and the prompting writes last is the difference
        between one interruption and twenty.
        """
        approval = self.runtime.approval
        threshold = THRESHOLD[approval.mode]
        listed = []
        for tool in self.runtime.toolbox:
            asks = tool.danger > threshold and tool.name not in approval.remembered
            listed.append({
                "name": tool.name,
                "description": tool.description,
                "danger": tool.danger.name.lower(),
                "parallel_safe": tool.parallel_safe,
                "needs_approval": asks,
                "refusable": asks and approval.ask is None,
                "schema": tool.schema,
            })
        return {
            "tools": listed,
            "mode": approval.mode,
            "depth": self.depth,
            "calls_left": max(0, self.max_calls - self.calls),
        }

    def _call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise _Invalid("call needs a string 'name'")
        args = params.get("args")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise _Invalid("call needs 'args' to be a JSON object")

        with self._lock:
            self.calls += 1
            spent = self.calls
        if spent > self.max_calls:
            # A budget failure is a result, not a protocol error: the script
            # asked correctly and is being told it has had enough.
            return {
                "ok": False,
                "approved": False,
                "content": "",
                "display": "",
                "error": f"this loopback has already run {self.max_calls} tool calls",
                "duration": 0.0,
                "data": {},
            }

        invocation = self._runtime.execute(ToolCall(id=new_id(), name=name, args=args))
        result = invocation.result
        if not invocation.approved:
            with self._lock:
                self.refusals += 1
        return {
            "ok": result.ok,
            "approved": invocation.approved,
            "content": result.content,
            "display": result.display,
            "error": result.error,
            "duration": round(result.duration, 4),
            "data": result.data,
        }

    def _log(self, params: dict[str, Any]) -> dict[str, Any]:
        message = params.get("message")
        if not isinstance(message, str) or not message.strip():
            raise _Invalid("log needs a non-empty string 'message'")
        fields = params.get("fields")
        if fields is not None and not isinstance(fields, dict):
            raise _Invalid("log needs 'fields' to be a JSON object")
        record = Record(message=message.strip()[:2000], fields=dict(fields or {}), depth=self.depth)
        with self._lock:
            self.records.append(record)
            if len(self.records) > MAX_LOGS:
                del self.records[0]
                self.dropped_logs += 1
        if self.log_hook is not None:
            try:
                self.log_hook(record)
            except Exception:
                # A UI that cannot paint a progress line must not fail the
                # script's call; the record is kept either way.
                pass
        return {"ok": True, "kept": len(self.records), "dropped": self.dropped_logs}

    def _ask(self, params: dict[str, Any]) -> dict[str, Any]:
        question = params.get("question")
        if not isinstance(question, str) or not question.strip():
            raise _Invalid("ask needs a non-empty string 'question'")
        # `default` is a hint for how to present the question, never an answer.
        # Honouring it with nobody attached would let a script write
        # `default=True` and grant itself a yes.
        hint = bool(params.get("default"))
        with self._lock:
            self.asks += 1
        if self.ask_hook is None:
            return {
                "answer": False,
                "asked": False,
                "reason": "no human is attached to this loopback, so the answer is no",
            }
        try:
            answer = bool(self.ask_hook(question.strip()[:2000], hint))
        except Exception as exc:
            return {"answer": False, "asked": False, "reason": f"asking failed: {type(exc).__name__}: {exc}"}
        return {"answer": answer, "asked": True, "reason": ""}


# -- publishing one for the duration of a call --------------------------------


@contextmanager
def serve(
    runtime: Runtime,
    *,
    depth: int | None = None,
    env: Mapping[str, str] | None = None,
    ask: Callable[[str, bool], bool] | None = None,
    log: Callable[[Record], None] | None = None,
    transport: str = "",
    max_calls: int = MAX_CALLS,
    publish_to: dict[str, str] | None = None,
) -> Iterator[Loopback]:
    """Publish a loopback for the body, and tear it down whatever happens.

    `env` is the environment the caller was handed — the depth is inherited
    from it, which is what makes nesting counted rather than hoped for.
    `publish_to` is the mapping a child process will be spawned with (usually
    `runtime.context.env`); its previous values are restored on the way out, so
    a tool call that finishes cannot leave a live-looking address behind for
    the next one.

    The loopback may fail to publish — nesting too deep, no writable temp
    directory — and the body still runs.  Check `lb.listening`.
    """
    loopback = Loopback(
        runtime,
        depth=depth_of(env) if depth is None else depth,
        ask=ask,
        log=log,
        transport=transport,
        max_calls=max_calls,
    )
    loopback.start()
    previous: dict[str, str | None] = {}
    if publish_to is not None:
        previous = {name: publish_to.get(name) for name in ENV_NAMES}
        publish_to.update(loopback.env())
    try:
        yield loopback
    finally:
        for name, value in previous.items():
            if value is None:
                publish_to.pop(name, None)  # type: ignore[union-attr]
            else:
                publish_to[name] = value  # type: ignore[index]
        loopback.shutdown()


# -- the client the executing script imports ----------------------------------
#
# One definition, kept as a string, because the client has to be importable by
# a process that knows nothing about offset: a script written to /tmp cannot
# `import offset.core.loopback`, and requiring it to could would mean requiring
# the child to share the parent's interpreter and its installed packages. The
# env variable names appear here as literals for the same reason; the tests
# assert they still match the constants above, so a rename cannot quietly
# break every script.

CLIENT_SNIPPET: Final = '''"""Call the offset agent that started this process.

Written by `offset.core.loopback`.  Pure stdlib, imports nothing from offset,
so it can be dropped next to a generated script and imported by it.

    import offset_loopback as agent

    if agent.available():
        agent.log("scanning", files=120)
        reply = agent.call("read", {"path": "pyproject.toml"})
        if reply["ok"]:
            print(reply["content"])
        elif agent.ask("may I overwrite the lockfile?"):
            ...

Every call returns the agent's own answer.  `ok` false means the agent refused
or the tool failed - the approval policy still applies, and it may say no.
Only a protocol fault raises `LoopbackError`.
"""

from __future__ import annotations

import json
import os
import socket
import threading

SOCKET_ENV = "OFFSET_LOOPBACK_SOCKET"
TOKEN_ENV = "OFFSET_LOOPBACK_TOKEN"
DEPTH_ENV = "OFFSET_LOOPBACK_DEPTH"

#: Seconds to wait for one answer.  A `call` can be slow: it may be sitting in
#: front of a human deciding whether to allow it.
TIMEOUT = 600.0


class LoopbackError(RuntimeError):
    """No agent is listening, or it rejected the frame outright."""


def available():
    """Whether this process was given a loopback at all."""
    return bool(os.environ.get(SOCKET_ENV) and os.environ.get(TOKEN_ENV))


def depth():
    """How many loopback hops separate this process from the human."""
    try:
        return max(0, int(os.environ.get(DEPTH_ENV) or "0"))
    except ValueError:
        return 0


class Client:
    """One connection.  Requests are serialised: replies come back in order,
    so a lock is all the framing this needs."""

    def __init__(self, address="", token="", timeout=TIMEOUT):
        address = address or os.environ.get(SOCKET_ENV) or ""
        token = token or os.environ.get(TOKEN_ENV) or ""
        if not address or not token:
            raise LoopbackError("no offset loopback is published to this process")
        if address.startswith("tcp://"):
            host, _, port = address[len("tcp://"):].rpartition(":")
            sock = socket.create_connection((host or "127.0.0.1", int(port)), timeout)
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            try:
                sock.connect(address)
            except OSError as exc:
                sock.close()
                raise LoopbackError("cannot reach the offset loopback: %s" % exc) from exc
        self._sock = sock
        self._buf = b""
        self._lock = threading.Lock()
        self._seq = 0
        self.hello = self._rpc("hello", {"token": token})

    # -- wire --------------------------------------------------------------

    def _rpc(self, method, params):
        with self._lock:
            self._seq += 1
            frame = json.dumps({"jsonrpc": "2.0", "id": str(self._seq), "method": method, "params": params})
            try:
                self._sock.sendall(frame.encode("utf-8") + b"\\n")
            except OSError as exc:
                raise LoopbackError("%s: the agent hung up (%s)" % (method, exc)) from exc
            line = self._readline()
        reply = json.loads(line)
        error = reply.get("error")
        if isinstance(error, dict):
            raise LoopbackError(str(error.get("message") or method))
        return reply.get("result")

    def _readline(self):
        while b"\\n" not in self._buf:
            try:
                chunk = self._sock.recv(65536)
            except OSError as exc:
                raise LoopbackError("the offset loopback went away (%s)" % exc) from exc
            if not chunk:
                raise LoopbackError("the offset loopback closed without answering")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\\n")
        return line.decode("utf-8", "replace")

    # -- methods -----------------------------------------------------------

    def tools(self):
        """Every callable tool, each with `needs_approval`."""
        return self._rpc("tools", {})["tools"]

    def call(self, name, args=None):
        """Run one tool. Returns `{ok, content, error, approved, data, ...}`."""
        return self._rpc("call", {"name": name, "args": dict(args or {})})

    def log(self, message, **fields):
        """Report progress the user can see while this script runs."""
        return self._rpc("log", {"message": message, "fields": fields})

    def ask(self, question, default=False):
        """Ask the human a yes/no question. False when nobody is attached."""
        return bool(self._rpc("ask", {"question": question, "default": bool(default)})["answer"])

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass  # closing a socket that already died is not news

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


_session = None
_session_lock = threading.Lock()


def connect(address="", token="", timeout=TIMEOUT):
    """A new connection, addressed explicitly or from the environment."""
    return Client(address, token, timeout)


def session():
    """The shared connection, made on first use.  Most scripts want this."""
    global _session
    with _session_lock:
        if _session is None:
            _session = Client()
        return _session


def tools():
    return session().tools()


def call(name, args=None):
    return session().call(name, args)


def log(message, **fields):
    return session().log(message, **fields)


def ask(question, default=False):
    return session().ask(question, default)
'''


def client_module_path(directory: str | os.PathLike[str] | None = None) -> Path:
    """Materialise `CLIENT_SNIPPET` and return the file to import.

    Defaults to `OFFSET_HOME`, so the module survives between calls and a
    script can put its parent on `sys.path` instead of carrying a copy.
    Rewritten only when the contents differ, so an upgrade takes effect and an
    unchanged file keeps its mtime.
    """
    target = Path(directory) if directory is not None else settings.home()
    target.mkdir(parents=True, exist_ok=True)
    path = target / CLIENT_NAME
    try:
        if path.read_text(encoding="utf-8") == CLIENT_SNIPPET:
            return path
    except (OSError, UnicodeDecodeError):
        pass  # missing, or unreadable rubbish: either way it gets rewritten
    # Written to a sibling and renamed: a half-written module on somebody's
    # `sys.path` is an ImportError in a subprocess nobody is watching.
    tmp = path.with_name(f".{CLIENT_NAME}.{os.getpid()}")
    tmp.write_text(CLIENT_SNIPPET, encoding="utf-8")
    os.replace(tmp, path)
    return path


_client_cache: dict[str, Any] = {}


def client(directory: str | os.PathLike[str] | None = None) -> Any:
    """`CLIENT_SNIPPET` as an imported module, for calling a loopback from
    inside this process.

    The snippet is the only client that exists, so the in-process caller uses
    it too: a second implementation for the parent's own use would be the one
    that stayed working while the one scripts actually import rotted.
    """
    import importlib.util

    path = client_module_path(directory)
    cached = _client_cache.get(str(path))
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(f"offset_loopback_{abs(hash(str(path)))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the loopback client from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _client_cache[str(path)] = module
    return module
