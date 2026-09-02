"""Driving a turn on somebody else's machine.

`core/daemon` already puts a full offset behind a socket, and
`clients/typescript` already speaks to it - but only an editor extension could.
The machine that has the API keys, the sixty-four gigabytes and the fast disk is
frequently not the machine somebody is typing on, and copying a checkout across
to it is not a workflow.  So this is the Python half of the same protocol: a
client, a small registry of named remotes, and one command that runs a prompt
over there while the events appear over here.

There is no second protocol.  Every frame on the wire is exactly what
`offset/core/bridge.py` serves and what `clients/typescript/src/client.ts`
consumes, which is why the two clients are kept deliberately parallel - the
frame ordering, the `hello`-before-anything rule and the buffer ceiling are the
same in both files.  When the bridge grows a method, `request` reaches it the
same day.

Four decisions carry the design.

**`hello` gates everything, locally too.** The bridge refuses any method sent
before `hello` and hangs up on the connection that tried.  A client that let a
caller send `status` first would therefore not fail that one call, it would lose
the socket - so `request` refuses the call itself rather than discovering the
policy from a dead connection.

**A version mismatch is named, never negotiated.** Bridge frames are versioned
because their shape changed once; a client that shrugged and carried on would
mis-read a payload and report the wrong thing about a file. If the remote speaks
a version this build does not, the message says both numbers.

**An unreachable remote says where it looked.** The whole point of a remote is
that it is somewhere else, so "connection refused" without an address is
useless: every failure here carries the address it tried, which is why
`RemoteError` has a field for it rather than relying on the message text.

**A tunnel never outlives the call that opened it.** `ssh -N -L` to reach a
loopback-bound daemon is the safe way to expose one, but an `ssh` left running
is both a hole and a file descriptor nobody will ever close.  The teardown is in
a `finally`, it signals the process *group*, and the child asks the kernel to
kill it if this process dies without running that `finally` at all - the same
discipline, and for the same measured reason, as `offset/tools/web/cdp.py`.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterator

from offset.core import settings
from offset.core.bridge import BRIDGE_VERSION, PROTOCOL, TOKEN_NAME

#: Where named remotes live.  Beside the bridge token and the session store, so
#: one `OFFSET_HOME` moves the whole lot - which is what makes the tests, and a
#: second checkout, able to have their own remotes.
REMOTES_NAME: Final = "remotes.json"

#: Bumped if the on-disk shape of `remotes.json` changes incompatibly.
CONFIG_VERSION: Final = 1

#: Seconds to wait for a TCP or unix connect.  Short: a remote that has not
#: answered a connect in this long is not merely busy, it is not there, and the
#: user is sitting in front of a prompt waiting to be told so.
CONNECT_TIMEOUT: Final = 10.0

#: Seconds to wait for an ordinary request's reply.  `status`, `diff` and
#: `sessions` are all local work on the far side; a minute would only hide a
#: wedged daemon.
DEFAULT_TIMEOUT: Final = 30.0

#: Seconds to wait for a `prompt` reply.  A turn is a model plus tools, so this
#: is a different order of magnitude from every other method and must not share
#: their ceiling: a fifteen-minute refactor answering at twelve minutes is a
#: success, and a thirty-second deadline would have thrown it away.
RUN_TIMEOUT: Final = 1800.0

#: Reader poll slice.  The granularity at which `dispose()` is noticed by the
#: reader thread, and small enough that a disposed client does not linger.
_TICK: Final = 0.1

#: Bytes read per `recv`.  One page-ish; frames are usually far smaller and a
#: tool result can be large.
_CHUNK: Final = 65536

#: Longest single frame accepted before the connection is abandoned.  The same
#: ceiling the TypeScript client uses, and for the same reason: the socket is
#: authenticated, but a peer that never sends a newline would otherwise grow
#: this buffer until the kernel killed offset.
MAX_FRAME: Final = 32 * 1024 * 1024

#: Seconds to wait for a freshly spawned `ssh -N -L` to start accepting on the
#: local end.  Long enough for a password-less handshake over a slow link,
#: short enough that a host which will never answer is reported rather than
#: hung on.
TUNNEL_TIMEOUT: Final = 20.0

#: Grace given to `ssh` between SIGTERM and SIGKILL during teardown.
TUNNEL_GRACE: Final = 3.0

#: `ssh` options for a tunnel offset opens itself.  `BatchMode` because there
#: is no terminal here to type a password into - without it a host wanting one
#: hangs until the readiness deadline and reports the wrong cause.  The keep
#: alives are what stop a silently dead link looking like a working tunnel.
SSH_OPTIONS: Final = (
    "-o", "BatchMode=yes",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
)

#: A remote's name is used as a command argument and shown in a table, so it is
#: restricted to what can be typed unquoted and cannot be confused for a path.
_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class RemoteError(Exception):
    """Anything that went wrong talking to a remote, with where it happened.

    `address` is separate from the message because callers format their own
    failures - the shell prints one line, a tool returns a `ToolResult` - and
    every one of them wants to name the address.  Recovering it by parsing the
    message text would be the sort of thing that quietly stops working.
    """

    __slots__ = ("address", "code")

    def __init__(self, message: str, address: str = "", code: int = 0) -> None:
        super().__init__(message)
        self.address = address
        self.code = code


# -- addresses --------------------------------------------------------------


def parse_address(address: str) -> tuple[str, str, int]:
    """`(kind, host_or_path, port)` for a remote address.

    Accepts what the daemon can bind: a unix socket path, `unix:<path>`,
    `host:port`, `tcp:host:port` and a bracketed IPv6 `[::1]:7799`.  A bare
    host with no port is refused rather than defaulted, because the daemon's
    TCP port is ephemeral unless it was pinned - guessing one would produce a
    connection refused against a port the user never mentioned.
    """
    spec = address.strip()
    if not spec:
        raise RemoteError("a remote needs an address, such as 127.0.0.1:7799", address)
    if spec.startswith("unix:"):
        path = spec[len("unix:"):].strip()
        if not path:
            raise RemoteError(f"{address!r} names no socket path", address)
        return "unix", os.path.expanduser(path), 0
    if spec.startswith("tcp:"):
        spec = spec[len("tcp:"):].strip()
    if spec.startswith(("/", "./", "../", "~")):
        return "unix", os.path.expanduser(spec), 0

    host, sep, port = spec.rpartition(":")
    if not sep:
        raise RemoteError(
            f"{address!r} is neither host:port nor a unix socket path", address
        )
    try:
        number = int(port)
    except ValueError:
        raise RemoteError(f"{port!r} in {address!r} is not a port number", address) from None
    if not 0 < number < 65536:
        raise RemoteError(f"port {number} in {address!r} is out of range", address)
    # Brackets are how an IPv6 literal is written with a port; the socket layer
    # wants the address without them.
    return "tcp", (host.strip("[]") or "127.0.0.1"), number


# -- named remotes ----------------------------------------------------------


@dataclass(slots=True)
class Remote:
    """One machine offset can be asked to run a turn on."""

    name: str
    address: str
    #: Where *this* machine keeps the remote's bridge token.  A path rather
    #: than the secret itself: `remotes.json` is a config file people put in
    #: dotfile repositories, and a token in it would be committed.
    token_path: str = ""
    #: The workspace on the far side.  Advisory - the daemon decides its own
    #: root - but it is what makes a listing of remotes readable.
    workspace: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "address": self.address,
            "token_path": self.token_path,
            "workspace": self.workspace,
        }

    @classmethod
    def from_json(cls, raw: Any) -> "Remote | None":
        """One entry, or None when the file has been hand-edited into rubbish.

        A single bad entry must not cost the user the rest of their remotes,
        so this returns None and the caller skips it.
        """
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name") or "").strip()
        address = str(raw.get("address") or "").strip()
        if not name or not address:
            return None
        return cls(
            name=name,
            address=address,
            token_path=str(raw.get("token_path") or ""),
            workspace=str(raw.get("workspace") or ""),
        )

    def token_file(self, home: Path | None = None) -> Path:
        """The token path, defaulting to this machine's own bridge token.

        The default is genuinely useful: an SSH tunnel to your own build box
        usually means the token was copied to `~/.offset/bridge.token`, and
        making everybody spell that out would be noise.
        """
        if self.token_path:
            return Path(os.path.expanduser(self.token_path))
        return (home if home is not None else settings.home()) / TOKEN_NAME

    def read_token(self, home: Path | None = None) -> str:
        """The secret, or a failure naming the file that was missing.

        A remote with no readable token cannot connect at all, and the
        overwhelmingly common cause is that the file was never copied across -
        so the path is in the message.
        """
        path = self.token_file(home)
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RemoteError(
                f"cannot read the token for {self.name} at {path}: {type(exc).__name__}: {exc}",
                self.address,
            ) from exc
        if not token:
            raise RemoteError(f"the token file {path} is empty", self.address)
        return token

    def summary(self) -> str:
        where = f"  {self.workspace}" if self.workspace else ""
        return f"{self.name:<16} {self.address}{where}"


def remotes_file(home: Path | None = None) -> Path:
    """Where the registry lives.  `home` is explicit for the same reason it is
    in `providers/catalogue`: a caller on a background thread must pass the
    directory it resolved, not re-resolve one later."""
    return (home if home is not None else settings.home()) / REMOTES_NAME


def remotes(home: Path | None = None) -> list[Remote]:
    """Every named remote, by name.  Never raises: an unreadable or
    hand-mangled config means "no remotes", not a traceback at startup."""
    try:
        raw = json.loads(remotes_file(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = raw.get("remotes") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []
    found: dict[str, Remote] = {}
    for item in entries:
        parsed = Remote.from_json(item)
        if parsed is not None:
            found[parsed.name] = parsed
    return [found[key] for key in sorted(found)]


def _write_remotes(entries: list[Remote], home: Path | None = None) -> Path:
    """Replace the registry atomically.

    Temp file in the same directory then `os.replace`, so an interrupted write
    leaves the previous list readable rather than a truncated file that would
    lose every remote the user had configured.
    """
    target = remotes_file(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".remotes.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(
                {"version": CONFIG_VERSION, "remotes": [r.to_json() for r in entries]},
                fh,
                indent=1,
            )
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def add_remote(
    name: str,
    address: str,
    *,
    token_path: str = "",
    workspace: str = "",
    home: Path | None = None,
) -> Remote:
    """Record a remote, replacing any entry of the same name.

    The address is parsed here rather than at connect time: a typo is worth
    reporting while the user is still looking at the command they typed, not
    twenty minutes later when they try to use it.
    """
    clean = name.strip()
    if not _NAME_RE.match(clean):
        raise RemoteError(
            f"{name!r} is not a usable remote name; "
            "use letters, digits, dot, dash or underscore",
            address,
        )
    parse_address(address)
    entry = Remote(
        name=clean,
        address=address.strip(),
        token_path=token_path.strip(),
        workspace=workspace.strip(),
    )
    kept = [r for r in remotes(home) if r.name != clean]
    kept.append(entry)
    _write_remotes(sorted(kept, key=lambda r: r.name), home)
    return entry


def remove_remote(name: str, *, home: Path | None = None) -> bool:
    """Forget a remote.  False when there was nothing by that name."""
    clean = name.strip()
    # Read once: two calls to `remotes()` would compare the file against
    # itself across a window in which another offset could have rewritten it.
    known = remotes(home)
    kept = [r for r in known if r.name != clean]
    if len(kept) == len(known):
        return False
    _write_remotes(kept, home)
    return True


def find_remote(name: str, *, home: Path | None = None) -> Remote:
    """One remote by name, or a failure listing the ones that do exist."""
    known = remotes(home)
    for entry in known:
        if entry.name == name.strip():
            return entry
    available = ", ".join(r.name for r in known) or "none are configured"
    raise RemoteError(f"no remote named {name!r}. known: {available}")


# -- the client -------------------------------------------------------------


@dataclass(slots=True)
class _Pending:
    """One request waiting for its reply.

    An `Event` rather than a condition variable per call: the reader thread
    fills the slot and sets it, the caller waits with a deadline, and neither
    needs to hold a lock while the other works.
    """

    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: str = ""
    code: int = 0


#: What a subscriber is handed: the notification name and its params.
EventHandler = Callable[[str, dict[str, Any]], None]


class RemoteAgent:
    """A connected offset daemon, over a unix socket or TCP.

    Deliberately the same shape as `clients/typescript/src/client.ts`: one
    socket, one reader, `request` public so a bridge method added tomorrow is
    reachable today, and notifications delivered to subscribers in wire order.

    Reconnection is *not* here, unlike the editor client.  An editor lives
    beside a program the user starts and stops and wants to re-attach; a remote
    turn is a single operation with a caller waiting on it, and silently
    reconnecting mid-turn would resume against a daemon that has forgotten the
    turn and hand back a reply belonging to nothing.
    """

    __slots__ = (
        "_buffered",
        "_closing",
        "_handlers",
        "_lock",
        "_next_id",
        "_pending",
        "_reader",
        "_sock",
        "address",
        "client_name",
        "connect_timeout",
        "greeting",
        "problem",
        "ready",
        "timeout",
        "token",
    )

    def __init__(
        self,
        address: str,
        *,
        token: str = "",
        name: str = "offset-remote",
        timeout: float = DEFAULT_TIMEOUT,
        connect_timeout: float = CONNECT_TIMEOUT,
        on_event: EventHandler | None = None,
    ) -> None:
        self.address = address
        self.token = token
        self.client_name = name
        self.timeout = max(0.5, timeout)
        self.connect_timeout = max(0.1, connect_timeout)
        #: The `hello` reply, once accepted.  Callers use it to learn the far
        #: side's workspace and method list rather than assuming this build's.
        self.greeting: dict[str, Any] = {}
        self.ready = False
        #: Why the connection ended, in words worth showing a user.
        self.problem = ""
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._pending: dict[int, _Pending] = {}
        self._handlers: list[EventHandler] = [on_event] if on_event is not None else []
        self._lock = threading.Lock()
        self._next_id = 1
        self._closing = False
        #: Bytes of a half-arrived frame.  Exposed through `buffered` because
        #: "the remote is mid-frame" and "the remote has said nothing" look
        #: identical from the outside and are diagnosed differently.
        self._buffered = 0

    # -- lifecycle ----------------------------------------------------------

    @property
    def buffered(self) -> int:
        """Bytes of an incomplete frame currently held."""
        return self._buffered

    def subscribe(self, handler: EventHandler) -> None:
        """Receive every notification the remote pushes.

        Subscribe *before* `connect`: a turn started immediately after
        connecting can publish `agent.started` before a later subscription
        lands, and the first event is the one that tells a UI to open its
        progress view.
        """
        self._handlers.append(handler)

    def connect(self) -> dict[str, Any]:
        """Open the socket and complete `hello`.  Returns the greeting.

        Returns only once `hello` has been accepted, for the reason the
        TypeScript client documents: until then the bridge dispatches nothing,
        so handing back an "open" client would hand back one whose every call
        fails - and worse, whose first call gets the socket hung up on.
        """
        if self._sock is not None:
            return self.greeting
        self._closing = False
        sock = self._open()
        self._sock = sock
        self._reader = threading.Thread(
            target=self._read_loop, args=(sock,), name="offset-remote-reader", daemon=True
        )
        self._reader.start()
        try:
            greeting = self.request("hello", {"token": self.token, "client": self.client_name})
        except RemoteError:
            self.dispose()
            raise
        if not isinstance(greeting, dict):
            self.dispose()
            raise RemoteError(f"{self.address} answered hello with {greeting!r}", self.address)
        version = str(greeting.get("version") or "")
        if version and version != BRIDGE_VERSION:
            self.dispose()
            raise RemoteError(
                f"{self.address} speaks bridge version {version}; this offset speaks "
                f"{BRIDGE_VERSION}. Upgrade whichever end is older",
                self.address,
            )
        self.greeting = greeting
        self.ready = True
        return greeting

    def _open(self) -> socket.socket:
        kind, where, port = parse_address(self.address)
        if kind == "unix":
            if not hasattr(socket, "AF_UNIX"):
                raise RemoteError(
                    f"this platform has no unix sockets, so {self.address} is unreachable; "
                    "use host:port",
                    self.address,
                )
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            try:
                sock.connect(where)
            except OSError as exc:
                sock.close()
                raise RemoteError(
                    f"cannot reach {self.address}: {type(exc).__name__}: {exc}", self.address
                ) from exc
        else:
            try:
                sock = socket.create_connection((where, port), timeout=self.connect_timeout)
            except OSError as exc:
                raise RemoteError(
                    f"cannot reach {self.address}: {type(exc).__name__}: {exc}", self.address
                ) from exc
        # Past the connect, the deadline is per-request rather than per-syscall,
        # so the reader only needs a slice short enough to notice `dispose`.
        sock.settimeout(_TICK)
        return sock

    def dispose(self) -> None:
        """Hang up, and fail every call still waiting.

        A caller blocked in `request` when the client is disposed would
        otherwise wait out its full timeout for a reply that can never arrive -
        which for `prompt` is half an hour of a UI claiming to be busy.
        """
        self._closing = True
        self.ready = False
        self._shutdown("the client was disposed")

    def _shutdown(self, why: str) -> None:
        sock, self._sock = self._sock, None
        self.ready = False
        if not self.problem:
            self.problem = why
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already gone; the close below is what matters
            try:
                sock.close()
            except OSError:
                pass
        with self._lock:
            waiting = list(self._pending.values())
            self._pending.clear()
        for pending in waiting:
            if not pending.done.is_set():
                pending.error = why
                pending.done.set()

    def __enter__(self) -> "RemoteAgent":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.dispose()

    # -- wire ---------------------------------------------------------------

    def request(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float | None = None
    ) -> Any:
        """Send one request and wait for its reply.

        Public, like the TypeScript client's, so a bridge method this build
        does not know about is still reachable.
        """
        sock = self._sock
        if sock is None:
            raise RemoteError(f"not connected to {self.address}", self.address)
        if method != "hello" and not self.ready:
            # Not a courtesy: the bridge answers a pre-`hello` method with one
            # error frame and then drops the connection, so sending this would
            # cost the socket rather than just the call.
            raise RemoteError(
                f"hello has not been accepted by {self.address}; connect first", self.address
            )

        with self._lock:
            ident = self._next_id
            self._next_id += 1
            pending = _Pending()
            self._pending[ident] = pending

        frame = json.dumps(
            {"jsonrpc": PROTOCOL, "id": ident, "method": method, "params": params or {}},
            ensure_ascii=False,
            default=str,
        ).encode("utf-8") + b"\n"
        try:
            sock.sendall(frame)
        except OSError as exc:
            with self._lock:
                self._pending.pop(ident, None)
            self._shutdown(f"{self.address}: {type(exc).__name__}: {exc}")
            raise RemoteError(
                f"cannot send {method} to {self.address}: {type(exc).__name__}: {exc}",
                self.address,
            ) from exc

        limit = self.timeout if timeout is None else max(0.5, timeout)
        if not pending.done.wait(limit):
            with self._lock:
                self._pending.pop(ident, None)
            raise RemoteError(
                f"{method} on {self.address} did not answer within {limit:g}s", self.address
            )
        if pending.error:
            raise RemoteError(pending.error, self.address, pending.code)
        return pending.result

    def _read_loop(self, sock: socket.socket) -> None:
        """The only code that reads the socket, and the only one that resolves
        a pending call or delivers an event.

        Reassembly lives here because the wire is a byte stream: one `recv` can
        return half a frame, two frames, or a frame and a half, and nothing
        about the framing survives assuming otherwise.
        """
        buffer = bytearray()
        while not self._closing:
            try:
                chunk = sock.recv(_CHUNK)
            except TimeoutError:
                continue  # the poll slice expired; check `_closing` and read on
            except OSError as exc:
                if not self._closing:
                    self._shutdown(f"{self.address}: {type(exc).__name__}: {exc}")
                return
            if not chunk:
                if not self._closing:
                    self._shutdown(f"{self.address} closed the connection")
                return
            buffer += chunk
            if len(buffer) > MAX_FRAME:
                self._shutdown(
                    f"{self.address} sent a frame larger than {MAX_FRAME} bytes without a newline"
                )
                return
            while True:
                cut = buffer.find(b"\n")
                if cut < 0:
                    break
                line = bytes(buffer[:cut])
                del buffer[: cut + 1]
                if line.strip():
                    self._dispatch(line)
            self._buffered = len(buffer)
        self._buffered = 0

    def _dispatch(self, line: bytes) -> None:
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return  # a frame we cannot read is not a reason to drop the socket
        if not isinstance(message, dict):
            return

        ident = message.get("id")
        if isinstance(ident, int) and not isinstance(ident, bool):
            with self._lock:
                pending = self._pending.pop(ident, None)
            if pending is None:
                return  # already timed out, or never ours
            failure = message.get("error")
            if isinstance(failure, dict):
                pending.error = str(failure.get("message") or "the remote refused the call")
                raw_code = failure.get("code")
                pending.code = raw_code if isinstance(raw_code, int) else 0
            else:
                pending.result = message.get("result")
            pending.done.set()
            return

        # No id: a notification, which is how the agent's events arrive.
        method = message.get("method")
        if not isinstance(method, str) or not method:
            return
        params = message.get("params")
        payload = params if isinstance(params, dict) else {}
        for handler in list(self._handlers):
            try:
                handler(method, payload)
            except Exception:
                # A subscriber's bug must not cost the connection, and there is
                # nobody on this thread to report it to: the caller is blocked
                # in `request` and would see a lost turn instead of a lost line.
                pass

    # -- the methods, mirroring the TypeScript client -----------------------

    def status(self) -> dict[str, Any]:
        reply = self.request("status")
        return reply if isinstance(reply, dict) else {}

    def sessions(self) -> list[dict[str, Any]]:
        reply = self.request("sessions")
        if isinstance(reply, list):
            return reply
        if isinstance(reply, dict):
            found = reply.get("sessions")
            return found if isinstance(found, list) else []
        return []

    def diff(self) -> list[dict[str, Any]]:
        reply = self.request("diff")
        if isinstance(reply, dict):
            found = reply.get("changes")
            return found if isinstance(found, list) else []
        return []

    def apply_edit(self, target: str, text: str) -> Any:
        return self.request("apply_edit", {"target": target, "text": text})

    def cancel(self) -> Any:
        return self.request("cancel")

    def prompt(self, text: str, *, timeout: float | None = RUN_TIMEOUT) -> dict[str, Any]:
        """Run a real turn.  Answers when the agent has finished, not started."""
        reply = self.request("prompt", {"text": text}, timeout=timeout)
        return reply if isinstance(reply, dict) else {"ok": bool(reply)}


# -- running a turn over there ----------------------------------------------


def run(
    remote: Remote | str,
    prompt: str,
    *,
    on_event: EventHandler | None = None,
    timeout: float = RUN_TIMEOUT,
    home: Path | None = None,
) -> dict[str, Any]:
    """Drive one turn on `remote`, streaming its events to `on_event`.

    Events are delivered on the reader thread, in wire order, *while* the call
    is still outstanding - that is the whole point: a UI that only learned what
    happened when `prompt` returned would show nothing for the length of a
    refactor. A handler must therefore be cheap and thread-safe; posting into a
    queue the UI drains is the intended shape.
    """
    target = remote if isinstance(remote, Remote) else find_remote(remote, home=home)
    client = RemoteAgent(
        target.address, token=target.read_token(home), name="offset-remote", timeout=timeout
    )
    if on_event is not None:
        client.subscribe(on_event)
    with client:
        return client.prompt(prompt, timeout=timeout)


# -- ssh tunnelling ---------------------------------------------------------


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this child when offset goes.

    `start_new_session=True` is what lets teardown signal `ssh`'s whole process
    group without signalling offset too, but detaching from the session also
    means a shell killed outright - rather than exiting through the `finally`
    below - leaves a port forward running that nobody can see and that still
    reaches the remote agent. `PR_SET_PDEATHSIG` closes exactly that gap.

    Linux only, and a best effort: `Tunnel.close()` remains the primary path,
    so a platform without it is no worse off than before.  Copied rather than
    imported from `offset/tools/web/cdp.py` because `core` does not depend on
    `tools`.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except (OSError, AttributeError, ValueError):
        return


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's process group, falling back to the child alone.

    `ssh -N -L` can fork; signalling only the pid we launched has been observed
    to leave the forwarding process behind.
    """
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def free_port() -> int:
    """A local port that is free right now.

    Bind-and-release, so there is a window in which something else could take
    it; `ExitOnForwardFailure=yes` is what turns that race into a reported
    failure instead of a tunnel that looks up but forwards nothing.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
    finally:
        probe.close()


def ssh_command(
    destination: str,
    local_port: int,
    remote_port: int,
    *,
    remote_host: str = "127.0.0.1",
    ssh: str = "ssh",
    options: tuple[str, ...] = SSH_OPTIONS,
) -> list[str]:
    """The argv for a tunnel to a loopback-bound remote daemon.

    Pure, and separate from spawning it, because the argv is the part worth
    asserting on: `-N` (no remote command), the local end pinned to `127.0.0.1`
    so the forward is not itself exposed to the network, and the remote end
    resolved on the far side.
    """
    if not destination.strip():
        raise RemoteError("a tunnel needs a user@host destination")
    forward = f"127.0.0.1:{local_port}:{remote_host}:{remote_port}"
    return [ssh, "-N", *options, "-L", forward, destination.strip()]


def _accepting(port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass(slots=True)
class Tunnel:
    """A running `ssh -N -L`, and the address it made reachable."""

    proc: subprocess.Popen
    local_port: int
    destination: str
    remote_port: int
    grace: float = TUNNEL_GRACE
    _closed: bool = False

    @property
    def address(self) -> str:
        """What to hand `RemoteAgent`."""
        return f"127.0.0.1:{self.local_port}"

    @property
    def alive(self) -> bool:
        return not self._closed and self.proc.poll() is None

    def wait_ready(self, timeout: float = TUNNEL_TIMEOUT) -> bool:
        """Whether the local end started accepting before the deadline.

        Polled rather than assumed: `ssh` returns from `fork` long before the
        forward exists, and a client that connected immediately would report
        "connection refused" against a tunnel that was merely still starting.
        """
        deadline = time.monotonic() + max(0.1, timeout)
        while True:
            if self.proc.poll() is not None:
                return False  # ssh gave up; no amount of waiting helps
            if _accepting(self.local_port):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_TICK)

    def close(self) -> None:
        """Reap `ssh` and its group.  Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        proc = self.proc
        if proc.poll() is None:
            _signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=self.grace)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
                try:
                    proc.wait(timeout=self.grace)
                except subprocess.TimeoutExpired:
                    pass
        else:
            proc.wait()  # reap, or the pid keeps answering kill(pid, 0)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def _spawn(argv: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # so close() can reap the whole tree
        preexec_fn=_die_with_parent,  # ...and so a killed shell reaps it too
    )


@contextmanager
def tunnel(
    destination: str,
    remote_port: int,
    *,
    local_port: int = 0,
    remote_host: str = "127.0.0.1",
    ssh: str = "ssh",
    timeout: float = TUNNEL_TIMEOUT,
    launch: Callable[[list[str]], subprocess.Popen] = _spawn,
) -> Iterator[Tunnel]:
    """Reach a loopback-bound remote daemon for the length of the block.

    The teardown is in `finally` and runs on every path - readiness failure,
    the caller raising, the caller returning early - because an `ssh` forward
    left behind is a route into the remote agent that nothing on this machine
    is watching, as well as a descriptor and a process that leak one per call.

    `launch` is injected so the spawn policy can be exercised without an
    `ssh` binary or a second machine.
    """
    port = local_port or free_port()
    argv = ssh_command(destination, port, remote_port, remote_host=remote_host, ssh=ssh)
    try:
        proc = launch(argv)
    except (OSError, ValueError) as exc:
        raise RemoteError(
            f"could not start {ssh} to {destination}: {type(exc).__name__}: {exc}",
            f"127.0.0.1:{port}",
        ) from exc
    live = Tunnel(proc=proc, local_port=port, destination=destination, remote_port=remote_port)
    try:
        if not live.wait_ready(timeout):
            code = proc.poll()
            gone = f"; {ssh} exited {code}" if code is not None else ""
            raise RemoteError(
                f"the tunnel to {destination} did not accept on 127.0.0.1:{port} "
                f"within {timeout:g}s{gone}",
                live.address,
            )
        yield live
    finally:
        live.close()


@contextmanager
def over_ssh(
    destination: str,
    remote_port: int,
    *,
    token: str = "",
    token_path: str = "",
    local_port: int = 0,
    timeout: float = TUNNEL_TIMEOUT,
    launch: Callable[[list[str]], subprocess.Popen] = _spawn,
) -> Iterator[RemoteAgent]:
    """A connected `RemoteAgent` reached through a tunnel that is then closed.

    The two halves are separate context managers so a caller who already has a
    tunnel does not have to open a second one, but this is the shape almost
    everybody wants and getting the nesting wrong is how a tunnel survives.
    """
    secret = token
    if not secret and token_path:
        secret = Path(os.path.expanduser(token_path)).read_text(encoding="utf-8").strip()
    with tunnel(destination, remote_port, local_port=local_port, timeout=timeout, launch=launch) as pipe:
        client = RemoteAgent(pipe.address, token=secret)
        with client:
            yield client


# -- shell wiring -----------------------------------------------------------
#
# `/remote run` streams events while it runs, but a command handler returns one
# Outcome at the end.  So events go two places: appended to the Outcome's lines,
# which is what a user reading scrollback afterwards wants, and pushed to a sink
# the app may install for live display.  The sink is a module-level hook rather
# than a state field so that wiring it is one line in the app and no change
# here.

_sink: EventHandler | None = None


def set_event_sink(handler: EventHandler | None) -> None:
    """Install the live display for remote events, or clear it.

    Called by the shell at startup.  One sink, not a list: there is one screen.
    """
    global _sink
    _sink = handler


def _line_for(name: str, params: dict[str, Any]) -> str:
    """One readable line per notification.

    Formatted here rather than in the app because the payload shapes are this
    module's business - they come off the wire - and a UI that had to know them
    would break the next time the bridge added a field.
    """
    if name == "agent.started":
        return f"  remote step {params.get('step', '?')} on {params.get('model', 'unknown')}"
    if name == "tool.started":
        return f"  remote {params.get('tool', 'tool')} started"
    if name == "tool.finished":
        mark = "ok" if params.get("ok") else "failed"
        return f"  remote {params.get('tool', 'tool')} {mark}"
    if name == "edit.applied":
        where = params.get("path") or params.get("tool") or "a file"
        return f"  remote edit: {where}"
    if name == "agent.finished":
        reason = params.get("reason", "done")
        error = params.get("error")
        return f"  remote finished: {reason}" + (f" ({error})" if error else "")
    return f"  remote {name}"


def _remote(state: Any, args: list[str]) -> Any:
    """`/remote add|list|remove|run`."""
    from offset.shell.commands import TONE_ERR, TONE_INFO, TONE_OK, Outcome

    home = settings.home()
    action = (args[0] if args else "list").lower()
    rest = args[1:]

    if action in ("list", "ls"):
        found = remotes(home)
        if not found:
            return Outcome(
                ["no remotes configured",
                 "add one with /remote add <name> <host:port> [token=<path>]"],
                TONE_INFO,
            )
        return Outcome([r.summary() for r in found], TONE_INFO)

    if action == "add":
        if len(rest) < 2:
            return Outcome.error("usage: /remote add <name> <address> [token=<path>] [workspace=<path>]")
        token_path = ""
        workspace = ""
        for extra in rest[2:]:
            key, sep, value = extra.partition("=")
            if not sep:
                return Outcome.error(f"{extra!r} is not key=value")
            if key in ("token", "token_path"):
                token_path = value
            elif key in ("workspace", "root"):
                workspace = value
            else:
                return Outcome.error(f"unknown option {key!r}; expected token= or workspace=")
        try:
            entry = add_remote(
                rest[0], rest[1], token_path=token_path, workspace=workspace, home=home
            )
        except RemoteError as exc:
            return Outcome.error(str(exc))
        return Outcome(
            [f"remote {entry.name} -> {entry.address}",
             f"token: {entry.token_file(home)}"],
            TONE_OK,
        )

    if action in ("remove", "rm", "forget"):
        if not rest:
            return Outcome.error("usage: /remote remove <name>")
        if not remove_remote(rest[0], home=home):
            return Outcome.error(f"no remote named {rest[0]!r}")
        return Outcome([f"forgot {rest[0]}"], TONE_OK)

    if action == "run":
        if len(rest) < 2:
            return Outcome.error("usage: /remote run <name> <prompt>")
        name = rest[0]
        prompt = " ".join(rest[1:]).strip()
        try:
            target = find_remote(name, home=home)
        except RemoteError as exc:
            return Outcome.error(str(exc))

        def job() -> Any:
            lines: list[str] = [f"{target.name} ({target.address}): {prompt}"]

            def observed(event: str, params: dict[str, Any]) -> None:
                lines.append(_line_for(event, params))
                if _sink is not None:
                    try:
                        _sink(event, params)
                    except Exception:
                        pass  # the display is never worth losing the turn over

            try:
                reply = run(target, prompt, on_event=observed, home=home)
            except RemoteError as exc:
                lines.append(str(exc))
                return Outcome(lines, TONE_ERR)
            text = str(reply.get("text") or reply.get("message") or "").strip()
            if text:
                lines.extend(text.splitlines())
            tone = TONE_OK if reply.get("ok", True) else TONE_ERR
            return Outcome(lines, tone)

        return Outcome(
            [f"running on {target.name} ({target.address})..."], TONE_INFO, job=job
        )

    return Outcome.error(
        f"unknown /remote action {action!r}; expected add, list, remove or run"
    )


def remote_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command(
            "remote",
            "run a turn on another machine's offset daemon",
            _remote,
            usage="/remote add <name> <address> [token=<path>] | /remote list | "
                  "/remote remove <name> | /remote run <name> <prompt>",
            aliases=("remotes",),
        ),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """`COMMANDS` on demand.

    Built lazily because the handler imports from `offset.shell.commands`,
    which imports this module: resolving at import time would be a cycle.  The
    re-check after building is the same guard `offset.core.tasks` needs, and
    for the same reason - importing the shell registry re-enters here before
    the outer call has stored anything, so a single access would otherwise
    produce two lists and register the command twice.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = remote_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
