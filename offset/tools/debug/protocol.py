"""The Debug Adapter Protocol on the wire: framing, channels, records.

DAP borrows LSP's framing — `Content-Length: N\\r\\n\\r\\n` then exactly N bytes
of JSON — and almost nothing else.  Where JSON-RPC has one envelope with an
`id`, DAP has three:

  * a *request* carries `seq`, `type: "request"`, `command`, `arguments`;
  * a *response* carries its own `seq`, `type: "response"`, `request_seq`
    naming the request it answers, and `success` plus a `body` or `message`;
  * an *event* carries `seq`, `type: "event"`, `event` and a `body`, and is
    never a reply to anything.

That third envelope is why this module exists as more than a rename of the MCP
client.  Almost everything worth knowing about a debuggee — that it stopped, why
it stopped, what it printed — arrives unsolicited, so the framing layer has to
deliver events with the same care as replies rather than treating them as
noise.  Correlation and ordering live in `client.py`; this module only knows
bytes, records, and how to stop owning a process.

Two rules stated out loud, both learned from adapters in the wild:

  * a stdout line that is not a header is dropped, not fatal.  Adapters print
    banners and warnings, and resynchronising at the next `Content-Length` is
    always safe because we have not yet consumed a body.  A broken
    *Content-Length* is the opposite: the stream can no longer be trusted, so
    that one is fatal.
  * an adapter is started in its own process group.  A debuggee is a child of
    the adapter, and a debuggee that outlives the session is the worst orphan
    in this codebase — it holds the port, the file lock, and the CPU.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import socket
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, IO

#: Longest single frame accepted.  An adapter stuck in a log loop must not be
#: able to grow this process without bound.
MAX_FRAME: Final = 8 << 20

#: Poll slice for queue waits: small enough that close() is responsive, large
#: enough that an idle session costs nothing.
_TICK: Final = 0.05

#: Sentinel enqueued by a reader thread when the peer will never speak again.
_GONE: Final = object()

ENCODING: Final = "utf-8"

#: How long to keep trying to reach a server-mode adapter while it boots.
CONNECT_GRACE: Final = 10.0


class ProtocolError(Exception):
    """A DAP-level failure: bad framing, or an adapter that cannot serve."""


class FramingError(ProtocolError):
    """The byte stream can no longer be resynchronised.  Fatal for a channel."""


class AdapterClosed(ProtocolError):
    """The adapter is gone, so no reply can ever arrive."""


# -- framing ----------------------------------------------------------------


def encode(message: dict[str, Any]) -> bytes:
    """One DAP frame: the header block, then the compact JSON body."""
    body = json.dumps(message, separators=(",", ":")).encode(ENCODING)
    return b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body


def read_frame(
    stream: IO[bytes],
    *,
    on_drop: Callable[[str], None] = lambda _why: None,
) -> dict[str, Any] | None:
    """Read one frame, or None at end of stream.

    `on_drop` is called with a reason for anything discarded that did not cost
    us synchronisation — a banner line, a body that is not a JSON object.
    """
    length = -1
    while True:
        line = stream.readline()
        if not line:
            return None
        text = line.strip()
        if not text:
            if length >= 0:
                break  # the blank line that terminates a real header block
            continue  # blank noise before any header; still resynchronised
        if b":" not in text:
            on_drop(f"line is not a header: {text[:80]!r}")
            continue
        name, _, value = text.partition(b":")
        if name.strip().lower() != b"content-length":
            continue  # Content-Type and friends carry nothing we need
        try:
            length = int(value.strip())
        except ValueError as exc:
            raise FramingError(f"unreadable Content-Length {value.strip()!r}") from exc
        if length < 0 or length > MAX_FRAME:
            raise FramingError(f"frame of {length} bytes is outside 0..{MAX_FRAME}")

    body = stream.read(length)
    if body is None or len(body) < length:
        return None  # the peer died part way through a body
    try:
        frame = json.loads(body.decode(ENCODING, "replace"))
    except ValueError:
        on_drop("body is not JSON")
        return read_frame(stream, on_drop=on_drop)
    if not isinstance(frame, dict):
        on_drop("body is not a JSON object")
        return read_frame(stream, on_drop=on_drop)
    return frame


def request_frame(seq: int, command: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    frame: dict[str, Any] = {"seq": seq, "type": "request", "command": command}
    # Adapters differ on whether `arguments` may be absent; sending `{}` is
    # accepted everywhere, so the empty case is normalised rather than omitted.
    frame["arguments"] = arguments if arguments is not None else {}
    return frame


@dataclass(slots=True)
class Response:
    """A reply, addressed to a request by `request_seq`."""

    seq: int
    request_seq: int
    command: str
    success: bool
    message: str = ""
    body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Response":
        body = raw.get("body")
        return cls(
            seq=_int(raw.get("seq")) or 0,
            request_seq=_int(raw.get("request_seq")) or 0,
            command=str(raw.get("command") or ""),
            success=raw.get("success") is True,
            message=str(raw.get("message") or ""),
            body=body if isinstance(body, dict) else {},
        )


@dataclass(slots=True)
class Event:
    """An unsolicited notification.  Never a reply, so it has no request_seq."""

    seq: int
    event: str
    body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Event":
        body = raw.get("body")
        return cls(
            seq=_int(raw.get("seq")) or 0,
            event=str(raw.get("event") or ""),
            body=body if isinstance(body, dict) else {},
        )


def classify(frame: dict[str, Any]) -> Response | Event | None:
    """Sort one frame into the trichotomy, or None for something unusable.

    A missing `type` is inferred from the fields present, because a handful of
    adapters omit it on responses.
    """
    kind = frame.get("type")
    if kind == "response" or (kind is None and "request_seq" in frame):
        return Response.parse(frame)
    if kind == "event" or (kind is None and "event" in frame):
        return Event.parse(frame)
    return None


# -- records ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceBreakpoint:
    """A breakpoint as the client asks for it, before the adapter rules on it."""

    line: int
    condition: str = ""
    hit_condition: str = ""
    log_message: str = ""

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"line": self.line}
        if self.condition:
            out["condition"] = self.condition
        if self.hit_condition:
            out["hitCondition"] = self.hit_condition
        if self.log_message:
            out["logMessage"] = self.log_message
        return out


@dataclass(slots=True)
class Breakpoint:
    """A breakpoint as the adapter ruled on it.

    `verified=False` is the interesting case and the one callers ignore at
    their peril: the adapter accepted the request and bound nothing, so
    execution will sail past the line the user is watching.
    """

    verified: bool
    id: int | None = None
    line: int | None = None
    source: str = ""
    message: str = ""

    @classmethod
    def parse(cls, raw: dict[str, Any], *, source: str = "") -> "Breakpoint":
        where = raw.get("source")
        where = where if isinstance(where, dict) else {}
        return cls(
            verified=raw.get("verified") is True,
            id=_int(raw.get("id")),
            line=_int(raw.get("line")),
            source=str(where.get("path") or where.get("name") or source),
            message=str(raw.get("message") or ""),
        )

    def describe(self) -> str:
        where = f"{_short(self.source)}:{self.line}" if self.line is not None else _short(self.source)
        state = "verified" if self.verified else "unverified"
        detail = f" ({self.message})" if self.message else ""
        return f"{where} {state}{detail}"


@dataclass(slots=True)
class StackFrame:
    id: int
    name: str
    line: int = 0
    column: int = 0
    source: str = ""

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "StackFrame":
        where = raw.get("source")
        where = where if isinstance(where, dict) else {}
        return cls(
            id=_int(raw.get("id")) or 0,
            name=str(raw.get("name") or "<anonymous>"),
            line=_int(raw.get("line")) or 0,
            column=_int(raw.get("column")) or 0,
            source=str(where.get("path") or where.get("name") or ""),
        )

    def describe(self) -> str:
        where = f"{_short(self.source)}:{self.line}" if self.source else f"line {self.line}"
        return f"{self.name} at {where}"


def frames_report(frames: list[StackFrame]) -> list[str]:
    """The stack as a model can read it: innermost first, with frame ids.

    The id is shown because every follow-up call (`scopes`, `evaluate`) needs
    it, and a model that cannot see it has to guess.
    """
    if not frames:
        return ["no stack: the program is not stopped"]
    return [f"#{i} [frame {f.id}] {f.describe()}" for i, f in enumerate(frames)]


@dataclass(slots=True)
class Scope:
    name: str
    variables_reference: int
    expensive: bool = False
    named: int = 0
    indexed: int = 0

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Scope":
        return cls(
            name=str(raw.get("name") or "scope"),
            variables_reference=_int(raw.get("variablesReference")) or 0,
            expensive=raw.get("expensive") is True,
            named=_int(raw.get("namedVariables")) or 0,
            indexed=_int(raw.get("indexedVariables")) or 0,
        )

    def describe(self) -> str:
        count = self.named + self.indexed
        size = f", {count} variables" if count else ""
        cost = ", expensive" if self.expensive else ""
        return f"{self.name} [ref {self.variables_reference}{size}{cost}]"


#: Longest rendered value.  A container's repr can be megabytes and the model
#: only needs enough to recognise it.
VALUE_LIMIT: Final = 240


@dataclass(slots=True)
class Variable:
    name: str
    value: str = ""
    type: str = ""
    variables_reference: int = 0
    children: list["Variable"] = field(default_factory=list)

    @property
    def expandable(self) -> bool:
        return self.variables_reference > 0

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Variable":
        return cls(
            name=str(raw.get("name") or ""),
            value=str(raw.get("value") if raw.get("value") is not None else ""),
            type=str(raw.get("type") or ""),
            variables_reference=_int(raw.get("variablesReference")) or 0,
        )

    def report(self, *, indent: int = 0, limit: int = 40) -> list[str]:
        """Nested structure flattened to bounded, indented text.

        A raw `variablesReference` is useless to a model — it cannot be read
        and cannot be followed without another round trip — so the tree is
        expanded before rendering and the reference is only mentioned when
        there is more to fetch than was expanded.
        """
        pad = "  " * indent
        value = " ".join(self.value.split())
        if len(value) > VALUE_LIMIT:
            value = value[: VALUE_LIMIT - 1] + "…"
        kind = f" : {self.type}" if self.type else ""
        more = ""
        if self.expandable and not self.children:
            more = f" (expandable, ref {self.variables_reference})"
        out = [f"{pad}{self.name}{kind} = {value}{more}".rstrip()]
        for child in self.children[:limit]:
            out.extend(child.report(indent=indent + 1, limit=limit))
        if len(self.children) > limit:
            out.append(f"{pad}  … {len(self.children) - limit} more")
        return out


@dataclass(slots=True)
class Thread:
    id: int
    name: str = ""

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Thread":
        return cls(id=_int(raw.get("id")) or 0, name=str(raw.get("name") or ""))

    def describe(self) -> str:
        return f"thread {self.id}" + (f" ({self.name})" if self.name else "")


@dataclass(slots=True)
class Capabilities:
    """What the adapter said it can do, in its own words plus the few flags
    the client branches on.

    Absence means "assume not": an adapter that is asked for something it
    never advertised answers `success: false`, and a failed configuration step
    is far worse than a feature we declined to use.
    """

    raw: dict[str, Any] = field(default_factory=dict)

    def supports(self, name: str) -> bool:
        return self.raw.get(name) is True

    @property
    def configuration_done(self) -> bool:
        return self.supports("supportsConfigurationDoneRequest")

    @property
    def function_breakpoints(self) -> bool:
        return self.supports("supportsFunctionBreakpoints")

    @property
    def conditional_breakpoints(self) -> bool:
        return self.supports("supportsConditionalBreakpoints")

    @property
    def hit_conditional_breakpoints(self) -> bool:
        return self.supports("supportsHitConditionalBreakpoints")

    @property
    def terminate_request(self) -> bool:
        return self.supports("supportsTerminateRequest")

    @property
    def exception_filters(self) -> list[dict[str, Any]]:
        filters = self.raw.get("exceptionBreakpointFilters")
        return [f for f in filters if isinstance(f, dict)] if isinstance(filters, list) else []

    def report(self) -> list[str]:
        declared = sorted(k for k, v in self.raw.items() if v is True)
        out = [f"{len(declared)} capabilities declared"]
        out.extend(f"  {name}" for name in declared)
        for entry in self.exception_filters:
            label = entry.get("label") or entry.get("filter")
            out.append(f"  exception filter {entry.get('filter')}: {label}")
        return out


# -- channels ---------------------------------------------------------------


class Channel(ABC):
    """Byte framing plus liveness.  Knows nothing about DAP semantics."""

    #: Frames or lines discarded because they were not usable.
    dropped: int = 0

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def send(self, message: dict[str, Any]) -> None: ...

    @abstractmethod
    def receive(self, timeout: float) -> dict[str, Any] | None:
        """The next frame, or None if `timeout` elapsed first.

        Raises AdapterClosed once the peer is gone, and keeps raising it.
        """

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def alive(self) -> bool: ...

    def diagnostics(self) -> str:
        """One line a human or a model can act on when this channel died."""
        return "channel closed"

    def stderr_tail(self) -> list[str]:
        return []


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, falling back to the child.

    The group is the point: the debuggee is a *grandchild* here, and killing
    only the adapter leaves the program under test running.
    """
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


class _Child:
    """The process-ownership half of both channels: spawn, reap, explain."""

    __slots__ = ("_proc", "_stderr", "_thread", "args", "command", "cwd", "env", "grace")

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: Path | str | None = None,
        grace: float = 2.0,
    ) -> None:
        self.command = command
        self.args = list(args or ())
        self.env = dict(env or {})
        self.cwd = cwd
        self.grace = grace
        self._proc: subprocess.Popen | None = None
        self._stderr: deque[str] = deque(maxlen=60)
        self._thread: threading.Thread | None = None

    @property
    def proc(self) -> subprocess.Popen | None:
        return self._proc

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def spawn(self, *, pipe_stdout: bool) -> subprocess.Popen:
        if self._proc is not None:
            raise ProtocolError(f"{self.command} was already started")
        argv = [self.command, *self.args]
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE if pipe_stdout else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.cwd) if self.cwd else None,
                env={**os.environ, **self.env},
                start_new_session=True,  # so close() can reap the whole tree
            )
        except (OSError, ValueError) as exc:
            raise ProtocolError(f"could not start {self.command!r}: {exc}") from exc
        self._thread = threading.Thread(
            target=self._pump_stderr, name=f"dap-err-{self._proc.pid}", daemon=True
        )
        self._thread.start()
        return self._proc

    def _pump_stderr(self) -> None:
        stream = self._proc.stderr if self._proc else None
        if stream is None:
            return
        try:
            for line in stream:
                self._stderr.append(line.decode(ENCODING, "replace").rstrip())
        except (OSError, ValueError):
            pass

    def tail(self) -> list[str]:
        return list(self._stderr)

    def describe(self) -> str:
        if self._proc is None:
            return f"{self.command} was never started"
        code = self._proc.poll()
        state = "still running" if code is None else f"exited {code}"
        tail = " | ".join(line for line in list(self._stderr)[-3:] if line)
        return f"{self.command} {state}" + (f": {tail[:300]}" if tail else "")

    def reap(self) -> None:
        """SIGTERM the group, then SIGKILL it.  Safe to call twice."""
        proc = self._proc
        if proc is None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
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
        if self._thread is not None:
            self._thread.join(timeout=self.grace)


class StdioChannel(Channel):
    """An adapter run as a child process, DAP frames on its stdin/stdout."""

    __slots__ = ("_child", "_closed", "_gone", "_queue", "_reader", "_write", "dropped")

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: Path | str | None = None,
        grace: float = 2.0,
    ) -> None:
        self._child = _Child(command, args, env=env, cwd=cwd, grace=grace)
        self._queue: queue.Queue[Any] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._write = threading.Lock()
        self._closed = False
        self._gone = False
        self.dropped = 0

    @property
    def command(self) -> str:
        return self._child.command

    @property
    def pid(self) -> int | None:
        return self._child.pid

    def start(self) -> None:
        proc = self._child.spawn(pipe_stdout=True)
        self._closed = False
        self._gone = False
        self._reader = threading.Thread(
            target=self._pump, name=f"dap-out-{proc.pid}", daemon=True
        )
        self._reader.start()

    @property
    def alive(self) -> bool:
        proc = self._child.proc
        return proc is not None and proc.poll() is None and not self._closed

    def diagnostics(self) -> str:
        return self._child.describe()

    def stderr_tail(self) -> list[str]:
        return self._child.tail()

    def send(self, message: dict[str, Any]) -> None:
        proc = self._child.proc
        if proc is None or proc.stdin is None:
            raise AdapterClosed(f"{self._child.command} is not running")
        if self._closed or proc.poll() is not None:
            raise AdapterClosed(self.diagnostics())
        try:
            with self._write:
                proc.stdin.write(encode(message))
                proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise AdapterClosed(f"{self.diagnostics()} ({exc})") from exc

    def receive(self, timeout: float) -> dict[str, Any] | None:
        if self._gone:
            raise AdapterClosed(self.diagnostics())
        try:
            item = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if item is _GONE:
            self._gone = True
            raise AdapterClosed(self.diagnostics())
        return item

    def _pump(self) -> None:
        proc = self._child.proc
        stream = proc.stdout if proc else None
        if stream is None:
            self._queue.put(_GONE)
            return
        try:
            while True:
                frame = read_frame(stream, on_drop=self._drop)
                if frame is None:
                    return
                self._queue.put(frame)
        except (FramingError, OSError, ValueError):
            return
        finally:
            self._queue.put(_GONE)

    def _drop(self, _why: str) -> None:
        self.dropped += 1

    def close(self) -> None:
        self._closed = True
        self._child.reap()
        if self._reader is not None:
            self._reader.join(timeout=self._child.grace)
        self._queue.put(_GONE)


def free_port() -> int:
    """A port the OS says is free right now.

    Racy in principle and correct in practice: server-mode adapters have no way
    to tell us which port they picked, so the client has to choose one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class SocketChannel(Channel):
    """An adapter that listens on TCP rather than speaking on its stdio.

    `dlv dap` and `rdbg --open` only work this way, so this is not an optional
    transport.  The adapter may be spawned by us (and is then owned by us, down
    to the process group) or already running, for attaching to a live process.
    """

    __slots__ = (
        "_child",
        "_closed",
        "_gone",
        "_queue",
        "_reader",
        "_sock",
        "_stream",
        "_write",
        "connect_grace",
        "dropped",
        "host",
        "port",
    )

    def __init__(
        self,
        host: str,
        port: int,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | str | None = None,
        grace: float = 2.0,
        connect_grace: float = CONNECT_GRACE,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_grace = connect_grace
        self._child = _Child(command, args, env=env, cwd=cwd, grace=grace) if command else None
        self._sock: socket.socket | None = None
        self._stream: IO[bytes] | None = None
        self._queue: queue.Queue[Any] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._write = threading.Lock()
        self._closed = False
        self._gone = False
        self.dropped = 0

    @property
    def pid(self) -> int | None:
        return self._child.pid if self._child else None

    def start(self) -> None:
        if self._child is not None:
            self._child.spawn(pipe_stdout=False)
        self._sock = self._connect()
        self._stream = self._sock.makefile("rb")
        self._closed = False
        self._gone = False
        self._reader = threading.Thread(
            target=self._pump, name=f"dap-sock-{self.port}", daemon=True
        )
        self._reader.start()

    def _connect(self) -> socket.socket:
        deadline = time.monotonic() + self.connect_grace
        last = ""
        while time.monotonic() < deadline:
            proc = self._child.proc if self._child else None
            if proc is not None and proc.poll() is not None:
                self.close()
                raise AdapterClosed(
                    f"adapter exited before it listened on {self.host}:{self.port}: "
                    f"{self._child.describe() if self._child else ''}"
                )
            try:
                return socket.create_connection((self.host, self.port), timeout=2.0)
            except OSError as exc:
                last = str(exc)
                time.sleep(_TICK)
        self.close()
        raise AdapterClosed(
            f"nothing accepted a connection on {self.host}:{self.port} within "
            f"{self.connect_grace:g}s ({last})"
        )

    @property
    def alive(self) -> bool:
        if self._closed or self._sock is None:
            return False
        proc = self._child.proc if self._child else None
        return proc is None or proc.poll() is None

    def diagnostics(self) -> str:
        where = f"dap on {self.host}:{self.port}"
        return f"{where}; {self._child.describe()}" if self._child else where

    def stderr_tail(self) -> list[str]:
        return self._child.tail() if self._child else []

    def send(self, message: dict[str, Any]) -> None:
        sock = self._sock
        if sock is None or self._closed:
            raise AdapterClosed(self.diagnostics())
        try:
            with self._write:
                sock.sendall(encode(message))
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise AdapterClosed(f"{self.diagnostics()} ({exc})") from exc

    def receive(self, timeout: float) -> dict[str, Any] | None:
        if self._gone:
            raise AdapterClosed(self.diagnostics())
        try:
            item = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if item is _GONE:
            self._gone = True
            raise AdapterClosed(self.diagnostics())
        return item

    def _pump(self) -> None:
        stream = self._stream
        if stream is None:
            self._queue.put(_GONE)
            return
        try:
            while True:
                frame = read_frame(stream, on_drop=self._drop)
                if frame is None:
                    return
                self._queue.put(frame)
        except (FramingError, OSError, ValueError):
            return
        finally:
            self._queue.put(_GONE)

    def _drop(self, _why: str) -> None:
        self.dropped += 1

    def close(self) -> None:
        self._closed = True
        for closable in (self._stream, self._sock):
            if closable is not None:
                try:
                    closable.close()
                except OSError:
                    pass
        self._stream = None
        self._sock = None
        if self._child is not None:
            self._child.reap()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        self._queue.put(_GONE)


# -- helpers ----------------------------------------------------------------


def _int(value: Any) -> int | None:
    """Adapters echo numbers back as strings often enough to matter."""
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _short(path: str) -> str:
    """A path a human can scan: the last two components, or the whole thing."""
    if not path:
        return "<unknown>"
    parts = Path(path).parts
    return str(Path(*parts[-2:])) if len(parts) > 2 else path
