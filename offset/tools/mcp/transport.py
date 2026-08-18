"""Framing for the two MCP transports: child process stdio, and HTTP POST.

Framing and process ownership live here; request correlation and the protocol
itself live in `client.py`.  The split exists because the ways these two fail
are completely different — a stdio server dies, an HTTP endpoint refuses a
connection — while the client above must see one signal, `TransportClosed`, and
never a caller left waiting on a reply that can no longer arrive.

Two rules worth stating out loud:

  * A frame that is not JSON is dropped, not fatal.  Servers print banners,
    warnings and tracebacks onto stdout; losing the session over one stray
    line would make offset unusable with real-world servers.
  * A stdio server is started in its own process group so closing it can reap
    the whole tree.  Servers spawn helpers, and a helper that outlives the
    session is an orphan holding the port or the file lock.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any, Final

from offset.providers.transport import HTTPFailure

#: Longest single frame accepted.  A server stuck in a print loop must not be
#: able to grow this process without bound.
MAX_FRAME: Final = 8 << 20

#: Poll slice for queue waits.  Small enough that a close is responsive,
#: large enough that idle transports cost nothing.
_TICK: Final = 0.05

#: Sentinel enqueued by a reader thread when the peer will never speak again.
_GONE: Final = object()

SESSION_HEADER: Final = "Mcp-Session-Id"


class TransportError(Exception):
    """Framing or connection failure, as opposed to a protocol-level error."""


class TransportClosed(TransportError):
    """The peer is gone.  Anything still waiting for a reply must give up."""


class Transport(ABC):
    """Byte framing plus liveness.  Knows nothing about JSON-RPC semantics."""

    #: Frames discarded because they were not a JSON object.
    dropped: int = 0

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def send(self, message: dict[str, Any]) -> None: ...

    @abstractmethod
    def receive(self, timeout: float) -> dict[str, Any] | None:
        """The next message, or None if `timeout` elapsed first.

        Raises TransportClosed once the peer is gone, and keeps raising it.
        """

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def alive(self) -> bool: ...

    def diagnostics(self) -> str:
        """One line a human or a model can act on when this transport died."""
        return "transport closed"


# -- stdio ------------------------------------------------------------------


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, falling back to the child."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


class StdioTransport(Transport):
    """A server run as a child process, newline-delimited JSON on its stdio."""

    __slots__ = (
        "_closed",
        "_gone",
        "_proc",
        "_queue",
        "_readers",
        "_stderr",
        "_write",
        "args",
        "command",
        "cwd",
        "dropped",
        "env",
        "grace",
    )

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
        self.dropped = 0
        self._proc: subprocess.Popen | None = None
        self._queue: queue.Queue[Any] = queue.Queue()
        self._readers: list[threading.Thread] = []
        self._stderr: deque[str] = deque(maxlen=40)
        self._gone = False
        self._closed = False
        self._write = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._proc is not None:
            raise TransportError("transport already started")
        argv = [self.command, *self.args]
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=str(self.cwd) if self.cwd else None,
                env={**os.environ, **self.env},
                start_new_session=True,  # so close() can reap the whole tree
            )
        except (OSError, ValueError) as exc:
            raise TransportError(f"could not start {self.command!r}: {exc}") from exc
        self._closed = False
        self._gone = False
        self._readers = [
            threading.Thread(target=self._pump_stdout, name=f"mcp-out-{self._proc.pid}", daemon=True),
            threading.Thread(target=self._pump_stderr, name=f"mcp-err-{self._proc.pid}", daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and not self._closed

    def diagnostics(self) -> str:
        if self._proc is None:
            return f"{self.command} was never started"
        code = self._proc.poll()
        tail = " | ".join(line for line in list(self._stderr)[-3:] if line)
        state = "still running" if code is None else f"exited {code}"
        return f"{self.command} {state}" + (f": {tail[:300]}" if tail else "")

    def close(self) -> None:
        """Stop the server and reap its process group; safe to call twice."""
        self._closed = True
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
        for reader in self._readers:
            reader.join(timeout=self.grace)
        self._queue.put(_GONE)

    # -- framing ------------------------------------------------------------

    def send(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise TransportClosed(f"{self.command} is not running")
        if self._closed or proc.poll() is not None:
            raise TransportClosed(self.diagnostics())
        line = json.dumps(message, separators=(",", ":")) + "\n"
        try:
            with self._write:
                proc.stdin.write(line)
                proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise TransportClosed(f"{self.diagnostics()} ({exc})") from exc

    def receive(self, timeout: float) -> dict[str, Any] | None:
        if self._gone:
            raise TransportClosed(self.diagnostics())
        try:
            item = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if item is _GONE:
            self._gone = True
            raise TransportClosed(self.diagnostics())
        return item

    def _pump_stdout(self) -> None:
        stream = self._proc.stdout if self._proc else None
        if stream is None:
            return
        try:
            for line in stream:
                text = line.strip()
                if not text:
                    continue
                if len(text) > MAX_FRAME:
                    self.dropped += 1
                    continue
                try:
                    frame = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    self.dropped += 1  # a banner or a traceback, not a frame
                    continue
                if not isinstance(frame, dict):
                    self.dropped += 1
                    continue
                self._queue.put(frame)
        except (OSError, ValueError):
            pass
        finally:
            self._queue.put(_GONE)

    def _pump_stderr(self) -> None:
        stream = self._proc.stderr if self._proc else None
        if stream is None:
            return
        try:
            for line in stream:
                self._stderr.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def stderr_tail(self) -> list[str]:
        return list(self._stderr)


# -- streamable HTTP --------------------------------------------------------


class HTTPTransport(Transport):
    """Streamable HTTP: each send is a POST whose replies are enqueued.

    Requests are deliberately never retried here.  A `tools/call` that reached
    the server may already have had its effect, so a transparent retry could
    run a remote tool twice; the manager retries the *connection* instead.

    A failed POST is turned into a JSON-RPC error response addressed to the
    request that caused it, because a caller waiting on a reply must be told,
    not left to time out.
    """

    __slots__ = (
        "_broken",
        "_closed",
        "_last",
        "_outbox",
        "_queue",
        "_sender",
        "dropped",
        "headers",
        "session",
        "timeout",
        "url",
    )

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.dropped = 0
        self.session: str | None = None
        self._queue: queue.Queue[Any] = queue.Queue()
        #: Outgoing messages go through ONE worker so they reach the server in
        #: submission order. A thread per message let `tools/list` overtake
        #: `notifications/initialized`, which the protocol forbids.
        self._outbox: queue.Queue[Any] = queue.Queue()
        self._sender: threading.Thread | None = None
        self._closed = False
        self._broken = False
        self._last = ""

    def start(self) -> None:
        self._closed = False
        self._broken = False
        self._ensure_sender()

    def _ensure_sender(self) -> None:
        if self._sender is not None and self._sender.is_alive():
            return
        self._sender = threading.Thread(target=self._pump, name="mcp-http-send", daemon=True)
        self._sender.start()

    def _pump(self) -> None:
        while True:
            message = self._outbox.get()
            if message is _GONE:
                return
            try:
                self._exchange(message)
            except Exception as exc:  # a dead sender would stall every later send
                self._last = f"{type(exc).__name__}: {exc}"

    @property
    def alive(self) -> bool:
        return not self._closed and not self._broken

    def diagnostics(self) -> str:
        return f"{self.url}: {self._last}" if self._last else self.url

    def close(self) -> None:
        self._closed = True
        self._outbox.put(_GONE)
        if self._sender is not None:
            self._sender.join(timeout=self.timeout if self.timeout < 5 else 5.0)
            self._sender = None
        self._queue.put(_GONE)

    def send(self, message: dict[str, Any]) -> None:
        if self._closed:
            raise TransportClosed(f"{self.url} is closed")
        if self._broken:
            raise TransportClosed(self.diagnostics())
        self._ensure_sender()
        self._outbox.put(message)

    def receive(self, timeout: float) -> dict[str, Any] | None:
        try:
            item = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if item is _GONE:
            raise TransportClosed(self.diagnostics())
        return item

    def _exchange(self, message: dict[str, Any]) -> None:
        try:
            headers, body = self._post(message)
        except HTTPFailure as exc:
            self._last = str(exc)
            # No connection at all: the endpoint is gone, not just this call.
            if exc.status == 0:
                self._broken = True
            self._fail(message, exc.detail() or str(exc))
            return
        session = headers.get(SESSION_HEADER.lower())
        if session:
            self.session = session
        for frame in self._frames(headers.get("content-type", ""), body):
            self._queue.put(frame)

    def _post(self, message: dict[str, Any]) -> tuple[dict[str, str], str]:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        sent = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self.session:
            sent[SESSION_HEADER] = self.session
        request = urllib.request.Request(self.url, data=payload, headers=sent, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                headers = {k.lower(): v for k, v in response.headers.items()}
                return headers, raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
            raise HTTPFailure(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise HTTPFailure(0, f"{exc.reason}") from exc
        except TimeoutError as exc:
            raise HTTPFailure(0, "request timed out") from exc

    def _frames(self, content_type: str, body: str) -> list[dict[str, Any]]:
        if not body.strip():
            return []  # 202 Accepted for a notification
        chunks = self._sse(body) if "event-stream" in content_type else [body]
        out: list[dict[str, Any]] = []
        for chunk in chunks:
            try:
                parsed = json.loads(chunk)
            except (json.JSONDecodeError, ValueError):
                self.dropped += 1
                continue
            for frame in parsed if isinstance(parsed, list) else [parsed]:
                if isinstance(frame, dict):
                    out.append(frame)
                else:
                    self.dropped += 1
        return out

    @staticmethod
    def _sse(body: str) -> list[str]:
        """Data payloads of an SSE body; multi-line data blocks are joined."""
        out: list[str] = []
        buffer: list[str] = []
        for line in body.splitlines():
            if line.startswith("data:"):
                buffer.append(line[5:].lstrip())
            elif not line.strip():
                if buffer:
                    out.append("\n".join(buffer))
                    buffer = []
        if buffer:
            out.append("\n".join(buffer))
        return out

    def _fail(self, message: dict[str, Any], detail: str) -> None:
        ident = message.get("id")
        if ident is None:
            return  # a notification has nobody waiting
        self._queue.put({
            "jsonrpc": "2.0",
            "id": ident,
            "error": {"code": -32000, "message": f"http transport: {detail[:300]}"},
        })
