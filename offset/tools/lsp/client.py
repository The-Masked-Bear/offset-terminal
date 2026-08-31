"""JSON-RPC 2.0 over a language server's stdio: handshake, sync, correlation.

A language server is a long-lived child process that talks in both directions
at once, and that is the whole difficulty.  Replies arrive out of order, the
server asks *us* questions mid-request, and the single most useful thing it
produces — diagnostics — is not a reply to anything.  So a reader thread owns
the pipe, every request is correlated by id and carries its own deadline, and
server-initiated requests are always answered.  A server left waiting for an
answer stops working, and a server that stops working looks exactly like a slow
one, which is the most expensive kind of bug to find.

Diagnostics get a deliberate policy rather than a convenience.  The server
pushes `textDocument/publishDiagnostics` whenever it feels like it: often
before we have asked anything, sometimes several times as an analysis settles,
and for a clean file frequently never at all.  Two obvious implementations are
both wrong.  Asking and waiting for a publication blocks forever on the clean
file; asking and reading whatever has arrived returns empty on the file that
was about to be reported.  So every publication is buffered per URI for the
life of the connection, a query returns the newest buffered set *immediately*
if that URI has ever been published for, and only an unheard-of URI waits — for
a bounded time, after which empty means "the server had nothing to say", which
is the truthful answer and the common case.  An empty array counts as a
publication: that is how a server retracts the errors it reported a moment ago.

Document versions are the other trap.  `didChange` must carry a version
strictly greater than the last one the server saw, or a conforming server
discards the change and answers every later question about a file it thinks
still has the old text.  The version counter and the notification that carries
it are therefore incremented and written under one lock, so two threads editing
two files cannot interleave into a decreasing sequence on the wire.

Framing, URIs and result shapes live in `protocol`; nothing here parses bytes.
"""

from __future__ import annotations

import io
import itertools
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

from offset.tools.lsp.protocol import (
    MAX_FRAME,
    CodeAction,
    Diagnostic,
    Framer,
    Location,
    Position,
    Range,
    Symbol,
    TextEdit,
    WorkspaceEdit,
    encode,
    to_uri,
)
from offset.tools.lsp.protocol import code_actions as parse_code_actions
from offset.tools.lsp.protocol import diagnostics as parse_diagnostics
from offset.tools.lsp.protocol import locations as parse_locations
from offset.tools.lsp.protocol import normalise_uri
from offset.tools.lsp.protocol import symbols as parse_symbols
from offset.tools.lsp.protocol import text_edits as parse_text_edits

CLIENT_NAME: Final = "offset"
CLIENT_VERSION: Final = "0.1.0"

#: Wait slice while a request is outstanding: the granularity of both the
#: deadline and cooperative cancellation.
_TICK: Final = 0.05

#: One read from the server's stdout.  The pipe hands over whatever arrived, so
#: this is a ceiling and never a promise.
_CHUNK: Final = 1 << 16

#: How long `diagnostics()` waits for a URI nothing has been published for.
#: Long enough for a server to finish parsing one freshly opened file, short
#: enough that a server which will never publish does not stall the turn.
DIAGNOSTIC_WAIT: Final = 2.5

#: How long the process gets to leave after `shutdown`/`exit` before the group
#: is signalled.
GRACE: Final = 2.0

#: Method -> (the capability that must be declared, how to say it in English).
#: Sending a request the server never claimed to serve earns either a
#: `-32601` or, from several real servers, silence until the deadline; a named
#: refusal here is both faster and something the caller can act on.
PROVIDERS: Final[dict[str, tuple[str, str]]] = {
    "textDocument/definition": ("definitionProvider", "go to definition"),
    "textDocument/typeDefinition": ("typeDefinitionProvider", "go to type definition"),
    "textDocument/implementation": ("implementationProvider", "go to implementation"),
    "textDocument/references": ("referencesProvider", "find references"),
    "textDocument/hover": ("hoverProvider", "hover"),
    "textDocument/documentSymbol": ("documentSymbolProvider", "document symbols"),
    "workspace/symbol": ("workspaceSymbolProvider", "workspace symbol search"),
    "textDocument/rename": ("renameProvider", "rename"),
    "textDocument/codeAction": ("codeActionProvider", "code actions"),
    "textDocument/formatting": ("documentFormattingProvider", "formatting"),
}

#: What offset tells the server it can do.  Nothing is advertised that is not
#: implemented below: `applyEdit` is absent because a server-driven edit is not
#: what the caller asked for, and every `dynamicRegistration` is false because
#: registrations are acknowledged rather than honoured.
CAPABILITIES: Final[dict[str, Any]] = {
    "general": {"positionEncodings": ["utf-16"]},
    "workspace": {
        "applyEdit": False,
        "configuration": True,
        "workspaceFolders": True,
        "didChangeConfiguration": {"dynamicRegistration": False},
        "symbol": {"dynamicRegistration": False},
    },
    "textDocument": {
        "synchronization": {
            "dynamicRegistration": False,
            "willSave": False,
            "willSaveWaitUntil": False,
            "didSave": False,
        },
        "publishDiagnostics": {"relatedInformation": True, "versionSupport": True},
        "definition": {"linkSupport": True},
        "typeDefinition": {"linkSupport": True},
        "implementation": {"linkSupport": True},
        "references": {"dynamicRegistration": False},
        "hover": {"contentFormat": ["markdown", "plaintext"]},
        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
        "rename": {"prepareSupport": True},
        "codeAction": {
            "isPreferredSupport": True,
            "disabledSupport": True,
            "dataSupport": True,
            "resolveSupport": {"properties": ["edit"]},
            "codeActionLiteralSupport": {"codeActionKind": {"valueSet": ["quickfix", "refactor", "source"]}},
        },
        "formatting": {"dynamicRegistration": False},
    },
    "window": {"workDoneProgress": True, "showMessage": {}},
}


class LSPError(Exception):
    """A protocol-level failure, including a JSON-RPC error response."""

    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class LSPTimeout(LSPError):
    """The server did not answer in time.  Never a user cancellation."""


class ServerGone(LSPError):
    """The process died, so no reply can ever arrive."""


class LSPCancelled(LSPError):
    """The caller asked to stop; the server was told with `$/cancelRequest`."""


class Unsupported(LSPError):
    """The server never declared the capability the request needs."""


@dataclass(slots=True)
class Document:
    """One open document, as the *server* believes it to be."""

    uri: str
    language_id: str
    version: int
    text: str


@dataclass(slots=True)
class _Pending:
    event: threading.Event
    reply: dict[str, Any] | None = None
    failure: Exception | None = None


def _declared(value: Any) -> bool:
    """Whether a `*Provider` field means yes.

    `{}` is a legitimate "supported, no options", so plain truthiness would
    reject exactly the servers that answer most tersely.
    """
    return value is True or isinstance(value, dict)


def _kill_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, falling back to the child.

    A language server routinely forks helpers — `jdtls` is a shell script
    around a JVM, `typescript-language-server` runs `tsserver` — and killing
    only the pid we hold leaves those behind holding the workspace open.
    """
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


class LSPClient:
    """One language server process.  Thread-safe: requests may overlap."""

    __slots__ = (
        "_closed",
        "_diag",
        "_diagnostics",
        "_doc_lock",
        "_docs",
        "_framer",
        "_ids",
        "_lock",
        "_orphans",
        "_pending",
        "_proc",
        "_readers",
        "_stderr",
        "_write",
        "args",
        "capabilities",
        "command",
        "dead_reason",
        "diagnostic_wait",
        "env",
        "grace",
        "handshake_timeout",
        "logs",
        "name",
        "on_diagnostics",
        "root",
        "server_info",
        "settings",
        "timeout",
    )

    def __init__(
        self,
        command: str,
        args: list[str] | tuple[str, ...] | None = None,
        *,
        root: Path | str,
        name: str = "",
        env: dict[str, str] | None = None,
        settings: dict[str, Any] | None = None,
        timeout: float = 20.0,
        handshake_timeout: float | None = None,
        diagnostic_wait: float = DIAGNOSTIC_WAIT,
        grace: float = GRACE,
        on_diagnostics: Callable[[str, list[Diagnostic]], None] | None = None,
    ) -> None:
        self.command = command
        self.args = list(args or ())
        self.root = Path(root)
        self.name = name or Path(command).name
        self.env = dict(env or {})
        self.settings = dict(settings or {})
        self.timeout = timeout
        self.handshake_timeout = handshake_timeout or max(timeout, 30.0)
        self.diagnostic_wait = diagnostic_wait
        self.grace = grace
        self.on_diagnostics = on_diagnostics
        self.capabilities: dict[str, Any] = {}
        self.server_info: dict[str, Any] = {}
        self.dead_reason: str = ""
        #: `window/logMessage` and `window/showMessage`, for `diagnose()`.
        self.logs: deque[str] = deque(maxlen=40)
        self._proc: subprocess.Popen | None = None
        self._framer = Framer(max_frame=MAX_FRAME)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._readers: list[threading.Thread] = []
        self._stderr: deque[str] = deque(maxlen=40)
        self._write = threading.Lock()
        self._closed = True
        self._orphans = 0
        self._docs: dict[str, Document] = {}
        self._doc_lock = threading.Lock()
        self._diag = threading.Condition()
        self._diagnostics: dict[str, list[Diagnostic]] = {}

    # -- lifecycle ----------------------------------------------------------

    @property
    def label(self) -> str:
        version = str(self.server_info.get("version") or "")
        title = str(self.server_info.get("name") or self.name)
        return f"{title} {version}".strip()

    @property
    def alive(self) -> bool:
        proc = self._proc
        return not self._closed and proc is not None and proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    def diagnose(self) -> str:
        """Why this server is not answering, in one line, for a person."""
        if self._proc is None:
            return f"{self.command} was never started"
        code = self._proc.poll()
        state = "still running" if code is None else f"exited {code}"
        tail = " | ".join(line for line in list(self._stderr)[-3:] if line)
        return f"{self.command} {state}" + (f": {tail[:300]}" if tail else "")

    def start(self) -> None:
        """Launch the process and its reader threads.  Does not handshake."""
        if self._proc is not None:
            raise LSPError(f"{self.name} is already started")
        argv = [self.command, *self.args]
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # raw pipes: framing is ours, not the io module's
                cwd=str(self.root),
                env={**os.environ, **self.env},
                start_new_session=True,  # so close() can reap the whole tree
            )
        except (OSError, ValueError) as exc:
            raise ServerGone(f"could not start {self.command!r}: {exc}") from exc
        self._closed = False
        self.dead_reason = ""
        self._readers = [
            threading.Thread(target=self._pump_stdout, name=f"lsp-out-{self._proc.pid}", daemon=True),
            threading.Thread(target=self._pump_stderr, name=f"lsp-err-{self._proc.pid}", daemon=True),
        ]
        for reader in self._readers:
            reader.start()

    def connect(self) -> dict[str, Any]:
        """Start, handshake, and return the server's own description of itself."""
        self.start()
        try:
            result = self.request(
                "initialize",
                {
                    "processId": os.getpid(),
                    "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
                    "locale": "en",
                    "rootUri": to_uri(self.root),
                    "rootPath": str(self.root),
                    "workspaceFolders": [{"uri": to_uri(self.root), "name": self.root.name}],
                    "capabilities": CAPABILITIES,
                    "initializationOptions": self.settings or None,
                    "trace": "off",
                },
                timeout=self.handshake_timeout,
            )
        except LSPError:
            self.close()  # never leave a half-initialised child behind
            raise
        if not isinstance(result, dict):
            self.close()
            raise LSPError(f"{self.name} answered initialize with {type(result).__name__}, not an object")
        caps = result.get("capabilities")
        self.capabilities = caps if isinstance(caps, dict) else {}
        info = result.get("serverInfo")
        self.server_info = info if isinstance(info, dict) else {}
        self.notify("initialized", {})
        if self.settings:
            # Several servers read their configuration only from this
            # notification and ignore `initializationOptions` entirely.
            self.notify("workspace/didChangeConfiguration", {"settings": self.settings})
        return self.server_info

    def shutdown(self, *, timeout: float = 3.0) -> None:
        """The polite sequence, then the process group.  Idempotent."""
        if self.alive:
            try:
                self.request("shutdown", timeout=timeout)
            except LSPError:
                pass  # a server too broken to say goodbye still gets killed
            self.notify("exit")
            proc = self._proc
            if proc is not None:
                try:
                    proc.wait(timeout=min(self.grace, timeout))
                except subprocess.TimeoutExpired:
                    pass
        self.close()

    def close(self) -> None:
        """Reap the process group and fail everything still waiting."""
        self._closed = True
        self._fail_all(ServerGone(self.dead_reason or "client closed"))
        with self._diag:
            self._diag.notify_all()  # a diagnostics wait must not outlive us
        proc = self._proc
        if proc is None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if proc.poll() is None:
            _kill_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=self.grace)
            except subprocess.TimeoutExpired:
                _kill_group(proc, signal.SIGKILL)
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
            if reader is not threading.current_thread():
                reader.join(timeout=self.grace)
        with self._doc_lock:
            self._docs.clear()

    # -- capabilities -------------------------------------------------------

    def supports(self, method: str) -> bool:
        entry = PROVIDERS.get(method)
        if entry is None:
            return True  # not a capability-gated method
        return _declared(self.capabilities.get(entry[0]))

    @property
    def prepare_rename_supported(self) -> bool:
        rename = self.capabilities.get("renameProvider")
        return isinstance(rename, dict) and bool(rename.get("prepareProvider"))

    @property
    def resolve_code_action_supported(self) -> bool:
        actions = self.capabilities.get("codeActionProvider")
        return isinstance(actions, dict) and bool(actions.get("resolveProvider"))

    def provides(self) -> list[str]:
        """The English names of everything this server declared, sorted."""
        return sorted(
            human for key, human in PROVIDERS.values() if _declared(self.capabilities.get(key))
        )

    def require(self, method: str) -> None:
        if self.supports(method):
            return
        key, human = PROVIDERS[method]
        have = ", ".join(self.provides()) or "nothing offset can use"
        raise Unsupported(
            f"{self.label} does not support {human}: it declared no {key}. it provides: {have}"
        )

    # -- JSON-RPC -----------------------------------------------------------

    def send(self, frame: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ServerGone(f"{self.name} is not running")
        if self._closed or proc.poll() is not None:
            raise ServerGone(self.diagnose())
        payload = encode(frame)
        try:
            with self._write:
                proc.stdin.write(payload)
                proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            self.dead_reason = self.diagnose()
            raise ServerGone(f"{self.dead_reason} ({exc})") from exc

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Fire-and-forget.  A dead server is not worth an exception here."""
        frame: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        try:
            self.send(frame)
        except ServerGone as exc:
            self.dead_reason = str(exc)

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> Any:
        """Call `method` and wait for its reply, with a hard deadline.

        The result is returned raw: LSP answers with a list, an object or
        `null` depending on the method, so narrowing it here would throw away
        exactly the shapes `protocol` exists to flatten.
        """
        if self._closed:
            raise ServerGone(self.dead_reason or f"{self.name} is not connected")
        ident = next(self._ids)
        slot = _Pending(threading.Event())
        with self._lock:
            self._pending[ident] = slot
        frame: dict[str, Any] = {"jsonrpc": "2.0", "id": ident, "method": method}
        if params is not None:
            frame["params"] = params
        try:
            self.send(frame)
        except ServerGone:
            self._forget(ident)
            raise

        budget = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, budget)
        while not slot.event.wait(_TICK):
            if stop is not None and stop():
                self._forget(ident)
                self.notify("$/cancelRequest", {"id": ident})
                raise LSPCancelled(f"{method} cancelled")
            if time.monotonic() >= deadline:
                self._forget(ident)
                self.notify("$/cancelRequest", {"id": ident})
                raise LSPTimeout(f"{method} did not answer within {budget:g}s ({self.diagnose()})")
        self._forget(ident)
        if slot.failure is not None:
            raise slot.failure
        reply = slot.reply or {}
        error = reply.get("error")
        if isinstance(error, dict):
            raise LSPError(
                str(error.get("message") or f"{method} was refused without a reason"),
                code=error.get("code") if isinstance(error.get("code"), int) else None,
                data=error.get("data"),
            )
        return reply.get("result")

    def _forget(self, ident: int) -> None:
        with self._lock:
            self._pending.pop(ident, None)

    def _fail_all(self, failure: Exception) -> None:
        with self._lock:
            waiting, self._pending = self._pending, {}
        for slot in waiting.values():
            slot.failure = failure
            slot.event.set()

    # -- reader -------------------------------------------------------------

    def _pump_stdout(self) -> None:
        stream = self._proc.stdout if self._proc is not None else None
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(_CHUNK)
                if not chunk:
                    break
                for message in self._framer.feed(chunk):
                    self._handle(message)
        except (OSError, ValueError):
            pass  # the pipe was closed under us, which close() already knows
        finally:
            self._die(self.dead_reason or self.diagnose())

    def _pump_stderr(self) -> None:
        stream = self._proc.stderr if self._proc is not None else None
        if stream is None:
            return
        try:
            for line in io.TextIOWrapper(stream, encoding="utf-8", errors="replace"):
                self._stderr.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def _die(self, reason: str) -> None:
        if self._closed and not self._pending:
            return
        self._closed = True
        self.dead_reason = reason or "server exited"
        self._fail_all(ServerGone(self.dead_reason))
        with self._diag:
            self._diag.notify_all()

    def _handle(self, message: dict[str, Any]) -> None:
        ident = message.get("id")
        if ident is not None and ("result" in message or "error" in message):
            key = ident if isinstance(ident, int) else _as_int(ident)
            with self._lock:
                slot = self._pending.pop(key, None) if key is not None else None
            if slot is None:
                self._orphans += 1  # a reply to a call we already gave up on
                return
            slot.reply = message
            slot.event.set()
            return
        method = message.get("method")
        if not isinstance(method, str):
            return
        params = message.get("params")
        if ident is not None:
            self._answer(ident, method, params)
            return
        if method == "textDocument/publishDiagnostics":
            self._publish(params)
        elif method in ("window/logMessage", "window/showMessage"):
            if isinstance(params, dict):
                self.logs.append(str(params.get("message") or "").strip())

    def _answer(self, ident: Any, method: str, params: Any) -> None:
        """Reply to a server-initiated request; silence would hang the server.

        The benign answers matter more than they look: `pyright` will not emit
        a single diagnostic until its `workspace/configuration` request is
        answered, and `gopls` blocks its own startup on the progress token it
        asked us to create.
        """
        result: Any
        if method == "workspace/configuration":
            items = params.get("items") if isinstance(params, dict) else None
            count = len(items) if isinstance(items, list) else 1
            # One `null` per item means "no override, use your defaults",
            # which is a real answer rather than an invented configuration.
            result = [None] * max(1, count)
        elif method == "workspace/workspaceFolders":
            result = [{"uri": to_uri(self.root), "name": self.root.name}]
        elif method in (
            "client/registerCapability",
            "client/unregisterCapability",
            "window/workDoneProgress/create",
        ):
            result = None
        elif method == "workspace/semanticTokens/refresh":
            result = None
        else:
            self._refuse(ident, method)
            return
        try:
            self.send({"jsonrpc": "2.0", "id": ident, "result": result})
        except ServerGone:
            pass

    def _refuse(self, ident: Any, method: str) -> None:
        body = {
            "jsonrpc": "2.0",
            "id": ident,
            "error": {"code": -32601, "message": f"offset does not implement {method}"},
        }
        try:
            self.send(body)
        except ServerGone:
            pass

    # -- documents ----------------------------------------------------------

    @property
    def documents(self) -> dict[str, Document]:
        with self._doc_lock:
            return dict(self._docs)

    def version_of(self, path: Path | str) -> int:
        doc = self._docs.get(to_uri(path))
        return doc.version if doc is not None else 0

    def sync(
        self,
        path: Path | str,
        *,
        text: str | None = None,
        language_id: str = "",
    ) -> tuple[str, str]:
        """Make the server's copy of `path` match `text`, or the file on disk.

        Returns `(uri, text)`.  Idempotent: syncing unchanged text sends
        nothing, because a `didChange` carrying identical content still costs a
        full re-analysis on every server worth using.
        """
        target = Path(path)
        uri = to_uri(target)
        if text is None:
            try:
                text = target.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise LSPError(f"cannot read {target}: {exc}") from exc
        # The whole critical section is held across the send: the version and
        # the notification carrying it must reach the pipe in one piece, or a
        # concurrent change to another file can interleave a lower version.
        with self._doc_lock:
            doc = self._docs.get(uri)
            if doc is None:
                doc = Document(uri, language_id or _language_id(target), 1, text)
                self._docs[uri] = doc
                self.notify(
                    "textDocument/didOpen",
                    {
                        "textDocument": {
                            "uri": uri,
                            "languageId": doc.language_id,
                            "version": doc.version,
                            "text": text,
                        }
                    },
                )
                return uri, text
            if doc.text == text:
                return uri, text
            doc.version += 1
            doc.text = text
            self.notify(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": doc.version},
                    # Full replacement.  `textDocumentSync: 1` is declared by
                    # nothing here, but every server accepts a whole-document
                    # change and incremental ranges buy nothing at this size.
                    "contentChanges": [{"text": text}],
                },
            )
        return uri, text

    def close_document(self, path: Path | str) -> bool:
        uri = to_uri(path)
        with self._doc_lock:
            if self._docs.pop(uri, None) is None:
                return False
            self.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
        return True

    # -- diagnostics --------------------------------------------------------

    def _publish(self, params: Any) -> None:
        if not isinstance(params, dict):
            return
        uri = params.get("uri")
        if not isinstance(uri, str) or not uri:
            return
        raw = params.get("diagnostics")
        items = parse_diagnostics(raw) if isinstance(raw, list) else []
        key = normalise_uri(uri)
        with self._diag:
            # Replace, never accumulate: a publication is the server's whole
            # current opinion about that file, including "nothing wrong now".
            self._diagnostics[key] = items
            self._diag.notify_all()
        hook = self.on_diagnostics
        if hook is not None:
            try:
                hook(key, items)
            except Exception:  # a bad hook must not kill the reader
                pass

    def published(self, path: Path | str) -> bool:
        """Whether anything has ever been published for this file."""
        with self._diag:
            return to_uri(path) in self._diagnostics

    def diagnostics(
        self,
        path: Path | str,
        *,
        wait: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> list[Diagnostic]:
        """The newest diagnostics for `path`, waiting only if none ever came.

        See the module docstring: buffered means immediate, and unheard-of
        means a bounded wait then empty.  Never blocks indefinitely.
        """
        key = to_uri(path)
        budget = self.diagnostic_wait if wait is None else wait
        deadline = time.monotonic() + max(0.0, budget)
        with self._diag:
            while key not in self._diagnostics:
                left = deadline - time.monotonic()
                if left <= 0 or self._closed or (stop is not None and stop()):
                    return []
                self._diag.wait(min(_TICK, left))
            return list(self._diagnostics[key])

    def all_diagnostics(self) -> dict[str, list[Diagnostic]]:
        with self._diag:
            return {uri: list(items) for uri, items in self._diagnostics.items()}

    # -- requests -----------------------------------------------------------

    def _at(
        self,
        method: str,
        path: Path | str,
        position: Position,
        *,
        extra: dict[str, Any] | None = None,
        text: str | None = None,
        language_id: str = "",
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> tuple[str, Any]:
        self.require(method)
        uri, _ = self.sync(path, text=text, language_id=language_id)
        params: dict[str, Any] = {"textDocument": {"uri": uri}, "position": position.wire()}
        if extra:
            params.update(extra)
        return uri, self.request(method, params, timeout=timeout, stop=stop)

    def definition(self, path: Path | str, position: Position, **kw: Any) -> list[Location]:
        return parse_locations(self._at("textDocument/definition", path, position, **kw)[1])

    def type_definition(self, path: Path | str, position: Position, **kw: Any) -> list[Location]:
        return parse_locations(self._at("textDocument/typeDefinition", path, position, **kw)[1])

    def implementation(self, path: Path | str, position: Position, **kw: Any) -> list[Location]:
        return parse_locations(self._at("textDocument/implementation", path, position, **kw)[1])

    def references(
        self,
        path: Path | str,
        position: Position,
        *,
        include_declaration: bool = True,
        **kw: Any,
    ) -> list[Location]:
        extra = {"context": {"includeDeclaration": include_declaration}}
        return parse_locations(self._at("textDocument/references", path, position, extra=extra, **kw)[1])

    def hover(self, path: Path | str, position: Position, **kw: Any) -> str:
        return _hover_text(self._at("textDocument/hover", path, position, **kw)[1])

    def document_symbols(
        self,
        path: Path | str,
        *,
        text: str | None = None,
        language_id: str = "",
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> list[Symbol]:
        self.require("textDocument/documentSymbol")
        uri, _ = self.sync(path, text=text, language_id=language_id)
        result = self.request(
            "textDocument/documentSymbol",
            {"textDocument": {"uri": uri}},
            timeout=timeout,
            stop=stop,
        )
        return parse_symbols(result, uri=uri)

    def workspace_symbols(
        self,
        query: str,
        *,
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> list[Symbol]:
        self.require("workspace/symbol")
        result = self.request("workspace/symbol", {"query": query}, timeout=timeout, stop=stop)
        return parse_symbols(result)

    def prepare_rename(self, path: Path | str, position: Position, **kw: Any) -> Range | None:
        """The span the server would rename, or `None` if it refuses.

        Only sent when `prepareProvider` is declared; a server without it is
        answering the rename itself, so asking first would be a wasted round
        trip and, on a few servers, a `-32601`.
        """
        if not self.prepare_rename_supported:
            return None
        result = self._at("textDocument/prepareRename", path, position, **kw)[1]
        if result is None:
            return None
        if isinstance(result, dict) and "range" in result:
            return Range.parse(result["range"])
        if isinstance(result, dict) and "start" in result:
            return Range.parse(result)
        return None

    def rename(self, path: Path | str, position: Position, new_name: str, **kw: Any) -> WorkspaceEdit:
        extra = {"newName": new_name}
        return WorkspaceEdit.parse(self._at("textDocument/rename", path, position, extra=extra, **kw)[1])

    def code_actions(
        self,
        path: Path | str,
        span: Range,
        *,
        only: list[str] | None = None,
        text: str | None = None,
        language_id: str = "",
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> list[CodeAction]:
        self.require("textDocument/codeAction")
        uri, _ = self.sync(path, text=text, language_id=language_id)
        context: dict[str, Any] = {
            # The diagnostics the server already published for this span are
            # what turn a bare refactor list into the quick fixes the caller
            # actually wanted; an empty context hides every one of them.
            "diagnostics": [d.raw for d in self._overlapping(uri, span)],
            "triggerKind": 2,  # automatic: not a user keystroke
        }
        if only:
            context["only"] = only
        result = self.request(
            "textDocument/codeAction",
            {"textDocument": {"uri": uri}, "range": span.wire(), "context": context},
            timeout=timeout,
            stop=stop,
        )
        return parse_code_actions(result)

    def resolve_code_action(
        self,
        action: CodeAction,
        *,
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> CodeAction:
        """Fill in an action's `edit`, which many servers omit until asked."""
        if action.edit is not None or not self.resolve_code_action_supported:
            return action
        raw: dict[str, Any] = {"title": action.title}
        if action.kind:
            raw["kind"] = action.kind
        if action.data is not None:
            raw["data"] = action.data
        result = self.request("codeAction/resolve", raw, timeout=timeout, stop=stop)
        resolved = CodeAction.parse(result)
        return resolved if resolved is not None else action

    def formatting(
        self,
        path: Path | str,
        *,
        tab_size: int = 4,
        insert_spaces: bool = True,
        text: str | None = None,
        language_id: str = "",
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> list[TextEdit]:
        self.require("textDocument/formatting")
        uri, _ = self.sync(path, text=text, language_id=language_id)
        result = self.request(
            "textDocument/formatting",
            {
                "textDocument": {"uri": uri},
                "options": {
                    "tabSize": tab_size,
                    "insertSpaces": insert_spaces,
                    "trimTrailingWhitespace": True,
                    "insertFinalNewline": True,
                },
            },
            timeout=timeout,
            stop=stop,
        )
        return parse_text_edits(result)

    def _overlapping(self, uri: str, span: Range) -> list[Diagnostic]:
        with self._diag:
            found = list(self._diagnostics.get(uri, ()))
        return [
            item
            for item in found
            if item.range.start.line <= span.end.line and item.range.end.line >= span.start.line
        ]


# -- helpers ----------------------------------------------------------------

#: Extension -> LSP `languageId` for the cases where the two differ.  Servers
#: match on this string, and one wrong value means every request is answered
#: about a document the server never parsed.
_LANGUAGE_IDS: Final[dict[str, str]] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".java": "java",
    ".lua": "lua",
    ".sh": "shellscript",
    ".json": "json",
    ".md": "markdown",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
}


def _language_id(path: Path) -> str:
    return _LANGUAGE_IDS.get(path.suffix.lower(), path.suffix.lstrip(".").lower() or "plaintext")


def _hover_text(result: Any) -> str:
    """Flatten every `Hover.contents` shape three protocol versions allow."""
    if not isinstance(result, dict):
        return ""
    parts: list[str] = []

    def one(item: Any) -> None:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            value = item.get("value")
            if value is not None:
                parts.append(str(value))

    contents = result.get("contents")
    if isinstance(contents, list):
        for item in contents:
            one(item)
    else:
        one(contents)
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _as_int(value: Any) -> int | None:
    """Servers sometimes echo a numeric id back as a string."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
