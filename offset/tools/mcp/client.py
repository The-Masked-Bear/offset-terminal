"""JSON-RPC 2.0 and the MCP protocol: handshake, correlation, calls.

The invariants this module defends, all three learned from clients that hang:

  * every request gets a fresh id from one monotonic allocator, because a
    repeated id silently pairs a reply with the wrong call;
  * every request has a deadline enforced on the *caller's* side, so a server
    that answers nothing costs one timeout rather than the session;
  * when the transport dies, everything still waiting is failed immediately
    with `ServerGone` instead of waiting out its deadline.

Errors the model or user has to react to are values (`CallOutcome.ok`); errors
that mean "this server cannot serve you" are exceptions, because there is no
useful result to hand back.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterator

from offset.tools.mcp.transport import Transport, TransportClosed

#: What offset asks for.  Servers commonly answer with an older one, which is
#: fine as long as it is one we can actually speak.
PROTOCOL_VERSION: Final = "2025-06-18"
ACCEPTED: Final = ("2025-06-18", "2025-03-26", "2024-11-05")

CLIENT_NAME: Final = "offset"
CLIENT_VERSION: Final = "0.1.0"

#: Wait slice while a request is outstanding: the granularity of both the
#: deadline and cooperative cancellation.
_TICK: Final = 0.05

#: Refuse to follow a paginated list forever; a server with a broken cursor
#: would otherwise loop until the deadline.
MAX_PAGES: Final = 50


class MCPError(Exception):
    """A protocol-level failure, including a JSON-RPC error response."""

    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class MCPTimeout(MCPError):
    """The server did not answer in time.  Never a user cancellation."""


class ServerGone(MCPError):
    """The transport died, so no reply can ever arrive."""


class MCPCancelled(MCPError):
    """The caller asked to stop; the server was told with notifications/cancelled."""


@dataclass(slots=True)
class RemoteTool:
    name: str
    description: str
    schema: dict[str, Any]
    #: `annotations.readOnlyHint`.  Absent means "assume it can hurt".
    read_only: bool = False
    title: str = ""

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "RemoteTool":
        annotations = raw.get("annotations")
        annotations = annotations if isinstance(annotations, dict) else {}
        schema = raw.get("inputSchema")
        return cls(
            name=str(raw.get("name") or ""),
            description=str(raw.get("description") or ""),
            schema=schema if isinstance(schema, dict) else {"type": "object", "properties": {}},
            read_only=annotations.get("readOnlyHint") is True,
            title=str(raw.get("title") or annotations.get("title") or ""),
        )


@dataclass(slots=True)
class Resource:
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Resource":
        return cls(
            uri=str(raw.get("uri") or ""),
            name=str(raw.get("name") or ""),
            description=str(raw.get("description") or ""),
            mime_type=str(raw.get("mimeType") or ""),
        )


@dataclass(slots=True)
class Prompt:
    name: str
    description: str = ""
    arguments: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> "Prompt":
        args = raw.get("arguments")
        return cls(
            name=str(raw.get("name") or ""),
            description=str(raw.get("description") or ""),
            arguments=[a for a in args if isinstance(a, dict)] if isinstance(args, list) else [],
        )


@dataclass(slots=True)
class CallOutcome:
    """A tools/call result.  `ok=False` is the server's own `isError`, which is
    a value the model must react to rather than a transport problem."""

    ok: bool
    content: str
    structured: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, result: dict[str, Any]) -> "CallOutcome":
        blocks = result.get("content")
        rendered = _render(blocks if isinstance(blocks, list) else [])
        structured = result.get("structuredContent")
        return cls(
            ok=result.get("isError") is not True,
            content=rendered,
            structured=structured if isinstance(structured, dict) else {},
        )


def _render(blocks: list[Any]) -> str:
    """Flatten MCP content blocks to text a model can read."""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} {block.get('mimeType') or 'unknown'}, not shown]")
        elif kind == "resource":
            inner = block.get("resource")
            inner = inner if isinstance(inner, dict) else {}
            body = inner.get("text")
            parts.append(str(body) if body is not None else f"[resource {inner.get('uri') or '?'}]")
        elif kind == "resource_link":
            parts.append(f"[resource {block.get('uri') or '?'}]")
    return "\n".join(p for p in parts if p)


@dataclass(slots=True)
class _Pending:
    event: threading.Event
    reply: dict[str, Any] | None = None
    failure: Exception | None = None


class MCPClient:
    """One connected server.  Thread-safe: tools may be called concurrently."""

    __slots__ = (
        "transport", "name", "version", "timeout", "handshake_timeout", "on_notification",
        "server_info", "capabilities", "protocol", "dead_reason", "notifications",
        "_ids", "_lock", "_pending", "_reader", "_closed", "_orphans",
    )

    def __init__(
        self,
        transport: Transport,
        *,
        name: str = CLIENT_NAME,
        version: str = CLIENT_VERSION,
        timeout: float = 30.0,
        handshake_timeout: float | None = None,
        on_notification: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.transport = transport
        self.name = name
        self.version = version
        self.timeout = timeout
        self.handshake_timeout = handshake_timeout or min(timeout, 15.0)
        self.on_notification = on_notification
        self.server_info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.protocol: str = ""
        self.dead_reason: str = ""
        self.notifications: deque[dict[str, Any]] = deque(maxlen=64)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._reader: threading.Thread | None = None
        self._closed = True
        self._orphans = 0

    # -- lifecycle ----------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        """Start the transport, handshake, and return the server's info."""
        self.transport.start()
        self._closed = False
        self.dead_reason = ""
        self._reader = threading.Thread(target=self._pump, name=f"mcp-client-{self.name}", daemon=True)
        self._reader.start()
        try:
            result = self.request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    # Nothing is advertised that offset does not implement: a
                    # server that trusts a false capability would wait forever
                    # for a reply we would never send.
                    "capabilities": {},
                    "clientInfo": {"name": self.name, "version": self.version},
                },
                timeout=self.handshake_timeout,
            )
        except MCPError:
            self.close()
            raise
        spoken = str(result.get("protocolVersion") or "")
        if spoken not in ACCEPTED:
            self.close()
            raise MCPError(
                f"server speaks MCP protocol {spoken or 'nothing recognisable'}; "
                f"offset speaks {', '.join(ACCEPTED)}"
            )
        self.protocol = spoken
        caps = result.get("capabilities")
        self.capabilities = caps if isinstance(caps, dict) else {}
        info = result.get("serverInfo")
        self.server_info = info if isinstance(info, dict) else {}
        self.notify("notifications/initialized")
        return self.server_info

    @property
    def alive(self) -> bool:
        return not self._closed and self.transport.alive

    def supports(self, area: str) -> bool:
        """Whether the server declared a capability, e.g. "tools"."""
        return isinstance(self.capabilities.get(area), dict)

    def close(self) -> None:
        """Idempotent.  Everything still waiting is failed, never abandoned."""
        self._closed = True
        self._fail_all(ServerGone(self.dead_reason or "client closed"))
        try:
            self.transport.close()
        except Exception:  # closing must not raise into a shutdown path
            pass
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)

    @property
    def description(self) -> str:
        title = self.server_info.get("name") or self.name
        return f"{title} {self.server_info.get('version') or ''}".strip()

    # -- JSON-RPC -----------------------------------------------------------

    def _next_id(self) -> int:
        with self._lock:
            return next(self._ids)

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Call `method` and wait for its reply, with a hard deadline.

        `stop` is polled while waiting so a tool can be cancelled without
        waiting the budget out; the server is told with notifications/cancelled.
        """
        if self._closed:
            raise ServerGone(self.dead_reason or "not connected")
        ident = self._next_id()
        slot = _Pending(threading.Event())
        with self._lock:
            self._pending[ident] = slot
        frame: dict[str, Any] = {"jsonrpc": "2.0", "id": ident, "method": method}
        if params is not None:
            frame["params"] = params
        try:
            self.transport.send(frame)
        except TransportClosed as exc:
            self._forget(ident)
            self.dead_reason = str(exc)
            raise ServerGone(f"{method}: {exc}") from exc

        budget = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, budget)
        while not slot.event.wait(_TICK):
            if stop is not None and stop():
                self._forget(ident)
                self._cancel_remote(ident, "cancelled by the user")
                raise MCPCancelled(f"{method} cancelled")
            if time.monotonic() >= deadline:
                self._forget(ident)
                self._cancel_remote(ident, "timed out")
                raise MCPTimeout(f"{method} did not answer within {budget:g}s")
        self._forget(ident)
        if slot.failure is not None:
            raise slot.failure
        reply = slot.reply or {}
        error = reply.get("error")
        if isinstance(error, dict):
            raise MCPError(
                str(error.get("message") or "server reported an error"),
                code=error.get("code") if isinstance(error.get("code"), int) else None,
                data=error.get("data"),
            )
        result = reply.get("result")
        return result if isinstance(result, dict) else {}

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Fire-and-forget.  A dead server is not worth an exception here."""
        frame: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            frame["params"] = params
        try:
            self.transport.send(frame)
        except TransportClosed as exc:
            self.dead_reason = str(exc)

    def _cancel_remote(self, ident: int, reason: str) -> None:
        self.notify("notifications/cancelled", {"requestId": ident, "reason": reason})

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

    def _pump(self) -> None:
        while not self._closed:
            try:
                message = self.transport.receive(_TICK)
            except TransportClosed as exc:
                self.dead_reason = str(exc) or "server exited"
                self._closed = True
                self._fail_all(ServerGone(self.dead_reason))
                return
            except Exception as exc:  # a transport bug must not strand callers
                self.dead_reason = f"transport failed: {exc}"
                self._closed = True
                self._fail_all(ServerGone(self.dead_reason))
                return
            if message is not None:
                self._handle(message)

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
        if ident is not None:
            self._answer(ident, method)
            return
        self.notifications.append(message)
        if self.on_notification is not None:
            try:
                self.on_notification(message)
            except Exception:  # a bad hook must not kill the reader
                pass

    def _answer(self, ident: Any, method: str) -> None:
        """Reply to a server-initiated request; silence would hang the server."""
        if method == "ping":
            body: dict[str, Any] = {"jsonrpc": "2.0", "id": ident, "result": {}}
        else:
            body = {
                "jsonrpc": "2.0",
                "id": ident,
                "error": {"code": -32601, "message": f"offset does not implement {method}"},
            }
        try:
            self.transport.send(body)
        except TransportClosed:
            pass

    # -- protocol calls -----------------------------------------------------

    def _pages(
        self,
        method: str,
        key: str,
        *,
        timeout: float | None = None,
    ) -> Iterator[list[Any]]:
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = self.request(method, params, timeout=timeout)
            items = result.get(key)
            yield [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []
            nxt = result.get("nextCursor")
            if not isinstance(nxt, str) or not nxt or nxt == cursor:
                return
            cursor = nxt

    def list_tools(self, *, timeout: float | None = None) -> list[RemoteTool]:
        if not self.supports("tools"):
            return []
        out: list[RemoteTool] = []
        for page in self._pages("tools/list", "tools", timeout=timeout):
            out.extend(tool for tool in (RemoteTool.parse(raw) for raw in page) if tool.name)
        return out

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> CallOutcome:
        if not self.supports("tools"):
            raise MCPError(f"{self.name} declares no tools capability")
        result = self.request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
            timeout=timeout,
            stop=stop,
        )
        return CallOutcome.parse(result)

    def list_resources(self, *, timeout: float | None = None) -> list[Resource]:
        if not self.supports("resources"):
            return []
        out: list[Resource] = []
        for page in self._pages("resources/list", "resources", timeout=timeout):
            out.extend(res for res in (Resource.parse(raw) for raw in page) if res.uri)
        return out

    def read_resource(self, uri: str, *, timeout: float | None = None) -> str:
        if not self.supports("resources"):
            raise MCPError(f"{self.name} declares no resources capability")
        result = self.request("resources/read", {"uri": uri}, timeout=timeout)
        contents = result.get("contents")
        parts: list[str] = []
        for item in contents if isinstance(contents, list) else []:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if text is not None:
                parts.append(str(text))
            elif item.get("blob") is not None:
                parts.append(f"[binary {item.get('mimeType') or 'unknown'} at {item.get('uri') or uri}]")
        return "\n".join(parts)

    def list_prompts(self, *, timeout: float | None = None) -> list[Prompt]:
        if not self.supports("prompts"):
            return []
        out: list[Prompt] = []
        for page in self._pages("prompts/list", "prompts", timeout=timeout):
            out.extend(prompt for prompt in (Prompt.parse(raw) for raw in page) if prompt.name)
        return out


def _as_int(value: Any) -> int | None:
    """Servers sometimes echo a numeric id back as a string."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
