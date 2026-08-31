"""RFC 6455 by hand, because the standard library has no WebSocket client.

The Chrome DevTools Protocol is only reachable over a WebSocket, and offset
takes exactly one runtime dependency.  So the framing lives here.  This is not
a general-purpose library and does not pretend to be one: no extensions, no
permessage-deflate, no autobahn compliance.  It is the subset a CDP client
needs, implemented properly, because the half-implemented version of this is
the classic source of silent corruption.

Three rules the protocol punishes you for getting wrong, so they are stated
here and enforced below:

  * **Every client frame is masked, every server frame is not.**  A browser
    reads an unmasked client frame as a protocol violation and closes the
    connection with no useful diagnostic at all.  Masking uses a fresh
    four-byte key per frame; reusing one is the same bug wearing a hat.
  * **A frame boundary is not a message boundary.**  Text is UTF-8 and a
    multi-byte character may be cut in half by fragmentation, so decoding
    happens once, on the reassembled payload, never per frame.
  * **Control frames arrive mid-message.**  A ping between two continuation
    frames must be answered and must not disturb the message being assembled.

The threading shape mirrors `offset.tools.mcp.transport`: a blocking socket, a
daemon reader thread that owns all reads and pushes finished messages onto a
queue, a sentinel so a waiting caller learns the peer is gone rather than
hanging, and a `close()` that is safe to call twice.
"""

from __future__ import annotations

import base64
import hashlib
import os
import queue
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlsplit

#: The magic string from RFC 6455 section 1.3.  The server appends it to the
#: client key and SHA-1s the result; that is the whole proof that we are
#: talking to a WebSocket endpoint and not to a cache replaying a 101.
GUID: Final = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT: Final = 0x0
OP_TEXT: Final = 0x1
OP_BINARY: Final = 0x2
OP_CLOSE: Final = 0x8
OP_PING: Final = 0x9
OP_PONG: Final = 0xA

#: Longest single frame accepted.  Generous because a full-page screenshot
#: arrives as one base64 string, but bounded: a confused peer must not be able
#: to grow this process without limit.
MAX_FRAME: Final = 64 << 20

#: Longest reassembled message.  Separate from `MAX_FRAME` because
#: fragmentation would otherwise be an unbounded-memory attack in disguise.
MAX_MESSAGE: Final = 64 << 20

#: A control frame carries at most 125 bytes and is never fragmented.
MAX_CONTROL: Final = 125

#: Socket read slice.  Short enough that `close()` is responsive, long enough
#: that an idle connection costs nothing measurable.
_TICK: Final = 0.2

#: Enqueued by the reader when the peer will never speak again.
_GONE: Final = object()

CLOSE_NORMAL: Final = 1000
CLOSE_PROTOCOL: Final = 1002
CLOSE_BAD_DATA: Final = 1007
CLOSE_TOO_BIG: Final = 1009


class WebSocketError(Exception):
    """Anything that makes this connection unusable."""


class HandshakeError(WebSocketError):
    """The upgrade did not happen, so there is no WebSocket to speak on."""


class ProtocolError(WebSocketError):
    """The peer sent bytes RFC 6455 does not allow."""


class WebSocketClosed(WebSocketError):
    """The peer is gone.  Anything still waiting must give up."""


# -- handshake ---------------------------------------------------------------


def accept_token(key: str) -> str:
    """The `Sec-WebSocket-Accept` a conforming server must return for `key`."""
    digest = hashlib.sha1((key.strip() + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def new_key() -> str:
    """A fresh client key: sixteen random bytes, base64'd, per section 4.1."""
    return base64.b64encode(os.urandom(16)).decode("ascii")


# -- framing -----------------------------------------------------------------


def mask_bytes(data: bytes, key: bytes) -> bytes:
    """XOR `data` with the repeating four-byte `key`.  Its own inverse.

    Done as one big-integer XOR rather than a Python-level loop: masking sits
    on the path of every screenshot, and a per-byte generator expression over
    a megabyte of base64 is measurably slower than the C-level bignum op.
    Padding goes on the *end* so the key stays aligned to the payload start,
    and the pad is sliced off afterwards.
    """
    if not data:
        return b""
    if len(key) != 4:
        raise ProtocolError(f"a masking key is four bytes, got {len(key)}")
    pad = -len(data) % 4
    width = len(data) + pad
    stream = key * (width // 4)
    mixed = (int.from_bytes(data + bytes(pad), "big") ^ int.from_bytes(stream, "big")).to_bytes(width, "big")
    return mixed[: len(data)] if pad else mixed


@dataclass(frozen=True, slots=True)
class Frame:
    """One wire frame, already unmasked."""

    opcode: int
    payload: bytes
    fin: bool = True
    #: Whether the MASK bit was set on the wire.  Kept because the direction
    #: rule is a protocol decision for the caller, not for the codec.
    masked: bool = False

    @property
    def control(self) -> bool:
        return self.opcode >= OP_CLOSE


def encode_frame(
    opcode: int,
    payload: bytes = b"",
    *,
    fin: bool = True,
    mask: bool = True,
    key: bytes | None = None,
) -> bytes:
    """Serialise one frame.  `mask=True` is the client direction."""
    if opcode >= OP_CLOSE and (len(payload) > MAX_CONTROL or not fin):
        raise ProtocolError("a control frame must be unfragmented and at most 125 bytes")
    if len(payload) > MAX_FRAME:
        raise ProtocolError(f"frame of {len(payload)} bytes exceeds the {MAX_FRAME} byte limit")
    head = bytearray()
    head.append((0x80 if fin else 0x00) | (opcode & 0x0F))
    flag = 0x80 if mask else 0x00
    size = len(payload)
    if size < 126:
        head.append(flag | size)
    elif size <= 0xFFFF:
        head.append(flag | 126)
        head += size.to_bytes(2, "big")
    else:
        head.append(flag | 127)
        head += size.to_bytes(8, "big")
    if not mask:
        return bytes(head) + payload
    chosen = key if key is not None else os.urandom(4)
    if len(chosen) != 4:
        raise ProtocolError(f"a masking key is four bytes, got {len(chosen)}")
    return bytes(head) + chosen + mask_bytes(payload, chosen)


def decode_frame(data: bytes | bytearray | memoryview) -> tuple[Frame, int] | None:
    """The first frame in `data` and how many bytes it used.

    `None` means the buffer is short and the caller should read more; that is
    the normal case on a stream socket, not a failure.
    """
    view = memoryview(data)
    if len(view) < 2:
        return None
    first, second = view[0], view[1]
    if first & 0x70:
        raise ProtocolError("reserved bits set but no extension was negotiated")
    fin = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    size = second & 0x7F
    at = 2
    if size == 126:
        if len(view) < at + 2:
            return None
        size = int.from_bytes(view[at : at + 2], "big")
        at += 2
    elif size == 127:
        if len(view) < at + 8:
            return None
        size = int.from_bytes(view[at : at + 8], "big")
        at += 8
        if size >> 63:
            raise ProtocolError("64-bit length with the high bit set")
    if size > MAX_FRAME:
        raise ProtocolError(f"frame of {size} bytes exceeds the {MAX_FRAME} byte limit")
    key = b""
    if masked:
        if len(view) < at + 4:
            return None
        key = bytes(view[at : at + 4])
        at += 4
    if len(view) < at + size:
        return None
    payload = bytes(view[at : at + size])
    at += size
    if opcode >= OP_CLOSE and (size > MAX_CONTROL or not fin):
        raise ProtocolError("a control frame must be unfragmented and at most 125 bytes")
    return Frame(opcode, mask_bytes(payload, key) if masked else payload, fin, masked), at


def close_payload(code: int, reason: str = "") -> bytes:
    """A close body: two-byte big-endian status then a UTF-8 reason."""
    body = code.to_bytes(2, "big") + reason.encode("utf-8")
    return body[:MAX_CONTROL]


def parse_close(payload: bytes) -> tuple[int, str]:
    """`(code, reason)` from a close body.  An empty body means 1005, no status."""
    if len(payload) < 2:
        return 1005, ""
    return int.from_bytes(payload[:2], "big"), payload[2:].decode("utf-8", "replace")


# -- messages ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Message:
    """A reassembled application message."""

    opcode: int
    data: bytes
    #: Set only for text messages, decoded once from the whole payload so a
    #: character split across continuation frames survives.
    text: str | None = None

    @property
    def is_text(self) -> bool:
        return self.opcode == OP_TEXT


# -- the connection ----------------------------------------------------------


class WebSocket:
    """A client connection: blocking writes, one reader thread, a queue out."""

    __slots__ = (
        "_buffer",
        "_closed",
        "_gone",
        "_lock",
        "_peer_closed",
        "_queue",
        "_reader",
        "_sent_close",
        "_sock",
        "close_code",
        "close_reason",
        "failure",
        "max_message",
        "pings_answered",
        "pongs_seen",
        "url",
    )

    def __init__(
        self,
        sock: socket.socket,
        *,
        url: str = "",
        leftover: bytes = b"",
        max_message: int = MAX_MESSAGE,
    ) -> None:
        self._sock = sock
        self.url = url
        self.max_message = max_message
        self._buffer = bytearray(leftover)
        self._queue: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()
        self._peer_closed = threading.Event()
        self._closed = False
        self._gone = False
        self._sent_close = False
        self.close_code: int | None = None
        self.close_reason = ""
        self.failure: str | None = None
        self.pings_answered = 0
        self.pongs_seen = 0
        sock.settimeout(_TICK)
        self._reader = threading.Thread(target=self._pump, name="ws-reader", daemon=True)
        self._reader.start()

    # -- construction -------------------------------------------------------

    @classmethod
    def connect(
        cls,
        url: str,
        *,
        timeout: float = 10.0,
        headers: dict[str, str] | None = None,
        max_message: int = MAX_MESSAGE,
    ) -> WebSocket:
        """Open a connection and complete the upgrade, or raise `HandshakeError`."""
        parts = urlsplit(url)
        secure = parts.scheme == "wss"
        if parts.scheme not in ("ws", "wss"):
            raise HandshakeError(f"not a websocket url: {url!r}")
        host = parts.hostname or "127.0.0.1"
        port = parts.port or (443 if secure else 80)
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        deadline = time.monotonic() + timeout
        try:
            sock = socket.create_connection((host, port), timeout=timeout)
        except OSError as exc:
            raise HandshakeError(f"could not reach {host}:{port}: {exc}") from exc
        try:
            if secure:
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
            key = new_key()
            request = [
                f"GET {target} HTTP/1.1",
                f"Host: {parts.netloc}",
                "Upgrade: websocket",
                "Connection: Upgrade",
                f"Sec-WebSocket-Key: {key}",
                "Sec-WebSocket-Version: 13",
            ]
            for name, value in (headers or {}).items():
                request.append(f"{name}: {value}")
            sock.settimeout(max(0.1, deadline - time.monotonic()))
            sock.sendall(("\r\n".join(request) + "\r\n\r\n").encode("ascii"))
            status, fields, leftover = _read_response(sock, deadline)
        except HandshakeError:
            sock.close()
            raise
        except OSError as exc:
            sock.close()
            raise HandshakeError(f"handshake with {host}:{port} failed: {exc}") from exc

        if status != 101:
            sock.close()
            raise HandshakeError(f"expected 101 switching protocols, got {status}")
        if fields.get("upgrade", "").lower() != "websocket":
            sock.close()
            raise HandshakeError(f"server did not upgrade: upgrade={fields.get('upgrade', '')!r}")
        expected = accept_token(key)
        offered = fields.get("sec-websocket-accept", "")
        if offered != expected:
            # Refusing here is the point of the handshake.  A wrong accept means
            # something other than our peer answered - a proxy, a cache, a
            # different protocol on the port - and framing it as WebSocket
            # would corrupt whatever it really is.
            sock.close()
            raise HandshakeError(f"bad sec-websocket-accept: expected {expected!r}, got {offered!r}")
        return cls(sock, url=url, leftover=leftover, max_message=max_message)

    # -- state --------------------------------------------------------------

    @property
    def alive(self) -> bool:
        return not self._closed and not self._gone

    def diagnostics(self) -> str:
        """One line a human or a model can act on."""
        if self.failure:
            return self.failure
        if self.close_code is not None:
            return f"peer closed with {self.close_code}" + (f": {self.close_reason}" if self.close_reason else "")
        if self._closed:
            return "connection closed locally"
        return f"connected to {self.url or 'peer'}"

    # -- sending ------------------------------------------------------------

    def send_text(self, text: str) -> None:
        self._write(OP_TEXT, text.encode("utf-8"))

    def send_bytes(self, data: bytes) -> None:
        self._write(OP_BINARY, data)

    def send(self, data: str | bytes) -> None:
        if isinstance(data, str):
            self.send_text(data)
        else:
            self.send_bytes(data)

    def ping(self, payload: bytes = b"") -> None:
        self._write(OP_PING, payload[:MAX_CONTROL])

    def pong(self, payload: bytes = b"") -> None:
        self._write(OP_PONG, payload[:MAX_CONTROL])

    def _write(self, opcode: int, payload: bytes) -> None:
        if self._closed or self._gone:
            raise WebSocketClosed(self.diagnostics())
        frame = encode_frame(opcode, payload, mask=True)
        try:
            with self._lock:
                self._sock.sendall(frame)
        except OSError as exc:
            self._fail(f"send failed: {type(exc).__name__}: {exc}")
            raise WebSocketClosed(self.diagnostics()) from exc

    # -- receiving ----------------------------------------------------------

    def recv(self, timeout: float = 10.0) -> Message | None:
        """The next message, or `None` if `timeout` elapsed with nothing to give.

        Raises `WebSocketClosed` once the peer is gone, and keeps raising it.
        """
        if self._gone:
            raise WebSocketClosed(self.diagnostics())
        try:
            item = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        if item is _GONE:
            self._gone = True
            raise WebSocketClosed(self.diagnostics())
        return item

    def drain(self) -> list[Message]:
        """Every message already queued, without blocking."""
        out: list[Message] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return out
            if item is _GONE:
                self._queue.put(_GONE)
                return out
            out.append(item)

    # -- teardown -----------------------------------------------------------

    def close(self, code: int = CLOSE_NORMAL, reason: str = "", *, grace: float = 1.0) -> None:
        """Close handshake then socket teardown.  Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if not self._gone:
            self._send_close(code, reason)
            # Politeness with a deadline: a peer that never answers must not
            # keep the caller here.
            self._peer_closed.wait(grace)
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._reader.join(timeout=grace + _TICK)
        self._queue.put(_GONE)

    def _send_close(self, code: int, reason: str) -> None:
        with self._lock:
            if self._sent_close:
                return
            self._sent_close = True
            try:
                self._sock.sendall(encode_frame(OP_CLOSE, close_payload(code, reason), mask=True))
            except OSError:
                pass  # the peer vanished first; nothing left to negotiate

    def _fail(self, why: str) -> None:
        if self.failure is None:
            self.failure = why

    # -- the reader ---------------------------------------------------------

    def _pump(self) -> None:
        """Own every read.  Assemble messages, answer control frames."""
        opcode = 0
        body = bytearray()
        assembling = False
        try:
            while not self._closed:
                parsed = self._next_frame()
                if parsed is None:
                    continue
                if parsed is _GONE:
                    break
                frame: Frame = parsed  # type: ignore[assignment]
                if frame.masked:
                    raise ProtocolError("a server frame must not be masked")
                if frame.control:
                    if self._handle_control(frame):
                        break
                    continue  # never disturbs the message under construction
                if frame.opcode == OP_CONT:
                    if not assembling:
                        raise ProtocolError("continuation frame with no message to continue")
                else:
                    if assembling:
                        raise ProtocolError(f"opcode {frame.opcode} arrived inside a fragmented message")
                    assembling, opcode, body = True, frame.opcode, bytearray()
                body += frame.payload
                if len(body) > self.max_message:
                    raise ProtocolError(f"message exceeded {self.max_message} bytes")
                if not frame.fin:
                    continue
                self._queue.put(_assemble(opcode, bytes(body)))
                assembling, body = False, bytearray()
        except ProtocolError as exc:
            self._fail(f"protocol error: {exc}")
            self._send_close(CLOSE_BAD_DATA if "utf-8" in str(exc) else CLOSE_PROTOCOL, str(exc)[:80])
        except (OSError, ValueError) as exc:
            self._fail(f"read failed: {type(exc).__name__}: {exc}")
        finally:
            self._queue.put(_GONE)

    def _next_frame(self) -> Frame | object | None:
        """One frame from the buffer, reading the socket when it is short."""
        while True:
            parsed = decode_frame(self._buffer)
            if parsed is not None:
                frame, used = parsed
                del self._buffer[:used]
                return frame
            try:
                chunk = self._sock.recv(65536)
            except TimeoutError:
                return None  # gives the loop a chance to notice `close()`
            except OSError as exc:
                if self._closed:
                    return _GONE
                raise ValueError(f"{type(exc).__name__}: {exc}") from exc
            if not chunk:
                if self._buffer:
                    self._fail("connection ended mid-frame")
                return _GONE
            self._buffer += chunk

    def _handle_control(self, frame: Frame) -> bool:
        """Answer a control frame.  True means the connection is finished."""
        if frame.opcode == OP_PING:
            # The pong must carry the ping's payload byte for byte; peers use
            # it as a round-trip token and drop mismatched replies.
            try:
                self._write(OP_PONG, frame.payload)
                self.pings_answered += 1
            except WebSocketClosed:
                return True
            return False
        if frame.opcode == OP_PONG:
            self.pongs_seen += 1
            return False
        if frame.opcode == OP_CLOSE:
            self.close_code, self.close_reason = parse_close(frame.payload)
            self._send_close(CLOSE_NORMAL, "")
            self._peer_closed.set()
            return True
        raise ProtocolError(f"unknown control opcode {frame.opcode}")


def _assemble(opcode: int, body: bytes) -> Message:
    """Turn a completed payload into a message, decoding text exactly once."""
    if opcode != OP_TEXT:
        return Message(opcode, body)
    try:
        return Message(opcode, body, body.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ProtocolError(f"text message is not valid utf-8: {exc}") from exc


def _read_response(sock: socket.socket, deadline: float) -> tuple[int, dict[str, str], bytes]:
    """Status, lowercased headers, and any bytes that followed the blank line.

    Keeping the leftover matters: a fast server can put its first frame in the
    same TCP segment as the 101, and throwing those bytes away loses a message.
    """
    buffer = bytearray()
    while b"\r\n\r\n" not in buffer:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HandshakeError("timed out waiting for the upgrade response")
        sock.settimeout(remaining)
        chunk = sock.recv(4096)
        if not chunk:
            raise HandshakeError("connection closed during the handshake")
        buffer += chunk
        if len(buffer) > 64 * 1024:
            raise HandshakeError("upgrade response headers are implausibly large")
    head, _, rest = bytes(buffer).partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ", 2)
    if len(parts) < 2 or not parts[1].isdigit():
        raise HandshakeError(f"malformed status line: {lines[0]!r}")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep:
            fields[name.strip().lower()] = value.strip()
    return int(parts[1]), fields, rest
