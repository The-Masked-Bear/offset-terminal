"""The hand-rolled WebSocket, tested against a real server on a real socket.

The server in this file deliberately does **not** import offset's codec.  It
does its own length parsing and its own per-byte unmasking, so a symmetrical
bug in `wsclient` cannot hide behind a symmetrical bug in the fixture.  Every
test here runs over loopback against `ThreadingHTTPServer`; none needs a
browser and none touches the network.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import queue
import socket
import struct
import threading

import pytest

from offset.tools.web.wsclient import (
    CLOSE_NORMAL,
    GUID,
    MAX_CONTROL,
    OP_BINARY,
    OP_CLOSE,
    OP_CONT,
    OP_PING,
    OP_PONG,
    OP_TEXT,
    HandshakeError,
    ProtocolError,
    WebSocket,
    WebSocketClosed,
    accept_token,
    decode_frame,
    encode_frame,
    mask_bytes,
    new_key,
    parse_close,
)

#: A string whose UTF-8 encoding is cut in half by `SPLIT_AT`, so the
#: continuation test cannot accidentally split on a character boundary.
SPLIT_TEXT = "spans: naïve → 日本語 ✓"
SPLIT_RAW = SPLIT_TEXT.encode("utf-8")
SPLIT_AT = next(i for i in range(1, len(SPLIT_RAW)) if SPLIT_RAW[i] & 0xC0 == 0x80)

PING_TOKEN = b"mid-message"


# -- an independent server ---------------------------------------------------


def _write_frame(out, opcode: int, payload: bytes, *, fin: bool = True, mask: bool = False) -> None:
    """Frame `payload` with hand-written length bytes.  No offset code involved."""
    first = (0x80 if fin else 0x00) | opcode
    size = len(payload)
    if size < 126:
        head = bytes((first, size | (0x80 if mask else 0)))
    elif size <= 0xFFFF:
        head = bytes((first, 126 | (0x80 if mask else 0))) + struct.pack("!H", size)
    else:
        head = bytes((first, 127 | (0x80 if mask else 0))) + struct.pack("!Q", size)
    if mask:
        key = b"\x11\x22\x33\x44"
        payload = bytes(byte ^ key[i % 4] for i, byte in enumerate(payload))
        head += key
    out.write(head + payload)
    out.flush()


class Received:
    """What the server saw, so a test can assert on the wire, not just the API."""

    def __init__(self) -> None:
        self.frames: queue.Queue = queue.Queue()
        self.masked: list[bool] = []
        self.lengths: list[int] = []


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: Received

    def log_message(self, *_args) -> None:  # keep pytest output readable
        pass

    # -- framing, written out longhand on purpose ---------------------------

    def _exact(self, count: int) -> bytes:
        data = self.rfile.read(count)
        if data is None or len(data) < count:
            raise EOFError("client hung up")
        return data

    def _read_frame(self) -> tuple[bool, int, bool, bytes]:
        first, second = self._exact(2)
        fin, opcode = bool(first & 0x80), first & 0x0F
        masked, size = bool(second & 0x80), second & 0x7F
        if size == 126:
            size = struct.unpack("!H", self._exact(2))[0]
        elif size == 127:
            size = struct.unpack("!Q", self._exact(8))[0]
        key = self._exact(4) if masked else b""
        body = self._exact(size) if size else b""
        if masked:
            body = bytes(byte ^ key[i % 4] for i, byte in enumerate(body))
        self.seen.masked.append(masked)
        self.seen.lengths.append(size)
        return fin, opcode, masked, body

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802  (the stdlib spells it this way)
        self.close_connection = True
        self.connection.settimeout(20.0)
        if self.path == "/plain":
            body = b"not a websocket"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        key = self.headers.get("Sec-WebSocket-Key", "")
        token = base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")
        if self.path == "/bad":
            token = "AAAAAAAAAAAAAAAAAAAAAAAAAAA="
        head = (
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + token.encode("ascii") + b"\r\n\r\n"
        )
        if self.path == "/greet":
            # The 101 and the first frame in one write, which is what a fast
            # server really does and what loses a message if the handshake
            # reader discards its leftover bytes.
            self.wfile.write(head)
            _write_frame(self.wfile, OP_TEXT, b"hello from the same segment")
            self._echo()
            return
        self.wfile.write(head)
        self.wfile.flush()
        if self.path == "/bad":
            return
        if self.path == "/split":
            self._split()
            return
        if self.path == "/masked":
            _write_frame(self.wfile, OP_TEXT, b"server frames must not be masked", mask=True)
            self._echo()
            return
        self._echo()

    def _echo(self) -> None:
        """Mirror data frames, answer pings, complete the close handshake."""
        try:
            while True:
                fin, opcode, _masked, body = self._read_frame()
                self.seen.frames.put((fin, opcode, body))
                if opcode == OP_CLOSE:
                    _write_frame(self.wfile, OP_CLOSE, body[:2] if len(body) >= 2 else b"")
                    return
                if opcode == OP_PING:
                    _write_frame(self.wfile, OP_PONG, body)
                elif opcode in (OP_TEXT, OP_BINARY):
                    _write_frame(self.wfile, opcode, body)
        except (EOFError, OSError, struct.error):
            return

    def _split(self) -> None:
        """One text message in two fragments with a ping wedged between them."""
        _write_frame(self.wfile, OP_TEXT, SPLIT_RAW[:SPLIT_AT], fin=False)
        _write_frame(self.wfile, OP_PING, PING_TOKEN)
        _write_frame(self.wfile, OP_CONT, SPLIT_RAW[SPLIT_AT:], fin=True)
        self._echo()


@pytest.fixture
def server():
    """A live WebSocket server; yields `(url_for_path, seen)`."""
    seen = Received()
    Handler.seen = seen
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        yield (lambda path: f"ws://127.0.0.1:{port}{path}"), seen
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


# -- the handshake -----------------------------------------------------------


def test_the_accept_token_matches_the_worked_example_in_rfc_6455():
    assert accept_token("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_a_client_key_is_sixteen_fresh_random_bytes():
    first, second = new_key(), new_key()
    assert len(base64.b64decode(first)) == 16
    assert first != second, "a reused nonce defeats the point of the handshake"


def test_a_correct_accept_token_is_accepted(server):
    url, _seen = server
    ws = WebSocket.connect(url("/echo"), timeout=10)
    try:
        assert ws.alive
    finally:
        ws.close()


def test_a_wrong_accept_token_is_refused(server):
    url, _seen = server
    with pytest.raises(HandshakeError) as caught:
        WebSocket.connect(url("/bad"), timeout=10)
    assert "sec-websocket-accept" in str(caught.value)


def test_a_plain_http_endpoint_is_not_treated_as_a_websocket(server):
    url, _seen = server
    with pytest.raises(HandshakeError) as caught:
        WebSocket.connect(url("/plain"), timeout=10)
    assert "101" in str(caught.value)


def test_a_frame_arriving_with_the_upgrade_response_is_not_lost(server):
    url, _seen = server
    ws = WebSocket.connect(url("/greet"), timeout=10)
    try:
        message = ws.recv(timeout=5)
        assert message is not None and message.text == "hello from the same segment"
    finally:
        ws.close()


def test_connecting_to_a_url_that_is_not_a_websocket_scheme_is_refused():
    with pytest.raises(HandshakeError):
        WebSocket.connect("http://127.0.0.1:1/x", timeout=1)


# -- masking -----------------------------------------------------------------


def test_masking_is_its_own_inverse():
    key = b"\xde\xad\xbe\xef"
    for payload in (b"", b"a", b"abcd", b"abcde", bytes(range(256)) * 7):
        assert mask_bytes(mask_bytes(payload, key), key) == payload


def test_a_client_frame_sets_the_mask_bit_and_hides_the_payload():
    wire = encode_frame(OP_TEXT, b"visible?", mask=True, key=b"\x01\x02\x03\x04")
    assert wire[1] & 0x80, "the MASK bit must be set on a client frame"
    assert wire[2:6] == b"\x01\x02\x03\x04"
    assert wire[6:] != b"visible?", "an unmasked payload means masking never ran"
    frame, used = decode_frame(wire)
    assert (frame.payload, frame.masked, used) == (b"visible?", True, len(wire))


def test_every_client_frame_gets_a_fresh_masking_key():
    keys = {encode_frame(OP_TEXT, b"same payload")[2:6] for _ in range(32)}
    assert len(keys) > 1, "one key for every frame is the bug this guards"


def test_a_server_unmasks_a_real_client_frame(server):
    url, seen = server
    ws = WebSocket.connect(url("/echo"), timeout=10)
    try:
        ws.send_text("masked on the way out")
        message = ws.recv(timeout=5)
        assert message is not None and message.text == "masked on the way out"
    finally:
        ws.close()
    assert seen.masked and all(seen.masked), "every client frame must arrive masked"


def test_a_masked_server_frame_is_rejected(server):
    url, _seen = server
    ws = WebSocket.connect(url("/masked"), timeout=10)
    try:
        with pytest.raises(WebSocketClosed):
            for _ in range(50):
                ws.recv(timeout=0.5)
        assert "masked" in (ws.failure or "")
    finally:
        ws.close()


# -- length encodings --------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "marker", "header"),
    [(0, 0, 2), (1, 1, 2), (125, 125, 2), (126, 126, 4), (65535, 126, 4), (65536, 127, 10)],
)
def test_the_length_field_uses_the_shortest_legal_encoding(size, marker, header):
    wire = encode_frame(OP_BINARY, b"z" * size, mask=False)
    assert wire[1] & 0x7F == marker
    frame, used = decode_frame(wire)
    assert len(frame.payload) == size
    assert used == header + size


@pytest.mark.parametrize("size", [0, 1, 125, 126, 127, 65535, 65536, 70000])
def test_all_three_payload_length_encodings_round_trip_over_a_socket(server, size):
    url, seen = server
    ws = WebSocket.connect(url("/echo"), timeout=15)
    payload = bytes((i * 7 + 3) & 0xFF for i in range(size))
    try:
        ws.send_bytes(payload)
        message = ws.recv(timeout=20)
        assert message is not None
        assert message.opcode == OP_BINARY
        assert message.data == payload, f"{size} bytes did not survive the round trip"
    finally:
        ws.close()
    assert size in seen.lengths


def test_decoding_a_truncated_frame_asks_for_more_rather_than_failing():
    for size in (10, 300, 70000):
        wire = encode_frame(OP_BINARY, b"y" * size, mask=True)
        for cut in (1, 2, 3, 4, 6, len(wire) - 1):
            assert decode_frame(wire[:cut]) is None, f"{cut} bytes of a {size}-byte frame"
        assert decode_frame(wire) is not None


def test_a_second_frame_in_the_buffer_is_left_alone():
    two = encode_frame(OP_TEXT, b"first") + encode_frame(OP_TEXT, b"second")
    first, used = decode_frame(two)
    assert first.payload == b"first"
    rest, _ = decode_frame(two[used:])
    assert rest.payload == b"second"


def test_an_oversized_message_ends_the_connection_instead_of_the_process(server):
    url, _seen = server
    ws = WebSocket.connect(url("/echo"), timeout=10, max_message=64)
    try:
        ws.send_bytes(b"x" * 4096)
        with pytest.raises(WebSocketClosed):
            for _ in range(50):
                ws.recv(timeout=0.5)
        assert "exceeded" in (ws.failure or "")
    finally:
        ws.close()


# -- fragmentation and control frames ---------------------------------------


def test_the_split_point_really_cuts_a_character_in_half():
    with pytest.raises(UnicodeDecodeError):
        SPLIT_RAW[:SPLIT_AT].decode("utf-8")


def test_a_utf8_character_split_across_continuation_frames_reassembles(server):
    url, _seen = server
    ws = WebSocket.connect(url("/split"), timeout=10)
    try:
        message = ws.recv(timeout=5)
        assert message is not None
        assert message.text == SPLIT_TEXT
    finally:
        ws.close()


def test_a_ping_between_continuation_frames_is_answered_and_does_not_corrupt_the_message(server):
    url, seen = server
    ws = WebSocket.connect(url("/split"), timeout=10)
    try:
        message = ws.recv(timeout=5)
        assert message is not None and message.text == SPLIT_TEXT
        pongs = []
        deadline = 40
        while deadline and not pongs:
            deadline -= 1
            try:
                fin, opcode, body = seen.frames.get(timeout=0.25)
            except queue.Empty:
                continue
            if opcode == OP_PONG:
                pongs.append((fin, body))
        assert pongs, "a ping must be answered even mid-message"
        assert pongs[0][1] == PING_TOKEN, "the pong must carry the ping's payload"
        assert ws.pings_answered == 1
    finally:
        ws.close()


def test_a_ping_we_send_comes_back_as_a_pong(server):
    url, _seen = server
    ws = WebSocket.connect(url("/echo"), timeout=10)
    try:
        ws.ping(b"tok")
        for _ in range(40):
            if ws.pongs_seen:
                break
            ws.recv(timeout=0.25)
        assert ws.pongs_seen == 1
    finally:
        ws.close()


def test_a_control_frame_may_not_be_fragmented_or_oversized():
    with pytest.raises(ProtocolError):
        encode_frame(OP_PING, b"x", fin=False)
    with pytest.raises(ProtocolError):
        encode_frame(OP_PING, b"x" * (MAX_CONTROL + 1))
    fragmented = bytearray(encode_frame(OP_PING, b"x", mask=False))
    fragmented[0] &= 0x7F  # clear FIN on a control frame
    with pytest.raises(ProtocolError):
        decode_frame(bytes(fragmented))


def test_reserved_bits_with_no_extension_negotiated_are_rejected():
    wire = bytearray(encode_frame(OP_TEXT, b"hi", mask=False))
    wire[0] |= 0x40
    with pytest.raises(ProtocolError):
        decode_frame(bytes(wire))


def test_a_continuation_frame_with_nothing_to_continue_is_a_protocol_error():
    # Exercised through the reader by feeding it a socket pair directly, which
    # is cheaper than another route and tests the same branch.
    left, right = socket.socketpair()
    ws = WebSocket(left, url="ws://pair/")
    try:
        right.sendall(encode_frame(OP_CONT, b"orphan", mask=False))
        with pytest.raises(WebSocketClosed):
            for _ in range(50):
                ws.recv(timeout=0.5)
        assert "continuation" in (ws.failure or "")
    finally:
        ws.close()
        right.close()


def test_text_and_binary_messages_keep_their_opcode(server):
    url, _seen = server
    ws = WebSocket.connect(url("/echo"), timeout=10)
    try:
        ws.send_bytes(b"\x00\xff\x00")
        first = ws.recv(timeout=5)
        ws.send_text("plain")
        second = ws.recv(timeout=5)
        assert first is not None and first.opcode == OP_BINARY and first.text is None
        assert second is not None and second.opcode == OP_TEXT and second.is_text
    finally:
        ws.close()


# -- closing -----------------------------------------------------------------


def test_a_close_carries_its_status_code_to_the_peer(server):
    url, seen = server
    ws = WebSocket.connect(url("/echo"), timeout=10)
    ws.close(1001, "going away")
    codes = []
    for _ in range(40):
        try:
            _fin, opcode, body = seen.frames.get(timeout=0.25)
        except queue.Empty:
            continue
        if opcode == OP_CLOSE:
            codes.append(parse_close(body))
            break
    assert codes and codes[0] == (1001, "going away")


def test_a_peer_close_is_reported_and_stops_further_sends(server):
    url, _seen = server
    ws = WebSocket.connect(url("/echo"), timeout=10)
    try:
        ws.send_text("bye")
        assert ws.recv(timeout=5) is not None
        ws._send_close(CLOSE_NORMAL, "")  # provoke the server's close reply
        with pytest.raises(WebSocketClosed):
            for _ in range(50):
                ws.recv(timeout=0.5)
        assert ws.close_code == CLOSE_NORMAL
        assert not ws.alive
        with pytest.raises(WebSocketClosed):
            ws.send_text("too late")
    finally:
        ws.close()


def test_closing_twice_is_harmless(server):
    url, _seen = server
    ws = WebSocket.connect(url("/echo"), timeout=10)
    ws.close()
    ws.close()
    assert not ws.alive


def test_an_empty_close_body_reads_as_no_status():
    assert parse_close(b"") == (1005, "")
    assert parse_close(struct.pack("!H", 1002) + b"nope") == (1002, "nope")


def test_a_dead_peer_raises_instead_of_hanging():
    left, right = socket.socketpair()
    ws = WebSocket(left, url="ws://pair/")
    try:
        right.close()
        with pytest.raises(WebSocketClosed):
            for _ in range(50):
                ws.recv(timeout=0.5)
    finally:
        ws.close()


def test_diagnostics_names_the_failure(server):
    url, _seen = server
    ws = WebSocket.connect(url("/echo"), timeout=10)
    try:
        assert "127.0.0.1" in ws.diagnostics()
    finally:
        ws.close()
