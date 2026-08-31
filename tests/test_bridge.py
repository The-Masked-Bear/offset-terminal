"""The editor bridge, driven over a real socket by a hand-rolled client.

There is no fake transport here.  Every test binds an actual unix socket in a
temporary `$OFFSET_HOME`, connects to it the way the extension does, and reads
newline-delimited JSON-RPC frames back.  The properties that matter are the
ones a mocked socket would quietly grant: that an unauthenticated peer is
refused and hung up on, that a client which stops reading is evicted without
stalling the agent's thread, and that the socket and the token are readable by
nobody but their owner.

The clock-sensitive tests shrink the bridge's own limits instead of sleeping:
a four-deep outbox and a fifth of a second of send patience reach the same code
path as the shipped values in a fraction of the time.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
import time

import pytest

from offset.core import bridge as bridge_mod
from offset.core.bridge import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    UNAUTHENTICATED,
    Bridge,
    Hooks,
    read_descriptor,
)

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="the bridge only publishes a file mode on unix"
)


# -- a client, written the way the extension is ------------------------------


class Editor:
    """The smallest honest client: one socket, one line per frame."""

    __slots__ = ("buf", "sock")

    def __init__(self, path, timeout: float = 5.0) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(str(path))
        self.buf = bytearray()

    def raw(self, body: bytes) -> None:
        self.sock.sendall(body)

    def call(self, method: str, ident=1, **params) -> None:
        frame = {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
        self.sock.sendall(json.dumps(frame).encode("utf-8") + b"\n")

    def frame(self) -> dict:
        while (nl := self.buf.find(b"\n")) < 0:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise EOFError("the bridge hung up")
            self.buf += chunk
        line = bytes(self.buf[:nl])
        del self.buf[: nl + 1]
        return json.loads(line)

    def hello(self, token: str) -> dict:
        self.call("hello", ident="hello", token=token)
        return self.frame()

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


@pytest.fixture
def serve(tmp_path, monkeypatch):
    """Build and start bridges, and make sure every one of them is stopped."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    running: list[Bridge] = []
    editors: list[Editor] = []

    def start(**kwargs) -> Bridge:
        kwargs.setdefault("home", tmp_path)
        kwargs.setdefault("hooks", Hooks(workspace=tmp_path))
        made = Bridge(**kwargs)
        problems = made.serve()
        assert problems == [], f"the bridge would not start: {problems}"
        running.append(made)
        return made

    start.editors = editors  # type: ignore[attr-defined]
    try:
        yield start
    finally:
        for editor in editors:
            editor.close()
        for made in running:
            made.shutdown()


@pytest.fixture
def attach(serve):
    """Connect an editor to a bridge and close it at the end of the test."""

    def connect(bridge: Bridge, *, token: str | None = None) -> Editor:
        editor = Editor(bridge.socket_path)
        serve.editors.append(editor)
        if token is not None:
            greeting = editor.hello(token)
            assert greeting["result"]["ok"], greeting
        return editor

    return connect


# -- authentication ----------------------------------------------------------


def test_an_unauthenticated_client_is_refused_and_hung_up_on(serve, attach):
    bridge = serve()
    editor = attach(bridge)
    editor.call("status", ident=1)

    answer = editor.frame()
    assert answer["id"] == 1
    assert answer["error"]["code"] == UNAUTHENTICATED
    assert "authenticate first" in answer["error"]["message"]
    with pytest.raises(EOFError):
        editor.frame()
    assert bridge.dropped == 1, "a socket that skipped hello must be counted as dropped"


def test_the_wrong_token_gets_one_error_frame_and_no_second_chance(serve, attach):
    bridge = serve()
    editor = attach(bridge)
    editor.call("hello", ident=9, token="not the token")

    answer = editor.frame()
    assert answer["error"]["code"] == UNAUTHENTICATED
    assert str(bridge.token_path) in answer["error"]["message"], \
        "the message must say which file holds the real token"
    with pytest.raises(EOFError):
        editor.frame()


def test_an_empty_token_is_not_accepted_even_before_one_is_published(serve, attach):
    bridge = serve()
    editor = attach(bridge)
    editor.call("hello", ident=1, token="")
    assert editor.frame()["error"]["code"] == UNAUTHENTICATED


def test_the_right_token_is_answered_with_the_wire_contract(serve, attach):
    bridge = serve()
    editor = attach(bridge)
    result = editor.hello(bridge.token)["result"]
    assert result["ok"] is True
    assert result["protocol"] == "2.0"
    assert result["pid"] == os.getpid()
    assert "prompt" in result["methods"] and "agent.started" in result["events"]


# -- broadcasting ------------------------------------------------------------


def test_two_concurrent_clients_both_receive_the_same_event(serve, attach):
    bridge = serve()
    first = attach(bridge, token=bridge.token)
    second = attach(bridge, token=bridge.token)

    delivered = bridge.publish("agent.started", {"step": 0, "model": "test"})
    assert delivered == 2, "both attached editors must be reached"

    for editor in (first, second):
        event = editor.frame()
        assert event["method"] == "agent.started"
        assert event["params"]["step"] == 0
        assert event["params"]["model"] == "test"
        assert "at" in event["params"], "every event is stamped"


def test_an_unauthenticated_socket_is_not_broadcast_to(serve, attach):
    bridge = serve()
    listener = attach(bridge, token=bridge.token)
    lurker = attach(bridge)  # connected, never said hello

    assert bridge.publish("agent.finished", {"reason": "done"}) == 1
    assert listener.frame()["method"] == "agent.finished"

    lurker.sock.settimeout(0.3)
    with pytest.raises((TimeoutError, socket.timeout)):
        lurker.frame()


def test_publishing_to_nobody_is_free(serve):
    bridge = serve()
    assert bridge.publish("agent.started", {"step": 0}) == 0


# -- back pressure -----------------------------------------------------------


def test_a_client_that_stops_reading_is_dropped_without_blocking_the_agent(serve, attach):
    """`publish` runs on the agent's thread, so the eviction has to happen
    without it ever waiting on a socket."""
    bridge = serve(queue_limit=4, send_timeout=0.2)
    # Attached, greeted, and from here on it never reads another byte.
    deaf = attach(bridge, token=bridge.token)
    assert deaf.sock.fileno() >= 0, "the deaf peer is still connected"
    assert [client["authenticated"] for client in bridge.clients()] == [True]

    blob = "x" * (128 * 1024)  # fills the kernel buffer in a handful of frames
    slowest = 0.0
    deadline = time.monotonic() + 20.0
    while bridge.dropped == 0 and time.monotonic() < deadline:
        started = time.monotonic()
        bridge.publish("tool.started", {"id": "t", "tool": "read", "args": {"blob": blob}})
        slowest = max(slowest, time.monotonic() - started)

    assert bridge.dropped == 1, "the deaf client should have been evicted"
    assert slowest < 1.0, f"publish blocked the agent for {slowest:.2f}s"
    assert bridge.clients() == [], "a dropped client must not linger in the registry"

    # The server itself is unharmed: a fresh editor still attaches and is served.
    assert bridge.listening
    fresh = attach(bridge, token=bridge.token)
    assert bridge.publish("agent.finished", {"reason": "done"}) == 1
    assert fresh.frame()["method"] == "agent.finished"


def test_a_disconnected_client_is_forgotten(serve, attach):
    bridge = serve()
    editor = attach(bridge, token=bridge.token)
    editor.close()
    for _ in range(100):
        if not bridge.clients():
            break
        time.sleep(0.02)
    assert bridge.clients() == [], "the reader must retire its client on EOF"


# -- dispatch ----------------------------------------------------------------


def test_an_unknown_method_returns_an_error_object_naming_the_alternatives(serve, attach):
    bridge = serve()
    editor = attach(bridge, token=bridge.token)
    editor.call("teleport", ident=7)

    answer = editor.frame()
    assert answer["jsonrpc"] == "2.0"
    assert answer["id"] == 7
    assert "result" not in answer, "an error frame must not also carry a result"
    assert answer["error"]["code"] == METHOD_NOT_FOUND
    message = answer["error"]["message"]
    assert "no method named 'teleport'" in message
    assert "apply_edit" in message and "prompt" in message

    # And the connection survives: an unknown method is not a hanging offence.
    editor.call("status", ident=8)
    assert editor.frame()["id"] == 8


def test_a_frame_that_is_not_json_is_a_parse_error_not_a_disconnect(serve, attach):
    bridge = serve()
    editor = attach(bridge, token=bridge.token)
    editor.raw(b"{ this is not json\n")

    answer = editor.frame()
    assert answer["error"]["code"] == PARSE_ERROR
    editor.call("status", ident=2)
    assert editor.frame()["id"] == 2


def test_a_frame_with_no_method_is_an_invalid_request(serve, attach):
    bridge = serve()
    editor = attach(bridge, token=bridge.token)
    editor.raw(json.dumps({"jsonrpc": "2.0", "id": 3}).encode() + b"\n")
    answer = editor.frame()
    assert answer["id"] == 3
    assert answer["error"]["code"] == INVALID_REQUEST


def test_a_handler_that_raises_becomes_an_error_frame(serve, attach):
    def hostile() -> dict:
        raise RuntimeError("the shell fell over")

    bridge = serve(hooks=Hooks(status=hostile))
    editor = attach(bridge, token=bridge.token)
    editor.call("status", ident=4)
    answer = editor.frame()
    assert "status failed: RuntimeError: the shell fell over" in answer["error"]["message"]

    editor.call("sessions", ident=5)
    assert "result" in editor.frame(), "one broken handler must not kill the connection"


def test_apply_edit_writes_the_file_and_announces_it(serve, attach, tmp_path):
    bridge = serve()
    editor = attach(bridge, token=bridge.token)
    editor.call("apply_edit", ident=6, path="notes.txt", text="hello from the editor\n")

    frames = [editor.frame(), editor.frame()]
    reply = next(f for f in frames if f.get("id") == 6)
    event = next(f for f in frames if f.get("method") == "edit.applied")
    assert reply["result"]["ok"] is True
    assert event["params"]["source"] == "editor"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "hello from the editor\n"


def test_apply_edit_refuses_a_missing_path(serve, attach):
    bridge = serve()
    editor = attach(bridge, token=bridge.token)
    editor.call("apply_edit", ident=1, text="body")
    assert editor.frame()["result"]["error"] == "apply_edit needs a non-empty 'path'"


def test_a_second_prompt_is_refused_while_a_turn_is_running(serve, attach):
    gate = threading.Event()

    def slow(text: str) -> tuple[bool, str]:
        gate.wait(10.0)
        return True, "done"

    bridge = serve(hooks=Hooks(prompt=slow))
    first = attach(bridge, token=bridge.token)
    second = attach(bridge, token=bridge.token)

    first.call("prompt", ident=1, text="do the thing")
    for _ in range(200):  # wait for the turn lock to actually be taken
        if bridge._turn.locked():
            break
        time.sleep(0.01)

    second.call("prompt", ident=2, text="do another thing")
    refusal = second.frame()
    assert refusal["result"]["ok"] is False
    assert "a turn is already running" in refusal["result"]["error"]

    gate.set()
    assert first.frame()["result"]["text"] == "done"


# -- the published files -----------------------------------------------------


def test_the_socket_and_the_token_are_readable_only_by_their_owner(serve):
    bridge = serve()
    for path in (bridge.socket_path, bridge.token_path, bridge.descriptor_path):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600, f"{path.name} is {oct(mode)}, not 0600"


def test_the_descriptor_carries_everything_an_editor_needs(serve, tmp_path):
    bridge = serve()
    descriptor, problem = read_descriptor(tmp_path)
    assert problem is None, problem
    assert descriptor["transport"] == "unix"
    assert descriptor["path"] == str(bridge.socket_path)
    assert descriptor["token"] == bridge.token
    assert descriptor["pid"] == os.getpid()


def test_read_descriptor_says_what_is_missing_rather_than_raising(tmp_path):
    descriptor, problem = read_descriptor(tmp_path)
    assert descriptor is None
    assert "no bridge descriptor" in problem and "is offset running?" in problem


def test_shutdown_removes_every_published_file(serve, tmp_path):
    bridge = serve()
    paths = (bridge.socket_path, bridge.token_path, bridge.descriptor_path)
    assert all(p.exists() for p in paths)
    bridge.shutdown()
    assert not any(p.exists() for p in paths), "a stopped bridge must leave nothing behind"
    assert bridge.token == "", "the token must not outlive the socket"


# -- stale and live sockets --------------------------------------------------


def test_a_stale_socket_from_a_dead_process_is_replaced(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    path = tmp_path / bridge_mod.SOCKET_NAME

    # Exactly what a killed offset leaves behind: the file, and nothing on it.
    corpse = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    corpse.bind(str(path))
    corpse.close()
    assert path.exists(), "the leftover file is the point of this test"
    assert not bridge_mod._alive(path)

    bridge = Bridge(home=tmp_path, hooks=Hooks(workspace=tmp_path))
    try:
        assert bridge.serve() == [], bridge.problems
        assert bridge.listening
        assert bridge_mod._alive(path), "the new bridge must own the same path"
    finally:
        bridge.shutdown()


def test_a_live_socket_is_not_stolen_from_the_offset_that_owns_it(serve, tmp_path):
    first = serve()
    second = Bridge(home=tmp_path, hooks=Hooks(workspace=tmp_path))
    try:
        problems = second.serve()
        assert problems, "the second bridge must refuse to bind"
        assert "already listening" in problems[0]
        assert not second.listening
        assert first.socket_path.exists(), "the refusal must not delete the live socket"
    finally:
        second.shutdown()
    assert bridge_mod._alive(first.socket_path), "the first bridge is still serving"


def test_a_report_names_the_socket_and_its_clients(serve, attach):
    bridge = serve()
    attach(bridge, token=bridge.token)
    lines = bridge.report()
    assert str(bridge.socket_path) in lines[0]
    assert any("1 attached" in line for line in lines)
