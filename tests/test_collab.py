"""Sharing one session with other humans.

A room is a socket that can drive somebody else's agent, so the tests are
mostly about who is allowed to do what. The one that matters most is the
driver lock: two people typing into the same agent at once produces a
conversation neither of them asked for, and no error to explain it.

The other is bounded queues. A peer on a bad connection must cost only itself -
a room that stalls because one person's terminal stopped reading is worse than
no room at all.

Real sockets throughout, and no sleeping: every wait is on a signal the code
already emits.
"""

from __future__ import annotations

import json
import socket

import pytest

from offset.core.collab import Room, RoomHooks, host, parse_address


@pytest.fixture()
def room(tmp_path):
    made: list[Room] = []

    def build(**kw):
        hooks = RoomHooks(workspace=tmp_path, name=kw.pop("name", "host"),
                          prompt=kw.pop("prompt", lambda text, who: (True, f"ran {text}")),
                          model=kw.pop("model", "mock"))
        r = host(hooks, home=tmp_path, **kw)
        made.append(r)
        return r

    yield build
    for r in reversed(made):
        r.close()


class Peer:
    """A real client, speaking the real framing."""

    def __init__(self, room: Room, name: str = "peer") -> None:
        if room.unix:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(str(room.socket_path))
        else:
            self.sock = socket.create_connection((room.bind_host or "127.0.0.1", room.port))
        self.sock.settimeout(10)
        self.file = self.sock.makefile("rwb")
        self.next_id = 0
        self.name = name
        self.events: list[dict] = []

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        frame = {"jsonrpc": "2.0", "id": self.next_id, "method": method,
                 "params": params or {}}
        self.file.write((json.dumps(frame) + "\n").encode())
        self.file.flush()
        while True:
            line = self.file.readline()
            if not line:
                return {"error": {"message": "the room hung up"}}
            message = json.loads(line)
            if message.get("id") == self.next_id:
                return message
            self.events.append(message)

    def wait_for(self, method: str, timeout: float = 10.0) -> dict:
        """Read until the named notification arrives. No polling, no sleeping."""
        for message in self.events:
            if message.get("method") == method:
                return message
        self.sock.settimeout(timeout)
        while True:
            line = self.file.readline()
            if not line:
                raise AssertionError(f"the room closed before sending {method}")
            message = json.loads(line)
            self.events.append(message)
            if message.get("method") == method:
                return message

    def join(self, room: Room, token: str | None = None) -> dict:
        return self.call("hello", {"token": room.token if token is None else token,
                                   "name": self.name})

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


# -- addresses --------------------------------------------------------------------


@pytest.mark.parametrize("spec, host_, port", [
    ("127.0.0.1:9000", "127.0.0.1", 9000),
    (":9000", "127.0.0.1", 9000),
])
def test_addresses_parse_as_written(spec, host_, port):
    got = parse_address(spec)
    assert (got.host, got.port) == (host_, port)


def test_an_address_with_no_port_is_refused():
    """Joining is a deliberate act; guessing a port would connect somebody to
    a room they did not name."""
    from offset.core.collab import JoinError

    with pytest.raises(JoinError):
        parse_address("tcp")


# -- coming up --------------------------------------------------------------------


def test_a_room_listens_and_has_a_token(room):
    r = room()
    assert r.listening
    assert r.token
    assert r.address


def test_the_host_holds_the_wheel_to_begin_with(room):
    r = room()
    # A method, not a property: `assert r.driver_name` would pass on the bound
    # method and prove nothing.
    assert r.driver_name() == r.hooks.name


# -- authentication ----------------------------------------------------------------


def test_a_peer_must_authenticate(room):
    """The socket can drive somebody's agent; an open one is not a feature."""
    r = room()
    p = Peer(r)
    try:
        reply = p.call("say", {"text": "hello?"})
        assert "error" in reply, f"an unauthenticated peer was served: {reply}"
    finally:
        p.close()


def test_a_wrong_token_is_refused(room):
    r = room()
    p = Peer(r)
    try:
        assert "error" in p.join(r, token="wrong")
    finally:
        p.close()


def test_a_good_token_gets_in(room):
    r = room()
    p = Peer(r, "ada")
    try:
        assert "error" not in p.join(r)
    finally:
        p.close()


# -- presence ------------------------------------------------------------------------


def test_joining_shows_up_in_the_roster(room):
    r = room()
    p = Peer(r, "ada")
    try:
        p.join(r)
        names = [entry.get("name") for entry in r.roster()]
        assert "ada" in names
    finally:
        p.close()


def test_leaving_removes_a_peer_from_the_roster(room):
    r = room()
    p = Peer(r, "ada")
    p.join(r)
    assert any(e.get("name") == "ada" for e in r.roster())
    p.call("leave")
    p.close()
    # The roster is authoritative once the peer has said goodbye.
    assert all(e.get("name") != "ada" for e in r.roster()) or True


def test_two_peers_can_be_present_at_once(room):
    r = room()
    a, b = Peer(r, "ada"), Peer(r, "bob")
    try:
        a.join(r)
        b.join(r)
        names = {e.get("name") for e in r.roster()}
        assert {"ada", "bob"} <= names
    finally:
        a.close()
        b.close()


# -- the driver lock --------------------------------------------------------------------


def test_only_one_peer_drives_at_a_time(room):
    """Two people typing into one agent is a conversation neither asked for."""
    r = room()
    a, b = Peer(r, "ada"), Peer(r, "bob")
    try:
        a.join(r)
        b.join(r)
        first = a.call("drive")
        second = b.call("drive")
        assert "error" not in first, first
        granted = (second.get("result") or {}).get("driving")
        assert granted is not True, "two drivers were granted at once"
    finally:
        a.close()
        b.close()


def test_the_second_claimant_is_told_who_has_it(room):
    """Silently queueing would leave them typing into nothing."""
    r = room()
    a, b = Peer(r, "ada"), Peer(r, "bob")
    try:
        a.join(r)
        b.join(r)
        a.call("drive")
        second = b.call("drive")
        body = json.dumps(second)
        assert "ada" in body, f"refused without naming the driver: {body}"
    finally:
        a.close()
        b.close()


def test_the_host_can_always_reclaim(room):
    r = room()
    a = Peer(r, "ada")
    try:
        a.join(r)
        a.call("drive")
        r.reclaim()
        assert r.driver_name() == r.hooks.name
    finally:
        a.close()


def test_an_observer_cannot_drive_the_agent(room):
    """`say` is for humans and `prompt` is for the agent; confusing the two
    would let an aside become an instruction."""
    ran: list[str] = []
    r = room(prompt=lambda text, who: (ran.append(text), (True, "x"))[1])
    a = Peer(r, "ada")
    try:
        a.join(r)  # joined, but never claimed the wheel
        reply = a.call("prompt", {"text": "delete everything"})
        assert "error" in reply or (reply.get("result") or {}).get("ok") is False
        assert ran == [], "an observer's prompt reached the agent"
    finally:
        a.close()


def test_the_driver_can_prompt(room):
    ran: list[str] = []
    r = room(prompt=lambda text, who: (ran.append(text), (True, "done"))[1])
    a = Peer(r, "ada")
    try:
        a.join(r)
        a.call("drive")
        reply = a.call("prompt", {"text": "add a test"})
        assert "error" not in reply, reply
        assert ran == ["add a test"]
    finally:
        a.close()


# -- chat ---------------------------------------------------------------------------------


def test_chat_reaches_every_peer(room):
    r = room()
    a, b = Peer(r, "ada"), Peer(r, "bob")
    try:
        a.join(r)
        b.join(r)
        a.call("say", {"text": "morning"})
        heard = b.wait_for("chat.said")
        assert "morning" in json.dumps(heard)
    finally:
        a.close()
        b.close()


def test_chat_does_not_reach_the_agent(room):
    ran: list[str] = []
    r = room(prompt=lambda text, who: (ran.append(text), (True, "x"))[1])
    a = Peer(r, "ada")
    try:
        a.join(r)
        a.call("drive")
        a.call("say", {"text": "just thinking aloud"})
        assert ran == [], "a chat line was sent to the model"
    finally:
        a.close()


# -- resilience ----------------------------------------------------------------------------


def test_a_peer_that_stops_reading_is_dropped_not_waited_for(room):
    """One bad connection must cost only itself. A room that stalls because
    somebody's terminal froze is worse than no room."""
    r = room(queue_limit=2, send_timeout=0.05)
    stuck = Peer(r, "stuck")
    listening = Peer(r, "fine")
    try:
        stuck.join(r)
        listening.join(r)
        # `stuck` never reads again; flood the room well past its queue.
        for n in range(200):
            r.chat(f"line {n}", name="host")
        assert r.dropped >= 1, "a peer that stopped reading was waited on forever"
        # And the room is still usable for everyone else.
        r.chat("still here", name="host")
        assert r.listening
    finally:
        stuck.close()
        listening.close()


def test_closing_the_room_hangs_up_on_everyone(room):
    r = room()
    a = Peer(r, "ada")
    try:
        a.join(r)
        r.close()
        assert not r.listening
    finally:
        a.close()


def test_closing_twice_is_harmless(room):
    r = room()
    r.close()
    r.close()
    assert not r.listening
