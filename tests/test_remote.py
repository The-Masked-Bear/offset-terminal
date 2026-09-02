"""Running a turn on another machine's daemon.

Nothing here mocks `socket`.  A remote client whose framing was only ever
exercised against a stub is a client that works until the day a tool result
arrives in two packets, which is the failure this module exists to make
impossible - so every test stands up a real listener speaking the real bridge
framing and talks to it over a real socket.

`FakeBridge` is deliberately not `offset.core.bridge.Bridge`: it has to be able
to misbehave on purpose (a bad token, a wrong version, half a frame) and a real
bridge cannot.  What it does share is the wire: `hello` before anything,
newline-delimited JSON-RPC 2.0, notifications with no id.

No test sleeps.  Ordering is proved by waiting on a `threading.Event` or by a
bounded `queue.get` that is required to time out, which is a real negative
signal rather than a guess about scheduling.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import pytest

from offset.core import bridge as bridge_mod
from offset.core import remote as rem


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Isolate `remotes.json` and the default token from the real `~/.offset`."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    return tmp_path


# -- a listener that speaks the real protocol --------------------------------


class Wire:
    """The server side of one connection, for handlers that frame their own."""

    def __init__(self, conn: socket.socket) -> None:
        self.conn = conn

    def raw(self, payload: bytes) -> None:
        self.conn.sendall(payload)

    def frame(self, message: dict) -> bytes:
        return json.dumps(message).encode("utf-8") + b"\n"

    def notify(self, name: str, params: dict | None = None) -> None:
        self.raw(self.frame({"jsonrpc": "2.0", "method": name, "params": params or {}}))

    def reply(self, ident, result) -> None:
        self.raw(self.frame({"jsonrpc": "2.0", "id": ident, "result": result}))


class FakeBridge:
    """A real socket serving the bridge's real framing, with a scriptable table.

    A handler returns a result, which is replied for it, or `None` to say "I
    wrote my own frames" - which is how the split-write and event-stream tests
    control byte boundaries.
    """

    def __init__(self, *, path=None, token="s3cret", version=bridge_mod.BRIDGE_VERSION):
        self.token = token
        self.version = version
        #: Every method name in the order it arrived, across all connections.
        #: This is what proves `hello` came first.
        self.calls: list[str] = []
        self.handlers: dict = {}
        self._stop = threading.Event()
        if path is not None:
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.bind(str(path))
            self.address = f"unix:{path}"
        else:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.bind(("127.0.0.1", 0))
            self.address = "127.0.0.1:%d" % self.server.getsockname()[1]
        self.server.listen(4)
        self.server.settimeout(0.1)
        self._threads: list[threading.Thread] = []
        self._accept = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept.start()

    def on(self, method: str, handler) -> None:
        self.handlers[method] = handler

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            thread = threading.Thread(target=self._serve, args=(conn,), daemon=True)
            self._threads.append(thread)
            thread.start()

    def _serve(self, conn: socket.socket) -> None:
        wire = Wire(conn)
        authenticated = False
        buffer = bytearray()
        conn.settimeout(0.1)
        try:
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except TimeoutError:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buffer += chunk
                while (cut := buffer.find(b"\n")) >= 0:
                    line = bytes(buffer[:cut])
                    del buffer[: cut + 1]
                    if not line.strip():
                        continue
                    message = json.loads(line)
                    method = message.get("method")
                    ident = message.get("id")
                    params = message.get("params") or {}
                    self.calls.append(method)

                    if not authenticated:
                        if method != "hello":
                            wire.raw(wire.frame({
                                "jsonrpc": "2.0", "id": ident,
                                "error": {"code": bridge_mod.UNAUTHENTICATED,
                                          "message": f"{method} was sent before hello"},
                            }))
                            return
                        if params.get("token") != self.token:
                            wire.raw(wire.frame({
                                "jsonrpc": "2.0", "id": ident,
                                "error": {"code": bridge_mod.UNAUTHENTICATED,
                                          "message": "the token does not match"},
                            }))
                            return
                        authenticated = True
                        wire.reply(ident, {
                            "ok": True, "version": self.version, "protocol": "2.0",
                            "workspace": "/remote/work",
                            "events": list(bridge_mod.EVENTS),
                            "methods": list(bridge_mod.METHODS),
                        })
                        continue

                    handler = self.handlers.get(method)
                    if handler is None:
                        wire.raw(wire.frame({
                            "jsonrpc": "2.0", "id": ident,
                            "error": {"code": bridge_mod.METHOD_NOT_FOUND,
                                      "message": f"no method named {method!r}"},
                        }))
                        continue
                    result = handler(params, wire, ident)
                    if result is not None:
                        wire.reply(ident, result)
        except OSError:
            # The client hung up mid-reply, which is exactly what the disposal
            # test does on purpose.  A real bridge drops such a client rather
            # than raising, and an unhandled thread exception here would be
            # reported against whichever later test happened to be running.
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self.server.close()
        except OSError:
            pass


@pytest.fixture
def bridge(tmp_path):
    """A unix-socket fake bridge, torn down whatever the test did."""
    made: list[FakeBridge] = []

    def build(**kwargs):
        kwargs.setdefault("path", tmp_path / "fake.sock")
        server = FakeBridge(**kwargs)
        made.append(server)
        return server

    yield build
    for server in made:
        server.close()


# -- hello ------------------------------------------------------------------


def test_hello_precedes_every_other_call(bridge):
    """The bridge hangs up on a pre-hello method, so the client must not send
    one: `hello` is the first frame on the wire, always."""
    server = bridge()
    server.on("status", lambda params, wire, ident: {"model": "m", "state": "idle"})
    with rem.RemoteAgent(server.address, token="s3cret") as client:
        client.status()
        client.status()
    assert server.calls == ["hello", "status", "status"]


def test_a_call_before_connecting_is_refused_locally(bridge):
    """Sending it would cost the socket, not just the call, so `request`
    refuses rather than letting the bridge teach us."""
    server = bridge()
    client = rem.RemoteAgent(server.address, token="s3cret")
    with pytest.raises(rem.RemoteError) as caught:
        client.status()
    assert "not connected" in str(caught.value)
    assert caught.value.address == server.address
    assert server.calls == []


def test_the_greeting_reports_the_remote_workspace(bridge):
    """A remote's workspace is the far side's answer, never this build's guess."""
    server = bridge()
    with rem.RemoteAgent(server.address, token="s3cret") as client:
        assert client.greeting["workspace"] == "/remote/work"
        assert client.ready


# -- authentication ---------------------------------------------------------


def test_a_bad_token_is_refused_and_the_client_is_not_ready(bridge):
    """A wrong token must fail loudly at connect, not silently produce a client
    whose every later call dies on a closed socket."""
    server = bridge()
    client = rem.RemoteAgent(server.address, token="wrong")
    with pytest.raises(rem.RemoteError) as caught:
        client.connect()
    assert "token" in str(caught.value)
    assert caught.value.code == bridge_mod.UNAUTHENTICATED
    assert not client.ready


def test_a_missing_token_file_names_the_path(tmp_path):
    """The usual cause is that the token was never copied across, so the
    message has to say which file was looked for."""
    entry = rem.Remote("box", "127.0.0.1:7799", token_path=str(tmp_path / "gone.token"))
    with pytest.raises(rem.RemoteError) as caught:
        entry.read_token()
    assert str(tmp_path / "gone.token") in str(caught.value)


def test_a_remote_without_a_token_path_uses_this_machines_bridge_token(home):
    """The common setup - a token copied to `~/.offset/bridge.token` - must not
    require spelling the path out."""
    (home / bridge_mod.TOKEN_NAME).write_text("copied-across\n", encoding="utf-8")
    assert rem.Remote("box", "127.0.0.1:7799").read_token() == "copied-across"


# -- version ----------------------------------------------------------------


def test_a_version_mismatch_names_both_versions(bridge):
    """Carrying on would mis-read a payload and report the wrong thing about a
    file, so the mismatch is stated rather than negotiated around."""
    server = bridge(version="99")
    client = rem.RemoteAgent(server.address, token="s3cret")
    with pytest.raises(rem.RemoteError) as caught:
        client.connect()
    message = str(caught.value)
    assert "99" in message
    assert bridge_mod.BRIDGE_VERSION in message
    assert not client.ready


# -- streaming --------------------------------------------------------------


def test_events_stream_back_in_order_while_the_turn_runs(bridge):
    """A UI that only learned what happened when `prompt` returned would show
    nothing for the length of a refactor, so every event must arrive before the
    reply and in the order the remote pushed it."""
    server = bridge()
    published = [
        ("agent.started", {"step": 1, "model": "big"}),
        ("tool.started", {"id": "a", "tool": "read"}),
        ("tool.finished", {"id": "a", "tool": "read", "ok": True}),
        ("edit.applied", {"source": "tool", "tool": "write", "id": "b"}),
        ("agent.finished", {"reason": "stop", "steps": 1, "text": "done"}),
    ]

    def handle_prompt(params, wire, ident):
        for name, payload in published:
            wire.notify(name, payload)
        return {"ok": True, "text": params["text"].upper(), "steps": 1}

    server.on("prompt", handle_prompt)

    seen: list[str] = []
    client = rem.RemoteAgent(server.address, token="s3cret")
    client.subscribe(lambda name, params: seen.append(name))
    with client:
        reply = client.prompt("refactor", timeout=10)

    assert reply["text"] == "REFACTOR"
    assert seen == [name for name, _ in published]


def test_run_streams_events_for_a_named_remote(bridge, home):
    """`run` is the whole feature: a name, a prompt, and the remote's events
    appearing locally."""
    server = bridge()

    def handle_prompt(params, wire, ident):
        wire.notify("agent.started", {"step": 1, "model": "big"})
        wire.notify("agent.finished", {"reason": "stop", "steps": 1, "text": "ok"})
        return {"ok": True, "text": "ok", "steps": 1}

    server.on("prompt", handle_prompt)
    (home / bridge_mod.TOKEN_NAME).write_text("s3cret", encoding="utf-8")
    rem.add_remote("box", server.address, home=home)

    seen: list[tuple[str, dict]] = []
    reply = rem.run("box", "go", on_event=lambda n, p: seen.append((n, p)), timeout=10, home=home)

    assert reply["ok"] is True
    assert [name for name, _ in seen] == ["agent.started", "agent.finished"]


def test_a_subscriber_that_raises_does_not_lose_the_turn(bridge):
    """A display bug is not worth a failed refactor."""
    server = bridge()

    def handle_prompt(params, wire, ident):
        wire.notify("agent.started", {"step": 1})
        return {"ok": True, "text": "survived"}

    server.on("prompt", handle_prompt)

    def explode(name, params):
        raise RuntimeError("the UI is broken")

    client = rem.RemoteAgent(server.address, token="s3cret", on_event=explode)
    with client:
        assert client.prompt("go", timeout=10)["text"] == "survived"


# -- framing ----------------------------------------------------------------


def test_a_frame_split_across_two_writes_is_reassembled(bridge):
    """The wire is a byte stream: one `recv` can return half a frame.  Proved
    deterministically - the client is required *not* to answer while it holds
    only the first half, and to answer once the remainder arrives."""
    server = bridge()
    wrote_half = threading.Event()
    send_rest = threading.Event()

    def handle_status(params, wire, ident):
        body = wire.frame({"jsonrpc": "2.0", "id": ident, "result": {"state": "idle", "model": "m"}})
        cut = len(body) // 2
        wire.raw(body[:cut])
        wrote_half.set()
        assert send_rest.wait(10), "the test never released the second half"
        wire.raw(body[cut:])
        return None  # framed by hand

    server.on("status", handle_status)

    answers: queue.Queue = queue.Queue()
    with rem.RemoteAgent(server.address, token="s3cret") as client:
        threading.Thread(
            target=lambda: answers.put(client.status()), daemon=True
        ).start()
        assert wrote_half.wait(10)
        # A bounded wait that must expire: if the client had acted on half a
        # frame, an answer would already be here.
        with pytest.raises(queue.Empty):
            answers.get(timeout=0.5)
        assert client.buffered > 0
        send_rest.set()
        assert answers.get(timeout=10) == {"state": "idle", "model": "m"}


def test_two_frames_in_one_write_are_both_delivered(bridge):
    """The other half of the same problem: a `recv` can also return a frame and
    a half, and a reader that handled one line per read would stall."""
    server = bridge()

    def handle_prompt(params, wire, ident):
        both = wire.frame({"jsonrpc": "2.0", "method": "agent.started", "params": {"step": 1}})
        both += wire.frame({"jsonrpc": "2.0", "id": ident, "result": {"ok": True, "text": "x"}})
        wire.raw(both)
        return None

    server.on("prompt", handle_prompt)
    seen: list[str] = []
    client = rem.RemoteAgent(server.address, token="s3cret", on_event=lambda n, p: seen.append(n))
    with client:
        assert client.prompt("go", timeout=10)["text"] == "x"
    assert seen == ["agent.started"]


def test_an_unreadable_frame_does_not_drop_the_connection(bridge):
    """A frame we cannot parse is the remote's problem; losing the socket over
    it would turn one bad line into a lost turn."""
    server = bridge()

    def handle_prompt(params, wire, ident):
        wire.raw(b"{this is not json}\n")
        return {"ok": True, "text": "still here"}

    server.on("prompt", handle_prompt)
    with rem.RemoteAgent(server.address, token="s3cret") as client:
        assert client.prompt("go", timeout=10)["text"] == "still here"


# -- reachability -----------------------------------------------------------


def test_an_unreachable_tcp_address_is_named(home):
    """"Connection refused" without an address is useless when the whole point
    of the thing is that it is somewhere else."""
    # Bind and release, so the port is one nothing is listening on.
    port = rem.free_port()
    address = f"127.0.0.1:{port}"
    client = rem.RemoteAgent(address, token="x", connect_timeout=2.0)
    with pytest.raises(rem.RemoteError) as caught:
        client.connect()
    assert address in str(caught.value)
    assert caught.value.address == address


def test_an_unreachable_unix_socket_is_named(tmp_path):
    """Same rule for the local transport, with the path in the message."""
    missing = tmp_path / "nothing.sock"
    address = f"unix:{missing}"
    client = rem.RemoteAgent(address, token="x")
    with pytest.raises(rem.RemoteError) as caught:
        client.connect()
    assert str(missing) in str(caught.value)


def test_a_remote_that_hangs_up_mid_call_fails_rather_than_waits(bridge):
    """A closed socket must resolve the outstanding call at once; waiting out a
    thirty-minute `prompt` deadline for a reply that cannot come is the bug."""
    server = bridge()

    def handle_prompt(params, wire, ident):
        wire.conn.close()
        return None

    server.on("prompt", handle_prompt)
    started = time.monotonic()
    with rem.RemoteAgent(server.address, token="s3cret") as client:
        with pytest.raises(rem.RemoteError) as caught:
            client.prompt("go", timeout=30)
    assert time.monotonic() - started < 20
    assert server.address in str(caught.value)


@pytest.mark.parametrize(
    "address, reason",
    [
        ("", "address"),
        ("just-a-host", "unix socket path"),
        ("127.0.0.1:notaport", "port number"),
        ("127.0.0.1:99999", "out of range"),
        ("unix:", "no socket path"),
    ],
)
def test_an_unusable_address_is_refused_when_it_is_typed(address, reason):
    """A typo is worth reporting while the user is still looking at the command
    they typed, not twenty minutes later."""
    with pytest.raises(rem.RemoteError) as caught:
        rem.parse_address(address)
    assert reason in str(caught.value)


@pytest.mark.parametrize(
    "address, expected",
    [
        ("127.0.0.1:7799", ("tcp", "127.0.0.1", 7799)),
        ("tcp:10.0.0.4:22", ("tcp", "10.0.0.4", 22)),
        ("[::1]:7799", ("tcp", "::1", 7799)),
        ("unix:/tmp/a.sock", ("unix", "/tmp/a.sock", 0)),
        ("/tmp/b.sock", ("unix", "/tmp/b.sock", 0)),
    ],
)
def test_every_address_the_daemon_can_bind_is_understood(address, expected):
    """The client must accept exactly what `offset daemon --listen` produces."""
    assert rem.parse_address(address) == expected


# -- disposal ---------------------------------------------------------------


def test_a_disposed_client_cancels_calls_still_in_flight(bridge):
    """Otherwise the caller waits out the full `prompt` timeout for a reply
    that can never arrive, and the UI claims to be busy for half an hour."""
    server = bridge()
    holding = threading.Event()
    release = threading.Event()

    def handle_prompt(params, wire, ident):
        holding.set()
        release.wait(10)  # never replies until the test says so
        return {"ok": True}

    server.on("prompt", handle_prompt)
    outcomes: queue.Queue = queue.Queue()
    client = rem.RemoteAgent(server.address, token="s3cret")
    client.connect()

    def call() -> None:
        try:
            outcomes.put(("ok", client.prompt("go", timeout=60)))
        except rem.RemoteError as exc:
            outcomes.put(("err", str(exc)))

    threading.Thread(target=call, daemon=True).start()
    assert holding.wait(10)
    client.dispose()
    kind, detail = outcomes.get(timeout=10)
    release.set()
    assert kind == "err"
    assert "disposed" in detail
    assert not client.ready


def test_disposal_is_idempotent(bridge):
    """`run` disposes through a context manager and a caller may dispose too;
    the second call must not raise on an already-closed socket."""
    server = bridge()
    client = rem.RemoteAgent(server.address, token="s3cret")
    client.connect()
    client.dispose()
    client.dispose()
    assert not client.ready


# -- the registry -----------------------------------------------------------


def test_remotes_round_trip_through_the_config_file(home):
    """A remote added in one session must be there in the next."""
    rem.add_remote("box", "10.0.0.4:7799", token_path="/keys/box.token", workspace="/srv/app", home=home)
    rem.add_remote("laptop", "unix:/tmp/offset.sock", home=home)

    found = rem.remotes(home)
    assert [r.name for r in found] == ["box", "laptop"]
    assert found[0].address == "10.0.0.4:7799"
    assert found[0].token_path == "/keys/box.token"
    assert found[0].workspace == "/srv/app"
    assert (home / rem.REMOTES_NAME).exists()


def test_adding_the_same_name_twice_replaces_it(home):
    """Re-pointing a remote at a new port is the common edit; two entries with
    one name would make `find_remote` a coin toss."""
    rem.add_remote("box", "10.0.0.4:7799", home=home)
    rem.add_remote("box", "10.0.0.9:8800", home=home)
    found = rem.remotes(home)
    assert len(found) == 1
    assert found[0].address == "10.0.0.9:8800"


def test_removing_a_remote_reports_whether_there_was_one(home):
    rem.add_remote("box", "10.0.0.4:7799", home=home)
    assert rem.remove_remote("box", home=home) is True
    assert rem.remove_remote("box", home=home) is False
    assert rem.remotes(home) == []


def test_an_unknown_remote_lists_the_known_ones(home):
    """A typo'd name should show what was available, not just fail."""
    rem.add_remote("box", "10.0.0.4:7799", home=home)
    with pytest.raises(rem.RemoteError) as caught:
        rem.find_remote("bxo", home=home)
    assert "box" in str(caught.value)


def test_a_corrupt_config_means_no_remotes_rather_than_a_traceback(home):
    """Startup reads this file; a hand-edited one must not stop offset."""
    (home / rem.REMOTES_NAME).write_text("{not json", encoding="utf-8")
    assert rem.remotes(home) == []


def test_one_mangled_entry_does_not_cost_the_others(home):
    """Losing every remote because one entry lost its address is a bad trade."""
    (home / rem.REMOTES_NAME).write_text(
        json.dumps({"version": 1, "remotes": [
            {"name": "good", "address": "10.0.0.4:7799"},
            {"name": "nameless"},
            "rubbish",
        ]}),
        encoding="utf-8",
    )
    assert [r.name for r in rem.remotes(home)] == ["good"]


@pytest.mark.parametrize("name", ["", "  ", "a b", "../etc", "with/slash", "-leading"])
def test_an_unusable_remote_name_is_refused(name, home):
    """The name is typed as a bare command argument and shown in a table."""
    with pytest.raises(rem.RemoteError):
        rem.add_remote(name, "10.0.0.4:7799", home=home)


def test_a_bad_address_is_refused_at_add_time(home):
    rem.add_remote("box", "10.0.0.4:7799", home=home)
    with pytest.raises(rem.RemoteError):
        rem.add_remote("other", "no-port-here", home=home)
    assert [r.name for r in rem.remotes(home)] == ["box"]


def test_a_partly_written_config_never_replaces_a_good_one(home, monkeypatch):
    """The write is atomic, so an interruption leaves the previous list intact
    rather than a truncated file that loses every remote."""
    rem.add_remote("box", "10.0.0.4:7799", home=home)

    def explode(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(rem.json, "dump", explode)
    with pytest.raises(OSError):
        rem.add_remote("laptop", "10.0.0.9:8800", home=home)
    monkeypatch.undo()
    assert [r.name for r in rem.remotes(home)] == ["box"]
    # And no temp file was left behind for the next reader to trip over.
    assert [p.name for p in home.iterdir() if p.name.startswith(".remotes.")] == []


# -- ssh tunnelling ---------------------------------------------------------


def test_the_ssh_argv_forwards_loopback_to_loopback():
    """`-N` so no remote command runs, and the local end pinned to 127.0.0.1 so
    the forward is not itself exposed to the network."""
    argv = rem.ssh_command("pi@box", 5000, 7799)
    assert argv[0] == "ssh"
    assert "-N" in argv
    assert argv[argv.index("-L") + 1] == "127.0.0.1:5000:127.0.0.1:7799"
    assert argv[-1] == "pi@box"
    assert "BatchMode=yes" in argv


def test_a_tunnel_without_a_destination_is_refused():
    with pytest.raises(rem.RemoteError):
        rem.ssh_command("   ", 1, 2)


#: A stand-in for `ssh -N -L`: it parses the forward out of the argv it was
#: given and actually listens on the local port, so readiness is real.  Written
#: as source rather than a helper module because it has to run in a child.
_LISTENER = """
import socket, sys, time
sock = socket.socket()
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("127.0.0.1", int(sys.argv[1])))
sock.listen(4)
while True:
    time.sleep(60)
"""


def _fake_ssh(argv: list[str]) -> subprocess.Popen:
    """Spawn a real child that listens where the argv said it would.

    Reads the port back out of `-L` rather than being told, so the argv the
    tunnel built is exercised rather than trusted.
    """
    forward = argv[argv.index("-L") + 1]
    local_port = forward.split(":")[1]
    return subprocess.Popen(
        [sys.executable, "-c", _LISTENER, local_port],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _dead(pid: int) -> bool:
    """Whether a pid we reaped is genuinely gone."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def test_no_tunnel_process_survives_the_call():
    """An `ssh` left running is a route into the remote agent that nothing on
    this machine is watching, plus a descriptor leaked once per call."""
    with rem.tunnel("pi@box", 7799, launch=_fake_ssh, timeout=15) as pipe:
        pid = pipe.proc.pid
        assert pipe.alive
        assert rem._accepting(pipe.local_port)
        assert pipe.address == f"127.0.0.1:{pipe.local_port}"
    assert not pipe.alive
    assert _dead(pid)


def test_a_tunnel_is_torn_down_even_when_the_body_raises():
    """The teardown is in `finally` precisely because the interesting exits are
    the unplanned ones."""
    captured: list[rem.Tunnel] = []

    with pytest.raises(ZeroDivisionError):
        with rem.tunnel("pi@box", 7799, launch=_fake_ssh, timeout=15) as pipe:
            captured.append(pipe)
            1 / 0

    assert captured
    assert not captured[0].alive
    assert _dead(captured[0].proc.pid)


def test_a_tunnel_that_never_listens_fails_and_leaves_nothing_behind():
    """A host that will never forward must be reported, not hung on - and the
    child that failed to forward must still be reaped."""
    spawned: list[subprocess.Popen] = []

    def never_listens(argv: list[str]) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        spawned.append(proc)
        return proc

    with pytest.raises(rem.RemoteError) as caught:
        with rem.tunnel("pi@box", 7799, launch=never_listens, timeout=1.0):
            pytest.fail("the body must not run when the tunnel is not ready")

    assert "pi@box" in str(caught.value)
    assert "127.0.0.1:" in str(caught.value)
    assert spawned and _dead(spawned[0].pid)


def test_a_tunnel_whose_ssh_dies_immediately_is_reported():
    """`ssh` exiting is a final answer; polling to the deadline would only
    delay it."""
    def dies(argv: list[str]) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-c", "raise SystemExit(255)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    with pytest.raises(rem.RemoteError) as caught:
        with rem.tunnel("pi@box", 7799, launch=dies, timeout=15):
            pytest.fail("unreachable")
    assert "exited 255" in str(caught.value)


def test_a_tunnel_that_cannot_be_spawned_names_the_binary():
    def missing(argv: list[str]) -> subprocess.Popen:
        raise FileNotFoundError("no such file: ssh")

    with pytest.raises(rem.RemoteError) as caught:
        with rem.tunnel("pi@box", 7799, launch=missing):
            pytest.fail("unreachable")
    assert "pi@box" in str(caught.value)


def test_over_ssh_yields_a_connected_client_and_closes_both_halves(tmp_path):
    """The nesting that almost everybody wants, in one place, so getting it
    wrong is not how a tunnel survives."""
    server = FakeBridge()  # a real TCP bridge on an ephemeral port
    server.on("status", lambda params, wire, ident: {"state": "idle", "model": "m"})
    remote_port = int(server.address.rsplit(":", 1)[1])

    def forward(argv: list[str]) -> subprocess.Popen:
        """A 'tunnel' that is really the bridge itself: the local port is
        pinned to the bridge's, so a client reaching the tunnel address reaches
        the bridge, and the spawned child is a genuine process to reap."""
        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    try:
        with rem.over_ssh(
            "pi@box", remote_port, token="s3cret", local_port=remote_port, launch=forward
        ) as client:
            assert client.ready
            assert client.status()["state"] == "idle"
            held = client
        assert not held.ready
        assert server.calls[0] == "hello"
    finally:
        server.close()


# -- the command ------------------------------------------------------------


def _state(home):
    """The two attributes `/remote` touches, and nothing else."""
    class State:
        workspace = home
    return State()


def test_the_command_lists_adds_and_removes(home):
    """The registry is reachable from the prompt, which is the only way most
    people will ever touch it."""
    from offset.shell.commands import TONE_ERR

    state = _state(home)
    run = rem.COMMANDS[0].run

    empty = run(state, ["list"])
    assert any("no remotes" in line for line in empty.lines)

    added = run(state, ["add", "box", "10.0.0.4:7799", "workspace=/srv/app"])
    assert any("10.0.0.4:7799" in line for line in added.lines)
    assert [r.name for r in rem.remotes(home)] == ["box"]

    listed = run(state, ["list"])
    assert any("/srv/app" in line for line in listed.lines)

    gone = run(state, ["remove", "box"])
    assert gone.lines == ["forgot box"]
    assert rem.remotes(home) == []

    assert run(state, ["remove", "box"]).tone == TONE_ERR


def test_the_command_refuses_an_unknown_action(home):
    from offset.shell.commands import TONE_ERR

    outcome = rem.COMMANDS[0].run(_state(home), ["frobnicate"])
    assert outcome.tone == TONE_ERR
    assert "frobnicate" in outcome.lines[0]


def test_run_from_the_command_streams_event_lines(bridge, home):
    """`/remote run` returns a job because a turn is slower than a keypress;
    the job's Outcome carries what happened over there."""
    server = bridge()

    def handle_prompt(params, wire, ident):
        wire.notify("tool.started", {"id": "a", "tool": "read"})
        wire.notify("tool.finished", {"id": "a", "tool": "read", "ok": True})
        wire.notify("agent.finished", {"reason": "stop", "steps": 1, "text": "all done"})
        return {"ok": True, "text": "all done", "steps": 1}

    server.on("prompt", handle_prompt)
    (home / bridge_mod.TOKEN_NAME).write_text("s3cret", encoding="utf-8")
    rem.add_remote("box", server.address, home=home)

    live: list[str] = []
    rem.set_event_sink(lambda name, params: live.append(name))
    try:
        outcome = rem.COMMANDS[0].run(_state(home), ["run", "box", "fix", "the", "bug"])
        assert outcome.job is not None
        finished = outcome.job()
    finally:
        rem.set_event_sink(None)

    text = "\n".join(finished.lines)
    assert "fix the bug" in text
    assert "read started" in text
    assert "read ok" in text
    assert "all done" in text
    assert live == ["tool.started", "tool.finished", "agent.finished"]


def test_run_from_the_command_reports_an_unreachable_remote(home):
    """The address has to be in the failure the user reads."""
    from offset.shell.commands import TONE_ERR

    port = rem.free_port()
    (home / bridge_mod.TOKEN_NAME).write_text("s3cret", encoding="utf-8")
    rem.add_remote("box", f"127.0.0.1:{port}", home=home)

    outcome = rem.COMMANDS[0].run(_state(home), ["run", "box", "go"])
    finished = outcome.job()
    assert finished.tone == TONE_ERR
    assert any(f"127.0.0.1:{port}" in line for line in finished.lines)


def test_run_from_the_command_refuses_an_unknown_remote(home):
    from offset.shell.commands import TONE_ERR

    outcome = rem.COMMANDS[0].run(_state(home), ["run", "nope", "go"])
    assert outcome.tone == TONE_ERR
    assert outcome.job is None


def test_commands_are_built_once(home):
    """The lazy `__getattr__` re-enters through the shell registry; a second
    list would register the command twice."""
    assert rem.COMMANDS is rem.COMMANDS
    assert [c.name for c in rem.COMMANDS] == ["remote"]
