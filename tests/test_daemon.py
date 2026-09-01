"""offset with no terminal attached.

The daemon is deliberately thin - it is `build_state` plus the bridge plus a
signal handler - so these tests are about the three things that thinness makes
easy to get wrong: that it actually binds where it was told, that it stops for
the right reasons, and that it closes the session down rather than dropping it.

`serve` takes an injected `build`, so nothing here needs a model, a network or
a real workspace.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import time
from pathlib import Path

import pytest

from offset.core import bridge as bridge_mod
from offset.core import daemon as daemon_mod


class FakeSession:
    def __init__(self) -> None:
        self.id = "sess-1"
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def __len__(self) -> int:
        return 0


class FakeEggs:
    def __init__(self) -> None:
        self.saved = False

    def save(self) -> None:
        self.saved = True


class FakeState:
    """Everything the bridge and the shutdown path reach for."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.model = "mock"
        self.session = FakeSession()
        self.eggs = FakeEggs()
        self.agent = None
        self.toolbox = []
        self.mcp = None


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    yield tmp_path
    bridge_mod.uninstall()


def build_for(workspace: Path):
    def build(root, *, model=None):
        state = FakeState(Path(root))
        state.model = model or "mock"
        return state

    return build


def run_daemon(tmp_path, **kwargs) -> tuple[threading.Thread, io.StringIO]:
    """Start `serve` on a thread and wait until it is actually listening."""
    out = io.StringIO()
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)

    def work() -> None:
        daemon_mod.serve(proj, build=build_for(proj), out=out, **kwargs)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        active = bridge_mod.active()
        if active is not None and active.listening:
            return thread, out
        time.sleep(0.01)
    raise AssertionError(f"the daemon never began listening: {out.getvalue()}")


# -- binding ------------------------------------------------------------------

def test_a_daemon_listens_without_a_terminal(tmp_path):
    thread, _ = run_daemon(tmp_path)
    try:
        bridge = bridge_mod.active()
        assert bridge is not None and bridge.listening
    finally:
        _stop(thread)


def test_the_descriptor_it_prints_is_enough_to_connect(tmp_path):
    """A user reads this and a client parses it: if it does not name where to
    connect, the daemon is useless even though it is running."""
    thread, out = run_daemon(tmp_path)
    try:
        # Only the descriptor has been printed; the closing line comes after
        # `run()` returns, which has not happened while we are still listening.
        printed = json.loads(out.getvalue())
        assert printed["transport"] in ("unix", "tcp")
        assert printed["address"]
        assert Path(printed["descriptor"]).exists()
        assert Path(printed["token_path"]).exists()
        assert printed["workspace"].endswith("proj")
    finally:
        _stop(thread)


def test_it_serves_tcp_when_asked(tmp_path):
    """The reason the daemon exists: a socket file is unreachable from another
    machine, so a remote editor needs a port."""
    thread, _ = run_daemon(tmp_path, listen="127.0.0.1:0")
    try:
        bridge = bridge_mod.active()
        assert not bridge.unix
        assert bridge.port > 0
        with socket.create_connection(("127.0.0.1", bridge.port), timeout=5) as sock:
            assert sock  # it accepts
    finally:
        bridge_mod.active().shutdown()
        _stop(thread)


def test_the_tcp_descriptor_names_the_host_it_actually_bound(tmp_path):
    thread, _ = run_daemon(tmp_path, listen="127.0.0.1:0")
    try:
        bridge = bridge_mod.active()
        descriptor = json.loads(bridge.descriptor_path.read_text())
        assert descriptor["transport"] == "tcp"
        assert descriptor["host"] == "127.0.0.1"
        assert descriptor["port"] == bridge.port
        assert descriptor["path"] == ""
    finally:
        bridge_mod.active().shutdown()
        _stop(thread)


@pytest.mark.parametrize(
    "spec, host, port",
    [
        ("", "127.0.0.1", 0),
        ("tcp", "127.0.0.1", 0),
        ("127.0.0.1:9931", "127.0.0.1", 9931),
        (":9931", "127.0.0.1", 9931),
        ("0.0.0.0:9931", "0.0.0.0", 9931),
        ("example.internal", "example.internal", 0),
    ],
)
def test_listen_addresses_are_parsed_as_written(spec, host, port):
    bridge = bridge_mod.Bridge(listen=spec)
    assert bridge._address() == (host, port)


def test_a_unix_bridge_is_still_the_default():
    """An editor on the same machine should not get a port it did not ask for:
    a socket file carries filesystem permissions, a port does not."""
    assert bridge_mod.Bridge().unix is (hasattr(socket, "AF_UNIX"))


def test_asking_for_a_port_turns_the_unix_socket_off():
    assert bridge_mod.Bridge(listen="tcp").unix is False


# -- lifecycle ----------------------------------------------------------------


def test_it_stops_when_told(tmp_path):
    thread, _ = run_daemon(tmp_path)
    bridge = bridge_mod.active()
    daemon = daemon_mod.Daemon(FakeState(tmp_path), bridge)
    daemon.stop("because a test said so")
    assert daemon.run() == "because a test said so"
    bridge.shutdown()
    _stop(thread)


def test_the_first_reason_to_stop_is_the_one_reported():
    """A SIGTERM during an idle shutdown should not rewrite history."""
    daemon = daemon_mod.Daemon(FakeState(Path(".")), bridge_mod.Bridge())
    daemon.stop("first")
    daemon.stop("second")
    assert daemon.reason == "first"


def test_idle_shutdown_is_off_unless_asked_for():
    daemon = daemon_mod.Daemon(FakeState(Path(".")), bridge_mod.Bridge())
    assert daemon.idle == 0.0


def test_an_idle_daemon_exits_when_it_was_given_a_deadline(tmp_path):
    thread, _ = run_daemon(tmp_path, idle=0.05)
    thread.join(timeout=15)
    assert not thread.is_alive(), "an idle daemon with a deadline never exited"
    assert bridge_mod.active() is None or not bridge_mod.active().listening


def test_shutdown_saves_the_session_rather_than_dropping_it(tmp_path):
    """A daemon that exits without saving has silently lost the user's work."""
    state = FakeState(tmp_path)
    daemon_mod._shutdown(state)
    assert state.session.closed
    assert state.eggs.saved


def test_one_failing_shutdown_step_does_not_skip_the_rest(tmp_path):
    state = FakeState(tmp_path)

    def explode() -> None:
        raise RuntimeError("no")

    state.eggs.save = explode
    daemon_mod._shutdown(state)
    assert state.session.closed, "a failing eggs.save stopped the session closing"


def test_a_bridge_that_cannot_bind_is_a_nonzero_exit(tmp_path, monkeypatch):
    """Silently exiting 0 having served nothing is the worst possible answer
    for something a supervisor is watching."""
    monkeypatch.setattr(bridge_mod.Bridge, "serve", lambda self: ["nope"])
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    status = daemon_mod.serve(proj, build=build_for(proj), out=io.StringIO())
    assert status == 1


def test_quiet_prints_nothing_on_the_happy_path(tmp_path):
    out = io.StringIO()
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)

    def work() -> None:
        daemon_mod.serve(proj, build=build_for(proj), out=out, quiet=True, idle=0.05)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(timeout=15)
    assert out.getvalue() == ""


def _stop(thread: threading.Thread) -> None:
    active = bridge_mod.active()
    if active is not None:
        active.shutdown()
    thread.join(timeout=5)
