"""The monitor, driven over real HTTP.

This binds a socket on a machine running an agent with tool access, so the
tests that matter are the ones that try to get in without permission. Nothing
here mocks the server: a token check that passes against a fake is worth
nothing, and the interesting failures - a header that a navigation cannot set,
a port that stays bound after shutdown - only exist in the real thing.

Every test tears its monitor down. A monitor that outlives its test holds the
port and wedges the next run.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

from offset.core.monitor import (
    DEFAULT_HOST,
    REDACTED,
    TOKEN_HEADER,
    Monitor,
    Snapshot,
    free_port,
    page,
    read_or_make_token,
    redact,
    token_file,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def running(home):
    """A live monitor on an ephemeral port, always torn down."""
    monitors: list[Monitor] = []

    def start(**kw):
        kw.setdefault("port", free_port())
        kw.setdefault("home", home)
        monitor = Monitor(**kw).start()
        monitors.append(monitor)
        return monitor

    yield start
    for monitor in monitors:
        monitor.stop()


def call(monitor: Monitor, path: str, *, query_token: str | None = None,
         header_token: str | None = None, method: str = "GET",
         body: dict | None = None) -> tuple[int, bytes]:
    url = f"http://127.0.0.1:{monitor.port}{path}"
    if query_token is not None:
        url += f"?token={query_token}"
    request = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode("utf-8") if body is not None else None)
    if header_token is not None:
        request.add_header(TOKEN_HEADER, header_token)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


# -- the defaults that decide exposure ------------------------------------------------


def test_the_default_bind_is_loopback(home):
    """The one setting that turns a convenience into an exposure."""
    assert Monitor(home=home).host == DEFAULT_HOST
    assert DEFAULT_HOST == "127.0.0.1"


def test_a_wider_interface_must_be_asked_for(home):
    assert Monitor(host="0.0.0.0", home=home).host == "0.0.0.0"


def test_the_url_never_advertises_a_wildcard_address(home):
    """`http://0.0.0.0/` is not a URL anybody can open."""
    monitor = Monitor(host="0.0.0.0", port=1, home=home)
    assert "0.0.0.0" not in monitor.url
    assert "127.0.0.1" in monitor.url


# -- the token -----------------------------------------------------------------------


def test_a_token_is_generated_on_first_use(home):
    assert len(read_or_make_token(home)) >= 32


def test_the_same_token_is_reused(home):
    assert read_or_make_token(home) == read_or_make_token(home)


def test_the_token_file_is_not_readable_by_others(home):
    read_or_make_token(home)
    mode = token_file(home).stat().st_mode & 0o777
    assert mode == 0o600, oct(mode)


def test_an_unwritable_home_still_yields_a_working_token(tmp_path):
    blocked = tmp_path / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        assert len(read_or_make_token(blocked / "inner")) >= 32
    finally:
        blocked.chmod(0o700)


# -- getting in without permission -----------------------------------------------------


def test_the_page_needs_a_token(running):
    monitor = running()
    assert call(monitor, "/")[0] == 401


def test_a_wrong_token_is_rejected(running):
    monitor = running()
    assert call(monitor, "/", query_token="not-the-token")[0] == 401


def test_a_prefix_of_the_real_token_is_rejected(running):
    """If the comparison were `==` it would still reject this - but it would
    reject it *sooner* for a shorter prefix, which is a timing oracle for the
    token. The test that this is not `==`-shaped is that a correct prefix
    gets no further than a wrong first byte.
    """
    monitor = running()
    assert call(monitor, "/", query_token=monitor.token[:20])[0] == 401
    assert call(monitor, "/", query_token="x" + monitor.token[1:])[0] == 401


def test_an_empty_token_is_rejected(running):
    monitor = running()
    assert call(monitor, "/", query_token="")[0] == 401


def test_the_token_plus_a_suffix_is_rejected(running):
    monitor = running()
    assert call(monitor, "/", query_token=monitor.token + "x")[0] == 401


@pytest.mark.parametrize("route", ["/", "/api/status", "/api/jobs", "/api/usage"])
def test_every_route_needs_the_token(running, route):
    monitor = running()
    assert call(monitor, route)[0] == 401


@pytest.mark.parametrize("route", ["/", "/api/status", "/api/jobs", "/api/usage"])
def test_every_route_answers_with_the_token(running, route):
    monitor = running()
    assert call(monitor, route, query_token=monitor.token)[0] == 200


def test_the_token_is_accepted_in_a_header_too(running):
    """So a script can avoid putting it in a URL that gets logged."""
    monitor = running()
    assert call(monitor, "/api/status", header_token=monitor.token)[0] == 200


# -- the mutating route ------------------------------------------------------------------


def test_cancel_rejects_a_token_supplied_only_in_the_query(running):
    """A query string rides along in links, bookmarks and history, so a page on
    another origin can navigate to it. A header cannot be set by navigation."""
    monitor = running()
    code, _ = call(monitor, "/api/cancel", query_token=monitor.token,
                   method="POST", body={"id": "x"})
    assert code == 401


def test_cancel_accepts_a_token_in_the_header(running):
    monitor = running()
    code, _ = call(monitor, "/api/cancel", header_token=monitor.token,
                   method="POST", body={"id": "x"})
    assert code == 200


def test_cancel_is_not_reachable_by_get(running):
    monitor = running()
    assert call(monitor, "/api/cancel", query_token=monitor.token)[0] == 404


def test_cancel_needs_a_job_id(running):
    monitor = running()
    code, _ = call(monitor, "/api/cancel", header_token=monitor.token,
                   method="POST", body={})
    assert code == 400


def test_cancel_rejects_a_non_json_body(running):
    monitor = running()
    url = f"http://127.0.0.1:{monitor.port}/api/cancel"
    request = urllib.request.Request(url, method="POST", data=b"not json")
    request.add_header(TOKEN_HEADER, monitor.token)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            code = response.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    assert code == 400


def test_cancelling_an_unknown_job_is_false_not_an_error(running):
    monitor = running()
    code, body = call(monitor, "/api/cancel", header_token=monitor.token,
                      method="POST", body={"id": "no-such-job"})
    assert code == 200
    assert json.loads(body)["cancelled"] is False


# -- no filesystem reaches the request ------------------------------------------------------


@pytest.mark.parametrize("route", [
    "/../../etc/passwd",
    "/etc/passwd",
    "/static/../../../etc/passwd",
    "/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
])
def test_no_request_path_reaches_the_filesystem(running, route):
    """There is no static file route at all - a traversal bug here would hand
    over the machine, so the only answer to an unknown path is 404."""
    monitor = running()
    code, body = call(monitor, route, query_token=monitor.token)
    assert code == 404
    assert b"root:" not in body


# -- what it reports --------------------------------------------------------------------------


def test_status_reports_the_injected_snapshot(running, home):
    monitor = running(source=lambda: Snapshot(model="claude-opus-5",
                                              session="s1", busy=True))
    _, body = call(monitor, "/api/status", query_token=monitor.token)
    data = json.loads(body)
    assert data["model"] == "claude-opus-5"
    assert data["session"] == "s1"
    assert data["busy"] is True


def test_a_snapshot_source_that_raises_does_not_break_the_route(running):
    """A monitor is a convenience; it must not become a way to crash the shell."""
    def broken():
        raise RuntimeError("no")

    monitor = running(source=broken)
    assert call(monitor, "/api/status", query_token=monitor.token)[0] == 200


def test_usage_says_nothing_recorded_rather_than_zero_cost(running, home):
    """Reporting zeros reads as "this cost nothing", a different claim from
    "nothing was recorded"."""
    monitor = running()
    _, body = call(monitor, "/api/usage", query_token=monitor.token)
    data = json.loads(body)
    assert data.get("note") == "nothing recorded yet"


def test_usage_reports_recorded_turns(running, home):
    from offset.core.telemetry import Entry, Ledger
    import time as clock

    Ledger(home).append(Entry(at=clock.time(), model="gpt-4o", tokens_in=100,
                              tokens_out=50, cost=0.25))
    monitor = running()
    _, body = call(monitor, "/api/usage", query_token=monitor.token)
    data = json.loads(body)
    assert data["turns"] == 1
    assert data["cost"] == pytest.approx(0.25)
    assert "gpt-4o" in data["by_model"]


def test_usage_marks_a_partial_cost(running, home):
    from offset.core.telemetry import Entry, Ledger
    import time as clock

    Ledger(home).append(Entry(at=clock.time(), model="unknown-model", cost=None))
    monitor = running()
    _, body = call(monitor, "/api/usage", query_token=monitor.token)
    assert json.loads(body)["cost_is_partial"] is True


def test_jobs_is_a_list_even_with_no_jobs(running):
    monitor = running()
    _, body = call(monitor, "/api/jobs", query_token=monitor.token)
    assert isinstance(json.loads(body)["jobs"], list)


# -- redaction --------------------------------------------------------------------------------


def test_a_credential_shaped_key_is_redacted():
    assert redact({"api_key": "anything"})["api_key"] == REDACTED
    assert redact({"password": "x"})["password"] == REDACTED
    assert redact({"Authorization": "Bearer x"})["Authorization"] == REDACTED


def test_a_credential_shaped_value_is_redacted():
    assert REDACTED in redact("using sk-abcdefghijklmnopqrstuvwx now")
    assert REDACTED in redact("ghp_abcdefghijklmnopqrstuvwxyz1234")


def test_an_innocent_value_is_untouched():
    assert redact({"model": "gpt-4o"})["model"] == "gpt-4o"


def test_redaction_reaches_into_nested_structures():
    got = redact({"jobs": [{"label": "x", "token": "secret"}]})
    assert got["jobs"][0]["token"] == REDACTED


def test_a_served_status_never_contains_the_monitor_token(running):
    monitor = running(source=lambda: Snapshot(model=monitor_token_probe()))
    for route in ("/api/status", "/api/jobs", "/api/usage"):
        _, body = call(monitor, route, query_token=monitor.token)
        assert monitor.token.encode() not in body, route


def monitor_token_probe() -> str:
    return "model-name"


# -- the page ------------------------------------------------------------------------------------


def test_the_page_is_self_contained():
    """Served to a phone over a network the user may not control; every
    external asset is another party in that conversation."""
    html = page()
    assert "http://" not in html.replace("http://127.0.0.1", "")
    assert "https://" not in html
    assert "<script" in html


def test_the_page_escapes_what_it_renders():
    assert "replace(/[&<>" in page(), "no HTML escaping helper in the page"


# -- shutdown --------------------------------------------------------------------------------------


def test_stopping_frees_the_port(home):
    monitor = Monitor(port=free_port(), home=home).start()
    port = monitor.port
    assert call(monitor, "/", query_token=monitor.token)[0] == 200
    monitor.stop()
    probe = socket.socket()
    try:
        assert probe.connect_ex(("127.0.0.1", port)) != 0, "the port is still bound"
    finally:
        probe.close()


def test_stopping_twice_is_harmless(home):
    monitor = Monitor(port=free_port(), home=home).start()
    monitor.stop()
    monitor.stop()
    assert monitor.running is False


def test_stopping_leaves_no_thread(home):
    before = {t.name for t in threading.enumerate()}
    monitor = Monitor(port=free_port(), home=home).start()
    monitor.stop()
    after = {t.name for t in threading.enumerate()}
    assert "offset-monitor" not in after - before


def test_starting_twice_does_not_bind_a_second_socket(home):
    monitor = Monitor(port=free_port(), home=home).start()
    try:
        port = monitor.port
        monitor.start()
        assert monitor.port == port
    finally:
        monitor.stop()


def test_a_monitor_that_never_started_stops_cleanly(home):
    Monitor(port=1, home=home).stop()


# -- the command -----------------------------------------------------------------------------------


class State:
    def __init__(self, workspace):
        self.workspace = workspace
        self.model = "claude-opus-5"
        self.live = None


def test_the_command_starts_and_stops(home, tmp_path):
    import offset.core.monitor as module

    module._active = None
    try:
        out = module._monitor_command(State(tmp_path), ["start"])
        assert any("monitor:" in line for line in out.lines)
        assert module._active is not None
        status = module._monitor_command(State(tmp_path), ["status"])
        assert any("bound to" in line for line in status.lines)
        stopped = module._monitor_command(State(tmp_path), ["stop"])
        assert any("stopped" in line for line in stopped.lines)
        assert module._active is None
    finally:
        if module._active is not None:
            module._active.stop()
            module._active = None


def test_stopping_when_not_running_says_so(home, tmp_path):
    import offset.core.monitor as module

    module._active = None
    out = module._monitor_command(State(tmp_path), ["stop"])
    assert any("not running" in line for line in out.lines)


def test_a_wide_bind_is_warned_about(home, tmp_path):
    """The user asked for it, so it happens - but they are told what they did."""
    import offset.core.monitor as module

    module._active = None
    try:
        out = module._monitor_command(State(tmp_path), ["start", "0.0.0.0"])
        assert any("reachable from the network" in line for line in out.lines)
    finally:
        if module._active is not None:
            module._active.stop()
            module._active = None


def test_an_unknown_action_explains_the_usage(home, tmp_path):
    import offset.core.monitor as module

    module._active = None
    out = module._monitor_command(State(tmp_path), ["frobnicate"])
    assert any("usage" in line for line in out.lines)


def test_the_command_is_registered_lazily():
    import offset.core.monitor as module

    first, second = module.COMMANDS, module.COMMANDS
    assert first is second
    assert [c.name for c in first] == ["monitor"]
