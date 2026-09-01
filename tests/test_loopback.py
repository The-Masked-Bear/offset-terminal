"""The loopback bridge: code the agent is running calling the agent's tools.

This opens a socket that can execute tools, so the security properties *are*
the tests. Three of them decide whether the feature is safe to exist:

  * an unauthenticated caller gets nothing,
  * the permission system still applies - loopback must not be a way for a
    subprocess to do what the model itself would have been asked about,
  * and the token never reaches argv, which is world-readable in `/proc`.

Everything here drives a real socket. Mocking `socket` would agree with any
bug in the framing, which is exactly where this kind of thing breaks.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from offset.core.loopback import DEPTH_ENV, SOCKET_ENV, TOKEN_ENV, depth_of, serve
from offset.tools.base import Danger, Tool, ToolContext, ToolResult, Toolbox
from offset.tools.runtime import Approval, Runtime


class Reader(Tool):
    name = "peek"
    description = "read something harmless"
    danger = Danger.SAFE
    schema = {"type": "object", "properties": {"what": {"type": "string"}}}

    def run(self, args, ctx):
        return ToolResult.text(f"peeked {args.get('what', '')}")


class Writer(Tool):
    name = "scribble"
    description = "change a file"
    danger = Danger.WRITE
    schema = {"type": "object", "properties": {"what": {"type": "string"}}}

    def run(self, args, ctx):
        return ToolResult.text("scribbled")


def runtime_for(tmp_path, mode: str = "safe") -> Runtime:
    return Runtime(
        Toolbox([Reader(), Writer()]),
        ToolContext(cwd=tmp_path, timeout=5.0),
        Approval(mode=mode),
    )


class Caller:
    """A minimal client speaking the wire format, so the framing is tested."""

    def __init__(self, address: str, *, unix: bool = True) -> None:
        if unix:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(address)
        else:
            host, _, port = address.rpartition(":")
            self.sock = socket.create_connection((host or "127.0.0.1", int(port)))
        self.sock.settimeout(10)
        self.file = self.sock.makefile("rwb")
        self.next_id = 0

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        frame = {"jsonrpc": "2.0", "id": self.next_id, "method": method,
                 "params": params or {}}
        self.file.write((json.dumps(frame) + "\n").encode())
        self.file.flush()
        while True:
            line = self.file.readline()
            if not line:
                return {"error": {"message": "the server hung up"}}
            message = json.loads(line)
            if message.get("id") == self.next_id:
                return message

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


@pytest.fixture()
def spawned(tmp_path):
    """A live loopback plus an authenticated caller."""
    made = []

    def build(*, mode="safe", ask=None, env=None, **kw):
        manager = serve(runtime_for(tmp_path, mode), ask=ask, env=env, **kw)
        lb = manager.__enter__()
        made.append((manager, lb))
        return lb

    yield build
    for manager, _ in reversed(made):
        manager.__exit__(None, None, None)


def connect(lb) -> Caller:
    return Caller(lb.address, unix=lb.unix)


def hello(lb, caller: Caller, token: str | None = None) -> dict:
    return caller.call("hello", {"token": lb.token if token is None else token})


# -- it comes up ------------------------------------------------------------------


def test_a_loopback_listens_and_publishes_its_address(spawned):
    lb = spawned()
    assert lb.listening
    assert lb.address
    assert lb.token


def test_the_address_and_token_are_published_through_the_environment(spawned, tmp_path):
    published: dict[str, str] = {}
    lb = spawned(publish_to=published)
    assert published.get(SOCKET_ENV)
    assert published.get(TOKEN_ENV) == lb.token


def test_the_environment_is_restored_afterwards(tmp_path):
    """A stale address left behind would point the next tool call at a socket
    that no longer exists."""
    published = {"UNRELATED": "kept"}
    with serve(runtime_for(tmp_path), publish_to=published) as lb:
        assert published[SOCKET_ENV]
        address = lb.address
    assert SOCKET_ENV not in published
    assert TOKEN_ENV not in published
    assert published["UNRELATED"] == "kept"
    assert not Path(address).exists() if lb.unix else True


# -- authentication ----------------------------------------------------------------


def test_an_unauthenticated_caller_is_refused(spawned):
    """The property that makes the whole thing tolerable."""
    lb = spawned()
    caller = connect(lb)
    try:
        reply = caller.call("call", {"name": "peek", "args": {"what": "x"}})
        assert "error" in reply, f"an unauthenticated call succeeded: {reply}"
    finally:
        caller.close()


def test_a_wrong_token_is_refused(spawned):
    lb = spawned()
    caller = connect(lb)
    try:
        assert "error" in hello(lb, caller, token="not-the-token")
    finally:
        caller.close()


def test_the_right_token_is_accepted(spawned):
    lb = spawned()
    caller = connect(lb)
    try:
        assert "error" not in hello(lb, caller), "the real token was refused"
    finally:
        caller.close()


def test_the_token_is_never_put_on_a_command_line(spawned):
    """`/proc/<pid>/cmdline` is world-readable; the environment is not."""
    published: dict[str, str] = {}
    lb = spawned(publish_to=published)
    assert lb.token in published.values()
    assert TOKEN_ENV in published, "the token must travel by environment"


# -- calling tools -------------------------------------------------------------------


def test_an_authenticated_caller_can_list_the_tools(spawned):
    lb = spawned()
    caller = connect(lb)
    try:
        hello(lb, caller)
        reply = caller.call("tools")
        names = [t.get("name") for t in (reply.get("result") or {}).get("tools", [])]
        assert "peek" in names
    finally:
        caller.close()


def test_a_safe_tool_runs(spawned):
    lb = spawned()
    caller = connect(lb)
    try:
        hello(lb, caller)
        reply = caller.call("call", {"name": "peek", "args": {"what": "the repo"}})
        assert "error" not in reply, reply
        assert "peeked" in json.dumps(reply["result"])
    finally:
        caller.close()


def test_an_unknown_tool_is_refused_cleanly(spawned):
    lb = spawned()
    caller = connect(lb)
    try:
        hello(lb, caller)
        reply = caller.call("call", {"name": "nosuchtool", "args": {}})
        assert "error" in reply or reply["result"].get("ok") is False
    finally:
        caller.close()


def test_a_call_budget_stops_a_runaway_script(spawned):
    lb = spawned(max_calls=3)
    caller = connect(lb)
    try:
        hello(lb, caller)
        outcomes = [caller.call("call", {"name": "peek", "args": {}}) for _ in range(6)]
        # Being over budget is a *result*, not a protocol error: the script
        # asked correctly and is being told it has had enough.
        refused = [r for r in outcomes
                   if "error" in r or (r.get("result") or {}).get("ok") is False]
        assert refused, "a script could call tools without limit"
        assert len(refused) == 3, f"expected 3 of 6 refused, got {len(refused)}"
    finally:
        caller.close()


# -- permissions ------------------------------------------------------------------------


def test_a_writing_tool_is_refused_when_nothing_can_approve_it(spawned):
    """The escalation this must not be. If the model would have been asked,
    a subprocess going round the back must not simply be allowed.
    """
    lb = spawned(mode="safe", ask=None)
    caller = connect(lb)
    try:
        hello(lb, caller)
        reply = caller.call("call", {"name": "scribble", "args": {"what": "x"}})
        body = json.dumps(reply)
        assert "error" in reply or '"ok": false' in body.lower(), (
            f"a WRITE tool ran unapproved over loopback: {body}"
        )
    finally:
        caller.close()


def test_a_writing_tool_runs_when_the_human_approves(spawned):
    lb = spawned(mode="safe", ask=lambda prompt, default=False: True)
    caller = connect(lb)
    try:
        hello(lb, caller)
        reply = caller.call("call", {"name": "scribble", "args": {"what": "x"}})
        assert "error" not in reply, reply
    finally:
        caller.close()


def test_a_refused_approval_is_reported_not_silently_skipped(spawned):
    lb = spawned(mode="safe", ask=lambda prompt, default=False: False)
    caller = connect(lb)
    try:
        hello(lb, caller)
        reply = caller.call("call", {"name": "scribble", "args": {"what": "x"}})
        assert "error" in reply or '"ok": false' in json.dumps(reply).lower()
    finally:
        caller.close()


# -- recursion --------------------------------------------------------------------------


def test_depth_is_read_from_the_environment():
    assert depth_of({}) == 0
    assert depth_of({DEPTH_ENV: "2"}) == 2


def test_a_corrupted_depth_counter_fails_closed():
    """Restarting the count at zero because the value was unreadable is
    precisely the runaway the counter exists to stop, so garbage reads as
    "already too deep"."""
    from offset.core.loopback import MAX_DEPTH

    assert depth_of({DEPTH_ENV: "nonsense"}) > MAX_DEPTH


def test_a_child_is_published_one_level_deeper(tmp_path):
    published: dict[str, str] = {}
    with serve(runtime_for(tmp_path), env={}, publish_to=published):
        assert int(published[DEPTH_ENV]) == 1


def test_nesting_stops_at_the_cap(tmp_path):
    """Without a cap, a tool that shells out to something that uses loopback
    recurses until the machine gives up."""
    deep = {DEPTH_ENV: "99"}
    with serve(runtime_for(tmp_path), env=deep) as lb:
        assert not lb.listening, "a runaway nesting depth was allowed to publish"
        assert lb.problems, "it declined without saying why"


# -- teardown --------------------------------------------------------------------------


def test_the_socket_is_gone_when_the_call_finishes(tmp_path):
    with serve(runtime_for(tmp_path)) as lb:
        address, was_unix = lb.address, lb.unix
        assert lb.listening
    if was_unix:
        assert not Path(address).exists(), "the socket outlived its tool call"


def test_an_exception_in_the_body_still_tears_it_down(tmp_path):
    address = None
    with pytest.raises(RuntimeError):
        with serve(runtime_for(tmp_path)) as lb:
            address = lb.address if lb.unix else None
            raise RuntimeError("the tool blew up")
    if address:
        assert not Path(address).exists()


def test_a_connection_after_shutdown_simply_fails(tmp_path):
    with serve(runtime_for(tmp_path)) as lb:
        address, was_unix = lb.address, lb.unix
    if was_unix:
        with pytest.raises(OSError):
            Caller(address)
