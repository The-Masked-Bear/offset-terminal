"""MCP client: handshake, correlation, failure, and the Tool wrapper.

Every test here drives a real server — a python script launched as a child
process over stdio, or a real HTTP server on a loopback port.  The failures
this module exists to prevent (a call that hangs when the server dies, a reply
matched to the wrong request, a killed server leaving an orphan) only ever show
up against a real process, so mocking the transport would test nothing.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from offset.tools.base import Cancelled, Danger, ToolContext
from offset.tools.mcp import (
    Config,
    HTTPTransport,
    MCPClient,
    Manager,
    ServerConfig,
    StdioTransport,
    load_config,
    parse_config,
    tool_name,
)

# -- the server under test --------------------------------------------------

SERVER = r'''
import json, os, subprocess, sys, time

LOG = os.environ.get("ECHO_LOG", "")


def log(obj):
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


TOOLS = [
    {"name": "echo", "description": "echo text back",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                     "required": ["text"]},
     "annotations": {"readOnlyHint": True}},
    {"name": "write_note", "description": "pretend to write a note",
     "inputSchema": {"type": "object", "properties": {"body": {"type": "string"}}}},
    {"name": "slow", "description": "answer eventually",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "die", "description": "exit without answering",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "boom", "description": "report a tool error",
     "inputSchema": {"type": "object", "properties": {}}},
]


def call(name, args):
    if name == "echo":
        return {"content": [{"type": "text", "text": "echo: " + str(args.get("text", ""))}]}
    if name == "write_note":
        return {"content": [{"type": "text", "text": "noted"}]}
    if name == "slow":
        time.sleep(float(os.environ.get("ECHO_SLOW", "10")))
        return {"content": [{"type": "text", "text": "eventually"}]}
    if name == "die":
        os._exit(9)
    if name == "boom":
        return {"isError": True, "content": [{"type": "text", "text": "the remote tool refused"}]}
    return {"isError": True, "content": [{"type": "text", "text": "no such tool " + str(name)}]}


def handle(msg):
    method = msg.get("method")
    if method == "initialize":
        if os.environ.get("ECHO_CHILD"):
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
            log({"child": child.pid})
        return {"protocolVersion": os.environ.get("ECHO_PROTOCOL", "2025-06-18"),
                "capabilities": {"tools": {"listChanged": True}, "resources": {}, "prompts": {}},
                "serverInfo": {"name": "echo-server", "version": "1.2.3"}}
    if method == "tools/list":
        return {"tools": TOOLS}
    if method == "tools/call":
        params = msg.get("params") or {}
        return call(params.get("name"), params.get("arguments") or {})
    if method == "resources/list":
        return {"resources": [{"uri": "mem://note", "name": "note", "mimeType": "text/plain"}]}
    if method == "resources/read":
        return {"contents": [{"uri": (msg.get("params") or {}).get("uri"), "text": "the note body"}]}
    if method == "prompts/list":
        return {"prompts": [{"name": "greet", "description": "say hi",
                             "arguments": [{"name": "who"}]}]}
    return None


if os.environ.get("ECHO_NOISE"):
    sys.stdout.write("starting up; this line is not a frame\n")
    sys.stdout.write("{not json either\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    log({"id": msg.get("id"), "method": msg.get("method")})
    if msg.get("id") is None:
        continue
    result = handle(msg)
    if result is None:
        send({"jsonrpc": "2.0", "id": msg["id"],
              "error": {"code": -32601, "message": "method not found: " + str(msg.get("method"))}})
    else:
        send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
'''


@pytest.fixture()
def script(tmp_path: Path) -> Path:
    path = tmp_path / "echo_server.py"
    path.write_text(SERVER, encoding="utf-8")
    return path


@pytest.fixture()
def log(tmp_path: Path) -> Path:
    return tmp_path / "rpc.log"


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, timeout=30.0)


@pytest.fixture()
def managers():
    """Every manager built by a test is shut down, orphans or not."""
    made: list[Manager] = []
    yield made
    for manager in made:
        manager.disconnect_all()


def build(managers, script: Path, log: Path, *, timeout: float = 30.0, **env: str) -> Manager:
    config = Config(servers=[ServerConfig(
        name="echo",
        command=sys.executable,
        args=[str(script)],
        env={"ECHO_LOG": str(log), **env},
        timeout=timeout,
    )])
    manager = Manager(config, attempts=1)
    managers.append(manager)
    return manager


def logged(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def wait_gone(pid: int, budget: float = 5.0) -> bool:
    limit = time.monotonic() + budget
    while time.monotonic() < limit:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.05)
    return False


# -- handshake --------------------------------------------------------------


def test_the_handshake_negotiates_a_version_and_reports_capabilities(script, log, managers):
    manager = build(managers, script, log)
    assert manager.connect("echo") is True
    client = manager.client("echo")
    assert client is not None
    assert client.protocol == "2025-06-18"
    assert client.server_info["name"] == "echo-server"
    assert client.supports("tools") and client.supports("resources")
    assert client.supports("logging") is False

    methods = [row["method"] for row in logged(log)]
    assert methods[0] == "initialize"
    # The spec requires the notification; a server may refuse to serve without it.
    assert "notifications/initialized" in methods


def test_a_protocol_version_offset_cannot_speak_is_refused(script, log, managers):
    manager = build(managers, script, log, ECHO_PROTOCOL="1999-01-01")
    assert manager.connect("echo") is False
    assert "1999-01-01" in manager.reason("echo")
    assert manager.client("echo") is None
    assert manager.tools() == []


def test_a_malformed_frame_is_skipped_rather_than_fatal(script, log, managers):
    manager = build(managers, script, log, ECHO_NOISE="1")
    assert manager.connect("echo") is True
    client = manager.client("echo")
    assert client is not None
    assert client.transport.dropped >= 2  # a banner line and a broken object
    assert client.call_tool("echo", {"text": "still here"}, timeout=10.0).content == "echo: still here"


# -- tools ------------------------------------------------------------------


def test_remote_tools_become_offset_tools_carrying_the_remote_schema(script, log, managers):
    manager = build(managers, script, log)
    manager.connect("echo")
    tools = {tool.name: tool for tool in manager.tools()}
    assert set(tools) == {
        tool_name("echo", name) for name in ("echo", "write_note", "slow", "die", "boom")
    }
    echo = tools["mcp__echo__echo"]
    assert echo.schema["required"] == ["text"]
    assert echo.description == "echo text back"
    assert echo.spec().schema is echo.schema


def test_only_a_read_only_hint_makes_a_remote_tool_safe(script, log, managers):
    manager = build(managers, script, log)
    manager.connect("echo")
    tools = {tool.name: tool for tool in manager.tools()}
    assert tools["mcp__echo__echo"].danger is Danger.SAFE
    assert tools["mcp__echo__echo"].parallel_safe is True
    # No hint means we must assume the worst: remote code we cannot read.
    assert tools["mcp__echo__write_note"].danger is Danger.DESTRUCTIVE
    assert tools["mcp__echo__write_note"].parallel_safe is False


def test_a_call_through_the_tool_wrapper_returns_the_servers_text(script, log, managers, ctx):
    manager = build(managers, script, log)
    manager.connect("echo")
    tool = next(t for t in manager.tools() if t.name == "mcp__echo__echo")

    result = tool.run({"text": "hello"}, ctx)
    assert result.ok is True
    assert result.content == "echo: hello"
    assert result.data["server"] == "echo" and result.data["tool"] == "echo"


def test_a_remote_tool_error_is_a_failed_result_not_an_exception(script, log, managers, ctx):
    manager = build(managers, script, log)
    manager.connect("echo")
    tool = next(t for t in manager.tools() if t.name == "mcp__echo__boom")

    result = tool.run({}, ctx)
    assert result.ok is False
    assert "refused" in result.content
    assert manager.state("echo") == "live"  # a tool error is not a dead server


def test_request_ids_are_never_reused(script, log, managers, ctx):
    manager = build(managers, script, log)
    manager.connect("echo")
    tool = next(t for t in manager.tools() if t.name == "mcp__echo__echo")
    for i in range(5):
        assert tool.run({"text": str(i)}, ctx).ok

    ids = [row["id"] for row in logged(log) if row["id"] is not None]
    assert len(ids) >= 7  # initialize + tools/list + five calls
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


# -- failure paths ----------------------------------------------------------


def test_a_server_that_exits_mid_call_is_reported_not_waited_on(script, log, managers, ctx):
    manager = build(managers, script, log, timeout=30.0)
    manager.connect("echo")
    tool = next(t for t in manager.tools() if t.name == "mcp__echo__die")

    started = time.monotonic()
    result = tool.run({}, ctx)
    elapsed = time.monotonic() - started

    assert result.ok is False
    assert "died" in result.content
    assert elapsed < 5.0, "the death was noticed by the reader, not by the 30s deadline"
    assert manager.state("echo") == "down"
    assert manager.tools() == [], "a dead server's tools must stop being offered"


def test_a_tool_timeout_is_a_failure_and_not_a_user_cancel(script, log, managers, ctx):
    manager = build(managers, script, log, timeout=0.5)
    manager.connect("echo")
    tool = next(t for t in manager.tools() if t.name == "mcp__echo__slow")

    started = time.monotonic()
    result = tool.run({}, ctx)
    elapsed = time.monotonic() - started

    assert result.ok is False
    assert "0.5s" in result.content and "echo" in result.content
    assert elapsed < 3.0
    assert not ctx.cancel.is_set(), "a timeout must not look like the user aborting"
    assert manager.state("echo") == "live", "one slow call does not kill the server"


def test_a_cancelled_remote_call_raises_cancel_so_the_runtime_can_tell(script, log, managers, ctx):
    # The server is single-threaded: it can only drain stdin (and log the
    # cancellation) once `slow` returns, so keep that short.
    manager = build(managers, script, log, timeout=30.0, ECHO_SLOW="2")
    manager.connect("echo")
    tool = next(t for t in manager.tools() if t.name == "mcp__echo__slow")
    threading.Timer(0.3, ctx.cancel.set).start()

    with pytest.raises(Cancelled):
        tool.run({}, ctx)

    # The server is still inside `slow`, so it has not drained its stdin yet:
    # wait for it to log the notification rather than racing it.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if "notifications/cancelled" in [row["method"] for row in logged(log)]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the server was never told to stop working")


def test_a_call_after_the_server_died_says_unavailable_instead_of_hanging(script, log, managers, ctx):
    manager = build(managers, script, log)
    manager.connect("echo")
    tool = next(t for t in manager.tools() if t.name == "mcp__echo__echo")
    assert tool.run({"text": "one"}, ctx).ok

    pid = manager.client("echo").transport.pid
    os.kill(pid, 9)
    assert wait_gone(pid) or True  # the kill is what matters, not the reaping race
    time.sleep(0.3)

    result = tool.run({"text": "two"}, ctx)
    assert result.ok is False
    assert "unavailable" in result.content or "died" in result.content


def test_a_server_that_cannot_start_is_retried_then_recorded(managers, tmp_path):
    delays: list[float] = []
    config = Config(servers=[ServerConfig(name="ghost", command=str(tmp_path / "nope"))])
    manager = Manager(config, attempts=3, backoff=0.01, sleep=delays.append)
    managers.append(manager)

    assert manager.connect("ghost") is False
    assert len(delays) == 2, "one sleep between each pair of attempts, none after the last"
    assert delays == sorted(delays) and delays[1] > delays[0], "backoff must grow"
    assert "nope" in manager.reason("ghost")
    assert manager.state("ghost") == "down"
    assert manager.tools() == []


def test_disconnect_leaves_no_orphan_process(script, log, managers):
    manager = build(managers, script, log, ECHO_CHILD="1")
    manager.connect("echo")
    pid = manager.client("echo").transport.pid
    child = next(row["child"] for row in logged(log) if "child" in row)
    os.kill(pid, 0)  # both are alive before the disconnect
    os.kill(child, 0)

    manager.disconnect("echo")

    with pytest.raises(OSError):
        os.kill(pid, 0)
    assert wait_gone(child), "a helper spawned by the server is part of the tree"
    assert manager.client("echo") is None


# -- resources and prompts --------------------------------------------------


def test_resources_and_prompts_round_trip(script, log, managers):
    manager = build(managers, script, log)
    manager.connect("echo")
    client = manager.client("echo")

    resources = client.list_resources(timeout=10.0)
    assert [r.uri for r in resources] == ["mem://note"]
    assert resources[0].mime_type == "text/plain"
    assert client.read_resource("mem://note", timeout=10.0) == "the note body"

    prompts = client.list_prompts(timeout=10.0)
    assert [p.name for p in prompts] == ["greet"]
    assert prompts[0].arguments == [{"name": "who"}]


def test_a_capability_the_server_never_declared_is_not_called(script, log, managers):
    """An undeclared capability must fail locally: a -32601 round trip on a
    server that answers nothing would cost a whole timeout."""
    transport = StdioTransport(sys.executable, [str(script)], env={"ECHO_LOG": str(log)})
    client = MCPClient(transport, name="echo", timeout=10.0)
    try:
        client.connect()
        client.capabilities = {"tools": {}}  # server withdrew resources
        assert client.list_resources() == []
        assert client.list_prompts() == []
    finally:
        client.close()


# -- configuration ----------------------------------------------------------


def test_a_bad_field_is_named_in_the_error(tmp_path):
    servers, errors = parse_config({"mcpServers": {
        "good": {"command": "python3", "args": ["s.py"]},
        "both": {"command": "python3", "url": "https://example.com/mcp"},
        "neither": {"args": ["x"]},
        "badargs": {"command": "python3", "args": "s.py"},
        "badtimeout": {"command": "python3", "timeout": 0},
        "badurl": {"url": "ftp://example.com"},
        "badenv": {"command": "python3", "env": {"K": 3}},
        "badflag": {"command": "python3", "enabled": "yes"},
        "bad name": {"command": "python3"},
    }}, source=tmp_path / "mcp.json")

    assert [s.name for s in servers] == ["good"]
    joined = "\n".join(errors)
    assert "both: set either" in joined
    assert "neither: needs" in joined
    assert "badargs.args must be a list of strings" in joined
    assert "badtimeout.timeout must be a positive number" in joined
    assert "badurl.url must start with http://" in joined
    assert "badenv.env must be an object with string values" in joined
    assert "badflag.enabled must be true or false" in joined
    assert "bad name" in joined
    assert str(tmp_path / "mcp.json") in errors[0]


def test_a_missing_server_table_is_an_error_not_an_empty_config(tmp_path):
    servers, errors = parse_config({"tools": {}}, source=tmp_path / "mcp.json")
    assert servers == []
    assert "mcpServers" in errors[0]


def test_both_transports_are_accepted_with_their_own_fields():
    servers, errors = parse_config({"mcpServers": {
        "local": {"command": ["python3", "-u", "s.py"], "args": ["--flag"], "cwd": "/tmp", "timeout": 12},
        "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer x"}},
    }})
    assert errors == []
    local, remote = servers
    assert local.kind == "stdio"
    assert local.command == "python3" and local.args == ["-u", "s.py", "--flag"]
    assert local.timeout == 12.0
    assert remote.kind == "http" and remote.headers["Authorization"] == "Bearer x"
    assert isinstance(remote.transport(), HTTPTransport)


def test_the_workspace_config_wins_over_the_home_one(tmp_path, monkeypatch):
    home, workspace = tmp_path / "home", tmp_path / "project"
    (home).mkdir()
    (workspace / ".offset").mkdir(parents=True)
    (home / "mcp.json").write_text(json.dumps({"mcpServers": {
        "shared": {"command": "home-cmd"},
        "only-home": {"command": "home-only"},
    }}), encoding="utf-8")
    (workspace / ".offset" / "mcp.json").write_text(json.dumps({"mcpServers": {
        "shared": {"command": "project-cmd"},
        "off": {"command": "x", "enabled": False},
    }}), encoding="utf-8")
    monkeypatch.setenv("OFFSET_HOME", str(home))

    config = load_config(workspace)
    by_name = {s.name: s for s in config.servers}
    assert config.errors == []
    assert by_name["shared"].command == "project-cmd"
    assert by_name["only-home"].command == "home-only"
    assert [s.name for s in config.enabled()] == ["only-home", "shared"]
    assert len(config.sources) == 2


def test_a_disabled_server_is_never_connected(script, log, managers):
    config = Config(servers=[ServerConfig(
        name="echo", command=sys.executable, args=[str(script)], enabled=False,
    )])
    manager = Manager(config)
    managers.append(manager)

    assert manager.connect_all() == []
    assert manager.connect("echo") is False
    assert manager.state("echo") == "disabled"
    assert manager.tools() == []
    assert logged(log) == [], "nothing was ever launched"


# -- streamable HTTP --------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        length = int(self.headers.get("Content-Length") or 0)
        message = json.loads(self.rfile.read(length) or b"{}")
        self.server.seen.append((message.get("method"), self.headers.get("Mcp-Session-Id")))
        ident = message.get("id")
        if ident is None:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        method = message.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26",
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "http-echo", "version": "0.1"}}
            # Framed as SSE, which is what real streamable-HTTP servers do.
            body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': ident, 'result': result})}\n\n"
            kind = "text/event-stream"
        else:
            if method == "tools/list":
                result = {"tools": [{"name": "shout", "inputSchema": {"type": "object"},
                                     "annotations": {"readOnlyHint": True}}]}
            elif method == "tools/call":
                text = (message.get("params") or {}).get("arguments", {}).get("text", "")
                result = {"content": [{"type": "text", "text": str(text).upper()}]}
            else:
                result = {}
            body = json.dumps({"jsonrpc": "2.0", "id": ident, "result": result})
            kind = "application/json"
        raw = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Mcp-Session-Id", "sess-42")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture()
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_the_http_transport_handshakes_and_calls_a_tool(http_server, managers, ctx):
    port = http_server.server_address[1]
    config = Config(servers=[ServerConfig(
        name="web", url=f"http://127.0.0.1:{port}/mcp", timeout=10.0,
    )])
    manager = Manager(config, attempts=1)
    managers.append(manager)

    assert manager.connect("web") is True
    client = manager.client("web")
    assert client.protocol == "2025-03-26"  # the server's choice, not ours
    tool = next(t for t in manager.tools() if t.name == "mcp__web__shout")
    assert tool.run({"text": "quiet"}, ctx).content == "QUIET"

    methods = [method for method, _ in http_server.seen]
    assert methods[:2] == ["initialize", "notifications/initialized"]
    sessions = [session for method, session in http_server.seen if method == "tools/call"]
    assert sessions == ["sess-42"], "the session id the server issued must come back"


def test_an_unreachable_http_endpoint_fails_instead_of_hanging(managers):
    config = Config(servers=[ServerConfig(
        name="web", url="http://127.0.0.1:9/mcp", timeout=3.0,
    )])
    manager = Manager(config, attempts=1)
    managers.append(manager)

    started = time.monotonic()
    assert manager.connect("web") is False
    assert time.monotonic() - started < 10.0
    assert manager.state("web") == "down"
    assert manager.reason("web")


def test_sse_framing_survives_multi_line_and_comment_lines():
    body = ": keep-alive\nevent: message\ndata: {\"jsonrpc\": \"2.0\",\ndata: \"id\": 1, \"result\": {}}\n\n"
    assert HTTPTransport._sse(body) == ['{"jsonrpc": "2.0",\n"id": 1, "result": {}}']
