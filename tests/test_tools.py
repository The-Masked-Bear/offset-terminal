"""Tool contract, runtime policy, built-ins, and user-authored tools.

The load-bearing claims under test: a bad call comes back as a message the
model can fix, a hung tool is abandoned rather than waited on, a batch is
never half-parallel, and a custom tool written in any language actually runs.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time

import pytest

from offset.providers.base import ToolCall
from offset.tools.base import Danger, Tool, ToolContext, ToolResult, Toolbox, validate
from offset.tools.builtin import Bash, builtin_tools
from offset.tools.custom import ExternalTool, discover, load_manifest
from offset.tools.runtime import Approval, Runtime


@pytest.fixture()
def ctx(tmp_path):
    return ToolContext(cwd=tmp_path, timeout=10.0)


@pytest.fixture()
def box():
    return Toolbox(builtin_tools())


def call(name: str, **args) -> ToolCall:
    return ToolCall(id="c1", name=name, args=args)


# -- validation -------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "count": {"type": "integer", "minimum": 1, "maximum": 10},
        "mode": {"type": "string", "enum": ["fast", "slow"]},
        "flag": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def test_validation_accepts_a_good_call():
    assert validate({"path": "a.py", "count": 3, "mode": "fast", "tags": ["x"]}, SCHEMA) == []


def test_validation_reports_every_problem_at_once():
    problems = validate({"count": 99, "mode": "sideways", "tags": ["ok", 5]}, SCHEMA)
    assert any("missing required" in p for p in problems)
    assert any("<= 10" in p for p in problems)
    assert any("one of" in p for p in problems)
    assert any("tags[1]" in p for p in problems)


def test_booleans_are_not_numbers():
    """`True` passing as an integer is a classic silent-corruption bug."""
    assert validate({"path": "a", "count": True}, SCHEMA)


def test_unexpected_arguments_rejected_only_when_closed():
    assert validate({"path": "a", "extra": 1}, SCHEMA)
    assert validate({"path": "a", "extra": 1}, {**SCHEMA, "additionalProperties": True}) == []


# -- approval ---------------------------------------------------------------


class Harmless(Tool):
    name = "harmless"
    danger = Danger.SAFE
    schema = {"type": "object", "properties": {}}

    def run(self, args, ctx):
        return ToolResult.text("fine")


class Risky(Tool):
    name = "risky"
    danger = Danger.DESTRUCTIVE
    schema = {"type": "object", "properties": {}}

    def run(self, args, ctx):
        return ToolResult.text("boom")


def test_modes_set_the_approval_threshold(ctx):
    box = Toolbox([Harmless(), Risky()])
    safe = Runtime(box, ctx, Approval(mode="safe"))
    yolo = Runtime(box, ctx, Approval(mode="yolo"))
    assert safe.execute(call("harmless")).result.ok
    assert not safe.execute(call("risky")).result.ok
    assert yolo.execute(call("risky")).result.ok


def test_a_declined_call_is_reported_not_raised(ctx):
    asked = []
    approval = Approval(mode="safe", ask=lambda tool, args: asked.append(tool.name) or False)
    got = Runtime(Toolbox([Risky()]), ctx, approval).execute(call("risky"))
    assert asked == ["risky"]
    assert not got.approved and "declined" in got.result.error


def test_remembering_an_approval_stops_the_asking(ctx):
    asks = []
    approval = Approval(mode="safe", ask=lambda t, a: asks.append(t.name) or True)
    runtime = Runtime(Toolbox([Risky()]), ctx, approval)
    runtime.execute(call("risky"))
    approval.remember("risky")
    runtime.execute(call("risky"))
    assert asks == ["risky"], "a remembered tool must not be asked again"


def test_denial_sticks(ctx):
    approval = Approval(mode="yolo")
    approval.deny("risky")
    got = Runtime(Toolbox([Risky()]), ctx, approval).execute(call("risky"))
    assert not got.result.ok and "denied earlier" in got.result.error


def test_without_an_approver_a_dangerous_call_fails_closed(ctx):
    got = Runtime(Toolbox([Risky()]), ctx, Approval(mode="safe", ask=None)).execute(call("risky"))
    assert not got.result.ok and "no approver" in got.result.error


# -- dispatch ---------------------------------------------------------------


def test_unknown_tool_lists_what_exists(ctx, box):
    got = Runtime(box, ctx).execute(call("teleport"))
    assert not got.result.ok
    assert "no tool named" in got.result.error and "read" in got.result.error


def test_invalid_arguments_come_back_as_a_fixable_message(ctx, box):
    got = Runtime(box, ctx).execute(ToolCall("c", "read", {}))
    assert not got.result.ok and "missing required argument 'path'" in got.result.error


def test_unparseable_arguments_are_rejected_with_instructions(ctx, box):
    broken = ToolCall("c", "read", {}, raw='{"path": ')
    got = Runtime(box, ctx).execute(broken)
    assert not got.result.ok and "valid JSON" in got.result.error


def test_a_crashing_tool_does_not_kill_the_turn(ctx):
    class Exploding(Tool):
        name = "explode"
        schema = {"type": "object", "properties": {}}

        def run(self, args, ctx):
            raise RuntimeError("kaboom")

    got = Runtime(Toolbox([Exploding()]), ctx).execute(call("explode"))
    assert not got.result.ok and "RuntimeError: kaboom" in got.result.error


def test_a_timeout_does_not_set_the_user_abort(tmp_path):
    """Per-call deadlines and turn-level aborts are separate signals."""

    class Hang(Tool):
        name = "hang"
        schema = {"type": "object", "properties": {}}

        def run(self, args, ctx):
            for _ in range(300):
                ctx.check()
                time.sleep(0.01)
            return ToolResult.text("never")

    class Quick(Tool):
        name = "quick"
        schema = {"type": "object", "properties": {}}

        def run(self, args, ctx):
            return ToolResult.text("instant")

    ctx = ToolContext(cwd=tmp_path, timeout=0.25)
    runtime = Runtime(Toolbox([Hang(), Quick()]), ctx)
    assert "budget" in runtime.execute(call("hang")).result.error
    assert not runtime.aborted, "a timeout is not an abort"
    assert runtime.execute(call("quick")).result.content == "instant", "the runtime must stay usable"


def test_a_hung_tool_is_abandoned(tmp_path):
    class Hang(Tool):
        name = "hang"
        schema = {"type": "object", "properties": {}}

        def run(self, args, ctx):
            for _ in range(500):
                ctx.check()
                time.sleep(0.01)
            return ToolResult.text("never")

    ctx = ToolContext(cwd=tmp_path, timeout=0.2)
    started = time.monotonic()
    got = Runtime(Toolbox([Hang()]), ctx).execute(call("hang"))
    assert not got.result.ok and "budget" in got.result.error
    assert time.monotonic() - started < 3.0


def test_cancellation_reaches_a_running_tool(tmp_path):
    entered = threading.Event()

    class Waiter(Tool):
        name = "wait"
        schema = {"type": "object", "properties": {}}

        def run(self, args, ctx):
            entered.set()
            for _ in range(500):
                ctx.check()
                time.sleep(0.01)
            return ToolResult.text("finished")

    ctx = ToolContext(cwd=tmp_path, timeout=5.0)
    runtime = Runtime(Toolbox([Waiter()]), ctx)
    threading.Timer(0.15, runtime.cancel).start()
    got = runtime.execute(call("wait"))
    assert entered.is_set()
    assert not got.result.ok and "cancel" in got.result.error.lower()


# -- batching ---------------------------------------------------------------


def test_batches_run_in_parallel_only_when_every_tool_agrees(ctx):
    class Slow(Tool):
        name = "slow"
        schema = {"type": "object", "properties": {}}

        def run(self, args, ctx):
            time.sleep(0.15)
            return ToolResult.text("ok")

    class SlowSerial(Slow):
        name = "serial"
        parallel_safe = False

    runtime = Runtime(Toolbox([Slow(), SlowSerial()]), ctx)
    calls = [call("slow"), call("slow"), call("slow")]
    started = time.monotonic()
    assert all(i.result.ok for i in runtime.execute_all(calls))
    parallel = time.monotonic() - started

    started = time.monotonic()
    runtime.execute_all([call("slow"), call("serial"), call("slow")])
    serial = time.monotonic() - started

    assert parallel < 0.35, "safe tools should have overlapped"
    assert serial > 0.4, "one unsafe tool must serialise the whole batch"


def test_batch_results_keep_request_order(ctx):
    class Echo(Tool):
        name = "echo"
        schema = {"type": "object", "properties": {"i": {"type": "integer"}}}

        def run(self, args, ctx):
            time.sleep(0.05 * (3 - args["i"]))  # finish in reverse
            return ToolResult.text(str(args["i"]))

    runtime = Runtime(Toolbox([Echo()]), ctx)
    got = runtime.execute_all([ToolCall(f"c{i}", "echo", {"i": i}) for i in range(3)])
    assert [g.result.content for g in got] == ["0", "1", "2"]


# -- built-ins --------------------------------------------------------------


def test_read_write_edit_round_trip(ctx, box):
    runtime = Runtime(box, ctx, Approval(mode="yolo"))
    runtime.execute(call("write", path="a.py", content="one\ntwo\nthree\n"))
    got = runtime.execute(call("read", path="a.py"))
    assert got.result.content == "1:one\n2:two\n3:three"

    assert runtime.execute(call("edit", path="a.py", old="two", new="2")).result.ok
    assert (ctx.cwd / "a.py").read_text() == "one\n2\nthree\n"


def test_edit_refuses_an_ambiguous_match(ctx, box):
    runtime = Runtime(box, ctx, Approval(mode="yolo"))
    runtime.execute(call("write", path="b.py", content="x\nx\n"))
    got = runtime.execute(call("edit", path="b.py", old="x", new="y"))
    assert not got.result.ok and "appears 2 times" in got.result.error
    assert runtime.execute(call("edit", path="b.py", old="x", new="y", all=True)).result.ok


def test_read_range(ctx, box):
    runtime = Runtime(box, ctx, Approval(mode="yolo"))
    runtime.execute(call("write", path="c.txt", content="\n".join(str(i) for i in range(1, 21))))
    got = runtime.execute(call("read", path="c.txt", offset=5, limit=3))
    assert got.result.content == "5:5\n6:6\n7:7"


def test_paths_cannot_escape_the_workspace(ctx, box):
    got = Runtime(box, ctx, Approval(mode="yolo")).execute(call("read", path="../../../etc/passwd"))
    assert not got.result.ok and "escapes the workspace" in got.result.error


def test_glob_and_grep(ctx, box):
    runtime = Runtime(box, ctx, Approval(mode="yolo"))
    runtime.execute(call("write", path="pkg/mod.py", content="def target():\n    pass\n"))
    runtime.execute(call("write", path="pkg/other.txt", content="target elsewhere\n"))
    assert "pkg/mod.py" in runtime.execute(call("glob", pattern="*.py")).result.content
    hits = runtime.execute(call("grep", pattern=r"def target")).result.content
    assert "pkg/mod.py:1:" in hits
    assert runtime.execute(call("grep", pattern="def target", glob="*.txt")).result.content == "(no matches)"


def test_grep_reports_a_bad_regex(ctx, box):
    got = Runtime(box, ctx).execute(call("grep", pattern="(unclosed"))
    assert not got.result.ok and "bad regular expression" in got.result.error


def test_bash_captures_output_and_exit_code(ctx, box):
    runtime = Runtime(box, ctx, Approval(mode="yolo"))
    got = runtime.execute(call("bash", command="echo hello && exit 3"))
    assert got.result.content == "hello"
    assert got.result.data["exit"] == 3 and not got.result.ok


def test_bash_kills_the_whole_process_group(ctx):
    runtime = Runtime(Toolbox([Bash()]), ctx, Approval(mode="yolo"))
    marker = ctx.cwd / "child-survived"
    started = time.monotonic()
    got = runtime.execute(call(
        "bash",
        command=f"(sleep 5; touch {marker}) & sleep 5",
        timeout=1,
    ))
    assert not got.result.ok and "timed out" in got.result.error
    assert time.monotonic() - started < 4.0
    time.sleep(0.4)
    assert not marker.exists(), "a background child outlived the process group kill"


def test_every_builtin_is_registered_and_specced(box):
    assert set(box.names()) == {"read", "write", "edit", "list", "glob", "grep", "bash", "fetch"}
    for spec in box.specs():
        assert spec.description and spec.schema.get("type") == "object"


def test_duplicate_registration_is_refused(box):
    with pytest.raises(ValueError):
        box.register(Bash())
    box.register(Bash(), replace=True)  # explicit override is allowed


# -- custom tools -----------------------------------------------------------


def test_python_plugin_is_discovered_and_runs(tmp_path, ctx):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "greet.py").write_text(
        "from offset.tools.base import Tool, ToolResult, Danger\n"
        "class Greet(Tool):\n"
        "    name = 'greet'\n"
        "    description = 'say hello'\n"
        "    danger = Danger.SAFE\n"
        "    schema = {'type': 'object', 'properties': {'who': {'type': 'string'}}, 'required': ['who']}\n"
        "    def run(self, args, ctx):\n"
        "        return ToolResult.text('hello ' + args['who'])\n",
        encoding="utf-8",
    )
    found = discover([plugins])
    assert not found.errors and [t.name for t in found] == ["greet"]

    runtime = Runtime(Toolbox(list(found)), ctx)
    assert runtime.execute(call("greet", who="bear")).result.content == "hello bear"


def test_a_broken_plugin_is_reported_not_fatal(tmp_path):
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "bad.py").write_text("this is not python(", encoding="utf-8")
    (plugins / "good.py").write_text(
        "from offset.tools.base import Tool, ToolResult\n"
        "class Fine(Tool):\n"
        "    name = 'fine'\n"
        "    schema = {'type': 'object', 'properties': {}}\n"
        "    def run(self, args, ctx):\n"
        "        return ToolResult.text('ok')\n",
        encoding="utf-8",
    )
    found = discover([plugins])
    assert [t.name for t in found] == ["fine"]
    assert len(found.errors) == 1 and "bad.py" in str(found.errors[0].source)


def test_executable_tool_in_any_language_works(tmp_path, ctx):
    """The point of the manifest route: a shell script is a first-class tool."""
    home = tmp_path / "tools"
    home.mkdir()
    script = home / "wordcount.sh"
    script.write_text(
        '#!/bin/sh\n'
        'read -r payload\n'
        'n=$(printf "%s" "$payload" | tr -cd " " | wc -c)\n'
        'printf \'{"ok": true, "content": "words: %s", "display": "wordcount"}\' "$n"\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    (home / "tool.json").write_text(json.dumps({
        "name": "wordcount",
        "description": "count words",
        "danger": "safe",
        "schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "command": ["./wordcount.sh"],
    }), encoding="utf-8")

    found = discover([home])
    assert not found.errors and [t.name for t in found] == ["wordcount"]
    got = Runtime(Toolbox(list(found)), ctx).execute(call("wordcount", text="a b c"))
    assert got.result.ok and got.result.content.startswith("words:")
    assert got.result.display == "wordcount"


def test_external_tool_falls_back_to_plain_stdout(ctx):
    tool = ExternalTool("plain", "echoes", {"type": "object", "properties": {}}, ["/bin/echo", "just text"], danger=Danger.SAFE)
    got = Runtime(Toolbox([tool]), ctx).execute(call("plain"))
    assert got.result.ok and got.result.content == "just text"


def test_external_tool_failure_is_captured(ctx):
    tool = ExternalTool("nope", "fails", {"type": "object", "properties": {}}, ["/bin/sh", "-c", "echo bad >&2; exit 2"], danger=Danger.SAFE)
    got = Runtime(Toolbox([tool]), ctx).execute(call("nope"))
    assert not got.result.ok and "exited 2" in got.result.error


def test_missing_executable_is_a_message_not_a_crash(ctx):
    tool = ExternalTool("ghost", "n/a", {"type": "object", "properties": {}}, ["/definitely/not/here"], danger=Danger.SAFE)
    got = Runtime(Toolbox([tool]), ctx).execute(call("ghost"))
    assert not got.result.ok and "could not start" in got.result.error


def test_manifest_may_declare_several_tools(tmp_path):
    manifest = tmp_path / "tool.json"
    manifest.write_text(json.dumps([
        {"name": "one", "command": ["/bin/true"]},
        {"name": "two", "command": ["/bin/true"], "danger": "destructive"},
    ]), encoding="utf-8")
    tools = load_manifest(manifest)
    assert [t.name for t in tools] == ["one", "two"]
    assert tools[1].danger is Danger.DESTRUCTIVE


def test_manifest_without_a_command_is_an_error(tmp_path):
    (tmp_path / "tool.json").write_text(json.dumps({"name": "x"}), encoding="utf-8")
    found = discover([tmp_path])
    assert not found.tools and "command" in found.errors[0].message


def test_duplicate_custom_names_are_skipped_with_a_report(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "tool.json").write_text(json.dumps({"name": "same", "command": ["/bin/true"]}), encoding="utf-8")
    found = discover([a, b])
    assert len(found.tools) == 1
    assert any("duplicate" in e.message for e in found.errors)


def test_custom_tools_join_the_builtins(tmp_path, ctx):
    plugins = tmp_path / "t"
    plugins.mkdir()
    (plugins / "tool.json").write_text(json.dumps({"name": "extra", "command": ["/bin/true"]}), encoding="utf-8")
    box = Toolbox(builtin_tools())
    for tool in discover([plugins]):
        box.register(tool)
    assert "extra" in box and len(box) == 9
    assert {s.name for s in box.specs()} >= {"read", "bash", "extra"}
