"""Three confirmed defects, and the fixes that close them.

Each of these was reproduced before it was fixed: a cancelled turn produced a
history providers reject, `cd` evaporated between calls, and `glob`/`grep`
happily walked whatever `.gitignore` was there to keep them out of.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from offset.core.agent import to_messages
from offset.core.entries import BRANCH_SUMMARY, COMPACTION, MESSAGE, TOOL_CALL, TOOL_RESULT
from offset.core.session import Session
from offset.providers.base import ToolCall
from offset.tools.base import ToolContext, Toolbox
from offset.tools.builtin import Bash, Glob, Grep, builtin_tools
from offset.tools.runtime import Approval, Runtime
from offset.tools.walk import walk


def call(name: str, **args) -> ToolCall:
    return ToolCall(id="c1", name=name, args=args)


@pytest.fixture()
def runtime(tmp_path):
    ctx = ToolContext(cwd=tmp_path, timeout=30.0)
    return Runtime(Toolbox(builtin_tools()), ctx, Approval(mode="yolo"))


# -- 1. dangling tool calls -------------------------------------------------


def test_an_interrupted_tool_call_still_gets_a_result(tmp_path):
    """Providers reject a tool call with no result; a cancelled turn made one."""
    s = Session.create(tmp_path)
    s.say("user", "read the file")
    s.say("assistant", "reading")
    s.append(TOOL_CALL, {"id": "call_1", "tool": "read", "args": {"path": "a.py"}})
    # ...and then the user hit ctrl-c, so no tool_result was ever written.

    msgs = to_messages(s.transcript())
    assert [m.role for m in msgs] == ["user", "assistant", "tool"]
    assert msgs[-1].tool_call_id == "call_1"
    assert "interrupted" in msgs[-1].text


def test_only_the_unanswered_call_is_synthesised(tmp_path):
    s = Session.create(tmp_path)
    s.say("user", "do two things")
    s.say("assistant", "working")
    s.append(TOOL_CALL, {"id": "a", "tool": "read", "args": {}})
    s.append(TOOL_CALL, {"id": "b", "tool": "glob", "args": {}})
    s.append(TOOL_RESULT, {"id": "a", "tool": "read", "content": "real result"})

    msgs = to_messages(s.transcript())
    results = {m.tool_call_id: m.text for m in msgs if m.role == "tool"}
    assert results["a"] == "real result"
    assert "interrupted" in results["b"]
    assert len(results) == 2


def test_every_call_has_exactly_one_result(tmp_path):
    s = Session.create(tmp_path)
    s.say("user", "go")
    s.append(TOOL_CALL, {"id": "x", "tool": "read", "args": {}})
    s.append(TOOL_RESULT, {"id": "x", "tool": "read", "content": "done"})
    msgs = to_messages(s.transcript())

    ids = [c.id for m in msgs if m.role == "assistant" for c in m.tool_calls]
    answers = [m.tool_call_id for m in msgs if m.role == "tool"]
    assert sorted(ids) == sorted(answers), "a duplicate or missing result breaks the request"


def test_an_orphan_result_is_dropped(tmp_path):
    """A result whose call is not on this branch cannot be sent."""
    s = Session.create(tmp_path)
    s.say("user", "go")
    s.append(TOOL_RESULT, {"id": "ghost", "tool": "read", "content": "from another branch"})

    msgs = to_messages(s.transcript())
    assert [m.role for m in msgs] == ["user"]
    assert all(m.role != "tool" for m in msgs)


def test_a_summary_entry_reaches_the_model(tmp_path):
    """branch_summary and compaction are conversational; they used to vanish."""
    s = Session.create(tmp_path)
    s.append(BRANCH_SUMMARY, {"text": "we tried the parser rewrite and it failed"})
    s.say("user", "what next?")

    msgs = to_messages(s.transcript())
    assert any("parser rewrite" in m.text for m in msgs), "the summary never reached the model"
    assert msgs[0].role == "user"


def test_a_compaction_entry_reaches_the_model(tmp_path):
    s = Session.create(tmp_path)
    s.append(COMPACTION, {"summary": "earlier: fixed the tokeniser"})
    s.say("user", "carry on")
    msgs = to_messages(s.transcript())
    assert any("tokeniser" in m.text for m in msgs)


def test_a_clean_history_is_left_alone(tmp_path):
    s = Session.create(tmp_path)
    s.say("user", "hi")
    s.say("assistant", "hello")
    before = to_messages(s.transcript())
    assert [m.role for m in before] == ["user", "assistant"]


# -- 2. persistent shell ----------------------------------------------------


def test_cd_survives_between_calls(runtime, tmp_path):
    """The reported defect: every command used to start back at the top."""
    (tmp_path / "sub").mkdir()
    runtime.execute(call("bash", command="cd sub"))
    got = runtime.execute(call("bash", command="pwd"))
    assert got.result.content.strip().endswith("sub"), got.result.content


def test_exports_survive_between_calls(runtime):
    runtime.execute(call("bash", command="export OFFSET_TEST_TOKEN=carried"))
    got = runtime.execute(call("bash", command='echo "[$OFFSET_TEST_TOKEN]"'))
    assert "[carried]" in got.result.content


def test_the_state_report_never_leaks_into_the_output(runtime):
    got = runtime.execute(call("bash", command="echo hello"))
    assert got.result.content.strip() == "hello", "the state probe leaked into the output"
    assert "__offset_" not in got.result.content
    assert "\x00" not in got.result.content


def test_exit_codes_survive_the_state_probe(runtime):
    got = runtime.execute(call("bash", command="echo out; exit 7"))
    assert got.result.data["exit"] == 7 and not got.result.ok
    assert got.result.content.strip() == "out"


def test_reset_forgets_the_directory(runtime, tmp_path):
    (tmp_path / "sub").mkdir()
    runtime.execute(call("bash", command="cd sub"))
    runtime.execute(call("bash", command="true", reset=True))
    got = runtime.execute(call("bash", command="pwd"))
    assert got.result.content.strip() == str(tmp_path.resolve())


def test_a_vanished_directory_falls_back_to_the_workspace(runtime, tmp_path):
    (tmp_path / "doomed").mkdir()
    runtime.execute(call("bash", command="cd doomed"))
    (tmp_path / "doomed").rmdir()
    got = runtime.execute(call("bash", command="pwd"))
    assert got.result.ok, got.result.error
    assert got.result.content.strip() == str(tmp_path.resolve())


def test_a_timeout_leaves_the_previous_directory_intact(runtime, tmp_path):
    (tmp_path / "kept").mkdir()
    runtime.execute(call("bash", command="cd kept"))
    timed_out = runtime.execute(call("bash", command="sleep 20", timeout=1))
    assert not timed_out.result.ok and "timed out" in timed_out.result.error
    got = runtime.execute(call("bash", command="pwd"))
    assert got.result.content.strip().endswith("kept")


def test_a_timed_out_command_still_shows_its_output(runtime):
    got = runtime.execute(call("bash", command="echo partial; sleep 20", timeout=1))
    assert "partial" in got.result.error
    assert "__offset_" not in got.result.error


# -- 3. gitignore-aware walking --------------------------------------------


def repo(root: Path, ignore: str) -> Path:
    (root / ".gitignore").write_text(ignore, encoding="utf-8")
    return root


def test_glob_honours_gitignore(runtime, tmp_path):
    repo(tmp_path, "build/\n*.log\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "artifact.py").write_text("x", encoding="utf-8")
    (tmp_path / "keep.py").write_text("x", encoding="utf-8")
    (tmp_path / "noisy.log").write_text("x", encoding="utf-8")

    got = runtime.execute(call("glob", pattern="*")).result.content
    assert "keep.py" in got
    assert "artifact.py" not in got, "an ignored directory was walked"
    assert "noisy.log" not in got


def test_ignored_files_can_be_asked_for_explicitly(runtime, tmp_path):
    repo(tmp_path, "secret/\n")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "hidden.py").write_text("x", encoding="utf-8")
    assert "hidden.py" not in runtime.execute(call("glob", pattern="*.py")).result.content
    assert "hidden.py" in runtime.execute(call("glob", pattern="*.py", ignored=True)).result.content


def test_grep_honours_gitignore(runtime, tmp_path):
    repo(tmp_path, "vendor/\n")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text("needle here\n", encoding="utf-8")
    (tmp_path / "mine.py").write_text("needle here\n", encoding="utf-8")

    got = runtime.execute(call("grep", pattern="needle")).result.content
    assert "mine.py" in got and "vendor" not in got


def test_negation_reinstates_a_file(tmp_path):
    repo(tmp_path, "*.log\n!keep.log\n")
    (tmp_path / "drop.log").write_text("x", encoding="utf-8")
    (tmp_path / "keep.log").write_text("x", encoding="utf-8")
    names = {p.name for p in walk(tmp_path)}
    assert "keep.log" in names and "drop.log" not in names


def test_a_nested_gitignore_applies_to_its_subtree(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
    (tmp_path / "pkg" / "a.tmp").write_text("x", encoding="utf-8")
    (tmp_path / "b.tmp").write_text("x", encoding="utf-8")
    names = {str(p.relative_to(tmp_path)) for p in walk(tmp_path)}
    assert "pkg/a.tmp" not in names
    assert "b.tmp" in names, "a nested rule must not leak upward"


def test_an_anchored_pattern_only_matches_at_the_root(tmp_path):
    repo(tmp_path, "/only.txt\n")
    (tmp_path / "only.txt").write_text("x", encoding="utf-8")
    (tmp_path / "deep").mkdir()
    (tmp_path / "deep" / "only.txt").write_text("x", encoding="utf-8")
    names = {str(p.relative_to(tmp_path)) for p in walk(tmp_path)}
    assert "only.txt" not in names
    assert "deep/only.txt" in names


def test_a_double_star_pattern_matches_at_any_depth(tmp_path):
    repo(tmp_path, "**/generated/**\n")
    (tmp_path / "a" / "generated").mkdir(parents=True)
    (tmp_path / "a" / "generated" / "x.py").write_text("x", encoding="utf-8")
    (tmp_path / "a" / "real.py").write_text("x", encoding="utf-8")
    names = {str(p.relative_to(tmp_path)) for p in walk(tmp_path)}
    assert "a/real.py" in names
    assert not any("generated" in n for n in names)


def test_skipping_a_large_ignored_tree_is_faster_than_walking_it(tmp_path):
    """The point of the walker: a big ignored directory costs nothing."""
    heavy = tmp_path / "node_modules_like"
    heavy.mkdir()
    for i in range(1200):
        (heavy / f"f{i}.js").write_text("x", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("node_modules_like/\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("x", encoding="utf-8")

    start = time.monotonic()
    respected = [p for p in walk(tmp_path, respect_gitignore=True)]
    fast = time.monotonic() - start

    start = time.monotonic()
    everything = [p for p in walk(tmp_path, respect_gitignore=False, prune=frozenset())]
    slow = time.monotonic() - start

    assert len(respected) < 10, f"the ignored tree was walked: {len(respected)} files"
    assert len(everything) > 1200
    assert fast < slow, f"respecting .gitignore ({fast:.3f}s) should beat walking it ({slow:.3f}s)"
