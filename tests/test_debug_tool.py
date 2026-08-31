"""The debug tools' behaviour when there is nothing to debug.

Almost every call a model makes to these tools will be on a shell with no live
session — it has to launch one first — so the session-less path is the common
case, not the edge. It shipped crashing: `SessionBook.current` is a method, and
reading it without the call returned the bound method, which is truthy, so the
`is None` guard never fired and every inspection died on `.client`.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from offset.tools.base import ToolContext
from offset.tools.debug import debug_tools
from offset.tools.debug.tool import book


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    work = tmp_path / "proj"
    work.mkdir()
    return ToolContext(cwd=work, root=work, cancel=threading.Event(), timeout=30.0)


@pytest.fixture
def tools():
    made = {t.name: t for t in debug_tools()}
    yield made
    book().release()  # never leave a debuggee behind


@pytest.mark.parametrize(
    "action", ["stack", "scopes", "variables", "evaluate", "threads", "output"]
)
def test_inspecting_without_a_session_refuses_rather_than_crashing(tools, ctx, action):
    result = tools["debug_inspect"].run({"action": action}, ctx)
    assert not result.ok
    assert "no debug session" in (result.error or "")
    assert "AttributeError" not in (result.error or ""), "a crash is not a refusal"


@pytest.mark.parametrize("action", ["continue", "step_over", "step_in", "step_out", "pause"])
def test_moving_without_a_session_refuses_rather_than_crashing(tools, ctx, action):
    result = tools["debug"].run({"action": action}, ctx)
    assert not result.ok
    assert "no debug session" in (result.error or "")
    assert "AttributeError" not in (result.error or "")


def test_status_and_terminate_are_answerable_with_no_session(tools, ctx):
    """These two describe the world rather than acting on it, so they succeed."""
    for action in ("status", "terminate"):
        result = tools["debug"].run({"action": action}, ctx)
        assert result.ok, f"{action}: {result.error}"
        assert "no debug session" in result.content


def test_the_session_book_hands_back_a_session_or_none_never_a_method(tools, ctx):
    """The specific mistake: `book().current` instead of `book().current()`."""
    from offset.tools.debug.tool import _current

    assert _current() is None, "with nothing launched this must be None, not a bound method"


def test_an_unknown_action_lists_the_real_ones(tools, ctx):
    result = tools["debug"].run({"action": "sideways"}, ctx)
    assert not result.ok
    assert "no debug action" in (result.error or "")
    assert "launch" in (result.error or ""), "the refusal should name what is available"


def test_launch_without_a_program_refuses(tools, ctx):
    result = tools["debug"].run({"action": "launch"}, ctx)
    assert not result.ok
    assert "needs a program" in (result.error or "")


def test_launching_something_that_is_not_there_refuses(tools, ctx):
    result = tools["debug"].run({"action": "launch", "program": "nope.py"}, ctx)
    assert not result.ok
    assert "no such file" in (result.error or "")


def test_a_breakpoint_can_be_set_before_any_session_exists(tools, ctx):
    """DAP only accepts breakpoints inside the configuration window, which is
    inside launch — so they have to be collectable beforehand."""
    (ctx.cwd / "app.py").write_text("x = 1\n", encoding="utf-8")
    result = tools["debug"].run({"action": "breakpoint", "file": "app.py", "line": 1}, ctx)
    assert result.ok, result.error
    assert "applies on launch" in result.content

    cleared = tools["debug"].run({"action": "clear_breakpoints"}, ctx)
    assert cleared.ok
    assert "cleared 1" in cleared.content


def test_a_breakpoint_outside_the_workspace_is_refused(tools, ctx):
    result = tools["debug"].run({"action": "breakpoint", "file": "/etc/passwd", "line": 1}, ctx)
    assert not result.ok, "a path escape must not be accepted"


def test_adapters_reports_what_this_machine_can_debug(tools, ctx):
    result = tools["debug"].run({"action": "adapters"}, ctx)
    assert result.ok
    # Whatever is installed, every known language gets a line saying yes or why not.
    assert "python" in result.content
    for line in result.content.splitlines():
        if line.strip().startswith("python"):
            assert "+" in line or "install" in line, f"unhelpful line: {line}"
