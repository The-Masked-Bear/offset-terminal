"""Slash commands and shell rendering.

Commands are pure functions over state, so all of this runs without a
terminal.  The interactive layer is smoke-tested separately against a real
pty, because that is the only honest way to test a TUI.
"""

from __future__ import annotations

import time

import pytest

from offset.core.agent import Agent, AgentConfig
from offset.core.session import Session
from offset.eggs.catalogue import build_engine
from offset.providers.mock import Mock
from offset.providers.registry import ModelInfo
from offset.shell import render
from offset.shell.commands import (
    BY_NAME,
    COMMANDS,
    ShellState,
    complete,
    dispatch,
    resolve_overlay,
)
from offset.tools.base import ToolContext, Toolbox
from offset.tools.builtin import builtin_tools
from offset.tools.runtime import Approval, Runtime


@pytest.fixture()
def state(tmp_path, monkeypatch):

    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "cfg"))
    session = Session.create(tmp_path / "sessions")
    toolbox = Toolbox(builtin_tools())
    approval = Approval(mode="auto-edit")
    runtime = Runtime(toolbox, ToolContext(cwd=tmp_path, timeout=5.0), approval)
    meta = ModelInfo("mock", "mock", "mock", 8192, 4096)
    agent = Agent(session, runtime, AgentConfig(model="mock"),
                  resolver=lambda _m: (Mock(), meta), provider=Mock())
    eggs = build_engine(tmp_path / "eggs.json")
    return ShellState(session, agent, toolbox, approval, eggs, tmp_path)


# -- dispatch ---------------------------------------------------------------


def test_plain_text_is_not_a_command(state):
    assert not dispatch(state, "please refactor the parser").handled


def test_unknown_command_suggests_something(state):
    got = dispatch(state, "/mdoel")
    assert not got.lines[0].startswith("/mdoel is fine")
    assert "unknown command" in got.lines[0]
    assert got.tone == "err"


def test_every_command_is_reachable_and_documented(state):
    for command in COMMANDS:
        assert command.summary, f"{command.name} has no summary"
        assert BY_NAME[command.name] is command
    # Wide enough that nothing is ellipsised, and tall enough that the
    # annotated layout is used at all: past forty commands the summaries are
    # dropped for a dense name grid when they will not fit, because losing a
    # summary beats losing a command. Whether a cramped pane degrades tidily
    # is a separate question, covered by the layout tests.
    state.width, state.height = 220, 80
    listing = dispatch(state, "/help")
    body = "\n".join(listing.lines)
    for command in COMMANDS:
        # /help is laid out in two columns, so a name need not start its line.
        assert f"/{command.name}" in body, f"/{command.name} is not documented"
        assert command.summary in body, f"/{command.name}'s summary is missing"


def test_a_summary_is_always_reachable_even_when_the_pane_is_tiny(state):
    """The grid drops summaries to fit. They must still be obtainable, or the
    command set becomes undiscoverable on a short terminal.

    The pane is short but not narrow: filtering is what makes the summaries
    reappear, and how a *narrow* pane ellipsises them is the layout tests'
    business, not this one's.
    """
    state.width, state.height = 200, 20
    for command in COMMANDS[:5]:
        body = "\n".join(dispatch(state, f"/help {command.name}").lines)
        assert command.summary in body, f"/{command.name}'s summary is unreachable"


def test_aliases_work(state):
    assert dispatch(state, "/?").lines
    assert dispatch(state, "/trophies").overlay is not None


def test_completion_offers_slash_commands():
    assert "/model" in complete("/mo")
    assert "/models" in complete("/mo")
    assert complete("/zzz") == []


# -- eggs never shadow real commands ---------------------------------------


def test_an_egg_command_still_fires(state):
    got = dispatch(state, "/bear")
    assert got.reveal is not None and got.reveal.lines == ["No."]


def test_a_bare_word_can_be_an_egg(state):
    assert dispatch(state, "sudo").reveal is not None


def test_a_sentence_is_never_mistaken_for_an_egg(state):
    """`rm` is an egg; "rm the old parser" is a request."""
    assert not dispatch(state, "rm the old parser").handled


def test_commands_win_over_eggs(state):
    """`/clear` is a real command and must not be hijacked by a joke."""
    got = dispatch(state, "/clear")
    assert got.reveal is None and "context cleared" in got.lines[0]


# -- model ------------------------------------------------------------------


def test_model_without_arguments_opens_the_picker(state):
    got = dispatch(state, "/model")
    assert got.overlay is not None and got.overlay.kind == "model"
    assert state.overlay is got.overlay
    assert got.overlay.items, "the picker must list models"


def test_model_with_an_argument_switches_immediately(state):
    got = dispatch(state, "/model deepseek-reasoner")
    assert state.agent.config.model == "deepseek-reasoner"
    assert "deepseek" in got.lines[0]


def test_picking_from_the_overlay_switches_model(state):
    dispatch(state, "/model")
    state.overlay.selected = 3
    chosen = state.overlay.payload[3]
    resolve_overlay(state, state.overlay, accepted=True)
    assert state.agent.config.model == chosen.id
    assert state.overlay is None


def test_cancelling_the_picker_changes_nothing(state):
    before = state.agent.config.model
    dispatch(state, "/model")
    resolve_overlay(state, state.overlay, accepted=False)
    assert state.agent.config.model == before and state.overlay is None


def test_switching_model_is_recorded_in_the_session(state):
    dispatch(state, "/model gpt-4o")
    assert any(e.type == "model_change" for e in state.session.all_entries())


def test_overlay_navigation_wraps(state):
    dispatch(state, "/model")
    panel = state.overlay
    panel.selected = 0
    panel.move(-1)
    assert panel.selected == len(panel.items) - 1
    panel.move(1)
    assert panel.selected == 0


# -- login ------------------------------------------------------------------


def test_login_opens_a_masked_field(state):
    got = dispatch(state, "/login anthropic")
    assert got.overlay.kind == "login" and got.overlay.secret


def test_a_submitted_key_is_stored_with_tight_permissions(state):
    from offset.providers.registry import credential, credentials_file, provider_for

    dispatch(state, "/login anthropic")
    state.overlay.buffer = "sk-not-a-real-key"
    got = resolve_overlay(state, state.overlay, accepted=True)
    assert "stored a key" in got.lines[0]
    assert credential(provider_for("anthropic")) == "sk-not-a-real-key"
    assert oct(credentials_file().stat().st_mode)[-3:] == "600"


def test_an_empty_key_is_refused(state):
    dispatch(state, "/login openai")
    assert resolve_overlay(state, state.overlay, accepted=True).tone == "err"


def test_the_secret_never_reaches_the_screen(state):
    dispatch(state, "/login anthropic")
    state.overlay.buffer = "sk-super-secret-value"
    painted = render.overlay(60, 8, state.overlay, 0.0)
    assert "sk-super-secret-value" not in painted
    assert "secret" not in painted.lower()


# -- tools and approval -----------------------------------------------------


def test_tools_lists_everything_as_enabled(state):
    got = dispatch(state, "/tools")
    for name in ("read", "write", "bash", "fetch"):
        assert any(line.startswith(name) for line in got.lines)
    assert "all enabled" in got.lines[-1]


def test_approval_mode_can_be_changed(state):
    assert "auto-edit" in dispatch(state, "/approve").lines[0]
    dispatch(state, "/approve yolo")
    assert state.approval.mode == "yolo"
    assert dispatch(state, "/approve sideways").tone == "err"


# -- session ----------------------------------------------------------------


def test_session_reports_where_it_lives(state):
    state.session.say("user", "hello")
    lines = dispatch(state, "/session").lines
    assert any(state.session.id in line for line in lines)
    assert any("entries" in line for line in lines)


def test_fork_leaves_the_original_alone(state):
    state.session.say("user", "one")
    before = len(state.session)
    got = dispatch(state, "/fork")
    assert "forked to" in got.lines[0]
    assert len(state.session) == before


def test_branch_needs_something_to_branch_from(state):
    assert dispatch(state, "/branch").tone == "err"
    state.session.say("user", "one")
    state.session.say("assistant", "a")
    state.session.say("user", "two")
    assert dispatch(state, "/branch").tone == "ok"


def test_tree_overlay_lists_the_conversation(state):
    state.session.say("user", "first")
    state.session.say("assistant", "reply")
    panel = dispatch(state, "/tree").overlay
    assert panel is not None and len(panel.items) == 2
    assert "active" in panel.notes


def test_jumping_in_the_tree_moves_the_leaf(state):
    state.session.say("user", "first")
    state.session.say("assistant", "reply")
    dispatch(state, "/tree")
    state.overlay.selected = 0
    resolve_overlay(state, state.overlay, accepted=True)
    assert state.session.transcript() == []


def test_clear_keeps_the_history_on_disk(state):
    state.session.say("user", "remember me")
    dispatch(state, "/clear")
    assert state.session.transcript() == []
    assert len(Session.open(state.session.path).all_entries().__iter__().__next__().id) == 26


def test_spec_validates_its_arguments(state):
    assert dispatch(state, "/spec").tone == "err"
    got = dispatch(state, "/spec 4 make the parser faster")
    assert "running 4 branches" in got.lines[0]
    assert got.job is not None, "/spec must actually queue work, not just say so"
    assert dispatch(state, "/spec 99 x").lines[0].startswith("running 6 branches"), "count is capped"


def test_spec_names_the_angles_and_models_up_front(state):
    got = dispatch(state, "/spec 3 speed up the parser")
    body = "\n".join(got.lines)
    for angle in ("minimal", "rewrite", "test-first"):
        assert angle in body
    assert "nothing touches your files until /adopt" in body


def test_spec_really_runs_when_its_job_is_executed(state, tmp_path, monkeypatch):
    """The job the command hands back must produce real ranked attempts."""
    from offset.providers.mock import Mock, script
    from offset.providers.registry import PROVIDERS

    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setitem(PROVIDERS, "mock", lambda: Mock([
        script(tool_calls=[("c1", "write", {"path": "app.py", "content": "VALUE = 2\n"})]),
        script("done"),
    ]))
    outcome = dispatch(state, "/spec 2 bump the value")
    report = outcome.job()

    assert state.spec_run is not None and len(state.spec_run.attempts) == 2
    assert any("/adopt 1" in line for line in report.lines)
    assert (tmp_path / "app.py").read_text() == "VALUE = 1\n", "the workspace must be untouched"

    adopted = dispatch(state, "/adopt 1")
    assert adopted.tone == "ok", adopted.lines
    assert (tmp_path / "app.py").read_text() == "VALUE = 2\n"

    assert dispatch(state, "/discard").tone == "ok"
    assert state.spec_run is None


def test_spec_with_no_arguments_shows_the_last_run(state):
    from offset.core.branches import BranchRun

    assert dispatch(state, "/spec").tone == "err"
    state.spec_run = BranchRun(task="earlier")
    assert dispatch(state, "/spec").lines == ["no branches ran"]


def test_adopt_needs_a_run_first(state):
    assert dispatch(state, "/adopt").tone == "err"
    assert "run /spec first" in dispatch(state, "/adopt 1").lines[0]


def test_verify_command_is_remembered(state):
    assert "(none" in dispatch(state, "/verify").lines[0]
    dispatch(state, "/verify pytest -q")
    assert state.verify_command == "pytest -q"
    assert "pytest -q" in dispatch(state, "/verify").lines[0]


def test_quit_asks_to_leave(state):
    assert dispatch(state, "/quit").quit


# -- rendering --------------------------------------------------------------


def test_transcript_paints_roles_differently(state):
    state.session.say("user", "a question")
    state.session.say("assistant", "an answer")
    painted = render.transcript(60, 10, list(state.session.transcript()), t=0.0)
    assert "a question" in painted and "an answer" in painted


def test_long_lines_wrap_instead_of_overflowing():
    text = "word " * 80
    lines = render._wrap(text.strip(), 30)
    assert all(len(line) <= 30 for line in lines)
    assert "".join(lines).replace(" ", "") == text.replace(" ", "")


def test_an_unbreakable_token_is_split_not_lost():
    lines = render._wrap("x" * 90, 20)
    assert all(len(line) <= 20 for line in lines)
    assert sum(line.count("x") for line in lines) == 90


def test_rendering_is_colour_free_when_asked(state, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    state.session.say("user", "hello")
    painted = render.transcript(50, 6, list(state.session.transcript()), t=0.0)
    assert "\x1b" not in painted


def test_status_bar_fits_its_width(state):
    for width in (40, 80, 120):
        row = render.status(width, state, busy=False, t=0.0)
        assert "\n" not in row


def test_reveal_panel_animates_frames(state):
    reveal = state.eggs.command("gravity")
    first = render.reveal_panel(50, reveal, 0.0)
    later = render.reveal_panel(50, reveal, 0.4)
    assert first != later, "frames must advance with time"


# -- approval handshake -----------------------------------------------------


def test_the_shell_actually_asks_before_a_dangerous_call(state):
    """Regression: `ask` used to be `lambda: False`, so `safe` mode silently
    denied every write instead of putting a question on screen."""
    import threading
    import types

    from offset.shell.app import Shell
    from offset.tools.builtin import Bash

    shell = object.__new__(Shell)
    shell.state = state
    shell.messages = []
    shell.approval_gate = None
    shell.approval_answer = False
    shell.app = types.SimpleNamespace(invalidate=lambda: None)
    state.approval.ask = shell.ask_approval
    state.approval.mode = "safe"

    answers: list[bool] = []
    asking = threading.Thread(
        target=lambda: answers.append(state.approval.check(Bash(), {"command": "rm -rf /"})[0]),
        daemon=True,
    )
    asking.start()

    for _ in range(200):
        if state.overlay is not None:
            break
        time.sleep(0.01)
    assert state.overlay is not None, "no approval question was ever shown"
    assert state.overlay.kind == "approve"
    assert "bash" in state.overlay.title
    assert any("rm -rf /" in line for line in state.overlay.items)

    shell.answer_approval(True)
    asking.join(timeout=5)
    assert answers == [True]
    assert state.overlay is None


def test_denying_a_call_reaches_the_waiting_tool(state):
    import threading
    import types

    from offset.shell.app import Shell
    from offset.tools.builtin import Bash

    shell = object.__new__(Shell)
    shell.state = state
    shell.messages = []
    shell.approval_gate = None
    shell.approval_answer = False
    shell.app = types.SimpleNamespace(invalidate=lambda: None)
    state.approval.ask = shell.ask_approval
    state.approval.mode = "safe"

    verdicts: list[tuple[bool, str]] = []
    asking = threading.Thread(
        target=lambda: verdicts.append(state.approval.check(Bash(), {"command": "ls"})),
        daemon=True,
    )
    asking.start()
    for _ in range(200):
        if state.overlay is not None:
            break
        time.sleep(0.01)

    shell.answer_approval(False)
    asking.join(timeout=5)
    assert verdicts and verdicts[0][0] is False
    assert "declined" in verdicts[0][1]


def test_always_allow_is_remembered(state):
    import threading
    import types

    from offset.shell.app import Shell
    from offset.tools.builtin import Bash

    shell = object.__new__(Shell)
    shell.state = state
    shell.messages = []
    shell.approval_gate = None
    shell.approval_answer = False
    shell.app = types.SimpleNamespace(invalidate=lambda: None)
    state.approval.ask = shell.ask_approval
    state.approval.mode = "safe"

    asking = threading.Thread(
        target=lambda: state.approval.check(Bash(), {"command": "ls"}),
        daemon=True,
    )
    asking.start()
    for _ in range(200):
        if state.overlay is not None:
            break
        time.sleep(0.01)
    shell.answer_approval(True, remember=True)
    asking.join(timeout=5)

    assert "bash" in state.approval.remembered
    allowed, _ = state.approval.check(Bash(), {"command": "ls"})
    assert allowed, "a remembered tool must not ask again"


# -- the newly wired commands ----------------------------------------------


def test_compact_reports_headroom_before_doing_anything(state):
    got = dispatch(state, "/compact")
    assert got.job is None, "an idle session must not summarise anything"
    assert "nothing to compact yet" in got.lines[0]
    assert "/compact now" in got.lines[1]


def test_compact_can_be_forced(state):
    state.session.say("user", "something to summarise")
    got = dispatch(state, "/compact now")
    assert got.job is not None, "forcing must actually queue the work"


def test_rewind_lists_nothing_before_any_write(state):
    assert "nothing has been snapshotted" in dispatch(state, "/rewind").lines[0]


def test_rewind_lists_snapshots_and_restores_one(state, tmp_path):
    from offset.core.snapshots import capture

    target = tmp_path / "app.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    capture(state.session, "app.py", tool="write", root=tmp_path)
    target.write_text("VALUE = 2\n", encoding="utf-8")

    listed = dispatch(state, "/rewind")
    assert "app.py" in listed.lines[0]
    assert "/rewind <number>" in "\n".join(listed.lines)

    restored = dispatch(state, "/rewind 1")
    assert restored.tone == "ok", restored.lines
    assert target.read_text() == "VALUE = 1\n"


def test_rewind_refuses_a_bad_number(state, tmp_path):
    from offset.core.snapshots import capture

    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    capture(state.session, "a.py", tool="write", root=tmp_path)
    assert dispatch(state, "/rewind 99").tone == "err"
    assert dispatch(state, "/rewind banana").tone == "err"


def test_mcp_says_so_when_nothing_is_configured(state):
    got = dispatch(state, "/mcp")
    assert "no MCP servers configured" in got.lines[0]
    assert any("mcp.json" in line for line in got.lines)


def test_sessions_lists_the_current_one(state):
    state.session.say("user", "hello from this session")
    got = dispatch(state, "/sessions")
    assert any("hello from this session"[:20] in line for line in got.lines)


def test_a_writing_tool_snapshots_before_it_writes(state, tmp_path):
    """The /rewind safety net only exists if the runtime takes the snapshot."""
    from offset.providers.base import ToolCall

    target = tmp_path / "notes.txt"
    target.write_text("before\n", encoding="utf-8")

    taken: list[str] = []
    state.agent.runtime.before_write = lambda tool, args: taken.append(tool.name)
    state.approval.mode = "yolo"
    state.agent.runtime.execute(ToolCall("c", "write", {"path": "notes.txt", "content": "after\n"}))

    assert taken == ["write"], "the hook must fire for a writing tool"
    assert target.read_text() == "after\n"


def test_a_reading_tool_does_not_snapshot(state, tmp_path):
    from offset.providers.base import ToolCall

    (tmp_path / "r.txt").write_text("x\n", encoding="utf-8")
    taken: list[str] = []
    state.agent.runtime.before_write = lambda tool, args: taken.append(tool.name)
    state.agent.runtime.execute(ToolCall("c", "read", {"path": "r.txt"}))
    assert taken == [], "reads cannot lose data, so they cost nothing"
