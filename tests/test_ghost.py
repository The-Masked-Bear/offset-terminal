"""Inline ghost-text suggestions.

The engine runs on the keypress path, so the invariant that matters most is not
"does it suggest the right thing" but "can it ever make typing feel slow". A
suggestion that arrives late is simply absent; a suggestion that blocks is a
stutter the user feels on every character.

The other half is restraint. Offering a completion the user cannot coherently
accept - mid-word, or with the cursor parked in the middle of the line - is
worse than offering nothing, because they have to look at it and dismiss it.
"""

from __future__ import annotations

import threading

import pytest

from offset.ui.ghost import Suggester, Suggestion, accept, accept_word


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "offset").mkdir()
    (tmp_path / "offset" / "ui").mkdir()
    (tmp_path / "offset" / "ui" / "ghost.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("hi\n")
    return tmp_path


def suggester(workspace, **kw) -> Suggester:
    return Suggester(workspace=workspace, **kw)


# -- what it offers -------------------------------------------------------------


def test_a_slash_command_completes(workspace):
    s = suggester(workspace, commands=lambda: {"models": (), "model": ()})
    found = s.suggest("/mode")
    assert found is not None
    assert (found.text.startswith("l") or found.text.startswith("ls"))
    s.close()


def test_history_completes_from_what_was_typed_before(workspace):
    s = suggester(workspace, history=["deploy the staging branch"])
    found = s.suggest("deploy the")
    assert found is not None
    assert found.text == " staging branch"
    s.close()


def test_the_most_recent_history_wins(workspace):
    """Two entries share a prefix; the one used last is the likelier intent."""
    s = suggester(workspace, history=["fix the parser newer", "fix the parser older"])
    found = s.suggest("fix the parser")
    assert found is not None
    assert "newer" in found.text
    s.close()


def test_remember_moves_a_repeat_to_the_front(workspace):
    s = suggester(workspace, history=["b task", "a task"])
    s.remember("a task")
    assert s.history[0] == "a task"
    assert s.history.count("a task") == 1, "a repeat was stored twice"
    s.close()


def test_remember_ignores_blank_input(workspace):
    s = suggester(workspace, history=[])
    s.remember("   ")
    assert s.history == ()
    s.close()


# -- restraint ------------------------------------------------------------------


def test_nothing_is_suggested_for_an_empty_buffer(workspace):
    s = suggester(workspace, history=["something"])
    assert s.suggest("") is None
    assert s.suggest("   ") is None
    s.close()


def test_nothing_is_suggested_when_the_cursor_is_not_at_the_end(workspace):
    """An appended completion would land in the middle of the user's own text,
    which they cannot accept coherently."""
    s = suggester(workspace, history=["deploy the staging branch"])
    assert s.suggest("deploy the", cursor=3) is None
    s.close()


def test_a_suggestion_is_only_the_remainder(workspace):
    """The renderer draws this dim after the cursor; returning the whole line
    would duplicate what the user already typed."""
    s = suggester(workspace, history=["deploy the staging branch"])
    found = s.suggest("deploy")
    assert found is not None
    assert not found.text.startswith("deploy")
    assert ("deploy" + found.text) == "deploy the staging branch"
    s.close()


# -- latency --------------------------------------------------------------------


def test_a_slow_source_yields_nothing_and_does_not_block(workspace):
    """The whole design constraint. A scan that misses its deadline must cost
    an absent suggestion, never a stalled keypress."""
    started = threading.Event()
    release = threading.Event()

    def glacial(path: str) -> tuple[str, ...]:
        started.set()
        release.wait(10)  # held until the assertion below has run
        return ("never.py",)

    s = suggester(workspace, scan=glacial, deadline=0.01)
    try:
        found = s.suggest("./")
        # It returned. That is the assertion: had it waited for the scan, this
        # line would not be reached until `release` is set.
        assert found is None or found.source != "path"
        assert started.wait(5), "the scan never even started"
    finally:
        release.set()
        s.close()


def test_a_scan_that_finishes_later_is_used_on_the_next_keypress(workspace):
    """Missing the deadline must not poison the result: the work was done and
    the next keystroke should benefit from it."""
    ready = threading.Event()
    s = suggester(workspace, deadline=0.001, on_ready=ready.set)
    try:
        s.suggest("offset/u")
        assert ready.wait(10), "the scan never completed"
        found = s.suggest("offset/u")
        assert found is not None, "the completed scan was thrown away"
    finally:
        s.close()


def test_repeated_keystrokes_do_not_queue_a_scan_each(workspace):
    """Debounce: typing eight characters must not mean eight directory walks."""
    calls: list[str] = []
    done = threading.Event()

    def counting(path: str) -> tuple[str, ...]:
        calls.append(path)
        done.set()
        return ("ui",)

    s = suggester(workspace, scan=counting, deadline=0.5)
    try:
        for n in range(1, 9):
            s.suggest("offset/"[:n] or "o")
        assert done.wait(5)
        assert len(calls) <= 3, f"one scan per keystroke: {calls}"
    finally:
        s.close()


# -- accepting ------------------------------------------------------------------


def test_accept_appends_the_whole_suggestion():
    assert accept("deploy", Suggestion(" the staging branch", "history")) == "deploy the staging branch"


def test_accept_word_takes_only_one():
    got = accept_word("deploy", Suggestion(" the staging branch", "history"))
    assert got == "deploy the"


def test_accept_word_takes_a_directory_separator_with_the_word():
    """A directory without its slash is not a position anything continues from."""
    got = accept_word("off", Suggestion("set/ui/ghost.py", "path"))
    assert got == "offset/"


def test_accepting_nothing_leaves_the_buffer_alone():
    assert accept("deploy", None) == "deploy"
    assert accept_word("deploy", None) == "deploy"
    assert accept("deploy", Suggestion("", "history")) == "deploy"


def test_a_closed_suggester_stops_cleanly(workspace):
    s = suggester(workspace)
    s.suggest("offset/")
    s.close()
    s.close()  # idempotent: a second close must not raise
