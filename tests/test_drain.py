"""Every event the loop can emit reaches the screen.

There is precedent for this file existing. `StreamError` was once unhandled in
`drain`, and the symptom was not a traceback - it was *total silence*: a failing
provider produced no reply, no error, forever. An event nobody renders is
indistinguishable from an agent that has hung.

So these tests do not check wording. They check that each event type produces
*some* observable change, and that the handler for it actually runs - which is
the part a "does it import?" check cannot tell you, because `isinstance(event,
Thing)` only fails when an event of that type actually arrives.
"""

from __future__ import annotations

import pytest

from offset.core.agent import Compacted, Finished
from offset.providers.base import StreamError, TextDelta, ThinkingDelta, Usage
from offset.shell.app import Shell, build_state


@pytest.fixture()
def shell(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OFFSET_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("OFFSET_NO_MODEL_FETCH", "1")
    proj = tmp_path / "proj"
    proj.mkdir()
    state = build_state(proj, model="mock")
    sh = Shell(state)
    yield sh
    try:
        state.session.close()
    except Exception:
        pass


def drained(sh: Shell, *events):
    """Push events through the real handler and report what changed."""
    before = (list(sh.messages), sh.live, sh.note, sh.busy)
    for event in events:
        sh.events.put(event)
    sh.drain()
    after = (list(sh.messages), sh.live, sh.note, sh.busy)
    return before != after


# -- the events ---------------------------------------------------------------


def test_text_reaches_the_live_buffer(shell):
    shell.events.put(TextDelta("hello"))
    shell.drain()
    assert "hello" in shell.live


def test_thinking_is_shown_as_a_note(shell):
    shell.events.put(ThinkingDelta("pondering"))
    shell.drain()
    assert shell.note


def test_a_stream_error_is_never_silent(shell):
    """The original bug: a failing provider that said nothing at all."""
    shell.events.put(StreamError("the provider refused", status=500))
    shell.drain()
    assert any(tone == "err" for tone, _ in shell.messages)


def test_an_auth_error_names_the_command_that_fixes_it(shell):
    shell.events.put(StreamError("invalid api key", status=401))
    shell.drain()
    assert any("/login" in line for _, line in shell.messages)


def test_a_compaction_is_announced(shell):
    """Silently dropping history the model can no longer see would make a
    forgetful answer look like a bad model rather than a full context."""
    shell.events.put(Compacted(before=60_566, after=8_351, summarised=69))
    shell.drain()
    assert any("compact" in line.lower() for _, line in shell.messages)


def test_the_compaction_notice_says_how_much_was_freed(shell):
    shell.events.put(Compacted(before=60_566, after=8_351, summarised=69))
    shell.drain()
    line = next(line for _, line in shell.messages if "compact" in line.lower())
    assert "69" in line
    assert "52,215" in line, "the freed figure should be before - after"


def test_the_compaction_notice_says_where_the_originals_went(shell):
    """Nothing is destroyed, and a user told only "compacted" would not know."""
    shell.events.put(Compacted(before=100, after=30, summarised=4))
    shell.drain()
    assert any("/tree" in line for _, line in shell.messages)


def test_a_sentinel_clears_the_busy_state(shell):
    shell.busy = True
    shell.live = "half a sentence"
    shell.events.put(None)
    shell.drain()
    assert not shell.busy
    assert shell.live == ""


def test_a_worker_exception_is_reported(shell):
    shell.events.put(RuntimeError("the worker fell over"))
    shell.drain()
    assert any("fell over" in line for _, line in shell.messages)


def test_finishing_is_handled(shell):
    assert drained(shell, Finished("max_steps", Usage(), 24, ""))


def test_draining_an_empty_queue_is_a_no_op(shell):
    shell.drain()
    assert shell.messages == []


# -- the general invariant ------------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        TextDelta("x"),
        ThinkingDelta("x"),
        StreamError("x", status=500),
        Compacted(before=10, after=5, summarised=1),
        RuntimeError("x"),
        Finished("max_steps", Usage(), 24, ""),
    ],
    ids=lambda e: type(e).__name__,
)
def test_no_event_is_silently_swallowed(shell, event):
    """The invariant that would have caught the `StreamError` silence, and
    that catches the next one: an event that changes nothing observable is
    indistinguishable from an agent that has stopped responding."""
    assert drained(shell, event), f"{type(event).__name__} produced no visible change"
