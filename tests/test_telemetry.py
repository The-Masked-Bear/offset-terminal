"""Token, cost and trace accounting.

The invariant that matters most is not accuracy, it is honesty: a model whose
price nobody knows must produce no cost figure at all. A confident zero looks
like an answer and nobody re-checks an answer, so it is worse than a blank.

Second only to that: telemetry must never be able to end a turn. Every test
that feeds it rubbish asserts the turn survives.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from offset.core import telemetry as tel
from offset.core.agent import Finished, StepStarted, ToolFinished, ToolStarted
from offset.core.telemetry import (
    Entry,
    Ledger,
    Recorder,

    cost_of,
    price_of,
    rollup,
)
from offset.providers.base import StreamError, ToolCall, Usage
from offset.tools.base import ToolResult
from offset.tools.runtime import Invocation


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    monkeypatch.delenv(tel.OFF_ENV, raising=False)
    return tmp_path


def entry(**kw) -> Entry:
    base = dict(at=time.time(), model="claude-sonnet-4", tokens_in=1000,
                tokens_out=500, seconds=1.0)
    base.update(kw)
    return Entry(**base)


# -- prices ---------------------------------------------------------------------


def test_a_dated_variant_is_the_same_model():
    """`claude-sonnet-4-20250514` is `claude-sonnet-4`; requiring every dated
    id in the table would guarantee it goes stale."""
    assert price_of("claude-sonnet-4-20250514") == price_of("claude-sonnet-4")


def test_a_version_bump_is_not_the_same_model():
    """The failure this guards: `gpt-5.6-luna` silently billed at gpt-5 rates.
    A new version is a new price until somebody says otherwise."""
    assert price_of("gpt-5") is not None
    assert price_of("gpt-5.6-luna") is None


def test_an_unknown_model_has_no_price_rather_than_zero():
    assert price_of("something-nobody-has-heard-of") is None
    assert cost_of("something-nobody-has-heard-of", 10_000, 10_000) is None


def test_a_local_model_is_free_not_unknown():
    """Saying "unknown" for a model running on this machine would be pedantic."""
    assert price_of("ollama/llama3") == (0.0, 0.0)
    assert cost_of("mock", 999, 999) == 0.0


def test_cost_is_computed_per_million_tokens():
    assert cost_of("claude-sonnet-4", 1_000_000, 1_000_000) == pytest.approx(18.0)
    assert cost_of("claude-sonnet-4", 500_000, 0) == pytest.approx(1.5)


def test_the_longest_matching_prefix_wins():
    """`gpt-4o-mini` must not be priced as `gpt-4o`."""
    assert price_of("gpt-4o-mini") != price_of("gpt-4o")


def test_a_user_price_file_overrides_the_table(isolated):
    (isolated / "pricing.json").write_text(json.dumps({"claude-sonnet-4": [1.0, 2.0]}))
    assert price_of("claude-sonnet-4", isolated) == (1.0, 2.0)


def test_a_broken_price_file_does_not_lose_the_built_in_table(isolated):
    (isolated / "pricing.json").write_text("{not json")
    assert price_of("claude-sonnet-4", isolated) is not None


def test_one_bad_price_entry_does_not_discard_the_others(isolated):
    (isolated / "pricing.json").write_text(
        json.dumps({"good-model": [1.0, 2.0], "bad-model": "nonsense"}))
    assert price_of("good-model", isolated) == (1.0, 2.0)


# -- the ledger --------------------------------------------------------------------


def test_an_entry_round_trips_through_jsonl(isolated):
    led = Ledger(isolated)
    led.append(entry(model="gpt-4o", tokens_in=7, tokens_out=9))
    back = led.read()
    assert len(back) == 1
    assert back[0].model == "gpt-4o"
    assert (back[0].tokens_in, back[0].tokens_out) == (7, 9)


def test_an_unpriced_turn_stores_a_null_cost_not_a_zero(isolated):
    led = Ledger(isolated)
    led.append(entry(model="unknown-thing", cost=None))
    assert led.read()[0].cost is None


def test_a_mangled_line_is_skipped_not_fatal(isolated):
    led = Ledger(isolated)
    led.append(entry())
    with led.path.open("a", encoding="utf-8") as fh:
        fh.write("{ this is not json\n")
    led.append(entry(model="gpt-4o"))
    models = [e.model for e in led.read()]
    assert models == ["claude-sonnet-4", "gpt-4o"]


def test_the_ledger_rotates_at_its_cap(isolated):
    led = Ledger(isolated, max_bytes=400)
    for _ in range(40):
        led.append(entry())
    assert led.path.with_suffix(".jsonl.1").exists(), "it never rotated"
    assert led.path.stat().st_size < 4000


def test_one_rotation_keeps_the_older_records_readable(isolated):
    """Rotating must not blind `/usage` to the turns either side of the roll.

    Exactly one rotation - the module keeps one older file on purpose, so a
    cap small enough to roll eighteen times legitimately discards the middle.
    """
    led = Ledger(isolated, max_bytes=600)
    for n in range(5):
        led.append(entry(tokens_in=n))
    assert led.path.with_suffix(".jsonl.1").exists(), "it never rotated"
    seen = {e.tokens_in for e in led.read()}
    assert 0 in seen, "the pre-rotation records are unreachable"
    assert 4 in seen, "the post-rotation records are unreachable"


def test_recording_can_be_switched_off(isolated, monkeypatch):
    monkeypatch.setenv(tel.OFF_ENV, "1")
    led = Ledger(isolated)
    assert led.append(entry()) is False
    assert led.read() == []


def test_an_unwritable_home_does_not_raise(isolated):
    """A full disk or a read-only home loses the measurement, not the turn.

    A real read-only directory, not a patched `Path.mkdir` - patching that
    globally breaks pytest's own temp handling and hangs the run.
    """
    blocked = isolated / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        assert Ledger(blocked / "inner").append(entry()) is False
    finally:
        blocked.chmod(0o700)


# -- rollups ------------------------------------------------------------------------


def test_rollup_groups_by_model():
    got = rollup([entry(model="a"), entry(model="a"), entry(model="b")], "model")
    assert got["a"].turns == 2
    assert got["b"].turns == 1


def test_rollup_sums_tokens_and_cost():
    got = rollup([entry(cost=1.5), entry(cost=2.25)], "model")["claude-sonnet-4"]
    assert got.tokens_in == 2000
    assert got.cost == pytest.approx(3.75)


def test_an_unpriced_turn_marks_the_total_partial():
    """A sum that quietly omits the unpriced turns reads as complete."""
    got = rollup([entry(cost=1.0), entry(cost=None)], "model")["claude-sonnet-4"]
    assert got.partial is True
    assert got.money().endswith("+")


def test_a_fully_priced_total_is_not_marked_partial():
    got = rollup([entry(cost=1.0)], "model")["claude-sonnet-4"]
    assert got.partial is False
    assert "+" not in got.money()


def test_failures_are_counted():
    got = rollup([entry(), entry(reason="error", error="boom")], "model")["claude-sonnet-4"]
    assert got.failures == 1


def test_rollup_by_day_uses_local_dates():
    day = time.strftime("%Y-%m-%d", time.localtime())
    assert day in rollup([entry()], "day")


def test_the_report_says_when_a_figure_is_incomplete():
    lines = tel.report(rollup([entry(cost=None)], "model"))
    assert any("no known price" in line for line in lines)


def test_an_empty_report_says_so():
    assert "nothing recorded yet" in " ".join(tel.report({}))


# -- watching the loop ----------------------------------------------------------------


def invocation(name: str = "read", ok: bool = True) -> Invocation:
    call = ToolCall(id=f"c-{name}", name=name, args={})
    return Invocation(call=call, result=ToolResult(ok=ok, content="x", display="did x"))


def drive(rec: Recorder, *, model="claude-sonnet-4", tools=1, fail=False) -> None:
    rec.observe(StepStarted(0, model))
    for n in range(tools):
        call = ToolCall(id=f"c-{n}", name="read", args={})
        rec.observe(ToolStarted(call))
        rec.observe(ToolFinished(Invocation(call=call,
                                            result=ToolResult(ok=True, content="x", display="ok"))))
    rec.observe(Usage(input=1000, output=500))
    if fail:
        rec.observe(StreamError("the provider fell over"))
    rec.observe(Finished("error" if fail else "stop", Usage(input=1000, output=500), 1, "done"))


def test_a_turn_produces_one_ledger_entry(isolated):
    rec = Recorder(Ledger(isolated), model="claude-sonnet-4")
    rec.start()
    drive(rec)
    entries = Ledger(isolated).read()
    assert len(entries) == 1
    assert entries[0].tokens_in == 1000
    assert entries[0].tokens_out == 500
    assert entries[0].cost == pytest.approx(cost_of("claude-sonnet-4", 1000, 500))


def test_tool_calls_are_counted(isolated):
    rec = Recorder(Ledger(isolated), model="claude-sonnet-4")
    rec.start()
    drive(rec, tools=3)
    assert Ledger(isolated).read()[0].tools == 3


def test_a_failed_turn_records_its_error(isolated):
    rec = Recorder(Ledger(isolated), model="claude-sonnet-4")
    rec.start()
    drive(rec, fail=True)
    got = Ledger(isolated).read()[0]
    assert got.reason == "error"
    assert "fell over" in got.error


def test_an_unpriced_model_records_a_null_cost(isolated):
    rec = Recorder(Ledger(isolated), model="brand-new-thing")
    rec.start()
    drive(rec, model="brand-new-thing")
    assert Ledger(isolated).read()[0].cost is None


def test_a_malformed_event_is_swallowed(isolated):
    """Telemetry must never be the reason a turn dies."""
    rec = Recorder(Ledger(isolated))
    rec.start()
    rec.observe(object())
    rec.observe(None)
    rec.observe(StepStarted(0, "m"))
    rec.observe(Finished("stop", Usage(), 1, ""))
    assert len(Ledger(isolated).read()) == 1


def test_events_outside_a_turn_are_ignored(isolated):
    rec = Recorder(Ledger(isolated))
    rec.observe(Usage(input=99, output=99))
    assert Ledger(isolated).read() == []


# -- traces --------------------------------------------------------------------------


def test_a_tool_call_nests_under_its_step(isolated):
    rec = Recorder(Ledger(isolated), model="m")
    rec.start()
    drive(rec, tools=1)
    kinds = [s.kind for s in rec.last.spans]
    assert kinds[:3] == ["turn", "step", "tool"]
    tool = rec.last.spans[2]
    step = rec.last.spans[1]
    assert tool.parent == 1 and step.parent == 0


def test_a_failing_tool_marks_its_span(isolated):
    rec = Recorder(Ledger(isolated), model="m")
    rec.start()
    rec.observe(StepStarted(0, "m"))
    call = ToolCall(id="c1", name="bash", args={})
    rec.observe(ToolStarted(call))
    rec.observe(ToolFinished(Invocation(call=call, result=ToolResult.fail("nope"))))
    rec.observe(Finished("stop", Usage(), 1, ""))
    tool = next(s for s in rec.last.spans if s.kind == "tool")
    assert tool.ok is False


def test_the_trace_is_bounded(isolated):
    """A turn with a thousand tool calls must not grow memory without limit."""
    rec = Recorder(Ledger(isolated), model="m")
    rec.start()
    for n in range(tel.MAX_SPANS + 50):
        rec.trace.open("tool", f"t{n}")
    assert len(rec.trace.spans) <= tel.MAX_SPANS
    assert rec.trace.dropped > 0


def test_the_trace_renders_and_says_what_it_dropped(isolated):
    rec = Recorder(Ledger(isolated), model="m")
    rec.start()
    for n in range(tel.MAX_SPANS + 5):
        rec.trace.open("tool", f"t{n}")
    lines = rec.trace.lines()
    assert any("more spans" in line for line in lines)


def test_span_text_is_clipped(isolated):
    rec = Recorder(Ledger(isolated), model="m")
    rec.start()
    index = rec.trace.open("tool", "x" * 5000)
    assert len(rec.trace.spans[index].name) <= tel.CLIP


# -- commands ---------------------------------------------------------------------------


def test_usage_reports_nothing_before_any_turn(isolated):
    from offset.shell.commands import Command  # noqa: F401  (proves no import cycle)

    out = tel._usage(None, [])
    assert any("nothing recorded" in line for line in out.lines)


def test_usage_reports_a_recorded_turn(isolated):
    led = Ledger(isolated)
    led.append(entry(model="gpt-4o", cost=0.5))
    out = tel._usage(None, ["models"])
    assert any("gpt-4o" in line for line in out.lines)


def test_usage_can_list_failures(isolated):
    led = Ledger(isolated)
    led.append(entry(reason="error", error="it broke"))
    out = tel._usage(None, ["failures"])
    assert any("it broke" in line for line in out.lines)


def test_trace_says_so_when_there_is_none(isolated):
    tel._active = None
    out = tel._trace(None, [])
    assert any("no trace" in line for line in out.lines)


def test_install_never_raises_on_a_broken_state(isolated):
    tel.install(object())              # no .session, no .model
    tel.observe(StepStarted(0, "m"))   # must not raise either


def test_a_first_step_without_an_explicit_start_does_not_deadlock(isolated):
    """This hung the agent once. `_observe` holds the recorder lock and opens
    the turn itself when nobody called `start()`, so the lock must be
    reentrant - and a hang is the one failure the blanket except cannot catch.
    """
    done = threading.Event()
    rec = Recorder(Ledger(isolated), model="m")

    def drive_it():
        rec.observe(StepStarted(0, "m"))
        rec.observe(Finished("stop", Usage(input=1, output=1), 1, ""))
        done.set()

    thread = threading.Thread(target=drive_it, daemon=True)
    thread.start()
    assert done.wait(timeout=5), "recorder deadlocked on its own lock"
    assert len(Ledger(isolated).read()) == 1
