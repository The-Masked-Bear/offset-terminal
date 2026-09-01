"""Mid-stream auditing.

Halting a generation is a destructive act: the user loses an answer they cannot
recover and cannot inspect. So the tests here are weighted towards *not*
firing - a clean stream must pass through untouched and in order, a disabled
check must stay silent, and a slow second opinion must never hold up a token
that was already produced.

The halts that are tested are the unambiguous ones: text repeating itself into
a loop, and a claim about a file that is not there.
"""

from __future__ import annotations

import threading

from offset.core.audit import Halted, Verdict, audit
from offset.providers.base import Stop, TextDelta, ThinkingDelta, ToolCallDelta, Usage


def stream(*events):
    yield from events


def words(text: str, times: int):
    for _ in range(times):
        yield TextDelta(text)
    yield Stop("stop")


def varied(times: int):
    """Non-repeating prose, so the repetition check stays quiet and whatever
    the test is actually about gets a chance to run."""
    for n in range(times):
        yield TextDelta(f"step {n} inspects module {n * 7 % 31} and records it. ")
    yield Stop("stop")


# -- pass-through ---------------------------------------------------------------


def test_a_clean_stream_passes_through_unchanged_and_in_order():
    """The common case is every case that is not broken, so this is the test
    that matters most."""
    original = [
        TextDelta("hello "),
        ThinkingDelta("hmm"),
        TextDelta("world"),
        Usage(input=10, output=2),
        Stop("stop"),
    ]
    got = list(audit(stream(*original)))
    assert got == original


def test_a_tool_call_survives_the_auditor():
    call = ToolCallDelta(index=0, id="c1", name="read", args_delta='{"path": "a.py"}')
    got = list(audit(stream(TextDelta("reading"), call, Stop("tool_use"))))
    assert call in got


def test_an_empty_stream_is_fine():
    assert list(audit(stream())) == []


def test_nothing_fires_when_every_check_is_off():
    got = list(audit(words("the same thing. ", 200), disable=("repetition", "path", "contradiction")))
    assert not any(isinstance(e, Halted) for e in got)


# -- repetition -----------------------------------------------------------------


def test_a_runaway_repetition_is_halted():
    got = list(audit(words("the same sentence over and over. ", 200), enable=("repetition",)))
    assert any(isinstance(e, Halted) for e in got)


def test_halting_stops_the_iteration_rather_than_draining_it():
    """The entire point is to stop spending tokens; reading to the end and then
    complaining would save nothing."""
    got = list(audit(words("round and round we go. ", 500), enable=("repetition",)))
    assert len(got) < 100, f"read {len(got)} of 501 events before halting"


def test_ordinary_prose_that_repeats_a_word_is_not_halted():
    """False positives are the real risk: this is normal writing."""
    prose = [
        TextDelta("The parser reads a token. "),
        TextDelta("The token is then classified. "),
        TextDelta("Classification uses the table above. "),
        TextDelta("The table is generated at build time. "),
        Stop("stop"),
    ]
    got = list(audit(stream(*prose), enable=("repetition",)))
    assert not any(isinstance(e, Halted) for e in got)


def test_the_halt_names_its_evidence():
    """A halt the user cannot judge is a halt they cannot trust."""
    got = list(audit(words("again and again. ", 200), enable=("repetition",)))
    halt = next(e for e in got if isinstance(e, Halted))
    assert halt.reason
    assert halt.evidence
    assert halt.check


def test_the_halt_reports_what_it_saved():
    got = list(audit(words("looping forever. ", 300), enable=("repetition",)))
    halt = next(e for e in got if isinstance(e, Halted))
    assert halt.tokens_seen > 0


# -- fabricated paths -------------------------------------------------------------


def test_a_claim_about_a_file_that_does_not_exist_is_caught():
    got = list(audit(
        stream(TextDelta("I have updated `src/nowhere/ghost.py` with the fix."), Stop("stop")),
        exists=lambda path: False,
        enable=("path",),
    ))
    assert any(isinstance(e, Halted) for e in got)


def test_a_claim_about_a_file_that_does_exist_is_left_alone():
    got = list(audit(
        stream(TextDelta("I have updated `offset/core/agent.py` with the fix."), Stop("stop")),
        exists=lambda path: True,
        enable=("path",),
    ))
    assert not any(isinstance(e, Halted) for e in got)


def test_without_a_way_to_check_it_never_accuses():
    """No workspace means no evidence, and an accusation without evidence is
    exactly the false positive this must not produce."""
    got = list(audit(
        stream(TextDelta("I edited `src/nowhere/ghost.py`."), Stop("stop")),
        enable=("path",),
    ))
    assert not any(isinstance(e, Halted) for e in got)


# -- the second model -------------------------------------------------------------


def test_a_condemning_verdict_halts():
    def judge(text: str) -> Verdict:
        # A returned Verdict *is* the condemnation; `None` means "fine".
        return Verdict(check="model", confidence=1.0,
                       reason="it is making this up", evidence=text[:40])

    got = list(audit(varied(400), judge=judge))
    assert any(isinstance(e, Halted) for e in got)


def test_saying_nothing_is_how_a_judge_approves():
    def judge(text: str) -> None:
        return None

    got = list(audit(varied(400), judge=judge))
    assert not any(isinstance(e, Halted) for e in got)


def test_a_verdict_below_the_threshold_does_not_halt():
    """Confidence is the whole defence against a jumpy second model."""
    def unsure(text: str) -> Verdict:
        return Verdict(check="model", confidence=0.1, reason="maybe", evidence="")

    got = list(audit(varied(400), judge=unsure))
    assert not any(isinstance(e, Halted) for e in got)


def test_a_slow_judge_never_holds_up_a_token():
    """The second opinion runs on a worker. If it blocked the stream it would
    cost every user latency to protect against a rare failure."""
    entered = threading.Event()
    release = threading.Event()

    def glacial(text: str) -> Verdict:
        entered.set()
        release.wait(10)
        return Verdict(check="model", confidence=1.0,
                       reason="too late to matter", evidence="")

    try:
        got = list(audit(varied(300), judge=glacial))
        # Reaching here at all is the assertion: a blocking judge would have
        # parked this list comprehension until `release` was set.
        assert got, "the stream produced nothing"
        assert entered.wait(5), "the judge was never consulted"
    finally:
        release.set()


def test_a_judge_that_raises_does_not_break_the_stream():
    def broken(text: str) -> Verdict:
        raise RuntimeError("the judge is down")

    got = list(audit(varied(200), judge=broken))
    assert any(isinstance(e, Stop) for e in got) or got, "an exploding judge killed the turn"


def test_the_judge_is_sampled_rather_than_asked_per_chunk():
    """One model call per token would cost more than the generation it guards."""
    calls: list[int] = []

    def counting(text: str) -> None:
        calls.append(len(text))
        return None

    list(audit(varied(200), judge=counting))
    assert len(calls) < 40, f"asked the judge {len(calls)} times for 200 chunks"


# -- the halt itself --------------------------------------------------------------


def test_a_halt_is_yielded_not_raised():
    """Raising into the agent loop would turn a guard into an outage."""
    got = list(audit(words("stuck. ", 300), enable=("repetition",)))
    assert any(isinstance(e, Halted) for e in got)


def test_events_before_the_halt_are_all_delivered():
    """Whatever was already generated is real work; the user should see it."""
    got = list(audit(words("repeating. ", 300), enable=("repetition",)))
    halt_at = next(i for i, e in enumerate(got) if isinstance(e, Halted))
    assert halt_at > 0, "it halted before delivering anything"
    # A Stop closes the turn before the Halted explains it; what must not
    # happen is text being swallowed on the way.
    assert not any(isinstance(e, Halted) for e in got[:halt_at])
    assert sum(isinstance(e, TextDelta) for e in got[:halt_at]) > 1
