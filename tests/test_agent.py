"""The agent loop and the multi-model scheduler.

Everything runs against scripted providers, so these assert behaviour of the
loop itself: that history survives an interrupted turn, that tool results feed
back correctly, that one dead model cannot take the others down, and that
concurrent seats still produce a stable, replayable order.
"""

from __future__ import annotations

import threading
import time

import pytest

from offset.core.agent import (
    Agent,
    AgentConfig,
    Finished,
    StepStarted,
    ToolFinished,
    ToolStarted,
    to_messages,
)
from offset.core.entries import MESSAGE, TOOL_CALL, TOOL_RESULT
from offset.core.multimodel import Ensemble, Opinion, Seat, normalise
from offset.core.session import Session
from offset.providers.base import Message, Request, TextDelta, Turn, Usage
from offset.providers.mock import Mock, script
from offset.providers.registry import ModelInfo
from offset.tools.base import Danger, Tool, ToolContext, ToolResult, Toolbox
from offset.tools.runtime import Approval, Runtime


class Echo(Tool):
    name = "echo"
    description = "echo back"
    danger = Danger.SAFE
    schema = {"type": "object", "properties": {"say": {"type": "string"}}, "required": ["say"]}

    def run(self, args, ctx):
        return ToolResult.text(f"echoed {args['say']}")


def make(tmp_path, scripts, tools=None, mode="yolo"):
    session = Session.create(tmp_path)
    ctx = ToolContext(cwd=tmp_path, timeout=5.0)
    runtime = Runtime(Toolbox(tools or [Echo()]), ctx, Approval(mode=mode))
    provider = Mock(scripts)
    meta = ModelInfo("mock", "mock", "mock", 8192, 4096)
    agent = Agent(
        session, runtime, AgentConfig(model="mock", max_steps=6),
        resolver=lambda _m: (provider, meta), provider=provider,
    )
    return agent, session, provider, runtime


# -- conversation rebuild ---------------------------------------------------


def test_entries_rebuild_into_provider_messages(tmp_path):
    s = Session.create(tmp_path)
    s.say("user", "hello")
    s.say("assistant", "calling a tool")
    s.append(TOOL_CALL, {"id": "c1", "tool": "echo", "args": {"say": "hi"}})
    s.append(TOOL_RESULT, {"id": "c1", "tool": "echo", "content": "echoed hi"})
    s.say("user", "thanks")

    msgs = to_messages(s.transcript())
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "user"]
    assert msgs[1].tool_calls[0].name == "echo"
    assert msgs[2].tool_call_id == "c1"


def test_orphan_tool_call_still_gets_an_assistant_turn(tmp_path):
    s = Session.create(tmp_path)
    s.append(TOOL_CALL, {"id": "c1", "tool": "echo", "args": {}})
    msgs = to_messages(s.transcript())
    assert msgs[0].role == "assistant" and msgs[0].tool_calls


# -- the loop ---------------------------------------------------------------


def test_a_plain_answer_ends_in_one_step(tmp_path):
    agent, session, _, _ = make(tmp_path, [script("here is the answer")])
    got = agent.send("question?")
    assert got.text == "here is the answer"
    assert got.steps == 1 and got.stop_reason == "stop"
    assert [e.text for e in session.transcript() if e.type == MESSAGE] == ["question?", "here is the answer"]


def test_tool_call_loops_back_into_the_model(tmp_path):
    agent, session, provider, _ = make(tmp_path, [
        script(tool_calls=[("c1", "echo", {"say": "hi"})]),
        script("the tool said hi"),
    ])
    got = agent.send("use the tool")
    assert got.steps == 2 and got.text == "the tool said hi"
    assert [i.result.content for i in got.invocations] == ["echoed hi"]

    # the second request must contain the tool result
    second = provider.requests[1]
    assert second.messages[-1].role == "tool"
    assert second.messages[-1].text == "echoed hi"


def test_every_step_is_persisted_as_it_happens(tmp_path):
    agent, session, _, _ = make(tmp_path, [
        script(tool_calls=[("c1", "echo", {"say": "x"})]),
        script("done"),
    ])
    agent.send("go")
    kinds = [e.type for e in session.transcript()]
    assert kinds == [MESSAGE, TOOL_CALL, TOOL_RESULT, MESSAGE]
    assert Session.open(session.path).transcript(), "history must survive a reload"


def test_events_arrive_in_a_usable_order(tmp_path):
    agent, _, _, _ = make(tmp_path, [
        script("thinking out loud", tool_calls=[("c1", "echo", {"say": "y"})]),
        script("finished"),
    ])
    seen = [type(e).__name__ for e in agent.run("go")]
    assert seen[0] == "StepStarted"
    assert "TextDelta" in seen
    assert seen.index("ToolStarted") < seen.index("ToolFinished")
    assert seen[-1] == "Finished"


def test_a_failing_tool_is_reported_back_not_fatal(tmp_path):
    agent, session, provider, _ = make(tmp_path, [
        script(tool_calls=[("c1", "nonexistent", {})]),
        script("I will try something else"),
    ])
    got = agent.send("go")
    assert got.steps == 2 and got.text == "I will try something else"
    assert "no tool named" in provider.requests[1].messages[-1].text


def test_step_budget_is_enforced(tmp_path):
    forever = [script(tool_calls=[(f"c{i}", "echo", {"say": str(i)})]) for i in range(20)]
    agent, _, _, _ = make(tmp_path, forever)
    agent.config.max_steps = 3
    got = agent.send("loop")
    assert got.steps == 3 and got.stop_reason == "max_steps"


def test_provider_error_stops_the_turn_and_is_recorded(tmp_path):
    agent, _, _, _ = make(tmp_path, [script("partial", error="upstream exploded")])
    got = agent.send("go")
    assert got.stop_reason == "error" and got.error == "upstream exploded"


def test_cancellation_stops_the_loop_but_keeps_results(tmp_path):
    class Slow(Tool):
        name = "slow"
        danger = Danger.SAFE
        schema = {"type": "object", "properties": {}}

        def run(self, args, ctx):
            for _ in range(200):
                ctx.check()
                time.sleep(0.01)
            return ToolResult.text("never")

    agent, session, _, runtime = make(
        tmp_path,
        [script(tool_calls=[("c1", "slow", {})]), script("should not get here")],
        tools=[Slow()],
    )
    threading.Timer(0.15, runtime.cancel).start()
    got = agent.send("go")
    assert got.stop_reason == "cancelled" and got.steps == 1
    assert any(e.type == TOOL_RESULT for e in session.transcript()), "the result must still be recorded"


def test_the_model_only_sees_the_active_branch(tmp_path):
    agent, session, provider, _ = make(tmp_path, [script("A"), script("B")])
    agent.send("first question")
    root = session.roots()[0]
    session.branch(root.id)
    agent.send("second question")
    assert [m.text for m in provider.requests[1].messages] == ["first question", "second question"]


def test_max_tokens_is_clamped_to_the_model(tmp_path):
    agent, _, provider, _ = make(tmp_path, [script("ok")])
    agent.config.max_tokens = 999_999
    agent.send("go")
    assert provider.requests[0].max_tokens == 4096


# -- multi-model ------------------------------------------------------------


def seat_with(model: str, role: str, text: str, *, weight: float = 1.0, fail: bool = False, delay: float = 0.0) -> Seat:
    def scripted(_req):
        if delay:
            time.sleep(delay)
        if fail:
            raise RuntimeError("provider down")
        return script(text)

    return Seat(model=model, role=role, weight=weight, provider=Mock(scripted))


def ensemble(*seats: Seat) -> Ensemble:
    meta = ModelInfo("x", "mock", "x", 8192, 4096)
    return Ensemble(seats, resolver=lambda m: (Mock(), meta))


def ask() -> Request:
    return Request(model="ignored", messages=[Message("user", "what is 2+2?")])


def test_seats_answer_concurrently_but_report_in_order():
    room = ensemble(
        seat_with("slowest", "planner", "four", delay=0.25),
        seat_with("middle", "implementer", "four", delay=0.1),
        seat_with("fastest", "critic", "five"),
    )
    started = time.monotonic()
    opinions = room.gather(ask())
    elapsed = time.monotonic() - started
    assert [o.seat.model for o in opinions] == ["slowest", "middle", "fastest"]
    assert elapsed < 0.5, "seats did not overlap"


def test_one_dead_provider_does_not_sink_the_others():
    room = ensemble(
        seat_with("healthy", "implementer", "four"),
        seat_with("broken", "critic", "", fail=True),
    )
    opinions = room.gather(ask())
    assert opinions[0].ok and opinions[0].text == "four"
    assert not opinions[1].ok and "provider down" in opinions[1].error


def test_race_returns_the_first_usable_answer():
    room = ensemble(
        seat_with("slow", "implementer", "slow answer", delay=0.3),
        seat_with("quick", "cheap", "quick answer"),
    )
    verdict = room.race(ask())
    assert verdict.winner.text == "quick answer"
    assert "quick" in verdict.reason


def test_race_skips_failures():
    room = ensemble(
        seat_with("broken", "cheap", "", fail=True),
        seat_with("working", "implementer", "real answer", delay=0.15),
    )
    assert room.race(ask()).winner.text == "real answer"


def test_vote_is_weighted_and_reports_dissent():
    room = ensemble(
        seat_with("a", "implementer", "four"),
        seat_with("b", "critic", "four"),
        seat_with("c", "cheap", "five", weight=1.5),
    )
    verdict = room.vote(ask())
    assert verdict.winner.text == "four"
    assert verdict.tally == {"four": 2.0, "five": 1.5}
    assert [o.text for o in verdict.dissent] == ["four", "five"]


def test_vote_ignores_case_and_spacing():
    room = ensemble(seat_with("a", "x", "Four\n"), seat_with("b", "y", "four"))
    assert room.vote(ask()).tally == {"four": 2.0}
    assert normalise("  Hello   World ") == "hello world"


def test_vote_with_no_answers_has_no_winner():
    room = ensemble(seat_with("a", "x", "", fail=True))
    verdict = room.vote(ask())
    assert verdict.winner is None and "no seat" in verdict.reason


def test_council_lets_a_judge_choose():
    room = ensemble(
        seat_with("a", "implementer", "answer alpha"),
        seat_with("b", "critic", "answer beta"),
    )
    judge = Seat(model="judge", role="referee", provider=Mock(lambda _r: script("1 because beta is tighter")))
    verdict = room.council(ask(), judge)
    assert verdict.winner.text == "answer beta"
    assert "chose [1]" in verdict.reason


def test_council_falls_back_to_voting_when_the_judge_fails():
    room = ensemble(
        seat_with("a", "implementer", "shared"),
        seat_with("b", "critic", "shared"),
        seat_with("c", "cheap", "different"),
    )

    def broken(_req):
        raise RuntimeError("judge offline")

    verdict = room.council(ask(), Seat(model="judge", role="referee", provider=Mock(broken)))
    assert verdict.winner.text == "shared"
    assert "fell back" in verdict.reason


def test_council_ignores_an_out_of_range_pick():
    room = ensemble(seat_with("a", "x", "one"), seat_with("b", "y", "two"))
    judge = Seat(model="j", role="referee", provider=Mock(lambda _r: script("99")))
    assert room.council(ask(), judge).winner is not None


def test_usage_is_summed_across_seats():
    def with_usage(model, n):
        return Seat(model=model, role="x", provider=Mock(lambda _r, n=n: script("hi", usage=Usage(input=n, output=n))))

    room = ensemble(with_usage("a", 10), with_usage("b", 5))
    verdict = room.vote(ask())
    assert verdict.usage.input == 15 and verdict.usage.output == 15


def test_streaming_interleaves_lanes_and_every_lane_terminates():
    room = ensemble(
        seat_with("a", "implementer", "alpha answer"),
        seat_with("b", "critic", "beta answer"),
    )
    lanes: dict[str, list[str]] = {}
    for seat, event in room.stream(ask()):
        if isinstance(event, TextDelta):
            lanes.setdefault(seat.model, []).append(event.text)
    assert "".join(lanes["a"]) == "alpha answer"
    assert "".join(lanes["b"]) == "beta answer"


def test_streaming_survives_a_broken_seat():
    room = ensemble(seat_with("good", "x", "fine"), seat_with("bad", "y", "", fail=True))
    seen = {seat.model for seat, _ in room.stream(ask())}
    assert seen == {"good", "bad"}, "a failing lane must still emit and close"


def test_relay_passes_work_down_the_roles():
    room = ensemble(
        seat_with("p", "planner", "the plan"),
        seat_with("i", "implementer", "the code"),
        seat_with("c", "critic", "the review"),
    )
    chain = room.relay(ask())
    assert [o.text for o in chain] == ["the plan", "the code", "the review"]


def test_routing_picks_the_heaviest_seat_for_a_role():
    light = seat_with("light", "critic", "x", weight=0.5)
    heavy = seat_with("heavy", "critic", "x", weight=2.0)
    room = ensemble(light, heavy)
    assert room.pick("critic") is heavy
    assert room.pick("nobody") is None
    assert set(room.by_role()) == {"critic"}


def test_an_empty_roster_is_refused():
    with pytest.raises(ValueError):
        Ensemble([])
