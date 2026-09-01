"""Splitting one feature across parallel coder models.

Two ways this goes badly and nothing notices. Two units that touch the same
file run at once and one silently overwrites the other, so the finished work
is missing a piece nobody can point at. Or a plan comes back with a dependency
cycle and the run either deadlocks or executes an arbitrary prefix of it.

Both are planning-time problems, so both are refused at planning time here.
Everything is driven through an injected architect and runner: no model is
called, and the scheduling is what is under test.
"""

from __future__ import annotations

import json
import threading

import pytest

from offset.core.decompose import (
    Plan,
    Unit,
    decompose,
    execute,
    parse_plan,
    plan_waves,
)


def unit(uid: str, *, files=(), consumes=(), provides=(), needs=()) -> Unit:
    return Unit(id=uid, task=f"do {uid}", files=tuple(files),
                provides=tuple(provides), consumes=tuple(consumes), needs=tuple(needs))


def reply_for(units: list[dict], goal: str = "a goal") -> str:
    return json.dumps({"goal": goal, "units": units})


def recorder(barrier: threading.Barrier | None = None):
    """A runner that records the order and concurrency it was driven with.

    `barrier` forces genuine overlap when a test needs to prove parallelism:
    without it a fast unit can finish before the next one starts, and the peak
    concurrency reads as one even though the pool is working correctly.
    """
    lock = threading.Lock()
    order: list[str] = []
    live: list[str] = []
    peak = {"n": 0}

    def run(plan, u: Unit) -> tuple[str, str]:
        with lock:
            live.append(u.id)
            peak["n"] = max(peak["n"], len(live))
            order.append(u.id)
        try:
            barrier.wait(2) if barrier is not None else None
            return f"{u.id} done", ""
        finally:
            with lock:
                live.remove(u.id)

    run.order = order      # type: ignore[attr-defined]
    run.peak = peak        # type: ignore[attr-defined]
    return run


# -- parsing --------------------------------------------------------------------


def test_a_plan_is_parsed_out_of_prose_wrapped_json():
    """Models narrate. Refusing anything that is not bare JSON would reject
    most real replies."""
    body = reply_for([{"id": "a", "task": "write the parser"}])
    plan = parse_plan("a goal", f"Sure! Here is the plan:\n```json\n{body}\n```\nHope that helps.")
    assert [u.id for u in plan.units] == ["a"]


def test_an_unparseable_reply_is_refused_rather_than_guessed():
    """Inventing a plan from an unreadable answer is how a run does something
    nobody asked for."""
    with pytest.raises(ValueError):
        parse_plan("a goal", "I would rather not, actually.")


def test_a_plan_with_no_units_is_refused():
    with pytest.raises(ValueError):
        parse_plan("a goal", reply_for([]))


def test_the_unit_budget_is_enforced():
    """A runaway architect must not be able to spawn a hundred agents."""
    many = [{"id": f"u{n}", "task": "work"} for n in range(200)]
    with pytest.raises(ValueError):
        parse_plan("a goal", reply_for(many), limit=12)


def test_declared_files_and_interfaces_survive_parsing():
    plan = parse_plan("g", reply_for([{
        "id": "a", "task": "t", "files": ["x.py"],
        "provides": ["Iface"], "consumes": [], "needs": [],
    }]))
    assert plan.units[0].files == ("x.py",)
    assert plan.units[0].provides == ("Iface",)


# -- scheduling -----------------------------------------------------------------


def test_independent_units_share_a_wave():
    plan = Plan(goal="g", units=(unit("a"), unit("b"), unit("c")))
    waves = plan_waves(plan)
    assert len(waves) == 1
    assert {u.id for u in waves[0].units} == {"a", "b", "c"}


def test_a_consumer_waits_for_its_provider():
    plan = Plan(goal="g", units=(
        unit("api", provides=["Store"]),
        unit("ui", consumes=["Store"]),
    ))
    waves = plan_waves(plan)
    assert [u.id for w in waves for u in w.units] == ["api", "ui"]
    assert len(waves) == 2


def test_an_explicit_dependency_is_honoured():
    plan = Plan(goal="g", units=(unit("second", needs=["first"]), unit("first")))
    waves = plan_waves(plan)
    assert waves[0].units[0].id == "first"


def test_two_units_touching_one_file_are_serialised():
    """The silent corruption this whole module has to prevent: concurrent
    edits to one file lose one of them, and nothing reports it."""
    plan = Plan(goal="g", units=(
        unit("a", files=["shared.py"]),
        unit("b", files=["shared.py"]),
    ))
    waves = plan_waves(plan)
    assert len(waves) == 2, "two writers to one file ended up in the same wave"


def test_units_touching_different_files_stay_parallel():
    plan = Plan(goal="g", units=(unit("a", files=["x.py"]), unit("b", files=["y.py"])))
    assert len(plan_waves(plan)) == 1


def test_a_cycle_is_refused_and_named():
    plan = Plan(goal="g", units=(
        unit("a", consumes=["B"], provides=["A"]),
        unit("b", consumes=["A"], provides=["B"]),
    ))
    with pytest.raises(ValueError) as caught:
        plan_waves(plan)
    message = str(caught.value).lower()
    assert "cycle" in message or "circular" in message
    assert "a" in message and "b" in message, "the cycle was refused without naming it"


def test_consuming_something_no_unit_provides_is_allowed():
    """It may already exist in the repository - most units consume interfaces
    nobody in *this* plan writes. Refusing would make the common case fail."""
    plan = Plan(goal="g", units=(unit("a", consumes=["AlreadyInTheRepo"]),))
    assert len(plan_waves(plan)) == 1


def test_needing_a_unit_that_is_not_in_the_plan_is_refused():
    """`needs` names a *unit*, so a missing one is a broken plan rather than
    an assumption about the repository."""
    plan = Plan(goal="g", units=(unit("a", needs=["nosuchunit"]),))
    with pytest.raises(ValueError):
        plan_waves(plan)


# -- execution ------------------------------------------------------------------


def test_every_unit_runs():
    plan = Plan(goal="g", units=(unit("a"), unit("b")))
    run = execute(plan, recorder())
    assert {u.id for u in run.plan.units if u.state == "done"} == {"a", "b"}


def test_a_wave_really_does_run_in_parallel():
    # Every unit waits at the barrier, so all four must be in flight at once
    # for any of them to finish. No sleeping, no guessing.
    gate = threading.Barrier(4, timeout=10)
    run_fn = recorder(gate)
    plan = Plan(goal="g", units=tuple(unit(f"u{n}") for n in range(4)))
    execute(plan, run_fn, workers=4)
    assert run_fn.peak["n"] == 4, f"a wave of four peaked at {run_fn.peak['n']}"


def test_a_dependent_never_starts_before_its_provider_finishes():
    run_fn = recorder()
    plan = Plan(goal="g", units=(
        unit("api", provides=["Store"]),
        unit("ui", consumes=["Store"]),
    ))
    execute(plan, run_fn)
    assert run_fn.order.index("api") < run_fn.order.index("ui")


def test_a_failure_blocks_its_dependents():
    def failing(plan, u: Unit) -> tuple[str, str]:
        if u.id == "api":
            return "", "the api unit fell over"
        return "ok", ""

    plan = Plan(goal="g", units=(
        unit("api", provides=["Store"]),
        unit("ui", consumes=["Store"]),
    ))
    run = execute(plan, failing)
    states = {u.id: u.state for u in run.plan.units}
    assert states["api"] == "failed"
    assert states["ui"] in ("blocked", "skipped"), states


def test_a_failure_does_not_stop_an_independent_subtree():
    """Isolation is the point of a graph: one bad branch must not cost the
    others their work."""
    def failing(plan, u: Unit) -> tuple[str, str]:
        if u.id == "bad":
            return "", "nope"
        return "ok", ""

    plan = Plan(goal="g", units=(unit("bad"), unit("good")))
    run = execute(plan, failing)
    states = {u.id: u.state for u in run.plan.units}
    assert states["bad"] == "failed"
    assert states["good"] == "done", "an unrelated unit was punished for a sibling"


def test_the_reason_a_unit_was_blocked_is_recorded():
    def failing(plan, u: Unit) -> tuple[str, str]:
        if u.id == "api":
            return "", "the api unit fell over"
        return "ok", ""

    plan = Plan(goal="g", units=(unit("api", provides=["S"]), unit("ui", consumes=["S"])))
    run = execute(plan, failing)
    blocked = next(u for u in run.plan.units if u.id == "ui")
    assert blocked.error, "a blocked unit with no reason is impossible to debug"


def test_cancelling_stops_further_waves():
    cancel = threading.Event()

    def once(plan, u: Unit) -> tuple[str, str]:
        cancel.set()
        return "ok", ""

    plan = Plan(goal="g", units=(unit("a", provides=["S"]), unit("b", consumes=["S"])))
    run = execute(plan, once, cancel=cancel)
    states = {u.id: u.state for u in run.plan.units}
    assert states["b"] != "done", "a cancelled run carried on to the next wave"


# -- end to end -----------------------------------------------------------------


def test_decompose_drives_the_architect_then_the_coders():
    asked: list[str] = []

    def architect(prompt: str) -> str:
        asked.append(prompt)
        return reply_for([
            {"id": "api", "task": "write the store", "provides": ["Store"]},
            {"id": "ui", "task": "use the store", "consumes": ["Store"]},
        ], goal="build a thing")

    run_fn = recorder()
    run = decompose("build a thing", architect=architect, runner=run_fn)
    assert asked and "build a thing" in asked[0]
    assert run_fn.order.index("api") < run_fn.order.index("ui")
    assert all(u.state == "done" for u in run.plan.units)


def test_an_architect_that_returns_rubbish_fails_before_anything_runs():
    ran: list[str] = []

    def architect(prompt: str) -> str:
        return "I have no idea what you mean."

    def runner(plan, u: Unit) -> tuple[str, str]:
        ran.append(u.id)
        return "ok", ""

    with pytest.raises(ValueError):
        decompose("build a thing", architect=architect, runner=runner)
    assert ran == [], "coders were dispatched against an unparsed plan"
