"""Several models working one task together on a plan they wrote themselves.

The properties under test are the ones that make this different from running the
same prompt N times: the plan's shape comes from the model, independent steps
really do run at once on different models, a read-only step cannot write even if
it tries, and a failure rewrites the part of the plan that has not happened yet.
"""

from __future__ import annotations

import threading
import time

import pytest

from offset.core.multimodel import Ensemble, Seat
from offset.core.workflow import (
    DONE,
    FAILED,
    MAX_STEPS,
    PENDING,
    SKIPPED,
    Plan,
    PlanError,
    Step,
    StepResult,
    parse_plan,
    readonly_toolbox,
    run_workflow,
)
from offset.tools.base import Danger, Toolbox
from offset.tools.builtin import builtin_tools


@pytest.fixture()
def roster() -> Ensemble:
    return Ensemble([Seat("opus", "planner"), Seat("sonnet", "implementer"), Seat("haiku", "critic")])


def scripted(**by_id: StepResult):
    """A worker whose answer per step is decided by the test."""
    seen: list[tuple[str, str]] = []

    def work(step: Step, seat: Seat, briefing: str) -> StepResult:
        seen.append((step.id, seat.model))
        return by_id.get(step.id, StepResult(text=f"{seat.model} did {step.id}"))

    work.seen = seen  # type: ignore[attr-defined]
    return work


# -- reading a plan out of a real reply --------------------------------------


def test_a_plan_survives_prose_and_a_code_fence():
    reply = (
        "Sure! Here's the plan:\n\n```json\n"
        '{"steps": [{"id": "look", "task": "read the parser", "role": "critic", "writes": false},'
        '{"id": "fix", "task": "rewrite it", "needs": ["look"]}]}\n```\nHope that helps!'
    )
    plan = parse_plan("make it faster", reply)
    assert [s.id for s in plan] == ["look", "fix"]
    assert plan.by_id("look").writes is False
    assert plan.by_id("fix").needs == ("look",)


def test_an_unparseable_reply_becomes_one_honest_step():
    """"I could not decompose this" is a plan: do the whole thing."""
    plan = parse_plan("make it faster", "I'm afraid I can't help with that.")
    assert len(plan) == 1
    assert plan.steps[0].task == "make it faster"


def test_an_empty_reply_becomes_one_step():
    assert len(parse_plan("goal", "")) == 1


def test_writes_defaults_to_true_when_unstated():
    """Guessing read-only would let a step race a writer."""
    plan = parse_plan("g", '{"steps": [{"id": "a", "task": "do something"}]}')
    assert plan.steps[0].writes is True


def test_only_an_explicit_false_makes_a_step_read_only():
    plan = parse_plan("g", '{"steps": [{"id": "a", "task": "t", "writes": "no"},'
                           '{"id": "b", "task": "t", "writes": false}]}')
    assert plan.by_id("a").writes is True
    assert plan.by_id("b").writes is False


def test_a_dependency_on_a_step_that_does_not_exist_is_dropped():
    plan = parse_plan("g", '{"steps": [{"id": "a", "task": "t", "needs": ["ghost"]}]}')
    assert plan.steps[0].needs == ()
    plan.validate()


def test_a_cycle_in_a_planners_output_is_flattened_not_fatal():
    plan = parse_plan("g", '{"steps": [{"id": "a", "task": "t", "needs": ["b"]},'
                           '{"id": "b", "task": "t", "needs": ["a"]}]}')
    plan.validate()
    assert all(s.needs == () for s in plan)


def test_a_plan_is_capped():
    steps = ", ".join('{"id": "s%d", "task": "t"}' % i for i in range(40))
    assert len(parse_plan("g", '{"steps": [%s]}' % steps)) <= MAX_STEPS


def test_a_step_with_no_task_is_ignored():
    plan = parse_plan("g", '{"steps": [{"id": "a"}, {"id": "b", "task": "real"}]}')
    assert [s.id for s in plan] == ["b"]


# -- the graph ---------------------------------------------------------------


def test_waves_group_what_can_start_together():
    plan = Plan("g", [
        Step("a", "t"), Step("b", "t"),
        Step("c", "t", needs=("a", "b")),
        Step("d", "t", needs=("c",)),
    ])
    assert [[s.id for s in wave] for wave in plan.waves()] == [["a", "b"], ["c"], ["d"]]


def test_a_real_cycle_is_refused_with_the_names_in_it():
    plan = Plan("g", [Step("a", "t", needs=("b",)), Step("b", "t", needs=("a",))])
    with pytest.raises(PlanError, match="cycle"):
        plan.validate()


def test_a_duplicate_id_is_refused():
    with pytest.raises(PlanError, match="share the id"):
        Plan("g", [Step("a", "t"), Step("a", "t")]).validate()


def test_a_step_that_needs_itself_is_refused():
    with pytest.raises(PlanError, match="itself"):
        Plan("g", [Step("a", "t", needs=("a",))]).validate()


def test_an_empty_plan_is_refused():
    with pytest.raises(PlanError):
        Plan("g", []).validate()


# -- running it --------------------------------------------------------------


def test_different_steps_land_on_different_models(roster):
    plan = Plan("g", [
        Step("survey", "t", "planner", writes=False),
        Step("build", "t", "implementer", needs=("survey",)),
        Step("review", "t", "critic", needs=("build",), writes=False),
    ])
    work = scripted()
    run = run_workflow(plan, roster, work)
    assert run.ok
    models = {step.id: step.model for step in run.steps}
    assert len(set(models.values())) == 3, f"three roles must reach three models: {models}"


def test_independent_read_only_steps_really_run_at_the_same_time(roster):
    """Not "could": two steps must actually overlap in time."""
    plan = Plan("g", [Step("a", "t", "planner", writes=False),
                      Step("b", "t", "critic", writes=False)])
    live = 0
    peak = 0
    lock = threading.Lock()

    def work(step, seat, briefing):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.15)
        with lock:
            live -= 1
        return StepResult(text="done")

    run_workflow(plan, roster, work)
    assert peak == 2, "two read-only steps in one wave must overlap"


def test_steps_that_write_never_overlap(roster):
    """They share one working tree, so they have to take turns."""
    plan = Plan("g", [Step("a", "t", "planner"), Step("b", "t", "critic")])
    live = 0
    peak = 0
    lock = threading.Lock()

    def work(step, seat, briefing):
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.1)
        with lock:
            live -= 1
        return StepResult(text="done")

    run_workflow(plan, roster, work)
    assert peak == 1, "two writing steps overlapped; they share the workspace"


def test_a_step_is_told_what_upstream_produced(roster):
    plan = Plan("g", [Step("first", "t"), Step("second", "t", needs=("first",))])
    briefings: dict[str, str] = {}

    def work(step, seat, briefing):
        briefings[step.id] = briefing
        return StepResult(text=f"result of {step.id}")

    run_workflow(plan, roster, work)
    assert "result of first" in briefings["second"], "downstream must see upstream's output"
    assert "result of first" not in briefings["first"]


def test_a_read_only_step_is_told_so(roster):
    plan = Plan("g", [Step("look", "t", writes=False)])
    briefings: list[str] = []
    run_workflow(plan, roster, lambda s, seat, b: briefings.append(b) or StepResult(text="x"))
    assert "read-only" in briefings[0]


def test_a_failure_skips_what_depended_on_it(roster):
    plan = Plan("g", [
        Step("build", "t"),
        Step("test", "t", needs=("build",)),
        Step("unrelated", "t", writes=False),
    ])
    run = run_workflow(plan, roster, scripted(build=StepResult(error="compiler exploded")))
    assert run.plan.by_id("build").state == FAILED
    assert run.plan.by_id("test").state == SKIPPED
    assert run.plan.by_id("unrelated").state == DONE, "an unrelated step must still run"
    assert not run.ok


def test_a_worker_that_raises_fails_only_its_own_step(roster):
    plan = Plan("g", [Step("boom", "t"), Step("fine", "t", writes=False)])

    def work(step, seat, briefing):
        if step.id == "boom":
            raise RuntimeError("kaboom")
        return StepResult(text="survived")

    run = run_workflow(plan, roster, work)
    assert run.plan.by_id("boom").state == FAILED
    assert "kaboom" in (run.plan.by_id("boom").error or "")
    assert run.plan.by_id("fine").state == DONE


def test_nothing_is_left_pending(roster):
    plan = Plan("g", [Step("a", "t"), Step("b", "t", needs=("a",))])
    run = run_workflow(plan, roster, scripted(a=StepResult(error="no")))
    assert all(s.state != PENDING for s in run.steps)


# -- adapting ----------------------------------------------------------------


def test_the_planner_rewrites_the_tail_after_a_failure(roster):
    plan = Plan("g", [Step("build", "t"), Step("test", "t", needs=("build",))])

    def revise(run, pending):
        assert [s.id for s in pending] == ["test"], "only unstarted steps may be replaced"
        return [Step("repair", "fix the build", "implementer"),
                Step("retest", "test again", "critic", needs=("repair",))]

    run = run_workflow(plan, roster, scripted(build=StepResult(error="broken")), revise=revise)
    assert [s.id for s in run.steps] == ["build", "repair", "retest"]
    assert run.plan.by_id("retest").state == DONE


def test_a_revision_happens_once_per_failure_not_once_per_wave(roster):
    """`run.failed` never empties, so reacting to it re-planned forever."""
    calls = {"n": 0}

    def revise(run, pending):
        calls["n"] += 1
        return [Step("repair", "fix it", "implementer"), Step("after", "then this", "critic",
                                                             needs=("repair",))]

    plan = Plan("g", [Step("build", "t"), Step("test", "t", needs=("build",))])
    run_workflow(plan, roster, scripted(build=StepResult(error="broken")), revise=revise)
    assert calls["n"] == 1


def test_revisions_are_bounded(roster):
    plan = Plan("g", [Step("a", "t"), Step("b", "t", needs=("a",)), Step("c", "t", needs=("b",))])
    calls = {"n": 0}

    def revise(run, pending):
        calls["n"] += 1
        return [Step(f"fresh{calls['n']}", "t", "implementer")]

    run_workflow(plan, roster, lambda s, seat, b: StepResult(error="always fails"),
                 revise=revise, max_revisions=1)
    assert calls["n"] == 1


def test_completed_work_is_never_rewritten(roster):
    plan = Plan("g", [Step("kept", "t", writes=False), Step("gone", "t"),
                      Step("later", "t", needs=("gone",))])
    run = run_workflow(plan, roster, scripted(gone=StepResult(error="broken")),
                       revise=lambda run, pending: [Step("replacement", "t", "implementer")])
    ids = [s.id for s in run.steps]
    assert "kept" in ids and run.plan.by_id("kept").state == DONE
    assert "replacement" in ids


def test_a_dangling_dependency_in_a_revision_is_repaired_not_discarded(roster):
    """Dropping one bad edge keeps the rethink; throwing it away wastes the call."""
    plan = Plan("g", [Step("a", "t"), Step("b", "t", needs=("a",))])
    run = run_workflow(plan, roster, scripted(a=StepResult(error="broken")),
                       revise=lambda run, pending: [Step("x", "t", needs=("nonexistent",))])
    assert "x" in [s.id for s in run.steps]
    assert run.plan.by_id("x").needs == ()
    assert run.plan.by_id("x").state == DONE


def test_a_revision_that_cannot_be_repaired_is_reported_and_ignored(roster):
    plan = Plan("g", [Step("a", "t"), Step("b", "t", needs=("a",))])
    run = run_workflow(plan, roster, scripted(a=StepResult(error="broken")),
                       revise=lambda run, pending: [Step("", "a step with no id")])
    assert any("unusable" in note for note in run.notes), run.notes
    assert [s.id for s in run.steps] == ["a", "b"], "the original plan must stand"


def test_a_reviser_that_declines_changes_nothing(roster):
    plan = Plan("g", [Step("a", "t"), Step("b", "t", needs=("a",))])
    run = run_workflow(plan, roster, scripted(a=StepResult(error="broken")),
                       revise=lambda run, pending: None)
    assert [s.id for s in run.steps] == ["a", "b"]
    assert run.revisions == 0


# -- the read-only guarantee -------------------------------------------------


def test_a_read_only_toolbox_has_no_writing_tool_in_it():
    """Enforced, not requested: the planner's claim is not trusted."""
    full = Toolbox(builtin_tools())
    limited = readonly_toolbox(full)
    assert limited.names(), "it must still be able to read"
    assert all(tool.danger <= Danger.SAFE for tool in limited)
    dropped = set(full.names()) - set(limited.names())
    assert "write" in dropped and "bash" in dropped, f"still writable: {sorted(limited.names())}"


def test_reading_tools_survive_the_filter():
    limited = readonly_toolbox(Toolbox(builtin_tools()))
    for expected in ("read", "grep", "glob", "list"):
        assert expected in limited.names(), f"{expected} is not a writing tool"


# -- reporting ---------------------------------------------------------------


def test_the_report_names_every_step_its_model_and_its_fate(roster):
    plan = Plan("g", [Step("a", "t", writes=False), Step("b", "t", needs=("a",))])
    run = run_workflow(plan, roster, scripted(b=StepResult(error="nope")))
    body = "\n".join(run.report())
    assert "a" in body and "b" in body
    assert "opus" in body or "haiku" in body or "sonnet" in body
    assert "fail" in body and "ok" in body


def test_the_outline_shows_what_runs_together():
    plan = Plan("g", [Step("a", "t", writes=False), Step("b", "t", writes=False),
                      Step("c", "t", needs=("a", "b"))])
    body = "\n".join(plan.outline())
    assert "2 at once" in body
    assert "read" in body and "edit" in body
