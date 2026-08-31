"""Persistent tasks: the state machine, not the model.

Every worker here is a script, so these tests pin the behaviour that actually
matters and that a live model would obscure: what is on disk after each
transition, what happens when the process dies mid-stage, and whether the fix
loop can run forever.
"""

from __future__ import annotations

import json

import pytest

from offset.core import tasks
from offset.core.tasks import (
    BLOCKED,
    COMPLETE,
    DONE,
    FAILED,
    FIX,
    IMPLEMENT,
    PENDING,
    PLAN,
    REPORT,
    RUNNING,
    SKIPPED,
    STOPPED,
    TEST,
    Stage,
    Task,
    create,
    drive,
    listing,
    load,
    path_for,
    prune,
    resume,
    save,
    step,
)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    return tmp_path


def always_fine(task: Task, stage: Stage) -> tuple[str, str]:
    return f"{stage.name} ok", ""


def failing_tests(task: Task, stage: Stage) -> tuple[str, str]:
    """Tests never pass, so the fix loop must be what stops this."""
    if stage.name == TEST:
        return "1 failed", "the test suite failed"
    return f"{stage.name} ok", ""


def test_a_new_task_is_on_disk_before_anything_runs(home):
    task = create("add auth")
    assert path_for(task.id).exists(), "a task must survive the process that made it"
    reloaded = load(task.id)
    assert reloaded is not None
    assert reloaded.goal == "add auth"
    assert [s.name for s in reloaded.stages] == [PLAN, IMPLEMENT, TEST, REPORT]
    assert all(s.state == PENDING for s in reloaded.stages)


def test_a_clean_run_completes_every_stage(home):
    task = drive(create("ship it"), always_fine)
    assert task.state == COMPLETE
    assert [s.state for s in task.stages] == [DONE] * 4
    assert load(task.id).state == COMPLETE, "the terminal state must be persisted"


def test_each_stage_is_persisted_as_it_settles(home):
    task = create("one stage at a time")
    step(task, always_fine)
    on_disk = load(task.id)
    assert on_disk.stages[0].state == DONE
    assert on_disk.stages[1].state == PENDING, "only one stage may advance per step"


def test_a_failing_test_stage_inserts_a_fix_and_a_retest(home):
    task = create("fix me", verify="pytest")
    step(task, always_fine)   # plan
    step(task, always_fine)   # implement
    step(task, failing_tests)  # test -> fails
    names = [s.name for s in task.stages]
    assert names == [PLAN, IMPLEMENT, TEST, FIX, TEST, REPORT], names
    assert task.stage(TEST).state == FAILED
    assert task.state != STOPPED, "one failure is not the end; that is what the loop is for"


def test_the_fix_loop_stops_at_its_bound_instead_of_running_forever(home):
    task = drive(create("impossible", max_fix=2), failing_tests, limit=50)
    assert task.state == STOPPED
    assert task.fixes == 2, f"spent {task.fixes} fixes against a ceiling of 2"
    blocked = [s for s in task.stages if s.state == BLOCKED]
    assert blocked, "the stage that ran out of attempts must say so"
    assert "2 fix attempt" in task.error


def test_stages_after_a_hard_failure_are_skipped_not_left_pending(home):
    def broken_implement(task: Task, stage: Stage) -> tuple[str, str]:
        if stage.name == IMPLEMENT:
            return "", "could not write the file"
        return "ok", ""

    task = drive(create("doomed"), broken_implement)
    assert task.state == STOPPED
    assert task.stage(IMPLEMENT).state == FAILED
    assert task.stage(REPORT).state == SKIPPED, "a skipped stage is not a pending one"


def test_resuming_does_not_rerun_a_completed_stage(home):
    task = create("resume me")
    step(task, always_fine)  # plan is done

    ran: list[str] = []

    def counting(task: Task, stage: Stage) -> tuple[str, str]:
        ran.append(stage.name)
        return "ok", ""

    resumed, why = resume(task.id, counting)
    assert why == ""
    assert resumed.state == COMPLETE
    assert PLAN not in ran, f"plan was re-run: {ran}"
    assert ran == [IMPLEMENT, TEST, REPORT], ran


def test_resuming_retries_a_stage_the_crash_left_running(home):
    """A process that died mid-stage recorded no outcome, so it did not happen."""
    task = create("crashed")
    task.stages[0].state = RUNNING
    task.stages[0].attempts = 1
    save(task)

    ran: list[str] = []

    def counting(task: Task, stage: Stage) -> tuple[str, str]:
        ran.append(stage.name)
        return "ok", ""

    resumed, why = resume(task.id, counting)
    assert why == ""
    assert PLAN in ran, "an interrupted stage must be tried again"
    assert resumed.state == COMPLETE


def test_resuming_an_unknown_task_is_an_error_not_a_crash(home):
    task, why = resume("NOSUCHTASK", always_fine)
    assert task is None
    assert "NOSUCHTASK" in why


def test_resuming_a_finished_task_says_so_and_changes_nothing(home):
    task = drive(create("already done"), always_fine)
    again, why = resume(task.id, always_fine)
    assert again is not None
    assert COMPLETE in why
    assert [s.state for s in again.stages] == [DONE] * 4


def test_a_worker_that_raises_fails_its_stage_rather_than_the_task_runner(home):
    def exploding(task: Task, stage: Stage) -> tuple[str, str]:
        raise RuntimeError("boom")

    task = step(create("explode"), exploding)
    assert task.stage(PLAN).state == FAILED
    assert "RuntimeError: boom" in task.stage(PLAN).error


def test_the_record_is_written_before_the_work_so_a_crash_is_visible(home):
    """`running` on disk during the work is what makes a crash detectable."""
    seen: list[str] = []

    def peeking(task: Task, stage: Stage) -> tuple[str, str]:
        seen.append(load(task.id).stages[0].state)
        return "ok", ""

    step(create("peek"), peeking)
    assert seen == [RUNNING], f"expected the stage to be persisted as running, saw {seen}"


def test_an_atomic_write_leaves_no_temp_files_behind(home):
    task = drive(create("tidy"), always_fine)
    leftovers = list(tasks.tasks_dir().glob(".*tmp"))
    assert not leftovers, f"temp files survived: {leftovers}"
    assert path_for(task.id).exists()


def test_a_corrupt_task_file_is_skipped_not_fatal(home):
    good = create("good")
    bad = tasks.tasks_dir() / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    found = listing()
    assert [t.id for t in found] == [good.id], "one bad file must not hide the rest"


def test_a_task_file_from_a_future_version_is_refused(home):
    task = create("future")
    raw = json.loads(path_for(task.id).read_text())
    raw["version"] = 999
    path_for(task.id).write_text(json.dumps(raw), encoding="utf-8")
    assert load(task.id) is None, "an unknown schema must not be half-read"


def test_listing_is_newest_first_and_filters_by_state(home):
    first = drive(create("older"), always_fine)
    second = create("newer")
    ids = [t.id for t in listing()]
    assert ids.index(second.id) < ids.index(first.id), "newest first"
    assert [t.id for t in listing(state=COMPLETE)] == [first.id]


def test_prune_keeps_the_newest_settled_tasks(home):
    for i in range(5):
        drive(create(f"task {i}"), always_fine)
    live = create("still going")
    removed = prune(keep=2)
    assert removed == 3
    remaining = {t.id for t in listing()}
    assert live.id in remaining, "an unfinished task must never be pruned"
    assert len(remaining) == 3


def test_report_names_the_stages_and_the_fix_budget(home):
    task = drive(create("report me", max_fix=1), failing_tests, limit=20)
    text = "\n".join(task.report())
    assert task.id in text
    assert TEST in text
    assert "fix attempts: 1/1" in text
