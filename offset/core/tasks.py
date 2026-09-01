"""Persistent multi-step tasks: plan, implement, test, fix, retest.

`core/workflow` already runs a plan as a wave DAG, and does it well, but it
lives entirely in memory: `ShellState.flow_run` is a field, so closing the
terminal loses the plan, the position in it, and every result. For a task that
means "implement auth, then keep going until the tests pass" that is the wrong
lifetime - the work outlasts the session.

So this module is not a second scheduler. It is a state machine whose every
transition is written to disk, and whose stages call into the existing
machinery. Three decisions:

**The file is the truth, not the object.** Every transition is persisted before
it is acted on, atomically, so a task interrupted between two stages resumes at
the boundary rather than repeating work or losing it. Resuming never re-runs a
stage that finished.

**The fix loop is bounded and the bound is visible.** `test -> fix -> retest`
with no ceiling is how an agent burns a budget overnight on an impossible
failure. Attempts are counted on the stage, the ceiling is a field, and
exhausting it is a terminal state that says so rather than a silence.

**A stage's work is injected.** The runner decides what happens next; it does
not know how to call a model or run a test suite. That keeps this file testable
without a network and lets `/task` supply real behaviour while a test supplies
a script.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final

from offset.core import settings
from offset.core.entries import new_id

#: Stage states.  `blocked` is distinct from `failed`: a blocked stage ran out
#: of attempts, which is a decision, while a failed one is the outcome of the
#: attempt it just made and may still be retried.
PENDING: Final = "pending"
RUNNING: Final = "running"
DONE: Final = "done"
FAILED: Final = "failed"
BLOCKED: Final = "blocked"
SKIPPED: Final = "skipped"

#: Task states.
ACTIVE: Final = "active"
COMPLETE: Final = "complete"
STOPPED: Final = "stopped"

PLAN: Final = "plan"
IMPLEMENT: Final = "implement"
TEST: Final = "test"
FIX: Final = "fix"
REPORT: Final = "report"

#: The default pipeline.  `fix` re-enters `test`, which is why the sequence is
#: expressed as stages plus a loop rather than a plain list.
PIPELINE: Final = (PLAN, IMPLEMENT, TEST, REPORT)

#: How many times to fix and retest before giving up.  Three is enough for a
#: typo, a missing import and a wrong assertion; beyond that the model is
#: usually not converging and a human should look.
MAX_FIX: Final = 3

VERSION: Final = 1


@dataclass(slots=True)
class Stage:
    """One step, and everything known about how it went."""

    name: str
    state: str = PENDING
    attempts: int = 0
    output: str = ""
    error: str = ""
    started: float = 0.0
    finished: float = 0.0

    @property
    def settled(self) -> bool:
        return self.state in (DONE, SKIPPED, BLOCKED)

    @property
    def seconds(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "attempts": self.attempts,
            "output": self.output,
            "error": self.error,
            "started": round(self.started, 6),
            "finished": round(self.finished, 6),
        }

    @classmethod
    def from_json(cls, raw: Any) -> Stage:
        if not isinstance(raw, dict):
            return cls(name="?", state=FAILED, error="unreadable stage record")
        return cls(
            name=str(raw.get("name") or "?"),
            state=str(raw.get("state") or PENDING),
            attempts=int(raw.get("attempts") or 0),
            output=str(raw.get("output") or ""),
            error=str(raw.get("error") or ""),
            started=float(raw.get("started") or 0.0),
            finished=float(raw.get("finished") or 0.0),
        )

    def line(self) -> str:
        mark = {DONE: "+", FAILED: "!", BLOCKED: "x", SKIPPED: "-", RUNNING: ">"}.get(self.state, " ")
        tail = f"  {self.seconds:.1f}s" if self.started else ""
        note = f"  {self.error[:50]}" if self.error else ""
        tries = f"  x{self.attempts}" if self.attempts > 1 else ""
        return f" {mark} {self.name:10s} {self.state:8s}{tries}{tail}{note}"


@dataclass(slots=True)
class Task:
    """A goal, the stages it decomposes into, and where it has got to."""

    id: str = field(default_factory=new_id)
    goal: str = ""
    cwd: str = ""
    state: str = ACTIVE
    stages: list[Stage] = field(default_factory=list)
    max_fix: int = MAX_FIX
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    error: str = ""
    verify: str = ""

    @property
    def current(self) -> Stage | None:
        """The next stage that still has work in it."""
        for stage in self.stages:
            if not stage.settled:
                return stage
        return None

    @property
    def finished(self) -> bool:
        return self.state in (COMPLETE, STOPPED)

    @property
    def fixes(self) -> int:
        """How many fix cycles have been entered.

        Counts the stages themselves, not their attempts: a fix stage is
        inserted with `attempts == 0` and only increments when it runs, so
        summing attempts reported zero at the moment the ceiling had to be
        tested and the loop inserted pairs forever.
        """
        return sum(1 for s in self.stages if s.name == FIX)

    def stage(self, name: str) -> Stage | None:
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "id": self.id,
            "goal": self.goal,
            "cwd": self.cwd,
            "state": self.state,
            "stages": [s.to_json() for s in self.stages],
            "max_fix": self.max_fix,
            "created": round(self.created, 6),
            "updated": round(self.updated, 6),
            "error": self.error,
            "verify": self.verify,
        }

    @classmethod
    def from_json(cls, raw: Any) -> Task | None:
        if not isinstance(raw, dict) or int(raw.get("version") or 0) != VERSION:
            return None
        stages = [Stage.from_json(s) for s in (raw.get("stages") or [])]
        return cls(
            id=str(raw.get("id") or new_id()),
            goal=str(raw.get("goal") or ""),
            cwd=str(raw.get("cwd") or ""),
            state=str(raw.get("state") or ACTIVE),
            stages=stages,
            max_fix=int(raw.get("max_fix") or MAX_FIX),
            created=float(raw.get("created") or time.time()),
            updated=float(raw.get("updated") or time.time()),
            error=str(raw.get("error") or ""),
            verify=str(raw.get("verify") or ""),
        )

    def report(self) -> list[str]:
        lines = [f"{self.id}  {self.state}", f"goal: {self.goal}"]
        lines.extend(s.line() for s in self.stages)
        if self.error:
            lines.append(f"error: {self.error}")
        if self.fixes:
            lines.append(f"fix attempts: {self.fixes}/{self.max_fix}")
        return lines

    def summary(self) -> str:
        done = sum(1 for s in self.stages if s.state == DONE)
        return f"{self.id[:10]}  {self.state:8s}  {done}/{len(self.stages)}  {self.goal[:44]}"


#: Does one stage's work.  Returns `(output, error)`; a non-empty error fails
#: the stage.  Injected so the runner needs neither a model nor a test suite.
Worker = Callable[[Task, Stage], tuple[str, str]]


def tasks_dir() -> Path:
    """Resolved on every call: `OFFSET_HOME` moves under tests and `--home`."""
    return settings.home() / "tasks"


def path_for(task_id: str) -> Path:
    return tasks_dir() / f"{task_id}.json"


def save(task: Task) -> Path:
    """Write the task out atomically.

    Temp file in the same directory then `os.replace`, so a crash mid-write
    leaves the previous state readable rather than a truncated file that would
    lose the whole task.
    """
    task.updated = time.time()
    target = path_for(task.id)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{task.id}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(task.to_json(), fh, indent=1)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def load(task_id: str) -> Task | None:
    """Read one task, or None if it is absent or unreadable."""
    target = path_for(task_id)
    if not target.exists():
        return None
    try:
        return Task.from_json(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def listing(*, state: str | None = None) -> list[Task]:
    """Every task, newest first."""
    root = tasks_dir()
    if not root.exists():
        return []
    found: list[Task] = []
    for entry in root.glob("*.json"):
        try:
            task = Task.from_json(json.loads(entry.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if task is None:
            continue
        if state is not None and task.state != state:
            continue
        found.append(task)
    return sorted(found, key=lambda t: t.updated, reverse=True)


def create(goal: str, *, cwd: Path | str = ".", verify: str = "", max_fix: int = MAX_FIX) -> Task:
    """A new task with the default pipeline, already on disk."""
    task = Task(
        goal=goal.strip(),
        cwd=str(Path(cwd).resolve()),
        verify=verify,
        max_fix=max_fix,
        stages=[Stage(name) for name in PIPELINE],
    )
    save(task)
    return task


def _insert_fix(task: Task) -> Stage:
    """Add a fix and a retest ahead of the report.

    The loop is expressed by growing the stage list rather than by jumping
    backwards, so the record shows every attempt that was made instead of
    overwriting the previous one. That is what makes a resumed task honest
    about how much has already been spent.
    """
    fix = Stage(FIX)
    retest = Stage(TEST)
    index = len(task.stages)
    for i, stage in enumerate(task.stages):
        if stage.name == REPORT and not stage.settled:
            index = i
            break
    task.stages[index:index] = [fix, retest]
    return fix


def step(task: Task, work: Worker) -> Task:
    """Advance the task by exactly one stage, persisting before and after.

    One stage per call so the caller keeps control: a shell can render between
    stages, a background job can check for cancellation, and a crash costs at
    most the stage that was in flight.
    """
    if task.finished:
        return task
    stage = task.current
    if stage is None:
        task.state = COMPLETE
        save(task)
        return task

    stage.state = RUNNING
    stage.attempts += 1
    stage.started = stage.started or time.time()
    save(task)  # persisted BEFORE the work, so a crash is visible as `running`

    try:
        output, error = work(task, stage)
    except Exception as exc:  # a worker fault is the stage's outcome, not a crash
        output, error = "", f"{type(exc).__name__}: {exc}"

    stage.output = output
    stage.error = error
    stage.finished = time.time()

    if not error:
        stage.state = DONE
        if task.current is None:
            task.state = COMPLETE
        save(task)
        return task

    # The test stage is the only one whose failure is expected and actionable.
    if stage.name == TEST:
        if task.fixes >= task.max_fix:
            stage.state = BLOCKED
            task.state = STOPPED
            task.error = f"tests still failing after {task.max_fix} fix attempt(s)"
            for later in task.stages:
                if later is not stage and not later.settled:
                    later.state = SKIPPED
            save(task)
            return task
        stage.state = FAILED
        _insert_fix(task)
        save(task)
        return task

    stage.state = FAILED
    task.state = STOPPED
    task.error = error
    for later in task.stages:
        # `stage` itself is FAILED, which is not a settled state, so without
        # this guard the loop below relabels the one stage that carries the
        # reason the task stopped.
        if later is not stage and not later.settled:
            later.state = SKIPPED
    save(task)
    return task


def drive(task: Task, work: Worker, *, limit: int = 24) -> Task:
    """Step until the task settles or `limit` stages have run.

    The limit is a stop against a worker that never settles a stage; it is not
    the fix ceiling, which `step` enforces separately.
    """
    for _ in range(max(1, limit)):
        if task.finished:
            break
        step(task, work)
    return task


def resume(task_id: str, work: Worker, *, limit: int = 24) -> tuple[Task | None, str]:
    """Continue a task from disk.  Completed stages are not re-run.

    A stage left `running` by a crash is returned to `pending` so it is tried
    again: the process died before recording an outcome, so the only honest
    reading is that the attempt did not happen.
    """
    task = load(task_id)
    if task is None:
        return None, f"no task {task_id!r}"
    if task.finished:
        return task, f"task {task_id} already {task.state}"
    for stage in task.stages:
        if stage.state == RUNNING:
            stage.state = PENDING
    save(task)
    return drive(task, work, limit=limit), ""


def prune(*, keep: int = 100) -> int:
    """Drop the oldest settled task files.  Returns how many were removed."""
    settled = [t for t in listing() if t.finished]
    if len(settled) <= keep:
        return 0
    removed = 0
    for task in settled[keep:]:
        try:
            path_for(task.id).unlink()
            removed += 1
        except OSError:
            continue
    return removed


# -- the shell surface ------------------------------------------------------


PLAN_SYSTEM: Final = """You are planning one coding task.
Reply with a short numbered list of the concrete edits required, no preamble.
Name files. If the task is one edit, say so in one line rather than inventing
steps to fill a list."""

FIX_SYSTEM: Final = """A test run failed. You are given its output.
Identify the root cause and make the smallest change that fixes it.
Do not restate the failure; change the code."""


def shell_worker(state: Any) -> Worker:
    """A worker that drives the real agent and the real test command.

    Built as a closure over the shell state rather than a method so the runner
    stays free of any knowledge of models, and so the tests can hand `step` a
    script instead.
    """
    from offset.core.agent import Agent, AgentConfig

    def work(task: Task, stage: Stage) -> tuple[str, str]:
        if stage.name == PLAN:
            result = state.agent.send(f"{PLAN_SYSTEM}\n\nTask: {task.goal}")
            return (result.text or "").strip(), result.error or ""

        if stage.name == IMPLEMENT:
            plan = task.stage(PLAN)
            brief = f"Task: {task.goal}"
            if plan is not None and plan.output:
                brief += f"\n\nYour plan:\n{plan.output}"
            brief += "\n\nMake the edits now."
            result = state.agent.send(brief)
            return (result.text or "").strip(), result.error or ""

        if stage.name == TEST:
            command = task.verify or state.verify_command
            if not command:
                return "no verify command configured; skipping the check", ""
            import subprocess

            try:
                proc = subprocess.run(
                    command, shell=True, cwd=task.cwd or None,
                    capture_output=True, text=True, timeout=600, errors="replace",
                )
            except subprocess.TimeoutExpired:
                return "", f"{command!r} did not finish within 600s"
            output = ((proc.stdout or "") + (proc.stderr or "")).strip()[-4000:]
            if proc.returncode == 0:
                return output or "tests passed", ""
            return output, f"{command!r} exited {proc.returncode}"

        if stage.name == FIX:
            failed = [s for s in task.stages if s.name == TEST and s.state == FAILED]
            evidence = failed[-1].output if failed else "the test run failed"
            result = state.agent.send(f"{FIX_SYSTEM}\n\n{evidence}")
            return (result.text or "").strip(), result.error or ""

        # report
        done = sum(1 for s in task.stages if s.state == DONE)
        return f"{done}/{len(task.stages)} stages completed", ""

    return work


def _task(state: Any, args: list[str]) -> Any:
    """`/task <goal>`, `/task resume <id>`."""
    from offset.auth import require_plus
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    if not require_plus("task"):
        return Outcome.error(
            "Offset Lite does not support /task.",
            "Upgrade to Offset Plus via 'offset upgrade <key>'.",
        )
    if not args:
        return Outcome.error("usage: /task <goal>", "or /task resume <id>")

    work = shell_worker(state)

    if args[0].lower() == "resume":
        if len(args) < 2:
            return Outcome.error("usage: /task resume <id>", "/tasks lists them")
        wanted = args[1]
        match = next((t for t in listing() if t.id == wanted or t.id.startswith(wanted)), None)
        if match is None:
            return Outcome.error(f"no task {wanted!r}", "/tasks lists them")

        def resume_job() -> Any:
            task, why = resume(match.id, work)
            if task is None:
                return Outcome.error(why)
            return Outcome(task.report(), TONE_OK if task.state == COMPLETE else TONE_INFO)

        return Outcome([f"resuming {match.id}..."], TONE_INFO, job=resume_job)

    goal = " ".join(args)
    created = create(goal, cwd=state.workspace, verify=state.verify_command or "")

    def job() -> Any:
        finished = drive(created, work)
        return Outcome(finished.report(), TONE_OK if finished.state == COMPLETE else TONE_INFO)

    return Outcome(
        [f"task {created.id[:10]}: {goal}", f"{len(created.stages)} stages, up to {created.max_fix} fixes"],
        TONE_INFO,
        job=job,
    )


def _tasks(state: Any, args: list[str]) -> Any:
    """`/tasks` - every task, newest first."""
    from offset.shell.commands import TONE_INFO, Outcome

    found = listing()
    if not found:
        return Outcome(["no tasks yet", "start one with /task <goal>"], TONE_INFO)
    return Outcome([t.summary() for t in found[:20]], TONE_INFO)


def task_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command("task", "plan, implement, test and fix until green", _task,
                usage="/task <goal> | /task resume <id>"),
        Command("tasks", "every task and where it got to", _tasks),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """`COMMANDS` on demand.

    Built lazily because the handlers import from `offset.shell.commands`,
    which imports this module: resolving at import time would be a cycle.

    The re-check after building is the same guard `offset.core.update` needs
    and for the same reason - importing the shell registry re-enters here
    before the outer call has stored anything, so a single access would
    otherwise produce two lists and register every command twice.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = task_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
