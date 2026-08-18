"""Several models working one task together, on a plan they wrote themselves.

The other two multi-model features here answer a question (`Ensemble.council`)
or try the whole task N ways in isolation (`branches`). This one is different: a
planner model decomposes the work into steps, says which role should do each and
what it depends on, and the runner executes that graph - independent steps at the
same time, on different models, against the same repository.

Three properties are worth stating because they are what make it usable rather
than a demo:

**The shape comes from the model, not from this file.** There is no fixed
planner-implementer-critic pipeline. The plan is a dependency graph produced at
runtime, so a refactor gets a different shape from a bug hunt.

**It adapts.** After every wave the runner offers the planner what actually
happened. A failed step can be replaced, retried differently, or abandoned, and
steps that were never reached can be rewritten. That is the whole reason the plan
is data rather than control flow.

**Read-only steps are read-only by construction.** Steps that only need to look
at the repository run concurrently, and they are handed a toolbox with the
writing tools removed - so "these can safely run together" is enforced here
rather than trusted to whatever the planner claimed. Steps that write run one at
a time, in declaration order, because they share one working tree.
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Iterator, Sequence

from offset.core.multimodel import Ensemble, Seat
from offset.tools.base import Danger, Tool, Toolbox

#: Step lifecycle.  A step is `skipped` when something it needed failed.
PENDING: str = "pending"
RUNNING: str = "running"
DONE: str = "done"
FAILED: str = "failed"
SKIPPED: str = "skipped"

#: How many times the planner may rewrite the remainder of a plan.
MAX_REVISIONS: int = 2

#: Hard ceiling on plan size, so a runaway planner cannot spend the afternoon.
MAX_STEPS: int = 12


class PlanError(ValueError):
    """A plan that cannot be executed as written."""


@dataclass(slots=True)
class Step:
    """One unit of work, and what became of it."""

    id: str
    task: str
    role: str = "implementer"
    needs: tuple[str, ...] = ()
    #: Whether the step may modify the workspace. Defaults to True: assuming a
    #: step is read-only when it is not would let it race another writer.
    writes: bool = True

    state: str = PENDING
    model: str = ""
    text: str = ""
    error: str | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.state == DONE

    def summary(self, width: int = 72) -> str:
        body = (self.text or self.error or "").strip().replace("\n", " ")
        return body[:width]


@dataclass(slots=True)
class Plan:
    """A dependency graph of steps."""

    goal: str
    steps: list[Step] = field(default_factory=list)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def by_id(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def validate(self) -> None:
        """Refuse a graph that cannot be run, naming the reason."""
        if not self.steps:
            raise PlanError("a plan with no steps")
        seen: set[str] = set()
        for step in self.steps:
            if not step.id:
                raise PlanError("a step with no id")
            if step.id in seen:
                raise PlanError(f"two steps share the id {step.id!r}")
            seen.add(step.id)
        for step in self.steps:
            for need in step.needs:
                if need == step.id:
                    raise PlanError(f"{step.id} depends on itself")
                if need not in seen:
                    raise PlanError(f"{step.id} depends on {need!r}, which is not in the plan")
        self.waves()  # raises on a cycle

    def waves(self) -> list[list[Step]]:
        """Steps grouped so everything in a group can start at once."""
        remaining = {s.id: set(s.needs) for s in self.steps}
        order = {s.id: i for i, s in enumerate(self.steps)}
        settled: set[str] = set()
        out: list[list[Step]] = []
        while remaining:
            ready = sorted((sid for sid, needs in remaining.items() if needs <= settled),
                           key=lambda sid: order[sid])
            if not ready:
                stuck = ", ".join(sorted(remaining))
                raise PlanError(f"these steps depend on each other in a cycle: {stuck}")
            out.append([self.by_id(sid) for sid in ready])  # type: ignore[misc]
            settled |= set(ready)
            for sid in ready:
                del remaining[sid]
        return out

    def outline(self) -> list[str]:
        """What the plan is, for a person about to authorise it."""
        lines: list[str] = []
        for depth, wave in enumerate(self.waves(), 1):
            together = "" if len(wave) == 1 else f"  ({len(wave)} at once)"
            lines.append(f"wave {depth}{together}")
            for step in wave:
                mark = "read" if not step.writes else "edit"
                needs = f" after {'+'.join(step.needs)}" if step.needs else ""
                lines.append(f"  {step.id:<10} {step.role:<12} {mark}  {step.task[:52]}{needs}")
        return lines


# -- turning a model's reply into a plan --------------------------------------

PLAN_SYSTEM: str = """You plan work for a team of coding models sharing one repository.

Reply with JSON only: {"steps": [...]}. Each step is an object with
  id      short slug, unique, referenced by other steps
  task    one imperative sentence, specific enough to act on alone
  role    planner | implementer | critic
  needs   list of step ids that must finish first (omit or [] if none)
  writes  true if the step edits files, false if it only reads them

Rules that matter:
- Steps with no dependency between them run at the same time on different
  models, so put independent work in separate steps.
- Only mark writes:false when the step genuinely just inspects the repository.
  A read-only step is given a toolbox with no writing tools at all.
- Prefer 2 to 5 steps. One step is a perfectly good plan for a small task.
- Never invent a verification step that runs a command you were not told exists.
"""


def plan_prompt(goal: str, *, roles: Sequence[str] = ()) -> str:
    seats = f"\nRoles actually available: {', '.join(roles)}." if roles else ""
    return f"Task: {goal}{seats}\n\nReply with the JSON plan."


def parse_plan(goal: str, reply: str) -> Plan:
    """Read a plan out of a model's reply.

    Models wrap JSON in prose, fences and apologies, so the first balanced
    object wins. A reply with no usable plan is not an error: the honest reading
    of "I could not decompose this" is a single step that does the whole job.
    """
    raw = _first_object(reply)
    steps: list[Step] = []
    if isinstance(raw, dict):
        listing = raw.get("steps")
        if isinstance(listing, list):
            for index, item in enumerate(listing[:MAX_STEPS]):
                step = _step_from(item, index)
                if step is not None:
                    steps.append(step)
    if not steps:
        return Plan(goal, [Step(id="do", task=goal, role="implementer")])

    known = {s.id for s in steps}
    for i, step in enumerate(steps):
        # A dependency on something that is not in the plan would make the whole
        # plan unrunnable, and it is always a planner slip rather than intent.
        steps[i] = replace(step, needs=tuple(n for n in step.needs if n in known and n != step.id))
    plan = Plan(goal, steps)
    try:
        plan.validate()
    except PlanError:
        # A cycle is the one thing that cannot be repaired by dropping edges
        # selectively without guessing which edge was meant, so flatten it.
        plan = Plan(goal, [replace(s, needs=()) for s in steps])
        plan.validate()
    return plan


def _step_from(item: Any, index: int) -> Step | None:
    if not isinstance(item, dict):
        return None
    task = str(item.get("task") or item.get("goal") or "").strip()
    if not task:
        return None
    needs = item.get("needs") or item.get("after") or ()
    if isinstance(needs, str):
        needs = [needs]
    writes = item.get("writes")
    return Step(
        id=str(item.get("id") or f"step{index + 1}").strip()[:24] or f"step{index + 1}",
        task=task,
        role=str(item.get("role") or "implementer").strip().lower() or "implementer",
        needs=tuple(str(n).strip() for n in needs if str(n).strip()),
        # Anything other than an explicit false means it might write.
        writes=not (writes is False or str(writes).strip().lower() == "false"),
    )


def _first_object(text: str) -> Any:
    """The first balanced JSON object in `text`, or None."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    for candidate in ([fenced.group(1)] if fenced else []) + [text]:
        start = candidate.find("{")
        while start != -1:
            depth, in_string, escaped = 0, False, False
            for position in range(start, len(candidate)):
                char = candidate[position]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(candidate[start:position + 1])
                        except json.JSONDecodeError:
                            break
            start = candidate.find("{", start + 1)
    return None


# -- running one -------------------------------------------------------------


@dataclass(slots=True)
class StepResult:
    """What a worker made of a step."""

    text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


#: A worker runs one step on one seat.  The shell supplies a real one; a test
#: supplies a scripted one.  Anything to do with agents, tools and worktrees
#: lives behind this, which is why this module has no opinion about them.
Worker = Callable[[Step, Seat, str], StepResult]

#: Given the plan so far, produce replacement steps for whatever has not run.
Reviser = Callable[["WorkflowRun", Sequence[Step]], list[Step] | None]


@dataclass(slots=True)
class WorkflowRun:
    """A plan, its assignments, and what happened when it ran."""

    plan: Plan
    assigned: dict[str, Seat] = field(default_factory=dict)
    revisions: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def steps(self) -> list[Step]:
        return self.plan.steps

    @property
    def failed(self) -> list[Step]:
        return [s for s in self.steps if s.state == FAILED]

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.state in (DONE, SKIPPED) for s in self.steps) \
            and any(s.state == DONE for s in self.steps)

    def context_for(self, step: Step) -> str:
        """What upstream steps produced, as the step's briefing."""
        pieces: list[str] = [f"Overall goal: {self.plan.goal}"]
        for need in step.needs:
            upstream = self.plan.by_id(need)
            if upstream is None:
                continue
            if upstream.ok and upstream.text:
                pieces.append(f"--- what {need} ({upstream.model}) produced ---\n{upstream.text.strip()}")
            elif upstream.error:
                pieces.append(f"--- {need} failed: {upstream.error} ---")
        pieces.append(f"Your step: {step.task}")
        if not step.writes:
            pieces.append("You have read-only tools for this step; report, do not edit.")
        return "\n\n".join(pieces)

    def report(self) -> list[str]:
        lines = [f"goal: {self.plan.goal}", ""]
        for depth, wave in enumerate(self.plan.waves(), 1):
            lines.append(f"wave {depth}:")
            for step in wave:
                mark = {DONE: "ok", FAILED: "fail", SKIPPED: "skip",
                        RUNNING: "..", PENDING: "--"}.get(step.state, step.state)
                seat = step.model or "unassigned"
                lines.append(f"  {mark:<5} {step.id:<10} {seat:<24} {step.seconds:5.1f}s  {step.summary(46)}")
        if self.revisions:
            lines += ["", f"the planner revised the plan {self.revisions}x"]
        lines += self.notes
        return lines


def readonly_toolbox(source: Iterable[Tool]) -> Toolbox:
    """The same tools with everything that can change anything removed.

    This is what makes running steps concurrently safe. A planner that marks a
    step read-only is not trusted about it: the step is handed a box that has no
    writing tool in it, so it cannot collide with a sibling even if it tries.
    """
    return Toolbox([tool for tool in source if tool.danger <= Danger.SAFE])


def run_workflow(
    plan: Plan,
    ensemble: Ensemble,
    work: Worker,
    *,
    revise: Reviser | None = None,
    max_revisions: int = MAX_REVISIONS,
    parallel: bool = True,
) -> WorkflowRun:
    """Execute `plan`, adapting it as results come in.

    Read-only steps in the same wave run together; writing steps run one after
    another because they share a working tree. A step whose dependency failed is
    skipped rather than run against a broken premise.
    """
    plan.validate()
    run = WorkflowRun(plan=plan)
    lock = threading.Lock()
    finished: set[str] = set()

    while True:
        _seat_plan(run, ensemble)
        wave = _next_wave(run, finished)
        if not wave:
            break

        readers = [s for s in wave if not s.writes]
        writers = [s for s in wave if s.writes]

        if parallel and len(readers) > 1:
            with ThreadPoolExecutor(max_workers=min(len(readers), 8)) as pool:
                list(pool.map(lambda step: _run_step(run, step, work, lock), readers))
        else:
            for step in readers:
                _run_step(run, step, work, lock)
        for step in writers:
            _run_step(run, step, work, lock)

        finished |= {s.id for s in wave}

        # Only a failure from THIS wave asks for a rethink. `run.failed` keeps
        # every failure for the report, so reacting to it directly re-planned on
        # every later wave for a failure that had already been dealt with.
        fresh_failures = [s for s in wave if s.state == FAILED]
        if revise is not None and fresh_failures and run.revisions < max_revisions:
            pending = [s for s in run.steps if s.state == PENDING and s.id not in finished]
            replacement = revise(run, pending)
            if replacement:
                _apply_revision(run, replacement, finished)
                run.revisions += 1

    _skip_unreachable(run)
    return run


def _next_wave(run: WorkflowRun, finished: set[str]) -> list[Step]:
    """The next group of steps whose dependencies have all settled."""
    for wave in run.plan.waves():
        pending = [s for s in wave if s.state == PENDING and s.id not in finished]
        if not pending:
            continue
        ready = [s for s in pending if all(_settled(run, need, finished) for need in s.needs)]
        return ready or pending
    return []


def _settled(run: WorkflowRun, need: str, finished: set[str]) -> bool:
    upstream = run.plan.by_id(need)
    return upstream is None or upstream.id in finished or upstream.state in (DONE, FAILED, SKIPPED)


def _seat_plan(run: WorkflowRun, ensemble: Ensemble) -> None:
    """Give every unassigned step a model.

    Seating the whole plan at once is what spreads the roster across the run.
    Doing it wave by wave restarted the ensemble's least-used counter each time,
    so the same seat won every wave and the rest of the roster sat idle.
    """
    waiting = [step for step in run.steps if step.id not in run.assigned]
    if not waiting:
        return
    for step, (_role, seat) in zip(waiting, ensemble.staff([s.role for s in waiting])):
        run.assigned[step.id] = seat
        step.model = seat.model


def _run_step(run: WorkflowRun, step: Step, work: Worker, lock: threading.Lock) -> None:
    seat = run.assigned[step.id]
    if any((run.plan.by_id(n) or step).state in (FAILED, SKIPPED) for n in step.needs):
        step.state = SKIPPED
        step.error = "a step it depended on did not finish"
        return
    briefing = run.context_for(step)
    step.state = RUNNING
    started = time.monotonic()
    try:
        result = work(step, seat, briefing)
    except Exception as exc:  # a broken worker fails its step, never the run
        result = StepResult(error=f"{type(exc).__name__}: {exc}")
    step.seconds = time.monotonic() - started
    with lock:
        if result.ok:
            step.state, step.text = DONE, result.text
        else:
            step.state, step.error = FAILED, result.error
            step.text = result.text


def _apply_revision(run: WorkflowRun, replacement: Sequence[Step], finished: set[str]) -> None:
    """Swap the not-yet-run tail of the plan for something else.

    Completed steps are never touched - they already changed the repository, and
    a plan that rewrote history would be lying about what happened.
    """
    keep = [s for s in run.steps if s.state != PENDING or s.id in finished]
    kept_ids = {s.id for s in keep}
    fresh: list[Step] = []
    for step in replacement[:MAX_STEPS]:
        if step.id in kept_ids:
            continue
        kept_ids.add(step.id)
        fresh.append(replace(step, needs=tuple(n for n in step.needs if n in kept_ids or
                                               n in {f.id for f in fresh})))
    candidate = Plan(run.plan.goal, keep + fresh)
    try:
        candidate.validate()
    except PlanError as exc:
        run.notes.append(f"the planner's revision was unusable ({exc}); carried on with the original")
        return
    run.plan = candidate
    run.notes.append(f"replaced {len(run.steps) - len(keep)} remaining step(s) after a failure")


def _skip_unreachable(run: WorkflowRun) -> None:
    for step in run.steps:
        if step.state == PENDING:
            step.state = SKIPPED
            step.error = step.error or "never reached"
