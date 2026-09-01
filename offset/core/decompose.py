"""Splitting one feature across an architect and a pool of coders.

`core/workflow` already runs a plan as a wave DAG and `core/tasks` already keeps
a plan's progress on disk, so this is not a third scheduler. What neither of
them has is a **contract between the units of work**, and that contract is the
only reason parallel coders are safe to run at all. Two failures motivate every
line here, and both were observed rather than imagined:

**Two coders editing one file.** Each reads the file, each writes it back, and
the second write lands on content it never saw. Nothing errors; the first
coder's work is simply gone. So a unit declares the files it will touch, and any
two units naming the same file are serialised at *planning* time - before a
single agent starts, while the fix is still a graph edge rather than a lost
diff.

**A consumer starting before its provider.** A unit that imports
`AuthToken` cannot run next to the unit that is still writing it; it fails on an
ImportError, or worse, invents its own incompatible version. So units name the
interfaces they `provides` and `consumes`, and the edge is derived from those
names rather than trusted to the architect remembering to also write `needs`.

Three consequences worth stating:

  * **The plan is data.** `parse_plan` gives you something inspectable and
    refusable before anything runs, and it refuses what it cannot read rather
    than inventing a plan - a decomposition that silently collapses to one unit
    has thrown away the only thing the user asked for.
  * **A cycle is refused, never half-run.** Flattening a cycle, which
    `core/workflow` does deliberately for its own smaller plans, would here mean
    running a consumer before its provider: exactly the bug the contract exists
    to prevent. So the cycle is named and the plan is rejected.
  * **Failure is isolated to a subtree.** A failed unit blocks the units that
    consume what it was going to provide, with the reason attached, and nothing
    else. A unit that merely shares a *file* with a failure is not blocked: it
    needed the other one to not be running, not to have succeeded.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Iterator, Mapping, Sequence

from offset.core.multimodel import Ensemble, Seat
# The state vocabulary is `core/tasks`', not a private copy: `blocked` there
# already means "a decision was taken not to run this", which is exactly what a
# unit whose provider failed is, and two modules disagreeing about the spelling
# of "done" is how a UI ends up with two legends.
from offset.core.tasks import BLOCKED, DONE, FAILED, PENDING, RUNNING, SKIPPED
# The scanner that pulls JSON out of prose, fences and apologies already exists
# and is already exercised by workflow's tests. A second copy would drift from
# it the first time a model found a new way to wrap an object, so the private
# name is imported on purpose rather than duplicated.
from offset.core.workflow import PlanError, _first_object

#: Hard ceiling on units in one plan. Twelve matches `workflow.MAX_STEPS` for
#: the same reason - a runaway architect must not spend the afternoon - and adds
#: one of its own: past a dozen units the interfaces between them stop being
#: describable in a single architect reply, so the contract degrades into
#: guesswork long before the budget bites.
MAX_UNITS: Final = 12

#: Coders running at once. Four, not the ensemble's eight: every coder here is a
#: full agent doing tool calls against one working tree, so the limit is review
#: bandwidth and disk contention rather than provider latency.
DEFAULT_WORKERS: Final = 4

#: Roles that may be given code to write, best first. The architect's own role
#: is deliberately last: spending the planner model on boilerplate is how a run
#: gets expensive without getting better.
CODER_ROLES: Final = ("implementer", "bulk", "cheap", "critic", "planner")

#: The role an architect seat should have. `default_roster` copies
#: `ModelInfo.role_hint` onto every seat, so this is the catalogue's own opinion
#: about which model plans well, not a second table to keep in step.
ARCHITECT_ROLE: Final = "planner"

_WHITE: Final = 0
_GREY: Final = 1
_BLACK: Final = 2


# -- the plan as data --------------------------------------------------------


def _texts(value: Any) -> tuple[str, ...]:
    """A tuple of non-empty strings from whatever the caller or model supplied.

    A bare string is treated as a one-item list because models write
    `"consumes": "AuthToken"` roughly as often as they write the list form, and
    refusing the plan over that would cost a whole architect round trip.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _norm_file(path: str) -> str:
    """One spelling per file, so collision detection cannot be fooled.

    `./core/x.py`, `core/x.py` and `core\\x.py` are the same file to the file
    system but three different dictionary keys, and a missed key means two
    coders writing one file at once.
    """
    cleaned = str(path).strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return "/".join(part for part in cleaned.split("/") if part not in ("", "."))


@dataclass(slots=True)
class Unit:
    """One coder's whole job, and the contract it signs with its siblings."""

    id: str
    task: str
    #: Every file this unit may touch. Two units sharing one entry are
    #: serialised; a unit that declares nothing cannot be protected, which is
    #: why `Graph.notes` says so out loud.
    files: tuple[str, ...] = ()
    #: Interfaces this unit creates. A consumer waits for all of them.
    provides: tuple[str, ...] = ()
    #: Interfaces this unit needs to exist before it starts.
    consumes: tuple[str, ...] = ()
    #: Ordering the interfaces do not express. Usually empty.
    needs: tuple[str, ...] = ()
    role: str = "implementer"

    state: str = PENDING
    model: str = ""
    text: str = ""
    error: str | None = None
    seconds: float = 0.0

    def __post_init__(self) -> None:
        # Normalised here rather than in the parser so a hand-built plan - the
        # shell's, a test's - is subject to exactly the same collision rules as
        # one a model wrote.
        self.id = str(self.id).strip()
        self.task = str(self.task).strip()
        self.role = (str(self.role).strip().lower() or "implementer")
        self.files = tuple(dict.fromkeys(f for f in (_norm_file(p) for p in _texts(self.files)) if f))
        self.provides = _texts(self.provides)
        self.consumes = _texts(self.consumes)
        self.needs = _texts(self.needs)

    @property
    def ok(self) -> bool:
        return self.state == DONE

    @property
    def settled(self) -> bool:
        return self.state in (DONE, FAILED, BLOCKED, SKIPPED)

    def summary(self, width: int = 66) -> str:
        body = (self.text or self.error or self.task).strip().replace("\n", " ")
        return body[:width]


@dataclass(frozen=True, slots=True)
class Graph:
    """The edges derived from a plan, separated by what they mean.

    Keeping the two apart is not tidiness. A data dependency failing must block
    its dependent, while a file-collision edge failing must not: the later unit
    needed the earlier one to not be *running*, and a failure satisfies that
    just as well as a success does. Merging the two sets would blame a whole
    subtree for a file it merely shares.
    """

    #: id -> ids whose output it needs (`needs` plus resolved `consumes`).
    needs: Mapping[str, tuple[str, ...]]
    #: `needs` plus file-collision edges. Scheduling reads this one.
    order: Mapping[str, tuple[str, ...]]
    #: What resolution had to decide, in the words an operator needs.
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class Plan:
    """A dependency graph of units, inspectable before anything runs."""

    goal: str
    units: list[Unit] = field(default_factory=list)

    def __iter__(self) -> Iterator[Unit]:
        return iter(self.units)

    def __len__(self) -> int:
        return len(self.units)

    def by_id(self, unit_id: str) -> Unit | None:
        return next((u for u in self.units if u.id == unit_id), None)

    def providers_of(self, interface: str) -> list[Unit]:
        return [u for u in self.units if interface in u.provides]

    def validate(self) -> Graph:
        """Refuse an unrunnable plan, naming the reason, and return its graph.

        Returning the graph rather than None is what stops `plan_waves` and
        `execute` from each having their own idea of what the edges are.
        """
        if not self.units:
            raise PlanError("a plan with no units")
        seen: set[str] = set()
        for unit in self.units:
            if not unit.id:
                raise PlanError("a unit with no id")
            if unit.id in seen:
                raise PlanError(f"two units share the id {unit.id!r}")
            if not unit.task:
                raise PlanError(f"unit {unit.id!r} has no task")
            seen.add(unit.id)
        return resolve(self)

    def graph(self) -> Graph:
        return self.validate()

    def outline(self) -> list[str]:
        """What the plan is, for a person about to authorise it."""
        graph = self.validate()
        lines: list[str] = [f"goal: {self.goal}"] if self.goal else []
        for wave in _layer(self.units, graph.order):
            together = "" if len(wave) == 1 else f"  ({len(wave)} at once)"
            lines.append(f"wave {wave.depth}{together}")
            for unit in wave:
                after = graph.order.get(unit.id, ())
                tail = f"  after {'+'.join(after)}" if after else ""
                lines.append(f"  {unit.id:<14} {unit.role:<12} {unit.task[:44]}{tail}")
                if unit.files:
                    lines.append(f"  {'':<14} files: {', '.join(unit.files)}")
                if unit.provides:
                    lines.append(f"  {'':<14} provides: {', '.join(unit.provides)}")
        lines.extend(f"note: {note}" for note in graph.notes)
        return lines


@dataclass(frozen=True, slots=True)
class Wave:
    """Units that may all start at once, and nothing that may not."""

    depth: int
    units: tuple[Unit, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(u.id for u in self.units)

    @property
    def parallel(self) -> bool:
        return len(self.units) > 1

    def __iter__(self) -> Iterator[Unit]:
        return iter(self.units)

    def __len__(self) -> int:
        return len(self.units)


# -- resolving the contract into edges ---------------------------------------


def _find_cycle(edges: Mapping[str, Sequence[str]], ids: Sequence[str]) -> list[str]:
    """One concrete cycle, as the path that closes it.

    Listing the members of a strongly connected clump is not enough to repair a
    plan: with five units tangled together the operator needs the actual loop,
    so this returns `[a, b, a]` and the caller renders the arrows.
    """
    colour: dict[str, int] = {}
    stack: list[str] = []

    def walk(node: str) -> list[str] | None:
        colour[node] = _GREY
        stack.append(node)
        for nxt in edges.get(node, ()):
            shade = colour.get(nxt, _WHITE)
            if shade == _GREY:
                return stack[stack.index(nxt):] + [nxt]
            if shade == _WHITE:
                found = walk(nxt)
                if found is not None:
                    return found
        stack.pop()
        colour[node] = _BLACK
        return None

    for start in ids:
        if colour.get(start, _WHITE) == _WHITE:
            found = walk(start)
            if found is not None:
                return found
    return []


def _refuse_cycle(edges: Mapping[str, tuple[str, ...]], ids: Sequence[str]) -> None:
    cycle = _find_cycle(edges, ids)
    if cycle:
        raise PlanError("these units depend on each other in a cycle: " + " -> ".join(cycle))


def _topo(ids: Sequence[str], edges: Mapping[str, tuple[str, ...]]) -> list[str]:
    """A total order consistent with `edges`, declaration order breaking ties.

    Serialisation edges are added along this order, and only along it, which is
    what makes them incapable of creating a cycle: every added edge points the
    same way as one fixed topological order, so the combined graph still has
    one.
    """
    position = {unit_id: index for index, unit_id in enumerate(ids)}
    remaining = {unit_id: set(edges.get(unit_id, ())) for unit_id in ids}
    settled: set[str] = set()
    out: list[str] = []
    while remaining:
        ready = sorted((u for u, need in remaining.items() if need <= settled), key=lambda u: position[u])
        if not ready:
            _refuse_cycle(edges, list(remaining))
            raise PlanError(f"these units cannot be ordered: {', '.join(sorted(remaining))}")
        for unit_id in ready:
            del remaining[unit_id]
        settled |= set(ready)
        out.extend(ready)
    return out


def resolve(plan: Plan) -> Graph:
    """Turn declared contracts into edges, refusing what cannot be run.

    Order matters here. Data dependencies are resolved and cycle-checked first,
    because a cycle is the architect's mistake and must be reported as one; only
    then are file collisions serialised, along the order the data dependencies
    already imply, so a serialisation edge can never manufacture a cycle and get
    a legitimate plan refused.
    """
    ids = [u.id for u in plan.units]
    known = set(ids)
    notes: list[str] = []

    providers: dict[str, list[str]] = {}
    for unit in plan.units:
        for name in unit.provides:
            providers.setdefault(name, []).append(unit.id)
    for name, owners in sorted(providers.items()):
        if len(owners) > 1:
            # Not refused: two units genuinely can build two halves of one
            # interface. Every consumer then waits for both, which is the only
            # reading that cannot be wrong.
            notes.append(f"{name!r} is provided by {' and '.join(owners)}; consumers wait for both")

    needs: dict[str, tuple[str, ...]] = {}
    for unit in plan.units:
        deps: list[str] = []
        for dep in unit.needs:
            if dep == unit.id:
                raise PlanError(f"{unit.id} depends on itself")
            if dep not in known:
                # Dropping the edge, which `workflow.parse_plan` can afford to
                # do, would here let a consumer start early - the precise bug
                # this module exists to prevent - so it is a refusal.
                raise PlanError(f"{unit.id} depends on {dep!r}, which is not in the plan")
            deps.append(dep)
        for name in unit.consumes:
            owners = [o for o in providers.get(name, ()) if o != unit.id]
            if not owners:
                if name in unit.provides:
                    continue  # a unit consuming what it also provides is just internal
                notes.append(
                    f"{unit.id} consumes {name!r}, which no unit provides; "
                    "assumed to exist in the repository already"
                )
                continue
            deps.extend(owners)
        needs[unit.id] = tuple(dict.fromkeys(deps))

    _refuse_cycle(needs, ids)

    order = {unit_id: set(deps) for unit_id, deps in needs.items()}
    ranking = _topo(ids, needs)
    position = {unit_id: index for index, unit_id in enumerate(ranking)}

    touching: dict[str, list[str]] = {}
    for unit in plan.units:
        for path in unit.files:
            touching.setdefault(path, []).append(unit.id)
    for path, owners in sorted(touching.items()):
        if len(owners) < 2:
            continue
        chain = sorted(dict.fromkeys(owners), key=lambda u: position[u])
        for earlier, later in zip(chain, chain[1:]):
            order[later].add(earlier)
        notes.append(f"{path}: {' then '.join(chain)}, one writer at a time")

    silent = [u.id for u in plan.units if not u.files]
    if silent:
        # Said plainly because the protection is opt-in from the architect's
        # side: a unit that names no file gets no collision check at all, and an
        # operator reading the plan should know which ones those are.
        notes.append(f"declare no files, so cannot be checked for collisions: {', '.join(silent)}")

    return Graph(
        needs={k: tuple(v) for k, v in needs.items()},
        order={k: tuple(sorted(v, key=lambda u: position[u])) for k, v in order.items()},
        notes=tuple(notes),
    )


def _layer(units: Sequence[Unit], order: Mapping[str, tuple[str, ...]]) -> list[Wave]:
    """Group units so everything in a group can start at once."""
    by_id = {u.id: u for u in units}
    position = {u.id: i for i, u in enumerate(units)}
    remaining = {u.id: set(order.get(u.id, ())) for u in units}
    settled: set[str] = set()
    waves: list[Wave] = []
    while remaining:
        ready = sorted((u for u, need in remaining.items() if need <= settled), key=lambda u: position[u])
        if not ready:
            _refuse_cycle(order, list(remaining))
            raise PlanError(f"these units cannot be scheduled: {', '.join(sorted(remaining))}")
        waves.append(Wave(len(waves) + 1, tuple(by_id[u] for u in ready)))
        settled |= set(ready)
        for unit_id in ready:
            del remaining[unit_id]
    return waves


def plan_waves(plan: Plan) -> list[Wave]:
    """The execution shape of a plan: refuses first, then groups."""
    return _layer(plan.units, plan.validate().order)


# -- reading a plan out of a model's reply ------------------------------------


ARCHITECT_SYSTEM: Final = """You are the architect. You write no code.

Split the request into units of work that different coders can do at the same
time, and reply with JSON only:

{"units": [
  {"id": "slug", "task": "one imperative sentence",
   "files": ["path/one.py"], "provides": ["InterfaceName"],
   "consumes": ["InterfaceName"], "needs": ["other-id"],
   "role": "implementer"}
]}

The fields that decide whether the run works:
  files     every file this unit will touch. Two units naming the same file are
            run one after the other, never together. A missing entry is how
            concurrent edits silently lose work, so be exact.
  provides  names of what this unit creates and others will import: a module, a
            class, a function signature.
  consumes  what this unit needs to exist first. It will not start until every
            unit providing those has finished.
  needs     unit ids that must finish first, for ordering no interface
            expresses. Usually empty; prefer provides and consumes.

Rules:
- Name an interface identically in the provider's provides and the consumer's
  consumes. A spelling difference is a race, not a typo.
- No cycles. If two units each consume what the other provides, they are one
  unit; say so.
- Give two units the same file only when both genuinely must edit it, and expect
  them to run in sequence when you do.
- Prefer 2 to 5 units. One unit is a fine answer for a small request.
- Do not invent tests, commands or files you were not told exist.
"""


def architect_prompt(goal: str, *, files: Sequence[str] = (), roles: Sequence[str] = ()) -> str:
    """What the architect is asked. Its reply must be `parse_plan`-able."""
    parts = [f"Request: {goal}"]
    if files:
        parts.append("Files that already exist and are relevant:\n" + "\n".join(f"  {f}" for f in files))
    if roles:
        parts.append(f"Coder roles actually available: {', '.join(roles)}.")
    parts.append("Reply with the JSON plan and nothing else.")
    return "\n\n".join(parts)


def _unit_from(item: Any, index: int) -> Unit:
    if not isinstance(item, dict):
        raise PlanError(f"unit {index + 1} is {type(item).__name__}, not an object")
    task = str(item.get("task") or item.get("goal") or item.get("description") or "").strip()
    if not task:
        raise PlanError(f"unit {index + 1} has no task")
    return Unit(
        id=str(item.get("id") or item.get("name") or f"u{index + 1}").strip()[:24] or f"u{index + 1}",
        task=task,
        files=_texts(item.get("files") or item.get("paths")),
        provides=_texts(item.get("provides") or item.get("exports")),
        consumes=_texts(item.get("consumes") or item.get("requires")),
        needs=_texts(item.get("needs") or item.get("after")),
        role=str(item.get("role") or "implementer"),
    )


def parse_plan(goal: str, reply: str, *, limit: int = MAX_UNITS) -> Plan:
    """Read a plan out of an architect's reply, or refuse it.

    Prose, apologies and fences around the JSON are tolerated because every
    model produces them. Everything else is refused, which is the opposite of
    `workflow.parse_plan`: that one falls back to a single step doing the whole
    job, and here that fallback would be a lie. The caller asked for a
    decomposition; quietly returning one undecomposed unit throws away the
    parallelism, the file safety and the interfaces all at once, and does it
    without saying so.
    """
    raw = _first_object(reply)
    if not isinstance(raw, dict):
        raise PlanError("the architect's reply contained no JSON object")
    listing = raw.get("units")
    if listing is None:
        listing = raw.get("steps")
    if not isinstance(listing, list):
        raise PlanError('the architect\'s reply has no "units" list')
    if not listing:
        raise PlanError("the architect returned an empty plan")
    if len(listing) > limit:
        # Truncating a dependency graph is worse than refusing it: the unit
        # dropped off the end is as likely as not to be a provider, and every
        # consumer left behind would then run against an interface nobody wrote.
        raise PlanError(f"the plan asks for {len(listing)} units; the cap is {limit}")

    plan = Plan(goal=goal.strip() or str(raw.get("goal") or "").strip(),
                units=[_unit_from(item, index) for index, item in enumerate(listing)])
    plan.validate()
    return plan


# -- choosing models ---------------------------------------------------------


def architect_seat(roster: Ensemble) -> Seat:
    """The seat that should plan. Falls back rather than returning None.

    `staff` fills a role nobody claims with the strongest idle seat, so a roster
    of local models still gets an architect instead of the whole feature
    silently declining to be decomposed.
    """
    return roster.pick(ARCHITECT_ROLE) or roster.staff((ARCHITECT_ROLE,))[0][1]


def coder_seats(roster: Ensemble, *, avoid: Seat | None = None) -> list[Seat]:
    """Seats fit to write code, best first."""
    def rank(seat: Seat) -> tuple[int, float, str]:
        role = CODER_ROLES.index(seat.role) if seat.role in CODER_ROLES else len(CODER_ROLES)
        return (role, -seat.weight, seat.model)

    ranked = sorted(roster.seats, key=rank)
    # `or ranked`: a single-seat roster means the architect also codes, which is
    # correct - one model doing both beats refusing to run.
    return [s for s in ranked if s is not avoid] or ranked


def assign(plan: Plan, roster: Ensemble, *, avoid: Seat | None = None) -> None:
    """Give every unit a model, rotating within its role.

    Rotating rather than picking the heaviest seat per role is the whole point:
    `pick` would hand all four implementer units to one model, and a wave of
    four units on one endpoint is a queue, not parallelism.
    """
    pool = coder_seats(roster, avoid=avoid)
    by_role: dict[str, list[Seat]] = {}
    cursor: dict[str, int] = {}
    for unit in plan.units:
        seats = by_role.setdefault(unit.role, [s for s in pool if s.role == unit.role] or pool)
        turn = cursor.get(unit.role, 0)
        unit.model = seats[turn % len(seats)].model
        cursor[unit.role] = turn + 1


# -- running the plan --------------------------------------------------------


#: Does one unit's work; returns `(output, error)` and a non-empty error fails
#: the unit. Takes the plan as well as the unit, mirroring `tasks.Worker`,
#: because a consumer's brief has to quote what its provider actually reported.
#: Injected so this module needs neither a model nor a repository to be tested.
Runner = Callable[[Plan, Unit], tuple[str, str]]

#: Given the architect prompt, returns the reply to parse. Injected for the same
#: reason.
Architect = Callable[[str], str]


@dataclass(slots=True)
class Run:
    """A plan, its waves, and what became of every unit."""

    plan: Plan
    graph: Graph
    waves: list[Wave]
    seconds: float = 0.0

    @property
    def units(self) -> list[Unit]:
        return self.plan.units

    @property
    def done(self) -> list[Unit]:
        return [u for u in self.units if u.state == DONE]

    @property
    def failed(self) -> list[Unit]:
        return [u for u in self.units if u.state == FAILED]

    @property
    def blocked(self) -> list[Unit]:
        return [u for u in self.units if u.state in (BLOCKED, SKIPPED)]

    @property
    def ok(self) -> bool:
        return bool(self.units) and all(u.state == DONE for u in self.units)

    def report(self) -> list[str]:
        lines: list[str] = []
        for wave in self.waves:
            lines.append(f"wave {wave.depth}" + ("" if len(wave) == 1 else f"  ({len(wave)} at once)"))
            for unit in wave:
                mark = {DONE: "ok", FAILED: "fail", BLOCKED: "blocked", SKIPPED: "skipped"}.get(unit.state, unit.state)
                model = f" {unit.model}" if unit.model else ""
                lines.append(f"  {unit.id:<14} {mark:<8}{model}  {unit.summary()}")
        lines.append(
            f"{len(self.done)}/{len(self.units)} units done"
            + (f", {len(self.failed)} failed" if self.failed else "")
            + (f", {len(self.blocked)} not run" if self.blocked else "")
            + f" in {self.seconds:.1f}s"
        )
        return lines


def brief(plan: Plan, unit: Unit) -> str:
    """The whole job as one unit's coder should see it.

    Upstream output is quoted rather than summarised: the consumer needs the
    provider's actual words about what it built, and a coder told only "auth is
    done" will guess the signature.
    """
    parts = [f"Overall goal: {plan.goal}", f"Your unit ({unit.id}): {unit.task}"]
    if unit.files:
        parts.append(
            "Touch only these files:\n" + "\n".join(f"  {f}" for f in unit.files)
            + "\n\nOther coders are working in this repository right now. Editing a file "
              "outside this list will collide with them and lose their work."
        )
    if unit.provides:
        parts.append(
            "You must leave these working and importable, because other units are "
            "waiting for exactly these names:\n" + "\n".join(f"  {p}" for p in unit.provides)
        )
    for name in unit.consumes:
        for owner in plan.providers_of(name):
            if owner.id == unit.id or not owner.text:
                continue
            parts.append(f"{name} was just built by {owner.id}, which reported:\n{owner.text}")
    return "\n\n".join(parts)


def _upstream_failure(unit: Unit, graph: Graph, by_id: Mapping[str, Unit]) -> str | None:
    """Why this unit must not run, or None.

    Only data dependencies are consulted. A unit serialised behind a *file* it
    shares carries on when that neighbour fails: it needed exclusive access, not
    a successful neighbour, and blocking it would be the module inventing a
    dependency the architect never declared.
    """
    for dep in graph.needs.get(unit.id, ()):
        upstream = by_id.get(dep)
        if upstream is None or upstream.state == DONE:
            continue
        if upstream.state == FAILED:
            return f"{dep} failed: {upstream.error or 'no reason given'}"
        if upstream.state in (BLOCKED, SKIPPED):
            # The chain is kept rather than collapsed to "upstream blocked", so
            # the last unit in a five-deep line still names the original fault.
            return f"{dep} did not run: {upstream.error or upstream.state}"
    return None


def _run_one(plan: Plan, unit: Unit, runner: Runner) -> None:
    unit.state, unit.error = RUNNING, None
    started = time.monotonic()
    try:
        text, error = runner(plan, unit)
    except Exception as exc:
        # Swallowed deliberately, and this is the point of the module: a coder
        # that raises - a provider 500, a bad tuple, a crash inside an agent -
        # fails its own unit and its dependents, and the independent subtrees
        # keep going. Re-raising here would end the whole run on one bad model.
        text, error = "", f"{type(exc).__name__}: {exc}"
    unit.seconds = time.monotonic() - started
    unit.text = (text or "").strip()
    if error:
        unit.state, unit.error = FAILED, str(error)
    else:
        unit.state = DONE


def _run_wave(plan: Plan, units: Sequence[Unit], runner: Runner, workers: int) -> None:
    """Run a wave. Each unit is owned by exactly one thread, so no locking.

    That ownership is guaranteed by the layering, not by convention: a unit
    appears in exactly one wave, and `_layer` puts two units that share a file
    in different waves.
    """
    if len(units) == 1:
        _run_one(plan, units[0], runner)
        return
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(units))), thread_name_prefix="coder") as pool:
        list(pool.map(lambda unit: _run_one(plan, unit, runner), units))


def execute(
    plan: Plan,
    runner: Runner,
    *,
    workers: int = DEFAULT_WORKERS,
    cancel: threading.Event | None = None,
) -> Run:
    """Walk the plan in waves, isolating failure to the subtree that caused it.

    Refuses before it runs anything: a plan validated at parse time can still be
    handed here after being edited, and half-executing a cycle is the one
    outcome with no honest recovery.
    """
    graph = plan.validate()
    waves = _layer(plan.units, graph.order)
    run = Run(plan=plan, graph=graph, waves=waves)
    by_id = {u.id: u for u in plan.units}
    started = time.monotonic()

    for wave in waves:
        ready: list[Unit] = []
        for unit in wave:
            if unit.state == DONE:
                continue  # a plan re-run after a fix must not redo finished work
            reason = _upstream_failure(unit, graph, by_id)
            if reason is not None:
                unit.state, unit.error = BLOCKED, reason
                continue
            if cancel is not None and cancel.is_set():
                unit.state, unit.error = SKIPPED, "the run was cancelled"
                continue
            ready.append(unit)
        if ready:
            _run_wave(plan, ready, runner, workers)

    run.seconds = time.monotonic() - started
    return run


def decompose(
    goal: str,
    *,
    architect: Architect,
    runner: Runner,
    roster: Ensemble | None = None,
    files: Sequence[str] = (),
    limit: int = MAX_UNITS,
    workers: int = DEFAULT_WORKERS,
    cancel: threading.Event | None = None,
) -> Run:
    """Plan with the architect, then run the plan with the coders.

    Raises `PlanError` if there is nothing to run: a refusal before any model
    was paid to edit anything is the cheapest possible outcome, so it is not
    softened into an empty `Run` the caller has to inspect.
    """
    if not goal.strip():
        raise PlanError("nothing to decompose")

    seat = architect_seat(roster) if roster is not None else None
    roles = sorted({s.role for s in coder_seats(roster, avoid=seat)}) if roster is not None else ()
    try:
        reply = architect(architect_prompt(goal, files=files, roles=roles))
    except PlanError:
        raise
    except Exception as exc:
        # Turned into a refusal rather than propagated: from the caller's side a
        # planner that times out and a planner that answers nonsense are the
        # same event - no plan - and both should read that way.
        raise PlanError(f"the architect failed: {type(exc).__name__}: {exc}") from exc

    plan = parse_plan(goal, reply or "", limit=limit)
    if roster is not None:
        assign(plan, roster, avoid=seat)
    return execute(plan, runner, workers=workers, cancel=cancel)


# -- the shell surface -------------------------------------------------------


def shell_architect(state: Any, roster: Ensemble | None = None) -> Architect:
    """An architect that asks the planner-hinted model, with no tools at all.

    No toolbox on purpose: the architect writes no code, and a planner given an
    edit tool starts implementing the first unit instead of describing the
    fifth.
    """
    from offset.providers.base import Message, Request

    def think(prompt: str) -> str:
        if roster is None:
            result = state.agent.send(f"{ARCHITECT_SYSTEM}\n\n{prompt}")
            if result.error:
                raise PlanError(f"the architect model failed: {result.error}")
            return result.text or ""
        seat = architect_seat(roster)
        opinion = roster.ask(seat, Request(
            model=seat.model,
            system=ARCHITECT_SYSTEM,
            messages=[Message("user", prompt)],
            max_tokens=4096,
        ))
        if not opinion.ok:
            raise PlanError(f"the architect model failed: {opinion.error}")
        return opinion.text

    return think


def shell_runner(state: Any, *, timeout: float = 900.0, cancel: threading.Event | None = None) -> Runner:
    """A runner giving each unit its own subagent, on the model it was assigned.

    A subagent rather than the shell's own agent because the units run at once:
    they need separate sessions and separate histories, and `SubagentRunner`
    already builds and bounds those. The user's approval policy object is shared
    rather than copied, so a coder cannot be a way around a `no`.
    """
    from offset.tools.agents import SubagentRunner
    from offset.tools.base import ToolContext

    def work(plan: Plan, unit: Unit) -> tuple[str, str]:
        spawner = SubagentRunner(
            model=unit.model or state.model,
            approval=state.approval,
            concurrency=DEFAULT_WORKERS,
        )
        ctx = ToolContext(
            cwd=state.workspace,
            root=state.workspace,
            timeout=timeout,
            cancel=cancel or threading.Event(),
        )
        result = spawner.run(brief(plan, unit), ctx)
        return result.text, (result.error or "")

    return work


def _decompose(state: Any, args: list[str]) -> Any:
    """`/decompose <goal>` - plan it with an architect, build it in parallel."""
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    if not args:
        return Outcome.error("usage: /decompose <goal>",
                            "an architect splits it up, then coders run the units in parallel")

    goal = " ".join(args)
    roster = state.ensemble
    plan_only = args[0].lower() == "--plan"
    if plan_only:
        goal = " ".join(args[1:])
        if not goal:
            return Outcome.error("usage: /decompose --plan <goal>")

    think = shell_architect(state, roster)
    work = shell_runner(state)

    def job() -> Any:
        try:
            if plan_only:
                plan = parse_plan(goal, think(architect_prompt(goal)))
                return Outcome(plan.outline(), TONE_INFO)
            run = decompose(goal, architect=think, runner=work, roster=roster)
        except PlanError as exc:
            # A refused plan is the designed outcome, not a crash: it is shown
            # as the architect's mistake so the user can rephrase the request.
            return Outcome.error(f"the plan was refused: {exc}")
        return Outcome(run.report(), TONE_OK if run.ok else TONE_INFO)

    return Outcome([f"decomposing: {goal}", "asking the architect..."], TONE_INFO, job=job)


def decompose_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command("decompose", "split a feature across parallel coder models", _decompose,
                usage="/decompose <goal> | /decompose --plan <goal>"),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """`COMMANDS` on demand.

    Built lazily, exactly as `core/tasks` does it: `offset.shell.commands`
    imports the subsystems, so a module-level `from ... import Command` here
    would be a cycle at interpreter start.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            _COMMANDS.extend(decompose_commands())
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
