"""Why one branch beat the others, in numbers a human can audit.

`Speculation.rank` sorts attempts lexicographically on (pass-tier, churn,
duration, name).  It is cheap, it is deterministic, and it is kept — but as a
*selector* it has two flaws that show up the moment branches are close:

  * **Churn is compared in absolute terms.**  A three-line patch beating a
    five-line patch is noise dressed up as a decision, while ten lines beating
    five hundred is a real signal.  One `<` cannot say both.  Here diff size is
    a saturating curve, so the marginal cost of a line falls as the diff grows.
  * **A verdict without a reason is not evidence.**  "Branch 2 won" is not
    something a user can argue with.  Every criterion below carries a
    normalised 0..1 sub-score *and* a sentence saying what it saw, and
    `Scorecard.lines()` prints the arithmetic that produced the winner.

Six criteria, weighted and then normalised over the ones that actually applied.
That normalisation is the load-bearing detail: a criterion with nothing to say
(no baseline to compare against, no linter installed, a runner that reported no
metrics) is *excluded from the denominator* rather than scored zero.  Absence
of evidence must not read as evidence of badness — a branch is not worse for
having run on a machine with no `ruff`.

Regressions are weighted above verification on purpose.  An unproven fix costs
the user their time; a fix that breaks something which already worked costs
them work they had banked.  The regression weight deliberately exceeds the
*combined* weight of every soft criterion, so no amount of small diffs, clean
lint or model flattery can rehabilitate a branch that broke a passing test.

Nothing here reaches the network unless a caller injects a reviewer, and
`reviewer=None` is the default: scoring is a local, offline, repeatable
measurement, and the model-as-judge is the optional garnish rather than the
mechanism.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Sequence

from offset.core.speculate import Attempt, Speculation, TestCounts

# -- criteria ---------------------------------------------------------------

VERIFICATION: Final = "verification"
REGRESSIONS: Final = "regressions"
DIFF: Final = "diff"
STATIC: Final = "static"
HEALTH: Final = "health"
REVIEW: Final = "review"


@dataclass(frozen=True, slots=True)
class Criterion:
    """One measurement: a normalised score, its weight, and what it saw.

    `applies=False` means "this criterion had nothing to measure".  Such a
    criterion is still reported — the user should see that no linter ran — but
    it contributes neither points nor weight, so it cannot tilt the result in
    either direction.
    """

    name: str
    score: float
    weight: float
    reason: str
    applies: bool = True

    @property
    def points(self) -> float:
        return self.score * self.weight if self.applies else 0.0

    def line(self) -> str:
        # Column-aligned so a reader can scan the scores down the page; the
        # inapplicable ones keep their row rather than vanishing, because
        # "no linter ran" is itself something the user needs to know.
        if not self.applies:
            return f"  {self.name:<13} --{'':<8}{self.reason}"
        return f"  {self.name:<13} {self.score:.2f} x{self.weight:<3g} {self.reason}"


@dataclass(frozen=True, slots=True)
class Weights:
    """How much each criterion counts.  The field names *are* the criterion
    names, so a scorecard row and its weight are always spelled the same way.

    Only the ratios matter: the total is normalised by the weight of whatever
    applied, so these numbers are a statement about priorities, not a scale.
    """

    #: The only direct evidence that the change does the job.  Nothing else
    #: observable substitutes for a suite that went green.
    verification: float = 4.0
    #: Above verification, and above the sum of every soft criterion below
    #: (1.0 + 1.0 + 0.5 + 1.5 = 4.0), because breaking something that already
    #: worked is worse than failing to prove something new works.
    regressions: float = 5.0
    #: A tie-breaker with teeth, not a headline.  Two branches that both pass
    #: should be separated by size; size should never outvote passing.
    diff: float = 1.0
    #: Equal to diff size: lint diagnostics are the same kind of evidence — a
    #: cheap, mechanical proxy for "someone will have to clean this up".
    static: float = 1.0
    #: Half of that.  How the agent got there is weak evidence about the
    #: result; a branch that flailed and still passed everything did pass.
    health: float = 0.5
    #: Enough to break a near-tie between two branches the machine cannot
    #: separate, never enough to overturn verification or a regression.
    review: float = 1.5


DEFAULT_WEIGHTS: Final = Weights()


@dataclass(slots=True)
class Scorecard:
    """One attempt's full breakdown, and the prose that explains it."""

    attempt: Attempt
    criteria: list[Criterion] = field(default_factory=list)
    total: float = 0.0

    @property
    def name(self) -> str:
        return self.attempt.approach.name

    @property
    def applied(self) -> list[Criterion]:
        return [c for c in self.criteria if c.applies]

    @property
    def strongest(self) -> Criterion | None:
        """The criterion contributing most of the total — the headline reason."""
        applied = self.applied
        return max(applied, key=lambda c: c.points) if applied else None

    def get(self, name: str) -> Criterion | None:
        return next((c for c in self.criteria if c.name == name), None)

    def __getitem__(self, name: str) -> Criterion:
        found = self.get(name)
        if found is None:
            raise KeyError(f"no criterion named {name}. available: {', '.join(c.name for c in self.criteria)}")
        return found

    def summary(self) -> str:
        top = self.strongest
        return f"{self.name} {self.total:.2f}" + (f" - {top.reason}" if top else "")

    def lines(self) -> list[str]:
        """The arithmetic, in the order it was applied.  Returns, never prints."""
        out = [f"{self.name} scored {self.total:.2f} of 1.00"]
        out.extend(c.line() for c in self.criteria)
        weight = sum(c.weight for c in self.applied)
        if weight:
            out.append(f"  {'total':<13} {sum(c.points for c in self.applied):.2f} / {weight:g} weight applied")
        return out


# -- the baseline -----------------------------------------------------------


#: pytest, go, cargo and unittest each name their failures differently, and a
#: regression can only be spotted by name: a *count* cannot tell a broken test
#: from a deleted one.
_FAILED_NAMES: Final = (
    re.compile(r"^FAILED\s+(\S+)", re.M),                       # pytest summary
    re.compile(r"^\s*--- FAIL: (\S+)", re.M),                   # go test -v
    re.compile(r"^test (\S+) \.\.\. FAILED", re.M),             # cargo test
    re.compile(r"^(?:FAIL|ERROR):\s+(\S+)", re.M),              # unittest
)
#: The same runners, on the way through.  Only verbose modes print these, which
#: is why the count fallback in `_regressions` exists at all.
_PASSED_NAMES: Final = (
    re.compile(r"^(\S+)\s+PASSED\b", re.M),                     # pytest -v
    re.compile(r"^\s*--- PASS: (\S+)", re.M),                   # go test -v
    re.compile(r"^test (\S+) \.\.\. ok", re.M),                 # cargo test
    re.compile(r"^(\S+) \([^)]*\) \.\.\. ok", re.M),            # unittest -v
)

#: Kept out of the way of a real approach name; `create` destroys a colliding
#: path, so the baseline must not be able to eat a branch.
_BASELINE_SLUG: Final = "baseline-check"


def _named(text: str, patterns: Sequence[re.Pattern[str]]) -> frozenset[str]:
    found: set[str] = set()
    for pattern in patterns:
        found.update(pattern.findall(text))
    return frozenset(found)


@dataclass(slots=True)
class Baseline:
    """What the verification command said about the workspace *before* branching.

    Without this there is no such thing as a regression: a failing test is only
    a regression if it used to pass.  `error` carries the reason we have no
    baseline, which is reported rather than silently treated as "all clear".
    """

    counts: TestCounts = field(default_factory=TestCounts)
    passing: frozenset[str] = frozenset()
    failing: frozenset[str] = frozenset()
    ref: str = ""
    error: str | None = None

    @property
    def known(self) -> bool:
        return self.error is None and (self.counts.known or bool(self.passing))

    def summary(self) -> str:
        if self.error:
            return f"no baseline: {self.error}"
        where = f" at {self.ref[:8]}" if self.ref else ""
        return f"baseline{where}: {self.counts.summary()}"


def measure_baseline(spec: Speculation) -> Baseline:
    """Run the same verification against the unmodified workspace, once.

    A throwaway workspace is created from `spaces.base()` — the commit that
    represents the tree *including uncommitted work* — and verified with
    `spec.verify`, so the baseline and the branches are measured by exactly the
    same code path.  Anything else would compare a green run against a
    differently-invoked red one.

    The measurement workspace is always destroyed, even under `keep`: it is not
    a candidate anybody could adopt.
    """
    if not spec.verify_command:
        return Baseline(error="no verification command configured")
    path = None
    try:
        path = spec.spaces.create(_BASELINE_SLUG)
        result = spec.verify(path)
    except Exception as exc:  # a baseline is a nicety; never fail the run for it
        return Baseline(error=f"{type(exc).__name__}: {exc}")
    finally:
        if path is not None:
            spec.spaces.destroy(path)
    base = getattr(spec.spaces, "base", None)
    try:
        ref = base() if callable(base) else ""
    except Exception:
        ref = ""
    return Baseline(
        counts=result.counts,
        passing=_named(result.output, _PASSED_NAMES),
        failing=_named(result.output, _FAILED_NAMES),
        ref=ref,
    )


# -- static analysis --------------------------------------------------------


#: `file:line:` and `file:line:col:` — ruff, pyflakes, mypy, gcc, everyone.
_POSIX_DIAG: Final = re.compile(r"^\s*\S[^\n]*?:\d+(?::\d+)?[:\s]", re.M)
#: TypeScript puts the position in brackets: `src/a.ts(3,10): error TS2345: ...`
_TSC_DIAG: Final = re.compile(r"^\S+\(\d+,\d+\):\s*error", re.M)
#: eslint's `compact` formatter: `src/a.js: line 3, col 10, Error - ...`
_ESLINT_DIAG: Final = re.compile(r"^\S+:\s+line \d+, col \d+", re.M)


@dataclass(frozen=True, slots=True)
class Linter:
    """A static analyser we will run *only* if the repo asked for it.

    `configs` pairs a filename with a substring that must appear inside it, so
    a shared file like `pyproject.toml` only counts when it actually carries
    that tool's section.  Running a linter nobody configured produces a wall of
    diagnostics about a style the project never adopted, which is noise
    masquerading as evidence.
    """

    name: str
    argv: tuple[str, ...]
    configs: tuple[tuple[str, str], ...]
    diagnostic: re.Pattern[str] = _POSIX_DIAG
    timeout: float = 60.0

    def configured(self, root: Path) -> str:
        for filename, marker in self.configs:
            candidate = root / filename
            if not candidate.is_file():
                continue
            if not marker:
                return filename
            try:
                if marker in candidate.read_text(encoding="utf-8", errors="replace"):
                    return filename
            except OSError:
                continue
        return ""


DEFAULT_LINTERS: Final[tuple[Linter, ...]] = (
    # --no-cache: the worktree is deleted minutes from now, so a cache written
    # into it is pure I/O for nothing.
    Linter(
        "ruff",
        ("ruff", "check", "--no-cache", "--output-format", "concise", "."),
        ((".ruff.toml", ""), ("ruff.toml", ""), ("pyproject.toml", "[tool.ruff")),
    ),
    # pyflakes has no configuration file at all, so "this is a Python project"
    # is the only gate available.  It is a fair one: pyflakes only reports
    # unambiguous errors, never style.
    Linter("pyflakes", ("pyflakes", "."), (("pyproject.toml", ""), ("setup.py", ""))),
    Linter(
        "mypy",
        ("mypy", "--no-error-summary", "--no-incremental", "."),
        (("mypy.ini", ""), (".mypy.ini", ""), ("setup.cfg", "[mypy]"), ("pyproject.toml", "[tool.mypy")),
        timeout=180.0,
    ),
    Linter(
        "tsc",
        ("tsc", "--noEmit", "-p", "tsconfig.json"),
        (("tsconfig.json", ""),),
        diagnostic=_TSC_DIAG,
        timeout=180.0,
    ),
    Linter(
        "eslint",
        ("eslint", ".", "-f", "compact"),
        (("eslint.config.js", ""), ("eslint.config.mjs", ""), (".eslintrc", ""),
         (".eslintrc.json", ""), (".eslintrc.js", ""), ("package.json", "eslintConfig")),
        diagnostic=_ESLINT_DIAG,
        timeout=180.0,
    ),
)


class Analysis:
    """Runs whichever linters this machine and this repo agree on.

    Discovery is cached for the life of one `score()` call: `shutil.which` per
    linter per attempt would stat the whole PATH once per branch for no reason.
    """

    __slots__ = ("_found", "linters")

    def __init__(self, linters: Sequence[Linter] = DEFAULT_LINTERS) -> None:
        self.linters = tuple(linters)
        self._found: dict[str, str | None] = {}

    def _binary(self, linter: Linter) -> str | None:
        if linter.argv[0] not in self._found:
            self._found[linter.argv[0]] = shutil.which(linter.argv[0])
        return self._found[linter.argv[0]]

    def run(self, root: Path) -> tuple[dict[str, int], list[str]]:
        """Diagnostic counts per usable linter, plus notes on the ones skipped."""
        counts: dict[str, int] = {}
        notes: list[str] = []
        for linter in self.linters:
            binary = self._binary(linter)
            if binary is None:
                continue  # not installed: silence, not a penalty
            config = linter.configured(root)
            if not config:
                continue  # installed but unconfigured: not this project's rule
            try:
                proc = subprocess.run(
                    [binary, *linter.argv[1:]], cwd=str(root), capture_output=True,
                    text=True, timeout=linter.timeout, errors="replace",
                )
            except subprocess.TimeoutExpired:
                notes.append(f"{linter.name} exceeded {linter.timeout:g}s")
                continue
            except OSError as exc:
                notes.append(f"{linter.name} would not start: {exc}")
                continue
            output = (proc.stdout or "") + (proc.stderr or "")
            found = len(linter.diagnostic.findall(output))
            # A linter reports findings with exit 1; anything higher with no
            # parsed diagnostics is the tool itself failing (bad config, syntax
            # error in its own rules) and must not be read as a clean run.
            if found == 0 and proc.returncode not in (0, 1):
                notes.append(f"{linter.name} exited {proc.returncode} without diagnostics")
                continue
            counts[linter.name] = found
        return counts, notes


# -- the reviewer -----------------------------------------------------------


#: Injected model-as-judge.  Given every attempt under comparison, returns the
#: winning approach name and a one-line reason, or `None` when it could not
#: say.  Called once per `score()`, never once per attempt.
Reviewer = Callable[[Sequence[Attempt]], tuple[str, str] | None]

#: A diff long enough to be a whole refactor still has to fit in a prompt
#: alongside its rivals; the head of a patch is where the intent lives.
REVIEW_BUDGET: Final = 6000


def council_reviewer(
    ensemble: object,
    judge: object,
    *,
    seats: Sequence[object] | None = None,
    criterion: str = "correctness first, then the smaller and clearer change",
) -> Reviewer:
    """Build a `Reviewer` from the ensemble's existing model-as-judge primitive.

    `Ensemble.council` already handles anonymising candidates, isolating a seat
    that dies, and degrading to a weighted vote when the judge answers
    unparseably.  Writing a second judge here would duplicate all of that and
    then drift from it, so this only builds the request and reads the branch
    name back out of the winning review.

    The provider stack is imported lazily: the common case is `reviewer=None`,
    and a purely local scorer should not drag the whole model registry into its
    import path.
    """
    from offset.providers.base import Message, Request

    def review(attempts: Sequence[Attempt]) -> tuple[str, str] | None:
        candidates = [a for a in attempts if a.diff.strip()]
        if len(candidates) < 2:
            return None
        listing = "\n\n".join(
            f"### {a.approach.name}\n{a.summary()}\n```diff\n{a.diff[:REVIEW_BUDGET]}\n```"
            for a in candidates
        )
        request = Request(
            model=getattr(judge, "model", ""),
            system=(
                "You are reviewing candidate patches for the same task. "
                "Name the single best branch on the first line, then one sentence of reasoning."
            ),
            messages=[Message("user", f"Criterion: {criterion}\n\n{listing}\n\nBest branch:")],
            max_tokens=400,
        )
        try:
            verdict = ensemble.council(request, judge, seats, criterion=criterion)  # type: ignore[attr-defined]
        except Exception:  # a reviewer is optional; its failure is not fatal
            return None
        if verdict.winner is None or not verdict.winner.text.strip():
            return None
        text = verdict.winner.text
        picked = _first_named(text, [a.approach.name for a in candidates])
        if picked is None:
            return None
        return picked, " ".join(text.split())[:160]

    return review


def _first_named(text: str, names: Sequence[str]) -> str | None:
    """The branch named earliest in the review, so the verdict beats the aside."""
    lowered = text.lower()
    hits = [(lowered.find(n.lower()), n) for n in names if n and n.lower() in lowered]
    return min(hits)[1] if hits else None


# -- scoring ----------------------------------------------------------------


#: Test count at which verification confidence is half-earned.  Ten green
#: tests is meaningfully more evidence than one; a hundred is barely more than
#: fifty, so the curve has to flatten.
COUNT_HALF: Final = 10.0
#: Changed lines at which the diff score is 0.5.  Forty lines is a focused
#: patch; the hyperbola makes 3-vs-5 lines a rounding error and 10-vs-500 a
#: verdict, which is exactly what `rank`'s absolute `<` on churn could not do.
DIFF_HALF: Final = 40.0
#: Diagnostics at which the static score is 0.5.  A handful is a review
#: comment; fifty is a different standard of care.
DIAG_HALF: Final = 5.0


def _falling(value: float, half: float) -> float:
    """1.0 at zero, 0.5 at `half`, asymptotic to 0.  Saturating by construction."""
    return half / (half + value) if value > 0 else 1.0


def _rising(value: float, half: float) -> float:
    """0.0 at zero, 0.5 at `half`, asymptotic to 1."""
    return value / (value + half) if value > 0 else 0.0


def _verification(attempt: Attempt, weight: float) -> Criterion:
    counts = attempt.verification.counts
    if attempt.error:
        return Criterion(VERIFICATION, 0.0, weight, f"the branch never finished: {attempt.error[:70]}")
    if attempt.verification.skipped:
        # Halfway, and applicable: "nobody checked" has to sit between proven
        # and disproven, which is where `rank` already put it.
        return Criterion(VERIFICATION, 0.5, weight, "nothing verified it")
    ratio = _ratio(counts)
    if not attempt.verification.ok:
        # Partial credit: 9 of 10 passing is a nearly-finished branch, 0 of 10
        # is a broken one, and flattening both to zero throws that away.
        return Criterion(VERIFICATION, 0.25 * ratio, weight, f"verification failed - {counts.summary()}")
    if counts.known and not counts.total:
        return Criterion(VERIFICATION, 0.6, weight, "verification passed but ran no tests")
    if not counts.known:
        return Criterion(VERIFICATION, 0.8, weight, "verification passed, no test counts to read")
    confidence = _rising(float(counts.total or 0), COUNT_HALF)
    return Criterion(VERIFICATION, 0.8 + 0.2 * confidence, weight, counts.summary())


def _ratio(counts: TestCounts) -> float:
    if not counts.known or not counts.total:
        return 0.0
    return max(0.0, min(1.0, (counts.passed or 0) / counts.total))


def _regressions(attempt: Attempt, baseline: Baseline | None, weight: float) -> Criterion:
    if baseline is None or not baseline.known:
        why = baseline.error if baseline and baseline.error else "no baseline to compare against"
        return Criterion(REGRESSIONS, 0.0, weight, why, applies=False)
    if attempt.verification.skipped:
        return Criterion(REGRESSIONS, 0.0, weight, "nothing ran, so nothing can be compared", applies=False)
    failing = _named(attempt.verification.output, _FAILED_NAMES)
    broken = sorted(baseline.passing & failing)
    if broken:
        shown = ", ".join(broken[:3]) + (f" and {len(broken) - 3} more" if len(broken) > 3 else "")
        return Criterion(REGRESSIONS, 0.0, weight, f"broke {len(broken)} test(s) that passed at the base: {shown}")
    counts = attempt.verification.counts
    # Fallback for runners that print no test names: fewer passes than the base
    # while something is failing.  Two guards, both load-bearing.  "While
    # failing" - a branch that legitimately deletes a test also lowers the pass
    # count.  And only when a name-level comparison was impossible: if the base
    # named its passes and the branch named its failures, that intersection is
    # the answer, and arithmetic must not overrule it by counting a brand-new
    # failing test as a regression.
    comparable = bool(failing and baseline.passing)
    if not comparable and counts.failed and counts.known \
            and baseline.counts.passed is not None and counts.passed is not None:
        shortfall = baseline.counts.passed - counts.passed
        if shortfall > 0:
            return Criterion(REGRESSIONS, 0.0, weight,
                             f"{shortfall} fewer tests pass than at the base, with {counts.failed} failing")
    return Criterion(REGRESSIONS, 1.0, weight, f"nothing that passed at the base fails here ({baseline.counts.summary()})")


def _diff(attempt: Attempt, weight: float) -> Criterion:
    churn = attempt.churn
    if churn == 0:
        # `rank` happily crowns a branch with an empty diff as the smallest
        # change of all.  A branch that changed nothing did not do the task.
        return Criterion(DIFF, 0.0, weight, "changed nothing at all - there is nothing to adopt")
    return Criterion(DIFF, _falling(float(churn), DIFF_HALF), weight, f"{churn} changed lines")


def _static(attempt: Attempt, analysis: Analysis, weight: float) -> Criterion:
    if not analysis.linters:
        return Criterion(STATIC, 0.0, weight, "static analysis not requested", applies=False)
    if attempt.path is None or not attempt.path.exists():
        return Criterion(STATIC, 0.0, weight, "no workspace left to analyse", applies=False)
    counts, notes = analysis.run(attempt.path)
    if not counts:
        why = "; ".join(notes) if notes else "no configured analyser on PATH"
        return Criterion(STATIC, 0.0, weight, why, applies=False)
    scores = [_falling(float(n), DIAG_HALF) for n in counts.values()]
    reason = ", ".join(f"{name}: {n} diagnostic{'' if n == 1 else 's'}" for name, n in sorted(counts.items()))
    return Criterion(STATIC, sum(scores) / len(scores), weight, reason + ("; " + "; ".join(notes) if notes else ""))


#: Stop reasons that mean the agent ran out of road rather than finishing.
FLAILED: Final = frozenset({"max_steps", "error", "timeout", "cancelled"})


def _health(attempt: Attempt, weight: float) -> Criterion:
    metrics = attempt.metrics
    if not metrics.observed:
        return Criterion(HEALTH, 0.0, weight, "the runner reported no agent metrics", applies=False)
    if metrics.error:
        return Criterion(HEALTH, 0.0, weight, f"the agent errored: {metrics.error[:60]}")
    score = 1.0
    notes: list[str] = [metrics.summary()]
    if metrics.tool_calls:
        share = metrics.tool_failures / metrics.tool_calls
        score -= 0.6 * share
        if metrics.tool_failures:
            notes.append(f"{metrics.tool_failures} of {metrics.tool_calls} tool calls failed")
    if metrics.stop_reason in FLAILED:
        # Hitting the step limit is the agent telling us it did not converge.
        score = min(score, 0.2)
        notes.append(f"stopped on {metrics.stop_reason}")
    return Criterion(HEALTH, max(0.0, score), weight, "; ".join(notes))


def _review(attempt: Attempt, pick: tuple[str, str] | None, weight: float) -> Criterion:
    if pick is None:
        return Criterion(REVIEW, 0.0, weight, "no reviewer was asked", applies=False)
    name, reason = pick
    if attempt.approach.name == name:
        return Criterion(REVIEW, 1.0, weight, f"the reviewer picked this branch: {reason}")
    return Criterion(REVIEW, 0.0, weight, f"the reviewer preferred {name}")


def score(
    attempts: Sequence[Attempt],
    *,
    weights: Weights = DEFAULT_WEIGHTS,
    reviewer: Reviewer | None = None,
    baseline: Baseline | None = None,
    linters: Sequence[Linter] = DEFAULT_LINTERS,
) -> list[Scorecard]:
    """Score every attempt and return the scorecards best first.

    Deterministic: the same attempts in any order produce the same ranking.
    Ties fall through to churn, then duration, then approach name — the same
    tiebreakers `Speculation.rank` uses, and for the same reason, so a replayed
    session shows the same winner.  Speed and name are *only* tiebreakers here;
    neither is a criterion, because being fast is not evidence of being right.
    """
    if not attempts:
        return []
    pick = _ask_reviewer(reviewer, attempts)
    analysis = Analysis(linters)
    cards: list[Scorecard] = []
    for attempt in attempts:
        criteria = [
            _verification(attempt, weights.verification),
            _regressions(attempt, baseline, weights.regressions),
            _diff(attempt, weights.diff),
            _static(attempt, analysis, weights.static),
            _health(attempt, weights.health),
            _review(attempt, pick, weights.review),
        ]
        applied = [c for c in criteria if c.applies]
        weight = sum(c.weight for c in applied)
        total = sum(c.points for c in applied) / weight if weight else 0.0
        cards.append(Scorecard(attempt, criteria, total))
    cards.sort(key=lambda c: (
        -round(c.total, 9),
        c.attempt.churn,
        round(c.attempt.duration, 3),
        c.name,
    ))
    return cards


def _ask_reviewer(reviewer: Reviewer | None, attempts: Sequence[Attempt]) -> tuple[str, str] | None:
    """One call for the whole set, and never a raise into the caller's turn."""
    if reviewer is None:
        return None
    try:
        pick = reviewer(attempts)
    except Exception:
        return None
    if not pick:
        return None
    name, reason = pick
    known = {a.approach.name for a in attempts}
    return (name, reason) if name in known else None

