"""Whether the winning branch made the program slower.

`offset/core/scoring.py` ranks speculative branches on verification,
regressions, diff size, static analysis and a reviewer. None of those notice a
correct fix that doubles a hot loop's runtime, so that fix wins the race and
the slowdown ships. This measures the thing nobody was measuring.

The whole module is shaped by one problem: **this runs on a Raspberry Pi, where
a single timed run is noise.** Three consequences, and the second is the one
most benchmark code gets wrong.

**A run is a distribution, not a number.** min, median, mean, stdev and count,
with a discarded warmup - the first run pays for cold caches and imports, and
letting it into the sample is how people conclude a change is slow when it is
not.

**A difference nobody can distinguish is reported as indistinguishable.** Not
as a small win. `compare()` looks at the spread of the samples themselves, and
if the two sets overlap it refuses to name a winner. This is the honesty
requirement: reporting "3% faster" from two twelve-sample runs on a loaded Pi
is a fabricated result, and a ranking built on fabricated results is worse than
one with no benchmark at all, because it moves confidently in random
directions.

**A crash is not a fast result.** A command that exits non-zero returns almost
instantly, which naive timing scores as a spectacular improvement. A failed
sample poisons its `Result` and the comparison says so.

Peak memory comes from `resource.getrusage(RUSAGE_CHILDREN)` - stdlib, no
dependency - because a change that halves runtime by buffering everything into
memory is not obviously a win either.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from offset.core import settings
from offset.core.scoring import Criterion

#: Runs kept in the sample, and runs thrown away first.  Twelve is enough for a
#: median to mean something and short enough that nobody stops using it.
RUNS: Final = 12
WARMUP: Final = 2

#: Per-run ceiling.  A benchmark that hangs must not hang the shell.
TIMEOUT: Final = 120.0

#: Below this, the timer is measuring process spawn rather than the program.
#: Reporting a winner between two 3ms commands is measuring the operating
#: system's mood.
FLOOR_SECONDS: Final = 0.005

#: Exit code stamped on a sample whose command exceeded the timeout.  A hung
#: benchmark needs different advice from a broken one - "raise the timeout" not
#: "fix your command" - so the two are not lumped together.
TIMED_OUT: Final = -9

#: A difference smaller than this fraction is not worth acting on even when the
#: samples happen not to overlap.  Two percent on a Pi is thermal drift.
NOISE: Final = 0.02

#: Where `/bench` looks for a command when given none.
CONFIG_NAMES: Final = ("bench.json", ".bench.json")

BENCH: Final = "performance"


@dataclass(slots=True, frozen=True)
class Sample:
    """One run."""

    seconds: float
    rss_kb: int = 0
    exit_code: int = 0
    bytes_out: int = 0

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(slots=True)
class Result:
    """A set of runs of the same command."""

    command: str
    samples: list[Sample] = field(default_factory=list)
    error: str = ""
    #: Carried so `summary()` can quote the limit that was actually hit rather
    #: than the module default, which may not be the one in force.
    timeout_used: float = TIMEOUT

    @property
    def times(self) -> list[float]:
        return [s.seconds for s in self.samples if s.ok]

    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def failures(self) -> int:
        return sum(1 for s in self.samples if not s.ok)

    @property
    def ok(self) -> bool:
        """Usable as a measurement.

        A single failed run invalidates the whole result rather than being
        averaged in: a crash returns instantly, so mixing it with real timings
        drags the mean down and makes a broken command look fast.
        """
        return bool(self.samples) and self.failures == 0 and not self.error

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else 0.0

    @property
    def mean(self) -> float:
        return statistics.fmean(self.times) if self.times else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def peak_rss_kb(self) -> int:
        return max((s.rss_kb for s in self.samples), default=0)

    @property
    def bytes_out(self) -> int:
        return max((s.bytes_out for s in self.samples), default=0)

    @property
    def timeouts(self) -> int:
        return sum(1 for s in self.samples if s.exit_code == TIMED_OUT)

    @property
    def too_fast(self) -> bool:
        """Whether the timer is measuring process spawn rather than the program."""
        return bool(self.times) and self.median < FLOOR_SECONDS

    def summary(self) -> str:
        if self.error:
            return f"failed: {self.error}"
        if self.timeouts:
            return (f"{self.timeouts} of {len(self.samples)} runs hit the "
                    f"{self.timeout_used:g}s timeout - raise it or benchmark "
                    f"something smaller")
        if self.failures:
            return (f"{self.failures} of {len(self.samples)} runs exited non-zero - "
                    f"a failing command is not a fast one")
        if not self.samples:
            return "no runs"
        note = "  (near the timer's floor)" if self.too_fast else ""
        return (f"median {self.median * 1000:.1f}ms  min {self.min * 1000:.1f}ms  "
                f"stdev {self.stdev * 1000:.1f}ms  n={self.count}"
                f"  peak {self.peak_rss_kb // 1024}MB{note}")

    def to_json(self) -> dict[str, Any]:
        return {
            "command": self.command, "at": time.time(),
            "samples": [{"seconds": s.seconds, "rss_kb": s.rss_kb,
                         "exit_code": s.exit_code, "bytes_out": s.bytes_out}
                        for s in self.samples],
            "error": self.error, "timeout": self.timeout_used,
        }

    @classmethod
    def from_json(cls, raw: Any) -> Result | None:
        if not isinstance(raw, dict):
            return None
        try:
            samples = [
                Sample(seconds=float(s.get("seconds") or 0.0),
                       rss_kb=int(s.get("rss_kb") or 0),
                       exit_code=int(s.get("exit_code") or 0),
                       bytes_out=int(s.get("bytes_out") or 0))
                for s in (raw.get("samples") or []) if isinstance(s, dict)
            ]
        except (TypeError, ValueError):
            return None
        return cls(command=str(raw.get("command") or ""), samples=samples,
                   error=str(raw.get("error") or ""),
                   timeout_used=float(raw.get("timeout") or TIMEOUT))


def _rss_kb() -> int:
    try:
        return int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    except (OSError, ValueError):
        return 0


@dataclass(slots=True)
class Benchmark:
    """A command, run enough times to say something about it."""

    command: str
    runs: int = RUNS
    warmup: int = WARMUP
    timeout: float = TIMEOUT
    cwd: Path | None = None
    env: dict[str, str] | None = None

    def _once(self) -> Sample:
        argv = shlex.split(self.command)
        if not argv:
            return Sample(seconds=0.0, exit_code=-1)
        before = _rss_kb()
        started = time.perf_counter()
        try:
            done = subprocess.run(
                argv, cwd=self.cwd, env=self.env, capture_output=True,
                timeout=self.timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            # `subprocess.run` kills the child before raising, so nothing is
            # left behind - but the sample is not a measurement, it is a
            # timeout, and must not be scored as a slow run.
            return Sample(seconds=self.timeout, exit_code=TIMED_OUT)
        except (OSError, ValueError):
            return Sample(seconds=0.0, exit_code=-1)
        elapsed = time.perf_counter() - started
        return Sample(
            seconds=elapsed,
            rss_kb=max(0, _rss_kb() - before) or _rss_kb(),
            exit_code=done.returncode,
            bytes_out=len(done.stdout or b"") + len(done.stderr or b""),
        )

    def run(self) -> Result:
        if not self.command.strip():
            return Result(command=self.command, error="no command to benchmark")
        try:
            shlex.split(self.command)
        except ValueError as exc:
            return Result(command=self.command, error=f"unparseable command: {exc}")

        for _ in range(max(0, self.warmup)):
            sample = self._once()
            if not sample.ok:
                # Failing in warmup is the command being broken, not slow.
                # Reporting it now saves twelve more runs of the same failure.
                return Result(command=self.command, samples=[sample],
                          timeout_used=self.timeout)
        samples = [self._once() for _ in range(max(1, self.runs))]
        return Result(command=self.command, samples=samples,
                      timeout_used=self.timeout)


# -- comparison -------------------------------------------------------------------------

FASTER: Final = "faster"
SLOWER: Final = "slower"
SAME: Final = "indistinguishable"


@dataclass(slots=True)
class Comparison:
    before: Result
    after: Result

    @property
    def usable(self) -> bool:
        return self.before.ok and self.after.ok

    @property
    def ratio(self) -> float:
        """After over before.  Below 1.0 is faster."""
        if not self.before.median:
            return 1.0
        return self.after.median / self.before.median

    @property
    def meaningful(self) -> bool:
        """Whether the difference is bigger than the measurement's own spread.

        The central judgement of this module.  Two conditions, both required:

        The change must exceed `NOISE`, because a 1% shift on a thermally
        throttled Pi is not a result.

        And the samples must not overlap: if the faster run's slowest time is
        still slower than the other's fastest, the two distributions are
        telling the same story and the medians differ by luck.  This is a
        deliberately conservative test rather than a t-test - with twelve
        non-normal samples a p-value would be false precision, and the cost of
        wrongly declaring a winner (a ranking that moves confidently in a
        random direction) is much higher than the cost of saying "cannot tell".
        """
        if not self.usable or self.before.too_fast or self.after.too_fast:
            return False
        if abs(1.0 - self.ratio) < NOISE:
            return False
        quick, slow = ((self.after, self.before) if self.ratio < 1.0
                       else (self.before, self.after))
        return max(quick.times) < min(slow.times)

    @property
    def verdict(self) -> str:
        if not self.meaningful:
            return SAME
        return FASTER if self.ratio < 1.0 else SLOWER

    @property
    def percent(self) -> float:
        return (self.ratio - 1.0) * 100.0

    def lines(self) -> list[str]:
        out = [f"before: {self.before.summary()}", f"after:  {self.after.summary()}"]
        if not self.usable:
            out.append("cannot compare: one side did not produce a clean run")
            return out
        if self.verdict == SAME:
            why = ("the runs are too short to time"
                   if self.before.too_fast or self.after.too_fast
                   else "the samples overlap, so the medians differ by luck")
            out.append(f"verdict: indistinguishable - {why}")
        else:
            out.append(f"verdict: {abs(self.percent):.1f}% {self.verdict} "
                       f"(median {self.before.median * 1000:.1f}ms -> "
                       f"{self.after.median * 1000:.1f}ms)")
        growth = self.after.peak_rss_kb - self.before.peak_rss_kb
        if self.before.peak_rss_kb and growth > self.before.peak_rss_kb * 0.2:
            out.append(f"note: peak memory grew {growth // 1024}MB - a speedup "
                       f"bought with memory is not obviously a win")
        return out


def compare(before: Result, after: Result) -> Comparison:
    return Comparison(before=before, after=after)


def bench_criterion(comparison: Comparison | None, weight: float) -> Criterion:
    """How measured performance folds into `/spec`'s ranking.

    `applies=False` whenever there is no usable measurement, which the scoring
    module excludes from the denominator.  An unbenchmarked branch must not be
    penalised - that would make the winner depend on whether a benchmark
    happened to be configured, which is not a property of the code.  And a
    branch whose comparison came out indistinguishable scores neutral rather
    than zero, for the same reason.
    """
    if comparison is None:
        return Criterion(BENCH, 0.0, weight, "no benchmark ran", applies=False)
    if not comparison.usable:
        return Criterion(BENCH, 0.0, weight,
                         "the benchmark did not produce a clean run", applies=False)
    if not comparison.meaningful:
        return Criterion(BENCH, 0.5, weight,
                         "no measurable difference in runtime", applies=False)
    if comparison.verdict == FASTER:
        # Saturating: 2x is a clear win, 10x is not five times better a reason
        # to pick this branch over a correct one.
        gain = min(1.0, (1.0 - comparison.ratio) * 2.0)
        return Criterion(BENCH, 0.5 + 0.5 * gain, weight,
                         f"{abs(comparison.percent):.0f}% faster")
    loss = min(1.0, (comparison.ratio - 1.0))
    return Criterion(BENCH, max(0.0, 0.5 - 0.5 * loss), weight,
                     f"{abs(comparison.percent):.0f}% slower")


# -- baselines and config ----------------------------------------------------------------


def baseline_path(command: str, home: Path | None = None) -> Path:
    """Where the recorded baseline for one command lives.

    Keyed by a hash of the command so two benchmarks in one repository do not
    overwrite each other, and so a command containing a path separator cannot
    escape the directory.
    """
    root = (home if home is not None else settings.home()) / "bench"
    digest = hashlib.sha256(command.strip().encode("utf-8")).hexdigest()[:16]
    return root / f"{digest}.json"


def save_baseline(result: Result, home: Path | None = None) -> Path | None:
    path = baseline_path(result.command, home)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_json(), separators=(",", ":")),
                        encoding="utf-8")
        return path
    except OSError:
        return None


def load_baseline(command: str, home: Path | None = None) -> Result | None:
    try:
        raw = json.loads(baseline_path(command, home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return Result.from_json(raw)


def configured_command(root: Path) -> str:
    """The command a repository names for itself, or "".

    Either `{"command": "..."}` or `{"bench": "..."}`, because both are the
    obvious guess and failing on the wrong one is a pointless obstacle.
    """
    for name in CONFIG_NAMES:
        try:
            raw = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict):
            for key in ("command", "bench", "benchmark"):
                value = raw.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(raw, str) and raw.strip():
            return raw.strip()
    return ""


# -- the command ---------------------------------------------------------------------------


def _bench_command(state: Any, args: list[str]) -> Any:
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    workspace = Path(getattr(state, "workspace", None) or Path.cwd())
    #: Resolved here, on the shell's thread.  A background writer that asks
    #: `settings.home()` for itself answers with whatever the environment says
    #: by then, which for an exited shell is the wrong directory.
    home = settings.home()

    save = False
    rest = list(args)
    if rest and rest[0] in ("--save", "-s", "save"):
        save = True
        rest = rest[1:]

    command = " ".join(rest).strip() or configured_command(workspace)
    if not command:
        return Outcome([
            "nothing to benchmark",
            "give a command - /bench python3 -c 'import offset'",
            (f"or put one in {workspace / CONFIG_NAMES[0]}: "
             '{"command": "python3 -m pytest -q"}'),
        ], TONE_INFO)

    result = Benchmark(command, cwd=workspace, env=dict(os.environ)).run()
    lines = [f"benchmarked: {command}", result.summary()]

    if save:
        path = save_baseline(result, home)
        lines.append(f"recorded as the baseline ({path})" if path
                     else "could not write the baseline")
        return Outcome(lines, TONE_OK if result.ok else TONE_INFO)

    baseline = load_baseline(command, home)
    if baseline is None:
        lines += ["", ("no baseline recorded for this command yet - "
                       "run /bench --save to record this run as one")]
        return Outcome(lines, TONE_OK if result.ok else TONE_INFO)

    lines += [""] + compare(baseline, result).lines()
    return Outcome(lines, TONE_OK if result.ok else TONE_INFO)


def bench_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command("bench", "time a command against its baseline", _bench_command,
                usage="/bench [--save] [command]"),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """Built on first access.  The re-check is the guard `tasks.py` carries:
    building imports the shell registry, which re-enters this module before the
    outer call has stored anything, so one check registers every command twice.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = bench_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
