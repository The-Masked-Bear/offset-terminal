"""Speculative branching: try several approaches at once, keep what passes.

The idea in one line — checkpoint the workspace, let N approaches (often N
different models) each attempt the task in a private copy, run the same
verification command against all of them, and rank the survivors by evidence
rather than by argument.

Isolation is the whole game.  Two agents editing one directory produce a mess
nobody can attribute, so each attempt gets its own filesystem:

  * `GitWorktrees` — a detached worktree per attempt, seeded from the current
    state *including uncommitted work* (via `git stash create`), which is what
    makes this usable mid-task rather than only on a clean tree;
  * `CopyWorkspaces` — a filtered directory copy, for workspaces that are not
    git repositories at all.

Evidence is only useful if it is kept.  A branch used to throw away everything
it learned except an exit code, so `observe` projects the runner's payload onto
`BranchMetrics` and `parse_test_output` turns a verification's own summary into
`TestCounts`.  Both are read once, when the `Attempt` is assembled, because an
untyped `detail` object re-interpreted at every call site is a bug waiting for
a runner that returns something else.

Nothing is adopted automatically.  Ranking proposes; the human disposes.
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

SKIP = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache", ".offset", "dist", "build"}


def _run(args: Sequence[str], cwd: Path | str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args), cwd=str(cwd), capture_output=True, text=True, timeout=timeout, errors="replace"
    )


# -- isolation strategies ---------------------------------------------------


class Workspaces(ABC):
    """Creates and tears down isolated copies of a workspace."""

    root: Path

    @abstractmethod
    def create(self, name: str) -> Path: ...

    @abstractmethod
    def diff(self, path: Path) -> str: ...

    @abstractmethod
    def destroy(self, path: Path) -> None: ...

    @abstractmethod
    def adopt(self, path: Path) -> tuple[bool, str]: ...


class GitWorktrees(Workspaces):
    """One detached git worktree per attempt, seeded from the live state."""

    def __init__(self, root: Path, container: Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.container = Path(container) if container else self.root / ".offset" / "branches"
        self._base: str | None = None

    @staticmethod
    def usable(root: Path) -> bool:
        got = _run(["git", "rev-parse", "--is-inside-work-tree"], root)
        return got.returncode == 0 and got.stdout.strip() == "true"

    def base(self) -> str:
        """A commit representing the workspace *right now*, dirt included."""
        if self._base:
            return self._base
        stashed = _run(["git", "stash", "create"], self.root)
        candidate = stashed.stdout.strip()
        if not candidate:
            head = _run(["git", "rev-parse", "HEAD"], self.root)
            candidate = head.stdout.strip()
        if not candidate:
            raise RuntimeError("the repository has no commits to branch from")
        self._base = candidate
        return candidate

    def create(self, name: str) -> Path:
        self.container.mkdir(parents=True, exist_ok=True)
        path = self.container / name
        if path.exists():
            self.destroy(path)
        got = _run(["git", "worktree", "add", "--detach", str(path), self.base()], self.root)
        if got.returncode != 0:
            raise RuntimeError(f"could not create worktree {name}: {got.stderr.strip()}")
        return path

    def diff(self, path: Path) -> str:
        # Staging inside a throwaway worktree is free, and it is the only way
        # to get new files into the diff.
        _run(["git", "add", "-A"], path)
        got = _run(["git", "diff", "--cached", self.base()], path)
        return got.stdout

    def destroy(self, path: Path) -> None:
        _run(["git", "worktree", "remove", "--force", str(path)], self.root)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    def adopt(self, path: Path) -> tuple[bool, str]:
        patch = self.diff(path)
        if not patch.strip():
            return True, "nothing to adopt"
        proc = subprocess.run(
            ["git", "apply", "--3way", "-"],
            cwd=str(self.root), input=patch, capture_output=True, text=True,
        )
        return proc.returncode == 0, (proc.stderr or "applied").strip()


class CopyWorkspaces(Workspaces):
    """A filtered copy per attempt, for directories that are not repositories."""

    def __init__(self, root: Path, container: Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.container = Path(container) if container else self.root / ".offset" / "branches"

    def create(self, name: str) -> Path:
        self.container.mkdir(parents=True, exist_ok=True)
        path = self.container / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        shutil.copytree(self.root, path, ignore=shutil.ignore_patterns(*SKIP), symlinks=True)
        return path

    def _files(self, base: Path) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for p in base.rglob("*"):
            rel = p.relative_to(base)
            if p.is_file() and not any(part in SKIP for part in rel.parts):
                out[str(rel)] = p
        return out

    def diff(self, path: Path) -> str:
        before, after = self._files(self.root), self._files(path)
        chunks: list[str] = []
        for rel in sorted(set(before) | set(after)):
            old = _text(before.get(rel))
            new = _text(after.get(rel))
            if old == new:
                continue
            chunks.extend(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}",
            ))
        return "".join(chunks)

    def destroy(self, path: Path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def adopt(self, path: Path) -> tuple[bool, str]:
        before, after = self._files(self.root), self._files(path)
        copied = 0
        for rel, src in after.items():
            target = self.root / rel
            if rel not in before or _text(src) != _text(before[rel]):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                copied += 1
        return True, f"copied {copied} file(s)"


def _text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def workspaces_for(root: Path) -> Workspaces:
    """Prefer worktrees; fall back to copies when this is not a repository."""
    root = Path(root)
    return GitWorktrees(root) if GitWorktrees.usable(root) else CopyWorkspaces(root)


# -- observation ------------------------------------------------------------


@dataclass(slots=True)
class TestCounts:
    """How many tests ran, when the runner said so.

    Every field is `None` rather than `0` when the output could not be parsed.
    "nothing failed" and "we have no idea whether anything failed" must not
    compare equal: the first is evidence, the second is silence, and a scorer
    that confuses them will happily crown a branch whose suite never ran.
    """

    passed: int | None = None
    failed: int | None = None
    skipped: int | None = None
    total: int | None = None

    @property
    def known(self) -> bool:
        return self.total is not None

    def summary(self) -> str:
        if not self.known:
            return "no test counts"
        parts = [f"{self.passed or 0} passed"]
        if self.failed:
            parts.append(f"{self.failed} failed")
        if self.skipped:
            parts.append(f"{self.skipped} skipped")
        return f"{', '.join(parts)} of {self.total}"


#: pytest and cargo both spell their tally as `N word`; the words differ.
_TALLY = re.compile(r"(\d+)\s+([a-z]+)")
_PYTEST_WORDS = {
    "passed": "passed", "xpassed": "passed",
    "failed": "failed", "error": "failed", "errors": "failed",
    "skipped": "skipped", "xfailed": "skipped", "deselected": "skipped",
}
_JEST = re.compile(r"^Tests:\s+([^\n]*?)(\d+)\s+total\s*$", re.M)
_JEST_WORDS = re.compile(r"(\d+)\s+(passed|failed|skipped|todo)")
_CARGO = re.compile(r"test result:\s*(?:ok|FAILED)\.\s*(\d+) passed;\s*(\d+) failed;\s*(\d+) ignored")
_GO = re.compile(r"^\s*--- (PASS|FAIL|SKIP): ", re.M)
_RAN = re.compile(r"^Ran (\d+) tests?\b", re.M)
_VERDICT = re.compile(r"^(?:OK|FAILED)\b(.*)$", re.M)
_KEYED = re.compile(r"(failures|errors|skipped)=(\d+)")


def parse_test_output(text: str) -> TestCounts:
    """Read a test runner's own summary rather than guessing from its exit code.

    Five formats cover nearly everything a project actually verifies with:
    jest, cargo, `go test -v`, pytest and unittest.  They are tried
    most-specific first, because the generic `N passed` scan that catches
    pytest would also half-read the others and report the wrong skip count.
    Unrecognised output yields empty counts — absent, not zero.
    """
    if not text:
        return TestCounts()
    for parser in (_jest_counts, _cargo_counts, _go_counts, _pytest_counts, _unittest_counts):
        counts = parser(text)
        if counts is not None:
            return _complete(counts)
    return TestCounts()


def _complete(counts: TestCounts) -> TestCounts:
    known = [n for n in (counts.passed, counts.failed, counts.skipped) if n is not None]
    if counts.total is None and known:
        counts.total = sum(known)
    if counts.passed is None and counts.total is not None:
        counts.passed = max(0, counts.total - (counts.failed or 0) - (counts.skipped or 0))
    return counts


def _jest_counts(text: str) -> TestCounts | None:
    found = _JEST.search(text)
    if found is None:
        return None
    counts = TestCounts(total=int(found.group(2)))
    for number, word in _JEST_WORDS.findall(found.group(1)):
        slot = "skipped" if word == "todo" else word
        setattr(counts, slot, (getattr(counts, slot) or 0) + int(number))
    return counts


def _cargo_counts(text: str) -> TestCounts | None:
    # A workspace prints one `test result:` line per test binary; sum them.
    hits = _CARGO.findall(text)
    if not hits:
        return None
    passed = sum(int(h[0]) for h in hits)
    failed = sum(int(h[1]) for h in hits)
    skipped = sum(int(h[2]) for h in hits)
    return TestCounts(passed, failed, skipped, passed + failed + skipped)


def _go_counts(text: str) -> TestCounts | None:
    marks = _GO.findall(text)
    if not marks:
        return None
    return TestCounts(marks.count("PASS"), marks.count("FAIL"), marks.count("SKIP"), len(marks))


def _pytest_counts(text: str) -> TestCounts | None:
    # The tally is the LAST such line: a rerun or a `-p no:randomly` banner
    # earlier in the output must not win over the summary at the bottom.
    for line in reversed(text.splitlines()):
        tally = [(int(n), _PYTEST_WORDS[w]) for n, w in _TALLY.findall(line) if w in _PYTEST_WORDS]
        if not tally:
            continue
        counts = TestCounts()
        for number, slot in tally:
            setattr(counts, slot, (getattr(counts, slot) or 0) + number)
        return counts
    return None


def _unittest_counts(text: str) -> TestCounts | None:
    ran = _RAN.search(text)
    verdicts = list(_VERDICT.finditer(text))
    if ran is None or not verdicts:
        return None
    total = int(ran.group(1))
    keyed = {key: int(value) for key, value in _KEYED.findall(verdicts[-1].group(1))}
    failed = keyed.get("failures", 0) + keyed.get("errors", 0)
    skipped = keyed.get("skipped", 0)
    return TestCounts(max(0, total - failed - skipped), failed, skipped, total)


@dataclass(slots=True)
class BranchMetrics:
    """What the agent actually did on a branch.

    `Attempt.detail` carries whatever the runner returned — a `RunResult` in
    practice, but the type stays `object` so a branch can be driven by anything
    at all.  This record is the typed projection of it, read exactly once by
    `observe`, so no consumer has to go spelunking in an untyped payload.
    """

    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    steps: int = 0
    stop_reason: str = ""
    tool_calls: int = 0
    tools: tuple[str, ...] = ()
    tool_failures: int = 0
    tool_time: float = 0.0
    error: str | None = None

    @property
    def observed(self) -> bool:
        """False when the runner told us nothing: absent, not healthy."""
        return bool(self.steps or self.tool_calls or self.stop_reason or self.tokens)

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def summary(self) -> str:
        if not self.observed:
            return "no agent metrics"
        parts = [f"{self.steps} step{'' if self.steps == 1 else 's'}", f"{self.tokens} tokens"]
        if self.tokens_cached:
            parts.append(f"{self.tokens_cached} cached")
        if self.tool_calls:
            failed = f", {self.tool_failures} failed" if self.tool_failures else ""
            parts.append(f"{self.tool_calls} tool call{'' if self.tool_calls == 1 else 's'}{failed}")
        if self.tools:
            parts.append("/".join(self.tools))
        if self.stop_reason and self.stop_reason != "stop":
            parts.append(f"stopped: {self.stop_reason}")
        return ", ".join(parts)


def _number(value: object) -> float:
    """A number from an untyped payload, or nothing.  `bool` is not a count."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def observe(detail: object) -> BranchMetrics:
    """Project an untyped runner payload onto typed fields, never raising.

    Defensive by contract.  `Runner` returns `object`, so every field is
    fetched with `getattr` and coerced: a runner that hands back `None`, a
    string or a test double yields empty metrics instead of exploding on a
    branch that has just spent a minute of real work.
    """
    metrics = BranchMetrics()
    if detail is None:
        return metrics
    usage = getattr(detail, "usage", None)
    metrics.tokens_in = int(_number(getattr(usage, "input", 0)))
    metrics.tokens_out = int(_number(getattr(usage, "output", 0)))
    metrics.tokens_cached = int(
        _number(getattr(usage, "cache_read", 0)) + _number(getattr(usage, "cache_write", 0))
    )
    metrics.steps = int(_number(getattr(detail, "steps", 0)))
    reason = getattr(detail, "stop_reason", "")
    metrics.stop_reason = reason if isinstance(reason, str) else ""
    failure = getattr(detail, "error", None)
    metrics.error = failure if isinstance(failure, str) else None
    invocations = getattr(detail, "invocations", None)
    # Only walk a concrete sequence: iterating an arbitrary object could
    # consume a generator the caller still needs, or block forever.
    seen: list[str] = []
    for call in invocations if isinstance(invocations, (list, tuple)) else ():
        metrics.tool_calls += 1
        name = getattr(getattr(call, "call", None), "name", "")
        if isinstance(name, str) and name and name not in seen:
            seen.append(name)
        result = getattr(call, "result", None)
        if getattr(result, "ok", True) is False:
            metrics.tool_failures += 1
        metrics.tool_time += _number(getattr(result, "duration", 0.0))
    metrics.tools = tuple(sorted(seen))
    return metrics


# -- attempts ---------------------------------------------------------------


@dataclass(slots=True)
class Approach:
    """One way of trying the task."""

    name: str
    prompt: str
    model: str | None = None
    note: str = ""


@dataclass(slots=True)
class Verification:
    ok: bool = True
    output: str = ""
    command: str = ""
    duration: float = 0.0
    skipped: bool = False
    #: `skipped` above already means "no verification ran at all", so the
    #: parsed tally lives in its own record rather than colliding with it.
    counts: TestCounts = field(default_factory=TestCounts)


@dataclass(slots=True)
class Attempt:
    approach: Approach
    path: Path | None = None
    diff: str = ""
    verification: Verification = field(default_factory=Verification)
    error: str | None = None
    duration: float = 0.0
    detail: object = None  # whatever the agent factory returned
    metrics: BranchMetrics = field(default_factory=BranchMetrics)

    @property
    def ok(self) -> bool:
        return self.error is None and self.verification.ok

    @property
    def churn(self) -> int:
        """Changed lines — the tie-breaker when two attempts both pass."""
        return sum(1 for line in self.diff.splitlines() if line[:1] in "+-" and not line.startswith(("+++", "---")))

    @property
    def state(self) -> str:
        if self.error:
            return "fail"
        if self.verification.skipped:
            return "idle"
        return "pass" if self.verification.ok else "fail"

    def summary(self) -> str:
        if self.error:
            return f"{self.approach.name}: {self.error[:60]}"
        verdict = "skipped" if self.verification.skipped else ("passed" if self.verification.ok else "failed")
        return f"{self.approach.name}: {verdict}, {self.churn} lines, {self.duration:.1f}s"


#: Called with (approach, worktree path); returns anything, raises to fail.
Runner = Callable[[Approach, Path], object]


class Speculation:
    """Runs approaches in isolation and ranks them by what actually happened."""

    __slots__ = ("keep", "root", "spaces", "verify_command", "verify_timeout")

    def __init__(
        self,
        root: Path | str,
        *,
        spaces: Workspaces | None = None,
        verify_command: Sequence[str] | str | None = None,
        verify_timeout: float = 300.0,
        keep: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.spaces = spaces or workspaces_for(self.root)
        self.verify_command = verify_command
        self.verify_timeout = verify_timeout
        self.keep = keep  # leave worktrees on disk for inspection

    # -- one attempt ------------------------------------------------------

    def attempt(self, approach: Approach, runner: Runner) -> Attempt:
        started = time.monotonic()
        record = Attempt(approach)
        try:
            record.path = self.spaces.create(_slug(approach.name))
        except Exception as exc:
            record.error = f"could not isolate: {exc}"
            record.duration = time.monotonic() - started
            return record
        try:
            record.detail = runner(approach, record.path)
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
        # Read the payload once, here, where the only Attempt is assembled.
        record.metrics = observe(record.detail)
        try:
            record.diff = self.spaces.diff(record.path)
        except Exception as exc:
            record.error = record.error or f"could not diff: {exc}"
        record.verification = self.verify(record.path)
        record.duration = time.monotonic() - started
        return record

    def verify(self, path: Path) -> Verification:
        if not self.verify_command:
            return Verification(ok=True, skipped=True, output="no verification command configured")
        command = self.verify_command
        started = time.monotonic()
        try:
            proc = subprocess.run(
                command if isinstance(command, str) else list(command),
                shell=isinstance(command, str),
                cwd=str(path), capture_output=True, text=True,
                timeout=self.verify_timeout, errors="replace",
            )
        except subprocess.TimeoutExpired:
            return Verification(False, f"verification exceeded {self.verify_timeout:g}s",
                                str(command), time.monotonic() - started)
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()[-4000:]
        return Verification(
            ok=proc.returncode == 0,
            output=output,
            command=str(command),
            duration=time.monotonic() - started,
            counts=parse_test_output(output),
        )

    # -- the fan-out ------------------------------------------------------

    def run(self, approaches: Sequence[Approach], runner: Runner, *, parallel: bool = True) -> list[Attempt]:
        """Try every approach.  Results come back in the order given."""
        if not approaches:
            return []
        if not parallel or len(approaches) == 1:
            return [self.attempt(a, runner) for a in approaches]
        with ThreadPoolExecutor(max_workers=min(8, len(approaches))) as pool:
            return list(pool.map(lambda a: self.attempt(a, runner), approaches))

    @staticmethod
    def rank(attempts: Iterable[Attempt]) -> list[Attempt]:
        """Passing first, then smaller changes, then faster.  Deterministic."""
        return sorted(
            attempts,
            key=lambda a: (
                0 if a.ok and not a.verification.skipped else (1 if a.ok else 2),
                a.churn,
                round(a.duration, 3),
                a.approach.name,
            ),
        )

    def adopt(self, attempt: Attempt) -> tuple[bool, str]:
        """Apply an attempt's changes to the real workspace."""
        if attempt.path is None:
            return False, "attempt never got a workspace"
        return self.spaces.adopt(attempt.path)

    def cleanup(self, attempts: Iterable[Attempt]) -> None:
        if self.keep:
            return
        for attempt in attempts:
            if attempt.path is not None:
                self.spaces.destroy(attempt.path)


def _slug(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in name.strip().lower())
    return safe.strip("-")[:40] or "attempt"
