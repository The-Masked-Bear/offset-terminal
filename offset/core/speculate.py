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

Nothing is adopted automatically.  Ranking proposes; the human disposes.
"""

from __future__ import annotations

import difflib
import os
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


@dataclass(slots=True)
class Attempt:
    approach: Approach
    path: Path | None = None
    diff: str = ""
    verification: Verification = field(default_factory=Verification)
    error: str | None = None
    duration: float = 0.0
    detail: object = None  # whatever the agent factory returned

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

    __slots__ = ("root", "spaces", "verify_command", "verify_timeout", "keep")

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
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return Verification(
            ok=proc.returncode == 0,
            output=output[-4000:],
            command=str(command),
            duration=time.monotonic() - started,
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
