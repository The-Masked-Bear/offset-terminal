"""Speculative branching: isolation, verification, ranking, adoption.

These run against real git repositories and real subprocesses, because the
claim under test is precisely that two attempts cannot see each other's edits.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from offset.core.speculate import (
    Approach,
    Attempt,
    CopyWorkspaces,
    GitWorktrees,
    Speculation,
    Verification,
    workspaces_for,
)


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    git("init", "-q", cwd=root)
    git("config", "user.email", "t@t", cwd=root)
    git("config", "user.name", "t", cwd=root)
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    git("add", "-A", cwd=root)
    git("commit", "-qm", "initial", cwd=root)
    return root


@pytest.fixture()
def plain(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return root


def writer(text: str):
    def run(approach: Approach, path: Path):
        (path / "app.py").write_text(text, encoding="utf-8")
        return f"wrote {len(text)} bytes"

    return run


# -- strategy selection -----------------------------------------------------


def test_git_repositories_get_worktrees(repo):
    assert GitWorktrees.usable(repo)
    assert isinstance(workspaces_for(repo), GitWorktrees)


def test_plain_directories_fall_back_to_copies(plain):
    assert not GitWorktrees.usable(plain)
    assert isinstance(workspaces_for(plain), CopyWorkspaces)


# -- isolation --------------------------------------------------------------


def test_attempts_cannot_see_each_other(repo):
    spec = Speculation(repo, keep=True)
    seen: dict[str, str] = {}

    def run(approach: Approach, path: Path):
        (path / "app.py").write_text(f"VALUE = {approach.name}\n", encoding="utf-8")
        time.sleep(0.05)  # give the other attempt a chance to interfere
        seen[approach.name] = (path / "app.py").read_text()
        return None

    attempts = spec.run([Approach("a", "p"), Approach("b", "p"), Approach("c", "p")], run)
    assert seen == {"a": "VALUE = a\n", "b": "VALUE = b\n", "c": "VALUE = c\n"}
    assert len({a.path for a in attempts}) == 3
    assert (repo / "app.py").read_text() == "VALUE = 1\n", "the real workspace was modified"
    spec.cleanup(attempts)


def test_uncommitted_work_is_carried_into_the_branch(repo):
    """Branching must work mid-task, not only from a clean tree."""
    (repo / "app.py").write_text("VALUE = 1\nDIRTY = True\n", encoding="utf-8")
    spec = Speculation(repo, keep=True)
    captured = {}

    def run(approach, path):
        captured["content"] = (path / "app.py").read_text()

    attempts = spec.run([Approach("dirty", "p")], run)
    assert "DIRTY = True" in captured["content"]
    spec.cleanup(attempts)


def test_new_files_appear_in_the_diff(repo):
    spec = Speculation(repo)

    def run(approach, path):
        (path / "brand_new.py").write_text("print('hi')\n", encoding="utf-8")

    [attempt] = spec.run([Approach("adds", "p")], run)
    assert "brand_new.py" in attempt.diff
    spec.cleanup([attempt])


def test_copy_strategy_isolates_and_diffs(plain):
    spec = Speculation(plain, spaces=CopyWorkspaces(plain))
    [attempt] = spec.run([Approach("x", "p")], writer("VALUE = 42\n"))
    assert "-VALUE = 1" in attempt.diff and "+VALUE = 42" in attempt.diff
    assert (plain / "app.py").read_text() == "VALUE = 1\n"
    spec.cleanup([attempt])


# -- verification -----------------------------------------------------------


def test_verification_decides_pass_and_fail(repo):
    spec = Speculation(repo, verify_command="python3 -c \"import app; assert app.VALUE == 2\"")
    attempts = spec.run([
        Approach("right", "p"),
        Approach("wrong", "p"),
    ], lambda a, p: (p / "app.py").write_text(f"VALUE = {2 if a.name == 'right' else 99}\n", encoding="utf-8"))

    by_name = {a.approach.name: a for a in attempts}
    assert by_name["right"].verification.ok and by_name["right"].state == "pass"
    assert not by_name["wrong"].verification.ok and by_name["wrong"].state == "fail"
    spec.cleanup(attempts)


def test_verification_is_marked_skipped_when_unconfigured(repo):
    spec = Speculation(repo)
    [attempt] = spec.run([Approach("a", "p")], writer("VALUE = 7\n"))
    assert attempt.verification.skipped and attempt.state == "idle"
    spec.cleanup([attempt])


def test_a_hanging_verification_is_cut_off(repo):
    spec = Speculation(repo, verify_command="sleep 30", verify_timeout=0.4)
    started = time.monotonic()
    [attempt] = spec.run([Approach("slow", "p")], writer("VALUE = 3\n"))
    assert not attempt.verification.ok and "exceeded" in attempt.verification.output
    assert time.monotonic() - started < 5.0
    spec.cleanup([attempt])


def test_a_crashing_runner_becomes_a_failed_attempt(repo):
    def explode(approach, path):
        raise RuntimeError("the model gave up")

    spec = Speculation(repo)
    [attempt] = spec.run([Approach("boom", "p")], explode)
    assert attempt.error and "the model gave up" in attempt.error
    assert attempt.state == "fail"
    spec.cleanup([attempt])


def test_one_failing_attempt_does_not_stop_the_others(repo):
    def sometimes(approach, path):
        if approach.name == "bad":
            raise RuntimeError("nope")
        (path / "app.py").write_text("VALUE = 5\n", encoding="utf-8")

    spec = Speculation(repo)
    attempts = spec.run([Approach("bad", "p"), Approach("good", "p")], sometimes)
    assert attempts[0].error and not attempts[1].error
    spec.cleanup(attempts)


# -- ranking ----------------------------------------------------------------


def make(name: str, *, ok: bool, skipped: bool = False, churn_lines: int = 0, duration: float = 1.0) -> Attempt:
    diff = "".join(f"+line{i}\n" for i in range(churn_lines))
    return Attempt(
        approach=Approach(name, "p"),
        diff=diff,
        verification=Verification(ok=ok, skipped=skipped),
        duration=duration,
    )


def test_ranking_prefers_evidence_then_smaller_change():
    ranked = Speculation.rank([
        make("big-pass", ok=True, churn_lines=50),
        make("failed", ok=False, churn_lines=1),
        make("small-pass", ok=True, churn_lines=2),
        make("unverified", ok=True, skipped=True, churn_lines=1),
    ])
    assert [a.approach.name for a in ranked] == ["small-pass", "big-pass", "unverified", "failed"]


def test_ranking_breaks_ties_by_speed_then_name():
    ranked = Speculation.rank([
        make("b", ok=True, churn_lines=2, duration=5.0),
        make("a", ok=True, churn_lines=2, duration=5.0),
        make("c", ok=True, churn_lines=2, duration=0.5),
    ])
    assert [a.approach.name for a in ranked] == ["c", "a", "b"]


def test_churn_counts_only_real_changes():
    attempt = Attempt(Approach("x", "p"), diff=(
        "--- a/f.py\n+++ b/f.py\n@@ -1 +1,2 @@\n-old line\n+new line\n+another\n"
    ))
    assert attempt.churn == 3


# -- adoption ---------------------------------------------------------------


def test_adopting_a_winner_lands_it_in_the_real_workspace(repo):
    spec = Speculation(repo, keep=True)
    attempts = spec.run([Approach("winner", "p")], writer("VALUE = 99\n"))
    ok, message = spec.adopt(attempts[0])
    assert ok, message
    assert (repo / "app.py").read_text() == "VALUE = 99\n"
    spec.cleanup(attempts)


def test_adopting_from_a_copy_workspace(plain):
    spec = Speculation(plain, spaces=CopyWorkspaces(plain), keep=True)
    attempts = spec.run([Approach("w", "p")], writer("VALUE = 77\n"))
    ok, _ = spec.adopt(attempts[0])
    assert ok and (plain / "app.py").read_text() == "VALUE = 77\n"
    spec.cleanup(attempts)


def test_adopting_a_no_op_is_harmless(repo):
    spec = Speculation(repo, keep=True)
    attempts = spec.run([Approach("idle", "p")], lambda a, p: None)
    ok, message = spec.adopt(attempts[0])
    assert ok and "nothing" in message
    spec.cleanup(attempts)


# -- housekeeping -----------------------------------------------------------


def test_cleanup_removes_every_worktree(repo):
    spec = Speculation(repo)
    attempts = spec.run([Approach("a", "p"), Approach("b", "p")], writer("VALUE = 2\n"))
    paths = [a.path for a in attempts]
    spec.cleanup(attempts)
    assert not any(p.exists() for p in paths)
    assert "branches" not in git("worktree", "list", cwd=repo).stdout


def test_keep_leaves_branches_for_inspection(repo):
    spec = Speculation(repo, keep=True)
    attempts = spec.run([Approach("keeper", "p")], writer("VALUE = 4\n"))
    spec.cleanup(attempts)
    assert attempts[0].path.exists()
    Speculation(repo).cleanup(attempts)


def test_awkward_names_become_safe_directories(repo):
    spec = Speculation(repo)
    [attempt] = spec.run([Approach("Rewrite The Parser!! / v2", "p")], writer("VALUE = 8\n"))
    assert attempt.path is not None and attempt.path.exists()
    assert "/" not in attempt.path.name and " " not in attempt.path.name
    spec.cleanup([attempt])


def test_attempts_actually_overlap_in_time(repo):
    """Parallel means parallel: three half-second attempts must not take 1.5s."""
    def slow(approach, path):
        time.sleep(0.4)
        (path / "app.py").write_text(f"VALUE = {approach.name!r}\n", encoding="utf-8")

    spec = Speculation(repo)
    started = time.monotonic()
    attempts = spec.run([Approach(n, "p") for n in ("a", "b", "c")], slow)
    assert time.monotonic() - started < 1.1
    spec.cleanup(attempts)


def test_summary_reads_like_a_verdict(repo):
    spec = Speculation(repo, verify_command="true")
    [attempt] = spec.run([Approach("tidy", "p")], writer("VALUE = 2\n"))
    assert attempt.summary().startswith("tidy: passed")
    spec.cleanup([attempt])
