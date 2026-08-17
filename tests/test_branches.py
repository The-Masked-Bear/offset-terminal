"""Speculative branching, end to end through real agents.

The provider is scripted, but everything else is real: real git worktrees, real
tool calls writing real files, a real verification command, real ranking.  This
is the test that would have caught `/spec` being a no-op.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from offset.core.agent import AgentConfig
from offset.core.branches import ANGLES, BranchRun, approaches, branch_runner, run_branches
from offset.providers.mock import Mock, script
from offset.providers.registry import PROVIDERS
from offset.core.speculate import Approach, Speculation


def git(*args: str, cwd: Path):
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
def writing_mock(monkeypatch):
    """A model that edits a file, so branches produce genuine diffs."""

    def factory():
        return Mock([
            script(tool_calls=[("c1", "write", {"path": "app.py", "content": "VALUE = 2\n"})]),
            script("changed VALUE to 2"),
        ])

    monkeypatch.setitem(PROVIDERS, "mock", factory)
    return factory


@pytest.fixture()
def config():
    return AgentConfig(model="mock", max_steps=4, max_tokens=200, system="terse")


# -- planning ---------------------------------------------------------------


def test_approaches_are_actually_different():
    plan = approaches(4, "make the parser faster")
    assert len(plan) == 4
    assert len({a.name for a in plan}) == 4
    assert len({a.prompt for a in plan}) == 4, "the branches would be four samples of one idea"
    assert all("make the parser faster" in a.prompt for a in plan)


def test_approach_count_is_bounded_by_the_angles_available():
    assert len(approaches(99, "task")) == len(ANGLES)
    assert len(approaches(0, "task")) == 1


def test_models_are_spread_across_approaches():
    plan = approaches(4, "task", ["a", "b"])
    assert [a.model for a in plan] == ["a", "b", "a", "b"]


def test_without_models_each_approach_uses_the_session_model():
    assert all(a.model is None for a in approaches(3, "task"))


# -- running ----------------------------------------------------------------


def test_branches_really_run_and_really_edit_their_own_worktree(repo, writing_mock, config):
    run = run_branches("bump the value", 3, workspace=repo, config=config, keep=True)

    assert len(run.attempts) == 3
    for attempt in run.attempts:
        assert attempt.error is None, attempt.error
        assert attempt.path is not None and attempt.path.exists()
        assert (attempt.path / "app.py").read_text() == "VALUE = 2\n"
        assert "VALUE = 2" in attempt.diff, "the edit did not show up in the diff"
    assert (repo / "app.py").read_text() == "VALUE = 1\n", "the real workspace was modified"
    run.speculation.keep = False
    run.speculation.cleanup(run.attempts)


def test_each_branch_gets_its_own_session(repo, writing_mock, config):
    run = run_branches("bump it", 2, workspace=repo, config=config, keep=True)
    sessions = [list((a.path / ".offset").glob("*.jsonl")) for a in run.attempts]
    assert all(len(s) == 1 for s in sessions), "each attempt needs its own history"
    assert sessions[0][0] != sessions[1][0]
    run.speculation.keep = False
    run.speculation.cleanup(run.attempts)


def test_verification_ranks_the_branches(repo, writing_mock, config):
    run = run_branches(
        "bump it", 2,
        workspace=repo, config=config,
        verify_command="python3 -c \"import app; assert app.VALUE == 2\"",
        keep=True,
    )
    assert all(a.verification.ok for a in run.attempts), [a.verification.output for a in run.attempts]
    assert all(a.state == "pass" for a in run.attempts)
    assert run.winner is not None
    run.speculation.keep = False
    run.speculation.cleanup(run.attempts)


def test_a_failing_verification_sinks_a_branch(repo, writing_mock, config):
    run = run_branches(
        "bump it", 2,
        workspace=repo, config=config,
        verify_command="python3 -c \"import app; assert app.VALUE == 99\"",
        keep=True,
    )
    assert all(not a.verification.ok for a in run.attempts)
    assert all(a.state == "fail" for a in run.attempts)
    run.speculation.keep = False
    run.speculation.cleanup(run.attempts)


def test_the_report_names_the_leader_and_how_to_take_it(repo, writing_mock, config):
    run = run_branches("bump it", 2, workspace=repo, config=config, verify_command="true", keep=True)
    report = run.report()
    assert any("/adopt 1" in line for line in report)
    assert any(a.approach.name in line for a in run.attempts for line in report)
    run.speculation.keep = False
    run.speculation.cleanup(run.attempts)


def test_adopting_the_winner_changes_the_real_workspace(repo, writing_mock, config):
    run = run_branches("bump it", 2, workspace=repo, config=config, verify_command="true", keep=True)
    ok, message = run.speculation.adopt(run.winner)
    assert ok, message
    assert (repo / "app.py").read_text() == "VALUE = 2\n"
    run.speculation.keep = False
    run.speculation.cleanup(run.attempts)


def test_branch_agents_do_not_stop_to_ask_permission(repo, writing_mock, config):
    """A write inside a throwaway worktree must not block on an approver."""
    runner = branch_runner(config)
    spec = Speculation(repo, keep=True)
    attempt = spec.attempt(Approach("minimal", "bump the value"), runner)
    assert attempt.error is None
    assert isinstance(attempt.detail.invocations[0].result.ok, bool)
    assert attempt.detail.invocations[0].result.ok, attempt.detail.invocations[0].result.error
    spec.keep = False
    spec.cleanup([attempt])


def test_a_run_with_no_attempts_reports_honestly():
    empty = BranchRun(task="nothing")
    assert empty.winner is None
    assert empty.report() == ["no branches ran"]


def test_branches_work_in_a_directory_that_is_not_a_repository(tmp_path, writing_mock, config):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    run = run_branches("bump it", 2, workspace=plain, config=config, keep=True)
    assert all(a.error is None for a in run.attempts)
    assert all("VALUE = 2" in a.diff for a in run.attempts)
    assert (plain / "app.py").read_text() == "VALUE = 1\n"
    run.speculation.keep = False
    run.speculation.cleanup(run.attempts)
