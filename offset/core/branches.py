"""Wiring speculation to actual agents.

`speculate` knows about isolation and ranking; `agent` knows about running a
turn.  This module is the only place that knows about both, which keeps the
dependency pointing one way.

The design decision worth stating: inside a throwaway worktree, branch agents
run unattended (`yolo`).  There is no human at the other end of four
simultaneous branches, and the blast radius is a directory that gets deleted.
The workspace the user actually cares about is only touched by `adopt`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from offset.core.agent import Agent, AgentConfig, RunResult
from offset.core.scoring import (
    DEFAULT_WEIGHTS,
    Baseline,
    Reviewer,
    Scorecard,
    Weights,
    measure_baseline,
    score,
)
from offset.core.session import Session
from offset.core.speculate import Approach, Attempt, Speculation, workspaces_for
from offset.tools.base import ToolContext, Toolbox
from offset.tools.builtin import builtin_tools
from offset.tools.runtime import Approval, Runtime

#: Framings that make branches genuinely different rather than four samples of
#: one idea.  Order matters: the first is the most conservative.
ANGLES: tuple[tuple[str, str], ...] = (
    ("minimal", "Make the smallest change that works. Touch as few lines as possible."),
    ("rewrite", "Rewrite the smallest unit that contains the problem, rather than patching around it."),
    ("test-first", "Write a failing test that captures the problem, then make it pass."),
    ("delete", "Prefer deleting code over adding it. Look for something to remove."),
    ("defer", "Move the work to where it is cheaper - cache it, defer it, or do it lazily."),
    ("stdlib", "Solve it with the standard library and no new abstractions."),
)


@dataclass(slots=True)
class BranchRun:
    task: str
    attempts: list[Attempt] = field(default_factory=list)
    speculation: Speculation | None = None
    #: Measured once, before the fan-out, so `regressions` has something to
    #: compare against.  `None` means nobody measured it, not "all clear".
    baseline: Baseline | None = None
    weights: Weights = DEFAULT_WEIGHTS
    reviewer: Reviewer | None = None
    _cards: list[Scorecard] | None = field(default=None, repr=False, compare=False)

    @property
    def scorecards(self) -> list[Scorecard]:
        """The weighted breakdown, best first.  Memoised: scoring shells out to
        linters, and `report()` alone asks for the ranking three times."""
        if self._cards is None:
            self._cards = score(
                self.attempts, weights=self.weights,
                reviewer=self.reviewer, baseline=self.baseline,
            )
        return self._cards

    def rescore(self, *, weights: Weights | None = None, reviewer: Reviewer | None = None) -> list[Scorecard]:
        """Re-run the scorer, optionally under different weights."""
        if weights is not None:
            self.weights = weights
        if reviewer is not None:
            self.reviewer = reviewer
        self._cards = None
        return self.scorecards

    @property
    def ranked(self) -> list[Attempt]:
        return [card.attempt for card in self.scorecards]

    @property
    def winner(self) -> Attempt | None:
        ranked = self.ranked
        return ranked[0] if ranked and ranked[0].error is None else None

    def report(self) -> list[str]:
        cards = self.scorecards
        if not cards:
            return ["no branches ran"]
        lines: list[str] = []
        for i, card in enumerate(cards, 1):
            attempt = card.attempt
            mark = {"pass": "+", "fail": "x", "idle": "?"}.get(attempt.state, "?")
            lines.append(
                f"{i}. {mark} {attempt.approach.name:<10} "
                f"{card.total:.2f}  {attempt.churn:>4} lines  {attempt.duration:>5.1f}s  "
                f"{'verified' if attempt.state == 'pass' else attempt.state}"
            )
            # The evidence, not just the verdict: a user comparing branches
            # wants the tally and the effort side by side with the score.
            if attempt.verification.counts.known:
                lines.append(f"      tests   {attempt.verification.counts.summary()}")
            if attempt.metrics.observed:
                lines.append(f"      agent   {attempt.metrics.summary()}")
            if attempt.error:
                lines.append(f"      {attempt.error[:70]}")
        if self.baseline is not None:
            lines.append("")
            lines.append(self.baseline.summary())
        lines.append("")
        lines.append(f"why {cards[0].name} leads:")
        lines.extend(cards[0].lines()[1:])
        lines.append("")
        best = cards[0].attempt
        lines.append(f"adopt the leader with /adopt 1  ({best.approach.name}, {best.churn} lines changed)")
        return lines


def approaches(count: int, task: str, models: Sequence[str] | None = None) -> list[Approach]:
    """Pick `count` distinct angles, spreading models across them if given."""
    chosen = ANGLES[: max(1, min(count, len(ANGLES)))]
    out: list[Approach] = []
    for i, (name, framing) in enumerate(chosen):
        out.append(Approach(
            name=name,
            prompt=f"{task}\n\nApproach constraint: {framing}",
            model=models[i % len(models)] if models else None,
            note=framing,
        ))
    return out


def branch_runner(
    base: AgentConfig,
    *,
    toolbox: Callable[[], Toolbox] = lambda: Toolbox(builtin_tools()),
    timeout: float = 180.0,
    api_key: str | None = None,
) -> Callable[[Approach, Path], RunResult]:
    """A runner that gives each attempt its own agent, session and toolbox."""

    def run(approach: Approach, path: Path) -> RunResult:
        session = Session.create(path / ".offset")
        runtime = Runtime(
            toolbox(),
            ToolContext(cwd=path, timeout=timeout),
            Approval(mode="yolo"),  # sandboxed worktree: nobody to ask
        )
        config = AgentConfig(
            model=approach.model or base.model,
            system=base.system,
            max_steps=base.max_steps,
            max_tokens=base.max_tokens,
            temperature=base.temperature,
            timeout=base.timeout,
        )
        agent = Agent(session, runtime, config, api_key=api_key)
        try:
            return agent.send(approach.prompt)
        finally:
            session.close()

    return run


def run_branches(
    task: str,
    count: int,
    *,
    workspace: Path | str,
    config: AgentConfig,
    models: Sequence[str] | None = None,
    verify_command: str | None = None,
    api_key: str | None = None,
    keep: bool = True,
    parallel: bool = True,
    baseline: bool = True,
    reviewer: Reviewer | None = None,
) -> BranchRun:
    """Try `count` approaches to `task` in isolation and rank what happened."""
    workspace = Path(workspace).resolve()
    spec = Speculation(
        workspace,
        spaces=workspaces_for(workspace),
        verify_command=verify_command,
        keep=keep,
    )
    # Before the fan-out, and only once: the branches are seeded from the same
    # base, so measuring it afterwards would race their worktrees for no gain.
    # Without this the regression criterion can never fire, which is the whole
    # difference between "these tests fail" and "you broke these tests".
    before = measure_baseline(spec) if baseline and verify_command else None
    plan = approaches(count, task, models)
    attempts = spec.run(plan, branch_runner(config, api_key=api_key), parallel=parallel)
    return BranchRun(task=task, attempts=attempts, speculation=spec,
                     baseline=before, reviewer=reviewer)
