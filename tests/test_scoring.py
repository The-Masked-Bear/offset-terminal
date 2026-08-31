"""Scoring branches: what was measured, and why the winner won.

The point of this layer is that a verdict comes with its reasons, so most of
these tests assert on a `Criterion` reason as well as on the ordering. Real
subprocesses throughout: a genuine git repo for the baseline, a genuine
executable for the static analyser. Only the reviewer is stubbed, because it is
the one thing that would otherwise reach a model.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from offset.core.agent import RunResult
from offset.core.scoring import (
    DEFAULT_LINTERS,
    DIFF,
    HEALTH,
    REGRESSIONS,
    REVIEW,
    STATIC,
    VERIFICATION,
    Analysis,
    Baseline,
    Linter,
    Weights,
    council_reviewer,
    measure_baseline,
    score,
)
from offset.core.speculate import (
    Approach,
    Attempt,
    BranchMetrics,
    Speculation,
    TestCounts as Counts,  # aliased: pytest tries to collect anything called Test*
    Verification,
    observe,
    parse_test_output,
)
from offset.providers.base import ToolCall, Usage
from offset.tools.base import ToolResult
from offset.tools.runtime import Invocation


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))


def make(
    name: str,
    *,
    ok: bool = True,
    skipped: bool = False,
    output: str = "",
    churn: int = 4,
    duration: float = 1.0,
    error: str | None = None,
    detail: object = None,
    path: Path | None = None,
) -> Attempt:
    """An attempt shaped exactly as `Speculation.attempt` would have left it."""
    return Attempt(
        approach=Approach(name, "prompt"),
        path=path,
        diff="".join(f"+line {i}\n" for i in range(churn)),
        verification=Verification(ok=ok, output=output, skipped=skipped, counts=parse_test_output(output)),
        error=error,
        duration=duration,
        detail=detail,
        metrics=observe(detail),
    )


# -- reading a runner's own summary ------------------------------------------


def test_pytest_summaries_are_read_as_counts():
    counts = parse_test_output("=========== 137 passed, 2 skipped in 4.21s ============")
    assert (counts.passed, counts.failed, counts.skipped, counts.total) == (137, None, 2, 139)

    counts = parse_test_output("==== 3 failed, 12 passed, 1 error in 1.02s ====")
    assert counts.passed == 12
    assert counts.failed == 4, "an error is a failure, not a category of its own"
    assert counts.total == 16


def test_unittest_summaries_are_read_as_counts():
    counts = parse_test_output("Ran 9 tests in 0.031s\n\nFAILED (failures=2, skipped=1)\n")
    assert (counts.passed, counts.failed, counts.skipped, counts.total) == (6, 2, 1, 9)

    counts = parse_test_output("Ran 4 tests in 0.002s\n\nOK\n")
    assert (counts.passed, counts.failed, counts.total) == (4, 0, 4)


def test_go_and_cargo_summaries_are_read_as_counts():
    go = parse_test_output(
        "=== RUN   TestAlpha\n--- PASS: TestAlpha (0.00s)\n"
        "=== RUN   TestBeta\n--- FAIL: TestBeta (0.01s)\n"
        "=== RUN   TestGamma\n--- SKIP: TestGamma (0.00s)\nFAIL\n"
    )
    assert (go.passed, go.failed, go.skipped, go.total) == (1, 1, 1, 3)

    cargo = parse_test_output(
        "test result: ok. 12 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n"
        "test result: FAILED. 3 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out\n"
    )
    assert (cargo.passed, cargo.failed, cargo.skipped, cargo.total) == (15, 2, 1, 18), \
        "a workspace prints one tally per test binary and they must be summed"


def test_output_nobody_recognises_yields_absent_counts_not_zero():
    for text in ("", "make: *** [Makefile:4: check] Error 1", "Segmentation fault"):
        counts = parse_test_output(text)
        assert not counts.known
        assert counts.passed is None and counts.failed is None and counts.total is None, \
            "'we do not know' must never be stored as 'nothing failed'"
        assert counts.summary() == "no test counts"


# -- projecting a runner payload onto metrics --------------------------------


def test_a_real_run_result_yields_correct_tokens_steps_and_tools():
    result = RunResult(
        text="done",
        steps=7,
        usage=Usage(input=1200, output=340, cache_read=800, cache_write=100),
        stop_reason="stop",
        invocations=[
            Invocation(ToolCall("1", "read"), ToolResult(ok=True, duration=0.25)),
            Invocation(ToolCall("2", "edit"), ToolResult(ok=False, error="no such file", duration=0.1)),
            Invocation(ToolCall("3", "read"), ToolResult(ok=True, duration=0.05)),
        ],
    )
    metrics = observe(result)
    assert metrics.observed
    assert (metrics.tokens_in, metrics.tokens_out, metrics.tokens_cached) == (1200, 340, 900)
    assert metrics.tokens == 1540
    assert metrics.steps == 7
    assert metrics.tool_calls == 3
    assert metrics.tools == ("edit", "read"), "each tool named once, in a stable order"
    assert metrics.tool_failures == 1
    assert metrics.tool_time == pytest.approx(0.40)
    assert "7 steps" in metrics.summary() and "1 failed" in metrics.summary()


def test_a_bare_string_or_none_yields_empty_metrics_rather_than_raising():
    for detail in (None, "just some text", 42, object(), {"steps": 4}):
        metrics = observe(detail)
        assert metrics == BranchMetrics(), f"{detail!r} should observe as nothing at all"
        assert not metrics.observed
        assert metrics.summary() == "no agent metrics"


def test_a_generator_payload_is_not_consumed_looking_for_tool_calls():
    calls = (c for c in ("read", "edit"))

    class Payload:
        invocations = calls

    assert observe(Payload()).tool_calls == 0
    assert list(calls) == ["read", "edit"], "the scorer must not drain the caller's iterator"


# -- verification ------------------------------------------------------------


def test_ten_passing_tests_beat_one_passing_test():
    cards = score([
        make("thorough", output="10 passed in 0.40s"),
        make("thin", output="1 passed in 0.01s"),
    ])
    assert [c.name for c in cards] == ["thorough", "thin"]
    assert cards[0][VERIFICATION].score > cards[1][VERIFICATION].score
    assert "10 passed" in cards[0][VERIFICATION].reason


def test_an_unverified_branch_sits_between_proven_and_disproven():
    cards = score([
        make("proven", output="4 passed in 0.10s"),
        make("unverified", skipped=True),
        make("broken", ok=False, output="4 failed in 0.10s"),
    ])
    assert [c.name for c in cards] == ["proven", "unverified", "broken"]
    assert cards[1][VERIFICATION].reason == "nothing verified it"


def test_a_nearly_green_failure_scores_above_a_total_failure():
    cards = score([
        make("nearly", ok=False, output="9 passed, 1 failed in 0.10s"),
        make("hopeless", ok=False, output="0 passed, 10 failed in 0.10s"),
    ])
    assert [c.name for c in cards] == ["nearly", "hopeless"]
    assert cards[0][VERIFICATION].score > cards[1][VERIFICATION].score > 0.0 - 1e-9


def test_a_branch_that_never_finished_scores_nothing_for_verification():
    card = score([make("crashed", error="RuntimeError: could not isolate")])[0]
    assert card[VERIFICATION].score == 0.0
    assert "never finished" in card[VERIFICATION].reason


def test_a_green_run_that_collected_no_tests_is_not_treated_as_proof():
    cards = score([
        make("real", output="6 passed in 0.20s"),
        make("empty", output="no tests ran in 0.01s\n0 passed"),
    ])
    assert [c.name for c in cards] == ["real", "empty"]
    assert "ran no tests" in cards[1][VERIFICATION].reason


# -- regressions -------------------------------------------------------------


BASE_PASSING = frozenset({
    "tests/test_core.py::test_alpha",
    "tests/test_core.py::test_beta",
    "tests/test_core.py::test_zeta",
})


@pytest.fixture()
def green_base():
    return Baseline(counts=Counts(6, 0, 0, 6), passing=BASE_PASSING, ref="deadbeefcafe")


def test_a_regression_loses_to_a_bigger_clean_diff(green_base):
    """The decisive case: same tally, but one broke something that worked."""
    regressed = make(
        "small-regression", ok=False, churn=2,
        output="FAILED tests/test_core.py::test_zeta - AssertionError\n5 passed, 1 failed in 0.30s",
    )
    honest = make(
        "big-clean", ok=False, churn=40,
        output="FAILED tests/test_core.py::test_brand_new - AssertionError\n5 passed, 1 failed in 0.30s",
    )
    cards = score([regressed, honest], baseline=green_base)

    assert cards[0][VERIFICATION].score == pytest.approx(cards[1][VERIFICATION].score), \
        "the tallies are identical, so only the regression may separate them"
    assert [c.name for c in cards] == ["big-clean", "small-regression"]
    assert cards[0].attempt.churn > cards[1].attempt.churn, "the winner has the LARGER diff"
    assert cards[1][REGRESSIONS].score == 0.0
    assert "broke 1 test" in cards[1][REGRESSIONS].reason
    assert "test_zeta" in cards[1][REGRESSIONS].reason
    assert cards[0][REGRESSIONS].score == 1.0


def test_a_regression_outweighs_every_soft_criterion_at_once(green_base):
    """Near-disqualifying means no combination of niceties can rescue it."""
    regressed = make(
        "flawless-but-broken", ok=False, churn=1, duration=0.1,
        output="FAILED tests/test_core.py::test_beta\n5 passed, 1 failed in 0.10s",
        detail=RunResult(steps=2, usage=Usage(input=10, output=5), stop_reason="stop"),
    )
    rival = make(
        "sloppy-but-safe", ok=False, churn=300, duration=90.0,
        output="FAILED tests/test_core.py::test_fresh\n5 passed, 1 failed in 0.10s",
        detail=RunResult(steps=24, stop_reason="max_steps", usage=Usage(input=90000, output=4000)),
    )
    cards = score([regressed, rival], baseline=green_base,
                  reviewer=lambda attempts: ("flawless-but-broken", "cleanest patch by far"))
    assert [c.name for c in cards] == ["sloppy-but-safe", "flawless-but-broken"]
    assert cards[1][REVIEW].score == 1.0, "the reviewer did back the loser, and lost anyway"


def test_fewer_passes_than_the_base_is_a_regression_even_with_no_test_names():
    """Runners that print no names still betray a regression by arithmetic."""
    base = Baseline(counts=Counts(10, 0, 0, 10))
    cards = score([
        make("shrank", ok=False, churn=3, output="7 passed, 3 failed in 0.10s"),
        make("held", ok=False, churn=3, output="10 passed, 3 failed in 0.10s"),
    ], baseline=base)
    assert [c.name for c in cards] == ["held", "shrank"]
    assert "3 fewer tests pass" in cards[1][REGRESSIONS].reason
    assert cards[0][REGRESSIONS].score == 1.0


def test_deleting_a_test_is_not_reported_as_a_regression():
    base = Baseline(counts=Counts(10, 0, 0, 10))
    card = score([make("pruned", output="8 passed in 0.10s")], baseline=base)[0]
    assert card[REGRESSIONS].score == 1.0, "fewer tests with nothing failing is not a break"


def test_without_a_baseline_regressions_are_absent_rather_than_forgiven():
    card = score([make("solo", output="5 passed in 0.10s")])[0]
    assert card[REGRESSIONS].applies is False
    assert "no baseline" in card[REGRESSIONS].reason
    assert card[REGRESSIONS].points == 0.0
    assert card[REGRESSIONS] not in card.applied, "an absent criterion must not dilute the total"


def test_a_skipped_verification_cannot_be_compared_to_a_baseline(green_base):
    card = score([make("untested", skipped=True)], baseline=green_base)[0]
    assert card[REGRESSIONS].applies is False


# -- diff size ---------------------------------------------------------------


def _totals(*churns: int) -> dict[int, float]:
    cards = score([make(f"n{c}", output="5 passed in 0.10s", churn=c) for c in churns])
    return {card.attempt.churn: card.total for card in cards}


def test_three_versus_five_changed_lines_is_nearly_a_tie():
    totals = _totals(3, 5)
    gap = totals[3] - totals[5]
    assert 0.0 < gap < 0.02, f"3 vs 5 lines should be noise, not a decision (gap {gap:.4f})"


def test_ten_versus_five_hundred_changed_lines_is_decisive():
    small = _totals(3, 5)
    large = _totals(10, 500)
    gap = large[10] - large[500]
    assert gap > 0.1, f"10 vs 500 lines must be a verdict (gap {gap:.4f})"
    assert gap > 10 * (small[3] - small[5]), \
        "the churn penalty has to saturate: the same 2-line delta cannot matter equally at 3 and at 500"


def test_the_diff_score_saturates_rather_than_scaling():
    totals = _totals(500, 1000)
    assert totals[500] - totals[1000] < 0.05, "past a point, more lines barely matter any more"


def test_a_branch_that_changed_nothing_cannot_win():
    cards = score([
        make("did-nothing", output="5 passed in 0.10s", churn=0),
        make("did-something", output="5 passed in 0.10s", churn=120),
    ])
    assert [c.name for c in cards] == ["did-something", "did-nothing"]
    assert cards[1][DIFF].score == 0.0
    assert "nothing to adopt" in cards[1][DIFF].reason


# -- static analysis ---------------------------------------------------------


@pytest.fixture()
def fakelint(tmp_path, monkeypatch):
    """A real analyser on a real PATH: it prints one diagnostic per line of `findings`."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "fakelint"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "found = pathlib.Path('findings').read_text().split()\n"
        "for i, name in enumerate(found, 1):\n"
        "    print(f'app.py:{i}:1: X001 {name}')\n"
        "sys.exit(1 if found else 0)\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return Linter("fakelint", ("fakelint",), (("fakelint.toml", ""),))


def worktree(root: Path, name: str, findings: int, *, config: bool = True) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "findings").write_text(" ".join(f"issue{i}" for i in range(findings)))
    if config:
        (path / "fakelint.toml").write_text("[fakelint]\n")
    return path


def test_a_configured_analyser_is_discovered_and_its_diagnostics_counted(tmp_path, fakelint):
    clean = worktree(tmp_path / "wt", "clean", 0)
    messy = worktree(tmp_path / "wt", "messy", 9)
    cards = score(
        [make("messy", output="5 passed in 0.1s", path=messy),
         make("clean", output="5 passed in 0.1s", path=clean)],
        linters=(fakelint,),
    )
    assert [c.name for c in cards] == ["clean", "messy"]
    assert cards[0][STATIC].score == 1.0
    assert cards[0][STATIC].reason == "fakelint: 0 diagnostics"
    assert cards[1][STATIC].reason == "fakelint: 9 diagnostics"
    assert cards[1][STATIC].score < 0.5


def test_an_analyser_the_repo_never_configured_is_not_run(tmp_path, fakelint):
    bare = worktree(tmp_path / "wt", "bare", 9, config=False)
    card = score([make("bare", output="5 passed in 0.1s", path=bare)], linters=(fakelint,))[0]
    assert card[STATIC].applies is False
    assert card[STATIC].reason == "no configured analyser on PATH"


def test_an_analyser_that_is_not_installed_costs_a_branch_nothing(tmp_path):
    absent = Linter("nosuchlint", ("offset-no-such-linter",), (("anything", ""),))
    with_tool = score([make("a", output="5 passed in 0.1s")], linters=())[0]
    without = score([make("a", output="5 passed in 0.1s")], linters=(absent,))[0]
    assert without[STATIC].applies is False
    assert without.total == pytest.approx(with_tool.total), \
        "a machine without linters must not score every branch lower"


def test_an_analyser_that_crashes_is_reported_not_counted_as_clean(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin2"
    bin_dir.mkdir()
    script = bin_dir / "brokenlint"
    script.write_text("#!/usr/bin/env python3\nimport sys\nsys.stderr.write('config error\\n')\nsys.exit(2)\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    path = tmp_path / "wt3"
    path.mkdir()
    (path / "brokenlint.toml").write_text("x\n")
    linter = Linter("brokenlint", ("brokenlint",), (("brokenlint.toml", ""),))
    card = score([make("a", output="5 passed in 0.1s", path=path)], linters=(linter,))[0]
    assert card[STATIC].applies is False
    assert "exited 2" in card[STATIC].reason


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed here")
def test_ruff_is_discovered_when_the_repo_actually_configures_it(tmp_path):
    path = tmp_path / "py"
    path.mkdir()
    (path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    (path / "app.py").write_text("import os\nimport sys\n")
    ruff = next(l for l in DEFAULT_LINTERS if l.name == "ruff")
    assert ruff.configured(path) == "pyproject.toml"
    counts, notes = Analysis((ruff,)).run(path)
    assert counts.get("ruff", 0) >= 2, f"two unused imports should be two diagnostics ({notes})"

    (path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    assert ruff.configured(path) == "", "a pyproject without a ruff section is not a ruff config"


# -- agent health ------------------------------------------------------------


def test_hitting_the_step_limit_costs_a_branch_its_health_score():
    cards = score([
        make("flailed", output="5 passed in 0.1s",
             detail=RunResult(steps=24, stop_reason="max_steps", usage=Usage(input=90000, output=3000))),
        make("converged", output="5 passed in 0.1s",
             detail=RunResult(steps=3, stop_reason="stop", usage=Usage(input=900, output=200))),
    ])
    assert [c.name for c in cards] == ["converged", "flailed"]
    assert cards[1][HEALTH].score <= 0.2
    assert "max_steps" in cards[1][HEALTH].reason
    assert cards[0][HEALTH].score == 1.0


def test_failed_tool_calls_lower_health_in_proportion():
    def result(failures: int, total: int) -> RunResult:
        calls = [
            Invocation(ToolCall(str(i), "edit"), ToolResult(ok=i >= failures, error=None if i >= failures else "no"))
            for i in range(total)
        ]
        return RunResult(steps=total, stop_reason="stop", invocations=calls, usage=Usage(input=10, output=5))

    cards = score([
        make("fumbled", output="5 passed in 0.1s", detail=result(3, 4)),
        make("smooth", output="5 passed in 0.1s", detail=result(0, 4)),
    ])
    assert [c.name for c in cards] == ["smooth", "fumbled"]
    assert cards[1][HEALTH].score < cards[0][HEALTH].score
    assert "3 of 4 tool calls failed" in cards[1][HEALTH].reason


def test_a_runner_that_reported_nothing_is_not_penalised_for_it():
    card = score([make("silent", output="5 passed in 0.1s")])[0]
    assert card[HEALTH].applies is False
    assert card[HEALTH].reason == "the runner reported no agent metrics"


# -- the optional reviewer ---------------------------------------------------


def test_scoring_works_fully_with_no_reviewer_and_no_analysers():
    cards = score([
        make("a", output="8 passed in 0.2s", churn=12),
        make("b", ok=False, output="6 passed, 2 failed in 0.2s", churn=4),
    ], linters=())
    assert [c.name for c in cards] == ["a", "b"]
    assert all(c.total > 0.0 for c in cards)
    for card in cards:
        assert card[REVIEW].applies is False
        assert card[STATIC].applies is False
        assert card.applied, "verification and diff size always apply"


def test_a_reviewer_can_break_a_near_tie():
    attempts = [
        make("terse", output="5 passed in 0.1s", churn=3),
        make("clearer", output="5 passed in 0.1s", churn=5),
    ]
    assert [c.name for c in score(attempts)] == ["terse", "clearer"]
    cards = score(attempts, reviewer=lambda a: ("clearer", "reads better and handles the empty case"))
    assert [c.name for c in cards] == ["clearer", "terse"]
    assert "reads better" in cards[0][REVIEW].reason
    assert cards[1][REVIEW].reason == "the reviewer preferred clearer"


def test_a_reviewer_that_raises_or_talks_nonsense_is_ignored():
    attempts = [make("a", output="5 passed in 0.1s", churn=3), make("b", output="5 passed in 0.1s", churn=5)]
    plain = [c.name for c in score(attempts)]

    def explode(_attempts):
        raise RuntimeError("provider on fire")

    for reviewer in (explode, lambda a: None, lambda a: ("branch-that-never-ran", "trust me")):
        cards = score(attempts, reviewer=reviewer)
        assert [c.name for c in cards] == plain
        assert cards[0][REVIEW].applies is False


def test_the_reviewer_is_asked_once_for_the_whole_set():
    calls: list[int] = []

    def reviewer(attempts):
        calls.append(len(attempts))
        return ("a", "because")

    score([make("a"), make("b"), make("c")], reviewer=reviewer)
    assert calls == [3], "one judgement over all branches, not one call per branch"


class _StubEnsemble:
    """Stands in for `Ensemble` only to capture the request it would have sent."""

    def __init__(self, text: str | None = None, boom: bool = False) -> None:
        self.text, self.boom, self.requests = text, boom, []

    def council(self, request, judge, seats, *, criterion):
        from offset.core.multimodel import Opinion, Seat, Verdict
        from offset.providers.base import Turn

        self.requests.append(request)
        if self.boom:
            raise RuntimeError("every seat failed")
        if self.text is None:
            return Verdict(None, "no seat produced an answer")
        return Verdict(Opinion(Seat(model="judge"), Turn(text=self.text)), "chose [0]")


def test_the_council_reviewer_reads_a_branch_name_out_of_the_ruling():
    from offset.core.multimodel import Seat

    ensemble = _StubEnsemble("rewrite\nIt fixes the cause rather than the symptom.")
    review = council_reviewer(ensemble, Seat(model="judge", role="referee"))
    attempts = [make("minimal", churn=3), make("rewrite", churn=30)]

    picked = review(attempts)
    assert picked is not None
    name, reason = picked
    assert name == "rewrite"
    assert "fixes the cause" in reason
    sent = ensemble.requests[0].messages[0].text
    assert "### minimal" in sent and "### rewrite" in sent, "the judge must see both diffs"


def test_the_council_reviewer_degrades_to_silence():
    from offset.core.multimodel import Seat

    judge = Seat(model="judge")
    attempts = [make("minimal", churn=3), make("rewrite", churn=30)]
    assert council_reviewer(_StubEnsemble(boom=True), judge)(attempts) is None
    assert council_reviewer(_StubEnsemble(None), judge)(attempts) is None
    assert council_reviewer(_StubEnsemble("neither of these is any good"), judge)(attempts) is None

    quiet = _StubEnsemble("minimal wins")
    assert council_reviewer(quiet, judge)([make("only", churn=3)]) is None
    assert quiet.requests == [], "one candidate is not a comparison; do not pay a model to say so"


# -- the scorecard as an explanation ----------------------------------------


def test_the_winners_scorecard_explains_itself_in_readable_prose(green_base):
    cards = score([
        make("minimal", churn=6, duration=12.4,
             output="tests/test_core.py::test_alpha PASSED\n8 passed in 0.60s",
             detail=RunResult(steps=5, stop_reason="stop", usage=Usage(input=4000, output=210),
                              invocations=[Invocation(ToolCall("1", "edit"), ToolResult(ok=True))])),
        make("rewrite", churn=210, output="8 passed in 0.60s"),
    ], baseline=green_base)

    lines = cards[0].lines()
    assert lines[0] == "minimal scored 0.95 of 1.00", lines[0]
    body = "\n".join(lines)
    assert "8 passed of 8" in body
    assert "nothing that passed at the base fails here" in body
    assert "6 changed lines" in body
    assert "5 steps" in body
    assert "no reviewer was asked" in body, "an absent criterion is still reported"
    assert "weight applied" in body
    assert cards[0].summary().startswith("minimal 0.95")
    assert cards[0].strongest.name == REGRESSIONS


def test_asking_for_a_criterion_that_does_not_exist_names_the_ones_that_do():
    card = score([make("a")])[0]
    with pytest.raises(KeyError) as caught:
        card["vibes"]
    assert "no criterion named vibes" in str(caught.value)
    assert VERIFICATION in str(caught.value)


# -- determinism -------------------------------------------------------------


def test_the_ranking_does_not_depend_on_the_input_order():
    attempts = [
        make("alpha", output="5 passed in 0.1s", churn=8, duration=2.0),
        make("bravo", output="5 passed in 0.1s", churn=8, duration=2.0),
        make("charlie", ok=False, output="5 failed in 0.1s", churn=1),
        make("delta", output="9 passed in 0.1s", churn=8),
    ]
    forwards = [c.name for c in score(attempts)]
    backwards = [c.name for c in score(list(reversed(attempts)))]
    assert forwards == backwards
    assert forwards[0] == "delta"
    assert forwards[1:3] == ["alpha", "bravo"], "identical branches fall back to the name, stably"


def test_scoring_an_empty_set_is_empty_not_an_error():
    assert score([]) == []


# -- the baseline, measured for real ----------------------------------------


RUNNER = """\
import app, sys
ok = app.VALUE == 1
print("tests/test_app.py::test_value " + ("PASSED" if ok else "FAILED"))
if not ok:
    print("FAILED tests/test_app.py::test_value - assert 2 == 1")
print(("1 passed" if ok else "1 failed") + " in 0.01s")
sys.exit(0 if ok else 1)
"""


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    git("init", "-q", cwd=root)
    git("config", "user.email", "t@t.t", cwd=root)
    git("config", "user.name", "t", cwd=root)
    (root / "app.py").write_text("VALUE = 1\n")
    (root / "runner.py").write_text(RUNNER)
    git("add", "-A", cwd=root)
    git("commit", "-qm", "init", cwd=root)
    return root


def test_the_baseline_runs_the_verification_against_the_untouched_tree(repo):
    spec = Speculation(repo, verify_command="python3 runner.py")
    base = measure_baseline(spec)

    assert base.known
    assert base.error is None
    assert base.counts.passed == 1 and base.counts.total == 1
    assert base.passing == {"tests/test_app.py::test_value"}
    assert base.failing == frozenset()
    assert base.ref, "a git baseline should name the commit it measured"
    assert "1 passed" in base.summary()


def test_the_baseline_includes_uncommitted_work(repo):
    (repo / "app.py").write_text("VALUE = 2\n")  # dirty, never committed
    base = measure_baseline(Speculation(repo, verify_command="python3 runner.py"))
    assert base.counts.failed == 1, "the baseline must be the tree as it is, not as it was committed"
    assert base.failing == {"tests/test_app.py::test_value"}
    assert base.passing == frozenset()


def test_the_baseline_workspace_is_always_removed_even_under_keep(repo):
    spec = Speculation(repo, verify_command="python3 runner.py", keep=True)
    measure_baseline(spec)
    assert not (repo / ".offset" / "branches" / "baseline-check").exists()
    assert "baseline-check" not in git("worktree", "list", cwd=repo).stdout


def test_without_a_verification_command_there_is_no_baseline(repo):
    base = measure_baseline(Speculation(repo))
    assert not base.known
    assert base.error == "no verification command configured"
    assert base.summary() == "no baseline: no verification command configured"


def test_an_unbranchable_workspace_yields_a_named_baseline_failure(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    git("init", "-q", cwd=empty)  # no commits: nothing to branch from
    base = measure_baseline(Speculation(empty, verify_command="true"))
    assert not base.known
    assert base.error and "could not create worktree" in base.error


def test_a_measured_baseline_makes_a_real_regression_visible(repo):
    """End to end: measure the base, then score a branch that broke it."""
    spec = Speculation(repo, verify_command="python3 runner.py")
    base = measure_baseline(spec)
    broke = spec.attempt(Approach("break-it", "p"), lambda a, path: (path / "app.py").write_text("VALUE = 9\n"))
    spec.cleanup([broke])

    card = score([broke], baseline=base)[0]
    assert broke.state == "fail"
    assert card[REGRESSIONS].score == 0.0
    assert "test_value" in card[REGRESSIONS].reason


# -- weights -----------------------------------------------------------------


def test_weights_are_a_statement_of_priority_not_a_scale():
    attempts = [make("a", output="9 passed in 0.1s", churn=4), make("b", output="9 passed in 0.1s", churn=90)]
    default = score(attempts)
    doubled = score(attempts, weights=Weights(verification=8.0, regressions=10.0, diff=2.0,
                                              static=2.0, health=1.0, review=3.0))
    assert [c.total for c in default] == pytest.approx([c.total for c in doubled]), \
        "the total is normalised, so scaling every weight changes nothing"

    churn_blind = score(attempts, weights=Weights(diff=0.0))
    assert churn_blind[0].total == pytest.approx(churn_blind[1].total)
