"""Performance measurement, and refusing to invent a result.

The contract worth defending hardest is negative: when two runs cannot be
distinguished, say so. A ranking built on differences that are actually noise
moves confidently in random directions, which is worse than one with no
benchmark at all - it is wrong *and* trusted.

Most tests here inject synthetic `Sample` lists rather than sleeping. Sleeping
to prove a statistical property is slow, flaky on a loaded Pi, and tests the
scheduler rather than the code. Two tests do run real subprocesses, because the
failure modes they cover - a crash, a timeout - only exist there.
"""

from __future__ import annotations

import json

import pytest

from offset.core.bench import (
    FASTER,
    NOISE,
    SAME,
    SLOWER,
    Benchmark,

    Result,
    Sample,
    bench_criterion,
    baseline_path,
    compare,
    configured_command,
    load_baseline,
    save_baseline,
)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    return tmp_path


def result(times, command="cmd", exit_code=0) -> Result:
    return Result(command=command,
                  samples=[Sample(seconds=t, exit_code=exit_code) for t in times])


# -- distributions -----------------------------------------------------------------


def test_a_result_reports_a_distribution_not_a_number():
    got = result([0.10, 0.12, 0.14, 0.20])
    assert got.min == pytest.approx(0.10)
    assert got.median == pytest.approx(0.13)
    assert got.mean == pytest.approx(0.14)
    assert got.stdev > 0
    assert got.count == 4


def test_stdev_of_a_single_sample_is_zero_not_an_error():
    assert result([0.1]).stdev == 0.0


def test_the_warmup_runs_are_not_in_the_sample():
    """The first run pays for cold caches and imports; letting it into the
    sample is how people conclude a change is slow when it is not."""
    got = Benchmark("python3 -c pass", runs=3, warmup=2).run()
    assert len(got.samples) == 3


def test_peak_memory_is_reported():
    got = Result(command="c", samples=[Sample(seconds=0.1, rss_kb=2048),
                                       Sample(seconds=0.1, rss_kb=4096)])
    assert got.peak_rss_kb == 4096


# -- failures are not fast results --------------------------------------------------


def test_a_failing_command_is_not_a_clean_result():
    """A crash returns almost instantly, which naive timing scores as a
    spectacular improvement."""
    assert result([0.001], exit_code=1).ok is False


def test_a_failing_result_says_the_command_failed():
    assert "non-zero" in result([0.001], exit_code=1).summary()


def test_one_failure_invalidates_the_whole_result():
    """Averaging a crash in with real timings drags the mean down and makes a
    broken command look fast."""
    mixed = Result(command="c", samples=[Sample(seconds=0.5),
                                         Sample(seconds=0.001, exit_code=1)])
    assert mixed.ok is False


def test_a_real_crash_is_detected():
    got = Benchmark("python3 -c 'raise SystemExit(3)'", runs=2, warmup=1).run()
    assert got.ok is False
    assert got.failures >= 1


def test_a_missing_binary_is_reported_not_raised():
    got = Benchmark("this-binary-does-not-exist-anywhere", runs=2, warmup=1).run()
    assert got.ok is False


def test_a_timeout_is_reported_and_kills_the_child():
    """A hung benchmark needs different advice from a broken command."""
    got = Benchmark("python3 -c 'import time; time.sleep(9)'",
                    runs=1, warmup=0, timeout=0.4).run()
    assert got.ok is False
    assert got.timeouts == 1
    assert "timeout" in got.summary()


def test_a_timeout_quotes_the_limit_that_was_in_force():
    got = Benchmark("python3 -c 'import time; time.sleep(9)'",
                    runs=1, warmup=0, timeout=0.4).run()
    assert "0.4s" in got.summary(), got.summary()


def test_a_warmup_failure_stops_before_the_real_runs():
    """Twelve more runs of the same failure tells nobody anything new."""
    got = Benchmark("python3 -c 'raise SystemExit(2)'", runs=12, warmup=1).run()
    assert len(got.samples) == 1


def test_an_empty_command_is_an_error_not_a_zero_time():
    assert Benchmark("   ").run().ok is False


def test_an_unparseable_command_is_an_error():
    assert Benchmark('python3 -c "unclosed').run().error


# -- the honesty requirement -----------------------------------------------------------


def test_identical_distributions_are_indistinguishable():
    same = result([0.10, 0.11, 0.12])
    assert compare(same, result([0.10, 0.11, 0.12])).verdict == SAME


def test_overlapping_samples_refuse_to_name_a_winner():
    """The central test.  Medians differ by 4%, but the slowest of one run is
    slower than the fastest of the other - so the medians differ by luck."""
    before = result([0.100, 0.104, 0.108, 0.112, 0.120])
    after = result([0.104, 0.108, 0.112, 0.116, 0.124])
    got = compare(before, after)
    assert got.ratio > 1.0, "the medians really do differ"
    assert got.meaningful is False
    assert got.verdict == SAME


def test_clearly_separated_samples_do_name_a_winner():
    got = compare(result([0.100, 0.101, 0.102]), result([0.400, 0.402, 0.404]))
    assert got.meaningful is True
    assert got.verdict == SLOWER


def test_a_clear_improvement_is_named_as_faster():
    got = compare(result([0.400, 0.402, 0.404]), result([0.100, 0.101, 0.102]))
    assert got.verdict == FASTER


def test_a_difference_below_the_noise_floor_is_not_a_result():
    """Non-overlapping but tiny: 1% on a thermally throttled Pi is drift."""
    before = result([0.10000, 0.10001])
    after = result([0.10050, 0.10051])
    got = compare(before, after)
    assert abs(1.0 - got.ratio) < NOISE
    assert got.meaningful is False


def test_runs_too_short_to_time_are_indistinguishable():
    """Below the floor the timer is measuring process spawn, not the program."""
    got = compare(result([0.0001, 0.0001]), result([0.0004, 0.0004]))
    assert got.verdict == SAME
    assert "too short" in " ".join(got.lines())


def test_a_comparison_against_a_crash_is_not_usable():
    got = compare(result([0.1, 0.1]), result([0.001], exit_code=1))
    assert got.usable is False
    assert "did not produce a clean run" in " ".join(got.lines())


def test_the_verdict_explains_why_it_could_not_tell():
    before = result([0.100, 0.104, 0.108, 0.112, 0.120])
    after = result([0.104, 0.108, 0.112, 0.116, 0.124])
    assert "overlap" in " ".join(compare(before, after).lines())


def test_memory_growth_is_flagged_even_when_faster():
    """A speedup bought by buffering everything into memory is not obviously
    a win."""
    before = Result(command="c", samples=[Sample(seconds=0.40, rss_kb=10_000)])
    after = Result(command="c", samples=[Sample(seconds=0.10, rss_kb=90_000)])
    assert any("memory grew" in line for line in compare(before, after).lines())


def test_a_real_slowdown_is_measured_end_to_end():
    """One test that actually runs both sides, since the synthetic ones cannot
    prove the timer is wired to the subprocess at all."""
    quick = Benchmark("python3 -c pass", runs=5, warmup=1).run()
    slow = Benchmark("python3 -c 'import time; time.sleep(0.05)'",
                     runs=5, warmup=1).run()
    assert quick.ok and slow.ok
    assert compare(quick, slow).verdict == SLOWER


# -- scoring --------------------------------------------------------------------------


def test_no_benchmark_is_excluded_from_scoring():
    """An unbenchmarked branch must not be penalised - that would make the
    winner depend on whether a benchmark was configured."""
    assert bench_criterion(None, 1.0).applies is False


def test_an_unusable_benchmark_is_excluded():
    got = compare(result([0.1]), result([0.001], exit_code=1))
    assert bench_criterion(got, 1.0).applies is False


def test_an_indistinguishable_result_is_excluded_rather_than_zeroed():
    got = compare(result([0.10, 0.11]), result([0.10, 0.11]))
    criterion = bench_criterion(got, 1.0)
    assert criterion.applies is False
    assert criterion.score == 0.5


def test_a_faster_branch_scores_above_neutral():
    got = compare(result([0.400, 0.402]), result([0.100, 0.101]))
    assert bench_criterion(got, 1.0).score > 0.5


def test_a_slower_branch_scores_below_neutral():
    got = compare(result([0.100, 0.101]), result([0.400, 0.402]))
    assert bench_criterion(got, 1.0).score < 0.5


def test_a_huge_speedup_does_not_score_unboundedly():
    """Ten times faster is not five times a better reason to pick a branch
    over a correct one."""
    got = compare(result([10.0, 10.1]), result([0.10, 0.11]))
    assert bench_criterion(got, 1.0).score <= 1.0


def test_the_criterion_reason_quotes_the_measurement():
    got = compare(result([0.100, 0.101]), result([0.400, 0.402]))
    assert "%" in bench_criterion(got, 1.0).reason


# -- baselines -------------------------------------------------------------------------


def test_a_baseline_round_trips(isolated):
    saved = result([0.1, 0.2], command="pytest -q")
    assert save_baseline(saved, isolated) is not None
    back = load_baseline("pytest -q", isolated)
    assert back is not None
    assert back.command == "pytest -q"
    assert back.count == 2


def test_two_commands_do_not_share_a_baseline(isolated):
    assert baseline_path("a", isolated) != baseline_path("b", isolated)


def test_a_command_with_slashes_cannot_escape_the_directory(isolated):
    """The filename is a hash, so a command is never a path."""
    path = baseline_path("../../etc/passwd", isolated)
    assert path.parent == isolated / "bench"
    assert path.name.endswith(".json")


def test_a_missing_baseline_is_none_not_an_error(isolated):
    assert load_baseline("never-run", isolated) is None


def test_a_corrupt_baseline_is_none_not_a_crash(isolated):
    path = baseline_path("cmd", isolated)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert load_baseline("cmd", isolated) is None


def test_the_timeout_survives_a_round_trip(isolated):
    saved = Result(command="c", samples=[Sample(seconds=0.1)], timeout_used=7.5)
    save_baseline(saved, isolated)
    assert load_baseline("c", isolated).timeout_used == 7.5


# -- repository config -------------------------------------------------------------------


def test_a_configured_command_is_discovered(tmp_path):
    (tmp_path / "bench.json").write_text(json.dumps({"command": "pytest -q"}))
    assert configured_command(tmp_path) == "pytest -q"


@pytest.mark.parametrize("key", ["command", "bench", "benchmark"])
def test_either_obvious_key_works(tmp_path, key):
    """Failing on the wrong guess is a pointless obstacle."""
    (tmp_path / "bench.json").write_text(json.dumps({key: "make test"}))
    assert configured_command(tmp_path) == "make test"


def test_a_bare_string_config_works(tmp_path):
    (tmp_path / "bench.json").write_text(json.dumps("go test ./..."))
    assert configured_command(tmp_path) == "go test ./..."


def test_no_config_is_an_empty_string(tmp_path):
    assert configured_command(tmp_path) == ""


def test_a_broken_config_is_not_an_error(tmp_path):
    (tmp_path / "bench.json").write_text("{{{")
    assert configured_command(tmp_path) == ""


# -- the command ---------------------------------------------------------------------------


class State:
    def __init__(self, workspace):
        self.workspace = workspace


def test_the_command_explains_itself_with_nothing_to_run(tmp_path):
    from offset.core.bench import _bench_command

    out = _bench_command(State(tmp_path), [])
    assert any("nothing to benchmark" in line for line in out.lines)


def test_the_command_says_there_is_no_baseline_yet(tmp_path):
    from offset.core.bench import _bench_command

    out = _bench_command(State(tmp_path), ["python3", "-c", "pass"])
    assert any("no baseline" in line for line in out.lines)


def test_saving_then_comparing_produces_a_verdict(tmp_path, isolated):
    from offset.core.bench import _bench_command

    saved = _bench_command(State(tmp_path), ["--save", "python3", "-c", "pass"])
    assert any("baseline" in line for line in saved.lines)
    again = _bench_command(State(tmp_path), ["python3", "-c", "pass"])
    assert any("verdict" in line for line in again.lines)


def test_the_command_is_registered_lazily():
    import offset.core.bench as module

    first, second = module.COMMANDS, module.COMMANDS
    assert first is second
    assert [c.name for c in first] == ["bench"]


def test_no_child_process_survives(isolated):
    """A benchmark that leaks a child wedges the next run of the suite."""
    import subprocess

    Benchmark("python3 -c 'import time; time.sleep(9)'",
              runs=1, warmup=0, timeout=0.3).run()
    found = subprocess.run(["pgrep", "-f", "time.sleep(9)"],
                           capture_output=True, text=True, check=False)
    assert found.returncode != 0, f"leaked: {found.stdout}"
