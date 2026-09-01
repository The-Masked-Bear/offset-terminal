"""Filesystem snapshots for workspace isolation.

Two things have to be true or this is worse than useless. A snapshot must be
genuinely independent - editing one side must not touch the other, or parallel
work silently corrupts itself. And releasing one must never, under any
circumstance, delete the original workspace.

The copy-on-write backends cannot be exercised on this machine's filesystem, so
they are driven through an injected runner rather than skipped: the thing worth
testing about them is the command built and the fallthrough when they fail, and
both are testable without btrfs.
"""

from __future__ import annotations

import os

import pytest

from offset.core.fsnapshot import Completed, available_backends, clear_probe_cache, detect, snapshot


@pytest.fixture(autouse=True)
def fresh_probes():
    """The probe result is cached per device; tests must not inherit it."""
    clear_probe_cache()
    yield
    clear_probe_cache()


@pytest.fixture()
def workspace(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.py").write_text("original\n")
    (root / "pkg").mkdir()
    (root / "pkg" / "b.py").write_text("also original\n")
    (root / ".gitignore").write_text("ignored/\n*.log\n")
    (root / "ignored").mkdir()
    (root / "ignored" / "junk.txt").write_text("junk\n")
    (root / "noisy.log").write_text("noise\n")
    return root


def runner_that(**outcomes):
    """A fake `Runner`: maps the first argv word to a return code."""
    seen: list[list[str]] = []

    def run(argv, **kw):
        seen.append(list(argv))
        code = outcomes.get(argv[0], 127)
        return Completed(code, "" if code == 0 else f"{argv[0]}: refused")

    run.seen = seen  # type: ignore[attr-defined]
    return run


# -- probing --------------------------------------------------------------------


def test_the_probe_answers_something_on_this_machine(workspace):
    found = detect(workspace)
    assert found is not None
    assert found.name


def test_the_probe_is_cached_per_device(workspace):
    """Probing runs real commands; doing it per snapshot would be a syscall
    storm on a machine that takes many."""
    run = runner_that()
    detect(workspace, runner=run)
    first = len(run.seen)
    detect(workspace, runner=run)
    assert len(run.seen) == first, "the probe ran again for the same device"


def test_clearing_the_cache_makes_it_probe_again(workspace):
    run = runner_that()
    detect(workspace, runner=run)
    first = len(run.seen)
    clear_probe_cache()
    detect(workspace, runner=run)
    assert len(run.seen) > first


def test_a_backend_that_errors_falls_through_to_the_next(workspace):
    """A machine with btrfs tools installed but no permission must not fail;
    it must quietly use something that works."""
    run = runner_that()  # every command returns 127
    found = detect(workspace, runner=run)
    assert found is not None
    assert found.instant is False, "no CoW backend worked, so this cannot be instant"


def test_available_backends_lists_what_was_tried(workspace):
    names = [b.name for b in available_backends(workspace, runner=runner_that())]
    assert names, "nothing was even attempted"
    assert names[-1] in ("copy", "fallback"), "the plain copy must always be last"


# -- snapshots ------------------------------------------------------------------


def test_a_snapshot_is_usable_and_reports_its_backend(workspace):
    with snapshot(workspace) as snap:
        assert snap.path.exists()
        assert (snap.path / "a.py").read_text() == "original\n"
        assert snap.backend.name


def test_the_snapshot_is_genuinely_independent(workspace):
    """The property the whole feature is for. If edits bleed across, parallel
    work corrupts itself and the corruption looks like a model mistake."""
    with snapshot(workspace) as snap:
        (snap.path / "a.py").write_text("changed in the snapshot\n")
        assert (workspace / "a.py").read_text() == "original\n"

        (workspace / "pkg" / "b.py").write_text("changed in the original\n")
        assert (snap.path / "pkg" / "b.py").read_text() == "also original\n"


def test_a_fallback_copy_admits_it_is_not_free(workspace):
    """Reporting a plain recursive copy as instant would let a caller take a
    hundred of them and wonder where the disk went."""
    snap = snapshot(workspace, runner=runner_that())
    try:
        assert snap.instant is False
        assert snap.bytes_copied > 0
    finally:
        snap.release()


def test_the_copy_fallback_honours_gitignore(workspace):
    snap = snapshot(workspace, runner=runner_that())
    try:
        assert (snap.path / "a.py").exists()
        assert not (snap.path / "ignored").exists(), "copied an ignored directory"
        assert not (snap.path / "noisy.log").exists(), "copied an ignored file"
    finally:
        snap.release()


# -- release --------------------------------------------------------------------


def test_release_removes_the_snapshot(workspace):
    snap = snapshot(workspace)
    where = snap.path
    snap.release()
    assert not where.exists()


def test_release_is_idempotent(workspace):
    snap = snapshot(workspace)
    snap.release()
    snap.release()  # must not raise
    assert snap.released


def test_release_never_touches_the_original(workspace):
    """The catastrophic failure. Guarded explicitly rather than trusted."""
    snap = snapshot(workspace)
    snap.release()
    assert workspace.exists()
    assert (workspace / "a.py").read_text() == "original\n"
    assert (workspace / "pkg" / "b.py").exists()


def test_a_snapshot_pointed_at_the_original_refuses_to_release_it(workspace):
    """Belt and braces. If a backend ever hands back the workspace itself,
    release must decline rather than delete the user's code."""
    from offset.core.fsnapshot import SnapshotError

    snap = snapshot(workspace)
    snap.path = workspace          # the mistake this guard exists for
    # Loudly, not quietly: a release that silently did nothing would leave the
    # caller believing it had cleaned up.
    with pytest.raises(SnapshotError, match="workspace itself"):
        snap.release()
    assert workspace.exists(), "release deleted the workspace it was pointed at"
    assert (workspace / "a.py").exists()


def test_the_context_manager_releases_on_the_way_out(workspace):
    with snapshot(workspace) as snap:
        where = snap.path
        assert where.exists()
    assert not where.exists()


def test_an_exception_inside_the_block_still_releases(workspace):
    where = None
    with pytest.raises(RuntimeError), snapshot(workspace) as snap:
        where = snap.path
        raise RuntimeError("boom")
    assert where is not None and not where.exists()


def test_root_is_never_required(workspace):
    """Every backend that needs privileges must fall through, not fail."""
    assert os.geteuid() != 0, "this test is meaningless as root"
    with snapshot(workspace) as snap:
        assert snap.path.exists()
