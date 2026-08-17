"""Invariants of the snapshot store.

The promise is narrow and absolute: after a rewind, either a file holds exactly
the bytes it held at the chosen point, or the user is told it does not.  These
tests defend that, the content-addressing that keeps the store small, and the
fact that the history survives a reload because it lives in the session log.
"""

from __future__ import annotations

import os

import pytest

from offset.core.session import Session
from offset.core.snapshots import (
    SNAPSHOT,
    Record,
    Store,
    capture,
    capture_all,
    history,
    records,
    restore,
    target_paths,
)


@pytest.fixture()
def ws(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


@pytest.fixture()
def s(tmp_path):
    sess = Session.create(tmp_path / "sessions")
    yield sess
    sess.close()


def edit(session, workspace, rel, content, *, tool="write", call=None, cap=None):
    """What a writing tool does: snapshot first, then change the file."""
    rec = capture(session, rel, tool=tool, call=call, root=workspace, cap=cap)
    target = workspace / rel
    if content is None:
        target.unlink()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return rec


# -- content addressing -----------------------------------------------------


def test_two_identical_writes_store_one_blob(s, ws):
    (ws / "a.py").write_text("same\n", encoding="utf-8")
    (ws / "b.py").write_text("same\n", encoding="utf-8")
    edit(s, ws, "a.py", "changed a\n")
    edit(s, ws, "b.py", "changed b\n")
    store = Store(ws)
    assert len(store.blobs()) == 1, "identical prior content must be stored once"
    assert len(records(s)) == 2, "both captures must still be indexed"


def test_recapturing_unchanged_content_does_not_grow_the_store(s, ws):
    (ws / "a.py").write_text("one\n", encoding="utf-8")
    edit(s, ws, "a.py", "one\n")  # a write that changes nothing
    edit(s, ws, "a.py", "two\n")
    assert Store(ws).blobs() == [Store.digest(b"one\n")]


def test_a_snapshot_of_a_missing_file_holds_no_blob(s, ws):
    rec = capture(s, "new.py", tool="write", root=ws)
    assert rec.absent and rec.hash is None
    assert Store(ws).blobs() == [], "nothing existed, so nothing should be stored"


# -- restore ----------------------------------------------------------------


def test_restore_returns_a_file_to_its_earlier_content(s, ws):
    (ws / "a.py").write_text("original\n", encoding="utf-8")
    mark = s.say("user", "change it")
    edit(s, ws, "a.py", "mangled\n")

    got = restore(s, mark.id, root=ws)
    assert got.ok, got.failed
    assert got.restored == ["a.py"]
    assert (ws / "a.py").read_text(encoding="utf-8") == "original\n"


def test_restoring_across_several_edits_lands_on_the_chosen_version(s, ws):
    (ws / "a.py").write_text("v1\n", encoding="utf-8")
    edit(s, ws, "a.py", "v2\n")
    mark = s.say("user", "from here")
    edit(s, ws, "a.py", "v3\n")
    edit(s, ws, "a.py", "v4\n")
    edit(s, ws, "a.py", "v5\n")

    got = restore(s, mark.id, root=ws)
    assert got.restored == ["a.py"]
    assert (ws / "a.py").read_text(encoding="utf-8") == "v2\n", "must be the state at the mark, not the oldest"


def test_restore_only_undoes_writes_after_the_chosen_point(s, ws):
    (ws / "a.py").write_text("a1\n", encoding="utf-8")
    (ws / "b.py").write_text("b1\n", encoding="utf-8")
    edit(s, ws, "a.py", "a2\n")
    mark = s.say("user", "from here")
    edit(s, ws, "b.py", "b2\n")

    got = restore(s, mark.id, root=ws)
    assert got.restored == ["b.py"]
    assert (ws / "a.py").read_text(encoding="utf-8") == "a2\n", "an earlier edit must be kept"


def test_a_file_created_after_the_mark_is_removed_on_rewind(s, ws):
    mark = s.say("user", "make a file")
    edit(s, ws, "pkg/new.py", "fresh\n")
    assert (ws / "pkg" / "new.py").exists()

    got = restore(s, mark.id, root=ws)
    assert got.removed == ["pkg/new.py"]
    assert not (ws / "pkg" / "new.py").exists()


def test_a_deleted_file_is_recreated_by_a_rewind(s, ws):
    (ws / "gone.py").write_text("still here\n", encoding="utf-8")
    mark = s.say("user", "delete it")
    edit(s, ws, "gone.py", None, tool="bash")
    assert not (ws / "gone.py").exists()

    got = restore(s, mark.id, root=ws)
    assert got.restored == ["gone.py"]
    assert (ws / "gone.py").read_text(encoding="utf-8") == "still here\n"


def test_restore_is_idempotent(s, ws):
    (ws / "a.py").write_text("original\n", encoding="utf-8")
    mark = s.say("user", "go")
    edit(s, ws, "a.py", "mangled\n")
    edit(s, ws, "made.py", "new\n")

    first = restore(s, mark.id, root=ws)
    assert first.changed == 2
    second = restore(s, mark.id, root=ws)
    assert second.ok
    assert second.changed == 0, "a second rewind must have nothing left to do"
    assert sorted(second.unchanged) == ["a.py", "made.py"]
    assert (ws / "a.py").read_text(encoding="utf-8") == "original\n"


def test_the_executable_bit_survives_a_rewind(s, ws):
    script = ws / "run.sh"
    script.write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    script.chmod(0o755)
    mark = s.say("user", "go")
    edit(s, ws, "run.sh", "#!/bin/sh\nfalse\n")
    script.chmod(0o644)

    restore(s, mark.id, root=ws)
    assert os.access(script, os.X_OK), "restoring content but not mode is not a restore"


def test_restore_of_an_unknown_entry_is_an_error_value(s, ws):
    (ws / "a.py").write_text("original\n", encoding="utf-8")
    edit(s, ws, "a.py", "mangled\n")

    got = restore(s, "NOPE", root=ws)
    assert got.error and "NOPE" in got.error
    assert not got.ok and got.changed == 0
    assert (ws / "a.py").read_text(encoding="utf-8") == "mangled\n", "a bad target must touch nothing"


def test_a_lost_blob_is_reported_rather_than_silently_skipped(s, ws):
    (ws / "a.py").write_text("original\n", encoding="utf-8")
    mark = s.say("user", "go")
    rec = edit(s, ws, "a.py", "mangled\n")
    Store(ws).path_for(rec.hash).unlink()

    got = restore(s, mark.id, root=ws)
    assert not got.ok
    assert got.failed and got.failed[0][0] == "a.py" and "missing" in got.failed[0][1]


# -- what cannot be snapshotted --------------------------------------------


def test_an_oversized_file_is_reported_not_ignored(s, ws):
    big = ws / "big.csv"
    big.write_text("x" * 5_000, encoding="utf-8")
    mark = s.say("user", "go")
    # The default cap is deliberately generous, so the test states its own.
    rec = edit(s, ws, "big.csv", "clobbered\n", tool="write", cap=1_000)

    assert rec.skipped and "too large" in rec.skipped
    assert rec.size == 5_000, "the real size must be recorded so the user can judge"
    assert Store(ws).blobs() == []

    got = restore(s, mark.id, root=ws)
    assert not got.ok, "a rewind that cannot restore a file must not claim success"
    assert got.failed == [("big.csv", rec.skipped)]
    assert (ws / "big.csv").read_text(encoding="utf-8") == "clobbered\n"


def test_the_size_cap_is_configurable_per_call(s, ws):
    (ws / "a.py").write_text("x" * 100, encoding="utf-8")
    rec = capture(s, "a.py", tool="write", root=ws, cap=10)
    assert rec.skipped and "too large" in rec.skipped
    assert capture(s, "a.py", tool="write", root=ws, cap=1000).stored


def test_a_binary_file_is_reported_not_ignored(s, ws):
    (ws / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")
    mark = s.say("user", "go")
    rec = edit(s, ws, "logo.png", "clobbered\n")

    assert rec.skipped and "binary" in rec.skipped
    assert Store(ws).blobs() == []
    got = restore(s, mark.id, root=ws)
    assert got.failed == [("logo.png", rec.skipped)]


def test_the_store_itself_is_never_snapshotted(s, ws):
    blob = Store(ws).dir / "ab" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_text("blob\n", encoding="utf-8")
    rec = capture(s, blob, tool="write", root=ws)
    assert rec.skipped and "snapshot store" in rec.skipped


# -- history and the session log -------------------------------------------


def test_history_lists_one_file_oldest_first(s, ws):
    (ws / "a.py").write_text("v1\n", encoding="utf-8")
    (ws / "b.py").write_text("bee\n", encoding="utf-8")
    edit(s, ws, "a.py", "v2\n")
    edit(s, ws, "b.py", "bee2\n")
    edit(s, ws, "a.py", "v3\n")

    got = history(s, "a.py", root=ws)
    assert [r.size for r in got] == [3, 3]
    assert [Store(ws).get(r.hash) for r in got] == [b"v1\n", b"v2\n"]
    assert all(r.path == "a.py" for r in got)
    assert [r.path for r in history(s, ws / "b.py", root=ws)] == ["b.py"]


def test_a_snapshot_records_the_tool_and_the_call_that_caused_it(s, ws):
    (ws / "a.py").write_text("v1\n", encoding="utf-8")
    call = s.append("tool_call", {"tool": "edit"})
    edit(s, ws, "a.py", "v2\n", tool="edit", call=call.id)

    rec = history(s, "a.py", root=ws)[0]
    assert (rec.tool, rec.call) == ("edit", call.id)
    assert rec.ts > 0 and rec.id, "the index entry must carry a timestamp and an id"


def test_snapshots_stay_out_of_the_conversation(s, ws):
    (ws / "a.py").write_text("v1\n", encoding="utf-8")
    first = s.say("user", "hello")
    edit(s, ws, "a.py", "v2\n")
    second = s.say("assistant", "done")

    assert second.parent == first.id, "a snapshot must never become a parent"
    assert s.leaf == second.id
    assert not any(e.type == SNAPSHOT for e in s.transcript())
    assert not any(e.type == SNAPSHOT for _d, e, _a, _l in s.rows())


def test_the_store_and_the_history_survive_a_reload(s, ws):
    (ws / "a.py").write_text("original\n", encoding="utf-8")
    mark = s.say("user", "go")
    edit(s, ws, "a.py", "mangled\n")
    s.close()

    again = Session.open(s.path)
    try:
        assert [r.path for r in history(again, "a.py", root=ws)] == ["a.py"]
        got = restore(again, mark.id, root=ws)
        assert got.restored == ["a.py"]
        assert (ws / "a.py").read_text(encoding="utf-8") == "original\n"
    finally:
        again.close()


def test_a_moved_workspace_restores_through_an_explicit_root(s, ws, tmp_path):
    (ws / "a.py").write_text("original\n", encoding="utf-8")
    mark = s.say("user", "go")
    edit(s, ws, "a.py", "mangled\n")

    moved = tmp_path / "moved"
    os.rename(ws, moved)
    got = restore(s, mark.id, root=moved)
    assert got.restored == ["a.py"]
    assert (moved / "a.py").read_text(encoding="utf-8") == "original\n"


# -- the runtime's side of the contract ------------------------------------


def test_target_paths_reads_the_tool_argument_convention():
    assert target_paths({"path": "a.py"}) == ["a.py"]
    assert target_paths({"paths": ["a.py", "b.py"]}) == ["a.py", "b.py"]
    assert target_paths({"command": "rm -rf /"}) == [], "guessing a shell's targets is worse than nothing"


def test_capture_all_captures_each_file_once(s, ws):
    (ws / "a.py").write_text("a\n", encoding="utf-8")
    (ws / "b.py").write_text("b\n", encoding="utf-8")
    got = capture_all(s, ["a.py", "b.py", "a.py"], tool="write", call="CALL", root=ws)
    assert [r.path for r in got] == ["a.py", "b.py"]
    assert {r.call for r in got} == {"CALL"}


def test_a_record_round_trips_through_the_session_log(s, ws):
    (ws / "a.py").write_text("v1\n", encoding="utf-8")
    rec = capture(s, "a.py", tool="write", call="CALL", root=ws)
    entry = s.entry(rec.id)
    assert entry is not None and entry.type == SNAPSHOT
    assert Record.from_entry(entry) == rec
