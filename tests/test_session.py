"""Invariants of the session store.

The store's whole value is that abandoning a branch loses nothing and that a
reload reproduces the exact state.  These tests defend those two properties
and the robustness rules that keep a damaged log openable.
"""

from __future__ import annotations

import json

import threading

import pytest

from offset.core.entries import CONVERSATIONAL, MESSAGE, Entry, new_id
from offset.core.session import Session


@pytest.fixture()
def s(tmp_path):
    sess = Session.create(tmp_path)
    yield sess
    sess.close()


# -- identifiers ------------------------------------------------------------


def test_ids_sort_chronologically():
    ids = [new_id() for _ in range(500)]
    assert ids == sorted(ids), "ids must sort lexicographically into time order"
    assert len(set(ids)) == 500, "ids collided"
    assert all(len(i) == 26 for i in ids)


def test_ids_stay_monotonic_inside_one_millisecond():
    ms = 1_700_000_000_000
    ids = [new_id(ms) for _ in range(100)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 100


# -- appending --------------------------------------------------------------


def test_append_chains_onto_the_leaf(s):
    a = s.say("user", "one")
    b = s.say("assistant", "two")
    assert a.parent is None
    assert b.parent == a.id
    assert s.leaf == b.id


def test_transcript_is_the_active_path_only(s):
    s.say("user", "one")
    s.say("assistant", "two")
    s.append("tool_call", {"tool": "read"})
    s.label(s.leaf, "here")
    assert [e.text for e in s.transcript() if e.type == MESSAGE] == ["one", "two"]
    assert all(e.type in CONVERSATIONAL for e in s.transcript())


# -- branching --------------------------------------------------------------


def test_branching_keeps_both_histories(s):
    root = s.say("user", "start")
    s.say("assistant", "approach A")
    a_leaf = s.leaf
    s.branch(root.id)
    s.say("assistant", "approach B")
    b_leaf = s.leaf

    assert a_leaf != b_leaf
    assert len(s.children(root.id)) == 2, "the abandoned branch was lost"
    assert [e.text for e in s.transcript()] == ["start", "approach B"]
    assert [e.text for e in s.transcript(a_leaf)] == ["start", "approach A"]


def test_branch_to_unknown_entry_is_refused(s):
    s.say("user", "hello")
    with pytest.raises(KeyError):
        s.branch("NOPE")


def test_reset_leaf_returns_to_the_root(s):
    s.say("user", "one")
    s.say("assistant", "two")
    s.reset_leaf()
    assert s.leaf is None
    assert s.transcript() == []
    fresh = s.say("user", "restart")
    assert fresh.parent is None
    assert len(s.roots()) == 2


def test_active_branch_is_ordered_first(s):
    root = s.say("user", "start")
    s.say("assistant", "A")
    s.branch(root.id)
    s.say("assistant", "B")
    kids = s.tree()[0].children
    assert kids[0].entry.text == "B", "the active branch must lead"
    assert kids[0].active and not kids[1].active


# -- persistence ------------------------------------------------------------


def test_reload_reproduces_tree_and_leaf(tmp_path):
    s = Session.create(tmp_path)
    root = s.say("user", "start")
    s.say("assistant", "A")
    s.branch(root.id)
    s.say("assistant", "B")
    s.label(root.id, "pivot")
    before = (s.leaf, [(d, e.id, a, l) for d, e, a, l in s.rows()])
    s.close()

    again = Session.open(s.path)
    assert (again.leaf, [(d, e.id, a, l) for d, e, a, l in again.rows()]) == before
    assert again.label_of(root.id) == "pivot"


def test_leaf_moves_are_replayed_in_order(tmp_path):
    s = Session.create(tmp_path)
    root = s.say("user", "start")
    s.say("assistant", "A")
    s.branch(root.id)
    s.branch(None)
    s.close()
    assert Session.open(s.path).leaf is None


def test_corrupt_lines_are_skipped_not_fatal(tmp_path):
    s = Session.create(tmp_path)
    s.say("user", "good")
    s.close()
    with s.path.open("a", encoding="utf-8") as fh:
        fh.write("{not json at all\n")
        fh.write(json.dumps({"id": 42, "type": "message"}) + "\n")  # bad id type
        fh.write("\n")
        fh.write(json.dumps({"id": new_id(), "type": MESSAGE, "data": {"role": "user", "text": "after"}}) + "\n")

    again = Session.open(s.path)
    assert again.skipped_lines == 2
    assert [e.text for e in again.all_entries() if e.type == MESSAGE] == ["good", "after"]


def test_duplicate_ids_keep_the_first(tmp_path):
    s = Session.create(tmp_path)
    first = s.say("user", "original")
    s.close()
    with s.path.open("a", encoding="utf-8") as fh:
        clone = Entry(id=first.id, type=MESSAGE, parent=None, data={"role": "user", "text": "impostor"})
        fh.write(clone.to_json() + "\n")
    again = Session.open(s.path)
    assert len(again) == 1
    assert again.entry(first.id).text == "original"


# -- damaged shapes ---------------------------------------------------------


def test_orphans_and_self_parents_become_roots(tmp_path):
    s = Session.create(tmp_path)
    s.say("user", "real root")
    s.close()
    orphan = Entry(id=new_id(), type=MESSAGE, parent="MISSING", data={"role": "user", "text": "orphan"})
    selfie = Entry(id=new_id(), type=MESSAGE, data={"role": "user", "text": "selfie"})
    selfie.parent = selfie.id
    with s.path.open("a", encoding="utf-8") as fh:
        fh.write(orphan.to_json() + "\n")
        fh.write(selfie.to_json() + "\n")

    again = Session.open(s.path)
    assert {e.text for e in again.roots()} == {"real root", "orphan", "selfie"}
    assert again.ancestry(selfie.id) == [again.entry(selfie.id)], "self-parent must not loop"


def test_ancestry_survives_a_parent_cycle(tmp_path):
    s = Session.create(tmp_path)
    a = Entry(id=new_id(), type=MESSAGE, data={"role": "user", "text": "a"})
    b = Entry(id=new_id(), type=MESSAGE, parent=a.id, data={"role": "user", "text": "b"})
    a.parent = b.id  # a cycle that only a hand-edited file could produce
    with s.path.open("a", encoding="utf-8") as fh:
        fh.write(a.to_json() + "\n")
        fh.write(b.to_json() + "\n")
    again = Session.open(s.path)
    assert len(again.ancestry(b.id)) <= 2  # terminates


# -- labels -----------------------------------------------------------------


def test_labels_are_last_write_wins_and_clearable(s):
    e = s.say("user", "mark me")
    s.label(e.id, "first")
    s.label(e.id, "second")
    assert s.label_of(e.id) == "second"
    s.label(e.id, None)
    assert s.label_of(e.id) is None


def test_labels_never_enter_the_tree(s):
    e = s.say("user", "one")
    s.label(e.id, "x")
    assert [entry.type for _, entry, _, _ in s.rows()] == [MESSAGE]
    assert s.leaf == e.id, "a label must not move the leaf"


# -- whole-session operations ----------------------------------------------


def test_fork_is_independent(tmp_path, s):
    s.say("user", "shared history")
    forked = s.fork(tmp_path / "forks")
    forked.say("assistant", "only in the fork")
    reloaded = Session.open(s.path)

    assert forked.path != s.path
    assert len(forked) == 2
    assert len(reloaded) == 1, "the fork wrote back into its parent"
    forked.close()


def test_compact_preserves_state_and_drops_churn(tmp_path):
    s = Session.create(tmp_path)
    root = s.say("user", "start")
    s.say("assistant", "A")
    for _ in range(20):
        s.branch(root.id)
        s.branch(s.roots()[0].id)
    s.label(root.id, "pivot")
    before_rows = [(d, e.id) for d, e, _, _ in s.rows()]
    before_leaf = s.leaf

    dropped = s.compact()
    assert dropped > 20
    assert [(d, e.id) for d, e, _, _ in s.rows()] == before_rows
    assert s.leaf == before_leaf
    assert s.label_of(root.id) == "pivot"
    s.close()

    assert Session.open(s.path).leaf == before_leaf


def test_compact_leaves_the_original_intact_on_failure(tmp_path, monkeypatch):
    s = Session.create(tmp_path)
    s.say("user", "precious")
    monkeypatch.setattr("offset.core.session.os.replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        s.compact()
    s.close()
    assert [e.text for e in Session.open(s.path).all_entries()] == ["precious"]
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".jsonl" and p != s.path], "temp file leaked"


# -- a damaged log stays openable --------------------------------------------


def test_a_session_with_undecodable_bytes_still_opens(tmp_path):
    """One bad byte used to make the whole session unopenable.

    Iteration decodes lazily, so the UnicodeDecodeError came out of the `for`
    line rather than the parse below it, and skipped past every guard.
    """
    path = tmp_path / "s.jsonl"
    good = json.dumps({"id": "01AAA", "type": "message", "data": {"role": "user", "text": "hi"}})
    path.write_bytes(good.encode("utf-8") + b"\n\xff\xfe torn write \n")

    session = Session.open(path)
    kept = list(session.all_entries())
    assert len(kept) == 1, "the readable entry must survive"
    assert session.skipped_lines == 1, "the damaged line must be counted, not ignored"


def test_a_damaged_session_is_still_writable(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_bytes(b"\xff\xfe not even close to json\n")
    session = Session.open(path)
    session.say("user", "carrying on")
    assert len(list(Session.open(path).all_entries())) == 1


def test_concurrent_appends_never_tear_a_line(tmp_path):
    """Eight threads writing at once must produce eight times the lines, all valid."""
    session = Session.create(tmp_path / "s")

    def spam(n: int) -> None:
        for i in range(30):
            session.say("user", f"thread {n} line {i}")

    threads = [threading.Thread(target=spam, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    reopened = Session.open(session.path)
    assert len(list(reopened.all_entries())) == 8 * 30
    assert reopened.skipped_lines == 0, "a torn line means the append was not atomic"


def test_a_very_deep_chain_does_not_blow_the_stack(tmp_path):
    """Six thousand turns is a long day, not an error."""
    session = Session.create(tmp_path / "s")
    for i in range(6000):
        session.say("user", f"deep {i}")
    assert len(list(session.transcript())) == 6000
