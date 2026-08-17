"""Invariants of context compaction.

Compaction is the one feature that decides to stop showing the model part of
its own history, so the tests here defend the two things that make that safe:
the boundary never lands inside a tool block, and nothing is destroyed — the
pre-compaction path stays on disk and stays reachable from the old leaf.
"""

from __future__ import annotations

import os

import pytest

from offset.core.agent import to_messages
from offset.core.compaction import (
    DEFAULT_THRESHOLD,
    Report,
    budget_for,
    compact,
    estimate_tokens,
    needs_compaction,
    outline,
    plan,
)
from offset.core.entries import COMPACTION, MESSAGE, TOOL_CALL, TOOL_RESULT
from offset.core.session import Session
from offset.providers.base import Message, ToolCall


@pytest.fixture()
def s(tmp_path):
    sess = Session.create(tmp_path)
    yield sess
    sess.close()


def exchange(sess: Session, n: int, *, tools: int = 0, size: int = 200) -> None:
    """One user turn and the assistant work it caused, in the order the agent
    records it: message, then calls, then results."""
    sess.say("user", f"request {n} " + "x" * size)
    sess.say("assistant", f"working on {n} " + "z" * size)
    for k in range(tools):
        sess.append(TOOL_CALL, {"id": f"c{n}-{k}", "tool": "read", "args": {"path": f"f{n}_{k}.py"}})
    for k in range(tools):
        sess.append(TOOL_RESULT, {"id": f"c{n}-{k}", "tool": "read", "content": "y" * size, "ok": True})


def unpaired(kept) -> list:
    """Results in `kept` whose call did not come with them: what a provider
    rejects outright."""
    calls = {e.data.get("id") for e in kept if e.type == TOOL_CALL}
    return [e for e in kept if e.type == TOOL_RESULT and e.data.get("id") not in calls]


def summarise(prompt: str) -> str:
    return "GOAL ship compaction\nDONE wrote the module\nFACTS none\nOPEN nothing"


# -- the estimate -----------------------------------------------------------


def test_the_estimate_never_falls_as_content_is_added():
    messages = [
        Message(role="user", text="a" * 40),
        Message(role="assistant", text="b" * 4000, thinking="t" * 100),
        Message(role="tool", text="c" * 800, tool_call_id="1"),
        Message(role="assistant", tool_calls=[ToolCall(id="1", name="read", args={"path": "x" * 200})]),
    ]
    running = [estimate_tokens(messages[:i]) for i in range(len(messages) + 1)]
    assert running == sorted(running), f"estimate went backwards: {running}"
    assert running[0] == 0
    assert running[1] < running[-1]


def test_the_estimate_is_characters_over_four_plus_framing():
    """Not a tokeniser, and it must not pretend to be one."""
    assert estimate_tokens([Message(role="user", text="a" * 400)]) == 4 + 100
    assert estimate_tokens([Message(role="user", text="a")]) == 4 + 1, "partial tokens round up"


def test_thinking_and_tool_arguments_are_counted():
    plain = Message(role="assistant", text="ok")
    with_thinking = Message(role="assistant", text="ok", thinking="q" * 400)
    with_call = Message(role="assistant", text="ok", tool_calls=[ToolCall(id="1", name="bash", args={"cmd": "ls" * 200})])
    assert estimate_tokens([with_thinking]) > estimate_tokens([plain])
    assert estimate_tokens([with_call]) > estimate_tokens([plain])


def test_unparsed_tool_arguments_are_counted_from_the_raw_text():
    raw = Message(role="assistant", tool_calls=[ToolCall(id="1", name="bash", args={}, raw="{" + "z" * 400)])
    empty = Message(role="assistant", tool_calls=[ToolCall(id="1", name="bash", args={})])
    assert estimate_tokens([raw]) > estimate_tokens([empty]) + 90


# -- the trigger ------------------------------------------------------------


def test_the_threshold_and_not_the_budget_decides_when_to_compact():
    messages = [Message(role="user", text="a" * 4000)]
    est = estimate_tokens(messages)
    assert needs_compaction(messages, est * 2, 0.5), "at exactly the threshold it is already time"
    assert not needs_compaction(messages, est * 2, 0.51)
    assert needs_compaction(messages, est * 2, 0.49)
    assert needs_compaction(messages, est, DEFAULT_THRESHOLD)


def test_no_budget_is_never_a_reason_to_throw_history_away():
    messages = [Message(role="user", text="a" * 100_000)]
    assert not needs_compaction(messages, 0)
    assert not needs_compaction(messages, -1)
    assert not needs_compaction([], 100)


def test_a_configured_trigger_beats_the_model_window(monkeypatch):
    from offset.core import settings

    monkeypatch.setattr(settings, "get", lambda key, default=None: 12_345 if key == "session.compactAt" else default)
    assert budget_for("mock") == 12_345
    monkeypatch.setattr(settings, "get", lambda key, default=None: 0 if key == "session.compactAt" else default)
    assert budget_for("mock") > 0, "with nothing configured the model's context window is the budget"
    assert budget_for("mock", override=0) == 0, "an explicit zero disables compaction"


# -- the boundary -----------------------------------------------------------


def test_the_first_user_message_and_the_recent_exchanges_always_survive(s):
    for n in range(8):
        exchange(s, n)
    entries = s.transcript()
    p = plan(entries, 2)
    assert p, "eight exchanges keeping two must leave something to summarise"
    first_user = next(e for e in entries if e.type == MESSAGE and e.role == "user")
    assert p.kept[0] is first_user
    assert first_user not in p.replaced
    users_kept = [e for e in p.kept if e.type == MESSAGE and e.role == "user"]
    assert [e.text.split()[1] for e in users_kept] == ["0", "6", "7"]
    accounted = [e.id for e in p.replaced] + [e.id for e in p.kept]
    assert sorted(accounted) == sorted(e.id for e in entries), "an entry was neither kept nor summarised"


def test_a_boundary_is_never_chosen_between_a_tool_call_and_its_results(s):
    for n in range(6):
        exchange(s, n, tools=2)
    entries = s.transcript()
    for keep in range(0, 9):
        p = plan(entries, keep)
        assert not unpaired(p.kept), f"keep_recent={keep} orphaned {unpaired(p.kept)}"
        assert p.boundary in (0, len(entries)) or entries[p.boundary].type != TOOL_RESULT


def test_a_result_that_arrived_after_the_next_user_message_drags_the_boundary_back(s):
    """A user can speak while a tool is still running, so results are not
    always inside the turn that asked for them."""
    s.say("user", "one")
    s.say("assistant", "reading")
    call = s.append(TOOL_CALL, {"id": "c1", "tool": "read", "args": {"path": "a.py"}})
    s.say("user", "two")
    s.append(TOOL_RESULT, {"id": "c1", "tool": "read", "content": "contents", "ok": True})
    s.say("assistant", "read it")
    s.say("user", "three")
    s.say("assistant", "done")

    entries = s.transcript()
    p = plan(entries, 2)
    assert entries[p.boundary] is call, "the boundary should have moved back onto the call"
    assert not unpaired(p.kept)
    assert call in p.kept


def test_a_result_with_no_call_id_moves_with_the_turn_it_belongs_to(s):
    s.say("user", "one")
    s.say("assistant", "one")
    s.say("user", "two")
    s.say("assistant", "two")
    s.append(TOOL_RESULT, {"tool": "read", "content": "no id at all"})
    s.say("user", "three")
    s.say("assistant", "three")
    entries = s.transcript()
    p = plan(entries, 2)
    assert entries[p.boundary].type != TOOL_RESULT


def test_a_prefix_of_nothing_but_summaries_is_not_worth_a_model_call(s):
    s.append(COMPACTION, {"text": "earlier: we fixed the tokeniser"})
    s.say("user", "carry on")
    s.say("assistant", "carrying on")
    assert not plan(s.transcript(), 6)
    assert not plan([], 6)


# -- compacting -------------------------------------------------------------


def test_compaction_puts_the_summary_first_and_shortens_the_transcript(s):
    for n in range(8):
        exchange(s, n, tools=1)
    before_tokens = estimate_tokens(to_messages(s.transcript()))
    report = compact(s, summarise, 10, 2)

    assert isinstance(report, Report) and report.done
    transcript = s.transcript()
    assert transcript[0].type == COMPACTION
    assert transcript[0].text.startswith("GOAL ship compaction")
    assert estimate_tokens(to_messages(transcript)) < before_tokens
    assert report.after < report.before and report.saved > 0
    assert not unpaired(transcript)

    # the summary reaches the model as content, not as a skipped entry
    messages = to_messages(transcript)
    assert "GOAL ship compaction" in messages[0].text

    # and it says what it stood in for
    assert report.replaced == len(transcript[0].data["replaced"])
    assert str(report.replaced) in transcript[0].data["note"]
    assert transcript[0].data["replaced_leaf"] == report.previous_leaf


def test_compaction_is_append_only_and_the_old_path_stays_reachable(s):
    for n in range(8):
        exchange(s, n, tools=1)
    old_leaf = s.leaf
    old_ancestry = [e.id for e in s.ancestry()]
    lines_before = len(s.path.read_text(encoding="utf-8").splitlines())

    report = compact(s, summarise, 10, 2)
    assert report and report.done and report.previous_leaf == old_leaf

    raw = s.path.read_text(encoding="utf-8")
    assert len(raw.splitlines()) > lines_before, "compaction must only ever add lines"
    for eid in old_ancestry:
        assert eid in raw, "an original entry left the file"
    assert [e.id for e in s.ancestry(old_leaf)] == old_ancestry, "the old path changed shape"

    # the new path shares no ids with the old one: nothing was re-parented
    new_ids = [e.id for e in s.ancestry()]
    assert not (set(new_ids) & set(old_ancestry) - {report.entry.id})
    copied = [e for e in s.transcript() if e.data.get("compacted_from")]
    assert [e.data["compacted_from"] for e in copied] == [e.id for e in _kept_originals(s, report)]

    reopened = Session.open(s.path)
    assert [e.id for e in reopened.transcript()] == new_ids, "a reload must reproduce the compacted path"
    assert [e.id for e in reopened.ancestry(old_leaf)] == old_ancestry


def _kept_originals(sess: Session, report: Report) -> list:
    replaced = set(report.entry.data["replaced"])
    return [e for e in sess.ancestry(report.previous_leaf) if e.id not in replaced]


def test_compacting_twice_with_nothing_new_to_do_is_a_no_op(s):
    for n in range(8):
        exchange(s, n)
    budget = estimate_tokens(to_messages(s.transcript()))
    assert compact(s, summarise, budget, 2).done

    assert not needs_compaction(to_messages(s.transcript()), budget), "one pass must get back under budget"
    entries_after, leaf_after = len(s), s.leaf
    asked: list[str] = []
    assert compact(s, lambda prompt: asked.append(prompt) or "x", budget, 2) is None
    assert not asked, "a no-op must not spend a model call"
    assert (len(s), s.leaf) == (entries_after, leaf_after)


def test_a_forced_compaction_with_only_a_summary_behind_it_still_does_nothing(s):
    for n in range(8):
        exchange(s, n)
    assert compact(s, summarise, 10, 2).done
    entries_after, leaf_after = len(s), s.leaf
    assert compact(s, summarise, 0, 6, force=True) is None
    assert (len(s), s.leaf) == (entries_after, leaf_after)


def test_being_under_budget_is_not_reported_as_work_done(s):
    for n in range(3):
        exchange(s, n)
    entries_after, leaf_after = len(s), s.leaf
    assert compact(s, summarise, 1_000_000, 2) is None
    assert (len(s), s.leaf) == (entries_after, leaf_after)


def test_a_failing_summariser_leaves_the_history_exactly_as_it_was(s):
    for n in range(8):
        exchange(s, n)
    entries_after, leaf_after = len(s), s.leaf

    def boom(prompt: str) -> str:
        raise RuntimeError("no credential for anthropic")

    report = compact(s, boom, 10, 2)
    assert report is not None and not report.done
    assert "no credential for anthropic" in report.error
    assert (len(s), s.leaf) == (entries_after, leaf_after)
    assert s.transcript()[0].type == MESSAGE
    assert report.before == report.after and report.saved == 0


def test_an_empty_summary_is_a_failure_and_not_an_empty_compaction(s):
    for n in range(8):
        exchange(s, n)
    entries_after = len(s)
    report = compact(s, lambda prompt: "   \n", 10, 2)
    assert report is not None and not report.done and report.error
    assert len(s) == entries_after
    assert not any(e.type == COMPACTION for e in s.all_entries())


def test_the_summariser_is_shown_the_turns_it_is_replacing(s):
    for n in range(8):
        exchange(s, n, tools=1)
    seen: list[str] = []
    assert compact(s, lambda prompt: seen.append(prompt) or summarise(prompt), 10, 2).done
    prompt = seen[0]
    assert "request 3" in prompt, "an old exchange is missing from the prompt"
    assert "tool read" in prompt
    assert "request 7" not in prompt, "the kept tail must not be summarised as well"


def test_tool_output_is_clipped_before_it_reaches_the_summariser(s):
    s.append(TOOL_RESULT, {"id": "c", "tool": "read", "content": "y" * 50_000, "ok": True})
    body = outline(list(s.all_entries()), clip=100)
    assert len(body) < 400 and body.endswith("\u2026")


# -- picking a session up again ---------------------------------------------


def test_session_list_is_newest_first_and_skips_a_corrupt_file(tmp_path):
    for i, name in enumerate(("oldest", "middle", "newest")):
        sess = Session.create(tmp_path, session_id=name)
        sess.say("user", f"hello from {name}")
        sess.say("assistant", "hi")
        sess.close()
        os.utime(sess.path, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))
    (tmp_path / "shredded.jsonl").write_text("{not json at all\nnor this\n", encoding="utf-8")
    (tmp_path / "binary.jsonl").write_bytes(b"\xff\xfe\x00not text either")
    for junk in ("shredded.jsonl", "binary.jsonl"):
        os.utime(tmp_path / junk, (1_700_009_999, 1_700_009_999))

    infos = Session.list(tmp_path)
    assert [i.id for i in infos] == ["newest", "middle", "oldest"]
    assert infos[0].messages == 2
    assert infos[0].first_line == "hello from newest"
    assert all(i.size > 0 and i.path.is_file() for i in infos)


def test_a_partly_corrupt_session_is_still_offered_with_its_damage_counted(tmp_path):
    sess = Session.create(tmp_path, session_id="dented")
    sess.say("user", "still readable")
    sess.close()
    with sess.path.open("a", encoding="utf-8") as fh:
        fh.write("half a line, no json here\n")

    (info,) = Session.list(tmp_path)
    assert info.id == "dented" and info.messages == 1 and info.skipped == 1
    assert Session.resume(sess.path).skipped_lines == 1


def test_resuming_reopens_at_the_recorded_leaf(tmp_path):
    sess = Session.create(tmp_path)
    sess.say("user", "one")
    second = sess.say("assistant", "two")
    sess.say("user", "three")
    sess.branch(second.id)
    sess.close()

    resumed = Session.resume(sess.path)
    assert resumed.leaf == second.id
    assert [e.text for e in resumed.transcript()] == ["one", "two"]
    with pytest.raises(FileNotFoundError):
        Session.resume(tmp_path / "never-existed.jsonl")
