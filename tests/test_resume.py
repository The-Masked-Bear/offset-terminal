"""Reopening an earlier session.

`Session.resume` was implemented and tested long before anything called it, so
the store could always do this and the product could not.  These tests pin the
wiring rather than the store: the CLI flag, the picker, and the slash command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offset.core.session import Session
from offset.shell.app import _pick_session


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    root = tmp_path / "home"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session_with(root: Path, *messages: str) -> Session:
    s = Session.create(root / "sessions")
    for m in messages:
        s.say("user", m)
    s.close()
    return s


def test_no_resume_flag_starts_a_fresh_session(home):
    first = _session_with(home, "hello")
    picked = _pick_session(home, None)
    assert picked.id != first.id, "absent --resume must not reopen anything"
    assert picked.transcript() == []


def test_bare_resume_takes_the_most_recent_session(home):
    _session_with(home, "older")
    newest = _session_with(home, "newer")
    picked = _pick_session(home, "")
    assert picked.id == newest.id
    assert [m.data["text"] for m in picked.transcript()] == ["newer"]


def test_resume_selects_by_full_id(home):
    wanted = _session_with(home, "the one")
    _session_with(home, "not the one")
    picked = _pick_session(home, wanted.id)
    assert picked.id == wanted.id
    assert [m.data["text"] for m in picked.transcript()] == ["the one"]


def test_resume_selects_by_id_prefix(home):
    wanted = _session_with(home, "prefix match")
    _session_with(home, "other")
    # ULIDs minted in the same millisecond share a long prefix, so a short one is
    # genuinely ambiguous; use enough of it to name a single session.
    picked = _pick_session(home, wanted.id[:20])
    assert picked.id == wanted.id


def test_an_ambiguous_prefix_resolves_to_the_most_recent_match(home):
    older = _session_with(home, "older")
    newer = _session_with(home, "newer")
    shared = 0
    for a, b in zip(older.id, newer.id):
        if a != b:
            break
        shared += 1
    if shared == 0:
        pytest.skip("ids diverged immediately; nothing ambiguous to assert")
    picked = _pick_session(home, older.id[:shared])
    assert picked.id == newer.id, "listing is newest-first, so the newest match wins"

def test_an_unknown_id_starts_fresh_rather_than_failing(home):
    _session_with(home, "existing")
    picked = _pick_session(home, "NOSUCHSESSION")
    assert picked.transcript() == [], "a typo must not cost the user their launch"


def test_resuming_with_no_sessions_at_all_is_not_an_error(home):
    picked = _pick_session(home, "")
    assert picked.transcript() == []


def test_a_resumed_session_appends_to_the_same_file(home):
    first = _session_with(home, "one")
    picked = _pick_session(home, first.id)
    picked.say("user", "two")
    picked.close()

    again = Session.open(Path(first.path))
    assert [m.data["text"] for m in again.transcript()] == ["one", "two"]
    assert again.path == first.path, "resume must continue the file, not copy it"


def test_resume_reopens_at_the_recorded_leaf(home):
    s = Session.create(home / "sessions")
    s.say("user", "first")
    second = s.say("user", "second")
    s.close()

    picked = _pick_session(home, s.id)
    assert picked.leaf == second.id


def test_the_cli_maps_continue_onto_the_most_recent(home, monkeypatch):
    newest = _session_with(home, "newest")
    seen: dict[str, object] = {}

    def fake_chat_main(**kwargs):
        seen.update(kwargs)
        return 0

    import offset.shell.app as app_mod
    monkeypatch.setattr(app_mod, "main", fake_chat_main)

    from offset.__main__ import main as cli
    assert cli(["chat", "--continue"]) == 0
    assert seen["resume"] == "", "--continue must request the most recent session"

    seen.clear()
    assert cli(["chat", "--resume", newest.id]) == 0
    assert seen["resume"] == newest.id

    seen.clear()
    assert cli(["chat"]) == 0
    assert seen["resume"] is None, "no flag must leave resume unset"


def test_bare_resume_flag_needs_no_argument(home):
    from offset.__main__ import main as cli
    seen: dict[str, object] = {}

    import offset.shell.app as app_mod
    original = app_mod.main
    app_mod.main = lambda **kw: (seen.update(kw), 0)[1]
    try:
        assert cli(["chat", "--resume"]) == 0
    finally:
        app_mod.main = original
    assert seen["resume"] == "", "bare --resume means the most recent"
