"""Project instruction files, and the scrollable transcript.

Both are things a user notices immediately when they are missing: a coding agent
that ignores AGENTS.md, and a transcript you cannot read back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from offset.core import context
from offset.core.entries import MESSAGE, Entry
from offset.shell.render import Transcript, transcript


# -- discovery --------------------------------------------------------------


@pytest.fixture()
def tree(tmp_path):
    """workspace/pkg, with a ceiling above the workspace so the walk terminates."""
    root = tmp_path / "root"
    (root / "work" / "pkg").mkdir(parents=True)
    return root


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_an_agents_file_is_found(tree):
    write(tree / "work" / "AGENTS.md", "always run the tests")
    found = context.discovered(tree / "work", ceiling=tree)
    assert [got.path.name for got in found] == ["AGENTS.md"]
    assert found[0].text == "always run the tests"


def test_every_accepted_name_is_read(tree):
    for name in ("OFFSET.md", "AGENTS.md", "CLAUDE.md"):
        write(tree / "work" / name, f"from {name}")
    names = [got.path.name for got in context.discovered(tree / "work", ceiling=tree)]
    assert names == ["OFFSET.md", "AGENTS.md", "CLAUDE.md"], "preference order must be stable"


def test_the_walk_goes_upward_closest_first(tree):
    write(tree / "work" / "AGENTS.md", "workspace rule")
    write(tree / "AGENTS.md", "outer rule")
    found = context.discovered(tree / "work" / "pkg", ceiling=tree)
    assert [got.depth for got in found] == [1, 2]
    assert "workspace rule" in found[0].text


def test_the_closest_file_gets_the_last_word(tree):
    write(tree / "AGENTS.md", "outer rule")
    write(tree / "work" / "AGENTS.md", "inner rule")
    block = context.assemble(tree / "work", ceiling=tree)
    assert block.index("outer rule") < block.index("inner rule"), (
        "the nearest file exists to override the one above it, so it must come last"
    )


def test_a_missing_file_is_not_an_error(tree):
    assert context.discovered(tree / "work", ceiling=tree) == []
    assert context.assemble(tree / "work", ceiling=tree) == ""


def test_an_empty_file_is_ignored(tree):
    write(tree / "work" / "AGENTS.md", "   \n\n")
    assert context.discovered(tree / "work", ceiling=tree) == []


def test_the_same_file_is_never_read_twice(tree):
    write(tree / "work" / "AGENTS.md", "once")
    found = context.discovered(tree / "work", ceiling=tree)
    assert len(found) == 1
    assert context.assemble(tree / "work", ceiling=tree).count("once") == 1


# -- frontmatter ------------------------------------------------------------


def test_frontmatter_is_parsed_and_stripped(tree):
    write(tree / "work" / "AGENTS.md", "---\nalwaysApply: false\nglobs: *.py, *.pyi\n---\npython rules")
    got = context.discovered(tree / "work", ceiling=tree)[0]
    assert got.text == "python rules", "the frontmatter must not reach the model"
    assert got.always is False and got.globs == ("*.py", "*.pyi")
    assert got.conditional


def test_a_conditional_file_only_applies_to_matching_paths(tree):
    write(tree / "work" / "AGENTS.md", "---\nalwaysApply: false\nglobs: *.sql\n---\nmigration rules")
    assert context.assemble(tree / "work", paths=["schema.sql"], ceiling=tree) != ""
    assert context.assemble(tree / "work", paths=["parser.py"], ceiling=tree) == ""
    assert context.assemble(tree / "work", ceiling=tree) == "", "no paths in play means no match"


def test_an_unconditional_file_always_applies(tree):
    write(tree / "work" / "AGENTS.md", "---\nalwaysApply: true\n---\nhouse style")
    assert "house style" in context.assemble(tree / "work", paths=[], ceiling=tree)


def test_unclosed_frontmatter_is_treated_as_body(tree):
    write(tree / "work" / "AGENTS.md", "---\nalwaysApply: false\nstill going")
    got = context.discovered(tree / "work", ceiling=tree)[0]
    assert "still going" in got.text
    assert got.always, "a malformed block must not silently disable the file"


def test_frontmatter_parsing_never_raises():
    assert context.parse_frontmatter("no frontmatter") == ({}, "no frontmatter")
    assert context.parse_frontmatter("") == ({}, "")
    assert context.parse_frontmatter("---\n---\nbody")[1] == "body"


# -- budget -----------------------------------------------------------------


def test_an_oversized_file_is_cut_with_a_visible_marker(tree):
    write(tree / "work" / "AGENTS.md", "x" * (context.MAX_FILE_BYTES + 5_000))
    got = context.discovered(tree / "work", ceiling=tree)[0]
    assert got.truncated
    assert got.text.endswith(context.CUT_MARKER.strip())
    assert "truncated" in got.label()


def test_the_total_budget_is_enforced_and_announced(tree):
    write(tree / "AGENTS.md", "a" * 4_000)
    write(tree / "work" / "AGENTS.md", "b" * 4_000)
    block = context.assemble(tree / "work", budget=2_000, ceiling=tree)
    assert len(block.encode()) < 6_000
    assert "truncated" in block or "omitted" in block, "a silent cut is the worst outcome"


def test_each_block_names_its_source(tree):
    write(tree / "work" / "AGENTS.md", "the rule")
    block = context.assemble(tree / "work", ceiling=tree)
    assert "instructions from" in block and "AGENTS.md" in block


def test_the_summary_explains_itself_when_nothing_is_found(tree):
    lines = context.summary(tree / "work")
    assert "no project instructions found" in lines[0]
    assert "AGENTS.md" in lines[1]


# -- scrollback -------------------------------------------------------------


def entries(count: int) -> list[Entry]:
    return [
        Entry(id=f"e{i:04d}", type=MESSAGE, data={"role": "assistant", "text": f"line number {i}"})
        for i in range(count)
    ]


def test_the_tail_is_shown_by_default():
    view = Transcript()
    body = transcript(60, 10, entries(50), view=view)
    assert "line number 49" in body
    assert "line number 0" not in body
    assert view.at_end


def test_scrolling_up_reveals_older_output():
    view = Transcript()
    rows = view.lines(60, entries(50))
    view.scroll(40, total=len(rows), height=10)
    body = transcript(60, 10, entries(50), view=view)
    assert "line number 49" not in body
    assert not view.at_end


def test_new_output_does_not_yank_the_view_away():
    """The whole point: reading history must not be interrupted."""
    view = Transcript()
    history = entries(50)
    rows = view.lines(60, history)
    view.scroll(30, total=len(rows), height=10)
    before = transcript(60, 10, history, view=view)
    after = transcript(60, 10, history + entries(5), view=view)
    assert before.split("\n")[0] == after.split("\n")[0], "the top line must not move"


def test_following_sticks_to_the_bottom():
    view = Transcript()
    transcript(60, 10, entries(20), view=view)
    assert view.follow
    body = transcript(60, 10, entries(40), view=view)
    assert "line number 39" in body


def test_to_end_resumes_following():
    view = Transcript()
    rows = view.lines(60, entries(50))
    view.scroll(30, total=len(rows), height=10)
    assert not view.follow
    view.to_end()
    assert view.follow and view.at_end
    assert "line number 49" in transcript(60, 10, entries(50), view=view)


def test_scrolling_clamps_at_both_ends():
    view = Transcript()
    rows = view.lines(60, entries(20))
    view.scroll(-100, total=len(rows), height=10)
    assert view.offset == 0
    view.scroll(10_000, total=len(rows), height=10)
    assert view.offset == max(0, len(rows) - 10)


def test_the_hidden_count_is_reported():
    view = Transcript()
    rows = view.lines(60, entries(50))
    view.scroll(25, total=len(rows), height=10)
    body = transcript(60, 10, entries(50), view=view)
    assert f"{view.offset} more below".replace(" ", "") in body.replace(" ", "").lower()


def test_wrapping_is_cached_across_frames():
    """At 12fps, re-wrapping the whole history every frame is the slow thing."""
    view = Transcript()
    history = entries(30)
    for _ in range(5):
        transcript(60, 10, history, view=view)
    assert view.wraps == 30, f"wrapped {view.wraps} times for 30 entries"


def test_a_resize_invalidates_the_cache():
    view = Transcript()
    history = entries(10)
    transcript(60, 10, history, view=view)
    first = view.wraps
    transcript(40, 10, history, view=view)
    assert view.wraps == first * 2, "a different width needs a different wrap"


def test_live_text_is_not_cached():
    view = Transcript()
    history = entries(3)
    transcript(60, 10, history, live="streaming...", view=view)
    before = view.wraps
    transcript(60, 10, history, live="streaming more...", view=view)
    assert view.wraps == before, "only entries are cached; live text is re-wrapped"
    assert "streaming more" in transcript(60, 10, history, live="streaming more...", view=view)


def test_a_short_transcript_has_no_scrollbar():
    view = Transcript()
    body = transcript(60, 20, entries(3), view=view)
    assert "\u2588" not in body.split("\n")[0][-1:], "nothing to scroll, nothing to draw"
