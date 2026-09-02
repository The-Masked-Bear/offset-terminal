"""Nothing may draw outside the box it was given.

Every bug in here was visible on screen: the welcome panel wrote its last rows
over its own bottom border and into its drop shadow, and /help built rows wider
than the pane so the right-hand column was cut off the edge of the terminal.
"""

from __future__ import annotations


import pytest

from offset.eggs.engine import Reveal
from offset.shell import render
from offset.shell.app import build_state
from offset.shell.commands import COMMANDS, dispatch


@pytest.fixture()
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NO_COLOR", "1")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return build_state(workspace, model="mock")


def rows(block: str) -> list[str]:
    return block.split("\n")


# -- the welcome panel -------------------------------------------------------


@pytest.mark.parametrize("height", range(4, 30))
def test_the_welcome_panel_is_exactly_the_height_it_was_given(height):
    assert len(rows(render.welcome(78, height, "/tmp/x", "mock"))) == height


@pytest.mark.parametrize("width", [30, 44, 45, 60, 100, 200])
def test_the_welcome_panel_is_exactly_the_width_it_was_given(width):
    for line in rows(render.welcome(width, 16, "/tmp/x", "mock")):
        assert len(line) == width, f"{len(line)} != {width}: {line!r}"


@pytest.mark.parametrize("height", range(9, 22))
def test_nothing_is_written_over_the_panel_border(height):
    """The regression: teaching rows landed on the bottom edge and the shadow."""
    body = render.welcome(78, height, "/tmp/x", "mock")
    for line in rows(body):
        stripped = line.strip()
        if not stripped.startswith("\u2599"):  # the bottom-left corner glyph
            continue
        rest = stripped.lstrip("\u2599")
        assert set(rest) <= {"\u2584", "\u259f", "\u2588", " "}, \
            f"text was drawn over the panel's bottom border: {line!r}"


@pytest.mark.parametrize("height", range(11, 24))
def test_the_panel_always_teaches_at_least_one_command(height):
    """A panel with room only for a headline is worse than a cramped one."""
    body = render.welcome(78, height, "/tmp/x", "mock").upper()
    assert "JUST TYPE" in body, f"height {height} taught nothing"


def test_a_tiny_terminal_gets_the_one_line_fallback():
    body = render.welcome(30, 5, "/tmp/x", "mock")
    assert "/HELP" in body.upper()
    assert len(rows(body)) == 5


# -- /help -------------------------------------------------------------------


@pytest.mark.parametrize("width", [40, 62, 63, 80, 100, 132, 200])
def test_help_never_exceeds_the_pane(state, width):
    state.width = width
    for line in dispatch(state, "/help").lines:
        assert len(line) <= width - 2, f"{len(line)} > {width - 2}: {line!r}"


@pytest.mark.parametrize("width", [40, 62, 80, 100, 200])
def test_every_command_is_listed_at_every_width(state, width):
    state.height = 60  # tall enough that width is the only variable
    state.width = width
    body = "\n".join(dispatch(state, "/help").lines)
    for command in COMMANDS:
        assert f"/{command.name}" in body, f"/{command.name} vanished at width {width}"


def test_a_narrow_pane_does_not_use_two_annotated_columns(state):
    """Two cramped columns beat neither.

    This used to assert one rendered line per command, which stopped being the
    right shape when the list grew past forty: at that size the annotated
    layout no longer fits the pane's *height*, and packing the bare names is
    strictly better than dropping the summaries' owners.  What still has to
    hold is that no line carries two summaries side by side.

    The detection has to say that and only that.  Counting words per line
    caught the bare-name grid too - the very shape the paragraph above
    endorses - so it began failing the moment the command list grew again.
    A paired *summary* is specifically a line with prose sitting between two
    command names; a line of nothing but names is the dense grid working.
    """
    state.width, state.height = 50, 60
    lines = [line for line in dispatch(state, "/help").lines if line.strip()]

    def pairs_summaries(line: str) -> bool:
        words = line.split()
        at = [i for i, word in enumerate(words) if word.startswith("/")]
        if len(at) < 2:
            return False
        return any(not words[i].startswith("/") for i in range(at[0], at[-1]))

    doubled = [line for line in lines if pairs_summaries(line)]
    assert not doubled, f"narrow pane paired summaries up anyway: {doubled[:2]}"


def test_a_narrow_pane_still_lists_every_command(state):
    """The grid may drop summaries; it may never drop a command.

    This is the invariant the test above was reaching for.  A user on a
    50-column terminal who cannot see that `/spec` exists has no way to
    discover it.
    """
    state.width, state.height = 50, 60
    body = " ".join(dispatch(state, "/help").lines)
    missing = [c.name for c in COMMANDS if f"/{c.name}" not in body]
    assert not missing, f"unreachable at 50 columns: {missing}"


def test_a_wide_pane_uses_two_columns(state):
    state.width, state.height = 140, 60
    lines = [line for line in dispatch(state, "/help").lines if line.strip()]
    assert len(lines) < len(COMMANDS), "a wide pane should pair the commands up"


@pytest.mark.parametrize("size", [(40, 20), (80, 24), (100, 32), (30, 14)])
def test_help_fits_the_rows_it_will_be_given(state, size):
    """The pane scrolls from the *top*, so emitting more lines than fit does
    not overflow harmlessly - it deletes the beginning of the list in silence.
    """
    from offset.shell.commands import MESSAGE_CHROME

    state.width, state.height = size
    lines = dispatch(state, "/help").lines
    budget = max(0, state.height - MESSAGE_CHROME)
    assert len(lines) <= budget, f"{len(lines)} lines into a {budget}-row pane"


def test_a_pane_too_small_for_the_list_says_so(state):
    """Silently showing two thirds of the commands is the failure this whole
    layout exists to avoid."""
    state.width, state.height = 40, 20
    body = "\n".join(dispatch(state, "/help").lines)
    shown = [c for c in COMMANDS if f"/{c.name}" in body]
    if len(shown) < len(COMMANDS):
        assert "more" in body, "commands were hidden without a word about it"


def test_help_can_be_filtered(state):
    state.width, state.height = 80, 24
    body = "\n".join(dispatch(state, "/help", ).lines)
    assert "/spec" in body
    narrowed = "\n".join(dispatch(state, "/help session").lines)
    assert "/sessions" in narrowed
    assert "/spec" not in narrowed, "the filter matched everything"


@pytest.mark.parametrize("width", [40, 80])
def test_help_never_overflows_its_pane(state, width):
    """Whatever the width, no rendered line may be wider than the pane.

    This used to assert that width 80 always truncated something, which was a
    fact about the command set rather than about the layout: adding a longer
    command name widened the name column until every summary fitted, and the
    test failed while the renderer was behaving correctly.
    """
    state.width = width
    lines = dispatch(state, "/help").lines
    too_wide = [line for line in lines if len(line) > width]
    assert not too_wide, f"lines exceeded {width} columns: {too_wide[:2]}"


def test_a_summary_too_long_for_the_pane_is_marked_as_cut(state):
    """A pane narrow enough to force a cut must say that it cut."""
    state.width = 40
    body = "\n".join(dispatch(state, "/help").lines)
    assert "\u2026" in body or ".." in body, "a cut summary must say it was cut"


# -- easter egg reveals ------------------------------------------------------


def egg(tone: str = "branch", lines: tuple[str, ...] = ("first line", "second line")) -> Reveal:
    return Reveal(egg_id="x", title="trophy unlocked", lines=list(lines),
                  tone=tone, frames=(), duration=3.0, achievement=None)


def test_a_reveal_types_itself_on_then_settles():
    early = render.reveal_panel(52, egg(), 100.0, elapsed=0.02)
    late = render.reveal_panel(52, egg(), 100.0, elapsed=3.0)
    assert "second line" in late
    assert "second line" not in early, "the body should not appear all at once"


def test_a_failing_reveal_glitches_instead():
    calm = render.reveal_panel(52, egg(tone="branch"), 100.0, elapsed=2.0)
    broken = render.reveal_panel(52, egg(tone="err"), 100.0, elapsed=2.0)
    assert "first line" in calm
    assert "first line" not in broken, "an err-toned reveal should be glitched"


def test_a_reveal_stays_inside_its_box():
    body = render.reveal_panel(52, egg(lines=tuple(f"line {i}" for i in range(6))), 100.0, elapsed=9.0)
    for line in rows(body):
        assert len(line) == 52, f"{len(line)} != 52: {line!r}"


def test_the_title_is_centred():
    body = rows(render.reveal_panel(60, egg(), 100.0, elapsed=9.0))
    title_row = next(line for line in body if "T R O P H Y" in line)
    left = len(title_row) - len(title_row.lstrip(" \u258c"))
    right = len(title_row) - len(title_row.rstrip(" \u2590"))
    assert abs(left - right) <= 2, f"title is not centred: {title_row!r}"
