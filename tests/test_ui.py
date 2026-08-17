"""Invariants of the design system.

These are not snapshot tests.  Each one defends a rule that, if broken, makes
the aesthetic wrong rather than merely different: shadows must stay hard and
on the diagonal, pressing must consume the shadow, tracked text must never
escape its box, and the canvas must never emit colour it was not asked for.
"""

from __future__ import annotations

import pytest

from offset.ui import anim, brutal
from offset.ui.canvas import Canvas
from offset.ui.tokens import (
    ASCII,
    BORDERS,
    G,
    INK,
    PAPER,
    PRESS,
    SHADOW_LG,
    SHADOW_MD,
    SHADOW_SM,
    SURFACE,
    Depth,
    Weight,
    fit,
    shadow_cells,
    text_width,
    to_16,
    to_256,
    track,
)


# -- geometry ---------------------------------------------------------------


def test_shadow_stays_on_the_diagonal():
    """A cell is ~2:1, so a 45-degree shadow needs exactly twice the columns."""
    for px in (4, 6, 8, 10, 12, 15, 18, 20, 23):
        dx, dy = shadow_cells(px)
        assert dx == 2 * dy, f"{px}px produced {dx}x{dy}, which is not 45 degrees"


def test_shadow_scale_is_monotonic_and_bounded():
    seen = [shadow_cells(px) for px in range(1, 40)]
    assert seen == sorted(seen)
    assert shadow_cells(0) == (0, 0)
    assert (SHADOW_SM, SHADOW_MD, SHADOW_LG) == ((2, 1), (4, 2), (6, 3))


def test_press_moves_into_its_own_shadow():
    """`translate(4px,4px)` + shadow collapse: the box lands where the shadow was."""
    up = Canvas(30, 8, fg=INK, bg=PAPER)
    down = Canvas(30, 8, fg=INK, bg=PAPER)
    brutal.slab(up, 2, 1, 20, 5, shadow=SHADOW_SM, pressed=False)
    brutal.slab(down, 2, 1, 20, 5, shadow=SHADOW_SM, pressed=True)
    corner = BORDERS[Weight.SLAB].tl
    assert up._ch[1 * 30 + 2] == corner
    assert down._ch[(1 + PRESS[1]) * 30 + (2 + PRESS[0])] == corner
    # nothing of the shadow survives the press
    assert down._ch.count(G.SHADOW) == 0
    assert up._ch.count(G.SHADOW) > 0


def test_shadow_is_offset_down_and_right_only():
    cv = Canvas(40, 12, fg=INK, bg=PAPER)
    brutal.slab(cv, 4, 2, 20, 6, shadow=SHADOW_MD)
    cells = [(i % 40, i // 40) for i, c in enumerate(cv._ch) if c == G.SHADOW]
    assert cells, "no shadow drawn"
    assert min(x for x, _ in cells) > 4, "shadow leaked to the left of the box"
    assert min(y for _, y in cells) > 2, "shadow leaked above the box"


def test_nothing_is_rounded():
    cv = Canvas(30, 8)
    brutal.slab(cv, 1, 1, 20, 5, weight=Weight.SLAB)
    for w in Weight:
        b = BORDERS[w]
        assert b.tl not in "\u256d\u256e\u256f\u2570", "rounded corner glyph in the border set"


# -- typography -------------------------------------------------------------


@pytest.mark.parametrize("width", range(1, 40))
def test_fit_never_exceeds_its_box(width):
    for s in ("models live", "qwen3 coder 480b", "a", "branch c / defer to runtime"):
        assert text_width(fit(s, width, spacing=2)) <= width


def test_fit_spends_tracking_before_truncating():
    """Wide spacing is the luxury; the letters are not."""
    s = "branches"
    roomy = fit(s, 40, spacing=2)
    snug = fit(s, len(s), spacing=2)
    assert roomy == track(s.upper(), 2)
    assert snug == s.upper()  # tracking gone, text intact


def test_fit_marks_truncation():
    out = fit("defer to the runtime", 8, spacing=1)
    assert text_width(out) <= 8
    assert out.endswith(".." if ASCII else "\u2026")


def test_fit_of_zero_width_is_empty():
    assert fit("anything", 0) == ""


# -- canvas -----------------------------------------------------------------


def test_drawing_outside_the_canvas_is_silent():
    cv = Canvas(10, 3)
    cv.put(-5, -5, "x")
    cv.put(99, 99, "x")
    cv.text(-3, 1, "overhang")
    cv.fill_rect(-4, -4, 100, 100, "#")
    assert len(cv._ch) == 30


def test_text_is_clipped_not_wrapped():
    cv = Canvas(8, 1)
    cv.text(0, 0, "abcdefghijklmno")
    assert "".join(cv._ch) == "abcdefgh"


def test_wide_characters_hold_the_grid():
    cv = Canvas(6, 1)
    cv.text(0, 0, "\u4f60\u597d")  # two double-width characters
    assert cv.render(Depth.NONE).rstrip() == "\u4f60\u597d"
    assert len(cv._ch) == 6


def test_no_escape_codes_without_colour():
    cv = Canvas(12, 3)
    brutal.slab(cv, 0, 0, 10, 3)
    assert "\x1b" not in cv.render(Depth.NONE)


def test_colour_is_emitted_only_on_change():
    cv = Canvas(20, 1, fg=INK, bg=PAPER)
    cv.text(0, 0, "aaaaaaaaaaaaaaaaaaaa", INK, PAPER)
    assert cv.render(Depth.TRUE).count("\x1b[") == 2  # one pen set, one reset


def test_truecolor_uses_the_exact_palette():
    cv = Canvas(3, 1, fg=INK, bg=PAPER)
    out = cv.render(Depth.TRUE)
    assert "38;2;17;17;17" in out  # #111111
    assert "48;2;244;244;240" in out  # #F4F4F0


def test_degradation_ladder_stays_in_range():
    for c in (INK, PAPER, SURFACE):
        assert 0 <= to_256(c) <= 255
        assert 0 <= to_16(c) <= 15


# -- animation --------------------------------------------------------------


def test_spring_overshoots_then_settles():
    """cubic-bezier(.175,.885,.32,1.275) must exceed 1.0 before returning."""
    peak = max(anim.SPRING(i / 200) for i in range(201))
    assert peak > 1.0, "the spring does not overshoot"
    assert anim.SPRING(0.0) == 0.0
    assert anim.SPRING(1.0) == pytest.approx(1.0, abs=1e-6)


def test_easing_is_deterministic_and_bounded():
    for i in range(0, 101):
        x = i / 100
        assert anim.EASE(x) == anim.EASE(x)
        assert -0.01 <= anim.EASE(x) <= 1.01


def test_count_up_arrives_exactly():
    assert anim.count_up(42, 0.0) == 0.0
    assert anim.count_up(42, 99.0) == pytest.approx(42.0)


def test_glitch_is_stable_within_a_frame():
    a = anim.glitch("speculative branching", 3.0)
    b = anim.glitch("speculative branching", 3.0)
    assert a == b
    assert len(a) == len("speculative branching")


def test_glitch_preserves_spaces():
    out = anim.glitch("a b c", 1.0, intensity=1.0)
    assert [i for i, c in enumerate(out) if c == " "] == [1, 3]


def test_marquee_never_changes_width():
    for t in (0.0, 0.37, 5.0, 123.4):
        assert len(anim.marquee(["one", "two"], 31, t)) == 31


def test_marquee_scrolls():
    assert anim.marquee(["alpha", "beta"], 20, 0.0) != anim.marquee(["alpha", "beta"], 20, 1.0)


def test_dot_bounce_has_exactly_one_active_dot():
    for i in range(24):
        assert anim.dot_bounce(i / 6.0).count(G.DOT) == 1


def test_shake_decays_to_rest():
    assert anim.shake(1.0, amp=3, dur=0.5) == 0


# -- components -------------------------------------------------------------


def test_components_survive_absurd_geometry():
    cv = Canvas(24, 10)
    for w, h in ((0, 0), (1, 1), (2, 2), (3, 30), (200, 3)):
        brutal.slab(cv, 0, 0, w, h)
        brutal.progress(cv, 0, 0, max(w, 1), 0.5)
    brutal.stat(cv, 0, 0, 6, value="123456", caption="a very long caption indeed")
    brutal.dropdown(cv, 0, 0, 8, ["one", "two"], 0, notes=["x"])
    brutal.diff_view(cv, 0, 0, 8, 4, [("+", "add"), ("-", "del"), (" ", "ctx")])


def test_progress_is_clamped():
    cv = Canvas(20, 1)
    brutal.progress(cv, 0, 0, 12, -5.0)
    assert cv._ch.count(G.BLOCK) == 0
    cv2 = Canvas(20, 1)
    brutal.progress(cv2, 0, 0, 12, 5.0)
    assert cv2._ch.count(G.BLOCK) == 10  # full span, borders excluded


def test_selected_row_is_a_full_bleed_block():
    cv = Canvas(30, 12, fg=INK, bg=PAPER)
    brutal.dropdown(cv, 0, 0, 26, ["alpha", "beta", "gamma"], 1)
    from offset.ui.tokens import YELLOW

    rows = {i // 30 for i, c in enumerate(cv._bg) if c == YELLOW}
    assert len(rows) == 2  # the title band and exactly one selected row


def test_masked_input_never_shows_the_secret():
    cv = Canvas(40, 8)
    brutal.masked_input(cv, 0, 0, 34, caption="api key", filled=11, t=0.0)
    body = cv.render(Depth.NONE)
    assert "sk-" not in body
    assert body.count(G.DOT) == 11
