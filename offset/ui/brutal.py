"""Neubrutalist components.

Every component obeys the same three construction rules:

  * a hard border, weight chosen from the 2/3/4/5 scale;
  * a flat fill, never a gradient;
  * a zero-blur shadow offset down-right, drawn as solid blocks.

Pressing anything moves it into its own shadow (`translate(4px, 4px)`, and the
shadow collapses to nothing) — the same interaction as the reference site.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from offset.ui import anim
from offset.ui.canvas import Canvas
from offset.ui.tokens import (
    BORDERS,
    G,
    GRID,
    INK,
    MINT,
    MUTED,
    PAPER,
    PINK,
    PRESS,
    RED,
    RGB,
    SHADOW_MD,
    SHADOW,
    SHADOW_SM,
    SURFACE,
    TONES,
    Weight,
    display,
    fit,
    ink_on,
    label,
    text_width,
    track,
)


def tone(name: str | RGB) -> RGB:
    return name if isinstance(name, tuple) else TONES[name]


# --------------------------------------------------------------------------
# the primitive: a box with a hard shadow
# --------------------------------------------------------------------------


def slab(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    weight: Weight = Weight.SLAB,
    fill: RGB = SURFACE,
    edge: RGB = INK,
    shadow: tuple[int, int] = SHADOW_MD,
    shadow_color: RGB = SHADOW,
    pressed: bool = False,
    title: str | None = None,
    title_tone: str | RGB = "accent",
    tracking: int = 1,
) -> tuple[int, int, int, int]:
    """Draw a panel; return the interior rect (x, y, w, h).

    When `pressed`, the whole box shifts onto the shadow's position and the
    shadow disappears — a physical press, not a colour change.
    """
    if w < 2 or h < 2:
        return (x, y, 0, 0)
    if pressed:
        x += PRESS[0]
        y += PRESS[1]
        shadow = (0, 0)

    dx, dy = shadow
    if dx or dy:
        # L-shaped: the box itself covers the overlap, so a full offset rect
        # is both correct and cheaper than computing the L.
        cv.fill_rect(x + dx, y + dy, w, h, G.SHADOW, shadow_color, shadow_color)

    b = BORDERS[weight]
    cv.fill_rect(x, y, w, h, " ", edge, fill)
    cv.hline(x + 1, y, w - 2, b.t, edge, fill)
    cv.hline(x + 1, y + h - 1, w - 2, b.b, edge, fill)
    cv.vline(x, y + 1, h - 2, b.l, edge, fill)
    cv.vline(x + w - 1, y + 1, h - 2, b.r, edge, fill)
    cv.put(x, y, b.tl, edge, fill)
    cv.put(x + w - 1, y, b.tr, edge, fill)
    cv.put(x, y + h - 1, b.bl, edge, fill)
    cv.put(x + w - 1, y + h - 1, b.br, edge, fill)

    inner = (x + 2, y + 1, w - 4, h - 2)
    if title is not None and h >= 4:
        band = tone(title_tone)
        cv.fill_rect(x + 1, y + 1, w - 2, 1, " ", INK, band)
        cv.text(x + 2, y + 1, fit(title, w - 4, spacing=tracking - 1), ink_on(band), band, True, max_w=w - 4)
        cv.hline(x + 1, y + 2, w - 2, BORDERS[Weight.HAIRLINE].t, INK, fill)
        inner = (x + 2, y + 3, w - 4, h - 4)
    return inner


def rule(cv: Canvas, x: int, y: int, w: int, *, weight: Weight = Weight.BOLD, color: RGB = INK, bg: RGB = PAPER) -> None:
    cv.hline(x, y, w, BORDERS[weight].t, color, bg)


# --------------------------------------------------------------------------
# type
# --------------------------------------------------------------------------

#: 5x3 block digits.  The site renders hero numbers in Archivo Black at ~80px;
#: this is the cell-grid equivalent.
_BIG: dict[str, tuple[str, ...]] = {
    "0": ("###", "# #", "# #", "# #", "###"),
    "1": ("  #", "  #", "  #", "  #", "  #"),
    "2": ("###", "  #", "###", "#  ", "###"),
    "3": ("###", "  #", "###", "  #", "###"),
    "4": ("# #", "# #", "###", "  #", "  #"),
    "5": ("###", "#  ", "###", "  #", "###"),
    "6": ("###", "#  ", "###", "# #", "###"),
    "7": ("###", "  #", "  #", "  #", "  #"),
    "8": ("###", "# #", "###", "# #", "###"),
    "9": ("###", "# #", "###", "  #", "###"),
    ".": ("   ", "   ", "   ", "   ", "  #"),
    ",": ("   ", "   ", "   ", "  #", " # "),
    "-": ("   ", "   ", "###", "   ", "   "),
    "+": ("   ", " # ", "###", " # ", "   "),
    ":": ("   ", " # ", "   ", " # ", "   "),
    "/": ("  #", "  #", " # ", "#  ", "#  "),
    "%": ("# #", "  #", " # ", "#  ", "# #"),
    "x": ("   ", "# #", " # ", "# #", "   "),
    "?": ("###", "  #", " ##", "   ", " # "),
    " ": ("  ", "  ", "  ", "  ", "  "),
}
BIG_H = 5


def big_text(cv: Canvas, x: int, y: int, s: str, fg: RGB = INK, bg: RGB | None = None, *, kern: int = 1) -> int:
    """Render `s` in the 5-row block face.  Returns width drawn."""
    col = x
    for chi, ch in enumerate(s):
        rows = _BIG.get(ch) or _BIG.get(ch.lower()) or _BIG["?"]
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                if cell == "#":
                    cv.put(col + c, y + r, G.BLOCK, fg, bg)
                elif bg is not None:
                    cv.put(col + c, y + r, " ", fg, bg)
        col += len(rows[0]) + (kern if chi < len(s) - 1 else 0)
    return col - x


def big_width(s: str, *, kern: int = 1) -> int:
    if not s:
        return 0
    return sum(len((_BIG.get(c) or _BIG["?"])[0]) for c in s) + kern * (len(s) - 1)


def heading(cv: Canvas, x: int, y: int, text: str, *, fg: RGB = INK, bg: RGB = PAPER, tracking: int = 2, underline: str | RGB | None = None, width: int | None = None) -> int:
    """Display type: uppercase, wide tracking, optional solid accent bar."""
    s = display(text, tracking)
    cv.text(x, y, s, fg, bg, True, max_w=width)
    w = min(text_width(s), width or 10**6)
    if underline is not None:
        cv.fill_rect(x, y + 1, w, 1, G.BLOCK, tone(underline), bg)
    return w


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------


def button(
    cv: Canvas,
    x: int,
    y: int,
    text: str,
    *,
    fill: str | RGB = "accent",
    pressed: bool = False,
    weight: Weight = Weight.BOLD,
    shadow: tuple[int, int] = SHADOW_SM,
    tracking: int = 1,
) -> int:
    """A 3-row button.  Returns its total width (excluding shadow)."""
    caption = display(text, tracking)
    w = text_width(caption) + 4
    ix, iy, iw, _ = slab(cv, x, y, w, 3, weight=weight, fill=tone(fill), shadow=shadow, pressed=pressed)
    cv.text(ix, iy, caption, ink_on(tone(fill)), tone(fill), True, max_w=iw)
    return w


def badge(cv: Canvas, x: int, y: int, text: str, *, fill: str | RGB = "info", tracking: int = 1) -> int:
    """A single-row filled chip; no border, no shadow — it is a label, not a control."""
    s = " " + display(text, tracking) + " "
    cv.text(x, y, s, ink_on(tone(fill)), tone(fill), True)
    return text_width(s)


def progress(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    frac: float,
    *,
    fill: str | RGB = "ok",
    bg: RGB = PAPER,
    weight: Weight = Weight.HAIRLINE,
) -> None:
    """Solid bar in a hard frame.  No gradient, no rounded cap."""
    b = BORDERS[weight]
    cv.put(x, y, b.l, INK, bg)
    cv.put(x + w - 1, y, b.r, INK, bg)
    span = max(0, w - 2)
    done = max(0, min(span, round(span * max(0.0, min(1.0, frac)))))
    cv.fill_rect(x + 1, y, done, 1, G.BLOCK, tone(fill), bg)
    cv.fill_rect(x + 1 + done, y, span - done, 1, " ", INK, GRID)


def stat(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    *,
    value: str,
    unit: str = "",
    caption: str,
    fill: str | RGB = "plain",
    accent: str | RGB = "accent",
    shadow: tuple[int, int] = SHADOW_MD,
    pressed: bool = False,
) -> int:
    """A hero counter card: big number, small unit, tracked caption.

    Height is fixed at 10 rows — 5 for the numerals, one for the caption, plus
    frame and breathing room.
    """
    h = 10
    ix, iy, iw, ih = slab(cv, x, y, w, h, weight=Weight.SLAB, fill=tone(fill), shadow=shadow, pressed=pressed)
    bw = big_width(value)
    big_text(cv, ix, iy + 1, value, ink_on(tone(fill)), tone(fill))
    if unit:
        uw = max(0, iw - bw - 2)
        cv.text(ix + bw + 2, iy + BIG_H - 1, fit(unit, uw), MUTED if tone(accent) == SURFACE else tone(accent), tone(fill), True, max_w=uw)
    cv.fill_rect(ix, iy + BIG_H + 1, iw, 1, BORDERS[Weight.HAIRLINE].t, INK, tone(fill))
    cv.text(ix, iy + BIG_H + 2, fit(caption, iw), MUTED, tone(fill), False, max_w=iw)
    return h


def ticker(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    items: Sequence[str],
    t: float,
    *,
    fill: str | RGB = "accent",
    speed: float = 6.0,
) -> None:
    """The masthead marquee: uppercase, star-separated, permanently scrolling."""
    strip = anim.marquee([i.upper() for i in items], w, t, speed=speed)
    cv.fill_rect(x, y, w, 1, " ", INK, tone(fill))
    cv.text(x, y, strip, INK, tone(fill), True, max_w=w)


def dropdown(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    items: Sequence[str],
    selected: int,
    *,
    title: str = "select",
    fill: str | RGB = "accent",
    notes: Sequence[str] | None = None,
    shadow: tuple[int, int] = SHADOW_MD,
) -> int:
    """The `/model` picker.  Selection is a full-bleed colour block."""
    h = len(items) + 5
    ix, iy, iw, _ = slab(cv, x, y, w, h, title=title, title_tone=fill, shadow=shadow)
    for i, item in enumerate(items):
        row = iy + i
        chosen = i == selected
        bgc = tone(fill) if chosen else SURFACE
        cv.fill_rect(ix - 1, row, iw + 2, 1, " ", INK, bgc)
        cv.text(ix, row, (G.ARROW + " ") if chosen else "  ", ink_on(bgc), bgc, True)
        note = notes[i] if notes and i < len(notes) else ""
        reserve = text_width(note) + 2 if note else 0
        cv.text(ix + 2, row, fit(item, max(0, iw - 2 - reserve)), ink_on(bgc), bgc, chosen, max_w=iw - 2)
        if note:
            cv.text(ix + iw - text_width(note), row, note, INK if chosen else MUTED, bgc, False)
    return h


def masked_input(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    *,
    caption: str = "api key",
    filled: int = 0,
    cursor: bool = True,
    t: float = 0.0,
    fill: str | RGB = "plain",
) -> int:
    """The `/login` field.  Never renders the secret, only its length."""
    h = 6
    ix, iy, iw, _ = slab(cv, x, y, w, h, title=caption, title_tone="branch", shadow=SHADOW_SM)
    cv.fill_rect(ix, iy, iw, 1, " ", INK, GRID)
    dots = G.DOT * min(filled, max(0, iw - 2))
    cv.text(ix + 1, iy, dots, INK, GRID, True, max_w=iw - 2)
    if cursor and anim.pulse(t, freq=2.4) > 0.5:
        cv.put(ix + 1 + text_width(dots), iy, G.BLOCK, INK, GRID)
    if h >= 7:
        cv.text(ix, iy + 2, label("input hidden"), MUTED, SURFACE, False, max_w=iw)
    return h


# --------------------------------------------------------------------------
# domain views
# --------------------------------------------------------------------------


def diff_view(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    lines: Iterable[tuple[str, str]],
    *,
    title: str = "diff",
) -> None:
    """Added/removed lines as solid colour blocks — readable at a glance."""
    ix, iy, iw, ih = slab(cv, x, y, w, h, title=title, title_tone="branch", shadow=SHADOW_MD)
    for i, (kind, text) in enumerate(lines):
        if i >= ih:
            break
        row = iy + i
        if kind == "+":
            cv.fill_rect(ix - 1, row, iw + 2, 1, " ", INK, MINT)
            cv.text(ix, row, "+ " + text, ink_on(MINT), MINT, True, max_w=iw)
        elif kind == "-":
            cv.fill_rect(ix - 1, row, iw + 2, 1, " ", INK, RED)
            cv.text(ix, row, "- " + text, ink_on(RED), RED, True, max_w=iw)
        else:
            cv.text(ix, row, "  " + text, MUTED, SURFACE, False, max_w=iw)


def branch_tree(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    nodes: Sequence[tuple[int, str, str]],
    *,
    title: str = "branches",
    t: float = 0.0,
) -> None:
    """Speculative branches: depth, label, state in {run, pass, fail, idle}."""
    ix, iy, iw, ih = slab(cv, x, y, w, h, title=title, title_tone="info", shadow=SHADOW_MD)
    marks = {"pass": (G.CHECK, MINT), "fail": (G.CROSS, RED), "idle": (G.DOT_EMPTY, MUTED)}
    for i, (depth, text, state) in enumerate(nodes):
        if i >= ih:
            break
        row = iy + i
        stem = ("  " * depth) + (G.ELBOW if depth else " ") + " "
        cv.text(ix, row, stem, MUTED, SURFACE, False, max_w=iw)
        mark, color = marks.get(state, (anim.radar_spin(t), TONES["info"]))
        col = ix + text_width(stem)
        cv.text(col, row, mark, color, SURFACE, True)
        avail = max(0, iw - (col + 2 - ix))
        cv.text(col + 2, row, fit(text, avail), INK, SURFACE, state == "pass", max_w=avail)


def achievement(cv: Canvas, x: int, y: int, w: int, name: str, detail: str, t: float = 0.0) -> int:
    """Trophy toast.  Pink, loud, and gone in a few seconds."""
    h = 6
    ix, iy, iw, _ = slab(cv, x, y, w, h, fill=PINK, weight=Weight.SLAB, shadow=SHADOW_MD)
    cv.text(ix, iy, anim.star_spin(t) + " " + display("achievement unlocked", 1), ink_on(PINK), PINK, True, max_w=iw)
    cv.text(ix, iy + 1, display(name, 1), ink_on(PINK), PINK, True, max_w=iw)
    cv.text(ix, iy + 2, detail, ink_on(PINK), PINK, False, max_w=iw)
    return h


def status_bar(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    *,
    left: str,
    right: str = "",
    t: float = 0.0,
    busy: bool = False,
    fill: str | RGB = "ink",
    text_color: RGB = PAPER,
) -> None:
    cv.fill_rect(x, y, w, 1, " ", text_color, tone(fill))
    head = (anim.radar_spin(t) if busy else G.ARROW) + " "
    tail = right.upper() if right else ""
    room = max(0, w - 2 - text_width(head) - (text_width(tail) + 3 if tail else 0))
    cv.text(x + 1, y, head + fit(left, room), text_color, tone(fill), True, max_w=w - 2)
    if tail and text_width(tail) + 4 < w:
        cv.text(x + w - 1 - text_width(tail), y, tail, text_color, tone(fill), False)
