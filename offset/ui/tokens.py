"""Design tokens — the DNA layer.

Every value here is transcribed from the reference implementation
(the-masked-bear.github.io/wifisense-pi) and re-expressed for a character
cell grid.  Two rules govern every translation:

  1. Shadows are HARD.  Zero blur, one flat colour, 45-degree offset.
  2. Nothing is rounded.  Corners are corners.

The web original works in CSS pixels on a square grid.  A terminal cell is
roughly 1:2 (w:h), so a 45-degree shadow needs twice as many columns as rows.
`shadow_cells()` is the ONLY place that conversion is allowed to happen.
"""

from __future__ import annotations

import os
import sys
import unicodedata
from enum import IntEnum
from typing import Final, NamedTuple

# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------


class RGB(NamedTuple):
    r: int
    g: int
    b: int


def _hex(value: int) -> RGB:
    return RGB((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


def blend(over: RGB, under: RGB, alpha: float) -> RGB:
    """Composite `over` onto `under` at `alpha` opacity."""
    inv = 1.0 - alpha
    return RGB(
        round(over.r * alpha + under.r * inv),
        round(over.g * alpha + under.g * inv),
        round(over.b * alpha + under.b * inv),
    )


# Palette — exact values from the reference stylesheet.
INK: Final = _hex(0x111111)  # --black
PAPER: Final = _hex(0xF4F4F0)  # --bg
SURFACE: Final = _hex(0xFFFFFF)  # --surface
MUTED: Final = _hex(0x555555)  # --text-secondary
YELLOW: Final = _hex(0xFFDE59)  # --yellow
PINK: Final = _hex(0xFF90E8)  # --pink
CYAN: Final = _hex(0x8CFFFB)  # --blue
MINT: Final = _hex(0xB2FF9E)  # --mint
RED: Final = _hex(0xFF5A5F)  # --red
GRID: Final = blend(INK, PAPER, 0.06)  # --grid-color over --bg

#: Semantic roles.  Accents are all light, so the ink on top is always INK —
#: that black-on-bright contrast IS the style; never invert it.
TONES: Final[dict[str, RGB]] = {
    "plain": SURFACE,
    "accent": YELLOW,  # primary action, focus
    "branch": PINK,  # speculative branches, forks
    "info": CYAN,  # models, streaming, telemetry
    "ok": MINT,  # success, diff additions
    "err": RED,  # failure, diff deletions
    "ink": INK,
    "paper": PAPER,
    "muted": MUTED,
    "grid": GRID,
}


class Depth(IntEnum):
    """Colour capability, in bits."""

    NONE = 0
    ANSI16 = 4
    ANSI256 = 8
    TRUE = 24


def detect_depth(stream=None) -> Depth:
    """Resolve colour depth from the environment.

    `OFFSET_COLOR` / `FORCE_COLOR` win over TTY detection so that piped
    captures (screenshots, tests, CI logs) keep their colour.
    """
    if os.environ.get("NO_COLOR"):
        return Depth.NONE
    forced = os.environ.get("OFFSET_COLOR") or os.environ.get("FORCE_COLOR")
    term = os.environ.get("TERM", "")
    if forced:
        return {"0": Depth.NONE, "4": Depth.ANSI16, "8": Depth.ANSI256}.get(forced, Depth.TRUE)
    if term == "dumb":
        return Depth.NONE
    stream = stream or sys.stdout
    if not hasattr(stream, "isatty") or not stream.isatty():
        return Depth.NONE
    if os.environ.get("COLORTERM", "") in ("truecolor", "24bit"):
        return Depth.TRUE
    if "256" in term:
        return Depth.ANSI256
    return Depth.ANSI16 if term else Depth.NONE


_CUBE: Final = (0, 95, 135, 175, 215, 255)
_ANSI16_RGB: Final = (
    (0, 0, 0), (205, 0, 0), (0, 205, 0), (205, 205, 0),
    (0, 0, 238), (205, 0, 205), (0, 205, 205), (229, 229, 229),
    (127, 127, 127), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (92, 92, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
)


def _cube_index(v: int) -> int:
    best, bd = 0, 1 << 30
    for i, c in enumerate(_CUBE):
        d = (c - v) * (c - v)
        if d < bd:
            best, bd = i, d
    return best


def to_256(c: RGB) -> int:
    r, g, b = c
    if abs(r - g) < 8 and abs(g - b) < 8:
        if r < 8:
            return 16
        if r > 248:
            return 231
        return 232 + (r - 8) * 24 // 247
    return 16 + 36 * _cube_index(r) + 6 * _cube_index(g) + _cube_index(b)


def to_16(c: RGB) -> int:
    best, bd = 0, 1 << 30
    for i, (r, g, b) in enumerate(_ANSI16_RGB):
        d = (r - c.r) ** 2 + (g - c.g) ** 2 + (b - c.b) ** 2
        if d < bd:
            best, bd = i, d
    return best


def fg_sgr(c: RGB, depth: Depth) -> str:
    if depth is Depth.TRUE:
        return f"38;2;{c.r};{c.g};{c.b}"
    if depth is Depth.ANSI256:
        return f"38;5;{to_256(c)}"
    i = to_16(c)
    return str(30 + i if i < 8 else 90 + i - 8)


def bg_sgr(c: RGB, depth: Depth) -> str:
    if depth is Depth.TRUE:
        return f"48;2;{c.r};{c.g};{c.b}"
    if depth is Depth.ANSI256:
        return f"48;5;{to_256(c)}"
    i = to_16(c)
    return str(40 + i if i < 8 else 100 + i - 8)


# --------------------------------------------------------------------------
# glyphs
# --------------------------------------------------------------------------

#: Terminals that cannot draw box characters get a coarser but structurally
#: identical frame.  The layout maths never changes; only the glyphs do.
ASCII: Final = bool(os.environ.get("OFFSET_ASCII")) or "UTF" not in (
    os.environ.get("LC_ALL") or os.environ.get("LC_CTYPE") or os.environ.get("LANG") or "UTF-8"
).upper()


class Weight(IntEnum):
    """Border weight.  Values are the reference stylesheet's CSS pixels."""

    HAIRLINE = 2  # border: 2px solid var(--black)
    BOLD = 3  # 3px — the workhorse
    DOUBLE = 4  # 4px
    SLAB = 5  # 5px — hero panels


class Border(NamedTuple):
    tl: str
    t: str
    tr: str
    l: str
    r: str
    bl: str
    b: str
    br: str


_UNICODE_BORDERS: Final[dict[Weight, Border]] = {
    Weight.HAIRLINE: Border("\u250c", "\u2500", "\u2510", "\u2502", "\u2502", "\u2514", "\u2500", "\u2518"),
    Weight.BOLD: Border("\u250f", "\u2501", "\u2513", "\u2503", "\u2503", "\u2517", "\u2501", "\u251b"),
    Weight.DOUBLE: Border("\u2554", "\u2550", "\u2557", "\u2551", "\u2551", "\u255a", "\u2550", "\u255d"),
    Weight.SLAB: Border("\u259b", "\u2580", "\u259c", "\u258c", "\u2590", "\u2599", "\u2584", "\u259f"),
}
_ASCII_BORDER: Final = Border("+", "-", "+", "|", "|", "+", "-", "+")

BORDERS: Final[dict[Weight, Border]] = (
    {w: _ASCII_BORDER for w in Weight} if ASCII else _UNICODE_BORDERS
)


class G:
    """Glyph constants.  All are single-width; see `char_width`."""

    BLOCK = "#" if ASCII else "\u2588"
    HALF_LEFT = "|" if ASCII else "\u258c"
    STAR = "*" if ASCII else "\u2605"
    DOT = "o" if ASCII else "\u25cf"
    DOT_EMPTY = "." if ASCII else "\u25cb"
    ARROW = ">" if ASCII else "\u25b6"
    CHECK = "x" if ASCII else "\u2713"
    CROSS = "!" if ASCII else "\u2717"
    BRANCH = "|" if ASCII else "\u2502"
    TEE = "+" if ASCII else "\u251c"
    ELBOW = "\\" if ASCII else "\u2514"
    SHADOW = "#" if ASCII else "\u2588"
    SPINNER = ("|", "/", "-", "\\") if ASCII else ("\u25dc", "\u25dd", "\u25de", "\u25df")
    STARS = ("*", "+", "x", "+") if ASCII else ("\u2726", "\u2727", "\u2736", "\u2737")
    GLITCH_POOL = "!<>-_\\/[]{}=+*^?#" if ASCII else "!<>-_\\/[]{}=+*^?#\u2591\u2592\u2593\u2588"


def char_width(ch: str) -> int:
    """1 for everything we ship; 2 for wide characters a user might paste."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def text_width(s: str) -> int:
    return sum(char_width(c) for c in s)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

#: `transform: translate(4px, 4px)` on press, quantised to cells.  Two columns
#: per row keeps the movement on the same 45-degree diagonal as the shadow.
PRESS: Final = (2, 1)


def shadow_cells(px: int) -> tuple[int, int]:
    """Map a CSS shadow offset to (columns, rows).

    The reference uses 4..23px.  A cell is ~2:1, so columns are doubled to
    hold the 45-degree angle.  Three visual steps cover the whole range.
    """
    if px <= 0:
        return (0, 0)
    if px <= 6:
        return (2, 1)
    if px <= 12:
        return (4, 2)
    return (6, 3)


SHADOW_SM: Final = shadow_cells(4)
SHADOW_MD: Final = shadow_cells(10)
SHADOW_LG: Final = shadow_cells(20)


# --------------------------------------------------------------------------
# typography
# --------------------------------------------------------------------------


def track(text: str, spacing: int = 1) -> str:
    """`letter-spacing`, monospace edition: pad between characters."""
    if spacing <= 0:
        return text
    pad = " " * spacing
    return pad.join(text)


def display(text: str, spacing: int = 2) -> str:
    """Archivo Black, 900, uppercase, wide tracking — the headline treatment."""
    return track(text.upper(), spacing)


def label(text: str) -> str:
    """A small caps label: uppercase, NO tracking.

    Letter-spacing belongs to headlines. The reference site tracks a 40px hero,
    not its body copy, and applying it to running UI text turned a status bar
    into "M O C K  1 5  T O O L S" - technically on-brand, actually unreadable.
    Tracking now has to be asked for, via `display()` or `fit(spacing=...)`.
    """
    return text.upper()


def fit(text: str, width: int, *, spacing: int = 0, upper: bool = True) -> str:
    """Tracked, uppercase text guaranteed to fit `width`.

    Tracking is the first thing sacrificed: the reference's wide letter
    spacing is a luxury, legibility is not.  Only once spacing has been spent
    down to zero does the string actually get cut.
    """
    if width <= 0:
        return ""
    s = text.upper() if upper else text
    for sp in range(spacing, -1, -1):
        out = track(s, sp)
        if text_width(out) <= width:
            return out
    cut = ".." if ASCII else "\u2026"
    if width <= len(cut):
        return s[:width]
    return s[: width - len(cut)] + cut
