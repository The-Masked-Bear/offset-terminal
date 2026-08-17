"""A character-cell canvas that renders to ANSI.

Flat parallel buffers, one entry per cell.  Everything clips silently at the
edges, so components never have to bounds-check their own geometry.

Style runs are coalesced at render time: an escape sequence is emitted only
when the pen actually changes, which keeps a full 100x34 repaint around a few
kilobytes instead of tens.
"""

from __future__ import annotations

from typing import Iterator

from offset.ui.tokens import INK, PAPER, RGB, Depth, bg_sgr, char_width, detect_depth, fg_sgr

_RESET = "\x1b[0m"
#: Written into the cell after a double-width character so the grid stays true.
_CONT = ""


class Canvas:
    __slots__ = ("w", "h", "_ch", "_fg", "_bg", "_bold")

    def __init__(self, w: int, h: int, *, fill: str = " ", fg: RGB = INK, bg: RGB = PAPER) -> None:
        if w < 0 or h < 0:
            raise ValueError("canvas dimensions must be non-negative")
        self.w = w
        self.h = h
        n = w * h
        self._ch: list[str] = [fill] * n
        self._fg: list[RGB] = [fg] * n
        self._bg: list[RGB] = [bg] * n
        self._bold = bytearray(n)

    # -- drawing ----------------------------------------------------------

    def put(
        self,
        x: int,
        y: int,
        ch: str,
        fg: RGB | None = None,
        bg: RGB | None = None,
        bold: bool = False,
    ) -> None:
        if not (0 <= x < self.w and 0 <= y < self.h):
            return
        i = y * self.w + x
        self._ch[i] = ch
        if fg is not None:
            self._fg[i] = fg
        if bg is not None:
            self._bg[i] = bg
        self._bold[i] = 1 if bold else 0

    def text(
        self,
        x: int,
        y: int,
        s: str,
        fg: RGB | None = None,
        bg: RGB | None = None,
        bold: bool = False,
        *,
        max_w: int | None = None,
    ) -> int:
        """Draw `s` at (x, y).  Returns the number of columns advanced."""
        if not (0 <= y < self.h) or not s:
            return 0
        limit = self.w if max_w is None else min(self.w, x + max_w)
        col = x
        row = y * self.w
        for ch in s:
            if col >= limit:
                break
            cw = char_width(ch)
            if cw == 0:
                continue
            if col >= 0:
                i = row + col
                self._ch[i] = ch
                if fg is not None:
                    self._fg[i] = fg
                if bg is not None:
                    self._bg[i] = bg
                self._bold[i] = 1 if bold else 0
                if cw == 2 and col + 1 < limit:
                    j = i + 1
                    self._ch[j] = _CONT
                    if bg is not None:
                        self._bg[j] = bg
            col += cw
        return col - x

    def fill_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        ch: str = " ",
        fg: RGB | None = None,
        bg: RGB | None = None,
        bold: bool = False,
    ) -> None:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.w, x + w), min(self.h, y + h)
        if x0 >= x1 or y0 >= y1:
            return
        flag = 1 if bold else 0
        for yy in range(y0, y1):
            base = yy * self.w
            for i in range(base + x0, base + x1):
                self._ch[i] = ch
                if fg is not None:
                    self._fg[i] = fg
                if bg is not None:
                    self._bg[i] = bg
                self._bold[i] = flag

    def hline(self, x: int, y: int, w: int, ch: str, fg: RGB | None = None, bg: RGB | None = None) -> None:
        self.fill_rect(x, y, w, 1, ch, fg, bg)

    def vline(self, x: int, y: int, h: int, ch: str, fg: RGB | None = None, bg: RGB | None = None) -> None:
        self.fill_rect(x, y, 1, h, ch, fg, bg)

    def blit(self, other: "Canvas", x: int, y: int) -> None:
        """Copy `other` onto this canvas at (x, y), clipping at the edges."""
        for yy in range(other.h):
            ty = y + yy
            if not (0 <= ty < self.h):
                continue
            src = yy * other.w
            dst = ty * self.w
            for xx in range(other.w):
                tx = x + xx
                if not (0 <= tx < self.w):
                    continue
                i, j = src + xx, dst + tx
                self._ch[j] = other._ch[i]
                self._fg[j] = other._fg[i]
                self._bg[j] = other._bg[i]
                self._bold[j] = other._bold[i]

    # -- output -----------------------------------------------------------

    def rows(self, depth: Depth) -> Iterator[str]:
        w, ch, fgs, bgs, bold = self.w, self._ch, self._fg, self._bg, self._bold
        plain = depth is Depth.NONE
        for y in range(self.h):
            base = y * w
            if plain:
                yield "".join(ch[base : base + w])
                continue
            out: list[str] = []
            cur_fg: RGB | None = None
            cur_bg: RGB | None = None
            cur_bold = 0
            for i in range(base, base + w):
                c = ch[i]
                if c == _CONT:
                    continue
                f, b, bo = fgs[i], bgs[i], bold[i]
                if f != cur_fg or b != cur_bg or bo != cur_bold:
                    parts = []
                    if bo != cur_bold:
                        parts.append("1" if bo else "22")
                    if f != cur_fg:
                        parts.append(fg_sgr(f, depth))
                    if b != cur_bg:
                        parts.append(bg_sgr(b, depth))
                    out.append("\x1b[" + ";".join(parts) + "m")
                    cur_fg, cur_bg, cur_bold = f, b, bo
            
                out.append(c)
            out.append(_RESET)
            yield "".join(out)

    def render(self, depth: Depth | None = None, *, home: bool = False) -> str:
        depth = detect_depth() if depth is None else depth
        body = "\n".join(self.rows(depth))
        return ("\x1b[H" + body) if home else body

    def __str__(self) -> str:
        return self.render(Depth.NONE)
