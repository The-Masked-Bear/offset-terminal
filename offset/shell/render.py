"""Drawing the shell.

Every function here returns an ANSI string produced by the same `Canvas` the
design system uses, so the interactive app and the static demo cannot drift
apart.  prompt_toolkit only ever receives finished pixels.
"""

from __future__ import annotations

from typing import Sequence

from offset.core.entries import MESSAGE, TOOL_CALL, TOOL_RESULT, Entry
from offset.shell.commands import Overlay, ShellState
from offset.ui import anim, brutal
from offset.ui.canvas import Canvas
from offset.ui.tokens import (
    CYAN,
    GRID,
    Depth,
    G,
    INK,
    MINT,
    MUTED,
    PAPER,
    PINK,
    RED,
    SURFACE,
    TONES,
    YELLOW,
    Weight,
    display,
    fit,
    label,
    text_width,
)

TICKER = (
    "offset",
    "speculative branching",
    "every tool on",
    "multi-model",
    "zero rounded corners",
)


def _wrap(text: str, width: int) -> list[str]:
    """Greedy wrap; never loses a character, never splits mid-word if it fits."""
    out: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            out.append("")
            continue
        line = ""
        for word in paragraph.split(" "):
            while text_width(word) > width:  # a single unbreakable monster
                room = width - text_width(line) - (1 if line else 0)
                if room <= 2:
                    out.append(line)
                    line = ""
                    room = width
                head, word = word[:room], word[room:]
                out.append((line + (" " if line else "") + head).rstrip())
                line = ""
            if text_width(line) + text_width(word) + (1 if line else 0) <= width:
                line += (" " if line else "") + word
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


def banner(width: int, t: float) -> str:
    cv = Canvas(width, 1, bg=PAPER)
    brutal.ticker(cv, 0, 0, width, TICKER, t, speed=6.0)
    return cv.render()


def _lines_for(entry: Entry, body: int) -> list[tuple[str, str]]:
    """One entry's wrapped display lines, as (kind, line) pairs."""
    if entry.type == MESSAGE:
        kind = "user" if (entry.role or "user") == "user" else "assistant"
        text = entry.text
    elif entry.type == TOOL_CALL:
        kind = "call"
        text = f"{entry.data.get('tool', '?')}  {entry.data.get('summary', '')}"
    elif entry.type == TOOL_RESULT:
        kind = "ok" if entry.data.get("ok", True) else "bad"
        text = str(entry.data.get("summary") or entry.data.get("content", ""))[:400]
    else:
        kind = "assistant"
        text = entry.text or str(entry.data.get("summary") or "")
    indent = 2 if kind in ("call", "ok", "bad") else 0
    out = [(kind, line) for line in _wrap(text, max(8, body - indent))]
    out.append(("gap", ""))
    return out


class Transcript:
    """The conversation as a scrollable buffer.

    Wrapping is cached per (entry, width): the Pi repaints at 12fps and
    re-wrapping the whole history every frame is the one thing here that would
    actually be slow.

    `offset` counts lines hidden BELOW the viewport, so 0 means "at the bottom".
    Scrolling up stops the view following new output - a transcript that yanks
    itself away while you are reading it is worse than no scrollback at all.
    """

    __slots__ = ("offset", "follow", "_cache", "_width", "_seen", "wraps")

    def __init__(self) -> None:
        self.offset = 0
        self.follow = True
        self._cache: dict[str, list[tuple[str, str]]] = {}
        self._width = 0
        #: The line count at the last paint, so appended output can be absorbed
        #: without moving what the reader is looking at.
        self._seen = 0
        #: Counts calls to the wrapper, so a test can prove the cache works.
        self.wraps = 0

    def anchor(self, total: int, height: int) -> int:
        """Keep the same content on screen as lines are appended.

        `offset` is measured from the bottom, so when the bottom moves a fixed
        offset would slide the viewport backwards in time. While scrolled up,
        growth is added to the offset instead; at the bottom we simply follow.
        """
        if self.follow:
            self.offset = 0
        elif total > self._seen:
            self.offset += total - self._seen
        self._seen = total
        self.offset = max(0, min(self.offset, max(0, total - height)))
        return self.offset

    def lines(self, width: int, entries: Sequence[Entry], live: str = "") -> list[tuple[str, str]]:
        body = width - 2
        if width != self._width:
            self._cache.clear()  # every wrap is width-dependent
            self._width = width
        out: list[tuple[str, str]] = []
        for entry in entries:
            cached = self._cache.get(entry.id)
            if cached is None:
                self.wraps += 1
                cached = _lines_for(entry, body)
                self._cache[entry.id] = cached
            out.extend(cached)
        if live:
            # Deliberately uncached: it changes on every frame by definition.
            out.extend([("assistant", line) for line in _wrap(live, max(8, body))])
        return out

    # -- movement ---------------------------------------------------------

    def scroll(self, delta: int, total: int = 0, height: int = 0) -> None:
        """Positive scrolls up (back in time)."""
        limit = max(0, total - height)
        self.offset = max(0, min(self.offset + delta, limit))
        self.follow = self.offset == 0

    def page(self, direction: int, height: int, total: int = 0) -> None:
        self.scroll(direction * max(1, height - 1), total, height)

    def to_end(self) -> None:
        self.offset = 0
        self.follow = True

    @property
    def at_end(self) -> bool:
        return self.offset == 0


def transcript(
    width: int,
    height: int,
    entries: Sequence[Entry],
    *,
    live: str = "",
    t: float = 0.0,
    view: Transcript | None = None,
) -> str:
    """The conversation, newest at the bottom.

    Without a `view` this is the tail, which is what the tests and the demo
    want; with one it is scrollable and shows how much is hidden.
    """
    body = width - 2
    if view is not None:
        painted = view.lines(width, entries, live)
    else:
        painted = []
        for entry in entries:
            painted.extend(_lines_for(entry, body))
        if live:
            painted.extend([("assistant", line) for line in _wrap(live, max(8, body))])

    total = len(painted)
    offset = view.anchor(total, height) if view is not None else 0
    end = total - offset
    start = max(0, end - height)
    visible = painted[start:end]

    cv = Canvas(width, height, bg=PAPER)
    for y, (kind, line) in enumerate(visible):
        if kind == "user":
            cv.fill_rect(0, y, width, 1, " ", INK, YELLOW)
            cv.text(1, y, line, INK, YELLOW, True, max_w=body)
        elif kind == "assistant":
            cv.text(1, y, line, INK, PAPER, False, max_w=body)
        elif kind == "call":
            cv.put(1, y, G.ARROW, CYAN, PAPER)
            cv.text(3, y, line, MUTED, PAPER, False, max_w=body - 2)
        elif kind == "ok":
            cv.put(1, y, G.HALF_LEFT, MINT, PAPER)
            cv.text(3, y, line, MUTED, PAPER, False, max_w=body - 2)
        elif kind == "bad":
            cv.put(1, y, G.HALF_LEFT, RED, PAPER)
            cv.text(3, y, line, RED, PAPER, False, max_w=body - 2)

    if offset > 0 and height > 0:
        cv.fill_rect(0, height - 1, width, 1, " ", INK, CYAN)
        cv.text(1, height - 1, fit(f"{offset} more below  \u2193 to follow", width - 2), INK, CYAN, True)
    _scrollbar(cv, width, height, total, start)
    return cv.render()


def _scrollbar(cv: Canvas, width: int, height: int, total: int, start: int) -> None:
    """A hard block thumb in the last column.  No gradient, no rounded cap."""
    if total <= height or height < 3 or width < 3:
        return
    span = max(1, round(height * height / total))
    top = round(start / total * height)
    top = min(top, height - span)
    for y in range(height):
        inside = top <= y < top + span
        cv.put(width - 1, y, G.BLOCK if inside else " ", INK if inside else GRID, PAPER if inside else GRID)


def prompt_row(width: int, text: str, *, busy: bool, t: float) -> str:
    """The input line's frame; prompt_toolkit draws the editable text itself."""
    cv = Canvas(width, 1, bg=PAPER)
    marker = anim.radar_spin(t) if busy else G.ARROW
    cv.text(0, 0, f" {marker} ", MUTED if busy else INK, PAPER, True)
    return cv.render()


def status(width: int, state: ShellState, *, busy: bool, t: float, note: str = "") -> str:
    cv = Canvas(width, 1, bg=PAPER)
    found, total = state.eggs.progress()
    left = note or f"{state.model}  {G.STAR}  {len(state.toolbox)} tools  {G.STAR}  {state.approval.mode}"
    right = f"{found}/{total} eggs  {G.STAR}  ctrl-c stop  ctrl-d quit"
    brutal.status_bar(cv, 0, 0, width, left=left, right=right, t=t)
    return cv.render()


def overlay(width: int, height: int, panel: Overlay, t: float) -> str:
    """Modal panels: the model picker, masked login, the tree, the trophies."""
    cv = Canvas(width, height, bg=PAPER)
    if panel.kind == "login":
        brutal.masked_input(cv, 0, 0, width, caption=panel.title, filled=len(panel.buffer), t=t)
        return cv.render()

    if panel.kind == "approve":
        # Red frame, because the question is always "may I do something that
        # changes your machine".
        rows = [line for line in panel.items if line != ""]
        height = len(rows) + 4
        x, y, iw, _ = brutal.slab(cv, 0, 0, width, height, fill=RED, weight=Weight.SLAB)
        cv.text(x, y, display(panel.title, 1), INK, RED, True, max_w=iw)
        for i, line in enumerate(rows):
            cv.text(x, y + 2 + i, line, INK, RED, i == len(rows) - 1, max_w=iw)
        return cv.render()

    rows = max(1, height - 5)
    start = max(0, min(panel.selected - rows // 2, len(panel.items) - rows))
    window = panel.items[start : start + rows]
    notes = panel.notes[start : start + rows] if panel.notes else None
    brutal.dropdown(
        cv, 0, 0, width, window, panel.selected - start,
        title=panel.title, notes=notes,
        fill="accent" if panel.kind == "model" else ("info" if panel.kind == "tree" else "branch"),
    )
    return cv.render()


def reveal_panel(width: int, egg, t: float) -> str:
    """An easter egg, drawn in a slab that owes nothing to the layout."""
    lines = egg.frames[int(t * 12) % len(egg.frames)] if egg.frames else egg.lines
    height = len(lines) + (4 if egg.title else 3)
    cv = Canvas(width, height, bg=PAPER)
    tone = egg.tone if egg.tone in TONES else "branch"
    inner = brutal.slab(cv, 0, 0, width, height, fill=TONES.get(tone, PINK), weight=Weight.SLAB)
    x, y, iw, _ = inner
    if egg.title:
        cv.text(x, y, display(egg.title, 1), INK, TONES.get(tone, PINK), True, max_w=iw)
        y += 1
    for i, line in enumerate(lines):
        cv.text(x, y + i, line, INK, TONES.get(tone, PINK), False, max_w=iw)
    return cv.render()


def message_block(width: int, lines: Sequence[str], tone: str) -> str:
    """Command output: a tinted left rule, no box, so it reads as chrome."""
    colour = TONES.get(tone, SURFACE)
    cv = Canvas(width, max(1, len(lines)), bg=PAPER)
    for y, line in enumerate(lines):
        cv.put(0, y, G.HALF_LEFT, colour if tone != "plain" else MUTED, PAPER)
        cv.text(2, y, line, INK if tone != "muted" else MUTED, PAPER, False, max_w=width - 2)
    return cv.render()
