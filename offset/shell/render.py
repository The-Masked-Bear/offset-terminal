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


def transcript(width: int, height: int, entries: Sequence[Entry], *, live: str = "", t: float = 0.0) -> str:
    """The conversation, newest at the bottom."""
    rows: list[tuple[str, str]] = []  # (kind, text)
    for entry in entries:
        if entry.type == MESSAGE:
            role = entry.role or "user"
            if role == "user":
                rows.append(("user", entry.text))
            else:
                rows.append(("assistant", entry.text))
        elif entry.type == TOOL_CALL:
            rows.append(("call", f"{entry.data.get('tool', '?')}  {entry.data.get('summary', '')}"))
        elif entry.type == TOOL_RESULT:
            kind = "ok" if entry.data.get("ok", True) else "bad"
            rows.append((kind, str(entry.data.get("summary") or entry.data.get("content", ""))[:400]))
    if live:
        rows.append(("assistant", live))

    body = width - 2
    painted: list[tuple[str, str]] = []
    for kind, text in rows:
        indent = 2 if kind in ("call", "ok", "bad") else 0
        for line in _wrap(text, max(8, body - indent)):
            painted.append((kind, line))
        painted.append(("gap", ""))

    cv = Canvas(width, height, bg=PAPER)
    visible = painted[-height:] if len(painted) > height else painted
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
    return cv.render()


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
