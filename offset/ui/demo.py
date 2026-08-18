"""Live showcase of the design system.

`python -m offset demo` runs it interactively; `--once` prints a single frame,
which is what makes the aesthetic testable in CI and in a piped capture.
"""

from __future__ import annotations

import os
import select
import shutil
import sys
import time
from dataclasses import dataclass, field

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
    YELLOW,
    detect_depth,
    display,
    fit,
    label,
)

TICKER = (
    "offset v0.1.0",
    "speculative branching",
    "multi-model by default",
    "every tool on",
    "zero rounded corners",
)

MODELS = ("claude opus 5", "gpt-5.2 codex", "gemini 3 pro", "deepseek v4", "qwen3 coder 480b", "local / llama.cpp")
MODEL_NOTES = ("planner", "implementer", "critic", "cheap", "bulk", "offline")

BRANCHES = (
    (0, "main", "idle"),
    (1, "a / rewrite parser", "pass"),
    (1, "b / patch in place", "fail"),
    (1, "c / defer to runtime", "run"),
    (2, "c1 / cache the scan", "pass"),
)

DIFF = (
    ("-", "def parse(src): return eval(src)"),
    ("+", "def parse(src):"),
    ("+", "    return _GRAMMAR.match(src)"),
    (" ", ""),
    ("-", "TIMEOUT = None"),
    ("+", "TIMEOUT = 30.0"),
)


@dataclass
class State:
    selected: int = 0
    pressed: int = -1
    pressed_at: float = -10.0
    secret_len: int = 7
    trophy_at: float = -10.0
    glitch_at: float = -10.0
    started: float = field(default_factory=time.monotonic)


def _columns(total: int, n: int, gutter: int) -> list[int]:
    """Widths for `n` equal columns, distributing the remainder left to right."""
    span = total - gutter * (n - 1)
    base, extra = divmod(span, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def compose(w: int, h: int, t: float, st: State) -> Canvas:
    cv = Canvas(w, h, fg=INK, bg=PAPER)
    m = 2
    inner = w - 2 * m

    # masthead ------------------------------------------------------------
    brutal.ticker(cv, 0, 0, w, TICKER, t, speed=7.0)

    # hero ----------------------------------------------------------------
    hero = "build four answers."
    if t - st.glitch_at < 0.6:
        hero = anim.glitch(hero, t, intensity=0.35)
    else:
        hero = anim.glitch_burst(hero, t, period=9.0)
    brutal.heading(cv, m, 2, hero, tracking=2, underline="accent", width=inner)
    cv.text(m, 4, display("keep the one that passes.", 2), INK, PAPER, True, max_w=inner)
    cv.text(m, 5, label("no wizards. no gradients. no rounded corners."), MUTED, PAPER, False, max_w=inner)

    y = 7

    # counters ------------------------------------------------------------
    ncards = 4 if inner >= 88 else 2
    gut = 6
    widths = _columns(inner, ncards, gut)
    cards = (
        ("6", "models", "live now"),
        ("4", "branches", "in flight"),
        ("42", "tools", "all armed"),
        ("61", "eggs", "hidden"),
    )[:ncards]
    x = m
    for i, (val, unit, cap) in enumerate(cards):
        shown = str(int(anim.count_up(float(val), t, dur=1.6, delay=0.12 * i)))
        brutal.stat(
            cv, x, y, widths[i],
            value=shown, unit=unit, caption=cap,
            accent=(YELLOW, PINK, CYAN, MINT)[i % 4],
        )
        x += widths[i] + gut
    y += 12

    # panels --------------------------------------------------------------
    if inner >= 96:
        pw = _columns(inner, 3, gut)
        ph = 11
        brutal.branch_tree(cv, m, y, pw[0], ph, BRANCHES, title="branches", t=t)
        brutal.dropdown(cv, m + pw[0] + gut, y, pw[1], MODELS, st.selected, title="model", notes=MODEL_NOTES)
        brutal.diff_view(cv, m + pw[0] + pw[1] + 2 * gut, y, pw[2], ph, DIFF, title="winner diff")
        y += ph + 3
    else:
        pw = _columns(inner, 2, gut)
        brutal.branch_tree(cv, m, y, pw[0], 11, BRANCHES, title="branches", t=t)
        brutal.dropdown(cv, m + pw[0] + gut, y, pw[1], MODELS, st.selected, title="model", notes=MODEL_NOTES)
        y += 14

    # controls ------------------------------------------------------------
    if y + 4 < h:
        bx = m
        for i, (cap, fill) in enumerate((("run all", "accent"), ("keep b", "ok"), ("kill", "err"), ("fork", "branch"))):
            down = st.pressed == i and (t - st.pressed_at) < 0.16
            bx += brutal.button(cv, bx, y, cap, fill=fill, pressed=down) + 5
            if bx > w - 12:
                break
        y += 4

    # meters + secret -----------------------------------------------------
    if y + 2 < h:
        frac = min(1.0, (t % 6.0) / 4.2)
        cv.text(m, y, fit("branch c", 11), MUTED, PAPER, False, max_w=11)
        brutal.progress(cv, m + 12, y, min(34, inner - 14), frac, fill="info")
        pct = f"{int(frac * 100):3d}%"
        cv.text(m + 12 + min(34, inner - 14) + 2, y, pct, INK, PAPER, True)
        bx = m + 12 + min(34, inner - 14) + 8
        if bx + 30 < w:
            bx += brutal.badge(cv, bx, y, "streaming", fill="info") + 2
            bx += brutal.badge(cv, bx, y, anim.dot_bounce(t), fill="ok") + 2
            brutal.badge(cv, bx, y, anim.star_spin(t) + " live", fill="branch")
        y += 2

    # trophy --------------------------------------------------------------
    if 0.0 <= t - st.trophy_at < 4.0 and y + 7 < h:
        brutal.achievement(cv, m, y, min(52, inner), "first blood", "a branch passed before you finished reading.", t)

    # status --------------------------------------------------------------
    brutal.status_bar(
        cv, 0, h - 1, w,
        left=f"offset  {G.STAR}  6 models  {G.STAR}  branch c running",
        right="q quit  space press  a trophy  g glitch",
        t=t,
    )
    return cv


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------


def frame(t: float, *, w: int | None = None, h: int | None = None, st: State | None = None, depth: Depth | None = None) -> str:
    size = shutil.get_terminal_size((100, 40))
    return compose(w or size.columns, h or size.lines, t, st or State()).render(depth)


def _read_keys(timeout: float) -> list[str]:
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return []
    data = os.read(sys.stdin.fileno(), 64).decode("utf-8", "replace")
    keys, i = [], 0
    while i < len(data):
        if data.startswith("\x1b[", i) and i + 2 < len(data):
            keys.append({"A": "up", "B": "down", "C": "right", "D": "left"}.get(data[i + 2], "esc"))
            i += 3
        elif data[i] == "\x1b":
            keys.append("esc")
            i += 1
        else:
            keys.append(data[i])
            i += 1
    return keys


def run(fps: float = 24.0) -> int:
    """Interactive loop.  Restores the terminal on every exit path."""
    import termios
    import tty

    st = State()
    depth = detect_depth()
    fd = sys.stdin.fileno()
    interactive = sys.stdin.isatty()
    saved = termios.tcgetattr(fd) if interactive else None
    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l")
    last_size = (0, 0)
    try:
        if interactive:
            tty.setraw(fd)
        t0 = time.monotonic()
        period = 1.0 / fps
        while True:
            t = time.monotonic() - t0
            size = shutil.get_terminal_size((100, 40))
            if (size.columns, size.lines) != last_size:
                out.write("\x1b[2J")
                last_size = (size.columns, size.lines)
            out.write(compose(size.columns, size.lines, t, st).render(depth, home=True))
            out.flush()
            for key in _read_keys(period) if interactive else []:
                if key in ("q", "esc", "\x03"):
                    return 0
                if key == "up":
                    st.selected = (st.selected - 1) % len(MODELS)
                elif key == "down":
                    st.selected = (st.selected + 1) % len(MODELS)
                elif key == " ":
                    st.pressed = (st.pressed + 1) % 4
                    st.pressed_at = t
                elif key == "a":
                    st.trophy_at = t
                elif key == "g":
                    st.glitch_at = t
            if not interactive:
                time.sleep(period)
                if t > 3.0:
                    return 0
    finally:
        if saved is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        out.write("\x1b[?25h\x1b[?1049l")
        out.flush()
    return 0
