"""The startup question: how much of this machine may offset touch?

This screen exists because the honest version of "full access" is frightening
when written down, and writing it down is the point.  Three rules it enforces:

  * the safe answer is the one you get by doing the obvious thing — Enter, or
    escape, or any misplaced keypress, all keep offset inside the workspace;
  * full access needs its own key, so it can never be reached by a user
    hammering Enter through a startup screen they did not read;
  * the copy names the actual consequences (any file, any command, launching
    apps, documents anywhere) instead of the word "permissions".
"""

from __future__ import annotations

import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from offset.core.permissions import Grant, Scope, grant
from offset.ui import brutal
from offset.ui.canvas import Canvas
from offset.ui.tokens import (
    CYAN,
    G,
    INK,
    MUTED,
    PAPER,
    RED,
    SHADOW_MD,
    SURFACE,
    Weight,
    display,
    fit,
    label,
    text_width,
)

HEADLINE: Final = "how much of this machine?"
STANDFIRST: Final = "offset runs tools for you. pick the blast radius before it starts."

WORKSPACE_TITLE: Final = "workspace only"
FULL_TITLE: Final = "full system access"

WORKSPACE_MEANS: Final = (
    "reads and writes files inside this folder",
    "runs commands here, and asks before the dangerous ones",
    "a path outside the folder is refused, not asked about",
)

#: The whole reason this module exists: say it plainly, in the user's terms.
FULL_MEANS: Final = (
    "read and write any file on this machine, not just this folder",
    "run arbitrary shell commands as you, with your privileges",
    "launch applications and open things on your desktop",
    "write documents anywhere, including outside the workspace",
)

SMALL_PRINT: Final = (
    "full access is remembered for this workspace only, never for others.",
    "revoke it any time with /permissions revoke.",
)

WORKSPACE_KEY: Final = "enter"
FULL_KEY: Final = "f"

#: Keys that mean "keep it in the workspace".  Escape and Enter are here on
#: purpose: refusing to choose must land on the safe answer.
_SAFE_KEYS: Final = frozenset({"enter", "\r", "\n", "w", "escape", "\x1b", "space", " ", "q"})
#: Only this one grants the machine.  Not "y", not Enter — a letter you have
#: to mean.
_FULL_KEYS: Final = frozenset({"f"})

TICKER: Final = (
    "permission required",
    "nothing runs until you answer",
    "workspace only is the default",
)


def decide(key: str) -> Scope | None:
    """Map a keypress to a choice.  Enter alone can never mean "full"."""
    if not key:
        return None
    name = key.lower()
    if name in _FULL_KEYS:
        return "full"
    if name in _SAFE_KEYS:
        return "workspace"
    return None


@dataclass(slots=True)
class Consent:
    """The state of the question: which workspace, and what was answered."""

    workspace: Path
    choice: Scope | None = None
    at: float = 0.0
    #: Which button is mid-press, for the 4px-translate animation.
    pressed: str = ""
    pressed_at: float = -10.0

    @property
    def answered(self) -> bool:
        return self.choice is not None

    @property
    def full(self) -> bool:
        return self.choice == "full"

    def press(self, key: str, t: float = 0.0) -> Scope | None:
        """Apply a keypress.  Returns the choice it made, or None if the key
        meant nothing — an unknown key must not resolve the screen."""
        got = decide(key)
        if got is None:
            return None
        self.choice = got
        self.at = time.time()
        self.pressed = got
        self.pressed_at = t
        return got

    def commit(self) -> Grant | None:
        """Persist the answer.  None while unanswered, so a caller cannot
        record a decision the user never made."""
        if self.choice is None:
            return None
        return grant(self.choice, self.workspace)


def copy_text() -> str:
    """The consent copy as plain prose — for tests, logs, and `/permissions`."""
    parts = [HEADLINE, STANDFIRST, WORKSPACE_TITLE, *WORKSPACE_MEANS, FULL_TITLE, *FULL_MEANS, *SMALL_PRINT]
    return "\n".join(parts)


def permission_badge(scope: Scope | None) -> str:
    """One line for the status bar.  Untracked: `status_bar` tracks it."""
    if scope == "full":
        return f"{G.CROSS} full system access"
    if scope == "workspace":
        return f"{G.CHECK} workspace only"
    return f"{G.DOT_EMPTY} permission not set"


def _bullets(cv: Canvas, x: int, y: int, w: int, lines: tuple[str, ...], bg, limit: int) -> int:
    """Wrapped bullet copy.  Wrapping at half width leaves room for tracking,
    which `fit` then spends down if the terminal is narrower still."""
    row = y
    budget = max(8, w // 2)
    for line in lines:
        for n, piece in enumerate(textwrap.wrap(line, budget) or [""]):
            if row >= y + limit:
                return row - y
            # Only the first row of a wrapped item carries the dot; a dot on
            # every wrapped row turned one sentence into three bullet points.
            if n == 0:
                cv.put(x, row, G.DOT, INK, bg)
            cv.text(x + 2, row, fit(piece, max(0, w - 2)), INK, bg, False, max_w=max(0, w - 2))
            row += 1
    return row - y


def _option(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    title: str,
    key: str,
    lines: tuple[str, ...],
    tone: str,
    pressed: bool,
    default: bool,
) -> None:
    ix, iy, iw, ih = brutal.slab(
        cv, x, y, w, h,
        weight=Weight.SLAB, fill=SURFACE, shadow=SHADOW_MD,
        title=title, title_tone=tone, tracking=1, pressed=pressed,
    )
    if iw <= 0 or ih <= 0:
        return
    cap = f"press {key}" + ("  (default)" if default else "")
    cv.text(ix, iy, fit(cap, iw), INK if default else MUTED, SURFACE, True, max_w=iw)
    _bullets(cv, ix, iy + 2, iw, lines, SURFACE, max(0, ih - 2))


def render_consent(width: int, height: int, workspace: Path | str, t: float = 0.0, st: Consent | None = None) -> str:
    """One frame of the consent screen.  Degrades rather than crashing: every
    component clips, so a 40x12 terminal still gets a usable question."""
    cv = Canvas(max(1, width), max(1, height), fg=INK, bg=PAPER)
    w, h = cv.w, cv.h
    m = 1 if w < 60 else 2
    inner = max(1, w - 2 * m)

    brutal.ticker(cv, 0, 0, w, TICKER, t, speed=6.0)
    brutal.heading(cv, m, 2, HEADLINE, tracking=2 if w >= 60 else 0, underline="accent", width=inner)
    cv.text(m, 4, fit(STANDFIRST, inner), MUTED, PAPER, False, max_w=inner)
    path = str(workspace)
    cv.text(m, 5, fit(f"workspace  {path}", inner), INK, PAPER, True, max_w=inner)

    down = st.pressed if st is not None and (t - st.pressed_at) < 0.16 else ""
    top = 7
    body = max(0, h - top - 2)
    if w >= 76 and body >= 8:  # two columns
        gut = 4
        left_w = (inner - gut) // 2
        right_w = inner - gut - left_w
        card = min(body, 12)
        _option(cv, m, top, left_w, card, title=WORKSPACE_TITLE, key=WORKSPACE_KEY,
                lines=WORKSPACE_MEANS, tone="ok", pressed=down == "workspace", default=True)
        _option(cv, m + left_w + gut, top, right_w, card, title=FULL_TITLE, key=FULL_KEY,
                lines=FULL_MEANS, tone="err", pressed=down == "full", default=False)
        # +2 clears the panel's own drop shadow, which the small print used to
        # collide with.
        y = top + card + 2
    else:  # stacked; the safe option stays first
        half = max(4, body // 2)
        _option(cv, m, top, inner, min(half, 8), title=WORKSPACE_TITLE, key=WORKSPACE_KEY,
                lines=WORKSPACE_MEANS, tone="ok", pressed=down == "workspace", default=True)
        y2 = top + min(half, 8) + 1
        _option(cv, m, y2, inner, max(0, min(h - y2 - 2, 9)), title=FULL_TITLE, key=FULL_KEY,
                lines=FULL_MEANS, tone="err", pressed=down == "full", default=False)
        y = y2 + max(0, min(h - y2 - 2, 9)) + 2

    if y < h - 1:
        for line in SMALL_PRINT:
            if y >= h - 1:
                break
            cv.text(m, y, fit(line, inner), MUTED, PAPER, False, max_w=inner)
            y += 1

    chosen = st.choice if st is not None else None
    if chosen is None:
        left = f"{WORKSPACE_KEY} keeps offset here  {G.STAR}  {FULL_KEY} hands over the machine"
    else:
        left = f"granted  {G.STAR}  {permission_badge(chosen)}"
    brutal.status_bar(cv, 0, h - 1, w, left=left, right=f"{path[-24:]}", t=t,
                      fill="err" if chosen == "full" else "ink",
                      text_color=INK if chosen == "full" else PAPER)
    return cv.render()


def summary_lines(scope: Scope | None, workspace: Path | str) -> list[str]:
    """What `/permissions` prints.  Names the consequence, not the setting."""
    if scope == "full":
        return [
            f"full system access, granted for {workspace}",
            "offset may read and write any file on this machine, run arbitrary",
            "commands, launch applications, and write documents anywhere.",
            "revoke with /permissions revoke.",
        ]
    if scope == "workspace":
        return [
            f"workspace only: {workspace}",
            "paths outside this folder are refused, not asked about.",
            "grant more with /permissions full.",
        ]
    return ["no permission recorded for this workspace yet.", "offset is behaving as workspace only."]


_UNUSED = (CYAN, RED, display, label, text_width)
