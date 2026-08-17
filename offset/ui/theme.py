"""Named palettes.

`offset.ui.tokens` holds the DNA palette and must keep holding it — it is the
transcription of the reference stylesheet.  But a user needs to be able to
swap the palette (high contrast on a sunlit Pi screen, monochrome over a bad
serial link, or their own file) without every drawing call importing a colour
constant.  So renderers ask for a *role* and this module decides what colour
that role is right now.

A broken `~/.offset/theme.json` must never stop the shell from drawing: the
loader degrades to the built-in palette and carries the reason on the theme
itself, so the UI can say what is wrong instead of dying.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Final, Mapping

from offset.ui.tokens import (
    CYAN,
    GRID,
    INK,
    MINT,
    MUTED,
    PAPER,
    PINK,
    RED,
    RGB,
    SURFACE,
    YELLOW,
    blend,
)

#: Every colour a renderer may ask for.  Adding a role here is a design
#: decision; adding a raw constant at a call site is a bug.
ROLES: Final = (
    "ink",
    "paper",
    "surface",
    "muted",
    "accent",
    "branch",
    "info",
    "ok",
    "err",
    "grid",
)

#: `offset.ui.tokens.TONES` names, mapped onto roles, so tone strings that
#: already exist in the codebase (egg tones, message tones) keep working.
_TONE_ROLE: Final[dict[str, str]] = {
    "plain": "surface",
    "accent": "accent",
    "branch": "branch",
    "info": "info",
    "ok": "ok",
    "err": "err",
    "ink": "ink",
    "paper": "paper",
    "muted": "muted",
    "grid": "grid",
}


class ThemeError(ValueError):
    """Every problem with a palette file, in one message.

    The user fixes a file; they should be told everything wrong with it at
    once rather than one error per reload.
    """


class Theme:
    """A resolved palette.  `note` explains why it is not what you asked for."""

    __slots__ = (*ROLES, "name", "note", "tones")

    def __init__(self, name: str, colors: Mapping[str, RGB], *, note: str = "") -> None:
        missing = [role for role in ROLES if role not in colors]
        if missing:
            raise ThemeError(f"{name}: missing role(s) {', '.join(missing)}")
        self.name = name
        self.note = note
        for role in ROLES:
            setattr(self, role, colors[role])
        self.tones = {tone: getattr(self, role) for tone, role in _TONE_ROLE.items()}

    def colors(self) -> dict[str, RGB]:
        return {role: getattr(self, role) for role in ROLES}

    def tone(self, name: str, default: str = "plain") -> RGB:
        """Resolve a tone string; unknown tones fall back rather than raise,
        because tones arrive from user data (eggs, command output)."""
        return self.tones.get(name) or self.tones[default]

    def replace(self, name: str, colors: Mapping[str, RGB], *, note: str = "") -> "Theme":
        merged = self.colors()
        merged.update(colors)
        return Theme(name, merged, note=note)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Theme({self.name!r})"


def _theme(name: str, **colors: RGB) -> Theme:
    """Built-in constructor: `grid` is derived unless stated, since it is only
    ever ink at 6% over the background."""
    colors.setdefault("grid", blend(colors["ink"], colors["paper"], 0.06))
    return Theme(name, colors)


def _rgb(value: int) -> RGB:
    return RGB((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


#: The reference palette, straight from tokens.
DNA: Final = Theme(
    "dna",
    {
        "ink": INK,
        "paper": PAPER,
        "surface": SURFACE,
        "muted": MUTED,
        "accent": YELLOW,
        "branch": PINK,
        "info": CYAN,
        "ok": MINT,
        "err": RED,
        "grid": GRID,
    },
)

#: Maximum separation: pure white ground, pure black ink, fully saturated
#: accents.  Still black-on-bright — the style never inverts.
CONTRAST: Final = _theme(
    "contrast",
    ink=_rgb(0x000000),
    paper=_rgb(0xFFFFFF),
    surface=_rgb(0xFFFFFF),
    muted=_rgb(0x1F1F1F),
    accent=_rgb(0xFFE500),
    branch=_rgb(0xFF00D0),
    info=_rgb(0x00E5FF),
    ok=_rgb(0x00FF66),
    err=_rgb(0xFF1A1A),
    grid=_rgb(0xD0D0D0),
)

#: One hue, five values.  Accents are separated by lightness alone, so the
#: layout still reads where colour does not survive.
MONO: Final = _theme(
    "mono",
    ink=_rgb(0x111111),
    paper=_rgb(0xF4F4F0),
    surface=_rgb(0xFFFFFF),
    muted=_rgb(0x555555),
    accent=_rgb(0xDCDCDC),
    branch=_rgb(0xBEBEBE),
    info=_rgb(0xCECECE),
    ok=_rgb(0xEAEAEA),
    err=_rgb(0x9A9A9A),
)

DEFAULT: Final = "dna"
BUILTIN: Final[dict[str, Theme]] = {t.name: t for t in (DNA, CONTRAST, MONO)}


def names() -> list[str]:
    return list(BUILTIN)


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------

#: Honours OFFSET_HOME, so a test (or a sandbox) can isolate the palette.
def home() -> Path:
    return Path(os.environ.get("OFFSET_HOME") or (Path.home() / ".offset"))


def path() -> Path:
    return home() / "theme.json"


def _colour(text: object) -> RGB:
    if not isinstance(text, str):
        raise ThemeError(f'expected "#rrggbb", got {text!r}')
    s = text.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ThemeError(f'expected "#rrggbb", got {text!r}')
    try:
        return _rgb(int(s, 16))
    except ValueError:
        raise ThemeError(f'expected "#rrggbb", got {text!r}') from None


def _decode(data: object, source: str) -> tuple[str, dict[str, RGB], str | None]:
    """(built-in the file asks for, colour overrides, explicit name).

    Splitting decode from construction is what lets the environment pick a
    different base while the file's overrides still apply.
    """
    if not isinstance(data, dict):
        raise ThemeError(f"{source}: expected an object, got {type(data).__name__}")

    problems: list[str] = []
    asked = data.get("theme", data.get("base", DEFAULT))
    if not isinstance(asked, str) or asked not in BUILTIN:
        problems.append(f"{source}: theme must be one of {', '.join(names())}, got {asked!r}")
        asked = DEFAULT

    raw = data.get("colors")
    if raw is None:
        raw = {k: v for k, v in data.items() if k not in ("theme", "base", "name", "colors")}
    if not isinstance(raw, dict):
        problems.append(f"{source}: colors must be an object, got {type(raw).__name__}")
        raw = {}

    colors: dict[str, RGB] = {}
    for key, value in raw.items():
        if key not in ROLES:
            problems.append(f"{source}: unknown role {key!r}; known roles are {', '.join(ROLES)}")
            continue
        try:
            colors[key] = _colour(value)
        except ThemeError as exc:
            problems.append(f"{source}: {key}: {exc}")

    if problems:
        raise ThemeError("; ".join(problems))

    name = data.get("name")
    return asked, colors, name if isinstance(name, str) and name else None


def parse(data: object, *, source: str = "theme") -> Theme:
    """Turn decoded JSON into a Theme, or raise with every problem listed."""
    asked, colors, name = _decode(data, source)
    base = BUILTIN[asked]
    return base.replace(name or ("custom" if colors else base.name), colors)


def read(file: Path) -> Theme:
    """Load one palette file.  Missing file, bad JSON and bad content all
    surface as ThemeError so callers have a single thing to catch."""
    return parse(_json(file), source=file.name)


def _json(file: Path) -> object:
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ThemeError(f"{file.name}: cannot read ({exc.strerror or exc})") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ThemeError(f"{file.name}: invalid JSON at line {exc.lineno} column {exc.colno}") from None


def load(file: Path | None = None) -> Theme:
    """The palette to draw with.  Never raises.

    Order: OFFSET_THEME names a built-in and wins over the file's own `theme`
    key (environment beats config, as everywhere else); the file's colour
    overrides always apply on top.  Any problem degrades to a built-in and is
    recorded in `note` — a bad palette is not a reason to stop drawing.
    """
    file = path() if file is None else file
    env = os.environ.get("OFFSET_THEME") or ""
    notes: list[str] = []
    if env and env not in BUILTIN:
        notes.append(f"OFFSET_THEME={env!r} is not a known theme; using {DEFAULT}")
        env = ""

    asked, colors, name = env or DEFAULT, {}, None
    if file.exists():
        try:
            asked, colors, name = _decode(_json(file), file.name)
            if env:
                asked = env
        except ThemeError as exc:
            notes.append(f"{exc} — using {env or DEFAULT}")
            asked = env or DEFAULT

    base = BUILTIN[asked]
    return base.replace(name or ("custom" if colors else base.name), colors, note="; ".join(notes))


# --------------------------------------------------------------------------
# the active theme
# --------------------------------------------------------------------------

_active: Theme | None = None


def active() -> Theme:
    """Memoised: the renderer asks for this on every frame."""
    global _active
    if _active is None:
        _active = load()
    return _active


def reload(file: Path | None = None) -> Theme:
    """Re-read from disk (after the user edits the file, or in a test)."""
    global _active
    _active = load(file)
    return _active


def use(name: str) -> str | None:
    """Switch to a built-in.  Returns a message when the name is unknown,
    because this is driven by user input, not by code."""
    global _active
    chosen = BUILTIN.get(name.strip().lower())
    if chosen is None:
        return f"unknown theme {name!r}; try {', '.join(names())}"
    _active = chosen
    return None


def reset() -> None:
    """Forget the memo; the next `active()` reloads."""
    global _active
    _active = None
