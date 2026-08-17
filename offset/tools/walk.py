"""Ignore-aware directory walking.

A search that reads `node_modules` is not a search, it is a stall.  The static
PRUNE list catches the usual suspects, but every repository carries its own
dead weight and has already written it down: `.gitignore`.  This module is the
only place that file is read, so `glob` and `grep` cannot drift apart on what
"the files in this project" means.

git's rules are the contract here, not fnmatch's: a pattern is relative to the
`.gitignore` that declares it, the last matching pattern wins (which is what
makes `!` re-include work), a trailing slash means directories only, a leading
or interior slash anchors, and `**` spans path segments.  An ignored directory
is never descended into — that pruning, not the matching, is where the speed
comes from, and it is also why a `!` cannot rescue a file whose parent
directory is excluded (git says the same).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Container, Iterator

#: Directories never worth walking; skipping them is the difference between a
#: glob that answers instantly and one that reads a virtualenv.
PRUNE: frozenset[str] = frozenset({
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".offset",
})

GITIGNORE = ".gitignore"


@dataclass(slots=True, frozen=True)
class Rule:
    """One `.gitignore` line, compiled."""

    regex: re.Pattern[str]
    negated: bool
    dir_only: bool


#: A compiled ignore file: how much of the candidate path to drop, what to put
#: back in front of it, and the rules themselves.  The two path fixups are what
#: let rules declared above the walk root and rules declared below it be
#: matched by the same loop.
@dataclass(slots=True, frozen=True)
class Frame:
    strip: int
    prefix: str
    rules: tuple[Rule, ...]


# -- pattern compilation ----------------------------------------------------


def _segment(seg: str) -> str:
    """Translate one path segment of a glob.  `*` and `?` never cross a slash."""
    out: list[str] = []
    i, n = 0, len(seg)
    while i < n:
        c = seg[i]
        i += 1
        if c == "*":
            while i < n and seg[i] == "*":
                i += 1
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1 if i < n and seg[i] in "!^" else i
            if j < n and seg[j] == "]":
                j += 1
            while j < n and seg[j] != "]":
                j += 1
            if j >= n:  # unterminated class: a literal bracket
                out.append(r"\[")
            else:
                body = seg[i:j].replace("\\", "\\\\")
                out.append("[" + ("^" + body[1:] if body[:1] == "!" else body) + "]")
                i = j + 1
        elif c == "\\" and i < n:
            out.append(re.escape(seg[i]))
            i += 1
        else:
            out.append(re.escape(c))
    return "".join(out)


def _body(pattern: str) -> str:
    segments = pattern.split("/")
    last = len(segments) - 1
    parts: list[str] = []
    for i, seg in enumerate(segments):
        if seg == "**":
            # Trailing `**` means "everything inside", so it needs a segment;
            # in the middle it means "any number of directories", including none.
            parts.append(".+" if i == last else "(?:[^/]+/)*")
        else:
            parts.append(_segment(seg) + ("" if i == last else "/"))
    return "".join(parts)


_TRAILING_SPACE = re.compile(r"(?<!\\)[ \t]+$")


def compile_rule(line: str) -> Rule | None:
    """Compile one `.gitignore` line, or None if it is blank or a comment."""
    text = _TRAILING_SPACE.sub("", line.rstrip("\r\n"))
    if not text or text.startswith("#"):
        return None
    negated = text.startswith("!")
    if negated:
        text = text[1:]
    elif text[:1] == "\\" and text[1:2] in ("!", "#"):
        text = text[1:]
    dir_only = text.endswith("/")
    text = text.rstrip("/")
    if not text:
        return None
    if text.startswith("/"):
        anchored, text = True, text.lstrip("/")
    else:
        anchored = "/" in text
    if not text:
        return None
    # An unanchored pattern matches at any depth; an anchored one is pinned to
    # the directory holding the .gitignore.
    head = "" if anchored else "(?:.*/)?"
    return Rule(re.compile("^" + head + _body(text) + "$"), negated, dir_only)


def compile_file(path: Path) -> tuple[Rule, ...]:
    """Compile a `.gitignore`.  A missing or unreadable file has no rules."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ()
    return tuple(r for r in (compile_rule(line) for line in text.splitlines()) if r is not None)


# -- matching ---------------------------------------------------------------


def _ignored(frames: tuple[Frame, ...], rel: str, is_dir: bool) -> bool:
    """Last match wins, outermost ignore file first — git's precedence exactly."""
    verdict = False
    for frame in frames:
        candidate = frame.prefix + rel[frame.strip:]
        for rule in frame.rules:
            if rule.dir_only and not is_dir:
                continue
            if rule.regex.match(candidate):
                verdict = not rule.negated
    return verdict


def _ancestor_frames(root: Path, boundary: Path | None) -> list[Frame]:
    """Ignore files declared above `root` still apply to what is inside it."""
    if boundary is None:
        return []
    try:
        inner = root.relative_to(boundary)
    except ValueError:
        return []
    frames: list[Frame] = []
    here = boundary
    for part in ("", *inner.parts):
        here = here / part if part else here
        if here == root:
            break
        rules = compile_file(here / GITIGNORE)
        if rules:
            frames.append(Frame(0, root.relative_to(here).as_posix() + "/", rules))
    return frames


# -- the walk ---------------------------------------------------------------


def walk(
    root: str | Path,
    *,
    respect_gitignore: bool = True,
    prune: Container[str] = PRUNE,
    boundary: Path | None = None,
    check: Callable[[], None] | None = None,
) -> Iterator[Path]:
    """Yield every file under `root` worth looking at, depth-first, name-sorted.

    `boundary` is an ancestor whose `.gitignore` files also apply — the
    workspace root, when the caller is searching a subdirectory of it.
    `check` is called once per directory so a long walk can be cancelled.
    """
    base = Path(root)
    frames: tuple[Frame, ...] = ()
    if respect_gitignore:
        top = _ancestor_frames(base, boundary)
        rules = compile_file(base / GITIGNORE)
        if rules:
            top.append(Frame(0, "", rules))
        frames = tuple(top)

    stack: list[tuple[str, tuple[Frame, ...]]] = [("", frames)]
    while stack:
        rel_dir, frames = stack.pop()
        if check is not None:
            check()
        here = base / rel_dir if rel_dir else base
        try:
            with os.scandir(here) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        children: list[tuple[str, tuple[Frame, ...]]] = []
        for entry in entries:
            name = entry.name
            if name in prune:
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                # A symlinked directory is never followed: it is the one way a
                # walk can fail to terminate.
                if not is_dir and entry.is_symlink() and entry.is_dir():
                    continue
            except OSError:
                continue
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if frames and _ignored(frames, rel, is_dir):
                continue
            if not is_dir:
                yield here / name
                continue
            inner = frames
            if respect_gitignore:
                rules = compile_file(here / name / GITIGNORE)
                if rules:
                    inner = frames + (Frame(len(rel) + 1, "", rules),)
            children.append((rel, inner))
        stack.extend(reversed(children))
