"""Project instruction files: what the repository wants the agent to know.

A coding agent that ignores the file the team wrote for exactly this purpose is
missing table stakes, so discovery is deliberately generous: several accepted
names, walked from the workspace up towards home, plus a global layer.

Two rules keep it honest:

  * order is stable and closest-first, because a nested file exists precisely to
    override the one above it;
  * the budget is enforced with a visible marker. Silently dropping half of
    someone's instructions is worse than telling them it was cut.

Frontmatter is optional. `alwaysApply: false` with `globs:` makes a file
conditional, so a rule about migrations does not follow you into the parser.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

#: Accepted names, in preference order within a single directory.
NAMES: Final[tuple[str, ...]] = (
    "OFFSET.md",
    "AGENTS.md",
    "CLAUDE.md",
    ".offset/instructions.md",
    ".cursorrules",
)

#: Per-file and total caps. Generous, but not "paste the whole wiki".
MAX_FILE_BYTES: Final = 32_000
MAX_TOTAL_BYTES: Final = 96_000
CUT_MARKER: Final = "\n[... truncated: this instruction file is longer than the budget ...]"


@dataclass(slots=True)
class Instructions:
    path: Path
    text: str
    always: bool = True
    globs: tuple[str, ...] = ()
    truncated: bool = False
    #: Distance from the workspace: 0 is the workspace itself, higher is further up.
    depth: int = 0

    @property
    def conditional(self) -> bool:
        return not self.always and bool(self.globs)

    def applies_to(self, paths: Sequence[str]) -> bool:
        """Unconditional files always apply; conditional ones need a match."""
        if not self.conditional:
            return self.always or not self.globs
        return any(
            fnmatch.fnmatch(candidate, pattern) or fnmatch.fnmatch(Path(candidate).name, pattern)
            for pattern in self.globs
            for candidate in paths
        )

    def label(self) -> str:
        bits = [str(self.path)]
        if self.conditional:
            bits.append(f"when {', '.join(self.globs)}")
        if self.truncated:
            bits.append("truncated")
        return "  ".join(bits)


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """A deliberately small YAML-ish reader: `key: value` lines between `---`.

    Pulling in a YAML parser for four keys would be the wrong trade, and a
    malformed block must degrade to "no frontmatter" rather than an error.
    """
    if not raw.startswith("---"):
        return {}, raw
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    fields: dict[str, str] = {}
    for i, line in enumerate(lines[1:], 1):
        if line.strip() in ("---", "..."):
            return fields, "\n".join(lines[i + 1 :]).lstrip("\n")
        key, sep, value = line.partition(":")
        if sep and key.strip():
            fields[key.strip().lower()] = value.strip().strip("\"'")
    return {}, raw  # never closed: treat the whole file as body


def _truthy(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in ("false", "no", "0", "off")


def _globs(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    cleaned = value.strip().strip("[]")
    return tuple(part.strip().strip("\"'") for part in cleaned.split(",") if part.strip())


def read_one(path: Path, *, depth: int = 0) -> Instructions | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    fields, body = parse_frontmatter(raw)
    truncated = False
    if len(body.encode("utf-8")) > MAX_FILE_BYTES:
        body = body.encode("utf-8")[:MAX_FILE_BYTES].decode("utf-8", "ignore") + CUT_MARKER
        truncated = True
    body = body.strip()
    if not body:
        return None
    return Instructions(
        path=path,
        text=body,
        always=_truthy(fields.get("alwaysapply")),
        globs=_globs(fields.get("globs")),
        truncated=truncated,
        depth=depth,
    )


def home_dir() -> Path:
    return Path(os.environ.get("OFFSET_HOME") or (Path.home() / ".offset"))


def discovered(workspace: Path | str, *, ceiling: Path | None = None) -> list[Instructions]:
    """Every instruction file that applies here, closest first.

    Walks from the workspace towards `ceiling` (the home directory by default),
    then adds the global file. Deduplicated by resolved path, so a workspace
    inside the home directory cannot read the same file twice.
    """
    start = Path(workspace).expanduser().resolve()
    stop = (ceiling or Path.home()).expanduser().resolve()
    found: list[Instructions] = []
    seen: set[Path] = set()

    here = start
    depth = 0
    while True:
        for name in NAMES:
            candidate = (here / name).resolve()
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            got = read_one(candidate, depth=depth)
            if got is not None:
                found.append(got)
        if here == stop or here == here.parent or stop not in here.parents:
            break
        here = here.parent
        depth += 1

    global_file = (home_dir() / "OFFSET.md").resolve()
    if global_file.is_file() and global_file not in seen:
        got = read_one(global_file, depth=depth + 1)
        if got is not None:
            found.append(got)
    return found


def assemble(
    workspace: Path | str,
    *,
    budget: int = MAX_TOTAL_BYTES,
    paths: Sequence[str] = (),
    ceiling: Path | None = None,
) -> str:
    """The block to append to the system prompt, or "" when there is nothing.

    `paths` are the files in play, used to decide whether a conditional file
    applies. Closest-first means a nested file's instructions are read last and
    therefore win, which is the convention everyone already expects.
    """
    chosen = [got for got in discovered(workspace, ceiling=ceiling) if got.applies_to(paths)]
    if not chosen:
        return ""
    blocks: list[str] = []
    spent = 0
    for got in reversed(chosen):  # furthest first, so the closest has the last word
        header = f"# instructions from {got.path}"
        chunk = f"{header}\n{got.text}"
        cost = len(chunk.encode("utf-8"))
        if spent + cost > budget:
            room = max(0, budget - spent)
            if room > len(header) + 80:
                clipped = chunk.encode("utf-8")[:room].decode("utf-8", "ignore")
                blocks.append(clipped + CUT_MARKER)
            else:
                blocks.append(f"[... {len(chosen) - len(blocks)} more instruction file(s) omitted: over budget ...]")
            break
        blocks.append(chunk)
        spent += cost
    return "\n\n".join(blocks)


def summary(workspace: Path | str) -> list[str]:
    """What `/context` prints."""
    found = discovered(workspace)
    if not found:
        return [
            "no project instructions found",
            "offset reads " + ", ".join(NAMES[:3]) + " walking up from the workspace",
        ]
    return [got.label() for got in found]
