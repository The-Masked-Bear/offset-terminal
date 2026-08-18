"""What the user has allowed offset to touch, and where that answer is kept.

A grant is not a preference: it is the record of a person having been shown
what full access means and having said yes.  Three consequences shape this
module.

  * It is stored per workspace.  Saying "yes, the whole machine" while fixing
    a toy project must not silently arm the agent the next time it is opened
    somewhere else, so a grant records the workspace it was made for and
    `current()` only answers for that path.
  * It is auditable.  `when` and `workspace` are written down so the user can
    read the file and see what they agreed to.
  * A corrupt or unreadable store means "no grant", never a crash and never an
    accidental escalation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from offset.core import settings
from typing import Final, Literal

Scope = Literal["workspace", "full"]

def config_dir() -> Path:
    """Read late, never cached at import - see `settings.home`."""
    return settings.home()


def permissions_file() -> Path:
    return config_dir() / "permissions.json"

#: Bump when the stored shape changes; an unknown version is ignored rather
#: than guessed at, which fails closed.
VERSION: Final = 1


@dataclass(slots=True)
class Grant:
    """A decision the user made, for one workspace, at one moment."""

    scope: Scope
    workspace: Path
    when: float

    @property
    def full(self) -> bool:
        return self.scope == "full"

    def root(self) -> Path | None:
        """The `ToolContext.root` this grant implies; None is the whole machine."""
        return None if self.full else self.workspace

    def to_json(self) -> dict[str, object]:
        return {"scope": self.scope, "workspace": str(self.workspace), "when": self.when}

    @classmethod
    def from_json(cls, raw: object) -> "Grant | None":
        if not isinstance(raw, dict):
            return None
        scope = raw.get("scope")
        workspace = raw.get("workspace")
        if scope not in ("workspace", "full") or not isinstance(workspace, str) or not workspace:
            return None
        when = raw.get("when")
        return cls(scope, Path(workspace), float(when) if isinstance(when, (int, float)) else 0.0)


def _canonical(workspace: Path | str) -> Path:
    """Resolve so that `.`, symlinks and trailing slashes cannot fake a match."""
    return Path(workspace).expanduser().resolve()


def _load() -> dict[str, Grant]:
    try:
        raw = json.loads(permissions_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != VERSION:
        return {}
    entries = raw.get("grants")
    if not isinstance(entries, dict):
        return {}
    out: dict[str, Grant] = {}
    for key, value in entries.items():
        grant = Grant.from_json(value)
        if grant is not None:
            out[key] = grant
    return out


def _store(grants: dict[str, Grant]) -> Path:
    """Write 0600, atomically — the temp file is created private too."""
    config_dir().mkdir(parents=True, exist_ok=True)
    payload = {
        "version": VERSION,
        "grants": {key: grant.to_json() for key, grant in grants.items()},
    }
    tmp = permissions_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, permissions_file())
    os.chmod(permissions_file(), 0o600)
    return permissions_file()


def current(workspace: Path | str) -> Grant | None:
    """The grant in force for `workspace`, or None if it was never asked."""
    return _load().get(str(_canonical(workspace)))


def grant(scope: Scope, workspace: Path | str) -> Grant:
    """Record the user's answer.  Overwrites any earlier one for this path."""
    if scope not in ("workspace", "full"):
        raise ValueError(f"unknown permission scope: {scope!r}")
    path = _canonical(workspace)
    grants = _load()
    made = Grant(scope, path, time.time())
    grants[str(path)] = made
    _store(grants)
    return made


def revoke(workspace: Path | str | None = None) -> bool:
    """Forget one workspace's grant, or every grant when given None."""
    grants = _load()
    if not grants:
        return False
    if workspace is None:
        _store({})
        return True
    key = str(_canonical(workspace))
    if key not in grants:
        return False
    del grants[key]
    _store(grants)
    return True


def granted() -> list[Grant]:
    """Every live grant, newest first — for `/permissions` and the UI."""
    return sorted(_load().values(), key=lambda g: g.when, reverse=True)


def root_for(workspace: Path | str) -> Path | None:
    """The boundary to hand `ToolContext.root`; the workspace until told otherwise."""
    made = current(workspace)
    return made.root() if made is not None else _canonical(workspace)


def mode_for(workspace: Path | str, fallback: str = "auto-edit") -> str:
    """The approval mode a grant implies.  Only "full" changes anything."""
    made = current(workspace)
    return "full" if made is not None and made.full else fallback
