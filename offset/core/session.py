"""The session store: an append-only entry log with a movable leaf.

Design in one paragraph.  Every turn appends an entry whose parent is the
current leaf, so the log is a tree and the "conversation" is just the path
from a root down to the leaf.  Moving the leaf somewhere else (`branch`) is
itself recorded as an entry, so the whole navigation history replays exactly
from disk with no separate index and no rewriting.  Nothing is ever mutated
or deleted; abandoning a branch costs nothing and loses nothing, which is the
property speculative branching is built on.

Robustness rules, learned from the reference harness:

  * an entry whose parent is missing, or which points at itself, is a root
    rather than an error — a truncated or hand-edited log still opens;
  * unparseable lines are skipped, counted, and reported, never fatal;
  * duplicate ids keep the first occurrence, so a doubled append is idempotent;
  * labels are append-only entries resolved last-write-wins.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from offset.core.entries import (
    BOOKKEEPING,
    CONVERSATIONAL,
    LABEL,
    LEAF,
    MESSAGE,
    SNAPSHOT,
    Entry,
    new_id,
)


@dataclass(slots=True)
class Node:
    entry: Entry
    depth: int
    active: bool
    label: str | None
    children: list["Node"]


@dataclass(slots=True, frozen=True)
class SessionInfo:
    """What a session picker needs without opening the session for writing."""

    id: str
    path: Path
    mtime: float
    messages: int
    first_line: str
    size: int
    skipped: int = 0


class Session:
    """An append-only session file plus the indexes derived from it."""

    __slots__ = ("_by_id", "_entries", "_fh", "_kids", "_labels", "_leaf", "_skipped", "id", "path")

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.id = self.path.stem
        self._entries: list[Entry] = []
        self._by_id: dict[str, Entry] = {}
        self._kids: dict[str | None, list[Entry]] = {}
        self._labels: dict[str, str] = {}
        self._leaf: str | None = None
        self._skipped = 0
        self._fh = None

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def create(cls, root: str | os.PathLike[str], *, session_id: str | None = None) -> "Session":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        s = cls(root / f"{session_id or new_id()}.jsonl")
        s.path.touch()
        return s

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> "Session":
        s = cls(path)
        s.load()
        return s

    @classmethod
    def resume(cls, path: str | os.PathLike[str]) -> "Session":
        """Reopen an existing session at its recorded leaf.

        Unlike `open`, a missing file is an error rather than an empty log:
        resuming something that was never written would silently hand the user
        a fresh conversation wearing an old session's name.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"no session at {p}")
        s = cls(p)
        s.load()
        s._repair_leaf()
        return s

    @classmethod
    def list(cls, root: str | os.PathLike[str]) -> list[SessionInfo]:
        """Newest-first metadata for a session picker.

        Builds no indexes on purpose: the picker has to stay fast with hundreds
        of logs.  A file that yields nothing usable — truncated, binary, not a
        session at all — is left out instead of breaking the picker.
        """
        base = Path(root)
        if not base.is_dir():
            return []
        out: list[SessionInfo] = []
        for path in base.glob("*.jsonl"):
            info = cls._describe(path)
            if info is not None:
                out.append(info)
        # mtime first, then the id, which is chronological: ties stay stable.
        out.sort(key=lambda i: (i.mtime, i.id), reverse=True)
        return out

    @staticmethod
    def _describe(path: Path) -> SessionInfo | None:
        try:
            stat = path.stat()
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        messages = skipped = 0
        first_line = ""
        usable = False
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = Entry.from_obj(json.loads(line))
            except (ValueError, json.JSONDecodeError):
                skipped += 1
                continue
            usable = True
            if entry.type != MESSAGE:
                continue
            messages += 1
            if not first_line and entry.role == "user":
                first_line = " ".join(entry.text.split())
        if not usable:
            return None
        return SessionInfo(
            id=path.stem,
            path=path,
            mtime=stat.st_mtime,
            messages=messages,
            first_line=first_line,
            size=stat.st_size,
            skipped=skipped,
        )

    def _repair_leaf(self) -> None:
        """A leaf pointing at something the log cannot hold a conversation on
        (truncated file, hand edit, a leaf aimed at bookkeeping) would present
        the session as empty.  Fall back to the newest usable entry."""
        target = self._by_id.get(self._leaf) if self._leaf is not None else None
        if self._leaf is None or (target is not None and target.type not in BOOKKEEPING):
            return
        self._leaf = None
        for e in reversed(self._entries):
            if e.type not in BOOKKEEPING:
                self._leaf = e.id
                return

    def load(self) -> "Session":
        self._entries.clear()
        self._by_id.clear()
        self._kids.clear()
        self._labels.clear()
        self._leaf = None
        self._skipped = 0
        if not self.path.exists():
            return self
        # `errors="replace"` rather than strict: iteration decodes lazily, so one
        # bad byte from a torn write or a failing disk raised out of the loop and
        # made the whole session unopenable. Replaced bytes fail the JSON parse
        # below instead, and are counted as skipped like any other damaged line.
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = Entry.from_obj(json.loads(line))
                except (ValueError, json.JSONDecodeError):
                    self._skipped += 1
                    continue
                self._index(entry)
        return self

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def skipped_lines(self) -> int:
        """Corrupt lines ignored at load time.  Surface this, never hide it."""
        return self._skipped

    # -- indexing ---------------------------------------------------------

    def _index(self, entry: Entry) -> None:
        if entry.id in self._by_id:
            return  # duplicate append: first write wins
        self._by_id[entry.id] = entry
        self._entries.append(entry)
        if entry.type == LEAF:
            target = entry.data.get("leaf")
            self._leaf = target if isinstance(target, str) else None
            return
        if entry.type == LABEL:
            target, text = entry.data.get("target"), entry.data.get("text")
            if isinstance(target, str):
                if text:
                    self._labels[target] = str(text)
                else:
                    self._labels.pop(target, None)
            return
        if entry.type == SNAPSHOT:
            # Recorded and addressable, but it is a fact about the workspace,
            # not a turn: it must never become a parent or the leaf.
            return
        parent = entry.parent
        if parent == entry.id or (parent is not None and parent not in self._by_id):
            parent = None  # orphan or self-parent: promote to root
        self._kids.setdefault(parent, []).append(entry)
        self._leaf = entry.id

    def _write(self, entry: Entry) -> Entry:
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(entry.to_json() + "\n")
        self._fh.flush()
        self._index(entry)
        return entry

    # -- appending --------------------------------------------------------

    def append(self, type: str, data: dict[str, Any] | None = None, *, parent: str | None = "\0") -> Entry:
        """Append an entry as a child of the current leaf (or of `parent`)."""
        at = self._leaf if parent == "\0" else parent
        return self._write(Entry(id=new_id(), type=type, parent=at, data=dict(data or {})))

    def say(self, role: str, text: str, **extra: Any) -> Entry:
        return self.append(MESSAGE, {"role": role, "text": text, **extra})

    def label(self, target: str, text: str | None) -> Entry:
        """Set or clear a label.  Append-only; the newest write wins."""
        return self._write(Entry(id=new_id(), type=LABEL, parent=None, data={"target": target, "text": text}))

    def label_of(self, target: str) -> str | None:
        return self._labels.get(target)

    # -- navigation -------------------------------------------------------

    @property
    def leaf(self) -> str | None:
        return self._leaf

    def branch(self, target: str | None) -> Entry:
        """Move the leaf.  The move itself is recorded, so replay is exact."""
        if target is not None and target not in self._by_id:
            raise KeyError(f"no such entry: {target}")
        return self._write(Entry(id=new_id(), type=LEAF, parent=None, data={"leaf": target}))

    def reset_leaf(self) -> Entry:
        return self.branch(None)

    def entry(self, eid: str) -> Entry | None:
        return self._by_id.get(eid)

    def children(self, eid: str | None) -> list[Entry]:
        """Children in stable chronological order (ties broken by id)."""
        return sorted(self._kids.get(eid, ()), key=lambda e: (e.ts, e.id))

    def roots(self) -> list[Entry]:
        return self.children(None)

    def ancestry(self, eid: str | None = None) -> list[Entry]:
        """Root-to-node path.  Cycle-safe."""
        node = self._by_id.get(eid if eid is not None else (self._leaf or ""))
        seen: set[str] = set()
        out: list[Entry] = []
        while node is not None and node.id not in seen:
            seen.add(node.id)
            out.append(node)
            parent = node.parent
            node = self._by_id.get(parent) if parent and parent != node.id else None
        out.reverse()
        return out

    def transcript(self, eid: str | None = None) -> list[Entry]:
        """The active path, conversational entries only — what a model sees."""
        return [e for e in self.ancestry(eid) if e.type in CONVERSATIONAL]

    def all_entries(self) -> Iterator[Entry]:
        return iter(self._entries)

    # -- tree -------------------------------------------------------------

    def tree(self, *, include: Iterable[str] | None = None) -> list[Node]:
        """Depth-first forest.  The active path is marked and ordered first."""
        active = {e.id for e in self.ancestry()}
        allowed = None if include is None else frozenset(include)

        def build(entry: Entry, depth: int) -> Node:
            kids = [c for c in self.children(entry.id) if allowed is None or c.type in allowed]
            kids.sort(key=lambda c: (c.id not in active, c.ts, c.id))
            return Node(
                entry=entry,
                depth=depth,
                active=entry.id in active,
                label=self._labels.get(entry.id),
                children=[build(c, depth + 1) for c in kids],
            )

        roots = [r for r in self.roots() if allowed is None or r.type in allowed]
        roots.sort(key=lambda r: (r.id not in active, r.ts, r.id))
        return [build(r, 0) for r in roots]

    def rows(self, *, include: Iterable[str] | None = None) -> list[tuple[int, Entry, bool, str | None]]:
        """Flattened tree, ready for the renderer."""
        out: list[tuple[int, Entry, bool, str | None]] = []

        def walk(nodes: list[Node]) -> None:
            for n in nodes:
                out.append((n.depth, n.entry, n.active, n.label))
                walk(n.children)

        walk(self.tree(include=include))
        return out

    # -- whole-session operations -----------------------------------------

    def fork(self, root: str | os.PathLike[str] | None = None, *, session_id: str | None = None) -> "Session":
        """Copy the entire session to a new file.  The original is untouched."""
        self.close()
        dest_dir = Path(root) if root is not None else self.path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{session_id or new_id()}.jsonl"
        shutil.copyfile(self.path, dest)
        return Session.open(dest)

    def compact(self) -> int:
        """Rewrite the file with bookkeeping collapsed to its final state.

        Returns the number of lines dropped.  This is the one operation that
        rewrites history, so it writes to a temporary file and renames — a
        crash leaves the original log intact.
        """
        keep = [e for e in self._entries if e.type not in BOOKKEEPING]
        tail = [Entry(id=new_id(), type=LABEL, parent=None, data={"target": t, "text": x}) for t, x in self._labels.items()]
        if self._leaf is not None:
            tail.append(Entry(id=new_id(), type=LEAF, parent=None, data={"leaf": self._leaf}))
        dropped = len(self._entries) - len(keep)
        self.close()
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".jsonl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for e in keep + tail:
                    fh.write(e.to_json() + "\n")
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise
        self.load()
        return dropped

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<Session {self.id} entries={len(self._entries)} leaf={self._leaf}>"
