"""File snapshots: the record that makes an edit undoable.

A writing tool records what a file looked like *before* it touched it.  The
bytes go into a content-addressed store under `<workspace>/.offset/snapshots`,
so a hundred writes of the same content cost one blob; the index — path, hash,
size, tool, which call did it — is appended to the session as a "snapshot"
entry, so the undo history reloads from disk together with the conversation and
needs no second database.

Three decisions worth stating.

  * A file that cannot be stored (binary, or larger than the cap) is *recorded
    as unstorable* rather than skipped, because a rewind that silently leaves a
    file at its new contents is far worse than one that names what it could not
    put back.
  * A snapshot of a file that did not exist is a null hash; that is what lets a
    rewind delete a file the agent created.
  * A snapshot is bookkeeping, never conversation: it is written as a root and
    must not become the parent of the next real entry.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable

from offset.core.entries import Entry
from offset.core.session import Session

#: The entry type.  Spelled literally here so this module does not depend on
#: the constant landing in `offset.core.entries` first; it is the same string.
SNAPSHOT: Final = "snapshot"

#: Files bigger than this are recorded but not stored.  A source file is
#: kilobytes; anything in the megabytes is a build artefact or an asset, and
#: filling the store with those costs the user disk for no undo they want.
DEFAULT_MAX_BYTES: Final = 2_000_000

#: Where the blobs live, relative to the workspace root.
STORE_DIR: Final = (".offset", "snapshots")

TOO_LARGE: Final = "not snapshotted, too large"
BINARY: Final = "not snapshotted, binary"


def max_bytes(override: int | None = None) -> int:
    """The size cap.  Caller beats settings, settings beat the built-in."""
    if override is not None:
        return int(override)
    try:
        from offset.core import settings
    except ImportError:  # settings layer not present: the built-in cap stands
        return DEFAULT_MAX_BYTES
    got = settings.get("snapshots.maxBytes", DEFAULT_MAX_BYTES)
    return int(got) if isinstance(got, (int, float)) and got > 0 else DEFAULT_MAX_BYTES


# -- the blob store ---------------------------------------------------------


class Store:
    """Content-addressed blobs.  The name *is* the hash, so writes dedupe."""

    __slots__ = ("dir", "root")

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dir = self.root.joinpath(*STORE_DIR)

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def path_for(self, digest: str) -> Path:
        """Two-character fanout: one directory per 1/256th of the keyspace."""
        return self.dir / digest[:2] / digest

    def has(self, digest: str) -> bool:
        return self.path_for(digest).exists()

    def put(self, data: bytes) -> str:
        """Store `data` if it is not already there.  Returns its hash."""
        digest = self.digest(data)
        target = self.path_for(digest)
        if target.exists():
            return digest
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, data)
        return digest

    def get(self, digest: str) -> bytes | None:
        try:
            return self.path_for(digest).read_bytes()
        except OSError:
            return None

    def blobs(self) -> list[str]:
        """Every hash held, for accounting and for tests."""
        if not self.dir.is_dir():
            return []
        return sorted(p.name for p in self.dir.glob("*/*") if p.is_file())

    def bytes_held(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob("*/*") if p.is_file()) if self.dir.is_dir() else 0

    def __repr__(self) -> str:
        return f"<Store {self.dir} blobs={len(self.blobs())}>"


def _atomic_write(target: Path, data: bytes, mode: int | None = None) -> None:
    """Write via a sibling temp file: a crash never leaves half a file."""
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        os.unlink(tmp)
        raise


# -- the index record -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Record:
    """One file's prior state, exactly as the session holds it.

    `hash is None` with no `skipped` reason means the file did not exist, which
    a rewind honours by deleting it again.
    """

    path: str  # workspace-relative posix, or absolute when outside the root
    root: str  # the workspace this was captured in; the store lives under it
    hash: str | None = None
    size: int = 0
    tool: str = ""
    call: str | None = None  # entry id of the tool call that overwrote it
    mode: int | None = None  # permission bits; a rewind that drops +x is a bug
    skipped: str | None = None  # why it could not be stored, for the user
    ts: float = 0.0
    id: str = ""  # the snapshot entry's own id

    @property
    def stored(self) -> bool:
        return self.hash is not None and self.skipped is None

    @property
    def absent(self) -> bool:
        """The file did not exist when it was captured."""
        return self.hash is None and self.skipped is None

    def to_data(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "root": self.root,
            "hash": self.hash,
            "size": self.size,
            "tool": self.tool,
            "call": self.call,
            "mode": self.mode,
            "skipped": self.skipped,
        }

    @classmethod
    def from_entry(cls, entry: Entry) -> "Record":
        d = entry.data
        digest = d.get("hash")
        mode = d.get("mode")
        return cls(
            path=str(d.get("path") or ""),
            root=str(d.get("root") or ""),
            hash=digest if isinstance(digest, str) else None,
            size=int(d.get("size") or 0),
            tool=str(d.get("tool") or ""),
            call=d.get("call") if isinstance(d.get("call"), str) else None,
            mode=int(mode) if isinstance(mode, int) else None,
            skipped=str(d["skipped"]) if d.get("skipped") else None,
            ts=entry.ts,
            id=entry.id,
        )

    def describe(self) -> str:
        """One line for the UI."""
        if self.skipped:
            return f"{self.path}  {self.skipped}"
        if self.absent:
            return f"{self.path}  (did not exist)"
        return f"{self.path}  {self.size}B  {(self.hash or '')[:12]}"


@dataclass(slots=True)
class Rewind:
    """What a restore did, and what it could not do.

    `failed` is the half that matters: the user has to know which files stayed
    at their new contents.
    """

    restored: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.failed

    @property
    def changed(self) -> int:
        return len(self.restored) + len(self.removed)

    def lines(self) -> list[str]:
        out = [f"restored {p}" for p in self.restored]
        out += [f"removed  {p}" for p in self.removed]
        out += [f"cannot restore {p}: {why}" for p, why in self.failed]
        return [self.error] if self.error else out


# -- capture ----------------------------------------------------------------


def target_paths(args: dict[str, Any]) -> list[str]:
    """The files a call is about to touch, by the convention the tools share.

    Writing built-ins take `path`; a few take `paths`.  Anything else (a shell
    command, say) is unknowable from the arguments and returns nothing rather
    than guessing.
    """
    out: list[str] = []
    one = args.get("path") or args.get("file")
    if isinstance(one, str) and one:
        out.append(one)
    many = args.get("paths")
    if isinstance(many, (list, tuple)):
        out += [p for p in many if isinstance(p, str) and p]
    return out


def capture(
    session: Session,
    path: str | os.PathLike[str],
    *,
    tool: str = "",
    call: str | None = None,
    root: str | os.PathLike[str] | None = None,
    cap: int | None = None,
) -> Record:
    """Record `path`'s current content before something overwrites it.

    Always appends an index entry, even when the content cannot be stored: the
    unstorable cases are exactly the ones a rewind must be able to report.
    """
    base = Path(root).expanduser().resolve() if root is not None else Path.cwd().resolve()
    store = Store(base)
    target = Path(path).expanduser()
    target = (base / target).resolve() if not target.is_absolute() else target.resolve()
    rel = _relative(target, base)

    data, size, mode, skipped = _read_prior(target, store, max_bytes(cap))
    digest = store.put(data) if data is not None else None
    record = Record(
        path=rel,
        root=str(base),
        hash=digest,
        size=size,
        tool=tool,
        call=call,
        mode=mode,
        skipped=skipped,
    )
    entry = _append(session, record.to_data())
    return Record(**{**_as_kwargs(record), "ts": entry.ts, "id": entry.id})


def capture_all(
    session: Session,
    paths: Iterable[str | os.PathLike[str]],
    *,
    tool: str = "",
    call: str | None = None,
    root: str | os.PathLike[str] | None = None,
    cap: int | None = None,
) -> list[Record]:
    """Capture several files for one call, in order, deduplicating repeats."""
    seen: set[str] = set()
    out: list[Record] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(capture(session, p, tool=tool, call=call, root=root, cap=cap))
    return out


def _read_prior(target: Path, store: Store, cap: int) -> tuple[bytes | None, int, int | None, str | None]:
    """The bytes to store, or a reason we are not storing them."""
    if store.dir == target or store.dir in target.parents:
        return None, 0, None, "not snapshotted, inside the snapshot store"
    try:
        st = target.lstat()
    except OSError:
        return None, 0, None, None  # did not exist: a rewind deletes it again
    if target.is_symlink():
        return None, 0, None, "not snapshotted, symlink"
    if not target.is_file():
        return None, 0, None, "not snapshotted, not a regular file"
    mode = st.st_mode & 0o7777
    if st.st_size > cap:
        return None, int(st.st_size), mode, f"{TOO_LARGE} ({st.st_size}B > {cap}B)"
    try:
        data = target.read_bytes()
    except OSError as exc:
        return None, int(st.st_size), mode, f"not snapshotted, unreadable: {exc}"
    if b"\0" in data:
        return None, len(data), mode, f"{BINARY} ({len(data)}B)"
    return data, len(data), mode, None


def _as_kwargs(record: Record) -> dict[str, Any]:
    return {
        "path": record.path,
        "root": record.root,
        "hash": record.hash,
        "size": record.size,
        "tool": record.tool,
        "call": record.call,
        "mode": record.mode,
        "skipped": record.skipped,
    }


def _relative(target: Path, base: Path) -> str:
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        return target.as_posix()  # outside the workspace: keep it absolute


def _append(session: Session, data: dict[str, Any]) -> Entry:
    """Append the index entry without letting it join the conversation.

    A snapshot records what a tool was about to overwrite, not something anyone
    said, so it is written as a root.  If this session build still advances the
    leaf for unrecognised types, the leaf is put back — the next real entry has
    to chain onto the last real one.
    """
    before = session.leaf
    entry = session.append(SNAPSHOT, data, parent=None)
    if session.leaf == entry.id and before != entry.id:
        session.branch(before)
    return entry


# -- reading the history ----------------------------------------------------


def records(session: Session) -> list[Record]:
    """Every snapshot in the session, oldest first."""
    return [Record.from_entry(e) for e in session.all_entries() if e.type == SNAPSHOT]


def history(
    session: Session,
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str] | None = None,
) -> list[Record]:
    """The snapshots taken of one file, oldest first."""
    want = _absolute(path, root)
    return [r for r in records(session) if _target_of(r, root) == want]


def _absolute(path: str | os.PathLike[str], root: str | os.PathLike[str] | None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    base = Path(root).expanduser().resolve() if root is not None else Path.cwd().resolve()
    return (base / p).resolve()


def _target_of(record: Record, root: str | os.PathLike[str] | None) -> Path:
    """Where this record's file lives now.  An explicit root overrides the
    captured one, so a session can be replayed against a moved workspace."""
    p = Path(record.path)
    if p.is_absolute():
        return p.resolve()
    base = Path(root).expanduser().resolve() if root is not None else Path(record.root)
    return (base / p).resolve()


# -- restore ----------------------------------------------------------------


def restore(
    session: Session,
    target_entry_id: str | None,
    *,
    root: str | os.PathLike[str] | None = None,
) -> Rewind:
    """Put every file back to how it was when `target_entry_id` was written.

    For each file, the state at that moment is what the *earliest* snapshot
    taken after that entry recorded — later snapshots describe intermediate
    versions and must be ignored.  Restoring twice is a no-op the second time,
    because a file already holding the wanted bytes is left alone.
    """
    entries = list(session.all_entries())
    if target_entry_id is None:
        # "Before everything": every snapshot in the session counts, which is
        # the only way to undo a write that was the session's first entry.
        at = -1
    else:
        at = next((i for i, e in enumerate(entries) if e.id == target_entry_id), -2)
        if at < 0:
            return Rewind(error=f"no such entry: {target_entry_id}")

    first: dict[Path, Record] = {}
    for entry in entries[at + 1 :]:
        if entry.type != SNAPSHOT:
            continue
        record = Record.from_entry(entry)
        first.setdefault(_target_of(record, root), record)

    out = Rewind()
    for target, record in first.items():
        _apply(target, record, root, out)
    out.restored.sort()
    out.removed.sort()
    out.unchanged.sort()
    out.failed.sort()
    return out


def _apply(target: Path, record: Record, root: str | os.PathLike[str] | None, out: Rewind) -> None:
    shown = record.path
    if record.skipped:
        out.failed.append((shown, record.skipped))
        return
    if record.hash is None:
        _remove(target, shown, out)
        return
    blob = Store(root if root is not None else record.root).get(record.hash)
    if blob is None:
        out.failed.append((shown, f"snapshot {record.hash[:12]} is missing from the store"))
        return
    try:
        if target.is_file() and target.read_bytes() == blob:
            out.unchanged.append(shown)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, blob, record.mode)
    except OSError as exc:
        out.failed.append((shown, f"cannot write: {exc}"))
        return
    out.restored.append(shown)


def _remove(target: Path, shown: str, out: Rewind) -> None:
    """Undo a creation.  Empty parent directories are left behind: we cannot
    tell which of them the tool made and which the user already had."""
    if not target.exists() and not target.is_symlink():
        out.unchanged.append(shown)
        return
    try:
        if target.is_dir() and not target.is_symlink():
            out.failed.append((shown, "a directory now stands where the file was"))
            return
        target.unlink()
    except OSError as exc:
        out.failed.append((shown, f"cannot remove: {exc}"))
        return
    out.removed.append(shown)
