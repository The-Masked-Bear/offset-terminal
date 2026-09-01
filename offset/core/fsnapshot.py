"""Filesystem snapshots: workspace isolation that costs nothing when it can.

Speculative branching needs N private copies of the workspace, and a plain
recursive copy of a real project is seconds of wall clock and hundreds of
megabytes of disk per branch.  Modern filesystems already solve this: btrfs and
zfs take a snapshot in constant time, XFS and btrfs share extents through
`cp --reflink`, and APFS clones a file with `clonefile(2)`.  This module reaches
for whichever of those the filesystem *actually* has, and falls back to a real
copy only when none of them work.

Three decisions worth stating.

  * Capability is **probed, never guessed**.  Reading `/proc/mounts` or
    branching on `os.name` produces a confident answer that is wrong on exactly
    the interesting machines: an overlayfs container whose lower layer is btrfs,
    a bind mount, a tmpfs `/tmp`, an XFS volume built without `reflink=1`.  So
    each backend runs its own cheap operation in a scratch directory on the
    target filesystem and reads the exit status.  The answer is cached per
    device id, because the probe costs a fork and the answer cannot change
    while the volume is mounted.
  * Falling short is **reported, not hidden**.  If every zero-cost path
    declined, `Snapshot.instant` is False and `bytes_copied` says what it cost,
    so the shell can tell the user that branching just wrote 400 MB rather
    than pretending the operation was free.
  * Root is **never required**.  A backend that exists but lacks the privilege
    to use it — `btrfs subvolume delete` without `user_subvol_rm_allowed`, zfs
    on a dataset owned by someone else — fails its probe or its creation and we
    move to the next candidate.  A missing privilege must degrade, never raise.

Teardown is the dangerous half.  `release()` deletes a directory tree, so it
refuses, loudly, any path that is the workspace itself, an ancestor of it, the
home directory or a filesystem root; `snapshot()` applies the same test to the
destination before creating anything, which is what makes the context manager
safe to use around code that raises.

One honest limitation of the fallback: it walks with the project's own
ignore-aware walker, so it reproduces neither `.gitignore`d files nor the
always-noise directories (`.git`, `node_modules`, `.venv`), and a directory
containing no files at all is not recreated.  The zero-cost backends clone
everything.  The fallback is therefore a cheap approximation of the workspace
rather than a faithful one, which is precisely why `instant=False` is part of
the public handle instead of an implementation detail.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Final, Iterator, Sequence

from offset.tools.walk import PRUNE, walk

#: Where snapshots live: a directory *beside* the workspace, never inside it.
#: Inside would mean the fallback copy walking into its own output and a
#: `cp -R` that refuses with "cannot copy a directory into itself"; beside
#: keeps the destination on the same filesystem, which every zero-cost backend
#: requires.
CONTAINER: Final = ".offset-fsnapshots"

#: Prefix for a generated snapshot directory name.  Includes the pid so two
#: agents sharing a workspace cannot collide on the same name.
PREFIX: Final = "snap-"

#: Prefix for probe scratch directories.  Dotted so it is invisible in a
#: listing during the fraction of a second it exists.
PROBE_PREFIX: Final = ".offset-probe-"

#: Seconds any backend command may take.  A snapshot is a metadata operation
#: and returns in milliseconds; a `cp -R` of a huge tree is the one case that
#: can legitimately run long, and it is the fallback's Python loop, not this.
#: A minute is therefore generous for everything that goes through here, and
#: bounded so a wedged `zfs` cannot hang the agent for ever.
COMMAND_TIMEOUT: Final = 60.0

#: Exit code reported when the tool is not installed at all, matching the
#: shell's convention so callers need not distinguish "absent" from "refused".
MISSING: Final = 127

#: Size of the reflink probe file.  Not a token 6 bytes: btrfs stores a very
#: small file inline in its metadata, where the clone ioctl has historically
#: been refused, so a tiny probe would report "no reflink" on a filesystem that
#: has it.  64 KiB is past any inlining threshold and still one page-aligned
#: write.
PROBE_BYTES: Final = 65_536


class SnapshotError(RuntimeError):
    """The caller asked for something that cannot or must not be done."""


class BackendError(RuntimeError):
    """One backend declined.  Always caught: the next candidate is tried."""


# -- running commands -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Completed:
    """A finished command, reduced to the two things any backend needs."""

    code: int
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0


#: A command runner.  One argument, no shell, no cwd: every path a backend
#: passes is already absolute, and keeping the signature this narrow is what
#: makes the btrfs/zfs/APFS paths testable on a machine that has none of them.
Runner = Callable[[Sequence[str]], Completed]


def system_runner(args: Sequence[str]) -> Completed:
    """Run a real command.  A missing tool is an exit code, not an exception,
    because "btrfs is not installed" and "btrfs said no" lead to the same
    place: try the next backend."""
    argv = [str(a) for a in args]
    try:
        got = subprocess.run(
            argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT, errors="replace"
        )
    except FileNotFoundError:
        return Completed(MISSING, f"{argv[0]}: not installed")
    except (OSError, subprocess.SubprocessError) as exc:
        # A timeout or a fork failure is indistinguishable from absence as far
        # as backend selection is concerned, and must not escape.
        return Completed(MISSING, f"{argv[0]}: {exc}")
    return Completed(got.returncode, (got.stdout + got.stderr).strip())


@contextmanager
def _scratch(path: Path) -> Iterator[Path | None]:
    """A throwaway directory guaranteed to be on the same filesystem as `path`.

    Created *inside* `path` when it is a directory, because its parent may well
    be a different filesystem — the workspace being a mount point is the whole
    case this module exists for, and probing the parent would then measure the
    wrong volume.  Yields None when nothing can be created there, which makes
    every backend correctly report itself unusable.
    """
    base = path if path.is_dir() else path.parent
    try:
        made = Path(tempfile.mkdtemp(prefix=PROBE_PREFIX, dir=base))
    except OSError:
        yield None
        return
    try:
        yield made
    finally:
        # `ignore_errors` matters: a btrfs subvolume the probe could not delete
        # is a directory `rmtree` cannot remove either, and a failed cleanup
        # must not turn a capability probe into an exception.
        shutil.rmtree(made, ignore_errors=True)


# -- the backend interface --------------------------------------------------


@dataclass(frozen=True, slots=True)
class Made:
    """What a backend produced.

    `handle` is whatever the backend needs handed back at teardown and cannot
    recompute from the path — the zfs snapshot tag, for instance.  Keeping it
    on the returned record rather than in a dict on the backend means a cached
    backend instance carries no per-snapshot state to leak.
    """

    bytes_copied: int = 0
    handle: str = ""


class Backend(ABC):
    """One way of making a private copy of a directory."""

    #: Shown to the user, and the stable name tests and settings refer to.
    name: ClassVar[str] = ""

    #: True when creation is a metadata operation whose cost does not grow with
    #: the size of the workspace.
    instant: ClassVar[bool] = True

    __slots__ = ("run",)

    def __init__(self, run: Runner | None = None) -> None:
        self.run = run if run is not None else system_runner

    @abstractmethod
    def usable(self, path: Path) -> bool:
        """Whether this really works on `path`'s filesystem, by trying it."""

    @abstractmethod
    def create(self, source: Path, dest: Path) -> Made:
        """Make `dest` a private copy of `source`, or raise `BackendError`."""

    def destroy(self, path: Path, handle: str = "") -> None:
        """Tear it down.  Only ever called through the guard in `_purge`."""
        shutil.rmtree(path, ignore_errors=True)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"<{type(self).__name__} instant={self.instant}>"


# -- btrfs ------------------------------------------------------------------


class Btrfs(Backend):
    """`btrfs subvolume snapshot`: constant time, writable, no data copied."""

    name = "btrfs"
    instant = True

    __slots__ = ()

    def usable(self, path: Path) -> bool:
        # `subvolume show` is a read and gates everything else.  Without it the
        # probe would create a subvolume on every non-btrfs machine only to
        # watch the create fail, and on a btrfs machine whose workspace is a
        # plain directory it would pass a probe that the real snapshot then
        # fails — a snapshot source must itself be a subvolume.
        if not self.run(["btrfs", "subvolume", "show", str(path)]).ok:
            return False
        with _scratch(path) as scratch:
            if scratch is None:
                return False
            source = scratch / "probe"
            if not self.run(["btrfs", "subvolume", "create", str(source)]).ok:
                return False
            clone = scratch / "probe-snapshot"
            ok = self.run(["btrfs", "subvolume", "snapshot", str(source), str(clone)]).ok
            # Deleting is the privilege we are really testing: a snapshot we
            # cannot remove is worse than no snapshot, since `release()` would
            # leave the user's disk filling up one branch at a time.
            removed = self.run(["btrfs", "subvolume", "delete", str(clone)]).ok if ok else True
            removed = self.run(["btrfs", "subvolume", "delete", str(source)]).ok and removed
            return ok and removed

    def create(self, source: Path, dest: Path) -> Made:
        got = self.run(["btrfs", "subvolume", "snapshot", str(source), str(dest)])
        if not got.ok:
            raise BackendError(f"btrfs: {got.output or 'the snapshot was refused'}")
        return Made()

    def destroy(self, path: Path, handle: str = "") -> None:
        if not self.run(["btrfs", "subvolume", "delete", str(path)]).ok:
            # Not a subvolume after all, or the privilege went away: the tree
            # is still a directory, and removing it recovers the space.
            shutil.rmtree(path, ignore_errors=True)


# -- zfs --------------------------------------------------------------------


class Zfs(Backend):
    """`zfs snapshot` plus `zfs clone`, mounted where the caller wants it."""

    name = "zfs"
    instant = True

    __slots__ = ()

    def _dataset(self, path: Path) -> str | None:
        """The dataset `path` *is*, not the one that contains it.

        Deliberately strict: `zfs list` answers for a mountpoint and errors for
        a subdirectory of one, and cloning the enclosing dataset would hand the
        caller a copy of far more than the workspace.
        """
        got = self.run(["zfs", "list", "-H", "-o", "name", str(path)])
        if not got.ok:
            return None
        first = got.output.splitlines()[0].strip() if got.output else ""
        return first or None

    def usable(self, path: Path) -> bool:
        return self._dataset(path) is not None

    def create(self, source: Path, dest: Path) -> Made:
        dataset = self._dataset(source)
        if dataset is None:
            raise BackendError("zfs: the workspace is not a dataset mountpoint")
        tag = f"{dataset}@{PREFIX}{dest.name}"
        got = self.run(["zfs", "snapshot", tag])
        if not got.ok:
            raise BackendError(f"zfs: {got.output or 'snapshot refused'}")
        # A clone is a dataset and needs a dataset name.  A sibling of the
        # source keeps the snapshot out of the workspace's own mountpoint; a
        # pool root has no sibling, so it takes a child instead.
        parent = dataset.rsplit("/", 1)[0] if "/" in dataset else dataset
        clone = f"{parent}/{PREFIX}{dest.name}"
        got = self.run(["zfs", "clone", "-o", f"mountpoint={dest}", tag, clone])
        if not got.ok:
            # Leave nothing behind: an orphaned snapshot pins the blocks of
            # every file the workspace later deletes.
            self.run(["zfs", "destroy", tag])
            raise BackendError(f"zfs: {got.output or 'clone refused'}")
        return Made(handle=tag)

    def destroy(self, path: Path, handle: str = "") -> None:
        if handle:
            # `-R` takes the dependent clone with it, which is exactly the
            # relationship we created, so one command undoes both.
            self.run(["zfs", "destroy", "-R", handle])
        if path.is_dir():
            # The mountpoint directory may survive the unmount.  Only the empty
            # husk is removed; anything still in it means the destroy failed and
            # deleting the contents by hand would delete the clone's data.
            with suppress(OSError):
                path.rmdir()


# -- APFS -------------------------------------------------------------------


class Apfs(Backend):
    """`cp -c`, which is `clonefile(2)` on APFS: copy-on-write per file."""

    name = "apfs"
    instant = True

    __slots__ = ()

    def usable(self, path: Path) -> bool:
        with _scratch(path) as scratch:
            if scratch is None:
                return False
            source = scratch / "probe"
            try:
                source.write_bytes(b"\0" * PROBE_BYTES)
            except OSError:
                return False
            # GNU cp has no `-c` at all and exits non-zero on the unknown
            # option, so this probe answers "no" on Linux without needing to
            # know it is Linux.
            return self.run(["cp", "-c", str(source), str(scratch / "clone")]).ok

    def create(self, source: Path, dest: Path) -> Made:
        got = self.run(["cp", "-Rc", str(source), str(dest)])
        if not got.ok:
            raise BackendError(f"apfs: {got.output or 'clonefile refused'}")
        return Made()


# -- reflink ----------------------------------------------------------------


class Reflink(Backend):
    """`cp --reflink=always`: shared extents on btrfs and reflink-capable XFS.

    `always`, never `auto`: `auto` silently performs a full copy when the clone
    is refused, which would report a zero-cost snapshot that quietly wrote the
    whole workspace — the exact dishonesty this module is built to avoid.
    """

    name = "reflink"
    instant = True

    __slots__ = ()

    def usable(self, path: Path) -> bool:
        with _scratch(path) as scratch:
            if scratch is None:
                return False
            source = scratch / "probe"
            try:
                source.write_bytes(b"\0" * PROBE_BYTES)
            except OSError:
                return False
            return self.run(["cp", "--reflink=always", str(source), str(scratch / "clone")]).ok

    def create(self, source: Path, dest: Path) -> Made:
        got = self.run(["cp", "-a", "--reflink=always", str(source), str(dest)])
        if not got.ok:
            raise BackendError(f"reflink: {got.output or 'clone refused'}")
        return Made()


# -- the fallback -----------------------------------------------------------


class Copy(Backend):
    """A real recursive copy.  Always works, and never pretends to be free."""

    name = "copy"
    instant = False

    __slots__ = ()

    def usable(self, path: Path) -> bool:
        # No probe: a copy needs nothing the caller does not already have, and
        # this backend is the reason `detect` can promise a non-empty answer.
        return True

    def create(self, source: Path, dest: Path) -> Made:
        try:
            dest.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise BackendError(f"copy: {exc}") from exc
        written = 0
        for found in walk(source, respect_gitignore=True, prune=PRUNE):
            relative = found.relative_to(source)
            target = dest / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                written += _clone_one(found, target)
            except OSError:
                # One unreadable file — a dangling symlink, a socket, a mode
                # the agent cannot read — must not abandon the whole snapshot.
                # The tree is still usable and the byte count stays honest.
                continue
        return Made(bytes_copied=written)


def _clone_one(source: Path, target: Path) -> int:
    """Copy one entry, preserving links.  Returns the bytes actually written.

    A symlink is recreated as a symlink rather than followed: following turns a
    single link into a second copy of its target, and a link pointing outside
    the workspace would drag unrelated data into the snapshot.
    """
    if source.is_symlink():
        target.symlink_to(os.readlink(source))
        return 0
    shutil.copy2(source, target, follow_symlinks=False)
    return source.stat().st_size


# -- probing ----------------------------------------------------------------

#: Probe order: cheapest and most complete first, the honest copy last.
BACKENDS: Final[tuple[type[Backend], ...]] = (Btrfs, Zfs, Apfs, Reflink, Copy)

#: device id (or path, when it cannot be stat'ed) and runner -> what works.
#: The runner is part of the key so an injected fake never reads a real
#: machine's cached answer, and a real probe never reads a fake's.
_PROBED: dict[tuple[Any, Runner], tuple[Backend, ...]] = {}

_LOCK = threading.Lock()


def clear_probe_cache() -> None:
    """Forget every probe.  For tests, and for after a mount changes."""
    with _LOCK:
        _PROBED.clear()


def _device(path: Path) -> Any:
    """The cache key for `path`'s filesystem.

    Capability belongs to a device, not a directory, so one probe answers for
    every workspace on the same volume.  A path that cannot be stat'ed falls
    back to its own name rather than to a shared sentinel, so an unreadable
    directory cannot poison the entry for a real one.
    """
    try:
        return os.stat(path).st_dev
    except OSError:
        return str(path)


def available_backends(
    path: str | os.PathLike[str] | None = None, *, runner: Runner | None = None
) -> list[Backend]:
    """Every backend that genuinely works on `path`, best first.

    Never empty: the copy fallback needs no support from anything.
    """
    where = Path(path).expanduser().resolve() if path is not None else Path.cwd()
    key = (_device(where), runner if runner is not None else system_runner)
    with _LOCK:
        cached = _PROBED.get(key)
    if cached is not None:
        return list(cached)
    # Probed outside the lock: a probe forks and a duplicate one on two threads
    # racing the same fresh device is wasteful but harmless, whereas holding
    # the lock across five subprocesses would stall every other snapshot.
    found = tuple(made for cls in BACKENDS if (made := cls(runner)).usable(where))
    with _LOCK:
        _PROBED[key] = found
    return list(found)


def detect(
    path: str | os.PathLike[str] | None = None, *, runner: Runner | None = None
) -> Backend:
    """The best backend for `path`.  The answer for a device is probed once."""
    return available_backends(path, runner=runner)[0]


# -- the handle -------------------------------------------------------------


def _refusal(path: Path, origin: Path) -> str | None:
    """Why deleting `path` would be a catastrophe, or None if it is safe.

    This is the guard that stands between a bug in snapshot naming and the
    user's actual work.  It is checked before creating and again before
    releasing, because the two happen at different times and a caller may have
    built the handle itself.
    """
    try:
        here = Path(path).expanduser().resolve()
        source = Path(origin).expanduser().resolve()
    except OSError:
        return "the path cannot be resolved"
    if here == source:
        return "it is the workspace itself"
    if here in source.parents:
        return "it contains the workspace"
    if here == Path(here.anchor):
        return "it is a filesystem root"
    with suppress(RuntimeError, OSError):
        if here == Path.home():
            return "it is the home directory"
    return None


def _purge(path: Path, origin: Path, backend: Backend, handle: str = "") -> None:
    """Remove a snapshot, or refuse to.  The only route to `Backend.destroy`."""
    refusal = _refusal(path, origin)
    if refusal is not None:
        raise SnapshotError(f"refusing to delete {path}: {refusal}")
    try:
        backend.destroy(path, handle)
    except OSError:
        # Teardown is best-effort by nature: the caller is usually already
        # unwinding, and a leftover directory is a disk-space problem, not a
        # reason to replace whatever exception brought us here.
        pass


@dataclass(slots=True)
class Snapshot:
    """A private, writable copy of a workspace, and how much it cost."""

    path: Path
    backend: Backend
    origin: Path
    instant: bool = True
    bytes_copied: int = 0
    handle: str = ""
    #: Backends that were tried and declined, in order.  Kept because "why is
    #: this not instant" is the first question a user asks.
    declined: tuple[str, ...] = ()
    released: bool = False

    def release(self) -> bool:
        """Tear the snapshot down.  True if this call is the one that did it.

        Idempotent, so a context manager and an explicit `release()` in the
        same block do not fight.  Refuses, by raising, to delete anything that
        is or contains the original workspace.
        """
        if self.released:
            return False
        _purge(self.path, self.origin, self.backend, self.handle)
        # Set only after the guard, so a refused release keeps refusing rather
        # than quietly reporting the snapshot as gone.
        self.released = True
        return True

    def describe(self) -> str:
        """One line for the UI, saying plainly what the snapshot cost."""
        if self.instant:
            return f"{self.backend.name} snapshot: instant, no data copied"
        return f"{self.backend.name}: {_human(self.bytes_copied)} copied — not free"

    def __enter__(self) -> Snapshot:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def __repr__(self) -> str:
        state = "released" if self.released else "live"
        return f"<Snapshot {self.path.name} {self.backend.name} {state}>"


def _human(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    scaled = float(size)
    for unit in ("KiB", "MiB", "GiB"):
        scaled /= 1024.0
        if scaled < 1024.0 or unit == "GiB":
            return f"{scaled:.1f} {unit}"
    return f"{size} B"  # unreachable: the loop always returns at GiB


# -- taking one -------------------------------------------------------------


def snapshot(
    workspace: str | os.PathLike[str],
    *,
    container: str | os.PathLike[str] | None = None,
    name: str | None = None,
    runner: Runner | None = None,
) -> Snapshot:
    """A private copy of `workspace`, by the cheapest means that actually works.

    Backends are tried in probe order and a failure falls through to the next,
    so a btrfs workspace whose user cannot delete subvolumes still gets a
    snapshot — a slower one, honestly labelled.
    """
    source = Path(workspace).expanduser().resolve()
    if not source.is_dir():
        raise SnapshotError(f"not a directory: {source}")
    box = Path(container).expanduser().resolve() if container is not None else source.parent / CONTAINER
    dest = box / (name or f"{PREFIX}{os.getpid()}-{uuid.uuid4().hex[:8]}")
    refusal = _refusal(dest, source)
    if refusal is not None:
        # Checked before anything is created: a destination we would refuse to
        # delete must never become a Snapshot, or `__exit__` would raise while
        # unwinding somebody else's exception.
        raise SnapshotError(f"refusing to snapshot into {dest}: {refusal}")
    try:
        box.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SnapshotError(f"cannot create {box}: {exc}") from exc

    declined: list[str] = []
    for backend in available_backends(source, runner=runner):
        try:
            made = backend.create(source, dest)
        except BackendError as exc:
            declined.append(str(exc))
            _purge(dest, source, backend)
            continue
        if not dest.is_dir():
            # A command that exits 0 and produces nothing is a broken backend,
            # not a snapshot.  Trusting the exit code alone once handed callers
            # a `.path` that did not exist.
            declined.append(f"{backend.name}: reported success but created nothing")
            _purge(dest, source, backend, made.handle)
            continue
        return Snapshot(
            path=dest,
            backend=backend,
            origin=source,
            instant=backend.instant,
            bytes_copied=made.bytes_copied,
            handle=made.handle,
            declined=tuple(declined),
        )
    raise SnapshotError("; ".join(declined) or f"no backend could snapshot {source}")


__all__ = [
    "BACKENDS",
    "Apfs",
    "Backend",
    "BackendError",
    "Btrfs",
    "Completed",
    "Copy",
    "Made",
    "Reflink",
    "Runner",
    "Snapshot",
    "SnapshotError",
    "Zfs",
    "available_backends",
    "clear_probe_cache",
    "detect",
    "snapshot",
    "system_runner",
]
