"""The editor bridge: one local socket, many editors, no framework.

offset is a terminal program and stays one.  An editor extension is a *view* of
a running agent, not a second agent, so the integration is a socket rather than
a rewrite: the shell keeps ownership of the session, the toolbox and the model,
and the bridge publishes what it is doing to whoever local asks.

Three decisions carry the design.

*The socket is a capability, so it is authenticated.*  `apply_edit` rewrites
files in the workspace, which makes an unauthenticated local socket a real
vulnerability rather than a theoretical one: on a shared machine any process
that can reach the path could rewrite the user's source.  A token generated per
run is written `0o600` beside the socket, the socket itself is created `0o600`,
and a connection that does not present the token in its first frame is answered
with one error and closed.  Two locks on one door, because the filesystem
permission alone is lost the moment the path is bind-mounted or the home
directory is group-writable.

*A dead editor must not stall the agent.*  Every client gets its own reader
thread, its own writer thread and its own bounded queue.  Publishing an event
is `put_nowait` into each queue and nothing else; a client that has stopped
reading fills its queue and is then dropped.  Backpressure is deliberately not
propagated — the alternative is an agent that pauses mid-turn because somebody
closed a laptop lid.

*A crash must not lock the user out.*  A socket file outlives the process that
created it, so a path that already exists is probed by connecting to it before
anything is unlinked: a live bridge is reported and left strictly alone, a dead
one is replaced.  Deleting first and asking later would let a second shell
silently steal the first shell's editors.

The framing is the same discipline as the MCP client (`offset/tools/mcp`):
newline-delimited JSON-RPC 2.0, requests carrying an `id`, events sent as
notifications without one.  Nothing here speaks HTTP, and nothing here imports
anything that is not stdlib.

Domain failures are results, not JSON-RPC errors: `apply_edit` on a path
outside the workspace answers `{"ok": false, "error": ...}`.  The `error`
member is reserved for protocol faults — bad JSON, unknown method, missing
token — so an editor can tell "you asked wrongly" from "that could not be
done".
"""

from __future__ import annotations

import difflib
import json
import os
import queue
import secrets
import socket
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final

from offset.core import settings
from offset.core.agent import Finished, StepStarted, ToolFinished, ToolStarted
from offset.core.entries import new_id
from offset.core.session import Session
from offset.providers.base import StreamError

#: JSON-RPC version string every frame carries.
PROTOCOL: Final = "2.0"

#: Bumped when the wire contract changes in a way an old extension would
#: misread.  The extension refuses to speak a version it does not know.
BRIDGE_VERSION: Final = "1"

SOCKET_NAME: Final = "bridge.sock"
TOKEN_NAME: Final = "bridge.token"
DESCRIPTOR_NAME: Final = "bridge.json"

#: Poll slice for the accept and writer loops: the granularity at which
#: `shutdown()` is noticed.
_TICK: Final = 0.1

#: Events one client may fall behind by before it is dropped.  Large enough
#: that a busy turn does not evict an editor that is merely repainting, small
#: enough that a dead editor cannot pin megabytes of turn history.
QUEUE_LIMIT: Final = 256

#: Seconds a connection may stay silent before it must have authenticated.
AUTH_GRACE: Final = 10.0

#: Longest single frame accepted.  Without a ceiling a client that never sends
#: a newline would grow the read buffer until the process died.
MAX_FRAME: Final = 4 * 1024 * 1024

#: Both the send deadline and the recv poll slice.  A blocked `sendall` is how
#: a client that stopped reading is finally noticed when its queue happens not
#: to have filled yet.
SEND_TIMEOUT: Final = 5.0

#: Concurrent editors.  A local bridge with more than this many clients is a
#: runaway reconnect loop, not a user with a lot of windows.
MAX_CLIENTS: Final = 32

#: Largest file the diff view will carry both versions of.  Beyond this the
#: change is still reported, with its texts omitted and `truncated` set.
MAX_DIFF_BYTES: Final = 512 * 1024

#: Files reported in one `diff` call.  A rebase gone wrong can dirty thousands;
#: an editor cannot show them and `git show` per file would take minutes.
MAX_CHANGES: Final = 200

#: git's hash of the empty tree.  Diffing against it is how a repository with
#: no commits yet is handled without a special case.
EMPTY_TREE: Final = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

PARSE_ERROR: Final = -32700
INVALID_REQUEST: Final = -32600
METHOD_NOT_FOUND: Final = -32601
INVALID_PARAMS: Final = -32602
INTERNAL_ERROR: Final = -32603
#: Outside the JSON-RPC reserved range, as the spec requires for application
#: errors.  The extension keys its "not authorised" state on exactly this.
UNAUTHENTICATED: Final = -32001

#: Every notification the bridge will ever push.  Declared so the extension can
#: be written against a closed set rather than against whatever it happens to
#: observe.
EVENTS: Final = (
    "agent.started",
    "agent.finished",
    "tool.started",
    "tool.finished",
    "edit.applied",
    "job.state",
)

#: Methods a client may call once authenticated.  `hello` is answered before
#: authentication and is therefore not in the dispatch table.
METHODS: Final = ("status", "sessions", "diff", "apply_edit", "cancel", "prompt")

#: Built-in tools that change files.  Only consulted when no toolbox was
#: supplied; a wired-in shell passes a `writes` hook that reads `Danger`
#: instead, which also covers custom and MCP tools.
_WRITE_TOOLS: Final = frozenset({"write", "edit"})

_STATUS: Final = {
    "M": "modified",
    "A": "added",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "T": "typechange",
    "U": "unmerged",
}


# -- pending changes --------------------------------------------------------
#
# The editor's diff view needs three things per file: a status, a unified diff,
# and both texts so `vscode.diff` can show them side by side.  Both texts are
# fetched anyway, so the unified diff is computed from them with `difflib`
# rather than by shelling out a second time: one subprocess per file instead of
# two, and the diff shown is guaranteed to be the diff between the two buffers
# the editor is displaying rather than a differently-configured git rendering.


def _git(args: list[str], cwd: Path | str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout, errors="replace"
    )


@dataclass(slots=True)
class Change:
    """One file that differs from the base commit."""

    path: str
    status: str
    additions: int = 0
    deletions: int = 0
    diff: str = ""
    original: str = ""
    current: str = ""
    #: Set when the file was too large or not decodable, in which case
    #: `original`, `current` and `diff` are empty and the editor shows counts
    #: only.  A silently empty diff would read as "no change".
    truncated: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
            "diff": self.diff,
            "original": self.original,
            "current": self.current,
            "truncated": self.truncated,
        }

    def report(self) -> list[str]:
        counts = f"+{self.additions} -{self.deletions}" if not self.truncated else "binary or oversized"
        return [f"{self.status:<10} {self.path}  ({counts})"]


def _decode(raw: bytes) -> str | None:
    """Text, or None when the bytes are not UTF-8 — which is how a binary file
    is detected without guessing from the extension."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _count(field_text: str) -> int:
    """A numstat column.  git writes `-` for a binary file."""
    try:
        return int(field_text)
    except ValueError:
        return 0


def _numstat(raw: str) -> dict[str, tuple[int, int]]:
    """`git diff --numstat -z`: `adds\\tdels\\tpath\\0`, but a rename writes an
    empty path followed by two more NUL-separated fields."""
    out: dict[str, tuple[int, int]] = {}
    fields = raw.split("\0")
    i = 0
    while i < len(fields):
        head = fields[i]
        i += 1
        if "\t" not in head:
            continue
        parts = head.split("\t")
        if len(parts) < 3:
            continue
        adds, dels, path = parts[0], parts[1], "\t".join(parts[2:])
        if not path and i + 1 < len(fields):
            path = fields[i + 1]  # rename: old then new; the new path is the file
            i += 2
        if path:
            out[path] = (_count(adds), _count(dels))
    return out


def _name_status(raw: str) -> dict[str, str]:
    """`git diff --name-status -z`: `code\\0path\\0`, renames adding a field."""
    out: dict[str, str] = {}
    fields = raw.split("\0")
    i = 0
    while i < len(fields):
        code = fields[i]
        i += 1
        if not code or i >= len(fields):
            continue
        path = fields[i]
        i += 1
        if code[0] in "RC":
            if i >= len(fields):
                break
            path = fields[i]
            i += 1
        if path:
            out[path] = _STATUS.get(code[0], "modified")
    return out


def pending_changes(
    root: str | os.PathLike[str], *, limit: int = MAX_DIFF_BYTES, cap: int = MAX_CHANGES
) -> tuple[list[Change], str | None]:
    """Everything in `root` that differs from HEAD, plus untracked files.

    Returns the changes and, separately, a reason the list may be empty.  A
    workspace that is not a git repository is not an error the caller should
    raise on — it is an explanation the editor shows in place of the diff list.
    """
    root = Path(root)
    if not root.is_dir():
        return [], f"{root} is not a directory"
    inside = _git(["rev-parse", "--is-inside-work-tree"], root)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return [], f"{root} is not inside a git work tree, so there is nothing to diff against"

    has_head = _git(["rev-parse", "--verify", "-q", "HEAD"], root).returncode == 0
    base = "HEAD" if has_head else EMPTY_TREE

    counts = _numstat(_git(["diff", base, "--numstat", "-z"], root).stdout)
    statuses = _name_status(_git(["diff", base, "--name-status", "-z"], root).stdout)
    untracked = [p for p in _git(["ls-files", "--others", "--exclude-standard", "-z"], root).stdout.split("\0") if p]
    for path in untracked:
        statuses.setdefault(path, "untracked")

    ordered = sorted(statuses)  # stable so a refresh does not reshuffle the list
    changes: list[Change] = []
    for path in ordered[:cap]:
        status = statuses[path]
        adds, dels = counts.get(path, (0, 0))
        change = Change(path=path, status=status, additions=adds, deletions=dels)

        target = root / path
        current: str | None = ""
        if status != "deleted":
            try:
                if target.is_file() and target.stat().st_size <= limit:
                    current = _decode(target.read_bytes())
                elif target.is_file():
                    current = None
            except OSError:
                current = None

        original = ""
        if has_head and status not in ("added", "untracked"):
            shown = _git(["show", f"{base}:{path}"], root)
            original = shown.stdout if shown.returncode == 0 else ""
            if len(original.encode("utf-8", "replace")) > limit:
                current = None

        if current is None:
            change.truncated = True
        else:
            change.original = original
            change.current = current
            change.diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                )
            )
            if status == "untracked":
                change.additions = current.count("\n") + (0 if current.endswith("\n") or not current else 1)
        changes.append(change)

    note = None
    if len(ordered) > cap:
        note = f"{len(ordered)} files changed; showing the first {cap}"
    return changes, note


# -- writing a file ---------------------------------------------------------


def apply_text(root: Path, rel: str, text: str) -> tuple[bool, str]:
    """Replace a workspace file's contents atomically.

    Written through `mkstemp` in the same directory then `os.replace`, so an
    editor that dies mid-write leaves the old file rather than half the new
    one.  The existing mode is carried over: a rewritten script that lost its
    executable bit is a bug report, and `mkstemp` creates `0o600`.
    """
    root = root.resolve()
    candidate = Path(rel)
    target = candidate if candidate.is_absolute() else root / candidate
    try:
        target = target.resolve()
    except OSError as exc:
        return False, f"{rel}: {type(exc).__name__}: {exc}"
    if target != root and not target.is_relative_to(root):
        return False, f"{rel} is outside the workspace {root}"
    if target.is_dir():
        return False, f"{rel} is a directory"

    try:
        mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else 0o644
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".offset-bridge-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            os.chmod(tmp, mode)
            os.replace(tmp, target)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        return False, f"{rel}: {type(exc).__name__}: {exc}"
    try:
        shown = str(target.relative_to(root))
    except ValueError:
        shown = str(target)
    return True, f"wrote {len(text)} characters to {shown}"


# -- what the bridge needs from a shell -------------------------------------


def _no_agent() -> tuple[bool, str]:
    return False, "no agent is attached to this bridge"


def _default_status() -> dict[str, Any]:
    return {"model": "", "session": "", "state": "detached"}


def _default_sessions() -> list[dict[str, Any]]:
    root = settings.home() / "sessions"
    return [
        {
            "id": info.id,
            "path": str(info.path),
            "mtime": info.mtime,
            "messages": info.messages,
            "title": info.first_line,
            "size": info.size,
            "skipped": info.skipped,
        }
        for info in Session.list(root)
    ]


def _default_writes(name: str) -> bool:
    return name in _WRITE_TOOLS


@dataclass(slots=True)
class Hooks:
    """Everything the bridge needs from a running shell, injected rather than
    imported, so the server can be started — and tested — on its own.

    Every default is a real answer rather than a placeholder: an unattached
    bridge reports honestly that it has no agent instead of pretending to have
    one, and still serves `sessions` and `diff`, which need no agent at all.
    """

    workspace: Path = field(default_factory=Path.cwd)
    #: Model, session id and coarse state.  The bridge overwrites `state` while
    #: it is itself driving a turn, because it knows that and the shell does not.
    status: Callable[[], dict[str, Any]] = _default_status
    #: Newest-first session metadata for the tree view.
    sessions: Callable[[], list[dict[str, Any]]] = _default_sessions
    #: Background jobs, in whatever shape their manager reports.  Each entry is
    #: expected to carry `id` and `state`; nothing else is read.
    jobs: Callable[[], list[dict[str, Any]]] = list
    #: Pending workspace changes for a given root.
    diff: Callable[[Path], tuple[list[Change], str | None]] = pending_changes
    #: Replace a file's contents; `(ok, message)`.
    apply: Callable[[Path, str, str], tuple[bool, str]] = apply_text
    #: Stop the running turn; `(ok, message)`.
    cancel: Callable[[], tuple[bool, str]] = _no_agent
    #: Run one turn to completion and return its reply text; `(ok, text)`.
    #: Blocking on purpose — the bridge owns the threading, not the hook.
    prompt: Callable[[str], tuple[bool, str]] = _no_agent
    #: Whether a finished tool call could have changed files, which is what
    #: tells the editor to refresh its diff view.
    writes: Callable[[str], bool] = _default_writes


# -- wire -------------------------------------------------------------------


def _encode(frame: dict[str, Any]) -> bytes:
    """One frame, one line.  `default=str` because a tool argument that arrived
    as JSON can still hold a Path by the time it is echoed back, and losing the
    whole event to a serialisation error would be worse than losing its type."""
    return json.dumps(frame, ensure_ascii=False, default=str).encode("utf-8") + b"\n"


def _reply(ident: Any, result: Any) -> bytes:
    return _encode({"jsonrpc": PROTOCOL, "id": ident, "result": result})


def _failure(ident: Any, code: int, message: str) -> bytes:
    return _encode({"jsonrpc": PROTOCOL, "id": ident, "error": {"code": code, "message": message}})


def _compose(text: str, selection: Any) -> str:
    """Fold an editor selection into the prompt the model actually sees."""
    if not isinstance(selection, dict):
        return text
    body = str(selection.get("text") or "")
    if not body:
        return text
    where = str(selection.get("path") or "the current file")
    start, end = selection.get("start_line"), selection.get("end_line")
    if isinstance(start, int) and isinstance(end, int):
        where = f"{where} lines {start}-{end}"
    return f"{text}\n\nSelection from {where}:\n\n{body}"


# -- one connected editor ---------------------------------------------------


class _Client:
    """A socket, a reader, a writer and a bounded queue between them.

    The queue is the whole point.  `publish` runs on the agent's thread and
    must never block, so it only ever puts; the writer thread is the only code
    that touches the socket for output, and it is allowed to be slow.
    """

    __slots__ = ("authenticated", "dropped", "id", "outbox", "reader", "since", "sock", "writer")

    def __init__(self, sock: socket.socket, limit: int) -> None:
        self.id = new_id()
        self.sock = sock
        self.outbox: queue.Queue[bytes | None] = queue.Queue(maxsize=max(1, limit))
        self.authenticated = False
        self.dropped: str | None = None
        self.since = time.time()
        self.reader: threading.Thread | None = None
        self.writer: threading.Thread | None = None

    def send(self, frame: bytes) -> bool:
        """Queue a frame.  False means the client is too far behind to keep."""
        if self.dropped is not None:
            return False
        try:
            self.outbox.put_nowait(frame)
        except queue.Full:
            return False
        return True

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "authenticated": self.authenticated,
            "queued": self.outbox.qsize(),
            "since": self.since,
            "dropped": self.dropped,
        }


# -- the server -------------------------------------------------------------


class Bridge:
    """A local JSON-RPC server publishing one agent to local editors."""

    __slots__ = (
        "_accept",
        "_busy",
        "_clients",
        "_lock",
        "_methods",
        "_owns_files",
        "_server",
        "_stop",
        "_turn",
        "descriptor_path",
        "dropped",
        "home",
        "hooks",
        "host",
        "listen",
        "port",
        "problems",
        "queue_limit",
        "send_timeout",
        "socket_path",
        "started",
        "token",
        "token_path",
    )

    def __init__(
        self,
        hooks: Hooks | None = None,
        *,
        home: str | os.PathLike[str] | None = None,
        queue_limit: int = QUEUE_LIMIT,
        send_timeout: float = SEND_TIMEOUT,
        listen: str = "",
    ) -> None:
        self.hooks = hooks or Hooks()
        #: Resolved now rather than at import, and never from a module-level
        #: constant: `--home` and the tests both move it after this file loads.
        self.home = Path(home) if home is not None else settings.home()
        self.socket_path = self.home / SOCKET_NAME
        self.token_path = self.home / TOKEN_NAME
        self.descriptor_path = self.home / DESCRIPTOR_NAME
        self.queue_limit = max(1, queue_limit)
        self.send_timeout = max(0.05, send_timeout)
        #: Empty means "a unix socket if this platform has them", which is the
        #: right answer for an editor on the same machine.  `tcp` or `host:port`
        #: forces a socket a remote client can reach - the daemon's reason to
        #: exist - and the token is what keeps that safe, so it is never
        #: optional.  Binding beyond loopback is a deliberate act, never a
        #: default.
        self.listen = listen.strip()
        self.token = ""
        self.host = ""
        self.port = 0
        self.started = 0.0
        #: Reasons a client was hung up on, counted rather than raised: an
        #: editor that fell behind is a fact about the editor, not a failure of
        #: the agent, but a bridge that drops everyone needs to be visible.
        self.dropped = 0
        self.problems: list[str] = []
        self._server: socket.socket | None = None
        self._accept: threading.Thread | None = None
        self._clients: dict[str, _Client] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        #: Held while a turn the bridge started is running.  Non-blocking
        #: acquisition is how a second editor is told "already busy" instead of
        #: being queued behind a turn it cannot see.
        self._turn = threading.Lock()
        self._busy = False
        self._owns_files = False
        self._methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "status": self._status,
            "sessions": self._sessions,
            "diff": self._diff,
            "apply_edit": self._apply_edit,
            "cancel": self._cancel,
            "prompt": self._prompt,
        }

    # -- lifecycle ----------------------------------------------------------

    @property
    def unix(self) -> bool:
        """Whether this bridge is on a unix domain socket.

        False on Windows, which has none with a filesystem mode, and false
        whenever a listen address was asked for: a remote client cannot reach a
        socket file on somebody else's machine.
        """
        return hasattr(socket, "AF_UNIX") and not self.listen

    @property
    def listening(self) -> bool:
        return self._server is not None and not self._stop.is_set()

    def serve(self) -> list[str]:
        """Bind, publish the descriptor, and accept in a daemon thread.

        Returns the reasons it could not start, empty on success.  A bridge is
        a convenience; a shell whose editor socket is unavailable must still
        run, so this never raises.
        """
        if self._server is not None:
            return []
        self.problems = []
        try:
            self.home.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.problems = [f"{self.home}: {type(exc).__name__}: {exc}"]
            return list(self.problems)

        server, problem = self._bind()
        if server is None:
            self.problems = [problem or "the bridge socket could not be bound"]
            return list(self.problems)

        self.token = secrets.token_urlsafe(32)
        problem = self._publish_descriptor()
        if problem:
            server.close()
            self.problems = [problem]
            return list(self.problems)

        self._server = server
        self._stop.clear()
        self.started = time.time()
        self._owns_files = True
        self._accept = threading.Thread(target=self._accept_loop, name="offset-bridge", daemon=True)
        self._accept.start()
        return []

    def _address(self) -> tuple[str, int]:
        """The host and port a TCP bridge should bind.

        `""` and `tcp` both mean loopback on an ephemeral port.  Anything else
        is `host`, `host:port` or `:port`, and a host that is not loopback is
        the caller deliberately exposing the agent to their network.
        """
        spec = self.listen
        if not spec or spec == "tcp":
            return "127.0.0.1", 0
        host, sep, port = spec.rpartition(":")
        if not sep:
            return spec, 0
        try:
            return (host or "127.0.0.1"), int(port)
        except ValueError:
            # `[::1]` and other bracketless colons: treat the whole thing as a
            # host rather than silently binding a port nobody asked for.
            return spec, 0

    def _bind(self) -> tuple[socket.socket | None, str | None]:
        if not self.unix:
            host, port = self._address()
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            server = socket.socket(family, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                server.bind((host, port))
                server.listen(MAX_CLIENTS)
            except OSError as exc:
                server.close()
                return None, f"{host}:{port}: {type(exc).__name__}: {exc}"
            bound = server.getsockname()
            self.host, self.port = host, int(bound[1])
            server.settimeout(_TICK)
            return server, None

        path = str(self.socket_path)
        # 108 on Linux, 104 on BSD; both are the size of sockaddr_un.sun_path.
        if len(path.encode("utf-8")) >= 104:
            return None, (
                f"{path} is too long for a unix socket path; "
                "set OFFSET_HOME to a shorter directory"
            )
        if self.socket_path.exists():
            if _alive(self.socket_path):
                return None, f"another offset is already listening on {path}"
            # Stale: the process that made it is gone, so the file is rubbish.
            try:
                self.socket_path.unlink()
            except OSError as exc:
                return None, f"{path} is stale but could not be removed: {exc}"

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Created under a restrictive umask rather than chmod'ed afterwards:
        # between bind and chmod the socket is connectable by anyone.
        previous = os.umask(0o177)
        try:
            server.bind(path)
            server.listen(MAX_CLIENTS)
        except OSError as exc:
            server.close()
            return None, f"{path}: {type(exc).__name__}: {exc}"
        finally:
            os.umask(previous)
        try:
            os.chmod(path, 0o600)  # belt and braces: some filesystems ignore umask
        except OSError:
            pass
        server.settimeout(_TICK)
        return server, None

    def _publish_descriptor(self) -> str | None:
        """Write the token and the discovery file, both `0o600`."""
        descriptor = {
            "version": BRIDGE_VERSION,
            "protocol": PROTOCOL,
            "transport": "unix" if self.unix else "tcp",
            "path": str(self.socket_path) if self.unix else "",
            "host": "" if self.unix else (self.host or "127.0.0.1"),
            "port": self.port,
            "token_path": str(self.token_path),
            "pid": os.getpid(),
            "started": time.time(),
            "events": list(EVENTS),
            "methods": list(METHODS),
        }
        for target, body in ((self.token_path, self.token), (self.descriptor_path, json.dumps(descriptor, indent=2))):
            problem = _write_secret(target, body)
            if problem:
                return problem
        return None

    def shutdown(self) -> None:
        """Stop accepting, hang up on everyone, remove the published files."""
        self._stop.set()
        server, self._server = self._server, None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        for client in self._snapshot():
            self._retire(client, "bridge shutting down")
        accept, self._accept = self._accept, None
        if accept is not None and accept is not threading.current_thread():
            accept.join(timeout=2.0)
        if self._owns_files:
            self._owns_files = False
            for path in (self.socket_path, self.token_path, self.descriptor_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self.token = ""

    def report(self) -> list[str]:
        """Human-facing state, for `/bridge` or a startup notice."""
        if self.problems:
            return ["editor bridge: not running"] + [f"  {p}" for p in self.problems]
        if not self.listening:
            return ["editor bridge: not started"]
        where = str(self.socket_path) if self.unix else f"127.0.0.1:{self.port}"
        clients = self._snapshot()
        lines = [f"editor bridge: listening on {where}", f"  token: {self.token_path}"]
        lines.append(f"  clients: {sum(1 for c in clients if c.authenticated)} attached, {self.dropped} dropped")
        for client in clients:
            lines.append(f"  {client.id[-8:]} queued={client.outbox.qsize()} auth={client.authenticated}")
        return lines

    # -- accepting ----------------------------------------------------------

    def _accept_loop(self) -> None:
        server = self._server
        while not self._stop.is_set() and server is not None:
            try:
                conn, _addr = server.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(_TICK)
                continue
            self._adopt(conn)

    def _adopt(self, conn: socket.socket) -> None:
        client = _Client(conn, self.queue_limit)
        conn.settimeout(self.send_timeout)
        with self._lock:
            if len(self._clients) >= MAX_CLIENTS:
                over = True
            else:
                over = False
                self._clients[client.id] = client
        if over:
            # Answered rather than dropped silently, so a reconnect loop shows
            # the reason in the editor's log instead of looking like a crash.
            try:
                conn.sendall(_failure(None, INTERNAL_ERROR, f"the bridge already has {MAX_CLIENTS} clients"))
            except OSError:
                pass
            _hangup(conn)
            return
        client.writer = threading.Thread(target=self._write_loop, args=(client,), name=f"bridge-tx-{client.id[-6:]}", daemon=True)
        client.reader = threading.Thread(target=self._read_loop, args=(client,), name=f"bridge-rx-{client.id[-6:]}", daemon=True)
        client.writer.start()
        client.reader.start()

    def _snapshot(self) -> list[_Client]:
        with self._lock:
            return list(self._clients.values())

    def clients(self) -> list[dict[str, Any]]:
        """Connected editors, for `status` and `report`."""
        return [c.payload() for c in self._snapshot()]

    def _retire(self, client: _Client, reason: str) -> None:
        """Remove a client and unblock both of its threads.  Idempotent."""
        with self._lock:
            existing = self._clients.pop(client.id, None)
        if client.dropped is None:
            client.dropped = reason
        try:
            client.outbox.put_nowait(None)  # wake the writer even if it is idle
        except queue.Full:
            pass
        if existing is not None or client.dropped:
            _hangup(client.sock)

    def _drop(self, client: _Client, reason: str) -> None:
        if client.dropped is None:
            self.dropped += 1
        self._retire(client, reason)

    # -- reading ------------------------------------------------------------

    def _read_loop(self, client: _Client) -> None:
        buf = bytearray()
        deadline = time.monotonic() + AUTH_GRACE
        try:
            while not self._stop.is_set() and client.dropped is None:
                try:
                    chunk = client.sock.recv(65536)
                except TimeoutError:
                    if not client.authenticated and time.monotonic() > deadline:
                        self._reject(client, None, "no hello frame arrived; the bridge requires a token")
                        return
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buf += chunk
                if len(buf) > MAX_FRAME:
                    self._reject(client, None, f"a single frame exceeded {MAX_FRAME} bytes")
                    return
                while (nl := buf.find(b"\n")) >= 0:
                    line = bytes(buf[:nl])
                    del buf[: nl + 1]
                    if line.strip():
                        self._handle(client, line)
                    if client.dropped is not None:
                        return
        finally:
            self._retire(client, client.dropped or "client disconnected")

    def _handle(self, client: _Client, line: bytes) -> None:
        try:
            message = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            client.send(_failure(None, PARSE_ERROR, f"frame is not valid JSON: {exc}"))
            return
        if not isinstance(message, dict):
            client.send(_failure(None, INVALID_REQUEST, "every frame must be a JSON object"))
            return

        ident = message.get("id")
        method = message.get("method")
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(method, str) or not method:
            client.send(_failure(ident, INVALID_REQUEST, "every request needs a string 'method'"))
            return
        if not isinstance(params, dict):
            client.send(_failure(ident, INVALID_PARAMS, f"params for {method} must be a JSON object"))
            return

        if not client.authenticated:
            if method != "hello":
                self._reject(client, ident, f"{method} was sent before hello; authenticate first")
                return
            offered = params.get("token")
            if not isinstance(offered, str) or not secrets.compare_digest(offered, self.token or new_id()):
                self._reject(client, ident, f"the token does not match {self.token_path}")
                return
            client.authenticated = True
            client.send(_reply(ident, self._greeting()))
            return

        if method == "hello":
            client.send(_reply(ident, self._greeting()))
            return

        handler = self._methods.get(method)
        if handler is None:
            client.send(_failure(
                ident,
                METHOD_NOT_FOUND,
                f"no method named {method!r}. available: " + ", ".join(sorted(self._methods)),
            ))
            return
        try:
            result = handler(params)
        except Exception as exc:  # a broken handler must not kill the connection
            client.send(_failure(ident, INTERNAL_ERROR, f"{method} failed: {type(exc).__name__}: {exc}"))
            return
        if ident is not None:
            client.send(_reply(ident, result))

    def _reject(self, client: _Client, ident: Any, why: str) -> None:
        """One error frame, then the connection goes.  A socket that can write
        files does not get to retry its way in."""
        client.send(_failure(ident, UNAUTHENTICATED, why))
        # Let the writer flush that single frame before the socket closes;
        # otherwise the editor sees a bare EOF and reports nothing useful.
        for _ in range(20):
            if client.outbox.empty():
                break
            time.sleep(0.01)
        self._drop(client, why)

    def _greeting(self) -> dict[str, Any]:
        return {
            "ok": True,
            "version": BRIDGE_VERSION,
            "protocol": PROTOCOL,
            "pid": os.getpid(),
            "workspace": str(self.hooks.workspace),
            "events": list(EVENTS),
            "methods": list(METHODS),
        }

    # -- writing ------------------------------------------------------------

    def _write_loop(self, client: _Client) -> None:
        while True:
            try:
                frame = client.outbox.get(timeout=_TICK)
            except queue.Empty:
                if self._stop.is_set() or client.dropped is not None:
                    return
                continue
            if frame is None:
                return
            try:
                client.sock.sendall(frame)
            except (TimeoutError, OSError) as exc:
                # A peer that is not reading blocks here until the deadline;
                # that deadline is the second half of the drop policy, the
                # first being a queue that fills.
                self._drop(client, f"send failed after {self.send_timeout:g}s: {type(exc).__name__}")
                return

    # -- events -------------------------------------------------------------

    def publish(self, event: str, payload: dict[str, Any] | None = None) -> int:
        """Push a notification to every authenticated client.

        Never blocks and never raises: this runs on the agent's own thread.
        Returns how many clients it reached, which is what makes "nobody is
        watching" cheap to detect.
        """
        if not self._clients:
            return 0
        frame = _encode({
            "jsonrpc": PROTOCOL,
            "method": event,
            "params": {"event": event, "at": time.time(), **(payload or {})},
        })
        delivered = 0
        for client in self._snapshot():
            if not client.authenticated:
                continue
            if client.send(frame):
                delivered += 1
            else:
                self._drop(client, f"fell more than {self.queue_limit} events behind")
        return delivered

    def observe(self, event: Any) -> None:
        """Mirror one agent-loop event onto the wire.

        A turn is one or more steps, so `StepStarted` becomes `agent.started`
        carrying its index: step 0 is the turn beginning and a later step
        re-asserts that the agent is still working, which is exactly what a
        status bar wants and costs nothing to ignore.
        """
        if isinstance(event, StepStarted):
            self._busy = True
            self.publish("agent.started", {"step": event.index, "model": event.model})
        elif isinstance(event, ToolStarted):
            self.publish("tool.started", {"id": event.call.id, "tool": event.call.name, "args": event.call.args})
        elif isinstance(event, ToolFinished):
            inv = event.invocation
            self.publish("tool.finished", {
                "id": inv.call.id,
                "tool": inv.call.name,
                "ok": inv.result.ok,
                "approved": inv.approved,
                "error": inv.result.error,
                "summary": inv.result.display or inv.result.content[:200],
                "duration": round(inv.result.duration, 4),
            })
            if inv.result.ok and self.hooks.writes(inv.call.name):
                self.publish("edit.applied", {"source": "tool", "tool": inv.call.name, "id": inv.call.id})
        elif isinstance(event, Finished):
            self._busy = False
            self.publish("agent.finished", {
                "reason": event.reason,
                "steps": event.steps,
                "text": event.text,
                "usage": {
                    "input": getattr(event.usage, "input", 0),
                    "output": getattr(event.usage, "output", 0),
                },
            })
        elif isinstance(event, StreamError):
            self._busy = False
            self.publish("agent.finished", {"reason": "error", "steps": 0, "text": "", "error": event.message})

    def emit_job(self, job_id: str, state: str, **detail: Any) -> int:
        """Announce a background job transition.  Called by whatever manager
        owns the job; the bridge keeps no job registry of its own."""
        return self.publish("job.state", {"id": job_id, "state": state, **detail})

    # -- methods ------------------------------------------------------------

    def _status(self, params: dict[str, Any]) -> dict[str, Any]:
        out = dict(self.hooks.status())
        jobs = list(self.hooks.jobs())
        running = sum(1 for j in jobs if str(j.get("state", "")).lower() in ("running", "starting"))
        out["jobs"] = jobs
        out["running_jobs"] = running
        out["workspace"] = str(self.hooks.workspace)
        out["clients"] = len(self._clients)
        out["dropped"] = self.dropped
        out["pid"] = os.getpid()
        out["version"] = BRIDGE_VERSION
        out["started"] = self.started
        if self._busy or self._turn.locked():
            out["state"] = "running"
        out.setdefault("state", "idle")
        return out

    def _sessions(self, params: dict[str, Any]) -> dict[str, Any]:
        limit = params.get("limit")
        sessions = self.hooks.sessions()
        if isinstance(limit, int) and limit > 0:
            sessions = sessions[:limit]
        return {"sessions": sessions}

    def _diff(self, params: dict[str, Any]) -> dict[str, Any]:
        root = Path(self.hooks.workspace)
        changes, note = self.hooks.diff(root)
        wanted = params.get("path")
        if isinstance(wanted, str) and wanted:
            changes = [c for c in changes if c.path == wanted]
        return {"root": str(root), "changes": [c.payload() for c in changes], "note": note}

    def _apply_edit(self, params: dict[str, Any]) -> dict[str, Any]:
        path = params.get("path")
        text = params.get("text")
        if not isinstance(path, str) or not path:
            return {"ok": False, "error": "apply_edit needs a non-empty 'path'"}
        if not isinstance(text, str):
            return {"ok": False, "error": "apply_edit needs 'text' as a string; send the whole new file"}
        ok, message = self.hooks.apply(Path(self.hooks.workspace), path, text)
        if ok:
            self.publish("edit.applied", {"source": "editor", "path": path, "characters": len(text)})
        return {"ok": ok, "path": path, "message" if ok else "error": message}

    def _cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        ok, message = self.hooks.cancel()
        return {"ok": ok, "message" if ok else "error": message}

    def _prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "error": "prompt needs a non-empty 'text'"}
        composed = _compose(text, params.get("selection"))
        wait = params.get("wait")
        wait = True if wait is None else bool(wait)

        if not self._turn.acquire(blocking=False):
            return {"ok": False, "error": "a turn is already running; call cancel first"}
        if not wait:
            threading.Thread(target=self._turn_loop, args=(composed,), name="bridge-turn", daemon=True).start()
            return {"ok": True, "accepted": True, "text": ""}
        try:
            ok, reply = self.hooks.prompt(composed)
        except Exception as exc:  # a provider blowing up is a result, not a fault
            self._turn.release()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self._turn.release()
        return {"ok": ok, "accepted": True, "text": reply} if ok else {"ok": False, "error": reply}

    def _turn_loop(self, prompt: str) -> None:
        try:
            ok, reply = self.hooks.prompt(prompt)
            if not ok:
                self.publish("agent.finished", {"reason": "error", "steps": 0, "text": "", "error": reply})
        except Exception as exc:
            self.publish("agent.finished", {"reason": "error", "steps": 0, "text": "", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self._turn.release()


# -- helpers ----------------------------------------------------------------


def _alive(path: Path) -> bool:
    """Whether something is actually listening on a socket file.

    Connecting is the only honest test.  A pid file would lie after a reboot
    recycled the number, and `st_mtime` says nothing about liveness.
    """
    if not hasattr(socket, "AF_UNIX"):
        return False
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.25)
    try:
        probe.connect(str(path))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _hangup(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


def _write_secret(target: Path, body: str) -> str | None:
    """`0o600`, atomically, in the target's own directory."""
    try:
        handle, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
        except OSError:
            Path(tmp).unlink(missing_ok=True)
            raise
    except OSError as exc:
        return f"{target}: {type(exc).__name__}: {exc}"
    return None


def read_descriptor(home: str | os.PathLike[str] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """What an editor does to find a running bridge: `(descriptor, problem)`.

    Exported because the extension is not the only client — a test, a script or
    a second offset needs the same three-line dance, and getting the token from
    the wrong file is the kind of mistake that silently disables auth.
    """
    root = Path(home) if home is not None else settings.home()
    descriptor_path = root / DESCRIPTOR_NAME
    if not descriptor_path.is_file():
        return None, f"no bridge descriptor at {descriptor_path}; is offset running?"
    try:
        raw = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{descriptor_path}: {type(exc).__name__}: {exc}"
    if not isinstance(raw, dict):
        return None, f"{descriptor_path} does not contain a JSON object"
    token_path = Path(raw.get("token_path") or (root / TOKEN_NAME))
    try:
        raw["token"] = token_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return None, f"{token_path}: {type(exc).__name__}: {exc}"
    return raw, None


# -- shell wiring -----------------------------------------------------------
#
# One process, one bridge.  Kept as a module-level instance for the same reason
# settings is: it describes this run, and every caller wants the same one.

_active: Bridge | None = None


def active() -> Bridge | None:
    """The installed bridge, or None."""
    return _active


def install(state: Any, *, jobs: Callable[[], list[dict[str, Any]]] | None = None,
            listen: str = "") -> None:
    """Start the editor bridge for a running shell.

    Idempotent, and never raises: a bridge that cannot bind records its reason
    in `active().problems` and the shell carries on without an editor view.
    Pass `jobs` to expose a background-job registry in `status` and the tree
    view; without one the bridge honestly reports no jobs.  Pass `listen` to
    bind a TCP socket instead of a unix one, which is what a daemon serving a
    remote editor needs.
    """
    global _active
    if _active is not None and _active.listening:
        return

    workspace = Path(getattr(state, "workspace", None) or Path.cwd())
    bridge = Bridge(Hooks(workspace=workspace), listen=listen)
    bridge.hooks.status = _status_of(state, bridge)
    bridge.hooks.cancel = _cancel_of(state)
    bridge.hooks.prompt = _prompt_of(state, bridge)
    bridge.hooks.writes = _writes_of(state)
    if jobs is not None:
        bridge.hooks.jobs = jobs
    bridge.serve()
    _active = bridge


def uninstall() -> None:
    """Stop the installed bridge and forget it."""
    global _active
    bridge, _active = _active, None
    if bridge is not None:
        bridge.shutdown()


def observe(event: Any) -> None:
    """Forward one agent-loop event to the installed bridge, if any.

    This is the single line the shell's event loop needs; everything else the
    bridge learns, it learns by being asked.
    """
    bridge = _active
    if bridge is not None:
        bridge.observe(event)


def _status_of(state: Any, bridge: Bridge) -> Callable[[], dict[str, Any]]:
    def status() -> dict[str, Any]:
        session = getattr(state, "session", None)
        agent = getattr(state, "agent", None)
        return {
            "model": getattr(getattr(agent, "config", None), "model", ""),
            "session": getattr(session, "id", ""),
            "session_path": str(getattr(session, "path", "")),
            "entries": len(session) if session is not None else 0,
            "state": "running" if bridge._busy else "idle",
        }

    return status


def _cancel_of(state: Any) -> Callable[[], tuple[bool, str]]:
    def cancel() -> tuple[bool, str]:
        runtime = getattr(getattr(state, "agent", None), "runtime", None)
        if runtime is None:
            return False, "no agent is attached to this bridge"
        runtime.cancel()
        return True, "the running turn was asked to stop"

    return cancel


def _prompt_of(state: Any, bridge: Bridge) -> Callable[[str], tuple[bool, str]]:
    """Drive a real turn and mirror its events.

    The bridge runs the loop itself rather than posting into the shell's input
    queue: an editor prompt has to work whether or not the TUI is at a prompt,
    and driving the loop here is what makes `tool.started`/`tool.finished`
    reach the editor for editor-initiated work.
    """

    def prompt(text: str) -> tuple[bool, str]:
        agent = getattr(state, "agent", None)
        if agent is None:
            return False, "no agent is attached to this bridge"
        reply, error = "", None
        for event in agent.run(text):
            bridge.observe(event)
            if isinstance(event, Finished):
                reply = event.text
            elif isinstance(event, StreamError):
                error = event.message
        if error is not None:
            return False, error
        return True, reply

    return prompt


def _writes_of(state: Any) -> Callable[[str], bool]:
    from offset.tools.base import Danger

    def writes(name: str) -> bool:
        toolbox = getattr(state, "toolbox", None)
        tool = toolbox.get(name) if toolbox is not None else None
        return tool is not None and getattr(tool, "danger", Danger.SAFE) >= Danger.WRITE

    return writes
