"""Whole-machine tools: the deliberately unrestricted set.

The built-ins in `offset.tools.builtin` are scoped to the workspace on purpose —
a coding agent that can silently rewrite `~/.ssh` is a liability.  These tools
are the opposite bargain, taken knowingly: the user asked for an assistant that
can act on the machine, so `system`/`file`/`open` reach anywhere the invoking
user can reach and every one of them is `Danger.FULL`.

That is safe only because of two things elsewhere:

* the startup grant — `Danger.FULL` sits above `THRESHOLD` for every mode but
  "full", so in any other mode each call goes to the approval prompt;
* `ToolContext.root` — when the session carries a boundary these tools honour
  it, and refusing is returned as a message rather than raised, so the model can
  pick a legal path instead of dying.

Output handling is the other load-bearing part.  A command may print gigabytes;
we keep a bounded head and a bounded tail while the process streams, so memory
stays flat and the model still sees both how a run started and how it ended.
"""

from __future__ import annotations

import os
import shutil
import signal
import stat as statmod
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from offset.tools.base import Danger, Tool, ToolContext, ToolResult
from offset.tools.builtin import MAX_BYTES

#: Kept from the start and the end of a command's output.  Middles are where
#: build logs repeat themselves; heads and tails are where the news is.
HEAD_CHARS = 20_000
TAIL_CHARS = 20_000
#: Lines forwarded live to the UI before we stop narrating.
EMIT_LINES = 400


def kill_group(proc: subprocess.Popen) -> None:
    """Kill the process *and* everything it spawned.

    A shell command that backgrounds a grandchild would otherwise outlive its
    own timeout and keep writing to a pipe nobody reads, which is how a hung
    turn becomes a hung machine.  `start_new_session=True` at spawn time is what
    makes this one signal reach all of them.
    """
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass


class _Sink:
    """Bounded head+tail accumulator fed by the reader thread."""

    __slots__ = ("_head", "_head_len", "_tail", "_tail_len", "total", "_emit", "_emitted", "_partial")

    def __init__(self, emit=None) -> None:
        self._head: list[str] = []
        self._head_len = 0
        self._tail: deque[str] = deque()
        self._tail_len = 0
        self.total = 0
        self._emit = emit
        self._emitted = 0
        self._partial = ""

    def feed(self, chunk: str) -> None:
        self.total += len(chunk)
        if self._head_len < HEAD_CHARS:
            room = HEAD_CHARS - self._head_len
            self._head.append(chunk[:room])
            self._head_len += min(room, len(chunk))
            chunk = chunk[room:]
            if not chunk:
                self._narrate()
                return
        self._tail.append(chunk)
        self._tail_len += len(chunk)
        while self._tail_len - len(self._tail[0]) >= TAIL_CHARS:
            self._tail_len -= len(self._tail.popleft())
        self._narrate(chunk)

    def _narrate(self, chunk: str | None = None) -> None:
        if self._emit is None or self._emitted >= EMIT_LINES:
            return
        text = self._partial + (chunk if chunk is not None else (self._head[-1] if self._head else ""))
        *lines, self._partial = text.split("\n")
        for line in lines:
            if self._emitted >= EMIT_LINES:
                return
            self._emitted += 1
            self._emit(line)

    def text(self) -> str:
        head = "".join(self._head)
        tail = "".join(self._tail)
        if not tail:
            return head
        dropped = self.total - len(head) - len(tail)
        return f"{head}\n... [{dropped} chars truncated] ...\n{tail}" if dropped > 0 else head + tail

    @property
    def truncated(self) -> bool:
        return self._tail_len > 0


class SystemExec(Tool):
    """Run any command, anywhere, with no sandbox.

    Unlike `bash` this is not confined to the workspace: `cwd` may be any
    directory on the machine and the command may do anything the user could do
    from a terminal.  That is the whole point of the tool, and the reason it is
    `Danger.FULL`: capability is granted once at startup (approval mode "full",
    or a prompt per call in every other mode) instead of being smuggled in per
    argument.
    """

    name = "system"
    description = (
        "Run any command anywhere on this machine (no workspace restriction). "
        "Args: command, cwd (any directory), timeout (seconds), stdin. "
        "Captures merged stdout+stderr and the exit code."
    )
    danger = Danger.FULL
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string"},
            "timeout": {"type": "number", "minimum": 0.1, "maximum": 3600},
            "stdin": {"type": "string"},
        },
        "required": ["command"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = args["command"]
        try:
            where = ctx.resolve(args["cwd"]) if args.get("cwd") else ctx.cwd
        except PermissionError as exc:
            return ToolResult.fail(str(exc))
        if not Path(where).is_dir():
            return ToolResult.fail(f"cwd is not a directory: {where}")

        budget = float(args.get("timeout") or ctx.timeout or 120.0)
        feed = args.get("stdin")
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=str(where),
                env={**os.environ, **ctx.env},
                stdin=subprocess.PIPE if feed is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                start_new_session=True,  # its own group: one signal kills the tree
            )
        except OSError as exc:
            return ToolResult.fail(f"could not start command: {exc}")

        sink = _Sink(ctx.emit)
        reader = threading.Thread(target=self._drain, args=(proc, sink), daemon=True)
        reader.start()
        if feed is not None:
            threading.Thread(target=self._feed, args=(proc, feed), daemon=True).start()

        reason = ""
        while proc.poll() is None:
            if ctx.cancel.is_set():
                reason = "cancelled by user"
                break
            if time.monotonic() - started > budget:
                reason = f"timed out after {budget:g}s"
                break
            time.sleep(0.05)
        return self._finish(proc, sink, reader, reason, command, where, started)

    @staticmethod
    def _drain(proc: subprocess.Popen, sink: _Sink) -> None:
        stream = proc.stdout
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    return
                sink.feed(chunk)
        except (ValueError, OSError):
            return

    @staticmethod
    def _feed(proc: subprocess.Popen, data: str) -> None:
        try:
            assert proc.stdin is not None
            proc.stdin.write(data)
            proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass

    def _finish(
        self,
        proc: subprocess.Popen,
        sink: _Sink,
        reader: threading.Thread,
        reason: str,
        command: str,
        where: Path,
        started: float,
    ) -> ToolResult:
        if reason:
            kill_group(proc)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            kill_group(proc)
        reader.join(timeout=2.0)
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        elapsed = time.monotonic() - started
        body = sink.text().strip()
        code = proc.returncode
        if reason:
            return ToolResult(
                ok=False,
                content=f"{reason}\n{body}".strip(),
                display=f"$ {command[:60]} -> {reason}",
                error=reason,
                data={"exit": code, "timed_out": "timed out" in reason, "cwd": str(where)},
                duration=elapsed,
            )
        return ToolResult(
            ok=code == 0,
            content=body or f"(no output, exit {code})",
            display=f"$ {command[:60]} -> exit {code}",
            error=None if code == 0 else f"exit {code}",
            data={"exit": code, "cwd": str(where), "truncated": sink.truncated, "chars": sink.total},
            duration=elapsed,
        )

    def preview(self, args: dict[str, Any]) -> str:
        where = args.get("cwd") or "."
        return f"system: {args.get('command', '')[:80]}  (in {where})"


#: Actions `file` understands, mapped to the extra argument each one needs.
FILE_ACTIONS = {
    "read": (),
    "write": ("content",),
    "append": ("content",),
    "delete": (),
    "copy": ("dest",),
    "move": ("dest",),
    "stat": (),
    "chmod": ("mode",),
}


class AnyFile(Tool):
    """Read and change any file on the machine, `~` included.

    The workspace `read`/`write`/`edit` tools stay as they are — cheap and
    boundaried — and this one exists for the cases they refuse: config in
    `~/.config`, a file on a mounted disk, `/etc` when the user says so.
    """

    name = "file"
    description = (
        "Read/write/append/delete/copy/move/stat/chmod any path on this machine "
        "(absolute, relative, or ~-expanded). Args: action, path, content, dest, mode, recursive."
    )
    danger = Danger.FULL
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(FILE_ACTIONS)},
            "path": {"type": "string"},
            "content": {"type": "string"},
            "dest": {"type": "string"},
            "mode": {"type": "string"},
            "recursive": {"type": "boolean"},
        },
        "required": ["action", "path"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = args["action"]
        if action not in FILE_ACTIONS:
            return ToolResult.fail(f"unknown action {action!r}; use one of {', '.join(sorted(FILE_ACTIONS))}")
        for needed in FILE_ACTIONS[action]:
            if args.get(needed) is None:
                return ToolResult.fail(f"action {action!r} needs {needed!r}")
        try:
            target = ctx.resolve(args["path"])
            dest = ctx.resolve(args["dest"]) if args.get("dest") else None
        except PermissionError as exc:
            return ToolResult.fail(str(exc))
        try:
            return getattr(self, f"_{action}")(target, dest, args)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            return ToolResult.fail(f"{action} {target}: {exc}")

    # -- actions ------------------------------------------------------------

    def _read(self, target: Path, _dest: Path | None, args: dict[str, Any]) -> ToolResult:
        if not target.exists():
            return ToolResult.fail(f"no such path: {target}")
        if target.is_dir():
            rows = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return ToolResult(content="\n".join(rows) or "(empty)", display=f"file read {target} ({len(rows)} entries)")
        size = target.stat().st_size
        if size > MAX_BYTES:
            return ToolResult.fail(f"{target} is {size} bytes (> {MAX_BYTES}); read it with system, e.g. sed -n")
        raw = target.read_bytes()
        try:
            body = raw.decode(args.get("encoding") or "utf-8")
        except UnicodeDecodeError:
            return ToolResult.fail(f"{target} is not text ({size} bytes); use system for binary content")
        return ToolResult(content=body, display=f"file read {target} ({size} bytes)", data={"bytes": size})

    def _write(self, target: Path, _dest: Path | None, args: dict[str, Any]) -> ToolResult:
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(args["content"], encoding="utf-8")
        if args.get("mode"):
            target.chmod(_mode(args["mode"]))
        n = len(args["content"].encode())
        verb = "overwrote" if existed else "created"
        return ToolResult(
            content=f"{verb} {target} ({n} bytes)",
            display=f"file {verb} {target}",
            data={"bytes": n, "created": not existed, "path": str(target)},
        )

    def _append(self, target: Path, _dest: Path | None, args: dict[str, Any]) -> ToolResult:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(args["content"])
        return ToolResult(
            content=f"appended {len(args['content'].encode())} bytes to {target}",
            display=f"file append {target}",
            data={"path": str(target), "size": target.stat().st_size},
        )

    def _delete(self, target: Path, _dest: Path | None, args: dict[str, Any]) -> ToolResult:
        if not target.exists() and not target.is_symlink():
            return ToolResult.fail(f"no such path: {target}")
        if target.is_dir() and not target.is_symlink():
            if not args.get("recursive") and any(target.iterdir()):
                return ToolResult.fail(f"{target} is a non-empty directory; pass recursive=true to remove it")
            shutil.rmtree(target) if args.get("recursive") else target.rmdir()
            return ToolResult(content=f"removed directory {target}", display=f"file delete {target}/")
        target.unlink()
        return ToolResult(content=f"deleted {target}", display=f"file delete {target}")

    def _copy(self, target: Path, dest: Path | None, args: dict[str, Any]) -> ToolResult:
        assert dest is not None
        if not target.exists():
            return ToolResult.fail(f"no such path: {target}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if target.is_dir():
            shutil.copytree(target, dest, dirs_exist_ok=bool(args.get("recursive", True)))
        else:
            shutil.copy2(target, dest)
        return ToolResult(content=f"copied {target} -> {dest}", display=f"file copy {dest}", data={"dest": str(dest)})

    def _move(self, target: Path, dest: Path | None, _args: dict[str, Any]) -> ToolResult:
        assert dest is not None
        if not target.exists() and not target.is_symlink():
            return ToolResult.fail(f"no such path: {target}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(dest))
        return ToolResult(content=f"moved {target} -> {dest}", display=f"file move {dest}", data={"dest": str(dest)})

    def _stat(self, target: Path, _dest: Path | None, _args: dict[str, Any]) -> ToolResult:
        if not target.exists() and not target.is_symlink():
            return ToolResult.fail(f"no such path: {target}")
        info = target.lstat()
        kind = "dir" if statmod.S_ISDIR(info.st_mode) else "link" if statmod.S_ISLNK(info.st_mode) else "file"
        mode = oct(statmod.S_IMODE(info.st_mode))[2:].rjust(3, "0")
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime))
        body = "\n".join(
            [
                f"path: {target}",
                f"type: {kind}",
                f"size: {info.st_size}",
                f"mode: {mode}",
                f"uid/gid: {info.st_uid}/{info.st_gid}",
                f"modified: {when}",
            ]
        )
        return ToolResult(
            content=body,
            display=f"file stat {target} ({kind}, {info.st_size} bytes)",
            data={"type": kind, "size": info.st_size, "mode": mode, "mtime": info.st_mtime},
        )

    def _chmod(self, target: Path, _dest: Path | None, args: dict[str, Any]) -> ToolResult:
        if not target.exists():
            return ToolResult.fail(f"no such path: {target}")
        bits = _mode(args["mode"])
        target.chmod(bits)
        return ToolResult(
            content=f"mode of {target} is now {oct(bits)[2:].rjust(3, '0')}",
            display=f"file chmod {oct(bits)[2:]} {target}",
            data={"mode": oct(bits)[2:]},
        )

    def preview(self, args: dict[str, Any]) -> str:
        extra = f" -> {args['dest']}" if args.get("dest") else ""
        return f"file {args.get('action')} {args.get('path')}{extra}"


def _mode(value: str | int) -> int:
    """Accept "644", "0644" or 420; reject anything that is not a mode."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text, 8)
    except ValueError as exc:
        raise ValueError(f"{value!r} is not an octal file mode like 644") from exc


def launch_argv(target: str, *, app: str | None = None, platform: str = sys.platform) -> list[str]:
    """The argv that would open `target`.

    Split out from the launch itself because the interesting part — did we build
    the right command for this platform — is testable on a machine with no
    display, where actually launching anything would be a lie.
    """
    if app:
        if platform == "darwin":
            return ["open", "-a", app, target]
        return [app, target]
    if platform == "darwin":
        return ["open", target]
    if platform.startswith("win"):
        return ["cmd", "/c", "start", "", target]
    return ["xdg-open", target]


def applications(limit: int = 200) -> list[str]:
    """Installed application names, cheaply: `.desktop` Name= lines, or /Applications."""
    names: list[str] = []
    seen: set[str] = set()
    if sys.platform == "darwin":
        for base in (Path("/Applications"), Path.home() / "Applications"):
            if not base.is_dir():
                continue
            for entry in sorted(base.glob("*.app")):
                if entry.stem not in seen:
                    seen.add(entry.stem)
                    names.append(entry.stem)
        return names[:limit]
    roots = [Path("/usr/share/applications"), Path("/usr/local/share/applications"), Path.home() / ".local/share/applications"]
    for base in roots:
        if not base.is_dir():
            continue
        for entry in sorted(base.glob("*.desktop")):
            if len(names) >= limit:
                return names
            try:
                for line in entry.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("Name="):
                        label = line[5:].strip()
                        if label and label not in seen:
                            seen.add(label)
                            names.append(label)
                        break
            except OSError:
                continue
    return names[:limit]


class OpenWith(Tool):
    """Hand a file or URL to the desktop, and never wait for it.

    The child is a GUI application: it can live for hours, so waiting on it
    would pin a worker thread for the rest of the session.  We detach it and
    report the argv instead, which is also the only thing worth asserting on a
    headless machine.
    """

    name = "open"
    description = (
        "Open a file or URL with the desktop default application, or with a named app. "
        "Args: target, app, list (true to list installed applications instead). Returns immediately."
    )
    danger = Danger.FULL
    schema = {
        "type": "object",
        "properties": {
            "target": {"type": "string"},
            "app": {"type": "string"},
            "list": {"type": "boolean"},
        },
        "required": [],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if args.get("list"):
            names = applications()
            return ToolResult(
                content="\n".join(names) or "(no application entries found)",
                display=f"open: {len(names)} applications",
                data={"applications": names},
            )
        target = args.get("target")
        if not target:
            return ToolResult.fail("open needs a target (a path or a URL), or list=true")
        if "://" not in target:
            try:
                resolved = ctx.resolve(target)
            except PermissionError as exc:
                return ToolResult.fail(str(exc))
            if not resolved.exists():
                return ToolResult.fail(f"no such path: {resolved}")
            target = str(resolved)
        argv = launch_argv(target, app=args.get("app"))
        if shutil.which(argv[0]) is None:
            return ToolResult.fail(f"{argv[0]} is not on PATH; name an application with `app`")
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(ctx.cwd),
                env={**os.environ, **ctx.env},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,  # detached: it outlives this turn
            )
        except OSError as exc:
            return ToolResult.fail(f"could not launch {argv[0]}: {exc}")
        return ToolResult(
            content=f"launched: {' '.join(argv)} (pid {proc.pid}, not waited on)",
            display=f"open {target}",
            data={"argv": argv, "pid": proc.pid, "display": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))},
        )

    def preview(self, args: dict[str, Any]) -> str:
        if args.get("list"):
            return "open: list installed applications"
        return f"open {args.get('target', '')}" + (f" with {args['app']}" if args.get("app") else "")


def system_tools() -> list[Tool]:
    """The whole-machine set.  Every one of these is `Danger.FULL`."""
    from offset.tools.documents import Documents

    return [SystemExec(), AnyFile(), Documents(), OpenWith()]
