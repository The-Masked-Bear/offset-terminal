"""The built-in tool set: files, search, and a shell.

All of them are registered by default.  What varies is the danger class, which
is what the approval policy reads: reading is free, writing asks in `safe`
mode, and the shell asks in anything but `yolo`.
"""

from __future__ import annotations

import fnmatch
import os
import re
import signal
import subprocess
import time
import uuid
from typing import Any

from offset.tools.base import Danger, Tool, ToolContext, ToolResult
from offset.tools.walk import PRUNE
from offset.tools.walk import walk as ignore_aware_walk

#: Directories never worth walking; skipping them is the difference between a
#: glob that answers instantly and one that reads a virtualenv.
PRUNE = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache", "dist", "build", ".offset"}
MAX_BYTES = 512_000


def _prune(parts: tuple[str, ...]) -> bool:
    return any(p in PRUNE for p in parts)


class ReadFile(Tool):
    name = "read"
    description = "Read a UTF-8 text file, optionally a line range. Returns numbered lines."
    danger = Danger.SAFE
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
        },
        "required": ["path"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = ctx.resolve(args["path"])
        if not target.exists():
            return ToolResult.fail(f"no such file: {args['path']}")
        if target.is_dir():
            return ToolResult.fail(f"{args['path']} is a directory; use list")
        if target.stat().st_size > MAX_BYTES:
            return ToolResult.fail(f"{args['path']} is larger than {MAX_BYTES} bytes; read a range")
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return ToolResult.fail(f"{args['path']} is not UTF-8 text")
        start = max(1, int(args.get("offset", 1)))
        end = start + int(args.get("limit", 2000)) - 1
        chosen = lines[start - 1 : end]
        body = "\n".join(f"{start + i}:{line}" for i, line in enumerate(chosen))
        return ToolResult(
            content=body,
            display=f"read {args['path']} ({len(chosen)} lines)",
            data={"lines": len(chosen), "total": len(lines)},
        )


class WriteFile(Tool):
    name = "write"
    description = "Create or overwrite a file with the given content."
    danger = Danger.WRITE
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = ctx.resolve(args["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(args["content"], encoding="utf-8")
        n = len(args["content"].encode())
        return ToolResult(
            content=f"{'overwrote' if existed else 'created'} {args['path']} ({n} bytes)",
            display=f"{'overwrote' if existed else 'created'} {args['path']}",
            data={"bytes": n, "created": not existed},
        )


class EditFile(Tool):
    name = "edit"
    description = "Replace an exact string in a file. The old string must appear exactly once unless `all` is set."
    danger = Danger.WRITE
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "all": {"type": "boolean"},
        },
        "required": ["path", "old", "new"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = ctx.resolve(args["path"])
        if not target.exists():
            return ToolResult.fail(f"no such file: {args['path']}")
        body = target.read_text(encoding="utf-8")
        old, new = args["old"], args["new"]
        hits = body.count(old)
        if hits == 0:
            return ToolResult.fail(f"{args['path']}: the old string does not appear")
        if hits > 1 and not args.get("all"):
            return ToolResult.fail(
                f"{args['path']}: the old string appears {hits} times; include more context or set all=true"
            )
        target.write_text(body.replace(old, new) if args.get("all") else body.replace(old, new, 1), encoding="utf-8")
        return ToolResult(
            content=f"edited {args['path']} ({hits if args.get('all') else 1} replacement(s))",
            display=f"edited {args['path']}",
            data={"replacements": hits if args.get("all") else 1},
        )


class ListDir(Tool):
    name = "list"
    description = "List directory entries. Directories are suffixed with a slash."
    danger = Danger.SAFE
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = ctx.resolve(args.get("path", "."))
        if not target.is_dir():
            return ToolResult.fail(f"not a directory: {args.get('path', '.')}")
        rows = sorted(
            (f"{p.name}/" if p.is_dir() else p.name) for p in target.iterdir() if p.name not in PRUNE
        )
        return ToolResult(content="\n".join(rows) or "(empty)", display=f"list {args.get('path', '.')} ({len(rows)})")


class Glob(Tool):
    name = "glob"
    description = "Find files matching a glob pattern, newest first."
    danger = Danger.SAFE
    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
            "ignored": {"type": "boolean", "description": "include files .gitignore excludes"},
        },
        "required": ["pattern"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.cwd.resolve()
        limit = int(args.get("limit", 200))
        found: list[tuple[float, str]] = []
        for path in ignore_aware_walk(
            root,
            respect_gitignore=not args.get("ignored"),
            # `ignored` has to stand the built-in prune list down as well:
            # leaving it in place meant asking for ignored files still never
            # entered node_modules, .venv or build, the exact directories
            # somebody sets this flag to reach.
            prune=() if args.get("ignored") else PRUNE,
            check=ctx.check,
        ):
            rel = path.relative_to(root)
            if fnmatch.fnmatch(str(rel), args["pattern"]) or fnmatch.fnmatch(path.name, args["pattern"]):
                try:
                    found.append((path.stat().st_mtime, str(rel)))
                except OSError:
                    continue
        found.sort(reverse=True)
        rows = [name for _, name in found[:limit]]
        return ToolResult(
            content="\n".join(rows) or "(no matches)",
            display=f"glob {args['pattern']} ({len(found)} matches)",
            data={"count": len(found), "truncated": len(found) > limit},
        )


class Grep(Tool):
    name = "grep"
    description = "Search file contents with a regular expression."
    danger = Danger.SAFE
    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
            "ignored": {"type": "boolean", "description": "include files .gitignore excludes"},
        },
        "required": ["pattern"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            needle = re.compile(args["pattern"])
        except re.error as exc:
            return ToolResult.fail(f"bad regular expression: {exc}")
        root = ctx.resolve(args.get("path", "."))
        limit = int(args.get("limit", 200))
        pattern = args.get("glob")
        hits: list[str] = []
        base = ctx.cwd.resolve()
        files = (
            iter([root])
            if root.is_file()
            else ignore_aware_walk(
                root,
                respect_gitignore=not args.get("ignored"),
                # `ignored` has to stand the built-in prune list down as well.
                # Leaving it in place meant asking for ignored files still never
                # entered node_modules, .venv or build - the exact directories
                # somebody sets this flag to reach.
                prune=() if args.get("ignored") else PRUNE,
                check=ctx.check,
            )
        )
        for path in files:
            if len(hits) >= limit:
                break
            try:
                rel = path.relative_to(base)
            except ValueError:
                rel = path
            if pattern and not fnmatch.fnmatch(path.name, pattern):
                continue
            try:
                if path.stat().st_size > MAX_BYTES:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if needle.search(line):
                    hits.append(f"{rel}:{n}:{line.strip()[:200]}")
                    if len(hits) >= limit:
                        break
        return ToolResult(
            content="\n".join(hits) or "(no matches)",
            display=f"grep {args['pattern']} ({len(hits)} hits)",
            data={"count": len(hits)},
        )


class Bash(Tool):
    """A shell whose working directory and exports survive between calls.

    `cd /tmp` followed by `pwd` has to print /tmp, or every multi-step shell
    recipe a model knows is wrong. Rather than babysitting a long-lived shell -
    which has to be restarted when it dies, and wedges if a command reads stdin
    - each call reports its final state on the way out and the next call
    restores it. A killed or timed-out command simply leaves the previous state
    in place.
    """

    name = "bash"
    description = (
        "Run a shell command. The working directory and exported variables "
        "persist between calls, so `cd` sticks."
    )
    danger = Danger.DESTRUCTIVE
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number", "minimum": 1, "maximum": 3600},
            "reset": {"type": "boolean", "description": "forget the persisted cwd and exports"},
        },
        "required": ["command"],
    }

    #: Variables that are meaningless to carry across calls.
    VOLATILE = frozenset({"_", "PWD", "OLDPWD", "SHLVL", "RANDOM", "LINENO", "BASHPID", "PPID"})

    def __init__(self) -> None:
        self._cwd: str | None = None
        self._exports: dict[str, str] = {}

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if args.get("reset"):
            self._cwd, self._exports = None, {}
        budget = float(args.get("timeout") or min(ctx.timeout, 120.0))
        started = time.monotonic()
        where = self._cwd if self._cwd and os.path.isdir(self._cwd) else str(ctx.cwd)
        sentinel = f"__offset_{uuid.uuid4().hex}__"
        script = (
            f"{args['command']}\n"
            f"__offset_rc=$?\n"
            f"printf '%s' {sentinel}\n"
            f"printf '%s\\0' \"$PWD\"\n"
            f"env -0\n"
            f"exit $__offset_rc\n"
        )
        proc = subprocess.Popen(
            script,
            shell=True,
            cwd=where,
            env={**os.environ, **self._exports, **ctx.env},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,  # its own group, so we can kill children too
        )
        try:
            while True:
                try:
                    out = proc.communicate(timeout=0.2)[0]
                    break
                except subprocess.TimeoutExpired:
                    if ctx.cancel.is_set() or time.monotonic() - started > budget:
                        self._kill(proc)
                        reason = "cancelled" if ctx.cancel.is_set() else f"timed out after {budget:g}s"
                        tail = (proc.communicate()[0] or "").split(sentinel)[0][-2000:]
                        return ToolResult.fail(f"{reason}\n{tail}".strip())
        finally:
            if proc.poll() is None:
                self._kill(proc)

        code = proc.returncode
        body, state = self._split(out or "", sentinel)
        if state is not None:
            self._absorb(state)
        body = body.strip()
        if len(body) > 40_000:
            body = body[:20_000] + "\n... [truncated] ...\n" + body[-20_000:]
        return ToolResult(
            ok=code == 0,
            content=body or f"(no output, exit {code})",
            display=f"$ {args['command'][:70]} -> exit {code}",
            error=None if code == 0 else f"exit {code}",
            data={"exit": code, "cwd": self._cwd or where},
        )

    @staticmethod
    def _split(out: str, sentinel: str) -> tuple[str, str | None]:
        """Separate the command's own output from the trailing state report."""
        head, marker, tail = out.rpartition(sentinel)
        return (head, tail) if marker else (out, None)

    def _absorb(self, state: str) -> None:
        fields = [f for f in state.split("\0") if f]
        if not fields:
            return
        self._cwd = fields[0]
        captured: dict[str, str] = {}
        for field in fields[1:]:
            key, sep, value = field.partition("=")
            if sep and key and key not in self.VOLATILE:
                captured[key] = value
        if captured:
            # Only keep what differs from this process's own environment, so the
            # persisted set stays the user's exports rather than a whole copy.
            self._exports = {k: v for k, v in captured.items() if os.environ.get(k) != v}

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


class Fetch(Tool):
    name = "fetch"
    description = "Fetch a URL over HTTP(S) and return its body as text."
    #: Not destructive locally, but it is egress: it can carry workspace
    #: contents off the machine, so it asks in anything stricter than yolo.
    danger = Danger.WRITE
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_bytes": {"type": "integer", "minimum": 1, "maximum": MAX_BYTES},
        },
        "required": ["url"],
    }

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        import urllib.error
        import urllib.request

        url = args["url"]
        if not url.startswith(("http://", "https://")):
            return ToolResult.fail("only http and https URLs are supported")
        cap = int(args.get("max_bytes") or 200_000)
        request = urllib.request.Request(url, headers={"User-Agent": "offset/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=min(ctx.timeout, 60.0)) as response:
                raw = response.read(cap + 1)
                status = response.status
                charset = response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            return ToolResult.fail(f"{url} returned HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            return ToolResult.fail(f"could not fetch {url}: {exc}")
        body = raw[:cap].decode(charset, "replace")
        return ToolResult(
            content=body + ("\n... [truncated] ..." if len(raw) > cap else ""),
            display=f"fetch {url} -> {status} ({len(body)} chars)",
            data={"status": status, "truncated": len(raw) > cap},
        )


def builtin_tools() -> list[Tool]:
    """Every built-in, enabled.  Order is the order the model sees them in."""
    return [ReadFile(), WriteFile(), EditFile(), ListDir(), Glob(), Grep(), Bash(), Fetch()]
