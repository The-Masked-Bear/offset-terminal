"""User-authored tools, loaded and working by default.

Two routes in, because the people who write tools do not all write Python:

  * a `.py` file dropped in `~/.offset/tools/` that defines `Tool` subclasses
    or a module-level `TOOLS` list — full in-process access;
  * a `tool.json` manifest naming any executable — the tool receives its
    arguments as JSON on stdin and answers with JSON (or plain text) on
    stdout, so a shell script, a Go binary or a Node file all work unchanged.

Discovery never raises.  A plugin with a syntax error is reported as a load
error and skipped, because one bad file must not stop the agent from starting.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from offset.tools.base import Danger, Tool, ToolContext, ToolResult

MANIFEST_NAMES = ("tool.json", "offset-tool.json")


@dataclass(slots=True)
class LoadError:
    source: Path
    message: str


@dataclass(slots=True)
class Discovery:
    tools: list[Tool] = field(default_factory=list)
    errors: list[LoadError] = field(default_factory=list)

    def __iter__(self):
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)


class ExternalTool(Tool):
    """A tool implemented by any executable, spoken to over JSON on stdio."""

    def __init__(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        command: list[str],
        *,
        danger: Danger = Danger.WRITE,
        parallel_safe: bool = True,
        timeout: float | None = None,
        source: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.schema = schema
        self.command = command
        self.danger = danger
        self.parallel_safe = parallel_safe
        self.timeout = timeout
        self.source = source
        self.env = env or {}

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        budget = self.timeout or ctx.timeout
        cwd = str(self.source.parent) if self.source else str(ctx.cwd)
        payload = json.dumps({"arguments": args, "cwd": str(ctx.cwd)})
        try:
            proc = subprocess.run(
                self.command,
                input=payload,
                capture_output=True,
                text=True,
                timeout=budget,
                cwd=cwd,
                env={**os.environ, **ctx.env, **self.env, "OFFSET_TOOL": self.name, "OFFSET_CWD": str(ctx.cwd)},
            )
        except subprocess.TimeoutExpired:
            return ToolResult.fail(f"{self.name} exceeded its {budget:g}s budget")
        except (OSError, ValueError) as exc:
            return ToolResult.fail(f"{self.name} could not start: {exc}")

        stdout = (proc.stdout or "").strip()
        if proc.returncode != 0 and not stdout:
            return ToolResult.fail(f"{self.name} exited {proc.returncode}: {(proc.stderr or '').strip()[:400]}")
        return self._interpret(stdout, proc.returncode, proc.stderr or "")

    def _interpret(self, stdout: str, code: int, stderr: str) -> ToolResult:
        """Accept a JSON envelope, or fall back to treating stdout as text."""
        try:
            obj = json.loads(stdout)
        except json.JSONDecodeError:
            return ToolResult(
                ok=code == 0,
                content=stdout or stderr.strip(),
                display=f"{self.name} -> exit {code}",
                error=None if code == 0 else f"exit {code}",
            )
        if not isinstance(obj, dict):
            return ToolResult(ok=code == 0, content=json.dumps(obj), display=self.name)
        ok = bool(obj.get("ok", code == 0))
        content = obj.get("content")
        if content is None:
            content = json.dumps({k: v for k, v in obj.items() if k not in ("ok", "display", "error")})
        return ToolResult(
            ok=ok,
            content=str(content),
            display=str(obj.get("display") or self.name),
            data=obj.get("data") if isinstance(obj.get("data"), dict) else {},
            error=None if ok else str(obj.get("error") or "tool reported failure"),
        )


def _danger(value: Any) -> Danger:
    if isinstance(value, str):
        return {
            "safe": Danger.SAFE,
            "write": Danger.WRITE,
            "destructive": Danger.DESTRUCTIVE,
            "full": Danger.FULL,
        }.get(value.lower(), Danger.WRITE)
    return Danger.WRITE


def load_manifest(path: Path) -> list[Tool]:
    """Build tools from a `tool.json`.  Accepts one object or a list."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw if isinstance(raw, list) else [raw]
    out: list[Tool] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest entries must be objects")
        name = entry.get("name")
        command = entry.get("command")
        if not name or not command:
            raise ValueError("a manifest entry needs `name` and `command`")
        if isinstance(command, str):
            command = [command]
        resolved = [str((path.parent / c).resolve()) if i == 0 and (path.parent / c).exists() else str(c)
                    for i, c in enumerate(command)]
        out.append(ExternalTool(
            name=str(name),
            description=str(entry.get("description") or f"user tool {name}"),
            schema=entry.get("schema") if isinstance(entry.get("schema"), dict) else {"type": "object", "properties": {}},
            command=resolved,
            danger=_danger(entry.get("danger")),
            parallel_safe=bool(entry.get("parallel_safe", True)),
            timeout=float(entry["timeout"]) if entry.get("timeout") else None,
            source=path,
            env=entry.get("env") if isinstance(entry.get("env"), dict) else None,
        ))
    return out


def load_python(path: Path) -> list[Tool]:
    """Import a plugin file and collect the tools it defines."""
    spec = importlib.util.spec_from_file_location(f"offset_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    declared = getattr(module, "TOOLS", None)
    if isinstance(declared, (list, tuple)):
        return [t if isinstance(t, Tool) else t() for t in declared]
    single = getattr(module, "TOOL", None)
    if single is not None:
        return [single if isinstance(single, Tool) else single()]
    return [
        obj()
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Tool) and obj is not Tool and obj.__module__ == module.__name__ and getattr(obj, "name", "")
    ]


def discover(dirs: Iterable[Path]) -> Discovery:
    """Find every user tool under the given directories."""
    found = Discovery()
    seen: set[str] = set()
    for directory in dirs:
        directory = Path(directory).expanduser()
        if not directory.is_dir():
            continue
        candidates = sorted(directory.rglob("*"))
        for path in candidates:
            if path.is_dir() or ".offset-skip" in path.parts:
                continue
            try:
                if path.name in MANIFEST_NAMES:
                    tools = load_manifest(path)
                elif path.suffix == ".py" and not path.name.startswith("_"):
                    tools = load_python(path)
                else:
                    continue
            except Exception as exc:  # a broken plugin is reported, never fatal
                found.errors.append(LoadError(path, f"{type(exc).__name__}: {exc}"))
                continue
            for tool in tools:
                if not getattr(tool, "name", ""):
                    found.errors.append(LoadError(path, "tool has no name"))
                elif tool.name in seen:
                    found.errors.append(LoadError(path, f"duplicate tool name {tool.name!r}, skipped"))
                else:
                    seen.add(tool.name)
                    found.tools.append(tool)
    return found


def default_dirs(workspace: Path | None = None) -> list[Path]:
    """Where user tools live: the workspace first, then the user's home."""
    home = Path(os.environ.get("OFFSET_HOME") or (Path.home() / ".offset"))
    dirs = [home / "tools"]
    if workspace:
        dirs.insert(0, Path(workspace) / ".offset" / "tools")
    return dirs
