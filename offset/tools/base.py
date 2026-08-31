"""Tool contract, argument validation, and the registry.

A tool declares how much damage it can do (`danger`) and whether it may run
alongside its siblings (`parallel_safe`).  Those two flags are the whole basis
for both the approval policy and the concurrency decision, so they belong to
the tool rather than to the caller guessing.

Argument validation lives here too.  Models produce plausible-looking wrong
arguments constantly; catching that with a clear message and handing it back is
far better than a stack trace, because the model can then fix its own call.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Final, Iterator

from offset.providers.base import ToolSpec


class Danger(IntEnum):
    """How much a call can hurt.  Drives approval, never capability."""

    SAFE = 0  # reads, searches, queries
    WRITE = 1  # creates or modifies files inside the workspace
    DESTRUCTIVE = 2  # deletes, runs arbitrary commands, touches the network
    FULL = 3  # reaches outside the workspace: any file, any app, the machine


#: Default for `ToolContext.root`.  A plain `None` default would make the
#: unrestricted case the accident rather than the decision, so "not given"
#: (this sentinel, replaced by `cwd`) and "explicitly unrestricted" (`None`)
#: have to be different values.
WORKSPACE: Final = Path("<workspace>")


class Cancelled(Exception):
    """Raised inside a tool when the user aborts the turn."""


@dataclass(slots=True)
class ToolResult:
    ok: bool = True
    content: str = ""  # what the model sees
    display: str = ""  # one line for the UI
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration: float = 0.0

    @classmethod
    def fail(cls, error: str, *, display: str = "") -> "ToolResult":
        return cls(ok=False, content=error, display=display or error, error=error)

    @classmethod
    def text(cls, content: str, *, display: str = "", **data: Any) -> "ToolResult":
        return cls(ok=True, content=content, display=display or content.splitlines()[0][:80] if content else "", data=data)


@dataclass(slots=True)
class ToolContext:
    """Everything a tool is allowed to know about the world."""

    cwd: Path
    #: Permission boundary for `resolve`.  `None` is the whole machine and is
    #: only ever set from an explicit user grant.
    root: Path | None = WORKSPACE
    cancel: threading.Event = field(default_factory=threading.Event)
    timeout: float = 120.0
    env: dict[str, str] = field(default_factory=dict)
    emit: Callable[[str], None] = lambda _line: None

    def __post_init__(self) -> None:
        if self.root is WORKSPACE:
            self.root = self.cwd

    def check(self) -> None:
        """Cooperative cancellation: long tools must call this in their loops."""
        if self.cancel.is_set():
            raise Cancelled("cancelled by user")

    def resolve(self, path: str) -> Path:
        """Absolute path, refusing escapes from `root` unless `root` is None."""
        supplied = Path(path).expanduser()
        target = (supplied if supplied.is_absolute() else self.cwd / supplied).resolve()
        if self.root is None:
            return target
        root = self.root.resolve()
        if root != target and root not in target.parents:
            raise PermissionError(f"path escapes the workspace: {path}")
        return target

    def unrestricted(self) -> "ToolContext":
        """A copy with no boundary, for a tool the user granted full access."""
        return ToolContext(cwd=self.cwd, root=None, cancel=self.cancel, timeout=self.timeout, env=self.env, emit=self.emit)


class Tool(ABC):
    name: str = ""
    description: str = ""
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    danger: Danger = Danger.SAFE
    #: False when two concurrent calls could interfere (shared cwd, same file).
    parallel_safe: bool = True

    @abstractmethod
    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.schema)

    def preview(self, args: dict[str, Any]) -> str:
        """A one-line description of what this call would do, for approval."""
        shown = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
        return f"{self.name}({shown})"

    def __repr__(self) -> str:
        return f"<Tool {self.name} danger={self.danger.name}>"


# -- argument validation ----------------------------------------------------

_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def validate(args: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Check `args` against a JSON-Schema subset.  Returns human-readable errors.

    Deliberately partial: required, type, enum, minimum/maximum and nested
    object/array item types.  Anything more elaborate belongs in the tool.
    """
    errors: list[str] = []
    if not isinstance(args, dict):
        return ["arguments must be an object"]

    for name in schema.get("required") or ():
        if name not in args:
            errors.append(f"missing required argument {name!r}")

    props = schema.get("properties") or {}
    for key, value in args.items():
        spec = props.get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                errors.append(f"unexpected argument {key!r}")
            continue
        errors.extend(_check(value, spec, key))
    return errors


def _check(value: Any, spec: dict[str, Any], path: str) -> list[str]:
    out: list[str] = []
    expected = spec.get("type")
    if isinstance(expected, str):
        wanted = _TYPES.get(expected)
        # bool is an int subclass; a boolean is never an acceptable number
        if wanted and (not isinstance(value, wanted) or (expected in ("number", "integer") and isinstance(value, bool))):
            return [f"{path}: expected {expected}, got {type(value).__name__}"]

    choices = spec.get("enum")
    if choices is not None and value not in choices:
        out.append(f"{path}: must be one of {', '.join(map(str, choices))}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in spec and value < spec["minimum"]:
            out.append(f"{path}: must be >= {spec['minimum']}")
        if "maximum" in spec and value > spec["maximum"]:
            out.append(f"{path}: must be <= {spec['maximum']}")

    if isinstance(value, str) and "maxLength" in spec and len(value) > spec["maxLength"]:
        out.append(f"{path}: longer than {spec['maxLength']} characters")

    if isinstance(value, list) and isinstance(spec.get("items"), dict):
        for i, item in enumerate(value):
            out.extend(_check(item, spec["items"], f"{path}[{i}]"))

    if isinstance(value, dict) and isinstance(spec.get("properties"), dict):
        out.extend(validate(value, spec))
    return out


# -- registry ---------------------------------------------------------------


class Toolbox:
    """The set of tools a session can call.  Every one of them is enabled."""

    __slots__ = ("_tools",)

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool, *, replace: bool = False) -> Tool:
        if not tool.name:
            raise ValueError("a tool needs a name")
        if tool.name in self._tools and not replace:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> bool:
        """Withdraw a tool.  True if it was there, False if it never was.

        The registry has to be able to shrink, not just grow: a reconnected MCP
        server publishes a different tool list, and leaving the old entries in
        place left names the model could call that resolved to a dead pipe.
        Removal is idempotent so a reload can be repeated safely.
        """
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
