"""Layered settings.

Why layers: a person has global preferences, a repository has project rules and
a single run has flags, and none of those three may quietly overwrite another's
file.  Only the user layer is ever written back, so `set()` cannot mutate a
checked-in `<workspace>/.offset/config.json`.

Why a declared schema: without one, a typo in a config file is
indistinguishable from a feature nobody implemented — the value is read, found
missing, and the default silently wins.  Every key is declared here, so an
unknown key, an unknown `OFFSET_*` variable and a value of the wrong type are
all *reported* (`problems()`) rather than dropped on the floor.

Why the write is debounced and atomic: `/set` in a tight loop must not mean one
fsync per keystroke, and a process dying mid-write must not leave a truncated
config that fails to parse on the next start.  The save re-reads the file under
a lock first, so an edit made in `$EDITOR` while a session is open survives.
"""

from __future__ import annotations

import atexit
import copy
import difflib
import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

#: Seconds between the last `set()` and the write it causes.
DEBOUNCE: Final = 0.25

#: Layer names, lowest precedence first.  `defaults` is the schema itself.
ORDER: Final = ("defaults", "user", "project", "env", "runtime")

#: `OFFSET_*` variables that are not settings.  Without this list every one of
#: them would be reported as a typo.
RESERVED: Final = frozenset(
    {
        "OFFSET_HOME",
        "OFFSET_COLOR",
        "OFFSET_ASCII",
        "OFFSET_LIVE",
        "OFFSET_LIVE_MODEL",
        "OFFSET_TOOL",
        "OFFSET_CWD",
        "OFFSET_CONFIG",
    }
)

_TRUE: Final = frozenset({"1", "true", "yes", "on"})
_FALSE: Final = frozenset({"0", "false", "no", "off"})
_CAMEL: Final = re.compile(r"(?<!^)(?=[A-Z])")


class SettingsError(ValueError):
    """A write that was refused: unknown key, bad type, bad choice."""


@dataclass(frozen=True, slots=True)
class Spec:
    """One declared setting: the sole source of its default and its type."""

    key: str
    kind: str  # bool | int | float | str | list | dict
    default: Any
    doc: str = ""
    choices: tuple[str, ...] = ()

    @property
    def env(self) -> str:
        """`tools.approvalMode` -> `OFFSET_TOOLS_APPROVAL_MODE`."""
        return "_".join(["OFFSET", *(_CAMEL.sub("_", part).upper() for part in self.key.split("."))])

    def coerce(self, value: Any) -> tuple[Any, str]:
        """Accept a typed value from JSON or a caller.  `("", msg)` on refusal."""
        kind = self.kind
        if kind == "bool":
            if isinstance(value, bool):
                return value, ""
        elif kind == "int":
            if isinstance(value, int) and not isinstance(value, bool):
                return value, ""
        elif kind == "float":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value), ""
        elif kind == "str":
            if isinstance(value, str):
                if self.choices and value not in self.choices:
                    return None, f"{self.key}: {value!r} is not one of {', '.join(self.choices)}"
                return value, ""
        elif kind == "list":
            if isinstance(value, (list, tuple)):
                return list(value), ""
        elif kind == "dict":
            if isinstance(value, dict):
                return dict(value), ""
        return None, f"{self.key}: expected {kind}, got {_name_of(value)} ({value!r})"

    def parse(self, text: str) -> tuple[Any, str]:
        """Accept the string form an environment variable has to use."""
        kind = self.kind
        if kind == "bool":
            low = text.strip().lower()
            if low in _TRUE:
                return True, ""
            if low in _FALSE:
                return False, ""
            return None, f"{self.key}: {text!r} is not a boolean"
        if kind in ("int", "float"):
            try:
                return (int(text, 10), "") if kind == "int" else (float(text), "")
            except ValueError:
                return None, f"{self.key}: {text!r} is not a {kind}"
        if kind == "list":
            if text.lstrip().startswith("["):
                return self._json(text)
            return [part.strip() for part in text.split(",") if part.strip()], ""
        if kind == "dict":
            return self._json(text)
        return self.coerce(text)

    def _json(self, text: str) -> tuple[Any, str]:
        try:
            return self.coerce(json.loads(text))
        except json.JSONDecodeError as exc:
            return None, f"{self.key}: {text!r} is not JSON ({exc.msg})"


def _name_of(value: Any) -> str:
    return {bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "dict"}.get(type(value), type(value).__name__)


# -- the schema -------------------------------------------------------------
#
# Adding a key here is what makes it readable, writable, settable from the
# environment and visible in `/settings`.  Nothing else needs editing.

SCHEMA: Final[tuple[Spec, ...]] = (
    Spec("model.default", "str", "claude-sonnet-4-20250514", "model a new session starts on"),
    Spec("model.roles", "dict", {}, "role -> model id for the multi-model ensemble"),
    Spec("model.thinking", "bool", False, "ask for thinking output where the model supports it"),
    Spec("tools.approvalMode", "str", "safe", "how much runs without asking", ("safe", "auto-edit", "yolo", "full")),
    Spec("tools.timeout", "float", 120.0, "seconds one tool call may take"),
    Spec("tools.disabled", "list", [], "tool names to hide from the model"),
    Spec("tools.parallel", "int", 4, "how many parallel-safe calls run at once"),
    Spec("tools.custom", "bool", True, "load user tools from ~/.offset/tools"),
    Spec("context.maxBytesPerFile", "int", 32_768, "cap on one instruction file"),
    Spec("context.maxBytesTotal", "int", 131_072, "cap on all instruction files together"),
    Spec("context.extraFiles", "list", [], "further instruction file names to look for"),
    Spec("session.compactAt", "int", 0, "token count that triggers compaction, 0 disables"),
    Spec("snapshots.enabled", "bool", True, "checkpoint the workspace before edits"),
    Spec("snapshots.maxBytes", "int", 2_000_000, "largest file a snapshot will copy"),
    Spec("speculate.branches", "int", 3, "default number of speculative branches"),
    Spec("speculate.keep", "bool", False, "keep branch worktrees after a run"),
    Spec("providers.baseUrls", "dict", {}, "provider -> base URL override"),
    Spec("providers.disabled", "list", [], "providers to ignore even when a key exists"),
    Spec("eggs.enabled", "bool", True, "easter eggs"),
    Spec("ui.ascii", "bool", False, "draw with ASCII only"),
)

BY_KEY: Final[dict[str, Spec]] = {spec.key: spec for spec in SCHEMA}
BY_ENV: Final[dict[str, Spec]] = {spec.env: spec for spec in SCHEMA}


def home() -> Path:
    """`~/.offset`, or wherever `OFFSET_HOME` points.  Read late, never cached
    at import: the tests and `--home` both move it after this module loads."""
    return Path(os.environ.get("OFFSET_HOME") or (Path.home() / ".offset"))


def _flatten(raw: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Dotted leaves of a config file.  Descent stops at a declared dict key so
    `{"model": {"roles": {...}}}` stays one value instead of becoming many."""
    if not isinstance(raw, dict):
        return
    for name, value in raw.items():
        key = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(value, dict) and key not in BY_KEY:
            yield from _flatten(value, key)
        else:
            yield key, value


def _merge(low: Any, high: Any) -> Any:
    """Deep merge for dicts so a higher layer can set one nested key without
    deleting its siblings.  Lists and scalars replace wholesale."""
    if isinstance(low, dict) and isinstance(high, dict):
        out = dict(low)
        for key, value in high.items():
            out[key] = _merge(out[key], value) if key in out else value
        return out
    return high


def _suggest(key: str) -> str:
    near = difflib.get_close_matches(key, list(BY_KEY), n=1, cutoff=0.6)
    return f"; did you mean {near[0]!r}?" if near else ""


class Settings:
    """The five layers and the one file that may be written."""

    __slots__ = ("workspace", "_home", "_layers", "_pending", "_problems", "_cache", "_lock", "_timer")

    def __init__(self, workspace: str | os.PathLike[str] | None = None) -> None:
        self.workspace = Path(workspace).resolve() if workspace else Path.cwd()
        self._home = home()
        self._layers: dict[str, dict[str, Any]] = {name: {} for name in ORDER[1:]}
        self._pending: dict[str, Any] = {}
        self._problems: list[str] = []
        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self.reload()

    # -- loading ----------------------------------------------------------

    def reload(self) -> "Settings":
        """Re-read every file and the environment.  Runtime overrides and
        unwritten `set()` values survive, because they came from this process."""
        with self._lock:
            self._home = home()
            self._problems.clear()
            self._cache.clear()
            self._layers["user"] = self._read(self.user_file)
            self._layers["project"] = self._read(self.project_file)
            self._layers["env"] = self._environment()
            self._layers["user"].update(self._pending)
        return self

    def use(self, workspace: str | os.PathLike[str]) -> "Settings":
        """Point the project layer at another workspace."""
        self.workspace = Path(workspace).resolve()
        return self.reload()

    @property
    def user_file(self) -> Path:
        return self._home / "config.json"

    @property
    def project_file(self) -> Path:
        return self.workspace / ".offset" / "config.json"

    def _read(self, path: Path) -> dict[str, Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except OSError as exc:
            self._note(f"{path}: unreadable ({exc.strerror})")
            return {}
        except json.JSONDecodeError as exc:
            self._note(f"{path}: not valid JSON (line {exc.lineno}: {exc.msg}) — ignored, nothing was changed")
            return {}
        if not isinstance(raw, dict):
            self._note(f"{path}: top level must be an object, found {_name_of(raw)}")
            return {}
        out: dict[str, Any] = {}
        for key, value in _flatten(raw):
            spec = BY_KEY.get(key)
            if spec is None:
                self._note(f"{path}: unknown setting {key!r}{_suggest(key)}")
                continue
            coerced, why = spec.coerce(value)
            if why:
                self._note(f"{path}: {why} — using {spec.default!r}")
                continue
            out[key] = coerced
        return out

    def _environment(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, text in os.environ.items():
            if not name.startswith("OFFSET_") or name in RESERVED:
                continue
            spec = BY_ENV.get(name)
            if spec is None:
                self._note(f"{name}: no such setting{_suggest(name[7:].lower())}")
                continue
            value, why = spec.parse(text)
            if why:
                self._note(f"{name}: {why} — using {spec.default!r}")
                continue
            out[spec.key] = value
        return out

    def _note(self, message: str) -> None:
        if message not in self._problems:
            self._problems.append(message)

    def problems(self) -> list[str]:
        """Every complaint since the last reload.  Show these; a silent config
        error is the bug this whole module exists to prevent."""
        with self._lock:
            return list(self._problems)

    # -- reading ----------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Effective value of a dotted key.

        `default` is only a courtesy for an *undeclared* key; a declared key
        always falls back to its schema default, so the schema stays the one
        place a default is written down.
        """
        with self._lock:
            if key in self._cache:
                value = self._cache[key]
            else:
                spec = BY_KEY.get(key)
                if spec is None:
                    self._note(f"read of unknown setting {key!r}{_suggest(key)}")
                    return default
                value = spec.default
                for name in ORDER[1:]:
                    layer = self._layers[name]
                    if key in layer:
                        value = _merge(value, layer[key])
                self._cache[key] = value
            return copy.deepcopy(value) if isinstance(value, (dict, list)) else value

    def source(self, key: str) -> str:
        """Which layer supplies `key` right now — for `/settings`."""
        with self._lock:
            if key not in BY_KEY:
                return "unknown"
            for name in reversed(ORDER[1:]):
                if key in self._layers[name]:
                    return name
            return "defaults"

    def effective(self) -> dict[str, Any]:
        return {spec.key: self.get(spec.key) for spec in SCHEMA}

    def layers(self) -> list[tuple[str, Path]]:
        """Name and origin of each layer, lowest precedence first.  The three
        layers that are not files name themselves in angle brackets."""
        return [
            ("defaults", Path("<schema>")),
            ("user", self.user_file),
            ("project", self.project_file),
            ("env", Path("<environment>")),
            ("runtime", Path("<runtime>")),
        ]

    # -- writing ----------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Write the user layer.  Raises `SettingsError` with a message worth
        showing when the key or the value is wrong."""
        spec = BY_KEY.get(key)
        if spec is None:
            raise SettingsError(f"unknown setting {key!r}{_suggest(key)}")
        coerced, why = spec.coerce(value)
        if why:
            raise SettingsError(why)
        with self._lock:
            self._layers["user"][key] = coerced
            self._pending[key] = coerced
            self._cache.pop(key, None)
            self._arm()

    def override(self, key: str, value: Any) -> None:
        """The runtime layer: CLI flags.  Never persisted."""
        spec = BY_KEY.get(key)
        if spec is None:
            raise SettingsError(f"unknown setting {key!r}{_suggest(key)}")
        coerced, why = spec.coerce(value)
        if why:
            raise SettingsError(why)
        with self._lock:
            self._layers["runtime"][key] = coerced
            self._cache.pop(key, None)

    def unset(self, key: str) -> bool:
        """Drop a key from the user layer, revealing the layer below."""
        with self._lock:
            if key not in self._layers["user"]:
                return False
            del self._layers["user"][key]
            self._pending.pop(key, None)
            self._cache.pop(key, None)
            self._removed().add(key)
            self._arm()
            return True

    def _removed(self) -> set[str]:
        return self._pending.setdefault("\0removed", set())  # type: ignore[return-value]

    def _arm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(DEBOUNCE, self.flush)
        self._timer.daemon = True  # a pending write must not hold up exit
        self._timer.start()

    def flush(self) -> bool:
        """Write pending changes now.  False (plus a `problems()` entry) when
        the write failed; the previous file is then still intact and the
        changes stay pending for the next attempt."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            pending = {k: v for k, v in self._pending.items() if k != "\0removed"}
            # NB: this module exports `set()` per the settings contract, which
            # shadows the builtin here - an empty tuple avoids the collision.
            removed = self._pending.get("\0removed") or ()
            if not pending and not removed:
                return True
            path = self.user_file
            raw = self._reread(path)
            for key, value in pending.items():
                _plant(raw, key, value)
            for key in removed:
                _uproot(raw, key)
            if not self._atomic(path, raw):
                return False
            self._pending.clear()
            return True

    def _reread(self, path: Path) -> dict[str, Any]:
        """The file as it is on disk right now, so an external edit made while
        this session was open is not lost by the write."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def _atomic(self, path: Path, raw: dict[str, Any]) -> bool:
        body = json.dumps(raw, indent=2, sort_keys=True) + "\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".config-", suffix=".json")
        except OSError as exc:
            self._note(f"{path}: cannot be written ({exc.strerror})")
            return False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())  # the rename is only atomic if the data is already down
            os.replace(tmp, path)
        except OSError as exc:
            self._note(f"{path}: write failed ({exc.strerror or exc}), the file on disk is unchanged")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
        return True


def _plant(raw: dict[str, Any], key: str, value: Any) -> None:
    """Set a dotted key in the nested shape a human edits by hand.  A branch
    that is currently a scalar is replaced: it cannot hold a child."""
    parts = key.split(".")
    node = raw
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _uproot(raw: dict[str, Any], key: str) -> None:
    parts = key.split(".")
    stack: list[tuple[dict[str, Any], str]] = []
    node = raw
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            return
        stack.append((node, part))
        node = child
    node.pop(parts[-1], None)
    for parent, part in reversed(stack):
        if not parent[part]:
            del parent[part]


# -- the process-wide instance ----------------------------------------------
#
# One instance, because settings describe one run.  The free functions are the
# contract everything else codes against.

_active = Settings()


def active() -> Settings:
    return _active


def configure(workspace: str | os.PathLike[str] | None = None, **overrides: Any) -> Settings:
    """Called once at startup: fix the workspace, apply CLI flags."""
    if workspace is not None:
        _active.use(workspace)
    for key, value in overrides.items():
        if value is not None:
            _active.override(key.replace("__", "."), value)
    return _active


def get(key: str, default: Any = None) -> Any:
    return _active.get(key, default)


def set(key: str, value: Any) -> None:  # noqa: A001 - the contract names it `set`
    _active.set(key, value)


def override(key: str, value: Any) -> None:
    _active.override(key, value)


def unset(key: str) -> bool:
    return _active.unset(key)


def layers() -> list[tuple[str, Path]]:
    return _active.layers()


def problems() -> list[str]:
    return _active.problems()


def source(key: str) -> str:
    return _active.source(key)


def effective() -> dict[str, Any]:
    return _active.effective()


def reload() -> Settings:
    return _active.reload()


def flush() -> bool:
    return _active.flush()


atexit.register(flush)  # a debounce must never eat the last `/set`
