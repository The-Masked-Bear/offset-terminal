"""Plugins: what a third party may add, and what it must be trusted to do.

`custom.py` answers "how does someone else's tool get in".  This module answers
the two questions that were left open, and they are both about consent.

**Consent to run.**  A plugin is a file in a directory, and importing it is
executing it.  Doing that at startup for anything that happens to be on disk
means a stray download, a stale experiment or a package that ships an entry
point gets the agent's full authority without anyone saying yes.  So every
plugin is content-addressed and checked against `~/.offset/plugins/trusted.json`
before it runs.  An unknown or changed plugin is *quarantined*: described on
screen, offered for approval, and not executed.  For a Python plugin quarantine
has to mean "not imported", because import is the dangerous act — so what a
quarantined plugin claims to export is read out of its syntax tree instead, and
`ast` never runs a line of it.  A manifest is data, so a quarantined manifest is
parsed and its tools are built, but they are not registered, and its own claim
about how dangerous it is carries no weight at all: an unapproved manifest
saying `"danger": "safe"` would clear the approval threshold in every mode,
which is precisely the gate it is trying to walk around.  Untrusted means FULL.

**Consent to be silent.**  Discovery must never raise, but the previous version
of "never raise" was to collect load errors and throw them away, so a plugin
with a syntax error looked exactly like a plugin that was never written.  Every
failure here is a value on the `Plugin` that caused it, and `/plugins` prints
them.

Beyond tools, a plugin may fill three extension points — `TOOLS`, `COMMANDS`
and `HOOKS` — and declare `PLUGIN_API` so a plugin written against a contract
this build does not implement is refused by name rather than half-loaded.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Final, Iterable, Mapping

from offset.core import settings
from offset.shell import commands as shell_commands
from offset.shell.commands import TONE_ERR, TONE_INFO, TONE_OK, Command, Outcome
from offset.tools.base import Tool, Toolbox
from offset.tools.custom import (
    MANIFEST_NAMES,
    default_dirs,
    import_plugin,
    load_manifest,
    tools_from,
)

#: The extension-point contract this build implements.  A plugin declaring a
#: different number is refused rather than loaded hopefully: the whole point of
#: the number is that we cannot know what changed.
PLUGIN_API: Final = 1

#: Where a pip-installed package advertises itself.
ENTRY_POINT_GROUP: Final = "offset.plugins"

#: Lifecycle points a plugin may hook.  A closed set on purpose — a mistyped
#: event name would otherwise be a hook that silently never fires.
EVENTS: Final = ("startup", "shutdown", "turn_start", "turn_end", "tool_call", "tool_result")

#: The names a plugin module may bind to fill an extension point.
EXPORTS: Final = ("TOOLS", "TOOL", "COMMANDS", "HOOKS")

PYTHON: Final = "python"
MANIFEST: Final = "manifest"
ENTRY_POINT: Final = "entrypoint"


# -- the trust store --------------------------------------------------------


def store_path() -> Path:
    """`$OFFSET_HOME/plugins/trusted.json`, resolved late so tests can move it."""
    return settings.home() / "plugins" / "trusted.json"


@dataclass(slots=True)
class Trust:
    """Which plugins the user approved, by name, and what they looked like.

    The value is a digest rather than a boolean because approval is of a
    *version*: a trusted plugin that is edited afterwards is a different
    plugin, and re-running it on the strength of the old decision would make
    the gate decorative.
    """

    path: Path
    entries: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Trust":
        target = Path(path) if path is not None else store_path()
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            # No store, or an unreadable one, means nothing is approved. That
            # is the safe reading: a corrupt file must quarantine plugins, not
            # wave them through.
            return cls(target, {})
        table = raw.get("plugins") if isinstance(raw, dict) and isinstance(raw.get("plugins"), dict) else raw
        if not isinstance(table, dict):
            return cls(target, {})
        return cls(target, {str(k): v for k, v in table.items() if isinstance(v, str)})

    def approves(self, name: str, digest: str) -> bool:
        return bool(digest) and self.entries.get(name) == digest

    def trust(self, name: str, digest: str) -> None:
        self.entries[name] = digest
        self.save()

    def distrust(self, name: str) -> bool:
        if self.entries.pop(name, None) is None:
            return False
        self.save()
        return True

    def save(self) -> None:
        """Rewrite the store atomically, and only readable by its owner."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"plugins": dict(sorted(self.entries.items()))}, indent=1)
        handle, temporary = tempfile.mkstemp(dir=str(self.path.parent), prefix=".trusted-")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                fh.write(body + "\n")
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            raise

    def report(self) -> list[str]:
        if not self.entries:
            return ["no plugins are trusted yet"]
        return [f"{name} {digest[:12]}" for name, digest in sorted(self.entries.items())]


def digest_of(paths: Iterable[Path]) -> str:
    """A content hash over every file a plugin can run.

    A manifest alone is not enough: it names an executable, and swapping that
    executable out would change what the approval actually covers while leaving
    the approved file untouched.  Only the base name goes into the hash, not the
    full path, so moving a plugin directory does not revoke its approval.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        try:
            body = path.read_bytes()
        except OSError:
            digest.update(b"\0unreadable\0")
            continue
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


# -- reading a plugin without running it ------------------------------------


@dataclass(slots=True)
class Peek:
    """What a Python plugin claims, read from its syntax tree."""

    api: int | None = None
    exports: tuple[str, ...] = ()
    error: str = ""


def peek(path: Path) -> Peek:
    """Read a plugin's declarations without executing a line of it.

    This is what lets an unapproved or incompatible plugin be described on
    screen and refused: a version check that had to import the module first
    would already have lost the argument it was there to win.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, ValueError) as exc:
        return Peek(error=f"{type(exc).__name__}: {exc}")
    api: int | None = None
    exports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        for name in names:
            if name == "PLUGIN_API":
                api = _literal_int(getattr(node, "value", None), api)
            elif name in EXPORTS:
                exports.append(name)
    return Peek(api=api, exports=tuple(dict.fromkeys(exports)))


def _literal_int(node: Any, fallback: int | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return fallback


def _api_complaint(api: int | None) -> str:
    """The refusal message, which has to name the version that was asked for."""
    if api is None or api == PLUGIN_API:
        return ""
    return f"asks for plugin api {api}; this build implements plugin api {PLUGIN_API}"


# -- one plugin -------------------------------------------------------------


@dataclass(slots=True)
class Plugin:
    """One plugin: where it came from, whether it ran, and what it gave us."""

    name: str
    kind: str = PYTHON
    origin: str = ""
    source: Path | None = None
    digest: str = ""
    api: int | None = None
    trusted: bool = False
    tools: list[Tool] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    hooks: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)
    #: Claimed extension points, for a plugin that was never executed.
    claims: tuple[str, ...] = ()
    error: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def quarantined(self) -> bool:
        """Loadable, but nobody has approved this version of it."""
        return self.ok and not self.trusted

    @property
    def active(self) -> bool:
        return self.ok and self.trusted

    def summary(self) -> str:
        parts: list[str] = []
        if self.tools:
            parts.append(f"{len(self.tools)} tool(s)")
        if self.commands:
            parts.append(f"{len(self.commands)} command(s)")
        if self.hooks:
            parts.append(f"{sum(len(v) for v in self.hooks.values())} hook(s)")
        if not parts and self.claims:
            parts.append("claims " + ", ".join(self.claims))
        return ", ".join(parts) or "nothing"

    def report(self) -> list[str]:
        if self.error:
            head = f"{self.name:<18} broken    {self.error}"
        elif self.trusted:
            head = f"{self.name:<18} loaded    {self.summary()}"
        else:
            head = f"{self.name:<18} untrusted {self.summary()} — /plugins trust {self.name}"
        lines = [head]
        lines.extend(f"  {note}" for note in self.notes)
        return lines


# -- collecting the extension points ---------------------------------------


def _commands_from(module: Any, notes: list[str]) -> list[Command]:
    raw = getattr(module, "COMMANDS", None)
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        notes.append("COMMANDS must be a list of Command objects")
        return []
    out: list[Command] = []
    for item in raw:
        if isinstance(item, Command) and item.name and callable(item.run):
            out.append(item)
        else:
            notes.append(f"COMMANDS entries must be Command objects, got {type(item).__name__}")
    return out


def _hooks_from(module: Any, notes: list[str]) -> dict[str, list[Callable[..., Any]]]:
    raw = getattr(module, "HOOKS", None)
    if raw is None:
        return {}
    items = list(raw.items()) if isinstance(raw, Mapping) else list(raw) if isinstance(raw, (list, tuple)) else None
    if items is None:
        notes.append("HOOKS must be a mapping of event name to callable")
        return {}
    out: dict[str, list[Callable[..., Any]]] = {}
    for item in items:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            notes.append("HOOKS entries must be (event, callable) pairs")
            continue
        event, handlers = item
        if event not in EVENTS:
            notes.append(f"unknown hook {event!r}; known events: {', '.join(EVENTS)}")
            continue
        for one in handlers if isinstance(handlers, (list, tuple)) else [handlers]:
            if not callable(one):
                notes.append(f"hook {event} is not callable")
                continue
            out.setdefault(str(event), []).append(one)
    return out


def _fill(plugin: Plugin, module: Any) -> None:
    """Read the three extension points off an executed plugin."""
    try:
        plugin.tools = [tool for tool in tools_from(module) if getattr(tool, "name", "")]
    except Exception as exc:
        plugin.notes.append(f"TOOLS could not be built: {type(exc).__name__}: {exc}")
        plugin.tools = []
    plugin.commands = _commands_from(module, plugin.notes)
    plugin.hooks = _hooks_from(module, plugin.notes)


# -- discovery --------------------------------------------------------------


def plugin_name(path: Path, root: Path) -> str:
    """A plugin's stable identity: what `/plugins trust` is typed against.

    A manifest is normally called `tool.json`, which says nothing, so a
    manifest in its own directory is named after that directory.
    """
    if path.name in MANIFEST_NAMES and path.parent != root:
        return path.parent.name
    return path.stem


def _manifest_binaries(path: Path) -> list[Path]:
    """The executables a manifest names, if they live beside it."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    out: list[Path] = []
    for entry in raw if isinstance(raw, list) else [raw]:
        if not isinstance(entry, dict):
            continue
        command = entry.get("command")
        first = command[0] if isinstance(command, list) and command else command
        if not isinstance(first, str):
            continue
        beside = path.parent / first
        if beside.is_file():
            out.append(beside)
    return out


def _candidates(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_dir() or ".offset-skip" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name in MANIFEST_NAMES or (path.suffix == ".py" and not path.name.startswith("_")):
            out.append(path)
    return out


def _load_manifest_plugin(path: Path, name: str, trust: Trust) -> Plugin:
    digest = digest_of([path, *_manifest_binaries(path)])
    trusted = trust.approves(name, digest)
    plugin = Plugin(name=name, kind=MANIFEST, origin=str(path), source=path,
                    digest=digest, trusted=trusted)
    try:
        # A manifest is data: parsing it runs nothing, so an unapproved one can
        # still be described. `trusted=False` clamps every tool it builds to
        # FULL, so its self-declared danger buys it nothing.
        plugin.tools = load_manifest(path, trusted=trusted)
    except Exception as exc:
        plugin.error = f"{type(exc).__name__}: {exc}"
        return plugin
    if not trusted:
        plugin.notes.append("danger clamped to full: the manifest is not approved")
    return plugin


def _load_python_plugin(path: Path, name: str, trust: Trust) -> Plugin:
    digest = digest_of([path])
    seen = peek(path)
    plugin = Plugin(name=name, kind=PYTHON, origin=str(path), source=path,
                    digest=digest, api=seen.api, claims=seen.exports)
    if seen.error:
        plugin.error = seen.error
        return plugin
    complaint = _api_complaint(seen.api)
    if complaint:
        plugin.error = complaint
        return plugin
    if not trust.approves(name, digest):
        plugin.notes.append("not imported: importing a plugin is running it")
        return plugin
    try:
        module = import_plugin(path)
    except BaseException as exc:  # a plugin may raise SystemExit at import
        plugin.error = f"{type(exc).__name__}: {exc}"
        return plugin
    declared = getattr(module, "PLUGIN_API", None)
    complaint = _api_complaint(declared if isinstance(declared, int) and not isinstance(declared, bool) else None)
    if complaint:
        plugin.api = declared
        plugin.error = complaint
        return plugin
    plugin.trusted = True
    _fill(plugin, module)
    return plugin


def _load_entry_point(entry: Any, trust: Trust) -> Plugin:
    value = getattr(entry, "value", "") or f"{getattr(entry, 'module', '')}:{getattr(entry, 'attr', '')}"
    # An installed package's files are not hashed: pip may replace them between
    # runs, so what the user approves is the import path itself — "yes, import
    # this module" — and the digest records exactly that sentence.
    digest = hashlib.sha256(f"{ENTRY_POINT_GROUP}:{entry.name}={value}".encode("utf-8")).hexdigest()
    plugin = Plugin(name=entry.name, kind=ENTRY_POINT, origin=value, digest=digest)
    if not trust.approves(entry.name, digest):
        plugin.notes.append(f"not imported: approve {value} with /plugins trust {entry.name}")
        return plugin
    try:
        loaded = entry.load()
    except BaseException as exc:
        plugin.error = f"{type(exc).__name__}: {exc}"
        return plugin
    if isinstance(loaded, ModuleType):
        declared = getattr(loaded, "PLUGIN_API", None)
        complaint = _api_complaint(declared if isinstance(declared, int) and not isinstance(declared, bool) else None)
        if complaint:
            plugin.api = declared
            plugin.error = complaint
            return plugin
        plugin.trusted = True
        _fill(plugin, loaded)
        return plugin
    if callable(loaded):
        try:
            loaded = loaded()
        except Exception as exc:
            plugin.error = f"{type(exc).__name__}: {exc}"
            return plugin
    if isinstance(loaded, ModuleType):
        plugin.trusted = True
        _fill(plugin, loaded)
        return plugin
    if isinstance(loaded, (list, tuple)) and all(isinstance(t, Tool) for t in loaded):
        plugin.trusted = True
        plugin.tools = [t for t in loaded if getattr(t, "name", "")]
        return plugin
    plugin.error = (
        f"entry point must resolve to a module, or to a callable returning a list of Tools, "
        f"got {type(loaded).__name__}"
    )
    return plugin


def discover(
    *,
    workspace: Path | str | None = None,
    dirs: Iterable[Path] | None = None,
    trust: Trust | None = None,
    entry_points: bool = True,
) -> "Registry":
    """Find every plugin, decide what may run, and never raise."""
    roots = [Path(d).expanduser() for d in (dirs if dirs is not None else default_dirs(
        Path(workspace) if workspace is not None else None))]
    store = trust if trust is not None else Trust.load()
    registry = Registry(trust=store, dirs=roots, entry_points=entry_points)
    seen: set[str] = set()
    for root in roots:
        for path in _candidates(root):
            name = plugin_name(path, root)
            if name in seen:
                registry.plugins.append(Plugin(
                    name=name, kind=MANIFEST if path.name in MANIFEST_NAMES else PYTHON,
                    origin=str(path), source=path,
                    error="shadowed by a plugin of the same name earlier on the path",
                ))
                continue
            seen.add(name)
            if path.name in MANIFEST_NAMES:
                registry.plugins.append(_load_manifest_plugin(path, name, store))
            else:
                registry.plugins.append(_load_python_plugin(path, name, store))
    if entry_points:
        registry.plugins.extend(_entry_point_plugins(store, seen))
    return registry


def _entry_point_plugins(trust: Trust, seen: set[str]) -> list[Plugin]:
    try:
        found = list(metadata.entry_points(group=ENTRY_POINT_GROUP))
    except Exception as exc:
        return [Plugin(name=ENTRY_POINT_GROUP, kind=ENTRY_POINT, origin=ENTRY_POINT_GROUP,
                       error=f"entry points could not be read: {type(exc).__name__}: {exc}")]
    out: list[Plugin] = []
    for entry in sorted(found, key=lambda e: e.name):
        if entry.name in seen:
            out.append(Plugin(name=entry.name, kind=ENTRY_POINT, origin=getattr(entry, "value", ""),
                              error="shadowed by a plugin file of the same name"))
            continue
        seen.add(entry.name)
        out.append(_load_entry_point(entry, trust))
    return out


# -- the extension-point registry ------------------------------------------


class Registry:
    """Every extension point a plugin can fill, and who filled it.

    The registry owns registration rather than its caller, for the same reason
    the MCP manager does: only it knows which Toolbox names and which slash
    commands came from a plugin, so only it can take them back out when that
    plugin is distrusted or reloaded.
    """

    __slots__ = ("_added", "_registered", "_table", "_toolbox", "dirs",
                 "entry_points", "notes", "plugins", "trust")

    def __init__(
        self,
        plugins: list[Plugin] | None = None,
        *,
        trust: Trust | None = None,
        dirs: Iterable[Path] | None = None,
        entry_points: bool = True,
    ) -> None:
        self.plugins: list[Plugin] = list(plugins or ())
        self.trust = trust if trust is not None else Trust.load()
        self.dirs: list[Path] = [Path(d) for d in (dirs or ())]
        self.entry_points = entry_points
        #: Things worth saying that nobody asked for yet: collisions from the
        #: last install, mostly. `/plugins` prints them.
        self.notes: list[str] = []
        self._toolbox: Toolbox | None = None
        self._registered: list[str] = []
        self._table: list[Command] | None = None
        self._added: list[Command] = []

    # -- accessors ----------------------------------------------------------

    @property
    def loaded(self) -> list[Plugin]:
        return [p for p in self.plugins if p.active]

    @property
    def quarantined(self) -> list[Plugin]:
        return [p for p in self.plugins if p.quarantined]

    @property
    def broken(self) -> list[Plugin]:
        return [p for p in self.plugins if not p.ok]

    def find(self, name: str) -> Plugin | None:
        return next((p for p in self.plugins if p.name == name), None)

    def tools(self) -> list[Tool]:
        """Every tool a trusted plugin contributes.  Quarantine yields none."""
        return [tool for plugin in self.loaded for tool in plugin.tools]

    def commands(self) -> list[Command]:
        return [command for plugin in self.loaded for command in plugin.commands]

    def hooks(self, event: str) -> list[Callable[..., Any]]:
        return [fn for plugin in self.loaded for fn in plugin.hooks.get(event, ())]

    def __len__(self) -> int:
        return len(self.loaded)

    # -- installing ---------------------------------------------------------

    def install(self, toolbox: Toolbox) -> list[str]:
        """Register every trusted plugin tool.  Returns the collisions."""
        self.withdraw()
        self._toolbox = toolbox
        clashes: list[str] = []
        for plugin in self.loaded:
            for tool in plugin.tools:
                if toolbox.get(tool.name) is not None:
                    clashes.append(f"plugin {plugin.name}: {tool.name} is already a tool, not registered")
                    continue
                toolbox.register(tool)
                self._registered.append(tool.name)
        self.notes = [*[n for n in self.notes if n not in clashes], *clashes]
        return clashes

    def withdraw(self) -> None:
        """Take back everything this registry put anywhere."""
        box = self._toolbox
        if box is not None:
            for name in self._registered:
                box.unregister(name)
        self._registered = []
        table = self._table
        if table is not None:
            for command in self._added:
                if command in table:
                    table.remove(command)
        self._added = []

    def publish_commands(self, table: list[Command] | None = None) -> list[str]:
        """Add plugin slash commands to the live command table.

        A plugin may not take a name the shell already uses.  Shadowing `/quit`
        or `/approve` would be a way for a plugin to trap the user in a session
        it controls, so the built-in wins and the clash is reported.
        """
        target = shell_commands.COMMANDS if table is None else table
        self._table = target
        taken = {c.name for c in target} | {a for c in target for a in c.aliases}
        clashes: list[str] = []
        for plugin in self.loaded:
            for command in plugin.commands:
                names = (command.name, *command.aliases)
                if any(n in taken for n in names):
                    clashes.append(f"plugin {plugin.name}: /{command.name} is already a command, not added")
                    continue
                target.append(command)
                self._added.append(command)
                taken.update(names)
        self.notes.extend(n for n in clashes if n not in self.notes)
        return clashes

    def fire(self, event: str, **payload: Any) -> list[str]:
        """Run every hook for `event`.  A failing hook is named, never fatal."""
        if event not in EVENTS:
            return [f"unknown hook event {event!r}; known events: {', '.join(EVENTS)}"]
        problems: list[str] = []
        for plugin in self.loaded:
            for hook in plugin.hooks.get(event, ()):
                try:
                    hook(**payload)
                except Exception as exc:
                    problems.append(f"plugin {plugin.name}: {event} hook failed: {type(exc).__name__}: {exc}")
        return problems

    # -- trust --------------------------------------------------------------

    def refresh(self) -> list[str]:
        """Re-discover from the same roots and re-install.  Returns notes."""
        box, table = self._toolbox, self._table
        self.withdraw()
        fresh = discover(dirs=self.dirs, trust=Trust.load(self.trust.path),
                         entry_points=self.entry_points)
        self.plugins = fresh.plugins
        self.trust = fresh.trust
        self.notes = []
        out: list[str] = []
        if box is not None:
            out.extend(self.install(box))
        if table is not None:
            out.extend(self.publish_commands(table))
        return out

    def approve(self, name: str) -> tuple[bool, list[str]]:
        """Trust one plugin at its current contents, then load it for real."""
        plugin = self.find(name)
        if plugin is None:
            known = ", ".join(p.name for p in self.plugins) or "none found"
            return False, [f"no plugin named {name!r}. known: {known}"]
        if not plugin.digest:
            return False, [f"{name} has nothing to hash, so it cannot be trusted"]
        if not plugin.ok:
            return False, [f"{name} is broken and trusting it would change nothing: {plugin.error}"]
        try:
            self.trust.trust(name, plugin.digest)
        except OSError as exc:
            return False, [f"could not write {self.trust.path}: {exc}"]
        lines = [f"{name} is trusted at {plugin.digest[:12]}"]
        lines.extend(self.refresh())
        after = self.find(name)
        if after is not None:
            lines.extend(after.report())
        return True, lines

    def revoke(self, name: str) -> tuple[bool, list[str]]:
        """Withdraw approval and unload whatever it had contributed."""
        if not self.trust.distrust(name):
            return False, [f"{name} was not trusted, so there was nothing to revoke"]
        lines = [f"{name} is no longer trusted"]
        lines.extend(self.refresh())
        return True, lines

    # -- rendering ----------------------------------------------------------

    def report(self) -> list[str]:
        if not self.plugins:
            searched = ", ".join(str(d) for d in self.dirs) or "nowhere"
            return [f"no plugins found in {searched}"]
        lines: list[str] = []
        for plugin in self.plugins:
            lines.extend(plugin.report())
        lines.append("")
        lines.append(
            f"{len(self.loaded)} loaded, {len(self.quarantined)} untrusted, {len(self.broken)} broken"
        )
        lines.extend(self.notes)
        return lines


# -- startup wiring ---------------------------------------------------------

_ACTIVE: Registry | None = None


def active() -> Registry | None:
    """The registry `install` built, or None before startup.

    `ShellState` is a slotted dataclass owned by the integrator, so there is
    nowhere on it to hang this; a module-level handle is the honest alternative
    to editing a dataclass this module does not own.
    """
    return _ACTIVE


def install(state: Any) -> None:
    """Wire MCP and plugins into a built `ShellState`.

    Called once, after the Toolbox exists.  Everything it has to say lands in
    `active().notes` rather than on stdout: startup is not a place to print,
    and `/plugins` is where the user goes to look.
    """
    global _ACTIVE
    notes: list[str] = []
    manager = getattr(state, "mcp", None)
    if manager is not None:
        notes.extend(manager.attach(state.toolbox))
    registry = discover(workspace=getattr(state, "workspace", None))
    notes.extend(registry.install(state.toolbox))
    notes.extend(registry.publish_commands())
    notes.extend(registry.fire("startup", state=state))
    registry.notes.extend(n for n in notes if n not in registry.notes)
    _ACTIVE = registry


def plugin_tools(workspace: Path | str | None = None) -> list[Tool]:
    """Every tool a trusted plugin contributes, for Toolbox construction."""
    return discover(workspace=workspace).tools()


# -- slash commands ---------------------------------------------------------

PLUGINS_USAGE: Final = "/plugins [trust <name>|distrust <name>|reload|trusted]"
MCP_USAGE: Final = "/mcp [reload [server]|connect <server>|disconnect <server>|resources [server]|read <server> <uri>]"


def _plugins(state: Any, args: list[str]) -> Outcome:
    """What loaded, what did not, and what needs approving."""
    global _ACTIVE
    registry = _ACTIVE
    if registry is None:
        # `/plugins` has to work whether or not startup called `install`, and
        # the registry it builds here is kept: without that, `trust` would
        # approve a plugin against a registry thrown away on the next keypress.
        registry = discover(workspace=getattr(state, "workspace", None))
        registry.install(state.toolbox)
        registry.publish_commands()
        _ACTIVE = registry
    verb = (args[0].lower() if args else "list")
    rest = args[1:]

    if verb in ("list", "ls", "status"):
        return Outcome(registry.report(), TONE_INFO)
    if verb == "trusted":
        return Outcome(registry.trust.report(), TONE_INFO)
    if verb == "reload":
        lines = registry.refresh()
        return Outcome([*registry.report(), *lines], TONE_OK)
    if verb in ("trust", "distrust"):
        if not rest:
            return Outcome.error(f"/plugins {verb} needs a plugin name.", PLUGINS_USAGE)
        ok, lines = registry.approve(rest[0]) if verb == "trust" else registry.revoke(rest[0])
        return Outcome(lines, TONE_OK if ok else TONE_ERR)
    return Outcome.error(f"no /plugins subcommand named {verb!r}.", PLUGINS_USAGE)


def _mcp_status(manager: Any) -> list[str]:
    rows = [
        f"{s.name:<16} {s.state:<10} {s.tools:>3} tools  {s.detail[:44]}"
        for s in manager.status()
    ]
    rows.extend(f"config: {problem}" for problem in manager.config.errors[:5])
    rows.extend(manager.collisions())
    if not rows:
        rows.append("no servers configured")
    return rows


def _mcp(state: Any, args: list[str]) -> Outcome:
    """MCP servers: what is running, and change it without a restart."""
    manager = state.mcp
    if manager is None:
        return Outcome([
            "no MCP servers configured",
            "add them to .offset/mcp.json or ~/.offset/mcp.json",
        ], TONE_INFO)
    verb = (args[0].lower() if args else "status")
    rest = args[1:]

    if verb in ("status", "list", "ls"):
        return Outcome(_mcp_status(manager), TONE_INFO)

    if verb == "reload":
        target = rest[0] if rest else None
        # Connecting spawns processes and handshakes with each of them, which
        # is far too slow to do on the keystroke that asked for it.
        def job() -> Outcome:
            lines = manager.reload(target)
            return Outcome([*lines, "", *_mcp_status(manager)], TONE_OK)
        return Outcome([f"reloading mcp {target or 'configuration'}..."], TONE_INFO, job=job)

    if verb in ("connect", "reconnect"):
        if not rest:
            return Outcome.error("/mcp connect needs a server name.", MCP_USAGE)
        name = rest[0]

        def job() -> Outcome:
            ok = manager.reconnect(name)
            head = (f"mcp {name}: connected, {len(manager.registered(name))} tool(s) registered"
                    if ok else f"mcp {name}: {manager.reason(name)}")
            return Outcome([head, *manager.collisions()], TONE_OK if ok else TONE_ERR)
        return Outcome([f"connecting to mcp {name}..."], TONE_INFO, job=job)

    if verb == "disconnect":
        if not rest:
            return Outcome.error("/mcp disconnect needs a server name.", MCP_USAGE)
        name = rest[0]
        if manager.config_for(name) is None:
            return Outcome.error(f"no mcp server named {name!r} is configured.", MCP_USAGE)
        withdrawn = len(manager.registered(name))
        manager.disconnect(name)
        return Outcome([f"mcp {name}: disconnected, {withdrawn} tool(s) withdrawn"], TONE_OK)

    if verb in ("resources", "prompts"):
        wanted = [rest[0]] if rest else None

        def job() -> Outcome:
            offerings = ([manager.offering(rest[0])] if wanted else manager.offerings())
            if not offerings:
                return Outcome(["no live mcp server to ask"], TONE_INFO)
            lines: list[str] = []
            for offering in offerings:
                lines.extend(offering.report())
            return Outcome(lines, TONE_OK if all(o.ok for o in offerings) else TONE_ERR)
        return Outcome(["listing mcp resources..."], TONE_INFO, job=job)

    if verb == "read":
        if len(rest) < 2:
            return Outcome.error("/mcp read needs a server and a resource uri.", MCP_USAGE)
        name, uri = rest[0], rest[1]

        def job() -> Outcome:
            got = manager.read_resource(name, uri)
            if not got.ok:
                return Outcome.error(f"mcp {name}: {got.error}")
            return Outcome([f"{name} {uri}", *got.text.splitlines()[:200]], TONE_OK)
        return Outcome([f"reading {uri} from mcp {name}..."], TONE_INFO, job=job)

    return Outcome.error(f"no /mcp subcommand named {verb!r}.", MCP_USAGE)


COMMANDS: Final[list[Command]] = [
    Command("plugins", "user plugins: what loaded, what needs trusting", _plugins,
            usage=PLUGINS_USAGE, aliases=("plugin",)),
    Command("mcp", "MCP servers: status, reload, connect, resources", _mcp, usage=MCP_USAGE),
]

__all__ = [
    "COMMANDS",
    "ENTRY_POINT_GROUP",
    "EVENTS",
    "PLUGIN_API",
    "Peek",
    "Plugin",
    "Registry",
    "Trust",
    "active",
    "digest_of",
    "discover",
    "install",
    "peek",
    "plugin_name",
    "plugin_tools",
    "store_path",
]
