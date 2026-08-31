"""Which debug adapter to run, and how to reach it.

Debuggers are not installed by default and cannot be vendored, so the only
honest design is to look for what is actually on this machine and say plainly
what is missing when nothing is.  A traceback because `debugpy` is absent tells
the user nothing they can act on; "no debug adapter for python - install one
with: pip install debugpy" tells them exactly what to do.

Candidates are ordered by how well they behave, not alphabetically.  `debugpy`
is preferred for Python because it is the reference implementation; `lldb-dap`
comes before `gdb --interpreter=dap` for native code because gdb's DAP bridge
is younger and its capability set is narrower.

The config file mirrors `mcp.json` deliberately, down to the two-file merge and
the errors-as-values contract: a user who has configured one subsystem should
not have to learn a second grammar for the next.  `dap.json` in `OFFSET_HOME`
holds machine-wide adapters, `<workspace>/.offset/dap.json` overrides per
project, and a malformed entry names its own field rather than raising.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable

from offset.core import settings
from offset.tools.debug.protocol import Channel, StdioChannel

#: The file name, in `OFFSET_HOME` and then in the workspace.
CONFIG_NAME: Final = "dap.json"

#: Either key is accepted at the top level, as `mcp.json` accepts two.
ADAPTER_KEYS: Final = ("adapters", "debugAdapters")

DEFAULT_TIMEOUT: Final = 20.0

#: Resolves an executable name to a path, or None.  Injectable for tests.
Which = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One way of getting a DAP server for a language.

    `module` is for adapters that are Python modules rather than executables:
    `debugpy` ships no launcher script on many installs, so the only reliable
    invocation is `python -m debugpy.adapter`.  Probing for the module rather
    than a binary is what makes detection work there.
    """

    command: str
    args: tuple[str, ...] = ()
    hint: str = ""
    module: str = ""

    def resolve(self, which: Which = shutil.which) -> tuple[str, tuple[str, ...]] | None:
        """The concrete argv for this candidate, or None if it is not here."""
        if self.module:
            if not _module_present(self.module):
                return None
            return sys.executable, ("-m", self.module, *self.args)
        found = which(self.command)
        if not found:
            return None
        return found, self.args


def _module_present(name: str) -> bool:
    """Whether an importable module exists, without importing it.

    `find_spec` executes parent packages, which for a debug adapter is both
    slow and a side effect at probe time, so the top-level name is checked
    against the finders directly.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(name.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


#: Language -> candidates, best first.
CANDIDATES: Final[dict[str, tuple[Candidate, ...]]] = {
    "python": (
        Candidate("python", ("--host", "127.0.0.1", "--port", "0"),
                  "pip install debugpy", module="debugpy.adapter"),
        Candidate("debugpy-adapter", (), "pip install debugpy"),
    ),
    "c": (
        Candidate("lldb-dap", (), "apt install lldb, or brew install llvm"),
        Candidate("lldb-vscode", (), "apt install lldb"),
        Candidate("gdb", ("--interpreter=dap",), "apt install gdb (12 or newer)"),
    ),
    "go": (
        Candidate("dlv", ("dap",), "go install github.com/go-delve/delve/cmd/dlv@latest"),
    ),
    "ruby": (
        Candidate("rdbg", ("--open", "--stop-at-load"), "gem install debug"),
    ),
    "node": (
        Candidate("js-debug-adapter", (), "npm i -g js-debug-adapter"),
    ),
}

#: Languages that share the native adapters.
CANDIDATES["cpp"] = CANDIDATES["c"]
CANDIDATES["rust"] = CANDIDATES["c"]

EXTENSIONS: Final[dict[str, str]] = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".js": "node",
    ".mjs": "node",
    ".ts": "node",
}


def language_for(path: Path | str) -> str:
    """The language of a file by extension, or an empty string."""
    return EXTENSIONS.get(Path(path).suffix.lower(), "")


def languages() -> list[str]:
    return sorted(CANDIDATES)


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """A user-declared adapter from `dap.json`."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    language: str = ""
    timeout: float = DEFAULT_TIMEOUT
    enabled: bool = True
    source: Path | None = None


@dataclass(frozen=True, slots=True)
class Config:
    adapters: tuple[AdapterConfig, ...] = ()
    errors: tuple[str, ...] = ()
    sources: tuple[Path, ...] = ()

    def for_language(self, language: str) -> AdapterConfig | None:
        """The configured adapter for a language, if the user declared one."""
        for adapter in self.adapters:
            if not adapter.enabled:
                continue
            if adapter.language == language or adapter.name == language:
                return adapter
        return None

    def report(self) -> list[str]:
        lines = [f"{a.name}: {a.command} {' '.join(a.args)}".rstrip() for a in self.adapters]
        lines.extend(f"config: {e}" for e in self.errors)
        return lines


def config_paths(workspace: Path | str | None = None) -> list[Path]:
    """Machine-wide first, workspace second: the later one wins."""
    paths = [settings.home() / CONFIG_NAME]
    if workspace is not None:
        paths.append(Path(workspace) / ".offset" / CONFIG_NAME)
    return paths


def load_config(workspace: Path | str | None = None, *, paths: Iterable[Path] | None = None) -> Config:
    """Read every config file that exists.  Never raises."""
    adapters: dict[str, AdapterConfig] = {}
    errors: list[str] = []
    sources: list[Path] = []
    for path in (list(paths) if paths is not None else config_paths(workspace)):
        if not path.exists():
            continue
        sources.append(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        parsed, problems = parse_config(raw, source=path)
        errors.extend(problems)
        for adapter in parsed:
            adapters[adapter.name] = adapter  # later file wins
    return Config(tuple(adapters.values()), tuple(errors), tuple(sources))


def parse_config(raw: Any, *, source: Path | None = None) -> tuple[list[AdapterConfig], list[str]]:
    """Validate one config document.  Problems name their own field."""
    where = f"{source}: " if source else ""
    if not isinstance(raw, dict):
        return [], [f"{where}the top level must be an object"]

    table = None
    for key in ADAPTER_KEYS:
        if isinstance(raw.get(key), dict):
            table = raw[key]
            break
    if table is None:
        return [], [f"{where}expected an {' or '.join(ADAPTER_KEYS)} object"]

    out: list[AdapterConfig] = []
    problems: list[str] = []
    for name, entry in sorted(table.items()):
        field_ = f"{where}adapters.{name}"
        if not isinstance(entry, dict):
            problems.append(f"{field_} must be an object")
            continue
        command = entry.get("command")
        args: list[str] = []
        if isinstance(command, list):
            if not command or not all(isinstance(x, str) for x in command):
                problems.append(f"{field_}.command must be a non-empty list of strings")
                continue
            command, args = command[0], list(command[1:])
        if not isinstance(command, str) or not command:
            problems.append(f"{field_}.command must be a string or list of strings")
            continue
        extra = entry.get("args", [])
        if extra:
            if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
                problems.append(f"{field_}.args must be a list of strings")
                continue
            args.extend(extra)
        env: dict[str, str] = {}
        if entry.get("env") is not None:
            if not isinstance(entry["env"], dict):
                problems.append(f"{field_}.env must be an object of strings")
                continue
            for key, value in entry["env"].items():
                if not isinstance(value, str):
                    problems.append(f"{field_}.env.{key} must be a string")
                    continue
                env[str(key)] = value
        timeout = entry.get("timeout", DEFAULT_TIMEOUT)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            problems.append(f"{field_}.timeout must be a positive number")
            continue
        enabled = entry.get("enabled", entry.get("enable", True))
        if not isinstance(enabled, bool):
            problems.append(f"{field_}.enabled must be true or false")
            continue
        out.append(AdapterConfig(
            name=str(name),
            command=command,
            args=tuple(args),
            env=env,
            cwd=str(entry.get("cwd", "") or ""),
            language=str(entry.get("language", "") or ""),
            timeout=float(timeout),
            enabled=enabled,
            source=source,
        ))
    return out, problems


@dataclass(frozen=True, slots=True)
class Launch:
    """A resolved adapter, ready to run."""

    language: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    timeout: float = DEFAULT_TIMEOUT
    source: str = "discovered"

    @property
    def label(self) -> str:
        return f"{self.command} {' '.join(self.args)}".strip()

    def channel(self) -> Channel:
        """A transport for this adapter.

        The environment is layered onto the real one rather than replacing it:
        an adapter that cannot see `PATH` or `HOME` fails in ways that look like
        a protocol bug.
        """
        env = dict(os.environ)
        env.update(self.env)
        return StdioChannel(
            self.command,
            list(self.args),
            env=env,
            cwd=self.cwd or None,
        )


def missing_message(language: str) -> str:
    """What to install, for a language with no adapter here."""
    candidates = CANDIDATES.get(language)
    if not candidates:
        known = ", ".join(languages())
        target = language or "that file type"
        return f"no debug adapter is known for {target}. offset can debug: {known}"
    hints = [c.hint for c in candidates if c.hint]
    unique: list[str] = []
    for hint in hints:
        if hint not in unique:
            unique.append(hint)
    return f"no debug adapter for {language}. install one with: " + "; or ".join(unique)


def choose(
    language: str,
    *,
    config: Config | None = None,
    which: Which = shutil.which,
) -> tuple[Launch | None, str]:
    """The adapter to use for a language.  Returns `(launch, "")` or `(None, why)`.

    A configured adapter wins outright, including when its command is missing -
    silently falling back to a discovered one would hide the user's own typo.
    """
    settings_config = config if config is not None else load_config()
    declared = settings_config.for_language(language)
    if declared is not None:
        found = which(declared.command)
        if not found and not Path(declared.command).exists():
            return None, (
                f"{CONFIG_NAME} declares {declared.command!r} for {language}, "
                "which is not on PATH"
            )
        return Launch(
            language=language,
            command=found or declared.command,
            args=declared.args,
            env=dict(declared.env),
            cwd=declared.cwd,
            timeout=declared.timeout,
            source=str(declared.source or CONFIG_NAME),
        ), ""

    for candidate in CANDIDATES.get(language, ()):
        resolved = candidate.resolve(which)
        if resolved is None:
            continue
        command, args = resolved
        return Launch(language=language, command=command, args=args), ""

    return None, missing_message(language)


def available(*, which: Which = shutil.which, config: Config | None = None) -> list[str]:
    """Languages this machine can actually debug right now."""
    return [lang for lang in languages() if choose(lang, config=config, which=which)[0] is not None]


def report(*, which: Which = shutil.which, config: Config | None = None) -> list[str]:
    """One line per language, saying what would run or what is missing."""
    lines = []
    for language in languages():
        launch, why = choose(language, config=config, which=which)
        if launch is None:
            lines.append(f"{language:8s} -  {why}")
        else:
            lines.append(f"{language:8s} +  {launch.label}  ({launch.source})")
    return lines
