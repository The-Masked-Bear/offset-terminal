"""Which language server to run, and the one process per workspace that runs it.

Discovery is deliberately dumb: a fixed table of the servers people actually
install, probed against `PATH` in a fixed order, first hit wins.  Nothing is
downloaded, nothing is guessed from a lock file, and no server is ever started
speculatively.  The alternative — inferring a toolchain from the project — gets
it wrong in exactly the repositories that are interesting, and gets it wrong
silently.

The order within a language is an opinion and worth stating.  For Python
`pyright-langserver` comes first because it answers `definition` and `hover`
correctly on typed code without configuration; `pylsp` second because it is a
pip install away; `ruff server` last because it is a superb linter and not a
navigator, so it is the right answer only when it is the only answer.

A missing server is the common case, not an error case.  It produces a sentence
naming the language, the commands that were looked for and how to install one
of them, because "no language server" without that list sends the reader to a
search engine.  Nothing here raises.

One process is kept per `(language, root)` and reused for the life of the
session.  Language servers are expensive to start and cheap to keep — indexing
a repository twice because two tool calls arrived is the difference between a
snappy session and a hot Pi — but they are also the easiest thing in this
process to orphan, so every one of them is reachable from `shutdown_all`.

Configuration overrides the table: `[$OFFSET_HOME]/lsp.json` then
`<workspace>/.offset/lsp.json`, workspace winning, exactly as `mcp.json` does.
It is read lazily rather than at construction, because `settings.home()` moves
under tests and under `--home`, and a cached path would quietly read the wrong
file.
"""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable

from offset.core import settings
from offset.tools.lsp.client import DIAGNOSTIC_WAIT, LSPClient, LSPError

CONFIG_NAME: Final = "lsp.json"

#: Both spellings are natural; accepting one only would be a papercut.
SERVER_KEYS: Final = ("servers", "languageServers")

DEFAULT_TIMEOUT: Final = 20.0


@dataclass(frozen=True, slots=True)
class Candidate:
    """One installable server for a language, and how to get it."""

    command: str
    args: tuple[str, ...] = ()
    install: str = ""

    @property
    def target(self) -> str:
        return " ".join([self.command, *self.args])

    @property
    def offer(self) -> str:
        return f"{self.command} ({self.install})" if self.install else self.command


_TYPESCRIPT: Final = (
    Candidate(
        "typescript-language-server",
        ("--stdio",),
        "npm install -g typescript-language-server typescript",
    ),
)
_CLANGD: Final = (Candidate("clangd", (), "apt install clangd, or brew install llvm"),)

#: Language -> the servers offset knows how to run, best first.
CANDIDATES: Final[dict[str, tuple[Candidate, ...]]] = {
    "python": (
        Candidate("pyright-langserver", ("--stdio",), "npm install -g pyright"),
        Candidate("pylsp", (), "pip install python-lsp-server"),
        Candidate("ruff", ("server",), "pip install ruff"),
    ),
    "typescript": _TYPESCRIPT,
    "javascript": _TYPESCRIPT,
    "go": (Candidate("gopls", (), "go install golang.org/x/tools/gopls@latest"),),
    "rust": (Candidate("rust-analyzer", (), "rustup component add rust-analyzer"),),
    "c": _CLANGD,
    "cpp": _CLANGD,
    "java": (Candidate("jdtls", (), "apt install jdtls, or unpack eclipse.jdt.ls"),),
    "lua": (
        Candidate("lua-ls", (), "apt install lua-language-server"),
        Candidate("lua-language-server", (), "apt install lua-language-server"),
    ),
}

#: Extension -> the key in `CANDIDATES`.  Deliberately not the LSP `languageId`
#: (`client._LANGUAGE_IDS` owns that): `.tsx` is `typescriptreact` to a server
#: but is served by the TypeScript one, and conflating the two picks nothing.
EXTENSIONS: Final[dict[str, str]] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".java": "java",
    ".lua": "lua",
}


def language_for(path: Path | str) -> str:
    """The `CANDIDATES` key for a file, or `""` when offset serves no server."""
    return EXTENSIONS.get(Path(path).suffix.lower(), "")


def languages() -> list[str]:
    return sorted(CANDIDATES)


# -- configuration ----------------------------------------------------------


@dataclass(slots=True)
class ServerConfig:
    """One language pinned to one command by an `lsp.json`."""

    language: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT
    enabled: bool = True
    source: Path | None = None

    @property
    def target(self) -> str:
        return " ".join([self.command, *self.args])


@dataclass(slots=True)
class Config:
    """Merged configuration plus every problem found, none of them fatal."""

    servers: dict[str, ServerConfig] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    sources: list[Path] = field(default_factory=list)

    def for_language(self, language: str) -> ServerConfig | None:
        return self.servers.get(language)


def config_paths(workspace: Path | str | None = None) -> list[Path]:
    """Lowest precedence first, matching how `load_config` merges them."""
    paths = [settings.home() / CONFIG_NAME]
    if workspace:
        paths.append(Path(workspace) / ".offset" / CONFIG_NAME)
    return paths


def load_config(workspace: Path | str | None = None, *, paths: Iterable[Path] | None = None) -> Config:
    """Read every config file, later files overriding earlier ones per language."""
    out = Config()
    for path in paths if paths is not None else config_paths(workspace):
        path = Path(path).expanduser()
        if not path.is_file():
            continue
        out.sources.append(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            out.errors.append(f"{path}: not valid JSON (line {exc.lineno}, column {exc.colno})")
            continue
        except OSError as exc:
            out.errors.append(f"{path}: unreadable ({exc.strerror or exc})")
            continue
        parsed, errors = parse_config(raw, source=path)
        out.errors.extend(errors)
        for server in parsed:
            out.servers[server.language] = server
    return out


def parse_config(raw: Any, *, source: Path | None = None) -> tuple[list[ServerConfig], list[str]]:
    """Validate one config document.  Errors name the offending field."""
    where = f"{source}: " if source else ""
    if not isinstance(raw, dict):
        return [], [f"{where}top level must be a JSON object"]
    table = next((raw[key] for key in SERVER_KEYS if key in raw), None)
    if table is None:
        return [], [f'{where}expected a "servers" object mapping a language to a command']
    if not isinstance(table, dict):
        return [], [f'{where}"servers" must be an object mapping a language to a command']

    servers: list[ServerConfig] = []
    errors: list[str] = []
    for language, entry in table.items():
        field_ = f"servers.{language}"
        problems: list[str] = []
        name = str(language).strip().lower()
        if not name:
            errors.append(f"{where}servers: a language key may not be empty")
            continue
        if not isinstance(entry, dict):
            errors.append(f"{where}{field_} must be an object")
            continue
        command, args = _command(entry, field_, problems)
        env = _strings(entry.get("env"), f"{field_}.env", problems)
        options = _object(entry.get("settings"), f"{field_}.settings", problems)
        timeout = _timeout(entry.get("timeout"), f"{field_}.timeout", problems)
        enabled = _flag(entry, field_, problems)
        if command is None and not problems:
            problems.append(f'{field_} needs "command"')
        if problems:
            errors.extend(f"{where}{problem}" for problem in problems)
            continue
        servers.append(
            ServerConfig(
                language=name,
                command=str(command),
                args=args,
                env=env,
                settings=options,
                timeout=timeout,
                enabled=enabled,
                source=source,
            )
        )
    return servers, errors


def _command(entry: dict[str, Any], field_: str, problems: list[str]) -> tuple[str | None, list[str]]:
    command = entry.get("command")
    extra = entry.get("args", [])
    inline: list[str] = []
    if command is None:
        return None, []
    if isinstance(command, list) and command and all(isinstance(part, str) for part in command):
        command, inline = command[0], list(command[1:])
    elif not isinstance(command, str):
        problems.append(f"{field_}.command must be a string, or a list of strings")
        return None, []
    if extra in (None, []):
        tail: list[str] = []
    elif isinstance(extra, list) and all(isinstance(arg, str) for arg in extra):
        tail = list(extra)
    else:
        problems.append(f"{field_}.args must be a list of strings")
        tail = []
    return command, inline + tail


def _strings(value: Any, field_: str, problems: list[str]) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(item, str) for item in value.values()):
        problems.append(f"{field_} must be an object with string values")
        return {}
    return {str(key): item for key, item in value.items()}


def _object(value: Any, field_: str, problems: list[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        problems.append(f"{field_} must be an object")
        return {}
    return dict(value)


def _timeout(value: Any, field_: str, problems: list[str]) -> float:
    if value is None:
        return DEFAULT_TIMEOUT
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        problems.append(f"{field_} must be a positive number of seconds")
        return DEFAULT_TIMEOUT
    return float(value)


def _flag(entry: dict[str, Any], field_: str, problems: list[str]) -> bool:
    if "enabled" in entry:
        if not isinstance(entry["enabled"], bool):
            problems.append(f"{field_}.enabled must be true or false")
            return True
        return bool(entry["enabled"])
    if entry.get("disabled") is not None:
        if not isinstance(entry["disabled"], bool):
            problems.append(f"{field_}.disabled must be true or false")
            return True
        return not entry["disabled"]
    return True


# -- choosing one ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Launch:
    """A command that exists on this machine, ready to be started."""

    language: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT
    #: Where the choice came from, for `status` — a config file, or PATH.
    source: str = "PATH"

    @property
    def target(self) -> str:
        return " ".join([self.command, *self.args])


#: How a command is looked up.  Injected so a test can decide what is installed
#: without putting scripts on the real PATH.
Which = Callable[[str], str | None]


def missing_message(language: str) -> str:
    """Why there is no server for this language, and what to install."""
    options = CANDIDATES.get(language)
    if not options:
        known = ", ".join(languages())
        return (
            f"offset has no language server for {language or 'this file type'}. "
            f"it knows: {known}. add one in {CONFIG_NAME} under \"servers\""
        )
    offers = "; ".join(option.offer for option in options)
    return (
        f"no {language} language server on PATH. install one of: {offers}. "
        f"or point offset at yours in {CONFIG_NAME}"
    )


def choose(language: str, *, config: Config | None = None, which: Which = shutil.which) -> tuple[Launch | None, str]:
    """The server to run for `language`, or the reason there is none.

    Errors are values: every caller of this is either a tool result or a status
    line, and both want the sentence rather than a traceback.
    """
    language = (language or "").strip().lower()
    if not language:
        return None, missing_message("")
    pinned = config.for_language(language) if config is not None else None
    if pinned is not None:
        if not pinned.enabled:
            return None, f"{language} language server is disabled in {pinned.source or CONFIG_NAME}"
        found = which(pinned.command)
        if not found:
            return None, (
                f"{pinned.source or CONFIG_NAME} sets {pinned.target!r} for {language}, "
                f"but {pinned.command!r} is not on PATH"
            )
        return (
            Launch(
                language=language,
                command=found,
                args=tuple(pinned.args),
                env=dict(pinned.env),
                settings=dict(pinned.settings),
                timeout=pinned.timeout,
                source=str(pinned.source or CONFIG_NAME),
            ),
            "",
        )
    for candidate in CANDIDATES.get(language, ()):
        found = which(candidate.command)
        if found:
            return Launch(language=language, command=found, args=candidate.args), ""
    return None, missing_message(language)


# -- the pool ----------------------------------------------------------------


@dataclass(slots=True)
class Status:
    """One line about one language, live or not."""

    language: str
    state: str  # live | down | idle
    detail: str = ""
    root: str = ""
    documents: int = 0

    def label(self) -> str:
        where = f" [{self.root}]" if self.root else ""
        return f"{self.language}: {self.state}{where}" + (f" — {self.detail}" if self.detail else "")


LIVE: Final = "live"
DOWN: Final = "down"
IDLE: Final = "idle"


class Servers:
    """Every language server this session owns, one per `(language, root)`.

    Starting is serialised by one lock.  Two languages cannot come up at once,
    which costs a few seconds on the second tool call of a mixed repository and
    buys the guarantee that never matters until it does: two threads asking for
    the same language cannot both spawn a server, leaving one of them
    unreferenced and unkillable.
    """

    __slots__ = (
        "_clients",
        "_config",
        "_lock",
        "_reasons",
        "_which",
        "diagnostic_wait",
        "emit",
        "timeout",
        "workspace",
    )

    def __init__(
        self,
        *,
        workspace: Path | str | None = None,
        config: Config | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        diagnostic_wait: float = DIAGNOSTIC_WAIT,
        emit: Callable[[str], None] = lambda _line: None,
        which: Which = shutil.which,
    ) -> None:
        self.workspace = Path(workspace) if workspace is not None else None
        self.timeout = timeout
        self.diagnostic_wait = diagnostic_wait
        self.emit = emit
        self._which = which
        self._config = config
        self._clients: dict[tuple[str, str], LSPClient] = {}
        self._reasons: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()

    # -- configuration ------------------------------------------------------

    @property
    def config(self) -> Config:
        """Read on first use, not at construction: `OFFSET_HOME` moves."""
        if self._config is None:
            self._config = load_config(self.workspace)
        return self._config

    def reload(self) -> Config:
        """Re-read `lsp.json`.  Running servers are untouched until restarted."""
        with self._lock:
            self._config = load_config(self.workspace)
        return self._config

    def language_for(self, path: Path | str) -> str:
        return language_for(path)

    def choose(self, language: str) -> tuple[Launch | None, str]:
        return choose(language, config=self.config, which=self._which)

    # -- clients ------------------------------------------------------------

    @staticmethod
    def _key(language: str, root: Path | str) -> tuple[str, str]:
        return (language.strip().lower(), str(Path(root).resolve()))

    def existing(self, language: str, root: Path | str) -> LSPClient | None:
        """The live client for this pair, without starting one."""
        key = self._key(language, root)
        with self._lock:
            client = self._clients.get(key)
            if client is not None and not client.alive:
                self._clients.pop(key, None)
                self._reasons[key] = client.dead_reason or client.diagnose()
                return None
            return client

    def reason(self, language: str, root: Path | str) -> str:
        return self._reasons.get(self._key(language, root), "")

    def client(self, language: str, root: Path | str) -> tuple[LSPClient | None, str]:
        """The server for this pair, started if need be.  Never raises.

        Returns `(client, "")` or `(None, why not)`.
        """
        language = (language or "").strip().lower()
        key = self._key(language, root)
        with self._lock:
            live = self.existing(language, root)
            if live is not None:
                return live, ""
            launch, problem = self.choose(language)
            if launch is None:
                self._reasons[key] = problem
                return None, problem
            client = LSPClient(
                launch.command,
                launch.args,
                root=Path(root),
                name=f"{language}/{Path(launch.command).name}",
                env=launch.env,
                settings=launch.settings,
                timeout=min(self.timeout, launch.timeout),
                diagnostic_wait=self.diagnostic_wait,
            )
            try:
                client.connect()
            except LSPError as exc:
                client.close()  # never leave a half-started child behind
                why = f"{launch.target} failed to start: {exc}"
                self._reasons[key] = why
                self.emit(f"lsp {language}: {why}")
                return None, why
            self._clients[key] = client
            self._reasons[key] = client.label
            return client, ""

    def for_file(self, path: Path | str, root: Path | str) -> tuple[LSPClient | None, str]:
        """The server for a file, chosen by its extension."""
        language = language_for(path)
        if not language:
            return None, missing_message(Path(path).suffix.lstrip(".").lower())
        return self.client(language, root)

    # -- shutdown -----------------------------------------------------------

    def shutdown(self, language: str, root: Path | str) -> bool:
        key = self._key(language, root)
        with self._lock:
            client = self._clients.pop(key, None)
            self._reasons.pop(key, None)
        if client is None:
            return False
        client.shutdown()
        return True

    def shutdown_all(self) -> None:
        """Stop every server and reap every process group.  Safe to call twice."""
        with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
            self._reasons.clear()
        for client in clients:
            try:
                client.shutdown()
            except Exception:  # a shutdown path may not raise, ever
                client.close()

    # -- reporting ----------------------------------------------------------

    def status(self) -> list[Status]:
        """Every language offset can serve here, live first, then the rest."""
        out: list[Status] = []
        with self._lock:
            live = dict(self._clients)
            reasons = dict(self._reasons)
        seen: set[str] = set()
        for (language, root), client in sorted(live.items()):
            seen.add(language)
            out.append(
                Status(
                    language=language,
                    state=LIVE if client.alive else DOWN,
                    detail=client.label if client.alive else client.diagnose(),
                    root=root,
                    documents=len(client.documents),
                )
            )
        for language in languages():
            if language in seen:
                continue
            launch, problem = self.choose(language)
            if launch is not None:
                out.append(Status(language=language, state=IDLE, detail=f"{launch.target} ({launch.source})"))
            else:
                down = next((why for (lang, _), why in reasons.items() if lang == language), "")
                out.append(Status(language=language, state=DOWN, detail=down or problem))
        return out

    def report(self) -> list[str]:
        lines = [status.label() for status in self.status()]
        config = self.config
        lines.extend(f"config {path}" for path in config.sources)
        lines.extend(f"config error: {problem}" for problem in config.errors)
        return lines
