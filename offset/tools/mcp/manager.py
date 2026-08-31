"""MCP configuration, server lifecycle, and the Tool wrapper.

The rule that shapes this whole module: a broken MCP server must degrade to
"its tools are unavailable, here is why" and never to a call that hangs or a
startup that blocks the shell.  So configuration problems are values that name
the offending field, a connect failure is retried a bounded number of times and
then recorded, and a server that dies has its tools withdrawn.

Config lives in `<workspace>/.offset/mcp.json` then `~/.offset/mcp.json`; the
workspace wins for a server defined in both, because the project is the more
specific statement of intent.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping

from offset.tools.base import Cancelled, Danger, Tool, ToolContext, ToolResult, Toolbox
from offset.tools.mcp.client import (
    CallOutcome,
    MCPCancelled,
    MCPClient,
    MCPError,
    MCPTimeout,
    Prompt,
    RemoteTool,
    Resource,
    ServerGone,
)
from offset.tools.mcp.transport import HTTPTransport, StdioTransport, Transport, TransportError

CONFIG_NAME: Final = "mcp.json"

#: A server name becomes part of a tool name, which models must be able to
#: reproduce exactly, so the character set is restricted rather than escaped.
NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: Both spellings are in the wild; accepting one only would be a papercut.
SERVER_KEYS: Final = ("mcpServers", "servers")

DEFAULT_TIMEOUT: Final = 60.0

#: `${VAR}` and `${env:VAR}`.  Anything else keeps its braces, so a header
#: value that merely looks like a template is passed through untouched.
VAR_RE: Final = re.compile(r"\$\{(?:env:)?([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(slots=True)
class ServerConfig:
    """One configured server.  Exactly one of `command` or `url` is set."""

    name: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    enabled: bool = True
    source: Path | None = None

    @property
    def kind(self) -> str:
        return "stdio" if self.command else "http"

    def target(self) -> str:
        if self.command:
            return " ".join([self.command, *self.args])
        return self.url or ""

    def transport(self) -> Transport:
        if self.command:
            return StdioTransport(self.command, self.args, env=self.env, cwd=self.cwd)
        return HTTPTransport(self.url or "", headers=self.headers, timeout=self.timeout)


@dataclass(slots=True)
class Config:
    """Merged configuration plus every problem found, none of them fatal."""

    servers: list[ServerConfig] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sources: list[Path] = field(default_factory=list)

    def enabled(self) -> list[ServerConfig]:
        return [s for s in self.servers if s.enabled]


def expand(text: str, *, environ: Mapping[str, str] | None = None) -> tuple[str, list[str]]:
    """Substitute `${VAR}` and `${env:VAR}` from the environment.

    Returns the expanded text and the names that were not set.  A secret does
    not belong in a config file that gets committed, so the file names the
    variable and the process supplies it; an unset variable is a configuration
    error rather than a literal `${TOKEN}` handed to a server, because that
    reaches the server looking like a credential and fails far from its cause.
    """
    env = os.environ if environ is None else environ
    missing: list[str] = []

    def sub(match: re.Match[str]) -> str:
        name = match.group(1)
        value = env.get(name)
        if value is None:
            missing.append(name)
            return ""
        return value

    return VAR_RE.sub(sub, text), missing


def _interpolate(value: str, field_: str, problems: list[str]) -> str | None:
    """`None` means a variable was missing: the caller must not carry on with a
    half-substituted string, so the server it belongs to is skipped."""
    text, missing = expand(value)
    if missing:
        for name in dict.fromkeys(missing):
            problems.append(f"{field_}: ${{{name}}} is not set in the environment")
        return None
    return text


def config_paths(workspace: Path | str | None = None) -> list[Path]:
    """Lowest precedence first, matching how `load_config` merges them."""
    home = Path(os.environ.get("OFFSET_HOME") or (Path.home() / ".offset"))
    paths = [home / CONFIG_NAME]
    if workspace:
        paths.append(Path(workspace) / ".offset" / CONFIG_NAME)
    return paths


def load_config(workspace: Path | str | None = None, *, paths: Iterable[Path] | None = None) -> Config:
    """Read and validate every config file, later files overriding earlier."""
    merged: dict[str, ServerConfig] = {}
    out = Config()
    for path in (paths if paths is not None else config_paths(workspace)):
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
        servers, errors = parse_config(raw, source=path)
        out.errors.extend(errors)
        for server in servers:
            merged[server.name] = server
    out.servers = sorted(merged.values(), key=lambda s: s.name)
    return out


def parse_config(raw: Any, *, source: Path | None = None) -> tuple[list[ServerConfig], list[str]]:
    """Validate one config document.  Errors name the offending field."""
    where = f"{source}: " if source else ""
    if not isinstance(raw, dict):
        return [], [f"{where}top level must be a JSON object"]
    table = next((raw[key] for key in SERVER_KEYS if key in raw), None)
    if table is None:
        return [], [f'{where}expected a "mcpServers" object mapping names to servers']
    if not isinstance(table, dict):
        return [], [f'{where}"mcpServers" must be an object mapping names to servers']

    servers: list[ServerConfig] = []
    errors: list[str] = []
    for name, entry in table.items():
        problems: list[str] = []
        field_ = f"mcpServers.{name}"
        if not NAME_RE.match(str(name)):
            errors.append(
                f"{where}{field_}: a server name may only contain letters, digits, "
                "dot, dash and underscore"
            )
            continue
        if not isinstance(entry, dict):
            errors.append(f"{where}{field_} must be an object")
            continue

        problems_before = len(problems)
        command, args = _command(entry, field_, problems)
        url = _url(entry, field_, problems)
        if command and url:
            problems.append(f'{field_}: set either "command" (stdio) or "url" (http), not both')
        elif not command and not url and len(problems) == problems_before:
            # A field that failed to interpolate has already been named; also
            # claiming the server has no command would bury the real cause.
            problems.append(f'{field_}: needs "command" (stdio) or "url" (http)')

        env = _strings(entry.get("env"), f"{field_}.env", problems)
        headers = _strings(entry.get("headers"), f"{field_}.headers", problems)
        timeout = _timeout(entry.get("timeout"), f"{field_}.timeout", problems)
        enabled = _flag(entry, field_, problems)
        cwd = entry.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            problems.append(f"{field_}.cwd must be a string")
            cwd = None
        elif isinstance(cwd, str):
            cwd = _interpolate(cwd, f"{field_}.cwd", problems)

        if problems:
            errors.extend(f"{where}{p}" for p in problems)
            continue
        servers.append(ServerConfig(
            name=str(name),
            command=command,
            args=args,
            env=env,
            url=url,
            headers=headers,
            cwd=cwd,
            timeout=timeout,
            enabled=enabled,
            source=source,
        ))
    return servers, errors


def _command(entry: dict[str, Any], field_: str, problems: list[str]) -> tuple[str | None, list[str]]:
    command = entry.get("command")
    args = entry.get("args", [])
    if command is None:
        parsed_args: list[str] = []
    elif isinstance(command, str):
        parsed_args = []
    elif isinstance(command, list) and command and all(isinstance(c, str) for c in command):
        command, parsed_args = command[0], list(command[1:])
    else:
        problems.append(f"{field_}.command must be a string, or a list of strings")
        return None, []
    if args in (None, []):
        extra: list[str] = []
    elif isinstance(args, list) and all(isinstance(a, str) for a in args):
        extra = list(args)
    else:
        problems.append(f"{field_}.args must be a list of strings")
        extra = []
    if command is not None:
        command = _interpolate(str(command), f"{field_}.command", problems)
        if command is None:
            return None, []
    whole: list[str] = []
    for index, arg in enumerate(parsed_args + extra):
        one = _interpolate(arg, f"{field_}.args[{index}]", problems)
        if one is None:
            return None, []
        whole.append(one)
    return command, whole


def _url(entry: dict[str, Any], field_: str, problems: list[str]) -> str | None:
    url = entry.get("url")
    if url is None:
        return None
    if not isinstance(url, str):
        problems.append(f"{field_}.url must be a string")
        return None
    url = _interpolate(url, f"{field_}.url", problems)
    if url is None:
        return None
    if not url.startswith(("http://", "https://")):
        problems.append(f"{field_}.url must start with http:// or https://")
        return None
    return url


def _strings(value: Any, field_: str, problems: list[str]) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
        problems.append(f"{field_} must be an object with string values")
        return {}
    out: dict[str, str] = {}
    for key, item in value.items():
        one = _interpolate(item, f"{field_}.{key}", problems)
        if one is not None:
            out[str(key)] = one
    return out


def _timeout(value: Any, field_: str, problems: list[str]) -> float:
    if value is None:
        return DEFAULT_TIMEOUT
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        problems.append(f"{field_} must be a positive number of seconds")
        return DEFAULT_TIMEOUT
    return float(value)


def _flag(entry: dict[str, Any], field_: str, problems: list[str]) -> bool:
    for key in ("enabled", "enable"):
        if key in entry:
            if not isinstance(entry[key], bool):
                problems.append(f"{field_}.{key} must be true or false")
                return True
            return bool(entry[key])
    if entry.get("disabled") is not None:  # the other common spelling
        if not isinstance(entry["disabled"], bool):
            problems.append(f"{field_}.disabled must be true or false")
            return True
        return not entry["disabled"]
    return True


# -- the tool wrapper -------------------------------------------------------


def tool_name(server: str, remote: str) -> str:
    """`mcp__<server>__<tool>`, with characters a model cannot reproduce removed."""
    safe = re.sub(r"[^A-Za-z0-9_]", "_", remote)
    return f"mcp__{re.sub(r'[^A-Za-z0-9_]', '_', server)}__{safe}"


class MCPTool(Tool):
    """A remote MCP tool wearing offset's Tool contract.

    Danger is DESTRUCTIVE by default on purpose: the body of a remote tool is
    opaque code on someone else's machine, so the only honest assumption is the
    worst one.  A server that declares `readOnlyHint` earns SAFE, since that is
    the server making a claim we can attribute to it.
    """

    def __init__(self, manager: "Manager", server: str, remote: RemoteTool, *, timeout: float | None = None) -> None:
        self.manager = manager
        self.server = server
        self.remote = remote.name
        self.read_only = remote.read_only
        self.timeout = timeout
        self.name = tool_name(server, remote.name)
        self.description = (remote.description or remote.title or f"remote tool {remote.name}").strip()
        self.schema = remote.schema
        self.danger = Danger.SAFE if remote.read_only else Danger.DESTRUCTIVE
        #: A read-only remote call is safe to overlap; anything that may mutate
        #: state we cannot see is serialised, because we cannot know what it
        #: shares with its siblings.
        self.parallel_safe = remote.read_only

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        ctx.check()
        client = self.manager.client(self.server)
        if client is None or not client.alive:
            why = self.manager.reason(self.server) or "not connected"
            return ToolResult.fail(f"mcp server {self.server!r} is unavailable: {why}")
        budget = min(self.timeout or ctx.timeout, ctx.timeout)
        started = time.monotonic()
        try:
            outcome = client.call_tool(self.remote, args, timeout=budget, stop=ctx.cancel.is_set)
        except MCPCancelled as exc:
            raise Cancelled(f"{self.name} cancelled") from exc
        except MCPTimeout:
            return ToolResult.fail(
                f"{self.name} got no answer from mcp server {self.server!r} within {budget:g}s"
            )
        except ServerGone as exc:
            self.manager.mark_dead(self.server, str(exc))
            return ToolResult.fail(f"mcp server {self.server!r} died during the call: {exc}")
        except MCPError as exc:
            return ToolResult.fail(f"{self.name}: {exc}")
        return self._result(outcome, time.monotonic() - started)

    def _result(self, outcome: CallOutcome, elapsed: float) -> ToolResult:
        if not outcome.ok:
            return ToolResult.fail(
                outcome.content or f"{self.name} reported failure without saying why",
                display=f"{self.name} failed",
            )
        content = outcome.content
        if not content and outcome.structured:
            content = json.dumps(outcome.structured, indent=1)
        return ToolResult(
            ok=True,
            content=content or f"{self.remote} returned no content",
            display=f"{self.name} -> {elapsed:.2f}s",
            data={"server": self.server, "tool": self.remote, "structured": outcome.structured},
        )

    def preview(self, args: dict[str, Any]) -> str:
        shown = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
        return f"{self.server}:{self.remote}({shown})"


# -- lifecycle --------------------------------------------------------------

#: A server's state, in the words the UI shows.
IDLE: Final = "idle"
LIVE: Final = "live"
DOWN: Final = "down"
OFF: Final = "disabled"


@dataclass(slots=True)
class Status:
    name: str
    state: str
    detail: str = ""
    tools: int = 0
    kind: str = "stdio"


@dataclass(slots=True)
class Offering:
    """What one server publishes besides its tools.

    Resources and prompts are listed lazily rather than cached at connect: a
    server is free to change them at any time, and a stale list shown to the
    user is worse than a round trip.
    """

    server: str
    resources: list[Resource] = field(default_factory=list)
    prompts: list[Prompt] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def report(self) -> list[str]:
        if self.error:
            return [f"{self.server}: {self.error}"]
        lines = [f"{self.server}: {len(self.resources)} resource(s), {len(self.prompts)} prompt(s)"]
        lines.extend(f"  resource {r.uri}{f' — {r.name}' if r.name else ''}" for r in self.resources)
        lines.extend(f"  prompt {p.name}{f' — {p.description}' if p.description else ''}" for p in self.prompts)
        return lines


@dataclass(slots=True)
class ResourceRead:
    """The body of one resource, or the reason there is none."""

    server: str
    uri: str
    text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class Manager:
    """Owns every MCP connection: connect, retry, withdraw, disconnect.

    Nothing here blocks on a server that is not answering: connect has a
    deadline per attempt and a bounded number of attempts, and a server that
    dies is marked down so its tools stop being offered instead of hanging the
    next call.
    """

    __slots__ = (
        "_clients",
        "_collisions",
        "_lock",
        "_reasons",
        "_registered",
        "_sleep",
        "_state",
        "_tools",
        "_toolbox",
        "attempts",
        "backoff",
        "config",
        "emit",
        "max_backoff",
        "workspace",
    )

    def __init__(
        self,
        config: Config | None = None,
        *,
        attempts: int = 3,
        backoff: float = 0.25,
        max_backoff: float = 4.0,
        sleep: Callable[[float], None] = time.sleep,
        emit: Callable[[str], None] = lambda _line: None,
        workspace: Path | str | None = None,
        toolbox: Toolbox | None = None,
    ) -> None:
        self.config = config or Config()
        self.attempts = max(1, attempts)
        self.backoff = backoff
        self.max_backoff = max_backoff
        self.emit = emit
        self._clients: dict[str, MCPClient] = {}
        self._state: dict[str, str] = {}
        self._reasons: dict[str, str] = {}
        self._tools: dict[str, list[MCPTool]] = {}
        self._sleep = sleep
        self._lock = threading.Lock()
        #: The names this manager put in the Toolbox, per server.  Withdrawal
        #: goes by recorded name rather than by `mcp__<server>__` prefix,
        #: because two server names can sanitise to the same prefix and a
        #: reload must never withdraw a neighbour's tool.
        self._registered: dict[str, list[str]] = {}
        #: Names a server asked for and could not have.  Kept rather than
        #: logged: `shell/app.py` used to `except ValueError: pass` around
        #: registration, so a collision cost the user half a server's tools
        #: with nothing on screen to say why.
        self._collisions: dict[str, list[str]] = {}
        self._toolbox = toolbox
        self.workspace = Path(workspace) if workspace is not None else None
        for server in self.config.servers:
            self._state[server.name] = IDLE if server.enabled else OFF

    @classmethod
    def from_workspace(cls, workspace: Path | str | None = None, **kwargs: Any) -> "Manager":
        # The workspace is kept, not just used: `reload()` has to re-read the
        # same pair of files, and without it a reload silently read a different
        # config to the one that was loaded at startup.
        kwargs.setdefault("workspace", workspace)
        return cls(load_config(workspace), **kwargs)

    # -- accessors ----------------------------------------------------------

    def config_for(self, name: str) -> ServerConfig | None:
        return next((s for s in self.config.servers if s.name == name), None)

    def client(self, name: str) -> MCPClient | None:
        client = self._clients.get(name)
        if client is not None and not client.alive:
            self.mark_dead(name, client.dead_reason or "server exited")
            return None
        return client

    def reason(self, name: str) -> str:
        return self._reasons.get(name, "")

    def state(self, name: str) -> str:
        return self._state.get(name, IDLE)

    def status(self) -> list[Status]:
        out: list[Status] = []
        for server in self.config.servers:
            out.append(Status(
                name=server.name,
                state=self.state(server.name),
                detail=self._reasons.get(server.name, "") or server.target(),
                tools=len(self._tools.get(server.name, ())),
                kind=server.kind,
            ))
        return out

    def tools(self) -> list[Tool]:
        """Every tool of every live server, as offset Tools."""
        out: list[Tool] = []
        for server in self.config.servers:
            out.extend(self._tools.get(server.name, ()))
        return out

    # -- the toolbox --------------------------------------------------------

    def attach(self, toolbox: Toolbox | None) -> list[str]:
        """Adopt a Toolbox and make it match this manager.  Returns collisions.

        The manager owns the registration rather than the caller, because only
        the manager knows which names came from which listing: a caller that
        registered the tools itself had no record of its own entries, so a
        reload could not withdraw them and stale names stayed callable.
        """
        if self._toolbox is not None and toolbox is not self._toolbox:
            for name in list(self._registered):
                self._withdraw(name)
        self._toolbox = toolbox
        return self.publish()

    @property
    def toolbox(self) -> Toolbox | None:
        return self._toolbox

    def registered(self, name: str) -> list[str]:
        """The tool names this manager currently holds in the Toolbox for `name`."""
        return list(self._registered.get(name, ()))

    def collisions(self) -> list[str]:
        """Every name a server could not claim, because something else holds it."""
        return [line for server in sorted(self._collisions) for line in self._collisions[server]]

    def publish(self, name: str | None = None) -> list[str]:
        """Re-sync the Toolbox for one server, or for all of them."""
        names = [name] if name is not None else [s.name for s in self.config.servers]
        out: list[str] = []
        for one in names:
            out.extend(self._publish(one))
        return out

    def _publish(self, name: str) -> list[str]:
        """Withdraw what this server had, then register what it has now.

        Withdrawal is unconditional and comes first: the previous listing is
        the only thing that can leave a name in the Toolbox resolving to a tool
        the server has stopped serving.
        """
        self._withdraw(name)
        box = self._toolbox
        if box is None:
            return []
        taken: list[str] = []
        clashes: list[str] = []
        for tool in self._tools.get(name, ()):
            if box.get(tool.name) is not None:
                clashes.append(
                    f"mcp {name}: {tool.name} is already taken by another tool, not registered"
                )
                continue
            box.register(tool)
            taken.append(tool.name)
        if taken:
            self._registered[name] = taken
        if clashes:
            self._collisions[name] = clashes
        else:
            self._collisions.pop(name, None)
        return clashes

    def _withdraw(self, name: str) -> list[str]:
        """Take this server's tools back out of the Toolbox.  Idempotent."""
        taken = self._registered.pop(name, [])
        box = self._toolbox
        if box is not None:
            for tool in taken:
                box.unregister(tool)
        return taken

    # -- connecting ---------------------------------------------------------

    def connect(self, name: str) -> bool:
        """Connect one server, retrying with bounded backoff.  Never raises."""
        server = self.config_for(name)
        if server is None:
            self._reasons[name] = f"no server named {name!r} is configured"
            self._state[name] = DOWN
            return False
        if not server.enabled:
            self._state[name] = OFF
            self._reasons[name] = "disabled in mcp.json"
            return False
        if self.state(name) == LIVE and self.client(name) is not None:
            return True

        last = ""
        for attempt in range(self.attempts):
            try:
                client = self._handshake(server)
            except (MCPError, TransportError) as exc:
                last = str(exc)
                self.emit(f"mcp {name}: attempt {attempt + 1} failed: {last}")
                if attempt < self.attempts - 1:
                    self._sleep(min(self.max_backoff, self.backoff * (2 ** attempt)))
                continue
            with self._lock:
                self._clients[name] = client
                self._state[name] = LIVE
                self._reasons[name] = client.description
            self._refresh(name)
            return True

        with self._lock:
            self._state[name] = DOWN
            self._reasons[name] = last or "could not connect"
        return False

    def _handshake(self, server: ServerConfig) -> MCPClient:
        client = MCPClient(
            server.transport(),
            name=server.name,
            timeout=server.timeout,
            on_notification=lambda msg, n=server.name: self._on_notification(n, msg),
        )
        try:
            client.connect()
        except Exception:
            client.close()  # never leave a half-started child behind
            raise
        return client

    def connect_all(self) -> list[str]:
        """Connect every enabled server.  Returns one message per failure."""
        failures: list[str] = []
        for server in self.config.enabled():
            if not self.connect(server.name):
                failures.append(f"mcp {server.name}: {self.reason(server.name)}")
        return failures

    def _refresh(self, name: str) -> None:
        """Re-list a server's tools.  A listing failure withdraws the server."""
        client = self._clients.get(name)
        server = self.config_for(name)
        if client is None or server is None:
            return
        try:
            remote = client.list_tools(timeout=min(server.timeout, 30.0))
        except (MCPError, TransportError) as exc:
            self.mark_dead(name, f"tools/list failed: {exc}")
            return
        with self._lock:
            self._tools[name] = [MCPTool(self, name, tool, timeout=server.timeout) for tool in remote]
        # Outside the lock, and after the listing is in place: a server may
        # announce a changed tool list from the reader thread, and the Toolbox
        # has to follow it or the model keeps seeing the old names.
        self._publish(name)

    def _on_notification(self, name: str, message: dict[str, Any]) -> None:
        if message.get("method") == "notifications/tools/list_changed":
            self._refresh(name)

    # -- failure and shutdown ----------------------------------------------

    def mark_dead(self, name: str, reason: str) -> None:
        """Withdraw a server's tools.  Idempotent, and safe from any thread."""
        with self._lock:
            client = self._clients.pop(name, None)
            self._tools.pop(name, None)
            self._state[name] = DOWN
            self._reasons[name] = reason or "server exited"
        self._withdraw(name)
        if client is not None:
            client.close()
        self.emit(f"mcp {name} is unavailable: {reason}")

    def disconnect(self, name: str) -> None:
        """Stop a server, reap its process tree, and withdraw its tools."""
        with self._lock:
            client = self._clients.pop(name, None)
            self._tools.pop(name, None)
            server = self.config_for(name)
            self._state[name] = OFF if server is not None and not server.enabled else IDLE
            self._reasons.pop(name, None)
        self._withdraw(name)
        self._collisions.pop(name, None)
        if client is not None:
            client.close()

    def disconnect_all(self) -> None:
        for name in list(self._clients):
            self.disconnect(name)

    # -- runtime changes ----------------------------------------------------

    def add_server(self, config: ServerConfig) -> bool:
        """Add, or replace, one server and connect it without a restart.

        Returns whether it came up.  A refusal is recorded in `reason(name)`
        rather than raised: the caller is a slash command whose whole job is to
        print the reason.
        """
        name = config.name
        if not NAME_RE.match(name):
            self._state[name] = DOWN
            self._reasons[name] = (
                "a server name may only contain letters, digits, dot, dash and underscore"
            )
            return False
        if self.config_for(name) is not None:
            self.disconnect(name)
        others = [s for s in self.config.servers if s.name != name]
        self.config.servers = sorted([*others, config], key=lambda s: s.name)
        self._state[name] = IDLE if config.enabled else OFF
        self._reasons.pop(name, None)
        return self.connect(name)

    def remove_server(self, name: str) -> bool:
        """Forget a server: stop it, withdraw its tools, drop its config."""
        if self.config_for(name) is None:
            return False
        self.disconnect(name)  # this is what withdraws from the Toolbox
        self.config.servers = [s for s in self.config.servers if s.name != name]
        for book in (self._state, self._reasons, self._tools, self._collisions):
            book.pop(name, None)
        return True

    def reconnect(self, name: str) -> bool:
        """Drop and re-establish one connection, re-listing its tools.

        The only honest way to pick up a server that was restarted behind our
        back: its old pipe can still be open, so `connect` would short-circuit
        on a client that will never answer again.
        """
        if self.config_for(name) is None:
            self._state[name] = DOWN
            self._reasons[name] = f"no server named {name!r} is configured"
            return False
        self.disconnect(name)
        return self.connect(name)

    def reload(self, name: str | None = None) -> list[str]:
        """Re-read `mcp.json` and make the running set match it.

        A server that vanished from the file is removed, one whose definition
        changed is reconnected, one that is new is connected, and — for a
        whole-file reload — one that is untouched is left alone, because a
        reload that dropped every working connection would be too expensive to
        reach for.  Naming a single server is an explicit request for that one,
        so it is always reconnected.
        """
        fresh = load_config(self.workspace)
        self.config.errors = fresh.errors
        self.config.sources = fresh.sources
        lines = [f"config: {problem}" for problem in fresh.errors]

        wanted = {s.name: s for s in fresh.servers}
        if name is None:
            known = [s.name for s in self.config.servers]
        else:
            known = [name] if self.config_for(name) is not None else []
            wanted = {name: wanted[name]} if name in wanted else {}
            if not wanted and not known:
                lines.append(f"mcp {name}: no server by that name is configured")

        for gone in [n for n in known if n not in wanted]:
            self.remove_server(gone)
            lines.append(f"mcp {gone}: removed, no longer in mcp.json")

        for server in wanted.values():
            before = self.config_for(server.name)
            same = before is not None and _signature(before) == _signature(server)
            if name is None and same and self.state(server.name) == LIVE:
                lines.append(f"mcp {server.name}: unchanged, {len(self._tools.get(server.name, ()))} tool(s)")
                continue
            if self.add_server(server):
                lines.append(f"mcp {server.name}: connected, {len(self._tools.get(server.name, ()))} tool(s)")
            elif not server.enabled:
                lines.append(f"mcp {server.name}: disabled in mcp.json")
            else:
                lines.append(f"mcp {server.name}: {self.reason(server.name)}")
        lines.extend(self.collisions())
        return lines

    # -- resources and prompts ---------------------------------------------

    def offering(self, name: str) -> Offering:
        """List one server's resources and prompts.  Never raises."""
        client = self.client(name)
        if client is None:
            return Offering(name, error=self.reason(name) or "not connected")
        try:
            resources = client.list_resources(timeout=self._budget(name))
            prompts = client.list_prompts(timeout=self._budget(name))
        except ServerGone as exc:
            self.mark_dead(name, str(exc))
            return Offering(name, error=f"server died: {exc}")
        except (MCPError, TransportError) as exc:
            return Offering(name, error=f"{type(exc).__name__}: {exc}")
        return Offering(name, resources=resources, prompts=prompts)

    def offerings(self) -> list[Offering]:
        """Every live server's listing, in configuration order."""
        return [self.offering(s.name) for s in self.config.servers if self.state(s.name) == LIVE]

    def read_resource(self, name: str, uri: str) -> ResourceRead:
        """Fetch one resource body as text.  A failure lands in `.error`."""
        client = self.client(name)
        if client is None:
            return ResourceRead(name, uri, error=self.reason(name) or "not connected")
        try:
            text = client.read_resource(uri, timeout=self._budget(name))
        except ServerGone as exc:
            self.mark_dead(name, str(exc))
            return ResourceRead(name, uri, error=f"server died: {exc}")
        except (MCPError, TransportError) as exc:
            return ResourceRead(name, uri, error=f"{type(exc).__name__}: {exc}")
        if not text:
            return ResourceRead(name, uri, error=f"{uri} carries no readable text")
        return ResourceRead(name, uri, text=text)

    def _budget(self, name: str) -> float:
        server = self.config_for(name)
        # A listing is metadata; spending a whole tool-call timeout on it would
        # freeze the command that asked for it.
        return min(server.timeout if server else DEFAULT_TIMEOUT, 30.0)

    def __len__(self) -> int:
        return len(self._clients)


def _signature(server: ServerConfig) -> tuple[Any, ...]:
    """Everything that changes how a server is reached, and nothing else.

    `source` is deliberately excluded: the same definition moving from the home
    file to the workspace file still describes the same process to talk to, and
    reconnecting it would drop a working session for no reason at all.
    """
    return (
        server.command,
        tuple(server.args),
        tuple(sorted(server.env.items())),
        server.url,
        tuple(sorted(server.headers.items())),
        server.cwd,
        server.timeout,
        server.enabled,
    )
