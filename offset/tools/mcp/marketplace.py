"""A marketplace for MCP servers: find one, read what it needs, record it.

Three decisions shape everything below, and each of them exists because the
obvious alternative is a way to get hurt.

**Installing is not running.**  `install` writes an entry into the user's
`mcp.json` and stops.  It never spawns the command, never resolves it on
`PATH`, never fetches a package.  The whole point of a marketplace is that a
one-word command can bring in code from a stranger, so the moment of first
execution must stay where the user can see it - the next connect, done by
`Manager`, with the server named in `/mcp`.  A marketplace that ran `npx` to
"verify" an install would be executing arbitrary code as a side effect of
browsing.

**Trust is never self-declared.**  A registry document carries a trust level,
and that level is treated as a claim, not a fact: the only servers offset calls
trusted are the ones in `BUILTIN`, which ship with the source you are reading.
A fetched entry is clamped to `COMMUNITY` at best, so a hostile registry cannot
promote itself past the confirmation gate, and for the same reason a fetched
entry never overrides a built-in id - otherwise `git` could be re-pointed at
someone else's command without the user noticing.  Untrusted servers install
only with `confirm=True`.

**The catalogue is cached, refreshed in the background, and never emptied by a
failure.**  This follows `offset.providers.catalogue` exactly, including the bug
that module had to fix: the home directory is resolved on the *calling* thread
and passed into the worker.  A daemon thread that asks `settings.home()` on its
own time answers with whatever `OFFSET_HOME` says by then, which for a shell
that has exited - or a test that has reverted its patch - is the user's real
`~/.offset`.  A refresh begun against a temporary home wrote its cache into the
real one, observably.

A server whose required environment variables are not set is still installed;
it is reported as unconfigured with the missing names listed.  That is the
honest outcome: the config file is the right place for `${GITHUB_TOKEN}` and the
process is the right place for its value, so refusing the install would only
force the user to choose between two correct halves.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Iterator, Mapping

from offset.core import settings
from offset.tools.mcp.manager import CONFIG_NAME, NAME_RE, SERVER_KEYS, expand

CACHE_NAME: Final = "mcp-registry.json"

#: Where user-added registries are remembered.  Kept apart from the catalogue
#: cache so that clearing a stale cache never loses the list of sources.
SOURCES_NAME: Final = "mcp-sources.json"

CACHE_VERSION: Final = 1

#: A registry of servers changes far more slowly than a model listing: new
#: servers appear over weeks, not hours.  A day is long enough to cost nothing
#: and short enough that a user who heard about a server yesterday finds it.
TTL: Final = 24 * 3600.0

#: A failure is retried sooner than a success, because the usual cause is a
#: machine that was briefly off the network rather than a registry that stopped
#: existing.
RETRY_TTL: Final = 900.0

TIMEOUT: Final = 8.0

#: Suppresses every network fetch, for an air-gapped machine or a test.  The
#: built-in list still works, so the marketplace is never simply dead.
NO_FETCH_ENV: Final = "OFFSET_NO_MCP_FETCH"

#: Trust levels, strongest first.  `TRUSTED` is reserved for `BUILTIN`; nothing
#: fetched can reach it, no matter what its document says.
TRUSTED: Final = "trusted"
COMMUNITY: Final = "community"
UNKNOWN: Final = "unknown"

#: What a fetched document is allowed to claim.  Anything else it says about
#: itself is downgraded to `UNKNOWN` rather than rejected: an unfamiliar word is
#: a reason for caution, not a reason to hide the entry.
FETCHED_LEVELS: Final = (COMMUNITY, UNKNOWN)

#: The official public registry.  Its entries arrive as `COMMUNITY` at best and
#: so still face the confirmation gate; it is a default source because a
#: marketplace that only ever shows a hard-coded list is a menu, not a market.
DEFAULT_SOURCES: Final[tuple[str, ...]] = (
    "https://registry.modelcontextprotocol.io/v0/servers",
)

#: Characters a server name may not contain, per `manager.NAME_RE`.  A registry
#: id like `io.github.owner/thing` is perfectly valid as an id and unusable as a
#: config key, so it is slugged rather than refused.
UNSAFE_NAME: Final = re.compile(r"[^A-Za-z0-9_.-]+")

#: Fetches a URL and returns parsed JSON.  Injected so tests never touch a
#: network and so a caller can supply its own transport.
Fetcher = Callable[[str], Any]


def _get(url: str) -> Any:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "offset"}, method="GET"
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


# -- what a server is --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Server:
    """One offer in the catalogue: enough to decide, and enough to install.

    `env` is the *names* of the variables the server needs, never their values.
    A marketplace entry that carried a secret would be a secret in a cache file.
    """

    id: str
    name: str = ""
    description: str = ""
    kind: str = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    url: str = ""
    env: tuple[str, ...] = ()
    homepage: str = ""
    trust: str = UNKNOWN
    source: str = "builtin"

    @property
    def trusted(self) -> bool:
        return self.trust == TRUSTED

    @property
    def label(self) -> str:
        return self.name or self.id

    @property
    def slug(self) -> str:
        """The name this server takes in `mcp.json`."""
        return UNSAFE_NAME.sub("-", self.id).strip("-.")

    @property
    def installable(self) -> bool:
        """False for an entry offset could not write out or could not reach.

        Both halves matter: a document may describe a server with no command
        and no url at all, and an id may slug down to nothing.
        """
        if not NAME_RE.match(self.slug):
            return False
        return bool(self.command) if self.kind == "stdio" else bool(self.url)

    def target(self) -> str:
        if self.kind == "stdio":
            return " ".join([self.command, *self.args]).strip()
        return self.url

    def missing(self, environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """Required variables that are not set, in the order declared."""
        env = os.environ if environ is None else environ
        return tuple(n for n in self.env if not (env.get(n) or "").strip())

    def entry(self) -> dict[str, Any]:
        """The `mcp.json` fragment for this server.

        Secrets are written as `${VAR}` templates, which `manager.expand`
        resolves from the process at connect time.  A value baked into the file
        would end up committed the first time the workspace config is used.
        """
        if self.kind == "stdio":
            body: dict[str, Any] = {"command": self.command}
            if self.args:
                body["args"] = list(self.args)
        else:
            body = {"url": self.url}
        if self.env:
            body["env"] = {n: f"${{{n}}}" for n in self.env}
        return body

    def line(self) -> str:
        """One listing row.  The trust level is shown wherever a server is."""
        return f"{self.id:<28} {self.trust:<9} {self.kind:<5} {(self.description or self.label)[:44]}"

    def detail(self, environ: Mapping[str, str] | None = None) -> list[str]:
        lines = [
            f"{self.label}  ({self.id})",
            f"trust:     {self.trust}" + ("" if self.trusted else "  - needs explicit confirmation"),
            f"transport: {self.kind}",
            f"runs:      {self.target() or '(nothing declared)'}",
        ]
        if self.description:
            lines.insert(1, self.description)
        if self.env:
            gone = self.missing(environ)
            state = ("missing: " + ", ".join(gone)) if gone else "all set"
            lines.append(f"env:       {', '.join(self.env)}  [{state}]")
        if self.homepage:
            lines.append(f"home:      {self.homepage}")
        lines.append(f"source:    {self.source}")
        if not self.installable:
            lines.append("this entry cannot be installed: no usable command or url")
        return lines

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "transport": self.kind,
            "command": self.command,
            "args": list(self.args),
            "url": self.url,
            "env": list(self.env),
            "homepage": self.homepage,
            "trust": self.trust,
            "source": self.source,
        }


def _strs(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Mapping):
        # `{"GITHUB_TOKEN": "a personal access token"}` documents the variable;
        # only its name is ours to keep.
        return tuple(str(k) for k in value if str(k).strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value if str(v).strip())
    return ()


def _level(raw: Any, *, fetched: bool) -> str:
    claim = str(raw or "").strip().lower()
    if not fetched:
        return TRUSTED
    return claim if claim in FETCHED_LEVELS else UNKNOWN


def _from_packages(entry: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """The official registry describes *how to obtain* a server, not a command.

    An npm package becomes `npx -y <name>`, a PyPI one `uvx <name>`, which is
    what the servers' own documentation tells people to type.  Nothing is run
    here; this only writes the command down.
    """
    packages = entry.get("packages")
    if not isinstance(packages, (list, tuple)):
        return "", ()
    for package in packages:
        if not isinstance(package, Mapping):
            continue
        registry = str(package.get("registry_name") or package.get("registry") or "").lower()
        name = str(package.get("name") or package.get("identifier") or "").strip()
        if not name:
            continue
        if registry in ("npm", "npmjs"):
            return "npx", ("-y", name)
        if registry in ("pypi", "python"):
            return "uvx", (name,)
    return "", ()


def _from_remotes(entry: Mapping[str, Any]) -> str:
    remotes = entry.get("remotes")
    if not isinstance(remotes, (list, tuple)):
        return ""
    for remote in remotes:
        if isinstance(remote, Mapping) and str(remote.get("url") or "").startswith("http"):
            return str(remote["url"])
    return ""


def parse_server(raw: Any, *, source: str = "builtin", fetched: bool = True) -> Server | None:
    """One catalogue entry, or None if it is not one.

    Lenient on shape and strict on trust: unfamiliar keys are ignored, but a
    fetched document never gets to name its own trust level.
    """
    if not isinstance(raw, Mapping):
        return None
    ident = str(raw.get("id") or raw.get("name") or "").strip()
    if not ident:
        return None

    kind = str(raw.get("transport") or raw.get("kind") or "").strip().lower()
    command = str(raw.get("command") or "").strip()
    args = _strs(raw.get("args"))
    url = str(raw.get("url") or "").strip()
    if not command and not url:
        command, args = _from_packages(raw)
        if not command:
            url = _from_remotes(raw)
    if kind not in ("stdio", "http"):
        kind = "http" if (url and not command) else "stdio"

    return Server(
        id=ident,
        name=str(raw.get("name") or ident),
        description=" ".join(str(raw.get("description") or "").split()),
        kind=kind,
        command=command,
        args=args,
        url=url,
        env=_strs(raw.get("env") or raw.get("environment") or raw.get("required_env")),
        homepage=str(raw.get("homepage") or raw.get("repository") or raw.get("website") or "").strip(),
        trust=_level(raw.get("trust"), fetched=fetched),
        source=source,
    )


def parse_registry(raw: Any, *, source: str, fetched: bool = True) -> tuple[list[Server], list[str]]:
    """Every server in one document, plus the reasons any were dropped."""
    body: Any = raw
    if isinstance(raw, Mapping):
        body = next((raw[k] for k in ("servers", "mcpServers", "data", "results") if k in raw), None)
    if isinstance(body, Mapping):
        # `{"id": {...}}` is as common as a list, and the key is the id.
        body = [{"id": k, **v} if isinstance(v, Mapping) else v for k, v in body.items()]
    if not isinstance(body, (list, tuple)):
        return [], [f"{source}: expected a list of servers"]

    found: list[Server] = []
    problems: list[str] = []
    for item in body:
        server = parse_server(item, source=source, fetched=fetched)
        if server is None:
            problems.append(f"{source}: an entry has no id")
            continue
        if not server.installable:
            problems.append(f"{source}: {server.id} declares no usable command or url")
            continue
        found.append(server)
    return found, problems


# -- what ships with offset --------------------------------------------------

#: The trusted set.  Every one of these is a first-party server from the
#: Model Context Protocol project or its documented reference list, and the
#: commands are the ones those projects publish.  Being in this tuple is the
#: only way a server counts as trusted, which is why the list is short.
BUILTIN: Final[tuple[Server, ...]] = tuple(
    Server(trust=TRUSTED, source="builtin", **kw)  # type: ignore[arg-type]
    for kw in (
        dict(id="filesystem", name="Filesystem",
             description="Read and write files under directories you name.",
             command="npx", args=("-y", "@modelcontextprotocol/server-filesystem", "."),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="git", name="Git",
             description="Read, search and inspect a git repository.",
             command="uvx", args=("mcp-server-git",),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="fetch", name="Fetch",
             description="Fetch a URL and convert it to markdown for the model.",
             command="uvx", args=("mcp-server-fetch",),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="memory", name="Memory",
             description="A knowledge graph the model can write notes into.",
             command="npx", args=("-y", "@modelcontextprotocol/server-memory"),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="sequential-thinking", name="Sequential Thinking",
             description="Step-by-step reasoning scratchpad as a tool.",
             command="npx", args=("-y", "@modelcontextprotocol/server-sequential-thinking"),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="time", name="Time",
             description="Current time and timezone conversion.",
             command="uvx", args=("mcp-server-time",),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="sqlite", name="SQLite",
             description="Query a local SQLite database.",
             command="uvx", args=("mcp-server-sqlite", "--db-path", "./data.db"),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="everything", name="Everything",
             description="The reference server: every protocol feature, for testing.",
             command="npx", args=("-y", "@modelcontextprotocol/server-everything"),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="github", name="GitHub",
             description="Issues, pull requests and code search on GitHub.",
             command="npx", args=("-y", "@modelcontextprotocol/server-github"),
             env=("GITHUB_PERSONAL_ACCESS_TOKEN",),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="slack", name="Slack",
             description="Read and post to Slack channels.",
             command="npx", args=("-y", "@modelcontextprotocol/server-slack"),
             env=("SLACK_BOT_TOKEN", "SLACK_TEAM_ID"),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="puppeteer", name="Puppeteer",
             description="Drive a headless browser: navigate, click, screenshot.",
             command="npx", args=("-y", "@modelcontextprotocol/server-puppeteer"),
             homepage="https://github.com/modelcontextprotocol/servers"),
        dict(id="brave-search", name="Brave Search",
             description="Web and local search through the Brave API.",
             command="npx", args=("-y", "@modelcontextprotocol/server-brave-search"),
             env=("BRAVE_API_KEY",),
             homepage="https://github.com/modelcontextprotocol/servers"),
    )
)


# -- the merged catalogue ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class Registry:
    """The catalogue as a value: what is on offer, when, and what went wrong."""

    servers: tuple[Server, ...] = ()
    fetched: float = 0.0
    errors: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.servers)

    def __iter__(self) -> Iterator[Server]:
        return iter(self.servers)

    def get(self, server_id: str) -> Server | None:
        """By exact id first, then by slug, then by unique prefix.

        The prefix rule is deliberately unambiguous-only: guessing between two
        candidates would mean installing something the user did not name.
        """
        wanted = (server_id or "").strip().lower()
        if not wanted:
            return None
        for server in self.servers:
            if server.id.lower() == wanted or server.slug.lower() == wanted:
                return server
        hits = [s for s in self.servers if s.id.lower().startswith(wanted)]
        if len(hits) == 1:
            return hits[0]
        hits = [s for s in self.servers if wanted in s.id.lower()]
        return hits[0] if len(hits) == 1 else None

    def search(self, query: str) -> list[Server]:
        """Match on id, name and description, best first.

        Description text counts: a user hunting for "browser" should find
        Puppeteer, whose id says nothing about browsers.
        """
        words = [w for w in re.split(r"[\s,]+", (query or "").lower()) if w]
        if not words:
            return sorted(self.servers, key=_rank_key)
        scored: list[tuple[float, Server]] = []
        for server in self.servers:
            ident = f"{server.id} {server.slug}".lower()
            name = server.name.lower()
            text = server.description.lower()
            score = 0.0
            for word in words:
                if word == server.id.lower() or word == server.slug.lower():
                    score += 8.0
                elif word in ident:
                    score += 4.0
                if word in name:
                    score += 2.0
                if word in text:
                    score += 1.0
            if score:
                scored.append((score, server))
        scored.sort(key=lambda pair: (-pair[0], _rank_key(pair[1])))
        return [server for _, server in scored]


def _rank_key(server: Server) -> tuple[int, str]:
    """Trusted first, then alphabetical: the safe answer is the visible one."""
    return (0 if server.trusted else 1, server.id.lower())


# -- sources -----------------------------------------------------------------


def sources_file(home: Path | None = None) -> Path:
    return (home if home is not None else settings.home()) / SOURCES_NAME


def sources(home: Path | None = None) -> list[str]:
    """Every registry to consult: the defaults plus whatever the user added."""
    out = list(DEFAULT_SOURCES)
    try:
        raw = json.loads(sources_file(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    added = raw.get("sources") if isinstance(raw, Mapping) else raw
    if isinstance(added, (list, tuple)):
        for item in added:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
    return out


def add_source(where: str, home: Path | None = None) -> bool:
    """Remember one more registry.  False if it was already known."""
    text = (where or "").strip()
    if not text or text in sources(home):
        return False
    path = sources_file(home)
    current = [s for s in sources(home) if s not in DEFAULT_SOURCES]
    _write_json(path, {"sources": [*current, text]})
    return True


def forget_source(where: str, home: Path | None = None) -> bool:
    """Drop a user-added registry.  A default cannot be forgotten this way."""
    text = (where or "").strip()
    current = [s for s in sources(home) if s not in DEFAULT_SOURCES]
    if text not in current:
        return False
    _write_json(sources_file(home), {"sources": [s for s in current if s != text]})
    return True


# -- the cache ---------------------------------------------------------------


def cache_file(home: Path | None = None) -> Path:
    """Where the catalogue cache lives.

    `home` is explicit for the background refresh, which must write to the
    directory that was current when it *started* - see the module docstring.
    """
    return (home if home is not None else settings.home()) / CACHE_NAME


def _write_json(path: Path, body: Any) -> None:
    """Atomic, because a half-written cache read on the next launch is a crash
    at startup rather than a stale list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass  # the temp file is already gone, or the directory is; either
            # way the original is untouched and the real error is re-raised
        raise


def _read_cache(home: Path | None = None) -> dict[str, Any]:
    try:
        raw = json.loads(cache_file(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return {}
    return raw


@dataclass(frozen=True, slots=True)
class Cached:
    """One source's last known contents, fresh or not."""

    source: str
    servers: tuple[Server, ...] = ()
    error: str = ""
    fetched: float = 0.0


def cached(source: str, home: Path | None = None) -> Cached | None:
    entry = (_read_cache(home).get("sources") or {}).get(source)
    if not isinstance(entry, dict):
        return None
    servers = [parse_server(s, source=source, fetched=True) for s in entry.get("servers") or []]
    return Cached(
        source,
        tuple(s for s in servers if s is not None and s.installable),
        str(entry.get("error") or ""),
        float(entry.get("fetched") or 0.0),
    )


def store(entry: Cached, home: Path | None = None) -> None:
    body = _read_cache(home) or {}
    body["version"] = CACHE_VERSION
    table = body.setdefault("sources", {})
    if not isinstance(table, dict):
        table = body["sources"] = {}
    table[entry.source] = {
        "fetched": entry.fetched,
        "error": entry.error,
        "servers": [s.to_json() for s in entry.servers],
    }
    _write_json(cache_file(home), body)


def stale(entry: Cached | None, *, now: float | None = None) -> bool:
    if entry is None:
        return True
    age = (time.time() if now is None else now) - entry.fetched
    return age > (RETRY_TTL if entry.error else TTL)


def catalogue(home: Path | None = None) -> Registry:
    """The built-in list with every cached source merged over it.

    Built-in ids win.  A fetched registry that also called itself `git` would
    otherwise be able to replace a trusted command with its own, which is the
    one substitution this whole module exists to prevent.
    """
    merged: dict[str, Server] = {}
    newest = 0.0
    problems: list[str] = []
    for source in sources(home):
        entry = cached(source, home)
        if entry is None:
            continue
        newest = max(newest, entry.fetched)
        if entry.error:
            problems.append(f"{source}: {entry.error}")
        for server in entry.servers:
            merged.setdefault(server.id, server)
    for server in BUILTIN:
        merged[server.id] = server
    return Registry(
        tuple(sorted(merged.values(), key=_rank_key)), newest, tuple(problems)
    )


def enabled() -> bool:
    return (os.environ.get(NO_FETCH_ENV) or "").strip().lower() not in ("1", "true", "yes", "on")


def _load(source: str, fetch: Fetcher) -> tuple[list[Server], list[str], str]:
    """One source's servers, its complaints, and the reason it failed."""
    try:
        if source.startswith(("http://", "https://")):
            raw = fetch(source)
        else:
            raw = json.loads(Path(source).expanduser().read_text(encoding="utf-8"))
    except Exception as exc:  # a registry is never worth a traceback
        return [], [], f"{type(exc).__name__}: {exc}"
    found, problems = parse_registry(raw, source=source, fetched=True)
    return found, problems, ""


def refresh(*, fetch: Fetcher | None = None, force: bool = False,
            home: Path | None = None, only: Iterable[str] | None = None) -> Registry:
    """Re-read every source whose cache has gone stale.  Never raises.

    A failed source keeps the servers it had.  One flaky launch must not empty
    a catalogue the user could search yesterday.
    """
    if not enabled():
        return catalogue(home)
    get = fetch if fetch is not None else _get
    wanted = list(only) if only is not None else sources(home)
    for source in wanted:
        previous = cached(source, home)
        if not force and not stale(previous):
            continue
        found, problems, error = _load(source, get)
        if error:
            keep = previous.servers if previous is not None else ()
            store(Cached(source, keep, error, time.time()), home)
            continue
        store(Cached(source, tuple(found), "; ".join(problems[:3]), time.time()), home)
    return catalogue(home)


def refresh_async(*, fetch: Fetcher | None = None, done: threading.Event | None = None,
                  home: Path | None = None) -> None:
    """Refresh on a background thread.  Startup waits for nothing.

    The home directory is resolved *here*, on the calling thread.  See the
    module docstring: a worker that asks `settings.home()` itself writes into
    whatever `OFFSET_HOME` says by the time it gets round to it.
    """
    if not enabled():
        if done is not None:
            done.set()
        return

    where = home if home is not None else settings.home()

    def work() -> None:
        try:
            refresh(fetch=fetch, home=where)
        except Exception:
            pass  # nothing about browsing a catalogue is worth a traceback
        finally:
            if done is not None:
                done.set()

    threading.Thread(target=work, name="offset-mcp-registry", daemon=True).start()


def refresh_on_start(state: Any) -> None:
    """Startup hook: begin the refresh, block nothing, raise nothing."""
    try:
        refresh_async(home=settings.home())
    except Exception:
        pass  # a marketplace that cannot warm its cache is still a marketplace


# -- the installed set -------------------------------------------------------


def config_file(home: Path | None = None) -> Path:
    """The user-level `mcp.json`.

    Installs go here, never into the workspace: a marketplace install is a
    statement about this machine, and writing into a repository would commit
    one user's choices to everybody who clones it.
    """
    return (home if home is not None else settings.home()) / CONFIG_NAME


def _read_config(home: Path | None = None) -> dict[str, Any]:
    try:
        raw = json.loads(config_file(home).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _table(doc: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The server table and the key it lives under, creating neither in the
    document until there is something to write."""
    for key in SERVER_KEYS:
        value = doc.get(key)
        if isinstance(value, dict):
            return key, value
    return SERVER_KEYS[0], {}


@dataclass(frozen=True, slots=True)
class Installed:
    """One configured server as the marketplace sees it."""

    name: str
    kind: str = "stdio"
    target: str = ""
    server: Server | None = None
    missing: tuple[str, ...] = ()

    @property
    def trust(self) -> str:
        return self.server.trust if self.server is not None else UNKNOWN

    @property
    def configured(self) -> bool:
        return not self.missing

    def line(self) -> str:
        state = "ready" if self.configured else "needs " + ", ".join(self.missing)
        return f"{self.name:<24} {self.trust:<9} {self.kind:<5} {state}"


def installed(home: Path | None = None, *, environ: Mapping[str, str] | None = None,
              known: Registry | None = None) -> list[Installed]:
    """Every server in the user's config, with trust and what it still needs.

    The raw document is read rather than `load_config`, because that function
    drops a server whose `${VAR}` is unset - and those are precisely the ones
    this listing has to name.
    """
    catalogue_ = known if known is not None else catalogue(home)
    _, table = _table(_read_config(home))
    out: list[Installed] = []
    for name, entry in table.items():
        if not isinstance(entry, dict):
            continue
        command = str(entry.get("command") or "")
        args = [str(a) for a in entry.get("args") or []]
        url = str(entry.get("url") or "")
        gone: list[str] = []
        for value in (entry.get("env") or {}).values() if isinstance(entry.get("env"), dict) else ():
            _, absent = expand(str(value), environ=environ)
            gone.extend(absent)
        out.append(Installed(
            name=str(name),
            kind="stdio" if command else "http",
            target=" ".join([command, *args]).strip() or url,
            server=catalogue_.get(str(name)),
            missing=tuple(dict.fromkeys(gone)),
        ))
    return sorted(out, key=lambda i: i.name.lower())


# -- install and remove ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Action:
    """What an install or a remove did, and what the user must still do."""

    ok: bool
    id: str = ""
    name: str = ""
    message: str = ""
    missing: tuple[str, ...] = ()
    changed: bool = False
    path: Path | None = None
    needs_confirmation: bool = False

    def lines(self) -> list[str]:
        out = [self.message] if self.message else []
        if self.missing:
            out.append("unconfigured: set " + ", ".join(self.missing))
        if self.path is not None and self.changed:
            out.append(f"written to {self.path}")
        return out


def install(server_id: str, *, confirm: bool = False, home: Path | None = None,
            environ: Mapping[str, str] | None = None, known: Registry | None = None) -> Action:
    """Record a server in the user's config.  Executes nothing, ever.

    An untrusted server is refused without `confirm=True`; that refusal is the
    trust gate, and it is here rather than in the command so that any caller -
    a command, a tool, a script - gets it.
    """
    catalogue_ = known if known is not None else catalogue(home)
    server = catalogue_.get(server_id)
    if server is None:
        return Action(False, id=server_id, message=f"no server matching {server_id!r} in the catalogue")
    if not server.installable:
        return Action(False, id=server.id, message=f"{server.id} declares no usable command or url")
    if not server.trusted and not confirm:
        return Action(
            False, id=server.id, name=server.slug, needs_confirmation=True,
            message=(f"{server.id} is {server.trust}, not on offset's trusted list "
                     f"- confirm the install explicitly to add it"),
        )

    path = config_file(home)
    doc = _read_config(home)
    key, table = _table(doc)
    replaced = server.slug in table
    table[server.slug] = server.entry()
    doc[key] = table
    try:
        _write_json(path, doc)
    except OSError as exc:
        return Action(False, id=server.id, name=server.slug, message=f"could not write {path}: {exc}")

    gone = server.missing(environ)
    verb = "updated" if replaced else "installed"
    note = f"{verb} {server.slug} ({server.trust})"
    if gone:
        note += " - unconfigured"
    return Action(True, id=server.id, name=server.slug, message=note,
                  missing=gone, changed=True, path=path)


def remove(server_id: str, *, home: Path | None = None, known: Registry | None = None) -> Action:
    """Drop a server from the user's config.  Idempotent by design.

    Removing something that is not there succeeds and says so.  The alternative
    - an error - makes `/market remove x` fail as a way of cleaning up after a
    partial install, which is exactly when a user reaches for it.
    """
    path = config_file(home)
    doc = _read_config(home)
    key, table = _table(doc)

    wanted = (server_id or "").strip()
    name = wanted if wanted in table else ""
    if not name:
        catalogue_ = known if known is not None else catalogue(home)
        server = catalogue_.get(wanted)
        if server is not None and server.slug in table:
            name = server.slug
    if not name:
        lowered = {n.lower(): n for n in table}
        name = lowered.get(wanted.lower(), "")
    if not name:
        return Action(True, id=wanted, message=f"{wanted} is not installed")

    del table[name]
    doc[key] = table
    try:
        _write_json(path, doc)
    except OSError as exc:
        return Action(False, id=wanted, name=name, message=f"could not write {path}: {exc}")
    return Action(True, id=wanted, name=name, message=f"removed {name}", changed=True, path=path)


def search(query: str, *, home: Path | None = None, known: Registry | None = None) -> list[Server]:
    """Servers matching `query` on id, name or description, best first."""
    return (known if known is not None else catalogue(home)).search(query)


def info(server_id: str, *, home: Path | None = None, known: Registry | None = None) -> Server | None:
    """One server in full, or None if nothing matches."""
    return (known if known is not None else catalogue(home)).get(server_id)


# -- the shell surface -------------------------------------------------------

#: Words that mean "yes, I know it is untrusted".  `/market install x` alone
#: must never be enough.
CONFIRM_FLAGS: Final = ("--yes", "-y", "--confirm", "--trust")

USAGE: Final = (
    "/market search <q> | info <id> | install <id> [--yes] | remove <id> | list | refresh | source"
)


def _market(state: Any, args: list[str]) -> Any:
    """`/market ...`, and the body of `/mcp market ...`."""
    from offset.shell.commands import TONE_ERR, TONE_INFO, TONE_OK, Outcome

    confirm = any(a in CONFIRM_FLAGS for a in args)
    rest = [a for a in args if a not in CONFIRM_FLAGS]
    action = (rest[0].lower() if rest else "list")
    rest = rest[1:]
    home = settings.home()

    if action in ("search", "find"):
        if not rest:
            return Outcome.error("usage: /market search <query>")
        hits = search(" ".join(rest), home=home)
        if not hits:
            return Outcome([f"nothing matching {' '.join(rest)!r}",
                            "/market refresh re-reads the registries"], TONE_INFO)
        return Outcome([s.line() for s in hits[:20]], TONE_INFO)

    if action in ("info", "show"):
        if not rest:
            return Outcome.error("usage: /market info <id>")
        server = info(rest[0], home=home)
        if server is None:
            return Outcome.error(f"no server matching {rest[0]!r}", "/market search finds them")
        here = {i.name for i in installed(home, known=Registry((server,)))}
        lines = server.detail()
        lines.append("installed" if server.slug in here else "not installed")
        return Outcome(lines, TONE_INFO)

    if action in ("install", "add", "get"):
        if not rest:
            return Outcome.error("usage: /market install <id> [--yes]")
        done = install(rest[0], confirm=confirm, home=home)
        if not done.ok:
            tail = ["re-run with --yes to accept the risk"] if done.needs_confirmation else []
            return Outcome([done.message, *tail], TONE_ERR)
        lines = [*done.lines(), *_reconnect(state)]
        return Outcome(lines, TONE_INFO if done.missing else TONE_OK)

    if action in ("remove", "rm", "uninstall", "delete"):
        if not rest:
            return Outcome.error("usage: /market remove <id>")
        done = remove(rest[0], home=home)
        if not done.ok:
            return Outcome([done.message], TONE_ERR)
        lines = [*done.lines(), *(_reconnect(state) if done.changed else [])]
        return Outcome(lines, TONE_OK if done.changed else TONE_INFO)

    if action in ("list", "ls", "installed"):
        here = installed(home)
        if not here:
            return Outcome(["no MCP servers installed",
                            "/market search <q> then /market install <id>"], TONE_INFO)
        return Outcome([i.line() for i in here], TONE_INFO)

    if action in ("refresh", "update", "sync"):
        def job() -> Any:
            found = refresh(force=True, home=home)
            trusted = sum(1 for s in found if s.trusted)
            lines = [f"{len(found)} server(s) in the catalogue",
                     f"{trusted} trusted, {len(found) - trusted} needing confirmation"]
            lines.extend(f"source: {problem}" for problem in found.errors[:3])
            return Outcome(lines, TONE_OK)

        return Outcome(["refreshing the MCP registries..."], TONE_INFO, job=job)

    if action in ("source", "sources", "registry"):
        if not rest:
            return Outcome([*sources(home), "",
                            "/market source add <url|path> | forget <url|path>"], TONE_INFO)
        verb = rest[0].lower()
        if verb in ("add", "+") and len(rest) > 1:
            added = add_source(rest[1], home)
            return Outcome([f"{'added' if added else 'already known'}: {rest[1]}",
                            "/market refresh reads it"], TONE_OK if added else TONE_INFO)
        if verb in ("forget", "remove", "rm", "-") and len(rest) > 1:
            gone = forget_source(rest[1], home)
            return Outcome([f"{'forgotten' if gone else 'not a user-added source'}: {rest[1]}"],
                           TONE_OK if gone else TONE_ERR)
        return Outcome.error("usage: /market source [add|forget] <url|path>")

    return Outcome.error(f"unknown subcommand {action!r}", f"usage: {USAGE}")


def _reconnect(state: Any) -> list[str]:
    """Make the running manager match the file we just changed.

    Without this the user installs a server, sees nothing in `/mcp`, and
    concludes the install silently failed.  A manager that refuses is not an
    error worth reporting as one: the next launch reads the file anyway.
    """
    manager = getattr(state, "mcp", None)
    reload = getattr(manager, "reload", None)
    if reload is None:
        return ["takes effect on the next launch"]
    try:
        changed = reload()
    except Exception:
        return ["takes effect on the next launch"]
    return [", ".join(changed)] if changed else []


def market_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command("market", "find, install and remove MCP servers", _market, usage=USAGE,
                aliases=("mcp-market",)),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """`COMMANDS` on demand.

    Built lazily because the handler imports from `offset.shell.commands`,
    which imports the MCP package: resolving at import time would be a cycle.
    The re-check after building is the same guard `offset.core.tasks` needs and
    for the same reason - importing the shell registry re-enters here before the
    outer call has stored anything, so a single access would otherwise build two
    lists and register every command twice.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = market_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
