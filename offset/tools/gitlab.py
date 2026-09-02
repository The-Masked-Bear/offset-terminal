"""GitLab, for the repositories offset could previously only see as git.

`core/forge` speaks GitHub and nothing else, so on a GitLab checkout every
forge-shaped command answered "no github remote here" — technically true and
completely useless.  This module is the GitLab counterpart, and it makes five
judgements worth stating.

**The remote is looked for, not assumed.**  `core/forge` asks git for the URL
of `origin`.  On GitLab that is wrong often enough to matter: the fork-based
workflow leaves `origin` pointing at a personal fork and `upstream` at the
project, and plenty of clones have a GitHub mirror sitting on `origin` with
GitLab on a second remote.  So `detect` reads *all* remotes from
`git remote -v` and picks the GitLab one, preferring `origin` only when it is
itself a candidate.

**The token is a header and only ever a header.**  It never goes into argv,
because the whole process table is world-readable, and never into a URL,
because URLs end up in error messages, logs and the transcript the model sees.
`GitLab.scrub` is the second line of that defence: every string a call returns
is passed through it, so a server that echoes the credential back in an error
body cannot leak it into the conversation either.

**Self-hosting cannot be sniffed, so it is declared.**  `gitlab.example.com`
is recognisable; `git.acme.internal` is not, and guessing that an unknown host
is a GitLab would produce API 404s dressed up as missing permissions.  A host
without "gitlab" in its name is refused with the name of the variable that
makes it work — `GITLAB_HOST`.

**Pagination is followed.**  GitLab's default page is twenty items.  A repo
with three hundred issues answering with twenty is not a small error; it is a
wrong answer that looks entirely right, and neither the user nor the model has
any way to tell.  `paged` follows `X-Next-Page` to the end, under a page
ceiling so that a broken proxy which always sets the header cannot spin.

**Failure degrades into a sentence naming the fix.**  No remote says so; no
token names the variable to set; a 401 or a 403 says which token scope is
missing rather than reporting the number.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from offset.core import vcs
from offset.tools.base import Danger, Tool, ToolContext, ToolResult

#: gitlab.com itself.  Self-hosted instances put the same API under their own
#: host, which is why the base is derived from the remote rather than fixed.
PUBLIC_HOST: Final = "gitlab.com"

#: Names the API path is built from.  v4 is the only version GitLab has shipped
#: since 9.0 and the only one worth supporting.
API_PREFIX: Final = "api/v4"

#: Where a self-hosted host is declared.  Required because a host that is not
#: called "gitlab" anything is indistinguishable from any other git server.
HOST_ENV: Final = "GITLAB_HOST"

#: Token sources, in precedence order.  The offset-specific name wins so a user
#: who already exports `GITLAB_TOKEN` for `glab` can point offset at a
#: different, narrower token without disturbing their shell.
TOKEN_KEYS: Final = ("OFFSET_GITLAB_TOKEN", "GITLAB_TOKEN")

#: The header GitLab authenticates personal access tokens with.  `Bearer` is
#: for OAuth tokens only and a PAT sent that way earns a 401 that looks like an
#: expiry, which is a confusing failure to hand a user.
TOKEN_HEADER: Final = "PRIVATE-TOKEN"

#: What a token is replaced with on its way out.  Not empty: an empty string
#: would make a leak look like a formatting bug rather than a redaction.
REDACTED: Final = "<gitlab-token>"

#: GitLab's own per-page maximum.  Asking for more is silently clamped, so this
#: is the largest number of round trips we can save.
PER_PAGE: Final = 100

#: How many items any one listing will accumulate.  High enough that a real
#: project's issue list arrives whole, bounded so a pathological project cannot
#: exhaust the context window.
MAX_ITEMS: Final = 1000

#: Page ceiling.  `X-Next-Page` is server-controlled, and a misconfigured proxy
#: that copies the header onto every response would otherwise loop forever.
MAX_PAGES: Final = 40

#: Lines of a failed job's trace to keep.  A GitLab trace is routinely
#: megabytes of setup noise; the failure is always at the end.
LOG_TAIL: Final = 120

#: Characters of an issue or MR body to render.  Enough to carry the report,
#: short enough that ten of them do not fill a context window.
BODY_BUDGET: Final = 2000

#: Rows a listing renders before it stops.  Matches what fits on a screen.
LIST_LIMIT: Final = 30

TIMEOUT: Final = 30.0

USER_AGENT: Final = "offset"

#: `git@host:group/project.git`, the scp-like dialect git accepts but which is
#: not a URL and so cannot be given to `urllib.parse`.
_SCP = re.compile(r"^(?:[^@/:]+@)?(?P<host>[^/:]+):(?P<path>[^:].*)$")

#: One line of `git remote -v`: name, whitespace, url, then `(fetch)`/`(push)`.
_REMOTE_LINE = re.compile(r"^(?P<name>\S+)\s+(?P<url>\S+)(?:\s+\((?:fetch|push)\))?$")

#: CSI escapes.  GitLab wraps every trace line in colour codes and terminates
#: each with `\x1b[0K`, which is noise the model pays tokens to read.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

#: GitLab's collapsible-section markers, which are data for the web UI and
#: nothing at all for a reader.
_SECTION = re.compile(r"section_(?:start|end):\d+:[A-Za-z0-9_.\-\[\]]+\s*")

#: Job statuses that mean the pipeline stopped because of this job.  `canceled`
#: is excluded on purpose: its log explains nothing.
FAILED_STATES: Final = ("failed",)


# -- one HTTP round trip ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class Call:
    """One request, as data, so a test can assert on what would be sent."""

    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes | None = None
    timeout: float = TIMEOUT


@dataclass(frozen=True, slots=True)
class Answer:
    """One response.  `error` is set only when there was no HTTP answer."""

    status: int = 0
    headers: Mapping[str, str] = field(default_factory=dict)
    text: str = ""
    error: str = ""

    def header(self, name: str) -> str:
        """Case-insensitive lookup: `X-Next-Page` arrives capitalised however
        the proxy in front of GitLab felt like capitalising it."""
        wanted = name.lower()
        for key, value in self.headers.items():
            if str(key).lower() == wanted:
                return str(value or "").strip()
        return ""


#: Performs one round trip.  Injected everywhere so nothing in the tests, and
#: nothing in a dry run, can reach a network.
Fetcher = Callable[[Call], Answer]

#: Runs `git <args>` in a directory and yields its stdout.  Injected so remote
#: parsing can be tested without a repository on disk.
Git = Callable[[Sequence[str], Path], str]


def urlopen_fetch(call: Call) -> Answer:
    """The real transport.  Never raises; a dead network is an `Answer`."""
    request = urllib.request.Request(call.url, data=call.body, method=call.method.upper())
    for key, value in call.headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=call.timeout) as answer:
            return Answer(
                status=answer.status,
                headers={k: v for k, v in answer.headers.items()},
                text=answer.read().decode("utf-8", errors="replace"),
            )
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # an unreadable body must not mask the status code
            body = ""
        return Answer(status=exc.code, headers={k: v for k, v in (exc.headers or {}).items()}, text=body)
    except urllib.error.URLError as exc:
        return Answer(error=f"could not reach {_host_of(call.url)}: {exc.reason}")
    except (TimeoutError, OSError) as exc:
        return Answer(error=f"could not reach {_host_of(call.url)}: {type(exc).__name__}: {exc}")


def git_stdout(args: Sequence[str], cwd: Path) -> str:
    """The default `Git`.  A failed git call is empty output, not an exception:
    every caller here already has to handle "no remotes"."""
    got = vcs.run(list(args), cwd)
    return got.text if got.ok else ""


def _host_of(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url).netloc or url
    except ValueError:
        return url


# -- the remote -------------------------------------------------------------


def is_gitlab_host(host: str, *, override: str = "") -> bool:
    """Whether `host` is a GitLab instance.

    The rule is deliberately narrow: an exact match against `GITLAB_HOST`, or
    "gitlab" appearing as a whole dotted label.  That accepts `gitlab.com`,
    `gitlab.example.org` and `code.gitlab.internal` and rejects `github.com`
    and every unrelated git server, which is the right way round — treating an
    unknown host as GitLab turns every subsequent 404 into a mystery.
    """
    name = (host or "").strip().lower()
    if not name:
        return False
    if override and name == override.strip().lower().removeprefix("https://").removeprefix("http://").rstrip("/"):
        return True
    return "gitlab" in name.split(".")


@dataclass(frozen=True, slots=True)
class Remote:
    """A GitLab remote, split into what the v4 API needs.

    `path` is the full namespace, subgroups included, because GitLab projects
    genuinely nest (`group/subgroup/project`) and truncating that to the last
    two segments addresses a project that does not exist.
    """

    host: str = ""
    path: str = ""
    url: str = ""
    #: Which git remote this came from, so a message can say `upstream` rather
    #: than leaving the user to guess which of their remotes was used.
    name: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.host and self.path)

    @property
    def project(self) -> str:
        """The URL-encoded project id.  GitLab accepts a namespaced path in
        place of a numeric id only when every slash is percent-encoded."""
        return urllib.parse.quote(self.path, safe="")

    @property
    def repo(self) -> str:
        return self.path.rpartition("/")[2]

    @property
    def namespace(self) -> str:
        return self.path.rpartition("/")[0]

    @property
    def api_base(self) -> str:
        return f"https://{self.host}/{API_PREFIX}"

    @property
    def web_url(self) -> str:
        return f"https://{self.host}/{self.path}"

    def report(self) -> list[str]:
        if self.error:
            return [self.error]
        via = f" via {self.name}" if self.name else ""
        return [f"{self.path} on {self.host}{via}"]


def parse_remote(url: str, *, name: str = "", host_override: str = "") -> Remote:
    """Split one git remote URL into a GitLab `Remote`.

    Handles `git@host:group/project.git`, `ssh://git@host:2222/group/project`,
    `https://user@host/group/sub/project.git` and a bare `host/group/project`.
    A remote that is not GitLab comes back as an error value naming
    `GITLAB_HOST`, rather than being parsed into something that will 404.
    """
    raw = (url or "").strip()
    if not raw:
        return Remote(name=name, error="no remote url to parse")

    host = ""
    path = ""
    if "://" in raw:
        scheme, _, rest = raw.partition("://")
        if scheme.lower() not in ("ssh", "git", "http", "https", "git+ssh"):
            return Remote(url=raw, name=name, error=f"unsupported remote scheme {scheme!r}: {raw}")
        authority, _, path = rest.partition("/")
        host = authority.rpartition("@")[2]
    else:
        found = _SCP.match(raw)
        if found:
            host, path = found.group("host"), found.group("path")
        elif "/" in raw:
            host, _, path = raw.partition("/")
        else:
            return Remote(url=raw, name=name, error=f"unrecognised remote url: {raw}")

    # A port is not part of the host as far as the API base is concerned, and
    # an ssh remote on 2222 still serves its API on 443.
    host = host.split(":", 1)[0].strip().lower()
    segments = [s for s in path.strip("/").split("/") if s]
    if segments:
        segments[-1] = segments[-1].removesuffix(".git")
    if not host or len(segments) < 2 or not segments[-1]:
        return Remote(host=host, url=raw, name=name, error=f"unrecognised remote url: {raw}")

    if not is_gitlab_host(host, override=host_override):
        return Remote(
            host=host,
            url=raw,
            name=name,
            error=(
                f"{host} is not a GitLab host, so the GitLab API does not apply here. "
                f"if it is a self-hosted GitLab, set {HOST_ENV}={host}"
            ),
        )
    return Remote(host=host, path="/".join(segments), url=raw, name=name)


def remote_urls(cwd: Path | str, *, git: Git = git_stdout) -> list[tuple[str, str]]:
    """Every configured `(name, url)` pair, in `git remote -v` order, deduped.

    `git remote -v` prints each remote twice, once for fetch and once for push,
    and a triangular workflow gives those two different URLs — hence dedup on
    the pair rather than on the name.
    """
    text = git(["remote", "-v"], Path(cwd))
    pairs: list[tuple[str, str]] = []
    for line in text.splitlines():
        found = _REMOTE_LINE.match(line.strip())
        if not found:
            continue
        pair = (found.group("name"), found.group("url"))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def detect(cwd: Path | str, *, git: Git = git_stdout, env: Mapping[str, str] | None = None) -> Remote:
    """The GitLab remote for this checkout.  Never raises.

    `origin` wins when it is a GitLab remote, because that is what the user
    means by "this project".  When it is not — a GitHub mirror on `origin` and
    the real project on `gitlab`, which is a common mirroring setup — the first
    GitLab remote in git's own order is used and named in the report.
    """
    where = env if env is not None else os.environ
    override = (where.get(HOST_ENV) or "").strip()
    pairs = remote_urls(cwd, git=git)
    if not pairs:
        return Remote(error="no git remotes are configured here, so there is no GitLab project to talk to")

    parsed = [parse_remote(url, name=name, host_override=override) for name, url in pairs]
    for candidate in parsed:
        if candidate.ok and candidate.name == "origin":
            return candidate
    for candidate in parsed:
        if candidate.ok:
            return candidate
    named = ", ".join(f"{name} -> {url}" for name, url in pairs[:4])
    return Remote(
        error=(
            f"no GitLab remote here; configured remotes: {named}. "
            f"if one of these is a self-hosted GitLab, set {HOST_ENV} to its host"
        )
    )


# -- credentials ------------------------------------------------------------


def glab_config_path(env: Mapping[str, str] | None = None) -> Path:
    """Where `glab` keeps `config.yml`, following glab's own precedence.

    Resolved from an explicit environment mapping so the caller can do it on
    its own thread; a background thread that reads `os.environ` and `Path.home`
    for itself has been the source of one real bug in this codebase already.
    """
    where = env if env is not None else os.environ
    if where.get("GLAB_CONFIG_DIR"):
        return Path(where["GLAB_CONFIG_DIR"]) / "config.yml"
    if where.get("XDG_CONFIG_HOME"):
        return Path(where["XDG_CONFIG_HOME"]) / "glab-cli" / "config.yml"
    home = where.get("HOME")
    base = Path(home) if home else Path.home()
    return base / ".config" / "glab-cli" / "config.yml"


#: Keys that appear *beside* a token inside a host block.  Listed explicitly so
#: the scanner below can tell a host key from a field key without a YAML
#: parser: anything not in here, with no value, at host depth, is a host.
_GLAB_FIELDS: Final = ("token", "api_host", "api_protocol", "git_protocol", "user", "username", "container_registry_domains")


def read_glab_token(path: Path, host: str = PUBLIC_HOST) -> str | None:
    """Pull one `token` out of `glab`'s `config.yml`.

    A targeted line scanner, not a YAML parser, and one on purpose: taking on
    a YAML dependency to read six lines of configuration is not a trade worth
    making.  It understands exactly the shape glab writes — a `hosts:` block at
    column zero, a host key beneath it, `token: <value>` beneath that — and
    returns None for anything else rather than guessing.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    in_hosts = False
    host_depth = -1
    current = ""
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        depth = len(raw) - len(raw.lstrip())
        key, sep, value = raw.strip().partition(":")
        if not sep:
            continue
        key = key.strip().strip("\"'")
        if depth == 0:
            in_hosts = key == "hosts"
            current = ""
            host_depth = -1
            continue
        if not in_hosts:
            continue
        if not value.strip() and key not in _GLAB_FIELDS and (host_depth < 0 or depth == host_depth):
            host_depth = depth
            current = key.lower()
            continue
        if current == host.lower() and key == "token":
            token = value.strip().strip("\"'")
            if token:
                return token
    return None


def find_token(host: str = PUBLIC_HOST, *, env: Mapping[str, str] | None = None) -> str:
    """A token for `host`: the environment first, then glab's config."""
    where = env if env is not None else os.environ
    for key in TOKEN_KEYS:
        value = (where.get(key) or "").strip()
        if value:
            return value
    return read_glab_token(glab_config_path(where), host) or ""


def no_token_message(host: str) -> str:
    """The sentence a missing credential earns.  It names the variable, because
    "unauthenticated" tells a user nothing they can act on."""
    return (
        f"no GitLab token for {host}: set {TOKEN_KEYS[1]} (or {TOKEN_KEYS[0]}) to a personal access "
        "token with the 'api' scope, or run 'glab auth login'"
    )


# -- one answer -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reply:
    """One API call's answer, or the reason there is none."""

    ok: bool
    status: int = 0
    data: Any = None
    text: str = ""
    error: str = ""
    #: How many pages were fetched.  Asserted on by the pagination tests and
    #: worth reporting: "300 issues over 3 pages" is a checkable claim.
    pages: int = 1

    @classmethod
    def fail(cls, error: str, *, status: int = 0, text: str = "") -> "Reply":
        return cls(False, status, None, text, error)

    @classmethod
    def good(cls, status: int, data: Any, text: str = "", *, pages: int = 1) -> "Reply":
        return cls(True, status, data, text, "", pages)

    @property
    def items(self) -> list[Any]:
        return list(self.data) if isinstance(self.data, list) else []

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default) if isinstance(self.data, dict) else default


def _decode(text: str) -> Any:
    """JSON when it is JSON.  The trace endpoint answers plain text."""
    stripped = (text or "").strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except ValueError:
        return None


def _said(payload: Any, fallback: str) -> str:
    """GitLab's own error sentence.  It uses `message` for most failures and
    `error` for OAuth ones, and `message` is sometimes a dict of field errors."""
    if isinstance(payload, dict):
        for key in ("message", "error", "error_description"):
            got = payload.get(key)
            if isinstance(got, str) and got.strip():
                return got.strip()
            if isinstance(got, dict):
                parts = [f"{k}: {', '.join(str(v) for v in vs)}" if isinstance(vs, list) else f"{k}: {vs}"
                         for k, vs in got.items()]
                if parts:
                    return "; ".join(parts)
            if isinstance(got, list) and got:
                return "; ".join(str(v) for v in got)
    return fallback


def diagnose(status: int, answer: Answer, payload: Any, *, path: str, host: str) -> str:
    """Turn an HTTP failure into a sentence that says what to do next.

    Every branch names a scope or a variable.  Reporting "403" and stopping
    leaves the user to work out that GitLab's `read_api` scope cannot post a
    note, which is exactly the sort of thing this should say out loud.
    """
    said = _said(payload, (answer.text or "").strip()[:200] or f"http {status}")
    if status == 401:
        return (
            f"gitlab rejected the token (401): {said}. this is a scope or lifetime problem, not a path "
            f"problem - the token is expired, revoked, or was created without the 'api' scope. set "
            f"{TOKEN_KEYS[1]} to a fresh personal access token with the 'api' scope"
        )
    if status == 403:
        return (
            f"gitlab refused (403): {said}. the token is valid but lacks the scope for {path} - reading "
            "needs the 'read_api' scope and creating a merge request or a note needs the full 'api' scope"
        )
    if status == 404:
        return (
            f"gitlab returned 404 for {path}. gitlab answers 404 rather than 403 for a project a token "
            f"cannot see, so either the path is wrong or the token needs the 'read_api' scope on "
            f"{host}"
        )
    if status == 409:
        return f"gitlab refused as a conflict (409): {said}"
    if status in (400, 422):
        return f"gitlab rejected the request ({status}): {said}"
    if status == 429:
        retry = answer.header("retry-after")
        wait = f" retry after {retry}s." if retry else ""
        return f"gitlab rate limit reached (429).{wait} the limit is per token and per ip"
    return f"gitlab returned {status} for {path}: {said}"


# -- the project ------------------------------------------------------------


@dataclass(slots=True)
class GitLab:
    """One GitLab project, reachable or not.

    `problem` is set when the client cannot be used at all — no remote, no
    token — so that every command can fail with one sentence before making a
    request that was always going to be refused.
    """

    remote: Remote = field(default_factory=Remote)
    token: str = ""
    #: A plain function held in a slot, never a class attribute, so it is
    #: stored rather than bound as a method when it is called back.
    fetch: Fetcher = urlopen_fetch
    timeout: float = TIMEOUT
    problem: str = ""

    def __post_init__(self) -> None:
        if not self.problem:
            if not self.remote.ok:
                self.problem = self.remote.error or "no GitLab remote here"
            elif not self.token:
                self.problem = no_token_message(self.remote.host)

    @property
    def ok(self) -> bool:
        return not self.problem

    def scrub(self, text: str) -> str:
        """Remove the token from anything on its way out.

        The choke point for the promise that the credential never reaches the
        transcript: a GitLab that echoes a bad header back in its error body,
        or a proxy that reflects the request, cannot leak it past here.
        """
        if not self.token or not text:
            return text
        return text.replace(self.token, REDACTED)

    def url_for(self, path: str, params: Mapping[str, Any] | None = None) -> str:
        """The absolute URL for an API path.  The token is never a parameter."""
        base = path if path.startswith(("http://", "https://")) else f"{self.remote.api_base}/{path.lstrip('/')}"
        pairs = [(k, str(v)) for k, v in (params or {}).items() if v not in (None, "")]
        if not pairs:
            return base
        joiner = "&" if "?" in base else "?"
        return f"{base}{joiner}{urllib.parse.urlencode(pairs)}"

    def headers(self, *, json_body: bool) -> dict[str, str]:
        head = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if json_body:
            head["Content-Type"] = "application/json"
        if self.token:
            head[TOKEN_HEADER] = self.token
        return head

    def _round_trip(self, call: Call, path: str) -> tuple[Answer | None, Reply | None]:
        """One round trip, with every failure already a sentence.

        Shared by `call` and `paged` so that the diagnosis of a 401 does not
        depend on which of them happened to make the request - when the two
        had their own copies, a paged listing reported a bare status code and
        a single read reported the missing scope.
        """
        try:
            answer = self.fetch(call)
        except Exception as exc:
            # A fetcher that raises is a bug in the fetcher or a transport
            # nobody anticipated; neither is a reason to take the shell down
            # in the middle of a command.
            return None, Reply.fail(
                self.scrub(f"could not reach {self.remote.host}: {type(exc).__name__}: {exc}")
            )
        if answer.error:
            return None, Reply.fail(self.scrub(answer.error), status=answer.status)
        if answer.status >= 400:
            reason = diagnose(answer.status, answer, _decode(answer.text), path=path, host=self.remote.host)
            return None, Reply.fail(self.scrub(reason), status=answer.status, text=self.scrub(answer.text))
        return answer, None

    def call(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Reply:
        """One request.  Never raises, and never returns the token."""
        if self.problem:
            return Reply.fail(self.problem)
        payload = json.dumps(dict(body)).encode("utf-8") if body is not None else None
        call = Call(
            method=method.upper(),
            url=self.url_for(path, params),
            headers=self.headers(json_body=payload is not None),
            body=payload,
            timeout=self.timeout,
        )
        answer, failed = self._round_trip(call, path)
        if failed is not None or answer is None:
            return failed or Reply.fail("no answer")
        return Reply.good(answer.status, _scrub_data(_decode(answer.text), self.token), self.scrub(answer.text))

    def paged(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        limit: int = MAX_ITEMS,
    ) -> Reply:
        """Follow `X-Next-Page` to the end of a listing.

        GitLab's default page is 20 items.  Returning the first page and
        calling it the answer is the silent-truncation bug this exists to
        prevent: a project with 300 issues must not report 20 of them with no
        indication that the rest exist.
        """
        if self.problem:
            return Reply.fail(self.problem)
        collected: list[Any] = []
        page = 1
        pages = 0
        while page and pages < MAX_PAGES and len(collected) < limit:
            query = dict(params or {})
            query["per_page"] = min(PER_PAGE, max(1, limit - len(collected)))
            query["page"] = page
            call = Call(
                method="GET",
                url=self.url_for(path, query),
                headers=self.headers(json_body=False),
                timeout=self.timeout,
            )
            answer, failed = self._round_trip(call, path)
            if failed is not None or answer is None:
                return failed or Reply.fail("no answer")
            pages += 1
            decoded = _decode(answer.text)
            if not isinstance(decoded, list):
                return Reply.fail(f"expected a list from {path}, got {type(decoded).__name__}")
            collected.extend(decoded)
            if not decoded:
                break
            nxt = answer.header("X-Next-Page")
            page = int(nxt) if nxt.isdigit() and int(nxt) > page else 0
        return Reply.good(200, _scrub_data(collected[:limit], self.token), pages=pages or 1)

    # -- issues ----------------------------------------------------------

    def issues(self, *, state: str = "opened", limit: int = MAX_ITEMS, search: str = "") -> Reply:
        return self.paged(
            f"projects/{self.remote.project}/issues",
            params={"state": _state(state), "order_by": "updated_at", "search": search},
            limit=limit,
        )

    def issue(self, iid: int) -> Reply:
        return self.call("GET", f"projects/{self.remote.project}/issues/{int(iid)}")

    # -- merge requests --------------------------------------------------

    def merge_requests(self, *, state: str = "opened", limit: int = MAX_ITEMS) -> Reply:
        return self.paged(
            f"projects/{self.remote.project}/merge_requests",
            params={"state": _state(state), "order_by": "updated_at"},
            limit=limit,
        )

    def merge_request(self, iid: int) -> Reply:
        return self.call("GET", f"projects/{self.remote.project}/merge_requests/{int(iid)}")

    def merge_requests_for(self, branch: str) -> Reply:
        return self.paged(
            f"projects/{self.remote.project}/merge_requests",
            params={"state": "opened", "source_branch": branch},
            limit=20,
        )

    def merge_request_changes(self, iid: int) -> Reply:
        return self.call("GET", f"projects/{self.remote.project}/merge_requests/{int(iid)}/changes")

    def create_merge_request(
        self,
        *,
        source: str,
        target: str,
        title: str,
        description: str = "",
        remove_source: bool = True,
        squash: bool = False,
    ) -> Reply:
        """Open a merge request.  GitLab names the fields `source_branch` and
        `target_branch`, not head and base; getting that wrong is a 400 with a
        message that does not say which field was missing."""
        if not source or not target:
            return Reply.fail("a merge request needs a source and a target branch")
        if not title.strip():
            return Reply.fail("a merge request needs a title")
        return self.call(
            "POST",
            f"projects/{self.remote.project}/merge_requests",
            body={
                "source_branch": source,
                "target_branch": target,
                "title": title.strip(),
                "description": description,
                "remove_source_branch": remove_source,
                "squash": squash,
            },
        )

    def comment(self, kind: str, iid: int, body: str) -> Reply:
        """Post a note on an issue or a merge request."""
        collection = _collection(kind)
        if not collection:
            return Reply.fail(f"comment on what? {kind!r} is neither 'issue' nor 'mr'")
        if not body.strip():
            return Reply.fail("a comment needs a body")
        return self.call(
            "POST",
            f"projects/{self.remote.project}/{collection}/{int(iid)}/notes",
            body={"body": body},
        )

    # -- CI --------------------------------------------------------------

    def pipelines(self, *, ref: str = "", status: str = "", limit: int = 20) -> Reply:
        return self.paged(
            f"projects/{self.remote.project}/pipelines",
            params={"ref": ref, "status": status, "order_by": "id", "sort": "desc"},
            limit=limit,
        )

    def pipeline(self, pipeline_id: int) -> Reply:
        return self.call("GET", f"projects/{self.remote.project}/pipelines/{int(pipeline_id)}")

    def jobs(self, pipeline_id: int, *, limit: int = 200) -> Reply:
        return self.paged(
            f"projects/{self.remote.project}/pipelines/{int(pipeline_id)}/jobs",
            limit=limit,
        )

    def job_log(self, job_id: int) -> Reply:
        """A job's raw trace.  Plain text, so `data` stays None by design."""
        return self.call("GET", f"projects/{self.remote.project}/jobs/{int(job_id)}/trace")

    def failure(self, pipeline_id: int, *, lines: int = LOG_TAIL) -> "Failure":
        """The failing job of a pipeline and the tail of its log.

        Two calls rather than one because GitLab has no "give me the failure"
        endpoint, and the whole trace is never what anyone wants: the useful
        part is at the end, and forwarding the setup noise in front of it
        spends most of a context window reaching twenty relevant lines.
        """
        listed = self.jobs(pipeline_id)
        if not listed.ok:
            return Failure(error=listed.error or "could not list the pipeline's jobs")
        broken = [j for j in listed.items if isinstance(j, dict) and str(j.get("status")) in FAILED_STATES]
        if not broken:
            return Failure(error=f"pipeline {pipeline_id} has no failed job")
        # Latest attempt wins: a retried job keeps the old one in the listing,
        # and the old log describes a failure that has already been fixed.
        chosen = max(broken, key=lambda j: int(j.get("id") or 0))
        job_id = int(chosen.get("id") or 0)
        got = self.job_log(job_id)
        if not got.ok:
            return Failure(
                job=job_id,
                name=str(chosen.get("name") or ""),
                stage=str(chosen.get("stage") or ""),
                error=got.error or "could not fetch the job log",
            )
        return Failure(
            job=job_id,
            name=str(chosen.get("name") or ""),
            stage=str(chosen.get("stage") or ""),
            text=tail(clean_trace(got.text), lines),
        )


@dataclass(frozen=True, slots=True)
class Failure:
    """A failed job, and the part of its log that explains why."""

    job: int = 0
    name: str = ""
    stage: str = ""
    text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def report(self) -> list[str]:
        if self.error:
            return [self.error]
        head = f"--- job {self.job} {self.name or '?'} ({self.stage or 'no stage'}) failed ---"
        return [head, *self.text.splitlines()]


def _state(state: str) -> str:
    """GitLab says `opened`, everyone types `open`.  Accept both."""
    name = (state or "").strip().lower()
    if name in ("", "open", "opened"):
        return "opened"
    if name in ("closed", "close"):
        return "closed"
    if name in ("merged",):
        return "merged"
    if name in ("all", "any", "*"):
        return "all"
    return name


def _collection(kind: str) -> str:
    name = (kind or "").strip().lower()
    if name in ("issue", "issues"):
        return "issues"
    if name in ("mr", "mrs", "merge_request", "merge_requests", "pr"):
        return "merge_requests"
    return ""


def _scrub_data(data: Any, token: str) -> Any:
    """Redact the token inside a decoded payload.

    Needed because `call` scrubs the raw text but callers render from `data`;
    a GitLab error that quotes the offending header back would otherwise reach
    the model through the parsed dict.
    """
    if not token or data is None:
        return data
    if isinstance(data, str):
        return data.replace(token, REDACTED)
    if isinstance(data, list):
        return [_scrub_data(item, token) for item in data]
    if isinstance(data, dict):
        return {key: _scrub_data(value, token) for key, value in data.items()}
    return data


# -- log handling -----------------------------------------------------------


def clean_trace(text: str) -> str:
    """Strip the parts of a GitLab trace that carry no meaning.

    Colour codes, `\\x1b[0K` line terminators and `section_start:` markers are
    roughly a third of the bytes of a real trace and none of its information.
    Progress output separated by bare carriage returns is split into lines so
    the tail is a tail of output rather than of one enormous line.
    """
    stripped = _SECTION.sub("", _ANSI.sub("", text or ""))
    lines = stripped.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept = [line.rstrip() for line in lines]
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def tail(text: str, lines: int = LOG_TAIL) -> str:
    """The last `lines` lines.  The failure is always at the end of a trace."""
    if lines <= 0:
        return ""
    found = (text or "").splitlines()
    return "\n".join(found[-lines:])


# -- rendering --------------------------------------------------------------


def issue_line(issue: Mapping[str, Any]) -> str:
    author = (issue.get("author") or {}).get("username") if isinstance(issue.get("author"), dict) else ""
    labels = ",".join(str(v) for v in (issue.get("labels") or [])[:3])
    marks = f"  [{labels}]" if labels else ""
    # Coerced rather than formatted directly: a `:8s` spec raises on anything
    # that is not a string, and an API field is whatever the server sent.
    state = str(issue.get("state") or "")
    title = str(issue.get("title") or "")[:60]
    return f"#{issue.get('iid')} {state:8s} {title}  @{author or '?'}{marks}"


def mr_line(mr: Mapping[str, Any]) -> str:
    author = (mr.get("author") or {}).get("username") if isinstance(mr.get("author"), dict) else ""
    draft = "draft " if mr.get("draft") or mr.get("work_in_progress") else ""
    state = str(mr.get("state") or "")
    title = str(mr.get("title") or "")[:56]
    return (
        f"!{mr.get('iid')} {state:8s} {draft}{title}  "
        f"{mr.get('source_branch')} -> {mr.get('target_branch')}  @{author or '?'}"
    )


def pipeline_line(pipeline: Mapping[str, Any]) -> str:
    status = str(pipeline.get("status") or "?")
    ref = str(pipeline.get("ref") or "?")
    sha = str(pipeline.get("sha") or "")[:8]
    return f"#{pipeline.get('id')} {status:10s} {ref:24s} {sha}"


def job_line(job: Mapping[str, Any]) -> str:
    status = str(job.get("status") or "?")
    stage = str(job.get("stage") or "?")
    name = str(job.get("name") or "?")[:36]
    return f"{status:10s} {stage:12s} {name}  id={job.get('id')}"


def issue_report(data: Mapping[str, Any]) -> list[str]:
    author = (data.get("author") or {}).get("username") if isinstance(data.get("author"), dict) else ""
    return [
        f"#{data.get('iid')} {data.get('title')}",
        f"state: {data.get('state')}  by {author or '?'}  {data.get('web_url') or ''}",
        "",
        str(data.get("description") or "")[:BODY_BUDGET],
    ]


def mr_report(data: Mapping[str, Any]) -> list[str]:
    author = (data.get("author") or {}).get("username") if isinstance(data.get("author"), dict) else ""
    merge = data.get("detailed_merge_status") or data.get("merge_status") or "?"
    return [
        f"!{data.get('iid')} {data.get('title')}",
        f"state: {data.get('state')}  by {author or '?'}  {data.get('web_url') or ''}",
        f"{data.get('source_branch')} -> {data.get('target_branch')}  merge status: {merge}",
        "",
        str(data.get("description") or "")[:BODY_BUDGET],
    ]


# -- construction -----------------------------------------------------------


def connect(
    cwd: Path | str,
    *,
    env: Mapping[str, str] | None = None,
    git: Git = git_stdout,
    fetch: Fetcher = urlopen_fetch,
    timeout: float = TIMEOUT,
    remote: Remote | None = None,
) -> GitLab:
    """A `GitLab` for the checkout at `cwd`.  Never raises.

    Every environment-derived value — the host override, the token, glab's
    config path — is resolved here, on the calling thread, and handed to the
    client as data.  A client that looked them up lazily would read them on
    whichever worker thread happened to make the first request, after the
    caller that knew the right home directory had gone; that mistake has been
    made in this codebase once already and is not worth making twice.

    `remote` is for the caller that has already run `detect` to decide whether
    this is a GitLab checkout at all: passing the answer back in saves a
    second `git remote -v` subprocess on that path, which is the whole cost of
    the decision.
    """
    where = dict(env) if env is not None else dict(os.environ)
    found = remote if remote is not None else detect(cwd, git=git, env=where)
    if not found.ok:
        return GitLab(remote=found, fetch=fetch, timeout=timeout, problem=found.error or "no GitLab remote here")
    return GitLab(remote=found, token=find_token(found.host, env=where), fetch=fetch, timeout=timeout)


@dataclass(slots=True)
class IssueForge:
    """A GitLab client in the shape a forge-agnostic caller asks for.

    `core/issue_to_pr` drives an issue through to a pull request over three
    methods and does not want to know which forge is underneath.  This is the
    GitLab side of that, and it carries the one translation GitLab needs:
    GitLab calls an issue's body its `description` and numbers it `iid`, while
    the caller reads `body` and `number`.  Aliasing here rather than teaching
    that module about GitLab keeps the knowledge of GitLab's field names in
    the file that already has it.
    """

    client: GitLab

    def read_issue(self, number: int) -> Reply:
        got = self.client.issue(number)
        if not got.ok or not isinstance(got.data, dict):
            return got
        data = dict(got.data)
        data.setdefault("body", data.get("description") or "")
        data.setdefault("number", data.get("iid"))
        return Reply.good(got.status, data, got.text)

    def comment(self, number: int, body: str) -> Reply:
        return self.client.comment("issue", number, body)

    def open_pr(self, *, title: str, body: str = "", head: str = "", base: str = "", draft: bool = False) -> Reply:
        # GitLab has no `draft` field to post: a merge request is a draft
        # exactly when its title begins with "Draft:", so the flag has to be
        # spent on the title or it is silently dropped.
        subject = f"Draft: {title}" if draft and not title.lower().startswith("draft:") else title
        return self.client.create_merge_request(source=head, target=base, title=subject, description=body)


def gitlab_forge(client: GitLab | None = None, *, cwd: Path | str = ".", **options: Any) -> IssueForge:
    """The forge adapter for this checkout, or over a client already built."""
    return IssueForge(client if client is not None else connect(cwd, **options))


# -- the model-facing tool --------------------------------------------------


class GitLabTool(Tool):
    """Read and write GitLab state for this repository."""

    name = "gitlab"
    description = (
        "Work with the GitLab project this repository belongs to: list and read issues and merge "
        "requests, open a merge request, post a comment, list pipelines and their jobs, and read the "
        "tail of a failed job's log. Requires a GitLab remote and a token with the 'api' scope."
    )
    #: WRITE rather than SAFE because `create_mr` and `comment` are side
    #: effects other people see.  Most actions here only read, but a tool's
    #: danger is the worst thing it can do, not the average.
    danger = Danger.WRITE
    parallel_safe = True
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "issues", "issue", "mrs", "mr", "create_mr", "comment",
                    "pipelines", "jobs", "log", "status",
                ],
                "description": "what to do",
            },
            "iid": {"type": "integer", "minimum": 1, "description": "issue or merge request number"},
            "state": {"type": "string", "description": "opened, closed, merged or all"},
            "kind": {"type": "string", "description": "for comment: 'issue' or 'mr'"},
            "body": {"type": "string", "description": "for comment: the note to post"},
            "title": {"type": "string", "description": "for create_mr"},
            "description": {"type": "string", "description": "for create_mr: the merge request body"},
            "source_branch": {"type": "string", "description": "for create_mr; defaults to the current branch"},
            "target_branch": {"type": "string", "description": "for create_mr; defaults to the default branch"},
            "ref": {"type": "string", "description": "for pipelines: limit to one branch"},
            "pipeline_id": {"type": "integer", "description": "for jobs and log"},
            "job_id": {"type": "integer", "description": "for log: one specific job"},
            "limit": {"type": "integer", "minimum": 1, "description": "how many items to return"},
        },
        "required": ["action"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        bits = [str(args.get("action") or "?")]
        for key in ("iid", "pipeline_id", "job_id", "title"):
            if args.get(key):
                bits.append(f"{key}={args[key]!r}")
        return "gitlab " + " ".join(bits)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = Path(getattr(ctx, "root", None) or ctx.cwd)
        env = dict(ctx.env) if getattr(ctx, "env", None) else None
        client = connect(root, env=env, timeout=ctx.timeout or TIMEOUT)
        if not client.ok:
            return ToolResult.fail(client.problem)
        content, problem = _dispatch(str(args.get("action") or "").strip(), args, client, root)
        if problem:
            # Scrubbed on the way out as well as at the call boundary: this is
            # the last point before the string reaches the transcript.
            return ToolResult.fail(client.scrub(problem))
        return ToolResult.text(client.scrub(content) or "nothing to report")


def _dispatch(action: str, args: dict[str, Any], client: GitLab, root: Path) -> tuple[str, str]:
    """`(content, error)` for one tool action.  Never raises."""
    limit = int(args.get("limit") or LIST_LIMIT)
    state = str(args.get("state") or "opened")

    if action == "status":
        return "\n".join(client.remote.report() + [f"token: {'yes' if client.token else 'no'}"]), ""

    if action == "issues":
        got = client.issues(state=state, limit=max(1, limit))
        if not got.ok:
            return "", got.error or "could not list issues"
        if not got.items:
            return f"no {_state(state)} issues", ""
        head = f"{len(got.items)} {_state(state)} issue(s) over {got.pages} page(s)"
        return "\n".join([head, *(issue_line(i) for i in got.items if isinstance(i, dict))]), ""

    if action == "mrs":
        got = client.merge_requests(state=state, limit=max(1, limit))
        if not got.ok:
            return "", got.error or "could not list merge requests"
        if not got.items:
            return f"no {_state(state)} merge requests", ""
        head = f"{len(got.items)} {_state(state)} merge request(s) over {got.pages} page(s)"
        return "\n".join([head, *(mr_line(m) for m in got.items if isinstance(m, dict))]), ""

    if action in ("issue", "mr"):
        iid = args.get("iid")
        if not isinstance(iid, int):
            return "", f"action {action!r} needs an iid"
        got = client.issue(iid) if action == "issue" else client.merge_request(iid)
        if not got.ok:
            return "", got.error or f"could not read {action} {iid}"
        data = got.data if isinstance(got.data, dict) else {}
        lines = issue_report(data) if action == "issue" else mr_report(data)
        return "\n".join(lines), ""

    if action == "create_mr":
        title = str(args.get("title") or "").strip()
        if not title:
            return "", "create_mr needs a title"
        source = str(args.get("source_branch") or "").strip() or vcs.current_branch(root).name
        target = str(args.get("target_branch") or "").strip() or vcs.default_branch(root).name
        if not source:
            return "", "not on a branch, so there is no source for a merge request"
        if not target:
            return "", "could not work out the target branch; pass target_branch"
        if source == target:
            return "", f"source and target are both {source}; a merge request needs two branches"
        got = client.create_merge_request(
            source=source, target=target, title=title, description=str(args.get("description") or "")
        )
        if not got.ok:
            return "", got.error or "could not open the merge request"
        return f"opened !{got.get('iid')} {got.get('web_url') or title}", ""

    if action == "comment":
        iid = args.get("iid")
        if not isinstance(iid, int):
            return "", "comment needs an iid"
        got = client.comment(str(args.get("kind") or "mr"), iid, str(args.get("body") or ""))
        if not got.ok:
            return "", got.error or "could not post the comment"
        return f"posted note {got.get('id')} on {_collection(str(args.get('kind') or 'mr'))} {iid}", ""

    if action == "pipelines":
        got = client.pipelines(ref=str(args.get("ref") or ""), limit=max(1, limit))
        if not got.ok:
            return "", got.error or "could not list pipelines"
        if not got.items:
            return "no pipelines for this project", ""
        return "\n".join(pipeline_line(p) for p in got.items if isinstance(p, dict)), ""

    if action == "jobs":
        pipeline_id = args.get("pipeline_id")
        if not isinstance(pipeline_id, int):
            return "", "jobs needs a pipeline_id; get one from action=pipelines"
        got = client.jobs(pipeline_id)
        if not got.ok:
            return "", got.error or "could not list the jobs"
        if not got.items:
            return f"pipeline {pipeline_id} has no jobs", ""
        return "\n".join(job_line(j) for j in got.items if isinstance(j, dict)), ""

    if action == "log":
        job_id = args.get("job_id")
        if isinstance(job_id, int):
            got = client.job_log(job_id)
            if not got.ok:
                return "", got.error or "could not fetch the job log"
            text = tail(clean_trace(got.text), LOG_TAIL)
            return text or "the job has no output", ""
        pipeline_id = args.get("pipeline_id")
        if not isinstance(pipeline_id, int):
            return "", "log needs a job_id or a pipeline_id"
        failed = client.failure(pipeline_id)
        if not failed.ok:
            return "", failed.error
        return "\n".join(failed.report()), ""

    return "", f"unknown action {action!r}; one of: issues, issue, mrs, mr, create_mr, comment, pipelines, jobs, log, status"


def gitlab_tools() -> list[Tool]:
    return [GitLabTool()]


# -- the shell surface ------------------------------------------------------


def _client(state: Any) -> tuple[GitLab | None, Any]:
    """The client for this session, or the `Outcome` explaining why not.

    Built here rather than inside the `job` closure on purpose: `connect`
    resolves the environment, and doing that on the worker thread the app runs
    jobs on is the bug `providers/catalogue` had to fix.
    """
    from offset.shell.commands import Outcome

    client = connect(state.workspace)
    if not client.ok:
        return None, Outcome.error(client.problem)
    return client, None


def _mrs(state: Any, args: list[str]) -> Any:
    """`/mrs [state]` - the project's merge requests."""
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    client, problem = _client(state)
    if client is None:
        return problem
    wanted = _state(args[0] if args else "opened")

    def job() -> Outcome:
        got = client.merge_requests(state=wanted)
        if not got.ok:
            return Outcome.error(got.error or "could not list merge requests")
        rows = [mr_line(m) for m in got.items if isinstance(m, dict)]
        if not rows:
            return Outcome([f"no {wanted} merge requests in {client.remote.path}"], TONE_INFO)
        head = f"{len(rows)} {wanted} merge request(s) in {client.remote.path}"
        return Outcome([head, *rows[:LIST_LIMIT]], TONE_OK)

    return Outcome([f"reading {wanted} merge requests..."], TONE_INFO, job=job)


def _issues(state: Any, args: list[str]) -> Any:
    """`/issues [state]`, `/issues <number>` - list or read."""
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    client, problem = _client(state)
    if client is None:
        return problem
    number = next((int(a) for a in args if a.isdigit()), 0)
    wanted = _state(next((a for a in args if not a.isdigit()), "opened"))

    def job() -> Outcome:
        if number:
            got = client.issue(number)
            if not got.ok:
                return Outcome.error(got.error or f"could not read issue {number}")
            data = got.data if isinstance(got.data, dict) else {}
            return Outcome(issue_report(data), TONE_OK)
        listed = client.issues(state=wanted)
        if not listed.ok:
            return Outcome.error(listed.error or "could not list issues")
        rows = [issue_line(i) for i in listed.items if isinstance(i, dict)]
        if not rows:
            return Outcome([f"no {wanted} issues in {client.remote.path}"], TONE_INFO)
        head = f"{len(rows)} {wanted} issue(s) in {client.remote.path}"
        return Outcome([head, *rows[:LIST_LIMIT]], TONE_OK)

    target = f"issue {number}" if number else f"{wanted} issues"
    return Outcome([f"reading {target}..."], TONE_INFO, job=job)


def _pipeline(state: Any, args: list[str]) -> Any:
    """`/pipeline [ref|id]` - the latest pipelines, and why the last one broke."""
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    client, problem = _client(state)
    if client is None:
        return problem
    wanted_id = next((int(a) for a in args if a.isdigit()), 0)
    ref = next((a for a in args if not a.isdigit()), "")

    def job() -> Outcome:
        pipeline_id = wanted_id
        rows: list[str] = []
        if not pipeline_id:
            listed = client.pipelines(ref=ref or vcs.current_branch(state.workspace).name, limit=8)
            if not listed.ok:
                return Outcome.error(listed.error or "could not list pipelines")
            found = [p for p in listed.items if isinstance(p, dict)]
            if not found:
                where = ref or "this branch"
                return Outcome([f"no pipelines for {where} in {client.remote.path}"], TONE_INFO)
            rows = [pipeline_line(p) for p in found]
            pipeline_id = int(found[0].get("id") or 0)
            if str(found[0].get("status")) not in FAILED_STATES:
                return Outcome(rows, TONE_OK)
        failed = client.failure(pipeline_id)
        if not failed.ok:
            return Outcome([*rows, failed.error], TONE_INFO)
        return Outcome([*rows, "", *failed.report()], TONE_OK)

    return Outcome(["reading pipelines..."], TONE_INFO, job=job)


def _mr(state: Any, args: list[str]) -> Any:
    """`/mr [title]` - push the current branch and open a merge request.

    A dirty tree refuses, for the same reason `/pr` does: a merge request that
    does not contain the work the user is looking at is worse than an error.
    The title comes from the arguments, or from the branch's own commits, so
    this command needs no model and no key to work.
    """
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    client, problem = _client(state)
    if client is None:
        return problem

    condition = vcs.status(state.workspace)
    if condition.dirty:
        return Outcome.error(
            "the working tree has uncommitted changes",
            "commit them first; a merge request would not contain them",
        )
    branch = vcs.current_branch(state.workspace)
    base = vcs.default_branch(state.workspace)
    if not branch.name:
        return Outcome.error("not on a branch")
    if not base.name:
        return Outcome.error("could not work out the default branch to merge into")
    if branch.name == base.name:
        return Outcome.error(
            f"you are on {base.name}, the default branch",
            "make a branch for the change first",
        )

    def job() -> Outcome:
        title = " ".join(args).strip()
        description = ""
        if not title:
            history = vcs.log(state.workspace, since=base.name, limit=20)
            commits = list(history.commits)
            if not commits:
                return Outcome.error(f"no commits on {branch.name} that {base.name} does not have")
            title = commits[0].subject or branch.name
            if len(commits) > 1:
                description = "\n".join(f"- {c.subject}" for c in commits)

        pushed = vcs.push(state.workspace, branch.name, set_upstream=True)
        if not pushed.ok:
            return Outcome.error(pushed.error or "could not push the branch")
        got = client.create_merge_request(
            source=branch.name, target=base.name, title=title, description=description
        )
        if not got.ok:
            return Outcome.error(got.error or "could not open the merge request")
        return Outcome([f"opened !{got.get('iid')}: {got.get('web_url') or title}"], TONE_OK)

    return Outcome([f"opening a merge request for {branch.name} -> {base.name}..."], TONE_INFO, job=job)


def gitlab_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command("mr", "open a merge request from this branch", _mr, usage="/mr [title]"),
        Command("mrs", "the project's merge requests", _mrs, usage="/mrs [opened|merged|closed|all]"),
        # No `/issue` alias on purpose: `/issues <number>` already reads one,
        # and the singular is the obvious name for an issue-to-MR workflow to
        # claim, so leaving it free costs nothing and avoids a clash.
        Command("issues", "the project's issues", _issues, usage="/issues [state] | /issues <number>"),
        Command("pipeline", "the latest pipelines and the failing job's log", _pipeline,
                usage="/pipeline [ref|id]", aliases=("ci",)),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """`COMMANDS` on demand.

    Built lazily because the handlers import from `offset.shell.commands`,
    which imports the tool subsystems: resolving at import time would be a
    cycle.  The re-check after building guards the re-entrant case, where
    importing the shell registry comes back through here before the outer call
    has stored anything and every command would be registered twice.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = gitlab_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
