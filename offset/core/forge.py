"""GitHub, reached whichever way actually works on this machine.

The client has two paths and tries them in a fixed order, because the ordering
is the whole design decision:

  1. **`gh`, the official CLI, when it is installed and authenticated.**  It
     already holds the user's credential — often an OAuth token in a keyring
     we have no business reading — it knows about GitHub Enterprise hosts, and
     it refreshes tokens on its own.  Shelling out to `gh api` costs a fork and
     removes every authentication question from this file.
  2. **Plain REST over `urllib`.**  For machines without `gh`: a token from
     `$GITHUB_TOKEN`/`$GH_TOKEN`, or failing that from `gh`'s own
     `hosts.yml`, read by a *deliberately narrow* line scanner — see
     `read_hosts_token`, which is not a YAML parser and does not pretend to be.

Both paths speak one method, `Backend.request(method, path, body, accept)`, so
every operation below is written once.  GraphQL is the same shape: POST to
`graphql` with a query in the body, which is what makes resolving a review
thread possible at all (REST cannot).

**Failure is a value, and the value has to be actionable.**  Three of GitHub's
answers are famously misleading and each gets its own message:

  * `403` with `X-RateLimit-Remaining: 0` is a rate limit, not a permission
    problem, and the reset time is the only useful thing to say.
  * `404` on a repository that exists means "your credential cannot see this" —
    GitHub deliberately hides private repositories behind 404 so that probing
    cannot enumerate them. The message must name both possibilities and say
    which is likely, depending on whether a credential was sent at all.
  * no credential at all is not "unauthorised", it is a missing setting, and
    the message names the setting.

CI logs get their own section.  A GitHub Actions job log is routinely tens of
megabytes of `##[group]`-wrapped noise; feeding it to a model is both useless
and expensive, so `extract_failure` finds the failing step and returns the
region around the error instead of the log.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from offset.core import vcs

#: github.com's own API base.  GitHub Enterprise lives at
#: `https://<host>/api/v3`, and `$GITHUB_API_URL` is GitHub's own convention
#: for overriding it (Actions sets it), which is also how the tests point this
#: client at a local server.
PUBLIC_API: Final = "https://api.github.com"

#: Environment variables the token is read from, in order.  `gh` itself
#: honours both names and so must we.
TOKEN_KEYS: Final = ("GITHUB_TOKEN", "GH_TOKEN")

API_VERSION: Final = "2022-11-28"
USER_AGENT: Final = "offset"

TIMEOUT: Final = 30.0

#: `gh api` reports the HTTP status in its stderr message rather than its exit
#: code, which is 1 for everything.
_GH_STATUS = re.compile(r"\(HTTP (\d{3})\)")


# -- one answer -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Reply:
    """One forge call's answer, or the reason there is none."""

    ok: bool
    status: int = 0
    data: Any = None
    text: str = ""
    error: str | None = None
    via: str = ""

    @classmethod
    def fail(cls, error: str, *, status: int = 0, via: str = "", text: str = "") -> "Reply":
        return cls(False, status, None, text, error, via)

    @classmethod
    def good(cls, status: int, data: Any, text: str, *, via: str = "") -> "Reply":
        return cls(True, status, data, text, None, via)

    @property
    def items(self) -> list[Any]:
        """The payload as a list, whatever shape GitHub chose to send."""
        if isinstance(self.data, list):
            return list(self.data)
        if isinstance(self.data, dict):
            for key in ("items", "check_runs", "workflow_runs", "jobs"):
                got = self.data.get(key)
                if isinstance(got, list):
                    return list(got)
        return []

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default) if isinstance(self.data, dict) else default


# -- credentials ------------------------------------------------------------


def gh_config_dir(env: dict[str, str] | None = None) -> Path:
    """Where `gh` keeps `hosts.yml`, following gh's own precedence."""
    where = env if env is not None else os.environ
    if where.get("GH_CONFIG_DIR"):
        return Path(where["GH_CONFIG_DIR"])
    if where.get("XDG_CONFIG_HOME"):
        return Path(where["XDG_CONFIG_HOME"]) / "gh"
    return Path.home() / ".config" / "gh"


def read_hosts_token(path: Path, host: str = "github.com") -> str | None:
    """Pull one `oauth_token` out of `gh`'s `hosts.yml`.

    This is a targeted line scanner, **not** a YAML parser, and it is one on
    purpose: adding a YAML dependency to read a four-line file is not a trade
    worth making.  It understands exactly the structure `gh` writes — a
    top-level host key at column zero, `oauth_token: <value>` somewhere beneath
    it — and returns None for anything else rather than guessing.  Quoted
    values and `users:` sub-blocks are handled because gh writes both.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    current = ""
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if not raw[0].isspace():
            current = raw.split(":", 1)[0].strip().strip("\"'")
            continue
        if current != host:
            continue
        key, sep, value = raw.strip().partition(":")
        if sep and key.strip() == "oauth_token":
            token = value.strip().strip("\"'")
            if token:
                return token
    return None


def find_token(host: str = "github.com", *, env: dict[str, str] | None = None) -> str | None:
    """A token for `host`, from the environment first, then gh's config."""
    where = env if env is not None else os.environ
    for key in TOKEN_KEYS:
        value = (where.get(key) or "").strip()
        if value:
            return value
    return read_hosts_token(gh_config_dir(where) / "hosts.yml", host)


def api_base(host: str = "github.com", *, env: dict[str, str] | None = None) -> str:
    """The REST base URL for `host`, honouring `$GITHUB_API_URL`."""
    where = env if env is not None else os.environ
    override = (where.get("GITHUB_API_URL") or "").strip().rstrip("/")
    if override:
        return override
    if host in ("", "github.com"):
        return PUBLIC_API
    return f"https://{host}/api/v3"


# -- transports -------------------------------------------------------------


class Backend:
    """One way of reaching GitHub.  Subclasses never raise; they return `Reply`."""

    name = "backend"

    __slots__ = ()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> Reply:
        raise NotImplementedError  # pragma: no cover - abstract by convention

    @staticmethod
    def _query(path: str, params: dict[str, Any] | None) -> str:
        if not params:
            return path
        pairs = [(k, str(v)) for k, v in params.items() if v is not None]
        if not pairs:
            return path
        joiner = "&" if "?" in path else "?"
        return f"{path}{joiner}{urllib.parse.urlencode(pairs)}"


def _decode(text: str) -> Any:
    """JSON when it is JSON.  A log endpoint answers plain text."""
    stripped = text.lstrip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        return None


def _message(payload: Any, fallback: str) -> str:
    """GitHub's own error sentence, plus its validation detail when present."""
    if not isinstance(payload, dict):
        return fallback
    said = str(payload.get("message") or "").strip()
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        detail = []
        for item in errors:
            if isinstance(item, dict):
                detail.append(str(item.get("message") or f"{item.get('field')}: {item.get('code')}"))
            else:
                detail.append(str(item))
        said = f"{said} ({'; '.join(detail[:3])})" if said else "; ".join(detail[:3])
    return said or fallback


def _reset_in(headers: dict[str, str]) -> str:
    """How long until the rate limit resets, in words."""
    raw = headers.get("x-ratelimit-reset") or ""
    try:
        seconds = int(float(raw)) - int(time.time())
    except (TypeError, ValueError):
        return "an unknown time"
    if seconds <= 0:
        return "now"
    if seconds < 120:
        return f"{seconds}s"
    return f"{seconds // 60}m"


def diagnose(status: int, headers: dict[str, str], payload: Any, text: str, *, path: str, authenticated: bool) -> str:
    """Turn an HTTP failure into a sentence that says what to do next."""
    lowered = {k.lower(): v for k, v in headers.items()}
    said = _message(payload, text.strip()[:200] or f"http {status}")
    if status == 401:
        return (
            f"github rejected the credential (401): {said}. "
            "the token is expired or malformed - set GITHUB_TOKEN to a fresh token with 'repo' scope, "
            "or run 'gh auth login'"
        )
    if status == 403 and lowered.get("x-ratelimit-remaining") == "0":
        limit = lowered.get("x-ratelimit-limit") or "?"
        extra = "" if authenticated else " authenticating raises the limit from 60 to 5000 per hour;"
        return (
            f"github rate limit exhausted ({limit}/hour), resets in {_reset_in(lowered)}."
            f"{extra} wait or set GITHUB_TOKEN to a different token"
        )
    if status == 403:
        return f"github refused (403): {said}. the credential is valid but lacks the scope for {path}"
    if status == 404:
        if not authenticated:
            return (
                f"github returned 404 for {path} with no credential sent. "
                "github answers 404 rather than 401 for private repositories, so this is most likely "
                "a missing token - set GITHUB_TOKEN - and only then a wrong path"
            )
        return (
            f"github returned 404 for {path}. the credential was accepted elsewhere, so either the path is "
            "wrong or the token cannot see this repository - a private repository needs 'repo' scope"
        )
    if status == 422:
        return f"github rejected the request (422): {said}"
    return f"github returned {status} for {path}: {said}"


class Rest(Backend):
    """Direct REST/GraphQL over `urllib`.  Used when `gh` is unavailable."""

    name = "rest"

    __slots__ = ("base", "timeout", "token")

    def __init__(self, token: str | None, *, base: str = PUBLIC_API, timeout: float = TIMEOUT) -> None:
        self.token = (token or "").strip() or None
        self.base = base.rstrip("/")
        self.timeout = timeout

    def url_for(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.base}/{path.lstrip('/')}"

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> Reply:
        target = self.url_for(self._query(path, params))
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(target, data=payload, method=method.upper())
        request.add_header("Accept", accept or "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", API_VERSION)
        request.add_header("User-Agent", USER_AGENT)
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as answer:
                raw = answer.read().decode("utf-8", errors="replace")
                return Reply.good(answer.status, _decode(raw), raw, via=self.name)
        except urllib.error.HTTPError as exc:
            raw = ""
            try:
                raw = exc.read().decode("utf-8", errors="replace")
            except Exception:  # a body we cannot read must not mask the status
                raw = ""
            headers = {k: v for k, v in (exc.headers or {}).items()}
            reason = diagnose(
                exc.code, headers, _decode(raw), raw,
                path=path, authenticated=bool(self.token),
            )
            return Reply.fail(reason, status=exc.code, via=self.name, text=raw)
        except urllib.error.URLError as exc:
            return Reply.fail(f"could not reach {self.base}: {exc.reason}", via=self.name)
        except (TimeoutError, OSError) as exc:
            return Reply.fail(f"could not reach {self.base}: {type(exc).__name__}: {exc}", via=self.name)


class GhCli(Backend):
    """`gh api`, which already holds the user's credential."""

    name = "gh"

    __slots__ = ("exe", "timeout")

    def __init__(self, exe: str = "gh", *, timeout: float = TIMEOUT) -> None:
        self.exe = exe
        self.timeout = timeout

    def authenticated(self) -> bool:
        """`gh auth token` is offline and instant, so it is the readiness probe.

        `gh auth status` reaches the network and would make starting a command
        cost a round trip; whether a token exists is all we need to know.
        """
        done = self._exec(["auth", "token"], None)
        return done is not None and done.returncode == 0 and bool(done.stdout.strip())

    def _exec(self, args: Sequence[str], stdin: str | None) -> subprocess.CompletedProcess | None:
        try:
            return subprocess.run(
                [self.exe, *args],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                errors="replace",
                input=stdin,
                env={**os.environ, "GH_PAGER": "", "CLICOLOR": "0", "NO_COLOR": "1"},
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> Reply:
        target = self._query(path, params).lstrip("/")
        args = ["api", target, "--method", method.upper()]
        if accept:
            args += ["-H", f"Accept: {accept}"]
        args += ["-H", f"X-GitHub-Api-Version: {API_VERSION}"]
        stdin = None
        if body is not None:
            # `--input -` sends the body verbatim, which keeps nested JSON
            # (a GraphQL variables object) intact where `-f key=value` would
            # flatten it into strings.
            args += ["--input", "-"]
            stdin = json.dumps(body)
        done = self._exec(args, stdin)
        if done is None:
            return Reply.fail(
                f"could not run {self.exe!r}; install the github cli or set GITHUB_TOKEN "
                "so the rest client can be used instead",
                via=self.name,
            )
        if done.returncode == 0:
            return Reply.good(200, _decode(done.stdout), done.stdout, via=self.name)

        noise = (done.stderr or done.stdout).strip()
        found = _GH_STATUS.search(noise)
        status = int(found.group(1)) if found else 0
        if status:
            payload = _decode(done.stdout) or _decode(noise)
            # gh only ever surfaces a status, so a rate limit has to be read
            # out of its own wording rather than a header.
            headers = {"x-ratelimit-remaining": "0"} if "rate limit" in noise.lower() else {}
            reason = diagnose(status, headers, payload, noise, path=target, authenticated=True)
            return Reply.fail(reason, status=status, via=self.name, text=noise)
        if "auth login" in noise or "not logged" in noise.lower():
            return Reply.fail(
                "the github cli is installed but not authenticated. run 'gh auth login', "
                "or set GITHUB_TOKEN and offset will use rest directly",
                via=self.name,
                text=noise,
            )
        return Reply.fail(f"gh api failed: {noise.splitlines()[0] if noise else 'no output'}", via=self.name, text=noise)


def choose_backend(
    host: str = "github.com",
    *,
    env: dict[str, str] | None = None,
    gh: str | None = None,
    prefer: str | None = None,
) -> tuple[Backend | None, str | None]:
    """The best available transport, or a message naming what to set.

    `prefer` (`$OFFSET_GITHUB_BACKEND`, `"gh"` or `"rest"`) exists so a user
    whose `gh` is authenticated as the wrong account can force the other path.
    """
    where = env if env is not None else os.environ
    wanted = (prefer or where.get("OFFSET_GITHUB_BACKEND") or "").strip().lower()
    token = find_token(host, env=where)
    found = gh or shutil.which("gh")

    if wanted == "rest" or (wanted != "gh" and where.get("GITHUB_API_URL") and token):
        # A pinned API base means somebody is deliberately talking to a
        # specific server; gh would ignore it and go somewhere else.
        if token:
            return Rest(token, base=api_base(host, env=where)), None
    if wanted != "rest" and found:
        cli = GhCli(found)
        if cli.authenticated():
            return cli, None
        if token:
            return Rest(token, base=api_base(host, env=where)), None
        return None, (
            "the github cli is installed but not authenticated, and no token is set. "
            "run 'gh auth login', or export GITHUB_TOKEN=<token with repo scope>"
        )
    if token:
        return Rest(token, base=api_base(host, env=where)), None
    return None, (
        "no github credential found. export GITHUB_TOKEN (a personal access token with "
        "'repo' scope) or install the github cli and run 'gh auth login'"
    )


# -- the repository ---------------------------------------------------------


#: GraphQL is the only way to see, or change, whether a review thread is
#: resolved; REST exposes the comments but not the thread they belong to.
THREADS_QUERY: Final = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) {
        nodes {
          id
          isResolved
          isOutdated
          path
          line
          comments(first: 20) {
            nodes { databaseId author { login } body path line }
          }
        }
      }
    }
  }
}
"""

RESOLVE_MUTATION: Final = """
mutation($id: ID!) {
  resolveReviewThread(input: {threadId: $id}) {
    thread { id isResolved }
  }
}
"""


@dataclass(frozen=True, slots=True)
class Thread:
    """One review conversation, with its resolution state."""

    id: str
    path: str = ""
    line: int | None = None
    resolved: bool = False
    outdated: bool = False
    comments: tuple[tuple[str, str], ...] = ()  # (author, body)

    @property
    def opener(self) -> str:
        return self.comments[0][1] if self.comments else ""

    def report(self) -> list[str]:
        where = f"{self.path}:{self.line}" if self.line else (self.path or "the pull request")
        mark = "resolved" if self.resolved else ("outdated" if self.outdated else "open")
        out = [f"{where} ({mark})"]
        out += [f"  {who}: {body.strip().splitlines()[0][:120] if body.strip() else '(empty)'}"
                for who, body in self.comments]
        return out


@dataclass(slots=True)
class Forge:
    """One GitHub repository, reachable or not.

    Built either from a checkout (`connect`) or directly with an owner, a repo
    and a backend, which is what makes it testable against a local server.
    An unreachable forge is still a usable object: every operation returns the
    same `error` instead of the caller having to check for None.
    """

    owner: str = ""
    repo: str = ""
    backend: Backend | None = None
    host: str = "github.com"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.backend is not None and self.error is None and bool(self.owner and self.repo)

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def via(self) -> str:
        return self.backend.name if self.backend else "nothing"

    def call(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        accept: str | None = None,
    ) -> Reply:
        """Any endpoint, with `{repo}` standing in for `repos/owner/name`."""
        if not self.ok:
            return Reply.fail(self.error or "no github repository to talk to")
        assert self.backend is not None  # `ok` guarantees it; keeps type checkers quiet
        resolved = path.replace("{repo}", f"repos/{self.owner}/{self.repo}")
        return self.backend.request(method, resolved, body=body, params=params, accept=accept)

    def graphql(self, query: str, variables: dict[str, Any]) -> Reply:
        answer = self.call("POST", "graphql", body={"query": query, "variables": variables})
        if not answer.ok:
            return answer
        errors = answer.get("errors")
        if isinstance(errors, list) and errors:
            said = "; ".join(str(e.get("message", e)) for e in errors[:3] if isinstance(e, (dict, str)))
            return Reply.fail(f"github graphql refused the query: {said}", status=answer.status, via=answer.via)
        return answer

    # -- pull requests ----------------------------------------------------

    def create_pr(
        self,
        *,
        title: str,
        head: str,
        base: str,
        body: str = "",
        draft: bool = False,
    ) -> Reply:
        if not title.strip():
            return Reply.fail("a pull request needs a title")
        return self.call("POST", "{repo}/pulls", body={
            "title": title.strip(), "head": head, "base": base, "body": body, "draft": bool(draft),
        })

    def get_pr(self, number: int) -> Reply:
        return self.call("GET", f"{{repo}}/pulls/{int(number)}")

    def pr_diff(self, number: int) -> Reply:
        """The unified diff GitHub itself renders, as text."""
        return self.call("GET", f"{{repo}}/pulls/{int(number)}", accept="application/vnd.github.v3.diff")

    def list_files(self, number: int, *, per_page: int = 100) -> Reply:
        return self.call("GET", f"{{repo}}/pulls/{int(number)}/files", params={"per_page": per_page})

    def find_pr(self, branch: str, *, state: str = "open") -> Reply:
        """The pull request opened from `branch`, if there is one."""
        if not branch.strip():
            return Reply.fail("no branch to look up a pull request for")
        return self.call("GET", "{repo}/pulls", params={
            "head": f"{self.owner.split('/')[0]}:{branch}", "state": state, "per_page": 10,
        })

    # -- comments ---------------------------------------------------------

    def list_pr_comments(self, number: int, *, per_page: int = 100) -> Reply:
        """Conversation comments — the ones not attached to a line of code."""
        return self.call("GET", f"{{repo}}/issues/{int(number)}/comments", params={"per_page": per_page})

    def list_review_comments(self, number: int, *, per_page: int = 100) -> Reply:
        """Inline review comments, each carrying its file and line."""
        return self.call("GET", f"{{repo}}/pulls/{int(number)}/comments", params={"per_page": per_page})

    def reply_to_comment(self, number: int, comment_id: int, body: str) -> Reply:
        if not body.strip():
            return Reply.fail("a reply needs a body")
        return self.call(
            "POST", f"{{repo}}/pulls/{int(number)}/comments/{int(comment_id)}/replies",
            body={"body": body},
        )

    def post_review(
        self,
        number: int,
        body: str,
        *,
        comments: Sequence[dict[str, Any]] = (),
        event: str = "COMMENT",
    ) -> Reply:
        payload: dict[str, Any] = {"body": body, "event": event}
        if comments:
            payload["comments"] = list(comments)
        return self.call("POST", f"{{repo}}/pulls/{int(number)}/reviews", body=payload)

    def list_review_threads(self, number: int) -> tuple[list[Thread], str | None]:
        """Every review thread with its resolution state, via GraphQL."""
        answer = self.graphql(THREADS_QUERY, {"owner": self.owner, "name": self.repo, "number": int(number)})
        if not answer.ok:
            return [], answer.error
        repository = (answer.get("data") or {}).get("repository") if isinstance(answer.data, dict) else None
        pull = (repository or {}).get("pullRequest") if isinstance(repository, dict) else None
        nodes = ((pull or {}).get("reviewThreads") or {}).get("nodes") if isinstance(pull, dict) else None
        if not isinstance(nodes, list):
            return [], f"github answered without review threads for pull request {number}"
        threads: list[Thread] = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            said = ((node.get("comments") or {}).get("nodes")) or []
            comments = tuple(
                (str(((c.get("author") or {}).get("login")) or "someone"), str(c.get("body") or ""))
                for c in said if isinstance(c, dict)
            )
            threads.append(Thread(
                id=str(node.get("id") or ""),
                path=str(node.get("path") or ""),
                line=node.get("line") if isinstance(node.get("line"), int) else None,
                resolved=bool(node.get("isResolved")),
                outdated=bool(node.get("isOutdated")),
                comments=comments,
            ))
        return threads, None

    def resolve_thread(self, thread_id: str) -> Reply:
        if not thread_id.strip():
            return Reply.fail("a thread needs an id to be resolved")
        return self.graphql(RESOLVE_MUTATION, {"id": thread_id})

    # -- checks -----------------------------------------------------------

    def list_checks(self, ref: str, *, per_page: int = 100) -> Reply:
        """Check runs for a commit, branch or tag."""
        if not ref.strip():
            return Reply.fail("checks need a ref to look at")
        return self.call(
            "GET", f"{{repo}}/commits/{urllib.parse.quote(ref, safe='')}/check-runs",
            params={"per_page": per_page},
        )

    def get_check_logs(self, job_id: int) -> Reply:
        """The plain-text log of one Actions job.  Can be enormous — clip it
        with `extract_failure` before it reaches a model."""
        return self.call("GET", f"{{repo}}/actions/jobs/{int(job_id)}/logs", accept="application/vnd.github.raw")


def connect(
    root: Path | str,
    *,
    remote: str = vcs.DEFAULT_REMOTE,
    env: dict[str, str] | None = None,
    gh: str | None = None,
    backend: Backend | None = None,
) -> Forge:
    """A `Forge` for the checkout at `root`.  Never raises."""
    if not vcs.is_repo(root):
        return Forge(error=f"{root} is not a git repository, so there is no pull request to work with")
    found = vcs.origin(root, remote)
    if not found.ok:
        return Forge(error=found.error or f"could not read the {remote!r} remote url")
    if not found.github:
        return Forge(
            owner=found.owner, repo=found.repo, host=found.host,
            error=f"{found.host} is not a github host; this workflow only speaks github",
        )
    if backend is not None:
        return Forge(found.owner, found.repo, backend, found.host)
    chosen, why = choose_backend(found.host, env=env, gh=gh)
    if chosen is None:
        return Forge(owner=found.owner, repo=found.repo, host=found.host, error=why)
    return Forge(found.owner, found.repo, chosen, found.host)


# -- CI log extraction ------------------------------------------------------


#: An Actions log line starts with an ISO timestamp and a space.  Stripping it
#: roughly halves the bytes and costs nothing in meaning.
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s?")

#: Ordered by how strongly each marker indicates the actual failure. The first
#: tier that matches wins, so a `##[error]` annotation beats a stray line
#: containing the word "error" in a dependency's output.
_TIERS: Final = (
    ("##[error]",),
    ("Traceback (most recent call last)", "panic: ", "Segmentation fault"),
    ("FAILED ", "FAIL ", "assertion failed", "AssertionError", "error TS", "npm ERR!",
     "error[E", "fatal error", "Fatal error", "FATAL:"),
    ("Error: ", "error: ", "ERROR:", "failed with exit code", "Process completed with exit code"),
)

_GROUP = re.compile(r"^##\[(?:group|section)\](.*)$")


@dataclass(slots=True)
class Excerpt:
    """The part of a CI log that explains the failure."""

    step: str = ""
    lines: list[str] = field(default_factory=list)
    matched: str = ""
    total: int = 0
    kept: int = 0
    reason: str = ""

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def report(self) -> list[str]:
        head = f"{self.kept} of {self.total} log lines"
        if self.step:
            head += f" from step {self.step!r}"
        if self.reason:
            head += f" ({self.reason})"
        return [head, *self.lines]


def extract_failure(
    log: str,
    *,
    before: int = 40,
    after: int = 20,
    max_lines: int = 160,
    max_chars: int = 12000,
) -> Excerpt:
    """Find the error region in a CI log and return only that.

    A job log is routinely tens of megabytes: dependency resolution, cache
    restores, and one interesting screenful.  Sending the whole thing to a
    model wastes the context window it needs for the fix, so this walks the log
    once, finds the *last* line matching the strongest tier of error markers
    present (the last one, because a build that retries fails again at the end),
    and returns a window around it plus the name of the `##[group]` it sits in.

    When no marker matches at all the tail is returned, clearly labelled — a
    silent empty result would look like a passing job.
    """
    raw = log.splitlines()
    if not raw:
        return Excerpt(total=0, kept=0, reason="the log was empty")
    stripped = [_STAMP.sub("", line).rstrip() for line in raw]

    hit = -1
    marker = ""
    for tier in _TIERS:
        for index in range(len(stripped) - 1, -1, -1):
            line = stripped[index]
            found = next((m for m in tier if m in line), "")
            if found:
                hit, marker = index, found
                break
        if hit >= 0:
            break

    if hit < 0:
        window = stripped[-min(max_lines, len(stripped)):]
        return _clip(Excerpt(
            step=_group_of(stripped, len(stripped) - 1),
            lines=window,
            total=len(stripped),
            kept=len(window),
            reason="no error marker found; showing the tail",
        ), max_chars)

    start = max(0, hit - max(0, before))
    end = min(len(stripped), hit + max(0, after) + 1)
    if end - start > max_lines:
        # Keep the error itself and what led to it; the trailing lines are
        # usually just the runner tidying up.
        start = max(0, end - max_lines)
    window = [line for line in stripped[start:end]]
    return _clip(Excerpt(
        step=_group_of(stripped, hit),
        lines=window,
        matched=marker,
        total=len(stripped),
        kept=len(window),
        reason=f"matched {marker!r} at line {hit + 1}",
    ), max_chars)


def _group_of(lines: Sequence[str], index: int) -> str:
    """The `##[group]` heading the line at `index` sits under."""
    for back in range(min(index, len(lines) - 1), -1, -1):
        found = _GROUP.match(lines[back])
        if found:
            return found.group(1).strip()
    return ""


def _clip(excerpt: Excerpt, max_chars: int) -> Excerpt:
    """Enforce the character budget from the front, keeping the error."""
    if max_chars <= 0:
        return excerpt
    while excerpt.lines and len("\n".join(excerpt.lines)) > max_chars:
        excerpt.lines.pop(0)
        excerpt.kept -= 1
        if not excerpt.reason.endswith("clipped"):
            excerpt.reason = f"{excerpt.reason}, clipped" if excerpt.reason else "clipped"
    return excerpt


# -- rendering --------------------------------------------------------------


def check_line(check: dict[str, Any]) -> str:
    name = str(check.get("name") or "check")
    status = str(check.get("status") or "")
    result = str(check.get("conclusion") or status or "pending")
    return f"{result:<12} {name}"


def failing_checks(reply: Reply) -> list[dict[str, Any]]:
    """The check runs a human would call broken, newest first per name."""
    bad = {"failure", "timed_out", "action_required", "cancelled", "startup_failure"}
    out = [c for c in reply.items if isinstance(c, dict) and str(c.get("conclusion") or "") in bad]
    out.sort(key=lambda c: (str(c.get("completed_at") or ""), str(c.get("name") or "")), reverse=True)
    return out


def job_id_of(check: dict[str, Any]) -> int | None:
    """A check run's id doubles as the Actions job id for GitHub Actions runs.

    Other providers (CircleCI, Buildkite) have no job log endpoint here at all,
    which is why this can return None and callers must handle it.
    """
    app = (check.get("app") or {}) if isinstance(check.get("app"), dict) else {}
    if str(app.get("slug") or "").lower() not in ("github-actions", ""):
        return None
    ident = check.get("id")
    return int(ident) if isinstance(ident, int) else None
