"""Talking to git, one subprocess at a time.

There is no git library here and there is not going to be one.  `git` is
already installed wherever this agent runs, it is the only implementation that
is bug-compatible with the user's own working copy, and shelling out to it
costs a fork rather than a dependency.  `offset.core.speculate` established the
pattern for worktrees; this module is the general vocabulary — branch, status,
diff, merge base, log, commit, push, remote — that the GitHub surface and
anything else needing repository facts can share instead of re-deriving.

Two rules shape the whole file.

  * **Nothing raises into a turn.**  Every call returns a record carrying its
    own `error`, so a missing git, a timeout, a detached HEAD and a repository
    that is not a repository at all are all just values the caller renders.
    `git` writing to stderr is normal, not exceptional.
  * **Nothing prints.**  Renderers are `report()` and return `list[str]`.

The one piece of real parsing is remote URLs, because git has two dialects for
the same thing — `git@host:owner/repo.git` (scp-like, no scheme) and
`https://host/owner/repo` — and a workflow that only understands one of them
breaks for half its users.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

#: Long enough for a big `git diff` on a cold cache, short enough that a
#: hanging credential prompt does not wedge the turn.  `GIT_TERMINAL_PROMPT=0`
#: below is what actually stops the prompt.
TIMEOUT: Final = 60.0

DEFAULT_REMOTE: Final = "origin"

#: Names to probe when the remote never told us which branch it considers
#: default.  Order is the order they are tried.
TRUNKS: Final = ("main", "master", "trunk", "develop")

#: Exit codes we invent for failures git never got to report.
MISSING: Final = 127
TIMED_OUT: Final = 124

#: `scp`-like remote: `git@github.com:owner/repo.git`.  No scheme, and the
#: colon separates host from path rather than naming a port.
_SCP = re.compile(r"^(?:(?P<user>[^@/]+)@)?(?P<host>[^:/@]+):(?P<path>[^:].*)$")


# -- one invocation ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run:
    """What one `git` invocation did.  `error` is None exactly when it worked."""

    args: tuple[str, ...]
    code: int
    out: str = ""
    err: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def text(self) -> str:
        return self.out.strip()

    @property
    def error(self) -> str | None:
        if self.ok:
            return None
        # git's own wording is kept verbatim — rewriting it loses the path or
        # ref the user needs to see — but the frame around it stays lowercase.
        detail = (self.err.strip() or self.out.strip()).splitlines()
        first = detail[0].strip() if detail else f"exited {self.code}"
        return f"git {self.args[0] if self.args else 'command'} failed: {first}"

    def lines(self) -> list[str]:
        return [line for line in self.out.splitlines() if line.strip()]


def run(args: Sequence[str], cwd: Path | str, *, timeout: float = TIMEOUT, stdin: str | None = None) -> Run:
    """Run `git <args>` in `cwd`.  Never raises; failures come back as `Run`."""
    argv = ["git", *args]
    try:
        done = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
            input=stdin,
            # A push that decides to ask for a password would otherwise hang
            # the agent until the timeout with nobody watching the tty.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"},
        )
    except FileNotFoundError:
        return Run(tuple(args), MISSING, err="git is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        return Run(tuple(args), TIMED_OUT, err=f"timed out after {timeout:.0f}s")
    except OSError as exc:
        return Run(tuple(args), MISSING, err=f"{type(exc).__name__}: {exc}")
    return Run(tuple(args), done.returncode, done.stdout or "", done.stderr or "")


def is_repo(cwd: Path | str) -> bool:
    got = run(["rev-parse", "--is-inside-work-tree"], cwd)
    return got.ok and got.text == "true"


def toplevel(cwd: Path | str) -> Path | None:
    """The working tree root, or None when `cwd` is not in a repository."""
    got = run(["rev-parse", "--show-toplevel"], cwd)
    return Path(got.text) if got.ok and got.text else None


# -- refs and branches ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ref:
    """A resolved object name."""

    sha: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.sha)

    @property
    def short(self) -> str:
        return self.sha[:12]


@dataclass(frozen=True, slots=True)
class Branch:
    """A branch name, or the reason there is not one."""

    name: str = ""
    detached: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.name)


def head(cwd: Path | str) -> Ref:
    got = run(["rev-parse", "HEAD"], cwd)
    return Ref(got.text) if got.ok else Ref(error=got.error)


def current_branch(cwd: Path | str) -> Branch:
    """The checked-out branch.  A detached HEAD is reported, not guessed at."""
    got = run(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd)
    if got.ok and got.text:
        return Branch(got.text)
    # `symbolic-ref` exits 1 on a detached HEAD with nothing on stderr, which
    # is a state rather than a failure, so it must not read as one.
    where = run(["rev-parse", "--short", "HEAD"], cwd)
    if where.ok and where.text:
        return Branch(where.text, detached=True)
    return Branch(error=where.error or "no commits yet, so there is no branch")


def branch_exists(cwd: Path | str, name: str) -> bool:
    return run(["rev-parse", "--verify", "--quiet", f"refs/heads/{name}"], cwd).ok


def default_branch(cwd: Path | str, *, remote: str = DEFAULT_REMOTE) -> Branch:
    """Which branch this repository treats as trunk, decided without a fetch.

    `refs/remotes/<remote>/HEAD` is the authoritative answer when the clone has
    one; a worktree created by `git init` or a shallow CI checkout often does
    not, so known trunk names are probed on the remote and then locally.
    """
    pointer = run(["symbolic-ref", "--quiet", "--short", f"refs/remotes/{remote}/HEAD"], cwd)
    if pointer.ok and pointer.text:
        name = pointer.text.split("/", 1)[1] if "/" in pointer.text else pointer.text
        if name:
            return Branch(name)
    for candidate in TRUNKS:
        if run(["rev-parse", "--verify", "--quiet", f"refs/remotes/{remote}/{candidate}"], cwd).ok:
            return Branch(candidate)
    for candidate in TRUNKS:
        if branch_exists(cwd, candidate):
            return Branch(candidate)
    return Branch(error=f"could not tell which branch is the default; looked for {', '.join(TRUNKS)}")


def merge_base(cwd: Path | str, left: str, right: str = "HEAD") -> Ref:
    """The commit `left` and `right` diverged from."""
    got = run(["merge-base", left, right], cwd)
    if got.ok and got.text:
        return Ref(got.text.split()[0])
    return Ref(error=got.error or f"no common ancestor between {left} and {right}")


# -- working tree state -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class Status:
    """Porcelain status, split by where the change lives."""

    staged: tuple[str, ...] = ()
    unstaged: tuple[str, ...] = ()
    untracked: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def dirty(self) -> bool:
        """Tracked changes only.  An untracked scratch file is not a reason to
        refuse work, and treating it as one made `/pr` unusable in a repo with
        a stray `notes.md`."""
        return bool(self.staged or self.unstaged)

    @property
    def paths(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for group in (self.staged, self.unstaged, self.untracked):
            for path in group:
                seen[path] = None
        return tuple(seen)

    def report(self) -> list[str]:
        if self.error:
            return [self.error]
        if not self.paths:
            return ["clean"]
        out: list[str] = []
        for label, group in (("staged", self.staged), ("unstaged", self.unstaged), ("untracked", self.untracked)):
            if group:
                out.append(f"{label:<10} {len(group)}: " + ", ".join(group[:6]) + (" ..." if len(group) > 6 else ""))
        return out


def status(cwd: Path | str) -> Status:
    got = run(["status", "--porcelain"], cwd)
    if not got.ok:
        return Status(error=got.error)
    staged: list[str] = []
    unstaged: list[str] = []
    untracked: list[str] = []
    for line in got.out.splitlines():
        if len(line) < 4:
            continue
        index, tree, path = line[0], line[1], line[3:].strip()
        # A rename is reported as `old -> new`; the new name is the one that
        # exists on disk and the only one worth showing.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        if index == "?" and tree == "?":
            untracked.append(path)
            continue
        if index not in " ?":
            staged.append(path)
        if tree not in " ?":
            unstaged.append(path)
    return Status(tuple(staged), tuple(unstaged), tuple(untracked))


# -- diffs and history ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Diff:
    """A unified diff, with what it was taken against."""

    text: str = ""
    against: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    @property
    def added(self) -> int:
        return sum(1 for line in self.text.splitlines() if line.startswith("+") and not line.startswith("+++"))

    @property
    def removed(self) -> int:
        return sum(1 for line in self.text.splitlines() if line.startswith("-") and not line.startswith("---"))

    def clip(self, limit: int) -> str:
        """The diff, truncated at a line boundary, saying so when it was cut."""
        if limit <= 0 or len(self.text) <= limit:
            return self.text
        cut = self.text[:limit]
        edge = cut.rfind("\n")
        if edge > 0:
            cut = cut[:edge]
        dropped = len(self.text) - len(cut)
        return f"{cut}\n... diff truncated, {dropped} more characters"

    def report(self) -> list[str]:
        if self.error:
            return [self.error]
        if self.empty:
            return [f"no changes against {self.against or 'the index'}"]
        return [f"{self.added} added, {self.removed} removed against {self.against or 'the index'}"]


def diff(
    cwd: Path | str,
    *,
    base: str | None = None,
    staged: bool = False,
    paths: Sequence[str] = (),
    context: int = 3,
) -> Diff:
    """The diff a reviewer would want.

    `base` uses three-dot range syntax (`base...HEAD`) deliberately: that is
    what a pull request shows — this branch's own work — whereas `base..HEAD`
    also contains everything that landed on the base since the fork point.
    """
    args = ["diff", f"--unified={max(0, context)}", "--no-color"]
    against = "the working tree"
    if base:
        args.append(f"{base}...HEAD")
        against = base
    elif staged:
        args.append("--cached")
        against = "the index"
    if paths:
        args += ["--", *paths]
    got = run(args, cwd)
    if not got.ok:
        return Diff(against=against, error=got.error)
    return Diff(got.out, against)


@dataclass(frozen=True, slots=True)
class Commit:
    sha: str
    subject: str = ""
    author: str = ""
    date: str = ""

    def line(self) -> str:
        who = f" ({self.author})" if self.author else ""
        return f"{self.sha[:8]} {self.subject}{who}"


@dataclass(frozen=True, slots=True)
class Log:
    commits: tuple[Commit, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def report(self) -> list[str]:
        if self.error:
            return [self.error]
        return [c.line() for c in self.commits] or ["no commits in that range"]

    def subjects(self) -> list[str]:
        return [c.subject for c in self.commits if c.subject]


#: Unit separator between fields, record separator between commits: a commit
#: subject can contain anything at all, including tabs and pipes.
_FIELD = "\x1f"
_RECORD = "\x1e"


def log(cwd: Path | str, since: str | None = None, until: str = "HEAD", *, limit: int = 50) -> Log:
    """Commits reachable from `until` but not `since`, newest first."""
    span = f"{since}..{until}" if since else until
    got = run(["log", f"--max-count={max(1, limit)}", f"--format=%H{_FIELD}%s{_FIELD}%an{_FIELD}%aI{_RECORD}", span], cwd)
    if not got.ok:
        return Log(error=got.error)
    commits: list[Commit] = []
    for record in got.out.split(_RECORD):
        record = record.strip("\n")
        if not record.strip():
            continue
        parts = record.split(_FIELD)
        if len(parts) < 4:
            continue
        commits.append(Commit(parts[0], parts[1], parts[2], parts[3]))
    return Log(tuple(commits))


@dataclass(frozen=True, slots=True)
class Files:
    paths: tuple[str, ...] = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def report(self) -> list[str]:
        if self.error:
            return [self.error]
        return list(self.paths) or ["no files changed"]


def changed_files(cwd: Path | str, *, base: str | None = None, staged: bool = False) -> Files:
    args = ["diff", "--name-only"]
    if base:
        args.append(f"{base}...HEAD")
    elif staged:
        args.append("--cached")
    got = run(args, cwd)
    if not got.ok:
        return Files(error=got.error)
    return Files(tuple(dict.fromkeys(line.strip() for line in got.out.splitlines() if line.strip())))


# -- changing things --------------------------------------------------------


def create_branch(cwd: Path | str, name: str, *, start: str | None = None, switch: bool = True) -> Run:
    """Create `name` (and check it out unless told otherwise)."""
    if not name.strip():
        return Run(("branch",), 1, err="a branch needs a name")
    if switch:
        args = ["switch", "--create", name] + ([start] if start else [])
    else:
        args = ["branch", name] + ([start] if start else [])
    return run(args, cwd)


def commit(cwd: Path | str, message: str, *, paths: Sequence[str] = (), stage_all: bool = False) -> Run:
    """Stage what was asked for, then commit.  An empty commit is refused."""
    if not message.strip():
        return Run(("commit",), 1, err="a commit needs a message")
    if paths:
        staged = run(["add", "--", *paths], cwd)
        if not staged.ok:
            return staged
    elif stage_all:
        staged = run(["add", "--all"], cwd)
        if not staged.ok:
            return staged
    return run(["commit", "--message", message], cwd)


def push(
    cwd: Path | str,
    *,
    remote: str = DEFAULT_REMOTE,
    branch: str | None = None,
    upstream: bool = True,
    force: bool = False,
    timeout: float = TIMEOUT,
) -> Run:
    """Push `branch` (default: the current one) to `remote`."""
    if branch is None:
        here = current_branch(cwd)
        if not here.ok or here.detached:
            return Run(("push",), 1, err=here.error or "cannot push a detached HEAD; check out a branch first")
        branch = here.name
    args = ["push"]
    if upstream:
        args.append("--set-upstream")
    if force:
        # Never a plain --force: refusing to clobber someone else's push is
        # the entire value of the lease.
        args.append("--force-with-lease")
    args += [remote, branch]
    return run(args, cwd, timeout=timeout)


# -- remotes ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Remote:
    """A remote URL, split into the three parts a forge API needs."""

    host: str = ""
    owner: str = ""
    repo: str = ""
    url: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.host and self.owner and self.repo)

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def parts(self) -> tuple[str, str, str]:
        return (self.host, self.owner, self.repo)

    @property
    def github(self) -> bool:
        return self.host == "github.com" or self.host.startswith("github.")

    def report(self) -> list[str]:
        if self.error:
            return [self.error]
        return [f"{self.slug} on {self.host}"]


def parse_remote(url: str) -> Remote:
    """`(host, owner, repo)` from either git URL dialect.

    Handles `git@host:owner/repo.git`, `ssh://git@host:22/owner/repo`,
    `https://user@host/owner/repo`, a bare `host/owner/repo`, and any of them
    with or without `.git` and a trailing slash.  Anything left over is an
    error value rather than a guess.
    """
    raw = (url or "").strip()
    if not raw:
        return Remote(error="no remote url to parse")

    host = ""
    path = ""
    if "://" in raw:
        scheme, rest = raw.split("://", 1)
        if scheme.lower() not in ("ssh", "git", "http", "https", "git+ssh"):
            return Remote(url=raw, error=f"unsupported remote scheme {scheme!r}: {raw}")
        authority, _, path = rest.partition("/")
        host = authority.rpartition("@")[2]
    else:
        found = _SCP.match(raw)
        if found:
            host, path = found.group("host"), found.group("path")
        elif "/" in raw:
            # `github.com/owner/repo`, which people do paste.
            host, _, path = raw.partition("/")
        else:
            return Remote(url=raw, error=f"unrecognised remote url: {raw}")

    # A port is not part of the host as far as an API base is concerned.
    host = host.split(":", 1)[0].lower().strip()
    segments = [s for s in path.strip("/").split("/") if s]
    if segments:
        segments[-1] = segments[-1].removesuffix(".git")
    if not host or len(segments) < 2 or not segments[-1]:
        return Remote(host=host, url=raw, error=f"unrecognised remote url: {raw}")
    # Everything before the repository is the owner, so GitLab-style subgroups
    # survive the round trip even though GitHub never has them.
    return Remote(host=host, owner="/".join(segments[:-1]), repo=segments[-1], url=raw)


def remote_url(cwd: Path | str, remote: str = DEFAULT_REMOTE) -> Run:
    """The configured URL for `remote`, as a `Run` so the failure is legible."""
    got = run(["remote", "get-url", remote], cwd)
    if got.ok and not got.text:
        return Run(got.args, 1, err=f"remote {remote!r} has no url configured")
    return got


def origin(cwd: Path | str, remote: str = DEFAULT_REMOTE) -> Remote:
    """Parse the repository's remote.  One call for the common case."""
    got = remote_url(cwd, remote)
    if not got.ok:
        listed = run(["remote"], cwd)
        known = ", ".join(listed.lines()) if listed.ok and listed.lines() else "none"
        return Remote(error=f"no remote named {remote!r}. configured remotes: {known}")
    return parse_remote(got.text)
