"""Self-update: find out whether a newer offset exists, then install it.

Four decisions shape this module.

The first is that finding out must cost the user nothing.  A coding agent is
started dozens of times a day, and an agent that pauses on a socket before it
draws a prompt is an agent people stop starting.  So the startup path is a
daemon thread that answers to nobody: it writes what it learned to a cache file
and dies, and every failure — no route to the host, DNS gone, GitHub down, a
body that is not JSON — resolves to silence.  There is nothing to report
because an update check is not something the user asked for.

The second is that the answer is remembered with a timestamp and only fetched
again after a day.  Without that, the network cost above is paid on every
launch, which is the same mistake in a slower disguise.

The third is that the comparison is real.  `importlib.metadata` hands out
version *strings* and no comparator; comparing those as text gets `0.10.0` and
`0.9.0` the wrong way round and calls `1.2.0rc1` newer than `1.2.0`.  So this
module carries a PEP 440 parser — epoch, release tuple, pre/post/dev segments,
local segment, ordered the way the specification says.  Anything it cannot
parse is not compared at all, because a wrong "update available" nag is worse
than no nag.

The fourth is that installing is done by whatever installed offset in the first
place, so the method is detected rather than assumed.  A source checkout is
refused outright: running `pip install --upgrade` over an editable install is
how somebody's working tree gets replaced by a release tarball.

Every outward-facing call takes its network access, its clock and its
subprocess runner as an argument, which is why the tests for this file never
open a socket.
"""

from __future__ import annotations

import json
import os
import re
import site
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Final, Sequence

from offset import __version__ as SOURCE_VERSION
from offset.core import settings

if TYPE_CHECKING:  # only for annotations: see `update_commands` for why
    from offset.shell.commands import Command, Outcome, ShellState

#: The distribution name, in `importlib.metadata` and on PyPI.
PACKAGE: Final = "offset"

#: Where releases are published.
REPO: Final = "The-Masked-Bear/offset-terminal"
GITHUB_URL: Final = f"https://api.github.com/repos/{REPO}/releases/latest"
PYPI_URL: Final = f"https://pypi.org/pypi/{PACKAGE}/json"

#: Seconds between successful checks.  A day; releases are not hourly events.
INTERVAL: Final = 86_400.0

#: Seconds before retrying a *failed* check.  Shorter than `INTERVAL` because
#: the usual cause is a laptop that was off the network for a minute, not a
#: repository that stopped existing.
RETRY_INTERVAL: Final = 3_600.0

#: Seconds one request may take.  Short: nobody is waiting for this, but a
#: hung socket still holds a thread.
TIMEOUT: Final = 6.0

USER_AGENT: Final = f"offset/{SOURCE_VERSION} (+https://github.com/{REPO})"

#: Set this to anything truthy and no check happens at all.
NO_CHECK_ENV: Final = "OFFSET_NO_UPDATE_CHECK"

#: The settings key with the same effect.
SETTING: Final = "update.check"

#: Auto-update switches, separate from the check switches: wanting to be told
#: about a release and wanting it installed unattended are different appetites,
#: and somebody who declines the second should still get the first.
NO_AUTO_ENV: Final = "OFFSET_NO_AUTO_UPDATE"
AUTO_SETTING: Final = "update.auto"

#: Set on the child after an auto-update re-executes, so a build that somehow
#: still reports itself stale cannot re-exec forever.  A loop here would be
#: unkillable from the terminal it is looping in.
REEXEC_ENV: Final = "OFFSET_UPDATED_REEXEC"

#: Bumped when the cache layout changes, so an old file is ignored rather than
#: misread.
CACHE_VERSION: Final = 1

_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})


# -- PEP 440 ----------------------------------------------------------------
#
# A trimmed version of the specification's own regex: enough for anything a
# release feed will ever carry, and it refuses what it does not understand.

_PATTERN: Final = re.compile(
    r"""^\s*v?
        (?:(?P<epoch>[0-9]+)!)?
        (?P<release>[0-9]+(?:\.[0-9]+)*)
        (?:[-_.]?(?P<pre_l>alpha|beta|preview|pre|a|b|c|rc)[-_.]?(?P<pre_n>[0-9]+)?)?
        (?P<post>-(?P<post_n1>[0-9]+)|[-_.]?(?:post|rev|r)[-_.]?(?P<post_n2>[0-9]+)?)?
        (?P<dev>[-_.]?dev[-_.]?(?P<dev_n>[0-9]+)?)?
        (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?
        \s*$""",
    re.VERBOSE | re.IGNORECASE,
)

#: Spellings the specification treats as the same marker.
_PRE_LETTERS: Final = {
    "a": "a", "alpha": "a",
    "b": "b", "beta": "b",
    "c": "rc", "pre": "rc", "preview": "rc", "rc": "rc",
}

_PRE_RANK: Final = {"a": 0, "b": 1, "rc": 2}

#: A `.devN` with no pre-release marker sorts below every marker; a version
#: with no marker at all sorts above them.  These are the two sentinels that
#: make `1.0.dev1 < 1.0a1 < 1.0rc1 < 1.0` come out in that order.
_DEV_ONLY: Final = (-1, 0, 0)
_FINAL: Final = (1, 0, 0)

_INF: Final = float("inf")


def _local_key(local: str) -> tuple[Any, ...]:
    """Order local segments the way PEP 440 does: numeric parts compare as
    numbers and outrank alphabetic ones, and having no local part at all is
    lower than having one."""
    if not local:
        return (0,)
    parts = tuple(
        (int(part), "") if part.isdigit() else (-1, part)
        for part in re.split(r"[-_.]", local.lower())
    )
    return (1, parts)


@dataclass(frozen=True, slots=True, eq=False)
class Version:
    """A parsed version, ordered by `key` rather than by field.

    Comparison is defined by hand because the dataclass default would compare
    `release` tuples, and `1.0` has to equal `1.0.0`.
    """

    text: str
    epoch: int = 0
    release: tuple[int, ...] = ()
    pre: tuple[str, int] | None = None
    post: int | None = None
    dev: int | None = None
    local: str = ""

    @property
    def key(self) -> tuple[Any, ...]:
        release = self.release
        while len(release) > 1 and release[-1] == 0:
            release = release[:-1]
        if self.pre is not None:
            marker = (0, _PRE_RANK[self.pre[0]], self.pre[1])
        elif self.dev is not None and self.post is None:
            marker = _DEV_ONLY
        else:
            marker = _FINAL
        return (
            self.epoch,
            release,
            marker,
            -1 if self.post is None else self.post,
            _INF if self.dev is None else self.dev,
            _local_key(self.local),
        )

    @property
    def prerelease(self) -> bool:
        return self.pre is not None or self.dev is not None

    def __str__(self) -> str:
        return self.text

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Version) and self.key == other.key

    def __lt__(self, other: "Version") -> bool:
        return self.key < other.key

    def __le__(self, other: "Version") -> bool:
        return self.key <= other.key

    def __gt__(self, other: "Version") -> bool:
        return self.key > other.key

    def __ge__(self, other: "Version") -> bool:
        return self.key >= other.key


def parse_version(text: str) -> Version | None:
    """A `Version`, or None when `text` is not something this can order.

    None rather than an exception: an unrecognised tag on a release feed has to
    end as silence, not as a traceback out of a background thread.
    """
    match = _PATTERN.match(text or "")
    if match is None:
        return None
    letter = match.group("pre_l")
    pre = None
    if letter is not None:
        pre = (_PRE_LETTERS[letter.lower()], int(match.group("pre_n") or 0))
    post = None
    if match.group("post") is not None:
        post = int(match.group("post_n1") or match.group("post_n2") or 0)
    dev = None
    if match.group("dev") is not None:
        dev = int(match.group("dev_n") or 0)
    return Version(
        text=(text or "").strip(),
        epoch=int(match.group("epoch") or 0),
        release=tuple(int(part) for part in match.group("release").split(".")),
        pre=pre,
        post=post,
        dev=dev,
        local=(match.group("local") or ""),
    )


def newer(candidate: str, than: str) -> bool:
    """True only when both sides parse and `candidate` is strictly greater.

    An unparseable side means "do not know", and not knowing must never show
    the user an update that may not exist.
    """
    left, right = parse_version(candidate), parse_version(than)
    return left is not None and right is not None and right < left


# -- what is installed, and by what -----------------------------------------


def installed_version() -> str:
    """The version of the offset that is running.

    The installed metadata wins over the source constant: a wheel built from an
    older tree still reports what was installed, which is what an update check
    is about.  A checkout with no metadata falls back to the constant.
    """
    try:
        return metadata.version(PACKAGE)
    except metadata.PackageNotFoundError:
        return SOURCE_VERSION
    except Exception:  # a broken dist-info must not stop the program starting
        return SOURCE_VERSION


def package_location() -> Path:
    """The directory the `offset` package sits in: site-packages for a real
    install, the repository root for a checkout or an editable install."""
    return Path(__file__).resolve().parents[2]


def _user_site() -> Path | None:
    try:
        return Path(site.getusersitepackages()).resolve()
    except (AttributeError, OSError, TypeError):
        return None


def _under(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _venv_python(where: Path) -> str:
    """The interpreter belonging to the environment `where` lives in.

    Walking up for `pyvenv.cfg` is what makes a pipx upgrade checkable: the
    version that changes is the one in pipx's own venv, never this process's.
    """
    for directory in (where, *where.parents):
        if (directory / "pyvenv.cfg").is_file():
            binary = directory / ("Scripts" if os.name == "nt" else "bin")
            candidate = binary / ("python.exe" if os.name == "nt" else "python")
            if candidate.exists():
                return str(candidate)
    return sys.executable


@dataclass(frozen=True, slots=True)
class Install:
    """How offset got onto the machine, and the one command that may update it.

    An empty `command` is not an oversight: it means this install must not be
    upgraded from inside offset, and `reason` says what to do instead.
    """

    method: str  # pipx | pip | pip-user | editable | unknown
    location: Path
    command: tuple[str, ...] = ()
    python: str = ""
    reason: str = ""

    @property
    def upgradable(self) -> bool:
        return bool(self.command)

    def report(self) -> list[str]:
        lines = [f"installed by {self.method} at {self.location}"]
        if self.command:
            lines.append("upgrade command: " + " ".join(self.command))
        if self.reason:
            lines.append(self.reason)
        return lines


def detect_install(
    location: Path | str | None = None,
    *,
    executable: str | None = None,
    user_site: Path | str | None = None,
) -> Install:
    """Work out which installer owns this copy of offset.

    Order matters.  A pipx venv contains a perfectly ordinary site-packages, so
    it has to be recognised first; a user-site directory is also called
    site-packages, so it has to be recognised before the plain case.  The
    checkout test comes before both because an editable install's path points
    at the working tree, and that is the one case where updating is refused.
    """
    where = Path(location).resolve() if location is not None else package_location()
    lowered = {part.lower() for part in where.parts}
    python = executable or _venv_python(where)

    if "pipx" in lowered and "venvs" in lowered:
        # pipx owns that venv; pip inside it would leave the shims behind.
        return Install("pipx", where, ("pipx", "upgrade", PACKAGE), python)

    if (where / "pyproject.toml").is_file() or (where / ".git").is_dir():
        return Install(
            "editable", where, (), python,
            f"offset runs from a source checkout at {where}; "
            "update it with 'git pull' there — pip would overwrite your working tree",
        )

    base = Path(user_site).resolve() if user_site is not None else _user_site()
    if base is not None and _under(where, base):
        return Install(
            "pip-user", where,
            (python, "-m", "pip", "install", "--upgrade", "--user", PACKAGE),
            python,
        )

    if where.name in ("site-packages", "dist-packages"):
        return Install(
            "pip", where,
            (python, "-m", "pip", "install", "--upgrade", PACKAGE),
            python,
        )

    return Install(
        "unknown", where, (), python,
        f"could not tell how offset was installed ({where}); "
        "update it the way you installed it",
    )


# -- the opt-out ------------------------------------------------------------


def enabled() -> bool:
    """Whether a check may happen at all.

    Two switches, because the two audiences differ: an environment variable for
    a CI image or a shell profile, a settings key for a person.
    """
    if (os.environ.get(NO_CHECK_ENV) or "").strip().lower() in _TRUTHY:
        return False
    return _setting()


def _setting() -> bool:
    """`update.check`, read without complaining about itself."""
    if SETTING in settings.BY_KEY and settings.get(SETTING, True) is False:
        return False
    return _flag_from_files(SETTING)


def _flag_from_files(dotted: str, default: bool = True) -> bool:
    """A boolean settings key, read straight from the two config files.

    The settings schema lives in a module this one does not own.  Until a key
    is declared there, `settings.get` files a "read of unknown setting"
    complaint that the user then sees in /settings — so the files are read
    directly instead, project layer last because it wins.

    Both spellings are accepted: a nested `{"update": {"auto": false}}` and a
    flat `{"update.auto": false}`, because a user editing JSON by hand will
    reasonably write either.
    """
    # `Settings._home` is captured when that object is built, so reading
    # `active.user_file` would point at whatever $OFFSET_HOME said at import
    # time.  The house rule is that home is resolved on every call.
    section_name, _, leaf = dotted.partition(".")
    value: Any = default
    for path in (settings.home() / "config.json", settings.active().project_file):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        section = raw.get(section_name)
        if isinstance(section, dict) and leaf in section:
            value = section[leaf]
        elif dotted in raw:
            value = raw[dotted]
    return bool(value) if isinstance(value, bool) else default


# -- the cache --------------------------------------------------------------


def cache_file() -> Path:
    """`$OFFSET_HOME/update.json`.  Resolved late; the tests move home."""
    return settings.home() / "update.json"


@dataclass(frozen=True, slots=True)
class UpdateInfo:
    """What the last look at the release feed found."""

    current: str
    latest: str = ""
    notes: str = ""
    url: str = ""
    published: str = ""
    checked_at: float = 0.0
    #: Answered from `update.json` without touching the network.
    cached: bool = False
    #: The user turned checks off, so nothing was looked at.
    disabled: bool = False
    error: str | None = None

    @property
    def available(self) -> bool:
        return newer(self.latest, self.current)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "current": self.current,
            "latest": self.latest,
            "checked_at": self.checked_at,
        }
        for name in ("notes", "url", "published"):
            value = getattr(self, name)
            if value:
                out[name] = value
        if self.error:
            out["error"] = self.error
        return out

    @classmethod
    def from_json(cls, raw: dict[str, Any], *, cached: bool = False) -> "UpdateInfo":
        return cls(
            current=str(raw.get("current") or ""),
            latest=str(raw.get("latest") or ""),
            notes=str(raw.get("notes") or ""),
            url=str(raw.get("url") or ""),
            published=str(raw.get("published") or ""),
            checked_at=float(raw.get("checked_at") or 0.0),
            cached=cached,
            error=str(raw["error"]) if raw.get("error") else None,
        )

    def report(self) -> list[str]:
        if self.disabled:
            return [f"offset {self.current}",
                    f"update checks are off ({NO_CHECK_ENV} or the {SETTING} setting)"]
        if self.error:
            return [f"offset {self.current}", f"could not check for updates: {self.error}"]
        if not self.latest:
            return [f"offset {self.current}", "no published release to compare against"]
        if not self.available:
            return [f"offset {self.current}", f"up to date; the latest release is {self.latest}"]
        lines = [f"offset {self.current} -> {self.latest} is available"]
        if self.published:
            lines.append(f"published {self.published}")
        if self.url:
            lines.append(self.url)
        if self.notes:
            lines += ["", *self.notes.strip().splitlines()[:12]]
        return lines


def _read_cache() -> dict[str, Any]:
    try:
        raw = json.loads(cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return {}
    return raw


def _write_cache(info: UpdateInfo) -> None:
    """Replace `update.json` atomically, and shrug if that is impossible: a
    cache that cannot be written costs one network call, nothing more."""
    path = cache_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".update-", suffix=".json")
    except OSError:
        return
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump({"version": CACHE_VERSION, **info.to_json()}, out, indent=1)
        os.replace(tmp, path)
    except OSError:
        Path(tmp).unlink(missing_ok=True)


def _fresh(current: str, stamp: float) -> UpdateInfo | None:
    """The previous answer, if it is still worth reusing.

    The stored `current` counts as much as the timestamp: after an upgrade the
    old answer would go on announcing an update that has already been applied.
    """
    raw = _read_cache()
    if not raw or raw.get("current") != current:
        return None
    checked = raw.get("checked_at")
    if not isinstance(checked, (int, float)) or isinstance(checked, bool):
        return None
    age = stamp - float(checked)
    window = RETRY_INTERVAL if raw.get("error") else INTERVAL
    if age < 0.0 or age >= window:
        # A negative age is a clock that moved backwards, not a fresh cache.
        return None
    try:
        return UpdateInfo.from_json(raw, cached=True)
    except (TypeError, ValueError):
        return None


# -- fetching ---------------------------------------------------------------

#: Fetches a URL and returns the decoded JSON body.  Injected everywhere, which
#: is what keeps the tests — and an offline user — off the network.
Fetcher = Callable[[str], Any]


def http_json(url: str, *, timeout: float = TIMEOUT) -> Any:
    """GET and decode.  Raises; the callers turn that into a message."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(1_000_000)
    return json.loads(body.decode("utf-8", "replace"))


@dataclass(frozen=True, slots=True)
class Release:
    """One published release, from whichever feed knew about it."""

    version: str
    notes: str = ""
    url: str = ""
    published: str = ""


def _from_github(payload: Any) -> Release | None:
    if not isinstance(payload, dict) or payload.get("draft") or payload.get("prerelease"):
        return None
    tag = str(payload.get("tag_name") or payload.get("name") or "").strip()
    if parse_version(tag) is None:
        return None
    return Release(
        version=tag.lstrip("vV"),
        notes=str(payload.get("body") or ""),
        url=str(payload.get("html_url") or ""),
        published=str(payload.get("published_at") or "")[:10],
    )


def _from_pypi(payload: Any) -> Release | None:
    info = payload.get("info") if isinstance(payload, dict) else None
    if not isinstance(info, dict):
        return None
    version = str(info.get("version") or "").strip()
    if parse_version(version) is None:
        return None
    return Release(version=version, url=f"https://pypi.org/project/{PACKAGE}/")


def latest_release(fetch: Fetcher | None = None) -> tuple[Release | None, str]:
    """The newest published release, or None and the reason there is none.

    GitHub first, because that is where the notes are; PyPI second, so a
    renamed repository or a rate-limited API does not silence updates for
    everybody.  Both feeds go through the same injected fetcher.
    """
    get = fetch or http_json
    problems: list[str] = []
    for url, reader in ((GITHUB_URL, _from_github), (PYPI_URL, _from_pypi)):
        host = urllib.parse.urlsplit(url).netloc or url
        try:
            release = reader(get(url))
        except Exception as exc:  # urllib raises a zoo; all of it means "no answer"
            problems.append(f"{host}: {type(exc).__name__}: {exc}")
            continue
        if release is not None:
            return release, ""
        problems.append(f"{host}: no usable release in the response")
    return None, "; ".join(problems)


# -- checking ---------------------------------------------------------------


def check(
    *,
    force: bool = False,
    fetch: Fetcher | None = None,
    now: Callable[[], float] | None = None,
) -> UpdateInfo:
    """Whether a newer offset exists.

    `force` is what the user typing /update means: check the network even if the
    cache is warm and even if the automatic check is switched off.  Without
    `force` this is allowed to answer from `update.json` and usually does.
    """
    clock = now or time.time
    current = installed_version()
    if not force and not enabled():
        return UpdateInfo(current=current, disabled=True)
    stamp = clock()
    if not force:
        cached = _fresh(current, stamp)
        if cached is not None:
            return cached
    release, why = latest_release(fetch)
    if release is None:
        info = UpdateInfo(current=current, checked_at=stamp,
                          error=why or "no release information")
    else:
        info = UpdateInfo(
            current=current, latest=release.version, notes=release.notes,
            url=release.url, published=release.published, checked_at=stamp,
        )
    _write_cache(info)
    return info


def check_async(
    *,
    on_update: Callable[[UpdateInfo], None] | None = None,
    fetch: Fetcher | None = None,
    now: Callable[[], float] | None = None,
    done: threading.Event | None = None,
) -> None:
    """Start the startup check and return immediately.

    Everything here is defensive.  The opt-out is honoured before the thread
    exists, so switching checks off costs not even a thread.  The thread is a
    daemon, so an unreachable host cannot hold up exit.  The body swallows
    every exception, because a traceback printed over the prompt by a check
    nobody asked for is a bug in this file, not news.  `done` exists so a test
    can wait for the thread without sleeping.
    """
    if not enabled():
        if done is not None:
            done.set()
        return

    def work() -> None:
        try:
            info = check(fetch=fetch, now=now)
            if info.available and on_update is not None:
                on_update(info)
        except Exception:
            pass  # silence is the whole contract; there is nobody to tell
        finally:
            if done is not None:
                done.set()

    threading.Thread(target=work, name="offset-update-check", daemon=True).start()


def auto_enabled() -> bool:
    """Whether an update may be installed without being asked.

    Gated on `enabled()` as well, so switching checks off switches auto-update
    off with it: it would be strange for a program told not to look to install
    something anyway.
    """
    if not enabled():
        return False
    if (os.environ.get(NO_AUTO_ENV) or "").strip().lower() in _TRUTHY:
        return False
    if os.environ.get(REEXEC_ENV):
        return False  # this process IS the update; do not go round again
    if AUTO_SETTING in settings.BY_KEY and settings.get(AUTO_SETTING, True) is False:
        return False
    return _flag_from_files(AUTO_SETTING)


@dataclass(frozen=True, slots=True)
class AutoOutcome:
    """What the startup auto-update did, if anything."""

    acted: bool = False
    before: str = ""
    after: str = ""
    error: str = ""
    skipped: str = ""

    def report(self) -> list[str]:
        if self.acted:
            return [f"offset updated itself: {self.before} -> {self.after}"]
        if self.error:
            return [f"offset could not update itself: {self.error}"]
        return []


def autoupdate(
    *,
    fetch: Fetcher | None = None,
    now: Callable[[], float] | None = None,
    target: Install | None = None,
    runner: Runner | None = None,
    prober: Probe | None = None,
    echo: Callable[[str], None] | None = None,
) -> AutoOutcome:
    """Install a waiting update before the shell loads.  Never raises.

    Three things make this safe enough to do unattended.

    It reads the CACHE, not the network.  The previous run's background check
    left an answer on disk; consulting it costs a file read, so an offline or
    slow start is not paid for at the one moment the user is watching. A fresh
    install with no cache therefore updates on its second launch, not its
    first, which is the right trade.

    It refuses anything it cannot prove it can do: an editable checkout, an
    unknown install method, a disabled switch. `apply` verifies the version
    actually moved rather than trusting the package manager's exit code.

    It does not try to swap code into a running interpreter.  Modules are
    already imported by the time anything here runs, so a mid-session
    replacement would leave half the program old. The caller re-executes.
    """
    if not auto_enabled():
        return AutoOutcome(skipped="auto-update is off")

    where = target or detect_install()
    if not where.upgradable:
        # Not an error: a git checkout is a perfectly good way to run offset,
        # it just is not one this can upgrade.
        return AutoOutcome(skipped=where.reason or f"{where.method} cannot self-upgrade")

    try:
        info = check(fetch=fetch, now=now)
    except Exception as exc:
        return AutoOutcome(error=f"{type(exc).__name__}: {exc}")
    if not info.available:
        return AutoOutcome(skipped="already up to date")

    if echo is not None:
        echo(f"offset {info.latest} is available; updating {info.current} -> {info.latest}")
    try:
        result = apply(info=info, target=where, runner=runner, prober=prober, echo=echo)
    except Exception as exc:
        return AutoOutcome(error=f"{type(exc).__name__}: {exc}")
    if not result.ok:
        return AutoOutcome(error=result.error or "the upgrade did not take")
    return AutoOutcome(acted=True, before=result.before, after=result.after or info.latest)


def reexec() -> None:
    """Replace this process with the freshly installed one.

    `execv` rather than a subprocess so the user keeps one process, one exit
    status and one terminal; the marker in the environment is what stops the
    new image doing this again.  Returns normally if the exec fails, because
    carrying on with the old version beats refusing to start.
    """
    child = dict(os.environ)
    child[REEXEC_ENV] = "1"
    try:
        os.execve(sys.executable, [sys.executable, "-m", "offset", *sys.argv[1:]], child)
    except OSError:
        return


def install(state: "ShellState") -> None:
    """Startup wiring: begin the background check, block nothing."""
    check_async()


# -- applying ---------------------------------------------------------------

#: Runs a command, handing each output line to the sink as it arrives, and
#: returns the exit status.
Runner = Callable[[Sequence[str], Callable[[str], None]], int]

#: Reports the version installed in the environment an upgrade just touched.
Probe = Callable[[Install], str]

_PROBE_CODE: Final = (
    "import importlib.metadata as m\n"
    "try: print(m.version('offset'))\n"
    "except Exception: pass\n"
)


def stream(command: Sequence[str], sink: Callable[[str], None]) -> int:
    """Run `command`, passing each line on as it appears.

    Merged stderr and line buffering are the point: an installer that prints
    nothing for forty seconds is indistinguishable from one that has hung, and
    a user watching a frozen screen kills it.
    """
    environment = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    try:
        proc = subprocess.Popen(
            list(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, bufsize=1, env=environment,
        )
    except OSError as exc:
        # 127 is the shell's "no such command": pipx not on PATH lands here.
        sink(f"{type(exc).__name__}: {exc}")
        return 127
    if proc.stdout is not None:
        with proc.stdout as out:
            for line in out:
                sink(line.rstrip())
    return proc.wait()


def probe(target: Install) -> str:
    """Ask the upgraded environment which version it has now.

    Asking this process would be wrong twice over: its metadata is already
    imported, and for pipx the environment that changed is not this one.
    Returning "" means the question could not be answered, which is reported as
    such rather than guessed at.
    """
    try:
        done = subprocess.run(
            [target.python or sys.executable, "-c", _PROBE_CODE],
            capture_output=True, text=True, timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


@dataclass(slots=True)
class UpdateResult:
    """What running the upgrade did.  `ok` is about the outcome, not the exit
    status: an installer that succeeds and changes nothing has failed."""

    ok: bool
    before: str
    after: str = ""
    method: str = ""
    command: tuple[str, ...] = ()
    output: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def verified(self) -> bool:
        return bool(self.after)

    def report(self) -> list[str]:
        if not self.ok:
            lines = [f"update failed: {self.error}" if self.error else "update failed"]
            if self.command:
                lines.append("ran: " + " ".join(self.command))
            return lines
        if self.after:
            return [f"updated {self.before} -> {self.after} via {self.method}",
                    "restart offset to run the new version"]
        return [f"{self.method} finished without error",
                f"could not confirm the installed version; it was {self.before} before",
                "restart offset and run 'offset update --check'"]


def apply(
    *,
    info: UpdateInfo | None = None,
    target: Install | None = None,
    runner: Runner | None = None,
    prober: Probe | None = None,
    echo: Callable[[str], None] | None = None,
) -> UpdateResult:
    """Run the one upgrade command this install supports, then prove it worked.

    An install method that cannot support an in-place upgrade is refused before
    anything runs, which is the difference between a refusal and a mess.
    """
    where = target or detect_install()
    before = info.current if info is not None else installed_version()
    if not where.upgradable:
        return UpdateResult(False, before, method=where.method,
                            error=where.reason or f"{where.method} installs cannot be upgraded from here")

    lines: list[str] = []

    def sink(line: str) -> None:
        lines.append(line)
        if echo is not None:
            echo(line)

    status = (runner or stream)(where.command, sink)
    if status != 0:
        return UpdateResult(False, before, "", where.method, where.command, lines,
                            f"{where.command[0]} exited {status}")
    after = (prober or probe)(where)
    if after and after == before:
        return UpdateResult(False, before, after, where.method, where.command, lines,
                            f"the upgrade reported success but offset is still {before}")
    return UpdateResult(True, before, after, where.method, where.command, lines)


# -- entry points -----------------------------------------------------------


def update_command(*, check_only: bool = False) -> int:
    """`offset update`, and `offset update --check`.

    0 means the answer can be trusted: up to date, an update reported, or an
    update installed and confirmed.  1 means it could not be done — nothing
    reachable, an installer that failed, or an install this must not touch.
    """
    info = check(force=True)
    for line in info.report():
        print(line)
    if info.error:
        return 1
    if check_only or not info.available:
        return 0

    where = detect_install()
    print("")
    for line in where.report():
        print(line)
    if not where.upgradable:
        return 1

    print("")
    result = apply(info=info, target=where, echo=print)
    print("")
    for line in result.report():
        print(line)
    return 0 if result.ok else 1


def _update(state: "ShellState", args: list[str]) -> "Outcome":
    """/update — check, and on request install.

    Both halves talk to the network or to a package manager, so both run as a
    background `job`; the keypress returns straight away with what is about to
    happen.
    """
    from offset.shell.commands import Outcome, TONE_ERR, TONE_INFO, TONE_OK

    word = args[0].lower() if args else ""
    if word in ("apply", "install", "now"):
        where = detect_install()
        if not where.upgradable:
            return Outcome.error(
                where.reason or f"{where.method} installs cannot be upgraded from here"
            )

        def upgrade() -> Outcome:
            result = apply(target=where)
            # The tail only: an installer's first hundred lines are never the
            # interesting ones, and the transcript is not a log viewer.
            return Outcome([*result.output[-40:], "", *result.report()],
                           TONE_OK if result.ok else TONE_ERR)

        return Outcome([f"running: {' '.join(where.command)}",
                        "this takes a moment; the output appears when it finishes"],
                       TONE_INFO, job=upgrade)

    if word:
        return Outcome.error(f"unknown argument {word!r}", "usage: /update [apply]")

    def look() -> Outcome:
        info = check(force=True)
        lines = info.report()
        if info.available:
            lines += ["", "run /update apply to install it"]
        tone = TONE_ERR if info.error else (TONE_OK if info.available else TONE_INFO)
        return Outcome(lines, tone)

    return Outcome([f"looking for something newer than offset {installed_version()}"],
                   TONE_INFO, job=look)


def update_commands() -> list["Command"]:
    """The slash commands, built on demand.

    `offset.shell.commands` imports half of `offset.core`, so importing it at
    the top of this file would make wiring /update into that registry a circular
    import.  Building the list inside a function moves the import to the moment
    the shell asks for it, by which time both modules exist.
    """
    from offset.shell.commands import Command

    return [
        Command("update", "check for a newer offset, and install it", _update,
                usage="/update [apply]"),
    ]


_COMMANDS: list["Command"] = []


def __getattr__(name: str) -> Any:
    """`COMMANDS` as a module attribute, without the import cycle.

    PEP 562: the shell can write `from offset.core.update import COMMANDS` and
    get the same list every time, while `import offset.core.update` on its own
    stays free of any dependency on the shell layer.

    The second guard is not redundant.  `update_commands()` imports
    `offset.shell.commands`, and that module's body asks this one for its
    COMMANDS in order to register them - so a first access re-enters here
    before the outer call has filled the list, and both copies then extended
    it.  Re-checking after the call is what makes the list built once rather
    than once per level of re-entry.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = update_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
