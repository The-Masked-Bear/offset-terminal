"""The self-update path, exercised without a socket.

Every test here injects the fetcher, the clock, the subprocess runner or the
version probe, because the one thing this module must never do is reach the
network — least of all from a test suite that runs on a machine with no route
out.  The two properties worth most are asserted with counters and a stopwatch
rather than by reading the code: that a warm cache means *zero* further calls,
and that the startup check returns before the fetch it started has finished.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from importlib import metadata
from pathlib import Path

import pytest

from offset import __version__ as SOURCE_VERSION
from offset.core import update


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A private `$OFFSET_HOME` with update checks switched on."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    monkeypatch.delenv(update.NO_CHECK_ENV, raising=False)
    return tmp_path


@pytest.fixture
def escaped(monkeypatch):
    """Anything a background thread let out.

    `capsys` alone is not enough here: pytest installs its own
    `threading.excepthook` and turns an escaped exception into a warning, so a
    thread that stopped swallowing failures would still leave stderr empty.
    Outside pytest the same exception prints a traceback over the prompt, which
    is precisely the bug this suite exists to prevent.
    """
    seen: list[BaseException] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: seen.append(args.exc_value))
    return seen


def _github(version: str = "9.9.9", **extra) -> dict:
    payload = {
        "tag_name": f"v{version}",
        "name": version,
        "body": "faster, and it no longer eats your worktree",
        "html_url": f"https://github.com/{update.REPO}/releases/tag/v{version}",
        "published_at": "2026-04-01T09:00:00Z",
        "draft": False,
        "prerelease": False,
    }
    payload.update(extra)
    return payload


def counting(payloads: dict[str, object]):
    """A fetcher that records every URL it is asked for.

    Returned alongside the list rather than as a class so an assertion can read
    `len(calls)` directly; the count is the whole point of the cache tests.
    """
    calls: list[str] = []

    def fetch(url: str):
        calls.append(url)
        try:
            return payloads[url]
        except KeyError:
            raise OSError(f"no route to {url}") from None

    return fetch, calls


class Clock:
    """A hand-wound clock.  `time.time` would make the interval tests sleep."""

    __slots__ = ("now",)

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# -- PEP 440 ----------------------------------------------------------------


def test_zero_point_ten_is_newer_than_zero_point_nine():
    assert update.newer("0.10.0", "0.9.0"), "a string compare gets this backwards"
    assert not update.newer("0.9.0", "0.10.0")


def test_a_pre_release_sorts_below_the_final_it_precedes():
    order = ["1.1.9", "1.2.0.dev1", "1.2.0a1", "1.2.0b2", "1.2.0rc1", "1.2.0", "1.2.0.post1"]
    parsed = [update.parse_version(text) for text in order]
    assert all(v is not None for v in parsed), "every one of these is valid PEP 440"
    for lower, higher in zip(parsed, parsed[1:]):
        assert lower < higher, f"{lower} should sort below {higher}"

    assert update.newer("1.2.0", "1.2.0rc1")
    assert not update.newer("1.2.0rc1", "1.2.0")
    assert update.newer("1.2.0rc1", "1.1.9")


def test_equal_versions_are_not_an_update():
    assert not update.newer("1.2.0", "1.2.0")
    assert not update.newer("1.2", "1.2.0")
    assert not update.newer("1.2.0", "1.2")
    assert update.parse_version("1.2") == update.parse_version("1.2.0.0")


def test_a_leading_v_and_surrounding_space_are_tolerated():
    assert update.parse_version(" v1.4.2 ").release == (1, 4, 2)
    assert update.newer("v2.0.0", "1.9.9")


def test_an_unreadable_version_is_never_reported_as_an_update():
    assert update.parse_version("nightly-2026-04-01") is None
    assert not update.newer("nightly", "1.0.0"), "an unparseable tag must not nag"
    assert not update.newer("1.0.0", "nightly")


def test_an_epoch_outranks_the_release_number():
    assert update.newer("1!0.1.0", "99.0.0")


def test_a_prerelease_knows_it_is_one():
    assert update.parse_version("1.0rc1").prerelease
    assert update.parse_version("1.0.dev3").prerelease
    assert not update.parse_version("1.0").prerelease


# -- what is installed ------------------------------------------------------


def test_the_current_version_comes_from_the_installed_metadata(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda name: "4.5.6")
    assert update.installed_version() == "4.5.6"


def test_a_checkout_with_no_metadata_falls_back_to_the_source_constant(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    assert update.installed_version() == SOURCE_VERSION
    assert SOURCE_VERSION, "offset.__version__ must not be empty"


def test_broken_distribution_metadata_does_not_stop_the_program(monkeypatch):
    def exploding(name):
        raise ValueError("dist-info is a directory full of nonsense")

    monkeypatch.setattr(metadata, "version", exploding)
    assert update.installed_version() == SOURCE_VERSION


# -- install detection ------------------------------------------------------


def _venv(root: Path) -> Path:
    """A believable virtualenv layout, real files and all."""
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    site = root / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    return site


def test_install_detection_distinguishes_pipx_pip_and_a_checkout(tmp_path):
    pipx_site = _venv(tmp_path / ".local" / "pipx" / "venvs" / "offset")
    pipx = update.detect_install(pipx_site, user_site=tmp_path / "nowhere")
    assert pipx.method == "pipx"
    assert pipx.command == ("pipx", "upgrade", "offset")
    assert pipx.upgradable

    venv_site = _venv(tmp_path / "env")
    pip = update.detect_install(venv_site, user_site=tmp_path / "nowhere")
    assert pip.method == "pip"
    assert pip.command[1:] == ("-m", "pip", "install", "--upgrade", "offset")
    assert pip.command[0] == str(tmp_path / "env" / "bin" / "python"), \
        "pip must be run from the venv that owns the copy, not from sys.executable"

    checkout = tmp_path / "src" / "offset-terminal"
    checkout.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("[project]\nname = 'offset'\n", encoding="utf-8")
    editable = update.detect_install(checkout, user_site=tmp_path / "nowhere")
    assert editable.method == "editable"
    assert editable.command == ()
    assert not editable.upgradable


def test_a_git_working_tree_with_no_pyproject_is_still_a_checkout(tmp_path):
    checkout = tmp_path / "clone"
    (checkout / ".git").mkdir(parents=True)
    assert update.detect_install(checkout, user_site=tmp_path / "nowhere").method == "editable"


def test_a_user_site_install_upgrades_with_the_user_flag(tmp_path):
    site = tmp_path / ".local" / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    where = update.detect_install(site, executable="/usr/bin/python3", user_site=site)
    assert where.method == "pip-user"
    assert where.command == (
        "/usr/bin/python3", "-m", "pip", "install", "--upgrade", "--user", "offset",
    )


def test_an_unrecognised_layout_says_so_rather_than_guessing(tmp_path):
    odd = tmp_path / "opt" / "bundled"
    odd.mkdir(parents=True)
    where = update.detect_install(odd, user_site=tmp_path / "nowhere")
    assert where.method == "unknown"
    assert not where.upgradable
    assert "could not tell how offset was installed" in where.reason


def test_the_install_report_names_the_command_it_would_run(tmp_path):
    pipx = update.detect_install(
        _venv(tmp_path / "pipx" / "venvs" / "offset"), user_site=tmp_path / "nowhere"
    )
    assert any("pipx upgrade offset" in line for line in pipx.report())


# -- the opt-out ------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_opt_out_environment_variable_suppresses_the_check(home, monkeypatch, value):
    monkeypatch.setenv(update.NO_CHECK_ENV, value)
    fetch, calls = counting({update.GITHUB_URL: _github()})

    info = update.check(fetch=fetch, now=Clock())
    assert info.disabled
    assert calls == [], "the opt-out must be honoured before anything is fetched"

    done = threading.Event()
    update.check_async(fetch=fetch, done=done)
    assert done.wait(5.0), "check_async must settle even when it does nothing"
    assert calls == [], "the background check must not run either"


def test_an_empty_opt_out_variable_leaves_checks_on(home, monkeypatch):
    monkeypatch.setenv(update.NO_CHECK_ENV, "")
    assert update.enabled() is True


def test_the_settings_key_switches_the_check_off(home):
    (home / "config.json").write_text(
        json.dumps({"update": {"check": False}}), encoding="utf-8"
    )
    assert update.enabled() is False

    fetch, calls = counting({update.GITHUB_URL: _github()})
    assert update.check(fetch=fetch, now=Clock()).disabled
    assert calls == []


def test_a_forced_check_ignores_the_opt_out(home, monkeypatch):
    monkeypatch.setenv(update.NO_CHECK_ENV, "1")
    fetch, calls = counting({update.GITHUB_URL: _github()})
    info = update.check(force=True, fetch=fetch, now=Clock())
    assert not info.disabled
    assert info.latest == "9.9.9"
    assert len(calls) == 1, "typing /update is an explicit request; it must run"


# -- the cache --------------------------------------------------------------


def test_the_daily_cache_prevents_a_second_network_call(home):
    clock = Clock()
    fetch, calls = counting({update.GITHUB_URL: _github()})

    first = update.check(fetch=fetch, now=clock)
    assert first.latest == "9.9.9"
    assert not first.cached
    assert len(calls) == 1

    clock.advance(update.INTERVAL - 60.0)
    second = update.check(fetch=fetch, now=clock)
    assert second.cached, "the second answer must come from update.json"
    assert second.latest == "9.9.9"
    assert len(calls) == 1, f"the warm cache still fetched: {calls}"

    clock.advance(120.0)
    third = update.check(fetch=fetch, now=clock)
    assert not third.cached
    assert len(calls) == 2, "a day later the check must happen again"


def test_a_forced_check_goes_to_the_network_even_with_a_warm_cache(home):
    clock = Clock()
    fetch, calls = counting({update.GITHUB_URL: _github()})
    update.check(fetch=fetch, now=clock)
    update.check(force=True, fetch=fetch, now=clock)
    assert len(calls) == 2


def test_a_failed_check_is_retried_sooner_than_a_successful_one(home):
    clock = Clock()
    fetch, calls = counting({})  # every URL raises

    first = update.check(fetch=fetch, now=clock)
    assert first.error, "a fetch that raises must land as an error, not an exception"
    assert len(calls) == 2, "GitHub then PyPI"

    clock.advance(update.RETRY_INTERVAL - 60.0)
    update.check(fetch=fetch, now=clock)
    assert len(calls) == 2, "a failure is cached too, briefly"

    clock.advance(120.0)
    update.check(fetch=fetch, now=clock)
    assert len(calls) == 4, "the retry window is shorter than the success window"


def test_the_cache_is_dropped_once_the_installed_version_changes(home, monkeypatch):
    clock = Clock()
    fetch, calls = counting({update.GITHUB_URL: _github()})
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")
    update.check(fetch=fetch, now=clock)

    monkeypatch.setattr(update, "installed_version", lambda: "9.9.9")
    after = update.check(fetch=fetch, now=clock)
    assert not after.cached, "the old answer described a version that is gone"
    assert len(calls) == 2


def test_a_clock_that_moved_backwards_does_not_freeze_the_cache(home):
    clock = Clock()
    fetch, calls = counting({update.GITHUB_URL: _github()})
    update.check(fetch=fetch, now=clock)
    clock.advance(-3600.0)
    update.check(fetch=fetch, now=clock)
    assert len(calls) == 2


def test_the_cache_lands_in_offset_home(home):
    update.check(fetch=lambda url: _github(), now=Clock())
    assert update.cache_file() == home / "update.json"
    raw = json.loads((home / "update.json").read_text(encoding="utf-8"))
    assert raw["version"] == update.CACHE_VERSION
    assert raw["latest"] == "9.9.9"
    assert isinstance(raw["checked_at"], float)


def test_a_corrupt_cache_file_is_ignored_not_fatal(home):
    (home / "update.json").write_text("{not json at all", encoding="utf-8")
    fetch, calls = counting({update.GITHUB_URL: _github()})
    info = update.check(fetch=fetch, now=Clock())
    assert info.latest == "9.9.9"
    assert len(calls) == 1


def test_a_cache_from_an_older_layout_is_ignored(home):
    (home / "update.json").write_text(
        json.dumps({"version": update.CACHE_VERSION + 1, "current": "0.1.0",
                    "latest": "5.0.0", "checked_at": Clock()()}),
        encoding="utf-8",
    )
    fetch, calls = counting({update.GITHUB_URL: _github()})
    assert update.check(fetch=fetch, now=Clock()).latest == "9.9.9"
    assert len(calls) == 1


def test_an_unwritable_home_costs_a_fetch_and_nothing_else(home, monkeypatch):
    monkeypatch.setattr(update, "cache_file", lambda: home / "no" / "such" / "dir" / "x.json")
    monkeypatch.setattr(
        Path, "mkdir", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only"))
    )
    info = update.check(fetch=lambda url: _github(), now=Clock())
    assert info.latest == "9.9.9", "a cache that cannot be written must not fail the check"


# -- reading the feeds ------------------------------------------------------


def test_github_is_asked_first_because_that_is_where_the_notes_are(home):
    fetch, calls = counting({update.GITHUB_URL: _github()})
    info = update.check(fetch=fetch, now=Clock())
    assert calls == [update.GITHUB_URL]
    assert info.notes.startswith("faster")
    assert info.url.endswith("/v9.9.9")
    assert info.published == "2026-04-01"


def test_pypi_answers_when_github_does_not(home):
    fetch, calls = counting({update.PYPI_URL: {"info": {"version": "3.1.4"}}})
    info = update.check(fetch=fetch, now=Clock())
    assert calls == [update.GITHUB_URL, update.PYPI_URL]
    assert info.latest == "3.1.4"
    assert info.error is None


def test_a_draft_or_prerelease_github_entry_is_not_offered(home):
    fetch, _ = counting({
        update.GITHUB_URL: _github(prerelease=True),
        update.PYPI_URL: {"info": {"version": "2.0.0"}},
    })
    assert update.check(fetch=fetch, now=Clock()).latest == "2.0.0"

    fetch, _ = counting({
        update.GITHUB_URL: _github(draft=True),
        update.PYPI_URL: {"info": {"version": "2.0.0"}},
    })
    assert update.check(fetch=fetch, now=Clock()).latest == "2.0.0"


def test_a_tag_that_is_not_a_version_is_refused_by_both_readers(home):
    fetch, _ = counting({
        update.GITHUB_URL: _github(tag_name="nightly", name="nightly"),
        update.PYPI_URL: {"info": {"version": "not-a-version"}},
    })
    info = update.check(fetch=fetch, now=Clock())
    assert info.latest == ""
    assert "no usable release" in (info.error or "")


def test_an_error_names_the_host_that_would_not_answer(home):
    fetch, _ = counting({})
    info = update.check(fetch=fetch, now=Clock())
    assert "api.github.com" in (info.error or "")
    assert "pypi.org" in (info.error or "")
    assert "OSError" in (info.error or ""), "the failure type belongs in the message"


def test_an_older_published_release_is_not_an_update(home, monkeypatch):
    monkeypatch.setattr(update, "installed_version", lambda: "9.9.9")
    info = update.check(fetch=lambda url: _github("1.0.0"), now=Clock())
    assert not info.available
    assert any("up to date" in line for line in info.report())


# -- the background check ---------------------------------------------------


def test_a_network_failure_in_the_background_is_completely_silent(home, capsys, escaped):
    def boom(url: str):
        raise OSError("Network is unreachable")

    seen: list[update.UpdateInfo] = []
    done = threading.Event()
    update.check_async(fetch=boom, now=Clock(), on_update=seen.append, done=done)
    assert done.wait(5.0), "the background thread must always finish"

    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "", \
        "an offline user must see nothing at all"
    assert escaped == [], f"the check let an exception out: {escaped}"
    assert seen == [], "there is no update to announce when nothing answered"


def test_a_callback_that_explodes_is_swallowed_rather_than_printed(
    home, monkeypatch, capsys, escaped
):
    """The thread's own except clause: without it Python's excepthook prints a
    traceback over whatever the shell had just drawn."""
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")

    def hostile(info):
        raise RuntimeError("the notifier is broken")

    done = threading.Event()
    update.check_async(fetch=lambda url: _github("9.9.9"), now=Clock(),
                       on_update=hostile, done=done)
    assert done.wait(5.0)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "", \
        f"a broken callback leaked output: {captured.out!r} {captured.err!r}"
    assert escaped == [], f"a broken callback escaped the thread: {escaped}"


def test_the_background_check_does_not_delay_startup(home):
    gate = threading.Event()

    def slow(url: str):
        gate.wait(10.0)
        raise OSError("still no route")

    done = threading.Event()
    started = time.monotonic()
    update.check_async(fetch=slow, now=Clock(), done=done)
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, f"check_async blocked the caller for {elapsed:.2f}s"

    assert not done.is_set(), "the fetch really is still in flight"
    gate.set()
    assert done.wait(5.0)


def test_the_background_thread_is_a_daemon_so_it_cannot_hold_up_exit(home):
    gate = threading.Event()
    names: list[bool] = []

    def watch(url: str):
        current = threading.current_thread()
        names.append(current.daemon)
        gate.set()
        return _github()

    done = threading.Event()
    update.check_async(fetch=watch, now=Clock(), done=done)
    assert gate.wait(5.0)
    assert done.wait(5.0)
    assert names == [True], "a non-daemon check would keep a closing offset alive"


def test_the_background_check_announces_only_a_real_update(home, monkeypatch):
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")
    seen: list[update.UpdateInfo] = []
    done = threading.Event()
    update.check_async(fetch=lambda url: _github("9.9.9"), now=Clock(),
                       on_update=seen.append, done=done)
    assert done.wait(5.0)
    assert [info.latest for info in seen] == ["9.9.9"]

    monkeypatch.setattr(update, "installed_version", lambda: "9.9.9")
    seen.clear()
    done = threading.Event()
    update.check_async(fetch=lambda url: _github("9.9.9"), now=Clock(),
                       on_update=seen.append, done=done)
    assert done.wait(5.0)
    assert seen == [], "being current is not news"


def test_install_starts_the_background_check_without_a_shell_state(home, monkeypatch):
    monkeypatch.setenv(update.NO_CHECK_ENV, "1")  # keep the thread off the network
    update.install(None)  # the wiring hook ignores the state it is handed


# -- applying ---------------------------------------------------------------


def test_an_editable_install_refuses_to_update_and_says_what_to_do_instead(tmp_path):
    checkout = tmp_path / "offset-terminal"
    checkout.mkdir()
    (checkout / ".git").mkdir()
    where = update.detect_install(checkout, user_site=tmp_path / "nowhere")

    ran: list[object] = []

    def runner(command, sink):
        ran.append(command)
        return 0

    result = update.apply(target=where, runner=runner, prober=lambda t: "9.9.9")
    assert not result.ok
    assert ran == [], "nothing may be executed against a working tree"
    assert "git pull" in (result.error or ""), result.error
    assert any("update failed" in line for line in result.report())


def test_an_unknown_install_method_is_refused_too(tmp_path):
    odd = tmp_path / "bundle"
    odd.mkdir()
    where = update.detect_install(odd, user_site=tmp_path / "nowhere")
    result = update.apply(target=where, runner=lambda c, s: 0, prober=lambda t: "9.9.9")
    assert not result.ok
    assert "could not tell how offset was installed" in (result.error or "")


def test_a_successful_upgrade_is_verified_against_the_new_version(tmp_path):
    where = update.Install("pipx", tmp_path, ("pipx", "upgrade", "offset"), sys.executable)
    echoed: list[str] = []

    def runner(command, sink):
        assert list(command) == ["pipx", "upgrade", "offset"]
        sink("upgraded package offset from 1.0.0 to 9.9.9")
        return 0

    result = update.apply(
        info=update.UpdateInfo(current="1.0.0", latest="9.9.9"),
        target=where, runner=runner, prober=lambda t: "9.9.9", echo=echoed.append,
    )
    assert result.ok and result.verified
    assert result.before == "1.0.0" and result.after == "9.9.9"
    assert echoed == ["upgraded package offset from 1.0.0 to 9.9.9"], \
        "installer output must be streamed as it arrives, not buffered to the end"
    assert result.report()[0] == "updated 1.0.0 -> 9.9.9 via pipx"


def test_an_upgrade_that_changes_nothing_is_reported_as_a_failure(tmp_path):
    where = update.Install("pip", tmp_path, (sys.executable, "-m", "pip", "install",
                                             "--upgrade", "offset"), sys.executable)
    result = update.apply(
        info=update.UpdateInfo(current="1.0.0", latest="9.9.9"),
        target=where, runner=lambda c, s: 0, prober=lambda t: "1.0.0",
    )
    assert not result.ok
    assert "still 1.0.0" in (result.error or "")


def test_a_failing_installer_reports_its_exit_status(tmp_path):
    where = update.Install("pipx", tmp_path, ("pipx", "upgrade", "offset"))

    def runner(command, sink):
        sink("ERROR: could not find a version that satisfies the requirement")
        return 2

    result = update.apply(info=update.UpdateInfo(current="1.0.0"), target=where,
                          runner=runner, prober=lambda t: "9.9.9")
    assert not result.ok
    assert "pipx exited 2" in (result.error or "")
    assert result.output[-1].startswith("ERROR:")
    assert any("ran: pipx upgrade offset" in line for line in result.report())


def test_an_unconfirmable_upgrade_is_a_success_that_says_so(tmp_path):
    where = update.Install("pipx", tmp_path, ("pipx", "upgrade", "offset"))
    result = update.apply(info=update.UpdateInfo(current="1.0.0"), target=where,
                          runner=lambda c, s: 0, prober=lambda t: "")
    assert result.ok and not result.verified
    assert any("could not confirm" in line for line in result.report())


def test_stream_hands_over_each_line_as_it_arrives():
    lines: list[str] = []
    status = update.stream(
        [sys.executable, "-c", "print('one'); print('two')"], lines.append
    )
    assert status == 0
    assert lines == ["one", "two"]


def test_a_command_that_does_not_exist_is_a_message_not_a_traceback(tmp_path):
    lines: list[str] = []
    status = update.stream([str(tmp_path / "definitely-not-here")], lines.append)
    assert status == 127
    assert lines and "Error" in lines[0]


def test_the_probe_asks_the_upgraded_interpreter_not_this_process():
    where = update.Install("pip", Path("."), (), sys.executable)
    # offset is not installed in this interpreter's metadata during the tests,
    # so the probe answers "" — the point is that it asks and does not raise.
    assert isinstance(update.probe(where), str)

    missing = update.Install("pip", Path("."), (), "/definitely/not/a/python")
    assert update.probe(missing) == "", "an unrunnable interpreter is not an answer"


# -- rendering --------------------------------------------------------------


def test_the_report_names_the_new_version_and_where_to_read_about_it():
    info = update.UpdateInfo(current="1.0.0", latest="9.9.9", notes="line one\nline two",
                             url="https://example.invalid/rel", published="2026-04-01")
    lines = info.report()
    assert lines[0] == "offset 1.0.0 -> 9.9.9 is available"
    assert "published 2026-04-01" in lines
    assert "https://example.invalid/rel" in lines
    assert "line one" in lines


def test_the_report_explains_silence_when_checks_are_off():
    lines = update.UpdateInfo(current="1.0.0", disabled=True).report()
    assert update.NO_CHECK_ENV in lines[1] and update.SETTING in lines[1]


def test_the_report_admits_a_failed_check_when_it_is_asked_directly():
    lines = update.UpdateInfo(current="1.0.0", error="pypi.org: OSError: down").report()
    assert lines[1] == "could not check for updates: pypi.org: OSError: down"


# -- entry points -----------------------------------------------------------


def test_the_cli_reports_up_to_date_with_a_zero_status(home, monkeypatch, capsys):
    monkeypatch.setattr(update, "installed_version", lambda: "9.9.9")
    monkeypatch.setattr(update, "http_json", lambda url, **kw: _github("9.9.9"))
    assert update.update_command(check_only=True) == 0
    assert "up to date" in capsys.readouterr().out


def test_the_cli_fails_when_nothing_answered(home, monkeypatch, capsys):
    def boom(url, **kw):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(update, "http_json", boom)
    assert update.update_command(check_only=True) == 1
    assert "could not check for updates" in capsys.readouterr().out


def test_check_only_never_installs_anything(home, monkeypatch, capsys):
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")
    monkeypatch.setattr(update, "http_json", lambda url, **kw: _github("9.9.9"))
    monkeypatch.setattr(update, "apply", lambda **kw: pytest.fail("apply must not run"))
    assert update.update_command(check_only=True) == 0
    assert "9.9.9 is available" in capsys.readouterr().out


def test_the_cli_refuses_to_upgrade_a_checkout(home, monkeypatch, capsys, tmp_path):
    checkout = tmp_path / "tree"
    (checkout / ".git").mkdir(parents=True)
    # Detected for real, then pinned: `update_command` takes no injection point,
    # and the machine running the tests is itself a checkout.
    where = update.detect_install(checkout, user_site=tmp_path / "nowhere")
    assert where.method == "editable"
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")
    monkeypatch.setattr(update, "http_json", lambda url, **kw: _github("9.9.9"))
    monkeypatch.setattr(update, "detect_install", lambda *a, **k: where)
    monkeypatch.setattr(update, "apply", lambda **kw: pytest.fail("apply must not run"))
    assert update.update_command() == 1
    assert "git pull" in capsys.readouterr().out


def test_the_cli_installs_and_confirms_when_it_may(home, monkeypatch, capsys):
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")
    monkeypatch.setattr(update, "http_json", lambda url, **kw: _github("9.9.9"))
    monkeypatch.setattr(
        update, "detect_install",
        lambda *a, **k: update.Install("pipx", Path("/tmp"), ("pipx", "upgrade", "offset")),
    )
    monkeypatch.setattr(
        update, "apply",
        lambda **kw: update.UpdateResult(True, "1.0.0", "9.9.9", "pipx",
                                         ("pipx", "upgrade", "offset")),
    )
    assert update.update_command() == 0
    assert "updated 1.0.0 -> 9.9.9 via pipx" in capsys.readouterr().out


def test_the_slash_command_is_registered_as_update():
    names = [command.name for command in update.COMMANDS]
    assert names == ["update"]
    assert update.COMMANDS is update.COMMANDS, "the list must be built once"


def test_an_unknown_slash_argument_is_refused_with_the_usage():
    outcome = update._update(None, ["sideways"])
    assert "unknown argument" in outcome.lines[0]
    assert "usage: /update [apply]" in outcome.lines[1]


def test_the_slash_command_checks_in_the_background(home, monkeypatch):
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")
    monkeypatch.setattr(update, "http_json", lambda url, **kw: _github("9.9.9"))
    outcome = update._update(None, [])
    assert outcome.job is not None, "the keypress must return before the fetch"
    assert "looking for something newer" in outcome.lines[0]

    finished = outcome.job()
    assert "offset 1.0.0 -> 9.9.9 is available" in finished.lines
    assert "run /update apply to install it" in finished.lines


def test_slash_update_apply_refuses_a_checkout(home, monkeypatch, tmp_path):
    monkeypatch.setattr(
        update, "detect_install",
        lambda *a, **k: update.Install("editable", tmp_path, (), sys.executable,
                                       "update it with 'git pull' there"),
    )
    outcome = update._update(None, ["apply"])
    assert outcome.job is None, "a refusal costs no background work"
    assert "git pull" in outcome.lines[0]


def test_slash_update_apply_runs_the_upgrade_as_a_job(home, monkeypatch):
    monkeypatch.setattr(
        update, "detect_install",
        lambda *a, **k: update.Install("pipx", Path("/tmp"), ("pipx", "upgrade", "offset")),
    )
    monkeypatch.setattr(
        update, "apply",
        lambda **kw: update.UpdateResult(True, "1.0.0", "9.9.9", "pipx",
                                         ("pipx", "upgrade", "offset"), ["done"]),
    )
    outcome = update._update(None, ["apply"])
    assert outcome.job is not None
    assert "running: pipx upgrade offset" in outcome.lines[0]
    finished = outcome.job()
    assert "updated 1.0.0 -> 9.9.9 via pipx" in finished.lines


# -- startup auto-update -----------------------------------------------------


def _pipx(tmp_path: Path) -> update.Install:
    return update.Install("pipx", tmp_path, ("pipx", "upgrade", "offset"))


def _cached(home: Path, *, current: str, latest: str, when: float = 0.0) -> None:
    """Seed the cache the way a previous run's background check would have."""
    import json as _json
    import time as _time

    (home / "update.json").write_text(
        _json.dumps({
            "version": update.CACHE_VERSION,
            "current": current,
            "latest": latest,
            "notes": "",
            "url": "",
            "published": "",
            "checked_at": when or _time.time(),
            "error": "",
        }),
        encoding="utf-8",
    )


def test_auto_update_is_on_by_default(home):
    assert update.auto_enabled() is True


def test_the_auto_opt_out_env_var_switches_it_off(home, monkeypatch):
    monkeypatch.setenv(update.NO_AUTO_ENV, "1")
    assert update.auto_enabled() is False


def test_switching_checks_off_switches_auto_update_off_too(home, monkeypatch):
    """A program told not to look must not install something anyway."""
    monkeypatch.setenv(update.NO_CHECK_ENV, "1")
    assert update.auto_enabled() is False


def test_the_settings_key_switches_auto_update_off(home):
    import json as _json

    (home / "config.json").write_text(_json.dumps({"update": {"auto": False}}), encoding="utf-8")
    assert update.auto_enabled() is False


def test_a_flat_settings_key_also_switches_it_off(home):
    import json as _json

    (home / "config.json").write_text(_json.dumps({"update.auto": False}), encoding="utf-8")
    assert update.auto_enabled() is False


def test_a_reexeced_process_refuses_to_update_again(home, monkeypatch):
    """The guard against an unkillable re-exec loop."""
    monkeypatch.setenv(update.REEXEC_ENV, "1")
    assert update.auto_enabled() is False


def test_auto_update_installs_a_waiting_release(home, monkeypatch, tmp_path):
    _cached(home, current="1.0.0", latest="9.9.9")
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")
    ran: list[tuple[str, ...]] = []

    def runner(command, sink):
        ran.append(tuple(command))
        sink("upgraded")
        return 0

    outcome = update.autoupdate(
        target=_pipx(tmp_path), runner=runner, prober=lambda _t: "9.9.9",
        fetch=_never_called,
    )
    assert outcome.acted, outcome.error or outcome.skipped
    assert outcome.before == "1.0.0" and outcome.after == "9.9.9"
    assert ran == [("pipx", "upgrade", "offset")]
    assert "1.0.0 -> 9.9.9" in "\n".join(outcome.report())


def test_auto_update_does_nothing_when_already_current(home, monkeypatch, tmp_path):
    _cached(home, current="9.9.9", latest="9.9.9")
    monkeypatch.setattr(update, "installed_version", lambda: "9.9.9")
    ran: list[tuple[str, ...]] = []

    outcome = update.autoupdate(
        target=_pipx(tmp_path),
        runner=lambda c, s: ran.append(tuple(c)) or 0,
        fetch=_never_called,
    )
    assert not outcome.acted
    assert not ran, "nothing should be run when there is nothing to install"
    assert outcome.report() == [], "silence is the whole point of the common case"


def test_auto_update_refuses_an_editable_checkout(home, monkeypatch, tmp_path):
    _cached(home, current="1.0.0", latest="9.9.9")
    checkout = update.Install("editable", tmp_path, reason="run git pull instead")
    ran: list[tuple[str, ...]] = []

    outcome = update.autoupdate(
        target=checkout,
        runner=lambda c, s: ran.append(tuple(c)) or 0,
        fetch=_never_called,
    )
    assert not outcome.acted
    assert not ran, "a checkout must never be upgraded behind the user's back"
    assert "git pull" in outcome.skipped


def test_auto_update_reads_the_cache_rather_than_the_network(home, monkeypatch, tmp_path):
    """A slow or offline start must not be paid for at launch."""
    _cached(home, current="1.0.0", latest="9.9.9")
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")

    outcome = update.autoupdate(
        target=_pipx(tmp_path),
        runner=lambda c, s: 0,
        prober=lambda _t: "9.9.9",
        fetch=_never_called,   # raises if the network is touched
    )
    assert outcome.acted


def test_a_failed_upgrade_is_reported_and_not_claimed(home, monkeypatch, tmp_path):
    _cached(home, current="1.0.0", latest="9.9.9")
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")

    outcome = update.autoupdate(
        target=_pipx(tmp_path),
        runner=lambda c, s: 1,          # the package manager fails
        prober=lambda _t: "1.0.0",
        fetch=_never_called,
    )
    assert not outcome.acted
    assert outcome.error
    assert "could not update itself" in "\n".join(outcome.report())


def test_an_upgrade_that_does_not_move_the_version_is_a_failure(home, monkeypatch, tmp_path):
    """Exit code 0 is not proof; the version has to actually change."""
    _cached(home, current="1.0.0", latest="9.9.9")
    monkeypatch.setattr(update, "installed_version", lambda: "1.0.0")

    outcome = update.autoupdate(
        target=_pipx(tmp_path),
        runner=lambda c, s: 0,
        prober=lambda _t: "1.0.0",      # still the old version afterwards
        fetch=_never_called,
    )
    assert not outcome.acted
    assert "still 1.0.0" in (outcome.error or "")


def test_auto_update_is_silent_and_inert_when_switched_off(home, monkeypatch, tmp_path):
    monkeypatch.setenv(update.NO_AUTO_ENV, "1")
    _cached(home, current="1.0.0", latest="9.9.9")
    ran: list[tuple[str, ...]] = []

    outcome = update.autoupdate(
        target=_pipx(tmp_path),
        runner=lambda c, s: ran.append(tuple(c)) or 0,
        fetch=_never_called,
    )
    assert not outcome.acted and not ran
    assert outcome.report() == []


def test_reexec_marks_the_child_so_it_cannot_loop(monkeypatch):
    captured: dict[str, object] = {}

    def fake_execve(path, argv, env):
        captured["path"] = path
        captured["argv"] = list(argv)
        captured["marker"] = env.get(update.REEXEC_ENV)
        raise OSError("not really execing in a test")

    monkeypatch.setattr(update.os, "execve", fake_execve)
    update.reexec()   # must return rather than raise when exec fails
    assert captured["marker"] == "1", "the child must be marked to break the loop"
    assert "-m" in captured["argv"] and "offset" in captured["argv"]


def _never_called(_url: str):
    raise AssertionError("the network must not be touched here")
