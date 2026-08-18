"""Costs that are paid on every start, and a flag that did not work.

Two of these are performance and one is correctness, but they were found the same
way: measuring what the program actually does rather than what it looks like it
does.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import time

import pytest

from offset.providers.base import ToolCall
from offset.shell.app import build_state

ROOT = pathlib.Path(__file__).resolve().parent.parent


def import_cost(module: str) -> float:
    """Seconds to import `module` in a fresh interpreter, best of three."""
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        done = subprocess.run([sys.executable, "-c", f"import {module}"],
                              capture_output=True, cwd=ROOT)
        assert done.returncode == 0, done.stderr.decode()[:400]
        best = min(best, time.perf_counter() - started)
    return best


def loaded_after(module: str) -> set[str]:
    """Which modules are in sys.modules once `module` has been imported."""
    code = f"import {module}, sys, json; print(json.dumps(sorted(sys.modules)))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr.decode()[:400]
    return set(json.loads(done.stdout.decode().splitlines()[-1]))


# -- what a start pays for ---------------------------------------------------


def test_importing_a_provider_does_not_open_the_network_stack():
    """`urllib.request` pulls http.client, email, socket and ssl behind it.

    Importing the provider *base* - dataclasses describing events - used to drag
    all of it in, because the package eagerly imported every concrete provider.
    """
    loaded = loaded_after("offset.providers.base")
    assert "http.client" not in loaded, "the http stack is loaded before anyone asks to use it"
    assert "urllib.request" not in loaded
    assert "ssl" not in loaded


def test_importing_the_agent_does_not_open_the_network_stack():
    """The agent reached auth, which reached oauth, which imported urllib."""
    loaded = loaded_after("offset.core.agent")
    assert "http.client" not in loaded
    assert "webbrowser" not in loaded, "nothing should touch the browser until /login"


def test_importing_oauth_does_not_start_an_http_server():
    loaded = loaded_after("offset.providers.oauth")
    assert "http.server" not in loaded
    assert "urllib.request" not in loaded
    assert "urllib.parse" in loaded, "url building is the cheap part and stays eager"


def test_the_provider_names_are_available_without_importing_providers():
    """`/models`, the login targets and the picker only want names."""
    code = ("import sys, json;"
            "from offset.providers.registry import PROVIDERS;"
            "names = sorted(PROVIDERS);"
            "print(json.dumps([names, 'offset.providers.anthropic' in sys.modules]))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr.decode()[:400]
    names, imported = json.loads(done.stdout.decode().splitlines()[-1])
    assert "anthropic" in names and "mock" in names
    assert not imported, "listing the providers must not import them"


def test_a_provider_still_resolves_when_it_is_actually_wanted():
    from offset.providers.registry import provider_for, resolve

    assert type(provider_for("mock")).__name__ == "Mock"
    assert type(resolve("claude-opus-4-20250514")[0]).__name__ == "Anthropic"


def test_an_unknown_provider_still_raises():
    from offset.providers.registry import provider_for

    with pytest.raises(KeyError):
        provider_for("not-a-provider")


@pytest.mark.parametrize("module,ceiling", [
    ("offset.providers.base", 0.34),
    ("offset.core.agent", 0.50),
])
def test_the_lower_layers_stay_cheap_to_import(module, ceiling):
    """A ceiling, not a benchmark: this catches a new eager import, not drift."""
    cost = import_cost(module)
    assert cost < ceiling, f"{module} took {cost:.3f}s to import (ceiling {ceiling}s)"


# -- the credential store ----------------------------------------------------


def test_the_credential_file_is_read_once_not_once_per_model(tmp_path, monkeypatch):
    """`available()` asks about every model in the catalogue.

    Each question re-read and re-parsed the same file: about thirty reads of the
    same bytes to build one roster, on every startup and every /models.
    """
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    from offset.providers import registry

    registry.credentials_file().parent.mkdir(parents=True, exist_ok=True)
    registry.credentials_file().write_text(json.dumps({"anthropic": "sk-one"}), encoding="utf-8")

    reads = {"n": 0}
    real = pathlib.Path.read_text

    def counted(self, *args, **kwargs):
        if self == registry.credentials_file():
            reads["n"] += 1
        return real(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", counted)
    registry.available()
    assert reads["n"] <= 1, f"{reads['n']} reads to build one roster"
    reads["n"] = 0
    for _ in range(5):
        registry.available()
    assert reads["n"] == 0, "an unchanged file should not be read again"


def test_an_edit_by_another_process_is_seen_immediately(tmp_path, monkeypatch):
    """A cache that hides a key written by a second offset is worse than no cache."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    from offset.providers import registry

    registry.credentials_file().parent.mkdir(parents=True, exist_ok=True)
    registry.credentials_file().write_text(json.dumps({"anthropic": "sk-one"}), encoding="utf-8")
    assert registry.credential("anthropic") == "sk-one"

    time.sleep(0.01)
    registry.credentials_file().write_text(json.dumps({"anthropic": "sk-two"}), encoding="utf-8")
    assert registry.credential("anthropic") == "sk-two", "an external write was cached away"

    registry.credentials_file().unlink()
    assert registry.credential("anthropic") is None, "a deleted file was cached away"


def test_a_corrupt_credential_file_is_not_cached_as_truth(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    from offset.providers import registry

    registry.credentials_file().parent.mkdir(parents=True, exist_ok=True)
    registry.credentials_file().write_text("{not json", encoding="utf-8")
    assert registry.credential("anthropic") is None

    time.sleep(0.01)
    registry.credentials_file().write_text(json.dumps({"anthropic": "sk-fixed"}), encoding="utf-8")
    assert registry.credential("anthropic") == "sk-fixed", "recovery must be visible"


# -- asking for ignored files -------------------------------------------------


@pytest.fixture()
def noisy_repo(tmp_path, monkeypatch):
    """A workspace with the directories every project ignores."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    root = tmp_path / "ws"
    root.mkdir()
    (root / ".gitignore").write_text("node_modules/\n.venv/\nbuild/\n", encoding="utf-8")
    for folder, count in (("src", 8), ("node_modules/pkg", 20), (".venv/lib", 12), ("build", 5)):
        where = root / folder
        where.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            (where / f"f{i}.py").write_text("needle = 1\n", encoding="utf-8")
    return build_state(root, model="mock")


def run_tool(state, name: str, args: dict) -> int:
    got = state.agent.runtime.execute(ToolCall("c", name, args)).result
    assert got.ok, got.error
    return len(got.content.splitlines())


def test_glob_ignores_the_noise_by_default(noisy_repo):
    assert run_tool(noisy_repo, "glob", {"pattern": "*.py", "limit": 2000}) == 8


def test_glob_ignored_actually_reaches_the_ignored_directories(noisy_repo):
    """The regression: the built-in prune list still excluded node_modules.

    Standing .gitignore down while keeping the hardcoded list meant the flag
    reached nothing it promised.
    """
    assert run_tool(noisy_repo, "glob", {"pattern": "*.py", "limit": 2000, "ignored": True}) == 45


def test_grep_ignores_the_noise_by_default(noisy_repo):
    assert run_tool(noisy_repo, "grep", {"pattern": "needle", "limit": 2000}) == 8


def test_grep_ignored_actually_reaches_the_ignored_directories(noisy_repo):
    assert run_tool(noisy_repo, "grep", {"pattern": "needle", "limit": 2000, "ignored": True}) == 45


def test_respecting_gitignore_is_still_the_faster_walk(noisy_repo):
    """If the flag works, the default has less to do."""
    default = run_tool(noisy_repo, "glob", {"pattern": "*.py", "limit": 2000})
    everything = run_tool(noisy_repo, "glob", {"pattern": "*.py", "limit": 2000, "ignored": True})
    assert everything > default * 4


def test_a_provider_can_still_be_registered_at_runtime():
    """Assignment is a real registration point, not just a test convenience."""
    from offset.providers.registry import PROVIDERS, provider_for

    class Stand:
        name = "stand-in"

    PROVIDERS["invented-today"] = Stand
    try:
        assert "invented-today" in PROVIDERS
        assert isinstance(provider_for("invented-today"), Stand)
    finally:
        del PROVIDERS["invented-today"]
    assert "invented-today" not in PROVIDERS


def test_an_override_of_a_built_in_wins_and_can_be_taken_back():
    from offset.providers.registry import PROVIDERS, provider_for

    class Stand:
        name = "stand-in"

    PROVIDERS["mock"] = Stand
    try:
        assert isinstance(provider_for("mock"), Stand)
    finally:
        del PROVIDERS["mock"]
    assert type(provider_for("mock")).__name__ == "Mock", "the built-in must come back"


def test_deleting_a_built_in_without_overriding_it_is_refused():
    from offset.providers.registry import PROVIDERS

    with pytest.raises(KeyError, match="built in"):
        del PROVIDERS["anthropic"]


# -- the first run on a machine with nothing configured ----------------------


@pytest.fixture()
def clean_machine(tmp_path, monkeypatch):
    """No credentials anywhere: not in the environment, not on disk."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    for name in list(__import__("os").environ):
        if name.endswith(("_API_KEY", "_CLIENTID", "_CLIENTSECRET")):
            monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_a_fresh_install_starts_on_a_model_it_can_actually_reach(clean_machine):
    """The regression: the configured default is a paid model.

    A new install therefore started on Claude with no key, and the very first
    message failed with an auth error before the user had done anything wrong.
    """
    from offset.shell.app import reachable_model

    assert reachable_model("claude-sonnet-4-20250514") == "mock"


def test_the_configured_default_is_honoured_once_it_can_be_reached(clean_machine, monkeypatch):
    from offset.shell.app import reachable_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert reachable_model("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"


def test_an_explicit_choice_always_wins(clean_machine):
    """`--model` is the user talking; it is not second-guessed."""
    state = build_state(clean_machine, model="claude-opus-4-20250514")
    assert state.model == "claude-opus-4-20250514"


def test_a_fresh_install_seats_a_roster_that_can_answer(clean_machine):
    state = build_state(clean_machine)
    assert [seat.model for seat in state.ensemble] == ["mock"], \
        "a council of unreachable models is worse than a council of one"


def test_reachability_is_stricter_than_availability(clean_machine):
    """A catalogue entry for ollama says nothing about ollama running."""
    from offset.providers.registry import available, reachable

    # `mock` is local too, and is always reachable by design; the point is about
    # models that need a server of their own.
    served = [m for m in available() if m.local and m.provider != "mock"]
    assert served, "the catalogue should list models that need a local server"
    assert not any(reachable(m.id) for m in served), \
        "nothing is listening here, so none of them are reachable"


def test_the_scripted_provider_is_always_reachable():
    from offset.providers.registry import reachable

    assert reachable("mock"), "it needs no key and no network; it must never be excluded"


# -- the suite must not read or write the real home --------------------------


def test_the_config_home_is_resolved_late_not_at_import():
    """Caching it at import made every test that set OFFSET_HOME a no-op.

    Worse than untidy: the suite then read - and wrote - the credential store of
    whoever was running it.
    """
    code = (
        "import os, json, tempfile;"
        "os.environ['OFFSET_HOME'] = first = tempfile.mkdtemp();"
        "from offset.providers import registry;"
        "a = str(registry.credentials_file());"
        "os.environ['OFFSET_HOME'] = second = tempfile.mkdtemp();"
        "b = str(registry.credentials_file());"
        "print(json.dumps([a.startswith(first), b.startswith(second)]))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=ROOT)
    assert done.returncode == 0, done.stderr.decode()[:400]
    before, after = json.loads(done.stdout.decode().splitlines()[-1])
    assert before, "the first home was not honoured"
    assert after, "moving OFFSET_HOME after import had no effect"


def test_every_state_file_agrees_where_home_is(tmp_path, monkeypatch):
    """Three modules used to compute this separately, in three places."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    from offset.core import permissions, settings
    from offset.providers import registry

    assert registry.config_dir() == settings.home()
    assert permissions.config_dir() == settings.home()
    assert registry.credentials_file().parent == settings.home()
    assert permissions.permissions_file().parent == settings.home()


def test_building_a_shell_writes_nothing_outside_the_given_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("OFFSET_HOME", str(home))
    workspace = tmp_path / "ws"
    workspace.mkdir()

    real_home = pathlib.Path.home() / ".offset"
    before = sorted(p.name for p in real_home.iterdir()) if real_home.exists() else []
    build_state(workspace, model="mock")
    after = sorted(p.name for p in real_home.iterdir()) if real_home.exists() else []
    assert before == after, "it touched the real home instead of the one it was given"
    assert home.exists(), "it should have used the home it was given"
