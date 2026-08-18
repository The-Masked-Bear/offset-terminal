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

    registry.CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    registry.CREDENTIALS_FILE.write_text(json.dumps({"anthropic": "sk-one"}), encoding="utf-8")

    reads = {"n": 0}
    real = pathlib.Path.read_text

    def counted(self, *args, **kwargs):
        if self == registry.CREDENTIALS_FILE:
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

    registry.CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    registry.CREDENTIALS_FILE.write_text(json.dumps({"anthropic": "sk-one"}), encoding="utf-8")
    assert registry.credential("anthropic") == "sk-one"

    time.sleep(0.01)
    registry.CREDENTIALS_FILE.write_text(json.dumps({"anthropic": "sk-two"}), encoding="utf-8")
    assert registry.credential("anthropic") == "sk-two", "an external write was cached away"

    registry.CREDENTIALS_FILE.unlink()
    assert registry.credential("anthropic") is None, "a deleted file was cached away"


def test_a_corrupt_credential_file_is_not_cached_as_truth(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    from offset.providers import registry

    registry.CREDENTIALS_FILE.parent.mkdir(parents=True, exist_ok=True)
    registry.CREDENTIALS_FILE.write_text("{not json", encoding="utf-8")
    assert registry.credential("anthropic") is None

    time.sleep(0.01)
    registry.CREDENTIALS_FILE.write_text(json.dumps({"anthropic": "sk-fixed"}), encoding="utf-8")
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
