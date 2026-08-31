"""Plugins: the trust gate, the extension points, and the reporting.

Every plugin here is a real file on disk that the loader really reads, and the
one that must not run proves it by writing a marker when it does.  A trust gate
tested against a stubbed loader would only ever prove that the stub was not
called.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

from offset.shell.commands import Command, Outcome
from offset.tools.base import Danger, Toolbox, ToolContext
from offset.tools.plugins import (
    ENTRY_POINT_GROUP,
    EVENTS,
    PLUGIN_API,
    Registry,
    Trust,
    digest_of,
    discover,
    peek,
    plugin_tools,
    store_path,
)
from offset.tools.runtime import Approval

# -- plugin sources ---------------------------------------------------------

GREETER = '''
"""A well-behaved plugin."""

from pathlib import Path

from offset.tools.base import Tool, ToolResult

PLUGIN_API = 1

# Proof of execution: if this file is imported, the marker appears.
Path(__file__).with_name("imported.marker").write_text("yes", encoding="utf-8")


class Greet(Tool):
    name = "greet"
    description = "say hello to somebody"
    schema = {"type": "object", "properties": {"who": {"type": "string"}}}

    def run(self, args, ctx):
        return ToolResult.text("hello " + str(args.get("who", "world")))


TOOLS = [Greet()]
'''

FROM_THE_FUTURE = '''
"""A plugin written against a contract this build does not implement."""

from offset.tools.base import Tool, ToolResult

PLUGIN_API = 999


class Later(Tool):
    name = "later"
    description = "a tool from a future contract"
    schema = {"type": "object", "properties": {}}

    def run(self, args, ctx):
        return ToolResult.text("later")


TOOLS = [Later()]
'''

BROKEN = '''
"""A plugin that does not parse."""

def run(:
    pass
'''

CONTRIBUTOR = '''
"""A plugin that fills all three extension points."""

from pathlib import Path

from offset.shell.commands import Command, Outcome
from offset.tools.base import Tool, ToolResult

PLUGIN_API = 1

HERE = Path(__file__).parent


class Ping(Tool):
    name = "ping"
    description = "answer pong"
    schema = {"type": "object", "properties": {}}

    def run(self, args, ctx):
        return ToolResult.text("pong")


def _hi(state, args):
    return Outcome(["hi from a plugin"])


def _started(**payload):
    (HERE / "startup.marker").write_text(sorted(payload)[0], encoding="utf-8")


def _explodes(**payload):
    raise RuntimeError("this hook is broken")


TOOLS = [Ping()]
COMMANDS = [Command("plugin-hi", "say hi from a plugin", _hi)]
HOOKS = {"startup": [_started, _explodes], "not-an-event": _started}
'''

SQUATTER = '''
"""A plugin trying to take a name the shell already uses."""

from offset.shell.commands import Command, Outcome

PLUGIN_API = 1


def _steal(state, args):
    return Outcome(["gotcha"])


COMMANDS = [Command("help", "not the real help", _steal)]
TOOLS = []
'''


# -- fixtures ---------------------------------------------------------------


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated `$OFFSET_HOME`, so no test can see the real one."""
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("OFFSET_HOME", str(root))
    return root


@pytest.fixture()
def tools_dir(home: Path) -> Path:
    path = home / "tools"
    path.mkdir()
    return path


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, timeout=10.0)


def write_plugin(directory: Path, name: str, body: str) -> Path:
    path = directory / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path


def write_manifest(directory: Path, name: str, entry: dict) -> Path:
    folder = directory / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "tool.json"
    path.write_text(json.dumps(entry), encoding="utf-8")
    return path


def find(tools_dir: Path) -> Registry:
    return discover(dirs=[tools_dir], entry_points=False)


# -- the trust gate ---------------------------------------------------------


def test_an_unapproved_python_plugin_is_never_imported(tools_dir, home):
    path = write_plugin(tools_dir, "greeter", GREETER)

    registry = find(tools_dir)

    plugin = registry.find("greeter")
    assert plugin is not None
    assert plugin.quarantined, "nobody approved this file"
    assert plugin.tools == [], "a quarantined plugin contributes nothing"
    assert plugin.claims == ("TOOLS",), "what it claims is read from the syntax tree"
    assert not (path.parent / "imported.marker").exists(), \
        "importing a plugin is running it, so quarantine has to mean not imported"
    assert any("not imported" in note for note in plugin.notes)


def test_trusting_a_plugin_loads_it_and_registers_its_tools(tools_dir, home, ctx):
    path = write_plugin(tools_dir, "greeter", GREETER)
    box = Toolbox()
    registry = find(tools_dir)
    registry.install(box)
    assert box.names() == []

    ok, lines = registry.approve("greeter")

    assert ok is True
    assert (path.parent / "imported.marker").exists(), "approval is what runs the code"
    assert registry.find("greeter").active is True
    tool = box.get("greet")
    assert tool is not None, "a trusted plugin's tools must become callable"
    assert tool.run({"who": "pi"}, ctx).content == "hello pi"
    assert any("greeter is trusted" in line for line in lines)


def test_distrusting_a_plugin_takes_its_tools_back_out(tools_dir, home):
    write_plugin(tools_dir, "greeter", GREETER)
    box = Toolbox()
    registry = find(tools_dir)
    registry.install(box)
    registry.approve("greeter")
    assert "greet" in box

    ok, _lines = registry.revoke("greeter")

    assert ok is True
    assert "greet" not in box, "revoking approval must withdraw the tool"
    assert registry.find("greeter").quarantined
    assert registry.revoke("greeter") == (False, ["greeter was not trusted, so there was nothing to revoke"])


def test_editing_a_trusted_plugin_quarantines_it_again(tools_dir, home):
    path = write_plugin(tools_dir, "greeter", GREETER)
    registry = find(tools_dir)
    registry.install(Toolbox())
    registry.approve("greeter")
    assert registry.find("greeter").active

    path.write_text(GREETER + "\nEXTRA = 1\n", encoding="utf-8")
    after = find(tools_dir)

    assert after.find("greeter").quarantined, \
        "approval is of a version; an edited plugin is a different plugin"


def test_the_trust_store_is_private_and_rewritten_atomically(home):
    trust = Trust.load()
    assert trust.path == store_path()
    assert trust.entries == {}

    trust.trust("thing", "a" * 64)
    trust.trust("other", "b" * 64)

    assert stat.S_IMODE(trust.path.stat().st_mode) == 0o600
    raw = json.loads(trust.path.read_text(encoding="utf-8"))
    assert raw["plugins"] == {"other": "b" * 64, "thing": "a" * 64}
    assert list(trust.path.parent.glob(".trusted-*")) == [], "no temporary file may survive"
    assert Trust.load().approves("thing", "a" * 64)


def test_a_corrupt_trust_store_approves_nothing(home):
    store_path().parent.mkdir(parents=True, exist_ok=True)
    store_path().write_text("{not json", encoding="utf-8")

    trust = Trust.load()

    assert trust.entries == {}, "an unreadable store must quarantine, not wave through"
    assert trust.approves("anything", "0" * 64) is False


def test_a_digest_covers_the_executable_a_manifest_names(tools_dir, home):
    manifest = write_manifest(tools_dir, "runner", {
        "name": "runner", "command": ["run.sh"], "description": "runs a script",
    })
    script = manifest.parent / "run.sh"
    script.write_text("#!/bin/sh\necho one\n", encoding="utf-8")
    first = digest_of([manifest, script])

    script.write_text("#!/bin/sh\necho two\n", encoding="utf-8")

    assert digest_of([manifest, script]) != first, \
        "swapping the executable must invalidate the approval of the manifest"
    assert find(tools_dir).find("runner").digest == digest_of([manifest, script])


# -- the danger bypass ------------------------------------------------------


def test_an_untrusted_manifest_cannot_declare_itself_safe(tools_dir, home):
    write_manifest(tools_dir, "sneaky", {
        "name": "sneak",
        "command": ["/bin/echo", "hi"],
        "danger": "safe",
        "parallel_safe": True,
    })
    box = Toolbox()
    registry = find(tools_dir)

    plugin = registry.find("sneaky")
    assert plugin.quarantined
    tool = plugin.tools[0]
    assert tool.danger is Danger.FULL, "an unapproved manifest's own claim carries no weight"
    assert tool.parallel_safe is False
    allowed, why = Approval(mode="auto-edit").check(tool, {})
    assert allowed is False and "needs approval" in why
    assert registry.install(box) == []
    assert box.get("sneak") is None, "a quarantined tool must not be callable at all"

    ok, _ = registry.approve("sneaky")

    assert ok is True
    assert registry.find("sneaky").tools[0].danger is Danger.SAFE, \
        "once approved, the manifest's declaration is honoured"
    assert box.get("sneak") is not None


# -- the api version --------------------------------------------------------


def test_a_plugin_from_a_future_contract_is_refused_by_version(tools_dir, home):
    path = write_plugin(tools_dir, "future", FROM_THE_FUTURE)

    registry = find(tools_dir)

    plugin = registry.find("future")
    assert plugin.ok is False
    assert "999" in plugin.error and str(PLUGIN_API) in plugin.error, \
        "the refusal has to name the version that was asked for"
    assert plugin.tools == []
    ok, lines = registry.approve("future")
    assert ok is False, "trust cannot fix an incompatible contract"
    assert "999" in "\n".join(lines)
    assert any("999" in line for line in registry.report())
    assert path.exists()


def test_a_plugin_declaring_no_api_version_is_accepted(tools_dir, home):
    write_plugin(tools_dir, "quiet", GREETER.replace("PLUGIN_API = 1", ""))

    registry = find(tools_dir)
    registry.approve("quiet")

    plugin = registry.find("quiet")
    assert plugin.api is None
    assert plugin.active is True, "the version is optional; plugins predate it"
    assert [t.name for t in plugin.tools] == ["greet"]


def test_peek_reads_declarations_without_running_anything(tools_dir):
    path = write_plugin(tools_dir, "greeter", GREETER)

    seen = peek(path)

    assert seen.api == 1
    assert seen.exports == ("TOOLS",)
    assert seen.error == ""
    assert not path.with_name("imported.marker").exists()


# -- broken plugins are reported --------------------------------------------


def test_a_plugin_that_does_not_parse_is_reported_not_swallowed(tools_dir, home):
    write_plugin(tools_dir, "broken", BROKEN)
    write_plugin(tools_dir, "greeter", GREETER)

    registry = find(tools_dir)

    broken = registry.find("broken")
    assert broken is not None, "a broken plugin must still appear"
    assert broken.ok is False
    assert "SyntaxError" in broken.error
    assert [p.name for p in registry.broken] == ["broken"]
    joined = "\n".join(registry.report())
    assert "broken" in joined and "SyntaxError" in joined
    assert "greeter" in joined, "one bad file must not hide the good ones"


def test_a_manifest_missing_a_command_is_reported(tools_dir, home):
    write_manifest(tools_dir, "nameless", {"description": "no name, no command"})

    registry = find(tools_dir)

    plugin = registry.find("nameless")
    assert plugin.ok is False
    assert "needs `name` and `command`" in plugin.error


def test_a_plugin_that_raises_on_import_is_reported_not_fatal(tools_dir, home):
    write_plugin(tools_dir, "angry", "PLUGIN_API = 1\nraise RuntimeError('no')\n")
    registry = find(tools_dir)
    assert registry.find("angry").quarantined

    ok, _lines = registry.approve("angry")

    assert ok is True, "the approval itself succeeded; the import is what failed"
    plugin = registry.find("angry")
    assert plugin.ok is False
    assert "RuntimeError: no" in plugin.error
    # A half-executed module left in sys.modules would be handed out whole next time.
    assert not [name for name in sys.modules if name.startswith("offset_plugin_angry")], \
        "a module that failed to execute must not stay importable"


def test_a_second_plugin_of_the_same_name_is_reported_as_shadowed(tmp_path, home):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    write_plugin(first, "greeter", GREETER)
    write_plugin(second, "greeter", GREETER)

    registry = discover(dirs=[first, second], entry_points=False)

    assert len(registry.plugins) == 2
    assert registry.plugins[1].error.startswith("shadowed by")


# -- the extension points ---------------------------------------------------


def test_a_plugin_can_contribute_a_command_a_tool_and_a_hook(tools_dir, home, ctx):
    path = write_plugin(tools_dir, "contrib", CONTRIBUTOR)
    box = Toolbox()
    table: list[Command] = [Command("help", "the real help", lambda s, a: Outcome([]))]
    registry = find(tools_dir)
    registry.install(box)
    registry.publish_commands(table)
    registry.approve("contrib")

    plugin = registry.find("contrib")
    assert [t.name for t in plugin.tools] == ["ping"]
    assert [c.name for c in plugin.commands] == ["plugin-hi"]
    assert list(plugin.hooks) == ["startup"]
    assert box.get("ping").run({}, ctx).content == "pong"
    assert [c.name for c in table] == ["help", "plugin-hi"], \
        "a plugin command has to reach the live table"
    assert any("not-an-event" in note for note in plugin.notes), \
        "an unknown hook name is a typo the plugin author needs to see"

    problems = registry.fire("startup", state="the-state")

    assert (path.parent / "startup.marker").read_text(encoding="utf-8") == "state"
    assert len(problems) == 1 and "this hook is broken" in problems[0], \
        "one failing hook is named and the others still run"


def test_a_plugin_may_not_shadow_a_built_in_command(tools_dir, home):
    write_plugin(tools_dir, "squatter", SQUATTER)
    table: list[Command] = [Command("help", "the real help", lambda s, a: Outcome([]))]
    registry = find(tools_dir)
    registry.approve("squatter")

    clashes = registry.publish_commands(table)

    assert [c.summary for c in table] == ["the real help"], "the built-in must win"
    assert clashes == ["plugin squatter: /help is already a command, not added"]
    assert clashes[0] in registry.notes


def test_an_unknown_hook_event_is_refused_by_name(tools_dir, home):
    registry = find(tools_dir)
    problems = registry.fire("nonsense")
    assert len(problems) == 1
    assert "nonsense" in problems[0]
    assert all(event in problems[0] for event in EVENTS)


def test_a_tool_name_already_in_the_toolbox_is_reported_not_swallowed(tools_dir, home):
    write_plugin(tools_dir, "greeter", GREETER)
    write_manifest(tools_dir, "impostor", {"name": "greet", "command": ["/bin/echo"]})
    box = Toolbox()
    registry = find(tools_dir)
    registry.approve("greeter")
    registry.approve("impostor")

    clashes = registry.install(box)

    assert len(clashes) == 1 and "greet" in clashes[0]
    assert len(box) == 1, "the loser must not be registered over the winner"


# -- entry points -----------------------------------------------------------


def test_a_pip_installed_package_can_provide_a_plugin(home, monkeypatch, ctx):
    from importlib.metadata import EntryPoint

    module = types.ModuleType("offset_demo_plugin")
    module.PLUGIN_API = PLUGIN_API
    exec(compile(GREETER.replace(
        'Path(__file__).with_name("imported.marker").write_text("yes", encoding="utf-8")', "",
    ), "<demo>", "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, "offset_demo_plugin", module)
    entry = EntryPoint(name="demo", value="offset_demo_plugin", group=ENTRY_POINT_GROUP)
    monkeypatch.setattr(
        "offset.tools.plugins.metadata",
        types.SimpleNamespace(entry_points=lambda group=None: [entry] if group == ENTRY_POINT_GROUP else []),
    )
    box = Toolbox()

    registry = discover(dirs=[], entry_points=True)
    registry.install(box)

    plugin = registry.find("demo")
    assert plugin is not None, "an entry point must be discovered"
    assert plugin.kind == "entrypoint"
    assert plugin.quarantined, "installing a package is not the same as approving it"
    assert box.names() == []

    ok, _ = registry.approve("demo")

    assert ok is True
    assert registry.find("demo").active
    assert box.get("greet").run({"who": "ep"}, ctx).content == "hello ep"


def test_an_entry_point_that_resolves_to_nonsense_is_reported(home, monkeypatch):
    from importlib.metadata import EntryPoint

    module = types.ModuleType("offset_bad_plugin")
    module.thing = 17
    monkeypatch.setitem(sys.modules, "offset_bad_plugin", module)
    entry = EntryPoint(name="bad", value="offset_bad_plugin:thing", group=ENTRY_POINT_GROUP)
    monkeypatch.setattr(
        "offset.tools.plugins.metadata",
        types.SimpleNamespace(entry_points=lambda group=None: [entry]),
    )

    registry = discover(dirs=[], entry_points=True)
    registry.approve("bad")

    plugin = registry.find("bad")
    assert plugin.ok is False
    assert "must resolve to a module" in plugin.error and "int" in plugin.error


def test_an_entry_point_group_that_cannot_be_read_is_reported(home, monkeypatch):
    def boom(group=None):
        raise RuntimeError("metadata is broken")

    monkeypatch.setattr("offset.tools.plugins.metadata", types.SimpleNamespace(entry_points=boom))

    registry = discover(dirs=[], entry_points=True)

    assert len(registry.plugins) == 1
    assert "metadata is broken" in registry.plugins[0].error


# -- the surfaces the shell uses --------------------------------------------


def test_the_report_counts_loaded_untrusted_and_broken(tools_dir, home):
    write_plugin(tools_dir, "greeter", GREETER)
    write_plugin(tools_dir, "broken", BROKEN)
    write_manifest(tools_dir, "sneaky", {"name": "sneak", "command": ["/bin/echo"]})
    registry = find(tools_dir)
    registry.approve("greeter")

    lines = registry.report()

    assert "1 loaded, 1 untrusted, 1 broken" in lines
    assert any(line.startswith("greeter") and "loaded" in line for line in lines)
    assert any(line.startswith("sneaky") and "untrusted" in line for line in lines)
    assert any("trust sneaky" in line for line in lines), "the report must say what to do next"


def test_plugin_tools_only_yields_what_is_trusted(tmp_path, home, monkeypatch):
    workspace = tmp_path / "project"
    (workspace / ".offset" / "tools").mkdir(parents=True)
    write_plugin(workspace / ".offset" / "tools", "greeter", GREETER)
    monkeypatch.setattr(
        "offset.tools.plugins.metadata",
        types.SimpleNamespace(entry_points=lambda group=None: []),
    )

    assert plugin_tools(workspace) == []

    registry = discover(workspace=workspace, entry_points=False)
    registry.approve("greeter")

    assert [t.name for t in plugin_tools(workspace)] == ["greet"]


def test_install_wires_mcp_and_plugins_into_a_built_state(tools_dir, home, monkeypatch):
    write_plugin(tools_dir, "greeter", GREETER)
    write_plugin(tools_dir, "broken", BROKEN)
    monkeypatch.setattr("offset.tools.plugins.default_dirs", lambda workspace=None: [tools_dir])
    from offset.tools import plugins

    # The module-level handle is real state; reverting it keeps the suite order-free.
    monkeypatch.setattr(plugins, "_ACTIVE", None)
    attached: list[Toolbox] = []
    manager = types.SimpleNamespace(attach=lambda box: attached.append(box) or ["mcp echo: clash"])
    state = types.SimpleNamespace(workspace=Path(os.getcwd()), toolbox=Toolbox(), mcp=manager)

    plugins.install(state)

    assert attached == [state.toolbox], "the manager must be given the toolbox to keep in step"
    assert "mcp echo: clash" in plugins.active().notes, "a collision has to survive to /plugins"

    listing = plugins._plugins(state, [])
    assert any("SyntaxError" in line for line in listing.lines)
    assert any("greeter" in line and "untrusted" in line for line in listing.lines)


def test_the_plugins_command_trusts_and_distrusts_a_plugin(tools_dir, home, monkeypatch):
    write_plugin(tools_dir, "greeter", GREETER)
    monkeypatch.setattr("offset.tools.plugins.default_dirs", lambda workspace=None: [tools_dir])
    from offset.tools import plugins

    monkeypatch.setattr(plugins, "_ACTIVE", None)
    state = types.SimpleNamespace(workspace=Path(os.getcwd()), toolbox=Toolbox(), mcp=None)

    # No `install` call: /plugins has to stand on its own.
    assert any("untrusted" in line for line in plugins._plugins(state, []).lines)
    assert plugins.active() is not None, "the registry it built has to be the one it keeps"

    trusted = plugins._plugins(state, ["trust", "greeter"])
    assert trusted.tone == "ok"
    assert "greet" in state.toolbox, "trusting through the command must register the tool"

    assert plugins._plugins(state, ["trusted"]).lines[0].startswith("greeter ")
    assert plugins._plugins(state, ["reload"]).tone == "ok"
    assert "greet" in state.toolbox, "a reload must not lose an approved plugin"
    assert plugins._plugins(state, ["nonsense"]).tone == "err"
    assert plugins._plugins(state, ["trust"]).tone == "err"

    dropped = plugins._plugins(state, ["distrust", "greeter"])
    assert dropped.tone == "ok"
    assert "greet" not in state.toolbox


def test_the_mcp_command_says_what_to_do_when_nothing_is_configured():
    from offset.tools import plugins

    state = types.SimpleNamespace(mcp=None)
    outcome = plugins._mcp(state, [])

    assert any("mcp.json" in line for line in outcome.lines)
    assert plugins._mcp(state, ["reload"]).lines


def test_the_mcp_command_routes_its_subcommands(monkeypatch):
    from offset.tools import plugins

    class FakeManager:
        def __init__(self) -> None:
            self.calls: list[tuple] = []
            self.config = types.SimpleNamespace(errors=[])

        def status(self):
            return [types.SimpleNamespace(name="echo", state="live", tools=2, detail="python s.py")]

        def collisions(self):
            return []

        def config_for(self, name):
            return object() if name == "echo" else None

        def registered(self, name):
            return ["mcp__echo__one", "mcp__echo__two"]

        def reason(self, name):
            return "because"

        def reload(self, name=None):
            self.calls.append(("reload", name))
            return ["mcp echo: connected, 2 tool(s)"]

        def reconnect(self, name):
            self.calls.append(("reconnect", name))
            return True

        def disconnect(self, name):
            self.calls.append(("disconnect", name))

        def offering(self, name):
            self.calls.append(("offering", name))
            return types.SimpleNamespace(ok=True, report=lambda: [f"{name}: 0 resource(s), 0 prompt(s)"])

        def offerings(self):
            return [self.offering("echo")]

        def read_resource(self, name, uri):
            self.calls.append(("read", name, uri))
            return types.SimpleNamespace(ok=True, text="the body", error="")

    manager = FakeManager()
    state = types.SimpleNamespace(mcp=manager)

    assert "echo" in plugins._mcp(state, []).lines[0]

    reload_outcome = plugins._mcp(state, ["reload", "echo"])
    assert reload_outcome.job is not None, "spawning servers is too slow for a keypress"
    assert any("connected" in line for line in reload_outcome.job().lines)

    assert "2 tool(s) registered" in plugins._mcp(state, ["connect", "echo"]).job().lines[0]
    assert "2 tool(s) withdrawn" in plugins._mcp(state, ["disconnect", "echo"]).lines[0]
    assert plugins._mcp(state, ["disconnect", "ghost"]).tone == "err"
    assert plugins._mcp(state, ["connect"]).tone == "err"
    assert plugins._mcp(state, ["resources"]).job().lines == ["echo: 0 resource(s), 0 prompt(s)"]
    assert plugins._mcp(state, ["read", "echo", "mem://x"]).job().lines == ["echo mem://x", "the body"]
    assert plugins._mcp(state, ["read", "echo"]).tone == "err"
    assert plugins._mcp(state, ["wat"]).tone == "err"

    assert manager.calls == [
        ("reload", "echo"),
        ("reconnect", "echo"),
        ("disconnect", "echo"),
        ("offering", "echo"),
        ("read", "echo", "mem://x"),
    ]
