"""Slash commands.

Kept deliberately free of prompt_toolkit: a command takes state and arguments
and returns an `Outcome` describing what to show.  That makes every command
testable without a terminal, and it means the same command set could drive a
different front end later without being rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from offset.core.agent import Agent
from offset.core.branches import BranchRun, approaches, run_branches
from offset.core.entries import CONVERSATIONAL, MESSAGE
from offset.core.multimodel import Ensemble, Seat
from offset.core.session import Session
from offset.eggs.engine import EggEngine, Reveal
from offset.providers.registry import (
    MODELS,
    ModelInfo,
    available,
    credential,
    forget_credential,
    info,
    provider_for,
    search,
    store_credential,
)
from offset.tools.base import Danger, Toolbox
from offset.tools.runtime import Approval

TONE_OK = "ok"
TONE_ERR = "err"
TONE_INFO = "info"


@dataclass(slots=True)
class Overlay:
    """A modal panel the app should draw over the transcript."""

    kind: str  # model | login | trophies | tree | none
    title: str = ""
    items: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    selected: int = 0
    buffer: str = ""
    secret: bool = False
    payload: Any = None

    def move(self, delta: int) -> None:
        if self.items:
            self.selected = (self.selected + delta) % len(self.items)


@dataclass(slots=True)
class Outcome:
    lines: list[str] = field(default_factory=list)
    tone: str = "plain"
    overlay: Overlay | None = None
    reveal: Reveal | None = None
    quit: bool = False
    handled: bool = True
    #: Work too slow for the keypress that asked for it.  The app runs this on
    #: a worker thread and applies whatever Outcome comes back.
    job: Callable[[], "Outcome"] | None = None

    @classmethod
    def say(cls, *lines: str, tone: str = "plain") -> "Outcome":
        return cls(list(lines), tone)

    @classmethod
    def error(cls, *lines: str) -> "Outcome":
        return cls(list(lines), TONE_ERR)

    @classmethod
    def miss(cls) -> "Outcome":
        return cls(handled=False)


@dataclass(slots=True)
class ShellState:
    session: Session
    agent: Agent
    toolbox: Toolbox
    approval: Approval
    eggs: EggEngine
    workspace: Path
    ensemble: Ensemble | None = None
    overlay: Overlay | None = None
    verify_command: str | None = None
    spec_run: BranchRun | None = None

    @property
    def model(self) -> str:
        return self.agent.config.model


Runner = Callable[[ShellState, list[str]], Outcome]


@dataclass(slots=True)
class Command:
    name: str
    summary: str
    run: Runner
    usage: str = ""
    aliases: tuple[str, ...] = ()


# -- individual commands ----------------------------------------------------


def _help(state: ShellState, args: list[str]) -> Outcome:
    width = max(len(c.name) for c in COMMANDS) + 2
    lines = [f"/{c.name:<{width}}{c.summary}" for c in COMMANDS]
    found, total = state.eggs.progress()
    lines += ["", f"{found}/{total} easter eggs found. Some of them are commands. Keep typing."]
    return Outcome(lines, TONE_INFO)


def _model(state: ShellState, args: list[str]) -> Outcome:
    """No argument opens the picker; an argument switches immediately."""
    if args:
        wanted = " ".join(args)
        matches = search(wanted)
        chosen = matches[0].id if matches else wanted
        state.agent.config.model = chosen
        state.agent._meta = None  # force re-resolution against the new id
        state.session.append("model_change", {"model": chosen})
        state.eggs.event("model_changed", model=chosen)
        meta = info(chosen)
        return Outcome([
            f"model: {meta.label}",
            f"provider {meta.provider}, context {meta.context:,}, max output {meta.max_output:,}",
        ], TONE_OK)

    catalogue = list(MODELS)
    ready = {m.id for m in available()}
    overlay = Overlay(
        kind="model",
        title="model",
        items=[m.label for m in catalogue],
        notes=[(m.role_hint or "-") + ("" if m.id in ready else " (no key)") for m in catalogue],
        selected=next((i for i, m in enumerate(catalogue) if m.id == state.model), 0),
        payload=catalogue,
    )
    state.overlay = overlay
    return Outcome(overlay=overlay)


def _models(state: ShellState, args: list[str]) -> Outcome:
    ready = {m.id for m in available()}
    lines = [
        f"{'*' if m.id == state.model else ' '} {m.label:<22} {m.provider:<11} "
        f"{'ready' if m.id in ready else 'no key':<7} {m.role_hint}"
        for m in MODELS
    ]
    return Outcome(lines, TONE_INFO)


def _login(state: ShellState, args: list[str]) -> Outcome:
    provider = args[0].lower() if args else info(state.model).provider
    overlay = Overlay(kind="login", title=f"{provider} api key", secret=True, payload=provider)
    state.overlay = overlay
    return Outcome(overlay=overlay)


def _logout(state: ShellState, args: list[str]) -> Outcome:
    provider = args[0].lower() if args else info(state.model).provider
    if forget_credential(provider):
        return Outcome([f"forgot the stored key for {provider}"], TONE_OK)
    return Outcome.error(f"no stored key for {provider}")


#: A missing entry here used to crash `/tools` outright, so the lookup falls
#: back to the enum's own name rather than raising on a tier added later.
DANGER_LABEL: dict[Danger, str] = {
    Danger.SAFE: "safe",
    Danger.WRITE: "write",
    Danger.DESTRUCTIVE: "danger",
    Danger.FULL: "SYSTEM",
}


def _tools(state: ShellState, args: list[str]) -> Outcome:
    rows = []
    for tool in sorted(state.toolbox, key=lambda t: (t.danger, t.name)):
        mark = DANGER_LABEL.get(tool.danger, tool.danger.name.lower())
        parallel = "parallel" if tool.parallel_safe else "serial"
        rows.append(f"{tool.name:<12} {mark:<7} {parallel:<9} {tool.description[:58]}")
    rows.append("")
    scope = "whole machine" if state.agent.runtime.context.root is None else "this workspace"
    rows.append(
        f"{len(state.toolbox)} tools, all enabled. approval mode: {state.approval.mode}, "
        f"reach: {scope}"
    )
    return Outcome(rows, TONE_INFO)


def _approve(state: ShellState, args: list[str]) -> Outcome:
    if not args:
        return Outcome([f"approval mode: {state.approval.mode}", "modes: safe, auto-edit, yolo"], TONE_INFO)
    mode = args[0].lower()
    if mode not in ("safe", "auto-edit", "yolo"):
        return Outcome.error(f"unknown mode {mode!r}; pick safe, auto-edit or yolo")
    state.approval.mode = mode  # type: ignore[assignment]
    note = {
        "safe": "reads run freely; everything else asks",
        "auto-edit": "reads and writes run; shell and network ask",
        "yolo": "nothing asks. you were warned",
    }[mode]
    return Outcome([f"approval mode: {mode}", note], TONE_OK if mode != "yolo" else TONE_ERR)


def _tree(state: ShellState, args: list[str]) -> Outcome:
    rows = state.session.rows(include=CONVERSATIONAL)
    if not rows:
        return Outcome(["the session tree is empty"], TONE_INFO)
    items, notes, active_index = [], [], 0
    for i, (depth, entry, active, label) in enumerate(rows):
        prefix = "  " * min(depth, 6)
        tag = f"[{label}] " if label else ""
        items.append(f"{prefix}{tag}{entry.summary(46)}")
        notes.append("active" if active else "")
        if active:
            active_index = i
    overlay = Overlay(kind="tree", title="session tree", items=items, notes=notes,
                      selected=active_index, payload=[e.id for _, e, _, _ in rows])
    state.overlay = overlay
    return Outcome(overlay=overlay)


def _branch(state: ShellState, args: list[str]) -> Outcome:
    users = [e for e in state.session.ancestry() if e.type == MESSAGE and e.role == "user"]
    if len(users) < 2:
        return Outcome.error("nothing to branch from yet")
    target = users[-2]
    state.session.branch(target.parent)
    state.eggs.event("tree_navigated")
    return Outcome([f"branched from: {target.summary(60)}", "the old branch is still in the tree"], TONE_OK)


def _fork(state: ShellState, args: list[str]) -> Outcome:
    forked = state.session.fork()
    state.eggs.event("session_forked")
    return Outcome([f"forked to {forked.path.name}", "this session is unchanged"], TONE_OK)


def _session(state: ShellState, args: list[str]) -> Outcome:
    s = state.session
    messages = sum(1 for e in s.all_entries() if e.type == MESSAGE)
    return Outcome([
        f"id       {s.id}",
        f"file     {s.path}",
        f"entries  {len(s)} ({messages} messages)",
        f"leaf     {s.leaf or '(root)'}",
        f"roots    {len(s.roots())}",
        f"skipped  {s.skipped_lines} corrupt line(s)",
    ], TONE_INFO)


def _trophies(state: ShellState, args: list[str]) -> Outcome:
    trophies = state.eggs.trophies()
    found = [(e, ok) for e, ok in trophies if ok]
    items = [f"{'x' if ok else '?'}  {egg.name if ok else '???'}" + (f"   {egg.hint}" if ok and egg.hint else "")
             for egg, ok in trophies]
    overlay = Overlay(kind="trophies", title=f"eggs {len(found)}/{len(trophies)}", items=items)
    state.overlay = overlay
    return Outcome(overlay=overlay)


def _spec(state: ShellState, args: list[str]) -> Outcome:
    """Really run N approaches in isolated worktrees, then rank them."""
    if not args:
        if state.spec_run is not None:
            return Outcome(state.spec_run.report(), TONE_INFO)
        return Outcome.error("usage: /spec <how many> <task>", "example: /spec 3 make the parser faster")
    try:
        count = int(args[0])
        task = " ".join(args[1:])
    except ValueError:
        count, task = 3, " ".join(args)
    if not task:
        return Outcome.error("give the branches something to attempt")
    count = max(2, min(count, 6))
    models = [s.model for s in state.ensemble] if state.ensemble else None

    def job() -> Outcome:
        run = run_branches(
            task, count,
            workspace=state.workspace,
            config=state.agent.config,
            models=models,
            verify_command=state.verify_command,
            keep=True,
        )
        state.spec_run = run
        for _ in run.attempts:
            state.eggs.event("branch_created")
        states = [a.state for a in run.attempts]
        if "pass" in states:
            state.eggs.event("branch_passed")
        if states and set(states) == {"pass"}:
            state.eggs.event("all_branches_passed")
        return Outcome(run.report(), TONE_OK)

    plan = approaches(count, task, models)
    return Outcome(
        [f"running {count} branches for: {task}"]
        + [f"  {a.name:<10} {(a.model or state.model)}" for a in plan]
        + ["", "each gets its own worktree; nothing touches your files until /adopt"],
        TONE_INFO,
        job=job,
    )


def _adopt(state: ShellState, args: list[str]) -> Outcome:
    run = state.spec_run
    if run is None or not run.attempts:
        return Outcome.error("no branch results yet; run /spec first")
    ranked = run.ranked
    try:
        index = int(args[0]) - 1 if args else 0
    except ValueError:
        return Outcome.error("usage: /adopt <number from the /spec list>")
    if not 0 <= index < len(ranked):
        return Outcome.error(f"pick a number between 1 and {len(ranked)}")
    attempt = ranked[index]
    if attempt.error:
        return Outcome.error(f"{attempt.approach.name} failed; adopting it would apply nothing useful")
    ok, message = run.speculation.adopt(attempt)
    if not ok:
        return Outcome.error(f"could not apply {attempt.approach.name}: {message}")
    return Outcome([
        f"adopted {attempt.approach.name} ({attempt.churn} lines)",
        message,
        "the other branches are still on disk; /discard removes them",
    ], TONE_OK)


def _discard(state: ShellState, args: list[str]) -> Outcome:
    run = state.spec_run
    if run is None or run.speculation is None:
        return Outcome.error("nothing to discard")
    run.speculation.keep = False
    run.speculation.cleanup(run.attempts)
    state.spec_run = None
    return Outcome(["removed the branch worktrees"], TONE_OK)


def _verify(state: ShellState, args: list[str]) -> Outcome:
    if not args:
        return Outcome([
            f"verification command: {state.verify_command or '(none - branches will be unranked)'}",
            "set one with /verify pytest -q",
        ], TONE_INFO)
    state.verify_command = " ".join(args)
    return Outcome([f"verification command: {state.verify_command}"], TONE_OK)


def _usage(state: ShellState, args: list[str]) -> Outcome:
    meta = info(state.model)
    key = credential(provider_for(meta.provider))
    return Outcome([
        f"model     {meta.label} ({meta.provider})",
        f"key       {'present' if key else 'missing'}",
        f"tools     {len(state.toolbox)} enabled",
        f"approval  {state.approval.mode}",
        f"workspace {state.workspace}",
    ], TONE_INFO)


def _clear(state: ShellState, args: list[str]) -> Outcome:
    state.session.reset_leaf()
    state.eggs.event("tree_navigated")
    return Outcome(["context cleared; the history is still on disk"], TONE_OK)


def _quit(state: ShellState, args: list[str]) -> Outcome:
    return Outcome(["bye"], quit=True)


COMMANDS: list[Command] = [
    Command("help", "this list", _help, aliases=("?",)),
    Command("model", "switch model, or open the picker", _model, usage="/model [name]"),
    Command("models", "list every model and whether it is usable", _models),
    Command("login", "store an API key (masked input)", _login, usage="/login [provider]"),
    Command("logout", "forget a stored API key", _logout, usage="/logout [provider]"),
    Command("tools", "every tool, its danger class and concurrency", _tools),
    Command("approve", "set the approval mode", _approve, usage="/approve safe|auto-edit|yolo"),
    Command("tree", "navigate the session tree", _tree),
    Command("branch", "reopen the previous user message as a new branch", _branch),
    Command("fork", "copy this session to a new file", _fork),
    Command("spec", "really run several approaches in parallel worktrees", _spec, usage="/spec <n> <task>"),
    Command("adopt", "apply one branch's changes to your workspace", _adopt, usage="/adopt <n>"),
    Command("discard", "delete the branch worktrees", _discard),
    Command("verify", "the command branches must pass", _verify, usage="/verify pytest -q"),
    Command("session", "where this session lives and how big it is", _session),
    Command("eggs", "the trophy room", _trophies, aliases=("trophies",)),
    Command("usage", "current model, key, tools, approval", _usage),
    Command("clear", "drop the context, keep the history", _clear),
    Command("quit", "leave", _quit, aliases=("exit",)),
]

BY_NAME: dict[str, Command] = {}
for _command in COMMANDS:
    BY_NAME[_command.name] = _command
    for _alias in _command.aliases:
        BY_NAME[_alias] = _command


def complete(prefix: str) -> list[str]:
    """Completions for the input box."""
    stem = prefix.lstrip("/").lower()
    return sorted({f"/{c.name}" for c in COMMANDS if c.name.startswith(stem)})


def dispatch(state: ShellState, text: str) -> Outcome:
    """Route a line of input.

    Order matters: real commands win, then easter eggs, then nothing — an egg
    must never shadow a command the user actually needs.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        reveal = state.eggs.command(stripped) if stripped and " " not in stripped else None
        return Outcome(reveal=reveal) if reveal else Outcome.miss()

    parts = stripped[1:].split()
    if not parts:
        return Outcome.miss()
    name, args = parts[0].lower(), parts[1:]

    command = BY_NAME.get(name)
    if command is not None:
        return command.run(state, args)

    reveal = state.eggs.command(name)
    if reveal is not None:
        return Outcome(reveal=reveal)
    guesses = complete(name)
    hint = f"  did you mean {guesses[0]}?" if guesses else ""
    return Outcome.error(f"unknown command /{name}{hint}", "try /help")


# -- overlay resolution -----------------------------------------------------


def resolve_overlay(state: ShellState, overlay: Overlay, *, accepted: bool) -> Outcome:
    """Apply whatever a modal was collecting once the user commits or cancels."""
    state.overlay = None
    if not accepted:
        return Outcome()  # dismissing a modal is not news

    if overlay.kind == "model":
        catalogue: Sequence[ModelInfo] = overlay.payload or []
        if not catalogue:
            return Outcome.error("no models to choose from")
        chosen = catalogue[overlay.selected]
        return _model(state, [chosen.id])

    if overlay.kind == "login":
        provider = str(overlay.payload or "")
        if not overlay.buffer:
            return Outcome.error("no key entered")
        store_credential(provider, overlay.buffer)
        return Outcome([f"stored a key for {provider}", "it lives in ~/.offset/credentials.json, mode 600"], TONE_OK)

    if overlay.kind == "tree":
        ids: Sequence[str] = overlay.payload or []
        if not ids:
            return Outcome.error("nothing to jump to")
        target = ids[overlay.selected]
        entry = state.session.entry(target)
        state.session.branch(entry.parent if entry and entry.role == "user" else target)
        state.eggs.event("tree_navigated")
        return Outcome([f"moved to: {entry.summary(60) if entry else target}"], TONE_OK)

    return Outcome()
