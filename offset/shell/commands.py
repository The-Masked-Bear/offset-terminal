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
from offset.core import compaction, context, snapshots
from offset.ui import theme
from offset.core.agent import to_messages
from offset.core.branches import BranchRun, approaches, run_branches
from offset.core.entries import CONVERSATIONAL, MESSAGE
from offset.core.multimodel import Ensemble, Seat
from offset.core.session import Session
from offset.eggs.engine import EggEngine, Reveal
from offset.providers import auth, oauth
from offset.providers.registry import (
    MODELS,
    ModelInfo,
    available,
    credential,
    info,
    provider_for,
    search,
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
    mcp: Any = None

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
    """Two columns: the list outgrew a single one, and then outgrew the pane."""
    names = [f"/{c.name}" for c in COMMANDS]
    width = max(len(n) for n in names) + 1
    cells = [f"{n:<{width}} {c.summary}" for n, c in zip(names, COMMANDS)]
    half = (len(cells) + 1) // 2
    left, right = cells[:half], cells[half:]
    gutter = max(len(row) for row in left) + 3
    lines = [
        f"{a:<{gutter}}{b}".rstrip() if b else a
        for a, b in zip(left, right + [""] * (len(left) - len(right)))
    ]
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


#: Providers a person can plausibly hold an account with.
LOGIN_TARGETS: tuple[str, ...] = ("anthropic", "openai", "google", "deepseek", "openrouter")


def _login(state: ShellState, args: list[str]) -> Outcome:
    """No argument opens the account picker; an argument goes straight there."""
    if args:
        return _login_provider(state, args[0].lower())

    ready = auth.oauth_providers()
    held = {c.provider: c for c in auth.accounts()}
    items, notes = [], []
    for name in LOGIN_TARGETS:
        items.append(name)
        cred = held.get(name)
        if cred is not None:
            notes.append(cred.label().split(": ", 1)[-1])
        elif name in ready and not auth.missing_config(name):
            notes.append("sign in with browser")
        else:
            notes.append("paste an api key")
    overlay = Overlay(kind="account", title="sign in", items=items, notes=notes, payload=list(LOGIN_TARGETS))
    state.overlay = overlay
    return Outcome(overlay=overlay)


def _login_provider(state: ShellState, provider: str) -> Outcome:
    """Browser flow when the provider offers one, otherwise a masked field."""
    if provider in auth.oauth_providers() and not auth.missing_config(provider):
        def job() -> Outcome:
            try:
                cred = auth.login_browser(provider, announce=lambda p: None)
            except Exception as exc:
                return Outcome.error(f"{provider} sign-in failed: {exc}", "try /login again, or paste a key")
            return Outcome([f"signed in: {cred.label()}"], TONE_OK)

        entry = oauth.app(provider)
        return Outcome(
            [f"opening your browser to sign in to {provider}", entry.note or "", "waiting for the callback..."],
            TONE_INFO,
            job=job,
        )

    absent = auth.missing_config(provider) if provider in auth.oauth_providers() else ()
    overlay = Overlay(kind="login", title=f"{provider} api key", secret=True, payload=provider)
    state.overlay = overlay
    lines = []
    if absent:
        lines = [f"{provider} browser sign-in needs {', '.join(absent)} in ~/.offset/config.json"]
    return Outcome(lines, TONE_INFO, overlay=overlay)


def _logout(state: ShellState, args: list[str]) -> Outcome:
    provider = args[0].lower() if args else info(state.model).provider
    if auth.forget(provider):
        return Outcome([f"signed out of {provider}"], TONE_OK)
    return Outcome.error(f"no stored credential for {provider}")


def _accounts(state: ShellState, args: list[str]) -> Outcome:
    held = auth.accounts()
    if not held:
        return Outcome(["no accounts yet. /login to add one."], TONE_INFO)
    lines = [c.label() for c in held]
    lines.append("")
    lines.append("browser sign-in available: " + (", ".join(
        p for p in auth.oauth_providers() if not auth.missing_config(p)
    ) or "none configured"))
    return Outcome(lines, TONE_INFO)


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


def _compact(state: ShellState, args: list[str]) -> Outcome:
    """Summarise the old part of the conversation so a long session can go on."""
    messages = to_messages(state.session.transcript())
    budget = compaction.budget_for(state.model)
    tokens = compaction.estimate_tokens(messages)
    forced = bool(args and args[0] == "now")
    if not forced and not compaction.needs_compaction(messages, budget):
        return Outcome([
            f"about {tokens:,} tokens of roughly {budget:,}; nothing to compact yet",
            "force it with /compact now",
        ], TONE_INFO)

    def job() -> Outcome:
        try:
            report = compaction.compact(
                state.session,
                compaction.model_summariser(state.model),
                budget=budget,
                threshold=0.0 if forced else 0.8,
            )
        except Exception as exc:
            return Outcome.error(f"could not compact: {exc}")
        after = compaction.estimate_tokens(to_messages(state.session.transcript()))
        return Outcome([
            f"compacted: about {tokens:,} tokens -> {after:,}",
            "the original entries are still on disk and still reachable in /tree",
        ], TONE_OK)

    return Outcome([f"summarising {tokens:,} tokens of history..."], TONE_INFO, job=job)


def _rewind(state: ShellState, args: list[str]) -> Outcome:
    """Put files back the way they were at a point in this session."""
    marks = snapshots.records(state.session)
    if not marks:
        return Outcome(["nothing has been snapshotted in this session yet"], TONE_INFO)
    if not args:
        lines = [f"{i + 1}. {m.path}  ({m.tool}, {m.skipped or 'stored'})" for i, m in enumerate(marks[-12:])]
        lines.append("")
        lines.append("rewind to one with /rewind <number>")
        return Outcome(lines, TONE_INFO)
    try:
        index = int(args[0]) - 1
    except ValueError:
        return Outcome.error("usage: /rewind <number from the /rewind list>")
    recent = marks[-12:]
    if not 0 <= index < len(recent):
        return Outcome.error(f"pick a number between 1 and {len(recent)}")
    # `restore` puts files back to how they were *as of* an entry, using the
    # snapshots taken after it. To undo snapshot N we therefore anchor on the
    # entry just before it.
    chosen = recent[index]
    entries = list(state.session.all_entries())
    at = next((i for i, e in enumerate(entries) if e.id == chosen.id), 0)
    anchor = entries[at - 1].id if at > 0 else None  # None = before everything
    got = snapshots.restore(state.session, anchor, root=state.workspace)
    lines = [f"restored {len(got.restored)} file(s)"] + [f"  {p}" for p in got.restored[:10]]
    for path, why in got.failed[:6]:
        lines.append(f"  could not restore {path}: {why}")
    return Outcome(lines, TONE_OK if got.ok else TONE_ERR)


def _mcp(state: ShellState, args: list[str]) -> Outcome:
    """What MCP servers are configured, and whether they answered."""
    manager = state.mcp
    if manager is None:
        return Outcome([
            "no MCP servers configured",
            "add them to .offset/mcp.json or ~/.offset/mcp.json",
        ], TONE_INFO)
    rows = []
    for status in manager.status():
        rows.append(f"{status.name:<16} {status.state:<12} {status.tools:>3} tools  {status.detail[:40]}")
    for problem in manager.config.errors[:5]:
        rows.append(f"config: {problem}")
    remote = [t.name for t in state.toolbox if t.name.startswith("mcp__")]
    rows.append("")
    rows.append(f"{len(remote)} remote tool(s) registered")
    return Outcome(rows or ["no servers"], TONE_INFO)


def _sessions(state: ShellState, args: list[str]) -> Outcome:
    """Recent sessions, newest first."""
    listed = Session.list(state.session.path.parent)
    if not listed:
        return Outcome(["no other sessions"], TONE_INFO)
    lines = []
    for i, meta in enumerate(listed[:15], 1):
        here = "*" if Path(meta.path) == state.session.path else " "
        lines.append(f"{here}{i:>3}. {meta.messages:>4} msgs  {meta.first_line[:44]}")
    return Outcome(lines, TONE_INFO)


def _context(state: ShellState, args: list[str]) -> Outcome:
    """Which project instruction files offset is obeying."""
    lines = context.summary(state.workspace)
    block = context.assemble(state.workspace)
    if block:
        lines = [*lines, "", f"{len(block)} characters appended to the system prompt"]
    return Outcome(lines, TONE_INFO)


def _theme(state: ShellState, args: list[str]) -> Outcome:
    """Switch palette, or say which one is in force."""
    if not args:
        current = theme.active()
        lines = [f"theme: {getattr(current, 'name', 'default')}",
                 "available: " + ", ".join(theme.names()),
                 f"a custom palette goes in {theme.path()}"]
        return Outcome(lines, TONE_INFO)
    problem = theme.use(args[0].lower())
    if problem:
        return Outcome.error(problem)
    return Outcome([f"theme: {args[0].lower()}", "redraw is immediate"], TONE_OK)


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
    Command("login", "sign in with your account, or paste an API key", _login, usage="/login [provider]"),
    Command("logout", "forget a stored credential", _logout, usage="/logout [provider]"),
    Command("accounts", "which accounts offset can use", _accounts),
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
    Command("sessions", "recent sessions, newest first", _sessions),
    Command("compact", "summarise old history to free up context", _compact, usage="/compact [now]"),
    Command("rewind", "restore files to an earlier point", _rewind, usage="/rewind [n]"),
    Command("mcp", "MCP servers and their remote tools", _mcp),
    Command("context", "project instruction files in force", _context),
    Command("theme", "switch palette", _theme, usage="/theme [name]"),
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

    if overlay.kind == "account":
        targets: Sequence[str] = overlay.payload or []
        if not targets:
            return Outcome.error("no providers to choose from")
        return _login_provider(state, targets[overlay.selected])

    if overlay.kind == "login":
        provider = str(overlay.payload or "")
        if not overlay.buffer:
            return Outcome.error("no key entered")
        try:
            cred = auth.login_api_key(provider, overlay.buffer)
        except auth.AuthError as exc:
            return Outcome.error(str(exc))
        return Outcome([
            f"stored a key for {provider}",
            "it lives in ~/.offset/credentials.json, mode 600",
            cred.label(),
        ], TONE_OK)

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
