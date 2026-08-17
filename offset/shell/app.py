"""The prompt_toolkit application.

Structure: a ticker, a transcript, an input line, a status bar, and a floating
overlay for modals.  Every region is an ANSI string produced by
`offset.shell.render`, so prompt_toolkit is doing layout and input handling
only — it never decides how anything looks.

The agent runs on a worker thread and pushes events onto a queue.  The UI
thread drains the queue on each refresh, which keeps streaming smooth without
the renderer ever blocking on the network.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import (
    ConditionalContainer,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from offset.core.agent import Agent, AgentConfig, Finished, ToolFinished
from offset.core.entries import CONVERSATIONAL
from offset.core.session import Session
from offset.eggs.catalogue import build_engine
from offset.providers.base import TextDelta, ThinkingDelta
from offset.providers.registry import CONFIG_DIR
from offset.shell import render
from offset.shell.commands import Outcome, Overlay, ShellState, complete, dispatch, resolve_overlay
from offset.tools.base import Toolbox, ToolContext
from offset.tools.builtin import builtin_tools
from offset.tools.custom import default_dirs, discover
from offset.tools.runtime import Approval, Runtime
from offset.ui.tokens import detect_depth
from offset.core import context, permissions, settings
from offset.shell.consent import Consent, decide, permission_badge, render_consent, summary_lines
from offset.tools.agents import subagent_tools
from offset.tools.system import system_tools
from offset.tools.todo import todo_tools
from offset.tools.websearch import web_search_tools
from offset.core.snapshots import capture_all, target_paths
from offset.tools.mcp import Manager as MCPManager
from offset.tools.mcp import load_config as load_mcp_config


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for option in complete(text):
            yield Completion(option, start_position=-len(text))


class Shell:
    """Owns the widgets, the worker thread, and the transient message log."""

    def __init__(self, state: ShellState, *, depth=None) -> None:
        self.state = state
        self.depth = depth or detect_depth()
        self.started = time.monotonic()
        self.messages: list[tuple[str, str]] = []  # (tone, line)
        self.reveal = None
        self.reveal_until = 0.0
        self.live = ""
        self.busy = False
        self.note = ""
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.approval_gate: threading.Event | None = None
        self.approval_answer = False
        #: The scrollable transcript. Without this the history was a fixed tail
        #: with no way to read back.
        self.view = render.Transcript()
        #: Present until the user has chosen a blast radius.  A workspace that
        #: was already granted skips the question entirely.
        self.consent: Consent | None = (
            None if permissions.current(state.workspace) else Consent(workspace=state.workspace)
        )

        self.buffer = Buffer(
            completer=SlashCompleter(),
            complete_while_typing=True,
            multiline=False,
            accept_handler=self._accept,
        )
        self.app = self._build()
        # The policy now has a human behind it, so `safe` mode asks instead of
        # refusing everything.
        self.state.approval.ask = self.ask_approval

    # -- permission consent -----------------------------------------------

    def _consent(self) -> ANSI:
        width, rows = self.size
        return ANSI(render_consent(width, rows, self.state.workspace, self.now(), self.consent))

    def _answer_consent(self, event) -> None:
        """Route one keypress to a scope.  Enter alone must never mean `full`."""
        raw = event.data or ""
        if not raw or not raw.isprintable():
            raw = {"c-m": "enter", "c-j": "enter", "escape": "escape"}.get(
                str(event.key_sequence[0].key), raw
            )
        scope = decide(raw)
        if scope is None:
            return
        self.apply_consent(scope)

    def apply_consent(self, scope: str) -> None:
        """Persist the grant and widen (or keep) the permission boundary."""
        permissions.grant(scope, self.state.workspace)
        context = self.state.agent.runtime.context
        context.root = permissions.root_for(self.state.workspace)
        self.state.approval.mode = permissions.mode_for(self.state.workspace)
        if self.consent is not None:
            self.consent.choice = scope
        for line in summary_lines(scope, self.state.workspace):
            self.messages.append(("err" if scope == "full" else "ok", line))

    # -- geometry ---------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        size = self.app.output.get_size() if hasattr(self, "app") else None
        return (size.columns, size.rows) if size else (100, 34)

    def now(self) -> float:
        return time.monotonic() - self.started

    # -- rendering --------------------------------------------------------

    def _banner(self) -> ANSI:
        return ANSI(render.banner(self.size[0], self.now()))

    def _message_rows(self) -> int:
        """Command output gets as much room as it needs, up to most of the
        screen.  Clipping `/help` to a handful of lines makes it useless."""
        return min(len(self.messages), max(0, self.size[1] - 10))

    def _transcript(self) -> ANSI:
        width, rows = self.size
        height = max(3, rows - 3 - self._message_rows())
        entries = [e for e in self.state.session.transcript() if e.type in CONVERSATIONAL]
        return ANSI(render.transcript(width, height, entries, live=self.live, t=self.now(), view=self.view))

    def _messages(self) -> ANSI:
        rows = self._message_rows()
        recent = self.messages[-rows:] if rows else []
        if not recent:
            return ANSI("")
        tone = recent[-1][0]
        return ANSI(render.message_block(self.size[0], [line for _, line in recent], tone))

    def _prompt(self) -> ANSI:
        return ANSI(render.prompt_row(4, "", busy=self.busy, t=self.now()))

    def _status(self) -> ANSI:
        return ANSI(render.status(self.size[0], self.state, busy=self.busy, t=self.now(), note=self.note))

    def _overlay(self) -> ANSI:
        panel = self.state.overlay
        if panel is None:
            if self.reveal and self.now() < self.reveal_until:
                return ANSI(render.reveal_panel(min(64, self.size[0] - 4), self.reveal, self.now()))
            return ANSI("")
        width, rows = self.size
        return ANSI(render.overlay(min(72, width - 4), min(len(panel.items) + 5, rows - 6), panel, self.now()))

    def _has_overlay(self) -> bool:
        return self.state.overlay is not None or bool(self.reveal and self.now() < self.reveal_until)

    # -- input ------------------------------------------------------------

    def _accept(self, buff: Buffer) -> bool:
        text = buff.text
        buff.reset()
        self.submit(text)
        return False

    def submit(self, text: str) -> None:
        if not text.strip() or self.busy:
            return
        self.state.eggs.touch()
        outcome = dispatch(self.state, text)
        if outcome.handled:
            # Each command replaces the last one's output; stacking it means
            # the useful lines scroll away behind stale ones.
            self.messages.clear()
            self._apply(outcome)
            return
        self._start_turn(text)

    def _apply(self, outcome: Outcome) -> None:
        for line in outcome.lines:
            self.messages.append((outcome.tone, line))
        if outcome.reveal is not None:
            self.reveal = outcome.reveal
            self.reveal_until = self.now() + outcome.reveal.duration
        if outcome.quit:
            self.app.exit()
        if outcome.job is not None:
            self._start_job(outcome.job)

    def _start_job(self, job) -> None:
        """Run slow command work off the UI thread (branch fan-outs, mostly)."""
        self.busy = True

        def work() -> None:
            try:
                self.events.put(("job", job()))
            except Exception as exc:
                self.events.put(RuntimeError(f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(None)

        self.worker = threading.Thread(target=work, name="offset-job", daemon=True)
        self.worker.start()

    # -- approval ---------------------------------------------------------

    def ask_approval(self, tool, args: dict) -> bool:
        """Called on a tool thread; blocks until the UI answers.

        Without this the approval modes are a lie: `safe` would deny silently
        instead of asking.
        """
        gate = threading.Event()
        self.approval_answer = False
        self.approval_gate = gate
        self.state.overlay = Overlay(
            kind="approve",
            title=f"allow {tool.name}?",
            items=[tool.preview(args)[:200], "", "Y allow    N deny    A always allow this tool"],
            payload=tool.name,
        )
        self.app.invalidate()
        if not gate.wait(timeout=180.0):
            self.state.overlay = None
            return False
        return self.approval_answer

    def answer_approval(self, allow: bool, *, remember: bool = False) -> None:
        """Called on the UI thread when the user presses a key."""
        panel = self.state.overlay
        if panel is None or panel.kind != "approve":
            return
        name = str(panel.payload or "")
        if remember and allow:
            self.state.approval.remember(name)
        self.approval_answer = allow
        self.state.overlay = None
        self.messages.append(("ok" if allow else "err", f"{name}: {'allowed' if allow else 'denied'}"))
        gate = self.approval_gate
        self.approval_gate = None
        if gate is not None:
            gate.set()

    # -- the turn ---------------------------------------------------------

    def _start_turn(self, text: str) -> None:
        self.busy = True
        self.live = ""
        self.messages.clear()
        self.view.to_end()  # you asked for this output; show it

        def work() -> None:
            try:
                for event in self.state.agent.run(text):
                    self.events.put(event)
            except Exception as exc:  # never let a worker die silently
                self.events.put(RuntimeError(f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(None)

        self.worker = threading.Thread(target=work, name="offset-turn", daemon=True)
        self.worker.start()

    def drain(self) -> None:
        """Pull whatever the worker produced since the last frame."""
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            if event is None:
                self.busy = False
                self.live = ""
                self.note = ""
                continue
            if isinstance(event, RuntimeError):
                self.messages.append(("err", str(event)))
                continue
            if isinstance(event, tuple) and event and event[0] == "job":
                self.messages.clear()
                self._apply(event[1])
                continue
            if isinstance(event, TextDelta):
                self.live += event.text
            elif isinstance(event, ThinkingDelta):
                self.note = "thinking"
            elif isinstance(event, ToolFinished):
                inv = event.invocation
                self.note = inv.result.display[:60]
                reveal = self.state.eggs.event("tool_call", suppress=not inv.result.ok)
                if reveal:
                    self.reveal, self.reveal_until = reveal, self.now() + reveal.duration
            elif isinstance(event, Finished):
                if event.reason == "cancelled":
                    self.state.eggs.event("turn_cancelled")
                reveal = self.state.eggs.event("turn_finished")
                if reveal:
                    self.reveal, self.reveal_until = reveal, self.now() + reveal.duration

    # -- wiring -----------------------------------------------------------

    def _keys(self) -> KeyBindings:
        keys = KeyBindings()

        def deciding() -> bool:
            return self.consent is not None and self.consent.choice is None

        pending = Condition(deciding)

        @keys.add("<any>", filter=pending, eager=True)
        @keys.add("enter", filter=pending, eager=True)
        @keys.add("escape", filter=pending, eager=True)
        def _(event):
            self._answer_consent(event)

        @keys.add("c-d")
        def _(event):
            event.app.exit()

        def approving() -> bool:
            panel = self.state.overlay
            return panel is not None and panel.kind == "approve"

        @keys.add("c-c")
        def _(event):
            if approving():
                self.answer_approval(False)
            elif self.busy:
                self.state.agent.runtime.cancel()
                self.messages.append(("err", "cancelling"))
            elif self.state.overlay is not None:
                self._apply(resolve_overlay(self.state, self.state.overlay, accepted=False))
            else:
                self.buffer.reset()

        @keys.add("escape", eager=True)
        def _(event):
            if approving():
                self.answer_approval(False)
            elif self.state.overlay is not None:
                self._apply(resolve_overlay(self.state, self.state.overlay, accepted=False))

        @keys.add("up")
        def _(event):
            panel = self.state.overlay
            if panel is not None:
                panel.move(-1)
            else:
                self.buffer.history_backward()

        @keys.add("down")
        def _(event):
            panel = self.state.overlay
            if panel is not None:
                panel.move(1)
            else:
                self.buffer.history_forward()

        @keys.add("enter")
        def _(event):
            panel = self.state.overlay
            if approving():
                self.answer_approval(True)
            elif panel is not None:
                self._apply(resolve_overlay(self.state, panel, accepted=True))
            else:
                self.buffer.validate_and_handle()

        @keys.add("<any>")
        def _(event):
            """Typing feeds the overlay when one is open, otherwise the buffer."""
            panel = self.state.overlay
            data = event.data
            if approving():
                answer = data.lower()
                if answer == "y":
                    self.answer_approval(True)
                elif answer == "n":
                    self.answer_approval(False)
                elif answer == "a":
                    self.answer_approval(True, remember=True)
                return
            if panel is None:
                self.state.eggs.touch()
                reveal = self.state.eggs.key(data)
                if reveal:
                    self.reveal, self.reveal_until = reveal, self.now() + reveal.duration
                self.buffer.insert_text(data)
            elif panel.kind == "login" and data.isprintable():
                panel.buffer += data

        def scroll_by(lines: int) -> None:
            width, rows = self.size
            entries = [e for e in self.state.session.transcript() if e.type in CONVERSATIONAL]
            height = max(3, rows - 3 - self._message_rows())
            total = len(self.view.lines(width, entries, self.live))
            self.view.scroll(lines, total, height)

        @keys.add("pageup")
        def _(event):
            scroll_by(max(1, self.size[1] - 6))

        @keys.add("pagedown")
        def _(event):
            scroll_by(-max(1, self.size[1] - 6))

        @keys.add("s-up")
        def _(event):
            scroll_by(1)

        @keys.add("s-down")
        def _(event):
            scroll_by(-1)

        @keys.add("end")
        def _(event):
            self.view.to_end()

        @keys.add("backspace")
        def _(event):
            panel = self.state.overlay
            if panel is not None and panel.kind == "login":
                panel.buffer = panel.buffer[:-1]
            else:
                self.buffer.delete_before_cursor()

        return keys

    def _build(self) -> Application:
        body = HSplit([
            Window(FormattedTextControl(self._banner), height=1),
            Window(FormattedTextControl(self._transcript)),
            Window(FormattedTextControl(self._messages),
                   height=lambda: Dimension.exact(self._message_rows())),
            VSplit([
                Window(FormattedTextControl(self._prompt), width=4),
                Window(BufferControl(buffer=self.buffer), height=1),
            ], height=1),
            Window(FormattedTextControl(self._status), height=1),
        ])
        deciding = Condition(lambda: self.consent is not None and self.consent.choice is None)
        root = FloatContainer(
            content=HSplit([
                # Until a blast radius is chosen, the consent screen IS the app.
                ConditionalContainer(Window(FormattedTextControl(self._consent)), filter=deciding),
                ConditionalContainer(body, filter=~deciding),
            ]),
            floats=[Float(Window(FormattedTextControl(self._overlay)), top=3, left=4)],
        )
        app = Application(
            layout=Layout(root, focused_element=self.buffer),
            key_bindings=self._keys(),
            full_screen=True,
            refresh_interval=0.08,
            mouse_support=False,
        )
        return app

    def run(self) -> None:
        stop = threading.Event()

        def pump() -> None:
            while not stop.is_set():
                self.drain()
                reveal = self.state.eggs.tick()
                if reveal:
                    self.reveal, self.reveal_until = reveal, self.now() + reveal.duration
                self.app.invalidate()
                time.sleep(0.08)

        ticker = threading.Thread(target=pump, name="offset-ui", daemon=True)
        ticker.start()
        try:
            self.app.run()
        finally:
            stop.set()
            self.state.eggs.save()
            self.state.session.close()


# -- construction -----------------------------------------------------------


def build_state(workspace: Path | str = ".", *, model: str | None = None, approval: str | None = None) -> ShellState:
    """Assemble a session, every tool, the eggs, and an agent."""
    workspace = Path(workspace).resolve()
    settings.configure(workspace)
    approval = approval or settings.get("tools.approvalMode", "auto-edit")
    # An explicit flag beats configuration; configuration beats the built-in.
    model = model or settings.get("model.default", None) or "mock"
    home = CONFIG_DIR  # honours OFFSET_HOME, so a test can isolate everything
    session = Session.create(home / "sessions")

    # Every tool ships enabled; what varies is whether a call is allowed.
    # `system_tools()` is the whole-machine set and already carries `document`.
    toolbox = Toolbox([
        *builtin_tools(),
        *system_tools(),
        *todo_tools(home / "todo"),
        *web_search_tools(),
        *subagent_tools(),
    ])
    found = discover(default_dirs(workspace))
    for tool in found:
        try:
            toolbox.register(tool)
        except ValueError:
            pass

    eggs = build_engine(home / "eggs.json")
    if found.tools:
        eggs.event("custom_tool_loaded", count=len(found.tools))

    # A grant from a previous run is honoured; a fresh workspace gets the
    # startup question instead, and until it is answered the boundary is the
    # workspace. `root=None` only ever comes from an explicit grant.
    grant = permissions.current(workspace)
    tool_context = ToolContext(
        cwd=workspace,
        root=permissions.root_for(workspace) if grant else workspace,
        timeout=120.0,
    )
    # `ask` is attached by the Shell, which is the only thing that can put a
    # question on screen; until then a dangerous call simply has no approver.
    policy = Approval(mode=permissions.mode_for(workspace, approval) if grant else approval)
    def snapshot(tool, args) -> None:
        """Record what a writing tool is about to overwrite, so /rewind works."""
        paths = target_paths(args)
        if paths:
            capture_all(session, paths, tool=tool.name, root=workspace)

    runtime = Runtime(toolbox, tool_context, policy, before_write=snapshot)
    # Project instructions belong in the system prompt, not in the first user
    # message: they are standing orders, not part of the conversation.
    instructions = context.assemble(workspace)
    system = f"{SYSTEM_PROMPT}\n\n{instructions}" if instructions else SYSTEM_PROMPT
    agent = Agent(session, runtime, AgentConfig(model=model, system=system))

    # MCP servers are optional and must never delay startup: a server that is
    # slow or absent costs its own tools, nothing else.
    mcp_manager = None
    try:
        mcp_config = load_mcp_config(workspace)
        if mcp_config.servers:
            mcp_manager = MCPManager(mcp_config)
            mcp_manager.connect_all()
            for remote in mcp_manager.tools():
                try:
                    toolbox.register(remote)
                except ValueError:
                    pass
    except Exception:
        mcp_manager = None

    state = ShellState(session, agent, toolbox, policy, eggs, workspace)
    state.mcp = mcp_manager
    return state


SYSTEM_PROMPT = """You are offset, a terminal coding agent.

Be terse. Prefer doing over describing. Use tools rather than guessing about
files. When several approaches are plausible, say so in one line and pick one.
Never claim a command succeeded without running it."""


def main(workspace: str = ".", model: str | None = None, approval: str | None = None) -> int:
    Shell(build_state(workspace, model=model, approval=approval)).run()
    return 0
