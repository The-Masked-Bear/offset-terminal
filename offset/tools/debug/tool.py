"""The model-facing debugger.

A debugger is worth having here for one reason: it replaces guessing with
observation.  Asked why a function returns the wrong value, a model without one
adds print statements, re-runs, and reasons about the output; with one it stops
on the line and reads the actual locals.  The second is shorter and correct.

Two tools, split by danger.  `debug` starts and drives a debuggee, which runs
arbitrary code, so it is DESTRUCTIVE and serial.  `debug_inspect` only reads an
already-stopped session - stack, scopes, variables, evaluate - and is SAFE, so
looking at state does not need the approval that starting a process does.

The session is deliberately global to the process and singular.  Two debuggees
under one agent means two sets of breakpoints, two `stopped` streams and an
ambiguous "continue"; the DAP spec allows it and it helps nobody here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from offset.tools.base import Danger, Tool, ToolContext, ToolResult
from offset.tools.debug.adapters import (
    Launch,
    choose,
    language_for,
    load_config,
    report as adapter_report,
)
from offset.tools.debug.client import (
    DebugError,
    RequestFailed,
    RequestTimeout,
    Session,
    SessionBook,
    open_session,
    stop_report,
)
from offset.tools.debug.protocol import ProtocolError, SourceBreakpoint, frames_report

#: Actions that start or move a debuggee.
DRIVE_ACTIONS: Final = (
    "launch",
    "attach",
    "continue",
    "step_over",
    "step_in",
    "step_out",
    "pause",
    "breakpoint",
    "clear_breakpoints",
    "terminate",
    "status",
    "adapters",
)

#: Actions that only read a stopped session.
READ_ACTIONS: Final = ("stack", "scopes", "variables", "evaluate", "threads", "output")

#: How long to wait for the debuggee to stop after a step or continue, when the
#: caller does not say.  Long enough for a real program to reach a breakpoint,
#: short enough that a program which never will does not hang the turn.
STOP_WAIT: Final = 10.0

#: Depth for rendering a variable tree.  Two levels shows a dict of objects
#: with their fields, which is the common case; deeper is usually noise and
#: costs a round trip per node.
VAR_DEPTH: Final = 2

#: Rows of a container to render before summarising.
VAR_BREADTH: Final = 30

_BOOK: Final = SessionBook()

#: Breakpoints the user has asked for but which no session has bound yet, kept
#: so `breakpoint` works before `launch` - the DAP window for setting them is
#: after `initialized` and before `configurationDone`, which is inside
#: `open_session`, so they have to be collected up front.
_PENDING: dict[str, list[SourceBreakpoint]] = {}


def book() -> SessionBook:
    return _BOOK


def pending_breakpoints() -> dict[str, list[SourceBreakpoint]]:
    return _PENDING


def _current() -> Session | None:
    # `current` is a method, not a property.  Reading it without the call
    # returned the bound method, which is always truthy, so the `is None`
    # guard below never fired and every inspection on a session-less shell
    # died on `.client` instead of saying there was no session.
    return _BOOK.current()


def _fail(exc: Exception) -> ToolResult:
    """Adapter faults are answers, not crashes."""
    if isinstance(exc, RequestTimeout):
        return ToolResult.fail(f"the debug adapter did not answer in time: {exc}")
    if isinstance(exc, RequestFailed):
        return ToolResult.fail(f"the debug adapter refused: {exc}")
    if isinstance(exc, (DebugError, ProtocolError)):
        return ToolResult.fail(str(exc))
    return ToolResult.fail(f"{type(exc).__name__}: {exc}")


def _resolve(ctx: ToolContext, path: str) -> tuple[Path | None, str]:
    try:
        return ctx.resolve(path), ""
    except PermissionError as exc:
        return None, str(exc)


def _render_variables(session: Session, ref: int, *, depth: int = VAR_DEPTH) -> list[str]:
    """A bounded, readable tree of a scope or a variable.

    Bounded on purpose: a single `self` in a web framework can expand to
    thousands of nodes, and a model that receives all of them learns nothing it
    could not have learned from thirty.
    """
    lines: list[str] = []

    def walk(reference: int, prefix: str, level: int) -> None:
        if reference <= 0 or level > depth:
            return
        try:
            found = session.client.variables(reference)
        except (DebugError, ProtocolError) as exc:
            lines.append(f"{prefix}<could not read: {exc}>")
            return
        for var in found[:VAR_BREADTH]:
            kind = f" ({var.type})" if getattr(var, "type", "") else ""
            lines.append(f"{prefix}{var.name}{kind} = {var.value}")
            child = getattr(var, "variables_reference", 0) or 0
            if child and level < depth:
                walk(child, prefix + "  ", level + 1)
        if len(found) > VAR_BREADTH:
            lines.append(f"{prefix}... {len(found) - VAR_BREADTH} more")

    walk(ref, "", 1)
    return lines


def _stopped_lines(session: Session) -> list[str]:
    stop = session.client.last_stop
    return stop_report(
        stop,
        exited=session.client.exited,
        exit_code=session.client.exit_code,
    )


class Debug(Tool):
    """Start and drive a debuggee."""

    name = "debug"
    description = (
        "Run a program under a debugger: set breakpoints, launch, step, continue. "
        "Stops on a breakpoint so state can be read with debug_inspect instead of "
        "guessing from print statements. Set breakpoints before launching."
    )
    danger = Danger.DESTRUCTIVE
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(DRIVE_ACTIONS), "description": "what to do"},
            "program": {"type": "string", "description": "for launch: the file to run"},
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "arguments for the debuggee",
            },
            "file": {"type": "string", "description": "for breakpoint: the source file"},
            "line": {"type": "integer", "minimum": 1, "description": "for breakpoint: 1-based line"},
            "condition": {
                "type": "string",
                "maxLength": 500,
                "description": "for breakpoint: only stop when this expression is true",
            },
            "function": {
                "type": "string",
                "maxLength": 200,
                "description": "for breakpoint: break on a function name instead of a line",
            },
            "thread_id": {"type": "integer", "description": "which thread to step or continue"},
            "port": {"type": "integer", "description": "for attach: the port to connect to"},
            "host": {"type": "string", "description": "for attach: the host to connect to"},
            "timeout": {
                "type": "number",
                "minimum": 0,
                "description": "seconds to wait for the debuggee to stop",
            },
        },
        "required": ["action"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        action = args.get("action", "?")
        target = args.get("program") or args.get("file") or ""
        return f"debug {action} {target}".strip()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(args.get("action", "")).strip()
        if action not in DRIVE_ACTIONS:
            return ToolResult.fail(
                f"no debug action {action!r}. available: {', '.join(sorted(DRIVE_ACTIONS))}"
            )
        try:
            return getattr(self, f"_{action}")(args, ctx)
        except Exception as exc:
            return _fail(exc)

    # -- inspection of the tool's own state --------------------------------

    def _adapters(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        config = load_config(getattr(ctx, "root", None) or ctx.cwd)
        lines = adapter_report(config=config)
        if config.errors:
            lines.append("")
            lines.extend(f"config: {e}" for e in config.errors)
        return ToolResult.text("\n".join(lines))

    def _status(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = _current()
        if session is None:
            waiting = sum(len(v) for v in _PENDING.values())
            note = f"no debug session. {waiting} breakpoint(s) waiting" if waiting else "no debug session"
            return ToolResult.text(note)
        lines = session.status()
        lines.extend(_stopped_lines(session))
        return ToolResult.text("\n".join(lines))

    # -- breakpoints --------------------------------------------------------

    def _breakpoint(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        function = str(args.get("function", "") or "")
        if function:
            _PENDING.setdefault("", []).append(SourceBreakpoint(line=0, condition=function))
            return ToolResult.text(f"will break on {function}() when the session starts")

        path_arg = args.get("file")
        line = args.get("line")
        if not path_arg or not isinstance(line, int):
            return ToolResult.fail("breakpoint needs file and line, or function")
        resolved, why = _resolve(ctx, str(path_arg))
        if resolved is None:
            return ToolResult.fail(why)

        wanted = SourceBreakpoint(line=line, condition=str(args.get("condition", "") or ""))
        key = str(resolved)
        _PENDING.setdefault(key, []).append(wanted)

        session = _current()
        if session is None:
            return ToolResult.text(f"breakpoint set at {path_arg}:{line} (applies on launch)")
        bound = session.client.set_breakpoints(key, _PENDING[key])
        verified = sum(1 for b in bound if getattr(b, "verified", False))
        return ToolResult.text(
            f"breakpoint at {path_arg}:{line}; {verified}/{len(bound)} verified in the live session"
        )

    def _clear_breakpoints(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        count = sum(len(v) for v in _PENDING.values())
        paths = list(_PENDING)
        _PENDING.clear()
        session = _current()
        if session is not None:
            for path in paths:
                if path:
                    try:
                        session.client.set_breakpoints(path, [])
                    except (DebugError, ProtocolError):
                        continue
        return ToolResult.text(f"cleared {count} breakpoint(s)")

    # -- lifecycle ----------------------------------------------------------

    def _launch(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        program = str(args.get("program", "") or "")
        if not program:
            return ToolResult.fail("launch needs a program")
        resolved, why = _resolve(ctx, program)
        if resolved is None:
            return ToolResult.fail(why)
        if not resolved.exists():
            return ToolResult.fail(f"no such file: {program}")

        busy = _BOOK.busy()
        if busy is not None:
            return ToolResult.fail(
                f"a debug session is already running ({busy.program or busy.id}); "
                "terminate it first"
            )

        language = language_for(resolved)
        if not language:
            return ToolResult.fail(
                f"offset does not know how to debug {resolved.suffix or 'that file type'}"
            )
        config = load_config(getattr(ctx, "root", None) or ctx.cwd)
        launch, why = choose(language, config=config)
        if launch is None:
            return ToolResult.fail(why)

        session = self._open(launch, resolved, args, ctx, request="launch")
        if isinstance(session, ToolResult):
            return session
        return self._settle(session, args, verb="launched")

    def _attach(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        port = args.get("port")
        if not isinstance(port, int):
            return ToolResult.fail("attach needs a port")
        busy = _BOOK.busy()
        if busy is not None:
            return ToolResult.fail("a debug session is already running; terminate it first")
        config = load_config(getattr(ctx, "root", None) or ctx.cwd)
        launch, why = choose("python", config=config)
        if launch is None:
            return ToolResult.fail(why)
        host = str(args.get("host", "127.0.0.1") or "127.0.0.1")
        session = self._open(
            launch, None, args, ctx, request="attach",
            config={"connect": {"host": host, "port": port}},
        )
        if isinstance(session, ToolResult):
            return session
        return self._settle(session, args, verb=f"attached to {host}:{port}")

    def _open(
        self,
        launch: Launch,
        program: Path | None,
        args: dict[str, Any],
        ctx: ToolContext,
        *,
        request: str,
        config: dict[str, Any] | None = None,
    ) -> Session | ToolResult:
        """Start the adapter and complete the handshake."""
        root = Path(getattr(ctx, "root", None) or ctx.cwd)
        body: dict[str, Any] = dict(config or {})
        if request == "launch" and program is not None:
            body.update({
                "program": str(program),
                "cwd": str(root),
                "args": [str(a) for a in (args.get("args") or [])],
                "console": "internalConsole",
                "justMyCode": False,
            })
        functions = [
            bp.condition for bp in _PENDING.get("", []) if getattr(bp, "condition", "")
        ]
        sources = {k: v for k, v in _PENDING.items() if k}
        try:
            session = open_session(
                launch.channel(),
                config=body,
                request=request,
                breakpoints=sources or None,
                functions=functions,
                adapter=launch.language,
                program=str(program) if program else "",
                session_id=_BOOK.mint(),
                timeout=launch.timeout,
            )
        except (DebugError, ProtocolError) as exc:
            return _fail(exc)
        _BOOK.hold(session)
        return session

    def _settle(self, session: Session, args: dict[str, Any], *, verb: str) -> ToolResult:
        """Wait for the first stop, and say what happened."""
        wait = args.get("timeout")
        timeout = float(wait) if isinstance(wait, (int, float)) and wait > 0 else STOP_WAIT
        stop = session.client.wait_stopped(timeout=timeout)
        lines = [f"{verb} {Path(session.program).name if session.program else session.id}"]
        lines.extend(session.configured.report())
        if stop is not None:
            lines.append(stop.describe())
            lines.extend(frames_report(session.client.stack_trace(stop.thread_id)[:5]))
        elif session.client.finished:
            lines.extend(_stopped_lines(session))
        else:
            lines.append(f"still running after {timeout:g}s; no breakpoint hit yet")
        out = session.client.output()
        if out:
            lines.append("--- output ---")
            lines.extend(chunk.text.rstrip("\n") for chunk in out[-20:])
        return ToolResult.text("\n".join(l for l in lines if l))

    # -- movement -----------------------------------------------------------

    def _thread(self, session: Session, args: dict[str, Any]) -> int:
        wanted = args.get("thread_id")
        if isinstance(wanted, int):
            return wanted
        stop = session.client.last_stop
        if stop is not None and stop.thread_id:
            return stop.thread_id
        threads = session.client.threads()
        return threads[0].id if threads else 1

    def _move(self, args: dict[str, Any], how: str) -> ToolResult:
        session = _current()
        if session is None:
            return ToolResult.fail("no debug session; launch one first")
        if session.client.finished:
            return ToolResult.text("\n".join(_stopped_lines(session)))
        thread = self._thread(session, args)
        wait = args.get("timeout")
        timeout = float(wait) if isinstance(wait, (int, float)) and wait > 0 else STOP_WAIT

        session.client.drain_stops()
        getattr(session.client, how)(thread)
        stop = session.client.wait_stopped(timeout=timeout)
        lines: list[str] = []
        if stop is not None:
            lines.append(stop.describe())
            lines.extend(frames_report(session.client.stack_trace(stop.thread_id)[:5]))
        elif session.client.finished:
            lines.extend(_stopped_lines(session))
        else:
            lines.append(f"still running after {timeout:g}s")
        out = session.client.output()
        if out:
            lines.append("--- output ---")
            lines.extend(chunk.text.rstrip("\n") for chunk in out[-20:])
        return ToolResult.text("\n".join(l for l in lines if l))

    def _continue(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self._move(args, "resume")

    def _step_over(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self._move(args, "step_over")

    def _step_in(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self._move(args, "step_in")

    def _step_out(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return self._move(args, "step_out")

    def _pause(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = _current()
        if session is None:
            return ToolResult.fail("no debug session; launch one first")
        session.client.pause(self._thread(session, args))
        stop = session.client.wait_stopped(timeout=STOP_WAIT)
        if stop is None:
            return ToolResult.text("asked the debuggee to pause; it has not stopped yet")
        return ToolResult.text(stop.describe())

    def _terminate(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        session = _current()
        if session is None:
            return ToolResult.text("no debug session")
        notes = _BOOK.release()
        return ToolResult.text("\n".join(["debug session ended", *notes]))


class DebugInspect(Tool):
    """Read a stopped debuggee: stack, scopes, variables, expressions."""

    name = "debug_inspect"
    description = (
        "Read the state of a stopped debuggee: stack frames, scopes, variable values, "
        "or evaluate an expression in a frame. Only meaningful while stopped at a "
        "breakpoint. This is what replaces adding print statements."
    )
    danger = Danger.SAFE
    parallel_safe = True
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(READ_ACTIONS), "description": "what to read"},
            "frame_id": {
                "type": "integer",
                "description": "which frame; defaults to the innermost stopped frame",
            },
            "thread_id": {"type": "integer", "description": "which thread; defaults to the stopped one"},
            "expression": {
                "type": "string",
                "maxLength": 2000,
                "description": "for evaluate: the expression to run in that frame",
            },
            "reference": {
                "type": "integer",
                "description": "for variables: a scope or variable reference from a previous call",
            },
        },
        "required": ["action"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        return f"debug_inspect {args.get('action', '?')}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(args.get("action", "")).strip()
        if action not in READ_ACTIONS:
            return ToolResult.fail(
                f"no debug_inspect action {action!r}. available: {', '.join(sorted(READ_ACTIONS))}"
            )
        session = _current()
        if session is None:
            return ToolResult.fail("no debug session; launch one first")

        try:
            if action == "output":
                chunks = session.client.output()
                if not chunks:
                    return ToolResult.text("no output from the debuggee yet")
                return ToolResult.text("\n".join(c.text.rstrip("\n") for c in chunks))

            if action == "threads":
                threads = session.client.threads()
                if not threads:
                    return ToolResult.text("the adapter reported no threads")
                return ToolResult.text("\n".join(f"{t.id}: {t.name}" for t in threads))

            stop = session.client.last_stop
            if stop is None and not session.client.finished:
                return ToolResult.fail(
                    "the debuggee is running, so its state cannot be read; "
                    "set a breakpoint and continue, or pause it"
                )

            thread = args.get("thread_id")
            thread_id = thread if isinstance(thread, int) else (stop.thread_id if stop else 1)

            if action == "stack":
                frames = session.client.stack_trace(thread_id)
                if not frames:
                    return ToolResult.text("no stack frames")
                return ToolResult.text("\n".join(frames_report(frames)))

            frame = args.get("frame_id")
            if not isinstance(frame, int):
                frames = session.client.stack_trace(thread_id)
                if not frames:
                    return ToolResult.fail("no stack frames to inspect")
                frame = frames[0].id

            if action == "scopes":
                scopes = session.client.scopes(frame)
                if not scopes:
                    return ToolResult.text("no scopes in that frame")
                lines = []
                for scope in scopes:
                    lines.append(f"{scope.name} (reference {scope.variables_reference})")
                    lines.extend(f"  {l}" for l in _render_variables(session, scope.variables_reference, depth=1))
                return ToolResult.text("\n".join(lines))

            if action == "variables":
                ref = args.get("reference")
                if isinstance(ref, int) and ref > 0:
                    lines = _render_variables(session, ref)
                    return ToolResult.text("\n".join(lines) or "no variables under that reference")
                scopes = session.client.scopes(frame)
                lines = []
                for scope in scopes:
                    lines.append(f"--- {scope.name} ---")
                    lines.extend(_render_variables(session, scope.variables_reference))
                return ToolResult.text("\n".join(lines) or "no variables in that frame")

            # evaluate
            expression = str(args.get("expression", "") or "")
            if not expression:
                return ToolResult.fail("evaluate needs an expression")
            outcome = session.client.evaluate(expression, frame_id=frame)
            return ToolResult.text("\n".join(outcome.report()))
        except Exception as exc:
            return _fail(exc)


def debug_tools() -> list[Tool]:
    """The pair.  They share the module-level session book by construction."""
    return [Debug(), DebugInspect()]
