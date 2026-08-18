"""Executing tool calls: approval, timeouts, cancellation, concurrency.

The three rules this module exists to enforce:

  * a call is validated before it runs, and a rejected call returns a message
    the model can act on rather than an exception;
  * a call that hangs is abandoned, not waited on forever, and cancellation is
    cooperative so a tool can clean up;
  * calls run concurrently only when every tool in the batch says it is safe —
    one unsafe tool serialises the whole batch, because a half-parallel batch
    is the hardest kind of bug to reproduce.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable, Literal, Sequence

from offset.providers.base import ToolCall
from offset.tools.base import Cancelled, Danger, Tool, ToolContext, ToolResult, Toolbox, validate

Mode = Literal["safe", "auto-edit", "yolo", "full"]

#: What each mode approves without asking.  "yolo" stops at DESTRUCTIVE on
#: purpose: leaving the workspace is a separate decision the user has to make.
THRESHOLD: dict[Mode, Danger] = {
    "safe": Danger.SAFE,
    "auto-edit": Danger.WRITE,
    "yolo": Danger.DESTRUCTIVE,
    "full": Danger.FULL,
}


@dataclass(slots=True)
class Approval:
    """Decides whether a call may proceed.

    `ask` is only consulted for calls above the mode's threshold.  Answers can
    be remembered per tool for the rest of the session, which is what makes
    `safe` mode usable instead of merely annoying.
    """

    mode: Mode = "auto-edit"
    ask: Callable[[Tool, dict], bool] | None = None
    remembered: set[str] = field(default_factory=set)
    denied: set[str] = field(default_factory=set)

    def check(self, tool: Tool, args: dict) -> tuple[bool, str]:
        if tool.name in self.denied:
            return False, f"{tool.name} was denied earlier in this session"
        if tool.danger <= THRESHOLD[self.mode] or tool.name in self.remembered:
            return True, ""
        if self.ask is None:
            return False, (
                f"{tool.name} needs approval ({tool.danger.name.lower()}) and no approver is attached"
            )
        return (True, "") if self.ask(tool, args) else (False, f"{tool.name} was declined by the user")

    def remember(self, name: str) -> None:
        self.remembered.add(name)
        self.denied.discard(name)

    def deny(self, name: str) -> None:
        self.denied.add(name)
        self.remembered.discard(name)


@dataclass(slots=True)
class Invocation:
    """A finished call: what was asked, what happened, how long it took."""

    call: ToolCall
    result: ToolResult
    approved: bool = True


class Runtime:
    __slots__ = ("_pool_size", "abort", "approval", "before_write", "context", "toolbox")

    def __init__(
        self,
        toolbox: Toolbox,
        context: ToolContext,
        approval: Approval | None = None,
        *,
        pool_size: int = 8,
        before_write: Callable[[Tool, dict], None] | None = None,
    ) -> None:
        self.toolbox = toolbox
        self.context = context
        self.approval = approval or Approval()
        self._pool_size = max(1, pool_size)
        #: Called just before a tool that can modify files runs, so a snapshot
        #: can be taken. Kept as a callback because the runtime has no business
        #: knowing about sessions; the shell supplies one that records to the
        #: session it owns.
        self.before_write = before_write
        #: A user abort for the whole turn.  Deliberately NOT the same signal
        #: as a per-call timeout: one means "stop everything", the other means
        #: "this one call ran long". Conflating them ends turns that should
        #: have carried on, and poisons every later call in the session.
        self.abort = threading.Event()

    # -- single call ------------------------------------------------------

    def execute(self, call: ToolCall) -> Invocation:
        started = time.monotonic()

        if call.raw is not None:
            return self._done(call, ToolResult.fail(
                f"arguments for {call.name} were not valid JSON; resend them as a JSON object"
            ), started)

        tool = self.toolbox.get(call.name)
        if tool is None:
            close = ", ".join(sorted(self.toolbox.names())[:12])
            return self._done(call, ToolResult.fail(f"no tool named {call.name!r}. available: {close}"), started)

        problems = validate(call.args, tool.schema)
        if problems:
            return self._done(call, ToolResult.fail("; ".join(problems)), started)

        allowed, why = self.approval.check(tool, call.args)
        if not allowed:
            return Invocation(call, ToolResult.fail(why), approved=False)

        if self.before_write is not None and tool.danger >= Danger.WRITE:
            try:
                self.before_write(tool, call.args)
            except Exception as exc:
                # A snapshot that cannot be taken is worth saying out loud, but
                # it must never be the reason a tool call fails.
                self.context.emit(f"snapshot skipped for {tool.name}: {exc}")

        return self._done(call, self._run_guarded(tool, call.args), started)

    def _run_guarded(self, tool: Tool, args: dict) -> ToolResult:
        """Run with a deadline, on a signal private to this call."""
        deadline = self.context.timeout
        ctx = replace(self.context, cancel=threading.Event())
        box: list[ToolResult] = []

        def target() -> None:
            try:
                box.append(tool.run(args, ctx))
            except Cancelled as exc:
                box.append(ToolResult.fail(str(exc) or "cancelled"))
            except PermissionError as exc:
                box.append(ToolResult.fail(f"refused: {exc}"))
            except Exception as exc:  # a broken tool must not kill the turn
                box.append(ToolResult.fail(f"{type(exc).__name__}: {exc}"))

        worker = threading.Thread(target=target, name=f"tool-{tool.name}", daemon=True)
        worker.start()
        limit = time.monotonic() + deadline
        while worker.is_alive():
            worker.join(0.05)
            if self.abort.is_set():
                ctx.cancel.set()
                worker.join(min(2.0, deadline))
                return ToolResult.fail(f"{tool.name} was cancelled by the user")
            if time.monotonic() >= limit:
                ctx.cancel.set()  # ask it to stop; it owns its own cleanup
                worker.join(min(2.0, deadline))
                return ToolResult.fail(f"{tool.name} exceeded its {deadline:g}s budget")
        return box[0] if box else ToolResult.fail(f"{tool.name} returned nothing")

    @staticmethod
    def _done(call: ToolCall, result: ToolResult, started: float) -> Invocation:
        result.duration = time.monotonic() - started
        return Invocation(call, result)

    # -- batches ----------------------------------------------------------

    def parallelisable(self, calls: Sequence[ToolCall]) -> bool:
        if len(calls) < 2:
            return False
        tools = [self.toolbox.get(c.name) for c in calls]
        return all(t is not None and t.parallel_safe for t in tools)

    def execute_all(self, calls: Sequence[ToolCall]) -> list[Invocation]:
        """Run a batch, preserving order regardless of completion order."""
        if not calls:
            return []
        if not self.parallelisable(calls):
            return [self.execute(c) for c in calls]
        with ThreadPoolExecutor(max_workers=min(self._pool_size, len(calls))) as pool:
            return list(pool.map(self.execute, calls))

    def cancel(self) -> None:
        """Abort the whole turn.  Reaches calls that are already running."""
        self.abort.set()
        self.context.cancel.set()

    @property
    def aborted(self) -> bool:
        return self.abort.is_set()

    def reset(self) -> None:
        self.abort = threading.Event()
        self.context.cancel = threading.Event()
