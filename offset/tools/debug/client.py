"""One debug session: correlation, the startup dance, and stop state.

`protocol.py` turns bytes into records.  This module turns records into a
session a caller can reason about, and it exists mostly because of two things
that DAP gets wrong from a client author's point of view.

The first is *ordering*.  A debug adapter is not ready to be configured when it
answers `initialize`; it is ready when it says so, by sending the `initialized`
*event*.  The sequence this module implements, and the only one that works
across adapters, is:

  1. send `initialize`, wait for its response — that response is where the
     capabilities come from, and every later branch depends on them;
  2. wait for the `initialized` **event**.  This is the step everyone skips,
     because the response has already arrived and looks like permission;
  3. send `setBreakpoints` / `setFunctionBreakpoints` /
     `setExceptionBreakpoints` — the configuration window is open now and only
     now;
  4. send `configurationDone`, closing the window;
  5. send `launch` or `attach`.

Two failures follow from getting that wrong.  `configurationDone` sent before
the `initialized` event is answered with an error by a strict adapter and
silently ignored by a lax one, which then never starts the program.
Breakpoints set after `configurationDone` bind late, so the debuggee sails
past the line the user is watching — the bug that looks like "breakpoints do
not work".  `configuration_done()` therefore *refuses* to send unless the event
has arrived; that guard is a hard invariant rather than a courtesy.

Step 5 has one wrinkle worth stating.  A minority of adapters withhold
`initialized` until they know what they are debugging, and the DAP
specification allows the `launch` response itself to be withheld until
`configurationDone` arrives.  Waiting on either one first deadlocks against
the other, so when the event has not appeared within `CONFIG_GRACE` the
request is *dispatched without waiting for its reply*, the event is waited for
again, configuration proceeds, and only then is the reply collected.  Both
paths keep the invariant: `configurationDone` never precedes `initialized`.

The second thing DAP gets wrong for us is that the interesting news is
unsolicited.  A program stops at a breakpoint, prints to stdout, or exits, and
each of those arrives as an event with no request to hang it on.  Polling for
them would be both wasteful and lossy, so every event is absorbed the moment
it is read: `stopped` goes onto a queue that `wait_stopped` blocks on, output
into a bounded buffer that survives until someone drains it, and exit into
latches.  A caller that resumes a program and then asks "why did it stop?"
gets an answer, with a deadline, and never a poll loop.
"""

from __future__ import annotations

import atexit
import itertools
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

from offset.tools.debug.protocol import (
    AdapterClosed,
    Breakpoint,
    Capabilities,
    Channel,
    Event,
    ProtocolError,
    Response,
    Scope,
    SourceBreakpoint,
    StackFrame,
    Thread,
    Variable,
    _int,
    classify,
    request_frame,
)

CLIENT_NAME: Final = "offset"

#: Wait slice while a request is outstanding: the granularity of the deadline
#: and of cooperative cancellation both.
_TICK: Final = 0.05

DEFAULT_TIMEOUT: Final = 30.0

#: How long to wait for the `initialized` event before falling back to
#: dispatching launch/attach first.  Adapters that behave send it within
#: milliseconds of the initialize response, so this only ever costs time on the
#: adapters that need the fallback.
CONFIG_GRACE: Final = 5.0

#: Teardown budget per step.  A debuggee wedged in a signal handler must not
#: hold the user's turn open.
GRACE: Final = 3.0

#: Debuggee output kept between drains.  Enough to see what a test printed,
#: bounded so a chatty program cannot grow this process.
OUTPUT_LIMIT: Final = 500

#: Events kept for `status`.  Diagnostic only.
EVENT_LIMIT: Final = 200

#: Deepest nesting `variables` will expand.  Every level is another round trip
#: per container, and a model cannot read more than this anyway.
MAX_DEPTH: Final = 3

#: Hard ceiling on `variables` round trips for one call, so a self-referential
#: structure costs a bounded amount rather than the whole deadline.
MAX_EXPANSIONS: Final = 64

#: Stop reasons worth naming in a report, in the adapters' own vocabulary.
STOP_REASONS: Final = (
    "breakpoint",
    "function breakpoint",
    "data breakpoint",
    "instruction breakpoint",
    "step",
    "exception",
    "entry",
    "pause",
    "goto",
)


class DebugError(ProtocolError):
    """A session-level failure: bad ordering, or an adapter that refused."""


class RequestFailed(DebugError):
    """The adapter answered `success: false`.  Carries what it said."""

    def __init__(self, command: str, message: str) -> None:
        super().__init__(f"{command} failed: {message or 'the adapter did not say why'}")
        self.command = command
        self.detail = message


class RequestTimeout(DebugError):
    """The adapter did not answer in time.  Never a user cancellation."""


class DebugCancelled(DebugError):
    """The caller asked to stop while a request was outstanding."""


# -- records ----------------------------------------------------------------


@dataclass(slots=True)
class Stop:
    """A `stopped` event: the only moment at which state is inspectable.

    `reason` is what a caller actually wants after resuming — breakpoint, step,
    exception, entry — and `thread_id` is what every follow-up request needs,
    so both are lifted out of the body rather than left in a dict.
    """

    reason: str = ""
    thread_id: int | None = None
    description: str = ""
    text: str = ""
    breakpoints: list[int] = field(default_factory=list)
    all_threads: bool = False

    @classmethod
    def parse(cls, body: Mapping[str, Any]) -> "Stop":
        hit = body.get("hitBreakpointIds")
        ids = [n for n in (_int(v) for v in hit) if n is not None] if isinstance(hit, list) else []
        return cls(
            reason=str(body.get("reason") or "unknown"),
            thread_id=_int(body.get("threadId")),
            description=str(body.get("description") or ""),
            text=str(body.get("text") or ""),
            breakpoints=ids,
            all_threads=body.get("allThreadsStopped") is True,
        )

    def describe(self) -> str:
        where = f" on thread {self.thread_id}" if self.thread_id is not None else ""
        detail = self.description or self.text
        hit = f" (breakpoint {', '.join(str(i) for i in self.breakpoints)})" if self.breakpoints else ""
        return f"stopped: {self.reason}{where}{hit}" + (f" — {detail}" if detail else "")


@dataclass(slots=True)
class OutputChunk:
    """One `output` event.  Adapters emit fragments, not lines."""

    category: str
    text: str


def output_report(chunks: Sequence[OutputChunk], *, limit: int = 200) -> list[str]:
    """Buffered output as whole lines, newest kept when there are too many.

    Consecutive chunks of one category are joined before being split, because
    an adapter routinely delivers half a line and then the rest of it; splitting
    each chunk on its own would shred every line in the debuggee's output.
    """
    lines: list[str] = []
    for category, group in groupby(chunks, key=lambda chunk: chunk.category):
        text = "".join(chunk.text for chunk in group)
        if not text:
            continue
        for line in text.splitlines() or [""]:
            lines.append(line if category == "stdout" else f"{category}: {line}")
    if len(lines) > limit:
        dropped = len(lines) - limit
        return [f"… {dropped} earlier output lines omitted", *lines[-limit:]]
    return lines


@dataclass(slots=True)
class Evaluation:
    """An `evaluate` reply.  A failed expression is a result, not an error:
    "there is no such name here" is exactly what the caller asked to find out."""

    ok: bool
    result: str = ""
    type: str = ""
    variables_reference: int = 0
    error: str = ""
    children: list[Variable] = field(default_factory=list)

    def report(self) -> list[str]:
        if not self.ok:
            return [f"evaluation failed: {self.error or 'the adapter did not say why'}"]
        kind = f" : {self.type}" if self.type else ""
        out = [f"{self.result}{kind}"]
        for child in self.children:
            out.extend(child.report(indent=1))
        return out


@dataclass(slots=True)
class Configured:
    """What the configuration window achieved, including what it could not.

    A file whose breakpoints were refused must not abort a launch: the rest of
    the session is still useful, and an unverified breakpoint is precisely the
    thing a caller needs told about.
    """

    source: dict[str, list[Breakpoint]] = field(default_factory=dict)
    functions: list[Breakpoint] = field(default_factory=list)
    exceptions: list[Breakpoint] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[Breakpoint]:
        out: list[Breakpoint] = []
        for placed in self.source.values():
            out.extend(placed)
        return [*out, *self.functions, *self.exceptions]

    def report(self) -> list[str]:
        out: list[str] = []
        for path in sorted(self.source):
            for placed in self.source[path]:
                out.append(f"  {placed.describe()}")
        for placed in self.functions:
            out.append(f"  function {placed.describe()}")
        unverified = sum(1 for placed in self.all if not placed.verified)
        if out:
            out.insert(0, f"{len(self.all)} breakpoints, {unverified} unverified")
        out.extend(f"  {why}" for why in self.problems)
        return out


@dataclass(slots=True)
class Pending:
    """A request in flight.  It carries its own `seq` so a caller can prove
    which reply it was handed — the correlation this module is here to get
    right, since responses may arrive in any order at all."""

    seq: int
    command: str
    event: threading.Event
    reply: Response | None = None
    failure: Exception | None = None


# -- the client -------------------------------------------------------------


class DebugClient:
    """One adapter connection.  Thread-safe: the reader runs alongside callers."""

    __slots__ = (
        "_closed",
        "_exited",
        "_initialized",
        "_known",
        "_last_stop",
        "_lock",
        "_orphans",
        "_output",
        "_pending",
        "_reader",
        "_seqs",
        "_stops",
        "_terminated",
        "breakpoint_updates",
        "capabilities",
        "channel",
        "dead_reason",
        "events",
        "exit_code",
        "name",
        "on_event",
        "timeout",
    )

    def __init__(
        self,
        channel: Channel,
        *,
        name: str = CLIENT_NAME,
        timeout: float = DEFAULT_TIMEOUT,
        on_event: Callable[[Event], None] | None = None,
    ) -> None:
        self.channel = channel
        self.name = name
        self.timeout = timeout
        self.on_event = on_event
        self.capabilities = Capabilities()
        self.dead_reason = ""
        self.exit_code: int | None = None
        self.events: deque[Event] = deque(maxlen=EVENT_LIMIT)
        self.breakpoint_updates: deque[Breakpoint] = deque(maxlen=64)
        self._output: deque[OutputChunk] = deque(maxlen=OUTPUT_LIMIT)
        self._stops: queue.Queue[Stop] = queue.Queue()
        self._initialized = threading.Event()
        self._exited = threading.Event()
        self._terminated = threading.Event()
        self._known: set[int] = set()
        self._last_stop: Stop | None = None
        self._seqs = itertools.count(1)
        self._lock = threading.Lock()
        self._pending: dict[int, Pending] = {}
        self._reader: threading.Thread | None = None
        self._closed = True
        self._orphans = 0

    # -- lifecycle ----------------------------------------------------------

    def open(self) -> None:
        """Start the channel and the reader.  No DAP traffic yet."""
        self.channel.start()
        self._closed = False
        self.dead_reason = ""
        self._reader = threading.Thread(target=self._pump, name=f"dap-{self.name}", daemon=True)
        self._reader.start()

    @property
    def alive(self) -> bool:
        return not self._closed and self.channel.alive

    @property
    def initialized(self) -> bool:
        return self._initialized.is_set()

    @property
    def exited(self) -> bool:
        return self._exited.is_set()

    @property
    def terminated(self) -> bool:
        return self._terminated.is_set()

    @property
    def finished(self) -> bool:
        """The debuggee is over, whatever the adapter is still doing."""
        return self._exited.is_set() or self._terminated.is_set()

    @property
    def last_stop(self) -> Stop | None:
        return self._last_stop

    @property
    def orphaned_replies(self) -> int:
        return self._orphans

    def close(self) -> None:
        """Idempotent.  Everything waiting is failed rather than abandoned."""
        self._closed = True
        self._fail_all(AdapterClosed(self.dead_reason or "session closed"))
        try:
            self.channel.close()
        except Exception:  # closing must never raise into a teardown path
            pass
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=GRACE)

    def diagnostics(self) -> str:
        return self.dead_reason or self.channel.diagnostics()

    # -- requests -----------------------------------------------------------

    def send(self, command: str, arguments: dict[str, Any] | None = None) -> Pending:
        """Dispatch a request and return its slot, without waiting for a reply.

        Needed for `launch`: the specification permits an adapter to withhold
        that response until `configurationDone`, so waiting for it first is a
        deadlock rather than a slow path.
        """
        if self._closed:
            raise AdapterClosed(self.diagnostics() or "not connected")
        with self._lock:
            seq = next(self._seqs)
            slot = Pending(seq=seq, command=command, event=threading.Event())
            self._pending[seq] = slot
        try:
            self.channel.send(request_frame(seq, command, arguments))
        except AdapterClosed as exc:
            self._forget(seq)
            self.dead_reason = str(exc)
            raise
        return slot

    def settle(
        self,
        slot: Pending,
        *,
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> Response:
        """Wait for one dispatched request's reply, with a hard deadline."""
        budget = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, budget)
        while not slot.event.wait(_TICK):
            if stop is not None and stop():
                self._forget(slot.seq)
                self._cancel_remote(slot.seq)
                raise DebugCancelled(f"{slot.command} cancelled")
            if time.monotonic() >= deadline:
                self._forget(slot.seq)
                self._cancel_remote(slot.seq)
                raise RequestTimeout(f"{slot.command} did not answer within {budget:g}s")
        self._forget(slot.seq)
        if slot.failure is not None:
            raise slot.failure
        if slot.reply is None:  # cannot happen: the event is only set with one
            raise DebugError(f"{slot.command} was answered with nothing")
        return slot.reply

    def request(
        self,
        command: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> Response:
        """Send and wait.  A `success: false` reply is returned, not raised:
        some commands (`evaluate`) fail as a matter of course."""
        return self.settle(self.send(command, arguments), timeout=timeout, stop=stop)

    def _cancel_remote(self, seq: int) -> None:
        """Tell the adapter to drop a request we stopped waiting for.

        Only when it advertised the capability: an unknown command earns an
        error response, and this is a path where we already gave up.
        """
        if not self.capabilities.supports("supportsCancelRequest"):
            return
        try:
            self.channel.send(request_frame(next(self._seqs), "cancel", {"requestId": seq}))
        except (AdapterClosed, OSError):
            pass

    def _forget(self, seq: int) -> None:
        with self._lock:
            self._pending.pop(seq, None)

    def _fail_all(self, failure: Exception) -> None:
        with self._lock:
            waiting, self._pending = self._pending, {}
        for slot in waiting.values():
            slot.failure = failure
            slot.event.set()

    # -- reader -------------------------------------------------------------

    def _pump(self) -> None:
        while not self._closed:
            try:
                frame = self.channel.receive(_TICK)
            except AdapterClosed as exc:
                self._die(str(exc) or "the adapter exited")
                return
            except Exception as exc:  # a channel bug must not strand callers
                self._die(f"channel failed: {type(exc).__name__}: {exc}")
                return
            if frame is not None:
                self._handle(frame)

    def _die(self, why: str) -> None:
        self.dead_reason = why
        self._closed = True
        self._fail_all(AdapterClosed(why))

    def _handle(self, frame: dict[str, Any]) -> None:
        record = classify(frame)
        if isinstance(record, Response):
            with self._lock:
                slot = self._pending.pop(record.request_seq, None)
            if slot is None:
                self._orphans += 1  # a reply to a request we already gave up on
                return
            slot.reply = record
            slot.event.set()
            return
        if isinstance(record, Event):
            self._absorb(record)
            return
        if frame.get("type") == "request":
            self._refuse(frame)

    def _refuse(self, frame: dict[str, Any]) -> None:
        """Answer a reverse request.  Silence would hang the adapter forever.

        `runInTerminal` and `startDebugging` are the two in the wild, and offset
        implements neither: it runs debuggees on the adapter's own console.
        Refusing turns a hang into a message the user can read.
        """
        command = str(frame.get("command") or "a request")
        seq = _int(frame.get("seq")) or 0
        reply = {
            "seq": next(self._seqs),
            "type": "response",
            "request_seq": seq,
            "command": command,
            "success": False,
            "message": f"offset does not implement the reverse request {command}",
        }
        try:
            self.channel.send(reply)
        except (AdapterClosed, OSError):
            pass

    def _absorb(self, event: Event) -> None:
        self.events.append(event)
        body = event.body
        name = event.event
        if name == "initialized":
            self._initialized.set()
        elif name == "stopped":
            stop = Stop.parse(body)
            if stop.thread_id is not None:
                self._known.add(stop.thread_id)
            self._last_stop = stop
            self._stops.put(stop)
        elif name == "continued":
            self._last_stop = None
        elif name == "output":
            text = str(body.get("output") or "")
            if text:
                self._output.append(OutputChunk(str(body.get("category") or "console"), text))
        elif name == "exited":
            self.exit_code = _int(body.get("exitCode"))
            self._exited.set()
        elif name == "terminated":
            self._terminated.set()
        elif name == "thread":
            ident = _int(body.get("threadId"))
            if ident is not None:
                if str(body.get("reason") or "") == "exited":
                    self._known.discard(ident)
                else:
                    self._known.add(ident)
        elif name == "breakpoint":
            raw = body.get("breakpoint")
            if isinstance(raw, dict):
                self.breakpoint_updates.append(Breakpoint.parse(raw))
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # a bad hook must not kill the reader
                pass

    # -- waiting ------------------------------------------------------------

    def wait_initialized(self, timeout: float) -> bool:
        return self._initialized.wait(max(0.0, timeout))

    def wait_terminated(self, timeout: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while time.monotonic() < deadline:
            if self.finished:
                return True
            time.sleep(_TICK)
        return self.finished

    def wait_stopped(
        self,
        timeout: float,
        *,
        stop: Callable[[], bool] | None = None,
    ) -> Stop | None:
        """The next `stopped` event, or None if the deadline passed first.

        A queue rather than a flag, because a stop that arrives while the caller
        is still reading the previous reply must not be lost — and because a
        caller that resumes and then asks why it stopped should never poll.
        None also covers "the program ended instead of stopping"; `finished`
        and `exit_code` say which.
        """
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                return self._stops.get(timeout=_TICK)
            except queue.Empty:
                pass
            if stop is not None and stop():
                raise DebugCancelled("wait for a stop cancelled")
            if self.finished or not self.alive:
                return None
            if time.monotonic() >= deadline:
                return None

    def drain_stops(self) -> None:
        """Forget stops that predate the request about to be sent.

        Without this, `continue` followed by `wait_stopped` would return the
        stop the caller was already looking at.
        """
        while True:
            try:
                self._stops.get_nowait()
            except queue.Empty:
                return

    def output(self, *, drain: bool = True) -> list[OutputChunk]:
        chunks = list(self._output)
        if drain:
            self._output.clear()
        return chunks

    # -- operations ---------------------------------------------------------

    def initialize(self, *, adapter_id: str = "offset", timeout: float | None = None) -> Capabilities:
        """The handshake.  Nothing is advertised that offset does not implement:
        an adapter that trusted a false claim would wait for a reply we would
        never send."""
        reply = self.request(
            "initialize",
            {
                "clientID": "offset",
                "clientName": "offset",
                "adapterID": adapter_id,
                "locale": "en-GB",
                "linesStartAt1": True,
                "columnsStartAt1": True,
                "pathFormat": "path",
                "supportsVariableType": True,
                "supportsVariablePaging": False,
                "supportsRunInTerminalRequest": False,
                "supportsStartDebuggingRequest": False,
                "supportsMemoryReferences": False,
                "supportsProgressReporting": False,
                "supportsInvalidatedEvent": False,
            },
            timeout=timeout,
        )
        _require(reply)
        self.capabilities = Capabilities(raw=dict(reply.body))
        return self.capabilities

    def configuration_done(self, *, timeout: float | None = None) -> bool:
        """Close the configuration window.  Refuses to run early.

        The guard is the point of this method: sending `configurationDone`
        before the `initialized` event is the single most common DAP client bug,
        and it fails in the worst way — the adapter accepts it and never runs
        the program.
        """
        if not self._initialized.is_set():
            raise DebugError(
                "configurationDone before the initialized event: wait for the event, "
                "set breakpoints, then close the configuration window"
            )
        if self.capabilities.raw and not self.capabilities.configuration_done:
            return False  # never advertised; sending it earns an error response
        _require(self.request("configurationDone", {}, timeout=timeout))
        return True

    def set_breakpoints(
        self,
        path: str,
        breakpoints: Sequence[SourceBreakpoint],
        *,
        timeout: float | None = None,
    ) -> list[Breakpoint]:
        """Replace the breakpoints for one file.  DAP has no "add one"."""
        self._demand_window("setBreakpoints")
        reply = self.request(
            "setBreakpoints",
            {
                "source": {"path": path, "name": _leaf(path)},
                "breakpoints": [bp.payload() for bp in breakpoints],
                "lines": [bp.line for bp in breakpoints],
                "sourceModified": False,
            },
            timeout=timeout,
        )
        _require(reply)
        return _breakpoints(reply.body, source=path)

    def set_function_breakpoints(
        self,
        names: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> list[Breakpoint]:
        self._demand_window("setFunctionBreakpoints")
        if self.capabilities.raw and not self.capabilities.function_breakpoints:
            raise DebugError("this adapter never advertised supportsFunctionBreakpoints")
        reply = self.request(
            "setFunctionBreakpoints",
            {"breakpoints": [{"name": name} for name in names]},
            timeout=timeout,
        )
        _require(reply)
        return _breakpoints(reply.body)

    def set_exception_breakpoints(
        self,
        filters: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> list[Breakpoint]:
        self._demand_window("setExceptionBreakpoints")
        reply = self.request(
            "setExceptionBreakpoints", {"filters": list(filters)}, timeout=timeout
        )
        _require(reply)
        return _breakpoints(reply.body)

    def _demand_window(self, command: str) -> None:
        if not self._initialized.is_set():
            raise DebugError(
                f"{command} before the initialized event: the adapter is not ready "
                "to be configured yet"
            )

    def launch(self, config: Mapping[str, Any], *, timeout: float | None = None) -> Response:
        reply = self.settle(self.send("launch", dict(config)), timeout=timeout)
        _require(reply)
        return reply

    def attach(self, config: Mapping[str, Any], *, timeout: float | None = None) -> Response:
        reply = self.settle(self.send("attach", dict(config)), timeout=timeout)
        _require(reply)
        return reply

    def resume(self, thread_id: int, *, timeout: float | None = None) -> bool:
        """`continue`.  Returns whether every thread was resumed."""
        self.drain_stops()
        reply = self.request("continue", {"threadId": thread_id}, timeout=timeout)
        _require(reply)
        return reply.body.get("allThreadsContinued") is not False

    def step_over(self, thread_id: int, *, timeout: float | None = None) -> None:
        self._step("next", thread_id, timeout)

    def step_in(self, thread_id: int, *, timeout: float | None = None) -> None:
        self._step("stepIn", thread_id, timeout)

    def step_out(self, thread_id: int, *, timeout: float | None = None) -> None:
        self._step("stepOut", thread_id, timeout)

    def _step(self, command: str, thread_id: int, timeout: float | None) -> None:
        self.drain_stops()
        _require(self.request(command, {"threadId": thread_id}, timeout=timeout))

    def pause(self, thread_id: int, *, timeout: float | None = None) -> None:
        self.drain_stops()
        _require(self.request("pause", {"threadId": thread_id}, timeout=timeout))

    def stack_trace(
        self,
        thread_id: int,
        *,
        start: int = 0,
        levels: int = 20,
        timeout: float | None = None,
    ) -> list[StackFrame]:
        reply = self.request(
            "stackTrace",
            {"threadId": thread_id, "startFrame": start, "levels": levels},
            timeout=timeout,
        )
        _require(reply)
        raw = reply.body.get("stackFrames")
        if not isinstance(raw, list):
            return []
        return [StackFrame.parse(item) for item in raw if isinstance(item, dict)]

    def scopes(self, frame_id: int, *, timeout: float | None = None) -> list[Scope]:
        reply = self.request("scopes", {"frameId": frame_id}, timeout=timeout)
        _require(reply)
        raw = reply.body.get("scopes")
        if not isinstance(raw, list):
            return []
        return [Scope.parse(item) for item in raw if isinstance(item, dict)]

    def variables(
        self,
        reference: int,
        *,
        depth: int = 1,
        limit: int = 40,
        timeout: float | None = None,
    ) -> list[Variable]:
        """One level of variables, with nested containers expanded `depth` deep.

        A bare `variablesReference` is worthless to a model — it cannot be read
        and cannot be followed without another turn — so the tree is walked here
        instead, under two bounds: `depth` (capped at MAX_DEPTH) and a shared
        ceiling on round trips, which is what makes a cyclic structure safe.
        """
        budget = [MAX_EXPANSIONS]
        return self._expand(reference, min(depth, MAX_DEPTH), limit, budget, timeout)

    def _expand(
        self,
        reference: int,
        depth: int,
        limit: int,
        budget: list[int],
        timeout: float | None,
    ) -> list[Variable]:
        if reference <= 0 or depth <= 0 or budget[0] <= 0:
            return []
        budget[0] -= 1
        reply = self.request("variables", {"variablesReference": reference}, timeout=timeout)
        _require(reply)
        raw = reply.body.get("variables")
        if not isinstance(raw, list):
            return []
        found = [Variable.parse(item) for item in raw if isinstance(item, dict)]
        for var in found[:limit]:
            if var.expandable:
                var.children = self._expand(var.variables_reference, depth - 1, limit, budget, timeout)
        return found

    def evaluate(
        self,
        expression: str,
        *,
        frame_id: int | None = None,
        context: str = "repl",
        depth: int = 1,
        timeout: float | None = None,
    ) -> Evaluation:
        arguments: dict[str, Any] = {"expression": expression, "context": context}
        if frame_id is not None:
            arguments["frameId"] = frame_id
        reply = self.request("evaluate", arguments, timeout=timeout)
        if not reply.success:
            return Evaluation(ok=False, error=reply.message)
        body = reply.body
        outcome = Evaluation(
            ok=True,
            result=str(body.get("result") if body.get("result") is not None else ""),
            type=str(body.get("type") or ""),
            variables_reference=_int(body.get("variablesReference")) or 0,
        )
        if outcome.variables_reference > 0 and depth > 0:
            outcome.children = self.variables(
                outcome.variables_reference, depth=depth, timeout=timeout
            )
        return outcome

    def threads(self, *, timeout: float | None = None) -> list[Thread]:
        reply = self.request("threads", {}, timeout=timeout)
        _require(reply)
        raw = reply.body.get("threads")
        if not isinstance(raw, list):
            return []
        found = [Thread.parse(item) for item in raw if isinstance(item, dict)]
        self._known.update(t.id for t in found)
        return found

    def thread_id(self, given: int | None = None, *, timeout: float | None = None) -> int | None:
        """The thread a request should address, in order of trustworthiness:
        what the caller said, the thread that stopped, the only one we know of,
        then whatever the adapter lists first."""
        if given is not None:
            return given
        stop = self._last_stop
        if stop is not None and stop.thread_id is not None:
            return stop.thread_id
        if len(self._known) == 1:
            return next(iter(self._known))
        try:
            found = self.threads(timeout=timeout)
        except ProtocolError:
            return None
        return found[0].id if found else None

    def disconnect(self, *, terminate: bool = True, timeout: float | None = None) -> None:
        reply = self.request(
            "disconnect",
            {"restart": False, "terminateDebuggee": terminate},
            timeout=GRACE if timeout is None else timeout,
        )
        _require(reply)

    def terminate(self, *, timeout: float | None = None) -> None:
        reply = self.request("terminate", {"restart": False}, timeout=GRACE if timeout is None else timeout)
        _require(reply)


# -- one session ------------------------------------------------------------


@dataclass(slots=True)
class Session:
    """A live debuggee plus the client that drives it.

    Numbered so a report can name which session it is talking about, and so a
    refusal to start a second one can say what is already running.
    """

    id: str
    client: DebugClient
    adapter: str = ""
    program: str = ""
    request: str = "launch"
    started: float = field(default_factory=time.monotonic)
    configured: Configured = field(default_factory=Configured)

    @property
    def pid(self) -> int | None:
        return getattr(self.client.channel, "pid", None)

    @property
    def live(self) -> bool:
        return self.client.alive and not self.client.finished

    def status(self) -> list[str]:
        client = self.client
        state = "running"
        if client.finished:
            code = client.exit_code
            state = f"finished (exit code {code})" if code is not None else "finished"
        elif client.last_stop is not None:
            state = client.last_stop.describe()
        elif not client.alive:
            state = f"adapter gone: {client.diagnostics()}"
        out = [
            f"session {self.id}: {state}",
            f"  adapter {self.adapter or 'unknown'} ({self.request}) pid {self.pid or '?'}"
            f", up {time.monotonic() - self.started:.1f}s",
        ]
        if self.program:
            out.append(f"  program {self.program}")
        out.extend(self.configured.report())
        return out

    def close(self, *, kill: bool | None = None) -> list[str]:
        """Take the debuggee down, then the adapter, then the process group.

        Each step is allowed to fail because the next one is stronger: a
        `terminate` the adapter ignores is followed by a `disconnect`, and that
        is followed by killing the adapter's whole process group — which is what
        actually guarantees no orphan, since the debuggee is a grandchild.
        An attach session detaches instead: we did not start that program and
        must not kill it.
        """
        kill = (self.request == "launch") if kill is None else kill
        client = self.client
        out: list[str] = []
        if kill and client.alive and client.capabilities.terminate_request:
            try:
                client.terminate()
                out.append("asked the adapter to terminate the debuggee")
            except ProtocolError as exc:
                out.append(f"terminate declined: {exc}")
        if client.alive:
            try:
                client.disconnect(terminate=kill)
                out.append("disconnected, debuggee killed" if kill else "disconnected, debuggee left running")
            except ProtocolError as exc:
                out.append(f"disconnect declined: {exc}")
            client.wait_terminated(GRACE / 2)
        client.close()
        out.append("adapter process group reaped")
        return out


class SessionBook:
    """At most one session at a time, and never an orphan.

    One is a deliberate limit, not a simplification: a debuggee holds ports,
    file locks, a terminal and a core, and a second one started by accident is
    a leak the user cannot see.  The teardown is registered with `atexit`
    because the process exiting is exactly when an orphaned debuggee would
    otherwise survive.
    """

    __slots__ = ("_count", "_live", "_lock")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: Session | None = None
        self._count = 0

    def current(self) -> Session | None:
        return self._live

    def mint(self) -> str:
        with self._lock:
            self._count += 1
            return f"dbg-{self._count}"

    def busy(self) -> Session | None:
        """The session that would block a new one, if there is one.

        A session whose debuggee has already finished does not block: its output
        is worth keeping until someone starts the next one, and no longer.
        """
        live = self._live
        if live is None:
            return None
        return live if live.live else None

    def hold(self, session: Session) -> None:
        blocking = self.busy()
        if blocking is not None:
            raise DebugError(
                f"session {blocking.id} is already running; terminate it first"
            )
        stale = self._live
        with self._lock:
            self._live = session
        if stale is not None:
            stale.close()  # finished, but the adapter process was still ours

    def release(self, *, kill: bool | None = None) -> list[str]:
        with self._lock:
            session, self._live = self._live, None
        if session is None:
            return []
        return [f"session {session.id}: " + "; ".join(session.close(kill=kill))]


#: The process-wide book.  Registered for teardown at exit so an interpreter
#: that dies mid-session still does not leave a debuggee behind.
SESSIONS: Final = SessionBook()
atexit.register(SESSIONS.release)


def open_session(
    channel: Channel,
    *,
    config: Mapping[str, Any],
    request: str = "launch",
    breakpoints: Mapping[str, Sequence[SourceBreakpoint]] | None = None,
    functions: Sequence[str] = (),
    exception_filters: Sequence[str] = (),
    adapter: str = "",
    program: str = "",
    session_id: str = "dbg",
    timeout: float = DEFAULT_TIMEOUT,
    on_event: Callable[[Event], None] | None = None,
) -> Session:
    """The startup sequence from the module docstring, start to finish.

    Raises on anything that leaves no usable session, and closes the channel
    first — a half-started adapter is exactly the orphan this package exists to
    avoid.  Breakpoints that the adapter refuses are recorded on the session
    instead: those still leave a session worth having.
    """
    client = DebugClient(channel, name=session_id, timeout=timeout, on_event=on_event)
    try:
        client.open()
        client.initialize(adapter_id=adapter or "offset", timeout=timeout)
        pending: Pending | None = None
        if not client.wait_initialized(min(timeout, CONFIG_GRACE)):
            # This adapter wants to know what it is debugging first.  Dispatch
            # without waiting: its response may legally be withheld until
            # configurationDone, which we have not sent yet.
            pending = client.send(request, dict(config))
            if not client.wait_initialized(timeout):
                raise DebugError(
                    "the adapter never sent the initialized event, so it can never be "
                    f"configured ({client.diagnostics()})"
                )
        configured = configure(
            client,
            breakpoints=breakpoints,
            functions=functions,
            exception_filters=exception_filters,
            timeout=timeout,
        )
        client.configuration_done(timeout=timeout)
        if pending is None:
            pending = client.send(request, dict(config))
        _require(client.settle(pending, timeout=timeout))
    except ProtocolError:
        client.close()
        raise
    return Session(
        id=session_id,
        client=client,
        adapter=adapter,
        program=program,
        request=request,
        configured=configured,
    )


def configure(
    client: DebugClient,
    *,
    breakpoints: Mapping[str, Sequence[SourceBreakpoint]] | None = None,
    functions: Sequence[str] = (),
    exception_filters: Sequence[str] = (),
    timeout: float | None = None,
) -> Configured:
    """Step 3: everything that may only be sent inside the window.

    One refusal is recorded and the rest carry on, because a session with two
    of three breakpoints bound is far more use than no session at all.
    """
    out = Configured()
    for path, wanted in sorted((breakpoints or {}).items()):
        try:
            out.source[path] = client.set_breakpoints(path, list(wanted), timeout=timeout)
        except ProtocolError as exc:
            out.problems.append(f"breakpoints in {path}: {exc}")
    if functions:
        try:
            out.functions = client.set_function_breakpoints(list(functions), timeout=timeout)
        except ProtocolError as exc:
            out.problems.append(f"function breakpoints: {exc}")
    if exception_filters:
        try:
            out.exceptions = client.set_exception_breakpoints(list(exception_filters), timeout=timeout)
        except ProtocolError as exc:
            out.problems.append(f"exception breakpoints: {exc}")
    return out


# -- helpers ----------------------------------------------------------------


def _require(reply: Response) -> Response:
    if not reply.success:
        raise RequestFailed(reply.command, reply.message)
    return reply


def _breakpoints(body: Mapping[str, Any], *, source: str = "") -> list[Breakpoint]:
    raw = body.get("breakpoints")
    if not isinstance(raw, list):
        return []
    return [Breakpoint.parse(item, source=source) for item in raw if isinstance(item, dict)]


def _leaf(path: str) -> str:
    return path.rsplit("/", 1)[-1] or path


def stop_report(stop: Stop | None, *, exited: bool, exit_code: int | None) -> list[str]:
    """Why execution is where it is, in one or two lines a model can act on."""
    if stop is not None:
        return [stop.describe()]
    if exited:
        return [f"the debuggee exited with code {exit_code}" if exit_code is not None else "the debuggee exited"]
    return ["still running: it has not stopped"]


def known_reason(reason: str) -> bool:
    """Whether a stop reason is one of the vocabulary DAP actually defines.

    Used only to decide whether to quote an adapter's wording verbatim.
    """
    return reason in STOP_REASONS


def iter_events(client: DebugClient, kind: str) -> Iterable[Event]:
    return (event for event in list(client.events) if event.event == kind)
