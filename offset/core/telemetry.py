"""What a turn cost, where the time went, and what went wrong.

`Usage` events have always flown past the agent loop and nobody caught them, so
the questions people actually ask - *what did that cost*, *which model burns the
most*, *why did last Tuesday fail* - had no answer at all. This records them.

Three decisions shape the rest of the file.

**An unknown price is `None`, never zero.** A model the table has not heard of
produces no cost figure. Reporting a confident zero for a model that in fact
charges is worse than reporting nothing, because a zero looks like an answer
and nobody checks it. Prices are indicative and go stale; `~/.offset/pricing.json`
overrides them, and the source of every figure is visible in the rollup.

**Telemetry may never cost a turn.** Every entry point swallows its own
failures. An accounting bug that kills a generation is a far worse bug than the
missing number it was trying to record, so `observe()` catches everything and
carries on.

**Cheap enough to leave on.** Traces cap the text they keep and never deep-copy
a tool payload; the ledger is append-only JSONL with a size cap and rotation.
Nothing here holds a whole session in memory.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from offset.core import settings

#: Where the ledger lives, relative to `settings.home()`.
LEDGER_NAME: Final = "telemetry.jsonl"

#: Rotate past this size.  One rotation is kept: enough to survive a rollover
#: mid-report, not enough to fill a Raspberry Pi's card.
MAX_BYTES: Final = 4 * 1024 * 1024

#: Longest text kept in a span.  A trace is for seeing shape, not for replaying
#: content - the session log already holds the content.
CLIP: Final = 200

#: Spans kept in memory for the live trace.  A turn that makes a thousand tool
#: calls should not be able to grow this without bound.
MAX_SPANS: Final = 500

#: Suppresses all recording.  For a test, or somebody who does not want a cost
#: file on disk at all.
OFF_ENV: Final = "OFFSET_NO_TELEMETRY"


# -- prices -------------------------------------------------------------------

#: US dollars per million tokens, `(input, output)`.
#:
#: Indicative, and certain to go stale - providers reprice and this file does
#: not know when.  A model that is not here yields `None` rather than a guess,
#: and `~/.offset/pricing.json` overrides any of it in the same shape.
PRICES: Final[dict[str, tuple[float, float]]] = {
    "claude-opus-4": (15.0, 75.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
    "claude-3-7-sonnet": (3.0, 15.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5": (1.25, 10.0),
    "o3": (2.0, 8.0),
    "o4-mini": (1.10, 4.40),
    "gemini-3-pro": (1.25, 10.0),
    "gemini-3-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "deepseek": (0.27, 1.10),
}

#: Anything local costs nothing to run, and saying "unknown" for it would be
#: pedantic rather than careful.
FREE_PREFIXES: Final = ("ollama", "llamacpp", "mock")


def _overrides(home: Path | None = None) -> dict[str, tuple[float, float]]:
    path = (home if home is not None else settings.home()) / "pricing.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, tuple[float, float]] = {}
    if isinstance(raw, dict):
        for name, pair in raw.items():
            try:
                out[str(name)] = (float(pair[0]), float(pair[1]))
            except (TypeError, ValueError, IndexError, KeyError):
                continue  # one bad entry must not discard the rest
    return out


def price_of(model: str, home: Path | None = None) -> tuple[float, float] | None:
    """Dollars per million `(input, output)` tokens, or None if unknown.

    Longest prefix wins, so `claude-sonnet-4-20250514` finds `claude-sonnet-4`
    without the table needing every dated variant.
    """
    bare = model.split("/")[-1].strip().lower()
    if any(bare.startswith(p) or model.startswith(p) for p in FREE_PREFIXES):
        return (0.0, 0.0)
    table = {**PRICES, **_overrides(home)}
    best = ""
    for prefix in table:
        if not bare.startswith(prefix) or len(prefix) <= len(best):
            continue
        # A dated or variant suffix is the same model: `claude-sonnet-4-20250514`
        # really is `claude-sonnet-4`. A *version* bump is not:
        # `gpt-5.6-luna` is not `gpt-5`, and charging it gpt-5 rates would be
        # the confident wrong number this whole module exists to avoid.
        rest = bare[len(prefix):]
        if rest and not rest.startswith("-"):
            continue
        best = prefix
    return table[best] if best else None


def cost_of(model: str, tokens_in: int, tokens_out: int,
            home: Path | None = None) -> float | None:
    price = price_of(model, home)
    if price is None:
        return None
    return (tokens_in / 1_000_000) * price[0] + (tokens_out / 1_000_000) * price[1]


# -- traces --------------------------------------------------------------------


@dataclass(slots=True)
class Span:
    """One timed thing: a whole turn, a step within it, or a tool call."""

    kind: str                      # "turn" | "step" | "tool"
    name: str
    started: float
    ended: float = 0.0
    ok: bool = True
    detail: str = ""
    parent: int = -1               # index into Trace.spans, -1 for the root

    @property
    def seconds(self) -> float:
        return max(0.0, (self.ended or time.time()) - self.started)

    def close(self, *, ok: bool = True, detail: str = "") -> None:
        self.ended = time.time()
        self.ok = ok
        if detail:
            self.detail = detail[:CLIP]


@dataclass(slots=True)
class Trace:
    """The span tree for one turn, bounded in size."""

    spans: list[Span] = field(default_factory=list)
    dropped: int = 0

    def open(self, kind: str, name: str, parent: int = -1) -> int:
        if len(self.spans) >= MAX_SPANS:
            self.dropped += 1
            return -1
        self.spans.append(Span(kind=kind, name=name[:CLIP], started=time.time(), parent=parent))
        return len(self.spans) - 1

    def close(self, index: int, *, ok: bool = True, detail: str = "") -> None:
        if 0 <= index < len(self.spans):
            self.spans[index].close(ok=ok, detail=detail)

    def lines(self) -> list[str]:
        """The tree, indented by depth, for `/trace`."""
        out: list[str] = []
        for i, span in enumerate(self.spans):
            depth = 0
            parent = span.parent
            seen = 0
            while parent >= 0 and seen < MAX_SPANS:
                depth += 1
                parent = self.spans[parent].parent
                seen += 1
            mark = " " if span.ok else "!"
            out.append(f"{mark} {'  ' * depth}{span.kind:<5} {span.name[:44]:<44} "
                       f"{span.seconds:6.2f}s {span.detail[:30]}")
        if self.dropped:
            out.append(f"  ...and {self.dropped} more spans, not recorded")
        return out


# -- the ledger -----------------------------------------------------------------


@dataclass(slots=True)
class Entry:
    """One turn, as recorded."""

    at: float
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    cost: float | None = None
    seconds: float = 0.0
    steps: int = 0
    tools: int = 0
    reason: str = "stop"
    error: str = ""
    session: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "at": round(self.at, 3), "model": self.model,
            "in": self.tokens_in, "out": self.tokens_out, "cached": self.tokens_cached,
            "cost": self.cost, "seconds": round(self.seconds, 3),
            "steps": self.steps, "tools": self.tools,
            "reason": self.reason, "error": self.error[:CLIP], "session": self.session,
        }

    @classmethod
    def from_json(cls, raw: Any) -> Entry | None:
        if not isinstance(raw, dict):
            return None
        try:
            cost = raw.get("cost")
            return cls(
                at=float(raw.get("at") or 0.0), model=str(raw.get("model") or ""),
                tokens_in=int(raw.get("in") or 0), tokens_out=int(raw.get("out") or 0),
                tokens_cached=int(raw.get("cached") or 0),
                cost=None if cost is None else float(cost),
                seconds=float(raw.get("seconds") or 0.0),
                steps=int(raw.get("steps") or 0), tools=int(raw.get("tools") or 0),
                reason=str(raw.get("reason") or ""), error=str(raw.get("error") or ""),
                session=str(raw.get("session") or ""),
            )
        except (TypeError, ValueError):
            return None


def ledger_file(home: Path | None = None) -> Path:
    return (home if home is not None else settings.home()) / LEDGER_NAME


def enabled() -> bool:
    return (os.environ.get(OFF_ENV) or "").strip().lower() not in ("1", "true", "yes", "on")


class Ledger:
    """Append-only turn history on disk.

    JSONL rather than SQLite: it is written once per turn and read rarely, a
    half-written final line costs one record instead of a corrupt database, and
    a user can grep it.
    """

    def __init__(self, home: Path | None = None, *, max_bytes: int = MAX_BYTES) -> None:
        #: Captured now, on the calling thread.  A background writer that asks
        #: `settings.home()` for itself answers with whatever the environment
        #: says by then, which for an exited shell is the wrong directory.
        #: That exact bug has been fixed here once already.
        self.home = home if home is not None else settings.home()
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return ledger_file(self.home)

    def append(self, entry: Entry) -> bool:
        """Record one turn.  Returns whether it landed; never raises."""
        if not enabled():
            return False
        line = json.dumps(entry.to_json(), separators=(",", ":"))
        with self._lock:
            try:
                self.home.mkdir(parents=True, exist_ok=True)
                self._rotate_if_needed()
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                return True
            except OSError:
                return False  # a full disk must not end the turn

    def _rotate_if_needed(self) -> None:
        try:
            if self.path.stat().st_size < self.max_bytes:
                return
        except OSError:
            return
        try:
            os.replace(self.path, self.path.with_suffix(".jsonl.1"))
        except OSError:
            pass  # rotation is housekeeping; failing it must not stop recording

    def read(self, *, since: float = 0.0, limit: int = 100_000) -> list[Entry]:
        """Every recorded turn, oldest first.  A mangled line is skipped."""
        out: list[Entry] = []
        for path in (self.path.with_suffix(".jsonl.1"), self.path):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = Entry.from_json(json.loads(line))
                except ValueError:
                    continue
                if entry is not None and entry.at >= since:
                    out.append(entry)
                    if len(out) >= limit:
                        return out
        return out


# -- rollups ---------------------------------------------------------------------


@dataclass(slots=True)
class Total:
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    #: True when at least one turn had no price, so `cost` is a floor and the
    #: report must say so rather than presenting a partial sum as the answer.
    partial: bool = False
    failures: int = 0
    seconds: float = 0.0

    def add(self, entry: Entry) -> None:
        self.turns += 1
        self.tokens_in += entry.tokens_in
        self.tokens_out += entry.tokens_out
        self.seconds += entry.seconds
        if entry.cost is None:
            self.partial = True
        else:
            self.cost += entry.cost
        if entry.error or entry.reason == "error":
            self.failures += 1

    @property
    def mean_seconds(self) -> float:
        return self.seconds / self.turns if self.turns else 0.0

    def money(self) -> str:
        if self.turns and self.partial and self.cost == 0.0:
            return "unpriced"
        return f"${self.cost:.4f}" + ("+" if self.partial else "")


def rollup(entries: Iterable[Entry], by: str = "model") -> dict[str, Total]:
    """Group turns by `model`, `day`, `session` or `reason`."""
    out: dict[str, Total] = {}
    for entry in entries:
        if by == "model":
            key = entry.model or "(unknown)"
        elif by == "day":
            key = time.strftime("%Y-%m-%d", time.localtime(entry.at))
        elif by == "session":
            key = entry.session or "(none)"
        elif by == "reason":
            key = entry.reason or "(none)"
        else:
            key = "all"
        out.setdefault(key, Total()).add(entry)
    return out


def report(totals: dict[str, Total], *, title: str = "") -> list[str]:
    lines = [title] if title else []
    if not totals:
        return lines + ["nothing recorded yet"]
    width = max(len(k) for k in totals)
    lines.append(f"{'':<{width}}  {'turns':>5} {'in':>9} {'out':>9} {'cost':>11} "
                 f"{'fail':>4} {'mean':>7}")
    for key, t in sorted(totals.items(), key=lambda kv: -kv[1].turns):
        lines.append(f"{key:<{width}}  {t.turns:>5} {t.tokens_in:>9,} {t.tokens_out:>9,} "
                     f"{t.money():>11} {t.failures:>4} {t.mean_seconds:>6.1f}s")
    if any(t.partial for t in totals.values()):
        lines.append("")
        lines.append("a '+' means some turns used a model with no known price; "
                     + "set one in ~/.offset/pricing.json")
    return lines


# -- watching the loop --------------------------------------------------------------


class Recorder:
    """Turns agent-loop events into a ledger entry and a trace.

    Fed the same events the UI sees.  It keeps one turn at a time: `Finished`
    closes the entry and writes it, and anything arriving outside a turn is
    ignored rather than guessed at.
    """

    def __init__(self, ledger: Ledger | None = None, *, model: str = "",
                 session: str = "") -> None:
        self.ledger = ledger if ledger is not None else Ledger()
        self.model = model
        self.session = session
        self.trace = Trace()
        self.last: Trace | None = None
        self._turn = -1
        self._step = -1
        self._tools: dict[str, int] = {}
        self._entry: Entry | None = None
        #: Reentrant on purpose. `_observe` holds this and, for the first
        #: `StepStarted` of a turn nobody opened explicitly, calls `start()` -
        #: which takes it again. With a plain Lock that is a deadlock, and a
        #: deadlock is the one failure `observe`'s blanket except cannot
        #: swallow: it would hang the agent rather than lose a measurement.
        self._lock = threading.RLock()

    def start(self, model: str = "") -> None:
        with self._lock:
            self._begin(model)

    def _begin(self, model: str = "") -> None:
        self.trace = Trace()
        self._turn = self.trace.open("turn", model or self.model or "turn")
        self._step = -1
        self._tools.clear()
        self._entry = Entry(at=time.time(), model=model or self.model,
                            session=self.session)

    def observe(self, event: Any) -> None:
        """Fold one event in.  Never raises: see the module docstring."""
        try:
            self._observe(event)
        except Exception:
            # Accounting must not be able to end a generation.  A dropped
            # measurement is a strictly smaller loss than a dropped turn.
            pass

    def _observe(self, event: Any) -> None:
        from offset.core.agent import Finished, StepStarted, ToolFinished, ToolStarted
        from offset.providers.base import StreamError, Usage

        with self._lock:
            if self._entry is None and not isinstance(event, StepStarted):
                return

            if isinstance(event, StepStarted):
                if self._entry is None:
                    self._begin(getattr(event, "model", "") or self.model)
                if self._step >= 0:
                    self.trace.close(self._step)
                self._step = self.trace.open("step", f"step {event.index}", self._turn)
                if self._entry is not None:
                    self._entry.steps = max(self._entry.steps, event.index + 1)

            elif isinstance(event, ToolStarted):
                name = getattr(event.call, "name", "tool")
                call_id = getattr(event.call, "id", name)
                self._tools[call_id] = self.trace.open("tool", name, self._step)
                if self._entry is not None:
                    self._entry.tools += 1

            elif isinstance(event, ToolFinished):
                inv = event.invocation
                call_id = getattr(getattr(inv, "call", None), "id", "")
                index = self._tools.pop(call_id, -1)
                ok = bool(getattr(getattr(inv, "result", None), "ok", True))
                self.trace.close(index, ok=ok,
                                 detail=str(getattr(inv.result, "display", ""))[:CLIP])

            elif isinstance(event, Usage) and self._entry is not None:
                self._entry.tokens_in += int(getattr(event, "input", 0) or 0)
                self._entry.tokens_out += int(getattr(event, "output", 0) or 0)
                self._entry.tokens_cached += int(getattr(event, "cache_read", 0) or 0)

            elif isinstance(event, StreamError) and self._entry is not None:
                self._entry.error = str(getattr(event, "message", ""))[:CLIP]

            elif isinstance(event, Finished):
                self._finish(event)

    def _finish(self, event: Any) -> None:
        entry = self._entry
        if entry is None:
            return
        usage = getattr(event, "usage", None)
        if usage is not None:
            # `Finished` carries the authoritative total; individual Usage
            # events may be partial or absent depending on the provider.
            entry.tokens_in = max(entry.tokens_in, int(getattr(usage, "input", 0) or 0))
            entry.tokens_out = max(entry.tokens_out, int(getattr(usage, "output", 0) or 0))
        entry.steps = max(entry.steps, int(getattr(event, "steps", 0) or 0))
        entry.reason = str(getattr(event, "reason", "stop"))
        entry.seconds = max(0.0, time.time() - entry.at)
        entry.cost = cost_of(entry.model, entry.tokens_in, entry.tokens_out, self.ledger.home)

        if self._step >= 0:
            self.trace.close(self._step)
        self.trace.close(self._turn, ok=entry.reason not in ("error",), detail=entry.reason)

        self.ledger.append(entry)
        self.last = self.trace
        self._entry = None
        self._step = -1


#: The recorder the shell installs, so `/usage` and `/trace` can reach it.
_active: Recorder | None = None


def active() -> Recorder | None:
    return _active


def install(state: Any) -> None:
    """Startup wiring: begin recording.  Never raises, never blocks."""
    global _active
    try:
        session = str(getattr(getattr(state, "session", None), "id", "") or "")
        _active = Recorder(Ledger(settings.home()),
                           model=str(getattr(state, "model", "") or ""),
                           session=session)
    except Exception:
        _active = None


def observe(event: Any) -> None:
    """The one line the shell's event loop needs."""
    recorder = _active
    if recorder is not None:
        recorder.observe(event)


# -- commands ---------------------------------------------------------------------


def _usage(state: Any, args: list[str]) -> Any:
    from offset.shell.commands import TONE_INFO, Outcome

    which = (args[0].lower() if args else "models")
    ledger = Ledger(settings.home())
    entries = ledger.read()
    if not entries:
        return Outcome(["nothing recorded yet - run a turn first",
                        f"ledger: {ledger.path}"], TONE_INFO)

    if which in ("today", "day", "days"):
        lines = report(rollup(entries, "day"), title="by day")
    elif which in ("session", "sessions"):
        lines = report(rollup(entries, "session"), title="by session")
    elif which in ("fail", "failures", "errors"):
        bad = [e for e in entries if e.error or e.reason == "error"]
        if not bad:
            return Outcome(["no failures recorded"], TONE_INFO)
        lines = [f"{time.strftime('%m-%d %H:%M', time.localtime(e.at))}  "
                 f"{e.model[:24]:<24} {e.reason:<10} {e.error[:60]}" for e in bad[-30:]]
        lines.insert(0, f"{len(bad)} failed turns, most recent last")
    else:
        lines = report(rollup(entries, "model"), title="by model")

    grand = Total()
    for e in entries:
        grand.add(e)
    total = (f"total: {grand.turns} turns, "
             f"{grand.tokens_in + grand.tokens_out:,} tokens, {grand.money()}")
    lines += ["", total]
    return Outcome(lines, TONE_INFO)


def _trace(state: Any, args: list[str]) -> Any:
    from offset.shell.commands import TONE_INFO, Outcome

    recorder = active()
    trace = (recorder.last or recorder.trace) if recorder else None
    if trace is None or not trace.spans:
        return Outcome(["no trace yet - run a turn first"], TONE_INFO)
    return Outcome(trace.lines(), TONE_INFO)


def telemetry_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        # `/cost`, not `/usage`: the shell already has a `/usage` that reports
        # the current model, key, tools and approval mode, and `_extend`
        # *replaces* a colliding name - so registering this as `/usage` would
        # silently delete a working command.  `/cost` is the better question
        # anyway: nobody asks "what is my usage", they ask what it cost.
        Command("cost", "tokens, cost and failures", _usage,
                usage="/cost [models|today|session|failures]",
                aliases=("tokens",)),
        Command("trace", "the last turn's span tree", _trace),
    ]


#: Built lazily: `offset.shell.commands` imports this module, so resolving the
#: command list at import time would be a cycle.
_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """Built on first access, not at import.

    The re-check after building is not redundant: `telemetry_commands()`
    imports the shell registry, which re-enters this module before the outer
    call has stored anything.  A single check produces two lists and registers
    every command twice.  `offset.core.tasks` carries the same guard for the
    same reason.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = telemetry_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
