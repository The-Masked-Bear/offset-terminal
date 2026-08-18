"""The shape every provider agrees on.

A request goes in, a stream of small immutable events comes out, and
`TurnBuilder` folds those events back into one `Turn`.  Keeping the streaming
form and the assembled form separate is what makes multi-model work later: the
UI consumes events as they arrive, while the scheduler compares finished turns.

One rule worth stating out loud: a tool call whose arguments do not parse is
never silently dropped.  It is surfaced on the turn as a malformed call, so the
caller can re-prompt the model instead of quietly losing an action.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Literal, Sequence

Role = Literal["system", "user", "assistant", "tool"]


# -- conversation ----------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    raw: str | None = None  # kept when `args` could not be parsed


@dataclass(slots=True)
class Message:
    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    thinking: str = ""
    name: str | None = None


@dataclass(slots=True)
class Request:
    model: str
    messages: list[Message]
    system: str | None = None
    tools: Sequence[ToolSpec] = ()
    max_tokens: int = 4096
    temperature: float | None = None
    stop: Sequence[str] = ()
    thinking_budget: int | None = None
    timeout: float = 300.0

    def with_model(self, model: str) -> "Request":
        """Same conversation, different model — the multi-model primitive."""
        return replace(self, model=model)


# -- streaming events -------------------------------------------------------


class Event:
    __slots__ = ()


@dataclass(slots=True)
class TextDelta(Event):
    text: str


@dataclass(slots=True)
class ThinkingDelta(Event):
    text: str


@dataclass(slots=True)
class ToolCallDelta(Event):
    index: int
    id: str | None = None
    name: str | None = None
    args_delta: str = ""


@dataclass(slots=True)
class Usage(Event):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input + other.input,
            self.output + other.output,
            self.cache_read + other.cache_read,
            self.cache_write + other.cache_write,
        )


@dataclass(slots=True)
class Stop(Event):
    reason: str = "stop"  # stop | tool_use | length | refusal | aborted | error


@dataclass(slots=True)
class StreamError(Event):
    message: str
    status: int | None = None
    retryable: bool = False


# -- assembled result -------------------------------------------------------


@dataclass(slots=True)
class Turn:
    text: str = ""
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "stop"
    error: str | None = None

    @property
    def malformed(self) -> list[ToolCall]:
        """Calls whose arguments never parsed — surfaced, never swallowed."""
        return [c for c in self.tool_calls if c.raw is not None]

    def to_message(self) -> Message:
        return Message(role="assistant", text=self.text, thinking=self.thinking, tool_calls=list(self.tool_calls))


class TurnBuilder:
    """Folds an event stream into a `Turn`."""

    __slots__ = ("_calls", "_error", "_order", "_stop", "_text", "_thinking", "_usage")

    def __init__(self) -> None:
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._calls: dict[int, dict[str, Any]] = {}
        self._order: list[int] = []
        self._usage = Usage()
        self._stop = "stop"
        self._error: str | None = None

    def feed(self, event: Event) -> Event:
        if isinstance(event, TextDelta):
            self._text.append(event.text)
        elif isinstance(event, ThinkingDelta):
            self._thinking.append(event.text)
        elif isinstance(event, ToolCallDelta):
            slot = self._calls.get(event.index)
            if slot is None:
                slot = {"id": None, "name": None, "args": []}
                self._calls[event.index] = slot
                self._order.append(event.index)
            if event.id:
                slot["id"] = event.id
            if event.name:
                slot["name"] = event.name
            if event.args_delta:
                slot["args"].append(event.args_delta)
        elif isinstance(event, Usage):
            self._usage = self._usage + event
        elif isinstance(event, Stop):
            self._stop = event.reason
        elif isinstance(event, StreamError):
            self._error = event.message
            self._stop = "error"
        return event

    def consume(self, events: Iterator[Event]) -> "TurnBuilder":
        for event in events:
            self.feed(event)
        return self

    def finish(self) -> Turn:
        calls: list[ToolCall] = []
        for i in self._order:
            slot = self._calls[i]
            blob = "".join(slot["args"]).strip()
            name = slot["name"] or ""
            cid = slot["id"] or f"call_{i}"
            if not blob:
                calls.append(ToolCall(id=cid, name=name, args={}))
                continue
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                calls.append(ToolCall(id=cid, name=name, args={}, raw=blob))
                continue
            if isinstance(parsed, dict):
                calls.append(ToolCall(id=cid, name=name, args=parsed))
            else:
                calls.append(ToolCall(id=cid, name=name, args={}, raw=blob))
        stop = self._stop
        if calls and stop == "stop":
            stop = "tool_use"  # providers disagree here; normalise it
        return Turn(
            text="".join(self._text),
            thinking="".join(self._thinking),
            tool_calls=calls,
            usage=self._usage,
            stop_reason=stop,
            error=self._error,
        )


# -- provider ---------------------------------------------------------------


class Provider(ABC):
    """A model endpoint.  Subclasses only translate; they never retry."""

    name: str = "provider"
    #: Environment variables searched for credentials, in order.
    env_keys: tuple[str, ...] = ()

    @abstractmethod
    def stream(
        self, request: Request, *, api_key: str | None = None, credential: Any = None
    ) -> Iterator[Event]:
        """Yield events until the turn ends.  Must not raise for HTTP errors;
        yield `StreamError` instead so the caller can decide.

        `credential` is an `offset.providers.auth.Credential` when the caller has
        one; it wins over `api_key` because it also covers OAuth, which needs a
        different header and may have just been refreshed.
        """

    def complete(self, request: Request, *, api_key: str | None = None, credential: Any = None) -> Turn:
        events = self.stream(request, api_key=api_key, credential=credential)
        return TurnBuilder().consume(events).finish()


def auth_header(api_key: str | None, credential: Any, fallback: dict[str, str]) -> dict[str, str]:
    """One place decides how a request is authenticated.

    A `Credential` knows its own header, including the Bearer form OAuth needs;
    without one we fall back to whatever the provider does with a bare key.
    """
    if credential is None:
        return fallback
    name, value = credential.header()
    return {name: value} if name else {}
