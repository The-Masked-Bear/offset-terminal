"""A scripted provider.

Not a stub standing in for missing work — a real, useful provider that replays
a fixed event script.  It is how the agent loop, the multi-model scheduler and
the UI get exercised deterministically with no network, no key and no cost.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator, Sequence

from offset.providers.base import (
    Event,
    Provider,
    Request,
    Stop,
    StreamError,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    Usage,
)


def script(
    text: str = "",
    *,
    thinking: str = "",
    tool_calls: Sequence[tuple[str, str, dict[str, Any]]] = (),
    usage: Usage | None = None,
    stop: str | None = None,
    chunk: int = 7,
    error: str | None = None,
) -> list[Event]:
    """Build an event script.  Text is split so streaming is exercised."""
    events: list[Event] = []
    if usage:
        events.append(Usage(input=usage.input, cache_read=usage.cache_read, cache_write=usage.cache_write))
    for i in range(0, len(thinking), chunk):
        events.append(ThinkingDelta(thinking[i : i + chunk]))
    for i in range(0, len(text), chunk):
        events.append(TextDelta(text[i : i + chunk]))
    for slot, (call_id, name, args) in enumerate(tool_calls):
        blob = json.dumps(args)
        events.append(ToolCallDelta(index=slot, id=call_id, name=name))
        for i in range(0, len(blob), chunk):  # arguments stream in fragments too
            events.append(ToolCallDelta(index=slot, args_delta=blob[i : i + chunk]))
    if error:
        events.append(StreamError(error))
    if usage:
        events.append(Usage(output=usage.output))
    events.append(Stop(stop or ("tool_use" if tool_calls else "stop")))
    return events


class Mock(Provider):
    """Replays scripts, one per turn, recording the requests it received."""

    name = "mock"
    env_keys = ()

    def __init__(self, scripts: Sequence[Sequence[Event]] | Callable[[Request], Sequence[Event]] | None = None) -> None:
        self._scripts = scripts
        self._turn = 0
        self.requests: list[Request] = []

    def stream(
        self, request: Request, *, api_key: str | None = None, credential: Any = None
    ) -> Iterator[Event]:
        self.requests.append(request)
        source = self._scripts
        if source is None:
            events: Sequence[Event] = script(f"mock reply to {request.model}")
        elif callable(source):
            events = source(request)
        else:
            events = source[self._turn] if self._turn < len(source) else script("")
        self._turn += 1
        yield from events

    @property
    def turns(self) -> int:
        return self._turn
