"""Ollama — the local path.

NDJSON rather than SSE, no API key, and tool arguments arrive as real objects
instead of JSON text.  Being able to run a model with no network at all is
what keeps this usable on a Pi that is offline or rate-limited.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

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
from offset.providers.sse import iter_json_frames
from offset.providers.transport import HTTPFailure, Retry, post_lines

DONE_REASONS = {"stop": "stop", "length": "length", "load": "error"}


def build_payload(request: Request) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for m in request.messages:
        if m.role == "tool":
            messages.append({"role": "tool", "content": m.text})
            continue
        entry: dict[str, Any] = {"role": m.role, "content": m.text}
        if m.tool_calls:
            entry["tool_calls"] = [{"function": {"name": c.name, "arguments": c.args}} for c in m.tool_calls]
        messages.append(entry)

    options: dict[str, Any] = {"num_predict": request.max_tokens}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.stop:
        options["stop"] = list(request.stop)

    payload: dict[str, Any] = {"model": request.model, "messages": messages, "stream": True, "options": options}
    if request.tools:
        payload["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
            for t in request.tools
        ]
    return payload


def parse(lines: Iterator[bytes]) -> Iterator[Event]:
    slot = 0
    for obj in iter_json_frames(lines):
        if obj.get("error"):
            yield StreamError(str(obj["error"]))
            continue
        message = obj.get("message") or {}
        if message.get("content"):
            yield TextDelta(message["content"])
        if message.get("thinking"):
            yield ThinkingDelta(message["thinking"])
        for call in message.get("tool_calls") or ():
            fn = call.get("function") or {}
            args = fn.get("arguments")
            yield ToolCallDelta(
                index=slot,
                id=f"ollama_{slot}",
                name=fn.get("name"),
                args_delta=args if isinstance(args, str) else json.dumps(args or {}),
            )
            slot += 1
        if obj.get("done"):
            if obj.get("prompt_eval_count") or obj.get("eval_count"):
                yield Usage(input=int(obj.get("prompt_eval_count") or 0), output=int(obj.get("eval_count") or 0))
            yield Stop(DONE_REASONS.get(str(obj.get("done_reason") or "stop"), "stop"))


class Ollama(Provider):
    name = "ollama"
    env_keys = ()

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def stream(
        self, request: Request, *, api_key: str | None = None, credential: Any = None
    ) -> Iterator[Event]:
        try:
            lines = post_lines(
                f"{self.base_url}/api/chat",
                build_payload(request),
                {"Accept": "application/x-ndjson"},
                timeout=request.timeout,
                retry=Retry(attempts=2),  # a local box is either up or it is not
            )
            yield from parse(lines)
        except HTTPFailure as exc:
            yield StreamError(exc.detail(), status=exc.status, retryable=exc.retryable)
            yield Stop("error")
