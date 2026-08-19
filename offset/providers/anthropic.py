"""Anthropic Messages API."""

from __future__ import annotations

from typing import Any, Iterator

from offset.providers.base import (
    auth_header,
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
from offset.providers.schema import normalise
from offset.providers.sse import iter_sse, loads
from offset.providers.transport import HTTPFailure, Retry, post_lines

STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_use",
    "max_tokens": "length",
    "refusal": "refusal",
}


def build_payload(request: Request) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for m in request.messages:
        if m.role == "system":
            continue  # carried in the top-level `system` field
        if m.role == "tool":
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.text}],
            })
            continue
        blocks: list[dict[str, Any]] = []
        if m.text:
            blocks.append({"type": "text", "text": m.text})
        for call in m.tool_calls:
            blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.args})
        messages.append({"role": m.role, "content": blocks or [{"type": "text", "text": ""}]})

    system = request.system or next((m.text for m in request.messages if m.role == "system"), None)
    payload: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
        "stream": True,
    }
    if system:
        payload["system"] = system
    if request.tools:
        payload["tools"] = [
            {"name": t.name, "description": t.description,
             "input_schema": normalise(t.schema, "anthropic")}
            for t in request.tools
        ]
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.stop:
        payload["stop_sequences"] = list(request.stop)
    if request.thinking_budget:
        payload["thinking"] = {"type": "enabled", "budget_tokens": request.thinking_budget}
        payload.pop("temperature", None)  # rejected alongside extended thinking
    return payload


def parse(lines: Iterator[bytes]) -> Iterator[Event]:
    """Translate the Anthropic SSE stream into events."""
    for name, data in iter_sse(lines):
        obj = loads(data)
        if obj is None:
            continue
        kind = name or obj.get("type") or ""
        if kind == "message_start":
            usage = (obj.get("message") or {}).get("usage") or {}
            yield Usage(
                input=int(usage.get("input_tokens") or 0),
                cache_read=int(usage.get("cache_read_input_tokens") or 0),
                cache_write=int(usage.get("cache_creation_input_tokens") or 0),
            )
        elif kind == "content_block_start":
            block = obj.get("content_block") or {}
            if block.get("type") == "tool_use":
                yield ToolCallDelta(index=int(obj.get("index") or 0), id=block.get("id"), name=block.get("name"))
        elif kind == "content_block_delta":
            delta = obj.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                yield TextDelta(delta.get("text") or "")
            elif dtype == "thinking_delta":
                yield ThinkingDelta(delta.get("thinking") or "")
            elif dtype == "input_json_delta":
                yield ToolCallDelta(index=int(obj.get("index") or 0), args_delta=delta.get("partial_json") or "")
        elif kind == "message_delta":
            usage = obj.get("usage") or {}
            if usage.get("output_tokens"):
                yield Usage(output=int(usage["output_tokens"]))
            reason = (obj.get("delta") or {}).get("stop_reason")
            if reason:
                yield Stop(STOP_REASONS.get(reason, "stop"))
        elif kind == "error":
            err = obj.get("error") or {}
            yield StreamError(str(err.get("message") or err), retryable=err.get("type") == "overloaded_error")


class Anthropic(Provider):
    name = "anthropic"
    env_keys = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")

    def __init__(self, base_url: str = "https://api.anthropic.com", version: str = "2023-06-01") -> None:
        self.base_url = base_url.rstrip("/")
        self.version = version

    def stream(
        self, request: Request, *, api_key: str | None = None, credential: Any = None
    ) -> Iterator[Event]:
        headers = auth_header(api_key, credential, {"x-api-key": api_key or ""})
        headers["anthropic-version"] = self.version
        try:
            lines = post_lines(
                f"{self.base_url}/v1/messages",
                build_payload(request),
                headers,
                timeout=request.timeout,
                retry=Retry(),
            )
            yield from parse(lines)
        except HTTPFailure as exc:
            yield StreamError(exc.detail(), status=exc.status, retryable=exc.retryable)
            yield Stop("error")


class ClaudePro(Anthropic):
    name = "claude-pro"
    env_keys = ("CLAUDE_PRO_API_KEY",)

    def stream(
        self, request: Request, *, api_key: str | None = None, credential: Any = None
    ) -> Iterator[Event]:
        token = credential.value if credential is not None else api_key
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        headers["anthropic-version"] = self.version
        headers["anthropic-beta"] = "oauth-2025-04-20"
        try:
            lines = post_lines(
                f"{self.base_url}/v1/messages",
                build_payload(request),
                headers,
                timeout=request.timeout,
                retry=Retry(),
            )
            yield from parse(lines)
        except HTTPFailure as exc:
            yield StreamError(exc.detail(), status=exc.status, retryable=exc.retryable)
            yield Stop("error")
