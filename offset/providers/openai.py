"""OpenAI chat completions, and every endpoint that copies it.

DeepSeek, Groq, Together, OpenRouter, vLLM and llama.cpp's server all speak
this shape, so they are configurations of this class rather than new code.
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
from offset.providers.sse import iter_sse, loads
from offset.providers.transport import HTTPFailure, Retry, post_lines

FINISH_REASONS = {
    "stop": "stop",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "length",
    "content_filter": "refusal",
}


def build_payload(request: Request, *, max_tokens_field: str = "max_tokens") -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for m in request.messages:
        if m.role == "tool":
            messages.append({"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.text})
            continue
        entry: dict[str, Any] = {"role": m.role, "content": m.text or None}
        if m.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": c.raw or json.dumps(c.args)},
                }
                for c in m.tool_calls
            ]
        if m.name:
            entry["name"] = m.name
        messages.append(entry)

    payload: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        max_tokens_field: request.max_tokens,
    }
    if request.tools:
        payload["tools"] = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
            for t in request.tools
        ]
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.stop:
        payload["stop"] = list(request.stop)
    return payload


def parse(lines: Iterator[bytes]) -> Iterator[Event]:
    for _, data in iter_sse(lines):
        if data.strip() == "[DONE]":
            continue
        obj = loads(data)
        if obj is None:
            continue
        if obj.get("error"):
            err = obj["error"]
            yield StreamError(str(err.get("message") if isinstance(err, dict) else err))
            continue
        usage = obj.get("usage")
        if isinstance(usage, dict):
            details = usage.get("prompt_tokens_details") or {}
            yield Usage(
                input=int(usage.get("prompt_tokens") or 0),
                output=int(usage.get("completion_tokens") or 0),
                cache_read=int(details.get("cached_tokens") or 0),
            )
        for choice in obj.get("choices") or ():
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                yield TextDelta(content)
            # DeepSeek-style reasoning models put chain-of-thought in its own field
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                yield ThinkingDelta(reasoning)
            for i, call in enumerate(delta.get("tool_calls") or ()):
                fn = call.get("function") or {}
                yield ToolCallDelta(
                    index=int(call.get("index", i)),
                    id=call.get("id"),
                    name=fn.get("name"),
                    args_delta=fn.get("arguments") or "",
                )
            reason = choice.get("finish_reason")
            if reason:
                yield Stop(FINISH_REASONS.get(reason, "stop"))


class OpenAI(Provider):
    name = "openai"
    env_keys = ("OPENAI_API_KEY",)

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        *,
        name: str | None = None,
        env_keys: tuple[str, ...] | None = None,
        max_tokens_field: str = "max_tokens",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_tokens_field = max_tokens_field
        if name:
            self.name = name
        if env_keys is not None:
            # `()` means "this endpoint takes no key" — never fall back to the
            # OpenAI default, or a local server receives a real credential.
            self.env_keys = env_keys

    def stream(self, request: Request, *, api_key: str | None = None) -> Iterator[Event]:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            lines = post_lines(
                f"{self.base_url}/chat/completions",
                build_payload(request, max_tokens_field=self.max_tokens_field),
                headers,
                timeout=request.timeout,
                retry=Retry(),
            )
            yield from parse(lines)
        except HTTPFailure as exc:
            yield StreamError(exc.detail(), status=exc.status, retryable=exc.retryable)
            yield Stop("error")


def deepseek() -> OpenAI:
    return OpenAI("https://api.deepseek.com/v1", name="deepseek", env_keys=("DEEPSEEK_API_KEY",))


def openrouter() -> OpenAI:
    return OpenAI("https://openrouter.ai/api/v1", name="openrouter", env_keys=("OPENROUTER_API_KEY",))


def llamacpp(base_url: str = "http://127.0.0.1:8080/v1") -> OpenAI:
    """A local llama.cpp server — no key, same protocol."""
    return OpenAI(base_url, name="llamacpp", env_keys=())
