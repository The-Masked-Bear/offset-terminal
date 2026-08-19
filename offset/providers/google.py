"""Google Gemini `streamGenerateContent`.

Two shape differences worth remembering: the assistant role is called `model`,
and function calls arrive whole rather than as JSON fragments — so a single
`ToolCallDelta` carries the complete argument object.
"""

from __future__ import annotations

import json
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

FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "refusal",
    "RECITATION": "refusal",
    "PROHIBITED_CONTENT": "refusal",
}


def build_payload(request: Request) -> dict[str, Any]:
    contents: list[dict[str, Any]] = []
    for m in request.messages:
        if m.role == "system":
            continue
        if m.role == "tool":
            contents.append({
                "role": "user",
                "parts": [{"functionResponse": {"name": m.name or m.tool_call_id or "tool", "response": {"result": m.text}}}],
            })
            continue
        parts: list[dict[str, Any]] = []
        if m.text:
            parts.append({"text": m.text})
        for call in m.tool_calls:
            part: dict[str, Any] = {"functionCall": {"name": call.name, "args": call.args}}
            if call.signature:
                part["thoughtSignature"] = call.signature
            parts.append(part)
        contents.append({"role": "model" if m.role == "assistant" else "user", "parts": parts or [{"text": ""}]})

    generation: dict[str, Any] = {"maxOutputTokens": request.max_tokens}
    if request.temperature is not None:
        generation["temperature"] = request.temperature
    if request.stop:
        generation["stopSequences"] = list(request.stop)
    if request.thinking_budget:
        generation["thinkingConfig"] = {"thinkingBudget": request.thinking_budget, "includeThoughts": True}

    payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
    system = request.system or next((m.text for m in request.messages if m.role == "system"), None)
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if request.tools:
        payload["tools"] = [{
            "functionDeclarations": [
                {"name": t.name, "description": t.description,
                 "parameters": normalise(t.schema, "google")}
                for t in request.tools
            ]
        }]
    return payload


def parse(lines: Iterator[bytes]) -> Iterator[Event]:
    slot = 0
    for _, data in iter_sse(lines):
        obj = loads(data)
        if obj is None:
            continue
        if obj.get("error"):
            err = obj["error"]
            yield StreamError(str(err.get("message") if isinstance(err, dict) else err), status=err.get("code") if isinstance(err, dict) else None)
            continue
        usage = obj.get("usageMetadata")
        if isinstance(usage, dict):
            yield Usage(
                input=int(usage.get("promptTokenCount") or 0),
                output=int(usage.get("candidatesTokenCount") or 0),
                cache_read=int(usage.get("cachedContentTokenCount") or 0),
            )
        for candidate in obj.get("candidates") or ():
            for part in (candidate.get("content") or {}).get("parts") or ():
                if part.get("text"):
                    if part.get("thought"):
                        yield ThinkingDelta(part["text"])
                    else:
                        yield TextDelta(part["text"])
                call = part.get("functionCall")
                if call:
                    yield ToolCallDelta(
                        index=slot,
                        id=call.get("id") or f"gemini_{slot}",
                        name=call.get("name"),
                        args_delta=json.dumps(call.get("args") or {}),
                        # Gemini 3 refuses the next request unless this comes
                        # back on the same part it arrived on.
                        signature=part.get("thoughtSignature"),
                    )
                    slot += 1
            reason = candidate.get("finishReason")
            if reason:
                yield Stop(FINISH_REASONS.get(reason, "stop"))


class Google(Provider):
    name = "google"
    env_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def __init__(self, base_url: str = "https://generativelanguage.googleapis.com/v1beta") -> None:
        self.base_url = base_url.rstrip("/")

    def stream(
        self, request: Request, *, api_key: str | None = None, credential: Any = None
    ) -> Iterator[Event]:
        url = f"{self.base_url}/models/{request.model}:streamGenerateContent?alt=sse"
        headers = auth_header(api_key, credential, {"x-goog-api-key": api_key} if api_key else {})
        try:
            lines = post_lines(url, build_payload(request), headers, timeout=request.timeout, retry=Retry())
            yield from parse(lines)
        except HTTPFailure as exc:
            yield StreamError(exc.detail(), status=exc.status, retryable=exc.retryable)
            yield Stop("error")

class GoogleAntigravity(Google):
    name = "google-antigravity"
    env_keys = ("ANTIGRAVITY_API_KEY", "GOOGLE_ANTIGRAVITY_API_KEY", "GEMINI_API_KEY")

    def stream(
        self, request: Request, *, api_key: str | None = None, credential: Any = None
    ) -> Iterator[Event]:
        # The model id includes the 'google-antigravity/' prefix; the backend needs the bare name.
        bare_model = request.model
        if bare_model.startswith("google-antigravity/"):
            bare_model = bare_model[len("google-antigravity/"):]
            
        req = Request(
            model=bare_model,
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stop=request.stop,
            timeout=request.timeout,
            thinking_budget=request.thinking_budget,
        )
        yield from super().stream(req, api_key=api_key, credential=credential)
