"""Google Gemini `streamGenerateContent`, and Antigravity over Cloud Code Assist.

Two shape differences worth remembering: the assistant role is called `model`,
and function calls arrive whole rather than as JSON fragments — so a single
`ToolCallDelta` carries the complete argument object.

A signed-in Antigravity account is a third shape again, and `GoogleAntigravity`
below is where that is spelled out.
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, Final, Iterator

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


def parse(lines: Iterator[bytes], *, envelope: str = "") -> Iterator[Event]:
    """Gemini SSE frames into events.

    `envelope` names a field the real payload is wrapped in.  Cloud Code Assist
    wraps every frame in `{"response": {...}}` and the generative-language API
    does not, but what is inside the wrapper is the same body, so this is the
    only difference the two backends need between them.
    """
    slot = 0
    for _, data in iter_sse(lines):
        obj = loads(data)
        if obj is None:
            continue
        if envelope:
            inner = obj.get(envelope)
            if not isinstance(inner, dict):
                continue
            obj = inner
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

#: Where a signed-in Antigravity account is served from.  Deliberately not
#: generativelanguage: an account token and an API key are different
#: credentials against different services, and neither accepts the other's.
#: The `daily-` host is Google's staging environment with a fraction of the
#: capacity - pointing at it by accident is the documented cause of the
#: 503 MODEL_CAPACITY_EXHAUSTED reports, so it is spelled out here once.
CLOUDCODE: Final = "https://cloudcode-pa.googleapis.com"

#: Identifies the caller to Cloud Code.  The backend routes quota on this, so
#: a request without it may be answered without the account's Antigravity
#: allowance behind it.
CLIENT_METADATA: Final[dict[str, str]] = {
    "ideType": "ANTIGRAVITY",
    "platform": "PLATFORM_UNSPECIFIED",
    "pluginType": "GEMINI",
}

ANTIGRAVITY_PREFIX: Final = "google-antigravity/"


def bare(model: str) -> str:
    """`google-antigravity/gemini-3.1-pro` -> `gemini-3.1-pro`."""
    return model[len(ANTIGRAVITY_PREFIX):] if model.startswith(ANTIGRAVITY_PREFIX) else model


def cloudcode_headers(token: str, *, stream: bool = False) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": "antigravity",
        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    }


def cloudcode_call(method: str, token: str, body: dict[str, Any], *,
                   timeout: float = 30.0) -> dict[str, Any]:
    """One unary Cloud Code call.

    Reuses `post_lines` rather than opening its own connection: the retry
    policy and the `HTTPFailure` shape are worth more than the streaming it
    does not need.  Lines are rejoined with the newline they were split on, so
    the JSON arrives byte-identical.
    """
    raw = b"\n".join(post_lines(
        f"{CLOUDCODE}/v1internal:{method}",
        body,
        cloudcode_headers(token),
        timeout=timeout,
        retry=Retry(),
    ))
    try:
        parsed = json.loads(raw.decode("utf-8", "replace") or "{}")
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def load_code_assist(token: str, *, timeout: float = 30.0) -> dict[str, Any]:
    """Project id, tier and credit balance for the signed-in account."""
    return cloudcode_call("loadCodeAssist", token,
                          {"metadata": dict(CLIENT_METADATA)}, timeout=timeout)


def available_models(token: str, project: str, *, timeout: float = 30.0) -> list[tuple[str, str]]:
    """`(id, display name)` for everything this account may currently call.

    Models whose quota is exhausted are still returned: knowing a model exists
    and is spent is more useful than it silently vanishing from the picker.
    """
    payload = cloudcode_call("fetchAvailableModels", token, {"project": project}, timeout=timeout)
    found = payload.get("models")
    if not isinstance(found, dict):
        return []
    out = []
    for model_id, meta in found.items():
        label = ""
        if isinstance(meta, dict):
            label = str(meta.get("displayName") or "")
        out.append((str(model_id), label))
    return out


class GoogleAntigravity(Google):
    """A signed-in Google Antigravity account, over Cloud Code Assist.

    Three things differ from the plain Gemini provider, and all three are
    required rather than cosmetic: the request is wrapped in an envelope that
    names the account's Cloud project, every SSE frame comes back wrapped in a
    `response` field, and the project id has to be asked for before the first
    message can be sent.

    An API key still works and takes the ordinary Gemini path - that is the
    mode Antigravity itself calls `modelProvider: gemini`, and it is the right
    answer for a headless box with no browser.
    """

    name = "google-antigravity"
    env_keys = ("ANTIGRAVITY_API_KEY", "GOOGLE_ANTIGRAVITY_API_KEY", "GEMINI_API_KEY")

    def __init__(self, base_url: str = "https://generativelanguage.googleapis.com/v1beta") -> None:
        super().__init__(base_url)
        #: Discovered once per token.  Keyed by the token itself so a re-login
        #: as a different account cannot inherit the previous project.
        self._projects: dict[str, str] = {}

    def project_for(self, token: str) -> str:
        cached = self._projects.get(token)
        if cached is not None:
            return cached
        info = load_code_assist(token)
        project = str(info.get("cloudaicompanionProject") or "")
        self._projects[token] = project
        return project

    def stream(
        self, request: Request, *, api_key: str | None = None, credential: Any = None
    ) -> Iterator[Event]:
        model = bare(request.model)
        token = getattr(credential, "value", None)

        if not token:
            # No account, but a key: this is the documented API-key mode, and
            # the ordinary Gemini endpoint is exactly right for it.
            yield from super().stream(_retarget(request, model), api_key=api_key, credential=None)
            return

        try:
            project = self.project_for(token)
        except HTTPFailure as exc:
            yield StreamError(
                f"could not load your Antigravity project: {exc.detail()}",
                status=exc.status, retryable=exc.retryable,
            )
            yield Stop("error")
            return

        envelope: dict[str, Any] = {
            "model": model,
            "request": build_payload(_retarget(request, model)),
            "requestType": "agent",
            "userAgent": "antigravity",
            "requestId": f"offset-{int(time.time() * 1000)}-{secrets.token_hex(4)}",
        }
        if project:
            envelope["project"] = project

        try:
            lines = post_lines(
                f"{CLOUDCODE}/v1internal:streamGenerateContent?alt=sse",
                envelope,
                cloudcode_headers(token, stream=True),
                timeout=request.timeout,
                retry=Retry(),
            )
            yield from parse(lines, envelope="response")
        except HTTPFailure as exc:
            yield StreamError(exc.detail(), status=exc.status, retryable=exc.retryable)
            yield Stop("error")


def _retarget(request: Request, model: str) -> Request:
    """The same request against a bare model id."""
    if request.model == model:
        return request
    return Request(
        model=model,
        messages=request.messages,
        system=request.system,
        tools=request.tools,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stop=request.stop,
        timeout=request.timeout,
        thinking_budget=request.thinking_budget,
    )
