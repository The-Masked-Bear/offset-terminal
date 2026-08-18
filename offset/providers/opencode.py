"""OpenCode Zen and OpenCode Go.

Two gateways from the OpenCode team, and a genuinely unusual shape: a single
gateway speaks *three different wire protocols* depending on which model you
ask for. Claude and Qwen arrive over the Anthropic Messages API, DeepSeek and
GLM over OpenAI chat completions, GPT and Grok over the OpenAI Responses API,
Gemini over Google's own endpoint.

So the routing table is the substance of this module, and it is transcribed
from the published endpoint tables rather than guessed:

  * https://opencode.ai/docs/zen/  - pay as you go, ids like `opencode/glm-5.2`
  * https://opencode.ai/docs/go/   - $10/month subscription, `opencode-go/kimi-k3`

The two disagree, which is exactly why this is a table and not a heuristic:
MiniMax is chat-completions on Zen and Messages on Go.

What is deliberately not here: the OpenAI Responses API. offset does not speak
it, and a model routed there returns a clear refusal naming the reason instead
of a mystery failure. That covers GPT and Grok; every open model on Go except
those two works.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Final, Iterator

from offset.providers import anthropic as anthropic_wire
from offset.providers import google as google_wire
from offset.providers import openai as openai_wire
from offset.providers.base import Event, Provider, Request, Stop, StreamError
from offset.providers.transport import HTTPFailure, Retry, post_lines

ZEN_BASE: Final = "https://opencode.ai/zen"
GO_BASE: Final = "https://opencode.ai/zen/go"

#: The four shapes a gateway model can arrive in.
MESSAGES: Final = "messages"
CHAT: Final = "chat"
RESPONSES: Final = "responses"
GEMINI: Final = "gemini"

#: Prefix -> wire format, longest prefix wins. Transcribed from the docs.
ZEN_ROUTES: Final[tuple[tuple[str, str], ...]] = (
    ("claude-", MESSAGES),
    ("qwen", MESSAGES),
    ("gemini-", GEMINI),
    ("gpt-", RESPONSES),
    ("grok-", RESPONSES),
    ("muse-", RESPONSES),
    ("deepseek-", CHAT),
    ("minimax-", CHAT),
    ("glm-", CHAT),
    ("kimi-", CHAT),
    ("mimo-", CHAT),
    ("hy3", CHAT),
    ("big-pickle", CHAT),
    ("laguna-", CHAT),
    ("nemotron-", CHAT),
)

#: Go routes some families differently from Zen; MiniMax is the clearest case.
GO_ROUTES: Final[tuple[tuple[str, str], ...]] = (
    ("gpt-", RESPONSES),
    ("grok-", RESPONSES),
    ("minimax-", MESSAGES),
    ("qwen", MESSAGES),
    ("glm-", CHAT),
    ("kimi-", CHAT),
    ("deepseek-", CHAT),
    ("mimo-", CHAT),
    ("hy3", CHAT),
)


def route(model: str, routes: tuple[tuple[str, str], ...]) -> str:
    """Which wire format this model speaks.  Longest prefix wins."""
    name = model.strip().lower()
    for prefix in ("opencode-go/", "opencode/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    best = ""
    found = CHAT  # the gateway's most common shape, and the safest guess
    for prefix, kind in routes:
        if name.startswith(prefix) and len(prefix) > len(best):
            best, found = prefix, kind
    return found


def bare(model: str) -> str:
    """`opencode/glm-5.2` -> `glm-5.2`; the gateway wants the bare id."""
    for prefix in ("opencode-go/", "opencode/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


@dataclass(slots=True)
class Catalogue:
    models: list[dict[str, Any]]
    error: str = ""


class OpenCodeGateway(Provider):
    """Shared behaviour; `OpenCodeZen` and `OpenCodeGo` only differ by base URL."""

    name = "opencode"
    env_keys = ("OPENCODE_API_KEY",)
    base = ZEN_BASE
    routes = ZEN_ROUTES
    #: What the docs call the model prefix in a config file.
    prefix = "opencode"

    def __init__(self, base: str | None = None) -> None:
        if base:
            self.base = base.rstrip("/")

    # -- routing ----------------------------------------------------------

    def kind(self, model: str) -> str:
        return route(model, self.routes)

    def url_for(self, model: str) -> str:
        kind = self.kind(model)
        if kind == MESSAGES:
            return f"{self.base}/v1/messages"
        if kind == GEMINI:
            return f"{self.base}/v1/models/{bare(model)}:streamGenerateContent?alt=sse"
        return f"{self.base}/v1/chat/completions"

    # -- streaming --------------------------------------------------------

    def stream(
        self, request: Request, *, api_key: str | None = None, credential: Any = None
    ) -> Iterator[Event]:
        kind = self.kind(request.model)
        if kind == RESPONSES:
            yield StreamError(
                f"{bare(request.model)} is served over the OpenAI Responses API, "
                "which offset does not speak yet. Pick a model served over "
                "messages or chat completions - /models lists them.",
                status=None,
                retryable=False,
            )
            yield Stop("error")
            return

        token = credential.value if credential is not None else api_key
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        scoped = request.with_model(bare(request.model))

        if kind == MESSAGES:
            payload = anthropic_wire.build_payload(scoped)
            headers["anthropic-version"] = "2023-06-01"
            parse = anthropic_wire.parse
        elif kind == GEMINI:
            payload = google_wire.build_payload(scoped)
            parse = google_wire.parse
        else:
            payload = openai_wire.build_payload(scoped)
            parse = openai_wire.parse

        try:
            lines = post_lines(
                self.url_for(request.model),
                payload,
                headers,
                timeout=request.timeout,
                retry=Retry(),
            )
            yield from parse(lines)
        except HTTPFailure as exc:
            yield StreamError(exc.detail(), status=exc.status, retryable=exc.retryable)
            yield Stop("error")

    # -- discovery --------------------------------------------------------

    def catalogue(self, api_key: str | None = None, *, timeout: float = 20.0) -> Catalogue:
        """Ask the gateway what it serves today.

        The published table goes stale; the endpoint does not. Failure returns
        an empty catalogue with the reason rather than raising, because a model
        picker that cannot reach the network should still open.
        """
        request = urllib.request.Request(
            f"{self.base}/v1/models",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            return Catalogue([], f"could not reach {self.base}: {exc}")
        rows = body.get("data") if isinstance(body, dict) else body
        if not isinstance(rows, list):
            return Catalogue([], "the gateway returned an unexpected model list")
        return Catalogue([row for row in rows if isinstance(row, dict)])


class OpenCodeZen(OpenCodeGateway):
    """Pay-as-you-go access to the models the OpenCode team benchmarked."""

    name = "opencode"
    base = ZEN_BASE
    routes = ZEN_ROUTES
    prefix = "opencode"


class OpenCodeGo(OpenCodeGateway):
    """A $10/month subscription covering open coding models."""

    name = "opencode-go"
    env_keys = ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY")
    base = GO_BASE
    routes = GO_ROUTES
    prefix = "opencode-go"


def zen() -> OpenCodeZen:
    return OpenCodeZen()


def go() -> OpenCodeGo:
    return OpenCodeGo()
