"""OpenCode Zen and OpenCode Go.

One gateway speaking three wire protocols is unusual enough that the routing
table is the whole risk: send Claude to chat/completions and it fails in a way
nobody can debug from the error. Every route here is checked against the
published endpoint tables at opencode.ai/docs/zen and /docs/go.
"""

from __future__ import annotations

import json

import pytest

from offset.providers import opencode
from offset.providers.base import Message, Request, Stop, StreamError, TurnBuilder
from offset.providers.opencode import CHAT, GEMINI, MESSAGES, RESPONSES, OpenCodeGo, OpenCodeZen
from offset.providers.registry import PROVIDERS, info, provider_for, resolve


def ask(model: str) -> Request:
    return Request(model=model, messages=[Message("user", "hello")], max_tokens=32)


# -- routing, straight from the published tables ----------------------------


@pytest.mark.parametrize("model,expected", [
    ("claude-opus-5", MESSAGES),
    ("claude-sonnet-4-6", MESSAGES),
    ("qwen3.7-max", MESSAGES),
    ("qwen3.5-plus", MESSAGES),
    ("deepseek-v4-pro", CHAT),
    ("minimax-m3", CHAT),          # chat on Zen...
    ("glm-5.2", CHAT),
    ("kimi-k2.7-code", CHAT),
    ("big-pickle", CHAT),
    ("mimo-v2.5-free", CHAT),
    ("nemotron-3-ultra-free", CHAT),
    ("gemini-3-flash", GEMINI),
    ("gpt-5.5", RESPONSES),
    ("grok-4.6", RESPONSES),
    ("muse-spark-1.2", RESPONSES),
])
def test_zen_routes_each_family_to_its_documented_protocol(model, expected):
    assert OpenCodeZen().kind(model) == expected


@pytest.mark.parametrize("model,expected", [
    ("minimax-m3", MESSAGES),      # ...but Messages on Go
    ("minimax-m2.7", MESSAGES),
    ("qwen3.8-max", MESSAGES),
    ("glm-5.3", CHAT),
    ("kimi-k3", CHAT),
    ("deepseek-v4-flash", CHAT),
    ("mimo-v2.5-pro", CHAT),
    ("hy3", CHAT),
    ("gpt-5.6-luna", RESPONSES),
    ("grok-4.5", RESPONSES),
])
def test_go_routes_differ_from_zen_where_the_docs_say_they_do(model, expected):
    assert OpenCodeGo().kind(model) == expected


def test_the_two_gateways_genuinely_disagree_about_minimax():
    """The reason this is a table and not a heuristic."""
    assert OpenCodeZen().kind("minimax-m3") == CHAT
    assert OpenCodeGo().kind("minimax-m3") == MESSAGES


def test_the_prefixed_id_form_routes_the_same():
    assert OpenCodeZen().kind("opencode/claude-opus-5") == MESSAGES
    assert OpenCodeGo().kind("opencode-go/kimi-k3") == CHAT


def test_an_unknown_model_falls_back_to_chat_completions():
    """The gateway's most common shape, and the least surprising guess."""
    assert OpenCodeZen().kind("something-released-tomorrow") == CHAT


# -- urls -------------------------------------------------------------------


def test_zen_urls_match_the_documented_endpoints():
    zen = OpenCodeZen()
    assert zen.url_for("claude-opus-5") == "https://opencode.ai/zen/v1/messages"
    assert zen.url_for("glm-5.2") == "https://opencode.ai/zen/v1/chat/completions"
    assert zen.url_for("gemini-3-flash").startswith(
        "https://opencode.ai/zen/v1/models/gemini-3-flash:streamGenerateContent"
    )


def test_go_urls_are_under_the_go_prefix():
    go = OpenCodeGo()
    assert go.url_for("kimi-k3") == "https://opencode.ai/zen/go/v1/chat/completions"
    assert go.url_for("minimax-m3") == "https://opencode.ai/zen/go/v1/messages"


def test_the_gateway_is_sent_the_bare_model_id():
    assert opencode.bare("opencode/glm-5.2") == "glm-5.2"
    assert opencode.bare("opencode-go/kimi-k3") == "kimi-k3"
    assert opencode.bare("glm-5.2") == "glm-5.2"


# -- the protocol offset does not speak -------------------------------------


def test_a_responses_model_refuses_clearly_instead_of_failing_oddly():
    events = list(OpenCodeZen().stream(ask("opencode/gpt-5.5"), api_key="k"))
    errors = [e for e in events if isinstance(e, StreamError)]
    assert errors, "a model we cannot speak to must say so"
    assert "Responses API" in errors[0].message
    assert "gpt-5.5" in errors[0].message
    assert not errors[0].retryable, "retrying an unsupported protocol is pointless"
    assert isinstance(events[-1], Stop) and events[-1].reason == "error"


def test_the_refusal_never_reaches_the_network(monkeypatch):
    from offset.providers import opencode as mod

    monkeypatch.setattr(mod, "post_lines", lambda *a, **k: pytest.fail("must not call out"))
    turn = TurnBuilder().consume(OpenCodeZen().stream(ask("grok-4.6"), api_key="k")).finish()
    assert turn.stop_reason == "error"


# -- payload and auth -------------------------------------------------------


def test_each_route_builds_the_payload_its_protocol_expects(monkeypatch):
    from offset.providers import opencode as mod

    seen: dict[str, tuple] = {}

    def capture(url, payload, headers, **kwargs):
        seen["call"] = (url, payload, headers)
        return iter([])

    monkeypatch.setattr(mod, "post_lines", capture)

    list(OpenCodeZen().stream(ask("opencode/claude-opus-5"), api_key="secret"))
    url, payload, headers = seen["call"]
    assert url.endswith("/v1/messages")
    assert payload["model"] == "claude-opus-5", "the prefix must be stripped"
    assert "max_tokens" in payload and headers["anthropic-version"] == "2023-06-01"
    assert headers["Authorization"] == "Bearer secret"

    list(OpenCodeGo().stream(ask("opencode-go/glm-5.3"), api_key="secret"))
    url, payload, headers = seen["call"]
    assert url.endswith("/zen/go/v1/chat/completions")
    assert payload["model"] == "glm-5.3"
    assert payload["stream"] is True


def test_a_credential_is_preferred_over_a_bare_key(monkeypatch):
    from offset.providers import opencode as mod
    from offset.providers.auth import Credential

    seen: dict[str, dict] = {}
    monkeypatch.setattr(mod, "post_lines",
                        lambda url, payload, headers, **k: seen.update(h=headers) or iter([]))
    cred = Credential("opencode", "api_key", "from-credential")
    list(OpenCodeZen().stream(ask("glm-5.2"), api_key="from-arg", credential=cred))
    assert seen["h"]["Authorization"] == "Bearer from-credential"


def test_an_http_failure_becomes_an_event(monkeypatch):
    from offset.providers import opencode as mod
    from offset.providers.transport import HTTPFailure

    def boom(*a, **k):
        raise HTTPFailure(402, json.dumps({"error": {"message": "add credits"}}))

    monkeypatch.setattr(mod, "post_lines", boom)
    events = list(OpenCodeZen().stream(ask("glm-5.2"), api_key="k"))
    assert any(isinstance(e, StreamError) and "add credits" in e.message for e in events)


# -- discovery --------------------------------------------------------------


def test_the_catalogue_degrades_instead_of_raising(monkeypatch):
    import urllib.error

    from offset.providers import opencode as mod

    monkeypatch.setattr(mod.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    got = OpenCodeZen().catalogue("key")
    assert got.models == [] and "could not reach" in got.error


# -- registration -----------------------------------------------------------


def test_both_gateways_are_registered():
    assert PROVIDERS["opencode"] is OpenCodeZen
    assert PROVIDERS["opencode-go"] is OpenCodeGo
    assert provider_for("opencode").name == "opencode"
    assert provider_for("opencode-go").env_keys[0] == "OPENCODE_GO_API_KEY"


def test_prefixed_model_ids_resolve_to_the_right_gateway():
    provider, meta = resolve("opencode/glm-5.2")
    assert provider.name == "opencode" and meta.provider == "opencode"
    provider, meta = resolve("opencode-go/kimi-k3")
    assert provider.name == "opencode-go"
    # an id released after this catalogue was written must still work
    assert info("opencode/brand-new-model").provider == "opencode"


def test_the_catalogue_only_advertises_models_offset_can_speak_to():
    from offset.providers.registry import MODELS

    listed = [m for m in MODELS if m.provider.startswith("opencode")]
    assert listed, "the gateways should appear in the picker"
    for meta in listed:
        gateway = provider_for(meta.provider)
        assert gateway.kind(meta.id) != RESPONSES, (
            f"{meta.id} is listed but offset cannot speak its protocol"
        )


def test_they_carry_the_right_header_shape():
    from offset.providers.auth import Credential

    assert Credential("opencode", "api_key", "k").header() == ("Authorization", "Bearer k")
    assert Credential("opencode-go", "api_key", "k").header() == ("Authorization", "Bearer k")
