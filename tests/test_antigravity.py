"""Google Antigravity: signing in to an account rather than pasting a key.

Antigravity looked wired up before this - it appeared in `/login`, it took a
key, it answered - because it was quietly the plain Gemini provider wearing an
Antigravity label.  A signed-in account does not have an API key and does not
talk to generativelanguage at all, so the tests that matter are the ones that
pin *which backend gets the request* and *what shape it is in*.
"""

from __future__ import annotations

import json

import pytest

from offset.providers import google as g
from offset.providers import oauth
from offset.providers.base import Message, Request, Stop, TextDelta, ToolCallDelta


def sse(*frames: dict) -> list[bytes]:
    out: list[bytes] = []
    for frame in frames:
        out.append(b"data: " + json.dumps(frame).encode())
        out.append(b"")
    return out


class Credential:
    def __init__(self, value: str) -> None:
        self.value = value


# -- ids ----------------------------------------------------------------------


def test_the_prefix_is_stripped_before_the_backend_sees_it():
    assert g.bare("google-antigravity/gemini-3.1-pro") == "gemini-3.1-pro"


def test_a_bare_id_is_left_alone():
    assert g.bare("gemini-3.1-pro") == "gemini-3.1-pro"


def test_only_the_leading_prefix_is_removed():
    assert g.bare("x/google-antigravity/y") == "x/google-antigravity/y"


# -- the wire -----------------------------------------------------------------


def test_cloud_code_frames_are_unwrapped():
    """Every Cloud Code SSE frame arrives inside a `response` envelope.  Read
    without unwrapping, the candidates are invisible and the turn looks empty.
    """
    frames = sse({"response": {"candidates": [
        {"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}
    ]}})
    events = list(g.parse(iter(frames), envelope="response"))
    assert any(isinstance(e, TextDelta) and e.text == "hello" for e in events)
    assert any(isinstance(e, Stop) for e in events)


def test_the_plain_gemini_shape_still_parses_without_an_envelope():
    frames = sse({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
    assert any(isinstance(e, TextDelta) for e in g.parse(iter(frames)))


def test_a_frame_missing_its_envelope_is_skipped_not_crashed():
    frames = sse({"traceId": "abc"}, {"response": {"candidates": [
        {"content": {"parts": [{"text": "ok"}]}}
    ]}})
    texts = [e.text for e in g.parse(iter(frames), envelope="response")
             if isinstance(e, TextDelta)]
    assert texts == ["ok"]


def test_a_tool_call_survives_the_envelope():
    frames = sse({"response": {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "read", "args": {"path": "a.py"}}}
    ]}}]}})
    calls = [e for e in g.parse(iter(frames), envelope="response")
             if isinstance(e, ToolCallDelta)]
    assert len(calls) == 1
    assert calls[0].name == "read"


# -- routing ------------------------------------------------------------------


def request(model: str = "google-antigravity/gemini-3.1-pro") -> Request:
    return Request(model=model, messages=[Message("user", "hi")], max_tokens=16)


def test_an_account_token_goes_to_cloud_code(monkeypatch):
    """The point of the whole file: a signed-in account must not be sent to
    generativelanguage, which does not accept its token."""
    seen: dict = {}

    def fake_post(url, payload, headers, *, timeout=None, retry=None):
        seen["url"], seen["payload"], seen["headers"] = url, payload, headers
        return iter(sse({"response": {"candidates": [
            {"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}
        ]}}))

    monkeypatch.setattr(g, "post_lines", fake_post)
    provider = g.GoogleAntigravity()
    provider._projects["TOKEN"] = "proj-1"  # skip the discovery round trip

    events = list(provider.stream(request(), credential=Credential("TOKEN")))

    assert g.CLOUDCODE in seen["url"]
    assert "streamGenerateContent" in seen["url"]
    assert "generativelanguage" not in seen["url"]
    assert seen["headers"]["Authorization"] == "Bearer TOKEN"
    assert any(isinstance(e, TextDelta) for e in events)


def test_the_request_is_wrapped_in_the_cloud_code_envelope(monkeypatch):
    seen: dict = {}

    def fake_post(url, payload, headers, *, timeout=None, retry=None):
        seen.update(payload)
        return iter(sse())

    monkeypatch.setattr(g, "post_lines", fake_post)
    provider = g.GoogleAntigravity()
    provider._projects["TOKEN"] = "proj-1"
    list(provider.stream(request(), credential=Credential("TOKEN")))

    assert seen["project"] == "proj-1"
    assert seen["model"] == "gemini-3.1-pro", "the prefix must not reach the backend"
    assert seen["requestType"] == "agent"
    assert "contents" in seen["request"], "the Gemini body belongs under `request`"


def test_an_api_key_takes_the_ordinary_gemini_path(monkeypatch):
    """Antigravity documents an API-key mode, and it is the right answer on a
    headless box.  It is the generativelanguage endpoint, not Cloud Code."""
    seen: dict = {}

    def fake_post(url, payload, headers, *, timeout=None, retry=None):
        seen["url"] = url
        return iter(sse())

    monkeypatch.setattr(g, "post_lines", fake_post)
    list(g.GoogleAntigravity().stream(request(), api_key="KEY"))

    assert "generativelanguage" in seen["url"]
    assert g.CLOUDCODE not in seen["url"]
    assert "gemini-3.1-pro:streamGenerateContent" in seen["url"]


def test_the_project_is_discovered_once_per_token(monkeypatch):
    """A round trip before every message would double the latency of a chat."""
    calls: list[str] = []

    def fake_post(url, payload, headers, *, timeout=None, retry=None):
        calls.append(url)
        if "loadCodeAssist" in url:
            return iter([json.dumps({"cloudaicompanionProject": "proj-9"}).encode()])
        return iter(sse())

    monkeypatch.setattr(g, "post_lines", fake_post)
    provider = g.GoogleAntigravity()
    for _ in range(3):
        list(provider.stream(request(), credential=Credential("TOKEN")))

    assert sum("loadCodeAssist" in u for u in calls) == 1
    assert provider._projects["TOKEN"] == "proj-9"


def test_a_different_account_does_not_inherit_the_first_project(monkeypatch):
    """Keyed by token, so re-logging in as somebody else cannot bill the
    previous account's project."""
    def fake_post(url, payload, headers, *, timeout=None, retry=None):
        if "loadCodeAssist" in url:
            return iter([json.dumps({"cloudaicompanionProject": "proj-B"}).encode()])
        return iter(sse())

    monkeypatch.setattr(g, "post_lines", fake_post)
    provider = g.GoogleAntigravity()
    provider._projects["TOKEN-A"] = "proj-A"
    assert provider.project_for("TOKEN-B") == "proj-B"
    assert provider._projects["TOKEN-A"] == "proj-A"


def test_a_failed_project_lookup_is_reported_not_raised(monkeypatch):
    from offset.providers.transport import HTTPFailure

    def fake_post(url, payload, headers, *, timeout=None, retry=None):
        raise HTTPFailure(403, '{"error":{"message":"no access"}}')

    monkeypatch.setattr(g, "post_lines", fake_post)
    events = list(g.GoogleAntigravity().stream(request(), credential=Credential("T")))
    assert any(getattr(e, "status", None) == 403 for e in events)
    assert any(isinstance(e, Stop) for e in events)


def test_the_model_list_comes_back_with_its_display_names(monkeypatch):
    def fake_post(url, payload, headers, *, timeout=None, retry=None):
        body = {"models": {"gemini-3.1-pro": {"displayName": "Gemini 3.1 Pro"}}}
        return iter([json.dumps(body).encode()])

    monkeypatch.setattr(g, "post_lines", fake_post)
    assert g.available_models("T", "proj") == [("gemini-3.1-pro", "Gemini 3.1 Pro")]


# -- the sign-in flow ---------------------------------------------------------


def test_antigravity_offers_a_browser_sign_in():
    """The reported complaint: it asked for an API key when the user wanted to
    link an account."""
    assert "google-antigravity" in oauth.flows()


def test_the_flow_points_at_the_registered_redirect():
    """Antigravity registers one exact URI.  Any other port is refused at the
    end of the flow, after the user has already signed in."""
    entry = oauth.APPS["google-antigravity"]
    assert entry.redirect_port == 51121
    assert entry.redirect_path == "/oauth-callback"


def test_the_scopes_cloud_code_requires_are_all_requested():
    scopes = oauth.APPS["google-antigravity"].scopes
    for needed in ("cloud-platform", "userinfo.email", "cclog", "experimentsandconfigs"):
        assert any(needed in s for s in scopes), needed


@pytest.mark.parametrize("provider", ["google-antigravity", "claude-pro", "openai-chatgpt"])
def test_environment_names_are_actually_settable(provider):
    """`export GOOGLE-ANTIGRAVITY_CLIENT_ID=x` is a shell syntax error, so the
    name offset printed was one nobody could act on."""
    for name in oauth.env_names(provider, "client_id"):
        assert "-" not in name, name
        assert name.replace("_", "").isalnum()


def test_the_documented_ecosystem_variable_is_accepted(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_CLIENT_ID", "from-env")
    assert oauth._configured("google-antigravity", "client_id") == "from-env"


def test_an_unconfigured_flow_names_the_redirect_uri_to_register():
    """The silent failure this prevents: a client registered without the exact
    URI fails with `redirect_uri_mismatch` only at the very end."""
    with pytest.raises(oauth.AuthConfigError) as caught:
        oauth.app("google-antigravity")
    message = str(caught.value)
    assert "51121/oauth-callback" in message
    assert "GOOGLE_ANTIGRAVITY_CLIENT_ID" in message


def test_the_oauth_app_declares_each_field_exactly_once():
    """The class body listed every field twice, which is how `redirect_port`
    nearly went missing."""
    import dataclasses

    names = [f.name for f in dataclasses.fields(oauth.OAuthApp)]
    assert len(names) == len(set(names))
