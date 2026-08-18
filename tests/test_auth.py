"""Credentials and the OAuth flows behind account sign-in.

No network here: the token endpoint is a stub, and the loopback server is
driven by a local request.  What is under test is the state machine — PKCE
correctness, expiry, refresh-once, storage permissions, and the rule that a
secret never reaches a log line.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import urllib.parse
import urllib.request

from dataclasses import replace

import pytest

from offset.providers import auth, oauth
from offset.providers.auth import API_KEY, OAUTH, Credential
from offset.providers.registry import credential as registry_credential
from offset.providers.registry import provider_for


def _poke(url: str) -> None:
    """Hit the callback once; a refused login answers 4xx, which is not an error here."""
    try:
        urllib.request.urlopen(url, timeout=5).read()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):

    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    for var in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "OPENAI_API_KEY",
                "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# -- PKCE -------------------------------------------------------------------


def test_the_code_challenge_follows_rfc7636():
    pkce = oauth.Pkce.create()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(pkce.verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    assert pkce.challenge == expected
    assert pkce.method == "S256"
    assert 43 <= len(pkce.verifier) <= 128, "RFC 7636 bounds"
    assert "=" not in pkce.challenge


def test_verifiers_are_not_reused():
    assert len({oauth.Pkce.create().verifier for _ in range(50)}) == 50
    assert len({oauth.new_state() for _ in range(50)}) == 50


# -- loopback ---------------------------------------------------------------


def test_the_loopback_captures_the_code():
    state = oauth.new_state()
    server = oauth.start_loopback(state, path="/callback")
    try:
        url = f"http://127.0.0.1:{server.port}/callback?code=the-code&state={state}"
        threading.Timer(0.05, lambda: urllib.request.urlopen(url, timeout=5).read()).start()
        assert server.wait(timeout=10) == "the-code"
    finally:
        server.close()


def test_a_mismatched_state_is_refused():
    """Without this check the callback is a CSRF hole."""
    server = oauth.start_loopback(oauth.new_state(), path="/callback")
    try:
        url = f"http://127.0.0.1:{server.port}/callback?code=x&state=not-the-state"
        threading.Timer(0.05, lambda: _poke(url)).start()
        with pytest.raises(oauth.AuthError) as caught:
            server.wait(timeout=3)
        assert "state mismatch" in str(caught.value)
    finally:
        server.close()


def test_a_provider_error_is_reported_verbatim():
    state = oauth.new_state()
    server = oauth.start_loopback(state, path="/callback")
    try:
        url = f"http://127.0.0.1:{server.port}/callback?error=access_denied&state={state}"
        threading.Timer(0.05, lambda: _poke(url)).start()
        with pytest.raises(oauth.AuthError) as caught:
            server.wait(timeout=3)
        assert "access_denied" in str(caught.value)
    finally:
        server.close()


# -- credentials ------------------------------------------------------------


def test_api_keys_never_expire():
    assert not Credential("anthropic", API_KEY, "sk-x").expired()
    assert not Credential("anthropic", API_KEY, "sk-x", expires_at=0).expired()


def test_an_oauth_token_expires_with_a_skew():
    token = Credential("google", OAUTH, "ya29", expires_at=1000.0)
    assert token.expired(now=lambda: 1000.0)
    assert token.expired(skew=60, now=lambda: 950.0), "inside the skew counts as expired"
    assert not token.expired(skew=60, now=lambda: 800.0)


def test_each_provider_gets_the_header_it_wants():
    assert Credential("anthropic", API_KEY, "sk-a").header() == ("x-api-key", "sk-a")
    assert Credential("google", API_KEY, "k").header() == ("x-goog-api-key", "k")
    assert Credential("openai", API_KEY, "sk-o").header() == ("Authorization", "Bearer sk-o")
    assert Credential("anthropic", OAUTH, "tok").header() == ("Authorization", "Bearer tok")
    assert Credential("ollama", API_KEY, "unused").header() == ("", "")


def test_a_credential_never_prints_its_secret():
    cred = Credential("openai", API_KEY, "sk-super-secret-value", account="me@example.com")
    for rendered in (repr(cred), str(cred), f"{cred}", cred.label()):
        assert "sk-super-secret-value" not in rendered
    assert "me@example.com" in cred.label()


def test_redaction_covers_credentials_and_bare_strings():
    cred = Credential("openai", API_KEY, "sk-abcdefghijklmnop")
    assert "sk-abcdefghijklmnop" not in auth.redact(f"failed with {cred.value}", cred)
    assert "plain-secret-value" not in auth.redact("saw plain-secret-value", "plain-secret-value")


# -- storage ----------------------------------------------------------------


def test_a_key_round_trips_at_mode_600(isolated):
    auth.login_api_key("anthropic", "sk-stored")
    assert auth.stored("anthropic").value == "sk-stored"
    assert oct((isolated / "credentials.json").stat().st_mode)[-3:] == "600"


def test_an_api_key_is_stored_as_a_plain_string(isolated):
    """Keeps the file readable, and compatible with the registry's flat shape."""
    auth.login_api_key("openai", "sk-flat")
    raw = json.loads((isolated / "credentials.json").read_text())
    assert raw["openai"] == "sk-flat"
    assert registry_credential(provider_for("openai")) == "sk-flat"


def test_an_oauth_token_is_stored_as_an_object_and_ignored_by_the_registry(isolated):
    auth.save(Credential("google", OAUTH, "ya29.token", refresh_token="r", expires_at=9e9))
    raw = json.loads((isolated / "credentials.json").read_text())
    assert raw["google"]["kind"] == OAUTH
    assert auth.stored("google").refresh_token == "r"
    # The registry only understands plain keys; an object must not become "{...}".
    assert registry_credential(provider_for("google")) is None


def test_a_key_typed_into_offset_beats_a_stale_vendor_variable(isolated, monkeypatch):
    """The regression this replaces cost a real user an afternoon.

    A vendor variable used to win. Somebody with an expired `GEMINI_API_KEY`
    exported from another tool pasted a working key into `/login`, was told it was
    stored, and then watched every request fail with the provider's own "API key
    not valid" - while offset quietly kept sending the expired one.
    """
    auth.login_api_key("anthropic", "typed-into-offset")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-shell-value")
    assert auth.load("anthropic").value == "typed-into-offset"


def test_a_vendor_variable_is_still_used_when_nothing_is_stored(isolated, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert auth.load("anthropic").value == "from-env"


def test_an_offset_specific_variable_wins_over_everything(isolated, monkeypatch):
    """`OFFSET_ANTHROPIC_KEY` names this program, so it cannot be a leftover."""
    auth.login_api_key("anthropic", "typed-into-offset")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-shell-value")
    monkeypatch.setenv("OFFSET_ANTHROPIC_KEY", "aimed-here")
    assert auth.load("anthropic").value == "aimed-here"


def test_forgetting_a_credential(isolated):
    auth.login_api_key("openai", "sk-x")
    assert auth.forget("openai")
    assert auth.stored("openai") is None
    assert not auth.forget("openai")


def test_an_empty_key_is_refused():
    with pytest.raises(auth.AuthError):
        auth.login_api_key("openai", "   ")


def test_a_corrupt_store_is_not_fatal(isolated):
    (isolated / "credentials.json").write_text("{ broken", encoding="utf-8")
    assert auth.stored("openai") is None
    auth.login_api_key("openai", "sk-recovered")
    assert auth.stored("openai").value == "sk-recovered"


def test_accounts_lists_disk_and_environment(isolated, monkeypatch):
    auth.login_api_key("openai", "sk-o")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    names = {c.provider for c in auth.accounts()}
    assert {"openai", "anthropic"} <= names
    assert all("sk-" not in c.label() for c in auth.accounts())


# -- refresh ----------------------------------------------------------------


def test_an_expired_token_is_refreshed_once_and_persisted(isolated, monkeypatch):
    auth.save(Credential("google", OAUTH, "old", refresh_token="r1", expires_at=0.0))
    calls: list[str] = []

    def fake_refresh(entry, refresh_token, *, post=None, now=None):
        calls.append(refresh_token)
        return oauth.Token(value="new", kind=OAUTH, refresh_token="r2", expires_at=4600.0)

    # `load` refuses to refresh through an unconfigured app, so point the lookup
    # at one that needs no registration.
    monkeypatch.setattr(oauth, "app", lambda _p: oauth.APPS["openrouter"])
    monkeypatch.setattr(oauth, "refresh", fake_refresh)
    got = auth.load("google", now=lambda: 1000.0)
    assert got.value == "new" and got.refresh_token == "r2"
    assert calls == ["r1"], "refresh must happen exactly once, with the stored token"
    assert auth.stored("google").value == "new", "the refreshed token must be saved"


def test_a_refresh_token_is_kept_when_the_server_omits_it(isolated, monkeypatch):
    auth.save(Credential("google", OAUTH, "old", refresh_token="keep-me", expires_at=0.0))
    monkeypatch.setattr(oauth, "app", lambda _p: oauth.APPS["openrouter"])
    monkeypatch.setattr(oauth, "refresh", lambda *a, **k: oauth.Token(
        value="new", kind=OAUTH, refresh_token=None, expires_at=9e9))
    assert auth.load("google", now=lambda: 10.0).refresh_token == "keep-me"


def test_a_failed_refresh_returns_the_stale_token_rather_than_raising(isolated, monkeypatch):
    auth.save(Credential("google", OAUTH, "stale", refresh_token="r", expires_at=0.0))
    monkeypatch.setattr(oauth, "refresh", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("refresh endpoint down")))
    got = auth.load("google", now=lambda: 5000.0)
    assert got is not None and got.value == "stale", "the provider's own 401 is the clearer error"


def test_a_token_without_a_refresh_token_is_not_refreshed(isolated, monkeypatch):
    auth.save(Credential("google", OAUTH, "orphan", expires_at=0.0))
    monkeypatch.setattr(oauth, "refresh", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not refresh without a refresh token")))
    assert auth.load("google", now=lambda: 9e9).value == "orphan"


def test_a_live_token_is_not_refreshed(isolated, monkeypatch):
    auth.save(Credential("google", OAUTH, "fresh", refresh_token="r", expires_at=9e9))
    monkeypatch.setattr(oauth, "refresh", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a live token must be used as-is")))
    assert auth.load("google").value == "fresh"


# -- what is actually offered ------------------------------------------------


def test_openrouter_is_ready_without_registration():
    """It is the one provider offering a public PKCE flow, so it needs nothing."""
    assert "openrouter" in auth.oauth_providers()
    assert auth.missing_config("openrouter") == ()


def test_google_asks_for_the_users_own_client_credentials():
    assert "google" in auth.oauth_providers()
    assert auth.missing_config("google"), "Google requires app registration"


def test_subscription_impersonation_is_not_offered():
    """Claude Pro/Max and ChatGPT tokens are vendor-locked; offering them would
    get the user a 401 and break the consumer terms."""
    assert "anthropic" not in oauth.APPS
    assert "openai" not in oauth.APPS
    assert "claude" not in oauth.APPS


def test_a_browser_login_refuses_when_config_is_missing():
    with pytest.raises(auth.AuthError) as caught:
        auth.login_browser("google")
    assert "clientId" in str(caught.value)
    assert isinstance(caught.value, oauth.AuthError), "one error type across the auth stack"


def test_an_unknown_provider_is_refused():
    with pytest.raises(oauth.AuthConfigError):
        oauth.app("definitely-not-a-provider")


def test_an_expiry_of_zero_survives_a_round_trip(isolated):
    """Regression: a falsy 0 expiry became None, so a dead token looked eternal."""
    auth.save(Credential("google", OAUTH, "tok", refresh_token="r", expires_at=0.0))
    back = auth.stored("google")
    assert back.expires_at == 0.0
    assert back.expired(now=lambda: 1000.0), "an epoch-0 expiry is long past"


# -- starting a sign-in without finishing it ---------------------------------


def test_begin_login_returns_something_showable_before_it_blocks(monkeypatch):
    """The url used to be computed, announced to a no-op, and discarded."""
    monkeypatch.setattr(oauth, "launch", lambda url: False)
    pending = auth.begin_login("openrouter")
    try:
        assert pending.url.startswith("https://"), pending.url
        assert pending.port > 0, "a loopback flow must say which port it is listening on"
        assert pending.kind == "loopback"
        assert not pending.opened, "no browser was opened, so it must not claim one was"
    finally:
        pending._server.close()


def test_begin_login_reports_when_a_browser_did_open(monkeypatch):
    monkeypatch.setattr(oauth, "launch", lambda url: True)
    pending = auth.begin_login("openrouter")
    try:
        assert pending.opened
    finally:
        pending._server.close()


def test_open_browser_false_never_launches_anything(monkeypatch):
    launched: list[str] = []
    monkeypatch.setattr(oauth, "launch", lambda url: launched.append(url) or True)
    pending = auth.begin_login("openrouter", open_browser=False)
    try:
        assert launched == [], "it must not touch the browser when told not to"
    finally:
        pending._server.close()


def test_a_device_flow_is_preferred_when_no_browser_could_open(monkeypatch):
    """The documented SSH fallback that was never actually implemented."""
    entry = oauth.app("openrouter")
    device = oauth.Device(user_code="WDJB-MJHT", verification_uri="https://example/device",
                          interval=5, expires_at=9e9, device_code="dc", complete_uri=None)
    monkeypatch.setattr(oauth, "launch", lambda url: False)
    monkeypatch.setattr(oauth, "app", lambda provider: replace(entry, device_url="https://example/device/code"))
    monkeypatch.setattr(oauth, "device_start", lambda entry, **kw: device)

    pending = auth.begin_login("openrouter")
    assert pending.kind == "device"
    assert pending.user_code == "WDJB-MJHT"
    assert pending.url == "https://example/device"
    assert pending.port == 0, "a device flow listens on nothing"


def test_a_loopback_flow_that_never_answers_is_an_error(monkeypatch):
    monkeypatch.setattr(oauth, "launch", lambda url: False)
    pending = auth.begin_login("openrouter")
    with pytest.raises(auth.AuthError):
        pending.finish(timeout=0.05)  # nobody ever visits the url


def test_a_finished_loopback_flow_stores_the_credential(monkeypatch):
    monkeypatch.setattr(oauth, "launch", lambda url: False)
    token = oauth.Token(value="tok-from-code", kind="oauth", refresh_token="r",
                        expires_at=9e9, account=None)
    monkeypatch.setattr(oauth, "exchange", lambda *a, **kw: token)

    pending = auth.begin_login("openrouter")
    threading.Timer(0.05, lambda: _reply_to(pending.url)).start()

    cred = pending.finish(timeout=5)
    assert cred.value == "tok-from-code"
    stored = auth.load("openrouter")
    assert stored is not None and stored.value == "tok-from-code", "it must persist, not just return"


def test_login_browser_still_works_end_to_end(monkeypatch, tmp_path):
    """The one-shot api is now composed from the split one; it must not drift."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    monkeypatch.setattr(oauth, "launch", lambda url: True)
    token = oauth.Token(value="tok", kind="oauth", refresh_token="r", expires_at=9e9, account=None)
    monkeypatch.setattr(oauth, "exchange", lambda *a, **kw: token)

    seen: list[auth.LoginProgress] = []
    monkeypatch.setattr(oauth, "launch", lambda url: _reply_to(url) or True)
    cred = auth.login_browser("openrouter", announce=seen.append, timeout=5)
    assert cred.value == "tok"
    assert seen[0].url, "it must announce the url before blocking"
    assert [p.done for p in seen][-1] is True, "it must announce that it finished"


def _reply_to(authorize_url: str) -> None:
    """Stand in for a browser: visit the callback the way the provider would.

    Reads the redirect target out of the authorize url rather than assuming it -
    each provider names that parameter differently, OpenRouter calls it
    `callback_url` - and echoes `state` only when it was given one.
    """
    query = urllib.parse.parse_qs(urllib.parse.urlparse(authorize_url).query)
    redirect = next(query[key][0] for key in ("redirect_uri", "callback_url", "redirect_url")
                    if key in query)
    reply = f"{redirect}?code=the-code"
    if "state" in query:
        reply += f"&state={query['state'][0]}"
    threading.Timer(0.05, lambda: _poke(reply)).start()


# -- the state parameter -----------------------------------------------------


def test_a_provider_that_takes_no_state_can_still_sign_in():
    """The regression that made the recommended zero-setup login impossible.

    `state` only travels with a `client_id`, so a public PKCE client like
    OpenRouter never receives one and cannot echo one back. The loopback used to
    demand it anyway, so every real sign-in died on "state mismatch".
    """
    entry = oauth.app("openrouter")
    assert not oauth.sends_state(entry), "openrouter is the public-client case"

    pkce, state = oauth.Pkce.create(), oauth.new_state()
    server = oauth.start_loopback(None, host=entry.redirect_host, path=entry.redirect_path)
    try:
        url = oauth.authorize_url(entry, pkce, state,
                                  f"http://{entry.redirect_host}:{server.port}{entry.redirect_path}")
        assert "state=" not in url, "no state is sent, so none may be required"
        _reply_to(url)
        assert server.wait(timeout=5) == "the-code"
    finally:
        server.close()


def test_a_registered_client_still_has_its_state_enforced():
    """Relaxing the check must not disarm CSRF protection where it applies."""
    state = oauth.new_state()
    server = oauth.start_loopback(state, host="127.0.0.1", path="/callback")
    try:
        threading.Timer(0.05, lambda: _poke(
            f"http://127.0.0.1:{server.port}/callback?code=x&state=forged")).start()
        with pytest.raises(auth.AuthError, match="state mismatch"):
            server.wait(timeout=5)
    finally:
        server.close()


def test_sends_state_follows_the_client_id():
    assert oauth.sends_state(replace(oauth.app("openrouter"), client_id="abc123"))
    assert not oauth.sends_state(replace(oauth.app("openrouter"), client_id=""))
