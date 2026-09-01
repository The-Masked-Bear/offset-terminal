"""OAuth 2.0 authorization code + PKCE, and the device-code fallback.

Why hand-rolled: the whole flow is stdlib plus a hash, and a dependency that
holds the keys to a user's account is a dependency you have to trust forever.

No invented constants: Every endpoint below was read off the provider's
own documentation or official open-source client. Anything we cannot verify
publicly defaults to None and raises `AuthConfigError` naming the key to set.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Final, Iterable

if TYPE_CHECKING:  # `http.server` costs a tenth of a second to import
    from http.server import HTTPServer

# `urllib.parse` above is cheap; `urllib.request`, `http.server` and `webbrowser`
# are not - between them they pull in http.client, email, socket and ssl, which
# is a third of a second on a Raspberry Pi. Nothing here is needed until somebody
# actually signs in, and the agent imports this module on every single start, so
# they are imported inside the three functions that use them.

#: RFC 8628 grant identifier.
DEVICE_GRANT: Final = "urn:ietf:params:oauth:grant-type:device_code"

#: RFC 8628 §3.5: the two errors that mean "keep polling".
PENDING: Final = frozenset({"authorization_pending", "slow_down"})

#: The page the browser lands on.  Deliberately plain: it is rendered by a
#: server that exists for one request and must never look like an app.
PAGE: Final = (
    b"<!doctype html><meta charset=utf-8><title>offset</title>"
    b"<body style=\"font:16px monospace;padding:3rem\">"
    b"<p>OFFSET HAS YOUR AUTHORISATION.</p><p>You can close this tab.</p></body>"
)

FAILED: Final = (
    b"<!doctype html><meta charset=utf-8><title>offset</title>"
    b"<body style=\"font:16px monospace;padding:3rem\">"
    b"<p>LOGIN REJECTED.</p><p>Return to the terminal for the reason.</p></body>"
)


class AuthError(Exception):
    """A login could not be completed.  Callers turn this into a message."""


class AuthConfigError(AuthError):
    """The flow is real but this machine has no credentials configured for it."""


# -- PKCE -------------------------------------------------------------------


def s256(verifier: str) -> str:
    """RFC 7636 §4.2: BASE64URL(SHA256(ASCII(verifier))), padding stripped."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


@dataclass(frozen=True, slots=True)
class Pkce:
    verifier: str
    challenge: str
    method: str = "S256"

    @classmethod
    def create(cls, verifier: str | None = None) -> Pkce:
        # 64 random bytes -> 86 chars, inside RFC 7636's 43..128 window.
        v = verifier or secrets.token_urlsafe(64)
        return cls(v, s256(v))


def new_state() -> str:
    """CSRF token for the redirect.  Compared with `secrets.compare_digest`."""
    return secrets.token_urlsafe(24)


# -- provider table ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OAuthApp:
    """One provider's flow, as documented by that provider.

    `needs` names the fields a user must supply themselves; `app()` refuses to
    build an app while any of them is empty, so a half-configured flow fails
    before it opens a browser rather than after.
    """

    provider: str
    authorize_url: str
    token_url: str
    client_id: str | None = None
    client_secret: str | None = None
    scopes: tuple[str, ...] = ()
    device_url: str | None = None
    redirect_host: str = "127.0.0.1"
    redirect_path: str = "/callback"
    #: Fixed loopback port, for a provider that registered one exact redirect
    #: URI and will refuse any other.  0 means "any free port", which is what
    #: most flows allow and what every provider here used before Antigravity.
    redirect_port: int = 0
    #: Query parameter carrying the loopback URL.  OpenRouter calls it
    #: `callback_url`; RFC 6749 calls it `redirect_uri`.
    redirect_param: str = "redirect_uri"
    #: "form" = RFC 6749 token endpoint.  "json" = OpenRouter's key exchange.
    exchange: str = "form"
    #: Field holding the secret in the token response.
    token_field: str = "access_token"
    #: What the exchange actually hands back.  OpenRouter mints a normal API
    #: key, so the credential is an api_key with no refresh cycle.
    kind: str = "oauth"
    #: Extra authorize-URL parameters (tuple of pairs: the app is frozen).
    auth_extra: tuple[tuple[str, str], ...] = ()
    #: Provider supports displaying the code for copy/paste when no loopback
    #: is reachable.  Set only where the provider documents it.
    paste_param: str = ""
    needs: tuple[str, ...] = ()
    note: str = ""


#: Endpoints verified 2026-08 against provider documentation.
#:
#: openrouter — https://openrouter.ai/docs/guides/overview/auth/oauth
#:   No client id exists or is required: the authorize URL takes callback_url +
#:   code_challenge + code_challenge_method, localhost callbacks are allowed on
#:   any port, and POST /api/v1/auth/keys returns {"key": ...} — a real API key
#:   the user owns and can revoke.  Omitting callback_url shows the code on
#:   screen for headless boxes (code_challenge then mandatory, 10 min expiry).
#: google — https://developers.google.com/identity/protocols/oauth2 and
#:   accounts.google.com/.well-known/openid-configuration.  Standard flow;
#:   client id and secret must be registered by the user (see `needs`).
APPS: Final[dict[str, OAuthApp]] = {
    "openrouter": OAuthApp(
        provider="openrouter",
        authorize_url="https://openrouter.ai/auth",
        token_url="https://openrouter.ai/api/v1/auth/keys",
        redirect_host="localhost",  # the host form OpenRouter documents
        redirect_param="callback_url",
        exchange="json",
        token_field="key",
        kind="api_key",
        paste_param="key_label",
        note="mints an API key you own; revoke it at openrouter.ai/keys",
    ),
    "google": OAuthApp(
        provider="google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        device_url="https://oauth2.googleapis.com/device/code",
        scopes=(
            "https://www.googleapis.com/auth/cloud-platform",
            "openid",
            "email",
        ),
        auth_extra=(("access_type", "offline"), ("prompt", "consent")),
        needs=("client_id", "client_secret"),
        note="requires your own OAuth client from console.cloud.google.com",
    ),
    "openai-chatgpt": OAuthApp(
        provider="openai-chatgpt",
        authorize_url="https://auth.openai.com/oauth/authorize",
        token_url="https://auth.openai.com/oauth/token",
        device_url="https://auth.openai.com/api/accounts/deviceauth/usercode",
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        scopes=(
            "openid",
            "profile",
            "email",
            "offline_access",
        ),
        redirect_host="localhost",
        redirect_path="/auth/callback",
        redirect_port=1455,
        auth_extra=(
            ("id_token_add_organizations", "true"),
            ("codex_cli_simplified_flow", "true"),
            ("originator", "codex_cli_rs"),
        ),
        note="Codex CLI sign-in; requires a ChatGPT Plus or Pro subscription",
    ),
    "claude-pro": OAuthApp(
        provider="claude-pro",
        authorize_url="https://claude.com/cai/oauth/authorize",
        token_url="https://platform.claude.com/v1/oauth/token",
        client_id="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        scopes=(
            "org:create_api_key",
            "user:profile",
            "user:inference",
            "user:sessions:claude_code",
            "user:mcp_servers",
            "user:file_upload",
        ),
        redirect_host="localhost",
        exchange="json_full",
        paste_param="redirect_uri",
        note="Claude Code sign-in; requires a Claude Pro or Max subscription",
    ),
    #: google-antigravity — https://antigravity.google/docs/cli/install/ for the
    #:   flow, and the Cloud Code Assist protocol for what the token is then used
    #:   against.  Standard Google OAuth with PKCE, but three details are not
    #:   negotiable: the redirect is a *fixed* port, because the registered URI is
    #:   exact; `cclog` and `experimentsandconfigs` are required alongside the
    #:   usual scopes; and the resulting token talks to cloudcode-pa, not to
    #:   generativelanguage.
    #:
    #:   Google does not publish the client id its own binary carries, and every
    #:   community client deliberately declines to embed it - a credential lifted
    #:   out of a shipped binary gets rotated, and takes every user with it.  So
    #:   this asks for yours, exactly as the `google` entry above does.  Create a
    #:   Desktop app client and give it the redirect URI printed by `/login`.
    "google-antigravity": OAuthApp(
        provider="google-antigravity",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=(
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/cclog",
            "https://www.googleapis.com/auth/experimentsandconfigs",
        ),
        redirect_host="localhost",
        redirect_path="/oauth-callback",
        redirect_port=51121,
        auth_extra=(("access_type", "offline"), ("prompt", "consent")),
        needs=("client_id", "client_secret"),
        note="sign in with your Google account; needs a Desktop OAuth client you own",
    ),
}


def _setting(key: str, default: Any = None) -> Any:
    """Read the settings layer if it is present; auth must not require it."""
    try:
        from offset.core import settings
    except ImportError:
        return default
    try:
        return settings.get(key, default)
    except (AttributeError, TypeError):
        return default


#: Environment names the wider ecosystem already uses.  Somebody who has
#: already configured another Antigravity client should not have to re-export
#: the same secret under a second name to satisfy us.
ENV_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "google-antigravity": ("ANTIGRAVITY",),
}


def env_names(provider: str, field_name: str) -> tuple[str, ...]:
    """Every environment variable that may carry one config field.

    Provider ids are hyphenated and shell variable names cannot be: `export
    GOOGLE-ANTIGRAVITY_CLIENT_ID=x` is a syntax error, so the name printed in
    the "configure this" message was one nobody could act on.  Hyphens
    normalise to underscores; the hyphenated form was never settable, so there
    is nothing to stay compatible with.
    """
    stem = provider.upper().replace("-", "_")
    field = field_name.upper()
    names = [f"OFFSET_{stem}_{field}", f"{stem}_{field}"]
    names.extend(f"{alias}_{field}" for alias in ENV_ALIASES.get(provider, ()))
    return tuple(names)


def _configured(provider: str, field_name: str) -> str | None:
    """Settings first, then env.  Empty strings count as absent."""
    camel = "".join(p.title() if i else p for i, p in enumerate(field_name.split("_")))
    value = _setting(f"auth.{provider}.{camel}")
    if not value:
        import os

        for name in env_names(provider, field_name):
            value = os.environ.get(name)
            if value:
                break
    return str(value) if value else None


def flows() -> list[str]:
    """Providers with a real OAuth flow, configured or not."""
    return sorted(APPS)


def app(provider: str) -> OAuthApp:
    """The flow for `provider`, overlaid with local configuration."""
    base = APPS.get(provider)
    if base is None:
        raise AuthConfigError(f"{provider} has no OAuth flow; use an API key")

    client_id = _configured(provider, "client_id") or base.client_id
    client_secret = _configured(provider, "client_secret") or base.client_secret
    raw_scopes = _setting(f"auth.{provider}.scopes")
    scopes = tuple(raw_scopes) if isinstance(raw_scopes, (list, tuple)) and raw_scopes else base.scopes

    resolved = OAuthApp(
        provider=base.provider,
        authorize_url=str(_setting(f"auth.{provider}.authorizeUrl") or base.authorize_url),
        token_url=str(_setting(f"auth.{provider}.tokenUrl") or base.token_url),
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
        device_url=base.device_url,
        redirect_host=base.redirect_host,
        redirect_path=base.redirect_path,
        redirect_param=base.redirect_param,
        redirect_port=base.redirect_port,
        exchange=base.exchange,
        token_field=base.token_field,
        kind=base.kind,
        auth_extra=base.auth_extra,
        paste_param=base.paste_param,
        needs=base.needs,
        note=base.note,
    )
    missing = _missing(resolved)
    if missing:
        camel = {"client_id": "clientId", "client_secret": "clientSecret"}
        keys = ", ".join(f"auth.{provider}.{camel.get(k, k)}" for k in missing)
        # The plain form, not the OFFSET_-prefixed one: shorter, and both work.
        env = ", ".join(env_names(provider, k)[1] for k in missing)
        hint = f"configure {keys} (or set {env})"
        if resolved.note:
            hint += f" \u2014 {resolved.note}"
        if resolved.redirect_port:
            # The failure this prevents is silent: an OAuth client registered
            # without this exact URI gets `redirect_uri_mismatch` at the very
            # end of the flow, after the user has already signed in.
            hint += (f"\nregister this redirect URI on the client: "
                     f"http://{resolved.redirect_host}:{resolved.redirect_port}"
                     f"{resolved.redirect_path}")
        raise AuthConfigError(hint)
    return resolved


def _missing(entry: OAuthApp) -> tuple[str, ...]:
    """Field names, in their real snake_case form, that are still empty."""
    return tuple(name for name in entry.needs if not getattr(entry, name, None))


def needs(provider: str) -> tuple[str, ...]:
    """Config keys still missing for `provider`; empty means ready to log in."""
    try:
        app(provider)
    except AuthConfigError as exc:
        base = APPS.get(provider)
        return _missing(base) if base else (str(exc),)
    return ()


# -- transport --------------------------------------------------------------

#: (url, form, json_body, headers, timeout) -> (status, decoded body).
#: Injected in tests; nothing here ever reaches a socket under pytest.
Poster = Callable[..., tuple[int, dict[str, Any]]]


def send(
    url: str,
    *,
    form: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    """POST and decode JSON, returning error bodies instead of raising.

    OAuth puts the interesting part of a failure *in* the 400 body
    (`{"error": "authorization_pending"}`), so an exception on non-2xx would
    throw away the only thing the caller needs.
    """
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        content = "application/json"
    else:
        body = urllib.parse.urlencode(form or {}).encode("utf-8")
        content = "application/x-www-form-urlencoded"
    sent = {"Content-Type": content, "Accept": "application/json", **(headers or {})}
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, data=body, headers=sent, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, _decode(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _decode(exc.read() if exc.fp else b"")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise AuthError(f"could not reach {urllib.parse.urlsplit(url).netloc}: {exc}") from exc


def _decode(raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid_response", "error_description": raw.decode("utf-8", "replace")[:200]}
    return data if isinstance(data, dict) else {"error": "invalid_response"}


# -- loopback redirect ------------------------------------------------------


@dataclass(slots=True)
class Loopback:
    """One-shot HTTP listener on 127.0.0.1 with an OS-assigned port.

    `state` is compared in constant time and a mismatch is fatal: a redirect we
    did not initiate must never be able to hand us a code.

    `handle_request` is driven from `wait()` rather than a thread so the caller
    keeps control of the timeout, and the handler's logging is silenced —
    BaseHTTPRequestHandler's default writes the full request line, complete
    with the authorization code, to stderr.
    """

    state: str | None
    path: str
    host: str
    _server: HTTPServer
    _code: str | None = None
    _error: str | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"

    def wait(self, timeout: float = 300.0, *, now: Callable[[], float] = time.monotonic) -> str:
        deadline = now() + timeout
        while self._code is None and self._error is None:
            left = deadline - now()
            if left <= 0:
                raise AuthError("timed out waiting for the browser redirect")
            self._server.timeout = min(left, 0.5)
            self._server.handle_request()
        if self._error:
            raise AuthError(self._error)
        return self._code or ""

    def close(self) -> None:
        self._server.server_close()

    def __enter__(self) -> Loopback:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def start_loopback(state: str | None, *, host: str = "127.0.0.1", path: str = "/callback",
                   port: int = 0) -> Loopback:
    """Bind the redirect listener.  Binds 127.0.0.1 whatever `host` we advertise.

    `state=None` means the authorize request carried no state, so there is
    nothing to compare against and any correct-path callback is accepted. The
    listener is still a single-use random port on the loopback interface.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer

    holder: dict[str, Loopback] = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self) -> None:
            loop = holder["loop"]
            parts = urllib.parse.urlsplit(self.path)
            if parts.path != loop.path:
                self._reply(404, b"not found")
                return
            query = urllib.parse.parse_qs(parts.query)
            got = (query.get("state") or [""])[0]
            error = (query.get("error") or [""])[0]
            code = (query.get("code") or [""])[0]
            if loop.state and not secrets.compare_digest(got, loop.state):
                loop._error = "state mismatch: that redirect did not come from the login you started"
                self._reply(400, FAILED)
                return
            if error:
                detail = (query.get("error_description") or [""])[0]
                loop._error = f"the provider refused: {error}{': ' + detail if detail else ''}"
                self._reply(400, FAILED)
                return
            if not code:
                loop._error = "the redirect carried no authorization code"
                self._reply(400, FAILED)
                return
            loop._code = code
            self._reply(200, PAGE)

        def _reply(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: Any) -> None:
            """Silence stderr: the default format includes the code."""

    server = HTTPServer(("127.0.0.1", port), Handler)
    loop = Loopback(state=state, path=path, host=host, _server=server)
    holder["loop"] = loop
    return loop


def launch(url: str) -> bool:
    """Try to open a browser.  False on a headless box, which is not an error."""
    try:
        import webbrowser

        return webbrowser.open(url)
    except (webbrowser.Error, OSError):
        return False


# -- authorization code flow ------------------------------------------------


def sends_state(entry: OAuthApp) -> bool:
    """Whether the authorize request will carry a `state` parameter.

    Only registered clients get one, because `state` travels with `client_id`.
    A provider that never received a state cannot echo one back, so the loopback
    must not demand it - requiring it unconditionally rejected every real
    OpenRouter sign-in with "state mismatch".
    """
    return bool(entry.client_id)


def authorize_url(entry: OAuthApp, pkce: Pkce, state: str, redirect_uri: str | None) -> str:
    """The URL the user visits.  `redirect_uri=None` selects paste mode."""
    params: dict[str, str] = {
        "code_challenge": pkce.challenge,
        "code_challenge_method": pkce.method,
    }
    if redirect_uri:
        params[entry.redirect_param] = redirect_uri
    elif entry.paste_param:
        params[entry.paste_param] = "https://platform.claude.com/oauth/code/callback" if entry.provider == "claude-pro" else "offset"
    else:
        raise AuthError(f"{entry.provider} needs a loopback redirect; it has no paste mode")
    if sends_state(entry):
        params["client_id"] = entry.client_id
        params["response_type"] = "code"
        params["state"] = state
    if entry.scopes:
        params["scope"] = " ".join(entry.scopes)
    params.update(dict(entry.auth_extra))
    join = "&" if urllib.parse.urlsplit(entry.authorize_url).query else "?"
    return f"{entry.authorize_url}{join}{urllib.parse.urlencode(params)}"


@dataclass(slots=True)
class Token:
    """What an exchange or refresh yielded, before it becomes a Credential."""

    value: str
    kind: str = "oauth"
    refresh_token: str | None = None
    expires_at: float | None = None
    account: str | None = None

    def __repr__(self) -> str:
        return f"Token(kind={self.kind!r}, account={self.account!r}, expires_at={self.expires_at!r})"


def exchange(
    entry: OAuthApp,
    code: str,
    pkce: Pkce,
    redirect_uri: str | None = None,
    *,
    post: Poster = send,
    now: Callable[[], float] = time.time,
) -> Token:
    if entry.exchange.startswith("json"):
        body = {
            "code": code,
            "code_verifier": pkce.verifier,
            "code_challenge_method": pkce.method,
        }
        if entry.exchange == "json_full":
            body["grant_type"] = "authorization_code"
            if redirect_uri:
                body["redirect_uri"] = redirect_uri
            if entry.client_id:
                body["client_id"] = entry.client_id
        status, data = post(entry.token_url, json_body=body)
    else:
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": pkce.verifier,
        }
        if redirect_uri:
            form["redirect_uri"] = redirect_uri
        if entry.client_id:
            form["client_id"] = entry.client_id
        if entry.client_secret:
            form["client_secret"] = entry.client_secret
        status, data = post(entry.token_url, form=form)
    return _token(entry, status, data, now=now)


def refresh(
    entry: OAuthApp,
    refresh_token: str,
    *,
    post: Poster = send,
    now: Callable[[], float] = time.time,
) -> Token:
    """Swap a refresh token for a fresh access token.

    Providers that rotate refresh tokens send a new one; Google does not, so the
    old one is carried forward.  Dropping it would turn every expiry into a
    re-login.
    """
    body = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    if entry.client_id:
        body["client_id"] = entry.client_id
    if entry.client_secret:
        body["client_secret"] = entry.client_secret
    if entry.exchange.startswith("json"):
        status, data = post(entry.token_url, json_body=body)
    else:
        status, data = post(entry.token_url, form=body)
    token = _token(entry, status, data, now=now)
    token.refresh_token = token.refresh_token or refresh_token
    return token


def _token(entry: OAuthApp, status: int, data: dict[str, Any], *, now: Callable[[], float]) -> Token:
    if status >= 400 or data.get("error"):
        raise AuthError(_reason(status, data))
    value = data.get(entry.token_field)
    if not isinstance(value, str) or not value:
        raise AuthError(f"{entry.provider} returned no {entry.token_field}")
    expires_in = data.get("expires_in")
    expires_at = now() + float(expires_in) if isinstance(expires_in, (int, float)) else None
    return Token(
        value=value,
        kind=entry.kind,
        refresh_token=data.get("refresh_token") or None,
        expires_at=expires_at,
        account=account_of(data.get("id_token"), provider=entry.provider),
        raw_token=data,
    )


def account_of(id_token: Any, provider: str = "") -> str | None:
    email = email_of(id_token)
    if provider == "openai-chatgpt" and isinstance(id_token, str):
        try:
            _, payload, _ = id_token.split(".")
            padding = "=" * (4 - (len(payload) % 4)) if len(payload) % 4 else ""
            claims = json.loads(base64.urlsafe_b64decode(payload + padding).decode("utf-8"))
            openai_auth = claims.get("https://api.openai.com/auth") or {}
            account_id = openai_auth.get("chatgpt_account_id")
            return account_id or email
        except Exception:
            pass
    return email


def _reason(status: int, data: dict[str, Any]) -> str:
    error = data.get("error")
    if isinstance(error, dict):  # Google nests it
        error = error.get("message") or error.get("status")
    detail = data.get("error_description") or data.get("message")
    parts = [str(p) for p in (error, detail) if p]
    return "; ".join(parts) or f"HTTP {status}"


def email_of(id_token: Any) -> str | None:
    """Email claim out of an OIDC id_token, for display only.

    The signature is deliberately not checked: this string is never used for a
    decision, only printed next to the account in the UI.  Treating it as
    authenticated would be a real vulnerability, so it never leaves the label.
    """
    if not isinstance(id_token, str) or id_token.count(".") != 2:
        return None
    payload = id_token.split(".")[1]
    try:
        raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        claims = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, json.JSONDecodeError):
        return None
    email = claims.get("email") if isinstance(claims, dict) else None
    return str(email) if email else None


# -- device code flow (RFC 8628) -------------------------------------------


@dataclass(slots=True)
class Device:
    user_code: str
    verification_uri: str
    interval: float
    expires_at: float
    device_code: str = field(repr=False, default="")
    complete_uri: str | None = None

    def prompt(self) -> str:
        return f"visit {self.complete_uri or self.verification_uri} and enter {self.user_code}"


def device_start(
    entry: OAuthApp,
    *,
    post: Poster = send,
    now: Callable[[], float] = time.time,
) -> Device:
    if not entry.device_url:
        raise AuthConfigError(f"{entry.provider} has no device flow")
    form = {"scope": " ".join(entry.scopes)}
    if entry.client_id:
        form["client_id"] = entry.client_id
    status, data = post(entry.device_url, form=form)
    if status >= 400 or data.get("error"):
        raise AuthError(_reason(status, data))
    code = data.get("device_code")
    user = data.get("user_code")
    uri = data.get("verification_uri") or data.get("verification_url")
    if not (code and user and uri):
        raise AuthError(f"{entry.provider} returned an incomplete device response")
    interval = data.get("interval")
    lifetime = data.get("expires_in")
    return Device(
        user_code=str(user),
        verification_uri=str(uri),
        # RFC 8628 §3.2: absent interval means 5 seconds, not "as fast as you like".
        interval=float(interval) if isinstance(interval, (int, float)) else 5.0,
        expires_at=now() + (float(lifetime) if isinstance(lifetime, (int, float)) else 600.0),
        device_code=str(code),
        complete_uri=str(data["verification_uri_complete"]) if data.get("verification_uri_complete") else None,
    )


def device_wait(
    entry: OAuthApp,
    device: Device,
    *,
    post: Poster = send,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> Token:
    """Poll until the user approves.

    `slow_down` adds 5s permanently (RFC 8628 §3.5) — resetting the interval
    afterwards is how a client gets itself rate-limited off the endpoint.
    """
    interval = device.interval
    form = {"grant_type": DEVICE_GRANT, "device_code": device.device_code}
    if entry.client_id:
        form["client_id"] = entry.client_id
    if entry.client_secret:
        form["client_secret"] = entry.client_secret
    while True:
        if now() >= device.expires_at:
            raise AuthError("the device code expired; start the login again")
        sleep(interval)
        status, data = post(entry.token_url, form=form)
        error = data.get("error") if isinstance(data.get("error"), str) else None
        if error in PENDING:
            if error == "slow_down":
                interval += 5.0
            continue
        return _token(entry, status, data, now=now)


def scopes_of(provider: str) -> Iterable[str]:
    """Used by the UI to show what a login will ask for."""
    entry = APPS.get(provider)
    return entry.scopes if entry else ()
