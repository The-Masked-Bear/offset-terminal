"""Credentials: API keys and OAuth tokens, resolved the same way.

Two kinds reach a provider. An API key is a constant; an OAuth token expires
and has to be refreshed before the request rather than after it fails. Both
arrive here as a `Credential`, so nothing above this layer has to care which
one it got.

What this deliberately does NOT do: impersonate a Claude Pro/Max or ChatGPT
subscription. Those tokens are restricted server-side to their vendors' own
clients, third-party use returns 401, and it breaks the consumer terms. The
supported paths are an API key, or an OAuth provider that actually offers one
(see `offset.providers.oauth.APPS`).

Storage is `~/.offset/credentials.json` at mode 0600, and it stays compatible
with the flat `{provider: "key"}` shape the registry already reads.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final

from offset.providers import oauth
from offset.providers import registry
from offset.providers.registry import provider_for

def _dir() -> Path:
    """Resolved lazily: capturing the constant at import time would ignore both
    a later OFFSET_HOME and any test that redirects the config directory."""
    return registry.CONFIG_DIR


def _file() -> Path:
    return registry.CREDENTIALS_FILE


API_KEY: Final = "api_key"
OAUTH: Final = "oauth"

#: How each provider wants to be told who you are.
HEADERS: Final[dict[str, tuple[str, str]]] = {
    # provider: (api-key header, oauth header)
    "anthropic": ("x-api-key", "Authorization"),
    "openai": ("Authorization", "Authorization"),
    "openrouter": ("Authorization", "Authorization"),
    "deepseek": ("Authorization", "Authorization"),
    "google": ("x-goog-api-key", "Authorization"),
    "llamacpp": ("Authorization", "Authorization"),
    "opencode": ("Authorization", "Authorization"),
    "opencode-go": ("Authorization", "Authorization"),
    "ollama": ("", ""),
    "mock": ("", ""),
}


#: One error type across the whole auth stack. Two separately-defined
#: `AuthError` classes meant a caller could catch the wrong one and still crash.
AuthError = oauth.AuthError
AuthConfigError = oauth.AuthConfigError


@dataclass(slots=True)
class Credential:
    provider: str
    kind: str = API_KEY
    value: str = ""
    refresh_token: str | None = None
    expires_at: float | None = None
    account: str | None = None

    # -- state ------------------------------------------------------------

    def expired(self, skew: float = 60.0, *, now: Callable[[], float] = time.time) -> bool:
        """True when the token is past its life, or close enough to it.

        API keys never expire; treating them as expiring would send everyone
        through a refresh path that does not exist for them.
        """
        if self.kind != OAUTH or self.expires_at is None:
            return False
        return now() >= self.expires_at - skew

    @property
    def renewable(self) -> bool:
        return self.kind == OAUTH and bool(self.refresh_token)

    def header(self) -> tuple[str, str]:
        """The header this provider expects, ready to send."""
        api_header, oauth_header = HEADERS.get(self.provider, ("Authorization", "Authorization"))
        name = oauth_header if self.kind == OAUTH else api_header
        if not name:
            return ("", "")
        if name == "Authorization":
            return (name, f"Bearer {self.value}")
        return (name, self.value)

    def label(self) -> str:
        """What the UI shows.  Never the secret."""
        who = self.account or ("api key" if self.kind == API_KEY else "signed in")
        tail = ""
        if self.kind == OAUTH and self.expires_at:
            left = int(self.expires_at - time.time())
            tail = f", expires in {left // 60}m" if left > 0 else ", expired"
        return f"{self.provider}: {who}{tail}"

    # -- serialisation ----------------------------------------------------

    def to_data(self) -> Any:
        """A bare string for an API key, an object for anything richer."""
        if self.kind == API_KEY and not self.account:
            return self.value
        return {
            "kind": self.kind,
            "value": self.value,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "account": self.account,
        }

    @classmethod
    def from_data(cls, provider: str, raw: Any) -> "Credential | None":
        if isinstance(raw, str):
            return cls(provider=provider, kind=API_KEY, value=raw) if raw else None
        if not isinstance(raw, dict):
            return None
        value = raw.get("value")
        if not isinstance(value, str) or not value:
            return None
        return cls(
            provider=provider,
            kind=str(raw.get("kind") or API_KEY),
            value=value,
            refresh_token=raw.get("refresh_token"),
            # `is not None`, not truthiness: an expiry of 0 is falsy, and losing
            # it makes an already-dead token look like one that never expires.
            expires_at=(
                float(raw["expires_at"]) if raw.get("expires_at") is not None else None
            ),
            account=raw.get("account"),
        )

    @classmethod
    def from_token(cls, provider: str, token: oauth.Token) -> "Credential":
        return cls(
            provider=provider,
            kind=token.kind or OAUTH,
            value=token.value,
            refresh_token=token.refresh_token,
            expires_at=token.expires_at,
            account=token.account,
        )

    def __repr__(self) -> str:  # never let a secret reach a log or a traceback
        return f"<Credential {self.provider} {self.kind} account={self.account!r} redacted>"

    __str__ = __repr__


# -- the store --------------------------------------------------------------


def _read() -> dict[str, Any]:
    try:
        raw = json.loads(_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write(data: dict[str, Any]) -> None:
    _dir().mkdir(parents=True, exist_ok=True)
    target = _file()
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, target)
    os.chmod(target, 0o600)


def save(cred: Credential) -> None:
    data = _read()
    data[cred.provider] = cred.to_data()
    _write(data)


def forget(provider: str) -> bool:
    data = _read()
    if provider not in data:
        return False
    del data[provider]
    _write(data)
    return True


def stored(provider: str) -> Credential | None:
    return Credential.from_data(provider, _read().get(provider))


def from_env(provider: str) -> Credential | None:
    """An API key in the environment, which always wins over the store."""
    try:
        keys = provider_for(provider).env_keys
    except KeyError:
        keys = ()
    for key in (*keys, f"{provider.upper()}_API_KEY", f"OFFSET_{provider.upper()}_KEY"):
        value = os.environ.get(key)
        if value:
            return Credential(provider=provider, kind=API_KEY, value=value, account=f"${key}")
    return None


def load(provider: str, *, post: oauth.Poster = oauth.send, now: Callable[[], float] = time.time) -> Credential | None:
    """Environment first, then the store, refreshing an expired token in place.

    A refresh that fails does not raise: the stale credential is returned so the
    caller gets the provider's own 401 and a clear "sign in again" message,
    rather than an exception from the wrong layer.
    """
    found = from_env(provider)
    if found is not None:
        return found
    found = stored(provider)
    if found is None or not found.expired(now=now):
        return found
    if not found.renewable:
        return found
    try:
        token = oauth.refresh(oauth.app(provider), found.refresh_token or "", post=post, now=now)
    except Exception:
        return found
    fresh = Credential.from_token(provider, token)
    if not fresh.refresh_token:
        fresh.refresh_token = found.refresh_token  # some servers only send it once
    fresh.account = fresh.account or found.account
    save(fresh)
    return fresh


def accounts() -> list[Credential]:
    """Every credential we hold, environment included, for `/accounts`."""
    out: list[Credential] = []
    seen: set[str] = set()
    for provider in sorted(_read()):
        cred = stored(provider)
        if cred is not None:
            out.append(cred)
            seen.add(provider)
    for provider in ("anthropic", "openai", "google", "deepseek", "openrouter",
                     "opencode", "opencode-go"):
        if provider in seen:
            continue
        cred = from_env(provider)
        if cred is not None:
            out.append(cred)
    return out


# -- logging in -------------------------------------------------------------


def login_api_key(provider: str, key: str) -> Credential:
    key = key.strip()
    if not key:
        raise AuthError("no key given")
    cred = Credential(provider=provider, kind=API_KEY, value=key)
    save(cred)
    return cred


def oauth_providers() -> list[str]:
    """Providers we can actually sign into with a browser."""
    return sorted(oauth.APPS)


def missing_config(provider: str) -> tuple[str, ...]:
    """Settings a provider still needs before a browser flow can start."""
    try:
        return oauth.needs(provider)
    except oauth.AuthConfigError:
        return ("unknown provider",)


@dataclass(slots=True)
class LoginProgress:
    """What the UI shows while a browser flow is in flight."""

    url: str = ""
    user_code: str = ""
    message: str = ""
    done: bool = False


def login_browser(
    provider: str,
    *,
    announce: Callable[[LoginProgress], None] = lambda _p: None,
    timeout: float = 300.0,
) -> Credential:
    """Authorization-code + PKCE through a loopback redirect.

    Falls back to the device-code flow when the provider offers one and no
    browser could be opened, which is the normal case over SSH.
    """
    entry = oauth.app(provider)
    absent = oauth.needs(provider)
    if absent:
        raise AuthError(f"{provider} needs {', '.join(absent)} configured first")

    pkce = oauth.Pkce.create()
    state = oauth.new_state()
    server = oauth.start_loopback(state, host=entry.redirect_host, path=entry.redirect_path)
    try:
        redirect_uri = f"http://{entry.redirect_host}:{server.port}{entry.redirect_path}"
        url = oauth.authorize_url(entry, pkce, state, redirect_uri)
        opened = oauth.launch(url)
        announce(LoginProgress(
            url=url,
            message="opened your browser; finish there" if opened
            else "open this url to finish signing in",
        ))
        code = server.wait(timeout=timeout)
    finally:
        server.close()
    if not code:
        raise AuthError("the browser never came back with an authorisation code")
    token = oauth.exchange(entry, code, pkce, redirect_uri)
    cred = Credential.from_token(provider, token)
    save(cred)
    announce(LoginProgress(message=f"signed in to {provider}", done=True))
    return cred


def login_device(
    provider: str,
    *,
    announce: Callable[[LoginProgress], None] = lambda _p: None,
) -> Credential:
    """For headless boxes: show a code, poll until the user confirms."""
    entry = oauth.app(provider)
    if not entry.device_url:
        raise AuthError(f"{provider} does not offer a device flow")
    device = oauth.device_start(entry)
    announce(LoginProgress(url=device.verification_uri, user_code=device.user_code,
                           message="enter this code in your browser"))
    token = oauth.device_wait(entry, device)
    cred = Credential.from_token(provider, token)
    save(cred)
    announce(LoginProgress(message=f"signed in to {provider}", done=True))
    return cred


def redact(text: str, *creds: Credential | str | None) -> str:
    """Scrub secrets out of anything about to be shown."""
    for cred in creds:
        secret = cred.value if isinstance(cred, Credential) else cred
        if secret and len(secret) > 8:
            text = text.replace(secret, f"{secret[:3]}\u2026{secret[-2:]}")
    return text
