"""Credentials: API keys and OAuth tokens, resolved the same way.

Two kinds reach a provider. An API key is a constant; an OAuth token expires
and has to be refreshed before the request rather than after it fails. Both
arrive here as a `Credential`, so nothing above this layer has to care which
one it got.

Storage is `~/.offset/credentials.json` at mode 0600.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

from offset.providers import oauth
from offset.providers import registry
from offset.providers.registry import provider_for

def _dir() -> Path:
    """Resolved lazily: capturing the constant at import time would ignore both
    a later OFFSET_HOME and any test that redirects the config directory."""
    return registry.config_dir()


def _file() -> Path:
    return registry.credentials_file()


API_KEY: Final = "api_key"
OAUTH: Final = "oauth"

#: How each provider wants to be told who you are.
HEADERS: Final[dict[str, tuple[str, str]]] = {
    # provider: (api-key header, oauth header)
    "anthropic": ("x-api-key", "Authorization"),
    "claude-pro": ("x-api-key", "Authorization"),
    "openai": ("Authorization", "Authorization"),
    "openai-chatgpt": ("Authorization", "Authorization"),
    "openrouter": ("Authorization", "Authorization"),
    "deepseek": ("Authorization", "Authorization"),
    "google": ("x-goog-api-key", "Authorization"),
    "google-antigravity": ("x-goog-api-key", "Authorization"),
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
    raw_token: dict[str, Any] | None = None

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
            "raw_token": self.raw_token,
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
            raw_token=raw.get("raw_token"),
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
            raw_token=getattr(token, "raw_token", None),
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
    """The stored credential first, then the environment.

    Matches `registry.source`: a credential established through `/login` beats an
    ambient vendor variable, because an expired `GEMINI_API_KEY` left over from
    another tool used to silently override a key the person had just pasted in.
    An OAuth token can only come from the store, so this also keeps a live
    signed-in session ahead of a stale shell variable.

    A refresh that fails does not raise: the stale credential is returned so the
    caller gets the provider's own 401 and a clear "sign in again" message,
    rather than an exception from the wrong layer.
    """
    aimed = os.environ.get(f"OFFSET_{provider.upper()}_KEY")
    if aimed:
        return Credential(provider=provider, value=aimed, kind=API_KEY)
    found = stored(provider)
    if found is None:
        return from_env(provider)
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
    for provider in ("anthropic", "claude-pro", "openai", "openai-chatgpt",
                     "google", "deepseek", "openrouter", "opencode", "opencode-go"):
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


@dataclass(slots=True)
class Pending:
    """A sign-in that has been started and is waiting on the person.

    Splitting the flow in two is what makes it usable on a headless box. The
    caller can put the url - or the device code - on screen the instant it
    exists, then block on `finish` off the UI thread. Doing both in one call
    meant the url was computed, handed to `announce`, and thrown away, so an
    ssh session showed "opening your browser" and then nothing at all for five
    minutes.
    """

    provider: str
    url: str
    opened: bool
    user_code: str | None = None
    _entry: Any = None
    _pkce: Any = None
    _server: Any = None
    _redirect_uri: str = ""
    _device: Any = None

    @property
    def port(self) -> int:
        """The loopback port the provider will redirect to, 0 for a device flow.

        Worth surfacing: over SSH the redirect lands on this machine's port, so
        the person needs it to forward one.
        """
        return getattr(self._server, "port", 0) or 0

    @property
    def kind(self) -> str:
        return "device" if self._device is not None else "loopback"

    def finish(self, *, timeout: float = 300.0) -> Credential:
        """Block until the person finishes, then store the credential."""
        if self._device is not None:
            token = oauth.device_wait(self._entry, self._device)
        else:
            try:
                code = self._server.wait(timeout=timeout)
            finally:
                self._server.close()
            if not code:
                raise AuthError("the browser never came back with an authorisation code")
            token = oauth.exchange(self._entry, code, self._pkce, self._redirect_uri)
        cred = Credential.from_token(self.provider, token)
        save(cred)
        return cred


def begin_login(provider: str, *, open_browser: bool = True) -> Pending:
    """Start a sign-in and return as soon as there is something to show.

    Prefers the device flow when no browser could be opened and the provider
    offers one - the normal case over SSH, where a loopback redirect lands on
    the wrong machine.
    """
    entry = oauth.app(provider)
    absent = oauth.needs(provider)
    if absent:
        raise AuthError(f"{provider} needs {', '.join(absent)} configured first")

    pkce = oauth.Pkce.create()
    state = oauth.new_state()
    server = oauth.start_loopback(state if oauth.sends_state(entry) else None,
                                  host=entry.redirect_host, path=entry.redirect_path,
                                  port=entry.redirect_port)
    redirect_uri = f"http://{entry.redirect_host}:{server.port}{entry.redirect_path}"
    url = oauth.authorize_url(entry, pkce, state, redirect_uri)
    opened = oauth.launch(url) if open_browser else False

    if not opened and entry.device_url:
        # No browser here, and the provider has a flow that does not need one.
        server.close()
        device = oauth.device_start(entry)
        return Pending(provider, device.verification_uri, False, device.user_code,
                       _entry=entry, _device=device)
    return Pending(provider, url, opened, None,
                   _entry=entry, _pkce=pkce, _server=server, _redirect_uri=redirect_uri)


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
    pending = begin_login(provider)
    announce(LoginProgress(
        url=pending.url,
        user_code=pending.user_code,
        message="opened your browser; finish there" if pending.opened
        else "enter this code in your browser" if pending.user_code
        else "open this url to finish signing in",
    ))
    cred = pending.finish(timeout=timeout)
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
