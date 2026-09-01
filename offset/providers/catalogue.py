"""What models exist right now, asked of the providers rather than remembered.

The static table in `registry.MODELS` was always going to go stale, and it did:
against a real key this machine found thirty-eight Google models live and four
in the table. A curated list is still worth having - it carries the context
windows and role hints no API reports - but it cannot be the only source,
because the day a provider ships something is the day somebody wants to use it.

So both. The table is the floor; each provider's own listing is merged over it.
Three rules keep that honest:

**Never block.** Listings are cached and refreshed on a background thread. A
cold cache shows the table, which is exactly what shipped before this file
existed, so the worst case is yesterday's behaviour rather than a slow start.

**Never ask without a credential.** These endpoints are authenticated. Calling
them unauthenticated earns a 401 per launch and tells the user nothing.

**Never invent.** Where the API reports a context window that wins. Where it
does not - and most do not - the number is inferred from the family, and the
inference is a documented table rather than a guess dressed up as a fact.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterable

from offset.core import settings
from offset.providers.registry import MODELS, ModelInfo, credential

#: How long a listing stays fresh: short enough to notice a launch the same
#: day, long enough that a busy shell is not re-asking every few minutes.
TTL: Final = 6 * 3600.0

#: A failure is retried sooner than a success. The usual cause is a laptop that
#: was briefly off the network, not a provider that stopped having models.
RETRY_TTL: Final = 900.0

TIMEOUT: Final = 8.0

CACHE_VERSION: Final = 1

#: Suppresses every live listing, for an air-gapped machine or a test.
NO_FETCH_ENV: Final = "OFFSET_NO_MODEL_FETCH"

#: Fetches a URL and returns parsed JSON. Injected so tests never touch a
#: network and so a caller can supply its own transport.
Fetcher = Callable[[str, dict[str, str]], Any]


def _get(url: str, headers: dict[str, str]) -> Any:
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def _post(url: str, headers: dict[str, str], body: dict[str, Any]) -> Any:
    payload = json.dumps(body).encode("utf-8")
    sent = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(url, data=payload, headers=sent, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


# -- inferring what the API does not say -------------------------------------

#: Context windows by family, longest prefix first, so a specific entry beats a
#: general one. Every number here is the provider's published figure.
CONTEXT: Final[tuple[tuple[str, int, int], ...]] = (
    ("gemini-3", 1_048_576, 65_536),
    ("gemini-2.5", 1_048_576, 65_536),
    ("gemini", 1_048_576, 8_192),
    ("claude-opus-4", 200_000, 32_000),
    ("claude-sonnet-4", 200_000, 64_000),
    ("claude-haiku-4", 200_000, 8_192),
    ("claude-3-7", 200_000, 64_000),
    ("claude-3-5", 200_000, 8_192),
    ("claude", 200_000, 32_000),
    ("gpt-5", 400_000, 128_000),
    ("gpt-4.1", 1_000_000, 32_768),
    ("gpt-4o", 128_000, 16_384),
    ("gpt-4", 128_000, 8_192),
    ("o4", 200_000, 100_000),
    ("o3", 200_000, 100_000),
    ("o1", 200_000, 100_000),
    ("deepseek", 64_000, 8_192),
    ("qwen", 256_000, 32_000),
    ("kimi", 256_000, 32_000),
    ("glm", 200_000, 32_000),
)

#: Reasons before answering. `o1`/`o3`/`o4` are reasoning models by definition;
#: the rest say so in the name.
THINKS: Final = re.compile(
    r"(^|/)o[134](-|$)|thinking|reason|gemini-3|gemini-2\.5|"
    r"claude-(opus|sonnet)-[45]|gpt-5"
)

#: Not a chat model. Listing endpoints return everything a provider hosts -
#: embeddings, speech, image, moderation - and offering those as coding models
#: only wastes the time it takes a user to discover they do not work.
NOT_CHAT: Final = re.compile(
    r"embed|tts|whisper|audio|speech|moderation|image|rerank|"
    r"dall-e|sora|veo|imagen|guard|safety|computer-use"
)

CHEAP: Final = re.compile(r"mini|flash|lite|small|haiku|tiny|nano|8b|7b|3b|free")
BIG: Final = re.compile(r"opus|pro|max|ultra|405b|70b")


def _shape(model_id: str) -> tuple[int, int]:
    bare = model_id.split("/")[-1].lower()
    for prefix, context, output in CONTEXT:
        if bare.startswith(prefix):
            return context, output
    return 128_000, 8_192  # a defensible floor, not precision we do not have


def _role(model_id: str) -> str:
    bare = model_id.split("/")[-1].lower()
    thinks, big = bool(THINKS.search(bare)), bool(BIG.search(bare))
    if thinks and big:
        return "planner"
    if thinks:
        return "critic"
    if CHEAP.search(bare):
        return "cheap"
    return "implementer"


def usable_as_chat(model_id: str) -> bool:
    """Whether this id is worth offering as a coding model."""
    return not NOT_CHAT.search(model_id.lower())


def describe(model_id: str, provider: str, *, label: str = "",
             context: int = 0, output: int = 0) -> ModelInfo:
    """A `ModelInfo` for a model the static table has never heard of."""
    inferred_context, inferred_output = _shape(model_id)
    bare = model_id.split("/")[-1]
    return ModelInfo(
        id=model_id,
        provider=provider,
        label=label or bare.replace("-", " "),
        context=context or inferred_context,
        max_output=output or inferred_output,
        tools=True,
        thinking=bool(THINKS.search(bare.lower())),
        local=provider in ("ollama", "llamacpp"),
        role_hint=_role(model_id),
    )


# -- per-provider listings ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class Listing:
    """One provider's live models, and whether the attempt worked."""

    provider: str
    models: tuple[ModelInfo, ...] = ()
    error: str = ""
    fetched: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error


def _openai_like(provider: str, base: str, key: str, fetch: Fetcher) -> list[ModelInfo]:
    """`GET /models` returning `{"data": [{"id": ...}]}`.

    OpenAI's shape, and therefore also DeepSeek's and most gateways'.
    """
    payload = fetch(f"{base.rstrip('/')}/models", {"Authorization": f"Bearer {key}"})
    out = []
    for entry in (payload or {}).get("data", []):
        model_id = str(entry.get("id") or "")
        if model_id and usable_as_chat(model_id):
            out.append(describe(model_id, provider))
    return out


def _anthropic(provider: str, base: str, key: str, fetch: Fetcher) -> list[ModelInfo]:
    payload = fetch(
        f"{base.rstrip('/')}/models?limit=1000",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    out = []
    for entry in (payload or {}).get("data", []):
        model_id = str(entry.get("id") or "")
        if model_id and usable_as_chat(model_id):
            out.append(describe(model_id, provider, label=str(entry.get("display_name") or "")))
    return out


def _google(provider: str, base: str, key: str, fetch: Fetcher) -> list[ModelInfo]:
    """Google reports what each model supports, so this filter is a fact."""
    payload = fetch(f"{base.rstrip('/')}/models?key={key}&pageSize=1000", {})
    out = []
    for entry in (payload or {}).get("models", []):
        name = str(entry.get("name") or "").removeprefix("models/")
        methods = entry.get("supportedGenerationMethods") or []
        if not name or "generateContent" not in methods or not usable_as_chat(name):
            continue
        out.append(describe(
            name, provider,
            label=str(entry.get("displayName") or ""),
            context=int(entry.get("inputTokenLimit") or 0),
            output=int(entry.get("outputTokenLimit") or 0),
        ))
    return out


def _antigravity(provider: str, base: str, key: str, fetch: Fetcher) -> list[ModelInfo]:
    """Whatever the signed-in Antigravity account may currently call.

    Two round trips, because the model list is scoped to the account's Cloud
    project and that has to be asked for first.  Ids are prefixed so they never
    collide with the same model reached through a plain Gemini key - the two go
    to different backends and bill against different things.
    """
    from offset.providers.google import ANTIGRAVITY_PREFIX, available_models, load_code_assist

    info = load_code_assist(key, timeout=TIMEOUT)
    project = str(info.get("cloudaicompanionProject") or "")
    out = []
    for model_id, label in available_models(key, project, timeout=TIMEOUT):
        if not usable_as_chat(model_id):
            continue
        out.append(describe(f"{ANTIGRAVITY_PREFIX}{model_id}", provider,
                            label=f"antigravity: {label or model_id}"))
    return out


def _ollama(provider: str, base: str, key: str, fetch: Fetcher) -> list[ModelInfo]:
    """Whatever has actually been pulled onto this machine."""
    payload = fetch(f"{base.rstrip('/')}/api/tags", {})
    out = []
    for entry in (payload or {}).get("models", []):
        name = str(entry.get("name") or "")
        if name:
            out.append(describe(name, provider))
    return out


def _openrouter(provider: str, base: str, key: str, fetch: Fetcher) -> list[ModelInfo]:
    """The one listing needing no credential, so it always works."""
    payload = fetch(f"{base.rstrip('/')}/models", {})
    out = []
    for entry in (payload or {}).get("data", []):
        model_id = str(entry.get("id") or "")
        if not model_id or not usable_as_chat(model_id):
            continue
        top = entry.get("top_provider") or {}
        out.append(describe(
            f"openrouter/{model_id}", provider,
            label=str(entry.get("name") or ""),
            context=int(entry.get("context_length") or 0),
            output=int(top.get("max_completion_tokens") or 0),
        ))
    return out


@dataclass(frozen=True, slots=True)
class Source:
    """How to ask one provider what it has."""

    base: str
    call: Callable[[str, str, str, Fetcher], list[ModelInfo]]
    #: Whether a credential is required before it is worth asking at all.
    needs_key: bool = True


SOURCES: Final[dict[str, Source]] = {
    "openai": Source("https://api.openai.com/v1", _openai_like),
    "anthropic": Source("https://api.anthropic.com/v1", _anthropic),
    "google": Source("https://generativelanguage.googleapis.com/v1beta", _google),
    "google-antigravity": Source("https://cloudcode-pa.googleapis.com", _antigravity),
    "deepseek": Source("https://api.deepseek.com/v1", _openai_like),
    "openrouter": Source("https://openrouter.ai/api/v1", _openrouter, needs_key=False),
    "ollama": Source("http://127.0.0.1:11434", _ollama, needs_key=False),
}


def enabled() -> bool:
    return (os.environ.get(NO_FETCH_ENV) or "").strip().lower() not in ("1", "true", "yes", "on")


def fetch_provider(provider: str, *, fetch: Fetcher | None = None) -> Listing:
    """Ask one provider what it has. Never raises."""
    source = SOURCES.get(provider)
    if source is None:
        return Listing(provider, error=f"no listing endpoint is known for {provider}")

    key = credential(provider) or ""
    if source.needs_key and not key:
        return Listing(provider, error="no credential, so the listing was not requested")

    try:
        found = source.call(provider, source.base, key, fetch or _get)
    except urllib.error.HTTPError as exc:
        return Listing(provider, error=f"the provider answered {exc.code}", fetched=time.time())
    except Exception as exc:
        return Listing(provider, error=f"{type(exc).__name__}: {exc}", fetched=time.time())
    return Listing(provider, tuple(found), fetched=time.time())


# -- the cache ---------------------------------------------------------------


def cache_file() -> Path:
    """Resolved late: `OFFSET_HOME` moves under tests and `--home`."""
    return settings.home() / "models.json"


def _read_cache() -> dict[str, Any]:
    try:
        raw = json.loads(cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
        return {}
    return raw


def _write_cache(body: dict[str, Any]) -> None:
    path = cache_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".models.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _to_json(model: ModelInfo) -> dict[str, Any]:
    return {
        "id": model.id, "provider": model.provider, "label": model.label,
        "context": model.context, "max_output": model.max_output,
        "tools": model.tools, "thinking": model.thinking,
        "local": model.local, "role_hint": model.role_hint,
    }


def _from_json(raw: Any) -> ModelInfo | None:
    if not isinstance(raw, dict) or not raw.get("id"):
        return None
    try:
        return ModelInfo(
            id=str(raw["id"]), provider=str(raw.get("provider") or ""),
            label=str(raw.get("label") or ""), context=int(raw.get("context") or 0),
            max_output=int(raw.get("max_output") or 0), tools=bool(raw.get("tools", True)),
            thinking=bool(raw.get("thinking")), local=bool(raw.get("local")),
            role_hint=str(raw.get("role_hint") or ""),
        )
    except (TypeError, ValueError):
        return None


def cached(provider: str) -> Listing | None:
    """What we last heard from a provider, fresh or not."""
    entry = _read_cache().get("providers", {}).get(provider)
    if not isinstance(entry, dict):
        return None
    models = [m for m in (_from_json(x) for x in entry.get("models", [])) if m is not None]
    return Listing(provider, tuple(models), str(entry.get("error") or ""),
                   float(entry.get("fetched") or 0.0))


def stale(listing: Listing | None, *, now: float | None = None) -> bool:
    if listing is None:
        return True
    age = (now if now is not None else time.time()) - listing.fetched
    return age > (RETRY_TTL if listing.error else TTL)


def store(listing: Listing) -> None:
    body = _read_cache() or {}
    body["version"] = CACHE_VERSION
    body.setdefault("providers", {})[listing.provider] = {
        "models": [_to_json(m) for m in listing.models],
        "error": listing.error,
        "fetched": listing.fetched,
    }
    _write_cache(body)


# -- the merged view ---------------------------------------------------------


def merged(*, include_live: bool = True) -> list[ModelInfo]:
    """The static table with every cached live listing merged over it.

    A live entry adds itself, but where the table already knows a model the
    table's `role_hint` and context win: those are the curated part a listing
    endpoint cannot tell us, and losing them would silently downgrade the
    multi-model roster.
    """
    out: dict[str, ModelInfo] = {m.id: m for m in MODELS}

    if include_live:
        for provider in SOURCES:
            listing = cached(provider)
            if listing is None:
                continue
            for model in listing.models:
                known = out.get(model.id)
                if known is None:
                    out[model.id] = model
                    continue
                out[model.id] = ModelInfo(
                    id=known.id, provider=known.provider,
                    label=known.label or model.label,
                    context=known.context or model.context,
                    max_output=known.max_output or model.max_output,
                    tools=known.tools,
                    thinking=known.thinking or model.thinking,
                    local=known.local,
                    role_hint=known.role_hint or model.role_hint,
                )
    return sorted(out.values(), key=lambda m: (m.provider, m.id))


def refresh(providers: Iterable[str] | None = None, *,
            fetch: Fetcher | None = None, force: bool = False) -> list[Listing]:
    """Re-ask every provider whose cache has gone stale. Never raises."""
    if not enabled():
        return []
    done: list[Listing] = []
    for provider in (providers if providers is not None else list(SOURCES)):
        if provider not in SOURCES:
            continue
        if not force and not stale(cached(provider)):
            continue
        listing = fetch_provider(provider, fetch=fetch)
        # A failed refresh must not erase what we already knew: a provider that
        # is briefly unreachable should not empty the picker.
        if listing.error:
            previous = cached(provider)
            if previous is not None and previous.models:
                listing = Listing(provider, previous.models, listing.error, time.time())
        store(listing)
        done.append(listing)
    return done


def refresh_async(*, fetch: Fetcher | None = None,
                  done: threading.Event | None = None) -> None:
    """Refresh on a background thread. Startup waits for nothing."""
    if not enabled():
        if done is not None:
            done.set()
        return

    def work() -> None:
        try:
            refresh(fetch=fetch)
        except Exception:
            pass  # a model listing is never worth a traceback over the prompt
        finally:
            if done is not None:
                done.set()

    threading.Thread(target=work, name="offset-model-listing", daemon=True).start()


def install(state: Any) -> None:
    """Startup wiring: begin the refresh, block nothing."""
    refresh_async()


def report() -> list[str]:
    """One line per provider, for `/models --refresh` and diagnostics."""
    lines = []
    for provider in sorted(SOURCES):
        listing = cached(provider)
        if listing is None:
            lines.append(f"{provider:19s} never asked")
            continue
        age = time.time() - listing.fetched
        when = f"{age / 60:.0f}m ago" if age < 3600 else f"{age / 3600:.1f}h ago"
        if listing.error:
            lines.append(f"{provider:19s} {len(listing.models):>4} cached  ({listing.error}, {when})")
        else:
            lines.append(f"{provider:19s} {len(listing.models):>4} live    ({when})")
    return lines
