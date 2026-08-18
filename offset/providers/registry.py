"""Model catalogue, provider construction, and credential lookup.

The catalogue is a convenience, never a gate: `resolve()` accepts any model id
and falls back to the provider implied by its prefix, so a model released
tomorrow works today without editing this file.

Credentials come from the environment first, then `~/.offset/credentials.json`
(created 0600).  Keys are never logged, never echoed, and never written into a
session file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from importlib import import_module
from typing import Any, Callable, Final

from collections.abc import Iterator, MutableMapping

from offset.core import settings
from offset.providers.base import Provider

def config_dir() -> Path:
    """Where offset keeps its own state.

    A function, not a constant: resolving `OFFSET_HOME` at import time meant a
    process that moved it afterwards - every test that thought it was isolated -
    silently kept reading the real one. `settings.home()` is the single place
    that answers this question.
    """
    return settings.home()


def credentials_file() -> Path:
    return config_dir() / "credentials.json"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    provider: str
    label: str
    context: int = 128_000
    max_output: int = 8_192
    tools: bool = True
    thinking: bool = False
    local: bool = False
    role_hint: str = ""  # a suggestion for the multi-model scheduler


MODELS: Final[tuple[ModelInfo, ...]] = (
    ModelInfo("claude-opus-4-20250514", "anthropic", "claude opus 4", 200_000, 32_000, thinking=True, role_hint="planner"),
    ModelInfo("claude-sonnet-4-20250514", "anthropic", "claude sonnet 4", 200_000, 64_000, thinking=True, role_hint="implementer"),
    ModelInfo("claude-3-5-haiku-20241022", "anthropic", "claude haiku 3.5", 200_000, 8_192, role_hint="cheap"),
    ModelInfo("gpt-4.1", "openai", "gpt-4.1", 1_000_000, 32_768, role_hint="implementer"),
    ModelInfo("gpt-4o", "openai", "gpt-4o", 128_000, 16_384, role_hint="implementer"),
    ModelInfo("gpt-4o-mini", "openai", "gpt-4o mini", 128_000, 16_384, role_hint="cheap"),
    ModelInfo("o3", "openai", "o3", 200_000, 100_000, thinking=True, role_hint="critic"),
    ModelInfo("o4-mini", "openai", "o4 mini", 200_000, 100_000, thinking=True, role_hint="critic"),
    ModelInfo("gemini-2.5-pro", "google", "gemini 2.5 pro", 1_048_576, 65_536, thinking=True, role_hint="critic"),
    ModelInfo("gemini-2.5-flash", "google", "gemini 2.5 flash", 1_048_576, 65_536, thinking=True, role_hint="cheap"),
    ModelInfo("deepseek-chat", "deepseek", "deepseek v3", 64_000, 8_192, role_hint="implementer"),
    ModelInfo("deepseek-reasoner", "deepseek", "deepseek r1", 64_000, 8_192, thinking=True, role_hint="critic"),
    ModelInfo("qwen2.5-coder:7b", "ollama", "qwen2.5 coder 7b", 32_768, 4_096, local=True, role_hint="bulk"),
    ModelInfo("llama3.2:3b", "ollama", "llama 3.2 3b", 131_072, 4_096, local=True, role_hint="cheap"),
    # OpenCode Zen - pay as you go. Only models offset can actually speak to
    # are listed; GPT and Grok there need the Responses API.
    ModelInfo("opencode/claude-opus-5", "opencode", "zen: claude opus 5", 200_000, 32_000, thinking=True, role_hint="planner"),
    ModelInfo("opencode/claude-sonnet-5", "opencode", "zen: claude sonnet 5", 200_000, 64_000, thinking=True, role_hint="implementer"),
    ModelInfo("opencode/claude-haiku-4-5", "opencode", "zen: claude haiku 4.5", 200_000, 8_192, role_hint="cheap"),
    ModelInfo("opencode/qwen3.7-max", "opencode", "zen: qwen3.7 max", 256_000, 32_000, role_hint="implementer"),
    ModelInfo("opencode/glm-5.2", "opencode", "zen: glm 5.2", 200_000, 32_000, role_hint="implementer"),
    ModelInfo("opencode/kimi-k3", "opencode", "zen: kimi k3", 256_000, 32_000, role_hint="planner"),
    ModelInfo("opencode/deepseek-v4-pro", "opencode", "zen: deepseek v4 pro", 128_000, 16_384, role_hint="critic"),
    ModelInfo("opencode/minimax-m3", "opencode", "zen: minimax m3", 200_000, 16_384, role_hint="bulk"),
    ModelInfo("opencode/deepseek-v4-flash-free", "opencode", "zen: deepseek v4 flash (free)", 128_000, 16_384, role_hint="cheap"),
    ModelInfo("opencode/big-pickle", "opencode", "zen: big pickle (free)", 128_000, 16_384, role_hint="cheap"),
    # OpenCode Go - $10/month subscription for open models.
    ModelInfo("opencode-go/kimi-k3", "opencode-go", "go: kimi k3", 256_000, 32_000, role_hint="planner"),
    ModelInfo("opencode-go/kimi-k2.7-code", "opencode-go", "go: kimi k2.7 code", 256_000, 32_000, role_hint="implementer"),
    ModelInfo("opencode-go/glm-5.3", "opencode-go", "go: glm 5.3", 200_000, 32_000, role_hint="implementer"),
    ModelInfo("opencode-go/qwen3.8-max", "opencode-go", "go: qwen3.8 max", 256_000, 32_000, role_hint="planner"),
    ModelInfo("opencode-go/minimax-m3", "opencode-go", "go: minimax m3", 200_000, 16_384, role_hint="bulk"),
    ModelInfo("opencode-go/deepseek-v4-pro", "opencode-go", "go: deepseek v4 pro", 128_000, 16_384, role_hint="critic"),
    ModelInfo("opencode-go/deepseek-v4-flash", "opencode-go", "go: deepseek v4 flash", 128_000, 16_384, role_hint="cheap"),
    ModelInfo("opencode-go/mimo-v2.5", "opencode-go", "go: mimo v2.5", 128_000, 16_384, role_hint="cheap"),
    ModelInfo("mock", "mock", "scripted mock", 8_192, 4_096, local=True, role_hint="test"),
)

BY_ID: Final[dict[str, ModelInfo]] = {m.id: m for m in MODELS}

#: Provider name to the module and attribute that builds it.
#:
#: Deliberately not imported at module level. Importing any one of these pulls in
#: `transport`, and with it `urllib.request` and `http.client` - a third of a
#: second on a Raspberry Pi, paid by every command including `--help`, to reach a
#: network nobody has asked to use yet. Each entry loads the first time that
#: provider is actually built.
_FACTORIES: Final[dict[str, tuple[str, str]]] = {
    "anthropic": ("anthropic", "Anthropic"),
    "openai": ("openai", "OpenAI"),
    "google": ("google", "Google"),
    "deepseek": ("openai", "deepseek"),
    "openrouter": ("openai", "openrouter"),
    "llamacpp": ("openai", "llamacpp"),
    "ollama": ("ollama", "Ollama"),
    "opencode": ("opencode", "OpenCodeZen"),
    "opencode-go": ("opencode", "OpenCodeGo"),
    "mock": ("mock", "Mock"),
}


#: Factories registered at runtime, which win over the lazy table above.
_OVERRIDES: dict[str, Callable[[], Provider]] = {}


def factory_for(name: str) -> Callable[[], Provider]:
    """The callable that builds provider `name`, imported on demand."""
    override = _OVERRIDES.get(name)
    if override is not None:
        return override
    entry = _FACTORIES.get(name)
    if entry is None:
        raise KeyError(f"unknown provider: {name}")
    module_name, attribute = entry
    module = import_module(f"offset.providers.{module_name}")
    return getattr(module, attribute)


class _Providers(MutableMapping[str, Callable[[], Provider]]):
    """The provider table: lazily imported, still an ordinary mapping.

    Reading a name costs nothing - `/models`, the login targets and the picker
    only ever want names, and they used to pay for the whole http stack to get
    them. Assignment stays supported because it is a real registration point: a
    test substitutes a scripted provider that way, and so could an extension.
    """

    __slots__ = ()

    def __getitem__(self, name: str) -> Callable[[], Provider]:
        return factory_for(name)

    def __setitem__(self, name: str, factory: Callable[[], Provider]) -> None:
        _OVERRIDES[name] = factory

    def __delitem__(self, name: str) -> None:
        if name in _OVERRIDES:
            del _OVERRIDES[name]
        elif name in _FACTORIES:
            raise KeyError(f"{name} is built in; override it instead of deleting it")
        else:
            raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        seen = dict.fromkeys(_FACTORIES)
        seen.update(dict.fromkeys(_OVERRIDES))
        return iter(seen)

    def __len__(self) -> int:
        return len(set(_FACTORIES) | set(_OVERRIDES))


PROVIDERS: Final[MutableMapping[str, Callable[[], Provider]]] = _Providers()

#: Used when a model id is not in the catalogue.  Prefix wins over guessing.
PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini", "google"),
    ("deepseek", "deepseek"),
    ("opencode-go/", "opencode-go"),
    ("opencode/", "opencode"),
    ("mock", "mock"),
)


def info(model_id: str) -> ModelInfo:
    """Catalogue entry, or a synthesised one for an unknown id."""
    known = BY_ID.get(model_id)
    if known:
        return known
    lowered = model_id.lower()
    provider = next((p for prefix, p in PREFIXES if lowered.startswith(prefix)), "ollama")
    return ModelInfo(model_id, provider, model_id, local=provider == "ollama")


def search(query: str) -> list[ModelInfo]:
    """Substring match over id and label, for the picker."""
    q = query.strip().lower()
    if not q:
        return list(MODELS)
    return [m for m in MODELS if q in m.id.lower() or q in m.label.lower()]


def provider_for(name: str) -> Provider:
    return factory_for(name)()


def resolve(model_id: str) -> tuple[Provider, ModelInfo]:
    """Everything needed to run a model: its transport and its limits."""
    meta = info(model_id)
    return provider_for(meta.provider), meta


# -- credentials ------------------------------------------------------------


#: Last parse of the credentials file, keyed by what would make it stale.
_CACHE: dict[str, Any] = {"stamp": None, "data": {}}


def _stored() -> dict[str, str]:
    """Plain API keys from disk, parsed at most once per change.

    `available()` asks about every model in the catalogue and each question used
    to re-read and re-parse this file: about thirty reads of the same bytes to
    build one roster, on every startup and every `/models`. Keying the cache on
    the file's identity rather than a timestamp of our own keeps a key written by
    another process - a second offset, or a text editor - visible immediately.
    """
    try:
        stat = credentials_file().stat()
        stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except OSError:
        _CACHE["stamp"], _CACHE["data"] = None, {}
        return {}
    if _CACHE["stamp"] == stamp:
        return _CACHE["data"]
    try:
        raw = json.loads(credentials_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _CACHE["stamp"], _CACHE["data"] = None, {}
        return {}
    # Richer entries (OAuth tokens) are objects; `offset.providers.auth` owns
    # those. Here we only surface plain API-key strings.
    data = {str(k): v for k, v in raw.items() if isinstance(v, str)} if isinstance(raw, dict) else {}
    _CACHE["stamp"], _CACHE["data"] = stamp, data
    return data


def credential(provider: Provider | str) -> str | None:
    """Environment first, then the on-disk store.  Never raises."""
    name = provider if isinstance(provider, str) else provider.name
    keys = () if isinstance(provider, str) else provider.env_keys
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    for key in (f"{name.upper()}_API_KEY", f"OFFSET_{name.upper()}_KEY"):
        value = os.environ.get(key)
        if value:
            return value
    return _stored().get(name)


def store_credential(provider: str, key: str) -> Path:
    """Persist a key with owner-only permissions."""
    config_dir().mkdir(parents=True, exist_ok=True)
    data = _stored()
    data[provider] = key
    tmp = credentials_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, credentials_file())
    os.chmod(credentials_file(), 0o600)
    return credentials_file()


def reachable(model_id: str) -> bool:
    """Whether this machine could run `model_id` right now.

    Stronger than `available`, which only asks whether a key exists: a local
    model needs a server that answers, and an ollama entry in the catalogue
    proves nothing about whether ollama is running. Used to pick the model a new
    session starts on, and to seat a roster - both of which used to choose models
    that could only fail.
    """
    meta = info(model_id)
    if meta.provider == "mock":
        return True  # scripted: no network, no key, always answers
    if meta.local:
        return listening(meta)
    return bool(credential(provider_for(meta.provider)))


def listening(meta: ModelInfo, timeout: float = 0.15) -> bool:
    """Whether a local server accepts connections, without waiting on it."""
    import socket
    from urllib.parse import urlsplit

    url = urlsplit(getattr(meta, "base_url", "") or "http://127.0.0.1:11434")
    try:
        with socket.create_connection((url.hostname or "127.0.0.1", url.port or 11434), timeout):
            return True
    except OSError:
        return False


def available() -> list[ModelInfo]:
    """Models we could actually run right now: local, or key present."""
    out: list[ModelInfo] = []
    for m in MODELS:
        if m.local or credential(provider_for(m.provider)):
            out.append(m)
    return out


def redact(text: str, *keys: str | None) -> str:
    """Scrub secrets out of anything about to be displayed or logged."""
    for key in keys:
        if key and len(key) > 8:
            text = text.replace(key, key[:3] + "\u2026" + key[-2:])
    return text


