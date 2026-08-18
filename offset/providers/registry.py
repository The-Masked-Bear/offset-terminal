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
from typing import Callable, Final

from offset.providers.anthropic import Anthropic
from offset.providers.base import Provider
from offset.providers.google import Google
from offset.providers.mock import Mock
from offset.providers.ollama import Ollama
from offset.providers.opencode import OpenCodeGo, OpenCodeZen
from offset.providers.openai import OpenAI, deepseek, llamacpp, openrouter

CONFIG_DIR: Final = Path(os.environ.get("OFFSET_HOME") or (Path.home() / ".offset"))
CREDENTIALS_FILE: Final = CONFIG_DIR / "credentials.json"


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

PROVIDERS: Final[dict[str, Callable[[], Provider]]] = {
    "anthropic": Anthropic,
    "openai": OpenAI,
    "google": Google,
    "deepseek": deepseek,
    "openrouter": openrouter,
    "llamacpp": llamacpp,
    "ollama": Ollama,
    "opencode": OpenCodeZen,
    "opencode-go": OpenCodeGo,
    "mock": Mock,
}

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
    factory = PROVIDERS.get(name)
    if factory is None:
        raise KeyError(f"unknown provider: {name}")
    return factory()


def resolve(model_id: str) -> tuple[Provider, ModelInfo]:
    """Everything needed to run a model: its transport and its limits."""
    meta = info(model_id)
    return provider_for(meta.provider), meta


# -- credentials ------------------------------------------------------------


def _stored() -> dict[str, str]:
    try:
        raw = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    # Richer entries (OAuth tokens) are objects; `offset.providers.auth`
    # owns those. Here we only surface plain API-key strings.
    return {str(k): v for k, v in raw.items() if isinstance(v, str)}


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
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _stored()
    data[provider] = key
    tmp = CREDENTIALS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, CREDENTIALS_FILE)
    os.chmod(CREDENTIALS_FILE, 0o600)
    return CREDENTIALS_FILE


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


