"""Server-sent events and newline-delimited JSON.

Both framings show up across providers: Anthropic and OpenAI speak SSE, Ollama
and Google's REST streaming speak NDJSON.  Neither parser ever raises on
malformed input — a mangled frame is skipped, because one bad chunk must not
kill a turn that is otherwise streaming fine.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Iterator


def iter_sse(lines: Iterable[bytes]) -> Iterator[tuple[str | None, str]]:
    """Yield `(event, data)` pairs.  Multi-line `data:` fields are joined."""
    event: str | None = None
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        if not line.strip():
            if data:
                yield event, "\n".join(data)
            event, data = None, []
            continue
        if line.startswith(":"):
            continue  # comment / keepalive
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
    if data:
        yield event, "\n".join(data)


def iter_json_frames(lines: Iterable[bytes]) -> Iterator[dict[str, Any]]:
    """Yield objects from NDJSON, skipping anything unparseable."""
    for raw in lines:
        line = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw).strip()
        if not line or line in ("[", "]"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        line = line.rstrip(",")
        if not line or line == "[DONE]":
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def loads(data: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
