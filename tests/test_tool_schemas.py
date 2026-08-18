"""Every shipped tool schema must survive every provider.

This is the test that was missing when a single malformed schema made *every*
message fail. Providers send all tools on every request, so one bad schema is
not a broken tool - it is a broken program, and the failure looks like the model
being unreachable rather than like a tool being wrong.

MCP makes it worse: those schemas come from servers nobody here controls, so the
normaliser has to hold rather than the authors being careful.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from offset.providers import anthropic, google, ollama, openai
from offset.providers.base import Message, Request, ToolSpec
from offset.providers.schema import normalise
from offset.tools.base import Toolbox
from offset.tools.builtin import builtin_tools
from offset.tools.system import system_tools
from offset.tools.todo import todo_tools
from offset.tools.websearch import web_search_tools


def every_tool() -> list[ToolSpec]:
    box = Toolbox([*builtin_tools(), *system_tools(), *web_search_tools()])
    for tool in todo_tools(Path(tempfile.mkdtemp())):
        box.register(tool)
    return box.specs()


def ask(tools) -> Request:
    return Request(model="m", messages=[Message("user", "hi")], tools=tools, max_tokens=64)


# -- the rule each provider actually enforces -------------------------------


def strict_problems(schema: dict, *, dialect: str) -> list[str]:
    """What each dialect actually refuses - not one blanket rule.

    Gemini's function declarations follow an OpenAPI 3.0 subset: ARRAY needs
    `items`, and every property needs a `type`. Anthropic wants a `type` on each
    property. OpenAI and Ollama take ordinary JSON Schema and are relaxed about
    both, so asserting the strictest rule everywhere would be inventing a
    constraint rather than testing one.
    """
    needs_type = dialect in ("anthropic", "google")
    needs_items = dialect == "google"

    issues: list[str] = []
    if schema.get("type") != "object":
        issues.append("top level must be an object")
    props = schema.get("properties")
    if not isinstance(props, dict):
        return [*issues, "missing properties"]
    for name, prop in props.items():
        if not isinstance(prop, dict):
            issues.append(f"{name}: not a schema")
            continue
        if needs_type and "type" not in prop and "enum" not in prop:
            issues.append(f"{name}: no type")
        if needs_items and prop.get("type") == "array" and "items" not in prop:
            issues.append(f"{name}: array without items")
        nested = prop.get("properties")
        if isinstance(nested, dict):
            issues += [f"{name}.{p}" for p in strict_problems(
                {"type": "object", "properties": nested}, dialect=dialect)]
    return issues


@pytest.mark.parametrize("dialect", ["anthropic", "openai", "google", "ollama"])
def test_every_shipped_tool_is_acceptable_after_normalising(dialect):
    for spec in every_tool():
        cleaned = normalise(spec.schema, dialect)
        problems = strict_problems(cleaned, dialect=dialect)
        assert not problems, f"{spec.name} for {dialect}: {problems}"


def test_the_schemas_are_already_clean_at_source():
    """The normaliser is a safety net, not a licence to write bad schemas.

    Both faults that broke every message - a bare anyOf and an array with no
    items - would be caught here.
    """
    for dialect in ("anthropic", "google"):
        for spec in every_tool():
            problems = strict_problems(spec.schema, dialect=dialect)
            assert not problems, f"{spec.name} ships a schema {dialect} rejects: {problems}"


# -- the two that actually broke it -----------------------------------------


def test_a_bare_anyof_never_reaches_anthropic():
    """`document.content` was `{"anyOf": [...]}` with no type, which Anthropic
    rejects, so every single message failed with a 400."""
    loose = {"type": "object", "properties": {"content": {"anyOf": [{"type": "string"}, {"type": "array"}]}}}
    content = normalise(loose, "anthropic")["properties"]["content"]
    assert content["type"] == "string", "a union must be given a concrete type"
    assert not strict_problems(normalise(loose, "anthropic"), dialect="anthropic")


def test_an_array_without_items_is_repaired():
    """`todo.tasks` was `{"type": "array"}`; Google rejects that."""
    loose = {"type": "object", "properties": {"tasks": {"type": "array"}}}
    cleaned = normalise(loose, "google")
    assert "items" in cleaned["properties"]["tasks"]


def test_normalising_never_mutates_the_input():
    original = {"type": "object", "properties": {"x": {"type": "array"}}}
    snapshot = json.dumps(original, sort_keys=True)
    normalise(original, "google")
    assert json.dumps(original, sort_keys=True) == snapshot


def test_normalising_is_idempotent():
    for dialect in ("anthropic", "openai", "google", "ollama"):
        for spec in every_tool():
            once = normalise(spec.schema, dialect)
            assert normalise(once, dialect) == once, f"{spec.name}/{dialect}"


# -- it is wired in, not merely present -------------------------------------


def test_anthropic_sends_normalised_schemas():
    loose = ToolSpec("loose", "d", {"type": "object", "properties": {"a": {"anyOf": [{"type": "string"}]}}})
    payload = anthropic.build_payload(ask([loose]))
    sent = payload["tools"][0]["input_schema"]
    assert not strict_problems(sent, dialect="anthropic"), sent


def test_openai_sends_normalised_schemas():
    """OpenAI takes plain JSON Schema, so the payload should be passed through
    with only the validator bookkeeping removed."""
    loose = ToolSpec("loose", "d", {"$schema": "https://json-schema.org/draft/2020-12/schema",
                                    "type": "object", "properties": {"a": {"type": "array"}}})
    sent = openai.build_payload(ask([loose]))["tools"][0]["function"]["parameters"]
    assert "$schema" not in sent
    assert not strict_problems(sent, dialect="openai")


def test_google_sends_normalised_schemas():
    loose = ToolSpec("loose", "d", {"type": "object", "properties": {"a": {"type": "array"}}})
    sent = google.build_payload(ask([loose]))["tools"][0]["functionDeclarations"][0]["parameters"]
    assert "items" in sent["properties"]["a"]


def test_ollama_sends_normalised_schemas():
    loose = ToolSpec("loose", "d", {"$schema": "x", "type": "object",
                                    "properties": {"a": {"type": "array"}}})
    sent = ollama.build_payload(ask([loose]))["tools"][0]["function"]["parameters"]
    assert "$schema" not in sent and not strict_problems(sent, dialect="ollama")


def test_a_hostile_remote_schema_cannot_break_a_request():
    """An MCP server's schema is written by someone else entirely."""
    hostile = ToolSpec("mcp__evil__thing", "from a server we do not control", {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "loose": {"type": "array"},
            "bare": {"anyOf": [{"type": "string"}, {"type": "number"}]},
            "nested": {"type": "object", "properties": {"deep": {"type": "array"}}},
        },
        "additionalProperties": True,
    })
    for dialect, payload in (
        ("anthropic", anthropic.build_payload(ask([hostile]))["tools"][0]["input_schema"]),
        ("openai", openai.build_payload(ask([hostile]))["tools"][0]["function"]["parameters"]),
        ("google", google.build_payload(ask([hostile]))["tools"][0]["functionDeclarations"][0]["parameters"]),
    ):
        assert not strict_problems(payload, dialect=dialect), f"{dialect}: {payload}"


# -- the document tool still takes rich content -----------------------------


def test_a_json_block_array_still_works_as_a_string():
    """The schema had to become a plain string, so the rich form arrives as
    JSON text and must still be understood."""
    from offset.tools.documents import blocks_from

    blocks = blocks_from(json.dumps([
        {"kind": "heading", "text": "Report"},
        {"kind": "bullets", "items": ["one", "two"]},
    ]))
    assert [b.kind for b in blocks] == ["heading", "bullets"]
    assert blocks[1].items == ("one", "two")


def test_plain_text_content_still_becomes_paragraphs():
    from offset.tools.documents import blocks_from

    assert [b.kind for b in blocks_from("first\n\nsecond")] == ["paragraph", "paragraph"]


def test_malformed_json_content_is_treated_as_text():
    from offset.tools.documents import blocks_from

    blocks = blocks_from('[{"kind": "heading", broken')
    assert blocks and blocks[0].kind == "paragraph"
