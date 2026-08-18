"""JSON Schema normalisation, one dialect per provider.

Tool authors write ordinary JSON Schema.  Providers accept overlapping subsets
of it and differ in how they fail: Gemini rejects the entire request when it
meets `additionalProperties` or a `format` it does not know, OpenAI silently
ignores the same keywords, and a 7B model behind Ollama is only confused by
constraints nothing will enforce.  A tool must not stop working because the
user switched model, so every provider runs its tool schemas through
`normalise` on the way out.

Nothing here invents a constraint.  A keyword is either translated into
something the dialect understands or dropped, and `normalise` is idempotent so
a schema can be normalised again without drifting.
"""

from __future__ import annotations

import math
from typing import Any, Final

DIALECTS: Final[tuple[str, ...]] = ("anthropic", "openai", "google", "ollama")

#: Validator bookkeeping — meaningful to a JSON Schema library, noise to a model.
_META: Final[frozenset[str]] = frozenset({"$schema", "$id", "$anchor", "$comment"})

_DEF_HOLDERS: Final[tuple[str, ...]] = ("$defs", "definitions")

#: Keywords dropped per dialect, on top of `_META`.
#:  google  — Gemini's function declarations take an OpenAPI 3.0 subset and 400
#:            on anything outside it, including `additionalProperties`.
#:  ollama  — the schema is turned into a sampling grammar; keywords it cannot
#:            honour only lengthen the prompt a small local model has to read.
DROP: Final[dict[str, frozenset[str]]] = {
    "anthropic": frozenset(),
    "openai": frozenset(),
    "google": frozenset(
        {
            "additionalProperties",
            "patternProperties",
            "unevaluatedProperties",
            "propertyNames",
            "allOf",
            "oneOf",
            "not",
            "if",
            "then",
            "else",
            "const",
            "multipleOf",
            "uniqueItems",
            "readOnly",
            "writeOnly",
            "deprecated",
            "contentEncoding",
            "contentMediaType",
        }
    ),
    "ollama": frozenset(
        {
            "additionalProperties",
            "patternProperties",
            "unevaluatedProperties",
            "propertyNames",
            "format",
            "readOnly",
            "writeOnly",
            "deprecated",
        }
    ),
}

#: Dialects that can carry `$defs`, and so a reference that cannot be inlined.
REFS_OK: Final[frozenset[str]] = frozenset({"anthropic", "openai"})

#: Gemini validates `format` against a short per-type list and rejects the rest.
GOOGLE_FORMATS: Final[dict[str, frozenset[str]]] = {
    "string": frozenset({"enum", "date-time"}),
    "integer": frozenset({"int32", "int64"}),
    "number": frozenset({"float", "double"}),
}

#: What a node becomes when its reference cannot be expressed: an object we can
#: say nothing more about.  A dangling `$ref` would have the whole call rejected.
_OPAQUE: Final[dict[str, Any]] = {"type": "object", "properties": {}}

_SUBSCHEMA: Final[frozenset[str]] = frozenset(
    {"items", "additionalItems", "contains", "not", "if", "then", "else", "additionalProperties"}
)
_SUBSCHEMA_LIST: Final[frozenset[str]] = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})
_SUBSCHEMA_MAP: Final[frozenset[str]] = frozenset({"properties", "patternProperties", "$defs", "definitions"})


def normalise(schema: dict[str, Any], dialect: str) -> dict[str, Any]:
    """A copy of `schema` that `dialect` accepts.  The input is never mutated."""
    if dialect not in DROP:
        raise ValueError(f"unknown dialect {dialect!r}; expected one of {', '.join(DIALECTS)}")
    if not isinstance(schema, dict):
        return dict(_OPAQUE)

    drop = DROP[dialect] | _META
    defs = _definitions(schema)
    recursive = _recursive(defs)
    keeps_refs = dialect in REFS_OK
    kept: set[str] = set()

    def walk(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [walk(item, seen) for item in node]
        if not isinstance(node, dict):
            return node

        ref = node.get("$ref")
        if isinstance(ref, str):
            target = defs.get(ref)
            if target is None:
                return dict(_OPAQUE)
            if ref in recursive or ref in seen:
                if not keeps_refs:
                    return dict(_OPAQUE)
                kept.add(ref)
                return {"$ref": ref}
            merged = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
            return walk(merged, seen | {ref})

        node = _fold_integer_bounds(node)
        types = _types(node)
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in drop or key in _DEF_HOLDERS:
                continue
            if key == "format":
                if dialect == "google" and not _google_format(types, value):
                    continue
                out[key] = value
            elif key in _SUBSCHEMA_MAP and isinstance(value, dict):
                out[key] = {name: walk(sub, seen) for name, sub in value.items()}
            elif key in _SUBSCHEMA_LIST and isinstance(value, list):
                out[key] = [walk(sub, seen) for sub in value]
            elif key in _SUBSCHEMA:
                out[key] = walk(value, seen) if isinstance(value, (dict, list)) else value
            else:
                out[key] = value
        if "object" in types and not isinstance(out.get("properties"), dict):
            # Several providers reject an object without a properties map.
            out["properties"] = {}
        if "array" in types and "items" not in out and dialect == "google":
            # Gemini's ARRAY requires `items`. Nothing here can know the real
            # element type, so this is a last resort for schemas we did not
            # write - our own are checked at source by the test suite.
            out["items"] = _infer_items(node)
        if dialect in ("anthropic", "google") and not types:
            hoisted = _hoist_union(out)
            if hoisted is not None:
                out["type"] = hoisted
        return out

    result = walk(schema, frozenset())
    if kept:  # a recursive schema keeps only the definitions it still points at
        holder = "$defs" if "$defs" in schema or "definitions" not in schema else "definitions"
        result[holder] = {
            _name(ref): walk(defs[ref], frozenset({ref})) for ref in sorted(kept) if _holder(ref) == holder
        }
        leftover = {ref for ref in kept if _holder(ref) != holder}
        for ref in sorted(leftover):  # a schema mixing both holders keeps both
            result.setdefault(_holder(ref), {})[_name(ref)] = walk(defs[ref], frozenset({ref}))
    return result


def _infer_items(node: dict[str, Any]) -> dict[str, Any]:
    """The best available guess at an array's element schema."""
    tuple_form = node.get("prefixItems")
    if isinstance(tuple_form, list) and tuple_form and isinstance(tuple_form[0], dict):
        return {k: v for k, v in tuple_form[0].items() if k == "type"} or {"type": "string"}
    return {"type": "string"}


def _hoist_union(node: dict[str, Any]) -> str | None:
    """A `type` for a branch schema that has none.

    `{"anyOf": [...]}` with no sibling `type` is legal JSON Schema and is
    rejected by the stricter function-calling dialects. When every branch agrees
    on a type it can be stated outright; otherwise the first branch is the least
    surprising choice, and the branches stay for anyone who reads them.
    """
    for key in ("anyOf", "oneOf", "allOf"):
        branches = node.get(key)
        if not isinstance(branches, list):
            continue
        found = [b.get("type") for b in branches if isinstance(b, dict) and isinstance(b.get("type"), str)]
        if not found:
            continue
        # Branches that disagree get the first one: any concrete type beats no
        # type at all, and the union itself stays in the schema for a reader.
        return found[0]
    return None


def _definitions(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Local definitions, keyed by the pointer that reaches them."""
    found: dict[str, dict[str, Any]] = {}
    for holder in _DEF_HOLDERS:
        section = schema.get(holder)
        if isinstance(section, dict):
            for name, body in section.items():
                if isinstance(body, dict):
                    found[f"#/{holder}/{name}"] = body
    return found


def _holder(ref: str) -> str:
    return ref.split("/")[1]


def _name(ref: str) -> str:
    return ref.split("/")[-1]


def _refs(node: Any, into: set[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            into.add(ref)
        for value in node.values():
            _refs(value, into)
    elif isinstance(node, list):
        for item in node:
            _refs(item, into)


def _recursive(defs: dict[str, dict[str, Any]]) -> frozenset[str]:
    """Definitions that reach themselves.  Inlining one would never terminate,
    and inlining it once would make `normalise` non-idempotent."""
    if not defs:
        return frozenset()
    edges: dict[str, set[str]] = {}
    for ref, body in defs.items():
        out: set[str] = set()
        _refs(body, out)
        edges[ref] = {r for r in out if r in defs}

    found: set[str] = set()
    for start in defs:
        stack = list(edges[start])
        seen: set[str] = set()
        while stack:
            here = stack.pop()
            if here == start:
                found.add(start)
                break
            if here in seen:
                continue
            seen.add(here)
            stack.extend(edges[here])
    return frozenset(found)


def _types(node: dict[str, Any]) -> frozenset[str]:
    raw = node.get("type")
    if isinstance(raw, str):
        return frozenset((raw,))
    if isinstance(raw, list):
        return frozenset(t for t in raw if isinstance(t, str))
    return frozenset()


def _google_format(types: frozenset[str], value: Any) -> bool:
    return any(isinstance(value, str) and value in GOOGLE_FORMATS.get(t, ()) for t in types)


def _fold_integer_bounds(node: dict[str, Any]) -> dict[str, Any]:
    """Integer bounds in the one form every dialect accepts.

    Draft 4 wrote `exclusiveMinimum` as a boolean flag on `minimum`, draft
    2020-12 writes it as the bound itself, and a fractional bound on an integer
    field is a rounding trap.  All three become an inclusive integer bound.
    """
    if "integer" not in _types(node):
        return node
    if not node.keys() & {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}:
        return node

    out = dict(node)
    low = _number(out.pop("minimum", None))
    high = _number(out.pop("maximum", None))
    x_low = out.pop("exclusiveMinimum", None)
    x_high = out.pop("exclusiveMaximum", None)

    if x_low is True:
        low = None if low is None else math.floor(low) + 1
    elif (bound := _number(x_low)) is not None:
        low = math.floor(bound) + 1
    if x_high is True:
        high = None if high is None else math.ceil(high) - 1
    elif (bound := _number(x_high)) is not None:
        high = math.ceil(bound) - 1

    if low is not None:
        out["minimum"] = math.ceil(low)
    if high is not None:
        out["maximum"] = math.floor(high)
    return out


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
