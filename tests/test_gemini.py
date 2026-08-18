"""What a real Gemini key exposed that captured streams could not.

Both bugs here were invisible until an actual Google key was pasted into an
actual session: the key was ignored in favour of a stale shell variable, and once
that was fixed, every tool call was rejected because Gemini 3 requires its own
thought signature to come back with the call.

Nothing here touches the network. The wire shapes are the ones observed from
`generativelanguage.googleapis.com`, transcribed.
"""

from __future__ import annotations

import json

import pytest

from offset.core.agent import to_messages
from offset.core.entries import Entry
from offset.providers import google
from offset.providers.base import Message, Request, ToolCall, ToolSpec, TurnBuilder

SIGNATURE = "CvIBAdHtim8Yz9F" * 12  # 508 chars in the wild; length is not the point


def sse(*objects: dict) -> list[bytes]:
    return [f"data: {json.dumps(obj)}\n".encode() for obj in objects] + [b"\n"]


def call_part(name: str, args: dict, *, signature: str | None = SIGNATURE) -> dict:
    """A candidate part exactly as Gemini 3 sends one for a tool call."""
    part: dict = {"functionCall": {"name": name, "args": args}}
    if signature is not None:
        part["thoughtSignature"] = signature
    return {"candidates": [{"content": {"parts": [part]}}]}


# -- reading the signature off the wire --------------------------------------


def test_a_thought_signature_is_carried_off_the_stream():
    events = list(google.parse(iter(sse(call_part("read", {"path": "greet.py"})))))
    deltas = [e for e in events if hasattr(e, "args_delta")]
    assert deltas, "the function call was not parsed at all"
    assert deltas[0].signature == SIGNATURE


def test_a_call_without_a_signature_is_still_a_call():
    """Older Gemini models send none, and must keep working."""
    events = list(google.parse(iter(sse(call_part("read", {"path": "x"}, signature=None)))))
    deltas = [e for e in events if hasattr(e, "args_delta")]
    assert deltas and deltas[0].signature is None


def test_the_builder_keeps_the_signature_on_the_assembled_call():
    turn = TurnBuilder().consume(
        google.parse(iter(sse(call_part("read", {"path": "greet.py"})))))
    finished = turn.finish()
    assert len(finished.tool_calls) == 1
    assert finished.tool_calls[0].signature == SIGNATURE
    assert finished.tool_calls[0].args == {"path": "greet.py"}


# -- sending it back ---------------------------------------------------------


def ask_with(call: ToolCall) -> dict:
    return google.build_payload(Request(
        model="gemini-3-flash-preview",
        messages=[
            Message("user", "read greet.py"),
            Message("assistant", "", tool_calls=[call]),
            Message("tool", "def greet(name): ...", tool_call_id=call.id, name=call.name),
        ],
        tools=[ToolSpec("read", "read a file", {"type": "object", "properties": {}})],
    ))


def function_parts(payload: dict) -> list[dict]:
    return [part for content in payload["contents"]
            for part in content["parts"] if "functionCall" in part]


def test_the_signature_goes_back_on_the_function_call_part():
    """The exact shape Google demands: on the part, beside the call.

    Without it every request after the first tool call is refused with "Function
    call is missing a thought_signature in functionCall parts", so tool use was
    impossible on Gemini 3 - which is every Gemini model a new key can reach.
    """
    parts = function_parts(ask_with(ToolCall("c1", "read", {"path": "greet.py"}, signature=SIGNATURE)))
    assert len(parts) == 1
    assert parts[0]["thoughtSignature"] == SIGNATURE
    assert parts[0]["functionCall"]["name"] == "read"


def test_no_signature_means_the_field_is_absent_not_null():
    """A null would be a value; Google's validator is not amused by one."""
    parts = function_parts(ask_with(ToolCall("c1", "read", {"path": "x"})))
    assert "thoughtSignature" not in parts[0]


def test_a_signature_survives_a_full_round_trip():
    turn = TurnBuilder().consume(
        google.parse(iter(sse(call_part("read", {"path": "greet.py"}))))).finish()
    parts = function_parts(ask_with(turn.tool_calls[0]))
    assert parts[0]["thoughtSignature"] == SIGNATURE


# -- surviving the session store ---------------------------------------------


def test_a_signature_survives_being_written_to_the_session_and_read_back():
    """History is rebuilt from entries, so an in-memory-only signature is lost.

    This is what still failed after the provider was fixed: the call went out
    correctly the first time and was rejected on the second step, because the
    reconstruction dropped what the provider had carefully kept.
    """
    entries = [
        Entry(id="e1", parent=None, type="message", ts=1.0,
              data={"role": "user", "text": "read greet.py"}),
        Entry(id="e2", parent="e1", type="tool_call", ts=2.0,
              data={"id": "c1", "tool": "read", "args": {"path": "greet.py"},
                    "signature": SIGNATURE}),
        Entry(id="e3", parent="e2", type="tool_result", ts=3.0,
              data={"id": "c1", "tool": "read", "content": "def greet(name): ..."}),
    ]
    messages = to_messages(entries)
    calls = [c for m in messages for c in m.tool_calls]
    assert calls, "the tool call did not survive the rebuild"
    assert calls[0].signature == SIGNATURE, "the signature was dropped rebuilding history"


def test_an_entry_written_without_a_signature_rebuilds_cleanly():
    entries = [
        Entry(id="e1", parent=None, type="tool_call", ts=1.0,
              data={"id": "c1", "tool": "read", "args": {"path": "x"}}),
    ]
    calls = [c for m in to_messages(entries) for c in m.tool_calls]
    assert calls and calls[0].signature is None


# -- the catalogue -----------------------------------------------------------


def test_no_retired_gemini_model_is_advertised():
    """Every `gemini-2.5-*` id answers 404 for a key made today.

    Verified against a real key: "This model is no longer available to new
    users". Shipping them meant a new Google account could not use a single
    Google model in the picker.
    """
    from offset.providers.registry import MODELS

    retired = [m.id for m in MODELS if m.provider == "google"
               and (m.id.startswith("gemini-2.") or m.id.startswith("gemini-1."))]
    assert not retired, f"these are gone for new keys: {retired}"


def test_the_google_models_we_ship_are_the_ones_that_answered():
    from offset.providers.registry import MODELS

    ids = {m.id for m in MODELS if m.provider == "google"}
    assert "gemini-3-flash-preview" in ids, "the model verified working must be offered"
    assert ids, "google must not be left with no models at all"


@pytest.mark.parametrize("model", ["gemini-3-flash-preview", "gemini-3.1-flash-lite"])
def test_every_google_model_declares_limits_the_api_reports(model):
    from offset.providers.registry import info

    meta = info(model)
    assert meta.context == 1_048_576
    assert meta.max_output == 65_536
