"""Wire-format conversion, streaming assembly, and retry behaviour.

Every provider is exercised against a captured stream of its own protocol, so
these tests fail when a translation is wrong rather than when a network is
down.  No test here touches the network.
"""

from __future__ import annotations

import json

import pytest

from offset.providers import anthropic, google, ollama, openai
from offset.providers.base import (
    Message,
    Request,
    Stop,
    StreamError,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolSpec,
    Turn,
    TurnBuilder,
    Usage,
)
from offset.providers.mock import Mock, script
from offset.providers.registry import (
    credential,
    info,
    provider_for,
    redact,
    resolve,
    search,
    store_credential,
)
from offset.providers.sse import iter_json_frames, iter_sse
from offset.providers.transport import HTTPFailure, Retry


def lines(text: str) -> list[bytes]:
    return [l.encode() for l in text.strip("\n").split("\n")]


READ_TOOL = ToolSpec("read", "read a file", {"type": "object", "properties": {"path": {"type": "string"}}})


def convo() -> Request:
    return Request(
        model="m",
        system="be terse",
        messages=[
            Message("user", "read setup.py"),
            Message("assistant", "", tool_calls=[ToolCall("c1", "read", {"path": "setup.py"})]),
            Message("tool", "file contents", tool_call_id="c1", name="read"),
        ],
        tools=[READ_TOOL],
    )


# -- framing ----------------------------------------------------------------


def test_sse_joins_multiline_data_and_skips_comments():
    got = list(iter_sse(lines("""
: keepalive
event: message
data: {"a":
data: 1}

data: tail
""")))
    assert got == [("message", '{"a":\n1}'), (None, "tail")]


def test_ndjson_skips_junk_and_done_markers():
    got = list(iter_json_frames(lines("""
{"ok": 1}
not json
data: {"ok": 2}
[DONE]
""")))
    assert got == [{"ok": 1}, {"ok": 2}]


# -- turn assembly ----------------------------------------------------------


def test_builder_assembles_text_thinking_and_calls():
    turn = TurnBuilder().consume(iter(script(
        "hello world",
        thinking="hmm",
        tool_calls=[("c1", "read", {"path": "a.py"})],
        usage=Usage(input=10, output=4),
    ))).finish()
    assert turn.text == "hello world"
    assert turn.thinking == "hmm"
    assert turn.tool_calls == [ToolCall("c1", "read", {"path": "a.py"})]
    assert (turn.usage.input, turn.usage.output) == (10, 4)
    assert turn.stop_reason == "tool_use"


def test_tool_calls_keep_their_order():
    turn = TurnBuilder().consume(iter(script(
        tool_calls=[("a", "first", {"i": 1}), ("b", "second", {"i": 2}), ("c", "third", {"i": 3})]
    ))).finish()
    assert [c.name for c in turn.tool_calls] == ["first", "second", "third"]


def test_malformed_arguments_are_surfaced_not_dropped():
    """A broken call must remain visible so the model can be re-prompted."""
    builder = TurnBuilder()
    builder.feed(ToolCallDelta(0, id="c1", name="read"))
    builder.feed(ToolCallDelta(0, args_delta='{"path": "unterminated'))
    turn = builder.finish()
    assert len(turn.tool_calls) == 1
    assert turn.malformed and turn.malformed[0].raw == '{"path": "unterminated'
    assert turn.tool_calls[0].args == {}


def test_non_object_arguments_are_treated_as_malformed():
    builder = TurnBuilder()
    builder.feed(ToolCallDelta(0, id="c", name="read", args_delta="[1,2,3]"))
    assert builder.finish().malformed


def test_empty_arguments_are_an_empty_object():
    builder = TurnBuilder()
    builder.feed(ToolCallDelta(0, id="c", name="ping"))
    turn = builder.finish()
    assert turn.tool_calls[0].args == {} and not turn.malformed


def test_stop_reason_is_normalised_when_tools_were_called():
    builder = TurnBuilder()
    builder.feed(ToolCallDelta(0, id="c", name="read", args_delta="{}"))
    builder.feed(Stop("stop"))
    assert builder.finish().stop_reason == "tool_use"


def test_stream_error_marks_the_turn():
    turn = TurnBuilder().consume(iter([TextDelta("part"), StreamError("boom")])).finish()
    assert turn.error == "boom" and turn.stop_reason == "error"


# -- anthropic --------------------------------------------------------------

ANTHROPIC_STREAM = """
event: message_start
data: {"type":"message_start","message":{"usage":{"input_tokens":12,"cache_read_input_tokens":3}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Reading"}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":"}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\\"a.py\\"}"}}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":25}}

event: message_stop
data: {"type":"message_stop"}
"""


def test_anthropic_stream_round_trip():
    turn = TurnBuilder().consume(anthropic.parse(iter(lines(ANTHROPIC_STREAM)))).finish()
    assert turn.text == "Reading"
    assert turn.tool_calls == [ToolCall("toolu_1", "read", {"path": "a.py"})]
    assert turn.stop_reason == "tool_use"
    assert (turn.usage.input, turn.usage.output, turn.usage.cache_read) == (12, 25, 3)


def test_anthropic_thinking_is_separated_from_text():
    stream = """
event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weighing"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"answer"}}
"""
    turn = TurnBuilder().consume(anthropic.parse(iter(lines(stream)))).finish()
    assert (turn.thinking, turn.text) == ("weighing", "answer")


def test_anthropic_payload_shape():
    p = anthropic.build_payload(convo())
    assert p["system"] == "be terse"
    assert p["tools"][0]["input_schema"] == READ_TOOL.schema
    assert p["messages"][1]["content"][0]["type"] == "tool_use"
    # a tool result is delivered as a user turn carrying a tool_result block
    assert p["messages"][2]["role"] == "user"
    assert p["messages"][2]["content"][0]["tool_use_id"] == "c1"


def test_anthropic_thinking_drops_temperature():
    req = Request(model="m", messages=[Message("user", "hi")], temperature=0.7, thinking_budget=1024)
    p = anthropic.build_payload(req)
    assert p["thinking"]["budget_tokens"] == 1024
    assert "temperature" not in p, "the API rejects both together"


# -- openai -----------------------------------------------------------------

OPENAI_STREAM = """
data: {"choices":[{"delta":{"content":"Hi"}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\\"path\\""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"b.py\\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: {"usage":{"prompt_tokens":8,"completion_tokens":16,"prompt_tokens_details":{"cached_tokens":4}},"choices":[]}

data: [DONE]
"""


def test_openai_stream_round_trip():
    turn = TurnBuilder().consume(openai.parse(iter(lines(OPENAI_STREAM)))).finish()
    assert turn.text == "Hi"
    assert turn.tool_calls == [ToolCall("call_1", "read", {"path": "b.py"})]
    assert turn.stop_reason == "tool_use"
    assert (turn.usage.input, turn.usage.output, turn.usage.cache_read) == (8, 16, 4)


def test_openai_reasoning_field_becomes_thinking():
    stream = 'data: {"choices":[{"delta":{"reasoning_content":"step one"}}]}'
    turn = TurnBuilder().consume(openai.parse(iter(lines(stream)))).finish()
    assert turn.thinking == "step one" and turn.text == ""


def test_openai_payload_shape():
    p = openai.build_payload(convo())
    assert p["messages"][0] == {"role": "system", "content": "be terse"}
    assert p["messages"][2]["tool_calls"][0]["function"]["arguments"] == json.dumps({"path": "setup.py"})
    assert p["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "file contents"}
    assert p["tools"][0]["function"]["parameters"] == READ_TOOL.schema


def test_openai_compatible_endpoints_are_configuration_not_code():
    ds, orx = openai.deepseek(), openai.openrouter()
    assert (ds.name, ds.env_keys) == ("deepseek", ("DEEPSEEK_API_KEY",))
    assert orx.base_url.endswith("openrouter.ai/api/v1")
    assert openai.llamacpp().env_keys == ()


# -- google -----------------------------------------------------------------

GOOGLE_STREAM = """
data: {"candidates":[{"content":{"parts":[{"text":"Sure"}]}}]}

data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"read","args":{"path":"c.py"}}}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":9}}

data: {"candidates":[{"finishReason":"STOP","content":{"parts":[]}}]}
"""


def test_google_stream_round_trip():
    turn = TurnBuilder().consume(google.parse(iter(lines(GOOGLE_STREAM)))).finish()
    assert turn.text == "Sure"
    assert turn.tool_calls[0].name == "read"
    assert turn.tool_calls[0].args == {"path": "c.py"}
    assert turn.stop_reason == "tool_use"
    assert (turn.usage.input, turn.usage.output) == (5, 9)


def test_google_uses_the_model_role_and_function_response():
    p = google.build_payload(convo())
    assert p["contents"][1]["role"] == "model"
    assert p["contents"][2]["parts"][0]["functionResponse"]["name"] == "read"
    assert p["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert p["tools"][0]["functionDeclarations"][0]["name"] == "read"


def test_google_thought_parts_are_thinking():
    stream = 'data: {"candidates":[{"content":{"parts":[{"text":"pondering","thought":true},{"text":"result"}]}}]}'
    turn = TurnBuilder().consume(google.parse(iter(lines(stream)))).finish()
    assert (turn.thinking, turn.text) == ("pondering", "result")


# -- ollama -----------------------------------------------------------------

OLLAMA_STREAM = """
{"message":{"content":"local "},"done":false}
{"message":{"content":"answer"},"done":false}
{"message":{"content":"","tool_calls":[{"function":{"name":"read","arguments":{"path":"d.py"}}}]},"done":false}
{"message":{"content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":7}
"""


def test_ollama_stream_round_trip():
    turn = TurnBuilder().consume(ollama.parse(iter(lines(OLLAMA_STREAM)))).finish()
    assert turn.text == "local answer"
    assert turn.tool_calls[0].args == {"path": "d.py"}
    assert (turn.usage.input, turn.usage.output) == (3, 7)
    assert turn.stop_reason == "tool_use"


def test_ollama_needs_no_key():
    assert provider_for("ollama").env_keys == ()


# -- transport --------------------------------------------------------------


def test_retry_prefers_the_server_instruction():
    r = Retry(base=1.0, cap=30.0)
    assert r.delay(0, retry_after=12.0) == 12.0
    assert r.delay(0, retry_after=999.0) == 30.0  # still capped


def test_backoff_grows_and_stays_capped():
    r = Retry(base=1.0, cap=8.0, jitter=0.0)
    assert [r.delay(i) for i in range(5)] == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_only_transient_statuses_retry():
    assert HTTPFailure(429, "").retryable
    assert HTTPFailure(503, "").retryable
    assert not HTTPFailure(400, "").retryable
    assert not HTTPFailure(401, "").retryable


def test_failure_extracts_the_provider_message():
    body = json.dumps({"error": {"message": "you are out of credits", "type": "billing"}})
    assert HTTPFailure(402, body).detail() == "you are out of credits"
    assert HTTPFailure(500, "<html>gateway</html>").detail() == "<html>gateway</html>"


def test_post_lines_retries_then_succeeds(monkeypatch):
    import urllib.error

    from offset.providers import transport

    calls, slept = {"n": 0}, []

    class Response:
        def __enter__(self):
            return iter([b'data: {"ok":1}'])

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(request.full_url, 429, "slow down", {"Retry-After": "2"}, None)
        return Response()

    monkeypatch.setattr(transport.urllib.request, "urlopen", fake_urlopen)
    got = list(transport.post_lines("http://x", {}, {}, retry=Retry(attempts=4), sleep=slept.append))
    assert got == [b'data: {"ok":1}']
    assert calls["n"] == 3
    assert slept == [2.0, 2.0], "the server's Retry-After must be honoured"


def test_post_lines_gives_up_on_permanent_errors(monkeypatch):
    import urllib.error

    from offset.providers import transport

    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(request.full_url, 401, "nope", {}, None)

    monkeypatch.setattr(transport.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(HTTPFailure) as caught:
        list(transport.post_lines("http://x", {}, {}, retry=Retry(attempts=5), sleep=lambda _: None))
    assert caught.value.status == 401
    assert calls["n"] == 1, "an auth failure must not be retried"


def test_http_errors_become_events_not_exceptions(monkeypatch):
    import urllib.error

    from offset.providers import transport

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, None)

    monkeypatch.setattr(transport.urllib.request, "urlopen", fake_urlopen)
    events = list(anthropic.Anthropic().stream(Request(model="m", messages=[Message("user", "hi")]), api_key="k"))
    assert any(isinstance(e, StreamError) for e in events)
    assert isinstance(events[-1], Stop) and events[-1].reason == "error"


# -- registry ---------------------------------------------------------------


def test_unknown_models_still_resolve():
    """A model released tomorrow must work without editing the catalogue."""
    provider, meta = resolve("claude-opus-9-20991231")
    assert provider.name == "anthropic" and meta.id == "claude-opus-9-20991231"
    assert info("gpt-6-turbo").provider == "openai"
    assert info("gemini-4-ultra").provider == "google"
    assert info("some-random-local-gguf").provider == "ollama"


def test_catalogue_search():
    assert any(m.id == "deepseek-reasoner" for m in search("deepseek"))
    assert len(search("")) == len(search(""))
    assert search("zzzz") == []


def test_credentials_prefer_the_environment(monkeypatch, tmp_path):

    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    store_credential("anthropic", "from-disk")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert credential(provider_for("anthropic")) == "from-env"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert credential(provider_for("anthropic")) == "from-disk"


def test_stored_credentials_are_owner_only(monkeypatch, tmp_path):

    monkeypatch.setenv("OFFSET_HOME", str(tmp_path))
    path = store_credential("openai", "sk-secret")
    assert oct(path.stat().st_mode)[-3:] == "600"


def test_missing_credentials_never_raise(monkeypatch, tmp_path):
    # An empty home, so this asks about nothing rather than about whatever the
    # person running the suite happens to have signed into.
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "empty"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    assert credential(provider_for("anthropic")) is None


def test_redaction_keeps_secrets_out_of_output():
    key = "sk-abcdefghijklmnop"
    assert key not in redact(f"auth failed for {key}", key)


# -- mock provider ----------------------------------------------------------


def test_mock_provider_completes_a_turn():
    provider = Mock([script("done", usage=Usage(input=1, output=2))])
    turn = provider.complete(Request(model="mock", messages=[Message("user", "go")]))
    assert isinstance(turn, Turn) and turn.text == "done"
    assert provider.turns == 1 and provider.requests[0].model == "mock"


def test_mock_can_script_per_request():
    provider = Mock(lambda req: script(f"saw {len(req.messages)} messages"))
    assert provider.complete(Request(model="mock", messages=[Message("user", "a")])).text == "saw 1 messages"


def test_request_can_be_retargeted_at_another_model():
    """The primitive multi-model relies on: same conversation, new model."""
    base = Request(model="a", messages=[Message("user", "hi")], tools=[READ_TOOL])
    other = base.with_model("b")
    assert other.model == "b" and base.model == "a"
    assert other.messages is base.messages and other.tools == base.tools
