"""Live checks against a real provider.

Opt-in: these are the only tests that touch the network or spend tokens, so
they stay dormant unless `OFFSET_LIVE=1` is set.  Run them after adding a key:

    OFFSET_LIVE=1 GEMINI_API_KEY=... python3 -m pytest tests/test_live.py -q -s

`test_an_invalid_key_becomes_an_event` needs no valid credential — it only
needs the network — so it is the one that can always prove the error path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from offset.core.agent import Agent, AgentConfig, Finished
from offset.core.entries import MESSAGE, TOOL_CALL, TOOL_RESULT
from offset.core.session import Session
from offset.providers.base import Message, Request, Stop, StreamError
from offset.providers.google import Google
from offset.providers.registry import credential, provider_for, redact
from offset.tools.base import ToolContext, Toolbox
from offset.tools.builtin import builtin_tools
from offset.tools.runtime import Approval, Runtime

pytestmark = pytest.mark.skipif(
    not os.environ.get("OFFSET_LIVE"),
    reason="set OFFSET_LIVE=1 to spend real tokens",
)

MODEL = os.environ.get("OFFSET_LIVE_MODEL", "gemini-2.0-flash")


def _key() -> str:
    key = credential(provider_for("google"))
    if not key:
        pytest.skip("no google credential available")
    return key


def _agent(tmp_path: Path, key: str, **config) -> tuple[Agent, Session]:
    session = Session.create(tmp_path / "sessions")
    runtime = Runtime(
        Toolbox(builtin_tools()),
        ToolContext(cwd=tmp_path, timeout=60.0),
        Approval(mode="yolo"),
    )
    agent = Agent(
        session, runtime,
        AgentConfig(model=MODEL, max_steps=4, max_tokens=400,
                    system="You are a terse coding agent. Use tools instead of guessing.",
                    **config),
        api_key=key,
    )
    return agent, session


def test_a_real_turn_streams_text():
    turn = Google().complete(
        Request(model=MODEL, messages=[Message("user", "Reply with exactly: OK")], max_tokens=20),
        api_key=_key(),
    )
    assert turn.error is None, turn.error
    assert turn.text.strip(), "the provider returned no text"
    assert turn.usage.input > 0 and turn.usage.output > 0, "usage was not reported"


def test_a_real_tool_call_round_trips(tmp_path):
    """The whole stack: model asks for a tool, we run it, it answers from the result."""
    (tmp_path / "VERSION.txt").write_text("offset 0.1.0\n", encoding="utf-8")
    agent, session = _agent(tmp_path, _key())

    finished = None
    for event in agent.run("Read the file VERSION.txt and tell me the exact version string in it."):
        if isinstance(event, Finished):
            finished = event

    kinds = [e.type for e in session.transcript()]
    assert TOOL_CALL in kinds, f"the model never called a tool: {kinds}"
    assert TOOL_RESULT in kinds, "no tool result was recorded"
    assert finished is not None and finished.reason in ("stop", "tool_use")
    answer = [e for e in session.transcript() if e.type == MESSAGE][-1].text
    assert "0.1.0" in answer, f"wrong final answer: {answer!r}"

    reopened = Session.open(session.path)
    assert [e.type for e in reopened.transcript()] == kinds, "history did not survive a reload"


def test_an_invalid_key_becomes_an_event_not_a_crash(tmp_path):
    """A rejected credential must degrade to a StreamError, over the real wire."""
    events = list(Google().stream(
        Request(model=MODEL, messages=[Message("user", "hello")], max_tokens=10),
        api_key="AIzaSyDefinitelyNotAValidKeyAtAll000000000",
    ))
    errors = [e for e in events if isinstance(e, StreamError)]
    assert errors, f"expected a StreamError, got {[type(e).__name__ for e in events]}"
    assert errors[0].status == 400
    assert not errors[0].retryable, "an invalid key must never be retried"
    assert "API key" in errors[0].message or "api key" in errors[0].message.lower()
    assert isinstance(events[-1], Stop) and events[-1].reason == "error"


def test_a_rejected_key_stops_the_loop_cleanly(tmp_path):
    agent, session = _agent(tmp_path, "AIzaSyDefinitelyNotAValidKeyAtAll000000000")
    result = agent.send("this should fail")
    assert result.stop_reason == "error"
    assert result.error and "key" in result.error.lower()
    assert result.steps == 1, "a permanent auth failure must not loop"


def test_secrets_never_reach_the_transcript(tmp_path):
    key = "AIzaSyDefinitelyNotAValidKeyAtAll000000000"
    agent, session = _agent(tmp_path, key)
    agent.send("this should fail")
    body = session.path.read_text(encoding="utf-8")
    assert key not in body, "the API key was written into the session file"
    assert key not in redact(f"failure for {key}", key)
