"""Compaction that happens without being asked.

`/compact` already existed, which meant a long session survived only if the
user noticed in time.  The failure it prevents is the worst one this program
has: the provider rejects an over-long request outright, so the session dies
exactly when it holds the most work.

These tests are about *when* it fires and what happens when it cannot, not
about summary quality - `test_compaction.py` already pins the boundary rules
and the "nothing is destroyed" invariant that this leans on.
"""

from __future__ import annotations

from offset.core import compaction
from offset.core.agent import Agent, AgentConfig, Compacted, Finished, to_messages
from offset.core.entries import TOOL_CALL, TOOL_RESULT
from offset.core.session import Session
from offset.providers.mock import Mock, script
from offset.providers.registry import ModelInfo
from offset.tools.base import Toolbox, ToolContext
from offset.tools.runtime import Approval, Runtime


def make(tmp_path, *, budget: int = 8192, **config):
    session = Session.create(tmp_path)
    runtime = Runtime(Toolbox([]), ToolContext(cwd=tmp_path, timeout=5.0), Approval(mode="yolo"))
    provider = Mock([script("done"), script("done")])
    meta = ModelInfo("mock", "mock", "mock", budget, 1024)
    agent = Agent(
        session, runtime, AgentConfig(model="mock", max_steps=2, **config),
        resolver=lambda _m: (provider, meta), provider=provider,
    )
    return agent, session


def bulk(session: Session, turns: int, size: int = 4000) -> None:
    """Enough history that any sane budget is exceeded."""
    for n in range(turns):
        session.say("user", f"request {n} " + "x" * size)
        session.say("assistant", f"answer {n} " + "z" * size)


def events(agent: Agent, prompt: str = "next") -> list:
    return list(agent.run(prompt))


# -- when it fires ------------------------------------------------------------


def test_a_short_session_is_left_alone(tmp_path):
    """The common case must cost nothing: no summariser call, no event."""
    agent, session = make(tmp_path)
    session.say("user", "hi")
    assert not [e for e in events(agent) if isinstance(e, Compacted)]


def test_a_long_session_compacts_itself_before_the_turn(tmp_path):
    """The whole point: nobody typed `/compact` and the turn still went out."""
    agent, session = make(tmp_path, budget=2_000)
    bulk(session, 8)
    before = compaction.estimate_tokens(to_messages(session.transcript()))

    seen = events(agent)

    compacted = [e for e in seen if isinstance(e, Compacted)]
    assert compacted, "history was over budget and nothing was compacted"
    assert compacted[0].summarised > 0
    after = compaction.estimate_tokens(to_messages(session.transcript()))
    assert after < before, "compaction ran but the transcript did not shrink"


def test_the_turn_still_completes_after_compacting(tmp_path):
    """Compaction is a means, not an end.  A turn that gets summarised and then
    never answers has traded one failure for another."""
    agent, session = make(tmp_path, budget=2_000)
    bulk(session, 8)
    finished = [e for e in events(agent) if isinstance(e, Finished)]
    assert finished and finished[0].reason == "stop"


def test_compaction_happens_before_the_first_request_goes_out(tmp_path):
    """Compacting after the request is built saves nothing: the oversized body
    has already been sent."""
    agent, session = make(tmp_path, budget=2_000)
    bulk(session, 8)
    order = [type(e).__name__ for e in events(agent)]
    assert "Compacted" in order
    assert order.index("Compacted") < order.index("StepStarted")


def test_it_can_be_switched_off(tmp_path):
    agent, session = make(tmp_path, budget=2_000, auto_compact=False)
    bulk(session, 8)
    assert not [e for e in events(agent) if isinstance(e, Compacted)]


def test_the_threshold_is_honoured(tmp_path):
    """At a threshold of 1.0 nothing compacts until the window is genuinely
    full, which is the setting for somebody who would rather see the error."""
    agent, session = make(tmp_path, budget=200_000, compact_at=1.0)
    bulk(session, 4)
    assert not [e for e in events(agent) if isinstance(e, Compacted)]


def test_nothing_is_destroyed(tmp_path):
    """The summary is a new root; the originals stay on disk.  This is what
    makes doing it automatically defensible at all."""
    agent, session = make(tmp_path, budget=2_000)
    bulk(session, 8)
    before_ids = {e.id for e in session.all_entries()}

    events(agent)

    after_ids = {e.id for e in session.all_entries()}
    assert before_ids <= after_ids, "compaction removed entries from the log"


# -- when it cannot ------------------------------------------------------------


def test_a_failing_summariser_does_not_cost_the_turn(tmp_path, monkeypatch):
    """Turning "your history is long" into "offset does not work" would be
    strictly worse than the over-long request this was avoiding."""
    agent, session = make(tmp_path, budget=2_000)
    bulk(session, 8)

    def broken(_model, **_kw):
        def summarise(_prompt: str) -> str:
            raise RuntimeError("the summariser is down")

        return summarise

    monkeypatch.setattr(compaction, "model_summariser", broken)

    seen = events(agent)
    assert not [e for e in seen if isinstance(e, Compacted)], "claimed a compaction that failed"
    assert [e for e in seen if isinstance(e, Finished)], "the turn never finished"


def test_a_summariser_returning_nothing_is_not_announced(tmp_path, monkeypatch):
    """`compact` reports rather than raises here and leaves history untouched,
    so saying "compacted" would be a lie about what the model can still see."""
    agent, session = make(tmp_path, budget=2_000)
    bulk(session, 8)
    monkeypatch.setattr(compaction, "model_summariser", lambda _m, **_k: lambda _p: "")

    seen = events(agent)
    assert not [e for e in seen if isinstance(e, Compacted)]
    assert [e for e in seen if isinstance(e, Finished)]


def test_an_unmeasurable_session_is_left_alone(tmp_path, monkeypatch):
    """If we cannot even size the history, rewriting it is a gamble."""
    agent, session = make(tmp_path, budget=2_000)
    bulk(session, 8)

    def explode(*_a, **_k):
        raise RuntimeError("cannot estimate")

    monkeypatch.setattr(compaction, "estimate_tokens", explode)
    seen = events(agent)
    assert not [e for e in seen if isinstance(e, Compacted)]
    assert [e for e in seen if isinstance(e, Finished)]


def test_the_event_reports_real_numbers(tmp_path):
    agent, session = make(tmp_path, budget=2_000)
    bulk(session, 8)
    compacted = next(e for e in events(agent) if isinstance(e, Compacted))
    assert compacted.before > compacted.after > 0
    assert compacted.summarised >= 1


def test_a_tool_block_is_never_split_by_an_automatic_run(tmp_path):
    """The invariant `test_compaction.py` pins for the manual path has to hold
    for the automatic one too: a `tool_use` without its result is rejected by
    the provider outright."""
    agent, session = make(tmp_path, budget=2_000)
    for n in range(8):
        session.say("user", f"request {n} " + "x" * 3000)
        session.say("assistant", f"working {n} " + "z" * 3000)
        session.append(TOOL_CALL, {"id": f"c{n}", "tool": "read", "args": {"path": f"f{n}.py"}})
        session.append(TOOL_RESULT, {"id": f"c{n}", "tool": "read", "content": "y" * 2000})

    events(agent)

    messages = to_messages(session.transcript())
    open_calls = {c.id for m in messages for c in m.tool_calls}
    answered = {m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id}
    assert open_calls <= answered, "a tool call survived compaction without its result"
