"""The agent loop.

One turn is: send the conversation, stream the reply, run whatever tools were
requested, and repeat until the model stops asking for tools.  Everything that
happens is written to the session as it happens — not at the end — so an
interrupted turn leaves a complete, replayable history rather than a gap.

The loop yields events instead of printing them.  That is what lets the same
loop drive the TUI, a test, and a background branch running headless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

from offset.core.entries import BRANCH_SUMMARY, COMPACTION, MESSAGE, TOOL_CALL, TOOL_RESULT, Entry
from offset.core.session import Session
from offset.providers.base import (
    Event,
    Message,
    Provider,
    Request,
    Stop,
    StreamError,
    ToolCall,
    ToolSpec,
    Turn,
    TurnBuilder,
    Usage,
)
from offset.providers.auth import load as load_credential
from offset.providers.registry import ModelInfo, resolve
from offset.tools.runtime import Invocation, Runtime


# -- loop events ------------------------------------------------------------


@dataclass(slots=True)
class StepStarted(Event):
    index: int
    model: str


@dataclass(slots=True)
class ToolStarted(Event):
    call: ToolCall


@dataclass(slots=True)
class ToolFinished(Event):
    invocation: Invocation


@dataclass(slots=True)
class Finished(Event):
    reason: str
    usage: Usage
    steps: int
    text: str = ""


# -- conversation rebuild ---------------------------------------------------


def to_messages(entries: Sequence[Entry]) -> list[Message]:
    """Turn session entries back into provider messages.

    Tool calls are stored as their own entries so the session tree can show
    them individually; here they are folded back onto the assistant turn that
    produced them, which is the shape every provider expects.

    Two repairs happen on the way, both for histories a cancelled turn leaves
    behind. Every provider rejects a tool call with no result, so an
    unanswered call gets a synthetic one, and a result whose call is no longer
    on this branch is dropped rather than sent as an orphan.
    """
    out: list[Message] = []
    known_calls: dict[str, str] = {}  # call id -> tool name
    answered: set[str] = set()

    for entry in entries:
        if entry.type == MESSAGE:
            role = entry.role or "user"
            if role == "system":
                continue
            out.append(Message(role=role, text=entry.text, thinking=entry.data.get("thinking") or ""))
        elif entry.type in (BRANCH_SUMMARY, COMPACTION):
            # A summary stands in for history the model can no longer see, so
            # it has to reach the model as content rather than be skipped.
            summary = entry.text or entry.data.get("summary") or ""
            if summary:
                out.append(Message(role="user", text=f"[earlier conversation, summarised]\n{summary}"))
        elif entry.type == TOOL_CALL:
            call = ToolCall(
                id=entry.data.get("id") or entry.id,
                name=entry.data.get("tool") or "",
                args=entry.data.get("args") or {},
            )
            known_calls[call.id] = call.name
            if not out or out[-1].role != "assistant":
                out.append(Message(role="assistant"))
            out[-1].tool_calls.append(call)
        elif entry.type == TOOL_RESULT:
            cid = entry.data.get("id")
            if cid is not None and cid not in known_calls:
                continue  # orphan: the call it answers is not on this branch
            if cid is not None:
                answered.add(cid)
            out.append(Message(
                role="tool",
                text=entry.data.get("content") or "",
                tool_call_id=cid,
                name=entry.data.get("tool"),
            ))

    if len(answered) == len(known_calls):
        return out

    repaired: list[Message] = []
    for message in out:
        repaired.append(message)
        if message.role != "assistant":
            continue
        for call in message.tool_calls:
            if call.id in answered:
                continue
            repaired.append(Message(
                role="tool",
                text="tool call was interrupted before it produced a result",
                tool_call_id=call.id,
                name=call.name,
            ))
    return repaired


# -- configuration ----------------------------------------------------------


@dataclass(slots=True)
class AgentConfig:
    model: str = "mock"
    system: str | None = None
    max_steps: int = 24
    max_tokens: int = 8192
    temperature: float | None = None
    thinking_budget: int | None = None
    timeout: float = 300.0


@dataclass(slots=True)
class RunResult:
    text: str = ""
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    stop_reason: str = "stop"
    error: str | None = None
    invocations: list[Invocation] = field(default_factory=list)


class Agent:
    """Drives one session against one model, with one toolbox."""

    __slots__ = ("session", "runtime", "config", "_resolve", "_provider", "_meta", "_key")

    def __init__(
        self,
        session: Session,
        runtime: Runtime,
        config: AgentConfig | None = None,
        *,
        resolver: Callable[[str], tuple[Provider, ModelInfo]] = resolve,
        provider: Provider | None = None,
        api_key: str | None = None,
    ) -> None:
        self.session = session
        self.runtime = runtime
        self.config = config or AgentConfig()
        self._resolve = resolver
        self._provider = provider
        self._meta: ModelInfo | None = None
        self._key = api_key

    # -- plumbing ---------------------------------------------------------

    def _endpoint(self) -> tuple[Provider, ModelInfo]:
        if self._provider is not None and self._meta is not None:
            return self._provider, self._meta
        provider, meta = self._resolve(self.config.model)
        if self._provider is not None:
            provider = self._provider
        self._meta = meta
        return provider, meta

    def _request(self, tools: Sequence[ToolSpec]) -> Request:
        _, meta = self._endpoint()
        return Request(
            model=self.config.model,
            messages=to_messages(self.session.transcript()),
            system=self.config.system,
            tools=tools if meta.tools else (),
            max_tokens=min(self.config.max_tokens, meta.max_output),
            temperature=self.config.temperature,
            thinking_budget=self.config.thinking_budget if meta.thinking else None,
            timeout=self.config.timeout,
        )

    def _record(self, turn: Turn) -> None:
        """Persist an assistant turn: the message, then each call separately."""
        if turn.text or turn.thinking:
            self.session.say("assistant", turn.text, thinking=turn.thinking)
        for call in turn.tool_calls:
            self.session.append(TOOL_CALL, {
                "id": call.id,
                "tool": call.name,
                "args": call.args,
                "summary": ", ".join(f"{k}={v!r}"[:40] for k, v in list(call.args.items())[:2]),
            })

    # -- the loop ---------------------------------------------------------

    def run(self, prompt: str | None = None) -> Iterator[Event]:
        """Run until the model stops asking for tools.  Yields as it goes."""
        if prompt is not None:
            self.session.say("user", prompt)
        # A new prompt is new intent: clear any abort left over from the
        # previous turn, so one cancellation cannot brick the session.
        self.runtime.reset()

        provider, meta = self._endpoint()
        # An explicit api_key beats everything (tests and branch agents pass one).
        # Otherwise resolve a Credential, which also covers OAuth and refreshes
        # a token that is about to expire before the request goes out.
        cred = None if self._key is not None else load_credential(provider.name)
        key = self._key if self._key is not None else (cred.value if cred and cred.kind == "api_key" else None)
        specs = self.runtime.toolbox.specs()
        total = Usage()
        last_text = ""
        reason = "stop"
        steps = 0

        for step in range(self.config.max_steps):
            steps = step + 1
            yield StepStarted(step, self.config.model)

            builder = TurnBuilder()
            for event in provider.stream(self._request(specs), api_key=key, credential=cred):
                yield builder.feed(event)
            turn = builder.finish()

            total = total + turn.usage
            last_text = turn.text or last_text
            self._record(turn)

            if turn.error:
                reason = "error"
                yield Finished(reason, total, steps, last_text)
                return

            if not turn.tool_calls:
                reason = turn.stop_reason
                break

            for call in turn.tool_calls:
                yield ToolStarted(call)
            # Results are always surfaced, even when the batch was cancelled:
            # the work already happened and the session already recorded it.
            for invocation in self._dispatch(turn.tool_calls):
                yield ToolFinished(invocation)
            if self.runtime.aborted:
                reason = "cancelled"
                break
        else:
            reason = "max_steps"

        yield Finished(reason, total, steps, last_text)

    def _dispatch(self, calls: list[ToolCall]) -> list[Invocation]:
        """Execute a batch and persist every result."""
        invocations = self.runtime.execute_all(calls)
        for inv in invocations:
            self.session.append(TOOL_RESULT, {
                "id": inv.call.id,
                "tool": inv.call.name,
                "content": inv.result.content,
                "ok": inv.result.ok,
                "summary": inv.result.display or inv.result.content[:60],
                "duration": round(inv.result.duration, 4),
            })
        return invocations

    # -- convenience ------------------------------------------------------

    def send(self, prompt: str | None = None) -> RunResult:
        """Run to completion and summarise.  Used by tests and headless work."""
        result = RunResult()
        for event in self.run(prompt):
            if isinstance(event, ToolFinished):
                result.invocations.append(event.invocation)
            elif isinstance(event, StreamError):
                result.error = event.message
            elif isinstance(event, Finished):
                result.text, result.usage = event.text, event.usage
                result.steps, result.stop_reason = event.steps, event.reason
        return result
