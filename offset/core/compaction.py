"""Context compaction: trading the oldest turns for a summary of them.

A long session eventually outgrows the model's context window, and the failure
is brutal: the provider rejects the whole request, so the session dies exactly
when it holds the most work.  Compaction replaces the old part of the active
path with one summary entry before that happens.

Three properties matter more than the summary's prose:

  * nothing is destroyed.  The summary is appended as a new root, the recent
    tail is re-appended after it, and the leaf ends up on that new chain.  The
    original entries stay in the file and stay reachable from the old leaf, so
    a compaction is a branch and can be walked back.  Entries are immutable, so
    re-appending the tail is the only way to put a summary in front of it —
    each copy records `compacted_from` rather than pretending to be a new turn.
  * the boundary never splits a tool call from its results.  Providers reject
    a result whose call is missing, and `offset.core.agent.to_messages` can
    only repair the other direction, so a boundary landing inside a tool block
    would turn one oversized request into a permanently broken one.
  * a compaction that could not summarise says so.  `compact` returns None for
    "nothing to do" and a `Report` carrying `error` when the summariser failed;
    it never appends a summary it did not get.

`summariser` is injected (prompt -> summary) so the decision logic is testable
without a model; `model_summariser` builds the real one.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Final, Sequence

from offset.core.agent import to_messages
from offset.core.entries import BRANCH_SUMMARY, COMPACTION, MESSAGE, TOOL_CALL, TOOL_RESULT, Entry
from offset.core.session import Session
from offset.providers.auth import load as load_credential
from offset.providers.base import Message, Request, TurnBuilder
from offset.providers.registry import info, resolve

#: English averages a little under four characters per token; four keeps the
#: estimate on the safe side for prose and close for code.
CHARS_PER_TOKEN: Final = 4

#: Per-message framing (role, delimiters) that providers bill for.  It only
#: matters once a transcript is hundreds of short tool results.
MESSAGE_OVERHEAD: Final = 4
CALL_OVERHEAD: Final = 8

DEFAULT_THRESHOLD: Final = 0.8
DEFAULT_KEEP_RECENT: Final = 6

#: Tool output is clipped in the summariser prompt: one file read can outweigh
#: every decision in the session, and the summariser has a window of its own.
CLIP: Final = 1500


# -- the estimate -----------------------------------------------------------


def _tokens(chars: int) -> int:
    return -(-chars // CHARS_PER_TOKEN)  # round up: never under-report


def _json(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Roughly what a request will cost, in tokens.

    Deliberately a heuristic and not a tokeniser: every provider tokenises
    differently, real tokenisers are large downloads that would still be wrong
    for the other providers, and the only decision this feeds is "summarise now
    or later" — an answer that survives being 20% out.  Characters are divided
    by `CHARS_PER_TOKEN`, rounded up, plus a small per-message and per-call
    envelope.  No term is ever negative and division rounds up, so the estimate
    is monotonic in content: appending a message can never lower it, which is
    what makes `needs_compaction` a one-way trigger rather than a flapping one.
    """
    total = 0
    for m in messages:
        total += MESSAGE_OVERHEAD + _tokens(len(m.text) + len(m.thinking))
        for call in m.tool_calls:
            body = call.raw if call.raw is not None else _json(call.args)
            total += CALL_OVERHEAD + _tokens(len(call.name) + len(body))
    return total


def over_budget(tokens: int, budget: int, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """Whether `tokens` has crossed `threshold` of `budget`.

    A budget of zero or less means "unknown, or the user switched compaction
    off", which is never a reason to throw history away.
    """
    if budget <= 0:
        return False
    return tokens >= budget * max(0.0, threshold)


def needs_compaction(messages: Sequence[Message], budget: int, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True while there is still room to compact *before* the failing request.

    The threshold is below 1.0 because the reply has to fit as well, and
    because compacting costs a model call that must happen while the context
    still has room for the summariser's own prompt.
    """
    return over_budget(estimate_tokens(messages), budget, threshold)


def budget_for(model: str, *, override: int | None = None) -> int:
    """The token budget to compact against.

    `session.compactAt` wins when set, because a user who names a number knows
    something the registry does not (a proxy with a smaller window, or a wish
    to keep requests cheap).  Zero means disabled, so it must not fall through
    to the model's context length.
    """
    if override is not None:
        return max(0, override)
    from offset.core import settings

    configured = settings.get("session.compactAt", 0)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = 0
    if configured:
        return configured
    try:
        return info(model).context
    except Exception:
        return 0


# -- choosing a boundary ----------------------------------------------------


@dataclass(slots=True, frozen=True)
class Plan:
    """What a summary may replace, and what has to survive verbatim."""

    boundary: int
    anchor: Entry | None
    replaced: tuple[Entry, ...]
    kept: tuple[Entry, ...]

    def __bool__(self) -> bool:
        return bool(self.replaced)


def _call_id(entry: Entry) -> str:
    """The provider-side call id, matching how `to_messages` pairs them."""
    return entry.data.get("id") or entry.id


def _pairing_safe(items: Sequence[Entry], boundary: int) -> int:
    """Drag a boundary back until it is not inside a tool block.

    Two repairs, applied until neither fires: a kept result whose call would be
    summarised away pulls the boundary back to that call, and a result that
    cannot be paired by id pulls it back over whatever it belongs to.  Both
    move backwards on purpose — keeping the call costs a few tokens, whereas
    skipping forward past the results would discard work the model just did.
    One pass is not enough: moving the boundary back puts more entries in the
    tail, which can expose the next unpaired result.
    """
    b = max(0, min(boundary, len(items)))
    while b > 0:
        wanted = {e.data.get("id") for e in items[b:] if e.type == TOOL_RESULT}
        wanted.discard(None)
        earliest = next(
            (i for i in range(b) if items[i].type == TOOL_CALL and _call_id(items[i]) in wanted),
            None,
        )
        if earliest is not None:
            b = earliest
            continue
        if b < len(items) and items[b].type == TOOL_RESULT:
            b -= 1
            continue
        break
    return b


def plan(entries: Sequence[Entry], keep_recent: int = DEFAULT_KEEP_RECENT) -> Plan:
    """Choose what a summary may replace.

    An exchange starts at a user message and runs to the next one, so only
    user-message starts are considered as boundaries — that alone keeps a
    summary out of the middle of an assistant turn.  Two things are never given
    up: the first user message, which states the task the whole session serves,
    and the last `keep_recent` exchanges, where the model needs verbatim detail
    about what it just did.  The boundary is then dragged back over any tool
    block it would have split.

    A prefix holding nothing but earlier summaries yields an empty plan:
    re-summarising a summary spends a model call and loses detail for no
    saving, which is what makes a second compaction with no new turns a no-op.
    """
    items = list(entries)
    users = [i for i, e in enumerate(items) if e.type == MESSAGE and e.role == "user"]
    anchor_at = users[0] if users else None
    anchor = items[anchor_at] if anchor_at is not None else None

    if keep_recent <= 0:
        boundary = len(items)
    elif len(users) > keep_recent:
        boundary = users[-keep_recent]
    else:
        boundary = anchor_at if anchor_at is not None else 0
    boundary = _pairing_safe(items, boundary)

    replaced = tuple(e for i, e in enumerate(items[:boundary]) if i != anchor_at)
    if not any(e.type != COMPACTION for e in replaced):
        return Plan(boundary=0, anchor=anchor, replaced=(), kept=tuple(items))

    tail = tuple(items[boundary:])
    keep_anchor = anchor is not None and anchor_at is not None and anchor_at < boundary
    return Plan(
        boundary=boundary,
        anchor=anchor,
        replaced=replaced,
        kept=((anchor,) if keep_anchor else ()) + tail,
    )


# -- the summariser prompt --------------------------------------------------


SYSTEM: Final = (
    "You compress part of a coding session into a briefing that will replace it. "
    "You report only what the transcript shows."
)

PROMPT: Final = """\
Summarise this part of a coding session so the summary can stand in for it.
Use these headings, in this order, with no preamble:
  GOAL   what the user is trying to achieve, in their words
  DONE   changes actually made, with file paths
  FACTS  what was learned: errors seen, versions, commands that work
  OPEN   what is unfinished, and the next step
Keep names, paths, numbers and error text. Drop pleasantries and tool output
that no longer matters. Never invent anything the transcript does not show.

--- transcript ---
"""


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def outline(entries: Sequence[Entry], clip: int = CLIP) -> str:
    """The entries being replaced, as plain text for a summariser."""
    lines: list[str] = []
    for e in entries:
        if e.type == MESSAGE:
            lines.append(f"{e.role or '?'}: {_clip(e.text, clip)}")
        elif e.type == TOOL_CALL:
            lines.append(f"tool {e.data.get('tool') or '?'}({_clip(_json(e.data.get('args') or {}), 300)})")
        elif e.type == TOOL_RESULT:
            head = "failed" if e.data.get("ok") is False else "result"
            body = e.data.get("content") or e.data.get("summary") or ""
            lines.append(f"  {head}: {_clip(str(body), clip)}")
        elif e.type in (BRANCH_SUMMARY, COMPACTION):
            lines.append(f"earlier summary: {_clip(e.text, clip * 2)}")
        else:
            lines.append(f"({e.type})")
    return "\n".join(lines)


def build_prompt(entries: Sequence[Entry]) -> str:
    return PROMPT + outline(entries)


# -- doing it ---------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Report:
    """What a compaction did, or why it did nothing."""

    entry: Entry | None
    replaced: int
    kept: int
    before: int
    after: int
    previous_leaf: str | None
    error: str | None = None

    @property
    def done(self) -> bool:
        return self.entry is not None

    @property
    def saved(self) -> int:
        return max(0, self.before - self.after)


def _note(p: Plan, previous_leaf: str | None) -> str:
    kinds = Counter(e.type for e in p.replaced)
    parts = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
    where = f"reachable from leaf {previous_leaf}" if previous_leaf else "still in the log"
    return f"stands in for {len(p.replaced)} entries ({parts}); the originals are {where}"


def compact(
    session: Session,
    summariser: Callable[[str], str],
    budget: int = 0,
    keep_recent: int = DEFAULT_KEEP_RECENT,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    force: bool = False,
) -> Report | None:
    """Fold the old part of the active path into a single summary entry.

    Returns None when there is nothing to do — under budget, or a prefix not
    worth summarising — so the caller can say that instead of implying work
    happened.  A summariser that raises or returns nothing yields a Report with
    `error` set and leaves the session exactly as it was.
    """
    previous = session.leaf
    entries = session.transcript()
    before = estimate_tokens(to_messages(entries))
    if not force and not over_budget(before, budget, threshold):
        return None

    p = plan(entries, keep_recent)
    if not p:
        return None

    try:
        summary = (summariser(build_prompt(p.replaced)) or "").strip()
    except Exception as exc:  # a provider failure must never cost the session
        return Report(None, len(p.replaced), len(p.kept), before, before, previous,
                      error=f"summariser failed: {exc}")
    if not summary:
        return Report(None, len(p.replaced), len(p.kept), before, before, previous,
                      error="the summariser returned nothing, so history is unchanged")

    # A new root: parenting the summary anywhere in the old path would keep
    # that path in the ancestry and save nothing.
    entry = session.append(
        COMPACTION,
        {
            "text": summary,
            "note": _note(p, previous),
            "replaced": [e.id for e in p.replaced],
            "replaced_leaf": previous,
            "kept": len(p.kept),
            "tokens_before": before,
        },
        parent=None,
    )
    for original in p.kept:
        session.append(original.type, {**original.data, "compacted_from": original.id})

    after = estimate_tokens(to_messages(session.transcript()))
    return Report(entry, len(p.replaced), len(p.kept), before, after, previous)


def model_summariser(
    model: str,
    *,
    api_key: str | None = None,
    resolver: Callable[[str], Any] = resolve,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> Callable[[str], str]:
    """A summariser that asks a model directly: no tools, no session, one turn.

    Kept out of `Agent` deliberately — a compaction must not append its own
    request to the session it is compacting.
    """

    def summarise(prompt: str) -> str:
        provider, meta = resolver(model)
        cred = None if api_key is not None else load_credential(provider.name)
        key = api_key if api_key is not None else (cred.value if cred and cred.kind == "api_key" else None)
        request = Request(
            model=model,
            messages=[Message(role="user", text=prompt)],
            system=SYSTEM,
            max_tokens=min(max_tokens, meta.max_output),
            timeout=timeout,
        )
        builder = TurnBuilder()
        for event in provider.stream(request, api_key=key, credential=cred):
            builder.feed(event)
        turn = builder.finish()
        if turn.error:
            # `compact` turns this into a Report the user can act on.
            raise RuntimeError(turn.error)
        return turn.text

    return summarise
