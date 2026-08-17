"""Session entries.

A session is an append-only log of entries, each pointing at its parent.  That
single shape gives history, branching and replay for free: to fork a
conversation you write a new child of an old parent, and nothing is ever
rewritten or deleted.

Identifiers are ULID-shaped — 48 bits of millisecond timestamp then 80 bits of
randomness, Crockford base32.  Sorting ids lexicographically therefore sorts
them chronologically, which is what keeps sibling ordering stable across a
reload without trusting wall-clock fields.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Final

_B32: Final = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_last_ms = 0
_last_rand = 0


def new_id(ms: int | None = None) -> str:
    """A monotonic, lexicographically sortable 26-character identifier."""
    global _last_ms, _last_rand
    now = int(time.time() * 1000) if ms is None else ms
    if now == _last_ms:
        _last_rand += 1  # same millisecond: keep strictly increasing
    else:
        _last_ms = now
        _last_rand = int.from_bytes(os.urandom(10), "big")
    # Timestamp first, then randomness: the concatenation, not the whole
    # buffer, is what has to come out big-endian.
    ts, rand = now, _last_rand & ((1 << 80) - 1)
    head: list[str] = []
    for _ in range(10):
        head.append(_B32[ts & 31])
        ts >>= 5
    tail: list[str] = []
    for _ in range(16):
        tail.append(_B32[rand & 31])
        rand >>= 5
    head.reverse()
    tail.reverse()
    return "".join(head) + "".join(tail)


# -- entry types ------------------------------------------------------------

MESSAGE: Final = "message"  # role in {user, assistant, system}
TOOL_CALL: Final = "tool_call"
TOOL_RESULT: Final = "tool_result"
BRANCH_SUMMARY: Final = "branch_summary"
CHECKPOINT: Final = "checkpoint"
MODEL_CHANGE: Final = "model_change"
LABEL: Final = "label"
LEAF: Final = "leaf"
COMPACTION: Final = "compaction"
SNAPSHOT: Final = "snapshot"

#: Entries that live in the log but never appear in the tree: LABEL and LEAF
#: record *how the tree was navigated*, SNAPSHOT records what the workspace
#: looked like.  None of them is something that was said.
BOOKKEEPING: Final = frozenset({LABEL, LEAF, SNAPSHOT})

#: The bookkeeping `Session.compact` may collapse to its final state.  A
#: snapshot is a fact about the world, not a navigation step, so "the latest
#: one" cannot stand in for the rest and it must survive a rewrite.
COLLAPSIBLE: Final = frozenset({LABEL, LEAF})

#: Entries a model actually sees when the prompt is rebuilt.  COMPACTION is
#: here because its entire purpose is to stand in for the turns it replaced.
CONVERSATIONAL: Final = frozenset({MESSAGE, TOOL_CALL, TOOL_RESULT, BRANCH_SUMMARY, COMPACTION})


@dataclass(slots=True)
class Entry:
    id: str
    type: str
    parent: str | None = None
    ts: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"id": self.id, "type": self.type, "parent": self.parent, "ts": round(self.ts, 6), "data": self.data},
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "Entry":
        if not isinstance(obj, dict):
            raise ValueError("entry must be an object")
        eid, etype = obj.get("id"), obj.get("type")
        if not isinstance(eid, str) or not isinstance(etype, str):
            raise ValueError("entry needs a string id and type")
        parent = obj.get("parent")
        if parent is not None and not isinstance(parent, str):
            raise ValueError("parent must be a string or null")
        data = obj.get("data")
        return cls(
            id=eid,
            type=etype,
            parent=parent,
            ts=float(obj.get("ts") or 0.0),
            data=data if isinstance(data, dict) else {},
        )

    # -- convenience ------------------------------------------------------

    @property
    def role(self) -> str | None:
        return self.data.get("role")

    @property
    def text(self) -> str:
        return self.data.get("text") or ""

    def summary(self, width: int = 48) -> str:
        """One line describing this entry, for the tree view."""
        if self.type == MESSAGE:
            body = " ".join(self.text.split())
            head = f"{self.role or '?'}: {body}"
        elif self.type == TOOL_CALL:
            head = f"{self.data.get('tool', 'tool')}({self.data.get('summary', '')})"
        elif self.type == TOOL_RESULT:
            head = f"-> {self.data.get('summary', 'result')}"
        elif self.type == BRANCH_SUMMARY:
            head = f"summary: {' '.join(self.text.split())}"
        elif self.type == COMPACTION:
            n = len(self.data.get("replaced") or ())
            head = f"compacted {n}: {' '.join(self.text.split())}"
        elif self.type == CHECKPOINT:
            head = f"checkpoint {self.data.get('ref', '')}"
        elif self.type == MODEL_CHANGE:
            head = f"model -> {self.data.get('model', '?')}"
        else:
            head = self.type
        return head if len(head) <= width else head[: width - 1] + "\u2026"
