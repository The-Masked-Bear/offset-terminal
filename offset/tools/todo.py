"""The task list the model keeps for itself.

Two invariants are the whole point of this module, and they exist because a
model left to free-form its own plan will either forget half of it or claim to
be working on four things at once:

  * exactly one task is `in_progress` whenever any task is still startable, so
    "what are you doing right now" always has one answer;
  * finishing a task immediately promotes the earliest still-open one, so the
    list never stalls in a state where nothing is being worked on.

Order is the order the tasks were written and never changes; a reload rebuilds
the same list, because the plan is on disk rather than in the conversation and
must survive a compaction or a restart.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from offset.tools.base import Danger, Tool, ToolContext, ToolResult

PENDING = "pending"
IN_PROGRESS = "in_progress"
DONE = "done"
DROPPED = "dropped"
BLOCKED = "blocked"

STATUSES: tuple[str, ...] = (PENDING, IN_PROGRESS, DONE, DROPPED, BLOCKED)

#: A task in one of these states can be picked up next.  `blocked` cannot:
#: promoting a blocked task would be a lie about what is runnable.
OPEN: frozenset[str] = frozenset({PENDING, IN_PROGRESS})

MARKS: dict[str, str] = {
    PENDING: " ",
    IN_PROGRESS: "~",
    DONE: "x",
    DROPPED: "-",
    BLOCKED: "!",
}

#: Where the list lives, relative to the workspace.  Same `.offset` directory
#: the branch machinery and the session store already use.
STORE = Path(".offset") / "todo.json"

MAX_TASKS = 200

#: Subagents share the workspace with their parent, so two runtimes really can
#: write this file at once.  One process-wide lock is cheaper than a merge.
_LOCK = threading.Lock()


@dataclass(slots=True)
class Task:
    id: str
    text: str
    phase: str = ""
    status: str = PENDING
    note: str = ""

    def line(self) -> str:
        mark = MARKS.get(self.status, "?")
        tail = f"  ({self.note})" if self.note else ""
        return f"[{mark}] {self.id}  {self.text}{tail}"


class TodoList:
    """The persisted list plus the invariants.  Every mutation re-enforces."""

    __slots__ = ("_counter", "path", "tasks")

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.tasks: list[Task] = []
        self._counter = 0

    # -- persistence ------------------------------------------------------

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "TodoList":
        """Read the list, treating a missing or corrupt file as an empty one.

        A damaged plan must not stop the turn: the model can rebuild it with
        `init`, which is strictly better than an exception it cannot act on.
        """
        todo = cls(path)
        try:
            raw = json.loads(todo.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return todo
        if not isinstance(raw, dict):
            return todo
        for obj in raw.get("tasks") or ():
            if not isinstance(obj, dict):
                continue
            ident, text = str(obj.get("id") or ""), str(obj.get("text") or "")
            if not ident or not text:
                continue
            status = str(obj.get("status") or PENDING)
            todo.tasks.append(Task(
                id=ident,
                text=text,
                phase=str(obj.get("phase") or ""),
                status=status if status in STATUSES else PENDING,
                note=str(obj.get("note") or ""),
            ))
        counter = raw.get("counter")
        todo._counter = counter if isinstance(counter, int) else len(todo.tasks)
        todo._enforce()
        return todo

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(
            {"version": 1, "counter": self._counter, "tasks": [asdict(t) for t in self.tasks]},
            indent=1,
        )
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(blob + "\n", encoding="utf-8")
        os.replace(tmp, self.path)  # a half-written plan is worse than none

    # -- invariants -------------------------------------------------------

    def _enforce(self) -> None:
        """Collapse to one `in_progress`, promoting the earliest open task."""
        running = [t for t in self.tasks if t.status == IN_PROGRESS]
        for extra in running[1:]:
            extra.status = PENDING  # first writer wins; the rest go back
        if running:
            return
        for task in self.tasks:
            if task.status == PENDING:
                task.status = IN_PROGRESS
                return

    @property
    def current(self) -> Task | None:
        for task in self.tasks:
            if task.status == IN_PROGRESS:
                return task
        return None

    @property
    def open_count(self) -> int:
        return sum(1 for t in self.tasks if t.status in OPEN)

    # -- lookup -----------------------------------------------------------

    def find(self, ref: str) -> Task | None:
        """Resolve a task reference.

        Models mangle ids, so an exact id, a 1-based position, and an
        unambiguous substring of the text all work.  Ambiguity resolves to
        nothing rather than to a guess.
        """
        ref = (ref or "").strip()
        if not ref:
            return None
        for task in self.tasks:
            if task.id == ref:
                return task
        if ref.isdigit():
            index = int(ref) - 1
            if 0 <= index < len(self.tasks):
                return self.tasks[index]
        lowered = ref.lower()
        hits = [t for t in self.tasks if lowered in t.text.lower()]
        return hits[0] if len(hits) == 1 else None

    # -- mutation ---------------------------------------------------------

    def _add(self, text: str, phase: str) -> Task:
        self._counter += 1
        task = Task(id=f"t{self._counter}", text=text, phase=phase)
        self.tasks.append(task)
        return task

    def init(self, items: Sequence[Any]) -> list[str]:
        """Replace the plan.  Returns problems; an empty plan is refused."""
        parsed, problems = _parse_items(items)
        if problems:
            return problems
        self.tasks.clear()
        self._counter = 0
        for text, phase in parsed:
            self._add(text, phase)
        self._enforce()
        return []

    def extend(self, items: Sequence[Any]) -> list[str]:
        parsed, problems = _parse_items(items)
        if problems:
            return problems
        if len(self.tasks) + len(parsed) > MAX_TASKS:
            return [f"that would exceed the {MAX_TASKS}-task limit; finish or drop some first"]
        for text, phase in parsed:
            self._add(text, phase)
        self._enforce()
        return []

    def start(self, ref: str) -> str:
        task = self.find(ref)
        if task is None:
            return f"no task matches {ref!r}"
        if task.status in (DONE, DROPPED):
            return f"{task.id} is already {task.status}; append a new task instead"
        for other in self.tasks:
            if other is not task and other.status == IN_PROGRESS:
                other.status = PENDING
        task.status, task.note = IN_PROGRESS, ""  # starting clears a block
        self._enforce()
        return ""

    def finish(self, ref: str) -> str:
        task = self.find(ref)
        if task is None:
            return f"no task matches {ref!r}"
        if task.status == DONE:
            return f"{task.id} is already done"
        task.status, task.note = DONE, ""
        self._enforce()
        return ""

    def drop(self, ref: str) -> str:
        task = self.find(ref)
        if task is None:
            return f"no task matches {ref!r}"
        task.status = DROPPED
        self._enforce()
        return ""

    def block(self, ref: str, why: str) -> str:
        task = self.find(ref)
        if task is None:
            return f"no task matches {ref!r}"
        if not why.strip():
            return "blocking a task needs a reason, so it can be unblocked later"
        task.status, task.note = BLOCKED, why.strip()
        self._enforce()
        return ""

    def unblock(self, ref: str) -> str:
        task = self.find(ref)
        if task is None:
            return f"no task matches {ref!r}"
        if task.status != BLOCKED:
            return f"{task.id} is not blocked"
        task.status, task.note = PENDING, ""
        self._enforce()
        return ""

    # -- rendering --------------------------------------------------------

    def render(self) -> str:
        """Plain text, grouped by phase in first-appearance order."""
        if not self.tasks:
            return "the task list is empty; call todo with op=init to plan the work"
        lines: list[str] = []
        phase: str | None = None
        for task in self.tasks:
            if task.phase != phase:
                phase = task.phase
                if phase:
                    lines.append(f"{phase}:")
            lines.append(("  " if task.phase else "") + task.line())
        current = self.current
        done = sum(1 for t in self.tasks if t.status == DONE)
        lines.append("")
        lines.append(f"{done}/{len(self.tasks)} done" + (f"; now: {current.text}" if current else "; nothing open"))
        return "\n".join(lines)

    def snapshot(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self.tasks]


def _parse_items(items: Sequence[Any]) -> tuple[list[tuple[str, str]], list[str]]:
    """Accept `["do a thing"]` or `[{"text": ..., "phase": ...}]`."""
    if not isinstance(items, (list, tuple)) or not items:
        return [], ["tasks must be a non-empty array of strings or {text, phase} objects"]
    out: list[tuple[str, str]] = []
    problems: list[str] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            text, phase = item.strip(), ""
        elif isinstance(item, dict):
            text, phase = str(item.get("text") or "").strip(), str(item.get("phase") or "").strip()
        else:
            problems.append(f"tasks[{i}]: expected a string or an object")
            continue
        if not text:
            problems.append(f"tasks[{i}]: empty task text")
            continue
        out.append((text, phase))
    if len(out) > MAX_TASKS:
        problems.append(f"at most {MAX_TASKS} tasks")
    return out, problems


def store_for(ctx: ToolContext, override: Path | None = None) -> Path:
    """Where the list lives.

    An override that names a directory gets the store placed inside it. Without
    this, being handed a directory produced `<dir>.json.tmp` and an
    `os.replace` onto the directory itself - an error at the far end of a
    wiring mistake rather than at the point of it.
    """
    if override is None:
        return ctx.cwd / STORE
    override = Path(override)
    return override / STORE if override.is_dir() else override


class Todo(Tool):
    name = "todo"
    description = (
        "Keep the plan for this piece of work: op=init replaces the list, append adds to it, "
        "start/done/drop/block/unblock move one task, view shows it. Exactly one task is in "
        "progress at a time and finishing one promotes the next automatically."
    )
    #: Writes a file inside the workspace, nothing more.
    danger = Danger.WRITE
    #: One shared file: concurrent calls would interleave load-modify-save.
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["init", "append", "start", "done", "drop", "block", "unblock", "view"]},
            "tasks": {"type": "array", "items": {"type": "string"},
                  "description": "task text, one per entry"},
            "id": {"type": "string"},
            "note": {"type": "string"},
        },
        "required": ["op"],
    }

    __slots__ = ("_path",)

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        #: Tests and headless runs pin the file; normally it follows the cwd.
        self._path = Path(path) if path is not None else None

    def preview(self, args: dict[str, Any]) -> str:
        op = args.get("op", "?")
        if op in ("init", "append"):
            return f"todo {op} ({len(args.get('tasks') or ())} tasks)"
        return f"todo {op} {args.get('id', '')}".strip()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        op = str(args.get("op") or "")
        with _LOCK:
            todo = TodoList.load(store_for(ctx, self._path))
            problem = self._apply(op, args, todo)
            if problem:
                return ToolResult(ok=False, content=problem, display=f"todo {op}: refused", error=problem)
            if op != "view":
                todo.save()
            current = todo.current
            return ToolResult(
                content=todo.render(),
                display=f"todo {op} -> {current.text[:48] if current else 'nothing open'}",
                data={"tasks": todo.snapshot(), "in_progress": current.id if current else None},
            )

    @staticmethod
    def _apply(op: str, args: dict[str, Any], todo: TodoList) -> str:
        ref = str(args.get("id") or "")
        if op == "view":
            return ""
        if op == "init":
            return "; ".join(todo.init(args.get("tasks") or []))
        if op == "append":
            return "; ".join(todo.extend(args.get("tasks") or []))
        if op in ("start", "done", "drop", "block", "unblock") and not ref:
            return f"op={op} needs the id of a task"
        if op == "start":
            return todo.start(ref)
        if op == "done":
            return todo.finish(ref)
        if op == "drop":
            return todo.drop(ref)
        if op == "block":
            return todo.block(ref, str(args.get("note") or ""))
        if op == "unblock":
            return todo.unblock(ref)
        return f"unknown op {op!r}"


def todo_tools(path: str | os.PathLike[str] | None = None) -> list[Tool]:
    return [Todo(path)]


def summary(items: Iterable[dict[str, Any]]) -> str:
    """One line for a status bar: `3/7 done - writing the parser`."""
    tasks = list(items)
    if not tasks:
        return ""
    done = sum(1 for t in tasks if t.get("status") == DONE)
    running = next((t for t in tasks if t.get("status") == IN_PROGRESS), None)
    tail = f" - {running['text']}" if running else ""
    return f"{done}/{len(tasks)} done{tail}"
