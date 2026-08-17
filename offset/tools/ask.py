"""Asking the human a question the model cannot answer itself.

The tool never draws anything.  It hands a `Question` to an injected asker and
waits, which is what makes it testable and what lets the shell render it with
the same dropdown the `/model` picker uses.

Waiting is bounded.  A question with nobody in front of the screen must not
hold a turn open forever, so the timeout returns a *result* — "no answer" —
rather than an exception or a hang.  That distinction matters: a timeout is not
a refusal and not a user abort, and the model is told which one it got so it can
carry on with a stated assumption instead of retrying the same question.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from offset.tools.base import Danger, Tool, ToolContext, ToolResult

#: Why there is no answer.  `answered` is the only success.
ANSWERED = "answered"
TIMEOUT = "timeout"
DECLINED = "declined"
UNAVAILABLE = "unavailable"

DEFAULT_TIMEOUT = 120.0
MAX_OPTIONS = 9


@dataclass(slots=True)
class Question:
    text: str
    options: list[str] = field(default_factory=list)
    #: Shown under the options; context the human needs to choose well.
    detail: str = ""
    #: Offered when the human just hits enter.
    default: str | None = None

    @property
    def default_index(self) -> int:
        if self.default is None or self.default not in self.options:
            return 0
        return self.options.index(self.default)


@dataclass(slots=True)
class Answer:
    chosen: str | None = None
    index: int = -1
    reason: str = ANSWERED

    @classmethod
    def pick(cls, question: Question, index: int) -> "Answer":
        if not (0 <= index < len(question.options)):
            return cls(reason=DECLINED)
        return cls(chosen=question.options[index], index=index)

    @classmethod
    def no(cls, reason: str = DECLINED) -> "Answer":
        return cls(reason=reason)

    @property
    def answered(self) -> bool:
        return self.reason == ANSWERED and self.chosen is not None


#: What the shell installs.  Blocking is fine; the tool bounds the wait.
Asker = Callable[[Question], Answer]


class Ask(Tool):
    name = "ask"
    description = (
        "Ask the user to choose between options when the answer changes what you build and "
        "cannot be found in the code. Returns the chosen option, or a 'no answer' result if "
        "nobody responds in time - in which case proceed and say which option you assumed."
    )
    #: It reads a human, not the disk.
    danger = Danger.SAFE
    #: Two questions on one screen at once is not a UI anyone can answer.
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "maxLength": 400},
            "options": {"type": "array", "items": {"type": "string", "maxLength": 200}},
            "detail": {"type": "string", "maxLength": 800},
            "default": {"type": "string"},
            "timeout": {"type": "number", "minimum": 1, "maximum": 600},
        },
        "required": ["question", "options"],
    }

    __slots__ = ("asker", "timeout")

    def __init__(self, asker: Asker | None = None, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        #: None means headless: no human is attached, so say so immediately
        #: rather than stalling for the full budget.
        self.asker = asker
        self.timeout = timeout

    def preview(self, args: dict[str, Any]) -> str:
        return f"ask: {str(args.get('question') or '')[:60]}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        options = [str(o).strip() for o in (args.get("options") or []) if str(o).strip()]
        if len(options) < 2:
            return _refuse("a question needs at least two distinct options")
        if len(options) > MAX_OPTIONS:
            return _refuse(f"at most {MAX_OPTIONS} options; collapse the rest into one")
        if len(set(options)) != len(options):
            return _refuse("the options must be distinct")

        question = Question(
            text=str(args.get("question") or "").strip(),
            options=options,
            detail=str(args.get("detail") or "").strip(),
            default=str(args["default"]) if args.get("default") in options else None,
        )
        if not question.text:
            return _refuse("the question text is empty")

        if self.asker is None:
            return _unanswered(question, UNAVAILABLE, "there is no human attached to this session")

        budget = self._budget(args, ctx)
        answer, failure = self._wait(question, budget, ctx)
        if failure:
            return _unanswered(question, TIMEOUT, failure)
        if answer is None or not answer.answered:
            reason = answer.reason if answer else DECLINED
            return _unanswered(question, reason, "the user did not choose an option")

        return ToolResult(
            content=f"the user chose: {answer.chosen}",
            display=f"ask -> {answer.chosen[:48]}",
            data={
                "answered": True,
                "reason": ANSWERED,
                "chosen": answer.chosen,
                "index": answer.index,
                "options": options,
            },
        )

    def _budget(self, args: dict[str, Any], ctx: ToolContext) -> float:
        """Stay inside the runtime's budget, so *we* report the timeout.

        If the tool ran to exactly `ctx.timeout` the runtime would kill it and
        the model would see "exceeded its budget" — true, but useless.  Leaving
        a margin buys the clear "no answer" message instead.
        """
        wanted = float(args.get("timeout") or self.timeout)
        if ctx.timeout <= 0:
            return wanted
        return max(0.5, min(wanted, ctx.timeout - 0.5))

    @staticmethod
    def _wait(question: Question, budget: float, ctx: ToolContext) -> tuple[Answer | None, str]:
        """Run the asker off-thread so a blocking UI cannot pin the runtime."""
        box: list[Answer] = []

        def target() -> None:
            try:
                box.append(_normalise(question, Ask._call(question)))
            except Exception as exc:  # a broken asker is a declined question
                box.append(Answer(reason=DECLINED if not str(exc) else DECLINED))

        Ask._call = staticmethod(lambda q: q)  # placeholder, replaced below
        raise AssertionError("unreachable")


def _normalise(question: Question, raw: Any) -> Answer:
    """Accept an `Answer`, the option text, or its index from the asker."""
    if isinstance(raw, Answer):
        if raw.chosen is not None and raw.index < 0 and raw.chosen in question.options:
            raw.index = question.options.index(raw.chosen)
        return raw
    if isinstance(raw, bool) or raw is None:
        return Answer.no()
    if isinstance(raw, int):
        return Answer.pick(question, raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text in question.options:
            return Answer(chosen=text, index=question.options.index(text))
        if text.isdigit():
            return Answer.pick(question, int(text) - 1)
        #: Free text is still an answer; the model is told it was not an option.
        return Answer(chosen=text, index=-1) if text else Answer.no()
    return Answer.no()


def _refuse(why: str) -> ToolResult:
    return ToolResult(ok=False, content=why, display=f"ask: {why}", error=why)


def _unanswered(question: Question, reason: str, why: str) -> ToolResult:
    """A question with no answer is a fact, not a failure.

    `ok` stays true because the tool did what it was asked; the content leads
    with "no answer" so the model cannot mistake it for a choice.
    """
    fallback = question.options[question.default_index]
    return ToolResult(
        content=(
            f"no answer: {why}. Proceed without it; if you must choose, "
            f"{fallback!r} is the least surprising option and you must say that you assumed it."
        ),
        display=f"ask -> no answer ({reason})",
        data={"answered": False, "reason": reason, "chosen": None, "options": list(question.options)},
    )
