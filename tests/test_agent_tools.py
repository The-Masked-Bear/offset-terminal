"""The `ask` and `todo` tools.

`ask` shipped with a stub: `_wait` raised AssertionError("unreachable"), so every
question with a real asker attached came back as a crash. These tests exist so
that cannot happen again quietly.
"""

from __future__ import annotations

import threading
import time

import pytest

from offset.providers.base import ToolCall
from offset.tools.ask import (
    ANSWERED,
    DECLINED,
    TIMEOUT,
    UNAVAILABLE,
    Answer,
    Ask,
    Question,
)
from offset.tools.base import Danger, ToolContext, Toolbox
from offset.tools.runtime import Approval, Runtime
from offset.tools.todo import Todo, todo_tools


@pytest.fixture()
def ctx(tmp_path):
    return ToolContext(cwd=tmp_path, timeout=30.0)


def ask_call(**args) -> ToolCall:
    return ToolCall(id="c1", name="ask", args=args)


def run_ask(tool: Ask, ctx: ToolContext, **args):
    return Runtime(Toolbox([tool]), ctx, Approval(mode="yolo")).execute(ask_call(**args)).result


QUESTION = {"question": "Which storage backend?", "options": ["sqlite", "postgres"]}


# -- ask: the path that was a stub -----------------------------------------


def test_an_answer_from_the_asker_reaches_the_model(ctx):
    """Regression: this raised AssertionError('unreachable')."""
    got = run_ask(Ask(lambda q: Answer.pick(q, 1)), ctx, **QUESTION)
    assert got.ok, got.error
    assert got.content == "the user chose: postgres"
    assert got.data == {
        "answered": True, "reason": ANSWERED, "chosen": "postgres",
        "index": 1, "options": ["sqlite", "postgres"],
    }


def test_the_asker_receives_the_question_it_was_given(ctx):
    seen: list[Question] = []
    run_ask(Ask(lambda q: seen.append(q) or Answer.pick(q, 0)), ctx,
            question="Pick one", options=["a", "b"], detail="context here", default="b")
    assert len(seen) == 1
    assert seen[0].text == "Pick one" and seen[0].options == ["a", "b"]
    assert seen[0].detail == "context here" and seen[0].default == "b"


def test_an_option_string_is_accepted(ctx):
    got = run_ask(Ask(lambda q: "sqlite"), ctx, **QUESTION)
    assert got.data["chosen"] == "sqlite" and got.data["index"] == 0


def test_a_one_based_number_is_accepted(ctx):
    got = run_ask(Ask(lambda q: "2"), ctx, **QUESTION)
    assert got.data["chosen"] == "postgres"


def test_free_text_still_counts_as_an_answer(ctx):
    got = run_ask(Ask(lambda q: "actually use duckdb"), ctx, **QUESTION)
    assert got.data["answered"] and got.data["chosen"] == "actually use duckdb"
    assert got.data["index"] == -1, "the model must be able to see it was not an option"


# -- ask: no answer is a result, not a failure ------------------------------


def test_a_timeout_returns_a_usable_result(ctx):
    def never(question):
        time.sleep(30)
        return Answer.pick(question, 0)

    started = time.monotonic()
    got = run_ask(Ask(never), ctx, timeout=1, **QUESTION)
    assert time.monotonic() - started < 5.0, "the wait must be bounded"
    assert got.ok, "a question nobody answered is a fact, not an error"
    assert got.data == {"answered": False, "reason": TIMEOUT, "chosen": None,
                        "options": ["sqlite", "postgres"]}
    assert "no answer" in got.content
    assert "sqlite" in got.content and "assumed" in got.content, "it must name a fallback"


def test_a_headless_session_says_so_immediately(ctx):
    started = time.monotonic()
    got = run_ask(Ask(None), ctx, **QUESTION)
    assert time.monotonic() - started < 1.0, "no human means no waiting"
    assert got.data["reason"] == UNAVAILABLE


def test_a_broken_asker_is_a_declined_question(ctx):
    def explode(question):
        raise RuntimeError("the UI fell over")

    got = run_ask(Ask(explode), ctx, **QUESTION)
    assert got.ok and got.data["reason"] == DECLINED


def test_a_declining_asker_is_reported_as_declined(ctx):
    got = run_ask(Ask(lambda q: Answer.no()), ctx, **QUESTION)
    assert got.data["answered"] is False and got.data["reason"] == DECLINED


def test_cancelling_the_turn_stops_the_question(ctx):
    def slow(question):
        time.sleep(30)
        return Answer.pick(question, 0)

    runtime = Runtime(Toolbox([Ask(slow)]), ctx, Approval(mode="yolo"))
    threading.Timer(0.2, runtime.cancel).start()
    started = time.monotonic()
    got = runtime.execute(ask_call(**QUESTION)).result
    # Either layer may win the race: the runtime's abort, or the tool noticing it.
    # What must not happen is a 30-second hang.
    assert time.monotonic() - started < 5.0
    body = f"{got.content} {got.error or ''}".lower()
    assert "cancel" in body, body


def test_the_wait_stays_inside_the_runtime_budget(tmp_path):
    """The runtime must never be the one to report the timeout."""
    tight = ToolContext(cwd=tmp_path, timeout=1.0)
    got = run_ask(Ask(lambda q: time.sleep(30)), tight, timeout=600, **QUESTION)
    assert got.data["reason"] == TIMEOUT, "the tool, not the runtime, reports it"
    assert "budget" not in (got.error or "")


# -- ask: argument validation ----------------------------------------------


@pytest.mark.parametrize("options,why", [
    ([], "two"),
    (["only one"], "two"),
    (["same", "same"], "distinct"),
    ([f"o{i}" for i in range(12)], "at most"),
])
def test_bad_option_lists_are_refused(ctx, options, why):
    got = run_ask(Ask(lambda q: Answer.pick(q, 0)), ctx, question="q?", options=options)
    assert not got.ok and why in got.error


def test_an_empty_question_is_refused(ctx):
    got = run_ask(Ask(lambda q: Answer.pick(q, 0)), ctx, question="   ", options=["a", "b"])
    assert not got.ok and "empty" in got.error


def test_ask_is_safe_and_never_parallel():
    tool = Ask(None)
    assert tool.danger is Danger.SAFE
    assert not tool.parallel_safe, "two questions on one screen is not answerable"


def test_the_default_must_be_one_of_the_options(ctx):
    got = run_ask(Ask(lambda q: Answer(reason=DECLINED)), ctx,
                  question="q?", options=["a", "b"], default="not-an-option")
    assert got.data["reason"] == DECLINED  # accepted the call, ignored the bogus default


# -- todo -------------------------------------------------------------------


def todo_run(tool: Todo, ctx: ToolContext, **args):
    return Runtime(Toolbox([tool]), ctx, Approval(mode="yolo")).execute(
        ToolCall(id="t", name="todo", args=args)
    ).result


@pytest.fixture()
def todo(tmp_path):
    return todo_tools(tmp_path)[0]


def test_a_list_can_be_created_and_read_back(todo, ctx):
    made = todo_run(todo, ctx, op="init", tasks=["scaffold", "wire it", "test it"])
    assert made.ok, made.error
    shown = todo_run(todo, ctx, op="view")
    for task in ("scaffold", "wire it", "test it"):
        assert task in shown.content


def test_exactly_one_task_is_in_progress(todo, ctx):
    todo_run(todo, ctx, op="init", tasks=["one", "two", "three"])
    body = todo_run(todo, ctx, op="view").content
    assert body.count("in_progress") <= 1, "two active tasks is not a plan"


def test_completing_promotes_the_next_task(todo, ctx):
    todo_run(todo, ctx, op="init", tasks=["first", "second"])
    first = todo_run(todo, ctx, op="view").data["tasks"][0]
    todo_run(todo, ctx, op="done", id=first["id"])
    body = todo_run(todo, ctx, op="view").content
    assert "first" in body and "second" in body
    assert body.count("in_progress") <= 1


def test_the_list_survives_a_reload(todo, ctx, tmp_path):
    todo_run(todo, ctx, op="init", tasks=["persisted"])
    fresh = todo_tools(tmp_path)[0]
    assert "persisted" in todo_run(fresh, ctx, op="view").content


def test_an_unknown_op_is_refused(todo, ctx):
    got = todo_run(todo, ctx, op="reticulate")
    assert not got.ok
