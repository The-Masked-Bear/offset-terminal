"""The multi-model features, actually reachable.

`ShellState.ensemble` used to be left at `None` for the whole life of the
program, so every feature built on it collapsed onto a single model: /spec ran
every branch on the same one, and nothing exposed vote, council, race or relay
at all. These tests exist so that cannot quietly happen again.
"""

from __future__ import annotations


import pytest

from offset.core.multimodel import Ensemble, Seat, seat_roster
from offset.shell.app import build_state
from offset.shell.commands import COMMANDS, STRATEGIES, dispatch


@pytest.fixture()
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    return build_state(workspace, model="mock")


# -- the roster exists at all -------------------------------------------------


def test_a_fresh_shell_has_a_roster(state):
    assert state.ensemble is not None, "startup must seat a roster, not leave it None"
    assert list(state.ensemble), "a roster with no seats is the same bug"


def test_the_active_model_leads_the_roster(state):
    first = list(state.ensemble)[0]
    assert first.model == state.model, "the model the user chose must lead"


def test_the_roster_only_holds_models_we_could_reach():
    roster = seat_roster("mock")
    assert "mock" in [seat.model for seat in roster]
    assert len(list(roster)) <= 4, "a council of everything is nobody's default"


def test_spec_spreads_across_the_roster(state):
    """The regression: /spec read state.ensemble, which was always None."""
    dispatch(state, "/seats mock claude-opus-4-20250514")
    out = dispatch(state, "/spec 2 make it faster")
    body = "\n".join(out.lines)
    assert "claude-opus-4-20250514" in body or "mock" in body
    assert out.job is not None, "a real run has to happen off-thread"


# -- staffing roles ----------------------------------------------------------


def test_every_role_gets_a_seat():
    """An unfilled role used to be skipped, so relay produced nothing."""
    roster = Ensemble([Seat("mock", "test"), Seat("other", "bulk")])
    staffed = roster.staff(("planner", "implementer", "critic"))
    assert len(staffed) == 3
    assert all(seat is not None for _, seat in staffed)


def test_a_role_holder_is_preferred_for_its_own_role():
    roster = Ensemble([Seat("a", "planner"), Seat("b", "implementer"), Seat("c", "critic")])
    assert [s.model for _, s in roster.staff(("planner", "implementer", "critic"))] == ["a", "b", "c"]


def test_an_idle_seat_beats_a_seat_that_already_spoke():
    """A critic that already answered as implementer would review its own work."""
    roster = Ensemble([Seat("a", "test"), Seat("b", "critic"), Seat("c", "cheap"), Seat("d", "bulk")])
    picks = [s.model for _, s in roster.staff(("planner", "implementer", "critic"))]
    assert len(set(picks)) == 3, f"three roles, four seats, no repeats expected: {picks}"


def test_more_roles_than_seats_walks_the_roster():
    roster = Ensemble([Seat("mock", "test"), Seat("opus", "planner")])
    picks = [s.model for _, s in roster.staff(("planner", "implementer", "critic"))]
    assert picks[0] != picks[1] and picks[1] != picks[2], f"no seat twice running: {picks}"


def test_one_seat_is_not_an_error():
    roster = Ensemble([Seat("solo", "bulk")])
    assert [s.model for _, s in roster.staff(("planner", "critic"))] == ["solo", "solo"]


def test_weight_breaks_ties():
    roster = Ensemble([Seat("light", "bulk", 0.5), Seat("heavy", "bulk", 9.0)])
    assert roster.staff(("planner",))[0][1].model == "heavy"


def test_staffing_an_empty_order_is_empty():
    assert Ensemble([Seat("a", "test")]).staff(()) == []


# -- the commands ------------------------------------------------------------


def test_both_commands_are_registered():
    names = {c.name for c in COMMANDS}
    assert {"seats", "council"} <= names


def test_seats_lists_the_roster(state):
    body = "\n".join(dispatch(state, "/seats").lines)
    assert "mock" in body and "seats, in order" in body


def test_seats_can_be_set_and_rebuilt(state):
    dispatch(state, "/seats mock claude-opus-4-20250514")
    assert [s.model for s in state.ensemble] == ["mock", "claude-opus-4-20250514"]
    dispatch(state, "/seats auto")
    assert list(state.ensemble)[0].model == "mock"


def test_seats_off_collapses_to_one_model(state):
    out = dispatch(state, "/seats off")
    assert len(list(state.ensemble)) == 1
    assert "one model" in "\n".join(out.lines)


def test_an_unknown_model_id_is_labelled_not_rejected(state):
    """The catalogue is deliberately open: a model released today must work."""
    body = "\n".join(dispatch(state, "/seats mock model-from-tomorrow").lines)
    assert "model-from-tomorrow" in body
    assert "unknown id" in body, "it must not claim the id merely lacks a key"


def test_council_needs_two_seats(state):
    dispatch(state, "/seats off")
    got = dispatch(state, "/council what now")
    assert not got.lines[0].startswith("judge"), got.lines
    assert "two seats" in "\n".join(got.lines)


def test_council_with_no_question_explains_itself(state):
    body = "\n".join(dispatch(state, "/council").lines)
    for name in STRATEGIES:
        assert name in body, f"{name} must be discoverable"


@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_strategy_runs_and_reports_each_seat(state, strategy):
    dispatch(state, "/seats mock mock")
    out = dispatch(state, f"/council {strategy} how should i cache this")
    assert out.job is not None, f"{strategy} must run off the UI thread"
    done = out.job()
    body = "\n".join(done.lines)
    assert "mock" in body, f"{strategy} reported nothing about any seat: {body}"


def test_the_judge_does_not_grade_its_own_answer(state, monkeypatch):
    """A judge in the answering pool marks its own homework."""
    seen: dict[str, object] = {}
    real = Ensemble.council

    def spy(self, request, judge, seats=None, **kw):
        seen["judge"] = judge
        seen["answering"] = list(seats) if seats is not None else list(self)
        return real(self, request, judge, seats, **kw)

    monkeypatch.setattr(Ensemble, "council", spy)
    dispatch(state, "/seats mock claude-opus-4-20250514 gemini-2.5-flash")
    dispatch(state, "/council judge which cache").job()
    assert seen["judge"] not in seen["answering"], "the judge answered its own question"


# -- /flow: several models on one task ---------------------------------------


PLAN_JSON = (
    '{"steps": ['
    '{"id": "survey", "task": "find the call sites", "role": "planner", "writes": false},'
    '{"id": "tests", "task": "read the tests", "role": "critic", "writes": false},'
    '{"id": "rewrite", "task": "rewrite it", "role": "implementer", "needs": ["survey", "tests"]},'
    '{"id": "review", "task": "review it", "role": "critic", "needs": ["rewrite"], "writes": false}]}'
)


@pytest.fixture()
def planning(monkeypatch):
    """Every model answers the planner with JSON and every step with prose."""
    from offset.providers import mock, registry
    from offset.providers.mock import script

    class Scripted(mock.Mock):
        def stream(self, request, *, api_key=None, credential=None):
            asked = request.messages[-1].text or ""
            body = PLAN_JSON if "JSON plan" in asked else f"{request.model} finished the step"
            yield from script(body)

    real = registry.resolve

    def resolve(model_id):
        return Scripted([]), real(model_id)[1]

    # Both the agent and the ensemble look the resolver up at call time, so the
    # registry is the only place that needs patching.
    monkeypatch.setattr(registry, "resolve", resolve)
    return resolve


def test_flow_is_registered():
    assert any(c.name == "flow" for c in COMMANDS)


def test_flow_needs_a_task(state):
    assert "usage" in "\n".join(dispatch(state, "/flow").lines)


def test_flow_runs_the_plan_the_planner_wrote(state, planning):
    dispatch(state, "/seats mock claude-opus-4-20250514 gemini-2.5-flash")
    out = dispatch(state, "/flow make the parser streaming")
    assert out.job is not None, "it must run off the UI thread"
    body = "\n".join(out.job().lines)
    for step in ("survey", "tests", "rewrite", "review"):
        assert step in body, f"{step} is missing from the report: {body}"
    assert "wave 1" in body and "wave 2" in body


def test_flow_spreads_the_work_across_the_roster(state, planning):
    """Seating wave by wave gave every step to the same model."""
    dispatch(state, "/seats mock claude-opus-4-20250514 gemini-2.5-flash")
    body = "\n".join(dispatch(state, "/flow make the parser streaming").job().lines)
    used = {model for model in ("mock", "claude-opus-4-20250514", "gemini-2.5-flash") if model in body}
    assert len(used) == 3, f"only {used} did any work"


def test_flow_records_the_run_for_later(state, planning):
    dispatch(state, "/seats mock mock")
    dispatch(state, "/flow do the thing").job()
    assert state.flow_run is not None
    assert len(state.flow_run.steps) == 4


def test_flow_survives_a_planner_that_cannot_be_reached(state):
    """No key, no plan - it must still do the work as a single step."""
    dispatch(state, "/seats mock mock")
    body = "\n".join(dispatch(state, "/flow do the thing").job().lines)
    assert "1 steps" in body or "wave 1" in body
