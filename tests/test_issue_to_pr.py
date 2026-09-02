"""Issue to pull request: the pipeline, not the model.

Every collaborator here is a double, so these tests pin the behaviour that
decides whether a maintainer welcomes or dreads an automated pull request:
whether the interpretation is on the record, whether a vague issue is refused
instead of guessed at, whether a failing verify can reach a mergeable pull
request, and whether a run interrupted or broken at the last step loses the
work it already paid for.

Nothing here reaches a network, a model or a git checkout.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from offset.core.issue_to_pr import (
    BLOCKED,
    BRANCH,
    COMPLETE,
    DONE,
    FAILED,
    IMPLEMENT,
    OPEN,
    PENDING,
    PIPELINE,
    PLAN,
    READ,
    REFUSED,
    RESTATE,
    REVIEW,
    SKIPPED,
    STOPPED,
    TEST,
    Crew,
    Run,
    Stage,
    create,
    drive,
    load,
    path_for,
    pr_body,
    resume,
    start,
    status,
    step,
    vagueness,
)

VERIFY = "pytest -q"

#: An issue that says what to change: it names a file, an error and what was
#: expected.  Anything less specific is the refusal case, tested separately.
GOOD_ISSUE = {
    "title": "Crash on an empty config file",
    "body": (
        "Running `offset` with an empty config.toml raises a Traceback out of "
        "offset/core/settings.py instead of saying the file is empty. Steps to "
        "reproduce: touch config.toml, then start offset. Expected a clear error."
    ),
}

DIFF = "diff --git a/offset/core/settings.py b/offset/core/settings.py\n+    return {}\n"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Isolate the run files so `runs_dir()` cannot touch a real home."""
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    return tmp_path


# -- doubles ------------------------------------------------------------------


class FakeForge:
    """A forge that records what it was asked to do.

    Answers with plain mappings, which is the shape a GitLab client or a test
    double naturally produces; the `Reply`-shaped alternative gets its own test.
    """

    def __init__(self, issue=None, *, open_answer=None, comment_answer=None):
        self.issue = dict(GOOD_ISSUE if issue is None else issue)
        self.open_answer = open_answer if open_answer is not None else {"html_url": "https://forge/pr/7"}
        self.comment_answer = comment_answer if comment_answer is not None else {"id": 1}
        self.reads: list[int] = []
        self.comments: list[tuple[int, str]] = []
        self.opened: list[dict] = []

    def read_issue(self, number):
        self.reads.append(number)
        return self.issue

    def comment(self, number, body):
        self.comments.append((number, body))
        return self.comment_answer

    def open_pr(self, *, title, body, head, base, draft):
        self.opened.append({"title": title, "body": body, "head": head, "base": base, "draft": draft})
        return self.open_answer


class Reply:
    """The `core.forge.Reply` shape, for the normalising path."""

    def __init__(self, ok, data=None, error=None):
        self.ok = ok
        self.data = data
        self.error = error


def thinker(**answers):
    """A `Think` answering per stage, recording the order it was called in."""
    said = {
        RESTATE: "The empty config file should produce a readable error, not a traceback.",
        PLAN: "1. guard the empty-file case in offset/core/settings.py",
        IMPLEMENT: "edited offset/core/settings.py",
        REVIEW: "- the guard matches the requirement\n- no unrelated changes",
    }
    said.update(answers)
    calls: list[str] = []

    def think(stage: str, prompt: str) -> tuple[str, str]:
        calls.append(stage)
        answer = said.get(stage, "")
        if isinstance(answer, tuple):
            return answer
        return answer, ""

    think.calls = calls  # type: ignore[attr-defined]
    return think


def runner(*, verify=(0, "12 passed"), branch=(0, ""), diff=(0, DIFF)):
    """A `Runner` keyed on what kind of command it was handed."""
    calls: list[str] = []

    def run(command: str, cwd) -> tuple[int, str]:
        calls.append(command)
        if command.startswith("git checkout"):
            return branch
        if command.startswith("git diff"):
            return diff
        return verify

    run.calls = calls  # type: ignore[attr-defined]
    return run


def crew(forge=None, think=None, run=None) -> Crew:
    return Crew(forge=forge or FakeForge(), think=think or thinker(), runner=run or runner())


def opened_body(forge: FakeForge) -> str:
    assert forge.opened, "no pull request was opened"
    return forge.opened[-1]["body"]


# -- the happy path -----------------------------------------------------------


def test_a_new_run_is_on_disk_before_any_stage_runs():
    """The restatement outlives the terminal, so the record must exist first."""
    run = create(4, verify=VERIFY)
    assert path_for(run.id).exists()
    reloaded = load(run.id)
    assert reloaded is not None
    assert [s.name for s in reloaded.stages] == list(PIPELINE)
    assert all(s.state == PENDING for s in reloaded.stages)


def test_the_pipeline_advances_one_stage_per_call():
    """`step` settles exactly one stage, in pipeline order, persisting each."""
    run = create(4, verify=VERIFY)
    team = crew()
    for expected in PIPELINE:
        assert run.current is not None
        assert run.current.name == expected
        step(run, team)
        settled = load(run.id)
        assert settled is not None
        assert settled.stage(expected).state == DONE, f"{expected} did not settle on disk"
    assert run.state == COMPLETE
    assert run.current is None


def test_a_finished_run_opens_a_ready_pull_request_with_the_restatement_in_it():
    forge = FakeForge()
    run = start(4, crew(forge), verify=VERIFY)
    assert run.state == COMPLETE
    assert len(forge.opened) == 1
    request = forge.opened[0]
    assert request["draft"] is False, "a passing verify must not be published as a draft"
    assert request["head"] == run.branch == "issue-4-crash-on-an-empty-config-file"
    assert "Closes #4" in request["body"]
    assert run.restatement in request["body"], "the interpretation must be reviewable in ten seconds"
    assert run.review in request["body"]
    assert run.pr_url == "https://forge/pr/7"


def test_the_restatement_is_written_before_the_plan_is_asked_for():
    """Order matters: a plan made before the requirement is stated is a guess."""
    think = thinker()
    run = start(4, crew(think=think), verify=VERIFY)
    assert think.calls == [RESTATE, PLAN, IMPLEMENT, REVIEW]
    assert run.stage(RESTATE).output == run.restatement


def test_a_reply_shaped_forge_answer_is_read_like_a_mapping():
    """GitHub's own client returns `Reply`, not a dict; both must work."""
    forge = FakeForge(open_answer=Reply(True, {"html_url": "https://forge/pr/9"}))
    forge.issue = dict(GOOD_ISSUE)
    forge.read_issue = lambda number: Reply(True, dict(GOOD_ISSUE))  # type: ignore[method-assign]
    run = start(4, crew(forge), verify=VERIFY)
    assert run.state == COMPLETE
    assert run.pr_url == "https://forge/pr/9"


def test_a_gitlab_style_url_is_still_found():
    forge = FakeForge(open_answer={"web_url": "https://gitlab/mr/3"})
    run = start(4, crew(forge), verify=VERIFY)
    assert run.pr_url == "https://gitlab/mr/3"


# -- surviving a restart ------------------------------------------------------


def test_a_run_reloaded_from_disk_mid_run_does_not_repeat_the_stages_it_paid_for():
    """The file is the truth: a restart resumes at the boundary."""
    run = create(4, verify=VERIFY)
    think = thinker()
    team = Crew(forge=FakeForge(), think=think, runner=runner())
    step(run, team)  # read
    step(run, team)  # restate
    assert think.calls == [RESTATE]

    reloaded = load(run.id)
    assert reloaded is not None
    assert reloaded.restatement == run.restatement
    assert reloaded.title == GOOD_ISSUE["title"]

    drive(reloaded, team)
    assert reloaded.state == COMPLETE
    assert think.calls == [RESTATE, PLAN, IMPLEMENT, REVIEW], "a reloaded run re-asked a settled stage"
    assert reloaded.stage(READ).attempts == 1


def test_a_stage_left_running_by_a_crash_is_retried_and_a_settled_one_is_not():
    run = create(4, verify=VERIFY)
    team = crew()
    step(run, team)  # read settles
    run.stage(RESTATE).state = "running"  # what a kill -9 mid-stage leaves behind
    from offset.core.issue_to_pr import save

    save(run)

    resumed, error = resume(run.id, team)
    assert error == ""
    assert resumed is not None
    assert resumed.state == COMPLETE
    assert resumed.stage(READ).attempts == 1, "a settled stage was run again"
    assert resumed.stage(RESTATE).attempts == 1


# -- refusing to guess --------------------------------------------------------


@pytest.mark.parametrize(
    "title, body",
    [
        ("", ""),
        ("login broken", "doesn't work"),
        ("Please make the whole thing faster and nicer than it currently is today", ""),
    ],
)
def test_an_issue_with_no_reproducible_ask_is_called_vague(title, body):
    assert vagueness(title, body) != ""


def test_a_specific_issue_is_not_refused():
    assert vagueness(GOOD_ISSUE["title"], GOOD_ISSUE["body"]) == ""


def test_a_vague_issue_gets_a_comment_and_no_pull_request():
    forge = FakeForge({"title": "login broken", "body": "doesn't work"})
    think = thinker()
    run = start(4, crew(forge, think=think), verify=VERIFY)

    assert run.state == REFUSED
    assert forge.opened == [], "a speculative pull request is worse than no pull request"
    assert len(forge.comments) == 1
    number, text = forge.comments[0]
    assert number == 4
    assert "cannot tell what to change" in text
    assert run.refusal
    assert think.calls == [], "no model call is worth making before the issue is readable"
    assert run.stage(RESTATE).state == SKIPPED


def test_the_model_may_refuse_too_and_that_also_stops_the_run():
    """The heuristic cannot catch everything; `UNCLEAR` is the model's way out."""
    forge = FakeForge()
    think = thinker(**{RESTATE: "UNCLEAR: which of the two config files do you mean?"})
    run = start(4, crew(forge, think=think), verify=VERIFY)

    assert run.state == REFUSED
    assert forge.opened == []
    assert "which of the two config files" in forge.comments[0][1]
    assert run.stage(PLAN).state == SKIPPED


def test_a_refusal_survives_a_forge_that_cannot_comment():
    """The issue is still unimplementable, and the silence is on the record."""

    class Mute(FakeForge):
        def comment(self, number, body):
            raise RuntimeError("403 no write access")

    forge = Mute({"title": "login broken", "body": "doesn't work"})
    run = start(4, crew(forge), verify=VERIFY)
    assert run.state == REFUSED
    assert forge.opened == []
    assert "did not post" in run.error
    assert "403" in run.error


def test_a_refused_run_is_not_resumable():
    """Only the reporter can unblock it, so retrying would just re-comment."""
    forge = FakeForge({"title": "login broken", "body": "doesn't work"})
    team = crew(forge)
    run = start(4, team, verify=VERIFY)
    again, why = resume(run.id, team)
    assert again is not None
    assert "refused" in why
    assert len(forge.comments) == 1


# -- a failing verify ---------------------------------------------------------


def test_a_failing_verify_opens_a_draft_carrying_the_failure():
    forge = FakeForge()
    run = start(4, crew(forge, run=runner(verify=(1, "E   assert 1 == 2\n1 failed"))), verify=VERIFY)

    assert run.state == COMPLETE, "the work must still reach a human"
    assert run.verified is False
    assert run.draft is True
    request = forge.opened[0]
    assert request["draft"] is True, "a failing verify must never open as ready to merge"
    assert "1 failed" in request["body"]
    assert "assert 1 == 2" in request["body"]
    assert run.restatement in request["body"]


def test_a_failing_verify_is_recorded_as_blocked_and_does_not_stop_the_pipeline():
    """`blocked` is settled, so the driver moves on rather than retrying."""
    run = start(4, crew(run=runner(verify=(3, "boom"))), verify=VERIFY)
    assert run.stage(TEST).state == BLOCKED
    assert run.stage(TEST).attempts == 1, "the verify command was run more than once"
    assert run.stage(REVIEW).state == DONE
    assert run.stage(OPEN).state == DONE


def test_an_unverified_run_is_also_a_draft():
    """No verify command configured is not evidence that anything works."""
    forge = FakeForge()
    run = start(4, crew(forge), verify="")
    assert run.state == COMPLETE
    assert forge.opened[0]["draft"] is True
    assert "draft" in forge.opened[0]["body"]


def test_an_empty_diff_stops_the_run_before_a_pull_request_exists():
    """A pull request with no change in it wastes the reviewer, not the agent."""
    forge = FakeForge()
    run = start(4, crew(forge, run=runner(diff=(0, "   \n"))), verify=VERIFY)
    assert run.state == STOPPED
    assert forge.opened == []
    assert "diff is empty" in run.error


def test_a_branch_that_cannot_be_created_stops_the_run_with_the_reason():
    forge = FakeForge()
    run = start(4, crew(forge, run=runner(branch=(128, "fatal: branch already exists"))), verify=VERIFY)
    assert run.state == STOPPED
    assert run.stage(BRANCH).state == FAILED
    assert "already exists" in run.error
    assert run.stage(IMPLEMENT).state == SKIPPED
    assert forge.opened == []


# -- the last step ------------------------------------------------------------


def test_a_forge_error_at_the_final_step_keeps_every_earlier_result():
    """The expensive part is the model calls; a 422 must not discard them."""
    forge = FakeForge(open_answer={"error": "422 head branch already exists"})
    run = start(4, crew(forge), verify=VERIFY)

    assert run.state == STOPPED
    assert "422" in run.error
    on_disk = load(run.id)
    assert on_disk is not None
    assert on_disk.restatement and on_disk.plan and on_disk.review and on_disk.diff
    assert on_disk.branch == run.branch
    assert on_disk.verified is True
    for name in (READ, RESTATE, PLAN, BRANCH, IMPLEMENT, TEST, REVIEW):
        assert on_disk.stage(name).state == DONE, f"{name} was lost"
    assert on_disk.stage(OPEN).state == FAILED


def test_resume_picks_up_at_the_failed_stage_and_repeats_nothing_before_it():
    forge = FakeForge(open_answer={"error": "502 bad gateway"})
    think = thinker()
    team = Crew(forge=forge, think=think, runner=runner())
    run = start(4, team, verify=VERIFY)
    assert run.state == STOPPED
    assert think.calls == [RESTATE, PLAN, IMPLEMENT, REVIEW]

    forge.open_answer = {"html_url": "https://forge/pr/11"}
    resumed, error = resume(run.id, team)

    assert error == ""
    assert resumed is not None
    assert resumed.state == COMPLETE
    assert resumed.pr_url == "https://forge/pr/11"
    assert think.calls == [RESTATE, PLAN, IMPLEMENT, REVIEW], "resuming re-ran a stage that had finished"
    assert forge.reads == [4], "resuming re-read the issue"
    assert resumed.stage(OPEN).attempts == 2
    assert resumed.restatement in forge.opened[-1]["body"]


def test_resume_reports_a_run_it_cannot_find():
    run, why = resume("nope", crew())
    assert run is None
    assert "nope" in why


def test_a_complete_run_is_not_driven_again():
    forge = FakeForge()
    team = crew(forge)
    run = start(4, team, verify=VERIFY)
    again, why = resume(run.id, team)
    assert again is not None
    assert "already complete" in why
    assert len(forge.opened) == 1


# -- the body and the report --------------------------------------------------


def test_the_body_puts_the_interpretation_above_the_plan():
    run = Run(issue=9, restatement="Make the empty config readable.", plan="1. guard it",
              review="- fine", verify=VERIFY, verified=True)
    body = pr_body(run)
    assert body.index("What I understood") < body.index("Plan")
    assert body.index(run.restatement) < body.index(run.plan)
    assert "Closes #9" in body


def test_the_body_of_a_failed_run_says_why_it_is_a_draft():
    run = Run(issue=9, restatement="x", verify=VERIFY, failure="E assert 1 == 2", verified=False)
    body = pr_body(run)
    assert "failed, so this is a draft" in body
    assert "assert 1 == 2" in body


def test_status_lists_runs_and_reports_one_by_prefix():
    run = start(4, crew(), verify=VERIFY)
    lines = status()
    assert any(run.id[:10] in line for line in lines)
    report = status(run.id[:8])
    assert any("understood as:" in line for line in report)
    assert any("pull request:" in line for line in report)
    assert status("zzzz") == ["no run 'zzzz'"]


def test_an_unreadable_run_file_is_ignored_rather_than_fatal():
    run = create(4, verify=VERIFY)
    path_for(run.id).write_text("{ not json", encoding="utf-8")
    assert load(run.id) is None
    assert status() == ["no issue runs yet", "start one with /issue <n>"]


def test_a_run_of_a_future_version_is_refused_rather_than_half_read():
    run = create(4, verify=VERIFY)
    raw = json.loads(path_for(run.id).read_text(encoding="utf-8"))
    raw["version"] = 99
    path_for(run.id).write_text(json.dumps(raw), encoding="utf-8")
    assert load(run.id) is None


def test_a_handler_fault_is_the_stages_outcome_not_a_crash():
    """A double that raises must fail the stage, not the process."""

    class Angry(FakeForge):
        def read_issue(self, number):
            raise TimeoutError("no route to host")

    run = start(4, crew(Angry()), verify=VERIFY)
    assert run.state == STOPPED
    assert run.stage(READ).state == FAILED
    assert "TimeoutError" in run.error


def test_the_commands_are_built_once_and_lazily():
    import offset.core.issue_to_pr as mod

    first = mod.COMMANDS
    assert [c.name for c in first] == ["issue"]
    assert mod.COMMANDS is first, "a second access built a second list"
    with pytest.raises(AttributeError):
        _ = mod.NOT_A_THING


def test_the_stage_vocabulary_is_the_one_tasks_already_uses():
    """One vocabulary, so `/issue status` and `/tasks` read the same way."""
    from offset.core import tasks

    assert Stage is tasks.Stage
    assert (DONE, FAILED, BLOCKED, SKIPPED) == (tasks.DONE, tasks.FAILED, tasks.BLOCKED, tasks.SKIPPED)


def test_the_slash_command_says_how_to_use_it_and_refuses_a_non_number(tmp_path):
    """The argument becomes a branch name and a forge path, so it is checked
    before any of that starts."""
    import offset.core.issue_to_pr as mod

    state = SimpleNamespace(workspace=tmp_path, verify_command=VERIFY, agent=None)
    handler = mod.COMMANDS[0].run

    assert "usage: /issue <number>" in handler(state, []).lines
    assert any("not an issue number" in line for line in handler(state, ["latest"]).lines)
    assert any("resume <id>" in line for line in handler(state, ["resume"]).lines)


def test_issue_status_goes_through_the_command_without_touching_a_forge(tmp_path):
    run = start(4, crew(), verify=VERIFY)
    import offset.core.issue_to_pr as mod

    state = SimpleNamespace(workspace=tmp_path, verify_command=VERIFY, agent=None)
    lines = mod.COMMANDS[0].run(state, ["status"]).lines
    assert any(run.id[:10] in line for line in lines)


# -- either forge -------------------------------------------------------------


def test_a_gitlab_remote_selects_the_gitlab_forge(tmp_path, monkeypatch):
    """A GitLab-hosted issue sent to api.github.com 404s after the branch
    already exists, so the remote decides before any stage runs."""
    from offset.core.issue_to_pr import forge_for
    from offset.tools import gitlab as gitlab_mod

    remote = gitlab_mod.Remote(host="gitlab.com", path="g/p")
    detected: list[int] = []
    built: list[object] = []

    def detect(*a, **k):
        detected.append(1)
        return remote

    def connect(cwd, **options):
        built.append(options.get("remote"))
        return SimpleNamespace(problem="")

    monkeypatch.setattr(gitlab_mod, "detect", detect)
    monkeypatch.setattr(gitlab_mod, "connect", connect)

    forge, why = forge_for(tmp_path)
    assert why == ""
    assert isinstance(forge, gitlab_mod.IssueForge)
    for call in ("read_issue", "comment", "open_pr"):
        assert callable(getattr(forge, call))
    assert len(detected) == 1
    assert built == [remote], "the detected remote must be handed back rather than re-detected"


def test_a_gitlab_remote_with_no_token_says_so_instead_of_falling_back(tmp_path, monkeypatch):
    """Falling through to GitHub here would report "no github remote here" for
    a repository that has a perfectly good GitLab one."""
    from offset.core.issue_to_pr import forge_for
    from offset.tools import gitlab as gitlab_mod

    monkeypatch.setattr(gitlab_mod, "detect", lambda *a, **k: gitlab_mod.Remote(host="gitlab.com", path="g/p"))
    monkeypatch.setattr(gitlab_mod, "connect", lambda *a, **k: SimpleNamespace(problem="set GITLAB_TOKEN"))

    forge, why = forge_for(tmp_path)
    assert forge is None
    assert why == "set GITLAB_TOKEN"


def test_a_github_remote_selects_the_github_forge(tmp_path, monkeypatch):
    from offset.core import forge as forge_mod
    from offset.core.issue_to_pr import GitHubForge, forge_for
    from offset.tools import gitlab as gitlab_mod

    monkeypatch.setattr(gitlab_mod, "detect", lambda *a, **k: gitlab_mod.Remote(error="no GitLab remote here"))
    monkeypatch.setattr(forge_mod, "connect", lambda *a, **k: SimpleNamespace(ok=True, error=None))

    forge, why = forge_for(tmp_path)
    assert why == ""
    assert isinstance(forge, GitHubForge)


def test_a_checkout_with_no_forge_at_all_is_a_reason_not_a_crash(tmp_path, monkeypatch):
    from offset.core import forge as forge_mod
    from offset.core.issue_to_pr import forge_for
    from offset.tools import gitlab as gitlab_mod

    monkeypatch.setattr(gitlab_mod, "detect", lambda *a, **k: gitlab_mod.Remote(error="no GitLab remote here"))
    monkeypatch.setattr(forge_mod, "connect", lambda *a, **k: SimpleNamespace(ok=False, error="no remote"))

    forge, why = forge_for(tmp_path)
    assert forge is None
    assert why == "no remote"


def test_the_github_adapter_speaks_the_three_calls_and_nothing_else():
    """The adapter is the only place that knows GitHub's paths."""
    from offset.core.issue_to_pr import GitHubForge

    sent: list[tuple] = []

    class Linked:
        def call(self, method, path, **options):
            sent.append((method, path, options.get("body")))
            return {"title": "t", "body": "b"}

        def create_pr(self, *, title, head, base, body, draft):
            sent.append(("create_pr", head, base, draft))
            return {"html_url": "https://github/pr/1"}

    forge = GitHubForge(Linked())
    forge.read_issue(4)
    forge.comment(4, "hello")
    forge.open_pr(title="t", body="b", head="h", base="main", draft=True)

    assert sent[0] == ("GET", "{repo}/issues/4", None)
    assert sent[1] == ("POST", "{repo}/issues/4/comments", {"body": "hello"})
    assert sent[2] == ("create_pr", "h", "main", True)
