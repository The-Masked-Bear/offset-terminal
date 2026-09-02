"""Issue to pull request, with the interpretation on the record.

`/task` already plans, implements and retests, and `tools/github` already opens
a pull request.  What neither does is start from the thing a maintainer
actually has - an issue number - and end at a review-ready change.  Wiring the
two together naively produces the worst artefact in this area: a confident pull
request that solves a different problem from the one that was reported.

So this pipeline is built around the three failures that make automated pull
requests unwelcome, and each one is a stage rather than a convention:

**The restatement is a stage, not decoration.**  Before planning anything the
agent writes what it believes the issue asks for, that sentence is persisted,
and it goes at the top of the pull request body.  A wrong interpretation is
then visible in ten seconds instead of after a reviewer has read the diff.

**A vague issue is refused, never guessed at.**  An issue with no reproducible
ask gets a comment asking for specifics and the run stops in `refused`.  Both
routes to that decision exist: a deterministic check on the issue text, and the
model answering `UNCLEAR: <question>` at the restate stage, which is the only
honest answer when the text does not say what to change.

**A failing verify blocks the merge, not the pull request.**  The run carries
the failure forward and opens a draft with the output in the body.  Stopping
the run instead would hide the work; opening it ready-to-merge would hide the
failure.  Unverified work - no verify command configured - is a draft for the
same reason.

Two further decisions, both about testability.  Every transition is written to
disk before it is acted on, atomically, so a restart resumes at the stage
boundary rather than replaying the model calls that have already been paid for;
this reuses `core.tasks`' stage vocabulary rather than inventing a second one.
And the three collaborators - the forge, the model and the command runner - are
injected as a `Crew`, so the whole pipeline is exercised without a network, a
model or a git checkout, and so GitHub and GitLab plug into the same code by
supplying `read_issue`, `comment` and `open_pr`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Protocol

from offset.core import settings
from offset.core.entries import new_id
from offset.core.tasks import (
    ACTIVE,
    BLOCKED,
    COMPLETE,
    DONE,
    FAILED,
    PENDING,
    RUNNING,
    SKIPPED,
    STOPPED,
    Stage,
)

#: The one state `core.tasks` has no word for.  A refused run is finished and
#: must never be resumed: the missing information is the reporter's to supply,
#: so retrying would only re-post the same comment.
REFUSED: Final = "refused"

READ: Final = "read"
RESTATE: Final = "restate"
PLAN: Final = "plan"
BRANCH: Final = "branch"
IMPLEMENT: Final = "implement"
TEST: Final = "test"
REVIEW: Final = "review"
OPEN: Final = "open-pr"

#: The stages, in order.  There is no fix loop here on purpose: `/task` owns
#: that, and a pipeline that keeps retrying until green is exactly how an
#: unattended agent spends a night on an impossible failure.  One verify run,
#: and the outcome goes in the body.
PIPELINE: Final = (READ, RESTATE, PLAN, BRANCH, IMPLEMENT, TEST, REVIEW, OPEN)

VERSION: Final = 1

#: Below this many words an issue has not described anything to implement.
#: Twelve is roughly "one sentence of context plus one of ask"; the issues that
#: fall below it in practice are "doesn't work", "please fix the login" and
#: "see slack", none of which a diff can answer.
MIN_WORDS: Final = 12

#: The model's own way out at the restate stage.  Cheaper than any heuristic:
#: the text either says what to change or it does not, and the model has just
#: read all of it.
UNCLEAR: Final = "UNCLEAR"

#: Characters of diff handed to the critic.  Large enough for a real change,
#: small enough to leave the model room to answer.
DIFF_BUDGET: Final = 24_000

#: Characters of command output kept.  A failing suite can emit megabytes; the
#: last few thousand characters are where the failure is.
OUTPUT_BUDGET: Final = 4_000

#: Seconds a verify command gets.  Matches `/task`'s ceiling.
VERIFY_TIMEOUT: Final = 600.0

#: Slug characters taken from the issue title for the branch name.
SLUG_CHARS: Final = 40

#: Stages to run in one `drive`.  Every stage settles in one attempt here, so
#: this is a stop against a handler that never settles, not a retry budget.
LIMIT: Final = 24


# -- what the pipeline is given ---------------------------------------------


class Forge(Protocol):
    """The three forge calls this pipeline makes.

    Deliberately smaller than `core.forge.Forge`: GitHub, GitLab and a test
    double all implement these three, so nothing below this line knows which
    forge it is talking to.  Implementations may return a mapping, a
    `Reply`-shaped object with `.ok`/`.error`/`.data`, or raise - all three are
    normalised by `_payload`.
    """

    def read_issue(self, number: int) -> Any: ...

    def comment(self, number: int, body: str) -> Any: ...

    def open_pr(self, *, title: str, body: str, head: str, base: str, draft: bool) -> Any: ...


#: One model call for one stage.  Takes the stage name rather than a system
#: prompt so an implementation picks its own wording, and so a test can script
#: by stage instead of by matching prompt text.  Returns `(text, error)`.
Think = Callable[[str, str], tuple[str, str]]

#: Runs one command in a directory and returns `(exit code, output)`.  Both the
#: branch stage and the verify stage go through this, which is why it is a
#: command runner rather than a "run the tests" hook.
Runner = Callable[[str, Path], tuple[int, str]]


@dataclass(frozen=True, slots=True)
class Crew:
    """The three collaborators a run needs, so no stage reaches for a global."""

    forge: Forge
    think: Think
    runner: Runner


@dataclass(frozen=True, slots=True)
class Done:
    """One stage's outcome, and how the run must treat it."""

    output: str = ""
    error: str = ""
    #: The error is real but the run must carry it forward instead of stopping.
    #: A failing verify is the case this exists for: blocking would mean the
    #: human never sees the draft pull request that explains the failure.
    carry: bool = False
    #: The run must not produce a pull request at all.  Set when the issue does
    #: not say what to change; the comment has already been posted.
    refuse: bool = False


# -- the record -------------------------------------------------------------


@dataclass(slots=True)
class Run:
    """One issue, the stages it goes through, and everything they produced.

    Every field a later stage reads is stored here rather than recomputed,
    because "resume" means the earlier stages are not going to run again: the
    restatement a reviewer sees has to come off disk, not out of a second model
    call that might phrase the interpretation differently.
    """

    id: str = field(default_factory=new_id)
    issue: int = 0
    state: str = ACTIVE
    stages: list[Stage] = field(default_factory=list)
    cwd: str = ""
    verify: str = ""
    base: str = ""
    branch: str = ""
    title: str = ""
    body: str = ""
    restatement: str = ""
    plan: str = ""
    notes: str = ""
    review: str = ""
    failure: str = ""
    verified: bool = False
    diff: str = ""
    pr_url: str = ""
    draft: bool = False
    refusal: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    @property
    def current(self) -> Stage | None:
        """The next stage with work left in it."""
        for stage in self.stages:
            if not stage.settled:
                return stage
        return None

    @property
    def finished(self) -> bool:
        return self.state in (COMPLETE, STOPPED, REFUSED)

    def stage(self, name: str) -> Stage | None:
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "id": self.id,
            "issue": self.issue,
            "state": self.state,
            "stages": [s.to_json() for s in self.stages],
            "cwd": self.cwd,
            "verify": self.verify,
            "base": self.base,
            "branch": self.branch,
            "title": self.title,
            "body": self.body,
            "restatement": self.restatement,
            "plan": self.plan,
            "notes": self.notes,
            "review": self.review,
            "failure": self.failure,
            "verified": self.verified,
            "diff": self.diff,
            "pr_url": self.pr_url,
            "draft": self.draft,
            "refusal": self.refusal,
            "error": self.error,
            "created": round(self.created, 6),
            "updated": round(self.updated, 6),
        }

    @classmethod
    def from_json(cls, raw: Any) -> Run | None:
        if not isinstance(raw, dict) or int(raw.get("version") or 0) != VERSION:
            return None
        return cls(
            id=str(raw.get("id") or new_id()),
            issue=int(raw.get("issue") or 0),
            state=str(raw.get("state") or ACTIVE),
            stages=[Stage.from_json(s) for s in (raw.get("stages") or [])],
            cwd=str(raw.get("cwd") or ""),
            verify=str(raw.get("verify") or ""),
            base=str(raw.get("base") or ""),
            branch=str(raw.get("branch") or ""),
            title=str(raw.get("title") or ""),
            body=str(raw.get("body") or ""),
            restatement=str(raw.get("restatement") or ""),
            plan=str(raw.get("plan") or ""),
            notes=str(raw.get("notes") or ""),
            review=str(raw.get("review") or ""),
            failure=str(raw.get("failure") or ""),
            verified=bool(raw.get("verified")),
            diff=str(raw.get("diff") or ""),
            pr_url=str(raw.get("pr_url") or ""),
            draft=bool(raw.get("draft")),
            refusal=str(raw.get("refusal") or ""),
            error=str(raw.get("error") or ""),
            created=float(raw.get("created") or time.time()),
            updated=float(raw.get("updated") or time.time()),
        )

    def report(self) -> list[str]:
        lines = [f"{self.id}  {self.state}  issue #{self.issue}", f"issue: {self.title or '(unread)'}"]
        if self.restatement:
            lines.append(f"understood as: {_oneline(self.restatement, 70)}")
        lines.extend(s.line() for s in self.stages)
        if self.refusal:
            lines.append(f"refused: {self.refusal}")
        if self.failure:
            lines.append(f"verify failed: {_oneline(self.failure, 70)}")
        if self.pr_url:
            lines.append(f"{'draft ' if self.draft else ''}pull request: {self.pr_url}")
        if self.error:
            lines.append(f"error: {self.error}")
        return lines

    def summary(self) -> str:
        done = sum(1 for s in self.stages if s.state == DONE)
        return f"{self.id[:10]}  {self.state:8s}  {done}/{len(self.stages)}  #{self.issue} {self.title[:40]}"


# -- storage ----------------------------------------------------------------


def runs_dir() -> Path:
    """Resolved on every call: `OFFSET_HOME` moves under tests and `--home`."""
    return settings.home() / "issue-runs"


def path_for(run_id: str, root: Path | None = None) -> Path:
    return (root or runs_dir()) / f"{run_id}.json"


def save(run: Run, root: Path | None = None) -> Path:
    """Write the run out atomically.

    Temp file in the same directory then `os.replace`, so a crash mid-write
    leaves the previous state readable rather than a truncated file that would
    lose the restatement and the plan along with it.
    """
    run.updated = time.time()
    target = path_for(run.id, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{run.id}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(run.to_json(), fh, indent=1)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def load(run_id: str, root: Path | None = None) -> Run | None:
    """Read one run, or None if it is absent or unreadable."""
    target = path_for(run_id, root)
    if not target.exists():
        return None
    try:
        return Run.from_json(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None


def listing(root: Path | None = None) -> list[Run]:
    """Every run, newest first."""
    where = root or runs_dir()
    if not where.exists():
        return []
    found: list[Run] = []
    for entry in where.glob("*.json"):
        try:
            run = Run.from_json(json.loads(entry.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if run is not None:
            found.append(run)
    return sorted(found, key=lambda r: r.updated, reverse=True)


def create(
    issue: int,
    *,
    cwd: Path | str = ".",
    verify: str = "",
    base: str = "main",
    root: Path | None = None,
) -> Run:
    """A new run with the full pipeline, already on disk."""
    run = Run(
        issue=int(issue),
        cwd=str(Path(cwd).resolve()),
        verify=verify.strip(),
        base=base.strip() or "main",
        stages=[Stage(name) for name in PIPELINE],
    )
    save(run, root)
    return run


# -- reading what the forge said --------------------------------------------


def _payload(value: Any) -> tuple[dict[str, Any], str]:
    """Normalise any forge answer to `(payload, error)`.

    Three shapes have to work: a plain mapping (a test double, or a forge that
    returns decoded JSON), a `core.forge.Reply` with `.ok`/`.error`/`.data`,
    and a mapping carrying its own `error` key.  Guessing wrong here would
    read a failure as an empty issue, which is precisely the input that must
    never reach the planner.
    """
    if value is None:
        return {}, "the forge returned nothing"
    ok = getattr(value, "ok", None)
    if ok is False:
        return {}, str(getattr(value, "error", None) or "the forge call failed")
    data = getattr(value, "data", None)
    if isinstance(data, dict):
        return dict(data), ""
    if isinstance(value, Mapping):
        said = value.get("error")
        if said:
            return {}, str(said)
        if value.get("ok") is False:
            return {}, "the forge call failed"
        return dict(value), ""
    if ok is True:
        # A truthful success with a payload this pipeline cannot use, which is
        # still success: `comment` legitimately answers with nothing useful.
        return {}, ""
    return {}, f"unreadable forge answer: {type(value).__name__}"


def _text_of(payload: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        got = payload.get(key)
        if isinstance(got, str) and got.strip():
            return got.strip()
    return ""


def _pr_url(payload: Mapping[str, Any]) -> str:
    """The URL a human should open.  `web_url` is GitLab's name for it."""
    return _text_of(payload, "html_url", "web_url", "url")


# -- the vagueness decision -------------------------------------------------

#: Signals that an issue names something concrete.  Each one is a thing a diff
#: can be aimed at; an issue with none of them is describing a feeling.
_FENCE = re.compile(r"```|^ {4}\S", re.M)
_PATH = re.compile(
    r"[\w./-]+\.(?:py|pyi|js|jsx|ts|tsx|go|rs|c|h|cc|cpp|java|rb|sh|toml|json|ya?ml|md|txt|cfg|ini)\b",
    re.I,
)
_FAULT = re.compile(
    r"traceback|exception|error[: ]|assertion|stack trace|segfault|exit(?:ed)? (?:code )?\d|status \d{3}|crash",
    re.I,
)
_SHAPE = re.compile(
    r"steps to reproduce|reproduce|repro\b|expected\b|actual\b|instead of|should (?:be|return|not|raise)"
    r"|when i\b|regression|no longer",
    re.I,
)
_SYMBOL = re.compile(r"`[^`\n]{2,}`|\b\w+\(\)|--?[a-z][a-z-]{2,}")


def vagueness(title: str, body: str) -> str:
    """Why this issue cannot be implemented, or `""` when it can.

    Deliberately a pure function of the text: the decision to refuse is the
    most consequential one this module makes, so it must be reproducible from
    the issue alone rather than from whatever the model felt like saying.
    """
    text = f"{title}\n{body}".strip()
    if not text:
        return "the issue has no title and no body"
    words = len(re.findall(r"\S+", text))
    if words < MIN_WORDS:
        return f"the issue is {words} words long, which is not a reproducible report"
    if not any(pattern.search(text) for pattern in (_FENCE, _PATH, _FAULT, _SHAPE, _SYMBOL)):
        return (
            "the issue names no file, symbol, command or error output, and does not say "
            "what was expected instead of what happened"
        )
    return ""


ASK_TEMPLATE: Final = """I would like to open a pull request for this, but I cannot tell what to change.

{why}.

Could you add:

- what you did, what happened, and what you expected instead
- the file, function or command involved
- the exact error output, if there is any

I will pick this up as soon as that is here, rather than guess at it."""


def ask_for_specifics(why: str) -> str:
    return ASK_TEMPLATE.format(why=why.rstrip("."))


# -- prompts ----------------------------------------------------------------


SYSTEMS: Final = {
    RESTATE: (
        "You are reading one issue before any code is written. In at most four sentences, "
        "say what you believe the issue asks for: the observed behaviour, the wanted "
        "behaviour, and where in the project you think it lives. Do not plan, do not "
        "apologise, do not restate the issue verbatim.\n"
        f"If the issue does not say what to change, reply with exactly '{UNCLEAR}: ' "
        "followed by the one question that would unblock you."
    ),
    PLAN: (
        "You are planning one change. Reply with a short numbered list of the concrete "
        "edits required, naming files. If it is one edit, say so in one line rather than "
        "inventing steps to fill a list."
    ),
    IMPLEMENT: "Make the edits now. Change code; do not describe what you would change.",
    REVIEW: (
        "You are reviewing a diff you did not write, against the requirement stated above. "
        "Answer in at most five bullets: anything the diff does that the requirement did not "
        "ask for, anything the requirement asked for that the diff does not do, and any bug "
        "you can see. If it is sound, say so in one line. Never comment on formatting."
    ),
}


def _prompt(run: Run, stage: str) -> str:
    """The user half of one stage's model call."""
    issue = f"Issue #{run.issue}: {run.title}\n\n{run.body}".strip()
    if stage == RESTATE:
        return issue
    if stage == PLAN:
        return f"{issue}\n\nWhat the issue asks for:\n{run.restatement}"
    if stage == IMPLEMENT:
        return f"What the issue asks for:\n{run.restatement}\n\nYour plan:\n{run.plan}\n\nMake the edits now."
    # review
    return (
        f"The requirement:\n{run.restatement}\n\n"
        f"The diff:\n{run.diff[:DIFF_BUDGET]}"
    )


# -- the stages -------------------------------------------------------------


def _tail(text: str, limit: int = OUTPUT_BUDGET) -> str:
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text


def _oneline(text: str, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def branch_name(issue: int, title: str) -> str:
    """A branch name a human can read, derived only from what is on disk.

    Deterministic on purpose: a resumed run recomputes it and must land on the
    same branch it already created.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")[:SLUG_CHARS].strip("-")
    return f"issue-{issue}-{slug}" if slug else f"issue-{issue}"


def _do_read(run: Run, crew: Crew) -> Done:
    payload, error = _payload(crew.forge.read_issue(run.issue))
    if error:
        return Done(error=error)
    run.title = _text_of(payload, "title")
    run.body = _text_of(payload, "body", "description")
    if not run.title and not run.body:
        return Done(error=f"issue #{run.issue} came back with no title and no body")

    why = vagueness(run.title, run.body)
    if not why:
        return Done(output=f"#{run.issue} {run.title}")
    return _refuse(run, crew, why)


def _refuse(run: Run, crew: Crew, why: str) -> Done:
    """Post the request for specifics and end the run.

    The comment failing does not make the issue implementable, so the refusal
    stands either way and the posting failure is recorded on the run instead of
    replacing it.  Swallowing the exception here is deliberate: a forge that
    cannot comment must not be reported as "the issue was fine".
    """
    run.refusal = why
    try:
        _, error = _payload(crew.forge.comment(run.issue, ask_for_specifics(why)))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if error:
        run.error = f"the request for specifics did not post: {error}"
    posted = "asked for specifics" if not error else "could not ask for specifics"
    return Done(output=f"refused: {why} ({posted})", refuse=True)


def _do_restate(run: Run, crew: Crew) -> Done:
    text, error = crew.think(RESTATE, _prompt(run, RESTATE))
    if error:
        return Done(error=error)
    text = (text or "").strip()
    if not text:
        return Done(error="the restatement came back empty")
    if text.upper().startswith(UNCLEAR):
        question = text[len(UNCLEAR) :].lstrip(": ").strip()
        return _refuse(run, crew, question or "I cannot tell what this issue asks for")
    run.restatement = text
    return Done(output=text)


def _do_plan(run: Run, crew: Crew) -> Done:
    text, error = crew.think(PLAN, _prompt(run, PLAN))
    if error:
        return Done(error=error)
    if not (text or "").strip():
        return Done(error="the plan came back empty")
    run.plan = text.strip()
    return Done(output=run.plan)


def _do_branch(run: Run, crew: Crew) -> Done:
    name = run.branch or branch_name(run.issue, run.title)
    code, output = crew.runner(f"git checkout -b {name}", Path(run.cwd or "."))
    if code != 0:
        return Done(output=_tail(output), error=f"could not create branch {name}: {_oneline(output, 120)}")
    run.branch = name
    return Done(output=name)


def _do_implement(run: Run, crew: Crew) -> Done:
    text, error = crew.think(IMPLEMENT, _prompt(run, IMPLEMENT))
    if error:
        return Done(error=error)
    run.notes = (text or "").strip()
    if not run.notes:
        return Done(error="the implementation step reported nothing")
    return Done(output=run.notes)


NO_VERIFY: Final = "no verify command is configured, so nothing was run"


def _do_test(run: Run, crew: Crew) -> Done:
    if not run.verify:
        # `verified` stays False, which is what makes the pull request a draft.
        # An unverified change presented as ready to merge is the same lie as a
        # failing one presented as ready to merge.
        return Done(output=NO_VERIFY)
    code, output = crew.runner(run.verify, Path(run.cwd or "."))
    if code == 0:
        run.verified = True
        run.failure = ""
        return Done(output=_tail(output) or "the verify command passed")
    run.verified = False
    run.failure = _tail(output) or f"{run.verify!r} exited {code}"
    return Done(output=run.failure, error=f"{run.verify!r} exited {code}", carry=True)


#: Committed work first, uncommitted second.  Which one holds the change
#: depends on whether the implement stage committed, and a critic handed an
#: empty diff reviews nothing while saying it reviewed everything.
DIFF_COMMANDS: Final = ("git diff {base}...HEAD", "git diff HEAD")


def _do_review(run: Run, crew: Crew) -> Done:
    diff = ""
    for template in DIFF_COMMANDS:
        code, output = crew.runner(template.format(base=run.base), Path(run.cwd or "."))
        if code == 0 and output.strip():
            diff = output
            break
    if not diff.strip():
        return Done(error="the diff is empty: nothing was changed, so there is nothing to open")
    run.diff = diff[:DIFF_BUDGET]

    text, error = crew.think(REVIEW, _prompt(run, REVIEW))
    if error:
        return Done(error=error)
    if not (text or "").strip():
        return Done(error="the self-review came back empty")
    run.review = text.strip()
    return Done(output=run.review)


def pr_title(run: Run) -> str:
    """The issue's own words, capped.  A reviewer scanning a list of pull
    requests wants the report's title, not a summary of the diff."""
    return _oneline(run.title or f"issue #{run.issue}", 72)


def pr_body(run: Run) -> str:
    """The pull request body: interpretation first, then verification.

    `Closes #n` is the whole point of the pipeline and both forges honour it.
    The order is chosen for a reviewer who will stop reading after the first
    screen: what the agent thought it was asked to do, then whether the tests
    passed, then the plan and the critic's notes.
    """
    parts = [f"Closes #{run.issue}.", "", "## What I understood", "", run.restatement or "(nothing recorded)"]
    parts += ["", "## Verification", ""]
    if run.verified:
        parts.append(f"`{run.verify}` passed.")
    elif run.failure:
        parts += [
            f"**`{run.verify}` failed, so this is a draft.**",
            "",
            "```",
            _tail(run.failure, 2_000),
            "```",
        ]
    else:
        parts.append(f"{NO_VERIFY.capitalize()}, so this is a draft.")
    if run.plan:
        parts += ["", "## Plan", "", run.plan]
    if run.review:
        parts += ["", "## Self-review", "", run.review]
    parts += ["", "---", f"Opened by offset from issue #{run.issue}; run `{run.id}`."]
    return "\n".join(parts)


def _do_open(run: Run, crew: Crew) -> Done:
    if not run.branch:
        return Done(error="no branch was created, so there is nothing to open a pull request from")
    run.draft = not run.verified
    payload, error = _payload(
        crew.forge.open_pr(
            title=pr_title(run),
            body=pr_body(run),
            head=run.branch,
            base=run.base,
            draft=run.draft,
        )
    )
    if error:
        return Done(error=error)
    run.pr_url = _pr_url(payload)
    kind = "draft pull request" if run.draft else "pull request"
    return Done(output=f"{kind}: {run.pr_url or '(no url returned)'}")


_HANDLERS: Final[dict[str, Callable[[Run, Crew], Done]]] = {
    READ: _do_read,
    RESTATE: _do_restate,
    PLAN: _do_plan,
    BRANCH: _do_branch,
    IMPLEMENT: _do_implement,
    TEST: _do_test,
    REVIEW: _do_review,
    OPEN: _do_open,
}


# -- the state machine ------------------------------------------------------


def _skip_rest(run: Run, stage: Stage) -> None:
    """Mark everything after `stage` as skipped.

    `stage` itself is excluded by identity: it is the one carrying the reason
    the run stopped, and relabelling it would erase that.
    """
    for later in run.stages:
        if later is not stage and not later.settled:
            later.state = SKIPPED


def step(run: Run, crew: Crew, *, root: Path | None = None) -> Run:
    """Advance the run by exactly one stage, persisting before and after.

    One stage per call so the caller keeps control: the shell can render
    between stages, a background job can notice cancellation, and a crash
    costs at most the stage that was in flight.
    """
    if run.finished:
        return run
    stage = run.current
    if stage is None:
        run.state = COMPLETE
        save(run, root)
        return run

    stage.state = RUNNING
    stage.attempts += 1
    stage.started = stage.started or time.time()
    save(run, root)  # persisted BEFORE the work, so a crash is visible as `running`

    handler = _HANDLERS.get(stage.name)
    if handler is None:
        done = Done(error=f"no handler for stage {stage.name!r}")
    else:
        try:
            done = handler(run, crew)
        except Exception as exc:  # a handler fault is the stage's outcome, not a crash
            done = Done(error=f"{type(exc).__name__}: {exc}")

    stage.output = done.output
    stage.error = done.error
    stage.finished = time.time()

    if done.refuse:
        stage.state = DONE
        run.state = REFUSED
        _skip_rest(run, stage)
        save(run, root)
        return run

    if done.error and done.carry:
        # `blocked` is a settled state, so the run moves on to the next stage
        # while the record still says this one failed.  Marking it `failed`
        # would leave it unsettled and the driver would run it again.
        stage.state = BLOCKED
        if run.current is None:
            run.state = COMPLETE
        save(run, root)
        return run

    if not done.error:
        stage.state = DONE
        if run.current is None:
            run.state = COMPLETE
        save(run, root)
        return run

    stage.state = FAILED
    run.state = STOPPED
    run.error = done.error
    _skip_rest(run, stage)
    save(run, root)
    return run


def drive(run: Run, crew: Crew, *, limit: int = LIMIT, root: Path | None = None) -> Run:
    """Step until the run settles or `limit` stages have run."""
    for _ in range(max(1, limit)):
        if run.finished:
            break
        step(run, crew, root=root)
    return run


def start(
    issue: int,
    crew: Crew,
    *,
    cwd: Path | str = ".",
    verify: str = "",
    base: str = "main",
    limit: int = LIMIT,
    root: Path | None = None,
) -> Run:
    """Create a run and drive it to settlement."""
    run = create(issue, cwd=cwd, verify=verify, base=base, root=root)
    return drive(run, crew, limit=limit, root=root)


def resume(run_id: str, crew: Crew, *, limit: int = LIMIT, root: Path | None = None) -> tuple[Run | None, str]:
    """Continue a run from disk.  Settled stages are never re-run.

    A stage left `running` by a crash is returned to `pending`: the process
    died before recording an outcome, so the only honest reading is that the
    attempt did not happen.  A `failed` stage is reset too, because the point
    of resuming a stopped run is to retry the one call that broke - a token
    that has since been fixed, a forge that was down - without paying for the
    model calls that already succeeded.
    """
    run = load(run_id, root)
    if run is None:
        return None, f"no run {run_id!r}"
    if run.state == COMPLETE:
        return run, f"run {run.id} is already complete"
    if run.state == REFUSED:
        return run, f"run {run.id} was refused: {run.refusal}"
    for stage in run.stages:
        if stage.state in (RUNNING, FAILED):
            stage.state = PENDING
        elif stage.state == SKIPPED:
            # Skipped stages were only skipped because a later-abandoned stage
            # failed; leaving them settled would open a pull request with no
            # restatement in it.
            stage.state = PENDING
    run.state = ACTIVE
    run.error = ""
    save(run, root)
    return drive(run, crew, limit=limit, root=root), ""


def status(run_id: str = "", *, root: Path | None = None) -> list[str]:
    """One run's report, or a line per run, newest first."""
    found = listing(root)
    if run_id:
        match = next((r for r in found if r.id == run_id or r.id.startswith(run_id)), None)
        if match is None:
            return [f"no run {run_id!r}"]
        return match.report()
    if not found:
        return ["no issue runs yet", "start one with /issue <n>"]
    return [r.summary() for r in found[:20]]


# -- the real collaborators -------------------------------------------------


def shell_runner(*, timeout: float = VERIFY_TIMEOUT) -> Runner:
    """A `Runner` over `subprocess`, with stderr folded into the output.

    `shell=False` after `shlex.split`: the commands come from a verify setting
    and from a branch name derived from an issue title, and an issue title is
    attacker-controlled text.
    """

    def run(command: str, cwd: Path) -> tuple[int, str]:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return 1, f"could not parse {command!r}: {exc}"
        if not argv:
            return 1, "empty command"
        try:
            proc = subprocess.run(
                argv,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout,
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return 1, f"{command!r} did not finish within {timeout:.0f}s"
        except OSError as exc:
            return 1, f"could not run {command!r}: {exc}"
        return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()

    return run


def shell_think(state: Any) -> Think:
    """A `Think` over the session's own agent.

    Goes through `state.agent.send` rather than a bare provider call so the
    implement stage has the tools it needs to actually edit files, and so the
    whole run appears in the transcript the user is looking at.
    """

    def think(stage: str, prompt: str) -> tuple[str, str]:
        system = SYSTEMS.get(stage, "")
        result = state.agent.send(f"{system}\n\n{prompt}" if system else prompt)
        return (result.text or "").strip(), result.error or ""

    return think


class GitHubForge:
    """`core.forge.Forge` behind the three calls this pipeline makes."""

    __slots__ = ("linked",)

    def __init__(self, linked: Any) -> None:
        self.linked = linked

    def read_issue(self, number: int) -> Any:
        return self.linked.call("GET", f"{{repo}}/issues/{int(number)}")

    def comment(self, number: int, body: str) -> Any:
        return self.linked.call("POST", f"{{repo}}/issues/{int(number)}/comments", body={"body": body})

    def open_pr(self, *, title: str, body: str, head: str, base: str, draft: bool) -> Any:
        return self.linked.create_pr(title=title, head=head, base=base, body=body, draft=draft)


def github_forge(workspace: Path) -> tuple[Forge | None, str]:
    """A GitHub-backed forge for the checkout at `workspace`, or why not."""
    from offset.core import forge as forge_mod

    linked = forge_mod.connect(workspace)
    if not linked.ok:
        return None, linked.error or "no github remote here"
    return GitHubForge(linked), ""


def forge_for(workspace: Path) -> tuple[Forge | None, str]:
    """The forge for this checkout, GitLab or GitHub, decided by its remote.

    GitLab is asked first because its detection is an exact host match - an
    explicit `GITLAB_HOST` or a `gitlab.*` domain - so it declines a GitHub
    checkout cleanly.  Asking GitHub first would send a GitLab-hosted issue to
    api.github.com and fail with a 404 that names the wrong problem, after the
    run had already created the branch.

    Both clients are built here, on the calling thread: they resolve a token
    and a config path from the environment, and a client that did that lazily
    would read them on whichever worker thread made the first request.
    """
    try:
        from offset.tools import gitlab as gitlab_mod
    except ImportError:
        # GitLab support is a tool module and may legitimately be absent; the
        # GitHub path below is the fallback, not a silent failure.
        return github_forge(workspace)

    found = gitlab_mod.detect(workspace)
    if not found.ok:
        return github_forge(workspace)
    # The detected remote goes back in: `connect` would otherwise run a second
    # `git remote -v`, which is the entire cost of the decision just made.
    client = gitlab_mod.connect(workspace, remote=found)
    if client.problem:
        return None, client.problem
    return gitlab_mod.gitlab_forge(client), ""


# -- the shell surface ------------------------------------------------------


def _issue(state: Any, args: list[str]) -> Any:
    """`/issue <n>`, `/issue resume <id>`, `/issue status [id]`."""
    from offset.shell.commands import TONE_INFO, TONE_OK, Outcome

    if not args:
        return Outcome.error("usage: /issue <number>", "or /issue resume <id>", "or /issue status")

    # Resolved here, on the calling thread, and closed over by the job below.
    # A daemon thread that calls `settings.home()` itself writes to the wrong
    # place once its caller has gone; the catalogue refresh had exactly that
    # bug.  Same for the workspace, which the shell may `cd` under us.
    root = runs_dir()
    workspace = Path(state.workspace)

    first = args[0].lower()

    if first == "status":
        return Outcome(status(args[1] if len(args) > 1 else "", root=root), TONE_INFO)

    if first == "resume":
        if len(args) < 2:
            return Outcome.error("usage: /issue resume <id>", "/issue status lists them")
        wanted = args[1]
        crew, why = _shell_crew(state, workspace)
        if crew is None:
            return Outcome.error(why)

        def resume_job() -> Any:
            run, error = resume(wanted, crew, root=root)
            if run is None:
                return Outcome.error(error)
            return Outcome(run.report(), TONE_OK if run.state == COMPLETE else TONE_INFO)

        return Outcome([f"resuming {wanted}..."], TONE_INFO, job=resume_job)

    number = first.lstrip("#")
    if not number.isdigit():
        return Outcome.error(f"{args[0]!r} is not an issue number", "usage: /issue <number>")

    crew, why = _shell_crew(state, workspace)
    if crew is None:
        return Outcome.error(why)
    base = _base_branch(workspace)
    created = create(
        int(number),
        cwd=workspace,
        verify=getattr(state, "verify_command", "") or "",
        base=base,
        root=root,
    )

    def job() -> Any:
        run = drive(created, crew, root=root)
        return Outcome(run.report(), TONE_OK if run.state == COMPLETE else TONE_INFO)

    return Outcome(
        [f"issue #{number} -> pull request, run {created.id[:10]}", f"{len(PIPELINE)} stages, base {base}"],
        TONE_INFO,
        job=job,
    )


def _shell_crew(state: Any, workspace: Path) -> tuple[Crew | None, str]:
    forge, why = forge_for(workspace)
    if forge is None:
        return None, why
    return Crew(forge=forge, think=shell_think(state), runner=shell_runner()), ""


def _base_branch(workspace: Path) -> str:
    """Trunk, decided without a fetch, falling back to a name rather than
    nothing: an empty base makes the forge reject the pull request at the end
    of a run that has already done all its work."""
    from offset.core import vcs

    branch = vcs.default_branch(workspace)
    return getattr(branch, "name", "") or "main"


def issue_commands() -> list[Any]:
    from offset.shell.commands import Command

    return [
        Command(
            "issue",
            "read an issue and open the pull request that closes it",
            _issue,
            usage="/issue <number> | /issue resume <id> | /issue status [id]",
        ),
    ]


_COMMANDS: list[Any] = []


def __getattr__(name: str) -> Any:
    """`COMMANDS` on demand.

    Built lazily because the handlers import from `offset.shell.commands`,
    which imports this module: resolving at import time would be a cycle.  The
    re-check after building is the guard `core.tasks` needs for the same
    reason - importing the shell registry re-enters here before the outer call
    has stored anything, so a single access would otherwise build two lists and
    register the command twice.
    """
    if name == "COMMANDS":
        if not _COMMANDS:
            built = issue_commands()
            if not _COMMANDS:
                _COMMANDS.extend(built)
        return _COMMANDS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
