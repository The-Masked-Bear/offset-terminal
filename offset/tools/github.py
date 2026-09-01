"""The GitHub-native workflow: `/pr`, `/review`, `/fix-ci`, `/resolve-comments`.

`core/vcs` knows about git and `core/forge` knows about GitHub; this module is
the part that decides what to do with them, and it makes three judgements worth
stating.

**Reviewing prints by default.** Posting a review to a repository is a side
effect other people see, and an agent that posts on every invocation is a
liability. `/review` renders the review for the user to read; `/review 12 post`
is the extra word that publishes it.

**CI logs are excerpted, not forwarded.** A failed GitHub Actions run is
routinely megabytes; feeding that to a model wastes most of a context window on
setup noise to reach twenty relevant lines. `forge.extract_failure` finds the
error region, and only that is sent.

**A dirty tree refuses.** Opening a pull request from a working tree with
uncommitted changes produces a PR that does not contain the work the user is
looking at, which is worse than an error message.

Everything here is licence-gated: these are the Plus workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from offset.core import forge as forge_mod
from offset.core import vcs
from offset.providers.base import Message, Request
from offset.shell.commands import TONE_ERR, TONE_INFO, TONE_OK, Command, Outcome, ShellState
from offset.tools.base import Danger, Tool, ToolContext, ToolResult

#: Diff sent to a model when writing a PR body or a review.  Large enough for a
#: real change, small enough to leave room for the answer.
DIFF_BUDGET: Final = 24_000

#: Review comments to render before summarising.
COMMENT_LIMIT: Final = 30

PR_SYSTEM: Final = """You write pull request descriptions.
Reply with a title on the first line, then a blank line, then the body.
The title is under 72 characters, imperative, and names the change not the files.
The body explains why the change was made and what a reviewer should look at.
No preamble, no markdown headings, no bullet list of every file."""

REVIEW_SYSTEM: Final = """You review code changes.
Report only what matters: correctness bugs, missing error handling, security
issues, and broken invariants. Quote the file and line you mean.
If the change is fine, say so in one line rather than inventing concerns.
Never comment on formatting a linter would catch."""

FIX_SYSTEM: Final = """You are given the failing region of a CI log.
Identify the single root cause and state the change that fixes it.
Be specific about the file and the line. If the log is inconclusive, say what
further output would settle it rather than guessing."""


def _plus(feature: str) -> Outcome | None:
    """The licence gate, in the shape every premium command uses."""
    from offset.auth import require_plus

    if require_plus(feature):
        return None
    return Outcome.error(
        f"Offset Lite does not support /{feature}.",
        "Upgrade to Offset Plus via 'offset upgrade <key>'.",
    )


def _ask(state: ShellState, system: str, prompt: str, *, max_tokens: int = 1200) -> tuple[str, str]:
    """One model call through the session's own agent config.

    Returns `(text, error)`.  Uses the agent's model rather than the ensemble so
    a user with one key configured is not told to set up a roster first.
    """
    from offset.providers.auth import load as load_credential
    from offset.providers.base import TurnBuilder
    from offset.providers.registry import resolve

    model = getattr(state.agent.config, "model", None) or state.model
    try:
        provider, meta = resolve(model)
    except Exception as exc:
        return "", f"could not resolve {model!r}: {exc}"
    try:
        key = load_credential(provider)
    except Exception:
        key = None
    request = Request(
        model=meta.id,
        messages=[Message("user", prompt)],
        system=system,
        max_tokens=min(max_tokens, meta.max_output or max_tokens),
    )
    try:
        turn = TurnBuilder().consume(provider.stream(request, api_key=key)).finish()
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"
    if turn.error:
        return "", turn.error
    return (turn.text or "").strip(), ""


def _forge(state: ShellState) -> tuple[forge_mod.Forge | None, Outcome | None]:
    linked = forge_mod.connect(state.workspace)
    if not linked.ok:
        return None, Outcome.error(linked.error or "no github remote here")
    return linked, None


def _pr_number(state: ShellState, args: list[str], linked: forge_mod.Forge) -> tuple[int, str]:
    """The PR to act on: the argument, or the one for this branch."""
    for arg in args:
        if arg.isdigit():
            return int(arg), ""
    branch = vcs.current_branch(state.workspace)
    if not branch.name:
        return 0, "not on a branch, so there is no pull request to infer"
    reply = linked.find_pr(branch.name)
    if not reply.ok:
        return 0, reply.error or "could not look up a pull request for this branch"
    items = reply.payload if isinstance(reply.payload, list) else []
    if not items:
        return 0, f"no open pull request for {branch.name}; pass a number"
    return int(items[0].get("number", 0)), ""


# -- /pr --------------------------------------------------------------------


def _pr(state: ShellState, args: list[str]) -> Outcome:
    """Summarise the branch and open a pull request for it."""
    gate = _plus("pr")
    if gate is not None:
        return gate

    linked, problem = _forge(state)
    if problem is not None:
        return problem

    condition = vcs.status(state.workspace)
    if condition.dirty:
        return Outcome.error(
            "the working tree has uncommitted changes",
            "commit them first; a pull request would not contain them",
        )
    branch = vcs.current_branch(state.workspace)
    base = vcs.default_branch(state.workspace)
    if not branch.name:
        return Outcome.error("not on a branch")
    if branch.name == base.name:
        return Outcome.error(
            f"you are on {base.name}, the default branch",
            "make a branch for the change first",
        )

    def job() -> Outcome:
        changes = vcs.diff(state.workspace, base=base.name)
        if not changes.text.strip():
            return Outcome.error(f"no difference between {branch.name} and {base.name}")

        title = " ".join(a for a in args if not a.isdigit()).strip()
        body = ""
        if not title:
            answer, why = _ask(
                state, PR_SYSTEM,
                f"Branch: {branch.name}\nBase: {base.name}\n\nDiff:\n{changes.text[:DIFF_BUDGET]}",
            )
            if why:
                return Outcome.error(f"could not write a description: {why}")
            lines = answer.splitlines()
            title = lines[0].strip() if lines else branch.name
            body = "\n".join(lines[1:]).strip()

        pushed = vcs.push(state.workspace, branch.name, set_upstream=True)
        if not pushed.ok:
            return Outcome.error(pushed.error or "could not push the branch")

        reply = linked.create_pr(title=title, body=body, head=branch.name, base=base.name)
        if not reply.ok:
            return Outcome.error(reply.error or "could not open the pull request")
        url = ""
        if isinstance(reply.payload, dict):
            url = str(reply.payload.get("html_url") or "")
        return Outcome([f"opened: {url or title}", f"via {linked.via}"], TONE_OK)

    return Outcome([f"opening a pull request for {branch.name} -> {base.name}..."], TONE_INFO, job=job)


# -- /review ----------------------------------------------------------------


def _review(state: ShellState, args: list[str]) -> Outcome:
    """Review a pull request.  Prints unless told to post."""
    gate = _plus("review")
    if gate is not None:
        return gate

    linked, problem = _forge(state)
    if problem is not None:
        return problem
    number, why = _pr_number(state, args, linked)
    if why:
        return Outcome.error(why)
    post = any(a.lower() in ("post", "publish") for a in args)

    def job() -> Outcome:
        got = linked.pr_diff(number)
        if not got.ok:
            return Outcome.error(got.error or f"could not read the diff for #{number}")
        text = got.text or ""
        if not text.strip():
            return Outcome([f"#{number} has an empty diff"], TONE_INFO)

        answer, problem = _ask(
            state, REVIEW_SYSTEM, f"Pull request #{number}\n\n{text[:DIFF_BUDGET]}", max_tokens=2000
        )
        if problem:
            return Outcome.error(f"could not review: {problem}")
        if not post:
            return Outcome(
                [f"review of #{number} (not posted; add 'post' to publish)", "", *answer.splitlines()],
                TONE_OK,
            )
        sent = linked.post_review(number, body=answer, event="COMMENT")
        if not sent.ok:
            return Outcome.error(sent.error or "could not post the review")
        return Outcome([f"posted a review on #{number}"], TONE_OK)

    verb = "reviewing and posting to" if post else "reviewing"
    return Outcome([f"{verb} #{number}..."], TONE_INFO, job=job)


# -- /fix-ci ----------------------------------------------------------------


def _fix_ci(state: ShellState, args: list[str]) -> Outcome:
    """Find the failing check, excerpt its log, and say what broke."""
    gate = _plus("fix-ci")
    if gate is not None:
        return gate

    linked, problem = _forge(state)
    if problem is not None:
        return problem

    def job() -> Outcome:
        ref = vcs.head(state.workspace).sha or "HEAD"
        checks = linked.list_checks(ref)
        if not checks.ok:
            return Outcome.error(checks.error or "could not read the checks for this commit")
        failed = forge_mod.failing_checks(checks)
        if not failed:
            lines = [forge_mod.check_line(c) for c in _check_runs(checks)][:COMMENT_LIMIT]
            return Outcome(["no failing checks on this commit", *lines], TONE_OK)

        lines: list[str] = [f"{len(failed)} failing check(s) on {ref[:8]}"]
        for check in failed[:3]:
            lines.append("")
            lines.append(forge_mod.check_line(check))
            job_id = forge_mod.job_id_of(check)
            if job_id is None:
                lines.append("  (no log available for this check)")
                continue
            log = linked.get_check_logs(job_id)
            if not log.ok:
                lines.append(f"  could not fetch the log: {log.error}")
                continue
            excerpt = forge_mod.extract_failure(log.text or "")
            if not excerpt.text.strip():
                lines.append("  the log has no recognisable failure region")
                continue
            lines.append(f"  --- {excerpt.reason} ---")
            lines.extend(f"  {l}" for l in excerpt.text.splitlines()[-40:])
            answer, why = _ask(state, FIX_SYSTEM, excerpt.text[:DIFF_BUDGET])
            if not why and answer:
                lines.append("  --- diagnosis ---")
                lines.extend(f"  {l}" for l in answer.splitlines())
        return Outcome(lines, TONE_ERR)

    return Outcome(["reading the failing checks..."], TONE_INFO, job=job)


def _check_runs(reply: forge_mod.Reply) -> list[dict[str, Any]]:
    payload = reply.payload if isinstance(reply.payload, dict) else {}
    runs = payload.get("check_runs")
    return runs if isinstance(runs, list) else []


# -- /resolve-comments ------------------------------------------------------


def _resolve_comments(state: ShellState, args: list[str]) -> Outcome:
    """List unresolved review threads, and optionally answer them."""
    gate = _plus("resolve-comments")
    if gate is not None:
        return gate

    linked, problem = _forge(state)
    if problem is not None:
        return problem
    number, why = _pr_number(state, args, linked)
    if why:
        return Outcome.error(why)
    answer_them = any(a.lower() in ("reply", "answer", "resolve") for a in args)

    def job() -> Outcome:
        threads, problem = linked.list_review_threads(number)
        if problem:
            return Outcome.error(problem)
        open_threads = [t for t in threads if not t.resolved]
        if not open_threads:
            return Outcome([f"#{number} has no unresolved review threads"], TONE_OK)

        lines = [f"{len(open_threads)} unresolved thread(s) on #{number}"]
        for thread in open_threads[:COMMENT_LIMIT]:
            lines.append("")
            lines.append(f"{thread.path}:{thread.line or '?'}  {thread.author}")
            lines.extend(f"  {l}" for l in (thread.body or "").splitlines()[:6])
            if not answer_them:
                continue
            reply, why = _ask(
                state, REVIEW_SYSTEM,
                f"A reviewer said this about {thread.path}:{thread.line}:\n\n{thread.body}\n\n"
                "Reply to the reviewer in one short paragraph. If they are right, say what "
                "you changed. If they are mistaken, say why, politely.",
                max_tokens=400,
            )
            if why or not reply:
                lines.append(f"  (could not draft a reply: {why or 'empty answer'})")
                continue
            sent = linked.reply_to_comment(number, thread.comment_id, reply)
            if not sent.ok:
                lines.append(f"  (could not post: {sent.error})")
                continue
            done = linked.resolve_thread(thread.id)
            lines.append(f"  replied{' and resolved' if done.ok else ''}")
        if not answer_them:
            lines.append("")
            lines.append("add 'reply' to answer and resolve them")
        return Outcome(lines, TONE_INFO)

    return Outcome([f"reading review threads on #{number}..."], TONE_INFO, job=job)


COMMANDS: list[Command] = [
    Command("pr", "open a pull request for this branch", _pr, usage="/pr [title]"),
    Command("review", "review a pull request", _review, usage="/review [n] [post]"),
    Command("fix-ci", "find and diagnose failing checks", _fix_ci, usage="/fix-ci [n]"),
    Command("resolve-comments", "unresolved review threads", _resolve_comments,
            usage="/resolve-comments [n] [reply]", aliases=("comments",)),
]


# -- the model-facing tool --------------------------------------------------


class GitHub(Tool):
    """Read pull requests, checks and review comments."""

    name = "github"
    description = (
        "Read GitHub state for this repository: pull request details and diffs, CI check "
        "results with the failing region of the log excerpted, and review comments. "
        "Read-only; use the /pr and /review commands to create or post anything."
    )
    danger = Danger.SAFE
    parallel_safe = True
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["pr", "diff", "files", "checks", "logs", "comments", "status"],
                "description": "what to read",
            },
            "number": {"type": "integer", "minimum": 1, "description": "pull request number"},
            "job_id": {"type": "integer", "description": "for logs: the failing job id from checks"},
        },
        "required": ["action"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        return f"github {args.get('action', '?')} {args.get('number', '')}".strip()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(args.get("action", "")).strip()
        root = Path(getattr(ctx, "root", None) or ctx.cwd)
        linked = forge_mod.connect(root)
        if not linked.ok:
            return ToolResult.fail(linked.error or "no github remote here")

        if action == "status":
            ref = vcs.head(root).sha or "HEAD"
            reply = linked.list_checks(ref)
            if not reply.ok:
                return ToolResult.fail(reply.error or "could not read checks")
            runs = _check_runs(reply)
            if not runs:
                return ToolResult.text(f"no checks reported for {ref[:8]}")
            return ToolResult.text("\n".join(forge_mod.check_line(c) for c in runs[:COMMENT_LIMIT]))

        number = args.get("number")
        if action in ("pr", "diff", "files", "comments") and not isinstance(number, int):
            return ToolResult.fail(f"action {action!r} needs a pull request number")

        if action == "pr":
            reply = linked.get_pr(int(number))
            if not reply.ok:
                return ToolResult.fail(reply.error or "could not read the pull request")
            data = reply.payload if isinstance(reply.payload, dict) else {}
            lines = [
                f"#{data.get('number')} {data.get('title')}",
                f"state: {data.get('state')}  by {(data.get('user') or {}).get('login')}",
                f"{data.get('head', {}).get('ref')} -> {data.get('base', {}).get('ref')}",
                "",
                str(data.get("body") or "")[:2000],
            ]
            return ToolResult.text("\n".join(lines))

        if action == "diff":
            reply = linked.pr_diff(int(number))
            if not reply.ok:
                return ToolResult.fail(reply.error or "could not read the diff")
            return ToolResult.text((reply.text or "")[:DIFF_BUDGET])

        if action == "files":
            reply = linked.list_files(int(number))
            if not reply.ok:
                return ToolResult.fail(reply.error or "could not list the files")
            items = reply.payload if isinstance(reply.payload, list) else []
            lines = [
                f"{f.get('status', '?'):9s} +{f.get('additions', 0):<5} -{f.get('deletions', 0):<5} {f.get('filename')}"
                for f in items
            ]
            return ToolResult.text("\n".join(lines) or "no files")

        if action == "comments":
            threads, problem = linked.list_review_threads(int(number))
            if problem:
                return ToolResult.fail(problem)
            if not threads:
                return ToolResult.text("no review comments")
            lines = []
            for t in threads[:COMMENT_LIMIT]:
                mark = "resolved" if t.resolved else "open"
                lines.append(f"[{mark}] {t.path}:{t.line or '?'} {t.author}: {(t.body or '')[:200]}")
            return ToolResult.text("\n".join(lines))

        # logs
        job_id = args.get("job_id")
        if not isinstance(job_id, int):
            return ToolResult.fail("logs needs a job_id; get one from action=checks")
        reply = linked.get_check_logs(job_id)
        if not reply.ok:
            return ToolResult.fail(reply.error or "could not fetch the log")
        excerpt = forge_mod.extract_failure(reply.text or "")
        if not excerpt.text.strip():
            return ToolResult.text("the log has no recognisable failure region")
        return ToolResult.text(f"--- {excerpt.reason} ---\n{excerpt.text}")


def github_tools() -> list[Tool]:
    return [GitHub()]
