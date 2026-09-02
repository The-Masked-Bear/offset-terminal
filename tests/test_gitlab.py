"""GitLab support.

The bugs these tests defend against are the ones that make a forge client
untrustworthy rather than merely broken.  A listing that silently stops at page
one is worse than an error, because nobody can tell.  A 401 reported as "401"
sends the user hunting for a wrong URL when the real answer is a token scope.
A credential that reaches an error message reaches the transcript, and from
there the model, the log and whatever the user pastes into a bug report.

Everything here uses an injected fetcher and an injected git.  Nothing reaches
a network and nothing needs a repository on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from offset.tools import gitlab as gl
from offset.tools.base import ToolContext

#: Shaped like a real personal access token so a leak is unmistakable.
TOKEN = "glpat-Sup3rSecretDoNotLog"


# -- fakes ------------------------------------------------------------------


class Server:
    """A fake GitLab.  Routes on (method, url fragment) and records every call."""

    def __init__(self) -> None:
        self.calls: list[gl.Call] = []
        self.routes: list[tuple[str, str, int, str, dict[str, str]]] = []

    def route(
        self,
        fragment: str,
        *,
        method: str = "GET",
        status: int = 200,
        body: object = None,
        headers: dict[str, str] | None = None,
    ) -> "Server":
        text = body if isinstance(body, str) else json.dumps(body if body is not None else [])
        self.routes.append((method.upper(), fragment, status, text, headers or {}))
        return self

    def __call__(self, call: gl.Call) -> gl.Answer:
        self.calls.append(call)
        for method, fragment, status, text, headers in self.routes:
            if method == call.method and fragment in call.url:
                return gl.Answer(status=status, headers=headers, text=text)
        raise AssertionError(f"unrouted {call.method} {call.url}")

    @property
    def urls(self) -> list[str]:
        return [c.url for c in self.calls]

    def body_of(self, index: int = -1) -> dict:
        raw = self.calls[index].body or b"{}"
        return json.loads(raw.decode("utf-8"))


def fake_git(output: str, seen: list[list[str]] | None = None):
    """A `Git` that answers with fixed `git remote -v` output."""

    def run(args, cwd: Path) -> str:
        if seen is not None:
            seen.append(list(args))
        return output

    return run


def client(server, *, token: str = TOKEN, host: str = "gitlab.com", path: str = "group/proj") -> gl.GitLab:
    return gl.GitLab(
        remote=gl.Remote(host=host, path=path, url=f"git@{host}:{path}.git", name="origin"),
        token=token,
        fetch=server,
    )


# -- remote parsing ---------------------------------------------------------


def test_an_https_gitlab_com_remote_is_parsed():
    """The plainest case, and the one the API base is derived from."""
    found = gl.parse_remote("https://gitlab.com/group/proj.git")
    assert found.ok
    assert found.host == "gitlab.com"
    assert found.path == "group/proj"
    assert found.api_base == "https://gitlab.com/api/v4"


def test_an_ssh_gitlab_com_remote_is_parsed():
    """`git@host:path` is not a URL, so it needs its own dialect."""
    found = gl.parse_remote("git@gitlab.com:group/proj.git")
    assert found.ok
    assert (found.host, found.path) == ("gitlab.com", "group/proj")


def test_a_self_hosted_host_is_recognised_by_its_name():
    found = gl.parse_remote("git@gitlab.example.org:team/tool.git")
    assert found.ok
    assert found.api_base == "https://gitlab.example.org/api/v4"


def test_a_self_hosted_host_without_gitlab_in_its_name_needs_the_override():
    """An unknown host cannot be sniffed, so the refusal names the variable
    that makes it work rather than guessing and 404-ing later."""
    refused = gl.parse_remote("git@git.acme.internal:team/tool.git")
    assert not refused.ok
    assert "GITLAB_HOST" in (refused.error or "")

    allowed = gl.parse_remote("git@git.acme.internal:team/tool.git", host_override="git.acme.internal")
    assert allowed.ok
    assert allowed.host == "git.acme.internal"


def test_a_non_gitlab_remote_is_refused():
    refused = gl.parse_remote("git@github.com:owner/repo.git")
    assert not refused.ok
    assert "not a GitLab host" in (refused.error or "")


def test_an_ssh_url_with_a_port_still_yields_an_https_api_base():
    """An ssh remote on 2222 serves its API on 443; carrying the port into the
    API base produced a connection refused that looked like an outage."""
    found = gl.parse_remote("ssh://git@gitlab.example.com:2222/team/tool.git")
    assert found.ok
    assert found.api_base == "https://gitlab.example.com/api/v4"


def test_a_subgroup_survives_the_round_trip():
    """GitLab projects nest.  Truncating to the last two segments addresses a
    project that does not exist."""
    found = gl.parse_remote("https://gitlab.com/group/sub/deeper/proj.git")
    assert found.path == "group/sub/deeper/proj"
    assert found.project == "group%2Fsub%2Fdeeper%2Fproj"
    assert found.repo == "proj"
    assert found.namespace == "group/sub/deeper"


def test_a_trailing_slash_and_a_missing_dot_git_are_both_accepted():
    assert gl.parse_remote("https://gitlab.com/group/proj/").path == "group/proj"
    assert gl.parse_remote("gitlab.com/group/proj").path == "group/proj"


@pytest.mark.parametrize("url", ["", "gitlab.com", "not a url", "ftp://gitlab.com/g/p"])
def test_an_unusable_remote_is_an_error_value_not_an_exception(url):
    found = gl.parse_remote(url)
    assert not found.ok
    assert found.error


# -- detection across several remotes ---------------------------------------


REMOTES = """\
origin\tgit@github.com:owner/mirror.git (fetch)
origin\tgit@github.com:owner/mirror.git (push)
gitlab\thttps://gitlab.com/group/proj.git (fetch)
gitlab\thttps://gitlab.com/group/proj.git (push)
"""


def test_detection_finds_the_gitlab_remote_when_origin_is_a_github_mirror():
    """The reason this reads every remote instead of asking for `origin`: a
    GitHub mirror on origin with GitLab on a second remote is a real setup."""
    found = gl.detect(Path("/nowhere"), git=fake_git(REMOTES), env={})
    assert found.ok
    assert (found.name, found.path) == ("gitlab", "group/proj")


def test_detection_prefers_origin_when_origin_is_itself_gitlab():
    text = (
        "upstream\tgit@gitlab.com:group/upstream.git (fetch)\n"
        "origin\tgit@gitlab.com:me/fork.git (fetch)\n"
    )
    found = gl.detect(Path("/nowhere"), git=fake_git(text), env={})
    assert (found.name, found.path) == ("origin", "me/fork")


def test_detection_honours_the_host_override_for_a_self_hosted_instance():
    text = "origin\tgit@git.acme.internal:team/tool.git (fetch)\n"
    env = {gl.HOST_ENV: "git.acme.internal"}
    assert gl.detect(Path("/nowhere"), git=fake_git(text), env=env).ok
    assert not gl.detect(Path("/nowhere"), git=fake_git(text), env={}).ok


def test_no_remote_at_all_says_so():
    found = gl.detect(Path("/nowhere"), git=fake_git(""), env={})
    assert not found.ok
    assert "no git remotes" in (found.error or "")


def test_a_repository_with_only_a_github_remote_names_what_it_found():
    text = "origin\tgit@github.com:owner/repo.git (fetch)\n"
    found = gl.detect(Path("/nowhere"), git=fake_git(text), env={})
    assert not found.ok
    assert "origin -> git@github.com:owner/repo.git" in (found.error or "")
    assert gl.HOST_ENV in (found.error or "")


def test_the_token_is_never_passed_to_git():
    """Detection shells out; argv is world-readable, so nothing secret goes
    anywhere near it."""
    seen: list[list[str]] = []
    gl.detect(Path("/nowhere"), git=fake_git(REMOTES, seen), env={"GITLAB_TOKEN": TOKEN})
    assert seen == [["remote", "-v"]]
    assert all(TOKEN not in arg for args in seen for arg in args)


def test_an_already_detected_remote_is_not_detected_twice():
    """A caller that ran `detect` to decide whether this is a GitLab checkout
    can hand the answer back rather than paying for a second subprocess."""
    seen: list[list[str]] = []
    git = fake_git(REMOTES, seen)
    found = gl.detect(Path("/nowhere"), git=git, env={})
    made = gl.connect(Path("/nowhere"), env={"GITLAB_TOKEN": TOKEN}, git=git, remote=found)
    assert made.ok
    assert made.remote is found
    assert seen == [["remote", "-v"]]


# -- credentials ------------------------------------------------------------


def test_a_missing_token_names_the_environment_variable(tmp_path):
    """"Unauthenticated" tells a user nothing they can act on."""
    made = gl.connect(
        tmp_path,
        env={"HOME": str(tmp_path), "GLAB_CONFIG_DIR": str(tmp_path)},
        git=fake_git("origin\tgit@gitlab.com:group/proj.git (fetch)\n"),
    )
    assert not made.ok
    assert "GITLAB_TOKEN" in made.problem
    assert "OFFSET_GITLAB_TOKEN" in made.problem
    assert "api" in made.problem


def test_either_environment_variable_carries_the_token(tmp_path):
    git = fake_git("origin\tgit@gitlab.com:group/proj.git (fetch)\n")
    for key in gl.TOKEN_KEYS:
        made = gl.connect(tmp_path, env={key: TOKEN, "HOME": str(tmp_path)}, git=git)
        assert made.ok, key
        assert made.token == TOKEN


def test_the_offset_specific_variable_wins(tmp_path):
    """So a user can point offset at a narrower token without disturbing the
    `GITLAB_TOKEN` their shell already exports for glab."""
    made = gl.connect(
        tmp_path,
        env={"GITLAB_TOKEN": "wide", "OFFSET_GITLAB_TOKEN": "narrow", "HOME": str(tmp_path)},
        git=fake_git("origin\tgit@gitlab.com:group/proj.git (fetch)\n"),
    )
    assert made.token == "narrow"


def test_glabs_own_config_is_read_when_the_environment_is_silent(tmp_path):
    (tmp_path / "config.yml").write_text(
        "hosts:\n"
        "  gitlab.com:\n"
        "    api_host: gitlab.com\n"
        f"    token: {TOKEN}\n"
        "    git_protocol: ssh\n"
        "  gitlab.example.org:\n"
        "    token: other-token\n",
        encoding="utf-8",
    )
    assert gl.read_glab_token(tmp_path / "config.yml", "gitlab.com") == TOKEN
    assert gl.read_glab_token(tmp_path / "config.yml", "gitlab.example.org") == "other-token"
    assert gl.read_glab_token(tmp_path / "config.yml", "gitlab.absent.org") is None


def test_a_missing_or_unparseable_glab_config_is_not_fatal(tmp_path):
    assert gl.read_glab_token(tmp_path / "nope.yml") is None
    (tmp_path / "junk.yml").write_text("{not yaml at all", encoding="utf-8")
    assert gl.read_glab_token(tmp_path / "junk.yml") is None


# -- requests ---------------------------------------------------------------


def test_the_token_travels_in_a_header_and_never_in_a_url():
    """A URL ends up in error messages, logs and the transcript; a header does
    not.  `PRIVATE-TOKEN` is also the only header a personal access token is
    accepted in - sending it as a Bearer earns a 401 that looks like expiry."""
    server = Server().route("issues", body=[])
    made = client(server)
    made.issues()
    call = server.calls[0]
    assert call.headers[gl.TOKEN_HEADER] == TOKEN
    assert TOKEN not in call.url
    assert "private_token" not in call.url


def test_the_project_path_is_encoded_for_the_api():
    server = Server().route("issues", body=[])
    made = client(server, path="group/sub/proj")
    made.issues()
    assert "projects/group%2Fsub%2Fproj/issues" in server.calls[0].url


# -- pagination -------------------------------------------------------------


def paging_server(total: int = 300, per_page: int = 100) -> Server:
    """A GitLab that pages exactly as the real one does: a full page plus an
    `X-Next-Page` header, and an empty header on the last page."""
    server = Server()
    pages = (total + per_page - 1) // per_page
    for index in range(1, pages + 1):
        start = (index - 1) * per_page
        chunk = [{"iid": n, "title": f"issue {n}", "state": "opened"}
                 for n in range(start + 1, min(start + per_page, total) + 1)]
        nxt = str(index + 1) if index < pages else ""
        # `&page=` and not `page=`: the bare form is also a substring of
        # `per_page=100`, so page two would be served page one's body.
        server.route(f"&page={index}", body=chunk, headers={"X-Next-Page": nxt, "X-Total-Pages": str(pages)})
    return server


def test_pagination_follows_the_next_page_header():
    """The silent-truncation bug: three hundred issues must not come back as
    the twenty or the hundred of the first page."""
    server = paging_server(300)
    got = client(server).issues(limit=1000)
    assert got.ok
    assert len(got.items) == 300
    assert got.pages == 3
    assert [i["iid"] for i in got.items[:2]] == [1, 2]
    assert got.items[-1]["iid"] == 300


def test_a_lower_case_next_page_header_is_still_followed():
    """Whatever proxy sits in front of GitLab capitalises headers its own way."""
    server = Server()
    server.route("&page=1", body=[{"iid": 1}], headers={"x-next-page": "2"})
    server.route("&page=2", body=[{"iid": 2}], headers={"x-next-page": ""})
    got = client(server).issues()
    assert [i["iid"] for i in got.items] == [1, 2]


def test_an_explicit_limit_stops_the_walk_early():
    server = paging_server(300)
    got = client(server).issues(limit=100)
    assert len(got.items) == 100
    assert got.pages == 1


def test_a_header_that_never_clears_cannot_spin_forever():
    """`X-Next-Page` is server-controlled.  A proxy copying it onto every
    response would otherwise loop until the process was killed."""
    server = Server().route("page=", body=[{"iid": 1}], headers={"X-Next-Page": "99"})
    got = client(server).issues(limit=gl.MAX_ITEMS)
    assert got.ok
    assert len(server.calls) <= gl.MAX_PAGES


def test_an_empty_page_ends_the_walk():
    server = Server().route("page=1", body=[], headers={"X-Next-Page": "2"})
    got = client(server).issues()
    assert got.ok
    assert got.items == []
    assert len(server.calls) == 1


# -- failure reporting ------------------------------------------------------


def test_a_401_is_reported_as_a_scope_problem():
    """Reporting the number sends the user hunting for a wrong URL."""
    server = Server().route("issues", status=401, body={"message": "401 Unauthorized"})
    got = client(server).issues()
    assert not got.ok
    assert got.status == 401
    assert "'api' scope" in got.error
    assert "GITLAB_TOKEN" in got.error


def test_a_403_names_the_scope_the_action_needs():
    server = Server().route("notes", method="POST", status=403, body={"message": "403 Forbidden"})
    got = client(server).comment("mr", 7, "looks good")
    assert not got.ok
    assert "read_api" in got.error and "api" in got.error


def test_a_404_explains_that_gitlab_hides_private_projects_behind_it():
    server = Server().route("issues/9", status=404, body={"message": "404 Project Not Found"})
    got = client(server).issue(9)
    assert not got.ok
    assert "read_api" in got.error


def test_gitlabs_own_field_errors_survive_into_the_message():
    server = Server().route(
        "merge_requests", method="POST", status=400,
        body={"message": {"source_branch": ["can't be blank"]}},
    )
    got = client(server).create_merge_request(source="a", target="b", title="t")
    assert not got.ok
    assert "source_branch" in got.error and "can't be blank" in got.error


def test_a_transport_failure_is_a_sentence_not_an_exception():
    def dead(call):
        raise OSError("network is unreachable")

    got = client(dead).issues()
    assert not got.ok
    assert "network is unreachable" in got.error


def test_a_client_with_no_remote_refuses_before_making_a_request():
    server = Server()
    made = gl.GitLab(remote=gl.Remote(error="no git remotes are configured here"), fetch=server)
    got = made.issues()
    assert not got.ok
    assert "no git remotes" in got.error
    assert server.calls == []


# -- the token never escapes ------------------------------------------------


def test_the_token_never_appears_in_anything_a_call_returns():
    """A GitLab that quotes the offending header back in its error body is the
    leak this guards: `error`, `text` and the decoded payload all pass through
    the same scrubber."""
    leak = {"message": f"invalid token {TOKEN}", "detail": [f"PRIVATE-TOKEN: {TOKEN}"]}
    server = Server().route("issues", status=401, body=leak)
    got = client(server).issues()
    assert TOKEN not in (got.error or "")
    assert TOKEN not in (got.text or "")
    assert gl.REDACTED in got.error


def test_the_token_is_scrubbed_out_of_a_successful_payload_too():
    server = Server().route("issues", body=[{"iid": 1, "description": f"debug: {TOKEN}"}])
    got = client(server).issues()
    assert TOKEN not in json.dumps(got.data)
    assert gl.REDACTED in got.items[0]["description"]


def test_the_token_never_appears_in_anything_the_tool_returns(monkeypatch, tmp_path):
    server = Server().route("issues", status=401, body={"message": f"bad token {TOKEN}"})
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(server))
    out = gl.GitLabTool().run({"action": "issues"}, ToolContext(cwd=tmp_path))
    assert not out.ok
    for text in (out.content, out.error or "", out.display):
        assert TOKEN not in text


# -- merge requests ---------------------------------------------------------


def test_creating_a_merge_request_posts_the_expected_body():
    """GitLab names the fields source_branch and target_branch, not head and
    base; getting that wrong is a 400 that does not say which field was
    missing."""
    server = Server().route(
        "merge_requests", method="POST", status=201,
        body={"iid": 12, "web_url": "https://gitlab.com/group/proj/-/merge_requests/12"},
    )
    got = client(server).create_merge_request(
        source="fix/thing", target="main", title="Fix the thing", description="why"
    )
    assert got.ok
    assert got.get("iid") == 12
    call = server.calls[0]
    assert call.method == "POST"
    assert call.headers["Content-Type"] == "application/json"
    assert server.body_of() == {
        "source_branch": "fix/thing",
        "target_branch": "main",
        "title": "Fix the thing",
        "description": "why",
        "remove_source_branch": True,
        "squash": False,
    }


def test_a_merge_request_without_a_title_or_a_branch_is_refused_locally():
    """Refused before the round trip: GitLab's own error for this is a 400
    whose message does not name the missing field."""
    server = Server()
    made = client(server)
    assert not made.create_merge_request(source="a", target="b", title="  ").ok
    assert not made.create_merge_request(source="", target="b", title="t").ok
    assert server.calls == []


def test_reading_one_merge_request_renders_both_branches():
    server = Server().route(
        "merge_requests/12",
        body={"iid": 12, "title": "Fix", "state": "opened", "source_branch": "fix",
              "target_branch": "main", "author": {"username": "kim"},
              "detailed_merge_status": "mergeable", "description": "body"},
    )
    lines = gl.mr_report(client(server).merge_request(12).data)
    assert "!12 Fix" in lines[0]
    assert "fix -> main" in lines[2]
    assert "mergeable" in lines[2]


def test_a_comment_goes_to_the_right_collection():
    server = Server().route("notes", method="POST", status=201, body={"id": 5})
    made = client(server)
    assert made.comment("issue", 3, "hello").ok
    assert made.comment("mr", 4, "hello").ok
    assert "issues/3/notes" in server.urls[0]
    assert "merge_requests/4/notes" in server.urls[1]
    assert server.body_of() == {"body": "hello"}


def test_a_comment_on_neither_an_issue_nor_an_mr_is_refused():
    server = Server()
    got = client(server).comment("wiki", 1, "hello")
    assert not got.ok
    assert "neither" in got.error
    assert server.calls == []


@pytest.mark.parametrize(
    "typed, sent",
    [("open", "opened"), ("opened", "opened"), ("", "opened"), ("closed", "closed"),
     ("merged", "merged"), ("all", "all")],
)
def test_the_state_people_type_is_translated_to_the_one_gitlab_wants(typed, sent):
    server = Server().route("merge_requests", body=[])
    client(server).merge_requests(state=typed)
    assert f"state={sent}" in server.calls[0].url


# -- pipelines and job logs -------------------------------------------------


TRACE = "".join(
    [
        "section_start:1700000000:step_script\r\x1b[0K\x1b[32;1m$ pytest\x1b[0;m\n",
        *[f"noise line {n}\n" for n in range(400)],
        "E   assert 1 == 2\n",
        "\x1b[31;1mFAILED tests/test_thing.py::test_it\x1b[0;m\n",
        "section_end:1700000900:step_script\r\x1b[0K",
    ]
)


def test_a_failed_pipeline_jobs_log_tail_is_extracted():
    """Two calls, because GitLab has no "give me the failure" endpoint, and a
    tail because a real trace is megabytes of setup noise with the failure at
    the very end."""
    server = Server()
    server.route("pipelines/77/jobs", body=[
        {"id": 900, "name": "test", "stage": "test", "status": "success"},
        {"id": 901, "name": "lint", "stage": "check", "status": "failed"},
    ], headers={"X-Next-Page": ""})
    server.route("jobs/901/trace", body=TRACE)

    failed = client(server).failure(77, lines=5)
    assert failed.ok
    assert (failed.job, failed.name, failed.stage) == (901, "lint", "check")
    assert "FAILED tests/test_thing.py::test_it" in failed.text
    assert "assert 1 == 2" in failed.text
    assert len(failed.text.splitlines()) <= 5
    assert "noise line 0" not in failed.text


def test_the_latest_attempt_of_a_retried_job_is_the_one_read():
    """A retried job leaves the old one in the listing, and the old log
    describes a failure that has already been fixed."""
    server = Server()
    server.route("pipelines/5/jobs", body=[
        {"id": 10, "name": "test", "stage": "test", "status": "failed"},
        {"id": 42, "name": "test", "stage": "test", "status": "failed"},
    ], headers={"X-Next-Page": ""})
    server.route("jobs/42/trace", body="the current failure")
    assert client(server).failure(5).job == 42


def test_a_pipeline_with_no_failed_job_says_so_rather_than_guessing():
    server = Server().route("pipelines/5/jobs", body=[{"id": 1, "status": "success"}],
                            headers={"X-Next-Page": ""})
    failed = client(server).failure(5)
    assert not failed.ok
    assert "no failed job" in failed.error


def test_the_trace_is_stripped_of_ansi_and_section_noise():
    """Colour codes and section markers are a third of the bytes of a real
    trace and none of its information."""
    cleaned = gl.clean_trace(TRACE)
    assert "\x1b[" not in cleaned
    assert "section_start" not in cleaned and "section_end" not in cleaned
    assert "$ pytest" in cleaned.splitlines()[0]


def test_the_tail_is_the_end_of_the_log():
    assert gl.tail("a\nb\nc\nd", 2) == "c\nd"
    assert gl.tail("a\nb", 10) == "a\nb"
    assert gl.tail("a\nb", 0) == ""


def test_pipelines_are_listed_newest_first_for_one_ref():
    server = Server().route("pipelines", body=[
        {"id": 9, "status": "failed", "ref": "main", "sha": "abcdef1234"},
    ], headers={"X-Next-Page": ""})
    got = client(server).pipelines(ref="main")
    assert got.ok
    assert "ref=main" in server.calls[0].url
    assert "sort=desc" in server.calls[0].url
    assert gl.pipeline_line(got.items[0]).startswith("#9 failed")


# -- the tool ---------------------------------------------------------------


def tool_run(monkeypatch, tmp_path, made: gl.GitLab, args: dict):
    monkeypatch.setattr(gl, "connect", lambda *a, **k: made)
    return gl.GitLabTool().run(args, ToolContext(cwd=tmp_path))


def test_the_tool_reports_a_missing_remote_before_doing_anything(monkeypatch, tmp_path):
    server = Server()
    made = gl.GitLab(remote=gl.Remote(error="no git remotes are configured here"), fetch=server)
    out = tool_run(monkeypatch, tmp_path, made, {"action": "issues"})
    assert not out.ok
    assert "no git remotes" in out.error
    assert server.calls == []


def test_the_tool_lists_issues_across_pages(monkeypatch, tmp_path):
    out = tool_run(monkeypatch, tmp_path, client(paging_server(300)), {"action": "issues", "limit": 1000})
    assert out.ok
    assert "300 opened issue(s) over 3 page(s)" in out.content


def test_the_tool_opens_a_merge_request_with_explicit_branches(monkeypatch, tmp_path):
    server = Server().route("merge_requests", method="POST", status=201,
                            body={"iid": 4, "web_url": "https://gitlab.com/g/p/-/merge_requests/4"})
    out = tool_run(monkeypatch, tmp_path, client(server), {
        "action": "create_mr", "title": "Do it", "source_branch": "work", "target_branch": "main",
    })
    assert out.ok
    assert "opened !4" in out.content
    assert server.body_of()["source_branch"] == "work"


def test_the_tool_refuses_a_merge_request_onto_itself(monkeypatch, tmp_path):
    server = Server()
    out = tool_run(monkeypatch, tmp_path, client(server), {
        "action": "create_mr", "title": "Do it", "source_branch": "main", "target_branch": "main",
    })
    assert not out.ok
    assert "two branches" in out.error
    assert server.calls == []


def test_the_tool_reads_a_failed_pipelines_log(monkeypatch, tmp_path):
    server = Server()
    server.route("pipelines/77/jobs", body=[{"id": 901, "name": "lint", "stage": "check", "status": "failed"}],
                 headers={"X-Next-Page": ""})
    server.route("jobs/901/trace", body=TRACE)
    out = tool_run(monkeypatch, tmp_path, client(server), {"action": "log", "pipeline_id": 77})
    assert out.ok
    assert "job 901 lint (check) failed" in out.content
    assert "FAILED tests/test_thing.py::test_it" in out.content


def test_the_tool_needs_an_id_for_the_actions_that_take_one(monkeypatch, tmp_path):
    server = Server()
    made = client(server)
    for args in ({"action": "issue"}, {"action": "mr"}, {"action": "jobs"}, {"action": "log"}):
        out = tool_run(monkeypatch, tmp_path, made, args)
        assert not out.ok, args
    assert server.calls == []


def test_an_unknown_action_lists_the_ones_that_exist(monkeypatch, tmp_path):
    out = tool_run(monkeypatch, tmp_path, client(Server()), {"action": "merge"})
    assert not out.ok
    assert "unknown action" in out.error
    assert "create_mr" in out.error


def test_the_tool_is_declared_as_a_writer():
    """Most actions read, but `create_mr` and `comment` are side effects other
    people see, and a tool's danger is the worst thing it can do."""
    tool = gl.gitlab_tools()[0]
    assert tool.name == "gitlab"
    assert tool.danger == gl.Danger.WRITE
    assert "create_mr" in tool.schema["properties"]["action"]["enum"]


# -- the forge adapter ------------------------------------------------------


def test_the_adapter_aliases_gitlabs_field_names_for_a_forge_agnostic_caller():
    """A caller that drives an issue to a merge request reads `body` and
    `number`; GitLab sends `description` and `iid`."""
    server = Server().route("issues/7", body={"iid": 7, "title": "Broken", "description": "detail"})
    got = gl.gitlab_forge(client(server)).read_issue(7)
    assert got.ok
    assert got.data["body"] == "detail"
    assert got.data["number"] == 7
    assert got.data["description"] == "detail"


def test_the_adapter_comments_on_the_issue_not_the_merge_request():
    server = Server().route("notes", method="POST", status=201, body={"id": 1})
    assert gl.gitlab_forge(client(server)).comment(7, "on it").ok
    assert "issues/7/notes" in server.urls[0]


def test_the_adapter_translates_head_and_base_into_gitlabs_field_names():
    server = Server().route("merge_requests", method="POST", status=201,
                            body={"iid": 9, "web_url": "https://gitlab.com/g/p/-/merge_requests/9"})
    got = gl.gitlab_forge(client(server)).open_pr(
        title="Fix it", body="closes #7", head="fix/7", base="main"
    )
    assert got.ok
    assert server.body_of()["source_branch"] == "fix/7"
    assert server.body_of()["target_branch"] == "main"
    assert server.body_of()["description"] == "closes #7"


def test_a_draft_merge_request_is_marked_in_its_title():
    """GitLab has no draft field to post; the flag would otherwise vanish."""
    server = Server().route("merge_requests", method="POST", status=201, body={"iid": 9})
    forge = gl.gitlab_forge(client(server))
    forge.open_pr(title="Fix it", head="fix", base="main", draft=True)
    assert server.body_of()["title"] == "Draft: Fix it"
    forge.open_pr(title="Draft: already", head="fix", base="main", draft=True)
    assert server.body_of()["title"] == "Draft: already"


# -- the commands -----------------------------------------------------------


def test_the_commands_are_built_lazily_and_only_once():
    """Built on attribute access because the handlers import the shell
    registry, which imports the tool subsystems."""
    first = gl.COMMANDS
    assert [c.name for c in first] == ["mr", "mrs", "issues", "pipeline"]
    # `/issue` is deliberately not an alias here: the singular is left free
    # for the issue-to-merge-request command that owns that verb.
    assert "issue" not in {alias for c in first for alias in c.aliases}
    assert gl.COMMANDS is first


def test_an_unknown_module_attribute_still_raises():
    with pytest.raises(AttributeError):
        gl.nonexistent  # noqa: B018


def test_issues_command_says_what_is_wrong_without_a_remote(monkeypatch, tmp_path):
    made = gl.GitLab(remote=gl.Remote(error="no git remotes are configured here"), fetch=Server())
    monkeypatch.setattr(gl, "connect", lambda *a, **k: made)
    out = gl._issues(SimpleNamespace(workspace=tmp_path), [])
    assert out.tone == "err"
    assert "no git remotes" in out.lines[0]
    assert out.job is None


def test_issues_command_renders_the_listing_on_its_worker_thread(monkeypatch, tmp_path):
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(paging_server(300)))
    started = gl._issues(SimpleNamespace(workspace=tmp_path), [])
    assert started.job is not None
    done = started.job()
    assert "300 opened issue(s)" in done.lines[0]
    assert len(done.lines) - 1 <= gl.LIST_LIMIT


def test_issues_command_reads_one_issue_when_given_a_number(monkeypatch, tmp_path):
    server = Server().route("issues/7", body={"iid": 7, "title": "Broken", "state": "opened",
                                              "author": {"username": "kim"}, "description": "detail"})
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(server))
    done = gl._issues(SimpleNamespace(workspace=tmp_path), ["7"]).job()
    assert done.lines[0] == "#7 Broken"
    assert "issues/7" in server.urls[0]


def test_mrs_command_lists_merge_requests(monkeypatch, tmp_path):
    server = Server().route("merge_requests", body=[
        {"iid": 3, "state": "opened", "title": "Fix", "source_branch": "fix",
         "target_branch": "main", "author": {"username": "kim"}},
    ], headers={"X-Next-Page": ""})
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(server))
    done = gl._mrs(SimpleNamespace(workspace=tmp_path), ["opened"]).job()
    assert "1 opened merge request(s) in group/proj" in done.lines[0]
    assert done.lines[1].startswith("!3")


def test_pipeline_command_appends_the_failing_jobs_log(monkeypatch, tmp_path):
    server = Server()
    server.route("pipelines?", body=[{"id": 77, "status": "failed", "ref": "main", "sha": "deadbeefcafe"}],
                 headers={"X-Next-Page": ""})
    server.route("pipelines/77/jobs", body=[{"id": 901, "name": "lint", "stage": "check", "status": "failed"}],
                 headers={"X-Next-Page": ""})
    server.route("jobs/901/trace", body=TRACE)
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(server))
    done = gl._pipeline(SimpleNamespace(workspace=tmp_path), ["main"]).job()
    text = "\n".join(done.lines)
    assert "#77 failed" in text
    assert "job 901 lint (check) failed" in text


def test_pipeline_command_stops_at_the_listing_when_the_pipeline_passed(monkeypatch, tmp_path):
    server = Server().route("pipelines?", body=[
        {"id": 78, "status": "success", "ref": "main", "sha": "abc"},
    ], headers={"X-Next-Page": ""})
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(server))
    done = gl._pipeline(SimpleNamespace(workspace=tmp_path), ["main"]).job()
    assert done.lines == ["#78 success    main                     abc"]


def test_mr_command_refuses_a_dirty_tree(monkeypatch, tmp_path):
    """A merge request that does not contain the work the user is looking at
    is worse than an error message."""
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(Server()))
    monkeypatch.setattr(gl.vcs, "status", lambda cwd: SimpleNamespace(dirty=True))
    out = gl._mr(SimpleNamespace(workspace=tmp_path), [])
    assert out.tone == "err"
    assert "uncommitted" in out.lines[0]
    assert out.job is None


def test_mr_command_refuses_from_the_default_branch(monkeypatch, tmp_path):
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(Server()))
    monkeypatch.setattr(gl.vcs, "status", lambda cwd: SimpleNamespace(dirty=False))
    monkeypatch.setattr(gl.vcs, "current_branch", lambda cwd: SimpleNamespace(name="main"))
    monkeypatch.setattr(gl.vcs, "default_branch", lambda cwd: SimpleNamespace(name="main"))
    out = gl._mr(SimpleNamespace(workspace=tmp_path), [])
    assert out.tone == "err"
    assert "default branch" in out.lines[0]


def test_mr_command_pushes_then_opens_the_merge_request(monkeypatch, tmp_path):
    server = Server().route("merge_requests", method="POST", status=201,
                            body={"iid": 21, "web_url": "https://gitlab.com/g/p/-/merge_requests/21"})
    pushed: list[str] = []
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(server))
    monkeypatch.setattr(gl.vcs, "status", lambda cwd: SimpleNamespace(dirty=False))
    monkeypatch.setattr(gl.vcs, "current_branch", lambda cwd: SimpleNamespace(name="fix/thing"))
    monkeypatch.setattr(gl.vcs, "default_branch", lambda cwd: SimpleNamespace(name="main"))
    monkeypatch.setattr(
        gl.vcs, "push",
        lambda cwd, branch, **kw: pushed.append(branch) or SimpleNamespace(ok=True, error=None),
    )
    done = gl._mr(SimpleNamespace(workspace=tmp_path), ["Fix", "the", "thing"]).job()
    assert pushed == ["fix/thing"]
    assert "opened !21" in done.lines[0]
    assert server.body_of()["title"] == "Fix the thing"


def test_mr_command_takes_its_title_from_the_branchs_commits(monkeypatch, tmp_path):
    """No model call and no API key needed to open a merge request: the commit
    subjects already say what the branch does."""
    server = Server().route("merge_requests", method="POST", status=201, body={"iid": 22})
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(server))
    monkeypatch.setattr(gl.vcs, "status", lambda cwd: SimpleNamespace(dirty=False))
    monkeypatch.setattr(gl.vcs, "current_branch", lambda cwd: SimpleNamespace(name="fix/thing"))
    monkeypatch.setattr(gl.vcs, "default_branch", lambda cwd: SimpleNamespace(name="main"))
    monkeypatch.setattr(gl.vcs, "push", lambda cwd, branch, **kw: SimpleNamespace(ok=True, error=None))
    monkeypatch.setattr(gl.vcs, "log", lambda cwd, since=None, limit=0: SimpleNamespace(
        commits=(SimpleNamespace(subject="Fix the thing"), SimpleNamespace(subject="Add a test")),
    ))
    gl._mr(SimpleNamespace(workspace=tmp_path), []).job()
    body = server.body_of()
    assert body["title"] == "Fix the thing"
    assert body["description"] == "- Fix the thing\n- Add a test"


def test_mr_command_reports_a_failed_push_instead_of_opening_nothing(monkeypatch, tmp_path):
    server = Server()
    monkeypatch.setattr(gl, "connect", lambda *a, **k: client(server))
    monkeypatch.setattr(gl.vcs, "status", lambda cwd: SimpleNamespace(dirty=False))
    monkeypatch.setattr(gl.vcs, "current_branch", lambda cwd: SimpleNamespace(name="fix/thing"))
    monkeypatch.setattr(gl.vcs, "default_branch", lambda cwd: SimpleNamespace(name="main"))
    monkeypatch.setattr(gl.vcs, "push", lambda cwd, branch, **kw: SimpleNamespace(ok=False, error="rejected"))
    done = gl._mr(SimpleNamespace(workspace=tmp_path), ["Fix"]).job()
    assert done.tone == "err"
    assert "rejected" in done.lines[0]
    assert server.calls == []
