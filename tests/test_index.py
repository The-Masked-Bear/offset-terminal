"""The workspace index and the retrieval built on it.

Every test runs against a real temporary repository and a real SQLite file: the
properties worth pinning here are incrementality and exclusion, and both are
about what the code does to the filesystem rather than about ranking maths.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from offset.core.index import Index, split_identifier, words_of
from offset.tools.base import ToolContext
from offset.tools.retrieve import bm25, expand, rank, retrieve_tools, select_context


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFSET_HOME", str(tmp_path / "home"))
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "auth.py").write_text(
        "import hashlib\n"
        "\n"
        "SALT = 'pepper'\n"
        "\n"
        "def get_user_authentication(token):\n"
        "    '''Verify a token and return the user.'''\n"
        "    return hashlib.sha256(token.encode()).hexdigest()\n"
        "\n"
        "class Session:\n"
        "    def refresh(self):\n"
        "        return get_user_authentication('x')\n",
        encoding="utf-8",
    )
    (root / "pkg" / "views.py").write_text(
        "from pkg.auth import get_user_authentication\n"
        "\n"
        "def index(request):\n"
        "    return get_user_authentication(request.token)\n",
        encoding="utf-8",
    )
    (root / "pkg" / "unrelated.py").write_text(
        "def render_template(name):\n    return name\n", encoding="utf-8"
    )
    (root / "notes.md").write_text("# notes\nnothing to see\n", encoding="utf-8")
    return root


@pytest.fixture
def index(repo):
    made = Index(repo)
    made.refresh()
    yield made
    made.close()


def test_indexing_twice_reparses_nothing(repo):
    made = Index(repo)
    first = made.refresh()
    assert first.parsed > 0, "the first pass has to do the work"
    second = made.refresh()
    assert second.scanned == first.scanned
    assert second.parsed == 0, f"a warm index re-parsed {second.parsed} file(s)"
    made.close()


def test_a_changed_file_is_reparsed_and_others_are_not(index, repo):
    (repo / "pkg" / "views.py").write_text(
        "def index(request):\n    return 1\n", encoding="utf-8"
    )
    stats = index.refresh()
    assert stats.parsed == 1, f"expected one re-parse, got {stats.parsed}"


def test_a_touched_but_unchanged_file_is_not_reparsed(index, repo):
    """mtime alone would re-parse this; the content hash is what prevents it."""
    target = repo / "pkg" / "auth.py"
    text = target.read_text()
    target.write_text(text, encoding="utf-8")  # same bytes, new mtime
    stats = index.refresh()
    assert stats.parsed == 0, "identical content must not be re-parsed"


def test_a_deleted_file_is_removed_from_the_index(index, repo):
    (repo / "pkg" / "unrelated.py").unlink()
    stats = index.refresh()
    assert stats.removed == 1
    assert index.file("pkg/unrelated.py") is None


def test_skipped_directories_are_never_walked(repo):
    junk = repo / "node_modules" / "left-pad"
    junk.mkdir(parents=True)
    (junk / "index.js").write_text("module.exports = 1\n", encoding="utf-8")
    cache = repo / "__pycache__"
    cache.mkdir()
    (cache / "auth.cpython-311.pyc").write_bytes(b"\x00\x01")

    made = Index(repo)
    made.refresh()
    paths = made.files()
    assert not any("node_modules" in p for p in paths), paths
    assert not any("__pycache__" in p for p in paths), paths
    made.close()


def test_gitignored_paths_are_excluded(repo):
    (repo / ".gitignore").write_text("secret.py\nbuilt/\n", encoding="utf-8")
    (repo / "secret.py").write_text("KEY = 'x'\n", encoding="utf-8")
    built = repo / "built"
    built.mkdir()
    (built / "out.py").write_text("x = 1\n", encoding="utf-8")

    made = Index(repo)
    made.refresh()
    paths = made.files()
    assert "secret.py" not in paths, paths
    assert not any(p.startswith("built/") for p in paths), paths
    made.close()


def test_python_symbols_are_ast_exact(index):
    names = {r["name"] for r in index.symbols_like("", limit=500)}
    assert "get_user_authentication" in names
    assert "Session" in names
    assert "refresh" in names, "a method must be extracted, not just its class"
    assert "SALT" in names, "a module-level assignment is a symbol"


def test_the_reverse_import_edge_is_available(index):
    graph = index.graph()
    # `imports_of` gives the statements as written; `dependencies_of` resolves
    # them to files in the tree, which is the edge worth asserting.
    assert "pkg/auth.py" in graph.dependencies_of("pkg/views.py")
    assert "pkg/views.py" in graph.importers_of("pkg/auth.py"), (
        "importers_of is the dependency-awareness edge"
    )

def test_a_large_file_is_recorded_but_not_parsed(repo):
    from offset.core.index import MAX_BYTES

    (repo / "huge.py").write_text("x = 1\n" * (MAX_BYTES // 4), encoding="utf-8")
    made = Index(repo)
    made.refresh()
    record = made.file("huge.py")
    assert record is not None, "the path should still be known"
    assert not record.parsed
    assert "large" in record.error
    made.close()


# -- identifier handling ----------------------------------------------------


def test_identifiers_split_on_camel_case_and_underscores():
    assert split_identifier("getUserAuth") == ["get", "user", "auth"]
    assert split_identifier("HTTP_PORT") == ["http", "port"]
    assert split_identifier("parseHTTPResponse") == ["parse", "http", "response"]


def test_words_of_yields_both_the_whole_identifier_and_its_pieces():
    found = set(words_of("def getUserAuth(): pass"))
    assert "getuserauth" in found
    assert {"get", "user", "auth"} <= found


def test_query_expansion_covers_the_compound_form():
    assert set(expand("getUserAuth")) >= {"getuserauth", "get", "user", "auth"}


# -- ranking ----------------------------------------------------------------


def test_a_camel_case_query_finds_the_snake_case_definition(index):
    found = rank(index, "getUserAuth", limit=5)
    paths = [r.path for r in found]
    assert "pkg/auth.py" in paths, paths


def test_bm25_prefers_many_occurrences_of_a_rare_term(repo):
    (repo / "rare_many.py").write_text("zygote\n" * 12, encoding="utf-8")
    (repo / "rare_one.py").write_text("zygote\nfiller\n", encoding="utf-8")
    made = Index(repo)
    made.refresh()
    scores = bm25(made, ["zygote"])
    assert scores["rare_many.py"][0] > scores["rare_one.py"][0]
    made.close()


def test_defining_a_symbol_outranks_merely_mentioning_it(index):
    found = rank(index, "Session", limit=5)
    assert found, "Session is defined in the fixture"
    assert found[0].path == "pkg/auth.py", [r.path for r in found]


def test_an_unmatched_query_returns_nothing_rather_than_everything(index):
    assert rank(index, "quetzalcoatl", limit=5) == []


def test_select_context_respects_its_budget(repo):
    chosen = select_context("authentication", root=repo, budget=40)
    assert chosen, "something should match"
    assert sum(s.tokens for s in chosen) <= 40 or len(chosen) == 1, (
        "the budget may only be exceeded by a single indivisible snippet"
    )


# -- the tools --------------------------------------------------------------


def _ctx(root: Path) -> ToolContext:
    return ToolContext(cwd=root, root=root, cancel=threading.Event(), timeout=60.0)


def test_the_search_tool_returns_ranked_snippets(repo):
    tool = next(t for t in retrieve_tools() if t.name == "search")
    result = tool.run({"query": "getUserAuth"}, _ctx(repo))
    assert result.ok, result.error
    assert "pkg/auth.py" in result.content


def test_the_symbols_tool_finds_a_definition(repo):
    tool = next(t for t in retrieve_tools() if t.name == "symbols")
    result = tool.run({"action": "defines", "name": "get_user_authentication"}, _ctx(repo))
    assert result.ok, result.error
    assert "pkg/auth.py" in result.content


def test_the_symbols_tool_suggests_near_misses(repo):
    tool = next(t for t in retrieve_tools() if t.name == "symbols")
    result = tool.run({"action": "defines", "name": "get_user_auth"}, _ctx(repo))
    assert result.ok
    assert "did you mean" in result.content
    assert "get_user_authentication" in result.content


def test_the_symbols_tool_reports_importers(repo):
    tool = next(t for t in retrieve_tools() if t.name == "symbols")
    result = tool.run({"action": "importers", "file": "pkg/auth.py"}, _ctx(repo))
    assert result.ok, result.error
    assert "pkg/views.py" in result.content


def test_search_without_a_query_is_a_clean_failure(repo):
    tool = next(t for t in retrieve_tools() if t.name == "search")
    result = tool.run({"query": "   "}, _ctx(repo))
    assert not result.ok
    assert "needs a query" in (result.error or "")


def test_both_tool_schemas_survive_every_provider_dialect():
    from offset.providers.schema import normalise

    for tool in retrieve_tools():
        for dialect in ("anthropic", "openai", "google", "ollama"):
            normalise(tool.schema, dialect)
