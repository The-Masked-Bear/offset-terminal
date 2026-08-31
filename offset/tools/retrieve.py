"""Finding the right code, and the tools that expose it.

Ranking is BM25 blended with two structural signals, and the blend is the whole
design, so it is worth stating why each part is there.

**BM25** is the lexical core. Roughly forty lines, no dependency, and far better
than substring counting because it does the two things counting cannot: it
saturates term frequency, so a file that says `parse` forty times does not
automatically beat one that says it four times in the right place; and it
weights by inverse document frequency, so a rare word in the query carries more
than a common one.

**Identifier expansion** is what makes `getUserAuth` find
`get_user_authentication`. It is lexical, not semantic: the query is split on
camelCase and underscores and the pieces are matched too. Calling that
"semantic search" would be a lie, and the docstrings here do not.

**Structural proximity** is the part a text search cannot do. A file that
imports, or is imported by, a file already matching is more likely to be
relevant than its lexical score alone suggests, and the symbol graph knows that.

`select_context` exists for the harness: given a query and a token budget, hand
back the snippets worth spending it on.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from offset.core.index import Index, open_index, split_identifier, words_of
from offset.tools.base import Danger, Tool, ToolContext, ToolResult

#: BM25 term-frequency saturation.  1.2 is the usual choice for prose; code
#: repeats identifiers far more, so a slightly lower value saturates sooner and
#: stops a long file of boilerplate from dominating.
K1: Final = 1.1

#: Length normalisation.  0.75 is standard and behaves well here: a long file
#: is penalised, but a legitimately large module is not buried.
B: Final = 0.75

#: Weight of the structural signal relative to the lexical score.  Kept well
#: below 1 deliberately: proximity is a tie-breaker between plausible files,
#: never a reason to return a file the query does not match at all.
GRAPH_WEIGHT: Final = 0.25

#: Weight for the query appearing in the path itself.  A query of `auth` should
#: favour `auth.py`, and this is cheap and reliable.
PATH_WEIGHT: Final = 0.35

#: Weight for a query word naming a defined symbol.  Strong, because "the file
#: that defines this" is usually the answer to "where is this".
SYMBOL_WEIGHT: Final = 0.6

#: Lines of context around a match in a snippet.
CONTEXT: Final = 4

#: Characters per token, for budgeting.  Deliberately conservative.
CHARS_PER_TOKEN: Final = 4

DEFAULT_LIMIT: Final = 8


@dataclass(frozen=True, slots=True)
class Snippet:
    path: str
    line: int
    text: str
    score: float
    reason: str = ""

    @property
    def tokens(self) -> int:
        return max(1, len(self.text) // CHARS_PER_TOKEN)

    def render(self) -> str:
        head = f"{self.path}:{self.line}"
        if self.reason:
            head += f"  ({self.reason})"
        return f"{head}\n{self.text}"


@dataclass(slots=True)
class Ranking:
    """A scored file, and why."""

    path: str
    lexical: float = 0.0
    graph: float = 0.0
    path_score: float = 0.0
    symbol: float = 0.0
    matched: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.lexical + self.graph + self.path_score + self.symbol

    def reason(self) -> str:
        parts = []
        if self.symbol:
            parts.append("defines it")
        if self.path_score:
            parts.append("path match")
        if self.graph:
            parts.append("imports a match")
        if self.matched:
            parts.append("terms: " + ", ".join(self.matched[:4]))
        return "; ".join(parts)


def expand(query: str) -> list[str]:
    """The query words, plus the pieces of any compound identifier in it.

    Lexical expansion, not embeddings.  `getUserAuth` becomes
    `getuserauth, get, user, auth`, which is what lets it match
    `get_user_authentication`.
    """
    out: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query):
        lowered = raw.lower()
        if lowered not in out:
            out.append(lowered)
        for piece in split_identifier(raw):
            if piece not in out:
                out.append(piece)
    return out


def bm25(index: Index, terms: Sequence[str]) -> dict[str, tuple[float, list[str]]]:
    """Score every file containing any term.  Returns `path -> (score, matched)`.

    The document-frequency and postings lookups are batched, so this is two
    queries plus one per term rather than a scan.
    """
    lengths = index.lengths()
    total = len(lengths)
    if not total or not terms:
        return {}
    average = sum(lengths.values()) / total or 1.0
    frequency = index.document_frequency(terms)

    scored: dict[str, list[float]] = {}
    matched: dict[str, list[str]] = {}
    for term in terms:
        appearances = frequency.get(term, 0)
        if not appearances:
            continue
        # A word in almost every file carries no signal; skipping it here rather
        # than at write time means the threshold can change without a reindex.
        if appearances > total * 0.6 and len(terms) > 1:
            continue
        idf = math.log(1.0 + (total - appearances + 0.5) / (appearances + 0.5))
        for path, count in index.postings(term):
            length = lengths.get(path, 0) or 1
            saturated = (count * (K1 + 1)) / (count + K1 * (1 - B + B * length / average))
            scored.setdefault(path, []).append(idf * saturated)
            matched.setdefault(path, []).append(term)
    return {path: (sum(parts), matched.get(path, [])) for path, parts in scored.items()}


def rank(
    index: Index,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    use_graph: bool = True,
) -> list[Ranking]:
    """Blend the lexical score with path, symbol and structural signals."""
    terms = expand(query)
    lexical = bm25(index, terms)

    rankings: dict[str, Ranking] = {}
    for path, (score, matched) in lexical.items():
        rankings[path] = Ranking(path=path, lexical=score, matched=matched)

    # Path affinity: `auth` should favour `auth.py`.
    for term in terms:
        for path in index.files():
            if term in path.lower():
                entry = rankings.setdefault(path, Ranking(path=path))
                entry.path_score = max(entry.path_score, PATH_WEIGHT * _best_lexical(lexical))

    # Defining a queried symbol is the strongest single signal.
    for term in terms:
        for row in index.symbols_named(term):
            entry = rankings.setdefault(row["path"], Ranking(path=row["path"]))
            entry.symbol = max(entry.symbol, SYMBOL_WEIGHT * _best_lexical(lexical))
        for row in index.symbols_like(term, limit=20):
            entry = rankings.setdefault(row["path"], Ranking(path=row["path"]))
            entry.symbol = max(entry.symbol, 0.4 * SYMBOL_WEIGHT * _best_lexical(lexical))

    if use_graph and rankings:
        try:
            graph = index.graph()
        except Exception:
            graph = None
        if graph is not None:
            seeds = sorted(rankings, key=lambda p: rankings[p].total, reverse=True)[:5]
            best = _best_lexical(lexical)
            for seed in seeds:
                for neighbour in graph.neighbourhood(seed, 1):
                    if neighbour == seed:
                        continue
                    entry = rankings.setdefault(neighbour, Ranking(path=neighbour))
                    entry.graph = max(entry.graph, GRAPH_WEIGHT * best)

    ordered = sorted(rankings.values(), key=lambda r: (-r.total, r.path))
    return [r for r in ordered if r.total > 0][:limit]


def _best_lexical(lexical: dict[str, tuple[float, list[str]]]) -> float:
    """The top lexical score, so structural weights are on the same scale.

    Without this the structural bonuses would be absolute numbers competing
    with a BM25 score whose magnitude depends on corpus size.
    """
    if not lexical:
        return 1.0
    return max(score for score, _ in lexical.values()) or 1.0


def snippets_for(index: Index, ranking: Ranking, query: str, *, budget: int = 900) -> Snippet | None:
    """The most relevant region of a file, with a little context."""
    text = index.text_of(ranking.path)
    if not text:
        return None
    terms = set(expand(query))
    lines = text.splitlines()
    best_line, best_hits = 0, 0
    for i, line in enumerate(lines):
        hits = sum(1 for word in words_of(line) if word in terms)
        if hits > best_hits:
            best_hits, best_line = hits, i
    if best_hits == 0:
        window = lines[:CONTEXT * 2]
        start = 0
    else:
        start = max(0, best_line - CONTEXT)
        window = lines[start:best_line + CONTEXT + 1]
    body = "\n".join(window)[:budget]
    return Snippet(
        path=ranking.path, line=start + 1, text=body,
        score=ranking.total, reason=ranking.reason(),
    )


def select_context(
    query: str,
    *,
    root: Path | str = ".",
    budget: int = 4000,
    limit: int = DEFAULT_LIMIT,
) -> list[Snippet]:
    """The snippets worth spending a token budget on, best first."""
    index = open_index(root)
    index.refresh()
    chosen: list[Snippet] = []
    spent = 0
    for ranking in rank(index, query, limit=limit):
        snippet = snippets_for(index, ranking, query)
        if snippet is None:
            continue
        if spent + snippet.tokens > budget and chosen:
            break
        chosen.append(snippet)
        spent += snippet.tokens
    return chosen


class Search(Tool):
    """Ranked code search over the indexed workspace."""

    name = "search"
    description = (
        "Search the workspace for code relevant to a query and return ranked snippets. "
        "Ranks by BM25 over identifiers, plus symbol definitions, path matches and import "
        "proximity. Splits compound identifiers, so getUserAuth matches "
        "get_user_authentication. Prefer this over grep when you do not know the exact text."
    )
    danger = Danger.SAFE
    parallel_safe = True
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "maxLength": 400, "description": "what to look for"},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 30,
                "description": "how many files to return; default 8",
            },
        },
        "required": ["query"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        return f"search {str(args.get('query', ''))[:60]}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult.fail("search needs a query")
        limit = args.get("limit")
        limit = int(limit) if isinstance(limit, int) and limit > 0 else DEFAULT_LIMIT

        root = Path(getattr(ctx, "root", None) or ctx.cwd)
        try:
            index = open_index(root)
            stats = index.refresh(cancel=lambda: ctx.cancel.is_set())
        except Exception as exc:
            return ToolResult.fail(f"could not index {root}: {type(exc).__name__}: {exc}")
        ctx.check()

        found = rank(index, query, limit=limit)
        if not found:
            return ToolResult.text(
                f"nothing matched {query!r} in {index.count()} indexed file(s)"
            )
        blocks: list[str] = []
        for ranking in found:
            snippet = snippets_for(index, ranking, query)
            if snippet is None:
                blocks.append(f"{ranking.path}  ({ranking.reason()})")
                continue
            blocks.append(snippet.render())
        head = f"{len(found)} match(es) in {index.count()} file(s)"
        if stats.parsed:
            head += f"; indexed {stats.parsed} changed file(s)"
        return ToolResult.text(head + "\n\n" + "\n\n".join(blocks))


class Symbols(Tool):
    """Where a symbol is defined and what refers to it."""

    name = "symbols"
    description = (
        "Find where a symbol is defined and what references it, using a parsed symbol "
        "graph. For Python this is AST-exact. Also answers which files import a file and "
        "which import it, so you can see what a change would affect."
    )
    danger = Danger.SAFE
    parallel_safe = True
    schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["defines", "references", "imports", "importers", "outline"],
                "description": "what to ask",
            },
            "name": {"type": "string", "maxLength": 200, "description": "the symbol to look up"},
            "file": {
                "type": "string",
                "description": "for imports/importers/outline: the file to ask about",
            },
        },
        "required": ["action"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        return f"symbols {args.get('action', '?')} {args.get('name') or args.get('file') or ''}".strip()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(args.get("action", "")).strip()
        root = Path(getattr(ctx, "root", None) or ctx.cwd)
        try:
            index = open_index(root)
            index.refresh(cancel=lambda: ctx.cancel.is_set())
        except Exception as exc:
            return ToolResult.fail(f"could not index {root}: {type(exc).__name__}: {exc}")
        ctx.check()

        if action in ("defines", "references"):
            name = str(args.get("name", "")).strip()
            if not name:
                return ToolResult.fail(f"action {action!r} needs a name")
            if action == "defines":
                rows = index.symbols_named(name)
                if not rows:
                    near = index.symbols_like(name, limit=8)
                    if near:
                        lines = [f"no symbol named {name!r}. did you mean:"]
                        lines.extend(f"  {r['name']}  {r['path']}:{r['line']}" for r in near)
                        return ToolResult.text("\n".join(lines))
                    return ToolResult.text(f"no symbol named {name!r}")
                lines = [
                    f"{r['kind']} {r['name']}{r['signature']}  {r['path']}:{r['line']}"
                    for r in rows
                ]
                return ToolResult.text("\n".join(lines))

            graph = index.graph()
            found = graph.references(name)
            if not found:
                return ToolResult.text(f"nothing references {name!r}")
            lines = [f"{loc.path}:{loc.line}" for loc in found[:60]]
            if len(found) > 60:
                lines.append(f"... {len(found) - 60} more")
            return ToolResult.text("\n".join(lines))

        target = str(args.get("file", "")).strip()
        if not target:
            return ToolResult.fail(f"action {action!r} needs a file")
        try:
            resolved = ctx.resolve(target)
        except PermissionError as exc:
            return ToolResult.fail(str(exc))
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            relative = target

        graph = index.graph()
        if action == "outline":
            record = index.file(relative)
            if record is None:
                return ToolResult.fail(f"{relative} is not indexed")
            rows = [
                r for r in index.symbols_like("", limit=500) if r["path"] == relative
            ] or index.symbols_named(Path(relative).stem)
            lines = [f"{relative}  ({record.language}, {record.lines} lines)"]
            lines.extend(
                f"  {r['kind']:9s} {r['name']}{r['signature']}  :{r['line']}"
                for r in sorted(rows, key=lambda r: r["line"])
            )
            return ToolResult.text("\n".join(lines))

        # Both directions return workspace paths: `imports_of` would give the
        # dotted names as written, which does not compare with `importers_of`.
        getter = graph.dependencies_of if action == "imports" else graph.importers_of
        found = getter(relative)
        if not found:
            verb = "imports nothing in this tree" if action == "imports" else "is imported by nothing"
            return ToolResult.text(f"{relative} {verb}")
        return ToolResult.text("\n".join(found))


def retrieve_tools(workspace: Path | str | None = None) -> list[Tool]:
    """The pair.  The index is opened per workspace on first use, not here."""
    return [Search(), Symbols()]
