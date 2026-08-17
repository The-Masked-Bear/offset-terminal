"""Searching the web from a tool call.

Two things are deliberately separate here.  `Backend` is the search itself —
swap it for a keyed API, a local index, or a fake in a test.  `WebSearch` is
the tool contract: argument checking, budgets, and turning a list of results
into something a model can read.

Nothing HTML ever reaches the model.  A results page is tens of kilobytes of
markup wrapped around three facts per hit, and every one of those bytes is
attacker-controlled text arriving in the model's context; title, URL and
snippet as plain lines is the whole useful payload.

The default backend is DuckDuckGo's HTML endpoint because it needs no key.
The price is that scraped markup, not a documented API, is the contract, so a
page this module cannot parse is reported as a parse failure rather than
being quietly rendered as "no results" — those two are different facts and
the model reacts to them differently.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Callable

from offset.tools.base import Danger, Tool, ToolContext, ToolResult

ENDPOINT = "https://html.duckduckgo.com/html/"

DEFAULT_RESULTS = 5
MAX_RESULTS = 15

#: The HTML endpoint answers 403 to an unrecognised client.  Claiming to be a
#: browser is the only way it works at all; pretending otherwise here would
#: just mean the tool never returns anything.
USER_AGENT = "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

#: Sponsored links wear the same class as organic ones and only differ by
#: going through the redirector.  An ad is not a search result.
AD_MARKERS = ("/y.js", "duckduckgo.com/y.js")


class SearchError(Exception):
    """A backend could not answer.  Carries text a model can act on."""


@dataclass(frozen=True, slots=True)
class Result:
    title: str
    url: str
    snippet: str = ""

    def line(self, index: int) -> str:
        body = f"{index}. {self.title}\n   {self.url}"
        return f"{body}\n   {self.snippet}" if self.snippet else body


class Backend(ABC):
    """One search engine.  `search` raises `SearchError` when it cannot try."""

    name: str = ""

    @abstractmethod
    def search(self, query: str, *, count: int, timeout: float) -> list[Result]: ...


# -- parsing ----------------------------------------------------------------


class _Page(HTMLParser):
    """Pulls (title, url, snippet) triples out of a DuckDuckGo HTML page.

    Written as a state machine over the two classes that carry the content,
    because the surrounding markup changes far more often than they do.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self.empty = False
        #: Anchors that looked like results, ads included.  Zero of them on a
        #: page that never said "no results" means the markup moved on.
        self.seen = 0
        self._in: str = ""
        self._buffer: list[str] = []
        self._href = ""
        self._ad = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = ""
        href = ""
        for key, value in attrs:
            if key == "class":
                classes = value or ""
            elif key == "href":
                href = value or ""
        if "no-results" in classes:
            self.empty = True
            return
        if tag != "a":
            return
        if "result__a" in classes:
            self.seen += 1
            # An ad's snippet follows its title, so the skip has to outlive
            # this tag; otherwise it lands on the previous organic result.
            self._ad = any(m in href for m in AD_MARKERS)
            self._in, self._buffer, self._href = ("" if self._ad else "title"), [], href
        elif "result__snippet" in classes:
            self._in, self._buffer = ("" if self._ad else "snippet"), []
            self._ad = False

    def handle_data(self, data: str) -> None:
        if self._in:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._in:
            return
        text = " ".join("".join(self._buffer).split())
        if self._in == "title":
            if text:
                self.results.append({"title": text, "url": _clean_url(self._href), "snippet": ""})
        elif self.results and not self.results[-1]["snippet"]:
            self.results[-1]["snippet"] = text
        self._in, self._buffer = "", []


def _clean_url(href: str) -> str:
    """Undo the `/l/?uddg=` redirector so the model sees the real destination."""
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    parts = urllib.parse.urlsplit(href)
    if parts.netloc.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parts.query).get("uddg", [""])[0]
        if target:
            return target
    return href


def parse(html: str) -> list[Result]:
    """Results in page order.  An unparseable page raises, an empty one does not."""
    page = _Page()
    try:
        page.feed(html)
        page.close()
    except Exception as exc:  # a malformed page is a backend failure, not zero hits
        raise SearchError(f"could not parse the search results page: {exc}") from exc
    if not page.results and not page.empty and "result__a" in html:
        raise SearchError("the search results page changed shape and could not be parsed")
    return [Result(r["title"], r["url"], r["snippet"]) for r in page.results if r["url"]]


# -- the default backend ----------------------------------------------------


def http_post(url: str, fields: dict[str, str], timeout: float) -> str:
    """Form-POST and return the body as text.  Replaceable in tests."""
    data = urllib.parse.urlencode(fields).encode()
    request = urllib.request.Request(
        url, data=data, headers={"User-Agent": USER_AGENT, "Accept": "text/html"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read(2_000_000).decode(charset, "replace")
    except urllib.error.HTTPError as exc:
        raise SearchError(f"the search endpoint returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise SearchError(f"could not reach the search endpoint: {exc}") from exc


class DuckDuckGo(Backend):
    """The keyless HTML endpoint.  `fetch` is injected so tests stay offline."""

    name = "duckduckgo"

    __slots__ = ("endpoint", "fetch")

    def __init__(
        self,
        *,
        endpoint: str = ENDPOINT,
        fetch: Callable[[str, dict[str, str], float], str] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.fetch = fetch or http_post

    def search(self, query: str, *, count: int, timeout: float) -> list[Result]:
        # kl=wt-wt is "no region": a result set that depends on where the
        # machine sits is not reproducible between two runs of the same agent.
        html = self.fetch(self.endpoint, {"q": query, "kl": "wt-wt"}, timeout)
        return parse(html)[:count]


# -- the tool ---------------------------------------------------------------


class WebSearch(Tool):
    name = "web_search"
    description = (
        "Search the web and get back a numbered list of title, URL and snippet. "
        "Use it for things outside the workspace - library docs, error messages, releases - "
        "then fetch the URLs worth reading in full."
    )
    #: No local damage, but the query leaves the machine.  Same reasoning as
    #: `fetch`: egress is a decision the user gets to make.
    danger = Danger.WRITE
    parallel_safe = True

    __slots__ = ("backend",)

    def __init__(self, backend: Backend | None = None) -> None:
        self.backend = backend or DuckDuckGo()
        self.schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 400},
                "count": {"type": "integer", "minimum": 1, "maximum": MAX_RESULTS},
            },
            "required": ["query"],
        }

    def preview(self, args: dict[str, Any]) -> str:
        return f"web_search {str(args.get('query') or '')[:60]!r}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult.fail("a search needs a non-empty query")
        count = max(1, min(MAX_RESULTS, int(args.get("count") or DEFAULT_RESULTS)))
        budget = min(ctx.timeout, 30.0) if ctx.timeout > 0 else 30.0

        try:
            results = self.backend.search(query, count=count, timeout=budget)
        except SearchError as exc:
            return ToolResult.fail(f"{self.backend.name}: {exc}")
        except Exception as exc:  # a broken backend is a message, not a dead turn
            return ToolResult.fail(f"{self.backend.name} search failed: {type(exc).__name__}: {exc}")
        ctx.check()

        if not results:
            # The search ran and the web had nothing. That is an answer, so
            # `ok` stays true; the content says so plainly enough that the
            # model retries with other words instead of reporting a failure.
            return ToolResult(
                content=f"no results for {query!r} on {self.backend.name}; try different or fewer words",
                display=f"web_search {query[:40]!r} -> no results",
                data={"query": query, "backend": self.backend.name, "count": 0, "results": []},
            )

        body = "\n".join(r.line(i) for i, r in enumerate(results, 1))
        return ToolResult(
            content=f"{len(results)} results for {query!r}:\n{body}",
            display=f"web_search {query[:40]!r} -> {len(results)} results",
            data={
                "query": query,
                "backend": self.backend.name,
                "count": len(results),
                "results": [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results],
            },
        )


def web_search_tools(backend: Backend | None = None) -> list[Tool]:
    return [WebSearch(backend)]
