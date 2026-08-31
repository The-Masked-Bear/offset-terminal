"""The model-facing browser.

Backend code can be checked by running it; a web page cannot.  Without a
browser the agent's only evidence that a UI works is that the markup it wrote
looks plausible, which is not evidence.  With one it can load the page, read
what is actually rendered, click a button and observe what happened.

Two decisions shape this surface.

The default view is the accessibility tree, not a screenshot.  A screenshot
costs a large image and still has to be interpreted; the AX tree is a few
hundred bytes of text that names every interactive element and its state, which
is what a model needs in order to decide what to click.  `snapshot` hands back
stable `[ref=eN]` handles so the next call can act on what it just read - the
alternative is guessing CSS selectors from a picture.

Screenshots return a PATH, never inline data.  A base64 PNG is hundreds of
kilobytes of context that the model cannot read anyway; a path can be looked at
by a human or passed to something that can.

Sessions are named and persist between calls, because a browser that closed
after every action could not test a login flow.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any, Final

from offset.tools.base import Danger, Tool, ToolContext, ToolResult
from offset.tools.web.cdp import (
    BrowserGone,
    CDPClient,
    CDPError,
    CDPTimeout,
    LaunchError,
    Page,
    attach,
    find_executable,
    launch,
)

ACTIONS: Final = (
    "open",
    "goto",
    "click",
    "type",
    "fill",
    "press",
    "scroll",
    "evaluate",
    "screenshot",
    "snapshot",
    "console",
    "close",
    "status",
)

#: Nodes of the accessibility tree to render.  Enough for a real page's
#: interactive surface; a document with more than this is usually a list, where
#: the first hundred rows tell you the shape.
SNAPSHOT_LIMIT: Final = 120

DEFAULT_SESSION: Final = "main"


class _Session:
    """One browser, one page, kept alive between tool calls."""

    __slots__ = ("client", "launched", "name", "page")

    def __init__(self, name: str, launched: Any, client: CDPClient, page: Page) -> None:
        self.name = name
        self.launched = launched
        self.client = client
        self.page = page

    @property
    def alive(self) -> bool:
        return self.client.alive and self.launched.alive

    def close(self) -> list[str]:
        notes: list[str] = []
        try:
            self.page.close()
        except (CDPError, OSError) as exc:
            notes.append(f"page: {exc}")
        try:
            self.client.close()
        except (CDPError, OSError) as exc:
            notes.append(f"client: {exc}")
        try:
            self.launched.close()
        except OSError as exc:
            notes.append(f"browser: {exc}")
        return notes


class _Pool:
    """Every live session.  One lock: two calls must not race a launch."""

    __slots__ = ("_lock", "_sessions")

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def get(self, name: str) -> _Session | None:
        with self._lock:
            session = self._sessions.get(name)
            if session is not None and not session.alive:
                self._sessions.pop(name, None)
                return None
            return session

    def put(self, session: _Session) -> None:
        with self._lock:
            self._sessions[session.name] = session

    def drop(self, name: str) -> list[str]:
        with self._lock:
            session = self._sessions.pop(name, None)
        return session.close() if session is not None else []

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)

    def close_all(self) -> list[str]:
        with self._lock:
            names = list(self._sessions)
        notes: list[str] = []
        for name in names:
            notes.extend(self.drop(name))
        return notes


_POOL: Final = _Pool()


def pool() -> _Pool:
    return _POOL


def close_all() -> list[str]:
    """Shut every browser down.  Called at shell exit."""
    return _POOL.close_all()


def _fail(exc: Exception) -> ToolResult:
    if isinstance(exc, LaunchError):
        return ToolResult.fail(str(exc))
    if isinstance(exc, CDPTimeout):
        return ToolResult.fail(f"the browser did not answer in time: {exc}")
    if isinstance(exc, BrowserGone):
        return ToolResult.fail(f"the browser exited: {exc}")
    if isinstance(exc, CDPError):
        return ToolResult.fail(str(exc))
    return ToolResult.fail(f"{type(exc).__name__}: {exc}")


class Browser(Tool):
    """Drive a real browser."""

    name = "browser"
    description = (
        "Drive a real headless browser to test a web page: load a URL, read the "
        "accessibility tree, click, type, run JavaScript, read console errors, take a "
        "screenshot. Prefer action=snapshot over screenshot - it returns element refs "
        "you can click. Sessions persist between calls."
    )
    danger = Danger.DESTRUCTIVE
    parallel_safe = False
    schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(ACTIONS), "description": "what to do"},
            "url": {"type": "string", "description": "for open/goto"},
            "selector": {
                "type": "string",
                "maxLength": 400,
                "description": "CSS selector, or a ref like e5 from a snapshot",
            },
            "text": {"type": "string", "maxLength": 4000, "description": "for type/fill"},
            "key": {
                "type": "string",
                "maxLength": 60,
                "description": "for press, e.g. Enter or Control+a",
            },
            "expression": {
                "type": "string",
                "maxLength": 4000,
                "description": "for evaluate: JavaScript to run in the page",
            },
            "session": {
                "type": "string",
                "maxLength": 60,
                "description": "which browser session; defaults to main",
            },
            "dx": {"type": "number", "description": "for scroll: horizontal pixels"},
            "dy": {"type": "number", "description": "for scroll: vertical pixels"},
            "full_page": {"type": "boolean", "description": "for screenshot: capture beyond the viewport"},
            "timeout": {"type": "number", "minimum": 0, "description": "seconds to wait"},
        },
        "required": ["action"],
    }

    def preview(self, args: dict[str, Any]) -> str:
        action = args.get("action", "?")
        target = args.get("url") or args.get("selector") or ""
        return f"browser {action} {target}".strip()[:80]

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = str(args.get("action", "")).strip()
        if action not in ACTIONS:
            return ToolResult.fail(
                f"no browser action {action!r}. available: {', '.join(sorted(ACTIONS))}"
            )
        name = str(args.get("session") or DEFAULT_SESSION)

        if action == "status":
            live = _POOL.names()
            if not live:
                return ToolResult.text("no browser sessions")
            return ToolResult.text("\n".join(f"{n}: live" for n in live))

        if action == "close":
            notes = _POOL.drop(name)
            return ToolResult.text("\n".join([f"closed {name}", *notes]))

        try:
            if action == "open":
                return self._open(name, args, ctx)

            session = _POOL.get(name)
            if session is None:
                if action == "goto":
                    return self._open(name, args, ctx)
                return ToolResult.fail(
                    f"no browser session {name!r}; open one first with action=open"
                )
            ctx.check()
            return self._act(action, session, args, ctx)
        except Exception as exc:
            return _fail(exc)

    # -- lifecycle ----------------------------------------------------------

    def _open(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Launch a browser, or adopt one already listening."""
        existing = _POOL.get(name)
        if existing is not None:
            url = str(args.get("url") or "")
            if url:
                existing.page.navigate(url)
                return ToolResult.text(f"{name}: {existing.page.url()}")
            return ToolResult.text(f"{name} is already open at {existing.page.url()}")

        port = args.get("port")
        if isinstance(port, int) and port > 0:
            launched = attach(port)
        else:
            if find_executable() is None:
                return ToolResult.fail(
                    "no browser found. install one with: apt install chromium, "
                    "or brew install --cask google-chrome"
                )
            launched = launch()

        client = CDPClient.open(launched)
        pages = client.pages()
        target = pages[0].target_id if pages else client.create_page()
        page = Page.attach(client, target)
        page.enable()
        session = _Session(name, launched, client, page)
        _POOL.put(session)

        url = str(args.get("url") or "")
        if url:
            page.navigate(url)
        lines = [f"opened {name}", f"url: {page.url()}", f"title: {page.title()}"]
        return ToolResult.text("\n".join(lines))

    # -- actions ------------------------------------------------------------

    def _act(self, action: str, session: _Session, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        page = session.page

        if action == "goto":
            url = str(args.get("url") or "")
            if not url:
                return ToolResult.fail("goto needs a url")
            page.navigate(url)
            return ToolResult.text(f"{page.url()}\ntitle: {page.title()}")

        if action == "snapshot":
            nodes = page.snapshot(limit=SNAPSHOT_LIMIT)
            if not nodes:
                return ToolResult.text("the page exposes no accessibility tree")
            lines = [node.line() for node in nodes]
            lines.append("")
            lines.append("act on a node with selector=e<N>")
            return ToolResult.text("\n".join(lines))

        if action == "click":
            selector = str(args.get("selector") or "")
            if not selector:
                return ToolResult.fail("click needs a selector or a ref")
            box = page.click(selector)
            return ToolResult.text(
                f"clicked {selector} at ({box.centre[0]:.0f}, {box.centre[1]:.0f})\n"
                f"url: {page.url()}"
            )

        if action == "type":
            text = str(args.get("text") or "")
            if not text:
                return ToolResult.fail("type needs text")
            selector = str(args.get("selector") or "")
            if selector:
                page.click(selector)
            page.type_text(text)
            return ToolResult.text(f"typed {len(text)} character(s)")

        if action == "fill":
            selector = str(args.get("selector") or "")
            if not selector:
                return ToolResult.fail("fill needs a selector")
            page.fill(selector, str(args.get("text") or ""))
            return ToolResult.text(f"filled {selector}")

        if action == "press":
            key = str(args.get("key") or "")
            if not key:
                return ToolResult.fail("press needs a key")
            page.press(key)
            return ToolResult.text(f"pressed {key}")

        if action == "scroll":
            page.scroll(dx=float(args.get("dx", 0) or 0), dy=float(args.get("dy", 0) or 0))
            return ToolResult.text("scrolled")

        if action == "evaluate":
            expression = str(args.get("expression") or "")
            if not expression:
                return ToolResult.fail("evaluate needs an expression")
            value = page.evaluate_value(expression)
            return ToolResult.text(repr(value) if value is not None else "undefined")

        if action == "console":
            messages = page.messages()
            if not messages:
                return ToolResult.text("no console output")
            return ToolResult.text("\n".join(messages))

        # screenshot
        data = page.screenshot(full_page=bool(args.get("full_page", False)))
        handle, path = tempfile.mkstemp(prefix="offset-shot-", suffix=".png")
        try:
            with open(handle, "wb") as fh:
                fh.write(data)
        except OSError as exc:
            return ToolResult.fail(f"could not save the screenshot: {exc}")
        return ToolResult.text(
            f"screenshot saved: {path} ({len(data) // 1024} KiB)",
            display=f"screenshot -> {Path(path).name}",
        )


def browser_tools() -> list[Tool]:
    return [Browser()]
