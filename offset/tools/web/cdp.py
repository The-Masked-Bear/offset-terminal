"""The Chrome DevTools Protocol, spoken directly down a raw WebSocket.

Driving a browser is the only way to see what a page actually renders, and
every convenient library for it (playwright, puppeteer's python ports,
selenium) is a dependency offset refuses to take.  What is left is the wire
protocol itself, and the wire protocol turns out to be small: JSON messages
with an integer `id`, replies correlated by that id, and unsolicited events
that carry a `method` instead.  That is the same shape as the MCP client in
`offset/tools/mcp/client.py`, so this module deliberately mirrors it — one
reader thread owning the socket, a pending-reply table under a lock, a hard
deadline on every call, and cooperative cancellation polled while waiting.
If you have read that file you have read this one.

Two decisions are worth defending.

**The port is never assumed.**  Chromium is launched with
`--remote-debugging-port=0` so the kernel picks a free port; anything else
collides with a browser the user already has open, or with a sibling process
on the same machine.  The chosen port is read back from the
`DevToolsActivePort` file Chromium writes into its user-data directory.  A
hardcoded 9222 is a bug that only shows up when two things run at once.

**Input is real input.**  Clicking is `DOM.getBoxModel` to find the element's
box followed by `Input.dispatchMouseEvent` at its centre, not
`element.click()` in JavaScript.  A synthetic JS click skips hit-testing, so
it happily "clicks" an element covered by a consent banner and reports
success; a dispatched mouse event hits whatever is actually on top, which is
the truth the caller asked for.

Everything here degrades into an exception carrying a lowercase, specific
message.  Nothing prints, nothing retries silently, and `close()` reaps the
browser's whole process group so a failed run cannot leave a headless
Chromium eating a Raspberry Pi.
"""

from __future__ import annotations

import base64
import itertools
import json
import os
import re
import shutil
import signal
import sys
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Sequence

from offset.tools.web.wsclient import WebSocket, WebSocketClosed, WebSocketError

#: Executables probed on PATH, best first.  Debian ships `chromium`, Ubuntu
#: shipped `chromium-browser` for years, and Google's own package is
#: `google-chrome`; a machine with none of them cannot run this tool at all.
BROWSER_NAMES: Final = ("chromium", "chromium-browser", "google-chrome", "chrome", "chrome-browser")

#: Written by Chromium into the user-data directory once the debugging socket
#: is listening: line one is the port, line two the browser target path.
PORT_FILE: Final = "DevToolsActivePort"

#: Wait slice while a reply is outstanding.  Short enough that cancellation
#: feels immediate, long enough that waiting costs no measurable CPU.
_TICK: Final = 0.05

#: Recent events kept for `wait_event`.  Bounded because a chatty page emits
#: lifecycle events forever and none of them are worth remembering for long.
_EVENT_MEMORY: Final = 512

#: Console lines kept per page.  The model reads the tail, not the history.
CONSOLE_MEMORY: Final = 200

#: Default ceiling on nodes rendered from an accessibility tree.  A real page
#: has thousands; a model needs the first few hundred.
AX_LIMIT: Final = 400

#: `Input.dispatchKeyEvent` modifier bits, from the protocol definition.
MOD_ALT: Final = 1
MOD_CTRL: Final = 2
MOD_META: Final = 4
MOD_SHIFT: Final = 8

_MODIFIERS: Final = {
    "alt": MOD_ALT,
    "option": MOD_ALT,
    "ctrl": MOD_CTRL,
    "control": MOD_CTRL,
    "meta": MOD_META,
    "cmd": MOD_META,
    "command": MOD_META,
    "super": MOD_META,
    "shift": MOD_SHIFT,
}

#: name -> (key, code, windowsVirtualKeyCode, text).  `text` is empty for keys
#: that must not insert a character; Chromium decides insertion from `text`
#: alone, so a stray value here types garbage into the focused field.
KEYS: Final[dict[str, tuple[str, str, int, str]]] = {
    "enter": ("Enter", "Enter", 13, "\r"),
    "return": ("Enter", "Enter", 13, "\r"),
    "tab": ("Tab", "Tab", 9, "\t"),
    "escape": ("Escape", "Escape", 27, ""),
    "esc": ("Escape", "Escape", 27, ""),
    "space": (" ", "Space", 32, " "),
    "backspace": ("Backspace", "Backspace", 8, ""),
    "delete": ("Delete", "Delete", 46, ""),
    "del": ("Delete", "Delete", 46, ""),
    "insert": ("Insert", "Insert", 45, ""),
    "home": ("Home", "Home", 36, ""),
    "end": ("End", "End", 35, ""),
    "pageup": ("PageUp", "PageUp", 33, ""),
    "pagedown": ("PageDown", "PageDown", 34, ""),
    "up": ("ArrowUp", "ArrowUp", 38, ""),
    "down": ("ArrowDown", "ArrowDown", 40, ""),
    "left": ("ArrowLeft", "ArrowLeft", 37, ""),
    "right": ("ArrowRight", "ArrowRight", 39, ""),
    "arrowup": ("ArrowUp", "ArrowUp", 38, ""),
    "arrowdown": ("ArrowDown", "ArrowDown", 40, ""),
    "arrowleft": ("ArrowLeft", "ArrowLeft", 37, ""),
    "arrowright": ("ArrowRight", "ArrowRight", 39, ""),
}

#: `ref=e5`, `[ref=e5]` or a bare `e5` — the three shapes a model writes back
#: after reading a snapshot.  Anything else is treated as a CSS selector.
_REF = re.compile(r"^\[?\s*(?:ref\s*=\s*)?(e\d+)\s*\]?$", re.IGNORECASE)

#: Flags that turn a desktop browser into a quiet, disposable automation
#: target.  Every one of them is here to stop Chromium doing something on its
#: own initiative: phoning home, updating, restoring tabs, or popping a
#: keyring prompt on a machine with no keyring.
_QUIET_FLAGS: Final = (
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-client-side-phishing-detection",
    "--disable-sync",
    "--disable-domain-reliability",
    "--disable-breakpad",
    "--no-service-autorun",
    "--metrics-recording-only",
    "--mute-audio",
    "--password-store=basic",
    "--use-mock-keychain",
    (
        "--disable-features=Translate,MediaRouter,OptimizationHints,"
        "CalculateNativeWinOcclusion,AcceptCHFrame,InterestFeedContentSuggestions"
    ),
)


# -- errors ------------------------------------------------------------------


class CDPError(Exception):
    """A protocol-level failure, including an error reply from the browser."""


class LaunchError(CDPError):
    """No browser could be started or found, so there is nothing to drive."""


class CDPTimeout(CDPError):
    """The browser did not answer in time.  Never a user cancellation."""


class BrowserGone(CDPError):
    """The connection died, so no reply can ever arrive."""


class CDPCancelled(CDPError):
    """The caller asked to stop while a call was outstanding."""


# -- discovery and launch ----------------------------------------------------


def find_executable(candidates: Sequence[str] = BROWSER_NAMES) -> str | None:
    """The first browser on PATH, or `None` when the machine has none."""
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def endpoint(port: int, *, host: str = "127.0.0.1", timeout: float = 5.0) -> str:
    """The browser's `webSocketDebuggerUrl` from `/json/version`.

    This is how an already-running browser is joined: the user started it with
    `--remote-debugging-port=<port>` and we ask it where to connect.
    """
    url = f"http://{host}:{port}/json/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError) as exc:
        raise LaunchError(f"no devtools endpoint on {host}:{port}: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise LaunchError(f"{url} did not answer with json: {exc}") from exc
    ws = payload.get("webSocketDebuggerUrl") if isinstance(payload, dict) else None
    if not isinstance(ws, str) or not ws:
        raise LaunchError(f"{url} answered without a websocketdebuggerurl")
    return ws


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, falling back to the child.

    Chromium is a process *tree* — zygote, renderers, gpu — so signalling only
    the launcher pid leaves renderers behind holding memory and a socket.
    """
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def _read_port_file(path: Path) -> tuple[int, str] | None:
    """`(port, browser path)` once Chromium has finished writing the file.

    Returns `None` while the file is absent or half-written: it is created and
    then filled, so a read can legitimately land in the middle.
    """
    try:
        lines = path.read_text("utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if len(lines) < 2 or not lines[0].strip().isdigit():
        return None
    port = int(lines[0].strip())
    target = lines[1].strip()
    if port <= 0 or not target.startswith("/"):
        return None
    return port, target


@dataclass(slots=True)
class Launch:
    """A browser this process is talking to, and how to let go of it.

    `proc` is `None` when we merely joined a browser the user was already
    running: `close()` must then leave it alone.  Killing a browser we did not
    start would take the user's tabs with it.
    """

    ws_url: str
    proc: subprocess.Popen | None = None
    user_data: Path | None = None
    #: True only when both the process and the profile are ours to destroy.
    owned: bool = False
    log: Path | None = None
    grace: float = 3.0
    _closed: bool = False

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    @property
    def alive(self) -> bool:
        if self._closed:
            return False
        return self.proc is None or self.proc.poll() is None

    def diagnostics(self) -> str:
        """One line naming what happened to the browser."""
        if self.proc is None:
            return f"attached to {self.ws_url}"
        code = self.proc.poll()
        if code is None:
            return f"browser pid {self.proc.pid} running on {self.ws_url}"
        return f"browser pid {self.proc.pid} exited {code}" + (f": {self.tail()}" if self.tail() else "")

    def tail(self, lines: int = 3) -> str:
        """The last few lines the browser wrote, for a failure message."""
        if self.log is None:
            return ""
        try:
            text = self.log.read_text("utf-8", errors="replace")
        except OSError:
            return ""
        kept = [line for line in text.splitlines() if line.strip()][-lines:]
        return " | ".join(kept)[:400]

    def close(self) -> None:
        """Reap the browser's process group and its throwaway profile."""
        self._closed = True
        proc = self.proc
        if proc is not None and self.owned:
            if proc.poll() is None:
                _signal_group(proc, signal.SIGTERM)
                try:
                    proc.wait(timeout=self.grace)
                except subprocess.TimeoutExpired:
                    _signal_group(proc, signal.SIGKILL)
                    try:
                        proc.wait(timeout=self.grace)
                    except subprocess.TimeoutExpired:
                        pass
            else:
                proc.wait()  # reap, or the pid keeps answering kill(pid, 0)
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if self.owned and self.user_data is not None:
            shutil.rmtree(self.user_data, ignore_errors=True)


def _die_with_parent() -> None:
    """Ask the kernel to kill this child when offset dies.

    `start_new_session=True` is what lets `close()` signal the browser's whole
    process group without signalling offset as well - but detaching from the
    session also means a shell that is killed outright, rather than exiting
    through its `finally`, leaves a headless browser running that nobody can
    see. Measured on this machine: eleven chromium processes surviving a
    SIGKILLed parent, which on a Raspberry Pi is most of the free memory.

    `PR_SET_PDEATHSIG` closes that gap: the kernel delivers SIGKILL to the
    child the moment its parent goes, however the parent went. Linux only, and
    a no-op everywhere else - the normal teardown path is unchanged, so a
    platform without it is no worse off than before.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except (OSError, AttributeError, ValueError):
        return  # best effort; close() is still the primary path


def launch(
    *,
    executable: str | None = None,
    headless: bool = True,
    user_data: Path | str | None = None,
    extra_args: Iterable[str] = (),
    start_url: str = "about:blank",
    timeout: float = 30.0,
) -> Launch:
    """Start a browser with an ephemeral debugging port and connect-ready url.

    The port is `0` — the kernel picks one — and is read back from
    `DevToolsActivePort`.  Chromium's stdio goes to a log file inside the
    profile: it is voluminous, uninteresting, and only wanted when the launch
    fails, at which point `Launch.tail()` produces the last few lines.
    """
    exe = executable or find_executable()
    if not exe:
        raise LaunchError(
            "no browser found on PATH. install one of: " + ", ".join(BROWSER_NAMES)
        )
    owned_profile = user_data is None
    profile = Path(tempfile.mkdtemp(prefix="offset-browser-")) if owned_profile else Path(user_data)
    try:
        profile.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise LaunchError(f"cannot create the browser profile {profile}: {exc}") from exc
    port_file = profile / PORT_FILE
    # A stale file from a previous run would be read as this run's port.
    try:
        port_file.unlink()
    except OSError:
        pass

    argv = [
        exe,
        "--remote-debugging-port=0",
        f"--user-data-dir={profile}",
        *_QUIET_FLAGS,
        *(["--headless=new"] if headless else []),
        *extra_args,
        start_url,
    ]
    log_path = profile / "browser.log"
    try:
        log = log_path.open("wb")
    except OSError as exc:
        raise LaunchError(f"cannot write the browser log {log_path}: {exc}") from exc
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # so close() can reap the whole tree
            preexec_fn=_die_with_parent,  # ...and so a killed shell reaps it too
        )
    except (OSError, ValueError) as exc:
        log.close()
        if owned_profile:
            shutil.rmtree(profile, ignore_errors=True)
        raise LaunchError(f"could not start {exe}: {exc}") from exc
    finally:
        log.close()  # the child holds its own duplicated descriptor

    started = Launch(ws_url="", proc=proc, user_data=profile, owned=True, log=log_path)
    deadline = time.monotonic() + max(1.0, timeout)
    while True:
        found = _read_port_file(port_file)
        if found is not None:
            port, target = found
            started.ws_url = f"ws://127.0.0.1:{port}{target}"
            if not owned_profile:
                # The profile is the caller's; only the process is ours.
                started.user_data = None
            return started
        if proc.poll() is not None:
            started.close()
            raise LaunchError(f"{exe} exited {proc.returncode} before listening: {started.tail()}")
        if time.monotonic() >= deadline:
            started.close()
            raise LaunchError(f"{exe} did not write {PORT_FILE} within {timeout:g}s: {started.tail()}")
        time.sleep(_TICK)


def attach(port: int, *, host: str = "127.0.0.1", timeout: float = 5.0) -> Launch:
    """Join a browser someone else started.  `close()` will not kill it."""
    return Launch(ws_url=endpoint(port, host=host, timeout=timeout), proc=None, owned=False)


# -- protocol ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    """One unsolicited message.  `session` is empty for browser-level events."""

    method: str
    params: dict[str, Any]
    session: str
    #: Monotonic arrival number, so a waiter can say "since I last looked"
    #: without racing the reader thread.
    seq: int


@dataclass(slots=True)
class _Pending:
    event: threading.Event
    reply: dict[str, Any] | None = None
    failure: Exception | None = None


class CDPClient:
    """One connected browser.  Thread-safe; sessions may be driven in parallel."""

    __slots__ = (
        "_closed",
        "_cond",
        "_events",
        "_ids",
        "_lock",
        "_pending",
        "_reader",
        "_seq",
        "dead_reason",
        "launch",
        "listeners",
        "skipped",
        "timeout",
        "ws",
    )

    def __init__(self, ws: WebSocket, *, launch: Launch | None = None, timeout: float = 20.0) -> None:
        self.ws = ws
        self.launch = launch
        self.timeout = timeout
        self.dead_reason = ""
        self.skipped = 0
        self.listeners: list[Callable[[Event], None]] = []
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._events: deque[Event] = deque(maxlen=_EVENT_MEMORY)
        self._seq = 0
        self._pending: dict[int, _Pending] = {}
        self._closed = False
        self._reader = threading.Thread(target=self._pump, name="cdp-reader", daemon=True)
        self._reader.start()

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def open(cls, target: Launch | str, *, timeout: float = 20.0) -> CDPClient:
        """Connect to a `Launch` (or a bare `ws://` url) and start reading."""
        started = target if isinstance(target, Launch) else Launch(ws_url=target, owned=False)
        try:
            ws = WebSocket.connect(started.ws_url, timeout=timeout)
        except WebSocketError as exc:
            started.close()
            raise BrowserGone(f"could not open the devtools socket: {exc}") from exc
        return cls(ws, launch=started, timeout=timeout)

    @property
    def alive(self) -> bool:
        return not self._closed and self.ws.alive

    def diagnostics(self) -> str:
        if self.dead_reason:
            return self.dead_reason
        if self.launch is not None:
            return self.launch.diagnostics()
        return self.ws.diagnostics()

    def close(self) -> None:
        """Idempotent.  Everything waiting is failed, then the browser dies."""
        was_open = not self._closed
        self._closed = True
        self._fail_all(BrowserGone(self.dead_reason or "client closed"))
        with self._cond:
            self._cond.notify_all()
        if was_open:
            try:
                self.ws.close()
            except Exception:  # closing must never raise into a shutdown path
                pass
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)
        if self.launch is not None:
            self.launch.close()

    # -- calls --------------------------------------------------------------

    def _next_id(self) -> int:
        with self._lock:
            return next(self._ids)

    def _forget(self, ident: int) -> None:
        with self._lock:
            self._pending.pop(ident, None)

    def _fail_all(self, exc: Exception) -> None:
        with self._lock:
            waiting = list(self._pending.values())
            self._pending.clear()
        for slot in waiting:
            slot.failure = exc
            slot.event.set()

    def send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        session: str = "",
        timeout: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Call `method` and wait for its reply, with a hard deadline.

        `stop` is polled while waiting so a cancelled turn does not sit out the
        whole budget.  The browser is left to finish whatever it started: CDP
        has no cancel notification, and pretending otherwise would be a lie.
        """
        if self._closed:
            raise BrowserGone(self.dead_reason or "not connected")
        ident = self._next_id()
        slot = _Pending(threading.Event())
        with self._lock:
            self._pending[ident] = slot
        frame: dict[str, Any] = {"id": ident, "method": method}
        if params:
            frame["params"] = params
        if session:
            frame["sessionId"] = session
        try:
            self.ws.send_text(json.dumps(frame))
        except WebSocketClosed as exc:
            self._forget(ident)
            self.dead_reason = str(exc)
            raise BrowserGone(f"{method}: {exc}") from exc
        except WebSocketError as exc:
            self._forget(ident)
            raise CDPError(f"{method}: could not be sent: {exc}") from exc

        budget = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + max(0.0, budget)
        while not slot.event.wait(_TICK):
            if stop is not None and stop():
                self._forget(ident)
                raise CDPCancelled(f"{method} cancelled")
            if time.monotonic() >= deadline:
                self._forget(ident)
                raise CDPTimeout(f"{method} did not answer within {budget:g}s")
        self._forget(ident)
        if slot.failure is not None:
            raise slot.failure
        reply = slot.reply or {}
        error = reply.get("error")
        if isinstance(error, dict):
            detail = error.get("data")
            message = str(error.get("message") or "failed")
            raise CDPError(f"{method}: {message}" + (f" ({detail})" if detail else ""))
        result = reply.get("result")
        return result if isinstance(result, dict) else {}

    # -- events -------------------------------------------------------------

    @property
    def cursor(self) -> int:
        """The current event number.  Take this *before* triggering the thing
        you intend to wait for, or the event can land before you look."""
        with self._cond:
            return self._seq

    def recent(self, *, method: str = "", session: str | None = None, limit: int = 50) -> list[Event]:
        with self._cond:
            events = list(self._events)
        chosen = [
            e
            for e in events
            if (not method or e.method == method) and (session is None or e.session == session)
        ]
        return chosen[-limit:]

    def wait_event(
        self,
        methods: str | Sequence[str],
        *,
        since: int,
        timeout: float,
        session: str | None = None,
        predicate: Callable[[Event], bool] | None = None,
    ) -> Event | None:
        """The first matching event numbered `since` or later, else `None`."""
        wanted = (methods,) if isinstance(methods, str) else tuple(methods)
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while True:
                for event in self._events:
                    if event.seq < since or event.method not in wanted:
                        continue
                    if session is not None and event.session != session:
                        continue
                    if predicate is not None and not predicate(event):
                        continue
                    return event
                left = deadline - time.monotonic()
                if left <= 0 or self._closed:
                    return None
                self._cond.wait(min(left, _TICK))

    def add_listener(self, fn: Callable[[Event], None]) -> Callable[[Event], None]:
        with self._lock:
            self.listeners.append(fn)
        return fn

    def remove_listener(self, fn: Callable[[Event], None]) -> None:
        with self._lock:
            if fn in self.listeners:
                self.listeners.remove(fn)

    # -- the reader ---------------------------------------------------------

    def _pump(self) -> None:
        """Own every read from the socket and dispatch what arrives."""
        try:
            while not self._closed:
                try:
                    message = self.ws.recv(timeout=_TICK)
                except WebSocketClosed as exc:
                    self.dead_reason = f"devtools socket closed: {exc}"
                    break
                except WebSocketError as exc:
                    self.dead_reason = f"devtools protocol error: {exc}"
                    break
                if message is None:
                    continue
                if message.text is None:
                    self.skipped += 1  # binary on a JSON channel is not ours
                    continue
                try:
                    frame = json.loads(message.text)
                except ValueError:
                    self.skipped += 1  # a corrupt frame is counted, never fatal
                    continue
                if isinstance(frame, dict):
                    self._dispatch(frame)
                else:
                    self.skipped += 1
        finally:
            self._closed = True
            self._fail_all(BrowserGone(self.dead_reason or "devtools socket closed"))
            with self._cond:
                self._cond.notify_all()

    def _dispatch(self, frame: dict[str, Any]) -> None:
        ident = frame.get("id")
        if isinstance(ident, int):
            with self._lock:
                slot = self._pending.get(ident)
            if slot is None:
                self.skipped += 1  # a reply to a call that already gave up
                return
            slot.reply = frame
            slot.event.set()
            return
        method = frame.get("method")
        if not isinstance(method, str):
            self.skipped += 1
            return
        params = frame.get("params")
        with self._cond:
            self._seq += 1
            event = Event(method, params if isinstance(params, dict) else {}, str(frame.get("sessionId") or ""), self._seq)
            self._events.append(event)
            self._cond.notify_all()
        with self._lock:
            listeners = list(self.listeners)
        for fn in listeners:
            try:
                fn(event)
            except Exception:  # a bad listener must not kill the reader
                self.skipped += 1

    # -- targets ------------------------------------------------------------

    def targets(self, *, timeout: float | None = None) -> list[TargetInfo]:
        raw = self.send("Target.getTargets", timeout=timeout).get("targetInfos")
        out: list[TargetInfo] = []
        for info in raw if isinstance(raw, list) else ():
            if isinstance(info, dict):
                out.append(TargetInfo.from_data(info))
        return out

    def pages(self, *, timeout: float | None = None) -> list[TargetInfo]:
        return [t for t in self.targets(timeout=timeout) if t.type == "page"]

    def create_page(self, url: str = "about:blank", *, timeout: float | None = None) -> str:
        result = self.send("Target.createTarget", {"url": url}, timeout=timeout)
        target = result.get("targetId")
        if not isinstance(target, str) or not target:
            raise CDPError("target.createtarget answered without a targetid")
        return target


@dataclass(frozen=True, slots=True)
class TargetInfo:
    target_id: str
    type: str
    title: str
    url: str
    attached: bool = False

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> TargetInfo:
        return cls(
            target_id=str(data.get("targetId") or ""),
            type=str(data.get("type") or ""),
            title=str(data.get("title") or ""),
            url=str(data.get("url") or ""),
            attached=bool(data.get("attached")),
        )


# -- accessibility -----------------------------------------------------------


def _ax_text(value: Any) -> str:
    """An AX property's value, which is always wrapped in `{"value": ...}`."""
    if isinstance(value, dict):
        inner = value.get("value")
        if isinstance(inner, (str, int, float, bool)):
            return str(inner)
        return ""
    return str(value) if isinstance(value, (str, int, float)) else ""


@dataclass(frozen=True, slots=True)
class AXNode:
    """One node of a flattened accessibility tree, addressable by `ref`."""

    ref: str
    role: str
    name: str
    value: str
    backend: int | None
    depth: int

    def line(self, *, width: int = 160) -> str:
        parts = [f"{'  ' * self.depth}{self.role or 'node'}"]
        if self.name:
            parts.append(f'"{self.name}"')
        if self.value and self.value != self.name:
            parts.append(f"value={self.value!r}")
        if self.backend is not None:
            parts.append(f"[ref={self.ref}]")
        line = " ".join(parts)
        return line if len(line) <= width else line[: width - 1] + "…"


def ax_nodes(raw: Sequence[Any], *, limit: int = AX_LIMIT) -> list[AXNode]:
    """Flatten `Accessibility.getFullAXTree` into readable, referenced nodes.

    Ignored nodes are dropped but their children are kept at the parent's
    depth: an `ignored` wrapper is an implementation detail of the page, and
    hiding its subtree would hide the button inside it.  Refs are assigned in
    document order so they are stable for as long as the page is, which is the
    contract the caller acts on after reading a snapshot.
    """
    by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for node in raw:
        if not isinstance(node, dict):
            continue
        ident = str(node.get("nodeId") or "")
        if ident and ident not in by_id:
            by_id[ident] = node
            order.append(ident)
    children_of_someone = {
        str(child)
        for node in by_id.values()
        for child in (node.get("childIds") or ())
    }
    roots = [ident for ident in order if ident not in children_of_someone] or order[:1]

    out: list[AXNode] = []
    seen: set[str] = set()
    stack: list[tuple[str, int]] = [(ident, 0) for ident in reversed(roots)]
    counter = 0
    while stack and len(out) < limit:
        ident, depth = stack.pop()
        if ident in seen or ident not in by_id:
            continue
        seen.add(ident)
        node = by_id[ident]
        kids = [str(c) for c in (node.get("childIds") or ())]
        if node.get("ignored"):
            stack.extend((kid, depth) for kid in reversed(kids))
            continue
        backend = node.get("backendDOMNodeId")
        counter += 1
        out.append(
            AXNode(
                ref=f"e{counter}",
                role=_ax_text(node.get("role")),
                name=_ax_text(node.get("name")),
                value=_ax_text(node.get("value")),
                backend=int(backend) if isinstance(backend, int) else None,
                depth=depth,
            )
        )
        stack.extend((kid, depth + 1) for kid in reversed(kids))
    return out


# -- one page ----------------------------------------------------------------


def key_event(spec: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(keyDown, keyUp)` parameter dicts for a key like `ctrl+a` or `Enter`."""
    pieces = [p for p in spec.replace(" ", "").split("+") if p] or [spec]
    name = pieces[-1]
    modifiers = 0
    for piece in pieces[:-1]:
        bit = _MODIFIERS.get(piece.lower())
        if bit is None:
            raise CDPError(f"unknown modifier {piece!r}. known: alt, ctrl, meta, shift")
        modifiers |= bit
    known = KEYS.get(name.lower())
    if known is not None:
        key, code, virtual, text = known
    elif len(name) == 1:
        key, code, virtual, text = name, f"Key{name.upper()}", ord(name.upper()), name
    else:
        raise CDPError(f"unknown key {name!r}. known: " + ", ".join(sorted(KEYS)))
    # A modified key must not also insert its character: ctrl+a selects, it
    # does not type an "a".  Only shift keeps text, and then in upper case.
    if modifiers & (MOD_ALT | MOD_CTRL | MOD_META):
        text = ""
    elif modifiers & MOD_SHIFT and len(text) == 1:
        text = text.upper()
        key = key.upper() if len(key) == 1 else key
    down: dict[str, Any] = {
        "type": "keyDown" if text else "rawKeyDown",
        "key": key,
        "code": code,
        "windowsVirtualKeyCode": virtual,
        "nativeVirtualKeyCode": virtual,
        "modifiers": modifiers,
    }
    if text:
        down["text"] = text
        down["unmodifiedText"] = text
    up = {"type": "keyUp", "key": key, "code": code, "windowsVirtualKeyCode": virtual, "modifiers": modifiers}
    return down, up


@dataclass(slots=True)
class Box:
    """An element's content box in CSS pixels, plus its centre."""

    x: float
    y: float
    width: float
    height: float

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def visible(self) -> bool:
        return self.width > 0 and self.height > 0


class Page:
    """One attached page target: navigation, input, evaluation, snapshots.

    The `stop` attribute is a predicate the owner sets once per tool call and
    every request then honours.  Threading a `stop=` argument through fifteen
    methods would be noise; a page belongs to exactly one caller at a time.
    """

    __slots__ = ("_refs", "client", "console", "detached", "session", "stop", "target_id", "timeout")

    def __init__(self, client: CDPClient, session: str, target_id: str, *, timeout: float = 20.0) -> None:
        self.client = client
        self.session = session
        self.target_id = target_id
        self.timeout = timeout
        self.console: deque[str] = deque(maxlen=CONSOLE_MEMORY)
        self.detached = False
        self.stop: Callable[[], bool] | None = None
        self._refs: dict[str, int] = {}
        client.add_listener(self._absorb)

    # -- lifecycle ----------------------------------------------------------

    @classmethod
    def attach(cls, client: CDPClient, target_id: str, *, timeout: float = 20.0) -> Page:
        """Attach flat: session messages arrive on the one socket, tagged."""
        result = client.send(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
            timeout=timeout,
        )
        session = result.get("sessionId")
        if not isinstance(session, str) or not session:
            raise CDPError(f"could not attach to target {target_id}")
        page = cls(client, session, target_id, timeout=timeout)
        page.enable()
        return page

    def enable(self) -> None:
        """Turn on the domains this class uses.

        `Accessibility` and `Log` are best-effort: an older build may not have
        them, and a snapshot that loses console history is far better than a
        page that refuses to open.
        """
        self.call("Page.enable")
        self.call("Runtime.enable")
        self.call("DOM.enable")
        self.call("Page.setLifecycleEventsEnabled", {"enabled": True})
        for optional in ("Accessibility.enable", "Log.enable"):
            try:
                self.call(optional)
            except CDPError:
                pass

    def close(self) -> None:
        """Detach and close the tab.  Never raises; this is a teardown path."""
        self.client.remove_listener(self._absorb)
        for method, params in (
            ("Target.closeTarget", {"targetId": self.target_id}),
            ("Target.detachFromTarget", {"sessionId": self.session}),
        ):
            try:
                self.client.send(method, params, timeout=2.0)
            except (CDPError, BrowserGone):
                pass
        self.detached = True

    # -- plumbing -----------------------------------------------------------

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.client.send(
            method,
            params,
            session=self.session,
            timeout=self.timeout if timeout is None else timeout,
            stop=self.stop,
        )

    def _absorb(self, event: Event) -> None:
        """Collect console output and notice being detached."""
        if event.method == "Target.detachedFromTarget" and event.params.get("sessionId") == self.session:
            self.detached = True
            return
        if event.session != self.session:
            return
        if event.method == "Runtime.consoleAPICalled":
            kind = str(event.params.get("type") or "log")
            args = event.params.get("args")
            rendered = " ".join(_remote_text(a) for a in args if isinstance(a, dict)) if isinstance(args, list) else ""
            self.console.append(f"{kind}: {rendered}".rstrip())
        elif event.method == "Runtime.exceptionThrown":
            details = event.params.get("exceptionDetails")
            self.console.append(f"exception: {_exception_text(details)}")
        elif event.method == "Log.entryAdded":
            entry = event.params.get("entry")
            if isinstance(entry, dict):
                self.console.append(f"{entry.get('level') or 'log'}: {str(entry.get('text') or '').strip()}")

    # -- navigation ---------------------------------------------------------

    def navigate(self, url: str, *, timeout: float = 30.0) -> str:
        """Go to `url` and wait for the load event.  Returns the settled url.

        Waiting on `Page.loadEventFired` *or* `Page.frameStoppedLoading` for the
        navigated frame, because a page that fails to load still stops loading
        and the caller deserves to be told rather than left at the deadline.
        """
        cursor = self.client.cursor
        result = self.call("Page.navigate", {"url": url}, timeout=timeout)
        failure = result.get("errorText")
        if failure:
            raise CDPError(f"could not load {url}: {failure}")
        frame = result.get("frameId")
        event = self.client.wait_event(
            ("Page.loadEventFired", "Page.frameStoppedLoading"),
            since=cursor,
            timeout=timeout,
            session=self.session,
            predicate=lambda e: e.method == "Page.loadEventFired" or e.params.get("frameId") == frame,
        )
        if event is None:
            raise CDPTimeout(f"{url} did not finish loading within {timeout:g}s")
        return self.url()

    def url(self) -> str:
        return str(self.evaluate_value("location.href") or "")

    def title(self) -> str:
        return str(self.evaluate_value("document.title") or "")

    def wait_for(self, expression: str, *, timeout: float = 10.0, interval: float = 0.1) -> bool:
        """Poll `expression` until it is truthy.  A predicate beats a sleep:
        a fixed sleep is either a waste or a flake, usually both."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                if self.evaluate_value(expression):
                    return True
            except CDPError:
                pass  # a not-yet-defined symbol is a "no", not a failure
            if time.monotonic() >= deadline:
                return False
            if self.stop is not None and self.stop():
                raise CDPCancelled("wait_for cancelled")
            time.sleep(interval)

    # -- evaluation ---------------------------------------------------------

    def evaluate(self, expression: str, *, await_promise: bool = False) -> dict[str, Any]:
        """`Runtime.evaluate`, raising the page's own exception as an error."""
        result = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
                "userGesture": True,
            },
        )
        details = result.get("exceptionDetails")
        if details:
            raise CDPError(f"evaluate failed: {_exception_text(details)}")
        remote = result.get("result")
        return remote if isinstance(remote, dict) else {}

    def evaluate_value(self, expression: str, *, await_promise: bool = False) -> Any:
        remote = self.evaluate(expression, await_promise=await_promise)
        if "value" in remote:
            return remote["value"]
        return remote.get("description")

    # -- elements -----------------------------------------------------------

    def snapshot(self, *, limit: int = AX_LIMIT) -> list[AXNode]:
        """The accessibility tree, and the refs the caller may now act on.

        This is the cheap, reliable view for a model: text and roles, no
        pixels, no guessing at coordinates.  Taking a snapshot replaces the
        ref table, because refs describe the tree the caller just read.
        """
        result = self.call("Accessibility.getFullAXTree")
        raw = result.get("nodes")
        nodes = ax_nodes(raw if isinstance(raw, list) else (), limit=limit)
        self._refs = {node.ref: node.backend for node in nodes if node.backend is not None}
        return nodes

    @property
    def refs(self) -> dict[str, int]:
        return dict(self._refs)

    def resolve(self, selector: str) -> dict[str, Any]:
        """`{"nodeId": n}` or `{"backendNodeId": n}` for a selector or ref."""
        selector = selector.strip()
        if not selector:
            raise CDPError("an element needs a css selector or a ref like ref=e5")
        match = _REF.match(selector)
        if match:
            ref = match.group(1).lower()
            backend = self._refs.get(ref)
            if backend is None:
                raise CDPError(f"no element for {ref}. take a snapshot first, then use its refs")
            return {"backendNodeId": backend}
        root = self.call("DOM.getDocument", {"depth": 0}).get("root")
        node = root.get("nodeId") if isinstance(root, dict) else None
        if not isinstance(node, int):
            raise CDPError("dom.getdocument answered without a root node")
        found = self.call("DOM.querySelector", {"nodeId": node, "selector": selector}).get("nodeId")
        if not isinstance(found, int) or found == 0:
            raise CDPError(f"no element matches {selector!r}")
        return {"nodeId": found}

    def box(self, selector: str, *, scroll: bool = True) -> Box:
        """The element's content box, scrolled into view first if asked.

        Scrolling first matters: `Input.dispatchMouseEvent` takes viewport
        coordinates, so a box below the fold is a click on whatever happens to
        be at those coordinates instead.
        """
        handle = self.resolve(selector)
        if scroll:
            try:
                self.call("DOM.scrollIntoViewIfNeeded", dict(handle))
            except CDPError:
                pass  # not scrollable, or already there; the box still stands
            handle = self.resolve(selector) if "nodeId" in handle else handle
        model = self.call("DOM.getBoxModel", dict(handle)).get("model")
        quad = model.get("content") if isinstance(model, dict) else None
        if not isinstance(quad, list) or len(quad) < 8:
            raise CDPError(f"{selector!r} has no box; it may be hidden or detached")
        xs = [float(quad[i]) for i in (0, 2, 4, 6)]
        ys = [float(quad[i]) for i in (1, 3, 5, 7)]
        return Box(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    # -- input --------------------------------------------------------------

    def click(self, selector: str, *, button: str = "left", clicks: int = 1) -> Box:
        """A real mouse press at the element's centre, hit-testing included."""
        target = self.box(selector)
        if not target.visible:
            raise CDPError(f"{selector!r} has a zero-sized box, so nothing can click it")
        x, y = target.centre
        common = {"x": x, "y": y, "button": button, "clickCount": max(1, clicks)}
        self.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
        self.call("Input.dispatchMouseEvent", {"type": "mousePressed", **common})
        self.call("Input.dispatchMouseEvent", {"type": "mouseReleased", **common})
        return target

    def press(self, key: str) -> None:
        down, up = key_event(key)
        self.call("Input.dispatchKeyEvent", down)
        self.call("Input.dispatchKeyEvent", up)

    def type_text(self, text: str) -> None:
        """Type as keystrokes, so a page's keydown handlers actually run."""
        for char in text:
            if char in "\n\r":
                self.press("Enter")
                continue
            if char == "\t":
                self.press("Tab")
                continue
            down, up = key_event(char) if len(char) == 1 else key_event("space")
            self.call("Input.dispatchKeyEvent", down)
            self.call("Input.dispatchKeyEvent", up)

    def fill(self, selector: str, value: str) -> None:
        """Focus the field, clear it, and insert `value` as real text input."""
        self.call("DOM.focus", dict(self.resolve(selector)))
        self.press("ctrl+a")
        if value:
            self.call("Input.insertText", {"text": value})
        else:
            self.press("delete")

    def scroll(self, *, dx: float = 0.0, dy: float = 0.0, x: float | None = None, y: float | None = None) -> None:
        """Wheel over a point, defaulting to the middle of the viewport."""
        if x is None or y is None:
            middle = self.evaluate_value("[innerWidth / 2 | 0, innerHeight / 2 | 0]")
            if isinstance(middle, list) and len(middle) == 2:
                x = float(middle[0]) if x is None else x
                y = float(middle[1]) if y is None else y
            else:
                x, y = (x if x is not None else 10.0), (y if y is not None else 10.0)
        self.call(
            "Input.dispatchMouseEvent",
            {"type": "mouseWheel", "x": x, "y": y, "deltaX": dx, "deltaY": dy},
        )

    # -- output -------------------------------------------------------------

    def screenshot(self, *, full_page: bool = False, timeout: float = 30.0) -> bytes:
        """PNG bytes.  Deliberately not base64 to the caller: a screenshot
        inlined into a model's context is tens of thousands of useless tokens.
        """
        params: dict[str, Any] = {"format": "png"}
        if full_page:
            metrics = self.call("Page.getLayoutMetrics", timeout=timeout)
            size = metrics.get("cssContentSize") or metrics.get("contentSize")
            if isinstance(size, dict):
                params["captureBeyondViewport"] = True
                params["clip"] = {
                    "x": 0,
                    "y": 0,
                    "width": float(size.get("width") or 0) or 1.0,
                    "height": float(size.get("height") or 0) or 1.0,
                    "scale": 1,
                }
        data = self.call("Page.captureScreenshot", params, timeout=timeout).get("data")
        if not isinstance(data, str) or not data:
            raise CDPError("page.capturescreenshot answered without image data")
        try:
            return base64.b64decode(data, validate=True)
        except (ValueError, TypeError) as exc:
            raise CDPError(f"screenshot was not valid base64: {exc}") from exc

    def messages(self, limit: int = 50) -> list[str]:
        return list(self.console)[-max(1, limit):]


def _remote_text(remote: dict[str, Any]) -> str:
    """A `Runtime.RemoteObject` as the one line a reader wants."""
    if "value" in remote:
        value = remote["value"]
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    for key in ("unserializableValue", "description", "className"):
        if remote.get(key):
            return str(remote[key])
    return str(remote.get("type") or "")


def _exception_text(details: Any) -> str:
    """The most useful single line out of a CDP `ExceptionDetails`."""
    if not isinstance(details, dict):
        return "unknown error"
    exception = details.get("exception")
    if isinstance(exception, dict):
        described = exception.get("description") or _remote_text(exception)
        if described:
            return str(described).splitlines()[0]
    text = str(details.get("text") or "").strip()
    return text or "unknown error"
