"""offset without a terminal.

The bridge has always been able to drive a real turn - `prompt` calls
`agent.run()` directly rather than posting into the TUI's input queue - but it
only existed while a TUI was open.  That is the wrong shape for the thing
people actually want: an editor, or a machine across an SSH hop, that can talk
to offset whether or not somebody is sitting in front of a terminal.

So this is the same session, the same toolbox and the same bridge, with the
terminal removed and a signal handler put in its place.  There is no second
protocol and no second agent: a daemon is a shell that nobody is looking at.

Two things are deliberate.

**Loopback unless told otherwise.** A unix socket by default, TCP only when
asked, and a non-loopback bind only when the address says so in full.  The
token is mandatory on every transport, but a token is not a reason to put an
agent that can run shell commands on an interface by accident.

**Idle shutdown is opt-in.** A daemon that exits when the last editor
disconnects is convenient on a laptop and wrong on a build box, so `--idle`
takes the number of seconds and defaults to never.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Final

from offset.core import bridge as bridge_mod

#: How often the run loop wakes to check for idleness and shutdown.  Short
#: enough that ctrl-c feels immediate, long enough to cost nothing.
TICK: Final = 0.25

#: Signals that mean "stop".  SIGHUP is included because a daemon started over
#: SSH should not outlive the connection unless it was detached on purpose.
STOP_SIGNALS: Final = ("SIGINT", "SIGTERM", "SIGHUP")


class Daemon:
    """A running headless offset, and the reasons it might stop."""

    def __init__(self, state: Any, bridge: bridge_mod.Bridge, *,
                 idle: float = 0.0) -> None:
        self.state = state
        self.bridge = bridge
        #: Seconds with no connected client before shutting down.  0 = never.
        self.idle = max(0.0, idle)
        self.stopping = threading.Event()
        self.reason = ""
        self._last_seen = time.time()

    @property
    def clients(self) -> int:
        return len(getattr(self.bridge, "_clients", ()) or ())

    def stop(self, reason: str = "asked to stop") -> None:
        if not self.stopping.is_set():
            self.reason = reason
            self.stopping.set()

    def describe(self) -> dict[str, Any]:
        """Everything a client needs to connect, and nothing secret.

        The token lives in its own `0o600` file and is named here rather than
        inlined: a descriptor gets pasted into issues.
        """
        bridge = self.bridge
        where = (str(bridge.socket_path) if bridge.unix
                 else f"{bridge.host or '127.0.0.1'}:{bridge.port}")
        return {
            "pid": os.getpid(),
            "transport": "unix" if bridge.unix else "tcp",
            "address": where,
            "descriptor": str(bridge.descriptor_path),
            "token_path": str(bridge.token_path),
            "workspace": str(getattr(self.state, "workspace", "") or ""),
            "model": str(getattr(self.state, "model", "") or ""),
            "tools": len(list(getattr(self.state, "toolbox", ()) or ())),
        }

    def run(self) -> str:
        """Block until something says stop.  Returns why."""
        while not self.stopping.wait(TICK):
            if self.clients:
                self._last_seen = time.time()
            elif self.idle and (time.time() - self._last_seen) > self.idle:
                self.stop(f"idle for {self.idle:.0f}s with no client")
        return self.reason or "stopped"


def _install_signals(daemon: Daemon) -> None:
    """Ask the process to stop on the usual signals.

    Wrapped because signal handlers can only be installed on the main thread,
    and a daemon embedded in something else (a test, a supervisor) may not be
    on one.  Failing to install a handler is not a reason to refuse to run.
    """
    def handler(number: int, _frame: Any) -> None:
        daemon.stop(f"signal {signal.Signals(number).name}")

    for name in STOP_SIGNALS:
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            signal.signal(number, handler)
        except (ValueError, OSError):
            pass  # not the main thread, or the platform has no such signal


def serve(
    workspace: str | os.PathLike[str] = ".",
    *,
    model: str | None = None,
    listen: str = "",
    idle: float = 0.0,
    quiet: bool = False,
    build: Callable[..., Any] | None = None,
    out=None,
) -> int:
    """Run offset headless until stopped.  Returns a process exit status.

    `build` is injected so a test can drive the whole lifecycle without
    constructing a real session, and so this module never imports the shell at
    module scope - the daemon is startup-critical and the shell is not cheap.
    """
    stream = out if out is not None else sys.stdout
    if build is None:
        from offset.shell.app import build_state as build

    root = Path(workspace).expanduser().resolve()
    state = build(root, model=model)

    jobs = None
    store = getattr(state, "jobs", None)
    if store is not None and hasattr(store, "list"):
        def jobs() -> list[dict[str, Any]]:
            try:
                return [dict(j) if isinstance(j, dict) else {"id": str(j)} for j in store.list()]
            except Exception:
                return []

    bridge_mod.install(state, jobs=jobs, listen=listen)
    bridge = bridge_mod.active()
    if bridge is None or not bridge.listening:
        problems = list(getattr(bridge, "problems", ())) or ["the bridge did not start"]
        for line in problems:
            print(f"offset daemon: {line}", file=sys.stderr)
        return 1

    daemon = Daemon(state, bridge, idle=idle)
    _install_signals(daemon)

    if not quiet:
        facts = daemon.describe()
        print(json.dumps(facts, indent=2), file=stream, flush=True)

    try:
        reason = daemon.run()
    finally:
        bridge_mod.uninstall()
        _shutdown(state)
    if not quiet:
        print(f"offset daemon: {reason}", file=stream, flush=True)
    return 0


def _shutdown(state: Any) -> None:
    """Give the session the same ending the TUI would.

    A daemon that exits without saving has silently lost the user's work, and
    one that leaves a browser or a language server running has leaked a process
    nobody can see.  Every step is independent: one failing must not skip the
    rest.
    """
    from offset.shell.app import _shutdown_debuggees, _shutdown_language_servers
    from offset.tools.web import close_all as close_browsers

    for step in (
        lambda: state.eggs.save(),
        lambda: state.session.close(),
        lambda: state.mcp.disconnect_all() if getattr(state, "mcp", None) else None,
        close_browsers,
        lambda: _shutdown_language_servers(state),
        lambda: _shutdown_debuggees(state),
    ):
        try:
            step()
        except Exception:
            pass


def command(args: Any) -> int:
    """`offset daemon` entry point."""
    return serve(
        getattr(args, "workspace", ".") or ".",
        model=getattr(args, "model", None),
        listen=getattr(args, "listen", "") or "",
        idle=float(getattr(args, "idle", 0) or 0),
        quiet=bool(getattr(args, "quiet", False)),
    )
