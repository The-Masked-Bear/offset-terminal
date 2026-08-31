"""A browser must not outlive the offset that started it.

`start_new_session=True` is what lets `close()` signal the browser's whole
process group without signalling offset too — but detaching also means a shell
that is killed outright, rather than exiting through its `finally`, used to
leave a headless browser running that nobody can see. Measured before the fix:
eleven chromium processes surviving a SIGKILLed parent, which on a Raspberry Pi
is most of the free memory.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("chromium") is None
    and shutil.which("chromium-browser") is None
    and shutil.which("google-chrome") is None,
    reason="no browser on PATH",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LAUNCHER = textwrap.dedent(
    """
    import sys, threading, time
    from pathlib import Path
    sys.path.insert(0, {root!r})
    from offset.tools.base import ToolContext
    from offset.tools.web import browser_tools
    tool = browser_tools()[0]
    ctx = ToolContext(cwd=Path('/tmp'), root=Path('/'),
                      cancel=threading.Event(), timeout=60.0)
    result = tool.run({{'action': 'open', 'url': 'data:text/html,<h1>x</h1>'}}, ctx)
    print('READY' if result.ok else 'FAILED: ' + (result.error or ''), flush=True)
    time.sleep(120)
    """
)


def _offset_browsers() -> list[int]:
    """Only browsers this project launched, identified by their profile dir."""
    found = subprocess.run(
        ["pgrep", "-f", "offset-browser-"], capture_output=True, text=True
    )
    return [int(p) for p in found.stdout.split() if p.strip().isdigit()]


@pytest.fixture
def no_browsers():
    """Refuse to run if the machine already has one, and tidy up afterwards."""
    if _offset_browsers():
        pytest.skip("an offset browser is already running; cannot attribute the result")
    yield
    for pid in _offset_browsers():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.5)
    return predicate()


def test_a_browser_dies_with_a_killed_offset(no_browsers):
    """The crash path: no `finally` runs, so only the kernel can clean up."""
    child = subprocess.Popen(
        [sys.executable, "-c", LAUNCHER.format(root=ROOT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        line = child.stdout.readline().strip() if child.stdout else ""
        if not line.startswith("READY"):
            pytest.skip(f"the browser would not start here: {line[:80]}")
        assert _wait_for(lambda: bool(_offset_browsers()), 30), "no browser appeared"

        child.kill()          # SIGKILL: nothing in the parent gets to tidy up
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()

    assert _wait_for(lambda: not _offset_browsers(), 25), (
        f"the browser outlived its killed parent: {_offset_browsers()}"
    )


def test_close_still_reaps_the_browser_on_the_normal_path(no_browsers):
    """The fix must not have broken ordinary teardown."""
    import threading
    from pathlib import Path

    from offset.tools.base import ToolContext
    from offset.tools.web import browser_tools, close_all

    tool = browser_tools()[0]
    ctx = ToolContext(cwd=Path("/tmp"), root=Path("/"),
                      cancel=threading.Event(), timeout=60.0)
    opened = tool.run({"action": "open", "url": "data:text/html,<h1>x</h1>"}, ctx)
    if not opened.ok:
        pytest.skip(f"the browser would not start here: {opened.error}")
    try:
        assert _offset_browsers(), "nothing was launched"
        tool.run({"action": "close"}, ctx)
        assert _wait_for(lambda: not _offset_browsers(), 20), "close left a browser behind"
    finally:
        close_all()
