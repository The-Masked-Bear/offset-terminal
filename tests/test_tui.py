"""End-to-end smoke test of the interactive shell.

A TUI is only honestly tested by running it: this spawns the real program in a
real pty, types real keystrokes, and reads the resulting screen back through a
VT100 emulator.  No mocking of the terminal, no asserting on ANSI strings.

The model is the scripted provider, so the test needs no network and no key.
"""

from __future__ import annotations

import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pyte = pytest.importorskip("pyte", reason="pyte renders the pty output into a readable grid")

REPO = Path(__file__).resolve().parents[1]
COLS, ROWS = 100, 32


class Terminal:
    """Runs offset in a pty and keeps a live picture of the screen."""

    def __init__(self, workspace: Path, *, model: str = "mock") -> None:
        self.screen = pyte.Screen(COLS, ROWS)
        self.stream = pyte.ByteStream(self.screen)
        import site

        # HOME is redirected so the run is isolated, which also hides the user
        # site directory: put it back on the path explicitly.
        extra = [str(REPO), *site.getsitepackages(), site.getusersitepackages()]
        env = {
            **os.environ,
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "COLUMNS": str(COLS),
            "LINES": str(ROWS),
            "PYTHONPATH": os.pathsep.join(extra),
            "OFFSET_HOME": str(workspace / ".offset-home"),
            "HOME": str(workspace / "home"),
        }
        (workspace / "home").mkdir(parents=True, exist_ok=True)
        self.pid, self.fd = pty.fork()
        if self.pid == 0:  # child
            os.chdir(str(workspace))
            os.execvpe(
                sys.executable,
                [sys.executable, "-m", "offset", "chat", "--model", model, "--workspace", str(workspace)],
                env,
            )
        self._resize()

    def _resize(self) -> None:
        import fcntl
        import struct
        import termios

        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

    def pump(self, seconds: float = 0.6) -> None:
        """Read whatever the program has drawn."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self.fd], [], [], 0.05)
            if not ready:
                continue
            try:
                data = os.read(self.fd, 65536)
            except OSError:
                return
            if not data:
                return
            self.stream.feed(data)

    def type(self, text: str, *, settle: float = 0.5) -> None:
        os.write(self.fd, text.encode())
        self.pump(settle)

    def key(self, code: str, *, settle: float = 0.4) -> None:
        codes = {
            "enter": b"\r",
            "down": b"\x1b[B",
            "up": b"\x1b[A",
            "escape": b"\x1b",
            "ctrl-d": b"\x04",
            "ctrl-c": b"\x03",
        }
        os.write(self.fd, codes[code])
        self.pump(settle)

    @property
    def text(self) -> str:
        return "\n".join(self.screen.display)

    def flat(self) -> str:
        """Screen text with tracking spaces collapsed, for easier matching."""
        return re.sub(r"\s+", " ", self.text)

    def squeeze(self) -> str:
        """Every space removed — survives letter-spacing entirely."""
        return re.sub(r"\s+", "", self.text)

    def wait_for(self, needle: str, timeout: float = 12.0) -> bool:
        target = re.sub(r"\s+", "", needle)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(0.25)
            if target in self.squeeze():
                return True
        return False

    def close(self) -> None:
        try:
            os.write(self.fd, b"\x04")
            time.sleep(0.3)
            os.kill(self.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        try:
            os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            pass


@pytest.fixture()
def term(tmp_path):
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    terminal = Terminal(tmp_path)
    # The banner is a marquee, so its words scroll off; the status bar is the
    # only text guaranteed to be on screen at any instant.
    assert terminal.wait_for("CTRL-D QUIT"), f"the shell never drew its chrome:\n{terminal.text}"
    yield terminal
    terminal.close()


# -- it starts --------------------------------------------------------------


def test_the_shell_starts_and_draws_its_chrome(term):
    assert term.wait_for("MOCK"), "the status bar should name the model"
    squeezed = term.squeeze()
    assert "TOOLS" in squeezed
    assert "AUTO-EDIT" in squeezed or "AUTOEDIT" in squeezed


def test_a_real_turn_streams_a_reply(term):
    term.type("summarise the plan")
    term.key("enter")
    assert term.wait_for("mockreply"), f"no reply appeared:\n{term.text}"
    assert "summarise the plan" in term.flat(), "the prompt should stay on screen"


def test_help_lists_commands(term):
    term.type("/help")
    term.key("enter")
    assert term.wait_for("/model"), term.text
    assert "/spec" in term.text and "/tools" in term.text


def test_the_model_picker_opens_and_switches(term):
    term.type("/model")
    term.key("enter")
    assert term.wait_for("CLAUDEOPUS"), f"the picker did not open:\n{term.text}"
    term.key("down")
    term.key("enter")
    assert term.wait_for("provider"), term.text


def test_login_masks_what_you_type(term):
    term.type("/login anthropic")
    term.key("enter")
    assert term.wait_for("APIKEY"), f"the login panel did not open:\n{term.text}"
    term.type("sk-super-secret")
    assert "sk-super-secret" not in term.text, "the key was echoed to the screen"
    assert "\u25cf" in term.text or "*" in term.text, "no masking dots were drawn"
    term.key("escape")


def test_tools_are_all_enabled(term):
    term.type("/tools")
    term.key("enter")
    assert term.wait_for("allenabled"), term.text
    for tool in ("read", "write", "bash", "fetch"):
        assert tool in term.text


def test_an_easter_egg_fires(term):
    term.type("/bear")
    term.key("enter")
    assert term.wait_for("AREYOUABEAR"), f"the bear did not answer:\n{term.text}"
    assert "No." in term.text


def test_the_trophy_room_opens(term):
    term.type("/eggs")
    term.key("enter")
    assert term.wait_for("EGGS"), term.text
    assert "?" in term.text, "undiscovered eggs should stay hidden"


def test_session_reports_itself(term):
    term.type("/session")
    term.key("enter")
    assert term.wait_for("entries"), term.text


def test_unknown_commands_are_told_off(term):
    term.type("/wharrgarbl")
    term.key("enter")
    assert term.wait_for("unknowncommand"), term.text


def test_ctrl_d_exits_cleanly(term):
    term.key("ctrl-d", settle=0.8)
    for _ in range(20):
        if os.waitpid(term.pid, os.WNOHANG) != (0, 0):
            break
        time.sleep(0.1)
    else:
        pytest.fail("the shell did not exit on ctrl-d")


def test_spec_really_runs_and_ranks_branches(term):
    """The old /spec claimed success and did nothing; this one must produce a
    ranked report from three isolated worktrees."""
    term.type("/spec 3 speed up the parser")
    term.key("enter")
    assert term.wait_for("adopttheleaderwith/adopt1"), term.text
    squeezed = term.squeeze()
    for angle in ("minimal", "rewrite", "test-first"):
        assert angle in squeezed, f"{angle} missing from:\n{term.text}"
    assert "lines" in squeezed, "the report should say how much each branch changed"


def test_adopt_without_a_run_is_refused(term):
    term.type("/adopt 1")
    term.key("enter")
    assert term.wait_for("run/specfirst"), term.text


def test_verify_command_is_settable(term):
    term.type("/verify pytest -q")
    term.key("enter")
    assert term.wait_for("pytest-q"), term.text
