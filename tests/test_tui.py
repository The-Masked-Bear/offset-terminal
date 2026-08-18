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
            "pageup": b"\x1b[5~",
            "pagedown": b"\x1b[6~",
            "end": b"\x1b[4~",
            "s-up": b"\x1b[1;2A",
            "s-down": b"\x1b[1;2B",
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
        """Case- and space-insensitive: the UI upper-cases and letter-spaces,
        and a test should not break when a label changes case."""
        target = re.sub(r"\s+", "", needle).lower()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.pump(0.25)
            if target in self.squeeze().lower():
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
def raw_term(tmp_path):
    """A terminal parked on the startup permission question."""
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    terminal = Terminal(tmp_path)
    yield terminal
    terminal.close()


@pytest.fixture()
def term(raw_term):
    """Past the permission question, on the safe choice."""
    assert raw_term.wait_for("HOW MUCH OF THIS MACHINE"), (
        f"the startup permission question never appeared:\n{raw_term.text}"
    )
    raw_term.key("enter")
    # The banner is a marquee, so its words scroll off; the status bar is the
    # only text guaranteed to be on screen at any instant.
    assert raw_term.wait_for("ctrl-c x2 quit"), f"the shell never drew its chrome:\n{raw_term.text}"
    return raw_term


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


# -- the startup permission question ----------------------------------------


def test_it_asks_for_permission_before_doing_anything(raw_term):
    """The user asked for this explicitly: decide the blast radius at startup."""
    assert raw_term.wait_for("HOW MUCH OF THIS MACHINE"), raw_term.text
    squeezed = raw_term.squeeze()
    assert "WORKSPACEONLY" in squeezed
    assert "FULLSYSTEMACCESS" in squeezed
    # The cards sit side by side, so a squeezed full-screen dump interleaves
    # their rows: match fragments that fit on one row, not whole sentences.
    lowered = squeezed.lower()
    for fragment in ("readandwriteany", "notjustthisfolder", "runarbitraryshell"):
        assert fragment in lowered, (
            f"the prompt must say plainly what full access means; missing {fragment!r}:\n{raw_term.text}"
        )
    assert "ctrl-cx2quit" not in squeezed.lower(), "the shell must not be usable before answering"


def test_enter_alone_never_grants_full_access(raw_term):
    assert raw_term.wait_for("HOW MUCH OF THIS MACHINE")
    raw_term.key("enter")
    assert raw_term.wait_for("ctrl-c x2 quit"), raw_term.text
    squeezed = raw_term.squeeze()
    assert "workspaceonly" in squeezed.lower(), raw_term.text
    assert "fullsystemaccess,granted" not in squeezed.lower(), "Enter must pick the safe option"


def test_pressing_f_grants_full_access(raw_term):
    assert raw_term.wait_for("HOW MUCH OF THIS MACHINE")
    raw_term.type("f", settle=0.8)
    assert raw_term.wait_for("ctrl-c x2 quit"), raw_term.text
    assert "FULL" in raw_term.squeeze()


def test_the_answer_is_remembered_for_next_time(tmp_path):
    """A second launch in the same workspace must not re-ask."""
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = Terminal(tmp_path)
    try:
        assert first.wait_for("HOW MUCH OF THIS MACHINE")
        first.key("enter")
        assert first.wait_for("ctrl-c x2 quit")
    finally:
        first.close()

    second = Terminal(tmp_path)
    try:
        assert second.wait_for("ctrl-c x2 quit"), f"the second launch never got going:\n{second.text}"
        assert "HOWMUCHOFTHISMACHINE" not in second.squeeze(), "the grant was not remembered"
    finally:
        second.close()


def test_the_document_and_system_tools_are_registered(term):
    term.type("/tools")
    term.key("enter")
    assert term.wait_for("allenabled"), term.text
    squeezed = term.squeeze()
    for tool in ("document", "system", "file", "open"):
        assert tool in squeezed, f"{tool} missing from the tool list:\n{term.text}"
    assert "danger" in squeezed


# -- scrollback and project context -----------------------------------------


def test_you_can_scroll_back_through_history(term):
    """Before this the transcript was a fixed tail with no way to read back."""
    for i in range(16):
        term.type(f"question {i}")
        term.key("enter", settle=0.12)
    assert term.wait_for("question 15"), term.text

    term.key("pageup", settle=0.8)
    squeezed = term.squeeze()
    assert "MOREBELOW" in squeezed, f"no hidden-line indicator:\n{term.text}"
    assert "question15" not in squeezed, "the view should have moved off the bottom"
    assert "\u2588" in term.text, "the scrollbar thumb should be drawn"


def test_end_returns_to_following(term):
    for i in range(16):
        term.type(f"item {i}")
        term.key("enter", settle=0.12)
    term.key("pageup", settle=0.6)
    assert "MOREBELOW" in term.squeeze()
    term.key("end", settle=0.6)
    assert "MOREBELOW" not in term.squeeze(), "END must jump back to the bottom"


def test_project_instructions_are_discovered(raw_term, tmp_path):
    (tmp_path / "AGENTS.md").write_text(
        "# House rules\nAlways run the tests before claiming a fix.\n", encoding="utf-8"
    )
    assert raw_term.wait_for("HOW MUCH OF THIS MACHINE")
    raw_term.key("enter")
    assert raw_term.wait_for("ctrl-c x2 quit")

    raw_term.type("/context")
    raw_term.key("enter")
    assert raw_term.wait_for("AGENTS.md"), raw_term.text
    assert "appended to the system prompt" in raw_term.text


# -- usability: the things that made it feel broken --------------------------


def test_two_ctrl_c_presses_quit(term):
    """What the user asked for: ctrl-c twice, not an undiscoverable ctrl-d."""
    term.key("ctrl-c", settle=0.5)
    assert "ctrl-c again" in term.text.lower(), f"the first press must say what happens next:\n{term.text}"

    term.key("ctrl-c", settle=0.8)
    for _ in range(30):
        if os.waitpid(term.pid, os.WNOHANG) != (0, 0):
            return
        time.sleep(0.1)
    pytest.fail("the second ctrl-c did not quit")


def test_one_ctrl_c_alone_does_not_quit(term):
    term.key("ctrl-c", settle=0.5)
    time.sleep(1.0)
    assert os.waitpid(term.pid, os.WNOHANG) == (0, 0), "a single ctrl-c must never quit"
    assert term.wait_for("ctrl-c x2 quit"), "the shell should still be running"


def test_typing_a_slash_shows_the_commands(term):
    """Regression: a completer was attached but nothing ever drew its results."""
    term.type("/mo", settle=1.2)
    body = term.text
    assert "/model" in body, f"no completion menu appeared:\n{body}"
    assert "/models" in body
    assert "switch model" in body, "the menu should carry each command's summary"


def test_the_completion_menu_narrows_as_you_type(term):
    term.type("/mod", settle=1.0)
    assert "/model" in term.text
    term.type("els", settle=1.0)
    assert "/models" in term.text


def test_an_empty_session_explains_itself(raw_term):
    """A blank twenty-row void reads as broken software."""
    assert raw_term.wait_for("HOW MUCH OF THIS MACHINE")
    raw_term.key("enter")
    assert raw_term.wait_for("ask for what you want"), raw_term.text
    squeezed = raw_term.squeeze().lower()
    for hint in ("/help", "/spec", "/model", "ctrl-ctwice"):
        assert hint.replace(" ", "") in squeezed, f"{hint} missing from the first screen"


def test_the_header_names_the_model_and_workspace(term):
    assert term.wait_for("offset"), term.text
    assert "mock" in term.text.lower()


def test_running_text_is_not_letter_spaced(term):
    """The status bar used to read 'M O C K  1 5  T O O L S'."""
    term.type("/tools")
    term.key("enter")
    assert term.wait_for("all enabled"), term.text
    status = term.text.rstrip().split("\n")[-1]
    assert "MOCK" in status, f"the model name should read as a word: {status!r}"
    assert "M O C K" not in status, f"running text must not be tracked: {status!r}"


def test_permissions_command_exists_because_the_consent_screen_promises_it(term):
    term.type("/permissions")
    term.key("enter")
    assert term.wait_for("workspace only"), term.text
    assert "revoke" in term.text.lower()


def test_permissions_can_grant_full_access_later(term):
    term.type("/permissions full")
    term.key("enter")
    assert term.wait_for("full system access"), term.text
    term.type("/tools")
    term.key("enter")
    assert term.wait_for("whole machine"), term.text
