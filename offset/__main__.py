"""Command line entry point."""

from __future__ import annotations

import argparse
import sys


def _autoupdate() -> None:
    """Install a waiting update, then become the new version.

    Deliberately quiet: it prints only when it actually does something, so the
    overwhelmingly common case of "already current" costs the user no output
    and no delay.  It reads the cache the previous run's background check left
    behind rather than the network, so an offline start is not paid for here.

    Every failure is swallowed. A program that refuses to start because it
    could not upgrade itself is worse than one running last week's build.
    """
    try:
        from offset.core.update import autoupdate, reexec
    except Exception:
        return
    try:
        outcome = autoupdate(echo=lambda line: print(line, flush=True))
    except Exception:
        return
    for line in outcome.report():
        print(line, flush=True)
    if outcome.acted:
        reexec()  # returns only if exec failed, in which case carry on


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offset", description="terminal coding agent")
    sub = parser.add_subparsers(dest="cmd")

    chat = sub.add_parser("chat", help="start an interactive session")
    chat.add_argument("--model", default=None,
                  help="model id; defaults to model.default from config, else the scripted mock")
    chat.add_argument("--workspace", default=".", help="directory the tools may touch")
    chat.add_argument("--approve", default=None,
                  choices=("safe", "auto-edit", "yolo", "full"),
                  help="approval mode; defaults to the stored permission grant")
    chat.add_argument("--resume", nargs="?", const="", default=None, metavar="ID",
                  help="carry on an earlier session; bare --resume takes the most recent")
    chat.add_argument("--continue", dest="continue_", action="store_true",
                  help="carry on the most recent session")

    sub.add_parser("login", help="sign in with your Google or GitHub account")
    sub.add_parser("sync", help="sync Offset Plus subscription status from your account")

    upgrade = sub.add_parser("upgrade", help="redeem a Gumroad licence key to unlock Offset Plus")
    upgrade.add_argument("key", help="the licence key from your Gumroad receipt")

    upd = sub.add_parser("update", help="check for a newer offset and install it")
    upd.add_argument("--check", action="store_true",
                 help="only report whether an update exists")

    demo = sub.add_parser("demo", help="render the design system")
    demo.add_argument("--once", action="store_true", help="print one frame and exit")
    demo.add_argument("--time", type=float, default=2.4, help="timestamp of the frame to print")
    demo.add_argument("--fps", type=float, default=24.0)
    demo.add_argument("--width", type=int, default=None)
    demo.add_argument("--height", type=int, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "login":
        from offset.auth import prompt_account_login
        prompt_account_login()
        return 0

    if args.cmd == "sync":
        from offset.auth import sync_command
        return sync_command()

    if args.cmd == "upgrade":
        from offset.auth import verify_direct_license_key
        return verify_direct_license_key(args.key)

    if args.cmd == "update":
        from offset.core.update import update_command
        return update_command(check_only=args.check)

    if args.cmd == "demo":
        from offset.ui import demo as demo_mod

        if args.once:
            print(demo_mod.frame(args.time, w=args.width, h=args.height))
            return 0
        return demo_mod.run(fps=args.fps)

    if args.cmd in (None, "chat"):
        # Before the shell, never during it: the modules a running session has
        # already imported cannot be swapped underneath it, so an update
        # applied mid-turn would leave half the program on the old version.
        # A successful update re-executes, so the user gets the new build in
        # the same invocation rather than being told to start again.
        _autoupdate()

        from offset.shell.app import main as chat_main

        resume = getattr(args, "resume", None)
        if getattr(args, "continue_", False) and resume is None:
            resume = ""  # bare --continue means "the most recent"
        return chat_main(
            workspace=getattr(args, "workspace", "."),
            model=getattr(args, "model", None),
            approval=getattr(args, "approve", None),
            resume=resume,
        )

    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
