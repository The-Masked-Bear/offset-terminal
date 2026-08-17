"""Command line entry point."""

from __future__ import annotations

import argparse
import sys


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

    demo = sub.add_parser("demo", help="render the design system")
    demo.add_argument("--once", action="store_true", help="print one frame and exit")
    demo.add_argument("--time", type=float, default=2.4, help="timestamp of the frame to print")
    demo.add_argument("--fps", type=float, default=24.0)
    demo.add_argument("--width", type=int, default=None)
    demo.add_argument("--height", type=int, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "demo":
        from offset.ui import demo as demo_mod

        if args.once:
            print(demo_mod.frame(args.time, w=args.width, h=args.height))
            return 0
        return demo_mod.run(fps=args.fps)

    if args.cmd in (None, "chat"):
        from offset.shell.app import main as chat_main

        return chat_main(
            workspace=getattr(args, "workspace", "."),
            model=getattr(args, "model", None),
            approval=getattr(args, "approve", None),
        )

    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
