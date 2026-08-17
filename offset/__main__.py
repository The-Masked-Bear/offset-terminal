"""Command line entry point."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="offset", description="terminal coding agent")
    sub = parser.add_subparsers(dest="cmd")

    demo = sub.add_parser("demo", help="render the design system")
    demo.add_argument("--once", action="store_true", help="print one frame and exit")
    demo.add_argument("--time", type=float, default=2.4, help="timestamp of the frame to print")
    demo.add_argument("--fps", type=float, default=24.0)
    demo.add_argument("--width", type=int, default=None)
    demo.add_argument("--height", type=int, default=None)

    args = parser.parse_args(argv)
    if args.cmd in (None, "demo"):
        from offset.ui import demo as demo_mod

        if getattr(args, "once", False):
            print(demo_mod.frame(args.time, w=args.width, h=args.height))
            return 0
        return demo_mod.run(fps=args.fps)
    parser.error(f"unknown command {args.cmd!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
