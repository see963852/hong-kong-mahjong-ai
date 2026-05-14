from __future__ import annotations

import argparse

from hkmahjong_ai.ui.app import run_gui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hong Kong Mahjong RL desktop app")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="Launch desktop GUI")
    gui.set_defaults(func=lambda args: run_gui())
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        run_gui()
        return
    args.func(args)


if __name__ == "__main__":
    main()
