from __future__ import annotations

import argparse

from hkmahjong_ai.compare import compare_models
from hkmahjong_ai.evaluate import evaluate_model
from hkmahjong_ai.gui import run_gui
from hkmahjong_ai.train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hong Kong Mahjong AI training platform")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="Launch desktop GUI")
    gui.set_defaults(func=lambda args: run_gui())

    train = sub.add_parser("train", help="Train a discard policy model")
    train.add_argument("--episodes", type=int, default=1000)
    train.add_argument("--model", default=None, help="Existing model JSON to continue training")
    train.add_argument("--out", default="models/model_latest.json")
    train.add_argument("--seed", type=int, default=None)
    train.set_defaults(func=lambda args: train_model(args.episodes, args.out, args.model, args.seed))

    evaluate = sub.add_parser("evaluate", help="Evaluate one model")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--episodes", type=int, default=500)
    evaluate.add_argument("--opponent", choices=["random", "heuristic"], default="heuristic")
    evaluate.add_argument("--seed", type=int, default=None)
    evaluate.set_defaults(
        func=lambda args: evaluate_model(args.model, args.episodes, args.opponent, args.seed)
    )

    compare = sub.add_parser("compare", help="Compare two model versions")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    compare.add_argument("--episodes", type=int, default=500)
    compare.add_argument("--seed", type=int, default=None)
    compare.set_defaults(func=lambda args: compare_models(args.a, args.b, args.episodes, args.seed))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        run_gui()
        return
    result = args.func(args)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
