from __future__ import annotations

import random

from .agents import HeuristicAgent, ModelAgent, RandomAgent
from .env import HKMahjongEnv
from .model import LinearDiscardModel


def evaluate_model(
    model_path: str,
    episodes: int = 500,
    opponent: str = "heuristic",
    seed: int | None = None,
) -> str:
    rng = random.Random(seed)
    model = LinearDiscardModel.load(model_path)
    stats = {"wins": 0, "losses": 0, "discard_losses": 0, "draws": 0}

    for episode in range(episodes):
        model_seat = episode % 4
        env = HKMahjongEnv(seed=rng.randrange(1_000_000_000))
        env.reset()
        agents = []
        for seat in range(4):
            seat_rng = random.Random(rng.randrange(1_000_000_000))
            if seat == model_seat:
                agents.append(ModelAgent(model, seat_rng, epsilon=0.0))
            elif opponent == "random":
                agents.append(RandomAgent(seat_rng))
            else:
                agents.append(HeuristicAgent(seat_rng))

        while not env.done:
            env.play_ai_turn(agents)
        result = env.result
        assert result is not None
        if result.winner is None:
            stats["draws"] += 1
        elif result.winner == model_seat:
            stats["wins"] += 1
        else:
            stats["losses"] += 1
            if result.loser == model_seat:
                stats["discard_losses"] += 1

    win_rate = stats["wins"] / max(1, episodes)
    return (
        f"model={model_path}, episodes={episodes}, opponent={opponent}, "
        f"wins={stats['wins']}, losses={stats['losses']}, draws={stats['draws']}, "
        f"discard_losses={stats['discard_losses']}, win_rate={win_rate:.3f}"
    )
