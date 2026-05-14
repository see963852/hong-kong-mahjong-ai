from __future__ import annotations

import random

from .agents import HeuristicAgent, ModelAgent
from .env import HKMahjongEnv
from .model import LinearDiscardModel


def compare_models(path_a: str, path_b: str, episodes: int = 500, seed: int | None = None) -> str:
    rng = random.Random(seed)
    model_a = LinearDiscardModel.load(path_a)
    model_b = LinearDiscardModel.load(path_b)
    wins = {"A": 0, "B": 0, "baseline": 0, "draws": 0}

    for episode in range(episodes):
        env = HKMahjongEnv(seed=rng.randrange(1_000_000_000))
        env.reset()
        a_seat = episode % 4
        b_seat = (a_seat + 2) % 4
        agents = []
        for seat in range(4):
            seat_rng = random.Random(rng.randrange(1_000_000_000))
            if seat == a_seat:
                agents.append(ModelAgent(model_a, seat_rng))
            elif seat == b_seat:
                agents.append(ModelAgent(model_b, seat_rng))
            else:
                agents.append(HeuristicAgent(seat_rng))

        while not env.done:
            env.play_ai_turn(agents)
        result = env.result
        assert result is not None
        if result.winner is None:
            wins["draws"] += 1
        elif result.winner == a_seat:
            wins["A"] += 1
        elif result.winner == b_seat:
            wins["B"] += 1
        else:
            wins["baseline"] += 1

    return (
        f"A={path_a}, B={path_b}, episodes={episodes}, "
        f"A_wins={wins['A']}, B_wins={wins['B']}, "
        f"baseline_wins={wins['baseline']}, draws={wins['draws']}"
    )
