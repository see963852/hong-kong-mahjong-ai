from __future__ import annotations

import random

from .agents import ModelAgent
from .env import HKMahjongEnv, GameResult
from .hand_eval import hand_potential
from .model import LinearDiscardModel


def train_model(
    episodes: int,
    out_path: str,
    model_path: str | None = None,
    seed: int | None = None,
    learning_rate: float = 0.02,
    epsilon: float = 0.12,
) -> str:
    rng = random.Random(seed)
    model = LinearDiscardModel.load(model_path) if model_path else LinearDiscardModel()
    wins = [0, 0, 0, 0]
    draws = 0

    for episode in range(episodes):
        env = HKMahjongEnv(seed=rng.randrange(1_000_000_000))
        env.reset()
        agents = [
            ModelAgent(model, random.Random(rng.randrange(1_000_000_000)), epsilon=epsilon, record=True)
            for _ in range(4)
        ]
        result = _play_episode(env, agents)
        rewards = _rewards(result, env)
        if result.winner is None:
            draws += 1
        else:
            wins[result.winner] += 1

        for pid, agent in enumerate(agents):
            per_step_reward = rewards[pid]
            for features in agent.decision_features:
                model.update(features, per_step_reward, learning_rate)
        model.games_trained += 1

    model.save(out_path)
    return (
        f"trained={episodes}, total_games={model.games_trained}, "
        f"wins={wins}, draws={draws}, saved={out_path}"
    )


def _play_episode(env: HKMahjongEnv, agents: list[ModelAgent]) -> GameResult:
    while not env.done:
        env.play_ai_turn(agents)
    assert env.result is not None
    return env.result


def _rewards(result: GameResult, env: HKMahjongEnv) -> list[float]:
    potentials = [
        hand_potential(player.counts(), len(player.melds))
        for player in env.players
    ]
    low = min(potentials)
    high = max(potentials)
    shaped = [0.0, 0.0, 0.0, 0.0]
    if high > low:
        shaped = [((score - low) / (high - low) - 0.5) * 0.12 for score in potentials]

    if result.winner is None:
        return shaped
    rewards = [-0.15 + shaped[i] for i in range(4)]
    rewards[result.winner] = 1.0
    if result.loser is not None and result.loser != result.winner:
        rewards[result.loser] = -0.7
    return rewards
