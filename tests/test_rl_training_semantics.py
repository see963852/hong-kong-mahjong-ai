from __future__ import annotations

import torch
import pytest

from hkmahjong_ai.env import GameResult
from hkmahjong_ai.replay_buffer import ReplayBuffer
from hkmahjong_ai.rl_encoder import ACTION_SIZE, STATE_DIM
from hkmahjong_ai.rl_train import EpisodeTransition, _commit_episode_transitions, _terminal_rewards


def transition(action: int, shaped_reward: float = 0.0) -> EpisodeTransition:
    return EpisodeTransition(
        state=torch.zeros(STATE_DIM),
        action=action,
        next_state=torch.ones(STATE_DIM),
        action_mask=torch.ones(ACTION_SIZE, dtype=torch.bool),
        next_action_mask=torch.ones(ACTION_SIZE, dtype=torch.bool),
        shaped_reward=shaped_reward,
    )


def test_episode_commit_writes_discounted_terminal_returns_as_terminal_samples() -> None:
    buffers = [ReplayBuffer(10) for _ in range(4)]
    episode_transitions = [
        [transition(0), transition(1, shaped_reward=0.2)],
        [transition(2)],
        [],
        [],
    ]

    _commit_episode_transitions(
        episode_transitions,
        buffers,
        terminal_rewards=[1.0, -0.7, -0.15, -0.15],
        gamma=0.5,
    )

    batch0 = buffers[0].sample(10, "cpu")
    assert sorted(batch0["rewards"].tolist()) == pytest.approx([0.5, 1.2])
    assert batch0["dones"].tolist() == [1.0, 1.0]

    batch1 = buffers[1].sample(10, "cpu")
    assert batch1["rewards"].tolist() == pytest.approx([-0.7])
    assert batch1["dones"].tolist() == [1.0]


def test_terminal_rewards_support_multiple_winners() -> None:
    result = GameResult(
        winner=1,
        loser=0,
        kind="multi_discard_win",
        turns=12,
        wall_remaining=40,
        message="multi win",
        winners=[1, 2],
        losers=[0],
    )

    assert _terminal_rewards(result) == pytest.approx([-1.4, 1.0, 1.0, -0.15])
