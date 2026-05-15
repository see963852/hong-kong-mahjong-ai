from __future__ import annotations

import torch
import pytest

from hkmahjong_ai.env import GameResult
from hkmahjong_ai.replay_buffer import ReplayBuffer
from hkmahjong_ai.rl_encoder import ACTION_CHOW, ACTION_KONG, ACTION_PASS, ACTION_PONG, ACTION_SIZE, STATE_DIM
from hkmahjong_ai.rl_model import ActorCriticNet, create_optimizer
from hkmahjong_ai.rl_train import (
    DRAW_REWARD,
    EpisodeTransition,
    _clear_buffers,
    _commit_episode_transitions,
    _progress_payload,
    _scheduled_exploration,
    _terminal_rewards,
    _update_model,
)


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

    assert _terminal_rewards(result) == pytest.approx([-1.0, 1.0, 1.0, -0.15])


def test_terminal_rewards_penalize_draws() -> None:
    result = GameResult(None, None, "draw", 88, 0, "draw")

    assert _terminal_rewards(result) == pytest.approx([DRAW_REWARD, DRAW_REWARD, DRAW_REWARD, DRAW_REWARD])


def test_progress_win_rate_is_total_game_level_not_player_win_sum() -> None:
    results = [
        GameResult(1, 0, "multi_discard_win", 10, 50, "win", winners=[1, 2], losers=[0]),
        GameResult(None, None, "draw", 20, 0, "draw"),
    ]

    payload = _progress_payload(
        episode=20,
        total_episodes=100,
        games_trained=20,
        wins=[0, 1, 1, 0],
        draws=10,
        recent_results=results,
        recent_losses=[],
        device="cpu",
        out_path="models/test.pt",
        stopped=False,
    )

    assert payload["win_rate"] == pytest.approx(0.5)
    assert payload["recent_win_rate"] == pytest.approx(0.5)


def test_discard_exploration_schedule_decays_between_bounds() -> None:
    assert _scheduled_exploration(0, 101, 0.75, 0.10) == pytest.approx(0.75)
    assert _scheduled_exploration(50, 101, 0.75, 0.10) == pytest.approx(0.425)
    assert _scheduled_exploration(100, 101, 0.75, 0.10) == pytest.approx(0.10)


def test_clear_buffers_drops_old_off_policy_samples() -> None:
    buffers = [ReplayBuffer(10) for _ in range(4)]
    buffers[0].push(
        torch.zeros(STATE_DIM),
        0,
        1.0,
        torch.zeros(STATE_DIM),
        True,
        torch.ones(ACTION_SIZE, dtype=torch.bool),
        torch.ones(ACTION_SIZE, dtype=torch.bool),
    )

    _clear_buffers(buffers)

    assert [len(buffer) for buffer in buffers] == [0, 0, 0, 0]


def test_claim_exploration_loss_does_not_backprop_to_chow_action() -> None:
    model = ActorCriticNet()
    optimizer = create_optimizer(model, lr=0.0)
    buffer = ReplayBuffer(10)
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool)
    mask[ACTION_PASS] = True
    mask[ACTION_PONG] = True
    mask[ACTION_KONG] = True
    buffer.push(
        torch.zeros(STATE_DIM),
        ACTION_PONG,
        1.0,
        torch.zeros(STATE_DIM),
        True,
        mask,
        mask,
    )

    _update_model(model, optimizer, buffer, batch_size=1, gamma=0.97, value_coeff=0.5, entropy_coeff=0.05, max_grad_norm=1.0, device=torch.device("cpu"))

    assert model.policy_head.bias.grad is not None
    assert model.policy_head.bias.grad[ACTION_CHOW].item() == pytest.approx(0.0)
