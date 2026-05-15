from __future__ import annotations

import random

import pytest
import torch

from hkmahjong_ai.env import HKMahjongEnv
from hkmahjong_ai.rl_agent import RLAgent
from hkmahjong_ai.rl_encoder import ACTION_KONG, discard_action_mask, encode_state
from hkmahjong_ai.rl_model import ActionSelection, ActorCriticNet, select_action


def test_choose_discard_rejects_illegal_model_action(monkeypatch: pytest.MonkeyPatch) -> None:
    env = HKMahjongEnv(seed=4)
    state = env.reset()
    model = ActorCriticNet()
    agent = RLAgent(model, record=True)

    def illegal_select(*args: object, **kwargs: object) -> ActionSelection:
        return ActionSelection(action=33, log_prob=0.0, entropy=0.0, value=0.0)

    monkeypatch.setattr("hkmahjong_ai.rl_agent.select_action", illegal_select)

    with pytest.raises(RuntimeError, match="illegal discard action"):
        agent.choose_discard(state, [0])

    assert agent.pop_decisions() == []


def test_select_action_uses_supplied_rng_for_sampling() -> None:
    model = ActorCriticNet()
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    state = torch.zeros(350)
    mask = discard_action_mask([0, 1, 2])
    rng_a = random.Random(123)
    rng_b = random.Random(123)

    seq_a = [select_action(model, state, mask, rng_a).action for _ in range(8)]
    seq_b = [select_action(model, state, mask, rng_b).action for _ in range(8)]

    assert seq_a == seq_b
    assert set(seq_a).issubset({0, 1, 2})


def test_select_action_does_not_change_model_mode() -> None:
    model = ActorCriticNet()
    model.train()

    select_action(model, torch.zeros(350), discard_action_mask([0]), random.Random(1))

    assert model.training


def test_choose_discard_rejects_empty_legal_actions() -> None:
    env = HKMahjongEnv(seed=5)
    state = env.reset()
    agent = RLAgent(ActorCriticNet())

    with pytest.raises(ValueError, match="without legal discard"):
        agent.choose_discard(state, [])


def test_choose_discard_can_use_heuristic_exploration(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ActorCriticNet()
    agent = RLAgent(model, rng=random.Random(1), discard_exploration=1.0)
    hand_counts = [0] * 34
    for tile in (0, 1, 2, 31):
        hand_counts[tile] += 1
    state = {"hand_counts": hand_counts, "open_melds": 0}

    def fail_select(*args: object, **kwargs: object) -> ActionSelection:
        raise AssertionError("model sampling should not run during forced discard exploration")

    monkeypatch.setattr("hkmahjong_ai.rl_agent.select_action", fail_select)

    assert agent.choose_discard(state, [0, 31]) == 31


def test_choose_kong_scores_multiple_kong_options(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ActorCriticNet()
    agent = RLAgent(model, rng=random.Random(1), deterministic=True)
    hand_counts = [0] * 34
    hand_counts[7] = 4
    hand_counts[8] = 1
    state = {"hand_counts": hand_counts, "open_melds": 1}
    options = [
        {"kind": "kong", "kong_type": "concealed", "consume": [7, 7, 7, 7], "tile": 7},
        {"kind": "kong", "kong_type": "added", "consume": [8], "tile": 8, "meld_index": 0},
    ]

    def choose_kong_action(*args: object, **kwargs: object) -> ActionSelection:
        return ActionSelection(action=ACTION_KONG, log_prob=0.0, entropy=0.0, value=0.0)

    monkeypatch.setattr("hkmahjong_ai.rl_agent.select_action", choose_kong_action)

    assert agent.choose_kong(state, options) == options[1]


def test_encode_state_dimension_matches_exported_constant() -> None:
    from hkmahjong_ai.rl_encoder import STATE_DIM

    env = HKMahjongEnv(seed=6)
    state = env.reset()
    encoded = encode_state(state)

    assert encoded.shape == (STATE_DIM,)
    assert STATE_DIM == 350
