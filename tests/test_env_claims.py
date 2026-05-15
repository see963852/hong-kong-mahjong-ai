from __future__ import annotations

import pytest

from hkmahjong_ai.env import HKMahjongEnv, Meld
from hkmahjong_ai.rl_encoder import ACTION_KONG, ACTION_PASS, ACTION_PONG


# These are whitebox regression tests. They intentionally set player hands,
# melds, wall, and last_discard directly to isolate rule-state transitions.
class RecordingAgent:
    def __init__(self) -> None:
        self.claim_options: list[dict] = []

    def choose_claim(self, state: dict, options: list[dict]) -> dict | None:
        self.claim_options.extend(options)
        return None


def test_paomazai_does_not_offer_chow_claims() -> None:
    env = HKMahjongEnv(seed=1)
    env.reset()
    env.players[1].hand = [0, 2]
    agents = [RecordingAgent() for _ in range(4)]

    result, claimed = env._resolve_claims(discarder=0, tile=1, agents=agents)

    assert result is None
    assert not claimed
    assert agents[1].claim_options == []


def test_legal_action_mask_returns_claim_mask_for_waiting_claim() -> None:
    env = HKMahjongEnv(seed=11)
    env.reset()
    env.players[1].hand = [3, 3, 3]
    env.players[0].discards = [3]
    env.last_discard = (0, 3)
    env.current_player = 0

    mask = env.legal_action_mask(1)

    assert mask[ACTION_PASS]
    assert mask[ACTION_PONG]
    assert mask[ACTION_KONG]
    assert not mask[3]


def test_legal_action_mask_returns_discard_mask_for_current_turn() -> None:
    env = HKMahjongEnv(seed=12)
    env.reset()
    env.current_player = 0
    env.players[0].hand = [1, 2, 2]

    mask = env.legal_action_mask(0)

    assert mask[1]
    assert mask[2]
    assert not mask[ACTION_PASS]


def test_legal_action_mask_ignores_stale_last_discard_for_current_player() -> None:
    env = HKMahjongEnv(seed=13)
    env.reset()
    env.current_player = 1
    env.players[1].hand = [3, 3, 3, 5]
    env.players[0].discards = [3]
    env.last_discard = (0, 3)

    mask = env.legal_action_mask(1)

    assert mask[3]
    assert mask[5]
    assert not mask[ACTION_PASS]


def test_unclaimed_discard_clears_pending_claim_window() -> None:
    env = HKMahjongEnv(seed=14)
    env.reset()
    env.current_player = 0
    env.players[0].hand = [5]
    env.players[1].hand = [1, 2, 3]
    env.players[2].hand = [5, 5, 8]
    env.wall = [7]

    env.discard(5, agents=None)
    mask = env.legal_action_mask(2)

    assert env.current_player == 1
    assert env.last_discard is None
    assert not mask[ACTION_PASS]
    assert not mask[ACTION_PONG]


def test_claim_clears_pending_discard_after_removing_claimed_tile() -> None:
    env = HKMahjongEnv(seed=15)
    env.reset()
    env.players[0].discards = [5, 5]
    env.last_discard = (0, 5)
    env.players[1].hand = [5, 5, 1]
    env.players[2].hand = [5, 5, 8]

    env._apply_claim(1, 0, {"kind": "pong", "tiles": [5, 5, 5], "consume": [5, 5], "discard_tile": 5})
    mask = env.legal_action_mask(2)

    assert env.players[0].discards == [5]
    assert env.last_discard is None
    assert not mask[ACTION_PASS]
    assert not mask[ACTION_PONG]


def test_declare_self_win_rejects_non_winning_hand() -> None:
    env = HKMahjongEnv(seed=16)
    env.reset()
    env.current_player = 0
    env.players[0].hand = [0, 1, 2]

    with pytest.raises(ValueError, match="cannot self-win"):
        env.declare_self_win(0)


def test_discard_win_supports_multiple_winners_and_rich_dealer() -> None:
    env = HKMahjongEnv(seed=2)
    env.reset()
    winning_wait = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    env.players[1].hand = list(winning_wait)
    env.players[2].hand = list(winning_wait)

    result, claimed = env._resolve_claims(discarder=0, tile=4, agents=None)

    assert result is not None
    assert result.winner == 1
    assert result.winners == [1, 2]
    assert result.losers == [0]
    assert result.kind == "multi_discard_win"
    assert result.score_delta == [-4, 2, 2, 0]
    assert result.next_dealer == 0
    assert env.dealer == 0
    assert not claimed


def test_kong_replacement_immediately_checks_self_draw() -> None:
    env = HKMahjongEnv(seed=3)
    env.reset()
    env.current_player = 0
    env.players[0].hand = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    env.wall = [4]

    result = env._draw_replacement(0)

    assert result is not None
    assert result.kind == "self_draw"
    assert env.done


def test_melded_kong_scores_from_discarder_and_replacement_draws() -> None:
    env = HKMahjongEnv(seed=4)
    env.reset()
    env.players[1].hand = [5, 5, 5] + env.players[1].hand
    env.players[0].discards = [5]
    env.last_discard = (0, 5)
    env.wall = [9]

    result = env._apply_claim(
        1,
        0,
        {"kind": "kong", "tiles": [5, 5, 5, 5], "consume": [5, 5, 5], "discard_tile": 5},
    )

    assert result is None
    assert env.players[0].discards == []
    assert env.players[1].melds[-1].kong_type == "melded"
    assert env.hand_score_delta == [-3, 3, 0, 0]
    assert 9 in env.players[1].hand


def test_concealed_kong_scores_all_others_and_replacement_draws() -> None:
    env = HKMahjongEnv(seed=5)
    env.reset()
    env.players[0].hand = [7, 7, 7, 7]
    env.wall = [9]

    result = env.declare_concealed_kong(0, 7)

    assert result is None
    assert env.players[0].melds[-1].kong_type == "concealed"
    assert env.players[0].hand == [9]
    assert env.hand_score_delta == [6, -2, -2, -2]


def test_added_kong_scores_all_others_when_not_robbed() -> None:
    env = HKMahjongEnv(seed=6)
    env.reset()
    env.players[0].hand = [8]
    env.players[0].melds = [Meld("pong", [8, 8, 8], 1)]
    env.wall = [9]

    result = env.declare_added_kong(0, 8, 0, agents=None)

    assert result is None
    assert env.players[0].melds[0].kind == "kong"
    assert env.players[0].melds[0].kong_type == "added"
    assert env.players[0].hand == [9]
    assert env.hand_score_delta == [3, -1, -1, -1]


def test_added_kong_can_be_robbed_by_multiple_winners_without_mutating_kong() -> None:
    env = HKMahjongEnv(seed=7)
    env.reset()
    env.players[0].hand = [4]
    env.players[0].melds = [Meld("pong", [4, 4, 4], 3)]
    winning_wait = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    env.players[1].hand = list(winning_wait)
    env.players[2].hand = list(winning_wait)

    result = env.declare_added_kong(0, 4, 0, agents=None)

    assert result is not None
    assert result.kind == "rob_kong"
    assert result.winners == [1, 2]
    assert result.losers == [0]
    assert result.score_delta == [-12, 6, 6, 0]
    assert result.next_dealer == 0
    assert env.players[0].hand == [4]
    assert env.players[0].melds[0].kind == "pong"
