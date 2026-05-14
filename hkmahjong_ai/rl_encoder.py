"""State encoding and action masking for PyTorch mahjong agents.

The action space is intentionally fixed and small:

- 0..33: discard that tile id
- 34: pass a claim
- 35: chow
- 36: pong
- 37: kong

Only legal actions are exposed through masks. The environment still decides
which concrete chow sequence is available when the actor selects chow.
"""

from __future__ import annotations

from typing import Any

import torch

from .tiles import TILE_COUNT

DISCARD_ACTIONS = TILE_COUNT
ACTION_PASS = 34
ACTION_CHOW = 35
ACTION_PONG = 36
ACTION_KONG = 37
ACTION_SIZE = 38
STATE_DIM = 316
MAX_WALL_TILES = 83
MAX_TURNS = 160

CLAIM_KIND_TO_ACTION = {
    "chow": ACTION_CHOW,
    "pong": ACTION_PONG,
    "kong": ACTION_KONG,
}
ACTION_TO_CLAIM_KIND = {value: key for key, value in CLAIM_KIND_TO_ACTION.items()}


def encode_state(state: dict[str, Any], device: torch.device | str | None = None) -> torch.Tensor:
    """Encode a player observation into a fixed-size float tensor."""
    features: list[float] = []

    features.extend(_normalized_counts(state.get("hand_counts", []), scale=4.0))

    for discards in state.get("discards", [[], [], [], []]):
        counts = [0] * TILE_COUNT
        for tile in discards:
            counts[tile] += 1
        features.extend(_normalized_counts(counts, scale=4.0))

    meld_counts = list(state.get("meld_counts", []))[:4]
    meld_counts += [0] * (4 - len(meld_counts))
    features.extend(min(1.0, count / 4.0) for count in meld_counts)

    all_meld_counts = state.get("all_meld_tile_counts", [])
    for player_index in range(4):
        counts = all_meld_counts[player_index] if player_index < len(all_meld_counts) else []
        features.extend(_normalized_counts(counts, scale=4.0))

    wall_remaining = float(state.get("wall_remaining", 0))
    features.append(max(0.0, min(1.0, wall_remaining / MAX_WALL_TILES)))

    current_player = int(state.get("player", state.get("current_player", 0)))
    features.extend(1.0 if current_player == seat else 0.0 for seat in range(4))

    turns = float(state.get("turns", 0))
    features.append(max(0.0, min(1.0, turns / MAX_TURNS)))

    if len(features) != STATE_DIM:
        raise ValueError(f"state encoder produced {len(features)} features, expected {STATE_DIM}")
    return torch.tensor(features, dtype=torch.float32, device=device)


def discard_action_mask(legal_discards: list[int], device: torch.device | str | None = None) -> torch.Tensor:
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool, device=device)
    for tile in legal_discards:
        if 0 <= tile < TILE_COUNT:
            mask[tile] = True
    return mask


def claim_action_mask(options: list[dict[str, Any]], device: torch.device | str | None = None) -> torch.Tensor:
    mask = torch.zeros(ACTION_SIZE, dtype=torch.bool, device=device)
    mask[ACTION_PASS] = True
    for option in options:
        action = CLAIM_KIND_TO_ACTION.get(option.get("kind", ""))
        if action is not None:
            mask[action] = True
    return mask


def option_for_claim_action(action: int, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    kind = ACTION_TO_CLAIM_KIND.get(action)
    if kind is None:
        return None
    for option in options:
        if option.get("kind") == kind:
            return option
    return None


def _normalized_counts(counts: list[int], scale: float) -> list[float]:
    data = list(counts)[:TILE_COUNT]
    data += [0] * (TILE_COUNT - len(data))
    return [max(0.0, min(1.0, count / scale)) for count in data]
