"""Runtime agent wrapper for Actor-Critic mahjong models."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

from .hand_eval import hand_potential
from .rl_encoder import (
    ACTION_PASS,
    claim_action_mask,
    discard_action_mask,
    encode_state,
    option_for_claim_action,
)
from .rl_model import ActorCriticNet, select_action


@dataclass(slots=True)
class RecordedDecision:
    player_id: int
    state: torch.Tensor
    action: int
    action_mask: torch.Tensor
    potential_before: float


class RLAgent:
    """Agent that uses an Actor-Critic model for discard and claim decisions."""

    def __init__(
        self,
        model: ActorCriticNet,
        device: torch.device | str = "cpu",
        rng: random.Random | None = None,
        deterministic: bool = False,
        record: bool = False,
    ):
        self.model = model
        self.device = torch.device(device)
        self.rng = rng or random.Random()
        self.deterministic = deterministic
        self.record = record
        self.decisions: list[RecordedDecision] = []

    def choose_discard(self, state: dict[str, Any], legal_discards: list[int]) -> int:
        state_tensor = encode_state(state, self.device)
        mask = discard_action_mask(legal_discards, self.device)
        selection = select_action(self.model, state_tensor, mask, self.rng, self.deterministic)
        action = selection.action if selection.action in legal_discards else legal_discards[0]
        self._record(state, state_tensor, action, mask)
        return action

    def choose_claim(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        state_tensor = encode_state(state, self.device)
        mask = claim_action_mask(options, self.device)
        selection = select_action(self.model, state_tensor, mask, self.rng, self.deterministic)
        action = selection.action
        self._record(state, state_tensor, action, mask)
        if action == ACTION_PASS:
            return None
        return option_for_claim_action(action, options)

    def pop_decisions(self) -> list[RecordedDecision]:
        decisions = self.decisions
        self.decisions = []
        return decisions

    def _record(self, state: dict[str, Any], state_tensor: torch.Tensor, action: int, mask: torch.Tensor) -> None:
        if not self.record:
            return
        potential = float(hand_potential(list(state["hand_counts"]), state.get("open_melds", 0)))
        self.decisions.append(
            RecordedDecision(
                player_id=int(state.get("player", state.get("current_player", 0))),
                state=state_tensor.detach().cpu(),
                action=int(action),
                action_mask=mask.detach().cpu(),
                potential_before=potential,
            )
        )
