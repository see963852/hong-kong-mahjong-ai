"""Runtime agent wrapper for Actor-Critic mahjong models."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch

from .hand_eval import hand_potential
from .rl_encoder import (
    ACTION_KONG,
    ACTION_PASS,
    CLAIM_KIND_TO_ACTION,
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
        claim_exploration: float = 0.15,
        discard_exploration: float = 0.0,
    ):
        self.model = model
        self.device = torch.device(device)
        self.rng = rng or random.Random()
        self.deterministic = deterministic
        self.record = record
        self.claim_exploration = claim_exploration
        self.discard_exploration = discard_exploration
        self.decisions: list[RecordedDecision] = []

    def choose_discard(self, state: dict[str, Any], legal_discards: list[int]) -> int:
        if not legal_discards:
            raise ValueError("RLAgent cannot choose a discard without legal discard options")
        state_tensor = encode_state(state, self.device)
        mask = discard_action_mask(legal_discards, self.device)
        if not self.deterministic and self.discard_exploration > 0 and self.rng.random() < self.discard_exploration:
            action = self._best_discard_action(state, legal_discards)
            self._record(state, state_tensor, action, mask)
            return action
        selection = select_action(self.model, state_tensor, mask, self.rng, self.deterministic)
        if selection.action not in legal_discards:
            raise RuntimeError(
                f"model selected illegal discard action {selection.action}; legal actions are {legal_discards}"
            )
        action = selection.action
        self._record(state, state_tensor, action, mask)
        return action

    def choose_claim(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        state_tensor = encode_state(state, self.device)
        mask = claim_action_mask(options, self.device)
        if not self.deterministic and options and self.rng.random() < self.claim_exploration:
            option = self.rng.choice(options)
            action = CLAIM_KIND_TO_ACTION[option["kind"]]
            self._record(state, state_tensor, action, mask)
            return option
        selection = select_action(self.model, state_tensor, mask, self.rng, self.deterministic)
        action = selection.action
        self._record(state, state_tensor, action, mask)
        if action == ACTION_PASS:
            return None
        return option_for_claim_action(action, options)

    def choose_kong(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not options:
            return None
        state_tensor = encode_state(state, self.device)
        mask = claim_action_mask(options, self.device)
        if not self.deterministic and self.rng.random() < self.claim_exploration:
            option = self.rng.choice(options)
            self._record(state, state_tensor, ACTION_KONG, mask)
            return option
        selection = select_action(self.model, state_tensor, mask, self.rng, self.deterministic)
        action = selection.action
        self._record(state, state_tensor, action, mask)
        if action == ACTION_PASS:
            return None
        if action == ACTION_KONG:
            return self._best_kong_option(state, options)
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

    def _best_kong_option(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any]:
        counts = list(state["hand_counts"])
        open_melds = int(state.get("open_melds", 0))
        best: tuple[int, float, dict[str, Any]] | None = None
        for option in options:
            next_counts = list(counts)
            for tile in option.get("consume", []):
                if 0 <= tile < len(next_counts):
                    next_counts[tile] -= 1
            score = hand_potential(next_counts, open_melds + 1)
            if option.get("kong_type") == "added":
                score += 1
            item = (score, self.rng.random(), option)
            if best is None or item > best:
                best = item
        return best[2] if best is not None else options[0]

    def _best_discard_action(self, state: dict[str, Any], legal_discards: list[int]) -> int:
        best: tuple[int, float, int] | None = None
        for tile in legal_discards:
            counts = list(state["hand_counts"])
            if 0 <= tile < len(counts):
                counts[tile] -= 1
            score = hand_potential(counts, state.get("open_melds", 0))
            item = (score, self.rng.random(), tile)
            if best is None or item > best:
                best = item
        return best[2] if best is not None else legal_discards[0]
