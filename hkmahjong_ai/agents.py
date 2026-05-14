from __future__ import annotations

import random
from typing import Any

from .hand_eval import hand_potential
from .model import LinearDiscardModel, discard_feature_keys


class RandomAgent:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose_discard(self, state: dict[str, Any], legal_discards: list[int]) -> int:
        return self.rng.choice(legal_discards)

    def choose_claim(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        return self.rng.choice(options) if options and self.rng.random() < 0.25 else None


class HeuristicAgent:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose_discard(self, state: dict[str, Any], legal_discards: list[int]) -> int:
        best: list[tuple[int, float, int]] = []
        for tile in legal_discards:
            counts = list(state["hand_counts"])
            counts[tile] -= 1
            score = hand_potential(counts, state.get("open_melds", 0))
            best.append((score, self.rng.random(), tile))
        best.sort(reverse=True)
        return best[0][2]

    def choose_claim(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        counts = list(state["hand_counts"])
        base = hand_potential(counts, state.get("open_melds", 0))
        best_option: tuple[int, float, dict[str, Any]] | None = None
        for option in options:
            next_counts = list(counts)
            for tile in option["consume"]:
                next_counts[tile] -= 1
            score = hand_potential(next_counts, state.get("open_melds", 0) + 1)
            item = (score, self.rng.random(), option)
            if best_option is None or item > best_option:
                best_option = item
        if best_option and best_option[0] >= base + 2:
            return best_option[2]
        return None


class ModelAgent:
    def __init__(
        self,
        model: LinearDiscardModel,
        rng: random.Random | None = None,
        epsilon: float = 0.0,
        record: bool = False,
    ):
        self.model = model
        self.rng = rng or random.Random()
        self.epsilon = epsilon
        self.record = record
        self.decision_features: list[list[str]] = []
        self.claim_helper = HeuristicAgent(self.rng)

    def choose_discard(self, state: dict[str, Any], legal_discards: list[int]) -> int:
        discard = self.model.choose(state, legal_discards, self.rng, self.epsilon)
        if self.record:
            self.decision_features.append(discard_feature_keys(state, discard))
        return discard

    def choose_claim(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        return self.claim_helper.choose_claim(state, options)

    def reset_records(self) -> None:
        self.decision_features.clear()
