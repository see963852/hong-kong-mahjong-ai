from __future__ import annotations

import random
from typing import Any

from .hand_eval import hand_potential


class RandomAgent:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose_discard(self, state: dict[str, Any], legal_discards: list[int]) -> int:
        return self.rng.choice(legal_discards)

    def choose_claim(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not options:
            return None
        strongest = {"kong": 0.9, "pong": 0.75}
        options = sorted(options, key=lambda opt: strongest.get(opt["kind"], 0), reverse=True)
        probability = strongest.get(options[0]["kind"], 0.5)
        return options[0] if self.rng.random() < probability else None

    def choose_kong(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        return self.rng.choice(options) if options and self.rng.random() < 0.85 else None


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
            if option["kind"] == "kong":
                score += 6
            elif option["kind"] == "pong":
                score += 4
            item = (score, self.rng.random(), option)
            if best_option is None or item > best_option:
                best_option = item
        if best_option is None:
            return None
        kind = best_option[2]["kind"]
        threshold = {"kong": base - 2, "pong": base - 2}.get(kind, base)
        if best_option[0] >= threshold:
            return best_option[2]
        return None

    def choose_kong(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not options:
            return None
        added = [option for option in options if option.get("kong_type") == "added"]
        return (added or options)[0]
