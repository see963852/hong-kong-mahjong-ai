from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any

from .tiles import is_suited, rank, suit_index, tile_name


def discard_feature_keys(state: dict[str, Any], discard: int) -> list[str]:
    counts = list(state["hand_counts"])
    before = counts[discard]
    counts[discard] -= 1
    keys = [
        "bias",
        f"tile:{discard}",
        f"name:{tile_name(discard)}",
        f"count_before:{before}",
        f"open_melds:{state.get('open_melds', 0)}",
        f"wall_bucket:{state.get('wall_remaining', 0) // 20}",
    ]

    if is_suited(discard):
        r = rank(discard)
        s = suit_index(discard)
        left = counts[discard - 1] if r > 1 else 0
        right = counts[discard + 1] if r < 9 else 0
        gap_left = counts[discard - 2] if r > 2 else 0
        gap_right = counts[discard + 2] if r < 8 else 0
        keys.extend(
            [
                f"suit:{s}",
                f"rank:{r}",
                f"terminal:{r in (1, 9)}",
                f"neighbors:{min(2, left + right)}",
                f"gaps:{min(2, gap_left + gap_right)}",
                f"isolated:{left + right + gap_left + gap_right == 0}",
            ]
        )
    else:
        keys.extend(["honor", f"honor_pair:{before >= 2}"])

    keys.append(f"keeps_pair:{counts[discard] >= 2}")
    keys.append(f"breaks_pair:{before == 2}")
    keys.append(f"breaks_triplet:{before >= 3}")
    return keys


@dataclass
class LinearDiscardModel:
    weights: dict[str, float] = field(default_factory=dict)
    version: int = 1
    games_trained: int = 0

    def score(self, state: dict[str, Any], discard: int) -> float:
        return sum(self.weights.get(key, 0.0) for key in discard_feature_keys(state, discard))

    def choose(
        self,
        state: dict[str, Any],
        legal_discards: list[int],
        rng: random.Random,
        epsilon: float = 0.0,
    ) -> int:
        if not legal_discards:
            raise ValueError("no legal discards")
        if rng.random() < epsilon:
            return rng.choice(legal_discards)
        scored = [(self.score(state, tile), rng.random(), tile) for tile in legal_discards]
        scored.sort(reverse=True)
        return scored[0][2]

    def update(self, feature_keys: list[str], reward: float, lr: float) -> None:
        if reward == 0:
            return
        scale = lr * max(-1.0, min(1.0, reward)) / math.sqrt(max(1, len(feature_keys)))
        for key in feature_keys:
            self.weights[key] = self.weights.get(key, 0.0) + scale

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": self.version,
                    "games_trained": self.games_trained,
                    "weights": self.weights,
                },
                f,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

    @classmethod
    def load(cls, path: str) -> "LinearDiscardModel":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            weights={str(k): float(v) for k, v in data.get("weights", {}).items()},
            version=int(data.get("version", 1)),
            games_trained=int(data.get("games_trained", 0)),
        )
