"""Lightweight opponent-pool snapshots for self-play stabilization."""

from __future__ import annotations

import random

import torch

from .rl_model import ActorCriticNet, clone_model


class OpponentPool:
    """Stores older model snapshots that can be sampled as fixed opponents."""

    def __init__(self, max_size: int = 12, device: torch.device | str = "cpu"):
        self.max_size = max_size
        self.device = torch.device(device)
        self._snapshots: list[ActorCriticNet] = []

    def __len__(self) -> int:
        return len(self._snapshots)

    def add(self, model: ActorCriticNet) -> None:
        self._snapshots.append(clone_model(model, self.device))
        if len(self._snapshots) > self.max_size:
            self._snapshots.pop(0)

    def sample(self, rng: random.Random) -> ActorCriticNet | None:
        if not self._snapshots:
            return None
        return clone_model(rng.choice(self._snapshots), self.device)
