"""Replay buffer for mahjong Actor-Critic training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import torch

from .rl_encoder import ACTION_SIZE, STATE_DIM


@dataclass(slots=True)
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    action_mask: torch.Tensor
    next_action_mask: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity random-sampling replay buffer."""

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self._items: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        action_mask: torch.Tensor,
        next_action_mask: torch.Tensor,
    ) -> None:
        self._items.append(
            Transition(
                state=state.detach().cpu(),
                action=int(action),
                reward=float(reward),
                next_state=next_state.detach().cpu(),
                done=bool(done),
                action_mask=action_mask.detach().cpu(),
                next_action_mask=next_action_mask.detach().cpu(),
            )
        )

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        if not self._items:
            raise ValueError("cannot sample from an empty replay buffer")
        batch = random.sample(list(self._items), min(batch_size, len(self._items)))
        return {
            "states": torch.stack([item.state for item in batch]).to(device).view(-1, STATE_DIM),
            "actions": torch.tensor([item.action for item in batch], dtype=torch.long, device=device),
            "rewards": torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device),
            "next_states": torch.stack([item.next_state for item in batch]).to(device).view(-1, STATE_DIM),
            "dones": torch.tensor([item.done for item in batch], dtype=torch.float32, device=device),
            "action_masks": torch.stack([item.action_mask for item in batch]).to(device).view(-1, ACTION_SIZE),
            "next_action_masks": torch.stack([item.next_action_mask for item in batch]).to(device).view(-1, ACTION_SIZE),
        }
