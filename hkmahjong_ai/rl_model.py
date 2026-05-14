"""PyTorch Actor-Critic model for Hong Kong mahjong self-play."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .rl_encoder import ACTION_SIZE, STATE_DIM


class ActorCriticNet(nn.Module):
    """Shared encoder with separate policy and value heads."""

    def __init__(self, state_dim: int = STATE_DIM, action_size: int = ACTION_SIZE, hidden_size: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if states.dim() == 1:
            states = states.unsqueeze(0)
        encoded = self.encoder(states)
        return self.policy_head(encoded), self.value_head(encoded).squeeze(-1)


@dataclass(slots=True)
class ActionSelection:
    action: int
    log_prob: float
    entropy: float
    value: float


def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    if action_mask.dim() == 1 and logits.dim() == 2:
        action_mask = action_mask.unsqueeze(0)
    return logits.masked_fill(~action_mask.bool(), -1.0e9)


def select_action(
    model: ActorCriticNet,
    state: torch.Tensor,
    action_mask: torch.Tensor,
    rng: random.Random | None = None,
    deterministic: bool = False,
) -> ActionSelection:
    """Select one legal action from the actor distribution."""
    del rng
    model.eval()
    with torch.no_grad():
        logits, value = model(state)
        legal_logits = masked_logits(logits, action_mask)
        dist = Categorical(logits=legal_logits)
        if deterministic:
            action_tensor = torch.argmax(legal_logits, dim=-1)
        else:
            action_tensor = dist.sample()
        log_prob = dist.log_prob(action_tensor)
        entropy = dist.entropy()
    return ActionSelection(
        action=int(action_tensor.item()),
        log_prob=float(log_prob.item()),
        entropy=float(entropy.item()),
        value=float(value.squeeze(0).item()),
    )


def create_optimizer(model: ActorCriticNet, lr: float = 3e-4) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=lr)


def save_rl_checkpoint(
    path: str,
    models: list[ActorCriticNet],
    optimizers: list[torch.optim.Optimizer] | None = None,
    games_trained: int = 0,
    version: int = 1,
    extra: dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    checkpoint: dict[str, Any] = {
        "version": version,
        "games_trained": games_trained,
        "state_dim": STATE_DIM,
        "action_size": ACTION_SIZE,
        "players": [model.state_dict() for model in models],
        "extra": extra or {},
    }
    if models:
        checkpoint["state_dict"] = models[0].state_dict()
    if optimizers is not None:
        checkpoint["optimizers"] = [optimizer.state_dict() for optimizer in optimizers]
    torch.save(checkpoint, path)


def load_rl_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    player_index: int = 0,
) -> tuple[ActorCriticNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    checkpoint_state_dim = int(checkpoint.get("state_dim", STATE_DIM))
    if checkpoint_state_dim != STATE_DIM:
        raise ValueError(
            f"checkpoint state_dim={checkpoint_state_dim} is incompatible with current STATE_DIM={STATE_DIM}; retrain the model"
        )
    model = ActorCriticNet().to(device)
    players = checkpoint.get("players")
    if players:
        state_dict = players[player_index % len(players)]
    else:
        state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def clone_model(model: ActorCriticNet, device: torch.device | str = "cpu") -> ActorCriticNet:
    cloned = ActorCriticNet().to(device)
    cloned.load_state_dict({key: value.detach().clone().to(device) for key, value in model.state_dict().items()})
    cloned.eval()
    return cloned
