"""Self-play Actor-Critic training, evaluation, and model comparison."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
import threading
import time
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .env import GameResult, HKMahjongEnv
from .hand_eval import hand_potential
from .opponent_pool import OpponentPool
from .replay_buffer import ReplayBuffer
from .rl_agent import RLAgent
from .rl_encoder import ACTION_KONG, ACTION_PONG, STATE_DIM, encode_state
from .rl_model import (
    ActorCriticNet,
    create_optimizer,
    load_checkpoint_data,
    load_rl_checkpoint,
    masked_logits,
    save_rl_checkpoint,
)

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True)
class EpisodeTransition:
    state: torch.Tensor
    action: int
    next_state: torch.Tensor
    action_mask: torch.Tensor
    next_action_mask: torch.Tensor
    shaped_reward: float


def train_rl(
    episodes: int,
    out_path: str,
    model_path: str | None = None,
    seed: int | None = None,
    batch_size: int = 64,
    capacity: int = 50_000,
    gamma: float = 0.97,
    lr: float = 3e-4,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.05,
    max_grad_norm: float = 1.0,
    snapshot_interval: int = 500,
    pool_probability: float = 0.15,
    shaped_reward_coeff: float = 0.01,
    updates_per_episode: int = 4,
    buffer_max_episodes: int = 200,
    device_name: str | None = None,
    progress_callback: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    update_interval: int = 50,
) -> str:
    """Train four independent Actor-Critic models through self-play.

    The function is GUI-friendly: callers can pass pause/stop events and a
    callback that receives rolling progress, win/draw statistics, and losses.
    """
    rng = random.Random(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    models, games_trained = _load_or_create_models(model_path, device)
    optimizers = [create_optimizer(model, lr) for model in models]
    buffers = [ReplayBuffer(capacity) for _ in range(4)]
    pool = OpponentPool(device=device)
    wins = [0, 0, 0, 0]
    draws = 0
    recent_results: list[GameResult] = []
    recent_losses: list[dict[str, float]] = []
    completed = 0
    stopped = False

    for episode in range(episodes):
        if _should_stop(stop_event):
            stopped = True
            break
        stopped = _wait_if_paused(pause_event, stop_event)
        if stopped:
            break

        env = HKMahjongEnv(seed=rng.randrange(1_000_000_000))
        env.reset()
        agents = _build_self_play_agents(models, pool, pool_probability, rng, device)
        episode_transitions: list[list[EpisodeTransition]] = [[], [], [], []]

        while not env.done:
            if _should_stop(stop_event):
                stopped = True
                break
            env.play_ai_turn(agents)
            _collect_agent_decisions(env, agents, episode_transitions, shaped_reward_coeff)
        if stopped:
            break

        assert env.result is not None
        terminal_rewards = _terminal_rewards(env.result)
        _commit_episode_transitions(episode_transitions, buffers, terminal_rewards, gamma)
        for _ in range(updates_per_episode):
            for seat in range(4):
                if len(buffers[seat]) >= max(8, batch_size // 2):
                    recent_losses.append(
                        _update_model(
                            models[seat],
                            optimizers[seat],
                            buffers[seat],
                            batch_size,
                            gamma,
                            value_coeff,
                            entropy_coeff,
                            max_grad_norm,
                            device,
                        )
                    )
        if len(recent_losses) > update_interval * 8:
            recent_losses = recent_losses[-update_interval * 8 :]

        if env.result.winner is None:
            draws += 1
        else:
            for winner in env.result.winners:
                wins[winner] += 1
        games_trained += 1
        completed = episode + 1
        recent_results.append(env.result)
        if len(recent_results) > update_interval:
            recent_results.pop(0)

        if snapshot_interval > 0 and completed % snapshot_interval == 0:
            for model in models:
                pool.add(model)
            save_rl_checkpoint(
                _snapshot_path(out_path, games_trained),
                models,
                optimizers,
                games_trained=games_trained,
                extra={"snapshot": True, "pool_size": len(pool)},
            )
            _clear_buffers(buffers)
        elif buffer_max_episodes > 0 and completed % buffer_max_episodes == 0:
            _clear_buffers(buffers)

        if progress_callback is not None and (completed % max(1, update_interval) == 0 or completed == episodes):
            progress_callback(
                _progress_payload(
                    episode=completed,
                    total_episodes=episodes,
                    games_trained=games_trained,
                    wins=wins,
                    draws=draws,
                    recent_results=recent_results,
                    recent_losses=recent_losses,
                    device=str(device),
                    out_path=out_path,
                    stopped=False,
                )
            )

    save_rl_checkpoint(
        out_path,
        models,
        optimizers,
        games_trained=games_trained,
        extra={
            "training": "self_play_actor_critic",
            "episodes": completed,
            "wins": wins,
            "draws": draws,
            "stopped": stopped,
        },
    )
    if progress_callback is not None:
        progress_callback(
            _progress_payload(
                episode=completed,
                total_episodes=episodes,
                games_trained=games_trained,
                wins=wins,
                draws=draws,
                recent_results=recent_results,
                recent_losses=recent_losses,
                device=str(device),
                out_path=out_path,
                stopped=stopped,
            )
        )
    return (
        f"rl_trained={completed}/{episodes}, total_games={games_trained}, wins={wins}, "
        f"draws={draws}, stopped={stopped}, device={device}, saved={out_path}"
    )


def evaluate_rl(model_path: str, episodes: int = 500, seed: int | None = None) -> str:
    """Evaluate the four checkpoint player models against each other."""
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, games_trained = _load_or_create_models(model_path, device)
    wins = [0, 0, 0, 0]
    draws = 0

    for _ in range(episodes):
        env = HKMahjongEnv(seed=rng.randrange(1_000_000_000))
        env.reset()
        agents = [
            RLAgent(models[seat], device=device, rng=random.Random(rng.randrange(1_000_000_000)), deterministic=True)
            for seat in range(4)
        ]
        while not env.done:
            env.play_ai_turn(agents)
        assert env.result is not None
        if env.result.winner is None:
            draws += 1
        else:
            for winner in env.result.winners:
                wins[winner] += 1

    return (
        f"model={model_path}, trained_games={games_trained}, episodes={episodes}, "
        f"wins={wins}, draws={draws}, draw_rate={draws / max(1, episodes):.3f}"
    )


def compare_rl(path_a: str, path_b: str, episodes: int = 500, seed: int | None = None) -> str:
    """Compare two RL checkpoints without heuristic opponents."""
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, _ = load_rl_checkpoint(path_a, device)
    model_b, _ = load_rl_checkpoint(path_b, device)
    wins = {"A": 0, "B": 0, "draws": 0}

    for episode in range(episodes):
        env = HKMahjongEnv(seed=rng.randrange(1_000_000_000))
        env.reset()
        a_seats = {episode % 4, (episode + 2) % 4}
        agents = []
        for seat in range(4):
            model = model_a if seat in a_seats else model_b
            agents.append(RLAgent(model, device=device, rng=random.Random(rng.randrange(1_000_000_000)), deterministic=True))
        while not env.done:
            env.play_ai_turn(agents)
        assert env.result is not None
        if env.result.winner is None:
            wins["draws"] += 1
        else:
            for winner in env.result.winners:
                if winner in a_seats:
                    wins["A"] += 1
                else:
                    wins["B"] += 1

    return (
        f"A={path_a}, B={path_b}, episodes={episodes}, "
        f"A_wins={wins['A']}, B_wins={wins['B']}, draws={wins['draws']}"
    )


def checkpoint_metadata(path: str) -> dict[str, Any]:
    """Load lightweight checkpoint metadata for GUI model management."""
    checkpoint = load_checkpoint_data(path, map_location="cpu")
    players = checkpoint.get("players") or []
    extra = checkpoint.get("extra") or {}
    return {
        "version": checkpoint.get("version", "-"),
        "games_trained": checkpoint.get("games_trained", 0),
        "state_dim": checkpoint.get("state_dim", "-"),
        "action_size": checkpoint.get("action_size", "-"),
        "players": len(players) or 1,
        "training": extra.get("training", "-"),
        "stopped": extra.get("stopped", False),
    }


def _build_self_play_agents(
    models: list[ActorCriticNet],
    pool: OpponentPool,
    pool_probability: float,
    rng: random.Random,
    device: torch.device,
) -> list[RLAgent]:
    agents: list[RLAgent] = []
    for seat in range(4):
        sampled_opponent = (
            pool.sample(rng)
            if pool_probability > 0 and len(pool) > 0 and rng.random() < pool_probability
            else None
        )
        agents.append(
            RLAgent(
                sampled_opponent or models[seat],
                device=device,
                rng=random.Random(rng.randrange(1_000_000_000)),
                deterministic=False,
                record=sampled_opponent is None,
            )
        )
    return agents


def _collect_agent_decisions(
    env: HKMahjongEnv,
    agents: list[RLAgent],
    episode_transitions: list[list[EpisodeTransition]],
    shaped_reward_coeff: float,
) -> None:
    for agent in agents:
        for decision in agent.pop_decisions():
            next_obs = env.observation(decision.player_id)
            next_state = encode_state(next_obs)
            next_mask = _next_mask(env, decision.player_id, decision.action_mask)
            potential_after = float(hand_potential(next_obs["hand_counts"], next_obs.get("open_melds", 0)))
            shaped = (potential_after - decision.potential_before) * shaped_reward_coeff
            episode_transitions[decision.player_id].append(
                EpisodeTransition(
                    state=decision.state,
                    action=decision.action,
                    next_state=next_state,
                    action_mask=decision.action_mask,
                    next_action_mask=next_mask,
                    shaped_reward=shaped,
                )
            )


def _commit_episode_transitions(
    episode_transitions: list[list[EpisodeTransition]],
    buffers: list[ReplayBuffer],
    terminal_rewards: list[float],
    gamma: float,
) -> None:
    for player_id, transitions in enumerate(episode_transitions):
        total = len(transitions)
        for index, transition in enumerate(transitions):
            discounted_terminal = terminal_rewards[player_id] * (gamma ** max(0, total - index - 1))
            # This buffer stores Monte Carlo return-style samples, so every
            # committed transition is terminal for bootstrap purposes.
            buffers[player_id].push(
                transition.state,
                transition.action,
                transition.shaped_reward + discounted_terminal,
                transition.next_state,
                True,
                transition.action_mask,
                transition.next_action_mask,
            )


def _next_mask(env: HKMahjongEnv, player_id: int, fallback: torch.Tensor) -> torch.Tensor:
    if env.done:
        return fallback.detach().cpu()
    try:
        return env.legal_action_mask(player_id)
    except Exception:
        return fallback.detach().cpu()


def _update_model(
    model: ActorCriticNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    value_coeff: float,
    entropy_coeff: float,
    max_grad_norm: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    batch = buffer.sample(batch_size, device)
    logits, values = model(batch["states"])
    legal_logits = masked_logits(logits, batch["action_masks"])
    dist = torch.distributions.Categorical(logits=legal_logits)
    log_probs = dist.log_prob(batch["actions"])
    entropy = dist.entropy().mean()
    probs = dist.probs
    claim_probability = probs[:, ACTION_PONG : ACTION_KONG + 1].sum(dim=1).clamp_min(1.0e-8)
    claim_exploration_loss = -0.02 * torch.log(claim_probability).mean()

    with torch.no_grad():
        _, next_values = model(batch["next_states"])
        targets = batch["rewards"] + gamma * (1.0 - batch["dones"]) * next_values
    advantages = targets - values
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = F.mse_loss(values, targets)
    loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy + claim_exploration_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "total_loss": float(loss.detach().cpu().item()),
        "entropy": float(entropy.detach().cpu().item()),
        "claim_exploration_loss": float(claim_exploration_loss.detach().cpu().item()),
    }


def _terminal_rewards(result: GameResult | None) -> list[float]:
    if result is None or result.winner is None:
        return [0.0, 0.0, 0.0, 0.0]
    rewards = [-0.15, -0.15, -0.15, -0.15]
    for winner in result.winners:
        rewards[winner] = 1.0
    for loser in result.losers:
        if loser not in result.winners:
            # TODO: Consider mapping score_delta to reward once scoring is stable.
            rewards[loser] = -1.0
    return rewards


def _load_or_create_models(
    model_path: str | None,
    device: torch.device,
) -> tuple[list[ActorCriticNet], int]:
    if not model_path:
        return [ActorCriticNet().to(device) for _ in range(4)], 0
    checkpoint = load_checkpoint_data(model_path, map_location=device)
    checkpoint_state_dim = int(checkpoint.get("state_dim", 0))
    if checkpoint_state_dim and checkpoint_state_dim != STATE_DIM:
        raise ValueError(
            f"checkpoint state_dim={checkpoint_state_dim} is incompatible with current state encoder; retrain the model"
        )
    players = checkpoint.get("players")
    games_trained = int(checkpoint.get("games_trained", 0))
    models: list[ActorCriticNet] = []
    if players:
        for seat in range(4):
            model = ActorCriticNet().to(device)
            model.load_state_dict(players[seat % len(players)])
            models.append(model)
    else:
        for _ in range(4):
            model = ActorCriticNet().to(device)
            model.load_state_dict(checkpoint["state_dict"])
            models.append(model)
    return models, games_trained


def _progress_payload(
    episode: int,
    total_episodes: int,
    games_trained: int,
    wins: list[int],
    draws: int,
    recent_results: list[GameResult],
    recent_losses: list[dict[str, float]],
    device: str,
    out_path: str,
    stopped: bool,
) -> dict[str, Any]:
    recent_count = max(1, len(recent_results))
    recent_draws = sum(1 for result in recent_results if result.winner is None)
    recent_won_games = sum(1 for result in recent_results if result.winner is not None)
    recent_turns = [result.turns for result in recent_results]
    losses = recent_losses[-20:]
    avg_policy_loss = _average([item["policy_loss"] for item in losses])
    avg_value_loss = _average([item["value_loss"] for item in losses])
    return {
        "episode": episode,
        "total_episodes": total_episodes,
        "percent": episode / max(1, total_episodes),
        "games_trained": games_trained,
        "wins": list(wins),
        "draws": draws,
        "win_rate": recent_won_games / recent_count,
        "draw_rate": draws / max(1, episode),
        "recent_win_rate": (recent_count - recent_draws) / recent_count,
        "recent_draw_rate": recent_draws / recent_count,
        "average_turns": _average(recent_turns),
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "device": device,
        "out_path": out_path,
        "stopped": stopped,
    }


def _average(values: list[float] | list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _clear_buffers(buffers: list[ReplayBuffer]) -> None:
    for buffer in buffers:
        buffer.clear()


def _snapshot_path(out_path: str, games_trained: int) -> str:
    path = Path(out_path)
    return str(path.with_name(f"{path.stem}_pool_{games_trained}{path.suffix or '.pt'}"))


def _should_stop(stop_event: threading.Event | None) -> bool:
    return stop_event is not None and stop_event.is_set()


def _wait_if_paused(pause_event: threading.Event | None, stop_event: threading.Event | None) -> bool:
    while pause_event is not None and pause_event.is_set():
        if _should_stop(stop_event):
            return True
        time.sleep(0.1)
    return False
