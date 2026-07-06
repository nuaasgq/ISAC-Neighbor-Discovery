from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import SimulationConfig, load_config  # noqa: E402
from isac_nd_sim.neural_contention_actor_critic import (  # noqa: E402
    AdaptiveGatedContentionGraphActorCritic,
    BalancedTopologyGatedContentionGraphActorCritic,
    ContentionGraphActorCritic,
    GatedContentionGraphActorCritic,
    TopologyAdaptiveGatedContentionGraphActorCritic,
)
from isac_nd_sim.marl_env import MODE_NAMES, MODE_TO_INDEX, MarlNeighborDiscoveryEnv  # noqa: E402
from isac_nd_sim.neural_scalegraph_beam_actor_critic import ScaleGraphBeamActorCritic  # noqa: E402
from isac_nd_sim.neural_shared_actor_critic import SharedBeamActorCritic  # noqa: E402
from isac_nd_sim.simulator import Action  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a real slot-level MARL policy for ISAC-assisted UAV neighbor discovery. "
            "The script logs per-step rewards, per-episode returns, resource usage, and held-out evaluation."
        )
    )
    parser.add_argument("--config", default="05_simulation/configs/paper_transfer_train_n10_b10_singlehop.yaml")
    parser.add_argument("--output", default="05_simulation/results_raw/marl_training")
    parser.add_argument("--algorithm", choices=["ippo", "mappo", "isac_mappo"], default="isac_mappo")
    parser.add_argument(
        "--network",
        choices=[
            "shared",
            "scalegraph_beam",
            "contention_shared",
            "gated_contention_shared",
            "adaptive_gated_contention_shared",
            "topology_adaptive_gated_contention_shared",
            "balanced_topology_gated_contention_shared",
        ],
        default="shared",
    )
    parser.add_argument("--reward-version", choices=["legacy", "collision_topology"], default="legacy")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--stochastic-eval", action="store_true", help="Sample from the policy during evaluation.")
    parser.add_argument("--eval-both", action="store_true", help="Run both deterministic and stochastic evaluations.")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--node-count", type=int, default=None)
    parser.add_argument("--azimuth-cells", type=int, default=None)
    parser.add_argument("--elevation-cells", type=int, default=None)
    parser.add_argument("--communication-range", type=float, default=None)
    parser.add_argument("--sensing-range", type=float, default=None)
    parser.add_argument("--false-alarm-rate", type=float, default=None)
    parser.add_argument("--miss-detection-rate", type=float, default=None)
    parser.add_argument("--angular-cell-offset-std", type=float, default=None)
    parser.add_argument("--sensing-period-slots", type=int, default=None)
    parser.add_argument("--mobility-model", default=None)
    parser.add_argument("--env-protocol", default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--candidate-mask", action="store_true", help="Use local ISAC candidate masks in beam sampling.")
    parser.add_argument("--candidate-score", action="store_true", help="Use local candidate scores in beam-token features.")
    parser.add_argument("--topology-deficit", action="store_true", help="Use local discovered-degree deficit token.")
    parser.add_argument("--rule-residual", action="store_true", help="Use local rule logits and beam priors as residual policy logits.")
    parser.add_argument("--rule-residual-scale", type=float, default=1.0)
    parser.add_argument("--disable-isac-features", action="store_true", help="Disable all ISAC/structured feature flags.")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--step-log-period", type=int, default=1)
    parser.add_argument("--resource-log-period", type=int, default=25)
    parser.add_argument("--max-rss-mb", type=float, default=12000.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=92.0)
    return parser.parse_args()


class CentralizedPooledCritic:
    """Scale-invariant centralized state-value critic for CTDE training."""

    def __new__(cls, input_dim: int, hidden_dim: int, torch_module: Any):
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )

            def forward(self, features: Any) -> Any:
                return self.net(features).squeeze(-1)

        return Module()


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for run_marl_training.py") from exc

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    validate_args(args)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = override_config(load_config(args.config), args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    feature_flags = resolved_feature_flags(args)
    env_protocol = resolved_env_protocol(args)
    policy = build_policy(
        str(getattr(args, "network", "shared")),
        cfg.n_beams,
        hidden_dim=int(args.hidden_dim),
        device="cpu",
        use_candidate_mask=feature_flags["candidate_mask"],
        use_candidate_score=feature_flags["candidate_score"],
        use_topology_deficit=feature_flags["topology_deficit"],
        use_rule_residual=feature_flags["rule_residual"],
        rule_residual_scale=float(args.rule_residual_scale),
    )
    centralized = str(args.algorithm) in {"mappo", "isac_mappo"}
    critic = None
    params = list(policy.parameters())
    if centralized:
        critic = CentralizedPooledCritic(central_feature_dim(), int(args.hidden_dim), torch)
        params += list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=float(args.learning_rate))

    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    global_step = 0

    for episode in range(int(args.episodes)):
        trajectory = collect_trajectory(
            cfg=cfg,
            policy=policy,
            torch_module=torch,
            seed=int(args.seed) + episode,
            episode=episode,
            env_protocol=env_protocol,
            global_step_start=global_step,
            step_log_period=int(args.step_log_period),
            args=args,
            resource_rows=resource_rows,
        )
        global_step += int(trajectory["slots"])
        losses = update_policy(
            trajectory=trajectory,
            policy=policy,
            critic=critic,
            optimizer=optimizer,
            torch_module=torch,
            functional=F,
            args=args,
            centralized=centralized,
        )
        row = build_episode_row(trajectory, losses, episode, global_step, args, cfg)
        episode_rows.append(row)
        step_rows.extend(trajectory["step_rows"])
        if should_checkpoint(episode + 1, int(args.checkpoint_interval)):
            save_checkpoint(output / f"checkpoint_ep{episode + 1:05d}.pt", policy, critic, optimizer, args, cfg, episode + 1, torch)
        if should_checkpoint(episode + 1, int(args.eval_interval)):
            eval_rows.extend(
            evaluate_policy(
                cfg=cfg,
                policy=policy,
                torch_module=torch,
                args=args,
                env_protocol=env_protocol,
                start_episode=episode + 1,
                seed_start=int(args.seed) + 1_000_000 + 1000 * (episode + 1),
                stochastic_eval=bool(args.stochastic_eval),
            )
            )
        flush_outputs(output, step_rows, episode_rows, eval_rows, resource_rows)

    if int(args.eval_episodes) > 0:
        eval_rows.extend(
            evaluate_policy(
                cfg=cfg,
                policy=policy,
                torch_module=torch,
                args=args,
                env_protocol=env_protocol,
                start_episode=int(args.episodes),
                seed_start=int(args.seed) + 2_000_000,
                stochastic_eval=bool(args.stochastic_eval),
            )
        )
    save_checkpoint(output / "final_model.pt", policy, critic, optimizer, args, cfg, int(args.episodes), torch)
    manifest = build_manifest(args, cfg, feature_flags, env_protocol, centralized, episode_rows, eval_rows)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    flush_outputs(output, step_rows, episode_rows, eval_rows, resource_rows)
    return manifest


def validate_args(args: argparse.Namespace) -> None:
    if int(args.episodes) <= 0:
        raise ValueError("--episodes must be positive.")
    if int(args.slots) <= 0:
        raise ValueError("--slots must be positive.")
    if int(args.ppo_epochs) <= 0:
        raise ValueError("--ppo-epochs must be positive.")
    if float(args.max_rss_mb) <= 0.0:
        raise ValueError("--max-rss-mb must be positive.")


def override_config(config: SimulationConfig, args: argparse.Namespace) -> SimulationConfig:
    replacements: dict[str, Any] = {
        "slots_per_episode": int(args.slots),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
    }
    optional_fields = {
        "node_count": "n_nodes",
        "azimuth_cells": "azimuth_cells",
        "elevation_cells": "elevation_cells",
        "communication_range": "communication_range_m",
        "sensing_range": "sensing_range_m",
        "false_alarm_rate": "false_alarm_rate",
        "miss_detection_rate": "miss_detection_rate",
        "angular_cell_offset_std": "angular_cell_offset_std",
        "sensing_period_slots": "sensing_period_slots",
    }
    for arg_name, field_name in optional_fields.items():
        value = getattr(args, arg_name)
        if value is not None:
            replacements[field_name] = value
    mobility = dict(config.mobility)
    if args.mobility_model is not None:
        mobility["model"] = str(args.mobility_model)
    replacements["mobility"] = mobility
    return replace(config, **replacements)


def resolved_feature_flags(args: argparse.Namespace) -> dict[str, bool]:
    if bool(args.disable_isac_features):
        return {
            "candidate_mask": False,
            "candidate_score": False,
            "topology_deficit": bool(args.topology_deficit),
            "rule_residual": False,
        }
    if str(args.algorithm) == "isac_mappo":
        return {
            "candidate_mask": True if not args.candidate_mask else bool(args.candidate_mask),
            "candidate_score": True if not args.candidate_score else bool(args.candidate_score),
            "topology_deficit": True if not args.topology_deficit else bool(args.topology_deficit),
            "rule_residual": True if not args.rule_residual else bool(args.rule_residual),
        }
    return {
        "candidate_mask": bool(args.candidate_mask),
        "candidate_score": bool(args.candidate_score),
        "topology_deficit": bool(args.topology_deficit),
        "rule_residual": bool(args.rule_residual),
    }


def build_policy(
    network: str, *args: Any, **kwargs: Any
) -> (
    SharedBeamActorCritic
    | ScaleGraphBeamActorCritic
    | ContentionGraphActorCritic
    | GatedContentionGraphActorCritic
    | AdaptiveGatedContentionGraphActorCritic
    | TopologyAdaptiveGatedContentionGraphActorCritic
):
    if str(network) == "shared":
        return SharedBeamActorCritic(*args, **kwargs)
    if str(network) == "scalegraph_beam":
        return ScaleGraphBeamActorCritic(*args, **kwargs)
    if str(network) == "contention_shared":
        return ContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "gated_contention_shared":
        return GatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "adaptive_gated_contention_shared":
        return AdaptiveGatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "topology_adaptive_gated_contention_shared":
        return TopologyAdaptiveGatedContentionGraphActorCritic(*args, **kwargs)
    if str(network) == "balanced_topology_gated_contention_shared":
        return BalancedTopologyGatedContentionGraphActorCritic(*args, **kwargs)
    raise ValueError(f"Unsupported network: {network}")


def resolved_env_protocol(args: argparse.Namespace) -> str:
    if args.env_protocol:
        return str(args.env_protocol)
    if bool(args.disable_isac_features):
        return "structured_marl_no_isac"
    return "isac_structured_marl"


def collect_trajectory(
    cfg: SimulationConfig,
    policy: SharedBeamActorCritic,
    torch_module: Any,
    seed: int,
    episode: int,
    env_protocol: str,
    global_step_start: int,
    step_log_period: int,
    args: argparse.Namespace,
    resource_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    env = MarlNeighborDiscoveryEnv(
        cfg,
        seed=seed,
        protocol=env_protocol,
        reward_version=str(getattr(args, "reward_version", "legacy")),
    )
    observations, _ = env.reset(seed=seed)
    old_log_probs = []
    rewards = []
    observations_by_step = []
    actions_by_step = []
    central_features = []
    step_rows = []
    cumulative_reward = 0.0
    truncated = False
    policy.train()

    while not truncated:
        slot = len(rewards)
        state = env.training_state()
        observations_by_step.append(copy_observations(observations))
        central_features.append(central_state_features(state, cfg))
        with torch_module.no_grad():
            step = policy.act(observations, deterministic=False)
        observations, reward, _terminated, truncated, info = env.step(step.actions)
        old_log_probs.append(step.log_probs.detach().cpu())
        reward_tensor = torch_module.as_tensor(reward, dtype=torch_module.float32)
        rewards.append(reward_tensor)
        actions_by_step.append(step.actions)
        cumulative_reward += float(reward_tensor.sum().item())
        global_step = global_step_start + slot + 1

        if step_log_period > 0 and slot % step_log_period == 0:
            true_edges = max(1, int(state["true_edges"].shape[0]))
            step_rows.append(
                {
                    "episode": episode,
                    "slot": slot,
                    "training_step": global_step,
                    "seed": seed,
                    "algorithm": str(args.algorithm),
                    "env_protocol": env_protocol,
                    "reward_sum": float(reward_tensor.sum().item()),
                    "reward_mean": float(reward_tensor.mean().item()),
                    "episode_cumulative_reward": cumulative_reward,
                    "new_edges_count": int(info["new_edges_count"]),
                    "discovered_edges": int(info["discovered_edges_count"]),
                    "true_edges": true_edges,
                    "discovery_rate": int(info["discovered_edges_count"]) / true_edges,
                    "empty_scan_ratio": float(info["empty_scan_ratio"]),
                    "collision_count": int(info["collision_count"]),
                    "lambda2": float(info["lambda2"]),
                    "lcc_ratio": float(info["lcc_ratio"]),
                    "scan_actions": int(info["scan_actions"]),
                    "tx_actions": int(info["tx_actions"]),
                    "rx_actions": int(info["rx_actions"]),
                    "sense_actions": int(info["sense_actions"]),
                    "idle_actions": int(info["idle_actions"]),
                }
            )
        if int(args.resource_log_period) > 0 and global_step % int(args.resource_log_period) == 0:
            snapshot = resource_snapshot()
            snapshot.update({"episode": episode, "slot": slot, "training_step": global_step})
            resource_rows.append(snapshot)
            enforce_resource_limits(snapshot, args)

    summary = env._sim.summarize(episode).as_dict()
    return {
        "episode": episode,
        "seed": seed,
        "slots": len(rewards),
        "observations": observations_by_step,
        "actions": actions_by_step,
        "old_log_probs": torch_module.stack([row.to(policy.device) for row in old_log_probs]),
        "rewards": torch_module.stack(rewards).to(policy.device),
        "central_features": torch_module.as_tensor(np.stack(central_features), dtype=torch_module.float32, device=policy.device),
        "step_rows": step_rows,
        "summary": summary,
    }


def update_policy(
    trajectory: dict[str, Any],
    policy: SharedBeamActorCritic,
    critic: Any,
    optimizer: Any,
    torch_module: Any,
    functional: Any,
    args: argparse.Namespace,
    centralized: bool,
) -> dict[str, float]:
    old_log_probs = trajectory["old_log_probs"]
    rewards = trajectory["rewards"]
    policy_losses = []
    value_losses = []
    entropy_values = []
    approx_kls = []
    clip_fracs = []

    if centralized:
        if critic is None:
            raise RuntimeError("Centralized critic is required for MAPPO-style training.")
        global_rewards = rewards.mean(dim=1)
        returns = discounted_returns_1d(global_rewards, float(args.gamma), torch_module)
        for _ in range(int(args.ppo_epochs)):
            log_probs, _local_values, entropies = evaluate_actions(policy, trajectory["observations"], trajectory["actions"])
            values = critic(trajectory["central_features"])
            advantages = normalize_advantages(returns - values.detach())
            ratio = torch_module.exp(log_probs - old_log_probs)
            clipped_ratio = torch_module.clamp(ratio, 1.0 - float(args.clip_epsilon), 1.0 + float(args.clip_epsilon))
            policy_loss = -torch_module.minimum(ratio * advantages[:, None], clipped_ratio * advantages[:, None]).mean()
            value_loss = functional.mse_loss(values, returns)
            entropy = entropies.mean()
            loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy
            optimizer.zero_grad()
            loss.backward()
            torch_module.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], float(args.max_grad_norm))
            optimizer.step()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropy_values.append(float(entropy.item()))
            approx_kls.append(float((old_log_probs - log_probs).mean().detach().item()))
            clip_fracs.append(float((torch_module.abs(ratio - 1.0) > float(args.clip_epsilon)).float().mean().item()))
        return loss_summary(policy_losses, value_losses, entropy_values, approx_kls, clip_fracs)

    returns = discounted_returns_2d(rewards, float(args.gamma), torch_module)
    for _ in range(int(args.ppo_epochs)):
        log_probs, local_values, entropies = evaluate_actions(policy, trajectory["observations"], trajectory["actions"])
        advantages = normalize_advantages(returns - local_values.detach())
        ratio = torch_module.exp(log_probs - old_log_probs)
        clipped_ratio = torch_module.clamp(ratio, 1.0 - float(args.clip_epsilon), 1.0 + float(args.clip_epsilon))
        policy_loss = -torch_module.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = functional.mse_loss(local_values, returns)
        entropy = entropies.mean()
        loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy
        optimizer.zero_grad()
        loss.backward()
        torch_module.nn.utils.clip_grad_norm_(policy.parameters(), float(args.max_grad_norm))
        optimizer.step()
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        entropy_values.append(float(entropy.item()))
        approx_kls.append(float((old_log_probs - log_probs).mean().detach().item()))
        clip_fracs.append(float((torch_module.abs(ratio - 1.0) > float(args.clip_epsilon)).float().mean().item()))
    return loss_summary(policy_losses, value_losses, entropy_values, approx_kls, clip_fracs)


def evaluate_actions(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    actions_by_step: Sequence[Sequence[Action]],
) -> tuple[Any, Any, Any]:
    torch = policy.torch
    from torch.distributions import Categorical

    log_prob_rows = []
    value_rows = []
    entropy_rows = []
    for observations, actions in zip(observations_by_step, actions_by_step, strict=True):
        step_log_probs = []
        step_values = []
        step_entropies = []
        for observation, action in zip(observations, actions, strict=True):
            mode_logits, beam_logits, value = policy.logits_value(observation, hard_mask=True)
            mode_dist = Categorical(logits=mode_logits)
            beam_dist = Categorical(logits=beam_logits)
            mode_idx = MODE_TO_INDEX[action.mode]
            mode_tensor = torch.as_tensor(mode_idx, dtype=torch.long, device=policy.device)
            log_prob = mode_dist.log_prob(mode_tensor)
            if action.mode != "idle":
                beam_tensor = torch.as_tensor(int(action.beam), dtype=torch.long, device=policy.device)
                log_prob = log_prob + beam_dist.log_prob(beam_tensor)
            step_log_probs.append(log_prob)
            step_values.append(value.squeeze(-1))
            step_entropies.append(mode_dist.entropy() + beam_dist.entropy())
        log_prob_rows.append(torch.stack(step_log_probs))
        value_rows.append(torch.stack(step_values))
        entropy_rows.append(torch.stack(step_entropies))
    return torch.stack(log_prob_rows), torch.stack(value_rows), torch.stack(entropy_rows)


def discounted_returns_2d(rewards: Any, gamma: float, torch_module: Any) -> Any:
    running = torch_module.zeros_like(rewards[-1])
    returns = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch_module.stack(returns)


def discounted_returns_1d(rewards: Any, gamma: float, torch_module: Any) -> Any:
    running = torch_module.zeros((), dtype=rewards.dtype, device=rewards.device)
    returns = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch_module.stack(returns)


def normalize_advantages(advantages: Any) -> Any:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def central_feature_dim() -> int:
    return 23


def central_state_features(state: dict[str, Any], cfg: SimulationConfig) -> np.ndarray:
    positions = np.asarray(state["positions"], dtype=np.float32)
    velocities = np.asarray(state["velocities"], dtype=np.float32)
    discovered = np.asarray(state["discovered_adjacency"], dtype=np.float32)
    true_adj = np.asarray(state["true_adjacency"], dtype=np.float32)
    belief = np.asarray(state["belief"], dtype=np.float32)
    n = max(1, int(cfg.n_nodes))
    possible_edges = max(1.0, n * (n - 1) / 2.0)
    area = np.asarray(cfg.area_size_m, dtype=np.float32)
    pos_norm = positions / np.maximum(area, 1e-6)
    speed = np.linalg.norm(velocities, axis=1)
    discovered_degree = discovered.sum(axis=1) / max(1, n - 1)
    true_degree = true_adj.sum(axis=1) / max(1, n - 1)
    upper = np.triu_indices(n, k=1)
    discovered_edges = float(discovered[upper].sum())
    true_edges = float(true_adj[upper].sum())
    return np.asarray(
        [
            float(state["slot"]) / max(1.0, float(cfg.slots_per_episode)),
            n / 100.0,
            cfg.n_beams / 2000.0,
            discovered_edges / possible_edges,
            true_edges / possible_edges,
            discovered_edges / max(1.0, true_edges),
            float(np.mean(belief)),
            float(np.std(belief)),
            float(np.max(belief)),
            *np.mean(pos_norm, axis=0).tolist(),
            *np.std(pos_norm, axis=0).tolist(),
            float(np.mean(speed) / max(1.0, _speed_scale(cfg))),
            float(np.std(speed) / max(1.0, _speed_scale(cfg))),
            float(np.mean(discovered_degree)),
            float(np.std(discovered_degree)),
            float(np.mean(true_degree)),
            float(np.std(true_degree)),
            float(np.mean(np.triu(discovered @ discovered, k=1)) / max(1, n)),
            float(np.mean(np.triu(true_adj @ true_adj, k=1)) / max(1, n)),
        ],
        dtype=np.float32,
    )


def _speed_scale(cfg: SimulationConfig) -> float:
    max_speed = float(cfg.mobility.get("max_speed_mps", 0.0) or 0.0)
    if max_speed > 0.0:
        return max_speed
    return float(cfg.mobility.get("speed_mean_mps", 15.0)) + 3.0 * float(cfg.mobility.get("speed_std_mps", 3.0))


def copy_observations(observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = []
    for observation in observations:
        row: dict[str, Any] = {}
        for key, value in observation.items():
            row[key] = value.copy() if hasattr(value, "copy") else value
        copied.append(row)
    return copied


def build_episode_row(
    trajectory: dict[str, Any],
    losses: dict[str, float],
    episode: int,
    global_step: int,
    args: argparse.Namespace,
    cfg: SimulationConfig,
) -> dict[str, Any]:
    rewards = trajectory["rewards"]
    summary = trajectory["summary"]
    row = {
        "episode": episode,
        "training_step": global_step,
        "seed": trajectory["seed"],
        "algorithm": str(args.algorithm),
        "slots": cfg.slots_per_episode,
        "n_nodes": cfg.n_nodes,
        "n_beams": cfg.n_beams,
        "episode_return_sum": float(rewards.sum().item()),
        "episode_return_mean_per_agent": float(rewards.sum(dim=0).mean().item()),
        "step_reward_mean": float(rewards.mean().item()),
        "step_reward_sum_mean": float(rewards.sum(dim=1).mean().item()),
    }
    row.update(losses)
    row.update({key: summary[key] for key in summary})
    return row


def evaluate_policy(
    cfg: SimulationConfig,
    policy: SharedBeamActorCritic,
    torch_module: Any,
    args: argparse.Namespace,
    env_protocol: str,
    start_episode: int,
    seed_start: int,
    stochastic_eval: bool,
) -> list[dict[str, Any]]:
    rows = []
    policy.eval()
    with torch_module.no_grad():
        eval_modes = (False, True) if bool(args.eval_both) else (bool(stochastic_eval),)
        eval_training_step = int(start_episode) * int(getattr(args, "slots", 1) or 1)
        for mode_index, use_stochastic in enumerate(eval_modes):
            for offset in range(int(args.eval_episodes)):
                seed = seed_start + 10_000 * mode_index + offset
                env = MarlNeighborDiscoveryEnv(
                    cfg,
                    seed=seed,
                    protocol=env_protocol,
                    reward_version=str(getattr(args, "reward_version", "legacy")),
                )
                observations, _ = env.reset(seed=seed)
                rewards = []
                truncated = False
                while not truncated:
                    step = policy.act(observations, deterministic=not use_stochastic)
                    observations, reward, _terminated, truncated, _info = env.step(step.actions)
                    rewards.append(torch_module.as_tensor(reward, dtype=torch_module.float32))
                rewards_tensor = torch_module.stack(rewards)
                summary = env._sim.summarize(start_episode + offset).as_dict()
                row = {
                    "phase": "eval_stochastic" if use_stochastic else "eval_deterministic",
                    "eval_after_episode": start_episode,
                    "training_step": eval_training_step,
                    "eval_episode": offset,
                    "seed": seed,
                    "algorithm": str(args.algorithm),
                    "env_protocol": env_protocol,
                    "episode_return_sum": float(rewards_tensor.sum().item()),
                    "episode_return_mean_per_agent": float(rewards_tensor.sum(dim=0).mean().item()),
                    "step_reward_mean": float(rewards_tensor.mean().item()),
                }
                row.update(summary)
                rows.append(row)
    policy.train()
    return rows


def loss_summary(
    policy_losses: list[float],
    value_losses: list[float],
    entropy_values: list[float],
    approx_kls: list[float],
    clip_fracs: list[float],
) -> dict[str, float]:
    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropy_values)),
        "approx_kl": float(np.mean(approx_kls)),
        "clip_fraction": float(np.mean(clip_fracs)),
    }


def should_checkpoint(index: int, interval: int) -> bool:
    return interval > 0 and index % interval == 0


def save_checkpoint(
    path: Path,
    policy: SharedBeamActorCritic,
    critic: Any,
    optimizer: Any,
    args: argparse.Namespace,
    cfg: SimulationConfig,
    episode: int,
    torch_module: Any,
) -> None:
    checkpoint = {
        "episode": int(episode),
        "algorithm": str(args.algorithm),
        "policy_state_dict": policy.model.state_dict(),
        "critic_state_dict": critic.state_dict() if critic is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "config": cfg.__dict__,
    }
    torch_module.save(checkpoint, path)


def flush_outputs(
    output: Path,
    step_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
) -> None:
    write_rows(output / "step_rewards.csv", step_rows)
    write_rows(output / "episode_metrics.csv", episode_rows)
    write_rows(output / "eval_episode_metrics.csv", eval_rows)
    write_rows(output / "resource_log.csv", resource_rows)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resource_snapshot() -> dict[str, Any]:
    try:
        import psutil
    except ImportError:
        return windows_resource_snapshot()
    process = psutil.Process()
    memory = psutil.virtual_memory()
    return {
        "rss_mb": process.memory_info().rss / (1024.0 * 1024.0),
        "process_cpu_percent": process.cpu_percent(interval=None),
        "system_memory_percent": memory.percent,
        "system_available_mb": memory.available / (1024.0 * 1024.0),
    }


def windows_resource_snapshot() -> dict[str, Any]:
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return empty_resource_snapshot()
    if not hasattr(ctypes, "windll"):
        return empty_resource_snapshot()

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        memory = MemoryStatusEx()
        memory.dwLength = ctypes.sizeof(MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            return empty_resource_snapshot()
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            rss_mb = current_process_rss_mb_fallback()
        else:
            rss_mb = counters.WorkingSetSize / (1024.0 * 1024.0)
        available_mb = memory.ullAvailPhys / (1024.0 * 1024.0)
        total_mb = max(memory.ullTotalPhys / (1024.0 * 1024.0), 1e-9)
        return {
            "rss_mb": rss_mb,
            "process_cpu_percent": "",
            "system_memory_percent": 100.0 * (1.0 - available_mb / total_mb),
            "system_available_mb": available_mb,
        }
    except Exception:
        return empty_resource_snapshot()


def current_process_rss_mb_fallback() -> float | str:
    try:
        import subprocess

        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-Process -Id {os.getpid()}).WorkingSet64",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        value = completed.stdout.strip()
        return float(value) / (1024.0 * 1024.0) if value else ""
    except Exception:
        return ""


def empty_resource_snapshot() -> dict[str, Any]:
    return {
        "rss_mb": "",
        "process_cpu_percent": "",
        "system_memory_percent": "",
        "system_available_mb": "",
    }


def enforce_resource_limits(snapshot: dict[str, Any], args: argparse.Namespace) -> None:
    rss = snapshot.get("rss_mb")
    memory_percent = snapshot.get("system_memory_percent")
    if isinstance(rss, (int, float)) and rss > float(args.max_rss_mb):
        raise RuntimeError(f"RSS memory limit exceeded: {rss:.1f} MB > {float(args.max_rss_mb):.1f} MB")
    if isinstance(memory_percent, (int, float)) and memory_percent > float(args.max_system_memory_percent):
        raise RuntimeError(
            f"System memory limit exceeded: {memory_percent:.1f}% > {float(args.max_system_memory_percent):.1f}%"
        )


def build_manifest(
    args: argparse.Namespace,
    cfg: SimulationConfig,
    feature_flags: dict[str, bool],
    env_protocol: str,
    centralized: bool,
    episode_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": str(args.algorithm),
        "network": str(getattr(args, "network", "shared")),
        "reward_version": str(getattr(args, "reward_version", "legacy")),
        "scope": "real_marl_training",
        "config": str(args.config),
        "output": str(args.output),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "slots_per_episode": int(cfg.slots_per_episode),
        "slot_duration_ms": float(cfg.slot_duration_s) * 1000.0,
        "node_count": int(cfg.n_nodes),
        "beam_count": int(cfg.n_beams),
        "azimuth_cells": int(cfg.azimuth_cells),
        "elevation_cells": int(cfg.elevation_cells),
        "communication_range_m": float(cfg.communication_range_m),
        "sensing_range_m": float(cfg.sensing_range_m),
        "env_protocol": env_protocol,
        "feature_flags": feature_flags,
        "centralized_training_decentralized_execution": bool(centralized),
        "decentralized_actor_observation": True,
        "centralized_critic_uses_training_state_only": bool(centralized),
        "logs_per_step_reward": True,
        "logs_episode_return": True,
        "torch_threads": int(args.torch_threads),
        "resource_limits": {
            "max_rss_mb": float(args.max_rss_mb),
            "max_system_memory_percent": float(args.max_system_memory_percent),
        },
        "stochastic_eval": bool(args.stochastic_eval),
        "eval_both": bool(args.eval_both),
        "final_train": episode_rows[-1] if episode_rows else {},
        "final_eval": eval_rows[-1] if eval_rows else {},
        "files": [
            "step_rewards.csv",
            "episode_metrics.csv",
            "eval_episode_metrics.csv",
            "resource_log.csv",
            "final_model.pt",
            "manifest.json",
        ],
    }


def main() -> None:
    print(json.dumps(run_training(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
