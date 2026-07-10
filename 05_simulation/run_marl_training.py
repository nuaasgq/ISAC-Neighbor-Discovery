from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
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
from isac_nd_sim.marl_env import (  # noqa: E402
    ACCESS_GATE_TO_INDEX,
    CANDIDATE_SOURCES,
    MODE_NAMES,
    MODE_TO_INDEX,
    REWARD_VERSIONS,
    MarlNeighborDiscoveryEnv,
)
from isac_nd_sim.neural_scalegraph_beam_actor_critic import ScaleGraphBeamActorCritic  # noqa: E402
from isac_nd_sim.neural_shared_actor_critic import SharedBeamActorCritic  # noqa: E402
from isac_nd_sim.simulator import (  # noqa: E402
    ACCESS_AGGRESSIVE,
    ACCESS_BACKOFF,
    ACCESS_NORMAL,
    MODE_RX,
    MODE_SENSE,
    MODE_TX,
    Action,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a real slot-level MARL policy for ISAC-assisted UAV neighbor discovery. "
            "The script logs per-step rewards, per-episode returns, resource usage, and held-out evaluation."
        )
    )
    parser.add_argument("--config", default="05_simulation/configs/twc_trainable_n10.yaml")
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
        default="contention_shared",
    )
    parser.add_argument("--reward-version", choices=REWARD_VERSIONS, default="discovery_first")
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
    parser.add_argument(
        "--candidate-source",
        choices=CANDIDATE_SOURCES,
        default="default",
        help="Source used to build MARL candidate_mask/candidate_score observations.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.985)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--separate-action-loss",
        action="store_true",
        help="Train mode, beam, and gate action factors with separate PPO losses.",
    )
    parser.add_argument("--beam-loss-coef", type=float, default=1.0)
    parser.add_argument("--gate-loss-coef", type=float, default=0.25)
    parser.add_argument(
        "--beam-rank-aux-coef",
        type=float,
        default=0.0,
        help="Auxiliary coefficient for fitting beam logits to local candidate-score rankings.",
    )
    parser.add_argument("--beam-rank-temperature", type=float, default=4.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--expert-bc-weight",
        type=float,
        default=0.0,
        help="Auxiliary behavior-cloning weight from a local rule expert. Zero disables expert guidance.",
    )
    parser.add_argument(
        "--expert-protocol",
        default="collision_aware_isac",
        help="Local simulator protocol used as the behavior-cloning expert when --expert-bc-weight > 0.",
    )
    parser.add_argument("--candidate-mask", action="store_true", help="Use local ISAC candidate masks in beam sampling.")
    parser.add_argument(
        "--candidate-score",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use local ISAC candidate scores in beam-token features (enabled by default for ISAC-MAPPO).",
    )
    parser.add_argument("--topology-deficit", action="store_true", help="Use local discovered-degree deficit token.")
    parser.add_argument("--rule-residual", action="store_true", help="Use local rule logits and beam priors as residual policy logits.")
    parser.add_argument("--rule-residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--contention-mode-prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt in to the hand-coded contention/topology mode-logit prior.",
    )
    parser.add_argument("--disable-contention-mode-prior", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable-isac-features", action="store_true", help="Disable all ISAC/structured feature flags.")
    parser.add_argument(
        "--forbid-sense",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-standalone-sense",
        action="store_true",
        help="Opt in to standalone SENSE actions; the default single-RF model senses only during TX.",
    )
    parser.add_argument(
        "--allow-idle",
        action="store_true",
        help="Opt in to IDLE; the default neighbor-discovery action space contains only TX/RX and beam selection.",
    )
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
        use_contention_mode_prior=contention_mode_prior_enabled(args),
        disabled_modes=disabled_modes_from_args(args),
    )
    setattr(policy, "_expert_bc_weight_cache", float(getattr(args, "expert_bc_weight", 0.0)))
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
    if float(getattr(args, "expert_bc_weight", 0.0)) < 0.0:
        raise ValueError("--expert-bc-weight must be nonnegative.")
    if float(getattr(args, "beam_loss_coef", 1.0)) < 0.0:
        raise ValueError("--beam-loss-coef must be nonnegative.")
    if float(getattr(args, "gate_loss_coef", 0.25)) < 0.0:
        raise ValueError("--gate-loss-coef must be nonnegative.")
    if float(getattr(args, "beam_rank_aux_coef", 0.0)) < 0.0:
        raise ValueError("--beam-rank-aux-coef must be nonnegative.")
    if float(getattr(args, "beam_rank_temperature", 4.0)) <= 0.0:
        raise ValueError("--beam-rank-temperature must be positive.")
    if bool(getattr(args, "separate_action_loss", False)) and str(getattr(args, "network", "")) == "scalegraph_beam":
        raise ValueError("--separate-action-loss is not implemented for scalegraph_beam.")


def disabled_modes_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    disabled_modes: list[str] = []
    if hasattr(args, "allow_standalone_sense"):
        disable_sense = not bool(args.allow_standalone_sense)
    else:
        disable_sense = bool(getattr(args, "forbid_sense", False))
    disable_sense = disable_sense or bool(getattr(args, "forbid_sense", False))
    if disable_sense:
        disabled_modes.append(MODE_SENSE)
    if hasattr(args, "allow_idle") and not bool(args.allow_idle):
        disabled_modes.append("idle")
    return tuple(disabled_modes)


def contention_mode_prior_enabled(args: argparse.Namespace) -> bool:
    enabled = bool(getattr(args, "contention_mode_prior", False))
    return enabled and not bool(getattr(args, "disable_contention_mode_prior", False))


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
            "candidate_mask": bool(args.candidate_mask),
            "candidate_score": True if args.candidate_score is None else bool(args.candidate_score),
            "topology_deficit": bool(args.topology_deficit),
            "rule_residual": bool(args.rule_residual),
        }
    return {
        "candidate_mask": bool(args.candidate_mask),
        "candidate_score": bool(args.candidate_score) if args.candidate_score is not None else False,
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
    use_contention_mode_prior = bool(kwargs.pop("use_contention_mode_prior", False))
    if str(network) == "shared":
        return SharedBeamActorCritic(*args, **kwargs)
    if str(network) == "scalegraph_beam":
        return ScaleGraphBeamActorCritic(*args, **kwargs)
    kwargs["use_contention_mode_prior"] = use_contention_mode_prior
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
    return "improved_rl_isac_tables"


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
        candidate_source=str(getattr(args, "candidate_source", "default")),
    )
    observations, _ = env.reset(seed=seed)
    old_log_probs = []
    old_mode_log_probs = []
    old_beam_log_probs = []
    old_gate_log_probs = []
    active_beam_masks = []
    rewards = []
    observations_by_step = []
    actions_by_step = []
    expert_actions_by_step = []
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
        if float(getattr(args, "expert_bc_weight", 0.0)) > 0.0:
            expert_actions_by_step.append(
                expert_actions_for_env(env, str(getattr(args, "expert_protocol", "collision_aware_isac")))
            )
        else:
            expert_actions_by_step.append([])
        with torch_module.no_grad():
            step = policy.act(observations, deterministic=False)
        observations, reward, _terminated, truncated, info = env.step(step.actions)
        old_log_probs.append(step.log_probs.detach().cpu())
        old_mode_log_probs.append(component_or_zeros(step.mode_log_probs, step.log_probs).detach().cpu())
        old_beam_log_probs.append(component_or_zeros(step.beam_log_probs, step.log_probs).detach().cpu())
        old_gate_log_probs.append(component_or_zeros(step.gate_log_probs, step.log_probs).detach().cpu())
        active_beam_masks.append(active_mask_tensor(step, step.actions, torch_module).detach().cpu())
        reward_tensor = torch_module.as_tensor(reward, dtype=torch_module.float32)
        rewards.append(reward_tensor)
        actions_by_step.append(step.actions)
        cumulative_reward += float(reward_tensor.sum().item())
        global_step = global_step_start + slot + 1

        if step_log_period > 0 and slot % step_log_period == 0:
            true_edges = max(1, int(state["true_edges"].shape[0]))
            row = {
                "episode": episode,
                "slot": slot,
                "training_step": global_step,
                "seed": seed,
                "algorithm": str(args.algorithm),
                "env_protocol": env_protocol,
                "reward_sum": float(reward_tensor.sum().item()),
                "reward_mean": float(reward_tensor.mean().item()),
                "reward_std_across_agents": float(reward_tensor.std(unbiased=False).item()),
                "reward_min_agent": float(reward_tensor.min().item()),
                "reward_max_agent": float(reward_tensor.max().item()),
                "positive_reward_agents": int((reward_tensor > 0.0).sum().item()),
                "episode_cumulative_reward": cumulative_reward,
                "new_edges_count": int(info["new_edges_count"]),
                "discovered_edges": int(info["discovered_edges_count"]),
                "active_discovered_edges": int(info.get("active_discovered_edges_count", info["discovered_edges_count"])),
                "true_edges": true_edges,
                "discovery_rate": int(info["discovered_edges_count"]) / true_edges,
                "empty_scan_ratio": float(info["empty_scan_ratio"]),
                "handshake_attempts": int(info.get("handshake_attempts", 0)),
                "handshake_successes": int(info.get("handshake_successes", 0)),
                "forward_decode_failures": int(info.get("forward_decode_failures", 0)),
                "ack_decode_failures": int(info.get("ack_decode_failures", 0)),
                "interference_limited_failures": int(info.get("interference_limited_failures", 0)),
                "phy_outage_failures": int(info.get("phy_outage_failures", 0)),
                "mean_handshake_sinr_db": float(info.get("mean_handshake_sinr_db", 0.0)),
                "collision_count": int(info["collision_count"]),
                "lambda2": float(info["lambda2"]),
                "knowledge_lambda2": float(info.get("knowledge_lambda2", info["lambda2"])),
                "lcc_ratio": float(info["lcc_ratio"]),
                "scan_actions": int(info["scan_actions"]),
                "tx_actions": int(info["tx_actions"]),
                "rx_actions": int(info["rx_actions"]),
                "sense_actions": int(info["sense_actions"]),
                "idle_actions": int(info["idle_actions"]),
                "access_gate_backoff_count": int(info.get("access_gate_backoff_count", 0)),
                "access_gate_normal_count": int(info.get("access_gate_normal_count", 0)),
                "access_gate_aggressive_count": int(info.get("access_gate_aggressive_count", 0)),
                "access_gate_backoff_ratio": float(info.get("access_gate_backoff_ratio", 0.0)),
                "access_gate_normal_ratio": float(info.get("access_gate_normal_ratio", 0.0)),
                "access_gate_aggressive_ratio": float(info.get("access_gate_aggressive_ratio", 0.0)),
            }
            row.update(beam_selection_diagnostics(observations_by_step[-1], step.actions))
            step_rows.append(row)
        if int(args.resource_log_period) > 0 and global_step % int(args.resource_log_period) == 0:
            snapshot = resource_snapshot()
            snapshot.update({"episode": episode, "slot": slot, "training_step": global_step})
            resource_rows.append(snapshot)
            enforce_resource_limits(snapshot, args)

    summary = env._sim.summarize(episode).as_dict()
    summary.update(env.access_gate_summary())
    return {
        "episode": episode,
        "seed": seed,
        "slots": len(rewards),
        "observations": observations_by_step,
        "actions": actions_by_step,
        "expert_actions": expert_actions_by_step,
        "old_log_probs": torch_module.stack([row.to(policy.device) for row in old_log_probs]),
        "old_mode_log_probs": torch_module.stack([row.to(policy.device) for row in old_mode_log_probs]),
        "old_beam_log_probs": torch_module.stack([row.to(policy.device) for row in old_beam_log_probs]),
        "old_gate_log_probs": torch_module.stack([row.to(policy.device) for row in old_gate_log_probs]),
        "active_beam_mask": torch_module.stack([row.to(policy.device) for row in active_beam_masks]),
        "rewards": torch_module.stack(rewards).to(policy.device),
        "central_features": torch_module.as_tensor(np.stack(central_features), dtype=torch_module.float32, device=policy.device),
        "step_rows": step_rows,
        "summary": summary,
    }


def component_or_zeros(component: Any, fallback: Any) -> Any:
    if component is None:
        return fallback * 0.0
    return component


def active_mask_tensor(step: Any, actions: Sequence[Action], torch_module: Any) -> Any:
    if getattr(step, "active_beam_mask", None) is not None:
        return step.active_beam_mask.bool()
    return torch_module.as_tensor([action.mode != "idle" for action in actions], dtype=torch_module.bool)


def beam_selection_diagnostics(observations: Sequence[dict[str, Any]], actions: Sequence[Action]) -> dict[str, Any]:
    candidate_counts: list[float] = []
    selected_scores: list[float] = []
    score_gaps: list[float] = []
    top1_hits = 0
    top3_hits = 0
    active = 0
    for observation, action in zip(observations, actions, strict=True):
        if action.mode == "idle":
            continue
        active += 1
        score = np.asarray(observation.get("candidate_score", np.zeros(0, dtype=np.float32)), dtype=float)
        mask = np.asarray(observation.get("candidate_mask", np.ones_like(score)), dtype=float) > 0.5
        candidates = np.flatnonzero(mask)
        candidate_counts.append(float(len(candidates)))
        if score.size == 0 or not 0 <= int(action.beam) < score.size or len(candidates) == 0:
            continue
        selected = float(score[int(action.beam)])
        visible_scores = score[candidates]
        best = float(np.max(visible_scores))
        rank = int(1 + np.sum(visible_scores > selected + 1e-12))
        selected_scores.append(selected)
        score_gaps.append(best - selected)
        top1_hits += int(rank == 1)
        top3_hits += int(rank <= 3)
    denom = max(1, active)
    return {
        "active_actions": int(active),
        "beam_candidate_count_mean_active": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "beam_selected_score_mean_active": float(np.mean(selected_scores)) if selected_scores else 0.0,
        "beam_score_gap_mean_active": float(np.mean(score_gaps)) if score_gaps else 0.0,
        "beam_top1_rate_active": float(top1_hits / denom),
        "beam_top3_rate_active": float(top3_hits / denom),
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
    mode_policy_losses = []
    beam_policy_losses = []
    gate_policy_losses = []
    value_losses = []
    entropy_values = []
    approx_kls = []
    clip_fracs = []
    bc_losses = []
    beam_rank_aux_losses = []
    beam_active_fracs = []
    separate_action_loss = bool(getattr(args, "separate_action_loss", False))
    beam_rank_aux_coef = float(getattr(args, "beam_rank_aux_coef", 0.0))

    if centralized:
        if critic is None:
            raise RuntimeError("Centralized critic is required for MAPPO-style training.")
        global_rewards = rewards.mean(dim=1)
        returns = discounted_returns_1d(global_rewards, float(args.gamma), torch_module)
        for _ in range(int(args.ppo_epochs)):
            action_eval = evaluate_action_components(policy, trajectory["observations"], trajectory["actions"])
            log_probs = action_eval["log_probs"]
            entropies = action_eval["entropies"]
            values = critic(trajectory["central_features"])
            advantages = normalize_advantages(returns - values.detach())
            if separate_action_loss:
                policy_loss, component = separated_action_policy_loss(
                    action_eval,
                    trajectory,
                    advantages,
                    float(args.clip_epsilon),
                    torch_module,
                    args,
                )
                mode_policy_losses.append(component["mode_policy_loss"])
                beam_policy_losses.append(component["beam_policy_loss"])
                gate_policy_losses.append(component["gate_policy_loss"])
                clip_fracs.append(component["clip_fraction"])
            else:
                policy_loss, clip_fraction = ppo_component_loss(
                    log_probs,
                    old_log_probs,
                    advantages,
                    float(args.clip_epsilon),
                    torch_module,
                )
                mode_policy_losses.append(0.0)
                beam_policy_losses.append(0.0)
                gate_policy_losses.append(0.0)
                clip_fracs.append(clip_fraction)
            value_loss = functional.mse_loss(values, returns)
            entropy = entropies.mean()
            bc_loss = behavior_cloning_loss(policy, trajectory["observations"], trajectory["expert_actions"], torch_module)
            beam_rank_aux_loss = optional_beam_ranking_aux_loss(
                policy,
                trajectory["observations"],
                torch_module,
                log_probs,
                beam_rank_aux_coef,
                temperature=float(getattr(args, "beam_rank_temperature", 4.0)),
            )
            loss = (
                policy_loss
                + float(args.value_coef) * value_loss
                - float(args.entropy_coef) * entropy
                + float(getattr(args, "expert_bc_weight", 0.0)) * bc_loss
                + beam_rank_aux_coef * beam_rank_aux_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch_module.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], float(args.max_grad_norm))
            optimizer.step()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropy_values.append(float(entropy.item()))
            bc_losses.append(float(bc_loss.item()))
            beam_rank_aux_losses.append(float(beam_rank_aux_loss.item()))
            beam_active_fracs.append(float(trajectory["active_beam_mask"].float().mean().detach().item()))
            approx_kls.append(float((old_log_probs - log_probs).mean().detach().item()))
        return loss_summary(
            policy_losses,
            value_losses,
            entropy_values,
            approx_kls,
            clip_fracs,
            bc_losses,
            mode_policy_losses,
            beam_policy_losses,
            gate_policy_losses,
            beam_rank_aux_losses,
            beam_active_fracs,
        )

    returns = discounted_returns_2d(rewards, float(args.gamma), torch_module)
    for _ in range(int(args.ppo_epochs)):
        action_eval = evaluate_action_components(policy, trajectory["observations"], trajectory["actions"])
        log_probs = action_eval["log_probs"]
        local_values = action_eval["values"]
        entropies = action_eval["entropies"]
        advantages = normalize_advantages(returns - local_values.detach())
        if separate_action_loss:
            policy_loss, component = separated_action_policy_loss(
                action_eval,
                trajectory,
                advantages,
                float(args.clip_epsilon),
                torch_module,
                args,
            )
            mode_policy_losses.append(component["mode_policy_loss"])
            beam_policy_losses.append(component["beam_policy_loss"])
            gate_policy_losses.append(component["gate_policy_loss"])
            clip_fracs.append(component["clip_fraction"])
        else:
            policy_loss, clip_fraction = ppo_component_loss(
                log_probs,
                old_log_probs,
                advantages,
                float(args.clip_epsilon),
                torch_module,
            )
            mode_policy_losses.append(0.0)
            beam_policy_losses.append(0.0)
            gate_policy_losses.append(0.0)
            clip_fracs.append(clip_fraction)
        value_loss = functional.mse_loss(local_values, returns)
        entropy = entropies.mean()
        bc_loss = behavior_cloning_loss(policy, trajectory["observations"], trajectory["expert_actions"], torch_module)
        beam_rank_aux_loss = optional_beam_ranking_aux_loss(
            policy,
            trajectory["observations"],
            torch_module,
            log_probs,
            beam_rank_aux_coef,
            temperature=float(getattr(args, "beam_rank_temperature", 4.0)),
        )
        loss = (
            policy_loss
            + float(args.value_coef) * value_loss
            - float(args.entropy_coef) * entropy
            + float(getattr(args, "expert_bc_weight", 0.0)) * bc_loss
            + beam_rank_aux_coef * beam_rank_aux_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch_module.nn.utils.clip_grad_norm_(policy.parameters(), float(args.max_grad_norm))
        optimizer.step()
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        entropy_values.append(float(entropy.item()))
        bc_losses.append(float(bc_loss.item()))
        beam_rank_aux_losses.append(float(beam_rank_aux_loss.item()))
        beam_active_fracs.append(float(trajectory["active_beam_mask"].float().mean().detach().item()))
        approx_kls.append(float((old_log_probs - log_probs).mean().detach().item()))
    return loss_summary(
        policy_losses,
        value_losses,
        entropy_values,
        approx_kls,
        clip_fracs,
        bc_losses,
        mode_policy_losses,
        beam_policy_losses,
        gate_policy_losses,
        beam_rank_aux_losses,
        beam_active_fracs,
    )


def expert_actions_for_env(env: MarlNeighborDiscoveryEnv, expert_protocol: str) -> list[Action]:
    """Return decentralized rule-expert actions from local simulator memory only."""

    old_protocol = env._sim.protocol
    env._sim.protocol = str(expert_protocol)
    try:
        actions: list[Action] = []
        for node in range(env.n_agents):
            mode = env._sim.select_mode(node, env._slot)
            beam = env._sim.select_beam(node, env._slot, mode, set(), set())
            gate = expert_access_gate_for_env(env, node, mode, int(beam), str(expert_protocol))
            actions.append(Action(mode, int(beam), gate))
        return actions
    finally:
        env._sim.protocol = old_protocol


def expert_access_gate_for_env(
    env: MarlNeighborDiscoveryEnv,
    node: int,
    mode: str,
    beam: int,
    expert_protocol: str,
) -> str:
    """Expose the rule expert's local collision-access decision as a gate label."""

    if expert_protocol not in {"collision_aware_isac", "budgeted_collision_aware_isac"}:
        return ACCESS_NORMAL
    sim = env._sim
    beam_collision = float(sim.collision_fail_count[node, beam])
    beam_success = float(sim.success_count[node, beam])
    beam_fail = float(sim.fail_count[node, beam])
    candidate_pool = sim.isac_candidate_pool(node, env._slot)
    degree = sum(1 for edge in sim.discovered_edges if node in edge)
    degree_need = max(0.0, float(sim.cfg.target_degree - degree)) / max(1.0, float(sim.cfg.target_degree))

    collision_pressure = 0.0
    failure_pressure = 0.0
    candidate_evidence = 0.0
    if len(candidate_pool) > 0:
        success = sim.success_count[node, candidate_pool]
        fail = sim.fail_count[node, candidate_pool]
        collision = sim.collision_fail_count[node, candidate_pool]
        belief = sim.belief[node, candidate_pool]
        collision_pressure = float(np.mean(collision / np.maximum(1.0, success + collision)))
        failure_pressure = float(np.mean(fail / np.maximum(1.0, success + fail)))
        candidate_evidence = float(np.clip(np.mean(belief + 0.15 * np.log1p(success)), 0.0, 1.0))

    if mode == MODE_TX and (beam_collision > 0.0 or collision_pressure >= 0.28 or beam_fail > beam_success + 1.0):
        return ACCESS_BACKOFF
    if (
        mode == MODE_RX
        and degree_need >= 0.35
        and candidate_evidence >= 0.20
        and beam_collision <= beam_success
        and collision_pressure <= 0.25
        and failure_pressure <= 0.65
    ):
        return ACCESS_AGGRESSIVE
    return ACCESS_NORMAL


def evaluate_actions(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    actions_by_step: Sequence[Sequence[Action]],
) -> tuple[Any, Any, Any]:
    action_eval = evaluate_action_components(policy, observations_by_step, actions_by_step)
    return action_eval["log_probs"], action_eval["values"], action_eval["entropies"]


def evaluate_action_components(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    actions_by_step: Sequence[Sequence[Action]],
) -> dict[str, Any]:
    torch = policy.torch
    from torch.distributions import Categorical

    log_prob_rows = []
    mode_log_prob_rows = []
    beam_log_prob_rows = []
    gate_log_prob_rows = []
    value_rows = []
    entropy_rows = []
    mode_entropy_rows = []
    beam_entropy_rows = []
    gate_entropy_rows = []
    active_mask_rows = []
    for observations, actions in zip(observations_by_step, actions_by_step, strict=True):
        step_log_probs = []
        step_mode_log_probs = []
        step_beam_log_probs = []
        step_gate_log_probs = []
        step_values = []
        step_entropies = []
        step_mode_entropies = []
        step_beam_entropies = []
        step_gate_entropies = []
        step_active_masks = []
        for observation, action in zip(observations, actions, strict=True):
            if getattr(policy, "supports_access_gate_action", False) and hasattr(policy, "action_logits_value"):
                mode_logits, beam_logits, gate_logits, value = policy.action_logits_value(observation, hard_mask=True)
            else:
                mode_logits, beam_logits, value = policy.logits_value(observation, hard_mask=True)
                gate_logits = None
            mode_dist = Categorical(logits=mode_logits)
            beam_dist = Categorical(logits=beam_logits)
            mode_idx = MODE_TO_INDEX[action.mode]
            mode_tensor = torch.as_tensor(mode_idx, dtype=torch.long, device=policy.device)
            mode_log_prob = mode_dist.log_prob(mode_tensor)
            beam_log_prob = torch.zeros((), dtype=mode_log_prob.dtype, device=policy.device)
            log_prob = mode_log_prob
            active_beam = action.mode != "idle"
            if action.mode != "idle":
                beam_tensor = torch.as_tensor(int(action.beam), dtype=torch.long, device=policy.device)
                beam_log_prob = beam_dist.log_prob(beam_tensor)
                log_prob = log_prob + beam_log_prob
            mode_entropy = mode_dist.entropy()
            beam_entropy = beam_dist.entropy()
            gate_log_prob = torch.zeros((), dtype=mode_log_prob.dtype, device=policy.device)
            gate_entropy = torch.zeros((), dtype=mode_entropy.dtype, device=policy.device)
            entropy = mode_entropy + beam_entropy
            if gate_logits is not None:
                gate_dist = Categorical(logits=gate_logits)
                gate_idx = ACCESS_GATE_TO_INDEX.get(getattr(action, "access_gate", "normal"), ACCESS_GATE_TO_INDEX["normal"])
                gate_tensor = torch.as_tensor(gate_idx, dtype=torch.long, device=policy.device)
                gate_log_prob = gate_dist.log_prob(gate_tensor)
                gate_entropy = gate_dist.entropy()
                log_prob = log_prob + gate_log_prob
                entropy = entropy + gate_entropy
            step_log_probs.append(log_prob)
            step_mode_log_probs.append(mode_log_prob)
            step_beam_log_probs.append(beam_log_prob)
            step_gate_log_probs.append(gate_log_prob)
            step_values.append(value.squeeze(-1))
            step_entropies.append(entropy)
            step_mode_entropies.append(mode_entropy)
            step_beam_entropies.append(beam_entropy)
            step_gate_entropies.append(gate_entropy)
            step_active_masks.append(torch.as_tensor(active_beam, dtype=torch.bool, device=policy.device))
        log_prob_rows.append(torch.stack(step_log_probs))
        mode_log_prob_rows.append(torch.stack(step_mode_log_probs))
        beam_log_prob_rows.append(torch.stack(step_beam_log_probs))
        gate_log_prob_rows.append(torch.stack(step_gate_log_probs))
        value_rows.append(torch.stack(step_values))
        entropy_rows.append(torch.stack(step_entropies))
        mode_entropy_rows.append(torch.stack(step_mode_entropies))
        beam_entropy_rows.append(torch.stack(step_beam_entropies))
        gate_entropy_rows.append(torch.stack(step_gate_entropies))
        active_mask_rows.append(torch.stack(step_active_masks))
    return {
        "log_probs": torch.stack(log_prob_rows),
        "mode_log_probs": torch.stack(mode_log_prob_rows),
        "beam_log_probs": torch.stack(beam_log_prob_rows),
        "gate_log_probs": torch.stack(gate_log_prob_rows),
        "values": torch.stack(value_rows),
        "entropies": torch.stack(entropy_rows),
        "mode_entropies": torch.stack(mode_entropy_rows),
        "beam_entropies": torch.stack(beam_entropy_rows),
        "gate_entropies": torch.stack(gate_entropy_rows),
        "active_beam_mask": torch.stack(active_mask_rows),
    }


def ppo_component_loss(
    log_probs: Any,
    old_log_probs: Any,
    advantages: Any,
    clip_epsilon: float,
    torch_module: Any,
    mask: Any | None = None,
) -> tuple[Any, float]:
    ratio = torch_module.exp(log_probs - old_log_probs)
    component_advantages = advantages
    while component_advantages.dim() < log_probs.dim():
        component_advantages = component_advantages.unsqueeze(-1)
    clipped_ratio = torch_module.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    loss_values = -torch_module.minimum(
        ratio * component_advantages,
        clipped_ratio * component_advantages,
    )
    clipped = (torch_module.abs(ratio - 1.0) > clip_epsilon).float()
    if mask is not None:
        mask = mask.bool()
        if not bool(mask.any().item()):
            return log_probs.sum() * 0.0, 0.0
        return loss_values[mask].mean(), float(clipped[mask].mean().detach().item())
    return loss_values.mean(), float(clipped.mean().detach().item())


def separated_action_policy_loss(
    action_eval: dict[str, Any],
    trajectory: dict[str, Any],
    advantages: Any,
    clip_epsilon: float,
    torch_module: Any,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    mode_loss, mode_clip = ppo_component_loss(
        action_eval["mode_log_probs"],
        trajectory["old_mode_log_probs"],
        advantages,
        clip_epsilon,
        torch_module,
    )
    active_mask = trajectory["active_beam_mask"].bool()
    beam_loss, beam_clip = ppo_component_loss(
        action_eval["beam_log_probs"],
        trajectory["old_beam_log_probs"],
        advantages,
        clip_epsilon,
        torch_module,
        mask=active_mask,
    )
    gate_loss, gate_clip = ppo_component_loss(
        action_eval["gate_log_probs"],
        trajectory["old_gate_log_probs"],
        advantages,
        clip_epsilon,
        torch_module,
    )
    total = mode_loss + float(getattr(args, "beam_loss_coef", 1.0)) * beam_loss
    total = total + float(getattr(args, "gate_loss_coef", 0.25)) * gate_loss
    return total, {
        "mode_policy_loss": float(mode_loss.detach().item()),
        "beam_policy_loss": float(beam_loss.detach().item()),
        "gate_policy_loss": float(gate_loss.detach().item()),
        "clip_fraction": float(np.mean([mode_clip, beam_clip, gate_clip])),
    }


def optional_beam_ranking_aux_loss(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    torch_module: Any,
    zero_like: Any,
    coefficient: float,
    temperature: float,
) -> Any:
    if float(coefficient) <= 0.0:
        return zero_like.sum() * 0.0
    return beam_ranking_aux_loss(policy, observations_by_step, torch_module, temperature)


def beam_ranking_aux_loss(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    torch_module: Any,
    temperature: float,
) -> Any:
    losses = []
    for observations in observations_by_step:
        if not observations or not all("candidate_score" in observation for observation in observations):
            continue
        if getattr(policy, "supports_access_gate_action", False) and hasattr(policy, "batched_action_logits_value"):
            _mode_logits, beam_logits, _gate_logits, _value = policy.batched_action_logits_value(observations, hard_mask=True)
        else:
            _mode_logits, beam_logits, _value = policy.batched_logits_value(observations, hard_mask=True)
        candidate_score = torch_module.stack(
            [
                torch_module.as_tensor(observation["candidate_score"], dtype=torch_module.float32, device=policy.device)
                for observation in observations
            ],
            dim=0,
        )
        candidate_mask = torch_module.stack(
            [
                torch_module.as_tensor(
                    observation.get("candidate_mask", np.ones_like(observation["candidate_score"])),
                    dtype=torch_module.float32,
                    device=policy.device,
                )
                for observation in observations
            ],
            dim=0,
        ) > 0.5
        valid_count = candidate_mask.sum(dim=-1)
        masked_scores = candidate_score.masked_fill(~candidate_mask, -1.0e9)
        score_max = masked_scores.max(dim=-1).values
        score_min = candidate_score.masked_fill(~candidate_mask, 1.0e9).min(dim=-1).values
        informative = (valid_count >= 2) & ((score_max - score_min) > 1.0e-6)
        if not bool(informative.any().item()):
            continue
        masked_beam_logits = beam_logits.masked_fill(~candidate_mask, -1.0e9)
        log_policy = torch_module.nn.functional.log_softmax(masked_beam_logits, dim=-1)
        target_logits = float(temperature) * (candidate_score - score_max.unsqueeze(-1))
        target_logits = target_logits.masked_fill(~candidate_mask, -1.0e9)
        target = torch_module.nn.functional.softmax(target_logits, dim=-1)
        per_agent = -(target * log_policy).sum(dim=-1)
        losses.append(per_agent[informative])
    if not losses:
        parameter = next(iter(policy.parameters()))
        return parameter.sum() * 0.0
    return torch_module.cat(losses).mean()


def behavior_cloning_loss(
    policy: SharedBeamActorCritic,
    observations_by_step: Sequence[Sequence[dict[str, Any]]],
    expert_actions_by_step: Sequence[Sequence[Action]],
    torch_module: Any,
) -> Any:
    weight = float(getattr(policy, "_expert_bc_weight_cache", 1.0))
    if weight <= 0.0:
        return torch_module.zeros((), dtype=torch_module.float32, device=policy.device)
    losses = []
    from torch.distributions import Categorical

    for observations, actions in zip(observations_by_step, expert_actions_by_step, strict=True):
        for observation, action in zip(observations, actions, strict=True):
            if getattr(policy, "supports_access_gate_action", False) and hasattr(policy, "action_logits_value"):
                mode_logits, beam_logits, gate_logits, _value = policy.action_logits_value(observation, hard_mask=False)
            else:
                mode_logits, beam_logits, _value = policy.logits_value(observation, hard_mask=False)
                gate_logits = None
            mode_dist = Categorical(logits=mode_logits)
            mode_tensor = torch_module.as_tensor(MODE_TO_INDEX[action.mode], dtype=torch_module.long, device=policy.device)
            loss = -mode_dist.log_prob(mode_tensor)
            if action.mode != "idle":
                beam_dist = Categorical(logits=beam_logits)
                beam_tensor = torch_module.as_tensor(int(action.beam), dtype=torch_module.long, device=policy.device)
                loss = loss - beam_dist.log_prob(beam_tensor)
            if gate_logits is not None:
                gate_dist = Categorical(logits=gate_logits)
                gate_idx = ACCESS_GATE_TO_INDEX.get(getattr(action, "access_gate", ACCESS_NORMAL), ACCESS_GATE_TO_INDEX[ACCESS_NORMAL])
                gate_tensor = torch_module.as_tensor(gate_idx, dtype=torch_module.long, device=policy.device)
                loss = loss - 0.10 * gate_dist.log_prob(gate_tensor)
            losses.append(loss)
    if not losses:
        return torch_module.zeros((), dtype=torch_module.float32, device=policy.device)
    return torch_module.stack(losses).mean()


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
    torch_rng_state = torch_module.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    was_training = bool(policy.model.training)
    policy.eval()
    try:
        with torch_module.no_grad():
            eval_modes = (False, True) if bool(args.eval_both) else (bool(stochastic_eval),)
            eval_training_step = int(start_episode) * int(getattr(args, "slots", 1) or 1)
            for mode_index, use_stochastic in enumerate(eval_modes):
                for offset in range(int(args.eval_episodes)):
                    seed = seed_start + 10_000 * mode_index + offset
                    torch_module.manual_seed(seed)
                    np.random.seed(seed)
                    env = MarlNeighborDiscoveryEnv(
                        cfg,
                        seed=seed,
                        protocol=env_protocol,
                        reward_version=str(getattr(args, "reward_version", "legacy")),
                        candidate_source=str(getattr(args, "candidate_source", "default")),
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
                    summary.update(env.access_gate_summary())
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
    finally:
        torch_module.random.set_rng_state(torch_rng_state)
        np.random.set_state(numpy_rng_state)
        if was_training:
            policy.train()
        else:
            policy.eval()
    return rows


def loss_summary(
    policy_losses: list[float],
    value_losses: list[float],
    entropy_values: list[float],
    approx_kls: list[float],
    clip_fracs: list[float],
    bc_losses: list[float] | None = None,
    mode_policy_losses: list[float] | None = None,
    beam_policy_losses: list[float] | None = None,
    gate_policy_losses: list[float] | None = None,
    beam_rank_aux_losses: list[float] | None = None,
    beam_active_fracs: list[float] | None = None,
) -> dict[str, float]:
    return {
        "policy_loss": float(np.mean(policy_losses)),
        "mode_policy_loss": float(np.mean(mode_policy_losses)) if mode_policy_losses else 0.0,
        "beam_policy_loss": float(np.mean(beam_policy_losses)) if beam_policy_losses else 0.0,
        "gate_policy_loss": float(np.mean(gate_policy_losses)) if gate_policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropy_values)),
        "approx_kl": float(np.mean(approx_kls)),
        "clip_fraction": float(np.mean(clip_fracs)),
        "expert_bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
        "beam_rank_aux_loss": float(np.mean(beam_rank_aux_losses)) if beam_rank_aux_losses else 0.0,
        "beam_active_fraction": float(np.mean(beam_active_fracs)) if beam_active_fracs else 0.0,
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
        "training_contract_version": "twc_trainable_v1",
        "feature_flags": resolved_feature_flags(args),
        "env_protocol": resolved_env_protocol(args),
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
    try:
        import torch

        torch_version = str(torch.__version__)
    except ImportError:  # pragma: no cover - training already requires torch
        torch_version = "unavailable"
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
        "candidate_source": str(getattr(args, "candidate_source", "default")),
        "allow_standalone_sense": bool(getattr(args, "allow_standalone_sense", False)),
        "allow_idle": bool(getattr(args, "allow_idle", True)),
        "disabled_modes": list(disabled_modes_from_args(args)),
        "expert_bc_weight": float(getattr(args, "expert_bc_weight", 0.0)),
        "expert_protocol": str(getattr(args, "expert_protocol", "collision_aware_isac")),
        "expert_gate_imitation": bool(float(getattr(args, "expert_bc_weight", 0.0)) > 0.0),
        "separate_action_loss": bool(getattr(args, "separate_action_loss", False)),
        "beam_loss_coef": float(getattr(args, "beam_loss_coef", 1.0)),
        "gate_loss_coef": float(getattr(args, "gate_loss_coef", 0.25)),
        "beam_rank_aux_coef": float(getattr(args, "beam_rank_aux_coef", 0.0)),
        "beam_rank_temperature": float(getattr(args, "beam_rank_temperature", 4.0)),
        "feature_flags": feature_flags,
        "contention_mode_prior": contention_mode_prior_enabled(args),
        "rule_residual_scale": float(getattr(args, "rule_residual_scale", 1.0)),
        "single_rf_chain": int(cfg.rf_chains) == 1,
        "isac_trigger": "tx_piggyback_only",
        "handshake_collision_model": (
            "unique_tx_and_unique_rx"
            if cfg.communication_phy_model == "ideal"
            else "two_phase_hello_ack_sinr_capture"
        ),
        "table_exchange_information": "confirmed_neighbor_positions_and_noisy_anonymous_sensing_reports",
        "communication_phy": communication_phy_manifest(cfg),
        "centralized_training_decentralized_execution": bool(centralized),
        "decentralized_actor_observation": True,
        "centralized_critic_uses_training_state_only": bool(centralized),
        "logs_per_step_reward": True,
        "logs_episode_return": True,
        "torch_threads": int(args.torch_threads),
        "command": [sys.executable, *sys.argv],
        "git_commit": git_revision(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": str(np.__version__),
            "torch": torch_version,
            "platform": platform.platform(),
        },
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


def git_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def communication_phy_manifest(cfg: SimulationConfig) -> dict[str, Any]:
    return {
        "model": cfg.communication_phy_model,
        "carrier_frequency_hz": cfg.communication_carrier_frequency_hz,
        "bandwidth_hz": cfg.communication_bandwidth_hz,
        "tx_power_w": cfg.communication_tx_power_w,
        "noise_figure_db": cfg.communication_noise_figure_db,
        "path_loss_exponent": cfg.communication_path_loss_exponent,
        "reference_distance_m": cfg.communication_reference_distance_m,
        "system_loss_db": cfg.communication_system_loss_db,
        "shadowing_std_db": cfg.communication_shadowing_std_db,
        "rician_k_db": cfg.communication_rician_k_db,
        "sinr_threshold_db": cfg.communication_sinr_threshold_db,
        "antenna_efficiency": cfg.communication_antenna_efficiency,
        "sidelobe_gain_db": cfg.communication_sidelobe_gain_db,
        "fading_enabled": cfg.communication_fading_enabled,
        "shadowing_enabled": cfg.communication_shadowing_enabled,
        "handshake": "two_phase_hello_ack_sinr_capture",
        "channel_seed_policy": "scenario_seed_only",
    }


def main() -> None:
    print(json.dumps(run_training(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
