from __future__ import annotations

import argparse
import csv
import json
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
from isac_nd_sim.marl_env import MODE_NAMES, MODE_TO_INDEX, MarlNeighborDiscoveryEnv  # noqa: E402
from isac_nd_sim.mobility import step_states  # noqa: E402
from isac_nd_sim.neural_shared_actor_critic import (  # noqa: E402
    SharedBeamActorCritic,
)
from isac_nd_sim.simulator import Action, NeighborDiscoverySimulator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a rule-expert-assisted behavior-cloning probe for the shared actor-critic MARL path."
    )
    parser.add_argument("--config", default="05_simulation/configs/mvp.yaml")
    parser.add_argument("--output", default="05_simulation/results_raw/actor_critic_imitation_probe")
    parser.add_argument("--bc-episodes", type=int, default=8)
    parser.add_argument("--rl-episodes", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Sample from the learned policy during evaluation instead of using argmax actions.",
    )
    parser.add_argument(
        "--eval-both",
        action="store_true",
        help="Evaluate the same trained policy with both deterministic and stochastic action selection.",
    )
    parser.add_argument("--slots", type=int, default=40)
    parser.add_argument("--node-count", type=int, default=None)
    parser.add_argument("--azimuth-cells", type=int, default=None)
    parser.add_argument("--elevation-cells", type=int, default=None)
    parser.add_argument("--communication-range", type=float, default=None)
    parser.add_argument("--sensing-range", type=float, default=None)
    parser.add_argument("--false-alarm-rate", type=float, default=None)
    parser.add_argument("--miss-detection-rate", type=float, default=None)
    parser.add_argument("--angular-cell-offset-std", type=float, default=None)
    parser.add_argument("--sensing-period-slots", type=int, default=None)
    parser.add_argument("--env-protocol", default="isac_structured_marl")
    parser.add_argument("--expert-protocol", default="improved_rl_isac")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--bc-coef", type=float, default=1.0)
    parser.add_argument("--beam-bc-coef", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.25)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--candidate-mask", action="store_true", help="Use local ISAC candidate masks during execution.")
    parser.add_argument("--candidate-score", action="store_true", help="Expose local candidate scores to the beam encoder.")
    parser.add_argument("--topology-deficit", action="store_true", help="Expose local discovered-degree deficit to the actor.")
    parser.add_argument("--rule-residual", action="store_true", help="Add local rule priors as residual logits.")
    parser.add_argument("--rule-residual-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--expert-seed-offset", type=int, default=7919)
    return parser.parse_args()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for run_actor_critic_imitation_probe.py") from exc

    validate_args(args)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = override_config(load_config(args.config), args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    policy = SharedBeamActorCritic(
        cfg.n_beams,
        hidden_dim=int(args.hidden_dim),
        device="cpu",
        use_candidate_mask=bool(args.candidate_mask),
        use_candidate_score=bool(args.candidate_score),
        use_topology_deficit=bool(args.topology_deficit),
        use_rule_residual=bool(args.rule_residual),
        rule_residual_scale=float(args.rule_residual_scale),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.learning_rate))
    history: list[dict[str, Any]] = []

    for episode in range(int(args.bc_episodes)):
        row = run_behavior_cloning_episode(
            cfg=cfg,
            policy=policy,
            optimizer=optimizer,
            torch_module=torch,
            functional=F,
            args=args,
            episode=episode,
        )
        history.append(row)

    for episode in range(int(args.rl_episodes)):
        row = run_actor_critic_episode(
            cfg=cfg,
            policy=policy,
            optimizer=optimizer,
            torch_module=torch,
            functional=F,
            args=args,
            episode=episode,
            seed=int(args.seed) + 100_000 + episode,
        )
        history.append(row)

    eval_rows: list[dict[str, Any]] = []
    eval_modes = (False, True) if bool(args.eval_both) else (bool(args.stochastic_eval),)
    for stochastic_eval in eval_modes:
        eval_rows.extend(
            evaluate_policy(
                cfg=cfg,
                policy=policy,
                torch_module=torch,
                args=args,
                start_episode=len(history) + len(eval_rows),
                seed_start=int(args.seed) + 200_000,
                stochastic_eval=stochastic_eval,
            )
        )
    history.extend(eval_rows)

    write_rows(output / "training_history.csv", history)
    manifest = build_manifest(args, cfg, history)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_args(args: argparse.Namespace) -> None:
    if int(args.bc_episodes) < 0 or int(args.rl_episodes) < 0 or int(args.eval_episodes) < 0:
        raise ValueError("Episode counts must be non-negative.")
    if int(args.bc_episodes) + int(args.rl_episodes) + int(args.eval_episodes) <= 0:
        raise ValueError("At least one BC, RL, or evaluation episode is required.")
    if int(args.slots) <= 0:
        raise ValueError("--slots must be positive.")


def override_config(config: SimulationConfig, args: argparse.Namespace) -> SimulationConfig:
    replacements: dict[str, Any] = {
        "slots_per_episode": int(args.slots),
        "episodes": max(1, int(args.bc_episodes) + int(args.rl_episodes) + int(args.eval_episodes)),
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
    return replace(config, **replacements)


def run_behavior_cloning_episode(
    cfg: SimulationConfig,
    policy: SharedBeamActorCritic,
    optimizer: Any,
    torch_module: Any,
    functional: Any,
    args: argparse.Namespace,
    episode: int,
) -> dict[str, Any]:
    seed = int(args.seed) + episode
    env = MarlNeighborDiscoveryEnv(cfg, seed=seed, protocol=str(args.env_protocol))
    observations, _info = env.reset(seed=seed)
    expert = NeighborDiscoverySimulator(
        cfg,
        protocol=str(args.expert_protocol),
        seed=seed + int(args.expert_seed_offset),
        scenario_seed=seed,
    )
    expert.reset()

    policy.train()
    mode_losses = []
    beam_losses = []
    entropies = []
    values = []
    rewards = []
    mode_correct = 0
    beam_correct = 0
    active_beam_correct = 0
    total_actions = 0
    active_actions = 0

    for slot in range(cfg.slots_per_episode):
        expert_actions, true_edges = select_expert_actions(expert, slot)
        loss_parts = behavior_cloning_loss(
            policy=policy,
            observations=observations,
            expert_actions=expert_actions,
            torch_module=torch_module,
            functional=functional,
        )
        mode_losses.append(loss_parts["mode_loss"])
        beam_losses.append(loss_parts["beam_loss"])
        entropies.append(loss_parts["entropy"])
        values.append(loss_parts["values"])
        mode_correct += int(loss_parts["mode_correct"])
        beam_correct += int(loss_parts["beam_correct"])
        active_beam_correct += int(loss_parts["active_beam_correct"])
        total_actions += int(loss_parts["total_actions"])
        active_actions += int(loss_parts["active_actions"])

        observations, reward, _terminated, truncated, _step_info = env.step(expert_actions)
        rewards.append(torch_module.as_tensor(reward, dtype=torch_module.float32))
        finish_expert_step(expert, slot, true_edges, expert_actions, episode)
        if truncated:
            break

    returns = discounted_returns(rewards, float(args.gamma), torch_module)
    value_tensor = torch_module.stack(values)
    mode_loss = torch_module.stack(mode_losses).mean()
    beam_loss = torch_module.stack(beam_losses).mean()
    entropy_bonus = torch_module.stack(entropies).mean()
    value_loss = functional.mse_loss(value_tensor, returns)
    loss = (
        float(args.bc_coef) * mode_loss
        + float(args.bc_coef) * float(args.beam_bc_coef) * beam_loss
        + float(args.value_coef) * value_loss
        - float(args.entropy_coef) * entropy_bonus
    )

    optimizer.zero_grad()
    loss.backward()
    torch_module.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    env_summary = env._sim.summarize(episode).as_dict()
    expert_summary = expert.summarize(episode).as_dict()
    return {
        "phase": "bc",
        "episode": episode,
        "seed": seed,
        "env_protocol": str(args.env_protocol),
        "expert_protocol": str(args.expert_protocol),
        "slots": cfg.slots_per_episode,
        "loss": float(loss.item()),
        "bc_mode_loss": float(mode_loss.item()),
        "bc_beam_loss": float(beam_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_bonus.item()),
        "reward_mean": float(torch_module.stack(rewards).mean().item()),
        "mode_accuracy": safe_divide(mode_correct, total_actions),
        "beam_accuracy": safe_divide(beam_correct, total_actions),
        "active_beam_accuracy": safe_divide(active_beam_correct, active_actions),
        **prefixed_summary("env", env_summary),
        **prefixed_summary("expert", expert_summary),
    }


def behavior_cloning_loss(
    policy: SharedBeamActorCritic,
    observations: Sequence[dict],
    expert_actions: Sequence[Action],
    torch_module: Any,
    functional: Any,
) -> dict[str, Any]:
    mode_logits_rows = []
    beam_logits_rows = []
    value_rows = []
    entropy_rows = []
    mode_targets = []
    beam_targets = []
    active_mask = []

    for observation, action in zip(observations, expert_actions, strict=True):
        mode_logits, beam_logits, value = policy.logits_value(observation, hard_mask=False)
        mode_logits_rows.append(mode_logits)
        beam_logits_rows.append(beam_logits)
        value_rows.append(value.squeeze(-1))
        entropy_rows.append(categorical_entropy(mode_logits, torch_module) + categorical_entropy(beam_logits, torch_module))
        mode_targets.append(MODE_TO_INDEX[action.mode])
        beam_targets.append(int(action.beam))
        active_mask.append(action.mode != "idle")

    mode_logits_tensor = torch_module.stack(mode_logits_rows)
    beam_logits_tensor = torch_module.stack(beam_logits_rows)
    value_tensor = torch_module.stack(value_rows)
    mode_target_tensor = torch_module.as_tensor(mode_targets, dtype=torch_module.long, device=policy.device)
    beam_target_tensor = torch_module.as_tensor(beam_targets, dtype=torch_module.long, device=policy.device)
    active_tensor = torch_module.as_tensor(active_mask, dtype=torch_module.bool, device=policy.device)

    mode_loss = functional.cross_entropy(mode_logits_tensor, mode_target_tensor)
    if bool(active_tensor.any().item()):
        beam_loss = functional.cross_entropy(beam_logits_tensor[active_tensor], beam_target_tensor[active_tensor])
    else:
        beam_loss = beam_logits_tensor.sum() * 0.0

    predicted_modes = torch_module.argmax(mode_logits_tensor, dim=1)
    predicted_beams = torch_module.argmax(beam_logits_tensor, dim=1)
    mode_correct = int((predicted_modes == mode_target_tensor).sum().item())
    beam_correct = int((predicted_beams == beam_target_tensor).sum().item())
    active_beam_correct = (
        int((predicted_beams[active_tensor] == beam_target_tensor[active_tensor]).sum().item())
        if bool(active_tensor.any().item())
        else 0
    )
    return {
        "mode_loss": mode_loss,
        "beam_loss": beam_loss,
        "values": value_tensor,
        "entropy": torch_module.stack(entropy_rows).mean(),
        "mode_correct": mode_correct,
        "beam_correct": beam_correct,
        "active_beam_correct": active_beam_correct,
        "total_actions": len(expert_actions),
        "active_actions": int(active_tensor.sum().item()),
    }


def run_actor_critic_episode(
    cfg: SimulationConfig,
    policy: SharedBeamActorCritic,
    optimizer: Any,
    torch_module: Any,
    functional: Any,
    args: argparse.Namespace,
    episode: int,
    seed: int,
) -> dict[str, Any]:
    env = MarlNeighborDiscoveryEnv(cfg, seed=seed, protocol=str(args.env_protocol))
    observations, _info = env.reset(seed=seed)
    log_probs = []
    values = []
    entropies = []
    rewards = []
    truncated = False
    policy.train()

    while not truncated:
        step = policy.act(observations, deterministic=False)
        observations, reward, _terminated, truncated, _step_info = env.step(step.actions)
        log_probs.append(step.log_probs)
        values.append(step.values)
        entropies.append(step.entropies)
        rewards.append(torch_module.as_tensor(reward, dtype=torch_module.float32))

    returns = discounted_returns(rewards, float(args.gamma), torch_module)
    value_tensor = torch_module.stack(values)
    log_prob_tensor = torch_module.stack(log_probs)
    entropy_tensor = torch_module.stack(entropies)
    advantages = returns - value_tensor.detach()
    policy_loss = -(log_prob_tensor * advantages).mean()
    value_loss = functional.mse_loss(value_tensor, returns)
    entropy_bonus = entropy_tensor.mean()
    loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch_module.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    env_summary = env._sim.summarize(episode).as_dict()
    return {
        "phase": "rl",
        "episode": episode,
        "seed": seed,
        "env_protocol": str(args.env_protocol),
        "expert_protocol": str(args.expert_protocol),
        "slots": cfg.slots_per_episode,
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy_bonus.item()),
        "reward_mean": float(torch_module.stack(rewards).mean().item()),
        **prefixed_summary("env", env_summary),
    }


def evaluate_policy(
    cfg: SimulationConfig,
    policy: SharedBeamActorCritic,
    torch_module: Any,
    args: argparse.Namespace,
    start_episode: int,
    seed_start: int,
    stochastic_eval: bool,
) -> list[dict[str, Any]]:
    rows = []
    policy.eval()
    with torch_module.no_grad():
        for offset in range(int(args.eval_episodes)):
            seed = seed_start + offset
            env = MarlNeighborDiscoveryEnv(cfg, seed=seed, protocol=str(args.env_protocol))
            observations, _info = env.reset(seed=seed)
            truncated = False
            rewards = []
            while not truncated:
                step = policy.act(observations, deterministic=not stochastic_eval)
                observations, reward, _terminated, truncated, _step_info = env.step(step.actions)
                rewards.append(torch_module.as_tensor(reward, dtype=torch_module.float32))
            summary = env._sim.summarize(start_episode + offset).as_dict()
            rows.append(
                {
                    "phase": "eval_stochastic" if stochastic_eval else "eval_deterministic",
                    "episode": start_episode + offset,
                    "seed": seed,
                    "env_protocol": str(args.env_protocol),
                    "expert_protocol": str(args.expert_protocol),
                    "slots": cfg.slots_per_episode,
                    "reward_mean": float(torch_module.stack(rewards).mean().item()),
                    **prefixed_summary("env", summary),
                }
            )
    return rows


def select_expert_actions(
    expert: NeighborDiscoverySimulator,
    slot: int,
) -> tuple[list[Action], set[tuple[int, int]]]:
    expert._beam_matrix_cache = None
    expert._distance_matrix_cache = None
    true_edges = expert.true_edges(expert.cfg.communication_range_m)
    for edge in true_edges:
        expert.first_true_slot.setdefault(edge, slot)
    expert.age += 1.0
    expert.belief *= expert.cfg.confidence_decay
    expert._candidate_pool_cache.clear()
    return expert.select_actions(slot, true_edges), true_edges


def finish_expert_step(
    expert: NeighborDiscoverySimulator,
    slot: int,
    true_edges: set[tuple[int, int]],
    actions: list[Action],
    episode: int,
) -> None:
    expert.snapshot_pre_sensing_candidates(slot)
    expert.update_action_counts(actions, slot)
    expert.update_empty_scan_counts(actions, true_edges)
    expert.update_sensing(actions, slot)
    expert._candidate_pool_cache.clear()
    new_edges = expert.resolve_discoveries(slot, actions, true_edges)
    if expert.cfg.slot_metric_period > 0 and slot % expert.cfg.slot_metric_period == 0:
        expert.per_slot_rows.append(expert.slot_metrics(episode, slot, true_edges, new_edges))
    step_states(
        expert.states,
        expert.cfg.area_size_m,
        expert.cfg.mobility,
        expert.cfg.slot_duration_s,
        slot,
        expert.mobility_rng,
    )
    expert._beam_matrix_cache = None
    expert._distance_matrix_cache = None


def categorical_entropy(logits: Any, torch_module: Any) -> Any:
    probabilities = torch_module.softmax(logits, dim=-1)
    log_probabilities = torch_module.log_softmax(logits, dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1)


def discounted_returns(rewards: Sequence[Any], gamma: float, torch_module: Any) -> Any:
    running = torch_module.zeros_like(rewards[-1])
    returns = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch_module.stack(returns)


def prefixed_summary(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "discovery_rate",
        "discovered_edges",
        "true_edges_seen",
        "lambda2",
        "empty_scan_ratio",
        "collision_count",
        "scan_actions",
        "tx_actions",
        "rx_actions",
        "sense_actions",
        "idle_actions",
        "piggyback_sense_actions",
        "lcc_ratio",
        "mean_delay_censored",
        "p95_delay_censored",
    ]
    return {f"{prefix}_{key}": summary[key] for key in keys if key in summary}


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(args: argparse.Namespace, cfg: SimulationConfig, history: list[dict[str, Any]]) -> dict[str, Any]:
    final_bc = next((row for row in reversed(history) if row.get("phase") == "bc"), {})
    final_eval = next((row for row in reversed(history) if str(row.get("phase", "")).startswith("eval_")), {})
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "shared_actor_critic_imitation_probe",
        "scope": "method_probe_not_paper_result",
        "interpretation_warning": (
            "This run is a behavior-cloning and wiring probe for the MARL stack. "
            "It should not be reported as a paper main result without longer training, "
            "multi-seed comparison, and frozen baselines."
        ),
        "config": str(args.config),
        "output": str(args.output),
        "expert_protocol": str(args.expert_protocol),
        "env_protocol": str(args.env_protocol),
        "bc_episodes": int(args.bc_episodes),
        "rl_episodes": int(args.rl_episodes),
        "eval_episodes": int(args.eval_episodes),
        "slots": int(cfg.slots_per_episode),
        "node_count": int(cfg.n_nodes),
        "beam_count": int(cfg.n_beams),
        "hidden_dim": int(args.hidden_dim),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "stochastic_eval": bool(args.stochastic_eval),
        "eval_both": bool(args.eval_both),
        "candidate_mask": bool(args.candidate_mask),
        "candidate_score": bool(args.candidate_score),
        "topology_deficit": bool(args.topology_deficit),
        "rule_residual": bool(args.rule_residual),
        "rule_residual_scale": float(args.rule_residual_scale),
        "uses_marl_env": True,
        "uses_shared_actor_critic": True,
        "teacher_forced_env_final": final_bc,
        "student_eval_final": final_eval,
        "files": ["training_history.csv", "manifest.json"],
    }


def main() -> None:
    print(json.dumps(run_probe(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
