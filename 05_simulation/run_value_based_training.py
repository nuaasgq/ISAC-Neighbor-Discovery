from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import load_config  # noqa: E402
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv  # noqa: E402
from isac_nd_sim.phy_sensing import SENSING_MEASUREMENT_MODES  # noqa: E402
from isac_nd_sim.value_decomposition import (  # noqa: E402
    VALUE_ACTION_CONTRACTS,
    VALUE_BASED_ALGORITHMS,
    IndependentReplayBuffer,
    JointReplayBuffer,
    JointTransition,
    ValueDecompositionLearner,
    requires_global_training_state,
    select_beam_only_actions,
    select_candidate_score_actions,
    select_local_actions,
)
from run_marl_training import (  # noqa: E402
    central_feature_dim,
    central_state_features,
    copy_observations,
    enforce_resource_limits,
    resource_snapshot,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IDQN/VDN/QMIX under the common ISAC ND contract.")
    parser.add_argument("--config", default="05_simulation/configs/twc_planar_n10_b15_random20.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--algorithm", choices=VALUE_BASED_ALGORITHMS, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=29260711)
    parser.add_argument("--node-count", type=int, default=None)
    parser.add_argument("--azimuth-cells", type=int, default=None)
    parser.add_argument("--elevation-cells", type=int, default=None)
    parser.add_argument("--sensing-measurement-mode", choices=SENSING_MEASUREMENT_MODES, default="noisy_count")
    parser.add_argument("--env-protocol", default="improved_rl_isac_tables")
    parser.add_argument("--candidate-source", default="residual_table")
    parser.add_argument("--reward-version", default="discovery_first")
    parser.add_argument("--reward-scope", choices=("team", "local"), default="team")
    parser.add_argument("--action-contract", choices=VALUE_ACTION_CONTRACTS, default="joint_role_beam")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mixer-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.10)
    parser.add_argument("--epsilon-decay-steps", type=int, default=6000)
    parser.add_argument("--role-uniform-mixture", type=float, default=0.60)
    parser.add_argument("--beam-uniform-mixture", type=float, default=0.80)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resource-log-period", type=int, default=100)
    parser.add_argument("--max-rss-mb", type=float, default=4096.0)
    parser.add_argument("--max-system-memory-percent", type=float, default=85.0)
    parser.add_argument("--eval-only-checkpoint", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = (
        "episodes",
        "slots",
        "batch_size",
        "replay_capacity",
        "warmup_steps",
        "update_interval",
        "target_update_interval",
        "epsilon_decay_steps",
    )
    for name in positive_ints:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if int(args.replay_capacity) < int(args.batch_size):
        raise ValueError("--replay-capacity must be at least --batch-size.")
    if not 0.0 <= float(args.epsilon_end) <= float(args.epsilon_start) <= 1.0:
        raise ValueError("epsilon must satisfy 0 <= end <= start <= 1.")
    if not 0.0 <= float(args.role_uniform_mixture) <= 1.0:
        raise ValueError("--role-uniform-mixture must be in [0, 1].")
    if not 0.0 <= float(args.beam_uniform_mixture) <= 1.0:
        raise ValueError("--beam-uniform-mixture must be in [0, 1].")


def resolved_config(args: argparse.Namespace):
    cfg = load_config(args.config)
    replacements: dict[str, Any] = {
        "episodes": int(args.episodes),
        "slots_per_episode": int(args.slots),
        "seed": int(args.seed),
        "sensing_measurement_mode": str(args.sensing_measurement_mode),
        "rendezvous_observation_enabled": False,
    }
    if args.node_count is not None:
        replacements["n_nodes"] = int(args.node_count)
    if args.azimuth_cells is not None:
        replacements["azimuth_cells"] = int(args.azimuth_cells)
    if args.elevation_cells is not None:
        replacements["elevation_cells"] = int(args.elevation_cells)
    return replace(cfg, **replacements)


def epsilon_at_step(args: argparse.Namespace, global_step: int) -> float:
    fraction = min(1.0, max(0.0, float(global_step) / max(1.0, float(args.epsilon_decay_steps))))
    return float(args.epsilon_start) + fraction * (float(args.epsilon_end) - float(args.epsilon_start))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_launch_provenance() -> dict[str, Any]:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        tracked_worktree_dirty: bool | None = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
    except Exception:
        git_commit = "unknown"
        tracked_worktree_dirty = None
    source_files = {
        "runner": Path(__file__).resolve(),
        "value_decomposition": SRC / "isac_nd_sim" / "value_decomposition.py",
    }
    return {
        "git_commit": git_commit,
        "tracked_worktree_dirty": tracked_worktree_dirty,
        "source_sha256": {name: file_sha256(path) for name, path in source_files.items()},
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for value-based training.") from exc

    validate_args(args)
    launch_provenance = capture_launch_provenance()
    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    replay_rng = np.random.default_rng(int(args.seed) + 17)
    role_rng = np.random.default_rng(int(args.seed) + 19)
    beam_gate_rng = np.random.default_rng(int(args.seed) + 23)
    beam_choice_rng = np.random.default_rng(int(args.seed) + 29)
    cfg = resolved_config(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    learner = ValueDecompositionLearner(
        str(args.algorithm),
        cfg.n_nodes,
        cfg.n_beams,
        central_feature_dim(),
        hidden_dim=int(args.hidden_dim),
        mixer_dim=int(args.mixer_dim),
        learning_rate=float(args.learning_rate),
        gamma=float(args.gamma),
        gradient_clip=float(args.gradient_clip),
        reward_scope=str(args.reward_scope),
        action_contract=str(args.action_contract),
    )
    if args.eval_only_checkpoint is not None:
        checkpoint_path = Path(args.eval_only_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        learner.load_checkpoint_state(checkpoint["learner"])
        eval_rows = evaluate_learner(learner, cfg, args)
        manifest = build_manifest(
            args,
            cfg,
            learner,
            [],
            eval_rows,
            [],
            launch_provenance,
        )
        manifest["scope"] = "common_contract_value_based_eval_only"
        manifest["source_checkpoint"] = str(checkpoint_path)
        manifest["source_checkpoint_sha256"] = file_sha256(checkpoint_path)
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        flush_outputs(output, [], [], eval_rows, [])
        return manifest
    replay = (
        IndependentReplayBuffer(
            cfg.n_nodes,
            int(args.replay_capacity),
            int(args.seed) + 31,
            reward_scope=str(args.reward_scope),
        )
        if str(args.algorithm) == "idqn"
        else JointReplayBuffer(int(args.replay_capacity))
    )
    episode_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    global_step = 0

    for episode in range(int(args.episodes)):
        env = MarlNeighborDiscoveryEnv(
            cfg,
            seed=int(args.seed) + episode,
            protocol=str(args.env_protocol),
            reward_version=str(args.reward_version),
            candidate_source=str(args.candidate_source),
            collect_slot_metrics=False,
            rich_info=False,
        )
        observations, _ = env.reset(seed=int(args.seed) + episode)
        episode_rewards: list[np.ndarray] = []
        episode_losses: list[dict[str, float]] = []
        role_trace = hashlib.blake2b(digest_size=12)
        uses_global_training_state = requires_global_training_state(str(args.algorithm))
        truncated = False
        while not truncated:
            slot = len(episode_rewards)
            state_features = (
                central_state_features(env.training_state(), cfg)
                if uses_global_training_state
                else np.zeros(central_feature_dim(), dtype=np.float32)
            )
            learner.eval()
            with torch.no_grad():
                q_values = learner.q_values(observations).cpu().numpy()
            epsilon = epsilon_at_step(args, global_step)
            role_mixture = max(float(args.role_uniform_mixture), epsilon)
            beam_mixture = max(float(args.beam_uniform_mixture), epsilon)
            if str(args.action_contract) == "beam_only_fixed_role":
                actions, action_indices = select_beam_only_actions(
                    q_values,
                    observations,
                    role_rng,
                    beam_gate_rng,
                    beam_choice_rng,
                    beam_uniform_mixture=beam_mixture,
                )
                role_mixture = 1.0
            else:
                actions, action_indices = select_local_actions(
                    q_values,
                    observations,
                    beam_gate_rng,
                    role_uniform_mixture=role_mixture,
                    beam_uniform_mixture=beam_mixture,
                )
            next_observations, rewards, _terminated, truncated, info = env.step(actions)
            role_trace.update(bytes(1 if action.mode == "tx" else 0 for action in actions))
            next_state_features = (
                central_state_features(env.training_state(), cfg)
                if uses_global_training_state
                else np.zeros(central_feature_dim(), dtype=np.float32)
            )
            replay.append(
                JointTransition(
                    observations=copy_observations(observations),
                    action_indices=action_indices.copy(),
                    rewards=np.asarray(rewards, dtype=np.float32).copy(),
                    next_observations=copy_observations(next_observations),
                    done=bool(truncated),
                    central_state=state_features.copy(),
                    next_central_state=next_state_features.copy(),
                )
            )
            observations = next_observations
            episode_rewards.append(np.asarray(rewards, dtype=np.float32))
            global_step += 1

            if (
                global_step >= int(args.warmup_steps)
                and len(replay) >= int(args.batch_size)
                and global_step % int(args.update_interval) == 0
            ):
                episode_losses.append(learner.update(replay.sample(int(args.batch_size), replay_rng)))
            if global_step % int(args.target_update_interval) == 0:
                learner.sync_targets()
            step_rows.append(
                {
                    "episode": episode,
                    "slot": slot,
                    "training_step": global_step,
                    "algorithm": str(args.algorithm),
                    "scenario_seed": int(args.seed) + episode,
                    "reward_sum": float(np.sum(rewards)),
                    "reward_mean": float(np.mean(rewards)),
                    "episode_cumulative_reward": float(np.sum(episode_rewards)),
                    "new_edges_count": int(info["new_edges_count"]),
                    "discovery_rate": len(env._sim.discovered_edges) / max(1, len(env._sim.first_true_slot)),
                    "epsilon": epsilon,
                    "role_uniform_mixture": (
                        "" if str(args.action_contract) == "beam_only_fixed_role" else role_mixture
                    ),
                    "fixed_tx_probability": (
                        0.5 if str(args.action_contract) == "beam_only_fixed_role" else ""
                    ),
                    "role_policy": (
                        "fixed_iid_bernoulli_0.5"
                        if str(args.action_contract) == "beam_only_fixed_role"
                        else "learned_with_uniform_mixture"
                    ),
                    "beam_uniform_mixture": beam_mixture,
                    "replay_size": len(replay),
                }
            )
            if int(args.resource_log_period) > 0 and global_step % int(args.resource_log_period) == 0:
                snapshot = resource_snapshot()
                snapshot.update({"episode": episode, "slot": slot, "training_step": global_step})
                resource_rows.append(snapshot)
                enforce_resource_limits(snapshot, args)

        reward_array = np.stack(episode_rewards)
        loss_summary = average_losses(episode_losses)
        row = {
            "episode": episode,
            "training_step": global_step,
            "seed": int(args.seed) + episode,
            "scenario_seed": int(args.seed) + episode,
            "algorithm": str(args.algorithm),
            "slots": int(cfg.slots_per_episode),
            "n_nodes": int(cfg.n_nodes),
            "n_beams": int(cfg.n_beams),
            "episode_return_sum": float(reward_array.sum()),
            "episode_return_mean_per_agent": float(reward_array.sum(axis=0).mean()),
            "step_reward_mean": float(reward_array.mean()),
            "epsilon": epsilon_at_step(args, global_step),
            "role_sequence_hash": role_trace.hexdigest(),
            **loss_summary,
            **env._sim.summarize(episode).as_dict(),
        }
        episode_rows.append(row)
        flush_outputs(output, step_rows, episode_rows, [], resource_rows)

    eval_rows = evaluate_learner(learner, cfg, args)
    checkpoint = {
        "learner": learner.checkpoint_state(),
        "args": vars(args),
        "config": str(args.config),
        "training_contract_version": "common_local_residual_value_v1",
    }
    torch.save(checkpoint, output / "final_model.pt")
    manifest = build_manifest(
        args,
        cfg,
        learner,
        episode_rows,
        eval_rows,
        resource_rows,
        launch_provenance,
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    flush_outputs(output, step_rows, episode_rows, eval_rows, resource_rows)
    return manifest


def evaluate_learner(
    learner: ValueDecompositionLearner,
    cfg: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    learner.eval()
    if str(args.action_contract) == "beam_only_fixed_role":
        variants = {
            "pure_learned_beam": ("learned_mix", 1.0, 0.0),
            "learned_beam_random_mix_0.2": ("learned_mix", 1.0, 0.2),
            "learned_beam_random_mix_0.5": ("learned_mix", 1.0, 0.5),
            "learned_beam_random_mix_0.8": ("learned_mix", 1.0, 0.8),
            "random_candidate_beam": ("learned_mix", 1.0, 1.0),
            "candidate_score_argmax": ("score_argmax", 1.0, 0.0),
            "candidate_score_proportional": ("score_proportional", 1.0, 0.0),
        }
    else:
        variants = {
            "matched_support": (
                "joint_learned_mix",
                float(args.role_uniform_mixture),
                float(args.beam_uniform_mixture),
            ),
            "greedy": ("joint_learned_mix", 0.0, 0.0),
            "learned_role_random_beam": ("joint_learned_mix", 0.0, 1.0),
            "random_role_learned_beam": ("joint_learned_mix", 1.0, 0.0),
            "random_uniform": ("joint_learned_mix", 1.0, 1.0),
        }
    for variant, (policy_kind, role_mixture, beam_mixture) in variants.items():
        for eval_episode in range(int(args.eval_episodes)):
            scenario_seed = int(args.seed) + 2_000_000 + eval_episode
            env = MarlNeighborDiscoveryEnv(
                cfg,
                seed=scenario_seed,
                protocol=str(args.env_protocol),
                reward_version=str(args.reward_version),
                candidate_source=str(args.candidate_source),
                collect_slot_metrics=False,
                rich_info=False,
            )
            observations, _ = env.reset(seed=scenario_seed)
            role_rng = np.random.default_rng(scenario_seed + 777)
            beam_gate_rng = np.random.default_rng(scenario_seed + 888)
            beam_choice_rng = np.random.default_rng(scenario_seed + 999)
            rewards_by_step: list[np.ndarray] = []
            role_trace = hashlib.blake2b(digest_size=12)
            beam_trace = hashlib.blake2b(digest_size=12)
            candidate_trace = hashlib.blake2b(digest_size=12)
            truncated = False
            while not truncated:
                with torch.no_grad():
                    q_values = learner.q_values(observations).cpu().numpy()
                for observation in observations:
                    candidate_trace.update(
                        np.packbits(
                            np.asarray(observation["candidate_mask"], dtype=np.uint8),
                            bitorder="little",
                        ).tobytes()
                    )
                if policy_kind == "learned_mix":
                    actions, _indices = select_beam_only_actions(
                        q_values,
                        observations,
                        role_rng,
                        beam_gate_rng,
                        beam_choice_rng,
                        beam_uniform_mixture=beam_mixture,
                    )
                elif policy_kind in ("score_argmax", "score_proportional"):
                    actions, _indices = select_candidate_score_actions(
                        observations,
                        role_rng,
                        beam_choice_rng,
                        selection="argmax" if policy_kind == "score_argmax" else "proportional",
                    )
                else:
                    actions, _indices = select_local_actions(
                        q_values,
                        observations,
                        beam_gate_rng,
                        role_uniform_mixture=role_mixture,
                        beam_uniform_mixture=beam_mixture,
                    )
                role_trace.update(bytes(1 if action.mode == "tx" else 0 for action in actions))
                beam_trace.update(
                    np.asarray([action.beam for action in actions], dtype=np.uint16).tobytes()
                )
                observations, rewards, _terminated, truncated, _info = env.step(actions)
                rewards_by_step.append(np.asarray(rewards, dtype=np.float32))
            reward_array = np.stack(rewards_by_step)
            rows.append(
                {
                    "phase": "eval_value_policy_ablation",
                    "policy_variant": variant,
                    "execution_policy_kind": policy_kind,
                    "eval_episode": eval_episode,
                    "episode": int(args.episodes) + eval_episode,
                    "seed": scenario_seed,
                    "scenario_seed": scenario_seed,
                    "algorithm": str(args.algorithm),
                    "env_protocol": str(args.env_protocol),
                    "role_uniform_mixture": (
                        "" if str(args.action_contract) == "beam_only_fixed_role" else role_mixture
                    ),
                    "fixed_tx_probability": (
                        0.5 if str(args.action_contract) == "beam_only_fixed_role" else ""
                    ),
                    "role_sequence_hash": role_trace.hexdigest(),
                    "beam_sequence_hash": beam_trace.hexdigest(),
                    "candidate_mask_sequence_hash": candidate_trace.hexdigest(),
                    "beam_uniform_mixture": beam_mixture,
                    "candidate_source": str(args.candidate_source),
                    "action_contract": str(args.action_contract),
                    "training_beam_random_floor": float(args.beam_uniform_mixture),
                    "training_epsilon_start": float(args.epsilon_start),
                    "training_epsilon_end": float(args.epsilon_end),
                    "training_epsilon_decay_steps": int(args.epsilon_decay_steps),
                    "episode_return_sum": float(reward_array.sum()),
                    "episode_return_mean_per_agent": float(reward_array.sum(axis=0).mean()),
                    "step_reward_mean": float(reward_array.mean()),
                    **env._sim.summarize(int(args.episodes) + eval_episode).as_dict(),
                }
            )
    return rows


def average_losses(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ("td_loss", "q_mean", "target_mean", "gradient_norm")
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


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


def build_manifest(
    args: argparse.Namespace,
    cfg: Any,
    learner: ValueDecompositionLearner,
    episode_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    launch_provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "common_contract_value_based_screen",
        "algorithm": str(args.algorithm),
        "config": str(args.config),
        "output": str(args.output),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "eval_episodes": int(args.eval_episodes),
        "slots_per_episode": int(cfg.slots_per_episode),
        "node_count": int(cfg.n_nodes),
        "beam_count": int(cfg.n_beams),
        "env_protocol": str(args.env_protocol),
        "candidate_source": str(args.candidate_source),
        "reward_version": str(args.reward_version),
        "reward_scope": str(args.reward_scope),
        "action_contract": str(args.action_contract),
        "training_contract_version": "common_local_residual_value_v1",
        "decentralized_execution": True,
        "centralized_training": learner.centralized_training,
        "independent_agent_parameters": learner.independent_parameters,
        "independent_agent_replay": learner.independent_parameters,
        "independent_agent_optimizers": learner.independent_parameters,
        "parameter_sharing": not learner.independent_parameters,
        "actor_observation": "local_anonymous_isac_and_table_state_only",
        "joint_action_controller": False,
        "action_space": (
            "per_agent_beam_only"
            if str(args.action_contract) == "beam_only_fixed_role"
            else "per_agent_tx_rx_times_beam"
        ),
        "role_policy": (
            "fixed_iid_bernoulli_0.5_not_learned"
            if str(args.action_contract) == "beam_only_fixed_role"
            else "learned_tx_rx"
        ),
        "stochastic_support": {
            "role_uniform_mixture": (
                None
                if str(args.action_contract) == "beam_only_fixed_role"
                else float(args.role_uniform_mixture)
            ),
            "fixed_tx_probability": (
                0.5 if str(args.action_contract) == "beam_only_fixed_role" else None
            ),
            "role_learned": str(args.action_contract) != "beam_only_fixed_role",
            "beam_uniform_mixture": float(args.beam_uniform_mixture),
            "beam_randomization_domain": "uniform_within_residual_candidate_mask",
            "beam_gate_rng_separate_from_choice_rng": True,
            "fixed_rng_draws_per_agent_slot": True,
            "training_epsilon_start": float(args.epsilon_start),
            "training_epsilon_end": float(args.epsilon_end),
            "training_epsilon_decay_steps": int(args.epsilon_decay_steps),
        },
        "replay": {
            "capacity": int(args.replay_capacity),
            "batch_size": int(args.batch_size),
            "warmup_steps": int(args.warmup_steps),
            "update_interval": int(args.update_interval),
            "target_update_interval": int(args.target_update_interval),
            "double_q_target": True,
        },
        "sensing_measurement": {
            "mode": cfg.sensing_measurement_mode,
            "identity_exposed_before_handshake": False,
            "common_protocol_interface": True,
        },
        "git_commit": launch_provenance["git_commit"],
        "provenance_capture": "process_launch",
        "tracked_worktree_dirty": launch_provenance["tracked_worktree_dirty"],
        "source_sha256": launch_provenance["source_sha256"],
        "command": list(sys.argv),
        "peak_rss_mb": max(
            (float(row["rss_mb"]) for row in resource_rows if isinstance(row.get("rss_mb"), (int, float))),
            default=0.0,
        ),
        "final_train": episode_rows[-1] if episode_rows else {},
        "final_eval": eval_rows[-1] if eval_rows else {},
        "files": (
            [
                "step_rewards.csv",
                "episode_metrics.csv",
                "eval_episode_metrics.csv",
                "resource_log.csv",
                "manifest.json",
            ]
            if args.eval_only_checkpoint is not None
            else [
                "step_rewards.csv",
                "episode_metrics.csv",
                "eval_episode_metrics.csv",
                "resource_log.csv",
                "final_model.pt",
                "manifest.json",
            ]
        ),
    }


def main() -> None:
    args = parse_args()
    manifest = run_training(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
