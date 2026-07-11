from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import load_config  # noqa: E402
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv  # noqa: E402
from isac_nd_sim.value_decomposition import (  # noqa: E402
    ValueDecompositionLearner,
    select_beam_only_actions,
    select_candidate_score_actions,
)
from run_marl_training import build_policy, write_rows  # noqa: E402


BeamScorer = Callable[[Sequence[dict[str, Any]]], np.ndarray]


class RecurrentPolicyScorer:
    def __init__(self, policy: Any, torch_module: Any):
        self.policy = policy
        self.torch = torch_module

    def reset(self, n_agents: int) -> None:
        self.policy.reset_recurrent_state(n_agents)

    def __call__(self, observations: Sequence[dict[str, Any]]) -> np.ndarray:
        with self.torch.no_grad():
            _mode_logits, beam_logits, _value = self.policy.advance_recurrent_logits(
                observations,
                hard_mask=True,
            )
            return beam_logits.cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common seven-policy beam-only checkpoint evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", choices=("value", "mappo"), required=True)
    parser.add_argument("--algorithm-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="05_simulation/configs/twc_planar_n10_b15_random20.yaml")
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=29260711)
    parser.add_argument("--env-protocol", default="improved_rl_isac_tables")
    parser.add_argument("--candidate-source", default="residual_table")
    parser.add_argument("--reward-version", default="discovery_first")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_value_scorer(checkpoint: dict[str, Any]) -> tuple[BeamScorer, dict[str, Any]]:
    import torch

    state = checkpoint["learner"]
    if state["action_contract"] != "beam_only_fixed_role":
        raise ValueError("Value checkpoint is not beam_only_fixed_role.")
    learner = ValueDecompositionLearner(
        state["algorithm"],
        state["n_agents"],
        state["n_beams"],
        state["state_dim"],
        hidden_dim=state["hidden_dim"],
        mixer_dim=int(state.get("mixer_dim", checkpoint.get("args", {}).get("mixer_dim", 32))),
        reward_scope=state["reward_scope"],
        action_contract=state["action_contract"],
    )
    learner.load_checkpoint_state(state)
    learner.eval()

    def scorer(observations: Sequence[dict[str, Any]]) -> np.ndarray:
        with torch.no_grad():
            return learner.q_values(observations).cpu().numpy()

    training_args = checkpoint.get("args", {})
    return scorer, {
        "checkpoint_algorithm": state["algorithm"],
        "action_contract": state["action_contract"],
        "n_agents": int(state["n_agents"]),
        "n_beams": int(state["n_beams"]),
        "training_exploration": {
            "type": "epsilon_greedy_candidate_uniform",
            "epsilon_start": training_args.get("epsilon_start"),
            "epsilon_end": training_args.get("epsilon_end"),
            "epsilon_decay_steps": training_args.get("epsilon_decay_steps"),
            "persistent_random_floor": training_args.get("beam_uniform_mixture"),
        },
    }


def load_mappo_scorer(checkpoint: dict[str, Any]) -> tuple[BeamScorer, dict[str, Any]]:
    import torch

    checkpoint_args = SimpleNamespace(**checkpoint["args"])
    action_contract = str(
        checkpoint.get("action_contract", checkpoint["args"].get("action_contract", ""))
    )
    if action_contract != "beam_only_fixed_role":
        raise ValueError("MAPPO checkpoint is not beam_only_fixed_role.")
    feature_flags = checkpoint["feature_flags"]
    n_beams = int(checkpoint["config"]["azimuth_cells"]) * int(
        checkpoint["config"]["elevation_cells"]
    )
    policy = build_policy(
        str(checkpoint_args.network),
        n_beams,
        hidden_dim=int(checkpoint_args.hidden_dim),
        device="cpu",
        use_candidate_mask=bool(feature_flags["candidate_mask"]),
        use_candidate_score=bool(feature_flags["candidate_score"]),
        use_topology_deficit=bool(feature_flags["topology_deficit"]),
        use_rule_residual=False,
        rule_residual_scale=1.0,
        use_contention_mode_prior=False,
        use_rendezvous_adapter=False,
        use_residual_measurement_features=bool(checkpoint_args.residual_measurement_features),
        role_probability_floor=0.0,
        beam_uniform_mixture=0.0,
        disabled_modes=("sense", "idle"),
        action_contract=action_contract,
        azimuth_cells=int(checkpoint["config"]["azimuth_cells"]),
        elevation_cells=int(checkpoint["config"]["elevation_cells"]),
    )
    policy.model.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    if hasattr(policy, "advance_recurrent_logits"):
        scorer: BeamScorer = RecurrentPolicyScorer(policy, torch)
    else:
        def scorer(observations: Sequence[dict[str, Any]]) -> np.ndarray:
            with torch.no_grad():
                _mode_logits, beam_logits, _value = policy.batched_logits_value(
                    observations,
                    hard_mask=True,
                )
                return beam_logits.cpu().numpy()

    return scorer, {
        "checkpoint_algorithm": checkpoint["algorithm"],
        "action_contract": action_contract,
        "n_agents": int(checkpoint["config"]["n_nodes"]),
        "n_beams": n_beams,
        "architecture_version": checkpoint.get("architecture_version", "legacy_unspecified"),
        "training_contract_version": checkpoint.get(
            "training_contract_version", "legacy_unspecified"
        ),
        "actor_network": checkpoint.get("actor_network", checkpoint["args"].get("network")),
        "critic_network": checkpoint.get("critic_network", "pooled"),
        "actor_state_reset": (
            "zero_each_episode" if hasattr(policy, "reset_recurrent_state") else "stateless"
        ),
        "training_exploration": {
            "type": "categorical_policy_sampling_plus_entropy",
            "entropy_coef": float(checkpoint_args.entropy_coef),
            "persistent_random_floor": float(checkpoint_args.beam_uniform_mixture),
        },
    }


def load_scorer(args: argparse.Namespace) -> tuple[BeamScorer, dict[str, Any]]:
    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.checkpoint_kind == "value":
        return load_value_scorer(checkpoint)
    return load_mappo_scorer(checkpoint)


def evaluate(args: argparse.Namespace, scorer: BeamScorer) -> list[dict[str, Any]]:
    cfg = replace(
        load_config(args.config),
        slots_per_episode=int(args.slots),
        episodes=int(args.eval_episodes),
        seed=int(args.seed),
        rendezvous_observation_enabled=False,
    )
    variants = {
        "pure_learned_beam": ("learned_mix", 0.0),
        "learned_beam_random_mix_0.2": ("learned_mix", 0.2),
        "learned_beam_random_mix_0.5": ("learned_mix", 0.5),
        "learned_beam_random_mix_0.8": ("learned_mix", 0.8),
        "random_candidate_beam": ("learned_mix", 1.0),
        "candidate_score_argmax": ("score_argmax", 0.0),
        "candidate_score_proportional": ("score_proportional", 0.0),
    }
    rows: list[dict[str, Any]] = []
    for variant, (policy_kind, beam_mixture) in variants.items():
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
            if hasattr(scorer, "reset"):
                scorer.reset(env.n_agents)
            role_rng = np.random.default_rng(scenario_seed + 777)
            gate_rng = np.random.default_rng(scenario_seed + 888)
            choice_rng = np.random.default_rng(scenario_seed + 999)
            role_trace = hashlib.blake2b(digest_size=12)
            beam_trace = hashlib.blake2b(digest_size=12)
            candidate_trace = hashlib.blake2b(digest_size=12)
            rewards_by_step: list[np.ndarray] = []
            truncated = False
            while not truncated:
                for observation in observations:
                    candidate_trace.update(
                        np.packbits(
                            np.asarray(observation["candidate_mask"], dtype=np.uint8),
                            bitorder="little",
                        ).tobytes()
                    )
                if policy_kind == "learned_mix":
                    actions, _indices = select_beam_only_actions(
                        scorer(observations),
                        observations,
                        role_rng,
                        gate_rng,
                        choice_rng,
                        beam_uniform_mixture=beam_mixture,
                    )
                else:
                    actions, _indices = select_candidate_score_actions(
                        observations,
                        role_rng,
                        choice_rng,
                        selection="argmax" if policy_kind == "score_argmax" else "proportional",
                    )
                role_trace.update(bytes(1 if action.mode == "tx" else 0 for action in actions))
                beam_trace.update(
                    np.asarray([action.beam for action in actions], dtype=np.uint16).tobytes()
                )
                observations, rewards, _terminated, truncated, _info = env.step(actions)
                rewards_by_step.append(np.asarray(rewards, dtype=np.float32))
            rewards_array = np.stack(rewards_by_step)
            row = {
                "phase": "eval_common_beam_only",
                "policy_variant": variant,
                "execution_policy_kind": policy_kind,
                "eval_episode": eval_episode,
                "scenario_seed": scenario_seed,
                "algorithm": str(args.algorithm_label),
                "checkpoint_kind": str(args.checkpoint_kind),
                "action_contract": "beam_only_fixed_role",
                "role_policy": "fixed_iid_bernoulli_0.5_not_learned",
                "fixed_tx_probability": 0.5,
                "beam_uniform_mixture": beam_mixture,
                "candidate_source": str(args.candidate_source),
                "role_sequence_hash": role_trace.hexdigest(),
                "beam_sequence_hash": beam_trace.hexdigest(),
                "candidate_mask_sequence_hash": candidate_trace.hexdigest(),
                "episode_return_sum": float(rewards_array.sum()),
                "episode_return_mean_per_agent": float(rewards_array.sum(axis=0).mean()),
                "step_reward_mean": float(rewards_array.mean()),
                **env._sim.summarize(eval_episode).as_dict(),
            }
            rows.append(row)
    return rows


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    if int(args.torch_threads) > 0:
        import torch

        torch.set_num_threads(int(args.torch_threads))
    scorer, checkpoint_metadata = load_scorer(args)
    cfg = load_config(args.config)
    if int(cfg.n_nodes) != int(checkpoint_metadata["n_agents"]):
        raise ValueError(
            f"Evaluation n_nodes={cfg.n_nodes} does not match checkpoint "
            f"n_agents={checkpoint_metadata['n_agents']}."
        )
    if int(cfg.n_beams) != int(checkpoint_metadata["n_beams"]):
        raise ValueError(
            f"Evaluation n_beams={cfg.n_beams} does not match checkpoint "
            f"n_beams={checkpoint_metadata['n_beams']}."
        )
    expected_label = str(checkpoint_metadata["checkpoint_algorithm"])
    allowed_labels = {expected_label}
    if args.checkpoint_kind == "mappo":
        allowed_labels.update({"mappo", "beam_only_mappo"})
    if str(args.algorithm_label) not in allowed_labels:
        raise ValueError(
            f"algorithm-label={args.algorithm_label!r} is incompatible with "
            f"checkpoint algorithm={expected_label!r}."
        )
    rows = evaluate(args, scorer)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "eval_episode_metrics.csv", rows)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "common_seven_policy_beam_only_evaluation",
        "algorithm": str(args.algorithm_label),
        "checkpoint_kind": str(args.checkpoint_kind),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "git_commit": git_revision(),
        "config": str(args.config),
        "slots": int(args.slots),
        "eval_episodes": int(args.eval_episodes),
        "seed": int(args.seed),
        "env_protocol": str(args.env_protocol),
        "candidate_source": str(args.candidate_source),
        "reward_version": str(args.reward_version),
        "variants": sorted({row["policy_variant"] for row in rows}),
        **checkpoint_metadata,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
