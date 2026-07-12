from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "05_simulation" / "src"
OFFPOLICY_ROOT = ROOT / "05_simulation" / "third_party" / "marlbenchmark_offpolicy"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(OFFPOLICY_ROOT) not in sys.path:
    sys.path.insert(0, str(OFFPOLICY_ROOT))

from isac_nd_sim.config import load_config
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv
from isac_nd_sim.neural_contention_actor_critic import observations_to_batched_contention_tensors
from isac_nd_sim.simulator import Action


OFFICIAL_REPOSITORY = "https://github.com/marlbenchmark/off-policy.git"
OFFICIAL_COMMIT = "41fd5eb46d12df2847e1c2e29842997ff2c24998"
POLICY_ID = "policy_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the pinned marlbenchmark/off-policy MATD3 on the ISAC ND environment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "05_simulation" / "configs" / "n10_b15_static_ideal_isac.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--slots", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=69260713)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--train-interval", type=int, default=100)
    parser.add_argument("--epsilon-anneal-steps", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


def official_components():
    if not (OFFPOLICY_ROOT / "offpolicy").is_dir():
        raise RuntimeError(
            "MATD3 submodule is missing. Run: git submodule update --init --recursive"
        )
    from gym.spaces import Box
    from offpolicy.algorithms.matd3.algorithm.MATD3Policy import MATD3Policy
    from offpolicy.algorithms.matd3.matd3 import MATD3
    from offpolicy.config import get_config
    from offpolicy.utils.mlp_buffer import MlpReplayBuffer
    from offpolicy.utils.util import MultiDiscrete

    return Box, MATD3Policy, MATD3, get_config, MlpReplayBuffer, MultiDiscrete


def flat_local_observations(
    observations: list[dict[str, Any]],
    n_beams: int,
) -> np.ndarray:
    """Flatten the same clean local direct-ISAC fields used by the MAPPO actor."""

    import torch

    tensors = observations_to_batched_contention_tensors(
        observations,
        torch.device("cpu"),
        torch,
        n_beams,
        measurement_feature_set="direct",
    )
    beam_features = tensors["beam_features"].clone()
    beam_features[..., 4] = 0.0
    beam_features[..., 5] = tensors["beam_collision_norm"]
    beam_features[..., 7] = 0.0
    contention = tensors["contention_state"].clone()
    contention[..., 6:9] = 0.0
    local_summary = tensors["local_summary"].clone()
    local_summary[..., 3] = 0.0
    candidate_stats = tensors["candidate_stats"].clone()
    candidate_stats[..., 0:3] = 0.0
    topology_deficit = torch.zeros_like(tensors["topology_deficit"])
    fields = (
        beam_features.flatten(start_dim=1),
        tensors["self_state"],
        local_summary,
        tensors["last_mode"],
        tensors["last_beam"],
        contention,
        candidate_stats,
        topology_deficit,
    )
    return torch.cat(fields, dim=-1).cpu().numpy().astype(np.float32, copy=False)


def decode_multidiscrete_actions(encoded: Any) -> list[Action]:
    values = encoded.detach().cpu().numpy() if hasattr(encoded, "detach") else np.asarray(encoded)
    if values.ndim != 2 or values.shape[1] != 26:
        raise ValueError(f"Expected [agents,26] one-hot actions, received {values.shape}.")
    modes = np.argmax(values[:, :2], axis=1)
    beams = np.argmax(values[:, 2:], axis=1)
    return [Action("tx" if int(mode) == 0 else "rx", int(beam)) for mode, beam in zip(modes, beams)]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in known:
                fieldnames.append(field)
                known.add(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_official_stack(args: argparse.Namespace, obs_dim: int, n_agents: int):
    import torch

    Box, MATD3Policy, MATD3, get_config, MlpReplayBuffer, MultiDiscrete = official_components()
    official_args = get_config().parse_args([])
    overrides = {
        "use_same_share_obs": True,
        "use_popart": False,
        "use_per": False,
        "use_value_active_masks": False,
        "use_huber_loss": False,
        "use_soft_update": True,
        "hidden_size": int(args.hidden_dim),
        "layer_N": 1,
        "use_ReLU": True,
        "use_feature_normalization": True,
        "use_orthogonal": True,
        "gain": 0.01,
        "lr": float(args.learning_rate),
        "opti_eps": 1.0e-5,
        "weight_decay": 0.0,
        "tau": 0.005,
        "epsilon_start": 1.0,
        "epsilon_finish": 0.05,
        "epsilon_anneal_time": int(args.epsilon_anneal_steps),
        "target_action_noise_std": 0.2,
        "max_grad_norm": 10.0,
        "gamma": 0.99,
    }
    for name, value in overrides.items():
        setattr(official_args, name, value)
    device = torch.device("cpu")
    action_dim = 2 + 24
    central_obs_dim = n_agents * obs_dim
    central_act_dim = n_agents * action_dim
    obs_space = Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
    share_obs_space = Box(-np.inf, np.inf, (central_obs_dim,), dtype=np.float32)
    action_space = MultiDiscrete([[0, 1], [0, 23]])
    policy_config = {
        "obs_space": obs_space,
        "share_obs_space": share_obs_space,
        "act_space": action_space,
        "cent_obs_dim": central_obs_dim,
        "cent_act_dim": central_act_dim,
    }
    policy = MATD3Policy(
        {"args": official_args, "device": device},
        policy_config,
    )
    policies = {POLICY_ID: policy}
    mapping = lambda _agent: POLICY_ID
    trainer = MATD3(official_args, n_agents, policies, mapping, device=device)
    policy_info = {POLICY_ID: policy_config}
    policy_agents = {POLICY_ID: list(range(n_agents))}
    buffer = MlpReplayBuffer(
        policy_info,
        policy_agents,
        int(args.buffer_size),
        True,
        False,
    )
    return official_args, policy, trainer, buffer


def insert_transition(
    buffer: Any,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_observations: np.ndarray,
    done: bool,
) -> None:
    n_agents = observations.shape[0]
    team_reward = float(np.mean(rewards))
    reward_rows = np.full((1, n_agents, 1), team_reward, dtype=np.float32)
    dones = np.full((1, n_agents, 1), float(done), dtype=np.float32)
    dones_env = np.asarray([[float(done)]], dtype=np.float32)
    valid = np.ones_like(dones, dtype=np.float32)
    payload = {
        POLICY_ID: observations[None, ...],
    }
    buffer.insert(
        1,
        payload,
        {POLICY_ID: observations.reshape(1, -1)},
        {POLICY_ID: actions[None, ...]},
        {POLICY_ID: reward_rows},
        {POLICY_ID: next_observations[None, ...]},
        {POLICY_ID: next_observations.reshape(1, -1)},
        {POLICY_ID: dones},
        {POLICY_ID: dones_env},
        {POLICY_ID: valid},
        {POLICY_ID: None},
        {POLICY_ID: None},
    )


def scalar(value: Any) -> float:
    return float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)


def evaluate_policy(
    cfg: Any,
    policy: Any,
    episodes: int,
    seed_start: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policy.actor.eval()
    for episode in range(episodes):
        env = MarlNeighborDiscoveryEnv(
            cfg,
            seed=seed_start + episode,
            protocol="improved_rl_isac_tables",
            reward_version="discovery_first",
            candidate_source="default",
        )
        observations, _info = env.reset(seed=seed_start + episode)
        returns = np.zeros(env.n_agents, dtype=np.float32)
        truncated = False
        while not truncated:
            local = flat_local_observations(observations, env.n_beams)
            encoded, _epsilon = policy.get_actions(local, explore=False)
            actions = decode_multidiscrete_actions(encoded)
            observations, rewards, _terminated, truncated, _info = env.step(actions)
            returns += np.asarray(rewards, dtype=np.float32)
        summary = env._sim.summarize(episode).as_dict()
        summary.update(
            {
                "phase": "eval_stochastic",
                "eval_episode": episode,
                "scenario_seed": seed_start + episode,
                "algorithm": "matd3_reference_discrete",
                "episode_return_mean_per_agent": float(np.mean(returns)),
            }
        )
        rows.append(summary)
    return rows


def main() -> None:
    import torch

    args = parse_args()
    torch.set_num_threads(int(args.torch_threads))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    cfg = load_config(args.config)
    if cfg.n_nodes != 10 or cfg.n_beams != 24 or int(args.slots) != 300:
        raise ValueError("The reference MATD3 contract requires N=10, B=24, and 300 slots.")
    cfg = cfg.__class__(**{**cfg.__dict__, "episodes": int(args.episodes), "slots_per_episode": int(args.slots)})
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    probe_env = MarlNeighborDiscoveryEnv(
        cfg,
        protocol="improved_rl_isac_tables",
        reward_version="discovery_first",
        candidate_source="default",
    )
    probe_observations, _ = probe_env.reset(seed=int(args.seed))
    observation_dim = flat_local_observations(probe_observations, cfg.n_beams).shape[1]
    official_args, policy, trainer, buffer = build_official_stack(args, observation_dim, cfg.n_nodes)

    episode_rows: list[dict[str, Any]] = []
    global_step = 0
    latest_train_info: dict[str, float] = {}
    for episode in range(int(args.episodes)):
        scenario_seed = int(args.seed) + episode
        env = MarlNeighborDiscoveryEnv(
            cfg,
            seed=scenario_seed,
            protocol="improved_rl_isac_tables",
            reward_version="discovery_first",
            candidate_source="default",
        )
        observations, _info = env.reset(seed=scenario_seed)
        returns = np.zeros(env.n_agents, dtype=np.float32)
        epsilon = 1.0
        truncated = False
        while not truncated:
            local = flat_local_observations(observations, env.n_beams)
            if global_step < int(args.warmup_steps):
                encoded = policy.get_random_actions(local)
            else:
                encoded, epsilon = policy.get_actions(local, t_env=global_step, explore=True)
            encoded_array = (
                encoded.detach().cpu().numpy() if hasattr(encoded, "detach") else np.asarray(encoded)
            ).astype(np.float32, copy=False)
            actions = decode_multidiscrete_actions(encoded_array)
            next_observations, rewards, _terminated, truncated, _info = env.step(actions)
            next_local = flat_local_observations(next_observations, env.n_beams)
            insert_transition(
                buffer,
                local,
                encoded_array,
                np.asarray(rewards, dtype=np.float32),
                next_local,
                bool(truncated),
            )
            returns += np.asarray(rewards, dtype=np.float32)
            observations = next_observations
            global_step += 1
            if (
                len(buffer) >= int(args.batch_size)
                and global_step >= int(args.warmup_steps)
                and global_step % int(args.train_interval) == 0
            ):
                trainer.prep_training()
                train_info, _priorities, _indices = trainer.shared_train_policy_on_batch(
                    POLICY_ID,
                    buffer.sample(int(args.batch_size)),
                )
                trainer.num_updates[POLICY_ID] += 1
                if bool(train_info.get("update_actor", False)):
                    policy.soft_target_updates()
                latest_train_info = {
                    key: scalar(value)
                    for key, value in train_info.items()
                    if key != "update_actor"
                }
                latest_train_info["update_actor"] = float(bool(train_info.get("update_actor", False)))
                trainer.prep_rollout()
        summary = env._sim.summarize(episode).as_dict()
        row = {
            "episode": episode,
            "training_step": global_step,
            "scenario_seed": scenario_seed,
            "algorithm": "matd3_reference_discrete",
            "episode_return_mean_per_agent": float(np.mean(returns)),
            "epsilon": float(epsilon or 0.0),
            "buffer_size": len(buffer),
            "official_update_count": int(trainer.num_updates[POLICY_ID]),
            **latest_train_info,
            **summary,
        }
        episode_rows.append(row)
        write_rows(output / "episode_metrics.csv", episode_rows)

    eval_rows = evaluate_policy(
        cfg,
        policy,
        int(args.eval_episodes),
        int(args.seed) + 2_010_000,
    )
    write_rows(output / "eval_episode_metrics.csv", eval_rows)
    checkpoint = {
        "actor_state_dict": policy.actor.state_dict(),
        "critic_state_dict": policy.critic.state_dict(),
        "target_actor_state_dict": policy.target_actor.state_dict(),
        "target_critic_state_dict": policy.target_critic.state_dict(),
        "training_step": global_step,
        "official_update_count": int(trainer.num_updates[POLICY_ID]),
        "observation_dim": observation_dim,
        "action_heads": [2, 24],
        "seed": int(args.seed),
    }
    torch.save(checkpoint, output / "final_model.pt")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "matd3_reference_discrete",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_classes": [
            "offpolicy.algorithms.matd3.algorithm.MATD3Policy.MATD3Policy",
            "offpolicy.algorithms.matd3.matd3.MATD3",
            "offpolicy.utils.mlp_buffer.MlpReplayBuffer",
        ],
        "adapter_changes": [
            "custom_isac_nd_environment_runner",
            "multidiscrete_heads_tx_rx_2_and_beam_24",
            "clean_local_direct_isac_flattening",
            "shared_team_mean_reward_for_official_shared_policy_trainer",
            "explicit_num_updates_increment_for_delayed_actor_schedule",
        ],
        "actor_global_state_access": False,
        "critic_input": "concatenated_local_actor_observations_training_only",
        "candidate_mask": False,
        "candidate_score": False,
        "expert_actions": False,
        "antisymmetric_role_head": False,
        "measurement_prediction_auxiliary": False,
        "node_count": cfg.n_nodes,
        "beam_count": cfg.n_beams,
        "slots_per_episode": int(args.slots),
        "episodes": int(args.episodes),
        "eval_episodes": int(args.eval_episodes),
        "training_steps": global_step,
        "seed": int(args.seed),
        "output": str(output),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
