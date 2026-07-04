from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "05_simulation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isac_nd_sim.config import load_config  # noqa: E402
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv  # noqa: E402
from isac_nd_sim.neural_shared_actor_critic import SharedBeamActorCritic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small shared actor-critic MARL probe.")
    parser.add_argument("--config", default="05_simulation/configs/mvp.yaml")
    parser.add_argument("--output", default="05_simulation/results_raw/actor_critic_probe")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--slots", type=int, default=40)
    parser.add_argument("--node-count", type=int, default=None)
    parser.add_argument("--azimuth-cells", type=int, default=None)
    parser.add_argument("--elevation-cells", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for run_actor_critic_probe.py") from exc

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    cfg = load_config(args.config)
    replacements: dict[str, Any] = {"slots_per_episode": int(args.slots), "episodes": 1, "seed": int(args.seed)}
    if args.node_count is not None:
        replacements["n_nodes"] = int(args.node_count)
    if args.azimuth_cells is not None:
        replacements["azimuth_cells"] = int(args.azimuth_cells)
    if args.elevation_cells is not None:
        replacements["elevation_cells"] = int(args.elevation_cells)
    cfg = replace(cfg, **replacements)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    policy = SharedBeamActorCritic(cfg.n_beams, hidden_dim=int(args.hidden_dim), device="cpu")
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.learning_rate))
    history: list[dict[str, Any]] = []

    for episode in range(int(args.episodes)):
        env = MarlNeighborDiscoveryEnv(cfg, seed=int(args.seed) + episode)
        observations, _info = env.reset(seed=int(args.seed) + episode)
        log_probs = []
        values = []
        entropies = []
        rewards = []
        truncated = False
        while not truncated:
            step = policy.act(observations, deterministic=False)
            observations, reward, _terminated, truncated, info = env.step(step.actions)
            log_probs.append(step.log_probs)
            values.append(step.values)
            entropies.append(step.entropies)
            rewards.append(torch.as_tensor(reward, dtype=torch.float32))

        returns = discounted_returns(rewards, float(args.gamma), torch)
        value_tensor = torch.stack(values)
        log_prob_tensor = torch.stack(log_probs)
        entropy_tensor = torch.stack(entropies)
        advantages = returns - value_tensor.detach()
        policy_loss = -(log_prob_tensor * advantages).mean()
        value_loss = F.mse_loss(value_tensor, returns)
        entropy_bonus = entropy_tensor.mean()
        loss = policy_loss + float(args.value_coef) * value_loss - float(args.entropy_coef) * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        summary = env._sim.summarize(episode).as_dict()
        history.append(
            {
                "episode": episode,
                "seed": int(args.seed) + episode,
                "reward_mean": float(torch.stack(rewards).mean().item()),
                "return_mean": float(returns.mean().item()),
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy_bonus.item()),
                "discovery_rate": float(summary["discovery_rate"]),
                "lambda2": float(summary["lambda2"]),
                "empty_scan_ratio": float(summary["empty_scan_ratio"]),
                "collision_count": float(summary["collision_count"]),
                "discovered_edges": float(summary["discovered_edges"]),
                "scan_actions": float(summary["scan_actions"]),
            }
        )

    write_rows(output / "training_history.csv", history)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "episodes": int(args.episodes),
        "slots": int(args.slots),
        "node_count": int(cfg.n_nodes),
        "beam_count": int(cfg.n_beams),
        "algorithm": "shared_actor_critic_probe",
        "decentralized_actor_observation": True,
        "centralized_training_value_baseline": True,
        "final": history[-1] if history else {},
        "files": ["training_history.csv", "manifest.json"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def discounted_returns(rewards, gamma: float, torch_module):
    running = torch_module.zeros_like(rewards[-1])
    returns = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch_module.stack(returns)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print(json.dumps(run_probe(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
