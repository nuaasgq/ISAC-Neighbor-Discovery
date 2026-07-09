from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_SRC = REPO_ROOT / "05_simulation" / "src"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))
if str(REPO_ROOT / "05_simulation") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "05_simulation"))

from isac_nd_sim.config import load_config  # noqa: E402
from isac_nd_sim.marl_env import MarlNeighborDiscoveryEnv  # noqa: E402
from isac_nd_sim.mobility import step_states  # noqa: E402
from isac_nd_sim.simulator import MODE_IDLE, MODE_RX, MODE_TX, Action, NeighborDiscoverySimulator  # noqa: E402
from run_marl_evaluate import (  # noqa: E402
    apply_eval_feature_overrides,
    build_policy,
    checkpoint_feature_flags_from_args,
    disabled_modes_from_flag,
    load_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize beam candidate dynamics for MARL and Wang-style baselines.")
    parser.add_argument("--config", default="05_simulation/configs/wang2025_reproduction_smoke.yaml")
    parser.add_argument(
        "--checkpoint",
        default=(
            "05_simulation/results_raw/marl_campaign/"
            "wang2025_aligned_n10_fixedhandshake_20260709/"
            "marl_wang_isac_tables_discovery_first/train/final_model.pt"
        ),
    )
    parser.add_argument("--output", default="06_analysis/paper_tables/candidate_dynamics_20260709")
    parser.add_argument("--seed", type=int, default=2026084001)
    parser.add_argument("--node-count", type=int, default=10)
    parser.add_argument("--slots", type=int, default=200)
    parser.add_argument("--node-id", type=int, default=0)
    parser.add_argument("--env-protocol", default="wang2025_isac_tables")
    parser.add_argument("--wang-policy", default="wang2025_isac_tables")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--disable-candidate-mask", action="store_true")
    parser.add_argument("--disable-candidate-score", action="store_true")
    parser.add_argument("--disable-topology-deficit", action="store_true")
    parser.add_argument("--disable-rule-residual", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (REPO_ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        load_config(REPO_ROOT / args.config),
        n_nodes=int(args.node_count),
        slots_per_episode=int(args.slots),
        episodes=1,
        seed=int(args.seed),
    )
    if not 0 <= int(args.node_id) < int(cfg.n_nodes):
        raise ValueError(f"--node-id must be in [0, {cfg.n_nodes}).")

    ours = run_ours_trace(args, cfg)
    wang = run_wang_trace(args, cfg)
    write_trace(output, ours, wang, cfg.n_beams, int(args.node_id))
    figure_paths = write_figures(output, ours, wang, cfg.n_beams, int(args.node_id))
    manifest = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "output": str(output),
        "seed": int(args.seed),
        "node_count": int(cfg.n_nodes),
        "slots": int(cfg.slots_per_episode),
        "node_id": int(args.node_id),
        "beam_count": int(cfg.n_beams),
        "env_protocol": str(args.env_protocol),
        "wang_policy": str(args.wang_policy),
        "figures": [str(path) for path in figure_paths],
        "csv_files": [
            "candidate_counts.csv",
            f"node{int(args.node_id)}_candidate_masks.csv",
            f"node{int(args.node_id)}_beliefs.csv",
            "actions_and_discoveries.csv",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def run_ours_trace(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to run MARL candidate diagnostics.") from exc

    if int(args.torch_threads) > 0:
        torch.set_num_threads(int(args.torch_threads))
    checkpoint = load_checkpoint(REPO_ROOT / args.checkpoint, torch)
    train_args = checkpoint.get("args", {})
    flags = apply_eval_feature_overrides(checkpoint_feature_flags_from_args(train_args), args)
    policy = build_policy(
        str(train_args.get("network", "contention_shared")),
        cfg.n_beams,
        hidden_dim=int(train_args.get("hidden_dim", 128)),
        device="cpu",
        use_candidate_mask=flags["candidate_mask"],
        use_candidate_score=flags["candidate_score"],
        use_topology_deficit=flags["topology_deficit"],
        use_rule_residual=flags["rule_residual"],
        rule_residual_scale=float(train_args.get("rule_residual_scale", 1.0)),
        disabled_modes=disabled_modes_from_flag(True),
    )
    policy.model.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    env = MarlNeighborDiscoveryEnv(
        cfg,
        protocol=str(args.env_protocol),
        seed=int(args.seed),
        reward_version=str(train_args.get("reward_version", "discovery_first")),
    )
    observations, _info = env.reset(seed=int(args.seed))
    trace = init_trace("ours_marl", cfg)
    rng_seed_base = int(args.seed)
    for slot in range(int(cfg.slots_per_episode)):
        record_ours_pre_action(trace, slot, env, observations)
        torch.manual_seed(rng_seed_base + slot)
        np.random.seed((rng_seed_base + slot) % (2**32 - 1))
        step = policy.act(observations, deterministic=bool(args.deterministic))
        actions = step.actions
        prev_edges = len(env._sim.discovered_edges)
        observations, _reward, _terminated, _truncated, info = env.step(actions)
        record_common_post_action(trace, slot, actions, len(env._sim.discovered_edges) - prev_edges, info)
    return trace


def run_wang_trace(args: argparse.Namespace, cfg: Any) -> dict[str, Any]:
    sim = NeighborDiscoverySimulator(cfg, protocol=str(args.env_protocol), seed=int(args.seed), scenario_seed=int(args.seed))
    sim.reset()
    trace = init_trace("wang2025", cfg)
    for slot in range(int(cfg.slots_per_episode)):
        record_wang_pre_action(trace, slot, sim)
        true_comm_edges = sim.true_edges(cfg.communication_range_m)
        for edge in true_comm_edges:
            sim.first_true_slot.setdefault(edge, slot)
        sim.age += 1.0
        sim.belief *= cfg.confidence_decay
        sim._candidate_pool_cache.clear()

        old_protocol = sim.protocol
        try:
            sim.protocol = str(args.wang_policy)
            actions = sim.select_actions(slot, true_comm_edges)
        finally:
            sim.protocol = old_protocol

        prev_edges = len(sim.discovered_edges)
        sim.snapshot_pre_sensing_candidates(slot)
        sim.update_action_counts(actions, slot)
        sim.update_empty_scan_counts(actions, true_comm_edges, slot)
        sim.update_sensing(actions, slot)
        sim._candidate_pool_cache.clear()
        new_edges = sim.resolve_discoveries(slot, actions, true_comm_edges)
        info = {
            "discovered_edges_count": len(sim.discovered_edges),
            "new_edges_count": len(new_edges),
            "empty_scan_ratio": sim.empty_scans / max(1, sim.scan_actions),
            "sensing_detection_rate": sim.sensing_detection_count / max(1, sim.sensing_target_observations),
        }
        record_common_post_action(trace, slot, actions, len(sim.discovered_edges) - prev_edges, info)
        step_states(sim.states, cfg.area_size_m, cfg.mobility, cfg.slot_duration_s, slot, sim.mobility_rng)
        sim._beam_matrix_cache = None
        sim._distance_matrix_cache = None
        sim._sensing_profile_cache.clear()
    return trace


def init_trace(method: str, cfg: Any) -> dict[str, Any]:
    slots = int(cfg.slots_per_episode)
    nodes = int(cfg.n_nodes)
    beams = int(cfg.n_beams)
    return {
        "method": method,
        "candidate_mask": np.zeros((slots, nodes, beams), dtype=np.float32),
        "candidate_score": np.zeros((slots, nodes, beams), dtype=np.float32),
        "belief": np.zeros((slots, nodes, beams), dtype=np.float32),
        "wang_active": np.zeros((slots, nodes, beams), dtype=np.float32),
        "wang_positive": np.zeros((slots, nodes, beams), dtype=np.float32),
        "counts": [],
        "actions": [],
    }


def record_ours_pre_action(trace: dict[str, Any], slot: int, env: MarlNeighborDiscoveryEnv, observations: list[dict]) -> None:
    for node, obs in enumerate(observations):
        candidate = np.asarray(obs["candidate_mask"], dtype=np.float32)
        score = np.asarray(obs["candidate_score"], dtype=np.float32)
        belief = np.asarray(obs["beam_belief"], dtype=np.float32)
        trace["candidate_mask"][slot, node] = candidate
        trace["candidate_score"][slot, node] = score
        trace["belief"][slot, node] = belief
        trace["counts"].append(
            {
                "method": trace["method"],
                "slot": slot,
                "node": node,
                "candidate_count": int(np.count_nonzero(candidate > 0.5)),
                "active_count": "",
                "positive_count": "",
                "belief_mean": float(np.mean(belief)),
                "belief_max": float(np.max(belief)),
                "score_max": float(np.max(score)),
                "discovered_edges_before": len(env._sim.discovered_edges),
            }
        )


def record_wang_pre_action(trace: dict[str, Any], slot: int, sim: NeighborDiscoverySimulator) -> None:
    for node in range(sim.cfg.n_nodes):
        belief = sim.belief[node].astype(np.float32, copy=False)
        active = ((sim.empty_beam_count[node] < 1.0) | (sim.success_count[node] > 0.05)).astype(np.float32)
        positive = (active > 0.5) & ((sim.belief[node] >= 0.55) | (sim.success_count[node] > 0.05))
        trace["belief"][slot, node] = belief
        trace["wang_active"][slot, node] = active
        trace["wang_positive"][slot, node] = positive.astype(np.float32)
        trace["counts"].append(
            {
                "method": trace["method"],
                "slot": slot,
                "node": node,
                "candidate_count": "",
                "active_count": int(np.count_nonzero(active > 0.5)),
                "positive_count": int(np.count_nonzero(positive)),
                "belief_mean": float(np.mean(belief)),
                "belief_max": float(np.max(belief)),
                "score_max": "",
                "discovered_edges_before": len(sim.discovered_edges),
            }
        )


def record_common_post_action(trace: dict[str, Any], slot: int, actions: list[Action], new_edges: int, info: dict[str, Any]) -> None:
    for node, action in enumerate(actions):
        trace["actions"].append(
            {
                "method": trace["method"],
                "slot": slot,
                "node": node,
                "mode": action.mode,
                "beam": int(action.beam),
                "new_edges_this_slot": int(new_edges),
                "discovered_edges_after": int(info.get("discovered_edges_count", "")),
                "empty_scan_ratio": info.get("empty_scan_ratio", ""),
                "sensing_detection_rate": info.get("sensing_detection_rate", ""),
            }
        )


def write_trace(output: Path, ours: dict[str, Any], wang: dict[str, Any], n_beams: int, node_id: int) -> None:
    write_rows(output / "candidate_counts.csv", ours["counts"] + wang["counts"])
    write_rows(output / "actions_and_discoveries.csv", ours["actions"] + wang["actions"])
    mask_rows = []
    belief_rows = []
    for trace in (ours, wang):
        for slot in range(trace["belief"].shape[0]):
            masks = {
                "ours_candidate": trace["candidate_mask"][slot, node_id],
                "wang_active": trace["wang_active"][slot, node_id],
                "wang_positive": trace["wang_positive"][slot, node_id],
            }
            for mask_name, values in masks.items():
                if not np.any(values):
                    continue
                row = {"method": trace["method"], "mask": mask_name, "slot": slot}
                row.update({f"beam_{beam}": float(values[beam]) for beam in range(n_beams)})
                mask_rows.append(row)
            row = {"method": trace["method"], "slot": slot}
            row.update({f"beam_{beam}": float(trace["belief"][slot, node_id, beam]) for beam in range(n_beams)})
            belief_rows.append(row)
    write_rows(output / f"node{node_id}_candidate_masks.csv", mask_rows)
    write_rows(output / f"node{node_id}_beliefs.csv", belief_rows)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(output: Path, ours: dict[str, Any], wang: dict[str, Any], n_beams: int, node_id: int) -> list[Path]:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.linewidth": 0.9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )
    paths: list[Path] = []
    slots = np.arange(ours["candidate_mask"].shape[0])
    ours_node_count = ours["candidate_mask"][:, node_id, :].sum(axis=1)
    wang_active_node_count = wang["wang_active"][:, node_id, :].sum(axis=1)
    wang_positive_node_count = wang["wang_positive"][:, node_id, :].sum(axis=1)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(slots, ours_node_count, label="Ours MARL candidate mask", color="#1f77b4", linewidth=2.0)
    ax.plot(slots, wang_active_node_count, label="Wang active set", color="#ff7f0e", linewidth=1.8)
    ax.plot(slots, wang_positive_node_count, label="Wang positive set", color="#2ca02c", linewidth=1.8)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Beam count")
    ax.set_title(f"Node {node_id} Candidate Set Size")
    ax.set_xlim(0, max(1, len(slots) - 1))
    ax.set_ylim(0, n_beams + 3)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = output / f"node{node_id}_candidate_count.png"
    fig.savefig(path)
    paths.append(path)
    plt.close(fig)

    mean_ours = ours["candidate_mask"].sum(axis=2).mean(axis=1)
    mean_wang_active = wang["wang_active"].sum(axis=2).mean(axis=1)
    mean_wang_positive = wang["wang_positive"].sum(axis=2).mean(axis=1)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(slots, mean_ours, label="Ours mean candidate mask", color="#1f77b4", linewidth=2.0)
    ax.plot(slots, mean_wang_active, label="Wang mean active set", color="#ff7f0e", linewidth=1.8)
    ax.plot(slots, mean_wang_positive, label="Wang mean positive set", color="#2ca02c", linewidth=1.8)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Mean beam count over nodes")
    ax.set_title("Network-Average Candidate Set Size")
    ax.set_xlim(0, max(1, len(slots) - 1))
    ax.set_ylim(0, n_beams + 3)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = output / "network_mean_candidate_count.png"
    fig.savefig(path)
    paths.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.8), sharex=True)
    heatmaps = [
        ("Ours MARL candidate mask", ours["candidate_mask"][:, node_id, :].T),
        ("Wang active set", wang["wang_active"][:, node_id, :].T),
        ("Wang positive set", wang["wang_positive"][:, node_id, :].T),
    ]
    for ax, (title, data) in zip(axes, heatmaps, strict=True):
        ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
        ax.set_ylabel("Beam")
        ax.set_title(title, fontsize=11)
    axes[-1].set_xlabel("Slot")
    fig.suptitle(f"Node {node_id} Candidate Heatmaps", y=0.995, fontsize=12)
    fig.tight_layout()
    path = output / f"node{node_id}_candidate_heatmaps.png"
    fig.savefig(path)
    paths.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.8), sharex=True)
    for ax, title, data in [
        (axes[0], "Ours MARL belief", ours["belief"][:, node_id, :].T),
        (axes[1], "Wang belief", wang["belief"][:, node_id, :].T),
    ]:
        im = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest", cmap="magma", vmin=0, vmax=1)
        ax.set_ylabel("Beam")
        ax.set_title(title, fontsize=11)
    axes[-1].set_xlabel("Slot")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="Belief")
    fig.suptitle(f"Node {node_id} Beam Belief Evolution", y=0.995, fontsize=12)
    path = output / f"node{node_id}_belief_heatmaps.png"
    fig.savefig(path, bbox_inches="tight")
    paths.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for trace, color, marker, label in [
        (ours, "#1f77b4", "o", "Ours MARL selected beam"),
        (wang, "#ff7f0e", "x", "Wang selected beam"),
    ]:
        rows = [row for row in trace["actions"] if int(row["node"]) == node_id and row["mode"] != MODE_IDLE]
        ax.scatter(
            [int(row["slot"]) for row in rows],
            [int(row["beam"]) for row in rows],
            s=12,
            alpha=0.75,
            marker=marker,
            color=color,
            label=label,
        )
    ax.set_xlabel("Slot")
    ax.set_ylabel("Selected beam")
    ax.set_title(f"Node {node_id} Executed Beam Trace")
    ax.set_xlim(0, max(1, len(slots) - 1))
    ax.set_ylim(-1, n_beams)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = output / f"node{node_id}_selected_beams.png"
    fig.savefig(path)
    paths.append(path)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    main()
