from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_SRC = REPO_ROOT / "05_simulation" / "src"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

from isac_nd_sim.config import SimulationConfig, load_config  # noqa: E402
from isac_nd_sim.mobility import step_states  # noqa: E402
from isac_nd_sim.simulator import Action, MODE_IDLE, MODE_RX, MODE_TX, NeighborDiscoverySimulator  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "05_simulation" / "configs" / "wang2025_reproduction_smoke.yaml"
DEFAULT_RAW = REPO_ROOT / "05_simulation" / "results_raw" / "marl_campaign" / "wang2025_aligned_discovery_first_20260709"
DEFAULT_OUTPUT = REPO_ROOT / "06_analysis" / "paper_tables" / "wang2025_aligned_discovery_first_20260709"

COMMON_ENV_PROTOCOL = "wang2025_isac_tables"

ACTION_POLICY_PROTOCOLS = (
    "uniform_trx_idle_random",
    "wang2025_isac_tables",
    "budgeted_collision_aware_isac",
)

METHOD_SPECS = (
    {
        "protocol": "marl_wang_isac_tables_discovery_first",
        "label": "MARL + Wang ISAC tables, discovery-first",
        "algorithm": "isac_mappo",
        "network": "contention_shared",
        "reward_version": "discovery_first",
        "env_protocol": COMMON_ENV_PROTOCOL,
        "disable_isac_features": False,
        "expert_bc_weight": 0.0,
        "expert_protocol": "budgeted_collision_aware_isac",
    },
)

LABELS = {
    "uniform_trx_idle_random": "Uniform TX/RX/IDLE random",
    "wang2025_isac_no_collab": "Wang ISAC action policy",
    "wang2025_comm_tables": "Wang neighbor-table action policy",
    "wang2025_isac_tables": "Wang sensing-table action policy",
    "budgeted_collision_aware_isac": "Budgeted ISAC rule",
    **{str(spec["protocol"]): str(spec["label"]) for spec in METHOD_SPECS},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate MARL in the same Wang2025-style environment: "
            "25-degree beams, 200-slot horizon, single RF, same sensing model, "
        "and a common Wang neighbor/sensing-table environment for every main row."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--node-counts", default="10,20,30,40,50")
    parser.add_argument("--train-episodes", type=int, default=60)
    parser.add_argument("--train-slots", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-slots", type=int, default=200)
    parser.add_argument("--train-node-count", type=int, default=10)
    parser.add_argument("--marl-eval-mode", choices=["deterministic", "stochastic", "both"], default="stochastic")
    parser.add_argument("--action-policies", default=",".join(ACTION_POLICY_PROTOCOLS))
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_counts = parse_int_list(args.node_counts)
    action_policies = parse_policy_list(args.action_policies)
    commands = planned_training_commands(args) + planned_eval_commands(args, node_counts)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "commands": commands,
                    "node_counts": node_counts,
                    "common_env_protocol": COMMON_ENV_PROTOCOL,
                    "action_policy_protocols": action_policies,
                    "methods": METHOD_SPECS,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    baseline_rows = [] if args.skip_baselines else run_baselines(args, node_counts, action_policies)
    baseline_aggregate = aggregate_rows(baseline_rows)
    write_csv(args.output / "baseline_target_metrics.csv", baseline_aggregate)

    train_manifests = train_methods(args)
    write_training_eval_gap(args, baseline_aggregate)
    marl_rows = evaluate_methods(args, node_counts)
    rows = baseline_rows + marl_rows
    aggregate = aggregate_rows(rows)

    write_csv(args.output / "per_episode_summary.csv", rows)
    write_csv(args.output / "aggregate_metrics.csv", aggregate)
    write_manifest(args, node_counts, action_policies, train_manifests, baseline_rows, marl_rows, aggregate)
    write_readme(args, node_counts, aggregate)
    print(
        json.dumps(
            {
                "output": rel(args.output),
                "raw_root": rel(args.raw_root),
                "node_counts": node_counts,
                "baseline_rows": len(baseline_rows),
                "marl_rows": len(marl_rows),
                "aggregate_rows": len(aggregate),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def planned_training_commands(args: argparse.Namespace) -> list[list[str]]:
    if args.skip_training:
        return []
    return [training_command(args, spec, idx) for idx, spec in enumerate(METHOD_SPECS)]


def planned_eval_commands(args: argparse.Namespace, node_counts: list[int]) -> list[list[str]]:
    commands: list[list[str]] = []
    for spec in METHOD_SPECS:
        checkpoint = args.raw_root / str(spec["protocol"]) / "train" / "final_model.pt"
        for node_count in node_counts:
            commands.append(eval_command(args, spec, checkpoint, int(node_count)))
    return commands


def train_methods(args: argparse.Namespace) -> list[Path]:
    manifests: list[Path] = []
    for idx, spec in enumerate(METHOD_SPECS):
        train_dir = args.raw_root / str(spec["protocol"]) / "train"
        manifest = train_dir / "manifest.json"
        if args.skip_training and manifest.exists():
            manifests.append(manifest)
            continue
        run_command(training_command(args, spec, idx))
        manifests.append(manifest)
    return manifests


def training_command(args: argparse.Namespace, spec: dict[str, Any], idx: int) -> list[str]:
    train_dir = args.raw_root / str(spec["protocol"]) / "train"
    cmd = [
        sys.executable,
        "05_simulation/run_marl_training.py",
        "--config",
        str(args.config),
        "--output",
        str(train_dir),
        "--algorithm",
        str(spec["algorithm"]),
        "--network",
        str(spec["network"]),
        "--reward-version",
        str(spec["reward_version"]),
        "--episodes",
        str(args.train_episodes),
        "--slots",
        str(args.train_slots),
        "--eval-episodes",
        "2",
        "--eval-interval",
        str(max(1, int(args.train_episodes) // 4)),
        "--checkpoint-interval",
        str(max(1, int(args.train_episodes) // 2)),
        "--node-count",
        str(args.train_node_count),
        "--env-protocol",
        str(spec["env_protocol"]),
        "--expert-protocol",
        str(spec["expert_protocol"]),
        "--expert-bc-weight",
        str(spec["expert_bc_weight"]),
        "--forbid-sense",
        "--stochastic-eval",
        "--seed",
        str(2026073001 + 1000 * idx),
        "--torch-threads",
        str(args.torch_threads),
        "--step-log-period",
        "10",
        "--resource-log-period",
        "200",
    ]
    if bool(spec["disable_isac_features"]):
        cmd.append("--disable-isac-features")
    return cmd


def run_baselines(args: argparse.Namespace, node_counts: list[int], action_policies: list[str]) -> list[dict[str, Any]]:
    cfg0 = load_config(args.config)
    rows: list[dict[str, Any]] = []
    for case_id, node_count in enumerate(node_counts):
        cfg = replace(
            cfg0,
            seed=int(cfg0.seed) + 910_000 + 10_000 * case_id,
            n_nodes=int(node_count),
            rf_chains=1,
            episodes=int(args.eval_episodes),
            slots_per_episode=int(args.eval_slots),
        )
        for policy_protocol in action_policies:
            episode_rows = run_action_policy_in_common_env(cfg, policy_protocol, COMMON_ENV_PROTOCOL)
            for row in episode_rows:
                row.update(common_row_fields(case_id, node_count, policy_protocol, "common_env_action_policy", 0))
                row["env_protocol"] = COMMON_ENV_PROTOCOL
                row["action_policy_protocol"] = policy_protocol
                rows.append(row)
    return rows


def run_action_policy_in_common_env(
    cfg: SimulationConfig,
    action_policy_protocol: str,
    env_protocol: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in range(int(cfg.episodes)):
        sim = NeighborDiscoverySimulator(
            cfg,
            protocol=env_protocol,
            seed=int(cfg.seed) + episode,
            scenario_seed=int(cfg.seed) + episode,
        )
        sim.reset()
        for slot in range(cfg.slots_per_episode):
            sim._beam_matrix_cache = None
            sim._distance_matrix_cache = None
            sim._sensing_profile_cache.clear()
            true_comm_edges = sim.true_edges(cfg.communication_range_m)
            for edge in true_comm_edges:
                sim.first_true_slot.setdefault(edge, slot)
            sim.age += 1.0
            sim.belief *= cfg.confidence_decay
            sim._candidate_pool_cache.clear()

            old_protocol = sim.protocol
            try:
                if str(action_policy_protocol) == "uniform_trx_idle_random":
                    actions = uniform_trx_idle_actions(sim)
                else:
                    sim.protocol = str(action_policy_protocol)
                    actions = sim.select_actions(slot, true_comm_edges)
            finally:
                sim.protocol = old_protocol

            sim.snapshot_pre_sensing_candidates(slot)
            sim.update_action_counts(actions, slot)
            sim.update_empty_scan_counts(actions, true_comm_edges, slot)
            sim.update_sensing(actions, slot)
            sim._candidate_pool_cache.clear()
            new_edges = sim.resolve_discoveries(slot, actions, true_comm_edges)
            if cfg.slot_metric_period > 0 and slot % cfg.slot_metric_period == 0:
                sim.per_slot_rows.append(sim.slot_metrics(episode, slot, true_comm_edges, new_edges))
            step_states(sim.states, cfg.area_size_m, cfg.mobility, cfg.slot_duration_s, slot, sim.mobility_rng)
            sim._beam_matrix_cache = None
            sim._distance_matrix_cache = None
            sim._sensing_profile_cache.clear()

        row = sim.summarize(episode).as_dict()
        row["protocol"] = str(action_policy_protocol)
        row["env_protocol"] = str(env_protocol)
        row["action_policy_protocol"] = str(action_policy_protocol)
        rows.append(row)
    return rows


def uniform_trx_idle_actions(sim: NeighborDiscoverySimulator) -> list[Action]:
    """Random action policy with the same no-standalone-SENSE action set."""

    raw_probs = np.asarray([sim.cfg.p_tx, sim.cfg.p_rx, sim.cfg.p_idle], dtype=float)
    if not np.isfinite(raw_probs).all() or float(raw_probs.sum()) <= 0.0:
        raw_probs = np.asarray([0.5, 0.5, 0.0], dtype=float)
    probs = raw_probs / float(raw_probs.sum())
    modes = (MODE_TX, MODE_RX, MODE_IDLE)
    actions: list[Action] = []
    for _node in range(sim.cfg.n_nodes):
        mode = str(sim.rng.choice(modes, p=probs))
        beam = 0 if mode == MODE_IDLE else int(sim.rng.integers(0, sim.cfg.n_beams))
        actions.append(Action(mode, beam))
    return actions


def evaluate_methods(args: argparse.Namespace, node_counts: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec_idx, spec in enumerate(METHOD_SPECS):
        checkpoint = args.raw_root / str(spec["protocol"]) / "train" / "final_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        for case_id, node_count in enumerate(node_counts):
            out = args.raw_root / str(spec["protocol"]) / f"eval_n{node_count}"
            run_command(eval_command(args, spec, checkpoint, int(node_count), out, spec_idx))
            for row in read_csv(out / "eval_episode_metrics.csv"):
                row.update(common_row_fields(case_id, node_count, spec["protocol"], "trained_marl", 1))
                row.update(
                    {
                        "train_node_count": int(args.train_node_count),
                        "train_slots": int(args.train_slots),
                        "train_episodes": int(args.train_episodes),
                        "env_protocol": str(spec["env_protocol"]),
                        "action_policy_protocol": str(spec["protocol"]),
                        "train_env_protocol": str(spec["env_protocol"]),
                        "train_network": str(spec["network"]),
                        "reward_version": str(spec["reward_version"]),
                        "forbid_sense": 1,
                    }
                )
                rows.append(row)
    return rows


def eval_command(
    args: argparse.Namespace,
    spec: dict[str, Any],
    checkpoint: Path,
    node_count: int,
    output: Path | None = None,
    spec_idx: int = 0,
) -> list[str]:
    out = output or args.raw_root / str(spec["protocol"]) / f"eval_n{node_count}"
    cmd = [
        sys.executable,
        "05_simulation/run_marl_evaluate.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(args.config),
        "--output",
        str(out),
        "--eval-episodes",
        str(args.eval_episodes),
        "--slots",
        str(args.eval_slots),
        "--node-count",
        str(node_count),
        "--env-protocol",
        str(spec["env_protocol"]),
        "--reward-version",
        str(spec["reward_version"]),
        "--forbid-sense",
        "--seed",
        str(2026074001 + 100_000 * spec_idx + 1000 * int(node_count)),
        "--torch-threads",
        str(args.torch_threads),
        "--resource-log-period",
        "200",
        "--no-resume",
    ]
    if args.marl_eval_mode == "stochastic":
        cmd.append("--stochastic")
    elif args.marl_eval_mode == "both":
        cmd.append("--eval-both")
    else:
        cmd.append("--deterministic")
    return cmd


def common_row_fields(case_id: int, node_count: int, protocol: object, family: str, trained: int) -> dict[str, Any]:
    protocol_text = str(protocol)
    return {
        "case_id": int(case_id),
        "node_count": int(node_count),
        "rf_chains": 1,
        "protocol": protocol_text,
        "method_label": LABELS.get(protocol_text, protocol_text),
        "method_family": family,
        "marl_trained": int(trained),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = ("protocol", "method_label", "method_family", "node_count", "rf_chains")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)
    metrics = (
        "discovery_rate",
        "mean_delay_censored",
        "p95_delay_censored",
        "lambda2",
        "lcc_ratio",
        "empty_scan_ratio",
        "collision_count",
        "sense_actions",
        "piggyback_sense_actions",
        "sensing_observations",
        "tx_actions",
        "rx_actions",
        "idle_actions",
    )
    out_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        out: dict[str, Any] = dict(zip(group_keys, key))
        out["n_episodes"] = len(group)
        out["marl_trained"] = max(int(float(row.get("marl_trained", 0) or 0)) for row in group)
        for metric in metrics:
            values = [float(row.get(metric, 0.0) or 0.0) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out_rows.append(out)
    return out_rows


def write_training_eval_gap(args: argparse.Namespace, baseline_aggregate: list[dict[str, Any]]) -> None:
    if args.skip_training:
        return
    target = best_baseline_target(baseline_aggregate)
    rows: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        eval_path = args.raw_root / str(spec["protocol"]) / "train" / "eval_episode_metrics.csv"
        if not eval_path.exists():
            continue
        groups: dict[str, list[dict[str, str]]] = {}
        for row in read_csv(eval_path):
            groups.setdefault(str(row.get("eval_after_episode", "")), []).append(row)
        for eval_after_episode, group in sorted(groups.items(), key=lambda item: int(float(item[0] or 0))):
            discovery_values = [float(row.get("discovery_rate", 0.0) or 0.0) for row in group]
            return_values = [float(row.get("episode_return_mean_per_agent", 0.0) or 0.0) for row in group]
            discovery_mean = mean(discovery_values) if discovery_values else 0.0
            return_mean = mean(return_values) if return_values else 0.0
            target_discovery = float(target.get("discovery_rate_mean", 0.0) or 0.0) if target else 0.0
            rows.append(
                {
                    "protocol": str(spec["protocol"]),
                    "eval_after_episode": eval_after_episode,
                    "eval_episodes": len(group),
                    "eval_discovery_rate_mean": discovery_mean,
                    "eval_return_mean_per_agent": return_mean,
                    "best_baseline_protocol": target.get("protocol", "") if target else "",
                    "best_baseline_label": target.get("method_label", "") if target else "",
                    "best_baseline_discovery_rate_mean": target_discovery,
                    "discovery_rate_gap_vs_best_baseline": discovery_mean - target_discovery,
                }
            )
    write_csv(args.output / "training_eval_gap.csv", rows)


def best_baseline_target(baseline_aggregate: list[dict[str, Any]]) -> dict[str, Any]:
    if not baseline_aggregate:
        return {}
    return max(baseline_aggregate, key=lambda row: float(row.get("discovery_rate_mean", 0.0) or 0.0))


def write_manifest(
    args: argparse.Namespace,
    node_counts: list[int],
    action_policies: list[str],
    train_manifests: list[Path],
    baseline_rows: list[dict[str, Any]],
    marl_rows: list[dict[str, Any]],
    aggregate_rows_: list[dict[str, Any]],
) -> None:
    cfg = load_config(args.config)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Wang-aligned MARL training/evaluation with Wang neighbor/sensing-table interaction.",
        "config": rel(args.config),
        "raw_root": rel(args.raw_root),
        "output": rel(args.output),
        "train_node_count": int(args.train_node_count),
        "node_counts": node_counts,
        "train_slots": int(args.train_slots),
        "eval_slots": int(args.eval_slots),
        "train_episodes": int(args.train_episodes),
        "eval_episodes": int(args.eval_episodes),
        "beam_cells": [int(cfg.azimuth_cells), int(cfg.elevation_cells)],
        "approx_beam_width_deg": [360.0 / float(cfg.azimuth_cells), 180.0 / float(cfg.elevation_cells)],
        "rf_chains": 1,
        "standalone_sense": "disabled for all main fair-comparison rows",
        "common_env_protocol": COMMON_ENV_PROTOCOL,
        "action_policy_protocols": list(action_policies),
        "baseline_first_workflow": "enabled; baseline_target_metrics.csv is written before MARL training.",
        "periodic_gap_analysis": "training_eval_gap.csv compares each MARL eval interval against the best baseline discovery rate.",
        "fairness_rule": "Main rows share the same environment dynamics; methods differ only in executed TX/RX/IDLE-and-beam action decisions.",
        "main_action_space": "TX/RX/IDLE plus one selected communication beam; no standalone SENSE in the main fair-comparison rows.",
        "marl_access_gate": "disabled in the main Wang-aligned MARL specification by using the non-gated contention_shared network.",
        "marl_table_exchange": "enabled through the common Wang neighbor and sensing table environment after successful first interaction",
        "marl_piggyback_isac": "TX-only under the common Wang protocol",
        "methods": METHOD_SPECS,
        "train_manifests": [rel(path) for path in train_manifests],
        "baseline_rows": len(baseline_rows),
        "marl_rows": len(marl_rows),
        "aggregate_rows": len(aggregate_rows_),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(args: argparse.Namespace, node_counts: list[int], aggregate: list[dict[str, Any]]) -> None:
    lines = [
        "# Wang-Aligned Discovery-First MARL Matrix",
        "",
        "This matrix is the strict environment-alignment entry point for MARL-vs-Wang experiments.",
        "",
        "- Base config: `05_simulation/configs/wang2025_reproduction_smoke.yaml`",
        "- Training and evaluation horizon: Wang-style 200 slots by default.",
        "- Beam grid: 15 azimuth x 7 elevation, approximately 25 degrees.",
        "- RF chains: 1.",
        "- Main comparison environment: fixed `wang2025_isac_tables` for Wang/rule action policies and MARL.",
        "- Only the executed TX/RX/IDLE-and-beam action policy changes across main rows.",
        "- MARL network in this matrix: non-gated `contention_shared`; no environment-side access-gate rewriting in the main comparison.",
        "- Standalone SENSE: disabled for MARL.",
        "- Uniform random baseline: TX/RX/IDLE only, with no standalone SENSE.",
        "- ISAC feedback in the common environment: TX-only piggyback sensing.",
        "- Table exchange in the common environment: Wang neighbor and sensing tables after successful first interaction.",
        "",
        "Top-line aggregate rows:",
        "",
    ]
    for node_count in node_counts:
        lines.append(f"## N={node_count}")
        for row in sorted([item for item in aggregate if int(item["node_count"]) == int(node_count)], key=lambda item: str(item["method_label"])):
            lines.append(
                f"- {row['method_label']}: discovery={float(row['discovery_rate_mean']):.4f}, "
                f"delay={float(row['mean_delay_censored_mean']):.1f}, "
                f"p95={float(row['p95_delay_censored_mean']):.1f}, "
                f"lambda2={float(row['lambda2_mean']):.3f}"
            )
        lines.append("")
    (args.output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_policy_list(text: str) -> list[str]:
    policies = [part.strip() for part in str(text).split(",") if part.strip()]
    unsupported = [policy for policy in policies if policy not in ACTION_POLICY_PROTOCOLS]
    if unsupported:
        raise ValueError(f"Unsupported action policies: {unsupported}. Supported: {list(ACTION_POLICY_PROTOCOLS)}")
    if not policies:
        raise ValueError("At least one action policy is required unless --skip-baselines is used.")
    return policies


def run_command(cmd: list[str]) -> None:
    print(json.dumps({"phase": "run_command", "cmd": cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    main()
