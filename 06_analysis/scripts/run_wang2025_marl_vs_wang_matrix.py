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

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_SRC = REPO_ROOT / "05_simulation" / "src"
if str(SIM_SRC) not in sys.path:
    sys.path.insert(0, str(SIM_SRC))

from isac_nd_sim.config import load_config  # noqa: E402
from isac_nd_sim.runner import run_detailed  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "05_simulation" / "configs" / "wang2025_reproduction_smoke.yaml"
DEFAULT_RAW = REPO_ROOT / "05_simulation" / "results_raw" / "marl_campaign" / "wang2025_marl_vs_wang_20260709"
DEFAULT_TABLES = REPO_ROOT / "06_analysis" / "paper_tables" / "wang2025_marl_vs_wang_20260709"
DEFAULT_FIGURES = REPO_ROOT / "06_analysis" / "paper_figures" / "wang2025_marl_vs_wang_20260709"

BASELINE_PROTOCOLS = (
    "uniform_random",
    "wang2025_isac_no_collab",
    "wang2025_comm_tables",
    "wang2025_isac_tables",
    "budgeted_collision_aware_isac",
)

METHOD_SPECS = (
    {
        "protocol": "marl_no_isac_txrxidle",
        "label": "MARL, no ISAC, TX/RX/IDLE",
        "algorithm": "mappo",
        "network": "contention_shared",
        "reward_version": "collision_topology",
        "disable_isac_features": True,
        "env_protocol": "structured_marl_no_isac",
        "expert_bc_weight": 0.0,
        "expert_protocol": "collision_aware_isac",
    },
    {
        "protocol": "marl_tx_isac_txrxidle",
        "label": "MARL + TX-coupled ISAC, TX/RX/IDLE",
        "algorithm": "isac_mappo",
        "network": "contention_shared",
        "reward_version": "collision_topology",
        "disable_isac_features": False,
        "env_protocol": "isac_structured_marl",
        "expert_bc_weight": 0.0,
        "expert_protocol": "collision_aware_isac",
    },
    {
        "protocol": "marl_tx_isac_gate_bc_txrxidle",
        "label": "MARL + TX-coupled ISAC + gate BC, TX/RX/IDLE",
        "algorithm": "isac_mappo",
        "network": "balanced_topology_gated_contention_shared",
        "reward_version": "collision_topology",
        "disable_isac_features": False,
        "env_protocol": "isac_structured_marl",
        "expert_bc_weight": 0.05,
        "expert_protocol": "budgeted_collision_aware_isac",
    },
)

LABELS = {
    "uniform_random": "Uniform Random",
    "wang2025_isac_no_collab": "Wang-like ISAC, no table exchange",
    "wang2025_comm_tables": "Wang-like + neighbor table",
    "wang2025_isac_tables": "Wang-like + sensing table",
    "budgeted_collision_aware_isac": "Budgeted ISAC rule",
    **{str(spec["protocol"]): str(spec["label"]) for spec in METHOD_SPECS},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train real TX/RX/IDLE MARL policies and compare them with Wang2025-style baselines."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--node-counts", default="10,20,30,40,50")
    parser.add_argument("--train-episodes", type=int, default=60)
    parser.add_argument("--train-slots", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-slots", type=int, default=200)
    parser.add_argument("--marl-eval-mode", choices=["deterministic", "stochastic", "both"], default="deterministic")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    node_counts = parse_int_list(args.node_counts)

    train_manifests = train_methods(args)
    baseline_rows = [] if args.skip_baselines else run_baselines(args, node_counts)
    marl_rows = evaluate_methods(args, node_counts)
    all_rows = baseline_rows + marl_rows
    aggregate_rows = aggregate_rows_by_method(all_rows)

    write_csv(args.output / "per_episode_summary.csv", all_rows)
    write_csv(args.output / "aggregate_metrics.csv", aggregate_rows)
    write_manifest(args, node_counts, train_manifests, len(baseline_rows), len(marl_rows), len(aggregate_rows))
    write_readme(args, node_counts, aggregate_rows)
    if not args.skip_figures:
        write_figures(args.figures, aggregate_rows)

    print(
        json.dumps(
            {
                "output": rel(args.output),
                "figures": rel(args.figures),
                "raw_root": rel(args.raw_root),
                "node_counts": node_counts,
                "train_manifests": [rel(Path(path)) for path in train_manifests],
                "baseline_rows": len(baseline_rows),
                "marl_rows": len(marl_rows),
                "aggregate_rows": len(aggregate_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def train_methods(args: argparse.Namespace) -> list[Path]:
    manifests: list[Path] = []
    for index, spec in enumerate(METHOD_SPECS):
        train_dir = args.raw_root / str(spec["protocol"]) / "train_n10"
        manifest_path = train_dir / "manifest.json"
        if args.skip_training and manifest_path.exists():
            manifests.append(manifest_path)
            continue
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
            str(max(0, int(args.train_episodes) // 2)),
            "--checkpoint-interval",
            str(max(0, int(args.train_episodes) // 2)),
            "--node-count",
            "10",
            "--forbid-sense",
            "--env-protocol",
            str(spec["env_protocol"]),
            "--expert-protocol",
            str(spec["expert_protocol"]),
            "--expert-bc-weight",
            str(spec["expert_bc_weight"]),
            "--seed",
            str(2026071201 + 1000 * index),
            "--torch-threads",
            str(args.torch_threads),
            "--step-log-period",
            "1",
            "--resource-log-period",
            "300",
        ]
        if bool(spec["disable_isac_features"]):
            cmd.append("--disable-isac-features")
        run_command(cmd)
        manifests.append(manifest_path)
    return manifests


def run_baselines(args: argparse.Namespace, node_counts: list[int]) -> list[dict[str, Any]]:
    base_cfg = load_config(args.config)
    rows: list[dict[str, Any]] = []
    for case_id, node_count in enumerate(node_counts):
        cfg = replace(
            base_cfg,
            seed=base_cfg.seed + 901_001 + 100_003 * case_id,
            n_nodes=node_count,
            rf_chains=1,
            episodes=int(args.eval_episodes),
            slots_per_episode=int(args.eval_slots),
        )
        episode_rows, _slot_rows, _edge_rows = run_detailed(cfg, BASELINE_PROTOCOLS)
        for row in episode_rows:
            row.update(
                {
                    "case_id": case_id,
                    "node_count": node_count,
                    "rf_chains": 1,
                    "method_family": "wang_or_rule_baseline",
                    "method_label": LABELS.get(str(row["protocol"]), str(row["protocol"])),
                    "marl_trained": 0,
                    "forbid_sense": "",
                    "disabled_modes": "",
                }
            )
            rows.append(row)
    return rows


def evaluate_methods(args: argparse.Namespace, node_counts: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_index, spec in enumerate(METHOD_SPECS):
        checkpoint = args.raw_root / str(spec["protocol"]) / "train_n10" / "final_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
        for case_id, node_count in enumerate(node_counts):
            eval_dir = args.raw_root / str(spec["protocol"]) / f"eval_n{node_count}"
            cmd = [
                sys.executable,
                "05_simulation/run_marl_evaluate.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(args.config),
                "--output",
                str(eval_dir),
                "--eval-episodes",
                str(args.eval_episodes),
                "--slots",
                str(args.eval_slots),
                "--node-count",
                str(node_count),
                "--env-protocol",
                str(spec["env_protocol"]),
                "--forbid-sense",
                "--seed",
                str(2026072201 + 100_000 * method_index + 1000 * node_count),
                "--torch-threads",
                str(args.torch_threads),
                "--resource-log-period",
                "400",
                "--no-resume",
            ]
            if str(args.marl_eval_mode) == "stochastic":
                cmd.append("--stochastic")
            elif str(args.marl_eval_mode) == "both":
                cmd.append("--eval-both")
            else:
                cmd.append("--deterministic")
            run_command(cmd)
            for row in read_csv(eval_dir / "eval_episode_metrics.csv"):
                row.update(
                    {
                        "case_id": case_id,
                        "node_count": node_count,
                        "rf_chains": 1,
                        "protocol": str(spec["protocol"]),
                        "method_family": "trained_marl",
                        "method_label": str(spec["label"]),
                        "marl_trained": 1,
                        "train_node_count": 10,
                        "train_slots": int(args.train_slots),
                        "train_episodes": int(args.train_episodes),
                        "forbid_sense": 1,
                        "disabled_modes": "sense",
                    }
                )
                rows.append(row)
    return rows


def run_command(cmd: list[str]) -> None:
    print(json.dumps({"phase": "run_command", "cmd": cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def aggregate_rows_by_method(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = ("protocol", "method_label", "method_family", "node_count", "rf_chains")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)

    metrics = (
        "discovery_rate",
        "collision_penalized_discovery_rate",
        "collision_count",
        "empty_scan_ratio",
        "mean_delay_censored",
        "p95_delay_censored",
        "lambda2",
        "lcc_ratio",
        "sense_actions",
        "piggyback_sense_actions",
        "tx_actions",
        "rx_actions",
        "idle_actions",
    )
    aggregate: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        out: dict[str, Any] = dict(zip(group_keys, key))
        out["n_episodes"] = len(group_rows)
        out["marl_trained"] = max(int(float(row.get("marl_trained", 0) or 0)) for row in group_rows)
        out["forbid_sense"] = max(int(float(row.get("forbid_sense", 0) or 0)) for row in group_rows if str(row.get("forbid_sense", "")).strip() != "") if any(str(row.get("forbid_sense", "")).strip() != "" for row in group_rows) else ""
        for metric in metrics:
            values = [float(row.get(metric, 0.0) or 0.0) for row in group_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregate.append(out)
    return aggregate


def write_figures(figures: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    plot_metric(rows, figures / "marl_vs_wang_discovery_rate.png", "discovery_rate_mean", "Discovery rate", plt)
    plot_metric(rows, figures / "marl_vs_wang_cpd.png", "collision_penalized_discovery_rate_mean", "CPD", plt)
    plot_metric(rows, figures / "marl_vs_wang_collision_count.png", "collision_count_mean", "Collision count", plt)
    plot_metric(rows, figures / "marl_vs_wang_lambda2.png", "lambda2_mean", "Lambda2", plt)
    plot_metric(rows, figures / "marl_vs_wang_sense_actions.png", "sense_actions_mean", "Standalone sense actions", plt)
    plot_metric(rows, figures / "marl_vs_wang_piggyback_sense_actions.png", "piggyback_sense_actions_mean", "TX-coupled ISAC observations", plt)


def plot_metric(rows: list[dict[str, Any]], path: Path, metric: str, ylabel: str, plt: Any) -> None:
    method_order = [
        "uniform_random",
        "wang2025_isac_no_collab",
        "wang2025_comm_tables",
        "wang2025_isac_tables",
        "budgeted_collision_aware_isac",
        "marl_no_isac_txrxidle",
        "marl_tx_isac_txrxidle",
        "marl_tx_isac_gate_bc_txrxidle",
    ]
    colors = {
        "uniform_random": "#6b7280",
        "wang2025_isac_no_collab": "#4f83cc",
        "wang2025_comm_tables": "#2ca25f",
        "wang2025_isac_tables": "#8856a7",
        "budgeted_collision_aware_isac": "#e15759",
        "marl_no_isac_txrxidle": "#8c564b",
        "marl_tx_isac_txrxidle": "#f28e2b",
        "marl_tx_isac_gate_bc_txrxidle": "#b07aa1",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for protocol in method_order:
        series = sorted(
            [row for row in rows if str(row.get("protocol")) == protocol],
            key=lambda item: int(item["node_count"]),
        )
        if not series:
            continue
        ax.plot(
            [int(row["node_count"]) for row in series],
            [float(row.get(metric, 0.0) or 0.0) for row in series],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=LABELS.get(protocol, protocol),
            color=colors.get(protocol),
        )
    ax.set_xlabel("Number of UAVs")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, frameon=True, ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_manifest(
    args: argparse.Namespace,
    node_counts: list[int],
    train_manifests: list[Path],
    baseline_rows: int,
    marl_rows: int,
    aggregate_rows: int,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Real trained MARL versus Wang2025-style baselines under the same single-RF Wang matrix.",
        "config": rel(args.config),
        "raw_root": rel(args.raw_root),
        "output": rel(args.output),
        "figures": rel(args.figures),
        "node_counts": node_counts,
        "rf_chains": [1],
        "train_node_count": 10,
        "train_episodes": int(args.train_episodes),
        "train_slots": int(args.train_slots),
        "eval_episodes": int(args.eval_episodes),
        "eval_slots": int(args.eval_slots),
        "marl_eval_mode": str(args.marl_eval_mode),
        "forbid_sense": True,
        "isac_abstraction": "Standalone SENSE is disabled for MARL; directional TX carries piggyback ISAC feedback.",
        "baseline_protocols": list(BASELINE_PROTOCOLS),
        "marl_protocols": [str(spec["protocol"]) for spec in METHOD_SPECS],
        "train_manifests": [rel(path) for path in train_manifests],
        "baseline_rows": baseline_rows,
        "marl_rows": marl_rows,
        "aggregate_rows": aggregate_rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(args: argparse.Namespace, node_counts: list[int], aggregate_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Wang2025 MARL-vs-Wang Matrix",
        "",
        "This corrective campaign compares real trained MARL checkpoints against Wang2025-style baselines under the same single-RF matrix.",
        "",
        f"- Node counts: {node_counts}",
        f"- Train node count: 10",
        f"- Train slots: {args.train_slots}",
        f"- Eval slots: {args.eval_slots}",
        f"- Eval episodes: {args.eval_episodes}",
        f"- MARL eval mode: {args.marl_eval_mode}",
        "- MARL action space: TX/RX/IDLE only; standalone SENSE is disabled.",
        "- ISAC feedback: TX-coupled piggyback sensing.",
        "",
        "Files:",
        "",
        "- `per_episode_summary.csv`",
        "- `aggregate_metrics.csv`",
        "- `manifest.json`",
        "",
        "Current top-line rows:",
        "",
    ]
    for row in sorted(aggregate_rows, key=lambda item: (int(item["node_count"]), str(item["protocol"]))):
        if int(row["node_count"]) != max(node_counts):
            continue
        lines.append(
            f"- {row['method_label']}: discovery={float(row['discovery_rate_mean']):.4f}, "
            f"CPD={float(row['collision_penalized_discovery_rate_mean']):.4f}, "
            f"collisions={float(row['collision_count_mean']):.1f}, "
            f"lambda2={float(row['lambda2_mean']):.4f}"
        )
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    main()
