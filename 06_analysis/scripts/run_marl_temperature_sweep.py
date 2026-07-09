from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = REPO_ROOT / "05_simulation" / "configs" / "wang2025_reproduction_smoke.yaml"
DEFAULT_CHECKPOINT_ROOT = (
    REPO_ROOT / "05_simulation" / "results_raw" / "marl_campaign" / "wang2025_marl_vs_wang_firstpass_20260709"
)
DEFAULT_RAW = REPO_ROOT / "05_simulation" / "results_raw" / "marl_campaign" / "marl_temperature_sweep_20260709"
DEFAULT_OUTPUT = REPO_ROOT / "06_analysis" / "paper_tables" / "marl_temperature_sweep_20260709"
DEFAULT_FIGURES = REPO_ROOT / "06_analysis" / "paper_figures" / "marl_temperature_sweep_20260709"

METHODS = (
    {
        "protocol": "marl_no_isac_txrxidle",
        "label": "MARL, no ISAC",
        "env_protocol": "structured_marl_no_isac",
    },
    {
        "protocol": "marl_tx_isac_txrxidle",
        "label": "MARL + TX-coupled ISAC",
        "env_protocol": "isac_structured_marl",
    },
    {
        "protocol": "marl_tx_isac_gate_bc_txrxidle",
        "label": "MARL + TX-coupled ISAC + gate BC",
        "env_protocol": "isac_structured_marl",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate existing MARL checkpoints under stochastic temperature sweeps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--temperatures", default="0.7,1.0,1.3,1.6,2.0")
    parser.add_argument("--node-counts", default="50")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--slots", type=int, default=200)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.raw_root.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    temperatures = parse_float_list(args.temperatures)
    node_counts = parse_int_list(args.node_counts)

    episode_rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        checkpoint = args.checkpoint_root / str(method["protocol"]) / "train_n10" / "final_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        for node_count in node_counts:
            for temperature in temperatures:
                eval_dir = (
                    args.raw_root
                    / str(method["protocol"])
                    / f"n{node_count}"
                    / f"temp_{format_temperature(temperature)}"
                )
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
                    str(args.slots),
                    "--node-count",
                    str(node_count),
                    "--env-protocol",
                    str(method["env_protocol"]),
                    "--forbid-sense",
                    "--stochastic",
                    "--mode-temperature",
                    str(temperature),
                    "--beam-temperature",
                    str(temperature),
                    "--gate-temperature",
                    str(temperature),
                    "--seed",
                    str(2026073301 + 100_000 * method_index + 1000 * node_count + int(round(100 * temperature))),
                    "--torch-threads",
                    str(args.torch_threads),
                    "--resource-log-period",
                    "400",
                    "--no-resume",
                ]
                run_command(cmd)
                for row in read_csv(eval_dir / "eval_episode_metrics.csv"):
                    row.update(
                        {
                            "protocol": str(method["protocol"]),
                            "method_label": str(method["label"]),
                            "node_count": int(node_count),
                            "temperature": float(temperature),
                            "mode_temperature": float(temperature),
                            "beam_temperature": float(temperature),
                            "gate_temperature": float(temperature),
                            "forbid_sense": 1,
                            "checkpoint": rel(checkpoint),
                        }
                    )
                    episode_rows.append(row)

    aggregate_rows = aggregate(episode_rows)
    write_csv(args.output / "per_episode_summary.csv", episode_rows)
    write_csv(args.output / "aggregate_metrics.csv", aggregate_rows)
    write_manifest(args, temperatures, node_counts, len(episode_rows), len(aggregate_rows))
    write_readme(args, temperatures, node_counts, aggregate_rows)
    if not args.skip_figures:
        write_figures(args.figures, aggregate_rows)

    print(
        json.dumps(
            {
                "output": rel(args.output),
                "figures": rel(args.figures),
                "raw_root": rel(args.raw_root),
                "temperatures": temperatures,
                "node_counts": node_counts,
                "episode_rows": len(episode_rows),
                "aggregate_rows": len(aggregate_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = ("protocol", "method_label", "node_count", "temperature")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)
    metrics = (
        "discovery_rate",
        "collision_penalized_discovery_rate",
        "collision_count",
        "lambda2",
        "empty_scan_ratio",
        "tx_actions",
        "rx_actions",
        "idle_actions",
        "sense_actions",
        "piggyback_sense_actions",
        "access_gate_backoff_ratio",
        "access_gate_normal_ratio",
        "access_gate_aggressive_ratio",
    )
    out_rows: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        out: dict[str, Any] = dict(zip(group_keys, key))
        out["n_episodes"] = len(group)
        for metric in metrics:
            values = [float(row.get(metric, 0.0) or 0.0) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        scan_actions = mean(float(row.get("scan_actions", 0.0) or 0.0) for row in group)
        out["tx_ratio_mean"] = out["tx_actions_mean"] / max(1.0, scan_actions)
        out["rx_ratio_mean"] = out["rx_actions_mean"] / max(1.0, scan_actions)
        out["idle_per_slot_node_mean"] = out["idle_actions_mean"] / max(1.0, float(out["node_count"]) * mean(float(row.get("slots", 0.0) or 0.0) for row in group))
        out_rows.append(out)
    return out_rows


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
    plot_metric(rows, figures / "temperature_discovery_rate.png", "discovery_rate_mean", "Discovery rate", plt)
    plot_metric(rows, figures / "temperature_cpd.png", "collision_penalized_discovery_rate_mean", "CPD", plt)
    plot_metric(rows, figures / "temperature_collision_count.png", "collision_count_mean", "Collision count", plt)
    plot_metric(rows, figures / "temperature_lambda2.png", "lambda2_mean", "Lambda2", plt)
    plot_metric(rows, figures / "temperature_tx_ratio.png", "tx_ratio_mean", "TX ratio among active scans", plt)


def plot_metric(rows: list[dict[str, Any]], path: Path, metric: str, ylabel: str, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colors = {
        "marl_no_isac_txrxidle": "#8c564b",
        "marl_tx_isac_txrxidle": "#f28e2b",
        "marl_tx_isac_gate_bc_txrxidle": "#b07aa1",
    }
    for method in METHODS:
        protocol = str(method["protocol"])
        series = sorted([row for row in rows if str(row["protocol"]) == protocol], key=lambda item: float(item["temperature"]))
        if not series:
            continue
        ax.plot(
            [float(row["temperature"]) for row in series],
            [float(row.get(metric, 0.0) or 0.0) for row in series],
            marker="o",
            linewidth=1.8,
            markersize=4.8,
            label=str(method["label"]),
            color=colors.get(protocol),
        )
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_manifest(
    args: argparse.Namespace,
    temperatures: list[float],
    node_counts: list[int],
    episode_rows: int,
    aggregate_rows: int,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Stochastic deployment-temperature sweep for existing Wang-style MARL checkpoints.",
        "config": rel(args.config),
        "checkpoint_root": rel(args.checkpoint_root),
        "raw_root": rel(args.raw_root),
        "output": rel(args.output),
        "figures": rel(args.figures),
        "temperatures": temperatures,
        "node_counts": node_counts,
        "eval_episodes": int(args.eval_episodes),
        "slots": int(args.slots),
        "forbid_sense": True,
        "temperature_applied_to": ["mode_logits", "beam_logits", "gate_logits"],
        "episode_rows": episode_rows,
        "aggregate_rows": aggregate_rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_readme(args: argparse.Namespace, temperatures: list[float], node_counts: list[int], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# MARL Temperature Sweep",
        "",
        "This sweep reuses the Wang-style first-pass MARL checkpoints and changes only stochastic deployment temperature.",
        "",
        f"- Temperatures: {temperatures}",
        f"- Node counts: {node_counts}",
        f"- Eval episodes: {args.eval_episodes}",
        f"- Slots: {args.slots}",
        "- Standalone SENSE remains disabled.",
        "",
        "Best CPD rows by method:",
        "",
    ]
    for method in METHODS:
        method_rows = [row for row in rows if str(row["protocol"]) == str(method["protocol"])]
        if not method_rows:
            continue
        best = max(method_rows, key=lambda row: float(row["collision_penalized_discovery_rate_mean"]))
        lines.append(
            f"- {best['method_label']}: temp={float(best['temperature']):.2f}, "
            f"discovery={float(best['discovery_rate_mean']):.4f}, "
            f"CPD={float(best['collision_penalized_discovery_rate_mean']):.4f}, "
            f"collisions={float(best['collision_count_mean']):.1f}, "
            f"lambda2={float(best['lambda2_mean']):.4f}"
        )
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(cmd: list[str]) -> None:
    print(json.dumps({"phase": "run_command", "cmd": cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def format_temperature(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


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
