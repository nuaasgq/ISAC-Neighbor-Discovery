from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_ROOT = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "round_trust_gated_tables_eval"
    / "B10_N100_3000slot_5ep_seed2026071091"
)
DEFAULT_TRAIN_ROOT = (
    ROOT / "05_simulation" / "results_raw" / "marl_campaign" / "budgeted_gate_bc_sweep_20260709_v2" / "train"
)
DEFAULT_TRANSFER_ROOT = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "marl_campaign"
    / "budgeted_gate_bc_sweep_20260709_v2_transfer"
)
DEFAULT_OUTPUT = ROOT / "06_analysis" / "paper_tables" / "marl" / "trust_gate_bc_sweep_20260709"
DEFAULT_FIGURES = ROOT / "06_analysis" / "paper_figures" / "marl_trust_gate_bc_20260709"

METRICS = [
    "discovery_rate",
    "collision_count",
    "collision_penalized_discovery_rate",
    "lambda2",
    "empty_scan_ratio",
    "scan_actions_per_discovery_censored",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate trust-gated table and Budgeted expert gate BC sweep outputs.")
    parser.add_argument("--protocol-root", default=str(DEFAULT_PROTOCOL_ROOT))
    parser.add_argument("--train-root", default=str(DEFAULT_TRAIN_ROOT))
    parser.add_argument("--transfer-root", default=str(DEFAULT_TRANSFER_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--figures", default=str(DEFAULT_FIGURES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_root = Path(args.protocol_root)
    train_root = Path(args.train_root)
    transfer_root = Path(args.transfer_root)
    output = Path(args.output)
    figures = Path(args.figures)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    protocol_rows = collect_protocol_rows(protocol_root)
    protocol_summary = summarize_protocols(protocol_rows)
    training_rows, eval_rows, training_curve_rows = collect_training_rows(train_root)
    eval_summary = summarize_eval_rows(eval_rows)
    transfer_rows = collect_transfer_rows(transfer_root)
    transfer_summary = summarize_transfer_rows(transfer_rows)

    write_rows(output / "protocol_eval_rows.csv", protocol_rows)
    write_rows(output / "protocol_eval_summary.csv", protocol_summary)
    write_rows(output / "bc_training_runs.csv", training_rows)
    write_rows(output / "bc_training_curves.csv", training_curve_rows)
    write_rows(output / "bc_eval_rows.csv", eval_rows)
    write_rows(output / "bc_eval_summary.csv", eval_summary)
    write_rows(output / "bc_transfer_rows.csv", transfer_rows)
    write_rows(output / "bc_transfer_summary.csv", transfer_summary)

    figures_written = plot_figures(protocol_summary, training_rows, training_curve_rows, eval_summary, transfer_summary, figures)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_root": str(protocol_root),
        "train_root": str(train_root),
        "transfer_root": str(transfer_root),
        "output": str(output),
        "figures": str(figures),
        "protocol_rows": len(protocol_rows),
        "protocol_summary_rows": len(protocol_summary),
        "bc_training_runs": len(training_rows),
        "bc_training_curve_rows": len(training_curve_rows),
        "bc_eval_rows": len(eval_rows),
        "bc_eval_summary_rows": len(eval_summary),
        "bc_transfer_rows": len(transfer_rows),
        "bc_transfer_summary_rows": len(transfer_summary),
        "figures_written": figures_written,
        "status": completion_status(protocol_summary, training_rows),
        "files": [
            "protocol_eval_rows.csv",
            "protocol_eval_summary.csv",
            "bc_training_runs.csv",
            "bc_training_curves.csv",
            "bc_eval_rows.csv",
            "bc_eval_summary.csv",
            "bc_transfer_rows.csv",
            "bc_transfer_summary.csv",
            "manifest.json",
            "README.md",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(build_readme(manifest, protocol_summary, training_rows), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def collect_protocol_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/eval_episode_metrics.csv")):
        protocol = path.parent.name
        for row in read_rows(path):
            row = dict(row)
            row["protocol_dir"] = protocol
            rows.append(row)
    return rows


def summarize_protocols(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        protocol = str(row.get("protocol", row.get("protocol_dir", "")))
        by_protocol.setdefault(protocol, []).append(row)
    out: list[dict[str, Any]] = []
    for protocol, group in sorted(by_protocol.items()):
        item: dict[str, Any] = {
            "protocol": protocol,
            "method": group[0].get("method", protocol),
            "episodes": len(group),
        }
        for metric in METRICS:
            values = numeric_values(group, metric)
            item[f"{metric}_mean"] = mean(values) if values else ""
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else ""
        out.append(item)
    return out


def collect_training_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    training_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    training_curve_rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        manifest = read_json(run_dir / "manifest.json")
        episode_rows = read_rows(run_dir / "episode_metrics.csv")
        eval_episode_rows = read_rows(run_dir / "eval_episode_metrics.csv")
        run_name = run_dir.name
        final_episode = episode_rows[-1] if episode_rows else {}
        first_episode = episode_rows[0] if episode_rows else {}
        row = {
            "run": run_name,
            "episodes_completed": len(episode_rows),
            "manifest_complete": bool(manifest),
            "expert_bc_weight": manifest.get("expert_bc_weight", parse_bc_weight(run_name)),
            "expert_protocol": manifest.get("expert_protocol", parse_expert_protocol(run_name)),
            "network": manifest.get("network", ""),
            "seed": manifest.get("seed", ""),
            "first_expert_bc_loss": first_episode.get("expert_bc_loss", ""),
            "final_expert_bc_loss": final_episode.get("expert_bc_loss", ""),
            "final_discovery_rate": final_episode.get("discovery_rate", ""),
            "final_collision_count": final_episode.get("collision_count", ""),
            "final_cpd": final_episode.get("collision_penalized_discovery_rate", ""),
            "final_backoff_ratio": final_episode.get("access_gate_backoff_ratio", ""),
            "final_aggressive_ratio": final_episode.get("access_gate_aggressive_ratio", ""),
            "final_model_exists": (run_dir / "final_model.pt").exists(),
            "updated": run_dir.stat().st_mtime,
        }
        training_rows.append(row)
        for episode_row in episode_rows:
            curve_item = dict(episode_row)
            curve_item["run"] = run_name
            curve_item["expert_bc_weight"] = row["expert_bc_weight"]
            curve_item["expert_protocol"] = row["expert_protocol"]
            curve_item["final_model_exists"] = row["final_model_exists"]
            training_curve_rows.append(curve_item)
        for eval_row in eval_episode_rows:
            eval_item = dict(eval_row)
            eval_item["run"] = run_name
            eval_item["expert_bc_weight"] = row["expert_bc_weight"]
            eval_item["expert_protocol"] = row["expert_protocol"]
            eval_rows.append(eval_item)
    return training_rows, eval_rows, training_curve_rows


def summarize_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("run", "")), str(row.get("eval_after_episode", "")), str(row.get("phase", "")))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (run, eval_after, phase), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "run": run,
            "expert_bc_weight": group[0].get("expert_bc_weight", ""),
            "expert_protocol": group[0].get("expert_protocol", ""),
            "eval_after_episode": eval_after,
            "phase": phase,
            "episodes": len(group),
        }
        for metric in METRICS:
            values = numeric_values(group, metric)
            item[f"{metric}_mean"] = mean(values) if values else ""
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else ""
        out.append(item)
    return out


def collect_transfer_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for run_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        if run_dir.name == "logs":
            continue
        manifest = read_json(run_dir / "manifest.json")
        azimuth_cells = manifest.get("azimuth_cells", parse_azimuth_cells(run_dir.name))
        beamwidth_deg = ""
        try:
            if float(azimuth_cells) > 0.0:
                beamwidth_deg = 360.0 / float(azimuth_cells)
        except (TypeError, ValueError):
            beamwidth_deg = parse_beamwidth_from_name(run_dir.name)
        if beamwidth_deg == "":
            beamwidth_deg = parse_beamwidth_from_name(run_dir.name)
        for row in read_rows(run_dir / "eval_episode_metrics.csv"):
            item = dict(row)
            item["transfer_run"] = run_dir.name
            item["manifest_complete"] = bool(manifest)
            item["checkpoint"] = manifest.get("checkpoint", "")
            item["beam_executor"] = manifest.get("beam_executor", "")
            item["mode_executor"] = manifest.get("mode_executor", "")
            item["beamwidth_deg"] = beamwidth_deg
            item["azimuth_cells"] = manifest.get("azimuth_cells", azimuth_cells)
            item["elevation_cells"] = manifest.get("elevation_cells", "")
            item["node_count"] = manifest.get("node_count", item.get("n_nodes", ""))
            item["slots_per_episode"] = manifest.get("slots_per_episode", item.get("slots", ""))
            rows.append(item)
    return rows


def summarize_transfer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("transfer_run", "")), str(row.get("beamwidth_deg", "")))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (run, beamwidth), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "transfer_run": run,
            "beamwidth_deg": beamwidth,
            "episodes": len(group),
            "node_count": group[0].get("node_count", ""),
            "slots_per_episode": group[0].get("slots_per_episode", ""),
            "beam_executor": group[0].get("beam_executor", ""),
            "mode_executor": group[0].get("mode_executor", ""),
            "manifest_complete": group[0].get("manifest_complete", False),
        }
        for metric in METRICS:
            values = numeric_values(group, metric)
            item[f"{metric}_mean"] = mean(values) if values else ""
            item[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0 if values else ""
        out.append(item)
    return out


def plot_figures(
    protocol_summary: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    training_curve_rows: list[dict[str, Any]],
    eval_summary: list[dict[str, Any]],
    transfer_summary: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11})
    written: list[str] = []
    if protocol_summary:
        path = output / "protocol_cpd_b10_n100.png"
        labels = [label_for_protocol(str(row["protocol"])) for row in protocol_summary]
        values = [float(row.get("collision_penalized_discovery_rate_mean") or 0.0) for row in protocol_summary]
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.bar(labels, values, color=["#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#76B7B2"][: len(labels)])
        ax.set_ylabel("Collision-penalized discovery")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(str(path))
    if training_rows:
        path = output / "bc_final_training_tradeoff.png"
        weights = [float(row.get("expert_bc_weight") or 0.0) for row in training_rows]
        discovery = [float(row.get("final_discovery_rate") or 0.0) for row in training_rows]
        collision = [float(row.get("final_collision_count") or 0.0) for row in training_rows]
        fig, ax1 = plt.subplots(figsize=(6.4, 4.8))
        ax2 = ax1.twinx()
        ax1.plot(weights, discovery, marker="o", color="#4E79A7", label="Discovery")
        ax2.plot(weights, collision, marker="s", color="#E15759", label="Collisions")
        ax1.set_xlabel("Expert BC weight")
        ax1.set_ylabel("Final train discovery")
        ax2.set_ylabel("Final train collisions")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(str(path))
    written.extend(plot_training_curves(training_curve_rows, output, plt))
    stochastic = [row for row in eval_summary if row.get("phase") == "eval_stochastic"]
    if stochastic:
        path = output / "bc_eval_cpd_by_checkpoint.png"
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        by_run: dict[str, list[dict[str, Any]]] = {}
        for row in stochastic:
            by_run.setdefault(str(row["run"]), []).append(row)
        for run, rows in by_run.items():
            rows = sorted(rows, key=lambda item: float(item.get("eval_after_episode") or 0.0))
            x = [float(row.get("eval_after_episode") or 0.0) for row in rows]
            y = [float(row.get("collision_penalized_discovery_rate_mean") or 0.0) for row in rows]
            ax.plot(x, y, marker="o", label=short_run_label(run))
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Stochastic eval CPD")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(str(path))
    if transfer_summary:
        path = output / "bc_transfer_cpd_by_beam.png"
        rows = sorted(transfer_summary, key=lambda item: float(item.get("beamwidth_deg") or 0.0))
        labels = [f"B={float(row.get('beamwidth_deg') or 0.0):.0f}" for row in rows]
        values = [float(row.get("collision_penalized_discovery_rate_mean") or 0.0) for row in rows]
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.bar(labels, values, color="#4E79A7")
        ax.set_xlabel("Beamwidth (deg)")
        ax.set_ylabel("Transfer CPD")
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(str(path))

        path = output / "bc_transfer_discovery_collision_by_beam.png"
        discovery = [float(row.get("discovery_rate_mean") or 0.0) for row in rows]
        collision = [float(row.get("collision_count_mean") or 0.0) for row in rows]
        fig, ax1 = plt.subplots(figsize=(6.4, 4.8))
        ax2 = ax1.twinx()
        x = list(range(len(rows)))
        ax1.plot(x, discovery, marker="o", color="#4E79A7", label="Discovery")
        ax2.plot(x, collision, marker="s", color="#E15759", label="Collisions")
        ax1.set_xticks(x, labels)
        ax1.set_xlabel("Beamwidth (deg)")
        ax1.set_ylabel("Transfer discovery")
        ax2.set_ylabel("Transfer collisions")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(str(path))
    return written


def plot_training_curves(rows: list[dict[str, Any]], output: Path, plt: Any) -> list[str]:
    if not rows:
        return []
    written: list[str] = []
    curve_specs = [
        ("bc_training_reward_by_step.png", "episode_return_mean_per_agent", "Mean return per agent", "#4E79A7"),
        ("bc_training_discovery_by_step.png", "discovery_rate", "Training discovery", "#59A14F"),
        ("bc_training_collision_by_step.png", "collision_count", "Training collisions", "#E15759"),
        ("bc_training_bc_loss_by_step.png", "expert_bc_loss", "Expert BC loss", "#F28E2B"),
    ]
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(str(row.get("run", "")), []).append(row)
    complete_runs = [
        run
        for run, group in by_run.items()
        if any(str(row.get("final_model_exists", "")).lower() == "true" for row in group)
    ]
    curve_runs = complete_runs if complete_runs else sorted(by_run)
    palette = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#76B7B2", "#9C755F"]
    for filename, metric, ylabel, color in curve_specs:
        path = output / filename
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        plotted = False
        for index, run in enumerate(sorted(curve_runs)):
            group = by_run[run]
            points = sorted(group, key=lambda item: float(item.get("training_step") or 0.0))
            x = numeric_values(points, "training_step")
            y = numeric_values(points, metric)
            if len(x) != len(y) or not x:
                continue
            ax.plot(x, y, color=palette[index % len(palette)], linewidth=1.4, alpha=0.9, label=short_run_label(run))
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(str(path))

    gate_metrics = [
        ("access_gate_backoff_ratio", "Backoff", "#4E79A7"),
        ("access_gate_normal_ratio", "Normal", "#59A14F"),
        ("access_gate_aggressive_ratio", "Aggressive", "#E15759"),
    ]
    path = output / "bc_training_gate_ratios_by_step.png"
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plotted = False
    selected_runs = complete_runs[:1] if complete_runs else sorted(by_run)[:1]
    for run in selected_runs:
        points = sorted(by_run[run], key=lambda item: float(item.get("training_step") or 0.0))
        x = numeric_values(points, "training_step")
        if not x:
            continue
        for metric, label, color in gate_metrics:
            y = numeric_values(points, metric)
            if len(y) != len(x):
                continue
            ax.plot(x, y, color=color, linewidth=1.4, label=label)
            plotted = True
    if plotted:
        ax.set_xlabel("Training step")
        ax.set_ylabel("Access-gate ratio")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        written.append(str(path))
    plt.close(fig)
    return written


def build_readme(manifest: dict[str, Any], protocol_summary: list[dict[str, Any]], training_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Trust-Gated Table and Budgeted Gate BC Sweep",
        "",
        f"Generated: {manifest['created_at']}",
        f"Status: {manifest['status']}",
        "",
        "## Protocol Summary",
        "",
    ]
    if protocol_summary:
        lines.append("| Protocol | Episodes | Discovery | Collisions | CPD |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in protocol_summary:
            lines.append(
                "| "
                + f"{row['protocol']} | {row['episodes']} | "
                + f"{fmt(row.get('discovery_rate_mean'))} | {fmt(row.get('collision_count_mean'))} | "
                + f"{fmt(row.get('collision_penalized_discovery_rate_mean'))} |"
            )
    else:
        lines.append("No protocol rows found yet.")
    lines.extend(["", "## Budgeted Expert BC Training", ""])
    if training_rows:
        lines.append("| Run | Episodes | BC weight | Final discovery | Final collisions | Final BC loss | Complete |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for row in training_rows:
            lines.append(
                "| "
                + f"{row['run']} | {row['episodes_completed']} | {fmt(row.get('expert_bc_weight'))} | "
                + f"{fmt(row.get('final_discovery_rate'))} | {fmt(row.get('final_collision_count'))} | "
                + f"{fmt(row.get('final_expert_bc_loss'))} | {row['final_model_exists']} |"
            )
    else:
        lines.append("No training rows found yet.")
    lines.extend(["", "## Transfer Evaluation", ""])
    if manifest.get("bc_transfer_summary_rows", 0):
        lines.append("See `bc_transfer_summary.csv` for N=100 transfer rows generated from final BC checkpoints.")
    else:
        lines.append("No transfer rows found yet.")
    lines.append("")
    return "\n".join(lines)


def completion_status(protocol_summary: list[dict[str, Any]], training_rows: list[dict[str, Any]]) -> str:
    expected_protocols = {
        "trust_gated_isac_tables",
        "improved_rl_isac_tables",
        "wang2025_isac_tables",
        "budgeted_collision_aware_isac",
        "uniform_random",
    }
    done_protocols = {str(row.get("protocol", "")) for row in protocol_summary if int(row.get("episodes", 0)) >= 5}
    complete_runs = sum(1 for row in training_rows if bool(row.get("final_model_exists")))
    if expected_protocols.issubset(done_protocols) and complete_runs >= 3:
        return "complete"
    return "partial"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def parse_bc_weight(run_name: str) -> str:
    marker = "_bc"
    if marker not in run_name:
        return ""
    tail = run_name.split(marker, 1)[1].split("_", 1)[0]
    try:
        return str(float(tail.replace("p", ".")))
    except ValueError:
        return ""


def parse_expert_protocol(run_name: str) -> str:
    if "_budgeted_" in run_name:
        return "budgeted_collision_aware_isac"
    if "_collaware_" in run_name:
        return "collision_aware_isac"
    return ""


def parse_azimuth_cells(run_name: str) -> str:
    if "_B10_" in run_name:
        return "36"
    if "_B15_" in run_name:
        return "24"
    return ""


def parse_beamwidth_from_name(run_name: str) -> str:
    if "_B10_" in run_name:
        return "10"
    if "_B15_" in run_name:
        return "15"
    return ""


def fmt(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def label_for_protocol(protocol: str) -> str:
    return {
        "trust_gated_isac_tables": "Trust-gated",
        "improved_rl_isac_tables": "Ours tables",
        "wang2025_isac_tables": "Wang tables",
        "budgeted_collision_aware_isac": "Budgeted",
        "uniform_random": "Random",
    }.get(protocol, protocol)


def short_run_label(run: str) -> str:
    if "_bc" in run:
        return "BC " + run.split("_bc", 1)[1].split("_", 1)[0].replace("p", ".")
    return run[:18]


if __name__ == "__main__":
    main()
