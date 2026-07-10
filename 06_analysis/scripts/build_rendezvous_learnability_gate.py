from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "05_simulation" / "results_raw" / "marl_rendezvous_learnability_gate_20260710"
TABLE_DIR = ROOT / "06_analysis" / "paper_tables" / "marl" / "rendezvous_learnability_gate_20260710"
FIGURE_DIR = ROOT / "06_analysis" / "paper_figures" / "marl" / "rendezvous_learnability_gate_20260710"
TRAIN_SEEDS = (20260705, 20260715, 20260725)
METHOD_ORDER = (
    "Uniform random",
    "Wang ISAC tables",
    "Adapter zero",
    "MARL + ISAC adapter",
)
COLORS = ("#7A7A7A", "#D28E2D", "#4C78A8", "#2A9D6F")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def collect_evaluation_rows() -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    inputs: list[Path] = []
    for train_seed in TRAIN_SEEDS:
        sources = (
            (
                "MARL + ISAC adapter",
                RAW_ROOT / f"adapter_explore_screen5_seed{train_seed}" / "eval_episode_metrics.csv",
                train_seed,
            ),
            (
                "Adapter zero",
                RAW_ROOT / f"adapter_zero_eval_seed{train_seed}" / "eval_episode_metrics.csv",
                train_seed,
            ),
            (
                "Uniform random",
                RAW_ROOT / f"protocol_controls_seed{train_seed}" / "uniform_random" / "eval_episode_metrics.csv",
                "",
            ),
            (
                "Wang ISAC tables",
                RAW_ROOT / f"protocol_controls_seed{train_seed}" / "wang2025_isac_tables" / "eval_episode_metrics.csv",
                "",
            ),
        )
        for method, path, source_train_seed in sources:
            inputs.append(path)
            for raw in read_csv(path):
                scenario_seed = int(float(raw.get("seed") or raw.get("scenario_seed") or 0))
                rows.append(
                    {
                        "method": method,
                        "train_seed": source_train_seed,
                        "scenario_seed": scenario_seed,
                        "discovered_edges": int(numeric(raw, "discovered_edges")),
                        "discovery_rate": numeric(raw, "discovery_rate"),
                        "aligned_handshake_opportunities": int(numeric(raw, "aligned_handshake_opportunities")),
                        "handshake_successes": int(numeric(raw, "handshake_successes")),
                        "empty_scan_ratio": numeric(raw, "empty_scan_ratio"),
                        "episode_return_mean_per_agent": numeric(raw, "episode_return_mean_per_agent"),
                    }
                )
    rows.sort(key=lambda row: (int(row["scenario_seed"]), METHOD_ORDER.index(str(row["method"]))))
    return rows, inputs


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = []
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        edges = np.asarray([float(row["discovered_edges"]) for row in selected], dtype=float)
        rates = np.asarray([float(row["discovery_rate"]) for row in selected], dtype=float)
        alignments = np.asarray([float(row["aligned_handshake_opportunities"]) for row in selected], dtype=float)
        successes = np.asarray([float(row["handshake_successes"]) for row in selected], dtype=float)
        empty = np.asarray([float(row["empty_scan_ratio"]) for row in selected], dtype=float)
        edge_std = float(edges.std(ddof=1)) if len(edges) > 1 else 0.0
        t95 = 2.5706 if len(edges) == 6 else 1.96
        half_width = t95 * edge_std / math.sqrt(max(1, len(edges)))
        summary.append(
            {
                "method": method,
                "episodes": len(selected),
                "nonzero_episodes": int(np.count_nonzero(edges > 0.0)),
                "mean_discovered_edges": float(edges.mean()),
                "std_discovered_edges": edge_std,
                "mean_edges_ci95_low": max(0.0, float(edges.mean()) - half_width),
                "mean_edges_ci95_high": float(edges.mean()) + half_width,
                "mean_discovery_rate": float(rates.mean()),
                "mean_alignments": float(alignments.mean()),
                "mean_handshake_successes": float(successes.mean()),
                "mean_empty_scan_ratio": float(empty.mean()),
            }
        )
    return summary


def collect_training_rows() -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    inputs: list[Path] = []
    for train_seed in TRAIN_SEEDS:
        path = RAW_ROOT / f"adapter_explore_screen5_seed{train_seed}" / "episode_metrics.csv"
        inputs.append(path)
        for raw in read_csv(path):
            rows.append(
                {
                    "train_seed": train_seed,
                    "episode": int(numeric(raw, "episode")),
                    "training_step": int(numeric(raw, "training_step")),
                    "episode_return_mean_per_agent": numeric(raw, "episode_return_mean_per_agent"),
                    "rendezvous_beam_hit_rate": numeric(raw, "rendezvous_beam_hit_rate"),
                    "rendezvous_mode_match_rate": numeric(raw, "rendezvous_mode_match_rate"),
                    "rendezvous_joint_action_rate": numeric(raw, "rendezvous_joint_action_rate"),
                    "rendezvous_beam_aux_loss": numeric(raw, "rendezvous_beam_aux_loss"),
                    "rendezvous_role_aux_loss": numeric(raw, "rendezvous_role_aux_loss"),
                    "adapter_beam_score_weight": numeric(raw, "rendezvous_adapter_beam_score_weight"),
                    "adapter_tx_role_weight": numeric(raw, "rendezvous_adapter_tx_role_weight"),
                    "adapter_rx_role_weight": numeric(raw, "rendezvous_adapter_rx_role_weight"),
                    "reciprocal_report_pair_count": int(numeric(raw, "reciprocal_report_pair_count")),
                    "reciprocal_scheduled_pair_count": int(numeric(raw, "reciprocal_scheduled_pair_count")),
                    "reciprocal_actor_pair_count": int(numeric(raw, "reciprocal_actor_pair_count")),
                    "aligned_handshake_opportunities": int(numeric(raw, "aligned_handshake_opportunities")),
                    "discovered_edges": int(numeric(raw, "discovered_edges")),
                }
            )
    return rows, inputs


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.7,
        }
    )


def plot_comparison(rows: list[dict[str, object]], summary: list[dict[str, object]]) -> Path:
    configure_plots()
    fig, ax = plt.subplots(figsize=(8, 6))
    means = np.asarray([float(row["mean_discovered_edges"]) for row in summary])
    stds = np.asarray([float(row["std_discovered_edges"]) for row in summary])
    x = np.arange(len(METHOD_ORDER))
    ax.bar(x, means, yerr=stds, capsize=4, color=COLORS, width=0.66, edgecolor="white", linewidth=0.8)
    rng = np.random.default_rng(20260710)
    for index, method in enumerate(METHOD_ORDER):
        values = [float(row["discovered_edges"]) for row in rows if row["method"] == method]
        jitter = rng.uniform(-0.09, 0.09, size=len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=30, color="#202020", alpha=0.78, zorder=3)
    ax.set_xticks(x, METHOD_ORDER)
    ax.set_ylabel("Discovered edges in 300 slots")
    ax.set_title("N=10, B=10 deg paired learnability gate")
    ax.set_ylim(bottom=0.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = FIGURE_DIR / "rendezvous_gate_method_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def episode_stats(rows: list[dict[str, object]], key: str) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for episode in range(5):
        values.append([float(row[key]) for row in rows if int(row["episode"]) == episode])
    array = np.asarray(values, dtype=float)
    return array.mean(axis=1), array.std(axis=1, ddof=1)


def plot_training(rows: list[dict[str, object]]) -> Path:
    configure_plots()
    episodes = np.arange(1, 6)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    panels = (
        ("rendezvous_beam_hit_rate", "Target-beam hit rate", COLORS[3]),
        ("rendezvous_joint_action_rate", "Joint beam-role rate", COLORS[2]),
        ("adapter_beam_score_weight", "Learned beam-evidence weight", "#8F5AA2"),
        ("rendezvous_beam_aux_loss", "Rendezvous beam auxiliary loss", COLORS[1]),
    )
    for ax, (key, title, color) in zip(axes.flat, panels, strict=True):
        mean, std = episode_stats(rows, key)
        ax.plot(episodes, mean, marker="o", color=color, linewidth=2.0)
        ax.fill_between(episodes, np.maximum(0.0, mean - std), mean + std, color=color, alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel("Training episode")
        ax.set_xticks(episodes)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 1].set_ylim(0.0, 1.0)
    fig.tight_layout()
    path = FIGURE_DIR / "rendezvous_adapter_training_diagnostics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_rows, evaluation_inputs = collect_evaluation_rows()
    training_rows, training_inputs = collect_training_rows()
    summary = summarize(evaluation_rows)
    paired_path = TABLE_DIR / "paired_evaluation_rows.csv"
    summary_path = TABLE_DIR / "method_summary.csv"
    training_path = TABLE_DIR / "training_diagnostics.csv"
    write_csv(paired_path, evaluation_rows)
    write_csv(summary_path, summary)
    write_csv(training_path, training_rows)
    figures = [plot_comparison(evaluation_rows, summary), plot_training(training_rows)]

    full = next(row for row in summary if row["method"] == "MARL + ISAC adapter")
    nonzero_deltas = [
        float(row["discovered_edges"])
        for row in evaluation_rows
        if row["method"] == "MARL + ISAC adapter" and float(row["discovered_edges"]) > 0.0
    ]
    manifest = {
        "scope": "n10_b10_300slot_rendezvous_learnability_gate",
        "training_seeds": list(TRAIN_SEEDS),
        "evaluation_episodes_per_training_seed": 2,
        "method_order": list(METHOD_ORDER),
        "headline": {
            "mean_discovered_edges": full["mean_discovered_edges"],
            "mean_discovery_rate": full["mean_discovery_rate"],
            "nonzero_episodes": full["nonzero_episodes"],
            "total_episodes": full["episodes"],
            "one_sided_exact_sign_p_excluding_ties": 0.5 ** len(nonzero_deltas),
            "interpretation": "learnability gate only; not a paper-level final performance claim",
        },
        "inputs": [{"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for path in evaluation_inputs + training_inputs],
        "outputs": [
            str(path.relative_to(ROOT))
            for path in (paired_path, summary_path, training_path, *figures)
        ],
    }
    manifest_path = TABLE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
