"""Analyze the round14 ten-seed main-table stability campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROTOCOL_ORDER = [
    "uniform_random",
    "skyorbs_like_skip_scan",
    "rl_no_isac",
    "improved_rl_no_isac",
    "improved_rl_isac",
]

PROTOCOL_LABELS = {
    "uniform_random": "Random",
    "skyorbs_like_skip_scan": "SkyOrbs-like",
    "rl_no_isac": "Learned no-ISAC",
    "improved_rl_no_isac": "Enhanced no-ISAC",
    "improved_rl_isac": "Enhanced ISAC",
}

METRICS = [
    "discovery_rate",
    "empty_scan_ratio",
    "lambda2",
    "mean_discovery_delay",
    "collision_count",
    "collision_penalized_discovery_rate",
    "discoveries_per_joule",
    "energy_per_discovery_censored_j",
]


def ci95(series: pd.Series) -> float:
    n = int(series.count())
    if n <= 1:
        return 0.0
    return float(1.96 * series.std(ddof=1) / np.sqrt(n))


def build_endpoint_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for protocol in PROTOCOL_ORDER:
        part = episodes[episodes["protocol"] == protocol]
        row = {"protocol": protocol, "label": PROTOCOL_LABELS[protocol], "n_pairs": int(len(part))}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1))
            row[f"{metric}_ci95"] = ci95(part[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_deltas(episodes: pd.DataFrame, treatment: str = "improved_rl_isac") -> pd.DataFrame:
    rows = []
    controls = [p for p in PROTOCOL_ORDER if p != treatment]
    for metric in METRICS:
        pivot = episodes.pivot_table(index="base_seed", columns="protocol", values=metric, aggfunc="first")
        for control in controls:
            deltas = (pivot[treatment] - pivot[control]).dropna()
            rows.append(
                {
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "n_pairs": int(deltas.count()),
                    "treatment_mean": float(pivot[treatment].mean()),
                    "control_mean": float(pivot[control].mean()),
                    "delta_mean": float(deltas.mean()),
                    "delta_std": float(deltas.std(ddof=1)),
                    "delta_ci95": ci95(deltas),
                    "positive_pairs": int((deltas > 0).sum()),
                    "negative_pairs": int((deltas < 0).sum()),
                    "zero_pairs": int((deltas == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
        }
    )
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#E45756"]
    labels = summary["label"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=240)
    for ax, metric, ylabel in [
        (axes[0], "discovery_rate", "Discovery rate"),
        (axes[1], "lambda2", r"Algebraic connectivity $\lambda_2$"),
    ]:
        means = summary[f"{metric}_mean"].to_numpy()
        errs = summary[f"{metric}_ci95"].to_numpy()
        ax.bar(x, means, yerr=errs, color=colors, edgecolor="#222222", linewidth=0.7, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figure_dir / "round14_main_table_discovery_lambda2.png")
    plt.close(fig)


def write_readme(source: Path, output: Path, figure_dir: Path, summary: pd.DataFrame, deltas: pd.DataFrame) -> None:
    proposed = summary[summary["protocol"] == "improved_rl_isac"].iloc[0]
    enhanced_no = summary[summary["protocol"] == "improved_rl_no_isac"].iloc[0]
    random = summary[summary["protocol"] == "uniform_random"].iloc[0]
    sky = summary[summary["protocol"] == "skyorbs_like_skip_scan"].iloc[0]
    rl_no = summary[summary["protocol"] == "rl_no_isac"].iloc[0]
    disc_deltas = deltas[deltas["metric"] == "discovery_rate"]
    lines = [
        "# Round14 Main-Table Ten-Seed Stability Check",
        "",
        f"- Source: `{source}`",
        f"- Figures: `{figure_dir}`",
        "- Setting: N=100, B=10 deg, Gauss-Markov mobility, 600 slots, density scaling, single-hop range.",
        "- Seeds: 20290704, 20291713, 20292722, 20293731, 20294740, 20295749, 20296758, 20297767, 20298776, 20299785.",
        "- Protocols: uniform random, SkyOrbs-like skip-scan, learned no-ISAC, enhanced no-ISAC, enhanced ISAC.",
        "",
        "## Key Result",
        "",
        f"Enhanced ISAC discovery is {proposed['discovery_rate_mean']:.4f} versus random {random['discovery_rate_mean']:.4f}, SkyOrbs-like {sky['discovery_rate_mean']:.4f}, learned no-ISAC {rl_no['discovery_rate_mean']:.4f}, and enhanced no-ISAC {enhanced_no['discovery_rate_mean']:.4f}.",
        f"Enhanced ISAC lambda2 is {proposed['lambda2_mean']:.4f}; all communication-only baselines have mean lambda2 {max(random['lambda2_mean'], sky['lambda2_mean'], rl_no['lambda2_mean'], enhanced_no['lambda2_mean']):.4f}.",
        "",
        "Discovery-rate paired deltas against all four communication-only controls are positive in "
        + ", ".join(f"{r.control}: {int(r.positive_pairs)}/{int(r.n_pairs)}" for r in disc_deltas.itertuples())
        + " paired seeds.",
        "",
        "Use this campaign as a stability check for the main N=100/B=10 baseline table. It does not replace the round13 collision-aware MAC refinement probe.",
        "",
        "## Reproduction",
        "",
        "Regenerate the raw data with `05_simulation/run_transfer_sweep.py` using the raw manifest in the source directory. Then run:",
        "",
        "```powershell",
        "python 06_analysis\\scripts\\analyze_round14_main_table.py --source 05_simulation\\results_raw\\round14_main_table_10seed_n100_b10 --output 06_analysis\\paper_tables\\round14_main_table_10seed_n100_b10 --figures 06_analysis\\paper_figures\\round14_main_table_10seed_n100_b10",
        "```",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("05_simulation/results_raw/round14_main_table_10seed_n100_b10"))
    parser.add_argument("--output", type=Path, default=Path("06_analysis/paper_tables/round14_main_table_10seed_n100_b10"))
    parser.add_argument("--figures", type=Path, default=Path("06_analysis/paper_figures/round14_main_table_10seed_n100_b10"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(args.source / "per_episode_summary.csv")
    summary = build_endpoint_summary(episodes)
    deltas = build_paired_deltas(episodes)
    summary.to_csv(args.output / "round14_endpoint_summary.csv", index=False)
    deltas.to_csv(args.output / "round14_paired_delta_summary.csv", index=False)
    (args.output / "raw_manifest.json").write_text((args.source / "manifest.json").read_text(encoding="utf-8"), encoding="utf-8")
    plot_summary(summary, args.figures)
    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "figures": [str(args.figures / "round14_main_table_discovery_lambda2.png")],
        "protocols": PROTOCOL_ORDER,
        "metrics": METRICS,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(args.source, args.output, args.figures, summary, deltas)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
