from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
PROTOCOL_ORDER = (
    "uniform_random",
    "improved_rl_no_isac",
    "ablation_isac_no_candidate_set",
    "ablation_isac_one_slot_delay",
    "improved_rl_isac",
)
PROTOCOL_LABELS = {
    "uniform_random": "Random",
    "improved_rl_no_isac": "Enhanced no ISAC",
    "ablation_isac_no_candidate_set": "No candidate set",
    "ablation_isac_one_slot_delay": "One-slot delay",
    "improved_rl_isac": "Proposed",
}
COLORS = {
    "uniform_random": "#6C757D",
    "improved_rl_no_isac": "#009E73",
    "ablation_isac_no_candidate_set": "#56B4E9",
    "ablation_isac_one_slot_delay": "#0072B2",
    "improved_rl_isac": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build round11 paired-seed stability tables and figures.")
    parser.add_argument("--source", default="05_simulation/results_raw/round11_paired_seed_campaign_main")
    parser.add_argument("--output", default="06_analysis/paper_tables/round11_paired_seed_campaign_main")
    parser.add_argument("--figures", default="06_analysis/paper_figures/round11_paired_seed_campaign_main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output)
    figure_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    aggregate = pd.read_csv(source / "aggregate_metrics.csv")
    episode = pd.read_csv(source / "per_episode_summary.csv")
    edges = pd.read_csv(source / "discovered_edges.csv")

    summary = build_endpoint_summary(aggregate)
    paired = build_paired_delta_table(episode, pd)
    cumulative = build_cumulative_curves(episode, edges, pd)

    summary.to_csv(output_dir / "round11_endpoint_summary.csv", index=False)
    paired.to_csv(output_dir / "round11_paired_delta_summary.csv", index=False)
    cumulative.to_csv(output_dir / "round11_cumulative_discovery_curves.csv", index=False)

    for name in ("aggregate_metrics.csv", "per_episode_summary.csv", "manifest.json", "README.md"):
        shutil.copyfile(source / name, output_dir / name)

    figures = write_figures(summary, paired, cumulative, figure_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "figures": figures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_endpoint_summary(aggregate):
    df = aggregate[aggregate["protocol"].isin(PROTOCOL_ORDER)].copy()
    keep = [
        "protocol",
        "beamwidth_deg",
        "n_episodes",
        "discovery_rate_mean",
        "discovery_rate_std",
        "collision_penalized_discovery_rate_mean",
        "collision_penalized_discovery_rate_std",
        "empty_scan_ratio_mean",
        "empty_scan_ratio_std",
        "lambda2_mean",
        "lambda2_std",
        "collision_count_mean",
        "collision_count_std",
    ]
    df = df[[col for col in keep if col in df.columns]].sort_values(["beamwidth_deg", "protocol"])
    for metric in (
        "discovery_rate",
        "collision_penalized_discovery_rate",
        "empty_scan_ratio",
        "lambda2",
        "collision_count",
    ):
        mean = f"{metric}_mean"
        std = f"{metric}_std"
        if mean in df.columns and std in df.columns:
            df[f"{metric}_ci95"] = 1.96 * df[std] / df["n_episodes"].pow(0.5)
    return df


def build_paired_delta_table(episode, pd):
    metrics = (
        "discovery_rate",
        "collision_penalized_discovery_rate",
        "empty_scan_ratio",
        "lambda2",
        "collision_count",
    )
    controls = (
        "uniform_random",
        "improved_rl_no_isac",
        "ablation_isac_no_candidate_set",
        "ablation_isac_one_slot_delay",
    )
    pair_cols = ["scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"]
    rows = []
    for beamwidth, beam_df in episode.groupby("beamwidth_deg"):
        treatment = beam_df[beam_df["protocol"] == "improved_rl_isac"]
        for control in controls:
            control_df = beam_df[beam_df["protocol"] == control]
            merged = treatment.merge(control_df, on=pair_cols, suffixes=("_treatment", "_control"))
            for metric in metrics:
                delta = merged[f"{metric}_treatment"] - merged[f"{metric}_control"]
                rows.append(
                    {
                        "beamwidth_deg": float(beamwidth),
                        "treatment": "improved_rl_isac",
                        "control": control,
                        "metric": metric,
                        "n_pairs": int(delta.shape[0]),
                        "treatment_mean": float(merged[f"{metric}_treatment"].mean()),
                        "control_mean": float(merged[f"{metric}_control"].mean()),
                        "delta_mean": float(delta.mean()),
                        "delta_std": float(delta.std(ddof=1)) if delta.shape[0] > 1 else 0.0,
                        "delta_ci95": float(1.96 * delta.std(ddof=1) / (delta.shape[0] ** 0.5)) if delta.shape[0] > 1 else 0.0,
                        "positive_pairs": int((delta > 0).sum()),
                        "negative_pairs": int((delta < 0).sum()),
                        "zero_pairs": int((delta == 0).sum()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["beamwidth_deg", "control", "metric"])


def build_cumulative_curves(episode, edges, pd):
    import numpy as np

    key_cols = ["protocol", "beamwidth_deg", "base_seed", "case_id", "episode"]
    max_slot = int(episode["slots"].max())
    slots = list(range(max_slot + 1))
    edge_groups = {key: group for key, group in edges.groupby(key_cols)}
    rows = []
    for key, summary in episode.groupby(key_cols):
        true_edges = float(summary["true_edges_seen"].iloc[0])
        if true_edges <= 0:
            continue
        discovered = edge_groups.get(key)
        discovery_slots = (
            np.sort(pd.to_numeric(discovered["discovery_slot"], errors="coerce").dropna().to_numpy())
            if discovered is not None
            else np.asarray([], dtype=float)
        )
        protocol, beamwidth, base_seed, case_id, episode_idx = key
        counts = np.searchsorted(discovery_slots, slots, side="right")
        for slot, count in zip(slots, counts):
            rows.append(
                {
                    "protocol": protocol,
                    "beamwidth_deg": float(beamwidth),
                    "base_seed": int(base_seed),
                    "case_id": int(case_id),
                    "episode": int(episode_idx),
                    "slot": int(slot),
                    "cumulative_discovery_rate": float(count) / true_edges,
                }
            )
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["protocol", "beamwidth_deg", "slot"], as_index=False)
        .agg(
            cumulative_discovery_rate_mean=("cumulative_discovery_rate", "mean"),
            cumulative_discovery_rate_std=("cumulative_discovery_rate", "std"),
            n=("cumulative_discovery_rate", "count"),
        )
        .fillna({"cumulative_discovery_rate_std": 0.0})
    )
    grouped["cumulative_discovery_rate_ci95"] = 1.96 * grouped["cumulative_discovery_rate_std"] / grouped["n"].pow(0.5)
    return grouped


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
        }
    )
    return plt


def write_figures(summary, paired, cumulative, figure_dir: Path) -> list[dict]:
    plt = setup_matplotlib()
    figures = []
    for beamwidth in (10.0, 15.0):
        figures.append(plot_cumulative(cumulative, beamwidth, figure_dir / f"round11_cumulative_discovery_b{int(beamwidth)}.png", plt))
    figures.append(plot_bar(summary, "discovery_rate_mean", "discovery_rate_ci95", "Discovery rate", figure_dir / "round11_discovery_rate.png", plt))
    figures.append(plot_bar(summary, "collision_penalized_discovery_rate_mean", "collision_penalized_discovery_rate_ci95", "Collision-penalized discovery", figure_dir / "round11_collision_penalized.png", plt))
    figures.append(plot_bar(summary, "lambda2_mean", "lambda2_ci95", r"$\lambda_2$", figure_dir / "round11_lambda2.png", plt))
    figures.append(plot_paired_delta(paired, figure_dir / "round11_paired_discovery_delta.png", plt))
    return figures


def plot_cumulative(cumulative, beamwidth: float, path: Path, plt) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    subset = cumulative[cumulative["beamwidth_deg"] == beamwidth]
    for protocol in PROTOCOL_ORDER:
        rows = subset[subset["protocol"] == protocol]
        if rows.empty:
            continue
        x = rows["slot"].to_numpy()
        y = rows["cumulative_discovery_rate_mean"].to_numpy()
        ci = rows["cumulative_discovery_rate_ci95"].to_numpy()
        ax.plot(x, y, label=PROTOCOL_LABELS[protocol], color=COLORS[protocol])
        ax.fill_between(x, y - ci, y + ci, color=COLORS[protocol], alpha=0.10, linewidth=0)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Cumulative discovery rate")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "cumulative", "beamwidth_deg": beamwidth}


def plot_bar(summary, metric: str, err_metric: str, ylabel: str, path: Path, plt) -> dict:
    import numpy as np

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(PROTOCOL_ORDER), dtype=float)
    width = 0.36
    for offset, beamwidth, color in [(-width / 2, 10.0, "#0072B2"), (width / 2, 15.0, "#E69F00")]:
        rows = summary[summary["beamwidth_deg"] == beamwidth].set_index("protocol")
        values = [float(rows.loc[p, metric]) if p in rows.index else 0.0 for p in PROTOCOL_ORDER]
        errs = [float(rows.loc[p, err_metric]) if p in rows.index and err_metric in rows.columns else 0.0 for p in PROTOCOL_ORDER]
        ax.bar(x + offset, values, yerr=errs, width=width, capsize=3, label=f"B={int(beamwidth)} deg", color=color, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS[p] for p in PROTOCOL_ORDER], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "bar", "metric": metric}


def plot_paired_delta(paired, path: Path, plt) -> dict:
    import numpy as np

    rows = paired[(paired["metric"] == "discovery_rate") & (paired["control"] != "uniform_random")].copy()
    controls = ["improved_rl_no_isac", "ablation_isac_no_candidate_set", "ablation_isac_one_slot_delay"]
    x = np.arange(len(controls), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for offset, beamwidth, color in [(-width / 2, 10.0, "#0072B2"), (width / 2, 15.0, "#E69F00")]:
        subset = rows[rows["beamwidth_deg"] == beamwidth].set_index("control")
        values = [float(subset.loc[c, "delta_mean"]) if c in subset.index else 0.0 for c in controls]
        errs = [float(subset.loc[c, "delta_ci95"]) if c in subset.index else 0.0 for c in controls]
        ax.bar(x + offset, values, yerr=errs, width=width, capsize=3, label=f"B={int(beamwidth)} deg", color=color, alpha=0.86)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS[c] for c in controls], rotation=20, ha="right")
    ax.set_ylabel("Discovery-rate delta vs proposed")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "paired_delta", "metric": "discovery_rate"}


def write_readme(output_dir: Path, manifest: dict) -> None:
    text = [
        "# Round11 Paired-Seed Campaign",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Source: `{manifest['source']}`",
        "",
        "This focused five-seed block evaluates N=100, B=10/B=15, Gauss-Markov mobility, 600 slots, and paired scenario seeds.",
        "It is designed to strengthen the main evidence chain without starting a broad Cartesian sweep.",
        "",
        "Use these results as stability and mechanism evidence. They complement, rather than replace, the original round3 main tables unless the manuscript is explicitly updated.",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
