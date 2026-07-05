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
    "skyorbs_like_skip_scan",
    "improved_rl_no_isac",
    "improved_rl_isac",
    "collision_aware_isac",
)
PROTOCOL_LABELS = {
    "uniform_random": "Random",
    "skyorbs_like_skip_scan": "SkyOrbs-like",
    "improved_rl_no_isac": "Enhanced no ISAC",
    "improved_rl_isac": "Proposed",
    "collision_aware_isac": "Collision-aware",
}
COLORS = {
    "uniform_random": "#6C757D",
    "skyorbs_like_skip_scan": "#56B4E9",
    "improved_rl_no_isac": "#009E73",
    "improved_rl_isac": "#E69F00",
    "collision_aware_isac": "#D55E00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build round15 3000-slot long-horizon evidence tables and figures.")
    parser.add_argument("--source", default="05_simulation/results_raw/round15_long_horizon_3000slot_n100_b10_b15")
    parser.add_argument("--output", default="06_analysis/paper_tables/round15_long_horizon_3000slot")
    parser.add_argument("--figures", default="06_analysis/paper_figures/round15_long_horizon_3000slot")
    parser.add_argument("--short-source", default="06_analysis/paper_tables/round13_collision_energy_10seed")
    parser.add_argument("--tag", default="round15")
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
    horizon = build_horizon_summary(episode, pd)
    horizon_comparison = build_horizon_comparison(horizon, Path(args.short_source), pd)

    tag = str(args.tag)
    summary.to_csv(output_dir / f"{tag}_endpoint_summary.csv", index=False)
    paired.to_csv(output_dir / f"{tag}_paired_delta_summary.csv", index=False)
    cumulative.to_csv(output_dir / f"{tag}_cumulative_discovery_curves.csv", index=False)
    horizon.to_csv(output_dir / f"{tag}_horizon_summary.csv", index=False)
    horizon_comparison.to_csv(output_dir / f"{tag}_horizon_comparison_600_vs_3000.csv", index=False)

    for name in ("aggregate_metrics.csv", "per_episode_summary.csv", "manifest.json", "README.md"):
        src = source / name
        if src.exists():
            shutil.copyfile(src, output_dir / name)

    figures = write_figures(summary, paired, cumulative, horizon_comparison, figure_dir, tag)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source),
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "short_source": str(args.short_source),
        "figures": figures,
        "tag": tag,
        "scope": "N=100, B=10/15 deg, 3000 slots, 5 ms slot, ten paired seeds, single-hop range.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, manifest, summary, paired)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_endpoint_summary(aggregate):
    keep = [
        "protocol",
        "beamwidth_deg",
        "n_episodes",
        "slots_mean",
        "slot_duration_ms",
        "discovery_rate_mean",
        "discovery_rate_std",
        "collision_penalized_discovery_rate_mean",
        "collision_penalized_discovery_rate_std",
        "mean_delay_censored_mean",
        "mean_delay_censored_std",
        "p95_delay_censored_mean",
        "p95_delay_censored_std",
        "empty_scan_ratio_mean",
        "empty_scan_ratio_std",
        "lambda2_mean",
        "lambda2_std",
        "collision_count_mean",
        "collision_count_std",
        "discoveries_per_joule_mean",
        "discoveries_per_joule_std",
        "energy_per_discovery_censored_j_mean",
        "energy_per_discovery_censored_j_std",
        "true_edges_seen_mean",
        "discovered_edges_mean",
    ]
    df = aggregate[aggregate["protocol"].isin(PROTOCOL_ORDER)].copy()
    df = df[[col for col in keep if col in df.columns]].sort_values(["beamwidth_deg", "protocol"])
    for metric in (
        "discovery_rate",
        "collision_penalized_discovery_rate",
        "mean_delay_censored",
        "p95_delay_censored",
        "empty_scan_ratio",
        "lambda2",
        "collision_count",
        "discoveries_per_joule",
        "energy_per_discovery_censored_j",
    ):
        mean = f"{metric}_mean"
        std = f"{metric}_std"
        if mean in df.columns and std in df.columns:
            df[f"{metric}_ci95"] = 1.96 * df[std] / df["n_episodes"].pow(0.5)
    return df


def build_paired_delta_table(episode, pd):
    pair_cols = ["scenario_seed", "node_count", "beamwidth_deg", "area_scale", "mobility_model"]
    treatments = ("improved_rl_isac", "collision_aware_isac")
    controls = ("uniform_random", "skyorbs_like_skip_scan", "improved_rl_no_isac", "improved_rl_isac")
    metrics = (
        "discovery_rate",
        "collision_penalized_discovery_rate",
        "mean_delay_censored",
        "empty_scan_ratio",
        "lambda2",
        "collision_count",
    )
    rows = []
    for beamwidth, beam_df in episode.groupby("beamwidth_deg"):
        for treatment in treatments:
            treatment_df = beam_df[beam_df["protocol"] == treatment]
            for control in controls:
                if treatment == control:
                    continue
                control_df = beam_df[beam_df["protocol"] == control]
                merged = treatment_df.merge(control_df, on=pair_cols, suffixes=("_treatment", "_control"))
                if merged.empty:
                    continue
                for metric in metrics:
                    if f"{metric}_treatment" not in merged.columns or f"{metric}_control" not in merged.columns:
                        continue
                    delta = merged[f"{metric}_treatment"] - merged[f"{metric}_control"]
                    rows.append(
                        {
                            "beamwidth_deg": float(beamwidth),
                            "treatment": treatment,
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
    return pd.DataFrame(rows).sort_values(["beamwidth_deg", "treatment", "control", "metric"])


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
                    "time_s": float(slot) * float(summary["slot_duration_ms"].iloc[0]) / 1000.0,
                    "cumulative_discovery_rate": float(count) / true_edges,
                }
            )
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["protocol", "beamwidth_deg", "slot", "time_s"], as_index=False)
        .agg(
            cumulative_discovery_rate_mean=("cumulative_discovery_rate", "mean"),
            cumulative_discovery_rate_std=("cumulative_discovery_rate", "std"),
            n=("cumulative_discovery_rate", "count"),
        )
        .fillna({"cumulative_discovery_rate_std": 0.0})
    )
    grouped["cumulative_discovery_rate_ci95"] = 1.96 * grouped["cumulative_discovery_rate_std"] / grouped["n"].pow(0.5)
    return grouped


def build_horizon_summary(episode, pd):
    grouped = (
        episode.groupby(["protocol", "beamwidth_deg", "slots", "slot_duration_ms"], as_index=False)
        .agg(
            n=("discovery_rate", "count"),
            discovery_rate_mean=("discovery_rate", "mean"),
            discovery_rate_std=("discovery_rate", "std"),
            collision_penalized_discovery_rate_mean=("collision_penalized_discovery_rate", "mean"),
            lambda2_mean=("lambda2", "mean"),
            mean_delay_censored_mean=("mean_delay_censored", "mean"),
            true_edges_seen_mean=("true_edges_seen", "mean"),
        )
        .fillna(0.0)
    )
    grouped["duration_s"] = grouped["slots"] * grouped["slot_duration_ms"] / 1000.0
    grouped["discovery_rate_ci95"] = 1.96 * grouped["discovery_rate_std"] / grouped["n"].pow(0.5)
    return grouped.sort_values(["beamwidth_deg", "protocol"])


def build_horizon_comparison(long_horizon, short_source: Path, pd):
    short_episode_path = short_source / "per_episode_summary.csv"
    if not short_episode_path.exists():
        return long_horizon.assign(horizon_label="3000 slots", source="round15_long_horizon")
    short_episode = pd.read_csv(short_episode_path)
    common_protocols = ["uniform_random", "improved_rl_no_isac", "improved_rl_isac", "collision_aware_isac"]
    short_episode = short_episode[
        short_episode["protocol"].isin(common_protocols) & short_episode["beamwidth_deg"].isin([10.0, 15.0])
    ].copy()
    short = build_horizon_summary(short_episode, pd)
    short = short[short["protocol"].isin(common_protocols)].copy()
    short["horizon_label"] = "600 slots"
    short["source"] = str(short_source)
    long = long_horizon[long_horizon["protocol"].isin(common_protocols)].copy()
    long["horizon_label"] = "3000 slots"
    long["source"] = "round15_long_horizon"
    comparison = pd.concat([short, long], ignore_index=True, sort=False)
    comparison["duration_s"] = comparison["slots"] * comparison["slot_duration_ms"] / 1000.0
    return comparison.sort_values(["beamwidth_deg", "protocol", "slots"])


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


def write_figures(summary, paired, cumulative, horizon_comparison, figure_dir: Path, tag: str) -> list[dict]:
    plt = setup_matplotlib()
    figures = []
    for beamwidth in sorted(cumulative["beamwidth_deg"].unique()):
        figures.append(plot_cumulative(cumulative, float(beamwidth), figure_dir / f"{tag}_cumulative_discovery_b{int(beamwidth)}.png", plt))
    figures.append(plot_bar(summary, "discovery_rate_mean", "discovery_rate_ci95", "Discovery rate", figure_dir / f"{tag}_discovery_rate.png", plt))
    figures.append(
        plot_bar(
            summary,
            "collision_penalized_discovery_rate_mean",
            "collision_penalized_discovery_rate_ci95",
            "Collision-penalized discovery",
            figure_dir / f"{tag}_collision_penalized.png",
            plt,
        )
    )
    figures.append(plot_bar(summary, "mean_delay_censored_mean", "mean_delay_censored_ci95", "Mean censored delay (slots)", figure_dir / f"{tag}_mean_delay.png", plt))
    figures.append(plot_bar(summary, "empty_scan_ratio_mean", "empty_scan_ratio_ci95", "Empty-scan ratio", figure_dir / f"{tag}_empty_scan_ratio.png", plt))
    figures.append(plot_bar(summary, "lambda2_mean", "lambda2_ci95", r"$\lambda_2$", figure_dir / f"{tag}_lambda2.png", plt))
    figures.append(plot_paired_delta(paired, "improved_rl_isac", "discovery_rate", figure_dir / f"{tag}_proposed_discovery_delta.png", plt))
    figures.append(
        plot_paired_delta(
            paired,
            "collision_aware_isac",
            "collision_penalized_discovery_rate",
            figure_dir / f"{tag}_collision_aware_penalized_delta.png",
            plt,
        )
    )
    if not horizon_comparison.empty:
        figures.append(plot_horizon_comparison(horizon_comparison, figure_dir / f"{tag}_horizon_discovery_600_vs_3000.png", plt))
    return figures


def plot_cumulative(cumulative, beamwidth: float, path: Path, plt) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    subset = cumulative[cumulative["beamwidth_deg"] == beamwidth]
    for protocol in PROTOCOL_ORDER:
        rows = subset[subset["protocol"] == protocol]
        if rows.empty:
            continue
        x = rows["time_s"].to_numpy()
        y = rows["cumulative_discovery_rate_mean"].to_numpy()
        ci = rows["cumulative_discovery_rate_ci95"].to_numpy()
        ax.plot(x, y, label=PROTOCOL_LABELS[protocol], color=COLORS[protocol])
        ax.fill_between(x, y - ci, y + ci, color=COLORS[protocol], alpha=0.10, linewidth=0)
    ax.set_xlabel("Time (s)")
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
    available_protocols = [p for p in PROTOCOL_ORDER if p in set(summary["protocol"])]
    x = np.arange(len(available_protocols), dtype=float)
    beamwidths = sorted(summary["beamwidth_deg"].unique())
    width = 0.72 / max(1, len(beamwidths))
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    for idx, beamwidth in enumerate(beamwidths):
        offset = (idx - (len(beamwidths) - 1) / 2) * width
        rows = summary[summary["beamwidth_deg"] == beamwidth].set_index("protocol")
        values = [float(rows.loc[p, metric]) if p in rows.index and metric in rows.columns else 0.0 for p in available_protocols]
        errs = [float(rows.loc[p, err_metric]) if p in rows.index and err_metric in rows.columns else 0.0 for p in available_protocols]
        ax.bar(x + offset, values, yerr=errs, width=width, capsize=3, label=f"B={int(beamwidth)} deg", color=palette[idx % len(palette)], alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS[p] for p in available_protocols], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "bar", "metric": metric}


def plot_paired_delta(paired, treatment: str, metric: str, path: Path, plt) -> dict:
    import numpy as np

    rows = paired[(paired["treatment"] == treatment) & (paired["metric"] == metric)].copy()
    if treatment == "improved_rl_isac":
        controls = ["uniform_random", "skyorbs_like_skip_scan", "improved_rl_no_isac"]
    else:
        controls = ["improved_rl_isac", "improved_rl_no_isac", "uniform_random"]
    controls = [control for control in controls if control in set(rows["control"])]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    if not controls:
        fig.savefig(path)
        plt.close(fig)
        return {"path": str(path), "type": "paired_delta", "metric": metric, "treatment": treatment, "empty": True}
    x = np.arange(len(controls), dtype=float)
    beamwidths = sorted(rows["beamwidth_deg"].unique())
    width = 0.72 / max(1, len(beamwidths))
    palette = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    for idx, beamwidth in enumerate(beamwidths):
        offset = (idx - (len(beamwidths) - 1) / 2) * width
        subset = rows[rows["beamwidth_deg"] == beamwidth].set_index("control")
        values = [float(subset.loc[c, "delta_mean"]) if c in subset.index else 0.0 for c in controls]
        errs = [float(subset.loc[c, "delta_ci95"]) if c in subset.index else 0.0 for c in controls]
        ax.bar(x + offset, values, yerr=errs, width=width, capsize=3, label=f"B={int(beamwidth)} deg", color=palette[idx % len(palette)], alpha=0.86)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS[c] for c in controls], rotation=20, ha="right")
    ax.set_ylabel(f"{PROTOCOL_LABELS[treatment]} delta")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "paired_delta", "metric": metric, "treatment": treatment}


def plot_horizon_comparison(comparison, path: Path, plt) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    protocols = ["uniform_random", "improved_rl_no_isac", "improved_rl_isac", "collision_aware_isac"]
    for ax, beamwidth in zip(axes, [10.0, 15.0]):
        subset = comparison[comparison["beamwidth_deg"] == beamwidth]
        for protocol in protocols:
            rows = subset[subset["protocol"] == protocol].sort_values("duration_s")
            if rows.empty:
                continue
            ax.errorbar(
                rows["duration_s"].to_numpy(),
                rows["discovery_rate_mean"].to_numpy(),
                yerr=rows["discovery_rate_ci95"].to_numpy() if "discovery_rate_ci95" in rows.columns else None,
                marker="o",
                capsize=3,
                color=COLORS[protocol],
                label=PROTOCOL_LABELS[protocol],
            )
        ax.set_title(f"B={int(beamwidth)} deg")
        ax.set_xlabel("Discovery window (s)")
        ax.set_xticks([3, 15])
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylabel("Discovery rate")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "horizon_comparison", "metric": "discovery_rate"}


def write_readme(output_dir: Path, manifest: dict, summary, paired) -> None:
    lines = [
        "# Round15 Long-Horizon 3000-Slot Evidence",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Source: `{manifest['source']}`",
        f"- Scope: {manifest['scope']}",
        "",
        "This run checks whether the earlier 600-slot, 3-second horizon is too short.",
        "At 5 ms per slot, 3000 slots correspond to 15 seconds.",
        "",
        "Endpoint summary:",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- B={float(row['beamwidth_deg']):g}, {row['protocol']}: "
            f"DR={float(row.get('discovery_rate_mean', 0.0)):.4f}, "
            f"CP-DR={float(row.get('collision_penalized_discovery_rate_mean', 0.0)):.4f}, "
            f"lambda2={float(row.get('lambda2_mean', 0.0)):.4f}"
        )
    lines.extend(
        [
            "",
            "Interpretation rule: use this block to discuss finite-horizon sensitivity.",
        "It should not replace the 600-slot stress result; it shows whether the method ordering persists when the access window is extended.",
        "The companion horizon-comparison table merges the round13 600-slot ten-seed block with this 3000-slot block for common protocols.",
    ]
    )
    if not paired.empty:
        key = paired[
            (paired["treatment"] == "improved_rl_isac")
            & (paired["control"].isin(["uniform_random", "skyorbs_like_skip_scan", "improved_rl_no_isac"]))
            & (paired["metric"] == "discovery_rate")
        ]
        if not key.empty:
            lines.append("")
            lines.append("Paired discovery-rate deltas for proposed:")
            for _, row in key.iterrows():
                lines.append(
                    f"- B={float(row['beamwidth_deg']):g}, vs {row['control']}: "
                    f"delta={float(row['delta_mean']):.4f} "
                    f"({int(row['positive_pairs'])}/{int(row['n_pairs'])} positive pairs)"
                )
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
