from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
PROTOCOL_ORDER = (
    "uniform_random",
    "skyorbs_like_skip_scan",
    "rl_no_isac",
    "improved_rl_no_isac",
    "improved_rl_isac",
)
PROTOCOL_LABELS = {
    "uniform_random": "Random",
    "skyorbs_like_skip_scan": "SkyOrbs-like",
    "rl_no_isac": "RL no ISAC",
    "improved_rl_no_isac": "Enhanced no ISAC",
    "improved_rl_isac": "Proposed",
}
COLORS = {
    "uniform_random": "#D55E00",
    "skyorbs_like_skip_scan": "#0072B2",
    "rl_no_isac": "#56B4E9",
    "improved_rl_no_isac": "#009E73",
    "improved_rl_isac": "#E69F00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pre-11am paper evidence figures from archived results.")
    parser.add_argument("--round10", default="06_analysis/paper_tables/round10_n100_b10_b15_extra_seeds")
    parser.add_argument("--output", default="06_analysis/paper_tables/pre11_evidence")
    parser.add_argument("--figures", default="06_analysis/paper_figures/pre11_evidence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round10 = Path(args.round10)
    output_dir = Path(args.output)
    figure_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    aggregate = pd.read_csv(round10 / "aggregate_metrics.csv")
    episode = pd.read_csv(round10 / "per_episode_summary.csv")
    edges = pd.read_csv(round10 / "discovered_edges.csv")
    tradeoff = build_tradeoff_table(aggregate)
    tradeoff.to_csv(output_dir / "round10_tradeoff_summary.csv", index=False)
    cumulative = build_cumulative_curves(episode, edges, pd)
    cumulative.to_csv(output_dir / "round10_cumulative_discovery_curves.csv", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "round10": str(round10),
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "figures": write_figures(tradeoff, cumulative, figure_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_tradeoff_table(aggregate):
    df = aggregate[aggregate["protocol"].isin(PROTOCOL_ORDER)].copy()
    df = df[df["beamwidth_deg"].isin([10.0, 15.0])]
    columns = [
        "protocol",
        "beamwidth_deg",
        "n_episodes",
        "discovery_rate_mean",
        "discovery_rate_std",
        "collision_penalized_discovery_rate_mean",
        "collision_penalized_discovery_rate_std",
        "lambda2_mean",
        "lambda2_std",
        "empty_scan_ratio_mean",
        "empty_scan_ratio_std",
        "collision_count_mean",
        "collision_count_std",
    ]
    return df[[column for column in columns if column in df.columns]].sort_values(["beamwidth_deg", "protocol"])


def build_cumulative_curves(episode, edges, pd):
    key_cols = ["protocol", "beamwidth_deg", "base_seed", "case_id", "episode"]
    episode = episode[episode["protocol"].isin(PROTOCOL_ORDER) & episode["beamwidth_deg"].isin([10.0, 15.0])].copy()
    edges = edges[edges["protocol"].isin(PROTOCOL_ORDER) & edges["beamwidth_deg"].isin([10.0, 15.0])].copy()
    max_slot = int(episode["slots"].max()) if "slots" in episode else 600
    slots = list(range(max_slot + 1))
    edge_groups = {key: group for key, group in edges.groupby(key_cols)}
    rows = []
    import numpy as np

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
                    "cumulative_discovered_edges": int(count),
                    "true_edges_seen": true_edges,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(["protocol", "beamwidth_deg", "slot"], as_index=False)
        .agg(
            cumulative_discovery_rate_mean=("cumulative_discovery_rate", "mean"),
            cumulative_discovery_rate_std=("cumulative_discovery_rate", "std"),
            cumulative_discovered_edges_mean=("cumulative_discovered_edges", "mean"),
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


def write_figures(tradeoff, cumulative, figure_dir: Path) -> list[dict]:
    plt = setup_matplotlib()
    figures = []
    for beamwidth in [10.0, 15.0]:
        figures.append(plot_cumulative(cumulative, beamwidth, figure_dir / f"round10_cumulative_discovery_b{int(beamwidth)}.png", plt))
    figures.append(plot_tradeoff_bars(tradeoff, "discovery_rate_mean", "discovery_rate_std", "Discovery rate", figure_dir / "round10_tradeoff_discovery_rate.png", plt))
    figures.append(plot_tradeoff_bars(tradeoff, "collision_penalized_discovery_rate_mean", "collision_penalized_discovery_rate_std", "Collision-penalized discovery", figure_dir / "round10_tradeoff_collision_penalized.png", plt))
    figures.append(plot_tradeoff_bars(tradeoff, "lambda2_mean", "lambda2_std", r"$\lambda_2$", figure_dir / "round10_tradeoff_lambda2.png", plt))
    figures.append(plot_tradeoff_bars(tradeoff, "empty_scan_ratio_mean", "empty_scan_ratio_std", "Empty-scan ratio", figure_dir / "round10_tradeoff_empty_scan.png", plt))
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


def plot_tradeoff_bars(tradeoff, metric: str, err_metric: str, ylabel: str, path: Path, plt) -> dict:
    import numpy as np

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(PROTOCOL_ORDER), dtype=float)
    width = 0.36
    for offset, beamwidth, color in [(-width / 2, 10.0, "#0072B2"), (width / 2, 15.0, "#E69F00")]:
        rows = tradeoff[tradeoff["beamwidth_deg"] == beamwidth].set_index("protocol")
        values = [float(rows.loc[p, metric]) if p in rows.index else 0.0 for p in PROTOCOL_ORDER]
        errs = [float(rows.loc[p, err_metric]) if p in rows.index and err_metric in rows else 0.0 for p in PROTOCOL_ORDER]
        ax.bar(x + offset, values, yerr=errs, width=width, capsize=3, label=f"B={int(beamwidth)} deg", color=color, alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels([PROTOCOL_LABELS[p] for p in PROTOCOL_ORDER], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "tradeoff_bar", "metric": metric}


def write_readme(output_dir: Path, manifest: dict) -> None:
    text = [
        "# Pre-11 Evidence Figures",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Source: `{manifest['round10']}`",
        "",
        "Figures emphasize the three evidence types that are common in ISAC beam-management papers:",
        "cumulative discovery over time, uncertainty-aware summary bars, and collision/overhead-aware tradeoffs.",
        "",
        "The round10 source is an extra-seed backup block, so use these figures as supplementary robustness evidence unless the manuscript text explicitly states this scope.",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
