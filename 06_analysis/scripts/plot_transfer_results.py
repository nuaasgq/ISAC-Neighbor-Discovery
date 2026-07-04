from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

FIGSIZE = (6.4, 4.8)
DPI = 300
COLORS = {
    "skyorbs_like_skip_scan": "#0072B2",
    "uniform_random": "#D55E00",
    "rl_no_isac": "#999999",
    "improved_rl_no_isac": "#009E73",
    "improved_rl_isac": "#E69F00",
    "ablation_isac_one_slot_delay": "#8A6BBE",
}
CONTINUOUS_CMAPS = {
    "sequential": "viridis",
    "diverging": "coolwarm",
}
PROTOCOL_ORDER = (
    "skyorbs_like_skip_scan",
    "uniform_random",
    "rl_no_isac",
    "improved_rl_no_isac",
    "ablation_isac_one_slot_delay",
    "improved_rl_isac",
)
METRICS = (
    ("discovery_rate_mean", "Discovery rate", "discovery_rate"),
    ("mean_discovery_delay_mean", "Mean delay (slots)", "mean_delay"),
    ("p95_discovery_delay_mean", "P95 delay (slots)", "p95_delay"),
    ("empty_scan_ratio_mean", "Empty-scan ratio", "empty_scan"),
    ("lambda2_mean", "Algebraic connectivity", "lambda2"),
    ("discovery_per_scan_action_mean", "Discoveries per scan action", "discovery_per_scan"),
    ("collision_penalized_discovery_rate_mean", "Collision-penalized discovery rate", "collision_penalized_discovery"),
    ("collision_normalized_efficiency_mean", "Collision-normalized efficiency", "collision_efficiency"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot node/beamwidth transfer sweep figures.")
    parser.add_argument("sweep_dirs", nargs="+", help="Directories with aggregate_metrics.csv.")
    parser.add_argument("--output", default="06_analysis/paper_figures/transfer_round1")
    return parser.parse_args()


def load_frame(paths: Iterable[str | Path]):
    import pandas as pd
    from pandas.errors import PerformanceWarning
    import warnings

    warnings.filterwarnings("ignore", category=PerformanceWarning)

    frames = []
    for path in paths:
        root = Path(path)
        csv_path = root / "aggregate_metrics.csv" if root.is_dir() else root
        frame = pd.read_csv(csv_path)
        frame = frame.assign(source=root.name if root.is_dir() else csv_path.parent.name)
        frames.append(frame)
    if not frames:
        raise ValueError("No sweep tables supplied.")
    return pd.concat(frames, ignore_index=True)


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )
    return plt


def protocol_sort_key(protocol: str) -> tuple[int, str]:
    try:
        return PROTOCOL_ORDER.index(protocol), protocol
    except ValueError:
        return len(PROTOCOL_ORDER), protocol


def save_line_plot(df, x_col: str, fixed_col: str, fixed_value, metric: str, ylabel: str, title: str, path: Path) -> dict:
    plt = setup_matplotlib()
    fig, ax = plt.subplots()
    subset = df[df[fixed_col] == fixed_value].copy()
    for protocol in sorted(subset["protocol"].unique(), key=protocol_sort_key):
        rows = subset[subset["protocol"] == protocol].sort_values(x_col)
        if rows.empty or metric not in rows:
            continue
        ax.plot(
            rows[x_col],
            rows[metric],
            marker="o",
            label=label_protocol(protocol),
            color=COLORS.get(protocol),
        )
    ax.set_title(title)
    ax.set_xlabel(label_axis(x_col))
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "status": "generated",
        "path": str(path),
        "metric": metric,
        "x": x_col,
        "fixed": fixed_col,
        "fixed_value": fixed_value,
        "figsize_inches": list(FIGSIZE),
        "pixel_size": [int(FIGSIZE[0] * DPI), int(FIGSIZE[1] * DPI)],
    }


def save_area_scale_plot(df, metric: str, ylabel: str, path: Path, node_count: int = 100) -> dict:
    plt = setup_matplotlib()
    subset = df[(df["node_count"] == node_count) & (df["protocol"] == "improved_rl_isac")].copy()
    if subset.empty or "area_scale" not in subset or metric not in subset:
        return {"status": "skipped", "reason": "missing area_scale comparison data", "path": str(path)}
    fig, ax = plt.subplots()
    for area_scale, rows in sorted(subset.groupby("area_scale")):
        rows = rows.sort_values("beamwidth_deg")
        ax.plot(rows["beamwidth_deg"], rows[metric], marker="o", label=str(area_scale))
    ax.set_title(f"N={node_count}: density scaling vs fixed area")
    ax.set_xlabel("Beamwidth (deg)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "status": "generated",
        "path": str(path),
        "metric": metric,
        "node_count": node_count,
        "figsize_inches": list(FIGSIZE),
        "pixel_size": [int(FIGSIZE[0] * DPI), int(FIGSIZE[1] * DPI)],
    }


def save_heatmap(df, metric: str, ylabel: str, path: Path, protocol: str = "improved_rl_isac") -> dict:
    import numpy as np

    plt = setup_matplotlib()
    subset = df[df["protocol"] == protocol].copy()
    if subset.empty or metric not in subset:
        return {"status": "skipped", "reason": f"missing {protocol}/{metric}", "path": str(path)}
    pivot = subset.pivot_table(index="node_count", columns="beamwidth_deg", values=metric, aggfunc="mean")
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots()
    image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(f"{label_protocol(protocol)}: {ylabel}")
    ax.set_xlabel("Beamwidth (deg)")
    ax.set_ylabel("Node count")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(value)) for value in pivot.index])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "status": "generated",
        "path": str(path),
        "metric": metric,
        "protocol": protocol,
        "figsize_inches": list(FIGSIZE),
        "pixel_size": [int(FIGSIZE[0] * DPI), int(FIGSIZE[1] * DPI)],
    }


def save_gain_heatmap(df, metric: str, baseline: str, path: Path) -> dict:
    import numpy as np

    plt = setup_matplotlib()
    proposed = df[df["protocol"] == "improved_rl_isac"].copy()
    base = df[df["protocol"] == baseline].copy()
    keys = ["mobility_model", "node_count", "beamwidth_deg"]
    merged = proposed.merge(base, on=keys, suffixes=("_proposed", "_baseline"))
    proposed_col = f"{metric}_proposed"
    baseline_col = f"{metric}_baseline"
    if merged.empty or proposed_col not in merged or baseline_col not in merged:
        return {"status": "skipped", "reason": f"missing gain columns for {metric}", "path": str(path)}
    if "delay" in metric:
        merged["gain"] = (merged[baseline_col] - merged[proposed_col]) / merged[baseline_col].clip(lower=1e-9) * 100.0
        title = f"Delay reduction vs {label_protocol(baseline)} (%)"
    else:
        merged["gain"] = (merged[proposed_col] - merged[baseline_col]) * 100.0
        title = f"Discovery gain vs {label_protocol(baseline)} (pp)"
    pivot = merged.pivot_table(index="node_count", columns="beamwidth_deg", values="gain", aggfunc="mean")
    pivot = pivot.sort_index().reindex(sorted(pivot.columns), axis=1)
    fig, ax = plt.subplots()
    image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set_title(title)
    ax.set_xlabel("Beamwidth (deg)")
    ax.set_ylabel("Node count")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(value)) for value in pivot.index])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Gain")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "status": "generated",
        "path": str(path),
        "metric": metric,
        "baseline": baseline,
        "figsize_inches": list(FIGSIZE),
        "pixel_size": [int(FIGSIZE[0] * DPI), int(FIGSIZE[1] * DPI)],
    }


def label_axis(column: str) -> str:
    if column == "beamwidth_deg":
        return "Beamwidth (deg)"
    if column == "node_count":
        return "Node count"
    return column.replace("_", " ").title()


def label_protocol(protocol: str) -> str:
    labels = {
        "skyorbs_like_skip_scan": "SkyOrbs-like",
        "uniform_random": "Random",
        "rl_no_isac": "Learned no-ISAC",
        "improved_rl_no_isac": "Enhanced no-ISAC",
        "ablation_isac_one_slot_delay": "One-slot delay",
        "improved_rl_isac": "Enhanced ISAC",
    }
    return labels.get(protocol, protocol)


def generate_transfer_figures(sweep_dirs: Iterable[str | Path], output: str | Path) -> dict:
    df = load_frame(sweep_dirs)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    area_scales = sorted(df["area_scale"].dropna().unique()) if "area_scale" in df else ["all"]
    for area_scale in area_scales:
        scale_df = df[df["area_scale"] == area_scale].copy() if "area_scale" in df else df.copy()
        prefix = f"{area_scale}_" if area_scale != "all" else ""
        for metric, ylabel, stem in METRICS:
            if metric not in scale_df:
                manifest.append({"status": "skipped", "metric": metric, "reason": "missing metric"})
                continue
            for node_count in sorted(scale_df["node_count"].unique()):
                if scale_df[scale_df["node_count"] == node_count]["beamwidth_deg"].nunique() > 1:
                    manifest.append(
                        save_line_plot(
                            scale_df,
                            "beamwidth_deg",
                            "node_count",
                            node_count,
                            metric,
                            ylabel,
                            f"{ylabel} vs beamwidth (N={int(node_count)}, {area_scale})",
                            output_dir / f"{prefix}beamwidth_{stem}_n{int(node_count)}.png",
                        )
                    )
            for beamwidth in sorted(scale_df["beamwidth_deg"].unique()):
                if scale_df[scale_df["beamwidth_deg"] == beamwidth]["node_count"].nunique() > 1:
                    beam_token = f"{beamwidth:g}".replace(".", "p")
                    manifest.append(
                        save_line_plot(
                            scale_df,
                            "node_count",
                            "beamwidth_deg",
                            beamwidth,
                            metric,
                            ylabel,
                            f"{ylabel} vs node count (beamwidth={beamwidth:g} deg, {area_scale})",
                            output_dir / f"{prefix}node_count_{stem}_b{beam_token}.png",
                        )
                    )
            if scale_df["node_count"].nunique() > 1 and scale_df["beamwidth_deg"].nunique() > 1:
                manifest.append(save_heatmap(scale_df, metric, ylabel, output_dir / f"{prefix}heatmap_proposed_{stem}.png"))
        if scale_df["node_count"].nunique() > 1 and scale_df["beamwidth_deg"].nunique() > 1:
            manifest.append(
                save_gain_heatmap(scale_df, "discovery_rate_mean", "uniform_random", output_dir / f"{prefix}heatmap_gain_discovery_vs_random.png")
            )
            manifest.append(
                save_gain_heatmap(scale_df, "mean_discovery_delay_mean", "uniform_random", output_dir / f"{prefix}heatmap_gain_delay_vs_random.png")
            )
            manifest.append(
                save_gain_heatmap(
                    scale_df,
                    "discovery_rate_mean",
                    "improved_rl_no_isac",
                    output_dir / f"{prefix}heatmap_gain_discovery_vs_improved_no_isac.png",
                )
            )
    if "area_scale" in df and df["area_scale"].nunique() > 1:
        for metric, ylabel, stem in METRICS:
            if metric in df:
                manifest.append(save_area_scale_plot(df, metric, ylabel, output_dir / f"area_scale_n100_{stem}.png"))
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "style": {
            "figsize_inches": list(FIGSIZE),
            "aspect_ratio": "4:3",
            "font_family": "Times New Roman with serif fallback",
            "dpi": DPI,
            "palette": COLORS,
            "continuous_cmap": CONTINUOUS_CMAPS,
        },
        "inputs": [str(Path(path)) for path in sweep_dirs],
        "counts": {
            "generated": sum(1 for entry in manifest if entry.get("status") == "generated"),
            "skipped": sum(1 for entry in manifest if entry.get("status") == "skipped"),
            "total": len(manifest),
        },
        "figures": manifest,
    }
    payload = to_plain_data(payload)
    (output_dir / "transfer_figure_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        f"# Transfer Figures\n\nGenerated figures: {payload['counts']['generated']}.\n",
        encoding="utf-8",
    )
    return payload


def to_plain_data(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    print(json.dumps(generate_transfer_figures(args.sweep_dirs, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
