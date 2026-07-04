from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


FIGSIZE = (6.4, 4.8)
DPI = 300
PROTOCOL_ORDER = (
    "improved_rl_no_isac",
    "ablation_isac_one_slot_delay",
    "improved_rl_isac",
)
COLORS = {
    "improved_rl_no_isac": "#009E73",
    "ablation_isac_one_slot_delay": "#8A6BBE",
    "improved_rl_isac": "#E69F00",
}
METRICS = {
    "discovery": ("discovery_rate_mean", "Discovery rate"),
    "lambda2": ("lambda2_mean", "Algebraic connectivity"),
    "collision_penalized": ("collision_penalized_discovery_rate_mean", "Collision-penalized discovery"),
    "empty_scan": ("empty_scan_ratio_mean", "Empty-scan ratio"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mobility-transfer robustness figures.")
    parser.add_argument("sweep_dirs", nargs="+", help="Directories or aggregate_metrics.csv files.")
    parser.add_argument("--output", default="06_analysis/paper_figures/round5_mobility_transfer")
    parser.add_argument("--node-count", type=int, default=100)
    return parser.parse_args()


def load_frame(paths: Iterable[str | Path]):
    import pandas as pd
    from pandas.errors import PerformanceWarning
    import warnings

    warnings.filterwarnings("ignore", category=PerformanceWarning)

    frames = []
    for item in paths:
        path = Path(item)
        csv_path = path / "aggregate_metrics.csv" if path.is_dir() else path
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame = frame.assign(source=path.name if path.is_dir() else path.parent.name)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No aggregate_metrics.csv files were found.")
    df = pd.concat(frames, ignore_index=True)
    for column in df.columns:
        if column in {"protocol", "mobility_model", "area_scale", "range_mode", "source"}:
            continue
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except (TypeError, ValueError):
            pass
    return add_derived_columns(df)


def add_derived_columns(df):
    if "collision_penalized_discovery_rate_mean" not in df and all(
        column in df for column in ("discovered_edges_mean", "true_edges_seen_mean", "collision_count_mean")
    ):
        denominator = (df["true_edges_seen_mean"].astype(float) + df["collision_count_mean"].astype(float)).clip(lower=1.0)
        df["collision_penalized_discovery_rate_mean"] = df["discovered_edges_mean"].astype(float) / denominator
    return df


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
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def generate_mobility_figures(sweep_dirs: Iterable[str | Path], output_dir: str | Path, node_count: int = 100) -> dict:
    df = load_frame(sweep_dirs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()
    manifest: list[dict] = []

    subset = df[df["node_count"] == node_count].copy() if "node_count" in df else df.copy()
    beamwidths = sorted(float(value) for value in subset["beamwidth_deg"].dropna().unique())
    for beamwidth in beamwidths:
        for metric_key, (metric_col, ylabel) in METRICS.items():
            manifest.append(
                save_grouped_bar(
                    subset,
                    metric_col,
                    ylabel,
                    output / f"mobility_{metric_key}_n{node_count}_b{beam_token(beamwidth)}.png",
                    plt,
                    beamwidth,
                )
            )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "style": {
            "figsize_inches": list(FIGSIZE),
            "aspect_ratio": "4:3",
            "font_family": "Times New Roman with serif fallback",
            "dpi": DPI,
            "palette": COLORS,
        },
        "inputs": [str(Path(path)) for path in sweep_dirs],
        "selection": {"node_count": node_count},
        "counts": {
            "generated": sum(1 for item in manifest if item["status"] == "generated"),
            "skipped": sum(1 for item in manifest if item["status"] == "skipped"),
            "total": len(manifest),
        },
        "figures": manifest,
    }
    (output / "mobility_figure_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output, payload)
    return payload


def save_grouped_bar(df, metric_col: str, ylabel: str, path: Path, plt, beamwidth: float) -> dict:
    import numpy as np

    if metric_col not in df:
        return skipped(path, metric_col, f"missing {metric_col}")
    subset = df[df["beamwidth_deg"].astype(float) == float(beamwidth)].copy()
    subset = subset[subset["protocol"].isin(PROTOCOL_ORDER)]
    if subset.empty:
        return skipped(path, metric_col, "no matching rows")
    grouped = (
        subset.groupby(["mobility_model", "protocol"], as_index=False)[metric_col]
        .mean()
        .sort_values(["mobility_model", "protocol"])
    )
    mobilities = sorted(grouped["mobility_model"].astype(str).unique())
    x = np.arange(len(mobilities), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for idx, protocol in enumerate(PROTOCOL_ORDER):
        series = grouped[grouped["protocol"] == protocol].set_index("mobility_model")[metric_col]
        values = [float(series.get(mobility, 0.0)) for mobility in mobilities]
        offset = (idx - (len(PROTOCOL_ORDER) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            label=label_protocol(protocol),
            color=COLORS.get(protocol, "#999999"),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_title(f"Mobility transfer: {ylabel} (beam={beamwidth:g} deg)")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Mobility model")
    ax.set_xticks(x)
    ax.set_xticklabels([label_mobility(item) for item in mobilities], rotation=18, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return generated(path, metric_col, int(grouped.shape[0]))


def label_protocol(protocol: str) -> str:
    labels = {
        "improved_rl_no_isac": "Enhanced-NoISAC",
        "ablation_isac_one_slot_delay": "One-slot delay",
        "improved_rl_isac": "Enhanced+ISAC",
    }
    return labels.get(protocol, protocol)


def label_mobility(mobility: str) -> str:
    labels = {
        "gauss_markov": "Gauss-Markov",
        "random_walk": "Random walk",
        "random_direction": "Random direction",
        "random_waypoint": "Random waypoint",
    }
    return labels.get(str(mobility), str(mobility).replace("_", " ").title())


def beam_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def generated(path: Path, metric: str, rows_used: int) -> dict:
    return {
        "status": "generated",
        "metric": metric,
        "filename": path.name,
        "path": str(path.resolve()),
        "rows_used": rows_used,
        "figsize_inches": list(FIGSIZE),
        "pixel_size": [int(FIGSIZE[0] * DPI), int(FIGSIZE[1] * DPI)],
    }


def skipped(path: Path, metric: str, reason: str) -> dict:
    return {
        "status": "skipped",
        "metric": metric,
        "filename": path.name,
        "path": str(path.resolve()),
        "reason": reason,
        "figsize_inches": list(FIGSIZE),
    }


def write_readme(output: Path, payload: dict) -> None:
    lines = [
        "# Mobility Transfer Figures",
        "",
        f"Generated at: {payload['generated_at_utc']}",
        f"Generated figures: {payload['counts']['generated']}",
        "",
        "## Generated",
        "",
    ]
    generated_items = [item for item in payload["figures"] if item["status"] == "generated"]
    if generated_items:
        for item in generated_items:
            lines.append(f"- `{item['filename']}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Skipped", ""])
    skipped_items = [item for item in payload["figures"] if item["status"] == "skipped"]
    if skipped_items:
        for item in skipped_items:
            lines.append(f"- `{item['filename']}`: {item['reason']}")
    else:
        lines.append("- None")
    lines.append("")
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = generate_mobility_figures(args.sweep_dirs, args.output, args.node_count)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
