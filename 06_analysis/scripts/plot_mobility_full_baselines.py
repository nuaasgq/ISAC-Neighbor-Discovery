from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
PROTOCOL_ORDER = (
    "uniform_random",
    "skyorbs_like_skip_scan",
    "rl_no_isac",
    "improved_rl_no_isac",
    "ablation_isac_one_slot_delay",
    "improved_rl_isac",
)
COLORS = {
    "uniform_random": "#D55E00",
    "skyorbs_like_skip_scan": "#0072B2",
    "rl_no_isac": "#999999",
    "improved_rl_no_isac": "#009E73",
    "ablation_isac_one_slot_delay": "#8A6BBE",
    "improved_rl_isac": "#E69F00",
}
METRICS = {
    "discovery": ("discovery_rate_mean", "discovery_rate_std", "Discovery rate"),
    "lambda2": ("lambda2_mean", "lambda2_std", "Algebraic connectivity"),
    "empty_scan": ("empty_scan_ratio_mean", "empty_scan_ratio_std", "Empty-scan ratio"),
    "collision_penalized": (
        "collision_penalized_discovery_rate_mean",
        "collision_penalized_discovery_rate_std",
        "Collision-penalized discovery",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot N=100 multi-mobility full-baseline supplement figures.")
    parser.add_argument(
        "--input",
        default="06_analysis/paper_tables/round8_n100_multimobility_full_baseline/combined_aggregate_metrics.csv",
    )
    parser.add_argument("--output", default="06_analysis/paper_figures/round8_n100_multimobility_full_baseline")
    parser.add_argument("--node-count", type=int, default=100)
    return parser.parse_args()


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
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def load_frame(path: str | Path):
    import pandas as pd

    df = pd.read_csv(path)
    for column in df.columns:
        if column in {"source_block", "protocol", "mobility_model"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def generate_figures(input_path: str | Path, output_dir: str | Path, node_count: int = 100) -> dict:
    df = load_frame(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()

    subset = df[df["node_count"].astype(int) == int(node_count)].copy()
    beamwidths = sorted(float(value) for value in subset["beamwidth_deg"].dropna().unique())
    manifest: list[dict] = []
    for beamwidth in beamwidths:
        beam_subset = subset[subset["beamwidth_deg"].astype(float) == beamwidth].copy()
        for metric_key, (mean_col, std_col, ylabel) in METRICS.items():
            filename = f"full_baseline_{metric_key}_n{node_count}_b{beam_token(beamwidth)}.png"
            manifest.append(save_grouped_bar(beam_subset, mean_col, std_col, ylabel, output / filename, plt, beamwidth))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "inputs": [str(Path(input_path).resolve())],
        "style": {
            "figsize_inches": list(FIGSIZE),
            "aspect_ratio": "4:3",
            "font_family": "Times New Roman with serif fallback",
            "dpi": DPI,
            "palette": COLORS,
        },
        "selection": {"node_count": node_count},
        "counts": {
            "generated": sum(1 for item in manifest if item["status"] == "generated"),
            "skipped": sum(1 for item in manifest if item["status"] == "skipped"),
            "total": len(manifest),
        },
        "figures": manifest,
    }
    (output / "full_baseline_figure_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(output, payload)
    return payload


def save_grouped_bar(df, mean_col: str, std_col: str, ylabel: str, path: Path, plt, beamwidth: float) -> dict:
    import numpy as np

    if mean_col not in df:
        return skipped(path, mean_col, f"missing {mean_col}")
    filtered = df[df["protocol"].isin(PROTOCOL_ORDER)].copy()
    if filtered.empty:
        return skipped(path, mean_col, "no matching protocol rows")

    mobilities = [item for item in ("gauss_markov", "random_walk", "random_direction", "random_waypoint") if item in set(filtered["mobility_model"])]
    x = np.arange(len(mobilities), dtype=float)
    width = min(0.12, 0.76 / max(1, len(PROTOCOL_ORDER)))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for idx, protocol in enumerate(PROTOCOL_ORDER):
        rows = filtered[filtered["protocol"] == protocol].set_index("mobility_model")
        values = [float(rows.loc[mobility, mean_col]) if mobility in rows.index else 0.0 for mobility in mobilities]
        if std_col in rows.columns:
            errors = [float(rows.loc[mobility, std_col]) if mobility in rows.index else 0.0 for mobility in mobilities]
        else:
            errors = None
        offset = (idx - (len(PROTOCOL_ORDER) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            capsize=2,
            label=label_protocol(protocol),
            color=COLORS.get(protocol, "#999999"),
            edgecolor="black",
            linewidth=0.4,
            error_kw={"elinewidth": 0.6, "capthick": 0.6},
        )

    ax.set_title(f"N=100 mobility baselines, beam={beamwidth:g} deg")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Mobility model")
    ax.set_xticks(x)
    ax.set_xticklabels([label_mobility(item) for item in mobilities], rotation=16, ha="right")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return generated(path, mean_col, int(filtered.shape[0]))


def label_protocol(protocol: str) -> str:
    labels = {
        "uniform_random": "Random",
        "skyorbs_like_skip_scan": "SkyOrbs-like",
        "rl_no_isac": "Learned no ISAC",
        "improved_rl_no_isac": "Enhanced no ISAC",
        "ablation_isac_one_slot_delay": "ISAC delay",
        "improved_rl_isac": "Enhanced ISAC",
    }
    return labels.get(protocol, protocol)


def label_mobility(mobility: str) -> str:
    labels = {
        "gauss_markov": "Gauss-Markov",
        "random_walk": "Random walk",
        "random_direction": "Random direction",
        "random_waypoint": "Random waypoint",
    }
    return labels.get(mobility, mobility.replace("_", " ").title())


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
        "# N=100 Mobility Full-Baseline Figures",
        "",
        f"Generated at: {payload['generated_at_utc']}",
        f"Generated figures: {payload['counts']['generated']}",
        "",
        "These figures visualize the combined round7/round8 table used to check whether the mobility-boundary result is caused by missing SkyOrbs-like or no-ISAC baselines.",
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
    skipped_items = [item for item in payload["figures"] if item["status"] == "skipped"]
    lines.extend(["", "## Skipped", ""])
    if skipped_items:
        for item in skipped_items:
            lines.append(f"- `{item['filename']}`: {item['reason']}")
    else:
        lines.append("- None")
    output.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = generate_figures(args.input, args.output, args.node_count)
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
