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
    "improved_rl_isac",
)
COLORS = {
    "uniform_random": "#D55E00",
    "skyorbs_like_skip_scan": "#0072B2",
    "rl_no_isac": "#999999",
    "improved_rl_no_isac": "#009E73",
    "improved_rl_isac": "#E69F00",
}
METRICS = {
    "discovery": ("discovery_rate_mean", "discovery_rate_std", "Discovery rate"),
    "empty_scan": ("empty_scan_ratio_mean", "empty_scan_ratio_std", "Empty-scan ratio"),
    "lambda2": ("lambda2_mean", "lambda2_std", "Algebraic connectivity"),
    "collision_penalized": (
        "collision_penalized_discovery_rate_mean",
        "collision_penalized_discovery_rate_std",
        "Collision-penalized discovery",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a compact single-scenario baseline stress comparison.")
    parser.add_argument("input", help="Directory or aggregate_metrics.csv file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", default="baseline_stress")
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
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    return plt


def load_frame(path: str | Path):
    import pandas as pd

    root = Path(path)
    csv_path = root / "aggregate_metrics.csv" if root.is_dir() else root
    df = pd.read_csv(csv_path)
    for column in df.columns:
        if column in {"protocol", "mobility_model", "area_scale", "range_mode"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df, csv_path


def generate_figures(input_path: str | Path, output_dir: str | Path, tag: str) -> dict:
    df, csv_path = load_frame(input_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt = setup_matplotlib()

    manifest = []
    for metric_key, (mean_col, std_col, ylabel) in METRICS.items():
        manifest.append(save_bar(df, mean_col, std_col, ylabel, output / f"{tag}_{metric_key}.png", plt))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "inputs": [str(csv_path.resolve())],
        "style": {
            "figsize_inches": list(FIGSIZE),
            "aspect_ratio": "4:3",
            "font_family": "Times New Roman with serif fallback",
            "dpi": DPI,
            "palette": COLORS,
        },
        "counts": {
            "generated": sum(1 for item in manifest if item["status"] == "generated"),
            "skipped": sum(1 for item in manifest if item["status"] == "skipped"),
            "total": len(manifest),
        },
        "figures": manifest,
    }
    (output / f"{tag}_figure_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output, payload)
    return payload


def save_bar(df, mean_col: str, std_col: str, ylabel: str, path: Path, plt) -> dict:
    import numpy as np

    if mean_col not in df:
        return skipped(path, mean_col, f"missing {mean_col}")
    filtered = df[df["protocol"].isin(PROTOCOL_ORDER)].copy()
    filtered["protocol_rank"] = filtered["protocol"].map({name: idx for idx, name in enumerate(PROTOCOL_ORDER)})
    filtered = filtered.sort_values("protocol_rank")
    if filtered.empty:
        return skipped(path, mean_col, "no matching protocol rows")

    labels = [label_protocol(item) for item in filtered["protocol"]]
    values = [float(item) for item in filtered[mean_col]]
    errors = [float(item) for item in filtered[std_col]] if std_col in filtered else None
    x = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=2,
        color=[COLORS.get(item, "#999999") for item in filtered["protocol"]],
        edgecolor="black",
        linewidth=0.5,
        error_kw={"elinewidth": 0.6, "capthick": 0.6},
    )
    ax.set_title("N=100, 3-degree stress comparison")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Protocol")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
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
        "improved_rl_isac": "Enhanced ISAC",
    }
    return labels.get(protocol, protocol)


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
        "# Baseline Stress Figures",
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
    payload = generate_figures(args.input, args.output, args.tag)
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
