from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
METRICS = {
    "discovery_rate_mean": "Discovery rate",
    "collision_penalized_discovery_rate_mean": "Collision-penalized discovery",
    "lambda2_mean": r"Algebraic connectivity $\lambda_2$",
    "collision_count_mean": "Collision count",
    "collisions_per_discovery_censored_mean": "Collisions per discovery",
}
METHOD_ORDER = [
    "uniform_random",
    "skyorbs_like",
    "mappo_no_isac",
    "contention_no_isac",
    "contention_actor",
]
METHOD_LABELS = {
    "uniform_random": "Uniform",
    "skyorbs_like": "SkyOrbs-like",
    "mappo_no_isac": "MARL no ISAC",
    "contention_no_isac": "Contention no ISAC",
    "contention_actor": "MARL+ISAC actor",
}
COLORS = {
    "uniform_random": "#9E9E9E",
    "skyorbs_like": "#6BAED6",
    "mappo_no_isac": "#FDAE6B",
    "contention_no_isac": "#BDBDBD",
    "contention_actor": "#31A354",
    "seed31": "#756BB1",
    "seed32": "#7E57C2",
    "seed33": "#E6550D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gated MARL seed-level transfer tradeoffs.")
    parser.add_argument(
        "--baseline-summary",
        action="append",
        default=None,
        help="Phase9/five-way summary CSV. May be repeated.",
    )
    parser.add_argument(
        "--gated-summary",
        action="append",
        required=True,
        help="Seed label and gated summary path, formatted as label=path.",
    )
    parser.add_argument("--output", default="06_analysis/paper_tables/marl/p10_gate_seed_tradeoff_comparison")
    parser.add_argument("--figures", default="06_analysis/paper_figures/marl/p10_gate_seed_tradeoff_comparison")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    output_dir = Path(args.output)
    figure_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    frames = load_baselines(pd, args.baseline_summary)
    gated_specs = parse_gated_specs(args.gated_summary)
    frames.extend(load_gated(pd, gated_specs))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["beamwidth_deg"] = combined["beamwidth_deg"].astype(float)
    combined = combined.sort_values(["beamwidth_deg", "plot_order", "method_label"]).reset_index(drop=True)
    combined.to_csv(output_dir / "seed_tradeoff_method_comparison.csv", index=False)
    core = combined[["checkpoint_tag", "method_label", "beamwidth_deg", "node_count", *METRICS.keys()]]
    core.to_csv(output_dir / "seed_tradeoff_core_metrics.csv", index=False)

    figures = write_figures(combined, figure_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_summary": [str(path) for path in baseline_summary_paths(args)],
        "gated_summary": [{"label": label, "path": str(path)} for label, path in gated_specs],
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "rows": int(len(combined)),
        "figures": figures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_baselines(pd, paths: list[str]) -> list:
    frames = []
    for path in paths or baseline_summary_paths(None):
        frame = pd.read_csv(path)
        frame = normalize_beamwidth(frame)
        frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
        frame["checkpoint_tag"] = "phase9_baseline"
        frame["method_label"] = frame["method"].map(METHOD_LABELS).fillna(frame["method_label"])
        frame["plot_order"] = frame["method"].map({method: index for index, method in enumerate(METHOD_ORDER)})
        frames.append(frame)
    return frames


def baseline_summary_paths(args: argparse.Namespace | None) -> list[str]:
    if args is not None and args.baseline_summary:
        return list(args.baseline_summary)
    return [
        "06_analysis/paper_tables/marl/phase9_fiveway_n100_b10_3000slot_10ep_stoch_all_methods/marl_transfer_summary.csv",
        "06_analysis/paper_tables/marl/phase9_fiveway_n100_b15_3000slot_10ep_stoch_all_methods/marl_transfer_summary.csv",
    ]


def parse_gated_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError("--gated-summary must be formatted as label=path.")
        label, path = value.split("=", 1)
        specs.append((label.strip(), Path(path.strip())))
    return specs


def load_gated(pd, specs: list[tuple[str, Path]]) -> list:
    frames = []
    for offset, (label, path) in enumerate(specs):
        frame = normalize_beamwidth(pd.read_csv(path))
        frame["checkpoint_tag"] = label
        frame["method"] = f"gated_contention_actor_{label}"
        frame["method_label"] = label
        frame["plot_order"] = len(METHOD_ORDER) + offset
        frames.append(frame)
    return frames


def normalize_beamwidth(frame):
    if "beamwidth_deg" not in frame.columns and "beam_width_deg" in frame.columns:
        frame = frame.rename(columns={"beam_width_deg": "beamwidth_deg"})
    return frame


def write_figures(frame, figure_dir: Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    figures = []
    labels = frame[["checkpoint_tag", "method_label", "plot_order"]].drop_duplicates().sort_values("plot_order")
    beamwidths = sorted(frame["beamwidth_deg"].dropna().astype(float).unique())
    x = np.arange(len(beamwidths))
    width = min(0.11, 0.78 / max(1, len(labels)))
    for metric, ylabel in METRICS.items():
        if metric not in frame.columns:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        for idx, row in enumerate(labels.itertuples(index=False)):
            tag = str(row.checkpoint_tag)
            label = str(row.method_label)
            values = []
            for beamwidth in beamwidths:
                subset = frame[
                    (frame["checkpoint_tag"].astype(str) == tag)
                    & (frame["method_label"].astype(str) == label)
                    & (frame["beamwidth_deg"].astype(float) == beamwidth)
                ][metric]
                values.append(float(subset.iloc[0]) if len(subset) else np.nan)
            offset = (idx - (len(labels) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=label,
                color=COLORS.get(tag, "#7E57C2"),
                edgecolor="black",
                linewidth=0.25,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(value)} deg" for value in beamwidths])
        ax.set_xlabel("Beamwidth")
        ax.set_ylabel(ylabel if "collision_count" not in metric else f"{ylabel} (log scale)")
        ax.set_title(f"{ylabel} under N=100 transfer")
        if metric in {"collision_count_mean", "collisions_per_discovery_censored_mean"}:
            ax.set_yscale("log")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, borderaxespad=0.0)
        fig.subplots_adjust(left=0.12, right=0.72, bottom=0.14, top=0.90)
        path = figure_dir / f"seed_tradeoff_{metric}.png"
        fig.savefig(path)
        plt.close(fig)
        figures.append({"metric": metric, "path": str(path)})
    return figures


if __name__ == "__main__":
    main()
