from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
METHODS = [
    ("legacy_shared", "Legacy reward", "#0072B2", "-"),
    ("collision_reward", "Collision reward", "#009E73", "--"),
    ("contention_actor", "Contention actor", "#D55E00", "-."),
]
METRICS = {
    "discovery_rate_mean": "Discovery rate",
    "collision_penalized_discovery_rate_mean": "Collision-penalized discovery rate",
    "collision_count_mean": "Collisions per episode",
    "collisions_per_discovery_censored_mean": "Collisions per discovery",
    "lambda2_mean": r"$\lambda_2$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot N=100 MARL method comparison curves.")
    parser.add_argument("--legacy", default="06_analysis/paper_tables/marl/phase3_n100_stress/marl_transfer_summary.csv")
    parser.add_argument("--collision", default="06_analysis/paper_tables/marl/phase4_shared_collision_transfer_probe/marl_transfer_summary.csv")
    parser.add_argument("--contention", default="06_analysis/paper_tables/marl/phase5_contention_shared_v2_transfer_probe/marl_transfer_summary.csv")
    parser.add_argument("--output", default="06_analysis/paper_tables/marl/phase5_method_comparison")
    parser.add_argument("--figures", default="06_analysis/paper_figures/marl/phase5_method_comparison")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    output_dir = Path(args.output)
    figure_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.append(load_method(pd, args.legacy, "legacy_shared"))
    rows.append(load_method(pd, args.collision, "collision_reward"))
    rows.append(load_method(pd, args.contention, "contention_actor"))
    combined = pd.concat(rows, ignore_index=True)
    combined = combined[combined["beamwidth_deg"].isin([10.0, 15.0, 30.0])].copy()
    combined.sort_values(["beamwidth_deg", "method"], inplace=True)
    combined.to_csv(output_dir / "marl_method_comparison.csv", index=False)

    figures = write_figures(combined, figure_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "legacy": str(args.legacy),
        "collision": str(args.collision),
        "contention": str(args.contention),
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "rows": int(len(combined)),
        "figures": figures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, manifest, combined)
    if not args.quiet:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_method(pd, path: str, method: str):
    frame = pd.read_csv(path)
    frame = frame[frame["slots_per_episode"].astype(float) == 3000.0].copy()
    frame["method"] = method
    return frame


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


def write_figures(frame, figure_dir: Path) -> list[dict]:
    plt = setup_matplotlib()
    figures = []
    for metric, ylabel in METRICS.items():
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for method, label, color, linestyle in METHODS:
            group = frame[frame["method"] == method].sort_values("beamwidth_deg")
            if group.empty:
                continue
            ci_col = metric.replace("_mean", "_ci95")
            yerr = group[ci_col] if ci_col in group.columns else None
            ax.errorbar(
                group["beamwidth_deg"],
                group[metric],
                yerr=yerr,
                marker="o",
                capsize=3,
                label=label,
                color=color,
                linestyle=linestyle,
            )
        ax.set_xlabel("Beamwidth (deg)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = figure_dir / f"marl_method_comparison_{metric.replace('_mean', '')}.png"
        fig.savefig(path)
        plt.close(fig)
        figures.append({"path": str(path), "metric": metric})
    return figures


def write_readme(output_dir: Path, manifest: dict, frame) -> None:
    lines = [
        "# MARL Method Comparison",
        "",
        f"- Created: {manifest['created_at']}",
        "- Scenario: zero-shot N=100 transfer, 3000-slot evaluation.",
        "",
        "```csv",
        frame.to_csv(index=False).strip(),
        "```",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
