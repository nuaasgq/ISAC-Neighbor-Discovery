from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


FIGSIZE = (6.4, 4.8)
DPI = 300
METHOD_STYLES = {
    "uniform_random": ("Uniform random", "#6C757D", ":"),
    "skyorbs_like": ("SkyOrbs-like", "#CC79A7", "--"),
    "mappo_no_isac": ("MAPPO no-ISAC", "#56B4E9", "-."),
    "contention_no_isac": ("Contention no-ISAC", "#009E73", (0, (3, 1, 1, 1))),
    "legacy_shared": ("Legacy reward", "#0072B2", "-"),
    "collision_reward": ("Collision reward", "#009E73", "--"),
    "contention_actor": ("Contention actor", "#D55E00", "-."),
}
FALLBACK_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#6C757D", "#E69F00"]
FALLBACK_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
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
    parser.add_argument("--combined-summary", action="append", default=None)
    parser.add_argument("--output", default="06_analysis/paper_tables/marl/phase5_method_comparison")
    parser.add_argument("--figures", default="06_analysis/paper_figures/marl/phase5_method_comparison")
    parser.add_argument("--slots", type=int, default=3000)
    parser.add_argument("--node-count", type=int, default=100)
    parser.add_argument("--phase", default="eval_stochastic", help="Use 'all' to keep all phases.")
    parser.add_argument("--beamwidths", type=float, nargs="+", default=[10.0, 15.0, 30.0])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import pandas as pd

    output_dir = Path(args.output)
    figure_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.combined_summary:
        combined = load_combined_summaries(pd, args.combined_summary)
    else:
        rows = []
        rows.append(load_method(pd, args.legacy, "legacy_shared"))
        rows.append(load_method(pd, args.collision, "collision_reward"))
        rows.append(load_method(pd, args.contention, "contention_actor"))
        combined = pd.concat(rows, ignore_index=True)
    combined = filter_comparison_frame(combined, args)
    if combined.empty:
        raise ValueError("No rows remain after applying method-comparison filters.")
    combined.sort_values(["beamwidth_deg", "method"], inplace=True)
    combined.to_csv(output_dir / "marl_method_comparison.csv", index=False)

    figures = write_figures(combined, figure_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "legacy": str(args.legacy),
        "collision": str(args.collision),
        "contention": str(args.contention),
        "combined_summary": [str(path) for path in args.combined_summary] if args.combined_summary else [],
        "filters": {
            "slots_per_episode": int(args.slots),
            "node_count": int(args.node_count),
            "phase": str(args.phase),
            "beamwidths": [float(value) for value in args.beamwidths],
        },
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
    frame["method"] = method
    return frame


def load_combined_summaries(pd, paths: list[str]):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "method" not in frame.columns:
            frame["method"] = frame.apply(infer_method, axis=1)
        else:
            frame["method"] = frame.apply(lambda row: str(row.get("method") or infer_method(row)), axis=1)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def infer_method(row: Any) -> str:
    network = str(row.get("train_network", "shared"))
    reward = str(row.get("train_reward_version", "legacy"))
    algorithm = str(row.get("train_algorithm", "unknown"))
    if network == "contention_shared" and reward == "collision_topology":
        return "contention_actor"
    if network == "shared" and reward == "collision_topology":
        return "collision_reward"
    if network == "shared" and reward == "legacy":
        return "legacy_shared"
    return f"{algorithm}_{network}_{reward}".replace("/", "_")


def filter_comparison_frame(frame, args: argparse.Namespace):
    filtered = frame.copy()
    if "slots_per_episode" in filtered.columns:
        filtered = filtered[filtered["slots_per_episode"].astype(float) == float(args.slots)]
    if "node_count" in filtered.columns:
        filtered = filtered[filtered["node_count"].astype(float) == float(args.node_count)]
    if str(args.phase).lower() != "all" and "phase" in filtered.columns:
        filtered = filtered[filtered["phase"].astype(str) == str(args.phase)]
    if "beamwidth_deg" in filtered.columns:
        beamwidths = {float(value) for value in args.beamwidths}
        filtered = filtered[filtered["beamwidth_deg"].astype(float).isin(beamwidths)]
    return filtered.copy()


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
        if metric not in frame.columns:
            continue
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for index, method in enumerate(method_order(frame)):
            label, color, linestyle = style_for_method(method, index)
            group = frame[frame["method"] == method].sort_values("beamwidth_deg")
            if group.empty:
                continue
            if "method_label" in group.columns and str(group["method_label"].iloc[0]) not in {"", "nan"}:
                label = str(group["method_label"].iloc[0])
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


def method_order(frame) -> list[str]:
    preferred = [
        "uniform_random",
        "skyorbs_like",
        "mappo_no_isac",
        "contention_no_isac",
        "legacy_shared",
        "collision_reward",
        "contention_actor",
    ]
    available = [str(value) for value in frame["method"].dropna().unique()]
    ordered = [method for method in preferred if method in available]
    ordered.extend(sorted(method for method in available if method not in set(ordered)))
    return ordered


def style_for_method(method: str, index: int) -> tuple[str, str, object]:
    if method in METHOD_STYLES:
        return METHOD_STYLES[method]
    label = method.replace("_", " ")
    color = FALLBACK_COLORS[index % len(FALLBACK_COLORS)]
    linestyle = FALLBACK_LINESTYLES[index % len(FALLBACK_LINESTYLES)]
    return label, color, linestyle


def write_readme(output_dir: Path, manifest: dict, frame) -> None:
    lines = [
        "# MARL Method Comparison",
        "",
        f"- Created: {manifest['created_at']}",
        (
            "- Scenario: zero-shot transfer, "
            f"N={manifest['filters']['node_count']}, "
            f"{manifest['filters']['slots_per_episode']}-slot evaluation, "
            f"phase={manifest['filters']['phase']}."
        ),
        "",
        "```csv",
        frame.to_csv(index=False).strip(),
        "```",
    ]
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
