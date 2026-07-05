from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
METRICS = {
    "discovery_rate": "Discovery rate",
    "mean_delay_censored": "Mean discovery delay (slots)",
    "p95_delay_censored": "P95 discovery delay (slots)",
    "empty_scan_ratio": "Empty-scan ratio",
    "lambda2": r"$\lambda_2$",
    "collision_count": "Collisions per episode",
}
COLORS = {
    "isac_mappo": "#0072B2",
    "mappo": "#D55E00",
    "ippo": "#009E73",
    "unknown": "#6C757D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and plot MARL transfer evaluation results.")
    parser.add_argument("--input-root", default="05_simulation/results_raw/marl_eval")
    parser.add_argument("--run-dir", action="append", default=None)
    parser.add_argument("--output", default="06_analysis/paper_tables/marl/transfer_current")
    parser.add_argument("--figures", default="06_analysis/paper_figures/marl/transfer_current")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    figure_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    run_dirs = resolve_run_dirs(args)
    rows = load_eval_rows(run_dirs, pd)
    if rows.empty:
        raise FileNotFoundError("No transfer evaluation rows found.")
    rows.to_csv(output_dir / "marl_transfer_eval_rows.csv", index=False)
    summary = summarize(rows, pd)
    summary.to_csv(output_dir / "marl_transfer_summary.csv", index=False)
    figures = write_figures(summary, figure_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(args.input_root),
        "run_dirs": [str(path) for path in run_dirs],
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "rows": int(len(rows)),
        "runs": int(rows["run"].nunique()),
        "figures": figures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, manifest, summary)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [Path(value) for value in args.run_dir]
    root = Path(args.input_root)
    runs = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("scope") == "marl_transfer_evaluation":
            runs.append(manifest_path.parent)
    return runs


def load_eval_rows(run_dirs: list[Path], pd):
    frames = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        data_path = run_dir / "eval_episode_metrics.csv"
        if not manifest_path.exists() or not data_path.exists() or data_path.stat().st_size == 0:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(data_path)
        frame["run"] = run_dir.name
        frame["train_algorithm"] = str(manifest.get("train_algorithm", "unknown"))
        frame["env_protocol"] = str(manifest.get("env_protocol", "unknown"))
        frame["node_count"] = int(manifest.get("node_count", 0))
        frame["beam_count"] = int(manifest.get("beam_count", 0))
        frame["slots_per_episode"] = int(manifest.get("slots_per_episode", 0))
        frame["azimuth_cells"] = int(manifest.get("azimuth_cells", 0))
        frame["elevation_cells"] = int(manifest.get("elevation_cells", 0))
        frame["beamwidth_deg"] = 360.0 / max(1, int(manifest.get("azimuth_cells", 1)))
        frame["communication_range_m"] = float(manifest.get("communication_range_m", 0.0))
        frame["sensing_range_m"] = float(manifest.get("sensing_range_m", 0.0))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize(rows, pd):
    group_keys = [
        "train_algorithm",
        "env_protocol",
        "phase",
        "node_count",
        "beamwidth_deg",
        "beam_count",
        "slots_per_episode",
        "communication_range_m",
        "sensing_range_m",
    ]
    records = []
    for key, group in rows.groupby(group_keys, dropna=False):
        record = dict(zip(group_keys, key, strict=True))
        record["eval_n"] = int(len(group))
        record["run_n"] = int(group["run"].nunique())
        for metric in METRICS:
            values = pd.to_numeric(group.get(metric), errors="coerce").dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            record[f"{metric}_ci95"] = 1.96 * record[f"{metric}_std"] / (len(values) ** 0.5) if len(values) > 1 else 0.0
        records.append(record)
    return pd.DataFrame(records).sort_values(["phase", "node_count", "beamwidth_deg", "train_algorithm"])


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


def write_figures(summary, figure_dir: Path) -> list[dict]:
    plt = setup_matplotlib()
    figures = []
    for metric, ylabel in METRICS.items():
        metric_col = f"{metric}_mean"
        if metric_col not in summary.columns:
            continue
        figures.append(plot_node_curve(summary, metric_col, f"{metric}_ci95", ylabel, figure_dir / f"marl_transfer_node_{metric}.png", plt))
        figures.append(plot_beam_curve(summary, metric_col, f"{metric}_ci95", ylabel, figure_dir / f"marl_transfer_beam_{metric}.png", plt))
    return figures


def plot_node_curve(summary, metric: str, ci_metric: str, ylabel: str, path: Path, plt) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    subset = summary[summary["phase"].astype(str).str.endswith("stochastic")].copy()
    if subset.empty:
        subset = summary
    for (algorithm, beamwidth, slots), group in subset.groupby(["train_algorithm", "beamwidth_deg", "slots_per_episode"]):
        group = group.sort_values("node_count")
        label = f"{algorithm}, {beamwidth:g} deg, {slots:g} slots"
        ax.errorbar(
            group["node_count"],
            group[metric],
            yerr=group[ci_metric],
            marker="o",
            capsize=3,
            label=label,
            color=COLORS.get(str(algorithm), COLORS["unknown"]),
        )
    ax.set_xlabel("Number of UAVs")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "node_transfer_curve", "metric": metric}


def plot_beam_curve(summary, metric: str, ci_metric: str, ylabel: str, path: Path, plt) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    subset = summary[summary["phase"].astype(str).str.endswith("stochastic")].copy()
    if subset.empty:
        subset = summary
    for (algorithm, node_count, slots), group in subset.groupby(["train_algorithm", "node_count", "slots_per_episode"]):
        group = group.sort_values("beamwidth_deg")
        label = f"{algorithm}, N={node_count:g}, {slots:g} slots"
        ax.errorbar(
            group["beamwidth_deg"],
            group[metric],
            yerr=group[ci_metric],
            marker="s",
            capsize=3,
            label=label,
            color=COLORS.get(str(algorithm), COLORS["unknown"]),
        )
    ax.set_xlabel("Beamwidth (deg)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "beam_transfer_curve", "metric": metric}


def write_readme(output_dir: Path, manifest: dict, summary) -> None:
    lines = [
        "# MARL Transfer Evaluation",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Runs loaded: {manifest['runs']}",
        f"- Rows loaded: {manifest['rows']}",
        "",
        "This table aggregates zero-shot evaluations of trained shared MARL policies under changed node counts and beam codebooks.",
    ]
    if not summary.empty:
        lines.extend(["", "## Summary", "", "```csv", summary.to_csv(index=False).strip(), "```"])
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
