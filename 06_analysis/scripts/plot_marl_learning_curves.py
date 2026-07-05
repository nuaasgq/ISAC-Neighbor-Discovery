from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
COLORS = {
    "isac_mappo": "#0072B2",
    "mappo": "#009E73",
    "ippo": "#D55E00",
    "no_isac": "#6C757D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and plot real MARL learning curves.")
    parser.add_argument("--input-root", default="05_simulation/results_raw/marl")
    parser.add_argument("--run-dir", action="append", default=None, help="Specific run directory; may be repeated.")
    parser.add_argument("--output", default="06_analysis/paper_tables/marl/current")
    parser.add_argument("--figures", default="06_analysis/paper_figures/marl/current")
    parser.add_argument("--rolling-window", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    figure_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    run_dirs = resolve_run_dirs(args)
    step_rows, episode_rows, eval_rows, resources, manifests = load_runs(run_dirs, pd)
    write_frame(step_rows, output_dir / "marl_step_rewards.csv")
    write_frame(episode_rows, output_dir / "marl_episode_metrics.csv")
    write_frame(eval_rows, output_dir / "marl_eval_episode_metrics.csv")
    write_frame(resources, output_dir / "marl_resource_log.csv")

    figures = write_figures(step_rows, episode_rows, eval_rows, resources, figure_dir, int(args.rolling_window))
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(args.input_root),
        "run_dirs": [str(path) for path in run_dirs],
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "runs": len(run_dirs),
        "step_rows": int(len(step_rows)),
        "episode_rows": int(len(episode_rows)),
        "eval_rows": int(len(eval_rows)),
        "resource_rows": int(len(resources)),
        "source_manifests": manifests,
        "figures": figures,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, manifest, episode_rows, eval_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [Path(value) for value in args.run_dir]
    input_root = Path(args.input_root)
    candidates = []
    for manifest_path in sorted(input_root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("scope") == "real_marl_training":
            candidates.append(manifest_path.parent)
    if not candidates:
        raise FileNotFoundError(f"No real MARL training runs found under {input_root}.")
    return candidates


def load_runs(run_dirs: list[Path], pd):
    step_frames = []
    episode_frames = []
    eval_frames = []
    resource_frames = []
    manifests = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_name = run_dir.name
        manifests.append({"run": run_name, "path": str(manifest_path), "algorithm": manifest.get("algorithm")})
        step_frames.append(load_csv(run_dir / "step_rewards.csv", pd, run_name, manifest))
        episode_frames.append(load_csv(run_dir / "episode_metrics.csv", pd, run_name, manifest))
        eval_frames.append(load_csv(run_dir / "eval_episode_metrics.csv", pd, run_name, manifest))
        resource_frames.append(load_csv(run_dir / "resource_log.csv", pd, run_name, manifest))
    return (
        concat_frames(step_frames, pd),
        concat_frames(episode_frames, pd),
        concat_frames(eval_frames, pd),
        concat_frames(resource_frames, pd),
        manifests,
    )


def load_csv(path: Path, pd, run_name: str, manifest: dict):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["run"] = run_name
    frame["algorithm"] = frame.get("algorithm", manifest.get("algorithm", "unknown"))
    frame["env_protocol"] = frame.get("env_protocol", manifest.get("env_protocol", "unknown"))
    frame["node_count"] = manifest.get("node_count", "")
    frame["beam_count"] = manifest.get("beam_count", "")
    frame["azimuth_cells"] = manifest.get("azimuth_cells", "")
    frame["elevation_cells"] = manifest.get("elevation_cells", "")
    frame["slot_duration_ms"] = manifest.get("slot_duration_ms", "")
    return frame


def concat_frames(frames: list, pd):
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_frame(frame, path: Path) -> None:
    if frame.empty:
        path.write_text("", encoding="utf-8")
        return
    frame.to_csv(path, index=False)


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


def write_figures(step_rows, episode_rows, eval_rows, resources, figure_dir: Path, rolling_window: int) -> list[dict]:
    plt = setup_matplotlib()
    figures = []
    if not step_rows.empty:
        figures.append(
            plot_step_curve(
                step_rows,
                "reward_mean",
                "Mean step reward",
                figure_dir / "marl_step_reward_curve.png",
                plt,
                rolling_window,
            )
        )
        figures.append(
            plot_step_curve(
                step_rows,
                "discovery_rate",
                "Discovery rate",
                figure_dir / "marl_step_discovery_curve.png",
                plt,
                rolling_window,
            )
        )
        figures.append(
            plot_step_curve(
                step_rows,
                "empty_scan_ratio",
                "Empty-scan ratio",
                figure_dir / "marl_step_empty_scan_curve.png",
                plt,
                rolling_window,
            )
        )
    if not episode_rows.empty:
        for metric, ylabel, filename in [
            ("episode_return_mean_per_agent", "Episode return per agent", "marl_episode_return_curve.png"),
            ("policy_loss", "Policy loss", "marl_policy_loss_curve.png"),
            ("value_loss", "Value loss", "marl_value_loss_curve.png"),
            ("entropy", "Policy entropy", "marl_entropy_curve.png"),
            ("discovery_rate", "Discovery rate", "marl_episode_discovery_curve.png"),
            ("lambda2", r"$\lambda_2$", "marl_episode_lambda2_curve.png"),
            ("collision_count", "Collisions per episode", "marl_episode_collision_curve.png"),
            ("empty_scan_ratio", "Empty-scan ratio", "marl_episode_empty_scan_curve.png"),
        ]:
            if metric in episode_rows.columns:
                figures.append(plot_episode_curve(episode_rows, metric, ylabel, figure_dir / filename, plt))
    if not eval_rows.empty:
        figures.append(plot_eval_curve(eval_rows, "discovery_rate", "Eval discovery rate", figure_dir / "marl_eval_discovery_curve.png", plt))
        if "lambda2" in eval_rows.columns:
            figures.append(plot_eval_curve(eval_rows, "lambda2", r"Eval $\lambda_2$", figure_dir / "marl_eval_lambda2_curve.png", plt))
    if not resources.empty:
        if "rss_mb" in resources.columns:
            figures.append(plot_resource_curve(resources, "rss_mb", "RSS memory (MB)", figure_dir / "marl_resource_rss_curve.png", plt))
        if "system_memory_percent" in resources.columns:
            figures.append(
                plot_resource_curve(
                    resources,
                    "system_memory_percent",
                    "System memory (%)",
                    figure_dir / "marl_resource_memory_curve.png",
                    plt,
                )
            )
    return figures


def plot_step_curve(rows, metric: str, ylabel: str, path: Path, plt, rolling_window: int) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for run, group in rows.groupby("run"):
        group = group.sort_values("training_step").copy()
        color = color_for(group)
        y = group[metric].rolling(max(1, rolling_window), min_periods=1).mean()
        ax.plot(group["training_step"], y, label=str(run), color=color, alpha=0.9)
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "step_curve", "metric": metric}


def plot_episode_curve(rows, metric: str, ylabel: str, path: Path, plt) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for run, group in rows.groupby("run"):
        group = group.sort_values("training_step")
        ax.plot(group["training_step"], group[metric], marker="o", markersize=3, label=str(run), color=color_for(group))
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "episode_curve", "metric": metric}


def plot_eval_curve(rows, metric: str, ylabel: str, path: Path, plt) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for run, group in rows.groupby("run"):
        grouped = group.groupby("eval_after_episode", as_index=False)[metric].mean()
        ax.plot(grouped["eval_after_episode"], grouped[metric], marker="s", markersize=4, label=str(run), color=color_for(group))
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "eval_curve", "metric": metric}


def plot_resource_curve(rows, metric: str, ylabel: str, path: Path, plt) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for run, group in rows.groupby("run"):
        group = group.sort_values("training_step")
        ax.plot(group["training_step"], group[metric], label=str(run), color=color_for(group))
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "resource_curve", "metric": metric}


def color_for(group) -> str:
    algorithm = str(group["algorithm"].iloc[0]) if "algorithm" in group and len(group) else "unknown"
    if "no_isac" in str(group.get("env_protocol", "")).lower():
        return COLORS["no_isac"]
    return COLORS.get(algorithm, "#6C757D")


def write_readme(output_dir: Path, manifest: dict, episode_rows, eval_rows) -> None:
    lines = [
        "# MARL Learning Curves",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Runs loaded: {manifest['runs']}",
        f"- Step rows: {manifest['step_rows']}",
        f"- Episode rows: {manifest['episode_rows']}",
        f"- Eval rows: {manifest['eval_rows']}",
        "",
        "These plots use true environment steps or training episodes as the x-axis.",
        "They are intended for real MARL runs, not CEM candidate-search traces.",
    ]
    if not episode_rows.empty:
        last = episode_rows.sort_values("training_step").groupby("run").tail(1)
        lines.extend(["", "## Final Train Rows", "", "```csv", last.to_csv(index=False).strip(), "```"])
    if not eval_rows.empty:
        last_eval = eval_rows.sort_values("eval_after_episode").groupby("run").tail(1)
        lines.extend(["", "## Final Eval Rows", "", "```csv", last_eval.to_csv(index=False).strip(), "```"])
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
