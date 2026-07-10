from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
RUNS = (
    ("clean_discovery", "Clean CTDE", "clean_ctde_planar_n10_b15_train30_seed24260720", "#0072B2"),
    ("clean_stable", "Clean CTDE + stable reward", "clean_ctde_planar_n10_b15_stable30_seed24260720", "#009E73"),
)
METHODS = (
    ("clean_discovery", "Clean CTDE", "clean_ctde_planar_n10_b15_train30_seed24260720/eval_episode_metrics.csv", "#0072B2"),
    ("clean_stable", "Clean CTDE + stable reward", "clean_ctde_planar_n10_b15_stable30_seed24260720/eval_episode_metrics.csv", "#009E73"),
    ("wang2025", "Wang2025", "clean_ctde_planar_n10_b15_train30_seed24260720/paired_baselines/wang2025_isac_tables/eval_episode_metrics.csv", "#D55E00"),
    ("uniform_random", "Uniform random", "clean_ctde_planar_n10_b15_train30_seed24260720/paired_baselines/uniform_random/eval_episode_metrics.csv", "#6C757D"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the planar clean-CTDE learnability screen.")
    parser.add_argument("--raw-root", default="05_simulation/results_raw")
    parser.add_argument("--output", default="06_analysis/paper_tables/clean_ctde_planar_n10_b15_focused_20260710")
    parser.add_argument("--figures", default="06_analysis/paper_figures/clean_ctde_planar_n10_b15_focused_20260710")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    raw_root = Path(args.raw_root)
    output = Path(args.output)
    figures = Path(args.figures)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    configure_plotting(plt)

    eval_frames = {}
    summary_rows = []
    for method, label, relative, _ in METHODS:
        frame = pd.read_csv(raw_root / relative).sort_values("scenario_seed")
        eval_frames[method] = frame
        values = frame["discovery_rate"].to_numpy(float)
        mean, std, low, high = mean_ci(values)
        summary_rows.append(
            {
                "method": method,
                "method_label": label,
                "episodes": len(values),
                "discovery_rate_mean": mean,
                "discovery_rate_std": std,
                "discovery_rate_ci95_low": low,
                "discovery_rate_ci95_high": high,
                "discovered_edges_mean": float(frame["discovered_edges"].astype(float).mean()),
                "tx_fraction_mean": float(frame["tx_actions"].astype(float).mean() / 3000.0),
                "aligned_opportunities_mean": float(frame["aligned_handshake_opportunities"].astype(float).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "method_summary.csv", index=False)

    reference = eval_frames["clean_discovery"]
    paired_rows = []
    for method, label, _, _ in METHODS[1:]:
        comparison = eval_frames[method]
        if not np.array_equal(reference["scenario_seed"].to_numpy(), comparison["scenario_seed"].to_numpy()):
            raise ValueError(f"Unpaired scenario seeds for {method}.")
        delta = reference["discovery_rate"].to_numpy(float) - comparison["discovery_rate"].to_numpy(float)
        mean, std, low, high = mean_ci(delta)
        paired_rows.append(
            {
                "reference": "clean_discovery",
                "comparison": method,
                "comparison_label": label,
                "paired_delta_mean": mean,
                "paired_delta_std": std,
                "paired_delta_ci95_low": low,
                "paired_delta_ci95_high": high,
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(delta),
                "wins": int(np.sum(delta > 0.0)),
                "ties": int(np.sum(delta == 0.0)),
                "losses": int(np.sum(delta < 0.0)),
            }
        )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "paired_deltas.csv", index=False)

    training_frames = {}
    selected_training = []
    selected_steps = []
    for method, label, run, _ in RUNS:
        episode_frame = pd.read_csv(
            raw_root / run / "episode_metrics.csv",
            usecols=["episode", "training_step", "discovery_rate", "discovered_edges", "tx_actions", "episode_return_sum"],
        )
        episode_frame["method"] = method
        episode_frame["method_label"] = label
        episode_frame["tx_fraction"] = episode_frame["tx_actions"].astype(float) / 3000.0
        training_frames[method] = episode_frame
        selected_training.append(
            episode_frame[
                ["method", "method_label", "episode", "training_step", "discovery_rate", "discovered_edges", "tx_fraction", "episode_return_sum"]
            ]
        )
        step_frame = pd.read_csv(raw_root / run / "step_rewards.csv", usecols=["training_step", "reward_mean"])
        step_frame["method"] = method
        step_frame["method_label"] = label
        step_frame["reward_200step_mean"] = step_frame["reward_mean"].astype(float).rolling(200, min_periods=1).mean()
        selected_steps.append(step_frame.iloc[::25][["method", "method_label", "training_step", "reward_200step_mean"]])
    pd.concat(selected_training, ignore_index=True).to_csv(output / "training_episode_summary.csv", index=False)
    pd.concat(selected_steps, ignore_index=True).to_csv(output / "training_step_reward_decimated.csv", index=False)

    figure_files = [
        plot_training_metric(training_frames, "discovery_rate", "Discovery rate", figures / "training_discovery_rate.png", plt),
        plot_training_metric(training_frames, "tx_fraction", "TX action fraction", figures / "training_tx_fraction.png", plt),
        plot_step_reward(raw_root, figures / "training_step_reward.png", plt),
        plot_method_summary(summary, figures / "evaluation_discovery_rate.png", plt),
        plot_paired(eval_frames, figures / "paired_episode_discovery.png", plt),
    ]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "clean_ctde_planar_n10_b15_learnability_screen",
        "raw_root": str(raw_root),
        "training_episodes": 30,
        "slots_per_episode": 300,
        "heldout_episodes": 10,
        "actor_contract": "clean_local_ctde_v1",
        "figure_size_inches": list(FIGSIZE),
        "font_family": "Times New Roman with serif fallback",
        "figures": figure_files,
        "files": ["method_summary.csv", "paired_deltas.csv", "training_episode_summary.csv", "training_step_reward_decimated.csv"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(output / "README.md", summary, paired)
    if not args.quiet:
        print(json.dumps(manifest, indent=2))


def mean_ci(values) -> tuple[float, float, float, float]:
    import numpy as np

    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    critical = 2.262 if len(values) == 10 else 1.96
    half = critical * std / math.sqrt(max(1, len(values)))
    return mean, std, mean - half, mean + half


def exact_sign_flip_p(delta) -> float:
    import numpy as np

    delta = np.asarray(delta, dtype=float)
    observed = abs(float(np.mean(delta)))
    statistics = (
        abs(float(np.mean(delta * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(delta))
    )
    exceedances = sum(value >= observed - 1e-12 for value in statistics)
    return exceedances / float(2 ** len(delta))


def configure_plotting(plt) -> None:
    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    return str(path)


def plot_training_metric(frames, metric: str, ylabel: str, path: Path, plt) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for method, label, _, color in RUNS:
        frame = frames[method]
        x = frame["episode"].astype(int) + 1
        raw = frame[metric].astype(float)
        smooth = raw.rolling(3, min_periods=1).mean()
        ax.plot(x, raw, color=color, alpha=0.22, linewidth=0.8)
        ax.plot(x, smooth, color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 30)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_step_reward(raw_root: Path, path: Path, plt) -> str:
    import pandas as pd

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for _, label, run, color in RUNS:
        frame = pd.read_csv(raw_root / run / "step_rewards.csv", usecols=["training_step", "reward_mean"])
        smooth = frame["reward_mean"].astype(float).rolling(200, min_periods=1).mean()
        ax.plot(frame["training_step"], smooth, color=color, linewidth=1.5, label=label)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean agent reward (200-step average)")
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_method_summary(frame, path: Path, plt) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    means = frame["discovery_rate_mean"].to_numpy(float)
    errors = means - frame["discovery_rate_ci95_low"].to_numpy(float)
    labels = frame["method_label"].tolist()
    colors = [color for _, _, _, color in METHODS]
    x = range(len(labels))
    ax.bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x), labels, rotation=17, ha="right")
    ax.set_ylabel("Discovery rate")
    ax.set_ylim(0.0, 0.72)
    return save_figure(fig, path)


def plot_paired(frames, path: Path, plt) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for method, label, _, color in METHODS:
        frame = frames[method]
        ax.plot(range(1, len(frame) + 1), frame["discovery_rate"], marker="o", linewidth=1.35, label=label, color=color)
    ax.set_xlabel("Held-out scenario")
    ax.set_ylabel("Discovery rate")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.0, 0.8)
    ax.legend(frameon=False, ncol=2)
    return save_figure(fig, path)


def write_readme(path: Path, summary, paired) -> None:
    best = summary.loc[summary["method"] == "clean_discovery"].iloc[0]
    random_delta = paired.loc[paired["comparison"] == "uniform_random"].iloc[0]
    wang_delta = paired.loc[paired["comparison"] == "wang2025"].iloc[0]
    text = f"""# Clean CTDE Planar Screen

- Clean CTDE discovery rate: {best['discovery_rate_mean']:.4f}.
- Paired gain over uniform random: {random_delta['paired_delta_mean']:+.4f}; exact p={random_delta['exact_sign_flip_p_two_sided']:.6f}.
- Paired gap to Wang2025: {wang_delta['paired_delta_mean']:+.4f}; exact p={wang_delta['exact_sign_flip_p_two_sided']:.6f}.
- Stable reward improves role balance but does not improve discovery; it is not selected.

The clean actor uses local candidate processing and post-handshake exchanged tables. Pair-derived phase, role hints, action targets, behavior cloning, rule residuals, and centralized execution guidance are disabled.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
