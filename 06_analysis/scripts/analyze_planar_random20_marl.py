from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
METHODS = (
    ("rule_guided_marl", "Rule-guided MARL", "seed23260800/eval_episode_metrics.csv", "#0072B2"),
    ("wang2025", "Wang2025", "paired_baselines_seed23260800/wang2025_isac_tables/eval_episode_metrics.csv", "#D55E00"),
    ("adapter_zero", "Adapter zero", "adapter_zero_seed23260800/eval_episode_metrics.csv", "#009E73"),
    ("no_isac_observation", "No ISAC observation", "no_isac_observation_seed23260800/eval_episode_metrics.csv", "#CC79A7"),
    ("uniform_random", "Uniform random", "paired_baselines_seed23260800/uniform_random/eval_episode_metrics.csv", "#6C757D"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the N=10 planar random-20% MARL validation run.")
    parser.add_argument(
        "--raw-root",
        default="05_simulation/results_raw/planar_n10_b15_marl_screen_20260710",
    )
    parser.add_argument(
        "--output",
        default="06_analysis/paper_tables/planar_n10_b15_random20_20260710",
    )
    parser.add_argument(
        "--figures",
        default="06_analysis/paper_figures/planar_n10_b15_random20_20260710",
    )
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

    frames: dict[str, object] = {}
    summary_rows = []
    for method, label, relative_path, _ in METHODS:
        frame = pd.read_csv(raw_root / relative_path)
        frame["method"] = method
        frame["method_label"] = label
        frames[method] = frame
        values = frame["discovery_rate"].astype(float).to_numpy()
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
                "aligned_opportunities_mean": float(
                    frame["aligned_handshake_opportunities"].astype(float).mean()
                ),
                "forward_decodes_mean": float(frame["forward_decodes"].astype(float).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "method_summary.csv", index=False)

    reference = frames["rule_guided_marl"].sort_values("scenario_seed")
    paired_rows = []
    for method, label, _, _ in METHODS[1:]:
        comparison = frames[method].sort_values("scenario_seed")
        if not np.array_equal(reference["scenario_seed"].to_numpy(), comparison["scenario_seed"].to_numpy()):
            raise ValueError(f"Scenario seeds are not paired for {method}.")
        delta = reference["discovery_rate"].to_numpy(float) - comparison["discovery_rate"].to_numpy(float)
        mean, std, low, high = mean_ci(delta)
        paired_rows.append(
            {
                "reference": "rule_guided_marl",
                "comparison": method,
                "comparison_label": label,
                "episodes": len(delta),
                "paired_delta_mean": mean,
                "paired_delta_std": std,
                "paired_delta_ci95_low": low,
                "paired_delta_ci95_high": high,
                "exact_sign_flip_p_two_sided": exact_sign_flip_p(delta),
                "rule_guided_marl_wins": int(np.sum(delta > 0.0)),
                "ties": int(np.sum(delta == 0.0)),
            }
        )
    pd.DataFrame(paired_rows).to_csv(output / "paired_deltas.csv", index=False)

    training = pd.read_csv(raw_root / "seed23260800/episode_metrics.csv")
    steps = pd.read_csv(raw_root / "seed23260800/step_rewards.csv")
    training.to_csv(output / "training_episode_metrics.csv", index=False)
    steps.to_csv(output / "training_step_rewards.csv", index=False)

    written = [
        plot_step_rewards(steps, figures / "training_step_reward.png", plt),
        plot_training_discovery(training, figures / "training_discovery_rate.png", plt),
        plot_method_summary(summary, figures / "evaluation_discovery_rate.png", plt),
        plot_paired_episodes(frames, figures / "paired_episode_discovery.png", plt),
    ]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": "planar_n10_b15_random20_rule_guided_upper_bound",
        "raw_root": str(raw_root),
        "output": str(output),
        "figures": str(figures),
        "figure_size_inches": list(FIGSIZE),
        "font_family": "Times New Roman with serif fallback",
        "methods": [method for method, _, _, _ in METHODS],
        "files": ["method_summary.csv", "paired_deltas.csv", "training_episode_metrics.csv", "training_step_rewards.csv"],
        "figure_files": written,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(manifest, indent=2))


def mean_ci(values) -> tuple[float, float, float, float]:
    import numpy as np

    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, 0.0, mean, mean
    std = float(np.std(values, ddof=1))
    critical = 2.262 if len(values) == 10 else 1.96
    half_width = critical * std / math.sqrt(len(values))
    return mean, std, mean - half_width, mean + half_width


def exact_sign_flip_p(delta) -> float:
    import numpy as np

    delta = np.asarray(delta, dtype=float)
    observed = abs(float(np.mean(delta)))
    exceedances = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(delta)):
        statistic = abs(float(np.mean(delta * np.asarray(signs))))
        exceedances += statistic >= observed - 1e-12
        total += 1
    return exceedances / total


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


def plot_step_rewards(frame, path: Path, plt) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = frame["training_step"].astype(int)
    reward = frame["reward_mean"].astype(float)
    smooth = reward.rolling(window=100, min_periods=1).mean()
    ax.plot(x, reward, color="#9E9E9E", linewidth=0.5, alpha=0.35, label="Per-step reward")
    ax.plot(x, smooth, color="#0072B2", linewidth=1.8, label="100-step moving average")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean agent reward")
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_training_discovery(frame, path: Path, plt) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = frame["episode"].astype(int) + 1
    y = frame["discovery_rate"].astype(float)
    ax.plot(x, y, color="#0072B2", marker="o", linewidth=1.8)
    ax.axhline(0.20, color="#6C757D", linestyle="--", linewidth=1.2, label="Random target (20%)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Discovery rate")
    ax.set_ylim(0.0, 0.8)
    ax.set_xticks(x)
    ax.legend(frameon=False)
    return save_figure(fig, path)


def plot_method_summary(frame, path: Path, plt) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = frame["method_label"].tolist()
    means = frame["discovery_rate_mean"].to_numpy(float)
    errors = means - frame["discovery_rate_ci95_low"].to_numpy(float)
    colors = [color for _, _, _, color in METHODS]
    x = range(len(labels))
    ax.bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x), labels, rotation=18, ha="right")
    ax.set_ylabel("Discovery rate")
    ax.set_ylim(0.0, 0.9)
    return save_figure(fig, path)


def plot_paired_episodes(frames, path: Path, plt) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    selected = {"rule_guided_marl", "wang2025", "uniform_random"}
    for method, label, _, color in METHODS:
        if method not in selected:
            continue
        frame = frames[method].sort_values("scenario_seed")
        ax.plot(
            range(1, len(frame) + 1),
            frame["discovery_rate"].astype(float),
            marker="o",
            linewidth=1.5,
            label=label,
            color=color,
        )
    ax.set_xlabel("Held-out scenario")
    ax.set_ylabel("Discovery rate")
    ax.set_xticks(range(1, len(frames["rule_guided_marl"]) + 1))
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, ncol=3)
    return save_figure(fig, path)


if __name__ == "__main__":
    main()
