from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


FIGSIZE = (6.4, 4.8)
DPI = 300
COLOR_CANDIDATE = "#6C757D"
COLOR_ELITE = "#0072B2"
COLOR_BEST = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render CEM candidate-evaluation trajectories for the selected shared-policy search run."
    )
    parser.add_argument("--training-dir", default="06_analysis/paper_tables/round2_transfer/training")
    parser.add_argument("--output", default="06_analysis/paper_tables/training_candidate_trajectory_20260705")
    parser.add_argument("--figures", default="06_analysis/paper_figures/training_candidate_trajectory_20260705")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_dir = Path(args.training_dir)
    output_dir = Path(args.output)
    figure_dir = Path(args.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    history = pd.read_csv(training_dir / "training_history.csv")
    elite = pd.read_csv(training_dir / "elite_history.csv")
    history = add_evaluation_index(history)
    elite = add_evaluation_index(elite)
    best_by_generation = build_best_by_generation(history, pd)

    history.to_csv(output_dir / "candidate_evaluation_history.csv", index=False)
    elite.to_csv(output_dir / "elite_evaluation_history.csv", index=False)
    best_by_generation.to_csv(output_dir / "generation_best_history.csv", index=False)
    for name in ("manifest.json", "best_config.yaml", "test_summary.csv"):
        src = training_dir / name
        if src.exists():
            shutil.copyfile(src, output_dir / name)

    figures = write_figures(history, elite, best_by_generation, figure_dir)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_dir": str(training_dir),
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "figures": figures,
        "note": "The x-axis is cumulative training environment steps; each point is one full candidate-policy episode evaluation.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output_dir, manifest, history, best_by_generation)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def add_evaluation_index(df):
    df = df.sort_values(["generation", "candidate"]).reset_index(drop=True).copy()
    if "evaluation_index" not in df.columns:
        df["evaluation_index"] = range(1, len(df) + 1)
    step_scale = (
        df.get("slots_per_episode", 1).astype(float)
        * df.get("episodes_per_seed", 1).astype(float)
        * df.get("seed_count", 1).astype(float)
    )
    df["training_step"] = (df["evaluation_index"].astype(float) * step_scale).astype(int)
    return df


def build_best_by_generation(history, pd):
    rows = []
    best_so_far = None
    for generation, group in history.groupby("generation", sort=True):
        best = group.sort_values("score", ascending=False).iloc[0].copy()
        if best_so_far is None or float(best["score"]) > float(best_so_far["score"]):
            best_so_far = best
        rows.append(
            {
                "generation": int(generation),
                "candidate": int(best["candidate"]),
                "evaluation_index": int(best["evaluation_index"]),
                "training_step": int(best["training_step"]),
                "generation_best_score": float(best["score"]),
                "generation_best_reward": float(best.get("reward_mean", best["score"])),
                "generation_best_discovery_rate": float(best.get("discovery_rate_mean", 0.0)),
                "best_so_far_score": float(best_so_far["score"]),
                "best_so_far_reward": float(best_so_far.get("reward_mean", best_so_far["score"])),
                "best_so_far_discovery_rate": float(best_so_far.get("discovery_rate_mean", 0.0)),
            }
        )
    return pd.DataFrame(rows)


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


def write_figures(history, elite, best_by_generation, figure_dir: Path) -> list[dict]:
    plt = setup_matplotlib()
    figures = []
    figures.append(
        plot_metric(
            history,
            elite,
            best_by_generation,
            "score",
            "best_so_far_score",
            "Selection score",
            figure_dir / "cem_step_score_curve.png",
            plt,
        )
    )
    figures.append(
        plot_metric(
            history,
            elite,
            best_by_generation,
            "reward_mean",
            "best_so_far_reward",
            "Episode reward",
            figure_dir / "cem_step_reward_curve.png",
            plt,
        )
    )
    figures.append(
        plot_metric(
            history,
            elite,
            best_by_generation,
            "discovery_rate_mean",
            "best_so_far_discovery_rate",
            "Discovery rate",
            figure_dir / "cem_step_discovery_curve.png",
            plt,
        )
    )
    figures.append(
        plot_metric(
            history,
            elite,
            best_by_generation,
            "empty_scan_ratio_mean",
            None,
            "Empty-scan ratio",
            figure_dir / "cem_step_empty_scan_curve.png",
            plt,
        )
    )
    return figures


def plot_metric(history, elite, best_by_generation, metric: str, best_metric: str | None, ylabel: str, path: Path, plt) -> dict:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(
        history["training_step"].to_numpy(),
        history[metric].to_numpy(),
        s=24,
        color=COLOR_CANDIDATE,
        alpha=0.55,
        label="Candidate evaluations",
    )
    elite_subset = elite[elite["rank"] == 1] if "rank" in elite.columns else elite
    ax.plot(
        elite_subset["training_step"].to_numpy(),
        elite_subset[metric].to_numpy(),
        marker="o",
        markersize=4,
        color=COLOR_ELITE,
        label="Generation best",
    )
    if best_metric is not None and best_metric in best_by_generation.columns:
        ax.step(
            best_by_generation["training_step"].to_numpy(),
            best_by_generation[best_metric].to_numpy(),
            where="post",
            color=COLOR_BEST,
            label="Best so far",
        )
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"path": str(path), "type": "cem_step_curve", "metric": metric}


def write_readme(output_dir: Path, manifest: dict, history, best_by_generation) -> None:
    best_row = history.sort_values("score", ascending=False).iloc[0]
    lines = [
        "# Step-Indexed Training Trajectory",
        "",
        f"- Created: {manifest['created_at']}",
        f"- Source: `{manifest['training_dir']}`",
        f"- Candidate evaluations: {len(history)}",
        f"- Final training step: {int(history['training_step'].max())}",
        f"- Generations: {int(history['generation'].nunique())}",
        f"- Best score: {float(best_row['score']):.4f}",
        f"- Best candidate: generation {int(best_row['generation'])}, candidate {int(best_row['candidate'])}",
        "",
        "Interpretation: the selected policy was produced by a CEM-style shared-parameter policy search.",
        "The x-axis is cumulative training environment steps, computed from candidate evaluations, episode length, episodes per seed, and training seeds.",
        "Each plotted point is still one full candidate-policy episode evaluation, not a per-gradient-update reward sample.",
        "Use these plots as step-indexed policy-search trace evidence only; do not describe them as a theoretical convergence proof.",
        "",
        "Generated files:",
        "- `candidate_evaluation_history.csv`",
        "- `elite_evaluation_history.csv`",
        "- `generation_best_history.csv`",
    ]
    if not best_by_generation.empty:
        last = best_by_generation.iloc[-1]
        lines.append(f"- Final best-so-far score: {float(last['best_so_far_score']):.4f}")
    output_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
