from __future__ import annotations

import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "beam_only_mappo_100200_n10_b15_seed29260711_v2"
)
OUTPUT = ROOT / "06_analysis" / "paper_tables" / "mappo_100k_gate_20260711"
REPORT = ROOT / "06_analysis" / "mappo_100k_gate_20260711.md"
STAGES = {
    "30k": RUN / "eval_ep00100" / "eval_episode_metrics.csv",
    "60k": RUN / "eval_ep00200" / "eval_episode_metrics.csv",
    "100.2k": RUN / "eval_final" / "eval_episode_metrics.csv",
}
VARIANTS = (
    "pure_learned_beam",
    "random_candidate_beam",
    "candidate_score_proportional",
    "candidate_score_argmax",
    "learned_beam_random_mix_0.2",
    "learned_beam_random_mix_0.5",
    "learned_beam_random_mix_0.8",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    if values.size < width:
        return np.full(values.shape, np.nan)
    result = np.full(values.shape, np.nan)
    result[width - 1 :] = np.convolve(values, np.ones(width) / width, mode="valid")
    return result


def plot_training(rows: list[dict[str, str]]) -> None:
    configure_plot()
    steps = np.asarray([float(row["training_step"]) for row in rows])
    discovery = 100.0 * np.asarray([float(row["discovery_rate"]) for row in rows])
    returns = np.asarray([float(row["episode_return_mean_per_agent"]) for row in rows])

    for values, ylabel, stem, color in (
        (discovery, "Training discovery rate (%)", "training_discovery_vs_step", "#0072B2"),
        (returns, "Episode return per UAV", "training_return_vs_step", "#009E73"),
    ):
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(steps, values, color=color, alpha=0.18, linewidth=0.7, label="Per episode")
        ax.plot(steps, rolling_mean(values, 20), color=color, linewidth=1.5, label="20-episode mean")
        ax.set_xlabel("Training environment step")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        save_figure(fig, stem)


def summarize_evaluations() -> tuple[list[dict[str, object]], dict[str, dict[str, list[dict[str, str]]]]]:
    summary: list[dict[str, object]] = []
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for stage, path in STAGES.items():
        rows = read_rows(path)
        grouped[stage] = {}
        for variant in VARIANTS:
            selected = [row for row in rows if row["policy_variant"] == variant]
            if len(selected) != 10:
                raise ValueError(f"{stage}/{variant}: expected ten paired scenarios, got {len(selected)}")
            grouped[stage][variant] = selected
            discovery = np.asarray([float(row["discovery_rate"]) for row in selected])
            summary.append(
                {
                    "checkpoint_stage": stage,
                    "policy_variant": variant,
                    "episodes": len(selected),
                    "discovery_rate": float(discovery.mean()),
                    "discovery_ci95_half_width": 2.262 * float(discovery.std(ddof=1)) / np.sqrt(len(discovery)),
                    "neighbor_knowledge_recall": float(
                        np.mean([float(row["neighbor_knowledge_recall"]) for row in selected])
                    ),
                    "mean_delay_censored": float(
                        np.mean([float(row["mean_delay_censored"]) for row in selected])
                    ),
                    "aligned_handshake_opportunities": float(
                        np.mean([float(row["aligned_handshake_opportunities"]) for row in selected])
                    ),
                }
            )
    return summary, grouped


def paired_test(
    stage: str,
    learned: list[dict[str, str]],
    control: list[dict[str, str]],
    control_name: str,
) -> dict[str, object]:
    learned_by_seed = {int(row["scenario_seed"]): float(row["discovery_rate"]) for row in learned}
    control_by_seed = {int(row["scenario_seed"]): float(row["discovery_rate"]) for row in control}
    if set(learned_by_seed) != set(control_by_seed):
        raise ValueError(f"{stage}/{control_name}: paired scenario seeds do not match")
    differences = np.asarray(
        [learned_by_seed[seed] - control_by_seed[seed] for seed in sorted(learned_by_seed)]
    )
    mean_difference = float(differences.mean())
    half_width = 2.262 * float(differences.std(ddof=1)) / np.sqrt(differences.size)
    observed = abs(mean_difference)
    permutations = (
        abs(float(np.mean(np.asarray(signs) * differences)))
        for signs in itertools.product((-1.0, 1.0), repeat=differences.size)
    )
    p_value = sum(value >= observed - 1e-15 for value in permutations) / (2**differences.size)
    return {
        "checkpoint_stage": stage,
        "comparison": f"pure_learned_beam - {control_name}",
        "paired_scenarios": int(differences.size),
        "mean_difference": mean_difference,
        "ci95_low": mean_difference - half_width,
        "ci95_high": mean_difference + half_width,
        "exact_sign_flip_p": p_value,
    }


def plot_checkpoint_gate(summary: list[dict[str, object]]) -> None:
    configure_plot()
    stages = list(STAGES)

    def values(variant: str) -> list[float]:
        return [
            100.0
            * float(
                next(
                    row["discovery_rate"]
                    for row in summary
                    if row["checkpoint_stage"] == stage and row["policy_variant"] == variant
                )
            )
            for stage in stages
        ]

    fig, ax = plt.subplots(figsize=(4, 3))
    x_values = np.arange(len(stages))
    ax.plot(x_values, values("pure_learned_beam"), marker="o", color="#0072B2", linewidth=1.6, label="Pure learned")
    ax.plot(x_values, values("random_candidate_beam"), linestyle="--", color="#D55E00", linewidth=1.2, label="Candidate random")
    ax.plot(
        x_values,
        values("candidate_score_proportional"),
        linestyle="-.",
        color="#009E73",
        linewidth=1.2,
        label="Score proportional",
    )
    ax.set_xticks(x_values, stages)
    ax.set_xlabel("Training environment step")
    ax.set_ylabel("Discovery rate (%)")
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, "checkpoint_discovery_gate")


def main() -> None:
    training_rows = read_rows(RUN / "episode_metrics.csv")
    if len(training_rows) != 334 or int(training_rows[-1]["training_step"]) != 100_200:
        raise ValueError("The expected 334-episode, 100,200-step training run is incomplete.")

    summary, grouped = summarize_evaluations()
    comparisons = []
    for stage in STAGES:
        comparisons.extend(
            [
                paired_test(
                    stage,
                    grouped[stage]["pure_learned_beam"],
                    grouped[stage][control],
                    control,
                )
                for control in ("random_candidate_beam", "candidate_score_proportional")
            ]
        )

    write_rows(OUTPUT / "checkpoint_evaluation_summary.csv", summary)
    write_rows(OUTPUT / "paired_control_comparisons.csv", comparisons)
    plot_training(training_rows)
    plot_checkpoint_gate(summary)

    def metric(stage: str, variant: str) -> float:
        return 100.0 * float(
            next(
                row["discovery_rate"]
                for row in summary
                if row["checkpoint_stage"] == stage and row["policy_variant"] == variant
            )
        )

    first50 = training_rows[:50]
    last50 = training_rows[-50:]
    mean_abs_kl = float(np.mean([abs(float(row["approx_kl"])) for row in training_rows]))
    zero_clip_fraction = float(np.mean([float(row["clip_fraction"]) == 0.0 for row in training_rows]))
    report = f"""# Beam-only MAPPO 100k sample-budget gate

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-11
- Verification Status: ANALYZED
- Scope: one MAPPO training seed, 334 episodes x 300 slots, 10 paired held-out scenarios per checkpoint

## Contract

The run contains 100,200 training environment steps (`N=10`, 15-degree planar codebook, fixed independent TX/RX probability 0.5). The actor selects only a beam from decentralized local observations. ISAC residual-table measurements and post-handshake exchanged tables are available; rule residuals, behavior cloning, rendezvous action targets, mode learning, and access-gate learning are disabled.

## Results

| Checkpoint | Pure learned | Candidate random | Score proportional |
|---|---:|---:|---:|
| 30k | {metric('30k', 'pure_learned_beam'):.2f}% | {metric('30k', 'random_candidate_beam'):.2f}% | {metric('30k', 'candidate_score_proportional'):.2f}% |
| 60k | {metric('60k', 'pure_learned_beam'):.2f}% | {metric('60k', 'random_candidate_beam'):.2f}% | {metric('60k', 'candidate_score_proportional'):.2f}% |
| 100.2k | {metric('100.2k', 'pure_learned_beam'):.2f}% | {metric('100.2k', 'random_candidate_beam'):.2f}% | {metric('100.2k', 'candidate_score_proportional'):.2f}% |

The first 50 training episodes average {100.0 * np.mean([float(row['discovery_rate']) for row in first50]):.2f}% discovery; the last 50 average {100.0 * np.mean([float(row['discovery_rate']) for row in last50]):.2f}%. Mean absolute PPO KL is {mean_abs_kl:.3e}, and {100.0 * zero_clip_fraction:.1f}% of episode updates have zero clip fraction. This indicates numerically stable but weak actor updates, not a clear convergence advantage.

## Interpretation

The 100k budget tests whether sample count alone rescues the current feed-forward MAPPO. It does not establish an algorithm-level advantage because there is only one training seed. The fixed final checkpoint must be reported even if an intermediate checkpoint is better. A mixed policy with a high random fraction is not evidence that the learned beam policy is effective.

## Statistical Fallacy Scan

Coverage: **11/11 checked**. The report avoids cross-configuration claims, individual-from-team inference, selective checkpoint reporting, denominator changes, post-treatment controls, uncorrected multiple-algorithm claims, causal attribution from one seed, and transfer/robustness claims. Paired tests are descriptive because ten evaluation scenarios do not replace independent training seeds.

## Artifacts

- `05_simulation/results_raw/beam_only_mappo_100200_n10_b15_seed29260711_v2/`
- `06_analysis/paper_tables/mappo_100k_gate_20260711/checkpoint_evaluation_summary.csv`
- `06_analysis/paper_tables/mappo_100k_gate_20260711/paired_control_comparisons.csv`
- `06_analysis/paper_tables/mappo_100k_gate_20260711/training_discovery_vs_step.png`
- `06_analysis/paper_tables/mappo_100k_gate_20260711/training_return_vs_step.png`
- `06_analysis/paper_tables/mappo_100k_gate_20260711/checkpoint_discovery_gate.png`
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
