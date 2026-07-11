from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "beam_only_algorithm_gate_n10_b15_30ep_seed29260711"
)
OUTPUT = ROOT / "06_analysis" / "paper_tables" / "beam_only_algorithm_gate_20260711"
ALGORITHMS = {
    "shared_idqn": "Shared-IDQN",
    "vdn": "VDN",
    "qmix": "QMIX",
    "mappo": "Beam-only MAPPO",
}
COLORS = {
    "shared_idqn": "#0072B2",
    "vdn": "#E69F00",
    "qmix": "#009E73",
    "mappo": "#CC79A7",
}
MIX_VARIANTS = (
    (0.0, "pure_learned_beam"),
    (0.2, "learned_beam_random_mix_0.2"),
    (0.5, "learned_beam_random_mix_0.5"),
    (0.8, "learned_beam_random_mix_0.8"),
    (1.0, "random_candidate_beam"),
)
NONLEARNING_VARIANTS = (
    "random_candidate_beam",
    "candidate_score_argmax",
    "candidate_score_proportional",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def paired_comparison(
    label: str,
    first: list[dict[str, str]],
    second: list[dict[str, str]],
) -> dict[str, object]:
    if len(first) != 10 or len(second) != 10:
        raise ValueError(f"{label}: expected exactly ten rows per variant.")
    first_by_seed = {int(row["scenario_seed"]): float(row["discovery_rate"]) for row in first}
    second_by_seed = {int(row["scenario_seed"]): float(row["discovery_rate"]) for row in second}
    if len(first_by_seed) != 10 or len(second_by_seed) != 10 or set(first_by_seed) != set(second_by_seed):
        raise ValueError(f"{label}: duplicate or unmatched paired scenario seeds.")
    differences = [first_by_seed[seed] - second_by_seed[seed] for seed in sorted(first_by_seed)]
    difference_mean = float(np.mean(differences))
    half_width = 2.262 * float(np.std(differences, ddof=1)) / np.sqrt(len(differences))
    observed = abs(difference_mean)
    sign_flips = (
        abs(sum(sign * value for sign, value in zip(signs, differences, strict=True)) / len(differences))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    )
    p_value = sum(value >= observed - 1e-15 for value in sign_flips) / (2 ** len(differences))
    return {
        "comparison": label,
        "paired_scenarios": len(differences),
        "mean_discovery_difference": difference_mean,
        "ci95_low": difference_mean - half_width,
        "ci95_high": difference_mean + half_width,
        "exact_sign_flip_p": p_value,
    }


def add_holm(rows: list[dict[str, object]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1]["exact_sign_flip_p"]))
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (len(rows) - rank) * float(row["exact_sign_flip_p"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 7.3,
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


def plot_mix(grouped: dict[str, dict[str, list[dict[str, str]]]]) -> None:
    configure_plot()
    fig, ax = plt.subplots(figsize=(4, 3))
    for algorithm, variants in grouped.items():
        samples = [
            np.asarray([float(row["discovery_rate"]) for row in variants[variant]]) * 100.0
            for _, variant in MIX_VARIANTS
        ]
        means = [float(values.mean()) for values in samples]
        ci95 = [2.262 * float(values.std(ddof=1)) / np.sqrt(len(values)) for values in samples]
        ax.errorbar(
            [mixture for mixture, _ in MIX_VARIANTS],
            means,
            yerr=ci95,
            color=COLORS[algorithm],
            marker="o",
            markersize=3.5,
            linewidth=1.35,
            capsize=2,
            label=ALGORITHMS[algorithm],
        )
    ax.set_xlabel("Evaluation candidate-random beam mixture")
    ax.set_ylabel("Discovery rate (%)")
    ax.set_xticks([0.0, 0.2, 0.5, 0.8, 1.0])
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    save_figure(fig, "algorithm_evaluation_random_mix")


def plot_pure_and_controls(grouped: dict[str, dict[str, list[dict[str, str]]]]) -> None:
    configure_plot()
    fig, ax = plt.subplots(figsize=(4, 3))
    x_values = np.arange(len(ALGORITHMS))
    pure_samples = [
        np.asarray([float(row["discovery_rate"]) for row in grouped[algorithm]["pure_learned_beam"]])
        * 100.0
        for algorithm in ALGORITHMS
    ]
    pure_means = [float(values.mean()) for values in pure_samples]
    pure_ci = [2.262 * float(values.std(ddof=1)) / np.sqrt(len(values)) for values in pure_samples]
    ax.bar(
        x_values,
        pure_means,
        yerr=pure_ci,
        color=[COLORS[algorithm] for algorithm in ALGORITHMS],
        width=0.62,
        capsize=3,
        label="Pure learned beam",
    )
    random_mean = mean(grouped["shared_idqn"]["random_candidate_beam"], "discovery_rate") * 100.0
    score_mean = mean(
        grouped["shared_idqn"]["candidate_score_proportional"], "discovery_rate"
    ) * 100.0
    ax.axhline(random_mean, color="#4D4D4D", linestyle="--", linewidth=1.2, label="Candidate random")
    ax.axhline(score_mean, color="#A65628", linestyle=":", linewidth=1.4, label="Score proportional")
    ax.set_xticks(x_values, [ALGORITHMS[key] for key in ALGORITHMS], rotation=15, ha="right")
    ax.set_ylabel("Discovery rate (%)")
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, "algorithm_pure_vs_nonlearning_controls")


def plot_training(training: dict[str, list[dict[str, str]]]) -> None:
    configure_plot()
    fig, ax = plt.subplots(figsize=(4, 3))
    for algorithm, rows in training.items():
        episodes = np.asarray([int(row["episode"]) + 1 for row in rows])
        values = np.asarray([float(row["discovery_rate"]) * 100.0 for row in rows])
        window = min(5, len(values))
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1 :],
            smoothed,
            color=COLORS[algorithm],
            linewidth=1.5,
            label=ALGORITHMS[algorithm],
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Training discovery rate (%)")
    ax.set_xlim(left=1)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    save_figure(fig, "algorithm_training_discovery")


def main() -> None:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    training: dict[str, list[dict[str, str]]] = {}
    summaries: list[dict[str, object]] = []
    training_summary: list[dict[str, object]] = []
    contracts: list[dict[str, object]] = []

    for algorithm in ALGORITHMS:
        with (RUN_ROOT / algorithm / "eval7" / "manifest.json").open(encoding="utf-8") as handle:
            evaluation_manifest = json.load(handle)
        with (RUN_ROOT / algorithm / "train" / "manifest.json").open(encoding="utf-8") as handle:
            training_manifest = json.load(handle)
        contracts.append(
            {
                "algorithm": algorithm,
                "training_git_commit": training_manifest["git_commit"],
                "evaluation_git_commit": evaluation_manifest["git_commit"],
                "checkpoint_sha256": evaluation_manifest["checkpoint_sha256"],
                "action_contract": evaluation_manifest["action_contract"],
                "candidate_source": evaluation_manifest["candidate_source"],
                "training_exploration": json.dumps(
                    evaluation_manifest["training_exploration"],
                    sort_keys=True,
                ),
            }
        )
        evaluation = read_rows(RUN_ROOT / algorithm / "eval7" / "eval_episode_metrics.csv")
        if len(evaluation) != 70:
            raise RuntimeError(f"{algorithm}: expected 70 evaluation rows, found {len(evaluation)}.")
        grouped[algorithm] = {}
        for variant in sorted({row["policy_variant"] for row in evaluation}):
            rows = [row for row in evaluation if row["policy_variant"] == variant]
            if len(rows) != 10:
                raise RuntimeError(f"{algorithm}/{variant}: expected 10 scenarios.")
            grouped[algorithm][variant] = rows
            summaries.append(
                {
                    "algorithm": algorithm,
                    "display_name": ALGORITHMS[algorithm],
                    "policy_variant": variant,
                    "episodes": len(rows),
                    "discovery_rate": mean(rows, "discovery_rate"),
                    "neighbor_knowledge_recall": mean(rows, "neighbor_knowledge_recall"),
                    "mean_delay_censored": mean(rows, "mean_delay_censored"),
                    "empty_scan_ratio": mean(rows, "empty_scan_ratio"),
                    "aligned_handshake_opportunities": mean(rows, "aligned_handshake_opportunities"),
                    "evaluation_beam_random_mixture": float(rows[0]["beam_uniform_mixture"]),
                }
            )
        training_rows = read_rows(RUN_ROOT / algorithm / "train" / "episode_metrics.csv")
        if len(training_rows) != 30:
            raise RuntimeError(f"{algorithm}: expected 30 training episodes.")
        training[algorithm] = training_rows
        training_summary.append(
            {
                "algorithm": algorithm,
                "episodes": len(training_rows),
                "first_10_discovery_rate": mean(training_rows[:10], "discovery_rate"),
                "last_10_discovery_rate": mean(training_rows[-10:], "discovery_rate"),
                "first_10_return_mean_per_agent": mean(
                    training_rows[:10], "episode_return_mean_per_agent"
                ),
                "last_10_return_mean_per_agent": mean(
                    training_rows[-10:], "episode_return_mean_per_agent"
                ),
            }
        )

    training_role_audit: list[dict[str, object]] = []
    for episode in range(30):
        hashes = {algorithm: training[algorithm][episode]["role_sequence_hash"] for algorithm in ALGORITHMS}
        identical = len(set(hashes.values())) == 1
        training_role_audit.append(
            {
                "episode": episode,
                "identical_role_sequence_across_algorithms": identical,
                "role_sequence_hash": next(iter(hashes.values())) if identical else "mismatch",
            }
        )
    if not all(row["identical_role_sequence_across_algorithms"] for row in training_role_audit):
        raise RuntimeError("Training role sequences differ across algorithms.")

    evaluation_role_audit: list[dict[str, object]] = []
    for scenario_index in range(10):
        scenario_seed = 29260711 + 2_000_000 + scenario_index
        selected = [
            row
            for algorithm in ALGORITHMS
            for rows in grouped[algorithm].values()
            for row in rows
            if int(row["scenario_seed"]) == scenario_seed
        ]
        hashes = {row["role_sequence_hash"] for row in selected}
        evaluation_role_audit.append(
            {
                "scenario_seed": scenario_seed,
                "rows_checked": len(selected),
                "identical_role_sequence": len(hashes) == 1,
                "role_sequence_hash": next(iter(hashes)) if len(hashes) == 1 else "mismatch",
            }
        )
    if not all(row["identical_role_sequence"] for row in evaluation_role_audit):
        raise RuntimeError("Evaluation role sequences differ across algorithms or variants.")

    stable_fields = (
        "discovery_rate",
        "neighbor_knowledge_recall",
        "mean_delay_censored",
        "empty_scan_ratio",
        "tx_actions",
        "rx_actions",
        "role_sequence_hash",
        "beam_sequence_hash",
        "candidate_mask_sequence_hash",
    )
    control_audit: list[dict[str, object]] = []
    for variant in NONLEARNING_VARIANTS:
        for scenario_index in range(10):
            scenario_seed = 29260711 + 2_000_000 + scenario_index
            rows = [
                next(
                    row
                    for row in grouped[algorithm][variant]
                    if int(row["scenario_seed"]) == scenario_seed
                )
                for algorithm in ALGORITHMS
            ]
            identical = all(
                row[field] == rows[0][field] for row in rows[1:] for field in stable_fields
            )
            control_audit.append(
                {
                    "policy_variant": variant,
                    "scenario_seed": scenario_seed,
                    "identical_across_algorithms": identical,
                    "fields_checked": "|".join(stable_fields),
                }
            )
    if not all(row["identical_across_algorithms"] for row in control_audit):
        raise RuntimeError("A non-learning control changed across algorithms.")

    comparisons: list[dict[str, object]] = []
    for algorithm, variants in grouped.items():
        for control in NONLEARNING_VARIANTS:
            comparisons.append(
                paired_comparison(
                    f"{algorithm}:pure_learned_beam - {control}",
                    variants["pure_learned_beam"],
                    variants[control],
                )
            )
    add_holm(comparisons)

    fixed_checkpoint_contrasts: list[dict[str, object]] = []
    algorithm_keys = list(ALGORITHMS)
    for first_index, first in enumerate(algorithm_keys):
        for second in algorithm_keys[first_index + 1 :]:
            fixed_checkpoint_contrasts.append(
                {
                    "contrast": f"{first}:pure - {second}:pure",
                    "first_discovery_rate": mean(grouped[first]["pure_learned_beam"], "discovery_rate"),
                    "second_discovery_rate": mean(grouped[second]["pure_learned_beam"], "discovery_rate"),
                    "descriptive_difference_only": mean(
                        grouped[first]["pure_learned_beam"], "discovery_rate"
                    )
                    - mean(grouped[second]["pure_learned_beam"], "discovery_rate"),
                    "inference_status": "not_tested_requires_independent_training_seeds",
                }
            )

    write_rows(OUTPUT / "evaluation_summary.csv", summaries)
    write_rows(OUTPUT / "training_summary.csv", training_summary)
    write_rows(OUTPUT / "run_contracts.csv", contracts)
    write_rows(OUTPUT / "paired_control_comparisons.csv", comparisons)
    write_rows(OUTPUT / "fixed_checkpoint_algorithm_contrasts.csv", fixed_checkpoint_contrasts)
    write_rows(OUTPUT / "training_role_sequence_audit.csv", training_role_audit)
    write_rows(OUTPUT / "evaluation_role_sequence_audit.csv", evaluation_role_audit)
    write_rows(OUTPUT / "nonlearning_control_reproducibility.csv", control_audit)
    plot_mix(grouped)
    plot_pure_and_controls(grouped)
    plot_training(training)


if __name__ == "__main__":
    main()
