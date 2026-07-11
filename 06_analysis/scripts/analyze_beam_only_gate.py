from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "05_simulation" / "results_raw"
RUN_ROOT = RAW / "beam_only_shared_idqn_gate_v2_n10_b15_30ep_seed29260711"
OUTPUT = ROOT / "06_analysis" / "paper_tables" / "beam_only_gate_20260711"

PALETTE = {
    "standard_epsilon": "#0072B2",
    "persistent_mix_0.8": "#D55E00",
}
DISPLAY_LABELS = {
    "standard_epsilon": "Standard epsilon training",
    "persistent_mix_0.8": "Persistent 0.8 random-guidance training",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run: str, variant: str, rows: list[dict[str, str]]) -> dict[str, object]:
    tx = sum(float(row["tx_actions"]) for row in rows)
    scans = sum(float(row["scan_actions"]) for row in rows)
    return {
        "training_regime": run,
        "evaluation_variant": variant,
        "episodes": len(rows),
        "discovery_rate": mean(rows, "discovery_rate"),
        "neighbor_knowledge_recall": mean(rows, "neighbor_knowledge_recall"),
        "mean_delay_censored": mean(rows, "mean_delay_censored"),
        "empty_scan_ratio": mean(rows, "empty_scan_ratio"),
        "aligned_handshake_opportunities": mean(rows, "aligned_handshake_opportunities"),
        "tx_fraction": tx / scans,
        "evaluation_beam_random_mixture": float(rows[0]["beam_uniform_mixture"]),
    }


def paired_comparison(
    label: str,
    first: list[dict[str, str]],
    second: list[dict[str, str]],
) -> dict[str, object]:
    if len(first) != len(second):
        raise ValueError(f"{label}: paired variants have different row counts.")
    first_by_seed = {int(row["scenario_seed"]): float(row["discovery_rate"]) for row in first}
    second_by_seed = {int(row["scenario_seed"]): float(row["discovery_rate"]) for row in second}
    if len(first_by_seed) != len(first) or len(second_by_seed) != len(second):
        raise ValueError(f"{label}: duplicate scenario seeds would overwrite paired samples.")
    if set(first_by_seed) != set(second_by_seed):
        raise ValueError(f"{label}: paired variants do not contain the same scenario seeds.")
    seeds = sorted(first_by_seed)
    differences = [first_by_seed[seed] - second_by_seed[seed] for seed in seeds]
    count = len(differences)
    if count != 10:
        raise ValueError(f"{label}: this diagnostic gate requires exactly 10 paired scenarios.")
    difference_mean = sum(differences) / count
    variance = sum((value - difference_mean) ** 2 for value in differences) / max(1, count - 1)
    critical = 2.262
    half_width = critical * (variance / count) ** 0.5
    observed = abs(difference_mean)
    sign_flip_values = (
        abs(sum(sign * value for sign, value in zip(signs, differences, strict=True)) / count)
        for signs in itertools.product((-1.0, 1.0), repeat=count)
    )
    p_value = sum(value >= observed - 1e-15 for value in sign_flip_values) / (2**count)
    return {
        "comparison": label,
        "paired_scenarios": count,
        "mean_discovery_difference": difference_mean,
        "ci95_low": difference_mean - half_width,
        "ci95_high": difference_mean + half_width,
        "exact_sign_flip_p": p_value,
    }


def add_holm_adjustment(rows: list[dict[str, object]]) -> None:
    ordered = sorted(
        enumerate(rows),
        key=lambda item: float(item[1]["exact_sign_flip_p"]),
    )
    running_max = 0.0
    family_size = len(rows)
    for rank, (original_index, row) in enumerate(ordered):
        adjusted = min(1.0, (family_size - rank) * float(row["exact_sign_flip_p"]))
        running_max = max(running_max, adjusted)
        rows[original_index]["holm_adjusted_p"] = running_max


def reconstruct_role_sequence(
    scenario_seed: int,
    *,
    n_agents: int = 10,
    slots: int = 300,
) -> tuple[int, int, str]:
    rng = np.random.default_rng(scenario_seed + 777)
    digest = hashlib.blake2b(digest_size=12)
    tx_actions = 0
    for _ in range(slots):
        roles = rng.random(n_agents) < 0.5
        tx_actions += int(roles.sum())
        digest.update(bytes(int(role) for role in roles))
    total_actions = n_agents * slots
    return tx_actions, total_actions - tx_actions, digest.hexdigest()


def configure_plot_style() -> None:
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


def plot_evaluation_mix(
    grouped: dict[str, dict[str, list[dict[str, str]]]],
) -> None:
    configure_plot_style()
    variants = (
        (0.0, "pure_learned_beam"),
        (0.2, "learned_beam_random_mix_0.2"),
        (0.5, "learned_beam_random_mix_0.5"),
        (0.8, "learned_beam_random_mix_0.8"),
        (1.0, "random_candidate_beam"),
    )
    fig, ax = plt.subplots(figsize=(4, 3))
    for run, by_variant in grouped.items():
        x_values = [mixture for mixture, _ in variants]
        samples = [
            np.asarray([float(row["discovery_rate"]) for row in by_variant[variant]]) * 100.0
            for _, variant in variants
        ]
        means = [float(values.mean()) for values in samples]
        ci95 = [
            2.262 * float(values.std(ddof=1)) / np.sqrt(len(values))
            if len(values) > 1
            else 0.0
            for values in samples
        ]
        ax.errorbar(
            x_values,
            means,
            yerr=ci95,
            color=PALETTE[run],
            marker="o",
            markersize=4,
            linewidth=1.5,
            capsize=2.5,
            label=DISPLAY_LABELS[run],
        )
    ax.set_xlabel("Evaluation candidate-random beam mixture")
    ax.set_ylabel("Discovery rate (%)")
    ax.set_xticks([0.0, 0.2, 0.5, 0.8, 1.0])
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, "beam_only_evaluation_random_mix")


def plot_training_curves(training_by_run: dict[str, list[dict[str, str]]]) -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(4, 3))
    for run, rows in training_by_run.items():
        episodes = np.asarray([int(row["episode"]) + 1 for row in rows])
        values = np.asarray([float(row["discovery_rate"]) * 100.0 for row in rows])
        window = min(5, len(values))
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        smoothed_episodes = episodes[window - 1 :]
        ax.plot(
            episodes,
            values,
            color=PALETTE[run],
            alpha=0.18,
            linewidth=0.8,
        )
        ax.plot(
            smoothed_episodes,
            smoothed,
            color=PALETTE[run],
            linewidth=1.6,
            label=f"{DISPLAY_LABELS[run]} (5-episode mean)",
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Training discovery rate (%)")
    ax.set_xlim(left=1)
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    save_figure(fig, "beam_only_training_discovery")


def main() -> None:
    paths = {
        "standard_epsilon": {
            "training": RUN_ROOT / "standard_epsilon",
            "evaluation": RUN_ROOT / "standard_epsilon_eval7",
        },
        "persistent_mix_0.8": {
            "training": RUN_ROOT / "persistent_mix_0p8",
            "evaluation": RUN_ROOT / "persistent_mix_0p8_eval7",
        },
    }
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    summaries: list[dict[str, object]] = []
    training_summaries: list[dict[str, object]] = []
    training_by_run: dict[str, list[dict[str, str]]] = {}
    contracts: list[dict[str, object]] = []
    role_audit: list[dict[str, object]] = []

    for run, run_paths in paths.items():
        evaluation_path = run_paths["evaluation"]
        training_path = run_paths["training"]
        with (evaluation_path / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        evaluation = read_rows(evaluation_path / "eval_episode_metrics.csv")
        variants = sorted({row["policy_variant"] for row in evaluation})
        grouped[run] = {}
        for variant in variants:
            selected = [row for row in evaluation if row["policy_variant"] == variant]
            grouped[run][variant] = selected
            summaries.append(summarize(run, variant, selected))

        by_seed: dict[int, list[dict[str, str]]] = {}
        for row in evaluation:
            by_seed.setdefault(int(row["scenario_seed"]), []).append(row)
        for seed, rows in by_seed.items():
            if len(rows) != 7:
                raise RuntimeError(f"Scenario {seed} has {len(rows)} variants; expected 7.")
            tx_values = {int(row["tx_actions"]) for row in rows}
            rx_values = {int(row["rx_actions"]) for row in rows}
            expected_tx, expected_rx, expected_hash = reconstruct_role_sequence(
                seed,
                n_agents=int(manifest["node_count"]),
                slots=int(manifest["slots_per_episode"]),
            )
            recorded_hash_values = [row.get("role_sequence_hash", "") for row in rows]
            all_hashes_recorded = all(recorded_hash_values)
            recorded_hashes = set(recorded_hash_values)
            counts_match_reconstruction = (
                tx_values == {expected_tx} and rx_values == {expected_rx}
            )
            hashes_match_reconstruction = all_hashes_recorded and recorded_hashes == {expected_hash}
            role_audit.append(
                {
                    "training_regime": run,
                    "scenario_seed": seed,
                    "variant_count": len(rows),
                    "identical_tx_count_across_variants": len(tx_values) == 1,
                    "identical_rx_count_across_variants": len(rx_values) == 1,
                    "counts_match_seeded_reconstruction": counts_match_reconstruction,
                    "all_role_hashes_recorded": all_hashes_recorded,
                    "identical_role_sequence_across_variants": len(recorded_hashes) == 1,
                    "recorded_hash_matches_reconstruction": hashes_match_reconstruction,
                    "role_sequence_proof": "recorded_hash",
                    "expected_role_sequence_hash": expected_hash,
                    "tx_actions": next(iter(tx_values)) if len(tx_values) == 1 else "mismatch",
                    "rx_actions": next(iter(rx_values)) if len(rx_values) == 1 else "mismatch",
                    "expected_tx_actions": expected_tx,
                    "expected_rx_actions": expected_rx,
                }
            )

        training = read_rows(training_path / "episode_metrics.csv")
        training_by_run[run] = training
        training_summaries.append(
            {
                "training_regime": run,
                "episodes": len(training),
                "first_10_discovery_rate": mean(training[:10], "discovery_rate"),
                "last_10_discovery_rate": mean(training[-10:], "discovery_rate"),
                "first_10_return_mean_per_agent": mean(training[:10], "episode_return_mean_per_agent"),
                "last_10_return_mean_per_agent": mean(training[-10:], "episode_return_mean_per_agent"),
                "last_10_td_loss": mean(training[-10:], "td_loss"),
            }
        )
        support = manifest["stochastic_support"]
        contracts.append(
            {
                "training_regime": run,
                "action_contract": manifest["action_contract"],
                "role_policy": manifest["role_policy"],
                "role_learned": support["role_learned"],
                "fixed_tx_probability": support["fixed_tx_probability"],
                "candidate_source": manifest["candidate_source"],
                "training_beam_random_floor": support["beam_uniform_mixture"],
                "training_epsilon_start": support["training_epsilon_start"],
                "training_epsilon_end": support["training_epsilon_end"],
                "training_epsilon_decay_steps": support["training_epsilon_decay_steps"],
                "beam_randomization_domain": support["beam_randomization_domain"],
                "beam_gate_rng_separate_from_choice_rng": support[
                    "beam_gate_rng_separate_from_choice_rng"
                ],
                "source_checkpoint": manifest["source_checkpoint"],
                "source_checkpoint_sha256": manifest["source_checkpoint_sha256"],
                "evaluation_git_commit": manifest["git_commit"],
                "tracked_worktree_dirty": manifest["tracked_worktree_dirty"],
            }
        )

    if not all(row["identical_tx_count_across_variants"] for row in role_audit):
        raise RuntimeError("Beam evaluation variants did not share the same TX sequence.")
    if not all(row["identical_rx_count_across_variants"] for row in role_audit):
        raise RuntimeError("Beam evaluation variants did not share the same RX sequence.")
    if not all(row["counts_match_seeded_reconstruction"] for row in role_audit):
        raise RuntimeError("Role counts do not match the seeded Bernoulli(0.5) reconstruction.")
    if not all(row["all_role_hashes_recorded"] for row in role_audit):
        raise RuntimeError("Every evaluation row must record a role-sequence hash.")
    if not all(row["identical_role_sequence_across_variants"] for row in role_audit):
        raise RuntimeError("Beam variants did not use identical per-slot role sequences.")
    if not all(row["recorded_hash_matches_reconstruction"] for row in role_audit):
        raise RuntimeError("Recorded role hashes do not match the seeded reconstruction.")

    comparisons: list[dict[str, object]] = []
    for run, variants in grouped.items():
        random_rows = variants["random_candidate_beam"]
        for variant in (
            "pure_learned_beam",
            "learned_beam_random_mix_0.2",
            "learned_beam_random_mix_0.5",
            "learned_beam_random_mix_0.8",
        ):
            comparisons.append(
                paired_comparison(
                    f"{run}:{variant} - {run}:random_candidate_beam",
                    variants[variant],
                    random_rows,
                )
            )
        for rule_variant in ("candidate_score_argmax", "candidate_score_proportional"):
            comparisons.append(
                paired_comparison(
                    f"{run}:pure_learned_beam - {run}:{rule_variant}",
                    variants["pure_learned_beam"],
                    variants[rule_variant],
                )
            )
    add_holm_adjustment(comparisons)
    checkpoint_contrast = [
        {
            "contrast": "standard_epsilon:pure - persistent_mix_0.8:pure",
            "standard_checkpoint_discovery_rate": mean(
                grouped["standard_epsilon"]["pure_learned_beam"], "discovery_rate"
            ),
            "persistent_checkpoint_discovery_rate": mean(
                grouped["persistent_mix_0.8"]["pure_learned_beam"], "discovery_rate"
            ),
            "descriptive_difference_only": mean(
                grouped["standard_epsilon"]["pure_learned_beam"], "discovery_rate"
            )
            - mean(grouped["persistent_mix_0.8"]["pure_learned_beam"], "discovery_rate"),
            "inference_status": "not_tested_requires_independent_training_seeds",
        }
    ]

    control_audit: list[dict[str, object]] = []
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
    for variant in (
        "random_candidate_beam",
        "candidate_score_argmax",
        "candidate_score_proportional",
    ):
        standard = {
            int(row["scenario_seed"]): row for row in grouped["standard_epsilon"][variant]
        }
        persistent = {
            int(row["scenario_seed"]): row for row in grouped["persistent_mix_0.8"][variant]
        }
        if set(standard) != set(persistent):
            raise RuntimeError(f"{variant}: checkpoint controls have different scenario seeds.")
        for seed in sorted(standard):
            matching_fields = all(
                standard[seed][field] == persistent[seed][field] for field in stable_fields
            )
            control_audit.append(
                {
                    "policy_variant": variant,
                    "scenario_seed": seed,
                    "identical_across_checkpoints": matching_fields,
                    "fields_checked": "|".join(stable_fields),
                }
            )
    if not all(row["identical_across_checkpoints"] for row in control_audit):
        raise RuntimeError("A non-learning control changed across checkpoints.")

    write_rows(OUTPUT / "evaluation_summary.csv", summaries)
    write_rows(OUTPUT / "paired_comparisons.csv", comparisons)
    write_rows(OUTPUT / "checkpoint_descriptive_contrast.csv", checkpoint_contrast)
    write_rows(OUTPUT / "training_summary.csv", training_summaries)
    write_rows(OUTPUT / "run_contracts.csv", contracts)
    write_rows(OUTPUT / "role_sequence_audit.csv", role_audit)
    write_rows(OUTPUT / "nonlearning_control_reproducibility.csv", control_audit)
    plot_evaluation_mix(grouped)
    plot_training_curves(training_by_run)


if __name__ == "__main__":
    main()
