from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FACTORIZATIONS = (
    "independent",
    "beam_conditioned",
    "beam_conditioned_antisymmetric",
)
DEFAULT_SEEDS = (29260711, 29261720, 29262729)
COLORS = {
    "independent": "#0072B2",
    "beam_conditioned": "#D55E00",
    "beam_conditioned_antisymmetric": "#009E73",
}
LABELS = {
    "independent": "Independent role",
    "beam_conditioned": "Beam-conditioned role",
    "beam_conditioned_antisymmetric": "Antisymmetric conditioned role",
}
ROW_METRICS = (
    "discovery_rate",
    "mean_delay_censored",
    "episode_return_mean_per_agent",
)
SEED_METRICS = (
    *ROW_METRICS,
    "beam_alignment_rate_active_ratio_of_sums",
    "aligned_role_complementarity_rate_ratio_of_sums",
    "handshake_success_conversion_rate_ratio_of_sums",
)
T_CRITICAL_95_DF2 = 4.302652729696142


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and aggregate the paired N=2, B=8 no-ISAC RMAPPO role-factorization matrix."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Exactly three comma-separated training seeds.",
    )
    parser.add_argument("--rolling-window", type=int, default=250)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_seeds(text: str) -> tuple[int, int, int]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("--seeds must contain exactly three distinct integers.")
    return seeds  # type: ignore[return-value]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def command_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag)
    return command[index + 1] if index + 1 < len(command) else None


def normalized_command(command: list[str]) -> list[str]:
    ignored_value_flags = {"--output", "--role-factorization"}
    normalized: list[str] = []
    index = 2
    while index < len(command):
        token = command[index]
        if token in ignored_value_flags:
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def validate_manifest(
    manifest: dict[str, Any],
    factorization: str,
    seed: int,
) -> None:
    prefix = f"{factorization}/seed_{seed}"
    expected = {
        "algorithm": "mappo",
        "action_contract": "joint_role_beam",
        "network": "recurrent_contention_shared",
        "role_factorization": factorization,
        "env_protocol": "structured_marl_no_isac",
        "seed": seed,
        "node_count": 2,
        "beam_count": 8,
        "azimuth_cells": 8,
        "elevation_cells": 1,
        "slots_per_episode": 16,
        "role_policy": "learned_mode",
    }
    for key, value in expected.items():
        require(manifest.get(key) == value, f"{prefix}: expected {key}={value!r}, got {manifest.get(key)!r}")
    require(bool(manifest.get("decoupled_role_tower")), f"{prefix}: decoupled role tower is required")
    require(bool(manifest.get("clean_ctde")), f"{prefix}: clean CTDE validation is required")
    require(not bool(manifest.get("rendezvous_observation_enabled")), f"{prefix}: rendezvous observations must be disabled")
    require(set(manifest.get("disabled_modes", ())) == {"sense", "idle"}, f"{prefix}: expected TX/RX-only execution")
    feature_flags = manifest.get("feature_flags", {})
    require(
        all(not bool(feature_flags.get(name)) for name in ("candidate_mask", "candidate_score", "topology_deficit", "rule_residual")),
        f"{prefix}: all ISAC/structured actor feature flags must be disabled",
    )
    command = [str(value) for value in manifest.get("command", ())]
    for flag in ("--disable-isac-features", "--no-candidate-score", "--no-rendezvous-observation", "--forbid-sense"):
        require(flag in command, f"{prefix}: training command is missing {flag}")
    require(command_value(command, "--training-scenario-mode") == "varying", f"{prefix}: training scenarios must vary")
    require(command_value(command, "--evaluation-scenario-mode") == "held_out", f"{prefix}: evaluation scenarios must be held out")
    require(command_value(command, "--beam-uniform-mixture") == "1.0", f"{prefix}: beam policy must be uniformly randomized")
    require("--eval-both" in command, f"{prefix}: deterministic and stochastic evaluation are both required")


def load_matrix(
    run_root: Path,
    seeds: tuple[int, int, int],
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[tuple[str, int], list[dict[str, str]]],
    dict[tuple[str, int], list[dict[str, str]]],
]:
    manifests: dict[tuple[str, int], dict[str, Any]] = {}
    training: dict[tuple[str, int], list[dict[str, str]]] = {}
    evaluation: dict[tuple[str, int], list[dict[str, str]]] = {}
    for factorization in FACTORIZATIONS:
        for seed in seeds:
            directory = run_root / factorization / f"seed_{seed}"
            paths = {
                "manifest": directory / "manifest.json",
                "training": directory / "episode_metrics.csv",
                "evaluation": directory / "eval_episode_metrics.csv",
            }
            for label, path in paths.items():
                require(path.is_file(), f"Missing {label} artifact: {path}")
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            train_rows = read_csv(paths["training"])
            eval_rows = read_csv(paths["evaluation"])
            validate_manifest(manifest, factorization, seed)
            expected_episodes = int(manifest["episodes"])
            expected_steps = expected_episodes * int(manifest["slots_per_episode"])
            require(len(train_rows) == expected_episodes, f"{directory}: incomplete training CSV")
            require(int(float(train_rows[-1]["training_step"])) == expected_steps, f"{directory}: expected {expected_steps} environment steps")
            eval_episodes = int(manifest["command"][manifest["command"].index("--eval-episodes") + 1])
            require(len(eval_rows) == 2 * eval_episodes, f"{directory}: incomplete deterministic/stochastic evaluation")
            manifests[(factorization, seed)] = manifest
            training[(factorization, seed)] = train_rows
            evaluation[(factorization, seed)] = eval_rows

    for seed in seeds:
        independent_manifest = manifests[("independent", seed)]
        independent_eval_seeds = [
            int(row["scenario_seed"])
            for row in evaluation[("independent", seed)]
            if row["phase"] == "eval_stochastic"
        ]
        for factorization in FACTORIZATIONS[1:]:
            comparator_manifest = manifests[(factorization, seed)]
            require(
                normalized_command([str(value) for value in independent_manifest["command"]])
                == normalized_command([str(value) for value in comparator_manifest["command"]]),
                f"seed {seed}: paired {factorization} command differs beyond output and role factorization",
            )
            comparator_eval_seeds = [
                int(row["scenario_seed"])
                for row in evaluation[(factorization, seed)]
                if row["phase"] == "eval_stochastic"
            ]
            require(
                independent_eval_seeds == comparator_eval_seeds,
                f"seed {seed}: paired {factorization} evaluation seeds differ",
            )
    return manifests, training, evaluation


def t_interval(values: Iterable[float]) -> tuple[float, float, float, float]:
    data = tuple(float(value) for value in values)
    require(len(data) == 3, "The predeclared t interval requires exactly three training seeds.")
    center = mean(data)
    sd = stdev(data)
    half_width = T_CRITICAL_95_DF2 * sd / math.sqrt(3.0)
    return center, sd, center - half_width, center + half_width


def bootstrap_interval(values: Iterable[float], replicates: int, rng: np.random.Generator) -> tuple[float, float]:
    data = np.asarray(tuple(float(value) for value in values), dtype=float)
    require(data.size == 3, "The paired bootstrap requires exactly three training seeds.")
    require(replicates >= 1000, "--bootstrap-replicates must be at least 1000.")
    indices = rng.integers(0, data.size, size=(replicates, data.size))
    samples = data[indices].mean(axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def summarize(
    evaluation: dict[tuple[str, int], list[dict[str, str]]],
    seeds: tuple[int, int, int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    paired_eval: list[dict[str, Any]] = []
    for factorization in FACTORIZATIONS:
        for training_seed in seeds:
            all_rows = evaluation[(factorization, training_seed)]
            rows = [row for row in all_rows if row["phase"] == "eval_stochastic"]
            deterministic_rows = [row for row in all_rows if row["phase"] == "eval_deterministic"]
            summary: dict[str, Any] = {
                "factorization": factorization,
                "training_seed": training_seed,
                "evaluation_scenarios": len(rows),
            }
            for metric in ROW_METRICS:
                summary[f"{metric}_mean"] = mean(float(row[metric]) for row in rows)
            summary["deterministic_discovery_rate_mean"] = mean(
                float(row["discovery_rate"]) for row in deterministic_rows
            )
            active = sum(float(row["active_undiscovered_pair_slots"]) for row in rows)
            aligned = sum(float(row["bilaterally_aligned_pair_slots"]) for row in rows)
            opportunities = sum(float(row["aligned_handshake_opportunities"]) for row in rows)
            successes = sum(float(row["handshake_successes"]) for row in rows)
            summary.update(
                active_undiscovered_pair_slots_sum=active,
                bilaterally_aligned_pair_slots_sum=aligned,
                aligned_handshake_opportunities_sum=opportunities,
                handshake_successes_sum=successes,
                beam_alignment_rate_active_ratio_of_sums=aligned / active if active > 0 else math.nan,
                aligned_role_complementarity_rate_ratio_of_sums=(
                    opportunities / aligned if aligned > 0 else math.nan
                ),
                handshake_success_conversion_rate_ratio_of_sums=(
                    successes / opportunities if opportunities > 0 else math.nan
                ),
            )
            summary["tx_ratio_mean"] = mean(
                float(row["tx_actions"]) / max(1.0, float(row["scan_actions"])) for row in rows
            )
            per_seed.append(summary)

    for training_seed in seeds:
        paired_by_method = {
            factorization: {
                int(row["scenario_seed"]): row
                for row in evaluation[(factorization, training_seed)]
                if row["phase"] == "eval_stochastic"
            }
            for factorization in FACTORIZATIONS
        }
        independent = paired_by_method["independent"]
        for factorization in FACTORIZATIONS[1:]:
            require(
                set(independent) == set(paired_by_method[factorization]),
                f"seed {training_seed}: unpaired {factorization} evaluation rows",
            )
        for scenario_seed in sorted(independent):
            row: dict[str, Any] = {"training_seed": training_seed, "scenario_seed": scenario_seed}
            for metric in ROW_METRICS:
                first = float(independent[scenario_seed][metric])
                row[f"independent_{metric}"] = first
                for factorization in FACTORIZATIONS[1:]:
                    second = float(paired_by_method[factorization][scenario_seed][metric])
                    row[f"{factorization}_{metric}"] = second
                    row[f"delta_{factorization}_{metric}"] = second - first
            paired_eval.append(row)

    rng = np.random.default_rng(bootstrap_seed)
    aggregate: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    by_key = {(str(row["factorization"]), int(row["training_seed"])): row for row in per_seed}
    for factorization in FACTORIZATIONS:
        for metric in (*SEED_METRICS, "tx_ratio"):
            field = f"{metric}_mean" if metric in (*ROW_METRICS, "tx_ratio") else metric
            values = [float(by_key[(factorization, seed)][field]) for seed in seeds]
            center, sd, low, high = t_interval(values)
            bootstrap_low, bootstrap_high = bootstrap_interval(values, bootstrap_replicates, rng)
            aggregate.append(
                {
                    "factorization": factorization,
                    "metric": metric,
                    "training_seeds": 3,
                    "mean": center,
                    "sd_across_training_seeds": sd,
                    "t95_low": low,
                    "t95_high": high,
                    "bootstrap95_low": bootstrap_low,
                    "bootstrap95_high": bootstrap_high,
                }
            )
    for comparator in FACTORIZATIONS[1:]:
        for metric in (*SEED_METRICS, "tx_ratio"):
            field = f"{metric}_mean" if metric in (*ROW_METRICS, "tx_ratio") else metric
            deltas = [
                float(by_key[(comparator, seed)][field])
                - float(by_key[("independent", seed)][field])
                for seed in seeds
            ]
            center, sd, low, high = t_interval(deltas)
            bootstrap_low, bootstrap_high = bootstrap_interval(deltas, bootstrap_replicates, rng)
            contrasts.append(
                {
                    "contrast": f"{comparator}-minus-independent",
                    "metric": metric,
                    "paired_training_seeds": 3,
                    "mean_paired_delta": center,
                    "sd_paired_delta": sd,
                    "paired_t95_low": low,
                    "paired_t95_high": high,
                    "paired_bootstrap95_low": bootstrap_low,
                    "paired_bootstrap95_high": bootstrap_high,
                    "positive_seed_deltas": sum(delta > 0.0 for delta in deltas),
                    "seed_deltas": json.dumps(deltas),
                }
            )
    return per_seed, aggregate, contrasts, paired_eval


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    width = max(1, min(int(window), values.size))
    cumulative = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    result = np.empty(values.size, dtype=float)
    for index in range(values.size):
        start = max(0, index + 1 - width)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result


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
            "figure.figsize": (4.0, 3.0),
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure: plt.Figure, output: Path, stem: str) -> None:
    figure.set_size_inches(4.0, 3.0, forward=True)
    figure.tight_layout(pad=0.7)
    figure.savefig(output / f"{stem}.png")
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)


def plot_training(
    training: dict[tuple[str, int], list[dict[str, str]]],
    seeds: tuple[int, int, int],
    rolling_window: int,
    output: Path,
) -> None:
    configure_plot()
    figure, axis = plt.subplots(figsize=(4.0, 3.0))
    for factorization in FACTORIZATIONS:
        matrices = []
        steps = None
        for seed in seeds:
            rows = training[(factorization, seed)]
            current_steps = np.asarray([int(float(row["training_step"])) for row in rows])
            if steps is None:
                steps = current_steps
            else:
                require(np.array_equal(steps, current_steps), f"{factorization}: training step grids differ")
            values = 100.0 * np.asarray([float(row["discovery_rate"]) for row in rows])
            matrices.append(rolling_mean(values, rolling_window))
        matrix = np.vstack(matrices)
        center = matrix.mean(axis=0)
        half_width = T_CRITICAL_95_DF2 * matrix.std(axis=0, ddof=1) / math.sqrt(3.0)
        axis.plot(steps, center, color=COLORS[factorization], linewidth=1.5, label=LABELS[factorization])
        axis.fill_between(steps, center - half_width, center + half_width, color=COLORS[factorization], alpha=0.14, linewidth=0)
    axis.set_xlabel("Training environment step")
    axis.set_ylabel("Discovery rate (%)")
    axis.set_xlim(left=0)
    axis.set_ylim(-2, 102)
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    axis.legend(frameon=False, loc="best")
    save_figure(figure, output, "training_discovery_curve")


def plot_evaluation(per_seed: list[dict[str, Any]], seeds: tuple[int, int, int], output: Path) -> None:
    configure_plot()
    by_key = {(str(row["factorization"]), int(row["training_seed"])): row for row in per_seed}
    data = {
        factorization: np.asarray(
            [100.0 * float(by_key[(factorization, seed)]["discovery_rate_mean"]) for seed in seeds]
        )
        for factorization in FACTORIZATIONS
    }
    centers = [float(data[factorization].mean()) for factorization in FACTORIZATIONS]
    errors = [
        T_CRITICAL_95_DF2 * float(data[factorization].std(ddof=1)) / math.sqrt(3.0)
        for factorization in FACTORIZATIONS
    ]
    figure, axis = plt.subplots(figsize=(4.0, 3.0))
    x = np.arange(len(FACTORIZATIONS))
    axis.bar(
        x,
        centers,
        yerr=errors,
        color=[COLORS[factorization] for factorization in FACTORIZATIONS],
        width=0.62,
        capsize=3,
        edgecolor="#333333",
        linewidth=0.5,
    )
    offsets = np.asarray((-0.05, 0.0, 0.05))
    for seed_index, seed in enumerate(seeds):
        values = [data[factorization][seed_index] for factorization in FACTORIZATIONS]
        axis.plot(x + offsets[seed_index], values, color="#555555", alpha=0.6, linewidth=0.7, zorder=3)
        axis.scatter(x + offsets[seed_index], values, color="#222222", s=13, zorder=4)
    axis.set_xticks(x, ("Independent", "Conditioned", "Antisymmetric"), rotation=10)
    axis.set_ylabel("Held-out discovery rate (%)")
    axis.set_ylim(bottom=0)
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.6)
    save_figure(figure, output, "evaluation_discovery_summary")


def write_report(
    output: Path,
    manifests: dict[tuple[str, int], dict[str, Any]],
    aggregate: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
) -> None:
    ordinary_discovery = next(
        row for row in contrasts
        if row["contrast"] == "beam_conditioned-minus-independent"
        and row["metric"] == "discovery_rate"
    )
    antisymmetric_discovery = next(
        row for row in contrasts
        if row["contrast"] == "beam_conditioned_antisymmetric-minus-independent"
        and row["metric"] == "discovery_rate"
    )
    ordinary_role_contrast = next(
        row
        for row in contrasts
        if row["contrast"] == "beam_conditioned-minus-independent"
        and row["metric"] == "aligned_role_complementarity_rate_ratio_of_sums"
    )
    antisymmetric_role_contrast = next(
        row
        for row in contrasts
        if row["contrast"] == "beam_conditioned_antisymmetric-minus-independent"
        if row["metric"] == "aligned_role_complementarity_rate_ratio_of_sums"
    )
    episodes = int(next(iter(manifests.values()))["episodes"])
    slots = int(next(iter(manifests.values()))["slots_per_episode"])
    aggregate_discovery = {
        str(row["factorization"]): row
        for row in aggregate
        if row["metric"] == "discovery_rate"
    }
    lines = [
        "# N=2, B=8 no-ISAC RMAPPO role-factorization matrix",
        "",
        "- Design: three paired independent training seeds.",
        f"- Budget per factorization/seed run: {episodes} episodes x {slots} slots = {episodes * slots:,} environment steps.",
        "- Contract: recurrent MAPPO with a learned role, uniformly randomized beam execution, decoupled role tower, and no ISAC actor features.",
        "- Evaluation: stochastic policy on matched held-out scenario seeds; deterministic results are retained as a secondary diagnostic.",
        "- Inference unit: independently trained policy seed (n=3), not evaluation episode.",
        "",
        "| Factorization | Discovery mean | 95% t interval | 95% seed-bootstrap interval |",
        "|---|---:|---:|---:|",
    ]
    for factorization in FACTORIZATIONS:
        row = aggregate_discovery[factorization]
        lines.append(
            f"| {LABELS[factorization]} | {100 * float(row['mean']):.2f}% | "
            f"[{100 * float(row['t95_low']):.2f}, {100 * float(row['t95_high']):.2f}]% | "
            f"[{100 * float(row['bootstrap95_low']):.2f}, {100 * float(row['bootstrap95_high']):.2f}]% |"
        )
    lines.extend(
        [
            "",
            "## Paired contrast",
            "",
            f"Ordinary conditioned minus independent discovery: {100 * float(ordinary_discovery['mean_paired_delta']):.2f} percentage points; "
            f"paired 95% t interval [{100 * float(ordinary_discovery['paired_t95_low']):.2f}, "
            f"{100 * float(ordinary_discovery['paired_t95_high']):.2f}] pp.",
            "",
            f"Antisymmetric conditioned minus independent discovery: "
            f"{100 * float(antisymmetric_discovery['mean_paired_delta']):.2f} percentage points; "
            f"paired 95% t interval [{100 * float(antisymmetric_discovery['paired_t95_low']):.2f}, "
            f"{100 * float(antisymmetric_discovery['paired_t95_high']):.2f}] pp.",
            "",
            f"Aligned-role complementarity deltas are "
            f"{100 * float(ordinary_role_contrast['mean_paired_delta']):.2f} pp for ordinary conditioning and "
            f"{100 * float(antisymmetric_role_contrast['mean_paired_delta']):.2f} pp for antisymmetric conditioning.",
            "",
            "Intervals quantify training-seed variation only. With three seeds they are necessarily wide and should be treated as a gate, not a definitive superiority claim.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    require(args.rolling_window >= 1, "--rolling-window must be positive")
    run_root = args.run_root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else ROOT / "06_analysis" / "tables" / run_root.name
    )
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty analysis directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifests, training, evaluation = load_matrix(run_root, seeds)
    per_seed, aggregate, contrasts, paired_eval = summarize(
        evaluation,
        seeds,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    write_csv(output / "per_training_seed_summary.csv", per_seed)
    write_csv(output / "aggregate_summary.csv", aggregate)
    write_csv(output / "paired_training_seed_contrasts.csv", contrasts)
    write_csv(output / "paired_evaluation_rows.csv", paired_eval)
    plot_training(training, seeds, args.rolling_window, output)
    plot_evaluation(per_seed, seeds, output)
    write_report(output, manifests, aggregate, contrasts)
    contract = {
        "run_root": str(run_root),
        "training_seeds": list(seeds),
        "factorizations": list(FACTORIZATIONS),
        "paired_contract_validated": True,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_seed": int(args.bootstrap_seed),
        "inference_unit": "independent_training_seed",
        "figure_size_inches": [4.0, 3.0],
        "font_family": "Times New Roman",
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
