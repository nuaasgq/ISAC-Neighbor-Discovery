from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "05_simulation" / "results_raw"
OUTPUT = ROOT / "06_analysis" / "paper_tables" / "common_measurement_residual_screen_20260711"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    available = [value for value in values if not math.isnan(value)]
    return sum(available) / len(available) if available else float("nan")


def tx_fraction(rows: Iterable[dict[str, str]]) -> float:
    materialized = list(rows)
    return sum(float(row["tx_actions"]) for row in materialized) / max(
        1.0, sum(float(row["scan_actions"]) for row in materialized)
    )


def sensing_mean(rows: list[dict[str, str]], key: str) -> float:
    if not any(float(row["sensing_observations"]) > 0.0 for row in rows):
        return float("nan")
    return mean(rows, key)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparison_contract(
    label: str,
    metrics_path: Path,
    metrics: list[dict[str, str]],
    capability_level: str,
) -> dict[str, object]:
    with (metrics_path.parent / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    support = manifest.get("stochastic_support") or {}
    scenario_seeds = sorted(int(row["scenario_seed"]) for row in metrics)
    sensing = manifest.get("sensing_measurement") or {}
    protocol = manifest.get("env_protocol", manifest.get("method", ""))
    learned_default = bool(manifest.get("checkpoint")) or manifest.get("scope") == "real_marl_training"
    default_executor = "policy" if learned_default else "protocol"
    default_candidate = "wang_table" if str(protocol).startswith("wang2025") else "none"
    return {
        "method": label,
        "capability_level": capability_level,
        "env_protocol": protocol,
        "candidate_source": manifest.get("candidate_source", default_candidate),
        "beam_executor": manifest.get("beam_executor", default_executor),
        "mode_executor": manifest.get("mode_executor", default_executor),
        "sensing_measurement_mode": (
            "disabled" if capability_level == "blind_no_isac_no_tables" else sensing.get("mode", "none")
        ),
        "eval_scenario_count": len(scenario_seeds),
        "eval_scenario_seed_min": min(scenario_seeds),
        "eval_scenario_seed_max": max(scenario_seeds),
        "slots_per_episode": int(manifest["slots_per_episode"]),
        "training_contract_version": manifest.get("training_contract_version", "not_learned"),
        "role_probability_floor": float(support.get("role_probability_floor", 0.0)),
        "beam_uniform_mixture": float(support.get("beam_uniform_mixture", 0.0)),
        "source_directory": str(metrics_path.parent.relative_to(ROOT)),
    }


def summarize(label: str, rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "method": label,
        "episodes": len(rows),
        "discovery_rate": mean(rows, "discovery_rate"),
        "neighbor_knowledge_recall": mean(rows, "neighbor_knowledge_recall"),
        "empty_scan_ratio": mean(rows, "empty_scan_ratio"),
        "aligned_handshake_opportunities": mean(rows, "aligned_handshake_opportunities"),
        "tx_fraction": tx_fraction(rows),
    }


def paired_comparison(
    label: str,
    first: list[dict[str, str]],
    second: list[dict[str, str]],
    metric: str = "discovery_rate",
) -> dict[str, object]:
    first_by_seed = {int(row["scenario_seed"]): float(row[metric]) for row in first}
    second_by_seed = {int(row["scenario_seed"]): float(row[metric]) for row in second}
    seeds = sorted(set(first_by_seed).intersection(second_by_seed))
    differences = [first_by_seed[seed] - second_by_seed[seed] for seed in seeds]
    count = len(differences)
    difference_mean = sum(differences) / count
    variance = sum((value - difference_mean) ** 2 for value in differences) / max(1, count - 1)
    # t(0.975, 9) for the fixed 10-scenario diagnostic screen.
    half_width = 2.262 * (variance / count) ** 0.5
    observed = abs(difference_mean)
    sign_flip_values = (
        abs(sum(sign * value for sign, value in zip(signs, differences, strict=True)) / count)
        for signs in itertools.product((-1.0, 1.0), repeat=count)
    )
    p_value = sum(value >= observed - 1e-15 for value in sign_flip_values) / (2**count)
    return {
        "comparison": label,
        "metric": metric,
        "paired_scenarios": count,
        "mean_difference": difference_mean,
        "ci95_low": difference_mean - half_width,
        "ci95_high": difference_mean + half_width,
        "exact_sign_flip_p": p_value,
    }


def main() -> None:
    corrected = RAW / "common_measurement_calibration_corrected_20260711"
    unbounded = RAW / "residual_v2_planar_n10_b15_train30_seed29260711"
    supported = RAW / "residual_v2_support_planar_n10_b15_train30_seed29260711"
    mechanism = RAW / "common_measurement_calibration_20260711"

    measurement_rows: list[dict[str, object]] = []
    for mode in ("ideal_count", "noisy_count", "binary_occupancy"):
        for method in ("uniform_random", "wang2025_isac_tables"):
            rows = read_rows(corrected / mode / method / "eval_episode_metrics.csv")
            summary = summarize(f"{method}:{mode}", rows)
            summary.update(
                sensing_count_mae=sensing_mean(rows, "sensing_count_mae"),
                per_target_sensing_recall=sensing_mean(rows, "per_target_sensing_recall"),
            )
            measurement_rows.append(summary)
    write_rows(OUTPUT / "measurement_mode_summary.csv", measurement_rows)

    final_paths = {
        "uniform_random": corrected / "noisy_count" / "uniform_random" / "eval_episode_metrics.csv",
        "wang_noisy_count": corrected / "noisy_count" / "wang2025_isac_tables" / "eval_episode_metrics.csv",
        "residual_random_uniform": unbounded / "diagnostics" / "local_random_uniform_mode" / "eval_episode_metrics.csv",
        "residual_v2_unbounded": unbounded / "eval_episode_metrics.csv",
        "residual_v2_support": supported / "diagnostics" / "full_with_target_status" / "eval_episode_metrics.csv",
        "support_learned_beam_uniform_mode": supported
        / "diagnostics"
        / "learned_beam_uniform_mode"
        / "eval_episode_metrics.csv",
        "support_local_random_learned_mode": supported
        / "diagnostics"
        / "local_random_learned_mode"
        / "eval_episode_metrics.csv",
    }
    final_rows = {label: read_rows(path) for label, path in final_paths.items()}
    write_rows(
        OUTPUT / "final_screen_summary.csv",
        [summarize(label, rows) for label, rows in final_rows.items()],
    )
    capability_levels = {
        "uniform_random": "blind_no_isac_no_tables",
        "wang_noisy_count": "common_phy_wang_table_policy",
        "residual_random_uniform": "residual_common_mechanism",
        "residual_v2_unbounded": "residual_common_mechanism",
        "residual_v2_support": "residual_common_mechanism",
        "support_learned_beam_uniform_mode": "residual_common_mechanism",
        "support_local_random_learned_mode": "residual_common_mechanism",
    }
    write_rows(
        OUTPUT / "final_screen_contracts.csv",
        [
            comparison_contract(label, final_paths[label], rows, capability_levels[label])
            for label, rows in final_rows.items()
        ],
    )

    default_rows = read_rows(mechanism / "default_rule_uniform_mode" / "eval_episode_metrics.csv")
    residual_rows = read_rows(mechanism / "residual_rule_uniform_mode" / "eval_episode_metrics.csv")
    comparisons = [
        paired_comparison("residual_random - default_random", residual_rows, default_rows),
        paired_comparison(
            "residual_support - wang_noisy_count",
            final_rows["residual_v2_support"],
            final_rows["wang_noisy_count"],
        ),
        paired_comparison(
            "residual_support - residual_random_uniform",
            final_rows["residual_v2_support"],
            final_rows["residual_random_uniform"],
        ),
        paired_comparison(
            "support_learned_beam - residual_random_uniform",
            final_rows["support_learned_beam_uniform_mode"],
            final_rows["residual_random_uniform"],
        ),
        paired_comparison(
            "support_learned_mode - residual_random_uniform",
            final_rows["support_local_random_learned_mode"],
            final_rows["residual_random_uniform"],
        ),
    ]
    write_rows(OUTPUT / "paired_comparisons.csv", comparisons)


if __name__ == "__main__":
    main()
