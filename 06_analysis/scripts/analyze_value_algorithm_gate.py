from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "05_simulation" / "results_raw"
VALUE_ROOT = RAW / "value_algorithm_gate_v2_n10_b15_30ep_seed29260711"
OUTPUT = ROOT / "06_analysis" / "paper_tables" / "value_algorithm_gate_20260711"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows: Iterable[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values)


def tx_fraction(rows: list[dict[str, str]]) -> float:
    return sum(float(row["tx_actions"]) for row in rows) / sum(float(row["scan_actions"]) for row in rows)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(method: str, variant: str, rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "method": method,
        "policy_variant": variant,
        "episodes": len(rows),
        "discovery_rate": mean(rows, "discovery_rate"),
        "neighbor_knowledge_recall": mean(rows, "neighbor_knowledge_recall"),
        "mean_delay_censored": mean(rows, "mean_delay_censored"),
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
    critical = 2.262 if count == 10 else 1.96
    half_width = critical * (variance / count) ** 0.5
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
    value_paths = {
        "idqn": VALUE_ROOT / "idqn_strict_team",
        "idqn_local_reward": VALUE_ROOT / "idqn_strict_local",
        "shared_idqn": VALUE_ROOT / "shared_idqn",
        "vdn": VALUE_ROOT / "vdn",
        "qmix": VALUE_ROOT / "qmix",
    }
    value_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    summaries: list[dict[str, object]] = []
    training_summaries: list[dict[str, object]] = []
    contracts: list[dict[str, object]] = []
    for method, path in value_paths.items():
        evaluation = read_rows(path / "eval_episode_metrics.csv")
        variants = sorted({row["policy_variant"] for row in evaluation})
        value_rows[method] = {}
        for variant in variants:
            selected = [row for row in evaluation if row["policy_variant"] == variant]
            value_rows[method][variant] = selected
            summaries.append(summarize(method, variant, selected))

        training = read_rows(path / "episode_metrics.csv")
        with (path / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        training_summaries.append(
            {
                "method": method,
                "training_episodes": len(training),
                "first_10_discovery_rate": mean(training[:10], "discovery_rate"),
                "last_10_discovery_rate": mean(training[-10:], "discovery_rate"),
                "first_10_return_mean_per_agent": mean(training[:10], "episode_return_mean_per_agent"),
                "last_10_return_mean_per_agent": mean(training[-10:], "episode_return_mean_per_agent"),
                "last_10_td_loss": mean(training[-10:], "td_loss"),
            }
        )
        contracts.append(
            {
                "method": method,
                "training_contract_version": manifest["training_contract_version"],
                "reward_scope": manifest["reward_scope"],
                "centralized_training": manifest["centralized_training"],
                "independent_agent_parameters": manifest["independent_agent_parameters"],
                "independent_agent_replay": manifest["independent_agent_replay"],
                "parameter_sharing": manifest["parameter_sharing"],
                "training_episodes": manifest["episodes"],
                "training_seed": manifest["seed"],
                "slots_per_episode": manifest["slots_per_episode"],
                "eval_episodes_per_variant": manifest["eval_episodes"],
                "peak_rss_mb": manifest["peak_rss_mb"],
            }
        )

    mappo_path = (
        RAW
        / "residual_v2_support_planar_n10_b15_train30_seed29260711"
        / "diagnostics"
        / "full_with_target_status"
        / "eval_episode_metrics.csv"
    )
    mappo_rows = read_rows(mappo_path)
    summaries.append(summarize("mappo_residual_v2_support", "matched_support", mappo_rows))

    comparisons: list[dict[str, object]] = []
    for method, variants in value_rows.items():
        random_rows = variants["random_uniform"]
        for variant in (
            "matched_support",
            "greedy",
            "learned_role_random_beam",
            "random_role_learned_beam",
        ):
            comparisons.append(
                paired_comparison(
                    f"{method}:{variant} - {method}:random_uniform",
                    variants[variant],
                    random_rows,
                )
            )
        comparisons.append(
            paired_comparison(
                f"{method}:matched_support - MAPPO:matched_support",
                variants["matched_support"],
                mappo_rows,
            )
        )

    comparisons.extend(
        [
            paired_comparison(
                "idqn:matched_support - shared_idqn:matched_support",
                value_rows["idqn"]["matched_support"],
                value_rows["shared_idqn"]["matched_support"],
            ),
            paired_comparison(
                "idqn_local_reward:matched_support - idqn:matched_support",
                value_rows["idqn_local_reward"]["matched_support"],
                value_rows["idqn"]["matched_support"],
            ),
            paired_comparison(
                "vdn:matched_support - shared_idqn:matched_support",
                value_rows["vdn"]["matched_support"],
                value_rows["shared_idqn"]["matched_support"],
            ),
            paired_comparison(
                "qmix:matched_support - shared_idqn:matched_support",
                value_rows["qmix"]["matched_support"],
                value_rows["shared_idqn"]["matched_support"],
            ),
        ]
    )

    write_rows(OUTPUT / "evaluation_summary.csv", summaries)
    write_rows(OUTPUT / "training_summary.csv", training_summaries)
    write_rows(OUTPUT / "paired_comparisons.csv", comparisons)
    write_rows(OUTPUT / "run_contracts.csv", contracts)


if __name__ == "__main__":
    main()
