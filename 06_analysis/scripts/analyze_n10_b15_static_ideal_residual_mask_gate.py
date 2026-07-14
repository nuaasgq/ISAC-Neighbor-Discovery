from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_n10_b15_static_ideal_paired_eval as paired_analysis  # noqa: E402


DEFAULT_FORMAL_ROOT = (
    ROOT / "05_simulation" / "results_raw" / "n10_b15_static_ideal_paired_eval_3seed"
)
DEFAULT_GATE_ROOT = (
    ROOT / "05_simulation" / "results_raw" / "n10_b15_static_ideal_residual_mask_gate_eval_formal"
)
DEFAULT_OUTPUT = ROOT / "06_analysis" / "n10_b15_static_ideal_residual_mask_gate_20260714"
DEFAULT_TRAIN_SEED = 59262731
RESIDUAL_METHOD = "mappo_residual_mask_isac"
COMPARATORS = (
    "mappo_direct_isac",
    "isac_candidate_pool_random",
    "wang2025_isac_tables",
)
METHODS = (*COMPARATORS, RESIDUAL_METHOD)
LABELS = {
    "mappo_direct_isac": "Direct-ISAC MAPPO",
    "isac_candidate_pool_random": "ISAC candidate random",
    "wang2025_isac_tables": "Wang2025",
    RESIDUAL_METHOD: "Residual-mask MAPPO",
}
COLORS = {
    "mappo_direct_isac": "#0072B2",
    "isac_candidate_pool_random": "#E69F00",
    "wang2025_isac_tables": "#D55E00",
    RESIDUAL_METHOD: "#009E73",
}
METRICS = (
    "discovery_rate",
    "mean_delay_censored",
    "discovery_curve_auc_normalized",
    "discovery_rate_at_50_slots",
    "discovery_rate_at_100_slots",
    "discovery_rate_at_150_slots",
    "discovery_rate_at_200_slots",
    "discovery_rate_at_300_slots",
    "networking_completion_slot_censored",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the single-seed residual-mask MAPPO gate.")
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--gate-root", type=Path, default=DEFAULT_GATE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--expected-episodes", type=int, default=50)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def validate_pairing(
    rows: list[dict[str, Any]],
    methods: tuple[str, ...] = METHODS,
    expected_episodes: int = 50,
) -> tuple[int, ...]:
    scenario_sets: dict[str, tuple[int, ...]] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != expected_episodes:
            raise ValueError(f"{method}: expected {expected_episodes} episodes, found {len(selected)}")
        episode_ids = {int(row["eval_episode"]) for row in selected}
        if episode_ids != set(range(expected_episodes)):
            raise ValueError(f"{method}: incomplete or duplicate eval_episode values")
        scenario_sets[method] = tuple(sorted(int(row["scenario_seed"]) for row in selected))
    reference = scenario_sets[methods[0]]
    for method in methods[1:]:
        if scenario_sets[method] != reference:
            raise ValueError(f"{method}: scenario seeds do not match the paired reference")
    return reference


def load_gate_rows(formal_root: Path, gate_root: Path, train_seed: int) -> list[dict[str, Any]]:
    gate_rows = [
        row
        for row in paired_analysis.load_evaluation_rows(gate_root)
        if row["method"] == RESIDUAL_METHOD and int(row["train_seed"]) == int(train_seed)
    ]
    gate_scenarios = {int(row["scenario_seed"]) for row in gate_rows}
    formal_rows = [
        row
        for row in paired_analysis.load_evaluation_rows(formal_root)
        if row["method"] in COMPARATORS
        and int(row["train_seed"]) == int(train_seed)
        and int(row["scenario_seed"]) in gate_scenarios
    ]
    return formal_rows + gate_rows


def load_gate_timeline(formal_root: Path, gate_root: Path, train_seed: int) -> list[dict[str, Any]]:
    gate = [
        row
        for row in paired_analysis.load_timeline_rows(gate_root)
        if row["method"] == RESIDUAL_METHOD and int(row["train_seed"]) == int(train_seed)
    ]
    gate_scenarios = {int(row["scenario_seed"]) for row in gate}
    formal = [
        row
        for row in paired_analysis.load_timeline_rows(formal_root)
        if row["method"] in COMPARATORS
        and int(row["train_seed"]) == int(train_seed)
        and int(row["scenario_seed"]) in gate_scenarios
    ]
    return formal + gate


def method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        result: dict[str, Any] = {"method": method, "label": LABELS[method], "episodes": len(selected)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_episode_sd"] = float(values.std(ddof=1))
        output.append(result)
    return output


def paired_deltas(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260714)
    by_method = {
        method: {int(row["scenario_seed"]): row for row in rows if row["method"] == method}
        for method in METHODS
    }
    seeds = sorted(by_method[RESIDUAL_METHOD])
    output: list[dict[str, Any]] = []
    for comparator in COMPARATORS:
        for metric in METRICS:
            differences = np.asarray(
                [
                    float(by_method[RESIDUAL_METHOD][seed][metric])
                    - float(by_method[comparator][seed][metric])
                    for seed in seeds
                ],
                dtype=float,
            )
            indices = rng.integers(0, len(differences), size=(bootstrap_samples, len(differences)))
            bootstrap = differences[indices].mean(axis=1)
            output.append(
                {
                    "reference": RESIDUAL_METHOD,
                    "comparator": comparator,
                    "metric": metric,
                    "paired_scenarios": len(differences),
                    "reference_minus_comparator": float(differences.mean()),
                    "paired_scenario_bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "paired_scenario_bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(curves: list[dict[str, Any]], output: Path) -> None:
    plt = paired_analysis.configure_plotting()
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for method in METHODS:
        selected = [row for row in curves if row["method"] == method]
        slots = np.asarray([row["slot"] for row in selected], dtype=int)
        mean = 100.0 * np.asarray([row["mean_discovery_rate"] for row in selected], dtype=float)
        low = 100.0 * np.asarray([row["ci95_low"] for row in selected], dtype=float)
        high = 100.0 * np.asarray([row["ci95_high"] for row in selected], dtype=float)
        ax.plot(slots, mean, color=COLORS[method], linewidth=2.0, label=LABELS[method])
        ax.fill_between(slots, low, high, color=COLORS[method], alpha=0.10)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Discovered neighbor links (%)")
    ax.set_xlim(1, 300)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(frameon=False)
    fig.savefig(output / "residual_mask_gate_discovery_vs_slot.png")
    fig.savefig(output / "residual_mask_gate_discovery_vs_slot.pdf")
    plt.close(fig)


def write_report(
    output: Path,
    summary: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    scenarios: tuple[int, ...],
    train_seed: int,
) -> None:
    lookup = {row["method"]: row for row in summary}
    delta_lookup = {(row["comparator"], row["metric"]): row for row in deltas}
    residual = lookup[RESIDUAL_METHOD]
    candidate = lookup["isac_candidate_pool_random"]
    direct = lookup["mappo_direct_isac"]
    candidate_auc = delta_lookup[("isac_candidate_pool_random", "discovery_curve_auc_normalized")]
    direct_auc = delta_lookup[("mappo_direct_isac", "discovery_curve_auc_normalized")]
    final_gap = float(residual["discovery_rate_mean"] - candidate["discovery_rate_mean"])
    promoted = (
        float(candidate_auc["reference_minus_comparator"]) > 0.0
        and float(direct_auc["reference_minus_comparator"]) > 0.0
        and final_gap >= -0.02
    )
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {date.today().isoformat()}",
        "- Verification Status: ANALYZED",
        "- Version Label: n10_b15_static_ideal_residual_mask_gate_v1",
        "",
        "## Single-Seed Gate Report",
        "",
        f"- Training seed: {train_seed}",
        f"- Paired scenarios: {scenarios[0]}--{scenarios[-1]} ({len(scenarios)})",
        "- Statistical unit: paired held-out scenario within one trained seed",
        f"- Promotion decision: {'PASS' if promoted else 'FAIL'}",
        "",
        "| Method | Final discovery | 50-slot | 100-slot | Mean delay | Curve AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = lookup[method]
        lines.append(
            f"| {LABELS[method]} | {100.0 * row['discovery_rate_mean']:.2f}% | "
            f"{100.0 * row['discovery_rate_at_50_slots_mean']:.2f}% | "
            f"{100.0 * row['discovery_rate_at_100_slots_mean']:.2f}% | "
            f"{row['mean_delay_censored_mean']:.2f} | "
            f"{row['discovery_curve_auc_normalized_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Attribution Boundary",
            "",
            "The residual candidate-random comparator uses the same local residual-table feasible set. "
            "Residual-mask MAPPO can claim an RL increment only from differences relative to that comparator; "
            "its absolute final discovery rate is not an RL contribution.",
            "",
            f"Residual-mask minus candidate-random AUC: "
            f"{candidate_auc['reference_minus_comparator']:.3f} "
            f"[{candidate_auc['paired_scenario_bootstrap_ci95_low']:.3f}, "
            f"{candidate_auc['paired_scenario_bootstrap_ci95_high']:.3f}].",
            f"Residual-mask minus Direct-ISAC AUC: {direct_auc['reference_minus_comparator']:.3f} "
            f"[{direct_auc['paired_scenario_bootstrap_ci95_low']:.3f}, "
            f"{direct_auc['paired_scenario_bootstrap_ci95_high']:.3f}].",
            f"Residual-mask final coverage versus candidate random: {100.0 * final_gap:+.2f} pp.",
            "",
            "### Statistical Boundary",
            "",
            "Paired-scenario bootstrap intervals condition on one trained policy seed. They characterize "
            "scenario variation only and cannot establish training-seed robustness or publication-level significance.",
            "The pilot is promoted only to decide whether a 1,000-episode, three-seed run is worth the compute.",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if int(args.bootstrap_samples) < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_gate_rows(args.formal_root.resolve(), args.gate_root.resolve(), int(args.train_seed))
    scenarios = validate_pairing(rows, expected_episodes=int(args.expected_episodes))
    timeline = load_gate_timeline(
        args.formal_root.resolve(), args.gate_root.resolve(), int(args.train_seed)
    )
    expected_timeline_rows = len(METHODS) * int(args.expected_episodes) * 45
    if len(timeline) != expected_timeline_rows:
        raise ValueError(
            f"Expected {expected_timeline_rows} edge timeline rows, found {len(timeline)}"
        )
    summary = method_summary(rows)
    deltas = paired_deltas(rows, int(args.bootstrap_samples))
    curves = paired_analysis.curve_rows(timeline)
    write_csv(output / "method_summary.csv", summary)
    write_csv(output / "paired_scenario_deltas.csv", deltas)
    write_csv(output / "discovery_curve.csv", curves)
    plot_curves(curves, output)
    write_report(output, summary, deltas, scenarios, int(args.train_seed))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
