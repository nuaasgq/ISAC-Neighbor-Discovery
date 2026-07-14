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


DEFAULT_EVAL_ROOT = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "n10_b15_static_ideal_single_mixture_formal_eval_3seed"
)
DEFAULT_TRAIN_ROOT = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "n10_b15_static_ideal_single_mixture_formal_3seed"
)
DEFAULT_BASELINE_ROOT = (
    ROOT / "05_simulation" / "results_raw" / "n10_b15_static_ideal_paired_eval_3seed"
)
DEFAULT_OUTPUT = ROOT / "06_analysis" / "n10_b15_static_ideal_single_mixture_formal"
METHODS = (
    "wang2025_isac_tables",
    "isac_candidate_pool_random",
    "mappo_direct_isac",
    "mappo_residual_mask_isac",
)
TRAINED_METHODS = ("mappo_direct_isac", "mappo_residual_mask_isac")
REFERENCE = "mappo_residual_mask_isac"
LABELS = {
    "wang2025_isac_tables": "Wang2025",
    "isac_candidate_pool_random": "ISAC candidate random",
    "mappo_direct_isac": "Direct-ISAC MAPPO",
    "mappo_residual_mask_isac": "Residual-mask MAPPO",
}
COLORS = {
    "wang2025_isac_tables": "#D55E00",
    "isac_candidate_pool_random": "#E69F00",
    "mappo_direct_isac": "#0072B2",
    "mappo_residual_mask_isac": "#009E73",
}
SUMMARY_METRICS = (
    "discovery_rate",
    "mean_delay_censored",
    "discovery_curve_auc_normalized",
    "discovery_rate_at_50_slots",
    "discovery_rate_at_100_slots",
    "discovery_rate_at_150_slots",
    "discovery_rate_at_200_slots",
    "networking_completion_slot_censored",
)
MECHANISM_METRICS = (
    "candidate_count_mean",
    "positive_beam_recall",
    "empty_opportunity_exclusion_rate",
    "selected_candidate_compliance",
    "selected_undiscovered_beam_rate",
    "tx_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the corrected single-mixture N=10 campaign.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_evaluation_rows(eval_root: Path, baseline_root: Path) -> list[dict[str, Any]]:
    corrected = [
        row
        for row in paired_analysis.load_evaluation_rows(eval_root)
        if row["method"] in TRAINED_METHODS
    ]
    baselines = [
        row
        for row in paired_analysis.load_evaluation_rows(baseline_root)
        if row["method"] in METHODS[:2]
    ]
    return baselines + corrected


def selected_timeline_rows(eval_root: Path, baseline_root: Path) -> list[dict[str, Any]]:
    corrected = [
        row
        for row in paired_analysis.load_timeline_rows(eval_root)
        if row["method"] in TRAINED_METHODS
    ]
    baselines = [
        row
        for row in paired_analysis.load_timeline_rows(baseline_root)
        if row["method"] in METHODS[:2]
    ]
    return baselines + corrected


def validate_pairing(rows: list[dict[str, Any]], expected_per_method: int = 150) -> set[tuple[int, int]]:
    key_sets: dict[str, set[tuple[int, int]]] = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != expected_per_method:
            raise ValueError(f"{method}: expected {expected_per_method} episodes, found {len(selected)}")
        keys = {(int(row["train_seed"]), int(row["scenario_seed"])) for row in selected}
        if len(keys) != expected_per_method:
            raise ValueError(f"{method}: duplicate paired scenario keys")
        key_sets[method] = keys
    reference = key_sets[METHODS[0]]
    for method in METHODS[1:]:
        if key_sets[method] != reference:
            raise ValueError(f"{method}: paired scenarios differ from the baseline reference")
    return reference


def seed_level_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        seeds = sorted({int(row["train_seed"]) for row in method_rows})
        record: dict[str, Any] = {
            "method": method,
            "label": LABELS[method],
            "seed_count": len(seeds),
            "episode_count": len(method_rows),
        }
        for metric in SUMMARY_METRICS:
            seed_means = np.asarray(
                [
                    np.mean(
                        [float(row[metric]) for row in method_rows if int(row["train_seed"]) == seed]
                    )
                    for seed in seeds
                ],
                dtype=float,
            )
            record[f"{metric}_mean"] = float(seed_means.mean())
            record[f"{metric}_seed_sd"] = float(seed_means.std(ddof=1))
        output.append(record)
    return output


def paired_delta_rows(rows: list[dict[str, Any]], samples: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260714)
    for metric in SUMMARY_METRICS:
        reference = {
            (int(row["train_seed"]), int(row["scenario_seed"])): float(row[metric])
            for row in rows
            if row["method"] == REFERENCE
        }
        for comparator in METHODS[:-1]:
            comparison = {
                (int(row["train_seed"]), int(row["scenario_seed"])): float(row[metric])
                for row in rows
                if row["method"] == comparator
            }
            point, low, high = paired_analysis.hierarchical_paired_bootstrap(
                reference, comparison, samples, rng
            )
            output.append(
                {
                    "reference": REFERENCE,
                    "comparator": comparator,
                    "metric": metric,
                    "paired_n": len(reference),
                    "reference_minus_comparator": point,
                    "hierarchical_bootstrap_ci95_low": low,
                    "hierarchical_bootstrap_ci95_high": high,
                }
            )
    return output


def load_training_rows(train_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in TRAINED_METHODS:
        for path in sorted((train_root / method).glob("seed_*/episode_metrics.csv")):
            train_seed = int(path.parent.name.removeprefix("seed_"))
            for item in read_csv(path):
                episode = int(item["episode"])
                rows.append(
                    {
                        "method": method,
                        "train_seed": train_seed,
                        "episode": episode,
                        "environment_step": (episode + 1) * 300,
                        "episode_return_sum": float(item["episode_return_sum"]),
                        "discovery_rate": float(item["discovery_rate"]),
                    }
                )
    return rows


def training_curve_rows(rows: list[dict[str, Any]], window: int = 50) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in TRAINED_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        seeds = sorted({int(row["train_seed"]) for row in method_rows})
        by_seed = {
            seed: sorted(
                [row for row in method_rows if int(row["train_seed"]) == seed],
                key=lambda row: int(row["episode"]),
            )
            for seed in seeds
        }
        common_episodes = sorted(set.intersection(*(set(int(row["episode"]) for row in items) for items in by_seed.values())))
        for episode in common_episodes:
            start = max(0, episode - window + 1)
            returns = []
            discoveries = []
            for seed in seeds:
                window_rows = [
                    row for row in by_seed[seed] if start <= int(row["episode"]) <= episode
                ]
                returns.append(float(np.mean([row["episode_return_sum"] for row in window_rows])))
                discoveries.append(float(np.mean([row["discovery_rate"] for row in window_rows])))
            output.append(
                {
                    "method": method,
                    "episode": episode,
                    "environment_step": (episode + 1) * 300,
                    "rolling_window_episodes": window,
                    "return_seed_mean": float(np.mean(returns)),
                    "return_seed_min": float(np.min(returns)),
                    "return_seed_max": float(np.max(returns)),
                    "discovery_seed_mean": float(np.mean(discoveries)),
                    "discovery_seed_min": float(np.min(discoveries)),
                    "discovery_seed_max": float(np.max(discoveries)),
                }
            )
    return output


def load_candidate_rows(eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method = REFERENCE
    for path in sorted((eval_root / method).glob("seed_*/candidate_pool_timeline.csv")):
        train_seed = int(path.parent.name.removeprefix("seed_"))
        for item in read_csv(path):
            row: dict[str, Any] = {
                "method": method,
                "train_seed": train_seed,
                "scenario_seed": int(item["scenario_seed"]),
                "eval_episode": int(item["eval_episode"]),
                "slot": int(item["elapsed_slots"]),
            }
            for metric in MECHANISM_METRICS:
                row[metric] = float(item[metric])
            rows.append(row)
    return rows


def mechanism_curve_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for slot in range(1, 301):
        selected = [row for row in rows if int(row["slot"]) == slot]
        record: dict[str, Any] = {"slot": slot, "samples": len(selected)}
        for metric in MECHANISM_METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
        output.append(record)
    return output


def plot_discovery(curves: list[dict[str, Any]], output: Path) -> None:
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
    ax.set(xlabel="Slot", ylabel="Discovered neighbor links (%)", xlim=(1, 300), ylim=(0, 100))
    ax.legend(frameon=False)
    fig.savefig(output / "discovery_vs_slot.png")
    fig.savefig(output / "discovery_vs_slot.pdf")
    plt.close(fig)


def plot_training(curves: list[dict[str, Any]], output: Path) -> None:
    plt = paired_analysis.configure_plotting()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    for method in TRAINED_METHODS:
        selected = [row for row in curves if row["method"] == method]
        steps = np.asarray([row["environment_step"] for row in selected], dtype=int)
        color = COLORS[method]
        for ax, metric, low_key, high_key in (
            (axes[0], "return_seed_mean", "return_seed_min", "return_seed_max"),
            (axes[1], "discovery_seed_mean", "discovery_seed_min", "discovery_seed_max"),
        ):
            mean = np.asarray([row[metric] for row in selected], dtype=float)
            low = np.asarray([row[low_key] for row in selected], dtype=float)
            high = np.asarray([row[high_key] for row in selected], dtype=float)
            if metric == "discovery_seed_mean":
                mean, low, high = 100.0 * mean, 100.0 * low, 100.0 * high
            ax.plot(steps, mean, color=color, linewidth=1.8, label=LABELS[method])
            ax.fill_between(steps, low, high, color=color, alpha=0.10)
    axes[0].set_ylabel("Episode return\n(50-episode mean)")
    axes[1].set_ylabel("Training discovery (%)\n(50-episode mean)")
    axes[1].set_xlabel("Environment step")
    axes[0].legend(frameon=False)
    fig.savefig(output / "training_convergence_vs_environment_step.png")
    fig.savefig(output / "training_convergence_vs_environment_step.pdf")
    plt.close(fig)


def plot_mechanism(curves: list[dict[str, Any]], output: Path) -> None:
    plt = paired_analysis.configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, constrained_layout=True)
    slots = np.asarray([row["slot"] for row in curves], dtype=int)
    panels = (
        ("candidate_count_mean_mean", "Candidate beams per UAV", 1.0),
        ("positive_beam_recall_mean", "Undiscovered-beam recall (%)", 100.0),
        ("empty_opportunity_exclusion_rate_mean", "Empty-beam exclusion (%)", 100.0),
        ("selected_undiscovered_beam_rate_mean", "Selected useful direction (%)", 100.0),
    )
    for ax, (metric, ylabel, scale) in zip(axes.flat, panels, strict=True):
        values = scale * np.asarray([row[metric] for row in curves], dtype=float)
        ax.plot(slots, values, color=COLORS[REFERENCE], linewidth=1.8)
        ax.set_ylabel(ylabel)
    axes[1, 0].set_xlabel("Slot")
    axes[1, 1].set_xlabel("Slot")
    fig.savefig(output / "residual_candidate_mechanism_vs_slot.png")
    fig.savefig(output / "residual_candidate_mechanism_vs_slot.pdf")
    plt.close(fig)


def write_report(
    output: Path,
    summaries: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> None:
    summary = {row["method"]: row for row in summaries}
    delta = {(row["comparator"], row["metric"]): row for row in deltas}
    auc_direct = delta[("mappo_direct_isac", "discovery_curve_auc_normalized")]
    auc_random = delta[("isac_candidate_pool_random", "discovery_curve_auc_normalized")]
    final_gap = float(
        summary[REFERENCE]["discovery_rate_mean"]
        - summary["isac_candidate_pool_random"]["discovery_rate_mean"]
    )
    compliance = min(float(row["selected_candidate_compliance"]) for row in candidate_rows)
    promoted = (
        float(auc_direct["reference_minus_comparator"]) > 0.0
        and float(auc_random["reference_minus_comparator"]) > 0.0
        and final_gap >= -0.02
        and compliance == 1.0
    )
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {date.today().isoformat()}",
        "- Verification Status: ANALYZED",
        "- Version Label: n10_b15_static_ideal_single_mixture_v1",
        "- Statistical unit: trained seed with paired held-out scenarios",
        "",
        "## Formal Result",
        "",
        f"Promotion decision: {'PASS' if promoted else 'FAIL'}",
        "",
        "| Method | Final discovery | 50-slot | 100-slot | Delay | Curve AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summary[method]
        lines.append(
            f"| {LABELS[method]} | {100 * row['discovery_rate_mean']:.2f}% | "
            f"{100 * row['discovery_rate_at_50_slots_mean']:.2f}% | "
            f"{100 * row['discovery_rate_at_100_slots_mean']:.2f}% | "
            f"{row['mean_delay_censored_mean']:.2f} | "
            f"{row['discovery_curve_auc_normalized_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Attribution Boundary",
            "",
            "Residual-mask versus Direct-ISAC isolates the local residual candidate-support mechanism "
            "under the same corrected MAPPO distribution. Residual-mask versus candidate-random isolates "
            "the learned policy increment within the same class of local residual candidate support.",
            f"Residual minus Direct AUC: {auc_direct['reference_minus_comparator']:.3f} "
            f"[{auc_direct['hierarchical_bootstrap_ci95_low']:.3f}, "
            f"{auc_direct['hierarchical_bootstrap_ci95_high']:.3f}].",
            f"Residual minus candidate-random AUC: {auc_random['reference_minus_comparator']:.3f} "
            f"[{auc_random['hierarchical_bootstrap_ci95_low']:.3f}, "
            f"{auc_random['hierarchical_bootstrap_ci95_high']:.3f}].",
            f"Residual final-discovery gap versus candidate-random: {100 * final_gap:+.2f} pp.",
            f"Minimum evaluated candidate-mask compliance: {100 * compliance:.2f}%.",
            "",
            "## Eleven-Fallacy Scan",
            "",
            "| Check | Status | Boundary |",
            "|---|---|---|",
            "| Causal attribution | Controlled | Direct and Residual differ only in local candidate support. |",
            "| Metric substitution | Controlled | Final discovery and curve AUC are both reported. |",
            "| Baseline fairness | Controlled | All methods use identical paired scenarios and horizons. |",
            "| Information leakage | Controlled | Truth is used only by offline diagnostics. |",
            "| Pseudoreplication | Controlled | Hierarchical bootstrap resamples seeds then scenarios. |",
            "| Seed robustness | Limited | Three training seeds; no stronger population claim is made. |",
            "| Censoring bias | Controlled | Delay and completion metrics remain horizon-censored. |",
            "| Cherry picking | Controlled | Promotion criteria were fixed before formal completion. |",
            "| Multiple comparisons | Descriptive | Intervals support mechanism analysis, not confirmatory multiplicity claims. |",
            "| Extrapolation | Restricted | Result applies to static N=10 ideal-ISAC only. |",
            "| Implementation fidelity | Audited | Double beam-mixture defect was fixed and legacy runs excluded. |",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if int(args.bootstrap_samples) < 100:
        raise ValueError("--bootstrap-samples must be at least 100")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation_rows = selected_evaluation_rows(args.eval_root.resolve(), args.baseline_root.resolve())
    if not args.allow_incomplete:
        validate_pairing(evaluation_rows)
    timeline_rows = selected_timeline_rows(args.eval_root.resolve(), args.baseline_root.resolve())
    if not args.allow_incomplete and len(timeline_rows) != len(METHODS) * 150 * 45:
        raise ValueError("Edge timeline is incomplete.")
    training_rows = load_training_rows(args.train_root.resolve())
    candidate_rows = load_candidate_rows(args.eval_root.resolve())
    if not args.allow_incomplete and len(candidate_rows) != 3 * 50 * 300:
        raise ValueError("Candidate-pool timeline is incomplete.")

    summaries = seed_level_summary(evaluation_rows)
    deltas = paired_delta_rows(evaluation_rows, int(args.bootstrap_samples))
    discovery_curves = paired_analysis.curve_rows(timeline_rows)
    training_curves = training_curve_rows(training_rows)
    mechanism_curves = mechanism_curve_rows(candidate_rows)
    write_csv(output / "method_summary.csv", summaries)
    write_csv(output / "hierarchical_paired_deltas.csv", deltas)
    write_csv(output / "discovery_curve.csv", discovery_curves)
    write_csv(output / "training_curve.csv", training_curves)
    write_csv(output / "candidate_mechanism_curve.csv", mechanism_curves)
    plot_discovery(discovery_curves, output)
    plot_training(training_curves, output)
    plot_mechanism(mechanism_curves, output)
    write_report(output, summaries, deltas, candidate_rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
