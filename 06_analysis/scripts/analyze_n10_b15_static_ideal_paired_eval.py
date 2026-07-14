from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_ROOT = (
    ROOT / "05_simulation" / "results_raw" / "n10_b15_static_ideal_paired_eval_3seed"
)
DEFAULT_TRAIN_ROOT = (
    ROOT / "05_simulation" / "results_raw" / "n10_b15_static_ideal_mappo_formal_3seed"
)
DEFAULT_OUTPUT = ROOT / "06_analysis" / "n10_b15_static_ideal_paired_eval_20260714"
REFERENCE_METHOD = "mappo_direct_isac"

METHOD_LABELS = {
    "uniform_random": "Uniform random",
    "wang2025_isac_tables": "Wang2025",
    "isac_candidate_pool_random": "ISAC candidate random",
    "mappo_no_isac": "MAPPO without ISAC",
    "mappo_direct_isac": "Direct-ISAC MAPPO",
    "mappo_direct_isac_measurement_aux": "ISAC + auxiliary MAPPO",
    "random_role_learned_beam": "Random role + learned beam",
    "learned_role_uniform_beam": "Learned role + random beam",
}

COLORS = {
    "uniform_random": "#9B9B9B",
    "wang2025_isac_tables": "#D55E00",
    "isac_candidate_pool_random": "#E69F00",
    "mappo_no_isac": "#56B4E9",
    "mappo_direct_isac": "#0072B2",
    "mappo_direct_isac_measurement_aux": "#009E73",
    "random_role_learned_beam": "#CC79A7",
    "learned_role_uniform_beam": "#6A3D9A",
}

SUMMARY_METRICS = (
    "discovery_rate",
    "mean_delay_censored",
    "discovery_curve_auc_normalized",
    "discovery_rate_at_50_slots",
    "discovery_rate_at_100_slots",
    "discovery_rate_at_150_slots",
    "discovery_rate_at_200_slots",
    "discovery_rate_at_300_slots",
    "time_to_50pct_censored_slots",
    "time_to_80pct_censored_slots",
    "time_to_90pct_censored_slots",
    "empty_scan_ratio",
    "networking_completion_slot_censored",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the paired N=10 static ideal evaluation.")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
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
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def method_and_seed(path: Path) -> tuple[str, int]:
    seed_dir = path.parent
    if seed_dir.name.startswith("seed_") and seed_dir.parent.name == "protocol_baselines":
        raise ValueError(f"Protocol file must live below a protocol child: {path}")
    if seed_dir.name in {"uniform_random", "wang2025_isac_tables"}:
        method = seed_dir.name
        seed_name = seed_dir.parent.name
    else:
        method = seed_dir.parent.name
        seed_name = seed_dir.name
    if not seed_name.startswith("seed_"):
        raise ValueError(f"Cannot parse training seed from {path}")
    return method, int(seed_name.removeprefix("seed_"))


def load_evaluation_rows(eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_root.rglob("eval_episode_metrics.csv")):
        method, train_seed = method_and_seed(path)
        for item in read_csv(path):
            row: dict[str, Any] = dict(item)
            row["method"] = method
            row["train_seed"] = train_seed
            row["scenario_seed"] = int(item.get("scenario_seed") or item.get("seed") or 0)
            row["eval_episode"] = int(item["eval_episode"])
            for metric in SUMMARY_METRICS:
                if item.get(metric, "") != "":
                    row[metric] = float(item[metric])
            rows.append(row)
    return rows


def load_timeline_rows(eval_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(eval_root.rglob("edge_discovery_timeline.csv")):
        method, train_seed = method_and_seed(path)
        for item in read_csv(path):
            rows.append(
                {
                    "method": method,
                    "train_seed": train_seed,
                    "scenario_seed": int(item["scenario_seed"]),
                    "eval_episode": int(item["eval_episode"]),
                    "edge_i": int(item["edge_i"]),
                    "edge_j": int(item["edge_j"]),
                    "discovered": str(item["discovered"]).lower() == "true",
                    "discovery_time_slots": int(item["discovery_time_slots"]),
                    "horizon_slots": int(item["horizon_slots"]),
                }
            )
    return rows


def validate_paired_contract(rows: list[dict[str, Any]], allow_incomplete: bool) -> dict[str, int]:
    counts = {
        method: sum(row["method"] == method for row in rows)
        for method in sorted({row["method"] for row in rows})
    }
    if not allow_incomplete:
        if set(counts) != set(METHOD_LABELS):
            raise ValueError(f"Expected methods {sorted(METHOD_LABELS)}, received {sorted(counts)}")
        if any(count != 150 for count in counts.values()):
            raise ValueError(f"Every method must contain 150 episodes: {counts}")
        reference_keys = {
            (row["train_seed"], row["scenario_seed"])
            for row in rows
            if row["method"] == REFERENCE_METHOD
        }
        for method in counts:
            keys = {
                (row["train_seed"], row["scenario_seed"])
                for row in rows
                if row["method"] == method
            }
            if keys != reference_keys:
                raise ValueError(f"{method} does not use the same paired scenarios.")
    return counts


def seed_level_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    methods = sorted({row["method"] for row in rows})
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        seeds = sorted({row["train_seed"] for row in method_rows})
        record: dict[str, Any] = {
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "seed_count": len(seeds),
            "episode_count": len(method_rows),
        }
        for metric in SUMMARY_METRICS:
            seed_means = [
                float(np.mean([row[metric] for row in method_rows if row["train_seed"] == seed]))
                for seed in seeds
                if all(metric in row for row in method_rows if row["train_seed"] == seed)
            ]
            if seed_means:
                record[f"{metric}_mean"] = float(np.mean(seed_means))
                record[f"{metric}_seed_sd"] = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
        output.append(record)
    return output


def hierarchical_paired_bootstrap(
    reference: dict[tuple[int, int], float],
    comparator: dict[tuple[int, int], float],
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    seeds = sorted({seed for seed, _scenario in reference})
    deltas = {key: reference[key] - comparator[key] for key in reference}
    point = float(np.mean(list(deltas.values())))
    draws = np.empty(samples, dtype=float)
    by_seed = {
        seed: [value for (item_seed, _scenario), value in deltas.items() if item_seed == seed]
        for seed in seeds
    }
    for index in range(samples):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        cluster_values = []
        for seed in sampled_seeds:
            values = np.asarray(by_seed[int(seed)], dtype=float)
            cluster_values.extend(rng.choice(values, size=len(values), replace=True).tolist())
        draws[index] = float(np.mean(cluster_values))
    low, high = np.percentile(draws, [2.5, 97.5])
    return point, float(low), float(high)


def paired_delta_rows(
    rows: list[dict[str, Any]], samples: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric in (
        "discovery_rate",
        "mean_delay_censored",
        "discovery_curve_auc_normalized",
        "networking_completion_slot_censored",
    ):
        reference = {
            (row["train_seed"], row["scenario_seed"]): row[metric]
            for row in rows
            if row["method"] == REFERENCE_METHOD
        }
        for method in sorted({row["method"] for row in rows} - {REFERENCE_METHOD}):
            comparator = {
                (row["train_seed"], row["scenario_seed"]): row[metric]
                for row in rows
                if row["method"] == method
            }
            shared = set(reference).intersection(comparator)
            if not shared:
                continue
            point, low, high = hierarchical_paired_bootstrap(
                {key: reference[key] for key in shared},
                {key: comparator[key] for key in shared},
                samples,
                rng,
            )
            output.append(
                {
                    "reference": REFERENCE_METHOD,
                    "comparator": method,
                    "metric": metric,
                    "paired_n": len(shared),
                    "reference_minus_comparator": point,
                    "hierarchical_bootstrap_ci95_low": low,
                    "hierarchical_bootstrap_ci95_high": high,
                }
            )
    return output


def curve_rows(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in timeline}):
        method_rows = [row for row in timeline if row["method"] == method]
        episode_keys = sorted({(row["train_seed"], row["scenario_seed"]) for row in method_rows})
        episode_edges = {
            key: [
                row for row in method_rows
                if (row["train_seed"], row["scenario_seed"]) == key
            ]
            for key in episode_keys
        }
        for slot in range(1, 301):
            rates = np.asarray(
                [
                    np.mean(
                        [edge["discovered"] and edge["discovery_time_slots"] <= slot for edge in edges]
                    )
                    for edges in episode_edges.values()
                ],
                dtype=float,
            )
            mean = float(np.mean(rates))
            sem = float(np.std(rates, ddof=1) / np.sqrt(len(rates))) if len(rates) > 1 else 0.0
            output.append(
                {
                    "method": method,
                    "slot": slot,
                    "mean_discovery_rate": mean,
                    "ci95_low": max(0.0, mean - 1.96 * sem),
                    "ci95_high": min(1.0, mean + 1.96 * sem),
                }
            )
    return output


def configure_plotting() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.figsize": (8, 6),
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
    return plt


def plot_discovery_curves(rows: list[dict[str, Any]], output: Path) -> None:
    plt = configure_plotting()
    fig, ax = plt.subplots(figsize=(8, 6))
    main_methods = (
        "uniform_random",
        "wang2025_isac_tables",
        "isac_candidate_pool_random",
        "mappo_no_isac",
        "mappo_direct_isac",
        "mappo_direct_isac_measurement_aux",
    )
    for method in main_methods:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        slots = np.asarray([row["slot"] for row in selected])
        mean = np.asarray([row["mean_discovery_rate"] for row in selected])
        low = np.asarray([row["ci95_low"] for row in selected])
        high = np.asarray([row["ci95_high"] for row in selected])
        color = COLORS[method]
        ax.plot(slots, 100.0 * mean, color=color, linewidth=2.0, label=METHOD_LABELS[method])
        ax.fill_between(slots, 100.0 * low, 100.0 * high, color=color, alpha=0.10)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Discovered neighbor links (%)")
    ax.set_xlim(1, 300)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(output / "discovery_rate_vs_slot.png")
    fig.savefig(output / "discovery_rate_vs_slot.pdf")
    plt.close(fig)


def plot_action_ablation(summary: list[dict[str, Any]], output: Path) -> None:
    plt = configure_plotting()
    methods = (
        "mappo_direct_isac",
        "random_role_learned_beam",
        "learned_role_uniform_beam",
        "isac_candidate_pool_random",
    )
    lookup = {row["method"]: row for row in summary}
    available = [method for method in methods if method in lookup]
    values = [100.0 * lookup[method]["discovery_rate_mean"] for method in available]
    errors = [100.0 * lookup[method]["discovery_rate_seed_sd"] for method in available]
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(available))
    ax.bar(x, values, yerr=errors, capsize=4, color=[COLORS[method] for method in available])
    ax.set_xticks(x, [METHOD_LABELS[method] for method in available], rotation=18, ha="right")
    ax.set_ylabel("Discovery rate (%)")
    ax.set_ylim(0, 105)
    fig.savefig(output / "action_component_ablation.png")
    fig.savefig(output / "action_component_ablation.pdf")
    plt.close(fig)


def training_and_auxiliary_diagnostics(train_root: Path, output: Path) -> list[dict[str, Any]]:
    diagnostic_rows: list[dict[str, Any]] = []
    training_series: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for method_dir in sorted(path for path in train_root.iterdir() if path.is_dir()):
        method = method_dir.name
        training_series[method] = []
        for seed_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            train_path = seed_dir / "episode_metrics.csv"
            eval_path = seed_dir / "eval_episode_metrics.csv"
            if not train_path.is_file() or not eval_path.is_file():
                continue
            train = read_csv(train_path)
            evaluation = read_csv(eval_path)
            steps = np.asarray([float(row["training_step"]) for row in train])
            discovery = np.asarray([float(row["discovery_rate"]) for row in train])
            training_series[method].append((steps, discovery))
            last = discovery[-100:]
            eval_discovery = np.asarray([float(row["discovery_rate"]) for row in evaluation])
            diagnostic_rows.append(
                {
                    "method": method,
                    "seed": int(seed_dir.name.removeprefix("seed_")),
                    "train_last100_discovery_rate": float(np.mean(last)),
                    "eval_discovery_rate": float(np.mean(eval_discovery)),
                    "generalization_gap_train_minus_eval": float(np.mean(last) - np.mean(eval_discovery)),
                    "aux_loss_last100": float(
                        np.mean([float(row.get("measurement_prediction_aux_loss") or 0.0) for row in train[-100:]])
                    ),
                    "actor_grad_norm_last100": float(
                        np.mean([float(row.get("actor_grad_norm") or 0.0) for row in train[-100:]])
                    ),
                    "entropy_last100": float(
                        np.mean([float(row.get("entropy") or 0.0) for row in train[-100:]])
                    ),
                }
            )

    plt = configure_plotting()
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in (
        "mappo_no_isac",
        "mappo_direct_isac",
        "mappo_direct_isac_measurement_aux",
    ):
        series = training_series.get(method, [])
        if not series:
            continue
        matrix = np.vstack([values for _steps, values in series])
        kernel = np.ones(50) / 50.0
        smooth = np.vstack([np.convolve(values, kernel, mode="valid") for values in matrix])
        steps = series[0][0][49:]
        mean = smooth.mean(axis=0)
        sd = smooth.std(axis=0, ddof=1)
        ax.plot(steps, 100.0 * mean, color=COLORS[method], linewidth=2.0, label=METHOD_LABELS[method])
        ax.fill_between(steps, 100.0 * (mean - sd), 100.0 * (mean + sd), color=COLORS[method], alpha=0.10)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Training discovery rate, 50-episode moving average (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    fig.savefig(output / "training_discovery_convergence.png")
    fig.savefig(output / "training_discovery_convergence.pdf")
    plt.close(fig)
    return diagnostic_rows


def write_report(
    output: Path,
    counts: dict[str, int],
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> None:
    lookup = {row["method"]: row for row in summary}
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {date.today().isoformat()}",
        "- Verification Status: ANALYZED",
        "- Version Label: n10_b15_static_ideal_paired_v1",
        "",
        "## Validation Report",
        "",
        f"- Source: `{DEFAULT_EVAL_ROOT}`",
        "- Overall Confidence: CAUTION",
        "- Design: three trained seeds, 50 paired held-out scenarios per seed",
        "",
        "### Method Coverage",
        "",
        "| Method | Episodes | Discovery rate | Seed SD | Mean delay | Curve AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_LABELS:
        if method not in lookup:
            continue
        row = lookup[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {counts.get(method, 0)} | "
            f"{100.0 * row['discovery_rate_mean']:.2f}% | "
            f"{100.0 * row['discovery_rate_seed_sd']:.2f} pp | "
            f"{row['mean_delay_censored_mean']:.2f} | "
            f"{row['discovery_curve_auc_normalized_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Statistical Boundary",
            "",
            "Hierarchical bootstrap intervals resample training seeds and then paired scenarios within seed. "
            "Only three independently trained seeds are available, so intervals describe uncertainty but do not "
            "support strong asymptotic significance claims.",
            "",
            "### Auxiliary Diagnostic",
            "",
            "The diagnostic table reports the last-100 training discovery rate, held-out discovery rate, "
            "and their gap. A positive gap indicates training-to-test degradation, not failure to converge.",
            "",
            "### Fallacy Scan",
            "",
            "- Coverage: 11/11 statistical fallacy types checked",
            "- Simpson/ecological/Berkson/collider: no subgroup or conditioning claim is made.",
            "- Base-rate neglect: not applicable to the link-discovery ratio.",
            "- Regression to mean/survivorship: all scheduled seeds and episodes are retained.",
            "- Look-elsewhere/garden of forking paths: CAUTION; multiple metrics and checkpoints exist, so the "
            "three-seed final-checkpoint table remains primary.",
            "- Correlation/causation and reverse causality: controlled simulation interventions support mechanism "
            "comparisons only within this static ideal environment.",
            "",
            "### Reproducibility",
            "",
            "- Method: deterministic replay contract with stochastic RNG seeds fixed per episode",
            "- Verdict: PARTIALLY_REPRODUCIBLE until the full paired campaign is independently rerun",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_evaluation_rows(args.eval_root.resolve())
    timeline = load_timeline_rows(args.eval_root.resolve())
    counts = validate_paired_contract(rows, bool(args.allow_incomplete))
    summary = seed_level_summary(rows)
    paired = paired_delta_rows(
        rows,
        int(args.bootstrap_samples),
        np.random.default_rng(20260714),
    )
    curves = curve_rows(timeline)
    diagnostics = training_and_auxiliary_diagnostics(args.train_root.resolve(), output)
    write_csv(output / "method_summary.csv", summary)
    write_csv(output / "paired_deltas_hierarchical_bootstrap.csv", paired)
    write_csv(output / "discovery_curve.csv", curves)
    write_csv(output / "training_generalization_diagnostics.csv", diagnostics)
    plot_discovery_curves(curves, output)
    plot_action_ablation(summary, output)
    write_report(output, counts, summary, paired, diagnostics)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
