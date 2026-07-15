from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "05_simulation" / "results_raw" / "paper_main_n10_b15_paired_eval_50ep"
DEFAULT_OUTPUT = ROOT / "06_analysis" / "paper_main_n10_b15_paired_eval_20260715"
TRAIN_SEEDS = (69260715, 69261724, 69262733)
METHOD_ORDER = ("uniform_random", "wang2025", "candidate_random", "residual_mask_mappo")
LABELS = {
    "uniform_random": "Blind random",
    "wang2025": "Wang2025",
    "candidate_random": "Residual candidate random",
    "residual_mask_mappo": "Residual-mask MAPPO",
}
COLORS = {
    "uniform_random": "#7F7F7F",
    "wang2025": "#0072B2",
    "candidate_random": "#E69F00",
    "residual_mask_mappo": "#009E73",
}
METRICS = (
    "discovery_rate",
    "discovery_rate_at_50_slots",
    "discovery_rate_at_100_slots",
    "discovery_rate_at_150_slots",
    "discovery_rate_at_200_slots",
    "mean_delay_censored",
    "discovery_curve_auc_normalized",
    "empty_scan_ratio",
    "aligned_handshake_opportunities",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the frozen paper-main paired evaluation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric_rows(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        scenario = int(row["scenario_seed"])
        item = {metric: float(row[metric]) for metric in METRICS}
        tx = float(row["tx_actions"])
        rx = float(row["rx_actions"])
        item["tx_ratio"] = tx / max(1.0, tx + rx)
        result[scenario] = item
    return result


def load_methods(input_root: Path) -> tuple[dict[str, dict[int, dict[str, float]]], dict[int, dict[int, dict[str, float]]]]:
    methods = {
        "uniform_random": numeric_rows(
            read_csv(input_root / "protocol_baselines" / "uniform_random" / "eval_episode_metrics.csv")
        ),
        "wang2025": numeric_rows(
            read_csv(
                input_root
                / "protocol_baselines"
                / "wang2025_isac_tables"
                / "eval_episode_metrics.csv"
            )
        ),
        "candidate_random": numeric_rows(
            read_csv(input_root / "residual_candidate_random" / "eval_episode_metrics.csv")
        ),
    }
    seed_rows: dict[int, dict[int, dict[str, float]]] = {}
    for seed in TRAIN_SEEDS:
        seed_rows[seed] = numeric_rows(
            read_csv(
                input_root
                / "residual_mask_mappo"
                / f"seed_{seed}"
                / "eval_episode_metrics.csv"
            )
        )
    scenarios = sorted(methods["uniform_random"])
    for name, rows in methods.items():
        if sorted(rows) != scenarios:
            raise ValueError(f"{name} does not use the common scenario set.")
    for seed, rows in seed_rows.items():
        if sorted(rows) != scenarios:
            raise ValueError(f"MAPPO seed {seed} does not use the common scenario set.")
    methods["residual_mask_mappo"] = {
        scenario: {
            metric: float(np.mean([seed_rows[seed][scenario][metric] for seed in TRAIN_SEEDS]))
            for metric in (*METRICS, "tx_ratio")
        }
        for scenario in scenarios
    }
    return methods, seed_rows


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean, mean
    half = float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))
    return mean, mean - half, mean + half


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(methods: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        scenarios = sorted(methods[method])
        row: dict[str, object] = {"method": method, "label": LABELS[method], "scenarios": len(scenarios)}
        for metric in (*METRICS, "tx_ratio"):
            values = np.asarray([methods[method][scenario][metric] for scenario in scenarios])
            mean, low, high = mean_ci(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return rows


def paired_rows(methods: dict[str, dict[int, dict[str, float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    main = methods["residual_mask_mappo"]
    scenarios = sorted(main)
    for baseline in ("uniform_random", "wang2025", "candidate_random"):
        for metric in (
            "discovery_rate",
            "discovery_rate_at_50_slots",
            "discovery_rate_at_100_slots",
            "discovery_rate_at_150_slots",
            "discovery_rate_at_200_slots",
            "mean_delay_censored",
            "discovery_curve_auc_normalized",
        ):
            delta = np.asarray(
                [main[scenario][metric] - methods[baseline][scenario][metric] for scenario in scenarios]
            )
            mean, low, high = mean_ci(delta)
            test = stats.ttest_1samp(delta, popmean=0.0)
            rows.append(
                {
                    "comparison": f"residual_mask_mappo-minus-{baseline}",
                    "metric": metric,
                    "paired_scenarios": len(scenarios),
                    "mean_delta": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_t_pvalue": float(test.pvalue),
                    "wins": int(np.sum(delta > 1e-12)),
                    "ties": int(np.sum(np.abs(delta) <= 1e-12)),
                    "losses": int(np.sum(delta < -1e-12)),
                }
            )
    return rows


def seed_summary_rows(seed_rows: dict[int, dict[int, dict[str, float]]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in TRAIN_SEEDS:
        scenarios = sorted(seed_rows[seed])
        row: dict[str, object] = {"training_seed": seed, "scenarios": len(scenarios)}
        for metric in (*METRICS, "tx_ratio"):
            values = np.asarray([seed_rows[seed][scenario][metric] for scenario in scenarios])
            mean, low, high = mean_ci(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return rows


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.figsize": (8, 6),
            "figure.dpi": 240,
            "savefig.bbox": "tight",
        }
    )


def plot_discovery_curves(methods: dict[str, dict[int, dict[str, float]]], output: Path) -> None:
    slots = np.asarray([0, 50, 100, 150, 200, 300])
    fields = (
        None,
        "discovery_rate_at_50_slots",
        "discovery_rate_at_100_slots",
        "discovery_rate_at_150_slots",
        "discovery_rate_at_200_slots",
        "discovery_rate",
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHOD_ORDER:
        scenarios = sorted(methods[method])
        means = [0.0]
        lows = [0.0]
        highs = [0.0]
        for field in fields[1:]:
            values = np.asarray([methods[method][scenario][field] for scenario in scenarios])
            mean, low, high = mean_ci(values)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        ax.plot(slots, means, marker="o", linewidth=2.0, markersize=4.5, label=LABELS[method], color=COLORS[method])
        ax.fill_between(slots, lows, highs, color=COLORS[method], alpha=0.10, linewidth=0)
    ax.set_xlabel("Slot")
    ax.set_ylabel("Direct neighbor discovery rate")
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=True, ncol=1, loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "paper_main_discovery_vs_slot.png")
    fig.savefig(output / "paper_main_discovery_vs_slot.pdf")
    plt.close(fig)


def plot_final_metrics(methods: dict[str, dict[int, dict[str, float]]], output: Path) -> None:
    labels = [LABELS[method] for method in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    for ax, metric, ylabel in (
        (axes[0], "discovery_rate", "Final discovery rate"),
        (axes[1], "mean_delay_censored", "Mean discovery delay (slot)"),
    ):
        means = []
        errors = []
        for method in METHOD_ORDER:
            scenarios = sorted(methods[method])
            values = np.asarray([methods[method][scenario][metric] for scenario in scenarios])
            mean, low, high = mean_ci(values)
            means.append(mean)
            errors.append(high - mean)
        ax.bar(x, means, yerr=errors, capsize=3, color=[COLORS[method] for method in METHOD_ORDER], width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.set_ylabel(ylabel)
        if metric == "discovery_rate":
            ax.set_ylim(0, 0.65)
        else:
            ax.set_ylim(180, 285)
    fig.tight_layout()
    fig.savefig(output / "paper_main_final_rate_delay.png")
    fig.savefig(output / "paper_main_final_rate_delay.pdf")
    plt.close(fig)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build_report(
    aggregate: list[dict[str, object]],
    paired: list[dict[str, object]],
    output: Path,
) -> None:
    by_method = {str(row["method"]): row for row in aggregate}
    delta_by_key = {(str(row["comparison"]), str(row["metric"])): row for row in paired}
    main = by_method["residual_mask_mappo"]
    lines = [
        "# Paper-main N=10/B=15 paired evaluation",
        "",
        "## Contract",
        "",
        "- 50 common held-out scenarios: seeds 79260715--79260764.",
        "- N=10, planar Gauss-Markov mobility, 15-degree beams, one RF chain, 300 slots at 5 ms/slot.",
        "- Noisy-count MIMO-OTFS sensing abstraction and close-in Rician/SINR communication PHY.",
        "- The MAPPO result averages three independently trained policies within each scenario before paired statistics.",
        "",
        "## Main results",
        "",
        "| Method | Slot 50 | Slot 100 | Slot 150 | Slot 200 | Final | Mean delay | TX ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = by_method[method]
        lines.append(
            "| {label} | {s50} | {s100} | {s150} | {s200} | {final} | {delay:.2f} | {tx} |".format(
                label=LABELS[method],
                s50=fmt_pct(float(row["discovery_rate_at_50_slots_mean"])),
                s100=fmt_pct(float(row["discovery_rate_at_100_slots_mean"])),
                s150=fmt_pct(float(row["discovery_rate_at_150_slots_mean"])),
                s200=fmt_pct(float(row["discovery_rate_at_200_slots_mean"])),
                final=fmt_pct(float(row["discovery_rate_mean"])),
                delay=float(row["mean_delay_censored_mean"]),
                tx=fmt_pct(float(row["tx_ratio_mean"])),
            )
        )
    lines.extend(["", "## Paired conclusions", ""])
    for baseline in ("uniform_random", "wang2025", "candidate_random"):
        row = delta_by_key[(f"residual_mask_mappo-minus-{baseline}", "discovery_rate")]
        lines.append(
            "- MAPPO minus {baseline}: {delta}; 95% CI [{low}, {high}], p={p:.3g}, W/T/L={wins}/{ties}/{losses}.".format(
                baseline=LABELS[baseline],
                delta=fmt_pct(float(row["mean_delta"])),
                low=fmt_pct(float(row["ci95_low"])),
                high=fmt_pct(float(row["ci95_high"])),
                p=float(row["paired_t_pvalue"]),
                wins=row["wins"],
                ties=row["ties"],
                losses=row["losses"],
            )
        )
    early = delta_by_key[("residual_mask_mappo-minus-wang2025", "discovery_rate_at_50_slots")]
    late = delta_by_key[("residual_mask_mappo-minus-wang2025", "discovery_rate_at_200_slots")]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"MAPPO reaches {fmt_pct(float(main['discovery_rate_at_50_slots_mean']))} by slot 50 and initially exceeds Wang by {fmt_pct(float(early['mean_delta']))}. The gap reverses by slot 200, where MAPPO minus Wang is {fmt_pct(float(late['mean_delta']))}.",
            "",
            "The main policy therefore learns useful early beam/role prioritization, but it does not sustain discovery. Its mean TX ratio is about 30%, versus 50% for all three baselines, which reduces late bidirectional rendezvous opportunities. The present campaign supports learnability and a gain over blind search, but it does not support superiority over Wang or the same residual candidate mechanism with random TX/RX and beam execution.",
            "",
            "This result should be treated as a method diagnosis, not selected-seed evidence. No further scale-transfer experiment is justified until the decentralized role policy avoids the persistent RX bias without rule-forced execution.",
        ]
    )
    (output / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    methods, seed_rows = load_methods(args.input.resolve())
    aggregate = aggregate_rows(methods)
    paired = paired_rows(methods)
    seed_summary = seed_summary_rows(seed_rows)
    write_csv(output / "aggregate_metrics.csv", aggregate)
    write_csv(output / "paired_deltas.csv", paired)
    write_csv(output / "training_seed_summary.csv", seed_summary)
    per_scenario = []
    for scenario in sorted(methods["residual_mask_mappo"]):
        row: dict[str, object] = {"scenario_seed": scenario}
        for method in METHOD_ORDER:
            for metric in (*METRICS, "tx_ratio"):
                row[f"{method}_{metric}"] = methods[method][scenario][metric]
        per_scenario.append(row)
    write_csv(output / "per_scenario_metrics.csv", per_scenario)
    configure_plotting()
    plot_discovery_curves(methods, output)
    plot_final_metrics(methods, output)
    build_report(aggregate, paired, output)
    manifest = {
        "input": str(args.input.resolve()),
        "output": str(output),
        "training_seeds": list(TRAIN_SEEDS),
        "evaluation_seed_start": 79260715,
        "evaluation_scenarios": 50,
        "pairing_unit": "scenario after averaging three training seeds",
        "files": sorted(path.name for path in output.iterdir() if path.is_file()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
