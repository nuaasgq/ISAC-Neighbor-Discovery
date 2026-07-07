"""Build the Phase10 learned-component ablation evidence package.

The raw evaluation directories are intentionally kept under the ignored
`05_simulation/results_raw` tree. This script commits only compact summary
tables, raw-file hashes, and manuscript-facing 4:3 figures.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev


RAW_ROOT = Path("05_simulation/results_raw/marl_campaign/phase10_learned_component_ablation_b10_5ep")
OUT_DIR = Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep")
FIG_DIR = Path("06_analysis/figures/marl/p10_learned_component_ablation_b10_3ep")
REPORT = Path("06_analysis/phase10_learned_component_ablation_report_20260707.md")

METHOD_ORDER = [
    "trained_full",
    "random_weights_full",
    "zero_weights_rule_only",
    "trained_no_rule_residual",
    "trained_no_candidate_mask",
]

DISPLAY_NAMES = {
    "trained_full": "Trained full",
    "random_weights_full": "Random weights",
    "zero_weights_rule_only": "Zero weights",
    "trained_no_rule_residual": "No rule residual",
    "trained_no_candidate_mask": "No candidate mask",
}

METRICS = [
    "discovery_rate",
    "collision_penalized_discovery_rate",
    "collision_count",
    "collisions_per_discovery_censored",
    "empty_scan_ratio",
    "lambda2",
    "energy_per_discovery_censored_j",
    "discoveries_per_joule",
    "episode_return_mean_per_agent",
]

PRIMARY_METRICS = [
    "discovery_rate",
    "collision_penalized_discovery_rate",
    "collision_count",
    "lambda2",
    "empty_scan_ratio",
]

COLORS = {
    "trained_full": "#0072B2",
    "random_weights_full": "#D55E00",
    "zero_weights_rule_only": "#CC79A7",
    "trained_no_rule_residual": "#009E73",
    "trained_no_candidate_mask": "#E69F00",
}

FIGSIZE = (6.4, 4.8)
DPI = 240


@dataclass(frozen=True)
class MetricStats:
    label: str
    metric: str
    episodes: int
    mean: float
    std: float
    ci95: float
    min_value: float
    max_value: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def ci95(std_value: float, n: int) -> float:
    return 1.96 * std_value / (n ** 0.5) if n > 0 else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(label: str) -> dict:
    path = RAW_ROOT / label / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metric_stats(label: str, rows: list[dict[str, str]]) -> list[MetricStats]:
    output: list[MetricStats] = []
    for metric in METRICS:
        values = [as_float(row.get(metric)) for row in rows]
        std_value = stdev(values) if len(values) > 1 else 0.0
        output.append(
            MetricStats(
                label=label,
                metric=metric,
                episodes=len(values),
                mean=mean(values),
                std=std_value,
                ci95=ci95(std_value, len(values)),
                min_value=min(values),
                max_value=max(values),
            )
        )
    return output


def collect_stats() -> tuple[list[MetricStats], list[dict[str, object]], list[dict[str, object]]]:
    stats: list[MetricStats] = []
    run_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for label in METHOD_ORDER:
        data_path = RAW_ROOT / label / "eval_episode_metrics.csv"
        manifest_path = RAW_ROOT / label / "manifest.json"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing ablation CSV: {data_path}")
        rows = read_csv(data_path)
        if len(rows) < 3:
            raise ValueError(f"Expected at least 3 rows for {label}; found {len(rows)}.")
        manifest = load_manifest(label)
        stats.extend(metric_stats(label, rows[:3]))
        run_rows.append(
            {
                "label": label,
                "display_name": DISPLAY_NAMES[label],
                "policy_ablation": manifest.get("policy_ablation", ""),
                "checkpoint_loaded": manifest.get("checkpoint_loaded", ""),
                "feature_flags": json.dumps(manifest.get("feature_flags", {}), ensure_ascii=False, sort_keys=True),
                "eval_episodes": manifest.get("eval_episodes", len(rows)),
                "slots_per_episode": manifest.get("slots_per_episode", ""),
                "node_count": manifest.get("node_count", ""),
                "beam_count": manifest.get("beam_count", ""),
                "seed_base": rows[0].get("seed", ""),
                "raw_output": (RAW_ROOT / label).as_posix(),
            }
        )
        for artifact_type, path in (
            ("eval_csv", data_path),
            ("resource_log", RAW_ROOT / label / "resource_log.csv"),
            ("manifest", manifest_path),
        ):
            source_rows.append(
                {
                    "label": label,
                    "artifact_type": artifact_type,
                    "path": path.as_posix(),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else "",
                    "sha256": sha256_file(path) if path.exists() else "",
                }
            )
    return stats, run_rows, source_rows


def summary_rows(stats: list[MetricStats]) -> list[dict[str, object]]:
    return [
        {
            "label": item.label,
            "display_name": DISPLAY_NAMES[item.label],
            "metric": item.metric,
            "episodes": item.episodes,
            "mean": item.mean,
            "std": item.std,
            "ci95": item.ci95,
            "min": item.min_value,
            "max": item.max_value,
        }
        for item in stats
    ]


def wide_metric_index(stats: list[MetricStats]) -> dict[tuple[str, str], MetricStats]:
    return {(item.label, item.metric): item for item in stats}


def delta_rows(stats: list[MetricStats]) -> list[dict[str, object]]:
    indexed = wide_metric_index(stats)
    rows: list[dict[str, object]] = []
    baseline = "trained_full"
    for label in METHOD_ORDER:
        for metric in PRIMARY_METRICS:
            base = indexed[(baseline, metric)].mean
            value = indexed[(label, metric)].mean
            rows.append(
                {
                    "label": label,
                    "display_name": DISPLAY_NAMES[label],
                    "metric": metric,
                    "mean": value,
                    "trained_full_mean": base,
                    "absolute_delta_vs_trained_full": value - base,
                    "relative_delta_vs_trained_full": (value - base) / max(abs(base), abs(value), 1e-12),
                }
            )
    return rows


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": FIGSIZE,
            "figure.dpi": DPI,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    return plt


def plot_efficiency(stats: list[MetricStats]) -> Path:
    plt = setup_matplotlib()
    indexed = wide_metric_index(stats)
    labels = METHOD_ORDER
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)
    for ax, metric, title in (
        (axes[0], "discovery_rate", "Discovery rate"),
        (axes[1], "collision_penalized_discovery_rate", "Collision-penalized discovery"),
    ):
        means = [indexed[(label, metric)].mean for label in labels]
        errs = [indexed[(label, metric)].ci95 for label in labels]
        ax.bar(x, means, yerr=errs, capsize=3, color=[COLORS[label] for label in labels], alpha=0.88)
        ax.set_title(title)
        ax.set_ylim(0.0, max(means) * 1.25)
        ax.set_xticks(list(x))
        ax.set_xticklabels([DISPLAY_NAMES[label] for label in labels], rotation=28, ha="right")
    fig.tight_layout()
    path = FIG_DIR / "learned_ablation_b10_discovery_efficiency.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_tradeoff(stats: list[MetricStats]) -> Path:
    plt = setup_matplotlib()
    indexed = wide_metric_index(stats)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    label_offsets = {
        "trained_full": (9, -2),
        "random_weights_full": (9, 6),
        "zero_weights_rule_only": (9, 6),
        "trained_no_rule_residual": (9, 10),
        "trained_no_candidate_mask": (9, 8),
    }
    for label in METHOD_ORDER:
        collisions = indexed[(label, "collision_count")].mean
        cpd = indexed[(label, "collision_penalized_discovery_rate")].mean
        discovery = indexed[(label, "discovery_rate")].mean
        ax.scatter(collisions, cpd, s=55 + 280 * discovery, color=COLORS[label], alpha=0.86, label=DISPLAY_NAMES[label])
        ax.annotate(
            DISPLAY_NAMES[label],
            (collisions, cpd),
            xytext=label_offsets[label],
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean collisions per episode")
    ax.set_ylabel("Collision-penalized discovery")
    ax.set_title("Learned component tradeoff at N=100, B=10")
    fig.tight_layout()
    path = FIG_DIR / "learned_ablation_b10_collision_tradeoff.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_report(stats: list[MetricStats], figures: list[Path]) -> None:
    indexed = wide_metric_index(stats)

    def mean_value(label: str, metric: str) -> float:
        return indexed[(label, metric)].mean

    trained_cpd = mean_value("trained_full", "collision_penalized_discovery_rate")
    random_cpd = mean_value("random_weights_full", "collision_penalized_discovery_rate")
    zero_cpd = mean_value("zero_weights_rule_only", "collision_penalized_discovery_rate")
    random_collision_reduction = 1.0 - mean_value("trained_full", "collision_count") / max(mean_value("random_weights_full", "collision_count"), 1e-12)
    zero_collision_reduction = 1.0 - mean_value("trained_full", "collision_count") / max(mean_value("zero_weights_rule_only", "collision_count"), 1e-12)
    lines = [
        "# Phase10 Learned-Component Ablation - 2026-07-07",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Verification Status: ANALYZED",
        "- Scope: N=100, B=10, 3000-slot stochastic transfer, 3 paired seed episodes",
        "",
        "## Summary",
        "",
        f"- Trained full CPD mean: {trained_cpd:.6f}; random-weight CPD mean: {random_cpd:.6f}; zero-weight/rule-only CPD mean: {zero_cpd:.6f}.",
        f"- Trained full reduces mean collisions by {random_collision_reduction:.1%} vs random weights and {zero_collision_reduction:.1%} vs zero-weight/rule-only.",
        "- Raw discovery and lambda2 are not always highest for the trained checkpoint; random/zero-weight rule-dominated policies discover more links by transmitting aggressively, but with much higher collision burden.",
        "- Disabling the hard candidate mask improves CPD in this 3-episode probe but also raises empty-scan ratio sharply, so it is an efficiency tradeoff signal rather than a ready replacement for the final policy.",
        "",
        "## Metric Means",
        "",
        "| Variant | Discovery | CPD | Collisions | Collisions/discovery | Empty scan | Lambda2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in METHOD_ORDER:
        lines.append(
            "| {name} | {disc:.4f} | {cpd:.4f} | {coll:.1f} | {cpdly:.3f} | {empty:.4f} | {lam:.3f} |".format(
                name=DISPLAY_NAMES[label],
                disc=mean_value(label, "discovery_rate"),
                cpd=mean_value(label, "collision_penalized_discovery_rate"),
                coll=mean_value(label, "collision_count"),
                cpdly=mean_value(label, "collisions_per_discovery_censored"),
                empty=mean_value(label, "empty_scan_ratio"),
                lam=mean_value(label, "lambda2"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This ablation separates learned weights from structured ISAC/rule priors but does not prove a universally dominant learned policy.",
            "The defensible claim is narrower: learned weights materially suppress collisions relative to random/zero-weight policies under the same ISAC features, while rule residuals and candidate-mask design control the discovery/collision/empty-scan tradeoff.",
            "",
            "## Generated Figures",
            "",
        ]
    )
    lines.extend(f"- `{path.as_posix()}`" for path in figures)
    lines.extend(
        [
            "",
            "## Generated Tables",
            "",
            f"- `{(OUT_DIR / 'ablation_metric_summary.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'ablation_vs_trained_full.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'ablation_run_index.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'ablation_source_file_hashes.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'manifest.json').as_posix()}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stats, run_rows, source_rows = collect_stats()
    figures = [plot_efficiency(stats), plot_tradeoff(stats)]
    write_csv(
        OUT_DIR / "ablation_metric_summary.csv",
        summary_rows(stats),
        ["label", "display_name", "metric", "episodes", "mean", "std", "ci95", "min", "max"],
    )
    write_csv(
        OUT_DIR / "ablation_vs_trained_full.csv",
        delta_rows(stats),
        ["label", "display_name", "metric", "mean", "trained_full_mean", "absolute_delta_vs_trained_full", "relative_delta_vs_trained_full"],
    )
    write_csv(
        OUT_DIR / "ablation_run_index.csv",
        run_rows,
        [
            "label",
            "display_name",
            "policy_ablation",
            "checkpoint_loaded",
            "feature_flags",
            "eval_episodes",
            "slots_per_episode",
            "node_count",
            "beam_count",
            "seed_base",
            "raw_output",
        ],
    )
    write_csv(
        OUT_DIR / "ablation_source_file_hashes.csv",
        source_rows,
        ["label", "artifact_type", "path", "exists", "size_bytes", "sha256"],
    )
    write_report(stats, figures)

    output_files = [
        OUT_DIR / "ablation_metric_summary.csv",
        OUT_DIR / "ablation_vs_trained_full.csv",
        OUT_DIR / "ablation_run_index.csv",
        OUT_DIR / "ablation_source_file_hashes.csv",
        REPORT,
        *figures,
    ]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase10 learned-component ablation",
        "raw_root": RAW_ROOT.as_posix(),
        "labels": METHOD_ORDER,
        "metrics": METRICS,
        "figures": [path.as_posix() for path in figures],
        "outputs": [path.as_posix() for path in output_files],
        "output_hashes": {path.as_posix(): sha256_file(path) for path in output_files},
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "created_at_utc": manifest["created_at_utc"],
                "labels": METHOD_ORDER,
                "output_dir": OUT_DIR.as_posix(),
                "figures": [path.as_posix() for path in figures],
                "report": REPORT.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
