"""Validate one independent Phase10 transfer re-run against the final table.

The raw re-run directory is intentionally left under the ignored
`05_simulation/results_raw` tree. This script extracts a compact, committed
evidence package with metric deltas and raw-file hashes so the manuscript can
state which part of the evidence chain has been independently re-run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable


DEFAULT_FINAL_CSV = Path("06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/marl_method_comparison.csv")
DEFAULT_RERUN_DIR = Path("05_simulation/results_raw/marl_campaign/p10_independent_rerun_gate31_b10_3000slot_10ep_seed41260731")
DEFAULT_OUTPUT_DIR = Path("06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation")
DEFAULT_REPORT = Path("06_analysis/phase10_independent_rerun_validation_20260707.md")

METRICS = [
    "discovery_rate",
    "collision_penalized_discovery_rate",
    "empty_scan_ratio",
    "lambda2",
    "collision_count",
]


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    original_mean: float
    rerun_mean: float
    rerun_std: float
    rerun_ci95: float
    absolute_diff: float
    relative_diff: float
    tolerance: float
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build independent Phase10 re-run validation package.")
    parser.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    parser.add_argument("--rerun-dir", type=Path, default=DEFAULT_RERUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--method", default="gated_contention_actor")
    parser.add_argument("--beamwidth", type=float, default=10.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def ci95(std: float, n: int) -> float:
    return 1.96 * std / (n ** 0.5) if n > 0 else 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_row(rows: list[dict[str, str]], method: str, beamwidth: float) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("method") == method and abs(as_float(row.get("beamwidth_deg")) - float(beamwidth)) < 1e-9
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one final row for method={method}, beamwidth={beamwidth}; found {len(matches)}.")
    return matches[0]


def compare_metrics(original: dict[str, str], rerun_rows: list[dict[str, str]], tolerance: float) -> list[MetricComparison]:
    comparisons: list[MetricComparison] = []
    for metric in METRICS:
        values = [as_float(row.get(metric)) for row in rerun_rows]
        rerun_mean = mean(values)
        rerun_std = stdev(values) if len(values) > 1 else 0.0
        original_mean = as_float(original.get(f"{metric}_mean"))
        absolute_diff = rerun_mean - original_mean
        denominator = max(abs(original_mean), abs(rerun_mean), 1e-12)
        relative_diff = abs(absolute_diff) / denominator
        comparisons.append(
            MetricComparison(
                metric=metric,
                original_mean=original_mean,
                rerun_mean=rerun_mean,
                rerun_std=rerun_std,
                rerun_ci95=ci95(rerun_std, len(values)),
                absolute_diff=absolute_diff,
                relative_diff=relative_diff,
                tolerance=float(tolerance),
                status="MATCH" if relative_diff <= float(tolerance) else "REVIEW",
            )
        )
    return comparisons


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def source_file_rows(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        exists = path.exists()
        rows.append(
            {
                "path": path.as_posix(),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else "",
                "sha256": sha256_file(path) if exists else "",
            }
        )
    return rows


def write_report(
    report: Path,
    comparisons: list[MetricComparison],
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    all_match = all(item.status == "MATCH" for item in comparisons)
    lines = [
        "# Phase10 Independent Re-Run Validation - 2026-07-07",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Verification Status: {'VERIFIED_PARTIAL' if all_match else 'ANALYZED_REVIEW'}",
        "- Scope: gated contention actor, N=100, B=10, 3000-slot stochastic transfer",
        "",
        "## Summary",
        "",
        f"- Original method row: `{summary['method']}`, B={summary['beamwidth_deg']} degrees.",
        f"- Independent re-run episodes: {summary['rerun_episodes']} with seed base {summary['seed_min']}--{summary['seed_max']}.",
        f"- Relative tolerance: {summary['relative_tolerance']:.2%}.",
        f"- Verdict: {'all checked metrics are within tolerance' if all_match else 'one or more metrics require review'}.",
        "",
        "## Metric Comparison",
        "",
        "| Metric | Original mean | Re-run mean | Re-run std | Re-run CI95 | Rel. diff | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in comparisons:
        lines.append(
            "| {metric} | {orig:.6f} | {rerun:.6f} | {std:.6f} | {ci:.6f} | {rel:.2%} | {status} |".format(
                metric=item.metric,
                orig=item.original_mean,
                rerun=item.rerun_mean,
                std=item.rerun_std,
                ci=item.rerun_ci95,
                rel=item.relative_diff,
                status=item.status,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a partial reproducibility check, not a full re-run of the entire Phase10 campaign.",
            "It verifies that the paper-facing default gated profile at B=10 remains within the predeclared tolerance under an independent stochastic seed range using the same checkpoint and transfer configuration.",
            "",
            "## Generated Files",
            "",
            f"- `{(output_dir / 'rerun_metric_summary.csv').as_posix()}`",
            f"- `{(output_dir / 'rerun_vs_original_comparison.csv').as_posix()}`",
            f"- `{(output_dir / 'rerun_source_file_hashes.csv').as_posix()}`",
            f"- `{(output_dir / 'manifest.json').as_posix()}`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    final_rows = read_csv(args.final_csv)
    rerun_data = read_csv(args.rerun_dir / "eval_episode_metrics.csv")
    manifest_path = args.rerun_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing re-run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    original = final_row(final_rows, args.method, args.beamwidth)
    comparisons = compare_metrics(original, rerun_data, args.relative_tolerance)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for metric in METRICS:
        values = [as_float(row.get(metric)) for row in rerun_data]
        metric_mean = mean(values)
        metric_std = stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(
            {
                "metric": metric,
                "episodes": len(values),
                "mean": metric_mean,
                "std": metric_std,
                "ci95": ci95(metric_std, len(values)),
                "min": min(values),
                "max": max(values),
            }
        )
    write_csv(
        args.output_dir / "rerun_metric_summary.csv",
        summary_rows,
        ["metric", "episodes", "mean", "std", "ci95", "min", "max"],
    )
    write_csv(
        args.output_dir / "rerun_vs_original_comparison.csv",
        [item.__dict__ for item in comparisons],
        [
            "metric",
            "original_mean",
            "rerun_mean",
            "rerun_std",
            "rerun_ci95",
            "absolute_diff",
            "relative_diff",
            "tolerance",
            "status",
        ],
    )
    source_files = [
        args.rerun_dir / "eval_episode_metrics.csv",
        args.rerun_dir / "resource_log.csv",
        args.rerun_dir / "progress.json",
        args.rerun_dir / "manifest.json",
    ]
    write_csv(
        args.output_dir / "rerun_source_file_hashes.csv",
        source_file_rows(source_files),
        ["path", "exists", "size_bytes", "sha256"],
    )

    seed_values = [int(float(row.get("seed", 0))) for row in rerun_data]
    summary = {
        "method": args.method,
        "beamwidth_deg": float(args.beamwidth),
        "rerun_episodes": len(rerun_data),
        "seed_min": min(seed_values) if seed_values else 0,
        "seed_max": max(seed_values) if seed_values else 0,
        "relative_tolerance": float(args.relative_tolerance),
    }
    write_report(args.report, comparisons, summary, args.output_dir)

    output_files = [
        args.output_dir / "rerun_metric_summary.csv",
        args.output_dir / "rerun_vs_original_comparison.csv",
        args.output_dir / "rerun_source_file_hashes.csv",
        args.report,
    ]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase10 independent rerun validation",
        "method": args.method,
        "beamwidth_deg": float(args.beamwidth),
        "rerun_dir": args.rerun_dir.as_posix(),
        "final_csv": args.final_csv.as_posix(),
        "relative_tolerance": float(args.relative_tolerance),
        "status_counts": {
            status: sum(1 for item in comparisons if item.status == status)
            for status in sorted({item.status for item in comparisons})
        },
        "all_metrics_match": all(item.status == "MATCH" for item in comparisons),
        "manifest": manifest,
        "outputs": [path.as_posix() for path in output_files],
        "output_hashes": {path.as_posix(): sha256_file(path) for path in output_files},
    }
    manifest_out = args.output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "created_at_utc": payload["created_at_utc"],
                "all_metrics_match": payload["all_metrics_match"],
                "status_counts": payload["status_counts"],
                "output_dir": args.output_dir.as_posix(),
                "report": args.report.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
