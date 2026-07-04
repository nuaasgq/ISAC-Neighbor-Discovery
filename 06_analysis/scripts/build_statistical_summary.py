from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime, timezone
from pathlib import Path


SOURCE_DEFS = (
    (
        "main_n100_density_transfer",
        "main",
        "06_analysis/paper_tables/round3_robustness/n100_density_multiseed/aggregate_metrics.csv",
    ),
    (
        "main_n100_fixed_transfer",
        "main",
        "06_analysis/paper_tables/round3_robustness/n100_fixed_multiseed/aggregate_metrics.csv",
    ),
    (
        "main_range_grid",
        "main",
        "06_analysis/paper_tables/round3_robustness/range_rc_rs_grid/aggregate_metrics.csv",
    ),
    (
        "main_error_profiles",
        "main",
        "06_analysis/paper_tables/round3_robustness/error_profiles/aggregate_metrics.csv",
    ),
    (
        "main_delay_ablation",
        "main",
        "06_analysis/paper_tables/round4_delay_ablation/aggregate_metrics.csv",
    ),
    (
        "mobility_boundary",
        "main_boundary",
        "06_analysis/paper_tables/round5_mobility_transfer/aggregate_metrics.csv",
    ),
    (
        "slot_duration_sensitivity",
        "supplement",
        "06_analysis/paper_tables/round6_slot_duration_sensitivity/aggregate_metrics.csv",
    ),
    (
        "round7_scale_beam_stress",
        "supplement",
        "06_analysis/paper_tables/round7_scale_beam_grid_light/aggregate_metrics.csv",
    ),
    (
        "round7_mobility_stress",
        "supplement",
        "06_analysis/paper_tables/round7_n100_multimobility_600slot/aggregate_metrics.csv",
    ),
    (
        "round7_error_profiles_quick",
        "supplement_sanity",
        "06_analysis/paper_tables/round7_error_profiles_quick/aggregate_metrics.csv",
    ),
    (
        "round7_error_profiles_light",
        "supplement",
        "06_analysis/paper_tables/round7_error_profiles_light/aggregate_metrics.csv",
    ),
    (
        "round8_mobility_missing_baselines",
        "supplement",
        "06_analysis/paper_tables/round8_n100_multimobility_missing_baselines_600slot/aggregate_metrics.csv",
    ),
    (
        "round8_error_profiles_b15_quick",
        "supplement_sanity",
        "06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_quick/aggregate_metrics.csv",
    ),
    (
        "round8_error_profiles_b15_full",
        "supplement",
        "06_analysis/paper_tables/round8_error_profiles_b15_gm_rw_600slot/aggregate_metrics.csv",
    ),
    (
        "round9_b3_full_baselines",
        "supplement_stress",
        "06_analysis/paper_tables/round9_n100_b3_full_baselines_600slot/aggregate_metrics.csv",
    ),
)

CONTEXT_COLUMNS = (
    "protocol",
    "node_count",
    "beamwidth_deg",
    "area_scale",
    "mobility_model",
    "range_mode",
    "communication_range_to_diagonal_ratio",
    "sensing_to_comm_range_ratio",
    "false_alarm_rate",
    "miss_detection_rate",
    "angular_cell_offset_std",
    "slot_duration_ms",
)

METRICS = (
    "discovery_rate",
    "empty_scan_ratio",
    "lambda2",
    "collision_count",
    "mean_discovery_delay",
    "collision_penalized_discovery_rate",
    "discovery_per_scan_action",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact statistical stability tables from archived sweeps.")
    parser.add_argument("--output", default="06_analysis/paper_tables/statistical_stability_summary")
    parser.add_argument("--node-count", type=int, default=100)
    return parser.parse_args()


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if math.isnan(value) or math.isinf(value):
        return ""
    return f"{value:.6g}"


def metric_values(row: dict[str, str], metric: str, n_episodes: int | None) -> dict[str, str]:
    mean = as_float(row.get(f"{metric}_mean"))
    std = as_float(row.get(f"{metric}_std"))
    ci95 = None
    if std is not None and n_episodes and n_episodes > 1:
        ci95 = 1.96 * std / math.sqrt(n_episodes)
    return {
        f"{metric}_mean": fmt(mean),
        f"{metric}_std": fmt(std),
        f"{metric}_ci95": fmt(ci95),
    }


def load_source(root: Path, block: str, tier: str, csv_path: str, node_count: int) -> list[dict[str, str]]:
    path = root / csv_path
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_node_count = as_float(row.get("node_count"))
            if row_node_count is not None and int(row_node_count) != node_count:
                continue
            n_episodes_float = as_float(row.get("n_episodes"))
            n_episodes = int(n_episodes_float) if n_episodes_float is not None else None
            out = {
                "evidence_block": block,
                "evidence_tier": tier,
                "source_csv": csv_path,
                "n_episodes": str(n_episodes or ""),
            }
            for column in CONTEXT_COLUMNS:
                out[column] = row.get(column, "")
            for metric in METRICS:
                out.update(metric_values(row, metric, n_episodes))
            rows.append(out)
    return rows


def write_readme(output_dir: Path, row_count: int) -> None:
    text = f"""# Statistical Stability Summary

Generated: {datetime.now(timezone.utc).isoformat()}

This directory contains compact mean/std/95% CI summaries extracted from archived `aggregate_metrics.csv` files for the manuscript and supplement.

Files:

- `statistical_stability_summary.csv`: normalized rows with evidence block, protocol, scenario context, `n_episodes`, metric means, metric standard deviations, and normal-approximation 95% confidence intervals.

Interpretation notes:

- The CI columns use `1.96 * std / sqrt(n_episodes)` and are intended as a concise reporting aid, not a replacement for paired statistical testing.
- Rows tagged `main` support current manuscript tables and figures.
- Rows tagged `main_boundary` support applicability-boundary claims.
- Rows tagged `supplement` are useful for reviewer-facing robustness evidence but should not replace the current main evidence chain without a separate promotion decision.
- Rows tagged `supplement_stress` are failure-boundary or extreme-regime checks; use them to bound claims rather than to advertise main performance.
- Rows tagged `supplement_sanity` are quick or one-seed checks; use them only to track trends while waiting for fuller multi-seed results.
- Total rows: {row_count}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for block, tier, csv_path in SOURCE_DEFS:
        rows.extend(load_source(root, block, tier, csv_path, args.node_count))

    fieldnames = (
        ["evidence_block", "evidence_tier", "source_csv", "n_episodes"]
        + list(CONTEXT_COLUMNS)
        + [name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_std", f"{metric}_ci95")]
    )
    with (output_dir / "statistical_stability_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_readme(output_dir, len(rows))


if __name__ == "__main__":
    main()
