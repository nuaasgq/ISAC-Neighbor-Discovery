from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCES = (
    (
        "round7_mobility_core",
        "06_analysis/paper_tables/round7_n100_multimobility_600slot/aggregate_metrics.csv",
    ),
    (
        "round8_missing_baselines",
        "06_analysis/paper_tables/round8_n100_multimobility_missing_baselines_600slot/aggregate_metrics.csv",
    ),
)

SUMMARY_COLUMNS = (
    "source_block",
    "protocol",
    "mobility_model",
    "beamwidth_deg",
    "node_count",
    "n_episodes",
    "discovery_rate_mean",
    "discovery_rate_std",
    "empty_scan_ratio_mean",
    "empty_scan_ratio_std",
    "lambda2_mean",
    "lambda2_std",
    "collision_count_mean",
    "collision_count_std",
    "collision_penalized_discovery_rate_mean",
    "collision_penalized_discovery_rate_std",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge N=100 mobility baseline aggregate tables.")
    parser.add_argument("--output", default="06_analysis/paper_tables/round8_n100_multimobility_full_baseline")
    return parser.parse_args()


def read_rows(root: Path, source_block: str, rel_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    path = root / rel_path
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out = {"source_block": source_block}
            for column in SUMMARY_COLUMNS:
                if column == "source_block":
                    continue
                out[column] = row.get(column, "")
            rows.append(out)
    return rows


def write_readme(output_dir: Path, row_count: int) -> None:
    text = f"""# N=100 Mobility Full-Baseline Table

Generated: {datetime.now(timezone.utc).isoformat()}

This table merges:

- `round7_n100_multimobility_600slot`: uniform random, improved no-ISAC, one-slot delayed ISAC, and full ISAC results.
- `round8_n100_multimobility_missing_baselines_600slot`: SkyOrbs-like skip scan and vanilla learned policy without ISAC.

The output is intended for reviewer-facing baseline-completeness checks under the mobility-transfer setting. It should be interpreted together with the mobility-boundary wording in the manuscript: Gauss-Markov and random-walk regimes are stronger for the proposed method, while random-direction and random-waypoint remain stress regimes.

Files:

- `combined_aggregate_metrics.csv`: compact merged mean/std table.

Total rows: {row_count}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for source_block, rel_path in DEFAULT_SOURCES:
        rows.extend(read_rows(root, source_block, rel_path))

    rows.sort(key=lambda row: (row["mobility_model"], float(row["beamwidth_deg"] or 0), row["protocol"]))
    with (output_dir / "combined_aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_readme(output_dir, len(rows))


if __name__ == "__main__":
    main()
