from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


DEFAULT_SOURCE = Path("06_analysis/paper_tables/wang2025_focused_single_rf_20260709")
TREATMENT = "budgeted_collision_aware_isac"
CONTROLS = (
    "wang2025_isac_no_collab",
    "wang2025_comm_tables",
    "wang2025_isac_tables",
    "improved_rl_isac",
    "uniform_random",
)
METRICS = (
    "discovery_rate",
    "collision_penalized_discovery_rate",
    "collision_count",
    "lambda2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired deltas for the focused Wang2025 single-RF matrix.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--treatment", default=TREATMENT)
    parser.add_argument("--controls", default=",".join(CONTROLS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    rows = read_rows(source / "per_episode_summary.csv")
    treatment = str(args.treatment)
    controls = [part.strip() for part in str(args.controls).split(",") if part.strip()]
    deltas = build_deltas(rows, treatment, controls)
    summary = summarize_deltas(deltas)
    write_rows(source / "paired_deltas_vs_budgeted.csv", deltas)
    write_rows(source / "paired_delta_summary.csv", summary)
    print(
        {
            "source": str(source),
            "paired_delta_rows": len(deltas),
            "summary_rows": len(summary),
            "treatment": treatment,
            "controls": controls,
        }
    )


def build_deltas(rows: list[dict[str, Any]], treatment: str, controls: list[str]) -> list[dict[str, Any]]:
    key_fields = ("node_count", "rf_chains", "episode")
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        grouped.setdefault(key, {})[str(row.get("protocol", ""))] = row

    out: list[dict[str, Any]] = []
    for key, protocol_rows in sorted(grouped.items()):
        treat_row = protocol_rows.get(treatment)
        if treat_row is None:
            continue
        for control in controls:
            control_row = protocol_rows.get(control)
            if control_row is None:
                continue
            item: dict[str, Any] = {
                "node_count": key[0],
                "rf_chains": key[1],
                "episode": key[2],
                "treatment": treatment,
                "control": control,
            }
            for metric in METRICS:
                item[f"{metric}_treatment"] = treat_row.get(metric, "")
                item[f"{metric}_control"] = control_row.get(metric, "")
                item[f"{metric}_delta"] = to_float(treat_row.get(metric)) - to_float(control_row.get(metric))
            out.append(item)
    return out


def summarize_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["node_count"]), str(row["rf_chains"]), str(row["control"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (node_count, rf_chains, control), group in sorted(groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]), item[0][2])):
        item: dict[str, Any] = {
            "node_count": node_count,
            "rf_chains": rf_chains,
            "control": control,
            "pairs": len(group),
        }
        for metric in METRICS:
            values = [to_float(row.get(f"{metric}_delta")) for row in group]
            item[f"{metric}_delta_mean"] = mean(values)
            item[f"{metric}_delta_std"] = pstdev(values) if len(values) > 1 else 0.0
            if metric == "collision_count":
                item[f"{metric}_improvement_count"] = sum(1 for value in values if value < 0.0)
            else:
                item[f"{metric}_improvement_count"] = sum(1 for value in values if value > 0.0)
        out.append(item)
    return out


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


if __name__ == "__main__":
    main()
