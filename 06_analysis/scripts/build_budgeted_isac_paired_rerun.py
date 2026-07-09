from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = (
    ROOT
    / "05_simulation"
    / "results_raw"
    / "round_budgeted_isac_paired_rerun"
    / "B10_N100_3000slot_3ep_seed2026071061"
)
OUT = ROOT / "06_analysis" / "paper_tables" / "marl" / "budgeted_isac_paired_rerun_b10_n100"

LABELS = {
    "budgeted_collision_aware_isac": "Budgeted ISAC",
    "wang2025_isac_no_collab": "Wang ISAC",
    "collision_aware_isac": "Collision-aware ISAC",
    "uniform_random": "Uniform random",
}

METRICS = [
    "discovery_rate",
    "collision_count",
    "collision_penalized_discovery_rate",
    "lambda2",
    "empty_scan_ratio",
    "scan_actions_per_discovery_censored",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episode_rows = collect_episode_rows()
    summary_rows = summarize(episode_rows)
    delta_rows = paired_deltas(episode_rows)
    write_csv(OUT / "paired_rerun_episode_rows.csv", episode_rows)
    write_csv(OUT / "paired_rerun_summary.csv", summary_rows)
    write_csv(OUT / "paired_rerun_deltas.csv", delta_rows)
    manifest = {
        "created_at": "2026-07-09",
        "raw_root": str(RAW.relative_to(ROOT)),
        "output": str(OUT.relative_to(ROOT)),
        "scenario": "B=10, N=100, 3000 slots, 3 paired episodes, single RF",
        "seed": 2026071061,
        "protocols": sorted(LABELS),
        "episode_rows": len(episode_rows),
        "summary_rows": len(summary_rows),
        "delta_rows": len(delta_rows),
        "files": [
            "paired_rerun_episode_rows.csv",
            "paired_rerun_summary.csv",
            "paired_rerun_deltas.csv",
            "manifest.json",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_episode_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for protocol, label in LABELS.items():
        path = RAW / protocol / "eval_episode_metrics.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            row = dict(row)
            row["protocol"] = protocol
            row["method"] = label
            row["beamwidth_deg"] = "10"
            row["node_count"] = "100"
            row["paired_seed"] = "2026071061"
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str | float | int]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    out = []
    for method, group in sorted(groups.items()):
        item: dict[str, str | float | int] = {"method": method, "episodes": len(group)}
        for metric in METRICS:
            item[f"{metric}_mean"] = avg(group, metric)
            item[f"{metric}_std"] = stdev(group, metric)
        out.append(item)
    return out


def paired_deltas(rows: list[dict[str, str]]) -> list[dict[str, str | float | int]]:
    by_protocol: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        by_protocol.setdefault(row["protocol"], {})[int(row["eval_episode"])] = row
    treatment = "budgeted_collision_aware_isac"
    controls = ["wang2025_isac_no_collab", "collision_aware_isac", "uniform_random"]
    out: list[dict[str, str | float | int]] = []
    for control in controls:
        for episode, treatment_row in sorted(by_protocol.get(treatment, {}).items()):
            control_row = by_protocol.get(control, {}).get(episode)
            if control_row is None:
                continue
            for metric in METRICS:
                out.append(
                    {
                        "treatment": LABELS[treatment],
                        "control": LABELS[control],
                        "eval_episode": episode,
                        "metric": metric,
                        "treatment_value": to_float(treatment_row.get(metric)),
                        "control_value": to_float(control_row.get(metric)),
                        "delta": to_float(treatment_row.get(metric)) - to_float(control_row.get(metric)),
                    }
                )
    summary_rows = []
    grouped: dict[tuple[str, str], list[dict[str, str | float | int]]] = {}
    for row in out:
        grouped.setdefault((str(row["control"]), str(row["metric"])), []).append(row)
    for (control, metric), group in grouped.items():
        deltas = [float(row["delta"]) for row in group]
        summary_rows.append(
            {
                "treatment": LABELS[treatment],
                "control": control,
                "eval_episode": "mean",
                "metric": metric,
                "treatment_value": "",
                "control_value": "",
                "delta": statistics.mean(deltas),
            }
        )
    return out + summary_rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def avg(rows: list[dict[str, str]], key: str) -> float:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    return statistics.mean(vals) if vals else float("nan")


def stdev(rows: list[dict[str, str]], key: str) -> float:
    vals = [to_float(row.get(key)) for row in rows]
    vals = [value for value in vals if math.isfinite(value)]
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
