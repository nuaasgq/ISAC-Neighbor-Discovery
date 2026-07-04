from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


EPISODE_METRICS = [
    "discovery_rate",
    "empty_scan_ratio",
    "mean_discovery_delay",
    "p90_discovery_delay",
    "p95_discovery_delay",
    "p99_discovery_delay",
    "lcc_ratio",
    "lambda2",
    "collision_count",
    "true_edges_seen",
    "discovered_edges",
    "moved_distance_mean_m",
]

SLOT_METRICS = [
    "true_edges",
    "true_edges_seen",
    "discovered_edges",
    "new_edges",
    "empty_scan_ratio",
    "collision_count",
    "lcc_ratio",
    "lambda2",
]

PLOT_METRICS = [
    ("empty_scan_ratio", "Empty scan ratio", "lower"),
    ("discovery_rate", "Discovery rate", "higher"),
    ("p95_discovery_delay", "P95 discovery delay (slots)", "lower"),
    ("p99_discovery_delay", "P99 discovery delay (slots)", "lower"),
    ("lcc_ratio", "Largest connected component ratio", "higher"),
    ("lambda2", "Algebraic connectivity lambda2", "higher"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate dynamic UAV neighbor-discovery runner outputs into paper-ready tables and figures.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more runner output directories or sweep roots containing per_episode_summary.csv.",
    )
    parser.add_argument(
        "--output",
        default="06_analysis",
        help="Analysis output root. Tables are written to <output>/tables and figures to <output>/figures.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plotting and write tables only.",
    )
    return parser.parse_args(argv)


def analyze(inputs: Iterable[str | Path], output_root: str | Path, no_plots: bool = False) -> dict:
    run_dirs = discover_run_dirs([Path(item) for item in inputs])
    if not run_dirs:
        raise FileNotFoundError("No per_episode_summary.csv files found under the requested input paths.")

    episode_rows, slot_rows, manifest_rows = load_rows(run_dirs)
    if not episode_rows:
        raise ValueError("Found run directories, but no episode rows were readable.")

    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    protocol_summary = summarize_episode_rows(episode_rows, ["protocol"])
    protocol_mobility_summary = summarize_episode_rows(episode_rows, ["protocol", "mobility_model"])
    slot_protocol_summary = summarize_slot_rows(slot_rows)

    write_csv(tables_dir / "protocol_summary.csv", protocol_summary)
    write_csv(tables_dir / "protocol_mobility_summary.csv", protocol_mobility_summary)
    write_csv(tables_dir / "slot_protocol_summary.csv", slot_protocol_summary)
    write_csv(tables_dir / "source_runs_manifest.csv", manifest_rows)

    figure_manifest = write_figures(protocol_summary, figures_dir, no_plots=no_plots)
    manifest = {
        "input_run_count": len(run_dirs),
        "episode_rows": len(episode_rows),
        "slot_rows": len(slot_rows),
        "tables": {
            "protocol_summary": str(tables_dir / "protocol_summary.csv"),
            "protocol_mobility_summary": str(tables_dir / "protocol_mobility_summary.csv"),
            "slot_protocol_summary": str(tables_dir / "slot_protocol_summary.csv"),
            "source_runs_manifest": str(tables_dir / "source_runs_manifest.csv"),
        },
        "figures": figure_manifest,
    }
    (output_root / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def discover_run_dirs(paths: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        candidates: list[Path] = []
        if path.is_file() and path.name == "per_episode_summary.csv":
            candidates = [path.parent]
        elif (path / "per_episode_summary.csv").exists():
            candidates = [path]
        elif path.exists():
            candidates = [candidate.parent for candidate in path.rglob("per_episode_summary.csv")]
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                run_dirs.append(resolved)
    return sorted(run_dirs, key=lambda item: str(item))


def load_rows(run_dirs: list[Path]) -> tuple[list[dict], list[dict], list[dict]]:
    episode_rows: list[dict] = []
    slot_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        per_episode = run_dir / "per_episode_summary.csv"
        per_slot = run_dir / "per_slot_metrics.csv"
        rows = [canonicalize_episode_row(row, run_id, run_dir) for row in read_csv(per_episode)]
        episode_rows.extend(rows)
        if per_slot.exists():
            slot_rows.extend(canonicalize_slot_row(row, run_id, run_dir) for row in read_csv(per_slot))
        manifest_rows.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "episode_rows": len(rows),
                "slot_rows": len(read_csv(per_slot)) if per_slot.exists() else 0,
            }
        )
    return episode_rows, slot_rows, manifest_rows


def canonicalize_episode_row(row: dict, run_id: str, run_dir: Path) -> dict:
    row = dict(row)
    row["run_id"] = run_id
    row["run_dir"] = str(run_dir)
    row.setdefault("mobility_model", "unknown")
    if "mean_discovery_delay" not in row and "mean_delay_censored" in row:
        row["mean_discovery_delay"] = row["mean_delay_censored"]
    if "p95_discovery_delay" not in row and "p95_delay_censored" in row:
        row["p95_discovery_delay"] = row["p95_delay_censored"]
    if "p90_discovery_delay" not in row and "p90_delay_censored" in row:
        row["p90_discovery_delay"] = row["p90_delay_censored"]
    if "p99_discovery_delay" not in row and "p99_delay_censored" in row:
        row["p99_discovery_delay"] = row["p99_delay_censored"]
    if "finite_time_discovery_rate" not in row and "discovery_rate" in row:
        row["finite_time_discovery_rate"] = row["discovery_rate"]
    return row


def canonicalize_slot_row(row: dict, run_id: str, run_dir: Path) -> dict:
    row = dict(row)
    row["run_id"] = run_id
    row["run_dir"] = str(run_dir)
    row.setdefault("mobility_model", "unknown")
    return row


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_episode_rows(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups = group_rows(rows, group_keys)
    summary_rows: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        output = {name: value for name, value in zip(group_keys, key)}
        output["episode_count"] = len(group)
        output["run_count"] = len({row.get("run_id", "") for row in group})
        for metric in EPISODE_METRICS:
            add_metric_summary(output, metric, [to_float(row.get(metric)) for row in group])
        summary_rows.append(output)
    return summary_rows


def summarize_slot_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    groups = group_rows(rows, ["protocol", "slot"])
    summary_rows: list[dict] = []
    for key, group in sorted(groups.items(), key=lambda item: (item[0][0], int(float(item[0][1])))):
        output = {"protocol": key[0], "slot": key[1], "sample_count": len(group)}
        for metric in SLOT_METRICS:
            add_metric_summary(output, metric, [to_float(row.get(metric)) for row in group])
        summary_rows.append(output)
    return summary_rows


def group_rows(rows: list[dict], keys: list[str]) -> dict[tuple[str, ...], list[dict]]:
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    return groups


def add_metric_summary(output: dict, metric: str, raw_values: list[float | None]) -> None:
    values = [value for value in raw_values if value is not None and math.isfinite(value)]
    output[f"{metric}_n"] = len(values)
    if not values:
        output[f"{metric}_mean"] = ""
        output[f"{metric}_std"] = ""
        output[f"{metric}_median"] = ""
        output[f"{metric}_min"] = ""
        output[f"{metric}_max"] = ""
        output[f"{metric}_ci95"] = ""
        return
    output[f"{metric}_mean"] = format_float(statistics.fmean(values))
    output[f"{metric}_std"] = format_float(statistics.stdev(values) if len(values) > 1 else 0.0)
    output[f"{metric}_median"] = format_float(statistics.median(values))
    output[f"{metric}_min"] = format_float(min(values))
    output[f"{metric}_max"] = format_float(max(values))
    output[f"{metric}_ci95"] = format_float(1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0)


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_float(value: float) -> str:
    return f"{value:.6g}"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_figures(summary_rows: list[dict], figures_dir: Path, no_plots: bool = False) -> list[dict]:
    manifest: list[dict] = []
    if no_plots:
        for metric, title, _direction in PLOT_METRICS:
            manifest.append({"metric": metric, "status": "skipped", "reason": "--no-plots"})
        write_figure_manifest(figures_dir, manifest)
        return manifest

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional environment dependency.
        for metric, _title, _direction in PLOT_METRICS:
            manifest.append({"metric": metric, "status": "skipped", "reason": f"matplotlib unavailable: {exc}"})
        write_figure_manifest(figures_dir, manifest)
        return manifest

    protocols = [row["protocol"] for row in summary_rows]
    for metric, title, direction in PLOT_METRICS:
        mean_key = f"{metric}_mean"
        ci_key = f"{metric}_ci95"
        values = [to_float(row.get(mean_key)) for row in summary_rows]
        ci95 = [to_float(row.get(ci_key)) or 0.0 for row in summary_rows]
        if not protocols or any(value is None for value in values):
            manifest.append({"metric": metric, "status": "skipped", "reason": f"missing {mean_key}"})
            continue
        figure_path = figures_dir / f"protocol_{metric}.png"
        colors = ["#4477AA" if direction == "higher" else "#CC6677" for _ in protocols]
        width = max(6.0, 0.75 * len(protocols) + 2.0)
        plt.figure(figsize=(width, 4.2))
        plt.bar(protocols, [float(value) for value in values], yerr=ci95, capsize=4, color=colors)
        plt.ylabel(title)
        plt.xlabel("Protocol")
        plt.xticks(rotation=20, ha="right")
        plt.grid(axis="y", alpha=0.25, linewidth=0.8)
        plt.tight_layout()
        plt.savefig(figure_path, dpi=220)
        plt.close()
        manifest.append({"metric": metric, "status": "generated", "path": str(figure_path)})
    write_figure_manifest(figures_dir, manifest)
    return manifest


def write_figure_manifest(figures_dir: Path, manifest: list[dict]) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = analyze(args.inputs, args.output, no_plots=args.no_plots)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
