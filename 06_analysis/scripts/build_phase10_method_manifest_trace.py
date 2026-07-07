"""Trace final Phase10 method rows back to analysis and raw manifests."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_FINAL_CSV = Path(
    "06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/marl_method_comparison.csv"
)
DEFAULT_FINAL_MANIFEST = Path(
    "06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/manifest.json"
)
DEFAULT_OUTPUT = Path("06_analysis/paper_tables/marl/p10_final_method_manifest_trace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build method-to-manifest trace table for Phase10 final evidence.")
    parser.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    parser.add_argument("--final-manifest", type=Path, default=DEFAULT_FINAL_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_repo_path(raw: str | None) -> str:
    if not raw:
        return ""
    text = str(raw).replace("\\", "/")
    marker = "05_simulation/"
    if marker in text:
        return text[text.index(marker) :]
    marker = "06_analysis/"
    if marker in text:
        return text[text.index(marker) :]
    return text


def beamwidth_from_manifest(payload: dict) -> float:
    if payload.get("beamwidth_deg") not in (None, ""):
        return float(payload["beamwidth_deg"])
    az = int(payload.get("azimuth_cells", 0) or 0)
    if az:
        return 360.0 / az
    return -1.0


def method_matches(raw_manifest: dict, method: str, methods_in_summary: set[str]) -> bool:
    if len(methods_in_summary) == 1 and method in methods_in_summary:
        return True
    raw_method = str(raw_manifest.get("method", "") or "")
    if raw_method == method:
        return True
    output = normalize_repo_path(raw_manifest.get("output", "")).lower()
    return method.lower() in output


def find_source_summaries(final_rows: list[dict[str, str]], final_manifest: dict) -> list[dict]:
    summaries: list[dict] = []
    for summary_path_raw in final_manifest.get("combined_summary", []):
        summary_path = Path(str(summary_path_raw).replace("\\", "/"))
        rows = read_csv(summary_path)
        summaries.append(
            {
                "summary_csv": summary_path,
                "summary_rows": rows,
                "analysis_manifest": summary_path.parent / "manifest.json",
                "methods": {row.get("method", "") for row in rows},
            }
        )
    return summaries


def load_raw_eval_manifests(analysis_manifest_path: Path) -> list[tuple[str, Path, dict]]:
    if not analysis_manifest_path.exists():
        return []
    payload = read_json(analysis_manifest_path)
    loaded: list[tuple[str, Path, dict]] = []
    for run_dir_raw in payload.get("run_dirs", []):
        run_dir = Path(str(run_dir_raw).replace("\\", "/"))
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            loaded.append((str(run_dir_raw), manifest_path, read_json(manifest_path)))
    return loaded


def derive_train_manifest(checkpoint_raw: str) -> str:
    checkpoint = normalize_repo_path(checkpoint_raw)
    if not checkpoint:
        return ""
    candidate = Path(checkpoint).parent / "manifest.json"
    return candidate.as_posix()


def exists_flag(path_text: str) -> bool:
    return bool(path_text) and Path(path_text).exists()


def compact_paths(paths: list[str]) -> str:
    unique = []
    seen = set()
    for path in paths:
        if path and path not in seen:
            unique.append(path)
            seen.add(path)
    return ";".join(unique)


def build_trace_rows(final_rows: list[dict[str, str]], summaries: list[dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    raw_cache: dict[Path, list[tuple[str, Path, dict]]] = {}
    for final_row in final_rows:
        method = final_row["method"]
        beam = float(final_row["beamwidth_deg"])
        matching_summary = None
        for summary in summaries:
            found = [
                row
                for row in summary["summary_rows"]
                if row.get("method") == method and abs(float(row.get("beamwidth_deg", -999)) - beam) < 1e-6
            ]
            if found:
                matching_summary = summary
                break
        if matching_summary is None:
            rows.append(
                {
                    **base_fields(final_row),
                    "trace_status": "missing_source_summary",
                    "analysis_summary_csv": "",
                    "analysis_manifest": "",
                    "raw_eval_manifests": "",
                    "train_manifests": "",
                    "checkpoints": "",
                    "config_paths": "",
                    "eval_run_count": "0",
                    "training_trace": "",
                }
            )
            continue

        analysis_manifest = matching_summary["analysis_manifest"]
        raw_entries = raw_cache.setdefault(analysis_manifest, load_raw_eval_manifests(analysis_manifest))
        selected = []
        for _run_dir, manifest_path, payload in raw_entries:
            if abs(beamwidth_from_manifest(payload) - beam) > 1e-6:
                continue
            if method_matches(payload, method, matching_summary["methods"]):
                selected.append((manifest_path, payload))

        raw_eval_manifests = [path.as_posix() for path, _payload in selected]
        checkpoints = [normalize_repo_path(payload.get("checkpoint", "")) for _path, payload in selected]
        train_manifests = [derive_train_manifest(path) for path in checkpoints if path]
        config_paths = [normalize_repo_path(payload.get("config", "")) for _path, payload in selected]
        baseline = final_row.get("train_algorithm") == "protocol_baseline"
        trace_status = "complete"
        if not raw_eval_manifests:
            trace_status = "missing_raw_eval_manifest"
        elif not baseline and not any(exists_flag(path) for path in train_manifests):
            trace_status = "missing_train_manifest"

        rows.append(
            {
                **base_fields(final_row),
                "trace_status": trace_status,
                "analysis_summary_csv": matching_summary["summary_csv"].as_posix(),
                "analysis_manifest": analysis_manifest.as_posix(),
                "raw_eval_manifests": compact_paths(raw_eval_manifests),
                "train_manifests": compact_paths(train_manifests),
                "checkpoints": compact_paths(checkpoints),
                "config_paths": compact_paths(config_paths),
                "eval_run_count": str(len(raw_eval_manifests)),
                "training_trace": "protocol_baseline_no_training" if baseline else "checkpoint_parent_manifest",
            }
        )
    return rows


def base_fields(row: dict[str, str]) -> dict[str, str]:
    keep = [
        "method",
        "method_label",
        "train_algorithm",
        "train_network",
        "train_reward_version",
        "env_protocol",
        "phase",
        "node_count",
        "beamwidth_deg",
        "slots_per_episode",
        "episodes",
        "run_n",
        "discovery_rate_mean",
        "collision_penalized_discovery_rate_mean",
        "lambda2_mean",
        "collision_count_mean",
    ]
    return {key: row.get(key, "") for key in keep}


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output: Path, manifest: dict) -> None:
    lines = [
        "# Phase10 Final Method Manifest Trace",
        "",
        f"Created: {manifest['created_at_utc']}",
        "",
        "This directory maps each final Phase10 method/beam row back to:",
        "",
        "- the final manuscript comparison CSV row,",
        "- the intermediate analysis summary CSV and manifest,",
        "- the raw evaluation manifest(s),",
        "- and, for learned policies, the checkpoint-parent training manifest(s).",
        "",
        "Protocol baselines have no training manifest and are marked `protocol_baseline_no_training`.",
        "",
        "Files:",
        "",
        "- `phase10_method_manifest_trace.csv`",
        "- `manifest.json`",
        "",
        f"Rows: {manifest['rows']}",
        f"Complete rows: {manifest['complete_rows']}",
        f"Incomplete rows: {manifest['incomplete_rows']}",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    final_rows = read_csv(args.final_csv)
    final_manifest = read_json(args.final_manifest)
    summaries = find_source_summaries(final_rows, final_manifest)
    trace_rows = build_trace_rows(final_rows, summaries)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "phase10_method_manifest_trace.csv"
    write_csv(csv_path, trace_rows)

    complete = sum(1 for row in trace_rows if row["trace_status"] == "complete")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase10 final N=100 B=10/B=15 method rows to analysis/raw/train manifests",
        "final_csv": args.final_csv.as_posix(),
        "final_manifest": args.final_manifest.as_posix(),
        "output_csv": csv_path.as_posix(),
        "rows": len(trace_rows),
        "complete_rows": complete,
        "incomplete_rows": len(trace_rows) - complete,
        "status_counts": {
            status: sum(1 for row in trace_rows if row["trace_status"] == status)
            for status in sorted({row["trace_status"] for row in trace_rows})
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["incomplete_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
