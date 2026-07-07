"""Build a tracked raw-bundle inventory for the Phase10 manuscript evidence.

The Phase10 method trace points to raw evaluation manifests, training manifests,
and compact model checkpoints under ``05_simulation/results_raw``. Those paths
are intentionally ignored by default, so this script creates an explicit
archive/index layer: each referenced raw file is checked for local existence,
Git tracking, size, and SHA256 hash.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TRACE_CSV = Path("06_analysis/paper_tables/marl/p10_final_method_manifest_trace/phase10_method_manifest_trace.csv")
OUT_DIR = Path("06_analysis/paper_tables/marl/p10_raw_bundle_archive")
REPORT = Path("06_analysis/phase10_raw_bundle_archive_report_20260707.md")


@dataclass(frozen=True)
class RawPath:
    artifact_type: str
    path: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_tracked_paths(paths: Iterable[Path]) -> set[str]:
    normalized = [path.as_posix() for path in paths]
    if not normalized:
        return set()
    try:
        result = subprocess.run(
            ["git", "ls-files", "--", *normalized],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip().replace("\\", "/").lower() for line in result.stdout.splitlines() if line.strip()}


def split_paths(value: str) -> list[Path]:
    paths: list[Path] = []
    for raw in str(value or "").split(";"):
        raw = raw.strip()
        if raw:
            paths.append(Path(raw.replace("\\", "/")))
    return paths


def collect_raw_paths(trace_rows: list[dict[str, str]]) -> list[RawPath]:
    collected: list[RawPath] = []
    for artifact_type, fields in (
        ("raw_eval_manifest", ("raw_eval_manifests",)),
        ("training_manifest", ("train_manifests",)),
        ("checkpoint", ("checkpoints",)),
    ):
        seen: set[str] = set()
        for row in trace_rows:
            for field in fields:
                for path in split_paths(row.get(field, "")):
                    key = path.as_posix().lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    collected.append(RawPath(artifact_type, path))
    return sorted(collected, key=lambda item: (item.artifact_type, item.path.as_posix().lower()))


def build_rows(raw_paths: list[RawPath]) -> list[dict[str, object]]:
    tracked = git_tracked_paths([item.path for item in raw_paths])
    rows: list[dict[str, object]] = []
    for item in raw_paths:
        exists = item.path.exists()
        rows.append(
            {
                "artifact_type": item.artifact_type,
                "path": item.path.as_posix(),
                "exists": exists,
                "git_tracked": item.path.as_posix().lower() in tracked,
                "size_bytes": item.path.stat().st_size if exists else "",
                "sha256": sha256_file(item.path) if exists else "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, object]], manifest: dict[str, object]) -> None:
    type_counts: dict[str, int] = {}
    type_bytes: dict[str, int] = {}
    for row in rows:
        artifact_type = str(row["artifact_type"])
        type_counts[artifact_type] = type_counts.get(artifact_type, 0) + 1
        type_bytes[artifact_type] = type_bytes.get(artifact_type, 0) + int(row["size_bytes"] or 0)

    lines = [
        "# Phase10 Raw Bundle Archive Report - 2026-07-07",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: archive",
        "- Verification Status: ANALYZED",
        "- Scope: raw files referenced by the Phase10 final method trace",
        "",
        "## Summary",
        "",
        f"- Raw files indexed: {manifest['raw_file_count']}.",
        f"- Existing files: {manifest['existing_file_count']}/{manifest['raw_file_count']}.",
        f"- Git-tracked files: {manifest['git_tracked_file_count']}/{manifest['raw_file_count']}.",
        f"- Total bytes: {manifest['total_size_bytes']}.",
        f"- Archive complete: {manifest['archive_complete']}.",
        "",
        "## Type Counts",
        "",
        "| Artifact type | Files | Bytes |",
        "|---|---:|---:|",
    ]
    for artifact_type in sorted(type_counts):
        lines.append(f"| {artifact_type} | {type_counts[artifact_type]} | {type_bytes[artifact_type]} |")

    missing = [row for row in rows if not row["exists"]]
    untracked = [row for row in rows if not row["git_tracked"]]
    lines.extend(["", "## Missing Or Untracked Files", ""])
    if not missing and not untracked:
        lines.append("None. All referenced raw files exist locally and are tracked by Git.")
    else:
        for row in missing:
            lines.append(f"- Missing: `{row['path']}`")
        for row in untracked:
            lines.append(f"- Untracked: `{row['path']}`")

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            f"- `{(OUT_DIR / 'phase10_raw_bundle_inventory.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'manifest.json').as_posix()}`",
            f"- `{(OUT_DIR / 'README.md').as_posix()}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    trace_rows = read_csv(TRACE_CSV)
    raw_paths = collect_raw_paths(trace_rows)
    rows = build_rows(raw_paths)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_csv = OUT_DIR / "phase10_raw_bundle_inventory.csv"
    manifest_json = OUT_DIR / "manifest.json"
    readme_md = OUT_DIR / "README.md"

    write_csv(
        inventory_csv,
        rows,
        ["artifact_type", "path", "exists", "git_tracked", "size_bytes", "sha256"],
    )

    existing_file_count = sum(1 for row in rows if row["exists"])
    git_tracked_file_count = sum(1 for row in rows if row["git_tracked"])
    total_size_bytes = sum(int(row["size_bytes"] or 0) for row in rows)
    output_files = [inventory_csv, readme_md, REPORT]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "raw files referenced by the Phase10 final method trace",
        "source_trace_csv": TRACE_CSV.as_posix(),
        "raw_file_count": len(rows),
        "existing_file_count": existing_file_count,
        "git_tracked_file_count": git_tracked_file_count,
        "total_size_bytes": total_size_bytes,
        "archive_complete": bool(rows) and existing_file_count == len(rows) and git_tracked_file_count == len(rows),
        "outputs": {
            "inventory_csv": inventory_csv.as_posix(),
            "report": REPORT.as_posix(),
            "readme": readme_md.as_posix(),
        },
        "raw_files": [
            {
                "artifact_type": row["artifact_type"],
                "path": row["path"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
            }
            for row in rows
        ],
    }

    write_report(rows, manifest)
    readme_md.write_text(
        "\n".join(
            [
                "# Phase10 Raw Bundle Archive",
                "",
                "Generated by `06_analysis/scripts/build_phase10_raw_bundle_manifest.py`.",
                "The inventory lists every raw evaluation manifest, training manifest, and checkpoint referenced by the Phase10 final method trace.",
                "",
                "- `phase10_raw_bundle_inventory.csv`: raw file path, type, size, Git tracking status, and SHA256.",
                "- `manifest.json`: machine-readable archive summary and raw file hashes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_files = [inventory_csv, readme_md, REPORT]
    manifest["output_hashes"] = {path.as_posix(): sha256_file(path) for path in output_files}
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "created_at_utc": manifest["created_at_utc"],
                "raw_file_count": manifest["raw_file_count"],
                "existing_file_count": existing_file_count,
                "git_tracked_file_count": git_tracked_file_count,
                "archive_complete": manifest["archive_complete"],
                "report": REPORT.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
