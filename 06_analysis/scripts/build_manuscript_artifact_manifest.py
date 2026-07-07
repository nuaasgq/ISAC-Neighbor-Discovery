"""Build a hash manifest for the current manuscript-facing artifacts.

The manifest is intentionally tied to the LaTeX sources and the Phase10 main
evidence chain. It does not walk the whole experiment archive because older
campaigns are retained for boundary evidence and should not be mixed into the
primary manuscript claim chain by accident.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")

DEFAULT_TEX_FILES = [
    Path("07_paper/ieee_twc_isac_nd/main.tex"),
    Path("07_paper/ieee_twc_isac_nd/supplement.tex"),
]

DEFAULT_DIRECT_ARTIFACTS = [
    Path("07_paper/ieee_twc_isac_nd/main.tex"),
    Path("07_paper/ieee_twc_isac_nd/supplement.tex"),
    Path("07_paper/ieee_twc_isac_nd/references.bib"),
    Path("07_paper/ieee_twc_isac_nd/README.md"),
    Path("06_analysis/scripts/build_manuscript_artifact_manifest.py"),
    Path("06_analysis/scripts/build_phase10_method_manifest_trace.py"),
    Path("06_analysis/scripts/build_phase10_independent_rerun_validation.py"),
    Path("06_analysis/scripts/build_phase10_learned_component_ablation.py"),
    Path("06_analysis/scripts/build_phase10_paired_significance.py"),
    Path("06_analysis/scripts/build_phase10_raw_bundle_manifest.py"),
    Path("06_analysis/scripts/build_research_goal_coverage_audit.py"),
    Path("06_analysis/scripts/build_submission_claim_strength_audit.py"),
    Path("06_analysis/figure_table_provenance_audit_20260707.md"),
    Path("06_analysis/claim_evidence_audit_20260707.md"),
    Path("06_analysis/citation_integrity_audit_20260707.md"),
    Path("06_analysis/phase10_statistical_validation_report_20260707.md"),
    Path("06_analysis/phase10_independent_rerun_validation_20260707.md"),
    Path("06_analysis/phase10_learned_component_ablation_report_20260707.md"),
    Path("06_analysis/phase10_paired_significance_report_20260707.md"),
    Path("06_analysis/phase10_raw_bundle_archive_report_20260707.md"),
    Path("06_analysis/range_abstraction_theory_note_20260707.md"),
    Path("06_analysis/research_goal_coverage_audit_20260707.md"),
    Path("06_analysis/submission_claim_strength_audit_20260707.md"),
    Path("06_analysis/submission_readiness_review_20260707.md"),
    Path("06_analysis/paper_tables/research_goal_coverage_audit/requirement_coverage.csv"),
    Path("06_analysis/paper_tables/research_goal_coverage_audit/evidence_inventory.csv"),
    Path("06_analysis/paper_tables/research_goal_coverage_audit/claim_risk_register.csv"),
    Path("06_analysis/paper_tables/research_goal_coverage_audit/raw_bundle_trace_status.csv"),
    Path("06_analysis/paper_tables/research_goal_coverage_audit/manifest.json"),
    Path("06_analysis/paper_tables/research_goal_coverage_audit/README.md"),
    Path("06_analysis/paper_tables/submission_claim_strength_audit/claim_boundary_checks.csv"),
    Path("06_analysis/paper_tables/submission_claim_strength_audit/risk_phrase_hits.csv"),
    Path("06_analysis/paper_tables/submission_claim_strength_audit/manifest.json"),
    Path("06_analysis/paper_tables/submission_claim_strength_audit/README.md"),
    Path("06_analysis/paper_tables/marl/p10_raw_bundle_archive/phase10_raw_bundle_inventory.csv"),
    Path("06_analysis/paper_tables/marl/p10_raw_bundle_archive/manifest.json"),
    Path("06_analysis/paper_tables/marl/p10_raw_bundle_archive/README.md"),
    Path("06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation/rerun_metric_summary.csv"),
    Path("06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation/rerun_vs_original_comparison.csv"),
    Path("06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation/rerun_source_file_hashes.csv"),
    Path("06_analysis/paper_tables/marl/p10_independent_rerun_gate31_b10_validation/manifest.json"),
    Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_metric_summary.csv"),
    Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_vs_trained_full.csv"),
    Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_run_index.csv"),
    Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_source_file_hashes.csv"),
    Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/manifest.json"),
    Path("06_analysis/figures/marl/p10_learned_component_ablation_b10_3ep/learned_ablation_b10_discovery_efficiency.png"),
    Path("06_analysis/figures/marl/p10_learned_component_ablation_b10_3ep/learned_ablation_b10_collision_tradeoff.png"),
    Path("06_analysis/paper_tables/marl/p10_paired_significance_primary/primary_paired_sign_tests.csv"),
    Path("06_analysis/paper_tables/marl/p10_paired_significance_primary/primary_paired_delta_values.csv"),
    Path("06_analysis/paper_tables/marl/p10_paired_significance_primary/source_file_hashes.csv"),
    Path("06_analysis/paper_tables/marl/p10_paired_significance_primary/manifest.json"),
    Path("06_analysis/figures/marl/p10_paired_significance_primary/phase10_primary_discovery_paired_deltas.png"),
    Path("06_analysis/figures/marl/p10_paired_significance_primary/phase10_empty_scan_paired_deltas.png"),
    Path("06_analysis/paper_tables/marl/p10_final_method_manifest_trace/phase10_method_manifest_trace.csv"),
    Path("06_analysis/paper_tables/marl/p10_final_method_manifest_trace/manifest.json"),
    Path("06_analysis/paper_tables/marl/p10_final_method_manifest_trace/README.md"),
    Path("06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/marl_method_comparison.csv"),
    Path("06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/manifest.json"),
    Path("06_analysis/paper_tables/marl/p10_gate_family_v2_v3_v4_tradeoff_comparison/seed_tradeoff_core_metrics.csv"),
    Path("06_analysis/paper_tables/marl/p10_gate_family_v2_v3_v4_tradeoff_comparison/seed_tradeoff_method_comparison.csv"),
    Path("06_analysis/paper_tables/marl/p10_gate_family_v2_v3_v4_tradeoff_comparison/manifest.json"),
    Path("06_analysis/paper_tables/marl/p10_gate_training_3seed_100ep_step_curves/marl_step_rewards.csv"),
    Path("06_analysis/paper_tables/marl/p10_gate_training_3seed_100ep_step_curves/marl_episode_metrics.csv"),
    Path("06_analysis/paper_tables/marl/p10_gate_training_3seed_100ep_step_curves/marl_eval_episode_metrics.csv"),
    Path("06_analysis/paper_tables/marl/p10_gate_training_3seed_100ep_step_curves/marl_resource_log.csv"),
    Path("06_analysis/paper_tables/marl/p10_gate_training_3seed_100ep_step_curves/manifest.json"),
]


@dataclass(frozen=True)
class Artifact:
    role: str
    source: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript artifact hash manifest.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--csv", type=Path, default=Path("06_analysis/manuscript_artifact_manifest_20260707.csv"))
    parser.add_argument("--json", type=Path, default=Path("06_analysis/manuscript_artifact_manifest_20260707.json"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_figures(tex_file: Path) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for line_no, line in enumerate(tex_file.read_text(encoding="utf-8").splitlines(), start=1):
        for match in INCLUDE_RE.finditer(line):
            raw_path = match.group(1)
            resolved = (tex_file.parent / raw_path).resolve()
            artifacts.append(
                Artifact(
                    role="manuscript_figure",
                    source=f"{tex_file.as_posix()}:{line_no}",
                    path=resolved,
                )
            )
    return artifacts


def collect_manifest_references(manifest_path: Path) -> list[Artifact]:
    """Collect directly referenced source files from selected JSON manifests."""
    if not manifest_path.exists() or manifest_path.suffix.lower() != ".json":
        return []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return []

    candidates: list[tuple[str, str]] = []

    def walk(value, label: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{label}.{key}" if label else str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{label}[{index}]")
        elif isinstance(value, str):
            normalized = value.replace("\\", "/")
            if normalized.endswith((".csv", ".json", ".md", ".yaml", ".yml")):
                candidates.append((label, value))

    walk(payload, "")
    artifacts: list[Artifact] = []
    for label, raw_path in candidates:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            rerun_dir = payload.get("rerun_dir")
            if label.startswith("manifest.files[") and rerun_dir and candidate.parent == Path("."):
                candidate = Path(str(rerun_dir).replace("\\", "/")) / candidate
            else:
                candidate = Path(raw_path.replace("\\", "/"))
        artifacts.append(
            Artifact(
                role="manifest_reference",
                source=f"{manifest_path.as_posix()}:{label}",
                path=candidate,
            )
        )
    return artifacts


def normalize_path(repo_root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def unique_artifacts(artifacts: list[Artifact], repo_root: Path) -> list[Artifact]:
    seen: set[tuple[str, str]] = set()
    output: list[Artifact] = []
    for artifact in artifacts:
        resolved = normalize_path(repo_root, artifact.path)
        key = (artifact.role, str(resolved).lower())
        if key in seen:
            continue
        seen.add(key)
        output.append(Artifact(artifact.role, artifact.source, resolved))
    return output


def relative_to_repo(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifacts: list[Artifact] = []

    for tex_file in DEFAULT_TEX_FILES:
        artifacts.extend(parse_figures(normalize_path(repo_root, tex_file)))
    for path in DEFAULT_DIRECT_ARTIFACTS:
        artifacts.append(Artifact("direct_evidence_artifact", "default_main_chain", path))

    for path in DEFAULT_DIRECT_ARTIFACTS:
        if path.name == "manifest.json":
            artifacts.extend(collect_manifest_references(normalize_path(repo_root, path)))

    artifacts = unique_artifacts(artifacts, repo_root)
    rows: list[dict[str, str | int | bool]] = []
    missing = 0
    for artifact in sorted(artifacts, key=lambda item: (item.role, str(item.path).lower())):
        exists = artifact.path.exists()
        if not exists:
            missing += 1
        rows.append(
            {
                "role": artifact.role,
                "source": artifact.source,
                "path": relative_to_repo(repo_root, artifact.path),
                "exists": exists,
                "size_bytes": artifact.path.stat().st_size if exists else "",
                "sha256": sha256_file(artifact.path) if exists else "",
            }
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "source", "path", "exists", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "current manuscript-facing LaTeX figures plus Phase10 main evidence artifacts",
        "csv": args.csv.as_posix(),
        "artifact_count": len(rows),
        "missing_count": missing,
        "roles": sorted({str(row["role"]) for row in rows}),
        "artifacts": rows,
    }
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: payload[k] for k in ("created_at_utc", "artifact_count", "missing_count", "roles")}, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
