"""Audit the Phase10 learned-component claim boundary.

The focused learned-component ablation does not support a universal
raw-discovery dominance claim. It can support a narrower paper claim if two
conditions hold:

1. The trained actor's weights improve the collision-efficiency operating point
   relative to random-weight and zero-weight controls under the same structured
   ISAC/rule interface.
2. The manuscript explicitly states that candidate masks, rule residuals, and
   aggressive access define a tradeoff rather than a universally dominant
   learned policy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SUMMARY_CSV = Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_metric_summary.csv")
RUN_INDEX_CSV = Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_run_index.csv")
SOURCE_HASH_CSV = Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/ablation_source_file_hashes.csv")
ABLATION_MANIFEST = Path("06_analysis/paper_tables/marl/p10_learned_component_ablation_b10_3ep/manifest.json")
MAIN_TEX = Path("07_paper/ieee_twc_isac_nd/main.tex")
SUPPLEMENT_TEX = Path("07_paper/ieee_twc_isac_nd/supplement.tex")
OUT_DIR = Path("06_analysis/paper_tables/marl/p10_learned_claim_boundary_audit")
REPORT = Path("06_analysis/phase10_learned_claim_boundary_audit_20260707.md")

METHOD_ORDER = [
    "trained_full",
    "random_weights_full",
    "zero_weights_rule_only",
    "trained_no_rule_residual",
    "trained_no_candidate_mask",
]

PRIMARY_METRICS = [
    "discovery_rate",
    "collision_penalized_discovery_rate",
    "collision_count",
    "empty_scan_ratio",
    "lambda2",
]


@dataclass(frozen=True)
class TextNeed:
    source: Path
    pattern: str
    label: str


TEXT_NEEDS = [
    TextNeed(
        MAIN_TEX,
        r"collision-efficiency claim rather than a universal raw-discovery dominance claim",
        "main narrow learned-claim boundary",
    ),
    TextNeed(
        SUPPLEMENT_TEX,
        r"random-weight and zero-weight/rule-only policies can produce higher raw discovery",
        "supplement aggressive-access caveat",
    ),
    TextNeed(
        SUPPLEMENT_TEX,
        r"three-episode probe",
        "supplement small-probe scope",
    ),
    TextNeed(
        SUPPLEMENT_TEX,
        r"learned weights help shape the collision-efficiency operating point",
        "supplement learned-weight scope",
    ),
    TextNeed(
        SUPPLEMENT_TEX,
        r"candidate mask and residual priors define the broader discovery/collision/empty-scan tradeoff",
        "supplement structure-prior boundary",
    ),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: str | int | float | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: str | int | float | None, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def metric_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get("label", ""), row.get("metric", "")): row for row in rows}


def load_episode_rows(source_rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = {}
    for row in source_rows:
        if row.get("artifact_type") != "eval_csv":
            continue
        label = row.get("label", "")
        path = Path(row.get("path", "").replace("\\", "/"))
        if label and path.exists():
            output[label] = read_csv(path)
    return output


def paired_seed_rows(episodes: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    trained = episodes.get("trained_full", [])
    by_label_seed: dict[tuple[str, str], dict[str, str]] = {}
    for label, rows in episodes.items():
        for row in rows:
            by_label_seed[(label, str(row.get("seed", "")))] = row

    rows: list[dict[str, object]] = []
    for trained_row in trained:
        seed = str(trained_row.get("seed", ""))
        for control in ("random_weights_full", "zero_weights_rule_only"):
            control_row = by_label_seed.get((control, seed))
            if not control_row:
                continue
            trained_collision = as_float(trained_row.get("collision_count"))
            control_collision = as_float(control_row.get("collision_count"))
            trained_cpd = as_float(trained_row.get("collision_penalized_discovery_rate"))
            control_cpd = as_float(control_row.get("collision_penalized_discovery_rate"))
            trained_discovery = as_float(trained_row.get("discovery_rate"))
            control_discovery = as_float(control_row.get("discovery_rate"))
            rows.append(
                {
                    "seed": seed,
                    "control": control,
                    "trained_collision_count": trained_collision,
                    "control_collision_count": control_collision,
                    "collision_delta_trained_minus_control": trained_collision - control_collision,
                    "trained_cpd": trained_cpd,
                    "control_cpd": control_cpd,
                    "cpd_delta_trained_minus_control": trained_cpd - control_cpd,
                    "trained_discovery_rate": trained_discovery,
                    "control_discovery_rate": control_discovery,
                    "discovery_delta_trained_minus_control": trained_discovery - control_discovery,
                    "collision_improved": trained_collision < control_collision,
                    "cpd_improved": trained_cpd > control_cpd,
                    "raw_discovery_not_dominant": trained_discovery < control_discovery,
                }
            )
    return rows


def find_text_need(need: TextNeed) -> tuple[bool, str]:
    text = read_text(need.source)
    compiled = re.compile(need.pattern, flags=re.IGNORECASE)
    for line_no, line in enumerate(text.splitlines(), start=1):
        if compiled.search(line):
            return True, f"{need.source.as_posix()}:{line_no}"
    return False, ""


def build_check_rows(summary_rows: list[dict[str, str]], paired_rows: list[dict[str, object]], run_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    indexed = metric_index(summary_rows)
    labels_present = {row.get("label", "") for row in run_rows}
    episode_counts = [
        as_int(indexed.get((label, "collision_count"), {}).get("episodes"))
        for label in METHOD_ORDER
    ]
    trained_collision = as_float(indexed.get(("trained_full", "collision_count"), {}).get("mean"))
    random_collision = as_float(indexed.get(("random_weights_full", "collision_count"), {}).get("mean"))
    zero_collision = as_float(indexed.get(("zero_weights_rule_only", "collision_count"), {}).get("mean"))
    trained_cpd = as_float(indexed.get(("trained_full", "collision_penalized_discovery_rate"), {}).get("mean"))
    random_cpd = as_float(indexed.get(("random_weights_full", "collision_penalized_discovery_rate"), {}).get("mean"))
    zero_cpd = as_float(indexed.get(("zero_weights_rule_only", "collision_penalized_discovery_rate"), {}).get("mean"))
    trained_discovery = as_float(indexed.get(("trained_full", "discovery_rate"), {}).get("mean"))
    random_discovery = as_float(indexed.get(("random_weights_full", "discovery_rate"), {}).get("mean"))
    zero_discovery = as_float(indexed.get(("zero_weights_rule_only", "discovery_rate"), {}).get("mean"))
    no_mask_empty = as_float(indexed.get(("trained_no_candidate_mask", "empty_scan_ratio"), {}).get("mean"))
    no_mask_cpd = as_float(indexed.get(("trained_no_candidate_mask", "collision_penalized_discovery_rate"), {}).get("mean"))
    no_rule_collision = as_float(indexed.get(("trained_no_rule_residual", "collision_count"), {}).get("mean"))
    no_rule_discovery = as_float(indexed.get(("trained_no_rule_residual", "discovery_rate"), {}).get("mean"))
    no_rule_lambda2 = as_float(indexed.get(("trained_no_rule_residual", "lambda2"), {}).get("mean"))
    trained_lambda2 = as_float(indexed.get(("trained_full", "lambda2"), {}).get("mean"))

    paired_vs_random = [row for row in paired_rows if row["control"] == "random_weights_full"]
    paired_vs_zero = [row for row in paired_rows if row["control"] == "zero_weights_rule_only"]
    text_hits = [find_text_need(need) for need in TEXT_NEEDS]

    checks = [
        {
            "check_id": "L01",
            "theme": "required ablation variants",
            "status": "PASS" if set(METHOD_ORDER).issubset(labels_present) else "REVIEW",
            "evidence": f"{len(labels_present & set(METHOD_ORDER))}/{len(METHOD_ORDER)} labels present",
            "boundary": "All learned/control feature variants needed to separate weights, mask, and residual priors are present.",
        },
        {
            "check_id": "L02",
            "theme": "minimum paired episodes",
            "status": "PASS" if episode_counts and min(episode_counts) >= 3 else "REVIEW",
            "evidence": f"episode_counts={episode_counts}",
            "boundary": "This is a focused three-episode probe, not a broad stochastic dominance test.",
        },
        {
            "check_id": "L03",
            "theme": "collision reduction vs random and zero weights",
            "status": "PASS" if trained_collision < 0.25 * random_collision and trained_collision < 0.25 * zero_collision else "REVIEW",
            "evidence": f"trained={trained_collision:.1f}, random={random_collision:.1f}, zero={zero_collision:.1f}",
            "boundary": "Supports collision-suppression contribution under the same structured interface.",
        },
        {
            "check_id": "L04",
            "theme": "CPD improvement vs random and zero weights",
            "status": "PASS" if trained_cpd > random_cpd and trained_cpd > zero_cpd else "REVIEW",
            "evidence": f"trained={trained_cpd:.4f}, random={random_cpd:.4f}, zero={zero_cpd:.4f}",
            "boundary": "Supports collision-efficiency, not raw discovery dominance.",
        },
        {
            "check_id": "L05",
            "theme": "paired seed sign consistency",
            "status": "PASS" if paired_rows and all(row["collision_improved"] and row["cpd_improved"] for row in paired_rows) else "REVIEW",
            "evidence": f"{sum(1 for row in paired_rows if row['collision_improved'] and row['cpd_improved'])}/{len(paired_rows)} paired control comparisons improve collision and CPD",
            "boundary": f"Paired controls: random={len(paired_vs_random)}, zero={len(paired_vs_zero)}.",
        },
        {
            "check_id": "L06",
            "theme": "raw discovery non-dominance acknowledged",
            "status": "PASS" if trained_discovery < random_discovery and trained_discovery < zero_discovery else "REVIEW",
            "evidence": f"trained={trained_discovery:.4f}, random={random_discovery:.4f}, zero={zero_discovery:.4f}",
            "boundary": "Prevents overclaiming learned raw-discovery superiority.",
        },
        {
            "check_id": "L07",
            "theme": "candidate-mask tradeoff",
            "status": "PASS" if no_mask_cpd > trained_cpd and no_mask_empty > 0.5 else "REVIEW",
            "evidence": f"no_mask_cpd={no_mask_cpd:.4f}, trained_cpd={trained_cpd:.4f}, no_mask_empty={no_mask_empty:.4f}",
            "boundary": "No-mask CPD is a boundary signal because empty scans increase sharply.",
        },
        {
            "check_id": "L08",
            "theme": "rule-residual tradeoff",
            "status": "PASS" if no_rule_collision < trained_collision and no_rule_discovery < trained_discovery and no_rule_lambda2 < trained_lambda2 else "REVIEW",
            "evidence": f"no_rule_collision={no_rule_collision:.1f}, trained_collision={trained_collision:.1f}, no_rule_discovery={no_rule_discovery:.4f}, trained_discovery={trained_discovery:.4f}",
            "boundary": "Rule residual raises raw discovery/topology while increasing collisions.",
        },
        {
            "check_id": "L09",
            "theme": "manuscript claim boundary",
            "status": "PASS" if all(hit for hit, _ in text_hits) else "REVIEW",
            "evidence": "; ".join(location for hit, location in text_hits if hit),
            "boundary": "; ".join(need.label for need, (hit, _) in zip(TEXT_NEEDS, text_hits) if not hit),
        },
    ]
    return checks


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def write_report(check_rows: list[dict[str, object]], paired_rows: list[dict[str, object]], manifest: dict[str, object]) -> None:
    lines = [
        "# Phase10 Learned-Claim Boundary Audit - 2026-07-07",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Verification Status: ANALYZED",
        "- Scope: focused learned-component ablation and manuscript claim boundary",
        "",
        "## Summary",
        "",
        f"- Checks passed: {manifest['passed_check_count']}/{manifest['check_count']}.",
        f"- Paired trained-vs-control comparisons: {manifest['paired_control_comparison_count']}.",
        f"- Collision/CPD paired improvements: {manifest['paired_collision_cpd_improvement_count']}/{manifest['paired_control_comparison_count']}.",
        f"- Bounded learned claim pass: {manifest['bounded_learned_claim_pass']}.",
        "",
        "## Boundary Interpretation",
        "",
        "The audit supports only a bounded learned-component claim: trained weights improve the collision-efficiency operating point relative to random-weight and zero-weight controls under the same structured ISAC/rule interface.",
        "It does not support universal raw-discovery dominance; the manuscript must keep the current caveats about aggressive access, candidate masks, rule residuals, and the three-episode probe.",
        "",
        "## Check Table",
        "",
        "| ID | Theme | Status | Evidence | Boundary |",
        "|---|---|---|---|---|",
    ]
    for row in check_rows:
        lines.append(
            f"| {md_cell(row['check_id'])} | {md_cell(row['theme'])} | {md_cell(row['status'])} | {md_cell(row['evidence'])} | {md_cell(row['boundary'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired Episode Evidence",
            "",
            "| Seed | Control | Collision delta | CPD delta | Raw discovery delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {seed} | {control} | {collision:.1f} | {cpd:.4f} | {disc:.4f} |".format(
                seed=row["seed"],
                control=row["control"],
                collision=row["collision_delta_trained_minus_control"],
                cpd=row["cpd_delta_trained_minus_control"],
                disc=row["discovery_delta_trained_minus_control"],
            )
        )

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            f"- `{(OUT_DIR / 'learned_claim_checks.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'paired_episode_claim_checks.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'manifest.json').as_posix()}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary_rows = read_csv(SUMMARY_CSV)
    run_rows = read_csv(RUN_INDEX_CSV)
    source_rows = read_csv(SOURCE_HASH_CSV)
    episodes = load_episode_rows(source_rows)
    paired_rows = paired_seed_rows(episodes)
    check_rows = build_check_rows(summary_rows, paired_rows, run_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks_csv = OUT_DIR / "learned_claim_checks.csv"
    paired_csv = OUT_DIR / "paired_episode_claim_checks.csv"
    manifest_json = OUT_DIR / "manifest.json"
    readme_md = OUT_DIR / "README.md"

    write_csv(checks_csv, check_rows, ["check_id", "theme", "status", "evidence", "boundary"])
    write_csv(
        paired_csv,
        paired_rows,
        [
            "seed",
            "control",
            "trained_collision_count",
            "control_collision_count",
            "collision_delta_trained_minus_control",
            "trained_cpd",
            "control_cpd",
            "cpd_delta_trained_minus_control",
            "trained_discovery_rate",
            "control_discovery_rate",
            "discovery_delta_trained_minus_control",
            "collision_improved",
            "cpd_improved",
            "raw_discovery_not_dominant",
        ],
    )

    passed_check_count = sum(1 for row in check_rows if row["status"] == "PASS")
    paired_collision_cpd = sum(1 for row in paired_rows if row["collision_improved"] and row["cpd_improved"])
    output_files = [checks_csv, paired_csv, readme_md, REPORT]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded learned-component claim audit",
        "source_files": [
            SUMMARY_CSV.as_posix(),
            RUN_INDEX_CSV.as_posix(),
            SOURCE_HASH_CSV.as_posix(),
            ABLATION_MANIFEST.as_posix(),
            MAIN_TEX.as_posix(),
            SUPPLEMENT_TEX.as_posix(),
        ],
        "check_count": len(check_rows),
        "passed_check_count": passed_check_count,
        "paired_control_comparison_count": len(paired_rows),
        "paired_collision_cpd_improvement_count": paired_collision_cpd,
        "bounded_learned_claim_pass": passed_check_count == len(check_rows) and paired_collision_cpd == len(paired_rows) and len(paired_rows) >= 6,
        "raw_episode_labels": sorted(episodes),
        "outputs": {
            "checks_csv": checks_csv.as_posix(),
            "paired_episode_claim_checks_csv": paired_csv.as_posix(),
            "report": REPORT.as_posix(),
            "readme": readme_md.as_posix(),
        },
    }

    write_report(check_rows, paired_rows, manifest)
    readme_md.write_text(
        "\n".join(
            [
                "# Phase10 Learned-Claim Boundary Audit",
                "",
                "Generated by `06_analysis/scripts/build_phase10_learned_claim_boundary_audit.py`.",
                "The audit verifies that the focused learned-component ablation supports only a bounded collision-efficiency claim and that the manuscript retains the required caveats.",
                "",
                "- `learned_claim_checks.csv`: required numeric and text-boundary checks.",
                "- `paired_episode_claim_checks.csv`: paired seed-level trained-vs-control deltas.",
                "- `manifest.json`: machine-readable audit summary.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_files = [checks_csv, paired_csv, readme_md, REPORT]
    manifest["output_hashes"] = {path.as_posix(): sha256_file(path) for path in output_files}
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "created_at_utc": manifest["created_at_utc"],
                "checks": f"{passed_check_count}/{len(check_rows)}",
                "paired_collision_cpd": f"{paired_collision_cpd}/{len(paired_rows)}",
                "bounded_learned_claim_pass": manifest["bounded_learned_claim_pass"],
                "report": REPORT.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
