"""Audit manuscript claim strength against explicit submission boundaries.

This script is intentionally conservative. It does not decide whether the
paper is strong enough; it checks whether high-risk claims in the main paper
and supplement are bounded by simulator scope, protocol-level abstraction,
baseline caveats, statistical scope, and learned-component limitations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


MAIN_TEX = Path("07_paper/ieee_twc_isac_nd/main.tex")
SUPPLEMENT_TEX = Path("07_paper/ieee_twc_isac_nd/supplement.tex")
OUT_DIR = Path("06_analysis/paper_tables/submission_claim_strength_audit")
REPORT = Path("06_analysis/submission_claim_strength_audit_20260707.md")


@dataclass(frozen=True)
class EvidenceNeed:
    source: Path
    pattern: str
    label: str


@dataclass(frozen=True)
class ClaimCheck:
    check_id: str
    theme: str
    rationale: str
    needs: tuple[EvidenceNeed, ...]


RISK_PATTERNS = [
    r"\bfirst\b",
    r"\bnovel\b",
    r"\boptimal(?:ity)?\b",
    r"\bguarantee(?:s|d)?\b",
    r"\bprove(?:s|d)?\b",
    r"\bproof\b",
    r"\bdominates?\b",
    r"\bdominant\b",
    r"\balways\b",
    r"\buniversal(?:ly)?\b",
    r"\bstate-of-the-art\b",
    r"\breal-world\b",
    r"\bhardware\b",
    r"\bsignificant(?:ly)?\b",
    r"\bcomplete\b",
    r"\bfull\b",
    r"\btheoretical\b",
    r"\bconvergence\b",
]

BOUNDARY_TERMS = [
    "not ",
    "not a ",
    "not as ",
    "not be ",
    "not prove",
    "not presented as",
    "not interpreted",
    "not replace",
    "rather than",
    "outside",
    "open boundary",
    "open issues",
    "boundary",
    "bounded",
    "caveat",
    "scope",
    "scoped",
    "deliberately scoped",
    "descriptive",
    "diagnostic",
    "tested",
    "single-hop",
    "finite-horizon",
    "simulator",
    "simulation",
    "protocol-level",
    "abstraction",
    "matched-support",
    "empirical",
    "inspired",
    "approximate",
    "not a strict reproduction",
    "not a full",
    "not a head-to-head",
    "not global",
    "not platform-calibrated",
    "only",
    "current evidence",
    "current manuscript",
    "remains",
    "remain",
    "should not",
    "does not",
    "do not",
    "used to",
]

LITERATURE_TERMS = [
    "deterministic methods",
    "oblivious nd",
    "introduced",
    "studied",
    "established",
    "\\cite",
]

BENIGN_TERMS = [
    "first, per-beam",
    "complete evidence map",
    "old occupancy evidence is decayed",
    "trained full",
    "full-feature controls",
    "full stochastic experiment re-run verification",
    "full-baseline stress",
]


CHECKS = [
    ClaimCheck(
        "C01",
        "tested transfer scope",
        "The abstract-level performance claim must be tied to the tested single-hop transfer setting.",
        (
            EvidenceNeed(MAIN_TEX, r"In the tested single-hop \$N=100\$ transfer setting", "tested N=100 single-hop scope"),
            EvidenceNeed(MAIN_TEX, r"trained at \$N=10\$, 10-degree beams, and 300 slots per episode", "small-source training scope"),
            EvidenceNeed(MAIN_TEX, r"evaluated without fine-tuning at \$N=100\$ over 3000-slot long-horizon tests", "zero-shot long-horizon transfer scope"),
        ),
    ),
    ClaimCheck(
        "C02",
        "distributed information boundary",
        "The system model must state what distributed execution does and does not assume.",
        (
            EvidenceNeed(MAIN_TEX, r"no central scheduler", "no central scheduler"),
            EvidenceNeed(MAIN_TEX, r"no undiscovered-neighbor state exchange", "no undiscovered-neighbor state exchange"),
            EvidenceNeed(MAIN_TEX, r"common navigation reference available from onboard localization and inertial sensing", "local navigation reference assumption"),
        ),
    ),
    ClaimCheck(
        "C03",
        "ISAC abstraction boundary",
        "ISAC should be presented as a protocol-level sensing service, not as a fully specified PHY/radar design.",
        (
            EvidenceNeed(MAIN_TEX, r"protocol-level input parameters", "protocol-level sensing inputs"),
            EvidenceNeed(MAIN_TEX, r"rather than treating equality as a hardware law", "equal-range hardware-law caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"not as a calibrated hardware statement", "supplement hardware calibration caveat"),
        ),
    ),
    ClaimCheck(
        "C04",
        "SkyOrbs-like baseline caveat",
        "The literature-inspired directional baseline must not be framed as a faithful SkyOrbs reproduction.",
        (
            EvidenceNeed(MAIN_TEX, r"not a strict reproduction of the complete SkyOrbs protocol", "main strict-reproduction caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"not a head-to-head claim against complete SkyOrbs", "supplement head-to-head caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"SkyOrbs-like results are used to contextualize deterministic communication-only scan scheduling", "supplement contextual-use caveat"),
        ),
    ),
    ClaimCheck(
        "C05",
        "MARL optimality boundary",
        "The learned actor contribution must not be described as an optimal MAC theorem.",
        (
            EvidenceNeed(MAIN_TEX, r"not an optimal MAC theorem", "main optimal-MAC caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"not.*globally optimal MAC policy", "supplement global-optimality caveat"),
        ),
    ),
    ClaimCheck(
        "C06",
        "training convergence boundary",
        "Training curves should be empirical diagnostics, not theoretical convergence evidence.",
        (
            EvidenceNeed(MAIN_TEX, r"not a proof of convergence", "main training convergence caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"not a theoretical convergence proof", "supplement training convergence caveat"),
        ),
    ),
    ClaimCheck(
        "C07",
        "topology metric boundary",
        "Topology-quality metrics should not be converted into a closed-loop consensus theorem.",
        (
            EvidenceNeed(MAIN_TEX, r"topology-quality proxy", "lambda2 proxy wording"),
            EvidenceNeed(MAIN_TEX, r"does not prove a closed-loop consensus-control convergence theorem", "closed-loop consensus caveat"),
        ),
    ),
    ClaimCheck(
        "C08",
        "statistical scope",
        "Confidence intervals and paired tests must be bounded to their declared comparison families.",
        (
            EvidenceNeed(MAIN_TEX, r"not global multiple-comparison-corrected hypothesis tests", "main non-global CI/test caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"gate-family operating points remain descriptive", "gate-family descriptive statistics caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"do not convert the gate-family frontier into a globally corrected superiority claim", "paired-significance boundary"),
        ),
    ),
    ClaimCheck(
        "C09",
        "gate-family frontier boundary",
        "Gate variants should be framed as an operating frontier, not as universally dominant replacements.",
        (
            EvidenceNeed(MAIN_TEX, r"operating-point ablations rather than as universally dominant replacements", "gate ablation caveat"),
            EvidenceNeed(MAIN_TEX, r"not a single gate that dominates every metric", "frontier wording"),
        ),
    ),
    ClaimCheck(
        "C10",
        "learned-component boundary",
        "The learned-weight contribution should be limited to collision-efficiency under structured priors.",
        (
            EvidenceNeed(MAIN_TEX, r"collision-efficiency claim rather than a universal raw-discovery dominance claim", "main learned-claim caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"learned weights help shape the collision-efficiency operating point", "supplement learned-weight scope"),
            EvidenceNeed(SUPPLEMENT_TEX, r"candidate mask and residual priors define the broader discovery/collision/empty-scan tradeoff", "structured-prior boundary"),
        ),
    ),
    ClaimCheck(
        "C11",
        "energy accounting boundary",
        "Energy metrics should remain diagnostic unless platform power states are calibrated.",
        (
            EvidenceNeed(MAIN_TEX, r"not presented as a platform-calibrated energy-optimality result", "main energy caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"not platform-calibrated energy-optimality results", "supplement energy caveat"),
        ),
    ),
    ClaimCheck(
        "C12",
        "MAC scope boundary",
        "Collision-aware MAC refinement must remain an open boundary rather than a solved full-MAC claim.",
        (
            EvidenceNeed(MAIN_TEX, r"complete collision- and energy-aware MAC remains outside the present scope", "main MAC scope caveat"),
            EvidenceNeed(SUPPLEMENT_TEX, r"does not close the full MAC-design problem", "supplement full-MAC caveat"),
        ),
    ),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8-sig").splitlines()


def find_pattern(lines: list[str], pattern: str) -> list[tuple[int, str]]:
    compiled = re.compile(pattern, flags=re.IGNORECASE)
    hits = []
    for line_no, line in enumerate(lines, start=1):
        if compiled.search(line):
            hits.append((line_no, " ".join(line.strip().split())))
    return hits


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def context_window(lines: list[str], line_no: int, radius: int = 1) -> str:
    start = max(1, line_no - radius)
    end = min(len(lines), line_no + radius)
    return " ".join(" ".join(lines[index - 1].strip().split()) for index in range(start, end + 1)).strip()


def classify_risk(context: str) -> str:
    lowered = context.lower()
    if any(term in lowered for term in BENIGN_TERMS):
        return "benign_usage"
    if any(term in lowered for term in LITERATURE_TERMS):
        return "literature_context"
    if any(term in lowered for term in BOUNDARY_TERMS):
        return "bounded"
    return "review_required"


def build_check_rows(source_lines: dict[Path, list[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for check in CHECKS:
        need_results = []
        for need in check.needs:
            hits = find_pattern(source_lines.get(need.source, []), need.pattern)
            need_results.append(
                {
                    "label": need.label,
                    "source": need.source.as_posix(),
                    "pattern": need.pattern,
                    "passed": bool(hits),
                    "line_numbers": ";".join(str(line_no) for line_no, _ in hits),
                    "snippets": " || ".join(snippet for _, snippet in hits[:3]),
                }
            )
        passed = all(result["passed"] for result in need_results)
        rows.append(
            {
                "check_id": check.check_id,
                "theme": check.theme,
                "status": "PASS" if passed else "REVIEW",
                "rationale": check.rationale,
                "evidence_count": sum(1 for result in need_results if result["passed"]),
                "required_count": len(need_results),
                "missing_labels": ";".join(result["label"] for result in need_results if not result["passed"]),
                "evidence_locations": " | ".join(
                    f"{result['label']}@{result['source']}:{result['line_numbers']}" for result in need_results if result["passed"]
                ),
                "evidence_snippets": " | ".join(str(result["snippets"]) for result in need_results if result["passed"]),
            }
        )
    return rows


def build_risk_rows(source_lines: dict[Path, list[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    compiled = [(pattern, re.compile(pattern, flags=re.IGNORECASE)) for pattern in RISK_PATTERNS]
    for source, lines in source_lines.items():
        for line_no, line in enumerate(lines, start=1):
            matched = sorted({pattern for pattern, regex in compiled if regex.search(line)})
            if not matched:
                continue
            context = context_window(lines, line_no)
            classification = classify_risk(context)
            rows.append(
                {
                    "source": source.as_posix(),
                    "line": line_no,
                    "risk_patterns": ";".join(matched),
                    "classification": classification,
                    "snippet": " ".join(line.strip().split()),
                    "context": context,
                }
            )
    return rows


def write_report(check_rows: list[dict[str, object]], risk_rows: list[dict[str, object]], manifest: dict[str, object]) -> None:
    lines = [
        "# Submission Claim-Strength Audit - 2026-07-07",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: sci-paper-polisher",
        "- Origin Mode: submission claim boundary audit",
        "- Verification Status: ANALYZED",
        "- Scope: main manuscript and supplement claim wording",
        "",
        "## Summary",
        "",
        f"- Required claim-boundary checks: {manifest['passed_required_checks']}/{manifest['required_check_count']} passed.",
        f"- Risk-phrase hits: {manifest['risk_hit_count']} total; {manifest['review_required_risk_hits']} require review.",
        f"- Overall pass: {manifest['all_required_checks_pass'] and manifest['review_required_risk_hits'] == 0}.",
        "",
        "## Required Boundary Checks",
        "",
        "| ID | Theme | Status | Evidence | Missing |",
        "|---|---|---|---|---|",
    ]
    for row in check_rows:
        lines.append(
            f"| {row['check_id']} | {row['theme']} | {row['status']} | {row['evidence_count']}/{row['required_count']} | {row['missing_labels']} |"
        )

    lines.extend(
        [
            "",
            "## Risk-Phrase Classification",
            "",
            "| Classification | Count |",
            "|---|---:|",
        ]
    )
    counts: dict[str, int] = {}
    for row in risk_rows:
        classification = str(row["classification"])
        counts[classification] = counts.get(classification, 0) + 1
    for classification in sorted(counts):
        lines.append(f"| {classification} | {counts[classification]} |")

    review_rows = [row for row in risk_rows if row["classification"] == "review_required"]
    lines.extend(["", "## Review-Required Hits", ""])
    if not review_rows:
        lines.append("None. All risk phrases are bounded by local caveats or appear in literature/contextual wording.")
    else:
        for row in review_rows:
            lines.append(f"- `{row['source']}:{row['line']}` {row['risk_patterns']}: {row['snippet']}")

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            f"- `{(OUT_DIR / 'claim_boundary_checks.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'risk_phrase_hits.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'manifest.json').as_posix()}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source_lines = {
        MAIN_TEX: read_lines(MAIN_TEX),
        SUPPLEMENT_TEX: read_lines(SUPPLEMENT_TEX),
    }

    check_rows = build_check_rows(source_lines)
    risk_rows = build_risk_rows(source_lines)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks_csv = OUT_DIR / "claim_boundary_checks.csv"
    risk_csv = OUT_DIR / "risk_phrase_hits.csv"
    manifest_json = OUT_DIR / "manifest.json"
    readme_md = OUT_DIR / "README.md"

    write_csv(
        checks_csv,
        check_rows,
        [
            "check_id",
            "theme",
            "status",
            "rationale",
            "evidence_count",
            "required_count",
            "missing_labels",
            "evidence_locations",
            "evidence_snippets",
        ],
    )
    write_csv(
        risk_csv,
        risk_rows,
        ["source", "line", "risk_patterns", "classification", "snippet", "context"],
    )

    passed_required_checks = sum(1 for row in check_rows if row["status"] == "PASS")
    review_required_risk_hits = sum(1 for row in risk_rows if row["classification"] == "review_required")
    output_files = [checks_csv, risk_csv, readme_md, REPORT]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "line-level claim-strength audit for main.tex and supplement.tex",
        "required_check_count": len(check_rows),
        "passed_required_checks": passed_required_checks,
        "all_required_checks_pass": passed_required_checks == len(check_rows),
        "risk_hit_count": len(risk_rows),
        "review_required_risk_hits": review_required_risk_hits,
        "source_files": [MAIN_TEX.as_posix(), SUPPLEMENT_TEX.as_posix()],
        "output_files": [path.as_posix() for path in output_files],
        "outputs": {
            "checks_csv": checks_csv.as_posix(),
            "risk_phrase_hits_csv": risk_csv.as_posix(),
            "report": REPORT.as_posix(),
            "readme": readme_md.as_posix(),
        },
    }

    write_report(check_rows, risk_rows, manifest)
    readme_md.write_text(
        "\n".join(
            [
                "# Submission Claim-Strength Audit",
                "",
                "Generated by `06_analysis/scripts/build_submission_claim_strength_audit.py`.",
                "The audit checks that manuscript claims remain bounded by the simulator, protocol-level ISAC abstraction, literature-baseline scope, statistics scope, and learned-component evidence.",
                "",
                "- `claim_boundary_checks.csv`: required boundary evidence checks.",
                "- `risk_phrase_hits.csv`: high-risk claim-word scan with local classification.",
                "- `manifest.json`: source and output trace for this audit.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_files = [checks_csv, risk_csv, readme_md, REPORT]
    manifest["output_hashes"] = {path.as_posix(): sha256_file(path) for path in output_files}
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "created_at_utc": manifest["created_at_utc"],
                "required_checks": f"{passed_required_checks}/{len(check_rows)}",
                "risk_hit_count": len(risk_rows),
                "review_required_risk_hits": review_required_risk_hits,
                "report": REPORT.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
