"""Audit the Phase10 beamwidth coverage boundary.

The final manuscript-facing MARL evidence is intentionally a 10-to-15 degree
transfer line. Earlier 3/5-degree and 30-degree sweeps remain useful boundary
evidence, but they are not part of the final Phase10 main comparison. This
audit checks that the data and manuscript wording preserve that boundary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


FINAL_METHOD_CSV = Path("06_analysis/paper_tables/marl/p10_final_b10_b15_method_comparison_with_v4/marl_method_comparison.csv")
STABILITY_CSV = Path("06_analysis/paper_tables/statistical_stability_summary/statistical_stability_summary.csv")
MAIN_TEX = Path("07_paper/ieee_twc_isac_nd/main.tex")
SUPPLEMENT_TEX = Path("07_paper/ieee_twc_isac_nd/supplement.tex")
OUT_DIR = Path("06_analysis/paper_tables/marl/p10_beamwidth_boundary_audit")
REPORT = Path("06_analysis/phase10_beamwidth_boundary_audit_20260707.md")

REQUIRED_FINAL_BEAMS = {10.0, 15.0}
REQUIRED_BOUNDARY_BEAMS = {3.0, 5.0}


@dataclass(frozen=True)
class TextNeed:
    source: Path
    pattern: str
    label: str


TEXT_NEEDS = [
    TextNeed(
        MAIN_TEX,
        r"10- and 15-degree narrow beams",
        "main contribution limits final narrow-beam line to 10/15 degrees",
    ),
    TextNeed(
        MAIN_TEX,
        r"3--5 degree cases and 30-degree legacy sweeps are retained only as supplementary boundary evidence",
        "main contribution labels 3/5 and 30 degrees as boundary evidence",
    ),
    TextNeed(
        MAIN_TEX,
        r"Long transfer tests.*10- and 15-degree beams for the final MARL evidence",
        "main settings table separates final evidence from boundary sweeps",
    ),
    TextNeed(
        MAIN_TEX,
        r"extremely narrow beams at 3 and 5 degrees, the archived 30-degree boundary sweep",
        "main limitation keeps 3/5/30 out of solved-case claims",
    ),
    TextNeed(
        SUPPLEMENT_TEX,
        r"Phase10 final MARL evidence focuses on 10/15 deg",
        "supplement coverage table states final focus",
    ),
    TextNeed(
        SUPPLEMENT_TEX,
        r"3 and 5 deg are stress regimes; 30 deg is not part of the final main comparison",
        "supplement coverage table states stress/historical boundary",
    ),
    TextNeed(
        SUPPLEMENT_TEX,
        r"The 3- and 5-degree cases are intentionally treated as failure-boundary regimes",
        "supplement beamwidth section states 3/5 failure boundary",
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


def beam_set(rows: list[dict[str, str]]) -> set[float]:
    return {as_float(row.get("beamwidth_deg")) for row in rows if row.get("beamwidth_deg", "") != ""}


def counts_by_key(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in keys)
        counts[key] = counts.get(key, 0) + 1
    return counts


def find_text_need(need: TextNeed) -> tuple[bool, str]:
    text = need.source.read_text(encoding="utf-8-sig") if need.source.exists() else ""
    compiled = re.compile(need.pattern, flags=re.IGNORECASE)
    for line_no, line in enumerate(text.splitlines(), start=1):
        if compiled.search(line):
            return True, f"{need.source.as_posix()}:{line_no}"
    return False, ""


def build_check_rows(final_rows: list[dict[str, str]], stability_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    final_beams = beam_set(final_rows)
    stability_beams = beam_set(stability_rows)
    final_methods_by_beam = counts_by_key(final_rows, "beamwidth_deg")
    stability_tier_counts = counts_by_key(stability_rows, "beamwidth_deg", "evidence_tier")
    text_hits = [find_text_need(need) for need in TEXT_NEEDS]
    text_locations = "; ".join(location for hit, location in text_hits if hit)
    missing_text = "; ".join(need.label for need, (hit, _) in zip(TEXT_NEEDS, text_hits) if not hit)
    b30_final_rows = [row for row in final_rows if abs(as_float(row.get("beamwidth_deg")) - 30.0) < 1e-9]
    b30_stability_rows = [row for row in stability_rows if abs(as_float(row.get("beamwidth_deg")) - 30.0) < 1e-9]

    checks = [
        {
            "check_id": "B01",
            "theme": "final Phase10 beamwidth set",
            "status": "PASS" if REQUIRED_FINAL_BEAMS.issubset(final_beams) and final_beams.issubset(REQUIRED_FINAL_BEAMS) else "REVIEW",
            "evidence": f"final_beams={sorted(final_beams)}",
            "boundary": "Final Phase10 method comparison is limited to 10/15 degrees.",
        },
        {
            "check_id": "B02",
            "theme": "method coverage per final beam",
            "status": "PASS" if all(final_methods_by_beam.get((f"{beam:.1f}",), 0) >= 9 for beam in REQUIRED_FINAL_BEAMS) else "REVIEW",
            "evidence": json.dumps({key[0]: value for key, value in sorted(final_methods_by_beam.items())}, ensure_ascii=False, sort_keys=True),
            "boundary": "Each final beam contains the full nine-method comparison.",
        },
        {
            "check_id": "B03",
            "theme": "narrow stress boundary retained",
            "status": "PASS" if REQUIRED_BOUNDARY_BEAMS.issubset(stability_beams) else "REVIEW",
            "evidence": f"stability_beams={sorted(stability_beams)}",
            "boundary": "3/5-degree data exist as stress/boundary evidence, not as final Phase10 main rows.",
        },
        {
            "check_id": "B04",
            "theme": "30-degree legacy exclusion",
            "status": "PASS" if not b30_final_rows and len(b30_stability_rows) > 0 else "REVIEW",
            "evidence": f"final_b30_rows={len(b30_final_rows)}, stability_b30_rows={len(b30_stability_rows)}",
            "boundary": "30-degree evidence exists only outside the final main comparison.",
        },
        {
            "check_id": "B05",
            "theme": "stability tier trace",
            "status": "PASS" if stability_tier_counts else "REVIEW",
            "evidence": json.dumps(
                {"|".join(key): value for key, value in sorted(stability_tier_counts.items()) if key[0] in {"3.0", "5.0", "10.0", "15.0", "30.0"}},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "boundary": "Beamwidth support is traceable by evidence tier.",
        },
        {
            "check_id": "B06",
            "theme": "manuscript boundary wording",
            "status": "PASS" if all(hit for hit, _ in text_hits) else "REVIEW",
            "evidence": text_locations,
            "boundary": missing_text,
        },
    ]
    return checks


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|")


def write_report(check_rows: list[dict[str, object]], manifest: dict[str, object]) -> None:
    lines = [
        "# Phase10 Beamwidth Boundary Audit - 2026-07-07",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        "- Verification Status: ANALYZED",
        "- Scope: final 10/15-degree Phase10 evidence plus 3/5/30-degree boundary wording",
        "",
        "## Summary",
        "",
        f"- Checks passed: {manifest['passed_check_count']}/{manifest['check_count']}.",
        f"- Final Phase10 beams: {manifest['final_beams']}.",
        f"- Stability-summary beams: {manifest['stability_beams']}.",
        f"- Bounded beamwidth pass: {manifest['bounded_beamwidth_pass']}.",
        "",
        "## Boundary Interpretation",
        "",
        "The audit supports a bounded beamwidth claim: the paper-facing Phase10 MARL comparison covers 10- and 15-degree narrow beams, while 3/5-degree and 30-degree results remain supplementary stress or historical boundary evidence.",
        "It does not convert the 3/5/30-degree archived sweeps into final-main performance claims.",
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
            "## Generated Files",
            "",
            f"- `{(OUT_DIR / 'beamwidth_boundary_checks.csv').as_posix()}`",
            f"- `{(OUT_DIR / 'manifest.json').as_posix()}`",
            f"- `{(OUT_DIR / 'README.md').as_posix()}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    final_rows = read_csv(FINAL_METHOD_CSV)
    stability_rows = read_csv(STABILITY_CSV)
    check_rows = build_check_rows(final_rows, stability_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks_csv = OUT_DIR / "beamwidth_boundary_checks.csv"
    manifest_json = OUT_DIR / "manifest.json"
    readme_md = OUT_DIR / "README.md"

    write_csv(checks_csv, check_rows, ["check_id", "theme", "status", "evidence", "boundary"])
    passed_check_count = sum(1 for row in check_rows if row["status"] == "PASS")
    final_beams = sorted(beam_set(final_rows))
    stability_beams = sorted(beam_set(stability_rows))
    output_files = [checks_csv, readme_md, REPORT]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "bounded beamwidth coverage audit",
        "source_files": [
            FINAL_METHOD_CSV.as_posix(),
            STABILITY_CSV.as_posix(),
            MAIN_TEX.as_posix(),
            SUPPLEMENT_TEX.as_posix(),
        ],
        "check_count": len(check_rows),
        "passed_check_count": passed_check_count,
        "final_beams": final_beams,
        "stability_beams": stability_beams,
        "bounded_beamwidth_pass": passed_check_count == len(check_rows),
        "outputs": {
            "checks_csv": checks_csv.as_posix(),
            "report": REPORT.as_posix(),
            "readme": readme_md.as_posix(),
        },
    }

    write_report(check_rows, manifest)
    readme_md.write_text(
        "\n".join(
            [
                "# Phase10 Beamwidth Boundary Audit",
                "",
                "Generated by `06_analysis/scripts/build_phase10_beamwidth_boundary_audit.py`.",
                "The audit checks that the final Phase10 beamwidth claim remains limited to 10/15-degree transfer while 3/5/30-degree evidence is treated as boundary support.",
                "",
                "- `beamwidth_boundary_checks.csv`: required data and manuscript-boundary checks.",
                "- `manifest.json`: machine-readable audit summary.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_files = [checks_csv, readme_md, REPORT]
    manifest["output_hashes"] = {path.as_posix(): sha256_file(path) for path in output_files}
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "created_at_utc": manifest["created_at_utc"],
                "checks": f"{passed_check_count}/{len(check_rows)}",
                "bounded_beamwidth_pass": manifest["bounded_beamwidth_pass"],
                "report": REPORT.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
