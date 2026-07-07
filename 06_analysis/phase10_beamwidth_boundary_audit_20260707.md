# Phase10 Beamwidth Boundary Audit - 2026-07-07

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Scope: final 10/15-degree Phase10 evidence plus 3/5/30-degree boundary wording

## Summary

- Checks passed: 6/6.
- Final Phase10 beams: [10.0, 15.0].
- Stability-summary beams: [3.0, 5.0, 10.0, 15.0, 30.0].
- Bounded beamwidth pass: True.

## Boundary Interpretation

The audit supports a bounded beamwidth claim: the paper-facing Phase10 MARL comparison covers 10- and 15-degree narrow beams, while 3/5-degree and 30-degree results remain supplementary stress or historical boundary evidence.
It does not convert the 3/5/30-degree archived sweeps into final-main performance claims.

## Check Table

| ID | Theme | Status | Evidence | Boundary |
|---|---|---|---|---|
| B01 | final Phase10 beamwidth set | PASS | final_beams=[10.0, 15.0] | Final Phase10 method comparison is limited to 10/15 degrees. |
| B02 | method coverage per final beam | PASS | {"10.0": 9, "15.0": 9} | Each final beam contains the full nine-method comparison. |
| B03 | narrow stress boundary retained | PASS | stability_beams=[3.0, 5.0, 10.0, 15.0, 30.0] | 3/5-degree data exist as stress/boundary evidence, not as final Phase10 main rows. |
| B04 | 30-degree legacy exclusion | PASS | final_b30_rows=0, stability_b30_rows=22 | 30-degree evidence exists only outside the final main comparison. |
| B05 | stability tier trace | PASS | {"10.0\|main": 75, "10.0\|main_boundary": 12, "10.0\|supplement": 60, "10.0\|supplement_backup": 5, "10.0\|supplement_sanity": 8, "15.0\|main": 31, "15.0\|main_boundary": 12, "15.0\|supplement": 60, "15.0\|supplement_backup": 5, "15.0\|supplement_sanity": 16, "3.0\|supplement": 12, "3.0\|supplement_stress": 5, "30.0\|main": 10, "30.0\|supplement": 12, "5.0\|main": 10, "5.0\|supplement": 12} | Beamwidth support is traceable by evidence tier. |
| B06 | manuscript boundary wording | PASS | 07_paper/ieee_twc_isac_nd/main.tex:64; 07_paper/ieee_twc_isac_nd/main.tex:64; 07_paper/ieee_twc_isac_nd/main.tex:301; 07_paper/ieee_twc_isac_nd/main.tex:481; 07_paper/ieee_twc_isac_nd/supplement.tex:45; 07_paper/ieee_twc_isac_nd/supplement.tex:45; 07_paper/ieee_twc_isac_nd/supplement.tex:146 |  |

## Generated Files

- `06_analysis/paper_tables/marl/p10_beamwidth_boundary_audit/beamwidth_boundary_checks.csv`
- `06_analysis/paper_tables/marl/p10_beamwidth_boundary_audit/manifest.json`
- `06_analysis/paper_tables/marl/p10_beamwidth_boundary_audit/README.md`
