# Research Goal Coverage Audit - 2026-07-07

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Scope: current MARL+ISAC neighbor-discovery manuscript evidence chain

## Summary

- Requirements checked: 18.
- Status counts: PASS=11, CAUTION=5, OPEN=2.
- Final Phase10 method rows: 18 across beams [10.0, 15.0].
- Training trace: 3 runs, 90000 step rows, 300 episode rows.
- Method trace completeness: 18/18.
- Artifact manifest: 109 artifacts, 0 missing.

## Requirement Coverage

| ID | Theme | Status | Evidence strength | Evidence summary |
|---|---|---|---|---|
| R01 | MARL implementation | PASS | code+tests | 6/6 core MARL code/test files are present; this proves an implemented MARL pipeline, not that learned weights dominate every rule residual. |
| R02 | Training trace | PASS | main | Training manifest reports 3 runs, 90000 step rows, 300 episode rows, 198 eval rows. |
| R03 | Final transfer | PASS | main | Final comparison has nodes=100, slots=3000, beams=10.0;15.0, rows=18. |
| R04 | Baseline completeness | PASS | main_with_caveat | Final methods cover 9/9 required method groups. |
| R05 | ISAC gain | PASS | main | Minimum actor discovery ratio is 22.31x vs random/SkyOrbs-like and 97.46x vs no-ISAC MARL across B=10/15. |
| R06 | Network/method innovation | PASS | main+code | Final table contains contention and four gate-family variants; code contains matching neural classes. |
| R07 | Beamwidth coverage and transfer | CAUTION | main+boundary | Stability summary beams=10.0;15.0;3.0;30.0;5.0; final Phase10 beams=10.0;15.0. B=30 exists only as archived boundary evidence. |
| R08 | Node-count scalability | PASS | main+supplement | Stability summary nodes=10;100;20;50; final main table is N=100 transfer. |
| R09 | Dynamic mobility | PASS | main+supplement | Mobility models in stability summary: gauss_markov;random_direction;random_walk;random_waypoint. |
| R10 | Range abstraction | CAUTION | supplement | Final Phase10 rows use Rc=900 m and Rs=900 m under single-hop transfer; range-grid evidence exists in robustness tables. |
| R11 | Time-scale assumption | PASS | main+supplement | Slot-duration values in stability summary: 1.0;10.0;100.0;20.0;5.0. |
| R12 | Statistical reliability | CAUTION | analyzed_not_verified | Final transfer episodes range 10-20; stability rows=345 with main rows=126. |
| R13 | Reproducibility trace | PASS | main | Method trace rows complete: 18/18. |
| R14 | Figure and artifact integrity | PASS | artifact_hash | Artifact manifest reports 109 artifacts and 0 missing paths; training manifest lists 19 training/resource figures. |
| R15 | Submission wording boundary | CAUTION | reviewer_audit | Existing readiness review records 195 supplement/supplement-stress rows and flags SkyOrbs-like/reproduction limits. |
| R16 | Independent reproduction | OPEN | missing_re_run | Current validation is artifact/hash and statistical analysis; no independent full stochastic Phase10 re-run is recorded in the validation report. |
| R17 | Learned component ablation | OPEN | missing_ablation | The code contains PPO/MAPPO-style learning and neural actors, but the final evidence does not yet include random-weight, frozen-rule, zero-residual, or no-candidate-mask ablations. |
| R18 | Raw bundle availability | CAUTION | local_trace_not_git_archive | Local method trace paths exist for 28/28 raw manifests and 8/8 checkpoints; 0/36 raw trace files are tracked by Git. |

## Highest-Value Remaining Work

- P2 `CAUTION`: Cover narrow beamwidths around 3-15 degrees, with final main transfer at 10->15 degrees. Next: If reviewers demand full stress coverage, rerun B=3/B=5 with the final Phase10 method set; B=30 is intentionally excluded from the final line.
- P2 `CAUTION`: Make communication/sensing range assumptions explicit and test range sensitivity. Next: Add or preserve theoretical/citation support for when Rs can equal Rc; avoid claiming this is hardware-calibrated.
- P2 `CAUTION`: Provide multi-seed/statistical summaries rather than single-run-only claims. Next: The next strongest evidence upgrade is one independent Phase10 transfer re-run for a key method/beam pair.
- P2 `CAUTION`: Keep claims aligned with simulator, protocol-level ISAC abstraction, and approximate literature baseline scope. Next: Before submission, run a line-level IEEE style and claim-strength pass.
- P1 `OPEN`: Convert at least part of the current status from ANALYZED to VERIFIED by re-running a selected final experiment. Next: Run one small independent final-transfer re-run, e.g., gated_contention_actor at B=10 or B=15, and compare within a predeclared tolerance.
- P1 `OPEN`: Separate learned actor contribution from strong rule priors, residual logits, candidate masks, and decentralized gates. Next: Run a focused learned-vs-rule ablation before making any claim that the neural component, rather than the structured prior alone, drives the performance gain.
- P2 `CAUTION`: Retain a local or archived raw-result bundle with manifests and checkpoint hashes for the final Phase10 evidence line. Next: Before submission or external release, decide whether to archive checkpoints or publish a separate checksum manifest for raw state_dict files.
- P2 `CAUTION`: Literature-inspired SkyOrbs-like baseline may be challenged as not a faithful reproduction. Next: Either retain explicit caveat or implement a stricter reproduction before submission.

## Boundary Interpretation

The current evidence is strong enough to support a manuscript draft centered on a real MAPPO-style MARL+ISAC neighbor-discovery method, small-to-large transfer, and gate-family collision/topology tradeoffs.
The evidence is not yet a fully verified replication package: the most important unresolved items are an independent re-run of a selected final Phase10 transfer experiment and a learned-vs-rule ablation that separates neural learning from structured priors, followed by line-level claim tightening before submission.

## Generated Files

- `06_analysis/paper_tables/research_goal_coverage_audit/requirement_coverage.csv`
- `06_analysis/paper_tables/research_goal_coverage_audit/evidence_inventory.csv`
- `06_analysis/paper_tables/research_goal_coverage_audit/claim_risk_register.csv`
- `06_analysis/paper_tables/research_goal_coverage_audit/raw_bundle_trace_status.csv`
- `06_analysis/paper_tables/research_goal_coverage_audit/manifest.json`
