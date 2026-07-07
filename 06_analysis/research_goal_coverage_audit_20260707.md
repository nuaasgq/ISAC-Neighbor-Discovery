# Research Goal Coverage Audit - 2026-07-07

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Scope: current MARL+ISAC neighbor-discovery manuscript evidence chain

## Summary

- Requirements checked: 18.
- Status counts: PASS=13, CAUTION=5, OPEN=0.
- Final Phase10 method rows: 18 across beams [10.0, 15.0].
- Training trace: 3 runs, 90000 step rows, 300 episode rows.
- Method trace completeness: 18/18.
- Artifact manifest: 142 artifacts, 0 missing.

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
| R10 | Range abstraction | PASS | main+supplement+theory_note | Final Phase10 rows use Rc=900 m and Rs=900 m as a matched-support single-hop setting; range sweeps cover Rc/D=0.65;0.85;1.05 and Rs/Rc=0.5;0.75;1.0;1.25. |
| R11 | Time-scale assumption | PASS | main+supplement | Slot-duration values in stability summary: 1.0;10.0;100.0;20.0;5.0. |
| R12 | Statistical reliability | CAUTION | analyzed_not_verified | Final transfer episodes range 10-20; stability rows=345 with main rows=126. |
| R13 | Reproducibility trace | PASS | main | Method trace rows complete: 18/18. |
| R14 | Figure and artifact integrity | PASS | artifact_hash | Artifact manifest reports 142 artifacts and 0 missing paths; training manifest lists 19 training/resource figures. |
| R15 | Submission wording boundary | CAUTION | reviewer_audit | Existing readiness review records 195 supplement/supplement-stress rows and flags SkyOrbs-like/reproduction limits. |
| R16 | Independent reproduction | PASS | verified_partial_rerun | Independent stochastic re-run for gated_contention_actor at B=10 has status_counts={'MATCH': 5}; this verifies one key final-transfer point, not the full campaign. |
| R17 | Learned component ablation | CAUTION | focused_ablation_mixed | Focused B=10/N=100/3000-slot learned-component ablation covers labels=['random_weights_full', 'trained_full', 'trained_no_candidate_mask', 'trained_no_rule_residual', 'zero_weights_rule_only']. Results separate learned weights from rule priors, but support a collision-efficiency claim rather than universal learned-policy dominance. |
| R18 | Raw bundle availability | CAUTION | local_trace_not_git_archive | Local method trace paths exist for 28/28 raw manifests and 8/8 checkpoints; 0/36 raw trace files are tracked by Git. |

## Highest-Value Remaining Work

- P2 `CAUTION`: Cover narrow beamwidths around 3-15 degrees, with final main transfer at 10->15 degrees. Next: If reviewers demand full stress coverage, rerun B=3/B=5 with the final Phase10 method set; B=30 is intentionally excluded from the final line.
- P2 `CAUTION`: Provide multi-seed/statistical summaries rather than single-run-only claims. Next: For stronger statistical evidence, extend independent re-runs to additional method/beam pairs or add a paired significance protocol with a predeclared correction boundary.
- P2 `CAUTION`: Keep claims aligned with simulator, protocol-level ISAC abstraction, and approximate literature baseline scope. Next: Before submission, run a line-level IEEE style and claim-strength pass.
- P2 `CAUTION`: Separate learned actor contribution from strong rule priors, residual logits, candidate masks, and decentralized gates. Next: Use conservative wording: learned weights suppress collisions versus random/zero-weight policies, while candidate masking and rule residuals define the discovery/collision/empty-scan tradeoff. Extend to B=15 or more seeds only if needed.
- P2 `CAUTION`: Retain a local or archived raw-result bundle with manifests and checkpoint hashes for the final Phase10 evidence line. Next: Before submission or external release, decide whether to archive checkpoints or publish a separate checksum manifest for raw state_dict files.
- P2 `CAUTION`: Literature-inspired SkyOrbs-like baseline may be challenged as not a faithful reproduction. Next: Either retain explicit caveat or implement a stricter reproduction before submission.

## Boundary Interpretation

The current evidence is strong enough to support a manuscript draft centered on a real MAPPO-style MARL+ISAC neighbor-discovery method, small-to-large transfer, and gate-family collision/topology tradeoffs.
The evidence is still not a fully verified replication package: independent re-run coverage is partial, learned-component evidence is a focused mixed ablation, and full-campaign reproducibility plus broader ablation seeds should remain bounded claims.

## Generated Files

- `06_analysis/paper_tables/research_goal_coverage_audit/requirement_coverage.csv`
- `06_analysis/paper_tables/research_goal_coverage_audit/evidence_inventory.csv`
- `06_analysis/paper_tables/research_goal_coverage_audit/claim_risk_register.csv`
- `06_analysis/paper_tables/research_goal_coverage_audit/raw_bundle_trace_status.csv`
- `06_analysis/paper_tables/research_goal_coverage_audit/manifest.json`
