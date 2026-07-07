# Phase10 Paired Significance Validation - 2026-07-07

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Scope: N=100, 3000-slot B=10/B=15 final Phase10 primary ISAC-vs-communication-only comparison

## Predeclared Testing Boundary

- Confirmatory family 1: discovery-rate deltas for `contention_actor` versus `uniform_random`, `skyorbs_like`, `mappo_no_isac`, and `contention_no_isac` at B=10 and B=15.
- Confirmatory family 2: empty-scan-ratio deltas for the same eight paired comparisons.
- Test: exact two-sided paired sign test over identical scenario seeds, excluding zero deltas.
- Multiple comparisons: Holm correction within each confirmatory family at alpha=0.05.
- Supportive only: lambda2 sign tests are reported for topology context but are not used as the confirmatory decision gate.

## Summary

- Confirmatory tests passing Holm-adjusted alpha=0.05: 16/16.
- Overall confirmatory verdict: PASS.
- Discovery deltas are positive in 10/10 to 10/10 paired seeds across the eight primary comparisons.
- Empty-scan deltas are negative in 10/10 to 10/10 paired seeds across the eight mechanism comparisons.

## Primary Discovery Family

| Beam | Control | Mean delta | 95% bootstrap CI | Sign count | Holm p |
|---:|---|---:|---:|---:|---:|
| 10 | uniform_random | 0.3401 | [0.3349, 0.3449] | +10/-0/00 | 0.01562 |
| 10 | skyorbs_like | 0.3398 | [0.3343, 0.3447] | +10/-0/00 | 0.01562 |
| 10 | mappo_no_isac | 0.3424 | [0.3373, 0.3473] | +10/-0/00 | 0.01562 |
| 10 | contention_no_isac | 0.3421 | [0.3368, 0.3470] | +10/-0/00 | 0.01562 |
| 15 | uniform_random | 0.4104 | [0.4028, 0.4182] | +10/-0/00 | 0.01562 |
| 15 | skyorbs_like | 0.4043 | [0.3971, 0.4117] | +10/-0/00 | 0.01562 |
| 15 | mappo_no_isac | 0.4211 | [0.4136, 0.4286] | +10/-0/00 | 0.01562 |
| 15 | contention_no_isac | 0.4189 | [0.4117, 0.4264] | +10/-0/00 | 0.01562 |

## Mechanism Empty-Scan Family

| Beam | Control | Mean delta | 95% bootstrap CI | Sign count | Holm p |
|---:|---|---:|---:|---:|---:|
| 10 | uniform_random | -0.8478 | [-0.8488, -0.8469] | +0/-10/00 | 0.01562 |
| 10 | skyorbs_like | -0.8480 | [-0.8490, -0.8471] | +0/-10/00 | 0.01562 |
| 10 | mappo_no_isac | -0.8476 | [-0.8484, -0.8468] | +0/-10/00 | 0.01562 |
| 10 | contention_no_isac | -0.8476 | [-0.8483, -0.8469] | +0/-10/00 | 0.01562 |
| 15 | uniform_random | -0.8154 | [-0.8180, -0.8127] | +0/-10/00 | 0.01562 |
| 15 | skyorbs_like | -0.8113 | [-0.8138, -0.8089] | +0/-10/00 | 0.01562 |
| 15 | mappo_no_isac | -0.8165 | [-0.8185, -0.8144] | +0/-10/00 | 0.01562 |
| 15 | contention_no_isac | -0.8163 | [-0.8183, -0.8142] | +0/-10/00 | 0.01562 |

## Statistical Fallacy Scan

- Simpson's paradox: checked; no subgroup reversal claim is made because the report stratifies by beamwidth.
- Ecological fallacy: checked; inference unit is scenario-seed episode, not individual UAV fairness.
- Berkson's paradox: checked; no selected-success-only sample is used.
- Collider bias: checked; no post-treatment control variable is introduced in the sign tests.
- Base-rate neglect: checked; discovery and empty-scan rates are paired directly, not diagnostic predictive values.
- Regression to the mean: checked; no pre/post extreme selection design is used.
- Survivorship bias: checked; all archived final evaluation episodes in the paired source files are included.
- Look-elsewhere effect: addressed by predeclared families and Holm correction.
- Garden of forking paths: bounded by fixed treatment/control sets, fixed metrics, and a written test boundary.
- Correlation-causation fallacy: not applicable to simulator treatment assignment, but hardware/generalization causality is not claimed.
- Reverse causality: not applicable to simulator treatment assignment.

## Interpretation Boundary

These tests strengthen the primary simulator claim that ISAC-assisted contention actor evaluation improves discovery and reduces empty scans versus communication-only controls under paired scenario seeds. They do not prove significance for every gate-family operating point, every mobility model, every beamwidth stress case, or any hardware-calibrated PHY implementation.

## Generated Figures

- `06_analysis/figures/marl/p10_paired_significance_primary/phase10_primary_discovery_paired_deltas.png`
- `06_analysis/figures/marl/p10_paired_significance_primary/phase10_empty_scan_paired_deltas.png`

## Source File Hashes

| Path | SHA256 |
|---|---|
| `06_analysis/paper_tables/marl/phase9_fiveway_n100_b10_3000slot_10ep_stoch_all_methods/marl_transfer_eval_rows.csv` | `00f82b9de91200fcbeda904da83992cb2112224369c7a9df02e6e9797f3df0f4` |
| `06_analysis/paper_tables/marl/phase9_fiveway_n100_b15_3000slot_10ep_stoch_all_methods/marl_transfer_eval_rows.csv` | `576ad72784f05e2c891eb8f5a94f66e029246415172073e69bd0a9d2a82cb4f4` |
