# Phase10 Gated Contention Transfer Results

Created: 2026-07-06

## Scope

This note summarizes the first formal N=100 transfer evaluation of the gated contention actor. The policy was trained at N=10 and B=10 deg, then evaluated at N=100 with 3000 slots and 10 stochastic episodes. Communication and sensing ranges were both set to 900 m for the single-hop stress case, matching the Phase9 comparison setting.

## Artifacts

- Raw evaluation:
  - `05_simulation/results_raw/marl_campaign/p10_gate31_n100/b10`
  - `05_simulation/results_raw/marl_campaign/p10_gate31_n100/b15`
- Phase10 transfer summary:
  - `06_analysis/paper_tables/marl/p10_gate31_n100_b10_b15_3000slot_10ep_stoch/marl_transfer_summary.csv`
- Phase9 + Phase10 method comparison:
  - `06_analysis/paper_tables/marl/p10_gate31_vs_phase9_b10_b15_method_comparison/marl_method_comparison.csv`
- Figures:
  - `06_analysis/paper_figures/marl/p10_gate31_n100_b10_b15_3000slot_10ep_stoch`
  - `06_analysis/paper_figures/marl/p10_gate31_vs_phase9_b10_b15_method_comparison`

All generated PNG figures are 1920 x 1440 pixels, i.e., 4:3.

## Key Comparison

| Beamwidth | Method | Discovery | CPD | Lambda2 | Collisions |
|---:|---|---:|---:|---:|---:|
| 10 deg | contention_actor | 0.3429 | 0.2263 | 14.3814 | 2563.7 |
| 10 deg | gated_contention_actor | 0.3020 | 0.2353 | 11.9043 | 1403.3 |
| 15 deg | contention_actor | 0.4233 | 0.1387 | 19.8887 | 10226.9 |
| 15 deg | gated_contention_actor | 0.3618 | 0.2114 | 16.1730 | 3535.0 |

## Interpretation

The gated contention actor improves collision-penalized discovery rate and substantially reduces collisions, but it does not dominate the previous contention actor on raw discovery rate or algebraic connectivity.

This supports a narrower and defensible claim:

> ISAC-assisted gated contention control improves finite-time effective discovery under contention-heavy N=100 transfer by suppressing redundant transmissions and collision-heavy alignment attempts.

It does not yet support a broad "all metrics are better" claim. The remaining technical weakness is that the access gate is somewhat conservative, especially at B=10 deg, where lambda2 drops from 14.3814 to 11.9043 despite a small CPD gain.

## Next Actions

- Complete seed20260733 training and generate step-axis training curves from all completed seeds.
- Tune the gate to recover topology quality:
  - increase topology-need contribution,
  - reduce collision-pressure penalty once local candidate confidence is high,
  - add a connectivity-recovery term or annealed gate threshold.
- Re-test at N=100/B=10 and B=15 after the topology-recovery adjustment.
- Then run the broader N/B transfer grid only for the best gated variant.

