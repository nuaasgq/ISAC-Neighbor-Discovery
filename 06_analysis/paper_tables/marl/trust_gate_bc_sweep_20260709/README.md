# Trust-Gated Table and Budgeted Gate BC Sweep

Generated: 2026-07-09T11:20:39
Status: partial

## Protocol Summary

| Protocol | Episodes | Discovery | Collisions | CPD |
|---|---:|---:|---:|---:|
| budgeted_collision_aware_isac | 5 | 0.7022 | 1300.8000 | 0.5563 |
| improved_rl_isac_tables | 5 | 0.7438 | 9247.0000 | 0.2608 |
| trust_gated_isac_tables | 5 | 0.7446 | 9516.4000 | 0.2569 |
| uniform_random | 5 | 0.0025 | 0.0000 | 0.0025 |
| wang2025_isac_tables | 5 | 0.6534 | 1886.0000 | 0.4739 |

## Budgeted Expert BC Training

| Run | Episodes | BC weight | Final discovery | Final collisions | Final BC loss | Complete |
|---|---:|---:|---:|---:|---:|---|
| train_n10_b10_balgate_bc0p15_budgeted_100ep_300slot_seed20260751 | 100 | 0.1500 | 0.2889 | 593.0000 | 3.0946 | True |
| train_n10_b10_balgate_bc0p3_budgeted_100ep_300slot_seed20260751 | 11 | 0.3000 | 0.5333 | 755.0000 | 6.6284 | False |

## Transfer Evaluation

No transfer rows found yet.
