# SkyOrbs-Like Baseline Scope Appendix

Date: 2026-07-05

Purpose: define the exact comparison boundary for the `skyorbs_like_skip_scan` baseline.
This is a baseline-scope appendix, not a strict reproduction claim.

## Current Baseline Role

The implemented baseline is a deterministic 3-D skip-scan reference inspired by SkyOrbs-style UAV directional neighbor discovery.
It is used to represent a communication-only directional scan-scheduling family under the same simulator information boundary as the other baselines.

## What Is Matched in the Current Simulator

| Aspect | Current implementation |
|---|---|
| Beam space | Same 3-D beam codebook and beamwidth as the evaluated protocols. |
| Information boundary | No ISAC occupancy prior, no undiscovered-neighbor state, no global topology knowledge. |
| Slot horizon and mobility | Same finite horizon, same UAV mobility trace, same range settings, and same scenario seeds as the proposed protocol. |
| Comparison metrics | Same discovery rate, empty-scan ratio, algebraic connectivity, delay, collision, and paired-seed reporting where available. |
| Reproducibility role | Included in round14 ten-seed N=100/B=10 main-table stability and round8/round9 supplement stress checks. |

## What Is Not Claimed

The current implementation does not claim to match every original SkyOrbs protocol detail.
In particular, it does not yet verify the original paper's exact beam-indexing order, scan-cycle synchronization assumptions, antenna switching schedule, motion assumptions, handshake details, or a published SkyOrbs trend/table in an isolated reproduction experiment.

## Required Wording

Use:

- `SkyOrbs-like skip-scan`
- `deterministic 3-D directional skip-scan reference`
- `inspired by SkyOrbs`

Avoid:

- `SkyOrbs reproduction`
- `complete SkyOrbs baseline`
- `we outperform SkyOrbs`

## Reviewer Response

If asked why SkyOrbs is not strictly reproduced, the best response is:

> We include a SkyOrbs-like deterministic 3-D skip-scan reference to represent communication-only directional scan scheduling under the same simulator information boundary. We do not claim a strict reproduction of the complete SkyOrbs protocol. The main claim is instead supported by random/no-ISAC learned baselines, a ten-seed main-table comparison, and mechanism ablations that isolate the ISAC candidate-set contribution.

## Remaining Upgrade

A strict reproduction would require:

1. Reconstructing the original beam-indexing and scan order.
2. Matching synchronization, listening/transmitting, and antenna-switching details.
3. Matching the original motion/range/discovery definitions or reporting deviations.
4. Reproducing at least one published SkyOrbs trend before using strict head-to-head language.
