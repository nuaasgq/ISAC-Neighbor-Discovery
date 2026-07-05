# SkyOrbs-Like Baseline Reproduction Checklist

Date: 2026-07-05

## Current Manuscript Position

The current simulator includes `skyorbs_like_skip_scan` as a deterministic directional scanning baseline. This is intentionally described as a SkyOrbs-like baseline, not a strict reproduction of the full SkyOrbs protocol.

This wording should remain until the checklist below is completed.
The supplement now includes a baseline-scope note, and `06_analysis/skyorbs_like_baseline_scope_appendix_20260705.md` records the exact non-reproduction boundary.

## Items Required for a Strict Baseline Claim

1. Confirm the original 3D beam-indexing and scanning order.
2. Confirm whether the original protocol assumes synchronized slots, known beam-codebook orientation, or common scan cycles.
3. Confirm antenna switching and listening/transmitting schedule details.
4. Match the original node motion assumptions or clearly state deviations.
5. Match the original communication range, angular coverage, and discovery success definition as closely as the paper allows.
6. Reproduce at least one published SkyOrbs trend or table in an isolated validation run.
7. Report deviations in a baseline appendix before using the phrase "SkyOrbs reproduction". A scope appendix now exists, but it documents non-reproduction rather than completing strict reproduction.

## Current Fair-Use Wording

Use:

> a SkyOrbs-like deterministic 3D skip-scan baseline

Use:

> a directional scan-scheduling baseline inspired by SkyOrbs

Avoid:

> the SkyOrbs baseline

Avoid:

> we reproduce SkyOrbs

## Why This Matters

The paper's primary comparison should not depend on overstating a reference baseline. The current data already include stronger internal ablations and no-ISAC baselines. The SkyOrbs-like curve is useful for directional-scan context, but the strongest claims should be framed around:

- random blind scanning,
- communication-only learning,
- improved communication-only learning,
- ISAC-assisted learning,
- and mechanism ablations inside the proposed protocol.
