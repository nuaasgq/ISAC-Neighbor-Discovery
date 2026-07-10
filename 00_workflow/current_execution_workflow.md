# Current Execution Workflow

This addendum records the active workflow for the MARL+ISAC neighbor-discovery paper track.

## Version-Control Rule

- Repository of record: `https://github.com/nuaasgq/ISAC-Neighbor-Discovery`.
- All follow-up development, including research notes, simulation code, experiment configs, analysis scripts, figures, tables, and IEEE draft updates, must be tracked in this repository.
- After each phase-level artifact is produced, run `git status`, review the changed scope, commit locally, and push to GitHub when network/authentication permits.
- If GitHub push is unavailable, a local Git commit is still required before continuing to the next major phase.
- Do not commit third-party source PDFs, oversized raw experiment dumps, temporary caches, or LaTeX intermediate files unless they are explicitly needed for reproducibility.

## Active Paper Evidence Line

- Training gate: small-scale decentralized environment with `N=10`, 10-degree beams, and 300 slots per episode.
- Transfer experiments remain blocked until the new rendezvous method passes an independent N=10 confirmation campaign. No 3000-slot or N=100 campaign is active.
- Boundary evidence: 3- and 5-degree cases are treated as extremely narrow stress regimes; the archived 30-degree sweep is retained only as historical boundary evidence.
- Main comparisons: uniform random, SkyOrbs-like directional baseline, communication-only MARL without ISAC, ISAC MARL variants, and decentralized gate-family ablations.
- Current mechanism extension: local anonymous position reports are reprojected into the current body frame and processed by a zero-initialized learned ISAC evidence adapter. Hard candidate masks, rule residuals, behavior cloning, and diagnostic bootstrap actions remain disabled.

## Current Gate Status (2026-07-10)

- The initial N=10/B=10/300-slot learnability gate passed on three training seeds and six held-out episodes.
- Full MARL + ISAC adapter: mean discovery rate 2.96%, 5/6 nonzero episodes, 8 total discovered edges.
- Uniform random, Wang ISAC tables, and trained-checkpoint adapter-zero controls: zero discoveries in the same six scenarios.
- This is an exploratory method-development result. It is not sufficient for a TWC claim because absolute discovery remains low and the first seed informed method tuning.
- The next blocking task is robust reciprocal-report formation and phase consistency, followed by an untouched-seed confirmation campaign.

## Discovery-First Metric Reset

- The primary objective is neighbor discovery performance, not collision
  suppression.
- Main evidence must prioritize discovery rate, discovery delay, cumulative
  discovered links, topology quality, and small-to-large transfer.
- Collision count, CPD, and collision-penalized discovery are diagnostic
  metrics only. They may support an overhead discussion when discovery and
  topology performance are comparable, but they cannot be used as the main
  proof that a method is better.
- Wang2025 uses a conflict mechanism, but its main evaluation is based on
  sensing accuracy/success and neighbor-discovery consumed slots. Follow that
  comparison logic for the next experiment round.
- Detailed replanning document:
  `00_workflow/discovery_first_replan_20260709.md`.

## Current Closeout Tasks

1. Audit existing reports and figures, and demote CPD/collision plots to
   diagnostic-only status.
2. Recompute the Wang-style single-RF comparison using discovery-first metrics:
   discovery rate, consumed slots, cumulative discoveries, delay, discovered
   edges, and topology quality.
3. Redesign MARL reward/evaluation around new discoveries, early discoveries,
   topology improvement, empty-scan reduction, and moderate access cost.
4. Keep standalone SENSE disabled; ISAC remains TX-coupled piggyback sensing.
5. Preserve the passed nonzero N=10/B=10 gate while improving reciprocal-report
   and phase opportunity rates; regression to zero is a failed learnability gate.
6. The position-ordered diagnostic has passed the nonzero gate only after
   reprojecting global ISAC position estimates into the current body-frame beam.
   Expose reprojected beam score, rendezvous phase, and role hint to MARL; do not
   hard-code the diagnostic bootstrap into the final actor.
7. Run an untouched-seed N=10 confirmation only after reciprocal scheduling is
   made robust; scale transfer remains blocked until that confirmation passes.
8. Keep every phase-level artifact committed and pushed; if push fails, keep
   the local commit before moving to the next major phase.
