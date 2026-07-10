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
- Transfer experiments are blocked until a trained policy beats the N=10 common-environment baselines. No 3000-slot or N=100 campaign is active.
- Boundary evidence: 3- and 5-degree cases are treated as extremely narrow stress regimes; the archived 30-degree sweep is retained only as historical boundary evidence.
- Main comparisons: uniform random, SkyOrbs-like directional baseline, communication-only MARL without ISAC, ISAC MARL variants, and decentralized gate-family ablations.
- Current mechanism extensions: `trust_gated_isac_tables` for guarded neighbor/sensing-table fusion, and Budgeted ISAC expert gate behavior cloning for MARL access-control learning.

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
5. Establish nonzero reciprocal rendezvous opportunities at N=10/B=10 before
   starting MARL; zero-discovery environments are a failed learnability gate.
6. Train small-scale MARL at N=10 only after the random, Wang, and rule-ISAC
   baselines are reproducible under the canonical SINR-PHY configuration.
7. Keep every phase-level artifact committed and pushed; if push fails, keep
   the local commit before moving to the next major phase.
