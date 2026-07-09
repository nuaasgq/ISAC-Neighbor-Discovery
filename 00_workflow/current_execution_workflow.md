# Current Execution Workflow

This addendum records the active workflow for the MARL+ISAC neighbor-discovery paper track.

## Version-Control Rule

- Repository of record: `https://github.com/nuaasgq/ISAC-Neighbor-Discovery`.
- All follow-up development, including research notes, simulation code, experiment configs, analysis scripts, figures, tables, and IEEE draft updates, must be tracked in this repository.
- After each phase-level artifact is produced, run `git status`, review the changed scope, commit locally, and push to GitHub when network/authentication permits.
- If GitHub push is unavailable, a local Git commit is still required before continuing to the next major phase.
- Do not commit third-party source PDFs, oversized raw experiment dumps, temporary caches, or LaTeX intermediate files unless they are explicitly needed for reproducibility.

## Active Paper Evidence Line

- Training setting: small-scale decentralized MARL training with `N=10`, 10-degree beams, and 300 slots per episode.
- Transfer setting: zero-shot large-scale evaluation with `N=100`, 3000 slots, and final main beamwidths of 10 and 15 degrees.
- Boundary evidence: 3- and 5-degree cases are treated as extremely narrow stress regimes; the archived 30-degree sweep is retained only as historical boundary evidence.
- Main comparisons: uniform random, SkyOrbs-like directional baseline, communication-only MARL without ISAC, ISAC MARL variants, and decentralized gate-family ablations.
- Current mechanism extensions: `trust_gated_isac_tables` for guarded neighbor/sensing-table fusion, and Budgeted ISAC expert gate behavior cloning for MARL access-control learning.

## Current Closeout Tasks

1. Run the Budgeted expert gate BC sweep at N=10, B=10, 300-slot training episodes.
2. Transfer-evaluate the resulting checkpoints at N=100, B=10/B=15, 3000 slots.
3. Add protocol baselines for `trust_gated_isac_tables`, `improved_rl_isac_tables`, `wang2025_isac_tables`, `budgeted_collision_aware_isac`, and `uniform_random`.
4. Promote learned-gate claims only if CPD or collision count improves without collapsing raw discovery.
5. Keep every phase-level artifact committed and pushed; if push fails, keep the local commit before moving to the next major phase.
