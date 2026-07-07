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

## Current Closeout Tasks

1. Keep the IEEE main draft and supplement aligned with Phase10 final results.
2. Verify all referenced figures exist and keep a 4:3 aspect ratio.
3. Compile `main.tex` and `supplement.tex`; logs should have no undefined references/citations, overfull boxes, or fatal errors.
4. Commit and push the checked manuscript state.
5. Continue any new experiment only when it addresses a specific paper claim gap.
