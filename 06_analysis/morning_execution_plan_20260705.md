# Morning Execution Plan Toward 11:00

Date: 2026-07-05
Deadline: 2026-07-05 11:00 Asia/Shanghai

## Objective

Make the current ISAC-assisted UAV-UAV narrow-beam neighbor-discovery study defensible as a paper draft package: compact IEEE manuscript, traceable figures, bounded claims, reproducible experiment archives, and clear supplement plan.

## Priority Order

1. Claim boundary and citation cleanup.
   - Keep the contribution as a cross-layer link-layer neighbor-discovery protocol.
   - Use `SkyOrbs-like`, not `SkyOrbs reproduction`.
   - Use `shared-parameter policy optimization`, not full MAPPO/QMIX/GNN MARL.
   - Treat 3/5-degree beams and random-direction/waypoint mobility as stress regimes.

2. Evidence completeness.
   - Main evidence: N=100 transfer, baseline comparison, range grid, error grid, mechanism ablation, training-score evolution.
   - Supplement evidence: broad 3--30 degree beam sweep, N=10--100 scaling, round7/round8 mobility boundary, B=15 error profiles, std/CI tables.

3. Verification gates.
   - `python -m pytest 05_simulation/tests`
   - `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `07_paper/ieee_twc_isac_nd`
   - Figure archive count and 4:3 aspect check.
   - `git status --short --branch`

4. Paper polish.
   - Remove unsupported superiority language.
   - Make all numerical claims traceable to archived CSV files.
   - Keep limitations explicit enough for TWC/TCOM review.

## Stop Condition for This Morning

Stop adding new experiments once the manuscript compiles cleanly and the remaining risks are mainly scope limitations rather than missing required evidence. At that point, spend time on writing, supplement organization, and reviewer-risk notes instead of launching new sweeps.
