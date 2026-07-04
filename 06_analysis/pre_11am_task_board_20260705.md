# Pre-11:00 Task Board - 2026-07-05

Current time anchor: after commit `ba9c617` (`Add extra-seed stability evidence`).

## Already Done

- IEEEtran main draft compiles cleanly.
- Supplement compiles cleanly.
- Unit tests pass: `25 passed`.
- Paper figures keep 4:3 aspect ratio.
- Main manuscript claim boundaries have been tightened:
  - ISAC is an imperfect occupancy prior, not an oracle.
  - Discovered graph is a finite-horizon neighbor-knowledge graph/cache, not arbitrary active-link connectivity.
  - Learning is shared-parameter protocol tuning, not full neural MARL.
  - SkyOrbs-like baseline is not a strict SkyOrbs reproduction.
- Paired delta summary added:
  - 125 rows.
  - Main N=100/B=10 paired discovery delta vs enhanced no-ISAC: +0.3648.
  - Round10 backup N=100/B=10 extra-seed paired discovery delta: +0.1731.
- Extra seed stability check added:
  - N=100, B=10/15, three new seeds.
  - Qualitative ordering remains positive, but absolute B=10 discovery is seed-sensitive.

## Remaining Before 11:00

1. **Push latest commit when GitHub connection is stable.**
   - Local commit ahead: `ba9c617`.
   - Previous push failed due SSL/TLS handshake, not local git state.

2. **Prepare final status report.**
   - Summarize current evidence, limitations, and exact files.
   - Explicitly state that results are paper-draft-ready but not final TWC submission-ready without more seeds/energy/SkyOrbs reproduction.

3. **Check final PDF artifacts visually if time allows.**
   - Main PDF: `07_paper/ieee_twc_isac_nd/main.pdf`.
   - Supplement PDF: `07_paper/ieee_twc_isac_nd/supplement.pdf`.

4. **Do not start a broad full-factor experiment.**
   - Full Cartesian sweep across N, beamwidth, mobility, errors, baselines would not finish with enough review time.
   - Extra seeds already show scenario sensitivity; more seed work should be planned as a dedicated campaign.

## After 11:00 Research Roadmap

1. **Seed campaign**
   - Promote main N=100/B=10/B=15 comparison from 3 seeds to 10+ seeds.
   - Use paired seed design.
   - Report median, IQR, bootstrap CI, and success probability of connected discovered-neighbor graph.

2. **Collision-aware protocol refinement**
   - Add adaptive Tx/Rx duty control under ISAC candidate concentration.
   - Optimize collision-penalized discovery rather than raw discovery only.
   - Evaluate B=15/30 where raw discovery is high but collisions are large.

3. **Neural MARL method innovation**
   - Keep the ISAC candidate set as an action-mask or top-k beam proposal.
   - Use factorized action heads: mode head, azimuth head, elevation head, candidate/refine head.
   - Compare value-decomposition, actor-critic, and policy-gradient variants under identical transfer tests.
   - Use shared local observations and no undiscovered-neighbor state at execution.

4. **Reference baseline campaign**
   - Decide whether to implement a closer SkyOrbs reproduction.
   - If not, keep current SkyOrbs-like baseline as scan-schedule class reference and avoid direct superiority claims over SkyOrbs.

5. **Physical-layer bridge**
   - Add a concise parameter mapping from sensing SNR/detection threshold to `P_fa`, `P_md`, and cell-offset std.
   - Do not turn the paper into a PHY paper; keep it as a protocol abstraction with calibrated ranges.
