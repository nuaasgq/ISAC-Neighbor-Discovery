# Pre-Submission Gap Triage

Date: 2026-07-05

This triage decides what is already sufficient for a bounded IEEE TWC/TCOM-style draft, what should be fixed by writing only, and what would require new experiments before external submission.

## Paper-Ready Now

| Item | Current evidence | How to use |
|---|---|---|
| Cross-layer problem positioning | Main manuscript models ISAC as an imperfect link-layer occupancy prior, not a PHY oracle. | Use as the central TWC/TCOM angle. |
| Main protocol mechanism | Candidate-set removal drops N=100/B=10 discovery from 0.3655 to 0.0313. | Claim candidate-beam refinement is the dominant observed mechanism at the main operating point. |
| Main baseline comparison | Round14 ten-seed N=100/B=10 proposed discovery 0.3652 and lambda2 13.2595; random, SkyOrbs-like, learned no-ISAC, and enhanced no-ISAC stay near zero, with 10/10 positive paired discovery deltas versus all four controls. | Keep as the main table. |
| N=10 to N=100 transfer | Tested under density and fixed-area scaling, with strongest behavior for 10--30 degree beams. | Claim bounded zero-shot transfer, not universal scalability. |
| Range, slot, and sensing-error sensitivity | Rc/D, Rs/Rc, 1--20 ms slots, false alarms, misses, and angular errors are covered. | Use to defend modeling assumptions and moderate-error robustness. |
| Collision-aware refinement | Round13 ten-seed probe improves collision-penalized discovery and assumed discoveries per joule at B=10/B=15. | Keep mostly in supplement; main text gets a one-sentence boundary mitigation. |
| Figure package | Main and supplement figures are 4:3, Times-style, and compiled into IEEE PDFs. | Good enough for an internal paper draft and advisor review. |

## Text-Only Before External Submission

| Task | Why it matters | Action |
|---|---|---|
| Tighten abstract/contributions around protocol-layer novelty | Avoid reviewer expectation of PHY beamforming or full MARL. | Use `round13_twc_tcom_revision_route_20260705.md` as the source. |
| Keep SkyOrbs wording uniform | Avoid implying strict reproduction or head-to-head superiority. | Always write `SkyOrbs-like`; state not a strict reproduction. |
| Make small-sample statistics visible | Main tables are mean-heavy; supplement has std/CI. | Add `mean over seeds; std/CI in supplement` wherever a table/figure could be misread. |
| Keep energy wording diagnostic | Round13 uses default powers but no platform calibration. | Write `assumed radio-state accounting`, never energy-optimality. |
| Clarify discovered graph versus active communication graph | Prevent topology metric overclaiming. | Continue using `discovered-neighbor graph/cache`. |
| Cite-backed PHY-to-protocol mapping | TWC/TCOM reviewers may ask why `Rs`, `Rc`, `P_fa`, `P_md`, angular error, and staleness are valid protocol inputs. | Main text and supplement now include a standards/literature-backed mapping; keep it as an abstraction, not calibration. |

## New Evidence Before External Submission

| Priority | Experiment or artifact | Rationale | Suggested scope |
|---:|---|---|---|
| P1 | Strict SkyOrbs reproduction or deeper baseline appendix | The current baseline is SkyOrbs-like only. | Either implement the missing original details or write a precise non-reproduction appendix. |
| P2 | Platform-calibrated radio-state power sensitivity | Energy accounting uses assumed powers. | Add a small sensitivity grid over plausible TX/RX/sensing/idle powers; do not change the main claim. |
| P3 | Calibrated PHY-to-ISAC parameter appendix | The cite-backed parameter mapping is now present, but a reviewer may still ask how the chosen false alarm/miss/angular-error values map to a named waveform, SNR, detector, aperture, and RCS model. | Optional next upgrade: add a compact illustrative link-budget/radar-equation appendix, not a full waveform design. |
| P4 | Collision-aware MAC extension | Round13 role control mitigates but does not solve collision-heavy regimes. | Extend role control to 30-degree and high-density settings if the paper is pushed toward MAC. |

## Do Not Prioritize Now

- A full MAPPO/QMIX/GNN-MARL rewrite: current evidence does not require it for a protocol paper.
- Another full Cartesian sweep over every node count, beamwidth, mobility, range, and error profile.
- Claiming 3-degree or 5-degree beams are solved.
- Claiming complete SkyOrbs superiority.
- Claiming platform-calibrated energy optimality.

## Submission Framing

The current package is strong enough for an internal TWC/TCOM-style draft and advisor review.
For external submission, the safest next evidence upgrade is now a precise baseline-scope appendix or calibrated PHY-to-ISAC parameter appendix; the 10-seed main-table confirmation is complete in round14 and the cite-backed PHY-to-protocol mapping has been added.
