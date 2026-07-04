# Reviewer Risk Register

Date: 2026-07-05

## P1 Risks

| Risk | Why a reviewer may object | Current mitigation | Manuscript posture |
|---|---|---|---|
| The learning method is not full neural MARL. | The project motivation includes multi-agent RL, but the strongest current evidence is shared-parameter/CEM policy optimization. | `main.tex` explicitly says neural actor-critic/value-decomposition variants are not used as main performance evidence; actor-critic probes are documented separately. | Use `shared-parameter policy optimization`, not full MAPPO/QMIX/GNN-MARL. |
| SkyOrbs is not strictly reproduced. | The baseline is inspired by SkyOrbs but not the full original protocol. | `skyorbs_baseline_reproduction_checklist.md`; baseline text says `SkyOrbs-like`. | Never claim reproduction. Use it as a deterministic 3-D skip-scan reference baseline. |
| Collision cost is high for the proposed policy. | Wider beams and ISAC-guided concentration increase collision attempts in dense settings. | Tables include collision counts; round7/round8 include collision-penalized discovery. | Report raw discovery with collision-aware metrics. State 15 degrees favors raw discovery/connectivity, 10 degrees favors collision-aware efficiency. |
| 3/5-degree beams are weak. | Extremely narrow beams create very sparse alignment opportunities in finite horizon. | Round7 scale/beam grid explicitly includes 3 and 5 degrees as stress regimes. | Write `evaluated over 3--30 degrees`, not `effective over 3--30 degrees`. |
| Abrupt mobility degrades results. | Random-direction and random-waypoint break smooth occupancy evolution. | Round5/round7 mobility results identify this as an applicability boundary. | State the method is strongest when occupancy evolves smoothly relative to the discovery horizon. |
| ISAC model is abstract. | TWC/TCOM reviewers may expect physical-layer sensing equations. | System model treats ISAC as a link-layer occupancy prior with false alarms, misses, angular offsets, range, and staleness; this is deliberate cross-layer abstraction. | Keep physical waveform/estimator outside scope; do not imply hardware-ready sensing. |

## P2 Risks

| Risk | Why a reviewer may object | Current mitigation | Manuscript posture |
|---|---|---|---|
| Tables only show means. | Statistical stability may be questioned. | `statistical_stability_summary.csv` contains mean/std/CI; main captions now say mean over three seeds. | Mention std/CI in supplement/reproducibility tables. |
| Energy efficiency is incomplete. | Recent UAV ND work jointly optimizes delay and power. | `energy_efficiency_extension_plan.md` defines the missing radio-state model. | Use scan/collision efficiency only; keep Joule-level energy as future extension. |
| B=15 looks good but collisions are large. | Raw discovery and connectivity can hide MAC cost. | Round8 B=15 full sweep includes collision counts and collision-penalized metrics. | Present B=15 as raw discovery/connectivity operating point, not universally optimal. |
| `Rs/Rc` saturation may be misread as a physical sensing law. | The simulator only confirms communication-range neighbors. | Manuscript says saturation is in the evaluated communication-neighbor-discovery abstraction. | Keep wording model-internal. |
| Training convergence may be overclaimed. | CEM score curve is empirical and finite. | Section renamed `Training-Score Evolution`; text says not a theoretical guarantee. | Do not use `convergence proof` language. |

## Strongest Reviewer Response Points

- The paper is not claiming oracle sensing: every edge still requires bidirectional handshake.
- Candidate-set refinement ablation is decisive and directly tied to the ISAC mechanism.
- Zero-shot transfer is demonstrated at `N=100` under density-preserving and fixed-area scaling.
- Missing mobility baselines have been filled; SkyOrbs-like and learned no-ISAC policies remain near zero in the same mobility settings.
- Round7/round8 stress tests strengthen the claim boundaries rather than hiding failure modes.
