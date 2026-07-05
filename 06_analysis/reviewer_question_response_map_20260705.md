# Reviewer Question Response Map - 2026-07-05

This file turns the current evidence package into concise response-ready arguments.

| Likely reviewer question | Short answer | Primary evidence | Boundary to state |
|---|---|---|---|
| Is this a physical-layer ISAC beamforming paper? | No. ISAC is abstracted as an imperfect beam-cell occupancy-prior service exposed to the link layer. | `main.tex` System Model and ISAC Prior; `supplement.tex` Range and Time-Scale Sensitivity. | No waveform, detector, or radar-equation calibration is claimed. |
| Why is this not just blind directional neighbor discovery? | The protocol uses sensed empty/occupied beam-cell feedback to suppress empty cells and refine candidate beams before confirmed handshakes. | `main.tex` Protocol Design; mechanism ablation; range and error figures. | Sensing does not create edges; edges require bidirectional handshakes. |
| Does the method beat SkyOrbs? | The paper does not claim a strict head-to-head win over complete SkyOrbs. It includes a SkyOrbs-inspired 3-D skip-scan baseline under the same information boundary. | `main.tex` Baselines and Limitations; `round8` and `round9` full-baseline figures. | Complete SkyOrbs reproduction is future work. |
| Is this a full MARL contribution? | No. The current learning evidence is shared-parameter protocol tuning, used to tune scalable local decision rules. | `main.tex` Shared-Parameter Protocol Tuning; training curves. | MAPPO/QMIX/GNN MARL are supported by the environment but are not main evidence. |
| Can a policy trained at N=10 transfer to N=100? | In the evaluated single-hop, 600-slot regimes, yes for the useful 10--30 degree and smoother-mobility region. | N=100 transfer tables; scale/beam heatmap; area-scaling figures. | Transfer is not claimed for all mobility models, all beamwidths, or multi-hop routing. |
| What happens at 3-degree beams? | It is an extreme stress/failure-boundary case. ISAC still reduces empty scans and obtains small nonzero discovery, but does not solve connectivity. | `round9_n100_b3_full_baselines_600slot`; supplement 3-degree table and figure. | Write "evaluated over 3--30 degrees", not "effective over 3--30 degrees". |
| Does `Rs > Rc` physically stop helping? | No physical range law is claimed. Marginal protocol gain saturates because only communication-range neighbors can be confirmed as links. | Range grid figure; ISAC abstraction paragraph. | `Rs`, error rates, and angular offsets are protocol abstraction parameters. |
| Why is algebraic connectivity used? | It is a topology-quality proxy linked to consensus convergence and graph connectivity. | Main metrics section and references to Fiedler/Olfati-Saber. | It is not a full consensus-dynamics simulation. |
| Are collisions ignored? | No. Collision count and collision-penalized discovery are reported; dense wider beams can improve raw discovery while increasing collisions. Round12 adds a local collision-aware role-control probe that improves collision-penalized discovery at B=10/B=15. | Main limitations; round11/round12 collision-penalized supplement figures. | Current method is still not a complete collision- and energy-aware MAC optimization. |
| Are the main N=100/B=10/B=15 results seed-stable? | A focused round11 five-seed paired campaign preserves the proposed-vs-baseline raw-discovery ordering at both beamwidths, with 5/5 positive paired deltas versus random, enhanced no-ISAC, candidate-set removal, and one-slot delay. | `round11_paired_seed_campaign_main` endpoint, cumulative, and paired-delta tables/figures. | Use as focused stability evidence; still not a 10+ seed final campaign. |
| Does higher discovery mean MAC efficiency is solved? | No. Round11 shows proposed raw discovery is highest, but B=15 collision-penalized discovery can favor the one-slot-delay variant. Round12 then shows the boundary can be mitigated by collision-aware local role control. | `round11_collision_penalized.png`; `round12_collision_penalized.png`; round11/round12 paired tables. | Role control is promising, but full MAC and energy scheduling remain next-stage protocol problems. |
| Is the 5 ms slot tuned? | Slot-duration sensitivity over 1--20 ms shows similar discovery in the Gauss-Markov N=100 B=10 setting. | `round6_slot_duration_sensitivity`; supplement range/timing figure. | This is still a protocol timescale study, not hardware timing validation. |

## Recommended Manuscript Tone

Use feasibility and mechanism language:

- "The results indicate..."
- "Within the evaluated single-hop finite-horizon regime..."
- "The main useful operating region is..."
- "This stress case bounds the claim..."
- "The implemented SkyOrbs-inspired baseline should not be interpreted as a full reproduction..."

Avoid overclaiming language:

- "solves 3--30 degree beams"
- "full MARL"
- "beats SkyOrbs"
- "physical sensing range law"
- "universal mobility robustness"
