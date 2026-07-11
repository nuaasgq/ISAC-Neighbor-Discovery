# Role-balanced joint MAPPO 100k gate

## Material Passport

- Origin: frozen three-seed training and common-random-number dev20 evaluation
- Date: 2026-07-12
- Verification status: ANALYZED
- Scope: N=10, planar 15-degree codebook, 300 slots, 334 episodes (100,200 environment steps)

## Contract

The decentralized actor uses only local ISAC residual-table observations, local candidate processing, local topology deficit, and received post-handshake tables. The centralized MPNN critic accesses global training state only. Standalone sensing, idle actions, handcrafted action recommendations, rule residual logits, and global execution guidance are disabled. The role-balance term is training-only and has coefficient 0.01.

## Training endpoint

| Training seed | Last-20 discovery (%) | Last-20 TX ratio (%) | Last-20 return/UAV |
|---:|---:|---:|---:|
| 29260711 | 54.44 | 45.23 | 8.921 |
| 29261711 | 48.22 | 35.85 | 8.141 |
| 29262711 | 53.89 | 50.36 | 8.889 |

## Frozen dev20 ablation

Values are mean +/- SD across three independently trained policies. All arms use the same 20 scenario seeds.

| Arm | Discovery (%) | Delay (slots) | TX ratio (%) | Aligned opportunities |
|---|---:|---:|---:|---:|
| A | 52.11 +/- 0.00 | 215.28 | 49.87 | 29.45 |
| B | 51.37 +/- 1.03 | 211.97 | 49.75 | 31.97 |
| C | 50.93 +/- 1.73 | 219.03 | 49.68 | 30.58 |
| D | 50.00 +/- 3.69 | 212.09 | 48.85 | 31.28 |

The learned-role effect with the learned beam policy is D-B = -1.37 pp (1/3 training seeds positive). The factorial interaction is -0.19 pp. The predeclared paper-level role-learning gate requires mean D-B >= 3 pp, at least 2/3 positive seeds, and no arm-level TX collapse outside 35%-65%.

## Decision

**FAIL.** This gate is a development evaluation, not the untouched final holdout. A failure means that no additional transfer matrix or paper-level MARL superiority claim should be launched from this checkpoint. A pass permits one untouched holdout evaluation before broader experiments.

## Artifacts

- `per_training_seed_ablation.csv`
- `aggregate_ablation.csv`
- `factorial_contrasts.csv`
- Eight 4:3 Times New Roman figures in PNG and PDF formats
