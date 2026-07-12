# N=10, B=15-Degree Static Ideal-ISAC Protocol Baselines

## Material Passport

- Artifact type: non-trained protocol baseline evaluation
- Verification status: ANALYZED
- Configuration: `05_simulation/configs/n10_b15_static_ideal_isac.yaml`
- Raw results: `05_simulation/results_raw/n10_b15_static_ideal_protocol_baselines_50ep`
- Evaluation: 50 episodes, 300 slots per episode, common seed 61270713

## Results

| Protocol | Discovery rate | Mean censored delay | `B/A` | `O/B` | Empty-scan ratio | TX ratio |
|---|---:|---:|---:|---:|---:|---:|
| Uniform random | 21.96% | 265.06 slots | 0.169% | 50.10% | 72.97% | 50.00% |
| Wang2025 ISAC tables | 94.13% | 128.11 slots | 1.583% | 49.08% | 14.14% | 49.91% |

## Interpretation Boundary

Neither baseline is trained. The Wang protocol's gain comes from its sensing-table beam pool
and table exchange while its TX/RX role remains random. The result establishes a demanding
rule-based reference for the later MAPPO checkpoints. A learned method that only beats uniform
random but remains substantially below Wang does not yet support a strong method claim in this
static ideal-ISAC setting.
