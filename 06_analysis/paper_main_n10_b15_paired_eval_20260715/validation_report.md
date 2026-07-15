# Paper-main N=10/B=15 paired evaluation

## Contract

- 50 common held-out scenarios: seeds 79260715--79260764.
- N=10, planar Gauss-Markov mobility, 15-degree beams, one RF chain, 300 slots at 5 ms/slot.
- Noisy-count MIMO-OTFS sensing abstraction and close-in Rician/SINR communication PHY.
- The MAPPO result averages three independently trained policies within each scenario before paired statistics.

## Main results

| Method | Slot 50 | Slot 100 | Slot 150 | Slot 200 | Final | Mean delay | TX ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Blind random | 3.47% | 7.20% | 10.40% | 13.69% | 19.56% | 269.68 | 50.03% |
| Wang2025 | 7.02% | 21.07% | 33.60% | 42.67% | 55.78% | 209.14 | 49.97% |
| Residual candidate random | 6.13% | 20.36% | 31.91% | 41.73% | 55.47% | 211.49 | 49.99% |
| Residual-mask MAPPO | 10.49% | 21.04% | 27.44% | 34.12% | 47.39% | 220.81 | 30.40% |

## Paired conclusions

- MAPPO minus Blind random: 27.84%; 95% CI [25.23%, 30.44%], p=1.43e-26, W/T/L=50/0/0.
- MAPPO minus Wang2025: -8.39%; 95% CI [-10.89%, -5.88%], p=1.75e-08, W/T/L=10/0/40.
- MAPPO minus Residual candidate random: -8.07%; 95% CI [-10.59%, -5.55%], p=4.92e-08, W/T/L=8/2/40.

## Interpretation

MAPPO reaches 10.49% by slot 50 and initially exceeds Wang by 3.47%. The gap reverses by slot 200, where MAPPO minus Wang is -8.55%.

The main policy therefore learns useful early beam/role prioritization, but it does not sustain discovery. Its mean TX ratio is about 30%, versus 50% for all three baselines, which reduces late bidirectional rendezvous opportunities. The present campaign supports learnability and a gain over blind search, but it does not support superiority over Wang or the same residual candidate mechanism with random TX/RX and beam execution.

This result should be treated as a method diagnosis, not selected-seed evidence. No further scale-transfer experiment is justified until the decentralized role policy avoids the persistent RX bias without rule-forced execution.
