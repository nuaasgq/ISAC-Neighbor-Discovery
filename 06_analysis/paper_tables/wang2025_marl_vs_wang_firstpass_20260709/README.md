# Wang2025 MARL-vs-Wang Matrix

This corrective campaign compares real trained MARL checkpoints against Wang2025-style baselines under the same single-RF matrix.

- Node counts: [10, 20, 30, 40, 50]
- Train node count: 10
- Train slots: 300
- Eval slots: 200
- Eval episodes: 3
- MARL action space: TX/RX/IDLE only; standalone SENSE is disabled.
- ISAC feedback: TX-coupled piggyback sensing.

Files:

- `per_episode_summary.csv`
- `aggregate_metrics.csv`
- `manifest.json`

Current top-line rows:

- Budgeted ISAC rule: discovery=0.5888, CPD=0.1914, collisions=2560.7, lambda2=18.7574
- MARL, no ISAC, TX/RX/IDLE: discovery=0.0000, CPD=0.0000, collisions=0.0, lambda2=0.0000
- MARL + TX-coupled ISAC + gate BC, TX/RX/IDLE: discovery=0.0003, CPD=0.0003, collisions=0.0, lambda2=0.0000
- MARL + TX-coupled ISAC, TX/RX/IDLE: discovery=0.0528, CPD=0.0463, collisions=174.3, lambda2=-0.0000
- Uniform Random: discovery=0.0090, CPD=0.0090, collisions=0.0, lambda2=-0.0000
- Wang-like + neighbor table: discovery=0.5401, CPD=0.0827, collisions=6796.3, lambda2=16.0006
- Wang-like ISAC, no table exchange: discovery=0.5170, CPD=0.1177, collisions=4201.7, lambda2=15.5967
- Wang-like + sensing table: discovery=0.5361, CPD=0.0819, collisions=6815.0, lambda2=17.2360
