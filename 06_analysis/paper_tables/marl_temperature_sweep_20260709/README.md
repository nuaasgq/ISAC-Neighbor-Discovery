# MARL Temperature Sweep

This sweep reuses the Wang-style first-pass MARL checkpoints and changes only stochastic deployment temperature.

- Temperatures: [0.7, 1.0, 1.3, 1.6, 2.0]
- Node counts: [50]
- Eval episodes: 3
- Slots: 200
- Standalone SENSE remains disabled.

Best CPD rows by method:

- MARL, no ISAC: temp=2.00, discovery=0.0035, CPD=0.0035, collisions=0.0, lambda2=0.0000
- MARL + TX-coupled ISAC: temp=0.70, discovery=0.2762, CPD=0.1717, collisions=745.7, lambda2=5.4821
- MARL + TX-coupled ISAC + gate BC: temp=0.70, discovery=0.3135, CPD=0.0748, collisions=3969.0, lambda2=6.4521
