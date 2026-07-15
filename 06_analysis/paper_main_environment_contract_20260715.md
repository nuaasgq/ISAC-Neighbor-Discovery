# Paper-main training environment contract

## Purpose

This contract replaces the static ideal-ISAC learning gate as the primary training environment. It freezes one defensible operating point before multi-seed training and avoids further method-ablation expansion.

## Environment

- 10 UAVs in a 3500 m by 3500 m planar region; every pair is inside the communication and sensing range filters.
- Gauss-Markov mobility at 3--15 m/s, with 9 m/s mean speed.
- 24 azimuth beams (15 degrees), one elevation cell, one RF chain.
- 300 slots per episode and 5 ms per slot.
- A TX action carries both the HELLO signal and the ISAC observation; there is no standalone sensing action.
- MIMO-OTFS radar-SNR sensing abstraction with anonymous noisy target counts, detection errors, angular errors, and position errors.
- Close-in path loss, normalized directional antenna gain, Rician fading, log-normal shadowing, interference, and SINR handshake decisions.
- Each actor sees only its own state, local measurements/history, and exchanged neighbor/sensing tables. Global state is critic-only during training.

## Frozen main method

- Residual-mask ISAC-MAPPO with a shared recurrent actor and centralized pooled critic.
- Joint TX/RX and beam decisions; beam-conditioned antisymmetric role head.
- Local residual candidate mask is the protocol mechanism under study.
- No behavior cloning, rule-logit residual, handcrafted contention prior, candidate-score prior, rendezvous recommendation, topology-truth action hint, or deterministic role assignment.

## Training campaign

- Three independent training seeds.
- 1000 episodes per seed, 300 environment steps per episode.
- Checkpoints every 100 episodes; per-episode rewards and environment-step coordinates are retained.
- Main training is followed by paired held-out evaluation against blind random, corrected Wang2025, and residual-candidate random execution. These baselines do not require training.

The static ideal setting remains a mechanism/learnability gate and must not be reported as the paper's primary operating environment.
