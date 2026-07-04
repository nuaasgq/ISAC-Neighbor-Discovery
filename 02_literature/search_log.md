# Sprint 1 Search Log

## Date

2026-07-04

## Purpose

Establish whether ISAC-assisted narrow-beam neighbor discovery for fully distributed UAV-UAV networks is a viable and novel direction.

## Search Tracks

### Track A: Directional Neighbor Discovery

Representative queries:

- `"neighbor discovery" "directional antennas" "UAV" "mmWave"`
- `"oblivious neighbor discovery" "airborne" "directional"`
- `"On 2-way neighbor discovery" "directional antennas" DOI`
- `"On Oblivious Neighbor Discovery" "Directional Antennas" "IEEE/ACM Transactions on Networking" DOI`
- `"SkyOrbs" "Fast 3-D Directional Neighbor Discovery" UAV`

Early included sources:

- Chen et al., IEEE/ACM TON 2017.
- Wang et al., Digital Communications and Networks 2021.
- Hong et al., Ad Hoc Networks 2023.
- Zhu et al., IEEE TMC 2024.
- Wang et al., IEEE TVT 2024.

### Track B: ISAC-Assisted Beam Management

Representative queries:

- `"Integrated sensing and communication-assisted beam rendezvous in airborne networks" DOI`
- `"Deep Learning-Based Predictive Bidirectional Beamforming" "ISAC-Enabled UAV Networks" DOI`
- `"Sensing-assisted accurate and fast beam management" "mmWave UAV" DOI`
- `"Seeing Is Not Always Believing" "ISAC-Assisted Predictive Beam Tracking" DOI`

Early included sources:

- Hong et al., Computer Communications 2024.
- Xu et al., IEEE TWC 2026.
- Cui et al., China Communications 2024.
- Cui et al., IEEE WCL 2024.

### Track C: Sensing Prior / Beam Probing Threats

Representative queries:

- `"CommRad" "Context-Aware Sensing-Driven Millimeter-Wave Networks"`
- `"Frame Structure and Protocol Design for Sensing-Assisted NR-V2X Communications"`
- `"Sensing-Assisted Adaptive Beam Probing" "Calibrated Multimodal Priors"`
- `"Enhancing THz/mmWave Network Beam Alignment With Integrated Sensing and Communication"`

Early included sources:

- Li et al., IEEE TMC 2024.
- Jain et al., ACM SenSys 2024.
- Orimogunje et al., IEEE WCL 2026.
- Chen et al., IEEE Communications Letters 2022.

## Preliminary Screening Rules

Include:

- Peer-reviewed or clearly indexed papers on directional neighbor discovery, UAV/airborne DND, ISAC-assisted beam management, or sensing-assisted beam probing.
- Works that may threaten novelty even if not UAV-specific.

Exclude or defer:

- Pure physical-layer waveform design without beam management/discovery connection.
- Pure routing/topology papers without directional discovery.
- Unverified preprints unless they are direct novelty threats.

## Current Interpretation

The direction remains viable if positioned as:

> fully distributed U2U pre-alignment neighbor discovery with ISAC-derived beam-cell occupancy priors and topology-aware finite-time prioritization.

The direction becomes weak if positioned only as:

> ISAC-assisted beam alignment or sensing-assisted beam search.

### Track D: Scalable MARL and Wireless Protocol Threats

Representative queries:

- `"Resource Management in Wireless Networks via Multi-Agent Deep Reinforcement Learning" TWC 2021`
- `"Learning Decentralized Wireless Resource Allocations with Graph Neural Networks" IEEE TSP 2022`
- `"Multi-Agent Reinforcement Learning-Based Distributed Channel Access" JSAC 2022`
- `"Enhanced reinforcement learning-based two-way transmit-receive directional antennas neighbor discovery" 2024`

Early included sources:

- Naderializadeh et al., IEEE TWC 2021.
- Wang, Eisen, and Ribeiro, IEEE TSP 2022.
- Guo et al., IEEE JSAC 2022.
- Wei et al., Ad Hoc Networks 2024/2025.

Interpretation:

Transferable MARL and GNN-based wireless resource allocation are not blank areas. The defensible gap is the combination of ISAC beam-cell prior, pre-alignment U2U narrow-beam neighbor discovery, topology-aware reward, and zero-shot small-to-large UAV swarm deployment.

### Track E: MARL Algorithm Families for Sprint 4

Representative queries:

- `"QPLEX" "Duplex Dueling Multi-Agent Q-Learning" OpenReview`
- `"Qatten" "cooperative multiagent reinforcement learning" arXiv`
- `"HAPPO" "HATRPO" "multi-agent reinforcement learning" arXiv`
- `"Multi-Agent Transformer" "sequence modeling" MARL NeurIPS`
- `"COMA" "Counterfactual Multi-Agent Policy Gradients" arXiv`
- `"Actor-Attention-Critic" "multi-agent reinforcement learning" ICML`
- `"Decomposed Soft Actor-Critic" "cooperative multi-agent reinforcement learning"`
- `"Multi-Agent Transformer" "sequence modeling" MARL NeurIPS`
- `"Multi-Agent Actor-Critic with Hierarchical Graph Attention Network"`
- `"Scalable Neighborhood-Based Multi-Agent Actor-Critic" arXiv`

Early included sources:

- VDN, QMIX, QTRAN/QTRAN++, QPLEX, Qatten, MAVEN, ACE.
- WQMIX, RiskQ, TransfQMix as enhanced value-factorization and risk/transfer candidates.
- IPPO/MAPPO, HAPPO/HATRPO, MAT.
- COMA, MAAC, MASAC / decomposed SAC.
- Hierarchical graph attention actor-critic and local-neighborhood centralized critic variants.

Interpretation:

The MARL part should be framed as an algorithm-family screening and problem-specific architecture design, not as a fixed MAPPO implementation. The likely method novelty is not the base optimizer itself, but the ISAC-aware beam representation, topology-aware credit assignment, uncertainty-aware exploration, and scale-invariant local observation design.

Network-structure note:

Attention critics, graph attention actor-critic, MAT-style sequence modeling, and K-neighborhood critics all suggest the same risk: full centralized critics do not scale cleanly. For this project, critic design should use local graph / top-K / pooling / mean-field summaries so that training on `N<=20` can still produce useful gradients for `N=50/100/200`.
