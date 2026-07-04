# Sensing-Assisted Accurate and Fast Beam Management for Cellular-Connected mmWave UAV Network

## Metadata

- Authors: Yanpeng Cui, Qixun Zhang, Zhiyong Feng, Qin Wen, Ying Zhou, Zhiqing Wei, Ping Zhang
- Venue: China Communications, 2024
- DOI: https://doi.org/10.23919/JCC.ea.2023-0140.202401
- Local PDF: user-provided, not committed to repository

## WHY

Beam management, including initial access and beam tracking, suffers from high delay and low alignment accuracy in cellular-connected mmWave UAV networks.

## HOW

The paper combines ISAC, computer vision, EKF-based beam tracking/prediction, and dual identity association to distinguish multiple UAVs in dynamic environments.

## WHAT

Real-world experiments and simulations show improvements in IA delay, association accuracy, tracking error, and communication performance.

## System Assumptions

- Cellular-connected UAV network.
- BS-side sensing and computer vision support.
- Multiple UAV identities are associated using sensing/visual information.
- Main scope is beam management, not distributed neighbor discovery.

## Relevance to This Project

Strong evidence that sensing can accelerate UAV beam management and initial access. It also clarifies that our paper must avoid a BS-centric problem statement.

## Gap

Does not address UAV-UAV fully distributed discovery, no-center coordination, no pre-alignment neighbor identity, or topology-quality-driven discovery.
