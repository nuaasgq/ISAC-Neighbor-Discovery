# Seeing Is Not Always Believing: ISAC-Assisted Predictive Beam Tracking in Multipath Channels

## Metadata

- Authors: Yanpeng Cui, Qixun Zhang, Zhiyong Feng, Qin Wen, Zhiqing Wei, Fan Liu, Ping Zhang
- Venue: IEEE Wireless Communications Letters, 2024
- DOI: https://doi.org/10.1109/LWC.2023.3303949
- Local PDF: user-provided, not committed to repository

## WHY

Existing ISAC-assisted predictive beamforming/tracking often relies on LoS channel assumptions. In multipath channels, the angle inferred from radar echoes may not correspond to the globally optimal communication beam direction.

## HOW

The paper uses reflected echoes to estimate kinematic parameters and applies EKF for angle prediction. It further introduces fine beam tracking to bridge the gap between radar-observed angles and communication-optimal beams.

## WHAT

The proposed method improves over conventional feedback-based tracking, but also shows that sensing observations are not always reliable communication alignment directions in multipath environments.

## System Assumptions

- Cellular-connected mmWave UAV.
- Ground BS aligns beams toward UAV.
- The problem is tracking after a communication target exists.

## Relevance to This Project

This is a critical cautionary reference. Our protocol cannot treat ISAC sensing as oracle truth. Beam-cell occupancy must include confidence, false alarm, miss detection, and angle error models.

## Gap

Does not solve pre-alignment neighbor discovery, distributed U2U operation, random Tx/Rx rendezvous, or topology-aware finite-time discovery.
