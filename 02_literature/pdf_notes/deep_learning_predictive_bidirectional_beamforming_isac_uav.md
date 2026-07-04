# Deep Learning-Based Predictive Bidirectional Beamforming in ISAC-Enabled UAV Networks

## Metadata

- Authors: Jinghan Xu, Xiaotian Zhou, Haixia Zhang, Yueheng Li
- Venue: IEEE Transactions on Wireless Communications, 2026
- DOI: https://doi.org/10.1109/TWC.2026.3664980
- Local PDF: user-provided, not committed to repository

## WHY

UAV mobility, position fluctuation, and attitude variation make accurate predictive beamforming difficult in ISAC-enabled UAV networks.

## HOW

The ground BS uses historical ISAC echoes from communication signals to track the UAV and predict transmit/receive beamforming matrices. The proposed HECTA-Net combines CNN, TCN, and attention to extract spatial-temporal features from historical echoes.

## WHAT

The method outperforms baselines and approaches a theoretical upper bound under randomized UAV motion patterns.

## System Assumptions

- Ground BS serves UAV.
- Communication object is known.
- ISAC echoes are available at BS.
- Main problem is predictive bidirectional beamforming, not discovery.

## Relevance to This Project

Useful for justifying that ISAC echoes can carry useful beam-selection information. Not directly usable as a neighbor discovery protocol because it is centralized, BS-led, and assumes an existing service relationship.

## Gap

Does not address fully distributed UAV-UAV pre-alignment neighbor discovery, unknown neighbor identity, random Tx/Rx scheduling, or topology-aware discovery.
