# 伪代码 v0

## Algorithm 1: I-TAP-ND Node Procedure

```text
Input:
  Beam-cell set B_i
  Exploration floor epsilon
  Mode probabilities p_sense, p_tx, p_rx
  Target degree d_target
  ISAC error-aware update parameters

State:
  p_i[m]      occupancy prior for local beam cell b_i^m
  U_i[m]      uncertainty for local beam cell b_i^m
  Age_i[m]    observation age for local beam cell b_i^m
  Succ_i[m]   successful discovery count for b_i^m
  Fail_i[m]   failed attempt count for b_i^m
  N_i^D       discovered neighbor set

Initialize:
  p_i[m] = 1 / |B_i|
  U_i[m] = 1
  Age_i[m] = infinity
  Succ_i[m] = 0
  Fail_i[m] = 0
  N_i^D = empty

For each slot t:
  1. Age all beam-cell observations:
       Age_i[m] <- Age_i[m] + 1 for all m

  2. Sample mode m_i(t) from {SENSE, TX, RX, IDLE}

  3. If m_i(t) == SENSE:
       O_i(t) <- perform local ISAC sensing
       For each observed local beam cell b_i^m:
           update p_i[m], U_i[m], Age_i[m]
       Continue to next slot

  4. Compute priority for each local beam cell b_i^m:
       r_i,m <- occupancy_term(p_i[m], U_i[m], Age_i[m])
       l_i,m <- link_quality_proxy(Succ_i[m], Fail_i[m])
       h_i,m <- pre_discovery_topology_proxy(N_i^D, b_i^m, d_target)
       w_i,m <- r_i,m * l_i,m * h_i,m

  5. Convert priority to beam selection distribution:
       P_i^B[m] <- (1 - epsilon) * softmax(w_i,m) + epsilon / |B_i|

  6. Sample local beam cell b_i^m(t) according to P_i^B

  7. If m_i(t) == TX:
       transmit discovery beacon on b_i^m(t)

  8. If m_i(t) == RX:
       listen for discovery beacon on b_i^m(t)
       If a valid beacon is received and handshake succeeds:
           add sender to N_i^D
           Succ_i[m] <- Succ_i[m] + 1
       Else:
           Fail_i[m] <- Fail_i[m] + 1

  9. If m_i(t) == IDLE:
       no discovery action
```

## Algorithm 2: ISAC Prior Update

```text
Input:
  Previous p_i[m], U_i[m]
  ISAC observation z_i,m(t)
  Observation confidence c_i,m(t)

For each local beam cell b_i^m:
  If b_i^m is reported occupied:
      p_i[m] <- (1 - rho) * p_i[m] + rho * c_i,m(t)
      U_i[m] <- max(U_min, U_i[m] * k_confirm)
      Age_i[m] <- 0
  Else:
      p_i[m] <- (1 - rho) * p_i[m]
      U_i[m] <- min(U_max, U_i[m] * k_uncertain)
```

Notes:

- This update is a protocol-level abstraction, not a physical-layer estimator.
- `Q_i[b]` is not ground truth; it is a noisy local belief.
- False alarms and miss detections are handled by exploration and handshake feedback.

## Algorithm 3: Topology Proxy

```text
Input:
  Discovered neighbor set N_i^D
  Local beam cell b_i^m
  Target degree d_target
  Directional discovery histogram H_i[b]

degree_deficit <- max(0, d_target - |N_i^D|)
direction_scarcity <- 1 / (1 + H_i[m])
unexplored_bonus <- 1 if Age_i[m] > age_threshold else 0

h_i,b <- 1
       + gamma_1 * degree_deficit
       + gamma_2 * direction_scarcity
       + gamma_3 * unexplored_bonus
```

This proxy intentionally avoids using global topology or identities of undiscovered neighbors.
