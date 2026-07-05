# Round12 Collision-Aware MAC Refinement Probe

- Created: 2026-07-05T08:49:07
- Source: `05_simulation\results_raw\round12_collision_aware_probe_v2`

This focused five-seed block reuses the round11 N=100, B=10/B=15, Gauss-Markov, 600-slot, density-scaled, single-hop setting.
It adds `collision_aware_isac`, a distributed ISAC variant that keeps the same candidate-set handshake interface but lowers TX probability under local candidate and collision pressure.
Use it as a mechanism-refinement probe: it supports the claim that the B=15 collision-penalized boundary can be mitigated by MAC-layer role control.
