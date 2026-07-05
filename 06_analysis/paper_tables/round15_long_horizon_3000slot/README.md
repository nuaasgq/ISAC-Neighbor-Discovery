# Round15 Long-Horizon 3000-Slot Evidence

- Created: 2026-07-05T12:58:58
- Source: `05_simulation\results_raw\round15_long_horizon_3000slot_n100_b10_b15`
- Scope: N=100, B=10/15 deg, 3000 slots, 5 ms slot, ten paired seeds, single-hop range.

This run checks whether the earlier 600-slot, 3-second horizon is too short.
At 5 ms per slot, 3000 slots correspond to 15 seconds.

Endpoint summary:
- B=10, collision_aware_isac: DR=0.7718, CP-DR=0.3919, lambda2=48.2810
- B=10, improved_rl_isac: DR=0.7735, CP-DR=0.3324, lambda2=48.7594
- B=10, improved_rl_no_isac: DR=0.0034, CP-DR=0.0034, lambda2=-0.0000
- B=10, skyorbs_like_skip_scan: DR=0.0036, CP-DR=0.0036, lambda2=-0.0000
- B=10, uniform_random: DR=0.0025, CP-DR=0.0025, lambda2=-0.0000
- B=15, collision_aware_isac: DR=0.8829, CP-DR=0.1093, lambda2=67.7875
- B=15, improved_rl_isac: DR=0.8808, CP-DR=0.0818, lambda2=67.8013
- B=15, improved_rl_no_isac: DR=0.0168, CP-DR=0.0168, lambda2=-0.0000
- B=15, skyorbs_like_skip_scan: DR=0.0179, CP-DR=0.0179, lambda2=-0.0000
- B=15, uniform_random: DR=0.0125, CP-DR=0.0125, lambda2=-0.0000

Interpretation rule: use this block to discuss finite-horizon sensitivity.
It should not replace the 600-slot stress result; it shows whether the method ordering persists when the access window is extended.
The companion horizon-comparison table merges the round13 600-slot ten-seed block with this 3000-slot block for common protocols.

Paired discovery-rate deltas for proposed:
- B=10, vs improved_rl_no_isac: delta=0.7700 (10/10 positive pairs)
- B=10, vs skyorbs_like_skip_scan: delta=0.7699 (10/10 positive pairs)
- B=10, vs uniform_random: delta=0.7710 (10/10 positive pairs)
- B=15, vs improved_rl_no_isac: delta=0.8640 (10/10 positive pairs)
- B=15, vs skyorbs_like_skip_scan: delta=0.8629 (10/10 positive pairs)
- B=15, vs uniform_random: delta=0.8683 (10/10 positive pairs)
