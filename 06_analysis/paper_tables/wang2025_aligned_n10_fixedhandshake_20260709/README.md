# Wang-Aligned Discovery-First MARL Matrix

This matrix is the strict environment-alignment entry point for MARL-vs-Wang experiments.

- Base config: `05_simulation/configs/wang2025_reproduction_smoke.yaml`
- Training and evaluation horizon: Wang-style 200 slots by default.
- Beam grid: 15 azimuth x 7 elevation, approximately 25 degrees.
- RF chains: 1.
- Main comparison environment: fixed `wang2025_isac_tables` for Wang/rule action policies and MARL.
- Only the executed TX/RX/IDLE-and-beam action policy changes across main rows.
- MARL network in this matrix: non-gated `contention_shared`; no environment-side access-gate rewriting in the main comparison.
- Standalone SENSE: disabled for MARL.
- Uniform random baseline: TX/RX/IDLE only, with no standalone SENSE.
- ISAC feedback in the common environment: TX-only piggyback sensing.
- Table exchange in the common environment: Wang neighbor and sensing tables after successful first interaction.

Top-line aggregate rows:

## N=10
- Budgeted ISAC rule: discovery=0.0622, delay=194.1, p95=200.0, lambda2=-0.000
- MARL + Wang ISAC tables, discovery-first: discovery=0.4711, delay=153.8, p95=200.0, lambda2=1.707
- Uniform TX/RX/IDLE random: discovery=0.0000, delay=200.0, p95=200.0, lambda2=0.000
- Wang sensing-table action policy: discovery=0.0978, delay=185.5, p95=200.0, lambda2=0.000
