# Round13 Collision-Aware MAC Refinement Probe

- Created: 2026-07-05T09:37:08
- Source: `05_simulation\results_raw\round13_collision_energy_10seed`

This focused paired-seed block reuses the round11 N=100, B=10/B=15, Gauss-Markov, 600-slot, density-scaled, single-hop setting.
It adds `collision_aware_isac`, a distributed ISAC variant that keeps the same candidate-set handshake interface but lowers TX probability under local candidate and collision pressure.
Use it as a mechanism-refinement probe: it supports the claim that the B=15 collision-penalized boundary can be mitigated by MAC-layer role control.

## Raw Sweep Command

The raw simulation directory is intentionally ignored by Git because it contains large per-slot and per-edge CSV files.
For a fresh clone, regenerate it with:

```powershell
python 05_simulation\run_transfer_sweep.py `
  --config 05_simulation\configs\paper_transfer_train_n10_b10_singlehop.yaml `
  --trained-config 06_analysis\paper_tables\round2_transfer\training\best_config.yaml `
  --output 05_simulation\results_raw\round13_collision_energy_10seed `
  --node-counts 100 `
  --beamwidth-degs 10,15 `
  --mobilities gauss_markov `
  --seeds 20290704,20291713,20292722,20293731,20294740,20295749,20296758,20297767,20298776,20299785 `
  --episodes-per-seed 1 `
  --slots 600 `
  --slot-metric-period 1 `
  --area-scale density `
  --range-mode singlehop `
  --protocols uniform_random,improved_rl_no_isac,ablation_isac_one_slot_delay,improved_rl_isac,collision_aware_isac `
  --train-node-count 10 `
  --train-beamwidth-deg 10 `
  --name round13_collision_energy_10seed
```

The expected raw manifest has 20 cases, 100 episode rows, 60000 slot rows, 131232 edge rows, and 10 aggregate rows.

## Post-Processing Command

```powershell
python 06_analysis\scripts\plot_round12_collision_aware.py `
  --source 05_simulation\results_raw\round13_collision_energy_10seed `
  --output 06_analysis\paper_tables\round13_collision_energy_10seed `
  --figures 06_analysis\paper_figures\round13_collision_energy_10seed `
  --tag round13
```

This produces the tracked endpoint summary, paired-delta summary, cumulative-discovery curves, manifest, and eight 1920 x 1440 PNG figures.

## Energy Accounting

Round13 uses assumed radio-state accounting under these default powers:

- `tx_power_w = 1.0`
- `rx_power_w = 0.6`
- `sense_power_w = 1.2`
- `idle_power_w = 0.05`
- `piggyback_sense_power_w = 0.2`

These are diagnostic accounting assumptions, not platform-calibrated energy-optimality claims.
For external supplement packaging, archive the raw `05_simulation\results_raw\round13_collision_energy_10seed` directory separately because it is not tracked by Git.
