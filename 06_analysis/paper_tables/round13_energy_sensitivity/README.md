# Round13 Radio-State Power Sensitivity

- Created: 2026-07-05T10:39:48
- Source: `05_simulation\results_raw\round13_collision_energy_10seed\per_episode_summary.csv`

This is a post-hoc reweighting of the round13 per-episode radio-state action counts.
It does not rerun the neighbor-discovery simulator and does not change discovery or collision outcomes.

## Main Result

Versus the one-slot-delay control, `collision_aware_isac` keeps positive discoveries-per-joule deltas in at least 10/10 paired seeds for every tested power profile and beamwidth.
Versus the proposed low-latency policy, the mean discoveries-per-joule delta is positive in 11/12 profile/beamwidth combinations; the weakest sign count is 4/10 paired seeds.
Boundary case versus the proposed policy: `rx_x2` at B=10 deg (mean delta -0.0836, 4/10 positive).
Thus the round13 energy result is useful diagnostic robustness evidence for TX-, sensing-, and idle-power variation, while RX-heavy platforms remain a boundary and no platform-calibrated energy optimality is claimed.

## Power Profiles

- `default`: tx=1.0, rx=0.6, sense=1.2, idle=0.05, piggyback=0.2. Simulator default radio-state accounting.
- `tx_x2`: tx=2.0, rx=0.6, sense=1.2, idle=0.05, piggyback=0.2. Transmit-heavy profile.
- `rx_x2`: tx=1.0, rx=1.2, sense=1.2, idle=0.05, piggyback=0.2. Receive-heavy profile.
- `sense_x2`: tx=1.0, rx=0.6, sense=2.4, idle=0.05, piggyback=0.4. Sensing and piggyback-sensing heavy profile.
- `idle_x4`: tx=1.0, rx=0.6, sense=1.2, idle=0.2, piggyback=0.2. Higher idle/listening baseline profile.
- `sense_half`: tx=1.0, rx=0.6, sense=0.6, idle=0.05, piggyback=0.1. Lower sensing-overhead profile.

## Outputs

- `energy_sensitivity_per_episode.csv`
- `energy_sensitivity_summary.csv`
- `energy_sensitivity_paired_deltas.csv`
- `manifest.json`
