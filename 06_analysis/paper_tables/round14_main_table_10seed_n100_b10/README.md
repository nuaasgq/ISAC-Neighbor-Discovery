# Round14 Main-Table Ten-Seed Stability Check

- Source: `05_simulation\results_raw\round14_main_table_10seed_n100_b10`
- Figures: `06_analysis\paper_figures\round14_main_table_10seed_n100_b10`
- Setting: N=100, B=10 deg, Gauss-Markov mobility, 600 slots, density scaling, single-hop range.
- Seeds: 20290704, 20291713, 20292722, 20293731, 20294740, 20295749, 20296758, 20297767, 20298776, 20299785.
- Protocols: uniform random, SkyOrbs-like skip-scan, learned no-ISAC, enhanced no-ISAC, enhanced ISAC.

## Key Result

Enhanced ISAC discovery is 0.3652 versus random 0.0005, SkyOrbs-like 0.0007, learned no-ISAC 0.0007, and enhanced no-ISAC 0.0006.
Enhanced ISAC lambda2 is 13.2595; all communication-only baselines have mean lambda2 0.0000.

Discovery-rate paired deltas against all four communication-only controls are positive in uniform_random: 10/10, skyorbs_like_skip_scan: 10/10, rl_no_isac: 10/10, improved_rl_no_isac: 10/10 paired seeds.

Use this campaign as a stability check for the main N=100/B=10 baseline table. It does not replace the round13 collision-aware MAC refinement probe.

## Reproduction

Regenerate the raw data with `05_simulation/run_transfer_sweep.py` using the raw manifest in the source directory. Then run:

```powershell
python 06_analysis\scripts\analyze_round14_main_table.py --source 05_simulation\results_raw\round14_main_table_10seed_n100_b10 --output 06_analysis\paper_tables\round14_main_table_10seed_n100_b10 --figures 06_analysis\paper_figures\round14_main_table_10seed_n100_b10
```
