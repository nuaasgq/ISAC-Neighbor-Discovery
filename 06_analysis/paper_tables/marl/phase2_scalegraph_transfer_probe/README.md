# MARL Transfer Evaluation

- Created: 2026-07-05T15:38:20
- Runs loaded: 4
- Rows loaded: 4

This table aggregates zero-shot evaluations of trained shared MARL policies under changed node counts and beam codebooks.

## Summary

```csv
train_algorithm,train_network,train_reward_version,env_protocol,phase,node_count,beamwidth_deg,beam_count,slots_per_episode,communication_range_m,sensing_range_m,eval_n,run_n,discovery_rate_mean,discovery_rate_std,discovery_rate_ci95,mean_delay_censored_mean,mean_delay_censored_std,mean_delay_censored_ci95,p95_delay_censored_mean,p95_delay_censored_std,p95_delay_censored_ci95,empty_scan_ratio_mean,empty_scan_ratio_std,empty_scan_ratio_ci95,lambda2_mean,lambda2_std,lambda2_ci95,collision_count_mean,collision_count_std,collision_count_ci95
isac_mappo,scalegraph_beam,collision_topology,isac_structured_marl,eval_stochastic,20,15.0,288,3000,900.0,900.0,1,1,0.868421052631579,0.0,0.0,855.9894736842106,0.0,0.0,3000.0,0.0,0.0,0.1298073079075717,0.0,0.0,13.878885313608029,0.0,0.0,1790.0,0.0,0.0
isac_mappo,scalegraph_beam,legacy,isac_structured_marl,eval_stochastic,20,15.0,288,3000,900.0,900.0,1,1,0.8631578947368421,0.0,0.0,812.4157894736842,0.0,0.0,3000.0,0.0,0.0,0.1140579710144927,0.0,0.0,13.462516843136978,0.0,0.0,2710.0,0.0,0.0
isac_mappo,scalegraph_beam,collision_topology,isac_structured_marl,eval_stochastic,50,15.0,288,3000,900.0,900.0,1,1,0.5346938775510204,0.0,0.0,1782.151836734694,0.0,0.0,3000.0,0.0,0.0,0.0239378254071859,0.0,0.0,16.86353890708644,0.0,0.0,8432.0,0.0,0.0
isac_mappo,scalegraph_beam,legacy,isac_structured_marl,eval_stochastic,50,15.0,288,3000,900.0,900.0,1,1,0.5355102040816326,0.0,0.0,1777.2579591836734,0.0,0.0,3000.0,0.0,0.0,0.020018161365084,0.0,0.0,16.95671099295796,0.0,0.0,12091.0,0.0,0.0
```
