# MARL Transfer Evaluation

- Created: 2026-07-05T16:46:52
- Runs loaded: 4
- Rows loaded: 4

This table aggregates zero-shot evaluations of trained shared MARL policies under changed node counts and beam codebooks.

## Summary

```csv
train_algorithm,train_network,train_reward_version,env_protocol,phase,node_count,beamwidth_deg,beam_count,slots_per_episode,communication_range_m,sensing_range_m,eval_n,episodes,run_n,discovery_rate_mean,discovery_rate_std,discovery_rate_ci95,collision_penalized_discovery_rate_mean,collision_penalized_discovery_rate_std,collision_penalized_discovery_rate_ci95,mean_delay_censored_mean,mean_delay_censored_std,mean_delay_censored_ci95,p95_delay_censored_mean,p95_delay_censored_std,p95_delay_censored_ci95,empty_scan_ratio_mean,empty_scan_ratio_std,empty_scan_ratio_ci95,lambda2_mean,lambda2_std,lambda2_ci95,largest_component_size_mean,largest_component_size_std,largest_component_size_ci95,lcc_ratio_mean,lcc_ratio_std,lcc_ratio_ci95,isolated_node_ratio_mean,isolated_node_ratio_std,isolated_node_ratio_ci95,collision_count_mean,collision_count_std,collision_count_ci95,collisions_per_discovery_censored_mean,collisions_per_discovery_censored_std,collisions_per_discovery_censored_ci95,discoveries_per_1000_scan_actions_mean,discoveries_per_1000_scan_actions_std,discoveries_per_1000_scan_actions_ci95,discoveries_per_joule_mean,discoveries_per_joule_std,discoveries_per_joule_ci95
isac_mappo,scalegraph_beam,collision_topology,isac_structured_marl,eval_stochastic,20,15.0,288,3000,900.0,900.0,1,1,1,0.868421052631579,0.0,0.0,0.0833333333333333,0.0,0.0,855.9894736842106,0.0,0.0,3000.0,0.0,0.0,0.1298073079075717,0.0,0.0,13.878885313608029,0.0,0.0,20.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1790.0,0.0,0.0,10.848484848484848,0.0,0.0,6.596306068601583,0.0,0.0,0.7118404959586874,0.0,0.0
isac_mappo,scalegraph_beam,legacy,isac_structured_marl,eval_stochastic,20,15.0,288,3000,900.0,900.0,1,1,1,0.8631578947368421,0.0,0.0,0.056551724137931,0.0,0.0,812.4157894736842,0.0,0.0,3000.0,0.0,0.0,0.1140579710144927,0.0,0.0,13.462516843136978,0.0,0.0,20.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,2710.0,0.0,0.0,16.524390243902438,0.0,0.0,5.942028985507246,0.0,0.0,0.6297864006459151,0.0,0.0
isac_mappo,scalegraph_beam,collision_topology,isac_structured_marl,eval_stochastic,50,15.0,288,3000,900.0,900.0,1,1,1,0.5346938775510204,0.0,0.0,0.0678264471367919,0.0,0.0,1782.151836734694,0.0,0.0,3000.0,0.0,0.0,0.0239378254071859,0.0,0.0,16.86353890708644,0.0,0.0,50.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,8432.0,0.0,0.0,12.873282442748092,0.0,0.0,9.94247028643422,0.0,0.0,1.118273658637149,0.0,0.0
isac_mappo,scalegraph_beam,legacy,isac_structured_marl,eval_stochastic,50,15.0,288,3000,900.0,900.0,1,1,1,0.5355102040816326,0.0,0.0,0.0492640432562331,0.0,0.0,1777.2579591836734,0.0,0.0,3000.0,0.0,0.0,0.020018161365084,0.0,0.0,16.95671099295796,0.0,0.0,50.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,12091.0,0.0,0.0,18.43140243902439,0.0,0.0,8.890936936692734,0.0,0.0,0.9966068272125318,0.0,0.0
```
