# MARL Learning Curves

- Created: 2026-07-06T10:40:35
- Runs loaded: 3
- Step rows: 90000
- Episode rows: 300
- Eval rows: 198

These plots use true environment steps or training episodes as the x-axis.
They are intended for real MARL runs, not CEM candidate-search traces.

## Final Train Rows

```csv
algorithm,approx_kl,clip_fraction,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,entropy,episode,episode_return_mean_per_agent,episode_return_sum,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,n_beams,n_nodes,p90_delay_censored,p95_delay_censored,p99_delay_censored,piggyback_sense_actions,policy_loss,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,step_reward_sum_mean,training_step,true_edges_seen,tx_actions,value_loss,run,network,reward_version,env_protocol,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
mappo,-6.287773482965831e-06,0.0,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1406,0.98805340829234,11.81725,11.81725,7.840667009353638,99,-4.084499835968018,-40.84499740600586,577,1.0,0.0,1,0.1,300.0,gauss_markov,2.9097086499643923,648,10,300.0,300.0,300.0,0,-3.68628516049796e-05,structured_marl_no_isac,721,1423,1423.0,20260832,20260832,1000,300,-0.0136149991303682,-0.136149987578392,30000,45,702,0.0191475981846451,train_n10_b10_mappo_no_isac_100ep_300slot_seed20260733,shared,legacy,structured_marl_no_isac,mappo_no_isac,MAPPO w/o ISAC,20260733,10,648,36,18,5.0
mappo,2.0450572386254692e-05,0.0,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1068,0.9916434540389972,10.38975,10.38975,7.821840047836304,99,-3.5914998054504395,-35.91499710083008,931,1.0,0.0,1,0.1,300.0,gauss_markov,2.6145206529075984,648,10,300.0,300.0,300.0,0,-1.6721566949762234e-05,structured_marl_no_isac,590,1077,1077.0,20260830,20260830,992,300,-0.011971665546298,-0.1197166591882705,30000,45,487,0.0117106018587946,train_n10_b10_mappo_no_isac_100ep_300slot_seed20260731,shared,legacy,structured_marl_no_isac,mappo_no_isac,MAPPO w/o ISAC,20260731,10,648,36,18,5.0
mappo,-3.657139235291318e-05,0.0,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1137,0.9878366637706344,9.247749999999998,9.247749999999998,7.822308778762817,99,-3.567499876022339,-35.67499542236328,1107,1.0,0.0,1,0.1,300.0,gauss_markov,2.377444509918951,648,10,300.0,300.0,300.0,0,-9.3527558240325e-05,structured_marl_no_isac,618,1151,1151.0,20260831,20260831,742,300,-0.0118916649371385,-0.1189166754484176,30000,45,533,0.0111222164705395,train_n10_b10_mappo_no_isac_100ep_300slot_seed20260732,shared,legacy,structured_marl_no_isac,mappo_no_isac,MAPPO w/o ISAC,20260732,10,648,36,18,5.0
```

## Final Eval Rows

```csv
algorithm,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,env_protocol,episode,episode_return_mean_per_agent,episode_return_sum,eval_after_episode,eval_episode,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,p90_delay_censored,p95_delay_censored,p99_delay_censored,phase,piggyback_sense_actions,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,true_edges_seen,tx_actions,run,network,reward_version,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms,training_step
mappo,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,0,0.0,10.778,10.778,structured_marl_no_isac,101,-2.371999740600586,-23.71999931335449,100,1,1256,1.0,0.0,1,0.1,300.0,gauss_markov,2.385437756787287,300.0,300.0,300.0,eval_deterministic,0,structured_marl_no_isac,0,0,0.0,21360732,21360732,1744,300,-0.0079066660255193,45,0,train_n10_b10_mappo_no_isac_100ep_300slot_seed20260731,shared,legacy,mappo_no_isac,MAPPO w/o ISAC,20260731,10,648,36,18,5.0,30000
mappo,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1146,0.9887834339948232,9.23375,9.23375,structured_marl_no_isac,100,-3.5855000019073486,-35.85499954223633,100,0,1099,1.0,0.0,1,0.1,300.0,gauss_markov,2.5383955375289746,300.0,300.0,300.0,eval_stochastic,0,structured_marl_no_isac,644,1159,1159.0,22270732,22270732,742,300,-0.0119516663253307,45,515,train_n10_b10_mappo_no_isac_100ep_300slot_seed20260732,shared,legacy,mappo_no_isac,MAPPO w/o ISAC,20260732,10,648,36,18,5.0,30000
mappo,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1438,0.990358126721763,11.949249999999996,11.949249999999996,structured_marl_no_isac,102,-4.151500225067139,-41.51499938964844,100,2,549,1.0,0.0,1,0.1,300.0,gauss_markov,2.893785495652961,300.0,300.0,300.0,eval_stochastic,0,structured_marl_no_isac,721,1452,1452.0,22270735,22270735,999,300,-0.0138383330777287,45,731,train_n10_b10_mappo_no_isac_100ep_300slot_seed20260733,shared,legacy,mappo_no_isac,MAPPO w/o ISAC,20260733,10,648,36,18,5.0,30000
```
