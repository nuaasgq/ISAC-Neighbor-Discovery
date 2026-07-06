# MARL Learning Curves

- Created: 2026-07-07T01:58:29
- Runs loaded: 1
- Step rows: 30000
- Episode rows: 100
- Eval rows: 66

These plots use true environment steps or training episodes as the x-axis.
They are intended for real MARL runs, not CEM candidate-search traces.

## Final Train Rows

```csv
algorithm,approx_kl,clip_fraction,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,entropy,episode,episode_return_mean_per_agent,episode_return_sum,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,n_beams,n_nodes,p90_delay_censored,p95_delay_censored,p99_delay_censored,piggyback_sense_actions,policy_loss,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,step_reward_sum_mean,training_step,true_edges_seen,tx_actions,value_loss,run,network,reward_version,env_protocol,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,-0.0001685211799728,0.0,10,0.7142857142857143,0.4545454545454545,0.4,1,25,27.17391304347826,3.146633102580239,0.0271739130434782,0.5555555555555556,834,0.9065217391304348,7.945,0.3178,5.957029581069946,99,-2.2239997386932373,-22.23999786376953,1548,0.0,3.017257753168048,10,1.0,195.7555555555556,gauss_markov,2.2826807332580707,648,10,300.0,300.0,300.0,920,-4.208022220220098e-05,isac_structured_marl,577,920,36.8,20260840,20260840,532,300,-0.0074133328162133,-0.0741333290934562,30000,45,343,0.1120906695723533,train_n10_b10_adaptive_gated_contention_actor_100ep_300slot_seed20260741,adaptive_gated_contention_shared,collision_topology,isac_structured_marl,adaptive_gated_contention_actor,Adaptive gated contention ISAC-MAPPO,20260741,10,648,36,18,5.0
```

## Final Eval Rows

```csv
algorithm,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,env_protocol,episode,episode_return_mean_per_agent,episode_return_sum,eval_after_episode,eval_episode,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,p90_delay_censored,p95_delay_censored,p99_delay_censored,phase,piggyback_sense_actions,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,training_step,true_edges_seen,tx_actions,run,network,reward_version,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,16,0.5897435897435898,0.3770491803278688,0.6956521739130435,1,23,23.069207622868607,2.8269419862340213,0.0230692076228686,0.5111111111111111,901,0.9037111334002006,8.136000000000001,0.3537391304347826,isac_structured_marl,102,-3.1599996089935303,-31.599998474121094,100,2,1504,0.0,2.040260066352949,10,1.0,196.55555555555557,gauss_markov,2.425995672891366,300.0,300.0,300.0,eval_stochastic,997,isac_structured_marl,608,997,43.34782608695652,22270743,22270743,499,300,-0.010533332824707,30000,45,389,train_n10_b10_adaptive_gated_contention_actor_100ep_300slot_seed20260741,adaptive_gated_contention_shared,collision_topology,adaptive_gated_contention_actor,Adaptive gated contention ISAC-MAPPO,20260741,10,648,36,18,5.0
```
