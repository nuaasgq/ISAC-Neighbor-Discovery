# MARL Learning Curves

- Created: 2026-07-07T03:30:28
- Runs loaded: 1
- Step rows: 30000
- Episode rows: 100
- Eval rows: 66

These plots use true environment steps or training episodes as the x-axis.
They are intended for real MARL runs, not CEM candidate-search traces.

## Final Train Rows

```csv
algorithm,approx_kl,clip_fraction,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,entropy,episode,episode_return_mean_per_agent,episode_return_sum,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,n_beams,n_nodes,p90_delay_censored,p95_delay_censored,p99_delay_censored,piggyback_sense_actions,policy_loss,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,step_reward_sum_mean,training_step,true_edges_seen,tx_actions,value_loss,run,network,reward_version,env_protocol,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,9.88036433158257e-06,0.0,51,0.4204545454545454,0.3854166666666667,1.3783783783783785,1,37,27.611940298507463,2.925479343743824,0.0276119402985074,0.8222222222222222,1148,0.8567164179104477,12.647499999999996,0.3418243243243242,5.978452920913696,99,-2.9089996814727783,-29.089994430541992,662,0.0,5.593512539699809,10,1.0,157.4,gauss_markov,2.7540514973939554,648,10,300.0,300.0,300.0,1340,-8.13827940149281e-05,isac_structured_marl,773,1340,36.21621621621622,20260841,20260841,998,300,-0.0096966652199625,-0.0969666615128517,30000,45,567,0.1883949711918831,train_n10_b10_topology_adaptive_gated_contention_actor_100ep_300slot_seed20260742,topology_adaptive_gated_contention_shared,collision_topology,isac_structured_marl,topology_adaptive_gated_contention_actor,Topology-adaptive gated ISAC-MAPPO,20260742,10,648,36,18,5.0
```

## Final Eval Rows

```csv
algorithm,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,env_protocol,episode,episode_return_mean_per_agent,episode_return_sum,eval_after_episode,eval_episode,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,p90_delay_censored,p95_delay_censored,p99_delay_censored,phase,piggyback_sense_actions,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,training_step,true_edges_seen,tx_actions,run,network,reward_version,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,51,0.3928571428571428,0.34375,1.5454545454545454,1,33,24.81203007518797,2.5899111189593267,0.0248120300751879,0.7333333333333333,1113,0.8368421052631579,12.74175,0.3861136363636364,isac_structured_marl,102,-3.576499462127685,-35.76499938964844,100,2,655,0.0,4.474176501358778,10,1.0,155.06666666666666,gauss_markov,2.546535382573338,300.0,300.0,300.0,eval_stochastic,1330,isac_structured_marl,746,1330,40.30303030303031,22270744,22270744,1015,300,-0.0119216665625572,30000,45,584,train_n10_b10_topology_adaptive_gated_contention_actor_100ep_300slot_seed20260742,topology_adaptive_gated_contention_shared,collision_topology,topology_adaptive_gated_contention_actor,Topology-adaptive gated ISAC-MAPPO,20260742,10,648,36,18,5.0
```
