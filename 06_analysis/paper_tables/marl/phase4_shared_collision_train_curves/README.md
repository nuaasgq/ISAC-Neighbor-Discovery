# MARL Learning Curves

- Created: 2026-07-05T18:54:12
- Runs loaded: 1
- Step rows: 6000
- Episode rows: 20
- Eval rows: 30

These plots use true environment steps or training episodes as the x-axis.
They are intended for real MARL runs, not CEM candidate-search traces.

## Final Train Rows

```csv
algorithm,approx_kl,clip_fraction,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,entropy,episode,episode_return_mean_per_agent,episode_return_sum,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,n_beams,n_nodes,p90_delay_censored,p95_delay_censored,p99_delay_censored,piggyback_sense_actions,policy_loss,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,step_reward_sum_mean,training_step,true_edges_seen,tx_actions,value_loss,run,network,reward_version,env_protocol,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,-0.0001148147950492,0.0,114,0.2297297297297297,0.2138364779874214,3.3529411764705883,1,34,20.19002375296912,2.4713343388272064,0.0201900237529691,0.7555555555555555,1357,0.8058194774346793,13.75775,0.404639705882353,5.969111919403076,19,-6.131500244140625,-61.31499862670898,399,0.0,4.074389096746426,10,1.0,161.7111111111111,gauss_markov,2.730637116802922,648,10,300.0,300.0,300.0,1684,-0.0002026359582867,isac_structured_marl,974,1684,49.52941176470589,20260724,20260724,917,300,-0.0204383321106433,-0.204383298754692,6000,45,710,0.3882741332054138,train_n10_b10_isac_mappo_shared_collision_topology_300slot,shared,collision_topology,isac_structured_marl,10,648,36,18,5.0
```

## Final Eval Rows

```csv
algorithm,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,env_protocol,episode,episode_return_mean_per_agent,episode_return_sum,eval_after_episode,eval_episode,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,p90_delay_censored,p95_delay_censored,p99_delay_censored,phase,piggyback_sense_actions,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,true_edges_seen,tx_actions,run,network,reward_version,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,129,0.2085889570552147,0.1954022988505747,3.794117647058824,1,34,19.733023795705165,2.500137875250473,0.0197330237957051,0.7555555555555555,1382,0.8020893789901334,13.59925,0.3999779411764705,isac_structured_marl,22,-6.443499565124512,-64.43498992919922,20,2,437,0.0,4.578695988103158,10,1.0,167.55555555555554,gauss_markov,3.2299376966359974,300.0,300.0,300.0,eval_stochastic,1723,isac_structured_marl,944,1723,50.6764705882353,22270707,22270707,840,300,-0.0214783307164907,45,779,train_n10_b10_isac_mappo_shared_collision_topology_300slot,shared,collision_topology,10,648,36,18,5.0
```
