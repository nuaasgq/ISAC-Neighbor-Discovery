# MARL Learning Curves

- Created: 2026-07-05T19:47:11
- Runs loaded: 1
- Step rows: 6000
- Episode rows: 20
- Eval rows: 30

These plots use true environment steps or training episodes as the x-axis.
They are intended for real MARL runs, not CEM candidate-search traces.

## Final Train Rows

```csv
algorithm,approx_kl,clip_fraction,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,entropy,episode,episode_return_mean_per_agent,episode_return_sum,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,n_beams,n_nodes,p90_delay_censored,p95_delay_censored,p99_delay_censored,piggyback_sense_actions,policy_loss,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,step_reward_sum_mean,training_step,true_edges_seen,tx_actions,value_loss,run,network,reward_version,env_protocol,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,-3.447604349149369e-05,0.0,92,0.2333333333333333,0.2043795620437956,3.2857142857142856,1,28,18.99592944369064,1.8697829716193657,0.0189959294436906,0.6222222222222222,1255,0.8514246947082768,14.975,0.5348214285714286,5.853566408157349,19,-5.990999698638916,-59.90999603271485,260,0.0,3.4563225499577417,10,1.0,171.77777777777777,gauss_markov,2.730637116802922,648,10,300.0,300.0,300.0,1474,-0.000137420141022,isac_structured_marl,765,1474,52.642857142857146,20260724,20260724,1266,300,-0.0199699979275465,-0.1997000128030777,6000,45,709,0.3126182556152344,train_n10_b10_isac_mappo_contention_shared_collision_topology_300slot,contention_shared,collision_topology,isac_structured_marl,10,648,36,18,5.0
```

## Final Eval Rows

```csv
algorithm,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,env_protocol,episode,episode_return_mean_per_agent,episode_return_sum,eval_after_episode,eval_episode,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,p90_delay_censored,p95_delay_censored,p99_delay_censored,phase,piggyback_sense_actions,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,true_edges_seen,tx_actions,run,network,reward_version,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,87,0.2983870967741935,0.2803030303030303,2.3513513513513518,1,37,24.025974025974023,2.497468781640229,0.024025974025974,0.8222222222222222,1292,0.8389610389610389,14.815,0.4004054054054053,isac_structured_marl,22,-4.3979997634887695,-43.97999572753906,20,2,276,0.0,5.766790456942358,10,1.0,150.57777777777778,gauss_markov,3.2299376966359974,300.0,300.0,300.0,eval_stochastic,1540,isac_structured_marl,799,1540,41.62162162162162,22270707,22270707,1184,300,-0.0146599989384412,45,741,train_n10_b10_isac_mappo_contention_shared_collision_topology_300slot,contention_shared,collision_topology,10,648,36,18,5.0
```
