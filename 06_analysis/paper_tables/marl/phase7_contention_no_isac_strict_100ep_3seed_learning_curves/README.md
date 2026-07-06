# MARL Learning Curves

- Created: 2026-07-06T08:10:29
- Runs loaded: 3
- Step rows: 90000
- Episode rows: 300
- Eval rows: 198

These plots use true environment steps or training episodes as the x-axis.
They are intended for real MARL runs, not CEM candidate-search traces.

## Final Train Rows

```csv
algorithm,approx_kl,clip_fraction,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,entropy,episode,episode_return_mean_per_agent,episode_return_sum,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,n_beams,n_nodes,p90_delay_censored,p95_delay_censored,p99_delay_censored,piggyback_sense_actions,policy_loss,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,step_reward_sum_mean,training_step,true_edges_seen,tx_actions,value_loss,run,network,reward_version,env_protocol,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms

mappo,6.127617420870202e-05,0.0,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1163,0.9889455782312924,7.637,7.637,7.737774133682251,99,-4.3379998207092285,-43.37999725341797,1372,1.0,0.0,1,0.1,300.0,gauss_markov,2.9097086499643923,648,10,300.0,300.0,300.0,0,-6.835317512710049e-05,structured_marl_no_isac,649,1176,1176.0,20260832,20260832,452,300,-0.0144599992781877,-0.1446000039577484,30000,45,527,0.0250557623803615,train_n10_b10_contention_no_isac_100ep_300slot_seed20260733,contention_shared,collision_topology,structured_marl_no_isac,contention_no_isac,Contention-aware MAPPO w/o ISAC,20260733,10,648,36,18,5.0

mappo,-5.7982026521741226e-05,0.0,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1577,0.98378041172801,9.8445,9.8445,7.814770698547363,99,-5.616000175476074,-56.15999984741211,766,1.0,0.0,1,0.1,300.0,gauss_markov,2.6145206529075984,648,10,300.0,300.0,300.0,0,-0.0004000767088339,structured_marl_no_isac,1074,1603,1603.0,20260830,20260830,631,300,-0.0187199991196393,-0.187199980020523,30000,45,529,0.0406062267720699,train_n10_b10_contention_no_isac_100ep_300slot_seed20260731,contention_shared,collision_topology,structured_marl_no_isac,contention_no_isac,Contention-aware MAPPO w/o ISAC,20260731,10,648,36,18,5.0

mappo,3.881313302554757e-05,0.0,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,990,0.9860557768924304,10.730250000000002,10.730250000000002,7.795147180557251,99,-4.41949987411499,-44.19499969482422,881,1.0,0.0,1,0.1,300.0,gauss_markov,2.377444509918951,648,10,300.0,300.0,300.0,0,-3.388222104216965e-05,structured_marl_no_isac,600,1004,1004.0,20260831,20260831,1115,300,-0.0147316670045256,-0.1473166495561599,30000,45,404,0.0164241660386323,train_n10_b10_contention_no_isac_100ep_300slot_seed20260732,contention_shared,collision_topology,structured_marl_no_isac,contention_no_isac,Contention-aware MAPPO w/o ISAC,20260732,10,648,36,18,5.0
```

## Final Eval Rows

```csv
algorithm,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,env_protocol,episode,episode_return_mean_per_agent,episode_return_sum,eval_after_episode,eval_episode,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,p90_delay_censored,p95_delay_censored,p99_delay_censored,phase,piggyback_sense_actions,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,true_edges_seen,tx_actions,run,network,reward_version,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms,training_step

mappo,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,2700,0.9,9.0,9.0,structured_marl_no_isac,101,-11.991999626159668,-119.91999816894533,100,1,0,1.0,0.0,1,0.1,300.0,gauss_markov,2.385437756787287,300.0,300.0,300.0,eval_deterministic,0,structured_marl_no_isac,3000,3000,3000.0,21360732,21360732,0,300,-0.0399733334779739,45,0,train_n10_b10_contention_no_isac_100ep_300slot_seed20260731,contention_shared,collision_topology,contention_no_isac,Contention-aware MAPPO w/o ISAC,20260731,10,648,36,18,5.0,30000

mappo,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1010,0.9824902723735408,10.749,10.749,structured_marl_no_isac,100,-4.51800012588501,-45.18000030517578,100,0,876,1.0,0.0,1,0.1,300.0,gauss_markov,2.5383955375289746,300.0,300.0,300.0,eval_stochastic,0,structured_marl_no_isac,593,1028,1028.0,22270732,22270732,1096,300,-0.0150600001215934,45,435,train_n10_b10_contention_no_isac_100ep_300slot_seed20260732,contention_shared,collision_topology,contention_no_isac,Contention-aware MAPPO w/o ISAC,20260732,10,648,36,18,5.0,30000

mappo,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1109,0.9892952720785012,7.44575,7.44575,structured_marl_no_isac,102,-4.333500385284424,-43.334999084472656,100,2,1419,1.0,0.0,1,0.1,300.0,gauss_markov,2.893785495652961,300.0,300.0,300.0,eval_stochastic,0,structured_marl_no_isac,637,1121,1121.0,22270735,22270735,460,300,-0.0144449993968009,45,484,train_n10_b10_contention_no_isac_100ep_300slot_seed20260733,contention_shared,collision_topology,contention_no_isac,Contention-aware MAPPO w/o ISAC,20260733,10,648,36,18,5.0,30000
```
