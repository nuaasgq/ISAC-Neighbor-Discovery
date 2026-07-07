# MARL Learning Curves

- Created: 2026-07-07T12:45:47
- Runs loaded: 2
- Step rows: 60000
- Episode rows: 200
- Eval rows: 132

These plots use true environment steps or training episodes as the x-axis.
They are intended for real MARL runs, not CEM candidate-search traces.

## Final Train Rows

```csv
algorithm,approx_kl,clip_fraction,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,entropy,episode,episode_return_mean_per_agent,episode_return_sum,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,n_beams,n_nodes,p90_delay_censored,p95_delay_censored,p99_delay_censored,piggyback_sense_actions,policy_loss,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,step_reward_sum_mean,training_step,true_edges_seen,tx_actions,value_loss,run,network,reward_version,env_protocol,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,-8.333200037789756e-05,0.0,40,0.4366197183098591,0.3647058823529411,1.2903225806451613,1,31,24.50592885375494,2.735797021511307,0.0245059288537549,0.6888888888888889,1111,0.8782608695652174,11.33125,0.3655241935483871,5.98006010055542,99,-3.629499912261963,-36.29499435424805,909,0.0,4.403548608600393,10,1.0,169.0888888888889,gauss_markov,2.321705073740433,648,10,300.0,300.0,300.0,1265,-0.0001020160928213,isac_structured_marl,721,1265,40.80645161290322,20260842,20260842,826,300,-0.0120983310043811,-0.1209833398461341,30000,45,544,0.2006070390343666,train_seed20260743,balanced_topology_gated_contention_shared,collision_topology,isac_structured_marl,balanced_topology_gated_contention_actor,Balanced topology gated ISAC-MAPPO,20260743,10,648,36,18,5.0
isac_mappo,-3.556466198642583e-05,0.0,21,0.58,0.4393939393939394,0.7241379310344828,1,29,25.77777777777778,2.1138952164009117,0.0257777777777777,0.6444444444444445,957,0.8506666666666667,13.718749999999998,0.4730603448275861,5.86663293838501,99,-3.102499723434448,-31.024995803833008,547,0.0,3.2263305385148646,10,1.0,168.93333333333334,gauss_markov,2.833361284957255,648,10,300.0,300.0,300.0,1125,1.3677756362628912e-05,isac_structured_marl,568,1125,38.793103448275865,20260843,20260843,1328,300,-0.0103416657075285,-0.1034166663885116,30000,45,557,0.1873952820897102,train_seed20260744,balanced_topology_gated_contention_shared,collision_topology,isac_structured_marl,balanced_topology_gated_contention_actor,Balanced topology gated ISAC-MAPPO,20260744,10,648,36,18,5.0
```

## Final Eval Rows

```csv
algorithm,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,env_protocol,episode,episode_return_mean_per_agent,episode_return_sum,eval_after_episode,eval_episode,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,p90_delay_censored,p95_delay_censored,p99_delay_censored,phase,piggyback_sense_actions,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,training_step,true_edges_seen,tx_actions,run,network,reward_version,method,method_label,training_seed,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,48,0.3924050632911392,0.3333333333333333,1.5483870967741935,1,31,22.979985174203115,2.6548483096752094,0.0229799851742031,0.6888888888888889,1141,0.8458117123795404,11.67675,0.3766693548387096,isac_structured_marl,101,-4.03749942779541,-40.37499237060547,100,1,819,0.0,3.903692154221975,10,1.0,163.44444444444446,gauss_markov,2.546535382573338,300.0,300.0,300.0,eval_stochastic,1349,isac_structured_marl,807,1349,43.516129032258064,22270744,22270744,832,300,-0.0134583311155438,30000,45,542,train_seed20260743,balanced_topology_gated_contention_shared,collision_topology,balanced_topology_gated_contention_actor,Balanced topology gated ISAC-MAPPO,20260743,10,648,36,18,5.0
isac_mappo,62,0.3608247422680412,0.3271028037383177,1.7714285714285714,1,35,33.49282296650718,2.531645569620253,0.0334928229665071,0.7777777777777778,925,0.8851674641148325,13.825,0.395,isac_structured_marl,102,-1.7879995107650757,-17.879993438720703,100,2,548,0.0,5.063934674747168,10,1.0,145.51111111111112,gauss_markov,2.6047979282487854,300.0,300.0,300.0,eval_stochastic,1045,isac_structured_marl,512,1045,29.857142857142858,22270746,22270746,1407,300,-0.0059599978849291,30000,45,533,train_seed20260744,balanced_topology_gated_contention_shared,collision_topology,balanced_topology_gated_contention_actor,Balanced topology gated ISAC-MAPPO,20260744,10,648,36,18,5.0
```
