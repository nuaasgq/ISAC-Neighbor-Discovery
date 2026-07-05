# MARL Learning Curves

- Created: 2026-07-05T16:23:06
- Runs loaded: 2
- Step rows: 2400
- Episode rows: 40
- Eval rows: 60

These plots use true environment steps or training episodes as the x-axis.
They are intended for real MARL runs, not CEM candidate-search traces.

## Final Train Rows

```csv
algorithm,approx_kl,clip_fraction,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,entropy,episode,episode_return_mean_per_agent,episode_return_sum,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,n_beams,n_nodes,p90_delay_censored,p95_delay_censored,p99_delay_censored,piggyback_sense_actions,policy_loss,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,step_reward_sum_mean,training_step,true_edges_seen,tx_actions,value_loss,run,network,reward_version,env_protocol,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,-8.184771104424726e-06,0.0,172,0.173076923076923,0.1658986175115207,4.777777777777778,1,36,20.942408376963357,2.5213173883354054,0.0209424083769633,0.8,1409,0.8196625945317044,14.27825,0.3966180555555555,5.968673229217529,19,-4.382499694824219,-43.82499694824219,377,0.0,5.483420182548479,10,1.0,157.4,gauss_markov,2.730637116802922,648,10,300.0,300.0,300.0,1719,-6.050801293655539e-05,isac_structured_marl,777,1719,47.75,20260724,20260724,904,300,-0.0146083319559693,-0.1460833251476287,6000,45,942,0.1275155767798423,train_n10_b10_isac_mappo_300slot,shared,legacy,isac_structured_marl,10,648,36,18,5.0
mappo,5.111002974445e-05,0.0,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1525,0.9857789269553976,9.809,9.809,7.846033334732056,19,-4.068999767303467,-40.68999862670898,888,1.0,0.0,1,0.1,300.0,gauss_markov,2.730637116802922,648,10,300.0,300.0,300.0,0,6.465832464108701e-06,structured_marl_no_isac,769,1547,1547.0,20260724,20260724,565,300,-0.0135633330792188,-0.1356333345174789,6000,45,778,0.0485958717763423,train_n10_b10_mappo_300slot,shared,legacy,structured_marl_no_isac,10,648,36,18,5.0
```

## Final Eval Rows

```csv
algorithm,collision_count,collision_normalized_efficiency,collision_penalized_discovery_rate,collisions_per_discovery_censored,connected_components,discovered_edges,discoveries_per_1000_scan_actions,discoveries_per_joule,discovery_per_scan_action,discovery_rate,empty_scan_count,empty_scan_ratio,energy_j,energy_per_discovery_censored_j,env_protocol,episode,episode_return_mean_per_agent,episode_return_sum,eval_after_episode,eval_episode,idle_actions,isolated_node_ratio,lambda2,largest_component_size,lcc_ratio,mean_delay_censored,mobility_model,moved_distance_mean_m,p90_delay_censored,p95_delay_censored,p99_delay_censored,phase,piggyback_sense_actions,protocol,rx_actions,scan_actions,scan_actions_per_discovery_censored,scenario_seed,seed,sense_actions,slots,step_reward_mean,true_edges_seen,tx_actions,run,network,reward_version,node_count,beam_count,azimuth_cells,elevation_cells,slot_duration_ms
isac_mappo,203,0.1361702127659574,0.1290322580645161,6.34375,1,32,18.669778296382727,2.276891331803547,0.0186697782963827,0.7111111111111111,1384,0.8074679113185531,14.05425,0.4391953125,isac_structured_marl,22,-4.952499866485596,-49.52499771118164,20,2,401,0.0,4.234363544894492,10,1.0,148.42222222222222,gauss_markov,2.742437369160833,300.0,300.0,300.0,eval_stochastic,1714,isac_structured_marl,820,1714,53.5625,21290707,21290707,885,300,-0.0165083333849906,45,894,train_n10_b10_isac_mappo_300slot,shared,legacy,10,648,36,18,5.0
mappo,0,0.0,0.0,0.0,10,0,0.0,0.0,0.0,0.0,1498,0.9887788778877888,9.83025,9.83025,structured_marl_no_isac,22,-4.030499458312988,-40.30499267578125,20,2,885,1.0,0.0,1,0.1,300.0,gauss_markov,3.2299376966359974,300.0,300.0,300.0,eval_stochastic,0,structured_marl_no_isac,783,1515,1515.0,22270707,22270707,600,300,-0.0134349977597594,45,732,train_n10_b10_mappo_300slot,shared,legacy,10,648,36,18,5.0
```
