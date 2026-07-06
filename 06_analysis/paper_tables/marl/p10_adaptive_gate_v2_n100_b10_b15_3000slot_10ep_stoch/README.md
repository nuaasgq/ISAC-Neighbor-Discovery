# MARL Transfer Evaluation

- Created: 2026-07-07T02:45:33
- Runs loaded: 2
- Rows loaded: 20

This table aggregates zero-shot evaluations of trained shared MARL policies under changed node counts and beam codebooks.

## Summary

```csv
method,method_label,train_algorithm,train_network,train_reward_version,env_protocol,phase,node_count,beamwidth_deg,beam_count,slots_per_episode,communication_range_m,sensing_range_m,eval_n,episodes,run_n,discovery_rate_mean,discovery_rate_std,discovery_rate_ci95,collision_penalized_discovery_rate_mean,collision_penalized_discovery_rate_std,collision_penalized_discovery_rate_ci95,mean_delay_censored_mean,mean_delay_censored_std,mean_delay_censored_ci95,p95_delay_censored_mean,p95_delay_censored_std,p95_delay_censored_ci95,empty_scan_ratio_mean,empty_scan_ratio_std,empty_scan_ratio_ci95,lambda2_mean,lambda2_std,lambda2_ci95,largest_component_size_mean,largest_component_size_std,largest_component_size_ci95,lcc_ratio_mean,lcc_ratio_std,lcc_ratio_ci95,isolated_node_ratio_mean,isolated_node_ratio_std,isolated_node_ratio_ci95,collision_count_mean,collision_count_std,collision_count_ci95,collisions_per_discovery_censored_mean,collisions_per_discovery_censored_std,collisions_per_discovery_censored_ci95,discoveries_per_1000_scan_actions_mean,discoveries_per_1000_scan_actions_std,discoveries_per_1000_scan_actions_ci95,discoveries_per_joule_mean,discoveries_per_joule_std,discoveries_per_joule_ci95
adaptive_gated_contention_actor,adaptive_gated_contention_actor,isac_mappo,adaptive_gated_contention_shared,collision_topology,isac_structured_marl,eval_stochastic,100,10.0,648,3000,900.0,900.0,10,10,1,0.22032323232323225,0.009791069797741202,0.006068567932947081,0.20266161477726846,0.006663814083791166,0.004130274760102891,2572.8780404040403,18.88383251577217,11.704320653785565,3000.0,0.0,0.0,0.10239943642776243,0.002803187657994457,0.0017374337107945756,7.550575437691985,0.9631750308705137,0.5969820690589476,100.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,429.6,75.95642025383883,47.0782770193548,0.3920241432767926,0.05556823862477232,0.034441551125132544,18.407823940916956,0.9269659824656807,0.5745394683451021,1.7109356524583508,0.08077197673983373,0.05006298985195517
adaptive_gated_contention_actor,adaptive_gated_contention_actor,isac_mappo,adaptive_gated_contention_shared,collision_topology,isac_structured_marl,eval_stochastic,100,15.0,288,3000,900.0,900.0,10,10,1,0.2934949494949494,0.01330807388666157,0.00824842965132537,0.22650317568379172,0.008271980489168563,0.005127026624824304,2397.843818181818,22.918350833232857,14.20494101417539,3000.0,0.0,0.0,0.03805184103038728,0.0017940426880774217,0.0011119591783235489,10.992504926426129,1.98538402407149,1.2305537670505886,100.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1463.7,157.62053588708972,97.69422028622438,1.0066960436774886,0.08987379794122866,0.05570435707895072,28.075114875453817,1.5552115901849373,0.9639301302214465,2.419822569506382,0.11780491964511595,0.07301624566772952
```
