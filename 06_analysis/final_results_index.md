# 最终结果索引 v2

日期：2026-07-04

本轮结果面向“ISAC 辅助窄波束无人机邻居发现”的论文初稿数据闭环，覆盖训练、四类动态运动模型测试、五类协议对比、论文图和可跟踪结果表。

## 产物位置

- 训练输出精简表：`06_analysis/paper_tables/final_round1/training/`
- 测试汇总表：`06_analysis/paper_tables/final_round1/{gauss_markov_20ep,random_walk_20ep,random_direction_10ep,random_waypoint_10ep}/`
- 论文图：`06_analysis/paper_figures/final_round1/`
- 绘图清单：`06_analysis/paper_figures/final_round1/paper_figure_manifest.json`
- 复现实验设计：`06_analysis/final_experiment_plan.md`

## 训练设置

训练对象是 `improved_rl_isac` 的共享策略参数，采用 CEM/black-box shared-policy search。训练命令使用 `paper_core_d1.yaml`，10 generations、population 10、每候选 2 episodes、300 slots，训练 seed 为 `20260704,20261713,20262722`，测试 seed 为 `20270704,20271713,20272722`。

最优参数来自 generation 8：

| 参数 | 数值 |
|---|---:|
| `alpha_occupancy` | 1.9833 |
| `softmax_beta` | 3.2337 |
| `exploration_floor` | 0.0567 |
| `confidence_decay` | 0.9937 |
| `piggyback_sensing_period_multiplier` | 0.5000 |

最优训练指标：score 123.902，发现率 0.9612，平均发现时延 49.90 slots，P95 时延 177.48 slots，空扫率 0.1857，lambda2 4.5933。独立测试汇总中，`improved_rl_isac` 发现率为 0.9621，平均时延为 48.95 slots，P95 为 192.08 slots。

## 五类对比协议

| 类别 | 协议名 | 作用 |
|---|---|---|
| 参考文献近似基线 | `skyorbs_like_skip_scan` | 类 SkyOrbs 三维 skip-scan 序列 |
| 完全随机 | `uniform_random` | 随机 mode/beam 邻居发现 |
| 强化学习但无 ISAC | `rl_no_isac` | 在线学习代理，无 sensing belief |
| 改进强化学习但无 ISAC | `improved_rl_no_isac` | memory/topology-aware 代理，无 sensing belief |
| 改进强化学习 + ISAC | `improved_rl_isac` | ISAC beam belief + memory/topology-aware 代理 |

## 测试矩阵

| 场景 | mobility | episodes | slots/episode | episode rows | slot rows |
|---|---|---:|---:|---:|---:|
| D1-GM | `gauss_markov` | 20 | 300 | 100 | 30000 |
| D1-RW | `random_walk` | 20 | 300 | 100 | 30000 |
| D1-RD | `random_direction` | 10 | 300 | 50 | 15000 |
| D1-RWP | `random_waypoint` | 10 | 300 | 50 | 15000 |

## 关键结果

### Gauss-Markov 20ep

| 协议 | 发现率 | 平均时延 | P95 | P99 | 空扫率 | lambda2 |
|---|---:|---:|---:|---:|---:|---:|
| `improved_rl_isac` | 0.9558 | 49.00 | 184.46 | 276.10 | 0.1698 | 4.2400 |
| `improved_rl_no_isac` | 0.8608 | 104.18 | 294.72 | 300.00 | 0.4478 | 3.5691 |
| `rl_no_isac` | 0.8463 | 110.66 | 298.92 | 300.00 | 0.4378 | 3.2260 |
| `uniform_random` | 0.8351 | 118.11 | 300.00 | 300.00 | 0.4346 | 3.3603 |
| `skyorbs_like_skip_scan` | 0.4882 | 179.62 | 300.00 | 300.00 | 0.4423 | 1.5919 |

相对 `uniform_random`：发现率 +12.07 个百分点，平均时延降低 58.51%，空扫率降低 60.94%。

### Random Walk 20ep

| 协议 | 发现率 | 平均时延 | P95 | P99 | 空扫率 | lambda2 |
|---|---:|---:|---:|---:|---:|---:|
| `improved_rl_isac` | 0.9569 | 77.43 | 243.59 | 282.07 | 0.3914 | 2.6037 |
| `improved_rl_no_isac` | 0.9171 | 93.78 | 278.65 | 295.93 | 0.4480 | 2.5076 |
| `rl_no_isac` | 0.9187 | 99.20 | 292.25 | 299.39 | 0.4393 | 2.3802 |
| `uniform_random` | 0.9077 | 106.11 | 293.77 | 300.00 | 0.4357 | 2.3526 |
| `skyorbs_like_skip_scan` | 0.5206 | 165.11 | 300.00 | 300.00 | 0.4376 | 1.1395 |

相对 `uniform_random`：发现率 +4.92 个百分点，平均时延降低 27.03%，空扫率降低 10.17%。

### Random Direction 10ep

| 协议 | 发现率 | 平均时延 | P95 | P99 | 空扫率 | lambda2 |
|---|---:|---:|---:|---:|---:|---:|
| `improved_rl_isac` | 0.9599 | 53.48 | 200.93 | 278.90 | 0.1625 | 3.7402 |
| `improved_rl_no_isac` | 0.8843 | 95.33 | 296.64 | 300.00 | 0.4414 | 3.1097 |
| `rl_no_isac` | 0.8853 | 103.89 | 298.95 | 300.00 | 0.4299 | 3.3580 |
| `uniform_random` | 0.8469 | 111.71 | 300.00 | 300.00 | 0.4307 | 2.7614 |
| `skyorbs_like_skip_scan` | 0.4887 | 186.86 | 300.00 | 300.00 | 0.4166 | 1.4812 |

相对 `uniform_random`：发现率 +11.30 个百分点，平均时延降低 52.12%，空扫率降低 62.26%。

### Random Waypoint 10ep

| 协议 | 发现率 | 平均时延 | P95 | P99 | 空扫率 | lambda2 |
|---|---:|---:|---:|---:|---:|---:|
| `improved_rl_isac` | 0.9844 | 42.20 | 128.86 | 235.57 | 0.0878 | 9.3798 |
| `improved_rl_no_isac` | 0.8601 | 105.57 | 300.00 | 300.00 | 0.4288 | 6.3081 |
| `rl_no_isac` | 0.9034 | 96.95 | 284.28 | 300.00 | 0.4012 | 6.9851 |
| `uniform_random` | 0.8866 | 105.11 | 300.00 | 300.00 | 0.3917 | 6.8026 |
| `skyorbs_like_skip_scan` | 0.3160 | 233.15 | 300.00 | 300.00 | 0.3949 | 0.8605 |

相对 `uniform_random`：发现率 +9.78 个百分点，平均时延降低 59.85%，空扫率降低 77.59%。

## 论文图

本轮生成 54 张 PNG，全部为 4:3 比例、1920x1440 像素、300 dpi。字体校验结果：Times New Roman 可用，字体文件为 `C:\Windows\Fonts\times.ttf`。配色统一为 Okabe-Ito 系列，协议顺序固定为：SkyOrbs-like、Random、RL-NoISAC、Improved-RL、Improved-RL+ISAC。

训练图包括：`train_score_curve.png`、`train_discovery_rate_curve.png`、`train_empty_scan_ratio_curve.png`、`train_delay_curve.png`、`train_collision_curve.png`、`train_connectivity_curve.png`。

测试图对四个运动模型分别覆盖：发现率、空扫率、平均时延、P90/P95/P99 时延、最大连通分量、lambda2、碰撞数、发现边数、slot 级发现边数、slot 级连通性。

## 复现命令

```powershell
$env:PYTHONPATH = "tmp\pydeps;05_simulation\src"
$env:MPLCONFIGDIR = "tmp\mplconfig"

python 05_simulation/run_training.py --config 05_simulation/configs/paper_core_d1.yaml --output 05_simulation/results_raw/final_cem_train_g10_p10 --generations 10 --population 10 --episodes 2 --slots 300 --seeds 20260704,20261713,20262722 --test-seeds 20270704,20271713,20272722 --test-episodes 2 --training-seed 20260704

python 05_simulation/run_trained_eval.py --config 05_simulation/configs/paper_core_d1.yaml --trained-config 05_simulation/results_raw/final_cem_train_g10_p10/best_config.yaml --output 05_simulation/results_raw/final_eval_gm_20ep --episodes 20 --seed 20280704 --mobility gauss_markov
python 05_simulation/run_trained_eval.py --config 05_simulation/configs/paper_core_d1.yaml --trained-config 05_simulation/results_raw/final_cem_train_g10_p10/best_config.yaml --output 05_simulation/results_raw/final_eval_rw_20ep --episodes 20 --seed 20280704 --mobility random_walk
python 05_simulation/run_trained_eval.py --config 05_simulation/configs/paper_core_d1.yaml --trained-config 05_simulation/results_raw/final_cem_train_g10_p10/best_config.yaml --output 05_simulation/results_raw/final_eval_rd_10ep --episodes 10 --seed 20280704 --mobility random_direction
python 05_simulation/run_trained_eval.py --config 05_simulation/configs/paper_core_d1.yaml --trained-config 05_simulation/results_raw/final_cem_train_g10_p10/best_config.yaml --output 05_simulation/results_raw/final_eval_rwp_10ep --episodes 10 --seed 20280704 --mobility random_waypoint

python 06_analysis/scripts/plot_paper_results.py 05_simulation/results_raw/final_eval_gm_20ep_analysis/tables 05_simulation/results_raw/final_eval_rw_20ep_analysis/tables 05_simulation/results_raw/final_eval_rd_10ep_analysis/tables 05_simulation/results_raw/final_eval_rwp_10ep_analysis/tables --training-dir 05_simulation/results_raw/final_cem_train_g10_p10 --output 06_analysis/paper_figures/final_round1
```

## 当前结论

在四个动态运动模型下，`improved_rl_isac` 同时保持最高发现率、最低或接近最低的长尾发现时延、最低空扫率和较高拓扑连通性。结果支持论文主张：将 ISAC 抽象为链路层可用的 beam occupancy prior 后，可以显著压缩三维窄波束邻居发现的无效搜索空间；在有限时间资源下，数据驱动策略与 ISAC 先验结合优于纯随机、无 ISAC RL 以及参考扫描序列近似基线。
