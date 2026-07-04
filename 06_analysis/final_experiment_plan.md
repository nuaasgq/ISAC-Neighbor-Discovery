# 最终实验设计 v2

日期：2026-07-04

## 研究问题

在完全分布式、对准前无邻居先验的位置/轨迹信息条件下，如何利用 ISAC 通信伴随感知能力为链路层窄波束邻居发现提供 beam occupancy prior，从而减少空波位扫描、压缩发现长尾，并优先建立支撑拓扑连通性的关键链路。

## 假设

H1：ISAC beam prior 可以显著降低空扫率，从而缩短平均和长尾发现时延。

H2：仅依赖随机或无 ISAC 学习策略不足以稳定解决三维高维窄波束搜索；引入 memory/topology 结构后会提升，但仍弱于 ISAC + 结构化策略。

H3：共享策略参数可以从小规模训练迁移到不同动态运动模型，性能退化可控。

## 方法矩阵

1. `skyorbs_like_skip_scan`：参考文献近似基线，模拟三维确定性 skip-scan。
2. `uniform_random`：完全随机 mode/beam 基线。
3. `rl_no_isac`：强化学习代理，无 ISAC belief。
4. `improved_rl_no_isac`：memory/topology-aware 代理，无 ISAC belief。
5. `improved_rl_isac`：ISAC beam belief + memory/topology-aware 代理。

## 训练设计

当前训练器采用 CEM/black-box shared-policy search，优化 `improved_rl_isac` 的共享协议参数。该训练器用于先得到可写论文的数据闭环，后续真实 MARL 训练器可以继承同一环境、指标和评估脚本。

训练命令：

```powershell
python 05_simulation/run_training.py --config 05_simulation/configs/paper_core_d1.yaml --output 05_simulation/results_raw/final_cem_train_g10_p10 --generations 10 --population 10 --episodes 2 --slots 300 --seeds 20260704,20261713,20262722 --test-seeds 20270704,20271713,20272722 --test-episodes 2 --training-seed 20260704
```

## 测试设计

| 场景 | mobility | episodes | 目的 |
|---|---|---:|---|
| D1-GM | `gauss_markov` | 20 | 主结果，连续相关机动 |
| D1-RW | `random_walk` | 20 | 随机游走鲁棒性 |
| D1-RD | `random_direction` | 10 | 方向保持型动态鲁棒性 |
| D1-RWP | `random_waypoint` | 10 | 航点型集群运动鲁棒性 |

所有测试统一使用五类协议、300 slots、相同 trained config，并固定 seed `20280704` 作为本轮可复现测试切分。

## 性能指标

核心指标：`discovery_rate`、`mean_discovery_delay`、`p90_discovery_delay`、`p95_discovery_delay`、`p99_discovery_delay`、`empty_scan_ratio`、`lcc_ratio`、`lambda2`、`discovered_edges`、`collision_count`。

训练指标：score、发现率、空扫率、平均/P95 时延、碰撞、lambda2/lcc、共享参数演化。

## 图形规范

论文图统一使用 4:3 比例、`figsize=(6.4, 4.8)`、300 dpi、Times New Roman、Okabe-Ito 统一配色。协议顺序固定为 `skyorbs_like_skip_scan`、`uniform_random`、`rl_no_isac`、`improved_rl_no_isac`、`improved_rl_isac`。

## 成功门槛

`improved_rl_isac` 在主场景和至少两个鲁棒性场景中需要满足：

1. 发现率最高或与最高值差距小于 2 个百分点。
2. 平均发现时延低于 `uniform_random` 和 `improved_rl_no_isac`。
3. 空扫率低于全部无 ISAC 方法。
4. P95 或 P99 至少一项明显优于 `uniform_random` 和 `improved_rl_no_isac`。
5. `lambda2` 不低于 `improved_rl_no_isac` 的同量级表现。

本轮结果满足上述门槛。
