# Round2 Transfer Results Index

日期：2026-07-05

## 核心设置

- 训练规模：`N=10`，`beamwidth=10 deg`，Gauss-Markov 运动，`slot_duration=5 ms`。
- 训练幕长：`1200 slots`。CEM 设置为 `5 generations x 8 population`，训练 seed 为 `20260704`，held-out seeds 为 `20270704, 20271713`。
- 迁移测试：`N=10,20,50,100`，`beamwidth=3,5,10,15,30 deg`。
- 区域缩放：`density` 等密度扩展与 `fixed` 固定区域高密度扩展。
- 首轮距离设置：`range_mode=singlehop`，对每个区域尺寸计算对角线 `D`，令 `R_c=R_s=1.05D`。该设置只用于让通信距离和感知距离都不成为瓶颈，不代表物理上默认 `R_s=R_c`。

## 通信/感知距离依据

通信距离与感知距离不能简单绑定。Friis 传输公式给出自由空间通信链路的一跳传播损耗；单站雷达/ISAC 感知通常包含往返传播和目标反射，雷达方程含 `R^4` 项。因此本文实验采用两步走：

1. 首轮 sanity：`R_c` 与 `R_s` 都大于区域对角线，验证邻居发现机制本身。
2. 后续敏感性：扫 `R_c/D` 与 `R_s/R_c`，覆盖 `R_s<R_c`、`R_s=R_c`、`R_s>R_c`。

可引用依据：

- Friis, "A Note on a Simple Transmission Formula," Proceedings of the I.R.E., 1946. `https://capmimo.ece.wisc.edu/capmimo_papers/friis_original_1946.pdf`
- MIT Lincoln Laboratory, "The Radar Equation." `https://www.ll.mit.edu/media/6946`
- Bomfin et al., "A System Level Analysis for Integrated Sensing and Communication," arXiv:2402.00750, 2024. `https://arxiv.org/abs/2402.00750`

## 训练结果

训练输出目录：`05_simulation/results_raw/transfer_train_n10_b10_singlehop_candidate_g5_p8`

归档表格：`06_analysis/paper_tables/round2_transfer/training`

最优共享参数：

| 参数 | 值 |
|---|---:|
| `alpha_occupancy` | 1.7290 |
| `softmax_beta` | 2.6355 |
| `exploration_floor` | 0.1408 |
| `confidence_decay` | 0.9874 |
| `piggyback_sensing_period_multiplier` | 0.6288 |

Held-out `all`：

| 指标 | 值 |
|---|---:|
| `reward_mean` | 104.1822 |
| `discovery_rate_mean` | 0.8444 |
| `discovered_edges_mean` | 38.0 / 45 |
| `mean_delay_censored_mean` | 485.19 slots |
| `empty_scan_ratio_mean` | 0.8957 |
| `lambda2_mean` | 6.0195 |
| `lcc_ratio_mean` | 1.0 |

训练图目录：`06_analysis/paper_figures/training_round2_candidate`

关键训练图：

- `train_reward_curve.png`
- `train_score_curve.png`
- `train_discovery_rate_curve.png`
- `train_empty_scan_ratio_curve.png`
- `train_connectivity_curve.png`

## 迁移矩阵

原始结果：

- `05_simulation/results_raw/transfer_matrix_singlehop_density_600slot`
- `05_simulation/results_raw/transfer_matrix_singlehop_fixed_600slot`

归档表格：

- `06_analysis/paper_tables/round2_transfer/density_600slot`
- `06_analysis/paper_tables/round2_transfer/fixed_600slot`

迁移图目录：`06_analysis/paper_figures/transfer_round2_600slot`

生成图数：111 张 PNG，全部为 `6.4 x 4.8 in`、300 dpi、4:3 比例、Times New Roman 字体族、统一 Okabe-Ito 配色。

### N=100 代表结果

`N=100, beamwidth=10 deg`：

| area scale | discovery rate | discovered edges | lambda2 | LCC |
|---|---:|---:|---:|---:|
| density | 0.3644 | 1804 / 4950 | 10.7912 | 1.0 |
| fixed | 0.3471 | 1718 / 4950 | 10.7137 | 1.0 |

`N=100` 的波束宽度趋势：

- `3 deg`：发现率约 1%，说明极窄波束仍是主要困难区。
- `5 deg`：发现率约 7-8%，但 fixed/density 下基本可形成较大连通分量。
- `10 deg`：可实现全连通，发现率约 35-36%。
- `15 deg`：发现率约 52-55%，lambda2 明显提升。
- `30 deg`：空扫率最低，但发现率未必最高，说明高密度下碰撞与候选竞争开始影响收益。

## 当前可写论文结论

1. ISAC 不应被建模为对全局 codebook 的先验，而应是由当前握手波束触发的局部 candidate-set refinement。
2. 在 10 deg 小规模训练后，模型能零样本迁移到 `N=100`，并在 `10/15/30 deg` 形成全连通拓扑。
3. 等密度扩展与固定区域高密度扩展的差异不大，`density` 在 `N=100, 10 deg` 略优，有利于支撑可扩展性叙事。
4. `3 deg` 和部分 `5 deg` 场景仍不足，后续应作为极窄波束 stress test，并考虑更强的多智能体结构或更长 discovery horizon。

