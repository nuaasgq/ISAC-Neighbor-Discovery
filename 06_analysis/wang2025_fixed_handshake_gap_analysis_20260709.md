# Wang2025 复现差异与 N=10 修正重跑结果

日期：2026-07-09

## 结论

修正 ISAC candidate pool 之后，single-RF 公平对比环境下的 Wang baseline 很低是合理的，不是随机错误或 sensing 完全失效。核心原因是当前公平对比把所有方法限制为单 RF、每 slot 一个实际通信波束、只把直接 TX/RX 握手计入发现；而王为栋论文的主要结果依赖多 RF/多波束、邻居表间接发现、以及“全网建网耗时”口径。

换句话说：

- 用 single RF 做公平 MARL 对比时，Wang 的发现率被物理动作空间压低。
- 用 paper-like RF=3/6 条件时，Wang 结果会立刻接近论文趋势。
- 因此后续论文实验必须分成两套：公平单 RF 对比，以及 Wang paper-like 复现对比。

## 本轮代码修正

修正位置：

- `05_simulation/src/isac_nd_sim/simulator.py`
- `05_simulation/tests/test_marl_env_contract.py`
- `06_analysis/scripts/run_wang2025_aligned_marl_matrix.py`

关键修正：

- ISAC candidate pool 不再作为额外握手波束。候选池只能影响后续选波和观测，不能在未实际选中波束时直接产生发现。
- 增加回归测试：即使 candidate pool 包含正确波束，只要实际 TX/RX 波束没选中，就不能发现。
- Wang-aligned 矩阵脚本支持 `--action-policies`，本轮只跑 `uniform_trx_idle_random` 和 `wang2025_isac_tables`。
- 脚本增加 baseline-first 流程：先输出 `baseline_target_metrics.csv`，再训练 MARL，并输出 `training_eval_gap.csv`。

## N=10 修正后公平对比结果

配置：N=10，15x7=105 beam cells，single RF，200 slots，5 evaluation episodes，common environment 为 `wang2025_isac_tables`。

结果目录：

- `06_analysis/paper_tables/wang2025_aligned_n10_fixedhandshake_20260709/`
- `05_simulation/results_raw/marl_campaign/wang2025_aligned_n10_fixedhandshake_20260709/`

| 方法 | 发现率均值 | 发现率标准差 | 平均时延 | 说明 |
|---|---:|---:|---:|---|
| MARL + Wang ISAC tables | 0.4711 | 0.0259 | 153.8 | 修正后最终随机评估 |
| Wang sensing-table action policy | 0.0978 | 0.0178 | 185.54 | 5 episodes baseline |
| Budgeted ISAC rule | 0.0622 | 0.0356 | 194.06 | 旧规则型方法，5 episodes baseline |
| Uniform TX/RX/IDLE random | 0.0000 | 0.0000 | 200.0 | 5 episodes baseline |

另做 20 episodes baseline 复核：

| 方法 | 发现率均值 | 发现率标准差 | 平均时延 |
|---|---:|---:|---:|
| Budgeted ISAC rule | 0.0633 | 0.0360 | 194.14 |
| Wang sensing-table action policy | 0.0567 | 0.0310 | 192.77 |
| Uniform TX/RX/IDLE random | 0.0089 | 0.0163 | 199.09 |

随机发现率低是符合理论的。single RF、105 beams、TX/RX 各 0.5 时，一条边单 slot 直接对准概率约为 `2*0.5*0.5/105^2 = 4.54e-5`，200 slots 期望发现率约 `0.0090`，与 20 episodes 复核结果一致。

旧规则 `Budgeted ISAC rule` 在 20 episodes 下略高于 Wang baseline，但数量级仍接近，说明在 single-RF、直接握手发现口径下，规则型 TX/RX 预算和碰撞规避只能缓解搜索问题，不能根本突破双向窄波束同时对准的乘法瓶颈。

## 训练周期差距

文件：`06_analysis/paper_tables/wang2025_aligned_n10_fixedhandshake_20260709/training_eval_gap.csv`

以 5 episodes 正式 baseline 中最强的 Wang baseline 发现率 `0.0978` 为目标线：

| Eval after episode | MARL eval 发现率 | 相对最强 baseline 差距 |
|---:|---:|---:|
| 25 | 0.4222 | +0.3244 |
| 50 | 0.4000 | +0.3022 |
| 75 | 0.3667 | +0.2689 |
| 100 | 0.4222 | +0.3244 |

这说明当前 MARL 在严格 single-RF 直接发现口径下明显优于 Wang/随机，但训练曲线不算稳定，后续仍需要更好的网络结构和奖励设计。

## 为什么王论文结果更好

王论文中的关键条件：

- 第 2.2 节：节点采用多射频 UPA 多天线结构，发送节点向各 RF 链输入相同 Hello 数据流，形成多波束。
- 第 4.2 节：每 slot 随机选择 `rf (rf <= NRF)` 个标志位为 1 的波束方向作为收发方向。
- 第 4.2 节：成功交互时交换邻居发现列表和感知列表；通信协同可把对方邻居表中未知节点写入本节点邻居表，即间接发现。
- 第 5 节表 4：RF 数为 1~6，波束宽度 25 度，时隙上限 200。
- 图 11 说明文字：多波束传输数量设为 3，比较无协同、通信协同、通感协同的建网耗时。
- 论文主要画的是“消耗时隙数/建网耗时”，不是当前直接边发现率 `discovered_edges / true_edges_seen`。

因此，当前 Wang single-RF baseline 不是王论文原始方法，而是为了和 MARL 在相同动作空间下公平比较的受限版本。

## RF 探针结果

为验证差距原因，固定 N=10、105 beams、200 slots、Wang sensing-table policy，仅改变 RF 数。

结果目录：

- `06_analysis/paper_tables/wang2025_rf_probe_20260709/`

| RF chains | 发现率均值 | 发现率标准差 | 平均时延 | lambda2 |
|---:|---:|---:|---:|---:|
| 2 | 0.5333 | 0.0757 | 134.44 | 1.929 |
| 3 | 0.8178 | 0.0442 | 86.06 | 4.885 |
| 6 | 0.9756 | 0.0232 | 37.86 | 8.500 |

这个探针基本解释了性能差距：Wang paper-like 方法在 RF=3/6 下确实能很快发现大量邻居；single RF 才是性能被压低的主因。

## 当前复现与论文的主要差异

高影响：

- 当前公平对比脚本强制 `rf_chains=1`；论文主结果使用多 RF/多波束，图 11 设置多波束传输数量为 3。
- 当前只把直接 TX/RX 握手成功计入 `discovered_edges`；论文通信协同会把邻居表中的未知节点写入本节点邻居表，属于间接发现。
- 当前主要指标是直接边发现率；论文主要指标是全网邻居表完成所需时隙。

中影响：

- 当前感知表是 per-beam belief/success/fail 的抽象；论文感知表包含目标位置、波束编号、Flag、潜在目标数、交互目标数和 SNR。
- 当前节点完成部分发现后仍按概率 TX/RX；论文中若节点已完成邻居发现，只进入接收状态并监听邻节点方向。
- 当前 RF>1 碰撞模型按 RX 节点聚合，可能比论文“同一波束方向冲突”更严格。

低影响：

- 当前保留 Gauss-Markov 运动，但 200 slots x 5 ms 只有 1 s，在 10 km 尺度下影响较小。
- 当前通信/感知范围大于 10 km 立方体对角线，和论文“任意节点在通信范围内”一致。

## 下一步建议

1. 保留当前 single-RF 公平 MARL 对比，不再把它叫做 Wang 原论文复现。
2. 新增 `wang2025_paperlike` 入口：RF=3，paper-like 邻居表间接发现，输出建网完成时隙。
3. 修正 RF>1 冲突模型为按 `(rx_node, rx_beam)` 判断冲突。
4. 指标拆开：`direct_edge_discovery_rate`、`neighbor_table_completion_ratio`、`networking_completion_slot`。
5. 再在 paper-like Wang 环境中比较我们的 MARL：先让 MARL 也能在同等多 RF 或抽象多波束能力下运行，否则比较会变成物理层能力差异，不是协议/学习方法差异。
