# MVP 仿真实验计划 v0

## 目标

建立离散 beam-cell 级仿真，验证 I-TAP-ND 是否在不使用 oracle 信息的前提下，相比盲扫和 ISAC-only 方法获得发现效率与拓扑质量收益；随后验证 MARL-I-TAP-ND 是否能在小规模训练后迁移到大规模 UAV 集群。

从 Sprint 5 起，MVP 主场景采用动态 UAV。`static` 仅作为调试模式，不作为主实验设定。

## MVP 问题

第一版只回答三个问题：

1. ISAC occupancy prior 是否减少空扫和平均发现时延？
2. 探索下界是否能缓解漏检导致的长尾问题？
3. topology-aware priority 是否在有限时隙内提升已发现拓扑质量？
4. MARL 策略是否能在 `N_train={10,20}` 训练后零微调迁移到 `N_test={30,50,100}`？

## 任务拆解

| 任务 | 内容 | 是否可并行 | 依赖 |
|---|---|---:|---|
| T1 模型定义 | 网络、波束、时隙、ISAC 误差、发现条件 | 否 | 无 |
| T2 协议基线 | 统一实现 7 类协议接口 | 是 | T1 |
| T3 指标模块 | 时延、空扫、连通性、lambda2、一致性收敛 | 是 | T1 |
| T4 参数扫描 | 节点数、波束宽度、误差、速度 | 是 | T1 |
| T5 MVP 跑通 | 固定场景 + 多 seed + 所有 baselines | 否 | T2/T3 |
| T6 MARL 训练接口 | MAPPO/IPPO 环境接口和规则专家轨迹 | 是 | T1/T2/T3 |
| T7 规模迁移评估 | 小规模训练，大规模零微调测试 | 否 | T6 |
| T8 结果判定 | 是否继续推进、是否调整创新点 | 否 | T5/T7 |

## 仿真对象

### 网络

- 节点数：MVP `N=30`；扫描 `N in {10, 20, 30, 50, 80}`
- 空间：三维长方体，例如 `1000m x 1000m x 300m`
- 位置：每个 episode 随机生成
- 移动：MVP 主场景采用高斯马尔可夫运动；随机方向、随机游走、随机航点用于鲁棒性扫描；边界默认反弹
- 真实邻居图：节点距离 `d_ij <= R_comm`
- 协议可见信息：本地 belief、已发现邻居、握手后局部信息
- 节点自状态：自身位置、速度、航向和姿态可用

### 三维运动模型

| 模型 | 用途 | 最小更新形式 |
|---|---|---|
| Gauss-Markov | 主场景，模拟平滑且时间相关的 UAV 运动 | `v(t+1)=alpha v(t)+(1-alpha) v_bar+sqrt(1-alpha^2) sigma_v w(t)`，`x(t+1)=x(t)+v(t+1) Delta t` |
| Random Walk | 局部抖动和强随机压力测试 | `x(t+1)=x(t)+eta(t)`，`eta(t) ~ N(0, sigma_step^2 I)` |
| Random Direction | 分段恒速随机方向运动 | 每 `T_dir` 个 slot 重采样三维方向和速度，其余时隙保持速度 |
| Random Waypoint | 任务式航点运动 | 随机选择三维航点，按速度飞向航点，到达后可短暂停留并重采样 |

姿态采用简化运动学映射：`yaw=atan2(v_y,v_x)`，`pitch=atan2(v_z,sqrt(v_x^2+v_y^2))`，`roll` 默认慢衰减到 0。后续如需要更真实机动，可加入基于转弯率的 bank angle 模型。

### 波束

- 本地水平 beam 数：`A in {12, 24, 36, 72}`
- 本地垂直 beam 数：`E in {4, 8, 12}`
- beam cell 总数：`M = A * E`
- 对准条件：TX 选择覆盖目标的本地 cell，RX 选择覆盖发送方的本地 cell
- 碰撞：MVP 默认同一 RX beam 内多个 TX 则失败

### 时隙

- 每个 episode `T` 个 discovery slots
- 每个节点独立选择 `SENSE / TX / RX / IDLE`
- MVP 使用同步时隙但无中心调度

### ISAC 误差

- 虚警率：MVP `0.02`；扫描 `p_fa in {0.00, 0.01, 0.05, 0.10}`
- 漏检率：MVP `0.15`；扫描 `p_md in {0.00, 0.10, 0.30, 0.50}`
- beam-cell 角度偏移：MVP `0.5 cell`；扫描 `sigma_cell in {0, 0.25, 0.5, 1.0, 2.0}`
- 感知刷新周期：MVP `5 slots`
- belief 更新：`q_i,b(t+1) = (1-rho) q_i,b(t) + rho * y_i,b(t)`

## Baselines

| 方法 | 说明 | 用途 |
|---|---|---|
| Uniform Random ND | 均匀随机选择模式和 beam cell | 随机盲扫基线 |
| Deterministic Scan ND | 固定顺序扫描 beam cells | 确定性扫描基线 |
| CRT/Oblivious-like ND | 简化 CRT/周期序列近似 | 无先验有界扫描基线 |
| SkyOrbs-like Skip Scan | 简化 3D 有序 skip-scan，先粗扫再细扫，不用 ISAC prior | 3D UAV DND 强基线近似 |
| ISAC-only ND | 使用 occupancy prior，不用 topology proxy | 验证 ISAC 剪枝收益 |
| Topology-only ND | 使用方向稀缺/度数代理，不用 ISAC prior | 验证 topology proxy 独立收益 |
| I-TAP-ND | 完整方法 | 主方法 |
| IQL/DRQN-I-TAP | 独立 Q 学习 + GRU 历史 | 值函数 sanity baseline |
| VDN-I-TAP | 加性值分解 | 轻量 team reward baseline |
| QMIX-I-TAP | 单调 mixing 值分解 | 经典 CTDE 值函数 baseline |
| QPLEX/Qatten-I-TAP | 增强值分解或 attention mixer | 复杂拓扑/碰撞信用分配候选 |
| MAVEN-I-TAP | 值函数 + committed exploration | 稀疏奖励和长尾发现候选 |
| IPPO-I-TAP | 参数共享 IPPO 学习 mode/beam | 轻量策略梯度 baseline |
| MAPPO-I-TAP | CTDE + 参数共享 MAPPO | 强策略梯度 baseline |
| HAPPO/HATRPO-I-TAP | sequential trust-region PG | 稳定策略更新候选 |
| COMA/MAAC/MASAC-I-TAP | counterfactual / attention / maximum-entropy actor-critic | 信用分配和探索候选 |
| MAT-I-TAP | multi-agent transformer / sequence modeling | 结构创新候选 |
| Oracle ND | 已知真实 occupied beam cells | 上界，不参与公平协议比较 |

## Metrics

### 发现效率

- 平均发现时延
- P90 / P95 / P99 发现时延
- 有限时间发现率
- 空扫比例
- 每发现一条链路所需时隙数

### 拓扑质量

- 已发现链路数
- 连通分量数
- 最大连通子图规模
- 代数连通度 `lambda_2`，仅评估用
- 关键链路发现率，后续定义

### 协同一致性

- 一致性误差曲线
- 达到误差阈值所需时间
- 相同 discovery budget 下的收敛速度
- 未发现链路时延采用 censor penalty：记为 `T_budget`，并单独报告发现率

### 鲁棒性

- 对 `p_fa` 的敏感性
- 对 `p_md` 的敏感性
- 对 `sigma_cell` 的敏感性
- 对 beam 数 `M` 的敏感性
- 对移动速度 `v in {0, 5, 15, 30}` m/s 的敏感性

### MARL 迁移性

- Zero-shot Gain
- Scale Retention
- 大规模每节点推理时间
- 最差 10% 节点发现率
- 孤立节点比例
- 固定密度扩展与固定区域压力测试

## 输出文件

建议每次运行输出：

```text
05_simulation/results_raw/
  mvp_002_dynamic_pruning_<timestamp>/
    config.yaml
    seed_manifest.json
    per_slot_metrics.csv
    per_episode_summary.csv
    discovered_edges.csv
    README.md
```

图表输出：

```text
06_analysis/figures/
  delay_mean_vs_beams.png
  delay_tail_vs_error.png
  empty_scan_ratio.png
  topology_quality_vs_time.png
  consensus_error_vs_time.png
```

## 首个 MVP 实验路径

首个动态实验：`mvp_002_dynamic_pruning`

固定设置：

- `N = 30`
- `A = 24, E = 8`
- `T_budget = 3000`
- 高斯马尔可夫三维运动，边界反弹
- `v_min=3 m/s, v_mean=15 m/s, v_max=30 m/s`
- `alpha=0.85, sigma_v=2 m/s`
- `p_fa = 0.02`
- `p_md = 0.15`
- `sigma_cell = 0.5`
- `50 seeds`
- 所有 baselines 同场景对比

第一轮只看四个结果：

1. `ISAC-only` 是否低于非 ISAC 基线的空扫比例。
2. `ISAC-only` 是否降低平均时延和 P95/P99。
3. `I-TAP-ND` 是否在发现率相近时提升 LCC、连通分量、lambda2 proxy。
4. `I-TAP-ND` 是否降低一致性收敛时间，且没有明显牺牲发现时延。

## MVP 判定标准

继续推进的最低标准：

1. `ISAC-only` 相比最佳非 ISAC 基线，平均时延下降不少于 30%，P95 下降不少于 25%，空扫比例下降不少于 40%。
2. `I-TAP-ND` 相比 `ISAC-only`，lambda2 proxy 或 LCC 提升不少于 10%，一致性收敛时间下降不少于 15%，平均时延损失不超过 10%。
3. Oracle ND 明显优于 I-TAP-ND，说明协议没有偷用 oracle。
4. 若只有 ISAC 有收益而 topology-aware 无收益，应收缩论文贡献。
5. 若只在 oracle 或极低误差下有效，应暂停进入复杂智能协议设计。

## Sprint 3 实现顺序

1. 数据结构：nodes、self state、beam cells、true topology、ISAC observations。
2. 基线：Uniform Random、Deterministic Scan、ISAC-only。
3. 规则主方法：I-TAP-ND。
4. 指标统计：delay、empty scan、discovery rate。
5. 拓扑指标：components、largest component、lambda_2。
6. 一致性仿真：基于已发现图运行简单平均一致性。
7. MARL 环境接口：observation、action、reward、done、info。
8. I-TAP-ND expert trajectory 生成。
9. IPPO/MAPPO 训练和小到大迁移评估。
10. 参数扫描和绘图。

## Sprint 5 动态仿真底座

当前代码入口：

```text
05_simulation/src/isac_nd_sim/
  config.py
  mobility.py
  beam.py
  simulator.py
  runner.py
```

运行 smoke 级实验：

```powershell
$env:PYTHONPATH='05_simulation/src'
python -m isac_nd_sim.runner --config 05_simulation/configs/mvp.yaml --output 05_simulation/results_raw/smoke_dynamic --episodes 2 --slots 100 --protocols uniform_random,isac_only,itap_nd
```

也可以直接使用仓库根入口和轻量配置：

```powershell
python 05_simulation/run_smoke.py --episodes 1 --slots 50 --protocols uniform_random,itap_nd --mobility gauss_markov
```

当前 runner 输出：

| 文件 | 内容 |
|---|---|
| `config.yaml` | 本次运行使用的配置副本 |
| `seed_manifest.json` | 每个 protocol-episode 的随机种子 |
| `per_episode_summary.csv` | 发现率、censored delay、空扫率、碰撞数、移动距离、LCC、`lambda2` 等 episode 级指标 |
| `per_slot_metrics.csv` | 每个 slot 的真实边数、累计发现边数、新发现边数、空扫率、连通分量、LCC、`lambda2` |
| `discovered_edges.csv` | 每条首次发现链路的端点、首次可见 slot、发现 slot 和时延 |
| `aggregate_metrics.json` | 按协议聚合的均值指标 |

运行测试：

```powershell
python -m pytest 05_simulation/tests -q
```

## Sprint 4 MARL 算法筛选顺序

配置入口：`05_simulation/configs/marl_algorithm_sweep.yaml`

### Phase 0：规则专家

- 跑通 I-TAP-ND。
- 输出规则 logits、expert trajectories 和 non-learning baseline metrics。

### Phase 1：轻量算法族筛选

| 候选 | 目的 | 淘汰规则 |
|---|---|---|
| IQL/DRQN | 检查纯本地值函数是否能利用 ISAC prior | 长期劣于 random 淘汰 |
| VDN | 检查 team reward 加性分解 | 劣于 IQL 或训练方差过大淘汰 |
| QMIX | 检查单调 mixing 是否足够 | 拓扑/碰撞指标劣于 VDN 时降级为 baseline |
| IPPO | 检查参数共享策略梯度 | 不稳定或空扫率高则淘汰 |
| MAPPO | 强 baseline | 不能直接定为主方法，只作为比较对象 |

### Phase 2：增强算法族筛选

- 值函数增强：QPLEX、Qatten、QTRAN++、MAVEN、ACE。
- 策略梯度增强：HAPPO、HATRPO、MAT。
- Actor-Critic：COMA、MAAC、MASAC / discrete SAC。

保留标准：

- 在 `N_train in [8,20]` 上稳定优于轻量 baseline。
- 在 `N_test in {30,50,100}` 上零微调保持正收益。
- 每节点 actor 推理复杂度不随全网 `N` 线性增长。

### Phase 3：结构创新优化

只在 Phase 2 胜出算法上加入：

- rule-residual beam attention。
- topology-aware attention critic / mixer。
- uncertainty-aware entropy control。
- scale-invariant neighbor pooling 或 beam-graph transformer。

最终论文方法必须证明：同一算法下，问题定制结构优于 vanilla 网络。
