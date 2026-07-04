# Sprint 4 任务拆解：MARL 算法族筛选与网络结构创新

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-07-04
- Verification Status: UNVERIFIED
- Version Label: sprint4_marl_algorithm_exploration_v0

## 目标

本轮不提前确定 `MARL-I-TAP-ND` 采用哪一种 MARL 算法，而是建立一套可执行的算法族筛选流程，在值函数、策略梯度、Actor-Critic 三类方法中横向比较，再引入面向 ISAC 邻居发现的网络结构创新。

最终目标是同时支撑两类贡献：

1. **机制创新**：ISAC beam-cell prior、链路层握手、拓扑优先、探索保底共同构成新的邻居发现机制。
2. **方法创新**：在机制之上设计适合三维窄波束邻居发现的可迁移 MARL 网络结构和训练流程。

## 新原则

- 不把 MAPPO 预设为唯一主方法。
- 先实现统一环境接口和规则专家，再比较算法族。
- 值函数类、策略梯度类、Actor-Critic 类都进入候选池。
- 网络结构创新应先绑定本问题的特殊输入：ISAC prior、自身姿态/位置、已发现邻居图、历史失败和拓扑代理。
- 小规模训练到大规模零微调迁移是硬约束，而不是附加实验。

## 并行任务线

| 任务线 | 内容 | 可并行 | 产物 |
|---|---|---:|---|
| A | 值函数/值分解算法族筛选 | 是 | `04_protocol/marl_algorithm_suite.md` |
| B | 策略梯度与 Actor-Critic 算法族筛选 | 是 | `04_protocol/marl_algorithm_suite.md` |
| C | 网络结构创新路线 | 是 | `04_protocol/neural_architecture_innovation.md` |
| D | 算法筛选实验矩阵 | 是 | `05_simulation/configs/marl_algorithm_sweep.yaml` |
| E | 文献与风险更新 | 是 | `02_literature/search_log.md`, `02_literature/gap_analysis.md` |
| F | 统一 MARL 环境接口设计 | 否，依赖 A-D | 后续 `05_simulation/src/` |

## 三阶段执行

### Phase 1：算法族 sanity check

目标是在小规模、低噪声、短训练预算下排除明显不适合的算法。

候选：

- Independent Q-Learning / DRQN。
- VDN。
- QMIX。
- IPPO。
- MAPPO。
- COMA 或 attention critic 简化版。

通过标准：

- 能稳定学习高于规则随机基线。
- 不明显破坏探索保底。
- 训练曲线无长期崩溃。

### Phase 2：增强算法族比较

在 Phase 1 的胜出算法上加入更强结构：

- QPLEX / QTRAN++ / Qatten / ACE / MAVEN。
- HAPPO / HATRPO。
- MAAC / MASAC。
- MAT 或轻量 multi-agent transformer。

通过标准：

- 在 `N_train in [8,20]` 上优于 I-TAP-ND 或至少学到与 I-TAP-ND 互补的策略。
- 在 `N_test in {30,50,100}` 上保持正收益。
- 推理输入维度不随全网 `N` 增长。

### Phase 3：问题定制网络结构

不再只换算法名，而是比较结构创新：

- ISAC beam-cell encoder。
- Neighbor set / discovered graph encoder。
- Temporal memory。
- Uncertainty-aware risk module。
- Rule-neural residual fusion。
- Scale-invariant pooling 或 mean-field summary。

通过标准：

- 相比同一算法的 vanilla 网络，有稳定收益。
- 收益来自 ISAC prior、拓扑目标和规模迁移，而不是训练预算堆叠。

## 当前执行检查点

1. 环境是否支持统一 `obs/action/reward/info` 接口？
2. mode 与 beam 是否采用分层动作，避免联合动作空间过大？
3. Critic 是否允许训练期使用全局真值，而 actor 执行期严格不使用？
4. 每类算法是否至少有一个轻量实现和一个增强实现？
5. 网络结构创新是否能被清晰消融？
6. 大规模测试是否零微调？
