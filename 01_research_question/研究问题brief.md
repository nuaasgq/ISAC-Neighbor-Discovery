# 研究问题 Brief v2

## 当前阶段

P3-P5 过渡：系统模型、协议设计、MARL 方法和可扩展仿真实验设计。

## 主研究问题 v2

在完全分布式、无中心控制、节点自身定位姿态可用但对准前未知邻居状态的三维窄波束 UAV-UAV 集群网络中，如何将 ISAC 感知得到的 beam-cell occupancy prior 与自身位置/姿态状态联合抽象为链路层可用信息，使邻居发现从均匀盲扫转变为规则-数据双驱动的非均匀概率搜索，并通过可迁移的多智能体强化学习策略在小规模训练后直接部署到大规模 UAV 集群中？

## 核心设定

**Self-localized but neighbor-unknown fully distributed UAV swarm.**

每个 UAV 可知道：

- 自身位置、速度、航向和姿态。
- 自身 beam codebook 与当前姿态下的空间指向映射。
- ISAC 感知得到的候选目标 beam-cell prior。
- 已完成握手的邻居信息。

每个 UAV 对准前不知道：

- 未发现邻居身份。
- 未发现邻居实时位置、速度和姿态。
- 未发现邻居当前 Tx/Rx/Sense 模式。
- 未发现邻居波束指向。
- 全局拓扑和全局连通性指标。

## 子问题

1. **ISAC 先验建模：** 在不假设已知邻居身份和实时状态的前提下，ISAC 能向链路层提供哪些合法信息？例如候选波位集合、占据概率、粗距离/多普勒、置信度、虚警率、漏检率、角度误差和更新时间。
2. **自定位辅助：** 自身位置和姿态如何帮助节点把 body-frame beam cell 映射到 global-frame 空间方向，并辅助扫描策略，而不等价于提前知道邻居状态？
3. **邻居发现调度：** 如何将 beam-cell occupancy prior、自身状态和已发现局部拓扑转换为完全分布式的 sensing / Tx / Rx / beam selection 策略，使其优于均匀随机扫描和确定性盲扫？
4. **误感知鲁棒性：** 在 multipath、遮挡、虚警、漏检、非通信目标和角度偏差存在时，如何避免错误剪枝导致邻居永久不可发现？
5. **可迁移 MARL：** 如何设计固定维度局部观测、参数共享、图/注意力聚合或 mean-field 表示，使小规模训练的策略能迁移到更大规模 UAV 集群？
6. **协同收益验证：** 发现时延降低是否能进一步转化为连通性、代数连通度代理或一致性收敛速度提升？

## Scope

### In Scope

- UAV-UAV 自组织网络。
- 完全分布式邻居发现。
- 节点自身位置、速度、航向和姿态可用。
- 对准前未知邻居状态。
- 三维窄波束扫描。
- ISAC 感知能力的链路层抽象。
- 规则启发式协议和多智能体强化学习协议。
- 小规模训练到大规模部署的策略迁移。
- 拓扑质量和一致性收敛相关指标。
- ISAC 虚警、漏检、角度误差敏感性。

### Out of Scope

- 纯物理层最优波形设计。
- BS-UAV 蜂窝接入作为主系统。
- 已建立链路后的 beam tracking 作为主问题。
- 依赖中心控制、全局调度或对准前邻居状态共享的方案。
- 假设所有节点在发现前共享彼此实时位置、轨迹或波束状态。
- 假设 ISAC 完美知道邻居位置或身份。

## 当前判断

方向可做，但主贡献必须从“ISAC 辅助波束对准”收窄为：

1. ISAC-assisted beam-cell pruning for pre-alignment neighbor discovery。
2. Self-localization-aware but neighbor-unknown discovery model。
3. Rule-guided and MARL-based uncertainty-aware beam/mode scheduling。
4. Topology-aware finite-time neighbor prioritization。
5. Small-to-large swarm transfer via scalable local observation and shared policy。
