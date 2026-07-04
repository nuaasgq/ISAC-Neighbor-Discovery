# 研究问题 Brief v1

## 当前阶段

P1 / Sprint 1：研究问题收敛与可做性判断。

## 主研究问题 v1

在完全分布式、无中心控制、无 GNSS/惯导/历史邻居状态辅助的三维窄波束 UAV-UAV 集群网络中，如何将 ISAC 感知得到的 beam-cell occupancy prior 抽象为链路层可用信息，使随机邻居发现从均匀盲扫转变为不确定性感知的非均匀概率搜索，并在有限发现时间内优先建立更有利于拓扑连通和一致性收敛的关键链路？

## 子问题

1. **ISAC 先验建模：** 在不假设已知邻居位置和身份的前提下，ISAC 能向链路层提供哪些合法信息？例如候选波位集合、占据概率、置信度、虚警率、漏检率、角度误差和更新时间。
2. **邻居发现调度：** 如何将 beam-cell occupancy prior 转换为完全分布式的 sensing / Tx / Rx / beam selection 策略，使其优于均匀随机扫描和确定性盲扫？
3. **误感知鲁棒性：** 在 multipath、遮挡、虚警、漏检和角度偏差存在时，如何避免错误剪枝导致邻居永久不可发现？
4. **拓扑质量优先：** 在对准前无法获得全局拓扑的条件下，如何设计本地可估计的拓扑价值代理，使有限发现预算优先用于关键方向或关键链路？
5. **协同收益验证：** 发现时延降低是否能进一步转化为连通性、代数连通度代理或一致性收敛速度提升？

## 初步 FINER 评估

| 维度 | 评分 | 理由 |
|---|---:|---|
| Feasible | 4/5 | 可以先做离散 beam-cell 级仿真，不需要完整物理层波形实现；ISAC 误差可参数化。 |
| Interesting | 5/5 | 三维窄波束邻居发现搜索空间爆炸是真问题，ISAC 提供新的跨层切入点。 |
| Novel | 4/5 | 单独的 ISAC beam alignment 和 3D UAV DND 已有强工作，但 fully distributed U2U ND + topology-aware priority 的组合缺口仍明显。 |
| Ethical | 4/5 | 属于通信网络协议研究；需避免涉及对抗用途细节和不可复现实验声明。 |
| Relevant | 5/5 | 直接关联 UAV swarm 自组网、窄波束接入、ISAC 和有限时间拓扑形成。 |
| **Average** | **4.4/5** | 具备继续推进价值。 |

## Scope

### In Scope

- UAV-UAV 自组织网络。
- 完全分布式邻居发现。
- 三维窄波束扫描。
- ISAC 感知能力的链路层抽象。
- 随机/概率式发现协议。
- 拓扑质量和一致性收敛相关指标。
- ISAC 虚警、漏检、角度误差敏感性。

### Out of Scope

- 纯物理层最优波形设计。
- BS-UAV 蜂窝接入作为主系统。
- 已建立链路后的 beam tracking 作为主问题。
- 依赖 GNSS、惯导、历史轨迹或中心控制的方案。
- 假设 ISAC 完美知道邻居位置或身份。

## 当前判断

方向可做，但主贡献必须明确表述为：

1. ISAC-assisted beam-cell pruning for pre-alignment neighbor discovery。
2. Uncertainty-aware non-uniform randomized discovery。
3. Topology-aware finite-time neighbor prioritization。

若只强调“ISAC 帮助波束对准”或“用学习方法选波束”，新颖性不足。
