# Gap Analysis v1

## 目标

识别本研究相对于以下三类工作的稳定差异：

1. 传统窄波束邻居发现。
2. UAV / airborne directional neighbor discovery。
3. ISAC-assisted beam management / beam tracking / beam rendezvous。

---

## 已确认的高风险近邻工作

| 类别 | 代表文献 | 威胁等级 | 为什么构成威胁 | 我们的差异点 |
|---|---|---|---|---|
| 3D UAV directional ND | SkyOrbs, IEEE TMC 2024 | High | 已覆盖 3D UAV、方向邻居发现、无强理想假设和快速扫描。 | 引入 ISAC 波位占据先验；从盲扫/skip scanning 转向感知引导概率搜索；加入拓扑质量目标。 |
| Airborne oblivious ND | Hong et al., Ad Hoc Networks 2023 | High | 已覆盖 airborne、无相对位置、异步、异构、匿名和 CRT 有界发现。 | 该文二维化并依赖扫描序列；本文关注三维波位置信图和有限时间拓扑形成。 |
| ISAC airborne beam rendezvous | Hong et al., Computer Communications 2024 | High | 已将 ISAC 和 airborne beam rendezvous 结合，题面很接近。 | 该文偏 beam prediction/rendezvous；本文研究对准前 neighbor discovery protocol、无已知通信对象和拓扑优先。 |
| Sensing-assisted protocol | Li et al., IEEE TMC 2024 NR-V2X | High | 已是 sensing-assisted frame/protocol design，且涉及 initial access。 | 该文是 gNB-V2I 基础设施场景；本文是 fully distributed UAV-UAV，无中心调度。 |
| Radar/context-assisted beam probing | CommRad 2024 / Orimogunje 2026 | High | “感知先验缩小 beam probing 集合”思想接近。 | 非 U2U、非邻居发现、非拓扑形成；本文需要处理双向发现、身份确认、随机 TX/RX 和拓扑质量。 |
| Learning-based DND | Wang et al., IEEE TVT 2024 | Medium | 已用 PFL/DDPG/MAML 改进方向邻居发现。 | 学习来自历史交互/训练，不是 ISAC 对准前感知；未解决三维空波位爆炸和拓扑优先。 |
| ISAC UAV-BS beam management | Cui et al., China Communications 2024; Xu et al., TWC 2026 | Medium-High | UAV + ISAC + beam management/beamforming 已有强工作。 | 它们是 BS 主导、已知服务关系或 IA/tracking；不是分布式 U2U neighbor discovery。 |
| Scalable wireless MARL | Naderializadeh et al., IEEE TWC 2021; Wang et al., IEEE TSP 2022 | High | 已明确讨论无线网络中小规模训练、不同规模迁移和 GNN 可迁移资源分配。 | 本文不能声称“首次可迁移无线 MARL”，应聚焦 ISAC prior 驱动的对准前 U2U 邻居发现。 |
| MARL MAC / access | Guo et al., IEEE JSAC 2022; MAPPO-MAC 2023 | High | 已将 MARL/MAPPO/QMIX 用于分布式链路层接入。 | 本文动作包含窄波束 sensing/Tx/Rx/beam 选择，目标是 neighbor discovery 和拓扑形成，不是通用 MAC。 |
| RL directional neighbor discovery | Enhanced RL two-way DND, Ad Hoc Networks 2024/2025 | High | 已用 RL 改进 two-way directional antenna neighbor discovery。 | 本文新增 ISAC beam-cell prior、自定位辅助、UAV-UAV 三维场景、拓扑优先和小到大迁移。 |
| General MARL algorithm families | VDN/QMIX/QPLEX/Qatten; MAPPO/HAPPO; COMA/MAAC/MAT | Medium-High | 值分解、策略梯度、Actor-Critic、Transformer MARL 都已有成熟算法，不能把基础算法本身包装成创新。 | 本文应做统一算法族筛选，并把创新放在 ISAC-aware 表征、topology-aware credit assignment、uncertainty-aware exploration 和 scale-invariant execution。 |

---

## 稳定 Gap

### G1. 从 beam alignment 转向 neighbor discovery

已有 ISAC 工作多聚焦 beam alignment、beam tracking、beam management、initial access 或 beam rendezvous。它们通常默认通信对象已知、由 BS/RSU/AP 主导，或至少已经存在服务关系。本文的问题是：在对准前、邻居身份未知、无中心调度时，节点如何发现潜在邻居。

### G2. 从基础设施辅助转向 fully distributed U2U

多数 sensing-assisted beam management 使用 BS、gNB、RSU、AP 或固定 radar 作为感知和决策中心。本文假设 UAV 集群完全分布式，每个节点只能基于本地感知和本地协议状态选择 sensing / Tx / Rx 及波束。

### G3. 从盲扫序列转向 ISAC beam-cell occupancy prior

传统 random / deterministic / quorum / CRT / skip scanning 方法主要优化扫描序列和工作模式匹配。本文将 ISAC 抽象为物理层能力，输出候选三维波位占据概率、置信度、虚警/漏检特性和更新时间，使链路层从均匀盲扫转为非均匀概率搜索。

### G4. 从发现时延单目标转向有限时间拓扑质量

现有 DND 指标主要是平均发现时延、最坏发现时延、发现率、miss detection、能耗或 beam management overhead。本文把有限时间内的拓扑质量作为目标之一：关键链路发现率、连通分量、代数连通度代理、一致性收敛速度。

### G5. 从完美感知转向误感知鲁棒协议

ISAC 感知角度不一定等于最佳通信方向，multipath、遮挡、虚警和漏检都会影响波位剪枝。本文不能把感知输出当作 oracle，而应引入置信度衰减、探索概率和链路层握手确认。

### G6. 从通用可迁移 MARL 转向 ISAC prior 驱动的可迁移邻居发现

无线网络中的可迁移 MARL、GNN 资源分配和分布式 MAC 已有较强工作。因此本文不能把“可迁移 MARL”本身作为唯一创新点。更稳妥的表述是：将可扩展 MARL 作为实现机制，服务于 ISAC-assisted pre-alignment narrow-beam neighbor discovery，并通过 topology-aware reward 和 beam-cell prior 融合体现领域特异性。

### G7. 从套用 MARL 算法转向问题定制结构

VDN、QMIX、QPLEX、MAPPO、HAPPO、COMA、MAAC 和 MAT 等算法已经是通用工具。本文不应声称基础优化器新，而应通过统一筛选证明算法选择合理，并进一步提出面向本问题的结构：ISAC beam-cell token、规则残差融合、拓扑注意力 critic / mixer、不确定性感知探索和可迁移邻居 pooling。

---

## 建议论文主张

**不是：**

> ISAC can help UAV beam alignment.

**而是：**

> ISAC-derived beam-cell occupancy priors can reshape fully distributed narrow-beam neighbor discovery from uniform blind scanning to uncertainty-aware probabilistic search, and topology-aware prioritization can improve finite-time swarm connectivity beyond delay-only discovery protocols.

---

## 需要继续验证的问题

1. Hong 2024 beam rendezvous 是否已经包含无已知通信对象的 discovery 过程？
2. SkyOrbs 2024 是否有任何可以等价为“空波位跳过”的先验机制？
3. Orimogunje 2026 adaptive beam probing 的 uncertainty-aware scheduling 是否可迁移为我们的主要算法基础，还是只是 baseline？
4. 拓扑质量指标是否能在完全分布式且未发现邻居前合理估计？
5. TWC/TCOM 审稿视角下，ISAC 抽象层是否足够具体，是否需要给出最小物理层感知误差模型？
6. MARL-I-TAP-ND 是否能在不借助大规模微调的情况下超过 I-TAP-ND 规则基线？
7. 可迁移 MARL 的新意是否足够依托 ISAC prior 和 neighbor discovery，而不是泛泛重复无线资源分配中的 GNN/MARL 迁移思想？
8. 在相同环境和训练预算下，值函数类、策略梯度类和 Actor-Critic 类哪个算法族更适合 mode+beam 邻居发现？
9. 网络结构创新相对于 vanilla MARL 的增益是否稳定，还是只来自更大模型和更长训练？
