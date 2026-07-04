# Gap Analysis

## 目标

识别本研究相对于以下三类工作的稳定差异：

1. 传统窄波束邻居发现。
2. UAV / airborne directional neighbor discovery。
3. ISAC-assisted beam management / beam tracking / beam rendezvous。

## 初始 Gap 假设

| 相关方向 | 已有重点 | 可能缺口 | 本研究切入点 |
|---|---|---|---|
| Directional neighbor discovery | 扫描序列、随机/确定性发现、异步无先验 | 三维窄波束搜索空间大，通常没有 ISAC 感知先验 | ISAC 辅助波位置信图降低空扫 |
| UAV beam tracking | 已建立或可预测链路的波束维护 | 不解决对准前未知邻居发现 | 将 discovery 与 tracking 区分 |
| ISAC beam management | 物理层 beamforming / initial access / BS-UAV | 多为蜂窝或单链路，不强调分布式拓扑形成 | UAV-UAV 分布式链路层协议 |
| Topology-aware networking | 连通性、关键链路、一致性收敛 | 通常不处理窄波束发现过程 | 有限发现预算下优先关键链路 |

## 待确认问题

- 是否已有 ISAC-assisted neighbor discovery 明确针对 fully distributed UAV-UAV 网络？
- 是否已有 beam rendezvous 工作已经包含对准前双向发现和协议调度？
- 是否已有工作把拓扑质量直接纳入窄波束邻居发现阶段？
