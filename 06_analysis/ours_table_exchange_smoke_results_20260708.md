# 我方表交换短对比结果

## 设置

- 协议：`improved_rl_isac` vs. `improved_rl_isac_tables`。
- 节点数：20, 50。
- RF 链数：3。
- 每幕长度：200 slots。
- 重复次数：3 episodes。
- 输出目录：`06_analysis/paper_tables/ours_table_exchange_smoke_20260708`。
- 图目录：`06_analysis/paper_figures/ours_table_exchange_smoke_20260708`。

## 结果

| N | Protocol | Discovery rate | Empty scan ratio | Collision count | Lambda2 |
|---:|---|---:|---:|---:|---:|
| 20 | `improved_rl_isac` | 0.758 | 0.273 | 2802.0 | 11.12 |
| 20 | `improved_rl_isac_tables` | 0.798 | 0.262 | 2828.3 | 11.24 |
| 50 | `improved_rl_isac` | 0.401 | 0.136 | 10598.3 | 11.60 |
| 50 | `improved_rl_isac_tables` | 0.413 | 0.129 | 11369.7 | 12.12 |

## 解释

表交换带来小幅正向收益：发现率和拓扑代数连通度提升，空扫率下降。这说明邻居信息表和感知表确实能把局部发现结果扩散为其他节点的候选波束先验。

但碰撞数也同步上升，尤其 N=50 时更明显。这说明表交换会让多个节点更集中地选择相似的高置信波束，如果没有冲突控制和多 RF 激活门控，收益会被接入冲突抵消。因此，表交换应作为下一版 MARL 状态与动作设计的一部分，而不能只作为规则增强。
