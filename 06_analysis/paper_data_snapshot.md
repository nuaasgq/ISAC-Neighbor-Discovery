# 论文数据快照：最终 round1

日期：2026-07-04

## 核心判据

本轮数据覆盖用户要求的五类对比：参考文献近似、完全随机、强化学习但无 ISAC、改进强化学习但无 ISAC、改进强化学习加 ISAC。四类动态运动模型均不是静止拓扑，分别为 Gauss-Markov、random walk、random direction、random waypoint。

## 训练收敛证据

CEM 训练从 generation 0 的 best score 119.64 提升到 generation 8 的 123.90，generation 9 保持 123.69。对应发现率从 0.9417 提高到 0.9612，P95 时延从 266.28 slots 降到 177.48 slots。该趋势说明共享策略参数已进入稳定高分区域。

最优共享参数：`alpha_occupancy=1.9833`，`softmax_beta=3.2337`，`exploration_floor=0.0567`，`confidence_decay=0.9937`，`piggyback_sensing_period_multiplier=0.5`。

## 跨场景表现

| 场景 | `improved_rl_isac` 发现率 | 相对随机发现率增益 | 相对随机平均时延降低 | 相对随机空扫率降低 |
|---|---:|---:|---:|---:|
| Gauss-Markov 20ep | 0.9558 | +12.07 pp | 58.51% | 60.94% |
| Random Walk 20ep | 0.9569 | +4.92 pp | 27.03% | 10.17% |
| Random Direction 10ep | 0.9599 | +11.30 pp | 52.12% | 62.26% |
| Random Waypoint 10ep | 0.9844 | +9.78 pp | 59.85% | 77.59% |

## 论文图与表

- 论文图目录：`06_analysis/paper_figures/final_round1/`
- 图像数量：54 张 PNG，全部 1920x1440，4:3，300 dpi。
- 字体：Times New Roman，校验路径 `C:\Windows\Fonts\times.ttf`。
- 表格目录：`06_analysis/paper_tables/final_round1/`。
- 原始大文件目录：`05_simulation/results_raw/`，该目录按仓库规则不提交，仅保留本机 raw 结果。

## 论文写作可用结论

1. ISAC beam prior 的主要收益来自降低空扫率，直接压缩三维窄波束搜索空间。
2. 无 ISAC 的 RL 或 memory/topology 策略能带来一定收益，但在发现率和长尾时延上弱于 `improved_rl_isac`。
3. `improved_rl_isac` 在四个运动模型下均保持稳定领先，支持“动态无人机集群中的分布式链路层邻居发现协议”定位。
4. 当前结果足以支撑方法验证型论文初稿的数据段落；若面向 TWC/TCOM 终稿，后续应继续把在线代理替换为真正 MARL 训练器，并增加规模迁移、ISAC 误检/漏检敏感性和 beam 数扩展实验。
