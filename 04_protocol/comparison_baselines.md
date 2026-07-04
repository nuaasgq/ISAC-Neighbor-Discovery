# 论文实验对比矩阵 v0

## 目标

论文数据必须覆盖以下五类对比，否则实验链条不完整：

1. 参考文献基线。
2. 完全随机盲扫。
3. 强化学习但没有 ISAC。
4. 改进强化学习但没有 ISAC。
5. 改进强化学习 + ISAC。

当前阶段已经提供 `rl_no_isac`、`improved_rl_no_isac` 和 `improved_rl_isac` 三个可运行的在线策略代理基线，用于完成五类对比闭环；后续深度 MARL 训练器接入后，应替换或补充这些代理策略。

## 参考文献基线

首选参考基线：

```text
SkyOrbs-like 3D directional neighbor discovery
```

来源：

- Zhu et al., "SkyOrbs: A Fast 3-D Directional Neighbor Discovery Algorithm for UAV Networks", IEEE Transactions on Mobile Computing, 2024.

选择理由：

- 题面直接覆盖 UAV networks。
- 目标是 3D directional neighbor discovery。
- 相比传统二维 CRT / quorum / random DND，更接近本文的三维窄波束邻居发现问题。
- 它不依赖 ISAC prior，适合作为强非感知基线。

当前实现状态：

- `skyorbs_like_skip_scan` 是工程近似版，不声明完全复现原文全部细节。
- 近似思想是三维有序 skip-scan / coarse-to-fine 风格的 beam-cell 遍历，避免完全均匀盲扫。
- 论文终稿前应继续核对 SkyOrbs 原文扫描序列、天线切换约束和同步/异步假设。

## 五类方法定义

| 类别 | 实验名称 | ISAC prior | RL | 改进结构 | 作用 |
|---|---|---:|---:|---:|---|
| 参考文献基线 | `skyorbs_like_skip_scan` | 否 | 否 | 否 | 对比 3D UAV DND 近邻工作 |
| 完全随机 | `uniform_random` | 否 | 否 | 否 | 下界与随机盲扫基线 |
| Learned policy no ISAC | `rl_no_isac` | 否 | 是 | 否 | 验证仅靠学习能否解决高维盲扫 |
| Enhanced learned policy | `improved_rl_no_isac` | 否 | 是 | 是 | 验证分层动作、记忆、拓扑奖励等方法创新的独立价值 |
| Enhanced learned policy + ISAC | `improved_rl_isac` | 是 | 是 | 是 | 验证 ISAC prior 与方法创新叠加收益 |

## 公平信息边界

| 方法 | 可用信息 | 不可用信息 |
|---|---|---|
| `skyorbs_like_skip_scan` | 自身状态、本地 beam codebook、固定扫描序列 | ISAC prior、未发现邻居真值、全局拓扑 |
| `uniform_random` | 自身状态、本地 beam codebook、随机 mode/beam | ISAC prior、历史学习策略、未发现邻居真值 |
| `rl_no_isac` | 自身状态、beam 选择历史、成功/失败/碰撞统计、已发现邻居摘要 | ISAC prior、未发现邻居真值、全局拓扑 |
| `improved_rl_no_isac` | 在 `rl_no_isac` 基础上增加分层动作、memory、topology-aware reward 或 local pooling | ISAC prior、未发现邻居真值、全局拓扑 |
| `improved_rl_isac` | `improved_rl_no_isac` + 通信伴随 ISAC beam-cell occupancy prior / uncertainty / staleness | 未发现邻居 ID、位置、波束状态、全局拓扑 |

## 最小成功判据

论文级主结果至少应满足：

1. `skyorbs_like_skip_scan` 优于或接近完全随机，说明参考基线不是弱基线。
2. `rl_no_isac` 相比完全随机有提升，但在窄波束高维空间下提升有限，支撑“仅靠 RL 不够”。
3. `improved_rl_no_isac` 优于 `rl_no_isac`，支撑方法结构创新。
4. `improved_rl_isac` 优于 `improved_rl_no_isac`，支撑 ISAC 机制创新。
5. `improved_rl_isac` 在发现率、P95/P99 时延、空扫率和拓扑质量中至少三类指标上优于参考文献基线。

## 当前规则协议消融链

在 MARL 尚未进入训练前，规则协议阶段先完成：

| 方法 | ISAC prior | 拓扑/历史项 | 目的 |
|---|---:|---:|---|
| `uniform_random` | 否 | 否 | 随机盲扫下界 |
| `deterministic_scan` | 否 | 否 | 固定序列确定性扫描 |
| `skyorbs_like_skip_scan` | 否 | 否 | 3D DND 近邻强基线近似 |
| `isac_only` | 是 | 否 | 单独证明 ISAC 空波位排除收益 |
| `topology_only` | 否 | 是 | 单独证明拓扑/历史规则收益 |
| `itap_nd` | 是 | 是 | 证明 ISAC + 拓扑/历史双驱动收益 |
| `oracle` | 真值 occupied beam | 否 | 上界，不参与公平协议比较 |

## 后续实现任务

| 模块 | 任务 |
|---|---|
| `skyorbs_like_skip_scan` | 继续细化三维 skip-scan 和 coarse-to-fine 规则 |
| `rl_no_isac` | 实现无 ISAC observation 的 MARL 环境和轻量 RL baseline |
| `improved_rl_no_isac` | 加入分层动作、GRU/per-beam memory、topology reward |
| `improved_rl_isac` | 加入通信伴随 ISAC beam prior、uncertainty token、rule residual |
| `analysis` | 输出五类方法的统一表格和图 |
