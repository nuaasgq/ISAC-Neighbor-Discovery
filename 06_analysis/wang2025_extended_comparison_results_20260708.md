# Wang2025 扩展对比结果记录

## 目的

本轮实验按王为栋等论文的 FANET 快速邻居发现思路扩展对比，在同一仿真环境中加入 MIMO-OTFS/ISAC 感知能力抽象、邻居信息表和邻居感知表机制，用于判断 Wang 风格机制、纯随机机制和当前 ISAC 引导方法在相同矩阵下的差异。

## 实验矩阵

- 节点数：10, 20, 30, 40, 50。
- RF 链数：1, 3, 6。王文表格中覆盖更完整的 RF 链数，这里先选低/中/高三个代表点做扩展对比。
- 时隙长度：5 ms。
- 每幕长度：200 slots。
- 重复次数：每个组合 5 episodes。
- 波束划分：15 个方位角单元、7 个俯仰角单元，对应约 24 deg 方位窄波束和 25.7 deg 俯仰波束。
- 感知抽象：30 GHz 载频、64 MHz 带宽、1 W 发射功率、1 m2 RCS，使用 radar-SNR 检测概率近似；这是 PHY-aware 抽象，不是完整 OTFS 波形接收机。
- 单跳假设：通信距离和感知距离足够覆盖区域对角线，用于先隔离邻居发现协议差异。

## 对比协议

- `uniform_random`：完全随机盲扫。
- `wang2025_isac_no_collab`：王文风格 ISAC 空波位排除，但不交换表。
- `wang2025_comm_tables`：王文风格通信邻居表交换。
- `wang2025_isac_tables`：王文风格通信邻居表 + 感知表交换。
- `improved_rl_isac`：当前仓库已有的拓扑感知 ISAC 规则基线。该方法不是最终 MARL 版本，当前只作为我方现有机制的中间对比项。

## 结果位置

- 聚合数据：`06_analysis/paper_tables/wang2025_extended_comparison_20260708/aggregate_metrics.csv`
- 每幕数据：`06_analysis/paper_tables/wang2025_extended_comparison_20260708/per_episode_summary.csv`
- 每时隙数据：`06_analysis/paper_tables/wang2025_extended_comparison_20260708/per_slot_metrics.csv`
- 完成时隙数据：`06_analysis/paper_tables/wang2025_extended_comparison_20260708/completion_slots.csv`
- 结果图：`06_analysis/paper_figures/wang2025_extended_comparison_20260708/`

## 关键发现

RF=3、N 从 10 到 50 时，完全随机的目标发现率只有 0.067 到 0.082 左右；所有 ISAC/表交换方法都显著优于随机盲扫，说明“感知先排空波位，再引导窄波束邻居发现”的方向成立。

在 Wang 风格矩阵中，表交换基线非常强。RF=3 时，`wang2025_comm_tables` 的发现率从 N=10 的 0.978 下降到 N=50 的 0.468；`wang2025_isac_tables` 从 1.000 下降到 0.467；当前 `improved_rl_isac` 从 0.920 下降到 0.400。也就是说，当前我方中间方法明显强于随机，但在 Wang 自身机制环境中还不能压过 Wang 风格表交换基线。

多 RF 链不是单调收益。N=50 时，完全随机发现率随 RF=1/3/6 从 0.0075、0.0738 提升到 0.2558；但 Wang 风格和当前 ISAC 方法在 RF 增加后下降或基本持平，例如 `wang2025_comm_tables` 为 0.538、0.468、0.474，`improved_rl_isac` 为 0.446、0.400、0.401。原因是多 RF 同时激活增加了冲突和重复扫描，若没有负载控制、退避和表信息置信度调度，额外 RF 链无法自动转化为更高发现率。

当前 `improved_rl_isac` 的空扫率显著低于 Wang 风格方法。RF=3、N=50 时，`improved_rl_isac` 空扫率为 0.138，而 `wang2025_comm_tables`、`wang2025_isac_tables` 分别为 0.297 和 0.297。这说明现有 ISAC 信念引导能够减少空波位扫描，但还没有把节省出来的扫描机会转化为更高握手成功率，瓶颈转移到多节点冲突、Tx/Rx 配对和表信息协同。

## 对论文创新点的含义

这轮结果不能支撑“当前 improved_rl_isac 已经优于 Wang 方法”的论文主张，但它支撑两个更清晰的创新方向：第一，ISAC 物理层感知能力用于快速筛除空波位是有效的；第二，仅靠单节点信念引导不够，必须把邻居表、感知表、多 RF 链负载控制、冲突抑制和 MARL 策略学习结合起来，形成真正的数据-规则双驱动跨层协议。

下一轮应把 Wang 的邻居信息表和邻居感知表机制纳入我方 MARL 环境状态，设计 collision-aware multi-RF action gating：智能体不仅选择波束，还要决定 RF 链激活数、Tx/Rx 角色、表信息可信度权重和候选波束调度。目标指标应从单纯发现率扩展为发现率、单位扫描发现数、碰撞惩罚发现率、拓扑代数连通度和完成时延联合优化。

## 版本边界

本轮代码已经具备 Wang 风格协议矩阵复现实验能力，但仍是协议层 PHY-aware 抽象。若面向 TWC/TCOM，后续需要补强两个层面：一是把 MIMO-OTFS 感知模型写成可解释的检测概率/虚警/漏检模型，并和波形参数建立显式关系；二是把 MARL 网络结构创新落到可训练、可迁移、可消融的实现上，而不是继续使用启发式规则冒充 MARL。
