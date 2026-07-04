# Sprint 3 任务拆解：MARL 与小到大规模迁移

## 目标

在已有规则协议 `I-TAP-ND` 的基础上，设计并实现 `MARL-I-TAP-ND`，验证小规模 UAV 集群训练得到的策略能否零微调迁移到更大规模集群。

## 新增研究设定

- UAV 可知道自身位置、速度、航向和姿态。
- UAV 可将 body-frame beam cell 映射到当前 global-frame 方向锥。
- 对准前仍未知未发现邻居身份、实时位置、波束状态和全局拓扑。
- ISAC 提供候选 beam-cell occupancy prior，而不是邻居确认。

## 任务线

| 任务线 | 内容 | 可并行 | 产物 |
|---|---|---:|---|
| A | POMDP/MARL 建模 | 是 | `04_protocol/marl_design.md` |
| B | 规模迁移实验设计 | 是 | `05_simulation/marl_scalability_plan.md` |
| C | 规则先验融合 | 是 | 更新 `04_protocol/protocol_design.md` |
| D | 仿真器接口设计 | 是 | 更新 `05_simulation/run_experiments.md` |
| E | 文献风险扫描 | 是 | 更新 `02_literature/gap_analysis.md` |
| F | 最小代码实现 | 否，依赖 A-D | `05_simulation/src/` |

## 推荐执行顺序

1. 更新系统模型和信息边界。
2. 设计 MARL observation/action/reward。
3. 设计参数共享和固定维输入，确保 actor 不依赖节点总数。
4. 实现规则仿真器和 I-TAP-ND。
5. 实现 MARL 训练接口。
6. 先跑 `N=10/20` 小规模训练，再零微调测试 `N=30/50/100`。

## 进入代码实现前的检查点

1. Actor 输入是否包含未发现邻居真值？
2. Critic 是否只在训练期使用全局信息？
3. 观测维度是否随 `N` 增长？
4. 大规模测试是否零微调？
5. MARL 是否至少包含 no-ISAC、no-topology、no-rule-prior 消融？
