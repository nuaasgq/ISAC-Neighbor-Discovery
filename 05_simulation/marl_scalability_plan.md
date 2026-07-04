# MARL 小规模训练到大规模迁移实验计划 v0

## 1. 目标

验证 `MARL-I-TAP-ND` 是否能在小规模 UAV 集群中训练，并在不微调的情况下直接部署到更大规模集群。

最低成功标准：

```text
N_train = {10, 20}
N_test  = {30, 50, 100}
```

在 `N_test=100` 时，MARL 相比最佳启发式基线仍保持正收益，并且每节点推理复杂度近似为 `O(M + K)`。

## 2. 规模泛化原则

| 原则 | 设计 |
|---|---|
| 参数共享 | 所有 UAV 使用同一个 actor `pi_theta(o_i)` |
| 分布式执行 | actor 只使用本地 observation |
| 固定维输入 | top-K beam / neighbor pooling / history encoder |
| 排列不变聚合 | mean/sum/attention/GNN pooling |
| Mean-field | 大规模场景用局部密度、模式比例、碰撞率近似群体影响 |
| Domain randomization | 随机化 `N`、密度、速度、ISAC 误差、beam 数 |

## 3. 训练和测试规模

| 阶段 | 节点数 | 目的 |
|---|---:|---|
| Train-S | `{10}` | 验证能学习 ISAC 排空和握手机制 |
| Train-M | `N in [8, 20]` | 主训练设置 |
| Validation | `{10, 20, 30}` | 观察早期规模退化 |
| Zero-shot Test | `{30, 50, 100, 200}` | 大规模直接迁移 |
| Stress Test | `{300, 500}` | 复杂度趋势测试，不作为主结论 |

两种扩展方式：

- 固定密度扩展：区域体积随 `N` 增大，主迁移实验。
- 固定区域扩展：区域不变、密度上升，拥塞压力测试。

## 4. 固定维 Observation

```text
o_i = [
  self_pose,
  self_motion,
  degree_state,
  beam_cell_features,
  pooled_neighbor_embedding,
  pooled_candidate_embedding,
  local_history_embedding
]
```

其中：

- `beam_cell_features` 不随节点数增长。
- 已发现邻居只取 top-K 或用 pooling。
- 候选目标不逐目标输入，而是聚合到 beam-cell occupancy prior。
- 位置输入使用归一化位置、局部方向、距离边界等，避免记忆固定区域。

## 5. Curriculum

| 阶段 | 设置 | 目标 |
|---|---|---|
| C0 | `N=5-10`，静态，低噪声，无碰撞 | 学会 beam prior 利用 |
| C1 | `N=10`，虚警/漏检/角度误差 | 学会保留探索 |
| C2 | `N=10-20`，移动、碰撞、握手失败 | 学会 Tx/Rx/Sense 协调 |
| C3 | `N in [8,20]`，随机密度/速度/误差/beam 宽度 | 训练规模泛化 |
| C4 | 多径误导、非通信目标、slot offset | 鲁棒性 |
| C5 | 冻结模型，测试 `N={30,50,100,200}` | 零微调迁移 |

## 6. 指标

| 类别 | 指标 |
|---|---|
| 发现效率 | 平均发现时延、P95/P99、发现率、空扫比例 |
| 拓扑质量 | LCC ratio、连通分量数、平均度、孤立节点比例、lambda2 proxy |
| 一致性 | average consensus 收敛迭代数或有限时间误差 |
| 迁移性 | Zero-shot Gain、Scale Retention |
| 复杂度 | 每 slot 推理时间、每节点计算量、内存 |
| 公平性 | 最差 10% 节点发现率、最小度、孤立节点比例 |

定义：

```text
Zero-shot Gain = Metric_RL(N_test) - Metric_best_baseline(N_test)
Scale Retention = Gain_RL(N_test) / Gain_RL(N_train_like_density)
```

## 7. 失败判据

| 失败类型 | 判据 |
|---|---|
| 性能迁移失败 | `N_test >= 50` 时 RL 不优于最佳启发式基线，或收益低于 5% |
| 收益保持失败 | `Scale Retention < 0.5` |
| 拓扑目标失败 | 相比 ISAC-only heuristic，LCC/孤立节点/consensus 无稳定改善 |
| 复杂度失败 | actor 输入或推理复杂度必须随全网 `N` 增长 |
| 拥塞崩溃 | `N >= 100` 时发现率低于随机/规则扫描 |
| 误感知脆弱 | `P_MD >= 0.3` 或 `sigma_cell >= 1` 时全面劣于启发式 |
| 需要大规模微调 | 大规模必须重新训练才能超过基线 |

## 8. 大规模仿真简化

- 用 KD-tree 或空间网格找通信半径内候选，避免全量 `O(N^2)`。
- beam-cell occupancy 只对局部候选节点更新。
- 碰撞判定按 `(receiver, beam-cell, slot)` 聚合。
- `lambda2` 只在 checkpoint 计算，大图用 sparse eigensolver 或代理。
- 大规模日志只保存聚合指标和关键事件。
