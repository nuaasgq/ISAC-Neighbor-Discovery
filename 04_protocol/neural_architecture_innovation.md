# 面向 ISAC 邻居发现的 MARL 网络结构创新 v0

## 目标

本文件定义区别于“套用现成 MARL”的网络结构创新路线。网络结构必须服务于本问题的特殊输入和目标：

- ISAC beam-cell occupancy prior。
- 自身位置、速度、航向和姿态。
- 本地 beam cell 到 global direction cone 的映射。
- 已发现邻居集合和局部拓扑摘要。
- 空扫、碰撞、漏检、多径误导等历史反馈。
- 小规模训练到大规模零微调迁移。

## 1. 总体结构

建议基础 actor 采用模块化结构：

```text
self_encoder
beam_prior_encoder
neighbor_set_encoder
history_encoder
uncertainty_encoder
rule_residual_fusion
mode_head
beam_head
```

训练期 critic / mixer 可以更强，但执行期 actor 必须只读本地观测。

## 2. Beam-cell Encoder

| 结构 | 输入 | 优点 | 风险 | 适用阶段 |
|---|---|---|---|---|
| Flatten MLP | 所有 beam-cell feature | 实现最简单 | 泛化弱，beam 数变化困难 | MVP |
| Shared beam scorer | 对每个 beam cell 用共享 MLP 打分 | 参数不随 `M` 增长过快，易解释 | 缺少邻近 beam 空间结构 | Phase 1 |
| 2D angular CNN | `azimuth x elevation` prior grid | 利用角度邻域和误差扩散 | codebook 变化时需插值 | Phase 2 |
| Beam attention | top-K beams 作为 token | 适合大 `M`，可解释 attention | top-K 可能漏掉低 prior 真邻居 | Phase 2 |
| Pose-aware spherical encoder | beam direction cone + `R_i(t)` + self pose | 更贴合三维波束几何，保留 azimuth wrap-around | 实现复杂，需要处理不同 beam codebook | Phase 3 |
| Rule-guided top-K beam token encoder | ISAC prior + uncertainty + topology proxy 选 top-K | 降低动作和注意力规模 | top-K 不能硬排除低 prior beam | Phase 2 |

推荐初始方案：

```text
beam_token_m = MLP([p_m, u_m, staleness_m, rhat_m, vhat_m,
                    recent_hit_m, recent_miss_m, collision_m,
                    topology_proxy_m, d_body_m, R_i d_body_m,
                    self_pose_i, action_history_m])
```

对所有 beam token 共享编码器，再输出 beam logits。

## 3. Neighbor Set / Graph Encoder

已发现邻居数随网络规模变化，不能直接拼接固定全局邻接矩阵。

| 结构 | 输入 | 迁移性 | 用法 |
|---|---|---:|---|
| Mean / max pooling | 已发现邻居记录 | 高 | Phase 1 baseline |
| Attention pooling | 已发现邻居记录 | 高 | 学习关键邻居和桥接链路 |
| Local GNN | 已发现一跳/二跳图 | 中-高 | 训练期和执行期均可，只用已发现图 |
| Hierarchical graph attention | agent-level + group/region-level summary | 中-高 | 可表达局部簇和桥接方向 |
| K-hop critic neighborhood | 训练期 critic 只读最近 K 个或 K-hop 相关节点 | 高 | 避免 centralized critic 输入随全网 `N` 增长 |
| Mean-field summary | 局部模式比例、碰撞率、密度 | 高 | 大规模压力测试 |

邻居记录可包括：

```text
neighbor_record_ij = [
  link_quality_ij,
  last_seen_ij,
  beam_to_j,
  relative_position_after_handshake,
  neighbor_degree_summary,
  two_hop_summary_optional
]
```

## 4. Temporal Memory

邻居发现是部分可观测过程，需要记住历史失败、碰撞和感知 stale 信息。

| 结构 | 作用 | 风险 |
|---|---|---|
| GRU | 低成本记忆局部历史 | 长程依赖有限 |
| Transformer memory | 处理长期探索和多阶段握手 | 成本高，易过拟合小规模 |
| Event memory | 只记录 beam-level hit/miss/collision counters | 可解释、稳定 | 表达能力弱 |
| Per-beam action-conditioned GRU | 每个 beam cell 记忆 Sense/Tx/Rx 结果 | 能区分空扫、无 ACK、碰撞、虚警、漏检 | 状态量随 beam 数增长 |

推荐 MVP：

```text
history_embedding = GRU([last_action, last_mode, selected_beam,
                         discovery_success, empty_scan, collision])
```

Phase 3 可加入 per-beam memory：

```text
h_i^m(t) = GRU_m([mode_i(t), beam_i^m, hit_i^m, miss_i^m,
                  collision_i^m, ack_fail_i^m, sensing_update_i^m])
```

## 5. Uncertainty-aware Module

ISAC prior 不能硬剪枝，因此需要显式建模不确定性。

候选设计：

| 模块 | 公式/机制 | 目的 |
|---|---|---|
| Exploration floor | `pi = eps/M + (1-eps) softmax(logits)` | 防止漏检永久不可发现 |
| Entropy scheduler | `alpha_i(t) = g(mean(u_i), H(p_i), P_MD_est)` | 不确定时提高探索 |
| Dual entropy | `alpha_mode` 控制 Sense/Tx/Rx/Idle，`alpha_beam` 控制 beam 搜索 | 防止 mode 学得过早确定，而 beam 仍需探索 |
| Risk value head | 输出 mean / quantile / CVaR-style beam value | 同时估计平均收益和长尾失败风险 |
| Beta belief head | 用 Beta 参数表示 occupancy uncertainty | 比单一 `p_m` 更适合虚警/漏检建模 |
| Risk penalty | 高置信多次失败后降低对应 beam residual | 抑制虚警 |
| Uncertainty token | 把 `u_m, staleness_m, sigma_cell` 作为 token feature | 让网络学习何时相信 ISAC |

## 6. Rule-Neural Residual Fusion

规则机制不应被神经网络覆盖，而应作为可解释骨架。

```text
score_m =
  beta * log(p_m + epsilon)
+ gamma * topology_proxy_m
- eta * staleness_m
+ clip(f_theta(beam_token_m, self_state, history), -delta, delta)
```

其中 `f_theta` 是 residual。这样论文可以解释为：

- 规则项提供机制可解释性。
- 神经项学习规则未覆盖的碰撞、时序和拓扑耦合。
- clipped residual 限制黑箱策略完全覆盖规则先验，便于稳定训练和消融解释。

## 7. 可形成论文方法创新的组合

### 方案 A：PABR-Net

全称：Pose-aware Action-memory Beam Risk Network。

- 姿态感知 spherical beam encoder。
- per-beam action-conditioned memory。
- uncertainty / quantile / CVaR-style risk head。
- rule-neural residual fusion。

优势：最贴合 ISAC + 三维 UAV 窄波束问题，能把“自身姿态可用”和“邻居未知”同时编码进方法。

### 方案 B：Rule-Guided Residual MARL

- I-TAP-ND 规则分数作为 base logits。
- 神经网络只学习 clipped residual。
- 分层 `mode -> beam` head。
- 适合 IPPO、MAPPO、HAPPO、MASAC、Q-value branch 等多种骨干。

优势：机制创新和方法创新结合最清楚，适合第一篇论文主线。

### 方案 C：Topology-Aware Attention Critic

- Actor 仍本地执行。
- Critic 用 discovered graph / candidate beam attention 做 credit assignment。
- 可接 MAAC、COMA、MAPPO critic 或 Qatten mixer。
- Critic 不能固定拼接全网节点状态，应使用 K-hop local graph、attention pooling 或 mean-field summary。

优势：能把拓扑奖励的信用分配作为方法创新。

### 方案 D：ScaleGraph-Beam MARL

- Top-K beam attention。
- DeepSets / local-GNN neighbor encoder。
- mean-field congestion summary。
- shared actor。

优势：重点服务“小规模训练、大规模直接迁移”。

### 方案 E：Scale-Invariant Beam-Graph Transformer

- beam tokens + neighbor tokens + history token。
- token pooling 输出固定维 embedding。
- 可接 MAT 或普通 actor-critic。
- 大规模版本只允许 local tokens 或 top-K tokens，不能把全网 agent 当作完整序列输入。

优势：最有方法新意，但实现和调参风险最高，应作为 Phase 3。

### 方案 F：K-hop Local Critic with Mean-field Congestion

- Actor 完全本地执行。
- Critic 训练期读取局部 K-hop discovered graph、top-K candidate beam tokens 和局部 Tx/Rx/Sense 比例。
- 用 mean-field summary 近似远端节点的拥塞影响。
- 可接 MAAC、MAPPO critic、MASAC critic 或 MADDPG-K 类思想。

优势：直接服务“小规模训练、大规模迁移”，并能回应 centralized critic 随 `N` 增长的问题。

## 8. 必做消融

| 消融 | 目的 |
|---|---|
| no-rule-residual | 验证规则骨架是否必要 |
| no-ISAC-prior-token | 验证感知先验贡献 |
| no-uncertainty-token | 验证误感知鲁棒性 |
| no-neighbor-pooling | 验证拓扑/已发现邻居信息贡献 |
| no-pose-encoding | 验证 UAV 自身姿态/位置对 beam 表示的贡献 |
| flat-MLP vs beam-token encoder | 验证 beam 结构编码 |
| angular CNN / spherical encoder / beam attention | 比较 beam-cell 结构化编码 |
| no-temporal-memory vs node-GRU vs per-beam action memory | 验证历史动作语义建模 |
| mean-only prior vs uncertainty/risk-aware head | 验证风险头对虚警/漏检和长尾的作用 |
| fixed-K vs pooling | 验证规模迁移结构 |
| entropy fixed vs uncertainty-adaptive | 验证漏检/虚警鲁棒性 |
| full critic vs K-hop/local critic | 验证 critic 尺度约束对迁移的作用 |
| agent sequence transformer vs local token transformer | 验证 MAT 类结构是否真的支持大规模迁移 |
| train `24x8`, test `36x8/72x12` | 验证跨波束粒度迁移 |
