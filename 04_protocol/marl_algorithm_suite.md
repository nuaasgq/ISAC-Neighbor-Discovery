# MARL 算法族候选池 v0

## 定位

`MARL-I-TAP-ND` 不应过早绑定到某一个具体 MARL 算法。更稳妥的路线是把 ISAC 辅助邻居发现抽象为统一 Dec-POMDP / CTDE 环境，在同一 observation、action、reward、baseline 和训练预算下比较多类 MARL 方法。

算法选择分两层：

1. **通用 MARL 优化器**：解决多智能体信用分配、非平稳、部分可观测和离散动作学习。
2. **问题定制网络结构**：利用 ISAC beam-cell prior、拓扑代理、邻居集合和历史失败模式。

最终论文主方法应是二者组合，而不是简单声称“使用某某 MARL 算法”。

## 1. 统一学习问题

每个 UAV 的执行期策略为：

```text
a_i(t) = {mode_i(t), beam_i(t)}
a_i(t) ~ pi_theta(a | o_i(t))
```

其中：

- `mode_i(t) in {Sense, Tx, Rx, Idle}`。
- `beam_i(t) in B_i`。
- actor 只使用本地 `o_i(t)`。
- critic / mixer / replay buffer 训练期可使用仿真全局状态，但执行期不可使用。

推荐使用分层动作：

```text
pi(a_i | o_i) = pi_mode(mode_i | o_i) * pi_beam(beam_i | mode_i, o_i)
```

这样避免 `4 * M` 的扁平大动作直接带来训练不稳定。

## 2. 值函数与值分解类

| 算法族 | 代表方法 | 适合点 | 主要风险 | 本项目用法 |
|---|---|---|---|---|
| Independent Q | IQL, DRQN, R2D2-like recurrent Q | 实现简单，可作为低成本 sanity baseline | 多智能体非平稳严重，难学协同 Tx/Rx | 先验证 beam prior 是否能被 Q 网络利用 |
| Additive factorization | VDN | 轻量、稳定、易迁移 | 加性假设过强，难表达拓扑奖励和碰撞耦合 | Phase 1 baseline |
| Monotonic factorization | QMIX | 经典 CTDE 值分解，适合离散动作 | 单调性限制，可能无法表达“局部牺牲换全局拓扑收益” | Phase 1 主值函数 baseline |
| Weighted monotonic factorization | WQMIX | 缓解 QMIX 投影偏差，重视更优联合动作 | 仍保留单调框架，需调权重策略 | QMIX 失效时的增强 baseline |
| Transform factorization | QTRAN / QTRAN++ | 表达能力强于 VDN/QMIX | 训练复杂，经验稳定性不一定好 | Phase 2 风险候选 |
| Duplex dueling | QPLEX | 更完整满足 IGM，表达能力强 | 实现复杂度高，对调参敏感 | Phase 2 值函数强基线 |
| Attention mixer | Qatten | 可学习 agent 重要性和局部贡献 | critic/mixer 可能随 `N` 增长，需要稀疏或 pooling | 适合拓扑贡献和拥塞贡献建模 |
| Transformer mixer | TransfQMix | 用 transformer 利用潜在图结构，强调可迁移 | 必须避免全网 token 直接导致规模爆炸 | Phase 3 结构候选 |
| Risk-sensitive factorization | RiskQ | 建模 VaR/CVaR/quantile 风险，适合漏检和长尾时延 | 实现成本高，风险指标解释要谨慎 | Phase 3 风险感知候选 |
| Action dependency | ACE | 用动作依赖缓解非平稳，适合强耦合决策 | 执行动作顺序设计可能引入中心化嫌疑 | 只在训练或顺序化近似中评估 |
| Committed exploration | MAVEN | 稀疏奖励和长时程协同探索更强 | 需要 latent exploration 设计，结果解释更难 | 用于发现奖励稀疏场景 |

### 与本问题的结合

值函数类适合 `mode + beam` 都离散的设定。需要重点处理：

- beam 动作空间很大时，使用 top-K candidate beam 或 beam scorer 先筛选。
- team reward 中的 topology 增益需要 credit assignment，QMIX/VDN 可能表达不足。
- QPLEX/Qatten/ACE 可作为 Phase 2 强候选，但不宜第一版就重实现所有复杂算法。

## 3. 策略梯度类

| 算法族 | 代表方法 | 适合点 | 主要风险 | 本项目用法 |
|---|---|---|---|---|
| Independent PG | IPPO | 参数共享、分布式执行、实现成本低 | 非平稳和信用分配弱 | Phase 1 轻量 PG baseline |
| Centralized critic PPO | MAPPO | 稳定、工程成熟、适合离散动作 | 不一定最优，容易被写成套算法 | Phase 1/2 强 baseline，不预设主方法 |
| Trust-region sequential update | HAPPO / HATRPO | 有单调改进理论，适合异构/顺序更新 | 训练开销较大，实现复杂 | Phase 2 强 PG 候选 |
| Sequence modeling | MAT | 把多智能体决策转成序列建模，可支持 agent 数变化 | 训练和推理顺序设计需要谨慎，执行期不可依赖全局顺序信息 | Phase 2/3 结构候选 |
| K-level / hierarchical PG | K-level PG, hierarchical PPO variants | 可自然对应 `mode -> beam` 两级动作 | 需要重新定义 advantage 和 credit assignment | 作为分层动作结构参考 |

### 与本问题的结合

策略梯度类适合概率搜索协议，因为自然输出 beam 分布：

```text
pi_beam(m | o_i) = epsilon / M + (1 - epsilon) * softmax(logit_i^m)
```

这与 I-TAP-ND 的探索下界一致。需要避免的问题是：

- actor 不能看到未发现邻居真值。
- centralized critic 不应成为执行期依赖。
- 若使用 MAT，需要把 agent sequence 视为训练期 credit assignment 工具，而不是中心调度。
- 对 `mode` 和 `beam` 应分别设置 entropy bonus；beam entropy 可随 ISAC 不确定性自适应变化。

## 4. Actor-Critic 与最大熵类

| 算法族 | 代表方法 | 适合点 | 主要风险 | 本项目用法 |
|---|---|---|---|---|
| Counterfactual AC | COMA | 对离散动作和信用分配友好 | 大动作空间下 counterfactual 计算重 | 小规模动作裁剪后评估 |
| Attention critic | MAAC | critic 只关注相关 agent，适合稀疏邻居关系 | 原始 MAAC 对连续/通用任务更多，需要适配离散 beam | 可与 discovered graph / candidate beam attention 结合 |
| Deterministic AC | MADDPG / MATD3 | 连续功率、波宽、轨迹控制可用 | 当前主动作是离散 mode/beam，不是首选 | 后续加入功率/波宽时再考虑 |
| Soft AC | MASAC / discrete SAC | 最大熵探索适合漏检、稀疏奖励和长尾发现 | 离散 SAC 稳定性需验证，熵温度敏感 | Phase 2 探索强化候选 |
| Hybrid SAC | MAHSAC / hybrid action SAC | 适合后续同时优化离散 mode/beam 和连续功率/波宽 | 第一阶段实现成本偏高 | 作为混合动作扩展路线 |

## 5. 分层动作训练约束

`mode + beam` 不是普通单头离散动作。推荐所有算法尽量共享同一动作接口：

```text
h_i = encoder(o_i)
mode_i ~ pi_mode(. | h_i)
beam_i ~ pi_beam(. | h_i, mode_i)
```

训练细节：

- `Idle` 不需要 beam loss。
- `Sense/Tx/Rx` 使用 conditional beam head。
- 新链路奖励同时分配给 Tx 与 Rx 两端。
- 延迟握手奖励需要 RNN memory 或 eligibility trace。
- ISAC prior 只作为 soft logit bias，不做硬 mask。
- flat `mode x beam` 只作为小 beam 数 sanity baseline。

## 6. Critic / Mixer 信息边界

| 模块 | 允许 | 禁止或不建议 |
|---|---|---|
| Actor 输入 | 自身状态、ISAC prior、本地历史、已发现邻居摘要 | 未发现邻居真实位置、全局拓扑、中心调度结果 |
| Actor 结构 | 参数共享、top-K beam、beam pooling、set pooling、GRU | 输入维度随全网 `N` 增长 |
| Critic 输入 | 仿真全局状态、发现图、联合动作、局部子图、全局统计 | 直接拼接固定 `N` 的全体节点状态作为唯一表达 |
| Critic 尺度策略 | graph pooling、attention pooling、mean-field summary、K-hop local critic | 训练期 critic 在大规模 `N` 下结构失效 |
| 执行复杂度 | 每节点 `O(M)` 或 `O(K)` | 执行期需要全网通信或全局图 |

## 7. 推荐筛选顺序

| 阶段 | 算法 | 目的 |
|---|---|---|
| S0 | Rule expert I-TAP-ND | 生成专家轨迹和规则基线 |
| S1 | IQL/DRQN, VDN, QMIX, IPPO, MAPPO | 找到可训练底线 |
| S2 | WQMIX, QPLEX, Qatten, HAPPO, MAAC, MASAC | 比较信用分配和探索能力 |
| S3 | MAT, TransfQMix, RiskQ, ACE, MAVEN | 验证序列决策、风险敏感、动作依赖和长程探索是否有额外收益 |
| S4 | 加入问题定制结构 | 形成最终方法创新 |

## 8. 方法创新落点

算法本身不应作为唯一创新点。建议最终方法命名围绕结构与机制：

```text
ISAC-Prior-Guided Scalable MARL Neighbor Discovery
```

候选技术组合：

1. **Rule-residual MARL**：规则 I-TAP-ND 给出 prior logits，神经网络学习 residual。
2. **Beam-cell attention critic**：critic 在训练期对 beam-cell、候选方向和已发现邻居贡献做注意力分解。
3. **Scale-invariant neighbor pooling**：已发现邻居集合和局部拥塞用固定维 pooling 表示，支持大规模迁移。
4. **Uncertainty-aware entropy control**：根据 ISAC prior 不确定性调节探索熵，防止漏检造成永久剪枝。
5. **PABR-Net**：姿态感知 beam token、per-beam action memory 和 risk head 共同建模三维 ISAC 波位信念。
6. **ScaleGraph-Beam MARL**：top-K beam attention、已发现邻居 pooling 和 mean-field congestion summary 支撑小到大规模迁移。
