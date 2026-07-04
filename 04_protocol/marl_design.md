# MARL-I-TAP-ND 方法设计 v0

## 1. 定位

`MARL-I-TAP-ND` 是规则协议 `I-TAP-ND` 的学习增强版。规则协议负责 ISAC belief 更新、探索下界、链路层握手和信息边界；MARL 负责学习每个时隙的 mode selection 和 beam-cell selection。

推荐关系：

| 层级 | 方法 |
|---|---|
| 规则基线 | I-TAP-ND：ISAC prior + topology-aware probabilistic discovery |
| 学习增强 | MARL-I-TAP-ND：学习 mode/beam 调度 |
| 安全退化 | 当 MARL 置信度低或场景分布外，退回 I-TAP-ND |

## 2. Dec-POMDP / MARL 建模

每个 UAV 是一个 agent。执行期去中心化：

```text
a_i(t) ~ pi_theta(a | o_i(t))
```

训练期可采用 CTDE：

- Actor：只读本地观测。
- Critic：训练时可读仿真全局状态、真实发现图和所有动作。
- 执行期：critic 不参与。

## 3. Observation

```text
o_i(t) = {
  self_state_i(t),
  beam_prior_i(t),
  sensing_uncertainty_i(t),
  discovered_neighbor_summary_i(t),
  local_discovery_history_i(t),
  action_history_i(t)
}
```

### 3.1 Self State

- 自身位置 `x_i(t)`。
- 自身速度 `v_i(t)`。
- 航向/俯仰/横滚或 quaternion。
- 当前 beam-cell。
- 上一时隙动作。
- 剩余发现时间或 episode progress。

### 3.2 ISAC Prior

Beam-cell feature：

```text
f_i^m(t) = [
  p_i^m(t),
  u_i^m(t),
  tau_i^m(t),
  rhat_i^m(t),
  vhat_i^m(t),
  recent_hit_i^m(t),
  recent_miss_i^m(t),
  recent_collision_i^m(t),
  topology_proxy_i^m(t)
]
```

编码方式：

- MVP：flatten beam-cell grid + MLP。
- 可扩展版本：top-K candidate beams + shared beam encoder。
- 高级版本：beam-cell attention 或轻量 CNN。

### 3.3 Discovered Neighbor Summary

对已发现邻居集合使用排列不变编码：

```text
h_i^N = Pool({enc(neighbor_record_ij) | j in N_i^D})
```

可选 `neighbor_record_ij`：

- 最近通信时间。
- 指向该邻居的 beam-cell。
- 链路质量。
- 握手后交换的位置/速度。
- 邻居本地度数摘要。

## 4. Action

MVP 动作：

```text
a_i(t) = {mode_i(t), beam_i(t)}
```

其中：

- `mode_i(t) in {Sense, Tx, Rx, Idle}`
- `beam_i(t) in B_i`

分层策略：

```text
pi(a_i) = pi_mode(mode | o_i) * pi_beam(beam | mode, o_i)
```

后续扩展：

- 发射功率 `{low, mid, high}`。
- sensing budget `{1, top-3, top-5}`。
- 可变波宽 `{narrow, medium, wide}`。

## 5. Reward

基础形式：

```text
r_i(t) =
  w_disc  * R_i^disc(t)
- w_delay * R_i^delay(t)
- w_empty * R_i^empty(t)
- w_col   * R_i^collision(t)
- w_sense * R_i^sense_cost(t)
- w_power * R_i^power(t)
+ w_topo  * R_i^topo(t)
```

### 5.1 Discovery Reward

```text
R_i^disc(t) = 1{new edge i-j discovered} * W_ij(t)
```

`W_ij(t)` 可包含：

- 链路质量。
- 方向多样性。
- 度数缺口。
- 发现越早奖励越大。

### 5.2 Topology Reward

训练阶段可用发现图的增量指标：

```text
R^topo(t) =
  c1 * Delta |E_D|
+ c2 * Delta LCC
+ c3 * Delta lambda2_proxy
- c4 * num_components
```

执行期 actor 不直接看到这些全局指标。

### 5.3 Penalty

- 空 Tx。
- 空 Rx。
- 碰撞。
- 反复 sensing 低价值 beam。
- 高置信 beam 多次失败后仍反复选择。

## 6. 推荐架构

主推荐：

```text
PS-MAPPO-GAT
```

即 Parameter-Shared MAPPO with Beam Prior Encoder and Graph/Attention Aggregation。

模块：

```text
self_encoder
beam_prior_encoder
neighbor_set_encoder
history_encoder
fusion
mode_head
beam_head
```

优先级：

1. IPPO：最轻量 baseline。
2. MAPPO：主方法。
3. MAPPO + attention/GNN pooling：规模迁移主方法。
4. Mean-field MARL：超大规模扩展版本。

## 7. 规则先验融合

### 7.1 Policy Bias

```text
logit_i^m = f_theta(o_i)^m
          + beta * log(p_i^m + epsilon)
          + gamma * topology_score_i^m
          - eta * staleness_i^m
```

### 7.2 Soft Action Mask

```text
pi_i^m = epsilon / M + (1 - epsilon) * softmax(logit_i^m)
```

不对低 ISAC prior beam 做硬 mask，避免漏检造成永久不可发现。

### 7.3 Imitation Pretraining

用 I-TAP-ND 生成专家轨迹，先行为克隆：

```text
min_theta CE(pi_theta(a|o), a_I-TAP)
```

再用 MAPPO fine-tune。

## 8. 消融实验

- no-ISAC prior。
- no-topology proxy。
- no-rule bias。
- no-neighbor pooling。
- IPPO vs MAPPO。
- small-scale only vs randomized-N training。
- zero-shot transfer vs fine-tuned transfer。

## 9. 风险

| 风险 | 处理 |
|---|---|
| 奖励稀疏 | imitation pretraining、dense empty-scan penalty、curriculum |
| 多智能体非平稳 | CTDE、MAPPO clipping、参数共享 |
| 规模过拟合 | variable-N training、domain randomization、固定维 observation |
| 过度依赖 ISAC | 训练随机化 `P_FA/P_MD/sigma_cell`，保留探索 |
| 黑箱难解释 | 输出 action heatmap、logit 分解、规则先验消融 |
