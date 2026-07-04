# 问题形式化 v0

## 1. 目标

在完全分布式 UAV-UAV 三维窄波束网络中，设计节点本地策略：

```text
pi_i: local_state_i(t) -> a_i(t) = (m_i(t), b_i^m(t))
```

使节点在有限时间预算 `T` 内以较低空扫和发现时延发现更多关键邻居链路，并提升已发现拓扑质量。

## 2. 本地状态

节点 `i` 在时隙 `t` 的本地状态为：

```text
local_state_i(t) = {
  O_i(t),
  N_i^D(t),
  d_i^D(t),
  discovery_history_i(t),
  failed_attempt_history_i(t)
}
```

其中不包含真实未发现邻居位置、真实全局拓扑或中心控制信息。

`O_i(t)` 是 ISAC 观测诱导的 beam-cell occupancy prior，不包含邻居 ID 或真实位置。

## 3. 多目标评价

主要评价目标包括：

```text
minimize   E[T_discovery]
minimize   P95(T_discovery), P99(T_discovery)
minimize   R_empty
maximize   F(T)
maximize   Q_topo(T)
minimize   T_consensus
```

其中 `Q_topo(T)` 可由多个评估指标构成：

```text
Q_topo_eval(T) = alpha_1 * |E_D(T)| / |E^*|
          + alpha_2 * S_max(T) / N
          + alpha_3 * normalized_lambda_2(T)
          - alpha_4 * C(T)
```

该式仅用于评估和离线分析。协议在线执行时不能读取 `E^*`、`lambda_2` 或 `C(T)` 的真实全局值。

## 4. 协议约束

### C1. 分布式约束

每个节点只能根据本地状态独立决策：

```text
a_i(t) = (m_i(t), b_i^m(t)) = pi_i(local_state_i(t))
```

### C2. 模式互斥约束

每个时隙只能选择一种模式：

```text
m_i(t) in {SENSE, TX, RX, IDLE}
```

### C3. 无 oracle 约束

策略 `pi_i` 不得使用：

- 未发现邻居的真实位置；
- 真实全局拓扑；
- 全局代数连通度；
- 中心调度结果；
- 未来时隙信息。

### C4. 探索下界约束

为避免 ISAC 漏检导致邻居永久不可发现，所有 beam cell 应保留非零探索概率：

```text
P_i^B(b,t) >= epsilon / M, for all b in B_i
```

### C5. 握手确认约束

ISAC 观测不能直接生成邻居链路。链路 `(i,j)` 只有在完成双向波束覆盖和链路层握手后才加入 `E_D(t)`。

## 5. 线上代理目标

协议执行时使用本地 beam-cell priority。该 priority 是在线代理，不是全局拓扑最优解：

```text
w_i,b(t) = r_i,b(t) * l_i,b(t) * h_i,b(t) / c_b
```

其中：

- `r_i,b(t)` 来自 ISAC occupancy prior。
- `l_i,b(t)` 是链路质量代理，例如感知强度、距离粗估计或历史握手成功率。
- `h_i,b(t)` 是拓扑价值代理。发现前只能由方向覆盖稀缺性、候选密度、已发现度数不足和长期未探索方向构成；发现后可加入确认邻居的局部信息。
- `c_b` 是扫描代价，可设为 1 或与机械/电子扫描开销相关。

beam selection distribution：

```text
P_i^B(b,t) = (1 - epsilon) * softmax_beta(w_i,b(t)) + epsilon / M
```

## 6. 核心假设检验

H1：相较均匀随机扫描，ISAC occupancy prior 可降低 `R_empty` 和平均发现时延。

H2：相较 ISAC-only，加入 topology-value proxy 可提升有限时间拓扑质量 `Q_topo(T)`。

H3：在中等虚警、漏检和角度误差下，探索下界和握手确认可防止性能灾难性退化。

H4：topology-aware priority 的收益应体现在离线评估指标 `Q_topo_eval(T)`，但其在线计算不得直接使用 `Q_topo_eval(T)` 中的真值项。
