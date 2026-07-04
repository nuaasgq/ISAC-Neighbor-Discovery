# 协议设计 v0

## 名称

ISAC-Assisted Topology-Aware Probabilistic Neighbor Discovery，简称 `I-TAP-ND`。

## 设计目标

在完全分布式、无中心、无对准前邻居先验的 UAV-UAV 三维窄波束网络中，利用 ISAC 生成的 beam-cell occupancy prior 减少空扫，并通过 topology-aware priority 在有限时间内优先发现更有价值的链路。

## 核心原则

1. ISAC 只提供候选 beam cell 先验，不直接确认邻居。
2. 协议保留随机性，不依赖中心调度。
3. 所有 beam cell 保留探索概率，避免漏检导致永久不可发现。
4. 链路必须通过发现 beacon 和握手确认。
5. topology-aware priority 只能使用本地代理，不使用真实全局拓扑。

## 节点状态

每个节点维护：

- `Q_i[b]`：beam cell occupancy score。
- `U_i[b]`：不确定性。
- `Age_i[b]`：距离上次感知的时隙数。
- `Fail_i[b]`：该 beam cell 上发现失败统计。
- `Succ_i[b]`：该 beam cell 上发现成功统计。
- `N_i^D`：已发现邻居集合。
- `D_i`：已发现节点度。
- `SectorCover_i[b]`：空间方向覆盖稀缺性代理。

## 时隙流程

每个时隙，节点独立执行：

1. 更新观测年龄和置信度衰减。
2. 根据 sensing schedule 决定是否进入 `SENSE`。
3. 若不 sensing，则选择 `TX`、`RX` 或 `IDLE`。
4. 根据 beam-cell priority 采样 beam cell。
5. 若 `TX`，发送 discovery beacon。
6. 若 `RX`，监听 discovery beacon。
7. 若 beacon 和波束互相覆盖，执行握手确认。
8. 若成功，更新已发现邻居和 beam cell 成功统计。
9. 若失败，更新失败统计和不确定性。

## Beam-Cell Priority

综合优先级：

```text
w_i,b = r_i,b * l_i,b * h_i,b / c_b
```

### Occupancy term

```text
r_i,b = decay(Age_i[b]) * Q_i[b] + uncertainty_bonus(U_i[b])
```

作用：优先扫描可能存在目标且观测较新的 beam cell，同时对不确定区域保留一定探索。

### Link-quality proxy

第一版可设：

```text
l_i,b = 1 + eta_s * Succ_i[b] - eta_f * Fail_i[b]
```

若 ISAC 可提供回波强度或距离粗估计，可进一步引入路径损耗代理。

### Topology-value proxy

第一版采用本地可计算的发现前代理：

```text
score_i(b) = alpha * q_i(b)
           + beta  * diversity_i(b)
           + gamma * degree_need_i
           - eta   * staleness_i(b)
```

其中：

- `q_i(b)` 是 ISAC occupancy prior。
- `diversity_i(b)` 表示该方向是否远离已发现邻居方向。
- `degree_need_i = max(0, k_target - d_i) / k_target`。
- `staleness_i(b)` 表示该 cell belief 多久未更新。

该设计不声称精确优化代数连通度，而是作为有限时间拓扑质量的本地代理。

发现后可以加入 confirmed topology priority，例如已确认邻居的局部度数摘要、链路质量或二跳摘要。但这些信息只能来自握手后的邻居交换，不能用于发现前决策。

## 模式选择

节点以概率选择模式：

```text
P(SENSE) = p_sense(t)
P(TX)    = p_tx(t)
P(RX)    = p_rx(t)
P(IDLE)  = 1 - p_sense - p_tx - p_rx
```

MVP 可设固定概率。后续可根据发现率、观测年龄和失败率自适应调整。

## 消融版本

1. `Uniform-Random-ND`：均匀随机 beam cell + 随机 TX/RX。
2. `ISAC-Only-ND`：使用 occupancy prior，但 `h_i,b = 1`。
3. `Topology-Only-ND`：不使用 ISAC prior，仅用方向稀缺性、度数需求和探索。
4. `I-TAP-ND`：完整方法。
5. `Oracle-ND`：直接知道真实 occupied beam cells，仅作上界。

## 预期优势

- 降低空扫比例。
- 降低平均发现时延。
- 缓解长尾发现时延。
- 在有限时间内提升已发现图的连通性。
- 对感知误差保留鲁棒性。

## 主要风险

- 若 ISAC prior 过强，协议可能被认为使用 oracle。
- 若 topology proxy 过弱，拓扑收益可能不显著。
- 若模式选择固定，可能难以适应不同密度和误差水平。
- 若同步时隙假设过强，后续需扩展异步版本。
- 若发现前 topology-aware 表述过强，可能被质疑使用未知身份或全局拓扑；必须坚持 proxy 表述。
