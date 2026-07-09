# MARL Advantage Audit, 2026-07-09

## 结论

当前 N=10、Wang-aligned、200-slot 结果不能解释为“MARL 网络训练后全面学会并显著超过 Wang/随机”。更准确的结论是：

1. MARL 执行策略没有发现直接 true-edge/真实邻居位置泄漏；观测使用的是自身状态、ISAC belief、局部历史、候选 mask/score 和拓扑缺口。
2. 之前汇总表里 MARL 与基线使用了不同 scenario seed，这是实验严谨性问题；但配对重跑后，基线仍远低于 MARL，因此 seed 不一致不是性能跃升主因。
3. 性能跃升主要来自 ISAC candidate hard mask、candidate score 和 rule residual/contention prior，而不是训练权重本身。
4. 训练权重目前只提供边际增益：trained_full 发现率 0.471，zero_weights_full 也能达到 0.422，random_weights_full 达到 0.409。

因此，当前方法如果写论文，不能把贡献重点放在“MARL 自主学习出了高性能策略”；应该转向“ISAC 辅助候选波束空间裁剪 + 规则先验/可学习残差 + 分布式动作决策”的跨层机制，并且必须进一步补强真正的学习贡献。

## 审计设置

- 配置：`05_simulation/configs/wang2025_reproduction_smoke.yaml`
- 场景：N=10，105 beams，约 25 度，single RF，200 slots，5 episodes
- 公共环境：`wang2025_isac_tables`
- MARL checkpoint：`05_simulation/results_raw/marl_campaign/wang2025_aligned_n10_fixedhandshake_20260709/marl_wang_isac_tables_discovery_first/train/final_model.pt`
- 配对测试种子：2026084001 到 2026084005
- 输出目录：`06_analysis/paper_tables/marl_advantage_audit_20260709/`

## 配对基线结果

| Method | Discovery rate | Mean delay | lambda2 |
| --- | ---: | ---: | ---: |
| Uniform TX/RX random | 0.0044 | 199.14 | 0.000 |
| Wang sensing-table action policy | 0.0267 | 196.76 | 0.000 |
| Budgeted ISAC rule | 0.0444 | 195.15 | 0.000 |
| MARL trained_full | 0.4711 | 153.80 | 1.707 |

说明：配对 seed 后，随机、Wang、旧规则仍很低。基线低不是 seed 导致的，但它也反过来说明 Wang 复现与论文原始高性能条件仍存在差距，后续需要继续拆解 Wang 的 RF 数、握手假设、表交换时机和冲突避免条件。

## MARL 消融结果

| Ablation | Discovery rate | Edges | Mean delay | lambda2 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| trained_full | 0.4711 | 21.2 | 153.80 | 1.707 | 当前完整模型 |
| trained_no_candidate_mask | 0.0889 | 4.0 | 192.33 | ~0 | 去掉 hard mask 后几乎退化 |
| trained_no_candidate_score | 0.3689 | 16.6 | 167.94 | 0.966 | score 有贡献，但不是最大因素 |
| trained_no_rule_residual | 0.2711 | 12.2 | 180.32 | 0.073 | 规则残差/模式先验贡献很大 |
| trained_no_mask_score_rule | 0.0000 | 0.0 | 200.00 | 0.000 | 去掉三类先验后完全失败 |
| random_weights_full | 0.4089 | 18.4 | 159.94 | 1.179 | 随机权重 + 强先验已很高 |
| zero_weights_full | 0.4222 | 19.0 | 159.09 | 1.310 | 零网络权重 + 强先验已很高 |
| zero_weights_no_rule | 0.2889 | 13.0 | 176.57 | 0.303 | hard mask + contention prior 仍很强 |

最关键证据是 `zero_weights_full` 和 `random_weights_full`。如果网络权重没有训练甚至全为零仍能接近完整模型，说明现阶段的“优势”主要来自工程化先验和动作空间裁剪，而不是 MARL 训练本身。

## 代码层原因

1. Hard candidate mask 直接裁剪 beam 动作空间。
   - `05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py:86`
   - `05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py:99`
   - `05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py:129`

   评估时 `act()` 调用 `hard_mask=True`，只要 candidate mask 非空，非候选 beam logits 被置为 `-1e9`。这相当于把 105 beam 的盲搜索压到约 sqrt(105) 个候选 beam，而 Wang/随机基线没有同等 hard mask。

2. Candidate mask 本身来自强规则。
   - `05_simulation/src/isac_nd_sim/marl_env.py:292`
   - `05_simulation/src/isac_nd_sim/marl_env.py:307`
   - `05_simulation/src/isac_nd_sim/marl_env.py:314`
   - `05_simulation/src/isac_nd_sim/marl_env.py:316`
   - `05_simulation/src/isac_nd_sim/marl_env.py:318`

   mask 由 belief、success、fail、recency、age 计算 raw_score，再取 top-k 和高 quantile belief。这个候选空间选择机制本身就是一个强启发式协议。

3. Rule residual 和 contention prior 直接改写 mode/beam logits。
   - `05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py:159`
   - `05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py:164`
   - `05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py:168`
   - `05_simulation/src/isac_nd_sim/neural_contention_actor_critic.py:172`

   即使网络权重置零，`candidate_score - 0.85 * collision_norm` 和 `_contention_mode_prior()` 仍会给出有效动作偏置，所以 zero-weight policy 仍然表现很好。

## 可信结论边界

可以保留的结论：

- ISAC 辅助排除空波位/形成候选波束池在窄波束邻居发现中非常有效。
- 只要候选 mask 与 piggyback sensing 合理，单 RF、200-slot 下也能显著提升发现率。
- 分布式策略中加入局部 contention/topology 状态能改善动作角色选择和拓扑连通性。

目前不能声称的结论：

- 不能说“MARL 网络结构创新是主要性能来源”。
- 不能说“训练模型全面学习出优于规则的策略”，因为零权重/随机权重已经接近完整模型。
- 不能把 trained_full 与 Wang/随机的巨大差距直接作为 MARL 学习贡献。

## 下一步修正方向

1. 把 `ISAC candidate hard mask` 单独定义为一个公平基线：`Wang + same candidate mask`、`Random mode + same candidate mask`、`Rule prior + same candidate mask`。
2. 把 MARL 贡献改成残差学习：固定同一 candidate mechanism，比较 zero/random/rule prior 与 trained residual 的增益。
3. 若要保留“网络结构创新”，需要让网络产生不可由规则直接得到的增益，例如学习动态 TX/RX 比例、候选 beam 排序、表交换后的重复边抑制、或 topology-aware edge prioritization。
4. 对 Wang 复现继续拆解：RF chains、冲突避免、重复边不回复、邻居信息表/感知表交换时机、论文中是否默认多 RF 或更宽候选响应窗口。
5. 重新设计主图：主结果不要只画 trained_full vs weak baselines，应加入 zero/random/rule-prior 消融，否则容易被审稿人指出性能来自手工规则。
