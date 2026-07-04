# Sprint 5 任务拆解：动态三维仿真底座

## 目标

构建 ISAC 辅助 UAV-UAV 窄波束邻居发现的动态三维仿真底座。本轮取消“主场景静态节点”的 MVP 假设，先实现可复用的移动模型层、规则协议 runner 和 smoke 测试，为后续 MARL 环境接口做准备。

## 更新设定

- UAV 具备自身定位能力：可使用自身位置、速度、航向、俯仰、横滚和 beam-to-global 方向映射。
- 完成发现握手前，UAV 仍不知道未发现邻居的身份、实时位置、速度、姿态、波束状态和全局拓扑。
- UAV 在三维空间中运动；`static` 仅作为调试模式，不作为主实验设定。
- 移动模型至少支持高斯马尔可夫、随机游走、随机方向和随机航点。

## 并行任务拆解

| 任务 | 内容 | 可并行 | 产物 |
|---|---|---:|---|
| S5-A | 审查移动模型公式、边界处理、姿态耦合和 smoke 测试条件 | 是 | 本 sprint 任务拆解与移动配置要求 |
| S5-B | 实现三维状态更新、边界策略和姿态推导 | 是 | `05_simulation/src/isac_nd_sim/mobility.py` |
| S5-C | 将默认主场景从 static 改为动态移动，并暴露移动参数 | 是 | `mvp.yaml`, `marl_mvp.yaml`, `marl_algorithm_sweep.yaml` |
| S5-D | 增加 smoke 测试：非静止、边界安全、可复现、速度约束和姿态有限 | 是 | `05_simulation/tests/test_mobility.py` |
| S5-E | 集成规则协议 runner 和最小输出文件 | 是 | `05_simulation/run_smoke.py`, `05_simulation/src/isac_nd_sim/runner.py` |
| S5-F | 设计后续 MARL env 合约 | 否 | 本轮侧向审查输出，不写文件 |

## 执行顺序

1. 实现移动模型和边界处理原语。
2. 更新配置，使默认 MVP 使用 `gauss_markov` 而不是 `static`。
3. 将移动状态接入 beam-cell 几何、ISAC belief 和 Tx/Rx 发现判定。
4. 增加 smoke 测试，确保不依赖外部训练库即可运行。
5. 输出 runner 所需的最小结果文件。
6. 下一轮再固化 MARL `reset/step` 环境接口。

## 移动模型要求

| 模型 | 实验作用 | 最小参数 |
|---|---|---|
| `gauss_markov` | 主 MVP 移动模型，模拟平滑且时间相关的 UAV 运动 | `alpha`, `speed_mean_mps`, `speed_std_mps`, `min_speed_mps`, `max_speed_mps` |
| `random_walk` | 局部抖动和强随机压力测试 | `random_walk_step_std_m`, `min_speed_mps`, `max_speed_mps` |
| `random_direction` | 分段恒速随机方向运动 | `direction_update_period_slots`, `speed_mean_mps`, `speed_std_mps` |
| `random_waypoint` | 任务式航点运动 | `waypoint_threshold_m`, `waypoint_pause_slots`, `speed_mean_mps`, `speed_std_mps` |

## 边界建议

默认使用 `reflect` 边界。它能让 UAV 保持在三维空域内，同时避免 `wrap` 带来的瞬移现象。`clip` 只适合调试，可能造成节点贴边；`wrap` 可作为固定密度合成压力测试选项，但不作为默认空域模型。

## Smoke 测试标准

| 检查项 | 通过条件 |
|---|---|
| 位置边界 | 所有坐标保持在 `[0, area_size]` 内 |
| 非静止运动 | 若干 slot 后至少有 UAV 发生非零位移 |
| 速度约束 | 速度不超过配置的 min/max 容差 |
| 姿态有效 | yaw、pitch、roll 均为有限值 |
| 可复现性 | 相同 seed 生成相同轨迹和汇总结果 |
| 边界行为 | 反弹边界能保持节点在空域内 |
| 模型覆盖 | 高斯马尔可夫、随机游走、随机方向、随机航点均可 step |

## 下一检查点

下一轮进入完整 MARL 环境接口：

- `reset(seed) -> observations, info`
- `step(actions) -> observations, rewards, terminated, truncated, info`
- 从当前动态几何生成 ISAC 观测。
- 在移动条件下判定 Tx/Rx beam 对准、冲突和握手成功。
- 明确 actor 执行期、critic 训练期和评估期的信息边界。
