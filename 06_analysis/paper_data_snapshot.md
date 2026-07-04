# 论文数据快照：五类对比闭环 v1

日期：2026-07-04

## 五类对比

本轮实验覆盖论文必须比较的五类方法：

| 类别 | 协议名 | 说明 |
|---|---|---|
| 参考文献基线 | `skyorbs_like_skip_scan` | SkyOrbs-like 3D skip-scan 工程近似 |
| 完全随机 | `uniform_random` | 随机 mode/beam 基线 |
| RL 无 ISAC | `rl_no_isac` | 在线 RL 代理基线，无 ISAC belief |
| 改进 RL 无 ISAC | `improved_rl_no_isac` | memory/topology-aware 代理基线，无 ISAC belief |
| 改进 RL + ISAC | `improved_rl_isac` | 通信伴随 ISAC belief + memory/topology-aware 代理 |

当前 `rl_no_isac / improved_rl_no_isac / improved_rl_isac` 是可运行的在线策略代理，用于完成论文实验链路和消融判据；后续深度 MARL 训练器应通过 `marl_env.py` 替换这些代理策略。

## 主结果：Gauss-Markov D1

命令：

```powershell
python 05_simulation/run_smoke.py `
  --config 05_simulation/configs/paper_core_d1.yaml `
  --output 05_simulation/results_raw/paper_core_d1_10ep_main `
  --episodes 10 `
  --protocols skyorbs_like_skip_scan,uniform_random,rl_no_isac,improved_rl_no_isac,improved_rl_isac `
  --mobility gauss_markov

python 06_analysis/scripts/analyze_dynamic_results.py `
  05_simulation/results_raw/paper_core_d1_10ep_main `
  --output 05_simulation/results_raw/paper_core_d1_10ep_main_analysis
```

核心统计：

| 协议 | 发现率 | 平均时延 | P95 | P99 | 空扫率 | lambda2 |
|---|---:|---:|---:|---:|---:|---:|
| `improved_rl_isac` | 0.934 | 61.57 | 243.29 | 290.36 | 0.197 | 4.521 |
| `improved_rl_no_isac` | 0.844 | 106.17 | 300.00 | 300.00 | 0.452 | 3.526 |
| `rl_no_isac` | 0.823 | 113.32 | 293.96 | 300.00 | 0.442 | 3.384 |
| `uniform_random` | 0.815 | 121.70 | 300.00 | 300.00 | 0.438 | 3.178 |
| `skyorbs_like_skip_scan` | 0.479 | 180.12 | 300.00 | 300.00 | 0.445 | 1.504 |

结论：`improved_rl_isac` 在发现率、平均发现时延、P95/P99 长尾、空扫率和拓扑质量上均领先，满足当前五类对比的主成功判据。

## 鲁棒性：Random Walk D1

命令：

```powershell
python 05_simulation/run_smoke.py `
  --config 05_simulation/configs/paper_core_d1.yaml `
  --output 05_simulation/results_raw/paper_core_d1_random_walk_10ep `
  --episodes 10 `
  --protocols skyorbs_like_skip_scan,uniform_random,rl_no_isac,improved_rl_no_isac,improved_rl_isac `
  --mobility random_walk
```

核心统计：

| 协议 | 发现率 | 平均时延 | P95 | P99 | 空扫率 | lambda2 |
|---|---:|---:|---:|---:|---:|---:|
| `improved_rl_isac` | 0.930 | 77.43 | 252.51 | 294.19 | 0.388 | 2.755 |
| `improved_rl_no_isac` | 0.915 | 92.90 | 291.01 | 294.85 | 0.436 | 2.533 |
| `rl_no_isac` | 0.913 | 99.18 | 288.77 | 298.36 | 0.429 | 2.672 |
| `uniform_random` | 0.872 | 113.51 | 298.84 | 300.00 | 0.432 | 2.267 |
| `skyorbs_like_skip_scan` | 0.489 | 175.71 | 300.00 | 300.00 | 0.441 | 0.906 |

结论：随机游走下 proposed 仍保持发现率和时延优势，说明主结论不是 Gauss-Markov 单一运动模型造成的偶然结果。

## 压力边界：无容差窄波束

命令：

```powershell
python 05_simulation/run_smoke.py `
  --config 05_simulation/configs/mobile_smoke.yaml `
  --output 05_simulation/results_raw/stress_mobile_smoke_h400_passive_isac `
  --episodes 5 `
  --slots 400 `
  --protocols skyorbs_like_skip_scan,uniform_random,rl_no_isac,improved_rl_no_isac,improved_rl_isac `
  --mobility gauss_markov
```

在 `alignment_tolerance_cells=0` 的压力场景下，所有协议发现率均低于 0.07，P95/P99 全部被时限截断。该结果应作为失败边界或高难度压力测试报告，不能作为主性能图。

## 论文写作状态

当前数据已经足够支撑一版“方法可行性 + 五类对比 + 鲁棒性运动模型”的结果段落，但还不够支撑最终 TWC/TCOM 投稿：

1. 需要把在线策略代理替换或补充为真实 MARL 训练结果。
2. 需要把 episode 数扩展到 20-30 seeds，并固定开发/测试 seed split。
3. 需要增加 ISAC 误差敏感性、beam 数扩展、节点规模扩展。
4. 需要把 SkyOrbs-like 近似继续校准到原文扫描序列。
