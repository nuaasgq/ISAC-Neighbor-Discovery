# ISAC-Neighbor-Discovery

面向完全分布式 UAV-UAV 集群网络的通感一体化辅助窄波束邻居发现研究。

## Research Focus

本项目研究如何将 ISAC 感知能力抽象为链路层邻居发现协议可用的先验信息，在无中心控制、对准前未知邻居状态的三维窄波束条件下，减少空波位扫描，并在有限时间内优先建立支撑拓扑质量提升的关键链路。

当前设定允许 UAV 知道自身位置、速度、航向和姿态，但不假设其在对准前知道未发现邻居的身份、实时位置、波束状态或全局拓扑。

从 Sprint 5 起，MVP 主场景采用三维动态 UAV 运动模型。默认运动模型为高斯马尔可夫，随机方向、随机游走和随机航点模型用于鲁棒性扫描；静态节点仅作为调试模式。

## Workflow

主控工作流见：

- `00_workflow/研究工作流.md`

当前阶段：

- P3-P5 过渡：系统模型、规则协议、动态三维仿真底座、MARL 算法族筛选、网络结构创新和可扩展仿真实验设计

当前新增任务线：

- `00_workflow/sprint_4_marl_algorithm_exploration.md`
- `00_workflow/sprint_5_dynamic_simulation_task_breakdown.md`
- `04_protocol/comparison_baselines.md`
- `04_protocol/marl_algorithm_suite.md`
- `04_protocol/neural_architecture_innovation.md`
- `05_simulation/configs/paper_comparison_matrix.yaml`
- `05_simulation/configs/marl_algorithm_sweep.yaml`
- `05_simulation/src/isac_nd_sim/mobility.py`
- `05_simulation/src/isac_nd_sim/simulator.py`
- `05_simulation/src/isac_nd_sim/runner.py`
- `05_simulation/configs/mobile_smoke.yaml`
- `05_simulation/tests/test_mobility.py`
- `05_simulation/tests/test_simulator_smoke.py`

快速运行动态 smoke test：

```powershell
python 05_simulation/run_smoke.py --episodes 1 --slots 50 --protocols uniform_random,itap_nd --mobility gauss_markov
python -m pytest 05_simulation/tests -q
```

## Repository Rules

- 所有研究文档、仿真代码、实验配置、数据处理脚本和论文草稿均以本仓库为版本控制主线。
- 不提交第三方 PDF 原文、大体积原始实验结果、临时缓存和编译中间文件。
- 阶段性产物完成后，先检查变更范围，再提交并推送远端仓库。
