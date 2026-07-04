# 符号表 v0

| 符号 | 含义 |
|---|---|
| `N` | UAV 节点数量 |
| `i, j` | 节点索引 |
| `t` | 离散时隙索引 |
| `A` | 本地水平扫描维度 beam cell 数 |
| `E` | 本地垂直扫描维度 beam cell 数 |
| `M` | 单节点 beam cell 总数，`M = A * E` |
| `B_i` | 节点 `i` 的 body-frame 本地 beam cell 集合 |
| `b_i^m` | 节点 `i` 的第 `m` 个本地 beam cell |
| `b` | 一个通用 beam cell 记号 |
| `m_i(t)` | 节点 `i` 在时隙 `t` 的模式，属于 `{SENSE, TX, RX, IDLE}` |
| `a_i(t)` | 节点 `i` 的动作，可包含模式和 beam-cell 选择 |
| `O_i(t)` | 节点 `i` 在时隙 `t` 的 ISAC 观测 |
| `z_i(t)` | 节点 `i` 的原始或抽象感知观测 |
| `p_i^m(t)` | 节点 `i` 对本地 beam cell `b_i^m` 的 occupancy prior |
| `q_i,b(t)` | `p_i^m(t)` 的等价通用写法 |
| `u_i^m(t)` | 节点 `i` 对 `b_i^m` 的不确定性 |
| `tau_i^m(t)` | `b_i^m` 最近一次观测时间 |
| `P_D` | ISAC 检测概率 |
| `p_fa` | ISAC 虚警概率 |
| `p_md` | ISAC 漏检概率 |
| `sigma_az` | 本地水平维度感知误差 |
| `sigma_el` | 本地垂直维度感知误差 |
| `sigma_cell` | beam-cell 级角度偏移强度 |
| `P_noncomm` | 感知目标为非通信目标的概率 |
| `P_multipath` | 多径误导概率 |
| `rho` | occupancy prior 更新或时间衰减系数 |
| `E^*` | 真实可通信链路集合，仅仿真可见 |
| `E_D(t)` | 截至时隙 `t` 已发现链路集合 |
| `N_i^D(t)` | 节点 `i` 已发现邻居集合 |
| `d_i^D(t)` | 节点 `i` 已发现度数 |
| `P_i^B(t)` | 节点 `i` 的 beam selection probability distribution |
| `epsilon` | 探索下界，保证低置信 beam cell 仍有非零扫描概率 |
| `w_i,b(t)` | beam cell `b` 的综合优先级权重 |
| `r_i,b(t)` | occupancy prior 贡献项 |
| `l_i,b(t)` | link-quality proxy |
| `h_i,b(t)` | topology-value proxy |
| `Q_i^m(t)` | 发现前拓扑代理优先级，也可作为 `h_i,b(t)` 的组成 |
| `c_b` | 扫描 beam cell `b` 的代价 |
| `T_ij` | 链路 `(i,j)` 的发现时延 |
| `F(T)` | 有限时间 `T` 内的发现率 |
| `R_empty` | 空扫比例 |
| `C(t)` | 已发现图的连通分量数 |
| `S_max(t)` | 最大连通子图规模 |
| `lambda_2(t)` | 已发现图 Laplacian 的代数连通度，评估用 |
| `T_consensus` | 一致性误差达到阈值所需时间 |
| `T_budget` | 邻居发现时间预算 |
