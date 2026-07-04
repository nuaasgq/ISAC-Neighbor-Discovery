# WHY-HOW-WHAT 文献扫描 v1

## 范围

本轮为 Sprint 1 首轮扫描，目标不是完整综述，而是判断课题是否可做、最近竞争点在哪里、研究边界如何收窄。

覆盖三类文献：

1. 用户提供的 3 篇 ISAC/UAV beam management PDF。
2. 传统 directional neighbor discovery / airborne neighbor discovery。
3. ISAC-assisted beam management / beam tracking / beam rendezvous。

---

## A. 用户提供的 3 篇 PDF

| 文献 | WHY | HOW | WHAT | 场景 | 是否邻居发现 | 对本课题的意义 | 差异/风险 |
|---|---|---|---|---|---|---|---|
| Xu et al., **Deep Learning-Based Predictive Bidirectional Beamforming in ISAC-Enabled UAV Networks**, IEEE TWC 2026, DOI: [10.1109/TWC.2026.3664980](https://doi.org/10.1109/TWC.2026.3664980) | UAV 高动态、位置波动和姿态变化导致波束预测困难。 | 地面 BS 利用通信信号回波做实时 UAV tracking，提出 HECTA-Net 从历史 ISAC echoes 中预测收发 beamforming 矩阵。 | 深度学习预测性能接近理论上界，面向随机 UAV 运动具有鲁棒性。 | UAV-BS，ground BS 主导；已知服务关系。 | 否，属于 predictive beamforming / tracking。 | 证明 ISAC echoes 可作为波束选择先验。 | 非 U2U、非分布式、非对准前邻居发现；若我们只写 ISAC beam prediction，会被该类工作覆盖。 |
| Cui et al., **Seeing Is Not Always Believing: ISAC-Assisted Predictive Beam Tracking in Multipath Channels**, IEEE WCL 2024, DOI: [10.1109/LWC.2023.3303949](https://doi.org/10.1109/LWC.2023.3303949) | LoS 假设下的 ISAC beam tracking 在 multipath 中可能失真。 | 基于反射回波估计运动参数，设计 EKF 角度预测和 fine beam tracking。 | 雷达回波观测角不总是最优通信对准方向，需要细粒度跟踪弥合 gap。 | Cellular mmWave UAV，BS-UAV。 | 否，属于 beam tracking。 | 对我们非常关键：ISAC 排空波位不能假设完美，必须建模虚警、漏检、角度误差、多径误导。 | 不是协议层邻居发现，但会成为审稿人质疑“感知先验可靠性”的依据。 |
| Cui et al., **Sensing-Assisted Accurate and Fast Beam Management for Cellular-Connected mmWave UAV Network**, China Communications 2024, DOI: [10.23919/JCC.ea.2023-0140.202401](https://doi.org/10.23919/JCC.ea.2023-0140.202401) | mmWave UAV 网络中 IA 和 beam tracking 延迟高、对准精度低。 | 结合 ISAC、computer vision 和 EKF，设计 beam tracking/prediction，并用 dual identity association 区分动态多 UAV。 | 降低 IA delay、tracking error，提高 association accuracy 和通信性能。 | Cellular-connected UAV，BS 侧主导，多 UAV。 | 部分涉及 IA，但不是 U2U neighbor discovery。 | 提供 sensing-assisted beam management 的强对照；说明 sensing 可降低 IA 开销。 | High risk：覆盖 UAV + sensing + IA，但集中式 BS、有视觉辅助、非分布式 U2U。 |

---

## B. 传统 Directional Neighbor Discovery / UAV DND

| 文献 | WHY | HOW | WHAT | 关键假设 | 与本课题差异 |
|---|---|---|---|---|---|
| Vasudevan et al., **On Neighbor Discovery in Wireless Networks with Directional Antennas**, 2005, technical report / INFOCOM-era foundational work | 方向天线下邻居发现需要双方波束互指，盲发现概率低。 | 随机 direct discovery 与 gossip。 | 给出同步/异步概率分析。 | 二维扇区天线，随机 TX/RX，无位置先验。 | 奠基随机方法；无 UAV、无 3D、无 ISAC、无拓扑质量目标。 |
| Zhang & Li, **Neighbor discovery in mobile ad hoc self-configuring networks with directional antennas**, IEEE TWC 2008, DOI: [10.1109/TWC.2008.05908](https://doi.org/10.1109/TWC.2008.05908) | 方向天线邻居发现常被认为比全向慢。 | 2-way random 与 scan-based 算法对比。 | 证明合理设计的方向发现可优于全向发现。 | 二维方向天线，无邻居方向先验。 | 可作为传统 random/scan 基线；没有感知先验。 |
| Cai & Wolf, **On 2-way neighbor discovery in wireless networks with directional antennas**, IEEE INFOCOM 2015, DOI: [10.1109/INFOCOM.2015.7218439](https://doi.org/10.1109/INFOCOM.2015.7218439) | 方向通信需要双向发现，单向发现不足。 | 建模 2-way TX/RX 扫描和握手机制。 | 给出平均发现时间分析。 | 无邻居方向先验，二维。 | 可作为 2-way handshake 模型基础；不解决三维空间膨胀。 |
| Chen et al., **On Oblivious Neighbor Discovery in Distributed Wireless Networks With Directional Antennas**, IEEE/ACM TON 2017, DOI: [10.1109/TNET.2017.2673862](https://doi.org/10.1109/TNET.2017.2673862) | 无同步、无协调、异构方向天线下仍需有界发现。 | 基于 CRT 设计扫描和模式序列。 | 给出 order-minimal 最坏发现时延。 | 完全分布式、无位置、异步、异构。 | 最接近“无先验确定性保证”；但仍是盲扫序列设计，不利用 ISAC 缩小有效波位空间。 |
| Wang et al., **Directional neighbor discovery in mmWave wireless networks**, Digital Communications and Networks 2021, DOI: [10.1016/j.dcan.2020.09.005](https://doi.org/10.1016/j.dcan.2020.09.005) | mmWave ad hoc 建链需要空间 rendezvous。 | HDND，设计 TX/RX 序列与连续旋转扫描。 | 给出发现条件和最坏时延。 | mmWave ad hoc，无控制信道、无先验、无协调。 | 窄波束盲扫强基线；无 UAV 三维和拓扑优先。 |
| Bai et al., **Cognitive Neighbor Discovery With Directional Antennas in Self-Organizing IoT Networks**, IEEE IoT-J 2021, DOI: [10.1109/JIOT.2020.3037067](https://doi.org/10.1109/JIOT.2020.3037067) | 随机 DND 具有长尾和低效扫描问题。 | 认知框架动态调整发现策略。 | 降低期望发现时间。 | 同步 randomized ND，历史发现反馈。 | 支撑“随机方法可学习但受搜索空间限制”的论述；学习信号不是 ISAC 感知。 |
| Hong et al., **Oblivious neighbor discovery algorithms in airborne networks with directional multi-antenna**, Ad Hoc Networks 2023, DOI: [10.1016/j.adhoc.2022.103074](https://doi.org/10.1016/j.adhoc.2022.103074) | Airborne network 中无相对位置、异步、异构多天线导致发现困难。 | CRT + MAND 多天线扫描序列。 | 建立成功发现条件和最坏时延界。 | 无相对位置、异步、异构、匿名；给定高度下二维化。 | Airborne 场景强相关；但没有 ISAC 波位占据先验，也不做三维拓扑质量优先。 |
| Lan et al., **3D directional neighbor discovery**, IJCS 2023, DOI: [10.1002/dac.5496](https://doi.org/10.1002/dac.5496) | 二维模型不足以描述三维方向发现。 | 三维 scan-based neighbor discovery。 | 扩展 3D 扫描空间和发现过程。 | 细节待全文复核。 | 解决 3D 扫描，但仍是规则扫描，不做感知先验或拓扑价值排序。 |
| Zhu et al., **SkyOrbs: A Fast 3-D Directional Neighbor Discovery Algorithm for UAV Networks**, IEEE TMC 2024, DOI: [10.1109/TMC.2024.3451991](https://doi.org/10.1109/TMC.2024.3451991) | 高动态 UAV 方向 ND 中同步、全向、先验假设过强。 | Skip scanning + 受限机械旋转路径。 | 降低 3D UAV ND 时延。 | UAV，高动态，弱化理想假设，考虑机械转动限制。 | 最强传统竞争文献；我们的差异必须稳定落在 ISAC 排空波位 + 拓扑质量优先。 |
| Yuan et al., **An Adaptive 3D Neighbor Discovery and Tracking Algorithm in Battlefield Flying Ad Hoc Networks with Directional Antennas**, Sensors 2024, DOI: [10.3390/s24175655](https://doi.org/10.3390/s24175655) | FANET 三维移动导致发现和跟踪困难。 | 3D 顺序扫描 + two-way handshake + 发现后位置交换跟踪。 | 改善发现时间和链路生存。 | GNSS 时间同步、节点 ID、发现后交换 3D 位置。 | 不满足我们“对准前无 GNSS/惯导/历史轨迹”的强约束。 |
| Wang et al., **Intelligent Beam Configuration for Neighbor Discovery in Ad Hoc Networks With Directional Antennas**, IEEE TVT 2024, DOI: [10.1109/TVT.2024.3437425](https://doi.org/10.1109/TVT.2024.3437425) | 移动 ad hoc 中 beam configuration 是 ND 核心瓶颈。 | Personalized FL + DDPG + MAML。 | 优于若干基线。 | 移动 ad hoc，方向通信，存在训练/联邦流程。 | 代表 learning/RL 路线；但不是 ISAC，且三维搜索空间过大时学习收益有限。 |

---

## C. ISAC / Sensing-Assisted Beam Management 近邻工作

| 文献 | WHY | HOW | WHAT | 场景 | 对准前 discovery | 协议层 | 拓扑感知 | 威胁 |
|---|---|---|---|---|---|---|---|---|
| Hong et al., **Integrated sensing and communication-assisted beam rendezvous in airborne networks**, Computer Communications 2024, DOI: [10.1016/j.comcom.2024.01.019](https://doi.org/10.1016/j.comcom.2024.01.019) | Airborne network 方向链路 beam training 开销大。 | 用 ISAC 感知角度做 3D beam prediction / rendezvous。 | 降低 beam training 和 link overhead，提高速率。 | Airborne nodes。 | 部分，beam rendezvous 而非 neighbor discovery。 | 偏物理/链路边缘。 | 否。 | High：题面和我们非常接近，必须精读并区分 discovery protocol。 |
| Li et al., **Frame Structure and Protocol Design for Sensing-Assisted NR-V2X Communications**, IEEE TMC 2024, DOI: [10.1109/TMC.2024.3389697](https://doi.org/10.1109/TMC.2024.3389697) | NR-V2X IA、connected mode、beam failure recovery 开销高。 | 设计 sensing-assisted frame structure/protocol。 | 降低 pilot/reference signal overhead。 | gNB-V2I。 | 部分：initial access。 | 是。 | 否。 | High：协议层和 IA 非常接近，但不是分布式 U2U。 |
| Jain et al., **CommRad: Context-Aware Sensing-Driven Millimeter-Wave Networks**, ACM SenSys 2024, DOI: [10.1145/3666025.3699363](https://doi.org/10.1145/3666025.3699363) | 5G NR beam training/feedback 开销高，雷达缺乏用户/反射体/遮挡物语义。 | radar + radio 协同，利用 context 缩小或维护 beam 方向。 | 实测 28 GHz testbed 提升吞吐并降低 beam management overhead。 | AP/BS + radar + UE。 | 部分：IA/beam management。 | 系统协议实现。 | 否。 | High：感知排除空波位思想接近；差异在基础设施辅助而非 U2U 分布式。 |
| Orimogunje et al., **Sensing-Assisted Adaptive Beam Probing With Calibrated Multimodal Priors and Uncertainty-Aware Scheduling**, IEEE WCL 2026, DOI: [10.1109/LWC.2026.3679539](https://doi.org/10.1109/LWC.2026.3679539) | Exhaustive codebook sweep 开销大。 | radar/LiDAR/camera 生成 beam prior，UCB 选择小 probe set。 | 在小 probe 数下达到高 Top-k 命中率。 | AP-UE，多模态侧信息。 | 部分：beam probing。 | 算法/调度层。 | 否。 | High：形式上接近“感知先验 + 小集合探测”；差异在 U2U、ND、拓扑。 |
| Chen et al., **Enhancing THz/mmWave Network Beam Alignment With Integrated Sensing and Communication**, IEEE Communications Letters 2022, DOI: [10.1109/LCOMM.2022.3171291](https://doi.org/10.1109/LCOMM.2022.3171291) | THz/mmWave beam alignment 易受 mobility/blockage 影响。 | ISAC 感知辅助 beam alignment。 | 降低 misalignment。 | Cellular BS-MT。 | 否。 | 偏分析。 | 否。 | Medium：说明 ISAC 可辅助网络级 beam alignment，但不覆盖 discovery。 |
| Yuan et al., **Radar-Assisted Predictive Beamforming for Vehicular Links: Communication Served by Sensing**, IEEE TWC 2020, DOI: [10.1109/TWC.2020.3015735](https://doi.org/10.1109/TWC.2020.3015735) | V2I beam tracking 依赖 pilots/feedback。 | RSU 复用 radar sensing，EKF 跟踪车辆参数。 | 降低 tracking overhead。 | RSU-vehicle。 | 否。 | 物理层。 | 否。 | Medium：经典 sensing-served communication 思路源头之一。 |
| González-Prelcic et al., **Radar aided beam alignment in mmWave V2I communications supporting antenna diversity**, ITA 2016, DOI: [10.1109/ITA.2016.7888145](https://doi.org/10.1109/ITA.2016.7888145) | V2I mmWave 初始波束搜索开销大。 | radar side information 辅助 beam alignment。 | 降低搜索复杂度。 | Infrastructure radar + V2I。 | 部分：beam alignment。 | 否。 | 否。 | Low-Medium：思想源头相关，但非 UAV/ISAC/ND。 |

---

## 初步结论

1. **方向可做，但不能写成“ISAC 辅助波束对准”。** 这一点已经被 V2I、UAV-BS、AP-UE、beam probing 和 beam tracking 大量覆盖。
2. **传统 DND 侧最强竞争是 SkyOrbs 2024、Hong 2023、Chen 2017。** 它们覆盖无先验、分布式、UAV/airborne、3D 或有界时延，但没有 ISAC 波位占据先验。
3. **ISAC 侧最强威胁是 Hong 2024 beam rendezvous、Li 2024 NR-V2X protocol、CommRad 2024、Orimogunje 2026 adaptive beam probing。** 它们证明“感知先验缩小 beam search”不是新概念。
4. **我们应守住的核心 gap 是组合式的：** fully distributed U2U + pre-alignment neighbor discovery + beam-cell occupancy prior + non-uniform randomized ND + topology-aware priority。
5. **拓扑质量是当前最有价值的第二贡献。** 现有工作多优化 beam alignment overhead、发现时延、能耗、吞吐，较少把有限时间邻居发现与关键链路、连通性、代数连通度或一致性收敛直接耦合。
