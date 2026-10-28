# SOURCES

外部资料登记。**未核验的条目不得进入 `paper/` 正文。**

字段：标题 / 来源 / 类型 / 与本项目的关系 / 状态（`verified` | `unverified` | `stale`）。

> 本次核验（2026-09-24，第 01 步）逐条给出了**核验方式与核验到的事实**。
> ⚠️ "状态 = verified" 指的是**该条目本身的信息已核对**（标题/作者/venue/年份/机制/结论量级），
> **不**表示我们复现了它，也**不**表示它的结论在本平台成立。

---

## 1. 论文与预印本

| 标题 | 来源 | 类型 | 与本项目的关系 | 状态 |
| --- | --- | --- | --- | --- |
| **Deadline-Aware Adaptive Prefill Chunking for Efficient Large Language Model Serving**（SLOWeave） | [arXiv:2609.07883](https://arxiv.org/abs/2609.07883)，v1 提交 2026-09-07；[全文](https://arxiv.org/html/2609.07883v1) | 预印本 | ⚠️ **与"成本模型自适应 chunk（第三臂）"最接近**。成本预测 + next-token deadline + 自适应 chunk | **verified**（2026-09-24，读摘要 + 全文 HTML） |
| **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve** | [OSDI '24 官方页](https://www.usenix.org/conference/osdi24/presentation/agrawal)；Agrawal 等（Georgia Tech / MSR India） | 会议论文（OSDI 2024） | 分块预填充 + stall-free 调度。本项目**固定 chunk 静态基线**的直接来源 | **verified**（读官方摘要页） |
| **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** | [OSDI '24 官方页](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)；Zhong 等（PKU / UCSD / StepFun） | 会议论文（OSDI 2024） | goodput 的**定义与讨论口径**；prefill-decode 干扰的量化。**多设备分离架构** | **verified**（读官方摘要页） |
| **DRLServe: Adaptive Prefill Chunking with Deep Reinforcement Learning for LLM Inference** | DOI [10.3233/FAIA250867](https://doi.org/10.3233/FAIA250867)（IOS Press / FAIA）；Sun, Bao, Wang（NUDT） | 会议论文（**ECAI 2025**） | 自适应 chunk 的**学习型**路线（RAPC 框架 + TAPPO 算法）。与第三臂同族 | **verified**（读出版社摘要 + 第三方著录） |
| _PagedAttention / vLLM_ | _待补会议年份与链接_ | 论文 | KV 分页管理与前缀复用的原始思想来源 | unverified |
| _SmoothQuant 类量化方法_ | _待补_ | 论文 | W8A8 精度-性能权衡的参考基线 | unverified |

**逐条核验记录**：

- **SLOWeave（最接近方法）**：选择"**预测能在最早活跃 decode 的 next-token deadline 前
  完成的最大 prefill chunk**"；**无需按负载调 chunk size**；对**单调迭代成本模型**做
  **对数时间搜索**；作者强调方法**刻意收窄**——不改编排路径、不改 kernel，
  与显存管理 / 分离式部署 / 模型并行**互补**。
  评测：event-driven 模拟器 + iteration-level GPU runtime；四类负载（chat / mixed /
  long / **bursty**）；TPOT 目标 10/25/50 ms；TTFT SLO 1000/1500/5000 ms。
  goodput = **同时**满足 TTFT 与请求内 P99 TPOT 的请求数 / trace 时长。
  结果量级：25 ms 目标下 goodput 比最强固定 chunk 基线 **+39%（mixed）/ +38%（long）**；
  10 ms 目标下升至 **3.3× / 2.4×**。
  ⚠️ **bursty 是"两状态到达过程"（高低速率交替），不是连续 load ramp**。
  ⚠️ **局限性（作者自述）**：评测覆盖的模型/加速器/负载/SLO 集合有限；真实迭代成本
  **可能非平滑**（kernel 边界、张量并行通信、显存压力、前缀命中）⇒ **标定漂移**需要更广泛
  部署研究；当前控制器假设**单一共享 TPOT 目标 + 每迭代一个 prefill chunk**；
  未建模抢占成本与 KV 传输；**在无限过载下不保证无饥饿**——持续过载时严格 TPOT 目标
  **可能没有安全的 prefill 预算**，需要准入控制或副本自动伸缩。
  ⚠️ 正文声称 simulator / cost parameters / workload generator / raw runs / aggregation code
  **随论文提供**，但**本次检索未找到可访问的仓库** ⇒ artifact**声明存在、位置未确认**。
- **Sarathi-Serve**：提出 **chunked-prefills**（把 prefill 切成近等大块）+
  **stall-free schedules**（在不暂停进行中 decode 的前提下把新请求加入 batch）
  ⇒ 支持大 batch 同时压低对时延的影响。量级：Mistral-7B 单 A100 服务容量 **2.6×**（对 vLLM）；
  Yi-34B 双 A100 **3.7×**；Falcon-180B + 流水并行 **5.6×**。源代码公开。
  ⚠️ 代码测试于 **CUDA 12.3 / H100 / A100**；其 README 自述是**研究原型**，
  与开源 vLLM **不具备完整功能对等**（只保留最关键功能）⇒ **只移植其调度规则时
  应标注为 policy 参考，不得声称为完整复现**。
- **DistServe**：把 prefill 与 decoding **分派到不同 GPU** ⇒ **消除 prefill-decode 干扰**，
  并按 TTFT / TPOT 需求**协同优化各阶段的资源分配与并行策略**。
  量级：可比系统下 **7.4× 更多请求** 或 **12.6× 更紧 SLO**（>90% 请求满足约束）。
  ⚠️ 是**多设备分离架构**，**不默认列为单卡同预算的必跑对照**；用于相关工作与
  **goodput 口径讨论**。
- **DRLServe**：RAPC（实时自适应分块框架，解耦推理与 chunk 调整）+ TAPPO
  （时间窗异步 PPO；首次把 LLM 推理建模为 MDP）。动作 = 从离散 chunk 集合中选；
  奖励 = TTFT 增量与 TBT 均值的分层归一化双目标 + 方差/标准差抑制项。
  基线：FasterTransformer / Orca / vLLM / **Sarathi-Serve** / RandomChunk。
  指标：TTFT、TBT 均值 / 方差 / 标准差。自报：TTFT **−50.9%**、TBT 均值 **−64.6%**、
  TBT 方差 **−99.0%**（相对 Sarathi-Serve）。
  - 会议与著录：**会议年份 = ECAI 2025**（FAIA 系列，IOS Press）。
    第三方著录（Scilit）记出版日 **2025-10-21**；
    ⚠️ **"出版社网页 2026 年上线"这一说法本次未核到** ⇒ 记为 **unverified**，不写入正文。

> 登记时补全：作者、年份、venue、DOI/arXiv 号、可访问链接。**只写记忆中的标题不算登记。**
> ⚠️ **相对 PagedAttention / vLLM 的前缀缓存机制**（本项目唯一未核验但被反复提及的一条）
> **仍为 unverified**；在本项目**不做前缀缓存相关主张**之前不急于补，
> 但一旦要写进相关工作就必须先核。

## 2. 官方文档与工具链

⚠️ **平台归属（2026-09-22 起）**：研究**主体平台 = MTT S4000（MUSA 栈）**，
昇腾为**次平台保留**。因此本节的昇腾侧条目属"次平台资料"，优先级低于 MTT 侧；
**两栈不可互换/不可比**。

| 名称 | 来源 | 用途 | 状态 |
| --- | --- | --- | --- |
| MTT / MUSA 相关（driver、MUSA SDK、torch_musa） | ⚠️ 待补版本与链接（**主体平台，优先级最高**） | 设备探测、驱动/SDK 口径、torch_musa 行为与已知限制 | unverified |
| CANN 文档 | _待补版本与链接_ | 算子、图模式（ACL Graph）、Profiling 工具口径（**次平台**） | unverified |
| torch_npu | _待补版本_ | PyTorch 适配层行为与已知限制（**次平台**） | unverified |
| vllm-ascend | _待补版本_ | 服务化推理基线与调度实现对照（**次平台**） | unverified |
| Ascend Profiling (msprof / MindStudio) | _待补_ | 采集硬件侧时间线与利用率（**次平台**） | unverified |

> ⚠️ **本机采集到的 MTT 侧工具版本口径**（`mthreads-gmi` 与 MUSA 版本）曾有**两处静默给错**，
> 成因与修法见 `README.md` §8（"采到的版本号必须锚定它自己的标签"）。
> 登记 MTT 侧资料时须沿用同一锚定纪律。

## 3. 基准、数据集与工作负载

| 名称 | 来源 | 用途 | 状态 |
| --- | --- | --- | --- |
| 本项目合成负载生成器 `benchmark/workloads.py` | 本仓库 | 四个扫描轴 + 七类点名原型 + **四种到达模型**（`closed` / `poisson` / `burst` / **`piecewise`**）；`load_ramp` 为 LVB 轨迹负载 | **verified**（本仓库代码） |
| 外部推理服务基准（如 ShareGPT 类真实轨迹） | _待补_ | ⚠️ **未采用**。本项目当前用**合成负载**，已知限制是**长度分布无重尾、会低估 P99**（登记在 `workloads.py` 末尾） | unverified |

> 工作负载来源需注明：序列长度分布、请求到达模式、是否有共享前缀。
> ⚠️ **SLOWeave 用的是"截断对数正态"长度分布 + 1000 请求/次**；本项目是均匀采样 +
> 每格 8 请求（P1S3 系列）。**两者不可直接比较**，做对照时必须对齐负载形状。

## 4. 社区实现与代码参考

| 项目 | 链接 | 参考点 | 状态 |
| --- | --- | --- | --- |
| sarathi-serve | [github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve) | chunked prefill 的**调度规则**可作 policy 参考；README 自述为研究原型（vLLM fork，功能不完整） | **verified**（代码公开；测试于 CUDA 12.3 / H100 / A100） |
| SLOWeave artifact | ⚠️ **未找到可访问仓库** | 论文声称随文提供 simulator / 成本参数 / 负载生成器 / raw runs / 聚合代码 | **unverified（位置未确认）** |
| DRLServe 代码 | ⚠️ 未查 | 论文未在本轮核验中给出代码入口 | **unverified** |
| DistServe 代码 | ⚠️ 未查（OSDI'24 官方页未列代码入口） | 属多设备架构，本项目不计划移植 | **unverified** |

> ⚠️ **第 13 步需要**：为"最接近方法"选定实现路径并记录**可行性 / 忠实度 / 额外限制 / 来源**。
> 直接使用作者实现 vs 按论文写 policy 参考版本，**必须写清是哪一种**，
> **不得**冒称"复现了完整系统"。判定项见 `RESEARCH_PLAN.md` §4.3。

## 5. 内部资料

| 名称 | 位置 | 说明 | 状态 |
| --- | --- | --- | --- |
| 研究规范 | `AGENTS.md`（正文 1–21 节 + 附录 A1–A7） | 研究口径与仓库操作契约（**附录 A 是旧契约，冲突以正文为准**） | verified |
| 工程与口径 | `README.md` | 负载维度 / 指标口径 / 标签与结论准入 / 数据完整性 / 阶段与环境 | verified |
| 结论溯源 | `EXPERIMENTS.md` | 登记表 + 已完成结论（C/D/E 系列）+ 负面结果与前置条件 | verified |
| 时间序日志 | `RESEARCH_LOG.md` | 含所有失败与否定结果的成因 | verified |
| 研究定位 | `RESEARCH_PLAN.md` | 问题收敛 / 主指标冻结 / 四层证据分级 / 第三臂重合风险 / 停止条件 | verified（第 01 步创建） |
| 执行计划 | `NEXT_SESSION_TASKS.md` | 16 步逐会话任务清单与依赖、完成记录 | verified |
| 成本画像 | `COST_PROFILE_S4000.md` | 真机前向成本画像与标定纪律（⚠️ 非 `MEASURED`） | verified |
| 排障手册 | `.workbuddy/memory/PLAYBOOK.md` + `YYYY-MM-DD.md` | 环境/排障/验证手法 + 逐日日志 | verified |

---

## 6. 对照表：已有机制 — 本项目现有能力 — 拟研究差异 — 验证方法

（2026-09-24 第 01 步产出。⚠️ **"拟研究差异"一列全部是待检验的命题，不是已确立的贡献**。）

| # | 已有机制 | 代表工作 | 本项目**现有能力** | **拟研究差异** | **验证方法** | 差异状态 |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | **固定 chunk 的分块预填充** + stall-free 调度 | Sarathi-Serve (OSDI'24) | ✅ 已实现（`chunk_policy`）；L2 真机已验；L3 已有 **C1/C2**（TTFT↔stall 权衡，40%~72%） | **无**（本项目把它当**静态基线**） | 第 06/10 步的静态候选（no_chunk / 512 / 1024 / 1536 / 2048） | 不主张 |
| M2 | **模拟器/运行时上的纯推理性能主张** | 全部 | ✅ 已有 harness + 遥测 + 四标签证据链 | **无** | — | 不主张 |
| M3 | **成本预测驱动的自适应 chunk**（deadline 约束、单调成本模型、对数搜索、无需按负载调参） | **SLOWeave** (arXiv 2609.07883) | ⚠️ **未实现**（第三臂待建）；⚠️ 成本画像**已探索但非正式证据** | **D1 负载轨迹形态**：SLOWeave 的时变到达只有 **bursty（两状态交替）**，**没有连续 load ramp / 恢复段**；本项目自变量即"低压→高压→恢复" | 第 06 步建轨迹负载 → 第 08 步标定容量 → 第 10 步静态响应面 → 第 14 步三臂正式比较（主指标 **`slo_goodput_tok_s`**，SLO 冻结） | ⬜ **待检验**（唯一可能站得住的差异之一） |
| M4 | **过载下的失败模式**（作者承认：持续过载时严格 TPOT 目标可能无安全 prefill 预算；不保证无饥饿） | SLOWeave（自述局限） | ✅ 已有 `slo_satisfaction_rate` / 失败·拒绝·未完成率三套账目 | **D2 可持续过载下的退化边界刻画**（把"仍可服务的高负载"与"超过物理容量"分开） | 第 09 步干预实验 + 第 15 步敏感性（含 SLO 敏感性、突变、恢复） | ⬜ **待检验** |
| M5 | **学习型自适应 chunk**（RL 学策略，MDP 建模） | DRLServe (ECAI 2025) | ❌ 无 | 不主张（路线不同：本项目是成本模型，不是学习型）；**作为"另一类自适应基线"或相关工作** | 第 13 步按需要选近邻对照 | 不主张 |
| M6 | **迭代级连续批处理 / 分页 KV / 前缀缓存** | vLLM / Orca / PagedAttention | ✅ 有 `cache_policy`（离线策略层）；⚠️ 真机上**从未开启**（`cache.enabled=false`） | 不主张（已有机制）；⚠️ **不得**据 `cached_tokens=0` 说"前缀缓存无收益" | — | 不主张 |
| M7 | **prefill/decode 物理分离** | DistServe / Splitwise (OSDI'24) | ❌ 无（单卡） | 不主张；仅用于**goodput 口径讨论**与相关工作 | — | 不主张（多设备架构，非本项目范围） |
| M8 | **阈值自适应控制器**（本项目自建：绝对阈值 + 滞回带，双向） | 本项目 | ✅ L1+L2 已验（双向动作落到 prefill）；❌ **L3 不成立**（符号跨重复翻转、c8 净负向） | 作为**第二臂**参与对照；同时是"固定阈值是否必然失效"这一**实验问题**的载体 | 第 04 步分层信号 → 第 09 步干预实验与阈值定稿 | ⬜ 待检验 |
| M9 | **单卡小显存场景下的适配** | （无直接对应） | ⚠️ 平台为**单卡 49152 MiB**，而 SLOWeave 主力是 8×A100/H100-80GB 量级 | **D3 适配性差异** | 第 11/14 步 | ❌ **不足以单独立论**（"换个硬件跑"不是贡献） |
| M10 | **准入控制 / 老化阈值**（过载时的外部手段） | SLOWeave 提及（未展开） | ❌ 无 | 不属于本研究范围，除非 D2 要求 | — | 不主张 |

**从本表推出的三条硬结论**：

1. **M1/M2/M6/M7 都不能作为贡献**——它们是已有机制，或本项目刻意不扩展的范围。
2. **唯一可能站得住的差异是 D1（负载轨迹形态）与 D2（过载边界）**；
   D3（换硬件）**单独不足以立论**。
3. ⇒ **第三臂在实现之前必须先验 D1/D2**（第 06/08/10 步的负载与静态响应面是前置）；
   若 D1/D2 都不成立，按 `RESEARCH_PLAN.md` 的停止条件 **S1** 处理（退为复现/适配定位）。

---

## 7. 登记纪律

1. 只登记**真正读过**且会引用的材料；不要为了凑数登记。
2. 每条必须能回答「它改变了本项目的哪个判断」——回答不出来的，删掉。
3. 引用前把状态改为 `verified`，并确认版本/年份与当前实现匹配；文档类资料必须标注版本号
   （如 MUSA SDK 3.1.0 / driver 2.7.0），否则半年后无法复现。
4. ⚠️ **两个字段必须分开记**：**会议年份**（如 DRLServe = ECAI 2025）与
   **出版社网页/著录的出版日期**（第三方著录记 2025-10-21）。混着写会在引用时给出错误的年份。
5. ⚠️ **"论文声称有 artifact" ≠ "artifact 可得"**。SLOWeave 即为此例：
   状态必须写实际核到的程度（本步 = `unverified（位置未确认）`），不得写成"代码可得"。
