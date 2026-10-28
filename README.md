# MTT S4000 LLM Inference Optimization

面向 **MTT S4000 / MUSA** 的 Qwen2.5-7B-Instruct 推理研究工程：请求感知调度、执行侧优化、成本预测与可复核实验。现有主要实测采用 **Transformers eager + 自建迭代级执行器**；vLLM-MUSA 是目标路线之一，不是这些结果的已验证运行引擎。

**发布整理：2026-10-28；已有实验与离线交付证据截止：2026-10-03。** 此次发布没有新增 GPU 实验。新成本模型、独立验证及三项诊断 pilot 尚待真机执行。

## 先看这些

- [已完成工作、实测效果与剩余任务](docs/PROJECT_STATUS.md)
- [发布范围、原始数据和版本身份说明](docs/PUBLICATION.md)
- [代码与文档导航](PROJECT_MAP.md)
- [实验登记与结论](EXPERIMENTS.md) / [证据资格](EVIDENCE_INDEX.md)
- [下一轮真机执行计划](handoff/optimization_20260929/REMOTE_EXECUTION_PLAN_20261003.txt)

## 当前结果

- **S01**：在已测 short_chat / 并发 8 / 64 请求条件下，TTFT p50 的逐轮配对改善中位数为 **4.31%**，5/5 轮同方向。TPOT 尚不足以判定改善，两臂 SLO 达标率均为 **2/64 = 3.125%**。结果见实验登记 §3.9，不外推至其他负载或引擎。
- **S04**：当前实现的局部负面诊断结果已归档，默认关闭；旧性能批次存在环境绑定缺口，不能作为正式量化收益证据。
- **新成本控制器**：执行形状、队列、KV 与整步计时契约已完成离线修复；**尚未证明真机性能收益**，旧 v7 不能直接用于新契约。

## 本地开始（无需 GPU）

使用 Python 3.10 或更新版本，在仓库根目录运行：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest tests/test_workload.py tests/test_metrics.py tests/test_controller_gate_mode.py tests/test_execution_shape.py tests/test_cost_execution_contract.py -q
python scripts/execution_shape_probe.py --manifest results/processed/execution_shape_manifest_20261003.json --dry-plan
python scripts/pilot_execution_plan.py
```

上述后两条只生成计划，不连接远程服务器、不执行模型推理。基础检查不覆盖全部历史测试、CPU 模型数值验收或 MUSA。完整测试入口为 `python scripts/run_tests.py --env` 和 `python scripts/run_tests.py --group all`；CPU 数值组需相应 PyTorch/Transformers 环境，远程编排需 Paramiko。硬件软件版本须在实际 MTT 主机核验，不在本机安装 MUSA 来代替真机。

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/ascend_opt/` | 策略、控制器、成本模型、共享执行形状和指标 |
| `benchmark/` | 真实/模拟后端、驱动、负载与实验编排 |
| `configs/` | 平台、实验和历史成本模型配置 |
| `scripts/` / `tests/` | 标定、分析、复现工具与回归测试 |
| `results/raw/` | 原始实验记录，保持只读 |
| `results/processed/` | 明确列入发布白名单的计划输入与派生证据快照 |
| `paper/` | 论文工作稿；不是已完成论文或已确认的全部结论 |
| `handoff/` | 交接材料；先看最新状态及真机计划 |

模型权重、远程凭据、完整本地助手日志和缓存不随本仓库发布。远程信息从用户私有配置读取；文档中的历史路径不能当作当前有效连接。

项目保留现有 **Proprietary** 许可声明；发布代码不等于授予新的开源许可。研究边界、指标定义与完整规范保留在下面原文及 [AGENTS.md](AGENTS.md)。

---

## 完整研究规范与历史说明

以下保留原研究 README。带日期的历史路线和阶段描述应结合上面的当前状态阅读。

> 🧭 **导航入口：先读 [`PROJECT_MAP.md`](PROJECT_MAP.md)** —— 文档路由表（"我要知道 X → 读 Y"）、
> 脚本/模块/配置索引、效力分层（哪些数字能引用）、环境速查、红线指针。
> 本文件是**口径与纪律**的权威之一，但不负责"去哪找 X"。

**研究主体平台是 MTT S4000**（摩尔线程 GPU，MUSA 软件栈）上的大语言模型推理性能优化研究工程。
昇腾 NPU 作为**次平台**保留支持——见 `AGENTS.md` 第 3 节（目标技术栈）与第 10 节（平台维度）。

> 术语：本仓库中 "device / GPU / 设备显存" 指目标加速器。仍写着 "NPU / 显存 / HBM" 的句子
> 读作 "目标加速器 / 其设备显存"，除非该句明确在讲昇腾——见 `AGENTS.md` 第 1 节的术语条款。
> 标识符不因这条而改名（改名会让老读端静默失效）。

**论文（暂定）**：《面向 MTT S4000 的请求感知型大语言模型推理部署优化研究》
（Request-Aware Large Language Model Inference Deployment Optimization on MTT S4000）

## 1. 研究问题

> **当工作负载特征变化时，自适应控制器能否做出比固定静态配置更好的推理配置决策？**

形式化地，系统状态为

```
s_t = [input_length, estimated_output_length, queue_length, concurrency,
       prefix_reuse_ratio, cache_hit_ratio, available_memory, NPU_utilization,
       TTFT, TPOT]
```

控制器选择动作

```
a_t = [batch_size, max_batched_tokens, chunk_size, prefix_cache_policy, graph_mode_policy]
```

优化目标是「在时延约束下最大化吞吐」：

```
maximize throughput
subject to  TTFT <= TTFT_SLO
            TPOT <= TPOT_SLO
            memory_usage <= available_NPU_memory
```

**没有任何优化被假定为普遍有益。** 所有收益主张必须有测量支撑。

### 本项目的贡献边界（重要）

贡献**不是**"我们启用了前缀缓存 / W8A8 / 更大的 batch"——这些机制已经存在。
贡献是一个**负载感知的协调机制**：决定这些已有机制在负载变化时应当如何配置。
因此对照组的正确形态是「静态配置 vs 自适应」，而不是「开/关某项优化」。

**收敛（2026-09-24，第 01 步）**——以下四条把上面的问题钉到可执行的形式，
完整版见 `RESEARCH_PLAN.md`：

1. **自变量 = 非平稳负载轨迹**（低压 → 高压 → 恢复，由 `piecewise` 的 λ(t) 表达），
   **不是**"某个并发档位下的稳态"。`closed` / `poisson` / `burst` 都表达不出这条轨迹（§3）。
2. **主指标冻结为 `slo_goodput_tok_s`**，并**必须**并列报告 `slo_satisfaction_rate`
   与 `slo_goodput_req_s`（§6 说明三者为何不可互换）。**SLO 值不因测试输赢而变**。
3. **对照至少三臂**：静态 / 阈值自适应 / 成本模型自适应。
4. ⚠️ **第三臂（成本模型控制器）是研究方案，不是已确立的贡献**：已有工作
   （SLOWeave 等）已经做过"成本预测 + deadline + 自适应 chunk"。
   ⇒ 它**必须先划定可检验差异**，或把定位退为**在同一平台/另一负载形态上复现与适配**。
   ⚠️ **硬件型号本身不构成创新点**——"在 MTT S4000 上做"不是贡献，"这类机制在
   非平稳负载下的行为与边界"才是。相关工作对照表见 `SOURCES.md` §6。

## 2. 研究范围

优化维度（每个维度对应 `configs/` 下的一个策略族）：

| 维度 | 配置目录 | 主要作用点 | 主关注指标 |
| --- | --- | --- | --- |
| 基线 | `configs/baseline/` | 无优化对照 | TTFT / TPOT / 吞吐 |
| 量化 | `configs/quantization/` | 权重与激活精度（W8A8 等） | 带宽占用、精度损失、时延 |
| 前缀缓存 | `configs/prefix_cache/` | 共享前缀的 KV 复用 | 命中率、TTFT |
| 分块预填充 | `configs/chunked_prefill/` | Prefill 与 Decode 混跑调度 | TTFT / P99 vs TPOT |
| 图模式 | `configs/graph_mode/` | ACL Graph / 静态图捕获 | 单步 launch 开销 |
| 自适应 | `configs/adaptive/` | 运行期在线调参 | SLO 达标率 / 达标吞吐 |

最小实验变体（规范第 5 节）：baseline；+量化；+图执行；+前缀缓存；+分块预填充；
+选定组合；+自适应控制器。**必须做消融，不得只报最终配置。**

模型范围：起步用 Qwen 系 **7B–8B Dense**。MoE / EP / EPLB 属后续扩展。

目标技术栈（**主体**）：`Linux (Ubuntu 22.04) → MTT driver (mtml / sgpu-dkms) → MUSA SDK →
torch_musa → PyTorch → vLLM → vLLM-MUSA`。
目标技术栈（**次平台，保留**）：`Linux → Ascend driver → CANN → torch_npu → PyTorch → vLLM →
vLLM-Ascend`。两条栈**不可互换、不可比**：一个平台上的 run 与另一个平台上的 run 是不同的
实验条件，数字不得并进同一个中位数。
MindIE 可作为参照实现或对照平台。

> **推理引擎尚未定案（2026-09-22）。** 上表是规范给的**目标**栈，不代表现在就按它部署。
> 当前路线是：先用 **Transformers** 做单请求验证（确认设备执行、输出正确性、显存占用），
> 跑通后接 `RealBackend`，再做连续批处理 / 前缀缓存 / 分块预填充实验，**最后**按这些功能
> 的可支持性选引擎。因此 **vLLM-MUSA 的部署细节现在都不适用**：
> - **权重转换**与**引擎块大小 64** 是**旧版 MTT-vLLM / MTTransformer** 栈的要求，
>   **不是所有 S4000 推理方式的通用要求**，不得机械照搬到 Transformers 路线。
> - 配置里的 `cache.block_size` 是**本处前缀缓存的模拟粒度**（`cache_policy` 消费），
>   与设备引擎参数无关，别混为一谈。
> - 引擎选定后，它自己的依赖与参数要求要**重新核对**，并按单变量纪律单独登记。

## 3. 负载维度

四个扫描轴（规范第 6 节，取值已钉进 `benchmark/workloads.py` 的常量）：

| 轴 | 取值 |
| --- | --- |
| 输入长度 | 128 / 512 / 2048 / 8192 / 16384 |
| 输出长度 | 128 / 512 |
| 并发 | 1 / 8 / 16 / 32 / 64 |
| 前缀复用比例 | 0% / 25% / 50% / 75% |

七类点名分布：短入长出、长入短出、高前缀复用、低前缀复用、低并发、高并发、混合。
这些原型**由轴函数生成**而非手写常量，避免"轴改了、原型没改"的隐性漂移。

**到达模型**（`WorkloadSpec.arrival`）：

| 模型 | 语义 | 能回答什么 |
| --- | --- | --- |
| `closed` | 固定客户端池，在途数不超过 `concurrency` | 稳态下的批效率（消除到达随机性） |
| `poisson` | 恒定速率、指数间隔 | 开放环稳态 |
| `burst` | 分段突发，段间确定性间隔 | 瞬时尖峰下的过载行为 |
| `piecewise` | **分段常量速率 λ(t)**，`rate_segments` 给出每段的 `(duration_s, qps)` | **负载水平的轨迹**（低压 → 高压 → 恢复）——只有它能表达 |

> `piecewise` 是研究问题「负载**变化**时控制器能否优于静态配置」的唯一自变量来源：
> closed 全程填满并发、poisson 是恒定速率、burst 只有尖峰，三者都没有"水平轨迹"。
> 段边界按**墙钟**对齐，因此"何时升压/降压"跨 run 是确定的。
> 总请求数由各段 `duration_s × qps` **决定**（不接受第二个真值，不一致即报错）。
> 段内分布 `rate_noise`：`jittered`（默认）让每段计数精确等于名义值且避开周期网格，
> `poisson` 只固定总数、各段实际计数会波动。
> ⚠️ 报告必须**逐段**读：`workload_fingerprint.arrival_segments` 同时给出名义与
> **实际**到达速率/计数，拿整 run 汇总值当"高压下的表现"是错的。
> ⚠️ `--dry-run` 下长时段 piecewise 负载会消耗大量步（空闲步按 `step_overhead_ms` 推进），
> 注意 `device.max_steps`。

```bash
# 查看四个扫描轴与展开后的待跑清单（每条负载标注它服务于哪个轴）
python benchmark/workloads.py --axes --matrix

# 列出注册表 / 查看某条负载的完整参数
python benchmark/workloads.py --list
python benchmark/workloads.py --show mixed
```

> 四个轴共用同一个锚点负载（`in=2048, out=128, c=8`），因此待跑清单会去重：
> 16 个轴取值 → **14 条负载**，锚点被标注为同时服务三个轴。
> 不去重会让同一条负载被当作多个实验条件，归因随之失真。

> 合成负载不代表所有真实生产环境。长度分布目前只有均匀采样、无重尾，
> 这会**低估 P99**——已知限制，登记在 `benchmark/workloads.py` 末尾。

## 4. 目录结构

```
ascend-llm-inference-optimization/
├── AGENTS.md            # 研究规范（正文 1–21 节）+ 操作契约（附录 A）——开工前必读
├── README.md            # 本文件
├── RESEARCH_LOG.md      # 时间序研究日志（append-only）
├── EXPERIMENTS.md       # 实验登记表（每条结论的唯一溯源入口）
├── SOURCES.md           # 外部资料与引用登记
├── configs/             # 按优化维度分组的实验配置
├── src/ascend_opt/      # 可复用库（纯逻辑层，禁止触达硬件 SDK）
├── benchmark/           # 执行层：跑基准、采指标、扫参数
├── scripts/             # 环境检查与一键执行脚本
├── results/             # raw 只读 / processed 聚合 / figures 图
├── tests/               # 单元与冒烟测试
└── paper/               # 论文素材：outline / methodology / experiments / results
```

`src/ascend_opt/` 的模块划分（对应规范第 12 节）：

| 规范要求 | 实现 | 职责 |
| --- | --- | --- |
| WorkloadMonitor | `monitor.py` | 把观测归一成状态向量 `s_t`；**只观测，不决策** |
| AdaptiveBatchController | `controller.py` | 批参数（`max_batched_tokens`、并发上限） |
| PrefixCachePolicy | `cache_policy.py` | 前缀留存 / 淘汰 |
| ChunkController | `controller.py` | Prefill 分块大小 |
| AdaptiveController | `controller.py` | 协调以上三者（消费 `s_t`） |
| — | `scheduler.py` / `chunk_policy.py` | 调度与分块策略实现（`ChunkLimits` 承载运行期上限） |
| — | `metrics.py` / `telemetry.py` | 指标口径 / 采集落盘 |
| — | `provenance.py` | 数据标签与可复现性元数据 |

**硬件边界**：`workload / metrics / monitor / provenance / scheduler / cache_policy /
chunk_policy / controller` 全部为纯逻辑层，禁止调用 `torch_npu` / ACL / `vllm-ascend`
运行时。硬件交互只允许出现在 `benchmark/` 与 `scripts/`。这条是整套离线验证
（`--dry-run`）能成立的前提。

### 自适应控制器的闭环契约

控制器按规范第 2 节消费状态向量 `s_t = f(观测)`、产出动作
`a_t = [batch_size, max_batched_tokens, chunk_size, ...]`。四条契约都对应
一个**不报错、也不让账目不自洽**的失效，因此必须写在文档里：

1. **双向可逆，不是棘轮。** 调小是防御（TPOT 超目标 / 显存临界），
   调大是恢复（压力消退）或为 prefill 加码（TTFT 超目标且 TPOT 有余量）。
   恢复的目标是**配置初值**，不是上界——恢复的语义是撤销防御，不是在安全区里探索。
   只有收缩没有恢复的控制器在负载回落后会把系统永久按在过保守的配置上。
   → `n_recovered` / `bottleneck="recover"` 可核验。
2. **建议值必须真的生效。** 运行期上限经 `chunk_policy.ChunkLimits` 显式传入策略，
   **不再由 frozen 的 `config.chunk_size` 封顶**。过去策略内部会把它压回原值，
   表现为"决策日志显示已调大、实际额度纹丝不动"。
   → `telemetry.jsonl` 同时记录 `*_requested`（建议）与 `*`（生效），
   `meta.json` 的 `controller_effect` 汇总差额。
3. **宣告的运行点必须可行。** `chunk_size <= max_batched_tokens`；不满足时控制器
   自己修复并计入 `n_infeasible_repaired`。宣告一个兑现不了的运行点，
   会让下游静默夹一次——那正好把第 2 条那个检查变成噪声。
4. **信号要么可信、要么显式为 `None`。** 缓存命中率与可用显存是决策输入，
   但离线路径上没有显存读数、缓存也可能是关的。未测信号**不参与决策**
   （既不触发收缩也不否决放大），却在 `signals` / `_unavailable_signals` 里显形。
   阈值默认 `None`（不启用），要启用必须在配置里显式写。
   注意：缓存**关闭**时命中率是 `None`（信号不存在），
   缓存开着但没命中是 `0.0`（测得没命中）——两者对策相反，不可合并。

三类必须固定住的行为在 `tests/test_controller_loop.py`：压力上升（收缩）、
压力消退（恢复）、负载切换（判据翻面）。

**控制信号的口径（P6，2026-09-24）**：控制器读的 `tpot_p90_ms` 来自**相邻两次 token
之间的间隔**，而某一步做了 prefill 时，落在这两步之间的间隔**直接包含该 prefill 的耗时**
⇒ "prefill 挤占"会被读成"decode 变慢"。现在把两者**分开记录**：

| 配置 | 含义 |
| --- | --- |
| `controller.signal_mode` | `legacy`（**默认**，只看 `tpot_p90`）/ `layered`（额外记录分层信号） |
| `controller.signal_window_expiry_ms` | 观测窗口的**过期上限**；**默认留空 = 不过期** |

* 分层量：`interval_had_prefill`（该间隔里**任意一步**做过 prefill）、
  `interval_decode_ms`（该区间里**纯 decode 步**的耗时之和），与用户感知的
  `intervals` **逐项对齐**；另有 `step_trace` 供**独立重算**。
* ⚠️ **两种模式在决策上等价**——本步只做**记录**，不改判据。
  "纯 decode 特征有效"不等于"应把所有决策交给它"，那是第 09 步的问题。
* ⚠️ **用户真实经历的停顿必须继续进入 ITL / TPOT / stall / SLO**，
  不得用"纯 decode"美化对外指标。
* ⚠️ 未测一律 `None`：`legacy` 下 `n_decode_samples` 是 **`None`（没有这个信号）**，
  而 `layered` 下空窗口是 **`0`（采集了但此刻没样本）**——两者对策不同。
* 细节与判据见 `EXPERIMENTS.md` §4.3.6。

## 5. 快速开始

```bash
# 0) 环境自查（驱动 / CANN / torch_npu / 设备型号与显存）
bash scripts/inspect_env.sh

# 1) 冒烟：不依赖真实硬件，验证 harness 与策略链路是否通
python -m benchmark.run_benchmark --config configs/baseline/default.yaml --dry-run

# 2) 单元测试——**两个入口，别混**（都必须用下面的 managed 解释器，见 §9）
#    先体检：打印解释器 / torch 可用性 / 该用哪一组
"C:/Users/13944/.workbuddy/binaries/python/versions/3.13.12/python.exe" scripts/run_tests.py --env

# 2a) 纯逻辑入口：**任意**解释器都要能跑；0 failure / 0 error；跳过项单独列出且不计入通过
"C:/Users/13944/.workbuddy/binaries/python/versions/3.13.12/python.exe" scripts/run_tests.py --group logic

# 2b) 数值入口：含 CPU 微型模型（需要 torch + transformers）；缺依赖直接退出码 2，不"跳过并通过"
"C:/Users/13944/.workbuddy/binaries/python/versions/3.13.12/python.exe" scripts/run_tests.py --group numeric

# 2c) 一次跑完（回归基线）。⚠️ 用错解释器会得到 8 errors / 25 skipped，那是**选错解释器**：
#     envs/default/Scripts/python.exe 只装了 paramiko（给 scripts/remote_run.py 用），没有 torch
"C:/Users/13944/.workbuddy/binaries/python/versions/3.13.12/python.exe" scripts/run_tests.py --group all

# 3) 基线实测（需要 MTT S4000 硬件）
bash scripts/run_baseline.sh

# 4) 聚合指标 -> results/processed（默认只收 MEASURED 数据）
python benchmark/collect_metrics.py --raw results/raw --out results/processed

# 5) 参数扫描（探索用，圈区间；不产生结论）
python benchmark/sweep.py --sweep configs/chunked_prefill/sweep.yaml --limit 5

# 6) 导出前置校验（原始明细 / 环境指纹 / 有效重复 / 实验登记；缺一即退出码 4）
python benchmark/preflight.py

# 7) 导出交付包（含汇总表、配置快照、环境指纹、图）。第 6 步会自动跑一遍；
#    无可交付的 MEASURED 数据时退出码 3，前置校验不过时退出码 4。
bash scripts/export_results.sh --tag v0.1

# 想复用同一个 run_id 重跑（覆盖已有产物）时必须显式声明——默认拒绝，见第 8 节
python -m benchmark.run_benchmark --config configs/baseline/default.yaml --dry-run --force
```

## 6. 指标定义

统一以下述定义为唯一口径，避免各实验各写一套：

- **TTFT**（Time To First Token）：从请求**到达**到首 token 返回，单位 ms。
  TTFT 是"等待 + 计算"的合成量，必须同时给出三段分解，否则无法判断该加容量还是该调分块：

  | 段 | 定义 | 成因 |
  | --- | --- | --- |
  | `admission_ms` | 到达 → 被服务端受理 | 链路 / 连接获取 / 模拟器步粒度 |
  | `queue_ms` | 受理 → 首次被调度 | **并发容量不足**（长 prefill 独占算力等） |
  | `prefill_exec_ms` | 首次被调度 → 首 token | 算力与分块策略 |

  要求：`admission + queue + prefill = TTFT`（逐请求精确成立；汇总里各段独立取分位，
  故 P50 之和不严格等于 TTFT 的 P50——**分位数不可加**）。

  ⚠️ **`arrival_ms` 的参考点（P5，2026-09-24 修）**：TTFT 的起点是**放行时刻**，
  **不是**"处理它的那一步的开始时刻"。两者在长步下差别巨大——关闭分块时一步可达数百毫秒，
  落在这一步内部的请求若用步开始时刻作参考点，它的真实等待会被**整段抹掉**。

  - 开放环：`arrival_ms` = **日程时刻**（`scheduled_arrival_ms`）；
  - 闭环：`arrival_ms` = 客户端腾出容量的时刻（此时 `scheduled ≡ arrival ≡ 步开始`）。
  - `arrival` 与 `enqueue` 是**两个时刻**：前者是放行，后者是被受理。
    `arrival → enqueue` 是 `admission_ms`（链路 / 连接获取 / **步粒度**），
    `enqueue → first_schedule` 才是 `queue_ms`（容量不足）。**合并它们会让"加容量"与"修链路"失去依据。**
  - **判据**：开放环下一整批的 `admission_ms` **恒为 0 ⇒ 旧口径**（该缺陷已修）；
    新口径下必有 `admission_ms > 0` 的请求。
  - ⚠️ **不得用"账目自洽"证明它**：`done_ms`、`ttft_ms`、`e2e_ms` 若共用同一个被覆盖的
    参考点，`accounting_error_ms` 照样是 0。详见 `EXPERIMENTS.md` §4.3.5。
- **TPOT / ITL**（Time Per Output Token）：decode 阶段每 token 平均间隔，单位 ms/token。
- **TPOT stall**：**单请求自身 token 间隔的 P99**，单位 ms。
  这一项必须单独存在：TPOT 是每请求的均值，一次数秒级的停顿会被 `gen_len` 除开、
  在均值里几乎消失，而用户感受到的正是那次停顿。忽略它会把"分块预填充修好了尾延迟"
  判成"测不出结论"。
- **E2E**：请求端到端时延 = TTFT + TPOT × (gen_len − 1)。
- **Throughput**：单位时间内完成的输出 token 数，单位 **tok/s**，需同时标注并发度。
- **达标口径（三个数，不可互相替代）**——刻意**不**用 `goodput` 这一个名字：
  它同时被读成"达标比例"和"达标吞吐"，而两者是不同的数，混名之后没人知道那列在说什么。

  | 字段 | 定义 |
  | --- | --- |
  | `slo_satisfaction_rate` | 达标请求数 / **全部提交**请求数 |
  | `slo_goodput_req_s` | 达标请求数 / 测量秒数 |
  | `slo_goodput_tok_s` | 达标请求的输出 token / 测量秒数 |

  **主指标冻结为 `slo_goodput_tok_s`**（2026-09-24，第 01 步；定义见 `RESEARCH_PLAN.md` §3）。
  另两列**必须同时报告**，不得只报主指标——达标比例回答"多少请求被服务好了"，
  请求达标吞吐回答"服务速度"，两者在长/短输出混合负载下会给出不同的排名。
  ⚠️ **SLO 阈值（`ttft_ms` / `tpot_ms`）在正式比较中冻结**：不得因为某臂结果不好而放宽；
  变更须按 AGENTS 附录 A7 在 `RESEARCH_LOG.md` 声明口径变更并标注影响范围，
  且旧批次必须按同一 `--candidates` 重算后才能与新批次并表（见下表后的重算说明）。

  分母是**全部提交**而不是成功数：用成功数做分母会把失败与未完成从账上抹掉，
  于是一批大量超时/失败的 run 反而显示出很高的"达标率"。
  未配置 SLO 时这三个量写 `None`（不可计算），**不写 0**。

- **判据本身必须是可达的**（否则这三个数不携带信息）。`SLO.satisfied_by()` 是**逐请求**
  判定（每个请求单独比 `m.tpot_ms > 阈值`），不是比汇总分位；同时配了 `ttft_ms` 与
  `tpot_ms` 时是**与**关系。所以阈值必须落在**实测逐请求区间内部**，否则整列退化成
  恒 0 或恒 1，在表里与"所有配置都很差"长得一样。

  | 硬件/模型 | `ttft_ms` | `tpot_ms` | 实测逐请求 TPOT 区间 | 定于 |
  | --- | --- | --- | --- | --- |
  | MTT S4000 / Qwen2.5-7B fp16 | 2000 | **100** | [61.7, 168.0] ms | 2026-09-24（原 50，低于下界） |

  标定与把关：`python scripts/calibrate_slo.py --raw results/raw --prefix P1S3B,P2,P1S3D`。
  ⚠️ 阈值改过之后，**旧批的达标率列与新批不是同一个判据**，不得并表；
  用同一 `--candidates` 重算旧批即可对齐（实测：`P1S3B` 按 100 重算后与 `P1S3D`
  **逐格相同**，全格同为 58/96 = 0.604）。
  **换硬件、换模型、换负载都要重采**，否则这份区间会失效
  （生成器里的 `_SLO_TPOT_REACHABLE_MS` 是同一份常量的静态兜底）。
  注意 `--dry-run` **验不了这一项**：一阶设备模型的 TPOT 比真机低约一个数量级，
  任何阈值都会读成 1.000。

- **控制门限必须与工作点分得开**（这是**决策口径**，与上面那条 SLO **报表口径**是两件事，
  但失效方式同源：判据本身不携带信息）。控制器判据是**绝对阈值 + 滞回带**——
  收缩要 `tpot_p90 > target·(1+h)`，放大要 `tpot_p90 < target·(1−h)`
  **且** `ttft_p90 > target_ttft·(1+h)`。门限若落在某负载档位的工作点**分布内部**，
  该格的决策会随重复之间的小扰动翻转 ⇒ **同一份配置的读数跨重复不可复现**。
  它不报错、不停止，**只是随机地动**（实测 `adaptive_c4` 三次重复 TTFT
  481.4 / 481.6 / 652.1 ms，**+43%**）。

  | 硬件/模型 | `target_tpot_ms` | `hysteresis` | 放大 / 收缩门限 | 定于 |
  | --- | --- | --- | --- | --- |
  | MTT S4000 / Qwen2.5-7B fp16 | **120** | **0.08** | 110.4 / 129.6 ms | 2026-09-24（原 110 / 0.1，门限 99.0 压在 c4 工作点 98.7 上） |

  标定：`python scripts/calibrate_controller_gate.py --prefix P1S3B,P2,P1S3D`。
  ⚠️ **格的身份是 `(批次, 格名)`**（2026-09-24 晚修正）：`--prefix` 给多批时**不再**把不同批的
  同一格并成一组。此前只按格名分组 ⇒ P1S3B 与 P2 的首轮被并成"同一条件的两次重复"
  （跨重复表里 c4 出现**两行 r0**，静态分类还不同而 `chunk 范围` 与瓶颈串**逐字相同**）
  ⇒ 看起来像"同条件重复却判定不同"，实际是**两个条件被混成一组**（未登记的混淆）。
  要问"某批的闭环行为"仍需 `--prefix` 只给那一批。
  它读 `controller_trace` 里控制器**实际读到**的信号（**不是**报表的
  `summary.tpot.p50`——c4 那格两者差 11.5 ms）。
  **判定口径（2026-09-24 晚修正）**：工具给每格一个四态 `verdict`
  （**必放大 / 必不放大 / 样本跨越门限 / 不可达**）＋"离门限最近的**点**相距多少 ms"；
  并**另设**「跨重复对照」一节比较**同一条件下不同重复**的判定是否一致
  （`重复间一致 / 重复间分歧 / 单一重复`）。⇒ **要一起读三个量**：`verdict` 回答
  "这条合并序列上门限决断吗"、最近点回答"离翻转多远"（110/0.08 是"决断但只有 0.6 ms"，
  等于没分开）、「重复间」回答"换个重复还这样吗"。
  ⚠️⚠️ **「样本跨越门限」≠「跨重复不可复现」**：它只看**一条合并后的序列**，而同一次运行内
  负载本来就可能在工作点之间移动，两种结果并存完全正常。旧名「抛硬币」把前者直接说成后者，
  是一次**未被证实的跳步**。
  ⚠️⚠️ **本工具不能充当"可复现性"的阻断门**（用户 2026-09-24 裁定）：它是**静态口径**
  （不含冷却、也不含"动作改变后续信号"这一反馈），能给的是"哪一格要盯着看"。
  若将来要接成**会失败**的门，判据必须是「**重复间一致**」，**不是** `verdict != 样本跨越门限`。
  ⚠️ 工具内置**口径核对**（每次运行自检，与 `--targets` 无关）：用每个 run **自己**的
  阈值（`meta.config.controller`）核对静态分类与轨迹里记下的分支计数是否逐条对上
  （`n_shrink==#tpot`、`n_grow+门槛拒==#ttft`；后者是因为**被 headroom 门槛拒时返回的
  `bottleneck` 仍是 `ttft`**），漂移即**退出码 6**；读不到阈值记「不可判定」而**不是**通过。
  ⚠️ 原文写的"**权威判据是 `max(headroom) − h`**"**已作废**：那个量出自一个把
  **收缩点**也算进去的输出列 ⇒ `min < h < max` 恒成立 ⇒"跨过门槛"**恒报**，
  且与它自己的「带内」列**自相矛盾**（详见 `EXPERIMENTS.md` §4.3.4）。
  ⚠️ 生成器的 `_assert_gate_separation` 只比"**中位数**到门限的距离"，是生成期的
  **静态下限**，看不见分布宽度 ⇒ **通过它 ≠ 门限已充分分离**。两者**不可能等价**
  （实测四个候选分歧方向相反：守卫拦下工具判决断的 110/0.08，却放行工具已看到
  8 号格最近点只有 1.1 ms 的 120/0.1——对照表见 §4.3.4），**不得**用它宣称"已充分分离"。
  ⚠️ 与 SLO 不同：**改门限会改变调度行为 ⇒ 必须重跑**；改 `run.slo` 只需重算。
  ⚠️ **"机制上不再可能"≠"可复现性已建立"**：`p1s3d` 确实让旧的那个拒绝
  （工作点 97.8–99.2 跨过 99.0）在构造上不可能，但 `P2` 用**旧门限**同样 3/3 收敛
  ⇒ n=3 分不开"新门限消除了它"与"没抽中"。**别把"余量抬了一个量级"读成"问题已解决"**
  （`EXPERIMENTS.md` §3.3 **E1/E8**）。
  ⚠️ **结构性障碍尚未解除**：工作点跨 61.8 → 571.7 ms（**9.3×**）而滞回带宽度是
  固定绝对值 `2h·target` ≈ 19 ms ⇒ 绝对阈值 + 单一 `(target, h)` 无法与多档工作点
  分开。⚠️ 理由要说**负载扫描**（负载连续变化会扫过整条区间 ⇒ 必然有档位落在带内），
  **不要**再说"c4 的区间跨过门限"（那是上面那个被污染列的产物），
  详见 `EXPERIMENTS.md` §4.3.4。**换负载/换硬件必须重采。**

- **结局账目**（分母同为全部提交数）：`failure_rate` / `rejection_rate` / `unfinished_rate`，
  另有原始计数 `n_failed` / `n_rejected` / `n_unfinished`。
  它们不是"不好的吞吐"，而是"服务没把活干完"，成因与对策都不同。
  `unfinished_rate > 0` 时必须与分位数一起看：时延分布只统计成功请求，
  未完成的那批（往往正是最慢的）被剔除在外，**尾延迟会被低估**。
- **Cache Hit Rate**：前缀缓存复用 token 数 / 总 prompt token 数。
- **设备显存**：单位 **MiB 或 GiB**（昇腾侧习惯称"NPU 显存 / HBM"，同一口径）。

口径纪律：

- 一律报告 **mean / median / P90 / P95 / P99**，不把 single-run 平均值当结论。
  `median` 与 `P50` 数值恒等，但两者都输出——论文正文用 median、工程表用 P50，
  并列能让口径自证。
- 分位数一律**附带样本量 n**：十几个样本的 P99 本质上就是最大值，只给分位不给 n，
  读者无法判断该不该信。
- **不可测的值保持 `None`，禁止填 0**。0 是"测出来是 0"，缺失是"没测"，语义相反；
  把缺失写成 0 会让"没发生"看起来像"极快"，而且读起来完全正常，因此不会被追查。
- **单位不混用**：吞吐只写 tok/s，显存只写 MiB/GiB。
- `summary.csv` 中按 `ttft_* / tpot_* / stall_* / e2e_* / admission_* / queue_* / prefill_*`
  前缀成组，便于直接绘图。
- 账目自查：`E2E` 必须等于「完成时刻 − 到达时刻」，偏差记为 `accounting_error_p99_ms`。
  参考点是**到达时刻**——闭环负载下请求在获得客户端容量时才产生。

### 动态负载的分阶段报告口径

`piecewise` 负载（`load_ramp`）的自变量是**负载轨迹**，整 run 一个汇总值回答不了它。
`src/ascend_opt/stage_report.py`（纯逻辑）+ `scripts/report_stages.py`（读产物、出报告）
把报告拆成**两套口径**，并规定**它们不能互相替代**：

| 口径 | 归属依据 | 回答什么 |
| --- | --- | --- |
| **cohort** | 请求的**到达/放行时刻**落在哪一段 | "这一段进来的请求，后来表现如何" |
| **window** | 事件**发生**在哪个墙钟窗口 | "这段时间系统在处理什么、积压多少" |

⚠️ **跨阶段请求同时出现在两处**（它是 cohort A 的成员，也在 window B 里完成）——
那不是重复计数，是**两个问题的两个答案**；混成一个口径就会既漏又重。

**三条预先定死的规则**（"预先定义"是要求，不能事后按结果解释）：

1. **达标判定时机**：请求**完成时**才判定；未完成 / 失败 / 被拒**一律不算达标**。
   实现上**直接调用** `SLO.satisfied_by`——**不在报告层另写一套阈值判断**，
   且 SLO 取自**该 run 自己**的 `meta.config.run.slo`（不是当前默认值，
   否则改过阈值之后旧批会被用新判据重算）。
2. **分母**：达标吞吐统一用**测量分母**（分窗口 = 窗口墙钟长度）。⚠️ 不用"最后一个
   完成时刻"，那会把排空尾段从分母里抹掉、吞吐虚高。
3. **恢复**：阈值与持续时间**必须显式给定**；没观察到 ⇒ 记
   `not_recovered` / `censored`，**用时记 `None`，不填 0**——
   "0 ms 恢复"是一个好得不可能的读数。

**另外两条**：

* **排空尾段单独成段**（`drain`）：它是"最后一批**完成**所花的时间"，**不是**负载的一段。
  并进最后一段会同时污染两件事——恢复段达标率被排空的慢请求拉低，而排空时长被算成"恢复耗时"。
* **低样本报 `n` 且不给 P95/P99**（`--min-n`，默认 20）：十几个样本的 P99 本质上就是最大值，
  报出来会诱导人相信一个没有统计意义的数字。
* **对账**：cohort 提交数之和必须等于请求总数、窗口完成数之和必须等于成功数；
  对不上就把差异**列出来**（退出码 1），不静默。

```bash
python scripts/report_stages.py --run-dir results/raw/<run_id>
python scripts/report_stages.py --run-dir <dir> --recovery-threshold 1 --recovery-hold-ms 1000
```

⚠️ 该 run 若**没有** `arrival_segments`（非 `piecewise` 负载），脚本会退化成"整 run 一段"
并**明确声明"分阶段结论不适用"**——不凭空造阶段。

## 7. 数据标签与结论准入

规范第 10 节把「绝不编造实验结果」定为绝对要求。本仓库把它实现为**类型**而非纪律：
每个 run 落盘时在 `meta.json` 与 `summary` 上强制盖章，聚合脚本按标签过滤。

| 标签 | 含义 | 可否进结论 |
| --- | --- | --- |
| `MEASURED` | 在权威执行主机上实测（本项目为远程 MTT S4000；次平台为昇腾主机） | ✅ 唯一可进入结论的标签 |
| `NOT_MEASURED` | 本应能测到但本次没测 / 硬件可用性未知 | ❌ |
| `HARDWARE_REQUIRED` | 依赖目标硬件，当前环境不具备 | ❌ |
| `TEST_DATA` | 合成数据（单元测试、`--dry-run` 链路验证） | ❌ 永不得进论文表格 |

关键判定：**`--dry-run` 产出一律是 `TEST_DATA`，即使机器上真有 NPU 也一样**
——数字是合成的，与机器无关。

聚合默认只收 `MEASURED`；需要核对链路口径时用 `--include-non-measured`
（该开关只改变**收哪些行**，不改变列结构——下面这些列在两种模式下都在，
否则"这张表里哪一行是模拟数据"就得靠外部记忆）。
汇总表带 `data_label`、`label_source`、`execution_environment`、
`hardware_platform`、`backend`、`precision`、`quantization` 七列身世信息。
`label_source` 有两条值：`explicit`（产物自带标签）与 `legacy_inferred`（旧产物缺标签，
只能按 `dry_run` 回推）。回推有一条硬规矩——**只允许降级，永不升格**：
`dry_run=True` 的旧产物判 `TEST_DATA`（数字必然是合成的，这是可靠推断），
其余旧产物一律判 `NOT_MEASURED`。理由：`dry_run=False` 推不出"实测"，
无 NPU 机器上跑真机路径同样是 `False`，而旧产物里又没有执行环境声明。
这条规矩堵的是一种很隐蔽的失效——它不抛错、不告警，只是把"没打标签"静默读成
"测过了"，最后产出一张看起来完全正常的汇总表。凡 `label_source` 不是 `explicit`
的 run，进结论前必须重新生成产物。
`execution_environment` 记的是**推理实际跑在哪台机器上**（`remote_ascend` /
`remote_mthreads` / `local` / `unknown`；一组重复若混了不同环境则写 `mixed`）。
它必须显形的原因：本机 run 与远程 run 都可能标 `MEASURED`，光看标签分不出来，
而两者的可比性完全不同。
`hardware_platform` 记的是**面向哪个平台**（`ascend` / `mthreads` / `unknown`），
与执行环境**分开**且**不互相推导**：平台是"硬件是什么"（事实，可被指纹校验），
执行环境是"推理在哪跑"（声明，决定标签）。声明 `ascend` 不代表推理就跑在远程主机上
（本机也可以是昇腾整机）。两个字段都会进 meta 并被下游按平台筛选，所以
**互相矛盾的组合在加载配置时直接报错**（如 `remote_ascend` + `mthreads`），
并给出可照抄的修法——放过去就是永久留下一份自相矛盾的溯源记录，
而下游只会读到其中一半，事后没人能发现。
`remote_mthreads` 是独立取值而不是复用 `remote_ascend`：复用会让 meta 里出现一句
与事实相反的话（S4000 的数据自称跑在昇腾上），比"没有这个声明"更糟。

这三个环境维度**不只是被记录，还参与分组**：`hardware_platform` 与 `backend` 进
重复分组键（见 `benchmark/collect_metrics.py::group_key`），平台还进绘图的
分组键。原因是一个已复现的静默失效——同负载、同配置下，两个平台的实验被并成
一组，中位数落在两者之间：一份取值 100、一份取值 200，图上出现一根 **150** 的柱，
一个谁都没测过的数，而且全程不报错。跨平台/跨后端合并的方向被刻意封死：
**宁可多拆一组**（多拆会落到 `insufficient_repeats` 而被看见），
**也不要并出一根假柱子**。

> **已决议（2026-09-22）：研究主体平台为 MTT S4000。** 规范末节已由
> `Remote Ascend Execution Environment` 修订为 `Remote Execution Environment`，
> 权威推理环境写作**远程 MTT S4000 主机**；`AGENTS.md` 第 3 节同步给出 MUSA 栈（主体）
> 与昇腾栈（次平台，保留）。代码侧本就平台无关（标签判定不认平台名，对每个远程取值
> 逐格一致），所以本轮规范修订**不改变任何判定结果**——它把规范口径与代码口径对齐，
> 而不是引入新行为。

进入 README / `paper/` 的任何数字，还必须同时满足：

1. 能对应到 `EXPERIMENTS.md` 中的一条登记记录；
2. 数据标签为 `MEASURED`；
3. **来自权威执行主机上真实执行过的命令**（规范末节：本机与合成结果不得表述为该平台实测；
   本项目权威主机是远程 MTT S4000 主机）；
4. 有 ≥3 次重复运行，且报告中给出离散度（稳健 CV = 1.4826·MAD / median；
   **注意项目里有两个都叫"稳健 CV"的量，别混用**：`summary.stability.*_robust_cv`
   是**一个 run 内跨请求**的离散度，`repeats.csv.*_robust_cv` 是**同一配置跨 N 次
   重复**的离散度。它们回答不同问题、阈值也不同，引用时必须写明是哪一种）；
5. 单变量变化，其余条件与对照 run 完全一致；
6. 负结果同样登记——"这个优化没用"与"这个优化有用"同等重要；
7. 导出时通过前置校验：原始明细、环境指纹、有效重复（≥3）、`EXPERIMENTS.md`
   登记四项齐全。这一条不是形式——它把上面 1–4 条从"纪律"变成"会被挡住"，
   因为这几项缺失时汇总表照样生成、图照样画出来，不报任何错。

**缺失必须显形**：取不到的观测值保持 `None`、元数据写缺失标记，**禁止填 0**。
0 会被读成"测出来是 0"，缺失是"没测"，两者语义相反。
例外：`0` 是真实测量值（如 `warmup_count=0`）时必须保留。

## 8. 结果组织约定

```
results/raw/<run_id>/
  meta.json         # 配置快照、代码版本、环境指纹、reproducibility（21 字段 + 执行环境）、
                    # data_label、controller_trace、system_states、controller_effect
  requests.jsonl    # 每请求一行：req_id, ttft, tpot, e2e, cached_tokens...
  telemetry.jsonl   # 时间序采样：队列深度、运行中批大小、chunk_size(生效)/chunk_size_requested(建议)...
  config.snapshot.yaml
results/processed/<run_id>.csv    # 聚合后的每请求指标
results/processed/summary.csv     # 跨 run 汇总表（喂给绘图的唯一数据源）
results/processed/repeats.csv     # 跨重复聚合（含离散度）
results/figures/                  # 由脚本生成，禁止手工截图
```

`meta.json` 里与控制器有关的三个顶层键（缺任何一个，"控制器是否真的在起作用"
就无从核验）：

- `system_states`：决策步上采集的 `s_t` 序列，与 `controller_trace` 逐步对齐；
- `controller_trace`：每次决策的 `reason` / `bottleneck` / `signals` /
  `_unavailable_signals` / `state`；
- `controller_effect`：`chunk_requested` 与 `chunk_effective` 的区间、
  恢复次数、显存收缩次数、以及"建议未兑现"的步数；外加
  `n_cycles` / `n_skipped_cooldown` / `decision_interval_steps`
  （用于核对**冷却到底有没有在工作**，见下）。

**冷却的单位是「观测周期」，不是推理步。** 配置项叫 `cooldown_cycles`
（一次可判定的 `observe()` 调用 = 一个周期；控制器关闭或无有效样本的调用
不消耗周期）。决策节拍另由 `control_interval_steps` 决定，两者单位不同。

这一条值得单独写，因为它是本轮修掉的一个**静默参数**：该字段原名
`cooldown_steps`、按推理步比较，而决策每 `control_interval_steps`（默认 8）
步才发生一次——于是"距上次调整 < 3 步"永远为假，冷却从未生效，扫这个参数
不会有任何效果，但它不会报错。落到产物里的可核对事实是
`decision_interval_steps`（应恰好等于 `control_interval_steps`）与
`n_skipped_cooldown`（恒为 0 就说明冷却窗口比决策节拍还短，参数是死的）。

`run_id` 命名：`YYYYMMDD-HHMMSS-<workload>-<config_stem>`。
`results/raw/**` 一旦写入即只读，不得修改、覆盖、重命名。

### 数据完整性的程序化约束（评审 item 3 + Part 1）

这些约束的共同点是**缺了都不会报错**——表照样生成、图照样画出来、导出包照样写着
「完成」。所以必须由代码主动拦，而不是靠人记得。

1. **run_id 不被覆盖。** `run_benchmark.py` 写入前检查目标目录里是否已有产物
   （`meta.json` / `telemetry.jsonl` / `requests.jsonl` / `config.snapshot.yaml`），
   有则抛 `RunDirConflict`、退出码 4。判据是**产物**而不是
   "目录存在"——`mkdir(parents=True)` 会提前建目录，一次中断留下的空目录不该把
   这个 run_id 永久堵死。CLI 还会在开跑前把**全部重复**的目标目录查一遍，
   避免留下"半批"产物被下游当成完整的一次重复。

   `--force` **不是万能钥匙**，它只放行**可丢弃的调试产物**，必须同时满足三条：
   目标不在 `results/raw` 之下、本次是 `--dry-run`、已有产物的 `data_label` 是
   `TEST_DATA`。三条缺一即拒绝，并把**具体是哪条不满足**写进报错。
   已有实测结果一律换新 `run_id` 重跑——覆盖这件事不会报错，只会让两条互相矛盾
   的结论共用一个 run_id，事后无法分辨。
2. **缺标签不升格。** 见第 7 节：回推只允许降级。
3. **导出前置校验。** `export_results.sh` 在拷贝前调用 `benchmark/preflight.py`。
   它**不只看文件在不在，而是逐 run 解析内容**：
   原始明细（`requests.jsonl` 逐行可解析且至少一条记录；`meta.json` 可解析、
   必需键齐全、`run_id` 与目录名自洽、`data_label` 与汇总行自洽；
   `config.snapshot.yaml` 可解析）、环境指纹（合法 JSON 对象且**显式声明**
   硬件可用）、有效重复（**从汇总行重新推导**成员数，不信 `repeats.csv`
   自报的 `sufficient_repeats`，并核对两者是否同步）、实验登记
   （`run_id` 必须作为**表格单元**出现在 `EXPERIMENTS.md` 的某一行里——
   正文提一嘴不算登记）。
   缺任一必查项即中止（退出码 4），并逐条指名是哪个 run 出的问题。
   其中**原始明细与内容类的必查项在任何模式下都是阻断**；
   其余（指纹可否用、重复够不够、登记在不在）对 `MEASURED` 行阻断、
   对非 `MEASURED` 行降为提示（诊断包不是交付物，可追溯性要求不适用，
   但缺陷仍会打印出来，降级不等于消失）。

4. **平台声明自洽。** `runtime.execution_environment` 与 `runtime.hardware_platform`
   是两个独立声明，都会被写进 meta 并被下游按平台筛选。远程执行声明蕴含一个平台
   （`remote_ascend`→`ascend`、`remote_mthreads`→`mthreads`），两者矛盾时
   **在加载配置时**报错并给出可照抄的修法。任一方为 `unknown` 或执行环境为 `local`
   则放行——"没声明"推不出"说错了"，而"本机就是一台整机直接在上面跑"是真实情形，
   允许它写，人才不会为了让流程走通而谎报平台。
   平台名还会被用进"该去哪台机器重跑"的解释串（`label_reason`）：写死昇腾会把
   S4000 的 `HARDWARE_REQUIRED` 指向错误的机器，而这句话只在必须重跑时才出现。

5. **平台声明必须能被指纹证伪。** 上一项只保证**两个声明之间**自洽——说 MTT 且说
   在远程跑，可以完全自洽同时与真机毫无关系（"声明 MTT、指纹明确是 Ascend"就是这样
   通过的）。因此同一个判据（`provenance.validate_platform_evidence`）接在三处，
   **共用同一份环境指纹快照**：

   | 位置 | 时机 | 作用 |
   | --- | --- | --- |
   | `run_benchmark.run_once` | **跑基准之前** | 不浪费一次数分钟的 run，也不在证据目录留下矛盾产物 |
   | `provenance.build_reproducibility_record` | 装配元数据时 | 直接构造 `ReproInputs` 的调用方（脚本、测试）同样产不出矛盾记录 |
   | `preflight.check_export` | 逐 run 导出检查 | 最后一道闸：矛盾产物不得带着"硬件实测"的名义进交付包 |

   任一方为 `unknown` 一律放行：**没证据 ≠ 证据矛盾**，否则所有未采集指纹的机器
   会一起跑不动。运行前读到的指纹**带着往下传**（`write_run_outputs` →
   `TelemetryCollector`），使"判定用的"与"写进 meta 的"是同一个对象，而不是
   同一路径读两次（那样会出现"判定时读到 A、落盘时读到 B"的产物，且不报错）。

6. **设备型号/卡数只有一个权威键名。** 采集端规范化后写 `device.model` /
   `device.count`，读端**优先读它**，再回落历史写法，最后才解析 `npu-smi` 文本。
   卡数的回落链刻意不写成 `a or b or c`：`0` 是**测量出来的**"一张卡都没有"，
   假值会被静默跳过并最终变成 `None`——"测到没有卡"与"没查到卡数"的修法完全不同。

7. **可用性只认设备记录，不认产品名。** 判据是 `provenance.device_probe_result`，
   `scripts/inspect_env.sh` 与单测共用**同一份**实现（此前两边各写一份，于是同一句
   `mthreads-gmi 2.7.0 supports MTT S4000`：解析器返回"卡数未知"、采集判定却给出
   `available=true`）。取值表：工具不存在→`False`；跑了数出 `>=1` 张→`True`；
   数出 `0` 张→`False`；**数不出（含只有产品名）→`None`**。三态合成见
   `normalize_availability`。

8. **采到的版本号必须锚定它自己的标签；版本文件候选要覆盖真实形状。** 采集 MTT 侧版本时
   踩过两个**静默给错**的坑——都不是报错，是产物看起来完全正常：

   - **驱动版本**：真实 `mthreads-gmi` 首行把两个版本印在**同一行**——
     `mthreads-gmi:1.14.0          Driver Version:2.7.0`。早先的正则把两者写成交替
     `(?:mthreads-gmi|driver version)`，而**交替总是先匹配到位置靠前的那个标签**，
     于是工具版本被当成了驱动版本（真值 `2.7.0` 被写成 `1.14.0`）。现在只认显式的
     `Driver Version`；只有工具版本行时写 `None`。**错版本比没版本危险**：它不报错、
     不留痕，而驱动版本正是"引擎块大小为何是 64"这类结论的依据。
   - **MUSA 版本**：候选文件只列了 `version` / `version.txt`，而该机装的是
     `version.json`，于是 MUSA 版本成了 `None`——与"这台机器根本没装 MUSA"在产物里
     **完全无法区分**。`version.json` 里同名组件很多（`mublas`/`muPP`/`mufft`…），
     **只有 `musa_toolkits`（缺失时退到 `musa_runtime`）才是 SDK 版本**；
     随手取"文件里第一个数字"会取到 `schema` 或别的组件。解析收在
     `provenance.parse_musa_version_file`（纯函数、可单测），脚本只负责调用它。

   ★ 这两个缺陷**都逃过了单测**，原因比缺陷本身更值得记：当时的测试样本是**手写的
   想象输出**（把两个版本分成了两行），而不是真实产物。**凡"解析外部工具输出"的测试，
   样本必须逐字抄自真机**；用想象样本写的测试只验证了想象。同理，MUSA 那条的测试样本
   要刻意把别的组件排在 `musa_toolkits` 前面，否则"取第一个数字"那种实现也能通过，
   用例就分不出对错。

**绘图纪律**：绘图逻辑在 `benchmark/plot_p99.py`（可被单元测试覆盖，
而不是内嵌在 shell heredoc 里）。分组键是 `ascend_opt.grouping.GROUP_DIMENSIONS`，
即 **(workload, config, hardware_platform, backend)**——不是 config 单独一项，
也不止于平台维度。缺了负载会让"同一份配置跑在两个负载上"的数字混进同一个中位数；
缺了平台或后端则会让跨平台、跨后端的数字混进去（前者已复现：100 与 200 合成一根
150 的柱）。这套维度**只有一份实现**，汇总（`collect_metrics.group_key`，额外带
`run_id` 批次前缀以便跨重复聚合）、绘图、导出校验三处都从它派生；三处各写一份时
曾经分叉成"表格按后端拆开、图上没拆"，两边都不报错，只是互相矛盾。
某口径没有可用分位数时**留空并在原位标 `n/a`**，绝不兜底成 0：
0 会被读成"P99 = 0 ms"，那是一个好得不可能的测量值，比缺一根柱子更容易误导。
被略去的组与缺失的口径都会写进 `.source.txt`，分组口径那行**由维度定义生成**——
写死会在规则变化后继续宣称旧规则。

图标题**按数据动态生成**，不再写死平台名：标题里印着 `Ascend` 而数据来自 MTT 时，
那句错话不影响观感，只会在有人照着它去复现时把人指到错误的机器上。
标签在数据里真有两个以上平台时才带 `@<platform>` 后缀（单平台时重复同一句是噪音）；
若截断后会出现**重名**，则放弃截断而不是让两根柱子顶着同一个名字。
`.source.txt` 同时记录 `platforms` / `backends` / `precisions` 三行——
图缩进论文后没人能再从标题里读出它是哪个平台的数据。

**派生数据卫生**：聚合时会回收 `results/processed/` 里 **raw 已无对应 run** 的
逐 run CSV（`--keep-orphans` 可关闭）。理由：这类残留不会报错，只会看起来像一份
正常数据——本仓库实际出现过旧 schema 残留（列里是 `success` 而没有 `outcome`，
也没有 TTFT 三段分解）。判据是"raw 里有没有同名目录"这种结构性事实，而不是
"本次有没有被选中"，所以 `--run-id X` 这类过滤运行不会误删其它 run 的明细。

**重复组以落盘清单为准**：`repeats.csv` 每行带 `member_run_ids`（分号分隔、已排序），
导出校验**读它**而不是按 `run_id` 前缀自行推导成员。理由同上——推导等于在第二处
维护同一条分组规则，且前缀法表达不了"同一批次被拆成两组（不同平台/后端）"：
它会把两组各 3 个成员合成 6，报出并不存在的 `repeats_count_mismatch` 假报错，
把真正的问题盖住。"不信自报旗标"这条纪律仍然成立，只是换了落地方式：**不读**
`sufficient_repeats`，而把 `member_run_ids` 与汇总表实际存在的 `run_id` 求交集
重新数。缺 `member_run_ids` 列（旧版聚合产物）时明确报 `repeats_members_missing`
要求重跑聚合——缺一条证据时正确的动作是补证据，不是猜。另有一条反方向的检查
`repeats_member_unaccounted`：汇总表里有 run 属于某批次却没有任何一组认领它，
说明聚合漏了它（只数已认领的成员发现不了漏掉的那些）。

`meta.json` 的 `reproducibility` 段按规范第 8 节保留 21 个字段
（`run_id` / `timestamp` / `git_commit` / `hostname_or_machine_id` / `device_model` /
`device_count` / `model` / `model_revision` / `precision` / `quantization` /
`tensor_parallel_size` / `input_length` / `output_length` / `concurrency` /
`batch_configuration` / `max_batched_tokens` / `chunk_size` / `prefix_cache_enabled` /
`graph_mode` / `random_seed` / `warmup_count` / `measurement_count`）。
取不到值的项写 `NOT_MEASURED` / `HARDWARE_REQUIRED`，并在 `_missing_fields` 中列出。

## 9. 研究阶段

按规范第 21 节推进，不跳阶段。

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| PHASE 0 | 仓库引导与环境盘点 | 🟢 完成——本机与**远程 MTT S4000 主机**两份指纹俱已采集（远程那份见 RESEARCH_LOG 的 2026-09-22 条目） |
| PHASE 1 | 真实后端部署（**引擎待定**） | 🟡 进行中——**Qwen 单请求生成已在真机验证通过**（0.5B + 7B：设备执行 / 输出正确性 / 显存占用，逐项见 RESEARCH_LOG 2026-09-22 条目），独立环境已建成。**`RealBackend` 数值正确性已验证通过**：四场景（单请求 / 多步 KV decode / 变长 batch / chunked prefill）数值等价 + 资源生命周期，0.5B 与 7B 双模型 `passed=true`。方案与执行记录见 `VALIDATION_PLAN_PHASE1.md`（§7.3 为**第二代**记录）。**② 于 2026-09-24 重跑**：判据修掉两处"看起来通过"的漏检（`V4b` 只验首个分岔步、`V3` 补位检查不覆盖合批 decode、槽位判据在 `retire` 前读而**恒假**）后重采，产物 `results/raw/phase1_validation_b/`（`580fcf66…` / `bc33a499…`），**自带可核验溯源戳** `git_commit=73dc2d7`、`git_dirty=false`、`validated=true`；第一代 `results/raw/phase1_validation/` **降级为"判据更弱的一代"**，不得据其宣称"完整路径已验收"。判据仍带缺陷那版的首跑证据留在 `results/raw/_pilot/phase1_validation_precheck/`（不进聚合）。逐项见 RESEARCH_LOG 2026-09-24 条目。**性能矩阵已跑完（36/36，首批 `MEASURED` 数据）**：3 臂（不分块 / 分块 / 自适应）× 并发 1/2/4/8 × 3 轮穿插执行，得到 **C1/C2**（分块预填充与不分块的 TTFT↔stall 权衡，差异 40%~72%，远超噪声底）与 **N1~N3**（含"`adaptive` vs `static_chunk` 测不出差异"）；⚠️ **本轮的 `adaptive` 臂因配置缺陷未测到它声称要测的东西**（`chunk_size_init` 等于下界、`target_tpot_ms` 不可达 ⇒ 控制器退化为单向棘轮），必须修配置后重跑。设计与完整读数见 `PERFORMANCE_MATRIX_PHASE1.md`，结论登记见 `EXPERIMENTS.md` §1.1/§3/§4。**修正方案已落地并已启动重跑（`P1S3B`）**：生成器改为 profile 驱动（`p1s3` 可逐字复现已执行配置、`p1s3b` 为修正版），修正三处 + 补一处——`chunk_size_init` 512→1024（且 `chunk.chunk_size` 取同值，保证 t=0 两臂逐字一致）、`target_tpot_ms` 50→110、`chunk_size_max` 16384→2048（超过 prompt 全长即"grow 了但不切分"的死区）、以及**原修正方案漏掉的 `target_ttft_ms` 2000→500**（基线值高于四档实测 TTFT P90 的全部取值 ⇒ grow 判据一次都不会触发，控制器仍只会收缩）；两个目标的取值区间均由 P1S3 实录信号反解。**两道可阻断的前置门**已建：生成期结构性可动自检（`_assert_controller_movable`）+ 开环可达性复放（`scripts/check_step3b_reachability.py`，结论：两方向可达、净方向符合"低并发净放大、高并发净收缩"、恢复路径非棘轮；**开环复放不预测闭环终态**）。设计与踩坑见 `PERFORMANCE_MATRIX_PHASE1B.md`、RESEARCH_LOG 同日条目。**修正版矩阵已跑完并入库**（`P1S3B-*`，36 份 `MEASURED`；结论见 `EXPERIMENTS.md` §1.2/§3.2，**可用结论只有 C1/C2**——`adaptive` 臂因"门限贴着工作点"跨重复不可复现，仍不可用）。**`P2` 完成计时口径的前后对照**：token 时间戳移到采样与取回之后，实测总作用 ≤0.25%、无既有结论翻转（`EXPERIMENTS.md` §4.3.2）。**`P3` 完成 SLO 判据重标定**：`run.slo.tpot_ms` 50 → **100**（原值低于实测逐请求 TPOT 下界 61.7 ⇒ 恒不可达 ⇒ 达标率三列恒 0，属"判据无效"而非"达标率低"；新 profile `p1s3c`，生成器新增 `_assert_slo_reachable` 守卫，见 `EXPERIMENTS.md` §4.3.3）。**`P4` 完成控制门限重标定**：`controller.target_tpot_ms`/`hysteresis` 110/0.1 → **120/0.08**（原放大门限 99.0 落在 c4 实测工作点 98.7 的邻域内，headroom 余量只剩 0.0112 对门槛 0.1000 ⇒ 该格行为跨重复不可复现，实测三次 TTFT 481.4/481.6/652.1；新 profile `p1s3d`，生成器新增 `_assert_gate_separation` 守卫，标定工具 `scripts/calibrate_controller_gate.py`，见 `EXPERIMENTS.md` §4.3.4）。⚠️ **P4 与 P3 不同：门限改动会改变调度行为 ⇒ `p1s3d` 必须重跑**（`p1s3c` 的产物不能借用）。⚠️ **且门限问题只有定量改善、未定性解决**——工作点跨 61.8→571.7 ms（9.3×）而滞回带宽仅 ≈19 ms，绝对阈值 + 单一 `(target,h)` 与多档工作点分不开，属判据形式的障碍（登记待办）。**`P1S3D` 重跑已完成并入库**（36/36 成功，36 份 `MEASURED`；登记 `EXPERIMENTS.md` §1.4、结论 §3.3 **E1–E9**）：① 旧门限下那个具体拒绝（工作点 97.8–99.2 跨过 99.0）在新门限 110.4 下**在构造上不再可能**（余量 0.0112 → 0.1052），且**作用面很窄**——除 c4 外 c1/c2/c8 的决策计数与 prefill 生效分布**逐格与 `P1S3B` 相同**；② ⚠️ **但不能说"c4 跨重复问题已解决"**——`P2`（**旧门限** ＋ 计时修复）的 c4 同样 3/3 收敛 ⇒ n=3 分不开"新门限消除了它"与"没抽中"，**D2 的限制只是被削弱**，并**更正 §1.3 注 2 的"与计时修复无关"为"归因未建立"**；③ **形式性障碍仍在**（理由要说**负载连续变化会扫过整条区间**；原文"c4 区间 [97.8, 571.7] 仍跨过新门限"取自一个把**收缩点**算进去的输出列，**已更正**，见 §6 与 `EXPERIMENTS.md` §4.3.4）：闭环出现**与"请求补入批"同步**的周期性调整（周期 = 补入批间隔 ≈ 一个请求的 e2e ≈ 129–132 个 decode 步，**由负载决定、非控制器固有周期**；触发机制 = 观测窗混入批边界的 prefill 步，见 §3.3 E7/E9）、`recover` 全程未触发、c4 的 6 次决策只有 1 次落到 prefill。达标率**首次携带信息**（阈值 100）：c1/c2 = 1.000、c4 = 0.125、c8 = 0.000，而 **`P4` 未改变任何一格**（按同阈值重算 `P1S3B`，结果逐格相同）。⚠️ 本批与 `P1S3B` **计时口径不同**（含 `readback_ms` 修复）⇒ 两批绝对 TTFT 不得并表。新增 `inspect_step3_runs.py --verify-config`：部署戳 `digest_scope` **不含 `configs/`**，故逐 run 核对快照（36/36 一致，反向对照用 `phase1_step3c` 返回 1）。路线剩余：连续批处理 / 前缀缓存 / 分块预填充实验 → **按功能支持度选引擎**（候选与依据见 RESEARCH_LOG 第 6 节）。远程主机已有 MTT S4000 / driver 2.7.0 / MUSA 3.1.0 / torch_musa 1.3.0，已装 `transformers 4.46.3` 并缓存 Qwen 权重；`huggingface.co` 不可达，须走 `hf-mirror.com` |
| PHASE 2 | 基准与遥测框架 | 🟢 已建成并离线验证（`--dry-run` 全链路）。**测试入口与真实基线**（2026-09-24 第 02 步）：用 managed 解释器 `versions/3.13.12/python.exe` ⇒ **800 tests，OK（0 failure / 0 error / 0 skipped）**；两个入口见 §5 第 2 步。⚠️ 曾把 `envs/default` 解释器（**没装 torch**）跑出的 `8 errors / 25 skipped` 记成"本机未装 torch"——**本机装了**，那是**选错解释器**；成因与两级入口见 `scripts/run_tests.py` |
| PHASE 3 | 基线负载表征 | ⬜ 未开始（等 PHASE 1） |
| PHASE 4 | 单项优化实验 | ⬜ 未开始 |
| PHASE 5 | 自适应控制器实现 | 🟡 策略层闭环已建成并离线验证（双向调节 + 信号闸门 + 建议/生效对照）；**真机行为未验证**，信号阈值待按实际设备显存与命中率分布标定 |
| PHASE 6 | 消融与对比实验 | ⬜ 未开始 |
| PHASE 7 | 统计分析与图表 | ⬜ 未开始 |
| PHASE 8 | 论文方法学与实验结果 | ⬜ 未开始（**在实验证据出现前不得进入**） |
| PHASE 9 | 相关工作对比与论文定稿 | ⬜ 未开始 |

当前环境的关键事实：**研究主体平台是 MTT S4000，权威推理环境是远程 MTT 主机，本机不是**
（规范末节 `Remote Execution Environment`）。开发机为 Windows 笔记本，`mthreads-gmi` 与
`npu-smi` 均不可用，`scripts/inspect_env.sh` 采集到的指纹中
`probes.ascend.result = false`、`probes.mthreads.result = false`，因而
`available = false`、`looks_like_ascend_host = false`、`looks_like_mthreads_host = false`。

> ⚠️ **2026-09-24 晚：远程实例已更换镜像（已实地核验，仅"可达性 + 环境盘点"部分）。**
>
> **核验方式**：`scripts/remote_run.py exec`（直连，非统计工具），逐项读原始输出。
>
> | 项 | 新镜像实测值 | 与 2026-09-22 记录比 |
> | --- | --- | --- |
> | 容器名 | `autodl-container-73d6479980-71280c16` | **变了**（旧 `…3dc442bf15-b061c1a5`） |
> | SSH 端口 | **14293** | **变了**（旧 14671） |
> | 驱动 / `mthreads-gmi` | 2.7.0 / 1.14.0 | 相同 |
> | MUSA SDK | **3.1.0**（`musa_toolkits` = `musa_runtime` = 3.1.0） | 相同 |
> | 设备 | MTT S4000 × 1，49152 MiB，0% util，占用 4 MiB，41 C | 相同 |
> | Python | 3.10.8 @ `/root/miniconda3`（`--system-site-packages` 的基底） | 相同 |
> | `torch` / `torch_musa` | **2.2.0**（`2.2.0a0+git8ac9b20`）/ **1.3.0+81caf0a** | 相同 |
> | 磁盘 | `/` overlay 30 G（用 53 M）；`/root/autodl-tmp` 50 G（**空**） | — |
> | 远端出网 | `https://hf-mirror.com` → **HTTP 200**（1.9 s） | 可用 |
>
> **✅ 三样丢失项已重建（2026-09-24 夜，按 `scripts/setup_remote_musa_env.sh`，幂等）**：
> ① `/root/musa-envs/qwen310` 已建（`venv --system-site-packages`，python 3.10.8）；
> ② `transformers 4.46.3` + `tokenizers 0.20.3` + `accelerate 1.1.1` + `safetensors 0.4.5` 已装
> （版本**照抄** 2026-09-22 记录，未升级）；
> ③ 模型缓存已重建（`Qwen2.5-0.5B-Instruct`，954 MB，
> revision `7ae557604adf67be50417f59c2c2f167def9a775`）。
> `activate_musa.sh` 由该脚本**逐字照抄**既有原文（2202 B，与 `C:\tmp\remote_scripts\` 的
> 原文 `diff` 相同）——⚠️ **"按文档描述恢复" ≠ "恢复成原来那份"**，第一版就是按描述合成的，
> 少了 `MODELS_DIR` 等三处；**有原文就照抄原文**。
> ✅ **真机正确性验收已通过（0.5B + 7B 双模型）**：`results/raw/phase1_validation_c/`
> —— 两个产物均 `passed=True` / `needs_review=[]` / 6 阶段全过；0.5B 戳 `2bdf7607c784`、
> 7B 戳 `654914ee40bf`，`git_dirty=False`、远端重算摘要相符；md5 两端一致。
> ⭐ **7B 与上一代 `phase1_validation_b` 逐项一致**（连"V3 的 v3_0 在第 4 个 token 分岔"都一样）
> ⇒ 新镜像**复现了同一套数值行为**，不只是"也过了"。
> ⚠️ 那两个 token 分岔是产物 `notes` 里记明的**近平局**（gap=0.0000 < 0.05，fp16 舍入即可翻转
> argmax），按三态分流规则**不阻断通过**；详见 `NEXT_SESSION_TASKS.md` 第 07 步记录 §六/§七。
>
> **一条仍然成立的前提（已复核，别照抄结论、要照抄判据）**：
> `strings /root/miniconda3/lib/libstdc++.so.6 | grep -c GLIBCXX_3.4.30` = **0**
> ⇒ conda 自带的 libstdc++ **仍然缺 `GLIBCXX_3.4.30`** ⇒
> **`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` 这条仍然必要**（成因见 §9 后文）。
> ⚠️ 验证这类改动必须**按 `tokenizers → torch` 的顺序导入**——按 `torch` 先导测不出这个坑。
>
> **⚠️⚠️ 测量有效性风险（新发现，必须在做任何性能实验前处理）**：
> 实测 **load average = 13.85 / 14.77 / 14.40，而 `nproc` = 15** ⇒ 所在主机**接近满载**。
> 探针已测定"主机提交成本与设备工作量同量级"（`COST_PROFILE_S4000.md`），
> 因此**CPU 争抢会污染计时**。⚠️ 未核验该 loadavg 是容器视角还是主机视角（lxcfs），
> **不得**直接当作本容器负载。⇒ 任何 `MEASURED` 批次开始前必须先记录 loadavg 并做稳定性对照；
> 若在波动，**不得**开始正式计时。
>
> ✅ **环境指纹已于 2026-09-24 夜重采**（`PYTHON=python3 bash scripts/inspect_env.sh`）：
> `platform=mthreads` / `available=True` / `device.model=MTT S4000` / `count=1` /
> `MTT_DRIVER_VERSION=2.7.0`（源 `mthreads-gmi:version-line`）/
> `MUSA_VERSION=3.1.0`（源 `file:/usr/local/musa/version.json`）/
> `TORCH_VERSION=2.2.0` / `TORCH_MUSA_VERSION=1.3.0+81caf0a`。
> ⚠️ `software.transformers_version = null` 是**用哪个解释器采指纹**的后果
> （用的是 base `python3`，而 transformers 装在 venv 里）——与原记录一致，非缺陷但须知情。
> ⚠️ 指纹文件不入库（`.gitignore` 排除 `results/raw/_env/`）⇒ **主机的持久记录只在本文件
> 与 `RESEARCH_LOG.md`**；换机或重采后要同步这两处。
> **换镜像 ⇒ 新的实验条件**：旧批次的 SLO 与控制门限标定**不自动适用**，不得与旧条件并表。
>
> **⚠️ 宿主机负载 ≠ 本容器负载（2026-09-25 实测）**：该主机**没有 lxcfs** ⇒
> `/proc/loadavg` 是**宿主机**的（实测 12.85 而宿主 **128** 核，`nproc` = **15**
> 是 cgroup 配额，`cpu.max = 1500000 100000`）。⚠️ **`nproc` 与 `/proc/cpuinfo` 给的核数
> 可以差 8.5 倍**（15 vs 128）——用错分母会把利用率读错一个数量级。
> 判断"会不会被抢 CPU"要用**容器级**仪器：`/sys/fs/cgroup/cpu.pressure`（实测全 0.00）。
> **开工前跑 `scripts/check_cpu_stability.py`**（三态：0 稳定 / 1 拒绝 / 2 无法判定）；
> 实测该主机 7×1s 纯 CPU 计算的稳健 CV = **0.003%** ⇒ 计时环境干净。
> （连通性诊断的判据与本次根因**未分离**的说明见 `PLAYBOOK.md` §5.1.1 / §5.1.2。）

远程 MTT 主机的指纹（**2026-09-22 采集，历史记录**——换镜像后除上表已复核项外**待重采**，
逐项见 RESEARCH_LOG）：`platform = mthreads`、
`available = true`、`device.model = MTT S4000`、`device.count = 1`、显存 49152 MiB、
driver **2.7.0**、MUSA SDK **3.1.0**、`torch 2.2.0` + `torch_musa 1.3.0+81caf0a`。
`vllm` **仍未安装**（故 `VLLM_MUSA_VERSION` 仍为 `None`）。

远程的推理环境（2026-09-22 建成，**续用**）：`venv --system-site-packages` 落在
`/root/musa-envs/qwen310`，从而保留下层 conda 里可用的 `torch` / `torch_musa` / MUSA，
只在本层加装 `transformers 4.46.3` 等模型依赖；权重缓存与大数据放 `/root/autodl-tmp`
（独立 50G 卷，`/root` 在 30G overlay 上）。

**跑任何远端 Python 前必须先 `source /root/musa-envs/qwen310/activate_musa.sh`。**
它设 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`——这不是预防性添加：
conda 自带的 libstdc++ 是 6.0.29（不含 `GLIBCXX_3.4.30`），而 `tokenizers` 的扩展
`.so` 无 RUNPATH、会继承 conda python 的 RPATH 把旧的那份先载进进程，同名 soname 被复用后
torch 再 `dlopen` 只能拿到旧的，于是 `import tokenizers` 之后 `import torch` 崩
（**导入顺序相关**，torch 单独没问题）。成因与 `/proc/self/maps` 实测见 RESEARCH_LOG。

`vllm` / `transformers` 在本机 `software_env` 段里有、在远程 `hardware_env` 段里没有——
这两段分开记录，正是为了不把"跑 harness 的那台机器"误读成"执行推理的那台机器"。

**指纹文件本身不入库**：`.gitignore` 明确排除 `results/raw/_env/`（它是某台机器某一刻的
本地状态，随仓库分发会被误读成"本项目的运行环境"）。因此**关于推理主机的持久记录只在本
文件与 `RESEARCH_LOG.md` 里**——换机器或重采指纹后要同步更新这两处，别指望从库里
diff 出来。

**设备侧探针产物入库、但不进结论**：`results/raw/_probe/<日期>/` 放"由设备侧探针直接生成、
没走 `benchmark/run_benchmark.py`"的产物与脚本（如 Qwen 单请求验证、带宽实测）。放这里是因为
`results/raw/` 的定位就是证据——不落盘，`RESEARCH_LOG` 引用的数字就变成不可复核的断言。
但这类产物**没有 harness 盖章、也没有 `meta.json`**，因此：① `collect_metrics.py` 的
`discover_runs()` 只认含 `meta.json` 的目录，不会把它当 run；② 它**不得**写进 `EXPERIMENTS.md`
的结论行，也**不得**当作 `MEASURED` 数据使用。

后果与正确动作：

- 本机只允许跑**离线路径**（`--dry-run`、单元测试、聚合与导出的干跑）。
- 本机跑"真机路径"的 run 会落成 `HARDWARE_REQUIRED`，**不能产出任何性能结论**；
  这不是"少了环境的实测"，是**不可用**的数据。
- 硬件相关操作（设备探测、驱动与 MUSA SDK / CANN 检查、模型加载、推理、基准、
  设备利用率监控、显存测量、profiling）必须在远程主机上执行，并在**那台机器上**
  采集环境指纹——指纹描述的是采集它的那台机器，在本地采一份远程指纹不成立。
- 若远程 MTT S4000 主机部署了服务、由本机驱动推理，需在配置里声明
  `runtime.execution_environment: remote_mthreads`（次平台的昇腾主机则为
  `remote_ascend`，见 `configs/mthreads/s4000_fp16.yaml`）。这样判定才会问对问题
  （"**执行推理的那台机器**有没有卡"），而不是拿本机指纹误杀真实测数据。
  声明本身不足为凭：仍需远程指纹确认设备可用。

## 10. 协作约定

代码边界、实验流程、测量纪律、写作约定与禁止事项见 **[AGENTS.md](AGENTS.md)**
（正文为研究规范，附录 A 为操作契约）。开工前请先读它。
