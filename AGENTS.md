# Project: MTT S4000 Large Language Model Inference Optimization

> 🧭 **导航入口：先读 [`PROJECT_MAP.md`](PROJECT_MAP.md)** —— 它给你"读什么、按什么顺序、
> 去哪找 X、脚本怎么调、环境怎么进"。**本文件是研究规范**（规则、口径、红线），
> 它回答"什么算证据"，**不**回答"去哪找"。两者互补，不要用一个替代另一个。

> **本文件由两部分组成，效力不同，不要混读：**
>
> - **正文（第 1–21 节 + Remote Execution Environment）**：研究规范，权威文本。
>   定义了研究问题、目标技术栈、实验方法、指标口径、可复现性要求、数据完整性规则，
>   以及**执行环境**（**研究主体平台是 MTT S4000；权威推理环境是远程 MTT 主机，本机不是**）。
>   改动必须谨慎。
> - **附录 A（A1–A7）**：仓库操作契约。规定在本仓库里**怎么干活**——
>   代码边界、流水线、禁止事项。它服务于正文，与正文冲突时以正文为准。
>
> 两者互补：正文说"要回答什么问题、什么算证据"，附录说"代码放在哪、提交前跑什么"。

---

# 1. Project purpose

This repository supports an academic research project on large language model inference optimization for **MTT S4000** GPUs (Moore Threads).

Provisional paper title:

《面向 MTT S4000 的请求感知型大语言模型推理部署优化研究》

English title:

Request-Aware Large Language Model Inference Deployment Optimization on MTT S4000

The project is not merely a deployment tutorial.

The research objective is to design, implement, and experimentally evaluate a workload-aware inference optimization framework for MTT S4000 systems.

**Terminology (normative).** The research subject is the MTT S4000. The target accelerator is
referred to as a "device" or "GPU", and its memory as "device memory". Where this document, the
README, or existing code still say "NPU" / "HBM" / "显存", read "the target accelerator / its
device memory" — unless the passage is explicitly about Ascend. Ascend NPU support is retained
as a **secondary** platform (see section 3, and the `hardware_platform` dimension in section 10);
work that targets Ascend must use the Ascend-specific terms and Ascend-specific probe tools.
Renaming identifiers is not required by this clause and must not be done silently.

The proposed framework should dynamically coordinate:

* batching
* Prefix Cache
* KV Cache usage
* Chunked Prefill / Split Prefill
* graph execution where applicable
* quantization configurations
* device memory pressure
* request queue state

Primary objectives:

1. reduce TTFT;
2. reduce TPOT and ITL;
3. improve token throughput;
4. improve device utilization;
5. control device memory consumption;
6. maintain performance under heterogeneous workloads.

---

# 2. Research question

The central research question is:

Can an adaptive controller make better inference configuration decisions than a fixed static configuration when workload characteristics change?

Let system state be:

s_t = [
input_length,
estimated_output_length,
queue_length,
concurrency,
prefix_reuse_ratio,
cache_hit_ratio,
available_memory,
NPU_utilization,
TTFT,
TPOT
]

The controller chooses an action:

a_t = [
batch_size,
max_batched_tokens,
chunk_size,
prefix_cache_policy,
graph_mode_policy
]

The optimization objective is conceptually:

maximize throughput

subject to:

TTFT <= TTFT_SLO

TPOT <= TPOT_SLO

memory_usage <= available_NPU_memory

No implementation should assume that one optimization is universally beneficial.

All optimization claims must be supported by measurements.

---

# 3. Target software stack

Preferred target stack — **primary platform (research subject): MTT S4000**:

Linux (Ubuntu 22.04 LTS)
→ MTT driver / firmware (`mtml`, `sgpu-dkms`)
→ MUSA SDK
→ torch_musa
→ PyTorch
→ vLLM
→ vLLM-MUSA

Preferred target stack — **secondary platform (retained): Ascend NPU**:

Linux
→ Ascend driver / firmware
→ CANN
→ torch_npu
→ PyTorch
→ vLLM
→ vLLM-Ascend

MindIE may be studied as a reference implementation or comparison platform where appropriate.

Do not assume exact versions.

Before implementation or benchmarking, inspect and record the actual environment.

Record at minimum:

* OS and kernel
* CPU architecture
* device model
* number of devices
* device memory capacity
* driver version
* firmware version
* MUSA SDK version (primary) / CANN version (secondary)
* Python version
* PyTorch version
* torch_musa version (primary) / torch_npu version (secondary)
* vLLM version
* vLLM-MUSA version (primary) / vLLM-Ascend version (secondary)
* model name
* model revision
* model precision / quantization
* Git commit of this repository

Never silently change the driver, firmware, MUSA SDK / CANN, or any system-wide accelerator software.

If an environment modification may affect an existing accelerator setup, explain the proposed change before making it.

The two stacks are **not interchangeable and not comparable**: a run on one platform and a run on
the other are different experimental conditions, and their numbers must never be pooled into one
median (section 10).

---

# 4. Initial model scope

Start with a Dense Transformer model.

Preferred first target:

Qwen-family Dense model around 7B–8B parameters.

Do not begin with a large MoE model unless the baseline Dense pipeline is complete.

MoE / Expert Parallel / EPLB is a later extension.

The implementation must not hard-code a single model architecture where avoidable.

---

# 5. Experimental methodology

The project must always distinguish:

1. baseline;
2. individual optimization;
3. combined optimization;
4. proposed adaptive method.

Minimum experimental variants:

Baseline

Baseline + quantization

Baseline + graph execution

Baseline + Prefix Cache

Baseline + Chunked Prefill

Baseline + selected combined optimizations

Baseline + proposed adaptive controller

Always run ablation experiments.

Do not report only the final optimized configuration.

---

# 6. Workload dimensions

Experiments should support systematic sweeps across at least:

Input length:

128
512
2048
8192
16384

Output length:

128
512

Concurrency:

1
8
16
32
64

Prefix reuse ratio:

0%
25%
50%
75%

Additional workload distributions should include:

* short-input / long-output;
* long-input / short-output;
* high-prefix-reuse workload;
* low-prefix-reuse workload;
* low-concurrency workload;
* high-concurrency workload;
* mixed workload.

Do not treat synthetic workloads as representative of every real production environment.

---

# 7. Metrics

Collect at minimum:

* TTFT
* TPOT
* ITL
* request throughput
* output token throughput
* total token throughput
* end-to-end latency
* NPU memory usage
* NPU utilization if available
* Prefix Cache hit ratio where applicable

Prefer distributions instead of averages alone.

Record where possible:

* mean
* median
* P90
* P95
* P99

For throughput use:

tokens / second

For memory use:

MiB or GiB

Never mix units.

---

# 8. Reproducibility requirements

Every benchmark execution must have a unique run ID.

Each raw result should preserve metadata including:

run_id
timestamp
git_commit
hostname_or_machine_id
device_model
device_count
model
model_revision
precision
quantization
tensor_parallel_size
input_length
output_length
concurrency
batch_configuration
max_batched_tokens
chunk_size
prefix_cache_enabled
graph_mode
random_seed
warmup_count
measurement_count

Every result used in the paper must be traceable to a raw result file.

Do not manually edit raw benchmark output.

Processed results must be generated programmatically.

---

# 9. Statistical requirements

Do not draw conclusions from one benchmark execution.

Use warm-up runs.

Use repeated measurements.

Report variability.

Flag unstable experiments.

Do not remove outliers unless the exclusion rule is documented before analysis.

Do not claim statistical significance unless an appropriate statistical test was actually performed.

---

# 10. Data integrity

This requirement is absolute:

NEVER INVENT EXPERIMENTAL RESULTS.

NEVER generate fake TTFT, TPOT, throughput, utilization, memory, or latency numbers.

If Ascend hardware is unavailable, create the benchmark framework and mark results as:

NOT_MEASURED

or

HARDWARE_REQUIRED

Synthetic values may only be used in unit tests and must be clearly labelled TEST_DATA.

Synthetic test data must never appear in paper result tables.

---

# 11. Baseline-first rule

Before implementing the adaptive optimization algorithm:

1. verify the model can run;
2. establish a reproducible baseline;
3. validate metric collection;
4. run baseline workload sweeps;
5. identify actual bottlenecks.

Do not optimize before the baseline is measurable.

Do not assume the bottleneck.

Measure it.

---

# 12. Optimization modules

The architecture should separate the following components.

## WorkloadMonitor

Responsible for observing:

* input length;
* queue depth;
* concurrency;
* Prefix Cache statistics;
* NPU memory pressure;
* measured latency;
* utilization if available.

## AdaptiveBatchController

Selects batching-related parameters based on observed system state.

## PrefixCachePolicy

Determines whether reusable prefixes should be retained or evicted.

A possible research metric may consider:

cache_value =
frequency × prefix_length × estimated_recompute_cost / memory_cost

This formula is a research starting point, not an assumed optimal solution.

## ChunkController

Dynamically selects Prefill chunk size.

Investigate trade-offs between:

* Prefill throughput;
* Decode interference;
* TTFT;
* TPOT.

## AdaptiveController

Coordinates all policies.

Keep the controller interface independent from the benchmark harness.

---

# 13. Implementation policy

Prefer small, inspectable changes.

Avoid unnecessary modification of upstream vLLM-Ascend in the first stage.

First implement instrumentation and policy prototypes around documented interfaces.

Only patch upstream scheduler/runtime components when measurements demonstrate that external control is insufficient.

If upstream source code must be changed:

* keep changes minimal;
* document exact upstream files;
* explain why each modification is necessary;
* keep patches reproducible;
* record upstream commit.

---

# 14. Code quality

Use Python type annotations for project code.

Use dataclasses or typed configuration models for experiment configuration.

Avoid hard-coded paths.

Avoid hard-coded device IDs.

All experiment parameters should be configurable.

Where practical, separate:

configuration
execution
measurement
analysis
visualization

Tests should cover:

* workload generation;
* metric parsing;
* experiment metadata;
* cache policy;
* controller decisions;
* result aggregation.

---

# 15. Repository responsibilities

configs/
contains reproducible experiment configurations.

src/
contains optimization logic.

benchmark/
contains benchmark orchestration and metric collection.

results/raw/
contains immutable raw results.

results/processed/
contains derived datasets.

results/figures/
contains automatically generated figures.

paper/
contains research text, tables, and figure references.

RESEARCH_LOG.md
records important observations, failed experiments, design decisions, and unexpected behavior.

EXPERIMENTS.md
contains the experiment matrix and execution status.

SOURCES.md
tracks external documentation and academic sources.

---

# 16. Academic integrity and source policy

Clearly distinguish:

* documented behavior;
* our implementation;
* our hypothesis;
* our measured observation;
* our interpretation.

Do not present an existing vLLM, vLLM-Ascend, CANN, MindIE, or academic technique as an original contribution.

Before describing something as a contribution, compare it against existing documented functionality.

For implementation facts, prefer primary sources such as:

* official Ascend documentation;
* official CANN documentation;
* official MindIE documentation;
* official vLLM documentation;
* official vLLM-Ascend documentation;
* upstream source code.

For research background, prefer peer-reviewed papers or original technical papers.

Record URLs, paper titles, dates, and versions in SOURCES.md.

If internet access is unavailable, do not fabricate references.
Insert a clearly marked citation TODO instead.

---

# 17. Paper contribution target

The intended contribution is NOT:

“we enabled Prefix Cache”

or

“we enabled W8A8”

or

“we increased Batch Size.”

These techniques already exist.

The intended contribution is a workload-aware coordination mechanism that decides how existing inference optimization mechanisms should be configured under changing workloads.

Potential final contribution structure:

Contribution 1:
workload characterization and inference bottleneck analysis **on MTT S4000**
(the research subject platform — see section 3 and "Remote Execution Environment";
this line read "Ascend" before the 2026-09-22 decision and was corrected on 2026-09-24).

Contribution 2:
request-aware joint optimization of batching, Prefix Cache, and Prefill chunking.

Contribution 3:
implementation and evaluation on **real MTT S4000 hardware** with extensive ablation experiments.

Do not claim these contributions as novel until the related-work review supports that claim.

⚠️ **Related-work check performed (2026-09-24, step 01).** The review in `SOURCES.md` §6 found that
Contribution 2 is **partially pre-empted** by published work — in particular SLOWeave
(arXiv 2609.07883) already performs cost-predictor-driven adaptive prefill chunking under a
next-token deadline, and DRLServe (ECAI 2025) already performs learned adaptive chunk sizing.
⇒ **Contribution 2 may not be claimed as novel.** Choose one of:
(a) state a **testable difference** and verify it (see `RESEARCH_PLAN.md` §4), or
(b) retreat to a **reproduce/adapt** positioning on this platform and load regime.
Also note: **the hardware model is not itself a contribution.**

---

# 18. Paper figures

All plots used in the paper must be generated from processed benchmark data.

Useful figures may include:

TTFT vs concurrency

TPOT vs concurrency

throughput vs concurrency

throughput vs input length

TTFT vs input length

HBM usage vs concurrency

Prefix Cache speedup vs reuse ratio

cache hit ratio vs cache budget

chunk size vs TTFT

chunk size vs TPOT

static scheduler vs adaptive scheduler

Pareto frontier:
latency vs throughput

Do not create misleading truncated axes without documenting them.

---

# 19. Optimization success criteria

The adaptive approach should be evaluated against static baselines.

Possible success indicators include:

* higher throughput under the same latency SLO;
* lower TTFT under comparable throughput;
* lower TPOT;
* better P99 latency;
* better memory efficiency;
* more stable performance across mixed workloads.

No fixed percentage improvement is assumed in advance.

Negative results must be preserved and analyzed.

---

# 20. Required behavior for Codex

Before modifying files:

1. inspect the repository;
2. read AGENTS.md;
3. inspect the execution environment;
4. identify available **target-accelerator hardware** (research subject platform: MTT S4000;
   Ascend NPU is the retained secondary platform — see "Remote Execution Environment");
5. identify installed software versions;
6. identify whether internet access is available;
7. summarize relevant constraints.

For substantial work:

1. state the intended change;
2. implement it;
3. test it;
4. report changed files;
5. report commands executed;
6. report validation results;
7. identify unresolved issues.

Never silently skip failed tests.

Never hide failed experiments.

Never fabricate benchmark success.

Never overwrite raw experimental data.

Do not modify unrelated files.

---

# 21. Research workflow

Follow these phases in order unless there is a documented reason to deviate.

PHASE 0
Repository bootstrap and environment inventory.

PHASE 1
Baseline vLLM-Ascend deployment.

PHASE 2
Benchmark and telemetry framework.

PHASE 3
Baseline workload characterization.

PHASE 4
Individual optimization experiments.

PHASE 5
Adaptive controller implementation.

PHASE 6
Ablation and comparison experiments.

PHASE 7
Statistical analysis and figures.

PHASE 8
Paper methodology and experimental results.

PHASE 9
Related-work comparison and final paper refinement.

At the end of each phase update:

RESEARCH_LOG.md
EXPERIMENTS.md
README.md when appropriate.

Do not jump directly to PHASE 8 before experimental evidence exists.

---

# Remote Execution Environment

The authoritative execution environment for this project is the **remote MTT S4000 host**
(Moore Threads). MTT S4000 is the research subject platform; Ascend NPU remains supported as a
secondary platform, but it is not the subject and its host is not the authoritative environment.

The local machine must NOT be treated as the inference environment.

All hardware-dependent operations must run on the remote host, including:

- device detection
- driver / firmware inspection
- MUSA SDK inspection (secondary platform: CANN inspection)
- torch_musa execution (secondary platform: torch_npu execution)
- vLLM-MUSA execution (secondary platform: vLLM-Ascend execution)
- model loading
- inference
- benchmarking
- device utilization monitoring
- device memory measurements
- profiling

Model files remain on the remote environment.

Do not copy large model weights to the local machine unless explicitly
requested.

Do not assume that a model path visible in documentation exists.
Discover and validate the actual model path before using it.

The discovery items listed in section 3 must be discovered **on the host that runs the
inference**, using that platform's own tools. The concrete key names are an implementation
detail of the harness (`scripts/inspect_env.sh`); the contract is the item list, not the
spelling of the keys.

An item that cannot be measured must be reported as missing (`null` / `NOT_MEASURED`). It must
never be guessed, copied from documentation, and never filled with a value that belongs to the
other platform — a fingerprint carrying the wrong platform's tool name is worse than an absent
fingerprint, because it reads as evidence.

All experimental results included in the paper must originate from actual commands executed on
the remote environment.

Local or synthetic benchmark results must never be represented as hardware measurements of
either platform.

Declaring `runtime.execution_environment` names **where** the inference runs. The declaration is
not evidence by itself: a remote declaration still requires a fingerprint collected on the
executing host that confirms the device is available (section 10).

---

# 附录 A：仓库操作契约

本附录规定在本仓库里**怎么干活**。它服务于正文；与正文冲突时以正文为准。

## A1 三条硬规则（不可协商）

1. **`results/raw/**` 只读。** 原始数据一旦落盘不得修改、覆盖、重命名。任何清洗、聚合、重算的产物写进 `results/processed/`，并在文件名或 meta 中标注来源 run_id。
2. **任何进入结论的数字，必须可溯源。** 必须能指向 `EXPERIMENTS.md` 中的一条登记记录，该记录包含 run_id、配置路径、重复次数与离散度。**且该 run 的数据标签必须是 `MEASURED`**（正文第 10 节）。
3. **单变量原则。** 一次实验只改一个维度。跨维度组合（如「量化 + 分块预填充」）必须作为独立实验单独登记，不允许把两个变量的收益混算成一个数字。

## A2 开工前必做

- **先确认你在哪台机器上。** 本机（开发机）**不是**推理环境，只允许跑离线路径：
  `--dry-run`、单元测试、聚合与导出的干跑。任何硬件相关操作——设备探测、驱动与
  MUSA SDK / CANN 检查、`torch_musa` / `torch_npu` 执行、模型加载、推理、基准、
  设备利用率监控、设备显存测量、profiling——必须在**远程 MTT S4000 主机**上执行
  （正文 `Remote Execution Environment` 一节）。
- 读 `RESEARCH_LOG.md` 末尾 3 条，确认当前处于哪个阶段（正文第 21 节）、上一个结论是什么。
- 读 `EXPERIMENTS.md` 中状态为 `pending` / `running` 的条目，避免重复跑同一条。
- 跑 `bash scripts/inspect_env.sh`，确认环境指纹与上次实验一致；若不一致，必须在日志中记为环境变更事件。
  **指纹必须在实际执行推理的那台主机上采集**：在本地采一份远程主机的指纹不成立，
  因为它描述的不是跑推理的那台机器。
- 若本次要推翻已有结论，先确认该结论对应的 run 是否仍在 `results/raw/` 中可复现。

## A3 代码约定

- **包布局**：源码在 `src/ascend_opt/`（src-layout）。导入统一写 `import ascend_opt`，不写相对路径 hack。
- **策略类统一接口**：所有策略（调度 / 缓存 / 分块 / 控制）必须
  - 继承 `ascend_opt` 中对应的 Policy 基类；
  - 用 dataclass 声明 `XxxConfig`，字段带默认值与单位注释；
  - 提供 `name()` 与 `describe()`（返回可直接写进 `meta.json` 的纯字典）；
  - 是**纯函数式决策**：给定输入返回决策，不产生副作用。
  - 被协调的子控制器（`ChunkController` / `AdaptiveBatchController`）刻意**无状态**：
    可变状态统一由 `ControllerState` 持有，避免出现两个真相来源。
- **硬件边界**：策略类与 `workload.py` / `metrics.py` / `monitor.py` / `provenance.py`
  **禁止**直接调用硬件 SDK（torch_npu、ACL、vllm-ascend 运行时）。
  硬件交互只允许出现在 `benchmark/` 与 `scripts/`，且**只能在权威执行主机上执行**
  （本项目为远程 MTT S4000 主机；次平台为昇腾主机）。
  在本机跑 `benchmark/` 的真机路径，得到的不是"少了环境的实测"，而是
  `HARDWARE_REQUIRED` 标签——它不是降级的数据，是**不可用**的数据。
- **不可测量必须显形**：任何取不到的观测值保持 `None` / 写缺失标记，
  **禁止填 0**。0 会被读成"测出来是 0"，而缺失是"没测"——两者语义相反。
- **可离线验证**：新增策略必须能在 `--dry-run`（无 NPU 的模拟路径）下跑通，
  并在 `tests/` 中补一条对应测试。
- **配置驱动**：实验参数一律通过 `configs/` 下的 YAML 注入，不硬编码在脚本里。
  命令行只允许覆盖少量显式声明的开关。配置段的**未知字段直接抛错**，不静默忽略。

## A4 实验流程（强制）

1. 在 `EXPERIMENTS.md` 新增一行，状态 `pending`，写好假设与预期。
2. 在 `configs/<维度>/` 下新建或复用配置，配置文件名即实验的 config_stem。
3. 起 run，run_id 形如 `20260918-143012-shared_prefix_4k-prefix_cache`。
4. 跑完把状态改为 `done` 或 `failed`；`failed` 也必须记录失败原因与报错摘要。
5. 若结果与预期不符，在 `RESEARCH_LOG.md` 记一条，并明确：是测量问题、环境问题，还是假设本身不成立。
6. 重复次数不足 3 次的 run，**不允许**对外表述为结论。

**每次实验的最小记录字段**：run_id / 日期 / 工作负载 / 配置路径 / 硬件与环境指纹 / 数据标签 / 重复次数 / 观测指标（P50·P95·P99）/ 离散度 / 结论一句话 / 状态。

## A5 测量纪律

- 预热必须在计时窗口**之外**：按时间预热（如 ≥30s）而非固定迭代数，等待 DVFS 爬坡稳定。
- 计时窗口内不得进行内存分配、JIT 编译、图捕获。
- 区分三类抖动：预热未完成的漂移、单次停顿（outlier）、系统性降频。前者修测量，后者修判据。
- 用稳健 CV（1.4826·MAD / median）替代普通 CV；普通标准差对 outlier 毫无抵抗力。
- 允许出现「本次测不出结论」这一状态，但必须与「测出了坏结论」分开表述。
- **TPOT 必须与 stall 并列报告**：TPOT 是每请求的均值，一次数秒级停顿会被
  `gen_len` 除开而在均值里消失；`stall`（单请求 token 间隔的 P99）才是
  用户实际感受到的卡顿。

## A6 写作约定

- `paper/` 只写有 `EXPERIMENTS.md` 支撑的内容，禁止先写结论再补实验。
- 所有图表由 `benchmark/` 或 `scripts/export_results.sh` 生成，产物落在 `results/figures/` 或 `paper/figures/`，并在图注中写明数据来源 run_id 与数据标签。
- 引用外部工作一律登记到 `SOURCES.md`，未核验的条目标记 `status: unverified`，不得带入正文。
- 正文第 16 节要求区分"文档行为 / 我们的实现 / 我们的假设 / 我们的观测 / 我们的解释"——
  写作时逐句自问这句话属于哪一类。

## A7 禁止事项

- 禁止修改或删除 `results/raw/` 下任何文件。
- 禁止编造实验结果，禁止让非 `MEASURED` 的数据进入表格或结论（正文第 10 节）。
- 禁止把单次运行的数字写进结论。
- 禁止在未登记实验的情况下直接改 `README.md` 中的性能数字。
- 禁止为「让图好看」而挑选 run、裁剪区间或改口径（口径变更必须在 `RESEARCH_LOG.md` 中声明并标注影响范围）。
- 禁止把已有的 vLLM / CANN / MindIE 功能当作本文的贡献（正文第 16、17 节）。
- **禁止把非 `MEASURED` 的数字说成"实测"。** 本机跑出的结果、`--dry-run` 的模拟结果、
  任何合成数据，在论文、README、图注、汇报、口头描述中都不得使用"实测""实测得到"
  "我们测到"这类表述（正文 `Remote Execution Environment` 一节）。
  论文中的每一个结果都必须来自**权威执行主机**上真实执行过的命令；
  该主机的平台由规范末节与 `runtime.execution_environment` 声明共同确定
  （本项目为远程 MTT S4000 主机）。
