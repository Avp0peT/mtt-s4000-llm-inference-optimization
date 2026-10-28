# PROJECT_MAP — 项目导航（AI / 人类单一入口）

> **本文件是导航层，不是权威。** 任何**口径、数字、纪律**以它指向的权威文档为准。
> 效力顺序：`AGENTS.md` **正文 1–21 节** > `README.md` > 其余。
> （`AGENTS.md` 附录 A1–A7 是**旧契约**，与正文冲突时以正文为准。）
>
> **维护义务（本文件唯一的纪律）**：新增/删除**脚本、模块、文档**时更新对应表格。
> 表格里只写**一行用途 + 何时用它**；**不复述**规则内容 —— 复述会造出第二处真相来源。
>
> 生成/核对时间：2026-09-29。规模：**16 份**顶层文档（含本文件，≈1.33 MB）、61 个脚本、
> 19 个 `src` 模块、9 个 `benchmark` 模块、74 个测试文件、315 个 run 产物目录。

---

发布阅读入口：[工作与状态](docs/PROJECT_STATUS.md)、[发布范围](docs/PUBLICATION.md)、[发布验证](docs/PUBLICATION_VALIDATION.md)。整理日期不等于新实验日期。

## 0. 30 秒理解

| 问题 | 答案（一句话 + 去哪看） |
| --- | --- |
| **这是什么** | 论文《面向 MTT S4000 的请求感知型 LLM 推理部署优化研究》的工程与证据仓库。**主体平台 = MTT S4000（MUSA）**，昇腾为次平台；两栈不可互换/不可比 → `AGENTS.md` §1–§3 |
| **核心问题** | 负载变化时，**自适应控制器能否优于固定静态配置**？贡献边界 = **这些机制如何随负载协调**，对照必须是**静态 vs 自适应** → `AGENTS.md` §2 |
| **现在在哪** | 见下方「当前状态」；逐日细节在 `RESEARCH_LOG.md` **尾部**、会话登记在 `OPTIMIZATION_TEST_SESSION.md` §0 |
| **下一步** | 离线执行形状修复与真机集中清单：`NEXT_BATCHES_PREREG.md` §13、`handoff/optimization_20260929/REMOTE_EXECUTION_PLAN_20261003.txt`；本轮不连接远端 |
| **红线（不可违反）** | §7 本文件（**指向，不复述**） |
| **环境** | 权威推理环境 = **远程 MTT S4000 主机**；本机只跑离线路径 → §9 本文件 |

### 当前状态（**2026-10-03**，随会话推进请更新此块）

> **最新恢复入口**：本轮修复了执行形状、步后队列、数值验收角色及整步成本排序。
> 新成本臂使用精确整步执行契约；旧 v7 未验证该契约，明确回退。
> 已有 S01RECHECK 不重跑；真机数值、定点标定/独立验证及三项 pilot 留待集中执行。
> 下面阶段计数与早期阻塞保留为历史概览，不替代 `NEXT_BATCHES_PREREG.md` §13。

- 阶段：PHASE 0 🟢 / 2 🟢 / 5 🟡（策略层闭环已建成、真机未验）；**PHASE 1 🟡 引擎待定**；3、4、6–9 ⬜
- 证据存量：**120 份 `MEASURED`**（`P1S3` 36 + `P1S3B` 36 + `P2` 12 + `P1S3D` 36）
  ＋执行优化线 `EXECOPT3` 12 / `EXECOPT4` 12 / `EXECOPT5` 10
- **可用结论只有 C1/C2**（分块 vs 不分块 TTFT↔stall 权衡，差 40%~72%）；`adaptive` 臂**至今不可用**
- 已裁定：B2 门限形式（`gate_mode`，默认 `absolute`，推荐 `normalized`）、**B1 五问**（OPT-03R）、
  **B4 小 chunk 可预测性 = 能**（128/192/256）、**S04 收尾**（默认关、停跑同协议批次、不扩混长、不进图捕获）
- ★ **S04 已归档为"限定实现与负载的局部负结果"**（尾段 `+70.9%`，两批独立一致）——详见
  `EXPERIMENTS.md` §3.8 与 `paper/sections/08_limitations.md` §8.6
- ★★ **证据资格分层已建立**（T1/T2/T3，`EVIDENCE_INDEX.md` §5）：执行优化线**三个真机性能批次
  都不是 T1**（`EXECOPT3/4` 有环境无源码；`EXECOPT5` 有源码无环境）⇒ **都只能进 T2**
- ★ **环境绑定 = 独立的一类边界**：产物指纹**不一定描述跑它的那台机器**（实测一例，10/10 份）；
  修法已落地（批次前置重采指纹 + 校验容器/设备身份），但**不补齐历史证据**
- **未闭**：`384` 未采样；§9.5（中间点被选中未证）；B1 **不可冻结**；**FZ-1 / FZ-2 两处语义待冻结**
- ⛔ **不得**开始 `OPT-04`（依赖 B1–B4 全部就绪）
- ✅ **S01 的独立复核已完成（2026-10-03，`S01RECHECK`）**：TTFT p50 **−4.31%（5/5 同向）**、
  p95 −4.13%、p99 −4.04%；TPOT **不足以判定**；吞吐改善 **4/5**；
  **SLO 达标率两臂同为 2/64 = 3.125%**；10/10、`complete=true`、29.0 min；
  **四类身份全闭合 ⇒ T1（本线第一个）**。详见 `EXPERIMENTS.md` §3.9
  ⚠️ **2026-10-03 分析口径更正**：初版把达标率写成 1.000、吞吐方向写成"4/5 更慢"，
  均已修正（原始数据未改）。判据**随分析脚本版本化**，**非**独立封存预注册。
- ⛔ **本次会话到此停止**：**未**启动 25 项机制批次、**未**动 S04/图捕获
- ⏭️ **下一步 = 下一批协议（尚未开跑）**：围绕真实问题「TPOT 多数超过 100 ms、
  SLO 达标率 3.125%」；**先做最小可辨别机制 pilot**，不上 5 臂 × 5 轮。
  开跑前先离线统一：成本模型（**实况是三版并存 v5/v6/v7**，见 `NEXT_BATCHES_PREREG.md` §9.2）、
  `static_small` 定义、负载与 FZ-1/FZ-2 语义。
  入口 `NEXT_BATCHES_PREREG.md`

> 📖 **本块只放定位级判断**。数字走 `EXPERIMENTS.md`；恢复指针走
> `handoff/optimization_20260929/STATE.json`；"哪些数字能进论文"走 `EVIDENCE_INDEX.md` §5。

---

## 1. 新会话冷启动：按这个顺序读

| # | 读什么 | 为什么 / 怎么读 |
| --- | --- | --- |
| 1 | **本文件 §0** | 30 秒知道在做什么、在哪 |
| 2 | **工作区级** `st/.workbuddy/memory/MEMORY.md`（约 11 KB，**整读**） | 项目**长期约定**：红线、口径、未入文档的约定。⚠️ 项目内 `.workbuddy/memory/MEMORY.md`（约 24 KB）是**另一份**，两份同日内容不同 —— 见 **§9.3** |
| 3 | **工作区级** `st/.workbuddy/memory/PLAYBOOK.md` | **排障/环境/验证手法**速查。约 45 KB，**按节读**。⚠️ 它在**仓库外**（工作区根），不在项目目录 |
| 4 | `RESEARCH_LOG.md` **尾部 200 行** | 最近发生了什么。⚠️ 248 KB **不要整读**，只读尾部或按日期 grep |
| 5 | `AGENTS.md` **正文 §1–§21** | 研究规范（要动手前读）。⚠️ 附录 A 是旧契约 |
| 6 | `README.md` §3/§6/§7/§8/§9 | 负载维度 / 指标口径 / 标签准入 / 数据完整性 / 阶段与环境 |
| 7 | 按任务查 **§2 文档路由表** | 只在需要时按需读 |

> ⚠️ **`HANDOFF.md` 已于 2026-09-29 声明被取代**（快照停在 2026-09-24，落后 5 天）。
> 它**不是**权威，且**未被 git 跟踪**。新会话请用本文件，不要用 `HANDOFF.md`。

---

## 2. 文档路由表：我要知道 X → 读 Y

| 我要知道 | 权威文档 | 体量 | 读法 |
| --- | --- | --- | --- |
| 研究问题 / 目标 / 技术栈 / 方法 / 指标口径 / 红线 | **`AGENTS.md`** 正文 §1–§21 | 26 KB | 整读；**附录 A1–A7 是旧契约** |
| 工程与口径：负载维度、指标定义、**标签与结论准入**、数据完整性、阶段与环境 | **`README.md`** §3/§6/§7/§8/§9 | 69 KB | 按节读 |
| **某条结论的出处**（唯一入口） | **`EXPERIMENTS.md`** §1 登记 / §3 结论 / §4.3 前置条件与标定 | 249 KB | **按节 grep**；只认 `MEASURED` 行 |
| 按时间发生了什么（含**失败与否定结果**的成因） | **`RESEARCH_LOG.md`** | 248 KB | **只读尾部**，或 `grep -n "<日期>\|<主题>"` |
| **已知缺口与限制**（不掩盖） | **`EVIDENCE_INDEX.md`** §3 | 26 KB | 整读 §3 表格 |
| ★ **哪些数字能进论文**（证据资格 T1/T2/T3） | **`EVIDENCE_INDEX.md`** **§5** | 同上 | 整读 §5 两张表；机械核对见 `tests/test_batch_evidence_eligibility.py` |
| ★ **下一批实验怎么跑**（配置 / 顺序 / 验收门 / 预算口径） | **`NEXT_BATCHES_PREREG.md`** | 约 12 KB | 整读；它**只含准备、不含测量值** |
| ★ **待冻结的调度语义**（会改变研究问题；裁定前不得开跑） | **`NEXT_SESSION_TASKS.md`** 的 **FZ-1 / FZ-2** | — | `grep -n "FZ-1"` |
| ★ **本地版本是否已部署**（不是"远端戳代表当前源码"） | `scripts/deploy_status.py`（离线，退出码 1 = 需重新部署） | — | 直接跑；记录在 `handoff/.../DEPLOYMENT.json` |
| ★ **论文哪一节写了什么 / 怎么重建正文** | **`paper/sections/*.md`**（权威源）→ `scripts/build_manuscript.py` | — | 改 sections，再构建；**不直接改** `paper/manuscript.md` |
| 真机前向成本画像与标定纪律 | **`COST_PROFILE_S4000.md`** | 13 KB | 整读；§6 = 会话内对照纪律 |
| 优化测试会话：登记表 / 共同执行要求 / 各会话执行记录 | **`OPTIMIZATION_TEST_SESSION.md`** §0/§1/§2A–2C | 25 KB | §0 找入口，§2x 读执行记录 |
| 路线图：每条路线的定义与验收 | **`OPTIMIZATION_ROADMAP_SESSIONS.md`** §2/§5 | 14 KB | 整读 |
| 八个执行优化候选的技术依据与实施建议 | `OPTIMIZATION_IMPLEMENTATION_PLANS.md`（建议材料） | 约25 KB | 选择优化方向时按方案读，现状回查源码/证据 |
| 交给其他软件的启动文本、分会话指令与计划 | `handoff/optimization_20260929/00_START_HERE.txt`（交接入口） | 12个会话 | 首次交接先读入口与计划；状态快照不替代实验登记 |
| 下一轮外包：成本预测与实际执行形状对齐 | `handoff/optimization_20260929/NEXT_SESSION_EXECUTION_SHAPE_20261003.txt` | 单会话 | 离线修复、真实缺失形状清单与补标定准备；不启动远端任务 |
| 本轮离线交付 | `handoff/optimization_20260929/OFFLINE_EXECUTION_SHAPE_20261003.txt` | 单会话 | 修复、证据、测试及尚未完成的硬件验证 |
| 集中真机流程 | `handoff/optimization_20260929/REMOTE_EXECUTION_PLAN_20261003.txt` | 单会话 | 新环境验收、整步标定、验证、派生配置和三项 pilot |
| 计划与真实前向的共享形状 | `src/ascend_opt/execution_shape.py` | 模块 | 参与请求、逐请求 KV、分组与精确整步查表键 |
| 新标定探针与模型导出 | `scripts/execution_shape_probe.py` / `scripts/export_execution_shape_model.py` | 脚本 | 先 dry-plan；远端采集，冻结训练表，再独立验证 |
| 新模型的 pilot 配置派生 | `scripts/prepare_execution_pilot.py` | 脚本 | 验证模型后生成全新配置目录，不覆盖 prereg 配置 |
| 16 步执行计划 + **历次完成记录** + **B1 预注册草案** | **`NEXT_SESSION_TASKS.md`** | **385 KB** | ⚠️ **绝不整读**：§1 定位步骤 → `grep -n "^## "` → 按节读；草案在**文末** |
| 研究定位收敛 / 四层证据分级 / 主指标 | **`RESEARCH_PLAN.md`** | 18 KB | 整读 |
| 文献入口 | **`SOURCES.md`** | 16 KB | 整读 |
| PHASE 1 验收判据（**两代**，以第二代为准） | `VALIDATION_PLAN_PHASE1.md` §7.3 | 23 KB | 按节读 |
| PHASE 1 性能矩阵（历史阶段产物） | `PERFORMANCE_MATRIX_PHASE1.md` / `…1B.md` | 29 KB / 25 KB | 按需；**不得**覆盖 `EXPERIMENTS.md` 的结论 |
| **交接导航**（**已过期，勿用**） | ~~`HANDOFF.md`~~ | 35 KB | ⛔ 2026-09-24 快照，落后 5 天 |
| 论文工作稿装配与占位 | `paper/manuscript.md`、`paper/sections/*`、`scripts/list_paper_placeholders.py` | — | 见 §3 论文组 |
| 排障 / 环境 / 验证手法 + 逐日日志 | `st/.workbuddy/memory/PLAYBOOK.md`（**仓库外**） + 两边 `YYYY-MM-DD.md` | 45 KB + 每份 1–180 KB | PLAYBOOK 按节；日志只读当日。⚠️ 两处 memory 见 §9.3 |

---

## 3. 文档效力分层（决定"能不能引用"）

| 层 | 文档 | 能不能作为结论/数字来源 |
| --- | --- | --- |
| **① 权威规范** | `AGENTS.md` 正文、`README.md` | ✅ 口径与纪律以它们为准 |
| **② 权威记录** | `EXPERIMENTS.md`（登记行 + `MEASURED`）、`RESEARCH_LOG.md`、`EVIDENCE_INDEX.md` | ✅ **结论只能出自这里**，且须落到一条表格行 |
| **③ 索引/导航** | **本文件**、各文档的 §0 登记表 | ❌ **只指路**；数字不得引用 |
| **④ 日志（append-only）** | `RESEARCH_LOG.md`、`.workbuddy/memory/YYYY-MM-DD.md` | ⚠️ 只增不改；**不得**用它替代 ② |
| **⑤ 快照（会过期）** | `HANDOFF.md`（已取代）、`PERFORMANCE_MATRIX_PHASE1*.md`、`VALIDATION_PLAN_PHASE1.md` | ❌ 历史阶段产物；引用前查是否已被 ② 取代 |
| **⑥ 临时/诊断** | `.workbuddy/memory/*.sh`、`*.py`、`.workbuddy/tmp/` | ❌ 排障脚本，**不是证据**（但 `.sh`/`.py` 按约定入库） |

**探针路径（`results/raw/_probe/**`、`results/raw/_costcalib/**`）不盖四标签、不进结论表。**

---

## 4. 脚本索引（`scripts/`，61 个）

> 调用约定：一律用 **venv 解释器** `C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe`；
> 远程脚本见 §9。多数脚本支持 `--help`。

### 4.1 成本标定与模型（第 11–12 步 / OPT-02/03）

| 脚本 | 用途 | 何时用 |
| --- | --- | --- |
| `costcalib_grid.py` | 标定**形状网格**的加载/校验/展开 | 要跑标定前 |
| `fwd_costcalib_probe.py` | 设备侧成本标定探针（协议 `cost-calib-v1`，真机） | 采标定数据 |
| `costcalib_fit.py` | 标定数据的**拟合与评估** → `results/processed/costcalib_eval*.json` | 采完后 |
| `export_cost_model.py` | 把标定**固化成模型文件** `configs/cost_calib/model_v*.json` | 需要新模型时 |
| `check_model_coverage.py` | 成本模型**覆盖率**（头部指标） | 换模型/负载前 |
| `sweep_lookup_buckets.py` | 分桶查表的**取舍曲线**（桶宽↔覆盖率↔保守代价，纯本地） | 调分桶参数 |
| `eval_lookup_deployment_path.py` | 按**部署路径**重评查表模型（含可复用函数：键/分桶/划分/度量） | 评估查表 |
| `slo_reach_frontier.py` | **SLO 可达边界**：哪些 `(batch,prefill,kv)` 够得着 SLO | 定工作带（B1 用） |
| `decompose_step_overhead.py` | 分解逐步缺失开销（`fwd_ms/pack_ms/readback_ms`） | 口径排查 |
| `calibrate_slo.py` | 把 `slo.tpot_ms` 标定到"能分辨" | 换 SLO 时 |
| `calibrate_controller_gate.py` | 把**控制器门限**标定到信号分布 | 换门限时 |
| `verify_calib_semantics.py` | 两个标定版本间的**语义核验** | 改口径后 |
| `gen_shape_grid_v7.py` / `gen_shape_grid_v8.py` | 生成形状网格（v8 = v7 + pf192）。⚠️ **网格不得原地改**，带 `--check` | 要加密采样点 |
| `analyze_b4_small_chunk.py` | **OPT-03/B4**：小 chunk 可预测性切片分析 | 小 chunk 相关 |
| `analyze_opt01_s14b2.py` | **OPT-01**：零成本只读分析既有 `S14B2` 产物 | 复用已有证据 |
| `_b4_inspect.py` | 打印标定探针产物的结构摘要（远端用，避开多层引号） | 远端核查 |

### 4.2 配置生成（**派生，不手写**）

| 脚本 | 用途 |
| --- | --- |
| `gen_phase1_step3_configs.py` | 第③步性能矩阵配置（派生自平台基线） |
| `gen_phase1_step6_configs.py` | 第⑥步静态扫描配置 |
| `gen_phase1_s6_configs.py` | S6 服务收益档三臂配置 |
| `gen_phase1_step15_configs.py` | 第 15 步消融配置 |
| `gen_phase1_d1_configs.py` / `gen_phase1_d1_scan_configs.py` | D1 配置集 / D1 低并发扫描配置 |

### 4.3 运行与编排（真机批次）

| 脚本 | 用途 | 何时用 |
| --- | --- | --- |
| `run_step3_matrix.py` | 按**穿插顺序**跑第③步矩阵 | 起矩阵批次 |
| `run_step6_scan.py` | 第⑥步静态扫描运行器（逐格 dry-run + 汇总表） | 起扫描 |
| `batch_lock.py` | 批次**互斥锁 + 生命期监视**（纯逻辑，可本机验证） | **每批起跑前** |
| `watch_batches.py` | 双批次看护（回拉补全 + 进度/到点自动分析） | 无人值守长跑 |
| `run_tests.py` | 两级测试入口（纯逻辑 / 含 CPU 微型模型的数值） | 跑测试 |

### 4.4 远程执行与回拉（§9 配用）

| 脚本 | 用途 |
| --- | --- |
| `remote_run.py` | **远程执行/传文件薄封装**（paramiko）。子命令 `exec/put/get/push/pull/stamp`。⚠️ 用 venv 解释器；⚠️ MSYS 路径转换坑见 §9 |
| `pull_incremental.py` | **增量取回**（只取本地没有的 run 目录 + 对数量） |
| `pull_chunked.py` | 分块回拉远端大文件 |
| `pull_s14b2_exec.py` | `S14B2` 回拉补全（走 `exec` 通道 + 远端分块 base64） |
| `decompress_pulled.py` | 解压已回拉的 `*.gz` |
| `remote_probe_run.py` | 只读：打印一个 run 目录的身份与汇总字段 |
| `check_remote_busy.py` | 远端并发检测（判是否真有基准在跑） |
| `check_cpu_stability.py` | **CPU 稳定性探针**（三态 rc=0/1/2）。⚠️ 真机批次前置 |
| `diag_remote_reachability.py` | 远端可达性诊断（把 PLAYBOOK §5.1 人工步骤做成一条命令） |

### 4.5 分析（**只读产物，不改任何东西**）

| 脚本 | 用途 |
| --- | --- |
| `analyze_step14.py` | 第 14 步四臂对比分析 |
| `analyze_s10_surface.py` | 第⑩步静态响应面 → 按**预先登记**规则冻结 `Best Static` |
| `analyze_s9_ab.py` | 第⑨步干预实验判据口径 A/B 对照 |
| `analyze_scan_batch.py` | `S14S` 扫描批：机械套用冻结判据 |
| `analyze_s14b3.py` | 批次 3（`lenmix`）覆盖率与回退成因分解 |
| `diagnose_scheduling.py` | **证据链三诊断**（预测误差 ↔ 服务收益）。⚠️ 遇截断 `meta.json` 会**响亮跳过 + rc=1**（不是崩） |
| `report_stages.py` | 按到达阶段 × 墙钟窗口两套口径报告动态负载 |
| `inspect_step3_runs.py` | 第③步产物的**重复独立性**与离散度（含 `--verify-config`） |
| `inspect_timing_scope.py` | P2 计时口径修复的**前后对照** |
| `inspect_s14d_execution.py` | D0 判决层 vs 实际执行并排 |
| `decouple_grid_split.py` | 2×2 解耦（新旧网格 × 哈希/秩保证划分） |
| `pick_d1_min.py` | 按**预注册规则**机械选 `chunk_size_min`（把判断从人手里拿走） |
| `bench_control_overhead.py` | 控制开销离线微基准 |

### 4.6 检查与闸门（判据的机器执行）

| 脚本 | 用途 |
| --- | --- |
| `d1_gate.py` | D1 可行性闸门（**三态** `PASS`/`FAIL`/`INCONCLUSIVE`） |
| `s6_benefit_gate.py` | S6 服务收益档闸门：**机械执行**预注册判据 |
| `check_step3_configs.py` / `check_step3b_reachability.py` | 第③步配置预检 / **开跑前**确认门限真会被触发 |
| `check_opt02_acceptance.py` | OPT-02 可复算验收（B2 × B3） |
| `check_traceability.py` | 追溯链自查（机械可查部分自动化） |
| `validate_fp16_path.py` | **PHASE 1 第②步**：真实 FP16 路径数值等价性（远程 S4000 上跑） |

### 4.7 论文、报告与图表

| 脚本 | 用途 |
| --- | --- |
| `build_manuscript.py` | `paper/sections/*.md` → `paper/manuscript.md`（可整篇阅读） |
| `list_paper_placeholders.py` | 清点论文占位标记（还剩多少坑、在哪个文件） |
| `emit_experiment_rows.py` | 从**产物**生成 `EXPERIMENTS.md` 登记行（数据驱动，不手抄） |
| `plot_step16_decouple.py` | 第 16 步 · P27 的两张图（只读、可重建） |

---

## 5. 配置索引（`configs/`）

| 目录 | 管什么 | 文件数 |
| --- | --- | --- |
| **`cost_calib/`** | 成本标定：`shape_grid_v1..v8.yaml`（网格）+ `model_v1..v7.json`（导出模型）。⚠️ **网格不得原地改**（另开 vNN）；⚠️ `model_v8.json` **尚未导出** | 13 |
| **`mthreads/`** | **真机批次配置**，按阶段分子目录：`phase1_step3/3b/3c/3d`、`step6`、`step13`、`step14/14b`、`step15/15_v5`、`d1`、`d1_scan`、`s6`；平台基线 `s4000_fp16.yaml` | 14 |
| `adaptive/` | 自适应臂配置 | 1 |
| `baseline/` | 基线配置 | 1 |
| `chunked_prefill/` | 分块预填充 | 2 |
| `prefix_cache/` | 前缀缓存 | 1 |
| `quantization/` | 量化（论文范围外） | 1 |
| `graph_mode/` | 图捕获/锁频探查（部署线 A2） | 1 |

> ⚠️ **部署戳 `digest_scope` 不含 `configs/`** ⇒ 配置不进源码戳，逐 run 核对靠
> `inspect_step3_runs.py --verify-config`。
> ⚠️ **改 `controller.*`（或 `benchmark/real_backend.py`）= 改调度行为 ⇒ 必须重跑**（P4 裁定）。

---

## 6. 代码模块索引

### 6.1 `src/ascend_opt/` —— **纯逻辑层**（禁止 torch_npu / ACL / vllm-ascend）

| 模块 | 职责 | 大小 |
| --- | --- | --- |
| `controller.py` | **运行期自适应控制器**（阈值臂；含 `gate_mode` 绝对/归一化） | 78 KB |
| `cost_policy.py` | **成本控制器**（用预测步成本在有限候选里选动作；含**双预算** `prefill_deadline_ms`） | 25 KB |
| `cost_model.py` | 成本模型加载/指纹比对/范围检查/预测 | 12 KB |
| `metrics.py` | 指标定义与聚合 | 33 KB |
| `provenance.py` | **数据来源标签与可复现性元数据** | 66 KB |
| `workload.py` | 工作负载建模（含 `arrival` 模型：`closed`/`closed_jittered`/`poisson`/`burst`/`piecewise`） | 24 KB |
| `stage_report.py` | 动态负载分阶段报告 | 24 KB |
| `telemetry.py` | 遥测采集与落盘 | 18 KB |
| `chunk_policy.py` | 分块预填充策略 + **`ChunkLimits`（运行期上限的唯一致源）** | 10 KB |
| `scheduler.py` | 调度策略 | 11 KB |
| `slo_weave.py` | **SLOWeave 参考策略**（第 13 步最接近方法对照） | 9 KB |
| `monitor.py` | 状态观测（`WorkloadMonitor`） | 9 KB |
| `cache_policy.py` | 前缀缓存策略 | 8 KB |
| `arm.py` | 策略臂的**决策与应用**（两路径共用的唯一实现） | 6 KB |
| `step_planner.py` | 一步的调度决策（批预算→分块→缓存→prefill 分配） | 10 KB |
| `execution.py` | 一次基准执行的结果契约 | 4 KB |
| `grouping.py` | 实验分组唯一定义（什么算"同一个实验"） | 4 KB |
| `run_ids.py` | `run_id` 解析的**唯一**实现 | 4 KB |
| `__init__.py` | 包入口 | 6 KB |

### 6.2 `benchmark/` —— 硬件交互只在这里

| 模块 | 职责 | 大小 |
| --- | --- | --- |
| `real_backend.py` | **真实硬件执行后端**（进程内 Transformers）。⚠️ 含闭环补位点；**改它 = 新实验条件** | 118 KB |
| `run_benchmark.py` | **基准执行入口**（含 `--dry-run` 离线模拟、`simulate()`） | 75 KB |
| `preflight.py` | 导出前置校验（每个数字都要有据可查） | 35 KB |
| `collect_metrics.py` | `results/raw` → `results/processed` 收敛 | 28 KB |
| `workloads.py` | 工作负载**注册表** | 23 KB |
| `plot_p99.py` / `plot_s10_surface.py` | P99 条形图 / 静态响应面图 | 18 KB / 8 KB |
| `arm_factory.py` | 策略臂**构造**（模拟器与真机共用） | 6 KB |
| `sweep.py` | 参数扫描（1D/多维网格批量起 run） | 7 KB |

### 6.3 `tests/` —— 74 个测试文件

两级入口：`scripts/run_tests.py`（纯逻辑 / 含 CPU 微型模型数值）。
全量：`python -m pytest tests/ -q`（≈1355 passed / 39 skipped，**约 30 分钟**）。

---

## 7. 红线（**指向，不复述**）

| 红线 | 权威处 |
| --- | --- |
| `src/ascend_opt/` 是**纯逻辑层**，硬件交互只在 `benchmark/`、`scripts/` | `AGENTS.md` 附录 A3；`README.md` §9 |
| 运行期上限**只有一处权威** = `chunk_policy.ChunkLimits` 显式传入 | `AGENTS.md` 附录 A3 |
| 控制器**双向**且恢复目标 = **配置初值**（只收缩不恢复 = 棘轮） | `AGENTS.md` §12 |
| 信号要么可信、要么显式 `None`；**不可测的值保持 `None` 禁止填 0** | `AGENTS.md` §7；`README.md` §7 |
| `results/raw/**` **只读**；单变量；重复 <3 不出结论；负面结果同等登记 | `AGENTS.md` §10；`README.md` §8 |
| **标识符一律不改名**（`ascend_opt` / `ASCEND_*` / `Ascend`） | `.workbuddy/memory/MEMORY.md` |
| `is_remote` 是**三态**（判定必须写 `is False`，不是 `not is_remote`） | `.workbuddy/memory/MEMORY.md` |
| 探针路径（`_probe`/`_costcalib`）**不盖四标签、不进结论表** | `README.md` §7；`OPTIMIZATION_TEST_SESSION.md` §1 |
| 改 `controller.*` 或 `benchmark/real_backend.py` ⇒ **改调度 ⇒ 必须重跑** | `EXPERIMENTS.md` §4.3.4（P4） |

---

## 8. 目录布局

```
ascend-llm-inference-optimization/
├── AGENTS.md  README.md  RESEARCH_PLAN.md  SOURCES.md        # 规范 · 口径 · 定位 · 文献
├── EXPERIMENTS.md  RESEARCH_LOG.md  EVIDENCE_INDEX.md        # 结论登记 · 时序日志 · 缺口台账
├── COST_PROFILE_S4000.md                                     # 真机成本画像
├── OPTIMIZATION_TEST_SESSION.md  OPTIMIZATION_ROADMAP_SESSIONS.md  # 会话登记 · 路线图
├── NEXT_SESSION_TASKS.md                                     # 16 步计划 + 完成记录 + B1 草案（385 KB）
├── HANDOFF.md                                                # ⛔ 已取代（2026-09-24 快照）
├── PERFORMANCE_MATRIX_PHASE1.md / …1B.md  VALIDATION_PLAN_PHASE1.md   # 历史阶段产物
├── PROJECT_MAP.md                                            # ← 本文件
├── src/ascend_opt/          # 纯逻辑层（19 模块）
├── benchmark/               # 硬件交互 + 执行入口 + 指标收敛（9 模块）
├── scripts/                 # 61 个工具（见 §4）
├── configs/                 # 8 个子目录（见 §5）
├── tests/                   # 74 个测试文件
├── paper/                   # manuscript.md / sections/ / tables/ / figures/ + 三份 brief
├── results/
│   ├── raw/          (315)  # ⚠️ 只读。run 产物 + `_probe/` + `_costcalib/`
│   ├── processed/    (175)  # 分析表（可重建）
│   ├── scan_step6/    (67)
│   ├── figures/        (2)
│   └── _incoming/      (4)  # 中转
└── .workbuddy/
    ├── memory/              # ⚠️ 见 §9「两处 memory」
    └── tmp/                 # 一次性诊断脚本与中转（git 忽略）
```

---

## 9. 环境与远程执行

### 9.1 权威环境

| 项 | 值 |
| --- | --- |
| **权威推理环境** | **远程 MTT S4000 主机**（本机只跑离线路径，`local` **永不**产生 `MEASURED`） |
| 本机角色 | Windows 开发机：写代码、离线分析、dry-run |
| 远程凭据 | `~/.workbuddy/remote.env`（**绝不入库**） |
| 远程工具 | `scripts/remote_run.py`（子命令 `exec/put/get/push/pull/stamp`） |

### 9.2 必须避开的坑（**详见仓库外** `st/.workbuddy/memory/PLAYBOOK.md`）

| 坑 | 规则 |
| --- | --- |
| 跑远端 python 前 | 必须 `source /root/musa-envs/qwen310/activate_musa.sh` |
| 非交互/后台 | 必须 `bash -c` 而非 `sh -c`（`source` 非内建 ⇒ 激活**静默失败**） |
| 取模型 | `HF_ENDPOINT=https://hf-mirror.com`（`huggingface.co` 不可达） |
| `remote_run.py` | **必须**用 venv 解释器（裸解释器无 paramiko） |
| 远端绝对路径 | 本机原生程序前加 `MSYS_NO_PATHCONV=1`，或用 `//root/...` 形态。⚠️ 它会**同时关掉本地**路径转换 ⇒ 本地目标一律写 `C:/...` |
| 长跑任务 | 后台 + 轮询 `pgrep`，**不要**阻塞等待；**同一批目标路径只能有一个写者** |
| 真机批次前置 | ① `check_cpu_stability.py`（三态）② 无并发 ③ GPU idle ④ provenance stamp ⑤ 回拉预案 |

### 9.3 ⚠️ 两处 memory（**已知重复，待你裁定**）

本项目存在**两套并行的 memory**，同日文件内容**不同**：

| 位置 | 内容 | 是否入库 | 最近写入 |
| --- | --- | --- | --- |
| `ascend-llm-inference-optimization/.workbuddy/memory/` | 逐日日志 `2026-09-22/23/24/26/27/28.md`（最大 177 KB）、`MEMORY.md`(24 KB)、`EXPERIMENTS.bak.*`(232 KB)、`REPORT_*.md`、**约 100 个排障 `.sh`/`.py`/`.log`** | ✅ **入库**（`.gitignore` 第 64–73 行明确保护） | 2026-09-28 19:52 |
| `st/.workbuddy/memory/`（工作区级） | 逐日日志 `2026-09-18/20/21/22/23/24/25/26/28.md`、`MEMORY.md`(11 KB)、**`PLAYBOOK.md`(45 KB)** | ❌ 仓库外 | 2026-09-28 22:46 |

**同日重复**：`09-22`、`09-23`、`09-24`、`09-26`、`09-28` 在两边**各有一份且内容不同**；
`09-27` 只在项目内，`09-18/20/21/25` 只在工作区。
⇒ **风险**：同一日期有两条真相。新会话**默认读**工作区级（系统托管、含 `PLAYBOOK.md`），
但**项目内那份才是入库的**。**清理需你裁定**（我不擅自删改 memory）。

---

## 10. 本文件的核对方式（自检）

```bash
# 脚本数是否与 §4 一致（应为 61）
ls scripts/*.py | wc -l
# 模块数（应为 19 / 9）
ls src/ascend_opt/*.py | wc -l ; ls benchmark/*.py | wc -l
# 顶层文档数与体量
ls -la *.md | awk '{print $5, $9}' | sort -rn
# 配置子目录
ls -d configs/*/
```
