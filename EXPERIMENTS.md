# EXPERIMENTS

> ## 📖 怎么读本文件（**249 KB，不要整读**）
>
> | 我要 | 怎么做 |
> | --- | --- |
> | **查某条结论/数字的出处** | ★ 这是**唯一入口** → §1 登记行（看 `数据标签` 列） |
> | 看按维度的结论汇总 | §3 |
> | 看前置条件与标定（P1–P4 等） | §4.3 |
> | 定位章节 | `grep -n "^## \|^### " EXPERIMENTS.md` |
> | 项目总入口 / 一切索引 | → **`PROJECT_MAP.md`** |
>
> ⚠️ **只有 `MEASURED` 的行可以进入结论与论文表格**；其余标签（`NOT_MEASURED` /
> `HARDWARE_REQUIRED` / `TEST_DATA`）**原样保留**，用来区分"没测"与"测出来没用"。
> ⚠️ 新增登记行由 `scripts/emit_experiment_rows.py` 从产物生成，**不要手抄**。

实验登记表。**每条进入论文或 README 的数字，都必须能在本表找到来源。**

编号规则：`EXP-<维度缩写>-<三位序号>`，维度缩写取 `base` / `quant` / `pc` / `cp` / `graph` / `adap`。
run_id 规则：`YYYYMMDD-HHMMSS-<workload>-<config_stem>`。

状态取值：`pending` | `running` | `done` | `failed` | `abandoned`（abandoned 需写原因）。

**数据标签**（规范第 10 节）：`MEASURED` | `NOT_MEASURED` | `HARDWARE_REQUIRED` | `TEST_DATA`。
**只有 `MEASURED` 的行可以进入结论与论文表格。** 其余标签一律在此表中原样保留，
以便区分"没测"与"测出来没用"。标签由 harness 自动写入 `meta.json`，
登记时从产物抄录，不要凭印象填写。

占位约定：未测到的数字写 `-`，**不要写 0**。0 是测量值，`-` 是缺失，两者语义相反。

**登记不是形式，是导出前置校验的一条必查项**：`scripts/export_results.sh` 在拷贝任何
内容之前会调用 `benchmark/preflight.py`，逐项核对原始明细、环境指纹、有效重复（≥3）
与**本表是否登记了该 run_id**。任一项缺失即中止导出（退出码 4），并指名是哪个 run。
本仓库刻意把"记得登记"变成"不登记就导不出去"——因为漏登记不会让任何东西报错，
只会让一个数字在半年后变成无从追溯的断言。

---

## 0. 最小实验变体清单（规范第 5 节）

必须逐项覆盖，且**不得只报最终优化配置**。做完一项勾一项。

- [ ] **Baseline**（对照，`configs/baseline/default.yaml`）
- [ ] Baseline + quantization（`configs/quantization/w8a8.yaml`）
- [ ] Baseline + graph execution（`configs/graph_mode/acl_graph.yaml`）
- [ ] Baseline + Prefix Cache（`configs/prefix_cache/prefix_cache.yaml`）
- [ ] Baseline + Chunked Prefill（`configs/chunked_prefill/chunked_4k.yaml`）
- [ ] Baseline + selected combined optimizations（组合改动，**必须单独登记**）
- [ ] Baseline + proposed adaptive controller（`configs/adaptive/adaptive_v1.yaml`）

> **平台归属（2026-09-22 起）**：研究主体平台是 **MTT S4000**，因此论文的基线是
> `configs/mthreads/s4000_fp16.yaml`。上表列出的仍是**昇腾侧**的配置路径（`configs/*/`）；
> 主体平台上的对应变体**尚未派生**。在派生完成之前，本表不得被当作"研究主体平台的变体
> 清单"来读。两平台数字不得并表（平台维度见 `AGENTS.md` 第 10 节）。

> 组合改动（如「量化 + 分块预填充」）不得与单项收益混算成一个数字——
> 见 AGENTS.md 附录 A1 第 3 条。vLLM-Ascend 的现有能力不得作为本文贡献
> （规范第 16、17 节）。

---

## 1. 登记表

| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-base-001 | _待填_ | _axis_in2048_out128_c8_ | `configs/baseline/default.yaml` | 无（对照） | 0/3 | - | - | - | - | - | - | - | - | - | - | pending |
| EXP-cp-001 | _待填_ | _axis_in2048_out128_c8_ | `configs/chunked_prefill/chunked_4k.yaml` | chunk_size | 0/3 | - | - | - | - | - | - | - | - | - | - | pending |

> 说明：上表两行是**格式示例**，占位值（如 `_待填_`）必须被真实值替换后才算完成登记。
> 达标口径分三列（`slo_satisfaction_rate` / `slo_goodput_req_s` / `slo_goodput_tok_s`），
> 不合并成一列 `Goodput`——它们不是同一个量，合并后没人知道那列在说什么。
> 导出前置校验要求 `run_id` 作为**表格单元**出现在本表某一行里：正文提到不算登记。

### 1.1 主体平台（MTT S4000）第 ③ 步性能矩阵 — 2026-09-23

编号用 `1xx` 段，与上面昇腾侧的示例编号隔开。全部 `MEASURED`，
逐格 3 次重复、穿插执行（同格重复相隔 19.3 分钟）。
设计与完整读数见 `PERFORMANCE_MATRIX_PHASE1.md`。

**`run_id` 约定偏差（如实登记）**：本表顶部的规则写的是
`YYYYMMDD-HHMMSS-<workload>-<config_stem>`，而这批实际是
`P1S3-<workload>-<cell>`——刻意不带时间戳，这样同一格的 3 次重复才能落在
`<id>` / `<id>-r1` / `<id>-r2` 上并被聚合器识别成一组。
规则文字与实现不一致，**尚未统一**；此处按实际 `run_id` 登记（导出校验比对的是实际值）。

| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-base-101 | P1S3-phase1_anchor-static_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3/static_c1.yaml` | 无（本矩阵对照：分块预填充**关**、控制器关） | 3/3 | MEASURED | 448.50/448.77/448.82 | 61.23/61.27 | 62.54 | 15.55 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0005 / stall.p99 0.0035 | done |
| EXP-base-102 | P1S3-phase1_anchor-static_c2 | phase1_anchor（in2048/out128，并发 2，8 请求） | `configs/mthreads/phase1_step3/static_c2.yaml` | 无（本矩阵对照：分块预填充**关**、控制器关） | 3/3 | MEASURED | 856.76/856.87/856.87 | 77.14/77.15 | 78.51 | 24.01 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0000 / tpot.p50 0.0003 / stall.p99 0.0006 | done |
| EXP-base-103 | P1S3-phase1_anchor-static_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3/static_c4.yaml` | 无（本矩阵对照：分块预填充**关**、控制器关） | 3/3 | MEASURED | 1655.40/1655.52/1655.52 | 87.71/87.72 | 88.47 | 39.96 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0001 / stall.p99 0.0041 | done |
| EXP-base-104 | P1S3-phase1_anchor-static_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3/static_c8.yaml` | 无（本矩阵对照：分块预填充**关**、控制器关） | 3/3 | MEASURED | 2278.44/3254.37/3406.43 | 140.10/144.57 | 458.49 | 50.69 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0002 / tpot.p50 0.0006 / stall.p99 0.0002 | done |
| EXP-cp-101 | P1S3-phase1_anchor-static_chunk_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3/static_chunk_c1.yaml` | chunk.enabled True（相对 static 只改这一处） | 3/3 | MEASURED | 630.00/631.28/631.47 | 61.28/61.34 | 62.17 | 15.21 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0002 / tpot.p50 0.0009 / stall.p99 0.0063 | done |
| EXP-cp-102 | P1S3-phase1_anchor-static_chunk_c2 | phase1_anchor（in2048/out128，并发 2，8 请求） | `configs/mthreads/phase1_step3/static_chunk_c2.yaml` | chunk.enabled True（相对 static 只改这一处） | 3/3 | MEASURED | 814.19/1226.22/1402.54 | 85.76/85.99 | 222.21 | 21.84 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0004 / tpot.p50 0.0009 / stall.p99 0.0001 | done |
| EXP-cp-103 | P1S3-phase1_anchor-static_chunk_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3/static_chunk_c4.yaml` | chunk.enabled True（相对 static 只改这一处） | 3/3 | MEASURED | 960.72/3054.29/3344.61 | 112.02/114.47 | 271.31 | 32.76 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0004 / stall.p99 0.0000 | done |
| EXP-cp-104 | P1S3-phase1_anchor-static_chunk_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3/static_chunk_c8.yaml` | chunk.enabled True（相对 static 只改这一处） | 3/3 | MEASURED | 3929.58/7358.83/7680.14 | 153.85/166.15 | 293.57 | 41.46 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0000 / stall.p99 0.0008 | done |
| EXP-adap-101 | P1S3-phase1_anchor-adaptive_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3/adaptive_c1.yaml` | controller.enabled True（相对 static_chunk 只改这一处） | 3/3 | MEASURED | 629.86/630.55/630.55 | 61.32/61.42 | 62.34 | 15.20 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0001 / stall.p99 0.0036 | done |
| EXP-adap-102 | P1S3-phase1_anchor-adaptive_c2 | phase1_anchor（in2048/out128，并发 2，8 请求） | `configs/mthreads/phase1_step3/adaptive_c2.yaml` | controller.enabled True（相对 static_chunk 只改这一处） | 3/3 | MEASURED | 814.76/1225.48/1401.49 | 85.96/86.03 | 222.44 | 21.81 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0003 / tpot.p50 0.0008 / stall.p99 0.0009 | done |
| EXP-adap-103 | P1S3-phase1_anchor-adaptive_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3/adaptive_c4.yaml` | controller.enabled True（相对 static_chunk 只改这一处） | 3/3 | MEASURED | 960.49/3053.19/3343.65 | 111.99/114.45 | 271.37 | 32.77 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0002 / tpot.p50 0.0000 / stall.p99 0.0007 | done |
| EXP-adap-104 | P1S3-phase1_anchor-adaptive_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3/adaptive_c8.yaml` | controller.enabled True（相对 static_chunk 只改这一处） | 3/3 | MEASURED | 3942.65/7386.51/7709.59 | 154.64/166.90 | 295.05 | 41.24 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0015 / tpot.p50 0.0012 / stall.p99 0.0039 | done |

> 上表读数 = 3 次重复的**中位数**（原始三值见 `PERFORMANCE_MATRIX_PHASE1.md` §7.2）。
> `离散度(稳健CV)` 是**跨 3 次重复**的稳健 CV，不是 `summary.stability.*_robust_cv`
> （那是**一个 run 内跨 8 个请求**的离散度）——两者同名不同口径，不可混用。
> **达标率三列恒为 0.000 不表示"稳定"**：它是 `target_tpot_ms=50` 对 7B 不可达
> （见 `PERFORMANCE_MATRIX_PHASE1.md` §6.2），本矩阵中不携带信息。

### 1.2 主体平台（MTT S4000）第 ③ 步**修正版**矩阵（P1S3B）— 2026-09-23

编号接 `1xx` 段用 `11x`。全部 `MEASURED`，逐格 3 次重复、穿插执行，
设计与完整读数见 `PERFORMANCE_MATRIX_PHASE1B.md`。**36/36 成功，0 失败。**
产物：`results/raw/P1S3B-*`（36 个目录，144 个文件）。

**相对 P1S3 改了哪些配置值（逐字段，作为"这不是同一格的重复"的证据）**：

| 字段 | P1S3（缺陷版） | P1S3B（修正版） | 为什么改 |
| --- | --- | --- | --- |
| `chunk.chunk_size` | 512 | **1024** | 见下"同一批内部单变量边界" |
| `controller.chunk_size_init` | 512 | **1024** | 原值等于下界 ⇒ 被扫描维度一步也动不了（维度死锁） |
| `controller.chunk_size_max` | 16384 | **2048** | 上界远超实际有用区间，grow 可达无意义值 |
| `controller.target_tpot_ms` | 50.0 | **110.0** | 50 对 7B 不可达 ⇒ 控制器只收不放、退化成单向棘轮 |
| `controller.target_ttft_ms` | 2000.0 | **500.0** | 原值**高于四档实测 TTFT P90 全部取值** ⇒ grow 判据一次都不触发（这是修正方案最初漏掉的第四处） |

> ⚠️ **必须显形的三条**（缺任何一条，上表数字都会被误用）：
>
> 1. **`static_chunk` 基线不是 P1S3 那一格的复现**：它的 `chunk.chunk_size` 由 512 改成了
>    **1024**。因此"P1S3B 的 `static_chunk` vs P1S3 的 `static_chunk`"**不是单变量对照**，
>    两批之间不得直接比较分块基线。同一批**内部**的两组对比才是单变量：
>    `static` vs `static_chunk`（只改 `chunk.enabled`）、
>    `static_chunk` vs `adaptive`（只改 `controller.enabled`，两者 chunk_size 均为 1024）。
> 2. **计时口径是修复前的**：本批 `telemetry.jsonl` 的 `note` 形如
>    `prefill_tok=… decode=… forwards=… fwd_ms=… pack_ms=…`，**不含 `readback_ms`**，
>    而修复后的 `benchmark/real_backend.py` 必然追加该字段（第 1941 行）。故本批
>    `ttft_ms`/`e2e_ms` 记的是 **logits 就绪时刻**，未含采样与设备→主机取回，
>    **系统性偏早**，偏差随 batch 增大。⇒ 批**内**两臂同口径同方向，相对比较成立；
>    **绝对 TTFT/E2E 不得与修复后口径的运行混用**。偏差量级待补的小规模前后对照给出。
> 3. **版本溯源未满足**：36/36 的 `reproducibility.git_commit` = `NOT_MEASURED`
>    （字段存在、值显式为未测，并列入 `_missing_fields`），`model_revision` = `'main'`
>    （未解析到具体 revision）。**不得**用当前 HEAD 回填，须在新一轮运行中真正记录。
>
> 另：`run.slo.tpot_ms` 在本批**仍是 50**（只有 `controller.target_tpot_ms` 改成了 110），
> 因此**达标率三列依旧恒为 0.000、不携带信息**——与 P1S3 同一问题，尚未修。

| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-base-111 | P1S3B-phase1_anchor-static_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3b/static_c1.yaml` | 无（本矩阵对照：分块**关**、控制器关） | 3/3 | MEASURED | 455.01/499.60/518.84 | 61.81/61.90 | 62.84 | 15.37 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0027 / stall.p99 0.0013 ⚠️见下 | done |
| EXP-base-112 | P1S3B-phase1_anchor-static_c2 | phase1_anchor（in2048/out128，并发 2，8 请求） | `configs/mthreads/phase1_step3b/static_c2.yaml` | 无（分块**关**、控制器关） | 3/3 | MEASURED | 869.25/869.57/869.57 | 77.98/78.05 | 79.45 | 23.74 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0011 / stall.p99 0.0073 | done |
| EXP-base-114 | P1S3B-phase1_anchor-static_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3b/static_c4.yaml` | 无（分块**关**、控制器关） | 3/3 | MEASURED | 1679.63/1679.75/1679.75 | 88.74/88.76 | 90.28 | 39.49 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0001 / stall.p99 0.0011 | done |
| EXP-base-118 | P1S3B-phase1_anchor-static_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3b/static_c8.yaml` | 无（分块**关**、控制器关） | 3/3 | MEASURED | 2312.06/3301.97/3456.04 | 141.78/146.32 | 464.73 | 50.07 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0006 / stall.p99 0.0009 | done |
| EXP-cp-111 | P1S3B-phase1_anchor-static_chunk_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3b/static_chunk_c1.yaml` | `chunk.enabled` True，`chunk_size=1024` | 3/3 | MEASURED | 507.57/508.40/508.46 | 61.82/61.88 | 62.77 | 15.30 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0002 / tpot.p50 0.0003 / stall.p99 0.0066 | done |
| EXP-cp-112 | P1S3B-phase1_anchor-static_chunk_c2 | phase1_anchor（in2048/out128，并发 2，8 请求） | `configs/mthreads/phase1_step3b/static_chunk_c2.yaml` | `chunk.enabled` True，`chunk_size=1024` | 3/3 | MEASURED | 569.49/900.62/1042.75 | 85.95/86.03 | 269.14 | 22.27 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0002 / stall.p99 0.0003 | done |
| EXP-cp-114 | P1S3B-phase1_anchor-static_chunk_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3b/static_chunk_c4.yaml` | `chunk.enabled` True，`chunk_size=1024` | 3/3 | MEASURED | 619.77/2183.33/2379.21 | 110.41/112.02 | 372.68 | 34.33 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0003 / tpot.p50 0.0001 / stall.p99 0.0005 | done |
| EXP-cp-118 | P1S3B-phase1_anchor-static_chunk_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3b/static_chunk_c8.yaml` | `chunk.enabled` True，`chunk_size=1024` | 3/3 | MEASURED | 2787.50/5103.91/5317.46 | 154.30/165.07 | 392.26 | 44.50 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0020 / tpot.p50 0.0007 / stall.p99 0.0005 | done |
| EXP-adap-111 | P1S3B-phase1_anchor-adaptive_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3b/adaptive_c1.yaml` | `controller.enabled` True（相对同批 `static_chunk` 只改这一处） | 3/3 | MEASURED | 507.77/508.29/508.48 | 61.89/61.99 | 63.31 | 15.29 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0008 / tpot.p50 0.0005 / stall.p99 0.0001 | done |
| EXP-adap-112 | P1S3B-phase1_anchor-adaptive_c2 | phase1_anchor（in2048/out128，并发 2，8 请求） | `configs/mthreads/phase1_step3b/adaptive_c2.yaml` | `controller.enabled` True（单变量） | 3/3 | MEASURED | 454.68/878.92/1038.58 | 85.75/85.81 | 241.18 | 22.48 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0006 / tpot.p50 0.0011 / stall.p99 0.0017 | done |
| EXP-adap-114 | P1S3B-phase1_anchor-adaptive_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3b/adaptive_c4.yaml` | `controller.enabled` True（单变量） | 3/3 | MEASURED | 481.57/2183.33/2378.98 | 110.15/110.76 | 569.49 | 34.74 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0006 / tpot.p50 0.0006 / stall.p99 0.0015 | done |
| EXP-adap-118 | P1S3B-phase1_anchor-adaptive_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3b/adaptive_c8.yaml` | `controller.enabled` True（单变量） | 3/3 | MEASURED | 2868.11/6259.64/6585.14 | 156.76/167.82 | 367.33 | 42.71 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0003 / stall.p99 0.0005 | done |

> ⚠️ **`EXP-base-111`（`static_c1`）含一次孤立事件**：第 1 次重复里 `r00003` 的
> `prefill_exec_ms` = **11136.2 ms**，而同一 run 其余 7 个请求的中位数是 455.0 ms
> （**24.5×**），`queue_ms`/`admission_ms` 均为 0、`outcome=success`。该事件只抬高了
> TTFT 的 p90/p95/p99（3661.8/7399.0/10388.8 ms，另两轮仅 ~455–519 ms），
> **`stall p99` 未捕获它**（仅 64.68 ms）——因为它是 prefill 事件而非 token 间隔。
> 后果：该格 `ttft.p90/p95/p99` 的跨重复 spread 达 **674%/1390%/1915%**。
> ⇒ 引用 p90 及以上分位时**必须附 n 并说明该格含单点事件**；p50 不受影响（455.01）。
> 两批 72 个 run 系统排查后仅此 1 例（判据：`prefill_exec > 10× 同 run 中位数`）。

> 上表读数 = 3 次重复的**中位数**。`离散度(稳健CV)` = **跨 3 次重复**的稳健 CV，
> 与 `summary.stability.*_robust_cv`（**一个 run 内跨 8 个请求**）同名不同口径，不可混用。

> 🔧 **2026-09-24 修正一处错指针**：本节的 12 行原先在"配置"列写
> `configs/mthreads/phase1_step3/…`，那是 **P1S3 那一版**。产物自证本批用的是
> `phase1_step3b`（`meta.json.config`：`chunk_size=1024` / `target_tpot_ms=110` /
> `target_ttft_ms=500`，而 `phase1_step3` 是 `512` / `50` / `2000`；`telemetry` 的
> `chunk_size` 列也分别是 1024 与 512）。**照错指针复现会加载到那版缺陷配置**，
> 而已修正。§1.1（编号 `10x`）的 `phase1_step3` 是对的，未动。

---

### 1.3 主体平台（MTT S4000）P2 —— token 时间戳口径修复的前后对照 — 2026-09-23/24

编号接 `12x`（`10x`=P1S3、`11x`=P1S3B、`12x`=本批；**末位仍是并发**）。
全部 `MEASURED`，4 格 × 3 轮 = 12 次运行、穿插执行，**12/12 成功，0 失败**。
产物：`results/raw/P2-*`（12 个目录，48 个文件）。
分析：`scripts/inspect_timing_scope.py`（前 = §1.2 的 `P1S3B-*`，后 = 本批）。

**这一批不是新的性能配置。** 四格的配置与 §1.2 **逐字节相同**
（`static_c1` / `static_c4` / `static_c8` / `adaptive_c4`，均取自
`configs/mthreads/phase1_step3b/`），所以两批之差就是两项代码改动的总作用：
① `ready_s` 的打点从"logits 算完"移到"token 可取回主机"（口径修正）；
② 写回循环里的逐元素 `int(next_ids[i])` 换成一次批量 D2H（真实提速）。
**两项不可分**（同一步做的工作量不同），故下表差值只作量级，不作分解。

| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-base-121 | P2-phase1_anchor-static_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3b/static_c1.yaml` | 无（同 §1.2；被测变量 = **计时口径**） | 3/3 | MEASURED | 455.12/455.35/455.39 | 61.84/61.95 | 64.00 | 15.39 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0002 / stall.p99 0.0032 | done |
| EXP-base-124 | P2-phase1_anchor-static_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3b/static_c4.yaml` | 无（同 §1.2；被测变量 = **计时口径**） | 3/3 | MEASURED | 1680.08/1680.15/1680.15 | 88.60/88.62 | 90.60 | 39.54 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0000 / tpot.p50 0.0001 / stall.p99 0.0021 | done |
| EXP-base-128 | P2-phase1_anchor-static_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3b/static_c8.yaml` | 无（同 §1.2；被测变量 = **计时口径**） | 3/3 | MEASURED | 2312.54/3302.22/3456.17 | 141.49/146.03 | 466.98 | 50.16 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0000 / stall.p99 0.0091 | done |
| EXP-adap-124 | P2-phase1_anchor-adaptive_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3b/adaptive_c4.yaml` | 无（同 §1.2；被测变量 = **计时口径**） | 3/3 | MEASURED | 481.97/2184.50/2380.24 | 109.95/110.54 | 569.67 | 34.80 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0002 / tpot.p50 0.0000 / stall.p99 0.0006 | done |

> ⚠️ **引用这一批时必须显形的四条**：
>
> 1. **两批不得并表**（§4.3 P2）。上表数字与 §1.2 的**同名格**是**不同口径**的两批，
>    不能合并、不能算"重复次数 6"。
> 2. **`adaptive_c4` 的两次读数不是同一件事**：§1.2 的 `EXP-adap-114`（TTFT P50 = 481.57）
>    是 3 次重复的**中位数**，掩盖了它三次是 `481.4 / 481.6 / 652.1` —— 其中一次
>    `chunk_size` 爬坡卡在 1536 未升到 2048。根因见 §4.3.2，~~与计时修复无关~~。
>    本批三次为 `482.0 / 481.9 / 482.0`（极差 0.128 ms）。⇒ 该格在 P1S3B 上**不具备
>    跨重复可复现性**，用它做"自适应 vs 静态"比较无效。
>    ⚠️ **2026-09-24 更正**：划掉的那句归因**不可判定**。P2（**旧门限** 110/0.1 ＋ 计时修复）
>    的 c4 三次**全部**收敛为 `{1024:8, 2048:4}`，与 P1S3D 相同；P1S3B 是 1/3 分叉。
>    n=3 分不开"计时修复消除了它"与"没抽中"⇒ 应读作**归因未建立**，详见 §3.3 **E8**。
> 3. **版本溯源已满足**：本批 12/12 的 `reproducibility` 为
>    `git_commit=87aeccc6c4718a39c8ddee46850d963897279981`、`git_source='stamp'`、
>    `git_dirty=False`、`model_revision=a09a35458c702b33eeacc393d103063234e8bc28`、
>    `model_revision_resolved=True`。这是 **P1 对新产物的验收在真机通过的证据**。
> 4. **本批 TTFT 的增量只在 c8 可分辨**（+0.484 ms，批内漂移 0.362 ms）；
>    c1/c4 的 +0.10 / +0.45 ms 都**小于各自批内漂移**（0.246 / 0.539 ms）⇒
>    这两格"口径修复对 TTFT 的影响"**测不出来**，只能说量级 ≤ 0.24 ms（直接测到的
>    被错记区间中位数）。明细见 §4.3.2。

---

### 1.4 主体平台（MTT S4000）P4 真机重跑（P1S3D）— 2026-09-24

编号接 `13x`（`10x`=P1S3、`11x`=P1S3B、`12x`=P2、`13x`=本批；**末位仍是并发**）。
全部 `MEASURED`，12 格 × 3 轮 = 36 次运行、穿插执行，**36/36 成功，0 失败**。
产物：`results/raw/P1S3D-*`（36 个目录，144 个文件，文件字节和 9 464 185）。
本批测的是 **P4 的门限改动**（`controller.target_tpot_ms` 110→120、
`controller.hysteresis` 0.1→0.08，配置目录 `configs/mthreads/phase1_step3d/`），
缺陷形状与改动清单见 §4.3.4。**为什么必须重跑**：`controller.*` 参与调度决策，
改了就会改变行为 ⇒ 不能像 P3（`run.slo.*`）那样只重算。

**输入新鲜度（本次新增的检查，见下 §3.3 E4）**：部署戳 `SOURCE_PROVENANCE.json` 的
`digest_scope` 只含 `src/ascend_opt` / `benchmark` / `scripts`，**不含 `configs/`** ⇒
"代码是这个提交"**推不出**"配置是这一版"。故本次逐 run 比对
`config.snapshot.yaml` 与 `configs/mthreads/phase1_step3d/` 下的同名文件：
**36/36 逐字段一致**（`scripts/inspect_step3_runs.py --verify-config`，退出码 0）。
反向对照：同一命令对 `phase1_step3c/` 返回 **1**，逐字段列出 3 处差异
（`controller.hysteresis` 0.1≠0.08、`controller.target_tpot_ms` 110≠120、`run.notes`）——
即这条检查**能失败**，不是"永远通过"的打印。

**溯源**：本批跑在 `git_commit=71e55abd1dd8e18189f5e1fd275f9a1b69a2e75e`、
`git_dirty=false`、树摘要 `3e833dd977bf10200d8aef4e8ef386b36521e8e544fcde501be7af3a68448c7b`。
⚠️ 该戳**不覆盖 `configs/`**（同上），配置侧的溯源由上面那条检查补上。

**必须与这些限制一起引用**：

> 1. **与 §1.2（P1S3B）不同计时口径**：本批在 P2 的修复**之后**运行，`note` 含
>    `readback_ms`，TTFT/E2E 记的是 **token 可取回主机**的时刻；P1S3B 记的是
>    logits 就绪时刻，**系统性偏早**。⇒ 两批的绝对 TTFT 差含"门限改动"与"计时修复"
>    **两项**，不可直接并表。计时项的量级上界已由 §4.3.2 给出：c1 +0.103 ms、
>    c4 +0.453 ms（**均小于各自批内漂移**，只能说 ≤0.24 ms）、c8 +0.484 ms。
>    §3.3 中凡跨 3B/3D 的比较都已按此限定。
> 2. **`static_chunk` 基线相对 §1.2 变了值**：`chunk.chunk_size` 由 512 → **1024**
>    （同 §1.2 相对 §1.1）。⇒ **跨批的分块基线不得直接比较**；本批**内部**的
>    `static` vs `static_chunk`（只改 `chunk.enabled`）与 `static_chunk` vs `adaptive`
>    （只改 `controller.enabled`，两者 `chunk_size` 均为 1024）才是单变量。
> 3. **`run.slo.tpot_ms` 由 50 → 100**（P3 的改动）。⇒ 本批达标率三列**首次携带信息**，
>    但**与 §1.1/§1.2/§1.3 的达标率列不同阈值、不可并表**。用
>    `scripts/calibrate_slo.py --candidates 100` 可把旧批按同一阈值重算（§3.3 E3 已做）。
> 4. **负载形状（2026-09-24 晚更正）**：每格 8 请求、并发固定，但**不是"全部同时到达"**。
>    负载是 `arrival: closed`（闭模型）：**前 `concurrency` 个立即到达，其余在执行槽释放时补位**，
>    故**补入批次数 = `num_requests / concurrency`**（c1/c2/c4/c8 = **8/4/2/1 批**；
>    `requests.jsonl` 的 `arrival_ms` 在 3 次重复里逐个一致，c1 批间隔 ≈8.37 s、c2 ≈11.37 s）。
>    ⇒ 本批**不是**"一次启动 + 一次排空"，而是 **`8/concurrency` 次"填满 → 该批预填充突发 →
>    该批排空"的重复瞬变**（只有最后一批真的排空）。**限制**：负载水平**始终填满到
>    `concurrency`**，没有"低压→高压→恢复"的轨迹 ⇒ "控制器在**负载水平变化**下如何表现"
>    本批**测不到**。**但限制之外有收获**：**补入扰动下的响应**测到了，并已定因（§3.3 E7/E9）。
>    ⚠️ 原写"全部同时到达"，与产物直接矛盾。源头：`_sample_arrivals` 对闭模型返回的是全 0 的
>    **计划**到达（"由执行器在补位时分配真实时刻"），**计划到达被当成了实际到达**。
> 5. `n_failed`/`n_rejected`/`n_unfinished` 全 0 —— 本负载**产生不出**这些结局，不是"测过没有"。

| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-base-131 | P1S3D-phase1_anchor-static_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3d/static_c1.yaml` | 无（本矩阵对照：分块**关**、控制器关） | 3/3 | MEASURED | 455.24/455.60/455.62 | 61.85/61.91 | 63.45 | 15.39 | 1.000 | 0.1203 | 15.396 | 0.000 | ttft.p50 0.0004 / tpot.p50 0.0024 / stall.p99 0.0088 | done |
| EXP-base-132 | P1S3D-phase1_anchor-static_c2 | phase1_anchor（in2048/out128，并发 2，8 请求） | `configs/mthreads/phase1_step3d/static_c2.yaml` | 无（分块**关**、控制器关） | 3/3 | MEASURED | 869.64/869.95/869.95 | 77.97/78.09 | 80.05 | 23.74 | 1.000 | 0.1856 | 23.761 | 0.000 | ttft.p50 0.0002 / tpot.p50 0.0014 / stall.p99 0.0120 | done |
| EXP-base-134 | P1S3D-phase1_anchor-static_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3d/static_c4.yaml` | 无（分块**关**、控制器关） | 3/3 | MEASURED | 1680.06/1680.26/1680.26 | 88.64/88.66 | 90.58 | 39.52 | 1.000 | 0.3088 | 39.526 | 0.000 | ttft.p50 0.0002 / tpot.p50 0.0001 / stall.p99 0.0074 | done |
| EXP-base-138 | P1S3D-phase1_anchor-static_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3d/static_c8.yaml` | 无（分块**关**、控制器关） | 3/3 | MEASURED | 2312.40/3302.35/3456.46 | 141.58/146.12 | 464.59 | 50.14 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0010 / stall.p99 0.0012 ⚠️见 §3.3 E5 | done |
| EXP-cp-131 | P1S3D-phase1_anchor-static_chunk_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3d/static_chunk_c1.yaml` | `chunk.enabled` True，`chunk_size=1024` | 3/3 | MEASURED | 508.13/509.07/509.14 | 61.85/61.96 | 64.71 | 15.29 | 1.000 | 0.1195 | 15.300 | 0.000 | ttft.p50 0.0007 / tpot.p50 0.0007 / stall.p99 0.0341 | done |
| EXP-cp-132 | P1S3D-phase1_anchor-static_chunk_c2 | phase1_anchor（in2048/out128，并发 2，8 请求） | `configs/mthreads/phase1_step3d/static_chunk_c2.yaml` | `chunk.enabled` True，`chunk_size=1024` | 3/3 | MEASURED | 570.00/900.98/1042.69 | 85.87/86.05 | 269.49 | 22.29 | 1.000 | 0.1741 | 22.287 | 0.000 | ttft.p50 0.0003 / tpot.p50 0.0002 / stall.p99 0.0003 | done |
| EXP-cp-134 | P1S3D-phase1_anchor-static_chunk_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3d/static_chunk_c4.yaml` | `chunk.enabled` True，`chunk_size=1024` | 3/3 | MEASURED | 619.83/2183.69/2379.50 | 110.28/111.88 | 372.36 | 34.38 | 0.125 | 0.0336 | 4.297 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0001 / stall.p99 0.0005 | done |
| EXP-cp-138 | P1S3D-phase1_anchor-static_chunk_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3d/static_chunk_c8.yaml` | `chunk.enabled` True，`chunk_size=1024` | 3/3 | MEASURED | 2787.57/5103.90/5317.26 | 153.87/164.68 | 391.75 | 44.60 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0002 / stall.p99 0.0007 | done |
| EXP-adap-131 | P1S3D-phase1_anchor-adaptive_c1 | phase1_anchor（in2048/out128，并发 1，8 请求） | `configs/mthreads/phase1_step3d/adaptive_c1.yaml` | `controller.enabled` True（相对同批 `static_chunk` 只改这一处） | 3/3 | MEASURED | 507.99/508.52/508.62 | 61.90/61.99 | 63.94 | 15.28 | 1.000 | 0.1194 | 15.277 | 0.000 | ttft.p50 0.0001 / tpot.p50 0.0005 / stall.p99 0.0181 | done |
| EXP-adap-132 | P1S3D-phase1_anchor-adaptive_c2 | phase1_anchor（in2048/out128，并发 2，8 请求） | `configs/mthreads/phase1_step3d/adaptive_c2.yaml` | `controller.enabled` True（单变量） | 3/3 | MEASURED | 454.96/878.68/1038.13 | 85.67/85.86 | 241.12 | 22.49 | 1.000 | 0.1756 | 22.474 | 0.000 | ttft.p50 0.0003 / tpot.p50 0.0012 / stall.p99 0.0006 | done |
| EXP-adap-134 | P1S3D-phase1_anchor-adaptive_c4 | phase1_anchor（in2048/out128，并发 4，8 请求） | `configs/mthreads/phase1_step3d/adaptive_c4.yaml` | `controller.enabled` True（单变量） | 3/3 | MEASURED | 481.91/2184.25/2380.09 | 110.01/110.65 | 569.51 | 34.79 | 0.125 | 0.0339 | 4.343 | 0.000 | ttft.p50 0.0002 / tpot.p50 0.0001 / stall.p99 0.0001 | done |
| EXP-adap-138 | P1S3D-phase1_anchor-adaptive_c8 | phase1_anchor（in2048/out128，并发 8，8 请求） | `configs/mthreads/phase1_step3d/adaptive_c8.yaml` | `controller.enabled` True（单变量） | 3/3 | MEASURED | 2867.12/6254.70/6579.55 | 156.08/167.13 | 367.00 | 42.87 | 0.000 | 0.000 | 0.000 | 0.000 | ttft.p50 0.0002 / tpot.p50 0.0000 / stall.p99 0.0001 | done |

> 上表时间列 = 3 次重复的**中位数**；`达标率` = 跨 3 轮的达标率中位数；
> `达标吞吐` 两列取**基准轮**读数（同 §1.1 的取法）。
> `离散度(稳健CV)` = **跨 3 次重复**的稳健 CV，与 `summary.stability.*_robust_cv`
> （**一个 run 内跨 8 个请求**）同名不同口径，不可混用。

> ⚠️ **`EXP-base-138`（`static_c8`）含一次孤立事件**：三轮 `stall p99` =
> `498.28 / 464.59 / 464.22`，其中首轮偏高 **7.3%**；稳健 CV 0.0012（看似极稳）
> 而普通 CV **0.0411**，被 `scripts/inspect_step3_runs.py` 的单点离群判据标出。
> ⇒ 引用本批的 `stall p99` 噪声底时须用被抬高后的值（**最大 3.41%**，见 §3.3 E5），
> 不要用其余 11 格的中位（0.09%）——后者会把该格的离群藏起来。

### 1.5 主体平台（MTT S4000）执行优化线 **EXECOPT3**（4 臂 × 3 轮）— 2026-09-30

编号用 `15x` 段（执行优化线，与论文线 OPT-xx 及上面 `1xx` 性能矩阵隔开）。
**这是执行优化线 S01/S02/S03 三个机制开关的第一次真机受控对照。**
协议：`short_chat`（in≈496 / out≈123，并发 8，64 请求），`max_num_seqs=32`，
分块预填充 8192，控制器**关**（`n_control_decisions=0`，实测）⇒ **无调度混淆**，
是纯执行路径 A/B。逐项 3 轮、**穿插**（pass-major），单轮 ~176 s，全程 35.2 min。
产物：`results/raw/EXECOPT3-short_chat-*`（12 个目录，48 个文件）。
批次收尾：`n_done=12/12`、`completed_passes=3`、`balanced=true`、`failures=[]`。

| 编号 | run_id | 工作负载 | 配置 | 变量（相对 baseline 只改这一处） | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-exec-151 | EXECOPT3-short_chat-s4000_fp16 | short_chat（in496/out123，并发 8，64 请求） | `configs/mthreads/exec_opt/s4000_fp16.yaml` | 无（对照：三开关全默认） | 3/3 | MEASURED | 196.98/375.84/1403.77 | 117.95/124.64 | 322.40 | 63.84 | 0.031 | 0.0163 | 1.91 | 0.000 | ttft.p50 0.117 / tpot.p50 0.444 | done |
| EXP-exec-152 | EXECOPT3-short_chat-s01_both | short_chat（in496/out123，并发 8，64 请求） | `configs/mthreads/exec_opt/s01_both.yaml` | `logits_last_only=True` + `sample_needed_only=True` | 3/3 | MEASURED | 188.78/367.74/1375.97 | 118.04/124.86 | 322.33 | 63.93 | 0.031 | 0.0164 | 1.91 | 0.000 | ttft.p50 0.117 / tpot.p50 0.105 | done |
| EXP-exec-153 | EXECOPT3-short_chat-s02_no_empty_cache | short_chat（in496/out123，并发 8，64 请求） | `configs/mthreads/exec_opt/s02_no_empty_cache.yaml` | `retire_empty_cache=False` | 3/3 | MEASURED | 196.79/375.85/1403.50 | 117.97/124.63 | 322.37 | 63.98 | 0.031 | 0.0163 | 1.91 | 0.000 | ttft.p50 0.117 / tpot.p50 0.056 | done |
| EXP-exec-154 | EXECOPT3-short_chat-s03_fuse_gate_up | short_chat（in496/out123，并发 8，64 请求） | `configs/mthreads/exec_opt/s03_fuse_gate_up.yaml` | `fuse_gate_up=True` | 3/3 | MEASURED | 196.32/373.14/1308.84 | 119.28/125.91 | 322.34 | 63.26 | 0.031 | 0.0163 | 1.90 | 0.000 | ttft.p50 0.117 / tpot.p50 0.021 | done |

> 上表读数 = 3 次重复的**中位数**。
> `离散度(稳健CV)` 是**跨 3 次重复**的稳健 CV（1.4826·MAD/中位），单位 `%`，
> 与 `summary.stability.*_robust_cv`（**一个 run 内跨 64 个请求**）同名不同口径。
> ⚠️ **本批噪声带极窄**：baseline 自身跨轮极差 `tpot_p50` **0.444%**、
> `ttft_p50` **0.117%** ⇒ 小于此的效应不可分辨（判定见 §3.6）。
> ⚠️ **受控性已核**：四臂 `requests.jsonl` 的 `prompt_len` 序列指纹
> 全等（`sha256[:16]=8589c6cfd8d6257c`）；**并做了否定测试** —— 故意用不存在
> 的字段名（`input_len`）也会得到"唯一指纹数=1"，说明**单看指纹相等不足以证明受控**，
> 因此另外人工核了 `prompt_len` 首两值（650 / 297，四臂一致且非空）。

### 1.6 EXECOPT4（S04 常驻 KV 缓冲，2026-09-30，真机）

协议与 EXECOPT3 **相同**（`short_chat`、并发 8、64 请求、控制器关、`max_num_seqs=32`、
分块预填充 8192），便于与 1.5 交叉印证。4 臂 × 3 轮穿插，12/12、`balanced=true`、
`failures=[]`、35.0 min。产物 `results/raw/EXECOPT4-*`。

⚠️ **本表读数口径**：TTFT/TPOT 取 3 轮**中位数**；
`pack_tel`/`fwd#tel` 来自 **telemetry 逐 step 求和**（= 正式测量段；见 §4.4 口径限制 2）。

| 编号 | run_id | 工作负载 | 配置 | 变量（只改这一处） | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P90 (ms) | 吞吐 (tok/s) | `pack_tel` (ms) | `fwd#tel` | 快路径/回退步数 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-exec-155 | EXECOPT4-short_chat-s4000_fp16 | short_chat（in496/out123，并发 8，64 请求） | `configs/mthreads/exec_opt/s4000_fp16.yaml` | 无（对照） | 3/3 | MEASURED | 197.13/938.75/1315.80 | 118.65/122.91 | 63.6 | 21,269.6 | 1,143 | — | done |
| EXP-exec-156 | EXECOPT4-short_chat-s01_both | 同上 | `configs/mthreads/exec_opt/s01_both.yaml` | `logits_last_only=True`+`sample_needed_only=True` | 3/3 | MEASURED | 188.80/899.85/1263.22 | 118.56/122.61 | 63.7 | 21,267.9 | 1,143 | — | done |
| EXP-exec-157 | EXECOPT4-short_chat-s04_pkv | 同上 | `configs/mthreads/exec_opt/s04_pkv.yaml` | `persistent_kv_slots=True`+`capacity=1024` | 3/3 | MEASURED | 197.12/938.48/1314.63 | 118.27/122.56 | 63.7 | 21,310.4 | 1,143 | **193/1067** | done |
| EXP-exec-158 | EXECOPT4-short_chat-s01_both_s04_pkv | 同上 | `configs/mthreads/exec_opt/s01_both_s04_pkv.yaml` | 上两行之和 | 3/3 | MEASURED | 188.72/900.41/1264.01 | 119.10/122.56※ | 55.8※ | 21,378.0 | 1,143 | **193/1067** | done |

> ※ **本表以"全窗口"列示，但它对 S04 是**误导性口径**（2026-10-01 更正）**：
> 12 个 run 里 4 个出现了**一次孤立单步长停顿**（13.9–19.3 s，**不是**持续慢态；见 §4.4 A3），
> `wall` 与吞吐差异是它的算术后果 ⇒ **服务指标本批暂不可归因**。
> ★★ 判 S04 **必须看"有效样本"**：快路径只覆盖 **77 / 1143 = 6.74%**（= 末尾连续
> `decode==1` 的单请求尾段，telemetry 第 1004 行起）。在那 77 步上 `pack` 总量
> **109.02 → 187.52 ms（+72.0%，3 轮极差 0.4%）**；对照组 `s01_both → 组合臂`
> **108.98 → 187.31（+71.9%）**。
> ⇒ 全窗口的 **+0.19%** 是**算术后果**，**不能**据此判"不可分辨"（见 §3.6 E20、§4.4 A4）。
> ⚠️ **2026-10-03 口径更正**：这里原先写"被 6.74% 覆盖率**稀释**"。`6.74%` 是**数量**
> 覆盖率，**不是**耗时占比 —— 尾段的**耗时**份额只有 **~0.52%**（EXECOPT5 实测
> baseline 0.5189%）。⇒ 解释全窗口差**必须用时间权重**，且可用加性核对验证
> （残差 0.047%、88.6% 来自尾段）。详见 **§4.4 A4**。
> ⚠️ 本批 S04 还带**已复现的数值缺陷**与**源码身份缺失**（§4.4 A5/A6）——
> 引用时两条都要带上。
> ⚠️ 分母口径：上表 `fwd#tel` = **前向组**（1,143）；**逻辑步**是 1,080（telemetry 行数）。

---

### 1.7 EXECOPT5（S04 常驻 KV 缓冲 · **5 轮独立复现**，2026-10-03，真机）

协议与 EXECOPT3/EXECOPT4 **相同**（`short_chat`、并发 8、64 请求、控制器关、
`max_num_seqs=32`、分块预填充 8192）。**2 臂 × 5 轮穿插 = 10/10**、
`completed_passes=5`、`balanced=true`、`failures=[]`、**`complete=true` / `under_run=false`**、
27.0 min（`avg_item_s=162.2`）。产物 `results/raw/EXECOPT5-*`（10 个目录，回拉后完整性 10/10）。

| 臂 | 配置 | 变量（只改这一处） | 重复 | `pack_tel` 5 轮区间（中位） | `fwd_tel` 5 轮区间 | 快路径/回退（含预热） | **尾段(77 步) `pack`Σ 中位** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | `configs/mthreads/exec_opt/s01_both.yaml` | 对照（`logits_last_only`+`sample_needed_only`） | 5/5 | 20,952.1–20,994.6（**20,968.8**） | 98,560.8–98,906.5 | 0 / 0 | **108.80 ms** |
| `s01_both_s04_pkv` | 同上 + PKV | `persistent_kv_slots=True` + `capacity=1024` | 5/5 | 21,007.9–21,092.0（**21,055.9**） | 98,688.1–98,852.6 | 193 / 1067 | **185.95 ms** |

> ★★ **本批的结论**（口径见 §4.4 A4）：
>
> | 口径 | 比值（pkv / baseline） | 5 轮极差 | 同号 |
> | --- | --- | --- | --- |
> | **尾段 77 步 `pack`Σ（有效样本）** | **+70.9%**（**5 轮逐轮配对比值的中位数**） | 16.3% | **5/5** |
> | 全窗口 1,080 步 `pack_tel` | +0.366% | 0.321% | 5/5 |
>
> ⚠️ 上表尾段那个数是**逐轮配对比值的中位数**（`median(r_i)`，`r_i = pkv_i / baseline_i`），
> **不是**"中位数之比"。两者本批恰好接近（+70.86% vs +70.91%），但**报的必须是前者**：
> 配对消掉了轮与轮之间的整体漂移，只有它才是"同一轮内两臂之差"。
>
> ★ **两个口径的换算关系（这是本批最该讲清的一点）**：
> `6.74%` 是**数量覆盖率**（快路径前向组数 / 正式窗口前向组数 = 77/1143），
> 它**不是**耗时占比。要解释"尾段 +70.9% 为什么在全窗口只剩 +0.37%"，
> 必须用**时间权重**：
>
> | 臂 | 尾段 `pack`Σ | 全窗口 `pack`Σ | **时间权重 = 尾段/全窗** |
> | --- | --- | --- | --- |
> | baseline | 108.80 ms | 20,968.8 ms | **0.5189%** |
> | pkv | 185.95 ms | 21,055.9 ms | **0.8831%** |
>
> ⇒ 尾段的**时间份额只有 ~0.52%**（而**数量**份额是 6.74%）——因为尾段是单请求纯 decode，
> 每步只搬 1 个 token 的 KV，**单位步的搬运量远低于**前段的批量步。
> **加性核对**：`baseline 全窗 + 尾段差 = 20,968.8 + 77.15 = 21,046.0` vs
> `pkv 全窗 21,055.9` ⇒ 残差 **9.9 ms（0.047%）**⇒ 全窗口 87.1 ms 的差里
> **77.15 ms（88.6%）直接来自尾段**，其余落在轮间散布内。
> ⇒ 正确表述：**不是"被稀释成没有"，而是"成本真实存在、但只占总耗时的约 0.9%"**。
>
> ⚠️ **关于 `fwd_tel`（本节与 §3.8 原来的写法已删除，理由见下）**：
> 原来写的是"两臂 `fwd_tel` 区间**重叠** ⇒ 机制只作用在 `pack`"。
> 这条**已撤回**：**区间重叠不是检验**，n=5 下重叠完全不能推出"无差异"。
> 本批**没有**判定 `fwd` 上是否存在差异的能力 ⇒ 只能记"**本批不足以判定 `fwd`**"。
> 机制层面的解释（"快路径改的是搬运/拼接，不是前向计算"）**是设计意图**，
> **不能**由本批数据证实或证伪。
>
> ⚠️ **逐轮配对明细**：
>
> | 轮 | baseline 尾段Σ | pkv 尾段Σ | 尾段比值 | 全窗比值 |
> | --- | --- | --- | --- | --- |
> | r0 | 107.56 | 187.05 | +73.9% | +0.587% |
> | r1 | 108.80 | 185.38 | +70.4% | +0.266% |
> | r2 | 111.25 | 185.32 | +66.6% | +0.426% |
> | r3 | 107.93 | 197.36 | +82.9% | +0.366% |
> | r4 | 108.83 | 185.95 | +70.9% | +0.292% |
>
> ⚠️ **`pack_drv`/`fwd_drv`/`fwd#drv` 跨臂不可比**：本批 `warmup` 轮数**两臂不同**
> （baseline 2 轮 / pkv 1 轮）—— 这正是 §4.4 口径限制 A1 说的那个陷阱，实测又出现一次。
> ⚠️ **每个 run 都恰有 1 个 >1 s 的单步**（`max_fwd≈1,310 ms @step0`，两臂**同样**存在，
> 10/10 轮）⇒ 那是**起手步**的固定现象；引用服务指标时须带这条。
> ⚠️ **本批的环境证据不完整**：产物里的 `hardware_env` 来自**另一个容器**
> （陈旧 5 天）；同期可用证据与缺失项见 **§4.4 A7**。
> **正式论文引用本批前必须先解决这一条**。
> ⚠️ 分母口径：`fwd#tel` = **前向组**（1,143）；**逻辑步**是 1,080（telemetry 行数）。
> 快路径 **193** 是**含预热**的总量，正式窗口是 **77**（`193-116`，116 来自预热轮）。

---

### 1.8 S01RECHECK（**S01 实际部署收益的独立复核**，2026-10-03，真机）

**为什么要独立复核**：`EXECOPT5` 的**两个臂都开着 S01**（其 `baseline` 臂本身就是
`s01_both.yaml`）⇒ 它**构造上不能**证明 S01 自身的收益。本批把对照换回
**同一条执行路径、同钉死 revision、两开关全关**的 `s01_reference.yaml`。

协议与 EXECOPT3/4/5 **相同**（`short_chat`、并发 8、64 请求、控制器关、
`max_num_seqs=32`、分块预填充 8192）。**2 臂 × 5 轮，AB/BA 交错 = 10/10**、
`completed_passes=5`、`balanced=true`、`failures=[]`、**`complete=true` / `under_run=false`**、
`budget_limited=false`、29.0 min（`avg_item_s=173.9`）。
产物 `results/raw/S01RECHECK-*`（10 个目录，回拉后完整性 10/10、`data_label=MEASURED`）。

**顺序表（预注册、计划/执行/续跑共用同一函数）**：
`r0 A,B / r1 B,A / r2 A,B / r3 B,A / r4 A,B`（`--order abba`）。
⚠️ 此前计划与执行**各写了一遍**格序 ⇒ 只改一处会让"计划 ABBA、实际 ABAB"且不报错。

| 臂 | 配置 | 变量（只改这一处） | 重复 |
| --- | --- | --- | --- |
| `s01_reference`（对照） | `configs/mthreads/exec_opt/s01_reference.yaml` | 两开关**全关**（= 原执行路径 + 同钉死 revision） | 5/5 |
| `s01_both`（处理） | `configs/mthreads/exec_opt/s01_both.yaml` | `logits_last_only=True` + `sample_needed_only=True` | 5/5 |

**单变量性（程序化核对，非目测）**：两臂有效配置差异**恰好是两个 S01 开关**
（`runtime.logits_last_only` / `runtime.sample_needed_only`）；
参照臂与平台基线 `s4000_fp16.yaml` 只差 `model_revision` 的**写法**（`main` → 钉死 commit）。
产物内 `switches` 字段同样证明：`baseline={}`、`s01_both={'logits_last_only':True,'sample_needed_only':True}`。

**★ 四类身份（本批全部闭合 ⇒ 属 T1，详见 `EVIDENCE_INDEX.md` §5）**：
源码 `tree_digest=472697170d43…`（`git_source=stamp`、`git_commit=16e82d2ea117`、`git_dirty=true` 如实记录）；
配置 = 两臂各自 `config.snapshot.yaml` 摘要（`2b88fee83e30` / `fdda5cc9ca34`，**不同**）；
模型 `a09a35458c702b33eeacc393d103063234e8bc28`；
环境 = **本批前置重采**的指纹（`collected_at=2026-10-03T15:37:35+0800`、容器 `05264bbf0d` == 实时机器标识、
`device_count=1`、`MTT S4000`、driver `2.7.0`、MUSA `3.1.0`、torch `2.2.0`、torch_musa `1.3.0+81caf0a`）。

**开跑前置（本次实际执行过）**：环境指纹读取改为按真实结构（`device.count` /
`discovery.values.MTT_DRIVER_VERSION`）+ 六项接受判据（首次采集同样查）；
真机**短启动**（导入自检 + 负对照）；**数值正确性验收**（两臂 token 序列逐位相同，
摘要 `1c4ecd6016654de0…`）。

## 2. 实验详情（每条一节的模板）

```
### EXP-<维度>-<序号> — <一句话假设>
- 假设：<可证伪的陈述，含预期方向与量级>
- 变量：<唯一改变量>，基线值 <A> → 实验值 <B>
- 不变项：模型 / 模型版本 / 精度 / 量化 / 序列长度分布 / 并发 / 硬件 / 驱动 /
          CANN / torch_npu / vLLM / vLLM-Ascend / 编译选项
- 方法：<如何测，预热策略（计时窗口之外），重复次数，SLO 定义>
- 数据标签：<从 meta.json 抄录，必须是 MEASURED 才可下结论>
- 观测：mean / median / P90 / P95 / P99（TTFT、TPOT、E2E、stall），
        吞吐（tok/s），显存峰值（MiB/GiB），离散度（稳健 CV）
- 结论：<一句话，含是否支持假设；不支持也要写>
- 缺口：<本次有哪些量没测到；这些量会怎样影响结论的强度>
- 意外发现：<与假设无关但值得记的现象>
- 关联 run：<run_id 列表>
- 遗留问题：<下一步/未解释的现象>
```

> 「缺口」字段不可省。规范第 10 节要求区分"没测"与"测出来是 0"；
> 把未测项写清楚，读者才能判断结论的适用范围。

---

## 3. 已完成的结论（供 README / paper 引用）

### 3.1 主体平台（MTT S4000）· 2026-09-23 · `MEASURED`

来源：`PERFORMANCE_MATRIX_PHASE1.md` §7；登记行见 §1.1；原始产物 36 个目录在
`results/raw/P1S3-*`。**噪声底必须先于结论引用**（见该文 §7.3.1）：
`TTFT p50` spread 中位 0.069% / 最大 0.415%；`TPOT p50` 0.134% / 0.720%；
`stall p99` 0.627% / 5.098%（均为 12 格上的 spread_pct）。

| 编号 | 结论（含方向与量级） | 依据 |
| --- | --- | --- |
| **C1** | 分块预填充（`chunk_size=512`）与不分块相比，**TTFT P50 差异随并发换向**：c1 **+40%**、c2 −5%、c4 **−42%**、c8 **+72%**。差异量级是噪声底最大值的 100 倍以上。 | `EXP-base-10x` vs `EXP-cp-10x`，`PERFORMANCE_MATRIX_PHASE1.md` §7.3.2 |
| **C2** | 分块预填充（`chunk_size=512`）的实质是**用 TTFT 换 stall**，且放大的是 **TTFT 的尾部**而非中位数：关分块时 TTFT 分位几乎重合（`static_c4` = 1655.40/1655.52/1655.52），开分块后尾部拉开（`static_chunk_c4` = 960.72/3054.29/3344.61）。**收益随并发变化，不得概括为"分块改善尾延迟"**——按每格 3 次重复取中位，stall P99 的相对变化**换向且非单调**：<br>c1 −0.6%（62.54→62.17，**低于**噪声底 0.627%，即无变化；且 TTFT P50 **+40.5%**）<br>**c2 +183.0%（78.51→222.21，恶化）**<br>**c4 +206.7%（88.47→271.31，恶化）**<br>**c8 −36.0%（458.49→293.57，改善）**<br>**只有 c8 的尾延迟改善**，且伴随**约 18% 的吞吐下降**（50.69→41.46 tok/s，**−18.2%**）与 `wall_ms` +22.3%。⇒ 正确表述是"分块预填充对尾延迟的作用**取决于并发**、也取决于 prefill 与 decode 是否重叠"；**"并发 ≥2 时 stall P99 改善"是错的**（该表述 2026-09-23 修正，原写于本行与 `PERFORMANCE_MATRIX_PHASE1.md` §7.3.2）。 | 同上 §7.3.2；**本次按 `results/raw/P1S3-*` 逐格 3 次重算**（三臂分桶、`static` vs `static_chunk`） |
| **C3** | 本平台在该负载下**重复性很好**：35/36 个「格 × 指标」的跨重复稳健 CV < 0.005，且已排除共享热状态（同格 3 次相隔 19.3 分钟、各自重新装载预热）。**该结论只对该负载/平台/测量链路成立，不得外推为噪声底。** | 该文 §7.2、§7.3.1 |
| **C4** | 测量方法学：**噪声底随分位升高而变大**，且尾部事件只影响分位、不影响中位数（`static_c1` 某轮 TTFT p99 515 ms 而 p50 仅 448.47）。**稳健 CV 会掩盖 n=3 的单点离群**（`static_c8` stall p99 稳健 CV 0.0002 vs 普通 CV 0.0213）。⇒ 只看 p50、或只报 CV，都会漏掉该看的东西。 | 该文 §7.3.1 |
| **C5** | **有缺陷的 `adaptive` 臂在四个并发档上与 `static_chunk` 基线逐档吻合（中位差值均 ≤0.5%）**（stall P99 三次原始值，`adaptive` 在前、`static_chunk` 在后）：c1 `62.34/65.13/62.18` vs `62.17/62.64/61.91`；c2 `223.30/222.31/222.44` vs `222.20/222.21/223.39`；c4 `271.37/271.88/271.25` vs `271.31/271.30/271.36`；c8 `295.82/295.05/293.71` vs `293.57/293.73/293.36`。⇒ 这**不是**"自适应无收益"，而是该臂**在配置上就等价于固定 512 分块**（`chunk_size_init == chunk_size_min == 512`，维度一步也动不了）。它为"P1S3 的 `adaptive` 臂作废"提供了**独立于代码审查的数值证据**。 | `results/raw/P1S3-*`，逐格 3 次 |

### 3.2 主体平台（MTT S4000）· 2026-09-23 · **修正版矩阵** `MEASURED`

来源：登记行见 §1.2，产物 `results/raw/P1S3B-*`（36 个目录）。
**先读 §1.2 顶部那三处"必须显形"的限制**——尤其第 2 条（计时口径是修复前的）。
噪声底（11 格，已剔除含孤立事件的 `static_c1`；口径 `spread = (max−min)/中位`）：

| 指标·分位 | spread 中位 | spread 最大 |
| --- | --- | --- |
| `ttft.p50` | 0.0962% | **35.4625%**（来自 `adaptive_c4`，成因见 **D2**，**不是**机器抖动） |
| `ttft.p90` | 0.1588% | 0.5793% |
| `ttft.p95` | 0.1690% | 0.5691% |
| `ttft.p99` | 0.2136% | 0.5616% |
| `tpot.p50` | 0.1662% | 0.9931% |
| `tpot.p99` | 0.1999% | 2.0257% |
| `stall.p50` | 0.5319% | 4.9116% |
| `stall.p99` | 1.0809% | 14.2125% |

| 编号 | 结论（含方向与量级） | 依据 |
| --- | --- | --- |
| **D1** | **自适应相对"同分块静态"（两臂 `chunk_size` 均为 1024，只改 `controller.enabled`）只在 prefill 主导的指标上有效，且方向随并发换向**：c1 TTFT P50 **+0.0%**（该格控制器零动作，见 D5）；c2 TTFT P50 **−20.2%**、stall P99 **−10.4%**；c4 TTFT P50 **−22.3%** 但 stall P99 **+52.8%（恶化）**；c8 TTFT P50 +2.9%、**TTFT P95/P99 +22.6%/+23.8%（尾部恶化）**、stall P99 −6.4%、吞吐 **−4.0%**。TPOT P50 在 c1–c4 的变化 ≤0.2%（符合"分块只调节 prefill"）。⇒ **不能笼统写"自适应更优"**：c4 是明确的 TTFT↔stall 权衡，c8 是净负向。 | §1.2 表；3 次重复中位数 |
| **D2** | **自适应臂的重复性系统性低于静态臂，且成因在控制器本身，不是机器抖动**：`adaptive_c4` 三次重复的 TTFT P50 = `481.37 / 481.57 / **652.15**`（spread 35.46%），而同一批 `static_chunk_c4` = `619.77 / 619.88 / 619.28`（spread 0.10%）。原因是**决策轨迹三轮不同**：第 3 次重复里 step 80 未产生动作、step 136 收缩到 1024、直到 step 184 才升到 2048；而另两次是 step 80 即达 2048、step 136 退到 1536（`n_clamped` 三轮分别为 8/6/**0**）。控制器以**带噪的实测信号**（TTFT P90 2217.9/2218.3/2220.0 ms）＋冷却计数为输入，因此路径本身不可复现。⇒ **后果**：c4 的 TTFT 优势按中位数是 −22.3%，但**最差一轮是 652.15 vs 619.77 = +5.2%（劣于基线）**，**符号在重复之间翻转**。故**当前证据不足以声称"c4 自适应稳定优于静态"**；必须先证明收益跨重复稳定，再谈幅度。<br>**（2026-09-24 更新）** P1S3D 重跑后：该限制**被削弱但未被解除**——旧门限下那个具体的拒绝机制（`tpot_p90` 97.8–99.2 跨过门限 99.0）在构造上已不可能（§3.3 **E1**），但 P2 用**旧门限**同样 3/3 收敛，故 n=3 无法把"新门限消除了它"与"没抽中"分开（§3.3 **E8**）。**引用 D2 时请一并引用 E1/E8。** | `results/raw/P1S3B-*adaptive_c4*`、`*static_chunk_c4*`，逐轮原始值；`controller_trace` |
| **D3** | **分块预填充在 `chunk_size=1024` 上复现了 P1S3 在 512 上得到的格局**（单变量 `chunk.enabled`，同批内部）：c2 TTFT P50 **−34.5%** 但 stall P99 **+238.8%**；c4 TTFT P50 **−63.1%** 但 stall P99 **+312.8%**；c8 TTFT P50 **+20.6%**、TTFT P95 +54.6% 而 stall P99 **−15.6%**、吞吐 **−11.1%**。⇒ 与 C2 更正后的表述**同向**，且这是在**另一个 chunk 取值**上的独立复现，支持"分块的尾延迟作用取决于并发与 prefill/decode 是否重叠"，而非某一取值的偶然。 | §1.2 表 `EXP-cp-11x` vs `EXP-base-11x` |
| **D4** | **修正后的控制器确实会双向动作**（对照 P1S3 的 `n_recovered=0`）：`adaptive_c2` 出现 grow 至 2048 的三次动作，`adaptive_c8` 出现收缩至 512 的动作并持续生效；`controller_trace` 的 `n_recovered`/`n_updates` 不再恒为 0。⇒ 第一处与第二处配置缺陷（`chunk_size_init == min` 死锁、`target_tpot_ms` 不可达）已修复，**第三处（`target_ttft_ms` 高于实测 TTFT P90 ⇒ grow 从不触发）也已修复**（改为 500 ms 后 grow 判据开始命中）。 | `controller_trace`；§1.2 配置对比表 |
| **D5** | **控制器动作确实落到了 prefill 上——但只有 c2/c4/c8 的部分动作**（`scripts/inspect_step3_runs.py --prefill-effect`）：c1 `changed` 决策 **0** 次（故 D1 中 c1 差异恒为 0，属"无事可做"而非"无收益"）；c2 10 次决策中 **3 次落到 prefill**（其后续 prefill 用 2048）、5 次因窗口内无 prefill 被后续决策覆盖、2 次此后再无 prefill；c4 6 次中 **1 次落到 prefill**、3 次被覆盖、2 次无对象；c8 6 次中 **1 次落到 prefill**（1024→512，此后 14 个 prefill 事件用 512）、5 次取值未变（`chunk_size` 已触下界 512，实际改的是 `max_batched_tokens`）。**"改了值却没生效"（`not_applied_to_prefill`）一次都没出现。** 另一独立自校验：两批 12 个基准 run 的实际 prefill 总 token **恒为 16384**（= 8 请求 × 2048），⇒ 控制器是**重新分配**了 prefill 工作，未丢失也未重复。 | `scripts/inspect_step3_runs.py --prefill-effect`；`tests/test_inspect_step3_runs.py`（36 项，含针对真机形状的读数断言） |

> **缺口（本节结论的适用范围，必须随结论一起引用）**
>
> 1. **计时口径为修复前**：本批 `note` 无 `readback_ms`，TTFT/E2E 记的是 logits 就绪时刻，
>    **系统性偏早**且偏差随 batch 增大（见 §1.2 限制 2）。D1/D2/D3 均为**同批内两臂相对比较**，
>    同口径同方向，故结论成立；但**绝对 TTFT 值不得与修复后口径的运行并表**。
> 2. **版本溯源未满足**：`git_commit` 36/36 为 `NOT_MEASURED`，`model_revision` 均为 `main`。
>    因此本批结论**无法**指认到具体源码版本；这是下一轮必须补齐的前置条件。
> 3. **负载形状**：每格仅 8 个请求、并发固定；到达是**闭模型补位**而非"全部同时到达"
>    （更正见 §1.4 限制 4）⇒ 是 `8/concurrency` 次"填满 → 该批预填充 → 排空"的重复瞬变，
>    **不是稳态**。D2 所揭示的"决策路径不可复现"很可能是**低请求数**放大出来的
>    （动作次数少、每次动作都显著改变全局）。**负载水平**（并发档）全程不变 ⇒
>    "负载水平变化下控制器如何表现"本批测不到；已有的是**补入扰动下的响应**（§3.3 E7/E9）。
> 4. **静态基线只扫了一个点**（`chunk_size=1024`）。"自适应优于**精心选择的**静态配置"
>    需要 512/1024/1536/2048 的完整基线，本批**不构成**该结论。
>    ⚠️ **2026-09-26 更正**：本条对**本批（P1S3）**成立，但**已不适用于当前状态** ——
>    S10（§3.5，15 次运行）后来补齐了 **512 / 1024 / 2048 / 不分块**（仍缺 `1536`）。
>    ⇒ 引用本条时**不得**再说"静态基线只扫了一个点"；改为"静态基线已扫 4 个候选（缺 1536），
>    但其负载/执行版本与当前不同，需先审计适用范围"。
> 5. `run.slo.tpot_ms=50` 未改 ⇒ 达标率三列仍恒为 0、不携带信息。

---

### 3.3 主体平台（MTT S4000）· 2026-09-24 · **P4 真机重跑（P1S3D）** `MEASURED`

来源：登记行见 §1.4，产物 `results/raw/P1S3D-*`（36 个目录）。**先读 §1.4 顶部那五条限制**。
本批要回答的唯一问题是 §4.3.4 末尾留下的那句"要回答'是否真的决断'，只能靠真机重跑"。

| 编号 | 结论（含方向与量级） | 依据 |
| --- | --- | --- |
| **E1** | **P4 在"静态判据"层面确实解除了 c4 的拒绝机制**：放大分支要求 `tpot_p90 < target·(1−h)`，旧门限 = `110×0.9` = **99.0**，而 §4.3.2 记录的实测工作点区间是 **97.8–99.2** ⇒ 该区间**跨过**旧门限（r2 的 step 80 就是以 `tpot_p90=99.23` **差 0.23 ms** 被拒，导致爬坡卡在 1536、后 4 个请求 TTFT 652.1）。新门限 = `120×0.92` = **110.4** ⇒ 同一区间**全部通过**，且上界距门限 **11.2 ms**（余量 0.0112 → 0.1052，**≈10×**）。**该拒绝对观测区间内任何取值都不再可能发生。** | §4.3.4 判据表；`scripts/calibrate_controller_gate.py --prefix P1S3D`（c4 `tpot_p90` min/p50/max = 97.9/98.8/571.2，放大 30/32、收缩 2/32） |
| **E2** | **P4 的作用面很窄：逐格比对 P1S3B 与 P1S3D，除 c4 外一切相同。** 决策计数逐格相同（c1 `129/0/0/128`、c2 `65/10/0/64`、c4 `33/6/0/32`、c8 `19/6/0/18`）；`prefill` 实际生效的 `chunk_size` 分布逐格相同（c2 `{1024:4, 2048:6}`、c8 `{1024:9, 512:14}`）；控制器读到的信号差 ≤0.5 ms（c8 `142.1→141.7`）。⇒ **门限改动没有把副作用扩散到别的格**；把它当成"整体重新调参"是过度解读。 | `--controller` / `--prefill-effect` 逐格两批对照；`calibrate_controller_gate.py --prefix P1S3B` vs `--prefix P1S3D` |
| **E3** | **达标率首次携带信息（阈值 50 → 100），而 P4 没有改变任何格的达标判定。** 阈值 100 下逐格 = c1 `1.000`／c2 `1.000`／c4 **`0.125`**／c8 **`0.000`**（三臂同）；全格 **58/96 = 0.604**。用 `scripts/calibrate_slo.py --prefix P1S3B --candidates 100` 把 **P1S3B 按同一阈值重算**，逐格与全格**完全相同**（同为 0.604）⇒ 门限改动与达标率无关，达标率只由**分块**决定。**副产物（本批的一条新权衡证据）**：c4 上开分块把达标率从 `static` 的 **8/8** 打到 `static_chunk`/`adaptive` 的 **1/8**（TPOT p50 `88.6 → 110.3` 跨过 100）——即"分块降 TTFT"在 c4 上是以**把 TPOT 推过 SLO** 换来的。 | §1.4 表；`scripts/calibrate_slo.py --candidates 100`（两批分别跑，读数同） |
| **E4** | **"部署戳通过"**不能**证明"输入是这一版"**：`SOURCE_PROVENANCE.json` 的 `digest_scope` 只含 `src/ascend_opt` / `benchmark` / `scripts`，**不含 `configs/`**。本批逐 run 补做了快照比对（36/36 与 `phase1_step3d/` 逐字段一致，退出码 0），并用 `phase1_step3c/` 做**反向对照**（退出码 **1**、逐字段列出 3 处差异）证明该检查**能失败**。另：该检查会**拒绝跨批次**的 `--tag`（子串匹配会把 P1S3/P1S3B/P2 一并扫进来，一个配置目录不可能同时是它们的输入）。 | `scripts/inspect_step3_runs.py --verify-config`；`tests/test_step3_runs_inspector.py`（新增 18 项，含 4 组突变验证） |
| **E5** | **本批噪声底（12 格，跨 3 次重复的稳健 CV）**：`TTFT p50` 中位 **0.0002** / 最大 **0.0007**；`TPOT p50` 中位 **0.0004** / 最大 **0.0024**；`stall p99` 中位 **0.0009** / 最大 **0.0341**。稳健 CV < 0.005 的「格 × 指标」**34/36**。⚠️ **`stall p99` 的最大值被单一格抬高**：`static_c8` 三轮 `498.28/464.59/464.22`（稳健 CV 0.0012 而普通 CV **0.0411**，单点离群被 MAD 抹平）。⇒ 引用本批 `stall p99` 的噪声底**必须用 3.41%**，用其余 11 格的中位 0.09% 会把该离群藏掉。 | §1.4 表；`scripts/inspect_step3_runs.py --tag P1S3D` |
| **E6** | **"自适应 vs 同分块静态"的格局在 P4 下重现（本批内部单变量，同口径，无跨批问题）**：c1 TTFT **−0.03%**（零动作，差异必为 0，属"无事可做"而非"无收益"）；**c2 TTFT −20.2%**（`570.00→454.96`）、stall p99 −10.5%、TPOT −0.23%；**c4 TTFT −22.2%**（`619.83→481.91`）但 **stall p99 +52.9%**（`372.36→569.51`）、TPOT −0.25%；**c8 TTFT +2.9%**、**TPOT P50 +1.45%**、stall p99 −6.3%。TPOT 在 c1–c4 的变化 ≤0.25%（符合"分块只调节 prefill"）。⇒ 与 **D1** 同向且量级一致：**c4 是明确的 TTFT↔stall 权衡，c8 是净负向**；"自适应更优"在 c8 上不成立。 | §1.4 表，3 次重复中位数；c1–c4 的差异绝对值均为噪声底最大值（TTFT 0.07%）的 ≥280 倍，c4 stall 为 (3.41%) 的 ≥15 倍 |
| **E7** | **P4 未解决 §4.3.4 记录的形式性障碍；闭环出现与"请求补入批次"同步的周期性调整（成因见 E9），本批**没有观察到控制器自激**；且多数决策没有落到 prefill 上。** c4 的**合格点在旧读数里"带内 0/32"——更正后的读法是：门槛拒 0/32 ⇒ 判据在该格上决断**（放大 30/32、收缩 2/32）。⚠️ 本节原写作"c4 的新门限仍被区间跨过"，那是**标定工具一个被收缩点污染的输出列的产物**，与它自己的"带内 0"自相矛盾，**已更正**（工具已修 + 反向控制测试，见 §4.3.4）；"形式障碍仍在"这一**结论保留**，但成立的理由是**负载连续变化会扫过整条区间**（与单格是否跨门限无关）。<br>⚠️ **步号归属更正（2026-09-24 晚）**：本节原把 `264 收缩 → 288 放大 → 392/416 收缩、放大` 接在 c4 之后，**那四步属 c2**。**c4 首轮只有 263 步**（`telemetry.jsonl` 263 行）、**共 6 次决策**：step 8 收缩（`TPOT P90=380.3`）→ 32/56/80 连续放大 → 136 收缩（`TPOT P90=571.2`）→ 160 放大，**到此结束**。**c2 首轮 515 步、共 10 次决策**：8 收缩 → 32/56/80 放大 → 136 收缩 → 160 放大 → **264 收缩 → 288 放大 → 392 收缩 → 416 放大**。<br>⚠️ **"反复调整"必须按成因拆成三件（原表述把三件并成了一句"周期性来回"）：** ① **补入驱动的周期性跟随**（c2 的 4 批、c4 的 2 批）——负载是闭模型补位（§1.4 限制 4），**补入批次数 = `num_requests / concurrency`**（c1/c2/c4/c8 = 8/4/2/1 批，3 次重复逐个一致）。每批一到，新 prefill 突发进入观测窗 ⇒ **该批之后第一个观测点收缩一次**（c2 批起点 step 0/129/257/385，收缩在 8/136/264/392 ⇒ 偏移 +8/+7/+7/+7 = 1 个 `control_interval_steps`）；突发过去后 `TPOT` 回落 ⇒ 按最小节拍（`cooldown_cycles=3` × 8 步 = 24 步）**连续放大 3 次回到上限**。**其"周期" = 补入批间隔 ≈ 一个请求的 e2e ≈ 129–132 个 decode 步，由负载决定，不是控制器的固有周期。** ② **稳态违约下的单向棘轮**（c8）——全程**只有 1 批**（8 请求同刻到达），`TPOT P90` 持续 ≈140 > 门限 ⇒ 每 24 步再收缩一次，共 **6 次、0 次放大**，值单调 7168→2048，**没有往复**。③ **零动作**（c1，8 批）——双目标全程在带内，**0 决策**。⇒ 本批能支持的是"**周期性调整跟随补入批**"；**不能**说"控制器自激振荡"（无补入的 c1 零动作、c8 单向），要检验自激需要**负载水平不变**的稳态段。<br>**`recover` 分支在所有格的所有重复里一次都没触发（`recover=0`）**。生效比例低：c2 的 10 次"已改"决策只有 **3 次**落到 prefill、c4 的 6 次只有 **1 次**、c8 的 6 次只有 **1 次**。⚠️ 另修一处**指标语义**：`n_decisions_changed` **不等于"取值变了"**——c8 的 6 次里有 **5 次 `value_unchanged`**（`chunk_size` 已触下界 512，实际动的是 `max_batched_tokens`）。**不得用 `changed` 计数当"动作次数"。** | `--controller`、`--prefill-effect`、`--cycles`（P1S3D）；`controller_trace` 的 `reason`/`step`/`signals` 字段；`requests.jsonl` 的 `arrival_ms`；`telemetry.jsonl` 的 `note` |
| **E8** | **⚠️ 归因更正：本批的 c4 收敛"不能"归给 P4。** P2（§1.3 的批次，配置为 `phase1_step3b` = **旧门限 110/0.1**，但已含计时修复）的 c4 **三次重复全部**是 `{1024:8, 2048:4}`——**与 P1S3D 相同**。而 P1S3B（旧门限、旧计时）3 次里有 1 次走到 `{1024:8, 1536:8}`。⇒ 观察到的是 P1S3B **1/3** 分叉、P2 **0/3**、P1S3D **0/3**，合计 **1/9**。这与"门限贴工作点 ⇒ 低概率触发"一致（E1 给出了机制），但**n=3 无法把"门限改动消除了它"与"P2 只是没抽中"分开**。<br>⇒ 因此：**§3.2 的 D2 限制只是被"削弱"，未被解除**。可以据 E1 说"已知的拒绝机制在构造上不再可能"；**不可以**说"c4 的跨重复可复现性已建立"——那需要更多重复（按 1/9 的经验频率，n=3 的 0 次分叉给不出统计保证）。<br>⚠️ 同时**修正 §1.3 注 2 的"与计时修复无关"**：P2 用**旧门限**却 3/3 收敛，与 P1S3B 的 1/3 分叉并列时，该归因**在 n=3 下不可判定**，应改写为"归因未建立"。 | `--prefill-effect` 逐重复（P2 三轮均 `{1024:8, 2048:4}`）；P1S3B `-r2` = `{1024:8, 1536:8}`；`calibrate_controller_gate.py --prefix P2`（旧门限下 c4 仍为 放大 29/32、收缩 2/32、带内 1/32） |
| **E9** | **⚠️ 收缩臂的成因是"`TPOT` 口径混入 prefill 步"，不是 decode 真的变慢；排除 prefill 步后，`tpot` 分支在 c2/c4 上**一次都不会触发**。** 控制器读的 `tpot_p90_ms` 来自 `recent_tpot[-32:]`，而 `recent_tpot` 存的是**相邻两次 token 发出之间的间隔**（`real_backend.py` 里 `recent_tpot.append(interval)`）⇒ **某一步做了 prefill，落在这两步之间的 token 间隔就直接包含该 prefill 的耗时**。观测窗 = **最近 32 个 token 间隔**，在 c2/c4 上 ≈ 决策点前 **8–16 步**（每步产 2/4 个 token），在 c8 上 ≈ **4 步**（每步 8 个）。实测（`telemetry.jsonl` 每步 `note` 给出 `prefill_tok` 与 `fwd_ms`）：c2 纯 decode 步 `fwd_ms=77.2`、窗内无 prefill 时 `tpot_p90` = **81.5–82.7**；批边界的 prefill 步（step 129/131，`fwd_ms=512.5`）落进窗口时 `tpot_p90` = **137.7–139.0**；收缩门限 = `120×1.08` = **129.6** ⇒ **纯 decode 的 81.8 在门限下、含 prefill 的 138.0 在门限上**。c4 同构且更极端：纯 decode **97.8–99.4**、批边界窗（含 4 个 prefill 步）**570.8–571.2**（5.8×）。**逐条核验（12 个 adaptive 运行 × 3 次重复）：每条由 `tpot` 分支触发的收缩，其观测窗内都含 prefill 步；每条由 `ttft` 分支触发的放大，窗内都没有 prefill。** **唯一例外**是 c8 的 step 32/56/80/104/128（窗内无 prefill、`tpot_p90` ≈140）——那是 8 并发下**真的**稳态违约，不是口径假象（即 E7 的成因 ②）。⇒ 这把长期登记的"控制器反馈口径待统一（token 级 ITL ≠ 请求级 TPOT）"从**疑点**变成**已测定的成因**：控制器把"prefill 挤占"读成"decode 变慢"，而它的响应是**调小 `chunk_size`**（prefill 更碎 ⇒ 含 prefill 的步更多）。⚠️ **该"反向放大"目前只是读数的推论，本批未做干预实验**，登记为待办。⇒ **2026-09-25 更新（见 §3.4 E10/E11）**：E10 在**开放环 piecewise**（负载连续变化）上**独立复现了本条的污染机制**（3 轮、动作步号逐位相同、触发时 `queue_depth=0`），并把"收缩的代价"量化到 **纯 decode 步 61→118 ms（+92%）**；而 **E11 否掉了"反向放大"这一单向说法**（9 次动作里 prefill 占比 3 升 5 降）。⇒ 本条成立的是"**污染存在且能触发动作**"；"收缩加重污染"**不成立**，不要引用。 | `real_backend.py`（`recent_tpot.append(interval)`、`recent_tpot[-32:]`）；`telemetry.jsonl` 每步 `note`（`prefill_tok`/`fwd_ms`）；`controller_trace[*].signals.tpot_p90_ms`；12 个 adaptive run 逐条扫描 |

> **缺口（§3.3 的适用范围）**
>
> 1. **跨 3B/3D 的任何绝对量比较都含两项变化**（门限 + 计时口径）。E2/E3/E8 用的是
>    "决策计数 / 生效分布 / 达标判定"这类**离散或阈值量**，对 ≤0.5 ms 的计时差免疫；
>    E1/E6 是**批内**读数，不受影响。**唯一不得不跨批的是 E8**，其结论本身就是"测不出"。
> 2. **非稳态**：每格 8 请求、**闭模型补位**而非"全部同时到达"（更正见 §1.4 限制 4）
>    ⇒ 是 `8/concurrency` 次重复瞬变，只有最后一批真的排空。**E7 里那个周期性调整
>    已由 E9 定因**（观测窗混入批边界的 prefill 步）。但它是否在**负载水平变化**
>    （低压→高压→恢复）下仍然出现，本批**测不到**（§3.2 缺口 3 同一问题）。
> 3. **静态基线仍只扫一个点**（`chunk_size=1024`，另在 `static` 臂上等价于"不分块"）。
>    要证"自适应优于**精心选择的**静态配置"，仍需 512/1024/1536/2048 的完整基线。
>    ⚠️ **2026-09-26 更正**：同 §3.2 的更正 —— 该说法**已过时**（S10 已扫 512/1024/2048/不分块，
>    仍缺 `1536`）。本条对**本批（P2）**成立，引用时须带上述限定。
> 4. **样本量**：全部结论建立在 n=3 上。E8 已经把这一点变成了一条显式结论，
>    其余各条的效应量（≥20%）远大于噪声底，受样本量影响较小。

---

### 3.4 主体平台（MTT S4000）· 2026-09-25 · **`load_ramp` pilot（P8FULL，开放环 piecewise）** `MEASURED`

> 轨迹：low 120 s @λ=0.08 → high 90 s @**λ=0.70** → recover 240 s @λ=0.08（共 92 请求、450 s）。
> 2 格（`ramp_static_c8` / `ramp_adaptive_c8`）× **3 轮穿插**；6 份产物同属一个溯源戳
> `69c16994d80b` / `dirty=False`；全部 `MEASURED`、`dropped=0`、`stall_reason` 空。
> 持续容量由 high 段独立定稿 **C = 0.356 req/s**（6 个 run 极差/中位 **0.0%**）。

| # | 结论 | 证据 |
| --- | --- | --- |
| **E10** | **★ E9 的机制在"负载水平连续变化"下同样出现，且可复现到步号；同时量化了收缩的代价。**（这直接回答了 §3.3 **限制 2** 登记的"是否在负载水平变化下仍然出现，本批**测不到**"。）<br>3 个 adaptive run 的**动作步号逐位相同**：`[1216, 1240, 1264, 1288, 1312, 1336, 2432, 2456, 2480]`（9 次，全部是收缩）。<br>首次动作：`t = 123.0 / 123.1 / 123.2 s`（**high 段刚开始**），**`queue_depth` 三个 run 都是 0**——即**在没有人排队时收缩**。<br>触发读数 vs 真实：报告 `tpot_p90 = 337.2 / 337.9 / 342.2 ms`；而同一观测窗（前 32 步）里 **纯 decode 步的 `fwd_ms` p50 只有 61.4 / 61.6 / 62.5 ms（n=28）**，**含 prefill 的步 p50 = 329.3 / 329.0 / 331.5 ms（n=4）** ⇒ **读数 ≈ 真实 decode 的 5.5 倍，且几乎全部来自那 4 个 prefill 步**。<br>**收缩的代价被直接量到**：5 次收缩把 `max_batched_tokens` 从 8192 压到 **2048**（下界），**纯 decode 步的 `fwd_ms` p50 随之从 61/62/62 ms 升到 118/118/118 ms（≈ +92%）**。<br>⇒ 这不只是"读数错"：**动作本身把真实的 decode 延迟弄慢了近一倍**，而它要修的"问题"（337 ms）在纯 decode 口径下并不存在。这解释了同一批里 adaptive 的 `tpot` p50 高于 static（182.4 vs 175.9 ms）。 | `results/raw/P8FULL-load_ramp-ramp_adaptive_c8{,-r1,-r2}`：`meta.controller_trace`（`changed=true` 的项）+ `telemetry.jsonl` 每步 `note`（`prefill_tok`/`decode`/`fwd_ms`）+ `max_batched_tokens` 列 |
| **E11** | **⚠️ E9 末尾那句"反向放大"（收缩 ⇒ prefill 更碎 ⇒ 含 prefill 的步更多 ⇒ 污染更重）在本批被否掉：它不是单向机制。** 逐次动作取"前 64 步 / 后 64 步"的**含 prefill 步占比**变化：`+31pp / −2pp / −42pp / −14pp / +25pp / +50pp / −12pp / −6pp / −3pp` ⇒ **3 升、5 降、1 持平**。<br>⚠️ 混淆因素：含 prefill 的步占比还由**到达与积压状态**决定（high 段前后 demand 差很多），所以这个相关性本身也不是干净的因果检验。<br>⇒ 正确表述收窄为：**"含 prefill 的间隔会污染控制器读数"已由 E9+E10 支持；"收缩会加重污染"证据不足，不得写成机制。** 要检验它需要**负载水平不变**的稳态段。 | 同上 9 次动作前后各 64 步的 `prefill_tok>0` 占比 |

#### 3.4.1 判据口径的**干预实验**（`edfc0b63` · 2 臂 × 3 轮穿插 · 同版本对照）

> A 臂 = `ramp_adaptive_c8`（判据 `tpot_p90`，含 prefill 污染）；B 臂 = `ramp_adaptive_decode`（判据 `decode_p90`，纯 decode）。
> 两臂**只差 `signal_criterion`**（`signal_mode: layered` 在两组共用）⇒ 单变量。6 份产物同戳 `edfc0b63c8c7` / `dirty=False`、
> 全 `MEASURED`、`n=92` 全成功、`dropped=0` ⇒ 可比性前置检查通过。

| # | 结论 | 证据 |
| --- | --- | --- |
| **E12** | **★ 干预**改变了触发值的**性质**（污染 5.5× → 1.33×），但**净收益很小、且代价是 stall**。<br>**① 触发值的性质**（报告判据 ÷ 同窗纯 decode `fwd_ms` p50）：A = **5.47 / 5.47 / 5.73**；B = **1.33 / 1.33 / 1.32** ⇒ 换判据后**报告的就是纯 decode 值**，而那批"在纯 decode 还只有 61 ms 时收缩"的动作消失（首次动作由 `t≈123.0 s` 推迟到 **`t≈131.2 s`**，且触发值 138.0 **> 门限 129.6** ⇒ 属**合法触发**：8 槽满批时纯 decode 真的变慢）。<br>**② 用户指标（3 次重复中位，组内极差）**：达标率 A `0.1957`(0) → B **`0.2065`**(0)（**18 → 19 个请求**）；`tpot` p50 `182.53`(0.24) → `180.79`(0.46)；**`stall` p99 `330.51`(20.5) → `410.30`(2.9)** ⇒ **回到 static 的水平（412.4）**。<br>**③ 怎么读**：机制成立（读数不再被污染），但 **净收益只有 +1 个达标请求**，且 B **放弃了 A 靠过度收缩换来的 stall 改善**（"低 stall"本来就是把 decode 摊薄的副产品）⇒ **这是权衡，不是"换判据就更好"**。<br>**④ 与 E10 的因果链**：E10 说"污染读数触发收缩、收缩又把纯 decode 从 61 推到 118 ms"。B 臂证明前半（污染触发**被消除**）；**后半仍在**——B 仍会合法收缩，之后纯 decode 同样到 118 ms。⇒ **"改判据"只能消除*不该有的*动作，不能消除*收缩本身*的代价。** | `results/raw/S9AB-*`（6 份）；`scripts/analyze_s9_ab.py --tag S9AB`；`meta.controller_trace` + `telemetry.jsonl` 的 `note` |
| **E13** | **⚠️ 第 09 步登记的 4 条预测 3 条不成立，其中一条是*预测本身*写错了（不是机制错）。** ① 判 fail（B 在 `t≈131 s`、`queue_depth=0` 时动了）——⚠️ 但这条登记**写得不好**：它把"队深 0"当成"没有真实压力"，而 8 槽全满、批内 8 条序列时纯 decode 本来就会变慢 ⇒ **post-hoc 观测**（不改判据）：B 的触发值 138 ms 是**纯 decode** 且超门限，属合法。② 判 fail（纯 decode 之后仍到 118 ms）——**预测设计缺陷**：它假设"不再收缩"，但 B 会**合法地**收缩。③ 判 fail（`tpot` p50 落在干预前一侧：180.8 离 182.4 比离 175.9 更近）⇒ **干预对 `tpot` p50 几乎没有改善**。<br>⇒ **登记为"预测与结果不符"，不事后改判据**；改判据的那条另列为 post-hoc 观测。 | 同上；判据与预测原文见 `NEXT_SESSION_TASKS.md` 第 09 步 §二/§五 |

**本批的限制**：

1. **臂间对照不构成结论**：static 达标率 `0.2174 ×3` vs adaptive `0.1957 ×3`、达标吞吐 5.64 vs 5.08（−10%）
   —— 组内极差为 **0**（可复现），但**效应量只有 2 个请求 / 92**，且是**进程内**重复；
   同时 adaptive 的 `stall p99` **低约 20%**（330 vs 412）而 `tpot` p50 高 3.7% ⇒ **是权衡不是纯损失**。
   ⇒ 登记为**待检验信号**，交由第 09（阈值冻结）/10（静态响应面）步处理；注意 **S5**。
2. **`max_batched_tokens` 被压到 2048 是下界**：本批只观察到"一路收缩到边界、0 次放大"，
   与 E7 成因 ② 同型 ⇒ **不能**据此说"控制器会振荡"。
3. **`recover` 分支本批是否触发未逐条核对**（E7 已登记它在 P1S3D 全格 0 次）；
   ⚠️ 且本批控制器**全程只朝一个方向动**，所以它是"单向棘轮"而非双向控制。


---

### 3.5 主体平台（MTT S4000）· 2026-09-25 · **静态响应面 + Best Static 冻结（S10）** `MEASURED`

> 5 个静态候选 × 3 轮穿插（**15 次运行**），同戳 `32f3fac5deef` / `dirty=False` / 全 `MEASURED` /
> `n=92` 全成功 / `dropped=0`。负载 = `load_ramp` 轨迹（low 120 s → high 90 s → recover 240 s）。
> 主指标按**段**取，用分阶段报告的 **window 口径**（分母 = 段墙钟长度），跨 3 次重复取**中位**。
> 选优规则（支配 + 平局）**在取数之前**已登记（`NEXT_SESSION_TASKS.md` 第 10 步 §二）。

**响应面（达标吞吐 tok/s，三段）**

| 格 | `chunk_size` | `budget` | low | **high** | recover |
| --- | --- | --- | --- | --- | --- |
| `ramp_static_c8` | 1024 | 8192 | 9.600 | 1.422 | 4.800 |
| `ramp_static_c8_ck512` | **512** | 8192 | 9.600 | **0.000** | 4.267 |
| `ramp_static_c8_ck2048` | 2048 | 8192 | 9.600 | 1.422 | **5.333** |
| `ramp_static_off_c8` | **off** | 8192 | 9.600 | 1.422 | **5.333** |
| `ramp_static_c8_bt2048` | 1024 | **2048** | 9.600 | 1.422 | 4.800 |

| # | 结论 | 证据 |
| --- | --- | --- |
| **E14** | **静态最优集不随负载段迁移；唯一明显的差异是"细分块在高负载段把达标吞吐打到 0"。** 逐段看：**low 段五格完全相同**（9.600）；**high 段 `ck512` 归零（0.000）而其余四格相同**（1.422）；**recover 段粗分块更好**（`ck2048`/`off` 5.333 > `c8`/`bt2048` 4.800 > `ck512` 4.267）。<br>⇒ 按预注册规则：**Pareto 前沿 = {`ck2048`, `off`}**（两者**逐位相同**），平局规则取 **`chunk.enabled=False` 优先** ⇒ **`Best Static` = `ramp_static_off_c8`（不分块 / budget 8192）**。<br>⇒ **S5 未触发**：没有单一候选支配其余全部（`off` 支配 3 个，但与 `ck2048` 完全并列）。<br>⚠️ **"最优点迁移"不成立**（同一批候选在三段都 ≥ 其余）——这是**否证**，不是"没测到"：迁移若存在，应表现为"某段最优的 `chunk` 值不同"，实测没有。 | `results/raw/S10-*`（15 份）；`scripts/analyze_s10_surface.py --tag S10`；图 `results/figures/s10_surface.png` + `.source.txt` |
| **E15** | **⚠️ 两条必须与 E14 一起读的限制，否则会误引。**<br>① **`spread(s) = 0`**：同一格的 3 次重复在段级达标吞吐上**逐位相同**（如 high 段 `1.422222 ×3`、`0.0 ×3`）⇒ 噪声底为 0，支配判据退化成"三段都不差、且至少一段严格更好"。原因是本负载是**开放环确定性**的（到达由 λ(t) 与外生种子决定，不经客户端容量反馈）。⇒ **不得**把"批内极差 0"读成"这台机器没有噪声"（那是**进程内、段级派生量**的极差，见第 08 步 §十）。<br>② **`ck2048` 与 `off` 是构造性等价，不是两个独立数据点**：prompt 恒为 **2048**，所以 `chunk_size = 2048` 的"分块"是**恒等操作**，两者三段逐位相同。⇒ 响应面实际只有 **4 个不同配置**；而这条等价**同时是一次仪器一致性检验**（它通过了：两个应当等价的配置给出逐位相同的数）。 | 同上；`results/processed/s10_surface.json` 的 `per_run` 原始值 |

**本节的限制**：

1. **`stall p99` 在 low / recover 段缺失**：这两段 cohort 的 n 分别是 **10 / 19**，低于分位所需
   的 20 ⇒ 分阶段报告**拒绝给出** P95/P99（第 05 步的设计）。图上的 stall panel 因此稀疏 ——
   这是**规则生效**，不是数据丢失。
2. **`budget` 轴只扫到 2048**，未见差异（1.422 / 4.800 与基线相同）；要证"预算无关"需要更宽的扫描。
3. **只扫 `phase1_anchor` 的请求形状**（prompt 2048 / gen 128）。E15② 的等价性**依赖**
   `prompt == 2048`，换形状即失效。

---

### 3.6 主体平台（MTT S4000）· 2026-09-30 · **执行优化线 EXECOPT3（S01/S02/S03 三开关真机受控对照）** `MEASURED`

> 协议与登记见 §1.5。**4 臂 × 3 轮穿插**，`short_chat`，控制器**关**（`n_control_decisions=0`）
> ⇒ 纯执行路径 A/B，无调度混淆。批次 `12/12`、`balanced=true`、`failures=[]`。

**判定用的噪声底（先立标尺，再看效应）**：baseline 自身跨 3 轮极差
`tpot_p50` = **0.444%**、`ttft_p50` = **0.117%**。**效应小于此即不可分辨。**

| # | 机制 | 相对 baseline（逐轮配对，3/3 同号） | 噪声底倍数 | **判定** |
| --- | --- | --- | --- | --- |
| **E16** | `logits_last_only` + `sample_needed_only`（`s01_both`） | **TTFT p50 −4.2%**（188.78 vs 196.98 ms）｜tpot p50 **无变化**（符号 3 轮为 `+ − +`，跨噪声）| **36×**（TTFT） | ✅ **机制成立且方向符合预期**：只算最后一个位置的 logits ⇒ 首 token 更快；decode 稳定态不受影响。**唯一一个测出正向收益的开关。** |
| **E17** | `retire_empty_cache=False`（`s02_no_empty_cache`） | tpot p50 −0.13%、ttft p50 −0.1%（均 3/3 同号）| **≈0.3×**（**低于**噪声底）| ⚪ **本协议下不可分辨**。符号虽一致，但幅度小于 baseline 自身抖动 ⇒ 只能说"在本负载/本批量下无可测差异"，**不得**写成"无收益"（机制在更大批量或更碎片化的分配模式下可能才显现）。 |
| **E18** | `fuse_gate_up=True`（`s03_fuse_gate_up`） | **tpot p50 +1.10%**（119.42 vs 118.12 ms）｜p90 +0.85%、p99 +1.02% 同向｜吞吐 −0.9% | **2.5×** | ⚠️ **真实的负面结果**：融合 gate-up **稳定地变慢**。机制定位：prefill / queue / stall **三段均无变化**，差异**只出现在 decode**（tpot p50/p90/p99 同向抬升 +0.8%~1.1%）⇒ 是**每步 decode 的固定开销**，不是被 prefill 挤占。推定成因：融合算子在 `eager` 图模式下引入了额外的 `.contiguous()` / 算子边界开销，超过省下的 kernel 启动时间。**该结论是真机实测，与"CPU 并行度不足"无关。** |
| **E19** | `s01_both` 第二次独立批次（EXECOPT4 的 EXP-exec-156） | **TTFT p50 −4.2%**（188.80 vs 197.13 ms），3/3 同号 | **≈30–100×** | ✅ **独立复现 E16**：不同批次、不同轮次抽样，同一协议下效应量与符号一致 ⇒ E16 由"单批"升格为**两批一致**。仍限 `short_chat` + `eager` + 并发 8。 |
| **E20** | `persistent_kv_slots=True` + `capacity=1024`（`s04_pkv`） | ★★ **在机制真正被行使的样本上，`pack` 变差 72%**：单请求纯 decode 尾段（**77 步 = 快路径的全部覆盖**）`pack` 总量 **109.02 → 187.52 ms**，3 轮 = 187.13 / 187.52 / 187.92（极差 **0.4%**）；对照组 `s01_both → s01_both_s04_pkv` 同样 **108.98 → 187.31（+71.9%）**。全窗口只有 **+0.19%**，那是被 **6.74% 覆盖率稀释**的结果<br>⚠️ **2026-10-03 口径更正**：`6.74%` 是**数量**覆盖率，不是耗时占比 ⇒ 解释全窗口差必须用**时间权重**（EXECOPT5 实测尾段耗时段份额仅 ~0.52%）＋加性核对（残差 0.047%）。详见 **§4.4 A4**。 | **≈ 175× 臂内噪声底**（尾段）| ⚠️ **真实的负面结果**（**本条推翻 2026-09-30 初次登记的"测不出收益"**）：快路径**没有省下搬运**，反而每步多做一次**整段 `copy_`**——把 HF 返回的**累积** KV 写进常驻缓冲；而原路径遇到等长历史**本来就不 padding**，写回只是**切视图**、不拷贝。⇒ 在**能行使**的地方（6.74%）它是**净成本**；在**不能行使**的地方（93.3%）它什么也没做。**不得**再写"差异在噪声底内不可分辨"。<br>⚠️ 这是**主机计时边界下的 `pack` 字段**，不能直接读成"GPU 搬运时间增加 72%"。 |

**本节必须一起引用的限制**：

1. **`s02` 的"不可分辨"不是"无收益"**（E17 已注明）：判定依据是"幅度 < 噪声底"，
   属于 §4.1 的**测不出结论**类，不是负面结论。
2. **`s03` 的负面结论限定在本协议内**：`graph_mode=eager`、`short_chat`、并发 8。
   换 `torch.compile` / 更大批量 / 更长序列时融合的摊销可能翻转；**不得**外推为"融合无用"。
3. **`n_samples=1080` 步 / `n_requests=64`**，单轮 176 s —— 效应量级在 1% 附近时，
   本批的统计功效**仅够**分辨 ≥0.44% 的 tpot 差异。
4. **`git_commit=NOT_MEASURED`（`git_source=stamp_stale`）**：与本仓库其他批次同一限制
   （见 §4.3 的 P1）——本批数字**没有源码版本绑定**。⚠️ 另发现一处**溯源写法不一致**：
   baseline 臂 `model_revision="main"`，而三个处理臂记录为固定 sha
   `a09a35458c702b33eeacc393d103063234e8bc28`。四臂实际加载的是**同一份权重快照**
   （revision 已在远端 pin，见 preflight 的"revision 快照在位"），故**不构成变量污染**；
   但该字段在 baseline 上写法不同，引用时须按"已 pin"理解。

---

### 3.7 主体平台（MTT S4000）· 2026-10-03 · **S04 强制重入的数值验证（`MEASURED`，正确性而非性能）**

**这条回答的问题**：S04 快路径在**重入**（成员退出后剩下的重新等长）时读到的
KV 是否与不开启时**逐位相同**。它**不**回答性能（性能见 §3.6 E20 / 后续 `EXECOPT5`）。

| 项 | 值 |
| --- | --- |
| 臂 | `baseline = configs/mthreads/exec_opt/s01_both.yaml`；处理臂由脚本派生（**只多** `persistent_kv_slots: true` + `persistent_kv_capacity: 1024`） |
| 场景 | `plen=9, plen_long=17, gen=24, steady=3, reenter=4, capacity=256`；**无预热** |
| 路径覆盖 | `n_fastpath_steps=7`、`n_fallback_steps=5`、**`n_resyncs=2`**（重入真的发生）；`fallback_reasons={'new_sequence': 3, 'unequal_kv_len': 2}` |
| 数值 | **token 23 个逐一相同**；**logits 摘要 23 个采集点全部相同**（`sha256` 逐位）；**重入时刻**与**结束时刻**的逐层 KV **逐位相同** |
| 记账 | `tokens_out`：baseline=23、pkv=23、**脚本实观察=23**（三方相等） |
| 源码身份 | `tree_digest=bff7f7080b67914e…`（= 本机重算值，= 远端戳） |
| 环境 | `torch 2.2.0`、`torch_musa 1.3.0+81caf0a`、`MODEL_REVISION=a09a35458c702b33eeacc393d103063234e8bc28` |
| 判定 | **PASS**（10/10 条预先冻结的判据全绿） |
| 证据 | `results/raw/_s04_verify/s04_reenact_verify_20261002T194537Z.json`（本机/远端 sha256 一致 `e14c75e03a7eb8e3…`） |

⚠️ **必须随结论引用的口径**：

1. **`warmup_deduction_verified = false`**：本次**未预热**，`measured_window.basis=no_warmup`。
   ⇒ "预热扣除"这条**从未被验证**。**不得**把它读成"已扣除预热"。
   要验证它必须显式 `--warmup-s > 0`，并检查 `basis == total_minus_warmup`
   且 `n_forwards_measured < n_forwards`（脚本就是这么判的）。
2. 本验证**只覆盖"快路径 → 回退 → 重入"**这条序列；**不含**过载 / SLO，
   **也不含**"回退之后未重入"的尾部（那条路径由单测覆盖）。
3. 这是**正确性**结论，**不构成**任何性能主张。性能由 `EXECOPT5`（2 臂 × 5 轮）回答，
   且 §3.6 E20 已登记的"尾段 `pack` +72%"是一条**独立**的负面结果，两者不互相削弱。

### 3.8 主体平台（MTT S4000）· 2026-10-03 · **EXECOPT5：S04 的 5 轮独立复现（`MEASURED`）**

> ⚠️⚠️ **本条是"限定实现与负载的局部负结果"**（*qualified local negative result*）。
> 它**只**说明：在**当前这份实现**（快路径每步对整段 KV 做一次 `copy_`）与
> **当前这份负载**（`short_chat`、并发 8、64 请求、`graph_mode=eager`、
> `max_num_seqs=32`、分块预填充 8192）下，S04 的常驻 KV 快路径在
> **它能被行使的样本上**是**净成本**。
> **不得**外推为"S04 思想无用"、"搬运优化一定失败"、
> 或"换成别的实现/负载也一定更慢"——本批**没有**测试那些。

**一句话**：S04 在**它能行使的样本**（末尾 77 步纯 decode）上使 `pack` **+70.9%**
（**5 轮逐轮配对比值的中位数**），且该结论在**独立第二批 5 轮**里**逐轮同号（5/5）**复现。

| 量 | baseline | pkv | 比值 |
| --- | --- | --- | --- |
| **尾段 77 步 `pack`Σ（有效样本）** | 108.80 ms | 185.95 ms | **+70.9%**（5 轮**配对比值**的中位数；极差 16.3%；**5/5 同号**）|
| 全窗口 1,080 步 `pack_tel` | 20,968.8 ms | 21,055.9 ms | +0.366% |
| **时间权重**（尾段 / 全窗） | **0.5189%** | **0.8831%** | — |

* 与 §3.6 E20（3 轮，`109.02→187.52 = +72.0%`）**独立一致**，相差 ~1.5%。
* ★ **换算是"时间权重"，不是"数量覆盖率"**：`6.74%`（= 77/1143）是**数量**份额；
  尾段占**耗时**的份额只有 **~0.52%**（baseline）/ **~0.88%**（pkv）——因为尾段是
  单请求纯 decode，**单位步搬运量远低于**前段的批量步。
  **加性核对**：`20,968.8 + 77.15 = 21,046.0` vs `21,055.9` ⇒ 残差 **9.9 ms（0.047%）**
  ⇒ 全窗口那 87.1 ms 的差里 **88.6% 直接来自尾段**。
  ⇒ 正确表述：**不是"被稀释得没有"，而是"成本真实存在、但只占总耗时的约 0.9%"**。
* ⚠️ **`fwd` 上本批不足以判定**：原先写的"两臂 `fwd_tel` 区间重叠 ⇒ 只影响 `pack`"
  已**撤回**——**区间重叠不是检验**，n=5 下重叠推不出"无差异"。
  机制解释（快路径改的是搬运/拼接）**是设计意图**，本批数据**不能**证实。
* ⚠️ 引用限制：`warmup` 轮数两臂不同（2 vs 1）⇒ **driver 口径不可跨臂比**；
  每个 run 恰有 1 个 >1 s 的**起手步**（两臂都有，10/10）⇒ 服务指标需专门口径；
  **环境证据不完整**（见 §4.4 A7：产物里的 `hardware_env` 来自**另一个容器**）。
* 证据：`results/raw/EXECOPT5-*`（10 目录）；批次 summary 见
  `/root/ascend_probe_d1/exec_opt_summary_EXECOPT5.json`。

---

### 3.9 主体平台（MTT S4000）· 2026-10-03 · **S01 的实际部署收益得到独立支持（`MEASURED`，T1）**

**结论**：在 `short_chat` / 并发 8 / 64 请求 / 控制器关 / S04 关 / 单变量只差两个
S01 开关的条件下，S01（`logits_last_only` + `sample_needed_only`）**降低 TTFT**。

> ★★ **2026-10-03 更正（分析口径）**：本节初版把 `SLO 达标率` 写成 **1.0000**，
> 那是**把"配对比值"读成了"绝对率"** —— 两臂达标率都是 **2/64 = 3.125%**，
> 比值恒为 1.0000 只说明"两臂相同"，与"达标率是 100%"无关。
> 同时初版把吞吐方向写成了"1/5 / 4/5"（按"更小=更快"的时延方向判），
> 而吞吐**越大越好** ⇒ 正确表述是 **改善 4/5、变差 1/5**。
> 两处均已按 `scripts/analyze_s01_recheck.py` 的四元口径修正。**原始数据未改**。

| 指标 | 对照绝对（中位） | 处理绝对（中位） | 配对比值中位（处理/对照） | 更好 / 更差 | 5 轮区间 |
| --- | --- | --- | --- | --- | --- |
| **TTFT p50（主指标）** | 190.6 ms | 182.9 ms | **0.9569（−4.31%）** | **5/5 更快** | [0.9547, 0.9630] |
| TTFT p95 | 914.3 ms | 877.1 ms | 0.9587（−4.13%） | 5/5 更快 | [0.8983, 0.9631] |
| TTFT p99 | 1279.1 ms | 1228.4 ms | 0.9596（−4.04%） | 5/5 更快 | [0.9087, 0.9646] |
| TPOT p50 | 117.8 ms | 117.6 ms | 0.9994（−0.06%） | 3/5 / 2/5 | [0.9903, 1.0199] |
| TPOT p90 | 121.7 ms | 121.5 ms | 0.9974（−0.26%） | 3/5 / 2/5 | [0.9891, 1.0098] |
| 吞吐（tok/s） | 64.2 | 64.4 | 1.0021（+0.21%） | **改善 4/5** | [0.9888, 1.0103] |
| 达标吞吐（tok/s） | 1.919 | 1.923 | 1.0021（+0.21%） | **改善 4/5** | [0.9886, 1.0104] |
| **SLO 达标率** | **0.0312 = 2/64** | **0.0312 = 2/64** | 1.0000（两臂**相同**） | 0 / 0 | [1.0000, 1.0000] |
| 正式窗口耗时（`summary.wall_ms`, ms） | 122430.9 | 122214.6 | 0.9979（−0.21%） | 4/5 更短 | [0.9885, 1.0076] |
| 前向+打包（`fwd_ms+pack_ms`, ms） | 121220.0 | 120747.9 | 0.9961（−0.39%） | 4/5 更短 | — |
| 失败 / 未完成 | **两臂全为 0**（5/5 轮） | 两臂全为 0 | — | — | — |
| 每轮 >1 s 的步数（`fwd`） | 1（**起手步**） | 1（**起手步**） | 1.0000 | 0 / 0 | — |

⚠️ 上表**两个耗时口径不得互换**：`summary.wall_ms` 是**正式窗口**（用户指定口径），
`fwd_ms+pack_ms` 是前向/打包累计（本批两者差约 **1000 ms ≈ 1%**）。

**判定（预注册规则，读数据之前冻结在 `scripts/analyze_s01_recheck.py` 的 `PREREG`）**：
主指标 TTFT p50 需 **≤ −2% 且 5/5 同向** ⇒ **SUPPORTED**。

**噪声底（本批同协议自比）**：对照臂自身跨 5 轮极差 `ttft_p50` **0.656%**、
`tpot_p50` 2.610% ⇒ 效应 ≈ **6.6×** 噪声底；且 5 轮配对比值**全部 < 1**（区间上界 0.9630）。

**与既有证据的关系**：`EXECOPT3` 曾测到 S01 的 TTFT p50 **−4.2%**（36× 噪声底）——
本批 **−4.31%** 与之**独立复现且高度一致**；而本批的四类身份**全部闭合**，
因此**这一条**可以作为量化主张的直接证据（`EXECOPT3` 当时因缺源码身份只能进 T2）。

⚠️ **适用范围（不得外推）**：
* 只对**本实现**（`Transformers` + `eager` + 自建迭代级执行器）与**本负载**
  （`short_chat`、并发 8、64 请求、`max_num_seqs=32`、分块预填充 8192）成立；
* **TPOT 未见变化**（方向 3/5 vs 2/5，区间跨 1.0）⇒ 本批**不足以判定** TPOT；
  机制上也自洽：S01 减的是**每步固定开销**，先体现在 TTFT。
* **达标率两臂同为 2/64 = 3.125%（配对比值 1.0000 只表示"相同"）** ⇒ 本负载
  **不区分两臂的服务质量**，因此**不能**据此声称"SLO 达标率提升"。
  ⚠️ 更需注意：**绝对达标率本身只有 3.125%** —— 这直接说明本负载的 TPOT
  （p50 ≈ 117 ms）**多数超过 SLO 的 100 ms**，"本负载下两臂服务质量相同"与
  "服务足够好"是**两件事**，不得混为一谈。
* `git_dirty=true`：`git_commit` 只是**分支点**，本批的源码身份以
  `tree_digest=472697170d43…` 为准。
* 短启动/数值验收产物（`results/raw/_smoke/*`）标 `NOT_MEASURED`，**不参与**本结论。

**判据的"事前性"声明（如实）**：`PREREG` 随 `scripts/analyze_s01_recheck.py`
**首次提交一同写入**，仓库里**没有**一份"实验开始之前独立留存"的副本可比对。
因此它的地位是**"随分析脚本版本化的判据"**，而**不是**"有独立封存证据的预注册"。
引用时**不得**升级表述为"已预注册"。

**证据入口**：`results/raw/S01RECHECK-*`（10 目录）；
分析 `results/processed/s01_recheck_analysis.json`；
复算 `python scripts/analyze_s01_recheck.py --tag S01RECHECK --control s01_reference`；
远端汇总 `/root/ascend_probe_d1/exec_opt_summary_S01RECHECK.json`。

## 4. 负面结果归档

规范第 19 节：负面结果必须保留并分析，不得只报好消息。
「本次测不出结论」与「测出了坏结论」要分开表述（AGENTS.md 附录 A5）。

### 4.1 测不出结论（**不是**负面结论）

| 编号 | 内容 | 为什么测不出 | 依据 |
| --- | --- | --- | --- |
| **N1** | `adaptive` vs `static_chunk`：TTFT P50 差 0.02%~0.33%、TPOT P50 差 ≤0.8 ms，**全部低于噪声底**，也落在同格三次重复的区间内。 | 不是"自适应无收益"，而是本轮 `adaptive` 臂因**配置缺陷**在数值上退化为 `static_chunk`（`chunk_size_init` 等于下界 ⇒ 被扫描维度不可动；`target_tpot_ms=50` 不可达 ⇒ 控制器只收不放、12/12 个自适应 run 的 `n_recovered=0`）。**该退化有数值旁证：四个并发档的 stall P99 与 `static_chunk` 中位差值均 ≤0.5%（见 C5）**，即它不只"像是"没动，而是**逐档等值**。 | `PERFORMANCE_MATRIX_PHASE1.md` §6.3、§7.3.3；C5 |
| **N2** | 失败/拒绝/未完成行为。 | 36 格全部 `n_failed=0`/`n_rejected=0`/`n_unfinished=0` —— 本负载**产生不出**这些结局，而非"测过且没有"。 | 同上 §7.3.3 |
| **N3** | 前缀缓存收益。 | `cache.enabled=false`，缓存从未开启过；`cached_tokens=0` 是结构后果。**不得**据此说"前缀缓存无收益"。 | 同上 §1.1、§7.3.3 |
| **N6** | `S14B3` 的 `cost` 臂：成本模型**没被用上**（`n_fallback=536/537=99.81%`，成因**单一** = `last_fallback_reason`「模型范围外（不可外推）」，`n_no_candidates`/`n_memory_conservative`/`n_prefill_starved` **全 0**）。 | 不是"成本模型自适应无收益"，而是 `model_v5` 的 `kv` 上界 **2048** ≪ `lenmix` 格的 `kv` 上界 **8192** ⇒ 查表在**结构上**必然回退。机制**没被执行**，收益既未验证也未否证。⚠️ **严禁**据此写"cost 更差"。 | 本文件 §`S14B3` 第 3–4 节；`results/processed/s14b3.json` |
| **N7** | `S14B3` 的 `static` 与 `threshold` 两臂**靠 `kind` 分不开**（`kind` 都是 `threshold`；`static` 的 `kind` 是惰性遗留）。 | 不是"两臂相同"，而是**判据写错**：分臂只能靠 `controller.enabled`（`False` / `True`）。任何按 `kind` 分组的脚本都会**静默合并**这两臂。 | 本文件 §`S14B3` 第 3 节末注 |

### 4.2 本轮无效（实验未测到它声称要测的东西）

- **`adaptive` 臂（P1S3，已作废）**：设计意图是"自适应切片 vs 静态切片"，实际测到的只是
  "不可达 TPOT 目标下控制器退化为单向棘轮"。
  引用本批 `adaptive` 数据时**必须**同时引用该结论边界。
  修正方案见 `PERFORMANCE_MATRIX_PHASE1.md` §6.3。
  **状态更新（2026-09-23）**：修正版矩阵 `P1S3B` 已跑完（36/36），修正**确实生效**
  （见 §3.2 **D4**：控制器开始双向动作，grow 判据开始命中）。因此该项
  **不再"无效"**——但换成了三条新的、更具体的限制，见下 §4.3。

- **N8 · `s03_fuse_gate_up`（EXECOPT3，2026-09-30）—— 测出了**坏**结论，不是"测不出"**：
  融合 gate-up 使 `tpot_p50` **+1.10%**（3/3 同号，2.5× 噪声底），吞吐 −0.9%。
  机制定位：差异**只在 decode 段**，prefill/queue/stall 不变 ⇒ 每步固定开销。
  ⇒ 归档为**负面结果**（机制在本协议下**有害**）。⚠️ 边界：仅 `eager` + `short_chat` + 并发 8；
  **不得**外推为"融合无用"（见 §3.6 E18 限制 2）。
  产物：`results/raw/EXECOPT3-short_chat-s03_fuse_gate_up*`（3 个目录）。

- **N9 · `s02_no_empty_cache`（EXECOPT3，2026-09-30）—— "本轮无效"（测不出）**：
  `retire_empty_cache=False` 相对 baseline 的 tpot/ttft 变化（−0.13% / −0.1%，3/3 同号）
  **均小于 baseline 自身跨轮极差**（0.444% / 0.117%）⇒ 本协议下**不可分辨**。
  ⇒ 归档为**测不出结论**（§4.1 类），**不是**"该开关无收益"。
  机制在更大批量 / 更碎片化的分配模式下可能才显现 —— 需另行设计协议。
  产物：`results/raw/EXECOPT3-short_chat-s02_no_empty_cache*`（3 个目录）。

### 4.3 前置条件（P1S3B 结论必须带着它们一起引用）

这几条**不是**实验失败，而是"结论目前还不能升级为可独立复核"的原因。
任何引用 §3.2 的地方都必须同时满足或明确声明它们未满足。

| 编号 | 条件 | 现状（可核对的证据） | 修法 |
| --- | --- | --- | --- |
| **P1** | **版本溯源** | ⬜ **对既有产物仍未满足**：`results/raw/P1S3B-*/meta.json` 的 `reproducibility.git_commit` = `NOT_MEASURED`（36/36），`model_revision` = `'main'`（36/36）；`_missing_fields` 含 `git_commit`。P1S3 同样如此。**注意：字段存在且显式标注"未测"，不是字段缺失**——harness 已如实披露。<br>🟢 **对新产物已满足**（2026-09-23 起，见 §4.3.1）。 | 已落地，见 §4.3.1。既有 72 个 run **不回填**——"这批数字没有源码版本"本身是记录的一部分，改掉它就是伪造溯源。 |
| **P2** | **计时口径统一** | 🟢 **已满足（2026-09-23 起的新产物）**：`P2-*` 的 `telemetry.jsonl` `note` 已含 `readback_ms`；实测该区间中位 **0.240 ms**，且 **N=1/4/8 取同一值**（一次性 D2H 的成本由启动/同步开销主导，不随批量增长）。既有 `P1S3B-*` 的 note 止于 `pack_ms=`（无该字段）⇒ **它们是"TTFT/E2E/ITL 记在 logits 就绪时刻"那一版口径**。 | 见 §4.3.2（已做小规模前后对照）。既有产物**不回填**；两批数字**不得并表**。 |
| **P3** | **SLO 可辨识** | 🟢 **已修（2026-09-24）**：`run.slo.tpot_ms` 50 → **100**（见 §4.3.3）。依据：逐请求 TPOT 实测区间 **[61.7, 168.0] ms**，50 **低于下界 ⇒ 恒不可达** ⇒ `slo_satisfaction_rate` 等三列恒为 0.000，**不携带信息**（不是"达标率低"，是"判据无效"）。<br>⚠️ **既有 `P1S3-*`/`P1S3B-*`/`P2-*` 的产物里这三列仍是 0.000**，那是旧阈值下的**真实记录**，**不回填**——但引用时不能把 0.000 读成"两种配置都很差"。 | 已落地：新 profile `p1s3c`（`configs/mthreads/phase1_step3c/`）只差 `run.slo.tpot_ms` 一个字段；生成器新增 `_assert_slo_reachable` 守卫；标定工具 `scripts/calibrate_slo.py`。既有产物的达标率**按新阈值重算**（不重跑，见 §4.3.3）。 |
| **P4** | **控制门限与工作点分得开** | 🟢 **已修（2026-09-24）**：`controller.target_tpot_ms`/`hysteresis` 110/0.1 → **120/0.08**（见 §4.3.4）。依据：110×0.9 = **99.0 落在 c4 实测工作点 97.8–99.2 内部**，放大判据的 headroom 余量只剩 **0.0112**（门槛 0.1000）⇒ 该格行为跨重复不可复现（三次重复 TTFT = 481.4 / 481.6 / **652.1**，+43%）。<br>⚠️ **`P1S3B-*` 的 `adaptive_c4` 因此不可作为单次读数引用**；引用该格须带三次重复区间。 | 已落地：新 profile `p1s3d`（`configs/mthreads/phase1_step3d/`），与 `p1s3c` 只差门限两参数；生成器新增 `_assert_gate_separation` 守卫；标定工具 `scripts/calibrate_controller_gate.py`。**门限改动会改变调度行为 ⇒ 必须重跑**（`p1s3c` 的产物不能借用）。 |

| **P5** | **开放环 `arrival` 的参考点** | 🟢 **已修（2026-09-24）**，且**对既有登记数据零影响**：`real_backend` 原先把 `s.arrival_ms` 赋成 `step_start`（本步开始时刻），而 `simulate()` 用 `scheduled_arrival` ⇒ **同一份语义、两个执行器只修了一边**。后果：开放环下"请求早已到达、只是本步才轮到处理"的等待被整段抹掉，TTFT 系统性低估；**而 `accounting_error` 依旧是 0**（同一个被覆盖的参考点自洽）。<br>✅ **全量扫描 141 个 `meta.json`：123 个 `MEASURED` 的到达模型全是 `closed`**（闭环下 `arrival ≡ scheduled ≡ step_start`，本修复**不改变**闭环行为）⇒ **既有结论一条都不受影响，不需要重跑**。唯一的开放环产物是 3 个 `cmp-adaptive_v1*`，标签 `TEST_DATA`，且走的是本就正确的模拟器。 | 见 §4.3.5。**新口径自本条起生效**；⚠️ 既有的 `TEST_DATA` 开放环产物**不重跑也不回填**（它们不进结论），但**任何新的开放环 `MEASURED` 批次都必须是本口径**——旧口径的开放环 `MEASURED` 数据**不存在**，因此当下没有"并表"风险；一旦将来出现，两者**不得并表**。<br>**判据（从产物即可区分，不必看源码版本）**：开放环下一整批的 `admission_ms` **恒为 0 ⇒ 旧口径**；新口径下必有 `admission_ms > 0` 的请求（步粒度决定的受理延迟）。 |
| **P6** | **控制信号的口径（停顿是 decode 慢，还是被 prefill 挤的）** | 🟢 **已修（2026-09-24，**记录层**）**：控制器读的 `tpot_p90_ms` 来自相邻两次 token 之间的间隔，而某步做了 prefill 时该间隔**直接包含 prefill 耗时** ⇒ E9 已定因："prefill 挤占"被读成"decode 变慢"，而控制器的响应是调小 `chunk_size`（prefill 更碎 ⇒ 含 prefill 的步更多）——**反向放大**。现在把两者**分开记**（`interval_had_prefill` / `interval_decode_ms`，用单调时钟判、**覆盖区间内全部执行步**）。<br>⚠️ **本步只做记录、不改判据** ⇒ 既有结论不受影响；`controller.signal_mode` **默认 `legacy`**，两种模式在决策上等价。 | 见 §4.3.6。**新口径要用必须显式写 `controller.signal_mode: layered`**，并会随 `state.describe()` 落进决策轨迹（否则事后分不出哪批用了哪种信号）。<br>⚠️ "纯 decode 特征有效"**不等于**"应把所有决策交给它"——那是第 09 步要检验的策略问题；用户真实经历的停顿**必须继续进入** ITL / TPOT / stall / SLO。 |

#### 4.3.1 P1 的修法已落地（2026-09-23）：部署戳 + 投放现场校验

**为什么需要它**：远端源码树是**拷贝**过去的、不带 `.git` ⇒ `git rev-parse HEAD`
必然失败 ⇒ `git_commit` 恒为 `NOT_MEASURED`（上表 36/36 就是这么来的）。

**做法**（`src/ascend_opt/provenance.py` + `scripts/remote_run.py`）：

* 本机 `build_source_stamp()` 生成 `SOURCE_PROVENANCE.json`，字段含
  `git_commit` / `git_dirty`（限作用域）/ `tree_digest` / `digest_scope` /
  `deployed_at` / `deployed_by`。取不到 commit、树脏（除非显式 `--allow-dirty`）、
  算不出摘要 —— 三种情况**都抛错**，不产出一个"凑合的戳"。
* 读端 `read_source_stamp()` **重算摘要比对**：戳会过期，不符即作废并把
  `git_commit` 置空（错的 commit 比缺的 commit 危险）。"作用域变了"与
  "源码被改过"**分开报**，因为修法不同。
* `detect_source_commit()` 三级优先：真 git 仓库 > 有效戳 > 取不到；
  `git_source` 取值 `git` / `stamp` / `stamp_stale` / `none`，
  后两者（"有戳但不可信" vs "从来没有戳"）不合并。
* **投放不是写文件，而是一次判定**：`scripts/remote_run.py stamp <远端根>` 在投完
  之后**立刻用远端那棵树里的 `read_source_stamp` 重算摘要**，不符返回非零并
  逐文件列出差异。`push` 默认在推完源码后自动做这件事。

**真机证据（2026-09-23）**：

* 远端以 `/root/miniconda3/bin/python3 3.10.8` 校验通过：
  `投放并校验通过：… 读回 commit=10632452af1c（来源 stamp）`；
  两台机器的摘要相同 `1b7b929575d9021551ffa669dfcea86f5942737cd9e7322177ca01aa2ea72910`
  （本机 3.13 / 远端 3.10.8 —— 摘要算法不依赖解释器版本）。
* 远端上 `detect_source_commit('.')` = `('10632452af1c…', 'stamp', False)`，
  `build_reproducibility_record(...)` 得到 `git_commit='1063245…'`、
  `git_source='stamp'`、`git_dirty=False`。
* 摘要作用域 = `src/ascend_opt` / `benchmark` / `scripts`（**不含** `configs/`：
  配置内容已在每个 run 的 `meta.json.config` 里逐份落盘，收进作用域只会让摘要
  随每次生成配置而失效）。
* **比对是逐字节的**，故 CRLF 与 LF 的差异会让戳失效。这是有意的：本仓
  `.gitattributes` 要求仓内一律 LF，且 CRLF 曾让远端 shell 脚本报
  `$'\r': command not found`。

**操作约定（新的部署纪律）**：

1. **先 `git commit`，再 `push` + 投放戳**。工作区脏时投放会被拒绝——HEAD 描述的
   不是将被部署的代码。
2. 远端**被部署源码的作用域内必须与本机逐字节相同**。远端自造的探针脚本一律放
   `/root/ascend_probe/adhoc/`（不删除，只是移出作用域），其存档在
   `results/raw/_probe/<日期>/`。
3. 戳里的 `git_commit` 是"**作用域内容所属的提交**"，不必然是最新 HEAD——只动了
   作用域外的提交（例如只往 `results/` 加证据）不会让戳失效。此时重新投放一次
   即可让标签跟上，但**不重新投放也不算错**。

#### 4.3.2 P2 的前后对照已做（2026-09-24）：口径修复的总作用 ≤ 0.25%

**做了什么**：真机跑 P2 批次（4 格 × 3 轮 = 12 次，登记见 §1.3）。四格配置与 P1S3B
**逐字节相同**，所以两批之差＝两项代码改动的总作用。分析器：`scripts/inspect_timing_scope.py`。

**一、被错记的那个区间有多长（直接测量，不是推断）**

`telemetry.jsonl` 的 `readback_ms` = "logits 算完" → "token 可取回主机"，
**正是修复前被记成 0 的那一段**。

| 格 | `readback_ms` p50 | 占设备计算（fwd） | 随批内序列数（N）的变化 |
| --- | --- | --- | --- |
| `static_c1` | 0.240 ms | 0.40% | N=1 → 0.240 |
| `static_c4` | 0.240 ms | 0.28% | N=4 → 0.240 |
| `static_c8` | 0.240 ms | 0.21% | N=8 → 0.240（`mixed` 步 0.52，因一次 step 含 2 个子前向，按和计） |

⇒ **不随 N 增长**：0.24 ms 由 kernel 启动 + 同步开销主导，不是传输量。
修复前是**逐元素** `int(next_ids[i])`（N 次小同步，且计在 `pack_ms` 里），
修复后是 1 次批量 D2H —— 这就是下表中 TPOT/E2E 变快的来源。

**二、对**报告值**的总作用**（前 = P1S3B，后 = P2；两项改动不可分）

| 指标（p50） | `c1` | `c4` | `c8` | 是否超过本批批内漂移 |
| --- | --- | --- | --- | --- |
| TTFT | +0.103 ms（+0.023%） | +0.453（+0.027%） | +0.484（+0.021%） | **仅 `c8` 超过**（漂移 0.362 ms；c1/c4 的漂移是 0.246/0.539） |
| TPOT | +0.032（+0.051%） | −0.139（−0.156%） | −0.294（−0.207%） | `c4`/`c8` 超过 |
| E2E | +4.12（+0.050%） | −17.17（−0.133%） | −37.08（−0.182%） | `c4`/`c8` 超过 |
| `pack_ms`（纯 decode 步） | −0.170 | −0.290 | −0.500 | 逐元素同步被移走的量 |

**结论**：所有指标的作用 **≤ 0.25%**，方向自洽（TTFT 变晚 = 口径修正；TPOT/E2E 变快 =
少做 N−1 次同步）⇒ **没有任何既有结论会因为这次修复而翻转**，但两批**不得并表**。
**反面也说清**：`c1`/`c4` 的 TTFT 增量小于各自批内漂移 ⇒ 这两格"修复对 TTFT 的影响"
**测不出来**，只能说量级 ≤ 0.24 ms（即上表第一行的直接测量值）。

**三、意外发现：`adaptive_c4` 在 P1S3B 上的异常不是修复造成的——是控制器门限贴着工作点**

§1.2 的 `EXP-adap-114` 记 `TTFT P50 = 481.57`，那是 3 次重复的**中位数**，掩盖了
三次实际是 `481.4 / 481.6 / **652.1**`。逐请求看，差的是**第二批到达的 4 个请求**：
`454.7` vs `652.1`（第一批 4 个两 run 完全一致，连 `t_ms` 都一致）。

根因（从 `meta.json.controller_trace` 直接读出，不靠推断）：控制器"放大 chunk"的门限是

```
tpot_p90 ≤ target_tpot_ms × (1 − hysteresis) = 110 × 0.9 = 99.0 ms
```

而实测工作点 `tpot_p90` 是 **97.8 – 99.2 ms** —— **门限落在工作点分布内部**，于是整条爬坡
是一连串近似抛硬币（实测每次"放大"决策的 headroom 在 0.1004 – 0.1110 之间，门限 0.1000）：

| run | 爬坡（step→`chunk_size`） | 关键拒绝点 | 后 4 个请求 TTFT |
| --- | --- | --- | --- |
| `P1S3B` base | 32→1024, 56→1536, 80→2048 | step 104 拒（`tpot_p90`=99.38） | 454.7 ✓ |
| `P1S3B` r1 | 32→1024, 56→1536, 80→2048 | step 104 拒（99.01） | 454.7 ✓ |
| **`P1S3B` r2** | 32→1024, 56→1536, **80 被拒**, 160, **184→2048** | **step 80 拒（99.23，差 0.23 ms）** | **652.1 ✗** |
| `P2` ×3 | 均在 step 88–96 之前到 2048 | — | 454.8 ✓ |

而负载的 `prompt_len = 2048` **正好等于 `chunk_size_max`**：2048 能把整个 prompt **一步**
装下（TTFT ≈ 455 ms），1536 要拆成 1536+512 两步（≈ 652 ms）—— **+43%**。
`r2` 直到 step 185（t ≈ 22.3 s）才升到 2048，而第二批请求在 step 129（t ≈ 14.4 s）就到了。

两条并列结论：

1. **0.23 ms（0.2%）的观测差异 ⇒ 197 ms（43%）的时延后果。** 门限贴着工作点时，
   控制器行为**跨重复不可复现**；因此 §1.2 的 `EXP-adap-114` 聚合值（以及任何拿它做的
   "自适应 vs 静态"比较）**无效**。
2. 这与 P1S3 那版的缺陷**同源、方向相反**：那版 `target_ttft_ms = 2000` **高于**全部实测值
   ⇒ 判据一次都不触发（只收不放，退化棘轮）；本版的门限**落在**工作点内部 ⇒ 随机触发。
   共同修法：**阈值必须按控制器实际读到的信号**（`controller_trace[*].signals.*_p90_ms`）
   **标定，不能按报表里的逐请求分位**——两者不是同一个量。本例：判据用的是决策时刻
   "最近 32 个样本"的 `tpot_p90`（那时只有第一批在跑，≈98–99），而 `summary.tpot.p50`
   是全程 8 个请求的 `110.15`。**这正是 P3 的标定依据。**

#### 4.3.3 P3 的修法已落地（2026-09-24）：`slo.tpot_ms` 50 → 100

**判据口径先说清**（两处最容易混）：

* `run.slo.tpot_ms` 是**达标率判据**，`controller.target_tpot_ms` 是**控制目标**，
  两个字段、两种用途。P1S3B 只改了后者（50 → 110），前者一直是 50 —— 这就是缺陷。
* `SLO.satisfied_by()`（`src/ascend_opt/metrics.py`）**逐请求**判 `m.tpot_ms > 阈值`，
  不是比汇总分位；且 SLO 是**与**关系（同时配了 ttft 与 tpot 时两个都要满足）；
  分母是**全部提交**请求数（含失败/未完成）。
  ⇒ 标定必须看**逐请求**分布，工具是 `scripts/calibrate_slo.py`。

**逐请求 TPOT（`P1S3B`，96 个成功请求，2026-09-24 统计）**：

| 层 | 逐请求 TPOT 区间 (ms) | 说明 |
| --- | --- | --- |
| c1（三臂） | 61.7 – 62.0 | 最松 |
| c2（stat / adap / schu） | 77.9 – 86.1 | |
| c4 `static`（不分块） | 88.7 | |
| c4 `static_chunk` / `adaptive` | 98.7 – 112.0 | 分块的代价 |
| c8（三臂） | 133.5 – 168.0 | 最紧 |

合并区间 = **[61.7, 168.0]**；旧阈值 50 **低于下界** ⇒ 恒不可达。

**新阈值 = 100 ms 的达标率**（`ttft_ms` 仍为 2000；重算，非重跑）：

| tpot 阈值 | c1 | c2 | c4 `stat` | c4 `schu`/`adap` | c8 | 全格 |
| --- | --- | --- | --- | --- | --- | --- |
| 50（旧） | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **100（新）** | 1.000 | 1.000 | 1.000 | **0.125** | 0.000 | 0.604 |

四条必须一起读的说明：

1. **定性分档在 90 / 100 / 110 三个候选上完全相同**（c1/c2 通过、c4 按臂分裂、c8 全失败），
   三个值都在**[88.7, 110.4]** 这个"c4 分块代价"区间内 ⇒ 结论对该选择**稳健**；
   差异只在 c4 的粒度（90 全否 / 100 给 1/8 / 110 给 4/8）。选 100 的理由是它同时
   等于 **10 tok/s** 这一外部可辩护的交互式边界。
2. **这个阈值对 `chunk_size` 敏感是有意为之**：c4 的"不分块 88.7 / 分块 110.4"分列阈值两侧，
   于是达标率能反映本项目要测的那个权衡。**代价**：标定与本负载/本 chunk 取值绑定
   （prompt 2048、chunk 1024），换负载或换 chunk 档位必须重采，
   否则 `_SLO_TPOT_REACHABLE_MS` 会在错误的区间上放行。
3. **`仅TTFT超 = 0`**：在 `ttft_ms = 2000` 下没有任何请求**只**因 TTFT 不达标
   （16 个请求是"两者都超"）。即 TTFT 判据在本负载下**从不单独起作用**——达标与否实际由
   TPOT 决定。这不改变结论，但引用"TTFT 达标率"时必须知道它当前不是一个独立的信息维度。
4. ⚠️ **`--dry-run` 无法验证这一项**：一阶设备模型的 TPOT 只有 **7.7 – 10.25 ms**
   （比真机低约一个数量级），任何阈值都会读成 `1.000`。这条再次印证"静默失效不会被
   任何运行时检查发现"——它只能靠**真机口径的逐请求数据**标定。

**改动清单**（单变量：只有 `run.slo.tpot_ms`）：

* 新增 profile `p1s3c` → `configs/mthreads/phase1_step3c/`，与 `p1s3b` **只差**
  `run.slo.tpot_ms`（50 → 100）与标签 `run.notes`（③b → ③c）；
* `p1s3` / `p1s3b` **逐字不变**（已执行证据不容改）——已用重生成 + `diff -r` 逐字节验证；
* 平台模板 `configs/mthreads/s4000_fp16.yaml` 与 `RunOptions.slo` 兜底默认同步为 100；
* 生成器新增守卫 `_assert_slo_reachable`（阈值低于下界或高于上界都拦下），
  `p1s3`/`p1s3b` 的旧值改为**已声明的已知缺陷**，需 `--allow-known-defects` 才放行。
* 现有产物**不重跑**：`run.slo` 不参与任何调度决策，改它只影响派生指标 ⇒
  达标率按新阈值**重算**即可（上表）。


#### 4.3.4 P4 的修法已落地（2026-09-24）：控制门限 110/0.1 → 120/0.08

**缺陷形状与前三道门都不同**：门限贴着工作点时，控制器**不报错、不停止，而是随机地动**。

控制器判据（`AdaptiveController._decide`，`src/ascend_opt/controller.py`）：

* **收缩**：`tpot_p90 > target·(1+h)`
* **放大**：`ttft_p90 > target_ttft·(1+h)` **且** headroom `(target − tpot_p90)/target > h`
  （等价于 `tpot_p90 < target·(1−h)`）——**两个条件都要满足**
* 分支顺序：`memory` ⇒ `tpot` ⇒ `ttft` ⇒ `recover` ⇒ `none`

⇒ `110/0.1` 的放大门限 = `110×0.9` = **99.0**。

**控制器实际读到的信号**（`controller_trace[*].signals.tpot_p90_ms`，P1S3B+P2 的
adaptive 臂**首轮**，`scripts/calibrate_controller_gate.py`，2026-09-24）：

| 格 | 决策点数 | tpot_p90 min / **p50** / max (ms) |
| --- | --- | --- |
| `adaptive_c1` | 128 | 61.8 / **62.0** / 65.0 |
| `adaptive_c2` | 64 | 81.6 / **81.9** / 291.6 |
| `adaptive_c4` | 64 | 97.8 / **98.7** / **571.7** |
| `adaptive_c8` | 18 | 116.2 / **142.1** / 415.8 |

⚠️ 这个量**不是**报表里的 `summary.tpot.p50`（c4 那格是 110.15）——标定阈值只能用
**控制器实际读到的那个量**。两者混用会把门限标到错的位置。

**c4 的余量只有 0.0112**：中位数 98.7 距门限 99.0 仅 **0.3 ms**；放大判据的
headroom 上限 0.1112 对门槛 0.1000 ⇒ 余量 **0.0112**。实测后果：同一份
`phase1_step3b/adaptive_c4.yaml` 三次重复的 TTFT = 481.4 / 481.6 / **652.1**（**+43%**）。

**候选对比**（`calibrate_controller_gate.py --targets 110,120,130 --hysteresis 0.1,0.08`）：

| target/h | 放大门限 | 收缩门限 | 滞回带宽 `2h·target` | c4 headroom max | **c4 余量**（=max−h） | c4 放大触发 | c8 收缩触发 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 110/0.1（旧） | 99.0 | 121.0 | 22.0 | 0.1112 | **0.0112** | 53/64 | 17/18 | ❌ 门限压在工作点上 |
| 110/0.08 | 101.2 | 118.8 | 17.6 | 0.1112 | 0.0312 | 60/64 | 17/18 | ❌ 只抬了余量，中位数仍贴门限 |
| 120/0.1 | 108.0 | 132.0 | 24.0 | 0.1852 | 0.0852 | 60/64 | 17/18 | ⚠️ 可用 |
| **120/0.08（新）** | **110.4** | **129.6** | **19.2** | 0.1852 | **0.1052** | 60/64 | 17/18 | ✅ 余量 ≈10× 于旧值 |
| 130/0.1 | 117.0 | **143.0** | 26.0 | 0.2479 | 0.1479 | 60/64 | **5/18** | ❌ 收缩门限跨过 c8 中位数 142.1 |

⚠️ **"c4 headroom max" 与 "c4 余量" 是两个量，不要混读**：前者是实测
`max((target − tpot_p90)/target)`（**只随 `target` 变**，与 `h` 无关）；后者是判据真正
关心的"离门槛还有多远"（= 前者 − `h`）。本表第一版把这两者填在同一列里，
会让 120/0.1（0.1852）看起来比 120/0.08（0.1052）更安全——**恰好相反**。

**余量的排序**：130/0.1 (0.1479) > **120/0.08 (0.1052)** > 120/0.1 (0.0852) >
110/0.08 (0.0312) > 110/0.1 (0.0112)。**但余量不是唯一判据**：
`h` 同时决定带宽（`2h·target`），而带宽越窄、决策越容易来回翻 ⇒
选 120/0.08 是在"余量"与"带宽"之间的折中（24.0 → 19.2 ms 的代价换 0.0852 → 0.1052）。


选 **120/0.08** 的理由：c4 余量 0.1052（≈10× 于旧值），c8 中位数仍高于收缩门限 12.5 ms。
**130/0.1 被否**不是因为余量不够（0.1479 最高），而是它把 c8 从一个确定行为
（收缩 17/18）变成不确定行为（**5/18**）——拿一个格的不确定性去换另一个格，不是改进。

**⚠️ 开环复放的门槛判定变了（2026-09-24 实测，`scripts/check_step3b_reachability.py`）**：
同一份复放（喂 P1S3 实录信号）在两套参数上给出**不同的净方向**判定：

| 配置 | 放大可达 | 收缩可达 | 恢复路径 | **净方向** | 格内双向振荡 |
| --- | --- | --- | --- | --- | --- |
| `phase1_step3b`（110/0.1） | 是 | 是 | 是（n_recovered=2） | 放大 [1,2,4] / 收缩 [8] ⇒ **符合预期** | [2, 4] |
| `phase1_step3d`（120/0.08） | 是 | 是 | 是（n_recovered=2） | 放大 [1,2,4] / 收缩 **[]** ⇒ **不符合预期** | [2, 4, **8**] |

两者都**通过**（判据是双向可达 + 恢复可达，不是净方向），但细节变了：
新参数下 c8 的末值从 512 回到 **1024（= 初值）**——放大门限由 99.0 抬到 110.4 后，
c8 也够到了放大分支（1 次放大 + 1 次收缩 ⇒ 净 0）。同时 `h` 由 0.1 降到 0.08
使带宽**变窄**（24.0 → 19.2 ms），格内振荡的格数增加一个。
⇒ **这是"修 c4 的抛硬币"与"c8 的决断性/带宽"之间的实测折中**，不是纯改进。
**本次不因此回退**（判定仍通过，且回退要再走一轮完整验证），但必须在设计核心实验时
把它带上：负载**连续变化**时，带宽变窄会让穿越更频繁。
⚠️ 净方向**只是开环复放的注记**，不预测闭环终态（脚本自己的免责声明）——
闭环里 chunk=1024 会让 `tpot_p90` 高于复放值（P1S3B：c4 不分块 88.7 / 分块 98.7–112.0），
收缩会更容易触发。**要回答"是否真的决断"，只能靠真机重跑。**

**结构性发现（本节最重要的结论，属"形式性障碍"而非"参数没调好"）**：

工作点的**分布宽度**比档位间的差距还大——`adaptive_c4` 单格的 `tpot_p90` 就从
**97.8 到 571.7**（跨 **5.8×**），四档合起来 **61.8 → 571.7**（跨 **9.3×**）；
而滞回带宽度是**固定绝对值** `2h·target` ≈ **19.2 ms**。
⇒ **绝对阈值 + 单一 `(target, h)` 无法同时与多个负载档位的工作点分开。**

120/0.08 **并未解决**这个形式问题，它只是把最坏余量抬了一个量级。

⚠️ **（2026-09-24 晚 更正）本节原先在这里写的是"c4 的区间 [97.8, 571.7] **仍跨过**
新门限 110.4"——那个读数取自标定工具一个**有缺陷的输出列**，它不能支撑"跨过门限"。**
该列把**收缩点**的 headroom（很大的负数）也算进区间统计，于是 ``min < h < max``
恒成立、"跨过门槛"**恒报**，并且与它自己的「带内」列**自相矛盾**——同一行里
"带内 0/64" 恰恰说明合格点**全部**过了门槛、判据是决断的。已修（工具 + 反向控制测试）；
修后同一个工具对 120/0.08 报 **无格落在门限内部（3 格决断 + 1 格不可达）、样本跨越门限 0 格**
（`不可达` 是**负载形状**的性质：c1 的 `ttft_p90` 全域未超放大门限 ⇒ 放大分支**从未被评估**，
**不是**"被门限拦住"），而它对 110/0.1 报 c4「样本跨越门限」
且"离放大门限最近的点相距 **0.0 ms**"（与实测的 481 vs 652 一致）。

⇒ **形式障碍依然存在，但理由要换成"负载扫描"这一条**（与"某一格跨过门限"无关）：
核心实验里负载是**连续变化**的，工作点会**扫过**整条 61.8 → 571.7 ms 区间，
而滞回带宽是**固定绝对值** ≈ 19 ms ⇒ 必然存在某些负载档位使门限落在带内。
这**不是**某一个固定档位的性质，所以消不掉，需要**判据形式**的改动
（例如门限按工作点归一化、或让 `(target, h)` 随负载档位调度）。
**登记为待办，在核心实验设计时处理**；不要把 `p1s3d` 读成"门限问题已解决"。

**改动清单**（单变量：门限两参数 + 标签）：

* 新增 profile `p1s3d` → `configs/mthreads/phase1_step3d/`，与 `p1s3c` **只差**
  `controller.target_tpot_ms`（110 → 120）、`controller.hysteresis`（0.1 → 0.08）
  与标签 `run.notes`（③c → ③d）——已用 `diff -r` 逐行核对；
* `p1s3` / `p1s3b` / `p1s3c` **逐字不变**——重生成 + `diff -r --brief` 零差异验证；
* 生成器新增守卫 `_assert_gate_separation`；`p1s3b`/`p1s3c` 的门限改为
  **已声明的已知缺陷**（需 `--allow-known-defects` 才放行）；
* `--profile` 默认值 3c → **3d**：默认必须是"干净的那一版"，否则无参数的
  `--check` 会被自己的守卫拒掉；
* 标定工具 `scripts/calibrate_controller_gate.py`（**只读，不写配置**）。
* ⚠️ **门限改动会改变调度行为 ⇒ `p1s3d` 必须重跑**；`p1s3c` 的产物**不能借用**。
  这与 P3 相反（`run.slo` 不参与决策，所以 P3 只需重算）。

**⚠️ 判据口径不一致（P4 自查发现 → 2026-09-24 晚查清，结论改写过一次）**：

生成器守卫 `_assert_gate_separation` 判的是"**中位数**到门限的距离 ≥ 5%·target"，
标定工具判的是"**实测的每一个点**上判据决断吗、离翻转多远"。P4 当时把后者
写成"`max(headroom) − h` 的余量"，并把它称为"权威口径"——**那个写法本身是错的**：
工具并没有真的在算一个判定，它打印的 `headroom min/max` 被**收缩点**污染
（见上文更正），所以"跨过门槛"恒报。已修工具：现在它给每格一个四态 `verdict`
（**必放大 / 必不放大 / 样本跨越门限 / 不可达**）＋"离门限最近的**点**相距多少 ms"，
并且把「ttft 条件不满足（放大分支根本没被评估）」与「被 headroom 门槛拒」
**拆成两列**（旧版合成 `n_band` 一个数 ⇒ 两者在表里同形，而结论相反）。
⚠️ **2026-09-24 晚再改两处（口径性质的）**：① 判定名由「抛硬币」改为
**「样本跨越门限」**——旧名读起来等于"跨重复不可复现"，而它只说明**这一条合并序列上**
门限不决断（同一次运行内负载本来就可能在工作点之间移动）；② 工具改为读**全部重复**，
并**另设**「跨重复对照」一节专门比"同一条件下不同重复的决策分歧"（`重复间一致 /
重复间分歧 / 单一重复`）——**这才是**可复现性的问法。**下表里的「抛硬币」一律读作
「样本跨越门限」。**

修好后查清了**为什么两处口径不可能统一**——它们互相都不包含对方，实测分歧方向相反：

| `(target, h)` | 守卫（中位数距离 ≥ 5%·target） | 标定工具（修正后） |
| --- | --- | --- |
| 110 / 0.1 | **拦下**（c4 距放大门限 0.3 ms） | 样本跨越门限 2 格：c2 最近点 1.0 ms、**c4 0.0 ms** |
| 110 / 0.08 | **拦下**（c4 距 2.5 ms） | 全格决断，但 c4 最近点仅 **0.6 ms**（决断但极脆） |
| 120 / 0.1 | **放行** | 全格决断，但 **c8 最近点仅 1.1 ms** ← 工具看到、守卫看不到 |
| **120 / 0.08（现行）** | **放行** | **3 格决断 + 1 格不可达**（c1 的 `ttft_p90` 全域未超放大门限 ⇒ 放大分支**从未被评估**）、**样本跨越门限 0 格**，最近点 **3.4–29.0 ms** |

> ⚠️ **口径注**：上表除最后一行外，都由**只读首轮**的旧版工具算出（2026-09-24 晚改为读
> **全部重复**）。样本变密后"最近点"可能**更小**——最后一行即从 3.5 变成 **3.4 ms**。
> 重跑其余三档会得到略有差异的值，**不要**把差异读成"门限变了"。
> 另：「全格决断」是旧措辞，精确说法是"**无格落在门限内部**"（四态里的 `不可达`
> **不是**决断，也**不是**"被门限拦住"）。

⇒ **不能靠"换个统计量"把两者合并**：若把守卫换成"最近点 + 同一个 5%·target margin"，
**120/0.08 会被自己拦下**（c8 收缩侧最近点 3.5 ms < 6.0 ms）——即**回溯否掉已入库的
`P1S3D`**，而它的合格点是全格决断的。所以正确的分工不是"统一"，是**各自明确适用范围**：

1. **生成器守卫 = 生成期的静态下限（floor）**：不解析产物、几秒可跑，负责挡住
   "110/0.1 这类明显贴着的值复发"。**不得**用它宣称"门限已充分分离"。
2. **标定工具 = 跑之前用产物做的读数（⚠️ 不是阻断门）**：给出每格的四态 `verdict`、
   "最近点距离"，以及**跨重复对照**（"重复间一致 / 分歧 / 单一重复"）。
   三者一起读：`verdict` 回答"这条合并序列上门限决断吗"，最近点回答"离翻转多远"
   （110/0.08 是"决断但只有 0.6 ms"，等于没分开），「重复间」回答"换个重复还这样吗"。
   ⚠️ **它不能充当"可复现性"的阻断门**（用户 2026-09-24 裁定）：它是**静态口径**
   （不含冷却，也不含"动作改变后续信号"这一反馈）⇒ 区分不了"**样本内**不决断"与
   "**跨重复**不一致"——而这两件事的结论相反。

**登记待办**：若要把第 2 条接成一道**真正会失败的**前置门，判据必须是
**「重复间一致」**（`cross_repeat_verdict`），**不是** `verdict != 样本跨越门限`
——后者会把"同一次运行内负载在工作点之间移动"误判成"不可复现"。在此之前，
核心实验设计时仍须处理"判据形式"（见上文"结构性发现"）。
⚠️ 引用本节时不要再说"权威判据是 `max(headroom) − h`"。

**⚠️ `--dry-run` 同样验不了这一项**：干跑没有真机信号，控制器读到的是模拟量
（与 §4.3.3 第 4 条同一个原因），门限标定只能靠真机 `controller_trace`。

**真机验证结果（2026-09-24，P1S3D，36/36）：登记行 §1.4，结论 §3.3 E1–E9。要点三条——**

1. **静态判据层面的目的达到了**：旧门限下那个具体的拒绝（`tpot_p90` 工作点区间
   97.8–99.2 跨过 99.0）在新门限 110.4 下**在构造上不可能**再发生（E1），余量 0.0112 → 0.1052。
2. **作用面很窄**：除 c4 外，c1/c2/c8 的决策计数、prefill 生效分布逐格与 P1S3B 相同（E2）。
3. **⚠️ 但不能说"c4 的跨重复可复现性问题已解决"**：P2 用**旧门限**同样 3/3 收敛
   ⇒ n=3 分不开"新门限消除了它"与"没抽中"（**E8**）。同时 §1.3 注 2 的"与计时修复无关"
   一并更正为"归因未建立"。**D2 的限制只是被削弱，未解除。**
4. **形式性障碍仍在**（本节的"结构性发现"未被 P4 触及）——但正确的理由是**负载扫描**：
   核心实验的负载**连续变化**会扫过整条 61.8 → 571.7 ms 区间，而滞回带宽是**固定** ≈19 ms
   ⇒ 必然存在某些负载档位使门限落在带内。另见闭环现象：**与"请求补入批"同步的周期性调整**
   （周期 = 补入批间隔 ≈ 一个请求的 e2e ≈ **129–132 个 decode 步**，由负载决定、**不是控制器的
   固有周期**；触发机制见 E9：观测窗混入批边界的 prefill 步）、`recover` 分支全程未触发（E7）。
   ⚠️ 原写作"c4 区间仍跨过新门限"，那个读数来自标定工具一个**被收缩点污染的列**
   （已更正，见 §4.3.4）。⇒ 核心实验设计时仍须处理"判据形式"。

#### 4.3.5 P5 的修法已落地（2026-09-24）：开放环 `arrival` 改用**日程时刻**

**缺陷形状与 P1–P4 都不同：它不在配置里，而在"两个执行器只修了一边"。**

`benchmark/real_backend.py` 与 `benchmark/run_benchmark.py::simulate()` 是**同一份调度语义的两份实现**
（一份接设备、一份走一阶模型）。模拟器早已改成：

```python
s.arrival_ms = scheduled_arrival      # 放行时刻（开放环 = 日程；闭环 = 本步开始）
s.enqueue_ms = step_start             # 受理时刻
```

而真实后端写的是 `s.arrival_ms = step_start`。**模拟器里还留着解释这段的注释**
（"为什么到达时刻不能写成 now"），真实后端没有 ⇒ 分叉发生且无人察觉。

**后果**：开放环下，若某请求的日程时刻落在**一个长步的内部**（关闭分块时一次算完整个长
prompt，一步可达数百毫秒），它的 TTFT 会从"首 token − **日程**"变成"首 token − **本步开始**"，
把中间那段真实等待**整段抹掉**。

**最隐蔽的一点**：`accounting_error_ms` 依然是 **0**。因为 `done_ms`、`ttft_ms`、`e2e_ms`
全部以**同一个被覆盖的参考点**计量 ⇒ 账目自洽。**所以不能用"账目自洽"证明口径正确。**
`tests/test_real_backend.py` 新增的用例专门把这一点算出来并断言：
现行读法 `accounting_error == 0`，**旧读法也算出 `== 0`**（两者只差一个整体平移）。

**为什么两个执行器会分叉而测试不报**：两边的"时刻单调性"断言不是同一组。
`tests/test_simulator.py` 有 `scheduled_arrival_ms <= arrival_ms`，
`tests/test_real_backend.py` **没有这一条**（只有 `arrival <= enqueue <= ...`）。
⇒ 已把缺的那条补上，并注明"**两边必须断言同一组不变式**"。

**修法与判据**（单变量：只有这一行，加注释）：

```diff
-                s.arrival_ms = step_start
+                s.arrival_ms = scheduled      # 放行时刻；与 simulate() 同一份语义
```

**离线假时钟反例**（`tests/test_real_backend.py::TestOpenLoopArrivalReference`，7 项）：
假驱动每步耗时 = `fixed + per_seq×批大小 + per_token×token 数`，取 `fixed=0 / per_seq=0 /
per_token=1 ms` 精确控制步长；日程 `r0@0 ms`（占满第 0 步，100 token ⇒ 100 ms）、
`r1@10 ms`（落在第 0 步内部）：

| 量 | 正确读法 | 旧读法 |
| --- | --- | --- |
| `scheduled_arrival_ms` | 10 | 10 |
| `arrival_ms` | **10** | 100 |
| `enqueue_ms` | 100 | 100 |
| `first_token_ms` | 101 | 101 |
| **`ttft_ms`** | **91** | **1** |
| `admission_ms` | 90 | 0 |
| `queue_ms` / `prefill_exec_ms` | 0 / 1 | 0 / 1 |
| `accounting_error_ms` | **0** | **0** ← 判据失效 |
| `SLO(ttft_ms=50).satisfied_by` | **False** | **True** ← 结论翻面 |

**反向对照（证明用例会失败）**：把那一行临时改回 `step_start` ⇒
**7 项里 5 项失败**，报错正是 `1.0 != 91.0`；还原后 44 项全过（原 37 → +7）。
（还原用备份文件比对 md5 与 `tr -cd '\r'`，避免"临时改坏"把 LF 写成 CRLF。）

**影响面（已用产物核实，不是推断）**：全量扫描 `results/raw/**/meta.json` 共 141 个 ⇒
`closed` 123（`MEASURED`）+ `closed` 15（`TEST_DATA`）+ `burst` 3（`TEST_DATA`）。
⇒ **既有 `MEASURED` 数据零影响、无需重跑**（闭环下 `arrival ≡ scheduled ≡ step_start`）。
本修复是**即将进行的开放环（`load_ramp`/`piecewise`）实验的前置条件**——
它与第 08 步的 pilot 关系，正如 §4.3.4 与第 12 步的关系。

#### 4.3.6 P6 的修法已落地（2026-09-24）：把"两种停顿"分开记录

**缺陷形状**：控制器读的 `tpot_p90_ms` 来自 `recent_tpot`，而 `recent_tpot` 存的是
**相邻两次 token 发出之间的间隔**。某一步做了 prefill 时，落在这两步之间的间隔
**直接包含该 prefill 的耗时**（`EXPERIMENTS.md` §3.3 **E9** 已用 `telemetry.jsonl`
逐步定因：c2 纯 decode 步 `fwd_ms=77.2`、窗内无 prefill 时 `tpot_p90 = 81.5–82.7`；
批边界的 prefill 步 `fwd_ms=512.5` 落进窗口时 `tpot_p90 = 137.7–139.0`；
**收缩门限 129.6** ⇒ 纯 decode 在门限下、含 prefill 在门限上）。

⇒ 控制器把"prefill 挤占"读成"decode 变慢"，而它的响应是**调小 `chunk_size`**
（prefill 更碎 ⇒ 含 prefill 的步更多）——**反向放大**。

**修法（只做记录，不改判据）**：用**两个单调时钟**把每步耗时归到 prefill 或"纯 decode"，
两次 emit 之间的时钟差就是该间隔的两个分解量。

```python
# 每步结束（⚠️ 在结算**之前**读相位：结算会把算完 prefill 的请求改成 decode/done）
_step_has_prefill = any(states[it.req.req_id].phase == "prefill" for it in batch)
if _step_has_prefill: prefill_clock_ms += _step_ms
else:                 decode_clock_ms  += _step_ms
```

**为什么用时钟差而不是"看最后一步"**：一个间隔可能跨很多步；
"只看发 token 的最后一步"会漏掉**夹在中间的** prefill。
时钟差天然覆盖区间内**全部执行步**（这是本步的硬要求）。

| 新增量 | 含义 |
| --- | --- |
| `interval_had_prefill` | 这个间隔里**任意一步**做过 prefill 吗（与 `intervals` 逐项对齐） |
| `interval_decode_ms` | 这个间隔里**纯 decode 步**一共花了多少（`interval` 大而它不变 ⇒ 是被挤的） |
| `step_trace` | 每步 `(时刻, 是否含 prefill, 纯 decode 耗时)` —— **可复核** |

**为什么 `step_trace` 要落盘**：有了它，"标记覆盖了全部步"这件事可以**从产物独立重算**，
不必信声明。测试里正是这么做的（按请求把间隔映射回步，逐项比对）。

**接线**：
* `controller.signal_mode`：**默认 `legacy`**（只看 `tpot_p90`，行为与判定**逐格不变**）；
  `layered` 才收集/记录，并随 `state.describe()` 落进决策轨迹。
  ⚠️ **两种模式在决策上等价**——本步刻意不改判据。
* `controller.signal_window_expiry_ms`：**默认 `None`** = 不过期（保既有行为）。
  设了值之后，距上次**收到新样本**超过上限 ⇒ 本周期返回"不动作"且**不消耗观察周期**，
  理由写进轨迹。此前没有任何过期规则 ⇒ 负载停顿时残留的**旧**样本仍会驱动收缩/放大。

**验证**（新增 13 项，全部离线）：
* 后端 7 项（`TestLayeredSignals`）：标记与 `step_trace` 独立重算逐项一致；
  `decode_ms` 在含 prefill 的间隔上明显小于 `interval`；三组样本等长；
  **原始 `intervals` 与用户指标一字未改**（不靠丢样本改善数字）；两次运行调度与 token 逐位相同。
* 控制器 6 项（`TestLayeredSignalWiring`）：取值校验（拼错报错）；`legacy` 下分层字段是
  `None` 而**不是 0**；`layered` 下被算出并进轨迹；空窗口 ⇒ `None`（不是 `False`）；
  **过期 ⇒ 不动作且不消耗周期**；不配上限时同一调用照常动作（反向对照）。
* 全量离线回归 **820 用例 / 真正执行 820 / 跳过 0 / 失败 0 / 错误 0**（807 → +13）。

⚠️ **一项能力边界（如实登记）**：验收要求的"含 prefill 标记覆盖全部步"，
在**真实夹具**里只能用"与 `step_trace` 独立重算一致"来证——因为本项目的分配器
**优先满足 decode**，"某个请求被整步饿住"在预算争抢下**构造不出来**，
所以真实夹具里每个间隔恰好只跨一步，两种读法给不出不同答案。
⇒ 另加一条**规格测试**（合成步序列：prefill 在中间、末步是纯 decode），
断言"时钟差读法 ≠ 只看最后一步"，并在 docstring 里写明它证的是**规则本身**。
**两者合起来**才是完整证据：一个是语义规格，一个是实现与规格一致。

### 4.4 执行优化批次（EXECOPT3/EXECOPT4）必须一起引用的口径限制

这几条**不是**批次失败，而是"这两个批次的数字能回答什么、不能回答什么"。
任何引用 §1.5 / §1.6 / §1.7 / §3.6 E16–E20 / §3.8 的地方都必须带着它们。
（2026-10-03 起 A1–A4、A7 对 **EXECOPT5** 同样适用。）

| 编号 | 限制 | 证据 | 影响与修法 |
| --- | --- | --- | --- |
| **A1** | ★ **`driver_stats` 的 `*_total` 与 `n_forwards`/`tokens_out` 含 warmup，而各臂 warmup 轮数可能不同** | `_drive_warmup_batch` 直接调 `self.forward(...)` ⇒ 预热与正式段共用累加器。EXECOPT4 实测 `warmup.iterations` = **2（基线、`s01_both`）vs 1（两 pkv 臂）**；`n_forwards` 1377 vs 1260，**差 117 = 恰好一轮预热的 1 prefill + 116 decode**。 | ⚠️ 直接比 `pack_ms_total` 会读出 **"+31%"** 的**假效应**（真值 +0.2%~0.5%）。<br>**修法已落地**：新增 `driver_stats.measured_window`（`n_forwards_measured` / `pack_ms_measured` / `fwd_ms_measured` / `tokens_out_measured` / `basis`），既有键语义**不变**。<br>⚠️ EXECOPT4 产物**早于该字段** ⇒ 该批只能用 **telemetry 逐 step 求和**替代（`scripts/analyze_exec_opt_batch.py` 已实现两路并列 + 交叉校验）。 |
| **A2** | **`pack_ms_total` 是"补位与切片的显存搬运"总量（不含设备计算），但只对**跨臂**比较有意义** | `benchmark/real_backend.py` 文件头定义；四臂 `pack_tel` 极差 0.08%~0.43%（`s04_pkv` 0.17%）。 | 单看一个臂的 `pack_ms` 无意义（绝对值随步数/批量变）。判据必须是**同批同协议下的跨臂差**，且要先扣 warmup（A1）。 |
| **A3** | ★ **存在的是「孤立单步长停顿」（13.9–19.3 s 一步），不是持续慢态**；受影响轮次的 wall/吞吐差异是它的算术后果，且**机制归因暂不可判** | EXECOPT4 逐 step 实测长停顿：`s4000_fp16` r0 `fwd_ms` **16,839 ms**@step779；`s01_both` r2 **13,865 ms**@993；`s01_both_s04_pkv` r0 `pack_ms` **16,961 ms**@360、r2 `fwd_ms` **19,288 ms**@456。常规轮 wall ≈123.4 s，受影响轮 137.1/140.2/140.9/143.0 s —— 差额与其单步停顿**逐轮对上**（123.4+13.9=137.3、123.6+16.8=140.4、123.5+17.0=140.5、123.2+19.3=142.5）。⚠️ 常规轮的 `fwd_ms` 单步上限只有 ~1.34 s（step 0 的批量预填充） | ⚠️ **服务指标（ttft/tpot/吞吐）暂不可归因**：受影响的 4 轮**跨 3 个不同臂** ⇒ 与机制无系统关联；但**也不能**说"与机制无关"——分配、同步、宿主机暂停、实现交互**都未被排除**。<br>**修法（待做）**：批次内**探测并标记长停顿步**（本批判据：单步 `fwd_ms`/`pack_ms` > 1000 ms 的步数；常规轮 = 1、受影响轮 = 2），保留**全部**有效运行、**不得**事后删"慢轮"。<br>⚠️ **2026-10-01 更正**：本条原写作"环境双模态、吞吐下降纯抽样假象"——那是**错的**（那不是持续慢态，也不足以断定与机制无关）。 |
| **A4** | ★★ **判 S04 必须按「有效样本」（机制真正被行使的那部分）；且**数量覆盖率 ≠ 耗时权重**，两者不可互相代用** | **数量覆盖率** = 快路径组数 / 正式窗口前向组数 = **77 / 1143 = 6.74%**（⚠️ **不是** 193/1260 = 15.3%：那个数既**没扣预热**、分母**也没扣**）。这 77 步是**末尾连续 `decode==1` 的单请求尾段**。<br>**时间权重**（= 尾段 `pack`Σ / 全窗口 `pack`Σ）：EXECOPT5 实测 baseline **0.5189%**、pkv **0.8831%**（EXECOPT4/EXECOPT5 的尾段 `pack`Σ 均 ~109 / ~186 ms，全窗 ~21,000 ms） | ⚠️ 全窗口的 **+0.19%（E20）/ +0.366%（EXECOPT5）** 是**算术后果**，**不能**据此判"无效/不可分辨"（初版登记就是这么误判的）。<br>★ **必须用时间权重解释，不能用数量覆盖率**：尾段的**数量**份额是 6.74%，但**耗时**份额只有 ~0.52%（baseline）——因为尾段每步只搬 1 个 token 的 KV，**单位步搬运量远低于前段的批量步**。两个份额混用会得出"稀释了 13 倍"这类**算不出来**的说法。<br>★ **加性核对**（EXECOPT5）：`20,968.8 + 77.15 = 21,046.0` vs `21,055.9` ⇒ 残差 **9.9 ms（0.047%）** ⇒ 全窗 87.1 ms 的差里 **88.6% 来自尾段**。<br>⇒ 正确表述：**"成本真实存在，但只占总耗时的约 0.9%"**，不是"被稀释成没有"。<br>⚠️ 报比值必须写明是**逐轮配对比值的中位数**（`median(pkv_i/baseline_i)`）还是"中位数之比"：EXECOPT5 两者为 +70.86% / +70.91%（本批接近，但**口径不同不可混用**）。<br>⚠️ 分母必须写清是**前向组**（1,143）还是**逻辑步**（1,080）。 |

| **A5** | ★★ **EXECOPT4 的 S04 两臂带**已复现的数值缺陷**：`fast → fallback → fast` 重入时读到过期的常驻缓冲** | 回退路径只更新 `_Slot.past`、**不碰 `_kv_buf`**；而重入时 `_pkv_ensure_buffers` 原实现是 `if buf is not None: continue`（已存在就跳过初始化）⇒ 缓冲里是旧内容。CPU 微型 Qwen 复现：同一请求重入后输出 token **60 → 36**（下一步 **62 → 4**）；逐层 KV 与基线最大差 **2.102e-01**。**形状全程合法、不报错**。已在 2026-10-01 修复（按"已同步长度"判脏并重同步），并补"加入→回退→退出→重入"的 logits/KV/token 对照用例（破坏验证：退回原实现 ⇒ KV 差 0.2102 被抓住） | ⚠️ **不能**据此断言"EXECOPT4 的所有请求都算错了"：本批快路径只覆盖 6.74%（且集中在**单请求尾段**），期间是否有"回退后重入"取决于该 run 的实际批组成，**未逐轮核**。⇒ 这批 S04 的 `pack` 结论（+72%）**仍成立**（那是搬运量，与数值无关），但**其正确性未获证明**；修复后**必须**重跑才能引用 S04 的数值正确性。 |
| **A6** | ★ **EXECOPT4 全部 12 份产物的源码身份是 `NOT_MEASURED`** | `reproducibility.git_commit = "NOT_MEASURED"`、`git_source = "stamp_stale"`（12/12）；`data_label` 仍是 `MEASURED`。成因：`exec_opt_batch.py::step_push` 把 `remote_run.py stamp` 的**退出码丢掉了** | ⚠️ **`MEASURED` 是测量标签，不能替代源码身份**；"批次结束后本地与远端文件一致"也**不能**证明"运行期一致"。<br>⇒ 引用这批数字时必须注明"无源码版本绑定"。<br>**修法已落地（2026-10-01）**：stamp 非零退出 **或** 缺少成功标记 ⇒ 直接 `StepFailed`、**禁止 launch**；断点续跑另加身份核验（源码来源可信 + `config.snapshot.yaml` 逐字节一致 + `model_revision` 一致），不符即拒绝复用并报错。历史产物**保留原样、不补标签**。 |
| **A7** | ★★ **EXECOPT5 的产物里嵌的是「另一个容器」的环境指纹（陈旧 5 天）** | 10 份产物的 `hardware_env` 写着 `collected_at = 2026-09-28T00:16:44+0800`、`hostname = autodl-container-cf99469eb6-b877706e`；而**实时** `reproducibility.hostname_or_machine_id = autodl-container-05264bbf0d-3b9c3488/root`。成因：`results/raw/_env/env.json` 是**共享路径**，`load_hardware_env` 只按**路径**寻找、**没有新鲜度或容器身份校验** ⇒ 5 天前、另一个容器采的指纹被原样写进全部产物，且产物**看上去完全正常** | ⚠️ 指纹承载 `mthreads_driver_version` / OS / CPU / `device_model` —— 这些正是**跨批次可比性的前提**。指纹来自 B 容器、数字来自 A 容器时，**"两批同环境"这句话没有证据**。<br>**同期已有证据（可引用）**：批次预检 `2026-10-03 03:46:58` 采到本容器的**实时**值 —— `PY 3.10.8` / `TORCH 2.2.0 MUSA 1.3.0+81caf0a` / `TF 4.46.3` / `TORCH_MUSA_AVAIL True` / `DEVS 1` / `REV_ENV a09a35458c70…` / `HF_HOME /root/autodl-tmp/hf`；另加计时探针（7×1.0 s，极差 0.000121 s，稳健 CV **0.005%**）与 `cpu.max = 1500000 100000`（容器可用 15 核）。<br>**仍缺失（明确登记为「测不到」，不回填）**：该容器的 **MUSA driver 版本**、OS/kernel、CPU 型号、以及新容器的 `device_model` 字符串（`device_model='MTT S4000'` 来自陈旧指纹）。<br>**处置**：原始产物**保持原样、不修改、不补标签**（本批 `hostname_or_machine_id` 与 `hardware_env.hostname` 的矛盾**保留为证据**）。<br>**修法已落地（2026-10-03）**：`exec_opt_batch` 新增前置步骤**重采环境指纹 + 校验容器/设备身份**（`fingerprint_identity_verdict`）—— 指纹描述的主机必须==实跑主机，否则**当场 `StepFailed` 禁止起跑**；换容器**允许**但记 `container_changed` 并显著告警；同时把 `plan` 与 `run` 的步骤序列合并为一张表（原先两处各写一遍 ⇒ 新步骤一度只出现在 `run` 里）。<br>⚠️ **正式论文引用 EXECOPT5 前，必须先解决本条的环境证据问题**。 |
> **判据可复现**：`scripts/analyze_exec_opt_batch.py --tag EXECOPT4`
> （两路口径并列、**覆盖率按正式窗口前向组**、**有效样本=单请求尾段**、
> 回退分桶闭合检查、单变量差集自算、孤立长停顿探测、`measured_window`×telemetry 交叉校验）。
> 产物字段路径 `context.extra.backend_stats.driver_stats`——**顶层没有**
> `execution_optimizations`，按记忆写路径会恒取到 `None` 并静默出空表。



## S13/S14 批次登记（2026-09-26 由 scripts/emit_experiment_rows.py 从产物生成）

> ⚠️ 为什么补这一段：复核发现登记表里**一条 `S14*` 都没有**，而纪律要求"每个结论都要指向一条登记行"。
> ⚠️ 各批次的**判据/模型/适用范围不同**（`S13COSTIN` 是误标批次；`S14B1/S14B2` 用的是默认 `tpot_p90` + 旧模型），
> **不得混算** —— 变量列已写明各批的 `controller.kind` / `criterion`。

<!-- S13COSTIN：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S13COSTIN-001 | `S13COSTIN-short_chat-costin_slo_weave` | short_chat | `costin_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 9003.08/11140.97/12913.47 | 242.38/242.70 | 428.01 | 30.76 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0012 | registered |
| EXP-S13COSTIN-002 | `S13COSTIN-short_chat-costin_slo_weave-r1` | short_chat | `costin_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 9005.19/11142.05/12917.62 | 242.46/242.76 | 428.04 | 30.75 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0006 | registered |
| EXP-S13COSTIN-003 | `S13COSTIN-short_chat-costin_slo_weave-r2` | short_chat | `costin_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 9000.85/11157.84/12941.76 | 242.35/242.59 | 428.23 | 30.75 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0005 | registered |
| EXP-S13COSTIN-004 | `S13COSTIN-short_chat-costin_threshold` | short_chat | `costin_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 8999.94/11139.47/12995.94 | 242.30/242.59 | 427.80 | 30.73 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0014 | registered |
| EXP-S13COSTIN-005 | `S13COSTIN-short_chat-costin_threshold-r1` | short_chat | `costin_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 9004.46/11146.72/12922.00 | 242.46/242.56 | 428.92 | 30.75 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0004 | registered |
| EXP-S13COSTIN-006 | `S13COSTIN-short_chat-costin_threshold-r2` | short_chat | `costin_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 9002.23/11140.40/12915.32 | 242.35/242.81 | 427.84 | 30.76 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0013 | registered |

<!-- S13COSTINB：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S13COSTINB-001 | `S13COSTINB-short_chat-costin_slo_weave` | short_chat | `costin_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 18565.94/25738.23/26700.69 | 147.32/214.27 | 668.18 | 20.93 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2694 | registered |
| EXP-S13COSTINB-002 | `S13COSTINB-short_chat-costin_slo_weave-r1` | short_chat | `costin_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 18570.76/25723.47/26698.26 | 147.30/214.15 | 668.26 | 20.93 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2692 | registered |
| EXP-S13COSTINB-003 | `S13COSTINB-short_chat-costin_slo_weave-r2` | short_chat | `costin_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 17338.94/25937.14/26919.22 | 146.76/208.80 | 671.33 | 21.57 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2238 | registered |
| EXP-S13COSTINB-004 | `S13COSTINB-short_chat-costin_threshold` | short_chat | `costin_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 8991.76/11130.08/12994.33 | 242.10/242.18 | 427.50 | 30.76 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0004 | registered |
| EXP-S13COSTINB-005 | `S13COSTINB-short_chat-costin_threshold-r1` | short_chat | `costin_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 8995.96/11150.39/12904.24 | 242.23/242.86 | 427.69 | 30.78 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0006 | registered |
| EXP-S13COSTINB-006 | `S13COSTINB-short_chat-costin_threshold-r2` | short_chat | `costin_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 8999.54/11146.40/12909.62 | 242.22/242.96 | 427.58 | 30.76 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0017 | registered |

<!-- S14B1：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S14B1-001 | `S14B1-short_chat-fixed_cost` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=None | - | MEASURED | 9063.07/11214.90/12995.79 | 244.00/244.22 | 430.16 | 30.55 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0009 | registered |
| EXP-S14B1-002 | `S14B1-short_chat-fixed_cost-r1` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=None | - | MEASURED | 9062.84/11218.85/12989.57 | 244.00/244.31 | 430.50 | 30.55 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0008 | registered |
| EXP-S14B1-003 | `S14B1-short_chat-fixed_cost-r2` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=None | - | MEASURED | 9064.85/11210.38/12994.64 | 244.05/244.36 | 430.17 | 30.55 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0011 | registered |
| EXP-S14B1-004 | `S14B1-short_chat-fixed_cost-r3` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=None | - | MEASURED | 9066.59/11221.65/13003.75 | 244.14/244.29 | 430.40 | 30.54 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0006 | registered |
| EXP-S14B1-005 | `S14B1-short_chat-fixed_cost-r4` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=None | - | MEASURED | 9062.42/11213.38/12991.14 | 243.95/244.29 | 430.52 | 30.56 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0012 | registered |
| EXP-S14B1-006 | `S14B1-short_chat-fixed_slo_weave` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 17345.09/25958.67/26929.94 | 146.64/208.75 | 671.03 | 21.56 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2257 | registered |
| EXP-S14B1-007 | `S14B1-short_chat-fixed_slo_weave-r1` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 17321.64/25977.19/26965.54 | 146.43/209.15 | 671.53 | 21.57 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2272 | registered |
| EXP-S14B1-008 | `S14B1-short_chat-fixed_slo_weave-r2` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 19667.37/26240.48/26912.31 | 147.55/209.43 | 643.85 | 19.79 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2425 | registered |
| EXP-S14B1-009 | `S14B1-short_chat-fixed_slo_weave-r3` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 17304.04/25930.72/26930.17 | 146.35/208.56 | 670.33 | 21.60 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2274 | registered |
| EXP-S14B1-010 | `S14B1-short_chat-fixed_slo_weave-r4` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=None | - | MEASURED | 17302.62/25937.02/26922.10 | 146.43/208.67 | 670.31 | 21.60 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2273 | registered |
| EXP-S14B1-011 | `S14B1-short_chat-fixed_static` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 8125.26/8133.73/9172.66 | 155.92/176.94 | 1308.26 | 39.45 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0159 | registered |
| EXP-S14B1-012 | `S14B1-short_chat-fixed_static-r1` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 8133.63/8138.88/9154.89 | 156.05/176.40 | 1302.26 | 39.43 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0204 | registered |
| EXP-S14B1-013 | `S14B1-short_chat-fixed_static-r2` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 8133.15/8139.48/9151.50 | 156.02/176.36 | 1302.21 | 39.44 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0211 | registered |
| EXP-S14B1-014 | `S14B1-short_chat-fixed_static-r3` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 8132.75/8141.84/9161.39 | 156.13/176.57 | 1302.14 | 39.42 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0192 | registered |
| EXP-S14B1-015 | `S14B1-short_chat-fixed_static-r4` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 8131.65/8136.19/9150.41 | 155.96/176.30 | 1301.78 | 39.45 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0210 | registered |
| EXP-S14B1-016 | `S14B1-short_chat-fixed_threshold` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 9060.07/11215.30/13089.77 | 243.95/244.33 | 430.80 | 30.52 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0009 | registered |
| EXP-S14B1-017 | `S14B1-short_chat-fixed_threshold-r1` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 9061.89/11212.28/13003.93 | 243.96/244.27 | 429.84 | 30.55 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0012 | registered |
| EXP-S14B1-018 | `S14B1-short_chat-fixed_threshold-r2` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 9062.57/11219.18/12992.03 | 243.96/244.30 | 430.57 | 30.55 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0012 | registered |
| EXP-S14B1-019 | `S14B1-short_chat-fixed_threshold-r3` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 9066.45/11220.04/13001.27 | 244.14/244.26 | 431.94 | 30.54 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0005 | registered |
| EXP-S14B1-020 | `S14B1-short_chat-fixed_threshold-r4` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=None | - | MEASURED | 9062.92/11214.45/12987.52 | 243.97/244.24 | 429.99 | 30.56 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0010 | registered |

---

## `S14D` · fixed 格的**正式结果**（v5 + host 项 + `decode_p90`）

> **登记于 2026-09-28**（用户裁定：`S14D` 作为 `fixed` 格的正式结果，批次 2 不重跑 `fixed`）。
>
> **为什么 `S14D` 才是 `fixed` 格的有效对照**：第 14 步批次 1（`S14B1`，commit `d20727d`）
> 的配置快照里**没有** `cost_model_path` ⇒ 加载后落成 `None` ⇒ `arm_factory` 回落默认
> `model_v1`；且**没有** `signal_criterion` ⇒ 落成默认 `tpot_p90`。
> `S14D`（commit `158f83a`）三项齐全：`cost_model_path=model_v5.json`、
> `signal_criterion=decode_p90`、`host_overhead_ms/_fit` 已在配置里。
> ⇒ **`S14B1` 是旧口径（v1 + `tpot_p90`），`S14D` 是 FREEZE §2/§4/§5 的口径**（除 host 项未进
> `describe()` 留存痕迹，见下）。
>
> ⚠️ **`S14D` 的产物不能自证 host 项生效**：`host_overhead_ms` / `host_overhead_fit` /
> `apply_safety_margin` 三项在 `e814e2a` 才进 `ControllerConfig.describe()`，而 `S14D` 早于它
> ⇒ 产物里这三项是 `None`。归因只能靠**仓库配置 + `git_commit`**（可追溯，但**不是产物自证**）。
> 这与 `EXPERIMENTS.md:1113` 的既有登记一致。
>
> ⚠️ **与 `S14B1` 的比较是"多变量"**（host 项 + `signal_criterion` + 模型 v1→v5 三处不同）
> ⇒ **只能作参考**，**不得**把任何差异单独归给 host 项（`EXPERIMENTS.md:1071`，§20.12）。
>
> ⚠️ **"离散度(稳健CV)"这一列是 `intra-run`（同一次运行内请求间）的离散度**，
> **不是** FREEZE §7.3 的判据。§7.3 判据是**跨重复**的稳健 CV（同格 ≥3 次重复），
> 本次按吞吐/TTFT 计算为：`cost 0.31%/0.28%`、`threshold 3.42%/5.66%`、
> `static 0.38%/0.14%`、`slo_weave 0.26%/0.26%` ⇒ **全部 ≤ 10%，闸门通过**。
> （两个统计量**分母不同**，混读会把"单次内的长尾"误读成"批次不可复现"。）

<!-- S14D：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S14D-001 | `S14D-short_chat-fixed_cost` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 8980.20/11109.15/12884.24 | 241.69/242.26 | 428.11 | 30.83 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0027 | registered |
| EXP-S14D-002 | `S14D-short_chat-fixed_cost-r1` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 9052.32/11245.06/13008.07 | 242.83/247.40 | 429.83 | 30.54 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0152 | registered |
| EXP-S14D-003 | `S14D-short_chat-fixed_cost-r2` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 9056.44/11185.32/12954.92 | 243.35/244.76 | 427.09 | 30.61 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0060 | registered |
| EXP-S14D-004 | `S14D-short_chat-fixed_cost-r3` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 9035.30/11181.62/12919.57 | 242.81/244.43 | 427.32 | 30.68 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0056 | registered |
| EXP-S14D-005 | `S14D-short_chat-fixed_cost-r4` | short_chat | `fixed_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 9026.49/11133.80/12952.62 | 242.19/244.52 | 427.27 | 30.71 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0069 | registered |
| EXP-S14D-006 | `S14D-short_chat-fixed_slo_weave` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 19505.19/26660.70/29146.02 | 135.13/210.07 | 689.48 | 20.32 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2787 | registered |
| EXP-S14D-007 | `S14D-short_chat-fixed_slo_weave-r1` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 19611.35/26761.65/29319.59 | 135.61/211.39 | 692.97 | 20.21 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2784 | registered |
| EXP-S14D-008 | `S14D-short_chat-fixed_slo_weave-r2` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 19613.14/26827.03/29366.54 | 135.93/210.48 | 690.83 | 20.20 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2764 | registered |
| EXP-S14D-009 | `S14D-short_chat-fixed_slo_weave-r3` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 19650.64/27169.92/29544.18 | 136.63/210.03 | 689.98 | 20.14 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2761 | registered |
| EXP-S14D-010 | `S14D-short_chat-fixed_slo_weave-r4` | short_chat | `fixed_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 19647.86/26920.32/29399.60 | 136.84/211.22 | 690.67 | 20.17 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2783 | registered |
| EXP-S14D-011 | `S14D-short_chat-fixed_static` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 8086.87/8093.32/9134.85 | 154.67/175.81 | 1305.50 | 39.69 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0152 | registered |
| EXP-S14D-012 | `S14D-short_chat-fixed_static-r1` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 8089.10/8098.15/9112.45 | 154.79/175.16 | 1299.96 | 39.69 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0199 | registered |
| EXP-S14D-013 | `S14D-short_chat-fixed_static-r2` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 8113.73/8149.11/9118.81 | 155.52/176.75 | 1300.53 | 39.52 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0229 | registered |
| EXP-S14D-014 | `S14D-short_chat-fixed_static-r3` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 8116.11/8196.18/9144.93 | 156.41/178.13 | 1305.43 | 39.41 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0197 | registered |
| EXP-S14D-015 | `S14D-short_chat-fixed_static-r4` | short_chat | `fixed_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 8108.58/8161.49/9152.79 | 156.18/177.19 | 1306.41 | 39.42 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0166 | registered |
| EXP-S14D-016 | `S14D-short_chat-fixed_threshold` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 7817.11/9531.79/10849.91 | 217.43/222.82 | 602.73 | 34.72 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0376 | registered |
| EXP-S14D-017 | `S14D-short_chat-fixed_threshold-r1` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 8806.94/10674.59/11376.90 | 229.51/243.67 | 564.75 | 31.84 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0783 | registered |
| EXP-S14D-018 | `S14D-short_chat-fixed_threshold-r2` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 8034.55/10814.93/11446.18 | 216.44/229.11 | 605.23 | 33.28 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0696 | registered |
| EXP-S14D-019 | `S14D-short_chat-fixed_threshold-r3` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 8744.30/10968.03/11479.85 | 227.47/245.46 | 561.84 | 31.73 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0792 | registered |
| EXP-S14D-020 | `S14D-short_chat-fixed_threshold-r4` | short_chat | `fixed_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 8483.28/10643.98/11283.36 | 220.21/241.63 | 649.92 | 32.59 | 0.000 | 0.00 | 0.00 | 0.000 | 0.1018 | registered |


---

## 第 15 步消融 · **预注册**（2026-09-26 登记，**尚未开跑**）

> 依据 `paper/experiments.md` §2「每个实验必须**预先**写下要做什么」与 §5 第 7 条。
> 配置已生成：`configs/mthreads/phase1_step15/`（生成器 `scripts/gen_phase1_step15_configs.py`，
> 单变量性由 `tests/test_step15_configs.py` 强制）。
> ⚠️ **事后补预期是本项目明令禁止的做法** —— 本节写于任何一步消融产生数据**之前**。

### 0. 参考点与**归因边界**（先说清哪一对才是单变量对照）

| 对 | 差异 | 可归因吗 |
| --- | --- | --- |
| `full` vs `no_host` / `no_margin` / `signal_tpot` / `chunk_only` / `budget_only` | **恰好一处**（由用例强制） | ✅ **只有这一族可以做因果归因** |
| `S14D`/`S14F` vs `S14B1`（84.62% D1） | 至少三处：host 项、`signal_criterion`（B1 是默认 `tpot_p90`）、成本模型（v1 → v5/v6） | ❌ **多变量**，只能作**参考**，**不得**把 D1 的下降单独归给 host 项（§20.12） |

**不变项清单**：模型 `Qwen2.5-7B-Instruct` fp16 / `short_chat` 1536→32 / `concurrency 16` /
`scheduler.max_num_seqs 8` / `chunk.policy fixed_chunk chunk_size 1024` / `cache off` /
SLO `tpot_ms 100`、`ttft_ms 2000` / 判据 `decode_p90` + `signal_mode layered` /
`target_tpot_ms 120`、`hysteresis 0.08` / 预热 30 s / **5 次重复** / 成本模型 `model_v6.json`。

**唯一变量**：见下表「改什么」列（其余逐字节相同，由用例强制）。

### 1. 六个条件与**可证伪的预期**（含方向与量级）

| 条件 | 改什么 | 可证伪的预期 | 否证形态（出现即按"预期不成立"如实记） |
| --- | --- | --- | --- |
| `full` | —（reference） | D1 明显低于 `S14B1` 的 **84.62%**（参考值，多变量）；模型回退率随 `lookup` 桶口径给出 | —（reference） |
| `no_host` | 去掉 `host_overhead_ms`/`fit` | **D1 相对 `full` 上升 ≥ +10pp**。依据：host @b8 = **25.29 ms**，占 120 ms 预算的 **21%** ⇒ b8 的"前向 104.8 ≤ 120 ⇒ 可行"正是错误接纳的来源 | D1 差 < **5pp** 或符号相反 ⇒ **host 项不是 D1 的主因**，必须改口 |
| `no_margin` | `apply_safety_margin: false` | 对 D1 影响**很小（< 3pp）**；`safety_margin` = **5.04%**，在 120 ms 上约 **6 ms** ⇒ 主要体现为**略微更保守**（D2 略降或持平） | D1 上升 > **10pp**、或 D3 显著变好 ⇒ 余量在起实质作用，**与预期相反**，需重新解释 |
| `signal_tpot` | 判据换回 `tpot_p90` | 读数被 prefill 污染（实测读数 ≈ 纯 decode 的 **5.5×**）⇒ 控制器**误判过载并收缩**：动作次数显著增加、**收缩为主**、D2 上升、TPOT p50 变差；**D1 不一定变差** | 与 `full` 在"动作次数/方向/D2"上**几乎一样** ⇒ 该负载下 prefill 污染不出现（须说明与 S9/S10 那批为何不同） |
| `chunk_only` | 冻结单步 token 预算 | 能动维度少一个 ⇒ 动作集更小；**预期明显差于 `full`**（分块一动就影响 decode 延迟） | 与 `full` 无差异 ⇒ 先看 `n_control_decisions`：**"没被用上 ≠ 用了没用"**，不得据此说"分块维度无用" |
| `budget_only` | 冻结分块 | 与 `full` **接近**（该格的主要限制项是 prefill 额度而非分块） | 明显差于 `full` ⇒ 说明分块维度在该格上是必需的（与预期相反） |

### 2. 失败判据（沿用 §4，不新造）

* 同一格**两次重复的稳健 CV > 10%** ⇒ **先修测量**，不得进入对照比较；
* 留出点 < 15 ⇒ 该格判据只能记 **「未判」**；
* 某臂全面胜出**不是**本节的必要条件 —— 结论形式是"**误差 → 错误决策 → 服务代价**"；
* 全部 goodput = 0 只揭示**过载边界**，**不能**用来证明一般性能优势。

### 3. 必须一起报的量（§2 的五组，一组都不许省）

**预测层**：留出误差**五个数**（train n / holdout n / 训练最小秩 / 留出最小秩 / 低估 p95）+ **模型回退率**；
**决策层**：**D1 可行超时率**、**D2 阻塞步占比 + Σ队列·步**、`n_steps_chunk_request_not_honoured`；
**服务层**：**D3 达标吞吐 / 达标率**、TTFT（p50/p90/p99）、TPOT（同）、**stall P99**、吞吐、墙钟；
**结局层**：失败 / 拒绝 / 未完成 率、账目自查误差 P99；
**溯源**：`git_commit` + `dirty` + `data_label` + **实际生效的判据**。

### 4. ⚠️ 开跑前必须满足的两个前置条件

1. **整树同步 + 重新打戳**：判据口径的三项（`host_overhead_ms` / `host_overhead_fit` /
   `apply_safety_margin`）**刚刚**才进 `ControllerConfig.describe()`（`e814e2a`）。
   远端当前源码早于它 ⇒ 若现在跑，产物**无法自证**"host 项有没有进判据"，
   而 `no_host` 这条消融的全部意义就在这一点上。
   ⇒ **必须先把整棵树同步过去并重新投放溯源戳**，再跑消融（`dirty` 的产物不得参与合并解释）。
   （同理：`S14D`/`S14F` 的产物**不会**带这三项；它们与 `S14B1` 的差异归因只能靠
   仓库配置 + `git_commit`，**不得**写成"产物已证明 host 项生效"。）
2. **`model_v6.json` 已在远端且已取回本地**（第 15 步的 6 个配置都指向它）。

### 5. 尚未决定（**先裁后跑**）

* **在哪个 cell 上做消融**：本节配置按 `fixed` 生成（它已有 D1 基线、且单次最省机时 ≈ 1.2 min）。
  若第 14 步结果显示 `fixed` 上 D1/D2 无区分度，则应改到 `slo_reach` 或 `ramp` 上重生成
  （生成器支持，改 `REFERENCE` 即可），**并在此处补一条修订说明**。
* 消融是否需要在**未参与调优的轨迹**上复跑（§1 阶段五）—— 待主结论形态确定后再定。

<!-- P8FULL：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-P8FULL-001 | `P8FULL-load_ramp-ramp_adaptive_c8` | load_ramp | `ramp_adaptive_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27181.19/88880.64/90737.76 | 181.90/182.59 | 329.88 | 25.96 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4458 | registered |
| EXP-P8FULL-002 | `P8FULL-load_ramp-ramp_adaptive_c8-r1` | load_ramp | `ramp_adaptive_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27537.99/89795.72/91650.64 | 182.69/183.47 | 330.77 | 25.96 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4466 | registered |
| EXP-P8FULL-003 | `P8FULL-load_ramp-ramp_adaptive_c8-r2` | load_ramp | `ramp_adaptive_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27402.01/89887.40/91573.92 | 182.36/184.00 | 340.23 | 25.96 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4468 | registered |
| EXP-P8FULL-004 | `P8FULL-load_ramp-ramp_static_c8` | load_ramp | `ramp_static_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24275.36/79663.61/81303.02 | 175.03/175.43 | 410.08 | 25.96 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4443 | registered |
| EXP-P8FULL-005 | `P8FULL-load_ramp-ramp_static_c8-r1` | load_ramp | `ramp_static_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24519.74/80546.03/82155.61 | 175.91/176.53 | 412.44 | 25.96 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4440 | registered |
| EXP-P8FULL-006 | `P8FULL-load_ramp-ramp_static_c8-r2` | load_ramp | `ramp_static_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24513.71/80812.00/82434.01 | 176.16/177.09 | 412.85 | 25.95 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4442 | registered |

<!-- PHASE1S8：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-PHASE1S8-001 | `PHASE1S8-load_ramp-ramp_static_c8` | load_ramp | `ramp_static_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 34428.80/80216.53/81899.85 | 175.69/175.81 | 411.27 | 31.39 | 0.120 | 0.03 | 3.78 | 0.000 | 1.0494 | registered |

<!-- PHASE1S8c：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-PHASE1S8c-001 | `PHASE1S8c-load_ramp-ramp_static_c8` | load_ramp | `ramp_static_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24300.76/79941.88/81619.36 | 175.35/175.62 | 410.78 | 25.96 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4445 | registered |

<!-- S10：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S10-001 | `S10-load_ramp-ramp_static_c8` | load_ramp | `ramp_static_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24271.38/79692.07/81338.25 | 175.17/175.24 | 408.74 | 25.96 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4446 | registered |
| EXP-S10-002 | `S10-load_ramp-ramp_static_c8-r1` | load_ramp | `ramp_static_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24300.80/79894.77/81551.87 | 175.33/176.76 | 415.85 | 25.96 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4442 | registered |
| EXP-S10-003 | `S10-load_ramp-ramp_static_c8-r2` | load_ramp | `ramp_static_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24319.76/79958.83/81618.21 | 175.41/175.46 | 409.12 | 25.96 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4439 | registered |
| EXP-S10-004 | `S10-load_ramp-ramp_static_c8_bt2048` | load_ramp | `ramp_static_c8_bt2048` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24290.68/79903.25/81560.00 | 175.34/175.45 | 409.13 | 25.96 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4444 | registered |
| EXP-S10-005 | `S10-load_ramp-ramp_static_c8_bt2048-r1` | load_ramp | `ramp_static_c8_bt2048` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24297.58/79884.77/81543.03 | 175.33/175.43 | 409.13 | 25.96 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4443 | registered |
| EXP-S10-006 | `S10-load_ramp-ramp_static_c8_bt2048-r2` | load_ramp | `ramp_static_c8_bt2048` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 24303.24/79908.59/81566.34 | 175.35/175.45 | 409.23 | 25.96 | 0.217 | 0.04 | 5.64 | 0.000 | 1.4442 | registered |
| EXP-S10-007 | `S10-load_ramp-ramp_static_c8_ck2048` | load_ramp | `ramp_static_c8_ck2048` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 22307.52/75571.48/77107.92 | 172.28/172.35 | 616.11 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4502 | registered |
| EXP-S10-008 | `S10-load_ramp-ramp_static_c8_ck2048-r1` | load_ramp | `ramp_static_c8_ck2048` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 22316.89/75559.04/77094.45 | 172.25/172.33 | 615.79 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4507 | registered |
| EXP-S10-009 | `S10-load_ramp-ramp_static_c8_ck2048-r2` | load_ramp | `ramp_static_c8_ck2048` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 22320.59/75596.35/77133.60 | 172.29/172.42 | 615.69 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4506 | registered |
| EXP-S10-010 | `S10-load_ramp-ramp_static_c8_ck512` | load_ramp | `ramp_static_c8_ck512` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27187.35/89192.96/90683.73 | 181.83/181.92 | 313.48 | 25.95 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4313 | registered |
| EXP-S10-011 | `S10-load_ramp-ramp_static_c8_ck512-r1` | load_ramp | `ramp_static_c8_ck512` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27205.84/89221.25/90713.85 | 181.85/181.95 | 313.51 | 25.95 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4312 | registered |
| EXP-S10-012 | `S10-load_ramp-ramp_static_c8_ck512-r2` | load_ramp | `ramp_static_c8_ck512` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27193.41/89190.86/90680.87 | 181.81/181.91 | 313.66 | 25.95 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4311 | registered |
| EXP-S10-013 | `S10-load_ramp-ramp_static_off_c8` | load_ramp | `ramp_static_off_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 22299.94/75559.08/77093.91 | 172.22/172.36 | 616.17 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4502 | registered |
| EXP-S10-014 | `S10-load_ramp-ramp_static_off_c8-r1` | load_ramp | `ramp_static_off_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 22313.17/75573.84/77110.29 | 172.28/172.46 | 616.35 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4501 | registered |
| EXP-S10-015 | `S10-load_ramp-ramp_static_off_c8-r2` | load_ramp | `ramp_static_off_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 22303.44/75594.42/77150.10 | 172.32/172.42 | 616.35 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4505 | registered |

<!-- S10C16：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S10C16-001 | `S10C16-phase1_anchor-static_chunk_c16` | phase1_anchor | `static_chunk_c16` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 14188.87/27843.19/28329.31 | 170.66/174.54 | 409.30 | 44.06 | 0.000 | 0.00 | 0.00 | 0.000 | 1.1818 | registered |
| EXP-S10C16-002 | `S10C16-phase1_anchor-static_chunk_c16-r1` | phase1_anchor | `static_chunk_c16` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 14109.89/27767.13/28253.11 | 170.72/174.57 | 408.97 | 44.12 | 0.000 | 0.00 | 0.00 | 0.000 | 1.1886 | registered |
| EXP-S10C16-003 | `S10C16-phase1_anchor-static_chunk_c16-r2` | phase1_anchor | `static_chunk_c16` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 14106.12/27765.25/28251.22 | 170.75/174.61 | 409.63 | 44.12 | 0.000 | 0.00 | 0.00 | 0.000 | 1.1890 | registered |
| EXP-S10C16-004 | `S10C16-phase1_anchor-static_chunk_c16_ck512` | phase1_anchor | `static_chunk_c16_ck512` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 15856.45/31291.28/32023.23 | 175.86/181.50 | 310.73 | 41.17 | 0.000 | 0.00 | 0.00 | 0.000 | 1.1108 | registered |
| EXP-S10C16-005 | `S10C16-phase1_anchor-static_chunk_c16_ck512-r1` | phase1_anchor | `static_chunk_c16_ck512` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 15850.24/31284.63/32015.83 | 175.81/181.47 | 310.46 | 41.18 | 0.000 | 0.00 | 0.00 | 0.000 | 1.1110 | registered |
| EXP-S10C16-006 | `S10C16-phase1_anchor-static_chunk_c16_ck512-r2` | phase1_anchor | `static_chunk_c16_ck512` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 15856.86/31292.62/32023.80 | 175.87/181.53 | 310.89 | 41.17 | 0.000 | 0.00 | 0.00 | 0.000 | 1.1110 | registered |

<!-- S9AB：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S9AB-001 | `S9AB-load_ramp-ramp_adaptive_c8` | load_ramp | `ramp_adaptive_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27430.25/89484.77/91337.93 | 182.53/182.72 | 330.47 | 25.96 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4470 | registered |
| EXP-S9AB-002 | `S9AB-load_ramp-ramp_adaptive_c8-r1` | load_ramp | `ramp_adaptive_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27506.49/89878.93/91729.75 | 182.71/183.57 | 330.51 | 25.96 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4464 | registered |
| EXP-S9AB-003 | `S9AB-load_ramp-ramp_adaptive_c8-r2` | load_ramp | `ramp_adaptive_c8` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27544.42/89637.98/91541.36 | 182.47/183.25 | 350.94 | 25.96 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4471 | registered |
| EXP-S9AB-004 | `S9AB-load_ramp-ramp_adaptive_decode` | load_ramp | `ramp_adaptive_decode` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 26715.08/87877.35/89685.93 | 180.79/182.82 | 409.79 | 25.96 | 0.206 | 0.04 | 5.36 | 0.000 | 1.4456 | registered |
| EXP-S9AB-005 | `S9AB-load_ramp-ramp_adaptive_decode-r1` | load_ramp | `ramp_adaptive_decode` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 26806.91/88274.00/90038.09 | 181.00/182.92 | 410.30 | 25.96 | 0.206 | 0.04 | 5.36 | 0.000 | 1.4456 | registered |
| EXP-S9AB-006 | `S9AB-load_ramp-ramp_adaptive_decode-r2` | load_ramp | `ramp_adaptive_decode` | controller.kind=None, shadow=None, criterion=None | - | MEASURED | 27051.62/87635.61/89290.13 | 180.54/182.38 | 412.71 | 25.96 | 0.206 | 0.04 | 5.36 | 0.000 | 1.4473 | registered |

---

## 说明：13 个老批次的 **run_id 登记**（2026-09-26 补，对应 D5 裁决）

**为什么补**：登记表里长期存在**两套口径** —— 新批次（`P1S3*` / `P2` / `S13COSTIN*` / `S14B1`）
以 **run_id** 为键，而 `S9AB` / `S10` / `S10C16` / `P8FULL` / `PHASE1S8[c]` 只用**配置名**
（`ramp_static_c8` 等）。导出闸门（`benchmark/preflight.py` 检查 4）要求 run_id 作为
**独立表格单元**出现 ⇒ 这 13 个批次在闸门口径下是**未登记**的
（`scripts/check_traceability.py` 检查 6 会报出来）。
⇒ 本块从**真实产物**补齐 run_id 行；**缺失信息写 `-`，不补造**。

**保留历史解释**：§3.5 的响应面表与 E9/E10/E14 的结论**一字未删**，本块只增加 run_id 粒度。
引用那几批结论时**仍应**指向 §3.x 的结论行；本块的作用是**溯源**（让每个数字能指到一条登记行）。

⚠️ **`离散度(稳健CV)` 列对 `load_ramp` 不适用 —— 不得用它判稳定性。**
该列由 `summary.stability` 算出，是**整段 run** 的稳健 CV；而 `load_ramp` 含
low(120 s) / high(90 s) / recover(240 s) 三段，TTFT 跨段差异（实测 P50 24.3 s vs P95 79.7 s）
**远大于**重复间噪声 ⇒ 该列在 ramp 上反映的是**负载形状**，不是测量稳定性
（实测 `S10` 该列为 **1.44**，即 144%）。
⇒ 判 ramp 的稳定性**必须按段**，做法见 §3.5（按段取中位、用分段 window 口径、跨 3 次重复）。
这与本项目已有的纪律同源：**把不同子群的样本混在一起，得到的区间/离散度没有意义**。

⚠️ 另：本块是**追溯用**的登记行。`S10` 的正式选优依据仍是 §3.5 的**按段**口径，
**不得**用本块整段汇总的 `达标吞吐` 去复核 `Best Static` 的冻结决定
（整段汇总会把 high 段的退化摊薄成"看起来还行"）。

---

## D1 · 候选下限的两步走 · **预注册**（2026-09-26 登记，**尚未开跑**）

> 配置已生成：`configs/mthreads/phase1_d1/`（生成器 `scripts/gen_phase1_d1_configs.py`，
> 带 `--check`；每份与参照的差异由它自己断言）。**模型**：`model_v7.json`（v7 = v6 + 小 chunk 组，
> **新增版本、不覆盖旧模型**）。依据：`NEXT_SESSION_TASKS.md` 的「D1 已裁（离线部分）」。
> ⚠️ 事后补预期是本项目明令禁止的做法 —— 本节写于**任何一步 D1 产生数据之前**。

### 0. 先说清一条容易读错的事：**① 不是单变量实验**
`chunk_size_min` 与 `candidate_chunks` 在①里**一起变**（512→128、`null`→显式 4 元列表）——
因为要让 128 真的可动，**必须**同时把下限降下去（`ChunkLimits` 会把候选夹回下限）。
⇒ ① 只是**可行性验证**（能不能被预测/选中/执行），**不得**用它做因果归因；
因果归因只在 **②**（逐批只差 `chunk_size_min` 一个值）。

### 1. 唯一变量
| 步 | tag | 唯一变量 | 取值 |
| --- | --- | --- | --- |
| ① 验证 | `S14G` | `candidate_chunks`（**并伴随** `chunk_size_min`） | `null`+512 → `[128,256,512,2048]`+128 |
| ② 对照 | `S14H` | `chunk_size_min`（`candidate_chunks` 保持 `null`） | 512 / 256 / 128（逐批一个值） |

**不变项清单**：模型 `Qwen2.5-7B-Instruct` fp16 / `short_chat` 1536→32 / `concurrency 16` /
`scheduler.max_num_seqs 8` / `chunk.policy fixed_chunk` / `cache off` /
SLO `tpot_ms 100`、`ttft_ms 2000` / 判据 `decode_p90` + `signal_mode layered` /
`target_tpot_ms 120`、`hysteresis 0.08` / 预热 30 s / 成本模型 `model_v7.json` /
宿主开销与余量沿用参照配置。

### 2. 可证伪的预期（含方向与量级）
**① `S14G`**（参照 = 同源码同模型的 `fixed_cost_min512.yaml`，即③系列里的 min=512）
* 遥测 `note` 里**必须**出现 `prefill_tok ∈ {128, 256}` 的步（**≥1 步**）；
* `arm_state.n_degraded` **显著低于**参照批（参照批以"降级/回退"为主）；
* D0 可行率 **> 5.7%**（`S14D` 的观测值）。
**否证形态**：128/256 一步都没被执行、或 D0 仍 ≈5.7%、或 `arm_state` 缺失
⇒ **① 判失败，停下补标定，不得进②**。

**② `S14H`**（三点：min=512 / 256 / 128，**互为对照**，同一模型/同一 SLO/同一负载）
* **机械检验**（最先看）：遥测里出现的**最小** `prefill_tok` 应随下限降低而降低；
* **主预期**：D1 **单调不增**：`D1(512) ≥ D1(256) ≥ D1(128)`，且 **`D1(128)` 比 `D1(512)` 低 ≥ 20pp**；
* **次预期**：D2（阻塞步占比）也随下限降低而下降；**TTFT 可能变差**（块更小 ⇒ 步更多）。
* **否证形态**：D1 不随下限下降而下降（或反向）⇒ 说明**候选集不是限制项**，
  必须重新归因（此时**不得**继续加小块，也**不得**把 D1 的变化归给候选集）。

### 3. 失败判据（沿用 §4，不新造）
* 同一格两次重复的**稳健 CV > 10%** ⇒ 先修测量，不得进入对照；
* 留出点 < 15 ⇒ 该格判据只能记「未判」；
* 某臂全面胜出**不是**必要条件 —— 结论形式是「**误差 → 错误决策 → 服务代价**」；
* 全部 goodput = 0 只揭示**过载边界**，**不能**证明一般性能优势。

### 4. 必须一起报的量（§2 五组，一组不许省）
**预测层**：留出误差五个数 + **模型回退率**；**决策层**：**D0 可行率**、**D1**、**D2**、
`n_steps_chunk_request_not_honoured`、**`arm_state` 的 `n_degraded`/`n_prefill_starved`
（**没有就写"数不出来"**，不得填 0）；**服务层**：D3 达标吞吐/达标率、TTFT(p50/p90/p99)、
TPOT、**stall P99**、吞吐、墙钟；**结局层**：失败/拒绝/未完成率、账目误差 P99；
**溯源**：`git_commit` + `dirty` + `data_label` + **实际生效的判据**。

### 5. ⚠️ 开跑前的三个前置条件
1. **整树同步 + 重新打戳**（D2）：本轮已修的多项（`arm_state` 落盘、判据口径三项、`--cfg`、
   `candidate_chunks`）**只有部署后才生效**；未部署就跑，产物里数不出降级/饥饿，
   `describe()` 也不记候选列表 ⇒ ① 的核心判据**无法核**。
2. **`model_v7.json` 已产出**（v7 标定 2 会话 → 拟合 → 导出）。
3. **① 通过**之后才允许起 ②。

### 6. 与既有批次的关系
**不得**与 `S14B1/S14B2/S14C1/S14C2` 混算；`S14D/S14E` 是**多变量**且产物不含 host 项标记，
只作**历史参考**。②的对照只在本系列**内部**成立（同模型/同 SLO/同负载）。

---

## D1② · **选取规则（先冻结，写在看到 ② 的数据之前）**

> 依据主控裁决："**选择'分别扫描不同下限'…并准确登记变量**"，以及"排期以条件满足为准"。
> 本节的作用是把 D3/D7 要用哪个下限这件事**从人的判断变成机械规则** ——
> 否则链会在"等一个判断"处空转。
> ⚠️ 本节写于 `S14H` **尚未产生数据**之前；事后改它是禁止的。

**判据只用同系列内部**（同模型 `model_v7` / 同源码树 / 同 SLO / 同负载），
三点 `D1(512) / D1(256) / D1(128)`（`S14H`，各 5 重复取中位）。

| 条件 | 采用的下限 |
| --- | --- |
| `D1(128) ≤ D1(512) − 20pp` **且** `D1(512) ≥ D1(256) ≥ D1(128)`（单调不增） | **min = 128** |
| `D1(128) ≤ D1(512) − 20pp` **但** 非单调（128 破单调） | **min = 256**（取不破单调的更小点，避开 128 的异常） |
| 以上都不满足（含"不降或反向"） | **维持 min = 512**（放弃小候选）⇒ **必须重新归因**：说明候选集**不是**限制项，**不得**继续加小块，也**不得**把 D1 的变化归给候选集 |
| **三点 D1 全为 0（该格无区分度）** | **不能据此选点**。改用 **D2** 作次级判据：`D2` 单调下降 ⇒ 取 **128**；`D2` 也无区分度 ⇒ **维持 512** 并记 **「未判」** |

**配套纪律**
1. 「无区分度」不等于「都没改善」，也不等于「都一样」——**三者写法不同**，结论里必须区分；
2. 本规则选出的下限**只是把 D3/D7 的配置钉住**，不构成"改善"的结论；
   改善与否仍要按 §6 的证据链（D0/D1/D2/D3）逐层报，并接受 §4 的失败判据（稳健 CV > 10% ⇒ 先修测量）；
3. ④ 的"公平对照"要求（D3）：新静态集须含**本规则选出的下限**并**保留不分块对照**，
   只补 S10 的实际缺口（S10 已有 512/1024/2048/不分块，缺 `1536` 与小 chunk）。

<!-- S14G：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S14G-001 | `S14G-short_chat-fixed_cost_min512` | short_chat | `fixed_cost_min512` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | NOT_MEASURED | 8980.81/11134.19/12887.01 | 241.80/242.09 | 427.03 | 30.83 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0004 | TEST_DATA（仅链路验证，不得进结论） |
| EXP-S14G-002 | `S14G-short_chat-fixed_cost_min512-r1` | short_chat | `fixed_cost_min512` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | NOT_MEASURED | 8970.79/11104.67/12876.59 | 241.53/241.93 | 427.05 | 30.86 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0007 | TEST_DATA（仅链路验证，不得进结论） |
| EXP-S14G-003 | `S14G-short_chat-fixed_cost_min512-r2` | short_chat | `fixed_cost_min512` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | NOT_MEASURED | 8977.09/11109.22/12886.93 | 241.63/242.17 | 427.21 | 30.85 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0015 | TEST_DATA（仅链路验证，不得进结论） |
| EXP-S14G-004 | `S14G-short_chat-fixed_cost_pilot` | short_chat | `fixed_cost_pilot` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | NOT_MEASURED | 11440.17/13429.98/13939.65 | 218.00/231.23 | 426.84 | 28.20 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0305 | TEST_DATA（仅链路验证，不得进结论） |
| EXP-S14G-005 | `S14G-short_chat-fixed_cost_pilot-r1` | short_chat | `fixed_cost_pilot` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | NOT_MEASURED | 11453.24/13442.06/13863.45 | 218.26/231.51 | 427.17 | 28.20 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0342 | TEST_DATA（仅链路验证，不得进结论） |
| EXP-S14G-006 | `S14G-short_chat-fixed_cost_pilot-r2` | short_chat | `fixed_cost_pilot` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | NOT_MEASURED | 11445.60/13441.78/13877.60 | 218.05/231.27 | 428.10 | 28.21 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0348 | TEST_DATA（仅链路验证，不得进结论） |

---

## D1① 闸门 · **修订记录**（2026-09-27）

> 本节按主控指令写：**保留原始失败记录和修订原因**。
> 原始记录**不删不改**（逐字原文见下 §1，远端 `/root/s14_d1.log` 仍在），
> 此处只登记"它为什么错、改了什么、改后怎么算"。

### 1. 原始失败记录（逐字，远端 `/root/s14_d1.log`，2026-09-27 03:01）

```
[d1] pilot  = {'n_runs': 0, 'small_steps': 0, 'min_pf': [], 'n_degraded': [], 'd0': 0.0476, 'has_arm_state': False}
[d1] min512 = {'n_runs': 0, 'small_steps': 0, 'min_pf': [], 'n_degraded': [], 'd0': 0.05405405405405406, 'has_arm_state': False}
   - 条件1：小 chunk(128/256) 一步都没被实际执行
   - 条件2：arm_state/n_degraded **数不出来**（源码未部署到位）⇒ 不可判
   - 条件3：D0 可行率 4.8% 未超过 5.7%
[d1] ⇒ **不启动 ②**（按预注册的否证形态，先补标定）
S14D1_GATE_FAIL
```

### 2. 真因：臂名解析存在**第三份**实现

| 出处 | 解析方式 | `S14G-short_chat-fixed_cost_pilot-r1` → |
| --- | --- | --- |
| `scripts/diagnose_scheduling.py` | 正则 `_RUN_RE` | **`cost_pilot`** ✅ |
| `scripts/analyze_step14.py` | 同一正则（复制了一份） | `cost_pilot` |
| **闸门（`run_s14_d1.sh` 内联 heredoc）** | 字符串替换 | **`fixed_cost_pilot`** ❌ |

⇒ 闸门拿 `cost_pilot` 去查一张按 `fixed_cost_pilot` 建的字典 ⇒ **取到空列表** ⇒
`n_runs: 0` ⇒ 条件1/条件2 的"一步都没执行""数不出来"**都是空列表的产物，不是事实**。

**两条反证（同一次运行的产物内，不是推断）**：

1. `telemetry.jsonl` 的 `note` 里 **小 chunk（128/256）prefill 步 = 168**（3 轮合计）⇒ 执行过；
2. `meta.json` 的 `controller_effect.arm_state` **就在产物里**：
   `{"n_decisions": 37, "n_degraded": 34, "n_prefill_starved": 17, ...}`
   ⇒ "源码未部署到位"不成立（部署戳 `9df0aefa` 晚于引入 `arm_state` 的提交）。

### 3. 顺带查出两处**独立**缺陷

| # | 缺陷 | 危害 |
| --- | --- | --- |
| A | 触发下一阶段用 `grep -c "S14D1_GATE_PASS" "$LOG"`，而 `$LOG` 是 **append-only** | 历史只要有过一次 PASS，之后**每一次**重跑都会误触发 ② |
| B | 闸门**完全不看 `data_label`** | 6 个产物全为 `NOT_MEASURED`（新根缺 `results/raw/_env/`）仍照判 |

### 4. 修订内容（`scripts/d1_gate.py` + 两个队列脚本 + 纯逻辑层）

| 项 | 改法 |
| --- | --- |
| **臂名解析** | 收归**唯一实现** `src/ascend_opt/run_ids.py`；`diagnose_scheduling.py`、`analyze_step14.py`、闸门全部 import 它；解析失败 ⇒ `INCONCLUSIVE`，**禁止**回退猜测（旧路正是回退到 `controller.kind`/字符串替换） |
| **有效性前置** | 判定**之前**逐臂查：重复数 / `data_label` / `label_source` / `execution_environment` / `hardware_platform` / `git_commit` / telemetry / `arm_state`；缺一样即 `INCONCLUSIVE` |
| **三态** | `PASS` / `FAIL` / **`INCONCLUSIVE`**（退出码 `0/1/2`，输入错 `3`）；**只有有效性全过才评估判据** |
| **触发方式** | 只认**本次调用**的退出码；结构化判定写 `results/processed/d1_gate_S14G.json`；**禁止** grep 累计日志 |
| **前置拒跑** | `results/raw/_env/env.json` 不存在 ⇒ `S14D1_ABORTED_NO_ENV`（让"缺指纹"**跑不起来**，而不是跑完作废） |
| **判据版本** | `--criteria` 显式登记；v1 = 原三条（数字原样保留），v2 见 §6 |

新增用例：`tests/test_run_ids.py`（9 例）+ `tests/test_d1_gate.py`（20 例），覆盖
**真实目录名 / 空集 / 非 MEASURED / 缺字段 / 解析异常 / 重复不足 / 臂名与诊断不一致 /
历史日志干扰 / 双向（PASS 与 FAIL 都必须可达）**。

### 5. 用 `S14G` 真实产物重算 v1（**诊断用，不是判定**）

新闸门对同一批产物给出 **`INCONCLUSIVE`**（24 条有效性理由：全批 `NOT_MEASURED`
+ `label_source`/`execution_environment`/`hardware_platform` 为 `None`），并把真实读数并列：

| 臂 | 重复 | 小 chunk 步 | 降级（**连报分母**） | 饥饿 | D0 可行率 |
| --- | --- | --- | --- | --- | --- |
| `cost_pilot`（处理） | 3 | **168** | **117 / 126 = 92.9%** | 45 | **4.76%** |
| `cost_min512`（同批参照） | 3 | 0 | 102 / 111 = 91.9% | 51 | **5.41%** |

⇒ 修正后的 v1 结果：

| 条件 | 旧闸门报 | **真实值** |
| --- | --- | --- |
| 条件1：小 chunk 被执行 | FAIL（"一步都没执行"） | ✅ **PASS（168 步）** |
| 条件2：降级减少 | FAIL（"数不出来"） | ❌ **FAIL（92.9% ≥ 91.9%，不降反升）** |
| 条件3：D0 > 5.7% | FAIL | ❌ **FAIL（4.76%）** |

⇒ 旧的"三条全失败"里**有一条是假的**；真实是 **2 条真失败 + 1 条通过**。
⚠️ **本次判定仍为 `INCONCLUSIVE`**（全批非 `MEASURED`）——上表只是
"若有效性成立会怎么判"，**不得**当判定或结论引用。

### 6. 未做（留待第 3/4 步）

* **判据 v2 未冻结**：`--criteria` 目前只实现 `v1`。v2（改用**同批参照**比较、
  连报降级率与分母、纳入预测误差与服务指标）在第 3 步模型评估完成后写，
  且**必须在重跑之前**登记。
* **`S14G` 重跑未启动**（将用新 tag）：前置是新根补采指纹（第 2 步）。


## 第 3 步 · **按部署路径重评查表模型**（2026-09-27）

> 用户指令第 3 步逐字要求：保留"线性模型未通过"的结论；**不要**把它自动扩大成
> "lookup 已失败"，也**不要**凭 0.11%/0.23% 宣布 lookup 已通过。使用与部署**完全相同**
> 的查表、分桶、余量及 host 计算路径重新评估，同时报告**覆盖率、低估、高估和回退**。

### 1. 先纠正一处**指代错误**（这是本轮最重要的发现）

`scripts/costcalib_fit.py` 报的 `查表` 行**不是部署路径**。三处实质差异：

| 维度 | `costcalib_fit.py` 的 `查表` 行 | **部署路径**（`CostModel.predict(mode="lookup")`） |
| --- | --- | --- |
| 键 | `(batch, prefill_tok, kv_len, n_forwards)` **四元** | `b{batch}_pf{pf}_kv{kv}` **三元**（不含 `n_forwards`） |
| 分桶 | 无（精确键） | **`lookup_bucket`**（`model_v7`：`pf_step=0`, `kv_step=256`） |
| 取值 | `median` | **`max`**（保守，绝不低估） |

⇒ 拿前者的数字回答"lookup 通过了吗"评的是**一个从未被部署的模型**。
配套问题：`costcalib_fit.py` 的 `**混合（部署形态）**` 行用的是 `predict_hybrid`，
它回退到**单超平面** `predict_segmented`（低估 p95 高达 277%~466%），
而部署配置里根本没有"回退到分段模型"这条 —— 部署是**回退到阈值控制器**。
⇒ 该行名为"部署形态"，实为另一个模型 ⇒ **不得**用它当部署证据。

### 2. 一处**必须避开的评估陷阱**（第一版脚本自己踩了）

`model_v7.json` 的 `lookup_table` 是建在**全部 220 条**记录上的（生产模型该用上全部数据）。
若拿它去评**同一批** 220 条：覆盖率恒 = 100%、低估恒 = 0% —— 那是**自匹配**，不是泛化。
第一版脚本就掉进去了：四处上下文（含"外推"）**全 100% 覆盖、全 0% 低估**。
⚠️ **这个"太好"本身就是告警** —— "外推"上下文覆盖率 100% 在逻辑上不可能成立。

⇒ 修正：按每个上下文的**训练划分重拟合键**（键/分桶/取值规则逐字来自部署实现），
只在训练划分上建表，再评留出。

### 3. 修正后的结果（部署路径 · `model_v7` · 220 条 / 2 会话）

| 上下文 | 覆盖率 | 回退 | 低估率 | 低估 p95 | 高估率 | 高估 p95 |
| --- | --- | --- | --- | --- | --- | --- |
| ① 按形状留出（秩保证 1/3） | **6.67%**（5/75） | **93.33%** | 0.00% | **n/a（无低估样本）** | 100% | 248.25% |
| ② 跨会话留出（s2） | 100.00%（110/110） | 0.00% | 0.00% | n/a | 100% | 16.29% |
| ③ 外推（全部 kv=2048） | **0.00%**（0/64） | **100.00%** | n/a | n/a | n/a | n/a |

（带余量口径；`no_margin` 见 `results/processed/lookup_deploy_eval_v7.json`。）

### 4. ★ 结构性诊断：**跨会话留出不是泛化考验**

② 覆盖率 100% 而 ① 只有 6.67% —— 看起来矛盾。真因：

> 两会话（`v7s1` / `v7s2`）是**同一张 107 键网格的两次遍历**，
> 键集合**交集 = 107/107 = 100%**。⇒ 留出会话的形状，训练会话里**全都有**。

⇒ ② 的 100% 是"**换了会话、没换形状**"，**不能**读成"查表泛化好"。
⇒ 判据必须以 **① 按形状留出** 为准（只有它真的留出了形状）。
该诊断已固化为脚本的显式输出（`cross_session_shape_overlap`）+ 用例。

### 4b. ⚠️ **不要把两个"覆盖率"混为一个**（本节与既有 §4 数字的关系）

既有论文 §4 引用的是 **53.57%**（kv_step=256），本节 ① 报的是 **6.67%** —— 二者**不矛盾，
但问的是两个不同问题**，必须分开读：

| 量 | 分母是什么 | 数值 | 回答的问题 |
| --- | --- | --- | --- |
| `coverage_lookup_bucketed_S14B1.json` | **真实服务形状**（S14B1 回放出的 280 个有效工作候选）| 命中 150 ⇒ **53.57%** | "部署到真实负载上，多少决策能命中？" |
| `lookup_bucket_sweep.json`（`loo_n=124`）| 标定网格自身（**留一**）| kv256 **53.57%** / kv1024 **100%** | "换桶宽，覆盖率与高估怎么权衡？" |
| 本节 ① | 标定网格的**秩保证留出**（75 条，形状真被留出）| **6.67%** | "面对**没测过的形状**，查表还命中多少？" |

⚠️ 还要叠加一层口径说明（`check_model_coverage.py:140`）：S14B1 那份 **53.57% 是"重放 +
假设 `kv`"**（旧产物未记 `kv_len`，回放时用 `--assume-kv` 近似）⇒ 它是**近似值**，
必须与"真轨迹"分开表述。新产物已修（记 `kv_len`），重跑后此数应被替换为真值。

⇒ 三者的差别正是**分母**：前两者分母里含大量**训练时见过的形状**（留一/回放），
所以覆盖率高；① 的分母是**刻意留出的新形状**，所以只有 6.67%。
⇒ **"查表覆盖率"必须永远连着分母报** —— 报一个裸数字会被读成部署结论。
⇒ 对"部署后会怎样"更有信息量的是 **S14B1 的 53.57%**（真实服务形状，但为近似回放）；
对"模型泛化能力"更有信息量的是 **① 的 6.67%**。两者都要，不可互相替代。

**具体缺口（`per_batch` 已点名，可直接决定补点）**：真实服务形状上
`b4 = 100%` 但 **`b5 = 0%`、`b6 = 0%`**（`b7 = 57.9%`、`b8 = 66.7%`、`b2 = 50%`）。
⇒ 缺口集中在**中高并发的 `batch ∈ {5,6,7}`**；而标定网格对 `b5/b6/b7` **只测了
`kv ∈ {512,768,1024,1536,2048}` 五个值** ⇒ 真实服务在这些 batch 上的 `kv` 落在
**桶之间的缝里**。⇒ 补点方向明确：**对 `batch ∈ {5,6,7,8}` 补 `kv` 取值**（而非全局加密），
或把 `kv_step` 从 256 收到 128（`lookup_bucket_sweep` 显示 128 与 256 同覆盖，
真正跳变在 512→1024 之间）。

### 5. 结论（**分层，不外扩**）

* ❌ **线性（分段）模型未通过** —— 沿用既有结论，**不变**（v7 低估 p95 16.02% > 5%）。
* ❌ **查表（部署路径）也未通过**，且**失败模式与线性模型完全不同**：
  不是"误差大"，而是"**覆盖率过低**"—— ① 仅 6.67% ⇒ **93.33% 的决策走回退**
  （即退回阈值控制器，成本模型等于没用上），③ 外推 100% 回退。
* ✅ 在**命中**的样本上，查表（无余量）低估 p95 = **0.07%**（①）/ **0.23%**（②）——
  这个"准"是**真的**（取 max ⇒ 构造上不低估），
  但**不得**据它宣布 lookup 通过：它的代价是 93% 的回退，覆盖率才是头部指标。
* ⚠️ 带余量后 ① 高估 p95 冲到 **248%** —— 真因已定位为**余量口径错配**：
  实测**查表路径自身**的低估 p95 = **0.068%**（① 留出、无余量），
  而 `model_v7.safety_margin` = **16.02%**（那是**分段模型**的留出低估误差）。
  ⇒ 余量被**高估约 236 倍**，把命中样本的预测整片推高 ⇒ 双重保守。
  这正是"同一个数被用在两条不同路径上"的后果（与 §4 的 `_kb()` 语义守卫同源）。

### 6. 待办（→ 第 4 步 / 冻结 v2 判据前必须处理）

* **覆盖率是主矛盾**：查表在真实服务形状上的覆盖率必须提升，
  否则第三臂在服务上是"回退到阈值"，无独立贡献。
  **具体缺口已点名**：`b5 = 0%`、`b6 = 0%`（`b7 = 57.9%`、`b8 = 66.7%`）⇒ 补点应
  **优先对 `batch ∈ {5,6,7}` 补 `kv` 取值**，不是全局加密网格。
* **余量口径错配**：`safety_margin` 应由**查表路径自己的**低估尾部推
  （实测 0.068%），而不是复用分段模型的 16.02%；
  或用**配置开关**（已有 `apply_safety_margin`）在查表路径上关掉余量，
  把"保守"交给"取 max + 未命中即回退"这两条**构造性**保证。
* **判据 v2 必须把覆盖率**作为准入条件（覆盖 < 阈值 ⇒ 不得判"通过"），
  并**连报回退次数及其分母**（与 D1 链 `d1_n` 同规）。

新增产物：`scripts/eval_lookup_deployment_path.py`、
`results/processed/lookup_deploy_eval_v7.json`、
`tests/test_eval_lookup_deployment_path.py`（14 例）。


## 第 4 步 · **判据 v2 冻结**（2026-09-27）

> 用户指令第 4 步逐字要求：保留旧判据及原流程未通过记录，**明确登记指标修订理由**；
> 以**同批参照**比较；报告可行 prefill 决策、降级次数**及其分母**、动作实际执行、
> 预测误差和服务指标；"如果新批次仍处于所有方案 goodput 为零的**过载区域**，
> 将它用于**机制诊断**；**服务收益另用预先登记的 SLO 可达负载验证**"。

### 1. 为什么必须改（**先有诊断，后有修订**）

第 3 步查明 S14G 的拒因 top1 逐字是
`预测 752.4 + host 14.2 = 766.5ms 超每步预算 120.0ms`。
量化后：

* 该预测**已含** `safety_margin=16.02%` ⇒ 无余量值 = 752.4 / 1.1602 ≈ **648.5 ms**；
* 648.5 ms vs 预算 120 ms ⇒ 仍是 **5.4× 超支**。

⇒ **超支不是余量造成的**（余量只贡献 16%）。真因是**本批落在过载区**：
这台机器上 `batch 8 + prefill` 的实际步成本**远高于** `target_tpot_ms=120` 所隐含的预算。
这与同批 `d3_goodput = 0.00`、`sla_satisfaction = 0.000` 完全一致。

⇒ 结论：**在过载区用"服务收益"判成败，等于判"负载不可达"，不是判机制。**
⇒ 故 v2 定位为**机制诊断档**；服务收益另开预注册的 SLO 可达负载批次。

### 2. v2 判据内容（**冻结，不得事后调**）

准入 + 四条机制，**全部连报分母**：

| 项 | 判什么 | 阈值/口径 | 不满足 ⇒ |
| --- | --- | --- | --- |
| **准入** | 决策级**模型覆盖率** | `< 50%` ⇒ 挡 | FAIL |
| 机制1 | 动作**真的被执行**（有 prefill 步）| `n_pf_steps > 0` | FAIL |
| 机制2 | 候选**真的被采纳** | `d0.n_feasible > 0` | FAIL |
| 机制3 | 降级/回退**被如实计数** | 两者键在、分母 `n_decisions` 非零 | FAIL |
| 机制4 | 预测误差**可算** | `d0.feasible_rate` 非 None | FAIL |

⚠️ **v2 明确不判**：`D3 服务收益`（goodput / 达标率）。

### 3. 修订理由（**逐条对照 v1，供审阅**）

| # | v1 的做法 | 问题 | v2 的改法 |
| --- | --- | --- | --- |
| 1 | 条件3 阈值 `5.7%` 取自 **S14D（另一个模型）** | **跨模型借用阈值** ⇒ 参照与被测不同源 | 改用**同批参照臂**（`cost_min512`）比较 |
| 2 | 只比降级**次数**（`t >= r`）| 忽略分母；"34 次"离开"37 次决策"没有意义 | 连报 `n_degraded/n_decisions` 与 `n_fallback/n_decisions` |
| 3 | 无覆盖率准入 | 模型没被用上时，收益数字**不是在验这个机制** | 加准入：覆盖 < 50% ⇒ FAIL |
| 4 | 条件3 把低 D0 当失败 | 过载区里 D0 天然低 ⇒ **把"负载不可达"报成"方法失败"** | D0 低**不判失败**，只作过载指示 |
| 5 | 判据里隐含"要服务收益" | 过载区 goodput 恒 0 ⇒ 永远 FAIL，无信息量 | 明确**不判收益**，另批验证 |

**准入阈值 50% 的依据**（不是拍脑袋）：第 3 步实测真实服务形状覆盖 **53.57%**
（`b4=100%`，`b5=b6=0%`）⇒ 50% 是"与已知部署点同级"的最低线。

### 4. 旧记录**保留**（不进 v2 的判定）

* S14G 原批（`NOT_MEASURED`）的 v1 判定原文与三条理由：见上文「D1① 闸门 · 修订记录」§5，
  **逐字未改**。
* 同一批在 v2 下会怎么判（**仅诊断，不是判定**）：
  准入 97.62% ✓、机制1–4 全过 ⇒ **PASS**。
  ⇒ 与 v1 的 `FAIL` 对照，**差异完全来自判据口径**，这正是本次修订要显式化的东西。

### 5. 附带修正：诊断产物新增**决策级覆盖率**

`scripts/diagnose_scheduling.py` 新增 `coverage` 块（该臂所有重复的分子分母**求和再相除**，
不是对率取中位数）。⚠️ 它与 `scripts/check_model_coverage.py` 的 run 级口径**互补**：

| 口径 | 分母 | 回答 |
| --- | --- | --- |
| `diagnose_scheduling.py` | **全部决策**（模型视角）| "模型有没有话可说？" |
| `check_model_coverage.py` | **有效工作决策**（剔除空闲，部署视角）| "部署上多少决策命中？" |

两者**都要报**，**不得互相替代**（分母不同）。实测 S14G：决策级 **97.6%**（123/126）
⇒ 模型**有话可说**，问题在**预算拒绝**（D0 4.76%）—— 这是**过载区**的签名，不是模型失效。

新增产物：`results/processed/d1_gate_S14G_v2.json`；
测试：`tests/test_d1_gate.py::TestCriteriaV2MechanismTier`（10 例）。

### 6. 第 4 步后半的**前置条件**：部署（`commit → push → 打戳`）

重跑 S14G 必须在**新版代码已在远端**的前提下进行，否则跑出来的产物 `git_commit`
会指向旧树（戳校验 `validated=false`），溯源失效。前置状态（2026-09-27）：

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| `commit` | ✅ 完成 | 本机 `HEAD=fb297b6`，作用域 `git_dirty=false` |
| `push` | ⚠️ **未完成** | 远端树仍为 `9df0aefa`；缺 `d1_gate.py` / `eval_lookup_deployment_path.py` / `run_ids.py` 等 **9 个文件** |
| `打戳` | ⚠️ **已投放但校验不通过** | 戳已落到远端（357 B，读回一致），但远端重算摘要 ⇒ `validated=false reason="摘要不符"` |

**打戳校验不通过是正确行为，不是缺陷**：戳指向 `fb297b6`、远端树是 `9df0aefa`，
`read_source_stamp` 如实判为不一致 ⇒ 没有把"版本未知"伪装成"版本已确认"
（`src/ascend_opt/provenance.py` 的设计意图）。**源码推上去后戳应自动转 `validated=true`**，
无需重投。

**远端连接不稳（已单变量分离，非代理问题）**：`http_proxy` 等四个变量在用户
「关闭 VPN」后仍在（关的是客户端，shell 环境变量未清除），但**清空后同样断**
⇒ 根因是**链路 RTT 高且抖动大**（直连 TCP 2.4–5.3 s、banner 偶超 8 s），
不是代理残留。因此 `push`（长连接传整包 tar）在此链路**必被 SIGTERM**，
部署改用**逐文件 `put` + 独立短连接 + 失败单独重试**。详见
`.workbuddy/memory/2026-09-27.md`「15:1x–15:5x」一节。

⚠️ 在 `push` 完成、远端戳转 `validated=true` 之前，**第 4 步后半（重跑 S14G）不得启动**
——否则新增的产物会带着无效溯源入库。

### 7. 收尾协议（`push` 结束后按序执行，每步有闸门）

推送是**长尾**操作：本会话实测单次连接成功率约 1/3，9 个文件在乐观/中位/悲观
三种情形下分别约 **4 / 9 / 20 分钟**（最坏上界 8×15×100 s ≈ 4 h，仅链路彻底不可用时）；
实测单文件耗时 5×~15 s 到 1×~15 s 不等。**推送结束后不是"就完了"，还有 6 步**：

| # | 动作 | 闸门（不满足就停） |
| --- | --- | --- |
| S1 | **核对 9 个文件哈希全部到位**：远端 `md5sum` 逐个比本机 | ✅ **9/9 一致**（16:1x 实测） |
| S2 | **查残留 `_push.tar`** 并清理（此前 push 中断留下的半个包） | ✅ 已清理（原 98304 B，`rm` 后 `ls` 确认 `No such file`） |
| S3 | **验证戳转 `validated=true`**：远端 `read_source_stamp` 重算 | ✅ **`validated=True`，读回 `commit=fb297b69d3cb` == 本机 HEAD** |
| S4 | **补采环境指纹**（用户指令第 2 步）：远端 `PYTHON=python3 bash scripts/inspect_env.sh` | ✅ `platform=mthreads` / `model=MTT S4000` / **`available=True`**；`env.json` 16:20 刷新 |
| S5 | **先小规模验证**：单格 × 1 轮（`--tag S14GV`） | ✅ 见下表——三条全过 |
| S6 | **重跑 S14G（新 tag）** + 用 **v2 判据**独立复算 | ⬜ **待定**（见 §7b 需先冻结负载） |

**S5 产物 `S14GV-short_chat-fixed_cost_pilot`（2026-09-27 16:26）三条全过**：

| 检查项 | 新产物（S14GV） | 旧产物（S14G） |
| --- | --- | --- |
| `data_label` | **`MEASURED`** ✅ | `NOT_MEASURED` ❌ |
| `hardware_env.available` | **`True` / `declared`** ✅ | `None` / `missing` ❌ |
| `reproducibility.git_commit` | **`fb297b69d3cb…`** ✅ | `9df0aefa…` ❌ |
| `git_dirty` | `False` ✅ | — |

**这张表是"部署 + 指纹补采真的生效"的唯一证据**：同一份配置（`fixed_cost_pilot`），
补上 `_env/env.json` 与新源码后，产物从"不可进结论"变成"可进结论"。
判定口径 = 三条**同时**成立才 PASS（缺任一 ⇒ 不扩大）。

⚠️ **S5 同时观察到**：`slo_goodput = 0.0`（64 请求）⇒ 本负载**仍落在过载区**。
这与 v2 = "机制诊断档、不判服务收益"的定位一致，**不是** S5 的失败条件
（S5 只验溯源与标签，不判收益）。

⚠️ **S4 留下的待办**：环境指纹 `discovery` 缺 9/12 项，其中
`MODEL_PATH` / `MODEL_NAME` / `MODEL_REVISION` / `MODEL_PRECISION` 需在远端**显式声明**。
不影响 `available` 判定，但**模型溯源会缺**——正式批次前应补。

### 7b. S6 之前必须先冻结的一件事：**负载**

旧 S14G 批是 `NOT_MEASURED`，**不能**作为对照。重跑前须先决定：

* 本负载（`short_chat`，64 请求）**已知落在过载区**（`slo_goodput=0.0`）⇒ 重跑只能做
  **机制诊断**（v2 档正是为此冻结的），**判不出服务收益**；
* 若要判**服务收益**，须按用户指令第 4 步末句**另开预注册的 SLO 可达负载批次**
  （降低到达率/缩短请求，使 SLO 可达）；
* 两条目的不同，**不得合并成一批** —— 否则又会回到"用服务收益判一个不可达负载"的老问题。

### 7c. ★ SLO 可达负载的定量依据（2026-09-27，基于 `S14GV` 实测分解）

**这部分是 §7b 那个决定的依据，不是估算**。数据源 = `S14GV-short_chat-fixed_cost_pilot`
（`MEASURED`，1 轮，334 步，`concurrency=16` / `max_num_seqs=8` / `prompt=1536` / `gen=32`）。

**顶层画像**（`summary`，全部为 p50）：

| 指标 | 实测 | SLO | 倍数 | 分解 |
| --- | --- | --- | --- | --- |
| TTFT | **11392 ms** | 2000 | **5.7×** | `queue=10687 ms`（**93.8%**）+ `prefill=701 ms`（6.1%） |
| TPOT | **217 ms** | 100 | **2.2×** | 见下（两个独立来源） |
| e2e | 18121 ms | — | — | — |
| `n_slo_satisfied` | **0/64** | — | — | — |

**★ TTFT 的根因是排队，不是算得慢**：`queue` 占 TTFT 的 **93.8%**，`prefill` 只占 6.1%。
`concurrency(16) > max_num_seqs(8)` ⇒ 8 个请求始终在排队。

**★ TPOT 的根因是两层，各占约一半**（这是本轮最重要的机制发现）。
把 334 步按 `scheduled_tokens == running`（纯 decode）分出 109 步：

| `running` | 步数 | 纯 decode TPOT p50 | 达标 |
| --- | --- | --- | --- |
| 1 | 3 | **79.7** | ✅ |
| 2 | 3 | **95.3** | ✅ |
| 3 | 3 | 106.3 | ❌ |
| 4 | 3 | **96.4** | ✅ |
| 5 | 3 | 107.4 | ❌ |
| 6 | 3 | 112.2 | ❌ |
| 7 | 3 | 123.3 | ❌ |
| **8** | **88** | **128.6** | ❌ **1.29×** |

⇒ **门槛在 `running ≈ 4`**：纯 decode 时 `running ≤ 4` 达标。
⇒ **第二层：prefill 混入再叠加 +143 ms**（`running=7/8` 的混合步 TPOT ≈ **271 ms**）。

**两个必须写清的纠正（我自己犯的，已复核）**：

1. **`t_ms` 是累积时间戳，不是每步耗时**。我第一版把它当耗时 ⇒ 算出"固定开销 86 s/步"
   —— 与"334 步 / 72 s 墙钟"直接矛盾。**差分后总和 72242 ms ≈ `wall_ms` 72307 ms** ✅
   才算对上。⇒ 教训：**任何"每步/每次"的量，先用总和核对一遍**，不核对就会得到一个
   看起来完整、方向完全错的分析。
2. **TPOT 的定义是 `tpot = mean(s.intervals)`（逐请求 token 间隔均值），不是 `dt / running`**。
   我第二版用 `dt/running` ⇒ 得出"`running=8` 时 TPOT=16 ms，怎么都达标"，
   与产物 `tpot p50=244.8` 矛盾。核对 `tpot/dt` 比值 **p50 = 1.001** ✅ ⇒ 确认 **TPOT ≈ dt**。
   ⇒ 教训：**用产物自己的字段，不要自己另算一个同名的量**——口径分叉必然给出相反结论。

**由此得到的负载建议（待预注册）**：

| 目标 | 做法 | 依据 |
| --- | --- | --- |
| TTFT 达标 | `concurrency ≤ max_num_seqs`（消掉排队） | queue 占 TTFT 93.8% |
| TPOT 达标 | `max_num_seqs ≤ 4`（纯 decode 门槛） | 上表 running≤4 |
| 避免 prefill 叠加 | 保证候选 chunk 小、不出现 `sched >> running` 的步 | +143 ms 叠加 |

⚠️ **本诊断的边界（不得越过）**：
* `S14GV` 是 **`--repeats=1`** ⇒ 本身**不是结论数据**（纪律：<3 重复不出结论）；
  它只用于**诊断**，用来**设计**下一批。
* `running ∈ {1,2,3,4,5,6,7}` 每档只有 **3 步** ⇒ 门槛"恰在 4"的位置**精度不足**，
  必须在真机上做**预注册的低并发扫描**（`concurrency ∈ {2,4,6,8}`）才能定论。
* 上表的"门槛"是在 **prompt=1536** 下测的；换 prompt 长度必须重测（prefill 项会变）。

**⇒ 因此 S6 的正确形态**：先跑**预注册的低并发扫描批**（同为机制诊断档），
用 ≥3 重复定出 `concurrency` 与 `max_num_seqs` 的达标组合，**再**用该组合跑 SLO 可达批次判服务收益。

### 7d. ★ 低并发扫描批 · **预注册（先冻结规则，再看数据）**

> ⚠️ **本节写于 `S14S`（扫描批）尚未产生任何数据之前。事后改它是禁止的。**
> 本节只做一件事：把"S6 用哪个负载组合"从人的判断变成**机械规则**。
> 与 §7c 的关系：§7c 用 `S14GV` **设计**了本扫描；本节**冻结**扫描怎么读。
> 三者的边界（**不得互相替代**）见下面的「三档分离」。

**1. 这批要回答的唯一问题**

`concurrency`（客户端在飞请求上限）与 `max_num_seqs`（批预算）取哪个组合时，
`short_chat`（`prompt=1536` / `gen=32`，N=64）的 **SLO 可达**？

**不回答**：控制器/分块/成本模型有没有收益。那是服务收益档的事。

**2. 轴与取值（依据 = §7c 的实测，不是估计）**

| 轴 | 取值 | 依据 |
| --- | --- | --- |
| `concurrency` | **2 / 4 / 6 / 8** | §7c：TTFT 93.8% 是排队 ⇒ 必须 ≤ `max_num_seqs` 才谈得上 TTFT 达标 |
| `max_num_seqs` | **4 / 8** | §7c：纯 decode 门槛在 `running ≈ 4` ⇒ 4 是预期可达侧、8 是已知超标侧（1.29×） |

⇒ 网格 = 4 × 2 = **8 格**。

⚠️ **锚点说明（2026-09-27 更正）**：`S14GV` 实测组合是 **`concurrency=16` / `max_num_seqs=8`**
（从产物 `config.workload_overrides` 核出，非凭记忆），**不在本网格内**（16 ∉ {2,4,6,8}）。
因此本网格**没有**与 `S14GV` 逐字段相同的锚点，**不能**用"某格复现 `S14GV`"来验证扫描有效性。
替代做法：`c8_m8` 是网格内最接近 `S14GV` 的一格（仅 `concurrency` 16→8），
它的读数应与 `S14GV` **方向一致**（TPOT 超标、TTFT 显著改善）——
这是**一致性检查，不是复现**，两者读数不同**不**自动算失败（`S14GV` 是 `--repeats=1`）。

**3. 被测对象固定**（与 `S14GV` 同一条代码路径，避免又引入第二个变量）

* 配置目录 = **`configs/mthreads/phase1_d1_scan/`**（**新目录**，不是 `phase1_d1/`：
  `phase1_d1/` 里的 4 份已被 D1①② 占用，且 `phase1_step15/` 被
  `tests/test_step15_configs.py` **逐字节钉住**，扫描配置不得混进去）；
  生成器 = `scripts/gen_phase1_d1_scan_configs.py`，测试 = `tests/test_d1_scan_configs.py`；
* 除 `concurrency` / `max_num_seqs` 外**逐字段相同**，底稿是
  `configs/mthreads/phase1_d1/fixed_cost_pilot.yaml`；
* `controller.kind=cost` / `signal_criterion=decode_p90` / `candidate_chunks=[128,256,512,2048]` /
  `cost_model_path=model_v7.json` —— 全部照抄 `fixed_cost_pilot.yaml`；
* `chunk.policy=fixed_chunk` / `chunk_size=1024`（**本批不测分块**，那是机制档的轴）。

⚠️ **已知覆盖缺口（必须与结果并排报，不得隐去）**：`model_v7` 查表在
`batch ∈ {3,5,6,7}` 上 `prefill_tok` **只覆盖 `{0,512,1024}`，缺 `{128,256}`**
（实测统计：b=1/2/4/8 有 128/256，b=3/5/6/7 没有）。本网格的 `running` 落在
`{2,4,5,6,8}` 一带，其中 5/6 会经过缺口键 ⇒ **命中率必须随结果一并报告**，
命中率低本身不构成"测不出"，但**不得**据此把 `n_degraded` 解释成"策略选择"。

**4. 读数与判据（**冻结**）**

对每一格、每一次重复，从 `meta.json` 的 `summary` 取
（⚠️ **键名已对真产物逐个核对**，见
`results/raw/S14GV-short_chat-fixed_cost_pilot/meta.json`；
分位块**是嵌套的**——`summary.ttft.p50`，**不是** `summary.ttft_ms_p50`）：

| 量 | 键（逐字） | 用途 |
| --- | --- | --- |
| 达标请求数 | `summary.n_slo_satisfied` | 分子 |
| 请求总数 | `summary.n_requests` | 分母 |
| TTFT p50 | `summary.ttft.p50` | 与 SLO 2000 比 |
| TPOT p50 | `summary.tpot.p50` | 与 SLO 100 比 |
| 排队分量 | `summary.queue.p50` | 归因 TTFT（§7c：占 93.8%） |
| 稳定度 | `summary.stability.*_robust_cv` | 第 4 节第 3 条的 CV 门槛 |
| 查表命中 | 由 `controller_effect` / trace 计 | **覆盖率，与误差并排报** |

⇒ 在冻结判据前**先核对键名**是必须的：凭记忆写 `ttft_ms_p50` 之类会**静默取到 `None`**，
使"全部不可达"与"读错了键"无法区分（本项目已踩过同类：局部 import 被 try 吞掉后
产物只多一个 error 字段、退出码仍为 0）。

单格判 **SLO 可达** 当且仅当（3 重复**全部**满足，取最差一次）：

1. `n_slo_satisfied / n_requests` **= 1.0**（不是"接近 1"）；
2. `ttft_ms_p50 ≤ 2000` **且** `tpot_ms_p50 ≤ 100`；
3. 3 重复的稳健 CV（`1.4826·MAD/median`）**≤ 10%**（否则先修测量，不判）。

**5. 由扫描结果选组合的规则（**冻结**）**

设 A = 满足第 4 节的可达组合集合，按 `concurrency` 降序排列（取能压出最大负载的那个）：

| 条件 | 选用的组合 |
| --- | --- |
| A 非空 | 取 A 中 **`concurrency` 最大**者；并列时取 `max_num_seqs` 更小者（更省资源） |
| A 为空，但存在 `ttft≤2000` 且 `tpot≤1.5×` 的组合 | **压低 gen_len 或缩短 prompt 再试一轮**；**不得**直接把超标组合当可达用 |
| A 为空且全部组合 `tpot > 1.5×` | ⇒ 报 **「SLO 在此负载下不可达」**：`short_chat(1536/32)` 本身超出 S4000 单卡能力 ⇒ 换更轻的负载（**记「未判」，不得改 SLO**） |

**6. 三档分离（**不得合并成一批**）**

| 档 | 目的 | 负载 | 现状 |
| --- | --- | --- | --- |
| 机制诊断 | 小 chunk 能否被预测/选中/执行 | `S14GV` 同款（过载也行） | ✅ 已跑 `S14GV` |
| **本档：负载可达性扫描** | 哪个组合 SLO 可达 | 同上，仅改 `concurrency` | **本批** |
| 服务收益 | 三臂（静态/阈值/成本模型）谁更优 | 用本档选出的组合 | 待本档出结果后另开 |

**7. 否定形态（预注册「什么样算失败」）**

* 若**全部 8 格**都不可达 ⇒ 本批**否定**"靠降并发就能让该负载可达"这一假设；
  须在 `RESEARCH_LOG.md` 记一条，并**重新归因**（是 prefill 太重，还是 decode 本体就超），
  **不得**继续加格、**不得**调 SLO、**不得**把结果写成"负载设计成功"。
* 若 `concurrency=8 / max_num_seqs=8` 这一格**与 `S14GV` 方向矛盾**（即 TPOT 达标、
  TTFT 也早已达标，看不出上表的两个根因）⇒ 说明 `S14GV` 的 `--repeats=1` 读数不可靠
  或环境漂了 ⇒ **先查环境指纹**（比对 `env.json`），在查明之前**整批不得进结论**。

**8. 执行纪律**

* `--repeats ≥ 3`（本项目纪律，`<3` 不出结论）；
* 穿插执行由 `run_step3_matrix.py` 保证（同格重复相隔整张表 = 8 格）；
* 起跑前 `pre-flight`：无并发批次、GPU idle、部署戳 `validated=true` 且 `commit=728cb98`
  （⚠️ 更新记录：本批新增 `scripts/gen_phase1_d1_scan_configs.py` 等，`scripts/` 在
  `SOURCE_DIGEST_SCOPE` 内 ⇒ 落盘即让旧戳 `fb297b6` 失效，重投后为 **`728cb98`**；
  此改动**未触碰** `src/ascend_opt/` 与 `benchmark/` ⇒ 被测代码不变）、
  `results/raw/_env/env.json` 在位（S4 已补）；
* 产物增量回拉，`results/raw/**` 只读。

**9. 耗时预估（依据全来自 `S14GV` 实测，不是估计）**

实测输入：`S14GV` 计时窗口 **72.31 s**（`n_requests=64`）；`warmup` **32.21 s / 5 迭代**
（`batch=8` / `prompt=1536` / `gen=32`，见产物 `backend_stats.warmup`）。

按 §7c 的逐 batch TPOT 表外推各格的计时窗口（`batch = min(concurrency, max_num_seqs)`）：

| 格 | batch | 窗口估算 |
| --- | --- | --- |
| c2_m4 / c2_m8 | 2 | ~98 s |
| c4_m4 / c4_m8 | 4 | ~49 s |
| c6_m4 | 4 | ~34 s |
| c6_m8 | 6 | ~40 s |
| c8_m4 | 4 | ~25 s |
| c8_m8 | 8 | ~33 s |

⇒ 一轮 8 格 ≈ **7 min**；3 轮 ≈ **21 min**（仅计时窗口）。

**每次运行都要重新装载模型**（`run_once` 每格新建 `RealBackend`，见
`benchmark/run_benchmark.py:1311`）。这是**有意的**：跨 run 复用后端会让上一次的
显存残留影响下一次，混淆"配置差异"与"残留状态"。因此 24 次运行各有一次装载 + 32 s warmup，
**总时长还要加上这部分**（实测 warmup 32 s/次 ⇒ 24×32 ≈ 13 min；装载未单列耗时，
按 `S14GV` 全程反推，单次合计约 90 s 量级）。

⇒ **本批总时长量级 ≈ 40 min**（区间 30–60 min，取决于装载与链路）。
**不得**为压缩时间改成"一个进程跑多格"——那会破坏"每次测量独立"这一前提。

**S1–S3 的必要性**：`push` 只保证"文件传过去了"，不保证"戳对得上树"。
S3 是唯一能证明"远端树 == 本机 `fb297b6`"的检查，且必须在 S1 通过后才可能成功。

**S4 的必要性**：`results/raw/_env/` 被 `.gitignore` 排除 ⇒ **每次部署到新根都会缺**，
本轮推送**不会**带来它；2026-09-27 已实测踩过一次（6 个 S14G 产物全被标 `NOT_MEASURED`）。
本机现存的 `_env/env.json` 是 **09-24** 的旧指纹，**不能**当作本次环境指纹（换根/换镜像须重采）。

**S6 的顺序不可颠倒**：`run_s14_d1.sh` 头部第 1 条前置就是"整树已同步并**重新投放溯源戳**"，
第 2 条是"`_env/env.json` 必须在位"——两条都被 S1–S4 覆盖。

### 8. 推送耗时预估的依据（可复算）

估算脚本 `.workbuddy/tmp/estimate_push.py`，输入全部来自本会话实测：

* 单次成功耗时：`exec "echo ping"` 实测 **11 / 15 / 27 s** ⇒ 中位 **15 s**；
* 单次失败耗时：多为握手阶段快速失败（`No existing session` / `auth timeout`）⇒ 保守 **25 s**；
* 重试间隔：脚本固定 `sleep 3`；单文件上限 **15** 次；
* 单次成功率：观测 A 组 1/3、B 组 3/3、push4 首文件 5 次才成 ⇒ 中位取 **0.35**。

据此：单文件期望尝试 2.85 次、期望耗时 **~70 s**；剩余 8 文件 ⇒ **~9 min**。
区间：乐观（p=0.60）**~4 min**、悲观（p=0.20）**~20 min**。
（探测与推送并发争用连接会让重试节奏变慢，已计入保守项。）
<!-- S14S：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S14S-001 | `S14S-short_chat-fixed_cost_scan_c2_m4` | short_chat | `fixed_cost_scan_c2_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1155.36/1258.12/1260.69 | 86.65/88.42 | 254.24 | 16.50 | 1.000 | 0.52 | 16.50 | 0.000 | 0.1302 | registered |
| EXP-S14S-002 | `S14S-short_chat-fixed_cost_scan_c2_m4-r1` | short_chat | `fixed_cost_scan_c2_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1155.37/1257.97/1258.99 | 86.66/88.37 | 253.25 | 16.51 | 1.000 | 0.52 | 16.51 | 0.000 | 0.1305 | registered |
| EXP-S14S-003 | `S14S-short_chat-fixed_cost_scan_c2_m4-r2` | short_chat | `fixed_cost_scan_c2_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1156.11/1258.20/1259.56 | 86.69/88.40 | 253.34 | 16.51 | 1.000 | 0.52 | 16.51 | 0.000 | 0.1295 | registered |
| EXP-S14S-004 | `S14S-short_chat-fixed_cost_scan_c2_m8` | short_chat | `fixed_cost_scan_c2_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1155.06/1258.53/1258.91 | 86.68/88.33 | 252.91 | 16.51 | 1.000 | 0.52 | 16.51 | 0.000 | 0.1310 | registered |
| EXP-S14S-005 | `S14S-short_chat-fixed_cost_scan_c2_m8-r1` | short_chat | `fixed_cost_scan_c2_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1155.60/1258.40/1259.10 | 86.60/88.28 | 252.90 | 16.51 | 1.000 | 0.52 | 16.51 | 0.000 | 0.1304 | registered |
| EXP-S14S-006 | `S14S-short_chat-fixed_cost_scan_c2_m8-r2` | short_chat | `fixed_cost_scan_c2_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1154.75/1258.07/1258.28 | 86.63/88.30 | 253.14 | 16.50 | 1.000 | 0.52 | 16.50 | 0.000 | 0.1320 | registered |
| EXP-S14S-007 | `S14S-short_chat-fixed_cost_scan_c4_m4` | short_chat | `fixed_cost_scan_c4_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1487.45/1961.66/1962.79 | 117.98/126.08 | 387.46 | 24.13 | 0.016 | 0.01 | 0.38 | 0.000 | 0.2946 | registered |
| EXP-S14S-008 | `S14S-short_chat-fixed_cost_scan_c4_m4-r1` | short_chat | `fixed_cost_scan_c4_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1487.04/1961.61/1962.40 | 117.90/126.00 | 387.22 | 24.13 | 0.016 | 0.01 | 0.38 | 0.000 | 0.2949 | registered |
| EXP-S14S-009 | `S14S-short_chat-fixed_cost_scan_c4_m4-r2` | short_chat | `fixed_cost_scan_c4_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1487.62/1961.99/1962.54 | 118.08/126.17 | 387.35 | 24.12 | 0.016 | 0.01 | 0.38 | 0.000 | 0.2948 | registered |
| EXP-S14S-010 | `S14S-short_chat-fixed_cost_scan_c4_m8` | short_chat | `fixed_cost_scan_c4_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1488.06/1961.97/1962.87 | 117.86/125.98 | 387.54 | 24.13 | 0.016 | 0.01 | 0.38 | 0.000 | 0.2936 | registered |
| EXP-S14S-011 | `S14S-short_chat-fixed_cost_scan_c4_m8-r1` | short_chat | `fixed_cost_scan_c4_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1487.33/1961.34/1961.89 | 117.90/126.00 | 387.21 | 24.13 | 0.016 | 0.01 | 0.38 | 0.000 | 0.2943 | registered |
| EXP-S14S-012 | `S14S-short_chat-fixed_cost_scan_c4_m8-r2` | short_chat | `fixed_cost_scan_c4_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1488.23/1962.11/1962.86 | 117.94/126.08 | 387.32 | 24.12 | 0.016 | 0.01 | 0.38 | 0.000 | 0.2937 | registered |
| EXP-S14S-013 | `S14S-short_chat-fixed_cost_scan_c6_m4` | short_chat | `fixed_cost_scan_c6_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 2408.49/6492.20/6495.18 | 115.72/125.95 | 661.98 | 24.90 | 0.000 | 0.00 | 0.00 | 0.000 | 0.7918 | registered |
| EXP-S14S-014 | `S14S-short_chat-fixed_cost_scan_c6_m4-r1` | short_chat | `fixed_cost_scan_c6_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 2408.32/6489.15/6494.72 | 115.71/125.91 | 661.39 | 24.91 | 0.000 | 0.00 | 0.00 | 0.000 | 0.7922 | registered |
| EXP-S14S-015 | `S14S-short_chat-fixed_cost_scan_c6_m4-r2` | short_chat | `fixed_cost_scan_c6_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 2409.79/6492.50/6497.40 | 115.70/125.92 | 661.52 | 24.90 | 0.000 | 0.00 | 0.00 | 0.000 | 0.7925 | registered |
| EXP-S14S-016 | `S14S-short_chat-fixed_cost_scan_c6_m8` | short_chat | `fixed_cost_scan_c6_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 2223.61/2830.98/3551.74 | 176.84/191.66 | 428.29 | 24.89 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0739 | registered |
| EXP-S14S-017 | `S14S-short_chat-fixed_cost_scan_c6_m8-r1` | short_chat | `fixed_cost_scan_c6_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 2225.62/2826.84/3550.86 | 176.82/191.63 | 427.33 | 24.89 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0721 | registered |
| EXP-S14S-018 | `S14S-short_chat-fixed_cost_scan_c6_m8-r2` | short_chat | `fixed_cost_scan_c6_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 2225.69/2830.53/3552.15 | 176.88/191.73 | 428.44 | 24.88 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0723 | registered |
| EXP-S14S-019 | `S14S-short_chat-fixed_cost_scan_c8_m4` | short_chat | `fixed_cost_scan_c8_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 6682.16/7210.43/7212.63 | 115.72/125.95 | 661.40 | 24.90 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0823 | registered |
| EXP-S14S-020 | `S14S-short_chat-fixed_cost_scan_c8_m4-r1` | short_chat | `fixed_cost_scan_c8_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 6681.90/7208.38/7214.75 | 115.75/125.93 | 661.13 | 24.91 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0815 | registered |
| EXP-S14S-021 | `S14S-short_chat-fixed_cost_scan_c8_m4-r2` | short_chat | `fixed_cost_scan_c8_m4` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 6677.44/7209.54/7213.09 | 115.71/125.94 | 661.06 | 24.90 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0823 | registered |
| EXP-S14S-022 | `S14S-short_chat-fixed_cost_scan_c8_m8` | short_chat | `fixed_cost_scan_c8_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 2342.20/4151.33/4481.94 | 218.57/231.74 | 428.68 | 28.16 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0688 | registered |
| EXP-S14S-023 | `S14S-short_chat-fixed_cost_scan_c8_m8-r1` | short_chat | `fixed_cost_scan_c8_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 2341.74/4152.41/4484.11 | 218.60/231.70 | 427.51 | 28.16 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0688 | registered |
| EXP-S14S-024 | `S14S-short_chat-fixed_cost_scan_c8_m8-r2` | short_chat | `fixed_cost_scan_c8_m8` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 2342.63/4154.33/4482.48 | 218.64/231.77 | 427.99 | 28.15 | 0.000 | 0.00 | 0.00 | 0.000 | 0.0693 | registered |


---

## 第 5 步 · **低并发扫描批 `S14S`**（2026-09-27，预注册见 §7d）

> **本批只判「SLO 可达性」**，不判服务收益（§7d 第 6 节「三档分离」）。
> 24 次运行（8 格 × 3 轮）全部 `MEASURED`，**24/24 成功、0 失败**。
> 产物：`results/raw/S14S-*`（24 个目录，96 个文件）。
> 源码戳：`git_commit=8eaf191c`（24/24）、`git_dirty=false`。

### 1. 实测读数（逐格 3 轮中位；判定用** 3 轮全部满足**，取最差一次）

| 格 | `concurrency` | `max_num_seqs` | `n_slo/n`（逐轮） | TTFT p50 | TPOT p50 | TPOT/SLO | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_cost_scan_c2_m4` | 2 | 4 | 64/64 · 64/64 · 64/64 | 1155.4 | 86.66 | 0.87× | **✅ 可达** |
| `fixed_cost_scan_c2_m8` | 2 | 8 | 64/64 · 64/64 · 64/64 | 1155.1 | 86.63 | 0.87× | **✅ 可达** |
| `fixed_cost_scan_c4_m4` | 4 | 4 | **1/64** · 1/64 · 1/64 | 1487.5 | 117.98 | 1.18× | ❌ 率不足（TPOT 超标） |
| `fixed_cost_scan_c4_m8` | 4 | 8 | **1/64** · 1/64 · 1/64 | 1488.1 | 117.86 | 1.18× | ❌ 率不足（TPOT 超标） |
| `fixed_cost_scan_c6_m4` | 6 | 4 | 0/64 ×3 | 2408.5 | 115.72 | 1.16× | ❌ 率不足（TTFT+TPOT 双超） |
| `fixed_cost_scan_c6_m8` | 6 | 8 | 0/64 ×3 | 2223.6 | 176.84 | 1.77× | ❌ 率不足（TTFT+TPOT 双超） |
| `fixed_cost_scan_c8_m4` | 8 | 4 | 0/64 ×3 | 6682.2 | 115.72 | 1.16× | ❌ 率不足（TTFT+TPOT 双超） |
| `fixed_cost_scan_c8_m8` | 8 | 8 | 0/64 ×3 | 2342.2 | 218.57 | 2.19× | ❌ 率不足（TTFT+TPOT 双超） |

> 分位块是**嵌套**的：`summary.ttft.p50`，**不是** `summary.ttft_ms_p50`（写错会静默取 `None`）。

### 2. ★ 机械套用 §7d 第 5 节选取规则（判据**在看到数据前冻结**）

```
A（可达集合，3 轮全过）= ['fixed_cost_scan_c2_m4', 'fixed_cost_scan_c2_m8']
⇒ 选用 fixed_cost_scan_c2_m4（concurrency=2 最大；并列取 m 更小）
```

⇒ **S6「服务收益档」的负载组合冻结为 `concurrency=2` / `max_num_seqs=4`**
（`configs/mthreads/phase1_d1_scan/fixed_cost_scan_c2_m4.yaml`）。

复算入口（可重复、不改判据）：`scripts/analyze_scan_batch.py results/raw --tag S14S`。

### 2b. ⚠️ 记录补正（2026-09-28）· 情形二当时被**漏报**，非判据更改

**本节的选定结论不变**（`A = {c2_m4, c2_m8}` ⇒ 选 `c2_m4`）。补正的是**记录完整性**：

§7d 第 5 节有三条情形。当时 `analyze_scan_batch.py` 的 `else` 分支**只印静态文案、
不检查格级条件**，因此**情形二的判定被静默丢掉**了。经复核（对真产物读数）：

| 格 | TTFT p50 | TPOT p50 | 达标率 | 情形一 | **情形二** |
| --- | --- | --- | --- | --- | --- |
| `c2_m4` / `c2_m8` | 1155 | 86.7 | 64/64 | ✅ | ✅ |
| **`c4_m4` / `c4_m8`** | **1487 / 1488** | **118.0 / 117.9** | **1/64** | ❌ | **✅ 为真** |
| `c6_*` / `c8_*` | ≥ 2224 | ≥ 115.7 | 0/64 | ❌ | ❌ |

⇒ **情形二（`ttft.p50 ≤ 2000` 且 `tpot.p50 ≤ 1.5×SLO = 150`）为真**，
触发格是 **`c4_m4` / `c4_m8`** —— 这两格**时延达标而达标率塌到 1/64**。
§7d 的分档**只看时延**，所以它们**属情形二，不属情形三**。
（若把达标率也并进情形二，它们会掉进情形三，从而读出"应报 SLO 不可达、换更轻负载"
这一**与本节结论相反**的建议 —— 那是分档语义被误用。）

⚠️ **触发情形二 ≠ 情形一为空**。本批情形一已非空、已按情形一选定；
情形二在此是**并存信息**（"再降一档负载值得试"），
**不得**据此覆盖情形一的结论。脚本已改为**逐条打印条件真值**并显式声明二者不并存。

* 本补正**未改任何判据**、**未改任何产物**、**结论未变**；只是把当时漏报的一条补齐。
* 该格 `c4_*` 的性质（TTFT 达标、TPOT 超 18%）在下面第 3 节「归因」里**本来就写到了**，
  §7d 的分析也点了名 —— 缺的只是**脚本没有机械地把情形二判定出来**。
* 测试：`tests/test_analyze_scan_batch.py`（13 用例），含"情形一非空时情形二仍须求值"
  与"情形二只看时延、不看达标率"两条；已做双向验证（注入旧行为 ⇒ 相应 3 条变红）。

### 3. 归因：TTFT 与 TPOT 的约束在**不同并发**上生效，二者不重合

* **`concurrency ≤ 2`**：TPOT = 86.7 ms，**低于** SLO 100 达 **13.3%** ⇒ 达标率 1.000。
  但 TTFT p50 = 1155 ms 中的 **~1030 ms 是排队**（`summary.queue.p50` ≈ 127 ms 是
  `c2_m8`，见逐格产物；§7c 已量化"TTFT 93.8% 是排队"）⇒ 降并发**换掉了达标率，代价是 TTFT**。
* **`concurrency = 4`**：TTFT p50 = 1487 ms 仍 **≤2000**，且 `p90` = 1961 ms 恰好贴线；
  但 TPOT = 118 ms **越线 18%** ⇒ 达标率塌到 1/64。**这一格是"TTFT 达标、TPOT 不达标"**，
  与 `c6`/`c8` 的**双超**性质不同。
* **`concurrency ≥ 6`**：TTFT 也越线（2408 / 6682 ms），`c8_m4` 的 TTFT 达 **3.3× SLO**
  （`max_num_seqs=4` 把 8 个并发请求卡在队列里）。
* **`max_num_seqs` 的效应只在 `concurrency=8` 时是"好"的**：`c8_m8`（TPOT 218.6）比
  `c8_m4`（115.7）**更差**在 TPOT，但把 TTFT 从 6682 ms 压到 2342 ms。
  ⇒ **`concurrency` 与 `max_num_seqs` 不是两个独立旋钮**：`max_num_seqs < concurrency`
  时多出来的请求只能排队，TTFT 线性恶化而 TPOT 不变。

### 4. 预注册的一致性检查（§7d 第 7 节，**不是复现**）

`c8_m8` 是网格内最接近 `S14GV`（`concurrency=16`/`max_num_seqs=8`）的一格。
预注册要求它**方向一致**：实测 TPOT = 218.6 ms（**超标**，与 `S14GV` 同向）、
TTFT = 2342 ms（比 `S14GV` 的 11440 ms **显著改善**，同向）⇒ **一致性检查通过，未触发否定形态**。

### 5. 边界与不得引申之处

* ❌ **不得**据本批说"降并发提升了服务收益"——本批没有对照臂，只测了**可行性**。
* ❌ **不得**把 `c2_m4` 的达标率 1.000 与既有 `P1S3*` 批次的达标率横向比较：本批
  `target_tpot_ms` 等控制器参数与 `P1S3B` 之前不同，且负载（1536/32）也不同。
* ⚠️ **查表覆盖缺口**（§7d 第 3 节已预先登记）：`model_v7` 在 `batch ∈ {3,5,6,7}` 上
  `prefill_tok` 缺 `{128,256}`。本批 `running` 落在 `{2,4,5,6,8}`，**命中率须随 S6 一并报**。
* ⚠️ 本批 `离散度(稳健CV)` 列对 `c4/c6/c8` 格偏大（最高 0.79）——那是**跨请求 TTFT 的
  排队离散**，不是测量不稳（TPOT 稳健 CV 全部 ≤ 0.10，见逐行）。判稳定性看 `tpot` 那一列。

---

## 第 6 步 · **S6 服务收益档 · 预注册**（2026-09-27 登记，**尚未开跑**）

> ⚠️ **本节写于 S6 产生任何数据之前。事后改它是禁止的。**
> 依据 = §7d 第 6 节「三档分离」的第三档 + §7d 第 5 节机械选出的负载组合。
> 配置已生成：`configs/mthreads/phase1_s6/`（生成器 `scripts/gen_phase1_s6_configs.py`，
> 带 `--check`；测试 `tests/test_s6_configs.py`，13 条）。

### 1. 这批要回答的唯一问题

在**已证明 SLO 可达**的负载（`c2_m4`）上，
**成本模型自适应**控制器（`kind=cost`）相对 **阈值自适应**（`kind=threshold`）
与 **静态**（控制器整组关闭）**有没有服务收益**？

**不回答**：小 chunk 能否被执行（那是机制诊断档 `S14G` 的事）；负载可达性（`S14S` 的事）。

### 2. 三臂定义（**只差两个开关**）

| 臂 | `controller.enabled` | `controller.kind` | 走哪条路径 |
| --- | --- | --- | --- |
| `static` | `false` | `threshold`（值不被读） | run 路径**整块跳过**控制器 |
| `threshold` | `true` | `threshold` | `AdaptiveController.observe()` → 规则 A/B/C/D |
| `cost` | `true` | `cost` | `observe()` **且** `apply_arm(kind="cost")` |

⚠️ **`static` 臂的 `kind` 必须是 `threshold`**，**不能**留底稿的 `cost`：
`ControllerConfig.__post_init__` 规定 `kind != "threshold"` 时 `enabled` 必须为 `True`
⇒ `kind: cost` + `enabled: false` 会在**加载期**抛 `ValueError`。
（这是**实测踩到的**：第一版生成器只做文本级校验，放过了它；
真加载器一构造就现形。⇒ 生成器现在**同时**做文本级与真加载器级两道校验。）

⚠️ 三臂可比性由「**controller 段除 `enabled`/`kind` 外逐字段相同**」保证，
由 `tests/test_s6_configs.py::TestOnlyTwoSwitchesDiffer` **逐字段**钉住。

### 3. 唯一变量与不变项

**唯一变量**：`controller.enabled` / `controller.kind`（见上表）。

**不变项清单**（三臂**逐字**相同，由测试断言）：
负载 `short_chat`（`prompt=1536` / `gen=32`，N=64）/ `concurrency 2` /
`scheduler.max_num_seqs 4` / `chunk.policy fixed_chunk`（`chunk_size 1024`）/
`chunk_size_min 128` / `candidate_chunks [128,256,512,2048]` / `cache off` /
SLO `tpot_ms 100`、`ttft_ms 2000` / 判据 `decode_p90` + `signal_mode layered` /
`target_tpot_ms 120`、`hysteresis 0.08`、`cooldown_cycles 3` /
成本模型 `model_v7.json` / 宿主开销表 / 预热 30 s / `runtime` 全段。

### 4. 可达性前提（**已由 `S14S` 满足**）

本负载由 §7d 第 5 节规则从 `S14S` 扫描批机械选出：
`A = 3 轮全过的格 = {c2_m4, c2_m8}` ⇒ 取 `concurrency` 最大者、并列取 `m` 更小者
⇒ **`c2_m4`**（`S14S` 实测：`n_slo/n = 64/64 × 3`、TPOT p50 = 86.66 ms = 0.87× SLO）。
⇒ 本批**不再是过载区**，`goodput` 这次**携带信息**（与 `S14G`/`S14GV` 不同）。

### 5. 可证伪的预期（含方向；**不预设量级**）

> ⚠️ 本条只给**方向与判据**，不给"应提升 X%"——量级是观测结果，不是预注册内容。
> 事前写定量幅度会让"没达到"被读成失败，而它可能只是幅度不同。

* **机械前提（最先看，不过则整批不判）**：
  * `cost` 臂的 `controller_trace` 里**必须**有 `cost_policy` 条目（>0 条）；
  * `threshold` 臂的轨迹里**不得**有 `cost_policy` 条目；
  * `static` 臂的轨迹**应为空**（控制器整组跳过）；
  * 三臂的 `arm_state` 字段按臂显形（`static` 无、另两臂有）。
  * **任一不满足 ⇒ 该臂根本没被跑到，整批判「接线失败」，不得当作"没收益"。**
* **主预期**：`cost` 臂的 **达标吞吐** 与 **达标率** 不低于 `threshold` 臂，
  且**至少一项严格高于 `static`**；（若三条臂读数在噪声内重合 ⇒ 见第 7 节否定形态。）
* **次预期**：`cost` 臂的 `arm_state.n_degraded` / `n_prefill_starved` 应**低于** `threshold`；
  `n_steps_chunk_request_not_honoured` 应接近 0。
* **反向预期（必须一起报）**：控制器可能**没有**收益甚至更差——若如此，按第 7 节登记，
  **不得**改判据去把它读成"有"。

**否证形态（预注册「什么样算失败」）**：
* 若三条臂的 `n_slo_satisfied/n` 与达标吞吐差异**全部落在 3 重复的稳健 CV 带内**
  ⇒ 记「**在本负载上三臂不可区分**」，**不得**声称任一臂更优；
* 若 `cost` 臂**严格劣于** `static` ⇒ 记「成本模型自适应在本负载上为**负收益**」，
  并归因（是决策错、还是成本模型在 `c2_m4` 的 `running` 上查表缺口所致——见第 6 节）；
* 若机械前提不过 ⇒ 记「**接线失败**」，**不是**「无收益」。

### 6. ★ 必须一起报的量（§2 五组，一组不许省）

* **预测层**：留出误差五个数（train n / holdout n / 训练最小秩 / 留出最小秩 / 低估 p95）
  + **模型回退率**；
* **决策层**：`arm_state` 的 `n_degraded` / `n_prefill_starved`、
  `n_steps_chunk_request_not_honoured`、`controller_trace` 的 `n_updates` /
  `n_skipped_cooldown`（**振荡证据**）、`chunk_requested` vs `chunk_effective` 差额
  （不为 0 说明决策被闸门静默压回）；
* **服务层**：`D3 达标吞吐` / `达标率`、TTFT(p50/p90/p99)、TPOT(p50/p90/p99)、
  **stall P99**、吞吐、墙钟；
* **结局层**：失败 / 拒绝 / 未完成率、账目自查误差 P99；
* **溯源**：`git_commit` + `dirty` + `data_label` + 实际生效的判据；
* ⚠️ **★ 查表命中率**：`model_v7` 在 `batch ∈ {3,5,6,7}` 上 `prefill_tok` 缺 `{128,256}`
  （§7d 第 3 节预登记）。本负载 `running ∈ {2,4,5,6,8}`（`S14S` 实测）
  ⇒ **命中率必须随本批一并报**；命中率低**本身不构成"测不出"**，
  但**不得**把 `n_degraded` 解释成"策略选择"。

### 7. 执行纪律

* `--repeats ≥ 3`；判据 = **3 轮全部满足**（取最差一次），与 `S14S` 同一口径；
* 穿插执行由 `run_step3_matrix.py` 保证（同格重复相隔整张表 = 3 格）；
* **起跑走唯一入口** `scripts/launch_batch.sh`（取锁 + 起跑一次 + 监视进程放锁），
  **不得**裸 `setsid nohup ... &`；
* `pre-flight` 全部满足才起跑：① 整树同步到**权威根**（`/root/ascend_probe_d1`）
  且戳 `validated` 且 `commit == 本机 HEAD`；② `results/raw/_env/env.json` 在位；
  ③ 无并发批次（`check_remote_busy.py --local --expect-idle` 返回 0）；
  ④ GPU idle；⑤ 产物增量回拉，`results/raw/**` 只读；
* **不得**与本批之外的机制诊断合并成一批（§7d 三档分离）。

### 8. 边界（不得引申）

* ❌ **不得**据本批说"成本模型自适应在一般负载上有收益"——本批只有一个负载
  （`short_chat` 1536/32）与一个并发组合（`c2_m4`）；换个负载必须重测。
* ❌ **不得**把本批达标率与 `S14S` 的 1.000 横比（`S14S` 无对照臂，不是收益证据）。
* ❌ **不得**把 `threshold` 臂的读数当作"既有 `P1S3*` 结论的复现"——
  控制器参数（`target_tpot_ms` 等）与负载都不同。
* ⚠️ 本批的**主指标是服务收益**（达标吞吐 / 达标率），**不是** stall 或平均时延；
  stall 只作解释性旁证（§2 已定：stall 与"服务收益"隔一层）。

### 9. 与前两档的边界（§7d 第 6 节「三档分离」，**不得互相替代**）

| 档 | 目的 | 负载 | 现状 |
| --- | --- | --- | --- |
| 机制诊断 | 小 chunk 能否被预测/选中/执行 | `S14GV` 同款（过载也行） | ✅ `S14G`/`S14GV` |
| 负载可达性扫描 | 哪个组合 SLO 可达 | 仅改 `concurrency`/`m` | ✅ `S14S` ⇒ 选 `c2_m4` |
| **服务收益（本档）** | 三臂谁更优 | **`c2_m4`（本档冻结）** | ✅ **已跑，判定 = FAIL（负收益）** |

---

## 第 6 步 · **S6 服务收益档 · 结果与判定**（2026-09-28，**判据未改**）

> ★ 本节**不改**上面 §S6 预注册的任何一条判据，只登记其实测执行结果。
> 判定由 `scripts/s6_benefit_gate.py`（唯一实现）机械给出，退出码 **1 = FAIL**。

### 1. 批次与溯源

* tag `S6`，9 次运行（3 臂 × 3 轮），**9/9 `MEASURED`**，0 失败 0 拒绝 0 未完成；
* 冻结提交 `25c5ab54497a`、`git_dirty=False`、`tree_digest=369bd1204360…`；
* 起跑 **2026-09-28 01:13:10**，完工 **01:35:46**（走唯一入口 `launch_batch.sh`，锁 `held=True`）；
* 产物：`results/raw/S6-*`（9 目录 × 4 文件）；登记行 = **EXP-S6-001 … EXP-S6-009**。

### 2. 机械前提（预注册 §5）：**全部通过，不是接线失败**

| 前提 | 实测 | 结论 |
| --- | --- | --- |
| `cost` 臂轨迹里有 `cost_policy` 条目（>0） | **480** 条（逐次 160/160/160） | ✅ 过 |
| `threshold` 臂**不得**有 `cost_policy` 条目 | **0** 条（轨迹 399 条，非空） | ✅ 过 |
| `static` 臂轨迹应为空 | **0** 条 | ✅ 过 |

⇒ 三臂**都被真的跑到**。本判定**不是**「接线失败」。

### 3. 主指标读数（每臂 3 轮，判定取**最差一次**，§7 口径）

| 臂 | 达标吞吐 req/s（3 轮） | 达标率 | 吞吐 tok/s | 墙钟 (s) | `n_steps` |
| --- | --- | --- | --- | --- | --- |
| `static` | 0.6003 / 0.6008 / 0.6007 | 1.0000 | 19.21–19.22 | 106.5–106.6 | 1057 |
| `threshold` | 0.6006 / 0.6007 / 0.6007 | 1.0000 | 19.22 | 106.5–106.6 | 1057 |
| **`cost`** | **0.5154 / 0.5156 / 0.5156** | 1.0000 | **16.49–16.50** | **124.1–124.2** | **1274** |

**效应 vs 噪声**：三臂 `goodput_min` 极差 **0.0852**，而最大稳健 CV = 0.0002468
⇒ 噪声带 0.000148 ⇒ **效应是噪声带的 575 倍**，**不构成"不可区分"**（§7 第一个否证形态**未**命中）。

⇒ **判定 = FAIL：`cost` 臂严格劣于 `static`（0.5154 < 0.6003，−14.14%）。**
按 §7 记「**成本模型自适应在本负载上为负收益**」。

### 4. ★ 归因（**必须与判定一起报**）：差异**全在排队**，不在一跳计算

| 臂 | 排队 p50 | 排队 p99 | `chunk_effective` | `chunk_requested` |
| --- | --- | --- | --- | --- |
| `static` | **0.0** | 120.6 | **1024 .. 1024**（恒定） | 1024 .. 1024 |
| `threshold` | **0.0** | 94.0 | **1024 .. 1024**（恒定） | 1024 .. 1024 |
| **`cost`** | **127.2** | **724.2** | **1024 .. 128**（收缩） | 1024 .. 1024 |

机制链（与 §7d 第 6 节「接了一半」的判据同族）：

1. `cost` 臂**确实动了**——`chunk_effective` 从 1024 收缩到 128
   （`n_steps_chunk_request_not_honoured = 1017`、`n_steps_prefill_below_chunk_request = 1209`）；
2. 收缩 ⇒ **步数 1057 → 1274（+20.5%）**，每步干的活变少；
3. ⇒ 请求**在队列里等更久**（排队 p50 `0 → 127.2 ms`、p99 `120.6 → 724.2 ms`）；
4. ⇒ TTFT p50 `588.9 → 1155.2 ms`（**×1.96**）、墙钟 `106.6 → 124.2 s`（**+16.5%**）；
5. ⇒ 达标吞吐 −14.14%。

⚠️ **`n_decisions_changed = 0` 不能用来说"该臂没动"**：`cost` 臂它就是 0，
但生效值确实在变。**判"该臂动过"必须看生效值（`chunk_effective`）的变化，不是决策计数。**

⚠️ 本负载 `c2_m4` 是 **SLO 可达**的（三臂达标率全 1.000），
即 **`cost` 的收缩是在"本来达标"的情况下主动放弃了吞吐** —— 不是"救回了超标"。

### 5. 边界（不得引申，与 §8 一致）

* ❌ **不得**据本节说"成本模型自适应在一般负载上有收益"——恰好相反，见上；
* ❌ **不得**据本节说"阈值自适应有效"——`threshold` 与 `static` 的差异
  （0.6006 vs 0.6003，**+0.05%**）远在噪声带（0.000148？否，该差值 0.0003 略高于带）
  量级 → 严格说**两者不可区分**，本次**没有**测出阈值臂的收益；
* ⚠️ 本节只覆盖 `c2_m4` 一个负载组合；换负载必须重测（§8）；
* ⚠️ 本判定的**方向与预注册 §5 的反向预期一致**（"控制器可能没有收益甚至更差"），
  按 §7 登记为**负收益**，**未**改任何判据。

<!-- S6：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S6-001 | `S6-short_chat-service_benefit_c2_m4_cost` | short_chat | `service_benefit_c2_m4_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1155.22/1258.42/1259.33 | 86.79/88.49 | 253.21 | 16.49 | 1.000 | 0.52 | 16.49 | 0.000 | 0.1311 | registered |
| EXP-S6-002 | `S6-short_chat-service_benefit_c2_m4_cost-r1` | short_chat | `service_benefit_c2_m4_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1155.87/1258.34/1259.24 | 86.69/88.36 | 253.38 | 16.50 | 1.000 | 0.52 | 16.50 | 0.000 | 0.1303 | registered |
| EXP-S6-003 | `S6-short_chat-service_benefit_c2_m4_cost-r2` | short_chat | `service_benefit_c2_m4_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 1155.00/1258.16/1258.83 | 86.74/88.41 | 253.50 | 16.50 | 1.000 | 0.52 | 16.50 | 0.000 | 0.1317 | registered |
| EXP-S6-004 | `S6-short_chat-service_benefit_c2_m4_static` | short_chat | `service_benefit_c2_m4_static` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 588.90/589.99/709.85 | 88.54/90.10 | 266.05 | 19.21 | 1.000 | 0.60 | 19.21 | 0.000 | 0.0929 | registered |
| EXP-S6-005 | `S6-short_chat-service_benefit_c2_m4_static-r1` | short_chat | `service_benefit_c2_m4_static` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 562.87/589.69/663.57 | 88.37/90.11 | 266.15 | 19.22 | 1.000 | 0.60 | 19.22 | 0.000 | 0.0698 | registered |
| EXP-S6-006 | `S6-short_chat-service_benefit_c2_m4_static-r2` | short_chat | `service_benefit_c2_m4_static` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 562.76/589.49/663.87 | 88.36/90.07 | 265.87 | 19.22 | 1.000 | 0.60 | 19.22 | 0.000 | 0.0697 | registered |
| EXP-S6-007 | `S6-short_chat-service_benefit_c2_m4_threshold` | short_chat | `service_benefit_c2_m4_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 562.87/589.74/664.10 | 88.51/90.23 | 266.33 | 19.22 | 1.000 | 0.60 | 19.22 | 0.000 | 0.0698 | registered |
| EXP-S6-008 | `S6-short_chat-service_benefit_c2_m4_threshold-r1` | short_chat | `service_benefit_c2_m4_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 562.99/589.77/664.19 | 88.34/90.08 | 266.24 | 19.22 | 1.000 | 0.60 | 19.22 | 0.000 | 0.0701 | registered |
| EXP-S6-009 | `S6-short_chat-service_benefit_c2_m4_threshold-r2` | short_chat | `service_benefit_c2_m4_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 562.66/589.39/663.56 | 88.39/90.11 | 265.90 | 19.22 | 1.000 | 0.60 | 19.22 | 0.000 | 0.0697 | registered |

## `S14B2` · **负载爬坡批（`load_ramp`）· 溯源与适用范围**（2026-09-28）

> ★ 本章**只补登记的元信息与适用范围边界**。逐行读数见下表；
> 单一 run 的数据形态分歧已由提交 `26c25bd` 定谳并写在表后注中，**此处不重复其论证**。

### 1. 批次与溯源

* **20 次运行**（4 臂 × 5 重复），`20/20 MEASURED`；
* 产物：`results/raw/S14B2-*`（20 目录，4 类文件齐；经三层校验全过）；
* 负载 = `load_ramp`（开放环 piecewise：low 120 s → high 90 s → recover 240 s）。
* runtime 源码戳：`reproducibility.git_source='stamp'`、`git_dirty=False`（20/20）。

### 2. ★ 适用范围告警（**引用本批数字前必读**）

⚠️ 本批**不是**可作三臂正式比较的数据源，**不得**用于填论文占位 `P12`：

1. **控制器模型是旧版**：本批用的是**默认判据**与**当时的成本模型**，
   与后续 `S14B3`/`S6` 批（`decode_p90` + `model_v7`）**不是同一套配置**
   ⇒ 跨批并表会把"配置差异"读成"臂间差异"；
2. `P12` 要求的是**同模型 / 同 SLO / 同负载**下的正式比较 —— 本批三条都不满足；
3. 本批 `slo.tpot_ms=50` 是**重标定前**的旧阈值且**不可达**，故 `slo_goodput` 列
   与 `S14B3`/`S6` 的达标率列**不得并表**
   （跨阈值并表须用 `scripts/calibrate_slo.py --candidates 100` 重算）。

**本批可以回答的问题**（且仅限这些）：负载水平**连续变化**时控制轨迹的形态，
以及**跨重复的可复现性** —— 见第 3 节。

### 3. 本批的真实价值：可复现性证据（不是性能结论）

* 各臂**臂内的 5 次重复**读数高度一致（如 `ramp_static` 5 次 TTFT P50 22301–22332 ms、
  达标率恒 0.228）⇒ 该臂**可复现**；
* 唯一例外是 **`ramp_slo_weave-r4`**（`EXP-S14B2-010`），其分歧已定谳为
  **控制轨迹分岔**，**不是**截断、**不是**测量噪声（见下表后注）；
* ⇒ 本批的结论形式是「**哪些臂可复现、哪个不可复现**」，
  **不是**「哪个臂更快」。

⚠️ **`ramp_slo_weave` 的 5 次重复不等长**：`-r4` 的 `meta.json` = 2,570,180 B，
同臂其余 = 2,843,008 B。已核实为**远端原生**（拉取前后字节数一致），非截断
⇒ **聚合该臂时不得当等长重复求均值**，须标注。

<!-- S14B2：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S14B2-001 | `S14B2-load_ramp-ramp_cost` | load_ramp | `ramp_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 27191.43/89189.79/90680.18 | 181.82/181.93 | 313.56 | 25.95 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4315 | registered |
| EXP-S14B2-002 | `S14B2-load_ramp-ramp_cost-r1` | load_ramp | `ramp_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 27212.08/89243.58/90733.99 | 181.88/181.97 | 313.32 | 25.95 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4312 | registered |
| EXP-S14B2-003 | `S14B2-load_ramp-ramp_cost-r2` | load_ramp | `ramp_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 27187.95/89193.79/90684.15 | 181.81/181.95 | 313.59 | 25.95 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4311 | registered |
| EXP-S14B2-004 | `S14B2-load_ramp-ramp_cost-r3` | load_ramp | `ramp_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 27220.93/89167.02/90656.30 | 181.86/181.96 | 313.29 | 25.95 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4316 | registered |
| EXP-S14B2-005 | `S14B2-load_ramp-ramp_cost-r4` | load_ramp | `ramp_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 27147.67/89234.41/90714.32 | 181.86/182.02 | 313.28 | 25.95 | 0.196 | 0.04 | 5.08 | 0.000 | 1.4312 | registered |
| EXP-S14B2-006 | `S14B2-load_ramp-ramp_slo_weave` | load_ramp | `ramp_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 84053.15/183270.57/192681.70 | 122.83/166.72 | 703.12 | 22.88 | 0.098 | 0.02 | 2.24 | 0.000 | 0.9025 | registered |
| EXP-S14B2-007 | `S14B2-load_ramp-ramp_slo_weave-r1` | load_ramp | `ramp_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 84010.91/183188.41/192598.96 | 122.82/166.68 | 702.96 | 22.89 | 0.098 | 0.02 | 2.24 | 0.000 | 0.9026 | registered |
| EXP-S14B2-008 | `S14B2-load_ramp-ramp_slo_weave-r2` | load_ramp | `ramp_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 84020.36/183217.76/192628.40 | 122.84/166.62 | 703.07 | 22.89 | 0.098 | 0.02 | 2.24 | 0.000 | 0.9025 | registered |
| EXP-S14B2-009 | `S14B2-load_ramp-ramp_slo_weave-r3` | load_ramp | `ramp_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 84128.97/183337.21/192753.51 | 122.86/166.71 | 703.13 | 22.88 | 0.098 | 0.02 | 2.24 | 0.000 | 0.9022 | registered |
| EXP-S14B2-010 | `S14B2-load_ramp-ramp_slo_weave-r4` | load_ramp | `ramp_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 65835.78/143089.94/153897.32 | 129.15/166.56 | 717.97 | 25.37 | 0.098 | 0.02 | 2.48 | 0.000 | 1.0508 | registered |
| EXP-S14B2-011 | `S14B2-load_ramp-ramp_static` | load_ramp | `ramp_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 22333.14/75486.01/77030.78 | 172.21/172.28 | 615.25 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4491 | registered |
| EXP-S14B2-012 | `S14B2-load_ramp-ramp_static-r1` | load_ramp | `ramp_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 22305.72/75575.02/77122.95 | 172.28/172.39 | 615.70 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4502 | registered |
| EXP-S14B2-013 | `S14B2-load_ramp-ramp_static-r2` | load_ramp | `ramp_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 22301.59/75587.16/77138.28 | 172.31/172.40 | 616.15 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4506 | registered |
| EXP-S14B2-014 | `S14B2-load_ramp-ramp_static-r3` | load_ramp | `ramp_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 22332.08/75634.53/77171.69 | 172.36/172.44 | 615.53 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4502 | registered |
| EXP-S14B2-015 | `S14B2-load_ramp-ramp_static-r4` | load_ramp | `ramp_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 22306.81/75572.98/77124.57 | 172.29/172.37 | 615.79 | 25.97 | 0.228 | 0.05 | 5.93 | 0.000 | 1.4500 | registered |
| EXP-S14B2-016 | `S14B2-load_ramp-ramp_threshold` | load_ramp | `ramp_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 26985.27/86790.06/88439.65 | 179.59/181.94 | 409.08 | 25.96 | 0.206 | 0.04 | 5.36 | 0.000 | 1.4470 | registered |
| EXP-S14B2-017 | `S14B2-load_ramp-ramp_threshold-r1` | load_ramp | `ramp_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 26969.45/86769.17/88423.65 | 179.56/181.95 | 409.13 | 25.97 | 0.206 | 0.04 | 5.36 | 0.000 | 1.4473 | registered |
| EXP-S14B2-018 | `S14B2-load_ramp-ramp_threshold-r2` | load_ramp | `ramp_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 26972.79/86766.55/88418.01 | 179.56/181.94 | 409.27 | 25.96 | 0.206 | 0.04 | 5.36 | 0.000 | 1.4470 | registered |
| EXP-S14B2-019 | `S14B2-load_ramp-ramp_threshold-r3` | load_ramp | `ramp_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 26996.40/86818.65/88472.51 | 179.60/182.08 | 409.00 | 25.97 | 0.206 | 0.04 | 5.36 | 0.000 | 1.4470 | registered |
| EXP-S14B2-020 | `S14B2-load_ramp-ramp_threshold-r4` | load_ramp | `ramp_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 26968.45/86769.53/88430.32 | 179.60/181.94 | 408.96 | 25.97 | 0.206 | 0.04 | 5.36 | 0.000 | 1.4472 | registered |

> ⚠️ **`EXP-S14B2-010`（`ramp_slo_weave-r4`）是控制轨迹分岔，不是测量噪声，也不是截断**
> （2026-09-28 查证；权威表述见 `paper/sections/07_benefits.md` §7.3，此处只留索引）。
> 证据（`meta.json` 双源比对 r0/r1/r4）：
> - `workload_fingerprint` **逐位相同**（`num_requests=92` / `prompt_len_mean=2048` /
>   `gen_len_mean=128` / `total_output_tokens=11776`）；
> - `dropped_requests=0`、`stall_reason=""` ⇒ **无丢请求、无 stall**；
> - 差异**只在控制器的步数**（`steps` 4201 vs 4886、`controller_decisions` 1052 vs 1222）
>   与 `chunk` 终值（`last` **1** vs 4）。
> ⇒ 负载相同而控制路径不同 = **`slo_weave` 的 deadline-aware 决策分岔**，不是运行被截断。
> 表中吞吐 25.37 vs 22.88、稳健 CV 1.0508 vs 0.9025 是**该分岔的后果**，**不是性能更好**。
> **三条纪律**：① `r4` **不得剔除**；② **不得**把 `2.481 > 2.239` 读成"该臂更好"；
> ③ **不得**跨臂合并 —— 各臂的可复现性**本身**就是结论。
> 该分岔与"自适应臂在门限贴工作点时跨重复不可复现"**同型**，是一条**可复现性发现**。

## `S14B3` · **`lenmix` 长度混合批 · 覆盖率与回退成因**（2026-09-28）

> ★ **本章按受命口径写：只报「覆盖率」与「回退成因分解」。**
> ⛔ **不得**据本章写"cost 更差"或任何臂间的优劣结论 —— 见第 4 节。
> 分析产物：`results/processed/s14b3.json`；生成器：`scripts/analyze_s14b3.py`。

### 1. 批次与溯源

* **20 次运行**（4 臂 × 5 重复），`20/20 MEASURED`，0 失败 0 拒绝 0 未完成；
* 产物：`results/raw/S14B3-*`（20 目录 × 4 文件 = 80 文件，163,375,613 B）；
  入库前经**三层校验**（4 类文件齐 + 字节数 == 远端 + JSON/JSONL 可解析）全过；
* 负载格 `lenmix`（混合长度），`chunk` 上界 **8192** —— 这是下述回退的直接前提。

**逐 run 溯源（`reproducibility.git_source`，逐条列明，不得合并）**：

| `git_source` | 运行数 | 具体 run |
| --- | --- | --- |
| `stamp` | **19** | 除下表 1 条外的全部 19 个 |
| `stamp_stale` | **1** | **`S14B3-mixed-lenmix_threshold`**（裸名目录，即该臂第 1 次重复） |

⚠️ **两条必须与数字一起读的事实**：

1. `stamp_stale` 落在**裸名** `lenmix_threshold` 上，不是 `-r1`。按本仓运行目录命名口径，
   **裸名目录是该 cell 的第 1 次重复（r0）**，第 2–5 次才是 `-r1…-r4`
   ⇒ 受影响的**不是**"最后一次重复"，而是**第 1 次**。引用时**不得**写成"末次重复溯源可疑"。
2. `stamp_stale` 的另一半语义是 `git_dirty` **无法判定**（该 run 为 `None`，其余 19 个为 `False`）。
   ⇒ 该 run 的 `git_commit` **不可信**；它带着 `data_label=MEASURED`
   （`MEASURED` 说的是"数据来自真机"，**不是**"源码版本已确认"）。这两件事**分开报**。

### 2. 覆盖率（主指标）

| 臂 | `chunk` 建议→生效（首→末） | `chunk` 覆盖率 | `max_batched_tokens` 建议→生效 | 覆盖率 |
| --- | --- | --- | --- | --- |
| `cost` | 1024 → 1536 | **1.0000** | 8192 → 2048 | **1.0000** |
| `slo_weave` | 1024 → 1 | **1.0000** | 8192 → 1 | **1.0000** |
| `static` | 1024 → 1024（恒定） | **1.0000** | 8192 → 8192（恒定） | **1.0000** |
| `threshold` | 1024 → 1536 | **1.0000** | 8192 → 2048 | **1.0000** |

* **覆盖率 = 控制器建议值被实际兑现的比例**，四臂**全部 1.0000**
  ⇒ **不存在"建议被闸门静默压回"**（`chunk_requested` 与 `chunk_effective` 逐格相同）。
* ⚠️ **覆盖率 1.0000 不等于"控制器在工作"**：`static` 臂**没有任何决策**
  （见第 3 节 `n_control_decisions=0`），它的覆盖率 1.0000 只是"常量建议恰好等于常量生效"
  —— 这是**同义反复**，没有信息量。**不得**把 `static` 的 1.0000 与另三臂的 1.0000 并列引用。

### 3. 回退成因分解（分母必须一起报）

| 臂 | `n_control_decisions` | `n_decisions_changed` | `n_decisions_recover` | `n_steps_chunk_request_not_honoured` | `n_steps_prefill_below_chunk_request` |
| --- | --- | --- | --- | --- | --- |
| `cost` | **1074** | 81 | **0** | 8 | 3565 |
| `slo_weave` | **4110** | 119 | **0** | 16222 | 16325 |
| `static` | **0** | 0 | 0 | 0 | 0 |
| `threshold` | **536** | 85 | **0** | **0** | 3581 |

| 臂 | `n_decisions` | `n_fallback` | **回退率** | `n_degraded` | `n_no_candidates` | `n_memory_conservative` | `n_prefill_starved` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cost` | 537 | **536** | **99.81%** | 1 | 0 | 0 | 0 |
| `threshold` | — | — | **数不出来** | — | — | — | — |
| `slo_weave` | — | — | **数不出来** | — | — | — | — |
| `static` | — | — | **数不出来** | — | — | — | — |

**回退成因（单一来源，非推测）**：`cost` 臂 `arm_state.last_fallback_reason`
= **「模型范围外（不可外推）」**，`n_no_candidates = n_memory_conservative = n_prefill_starved = 0`
⇒ **536 次回退全部来自"查表范围外"，没有一次来自候选耗尽、显存保守或预填充饥饿。**

**成因分解（与 §第 3 步/§4 的查表边界一致）**：
`model_v5` 只标定了 `batch 1–8`、`kv` 上限 **2048**；而 `lenmix` 格的 `kv` 上界是 **8192**
⇒ 查表**在结构上**必须回退。**这是配置与模型的适用范围错配，不是控制器"选择"了回退**，
更不是"回退让性能变差"。

### 4. ⛔ 明确**不**能从本章得出的结论（三条）

1. ⛔ **不得**把 99.81% 的回退率读成"`cost` 更差"。
   **回退率高只说明"成本模型没被用上"**；一个没被执行的机制，其收益既没被验证、
   **也没被否证**。要判 `cost`，须先在**它适用**的 `kv ≤ 2048` 形状上重跑（§第 3 步）。
2. ⛔ **不得**把四臂并列比时延/吞吐。本章**只**报覆盖率与回退成因；
   表中 TTFT/TPOT/吞吐等列（见下表）仅供索引，**引用前必须回到 `meta.json` 核对口径**
   （`static` 臂的判据是 `tpot_p90`，另三臂是 `decode_p90` ⇒ **不同判据，不可横比**）。
3. ⛔ **不得**把 `slo_weave` 的 `chunk 建议→生效 1024 → 1`（末值 1）读成"控制器崩了/数据坏了"。
   该臂 5 次重复读数**逐格完全相同**（`n_steps` 16439、`n_control_decisions` 4110、
   `not_honoured` 16222），是**高度可复现的稳态**；`last=1` 是该臂策略的**已知终端行为**，
   与 `S14B2` 的 `slo_weave` 分岔（§下方 `EXP-S14B2-010` 注）**不是同一现象**，不得混提。

> ⚠️ **两臂 `kind` 相同、必须靠 `enabled` 分臂**：`static` 与 `threshold` 的
> `controller.kind` **都是** `threshold`（`static` 臂的 `kind` 是惰性遗留），
> 区分只能靠 `controller.enabled`（`False` / `True`）。
> **新增风险**：任何"按 `kind` 分臂"的脚本都会把这两臂**合并**（见 §4.1 **N6**）。

<!-- S14B3：由 scripts/emit_experiment_rows.py 从产物生成 -->
| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | 达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-S14B3-001 | `S14B3-mixed-lenmix_cost` | mixed | `lenmix_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 108573.07/142319.99/151148.34 | 318.62/344.04 | 699.10 | 24.66 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2532 | registered |
| EXP-S14B3-002 | `S14B3-mixed-lenmix_cost-r1` | mixed | `lenmix_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 108659.16/142832.55/151664.33 | 318.23/344.00 | 698.96 | 24.66 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2550 | registered |
| EXP-S14B3-003 | `S14B3-mixed-lenmix_cost-r2` | mixed | `lenmix_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 108673.27/143740.34/152272.91 | 319.23/345.01 | 851.78 | 24.58 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2565 | registered |
| EXP-S14B3-004 | `S14B3-mixed-lenmix_cost-r3` | mixed | `lenmix_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 108672.32/143663.53/152211.15 | 319.21/347.33 | 780.93 | 24.60 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2564 | registered |
| EXP-S14B3-005 | `S14B3-mixed-lenmix_cost-r4` | mixed | `lenmix_cost` | controller.kind=cost, shadow=False, criterion=decode_p90 | - | MEASURED | 108663.82/143141.44/151197.29 | 318.29/346.42 | 699.36 | 24.61 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2556 | registered |
| EXP-S14B3-006 | `S14B3-mixed-lenmix_slo_weave` | mixed | `lenmix_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 275052.97/342169.87/355897.69 | 131.49/190.20 | 725.63 | 16.55 | 0.000 | 0.00 | 0.00 | 0.000 | 0.3690 | registered |
| EXP-S14B3-007 | `S14B3-mixed-lenmix_slo_weave-r1` | mixed | `lenmix_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 274954.07/342065.42/355797.96 | 131.48/190.14 | 725.48 | 16.56 | 0.000 | 0.00 | 0.00 | 0.000 | 0.3692 | registered |
| EXP-S14B3-008 | `S14B3-mixed-lenmix_slo_weave-r2` | mixed | `lenmix_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 274954.54/342084.40/355822.31 | 131.46/190.19 | 725.52 | 16.56 | 0.000 | 0.00 | 0.00 | 0.000 | 0.3691 | registered |
| EXP-S14B3-009 | `S14B3-mixed-lenmix_slo_weave-r3` | mixed | `lenmix_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 274985.25/342157.87/355881.27 | 131.48/190.22 | 725.23 | 16.56 | 0.000 | 0.00 | 0.00 | 0.000 | 0.3690 | registered |
| EXP-S14B3-010 | `S14B3-mixed-lenmix_slo_weave-r4` | mixed | `lenmix_slo_weave` | controller.kind=slo_weave, shadow=False, criterion=decode_p90 | - | MEASURED | 274972.00/342116.64/355900.00 | 131.47/190.19 | 724.99 | 16.56 | 0.000 | 0.00 | 0.00 | 0.000 | 0.3695 | registered |
| EXP-S14B3-011 | `S14B3-mixed-lenmix_static` | mixed | `lenmix_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 103374.81/133080.08/142286.03 | 312.24/333.65 | 2075.74 | 25.89 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2756 | registered |
| EXP-S14B3-012 | `S14B3-mixed-lenmix_static-r1` | mixed | `lenmix_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 103528.14/133183.50/142443.02 | 312.52/333.87 | 2075.70 | 25.86 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2751 | registered |
| EXP-S14B3-013 | `S14B3-mixed-lenmix_static-r2` | mixed | `lenmix_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 103551.58/133189.96/142410.40 | 312.47/334.07 | 2086.35 | 25.86 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2752 | registered |
| EXP-S14B3-014 | `S14B3-mixed-lenmix_static-r3` | mixed | `lenmix_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 103572.20/133191.45/142420.22 | 312.51/333.87 | 2075.89 | 25.86 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2756 | registered |
| EXP-S14B3-015 | `S14B3-mixed-lenmix_static-r4` | mixed | `lenmix_static` | controller.kind=threshold, shadow=False, criterion=tpot_p90 | - | MEASURED | 103472.60/133179.45/142466.22 | 312.55/333.97 | 2083.63 | 25.86 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2752 | registered |
| EXP-S14B3-016 | `S14B3-mixed-lenmix_threshold` | mixed | `lenmix_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 108497.20/142245.07/149386.82 | 318.07/348.26 | 681.56 | 24.75 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2593 | registered |
| EXP-S14B3-017 | `S14B3-mixed-lenmix_threshold-r1` | mixed | `lenmix_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 108491.19/143011.22/149443.13 | 318.88/343.75 | 681.23 | 24.73 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2581 | registered |
| EXP-S14B3-018 | `S14B3-mixed-lenmix_threshold-r2` | mixed | `lenmix_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 108766.13/143217.38/149951.26 | 319.19/345.05 | 681.32 | 24.69 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2627 | registered |
| EXP-S14B3-019 | `S14B3-mixed-lenmix_threshold-r3` | mixed | `lenmix_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 108747.26/142659.31/149395.56 | 318.91/343.74 | 682.44 | 24.72 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2590 | registered |
| EXP-S14B3-020 | `S14B3-mixed-lenmix_threshold-r4` | mixed | `lenmix_threshold` | controller.kind=threshold, shadow=False, criterion=decode_p90 | - | MEASURED | 108789.20/144129.03/150543.44 | 319.45/344.91 | 825.30 | 24.68 | 0.000 | 0.00 | 0.00 | 0.000 | 0.2669 | registered |


## 离线协议准备登记 · 2026-10-03 · EXECUTION_SHAPE

状态：OFFLINE_VERIFIED / 真机 NOT_MEASURED。本条不是硬件实验结果，不新增性能数值。
修复预测输入与生产执行形状、队列状态和角色验收；6形状清单源自合法TEST_DATA fixtures，
不构成负载覆盖率。新整步采样、训练冻结、独立验证与三项pilot协议见 NEXT_BATCHES_PREREG.md §13。
真机短验收/3训练进程/3验证进程/pilot均尚未执行；run_id由届时唯一tag分配。
375项联合回归与CPU语义验证见 handoff/optimization_20260929/OFFLINE_EXECUTION_SHAPE_20261003.txt。
旧processed演练文件被误覆盖且原字节未恢复的事件已记入RESEARCH_LOG；raw和历史模型未修改。
