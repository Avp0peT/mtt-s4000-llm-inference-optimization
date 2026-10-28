# 离线收尾交付报告（2026-10-03 第六轮）

**范围声明**：本轮**未启动任何 GPU 性能矩阵**，**未触碰远程主机**，`results/raw/**` 只读。
已通过的固定数值排程与 S01RECHECK 性能产物**原样保留**。
本轮全部产物的证据等级 = `TEST_DATA` / `PLAN_ONLY` —— **不得**升级为 S4000 证据。

---

## 1. 真实决策链演练结果

**入口**：`scripts/dryplan_decision_chain.py`
**产物**：`results/processed/dryplan_decision_chain.json`（28,716 B，CR=0）
**链路**：`load_config → build_arm → plan_step / CostPolicy.decide`

★ 关键点：`CostPolicy.decide()` **内部**对每个候选调 `plan_step()` ⇒
`(batch, tokens_prefilled, kv_len)` 是**调度器推导的**，**不是脚本手选的**。
这正是"预算函数示例"与"真实决策链"的分界。

### 1.1 计划 cell 与四个状态

计划 cell = `prompt_len 1536 / gen_len 32 / concurrency 16 / max_num_seqs 8`。

| 状态 | 为什么在计划负载下**可达** |
| --- | --- |
| `batch_start_prefill` | 连续批调度器在空载时从 waiting 取满批 ⇒ **第 0 步必然**是这种形状 |
| `steady_decode` | 批起步结束、下一波到达前的中段（`conc 16 > max_num_seqs 8` ⇒ 中段**必现**）|
| `mixed_admit` | `conc 16 > max_num_seqs 8` ⇒ 前 8 个解码时，后 8 个已在 waiting 排队 |
| `contended_prefill` | `prompt 1536 > 单步 token 预算` ⇒ 起步请求跨多步 ⇒ prefilling 非空是**常态** |

### 1.2 两臂逐状态结果（**如实**：不要求两臂不同）

| 状态 | cost_single（`prefill_deadline_ms=null`） | cost_dual（`prefill_deadline_ms=2000`）|
| --- | --- | --- |
| `batch_start_prefill` | 4 候选**全部查表缺失** ⇒ 回退阈值控制器，`chosen=null` | **完全相同**（预算是 prefill 分支的唯一适用场景，但**预测不出来**）|
| `steady_decode` | 4 候选全预测 115.53 → `115.5+25.3=140.8 > 120` ⇒ **降级取最省者** `{chunk:512, max_tokens:2048}` | **完全相同** |
| `mixed_admit` | 同上，且 `⚠️ prefill 饥饿` | **完全相同** |
| `contended_prefill` | 2 候选预测 833.65（`833.6+17.9=851.5 > 120`）⇒ 降级；另 2 候选**模型范围外** | **完全相同** |

**汇总**：`cost_single` 4 状态中 查表全缺失 1 / 有动作 3；`cost_dual` **同**。

★ **两臂在本演练中动作完全一致** —— 这是**允许**的结果（验收要求的是
"决策过程真实、结果如实"，**不要求两臂必须不同**）。
★ 但必须看清一致的原因：**双预算的差异本应体现在 `batch_start_prefill`**，
而那一格**恰好是查表缺失、直接回退** ⇒ 双预算**没有被行使过**。
⇒ 因此**不得**把"两臂一致"解读为"双预算无效"（**没测到 ≠ 测出无收益**）。

### 1.3 ★ 主要产出：v7 网格的结构性空洞

**现象**：起步步形状 = `(batch≥2, kv_len=0)`，落在 v7 网格的**结构性空洞**里。

**定性**：这不是模型精度问题，是**网格分层**问题。
对原始标定产物 `costcalib_v7_s1.json` + `costcalib_v7_s2.json`（**222 点**）实测：

```
kv 网格 = [0, 512, 768, 1024, 1536, 2048]
batch   = [1, 2, 3, 4, 5, 6, 7, 8]
pf      = [0, 128, 256, 512, 1024, 2048]

★ 空洞：kv_len=0 这一层**只采了 batch=[1]**；
        batch≥2 的 kv 维**从 512 起** ⇒ (batch≥2, kv=0) **无任何点**
```

**被挡住的起步形状**（本 cell 实际会生成的）：

| batch | pf | kv | 在标定范围内 | lookup | 为什么会出现 |
| --- | --- | --- | --- | --- | --- |
| 2 | 512 | 0 | ✗ | `None` | 起步步：chunk=512，持 2 个序列 |
| 8 | 512 | 0 | ✗ | `None` | 起步步：chunk=512，`max_num_seqs=8` 满批 |
| 2 | 1024 | 0 | ✗ | `None` | 起步步：chunk=1024 |
| 8 | 1024 | 0 | ✗ | `None` | 起步步：chunk=1024 满批 |
| 8 | 1536 | 0 | ✗ | `None` | 起步步：整段 1536 prompt 一次进 |

### 1.4 最小补标定方案

| 项 | 内容 |
| --- | --- |
| **原则** | **另开一轮**（**不得原地改网格** —— 会让 `grid_sha256` 变、血缘断）|
| **补哪一维** | 在 `kv_len=0` 这一层补 `batch ∈ {2,4,8}`（现在只有 1）|
| **补什么点** | 每个 `(batch∈{2,4,8}, kv=0)` 补 `pf ∈ {512,1024,1536}`（512/1024 是真实候选，1536 = 本 cell 整段 prompt）|
| **绑定约束** | 补的点**必须来自本批计划 cell**（prompt 1536 / conc 16 / max_num_seqs 8）的**第一步形状**，而不是又一组一般网格 |
| **新增点数估计** | **约 9 个** |
| **阻断性** | 补完前，本批**起步步一律回退** ⇒ 双预算在起步步上**不可被行使** ⇒ **不得**算成"机制无效" |

> **⇒ 结论：先补标定；不要直接把本演练扩大为性能收益矩阵。**
> （`coverage_gap.consequence_for_expansion` 字段原文即此。）

### 1.5 原四点表改名

旧"四点表"已改称 **预算函数示例**，落盘标记：
- `artifact_role = "budget_function_example"`
- 带 `not_evidence_of` 字段说明它**不**证明什么
- 对应测试：`tests/test_prereg_batch_configs.py::test_artifact_records_the_relaxation`
  断言 `artifact_role == "budget_function_example"`

⇒ **不得**再称"本批机制已触发"。

---

## 2. 三项 pilot 执行清单

**入口**：`scripts/pilot_execution_plan.py`
**产物**：`results/processed/pilot_plan.json`（`evidence_class = PLAN_ONLY`）

### 2.1 配置路径与实际命令

| 项 | 臂 | 矩阵格 | 配置路径 | 控制器 |
| --- | --- | --- | --- | --- |
| P1 | `static_small` | `baseline` | `configs/mthreads/prereg_batch/static_small.yaml` | 关 |
| P2 | `cost_single` | `cost_single` | `configs/mthreads/prereg_batch/cost_single.yaml` | 开（单预算 `prefill_deadline_ms=null`）|
| P3 | `cost_dual` | `cost_dual` | `configs/mthreads/prereg_batch/cost_dual.yaml` | 开（双预算 `prefill_deadline_ms=2000`）|

```bash
python scripts/run_exec_opt_matrix.py \
    --tag PILOT-<日期> \
    --cfg-dir configs/mthreads/prereg_batch \
    --baseline configs/mthreads/prereg_batch/static_small.yaml \
    --arms cost_single cost_dual \
    --repeats 1 \
    --order block \
    --budget-minutes 45 \
    --progress-file /root/pilot_progress.json \
    --summary-file /root/pilot_summary.json
```

### 2.2 ★ 计划期就能核对任务数 = **3**

```
repeats       : 1
计划任务      : 3 项 = 臂数 3 × 轮数 1
明细          : baseline#r0, cost_single#r0, cost_dual#r0
```

**两个默认值陷阱**（不写就中 —— 会把 3 次变成 15 次）：

| 陷阱 | 默认行为 | 后果 | 本轮处置 |
| --- | --- | --- | --- |
| `--repeats` 默认 `None` | 回落到 `--fill-budget` | 跑 3~50 次"填满预算" | **显式 `--repeats 1`** |
| `--baseline` 有默认值 | `cells = ["baseline"] + arms` | 再白加一个 cell | `--baseline` **指向 P1 的 `static_small.yaml`** |

**独立交叉核对**（不靠人读命令）：脚本调用 `run_exec_opt_matrix --dry-plan`
并解析其 JSON，确认 `cells=['baseline','cost_single','cost_dual']`、**3 items**、`repeats=1`。
⇒ 校验通过：`expected_total_runs = n_planned_runs = 3`，`matrix_dryplan_ok = true`。

### 2.3 FZ A/A 状态一致性

```
FZ-1: A（逐步择一）—— 当前实现
FZ-2: A（本步没被服务）—— 当前实现
utterance_ceiling: 只能说『在（逐步择一）语义下 / 在（本步没被服务）口径下』；
                   **不得**把 A 写成『已选定最优语义』
```

⇒ 配置、文档、FZ 状态三处**一致**（`validation_problems = []`）。

---

## 3. 真机短验收命令

**入口**：`scripts/verify_s01_numeric_acceptance.py`
**★ 真机必须显式 `--driver auto`**：

```bash
# 先激活环境；★ 必须 bash -c（sh -c 下 source 是内建命令，会静默失败）
bash -c 'source /root/musa-envs/qwen310/activate_musa.sh && \
  python scripts/verify_s01_numeric_acceptance.py \
    --config-a configs/mthreads/exec_opt/s01_reference.yaml \
    --config-b configs/mthreads/exec_opt/s01_both.yaml \
    --driver auto \
    --json results/processed/s01_numeric-<唯一编号>.json'
```

### 3.1 为什么必须显式写 `--driver auto`

| # | 理由 |
| --- | --- |
| ① | `--driver` **默认值是 `cpu`** ⇒ 真机上漏写会**静默跑 CPU 微型驱动**，产物标 `LOCAL_CPU_MICRO` ⇒ **把本机证据当成真机证据**（最危险的形态）|
| ② | `--driver auto` 会**额外**核验三件事（`_hardware_problems`，判据**拒绝式**）：**实际 MUSA 设备** / **模型 resolved revision** / **优化真的生效** |

**附带守卫**：`--require-real-hw` —— 不写 `--driver auto` 就直接拒绝。

**反例实测**（本轮已跑）：

```
$ python scripts/verify_s01_numeric_acceptance.py ... --require-real-hw --json /tmp/x.json
✗ --require-real-hw 要求 --driver auto，当前是 'cpu'
  默认值是 'cpu' ⇒ 漏写会让本机微型驱动的结果被当成真机证据。
→ EXIT_CODE = 2
→ /tmp/x.json **未生成**（拒绝时不留半成品）
```

### 3.2 唯一产物 + 四类身份绑定

- **禁止覆盖**：`--json` 目标已存在 ⇒ `exit 2`，**不静默覆盖历史证据**（除非显式 `--allow-overwrite`）。
- **4 类 identity**：源码 / 配置 / 模型（resolved revision）/ 环境，各带 `_file_sha256` 或捕获值，
  连同 `evidence_scope` 一起落盘。

### 3.3 ★ 证据等级不升级

| 路径 | 证据等级 | 能证明什么 |
| --- | --- | --- |
| `--driver cpu`（**默认**）| `LOCAL_CPU_MICRO` | **接口与判据语义**；能在本机证明"判据本身会 FAIL" |
| `--driver auto`（真机）| 待真机跑 | 硬件数值行为 |

★ `LOCAL_CPU_MICRO` **不是**硬件数值行为证据 ⇒ **不得**在报告里写成"S4000 已验证"。
本轮**保留现有 CPU 证据等级**，**不**把它升级为 S4000 证据。

---

## 4. 四项小修

### 4.(a) TTFT 两侧有限且 > 0

`scripts/analyze_s01_recheck.py` 新增 `_finite_positive()` + `POSITIVE_LATENCY_METRICS`
（`ttft_p50/p95/p99`）+ `_ttft_positivity_problems()`；`check_valid_numeric` 增第 ⑤ 条
（**独立于主指标**）；汇总增 `nonpositive_control` / `nonpositive_treatment`；控制台增 ③c 打印块。

- **实跑**：`ttft_p50 −4.31%（5/5）`、`ttft_p95 −4.13%`、`ttft_p99 −4.04%`，非正值 **0/0** ✅
- **破坏性对照**：注入 `ttft.p50 = 0.0` ⇒ **rc=1、UNKNOWN**，理由被打印 ✅
- ⚠️ 已登记一个真实盲点：**只靠主指标闸门会让 `ttft_p95 = -12` 混过去** ⇒ 故必须逐指标查。

### 4.(b) 非有限诊断值 → `null` + 说明，严格 JSON

两个脚本都加 `sanitize_json(obj)`：递归把 `NaN/±Inf` 换成 `None`，
并追加 `"<path> = <repr> 不是有限数 ⇒ 已置 \`null\`（非有限值不可作诊断依据）"` 到 `notes`。
写出改为 `json.dumps(..., allow_nan=False)`，notes 落 `_json_notes`。

- **实跑**：注入 `ttft.p95 = Infinity` ⇒ 产物**无 `Infinity`/`NaN` 字样**、
  `strict json.loads` 通过、`_json_notes` 列出 3 条路径 ✅
- **反例**：`allow_nan=False` 直dump 原始值 ⇒ `ValueError` ✅

### 4.(c) 本机必测脚本失败**不得**转 skip

分类 13 处裸 `skipTest`：

| 类型 | 处置 | 行为 |
| --- | --- | --- |
| **本机必测输入**（仓库内配置/脚本/离线产物）| `tests/_tiny_support.require_local_file()` | **`AssertionError`（红）** |
| **外部依赖**（远端产物未回拉 / 部署态 / 环境依赖）| `ExternalArtifactMissing(unittest.SkipTest)` | 显式 skip，**带理由** |

- `tests/test_must_run_not_skipped.py`（**5 例**）扫禁止清单文件里的裸 `skipTest`，
  并**实跑**两个端到端类断言输出无 `" skipped"`。
- **破坏性对照**：把 `results/processed/prereg_states.json` 改名 ⇒ **2 failed（不是 skipped）**；
  还原后恢复全绿 ✅

### 4.(d) 删除"P1 必须落在 (0,1)"规则

- 主行与**同族表述**（开区间 `(0,1)`）**都**已废除 —— 两条本质是**同一个错误**。
- 两条旧规则**以引用块形态留存**（`❌ 旧规则 1` / `❌ 旧规则 2`），保持可追溯，
  且天然避开"当前规则"扫描器。
- 终局判据（**按三臂结果与机制行为**，不按数值区间）：

| 判据 | 内容 |
| --- | --- |
| **E1** | 机制**被真正行使**（该臂的 `budget_kind` 有落盘 / 步级轨迹有变化）|
| **E4** | 身份**完整可信**（源码 / 配置 / 模型 / 环境四类可核）|
| **E3** | 达标率**已登记绝对水平**（分子/分母两个整数一并写出）⇒ **仅作解释** |

★ 三个达标率数值 **都只用于解释**，**都不单独构成"停"或"续"**。

### 4.(e) 预算说明与实际停止逻辑一致

`NEXT_BATCHES_PREREG.md` §10.4 / §11 改为与 `benchmark/exec_opt_plan.py:157 decide_stop` **实际语义**对齐：

| 事实 | 说明 |
| --- | --- |
| **只在整轮边界判定** | `need = avg_item_s * remaining_items_this_pass * 1.15`；`elapsed + need > budget` 才停 |
| **不打断**正在跑的 run | 它是一个**干净的收尾守卫**，**不是硬超时** |
| `avg_item_s is None` ⇒ **不停** | |
| ★ **真实动机** | **保持重复均衡** —— 宁可少跑一整轮，也不要"有的臂 5 次、有的臂 4 次"（不均衡会让配对判据失效）|

- 反例测试 `test_budget_flag_semantics_match_decide_stop` 断言文中含
  `decide_stop` / `不是` / `硬超时` / `整轮` / `不打断` / `均衡`。

---

## 5. 针对性测试与逐项 skip 原因

### 5.1 八个涉及文件的测试结果

```
tests/test_dryplan_decision_chain.py         11 passed
tests/test_pilot_execution_plan.py           15 passed
tests/test_pilot_flow_rules.py               22 passed
tests/test_prereg_batch_configs.py           25 passed
tests/test_verify_s01_numeric_offline.py     21 passed
tests/test_verify_s01_realhw_entry.py        20 passed
tests/test_s01_recheck_analysis.py           49 passed
tests/test_must_run_not_skipped.py            5 passed
──────────────────────────────────────────────────────
合计                                        168 passed, 0 skipped
```

### 5.2 逐项 skip 原因（全仓）

**本轮的八个文件：`0 skipped`** —— 没有把"没跑"混进"通过"。

**其余文件的 skip 全部在 whitelist 内，逐项原因如下**
（来源：`tests/test_must_run_not_skipped.py::_ALLOWED_RAW_SKIP`，**13 条**）：

| 文件 | 允许 skip 的原因 |
| --- | --- |
| `test_batch_evidence_eligibility.py` | 远端批次产物未回拉 ⇒ **外部依赖** |
| `test_batch_lock.py` | `launch_batch.sh` 由部署步骤生成 ⇒ **部署态依赖** |
| `test_check_remote_busy.py` | 需远端进程表 ⇒ **外部依赖** |
| `test_deploy_status.py` | 部署摘要由远端写入 ⇒ **外部依赖** |
| `test_env_fingerprint_identity.py` | 远端环境指纹产物 ⇒ **外部依赖** |
| `test_eval_lookup_deployment_path.py` | 真实评测产物 ⇒ **外部依赖** |
| `test_exec_opt_plan.py` | 本机 bash / 既有批次产物 ⇒ **环境依赖** |
| `test_pull_incremental.py` | 托管 venv 是可选的 ⇒ **环境依赖** |
| `test_run_tests_harness.py` | 需要 pytest 才跑真实链路 ⇒ **环境依赖** |
| `test_s01_recheck_analysis.py` | 已改用 `ExternalArtifactMissing`（本条只是注释残留）|
| `test_source_stamp.py` | 需要可用 git ⇒ **环境依赖** |
| `test_stage_report.py` | 登记在案的**已知缺陷** ⇒ 显式登记 |
| `test_step6_scan_configs.py` | step6 配置由生成器产出 ⇒ **生成态依赖** |

**分类口径**：
- **外部依赖 / 部署态 / 生成态**（8 条）= 产物由远端或部署/生成步骤产出，本机**允许** skip；
- **环境依赖**（4 条）= 需要 git / bash / pytest 等本机可选件；
- **已知缺陷**（1 条）= 显式登记，不是"忘了修"。

★ **禁止清单**（这些文件**不得**出现裸 `skipTest`）：
`tests/test_prereg_batch_configs.py`、`tests/test_verify_s01_numeric_offline.py`
—— 它们的输入**都在仓库内**，缺了就是仓库坏 ⇒ 必须红。

---

## 6. 落盘清单与身份

| 文件 | 类型 | CR |
| --- | --- | --- |
| `scripts/dryplan_decision_chain.py` | 新 | 0 |
| `scripts/pilot_execution_plan.py` | 新 | 0 |
| `scripts/verify_s01_numeric_acceptance.py` | 改 | 0 |
| `scripts/analyze_s01_recheck.py` | 改 | 0 |
| `tests/_tiny_support.py` | 改 | 0 |
| `tests/test_must_run_not_skipped.py` | 新 | 0 |
| `tests/test_s01_recheck_analysis.py` | 改 | 0 |
| `tests/test_prereg_batch_configs.py` | 改 | 0 |
| `tests/test_verify_s01_numeric_offline.py` | 改 | 0 |
| `tests/test_verify_s01_realhw_entry.py` | 改 | 0 |
| `tests/test_pilot_flow_rules.py` | 改 | 0 |
| `NEXT_BATCHES_PREREG.md` | 改（§10/§10.3/§10.4/§11）| 0 |
| `RESEARCH_LOG.md` | 追加本轮小节 | 0 |
| `handoff/optimization_20260929/DEPLOYMENT.json` | 追加 `tool_changes_2026_10_03_fifth_wave` | 0 |
| `handoff/optimization_20260929/STATE.json` | 追加 `session_2026_10_03_offline_fix4` | 0 |
| `results/processed/dryplan_decision_chain.json` | 产物 | 0 |
| `results/processed/pilot_plan.json` | 产物 | 0 |
| `.workbuddy/memory/PLAYBOOK.md`（**仓库外**）| 追加条目 42–44 | 0 |

---

## 7. 未决 / 不做的边界

| 项 | 状态 |
| --- | --- |
| GPU 性能矩阵 | **未启动**（本轮范围外）|
| 远程主机 | **未触碰** |
| `results/raw/**` | **只读**，未改 |
| v7 网格 | **未改**（改会让 `grid_sha256` 变 ⇒ 血缘断）⇒ 补标定须**另开一轮** |
| 双预算是否有效 | **未裁定** —— 本演练中它在唯一适用场景（起步步）上**未被行使** |
| 补标定后做什么 | 先补 `kv=0` 层（≈9 点）⇒ 再考虑是否扩大为性能矩阵 |
| SSH 根因 | 仍为**待确认**（不写成"端口已变化"）|
