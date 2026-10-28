# 优化测试会话（SESSION ENTRY）

> **分工**：`OPTIMIZATION_ROADMAP_SESSIONS.md` 回答"**有哪些优化、各自预期与依据**"
> （路线权威，9 条 A/B/C/D 类），**本文**回答"**这一次优化测试会话怎么开、做什么、
> 怎么算过**"。本文不复制路线图内容；冲突时以路线图、`EXPERIMENTS.md`、
> `results/raw/**` 为准。
>
> 会话编号：`OPT-<两位序号>`——与论文线 `NEXT_SESSION_TASKS.md` 的步骤号**互不干扰**。

---

## 0. 会话登记表（追加式，每次会话收尾时更新）

| 会话 | 日期 | 路线 | 目标 | 结论 | 证据/产物 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| OPT-01 | 2026-09-28 | S0 | 判定 `load_ramp` low 段是否落慢态 + 复核最优工作带 | (1) **未落慢态**（b1 low n=21817、p50 **61.29**、max **65.64** ms，全 19 run 全 4 段**零**步 `fwd_ms ≥ 72.2`）；(2) **c4–c6 不达标**（干净工作点 c4 = **101.26** ms > SLO 100；实测达标档位集合 = **{b1, b2}**） | `scripts/analyze_opt01_s14b2.py`；`results/processed/opt01_s14b2.json`；执行记录见 §2A | done |
| OPT-02 | 2026-09-28 | B1+B2+B3 | 门限形式改造 / 候选集中间点 / 收益负载设计 | (1) **B2 完成**：新增 `gate_mode`（默认 `absolute`）；选"按工作点归一化"并给出论证；★ **附带修掉一个 P0**——`decode_p90` 在持续过载时读出 `0.0` 而非 `None`，实测 3 臂共 **64 次**决策据此在 TTFT 13.8–190 s 时**放大 chunk**；(2) **B3 部分**：候选列表含中间值且被记录 ✅、反向对照 ✅，但"中间点被**选中**"**未证明**；(3) **B1 草案**：`c4–c6` 前提已更正、不可冻结 | `src/ascend_opt/controller.py`；`tests/test_controller_gate_mode.py`；`scripts/check_opt02_acceptance.py`；`NEXT_SESSION_TASKS.md` 文末两节 | done（B1 待评审） |
| OPT-03 | 2026-09-28 | B4 | 小 chunk 可预测性标定（真机） | ★ **能**（128/192/256 三档全部可预测）：键各 **8** 个、本批表对已采 48 形状 **100%** 命中、会话内重复性中位 **0.01%**、**跨会话留出低估 p95 0.07%**（阈值 5%）/ `\|rel\|` p90 **0.06%**；pf192 单独 0.06%/0.06%（与 128/256 同级）。★ 前置更正：任务书写「v6」，实际 **v7 已采 128/256**，**唯一缺口是 192** ⇒ 新开 **v8** 网格补 192（+8 点）并真机复采两会话。⚠️ **384 仍是缺口**（B1 草案 §1.4 点名的三档只关闭两档半）；⚠️ 四套覆盖率不可互换（见 §2B） | `configs/cost_calib/shape_grid_v8.yaml`；`scripts/gen_shape_grid_v8.py`；`scripts/analyze_b4_small_chunk.py`；`results/raw/_probe/20260928/*`；`results/processed/b4_small_chunk_v8.json`；执行记录见 §2B | done |
| OPT-03R | 2026-09-28 | B1 评审 | 裁定 B1 预注册草案的 5 个评审问题 | 五条全部裁定：**① 选 A（双预算）+ 同批单预算版本归因**；**② 0.3–0.7 是目标不是判据**（不达标照跑照记）；**③ 两条静态基线足够、不加 `static_chunk@1024`**；**④ 用闭环补位**（⚠️ 改执行路径 ⇒ 重核 digest）；**⑤ 先跑便宜替代**（1.5–2.5 h 的"双预算 × 两臂"小批）。★ 同时把 OPT-03 结果回灌草案：§1.4 结论 2 加"B 由不可行→可行"更正、§9.1 改"部分闭合"、§9.5 标注"选择规则"成唯一堵点。⚠️ **B1 仍不可冻结**（`384` + §9.5 未闭；且 A+归因 = 40 run ≈16 h > 8 h 上限，须分夜/减重复，留待便宜替代出结果后再裁） | `NEXT_SESSION_TASKS.md` §1.4 / §9 / §10.1–10.2；执行记录见 §2C | done（B1 仍待闭合两项） |
| **S00** | 2026-09-29 | 执行优化线（handoff） | 交接包 `optimization_20260929` 的公共前置：状态核验、阻断项证据、基线冻结 | **本地完成、真机 REMOTE_PENDING**。★ 四列状态：执行优化线 P1/P2a/P3 **均未实现**（不是"测了没收益"）。★ §5.3 **两处已修**：SLOWeave `safety_margin` **重复应用**（1.1602→1.3461）+ 稀疏 `lookup` **连续二分**（`L=1536`/320 ms ⇒ 旧给 0、真值 1024）——均已加反例用例并做**破坏验证**。★ §5.1（双预算忽略受阻 decode，实测 6 个候选）/§5.2（`closed_jittered` 入队早于到达，实测 **16/16**）**只登记未修**（属新策略语义 ⇒ S07 预注册）。★ `model_v8.json` **不存在** ⇒ `pf192` 不在部署表。★ 远端 SSH banner 失败 ⇒ REMOTE_PENDING | `scripts/audit_s00_blockers.py`；`results/processed/s00_audit.json`；`tests/test_slo_weave_sparse_lookup.py`；`tests/test_s00_known_gaps.py`；执行记录见 §5 | done（本地）|
| **S01** | 2026-09-29 | P1 输出层冗余 | 仅末位置 logits + 只采 need_token 行，两个独立开关 | 代码 ✅ / 离线 ✅（12 用例 + 破坏验证 5 failed）/ 真机 ❌ / 结论 ❌。能力**装载期判定**（不支持则显式回退并记录，不静默吞参数）；中间 chunk 的 KV 照常推进 | `benchmark/real_backend.py`；`tests/test_optim_s01_logits_sampling.py`；`configs/mthreads/exec_opt/s01_*.yaml`；记录见 §5.2 | done（本地）|
| **S02** | 2026-09-29 | P2a allocator 生命周期 | `retire_empty_cache` 开关 + `flush_allocator_cache()` | 代码 ✅ / 离线 ✅（7 用例 + 破坏验证 3 failed）/ 真机 ❌ / 结论 ❌。默认 `True` = 原行为；关掉后**槽位仍逐请求释放**（无泄漏） | 同上；`tests/test_optim_s02_allocator.py`；`configs/mthreads/exec_opt/s02_no_empty_cache.yaml`；记录见 §5.3 | done（本地）|
| **S03** | 2026-09-29 | P3 线性投影融合 | QKV 3→1、gate/up 2→1，独立开关 + 形状选择 | 代码 ✅ / 离线 ✅（16 用例 + 破坏验证 1 failed）/ 真机 ❌ / 结论 ❌。★ 实测两条机制事实：**QKV 的窄片必须 contiguous（每层每次 3 次拷贝）**，而 **gate/up 不需要** ⇒ 两者净收益前景不同；`state_dict` 前后逐键相同、可原样回退 | `benchmark/linear_fusion.py`；`scripts/inspect_linear_shapes.py`；`tests/test_optim_s03_linear_fusion.py`；`configs/mthreads/exec_opt/s03_*.yaml`；记录见 §5.4 | done（本地）|
| **EXEC-RUN** | 2026-09-29 | 真机运行脚本 | 让远端**尽可能自动化、跑满、少干预** | 代码 ✅ / 离线 ✅（39 用例 + 破坏验证 5 failed）/ 真机 ❌（远端不可达）。一条本机命令完成 核验→推送→起跑→轮询→回拉；远端**自驱动**（穿插顺序、均衡重复、预算填充、30 s 心跳、STOP 协作停止、断点续跑、按产物判成功）。★ 顺带发现并补上**陈旧批次锁**这个"什么都不跑且不报错"的失效（含安全阀）。★ 本地实跑 `launch_batch.sh` 全链路 + 真实锁产物验证 | `scripts/exec_opt_batch.py`；`scripts/run_exec_opt_matrix.py`；`scripts/clear_stale_batch_lock.py`；`benchmark/exec_opt_plan.py`；`handoff/optimization_20260929/EXEC_OPT_RUNBOOK.md`；记录见 §5.5 | done（本地）|

---

## 1. 共同执行要求（每次优化测试会话都适用）

1. **先查产物再动手**：任何"该优化有多少收益"的主张，先去看已有产物能不能回答；
   能回答就不做新实验。已排除项（路线图 §1）不得重复投入。
2. **只读优先**：`results/raw/**` 只读（不改、不覆盖、不重命名、不补标签）。
   分析脚本产物进 `results/processed/`，图进 `results/figures/`。
3. **探针分离**：不经 harness 的测量产物进 `results/raw/_probe/<日期>/`，
   **不盖四标签、不进结论表**。
4. **血缘纪律**：改执行路径（`benchmark/real_backend.py` 等）或改负载/配置 =
   **新实验条件** ⇒ SLO、控制门限、成本模型、静态基线全部重采，旧批不得并表；
   **所有臂共用同一条执行路径**。
5. **换行符与字段名**：写回源文件一律 `newline="\n"`；
   读产物字段**必须先从产物现读**（本次已核对 telemetry 顶层键，见 §2.2），
   不得凭记忆写字段名。
6. **不可判 ≠ 未发生**：样本不足时记"不可判"并列出所需最小补充测量，
   不得写成否定结论。
7. 收尾输出：变更文件、实际验证命令、结论与证据路径、剩余风险、下一步入口，
   并更新 §0 登记表。

---

## 2. OPT-01（路线 S0）：零成本只读分析

**目标**：用本地已有 `S14B2` 产物回答两个问题，**不产生任何新实验条件**。

1. `load_ramp` 的 **low 段是否落入设备慢态**（部署损失 + 控制器信号污染的潜在来源）；
2. **最优工作带复核**：若负载设计在 `c4–c6`，TPOT 是否落在 SLO 100 ms 内。

### 2.1 前置核查

- 本地 `results/raw/S14B2-load_ramp-ramp_*/telemetry.jsonl` 完整的 run 数
  ⚠️ **分母随回拉进度变化，引用前必须现测**：2026-09-28 13:08 实测 **11/20**
  （`EVIDENCE_INDEX.md` §1.7b），2026-09-28 **20:16 实测 = 19/20**
  （本地 17:51–17:57 又落盘 12 个目录）⇒ **不再写死「11」。**
- 段边界定义取自 `EVIDENCE_INDEX.md` §1.5：`low(0–120s)` → `high(120–210s)`
  → `recover(210–450s)`；负载终点后另设 **`drain(≥450s)`** 段
  （队列排空、无新到达 ⇒ **干净工作点**，本会话新增的判据来源）。
- ~~⚠️ **`t_ms` 基准必须先对齐**：实测 `static` run 首行 `step=0, t_ms=10143.23`
  ⇒ 墙钟起点带了约 10 s 偏移，**不能直接把 `t_ms` 当段边界用**。~~
  **⚠️ 该前提已于 2026-09-28 OPT-01 证伪，勿再按它处理**：
  `arrival_segments` 与 telemetry `t_ms` **是同一时钟**。判据（可复算）：
  按 `anchor=0` 数段内到达数，**19/19 个 run 恰好复现**声明的
  `realized_count=[10,63,19]`；按 `anchor=min(scheduled_arrival_ms)≈10133` 平移
  则得 `[17,57,18]`，**与声明矛盾**（0/19 复现）。
  ⇒ 首行 `t_ms≈10143` 是 `low` 段**首次到达**（`qps=0.08` ⇒ 平均间隔 12.5 s），
  **不是**墙钟偏移。**段边界直接用 `t_ms`**（`stage_report` 本来就是这么做的）。
  证伪代码：`scripts/analyze_opt01_s14b2.py` 的「t_ms 基准」节（每次运行都重做）。
- 先查 `scripts/` 是否已有逐步耗时分析（既有 `scripts/analyze_step14.py`、
  `scripts/autoq/remote_recovery_s14b2.py`）——**优先复用/扩展，不新写重复实现**。
  OPT-01 实际复用：`decompose_step_overhead.parse_note`（`fwd_ms` 唯一解析实现）、
  `ascend_opt.stage_report.build_stage_specs`（段边界唯一来源）、
  `ascend_opt.run_ids.parse_run_id`。
- ⚠️ **`running` 不是分档键**（OPT-01 现测）：纯 decode 步里 `fwd_ms` **只随
  `note.decode` 变**（`(running=8, decode=1)` 的中位 61.41 ≈ 档位 1 的 61.33），
  且 `running ≠ decode` 的步占 **20.6%（14185/68811）** ⇒ 按 `running` 分档会把
  这批步贴错标签。**两侧都报，判据用 `decode`**（= 顶层 `scheduled_tokens`）。

### 2.2 已核对的字段（本次现读，勿凭记忆改写）

telemetry 顶层键（实测 `S14B2-load_ramp-ramp_static`）：
`chunk_size` / `chunk_size_requested` / `max_batched_tokens` /
`max_batched_tokens_requested` / `mem_mb` / **`note`** / `prefill_budget_effective` /
`queue_depth` / **`running`** / `scheduled_tokens` / `step` / **`t_ms`** /
`throughput_tok_s` / `tpot_ms` / `ttft_ms` / `util_pct`。

⚠️ **`fwd_ms` 不在顶层**——它在 `note` 文本里（形如
`prefill_tok=… decode=… forwards=… fwd_ms=… pack_ms=…`），必须正则提取。
⚠️ `tpot_ms` / `util_pct` / `mem_mb` **可以是 `null`**（不可测），聚合前先过滤，
不得把 `null` 当 0。

### 2.3 步骤 1：逐段耗时分布（判慢态）

> ⚠️ **OPT-01（2026-09-28）执行后对本节的修正**：分档键改为 `note.decode`；
> 绝对快/慢带**只对 b1 有效**。原文（按 `running` 分档、把 76–82 ms 当所有档的
> 慢态线）会产出假判定 —— 见 §2.1 末两条与 `RESEARCH_LOG.md` 当日条目。

1. 按 **`decode`** 分档（= 该步真实前向的序列数 = 顶层 `scheduled_tokens`），
   每档再按段（low / high / recover / **drain**）分组；`running` 分布**并列报出**；
2. 每组统计**纯 decode**（`note.prefill_tok == 0`）`fwd_ms` 的
   n / p10 / p25 / p50 / p75 / p90 / min / max，**必须报 n**；
3. 判据（对照 `COST_PROFILE_S4000.md` §2/§4）：
   - **b1**（唯一有独立快/慢标定的档）：快态上界 68.86 ms（=59.88×1.15，
     容差理由见脚本常量注释）｜慢态下界 72.20 ms（=76×0.95）；
     慢态特征 = 中位 ≥ 慢态下界，或出现 ≥ 慢态下界的样本 / 连续步；
   - **其余档位**（b2 的正常值本就是 74–81 ms）：**不得套 b1 的绝对带**
     （会误判成"慢态"），只判**形状**：`p90/p10 ≤ 1.02` ⇒ 单态；否则疑双峰；
4. 三态结论：**落慢态 / 未落 / 不可判**（样本不足）。不可判不得写成"未落"。

### 2.4 步骤 2：最优工作带复核

1. 统计各段实测 **`decode`** 分布（并列出 `running` 分布，两者不同）；
2. 用实测 `tpot_ms`（非快态表）按 `decode` 分档对照 SLO=100 ms，
   回答"c4–c6 设计下 TPOT 是否可达"；
   ★ **优先用 `drain` 段**（负载已结束、队列排空、跳过率 0%）的档位值 ——
   那是**干净工作点**；`high`/`recover` 的档位值受积压污染；
3. ⚠️ 快态标定表（b1 59.9 / b2 74.3 / b4 83.3 / b8 109.2）是**性能下界**，
   与服务协议实测不可直接等同——只作形状参考，判据用实测 `tpot_ms`；
4. ⚠️ **不得报"最大可行批"这个单个数**：实测 batch 成本曲线**非单调**
   （实测 b3 105.8 ms > b4 89.6 ms `fwd_ms`），"最大档"在非单调曲线下无定义；
   只报**达标档位集合**；
5. ⚠️ 交叉核对：`requests.jsonl` 的逐请求 `tpot_ms` 是 SLO 自己的字段。
   `low` 段两口径应接近（实测 63.23 vs 63.15 ⇒ 口径通过核验）；
   过载段请求级**更高** ⇒ 步级 `tpot_ms` 是**下界**（方向对结论安全）。

### 2.5 验收

- 两项各有明确判定（**是 / 否 / 不可判** + 依据数字 + n）；
- 脚本落 `scripts/`（或复用既有脚本的扩展），产物落 `results/processed/`；
- 结论写进 `RESEARCH_LOG.md` 当日条目并更新本文 §0 登记表；
- **未修改** `results/raw/**`、未改 `benchmark/real_backend.py`、未产生新实验条件。

### 2.6 停手条件

- 若某档位 n 过小（分档后无法支撑分位数）⇒ 记"不可判" + 列出所需最小补充
  测量（哪台机器、什么负载、多少样本），**不得**用相邻档位外推填补；
- 若发现既有两个脚本即可完整回答 ⇒ 不新写脚本，直接出结论并登记。

---

## 2A. OPT-01 执行记录（2026-09-28，零成本只读）

> `OPT-01` 已完成 ⇒ 本节是**执行结果**，不是待办。两项判定：

| # | 问题 | 判定 | 依据数字（n） |
| --- | --- | --- | --- |
| 1 | `load_ramp` 的 low 段是否落入设备慢态 | **否（未落慢态）** | low 段 b1 纯 decode `fwd_ms`：n=**21817**、p50 **61.29**、p90 61.47、max **65.64** ms（慢态下界 76）；全部 19 run × 4 段**零**步 ≥72.2 ms；各臂 low-b1 中位 61.28–61.31（跨臂一致） |
| 2 | 负载设计在 c4–c6 时 TPOT 是否落 SLO 100 ms 内 | **否** | 干净工作点（`drain` 段，队列 p50=0、跳过率 0.0%）：b1 **63.34** / b2 **83.96** / b3 **113.74** / b4 **101.26** ms；c5 118.4 / c6 123.9 ⇒ 达标档位集合 = **{b1, b2}**，b4 已超 +1.26 ms |

**变更文件**

| 文件 | 变更 |
| --- | --- |
| `scripts/analyze_opt01_s14b2.py` | **新增**（只读分析；复用 `parse_note` / `build_stage_specs` / `parse_run_id`） |
| `results/processed/opt01_s14b2.json` | **新增**产物（`results/processed/` 在 `.gitignore` 内，不入库） |
| `OPTIMIZATION_TEST_SESSION.md` | §0 登记表、§2.1（证伪 10 s 偏移前提）、§2.3/§2.4（分档键与判据修正）、本节 |
| `RESEARCH_LOG.md` | 新增 2026-09-28 OPT-01 条目 |

**实际验证命令**

```bash
"C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe" \
  scripts/analyze_opt01_s14b2.py --tag S14B2 --json results/processed/opt01_s14b2.json
"C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe" scripts/run_tests.py
git status --short -- results/raw benchmark/real_backend.py    # 两者均空 = 未触碰
```

- 脚本 rc=**1**（分母 19/20，`threshold-r1` 的 `telemetry.jsonl` 尾部截断）
  ——**非零 ≠ 方法被否证**，只表示样本不齐。
- `scripts/run_tests.py`：**1357 用例 / 真正执行 1323 / 跳过 34 / 失败 0 / 错误 0**。
- `results/raw/**` 与 `benchmark/real_backend.py` 的 `git status` **均为空**。

**剩余风险（不掩盖）**

1. ⚠️ **实测 batch 成本曲线非单调**（b3 105.8 > b4 89.6 ms），
   ~~**成因未定位**，登记待查~~ **⇒ 2026-09-28 OPT-02 更正：形状已由第二条独立来源确认**——
   成本模型 `model_v6`（同一标定血缘、非本报表口径）**独立复现同一凸起**
   （模型 b1..b4 = 62.34 / 79.25 / **105.93** / **88.62**）。
   ⇒ 它**不是**采样噪声、也不是报表口径错误，而是**可信的曲线形状**；
   同时口径一致核对后模型在纯 decode 档上准确到 **≤1.5%**
   （模型前向 + 实测 `pack_ms` vs 实测 `tpot`：−0.41 / +0.52 / +0.27 / +1.44 ms）。
   **机制仍未定位**（可能与 batch 分组/padding 有关），但后果不变且更硬：
   路线图 §0「SLO 可行最大批 ≈ b6」所依赖的**单调内插不成立**；
   任何"c6 能达标"的算术都**不可用**，必须实测。详见 `NEXT_SESSION_TASKS.md` 的
   「OPT-02 完成记录」§二 与「B1 预注册草案」§1.5。
2. ⚠️ **结论 2 只覆盖"瞬时前向批 = c"这一读法**。它**不是**"并发 c 的稳态"
   的完整等价物：本批没有 c4–c6 的**稳态**负载格（b≥3 只出现在 `high`
   与 `recover` 的积压段，`drain` 是唯一的干净点，但 c4 只有 536 步）。
   要写成论文级结论，需 B4/收益实验的**真机稳态格**。
3. ⚠️ **口径宽度**：步级 `tpot_ms` 在过载段**低估**请求级 TPOT
   （`high` 172.4 vs 步级 b8 145.4），且 `skipping` 不是充分解释
   （b8 跳过率 0.0% 而仍偏低）。方向安全（结论 2 因此**保守**），但两口径
   **不得并列当同一分布**。
4. ⚠️ **结论 1 是"有限观察"**：只覆盖 `S14B2` 一个负载 + 一次会话 + 一台机器。
   它**不构成**"该设备不会进慢态"；`COST_PROFILE` §4 的慢态现象来自
   **另一种协议**（sync-per-call / 低占空比），本次未复现**不代表**它不存在。
5. ⚠️ `slo_weave-r4` 虽产物完整，但属**已知控制轨迹分岔**（步数 4201 vs 4886），
   本会话把它**计入分母**（19）但**未单列**；引用跨重复统计时须沿用
   `EVIDENCE_INDEX.md` §1.6 的纪律（不得剔除、不得读作收益）。

**下一步入口**：`OPT-02`（B1+B2+B3 离线准备，见 §3）。本会话**未**开始 OPT-02。

---

## 2B. OPT-03 执行记录（2026-09-28，B4 小 chunk 标定，真机）

**任务书问题（逐字）**：「小 chunk 能否被可靠预测（lookup 有键且留出误差达标）？」

### 2B.1 前置核查与一处更正

- 远程 S4000 可达（`autodl-container-cf99469eb6-b877706e`，GPU 空闲）；
  `check_cpu_stability.py 7 1.0` → **rc=0（0 稳定）**，稳健 CV **0.007%**、PSI 全 0。
- ★ **更正任务书前提**：任务书写「补 **v6** 网格的 128/192/256」，实测 **v7 已存在**
  且其 `mixed_small_chunk` 组**已采 128/256**（`costcalib_v7_s{1,2}.json`）。
  `model_v7.json` 键数：pf128 **8** / pf256 **8** / **pf192 = 0**
  ⇒ **128/256 无需重采，唯一缺口是 192。**

### 2B.2 做法

新开 **v8** 网格（`protocol_version: cost-calib-v8`）——**不原地改 v7**：
① 网格纪律要求"加密采样点必须另开一轮"；② 原地改会使 `grid_sha256` 失配、血缘断。
**v8 只改一处**：`mixed_small_chunk.prefill_tok` `[128,256] → [128,192,256]`（+8 点 ⇒ 24 点）。
变更理由**采数前**写入文件头；不变量由 `scripts/gen_shape_grid_v8.py --check` 钉住。

真机两会话（快态协议 + 会话内对照，`COST_PROFILE` §6）：
`20260928T221000Z-autodl-c-v8s1`、`20260928T222500Z-autodl-c-v8s2`，
各 **24/24 点**、`complete: true`、`grid_sha256 = 3473d023e269706f`。

### 2B.3 ★ 结论：能

| 判据 | v8 实测 | 阈值 | 判定 |
| --- | --- | --- | --- |
| 键完整性 | pf128 / **pf192** / pf256 各 **8** 键 | 三档齐 | ✅ |
| 本批表对已采 48 形状命中 | **100%**（48/48） | 100% | ✅ |
| 会话内重复性（同形状跨会话） | 中位 **0.01%** / 最大 0.07% | ≤5% | ✅ |
| ★ 跨会话留出·低估 p95 | **0.07%** | ≤5% | ✅ |
| ★ 跨会话留出·`\|rel\|` p90 | **0.06%** | ≤10% | ✅ |
| pf192 单独 | **0.06%** / 0.06% | 同上 | ✅ |

- ★ **两条可证伪预期均成立**（采数前登记、采完核对）：
  ① **pf 维单调** —— 192 落在 128 与 256 之间（b4: 183.52/**229.36**/277.15；
  b8: 303.20/**431.56**/507.76）；② **kv 不敏感** —— 同 batch+pf 下 kv 1024→2048
  中位 `|Δ|` **0.02%**（故"同 batch+pf 补一个键"安全）。

### 2B.4 ⚠️ 四套覆盖率**不可互换**（本会话最容易读错的地方）

| 口径 | 覆盖率 | 它回答什么 |
| --- | --- | --- |
| 本批表对**已采形状** | **100%** | 运行时落在已采样形状上，必然命中 |
| **按形状留出**（秩保证 1/3） | **0.00%** | 对**全新形状**无键 ⇒ 回退（是泛化考验，**不是**"不可预测"）|
| **跨会话留出** | **100%** | 换会话、不换形状 ⇒ 键全等；报精度与会话漂移 |
| 全形状混算（`lookup_deploy_eval_v7.json`） | 6.67% | 被大 prefill 留出形状拖低，**与上三者都不同** |

且**按形状留出**下用"同 batch+pf 最近邻补键"（非部署路径，仅诊断）误差降到 **0.09%**。

### 2B.5 口径边界（不得外推）

- 本结论**只对 pf ∈ {128,192,256}**、**只对已采这 24 个形状**成立；
  **384 仍未采样**（B1 草案 §1.4 点名的 `{128,256,384}` 只关闭了前两档 +192）。
- **不触发**任务书的附带条款「若不能则 B1 须调整小 chunk 候选」——答案是"能"。
  用户裁定：**保持现状**，B1 候选集留待 B1 自己的 5 个评审问题裁定。

### 2B.6 变更与验证

- 新增：`configs/cost_calib/shape_grid_v8.yaml`、`scripts/gen_shape_grid_v8.py`、
  `scripts/analyze_b4_small_chunk.py`、`scripts/_b4_inspect.py`；
  `results/raw/_probe/20260928/costcalib_v8_s{1,2}.json`；
  `results/processed/{costcalib_eval_v8,b4_small_chunk_v8,b4_small_chunk_v7}.json`。
- 验证：v8 网格校验 **0 问题 / 24 点**；`gen_shape_grid_v8.py --check` 通过；
  两会话 `complete: true`；回拉 **md5 远端==本地**、CR=0；
  `analyze_b4_small_chunk.py` 判定"能"；**两条否定测试**均按预期失败
  （空小 chunk ⇒ rc=2；假 `--artifacts` ⇒ 告警且血缘为 `None`）；
  全量 `pytest` **1355 passed / 39 skipped / 0 failed**；
  `git diff --stat -- results/raw/` **为空**（只读区未被触碰）。

### 2B.7 剩余风险

1. **384 仍未标定**（B1 草案 §1.4 的第三档）⇒ 若 B1 要用 384 作为候选，须再补一轮。
2. 两会话是**同一网格的两次遍历** ⇒ 跨会话留出**不考验新形状**；"新形状"能力只由
   "按形状留出 + 最近邻诊断"支持（0.09%），**不是**部署路径数字。
3. 本批是**探针路径**（无 harness 盖章、无四标签）⇒ 结论只进 `RESEARCH_LOG`，
   **不得**进 `EXPERIMENTS.md` 结论表，也不得据此直接挑最终参数。
4. `model_v8.json` **尚未导出**（v8 数据仅存在于 eval 层）⇒ 若要让 192 真正进入
   部署查表，须走 `export_cost_model.py` 并配套重跑验证（`PENDING_MODELS` 守卫）。
5. 跨硬件/模型/负载须重采（`COST_PROFILE` §6 绝对值漂移 1.3–4%）。

### 2B.8 下一步

- **B1 仍不可冻结**（§10 五个评审问题待裁定；384 仍为缺口）。
- ⛔ **不得**开始 `OPT-04`。

（⇒ 上述"待裁定"已在同会话内完成，见 §2C。）

---

## 2C. OPT-03R 执行记录（B1 评审问题裁定，2026-09-28）

### 2C.1 做了什么

裁定 B1 预注册草案（`NEXT_SESSION_TASKS.md` 文末）的 5 个评审问题，并把 OPT-03（B4）
的结果**回灌**草案中因此过时的三处。**未跑真机、未改 `results/raw/**`、未改 `src/`。**

### 2C.2 ★ 裁定前的必要更正（否则会按过期信息裁定）

| 位置 | 原文（OPT-02 时） | 现状（OPT-03 后） |
| --- | --- | --- |
| §1.4 结论 2 | 方案 B（压小 chunk）**做不了** | ★ **改为"可行"**——128/192/256 已可预测；但 `384` 与 §9.5 未闭 |
| §9.1 | 缺口 3 **未闭合**（128/256/384 全部不可判） | ★ **部分闭合**——`pf ∈ {128,192,256}` 有键；`384` 仍缺 |
| §9.5 | 中间点"被选中"未证明 | ★ 补注：候选 192 已可预测 ⇒ **堵点只剩"选择规则"**，不是"没有可用小 chunk" |

### 2C.3 ★ 五条裁定

| # | 裁定 | 关键后果 |
| --- | --- | --- |
| 1 | **A（双预算）+ 同批跑单预算版本归因** | 机时翻倍至 **40 run ≈16 h** ⇒ 单夜放不下（冲突见 2C.4） |
| 2 | **0.3–0.7 是"目标"，不是判据** | 只写死"可达性前置检查"；不达标照跑照记；**禁止**朝区间调负载（制造 oracle） |
| 3 | **两条基线足够，不加 `static_chunk@1024`** | 历史锚改为**引用** S14B2 数字并注明**不可并表**，不重跑 |
| 4 | **闭环补位** | ⚠️ 改执行路径 = 红线 ⇒ 预注册须写明并重核 `git_commit`/`digest` |
| 5 | **先跑便宜替代**（1.5–2.5 h） | 「双预算 × {cost, threshold}」小批 ⇒ 产出**正是 9.2 与 9.5 的闭合证据** |

### 2C.4 ⚠️ 自查出的机时冲突（未掩盖）

§7 原写「20 run ≈8.0 h，刚好一夜、无余量」；Q1 裁定 A+归因后为 **40 run ≈16 h**
⇒ **必须显式二选一**：**(a) 重复降到 3**（24 run ≈9.6 h，仍略超）或
**(b) 分两夜且明确两夜不得并表**。**不得**不写就按 16 h 起跑。
⇒ 该项**留待便宜替代出结果后**与 9.2/9.5 一起再裁（那时才知是否值得 16 h）。

### 2C.5 变更与验证

- 变更：`NEXT_SESSION_TASKS.md`（§1.4 加更正框、§9 表 4 行改写、§10 加 10.1/10.2）；
  本文件（§0 加 `OPT-03R` 行、§2C 本节）。
- 验证：`tr -cd '\r' | wc -c` = **0**（两份均 LF）；`grep` 命中五条裁定各 ≥1 次；
  §10.1/§10.2 锚点在位（行 5566/5576）。

### 2C.6 下一步

- **可立即执行**：便宜替代（Q5），入口 = 写 `slo_reach` 负载 + 双预算 deadline 的
  **纯逻辑层**实现与单测（不碰真机），再起 1.5–2.5 h 小批。
- **B1 仍不可冻结**：`384`（Q1/§9.1）+ §9.5 两处未闭。
- ⛔ **不得**开始 `OPT-04`。

---

## 3. 后续会话索引（不复制内容，指向路线图）

| 会话 | 路线 | 入口 |
| --- | --- | --- |
| OPT-02 | B1+B2+B3 离线准备 | 路线图 §5「B1+B2+B3」 |
| OPT-03 | B4 小 chunk 标定（真机） | 路线图 §5「B4」 |
| OPT-04 | 收益实验（P29，真机） | 路线图 §5「收益实验」 |
| OPT-05 | A1 Host 重叠（部署线） | 路线图 §5「A1」 |
| OPT-06 | A2 图捕获/锁频探查 | 路线图 §5「A2」 |
| OPT-07 | D1 W8A8 探针（仅部署需求时） | 路线图 §5「D1」 |

依赖顺序见路线图 §4；论文线（OPT-02→04）与部署线（OPT-05/06/07）解耦，
共享 OPT-01 这个零成本起点。

---

## 4. 启动语（复制即用）

```text
项目目录：C:\Users\13944\WorkBuddy\st\ascend-llm-inference-optimization
请读取 OPTIMIZATION_TEST_SESSION.md，执行 OPT-01，先遵守其「共同执行要求」。
先核查本地 S14B2 产物在位情况与已有分析脚本，能复用就不新写。
完成本会话后更新该文件的会话登记表与 RESEARCH_LOG.md，列出变更、验证命令、
结论与证据路径、剩余风险。不要自动开始 OPT-02。
```

下一会话把 `OPT-01` 换成 `OPT-02`，依此类推。

---

## 5. 执行优化线（handoff 包 `optimization_20260929`，编号 `S00`–`S11`）

> **与本文 §1–§4 的关系**：§1–§4 是**论文线**（控制器/成本模型，编号 `OPT-xx`）；
> 本节是交接包带来的**执行优化线**（输出层/allocator/线性融合等，编号 `S00`–`S11`）。
> 两条线的**纪律相同**（§1 共同执行要求全部适用），但**编号互不干扰**：
> `S00` ≠ `OPT-04`，`S07` ≠ `OPT-07`。
> 路线权威：`handoff/optimization_20260929/01_PROJECT_PLAN.md`（阶段/依赖/验收）+
> `REFERENCE_8_PLANS.md`（8 个方案 P1–P8 的技术依据）。

### 5.1 S00 执行记录（2026-09-29，本地）

**目标**：建立可信起点 —— 状态核验、阻断项证据、基线冻结。**不**开始 P1/P2a/P3。

**四列状态（来源 = 源码 + `EXPERIMENTS.md` + `EVIDENCE_INDEX.md`）**

| 机制 | 已实现 | 离线已验 | 真机已测 | 可作结论 |
| --- | --- | --- | --- | --- |
| **P1** 末位置 logits / 按需采样（S01） | ❌ 无 | ❌ | ❌ | ❌ |
| **P2a** retire 与 allocator 清理分离（S02） | ❌ `retire()` 每次摘槽即 `musa.empty_cache()`（`benchmark/real_backend.py:1374-1378`） | ❌ | ❌ | ❌ |
| **P3** QKV / gate-up 融合（S03） | ❌ | ❌ | ❌ | ❌ |
| **P4/P5/P6/P8**（S04–S09） | ❌ | ❌ | ❌ | ❌ |
| **P7** 执行形状感知调度（S07） | 🟡 `cost`/`slo_weave` 臂 + 双预算开关（未提交） | 🟡 单测覆盖 | 🟡 有四臂产物，`adaptive` 不可用 | ❌ |

⇒ **执行优化线三项优先方向目前都只是"未实现"**，不是"测了没收益"。

**§5 阻断项核验（`01_PROJECT_PLAN.md` §5）**

| # | 项 | 实测 | 处理 |
| --- | --- | --- | --- |
| §5.1 | 双预算忽略受阻 decode | 仍存在（12×2048 局面 ⇒ **6** 个候选被 prefill 预算放行） | **只登记**（新策略语义 ⇒ S07 预注册）；`tests/test_s00_known_gaps.py` |
| §5.2 | `closed_jittered` 未来 release | 仍存在（**16/16** 请求 `enqueue < arrival`） | **只登记**（真机"等待"= GPU 空转；须预注册裁定） |
| §5.3a | SLOWeave `safety_margin` 重复应用 | ★ 确认（1.1602 → **1.3461**） | ✅ **已修** |
| §5.3b | 稀疏 `lookup` 连续二分 | ★ 确认（`L=1536`/320 ms ⇒ 旧给 **0**，真值 1024） | ✅ **已修** |

⚠️ **§5.3 的修复改变了 `slo_weave` 臂的判据倍率** ⇒ 既有含该臂的产物
（`EXP-S14B3-006..010`）**是其修复前口径**，**不得与修复后并表**。

**变更文件**

| 文件 | 变更 |
| --- | --- |
| `src/ascend_opt/slo_weave.py` | 修 `_predict_ms`（`apply_margin=False`）；`decide()` 加 `candidates` 参数 + 离散枚举路径；docstring 说明两条路径 |
| `src/ascend_opt/cost_model.py` | 新增 `lookup_prefill_candidates()`（稀疏表的"有键形状"查询） |
| `tests/test_slo_weave_sparse_lookup.py` | **新增** 13 用例（含"旧二分复现缺陷"的反例对照） |
| `tests/test_s00_known_gaps.py` | **新增** 4 用例（特征化 §5.1/§5.2 两个缺口） |
| `scripts/audit_s00_blockers.py` | **新增**（只读证据脚本） |
| `results/processed/s00_audit.json` | **新增**产物（`results/processed/` 不入库） |

**基线协议（冻结）**

- 权威平台 = **远程 MTT S4000（MUSA）**；本机 `local` **永不**产生 `MEASURED`。
- 平台基线 `configs/mthreads/s4000_fp16.yaml`：FP16 / 单卡 / `graph_mode: eager` /
  `tensor_parallel_size: 1` / `max_num_seqs: 32`；分块、缓存、自适应**全关**。
- 外部 SLO = `ttft_ms 2000` + **逐请求** `tpot_ms 100`。
  ⚠️ 配置里的 `controller.target_tpot_ms` 是**控制器内部目标**，**不是**外部 SLO，务必分开写。
- ⚠️ **未 pin**：基线配置写 `runtime.model_revision: "main"`，**不是**交接记录的
  `a09a35458c702b33eeacc393d103063234e8bc28` ⇒ 执行优化线须用**新配置**钉死 revision
  （原地改历史配置 = 新实验条件）。
- **vLLM-MUSA 成熟引擎基线**：本仓库**无**该基线的配置/脚本/产物（`SOURCES.md` 里
  相关条目均为 `unverified`）⇒ 兼容性调查**须在远程**采样实际 driver/MUSA SDK/torch_musa
  版本后进行，**未证可用前不安装新驱动**。状态 = `REMOTE_PENDING`。
- ⚠️ `--dry-run` 只证"链路可跑"，**不能**验过载 / SLO / 任何判据逻辑。

**验证命令**

```bash
python -m pytest tests/test_slo_weave.py tests/test_slo_weave_sparse_lookup.py \
                 tests/test_s00_known_gaps.py -q
python scripts/audit_s00_blockers.py --json results/processed/s00_audit.json
python -m pytest tests/ -q
```

- 定向 **31 passed**（27 + 4）；**破坏验证**：还原旧行为 ⇒ **5 failed**，恢复 ⇒ 全绿。
- **全量回归**：`PYTHONPATH= python -m pytest tests/ -q`
  ⇒ **1433 passed / 1 skipped / 746 subtests / 0 failed / EXIT=0**（11:35）。
  ⚠️ 必须**清 `PYTHONPATH`**（否则 env 垫片造成**假回归**）；⚠️ 而 1378→1433 的 +55
  可解释为 **+17（本会话新增）+ 38（此前因解释器缺 torch 被 skip，现实跑）** ⇒ 无回归。
- `results/raw/**`（已跟踪文件）与 `benchmark/{real_backend,run_benchmark}.py`
  **均未被触碰** —— 后者工树里的 ` M` 是**前一会话**的未提交改动
  （mtime 2026-09-28 23:2x < 本会话开始 2026-09-29 19:43）。
- 未产生任何新实验条件；未跑真机。

**剩余风险**

1. §5.1/§5.2 **只有证据、没有修复**（属新策略语义，须 S07 预注册）。
2. §5.3 修复**未在真机验证**，且会改变 `slo_weave` 臂真机行为。
3. §5.2 证据来自 dry-run 模拟器 ⇒ 证明的是**代码语义**，不是真机现象。
4. 远端不可达（SSH banner 失败）⇒ 真机项全部 `REMOTE_PENDING`。
5. 基线配置写 `model_revision: main`（**未 pin**）⇒ 新线须用**新配置**钉死 revision。

**下一步**：S01（P1 输出层冗余），入口 = `REFERENCE_8_PLANS.md` §P1 +
`benchmark/real_backend.py` 前向入口（约 1214 行）。

### 5.2 S01 执行记录（2026-09-29，本地）

**目标**：利用 Qwen 大词表结构，去掉不需要的输出计算。两个**可独立开关**的机制。

**实现**（`benchmark/real_backend.py` 的 `TransformerDriver`，默认全关）

| 开关 | 作用 | 落盘字段 |
| --- | --- | --- |
| `logits_last_only` | 前向传 `num_logits_to_keep=1`（只算最后位置） | `execution_optimizations.logits_last_only.{requested,effective,unsupported_reason,n_calls_*}` |
| `sample_needed_only` | 只对 `need_token=True` 的行采样 / D2H，按**行下标**映射回请求 | `…sample_needed_only.{n_sampled_rows,n_sampling_rows_skipped}` |

- ⚠️ **可用性在装载期判定**（`_detect_num_logits_to_keep`）：签名显式声明 ⇒ 用；
  签名不接受 ⇒ **不传**；签名带 `**kwargs` 判不了 ⇒ 用一次极小前向探测。
  三条路径都写 `effective` 与 `unsupported_reason` ⇒ **不静默吞参数**。
  探测发生在计时窗口之外，且不计入 `n_forwards`。
- 中间 chunk 即使 `need_token=False`，**KV 与 `past_len` 照常推进**（跳过的是采样，不是前向）。

**验收（本地）**

| 判据 | 结果 |
| --- | --- |
| 末位置 logits ≡ 全位置 logits 的最后一列 | ✅ `allclose`（atol 1e-5） |
| KV / `past_len` 与关闭时逐项相同 | ✅ |
| 多步 greedy 生成 token 序列一致 | ✅ |
| `need_token` 全真 / 全假 / 混合三种场景 | ✅ 全假的**一次都不采样**；混合的**行映射不错位** |
| 两开关组合 = 基线 | ✅ 含 decode1 负对照 |
| 不支持的模型 ⇒ **显式回退且记录** | ✅ `effective=False` + reason 含参数名，且 token 仍正确 |
| **破坏验证** | ✅ 还原旧行为 ⇒ **5 failed**，恢复 ⇒ 全绿 |

**⚠️ 性能未测**（TTFT / prefill 时间 / 显存）：本地无 MUSA ⇒ `REMOTE_PENDING`。
**不得**把"输出位置数少了 k 倍"写成"端到端快 k 倍" —— 减少的是输出层重复投影，
不是整个 Transformer 的计算量。

### 5.3 S02 执行记录（2026-09-29，本地）

**目标**：把"逐请求删 slot"与"清 allocator 缓存"分离，验证后者不必要地压在周转路径上。

**实现**：`TransformerDriver.retire()` 保留逐请求摘槽（**必需**，否则驻留槽位累积到 OOM），
把 `musa.empty_cache()` 变成受开关控制的行为（`retire_empty_cache`，**默认 `True` = 原行为**）；
新增 `flush_allocator_cache()` 供 run 边界显式调用，**返回值表示是否真的调用**（不虚报）。
落盘：`execution_optimizations.retire_empty_cache.{requested_at_retire,n_empty_cache_calls,n_suppressed}`。

**验收（本地）**

| 判据 | 结果 |
| --- | --- |
| 默认：每次真摘槽位就清一次缓存 | ✅（假 `torch.musa` 替身计数） |
| 关掉后：槽位数归零（**无活跃引用泄漏**），`empty_cache` 一次不调 | ✅ |
| 高周转 10 轮「进 1 出 1」⇒ `empty_cache` 次数不随请求数增长 | ✅ |
| `flush_allocator_cache()`：可用时 `True`；`torch.musa` 缺失时 `False` | ✅ |
| 只改"何时清缓存"不改变任何生成结果 | ✅ |
| **破坏验证** | ✅ 无视开关 ⇒ **3 failed** |

**⚠️ 未测**：`allocated` vs `reserved`、分配重试、OOM、请求边界 ITL、真机端到端吞吐
⇒ `REMOTE_PENDING`。**`reserved` 变高本身不算失败**（那是缓存复用的预期表现），
判据是"活跃内存不无限增长 + 无 OOM"。
**⚠️ 未接线**：本仓库**没有**内存压力控制器 ⇒ "如何区分 allocated 与可复用 reserved"
目前只能靠 `sample_device()` 的直接读数，**不存在**自动判据。记为待办，不假装已有。

### 5.4 S03 执行记录（2026-09-29，本地）

**目标**：验证小批量 decode 与 prefill 能否从 QKV / gate-up 融合中受益。

**实现**：新增 **`benchmark/linear_fusion.py`**（纯逻辑，顶层不 import torch）。

- 权重在同一装载动作里**沿输出维拼接一次**；原 `nn.Linear` 的 `weight`/`bias`
  **一个字节都不动** ⇒ `state_dict` 键与值前后**逐键相同**，随时可原样回退。
- 两个融合**各自独立开关**（`fuse_qkv` / `fuse_gate_up`），可单独消融；
  `linear_fuse_shapes` 提供按形状的选择（`None` = 全形状）。
- ★ **复用守卫**：QKV 靠"q 先算、k/v 复用"避免算三遍，复用以
  `(data_ptr, shape)` **与剩余消费计数**双重校验；任一条不成立 ⇒ 回退原 `F.linear`。

**★ 实测到的两条机制事实（本会话最有价值的部分）**

1. **两种融合的切分代价不对称**：
   - **QKV 必须 `contiguous()`** —— HF 的 attention 对 q/k/v 做 `.view(...)`，
     而"从融合输出切出的窄片"**不是连续张量**（行 stride 仍是融合宽度）。
     实测（微型模型）每层每次融合产生 **3** 次拷贝、字节数记在
     `bytes_copied` 里。⇒ 融合省下的 2 次 launch **要与这 3 次拷贝相抵**，
     净收益**不保证**（正是"不预设正收益"的机制来源）。
   - **gate/up 不需要 `contiguous()`** —— 下游是 SiLU 与逐元素乘法，对非连续张量天然可用
     ⇒ `split_copies = 0`、`bytes_copied = 0` ⇒ **这一支更可能是净收益**。
2. **形状选择是必要的**：prefill 与 decode 的 M 差三个数量级，用同一套规则必然有一端受损。

**验收（本地）**

| 判据 | 结果 |
| --- | --- |
| 任一/任一组合 ⇒ 逐 token 与逐层 KV 等于基线 | ✅ |
| 默认关闭 ⇒ 不装融合、`state_dict` 不变 | ✅ |
| 安装不改 `state_dict`（键与值） | ✅ |
| ★ 反例：k/v 收到**不同**输入 ⇒ 必须回退，不得复用陈旧值 | ✅（去掉守卫即失败：`5 failed` 反向验证） |
| 反例：v 先于 q 被调用 ⇒ 回退 | ✅ |
| 形状选择：集合外的 M 回退，且结果仍正确 | ✅ |
| `uninstall_fusions` 恢复类上的 forward | ✅ |
| **破坏验证** | ✅ 去掉复用守卫 ⇒ **1 failed**（且报"静默算错"） |

**⚠️ 未测**：M=1/2/4/8/128/512 的**算子耗时与净收益**、真实 7B 的
`3584/512/512 → 4608` 与 `18944×2 → 37888` 是否与声明一致 ⇒ `REMOTE_PENDING`。
已备 `scripts/inspect_linear_shapes.py`（`--tiny` 本地自检通过，真机读实际权重）。

**变更文件（S01–S03 合计）**

| 文件 | 变更 |
| --- | --- |
| `benchmark/real_backend.py` | 6 个执行优化开关 + 装载期能力判定 + `flush_allocator_cache()` + `execution_optimizations` 落盘 |
| `benchmark/linear_fusion.py` | **新增**（融合安装/卸载/清单） |
| `scripts/gen_exec_opt_configs.py` | **新增**（从平台基线派生 8 份 A/B 配置，含 `--check`） |
| `scripts/inspect_linear_shapes.py` | **新增**（读实际模型的投影形状/bias/dtype/stride） |
| `configs/mthreads/exec_opt/*.yaml` | **新增** 8 份（派生，revision 钉死） |
| `tests/_tiny_support.py` | **新增**（CPU 微型模型夹具，三会话共用） |
| `tests/test_optim_s01_logits_sampling.py` | **新增** 12 用例 |
| `tests/test_optim_s02_allocator.py` | **新增** 7 用例 |
| `tests/test_optim_s03_linear_fusion.py` | **新增** 16 用例 |

**下一步**：S04/S05/S06 需先冻结执行路径（按 profile 决定）；或按 `01_PROJECT_PLAN.md` §D
在路径冻结后直接准备 S07。**三项的收益结论一律待真机**。

### 5.5 EXEC-RUN 执行记录（2026-09-29，本地）

**目标（用户原话）**：准备好真机运行脚本，远端尽可能自动化、跑满、减少本地干预。

**交付：三层，各自可独立使用**

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 纯逻辑 | `benchmark/exec_opt_plan.py` | 穿插顺序、均衡重复的停机判据、进度/心跳载荷、原子写、臂解析 |
| 远端自驱动 | `scripts/run_exec_opt_matrix.py` | 无人值守跑臂 × 重复；30 s 心跳；STOP 协作停止；预算填充；断点续跑；**按产物判成功** |
| 本机一键 | `scripts/exec_opt_batch.py` | 核验 → 推送 → 单次起跑 → 轮询 → 回拉；`plan/preflight/run/status/stop/pull` |
| 安全阀 | `scripts/clear_stale_batch_lock.py` | 清理**陈旧**批次锁（四条条件，默认只报告） |
| 清单 | `handoff/optimization_20260929/EXEC_OPT_RUNBOOK.md` | 三条命令 + 失败判据表 + 红线 + 已知边界 |

**一条命令**（本机）：

```bash
python scripts/exec_opt_batch.py run --tag EXECOPT1 --stage screen --budget-minutes 240
```

**复用而不是重写**：`launch_batch.sh`（锁/setsid/监视/激活）、`push_files.sh`（逐文件
sha 校验）、`pull_incremental.py`（增量 + 完整性核对）、`check_remote_busy.py`、
`check_cpu_stability.py`、`diag_remote_reachability.py`、
`run_step3_matrix.read_run_status`（产物判定）。本会话只做**编排**。

**"跑满"是怎么定义的（不是"能塞多少塞多少"）**

* 默认 `--fill-budget`：**整轮整轮**跑到预算不够为止；预算不足跑完当前一轮 ⇒ 干净收尾
  （宁可少一整轮，也不出现"有的臂 5 次、有的臂 4 次"）。收尾时显式检查 `balanced`。
* `--min-repeats 3` 是不低于 3 次的下限；`--max-repeats` 是跑飞保护。
* 实测节拍由第一批自动测出（`avg_item_s`），后续停止判据用它 + 15% 余量。

**★ 本会话发现并修掉的一个新失效：陈旧批次锁**

`launch_batch.sh` 的锁由**监视进程**在批次结束后释放。本地实跑（`--activate` 全链路）
暴露：**本机 Git for Windows 没有 `setsid`** ⇒ 监视进程随父 shell 被整组杀掉 ⇒
锁永久留下。远端有 `setsid`，行为不同，但这说明"监视进程死掉 ⇒ 锁永留"是一条真实路径，
而它的症状是**之后所有批次都起不来且不报错** —— 无人值守场景下最贵的失效。

⇒ 新增 `scripts/clear_stale_batch_lock.py`，删除必须同时满足四条：
① 只删锁根目录下 `*.lock.d`；② 记录的 pid **确已不存在**；③ 年龄 > 10 min；
④ 调用方另行确认过"远端没有基准在跑"（编排器把它排在 `check_remote_busy` **之后**）。

⚠️ 自查到的一处**真错**：第一版解析器把 `holder` 当成"标记文件"跳过，
而**真实布局**是记录就写在 `holder`（`batch_lock.py:169`）⇒ 扫描恒得 `pid=None`
⇒ 清理**永不触发**，而报告看起来像"没有陈旧锁"。
根因是**夹具按想象造**（`{"tag":…}.json`）。已改为 ① 复用 `batch_lock.BatchLocks.holder()`
（唯一实现）；② 夹具按真实布局写 `holder`；③ 加一条**用真实 `batch_lock.py` 取锁**的用例。
同处还修了平台差异：不存在的 pid 在 Linux 抛 `ProcessLookupError`、在 **Windows 抛
`OSError(winerror=87)`**（实测），只捕前者会让清理在 Windows 上永不触发。

**验证（本地）**

| 项 | 结果 |
| --- | --- |
| `tests/test_exec_opt_plan.py` | **25 passed** |
| `tests/test_clear_stale_batch_lock.py` | **14 passed** |
| **全量回归** `PYTHONPATH= pytest tests/ -q` | **1507 passed / 1 skipped / 756 subtests / 0 failed**（1468 + 39 = 1507，口径可解释）|
| `launch_batch.sh` **全链路本地实跑**（含 `--activate` 激活层、锁、监视进程） | ✅ 激活生效（`ACT=yes`）、锁已取、`BATCH_LAUNCH_OK`、rc=0 |
| 清理器对**真实** `batch_lock.py` 产物 | ✅ 识别并删除两个卡死锁（pid 已不存在）|
| 破坏验证 | ✅ 还原错解析器 ⇒ **5 failed**，恢复 ⇒ 14 passed |
| 离线路演 `exec_opt_batch.py plan` | ✅ 打印完整命令序列（不触网） |
| `run_exec_opt_matrix.py --dry-plan` | ✅ screen 4 臂 / full 9 臂，顺序为穿插 |

**限制（不掩盖）**

1. **远端不可达** ⇒ 本清单**未在远端实跑过**；全部验证是本地组件级 + 离线路演。
2. **"批次结束自动放锁"本地验不了**（缺 `setsid`）—— 这是远端的契约，只能到远端确认；
   已用安全阀兜住后果。
3. `--budget-minutes 240` 是**默认值不是测量结果**：真实节拍靠第一批实测。
4. 交接包 `STATE.json` 的额度字段仍为 `null`（未知）。

**下一步**：环境恢复后 `preflight` → `run --stage screen`；有信号再 `--stage full`。

---

### 5.6 EXECOPT 真机批次执行记录（2026-09-30，远端 S4000）

**环境恢复**：用户给出 SSH 凭据（`SSH_HOST_FROM_PRIVATE_CONFIG:28080`）。
上一节"远端不可达"的限制**已解除**；本轮起在**真机**上跑批次。
⚠️ 连上过程**连环踩了 7 个坑**（逐一记成因见 `st/.workbuddy/memory/2026-09-30.md`），
其中**坑 ②/③/④/⑤ 属"引号/编码/转义"家族**，**坑 ⑦ 是本章最贵的一个**：

> **★ `bash -c` 会重置 PATH。** `launch_batch.sh --activate` 确实执行了（日志有
> `[activate_musa] venv=...`），但**承载真命令的内层 `bash -c` 重新加载 profile**，
> 把 venv 的 `bin` 从 PATH 头部挤掉（→ `/root/miniconda3/bin`）⇒
> 真机报 `ModuleNotFoundError: No module named 'transformers'`（**EXECOPT1 批次即死于此**）。
> ⇒ 修：解释器一律用**绝对路径**；并加**起跑路径守卫** —— 核验**必须用与起跑同形态的
> 调用**，否则"核验通过、起跑炸"。

**★ 新失效（纯逻辑）：EXECOPT2 静默少跑一轮（8/12）**

| 批次 | 结果 | 处置 |
| --- | --- | --- |
| `EXECOPT1` | **失败**（`ModuleNotFoundError`，坑 ⑦） | 产物**不可用**，留作反例 |
| `EXECOPT2` | **8/12**（应 12），`failures=[]`、无报错 | 产物**不作结论**（少一轮 + 跨轮条件漂移）|
| `EXECOPT3` | 完整 12 项（本轮） | 见下 |

根因（`scripts/run_exec_opt_matrix.py`）：**同一个 `total_planned` 既做报告量、又做停止判据，
且每轮被覆写**。L264 正确写 `len(cells)*repeats`（=12），L372（**已删**）每轮改成
`len(cells)*(pass_index+1)` ⇒ r1 后 `decide_stop(items_done=8, items_total=4)` ⇒ 判"全部完成"。
⇒ 纯逻辑 bug，**与硬件无关、且完全静默**。

★ **可复用规则**：**报告用的派生量与判据用的阈值必须来自不同来源** —— 同一变量一旦既描述
"计划"又随进度被改写，停止判据就会**随进度自毁**，且不报错。

**修与验证**：删 L372 + 新增 `TestMultiPassProgression`（4 用例）。
**破坏验证**：`items_done >= items_total` → `>` ⇒ **3 failed**（含新增 2 条）⇒ 测试真能抓。

**EXECOPT3 起跑（preflight 全绿）**

```
[ok] 起跑路径守卫：经 bash -c 用 /root/musa-envs/qwen310/bin/python3 仍可 import（与起跑同形态）
[ok] revision 快照在位：a09a35458c702b33eeacc393d103063234e8bc28
IDLE：远端无基准进程 / [stale-lock] 没有锁 ⇒ 无陈旧锁
稳健 CV : 0.005%  ✓ 稳定（≤1%）⇒ 计时环境可用
```
命令：`python scripts/exec_opt_batch.py run --stage screen --tag EXECOPT3 --budget-minutes 240`

**中途抽样**：r0 四项全成功（`baseline` 3.0 / `s01_both` 2.8 / `s02_no_empty_cache` 2.9 /
`s03_fuse_gate_up` 3.0 min）⇒ **三个开关都能在真机执行**；r1 同样 4/4 成功；
★ **批次未在 8/12 提前停止** ⇒ **"少跑一轮"的修复在真机上得到确认**。
每项 ~3.0 min ⇒ 12 项约 36 min。⚠️ 进度文件 `n_done` 轮间刷新有滞后，**以完成行为准**。

**限制（不掩盖）**

1. 本轮只证明**机制能执行 + 链路可靠**；**收益结论一律待 12/12 回拉 + `EXPERIMENTS.md` 登记**。
2. `EXECOPT1`/`EXECOPT2` 产物**不作结论**（一个环境失败、一个少一轮）。
3. `--budget-minutes 240` 是默认值；真实节拍 ~3.0 min/项（EXECOPT3 实测）。

**下一步**：EXECOPT3 → 12/12 ⇒ `pull` ⇒ `EXPERIMENTS.md` 登记四臂对照（**仍不写收益结论**）。

---

### 5.7 EXECOPT3 收尾：12/12 完成 + 发现并修掉"等待侧只认一条退出通道"（2026-09-30）

**批次结果（真机，`EXECOPT3`）**：`n_done=12/12`、`completed_passes=3`、**`balanced=true`**、
**`failures=[]`**、`elapsed_s=2114.6`（35.2 min）、`avg_item_s=176.2`、`stopped_by_budget=true`。
产物 12 个目录全部回拉 + 完整性核对通过（`4 个文件齐、data_label 与 n_requests 有值`）。
⇒ **`EXECOPT2` 的"静默少跑一轮"已确认修复**（本轮真机没在 8/12 停）。

**★ 新失效（第九个坑）：本机等待侧只认一条退出通道 ⇒ 远端跑完、本机空转 1 h+**

远端 35 min 就跑完 12/12 并写了 `exec_opt_summary_EXECOPT3.json`，但**本机 `run` 进程
继续轮询了 1 h 13 min**（最后是我手动终止的）。根因：

| 退出通道 | 谁写 | 何时写 |
| --- | --- | --- |
| `progress.stop.stop` | runner | **仅**协作式停止（STOP 文件）|
| `summary` 文件 | runner 收尾 | **总是**（预算收尾 / 全部完成 / 中断都写）|
| `n_done >= n_total` | runner | 项数跑满时 |

`step_wait` **只查了第一条**。而 EXECOPT3 是**预算收尾**（`stopped_by_budget=true`）
⇒ `stop` 字段**根本不写**（实测 progress.json 的 keys 里没有 `stop`）⇒ 条件永假
⇒ 轮询到 `--wait-timeout-minutes`（**默认 600 min**）为止。
症状：**远端早已空闲、本机显示"在跑"** —— 与"卡住"不可区分。

⇒ **修法**：把退出判据提到**纯逻辑层**（`benchmark/exec_opt_plan.py::decide_wait_exit`，
返回 `WaitExit(done, channel, reason)`），`step_wait` 只做 I/O。三条通道全识别，
`summary` 为**权威**（runner 最后一步写入），`all_items` 为**冗余**兜底。

**验证**

| 项 | 结果 |
| --- | --- |
| `tests/test_exec_opt_plan.py` | **37 passed**（29 → 37，+8 新用例）|
| 新增 `TestWaitExitDecision` | 8 用例，含**否定测试** `test_old_logic_would_hang`（证明旧判据在预算收尾时**不**退出）|
| **破坏验证** | 关掉 `summary` + `all_items` 两条通道 ⇒ **3 failed**（`test_summary_is_the_authoritative_exit` / `test_all_items_is_redundant_fallback` / `test_old_logic_would_hang`）；还原后 37 passed，与备份 `diff` **逐字节一致**、CR=0 |
| **全量回归** | `pytest tests/ -q --ignore=tests/test_validate_fp16_path.py --junitxml=…` ⇒ **2204 tests / 0 failures / 0 errors / 52 skipped**（junit XML 口径；命令行汇总行被本环境的临时目录清理钩子截断，故改用 XML）|

**收益结论（已登记 `EXPERIMENTS.md` §3.6 / §1.5，这里只摘要；细则以那里为准）**

- **E16 `s01_both`（logits_last_only + sample_needed_only）✅ 正向**：
  TTFT p50 **−4.2%**（196.98 → 188.78 ms），3/3 同号，**36× 噪声底**；tpot 无变化。
- **E17 `s02_no_empty_cache` ⚪ 不可分辨**：−0.13% / −0.1%，**低于**噪声底（0.444% / 0.117%）
  ⇒ 属"测不出"，**不是**"无收益"。
- **E18 `s03_fuse_gate_up` ⚠️ 负面**：tpot p50 **+1.10%**（3/3 同号，**2.5× 噪声底**）、
  吞吐 −0.9%；差异**只在 decode 段** ⇒ 每步固定开销。已归档 §4.2 N8。

⚠️ **仍不写"论文级收益结论"**：本批是**单协议**（`short_chat`、并发 8、`eager`），
`git_commit=NOT_MEASURED`；E18 的负号**不得**外推为"融合无用"。

**限制与未决**

1. `--wait-timeout-minutes` 默认 **600 min** 太大 —— 即便修好退出通道，它仍是"卡住"时
   最后一道兜底。建议默认降到 **120 min**（本轮实测单批 35 min），**待裁定**。
2. EXECOPT3 的 `git_commit` 仍是 `NOT_MEASURED`（`stamp_stale`）⇒ 本批数字无源码版本绑定。
3. `s02` 若要给出"有/无收益"，须另设计协议（更大批量 / 更碎片化分配）。

**下一步**：S04/S05/S06 需先**冻结执行路径**；或按本批结论把 `s01_both` 定为**后续批次的默认基线**。

---

### 5.8 执行路径冻结 + S04/S05/S06 取舍（2026-09-30，本地判定，不外推）

> 依据：`01_PROJECT_PLAN.md` §1/§4 与 §D；`S03` 任务书结尾逐字要求；`S04/S05/S06` 三份
> 任务书的**前置条件**。本节的每个数字都**从 EXECOPT3 产物现读复算**，不引用记忆值。

#### 5.8.1 ★ 冻结的执行路径（本阶段结论）

**保留并冻结**（作为后续批次的默认基线）：

| 开关 | 判定 | 依据（EXECOPT3，3 轮） |
| --- | --- | --- |
| `logits_last_only` + `sample_needed_only`（**`s01_both`**） | ✅ **保留**，定为后续批次默认基线 | TTFT p50 **−4.17%**（196.988 → 188.769 ms），3/3 同号，**35.6× 噪声底**；tpot 不变（−0.03%）；两开关 `effective=true` 且 `n_calls_last_only=1377`/`n_sampled_rows=15353`（非零 = 真走到） |
| `retire_empty_cache=false`（`s02`） | ⚪ **不冻结、不启用** | −0.09% / −0.25%，**低于**噪声底（0.117% / 0.444%）⇒ **测不出**（≠ 无收益）；要判需另设计协议 |
| `fuse_gate_up=true`（`s03`） | ⚠️ **不启用**（有效负结果） | tpot p50 **+1.02%**（3/3 同号，**2.3× 噪声底**）；差异只在 decode 段 ⇒ 每步固定开销；已登记 §4.2 N8 |

⚠️ **冻结的是"这批数字在**单协议**下支持的选择"，不是"论文级结论"**：本批 `short_chat`、
并发 8、`eager`、`git_commit=NOT_MEASURED`；§5.7 的三条限制全部仍然成立。

#### 5.8.2 四臂受控性复核（本次现读，含"配了没走到"判定）

| 臂 | `n_forwards` | `pack_ms_total` | `fwd_ms_total` | `last_only` 调用 | `sample` 跳过计数 | `retire` 计数 | `linear_fusion` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `s4000_fp16` | 1377 | 26725.5 | 141670.7 | 0 (all=1377) | 0/0 | calls=63, supp=0 | null |
| `s01_both` | 1377 | 26764.2 | 136256.7 | **1377** (all=0) | **15353**/0 | calls=63, supp=0 | null |
| `s02_no_empty_cache` | 1377 | 26750.4 | 142755.3 | 0 (all=1377) | 0/0 | calls=0, **supp=63** | null |
| `s03_fuse_gate_up` | 1377 | 26771.3 | 149848.6 | 0 (all=1377) | 0/0 | calls=63, supp=0 | gate_up 28 组 / fused 38556 / fallback 0 |

- **计数与 `requested` 逐臂一致**（requested=true 的臂计数非零、requested=false 的臂计数为 0）
  ⇒ **没有"配了但没走到"**。`s02` 的 `n_suppressed=63` 正是"关掉了 empty_cache"的**正向证据**
  （抑制发生 63 次、调用 0 次），不是"没生效"。
- 配置层面核对：四个 URL 的 yaml 差异**只有** `execution_optimizations` 下那一行开关
  （其余 92 行同源），⇒ **单变量成立**。

#### 5.8.3 S04/S05/S06 取舍（依 profile 证据，不依推测）

| 会话 | 任务书前置 | 现有证据 | 裁定 |
| --- | --- | --- | --- |
| **S04** 持久KV与输入缓冲 | "**只有 profile/分配证据支持才进入**" | ★ **满足**：`pack_ms` 段（定义在 `benchmark/real_backend.py:55` = "补位与切片的显存搬运，**不含任何设备计算**"）稳定占 `pack+fwd` 的 **15.16%–16.42%**；且该段在**四臂间极差仅 0.17%**（26725/26764/26750/26771 ms）⇒ 是**独立于 S01/S02/S03、未被触碰的热点**。代码层面：每步每槽位 ≥2 次全量 KV 分配 + 拷贝（入口 `_pack_past` 的 `F.pad`+`torch.cat`，`real_backend.py:1489-1509`；出口写回段逐层逐行切片重建 `[i:i+1,:,pad:pad+keep,:]`，`real_backend.py:1426-1432`） | ✅ **进入 S04** |
| **S05** GQA 支持度 | "先记录实际 `attention` 类与内核，检查 `repeat_kv`/reshape 后的 storage、分配、拷贝和耗时… 若不存在显著复制/热点，记录结果结束" | ⚠️ **无该粒度证据**：`src/` 与 `benchmark/` 中**无任何** `repeat_kv`/`enable_gqa` 实现（grep 零命中）；`repeat_kv` 发生在 attention **内部**，计入 `fwd_ms`，当前 harness 不拆出该量 | ⏸ **先做探针**（P4 能力探测），**不直接实现**；无热点即结束该分支 |
| **S06** 同步优化与 decode 图捕获 | "依赖执行路径冻结**和 S04 提供的稳定地址/容量/slot 映射**" | 执行路径本回合已冻结 ✅；但**地址/容量/slot 映射须由 S04 交付** | ⛔ **被 S04 阻塞**，S04 完成后再判 |

**⇒ 选定顺序：`S04` → （S04 交付地址/容量/slot 映射后）`S06`；`S05` 并行先做能力探针。**
S08/S09 保持 `OPTIONAL_NOT_SELECTED`（默认不启动）。

#### 5.8.4 四状态（本回合）

| 项 | 状态 |
| --- | --- |
| 代码实现 | 本回合**无代码改动**（仅只读复核 + 决策登记）|
| 离线验证 | 不适用（无实现）|
| 真机实测 | S01–S03 四臂 **12/12 已有 MEASURED**（EXECOPT3）；本回合未新增真机运行 |
| 结论资格 | **选择资格成立**（冻结哪些开关 / 进哪个会话）；**收益资格不成立**（单协议、`git_commit=NOT_MEASURED`）|

**下一步**：进入 **S04**——先按任务书做"成员固定、历史等长 decode 批"的 fast-path 与
独立回退，交付 fast-path 适用条件、回退计数、独立消融与资源生命周期证据；
S05 并行只做能力探针；S06 等 S04 的地址/容量/slot 映射。

---

### 5.9 S04 真机批次 EXECOPT4 + 三处编排/口径缺陷（2026-09-30，远端 S4000）

> ## ⚠️⚠️ 2026-10-01 更正（**本节以下三处结论已作废，以本节末 5.9.9 为准**）
>
> 独立复核（用户，2026-10-01）指出并被我复现确认了**三处实质误判**：
>
> | 原写法（错） | 事实 | 影响 |
> | --- | --- | --- |
> | 「覆盖率 **193/1260 = 15.3%**」 | **193 含预热**（预热 1 轮 = 116 步纯 decode，也命中快路径），分母 1260 也没扣 ⇒ 正解 **77 / 1143 = 6.74%** | 把"几乎没测到"说成"测到了 15%" |
> | 「`pack_ms` 无下降，差异 ≤ 噪声 ⇒ **测不出收益**」 | 全窗口 +0.19% 是 **6.74% 覆盖率稀释**的结果；在**有效样本**（单请求纯 decode 尾段 77 步）上 `pack` 总量 **109.02 → 187.52 ms = +72.0%**（3 轮极差 0.4%） | 把**真实的负效应**说成"不可分辨" |
> | 「环境**双模态**已证实；吞吐下降是**抽样假象**」 | 那是每轮**一次孤立单步长停顿**（13.9–19.3 s），不是持续慢态；wall 差额与其**逐轮对上**；成因未定位 ⇒ **机制归因暂不可判** | 过度断言，掩盖了未解释的长停顿 |
>
> 另有两处**我方漏检**（比上述更严重，属正确性与证据链）：
> （a）**S04 `fast→fallback→fast` 重入读过期缓冲** ⇒ 数值错（我方既有用例因"从不移除混长请求"
> 而未覆盖重入）；`tokens_out` 漏计快路径产出；
> （b）**12/12 产物 `git_commit=NOT_MEASURED`**，而编排忽略了 stamp 的退出码。
> 详见 **5.9.9**（修复与反例）与 `EXPERIMENTS.md` §4.4 A3–A6。
>
> ⚠️ 本节 5.9.1 的**批次事实**（12/12、balanced、臂开关、单变量）与
> 5.9.2 的**回退计数**（193/1067、`unequal_kv_len=1002`、`capacity_exceeded` 一次未现）
> **仍然成立**；5.9.3 / 5.9.4 的**推送守卫与 warmup 口径缺陷本身**也成立。
> 作废的只有：覆盖率的**数值**、`pack` 的**结论**、以及**双模态归因**。

#### 5.9.1 批次事实

`exec_opt_batch.py run --tag EXECOPT4 --arms s01_both s01_both_s04_pkv s04_pkv`，
预算 240 min，**4 臂 × 3 轮 = 12 项**，穿插（pass-major）。
`n_done=12/12`、`completed_passes=3`、`balanced=true`、`failures=[]`、
`elapsed_s=2101.8`（35.0 min）、`avg_item_s=175.1`。
产物 `results/raw/EXECOPT4-*`（12 目录）。全部 `data_label=MEASURED`。

臂开关（从产物 `config.runtime` 现读，非从配置推断）：

| 臂 | 开关 |
| --- | --- |
| `s4000_fp16` | （无） |
| `s01_both` | `logits_last_only=true`、`sample_needed_only=true` |
| `s04_pkv` | `persistent_kv_slots=true`、`persistent_kv_capacity=1024` |
| `s01_both_s04_pkv` | 上面两者之和（**单变量成立**：相对 `s01_both` 只多 2 个键） |

#### 5.9.2 ★ 机制确实生效，但**覆盖面只有 15.3%**

| 臂 | `n_fastpath_steps` | `n_fallback_steps` | `fallback_reasons` |
| --- | --- | --- | --- |
| `s04_pkv` | **193**（3/3 轮完全相同） | **1067** | `unequal_kv_len=1002`、`new_sequence=65` |
| `s01_both_s04_pkv` | **193**（3/3 轮完全相同） | **1067** | 同上 |

- 分桶**闭合**（65+1002=1067 ✓），逐轮**逐位相同** ⇒ 判据是确定性的。
- ★ **阻塞项是 `unequal_kv_len`（1002 次）而不是容量**：`capacity_exceeded` **一次都没出现**。
  连续批处理下各序列的 `past_len` 天然相差一个常数偏移（按到达时刻），
  「恰好全等」是**稀有时刻** ⇒ 现判据（要求严格等长）在 `short_chat` 上**结构性罕见**。
- `bytes_allocated` 终态为 0 **不等于没分配**（见 5.9.4 缺陷③）；分配证据是
  `capacity=1024 / n_kv_heads=4 / head_dim=128 / dtype=float16` 非空。

#### 5.9.3 ★ 判据结果：`pack_ms` **没有下降**（报告 +0.2%~0.5%）

口径说明见 5.9.4 缺陷② —— `driver_stats.*_total` **含 warmup** 且各臂 warmup 轮数不同
（2 vs 1，差 117 次前向）。**唯一可比的是 telemetry 求和**（正式测量段，逐 step `note` 拆解）：

| 臂 | `pack_tel` 中位 (ms) | 相对其基线 | `fwd#tel`（前向次数） |
| --- | --- | --- | --- |
| `s4000_fp16` | 21,269.6 | — | 1,143 |
| `s04_pkv` | 21,310.4 | **+0.19%** | 1,143 |
| `s01_both` | 21,267.9 | — | 1,143 |
| `s01_both_s04_pkv` | 21,378.0 | **+0.52%** | 1,143 |

- ★ **`fwd#tel` 四臂全等（1,143）** ⇒ 正式段的设备前向量**完全相同** ⇒ 受控成立。
- 逐轮 `pack_tel` 稳定性：`s04_pkv` 21,288.2–21,324.0（**0.17%**）、
  `s01_both` 21,251.2–21,268.4（0.08%）、基线 21,186.8–21,277.6（0.43%）。
  **差异（+0.2%~0.5%）与噪声同量级或更小 ⇒ 判"无收益"**（不是"有效应但方向错"，是**测不出**）。
- ⚠️ `s01_both_s04_pkv` 的 **r0 有一个 `pack_tel` 离群点**（38,273.4，+79% vs 同臂另两轮
  21,352.0 / 21,378.0）。★ **成因未定位**：它**不能**用慢模态解释——同一慢模态里的
  另外 3 个 run（`s4000_fp16` r0、`s01_both` r2、combo r2）的 `pack_tel` 都在 21,186–21,378
  （正常范围）。⇒ 登记为**未解释的离群点**，不得归因给 S04、也不得归因给慢模态。
  影响有限：即使把它算进去，组合臂相对 `s01_both` 的 `pack_tel` 也只从 +0.52% 变成
  +0.52%（中位数对单个离群点不敏感）。

**⇒ S04 第一版（等长判据 + 定容 1024）在本协议下没有降低 `pack_ms`。**
按任务书"若不存在显著复制/热点，记录结果结束"的同族纪律：**这是一条有效的负面结果**，
但**不得**外推为"常驻 KV 缓冲无用"——因为**快路径只有 15.3% 的步真的走了**，
而回退路径仍做原来的补位/切片搬运（这正是"机制没被行使"与"机制无效"必须分开的情形）。

#### 5.9.4 ★ 本批暴露的三处缺陷（都属"报告成功但事实不符"族）

**① `exec_opt_batch.py` 推送**静默不推**（已修，含两道守卫 + 破坏验证）**

`step_push` 曾用「取以两个空格开头的整行」解析 `batch_sha_diff.py` 的**人类报告**。
而报告里 `sha 不同` 的行是
`  scripts/x.py  local=<sha12> remote=<sha12>`，末尾还有「建议命令」区块（续行也以两空格开头）
⇒ 解析出的是**伪路径** ⇒ `push_files.sh` 判「跳过（不存在）」⇒ `N=0 / ok=0 / skip=0 / fail=0`
⇒ **退出码 0** ⇒ 编排打印"推送源码与配置"后照常起跑，**远端一个文件都没更新**。
实测后果：远端 `scripts/run_exec_opt_matrix.py` 落后本机，于是本批 summary 的
`switches` 里 `s04_pkv` 显示为 `{}`（缺 pkv 两个键）——**报告缺口，非实验失效**
（`benchmark/real_backend.py` 与全部 S04 配置当时已与本地一致，130 个文件比对通过）。

修法（三处）：`batch_sha_diff.py` 增 `--emit-paths`（只吐路径、无后缀）；
`step_push` 改消费它；并加**守卫 1**（清单每项必须是本机真实文件，否则报错停跑）与
**守卫 2**（解析 `push_files: ok/skip/fail (共 N)`，要求 `fail=0`、`N==清单长度`、`ok+skip==N`）。
本地单测 `tests/test_exec_opt_plan.py::TestPushTodoParsing/TestPushSummaryGuard`（12 例），
破坏验证 2 组（退回旧解析 ⇒ 2 failed；守卫 2 恒返回 None ⇒ 4 failed），还原后 57 passed、`CR=0`。

**② `driver_stats` 的 `*_total` 含 warmup，而各臂 warmup 轮数不同（未修，已加新键）**

`_drive_warmup_batch` 直接调 `self.forward(...)` ⇒ 预热与正式段**共用同一累加器**
（`_n_forwards` / `_pack_s` / `_fwd_s` / `_tokens_out`），而 `wall_ms` 明确排除预热。
预热轮数由"累计跑到 `target_s=30`"决定 ⇒ 本批实测 **2 轮 vs 1 轮**
⇒ `n_forwards` 1377 vs 1260，**差 117 = 恰好一轮预热的 1 次 prefill + 116 次 decode**。
若直接比 `pack_ms_total`，会读出 **"+31%"** 的假效应（真相是 +0.2%~0.5%）。

修法：**不改既有键语义**（否则与历史产物不可比），改为在 `warmup()` **循环之后**记录
累加器零点，`stats()` **新增** `measured_window` 组
（`n_forwards_measured` / `pack_ms_measured` / `fwd_ms_measured` / `tokens_out_measured` / `basis`）。
⚠️ 快照必须在循环**之后**：放循环之前等于什么也没扣（本地单测抓到过）。
单测 `tests/test_optim_s04_persistent_kv.py::TestMeasuredWindowScope`，
破坏验证 1 组（零点不记录 ⇒ 1 failed）。本批产物**早于该字段** ⇒ 只能用 telemetry 求和替代。

**③ `persistent_kv_slots.bytes_allocated` 终态恒 0，无法区分"没分配"与"分配后释放"（已修）**

`retire()` 会把驻留字节扣回去（设计成"当前驻留"），而 run 收尾时所有请求都已 retire
⇒ 报告里恒为 `0`——**跑了 193 步快路径却显示 0 字节**。
修法：新增 `bytes_peak`（峰值）与 `n_allocations`（累计分配次数），二者不因释放而归零。
单测 `TestAllocationVisibleAfterRetire`（含"未开机制时峰值为 0"的反向对照），
破坏验证 1 组（不记峰值/次数 ⇒ 1 failed）。

> ⚠️ **破坏验证方法本身的一处教训**：首次注入破坏时，我把注入行放在了随后的真赋值**之前**
> ⇒ 被覆盖成死代码 ⇒ 用例照常通过。**"破坏后用例没红"首先怀疑注入无效**，不要据此
> 认定"用例不敏感"。另：备份必须**一次备好、各次从同一份恢复**——"每次注入前先备份当前文件"
> 会把已带破坏的版本存成备份（本轮第一次就踩到）。

#### 5.9.5 ⚠️ 环境双模态 ⇒ 服务指标（ttft/tpot/吞吐）本批**不确定**

按 `recorded_at` 排序的 12 个 run，`wall_ms` 明显分成两簇：

| 模式 | wall | 吞吐 | 出现的 run |
| --- | --- | --- | --- |
| 快 | 123.2–123.6 s | 63.6–63.8 | 8 个（含 `s04_pkv` **全部 3 轮**）|
| **慢** | 137.1–143.0 s | 55.0–57.4 | **4 个**：`s4000_fp16` r0、`s01_both` r2、`s01_both_s04_pkv` r0、r2 |

- 慢模态**跨 3 个不同臂分布**，且在时间上成簇（≈19:54 与 ≈20:20–20:26 两个窗口）
  ⇒ **是机器/环境瞬变，不是机制效应**。
- 因此 `s01_both_s04_pkv` 表面上的「吞吐 −12.4%、`tpot90` 212→224 ms」**不能归因给 S04**：
  它只是 **2/3 轮落在慢模态**，而慢模态率约 33%（4/12），n=3 时完全在偶然范围内。
- **服务指标本批不可用**；**`pack_tel` 的结论不受影响**。★ 分层中位数（12 个 run 按
  `wall_ms` 分簇的**实测**对比，非推断）：

  | 模态 | n | `pack_tel` 中位 (ms) | `fwd_tel` 中位 (ms) |
  | --- | --- | --- | --- |
  | 快（wall ≤ 130 s） | 8 | 21,282.9 | 101,167.8 |
  | 慢（wall > 130 s） | 4 | 21,314.6（**+0.15%**） | 116,437.2（**+15.1%**） |

  ⇒ **`fwd_tel` 被慢模态抬高约 15% ⇒ 本批不可比**（与 `EXPERIMENTS.md` §4.4 A3 一致）；
  `pack_tel` 只差 **0.15%** ⇒ 慢模态**不系统影响**它（若剔除那个未解释离群点，
  慢簇的 `pack_tel` 反而略低）。
  ⚠️ 那条离群点（`s01_both_s04_pkv` r0 = 38,273.4）**成因未定位**，且**不能用慢模态解释**
  ——同簇另 3 个 run 的 `pack_tel` 都在正常范围。

#### 5.9.6 与 EXECOPT3 的交叉印证

`s01_both` 的 **TTFT p50** 本批 188.80 ms vs 基线 197.13 ms（**−4.2%**，3/3 同号），
与 EXECOPT3（188.78 vs 196.98，−4.2%）**独立复现**。
臂内跨轮极差：`ttft50` 0.03%–0.14% ⇒ 该效应约 **30–100× 噪声底**。
⇒ E16 由"单批"升格为**两次独立批次一致**（仍限 `short_chat` + `eager` + 并发 8）。

#### 5.9.7 四状态（本回合）

| 项 | 状态 |
| --- | --- |
| 代码实现 | ✅ S04 快路径 + 3 处报告/编排修复（均默认不改变既有行为；`tests/` 81 passed 受影响集）|
| 离线验证 | ✅ 单测 16 例（S04）+ 12 例（推送守卫）+ 8 例（分析器）；破坏验证 4 组 |
| 真机实测 | ✅ EXECOPT4 12/12 `MEASURED`；机制生效（193/1067）|
| 结论资格 | ⚪ **仅"无收益"成立**（`pack_tel` +0.19%/+0.52%，差异 ≤ 噪声）；**服务指标不合格**（双模态）；**不得**外推为"机制无用" |

#### 5.9.8 下一步（须先裁定）

1. **S04 是否继续**：现判据（严格等长）在连续批处理下只有 15.3% 覆盖面。
   要让它可被行使，须把缓冲改为**按槽位独立偏移/长度**（每个槽写自己的位置，
   配 per-slot 长度），即向 paged KV 靠一步。**这是一次真实的设计扩张**，
   且**必须**先证明「每步全量 `copy_` 已累积 KV」不是净增成本（本版 `_pkv_writeback` 整段写回，
   机制上可能比原来的 `F.pad`+`cat` 更贵——与实测"+0.2%~0.5%"一致）。
2. **服务指标协议**：需先**探测并标记慢模态**（或把每轮时长纳入判据、按模态分层），
   否则 3 轮里落 1–2 个慢轮就会造出假效应。**不得**在未分层时比较 tpot/吞吐。
3. **S06**：S04 已交付固定地址缓冲 + 定容量 + 槽位映射，任务书要的"稳定地址/容量/slot 映射"
   在设计上已具备；但其**收益前提**受第 1 条制约 ⇒ 建议**先裁定第 1 条**再开 S06。
4. **S07 预注册 §5.1/§5.2** 仍待裁定（`--dry-run` 验不了判据；§5.2 只改时间戳）。

---

### 5.9.9 正确性与证据链修复轮（2026-10-01，**本轮不做新性能实验**）

执行用户 2026-10-01 复核后的 5 条指示（修正确性与溯源，暂缓扩大 S04）。**未启动任何远程实验。**

#### (1) ★ S04 `fast→fallback→fast` 重入读过期缓冲（数值错，已修 + 反例）

**机制**：回退路径只更新 `_Slot.past`（`real_backend.py` 通用分支），**完全不碰
`_kv_buf`**；而重入时 `_pkv_ensure_buffers` 的原实现是 `if buf is not None: continue`
⇒ 缓冲里还是**上一次快路径**的内容 ⇒ 形状合法、不报错、数值悄悄错。

**独立复现（CPU 微型 Qwen）**：同一请求重入后输出 token **60 → 36**、下一步 **62 → 4**；
逐层 KV 与基线最大差 **2.102e-01**。

**修法**：给每个槽位记录 `_pkv_synced_len[req_id]`（= "缓冲前 L 行与 `s.past` 一致"）。
判据从"缓冲在不在"改为"**已同步长度是否等于当前 `past_len`**"——回退只推进 `past_len`、
不碰缓冲 ⇒ 长度必然落后 ⇒ 重入时**重同步**（整段 `copy_` 回来后标记 clean）。
写回后更新该长度；`retire` 一并 `pop`（防 `req_id` 复用时被判"已同步"）。
重同步单独计数 `n_resyncs`（它是真实成本，不隐藏）。

**新增用例**（`tests/test_optim_s04_persistent_kv.py::TestReattachAfterFallback`）：
`等长快路径 → 混长回退 → **新请求退出** → 重入快路径 → 继续`，对照 **token 序列 + 逐层 KV**。
★ **为什么旧用例漏了**：既有 `_fallback_scenario` 补入混长请求后**从不移除它**
⇒ 批一直是混长 ⇒ 快路径**再也没有被重新进入** ⇒ 重入路径一次都没覆盖。
**破坏验证**：把判据退回 `if buf is not None` ⇒ 用例失败，报文
"`a` 第 0 层 K 在重入快路径后与基线不同（最大差 **2.102e-01**）"。
（token 在该小场景里恰好还没分叉 —— 加逐层 KV 对照是对的。）

#### (2) ★ 快路径漏计 `tokens_out`（已修 + 反例）

`self._tokens_out += 1` 原**只在通用分支**；快路径分支只 `record_token` + 填 `produced`
⇒ 快路径产出的 token 全被漏记。**CPU 微型复现**：实际 17 个 token、只记 9 个。
⇒ `measured_window` 也救不了（它只是相减，漏计照样相减）。
**修法**：快路径分支补同一次计数。**用例** `TestTokensOutCountsFastpath`
（断言 `stats()["tokens_out"] == 请求数 ×(1+steps)`，且基线同口径）。
**破坏验证**：删掉该计数行 ⇒ 用例失败。

#### (3) ★ 分桶与分母口径（已修）

* `fast`/`fallback`/`reasons` **拆成 warmup 与正式窗口**：`_counter_snapshot` 纳入
  `pkv_fastpath/fkv_fallback/pkv_resyncs`，`measured_window` 给出
  `pkv_fastpath_measured` / `pkv_fallback_measured` / `pkv_resyncs_measured`
  以及两个覆盖率（`pkv_coverage_of_forward_groups` / `..._of_logical_steps`）。
* **两个分母显式化**：`n_forwards`（= 前向**组**，`_shape_groups` 可能把一步拆成多组）
  与新增的 `n_logical_steps`（= `forward()` 调用数）。混形状批次下两者不等
  （EXECOPT4：**1,143 vs 1,080**）⇒ 报覆盖率必须写明用哪个。
* 注释写明 `n_fastpath_steps`/`n_fallback_steps` 是**组级**计数。

#### (4) ★ 部署校验闭环（已修 + 反例）

`step_push` 末尾 `_remote(["stamp", ...], check=False)` **丢掉了退出码** ⇒
`do_stamp` 返回 4/6（生成不了可信戳 / 远端重算摘要不符）也照样进 `step_launch`
⇒ 12/12 产物 `git_commit=NOT_MEASURED` + `git_source=stamp_stale`，
而 `data_label` 仍是 `MEASURED`。**"有测量标签、没有源码身份"。**

**修法**（两个判据都要过）：
* `stamp_complaint(rc, out)`：`rc != 0` ⇒ 拒；`rc == 0` 但输出缺
  `投放并校验通过` 正标记 ⇒ **也拒**（"没报错"不等于"验过"）；
* **断点续跑加身份核验** `identity_complaint()`：源码来源必须可信
  （`git`/`stamp`，拒 `stamp_stale`/`none`）+ 产物里的 `config.snapshot.yaml`
  与当前配置**逐字节一致** + `model_revision` 一致；不符 ⇒ **拒绝复用**并报错，
  不再只看"请求完成数"。

**用例**：`TestStampGuard`（4 例，含"rc=0 但无正标记必须拦"的否定测试 +
"标记常量必须与 `remote_run.py` 实际打印串一致"）、`TestResumeIdentityCheck`（5 例，
含 `stamp_stale` 被拒——**那正是本批 12 份产物的形态**）。
**破坏验证**：两个守卫各自恒返回 `None` ⇒ 各 3 failed。

#### (5) ★ 从原始遥测重建 EXECOPT4 分析（已撤回错误归因）

`scripts/analyze_exec_opt_batch.py` 新增三节，全部从**原始 telemetry** 重算：

* **①c 覆盖率**：分母 = **正式窗口**前向组（telemetry `forwards` 求和 1,143），
  分子 = 快路径组数 − 预热贡献（`iterations × (gen_len-1)` = 116）⇒ **6.74%**；
  缺 `measured_window` 时**标注来源**（`derived(warmup iters*(gen_len-1))`）。
* **①d 有效样本**：末尾**连续** `decode==1` 的段（`tail_single_decode`）= 77 步；
  尾段 `pack` 总量 3 轮中位数：**109.02 → 187.52 ms（+72.0%）**、
  **108.98 → 187.31（+71.9%）**（逐轮 187.13 / 187.52 / 187.92，极差 0.4%）。
* **①e 孤立长停顿**：单步 `fwd_ms`/`pack_ms`：`s4000_fp16` r0 **16,839 ms**@779、
  `s01_both` r2 **13,865 ms**@993、组合 r0 `pack` **16,961 ms**@360、r2 **19,288 ms**@456；
  常规轮单步上限仅 ~1.34 s（step 0 预填充）。**wall 差额逐轮对上**
  （123.4+13.9=137.3、123.6+16.8=140.4、123.5+17.0=140.5、123.2+19.3=142.5）。

**改写后的结论**（写入 `EXPERIMENTS.md` §3.6 E20、§4.4 A3/A4）：

> **在机制真正被行使的 6.74% 样本上，S04 的 `pack` 变差 72%（≈175× 臂内噪声底）
> ⇒ 真实的负效应，不是"不可分辨"。**
> 机制上也说得通：**原路径遇到等长历史本来就不 padding**，写回只是**切视图**；
> 而快路径每步多做一次**整段 `copy_`**（把 HF 返回的累积 KV 写进常驻缓冲）⇒ 净增搬运。
> **不得**再写成"差异在噪声底内"或"吞吐下降是抽样假象"；
> 服务指标因**未解释的单步长停顿**暂不可归因（保留全部有效运行，不删"慢轮"）。

#### (6) 本轮四状态

| 项 | 状态 |
| --- | --- |
| 代码实现 | ✅ 数值缺陷修复 + `tokens_out` 修复 + 分桶/双分母 + 部署校验闭环 + 续跑身份核验 |
| 离线验证 | ✅ 单测 `test_optim_s04_persistent_kv.py` 19 例、`test_exec_opt_plan.py` 67 例、`test_analyze_exec_opt_batch.py` 15 例；★ **改动后全量回归 1535 passed / 1 skipped / 760 subtests / 9m34s**（`--ignore=tests/test_validate_fp16_path.py`）；反例/破坏验证 **6 组** |
| 真机实测 | ❌ **本轮未跑**（按指示"暂不启动新的正式性能批次"） |
| 结论资格 | ⚪ EXECOPT4 的 `pack` 结论**改写为负效应**（且带 A5/A6 两条限制）；**数值正确性与源码身份均未获证明** |

#### (7) 最小真机验证计划（待批准）

1. **前置**：本地全量回归通过 + `stamp` 单独跑到通过（否则不开跑）。
2. **一次小批**（新 tag，如 `EXECOPT5`）：`s01_both` 与 `s01_both_s04_pkv` 两臂 × **5 轮**，
   同一协议（`short_chat`）。目的只有一个：**在带身份绑定的产物上确认
   ①快路径数值正确（逐 token 与基线一致）②尾段 `pack` 仍 +72%**。
3. **判据**（预先冻结）：`n_resyncs > 0`（证明重入路径真的被走到）、
   `tokens_out == 请求数×(1+每请求步数)`、尾段 `pack` 中位仍显著高于对照、
   且 5 轮**无**"超 1 s 步数 > 1"的轮次（否则该轮按 A3 单独标注）。
4. **不做**：不扫容量、不扩混长实现、不进图捕获。

#### (8) 远端连接恢复（2026-10-02，**端口第二次变更**）

**(a) 现象与分诊。** 上一轮末远端报 `Error reading SSH protocol banner`，
第一眼像"网络不可达"。`scripts/diag_remote_reachability.py` 给出**否定证据**：

| 条目 | 结果 | 含义 |
| --- | --- | --- |
| **目标端口** | **EOF（对端主动关闭）** | 主机在线、**该端口没有服务** |
| 对照 A：同主机 :9 | EOF | "主机在线"的基线 |
| 对照 B：同类路由 SSH | BANNER | **线路本身正常** |
| 对照 D：通用出网 HTTP | TIMEOUT_AFTER_CONNECT | 隧道假活 |

⇒ **不是网络问题，是映射端口变了**。这与 2026-09-24（换镜像 `14671 → 14293`）
是同族现象的**第二次实证** ⇒ "重启/迁移会改端口"从推断升级为规律。

**(b) 处置与再验证（两步，缺一不可）。** 用户从控制台取到新端口 **23576**；
只改 **`~/.workbuddy/remote.env`（仓库外）**：

* **第 1 步 banner**：新端口 `SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.11`（旧值 28080 现 EOF）。
* **第 2 步认证**（banner ≠ 认证通过）：`remote_run.py exec "hostname; whoami"` ⇒
  `autodl-container-05264bbf0d-3b9c3488` / `root`，`/root/ascend_probe_d1` 仍在。

**(c) 凭据卫生（已核）。** `git ls-files | grep -i env` 为空、
`.gitignore` 第 56–59 行已设防（`remote.env` / `*.remote.env`）、
仓库内无副本；`MEMORY.md` 与 `PROJECT_MAP.md` **只指向 `remote.env`、不复述端口**
⇒ 端口变更**无需**改任何文档（单一真相源纪律生效）。

**(d) 远端现状。** **纯部署目录、无 `.git`**（符合设计）。现存戳
`deployed_at = 2026-09-30T19:51:25+0800`、`git_commit = 16e82d2e`、`git_dirty = true`
⇒ 与本地 HEAD（`16e82d2`）一致 ⇒ **上轮确实没推送成功**，远端尚无本轮的五处修复。

**(e) 推送与打戳（本轮实际执行）。**

* **首次推送**：`ok=87 skip=120 fail=0（共 207）`、耗时 37 min。
* ⚠️ **首次推送的 `--include` 静默失效**（见 (f)）⇒ 实际只推了默认的 `py,sh`，
  **全部 `.yaml/.yml/.json` 未进清单**。发现路径：核 configs 整目录时**文件数**
  `远端 138 vs 本地 144`（差 6，全在 `configs/mthreads/phase1_step15_v5/`）。
* **补推**：用**正确顺序**重推 `configs` ⇒ `144/144` 逐文件 sha 一致。
* **`stamp`**：`投放并校验通过`、`rc=0`，远端以 `/root/miniconda3/bin/python3 3.10.8`
  重算摘要相符；`tree_digest = 2b79ff65f275…`、`commit=16e82d2ea117`、`dirty=True`。
* **终验**（`digest_scope` 三目录逐文件）：`src/ascend_opt` 19/19、`benchmark` 11/11、
  `scripts` 77/77 —— **`diff=0`**。
* ⚠️ 过程中一次**并发写者**失误：后台推送脚本运行期间，我用 Python 改写了
  `scripts/push_files.sh`（破坏验证），bash 逐行读脚本 ⇒ 报
  `line 116: syntax error near unexpected token '('`。**后果已核**：149 项
  均在报错前处理完（全部 `SKIP`），且事后文件数/逐文件 sha 均一致 ⇒ **未损坏产物**。

**(f) ★ 发现并修掉 `push_files.sh` 的 `--include` 静默降级（新增守卫 + 回归）。**

`--include` 的解析循环（`push_files.sh:28-34`）在**第一个位置参数**处 `break`。
若把 `--include` 写在位置参数**之后**（我本轮就是这么写的），它会
**留在 `"$@"` 里被当成路径** ⇒ 打出「跳过（不存在）：--include」⇒
`INCLUDE` **静默退回默认 `py,sh`** ⇒ 配置类文件一个都不推，而输出是
`ok/skip/fail` 正常、**退出码 0**。

⇒ 这正是 PLAYBOOK §1 那条「**代码是新的、配置是旧的**」的**实际成因**，
也是「**静默降级成默认值**」这一失效模式的具体形态。

**修法**：位置参数里若出现 `--include` / `--include=…` 形态 ⇒
**明确报错 + `exit 2`**（不再静默降级）。
**回归**：`tests/test_exec_opt_plan.py::TestPushIncludePositionGuard`（4 例）——
两条错序（分离形态 / `=` 形态）必须 `rc=2`；两条**正向对照**
（不写 `--include` 时退回默认是**设计行为**、正确顺序不得触发守卫）证明守卫**有判别力**。
**破坏验证**：整段删掉守卫 ⇒ 2 条错序用例变红、2 条正向对照**保持绿**（未误伤）；
从**同一份备份**还原后 sha 逐字节一致（`e9fd4465d8ea7ccb`）、LF 干净。

**(g) 收尾状态。** 已按序完成"先推送 → 再让 `stamp` 独立跑到通过"；
**真机批次仍未启动**（须待用户批准 §5.9.9(7) 的最小验证计划）。
⚠️ 端口/provider 变更**不改变实验条件**（同实例同镜像），
但**下一次真机批次仍必须重新打戳**（`--allow-dirty` 会记 `git_dirty=true`，见 A6）。

---

### 5.9.10 用户 6 条修订指令执行轮（2026-10-02 续）

**(a) ★★ 更正：遗漏用例是 13 个，不是 74 个（我上一轮报错了）。**

我上一轮把"pytest 补跑 **74** 项"（`test_plot_step16_decouple`=13、
`test_source_stamp`=61）写进了汇总。**74 是错的**，成因是我自己的触发条件是
`pytest 收集数 > unittest 收集数`：

* pytest 会把 `@pytest.mark.parametrize` **展开成多条 nodeid**，而 unittest 只当
  **一个**测试方法 ⇒ `n_py` **天然偏大**，即使该文件里根本没有 unittest 收不到的用例；
* 实测两类假阳性：`test_source_stamp`（unittest 48 / pytest 61）、
  `test_validate_fp16_path`（23 / 48）—— 它们的**差集是空的**。

**真值（直接核过，非印象）**：

```
tests.test_plot_step16_decouple   unittest_have = 0    pytest_collect = 13
```

⇒ **真正被静默漏掉的只有这 13 个**，正是用户第 1 条点名的数字。
⇒ 判据已改为 **差集非空**（`todo = pytest_nodeids − unittest_have`）才计入补跑，
且**不再打空行**（旧的 `[pytest] 通过 0` 看着像"补跑跑了 0 个"，实际是"没有要补的"）。

**(b) ★ 退出码**不是**无条件权威：区分"收尾噪音"与"收集期崩溃"。**

补跑 `test_plot_step16_decouple` 时 `13 passed`，而 `returncode` 非零 ⇒
旧规则 `if rc != 0 and not (failed or errored): errors = max(errors,1)`
把它强行记成 **`errors=1`**（一个假失败）。

真凶经查是**本机沙箱的 `safe-delete` 垫片**：pytest 结束时清理
`Temp/pytest-of-<user>/garbage-*`，`rm` 被替换成"送入回收站"的实现，
拿到 `\\?\C:\...` 扩展路径后 Windows 报 `Errno 53 找不到网络路径`，
垫片 fail-closed、把一段 JSON 打到 **stdout**、并**返回非零退出码**。
⇒ 这是**环境垫片**行为，**不是**被测代码有问题。

**但守卫不能因此删掉**（它的本意是拦"收集期就崩了"）。两者的**可分辨差异**：
崩溃 ⇒ **没有** `N passed` 汇总行；垫片噪音 ⇒ **有**汇总行且 `N` 与 nodeid 数一致。
⇒ 新判据：有汇总行 **且** `stat.run == len(todo)` ⇒ 不记 error，但**留痕**
（`skip_reasons` 写明"退出码非零来自收尾阶段"），事后可查；
否则**照旧**记 error。回归见 `tests/test_run_tests_harness.py`（7 例，含两条否定测试）。

**(c) ★★ 续跑身份：从"比 commit"升级到"比内容摘要"，三处"缺席即通过"改"缺席即拒绝"。**

1. **发现一个真缺陷**：`read_source_stamp` 内部 `want = stamp["tree_digest"]`
   反复读它做校验，**却从没写进返回值** ⇒ `detect_source_identity` 读
   `stamp.get("tree_digest")` **恒得 `None`**，而 `validated` 仍是 `True`。
   症状 = "戳明明有效，摘要却永远不可用"。已补 `out["tree_digest"] = want`。
2. `ReproducibilityRecord` 新增 `tree_digest` 字段（写端落盘；**不进**
   `REQUIRED_REPRO_FIELDS`，否则所有历史 run 一夜之间"元数据不完整"）。
3. `identity_complaint` 三处**缺席即通过**（旧写法在各侧为空时**短路跳过比对**）：

   | 旧写法 | 错在哪 |
   | --- | --- |
   | `if src is not None and src not in TRUSTED` | 产物没这个字段就放行 |
   | `if snap.is_file() and cfg_file.is_file()` | 少一个就不比 |
   | `if model_revision and got and got != model_revision` | 任一侧为空就不比 |

   三者都把"**测不出**"读成了"**没问题**"。新语义：**核不了 ⇒ 拒绝复用**。
4. **锚定真实 baseline 路径**：`cfg_path_of["baseline"] = baseline_path`。
   旧写法 `cfg_dir / f"{name}.yaml"` 对 baseline 臂指向**不存在的文件**
   ⇒ 那个 `is_file()` 恒假 ⇒ **配置身份从来不比**。
5. **破坏验证**（7 条守卫逐条短路，对应 case 全部从 REJECT 变 PASS）：
   源码身份缺 / 本机摘要测不出 / 产物无摘要 / 摘要漂移 / 缺快照 /
   当前配置不存在 / 模型 revision 缺失 —— **7/7 对得上**。
6. ★ **边界**：`git_source == "git"`（真 git 树）在**摘要**这一关**必然拒绝**
   （`detect_source_identity` 对 git 树刻意不给摘要）——这是**有意的保守**，
   已用 `test_git_source_without_digest_is_conservatively_rejected` 冻结。

**(d) ★ 新增真机验证脚本**（第 4 项交付物，⏳ 待执行）：
`scripts/verify_s04_reenact_numerics.py` —— **独立**于单测，自己驱动 `RealBackend`
造"快路径 → 混长回退 → 成员退出 → **重入**"序列，7 条判据**预先冻结**，
并显式声明 `does_not_assert`（不含性能、不含过载/SLO）。
**本地微型模型已验证场景本身**能造出
`fastpath=7 / fallback=5 / resyncs=2`（`unequal_kv_len×2`）、
`tokens equal=True`、逐层 KV `same=True`、`tokens_out 23==23`。

**(e) 修掉 `run_tests.py` 的性能回退**：`_pytest_collect_count` 与
`_run_pytest_subset` 原先**各起一次** `--collect-only` ⇒ 每模块 2 次 pytest 启动，
`--group logic` 从 ~7 min 涨到 >20 min。已合并为一次（`_pytest_collect_nodeids`
同时返回收集数与 nodeid 列表）。

**真机批次仍未启动**（须待用户批准第 4、5 项）。




### 5.9.11 重推 + 重打戳 + 破坏性对照（2026-10-02 续·核实轮）

**(a) ★★ 远端源码树确已过期，且差集是"按内容"算出来的。**

第 1–3 项改动落在 4 个文件、又新增 1 个 ⇒ 部署戳的 `tree_digest` 必变。
用 `iter_scope_files()` 的 **`(rel, sha)`** 两边对拉（**不是**比文件名、也不是比目录清单），
得**精确差集**：

| 类别 | 数 | 文件 |
| --- | --- | --- |
| 只在本地 | 1 | `scripts/verify_s04_reenact_numerics.py` |
| 内容不同 | 4 | `push_files.sh`、`run_exec_opt_matrix.py`、`run_tests.py`、`provenance.py` |
| 只在远端 | 0 | — |

漂移实测：远端旧戳 `2b79ff65f27534a9…` ≠ 本机 `af3ec669d0678b4e…`。

处置：先 `push_files.sh --dry-run`（rc=0、**未建立任何连接**）核清单 → 真推
**5/5 OK**（成功判据 = 远端 sha256 == 本机 sha256，**不是**"put 返回 0"）
→ `remote_run.py stamp /root/ascend_probe_d1 --allow-dirty` ⇒ `投放并校验通过`，
读回 `tree_digest = af3ec669d0678b4ed34ad5cf97b61c205812dbedf998bb7d9df50000fbd13af9`
= **本机值**。两侧对齐。

**(b) ★ `read_source_stamp` 的修复在真机上得到证实（不是靠本机单测推的）。**

远端用远端解释器读真实的戳：

```
detect_source_identity('/root/ascend_probe_d1')
= ('16e82d2e…', 'stamp', True, 'af3ec669d0678b4ed34ad5cf97b61c205812dbedf998bb7d9df50000fbd13af9')
```

`source='stamp'` 且**第 4 位真的带上了摘要** —— 修复前这一位恒为 `None`，
而 `validated` 照样是 `True`（正是 §5.9.10(c) 记的那个缺陷）。

**(c) 真实 EXECOPT4 产物按新判据仍被拒绝 —— 这是设计目标，不是故障。**

`results/raw/EXECOPT4-short_chat-s01_both{,_s04_pkv}` 的 `meta.json` 里
`git_source='stamp_stale'`、`git_commit='NOT_MEASURED'`、`tree_digest=None`。
**两种**输入（`source_digest=None` 与 `source_digest=活摘要`）**都**返回
`源码身份不可信（git_source='stamp_stale'…）`
⇒ 这批产物**不得复用**；第 4、5 项必须跑**新 tag**。

⚠️ **必须写明的边界**：`resolve_source_digest(_ROOT)` 读的是**本机**戳，
而**本机没有戳**（戳只投到远端、且不入库）⇒ 在本机跑 `run_exec_opt_matrix`
会对**一切**既有产物拒绝。**这不是缺陷**（该脚本的用法就是远端），
但若不在报告里写明，下一轮会被读成"判据坏了"。

**(d) ★★ 第 2 项的"不联网"是**证过**的 —— 用破坏性对照。**

`test_dry_run_never_touches_network` 的判据取**可观测外部事实**：
把远端主机给成 `//192.0.2.1:1`（TEST-NET-1，RFC 5737，**保证不可路由**），
要求 ① `rc=0`、② 输出含"未建立任何连接"、③ **墙钟远小于** 20 s。
dry-run 实测 **7.9 s 内两项全过**（与正向对照 `test_missing_include_falls_back_to_default_silently` 同批）。

**破坏性对照（本轮新做）**：把 `push_files.sh` 里 dry-run 的早退条件
从 `-eq 1` 改成 `-eq 2`（即**永不成立**）⇒ 脚本会真的走下去建连。
结果：该测试**变红**，`subprocess.TimeoutExpired`（**60 s 超时，卡在真实的 TCP connect**）
⇒ 证明这条测试**真的在区分**"没建连接"与"停得早"，而不是"进程停得快就算过"。
还原后 `diff` 为空、`TestPushIncludePositionGuard` **6/6 绿**。

**(e) 第 4/5 项的前置已逐条核实（不是"应该没问题"）。**

* `scripts/verify_s04_reenact_numerics.py`：AST 解析 + 实际 import 均 OK；
* 目标配置 `configs/mthreads/exec_opt/s01_both_s04_pkv.yaml` 存在；与
  `s01_both.yaml` 的差集 = **仅** `runtime.persistent_kv_slots: true`
  （`s01_both.yaml` **没有**这个键 ⇒ 默认 false）⇒ **单变量成立**；
* **模型就绪**：远端 `snapshots/a09a35458c702b33eeacc393d103063234e8bc28` 存在，
  4 个 `safetensors` 分片齐全；两份 exec_opt 配置钉死的 `model_revision`
  与该 snapshot **逐字符相同**。

⚠️ **我自己的一次误判（登记）**：一度把缓存 revision 抄成 `…d103…`、与配置
`…d043…` 不符，据此判 `REV_MISSING`。从 yaml **精确读出**后确认两者**都是**
`a09a35458c702b33eeacc393d103063234e8bc28` ⇒ 是**我的转录错误**，不是环境问题。
教训同 PLAYBOOK「字段名/取值必须现读，不能凭记忆」。

**(f) 本机测试结果（本轮实跑）。**

| 目标 | 结果 |
| --- | --- |
| `tests/test_exec_opt_plan.py`（含全部续跑守卫 + push 守卫） | **82 passed, 4 subtests passed, RC=0** |
| `tests/test_run_tests_harness.py`（第 1 项的自守卫） | **7 passed** |
| `TestPushIncludePositionGuard`（单独） | **6 passed**（破坏性对照后仍全绿） |
| `tests/test_batch_lock.py`（隔离，干净负载） | **42 passed, 20 subtests passed** |

⚠️ **上一轮 `_regr4.txt` 的内容是"没有找到 S14S-* 产物"** —— 成因是我给的重跑命令
**漏掉了 `scripts/run_tests.py` 路径参数**，于是它退回了默认的 S14 扫描路径。
**这不构成任何 `test_batch_lock` flake 证据**，已在 §5.9.10 的对照里更正。

**(g) 真机批次仍未启动**（须待用户批准第 4、5 项）。⛔ 不得开始 OPT-04。

**(h) ★★ 第 1 项的退出码判据发现**第二形态**（`--group all` 实测，本轮修复）。**

`_regr3` 那次 `13 passed` + rc≠0 被我的第一条判据（"**有** `N passed` 汇总行"）挡住，
看起来已经修好。但本轮 `--group all`（干净负载、18 min 跑完）**又出现同一模块报错**，
且形态**不同**：

```
[逻辑] tests.test_plot_step16_decouple              通过   0  ...  <== 未执行任何用例
[pytest] tests.test_plot_step16_decouple [pytest]   通过   0  失败 0  错误 1  <== 失败
     1 × pytest 退出码 1 但未数出失败：.............
```

关键在最后那行：捕获到的输出**只有进度点 `.............`（13 个）**，
**`13 passed` 那行根本没打出来**。成因：`-q` 下汇总行与收尾是同帧输出的，
沙箱 `safe-delete` 垫片一崩就**整段丢** ⇒ `has_summary=False` ⇒ 落进 else
⇒ **13 个通过又被记成 `errors=1`**，并且因为 `stat.run` 仍是 0，
下游"非数值模块被执行数为 0"的守卫又把它读成"整体跳过"（**两个**假失败）。

⇒ **修法**（`scripts/run_tests.py::_run_pytest_subset`）：补第二判据 ——
**进度点覆盖了全部待跑 nodeid** 同样等价于"都跑到结果了"，并把 `stat.run` 补记。
这**不是**放宽：**收集期崩溃时一个点都不会有**（pytest 还没进到执行阶段）。

逐形态验证（5 种，含 2 条否定）：

| 形态 | dots | 有汇总行 | 判"都跑到" | 期望 |
| --- | --- | --- | --- | --- |
| 点盖满 + 无汇总行（**本次真实**） | 13 | 否 | ✔ | ✔ |
| 有汇总行 | 0 | 是 | ✔ | ✔ |
| 点不满（8/13） | 8 | 否 | ✘ | ✘ |
| 完全无输出（收集崩） | 0 | 否 | ✘ | ✘ |
| 点盖满 + 带 `[ 82%]` | 15 | 否 | ✔ | ✔ |

**(i) ★★★ 第三形态：**汇总行与进度点可以同时被摧毁** ⇒ 停止修补解析，改换证据源。**
（2026-10-02 第五轮）

在 (h) 里我刚把"进度点"补成第二判据，紧接着 `_regr6`（干净负载、`--group all`）
**同一模块第三次报错**，且这次**连进度点都只剩一行被打断的**：
沙箱 `safe-delete` 垫片不仅吞掉汇总行，还会把自己的 JSON 打到 stdout、
**挤进/截断**进度点那一行。

⇒ 两次都在**同一个**证据族（stdout 解析）里打补丁。**根因是判据与故障同源**：
`-q` 的收尾输出（汇总行）与执行进度（点）**都走 stdout**，而故障恰恰发生在收尾阶段。
继续加正则只会得到"第三形态"。

⇒ **修法（终局）：换证据来源，不是修补解析。** 新增旁路插件
`scripts/_pytest_outcome_plugin.py`：用 pytest **自己的 hook**
（`pytest_runtest_logreport` / `pytest_collection_finish` / `pytest_sessionfinish`）
把逐结局计数写进 `PYTEST_OUTCOME_JSON` 指定的文件。
它与 stdout **完全无关**，且在 `sessionfinish` **立刻落盘**
⇒ 收尾清理怎么崩都不影响已写出的内容。

**三层判据（优先级从高到低）**：

1. **旁路 JSON（主证据）**：`collected == len(todo)` **且** `passed+failed+error+skipped == len(todo)`；
2. 汇总行（回退，`has_summary and stat.run == len(todo)`）；
3. 进度点覆盖（回退，`n_dots >= len(todo)`）。

⚠️ 旁路 JSON 的**自身**也要有否定测试：`collected` 与 todo **不符**时必须判 `error`
（否则"文件存在"就变成新的"缺席即通过"）。见下。

**(j) ★ 最终三轮对照（同机，唯一变量 = 代码版本 + 解释器）。**

| 运行 | 代码 | 解释器 | 用例 | 真正执行 | 跳过 | 失败 | 错误 | `EXIT` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `_regr6` | 修复前（仅点判据） | 3.13.12 | 1605 | 1604 | 1 | 0 | **1** | **1** |
| `_regr7` | 修复后（旁路主证据） | **venv** | 1612 | 1524 | **88** | 0 | 0 | **1** |
| `_regr8` | 修复后（旁路主证据） | 3.13.12 | **1620** | **1619** | 1 | **0** | **0** | **0** |

`_regr8` 的收尾两行（**这就是第 1 项的验收证据**）：
```
用例 1620｜真正执行 1619｜跳过 1｜失败 0｜错误 0
✅ OK：0 failure / 0 error；纯逻辑模块全部真的执行了
```
并且跳过原因**自己标出了判据来源**：
```
1 × pytest 退出码 1，但 13 个用例全部有结果（判据：旁路 JSON（collected=13，合计=13））
```
⇒ 旁路文件是**这一次判定真正用到的依据**，不是"加了但没用上"。

**(k) ★★ `_regr7` 的 4 个新 `✗` 是**我选错解释器**，不是代码回归（如实登记）。**

形态极具误导性：`tests.test_optim_s01/s02/s03/s04` 从 `_regr6` 的
`通过 12/7/16/19、跳过 0` 翻成 `通过 0、跳过 12/7/16/19`，
其中**四组计数（12/7/16/19）逐个不变，只是"通过"翻成"跳过"**
⇒ 四个模块**同时**发生**同一方向**的变化 ⇒ **系统性信号**，先找共同因，别逐个查代码。

根因：这四个模块用 `@unittest.skipIf(T.torch is None, …)` 门控；
`_regr7` 我用的是 **venv** `envs/default/Scripts/python.exe`（`import torch` ⇒ `ModuleNotFoundError`
⇒ `T.torch is None`）⇒ **合法整体跳过**（跳过数与用例总数完全对上）。
`_regr6`/`_regr8` 用的是 **`versions/3.13.12/python.exe`**（**torch 2.10.0**）⇒ 正常执行。

⇒ **两个解释器的分工**（本机，已入 PLAYBOOK §2 第 21 条）：

| 解释器 | torch | pytest/paramiko | 用途 |
| --- | --- | --- | --- |
| `versions/3.13.12/python.exe` | ✅ 2.10.0 | ❌ | **跑回归**（`run_tests.py`）|
| `envs/default/Scripts/python.exe` | ❌ | ✅ | **远程工具**（`remote_run.py` / `push_files.sh`）|

⇒ ⚠️ 第二层陷阱：`run_tests.py` 的"非数值模块被执行数为 0 ⇒ 硬失败"守卫**本身是对的**，
它抓到的是"**我选错了运行时**"——只是它的报错文案指向**模块**而不是**解释器**。
⇒ **收尾纪律**：汇报"回归通过/失败"时**必须同时报出解释器**；
只看 `EXIT=1` 而不看跳过总数（1 vs 88）会把"自己选错"读成"代码坏了"。
（`run_tests.py` 第 91–97 行**早就写明**了这一点——venv 只装了 paramiko，
用它跑测试会得到大量 skipped/error，**那是选错解释器，不是代码问题**。）

**(l) 第 3 项闭环：本轮重推 + 重打戳（差分只 2 个文件，全部通过）。**

内容 sha256 逐文件比对（清单 114 个）⇒ **精确 2 个文件差**：

| 类别 | 数 | 文件 |
| --- | --- | --- |
| 只在本地 | 1 | `scripts/_pytest_outcome_plugin.py`（新增） |
| 内容不同 | 1 | `scripts/run_tests.py` |
| 只在远端 | 0 | — |

⚠️ `tests/test_run_tests_harness.py` **不在** delta —— `tests/` **不在** `digest_scope`
（scope = `src/ascend_opt` / `benchmark` / `scripts`）⇒ 本就不部署。这是**预期**，不是遗漏。

处置：`push_files.sh` 真推 **2/2 OK**（判据 = 远端 sha256 == 本机 sha256）
⇒ `stamp --allow-dirty` ⇒ `投放并校验通过`（正标记），`digest = 17997a944cb4…`。

* 本机 `compute_tree_digest` = `17997a944cb43bcf3ee157afd877f551cc481c870b579ddd243025702b4d0453`（125 个 scope 文件）
* 远端 `SOURCE_PROVENANCE.json` 的 `tree_digest` = **同一串**
* 远端 `detect_source_identity('/root/ascend_probe_d1')` =
  `('16e82d2e…', 'stamp', True, '17997a944cb4…')` ⇒ **`read_source_stamp` 修复在真机再次复现**
* 远端 `resolve_source_digest` = `17997a944cb4…`

**(m) 第 3 项的反向对照（缺一不可，全部本轮实测）。**

* `no_reproducibility`（无 `git_source` 键）⇒ 「源码身份**缺失**…拒绝复用」✔
* `empty_git_source=''` ⇒ 「源码身份**缺失**…拒绝复用」✔
* `unknown_git_source='guess'` ⇒ 「源码身份不可信」✔
* 真实 `EXECOPT4-short_chat-s01_both` / `-r1` / `-s04_pkv` 三份 ⇒
  `git_source='stamp_stale'`、`git_commit='NOT_MEASURED'`、`tree_digest=None`
  ⇒ **全部拒绝复用** ✔（设计目标）

⚠️ **边界（同 (c)，必须写明）**：`resolve_source_digest(本机 root)` 返 `None`，
因为戳**只在远端**。⇒ 本机跑 `run_exec_opt_matrix` 会对**一切**既有产物拒绝。
**这不是缺陷**（脚本的用法本就是远端），但不写明会被下一轮读成"判据坏了"。

**(n) 第 4/5 项前置（本轮再核一遍，仍全绿）。**

> ⚠️⚠️ **本节已于 2026-10-03 更正（原文保留在下方，便于追溯）。**
>
> 下面这些核查**逐条成立**（文件在、sha 一致、配置单变量、判据已冻结），
> **但它们证明不了一件事：脚本能跑**。2026-10-03 实测第一次真调用即
> `AttributeError: 'dict' object has no attribute 'runtime'`
> （`build_driver` 收到 `yaml.safe_load` 的 dict，而真驱动读 `cfg.runtime`）。
>
> ⇒ 所以本节末尾那句"**前置全部就绪**"是**过度声明**，已撤回。
> **AST/import OK ≠ 能跑**；"文件都在" ≠ "链路通"。
> 更正后的口径：真机验证前，必须先在**离线集成**上跑通
> `--plan-only`（读配置/派生两臂/采集身份/定输出路径/校验单变量性）
> 与判据的正反例（见 `tests/test_verify_s04_cli.py`）。
> 本轮另补了 logits 摘要对照（`capture_logits_digest`，默认关）
> 与"无预热不得宣称预热扣除已验证"的口径判据。

* `scripts/verify_s04_reenact_numerics.py` 本机/远端 sha256 **逐字节相同**
  （`1d15804a9edd7dabe9740c263a1b3de18f2bb8f2d0964bbb49e843dec54a7ff4`）⇒ 部署副本是**当前**版本；
* 远端 AST 解析 OK；MUSA 环境激活正常（`torch 2.2.0`，`MODEL_REVISION=a09a35458c70…`）；
* 单变量性再核：`s01_both_s04_pkv.yaml` 相对 `s01_both.yaml` **只多两行**——
  `persistent_kv_slots: true` + `persistent_kv_capacity: 1024`。
  **后者在 `slots=False` 时不生效**（`benchmark/real_backend.py` 的读取点已核：
  `None if rt.get("persistent_kv_capacity") is None else int(...)`，而
  `_pkv_capacity` 只在 `_persistent_kv_requested` 为真时才被用到）
  ⇒ 两行 = **一个逻辑变量**（"开 PKV"），**不是两个**。
* 7 条判据**预先冻结**在脚本 docstring（含最关键的 `n_resyncs > 0`：
  没有它，脚本等于只验"纯快路径"，而缺陷**只在重入时**出现）。

**(o) 第 6 项全程守住：**`S04` 默认关闭**（`s01_both.yaml` 无 `persistent_kv_slots` 键 ⇒ false）；
本轮**未**扩混长实现、**未**进入图捕获。⛔ 不得开始 OPT-04。**

**(p) 本轮 6 项总判定：见 `NEXT_SESSION_TASKS.md` 的收尾结论。**

对应回归 `tests/test_run_tests_harness.py` 增至 **10 例**（新增 3 例：
第二形态、点不满的否定测试、百分比后缀）。

⚠️ **教训**：第一形态的修法**依赖"汇总行存在"**，而汇总行恰恰是**最脆弱**的那部分
（与崩溃同帧）。**判据不能建立在会被同一故障摧毁的证据上**——
必须另找一条**独立**的、"都跑到结果了"的证据（这里是进度点）。

---

### 5.9.12 用户第 6 条指令：按当前源码复核并修复 6 项（2026-10-03）

**用户口径**：**"未执行或未完成不得写成通过"**。本轮先定位（每个都实测复现）、再改、
再用**破坏性对照**证明判据真的在测东西。

**(a) ★★★ 找到 5 个真实缺陷，第 1 个是"我自己的解药变成了毒"。**

| # | 缺陷 | 实测证据 |
| --- | --- | --- |
| 1 | 旁路插件**漏记 `teardown` 相位**（只记 setup/call）⇒ 配上"计数齐全即豁免非零退出码"，把一个**真实的夹具收尾缺陷**读成通过 | pytest：`1 passed, 1 error`；旧旁路：`error=0` 且"看起来齐全" |
| 2 | `verify_s04_reenact_numerics.py` **根本跑不起来** | 第一次真调用即 `AttributeError: 'dict' object has no attribute 'runtime'` |
| 3 | 固定轮数 `--repeats 5` **只跑 1 轮**，且 **rc=0** | `run_once` 调用 2 次、`n_done=2/10`、`balanced=True`、`failures=[]`、`rc=0` |
| 4 | 配置身份判据**恒判不一致** ⇒ 续跑永远不复用 | 写端重新序列化 vs 读端比原始字节（本仓配置都带注释）|
| 5 | 测试里的裸 `"bash"` 探测 ⇒ **整个 pytest 进程被 SIGTERM** | Windows 解析到被拉黑的 `wsl.exe`；`except (OSError, SubprocessError)` 接不住 |

**(b) 修法与破坏性对照（每条新判据都要能被"改回旧行为"打红）。**

| 缺陷 | 修法 | 破坏性对照 | 结果 |
| --- | --- | --- | --- |
| 1 | 插件逐相位（setup/call/**teardown**）+ 会话级异常（收集失败/internal error）+ `expected`/`reported` nodeid 集合；判据抽成**纯函数** `_classify_pytest_run`：① nodeid **双向对账** ② 逐相位核对 ③ 会话异常 ④ 非零退出码**只在命中已知外部签名**时降级 | 改回"只看 setup" | teardown 用例 **变红**（`0 != 1`）|
| 1 | 删除"计数齐全即豁免" | 改回 `notes.append` | 无法归因的非零退出码用例 **变红** |
| 1 | ★ 自动补跑路径也补跑了 `stat.run`，避免"相位记录缺失"引发下游"整块跳过"假失败 | — | 负例：`collected=7 vs todo 13` ⇒ error |
| 2 | 改 `load_config` + `dataclasses.replace`；补 logits 摘要（`capture_logits_digest`，**默认关**）、**重入时刻** KV、`tokens_out` **三方对账**、预热口径、唯一输出路径 + 拒绝覆盖、四类身份记录、`--plan-only` 离线路径 | —（原为崩溃） | 新增 **23** 例 |
| 3 | `break`→`continue`；新增 `completion_flags`（`n_planned_fixed`/`complete`/`under_run`）；外层编排补 `--baseline`/`--repeats` | `continue`→`break` | 调用数 `2 != 10`、`6 != 2` ⇒ **2 条变红** |
| 4 | 统一为**参数内容的规范摘要** `config_data_digest()`（`sort_keys` + 紧凑分隔符），两侧**同一解析器** | 退回"比原始字节" | 真实写端快照用例 **变红** |
| 5 | 只探明确的 PortableGit 路径（glob 版本目录） | —（环境级） | 全量回归不再被 SIGTERM |

**(c) ★ 我自己的一次误判（自行更正，未改坏代码）。**
先入为主以为 `forward()` 返回内部 `_GroupOutcome`，据此把脚本原来的 `out.slots`
判成第二个缺陷。核对**公开签名**才发现：

    TransformerDriver.forward(batch) -> StepOutcome{slots: list[SlotResult]}

⇒ **原来的写法是对的**。教训：**改之前先核公开签名**——
把"对的东西"改掉会引入**新的**缺陷，而它同样会藏在"更早的地方崩过所以没跑到"后面。

**(d) 离线验收：统一一次全量回归（`_regr9`）。**

```
用例 1698｜真正执行 1697｜跳过 1｜失败 0｜错误 0
✅ OK：0 failure / 0 error；纯逻辑模块全部真的执行了
EXIT=0
```

较 `_regr8`（1620）新增 **78** 例（全部是新增测试，无一处放宽）：
harness 12→21、`test_verify_s04_cli` 23、`test_exec_opt_matrix_main` 22、
`test_config_identity_contract` 20、`TestLogitsDigest` 4。
本轮**不再出现**沙箱垫片造成的非零退出码。

**(e) 真机数值验证：PASS（10/10）。** 详见 `EXPERIMENTS.md` §3.7。
`resyncs=2`、token 23 全同、logits 23 点全同（`sha256` 逐位）、
重入与结束时刻逐层 KV 逐位相同、`tokens_out` 23/23/23。
⚠️ `warmup_deduction_verified=false`——**未预热，故"预热扣除"未被验证**。

**(f) 性能实验：已执行完成 —— `EXECOPT5`（2 臂 × 5 轮 = 10 项，新 tag）。**

两门已满足：① 数值门 PASS；② 完整性已验（远端 `--dry-plan` = 10 项、
真实 `main` 离线模拟 = 10 项、少跑 ⇒ rc=1）。本轮数值门未失败，故未触发"立即停止性能链"。

批次 summary（**第 3 项那套完成度判据在生产里的第一次生效**）：

```
n_items=10  n_done=10  n_planned_fixed=10
complete=true  under_run=false  budget_limited=false
completed_passes=5  balanced=true  n_skipped_existing=0
dry_run=false  failures=[]  elapsed_s=1621.7  avg_item_s=162.2
```

⇒ **10/10、五轮齐、均衡、零失败**；回拉后完整性核对 **10/10 个目录**（4 文件齐）。

**结果（口径 = §4.4 A4）**：

| 口径 | baseline | pkv | 比值 | 5 轮极差 | 同号 |
| --- | --- | --- | --- | --- | --- |
| **尾段 77 步 `pack`Σ（有效样本）** | 108.80 ms | 185.95 ms | **+70.9%**（**逐轮配对比值的中位数**） | 16.3% | **5/5** |
| 全窗口 1,080 步 `pack_tel` | 20,968.8 ms | 21,055.9 ms | +0.366% | 0.321% | 5/5 |
| **时间权重**（尾段/全窗） | **0.5189%** | **0.8831%** | — | — | — |

⇒ **独立复现 §3.6 E20（+72.0%，3 轮）**：本批 5 轮给 **+70.9%**，且多了**逐轮配对同号 5/5**。
**S04 = 限定实现与负载的局部负结果**（*qualified local negative result*）：
在**当前实现**（每步整段 `copy_`）与**当前负载**（`short_chat`/并发 8/64 请求/`eager`/`max_num_seqs=32`）
下，快路径在**能行使**的样本上是**净成本**。**不得外推。**

★ **两个口径的换算**：`6.74%` 是**数量覆盖率**（77/1143），**不是**耗时占比。
尾段的**时间权重只有 ~0.52%**（baseline）—— 因为尾段是单请求纯 decode，
单位步搬运量远低于前段批量步。**加性核对**：`20,968.8 + 77.15 = 21,046.0` vs `21,055.9`
⇒ 残差 **9.9 ms（0.047%）** ⇒ 全窗 87.1 ms 的差里 **88.6% 来自尾段**。
⇒ 正确表述：**成本真实存在，但只占总耗时的约 0.9%**（**不是**"被稀释成没有"）。

⚠️ **已删除的表述**：原来写的"`fwd_tel` 两臂区间**重叠** ⇒ 机制只作用在 `pack`"
**已撤回** —— **区间重叠不是检验**，n=5 下推不出"无差异"。
本批只能记"**不足以判定 `fwd`**"；机制解释是**设计意图**，本批数据不能证实。

⚠️ 引用时必须带上：
① `warmup` 轮数两臂**不同**（baseline 2 / pkv 1）⇒ `pack_drv`/`fwd_drv`/`fwd#drv` **不可跨臂比**；
② 每个 run **恰有 1 个 >1 s 的单步**（`max_fwd≈1,310 ms @step0`，**两臂都有**，10/10 轮）
   ⇒ 是**起手步**的固定现象；服务指标引用须带这条；
③ 覆盖率分母是**前向组**（1,143），**逻辑步**是 1,080；
④ **环境证据不完整**（§4.4 A7：产物里的 `hardware_env` 来自另一个容器）
   ⇒ **正式论文引用前必须先解决**。

详见 `EXPERIMENTS.md` §1.7（登记）、§3.8（结论）、§4.4 A4/A7（口径）。

**(g) 用户裁定（2026-10-03）与后续范围。** 原文要点：S04 当前实现**保持默认关闭**、
**停止重复同协议性能批次**、**暂不扩展混长槽位**、**不进入依赖它的图捕获**、
**不为重复确认负结果继续消耗机时**。

| 条目 | 裁定 | 落点 |
| --- | --- | --- |
| S04 当前实现 | **保持默认关闭** | `s01_both.yaml` 无该键 ⇒ false；**不得**改成默认开 |
| 同协议性能批次 | ⛔ **停止重复** | EXECOPT4（3 轮）+ EXECOPT5（5 轮）已**两批独立一致**（+72.0% / +70.9%）|
| 混长槽位扩展 | ⏸️ **暂不扩展** | 属**设计扩张**，需另立裁定 |
| S06 图捕获 | ⏸️ **暂缓**；S06 **先限于计时与同步诊断** | 前置（输入仍动态创建、KV 仍拼接）未全部满足 |
| S07 | ✅ **可继续离线协议设计** | 不需机时 |
| **OPT-04** | ⛔ **仍按其独立前置条件验收**，不因 S04 实验结束而自动放行 | — |
| 论文引用 EXECOPT5 | ⚠️ **先解决环境证据与统计表述问题** | §4.4 A7 + A4 |

**S04 归档定性：限定实现与负载的局部负结果**（*qualified local negative result*）——
只对"当前实现（每步整段 `copy_`）+ 当前负载（`short_chat`/并发 8/64 请求/`eager`/`max_num_seqs=32`）"
成立，**不得外推**。

**(h) 收尾 5 项的处置（本轮）**

1. **环境指纹异机**（§4.4 A7）：核查确认 —— 产物里 `hardware_env` 采集于
   `2026-09-28`、容器 `…cf99469eb6…`，而实跑容器是 `…05264bbf0d…`。
   **同期已有证据**：批次预检 `2026-10-03 03:46:58` 的实时值（torch/MUSA/TF/设备数/REV/
   HF_HOME/计时稳定性/容器可用 CPU）。**仍缺失**：该容器的 MUSA driver 版本、OS/kernel、
   CPU 型号（**登记为"测不到"，不回填**）。原始产物**保持原样**。
   **修法已落地**：批次前置新增**重采环境指纹 + 校验容器/设备身份**
   （指纹主机必须==实跑主机，否则 `StepFailed` 禁止起跑；换容器允许但记 `container_changed`）；
   并把 `plan`/`run` 的步骤序列**合并为一张表**（原先两处各写一遍，新步骤一度只出现在 `run` 里）。
2. **结论口径**：`6.74%` 明确为**数量覆盖率**，换算改用**时间权重**（baseline 0.5189% /
   pkv 0.8831%）＋**加性核对**（残差 0.047%，88.6% 来自尾段）；比值标注为
   **逐轮配对比值的中位数**；**删除**"`fwd` 区间重叠 ⇒ 无差异/只影响 `pack`"；
   归档为**局部负结果**。
3. **两个残余漏洞已修 + 真实反例**：① 运行器原来**只读进程退出码、从不读**旁路文件里的
   `exitstatus`（pytest 自报）⇒ 现在**两处必须一致**，缺失或不一致都判失败（3 条反例）；
   ② 数值对照的**共同缺失**：两侧都空时 `{} == {}` 恒真 ⇒ `_compare_kv` 现在要求
   **非空且层列表非空**（token 同理）；logits 早已如此。
   另：数值产物新增**轻量摘要**（token/logits/KV 的 sha256+标量，**不存张量**），
   使事后复核**不必重跑真机**。`STATE.next_action` 已写明**不要重复执行已完成任务**。
4. **下一阶段优先复核 S01 的实际部署收益**（⚠️ EXECOPT5 两臂**都开着 S01** ⇒ 构造上
   不能证明 S01 自身收益）：预注册见 `NEXT_SESSION_TASKS.md` —— 原始基线 vs S01、
   可信环境＋可信源码记录、两臂等轮、**报服务指标**并同时报每轮 >1 s 起手步数。
5. S06/S07/OPT-04 的边界见上表。
