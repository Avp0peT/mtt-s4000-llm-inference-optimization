# 证据索引（EVIDENCE INDEX）

> **用途**：把论文里每一个数字/结论**反向**指到它的证据位置与版本血缘。
> **权威源仍是 `EXPERIMENTS.md`（登记）与 `results/raw/**`（产物）**；
> 本文件只做**索引与血缘分离**，不产生新数字，不复制结论正文。
> 每行都可用"产物目录 → `meta.json`"复核。

## 0. 三条读法（先看，避免误用）

1. **血缘必须分开看**：不同 `git_commit` / 不同成本模型 / 不同负载的批次**不得并表**。
   见 §2 血缘表。论文中"v5"与"v7"的结果**分开呈现**。
2. **负载区不同则结论不可比**：`ramp`/`lenmix` 在**过载区**（全 run 达标率 ≈0.1~0.23）；
   `c2_m4` 在 **SLO 可达区**（达标率 = 1.000）。前者答"容量边界 + 权衡形状"，
   后者答"服务收益"。**两类结论不得互相代替。**
   ⚠️ **`ramp` 内部是分段的**：`low` 段可达（0.900）、`high` 段崩塌（0.016）、
   `recover` 段部分恢复（0.526）。**"过载"是对整 run 的粗描述，不要抹平分段结构**——
   区分度恰恰在分段上（见 §1.5）。
3. **缺 ≠ 0**：产物里没有的字段写 `-`（缺失），不写 0（测量值）。
   例：`static`/`threshold` 臂的 `arm_state` 是 `None`（该字段只对带策略状态者产出），
   **不得**读成"回退 0 次"。

## 1. 主体平台（MTT S4000 / MUSA）

### 1.1 机制诊断档

| 结论/数字 | 证据位置 | 血缘（commit / 模型） | 负载 | 备注 |
| --- | --- | --- | --- | --- |
| 一步成本分解：`130.52 ms = 前向 104.79 + host 25.25` | `EXPERIMENTS.md` §7c；产物 `S14GV-short_chat-fixed_cost_pilot` | `fb297b69` / `model_v7` | `fixed` c16/m8, prompt1536 | **1 轮，仅诊断**（<3 重复不出结论） |
| TTFT 的 93.8% 是**排队**不是算得慢 | 同上（`queue=10687` / `prefill=701`） | 同上 | 同上 | 用于**设计** S14S，不作结论 |
| 纯 decode 门槛在 `running ≤ 4` | 同上（逐 `running` 分档表） | 同上 | 同上 | 精度不足（每档仅 3 步）⇒ 须真机扫描 |
| `signal_criterion` 冻结值 = `decode_p90` | `EXPERIMENTS.md` 第 4 步 §2（判据 v2 冻结） | — | — | ✅ **三个 `enabled=true` 臂实测均为 `decode_p90`**（19 run 核验；此前"错配"记录**已撤回**，见 §3 缺口 1） |

### 1.2 负载可达性档（`S14S`，24 次运行全 MEASURED）

| 结论 | 证据位置 | 血缘 | 备注 |
| --- | --- | --- | --- |
| `c2_m4` / `c2_m8` 三臂可达（64/64 × 3 轮） | `EXPERIMENTS.md` §第5步 §1 | `8eaf191c` / `model_v7` | |
| `c4_*` 时延达标但达标率塌到 1/64 | 同上 §2b | 同上 | 情形二为真（当时被漏报，已补正） |
| 可达组合冻结 = `concurrency=2`/`max_num_seqs=4` | 同上 §2（机械套用预注册规则） | 同上 | 复算：`scripts/analyze_scan_batch.py results/raw --tag S14S` |

### 1.3 服务收益档（`S6`，9 次运行全 MEASURED）★

| 结论 | 证据位置 | 血缘 | 备注 |
| --- | --- | --- | --- |
| **判定 FAIL：`cost` 劣于 `static` −14.14%** | `EXPERIMENTS.md` §第6步 §3；`scripts/s6_benefit_gate.py`（唯一实现，rc=1=FAIL） | `25c5ab54` / `model_v7` | 效应 = 噪声带 575 倍 |
| 因果链：`chunk_effective` 1024→128 ⇒ 步数 +20.5% ⇒ queue p50 0→127.2 ms ⇒ TTFT ×1.96 ⇒ goodput −14.14% | 同上 §4 | 同上 | **全项目最完整的机制→代价链** |
| 该负载三臂达标率全 1.000 | 同上 §3 | 同上 | ⇒ `cost` 是**主动放弃吞吐**，非"救回超标" |
| ❌ 不得说"cost 有收益"；❌ 不得说"threshold 有效"（+0.05% 不可区分） | 同上 §5 | 同上 | 边界条款，照抄 |
| 登记行 `EXP-S6-001 … 009` | `EXPERIMENTS.md` §第6步 末表 | 同上 | 9 行 MEASURED |

### 1.4 第 14 步四臂（`S14B2` `ramp` · **已完成 20/20** / `S14B3` `lenmix` · **运行中**）

> **`S14B2` 状态复核（2026-09-28 06:14）**：远端 real 目录（`meta.json` 非空）= **20**，
> 分臂 **static 5 / threshold 5 / cost 5 / slo_weave 5** ⇒ 四臂 × 5 重复齐。
> ⚠️ 目录命名是 `S14B2-load_ramp-ramp_<arm>[-rN]`（**连字符**，非 `S14B2_*`）。
>
> **`S14B3`（lenmix）实时复核（2026-09-28 07:51，只读远端复采，未复用 06:14 的观察）**：
> * 队列脚本 `run_auto_queue.sh` PID = **343395**（`etime 01:39:22`，PPID=1 已脱离）。
>   ⚠️ 此前记的 `343438` **不是**队列脚本 PID ⇒ 以后引用 PID 一律**当场复采**。
> * 日志 `/root/autoq/queue.log.b3`（mtime **07:24** = 最后一次 run 完成）。
> * **计划 20 项、pass-major 顺序**：`pass 0 = static → threshold → cost → slo_weave`，
>   再 `r1…r4` 各一遍（读 `queue.log.b3` 第 8–27 行）⇒ 与命名
>   `S14B3-mixed-lenmix_<arm>-r{0..4}` 一致。
> * 已完成 3 项（目录 mtime：`static` 06:36 / `threshold` 07:00 / `cost` 07:24）
>   ⇒ **节拍 ≈ 24 min/run** ⇒ 20 run ≈ **8.0 h**，起跑 06:14 ⇒ 结束 **≈ 14:15**
>   （与任务书 8 h 新机时上限锚点 ≈14:14 **几乎重合** ⇒ 本夜不再起新批次）。
>   ⚠️ 节拍由**目录 mtime**量得（含起跑/落盘开销），比早期按两个最快 run 的 `wall_ms`
>   推得的 21 min **更保守**；只按最快臂估会**系统性乐观**。

| 结论 | 证据位置 | 血缘 | 状态 |
| --- | --- | --- | --- |
| 跨重复稳健 CV **0.0000–0.0009** ⇒ 数字可信 | 远端探针 `scripts/autoq/remote_probe_s14b2.py` | `7051ea62` / `model_v5` | 中途核对（15 格时） |
| 四臂**全 run 级**达标率 0.098–0.228，**无区分度** | 同上 | 同上 | ⚠️ 全 run 口径；**逐段口径有区分度，见下** |
| 方向非单向：`static` goodput 最高但 stall P99 最差（615.6）；`cost` 反之（313.4） | 同上 | 同上 | **是权衡，不是胜负** |
| `cost` 降级分解：`n_fallback=4~5, n_degraded=150~151, n_starved=114` | 同上（读 `controller_effect.arm_state`） | 同上 | 真机证实"**并非**结构上无法发 prefill" |
| 历史对照 `S14B1`（20 次） | `EXPERIMENTS.md` §S14B1 | `d20727d5` / 无模型 | 旧执行版本，仅历史参考 |

### 1.5 ★ 逐段 SLO 证据（`S14B2`，18 个真 run）—— **负载变化现在测到了**

> 负载结构：`low(0–120s)` → `high(120–210s)` → `recover(210–450s)`；恢复判据
> `queue_depth<1` 连续保持 5s。**逐段达标率由项目自身 `SLO.satisfied_by` 现场派生**
> （`requests.jsonl` 不自带该字段，见 §3 缺口 9）。
> 复算：`python scripts/autoq/remote_recovery_s14b2.py`（远端，只读）。

| 臂 | 控制器 | chunk 实际区间 | low | **high** | **recover** | drain goodput |
| --- | --- | --- | --- | --- | --- | --- |
| **`static`** | **`enabled=false`**（开环） | 1024 **恒定** | 0.900 | **0.016** | **0.526** | 37.2 |
| `threshold` | `enabled=true` | [512,2048] | 0.900 | 0.016 | 0.421 | 36.3 |
| `cost`(v5) | `enabled=true` | [512,2048] | 0.900 | **0.000** | 0.421 | 33.3 |
| `slo_weave` | `enabled=true` | [512,2048] | 0.800 | 0.016 | **0.000** | **0.000** |

| 结论 | 证据位置 | 血缘 | 备注 |
| --- | --- | --- | --- |
| ★ **`static` 两段都最好 ⇒ 三个闭环控制器均未胜过开环静态** | 上表 | `7051ea62` / `model_v5` | 与 `S6` 负结果**方向一致**，但**另一血缘**（跨血缘互证，**非**混算） |
| ★ `cost` 在 high 段 **0.000 < threshold 0.016** | 上表 | 同上 | 收缩到 512 + `n_degraded≈150` ⇒ 代价大于收益 |
| ★ `slo_weave` **恢复段恒 0.000**、q50 保持 19.0 ⇒ **未恢复** | 上表 | 上表 | 队列排空 ≠ 达标率恢复，两者分开报 |
| ★ **"负载变化测不到"的旧结论被推翻** | 上表（0.900→0.016→0.526） | 同上 | 轨迹确实穿过 可达 → 崩塌 → 部分恢复 |
| 控制器动作：`cost` 加码 40 / 恢复 0 / 收缩 52（`threshold` 45/0/60，`slo_weave` 32/0/36） | 同上（按 `reason` 分类） | 同上 | ⚠️ **case C（恢复到初值）一次未发生**；峰值 > init 是合法 case B |
| 同臂跨重复**逐段值完全一致** | 同上（18→19 run 逐个打印） | 同上 | 数据可信 |

### 1.6 逐运行身份核验（`S14B2`，19 run × 28 维度）

| 结论 | 证据位置 | 备注 |
| --- | --- | --- |
| ✅ **全部唯一性维度一致、臂内配置一致** | `scripts/autoq/remote_identity_s14b2.py`（远端只读） | `2026-09-28.md` §23 |
| 标签 `MEASURED` ×19；`git_dirty=False`；`git_source=stamp` | 同上 | 与 v5 冻结方案一致 |
| 冻结量实测：SLO `2000/100.0`、`target_tpot=120.0`、`hysteresis=0.08`、`max_num_seqs=8` | 同上 | **既定前提已生效** |
| 环境：py3.10.8 / torch2.2.0 / transformers4.46.3 / driver2.7.0 / `backend=real` | 同上 | |
| ⚠️ `musa_version=None`、`vllm_version=None` ⇒ **MUSA 运行时版本未采到** | 同上 | **缺失，不是 0**；论文若引需补采 |

### 1.7 ★ `ramp` 的跨重复**确定性**（`S14B2`，n=5 全列）—— P15 的**方法学否定**素材

> 远端只读核验（逐目录读 `meta.json` 的 `summary` + `controller_effect`）。
> ⚠️ 这是**同一批 20 个 run** 的另一种切法（按臂看重复），不是新数据。

| 臂 | 达标吞吐 tok/s（r0/r1/r2/r3/r4） | 极差 | `n_steps` | `dec` |
| --- | --- | --- | --- | --- |
| `static` | 5.928 ×5 | **0.0** | 3736 ×5 | 0（控制器关） |
| `cost` | 5.077 ×5 | **0.0** | 3572 ×5 | 894 ×5 |
| `threshold` | 5.362 ×4 + **5.363**(r4) | 0.001 | 3570 ×5 | 447 ×5 |
| `slo_weave` | 2.239 ×3 + **2.481**(r4) | **0.243** | r0–r3 **4886**，**仅 r4 4201** | 1222 vs **1052** |

| 结论 | 证据位置 | 血缘 | 状态 |
| --- | --- | --- | --- |
| ★ `ramp` 是**确定性**负载：`static`/`cost` **5 次逐位相同** | 上表 | `7051ea62` | **由 n=5 确证**（原登记只说 n=3 极差 0） |
| ⚠️ **"极差 0" 在 n=5 下不普遍成立**：`threshold` 0.001、`slo_weave` **0.243（+10.8%）** | 上表 | 同上 | **修正既有登记**（原文只对 `static` 成立于 n=3） |
| ★ `slo_weave-r4` 是**控制轨迹分岔**，非噪声：请求数/令牌量/达标数四次**完全相同**（92/188416/11776/9），差异**只在 `n_steps`**（4201 vs 4886）与 chunk 终值（1 vs 4） | 上表 + 逐字段比对 | 同上 | **与项目级既有失效模式同型**（adaptive 臂跨重复不可复现） |
| ⇒ **`ramp` 不能提供置信区间**（4 个相同点 + 1 个离群点 ⇒ 任何 CI 由那 1 点决定） | 上表 | 同上 | **方法学否定结论**，非"待跑" |
| ⚠️ `slo_weave-r4` **不得剔除**、不得读作"该臂更好"（2.481 > 2.239 会**看起来像**收益） | — | 同上 | 纪律 |

### 1.7b ★ D0→D1→D2→D3 链（`S14B2` `ramp`）

> ⚠️⚠️ **分母必须按当前实测重算，不得沿用 "16 run"**（`2026-09-28.md` §44）。
> 13:08 逐文件实测：`meta.json` 完好 **10**/20、`telemetry.jsonl` 完好 **11**/20、
> `requests.jsonl` 完好 **19**/20。**本表的分母 = 同时有可用 `meta.json` 与 `telemetry.jsonl` 的 run**。
> 下方表格是**此前（16 run 在位时）**算出的数，**在补齐前引用必须写明"分母已变"**；
> 补齐后须整表重算。⚠️ 缺文件是**执行瓶颈**，不是"这些 run 没测出来"，更不是方法被否证。
> 另：`ramp` 是确定性负载，**不能提供置信区间**（见 §1.7）。

> 复算：`python scripts/diagnose_scheduling.py --tag S14B2`（只读；本地回拉部分即可跑）。
> ⚠️ **本表 n=10（本地可解析的 run 数）**，是 2026-09-28 13:10 实测重算的结果；
> 另 8 个 `meta.json` 被**响亮跳过**（脚本 rc=1，输出中逐条列出截断位置）⇒ 见缺口 8。
> ⇒ 数值随回拉进度会变，**以 20/20 后为准**；当前引用须写明分母 = 10。
> ⚠️ D1 的 deadline 取自配置（轨迹没记 `budget_ms`）⇒ **只作臂间横向比较**（脚本自述）。
> ⚠️ 下列 `n` 是**该臂被成功解析的 run 数**（`cost` 3、`threshold` 3、`static` 2、`slo_weave` 2）。

| 臂 | n | 决策数 | 候选数 | 可行率 | D1 | D2 阻塞步占比 | D2 `Σ队列·步` | **D3 达标吞吐** | D3 达标率 | TTFT 排队中位 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `static` | 2 | 0 | 0 | — | — | **0.00%** | **0** | **5.93** | 0.228 | **21 597** |
| `threshold` | 3 | 0 | 0 | — | — | 23.64% | 13 315 | 5.36 | 0.206 | 25 759 |
| `cost` | 3 | 1 341 | 3 294 | **79.7%** | **2.06%** | 23.46% | 13 186 | 5.08 | 0.196 | 25 993 |
| `slo_weave` | 2 | 2 970 | **0** | — | — | **58.31%** | **48 383.5** | 2.24 | 0.098 | 77 006 |

> ⭐ **臂间横向结论在 n=10 与 n=16 下完全一致**（`static` D2=0%/Σ队列=0 最好、`slo_weave` 最差、
> `cost` 的 D0=79.7%/D1=2.06% 稳定）⇒ **方向不依赖分母**，这提高了该结论的可信度，
> 但**具体数值**（尤其绝对值）**不得**在补齐全 20 个之前写进论文。

> ⚠️ **与 11-run 版（上一版）相比，`static`/`threshold` 的 D0 仍是"0 决策 0 候选"**——
> 因为这两臂的控制器是**停用**的（`kind=threshold` 但 `enabled=false`）⇒ **不产生候选**，
> 这是**设计**，不是"没测到"。**臂间比较 D0 只在 `cost` 与 `slo_weave` 之间有信息量。**

| 结论 | 备注 |
| --- | --- |
| ★ **`static` 队列阻塞恰为 0**（D2 = 0.00%、`Σ队列·步` = 0）⇒ 这是它拿下最高 D3 的**机制性解释** | "开环不分块"在该平台上**不产生排队的等待** |
| ★ **`cost` 的 D0 可行率 79.7%** ⇒ 候选**大量被用上**（不是"没被用上"）；被拒主因 = `预测 182.6 + host 0 = 182.6 ms 超预算 120.0`（×614）与 `模型范围外`（×54） | ⇒ **不得**把 `cost` 的劣化归因为"没被用上" |
| ★ ⚠️ **`slo_weave` 候选数 = 0（2970 次决策一个候选都没出）** —— 这是**该策略的结构性属性**，**不是**"回退"：其 `budget_ms = inf`（`budget_unbounded = true`，见 `telemetry.py:325`）⇒ **无活跃 decode 时不评估可行性**，直接给出 `chunk`，故不产生候选 | ⇒ **不得**读成"它试过了但都不可行" |
| ★ `slo_weave` 的 D2 = **58.31%**、`Σ队列·步` = 48 369 ⇒ 与 §1.5 的"恢复段恒 0.000"互相印证 | 两个独立口径同向 |
| `cost` 的 D1 = 2.06% **远低于** §6.4 记的历史值 84.62% | ⚠️ **口径不同**（`d1_n`、判据版本、负载格均不同）⇒ **不得**并列解读；只作本批内横向 |
| ⚠️ **截断产物的处理**：`diagnose_scheduling.py` 已改为**响亮跳过 + rc=1**（原先会在 `json.loads` 处整脚本崩掉 ⇒ 一个数都出不来，像"脚本坏了"） | 修复属**分析层**、不改推理/调度行为 |

### 1.8 ⚠️ `S14B3` 一个 run 的**溯源中断**（我的操作造成，已修复）

| 结论 | 证据位置 | 备注 |
| --- | --- | --- |
| ⚠️ `S14B3-mixed-lenmix_threshold`（07:00）`git_source=stamp_stale` / `git_commit=NOT_MEASURED`，**却仍 `data_label=MEASURED`** | 远端 `meta.json` `reproducibility` | **标签与溯源自相矛盾** ⇒ 不得当正常带戳结果 |
| 成因：`SOURCE_DIGEST_SCOPE` **含 `scripts/`**；我把自写 `remote_probe_run.py` `put` 进远端 `scripts/` ⇒ 摘要 `1ec18088`→`0df43870` | `provenance.py:563` + 摘要复算 | **违反任务书"不得改跑中批次的源码/戳"** |
| 修复：删该文件后复算摘要 = **`1ec18088…`（与戳逐字相同）** ⇒ 树已恢复，后续 run 正常 | 摘要复算实测 | 判据 = **摘要逐字相等**，不是"文件删了" |
| 受影响范围：**仅 `lenmix_threshold` 一个 run**（`lenmix_static` 06:36 在事故前，正常） | 逐 run `recorded_at` | 纪律见 `MEMORY.md` §16 |



## 2. 血缘表（**禁止跨行并表**）

| 批次 | `git_commit` | 成本模型 | 负载 | 负载区 | n | 用途 |
| --- | --- | --- | --- | --- | --- | --- |
| `S10` | `32f3fac5` | — | 静态扫描 | — | 15 | 静态基线（缺 1536，见 §3） |
| `S14B1` | `d20727d5` | 无 | fixed | 过载 | 20 | 历史参考 |
| `S14D` | `158f83a0` | `model_v5`（cost/slo_weave 臂） | fixed | 过载 | 20 | 第14步机制 |
| **`S14B2`** | **`7051ea62`** | **`model_v5`** | **ramp** | **过载（分段）** | 20 | 第14步（**已完成 20/20**） |
| **`S14B3`** | **`7051ea62`** | **`model_v5`** | **lenmix** | **过载/边界** | 0→20 | 第14步（**运行中**，pid 343438） |
| `S14S` | `8eaf191c` | `model_v7` | fixed 低并发扫描 | **可达** | 24 | 负载可达性 |
| **`S6`** | **`25c5ab54`** | **`model_v7`** | **`c2_m4`** | **可达** | 9 | **服务收益（负结果）** |
| `S14G`/`S14GV` | `9df0aefa`/`fb297b69` | `model_v7` | fixed | 过载 | 6+1 | 机制诊断 |
| 第15步消融 `phase1_step15_v5`（已有 6 格） | 待定 | **`model_v5`** | `fixed`(short_chat) c16/m8 | 待测 | 0 | 参考点 = **第14步 `fixed`**，**非** S6；`--check` rc=0 |
| 第15步消融 `S14K`（"S6 条件匹配"） | — | `model_v7`（同 S6） | `c2_m4` | 可达 | 0 | ⚠️ **不可定义**：下限须由 `pick_d1_min` 规则给出，该规则**无有效输入** ⇒ 见 §3 缺口 11 |

## 3. 已知缺口与限制（不掩盖）

| # | 缺口 | 状态 | 依据 |
| --- | --- | --- | --- |
| 1 | ~~`static`/`threshold` 臂 `signal_criterion` 实为 `tpot_p90`~~ **← 已撤回** | **不存在该缺陷**：三个 `enabled=true` 臂一律 `decode_p90` | 19 run 核验（`2026-09-28.md` §21）；原判系误读**停用**控制器上的默认字段 |
| 1b | ★ `meta.config.controller` 与 `config.snapshot.yaml` **对停用控制器口径不一致**（`static` 臂：meta 有 `kind=threshold`/`tpot_p90`，快照无） | **未修**（改则动 `src/` ⇒ 影响源码戳 ⇒ 须等批次收尾作新版本） | 同上 §21；**"错值比缺值危险"**的实例 |
| 2 | 静态基线缺 `1536` | 未补 | `EXPERIMENTS.md` 第4步 §五（③ 撤回后：应**审计 S10 适用性**而非从零补） |
| 3 | 小 chunk 是否**能被可靠预测**（非查表未命中） | **已验（部分关闭，OPT-03）**：**128/192/256 均可预测**（v7 已采 128/256；v8 补 192 ⇒ `lookup` 有键）。★ 本批表对已采 48 形状 **100% 命中**、会话内重复性中位 **0.01%**、**最严**（跨会话）留出低估 `p95` **0.07%**（阈 5%）、`\|rel\| p90` **0.06%**（阈 10%）；`pf192` 单列同档（0.06%/0.06%）。⚠️ **`384` 仍未采样 ⇒ 缺口未完全关闭**（**缺 ≠ 0**，写为残留缺口）。⚠️ 探针路径（`_probe`/`_costcalib`，**不盖四标签、不进 `EXPERIMENTS.md` 结论表**） | 网格 `configs/cost_calib/shape_grid_v8.yaml`（119 点，`grid_sha256=3473d023e269706f`）；产物 `results/processed/b4_small_chunk_v8.json` + `costcalib_eval_v8.json`；原机采样 `results/raw/_probe/20260928/costcalib_v8_s{1,2}.json`（md5 远端==本地、CR=0）；结论见 `RESEARCH_LOG.md`「OPT-03」段。⚠️ 部署表尚未含 192（`model_v8.json` 未导出） |
| 4 | 候选只取 `min`/`max` ⇒ 改下限**不会**自动产生中间点 | 未改 | `NEXT_SESSION_TASKS.md` §三 |
| 5 | `slo_reach` 配置 | **未建**（其"阻塞"前提已撤回；**缺口 3 已部分关闭**（128/192/256 可预测）⇒ 阻塞条件不再整体成立，但 `384` 尚缺，是否影响 `slo_reach` 需新判）；⚠️ **且受 8 h 机时上限约束**：批次 3 起于 06:14、实测节拍 ≈24 min/run ⇒ 结束 ≈14:15，无余量 ⇒ **本夜不起新批次**（`2026-09-28.md` §35） | 同上；缺口 3 见本表第 3 行 |
| 6 | `budget_ms=Infinity` 写非标准 JSON | 未修（技术债） | `NEXT_SESSION_TASKS.md` |
| 7 | ~~远端证据目录残留 8 个回拉暂存 `.tar`~~ **已复查：`ls /root/.../raw \| grep -c '^_S14'` = 0** ⇒ 残留已不存在（旧记录过期） | **已消解**（无需再清） | `2026-09-28.md` §39 实测 |
| 8 | ★ `S14B2` **本地回拉卡在传输瓶颈**（**11/20 完整**，判据 = `meta.json` **>100 KB**）。远端 20/20 齐全且已核验身份/确定性 ⇒ **卡点是传输，不是数据、不是方法** | ⏸ **今晚不追**。`sftp.get` 在满负载下对 ≥100 KB 文件**挂起**（13 分钟恒 0 字节、不返回不报错）；`exec`+分块 base64 能落盘但只有 **≈24–50 KB/分钟**，而缺口 ≈**28 MB** ⇒ 需 **9–19 小时**，8 h 上限只剩 ~77 分钟 ⇒ **带宽不够，今晚不可能 20/20** | `2026-09-28.md` §29/§36/**§40/§41**。⚠️ 按纪律这是**执行瓶颈 / 不可判**，**不是**负结果、**不是**方法被否证。⭐ **批次 3 收尾后第一件事**：重测一次 2.6 MB 的 `get`，坐实"负载争用"还是"机器固有"（现只有**假设**）。已备 `scripts/pull_s14b2_exec.py`（分块 base64 通道，含单写者锁与原子替换） |
| 9 | ★ **`requests.jsonl` 缺 `slo_satisfied` ⇒ 逐段达标率静默归零** | **未修**（改 `describe()` 会改产物字节 ⇒ 动 `SOURCE_DIGEST_SCOPE` 内的 `src/` ⇒ 须等批次收尾作新版本） | `2026-09-28.md` §18；复现：cohort 0 vs summary 18；契约用例已加（skip 标记） |
| 10 | `WindowReport.n_control_actions` 恒为 0（而 `n_changed`=23）⇒ 该字段未接线 | 未修 | `2026-09-28.md` §19 |
| 11 | ★ **第15步"S6 条件匹配消融"（`S14K`）不存在**：其 `chunk_size_min` 须由「D1② 选取规则」选出，而该规则的输入（`S14H` 三点 `D1`）从未产生 ⇒ `pick_d1_min.json` 不存在 ⇒ **下限不可判（不是 128/256/512 中任何一个）** | **不可定义**（按纪律**不填下限**） | `NEXT_SESSION_TASKS.md` §五（5240 行，已裁定）；`EXPERIMENTS.md:1330` 规则四分支**一条未触发**；实测 `results/processed/pick_d1_min.json` 缺失 |
| 12 | `phase1_step15_v5/`（已有 6 格）**参考点是第14步 `fixed`，不是 S6** | 可用但**口径须写清** | 实测逐行 diff：与 S6 `cost` 格在 `model`/`concurrency`/`max_num_seqs`/`chunk_min` **四处同时不同** ⇒ **不得**用它替代 `S14K` 的结论 |
| 13 | ⚠️ ★ **`S14B3-lenmix_threshold` 的 `git_source=stamp_stale`**（标签 `MEASURED` 与 `git_commit` 缺失冲突） | **已修复成因，run 本身须显式标注** | 本文件 §1.8；`MEMORY.md` §16 |
| 14 | `meta.json` 达 **2.6 MB**（`controller_trace` 783 KB + `context` 386 KB + `system_states` 177 KB）⇒ 回拉慢而易断 | **已用 gz 中转缓解**（2.6 MB→107 KB） | §1.8 / `MEMORY.md` §15；**根治需精简 `meta`**（另开新版本） |
| 15 | ~~`diagnose_scheduling.py` 遇**截断** `meta.json` 会**整脚本崩掉**（一个数都出不来）~~ | **已修**（截断⇒**响亮跳过 + rc=1**，`skipped_truncated_products` 进 JSON 报告；3 个回归用例，**双向验证**已做） | `2026-09-28.md` §37；提交 `c521b79` |
| 16 | ★ **派生字段 `software_env.musa_version` 跨全部 283 份产物恒为 `None`**（原文在 `hardware_env.discovery.values.MUSA_VERSION`，接线断一节） | **未修**（非缺陷性沉默，但引用 SDK 版本须锚 `discovery.values`） | 本日 §34；`MEMORY.md` §17 |
| 17 | 24 份产物 `hardware_env` **整块退化**为 `available_source:"missing"`（18 份 `TEST_DATA`/simulate + 6 份 `NOT_MEASURED` 真机 pilot） | **`MEASURED` 258/258 全带指纹** ⇒ 不影响结论；记为**工具缺口**（真机批次漏跑 `inspect_env.sh` 会静默丢指纹） | 本日 §34 |

| 27 | ★ 本机**沙箱内子进程无法从外部终止**（`os.kill`→`PermissionError`；`taskkill //PID //F`→「拒绝访问」；Win32 `TerminateProcess`→返回 False；WMI `Terminate`→被安全策略拦） | **已用"协作式停止"绕过**：长跑脚本自查 stop 文件（`pull_s14b2_robust.sh` 的 `check_stop()`、`pull_s14b2_exec.py` 的 `STOP`） | 本日 §40。⚠️ **新写的长跑脚本必须自带 stop 文件开关**，否则一旦卡住只能等它自己跑完（实测被卡的 get 子进程会不断新生、越积越多） |
| 28 | ★ **SLOWeave 基线两处口径错误**：① `safety_margin` 被**重复应用**（`CostModel.predict` 默认 `apply_margin=True`，`_predict_ms` 未关掉，`_margin_ms` 又乘一次 ⇒ `(1+m)²`；实测 `model_v7` `m=16.02%` ⇒ 1.1602 变 **1.3461**）；② **稀疏 `lookup` 上做连续二分**（缺键=`未知`被当成"更大者全不可行"；实测 `b1/kv1024` `L=1024`/预算 400 ms ⇒ 给 512（真值 1024）；`L=1536`/预算 320 ms ⇒ 给 **0**，而 `c=512` 只需 194.4 ms） | ✅ **已修**（`src/ascend_opt/slo_weave.py` + `cost_model.lookup_prefill_candidates`）；⚠️ **修复改变该臂判据倍率 ⇒ 既有含 `slo_weave` 臂的产物（`EXP-S14B3-006..010`）不得与修复后并表** | `RESEARCH_LOG.md` 2026-09-29「S00」§三；反例用例 `tests/test_slo_weave_sparse_lookup.py`（含**破坏验证**）；证据 `results/processed/s00_audit.json` |
| 29 | ★★ **执行优化线两个策略语义缺口** —— ✅ **两者均已修（2026-10-03，用户裁定后）** |

**① 双预算忽略受阻 decode**：`_budget_for` 原来只看候选**自己**的 `tokens_prefilled>0`
⇒ 本步还有 decode 在跑（或没被服务）时，那些 decode 也被按 **prefill 预算**（2000 ms，
而 decode 只有 100 ms）放行。**修法**：新增 `decode_at_risk = decode_served or decode_blocked`，
为真 ⇒ **必须**用 decode 预算（逐 token 护栏）；prefill 预算**只在纯 prefill 步**适用。
⚠️ `decode_at_risk` **默认 `True`**（保守）：默认 `False` 会让忘记传参的调用点静默回到旧缺口。
⚠️ **没有改任何外部 SLO 取值**（100/2000 原样），改的是"哪一步该用哪一个"。

**② `closed_jittered` 只改时间戳、请求仍提前入队**：抖动原由 `(seed, req_id, **step**)` 派生
⇒ 到达时刻是**调度的函数**（跑之前无定义）；请求仍在 `step_start` 入队。
**修法**：抖动改由 `(seed, **客户端**, **轮次**)` 派生（两者都是负载自身属性），
并**预先算成一张表**（`closure_jitter_by_request`）；执行器**等到点才放行**。
⚠️ 两条执行路径（模拟器 / 真机）**共用同一份实现**（红线：同一指标必须是同一个量）。

**离线验收（`scripts/audit_s00_blockers.py`，新证据 `results/processed/s00_audit_20261003.json`）**：
`dual_budget.acceptance_pass = true`（**混合步**被 prefill 预算放行数 **0**，`budget_kinds=["decode"]`；
纯 prefill 步仍是 6 —— 那是**按设计**服务 TTFT，旧版把这两个数混为一谈）；
`jitter.acceptance_pass = true`（倒置 **16/16 → 0/16**；`n_distinct_arrivals=16` **反向守卫**
——不许靠"把抖动抹平成 0"来消除倒置；`arrival_span=15743.2 ms`；受理延迟 ≥ +0.07 ms）。
另：**建议动作能落实** —— `apply_arm(kind="cost", shadow=False)` 返回的建议值必须真的进入
下一步的 `plan_step`（`tests/test_s00_known_gaps.py::TestRecommendedActionTakesEffect`），
而 `shadow=True` **只记录不生效**。

**状态：离线验证通过、真机待验。本项不宣称任何服务收益。**
⚠️ 两张旧的特征化用例（断言"缺口仍在"）已按它们自己的约定改为**正确性断言**：
`test_s00_known_gaps.py` 与 `test_closed_loop_jitter.py`。
⚠️ **`closed_jittered` 从未用于任何配置或产物**（`configs/` 与 `results/raw/` 均为 0 命中）
⇒ 本次语义变更**不作废任何历史数据**。 |

## 4. 复算入口（每个数字都应可重算）

| 要算什么 | 命令 |
| --- | --- |
| 第14步四臂对比 | `python scripts/analyze_step14.py --tag <TAG> --high-start-s 120 --high-end-s 210` |
| 负载可达性（S14S） | `python scripts/analyze_scan_batch.py results/raw --tag S14S` |
| 服务收益判定（S6） | `python scripts/s6_benefit_gate.py`（`rc=1` = FAIL） |
| 登记行生成 | `python scripts/emit_experiment_rows.py --tag <TAG>` |
| 批次运行期只读核对 | `python scripts/autoq/remote_probe_s14b2.py`（远端） |
| ★ 跨重复**确定性**核验（§1.7） | 远端只读：逐目录读 `summary`+`controller_effect` 的 `n_steps`/`dec`/`slo_goodput_tok_s` |
| ★ 溯源戳复算（§1.8） | 远端 `python3 -c "import sys;sys.path.insert(0,'./src');from ascend_opt.provenance import compute_tree_digest;print(compute_tree_digest('.'))"` → 与 `SOURCE_PROVENANCE.json` 的 `tree_digest` 逐字比对 |
| ★ 逐段达标/恢复分析 | `python scripts/autoq/remote_recovery_s14b2.py`（远端；**自带 `slo_satisfied` 派生**） |
| 回拉产物 | `python scripts/pull_incremental.py --tag <TAG> [--expect N]` |
| ★ **EXECOPT 批次用时分解**（尾段／非尾段／全窗 + 两个统计口径 + 可加性） | `python scripts/analyze_exec_opt_batch.py --tag EXECOPT5`（**只读** `results/raw/EXECOPT5-*`；段标 `①f`） |
| ★ **批次证据资格复核**（§5 那张表怎么来的） | 逐目录读 `meta.json`：`reproducibility.git_source` / `.tree_digest` / `.hostname_or_machine_id`、`hardware_env.collected_at`；判据见本文件 §5 |
| 契约用例（产物行 ↔ 逐段报告） | `python -m pytest tests/test_stage_report.py -q -rs` |

---

## 5. 批次证据资格清单（**可正式引用／仅作诊断／证据待补**）

> ★ 为什么单独列这一张（2026-10-03）：此前每条结论都写了"引用时须带某某限制"，
> 但没有一个**总表**回答"这一批到底能不能进论文"。于是"带上限制"变成了口头禅，
> 限制不被真正执行。这张表把资格**分级**，让"能不能引用"是一个**可核对的判断**。
>
> **三级的判据（都从产物现读，不靠回忆）**：
> * **T1 可正式引用** —— ① 源码身份闭合（`git_source` 可信**且** `tree_digest` 可复算）；
>   ② **容器身份闭合**（实时 `hostname_or_machine_id` == 该批环境指纹描述的机器）；
>   ③ 判据与口径已冻结（有预注册或明确登记）。
> * **T2 仅作诊断** —— 可用于**机制方向**判断与内部决策，**不得**作为论文的量化主张。
> * **T3 证据待补** —— 关键证据缺失，须重跑或补采后才能使用。
>
> ⚠️ **`data_label=MEASURED` 不等于 T1**。`MEASURED` 只说明"数字是测出来的"，
> 不说明"这批数字绑定了哪份代码、跑在哪台机器上"。下表 **三个真机批次没有一个同时闭合约 ① 与 ②**。

### 5.1 执行优化线（S01–S05 / S04）

产物字段现读结果（`results/raw/<dir>/meta.json` 的 `reproducibility` + `hardware_env`）：

| 批次 | 目录数 | `git_source` | `tree_digest` | `data_label` | 指纹 `collected_at` | 实时 `hostname_or_machine_id` | 环境与容器是否同源 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `EXECOPT3` | 12 | `stamp_stale` | — | `MEASURED` | 2026-09-28 | `…cf99469eb6…/root` | ✅ **同源** |
| `EXECOPT4` | 12 | `stamp_stale` | — | `MEASURED` | 2026-09-28 | `…cf99469eb6…/root` | ✅ **同源** |
| `EXECOPT5` | 10 | **`stamp`** | **`bff7f7080b67…`** | `MEASURED` | ⚠️ **2026-09-28（陈旧）** | `…05264bbf0d…/root` | ❌ **异机** |
| S04 数值验证 | 1 | **`stamp`**（`validated=true`） | **`bff7f7080b67…`** | —（正确性产物） | 运行时**现采**（`environment.hostname`） | `…05264bbf0d…` | ✅ **同源** |

| `S01RECHECK` | 10 | **`stamp`**（`validated=true`） | **`472697170d43…`** | `MEASURED` | ✅ **本批前置重采**（`2026-10-03T15:37:35+0800`） | `…05264bbf0d…` | ✅ **同源** |

⇒ **资格判定**：

| 批次 / 结论 | 资格 | 理由（缺哪一条） | 允许怎么用 |
| --- | --- | --- | --- |
| `EXECOPT3`：`s01_both` TTFT p50 −4.2%；`s03_fuse_gate_up` tpot +1.1%（负结果） | **T2 仅作诊断** | 缺 ① **源码身份**（`stamp_stale`、无 `tree_digest`） | 机制方向与内部决策；**论文引用须注明"无源码版本绑定"**，且不得作为唯一证据 |
| `EXECOPT4`：S04 尾段 `pack` +72.0% | **T2 仅作诊断** | 缺 ① 源码身份；**且**该批 S04 两臂带**已复现的数值缺陷**（§4.4 A5）⇒ 只 `pack` 结论可用，数值正确性不可用 | 仅作"尾段成本偏高"的**诊断**；**不得**据此称 S04 正确或错误 |
| `EXECOPT5`：尾段 `pack` +70.9%（5/5 同号）、非尾段 +2.83 ms、全窗 +0.37% | **T2 仅作诊断** | 缺 ② **环境绑定**（指纹异机，§4.4 A7）——源码身份是**闭合**的 | 与 EXECOPT4 **互相印证**（两批独立一致）⇒ 可作为"局部负结果"的**诊断**证据；**论文引用前必须先解决 A7** |
| **`S01RECHECK`：S01 的 TTFT p50 −4.31%（5/5 同向，主指标判据随脚本版本化）** | **T1 可正式引用** | ①②③ **全部闭合**：源码 `stamp` + `tree_digest` 可复算；环境指纹**本批前置重采**且容器与实时实例一致；判据**读数据前冻结**在分析器 `PREREG` 里 | 可作**量化主张**的直接证据；⚠️ 仍须带 §3.9 的适用范围（本实现 + 本负载；TPOT **不足以判定**；达标率两臂同为 **2/64 = 3.125%** ⇒ 不得声称提升，且绝对达标率本身很低） |
| S04 数值验证：`verdict=PASS`（重入 KV 逐位相同等 10 条） | **T1 可正式引用（附残留）** | ①② **都闭合**（源码 `stamp` 校验通过 + `tree_digest` 可复算；容器/OS/kernel/torch/MUSA 运行时现采）。**残留缺口：该容器的 MUSA 驱动版本未采集**（见 §3 缺口 16 同族） | 可用作**正确性**结论（正确性不依赖驱动版本）；⚠️ **不得**据此主张任何**性能**结论；⚠️ 引用时带 `warmup_deduction_verified=false` |

### 5.2 一句话总纲

* **执行优化线的三个真机性能批次全部是 T2**：源码身份与容器身份**从来没有同时闭合过** ——
  `EXECOPT3/4` 有环境、没源码；`EXECOPT5` 有源码、没环境。
  ⇒ 这是"三批独立的性能主张"目前**都进不了论文**的**唯一**原因，且**成因不同**、修法也不同。
* ★★ **`S01RECHECK`（2026-10-03）是这条线上第一个 T1**：它把"源码身份 + 环境绑定 + 判据冻结"
  三件一起做齐 —— 因为**前置步骤本身**被修过（指纹按真实结构读 + 六项接受判据 + 新鲜度）。
  ⇒ 这也是"修前置"的直接回报：**同样的两臂，才能从 T2 升到 T1**。
  ⚠️ 更正（2026-10-03）：判据是**随分析脚本版本化**的，**没有**独立的事前封存副本
  ⇒ 表述**不得**写成"已预注册"（详见 `EXPERIMENTS.md` §3.9 末段）。
* **S04 数值验证是这条线上唯一接近 T1 的产物**，因为它**自己现采**环境
  （不读那份共享指纹）——这也反过来印证了 A7 的修法方向（批次前置必须重采）。
* 其余批次（`S14B2`/`S14S`/`S6` 等调度线）见 §1；本表只覆盖执行优化线与 S04。

---

*本索引由 `2026-09-28` 会话核对产物生成；血缘与缺口均取自产物 `meta.json` 与 `EXPERIMENTS.md`，
非人工转录。发现与新证据冲突时，**以 `EXPERIMENTS.md` 与 `results/raw/**` 为准**并回来更正本文件。*
