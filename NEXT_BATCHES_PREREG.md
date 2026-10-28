# 下一批实验 · **预注册**（2026-10-03 准备，**尚未开跑**）

> **本文件是"准备"，不是"结果"。** 里面没有任何测量值；所有数字要么是从**产物现读**
> 的事实，要么明确标为 `待估`。生成配置 ≠ 实验完成。
>
> ⚠️ **本批次全部需要真机**（cost 臂的 dry-plan 都要在标定环境做，见 §6）。
> 本文件只做**离线准备**：配置、清单、判据、预算口径。

---

## 1. 成本模型 / 采样网格 / 候选形状 —— 现状清点

**部署模型**：`configs/cost_calib/model_v7.json`（**本格打算唯一在用的**——⚠️ 但现读
`configs/mthreads/**/*.yaml` 发现 **v5 / v6 / v7 三版并存**，见 §9.2 的更正）

| 项 | 值（从产物现读） |
| --- | --- |
| `lookup_table` 键数 | **107** |
| `grid_sha256` | `5c3b66f75c89522c` |
| batch 覆盖 | `1–8`（`calibrated_range` 与查表键一致） |
| `prefill_tok` 覆盖 | `0, 128, 256, 512, 1024, 2048` |
| `kv_len` 覆盖 | `0, 512, 768, 1024, 1536, 2048` |

⚠️ **`model_v8.json` 不存在**（`configs/cost_calib/` 只有 v1、v4–v7）
⇒ **v8 的数据只存在于探针层，没有导出为模型**。

**采样网格**：`shape_grid_v1..v8.yaml`。v8 的 `axes`：

```
batch      = [1,2,3,4,5,6,7,8]
prefill_tok= [0,128,192,256,512,1024,2048]   ← 比 v7 多一个 192
kv_len     = [0,512,768,1024,1536,2048]
```

**原始标定产物**（`results/raw/_costcalib/*.json`，每版 2 个会话）：

| 版本 | `grid_sha256` | 点数 | `complete` |
| --- | --- | --- | --- |
| v7 | `5c3b66f75c89522c` | 111 | True |
| v8 | `3473d023e269706f` | **24** | True |

⚠️ 这些产物的 `note` 自己写着：
**"设备侧探针，无 harness 盖章；结论只能进 RESEARCH_LOG，不得进 EXPERIMENTS.md 结论行"**。

---

## 2. 三类划分：**可复用 / 需锚点核验 / 真正缺失**

### 2.1 可复用（直接可用，无需再采）

* **v7 的 107 键** —— 既有全部批次用的就是它。`grid_sha256` 固定
  ⇒ 血缘清楚，**不得**在原地改网格（改了就断血缘，须另开一轮）。

### 2.2 ★ 需锚点核验（**有数据，但不能直接用**）

v8 的 24 点 = `{b1,b2,b4,b8} × {pf128,192,256} × {kv1024,2048}`。
其中：

* **16 点落在 v7 已采形状上** ⇒ 这 16 点正是**锚点**：同形状、**跨会话**
  （v7 会话 `20260926T…`、v8 会话 `20260928T…`）⇒ 可以直接比"是否同分布"。
* **8 点是净新增**：`{b1,b2,b4,b8} × pf192 × {kv1024,2048}`。

⇒ **锚点核验的判据（开跑前冻结）**：在那 16 个重叠形状上，
`|median_v8 − median_v7| / median_v7` 必须 **≤ 5%**（与既有留出低估的阈值同族）。
**超过就说明两个会话不可比**，此时 v8 的 8 个新点**不得**并入 v7 的表
（缺 pf192 ≠ 假装它不存在）。

⚠️ 另**两条硬限制**（来自既有记录，不得绕过）：
① 探针路径**不得盖四标签、不得进 `EXPERIMENTS.md` 结论表**；
② **`model_v8.json` 不能直接替代 `model_v7`**（v8 只采 24 点 ⇒ 导出会丢 v7 的 107 键）。

### 2.3 真正缺失（**没有数据**，且**本次不补**）

| 缺口 | 状态 | 为什么本次不补 |
| --- | --- | --- |
| `pf = 384` | **v7、v8 都没有**（已核：两处 `prefill_tok` 列表都不含 384） | 补它 = 扩网格 = 另开一轮标定 + 真机机时；与"不为增加覆盖率盲目扩网格"冲突 |
| `batch > 8` | v7/v8 都只到 8 | 同上。且 `max_num_seqs=8`（第14步 cell）下本就够用 |
| `kv_len > 2048` | v7/v8 都只到 2048 | 同上；`prompt_len=1536 + gen 32` 落在范围内 |
| v8 网格的其余组合（如 pf∈{0,512,1024,2048} × 各 batch） | 未采 | 属于"扩网格"，本次**不做** |

⇒ **结论：本次不扩网格、不伪造新标定。** 需要 pf384 或更大 batch 时，**另开一轮**。

---

## 3. 静态基线：**两条候选，选谁由校准数据决定**

| 基线 | 定义 | 文件 | 状态 |
| --- | --- | --- | --- |
| **`static_nochunk`** | `chunk.enabled=false` + 控制器关（budget 8192） | `configs/mthreads/phase1_step14/fixed_static.yaml` | **复用**（不复制） |
| **`static_small`** | `chunk.enabled=true` + `chunk_size=1024` + 控制器关 | `configs/mthreads/next_batch/static_small.yaml` | **新派生** |

⚠️ **`static_small` 的 `chunk_size` 取 1024**：这是自适应臂的 `chunk_size_init`
⇒ 静态基线必须与自适应臂**同起点**，否则量到的是"换了个 chunk 大小"而不是"控制器的收益"。

⚠️ **Best Static 只能在校准数据上选**：本文件**不选**、也不建议选哪个。
两条都留着、选择过程必须落盘（候选集与选择依据），**不得**在测试集上挑一个更好看的。
（既有 S10/E14 已把 `Best Static` 冻结为 Pareto 前沿 `{ck2048, off}` 平局取 `off`
⇒ `static_nochunk` 是**已有结论**；`static_small` 是**本轮新增的对照**，两者并存。）

⚠️ **不重复造配置**：`static_nochunk` 与 `threshold` 直接用冷冻的第 14 步配置。
复制一份 = 第二处真相（改一处另一处不跟着变，产物看不出区别）。
生成器 `scripts/gen_next_batch_configs.py` **只派生真正缺的两份**。

---

## 4. 批次一：**S01 实际部署收益的独立复核**（优先级 1）

**为什么要重做**：`EXECOPT5` 的**两个臂都开着 S01**（`baseline` 臂 = `s01_both.yaml`）
⇒ **构造上不能**证明 S01 自身的收益。

| 项 | 内容 |
| --- | --- |
| 对照 | `configs/mthreads/s4000_fp16.yaml`（S01 全关）**×** `configs/mthreads/exec_opt/s01_both.yaml`（S01 两开关） |
| 单变量 | ⚠️ **实测差异是 3 个键，不是 2 个**（见 §7）；其中 2 个是要测的开关，第 3 个是 **revision 的写法** |
| 负载 | 各自配置内的 `workload`（exec_opt 的 `short_chat`） |
| 重复 / 顺序 | **固定 5 轮**，穿插：`baseline#r0 → s01#r0 → baseline#r1 → … → s01#r4`（= **10 项**） |
| 判据入口 | `python scripts/run_exec_opt_matrix.py --tag <TAG> --repeats 5 --baseline configs/mthreads/s4000_fp16.yaml --arms s01_both --cfg-dir configs/mthreads/exec_opt` |
| 完成度判据 | `complete=true` / `under_run=false`（少跑即失败 —— 2026-10-03 已实现并实测） |
| **必须报的服务指标** | TTFT p50/p95/p99、TPOT p50/p90、达标吞吐（tok/s）、SLO 达标率 |
| **必须同时报** | 每轮 **>1 s 的步数**（EXECOPT5 实测：两臂各 **1** 个/轮，都在 step 0） |
| 环境 | 批次前置会**重采指纹并校验容器/设备身份**（§4.4 A7 的修法）；容器变则记 `container_changed` |
| 源码 | `stamp` 投放并校验通过；`git_dirty` 如实记录 |
| 预计产物 | `results/raw/<TAG>-short_chat-s4000_fp16{,-r1..-r4}`、`…-s01_both{,-r1..-r4}`（**10 个目录**） |
| **预算** | **≈ 27 min**（10 项 × **实测 162.2 s/项**，来自 EXECOPT5 **同 workload 同配置族** ⇒ 可直接沿用）。建议给 `--budget-minutes 240` 留余量 |

⚠️ **`--baseline` 的正确路径是 `configs/mthreads/s4000_fp16.yaml`**，
**不是** `configs/mthreads/exec_opt/s4000_fp16.yaml`（后者**不存在** —— 实测 `load_config` 报
`FileNotFoundError`）。这与 `BASELINE_DEFAULT` 的取值一致。

⚠️ **不要求**差异为正或负 —— 这是**复核已有决定**，不是找收益。

---

## 5. 批次二：**机制小批**（单预算 cost / 修正双预算 cost / threshold + 两条静态基线）

目的：在一个**可控的小批**里同时看三件事 ——
① 双预算修正后，混合步不再被 prefill 预算旁路（**机制**）；
② 控制器相对**同起点静态基线**的效果（**收益方向**）；
③ 最接近方法对照的相对位置。

| 臂 | 文件 | `controller.kind` | 双预算 | 状态 |
| --- | --- | --- | --- | --- |
| `static_nochunk` | `phase1_step14/fixed_static.yaml` | —（关） | — | 复用 |
| `static_small` | `next_batch/static_small.yaml` | —（关） | — | **新派生** |
| `threshold` | `phase1_step14/fixed_threshold.yaml` | `threshold` | — | 复用 |
| `cost_single` | `phase1_step14/fixed_cost.yaml` | `cost` | `None`（关） | 复用 |
| **`cost_dual`** | `next_batch/cost_dual.yaml` | `cost` | **2000.0**（开） | **新派生** |

⚠️ **两条 cost 臂只差 `controller.prefill_deadline_ms` 一项**（程序化核对，见 §7）。

### 5.1 ★ 本批暴露的一个真实缺口（已在本次**接通**）

`prefill_deadline_ms` 此前**在配置层根本不存在**：`ControllerConfig` 没这个字段，
`arm_factory` 也从不传它 ⇒ 双预算**只能在代码与单测里可达，任何批次都开不了它**
⇒ 产物里永远看不到"这个臂到底开没开双预算"。

本次已接通：`ControllerConfig.prefill_deadline_ms`（默认 `None`）
+ `describe()` 落盘 + `arm_factory` 透传。
⚠️ **接线本身不改变任何语义**：默认 `None` ⇒ 既有批次逐字节不变。

### 5.2 负载与顺序

| 项 | 内容 |
| --- | --- |
| 负载 | 各臂配置自带的 `workload`（第 14 步 cell：`short_chat`，`prompt_len=1536`，`gen_len=32`，`concurrency=16`，`max_num_seqs=8`） |
| 臂数 | **5** |
| 重复 | **建议 5 轮**（既有冻结配置 `run.repeats: 5`） |
| 顺序 | **穿插**：每轮扫完整张 5 臂表（`static_nochunk → static_small → threshold → cost_single → cost_dual`），共 **25 项** |
| 预计产物 | 25 个 run 目录；另加 `exec_opt_summary_<TAG>.json` |
| **预算** | ⚠️ **待估**。本 cell 的**实际节拍没有 pilot 数据**（它与 exec_opt 的 `short_chat` **不是同一组参数**：prompt 1536 vs 496、gen 32 vs 123、concurrency 16 vs 8）⇒ **不得**照搬 162.2 s/项。必须先跑 **1 项 pilot** 测节拍，再按 `25 × pilot_s` 申请预算 |

⚠️ **前置（未满足则不得开跑）**：
① 单变量性程序化核对（§7）；
② `cost` 臂的 dry-plan 必须在**标定环境**做（见 §6）；
③ 预注册写死：`prefill_deadline_ms` 的取值理由（= `run.slo.ttft_ms` 同源）；
④ ← 若 §5.3 的两处**待冻结**语义尚未裁定，**不得**开跑 `cost_dual`。

### 5.3 ⏸️ 待冻结（**改变研究问题**，裁定前不得开跑）

见 `NEXT_SESSION_TASKS.md` 的 **FZ-1 / FZ-2**：

* **FZ-1**："双预算"是"**逐步择一**"（当前实现）还是"**步内拆分两块**"
  （后者需要成本模型给出 decode/prefill **分量预测** ⇒ 需重做标定）。
* **FZ-2**："受阻 decode"的判定口径是"**本步没被服务**"（当前）还是
  "**逐请求间隔将超标**"（后者与外部 SLO 的 TPOT 目标耦合 ⇒ 改变问题）。

⚠️ 在 FZ-1/FZ-2 裁定前，`cost_dual` 的结论**只能**表述为
"在（A/A）这组语义下、**离线验收通过**"，**不得**表述为"双预算有效/无效"。

---

## 6. 离线 dry-plan 的**已验证边界**（含一条否定对照）

| 配置 | 本机 dry-plan | 证据 |
| --- | --- | --- |
| `static_small.yaml` | ✅ **可以** | 5 次 run 全成，产物 `data_label=TEST_DATA`、`backend=simulate`；且**把 socket 关掉再跑一遍仍然成功** ⇒ 该路径**不连远端** |
| `fixed_cost.yaml` / `cost_dual.yaml` | ❌ **不可以** | 成本模型加载时**指纹比对**拒绝：本机 `torch 2.10.0+cu130` vs 标定 `2.2.0`、`transformers 4.50.0` vs `4.46.3`、`python 3.13.14` vs `3.10.8` ⇒ **正确地**拒绝 |

⚠️ 上表第二条**不是"环境坏了"**，而是"本机不是标定环境"。
⇒ **cost 臂的一切 dry-plan / 正式运行都必须在标定环境（远端 MTT S4000）做**。
（把两者混为一谈，会让人去修一个不该修的东西。）

⚠️ 与既有纪律一致：所有 dry-run 与模拟器产物**都是 `TEST_DATA`，不得进结论**。

---

## 7. 必须**程序化**核对（不靠目测）的两处单变量性

### 7.1 S01 复核对：**实测差异是 3 个键**（已核对，非推测）

```
runtime.logits_last_only      ← S01 开关（要测的那个）
runtime.sample_needed_only    ← S01 开关（要测的那个）
runtime.model_revision        ← ★ `main` → 钉死的 commit（写法差异）
```

⚠️ 第三个**不是**第二个变量，而是**同一份权重的写法差异**（生成器把 `main`
钉死成实测解析到的 commit）。但这**只在 `main` 今天仍解析到同一个 commit 时成立**
—— 否则它就是**真的**换了权重，两臂不再单变量。

⇒ **开跑前置（必须执行，离线侧无法验证）**：在远端确认
`main` 仍指向 `a09a35458c702b33eeacc393d103063234e8bc28`
（`git ls-remote` 或读缓存 snapshot 的 `refs`）。
⚠️ 这一条**不写成恒真的测试** —— 一个不会失败的断言比不写更坏。
已把差异集钉成测试（`tests/test_next_batch_configs.py::TestS01RecheckPairIsSingleVariable`）：
**差异集一变就报红**。

### 7.2 机制小批：两条 cost 臂

`fixed_cost.yaml` vs `cost_dual.yaml` ⇒ 差集必须**恰好**是
`controller.prefill_deadline_ms`。**已实现为测试**
（`tests/test_next_batch_configs.py::TestSingleVariableBetweenCostArms`）。

⚠️ 通例：**"目测 diff 过了"不算核对** —— 本仓有过"按记忆写字段名 ⇒ 恒取到 0
⇒ 四臂指纹全相同"的先例。任何"只有一个变量"的主张都必须有一条**会失败的**检查。

---

## 8. 交付清单（本文件配套的产物）

| 项 | 路径 | 状态 |
| --- | --- | --- |
| 配置生成器 | `scripts/gen_next_batch_configs.py`（含 `--check`） | ✅ 已实现 |
| 派生配置 | `configs/mthreads/next_batch/{static_small,cost_dual}.yaml` | ✅ 已生成（LF、可复现） |
| 双预算开关接线 | `src/ascend_opt/controller.py` + `benchmark/arm_factory.py` | ✅ 已接通（默认关） |
| 配置/接线测试 | `tests/test_next_batch_configs.py`（16 例） | ✅ 通过 |
| 机制缺口登记 | `EVIDENCE_INDEX.md` §3 第 29 行 | ✅ 已更新 |
| 待冻结语义 | `NEXT_SESSION_TASKS.md` FZ-1 / FZ-2 | ✅ 已登记 |

---

*本文件由 2026-10-03 离线会话生成（§1–§8）；**§9–§12 由同日第二次离线会话追加**。
所有事实性数字均从 `results/raw/**` 与 `configs/**` 现读；
标注 `待估` 的地方**没有**数据支撑，开跑前必须先测 pilot。*

---

## 9. ⏯️ 口径统一（2026-10-03 追加，**开跑前必须落地**）

下一批围绕**真实问题**组织：

> **TPOT 多数超过 100 ms；SLO 达标率仅 2/64 = 3.125%**（来源：`S01RECHECK`，10/10 轮，
> 两臂一致）。⇒ 本格要回答的不是"控制器能不能省 TTFT"，而是
> **"在当前执行基线下，控制器能否把 TPOT 拉回 SLO 内"**。

### 9.1 ★ `static_small` 的定义冲突（**两处文档互相矛盾，必须以本节为准**）

| 出处 | 说法 | `chunk_size` |
| --- | --- | --- |
| `NEXT_SESSION_TASKS.md` §3（第 5783 行） | = 自适应臂的 **`chunk_size_min`** | → 该 cell 为 **512** |
| `NEXT_BATCHES_PREREG.md` §3、`scripts/gen_next_batch_configs.py` | = 自适应臂的 **`chunk_size_init`** | → **1024** |

**程序化事实（现读 `configs/mthreads/phase1_step14/fixed_{static,cost,threshold}.yaml`）**：
`chunk_size_init = 1024`、`chunk_size_min = 512`、`chunk_size_max = 2048`、`chunk_size_step = 512`。
⚠️ **本 cell 的 `chunk_size_min` 是 512，不是 128**（128 是 `phase1_d1/fixed_cost_min128.yaml`
那**另一个** cell 的下界）—— 把两者混为一谈会派生出**错的**基线。

**裁定（本节生效）**：**两份基线并存**，因为它们回答两个不同的问题，**不得**只留一个：

| 基线 | `chunk_size` | 回答的问题 |
| --- | --- | --- |
| `static_nochunk` | —（不分块） | 历史最强静态（S10/E14 的 Pareto 平局取 `off`） |
| `static_small@1024` | **1024**（= `chunk_size_init`） | **同起点**：控制器相对"起点配置"的收益 |
| ★ `static_small@512`（**新增候选**） | **512**（= `chunk_size_min`） | **同范围下界**：收益来自"**会调**"还是"**能调**" |

⚠️ **Best Static 仍只能在**校准数据**上选（§3 的纪律不变），且选择过程必须落盘。
⚠️ `static_small@512` 目前**不存在**：需要时用生成器另派生一份（**不手改**），
且派生后必须过 §7 的单变量核对。

### 9.2 ★ 成本模型 **三版本并存**（v5 / v6 / v7，**以本节为准**）

⚠️ **本节 2026-10-03 二次更正**：初版标题写的是"v5 / v7 不一致"，把它当成**两版**问题。
**那是错的** —— 我靠"两处对读"（`phase1_step14` vs `phase1_d1`）下的结论，
漏掉了 `phase1_step15/*` 与 `phase1_step14b/*` 这**两个整目录都指向 v6**。
发现方式：把"在用版本集合"写成断言后**测试立刻变红**（详见 §9.2.1）。
⇒ **教训**：版本清点必须**穷举 `configs/mthreads/**/*.yaml`**，不能抽样两处就下结论。

**程序化事实（现读 `configs/mthreads/**/*.yaml` 的 `controller.cost_model_path`，穷举）**：

| 使用者（目录/文件） | 引用的模型 | 键数 | `grid_sha256` |
| --- | --- | --- | --- |
| `phase1_step14/fixed_cost.yaml`（**本小批的 `cost_single` 来源**） | **`model_v5.json`** | 91 | `c0ed35000aae61a0` |
| `next_batch/cost_dual.yaml`（本小批新派生，继承自上面） | **`model_v5.json`** | 91 | `c0ed35000aae61a0` |
| `phase1_step15/*`（`fixed_cost*` / `ramp_*` / `lenmix_*`） | **`model_v6.json`** | — | — |
| `phase1_step14b/*`（`ramp_*` / `lenmix_*`） | **`model_v6.json`** | — | — |
| `phase1_d1/*`、`phase1_d1_scan/*` | **`model_v7.json`** | **107** | `5c3b66f75c89522c` |
| `phase1_s6/*`（`service_benefit_*`） | **`model_v7.json`** | 107 | `5c3b66f75c89522c` |

⇒ **在用集合 = `{model_v5.json, model_v6.json, model_v7.json}`（三个）**，
与 §1 声明的"唯一在用 = v7"**三处**矛盾（不是两处）。

#### 9.2.1 这条是**跑出来的**，不是看出来的

`tests/test_next_batch_configs.py::TestCostModelVersionConsistency::test_model_versions_in_use_are_enumerable`
最初断言 `used == {v5, v7}`，**当次即红**：

```
AssertionError: Items in the first set but not the second:
E       'model_v6.json' : 在用模型版本集合变了：['model_v5.json','model_v6.json','model_v7.json']
```

⇒ 该断言现已改为钉住**真实的三元集合**。**任何**未来的版本增删都会让它红，
从而**强制**同步更新本节（这就是"让不一致无法被静默忽略"的实现方式）。

**裁定（本节生效）**：**统一到 `model_v7.json`**，理由：
① 它是**键数最多**（107 ⊃ 91，v6 未单列键数 ⇒ 需开跑前补记）且 `complete=True` 的版本；
② `phase1_d1`/`phase1_s6` 已按它跑过，**换回 v5/v6 会与那些批次的锚点不可比**；
③ §2.2 已确认 v8 **不能**替代 v7（会丢 107 键）。
⚠️ **改法**：把 `next_batch/{cost_dual}.yaml` 的 `controller.cost_model_path` 改为 v7，
**并同步派生 `static_small@512`**；改完必须重跑 §7 的单变量核对（v5→v7 是**一个显式变量**，
必须在预注册里写明白：**本小批用的是 v7，不是 v5、也不是 v6**）。
⚠️ **不得**就地改 `phase1_step14/fixed_cost.yaml` / `phase1_step15/*` / `phase1_step14b/*`
（冷冻配置 = 历史血缘，改了会断）。
⇒ `cost_single` 也应**另派生**一份指向 v7 的 `cost_single_v7.yaml`，与 `cost_dual` 一起进批次。
⚠️ **v6 的键数/`grid_sha256` 当前未记入本表**（我只确认了它的**使用者**，未读文件内容）
⇒ 这是本节的**已知空缺**，在 §12 的 G 门里作为一条"开跑前补齐"项，**不猜**。

### 9.3 FZ-1 / FZ-2 语义（**未裁定 ⇒ 不得开跑 `cost_dual`**）

见 `NEXT_SESSION_TASKS.md` 第 319–339 行，此处不复制（避免第二处真相来源）。
本文件只强调**对下一批的约束**：

| 冻结项 | 未裁定时的表述上限 |
| --- | --- |
| **FZ-1**（双预算 = 逐步择一 / 步内拆分） | 只能说"在（逐步择一）语义下、离线验收通过" |
| **FZ-2**（受阻 decode = 本步没被服务 / 逐请求间隔将超标） | 只能说"在（本步没被服务）口径下" |

⚠️ **FZ-1/FZ-2 只影响调度侧研究**，**不阻塞**已完成的 S01 复核（控制器关）。

---

## 10. ★ 最小可辨别机制 pilot（**先跑这个，不要直接上 5 臂 × 5 轮**）

**目的**：在**一次短跑**里先回答三个"能不能分辨"的问题，再决定要不要投 25 项机时。

| 问题 | 为什么必须先答 |
| --- | --- |
| ① 本 cell 的节拍（s/项）是多少？ | §5.2 已标注 **`待估`**：本 cell（prompt 1536 / gen 32 / conc 16 / `max_num_seqs=8`）与 exec_opt 的 `short_chat` **不是**同一组参数 ⇒ **不得**照搬 162.2 s/项 |
| ② 本 cell 的 TPOT 分布落在 SLO 的哪一侧？ | 真实问题是"TPOT 多数超 100 ms" ⇒ pilot 要报**达标率绝对值 + 分母**（**只作解释**；扩大与否看 §10.3 的 E1/E4，**不看**达标率落在哪个区间） |
| ③ 双预算修正是否**真的**改变了步级行为？ | 若 `prefill_deadline_ms` 开了与没开**步级轨迹完全相同** ⇒ 机制没被行使，"下结论"等于没测 |

**pilot 设计（3 项，不是 25 项）**：

| 项 | 臂 | 轮数 | 用途 |
| --- | --- | --- | --- |
| P1 | `static_small@1024`（同起点静态） | **1** | 测节拍 + 测该 cell 的 TPOT/达标率基线 |
| P2 | `cost_single`（新整步模型，待验证后派生配置） | **1** | 测控制器的步级轨迹与回退率 |
| P3 | `cost_dual`（同一新模型） | **1** | 测双预算是否改变步级轨迹（与 P2 逐 step 对比） |

**★ 执行清单由脚本**产出一处权威**：`scripts/pilot_execution_plan.py`
（配置路径 / 臂名 / 轮数 / 真机命令**都在那里定义**，本文档只引用，**不复述**）。

```bash
python scripts/pilot_execution_plan.py                  # 打印清单 + 校验（应显示 3 项）
python scripts/pilot_execution_plan.py --help           # 真机命令需已核验绝对目录/Python及新 cfg-dir
```

⚠️ **两个默认值陷阱**是"以为跑 3 项、实际跑 15 项"的成因，必须显式规避：

| 陷阱 | 默认行为 | 后果 | 正确写法 |
| --- | --- | --- | --- |
| `--repeats` 不给 | 退到 **`--fill-budget`**（3~50 轮） | 3 项变 15 项 | 显式 **`--repeats 1`** |
| `--baseline` 用默认值 | `cells = ["baseline"] + arms` ⇒ 多一格 | 3 臂变 **4** 格 | **P1 走 `--baseline <static_small.yaml>`**，`--arms` 只放 `cost_single cost_dual` |

✅ **计划期可核对**：脚本会**真的**调 `run_exec_opt_matrix --dry-plan` 并解析 JSON，
断言 `cells == ["baseline","cost_single","cost_dual"]`、**恰好 3 项**、`repeats == 1`
（不靠人眼数）。

### 10.1 ★ pilot 的**目的**（先说清它**不能**做什么）
**pilot 用于检查三件事**：① **机制**是否被行使（步级轨迹真的变了）；
② **覆盖**是否成立（判据有定义域、可达标也可是不达标）；
③ **节拍**（s/项）是否与预算相符。

⚠️ **pilot 不用于宣布性能优势**。3 项 × 1 轮，**没有** 5/5 配对、**没有**重复，
按 `AGENTS.md` 正文第 9 节（重复 <3 不出结论）它**在构造上**就不够。
⇒ pilot 的任何数字**不得**写成"更好/更差"，**不得**进 `EXPERIMENTS.md` 结论表。
它只回答"**要不要投 25 项**"。

---

### 10.2 开跑前检查（**跑之前**做；任一不过 ⇒ **不开跑**）

这些是**准备度**检查，与"跑出来好不好"无关。**它们不因 pilot 结果而改变。**

| # | 检查 | 不过怎么办 |
| --- | --- | --- |
| A1 | `configs/mthreads/prereg_batch/` 三份配置与生成器**逐字节一致**（`gen_prereg_batch_configs.py --check`） | 重新生成；**不得**手改派生文件 |
| A2 | 三条配置都能 `load_config` 通过（尤其 `controller` 段无未知字段） | 修生成器；**不得**往 YAML 塞非配置字段 |
| A3 | S01 开关在**配置里**显式为 `true`（不是靠默认值） | 修生成器 |
| A4 | 内部预算 120 ms 与外部 SLO 100 ms **是两个数**，且 `target_tpot_ms=120` | 修生成器 |
| A5 | `max_num_seqs = 8` | 修配置 |
| A6 | 远端可达 + 该批前置**重采**环境指纹并校验容器/设备身份（`G7`） | **停** —— 这是当前硬阻塞 |
| A7 | `main` 仍解析到 `a09a35458c702b33eeacc393d103063234e8bc28` | **停**，核权重 |
| A8 | `pilot` 的预算档位已确定（见 §10.4） | 先定预算再开跑 |

### 10.3 pilot 结束后的**扩大条件**（跑之后才判）

⚠️ **这里**才是"要不要投 25 项"的判据。**与 §10.2 分开**，因为混在一起会出现
"因为某个准备度检查没过，就误判成'机制无效'"这类跨层归因。

| # | 判据 | 含义 | 动作 |
| --- | --- | --- | --- |
| E1 | P2/P3 的**步级轨迹不同**（`budget_kind` 出现过 `prefill`，且逐 step 有差异） | 机制被行使 | 否则：先修接线，**不扩大** |
| E2 | P1 实测**节拍**已知（s/项） | 才能算 25 项的预算 | 未知则**不申请** 25 项机时 |
| E3 | 该 cell 的达标率**已登记绝对水平**（分子/分母两个整数一并写出） | 可判区分度 | ⚠️ **不作扩大与否的开关**，见下方**修订说明** |
| E4 | 三项 `data_label == MEASURED` 且身份一致 | 数据可用 | 否则核环境 |

**★★ 2026-10-03 修订（用户点名两次，第二次为终裁）：删除"达标率必须落在 (0,1)"这条规则。**

第一次修订删掉了下面第一条，但**又换成了一个同样错误的规则** —— 下面第二条。两者同族：

> ❌ 旧规则 1："`slo_satisfaction_rate ∈ {0, 1}` ⇒ 该 cell 无区分度 ⇒ **停**"
>
> ❌ 旧规则 2："达标率**必须落在开区间 (0,1)** 才算有区分度（否则视为无区分度）"

**共同错误**：都在用**一个数值区间**去决定"要不要扩大"。
扩大与否要看的是**三臂的实际结果与机制是否真的被行使**（E1/E4），
不是一个单臂指标落在哪里。

⇒ **现行规则（终态）**：

| # | 判据 | 含义 | 动作 |
| --- | --- | --- | --- |
| E1b | P2/P3 的 `budget_kind` / 逐 step 轨迹**真的不同**（E1） | 机制被行使 | 机制没被行使 ⇒ 先修接线，**不扩大** |
| E4b | 三臂 `data_label == MEASURED` 且身份一致（E4） | 结果可用 | 身份不一致 ⇒ 核环境，**不扩大** |
| E3b | 达标率的**绝对水平已登记**（分子/分母整数） | 供解释，不作开关 | **无条件登记**；数值高低**本身不决定**扩大与否 |

* 达标率的三个取值（`0` / `∈ (0,1)` / `1`）**都只用于解释**，**都不单独构成"停"或"续"**：
  * `= 1.0` ⇒ SLO 对该 cell **偏松**（解释：该 cell 的机制差异可能被 SLO 掩盖）；
  * `= 0.0` ⇒ SLO 对该 cell **偏紧**或**判据口径**待核（解释：先分辨成因）；
  * `∈ (0,1)` ⇒ 该 cell 对当前 SLO **有区分度**。
  三者都是**对该 cell 的描述**，**不是"要不要投 25 项"的判据**。
* **显著优于 / 劣于**的判定仍走 §10 上方的三态（`SUPPORTED` / `REGRESSED` / `UNCLEAR`），
  与达标率取值无关。

⚠️ **`0.0` 与 `1.0` 都必须附"分母"**（`n_slo_satisfied` / `n_requests` 两个整数）：
只看一个比值无法区分"真没达标"和"分母是 0 / 字段缺失"。

### 10.4 ★ 预算：**30 分钟与 45 分钟的冲突已统一**

原文同时出现两个数（§10 表写 "≤30 min"、§11 表写 "pilot 用 45"）且**没有说明关系**。
⇒ 现在只保留**一种**表述：

* **pilot 的机时上限 = 45 min**（`--budget-minutes 45`）；
* 其中 **30 min 是"预期用量"**（= 历史同族 162.2 s/项 × 3 ≈ 8 min，留 ~3.7× 余量），
  **45 min 是"上限"**。
* ⚠️ `--budget-minutes` 的**真实语义**（与 `benchmark/exec_opt_plan.py:157 decide_stop`
  逐条对齐，**不是**"硬超时"）：

  | 情形 | `decide_stop` 的行为 |
  | --- | --- |
  | 该跑的都跑完了（`items_done >= items_total`） | 停 —— "全部项已完成" |
  | 预算够跑完**当前这一轮**（`elapsed + need ≤ budget`，`need = avg × 本轮剩余 × 1.15`） | **继续** |
  | 预算**不够跑完当前这一轮** | 停 —— **不开始**这一轮（避免"有的臂 5 次、有的臂 4 次"） |
  | 还没跑完任何一项（`avg_item_s is None`） | **不因预算停**（否则一项都跑不了） |
  | 发现 STOP 文件 | 协作式停止（当前 run 跑完后退出） |

  ⇒ 它是**"整轮边界上的干净收尾守卫"**：**只在整轮之间**决定要不要开下一轮，
  **不打断**正在跑的 run，**不是**到点就 kill。
  ⚠️ **不会**因为"过了 45 min"就截断当前这一轮；只在**下一轮开跑前**发现余额不足才收尾。
  ★ 这条设计的**真实动机**是**保持重复均衡**（宁可少一整轮，也不要"有的臂 5 次、
  有的臂 4 次"）—— 不均衡的重复会让配对判据失效。
* ⚠️ 收尾时：**已完成项保留、未完成明确登记、不从头重复**；`summary` 文件**总是**写出，
  它是本机等待侧的**权威退出通道**（见 `decide_wait_exit`）。
* ⚠️ **两个数不是两个预算**。原表的写法（一处 30、一处 45、无说明）会让读者
  以为"到底按哪个算"，属于**同一件事的第二处真相**。⇒ 统一为：
  **45 min 上限 / 30 min 预期**，且**只有这一处**定义。

🚫 **已废止的旧规则（保留记录，不再生效）**：
> "P1 的 `slo_satisfaction_rate ∈ {0, 1}` ⇒ 该 cell 无区分度 ⇒ **停**"
>
> "达标率**必须落在开区间 (0,1)** 才算有区分度（否则视为无区分度）"

两条**同族错误**、**都已废止**：都用一个数值区间去决定"要不要扩大"。
**替代规则** = §10.3 现行表（扩大与否看 E1 机制是否被行使 + E4 身份，达标率只作解释）。

⚠️ **pilot 的产物不得进结论表**（与 §6 的 dry-run 纪律同族）：它只用于**决定是否投 25 项**。

**旧成本模型在新执行基线下需核验的锚点**（用户点名）：

| 锚点 | 现状 | 下一批怎么核 |
| --- | --- | --- |
| `model_v7` 的 **batch 维准确性** | 纯 decode 档准确到 ≤1.5%，但**口径必须一致**（模型 `total_ms` 是前向，实测 `tpot` 是整步） | 在 pilot 的 P2 里比对"模型预测 vs 实测 `fwd_ms`"（**同口径**，不拿 `tpot` 比） |
| **`prefill_tok ≥ 512` 的步** | 单 per-step deadline 与"整块 prefill 步"**构造上不相容**（全档超 120 ms） ⇒ 双预算是为此设计 | P3 需证明这些步**真的**走了 prefill 预算分支（产物里有 `budget_kind`） |
| **`kv_len > 2048`** | 未采样（§2.3） | 本 cell 的 `prompt_len=1536 + gen 32` 落在范围内 ⇒ **不需要**；若改 cell 则须先核 |

---

## 11. 下一批的预算（**先列机时，不自动开跑**）

| 阶段 | 项数 | 单件估计 | 机时估计 | 依据 |
| --- | --- | --- | --- | --- |
| **pilot（§10）** | 3 | **待测**（P1 实测后回填） | **上限 45 min / 预期 30 min**（见 §10.4；历史同族 162.2 s/项 × 3 ≈ 8 min，留余量） | ⚠️ `待估`，先跑 P1 |
| 5 臂 × 5 轮（§5） | 25 | `25 × pilot_s` | **待测算** | **pilot 完成后**才能申请 |
| （建议）`--budget-minutes` | — | — | pilot 用 **45**；正式批按 `25 × pilot_s / 60 × 1.5` | 它是**整轮边界上的干净收尾守卫**（`decide_stop`），**不打断**正在跑的 run |

⚠️ **pilot 预算只有一处定义**（§10.4）：**45 min 上限 / 30 min 预期**。
本节不再重复另一套数（避免"同一件事的第二处真相"）。

⚠️ **纪律**：首项完成后**更新剩余时间估算**；异常或预算不足时**不继续扩大**，
已完成项保留、未完成明确登记、**不从头重复**。

---

## 12. 下一批的**前置门**（全过才可开跑；缺一即停）

| # | 门 | 现状 |
| --- | --- | --- |
| G1 | **三版本统一**（§9.2）：派生 `cost_single` + `cost_dual` 指向 v7（现为 v5） | ✅ **已做**（`prereg_batch/cost_*.yaml`，显式 `model_v7.json`） |
| G1b | **补记 `model_v6.json` 的键数 / `grid_sha256`**（§9.2 的空缺项，不猜） | ✅ **已做**（v6 `grid=3de2fb28b6a47a6f` / 104 records，见生成器文件头与 `--check` 断言） |
| G2 | `static_small` 派生（§9.1） | ✅ **已做**（`prereg_batch/static_small.yaml`，chunk=1024 与自适应臂同起点） |
| G3 | 单变量性程序化核对（§7）在**改过模型路径之后**重跑 | ✅ **已做**（`TestCostArmsAreSingleVariable`：逐行 diff 恰好 1 处） |
| G3b | **S01 开关 / 模型 revision / 120 ms 内部预算 / 100 ms 外部 SLO 在配置里显式存在** | ✅ **已做**（`TestRequiredFieldsAreExplicitNotDefaulted`） |
| G4 | FZ-1 / FZ-2 范围 | 当前 A/A 实现口径已明确；不声称是最优语义 |
| G5 | 远端可达 | 本轮离线，**未检查**；历史失败不代表当前状态 |
| G6 | pilot 的开跑前检查（§10.2）全过 | ⬜ 待 pilot |
| G7 | 环境指纹**本批前置重采**并校验容器/设备身份（A7 的修法） | ⬜ 待部署 |
| G8 | 固定 revision snapshot 存在，实际 resolved revision 等于配置 SHA | ⬜ 待远端核；不依赖浮动 main |
| G9 | `tests/test_next_batch_configs.py` + `tests/test_prereg_batch_configs.py` + `tests/test_pilot_flow_rules.py` 全绿 | ✅ 24 + 24 + 17 passed（2026-10-03 现跑） |

⚠️ **撤回“G5 是唯一硬阻塞”**：除连接/身份外，旧 v7 与新执行契约未对齐；
必须完成 §13 的短验收、整步标定和独立验证。旧连接诊断如下，仅作历史参考。

⚠️ **根因保持待确认**：症状是"主机在线、该端口无 SSH 服务"。

> **诊断签名只是假设（不是结论）。** 症状（TCP 可连、**无 banner**）在历史上**两次**
> 确实由映射端口变更引起；但把它读成结论就是**未验证的因果跳跃** —— 这里**不写成"端口已经变化"**。
> 当前**假设**：是映射端口变更（历史两次实证）；**竞争假设**：该端口后服务未起 /
> 被占用 / 上游策略变更。**本轮未定位根因 ⇒ 不写成"端口已经变化"。**

⇒ 下一步 = 用户在 AutoDL 控制台**核对当前映射端口**（而不是猜一个再试）→
只改仓库外 `~/.workbuddy/remote.env` → **单次**探测。

## 13. 当前执行形状协议（2026-10-03；本轮仅离线）

本节替代此前“只剩连接”“先补 9 点”的执行指针；历史结果与模型保持不变。
用户决定先完成离线工作，真机任务留待集中执行。本轮未检查当前远端状态。

- 生产成本臂从共享 `StepPlan → ForwardSpec` 构造逐请求 token/KV/need_token 与实际分组。
  新接纳但未获得计算配额的请求不再计入 batch；续 prefill 计入。
  控制器使用步后重新分区的队列，decode KV 不包含尚未送入模型的最后一个采样 token。
- 标量历史重放仍可用，但标为 `legacy_scalar_diagnostic`；生产工厂强制新执行签名。
  旧 v7 无新契约，返回 `execution_contract_unvalidated`，不能解释成模型已否证或预算不可行。
- 当前自动清单是 **6 个整步形状、32 次候选评估、14 处旧 batch 修正**。
  它来自合法独立合成状态（TEST_DATA），不是生产到达轨迹的覆盖率保证。
  入口 `results/processed/execution_shape_manifest_20261003.json`。
- 一个点是一次完整 `TransformerDriver.forward`，包含其所有执行组。
  `fwd_ms` 是各实际组前向累计；`host_ms = driver elapsed_ms - fwd_ms`；
  pack/readback 另列诊断，不重复加到 host。prime、retire 和预热在测量窗外。
  该量不包含整个服务请求的排队与所有后端外部工作，不能声称硬 SLO 保证。
- 每个 split 恰好 **3 个独立进程**，每个进程每形状 30 条记录，逐形状预热至少 30 秒。
  训练期每进程取 total 中位的成对 fwd/host，再取三个训练进程中 total 最大者。
  固定额外余量为 0；训练表冻结之后再采 3 个验证进程，验证数据不参与改表。
  沿用低估相对 p95 ≤5%、绝对相对误差 p90 ≤10%、跨会话漂移 ≤10% 的现有阈值。
  绝对误差门在旧研究中是事后补充；在本协议新数据产生前固定。只检验同形状跨进程，
  不声称未见形状泛化；不把 30 条记录或 token 当独立实验重复。
- `execution_shape_probe.py` 默认仅 dry-plan。`export_execution_shape_model.py` 先 freeze，
  后 validate；缺身份/不完整/验证失败均不导出正式模型。输出禁止覆盖。
  `prepare_execution_pilot.py` 只接受通过验证的 MEASURED 新模型，派生到全新配置目录。
  历史 prereg 配置与旧成本模型不修改；新增模型/配置后须同步并核对各自 SHA。
  检查部署戳仍有效；若期间修改执行代码则重新部署打戳并重新判断标定可用性。
- 三项 pilot 仍为 baseline/cost_single/cost_dual，各一次，使用派生目录显式 `--cfg-dir`。
  默认目录指向旧 v7，仅能核对计划次数，不能当作新模型已就绪。
  新 `prediction_execution_match` 只绑定下一步；有新到达等造成形状变化时，不纳入预测误差比较。
  应并报形状匹配率、预测覆盖率、回退原因和动作是否行使，再决定是否扩大实验。
- **撤回旧 30 分钟预期及历史 162.2 秒/项外推**。45 分钟仅为矩阵整轮边界预算参数，
  不保证墙钟上限，不会中断正在运行的一轮；标定与验收另计，用实际节拍估算。

集中执行步骤与命令见 `handoff/optimization_20260929/REMOTE_EXECUTION_PLAN_20261003.txt`。
本轮离线报告与测试见同目录 `OFFLINE_EXECUTION_SHAPE_20261003.txt`。
