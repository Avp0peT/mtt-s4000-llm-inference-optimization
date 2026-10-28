# 3 平台与执行特征（阶段一）

> 证据入口：`COST_PROFILE_S4000.md`、`results/raw/_probe*`、`EXPERIMENTS.md` 的 E9/E10/E12 登记行、
> `scripts/check_model_coverage.py`、`scripts/decompose_step_overhead.py`。
> **本节不含任何"待解释"的机制推断**（那些在 §4）。

## 3.1 硬件与环境指纹

| 项 | 值 |
| --- | --- |
| 平台 | `mthreads`（MUSA 栈，**非 CUDA**） |
| 设备 | **MTT S4000 × 1**，设备显存 **49152 MiB**，采集时 0% util / 占用 4 MiB / 41–42 °C |
| 驱动 | MTT driver **2.7.0**（源 `mthreads-gmi` 的 `Driver Version` 行） |
| SDK | MUSA SDK **3.1.0**（源 `file:/usr/local/musa/version.json` 的 `musa_toolkits`） |
| 框架 | `torch 2.2.0`（`2.2.0a0+git8ac9b20`）+ `torch_musa 1.3.0+81caf0a` |
| 模型 | **`Qwen2.5-7B-Instruct`**，精度 **fp16**，单卡 |
| 执行器 | 本项目自建的**迭代级**执行器（每步一次设备前向批）；**不使用** vLLM 栈 |
| 宿主 | Python 3.10.8；容器 `cpu.max = 1500000 100000`（`nproc` = 15，宿主 128 核） |

⚠️ **两栈不可比**：次平台（昇腾）的配置与产物不与本节并表。

⚠️ **口径纪律（本项目踩过的两处静默给错，写入这里是提醒读者为何这些值可信）**：
`mthreads-gmi` 首行把**工具版本与驱动版本印在同一行**，用交替正则取值会先命中位置靠前的标签
⇒ 把工具版本 1.14.0 记成驱动版本；MUSA 版本只在 `version.json` 的 `musa_toolkits`
（缺失时退到 `musa_runtime`）才是 SDK 版本，随手取"文件里第一个数字"会取到别的组件。
⇒ 本节的值一律**锚定它自己的标签**。

**本节指纹的产物锚点（P03，2026-09-28 补齐）**：以下值**逐项**从 `MEASURED` 产物的
`reproducibility` 段抄录（样例 `S14B2-load_ramp-ramp_static`，`git_commit=7051ea62`）：

| 项 | 产物字段 | 值 |
| --- | --- | --- |
| 设备 / 卡数 | `reproducibility.device_model` / `.device_count` | `MTT S4000` / `1` |
| 驱动 | `hardware_env.device.mthreads_gmi_info` 的 `Driver Version` 行 | `2.7.0` |
| gmi 工具 | 同上首行 | `mthreads-gmi:1.14.0` |
| SDK | `hardware_env.discovery.values.MUSA_VERSION`（`sources` = `file:/usr/local/musa/version.json`） | `3.1.0` |
| 框架 | `software_env.torch_version` + `hardware_env.software.torch_musa_version` | `2.2.0` + `1.3.0+81caf0a` |
| 模型 / 精度 / 并行 | `reproducibility.model` / `.precision` / `.tensor_parallel_size` | `Qwen/Qwen2.5-7B-Instruct`（`model_revision=a09a3545…`）/ `fp16` / `1` |
| 批配置 | `reproducibility.batch_configuration` / `.max_batched_tokens` / `.chunk_size` | `policy=continuous_batching;max_num_seqs=8` / `8192` / `1024` |
| 溯源 | `reproducibility.git_commit` / `.git_source` / `.git_dirty` | `7051ea62…` / `stamp` / `false` |

⚠️ **产物里唯一"取不到"的派生字段：`software_env.musa_version` 恒为 `None`**（遍历 283 份
产物，无一例外）。**这不是没采集**：SDK 版本原文在 `hardware_env.discovery.values.MUSA_VERSION`
里（见上表）。原因是**接线断了一节**——`telemetry.software_env()` 只从 `mthreads-gmi` 文本
解析驱动/MUSA（而该主机 `mthreads-gmi` **不打印独立 SDK 行**），**没有**接
`discovery.values`。按"取不到就 `None`、不猜"的设计，这是**正确的沉默**，不是错值。
⇒ 引用本节 SDK 值时**锚 `discovery.values.MUSA_VERSION`**，不要引用派生字段。

⚠️ **一处必须一起读的完整性事实**：24 份产物的 `hardware_env` 整块退化为
`{available_source: "missing", reason: "未找到 env.json…"}`（`inspect_env.sh` 未投该文件）——
按标签分：18 份是 `TEST_DATA`（本机 `simulate`，缺得合理）、6 份是 `NOT_MEASURED`（真机 pilot），
而 **`MEASURED` 产物 258/258 全部带完整指纹** ⇒ **本节结论不受影响**。
⚠️ 但因缺 gmi 原文者 `mthreads_driver_version` **也是 `null`** ⇒ **跨批次比"驱动换没换"前，
先确认两边都有指纹**，否则会把"没采到"读成"没装"。

## 3.2 模型与执行形状

被调度的"一步"可以用三个量加一个计数描述：`batch`（正在解码的序列数）、
`prefill_tok`（这一步新吃进的 prefill token 数）、`kv_len`（上下文长度）、
`n_forwards`（本步串行前向数）。**真实服务步的形状分布**由 `scripts/check_model_coverage.py`
在真实轨迹上普查得到 —— 它同时是"标定网格该往哪里补"的依据（§4.2 的覆盖问题由此提出）。

被测负载之一的形状（`short_chat` 族）：求入长度 1536、并发 16；另一族为
`phase1_anchor`（prompt 恒 2048、gen 128）。⚠️ `chunk_size = 2048` 在该负载上
**等于整段 prompt** ⇒ "分块"是**恒等操作**，与"不分块"**构造性等价**（§6 的响应面由此只有 4 个
不同配置，而不是 5 个）。

⚠️ **字段口径**：执行侧记录的 `prefill_tok` 是**全步总量**（一步内所有序列新增的 prefill token 之和），
而"每条序列各做多少"是另一个量。这两者的混淆是 §4.3 的主题，**本节只登记字段定义**。

## 3.3 块 A 与块 B：不能只拟合快态最小值

* **块 A（快态平台）**：固定形状、连续提交（async 块：连续提交 N 次、块尾 sync 一次 ⇒ 占空比 ≈100%），
  取**块最小值**，测的是稳态吞吐**下界**；
* **块 B（服务协议）**：逐步 `sync` + KV 真实生长，即真实服务路径。

实测两者差值随 prefill 增大：**b1/pf512 时 156.03 ms（A） vs 166.85 ms（B）**。
⇒ 结论（事实）：**块 A 只能作为性能下界参考**，用它拟合 SLO 预测会**系统性偏乐观**。

⚠️ **这两数不是 `MEASURED`**（设备侧探针、无 `run_id`、不盖标签）⇒ 只作平台特征，
不进性能结论表。此外有一条**长期有效的标定纪律**：**形状比值可信（会话内 ±0.3%）、
绝对值跨会话漂移 1.3–4%** ⇒ 成本模型只能用**同会话内**对照，不得跨会话并表。

设备还有**多档性能态**（b1 快 ~57–60 ms / 慢 ~76–82 ms，在几秒尺度双向翻转），
**占空比是开关**；它直接决定"怎么测"。因此本项目的可信标定协议固定为
**async 块 + warmup + 取块最小值**。
⚠️ **有限观察**（不是"已排除全部原因"）：采样时刻看不到 DVFS（gfx clock 恒 1660 MHz）、
热节流（恒 42 °C）、进程争抢（无其他 GPU 进程）—— 这不等于这些机制不存在，
尤其 `clock 的 min == max` 可能本身就是被限频策略钉住的**结果**。

## 3.4 分层信号：为什么裸 TPOT 会被 prefill 污染

调度器要读一个"延迟信号"来决定动作。若直接读**每步的 TPOT**，那么"这一步做了 prefill"
会让该步对应的 token 间隔被拉长，于是读数反映的是**这一步干了多少活**，而不是"服务是否在变慢"。

**实测（登记行 E10，`P8FULL`，`load_ramp`，3 轮）**：一次观测窗内，控制器读到的
`tpot_p90` 是 **337.2 / 337.9 / 342.2 ms**，而同一窗口里**纯 decode 步**的 `fwd_ms` p50 只有
**61.4 / 61.6 / 62.5 ms（n=28）**、**含 prefill 的步** p50 为 **329.3 / 329.0 / 331.5 ms（n=4）**
⇒ **读数 ≈ 真实 decode 的 5.5 倍，且几乎全部来自那 4 个 prefill 步**。

⚠️ 后果**不只是"读数错"**：这 9 次动作**全部是收缩**（把 `max_batched_tokens` 从 8192 压到
下界 2048），**纯 decode 步的 `fwd_ms` p50 随之从 61/62/62 ms 升到 118/118/118 ms（≈ +92%）**
—— 动作本身把真实的 decode 延迟弄慢了近一倍，而它要修的"问题"（337 ms）在纯 decode 口径下
并不存在。**触发时 `queue_depth` 三个 run 都是 0**（即**没有人排队时收缩**）。

⇒ 本项目的判据因此**冻结为 `decode_p90`**（只统计纯 decode 步），并把
`signal_mode: layered` 写进配置与产物（`describe()` 记录 `signal_mode` / `signal_criterion`）。

⚠️ 这条冻结不是"更好的读数"的口味问题：它是**判据的可比性前提** ——
换读数会改变调度（改门限 / 改判据都会改调度 ⇒ 必须重跑）。
⚠️ 同时注意：**用户真实经历的 prefill 停顿必须继续进入 ITL / TPOT / stall / SLO**，
不得用"纯 decode"美化对外指标。分层是**内部诊断量**，不是对外指标。

**该改动的干预实验（登记行 E12，单变量 `signal_criterion`，2 臂 × 3 轮）**：
换判据后**报告值 ÷ 同窗纯 decode** 从 **5.47 / 5.47 / 5.73** 降到 **1.33 / 1.33 / 1.32**
（即报告的就是纯 decode 值），且"在纯 decode 还只有 61 ms 时收缩"的动作消失
（首次动作由 `t ≈ 123.0 s` 推迟到 **`t ≈ 131.2 s`**，且触发值 138.0 **> 门限 129.6**
⇒ 属**合法触发**：8 槽满批时纯 decode 真的变慢）。
但**净收益很小、且代价是 stall**：达标率 **0.1957 → 0.2065**（**18 → 19 个请求**）；
`tpot` p50 **182.53 → 180.79**；**`stall` p99 330.51 → 410.30（回到 static 的 412.4 水平）**。
⇒ 机制成立，但这是**权衡**，不是"换判据就更好"。

## 3.5 判据的组成部分：整步 ≠ 前向

调度器比的是"这一步还来不来得及"，而**模型只预测前向**。

**实测（登记行 S14B1，`scripts/decompose_step_overhead.py` 的 65 组配对）**：
**整步 130.52 ms = 前向 104.79 ms + host 25.25 ms**（`pack_ms` 24.94 + `readback_ms` 0.31）。
模型对**前向**是中位 **−4.4%**（准），对**整步**是 **−23.2%**。

⇒ 判据必须写成 `预测整步 = 前向预测 + host(该 batch)`，其中 host 是**本机 + 当前执行路径**的量
（含 python 侧打包 / mask 组装），**不是**芯片性质 ⇒ 换执行路径或改 batch 上限**必须重测**。

⚠️ **判据口径的三项必须进产物**（`host_overhead_ms` / `host_overhead_fit` / `apply_safety_margin`），
否则事后无法区分"这次跑的是哪种口径"。本项目实测踩过：两批产物的 `config.controller`
在这一点上**完全相同**，只能靠配置 + `git_commit` 反推。
【待补·P04】该修复后的产物自证示例（来源：`S14G` 起的批次；当前状态：修复已落地，批次待跑）。

⚠️ **一条与"改判据"同源的纪律**：`host` 项**不进模型**（模型不负责 host），
进的是**判据**；且它与"模型精度"是两件事——**"模型很准"与"判据偏松"可以同时成立**
（−4.4% vs −23.2% 就是同一次测量里的两个数）。

## 3.6 复现性与数据完整性

* 每个产物带 `git_commit` + `git_dirty` + `data_label` + **实际生效的判据**（`describe()`）；
* 同一批（同一 tag）内**不得**混不同 `git_commit` / `dirty`；
* `dirty=true` 的批次**不得**与其它批次合并解释；
* 远端源码树是**拷贝**过去的、不带 `.git` ⇒ 用**部署戳**（`SOURCE_PROVENANCE.json`）钉住版本，
  并在投放后**立刻用远端那棵树重算摘要**校验；戳会过期，不符即作废并把 `git_commit` 置空
  （**错的 commit 比缺的 commit 危险**）；
* ⚠️ **戳的作用域不含 `configs/`** ⇒ "代码是这个提交"**推不出**"配置是这一版"，
  配置侧须逐 run 比对 `config.snapshot.yaml`（本项目已做，并用一个**反向对照**证明该检查**能失败**）；
* 未测的值保持缺失（`-`），**不得**填 0（0 是测量值，与缺失语义相反）。
