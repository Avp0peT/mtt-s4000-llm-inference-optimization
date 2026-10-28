# 项目工作整理与当前状态

整理日期：**2026-10-28**。本页所依据的实验与离线交付记录截止 **2026-10-03**；整理日期不代表新增实验日期。本次整理没有运行新的模型推理、GPU 测量或性能矩阵，也没有刷新远端环境状态。

项目已经具备策略、执行、测量、分析和证据追踪代码。已有一项范围明确的 S01 首 token 时延改善得到真机证据支持；新执行形状契约及成本预测链路完成离线验证，尚无通过该契约独立验证的真机成本模型。**目前不能声称新控制器有效，或整体服务质量已经改善。**

## 研究定位

研究主体是 **MTT S4000 / MUSA** 上的 Qwen2.5-7B-Instruct 推理。当前已有主要实验采用 Transformers、eager 模式和自建迭代级执行器；这些结果不代表 vLLM-MUSA 或其他执行引擎的性能。Ascend 支持作为次平台保留，两平台的数据不得合并比较。

研究主线是**成本预测的可靠性、预测与实际执行失配所产生的调度代价，以及方法的适用边界**。请求感知的批处理、前缀缓存和预填充分块是被研究的机制；实现了这些功能、覆盖了某张查找表或通过了单元测试，都不足以证明新的研究贡献或控制器优势。

定位依据：[RESEARCH_PLAN.md](../RESEARCH_PLAN.md) 的 2026-10-03 状态更新；研究与证据规则见 [AGENTS.md](../AGENTS.md)。

## 已完成的工作

| 层次 | 已完成内容 | 当前可支持的判断 |
| --- | --- | --- |
| 代码框架 | 配置驱动的负载生成、指标采集、策略与控制器、迭代执行、批次编排、结果分析和源码溯源 | 能审查实现、执行离线验证和准备可追踪实验；功能存在不等于性能有效 |
| 执行优化 | S01 logits/采样裁剪、S02 缓存释放控制、S03 线性层融合、S04 常驻 KV 缓冲等实现与对应测试 | 不同分支的真机与证据资格不同，不能合称为“全部优化成功” |
| 执行形状修复 | 计划与实际驱动共享参与请求、逐请求 token/KV、`need_token` 和前向分组；修正步后队列、decode KV 与成本排序 | 已通过离线及 CPU 微型模型检查；当前生产负载上的匹配率和覆盖率仍待真机核验 |
| 成本模型工具 | 整步标定探针、训练冻结与独立验证、模型导出、pilot 配置派生、决策链重放 | 工具完成离线验证；尚未生成满足新契约的合格实测模型和正式派生配置 |
| 实验与证据管理 | 真机结果登记、正负结果分析、身份校验、T1/T2/T3 资格分层、保留历史更正 | `MEASURED` 表示实际测量，不自动获得论文量化引用资格 |

主要代码入口：

- 策略和共享形状：[src/ascend_opt/](../src/ascend_opt/)，重点是 [execution_shape.py](../src/ascend_opt/execution_shape.py)、[step_planner.py](../src/ascend_opt/step_planner.py)、[cost_model.py](../src/ascend_opt/cost_model.py) 和 [controller.py](../src/ascend_opt/controller.py)。
- 执行与采集：[benchmark/](../benchmark/)，包括 [real_backend.py](../benchmark/real_backend.py) 和 [exec_opt_plan.py](../benchmark/exec_opt_plan.py)。
- 配置、脚本与测试：[configs/](../configs/)、[scripts/](../scripts/) 和 [tests/](../tests/)。

最新离线交付记录的验证结果为 **21 个相关模块联合回归：375 passed、87 subtests passed，0 fail/error/skip**。后续重放 CLI 定向检查为 23 passed、10 subtests passed；两组有重叠，不能相加，也不代表重新运行了整个历史测试仓库。这些是截至 2026-10-03 的既有记录，本次整理未重新执行。

CPU 微型模型验证仅具有 `LOCAL_CPU_MICRO` 资格；合成状态、FakeDriver 和 dry-run 保持 `TEST_DATA`。它们支持逻辑与链路检查，不能作为 S4000 性能、模型误差或生产覆盖率证据。详见 [离线交付报告](../handoff/optimization_20260929/OFFLINE_EXECUTION_SHAPE_20261003.txt)。

## 真机结果与可以下的结论

### S01：限定条件下，TTFT 改善得到支持

`S01RECHECK` 对比 `logits_last_only` 与 `sample_needed_only` 两个 S01 开关的关闭/开启状态，共 **2 臂 × 5 轮**，10 份产物。条件为 `short_chat`、并发 8、每轮 64 请求、控制器关闭、S04 关闭、`max_num_seqs=32`、预填充预算 8192，并使用上述 Transformers eager 执行路径。

| 指标 | 对照臂中位数 | S01 臂中位数 | 逐轮配对比值的中位数 | 解释 |
| --- | --- | --- | --- | --- |
| TTFT p50 | 190.6 ms | 182.9 ms | 0.9569，即 **−4.31%** | **5/5 轮更快**，支持这一范围内的 TTFT 改善 |
| TTFT p95 | 914.3 ms | 877.1 ms | 0.9587，即 −4.13% | 5/5 轮更快 |
| TPOT p50 | 117.8 ms | 117.6 ms | 0.9994，即 −0.06% | 方向不一致，**不足以判定改善** |
| 输出吞吐 | 64.2 tok/s | 64.4 tok/s | 1.0021，即 +0.21% | 4/5 轮改善，不能表述成整体显著提速 |
| SLO 达标率 | **2/64 = 3.125%** | **2/64 = 3.125%** | 1.0000 | **没有提升**；比值为 1 不代表达标率为 100% |

这里的百分比来自**逐轮配对比值的中位数**，不能用表中两臂中位数直接相除来替代。低绝对达标率仍表明当前服务质量存在问题，S01 的 TTFT 改善没有解决这一问题。

该批源码、环境、配置/模型身份与判据登记形成闭环，现有证据审计将其列为 **T1，可作限定范围的量化引用**。判据随分析脚本版本化；没有独立的事前封存副本，不能升级表述为“已独立预注册”。依据：[EXPERIMENTS.md](../EXPERIMENTS.md) §3.9、[EVIDENCE_INDEX.md](../EVIDENCE_INDEX.md) §5、[分析脚本](../scripts/analyze_s01_recheck.py)。

### S04：保留局部负面结果，默认关闭

修复后的 S04 已完成重入路径数值验证；这项正确性证据不等于性能收益。在 `EXECOPT5` 的 5 轮独立复现中，快路径真正行使的末尾 77 步纯 decode 样本上，主机计时边界下的 `pack` 成本约 **+70.9%**，5/5 同向；全窗口约 **+0.37%**。

77/1143 = 6.74% 是前向组的**数量覆盖率**，不是耗时权重。上述 `pack` 字段也不能直接解释成 GPU 搬运时间。该负面结果只约束当前整段 KV 复制实现与当前负载，不可外推为所有常驻缓冲方案无效。

`EXECOPT5` 的环境指纹与实际执行容器不一致，因此性能证据仍为 **T2，仅作机制诊断和内部决策**；不能直接作为论文量化主张。历史 `EXECOPT3/4` 也有源码身份等缺口，保留 T2 资格。S04 继续默认关闭，停止重复同协议性能批次；图捕获与混长扩展仍未因此获准推进。依据：[EXPERIMENTS.md](../EXPERIMENTS.md) §3.7–§3.8、§4.4，以及 [EVIDENCE_INDEX.md](../EVIDENCE_INDEX.md) §5。

### 新成本控制器：效果尚未验证

最新代码要求 `execution-shape-v1`：一个预测点必须对应完整驱动逻辑步及其实际执行分组，绑定执行开关、硬件/软件/权重和执行源码身份。

**旧标量 `model_v7` 不可直接用于新契约下的生产成本臂。** 当前返回 `execution_contract_unvalidated`，表示契约未验证；不能把它解释成预算不可行或模型已经被实测否证，也不能拼加旧标量采样点来充当新整步模型。

离线清单来自四个合法独立合成状态：32 次候选评估、14 处旧 batch 输入修正、6 个去重整步形状。它用于准备验证，不证明生产到达轨迹可达性、全负载覆盖率或未见形状泛化。预测只绑定下一实际步骤；新到达、补位或执行组变化造成的不同形样本，不具备直接计算模型预测误差的资格。

因此，当前没有“新控制器优于静态基线”的结论，也没有满足新契约的合格真机成本模型。历史成本模型的负面结果与适用限制继续保留。依据：[当前协议](../NEXT_BATCHES_PREREG.md) §13、[最新状态](../handoff/optimization_20260929/STATE.json) 和 [离线交付报告](../handoff/optimization_20260929/OFFLINE_EXECUTION_SHAPE_20261003.txt)。

## 下一步执行顺序

以下为**尚待执行的流程**，不是已完成记录。完整操作说明在 [集中真机执行计划](../handoff/optimization_20260929/REMOTE_EXECUTION_PLAN_20261003.txt)，验收口径在 [NEXT_BATCHES_PREREG.md](../NEXT_BATCHES_PREREG.md) §13。

1. **环境与部署身份验收，再做短数值验收。** 核对实际执行环境、权重、代码和配置身份，使用 [verify_s01_numeric_acceptance.py](../scripts/verify_s01_numeric_acceptance.py) 验证当前代码；失败即保留证据并停止后续实验。
2. **3 个独立训练进程标定，再冻结训练表。** 用 [execution_shape_probe.py](../scripts/execution_shape_probe.py) 采集完整逻辑步成本，按协议记录前向与 host 成本，不重复计费。当前清单每进程每形状 30 条记录，逐形状预热至少 30 秒。
3. **冻结后执行 3 个独立验证进程。** 验证数据不参与改表；仅比较同形状跨进程数据。按协议检查低估相对误差 p95 ≤5%、绝对相对误差 p90 ≤10%、跨会话漂移 ≤10%。缺身份、证据不完整或验证失败，均不得导出正式模型。
4. **导出模型并派生新配置，完成离线重放。** 使用 [export_execution_shape_model.py](../scripts/export_execution_shape_model.py)、[prepare_execution_pilot.py](../scripts/prepare_execution_pilot.py) 和 [dryplan_decision_chain.py](../scripts/dryplan_decision_chain.py)；历史模型和冻结配置保留原样。
5. **运行恰好 3 项诊断 pilot。** `baseline`、`cost_single`、`cost_dual` 各一次，由 [pilot_execution_plan.py](../scripts/pilot_execution_plan.py) 核对清单。先报告形状匹配率、预测覆盖率、回退原因和动作是否真正行使，再决定是否扩大。
6. **通过上述门槛后，另行开展正式对照、重复实验与论文工作。** 需要匹配条件的静态/成本臂对照、消融、统计分析和可复算图表。三项单次 pilot 本身不能支持正式收益结论。

S01RECHECK 同协议性能批次不重跑，S04 保持默认关闭；已撤回的固定 9 点旧补标定方案不再执行，也不自动扩大成 25 项收益矩阵。总机时尚未知，需先测首个训练进程的真实节拍；矩阵的 45 分钟预算参数不是整条流程的墙钟上限。

## 证据查阅与发布范围

本页是阅读入口，不替代完整实验记录。以下路径均相对仓库根目录；原始结果保持只读，缺失值、失败、否定结果和历史证据缺口不得通过整理或发布而改写。

| 内容 | 完整证据位置 |
| --- | --- |
| 研究定位与实验登记 | `RESEARCH_PLAN.md`；`EXPERIMENTS.md`；`RESEARCH_LOG.md`；`EVIDENCE_INDEX.md` |
| 最新交付与恢复状态 | `handoff/optimization_20260929/STATE.json`；同目录 `STATE_FULL_20261003.json`、`OFFLINE_EXECUTION_SHAPE_20261003.txt`、`REMOTE_EXECUTION_PLAN_20261003.txt` |
| S01 真机性能 | `results/raw/S01RECHECK-*`（10 个目录）；`results/processed/s01_recheck_analysis.json`；`scripts/analyze_s01_recheck.py` |
| S04 性能与正确性 | `results/raw/EXECOPT4-*`、`results/raw/EXECOPT5-*`；`results/raw/_s04_verify/`；`scripts/analyze_exec_opt_batch.py` |
| 执行形状候选与清单 | `results/processed/dryplan_execution_shape_20261003.json`；`results/processed/execution_shape_manifest_20261003.json` |
| 最新离线回归 | `results/processed/offline_execution_shape_tests_final_20261003.log` 与 `.xml`；`results/processed/offline_execution_shape_replay_cli_20261003.log` 与 `.xml` |
| CPU 与合成链路证明 | `results/processed/s01num-offline-rolefix-*.json`；`results/processed/execution-shape-cpu-*.json`；`results/processed/prediction-execution-match-*.json` |

如果发布包没有附带某项历史记录或实验产物，请按上列相对路径查阅另行提供的数据/交接目录及其说明。缺少对应证据时，不能仅凭本页摘要独立复核数字，也不能将缺失材料理解为已通过验证。

历史派生产物 `results/processed/dryplan_decision_chain.json` 曾被误覆盖，原字节未恢复；现有记录明确将其标为失效。新证据应从上述带日期的执行形状产物读取，不能把旧占位文件当作有效结果。原始实验数据、历史模型及冻结配置未因此改写。
