"""成本控制器：用**预测的每步成本**在有限候选里选动作（第 12 步）。

与阈值控制器的关系
------------------
并列的两条臂（`AdaptiveController` = 阈值；本模块 = 成本模型）。
**两者复用同一个 `plan_step`**（`step_planner`）与同一套 `ChunkLimits`（运行期上限的
**唯一**权威）—— 只换"怎么挑"，不换"挑什么"。这样"策略差异"才与"执行路径差异"分离。

预测的形状怎么来
----------------
对每个候选 `(chunk, max_tokens)` 调一次 `plan_step`，得到它**真的会产生的**计划，
再把计划折成成本模型的输入形状：

生产工厂启用 ``execution-shape-v1``：从共享的计划翻译函数提取本步真正获分配
的请求，包含续 prefill，排除只接纳而未计算的请求；保留每条 token/KV/need_token
以及按 token 数降序的实际前向组。模型必须精确命中整个逻辑前向向量与执行签名。
未验证的旧标定不自动复用，多组成本不由单组成本擅自相加。

旧 ``len(decode_ids)+len(new_start_ids)`` 会把接纳数误当执行数。仅保留在直接构造
无执行签名策略的历史标量重放 API 中，明确标 ``legacy_scalar_diagnostic``；
它不提供生产执行证据，``build_arm`` 不会进入该路径。

⇒ 预测的是"**这个候选的执行计划**"的成本，而不是"这个候选本身"的成本。

★ 预先记录：每步预算（per-step deadline）的映射
-----------------------------------------------
本模块引入一条**每步预算** `step_deadline_ms := target_tpot_ms`（**直接取已标定的 SLO 阈值**，
不新增一个待标定的数）：

* **映射依据**：一步里 `batch` 条序列并发解码，每条序列**每 token 经历的等待**≈这一大步的墙钟
  ⇒ `predicted_step_ms ≤ target_tpot_ms` 就是把"逐请求 TPOT ≤ 目标"表达成**每步**的形式。
* ⚠️ **它不是 `TTFT` 约束，也不是逐 token `ITL` 约束**（验收明确说这三者不是同一个约束）：
  本模块预测并约束的是**步成本**，它**只**约束解码序列的每 token 延迟。
  首 token 延迟由"prefill 多久能被推进"决定 ⇒ 那条由**第二优先规则**（本步 prefill 进度）
  承担，**不由**成本模型预测。
* ⇒ 因此本模块**不声称硬 SLO 保证**：它保证的是"在模型成立的前提下，选出的计划不会让
  步成本超预算"，并**保守回退**（见下）。

★ 双预算（方案 A，`B1` 草案 §1.4 / OPT-03R 裁定 Q1）
----------------------------------------------------
**单一** per-step deadline 与"整块 prefill 步"**在构造上不相容**：`prefill_tok ≥ 512`
的步前向 ≈135 ms 已超 120 ms ⇒ cost 臂在这些步上**必然** degraded/fallback。
用**同一个数**同时约束 TPOT 与 TTFT，是把两个不同的承诺混成一个数。
⇒ 本模块支持把预算拆成两条（**默认不启用**，`prefill_deadline_ms=None` 时行为逐字节不变）：

* `step_deadline_ms` = **decode 预算**（服务 TPOT）
* `prefill_deadline_ms` = **prefill 预算**（服务 TTFT）
* 归类由候选**自己的计划**决定（有 prefill 且无 decode 风险才用 prefill 预算）
* 每条候选落盘 `budget_ms` / `budget_kind` ⇒ "双预算是否真被用上"可**数出来**

⚠️ **不声称**拆分后预算就"更松"：它只是把两个承诺分开表达，取值仍须由标定决定。

选择规则（**预先记录**，避免事后调权重）
----------------------------------------
1. **可行** = 预测在范围**内** 且 `predicted ≤ step_deadline_ms` 且计划无 `stall_reason`；
2. 可行集非空 ⇒ 取 `tokens_prefilled` **最大**者（推进 prefill 就是压 TTFT）；
   并平则依次比：预测步成本更小 → `max_tokens` 更小 → `chunk` 更小（确定性 + 保守）；
3. 有预测但**全部超预算** ⇒ 取预测成本**最小**者，并标 `degraded=True`（**显式记录降级**，
   不静默）；
4. **一个预测都没有**（全越界 / 模型不可用）⇒ **回退到阈值控制器**并**计数**。

⚠️ 规则里没有手调的权重 —— 权重是"事后调参"最容易的入口。

在线输入的边界（验收第 3 条）
-----------------------------
本模块只看**当前**的队列/在解码/在 prefill 状态与**已等待时间**；
**不使用**未来到达、最终测试标签或事后真实输出长度。
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any, Sequence

from .chunk_policy import ChunkLimits
from .cost_model import CostModel, PredictedCost
from .step_planner import StepPlan, plan_step
from .execution_shape import plan_execution_shape
from .workload import Request

__all__ = ["CostPolicyConfig", "CostPolicyState", "CandidateEvaluation", "CostDecision",
           "CostPolicy"]


@dataclass(frozen=True)
class CostPolicyConfig:
    """成本控制器的配置。**默认关闭** ⇒ 既有批次的行为逐字节不变。"""

    enabled: bool = False
    # All build_arm production callers supply this. None retains the historical
    # scalar research API for old replay/tests; it is not execution evidence.
    execution_signature: dict[str, Any] | None = None
    #: 候选的分块大小与单步 token 预算。**不是**"无限搜索"，是有限候选集。
    candidate_chunks: tuple[int, ...] = (512, 1024, 2048)
    candidate_max_tokens: tuple[int, ...] = (2048, 4096, 8192)
    #: 每步预算（= 逐请求 TPOT 目标；映射依据见模块 docstring）。
    #:
    #: ★ **B1 §1.4 方案 A（双预算）**：本字段即 **decode 预算** —— 服务 TPOT。
    #: 当 `prefill_deadline_ms` 为 `None`（默认）时，**它就是唯一的预算**，
    #: 行为与引入双预算之前**逐字节相同**。
    step_deadline_ms: float | None = 100.0
    #: ★ **prefill 步的预算**（ms）—— 服务 **TTFT**。`None`（默认）= **不启用双预算**。
    #:
    #: 为什么需要（`B1` 草案 §1.4 结论 1，构造性发现）：单一 per-step deadline
    #: 与"整块 prefill 步"**在构造上不相容** —— 一个 `prefill_tok ≥ 512` 的步
    #: （前向 ≈135 ms）已经超 120 ms ⇒ cost 臂在这些步上**必然** degraded/fallback
    #: （S14B2 实测 `n_degraded ≈ 150`）。用**同一个** deadline 同时约束 TPOT 与 TTFT，
    #: 是把两个不同的承诺混成一个数。
    #:
    #: 归类规则（**预先记录，无手调权重**）：按**该候选自己的计划**判定，**且**
    #: 要求**本步没有 decode 处于风险中** ——
    #: `plan.tokens_prefilled > 0` **且** `decode_at_risk == False`
    #: ⇒ 用本预算；否则用 `step_deadline_ms`。
    #:
    #: ⚠️ **2026-10-03 修（缺口 §5.1）**：原规则只看 `tokens_prefilled > 0`，
    #: 于是本步还有 decode 在跑（或没被服务）时，那些 decode 也被按**本预算**
    #: （如 2000 ms，而 decode 只有 100 ms）放行 ⇒ 实测 12×2048 的等待局面下
    #: 有 6 个候选被 prefill 预算放行、而按 decode 预算会不可行。
    #: 方向是**放行过松**：逐 token 护栏在这些步上**根本不生效**，
    #: 而产物里 `budget_kind="prefill"` 看上去完全正常。
    #: ⇒ 现在：**只要本步服务了 decode token，或让某个 decode 请求没拿到 token**，
    #: 就必须用 `step_deadline_ms`。本预算只在**纯 prefill 步**适用。
    #: 详见 `_budget_for` 的 docstring（两个护栏的语义对照表）。
    #:
    #: 这样一步内不同候选可各自归档，不引入"整步级"分类的歧义。
    #:
    #: ⚠️ 语义提醒：这不是"放宽预算"——若把 prefill 预算设得比 decode 大，
    #: 那是**明确声明**"纯 prefill 步的墙钟可以更久，因为它不服务逐 token 延迟"。
    #: 取值本身须由标定决定，**不得**为凑结论事后调。
    #: ⚠️ 本预算的取值应当与 `slo.ttft_ms` **同源**（服务的是同一个承诺），
    #: 不一致时要在预注册里写明理由。
    prefill_deadline_ms: float | None = None
    #: 预测不可用（越界/无模型）时是否回退到阈值控制器。默认**回退**。
    fallback_to_threshold: bool = True
    #: **逐步的 host 侧开销**（ms，按 batch）—— 这一步**必须计入预算判据**。
    #:
    #: 为什么：模型预测的是**前向**（`fwd_ms`），而 deadline 约束的是**整步**。
    #: 实测（`scripts/decompose_step_overhead.py`，S14B1 的 65 组配对）：
    #: 整步 130.52 ms = 前向 104.79 + **host 25.25**（`pack_ms` 24.94 + `readback_ms` 0.31），
    #: 而**模型对前向是中位 −4.4%（准的）**，对整步是 −23.2%。
    #: ⇒ 不把 host 算进去，就是"拿前向的预测去比整步的 deadline"⇒ 判据口径错，
    #: 这直接造成 D1（预测可行、实际超时）= 84.6%。
    #:
    #: ⚠️ **它不进模型**（模型不负责 host）；而且它是**本机 + 当前执行路径**的量，
    #: 换路径/换 batch 上限都要重测。
    host_overhead_ms: dict[int, float] | None = None
    #: 上表的线性拟合 `(斜率, 截距)`（ms 对 batch），用于表里**没列到**的 batch。
    #: 实测三点 (2,3.06)/(4,10.76)/(8,25.29) ⇒ 斜率 ≈ 3.705、截距 ≈ −4.35（拟合里 b4 预测 10.47 vs 实测 10.76）。
    host_overhead_fit: tuple[float, float] | None = None
    #: 是否在预测上乘**安全余量**（模型文件里的 `safety_margin`，= 留出形状上的低估 p95）。
    #:
    #: 默认 ``True``（既有行为**逐字节不变**）。
    #: 存在的理由：第 15 步消融要求"**去掉误差裕量**"这一条能被**配置**表达。
    #: ⚠️ 不要用"改模型文件里的 `safety_margin`"来表达它 —— 那会派生出一个
    #: 与标定结果不一致的第二份模型文件（同一份标定两个真相）。
    #: ⚠️ 如实说明边界：`safety_margin` 是**特定留出划分下**的留出误差，**不是泛化精度**
    #: （见 `model_v*.json` 的 `accuracy_caveat`）⇒ 去掉它测的是"这一项有没有效果"，
    #: 不是"低估尾部是否已被覆盖"。
    apply_safety_margin: bool = True


@dataclass
class CostPolicyState:
    """计数器。**回退与降级都要能数出来** —— 数不出来的失效等于没发生。"""

    n_decisions: int = 0
    n_fallback: int = 0
    n_degraded: int = 0
    n_no_candidates: int = 0
    #: 因显存压力而改取**最保守**候选的次数。
    n_memory_conservative: int = 0
    #: **prefill 饥饿**的次数：有 prefill 积压、而所有候选本步都推进不了 prefill。
    n_prefill_starved: int = 0
    #: **空闲步**的次数：所有候选都 `batch==0` 且无 prefill ⇒ 本步没活可做。
    #: ⚠️ 必须与 `n_fallback` **分开**：把"空闲"算成"回退"会让回退率被空闲步淹没，
    #: 那个计数器就再也反映不出"模型是否在用"（实测踩到：25000 次决策里 23598 次"回退"，
    #: 绝大多数其实是空闲步）。
    n_idle: int = 0
    last_fallback_reason: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "n_decisions": self.n_decisions,
            "n_fallback": self.n_fallback,
            "n_degraded": self.n_degraded,
            "n_no_candidates": self.n_no_candidates,
            "n_memory_conservative": self.n_memory_conservative,
            "n_prefill_starved": self.n_prefill_starved,
            "n_idle": self.n_idle,
            "last_fallback_reason": self.last_fallback_reason,
        }


@dataclass(frozen=True)
class CandidateEvaluation:
    """一个候选的**完整**评估记录：候选、计划、预测、可行性与否决原因。

    验收要求"记录候选、成本、选择原因" ⇒ 被否决的候选也要留（否则"为什么没选它"
    事后无从回答）。
    """

    chunk: int
    max_tokens: int
    #: `None` = 预测不可用（越界或模型缺失）—— **不填 0**。
    predicted_ms: float | None
    tokens_prefilled: int
    batch: int
    #: **决策的输入必须落盘**：没有它，事后无法把候选形状**重放**一遍
    #: （换预测方式、换模型时都要重放）⇒ 实测踩到过：想量"查表方式的覆盖率"却
    #: 因为轨迹里没有 `kv_len` 而重放不了。默认 0 只为兼容旧调用点。
    kv_len: int = 0
    feasible: bool = False
    #: ★ **B1 §1.4 方案 A（双预算）落盘字段**：该候选被判定时**用的预算**（ms）。
    #: `None` = 该候选没有预测（无从判定）⇒ **不填 0**（0 会被读成"预算为 0，必超"）。
    budget_ms: float | None = None
    #: 该预算的**种类**：`"decode"`（服务 TPOT）或 `"prefill"`（服务 TTFT）。
    #: 单独记它，是为了让"双预算是否真的被用上"能被**数出来**，而不是靠读代码相信。
    budget_kind: str = ""
    reject_reason: str = ""
    execution_shape: dict[str, Any] | None = None
    legacy_batch: int | None = None
    coverage_status: str = "legacy_scalar_diagnostic"
    coverage_reason: str = "历史聚合接口，不具备生产执行形状验证资格"
    host_ms: float | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk, "max_tokens": self.max_tokens,
            "predicted_ms": (None if self.predicted_ms is None else round(self.predicted_ms, 3)),
            "tokens_prefilled": self.tokens_prefilled, "batch": self.batch,
            "kv_len": self.kv_len,
            "feasible": self.feasible, "reject_reason": self.reject_reason,
            "budget_ms": (None if self.budget_ms is None or not math.isfinite(self.budget_ms)
                          else round(self.budget_ms, 4)),
            "budget_unbounded": self.budget_ms is not None and math.isinf(self.budget_ms),
            "budget_kind": self.budget_kind,
            "execution_shape": self.execution_shape,
            "legacy_batch": self.legacy_batch,
            "coverage_status": self.coverage_status,
            "coverage_reason": self.coverage_reason,
            "host_ms": self.host_ms,
            "predicted_step_ms": (None if self.predicted_ms is None or self.host_ms is None
                                  else self.predicted_ms + self.host_ms),
        }


@dataclass
class CostDecision:
    changed: bool
    reason: str
    chunk: int | None = None
    max_tokens: int | None = None
    candidates: list[CandidateEvaluation] = field(default_factory=list)
    degraded: bool = False
    #: 非空 = 本步**回退**了，内容即原因（与 `CostPolicyState.n_fallback` 配对）。
    fallback_reason: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "changed": self.changed, "reason": self.reason,
            "chunk": self.chunk, "max_tokens": self.max_tokens,
            "degraded": self.degraded, "fallback_reason": self.fallback_reason,
            "candidates": [c.describe() for c in self.candidates],
        }


class CostPolicy:
    """按预测的每步成本选 `(chunk, max_tokens)`。纯逻辑，不含任何硬件依赖。"""

    def __init__(self, config: CostPolicyConfig, model: CostModel | None = None) -> None:
        self.config = config
        self.model = model
        self.state = CostPolicyState()

    # ------------------------------------------------------------------ 候选
    def _candidates(self, limits: ChunkLimits, chunk_ceiling: int,
                    budget_ceiling: int) -> list[tuple[int, int]]:
        """按候选集生成 `(chunk, max_tokens)`，**再用运行期上限夹取**。

        ⚠️ 上限只在 `ChunkLimits` 一处权威（本仓库的红线）⇒ 这里只是"喂进去"，
        不自己再算一遍天花板。
        """
        if limits.chunk_size is not None:
            chunk_ceiling = min(chunk_ceiling, limits.chunk_size)
        if limits.max_num_batched_tokens is not None:
            budget_ceiling = min(budget_ceiling, limits.max_num_batched_tokens)
        out: list[tuple[int, int]] = []
        for c in sorted(set(self.config.candidate_chunks)):
            for m in sorted(set(self.config.candidate_max_tokens)):
                cc, mm = min(c, chunk_ceiling), min(m, budget_ceiling)
                if cc > 0 and mm > 0 and (cc, mm) not in out:
                    out.append((cc, mm))
        return out

    # ------------------------------------------------------------------ 决策
    def decide(self, *, scheduler: Any, chunk_policy: Any, cache: Any,
               waiting: Sequence[Request], decoding: Sequence[Request],
               prefilling: Sequence[tuple[Request, int]],
               limits: ChunkLimits, chunk_ceiling: int, budget_ceiling: int,
               max_num_seqs: int, chunk_enabled: bool, kv_len: int,
               step: int = 0, memory_veto: bool = False,
               memory_critical: bool = False,
               per_request_kv_lengths: dict[str, int] | None = None) -> CostDecision:
        """选一个 `(chunk, max_tokens)`。**没有任何候选可评估时明确回退**。

        ``memory_veto`` / ``memory_critical``
            与阈值控制器同源的两个显存信号（`available_memory_mb` 低于两个门限）。
            成本模型**不预测显存**，所以这两条必须由调用方给；
            `critical` 时本策略改取**最保守**候选（最小 `max_tokens`/`chunk`），
            理由：显存吃紧时"少给 token"是唯一不需要模型就能确定的安全方向。
        """
        self.state.n_decisions += 1
        cands = self._candidates(limits, chunk_ceiling, budget_ceiling)
        if not cands:
            return self._fallback("没有候选（候选集为空或上限夹到 0）")

        evals: list[CandidateEvaluation] = []
        for chunk, max_tokens in cands:
            plan: StepPlan = plan_step(
                scheduler=scheduler, chunk_policy=chunk_policy, cache=cache,
                waiting=waiting, decoding=decoding, prefilling=prefilling,
                cur_chunk=chunk, cur_max_tokens=max_tokens,
                max_num_seqs=max_num_seqs, chunk_enabled=chunk_enabled, step=step,
            )
            batch = len(plan.decode_ids) + len(plan.new_start_ids)
            pf = int(plan.tokens_prefilled)
            evidence: dict[str, Any] = {"legacy_batch": batch}
            exact = self.config.execution_signature is not None
            shape = None
            if exact:
                shape = plan_execution_shape(
                    plan, waiting=waiting, decoding=decoding, prefilling=prefilling,
                    per_request_kv_lengths=per_request_kv_lengths).describe()
                batch = shape["batch"]
                evidence.update(execution_shape=shape, coverage_status="no_model",
                                coverage_reason="没有加载成本模型")
            if self.model is None:
                evals.append(CandidateEvaluation(chunk, max_tokens, None, pf, batch, kv_len=kv_len,
                                                 feasible=False,
                                                 reject_reason="没有加载成本模型", **evidence))
                continue
            if exact:
                pred, status, detail = self.model.predict_execution(
                    shape, execution_signature=self.config.execution_signature,
                    apply_margin=self.config.apply_safety_margin)
                evidence.update(coverage_status=status, coverage_reason=detail)
            else:
                # Historical scalar replay only; build_arm never selects this.
                pred = self.model.predict(batch=batch, prefill_tok=pf, kv_len=kv_len,
                                          apply_margin=self.config.apply_safety_margin)
            if pred is None:
                evals.append(CandidateEvaluation(chunk, max_tokens, None, pf, batch, kv_len=kv_len,
                                                 feasible=False,
                                                 reject_reason=(evidence["coverage_reason"] if exact else
                                                                "模型范围外（不可外推）"), **evidence))
                continue
            # ⚠️ 判据是"**整步** ≤ 预算"，不是"前向 ≤ 预算"：
            # 模型预测的是前向，host 侧开销（打包/构造等）必须显式加上去。
            _over = pred.host_ms if exact else self.host_overhead(int(batch))
            evidence["host_ms"] = _over
            # ★ B1 §1.4 方案 A：**按该候选自己的成分**选预算（无手调权重）。
            # prefill 步服务 TTFT、纯 decode 步服务 TPOT —— 两者是不同的承诺。
            # ★★ 2026-10-03 修（缺口 §5.1）：上面那句"按候选自己的成分"**不够** ——
            #    还必须看**本步有没有 decode 处于风险中**。两种风险：
            #      ① 本步**服务**了 decode token ⇒ 这些 token 被 prefill 工作推后；
            #      ② 本步让某个 decode 请求**没拿到** token ⇒ 它的间隔被拉长。
            #    任一成立 ⇒ 逐 token 护栏必须适用（用 decode 预算），
            #    否则宽松的 prefill 预算会把这些 decode 放行到远超 TPOT 承诺。
            _served = set(plan.decode_ids)
            decode_served = bool(plan.decode_ids)
            decode_blocked = any(getattr(r, "req_id", None) not in _served
                                 for r in decoding)
            decode_at_risk = decode_served or decode_blocked
            _kind, _budget = self._budget_for(tokens_prefilled=pf,
                                              decode_at_risk=decode_at_risk)
            feasible = (pred.total_ms + _over) <= _budget
            reason = "" if feasible else (
                f"预测 {pred.total_ms:.1f} + host {_over:.1f} = {pred.total_ms + _over:.1f}ms "
                f"超{'prefill' if _kind == 'prefill' else '每步'}预算 {_budget:.1f}ms")
            evals.append(CandidateEvaluation(chunk, max_tokens, pred.total_ms, pf, batch, kv_len=kv_len,
                                             feasible=feasible, reject_reason=reason,
                                             budget_ms=_budget, budget_kind=_kind, **evidence))

        # **空闲步**短路：所有候选都是 batch=0 且没有 prefill ⇒ 本步没活可做。
        # 模型没有 batch=0 的标定点（也不该有：那是"什么都不做"），所以这里
        # **不查模型**、也**不算回退** —— 直接说"无变化"。
        if evals and all(e.batch == 0 and e.tokens_prefilled == 0 for e in evals):
            self.state.n_idle += 1
            return CostDecision(False, "本步无活可做（无 decode、无 prefill）",
                                candidates=evals)

        predicted = [e for e in evals if e.predicted_ms is not None]
        if not predicted:
            # ⚠️ 汇总原因必须来自**各候选的实际原因**。硬写"全部在范围外"会在
            # "没有加载模型"时给出错误解释 —— 同一个"没有预测"有两种完全不同的成因，
            # 报错文本把它们混成一个，事后就会照错的线索去查。
            reasons = sorted({e.reject_reason for e in evals if e.reject_reason})
            why = "；".join(reasons) if reasons else "没有候选可评估"
            if self.config.fallback_to_threshold:
                return self._fallback(why, evals)
            return CostDecision(False, f"不回退：{why}", candidates=evals,
                                fallback_reason=why)

        # **prefill 饥饿**：有 prefill 积压、而所有候选本步都推进不了 prefill。
        # 这不是"没动作"，而是"动作存在、但方向全错"——必须能数出来（否则它只会
        # 表现为"TTFT 莫名其妙地高"，而那不是任何一处报错）。
        backlog = bool(prefilling) or bool(waiting)
        starved = backlog and all(e.tokens_prefilled == 0 for e in predicted)
        if starved:
            self.state.n_prefill_starved += 1

        feasible = [e for e in predicted if e.feasible]
        if feasible:
            if memory_critical:
                # 显存吃紧 ⇒ 取**最保守**（最小 max_tokens → 最小 chunk），不看 prefill 进度
                chosen = sorted(feasible, key=lambda e: (e.max_tokens, e.chunk))[0]
                self.state.n_memory_conservative += 1
                tail = "；⚠️ prefill 饥饿" if starved else ""
                return CostDecision(True,
                                    f"显存吃紧 ⇒ 取最保守候选（bt={chosen.max_tokens}）{tail}",
                                    chosen.chunk, chosen.max_tokens, evals)
            chosen = sorted(feasible, key=lambda e: (-e.tokens_prefilled, e.predicted_ms + e.host_ms,
                                                     e.max_tokens, e.chunk))[0]
            tail = "；⚠️ prefill 饥饿" if starved else ""
            return CostDecision(True,
                                f"可行候选 {len(feasible)}/{len(predicted)}；"
                                f"取 prefill 进度最大者（{chosen.tokens_prefilled} tok）{tail}",
                                chosen.chunk, chosen.max_tokens, evals)
        # 有预测但全部超预算 ⇒ 取**最省**的那个，并显式记录降级
        chosen = sorted(predicted, key=lambda e: (e.predicted_ms + e.host_ms, e.max_tokens, e.chunk))[0]
        self.state.n_degraded += 1
        tail = "；⚠️ prefill 饥饿" if starved else ""
        # ⚠️ 打印的预算必须取**被选中候选自己用的那个**（双预算下它可能是 prefill 预算）
        # —— 硬写 `self._deadline()` 会在 prefill 步上报错口径，事后照它查会查到错的线索。
        _cb = chosen.budget_ms if chosen.budget_ms is not None else self._deadline()
        _cl = "prefill" if chosen.budget_kind == "prefill" else "每步"
        return CostDecision(True,
                            f"全部超{_cl}预算 ⇒ 降级取最省者"
                            f"（前向 {chosen.predicted_ms:.1f} + host {chosen.host_ms:.1f} = "
                            f"{chosen.predicted_ms + chosen.host_ms:.1f}ms > {_cb:.1f}ms）{tail}",
                            chosen.chunk, chosen.max_tokens, evals, degraded=True)

    # ------------------------------------------------------------------ 回退
    def _fallback(self, why: str, evals: list[CandidateEvaluation] | None = None) -> CostDecision:
        self.state.n_fallback += 1
        self.state.last_fallback_reason = why
        return CostDecision(False, f"回退到阈值控制器：{why}",
                            candidates=evals or [], fallback_reason=why)

    def host_overhead(self, batch: int) -> float:
        """该 batch 的 host 侧开销（ms）。表里没有就用线性拟合；都没有则 0（并如实标注）。

        ★ 公开名：判据口径的**组成部分**，任何"这条判据在当前 SLO 下够得着吗"的工具
        （如 `scripts/slo_reach_frontier.py`）都必须用**这一份**实现 ——
        另写一套"表命中否则 0"的简化版会与真实判据不一致（少掉线性拟合那一段）。
        """
        cfg = getattr(self, "config", None)
        tbl = (getattr(cfg, "host_overhead_ms", None) or {}) if cfg else {}
        if batch in tbl:
            return float(tbl[batch])
        fit = getattr(cfg, "host_overhead_fit", None) if cfg else None
        if fit:
            return max(0.0, fit[0] * batch + fit[1])
        return 0.0

    #: 旧名（私有）保留为别名：既有用例按旧名调用。
    _host_overhead = host_overhead

    def _deadline(self) -> float:
        """**decode/单一**预算（向后兼容名）。双预算启用时 prefill 步用 `_budget_for`。"""
        d = self.config.step_deadline_ms
        return float(d) if d is not None else float("inf")

    def _budget_for(self, *, tokens_prefilled: int,
                    decode_at_risk: bool = True) -> tuple[str, float]:
        """给定候选计划，返回 `(budget_kind, budget_ms)`。

        ★ **B1 §1.4 方案 A**：prefill 步与 decode 步用**不同的**承诺。
        规则（预先记录、无权重）：
          * `tokens_prefilled > 0` **且** `prefill_deadline_ms` 已配置(非 None)
            **且** 本步**没有**任何 decode 处于风险中（`decode_at_risk=False`）
            ⇒ `("prefill", prefill_deadline_ms)`（服务 TTFT）
          * 否则 ⇒ `("decode", step_deadline_ms)`（服务 TPOT；`None` ⇒ `inf`）

        ⚠️ `prefill_deadline_ms is None`（默认）时，**任何**成分都走 decode 预算
        ⇒ 与引入双预算之前**逐字节相同**（既有批次不变）。

        ★★ 为什么要 `decode_at_risk`（缺口 §5.1；2026-10-03 修，**已裁定**）
        -------------------------------------------------------------------
        原判据只看**候选自己**的 `tokens_prefilled > 0`。于是当本步还有 decode
        请求在等 token（或正在被服务）时，那些 decode 请求会被按 **prefill 预算**
        （宽松得多：2000 ms vs 100 ms）放行 ⇒ 实测 12 条 2048-token 等待局面下
        有 **6** 个候选被 prefill 预算放行、而按 decode 预算会不可行。

        缺口的方向是**放行过松**：它让"逐 token 护栏"在这些步上**完全不生效**，
        而产物里 `budget_kind="prefill"` 看上去完全正常 —— 这正是本项目最怕的形态
        （数字能跑通、口径悄悄失效）。

        ★ **两个护栏的语义（不要把两者混同，也不要改外部 SLO 值）**：

        | 护栏 | 保护什么 | 何时适用 |
        | --- | --- | --- |
        | **逐 token 护栏**（`step_deadline_ms`） | **每个 decode token 之间的间隔**（逐 token 上界） | 只要本步**服务**了 decode token，**或**本步让某个 decode 请求**没能**拿到 token（它的间隔被拉长） |
        | **prefill TTFT 预算**（`prefill_deadline_ms`） | 新起步请求的**首 token 时刻**（请求级） | 只在**没有**任何 decode 受影响时（纯 prefill 步） |

        ⚠️ **请求平均 TPOT** 与**逐 token 护栏**是两件事：前者是**请求级平均**，
        允许个别 token 慢、只要整段平均达标；后者是**每一个 token** 的上界。
        本函数实施的是后者（这也是 `step_deadline_ms` 一直以来的口径）；
        本修正**没有**引入任何"平均"口径的放宽 —— 恰恰相反，它堵住了
        "在某些步上逐 token 护栏根本没被应用"这个洞。

        ⚠️ `decode_at_risk` 的**默认值是 `True`（保守）**，不是 `False`：
        默认 `False` 会让任何忘记传参的调用点**静默回到旧缺口**；
        默认 `True` 最坏只是"该放宽时没放宽"（少给 token），方向安全。
        当前唯一的调用点（`decide`）**总是显式传值**，默认值只是最后一道保护。
        """
        cfg = self.config
        pf_d = getattr(cfg, "prefill_deadline_ms", None)
        if tokens_prefilled > 0 and pf_d is not None and not decode_at_risk:
            return "prefill", float(pf_d)
        return "decode", self._deadline()

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "prediction_contract": ("execution-shape-v1" if self.config.execution_signature
                                    is not None else "legacy_scalar_diagnostic"),
            "execution_signature": self.config.execution_signature,
            "candidate_chunks": list(self.config.candidate_chunks),
            "candidate_max_tokens": list(self.config.candidate_max_tokens),
            "step_deadline_ms": self.config.step_deadline_ms,
            "prefill_deadline_ms": getattr(self.config, "prefill_deadline_ms", None),
            "model": (self.model.describe() if self.model else None),
            "state": self.state.describe(),
        }
