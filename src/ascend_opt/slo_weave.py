"""SLOWeave 参考策略（第 13 步的**最接近方法对照**）。

来源与忠实程度（**照实说明，不冒称复现了完整系统**）
----------------------------------------------------
* **论文**：Siyu Song, Qi Bai, Jinbo Hao, Kai Li, Chenchen Wang, Jiayu Sun,
  *Deadline-Aware Adaptive Prefill Chunking for Efficient Large Language Model Serving*,
  arXiv:**2609.07883**（v1，2026-09-07 提交）。
* **作者制品**：摘要称 "The simulator, cost parameters, workload generator, raw runs,
  and aggregation code accompany the paper"，但**正文与 HTML 版均未给出 URL**
  （arXiv 的 Code/Data/Media 关联区亦未查到可下载地址）。
  ⇒ **没有使用作者代码**；本模块是**按论文 Algorithm 1 与 §3.3/§5.1 的公式自行实现**的
  **policy 参考版本**。
* ⚠️ **这不是复现完整系统**。相对于论文，本实现**缺少**以下部分，且**不声称**具备：
  1. 作者的事件驱动模拟器与 GPU runtime（本实现挂在**本项目自己的**模拟器/真机路径上）；
  2. §3.4 的两类生产控制（**准入控制**与 **aging 阈值**）—— 本实现只做 chunk 选择；
  3. §3.5 的**异构优先级与逐类差异化裕度**（本实现只有单一 TPOT 目标）；
  4. 作者的 P50/P99 网格 profiling 流程（本实现复用**本项目的**成本模型与其低估尾部裕度）；
  5. 多个 pending prefill 的背包扩展（论文提 first-fit；本实现只服务**最早**那一条，
     即论文 Algorithm 1 的单 prefill 形式）。

忠实实现的部分（逐条对应论文）
------------------------------
* **deadline**：`d_i = ℓ_i + D`（ℓ_i = 该请求最近一个 token 的完成时刻），`D` = TPOT 目标；
* **迭代预算**：`B_t = max(0, min_i (d_i − s_t))`（最早剩余松弛）；
* **选择**：`c* = max{ c ≤ min(C_max, L_t) : T(n, c) + δ(n, c) ≤ B_t }`。
  ⚠️ 论文用**二分**在 `c` 上搜索（依赖 `T` 对 `c` 单调 ⇒ 对数次查询），本实现**分两条路径**
  （2026-09-29 修，见 `decide()` 的 docstring）：
  * `segmented` 模型（`T` 单调、可行域连续）⇒ **二分**，与论文一致；
  * `lookup` 模型（**稀疏**：只有实测形状有键）⇒ **枚举**「有键的候选形状」取最大可行者
    —— 稀疏表上二分的前提不成立（实测会把可行预算误判成 `chunk = 0`）。
  ⚠️ **δ 只乘一次**：`_predict_ms` 取的是**不含**余量的 `T`（`apply_margin=False`），
  δ 由 `_margin_ms` 单独给出 —— 两处都乘就是"重复应用安全余量"（实测 1.1602 → 1.3461）。
* **无 decode 活跃**：`A_t = ∅` ⇒ 直接取 `min(C_max, L_t)`（加速 TTFT），`C_max = 4096`（论文 §4.3）；
* **没有正 chunk 能适配** ⇒ 执行一次 **decode-only** 迭代；
* **oldest-first**：只考虑最早那个 pending prefill（论文实现亦为 oldest-first，防重排序）；
* **δ 的来源**：论文建议"分位数回归或经验高分位残差表"⇒ 本实现取**本项目成本模型的
  低估误差尾部**（`safety_margin`，来自留出形状上的低估 p95），**相对形式**乘上去。

与本项目 `CostPolicy` 的关系（**重要**）
----------------------------------------
两者实质重合：`CostPolicy` 的选择规则（可行集取 `tokens_prefilled` 最大者、以每步预算为约束）
**就是** SLOWeave 的极大性目标，且两者的 deadline 映射相同。
⇒ 本模块的存在是为了**把"最接近方法"作为独立、可指认的对照臂**跑出来，
而不是为了宣称一个算法创新。**论文的优越性主张据此收窄**（见第 13 步记录）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .cost_model import CostModel

__all__ = ["SLOWeaveConfig", "SLOWeaveDecision", "SLOWeavePolicy"]


@dataclass(frozen=True)
class SLOWeaveConfig:
    #: `D`：TPOT 目标（ms）。论文用 10 / 25 / 50 ms 三档；本项目沿用已标定的 SLO 阈值。
    tpot_target_ms: float = 100.0
    #: `C_max`：无 decode 活跃时的最大 chunk（论文 §4.3 取 4096）。
    c_max: int = 4096
    enabled: bool = False


@dataclass(frozen=True)
class SLOWeaveDecision:
    chunk: int
    budget_ms: float
    earliest_deadline_ms: float | None
    n_active: int
    #: `True` = 没有正 chunk 能适配 ⇒ 本次应执行 **decode-only** 迭代。
    decode_only: bool
    margin_ms: float
    reason: str

    def describe(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk,
            # ⚠️ `inf` 会被标准化成 `None`（见 telemetry._json_safe）⇒ 用显式布尔
            # 字段保留"无活跃 decode ⇒ 预算无上限"这个信息，别让它只剩一个 `None`。
            "budget_ms": (None if self.budget_ms == float("inf")
                          else round(self.budget_ms, 4)),
            "budget_unbounded": self.budget_ms == float("inf"),
            "earliest_deadline_ms": (None if self.earliest_deadline_ms is None
                                     else round(self.earliest_deadline_ms, 3)),
            "n_active": self.n_active,
            "decode_only": self.decode_only,
            "margin_ms": round(self.margin_ms, 4),
            "reason": self.reason,
        }


class SLOWeavePolicy:
    """按论文 Algorithm 1 选 chunk。纯逻辑。"""

    def __init__(self, config: SLOWeaveConfig, model: CostModel | None = None) -> None:
        self.config = config
        self.model = model
        #: 回退计数（超出成本模型范围 / 无预测器时的 decode-only）。
        self.n_fallback = 0
        self.n_decode_only = 0
        self.n_no_prefill = 0

    # ---------------------------------------------------------------- 成本预测
    def _predict_ms(self, n_active: int, chunk: int, kv_len: int) -> float | None:
        """`T(n, c)`（**不含** δ）。**越界/缺键返回 `None`**（调用方按"不可行"处理，不外推）。

        ⚠️ **必须传 `apply_margin=False`**（2026-09-29 修）。
        `CostModel.predict` 的默认是 `True`，会在返回值里**先乘一次** `safety_margin`；
        而本模块的 δ（:meth:`_margin_ms`）**就是**那次低估尾部余量。
        不传 ⇒ 判据变成 `T·(1+m) + T·(1+m)·m = T·(1+m)²`
        —— 实测 `model_v7` 的 `m = 16.02%` 把 `1.1602` 放大成 **`1.3461`**
        （多乘一次 16%），即"**重复应用安全余量**"：判据比设计保守得多，
        且"余量买到的东西"与成本臂**不是同一个量** ⇒ 两臂不可比。
        """
        if self.model is None:
            return None
        p = self.model.predict(batch=n_active, prefill_tok=chunk, kv_len=kv_len,
                               apply_margin=False)
        return None if p is None else p.total_ms

    def _margin_ms(self, predicted_ms: float) -> float:
        """`δ(n, c)`：相对裕度 × 预测值（论文 §5.1 的"高分位残差"用经验表，本实现用相对形式）。"""
        if self.model is None:
            return 0.0
        return predicted_ms * float(self.model.payload.get("safety_margin") or 0.0)

    # ---------------------------------------------------------------- 决策
    def decide(self, *, now_ms: float, deadlines: Sequence[float],
               pending_prefill: Sequence[tuple[str, int]], kv_len: int,
               candidates: Sequence[int] | None = None) -> SLOWeaveDecision:
        """按 Algorithm 1 选 chunk。

        ``deadlines``
            活跃 decode 的 `d_i = ℓ_i + D`（**由调用方算**：需要"上一 token 完成时刻"，
            那是状态机的事，本模块不猜）。
        ``pending_prefill``
            **按 oldest-first 排序**的待办 prefill：`(req_id, remaining_tokens)`。
        ``candidates``
            **显式候选 chunk 集**。给定时按**枚举**取"最大的可行者"，
            **不**依赖 `T(n, c)` 的单调性/连续性。
            `None` ⇒ 自动判定：`lookup` 模型（稀疏表）⇒ 用 :meth:`CostModel.lookup_prefill_candidates`
            取**有键的形状**枚举；其余（`segmented`）⇒ 沿用论文的**二分**。

        ⚠️ 为什么默认不能一律二分（2026-09-29 实测反例）
        --------------------------------------------------
        二分需要 "`T` 单调 **且** 可行域连续"。稀疏 `lookup` 上缺键返回 `None`
        （= 未知，不是不可行），二分却按 `hi = mid - 1` 把整段更大的 `c` 一并剪掉
        ⇒ 实测 `b1/kv1024`：可预测 `c ∈ {128,256,512,1024}`，`mid=2048` 缺键后
        一路收缩到 **`chunk = 0`**，而 `c=1024`（316.9 ms）在预算 400 ms 下**本可行**。
        """
        if not pending_prefill:
            self.n_no_prefill += 1
            return SLOWeaveDecision(0, 0.0, None, len(deadlines), True, 0.0,
                                    "没有待办 prefill ⇒ 本步无事可做")
        # L ← min(C_max, remaining(最早的 pending prefill))
        remaining_earliest = int(pending_prefill[0][1])
        L = min(int(self.config.c_max), max(0, remaining_earliest))

        # A_t = ∅ ⇒ 没有 decode 截止时间约束 prefill ⇒ 用大 cap 加速 TTFT
        if not deadlines:
            return SLOWeaveDecision(L, float("inf"), None, 0, False, 0.0,
                                    f"无活跃 decode ⇒ 取 C_max 与待办量的较小者（{L}）")

        n_active = len(deadlines)
        earliest = min(deadlines)
        # B_t = max(0, min_i (d_i − s_t))
        budget = max(0.0, earliest - float(now_ms))
        if budget <= 0.0:
            self.n_decode_only += 1
            return SLOWeaveDecision(0, budget, earliest, n_active, True, 0.0,
                                    "最早 deadline 已过期 ⇒ decode-only")

        # ---- 离散候选路径：不假设单调/连续，逐个评估，取最大可行者 ----
        cand = candidates
        if cand is None and self.model is not None:
            cand = self.model.lookup_prefill_candidates(
                batch=n_active, kv_len=kv_len, cap=L)
        if cand is not None:
            best, best_margin = 0, 0.0
            for c in sorted({int(x) for x in cand}):
                if c <= 0 or c > L:
                    continue
                pred = self._predict_ms(n_active, c, kv_len)
                if pred is None:            # 缺键/越界 ⇒ **未知**，跳过（不是不可行）
                    continue
                m = self._margin_ms(pred)
                if pred + m <= budget:
                    best, best_margin = c, m
            if best <= 0:
                self.n_decode_only += 1
                return SLOWeaveDecision(0, budget, earliest, n_active, True, 0.0,
                                        f"候选集内没有 chunk 能在预算 {budget:.1f}ms 内完成"
                                        f"（候选 {len(cand)} 个）⇒ decode-only")
            return SLOWeaveDecision(best, budget, earliest, n_active, False, best_margin,
                                    f"最早松弛 {budget:.1f}ms ⇒ 取候选集中最大可行 chunk {best}")

        # ---- 连续路径（论文形式）：segmented 模型下 T 单调、可行域连续 ----
        # 二分：max{ c ≤ L : T(n, c) + δ ≤ B }。**单调性由成本模型保证**
        # （prefill 系数非负）；越界（None）一律当**不可行**，不外推。
        lo, hi = 0, L
        while lo < hi:
            mid = (lo + hi + 1) // 2
            pred = self._predict_ms(n_active, mid, kv_len)
            if pred is None:
                hi = mid - 1
                continue
            if pred + self._margin_ms(pred) <= budget:
                lo = mid
            else:
                hi = mid - 1

        if lo <= 0:
            self.n_decode_only += 1
            return SLOWeaveDecision(0, budget, earliest, n_active, True, 0.0,
                                    f"没有正 chunk 能在预算 {budget:.1f}ms 内完成 ⇒ decode-only")
        pred_lo = self._predict_ms(n_active, lo, kv_len) or 0.0
        return SLOWeaveDecision(lo, budget, earliest, n_active, False,
                                self._margin_ms(pred_lo),
                                f"最早松弛 {budget:.1f}ms ⇒ 取最大可行 chunk {lo}")

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "tpot_target_ms": self.config.tpot_target_ms,
            "c_max": self.config.c_max,
            "n_decode_only": self.n_decode_only,
            "n_no_prefill": self.n_no_prefill,
            "model": (self.model.describe() if self.model else None),
        }
