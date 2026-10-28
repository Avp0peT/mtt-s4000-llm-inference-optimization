#!/usr/bin/env python3
"""动态负载（``piecewise``）的**分阶段报告**。

为什么需要它
------------
研究问题的自变量是**负载水平的轨迹**（低压 → 高压 → 恢复，`piecewise` 的 λ(t)）。
可"整 run 一个汇总值"回答不了这个自变量带来的一切问题：

* **低压段**的 TTFT 与**高压段**的 TTFT 混进同一个中位数 ⇒ 那个数既不是低压的、
  也不是高压的，而是谁都没测过的；
* **恢复段**有没有真的回落、**多久**回落，整 run 平均值看不见；
* 队深与活跃数**在什么时候**涨、**什么时候**掉，只有按时间摊开才看得见；
* 高压段末尾到达的请求会在恢复段完成——它算哪一段的？

⇒ 本模块把报告拆成**两套口径**，并**规定它们不能互相替代**：

| 口径 | 归属依据 | 回答什么 |
| --- | --- | --- |
| **cohort**（按到达阶段） | 请求的**到达/放行时刻**落在哪一段 | "这一段进来的请求，后来表现如何" |
| **window**（按墙钟窗口） | 事件**发生**在哪个时间窗 | "这段时间里系统在处理什么、积压多少" |

⚠️ **跨阶段请求同时出现在两处**（它是 cohort A 的成员，也在 window B 里完成）——
这不是重复计数，是**两个问题的两个答案**。混成一个口径就会既漏又重。

三条预先定死的规则（验收要求"预先定义"，所以写在这里而不是事后解释）
----------------------------------------------------------------
1. **达标判定时机**：请求**完成时**才判定（成功 + 所有 SLO 字段可得）。
   未完成 / 失败 / 被拒**一律不算达标**——与 :meth:`SLO.satisfied_by` 一致。
2. **达标吞吐的分母**：统一用**测量分母**（run 级 = ``measured_window_ms``；
   分窗口 = 该窗口的墙钟长度）。⚠️ 不用"最后一个完成时刻"当分母，
   那会把排空尾段从分母里抹掉，吞吐虚高。
3. **恢复的判据**：见 :class:`RecoveryRule`，**阈值与持续时间都必须显式给定**。
   没观察到恢复 ⇒ 记 ``not_recovered`` / ``censored``，**绝不填 0** ——
   "0 ms 恢复"是一个好得不可能的读数，比"未恢复"更容易误导。

低样本
------
每段/每窗都带 ``n``。低于 ``min_n_for_percentiles`` 时**不给 P95/P99**
（返回 ``None`` 并标 ``insufficient_samples``）：十几个样本的 P99 本质上就是最大值，
报出来会诱导人相信一个没有统计意义的数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ascend_opt.metrics import OUTCOME_FAILED, OUTCOME_REJECTED, OUTCOME_SUCCESS, OUTCOME_UNFINISHED, percentile

__all__ = [
    "RecoveryRule",
    "StageSpec",
    "StageReport",
    "build_stage_specs",
    "build_stage_report",
    "DRAIN_LABEL",
]

#: "排空尾段"的标签。它**不是**负载的一段，而是最后一批到达**完成**所花的时间。
#: 单独列出来的理由：把它并进最后一段（recover）会同时污染两件事——
#: 恢复段的达标率被排空的慢请求拉低，而排空时长被算成"恢复耗时"。
DRAIN_LABEL = "drain"


def _opt(value: Any) -> Any:
    """缺失一律 ``None``。**不做 0 填充**：0 是"测出来是 0"，缺失是"没测"。"""
    return value


def _pct(values: Sequence[float], p: float, *, min_n: int) -> float | None:
    """分位数；样本不足时返回 ``None``。

    样本量必须与分位一起看：给十几个样本算 P99，本质上是把最大值换个名字。
    返回 ``None`` 会逼调用方去补样本，返回一个数则不会。
    """
    if len(values) < max(1, min_n):
        return None
    return percentile(list(values), p)


def _median(values: Sequence[float], *, min_n: int) -> float | None:
    return _pct(values, 50, min_n=min_n)


# --------------------------------------------------------------------------- #
# 输入
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StageSpec:
    """负载轨迹里的一段（来自 ``workload_fingerprint.arrival_segments``）。

    ``realized_count`` 是**实际**落到设备上的请求数（与 ``nominal_count`` 可能不同，
    ``poisson`` 噪声下尤其明显）。报告里两个都给：名义值是设计意图，实际值才是负载。
    """

    label: str
    start_ms: float
    end_ms: float
    nominal_qps: float | None = None
    nominal_count: float | None = None
    realized_count: int | None = None

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


def build_stage_specs(arrival_segments: Iterable[Mapping[str, Any]]) -> list[StageSpec]:
    """把 ``arrival_segments`` 转成 :class:`StageSpec`（保持原顺序）。"""
    out: list[StageSpec] = []
    for seg in arrival_segments:
        out.append(
            StageSpec(
                label=str(seg.get("label", "")),
                start_ms=float(seg["start_ms"]),
                end_ms=float(seg["end_ms"]),
                nominal_qps=_opt(seg.get("nominal_qps")),
                nominal_count=_opt(seg.get("nominal_count")),
                realized_count=_opt(seg.get("realized_count")),
            )
        )
    return out


@dataclass(frozen=True)
class RecoveryRule:
    """恢复的判据，**必须显式给定**（阈值 + 持续时间）。

    "恢复了"是一个**有参数**的判断。参数不写死就会在事后按结果挑一个，
    于是"恢复了"永远成立。

    * ``metric``：用哪个量判恢复。默认 ``queue_depth``（最直接反映积压）。
    * ``threshold``：低于此值算"回到低压水平"。
    * ``hold_ms``：**必须连续保持**这么久才算恢复。只取瞬时值会把一次抖动读成恢复。
    * ``from_ms``：从哪一刻开始找（通常是高压段结束时刻）。
    * ``until_ms``：观察窗的右端；到这儿还没恢复 ⇒ **删失（censored）**，
      而不是"没恢复"——两者含义不同（前者是"还没来得及看"，后者是"看完了也没回落"）。
    """

    metric: str = "queue_depth"
    threshold: float = 1.0
    hold_ms: float = 1000.0
    from_ms: float | None = None
    until_ms: float | None = None


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #


@dataclass
class CohortReport:
    """**按到达阶段**看一批请求（cohort）。归属依据 = 请求的到达/放行时刻。"""

    label: str
    n_submitted: int = 0
    n_success: int = 0
    n_failed: int = 0
    n_rejected: int = 0
    n_unfinished: int = 0
    #: 到达在本段、**完成在别的段**的请求数。它不是异常，是必然（高压段末尾的请求
    #: 会在恢复段完成）；单列出来是为了让"cohort 与 window 不是同一口径"可见。
    n_cross_stage: int = 0
    completion_by_stage: dict[str, int] = field(default_factory=dict)
    n_slo_satisfied: int = 0
    ttft_ms_p50: float | None = None
    ttft_ms_p95: float | None = None
    ttft_ms_p99: float | None = None
    tpot_ms_p50: float | None = None
    tpot_ms_p95: float | None = None
    stall_ms_p99: float | None = None
    #: 达标吞吐按**完成窗口**归属；cohort 上给的是"这批请求贡献的总量"，
    #: 分母仍是统一测量分母（见模块 docstring 规则 2）。
    goodput_tok: int | None = None
    insufficient_samples: bool = False

    def describe(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class WindowReport:
    """**按墙钟窗口**看发生了什么（事件按发生时刻归属）。"""

    label: str
    start_ms: float
    end_ms: float
    duration_ms: float = 0.0
    #: 到达（受理）与完成的**事件计数**——与 cohort 的"提交数"是两个不同的量。
    n_admitted: int = 0
    n_completed: int = 0
    n_failed: int = 0
    n_rejected: int = 0
    tokens_out: int = 0
    n_slo_satisfied: int = 0
    #: 达标吞吐：分子按**完成窗口**归属，分母 = 本窗口墙钟长度（统一分母）。
    slo_goodput_tok_s: float | None = None
    slo_goodput_req_s: float | None = None
    slo_satisfaction_rate: float | None = None
    queue_depth_max: int | None = None
    queue_depth_p50: float | None = None
    active_max: int | None = None
    #: 本窗口内发生的控制动作（来自 ``controller_trace``）。
    n_control_actions: int = 0
    chunk_size_min: int | None = None
    chunk_size_max: int | None = None

    def describe(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class RecoveryReport:
    """恢复时间。**未观察到就是未观察到**，不填 0。"""

    status: str = "not_applicable"      # recovered | not_recovered | censored | not_applicable
    at_ms: float | None = None
    duration_ms: float | None = None
    rule: str = ""
    threshold: float | None = None
    hold_ms: float | None = None

    def describe(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class StageReport:
    stages: list[StageSpec] = field(default_factory=list)
    cohorts: list[CohortReport] = field(default_factory=list)
    windows: list[WindowReport] = field(default_factory=list)
    recovery: RecoveryReport = field(default_factory=RecoveryReport)
    #: 对账结果。``ok=False`` 时**必须**把差异列出来——不静默。
    reconciliation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "stages": [s.__dict__ for s in self.stages],
            "cohorts": [c.describe() for c in self.cohorts],
            "windows": [w.describe() for w in self.windows],
            "recovery": self.recovery.describe(),
            "reconciliation": self.reconciliation,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def _stage_of(t_ms: float | None, stages: Sequence[StageSpec],
              tail: StageSpec | None) -> str | None:
    """某个时刻落在哪一段。超出最后一段 ⇒ 归入"排空尾段"。

    ⚠️ 用**半开区间** ``[start, end)``：边界上的请求只能属于一段，否则同一请求
    会被两段各数一次（重复计数）。最后一段的右端按**闭区间**处理
    （``end_ms`` 上的到达属于它，不属于排空段）——因为排空段是"最后一批**完成**"的
    时间，不是"还有请求到达"的时间。
    """
    if t_ms is None:
        return None
    for i, s in enumerate(stages):
        last = i == len(stages) - 1
        if s.start_ms <= t_ms < s.end_ms or (last and t_ms == s.end_ms):
            return s.label
    if tail is not None and t_ms >= tail.start_ms:
        return tail.label
    return None


def build_stage_report(
    *,
    requests: Sequence[Mapping[str, Any]],
    stages: Sequence[StageSpec],
    telemetry: Sequence[Mapping[str, Any]] = (),
    actions: Sequence[Mapping[str, Any]] = (),
    measured_window_ms: float | None = None,
    recovery_rule: RecoveryRule | None = None,
    min_n_for_percentiles: int = 20,
    counts_toward_drain: bool = True,
) -> StageReport:
    """生成分阶段报告。

    ``requests`` 每项至少要有的键（缺了按 ``None`` 处理，**不填 0**）：
    ``arrival_ms`` / ``done_ms`` / ``outcome`` / ``ttft_ms`` / ``tpot_ms`` /
    ``tpot_p99_ms`` / ``gen_len`` / ``slo_satisfied``。

    ``telemetry`` 至少要有 ``t_ms`` / ``queue_depth`` / ``running``；
    可选 ``chunk_size``。
    """
    rep = StageReport(stages=list(stages))
    n = len(requests)

    # ---- 排空尾段：最后一批到达**完成**所花的时间 ----------------------------
    # 它不是负载的一段。只有在真的存在"最后一段结束之后才完成"的请求时才建立，
    # 否则会凭空造出一个空窗口，让人以为负载比实际长。
    drain_start = stages[-1].end_ms if stages else None
    done_times = [r.get("done_ms") for r in requests if r.get("done_ms") is not None]
    tail: StageSpec | None = None
    if drain_start is not None and done_times and max(done_times) > drain_start:
        tail = StageSpec(label=DRAIN_LABEL, start_ms=drain_start,
                         end_ms=max(done_times), realized_count=None)
    all_stages = list(stages) + ([tail] if tail else [])

    # ---- cohort：按**到达时刻**归属 ----------------------------------------
    cohorts: dict[str, CohortReport] = {s.label: CohortReport(label=s.label)
                                        for s in all_stages}
    unassigned: list[Any] = []
    for r in requests:
        label = _stage_of(r.get("arrival_ms"), stages, None)   # ⚠️ 到达只归到负载段
        if label is None:
            unassigned.append(r)
            continue
        c = cohorts.setdefault(label, CohortReport(label=label))
        c.n_submitted += 1
        outcome = r.get("outcome")
        if outcome == OUTCOME_SUCCESS:
            c.n_success += 1
        elif outcome == OUTCOME_FAILED:
            c.n_failed += 1
        elif outcome == OUTCOME_REJECTED:
            c.n_rejected += 1
        else:
            c.n_unfinished += 1
        # ⚠️ 达标必须是"成功 **且** 自报达标"。
        # 只信自报字段会把失败/被拒/未完成的请求算成达标——而它们**不可能**达标
        # （与 :meth:`SLO.satisfied_by` 一致）。产物里那个字段若因任何原因写错，
        # 这里就是最后一道闸；"没跑完却达标"会把达标率整体抬高。
        satisfied = (
            r.get("slo_satisfied") is True and outcome == OUTCOME_SUCCESS
        )
        if satisfied:
            c.n_slo_satisfied += 1
        done_label = _stage_of(r.get("done_ms"), stages, tail)
        if done_label is not None:
            c.completion_by_stage[done_label] = c.completion_by_stage.get(done_label, 0) + 1
            if done_label != label:
                c.n_cross_stage += 1

    # cohort 的分位数只统计**成功**请求（未完成的没有时延可言，混进来会把分布拉偏）
    for c in cohorts.values():
        rows = [r for r in requests
                if _stage_of(r.get("arrival_ms"), stages, None) == c.label
                and r.get("outcome") == OUTCOME_SUCCESS]
        ttfts = [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
        tpots = [r["tpot_ms"] for r in rows if r.get("tpot_ms") is not None]
        stalls = [r["tpot_p99_ms"] for r in rows if r.get("tpot_p99_ms") is not None]
        c.insufficient_samples = len(rows) < min_n_for_percentiles
        c.ttft_ms_p50 = _median(ttfts, min_n=1)
        c.ttft_ms_p95 = _pct(ttfts, 95, min_n=min_n_for_percentiles)
        c.ttft_ms_p99 = _pct(ttfts, 99, min_n=min_n_for_percentiles)
        c.tpot_ms_p50 = _median(tpots, min_n=1)
        c.tpot_ms_p95 = _pct(tpots, 95, min_n=min_n_for_percentiles)
        c.stall_ms_p99 = _pct(stalls, 99, min_n=min_n_for_percentiles)
        c.goodput_tok = (
            sum(int(r.get("gen_len") or 0) for r in rows if r.get("slo_satisfied") is True)
            if rows
            else None
        )

    # ⚠️ 把局部字典**挂回报告**（按段序，保证顺序稳定）。
    # 这一步曾经漏掉：cohorts 在函数里建好了、却没人读得到 ⇒ 报告里 cohort 为空。
    # 是测试（`KeyError: 'low'`）把它抓出来的——"算出来了但没接上去"正是本项目最警惕的那类。
    rep.cohorts = [cohorts[s.label] for s in all_stages if s.label in cohorts]

    # ---- window：按**事件发生时刻**归属 ------------------------------------
    tel_by_label: dict[str, list[Mapping[str, Any]]] = {s.label: [] for s in all_stages}
    for t in telemetry:
        label = _stage_of(t.get("t_ms"), stages, tail)
        if label is not None:
            tel_by_label.setdefault(label, []).append(t)

    act_by_label: dict[str, int] = {}
    for a in actions:
        label = _stage_of(a.get("t_ms"), stages, tail)
        if label is not None:
            act_by_label[label] = act_by_label.get(label, 0) + 1

    for s in all_stages:
        w = WindowReport(label=s.label, start_ms=s.start_ms, end_ms=s.end_ms,
                         duration_ms=s.duration_ms)
        # 事件：受理按 arrival_ms、完成按 done_ms、拒绝按 done_ms（被拒也是一次"发生"）
        w.n_admitted = sum(1 for r in requests
                           if _stage_of(r.get("arrival_ms"), stages, tail) == s.label)
        done_here = [r for r in requests
                     if _stage_of(r.get("done_ms"), stages, tail) == s.label]
        w.n_completed = sum(1 for r in done_here if r.get("outcome") == OUTCOME_SUCCESS)
        w.n_failed = sum(1 for r in done_here if r.get("outcome") == OUTCOME_FAILED)
        w.n_rejected = sum(1 for r in done_here if r.get("outcome") == OUTCOME_REJECTED)
        w.n_slo_satisfied = sum(
            1 for r in done_here
            if r.get("slo_satisfied") is True and r.get("outcome") == OUTCOME_SUCCESS
        )
        w.tokens_out = sum(int(r.get("gen_len") or 0) for r in done_here
                           if r.get("outcome") == OUTCOME_SUCCESS)

        # 分母：run 级用 measured_window_ms；分窗口用**窗口墙钟长度**。
        denom_s = (measured_window_ms if (s.label == DRAIN_LABEL and measured_window_ms)
                   else s.duration_ms) / 1000.0
        if denom_s > 0:
            tok = sum(int(r.get("gen_len") or 0) for r in done_here
                      if r.get("slo_satisfied") is True
                      and r.get("outcome") == OUTCOME_SUCCESS)
            w.slo_goodput_tok_s = round(tok / denom_s, 6)
            w.slo_goodput_req_s = round(w.n_slo_satisfied / denom_s, 6)
            # 达标率的分母是**全部提交**（不是成功数）——否则失败与未完成会被抹掉
            submitted = w.n_admitted
            w.slo_satisfaction_rate = (
                round(w.n_slo_satisfied / submitted, 6) if submitted else None
            )

        rows_t = tel_by_label.get(s.label, [])
        qd = [t["queue_depth"] for t in rows_t if t.get("queue_depth") is not None]
        run = [t["running"] for t in rows_t if t.get("running") is not None]
        ch = [t["chunk_size"] for t in rows_t if t.get("chunk_size") is not None]
        w.queue_depth_max = max(qd) if qd else None
        w.queue_depth_p50 = round(percentile(qd, 50), 3) if qd else None
        w.active_max = max(run) if run else None
        w.chunk_size_min = min(ch) if ch else None
        w.chunk_size_max = max(ch) if ch else None
        w.n_control_actions = act_by_label.get(s.label, 0)
        rep.windows.append(w)

    # ---- 恢复 -------------------------------------------------------------
    rep.recovery = _recovery(telemetry, stages, tail, recovery_rule)

    # ---- 对账 -------------------------------------------------------------
    rep.reconciliation = _reconcile(requests, cohorts, rep.windows, n)
    if unassigned:
        rep.notes.append(
            f"⚠️ 有 {len(unassigned)} 个请求的到达时刻落在任何一段之外"
            f"（共 {n} 个），它们**没有被计入任何 cohort**：这是对账不通过的一种，"
            "不要当成'这一段没负载'。"
        )
    if not counts_toward_drain:
        rep.notes.append("⚠️ 已关闭排空尾段计数：请确认这是有意的。")
    return rep


def _recovery(
    telemetry: Sequence[Mapping[str, Any]],
    stages: Sequence[StageSpec],
    tail: StageSpec | None,
    rule: RecoveryRule | None,
) -> RecoveryReport:
    """按**预先给定**的规则判断恢复。没观察到就说没观察到。"""
    if rule is None or not stages:
        return RecoveryReport(status="not_applicable",
                              rule="未给定 RecoveryRule ⇒ 不判恢复")
    start = rule.from_ms if rule.from_ms is not None else stages[-1].start_ms
    until = rule.until_ms
    if until is None:
        until = max((t.get("t_ms") or 0.0) for t in telemetry) if telemetry else start

    pts = sorted(
        (float(t["t_ms"]), float(t[rule.metric]))
        for t in telemetry
        if t.get("t_ms") is not None and t.get(rule.metric) is not None
        and t["t_ms"] >= start
    )
    desc = (f"{rule.metric} < {rule.threshold} 且连续保持 {rule.hold_ms} ms，"
            f"自 {start} ms 起观察")
    if not pts:
        return RecoveryReport(status="censored", rule=desc,
                              threshold=rule.threshold, hold_ms=rule.hold_ms)
    hold_start: float | None = None
    for t_ms, value in pts:
        if value < rule.threshold:
            if hold_start is None:
                hold_start = t_ms
            if t_ms - hold_start >= rule.hold_ms:
                return RecoveryReport(
                    status="recovered", at_ms=t_ms,
                    duration_ms=t_ms - start, rule=desc,
                    threshold=rule.threshold, hold_ms=rule.hold_ms,
                )
        else:
            # 中间抬起来一次 ⇒ 连续段作废，从头再数。
            # 只取瞬时值会把一次抖动读成恢复。
            hold_start = None
    observed_end = max(t for t, _ in pts)
    status = "censored" if observed_end < until else "not_recovered"
    return RecoveryReport(status=status, rule=desc,
                          threshold=rule.threshold, hold_ms=rule.hold_ms)


def _reconcile(
    requests: Sequence[Mapping[str, Any]],
    cohorts: Mapping[str, CohortReport],
    windows: Sequence[WindowReport],
    n_total: int,
) -> dict[str, Any]:
    """对账：**每一条都要能对上**，对不上就把差异列出来（不静默）。"""
    by_cohort = sum(c.n_submitted for c in cohorts.values())
    by_outcome = sum(c.n_success + c.n_failed + c.n_rejected + c.n_unfinished
                     for c in cohorts.values())
    n_success_total = sum(1 for r in requests if r.get("outcome") == OUTCOME_SUCCESS)
    n_done_total = sum(1 for r in requests if r.get("done_ms") is not None)
    completed_in_windows = sum(w.n_completed for w in windows)
    problems: list[str] = []
    if by_cohort != n_total:
        problems.append(
            f"cohort 提交数之和 {by_cohort} ≠ 请求总数 {n_total}"
            f"（差 {n_total - by_cohort}）——有请求没被任何一段认领"
        )
    if by_outcome != by_cohort:
        problems.append(
            f"结局分类之和 {by_outcome} ≠ cohort 提交数 {by_cohort}"
            f"——有请求的 outcome 不在四态之内"
        )
    if completed_in_windows != n_success_total:
        problems.append(
            f"窗口内完成数之和 {completed_in_windows} ≠ 成功总数 {n_success_total}"
            f"——有完成事件落在所有窗口之外（排空尾段没建？）"
        )
    return {
        "ok": not problems,
        "n_requests": n_total,
        "n_cohort_submitted": by_cohort,
        "n_by_outcome": by_outcome,
        "n_success": n_success_total,
        "n_with_done_ms": n_done_total,
        "n_completed_in_windows": completed_in_windows,
        "problems": problems,
    }
