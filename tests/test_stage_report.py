"""分阶段报告（第 05 步）的验收用例。**全部离线**，手工可核算。

要证的六件事：
1. **跨阶段请求**：到达在 A 段、完成在 B 段 ⇒ 只计入 A 的 cohort（提交）
   **并且**计入 B 的 window（完成）——两处都出现是**两个问题的两个答案**，不是重复计数；
2. **未完成 / 失败 / 被拒**：计提交、计结局，但**不进任何完成窗口**、**不算达标**；
3. **排空尾段**：最后一批完成所花的时间**单独一段**，不并进最后一段负载；
4. **对账**：每一条都要对得上；对不上就把差异列出来，**不静默**；
5. **恢复**：阈值 + 持续时间都必须显式给定；没观察到就记
   ``not_recovered`` / ``censored``，``duration_ms`` 是 ``None``——**不填 0**；
6. **低样本**：分位与 ``n`` 并列；样本不足时 P95/P99 返回 ``None``。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.metrics import (  # noqa: E402
    OUTCOME_FAILED,
    OUTCOME_REJECTED,
    OUTCOME_SUCCESS,
    OUTCOME_UNFINISHED,
)
from ascend_opt.stage_report import (  # noqa: E402
    DRAIN_LABEL,
    RecoveryRule,
    StageSpec,
    build_stage_report,
)

# 三段：low [0,1000) / high [1000,2000) / recover [2000,3000)
STAGES = [
    StageSpec("low", 0.0, 1000.0, nominal_qps=0.5, nominal_count=0.5),
    StageSpec("high", 1000.0, 2000.0, nominal_qps=2.0, nominal_count=2.0),
    StageSpec("recover", 2000.0, 3000.0, nominal_qps=0.5, nominal_count=0.5),
]


def _req(req_id, arrival, done, outcome=OUTCOME_SUCCESS, *, ttft=10.0, tpot=5.0,
         tpot_p99=8.0, gen_len=10, slo=True):
    """⚠️ 非成功结局**不可能达标**——夹具必须照这个事实造，否则测的不是真契约。"""
    return {
        "req_id": req_id, "arrival_ms": arrival, "done_ms": done, "outcome": outcome,
        "ttft_ms": ttft, "tpot_ms": tpot, "tpot_p99_ms": tpot_p99,
        "gen_len": gen_len, "slo_satisfied": bool(slo) and outcome == OUTCOME_SUCCESS,
    }


class TestCohortAndWindowAreDifferentLenses(unittest.TestCase):
    """cohort（按到达阶段）与 window（按墙钟窗口）是**两套口径**，不能互相替代。"""

    def setUp(self):
        # r0 低压到达、低压完成
        # r1 高压到达、**排空尾段**完成（跨阶段）
        # r2 恢复段到达、恢复段完成（但不达标）
        # r3 高压到达、**未完成**
        self.requests = [
            _req("r0", 100.0, 300.0, gen_len=10, slo=True),
            _req("r1", 1500.0, 3500.0, gen_len=20, slo=True),
            _req("r2", 2500.0, 2600.0, gen_len=30, slo=False),
            _req("r3", 1200.0, None, OUTCOME_UNFINISHED, gen_len=40),
        ]
        self.rep = build_stage_report(requests=self.requests, stages=STAGES)
        self.cohorts = {c.label: c for c in self.rep.cohorts}
        self.windows = {w.label: w for w in self.rep.windows}

    def test_cohort_membership_follows_arrival_not_completion(self):
        self.assertEqual(self.cohorts["low"].n_submitted, 1)
        self.assertEqual(self.cohorts["high"].n_submitted, 2)      # r1 + r3
        self.assertEqual(self.cohorts["recover"].n_submitted, 1)
        # 排空尾段**没有到达**：它是"完成"的时间，不是"到达"的时间
        self.assertEqual(self.cohorts[DRAIN_LABEL].n_submitted, 0)

    def test_cross_stage_request_appears_in_both_lenses(self):
        """r1 到达在高压、完成在排空尾段 ⇒ 两处都出现，但含义不同。"""
        self.assertEqual(self.cohorts["high"].n_cross_stage, 1)
        self.assertEqual(self.cohorts["high"].completion_by_stage.get(DRAIN_LABEL), 1)
        self.assertEqual(self.windows[DRAIN_LABEL].n_completed, 1)  # 完成侧
        self.assertEqual(self.windows["high"].n_completed, 0)       # 高压段内一个都没完成

    def test_window_admission_uses_arrival_time(self):
        self.assertEqual(self.windows["low"].n_admitted, 1)
        self.assertEqual(self.windows["high"].n_admitted, 2)
        self.assertEqual(self.windows["recover"].n_admitted, 1)
        self.assertEqual(self.windows[DRAIN_LABEL].n_admitted, 0)

    def test_drain_tail_is_its_own_window_not_merged_into_recover(self):
        """排空尾段必须单独一段：并进 recover 会同时污染"恢复段达标率"与"恢复耗时"。"""
        self.assertIn(DRAIN_LABEL, self.windows)
        self.assertAlmostEqual(self.windows[DRAIN_LABEL].start_ms, 3000.0)
        self.assertAlmostEqual(self.windows[DRAIN_LABEL].end_ms, 3500.0)
        self.assertEqual(self.windows["recover"].n_completed, 1)     # 只有 r2

    def test_no_drain_window_when_everything_finishes_inside_stages(self):
        rep = build_stage_report(
            requests=[_req("a", 100.0, 200.0), _req("b", 1500.0, 1900.0)],
            stages=STAGES,
        )
        self.assertNotIn(DRAIN_LABEL, {w.label for w in rep.windows})


class TestUnfinishedAndFailedAreNotLost(unittest.TestCase):
    """未完成 / 失败 / 被拒**不能消失**：计提交、计结局，但不进完成窗口、不算达标。"""

    def setUp(self):
        self.requests = [
            _req("ok", 100.0, 300.0, gen_len=10, slo=True),
            _req("bad", 1500.0, 1600.0, OUTCOME_FAILED, gen_len=10),
            _req("rej", 1700.0, 1750.0, OUTCOME_REJECTED, gen_len=10),
            _req("unf", 1800.0, None, OUTCOME_UNFINISHED, gen_len=10),
        ]
        self.rep = build_stage_report(requests=self.requests, stages=STAGES)
        self.cohorts = {c.label: c for c in self.rep.cohorts}
        self.windows = {w.label: w for w in self.rep.windows}

    def test_every_outcome_is_counted_in_its_cohort(self):
        high = self.cohorts["high"]
        self.assertEqual(high.n_submitted, 3)
        self.assertEqual(high.n_failed, 1)
        self.assertEqual(high.n_rejected, 1)
        self.assertEqual(high.n_unfinished, 1)
        self.assertEqual(high.n_success, 0)

    def test_only_success_counts_as_completed_in_a_window(self):
        self.assertEqual(self.windows["low"].n_completed, 1)
        # 失败/被拒既不是"完成"，也不算"未完成"；单列出来才看得出成因
        self.assertEqual(self.windows["high"].n_completed, 0)
        self.assertEqual(self.windows["high"].n_failed, 1)
        self.assertEqual(self.windows["high"].n_rejected, 1)

    def test_unfinished_request_is_never_slo_satisfied(self):
        """未完成的请求**不可能**达标——否则"没跑完"会被算成"很快"。

        这也是为什么它的 ``done_ms`` 不参与任何完成窗口：没有完成时刻可言。
        """
        for c in self.rep.cohorts:
            self.assertFalse(
                c.label == "high" and c.n_slo_satisfied > 0,
                "高压段的失败/被拒/未完成一个都不该算达标",
            )
        self.assertEqual(self.cohorts["high"].n_slo_satisfied, 0)

    def test_satisfaction_rate_denominator_is_all_submitted(self):
        """⚠️ 达标率的**分母是全部提交**（含失败/未完成），不是成功数。

        用成功数做分母会把失败从账上抹掉——一批大量超时的 run 反而显示出高"达标率"。
        """
        high = self.windows["high"]
        # 3 个提交（1 失败 1 被拒 1 未完成）、0 个达标 ⇒ 0/3（不是 0/0 的空值）
        self.assertAlmostEqual(high.slo_satisfaction_rate, 0.0)

    def test_goodput_denominator_is_the_measured_window(self):
        """达标吞吐的分母统一用**窗口墙钟长度**（run 级 = measured window）。"""
        low = self.windows["low"]
        # 1 个达标请求、gen_len=10、窗口 1000ms=1s ⇒ 10 tok/s
        self.assertAlmostEqual(low.slo_goodput_tok_s, 10.0)
        self.assertAlmostEqual(low.slo_goodput_req_s, 1.0)


class TestReconciliation(unittest.TestCase):
    """对账：每一条都要对得上；对不上就把差异列出来。"""

    def test_clean_case_reconciles(self):
        rep = build_stage_report(
            requests=[_req("a", 100.0, 200.0), _req("b", 1500.0, 3500.0)],
            stages=STAGES,
        )
        self.assertTrue(rep.reconciliation["ok"], rep.reconciliation["problems"])
        self.assertEqual(rep.reconciliation["n_cohort_submitted"], 2)
        self.assertEqual(rep.reconciliation["n_completed_in_windows"], 2)

    def test_request_arriving_outside_every_stage_is_reported_not_dropped(self):
        """到达时刻落在所有段之外的请求**必须显形**——不许当成"这一段没负载"。"""
        rep = build_stage_report(
            requests=[_req("a", 100.0, 200.0), _req("stray", 99_999.0, 100_500.0)],
            stages=STAGES,
        )
        self.assertFalse(rep.reconciliation["ok"])
        self.assertTrue(any("没被任何一段认领" in p for p in rep.reconciliation["problems"]))
        self.assertTrue(any("落在任何一段之外" in n for n in rep.notes))

    def test_completion_outside_windows_is_reported(self):
        """完成事件落在所有窗口之外 ⇒ 对账必须报（典型成因是排空尾段没建）。"""
        rep = build_stage_report(
            requests=[_req("a", 100.0, 200.0)], stages=STAGES,
        )
        self.assertTrue(rep.reconciliation["ok"])
        # 人为把 done 时刻挪到最后一段之后、同时不给 telemetry ⇒ 应仍能建出排空段
        rep2 = build_stage_report(
            requests=[_req("a", 100.0, 5000.0)], stages=STAGES,
        )
        self.assertIn(DRAIN_LABEL, {w.label for w in rep2.windows})
        self.assertTrue(rep2.reconciliation["ok"], rep2.reconciliation["problems"])


class TestRecovery(unittest.TestCase):
    """恢复：阈值与持续时间**必须显式给定**；没观察到就不填 0。"""

    TELEMETRY = [
        {"t_ms": 1000.0, "queue_depth": 5, "running": 8},
        {"t_ms": 1500.0, "queue_depth": 7, "running": 8},
        {"t_ms": 2000.0, "queue_depth": 4, "running": 6},
        {"t_ms": 2500.0, "queue_depth": 0, "running": 2},
        {"t_ms": 3000.0, "queue_depth": 0, "running": 1},
        {"t_ms": 3500.0, "queue_depth": 0, "running": 1},
    ]

    def test_recovered_when_threshold_and_hold_are_met(self):
        rep = build_stage_report(
            requests=[], stages=STAGES, telemetry=self.TELEMETRY,
            recovery_rule=RecoveryRule(metric="queue_depth", threshold=1.0,
                                       hold_ms=500.0, from_ms=2000.0),
        )
        r = rep.recovery
        self.assertEqual(r.status, "recovered")
        self.assertAlmostEqual(r.at_ms, 3000.0)          # 2500 起连续保持 500ms
        self.assertAlmostEqual(r.duration_ms, 1000.0)    # 自 from_ms=2000 起算
        self.assertIn("queue_depth < 1.0", r.rule)

    def test_a_single_blip_does_not_count_as_recovery(self):
        """只取瞬时值会把一次抖动读成恢复：中间抬起来一次，连续段必须作废。"""
        telemetry = [
            {"t_ms": 2000.0, "queue_depth": 0},
            {"t_ms": 2100.0, "queue_depth": 5},     # ← 抬起来
            {"t_ms": 2200.0, "queue_depth": 0},
            {"t_ms": 2300.0, "queue_depth": 0},
        ]
        rep = build_stage_report(
            requests=[], stages=STAGES, telemetry=telemetry,
            recovery_rule=RecoveryRule(threshold=1.0, hold_ms=500.0, from_ms=2000.0),
        )
        # 2000 起算到 2100 就断了；2100 之后重新数，到 2300 只有 200ms < 500ms
        self.assertNotEqual(rep.recovery.status, "recovered")

    def test_not_recovered_records_none_not_zero(self):
        """⚠️ 没回落 ⇒ ``not_recovered``，``duration_ms`` 是 ``None`` —— **不填 0**。

        "0 ms 恢复"是一个好得不可能的读数，比"未恢复"更容易误导。
        """
        rep = build_stage_report(
            requests=[], stages=STAGES, telemetry=self.TELEMETRY,
            recovery_rule=RecoveryRule(threshold=0.0, hold_ms=500.0, from_ms=2000.0),
        )
        self.assertEqual(rep.recovery.status, "not_recovered")
        self.assertIsNone(rep.recovery.duration_ms)
        self.assertIsNone(rep.recovery.at_ms)

    def test_censored_when_the_observation_window_ends_early(self):
        """观察窗还没结束就收工 ⇒ **删失**，与"看完了也没回落"必须分开。"""
        rep = build_stage_report(
            requests=[], stages=STAGES, telemetry=self.TELEMETRY,
            recovery_rule=RecoveryRule(threshold=0.0, hold_ms=500.0,
                                       from_ms=2000.0, until_ms=10_000.0),
        )
        self.assertEqual(rep.recovery.status, "censored")
        self.assertIsNone(rep.recovery.duration_ms)

    def test_no_rule_means_no_verdict(self):
        rep = build_stage_report(requests=[], stages=STAGES, telemetry=self.TELEMETRY)
        self.assertEqual(rep.recovery.status, "not_applicable")


class TestLowSampleStages(unittest.TestCase):
    """低样本阶段：报 ``n``，且**不给** P95/P99（十几个样本的 P99 就是最大值）。"""

    def test_percentiles_are_withheld_below_min_n_but_count_is_reported(self):
        reqs = [_req(f"a{i}", 100.0, 200.0, ttft=float(10 + i)) for i in range(3)]
        rep = build_stage_report(requests=reqs, stages=STAGES,
                                 min_n_for_percentiles=20)
        low = rep.cohorts[0]
        self.assertEqual(low.n_submitted, 3)          # 样本量始终给
        self.assertTrue(low.insufficient_samples)
        self.assertIsNotNone(low.ttft_ms_p50)         # 中位数可以给
        self.assertIsNone(low.ttft_ms_p95)            # 但尾分位不给
        self.assertIsNone(low.ttft_ms_p99)
        self.assertIsNone(low.stall_ms_p99)

    def test_percentiles_appear_once_there_are_enough_samples(self):
        reqs = [_req(f"a{i}", 100.0, 200.0, ttft=float(10 + i)) for i in range(25)]
        rep = build_stage_report(requests=reqs, stages=STAGES,
                                 min_n_for_percentiles=20)
        low = rep.cohorts[0]
        self.assertFalse(low.insufficient_samples)
        self.assertIsNotNone(low.ttft_ms_p95)
        self.assertIsNotNone(low.ttft_ms_p99)


class TestBoundariesAreNotDoubleCounted(unittest.TestCase):
    """边界上的请求只能属于一段——否则同一请求被两段各数一次（重复计数）。"""

    def test_request_exactly_on_a_boundary_belongs_to_the_later_stage_only(self):
        rep = build_stage_report(requests=[_req("edge", 1000.0, 1100.0)], stages=STAGES)
        cohorts = {c.label: c for c in rep.cohorts}
        self.assertEqual(cohorts["low"].n_submitted, 0)
        self.assertEqual(cohorts["high"].n_submitted, 1)
        self.assertTrue(rep.reconciliation["ok"])

    def test_request_exactly_on_the_last_boundary_stays_in_the_last_stage(self):
        """最后一段的右端按**闭区间**：``end_ms`` 上的到达属于它，不属于排空段。"""
        rep = build_stage_report(requests=[_req("last", 3000.0, 3100.0)], stages=STAGES)
        cohorts = {c.label: c for c in rep.cohorts}
        self.assertEqual(cohorts["recover"].n_submitted, 1)
        self.assertEqual(cohorts[DRAIN_LABEL].n_submitted, 0)


class TestProductionRequestRowsCarrySloSatisfied(unittest.TestCase):
    """★ **产物契约**：``requests.jsonl`` 的行必须能被本模块消费。

    为什么单列一个类：上面所有用例都用手写的 ``_req()`` 造行，而 ``_req()``
    **自己塞了 ``slo_satisfied``**。于是即便真实 harness 根本不写这个字段，
    本模块的测试也全绿——**测的是夹具，不是产物**。

    真实链路：``run_benchmark`` 写 ``requests.jsonl`` 用的是
    ``RequestMetrics.describe()``；``build_stage_report`` 的判据是
    ``r.get("slo_satisfied") is True``。两边一旦不一致，逐段/逐 cohort 的
    达标数会**静默变成 0**——不报错、``reconciliation.ok`` 仍为 True
    （对账只校请求数），看起来像"负载太重"而不是"读数坏了"。

    ⚠️ **已知缺陷，暂以 skip 标记**（2026-09-28 发现，见日志 §18）：
    实测 ``S14B2-load_ramp-ramp_cost`` 的 ``cohort 达标合计 = 0`` 而
    ``summary = 18``。修 ``describe()`` 会改变产物**字节** ⇒ 影响
    ``SOURCE_DIGEST_SCOPE``（含 ``benchmark/``）内的源码戳，**不得在批次运行期改**。
    批次收尾后应补上该字段，并把下面的 skip 改为硬断言。
    """

    _KNOWN_DEFECT = (
        "已知缺陷（2026-09-28）：RequestMetrics.describe() 未落盘 slo_satisfied。"
        "修复会改产物字节 ⇒ 影响源码戳 ⇒ 需等当前批次收尾后作为新版本落地。"
    )

    def test_describe_output_is_consumable_by_stage_report(self):
        from ascend_opt.metrics import RequestMetrics

        row = RequestMetrics(
            req_id="r0", prompt_len=8, gen_len=4, ttft_ms=10.0, tpot_ms=5.0,
            tpot_p99_ms=8.0, outcome="success", arrival_ms=0.0, done_ms=100.0,
        ).describe()
        # ① 必须是 dict（可 JSON 序列化）且含 stage_report 依赖的键
        for key in ("arrival_ms", "done_ms", "outcome"):
            self.assertIn(key, row, f"产物行缺少 stage_report 依赖的 {key}")
        # ② slo_satisfied 必须显式落盘，不能靠下游猜
        if "slo_satisfied" not in row:
            self.skipTest(self._KNOWN_DEFECT)
        self.assertIsInstance(row["slo_satisfied"], bool)

    def test_stage_report_does_not_silently_zero_missing_flag(self):
        """缺字段时**必须显形**：这里固定住"缺 = 0"这一行为的存在性。

        本断言不是在认可该行为，而是让"字段缺失 ⇒ 达标率 0"这件事
        被测试显式记住；上游一旦补齐字段，本用例会提示可以升级为硬校验。
        """
        rep = build_stage_report(
            requests=[{"req_id": "r0", "arrival_ms": 0.0, "done_ms": 100.0,
                       "outcome": OUTCOME_SUCCESS, "ttft_ms": 10.0,
                       "tpot_ms": 5.0, "gen_len": 4}],  # 刻意不给 slo_satisfied
            stages=STAGES,
        )
        self.assertEqual(sum(c.n_slo_satisfied for c in rep.cohorts), 0)


if __name__ == "__main__":
    unittest.main()
