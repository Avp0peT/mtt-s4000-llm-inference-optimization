"""指标层的单元测试。

跑法：python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.metrics import (  # noqa: E402
    OUTCOME_FAILED,
    OUTCOME_REJECTED,
    OUTCOME_UNFINISHED,
    RequestMetrics,
    SLO,
    coeff_variation,
    compare,
    percentile,
    robust_cv,
    summarize,
)


class TestPercentile(unittest.TestCase):
    def test_known_values(self):
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(values, 0), 1.0)
        self.assertAlmostEqual(percentile(values, 50), 2.5)
        self.assertAlmostEqual(percentile(values, 100), 4.0)
        self.assertAlmostEqual(percentile(values, 25), 1.75)

    def test_single_value(self):
        self.assertAlmostEqual(percentile([7.5], 99), 7.5)

    def test_unsorted_input_is_handled(self):
        self.assertAlmostEqual(percentile([4.0, 1.0, 3.0, 2.0], 50), 2.5)

    def test_empty_raises_instead_of_returning_nan(self):
        # 返回 nan 会静默污染整张汇总表，必须显式失败
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_out_of_range_q_raises(self):
        with self.assertRaises(ValueError):
            percentile([1.0], 101)


class TestStability(unittest.TestCase):
    def test_robust_cv_zero_for_constant_series(self):
        self.assertAlmostEqual(robust_cv([5.0, 5.0, 5.0, 5.0]), 0.0)

    def test_robust_cv_ignores_single_outlier(self):
        # 这是选用稳健 CV 的核心理由：一个 outlier 不该让"稳定性"判据失效
        stable = [10.0, 10.1, 9.9, 10.05, 10.02]
        polluted = stable + [500.0]
        self.assertLess(robust_cv(polluted), 0.10)
        # 普通 CV 会被这个 outlier 拉高一个数量级
        self.assertGreater(coeff_variation(polluted), 5 * coeff_variation(stable))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            robust_cv([])


class TestSummarize(unittest.TestCase):
    def _metrics(self):
        return [
            RequestMetrics("r0", 100, 10, ttft_ms=100.0, tpot_ms=10.0),
            RequestMetrics("r1", 100, 10, ttft_ms=200.0, tpot_ms=20.0),
            RequestMetrics("r2", 100, 10, ttft_ms=300.0, tpot_ms=30.0),
        ]

    def test_distribution_and_throughput(self):
        summary = summarize(
            self._metrics(),
            run_id="t",
            workload="w",
            config="c",
            wall_ms=1000.0,
        )
        self.assertEqual(summary.n_requests, 3)
        self.assertAlmostEqual(summary.ttft["p50"], 200.0)
        # [10, 20, 30] 的 99 分位按线性插值 = 20*0.02 + 30*0.98 = 29.8
        # （与 numpy.percentile 默认行为一致，不是 30；口径必须写清楚）
        self.assertAlmostEqual(summary.tpot["p99"], 29.8)
        # 30 个输出 token / 1 秒
        self.assertAlmostEqual(summary.throughput_tok_s, 30.0)
        self.assertAlmostEqual(summary.ttft["max"], 300.0)

    def test_e2e_formula(self):
        m = RequestMetrics("r", 10, 5, ttft_ms=100.0, tpot_ms=10.0)
        # 100 + 10 * (5 - 1) = 140
        self.assertAlmostEqual(m.e2e_ms, 140.0)

    def test_e2e_for_single_token_equals_ttft(self):
        m = RequestMetrics("r", 10, 1, ttft_ms=100.0, tpot_ms=10.0)
        self.assertAlmostEqual(m.e2e_ms, 100.0)

    def test_slo_satisfaction_rate_uses_all_submitted_as_denominator(self):
        summary = summarize(
            self._metrics(),
            run_id="t",
            workload="w",
            config="c",
            wall_ms=1000.0,
            slo=SLO(ttft_ms=250.0),
        )
        # 只有 r0、r1 满足 TTFT <= 250ms。分母是**全部提交**（3），不是成功数。
        self.assertEqual(summary.n_slo_satisfied, 2)
        self.assertAlmostEqual(summary.slo_satisfaction_rate, 2 / 3)
        # 达标吞吐另给两个口径：请求数/秒 与 token/秒
        self.assertAlmostEqual(summary.slo_goodput_req_s, 2.0)
        self.assertAlmostEqual(summary.slo_goodput_tok_s, 20.0)

    def test_failures_shrink_slo_rate_instead_of_being_dropped(self):
        """失败请求必须留在分母里，且不算达标。

        这是被修掉的口径缺陷：分母取成功数时，一批"大量请求失败"的 run
        反而显示出很高的达标率——越失败看起来越健康。
        """
        metrics = self._metrics()
        # 时延刻意给得很低：如果只看时延不看结局，它会被算成达标。
        metrics.append(
            RequestMetrics("bad", 100, 10, ttft_ms=1.0, tpot_ms=1.0,
                           outcome=OUTCOME_FAILED)
        )
        summary = summarize(
            metrics, run_id="t", workload="w", config="c", wall_ms=1000.0,
            slo=SLO(ttft_ms=250.0),
        )
        self.assertEqual(summary.n_requests, 4)
        self.assertEqual(summary.n_success, 3)
        self.assertEqual(summary.n_failed, 1)
        self.assertAlmostEqual(summary.failure_rate, 0.25)
        # 分母是 4（不是 3）：达标 2 个 -> 0.5
        self.assertAlmostEqual(summary.slo_satisfaction_rate, 0.5)
        self.assertEqual(summary.n_slo_satisfied, 2)

    def test_outcomes_are_counted_separately(self):
        metrics = self._metrics()
        metrics.append(RequestMetrics("f", 10, 1, 1.0, 1.0, outcome=OUTCOME_FAILED))
        metrics.append(RequestMetrics("j", 10, 1, 1.0, 1.0, outcome=OUTCOME_REJECTED))
        metrics.append(RequestMetrics("u", 10, 1, 1.0, 1.0, outcome=OUTCOME_UNFINISHED))
        summary = summarize(metrics, run_id="t", workload="w", config="c", wall_ms=1000.0)
        self.assertEqual(summary.n_failed, 1)
        self.assertEqual(summary.n_rejected, 1)
        self.assertEqual(summary.n_unfinished, 1)
        self.assertEqual(summary.n_success, 3)
        self.assertEqual(summary.n_requests, 6)
        # 三率的分母都是全部提交数，且互不重叠
        self.assertAlmostEqual(
            summary.failure_rate + summary.rejection_rate + summary.unfinished_rate,
            0.5,
        )

    def test_failed_requests_are_excluded_from_distributions(self):
        metrics = self._metrics()
        metrics.append(
            RequestMetrics("bad", 100, 10, ttft_ms=99999.0, tpot_ms=99999.0,
                           outcome=OUTCOME_FAILED)
        )
        summary = summarize(metrics, run_id="t", workload="w", config="c", wall_ms=1000.0)
        self.assertEqual(summary.n_requests, 4)
        self.assertEqual(summary.n_success, 3)
        self.assertAlmostEqual(summary.ttft["max"], 300.0)
        # 样本量必须一起报：P99 的可靠性完全取决于这个数
        self.assertEqual(summary.ttft["n"], 3)

    def test_unconfigured_slo_yields_none_not_zero(self):
        """没配 SLO 时三个量都是 None（不可计算），不能写成 0。

        写 0 会把"没配"伪装成"一个都没达标"，于是一个正常 run 看起来像彻底失败。
        """
        summary = summarize(
            self._metrics(), run_id="t", workload="w", config="c", wall_ms=1000.0
        )
        self.assertIsNone(summary.slo_satisfaction_rate)
        self.assertIsNone(summary.slo_goodput_req_s)
        self.assertIsNone(summary.slo_goodput_tok_s)
        self.assertIsNone(summary.describe()["slo_satisfaction_rate"])
        self.assertIsNone(summary.flat_row()["slo_satisfaction_rate"])

    def test_wall_ms_zero_does_not_divide_by_zero(self):
        summary = summarize(
            self._metrics(), run_id="t", workload="w", config="c", wall_ms=0.0
        )
        self.assertAlmostEqual(summary.throughput_tok_s, 0.0)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            summarize([], run_id="t", workload="w", config="c", wall_ms=1.0)

    def test_all_failed_still_produces_a_row_with_empty_distributions(self):
        """全部失败时**不能抛错**。

        抛异常会让这一行从汇总表里整个消失，而表里少一行看起来像"这个配置还没跑"，
        恰恰把最该被看见的一次运行藏了起来。正确行为是：照常出这一行，
        时延分布为空、失败数为 n、达标率为 0。
        """
        summary = summarize(
            [RequestMetrics("r", 1, 1, 0.0, 0.0, outcome=OUTCOME_FAILED)],
            run_id="t", workload="w", config="c", wall_ms=1.0,
            slo=SLO(ttft_ms=100.0),
        )
        self.assertEqual(summary.n_requests, 1)
        self.assertEqual(summary.n_failed, 1)
        self.assertEqual(summary.n_success, 0)
        # 空分布而不是一串 0——0 会被读成"测出来就是 0"
        self.assertEqual(summary.ttft, {})
        self.assertEqual(summary.stability, {})
        self.assertAlmostEqual(summary.failure_rate, 1.0)
        self.assertAlmostEqual(summary.slo_satisfaction_rate, 0.0)
        self.assertAlmostEqual(summary.throughput_tok_s, 0.0)


class TestTimingDecomposition(unittest.TestCase):
    """TTFT 的三段分解：受理 / 排队 / prefill 执行。

    只报 TTFT 总数会让"该加容量"与"该调分块"两种对策失去依据——
    而排队变差、计算变好时总数甚至可能是持平的。
    """

    def test_three_segments_sum_to_ttft(self):
        m = RequestMetrics(
            "r", 100, 10, ttft_ms=1000.0, tpot_ms=10.0,
            scheduled_arrival_ms=100.0, arrival_ms=100.0,
            enqueue_ms=180.0, first_schedule_ms=600.0, first_token_ms=1100.0,
            done_ms=1190.0,
        )
        self.assertAlmostEqual(m.admission_ms, 80.0)     # 到达 -> 受理
        self.assertAlmostEqual(m.queue_ms, 420.0)        # 受理 -> 首次调度
        self.assertAlmostEqual(m.prefill_exec_ms, 500.0)  # 首次调度 -> 首 token
        self.assertAlmostEqual(
            m.admission_ms + m.queue_ms + m.prefill_exec_ms, m.ttft_ms
        )

    def test_admission_and_queue_are_distinct(self):
        """受理延迟与排队等待不能混同成一个数。

        混同之后，"加容量"与"修链路/改步粒度"两种对策就没有依据了。
        """
        m = RequestMetrics(
            "r", 10, 2, ttft_ms=100.0, tpot_ms=5.0,
            arrival_ms=10.0, enqueue_ms=60.0,
            first_schedule_ms=110.0, first_token_ms=110.0,
        )
        self.assertAlmostEqual(m.admission_ms, 50.0)
        self.assertAlmostEqual(m.queue_ms, 50.0)
        self.assertAlmostEqual(m.prefill_exec_ms, 0.0)

    def test_summary_reports_the_three_segments(self):
        metrics = [
            RequestMetrics(
                f"r{i}", 10, 2, ttft_ms=100.0, tpot_ms=5.0,
                arrival_ms=0.0, enqueue_ms=10.0,
                first_schedule_ms=30.0, first_token_ms=100.0, done_ms=105.0,
            )
            for i in range(3)
        ]
        summary = summarize(metrics, run_id="t", workload="w", config="c", wall_ms=1000.0)
        self.assertAlmostEqual(summary.admission["p50"], 10.0)
        self.assertAlmostEqual(summary.queue["p50"], 20.0)
        self.assertAlmostEqual(summary.prefill["p50"], 70.0)

    def test_unmeasured_timestamps_stay_none_not_zero(self):
        """从未发生的时刻保持 None，而不是 0。

        写成 0 会把"没发生"报成"耗时 0ms"——最慢的请求看起来最快，
        而且这个数字读起来完全正常，因此不会被追查。
        """
        m = RequestMetrics("r", 10, 2, ttft_ms=None, tpot_ms=0.0,
                           outcome=OUTCOME_UNFINISHED)
        self.assertIsNone(m.admission_ms)
        self.assertIsNone(m.queue_ms)
        self.assertIsNone(m.prefill_exec_ms)
        self.assertIsNone(m.e2e_ms)
        self.assertIsNone(m.accounting_error_ms())
        described = m.describe()
        self.assertIsNone(described["ttft_ms"])
        self.assertIsNone(described["queue_ms"])

    def test_zero_is_a_legitimate_timestamp_not_a_sentinel(self):
        """第一步（t=0）就被调度是合法情形，不能与"从未调度"混为一谈。

        这条守的是哨兵设计本身：若用 0 表示"未设置"，这个请求的 prefill 段
        会算出 0，而它实际算了 1210ms——一个"没测"被报成了"零成本"。
        """
        m = RequestMetrics(
            "r", 100, 2, ttft_ms=1210.5, tpot_ms=10.0,
            scheduled_arrival_ms=0.0, arrival_ms=0.0, enqueue_ms=0.0,
            first_schedule_ms=0.0, first_token_ms=1210.5, done_ms=1220.5,
        )
        self.assertEqual(m.prefill_exec_ms, 1210.5)
        # 这两个 0 是真实的 0（确实没等），不是"没测"
        self.assertEqual(m.admission_ms, 0.0)
        self.assertEqual(m.queue_ms, 0.0)

    def test_unmeasurable_latency_is_not_satisfied(self):
        """没有可测时延的请求不算达标——**不可测 ≠ 达标**。

        否则一个"连首 token 都没产出来"的请求会因为"没超过阈值"而被算作达标。
        这里刻意构造一个自相矛盾的输入（标成成功却没有 ttft）来证明守卫真的生效：
        真实路径里这个组合不该出现，但判据不能依赖调用方守规矩。
        """
        m = RequestMetrics("r", 10, 2, ttft_ms=None, tpot_ms=0.0)
        self.assertFalse(SLO(ttft_ms=9999.0).satisfied_by(m))


class TestCompare(unittest.TestCase):
    def _run(self, run_id, ttft_scale=1.0, tpot_scale=1.0, spread=0.10, n=100):
        """构造一个**组内有离散度**的 run。

        这一点很关键：如果一次 run 内所有请求时延完全相同，稳健 CV 就是 0，
        噪声地板也随之变成 0，任何微小的差异都会被判为"显著"。
        真实测量不会出现这种情况，所以测试也必须用有离散度的数据，
        否则测的是一个不会发生的场景。
        """
        metrics = []
        for i in range(n):
            factor = 1.0 + spread * ((i % 21) - 10) / 10.0   # 0.90 ~ 1.10
            metrics.append(
                RequestMetrics(
                    f"a{i}",
                    100,
                    10,
                    ttft_ms=1000.0 * ttft_scale * factor,
                    tpot_ms=100.0 * tpot_scale * factor,
                )
            )
        return summarize(metrics, run_id=run_id, workload="w", config="c", wall_ms=5000.0)

    def test_large_improvement_is_positive(self):
        result = compare(self._run("base"), self._run("cand", 0.5, 0.5))
        self.assertTrue(result["verdict"].startswith("positive"))
        self.assertLess(result["ttft"]["p99_delta_pct"], -40)

    def test_tiny_difference_is_inconclusive_not_positive(self):
        # 这条测试守的是最容易犯的错：把噪声当收益。
        # 组内稳健 CV 约 7%，3% 的"改善"落在噪声地板之内。
        result = compare(self._run("base"), self._run("cand", 0.97, 0.97))
        self.assertGreater(result["noise_floor_pct"], 3.0)
        self.assertTrue(result["verdict"].startswith("inconclusive"))

    def test_mixed_tradeoff_is_reported_as_mixed(self):
        result = compare(self._run("base"), self._run("cand", 2.0, 0.5))
        self.assertTrue(result["verdict"].startswith("mixed"))

    def test_workload_mismatch_produces_warning(self):
        base = self._run("base", 1000.0, 100.0)
        cand = summarize(
            [RequestMetrics("a", 100, 10, 100.0, 10.0)],
            run_id="cand",
            workload="other",
            config="c",
            wall_ms=1000.0,
        )
        result = compare(base, cand)
        self.assertTrue(any("负载不同" in w for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
