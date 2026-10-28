"""工作负载生成器的单元测试。

重点是**确定性与输入校验**：
生成器不确定，跨 run 对照就失去意义；校验缺失，错误的负载会静默跑出一堆
「看起来正常但口径不对」的数字。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.workload import (  # noqa: E402
    RateSegment,
    WorkloadSpec,
    build_requests,
    summarize_workload,
)


class TestDeterminism(unittest.TestCase):
    def test_same_spec_same_output(self):
        spec = WorkloadSpec(name="t", num_requests=32, prompt_len_min=100,
                            prompt_len_max=400, gen_len_min=8, gen_len_max=32, seed=7)
        first = build_requests(spec)
        second = build_requests(spec)
        self.assertEqual([r.describe() for r in first], [r.describe() for r in second])

    def test_different_seed_changes_sequence(self):
        base = dict(name="t", num_requests=64, prompt_len_min=100,
                    prompt_len_max=400, gen_len_min=8, gen_len_max=32)
        a = build_requests(WorkloadSpec(seed=1, **base))
        b = build_requests(WorkloadSpec(seed=2, **base))
        self.assertNotEqual([r.prompt_len for r in a], [r.prompt_len for r in b])

    def test_lengths_within_declared_range(self):
        spec = WorkloadSpec(name="t", num_requests=128, prompt_len_min=50,
                            prompt_len_max=80, gen_len_min=4, gen_len_max=9, seed=3)
        for req in build_requests(spec):
            self.assertTrue(50 <= req.prompt_len <= 80)
            self.assertTrue(4 <= req.gen_len <= 9)


class TestValidation(unittest.TestCase):
    def test_shared_prefix_longer_than_prompt_is_rejected(self):
        spec = WorkloadSpec(name="t", num_requests=4, prompt_len_min=64,
                            prompt_len_max=128, shared_prefix_len=256)
        with self.assertRaises(ValueError):
            build_requests(spec)

    def test_zero_gen_len_is_rejected(self):
        # gen_len=0 会让 TPOT 无定义，必须在入口拦掉
        spec = WorkloadSpec(name="t", num_requests=4, gen_len_min=0, gen_len_max=4)
        with self.assertRaises(ValueError):
            build_requests(spec)

    def test_inverted_range_is_rejected(self):
        with self.assertRaises(ValueError):
            build_requests(WorkloadSpec(name="t", prompt_len_min=200, prompt_len_max=100))

    def test_non_positive_num_requests_is_rejected(self):
        with self.assertRaises(ValueError):
            build_requests(WorkloadSpec(name="t", num_requests=0))

    def test_unknown_field_in_from_dict_is_rejected(self):
        # 配置里写错字段名必须立刻炸，不能静默忽略
        with self.assertRaises(ValueError):
            WorkloadSpec.from_dict({"name": "t", "promt_len_min": 64})


class TestArrivalModels(unittest.TestCase):
    def test_closed_model_arrives_immediately(self):
        spec = WorkloadSpec(name="t", num_requests=8, arrival="closed", concurrency=4)
        self.assertTrue(all(r.arrival_ms == 0.0 for r in build_requests(spec)))

    def test_poisson_arrivals_are_monotonic(self):
        spec = WorkloadSpec(name="t", num_requests=64, arrival="poisson",
                            request_rate_qps=4.0, seed=11)
        arrivals = [r.arrival_ms for r in build_requests(spec)]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertGreater(arrivals[-1], 0.0)

    def test_burst_groups_requests_together(self):
        spec = WorkloadSpec(name="t", num_requests=12, arrival="burst",
                            concurrency=4, request_rate_qps=2.0, seed=1)
        arrivals = [r.arrival_ms for r in build_requests(spec)]
        # 前 4 个同一时刻到达
        self.assertEqual(len(set(arrivals[:4])), 1)
        self.assertLess(arrivals[0], arrivals[4])

    def test_unknown_arrival_model_raises(self):
        spec = WorkloadSpec(name="t", num_requests=4, arrival="wishful")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_requests(spec)


class TestPiecewiseArrival(unittest.TestCase):
    """分段常量速率 λ(t) 到达模型。

    存在的理由：closed / poisson / burst 都表达不出「低压 → 高压 → 恢复」的
    **负载水平轨迹**，而那正是本研究的核心问题（控制器在负载变化下如何表现）
    唯一的自变量。这个模型必须让"何时升压/降压"是**墙钟上确定的**，
    否则跨 run 无法对照。
    """

    SEGMENTS = (
        RateSegment(duration_s=30.0, qps=0.5, label="low"),
        RateSegment(duration_s=30.0, qps=2.0, label="high"),
        RateSegment(duration_s=30.0, qps=0.5, label="recover"),
    )

    def _spec(self, **kw):
        base = dict(
            name="ramp", arrival="piecewise", rate_segments=self.SEGMENTS,
            prompt_len_min=512, prompt_len_max=512, gen_len_min=8, gen_len_max=8,
            num_requests=90,  # = round(15) + round(60) + round(15)，由段定义决定
            seed=3,
        )
        base.update(kw)
        return WorkloadSpec(**base)

    def test_num_requests_is_derived_from_segments(self):
        # jittered：每段取整后求和 = round(15) + round(60) + round(15) = 90
        reqs = build_requests(self._spec())
        self.assertEqual(len(reqs), 90)

    def test_num_requests_mismatch_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            WorkloadSpec.from_dict({
                "name": "ramp", "arrival": "piecewise", "num_requests": 64,
                "rate_segments": [
                    {"duration_s": 30.0, "qps": 0.5},
                    {"duration_s": 30.0, "qps": 2.0},
                ],
            })
        self.assertIn("num_requests", str(ctx.exception))
        self.assertIn("75", str(ctx.exception))

    def test_mismatch_is_rejected_for_direct_construction_too(self):
        # 直接构造也要受同一条约束，否则"配置路径报错、代码路径静默"
        with self.assertRaises(ValueError):
            self._spec(num_requests=64)

    def test_segments_are_required(self):
        with self.assertRaises(ValueError):
            self._spec(rate_segments=())

    def test_bad_segment_values_are_rejected(self):
        for bad in ({"duration_s": 0.0, "qps": 1.0}, {"duration_s": 10.0, "qps": 0.0}):
            with self.assertRaises(ValueError):
                self._spec(rate_segments=(RateSegment(**bad),), num_requests=1)

    def test_unknown_segment_field_is_rejected(self):
        # 段内字段拼错必须炸：静默退化会让整段负载消失而产物里看不出来
        with self.assertRaises(ValueError):
            WorkloadSpec.from_dict({
                "name": "ramp", "arrival": "piecewise",
                "rate_segments": [{"duration_s": 10.0, "qps_s": 1.0}],
            })

    def test_segments_without_piecewise_is_rejected(self):
        with self.assertRaises(ValueError):
            WorkloadSpec.from_dict({
                "name": "x", "arrival": "closed",
                "rate_segments": [{"duration_s": 10.0, "qps": 1.0}],
            })

    def test_jittered_phase_counts_are_exact(self):
        arrivals = [r.arrival_ms for r in build_requests(self._spec())]
        first = [a for a in arrivals if a < 30000.0]
        second = [a for a in arrivals if 30000.0 <= a < 60000.0]
        third = [a for a in arrivals if a >= 60000.0]
        self.assertEqual((len(first), len(second), len(third)), (15, 60, 15))

    def test_arrivals_stay_inside_their_segment(self):
        arrivals = [r.arrival_ms for r in build_requests(self._spec())]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertTrue(all(0.0 <= a < 90000.0 for a in arrivals))

    def test_jittered_is_not_on_a_periodic_grid(self):
        # 严格等间隔会与执行器步长网格相位锁定（P1S3D E7 的教训）：
        # 同段内相邻间隔必须不全相等。
        arrivals = [r.arrival_ms for r in build_requests(self._spec())]
        second = [a for a in arrivals if 30000.0 <= a < 60000.0]
        gaps = [round(b - a, 3) for a, b in zip(second, second[1:])]
        self.assertGreater(len(set(gaps)), 1)

    def test_poisson_mode_keeps_total_count(self):
        reqs = build_requests(self._spec(rate_noise="poisson"))
        # poisson 总数 = round(Σ λ·d) = round(90) = 90
        self.assertEqual(len(reqs), 90)
        arrivals = [r.arrival_ms for r in reqs]
        self.assertEqual(arrivals, sorted(arrivals))
        self.assertTrue(all(0.0 <= a < 90000.0 for a in arrivals))

    def test_poisson_phase_counts_fluctuate(self):
        # 记录这条不是为了断言"一定波动"（同 seed 固定），而是把语义钉住：
        # poisson 模式**不保证**段内计数等于名义值，分析时必须读实际到达。
        reqs = build_requests(self._spec(rate_noise="poisson"))
        second = [r.arrival_ms for r in reqs if 30000.0 <= r.arrival_ms < 60000.0]
        self.assertNotEqual(len(second), 0)

    def test_same_seed_same_output(self):
        a = [r.arrival_ms for r in build_requests(self._spec())]
        b = [r.arrival_ms for r in build_requests(self._spec())]
        self.assertEqual(a, b)

    def test_from_dict_coerces_plain_dicts(self):
        spec = WorkloadSpec.from_dict({
            "name": "ramp", "arrival": "piecewise",
            "rate_segments": [
                {"duration_s": 20.0, "qps": 1.0, "label": "low"},
                {"duration_s": 20.0, "qps": 3.0, "label": "high"},
            ],
            "seed": 1,
        })
        self.assertIsInstance(spec.rate_segments[0], RateSegment)
        self.assertEqual(spec.num_requests, 80)
        self.assertEqual(len(build_requests(spec)), 80)

    def test_fingerprint_reports_nominal_and_realized(self):
        spec = self._spec()
        reqs = build_requests(spec)
        fp = summarize_workload(reqs, spec)
        segs = fp["arrival_segments"]
        self.assertEqual(len(segs), 3)
        self.assertEqual([s["label"] for s in segs], ["low", "high", "recover"])
        for s in segs:
            # 名义与实际必须**同时**给出：只报名义等于假定"想要多少就来多少"
            self.assertIn("nominal_qps", s)
            self.assertIn("realized_qps", s)
            self.assertIn("realized_count", s)
        self.assertEqual(segs[1]["nominal_count"], 60.0)
        self.assertEqual(segs[1]["realized_count"], 60)


class TestPrefixGroups(unittest.TestCase):
    def test_no_groups_means_no_shared_prefix(self):
        spec = WorkloadSpec(name="t", num_requests=16, prompt_len_min=256,
                            prompt_len_max=256, shared_prefix_len=128, prefix_groups=0)
        self.assertTrue(all(r.shared_prefix_len == 0 for r in build_requests(spec)))

    def test_groups_are_stable_and_bounded(self):
        spec = WorkloadSpec(name="t", num_requests=32, prompt_len_min=512,
                            prompt_len_max=512, shared_prefix_len=256,
                            prefix_groups=3, seed=5)
        reqs = build_requests(spec)
        groups = {r.prefix_group for r in reqs}
        self.assertTrue(groups <= {"g000", "g001", "g002"})
        self.assertTrue(all(r.shared_prefix_len == 256 for r in reqs))
        # 分组由 req_id 哈希决定，与生成顺序无关
        self.assertEqual(
            [r.prefix_group for r in reqs], [r.prefix_group for r in build_requests(spec)]
        )

    def test_unique_prefix_len(self):
        spec = WorkloadSpec(name="t", num_requests=4, prompt_len_min=1024,
                            prompt_len_max=1024, shared_prefix_len=768, prefix_groups=1)
        for req in build_requests(spec):
            self.assertEqual(req.unique_prefix_len, 256)
            self.assertEqual(req.total_tokens, req.prompt_len + req.gen_len)


class TestFingerprint(unittest.TestCase):
    def test_totals_match_inputs(self):
        spec = WorkloadSpec(name="t", num_requests=10, prompt_len_min=100,
                            prompt_len_max=100, gen_len_min=10, gen_len_max=10,
                            prefix_groups=1, shared_prefix_len=64, seed=0)
        reqs = build_requests(spec)
        fp = summarize_workload(reqs)
        self.assertEqual(fp["num_requests"], 10)
        self.assertEqual(fp["total_prompt_tokens"], 1000)
        self.assertEqual(fp["total_output_tokens"], 100)
        self.assertEqual(fp["reusable_prompt_tokens"], 640)
        self.assertEqual(fp["distinct_prefix_groups"], 1)

    def test_empty_list_is_handled(self):
        self.assertEqual(summarize_workload([]), {"num_requests": 0})


if __name__ == "__main__":
    unittest.main()
