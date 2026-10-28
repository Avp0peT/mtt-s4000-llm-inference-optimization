"""扫描轴与规范原型的测试（规范第 6 节）。

这些测试把规范正文的取值**钉在代码里**。规范改了轴而代码没跟上，
是这类实验里最隐蔽的漂移：图还是能出，横轴却已经不是当初约定那个，
而 run_id 上完全看不出来。
"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import workloads as wl  # noqa: E402
from ascend_opt.workload import build_requests  # noqa: E402


class TestAxisConstantsMatchSpec(unittest.TestCase):
    """规范第 6 节逐条列出的取值，改动即实验无效。"""

    def test_input_length_axis(self):
        self.assertEqual(wl.INPUT_LENGTH_AXIS, (128, 512, 2048, 8192, 16384))

    def test_output_length_axis(self):
        self.assertEqual(wl.OUTPUT_LENGTH_AXIS, (128, 512))

    def test_concurrency_axis(self):
        self.assertEqual(wl.CONCURRENCY_AXIS, (1, 8, 16, 32, 64))

    def test_prefix_reuse_axis(self):
        self.assertEqual(wl.PREFIX_REUSE_AXIS, (0.0, 0.25, 0.50, 0.75))


class TestBuildAxisWorkload(unittest.TestCase):
    def test_lengths_are_single_point_not_range(self):
        spec = wl.build_axis_workload(
            input_length=2048, output_length=128, concurrency=8
        )
        self.assertEqual(spec.prompt_len_min, spec.prompt_len_max, 2048)
        self.assertEqual(spec.gen_len_min, spec.gen_len_max, 128)

    def test_zero_reuse_disables_prefix_grouping(self):
        # "没有共享前缀" 与 "有共享前缀但长度为 0" 必须在元数据里可区分。
        spec = wl.build_axis_workload(
            input_length=2048, output_length=128, concurrency=8, prefix_reuse_ratio=0.0
        )
        self.assertEqual(spec.shared_prefix_len, 0)
        self.assertEqual(spec.prefix_groups, 0)

    def test_reuse_ratio_becomes_absolute_shared_prefix_length(self):
        spec = wl.build_axis_workload(
            input_length=4096, output_length=128, concurrency=8, prefix_reuse_ratio=0.75
        )
        self.assertEqual(spec.shared_prefix_len, 3072)
        self.assertGreater(spec.prefix_groups, 0)

    def test_requests_are_actually_buildable(self):
        for ratio in wl.PREFIX_REUSE_AXIS:
            with self.subTest(ratio=ratio):
                spec = wl.build_axis_workload(
                    input_length=4096, output_length=128, concurrency=8,
                    prefix_reuse_ratio=ratio,
                )
                self.assertGreater(len(build_requests(spec)), 0)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            wl.build_axis_workload(input_length=0, output_length=128, concurrency=8)
        with self.assertRaises(ValueError):
            wl.build_axis_workload(input_length=128, output_length=0, concurrency=8)
        with self.assertRaises(ValueError):
            wl.build_axis_workload(input_length=128, output_length=8, concurrency=0)
        with self.assertRaises(ValueError):
            wl.build_axis_workload(
                input_length=128, output_length=8, concurrency=1, prefix_reuse_ratio=1.5
            )


class TestSpecArchetypes(unittest.TestCase):
    """规范第 6 节点名的七类分布，必须都能直接取到并生成请求。"""

    SPEC_ARCHETYPES = (
        "short_in_long_out",
        "long_in_short_out",
        "high_prefix_reuse",
        "low_prefix_reuse",
        "low_concurrency",
        "high_concurrency",
        "mixed",
    )

    def test_all_spec_archetypes_are_registered(self):
        for name in self.SPEC_ARCHETYPES:
            with self.subTest(name=name):
                self.assertIn(name, wl.WORKLOADS)

    def test_all_registered_workloads_are_buildable(self):
        # 注册表里任何一条负载都必须真的能生成请求——否则实验跑到一半才炸。
        for name in wl.list_workloads():
            with self.subTest(name=name):
                spec = wl.get_workload(name)
                self.assertGreater(len(build_requests(spec)), 0, name)

    def test_low_and_high_concurrency_differ_only_in_concurrency(self):
        # 单变量：这两条一旦在别的维度上也不同，就不是一组可比的对照。
        low = wl.get_workload("low_concurrency")
        high = wl.get_workload("high_concurrency")
        self.assertEqual(low.concurrency, 1)
        self.assertEqual(high.concurrency, 64)
        for field in ("prompt_len_min", "prompt_len_max", "gen_len_min", "gen_len_max"):
            self.assertEqual(getattr(low, field), getattr(high, field), field)

    def test_high_and_low_prefix_reuse_differ_only_in_reuse(self):
        high = wl.get_workload("high_prefix_reuse")
        low = wl.get_workload("low_prefix_reuse")
        self.assertEqual(high.concurrency, low.concurrency)
        self.assertEqual(high.prompt_len_min, low.prompt_len_min)
        self.assertGreater(high.shared_prefix_len, low.shared_prefix_len)

    def test_project_local_workloads_are_kept(self):
        for name in ("short_chat", "long_context_qa", "shared_prefix_4k", "mixed_burst"):
            with self.subTest(name=name):
                self.assertIn(name, wl.WORKLOADS)


class TestMatrixSpecs(unittest.TestCase):
    def test_axes_are_fully_covered(self):
        """四个轴的每个取值点都必须出现在清单里，一个都不能漏。"""
        counts = Counter(a for e in wl.matrix_specs() for a in e["_axes"])
        self.assertEqual(counts["input_length"], len(wl.INPUT_LENGTH_AXIS))
        self.assertEqual(counts["output_length"], len(wl.OUTPUT_LENGTH_AXIS))
        self.assertEqual(counts["concurrency"], len(wl.CONCURRENCY_AXIS))
        self.assertEqual(counts["prefix_reuse"], len(wl.PREFIX_REUSE_AXIS))

    def test_shared_anchor_is_deduplicated(self):
        """四条轴共用同一个锚点，去重后只能出现一次。

        不去重的后果不只是多跑几遍：同一条负载配上不同轴的配置文件，
        人眼会以为它们是不同条件，实际完全一样。
        """
        entries = list(wl.matrix_specs())
        names = [e["name"] for e in entries]
        self.assertEqual(len(names), len(set(names)), f"存在重复负载：{names}")

    def test_shared_anchor_is_flagged(self):
        entries = list(wl.matrix_specs())
        anchors = [e for e in entries if e["_is_anchor"]]
        self.assertEqual(len(anchors), 1, "应当恰好有一个被多轴共用的锚点")
        self.assertEqual(
            sorted(anchors[0]["_axes"]),
            ["concurrency", "input_length", "output_length"],
        )

    def test_every_entry_has_a_constant_request_count(self):
        # 单变量原则在负载层面的落实：请求数固定，否则横轴变化无法归因。
        counts = {e["num_requests"] for e in wl.matrix_specs()}
        self.assertEqual(len(counts), 1, f"请求数应恒定，实际有 {counts}")

    def test_entries_are_valid_workload_payloads(self):
        from ascend_opt.workload import WorkloadSpec

        for entry in wl.matrix_specs():
            with self.subTest(name=entry["name"]):
                spec = WorkloadSpec.from_dict(entry)
                self.assertGreater(len(build_requests(spec)), 0)

    def test_underscore_keys_are_annotations_not_fields(self):
        # 下划线开头的键是注释，必须被 from_dict 忽略而不是报未知字段。
        entry = next(iter(wl.matrix_specs()))
        self.assertTrue(any(k.startswith("_") for k in entry))
        from ascend_opt.workload import WorkloadSpec

        WorkloadSpec.from_dict(entry)   # 不应抛错

    def test_unknown_field_still_raises(self):
        # 注释豁免不能削弱"拼错字段必须报错"这条严格性。
        from ascend_opt.workload import WorkloadSpec

        entry = dict(next(iter(wl.matrix_specs())))
        entry["prompt_lens_min"] = 100      # 拼错
        with self.assertRaises(ValueError):
            WorkloadSpec.from_dict(entry)

    def test_request_count_is_overridable(self):
        counts = {e["num_requests"] for e in wl.matrix_specs(requests=8)}
        self.assertEqual(counts, {8})


if __name__ == "__main__":
    unittest.main()
