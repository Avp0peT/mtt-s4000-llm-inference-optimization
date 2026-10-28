"""P99 对比图的绘制纪律（评审 Part 1 #3）。

这批测试的存在理由：绘图逻辑原先内嵌在 ``export_results.sh`` 的 heredoc 里，
**没有任何测试**，于是两个缺陷静默存活到了评审——

  A. 空分位块被兜底成 ``0.0`` → 图上出现 "TPOT P99 = 0 ms" 的柱子。
     0 看起来是一个好得不可能的**测量值**，比缺一根柱子更容易误导。
  B. 分组只按 ``config`` → "同一份配置跑在两个负载上"的数字被混进同一个
     中位数，那个数谁都没测过。

两者都不是"报错的"缺陷，而是"看起来正常"的缺陷——所以必须由测试固定。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_p99  # noqa: E402


def row(
    *,
    run_id: str = "r0",
    workload: str = "short_chat",
    config: str = "baseline",
    ttft: str = "100",
    tpot: str = "10",
    stall: str = "1",
    label: str = "MEASURED",
    platform: str = "unknown",
    backend: str = "unknown",
    precision: str = "",
) -> dict[str, str]:
    """一行汇总表。空字符串代表该口径**没有可用分位数**（合法状态）。"""
    return {
        "run_id": run_id,
        "workload": workload,
        "config": config,
        "ttft_p99": ttft,
        "tpot_p99": tpot,
        "stall_p99": stall,
        "data_label": label,
        "hardware_platform": platform,
        "backend": backend,
        "precision": precision,
    }


class TestMissingIsNeverZero(unittest.TestCase):
    """缺失必须是 ``None``，绝不能被兜底成 0。"""

    def test_median_of_empty_is_none_not_zero(self):
        """这条就是那个缺陷的最小复现：旧实现返回 0.0。"""
        self.assertIsNone(plot_p99.median_or_none([]))
        self.assertIsNone(plot_p99.median_or_none([None, None]))

    def test_median_of_values_is_ordinary(self):
        self.assertEqual(plot_p99.median_or_none([1.0, 3.0]), 2.0)
        self.assertEqual(plot_p99.median_or_none([1.0, 2.0, 6.0]), 2.0)

    def test_blank_cell_reads_as_none(self):
        self.assertIsNone(plot_p99._fnum(""))
        self.assertIsNone(plot_p99._fnum(None))
        self.assertIsNone(plot_p99._fnum("   "))
        self.assertEqual(plot_p99._fnum("12.5"), 12.5)
        # 非数字也不能被当成 0
        self.assertIsNone(plot_p99._fnum("n/a"))

    def test_missing_metric_stays_none_in_panel(self):
        """TPOT 整列为空 → 该面板此组是 None，**不是** 0。"""
        plan = plot_p99.build_plan([row(tpot="")])
        self.assertEqual(plan.panels["tpot"], [None])
        self.assertIn("tpot", plan.omitted["short_chat/baseline"])
        # TTFT 仍在，所以整组没被跳过
        self.assertEqual(plan.labels, ["short_chat/baseline"])

    def test_panel_never_contains_zero_for_missing(self):
        """泛化断言：没有一行提供 stall 时，stall 面板不得出现任何数字。"""
        rows = [row(run_id="r0", stall=""), row(run_id="r1", stall="")]
        plan = plot_p99.build_plan(rows)
        self.assertTrue(all(v is None for v in plan.panels["stall"]))
        self.assertNotIn(0, [v for v in plan.panels["stall"] if v is not None])


class TestGroupingIsWorkloadPlusConfig(unittest.TestCase):
    """分组键必须是 (workload, config)，不是 config 单独一项。"""

    def test_same_config_on_two_workloads_stays_separate(self):
        """同一份配置跑在两个负载上 → 两组，中位数各自独立。

        旧实现会把它们合成一组，中位数 = (100 + 1000)/2 = 550——
        一个既非 A 也非 B、谁都没测过的数。
        """
        plan = plot_p99.build_plan([
            row(run_id="a", workload="short_chat", config="baseline", ttft="100"),
            row(run_id="b", workload="long_context_qa", config="baseline", ttft="1000"),
        ])
        self.assertEqual(
            sorted(plan.labels),
            ["long_context_qa/baseline", "short_chat/baseline"],
        )
        by_label = dict(zip(plan.labels, plan.panels["ttft"]))
        self.assertEqual(by_label["short_chat/baseline"], 100.0)
        self.assertEqual(by_label["long_context_qa/baseline"], 1000.0)

    def test_repeats_of_one_group_are_medians(self):
        plan = plot_p99.build_plan([
            row(run_id="r0", ttft="100"),
            row(run_id="r1", ttft="200"),
            row(run_id="r2", ttft="300"),
        ])
        self.assertEqual(plan.labels, ["short_chat/baseline"])
        self.assertEqual(plan.panels["ttft"], [200.0])
        self.assertEqual(plan.run_ids["short_chat/baseline"], ["r0", "r1", "r2"])

    def test_label_count_matches_bar_count(self):
        """标签必须与柱子一一对应，否则每根柱子都顶着别人的名字。"""
        plan = plot_p99.build_plan([
            row(run_id="a", workload="w1", config="c"),
            row(run_id="b", workload="w2", config="c"),
            row(run_id="c", workload="w3", config="c"),
        ])
        self.assertEqual(len(plan.labels), len(plan.display_labels))
        for metric in plot_p99.METRICS:
            self.assertEqual(len(plan.panels[metric]), len(plan.labels))


class TestSkippingAndSourceRecord(unittest.TestCase):
    """无数据的组要被略去并**记录下来**，不能只从数据里消失。"""

    def test_group_without_ttft_is_skipped_and_recorded(self):
        plan = plot_p99.build_plan([
            row(run_id="a", workload="w_ok", ttft="100"),
            row(run_id="b", workload="w_dead", ttft=""),
        ])
        self.assertEqual(plan.labels, ["w_ok/baseline"])
        self.assertEqual(plan.skipped, ["w_dead/baseline"])

    def test_nothing_plottable_is_reported(self):
        plan = plot_p99.build_plan([row(ttft="")])
        self.assertFalse(plan.has_data)
        self.assertEqual(plan.skipped, ["short_chat/baseline"])

    def test_source_file_records_missing_and_runs(self):
        """被省略的口径必须落盘，否则事后无法解释这根柱子为什么不在。"""
        plan = plot_p99.build_plan([
            row(run_id="r0", stall="", tpot=""),
            row(run_id="r1", workload="w2", ttft="", stall="2"),
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.source.txt"
            plot_p99.write_source(plan, path)
            text = path.read_text(encoding="utf-8")
        # 分组口径那行必须写出**全部**维度。少写一个（曾经漏了 backend）
        # 会让复核这张图的人以为模拟与真机的中位数合并过了——图是对的，
        # 说明是错的，比两者都错更难发现。
        self.assertIn(
            "grouping: (workload, config, hardware_platform, backend)", text
        )
        self.assertIn("不画成 0", text)
        self.assertIn("missing_in_group: short_chat/baseline -> tpot, stall", text)
        self.assertIn("omitted_groups: w2/baseline", text)
        self.assertIn("r0", text)
        # 环境维度必须落盘：图缩进论文后没人能再从标题读出它是哪个平台的数据。
        self.assertIn("platforms:", text)
        self.assertIn("backends:", text)
        self.assertIn("precisions:", text)

    def test_omitted_messages_are_surfaced(self):
        plan = plot_p99.build_plan([row(stall="")])
        text = "\n".join(plot_p99.describe_plan(plan))
        self.assertIn("stall", text)

    def test_complete_data_has_no_omissions(self):
        """正向对照：数据齐时不该出现任何 omitted / skipped。"""
        plan = plot_p99.build_plan([row(run_id="r0"), row(run_id="r1")])
        self.assertTrue(plan.has_data)
        self.assertEqual(plan.omitted, {})
        self.assertEqual(plan.skipped, [])
        self.assertIsNotNone(plan.panels["stall"][0])


class TestGroupingIsPlatformIsolated(unittest.TestCase):
    """分组必须按平台隔离（评审 #2 的复现：100 与 200 被合成了一根 150 的柱）。

    "跨平台合并"的可怕之处在于它**不报错**：表照样生成、图照样画出来，
    只是图上多了一根谁都没测过的柱子。所以必须由测试钉住。
    """

    def test_two_platforms_never_share_a_bar(self):
        plan = plot_p99.build_plan([
            row(run_id="a", platform="ascend", ttft="100"),
            row(run_id="b", platform="mthreads", ttft="200"),
        ])
        self.assertEqual(len(plan.labels), 2, "两个平台必须各占一根柱")
        values = sorted(v for v in plan.panels["ttft"] if v is not None)
        self.assertEqual(values, [100.0, 200.0])
        # 关键：不得出现 150 —— 那是两个平台的合并中位数。
        self.assertNotIn(150.0, values)

    def test_multi_platform_labels_name_the_platform(self):
        """数据里真有两个平台时，标签必须写出是哪一个。

        只隔离分组而不写名称，读者看得到两根柱子却分不清哪根是谁的。
        """
        plan = plot_p99.build_plan([
            row(run_id="a", platform="ascend", ttft="100"),
            row(run_id="b", platform="mthreads", ttft="200"),
        ])
        self.assertTrue(plan.multi_platform)
        self.assertEqual(
            sorted(plan.labels),
            ["short_chat/baseline@ascend", "short_chat/baseline@mthreads"],
        )

    def test_single_platform_keeps_labels_short(self):
        """只有一个平台时标签不加后缀——重复同一句是噪音，分组正确性不受影响。"""
        plan = plot_p99.build_plan([
            row(run_id="a", platform="ascend"),
            row(run_id="b", platform="ascend"),
        ])
        self.assertFalse(plan.multi_platform)
        self.assertEqual(plan.labels, ["short_chat/baseline"])

    def test_missing_platform_column_does_not_split(self):
        """缺这一列（旧产物）时不得把同一组拆开——缺证据不等于"换平台了"。"""
        rows = [
            {"run_id": "a", "workload": "w", "config": "c",
             "ttft_p99": "100", "tpot_p99": "1", "stall_p99": "1",
             "data_label": "MEASURED"},
            {"run_id": "a", "workload": "w", "config": "c",
             "ttft_p99": "100", "tpot_p99": "1", "stall_p99": "1",
             "data_label": "MEASURED"},
        ]
        plan = plot_p99.build_plan(rows)
        self.assertEqual(plan.labels, ["w/c"])
        self.assertEqual(plan.panels["ttft"], [100.0])

    def test_truncation_never_creates_duplicate_labels(self):
        """截断后不许重名。

        加了平台后缀后标签普遍变长，截断到同一前缀的可能性大幅上升；一旦重名，
        两根不同的柱子会顶着同一个名字，读者会把它们当成同一组。
        """
        long_workload = "long_context_qa_repeat_heavy"
        plan = plot_p99.build_plan([
            row(run_id="a", workload=long_workload, platform="ascend"),
            row(run_id="b", workload=long_workload, platform="mthreads"),
        ])
        display = plan.display_labels
        self.assertEqual(len(display), 2)
        self.assertEqual(len(set(display)), 2, f"截断后重名：{display}")
        # 显示标签与真实标签必须一一对应
        self.assertEqual(
            dict(zip(display, plan.labels)).keys(), set(display)
        )

    def test_display_labels_are_truncated_when_unique(self):
        plan = plot_p99.build_plan([
            row(run_id="a", workload="w", config="c"),
        ])
        self.assertEqual(plan.display_labels, plan.labels)


class TestFigureTextIsAscii(unittest.TestCase):
    """图内文字必须全 ASCII。

    默认字体（DejaVu Sans）没有中文字形，中文只会渲染成豆腐块——读者会以为
    图坏了。中文解释归 .source.txt 与终端输出（UTF-8 文本，不经字体）。
    """

    def test_suptitle_is_ascii_for_every_platform(self):
        """标题按数据动态生成，因此每个平台的取值都要单独验一遍。"""
        for platform in ("unknown", "ascend", "mthreads"):
            with self.subTest(platform=platform):
                plan = plot_p99.build_plan([row(platform=platform)])
                offenders = [c for c in plot_p99.suptitle(plan) if ord(c) > 127]
                self.assertEqual(
                    offenders, [],
                    f"图内标题含非 ASCII 字符会被渲染成豆腐块：{offenders!r}",
                )

    def test_suptitle_names_the_actual_platform_not_a_hardcoded_one(self):
        """写死平台名的标题会在数据换平台后继续印老名字。

        那句错话不影响观感，只会在有人照着它复现时把人指到错误的机器上。
        """
        plan = plot_p99.build_plan([row(platform="mthreads")])
        title = plot_p99.suptitle(plan)
        self.assertIn("mthreads", title)
        self.assertNotIn("Ascend", title)

    def test_suptitle_reports_backend_and_precision(self):
        plan = plot_p99.build_plan([
            row(platform="ascend", backend="real", precision="fp16"),
        ])
        title = plot_p99.suptitle(plan)
        self.assertIn("real", title)
        self.assertIn("fp16", title)

    def test_metric_titles_are_ascii(self):
        for metric, title in plot_p99._METRIC_TITLES.items():
            self.assertTrue(
                all(ord(c) < 128 for c in title),
                f"子图标题 {metric}={title!r} 含非 ASCII 字符",
            )


if __name__ == "__main__":
    unittest.main()
