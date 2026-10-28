"""`plot_s10_surface.build_plan` 的用例（纯函数，不需要 matplotlib）。

为什么单独测 `build_plan`：它是"图上会出现什么"的唯一决定者。
本仓库已经用真事付过一次学费（`plot_p99.py` 开头记的）：绘图逻辑内嵌、
不可测 ⇒ "空分位兜底成 0"与"跨负载混成一个中位数"两个缺陷一路静默存活到评审。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_s10_surface import build_plan  # noqa: E402

PLAN = {
    "tag": "S10",
    "segments": ["low", "high", "recover"],
    "cells": ["c_a", "c_b", "best_static"],
    "metrics": [
        {"key": "goodput_tok_s", "label": "goodput", "lens": "window"},
        {"key": "ttft_p50_ms", "label": "TTFT p50", "lens": "cohort"},
    ],
    "per_cell_metrics": {
        "c_a": {
            "low": {"goodput_tok_s": {"median": 10.0, "n": 3},
                    "ttft_p50_ms": {"median": 500.0, "n": 3}},
            "high": {"goodput_tok_s": {"median": 5.0, "n": 3},
                     "ttft_p50_ms": {"median": 36000.0, "n": 3}},
            "recover": {"goodput_tok_s": {"median": 8.0, "n": 3},
                        "ttft_p50_ms": {"median": 600.0, "n": 3}},
        },
        "c_b": {
            "low": {"goodput_tok_s": {"median": 9.0, "n": 3},
                    "ttft_p50_ms": {"median": None, "n": 0}},      # ← 缺
            "high": {"goodput_tok_s": {"median": 6.0, "n": 3},
                     "ttft_p50_ms": {"median": 30000.0, "n": 3}},
            "recover": {"goodput_tok_s": {"median": None, "n": 0},  # ← 缺
                        "ttft_p50_ms": {"median": 700.0, "n": 3}},
        },
    },
    "best_static": "c_a",
    "per_run": [{"run": "S10-c_a"}, {"run": "S10-c_b"}],
}


class TestBuildPlan(unittest.TestCase):
    def setUp(self):
        self.plan = build_plan(PLAN)

    def test_plan_is_pure_and_needs_no_matplotlib(self):
        """能在只有纯 Python 的环境下算出计划（本用例本身就没 import matplotlib）。"""
        self.assertEqual(self.plan.segments, ["low", "high", "recover"])
        self.assertEqual(self.plan.cells, ["c_a", "c_b"])       # best_static 不是候选
        self.assertEqual(len(self.plan.panels), 2)
        self.assertEqual(self.plan.best_static, "c_a")

    def test_each_panel_keeps_its_lens(self):
        """口径必须**跟着 panel 走**：吞吐 window、延迟 cohort，不许混。"""
        self.assertEqual([p.lens for p in self.plan.panels], ["window", "cohort"])

    def test_missing_values_are_absent_not_zero(self):
        """★ 缺口**缺席**，不填 0。

        填 0 会在图上画出一根"好得不可能的测量值"的柱子 —— 比缺一根柱子更容易误导。
        """
        gp = next(p for p in self.plan.panels if p.key == "goodput_tok_s")
        self.assertNotIn("recover", gp.values["c_b"], "缺的段不该出现在 values 里")
        self.assertIsNone(gp.values["c_b"].get("recover"))
        self.assertTrue(any("c_b/recover" in m for m in gp.missing), gp.missing)

        ttft = next(p for p in self.plan.panels if p.key == "ttft_p50_ms")
        self.assertNotIn("low", ttft.values["c_b"])

    def test_present_values_survive(self):
        gp = next(p for p in self.plan.panels if p.key == "goodput_tok_s")
        self.assertAlmostEqual(gp.values["c_a"]["high"], 5.0)
        self.assertAlmostEqual(gp.values["c_b"]["high"], 6.0)

    def test_run_ids_are_collected_for_the_source_file(self):
        self.assertEqual(self.plan.run_ids, ["S10-c_a", "S10-c_b"])

    def test_empty_plan_does_not_crash(self):
        """空计划也要能构造（图会空着，但不该抛）。"""
        p = build_plan({})
        self.assertEqual(p.panels, [])
        self.assertEqual(p.cells, [])
        self.assertTrue(p.suptitle)


if __name__ == "__main__":
    unittest.main()
