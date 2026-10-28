"""`ascend_opt.run_ids` 的用例 —— **唯一**的 run_id/臂名解析。

为什么值得单独测
----------------
2026-09-27 的 D1① 假失败，根因就是"臂名解析有第二/第三份实现"：
闸门用字符串替换得到 `fixed_cost_pilot`，诊断工具用正则得到 `cost_pilot`
⇒ 闸门在自己字典里查不到数据，报出两条**假失败**。
所以这里第一条用例就钉住"**用哪个解析**"，第二条钉住"解析不出来就 `None`，
**不许猜**"。
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.run_ids import arm_of, normalize_arm, parse_run_id  # noqa: E402

#: 这些是**真实存在过的目录名/run_id**，逐字抄自 `results/raw/`（不是想象出来的样本）。
REAL_RUN_IDS = {
    "S14G-short_chat-fixed_cost_pilot": ("S14G", "short_chat", "fixed", "cost_pilot", None),
    "S14G-short_chat-fixed_cost_pilot-r1": ("S14G", "short_chat", "fixed", "cost_pilot", 1),
    "S14G-short_chat-fixed_cost_min512-r2": ("S14G", "short_chat", "fixed", "cost_min512", 2),
    "S14B1-short_chat-fixed_slo_weave-r4": ("S14B1", "short_chat", "fixed", "slo_weave", 4),
    "P1S3-phase1_anchor-static_chunk_c4": ("P1S3", "phase1_anchor", "static", "chunk_c4", None),
    "P1S3D-phase1_anchor-adaptive_c1-r2": ("P1S3D", "phase1_anchor", "adaptive", "c1", 2),
    "S10-load_ramp-ramp_static_c8_ck512": ("S10", "load_ramp", "ramp", "static_c8_ck512", None),
    "P8FULL-load_ramp-ramp_adaptive_c8": ("P8FULL", "load_ramp", "ramp", "adaptive_c8", None),
}


class TestParseRealNames(unittest.TestCase):
    def test_real_run_ids_resolve_as_expected(self):
        for rid, exp in REAL_RUN_IDS.items():
            with self.subTest(rid=rid):
                r = parse_run_id(rid)
                self.assertIsNotNone(r, f"真实 run_id 解析失败：{rid}")
                self.assertEqual((r.tag, r.workload, r.cell, r.arm, r.pass_index), exp)

    def test_the_regression_that_caused_two_false_failures(self):
        """★ 钉住 2026-09-27 那次假失败：旧写法得到的是 **cell+arm**，不是臂名。"""
        rid = "S14G-short_chat-fixed_cost_pilot-r1"
        old = re.sub(r"-r\d+$", "", rid.split("-", 1)[1]).replace("short_chat-", "")
        self.assertEqual(old, "fixed_cost_pilot", "旧写法的行为应被如实记录")
        self.assertEqual(arm_of(rid), "cost_pilot", "新实现必须给臂名，而不是 cell+arm")
        self.assertNotEqual(old, arm_of(rid), "两者必须不同——这正是那次假失败的根因")

    def test_group_key_collapses_repeats(self):
        keys = {parse_run_id(f"S14G-short_chat-fixed_cost_pilot{f}").group_key
                for f in ("", "-r1", "-r2")}
        self.assertEqual(keys, {"S14G-short_chat-fixed_cost_pilot"},
                         "同一格的重复必须共用一个分组键")

    def test_pass_index_none_when_no_suffix(self):
        self.assertIsNone(parse_run_id("S14G-short_chat-fixed_cost_pilot").pass_index)


class TestParseFailuresAreNotGuessed(unittest.TestCase):
    """解析失败必须显式 `None`，**不得**回退成任何猜测值。"""

    def test_unparsable_returns_none(self):
        for bad in ("", "nonsense", "S14G-short_chat-inB", "20260925-225848-short_chat-inB"):
            with self.subTest(bad=bad):
                r = parse_run_id(bad)
                if bad == "20260925-225848-short_chat-inB":
                    # 这一条是**语法成立、语义无意义**（时间戳当 tag）—— 如实记录边界：
                    # 解析"成功"但臂名是垃圾 ⇒ 调用方必须做交叉核对，不能只看"解析出来了没"。
                    self.assertIsNotNone(r)
                    self.assertEqual(r.tag, "20260925")
                else:
                    self.assertIsNone(r, f"应判解析失败：{bad!r}")

    def test_arm_of_none_on_failure(self):
        self.assertIsNone(arm_of("nonsense"))

    def test_normalize_arm_has_explicit_fallback(self):
        """兜底函数只用于"拿不到合法 run_id"，且对合法输入必须与权威一致。"""
        self.assertEqual(normalize_arm("S14G-short_chat-fixed_cost_pilot-r1"), "cost_pilot")
        self.assertEqual(normalize_arm("weird-name"), "name")  # 兜底：去掉第一段


if __name__ == "__main__":
    unittest.main()
