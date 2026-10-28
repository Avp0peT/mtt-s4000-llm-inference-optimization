"""`pick_d1_min` 的四条分支都要有用例 —— 否则"预注册规则"只是注释。

规则在 `EXPERIMENTS.md` 的「D1② · 选取规则（先冻结）」。本文件的角色是证明
**每条分支都可达、且走对了**（含"缺项/无区分度"这两个最容易写错的保守分支）。

⚠️ 特别钉住两条：
* **三点 D1 全为 0 ⇒ 是「无区分度」，不是「都没改善」**，也不是「都一样」
  ⇒ 必须转 D2，而不是直接判"小候选无用"；
* **缺项（`None`）不得当 0**（填 0 会让"数不出来"伪装成"测出来是 0"）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pick_d1_min import pick_min  # noqa: E402


def _d(v512, v256, v128):
    return {"512": v512, "256": v256, "128": v128}


class TestMainRule(unittest.TestCase):
    def test_monotone_and_margin_met_picks_128(self):
        got, why = pick_min(_d(0.60, 0.35, 0.30), _d(0.2, 0.1, 0.05))
        self.assertEqual(got, 128)
        self.assertIn("主规则", why)

    def test_margin_met_but_not_monotone_picks_256(self):
        got, why = pick_min(_d(0.60, 0.65, 0.30), _d(0.2, 0.1, 0.05))
        self.assertEqual(got, 256)
        self.assertIn("非单调", why)

    def test_margin_not_met_keeps_512_and_says_reattribute(self):
        got, why = pick_min(_d(0.60, 0.58, 0.50), _d(0.2, 0.1, 0.05))
        self.assertEqual(got, 512)
        self.assertIn("重新归因", why)
        self.assertIn("不得", why, "否证分支必须写明'不得把 D1 变化归给候选集'")

    def test_reverse_direction_keeps_512(self):
        got, _ = pick_min(_d(0.30, 0.40, 0.55), _d(0.2, 0.1, 0.05))
        self.assertEqual(got, 512)


class TestNoDiscriminationBranch(unittest.TestCase):
    def test_all_zero_d1_falls_back_to_d2(self):
        got, why = pick_min(_d(0.0, 0.0, 0.0), _d(0.30, 0.20, 0.10))
        self.assertEqual(got, 256)
        self.assertIn("无区分度", why)
        self.assertIn("转 D2", why)

    def test_all_zero_d1_and_flat_d2_is_undecided(self):
        got, why = pick_min(_d(0.0, 0.0, 0.0), _d(0.20, 0.20, 0.20))
        self.assertEqual(got, 512)
        self.assertIn("未判", why)
        # ⚠️ 断言要精确：`why` 里确实会出现"小候选无用"这四个字，但它是**禁令**的一部分
        # （"「未判」登记，**不得写成**'小候选无用'"）⇒ 只能断言禁令在、不能断言字面不出现。
        self.assertIn("不得写成", why, "必须显式禁止把'无区分度'写成'小候选无用'")

    def test_all_zero_d1_and_missing_d2_is_undecided(self):
        got, why = pick_min(_d(0.0, 0.0, 0.0), _d(None, 0.2, 0.1))
        self.assertEqual(got, 512)
        self.assertIn("未判", why)


class TestMissingIsNotZero(unittest.TestCase):
    def test_any_missing_d1_is_undecided_not_zero(self):
        got, why = pick_min(_d(0.60, None, 0.30), _d(0.2, 0.1, 0.05))
        self.assertEqual(got, 512)
        self.assertIn("不可判", why)
        self.assertIn("不得把缺项当 0", why)

    def test_missing_does_not_accidentally_satisfy_the_rule(self):
        """若把 `None` 当 0，`D1(128)=None` 会被读成"降到 0"从而误选 128 —— 钉住这一点。"""
        got, _ = pick_min(_d(0.60, None, None), _d(0.2, 0.1, 0.05))
        self.assertNotEqual(got, 128)


if __name__ == "__main__":
    unittest.main()
