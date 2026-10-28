"""策略臂的**选择开关**与**溯源**用例（第 12/13 步）。

验收要求"shadow 仅证明预测/决策可记录，闭环才验证动作实际生效" ⇒
**产物必须能分辨二者**。这两条在实测里都踩到过：
* 加了 `kind`/`shadow` 却漏进 `describe()` ⇒ 两臂产物看起来一模一样；
* `kind: cost` + `enabled: False` ⇒ 成本臂一次都没被调用，而配置写着 kind=cost。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.controller import ControllerConfig  # noqa: E402


class TestArmSelectionGuards(unittest.TestCase):
    def test_default_is_threshold_and_not_shadow(self):
        """默认必须是阈值臂、且不 shadow ⇒ 既有批次逐字节不变。"""
        c = ControllerConfig()
        self.assertEqual(c.kind, "threshold")
        self.assertFalse(c.shadow)

    def test_unknown_kind_is_rejected_at_load(self):
        with self.assertRaises(ValueError):
            ControllerConfig(kind="nope")

    def test_non_threshold_kind_requires_enabled(self):
        """★ 静默陷阱：控制块在 `if enabled` 里 ⇒ `enabled=False` 时该臂根本不被调用。"""
        for kind in ("cost", "slo_weave"):
            with self.assertRaises(ValueError):
                ControllerConfig(kind=kind, enabled=False)

    def test_shadow_is_allowed_on_any_non_threshold_kind(self):
        for kind in ("cost", "slo_weave"):
            c = ControllerConfig(kind=kind, enabled=True, shadow=True)
            self.assertTrue(c.shadow)

    def test_c_max_must_be_positive(self):
        with self.assertRaises(ValueError):
            ControllerConfig(kind="slo_weave", enabled=True, slo_weave_c_max=0)


class TestArmTraceability(unittest.TestCase):
    def test_describe_carries_kind_and_shadow(self):
        """★ 产物必须能分辨"跑的是哪条臂"与"是不是 shadow"。"""
        d = ControllerConfig(kind="slo_weave", enabled=True, shadow=True).describe()
        self.assertEqual(d["kind"], "slo_weave")
        self.assertTrue(d["shadow"])
        self.assertEqual(d["slo_weave_c_max"], 4096)

    def test_describe_distinguishes_shadow_from_closed_loop(self):
        a = ControllerConfig(kind="cost", enabled=True, shadow=True).describe()
        b = ControllerConfig(kind="cost", enabled=True, shadow=False).describe()
        self.assertNotEqual(a["shadow"], b["shadow"], "shadow 与闭环必须在产物里分得开")

    def test_describe_distinguishes_threshold_from_the_other_arms(self):
        k = {ControllerConfig(kind=x, enabled=True).describe()["kind"]
             for x in ("threshold", "cost", "slo_weave")}
        self.assertEqual(k, {"threshold", "cost", "slo_weave"})


if __name__ == "__main__":
    unittest.main()
