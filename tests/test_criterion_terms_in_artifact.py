"""判据口径的**三件必须进产物** + 消融开关真的接上了（第 14/15 步）。

为什么要有这一条
----------------
第 14 步的主线是"判据从**前向 ≤ 预算**改成**整步 ≤ 预算**"（前向 104.8 + host 25.3
= 整步 130.5 ms @b8）。但 `ControllerConfig.describe()` 原先**没有**记
`host_overhead_ms` / `host_overhead_fit` ⇒ **产物里看不出某次运行到底有没有把 host 项
算进判据**。实测：`S14B1`（无 host 项）与 `S14D`（有 host 项）的 `config.controller`
在这一点上完全一样，只能靠仓库配置 + `git_commit` 反推（可追溯，但**不是产物自证**）。

这与当年 `signal_criterion` 漏记是同一个形状（"冻结/落地的决定没进产物"），
所以按同一条纪律办：**凡是会改变判据的字段，必须进 `describe()`**。

第 15 步消融"去掉误差裕量"要求这件事能**由配置**表达（不许靠改模型文件 ——
那会造出同一份标定的第二份真相）。本文件同时证明那条开关真的从配置走到了模型调用。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.controller import ControllerConfig  # noqa: E402
from ascend_opt.cost_model import CostModel  # noqa: E402
from ascend_opt.cost_policy import CostPolicyConfig  # noqa: E402
from run_benchmark import load_config  # noqa: E402

#: 判据口径的组成部分 —— 缺任一项，产物就无法自证"这次跑的是哪种口径"。
_CRITERION_TERMS = ("host_overhead_ms", "host_overhead_fit", "apply_safety_margin")


class TestDescribeRecordsCriterionTerms(unittest.TestCase):
    def test_default_describe_has_all_terms(self):
        d = ControllerConfig().describe()
        for k in _CRITERION_TERMS:
            self.assertIn(k, d, f"describe() 少了 {k} ⇒ 产物看不出判据口径")

    def test_describe_records_the_actual_values(self):
        c = ControllerConfig(host_overhead_ms={2: 3.06, 8: 25.29},
                             host_overhead_fit=[3.705, -4.35],
                             apply_safety_margin=False)
        d = c.describe()
        self.assertEqual(d["host_overhead_ms"], {2: 3.06, 8: 25.29})
        self.assertEqual(d["host_overhead_fit"], [3.705, -4.35])
        self.assertIs(d["apply_safety_margin"], False)

    def test_absent_host_term_is_recorded_as_none_not_zero(self):
        """**不可测/未设**必须留在 `None`，不许填 0（0 是"host 开销真的是 0"这个断言）。"""
        d = ControllerConfig().describe()
        self.assertIsNone(d["host_overhead_ms"])
        self.assertIsNone(d["host_overhead_fit"])


class TestMarginSwitchReachesTheModel(unittest.TestCase):
    """`apply_safety_margin=false` 必须真的改变预测值（否则消融测的是空气）。"""

    MODEL = _ROOT / "configs" / "cost_calib" / "model_v4.json"

    def test_predict_with_and_without_margin_differ(self):
        m = CostModel.load_offline(self.MODEL)
        margin = float(m.payload.get("safety_margin") or 0.0)
        self.assertGreater(margin, 0.0, "该模型没有安全余量 ⇒ 本用例证明不了任何事")
        common = dict(batch=2, prefill_tok=0, kv_len=1024)
        with_m = m.predict(**common, apply_margin=True)
        without = m.predict(**common, apply_margin=False)
        self.assertIsNotNone(with_m)
        self.assertIsNotNone(without)
        self.assertAlmostEqual(without.total_ms, with_m.total_ms / (1.0 + margin), places=6)

    def _cfg_with_model(self, tmp: Path, *, margin_off: bool = False):
        """拿真实配置当底子，只把模型路径换成本地存在的那份（其余逐字不动）。

        ⚠️ 插入点用**锚点断言**：锚点变了就让用例失败，而不是"插不进去、于是悄悄
        测的是默认值"（那正是"守卫恒真"的成因之一）。
        """
        src = (_ROOT / "configs" / "mthreads" / "phase1_step15" /
               "fixed_cost_full.yaml").read_text(encoding="utf-8")
        src = src.replace("cost_model_path: configs/cost_calib/model_v6.json",
                          "cost_model_path: configs/cost_calib/model_v4.json")
        src = src.replace("cost_model_fingerprint_source: observed",
                          "cost_model_fingerprint_source: declared")
        if margin_off:
            anchor = "  signal_criterion: decode_p90\n"
            self.assertIn(anchor, src, "锚点不在 ⇒ 本用例会静默失效，必须更新")
            src = src.replace(anchor, anchor + "  apply_safety_margin: false\n")
        p = tmp / "cfg.yaml"
        p.write_text(src, encoding="utf-8", newline="\n")
        return load_config(p)

    def test_wiring_default_is_on(self):
        from arm_factory import build_arm
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg_with_model(Path(td))
            self.assertTrue(cfg.controller.apply_safety_margin, "配置层默认应为 True")
            policy = build_arm(cfg)
        self.assertIsNotNone(policy, "没构造出 cost 臂")
        self.assertTrue(policy.config.apply_safety_margin,
                        "默认应**乘**余量（否则既有批次的判据会静默变松）")

    def test_wiring_off_is_honoured(self):
        from arm_factory import build_arm
        with tempfile.TemporaryDirectory() as td:
            cfg = self._cfg_with_model(Path(td), margin_off=True)
            self.assertFalse(cfg.controller.apply_safety_margin, "配置没读进去 ⇒ 用例前提不成立")
            policy = build_arm(cfg)
        self.assertFalse(policy.config.apply_safety_margin,
                         "配置里关了余量，策略层却还在乘 ⇒ 第 15 步消融测不到东西")


class TestCostPolicyConfigDefault(unittest.TestCase):
    def test_default_keeps_margin_on(self):
        self.assertTrue(CostPolicyConfig().apply_safety_margin)


if __name__ == "__main__":
    unittest.main()
