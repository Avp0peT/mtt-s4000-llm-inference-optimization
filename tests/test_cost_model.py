"""成本模型的加载/指纹/范围用例（第 12 步的模型层）。

这些用例直接对应**验收里的几条**：
* "模型带硬件、软件、模型版本、精度和标定范围指纹，**失配不能静默使用**" ⇒ `TestFingerprint`；
* "范围外回退规则" ⇒ `TestRange`；
* "有保守回退但不声称硬 SLO 保证" ⇒ 余量从**低估尾部**推，见 `TestMargin`。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.cost_model import CostModel, CostModelError  # noqa: E402

MODEL = _ROOT / "configs" / "cost_calib" / "model_v1.json"

#: 一份"与标定相符"的 observed（值取自标定产物里的 `fingerprint.expected`）。
GOOD = {
    "hardware": {"device_model": "MTT S4000", "driver_version": "2.7.0"},
    "software": {"torch": "2.2.0", "torch_musa": "1.3.0+81caf0a",
                 "transformers": "4.46.3", "python": "3.10.8"},
    "model": {"id": "Qwen/Qwen2.5-7B-Instruct",
              "revision": "a09a35458c702b33eeacc393d103063234e8bc28"},
    "precision": "fp16",
}


def _mutate(path: Path, fn) -> Path:
    """写一份改过的模型文件（用于构造缺陷）。"""
    payload = json.loads(MODEL.read_text(encoding="utf-8"))
    fn(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8", newline="\n")
    return path


class TestLoadShippedModel(unittest.TestCase):
    def test_shipped_model_loads_with_matching_fingerprint(self):
        m = CostModel.load(MODEL, GOOD, expect_protocol="cost-calib-v1")
        self.assertEqual(m.describe()["model_version"], "cost-model-v1")
        self.assertEqual(m.relaxations, [], "指纹全确认相符时不该有放宽记录")

    def test_protocol_mismatch_is_rejected(self):
        """协议版本不符 ⇒ 拒（"模型过期"的一种）。"""
        with self.assertRaises(CostModelError) as cm:
            CostModel.load(MODEL, GOOD, expect_protocol="cost-calib-v2")
        self.assertIn("protocol_version", str(cm.exception))


class TestFingerprint(unittest.TestCase):
    def test_driver_version_mismatch_is_rejected_and_named(self):
        """★ 验收第 5 条：驱动换了还照用，就是"静默使用"。"""
        bad = dict(GOOD)
        bad["hardware"] = dict(GOOD["hardware"], driver_version="2.8.0")
        with self.assertRaises(CostModelError) as cm:
            CostModel.load(MODEL, bad)
        self.assertIn("driver_version", str(cm.exception))
        self.assertIn("2.8.0", str(cm.exception))

    def test_model_revision_mismatch_is_rejected(self):
        bad = dict(GOOD)
        bad["model"] = dict(GOOD["model"], revision="deadbeef")
        with self.assertRaises(CostModelError):
            CostModel.load(MODEL, bad)

    def test_precision_mismatch_is_rejected(self):
        bad = dict(GOOD, precision="bf16")
        with self.assertRaises(CostModelError):
            CostModel.load(MODEL, bad)

    def test_unmeasured_leaf_is_rejected_in_strict_mode(self):
        """★ "没确认"与"确认相符"是两件事 —— 默认把前者也拒掉。"""
        bad = dict(GOOD)
        bad["hardware"] = dict(GOOD["hardware"], driver_version=None)
        with self.assertRaises(CostModelError) as cm:
            CostModel.load(MODEL, bad, strict=True)
        self.assertIn("未确认", str(cm.exception))

    def test_unmeasured_leaf_is_allowed_but_recorded_in_non_strict(self):
        """放宽必须**显式**，而且留下记录 —— 不然就成了静默放宽。"""
        bad = dict(GOOD)
        bad["hardware"] = dict(GOOD["hardware"], driver_version=None)
        m = CostModel.load(MODEL, bad, strict=False)
        self.assertTrue(any("driver_version" in r for r in m.relaxations), m.relaxations)

    def test_non_strict_still_rejects_a_real_mismatch(self):
        """放宽的是"未测"，不是"不符" —— 后者任何模式下都拒。"""
        bad = dict(GOOD)
        bad["software"] = dict(GOOD["software"], torch="2.3.0")
        with self.assertRaises(CostModelError):
            CostModel.load(MODEL, bad, strict=False)

    def test_missing_expected_value_in_model_is_rejected(self):
        """模型里没写期望值 ⇒ 无从比对 ⇒ 拒（不能因为"没写"就放过）。"""
        with tempfile.TemporaryDirectory() as td:
            p = _mutate(Path(td) / "m.json",
                        lambda d: d["fingerprint"]["hardware"].pop("driver_version"))
            with self.assertRaises(CostModelError) as cm:
                CostModel.load(p, GOOD)
            self.assertIn("无法比对", str(cm.exception))


class TestRange(unittest.TestCase):
    def setUp(self):
        self.m = CostModel.load(MODEL, GOOD)

    def test_in_range_point_predicts(self):
        p = self.m.predict(batch=1, prefill_tok=0, kv_len=512)
        self.assertIsNotNone(p)
        self.assertGreater(p.per_forward_ms, 0)

    def test_unseen_batch_is_out_of_range(self):
        self.assertIsNone(self.m.predict(batch=16, prefill_tok=0, kv_len=512))

    def test_prefill_at_a_batch_that_had_no_prefill_data_is_out_of_range(self):
        """★ 网格里 `batch=2` 只有纯 decode ⇒ 它的 `prefill_tok` 范围是 `[0,0]`。

        这正是"训练集里没出现过的组合一律算越界"的价值：**没数据的地方不会被静默外推**。
        """
        self.assertIsNotNone(self.m.predict(batch=2, prefill_tok=0, kv_len=512))
        self.assertIsNone(self.m.predict(batch=2, prefill_tok=512, kv_len=512))

    def test_kv_beyond_calibration_is_out_of_range(self):
        self.assertIsNone(self.m.predict(batch=1, prefill_tok=0, kv_len=4096))

    def test_out_of_range_returns_none_not_a_guess(self):
        """★ 预注册 §六：越界必须回退。返回 `None` 就是"我算不了"，不是"我给个近似值"。"""
        self.assertIsNone(self.m.predict(batch=99, prefill_tok=0, kv_len=0))

    def test_negative_prediction_is_treated_as_unusable(self):
        """线性段外推出负成本时，返回 `None` 而不是一个负数。"""
        with tempfile.TemporaryDirectory() as td:
            def wreck(d):
                d["coefficients"]["1"] = [-1000.0, 0.0, 0.0]
            p = _mutate(Path(td) / "m.json", wreck)
            m = CostModel.load(p, GOOD)
            self.assertIsNone(m.predict(batch=1, prefill_tok=0, kv_len=512))


class TestMarginAndForwards(unittest.TestCase):
    def setUp(self):
        self.m = CostModel.load(MODEL, GOOD)

    def test_margin_matches_the_underestimation_tail_from_calibration(self):
        """余量必须等于标定时的**低估 p95**（1.7669%），不能另取一个数。"""
        p = self.m.predict(batch=1, prefill_tok=0, kv_len=512)
        self.assertAlmostEqual(p.margin, 0.01766943457486417, places=10)

    def test_margin_increases_the_prediction(self):
        with_m = self.m.predict(batch=1, prefill_tok=0, kv_len=512, apply_margin=True)
        without = self.m.predict(batch=1, prefill_tok=0, kv_len=512, apply_margin=False)
        self.assertAlmostEqual(with_m.per_forward_ms, without.per_forward_ms * (1.0 + with_m.margin),
                               places=6)

    def test_n_forwards_is_a_multiplier_not_part_of_the_range(self):
        """加和性有实测依据（n_forwards 1→4 单前向成本逐位相同）⇒ 它作乘子。"""
        one = self.m.predict(batch=1, prefill_tok=1024, kv_len=1024, n_forwards=1)
        four = self.m.predict(batch=1, prefill_tok=1024, kv_len=1024, n_forwards=4)
        self.assertAlmostEqual(four.per_forward_ms, one.per_forward_ms, places=9)
        self.assertAlmostEqual(four.total_ms, one.per_forward_ms * 4, places=6)
        # 且它**不参与**范围判断：给个很大的 n_forwards 仍然给得出预测
        self.assertIsNotNone(self.m.predict(batch=1, prefill_tok=0, kv_len=512, n_forwards=64))


if __name__ == "__main__":
    unittest.main()
