"""`_json_safe` 与模型文件的边界标注用例（第 14 步技术债）。"""
from __future__ import annotations
import json, sys, unittest
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from ascend_opt.cost_model import CostModel
from ascend_opt.telemetry import _json_safe


class TestJsonSafe(unittest.TestCase):
    def test_normalises_non_finite_floats(self):
        s = _json_safe({"a": 1.0, "b": float("inf"), "c": [float("nan"), {"d": float("-inf")}]})
        self.assertEqual(s["b"], None)
        self.assertEqual(s["c"], [None, {"d": None}])
        json.dumps(s, allow_nan=False)          # ★ 必须是标准 JSON

    def test_raw_payload_would_fail(self):
        """★ 反证：不标准化时会**真的失败** —— 否则这条保护就没意义。"""
        with self.assertRaises(ValueError):
            json.dumps({"b": float("inf")}, allow_nan=False)

    def test_keeps_finite_and_other_types(self):
        s = _json_safe({"a": 1.5, "b": "x", "c": None, "d": (1, 2)})
        self.assertEqual(s, {"a": 1.5, "b": "x", "c": None, "d": [1, 2]})


class TestModelBoundaryAnnotation(unittest.TestCase):
    def test_model_file_carries_boundary_and_caveat(self):
        """模型**本体**要带适用边界与精度告诫（不然只有文档里有）。"""
        m = CostModel.load_offline(_ROOT / "configs" / "cost_calib" / "model_v1.json")
        self.assertIn("applicability_boundary", m.payload)
        self.assertIn("accuracy_caveat", m.payload)
        self.assertEqual(m.payload["applicability_boundary"]["kv_len_max"], 2048)
        # 告诫里必须点明"不是泛化精度"与"余量站不住"
        blob = json.dumps(m.payload["accuracy_caveat"], ensure_ascii=False)
        self.assertIn("不是泛化精度", blob)
        self.assertIn("余量", blob)

    def test_boundary_still_loads_and_predicts(self):
        m = CostModel.load_offline(_ROOT / "configs" / "cost_calib" / "model_v1.json")
        self.assertIsNotNone(m.predict(batch=4, prefill_tok=512, kv_len=1024))
        self.assertIsNone(m.predict(batch=3, prefill_tok=0, kv_len=1024),
                          "batch=3 不在网格里 ⇒ 必须回退（None）")


if __name__ == "__main__":
    unittest.main()
