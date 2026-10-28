"""`scripts/eval_lookup_deployment_path.py` 的用例（第 3 步）。

重点钉住三类**会静默出错**的地方：
1. **键格式与部署实现同构** —— 导出器写 `b{batch}_pf{pf}_kv{kv}`，
   `cost_model.predict(mode='lookup')` 读同一拼法；任一处漂移 ⇒ 查表恒未命中
   （回退率 100%）而**不报错**。
2. **评估不得用模型自带的 lookup_table**（它建在全部记录上 ⇒ 自匹配 ⇒ 覆盖恒 100%）。
   实测踩到：第一版脚本四处上下文全 100%，连"外推"都是 100%。
3. **跨会话留出必须被识别为"换会话、没换形状"** —— 否则 100% 会被读成泛化好。

运行器（**两个都要能跑**）
--------------------------
本仓库的**主**回归入口是 `python -m unittest discover`。本文件原先用 pytest 风格
（`pytest.approx` / `pytest.skip` / `tmp_path`），于是：
  * 缺 pytest 时被 discover 报成 `unittest.loader._FailedTest` ——
    **"收集失败"不是"用例失败"**，14 条断言实际**从未**进过默认回归
    （实测 2026-09-27：全量回归 `errors=1`，定位即本文件）；
  * 装了 pytest 后 unittest 仍**收不到**这些类 ⇒ `Ran 0 tests`，
    比报 error 更危险（看起来"跑了"，实际一条没跑）。
⇒ 现在改为**双运行器**：类继承 `unittest.TestCase`，`approx` 用本地 shim，
   `skip` / `tmp_path` 用 stdlib 等价物。两个入口都能真跑。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import eval_lookup_deployment_path as E  # noqa: E402
from ascend_opt.cost_model import CostModel  # noqa: E402

#: 桶配置逐字照抄 model_v7.json（kv 桶 256、pf 无桶）。
_BUCKET_V7 = {"pf_step": 0, "kv_step": 256}


class _Approx:
    """`pytest.approx` 的最小替代（只用到了 `==` 比较，不需要 full pytest）。

    ⚠️ 不要用 `round(a, n) == round(b, n)` 代替：那对"接近但不相等"会**悄悄判过**，
    正是本项目最怕的形态。这里显式给相对容差。
    """

    def __init__(self, expected: float, rel: float = 1e-6, abs_: float = 1e-9) -> None:
        self.expected = float(expected)
        self.rel = rel
        self.abs = abs_

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        try:
            o = float(other)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return NotImplemented
        return abs(o - self.expected) <= max(
            self.abs, self.rel * max(abs(o), abs(self.expected)))


def approx(expected: float, **kw) -> _Approx:
    return _Approx(expected, **kw)


def _rec(batch, pf, kv, ms, key=None, session="s1", nf=1):
    return {"session": session, "key": key or f"b{batch}_pf{pf}_kv{kv}_nf{nf}",
            "batch": batch, "prefill_tok": pf, "kv_len": kv, "n_forwards": nf,
            "actual_ms": ms, "actual_p90_ms": ms, "block_a_ms": ms}


@contextmanager
def _tmp_dir():
    """`tmp_path` fixture 的 stdlib 等价物。"""
    d = tempfile.mkdtemp(prefix="eval_lookup_test_")
    try:
        yield Path(d)
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


class TestKeyFormatMatchesDeployment(unittest.TestCase):
    """键必须与 `cost_model.predict` 的 lookup 分支**逐字同构**。"""

    def test_bucketing_matches_cost_model(self):
        """用真实 `CostModel.predict` 反证：本脚本造的键若能命中，说明格式一致。"""
        rec = _rec(batch=4, pf=512, kv=1600, ms=100.0)   # kv 1600 → 桶 1536
        tbl = {E.build_lookup_key(rec, _BUCKET_V7): {"max": 123.0, "median": 120.0, "n": 1}}
        payload = {"coefficients": {}, "calibrated_range": {},
                   "safety_margin": 0.0, "default_mode": "lookup",
                   "lookup_bucket": _BUCKET_V7, "lookup_table": tbl}
        cm = CostModel(payload=payload)
        pred = cm.predict(batch=4, prefill_tok=512, kv_len=1600, mode="lookup",
                          apply_margin=False)
        # 键一致 ⇒ 必须命中（若本脚本的桶算法与 predict 不同，这里会是 None）
        self.assertIsNotNone(
            pred, "键格式与部署 predict 不一致 ⇒ 查表恒未命中（静默失效）")
        self.assertEqual(pred.total_ms, approx(123.0))

    def test_kv_bucket_rounds_down(self):
        """桶 = **向下取整**（floor），与 `cost_model.predict` 的 `//` 同构。

        ⚠️ 桶边界要按 256 的整数倍算：1600→1536、1536→1536、1535→**1280**。
        写用例时若凭"1535≈1536"直觉写，会得到一个**实现正确、用例错**的假失败。
        """
        self.assertEqual(E.build_lookup_key(_rec(1, 0, 1600, 50.0), _BUCKET_V7), "b1_pf0_kv1536")
        self.assertEqual(E.build_lookup_key(_rec(1, 0, 1536, 50.0), _BUCKET_V7), "b1_pf0_kv1536")
        self.assertEqual(E.build_lookup_key(_rec(1, 0, 1535, 50.0), _BUCKET_V7), "b1_pf0_kv1280")

    def test_pf_step_zero_means_exact(self):
        self.assertEqual(E.build_lookup_key(_rec(2, 333, 512, 50.0), _BUCKET_V7), "b2_pf333_kv512")

    def test_no_bucket_means_exact_key(self):
        self.assertEqual(E.build_lookup_key(_rec(2, 333, 777, 50.0), None), "b2_pf333_kv777")


class TestNoSelfMatchLeakage(unittest.TestCase):
    """评估**不得**用模型自带的 lookup_table。"""

    def test_refit_uses_only_train(self):
        """只在训练划分上建的键，留出划分的新形状必须**未命中**。"""
        train = [_rec(1, 0, 512, 50.0, key="a")]
        holdout = [_rec(1, 0, 1024, 90.0, key="b")]      # 新 kv 形状
        tbl = E.refit_lookup_table(train, _BUCKET_V7)
        self.assertIn("b1_pf0_kv512", tbl)
        self.assertNotIn("b1_pf0_kv1024", tbl)

    def test_predictor_returns_none_for_unseen_shape(self):
        """部署路径预测器对**未在训练集出现**的形状必须返回 `None`（回退），不是 0。"""
        train = [_rec(1, 0, 512, 50.0, key="a")]
        base = CostModel(payload={"coefficients": {}, "calibrated_range": {},
                                  "safety_margin": 0.0, "default_mode": "lookup",
                                  "lookup_bucket": _BUCKET_V7, "lookup_table": {}})
        pred = E.make_predictor(base, train, _BUCKET_V7, apply_margin=True)
        self.assertIsNotNone(pred(_rec(1, 0, 512, 50.0)))   # 见过
        self.assertIsNone(pred(_rec(1, 0, 1024, 90.0)))     # 没见过 ⇒ 回退

    def test_measure_counts_miss_as_fallback_not_zero(self):
        recs = [_rec(1, 0, 512, 50.0, key="a"), _rec(1, 0, 1024, 90.0, key="b")]
        m = E.measure(recs, lambda r: 60.0 if r["kv_len"] == 512 else None)
        self.assertEqual(m["n_hit"], 1)
        self.assertEqual(m["n_miss_fallback"], 1)
        self.assertEqual(m["coverage"], approx(0.5))
        self.assertEqual(m["fallback_rate"], approx(0.5))
        # 未命中**不得**被算作一个 0 误差样本
        self.assertIsNotNone(m["abs_rel_p50"])

    def test_lookup_max_value_never_underestimates_within_key(self):
        """同一键的多条记录放进去，取 max ⇒ 余量前不该出现低估。"""
        train = [_rec(1, 0, 512, 50.0, key="a1"), _rec(1, 0, 512, 80.0, key="a2")]
        tbl = E.refit_lookup_table(train, _BUCKET_V7)
        self.assertEqual(tbl["b1_pf0_kv512"]["max"], approx(80.0))


class TestMarginAccounting(unittest.TestCase):
    """余量前后必须分开报 —— 混在一起说不清余量贡献了多少。"""

    def test_margin_increases_prediction(self):
        train = [_rec(1, 0, 512, 100.0, key="a")]
        base = CostModel(payload={"coefficients": {}, "calibrated_range": {},
                                  "safety_margin": 0.20, "default_mode": "lookup",
                                  "lookup_bucket": _BUCKET_V7, "lookup_table": {}})
        p_no = E.make_predictor(base, train, _BUCKET_V7, apply_margin=False)
        p_yes = E.make_predictor(base, train, _BUCKET_V7, apply_margin=True)
        r = _rec(1, 0, 512, 100.0)
        self.assertEqual(p_no(r), approx(100.0))
        self.assertEqual(p_yes(r), approx(120.0))

    def test_measure_reports_both_sides(self):
        recs = [_rec(1, 0, 512, 100.0, key="a"), _rec(1, 0, 512, 50.0, key="b")]
        # 预测恒 100 ⇒ 对 50 高估 100%、对 100 无误差
        m = E.measure(recs, lambda r: 100.0)
        self.assertTrue(m["underest_p95"] is None or m["underest_p95"] == 0.0)
        self.assertEqual(m["overest_p95"], approx(1.0))


class TestCrossSessionOverlapDiagnostic(unittest.TestCase):
    """跨会话留出若只是"同一网格跑两遍"，必须被点名。"""

    def test_identical_shape_sets_flagged(self):
        recs = [_rec(1, 0, 512, 50.0, key="a", session="s1"),
                _rec(1, 0, 512, 52.0, key="b", session="s2")]
        d = E._cross_session_shape_overlap(recs, _BUCKET_V7)
        self.assertIs(d["applicable"], True)
        self.assertEqual(d["n_intersect"], 1)
        self.assertEqual(d["intersect_frac"], approx(1.0))

    def test_disjoint_shape_sets_not_flagged(self):
        recs = [_rec(1, 0, 512, 50.0, key="a", session="s1"),
                _rec(1, 0, 1024, 90.0, key="b", session="s2")]
        d = E._cross_session_shape_overlap(recs, _BUCKET_V7)
        self.assertEqual(d["n_intersect"], 0)

    def test_single_session_inapplicable(self):
        recs = [_rec(1, 0, 512, 50.0, key="a", session="s1")]
        d = E._cross_session_shape_overlap(recs, _BUCKET_V7)
        self.assertIs(d["applicable"], False)


class TestRealArtifactShape(unittest.TestCase):
    """用**真实产物形状**跑通端到端（防"按想象写解析"）。"""

    def test_end_to_end_on_real_v7(self):
        ev_path = _ROOT / "results" / "processed" / "costcalib_eval_v7.json"
        model_path = _ROOT / "configs" / "cost_calib" / "model_v7.json"
        if not ev_path.exists() or not model_path.exists():
            self.skipTest("缺少真实产物（离线环境）")
        with _tmp_dir() as tmp:
            out = tmp / "out.json"
            rc = E.main(["--eval", str(ev_path), "--model", str(model_path),
                         "--diagnose", "--json", str(out)])
            self.assertEqual(rc, 0)
            d = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(d["default_mode"], "lookup")
        # ★ 判据必须落在 ① 按形状留出（唯一真的留出形状的上下文）
        self.assertTrue(any("按形状留出" in k for k in d["contexts"]))
        c1 = [v for k, v in d["contexts"].items() if "按形状留出" in k][0]
        hm = c1["holdout_with_margin"]
        # 覆盖率**不得**是 100%（那是自匹配的标志）
        self.assertLess(hm["coverage"], 1.0,
                        "① 上下文的部署路径覆盖率 = 100% ⇒ 疑似用了模型自带的 "
                        "lookup_table（自匹配）")
        # 未命中必须计入回退
        self.assertGreater(hm["n_miss_fallback"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
