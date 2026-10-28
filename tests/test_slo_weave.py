"""SLOWeave 参考策略的用例（第 13 步的最接近方法对照）。

断言逐条对应**论文里写明的东西**（Algorithm 1 / Prop. 1 / §3.3 的 δ / §4.3 的 C_max），
不是"实现成什么样就测什么样"。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.cost_model import CostModel  # noqa: E402
from ascend_opt.slo_weave import SLOWeaveConfig, SLOWeavePolicy  # noqa: E402

MODEL = _ROOT / "configs" / "cost_calib" / "model_v1.json"
KV = 1024


def policy(*, c_max: int = 4096, model: bool = True) -> SLOWeavePolicy:
    m = CostModel.load_offline(MODEL) if model else None
    return SLOWeavePolicy(SLOWeaveConfig(enabled=True, c_max=c_max), m)


class TestAlgorithm1(unittest.TestCase):
    def test_no_pending_prefill_returns_zero(self):
        p = policy()
        d = p.decide(now_ms=0.0, deadlines=[100.0], pending_prefill=[], kv_len=KV)
        self.assertEqual(d.chunk, 0)
        self.assertTrue(d.decode_only)

    def test_no_active_decode_uses_the_large_cap(self):
        """§4.3 / Algorithm 1 第 4 行：`A_t = ∅` ⇒ 取 `min(C_max, 剩余)` 加速 TTFT。"""
        p = policy(c_max=4096)
        d = p.decide(now_ms=0.0, deadlines=[], pending_prefill=[("r", 2048)], kv_len=KV)
        self.assertEqual(d.chunk, 2048)
        self.assertFalse(d.decode_only)

    def test_c_max_caps_the_chunk(self):
        p = policy(c_max=512)
        d = p.decide(now_ms=0.0, deadlines=[], pending_prefill=[("r", 4096)], kv_len=KV)
        self.assertEqual(d.chunk, 512)

    def test_nonexpired_budget_yields_a_positive_chunk(self):
        p = policy()
        d = p.decide(now_ms=0.0, deadlines=[500.0], pending_prefill=[("r", 2048)], kv_len=KV)
        self.assertGreater(d.chunk, 0)
        self.assertFalse(d.decode_only)
        self.assertAlmostEqual(d.budget_ms, 500.0)

    def test_expired_deadline_is_decode_only(self):
        """`B_t = max(0, ·)` ⇒ 预算 ≤ 0 时不能塞任何 prefill。"""
        p = policy()
        d = p.decide(now_ms=600.0, deadlines=[500.0], pending_prefill=[("r", 2048)], kv_len=KV)
        self.assertEqual(d.chunk, 0)
        self.assertTrue(d.decode_only)

    def test_budget_is_the_earliest_slack(self):
        p = policy()
        d = p.decide(now_ms=100.0, deadlines=[300.0, 900.0],
                     pending_prefill=[("r", 2048)], kv_len=KV)
        self.assertAlmostEqual(d.budget_ms, 200.0)     # 300 - 100
        self.assertAlmostEqual(d.earliest_deadline_ms, 300.0)

    def test_uses_only_the_oldest_pending_prefill(self):
        """论文实现是 oldest-first（防 pending prefill 之间重排序）⇒ 只看第一条。"""
        p = policy()
        a = p.decide(now_ms=0.0, deadlines=[], pending_prefill=[("old", 512), ("new", 4096)],
                     kv_len=KV)
        self.assertEqual(a.chunk, 512)


class TestMaximality(unittest.TestCase):
    def test_chosen_chunk_is_the_largest_that_fits(self):
        """★ 论文 Prop. 1(ii)：处理的 prefill token 数不少于任何其它"截止安全"的 chunk。

        判据：选中的 `c` 满足约束，而 `c+1` **不**满足（在 `c < L` 时）。
        """
        p = policy(c_max=4096)
        d = p.decide(now_ms=0.0, deadlines=[200.0], pending_prefill=[("r", 4096)], kv_len=KV)
        c = d.chunk
        self.assertGreater(c, 0)
        self.assertLess(c, 4096, "预算 200ms 不该容得下整段 4096")
        B = d.budget_ms
        ok_c = p._predict_ms(1, c, KV)
        ok_c1 = p._predict_ms(1, c + 1, KV)
        self.assertIsNotNone(ok_c)
        self.assertLessEqual(ok_c + p._margin_ms(ok_c), B, "选中的 c 必须满足约束")
        if ok_c1 is not None:
            self.assertGreater(ok_c1 + p._margin_ms(ok_c1), B, "c+1 必须已经违反约束")

    def test_more_budget_never_shrinks_the_chunk(self):
        """单调性：预算更大 ⇒ 选中的 chunk **不会更小**。"""
        p = policy()
        chunks = [p.decide(now_ms=0.0, deadlines=[b], pending_prefill=[("r", 4096)],
                           kv_len=KV).chunk
                  for b in (100.0, 200.0, 400.0, 800.0, 1600.0)]
        self.assertEqual(chunks, sorted(chunks), chunks)


class TestMarginAndRanges(unittest.TestCase):
    def test_margin_is_actually_applied(self):
        """§3.3：判据是 `T̂ + δ ≤ B`，不是 `T̂ ≤ B`。

        构造一个"只有不算 δ 才刚好过"的预算：加 δ 之后必须被拒（chunk 变小）。
        """
        p = policy()
        base = p._predict_ms(1, 256, KV)
        self.assertIsNotNone(base)
        delta = p._margin_ms(base)
        self.assertGreater(delta, 0.0, "本项目成本模型带安全余量，δ 应大于 0")
        # 预算刚好等于 base（不含 δ）⇒ 256 必须不通过
        d_strict = p.decide(now_ms=0.0, deadlines=[base], pending_prefill=[("r", 4096)], kv_len=KV)
        self.assertLess(d_strict.chunk, 256)
        # 预算给到 base + δ ⇒ 256 可以过
        d_ok = p.decide(now_ms=0.0, deadlines=[base + delta + 1e-6],
                        pending_prefill=[("r", 4096)], kv_len=KV)
        self.assertGreaterEqual(d_ok.chunk, 256)

    def test_out_of_range_is_infeasible_not_extrapolated(self):
        """越界 ⇒ 当**不可行**（不外推）⇒ 落到 decode-only，并计数。"""
        p = policy()
        d = p.decide(now_ms=0.0, deadlines=[5000.0], pending_prefill=[("r", 2048)],
                     kv_len=99999)                      # kv 远超标定范围
        self.assertEqual(d.chunk, 0)
        self.assertTrue(d.decode_only)
        self.assertGreaterEqual(p.n_decode_only, 1)

    def test_no_model_means_decode_only(self):
        p = policy(model=False)
        d = p.decide(now_ms=0.0, deadlines=[5000.0], pending_prefill=[("r", 2048)], kv_len=KV)
        self.assertEqual(d.chunk, 0)
        self.assertTrue(d.decode_only)


class TestBookkeeping(unittest.TestCase):
    def test_counters_and_describe(self):
        p = policy()
        p.decide(now_ms=0.0, deadlines=[500.0], pending_prefill=[("r", 2048)], kv_len=KV)
        p.decide(now_ms=0.0, deadlines=[500.0], pending_prefill=[], kv_len=KV)
        p.decide(now_ms=1e9, deadlines=[500.0], pending_prefill=[("r", 2048)], kv_len=KV)
        self.assertEqual(p.n_no_prefill, 1)
        self.assertEqual(p.n_decode_only, 1)
        json.dumps(p.describe(), ensure_ascii=False)

    def test_deterministic(self):
        p = policy()
        a = p.decide(now_ms=0.0, deadlines=[200.0], pending_prefill=[("r", 4096)],
                     kv_len=KV).describe()
        b = p.decide(now_ms=0.0, deadlines=[200.0], pending_prefill=[("r", 4096)],
                     kv_len=KV).describe()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
