"""SLOWeave 基线的两个**反例**测试（2026-09-29 修，S00 阻断项 §5.3）。

背景：`01_PROJECT_PLAN.md` §5.3 要求「使用 SLOWeave 对照时**只应用一次**安全余量；
稀疏 lookup 不能当成连续单调可行域二分」。本文件用**实测模型 `model_v7`**
（`default_mode = lookup`，`safety_margin = 16.02%`）把两条都钉住。

⚠️ 两个用例**在修复前都会失败**（这就是"反例测试"的意义 —— 不是"实现成什么样就测什么样"）：
  * `test_safety_margin_applied_exactly_once`：旧代码让判据乘 `(1+m)² = 1.3461`；
  * `test_sparse_lookup_does_not_collapse_to_decode_only`：旧二分在 `b1/kv1024`
    上从缺键的 `mid=2048` 起一路收缩到 `chunk = 0`，而 `c=1024` 在 400 ms 预算下本可行。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.cost_model import CostModel  # noqa: E402
from ascend_opt.slo_weave import SLOWeaveConfig, SLOWeavePolicy  # noqa: E402

V7 = _ROOT / "configs" / "cost_calib" / "model_v7.json"
V1 = _ROOT / "configs" / "cost_calib" / "model_v1.json"
KV = 1024


def lookup_policy() -> SLOWeavePolicy:
    return SLOWeavePolicy(SLOWeaveConfig(enabled=True, c_max=4096),
                          CostModel.load_offline(V7))


class TestMarginAppliedExactlyOnce(unittest.TestCase):
    """§3.3 的判据是 `T̂ + δ ≤ B`，而 `δ = T̂ · safety_margin` ⇒ 总计 **一次** `(1+m)`。"""

    def test_safety_margin_applied_exactly_once(self):
        m = CostModel.load_offline(V7)
        sm = float(m.payload["safety_margin"])
        self.assertGreater(sm, 0.0)

        p = SLOWeavePolicy(SLOWeaveConfig(enabled=True), m)
        raw = m.predict(batch=1, prefill_tok=256, kv_len=KV, apply_margin=False).total_ms
        t = p._predict_ms(1, 256, KV)
        delta = p._margin_ms(t)

        # `_predict_ms` 必须是**不含**余量的 T（否则 δ 就被乘了第二次）。
        self.assertAlmostEqual(t, raw, places=9)
        # 判据里实际用的是 `t + delta`，其相对 raw 的倍数必须恰为 `1 + m`。
        self.assertAlmostEqual((t + delta) / raw, 1.0 + sm, places=9)

    def test_cost_policy_and_slo_weave_use_the_same_margin_multiple(self):
        """两臂的余量倍率必须相同 —— 否则"余量买到的东西"两臂不可比。"""
        m = CostModel.load_offline(V7)
        sm = float(m.payload["safety_margin"])
        with_m = m.predict(batch=1, prefill_tok=256, kv_len=KV, apply_margin=True).total_ms
        without = m.predict(batch=1, prefill_tok=256, kv_len=KV, apply_margin=False).total_ms
        self.assertAlmostEqual(with_m / without, 1.0 + sm, places=9)


class TestSparseLookupCandidates(unittest.TestCase):
    """`lookup` 表是稀疏的：候选只能是**有键的形状**。"""

    def test_candidates_are_the_table_keys_for_this_bucket(self):
        m = CostModel.load_offline(V7)
        cand = m.lookup_prefill_candidates(batch=1, kv_len=KV, cap=4096)
        self.assertEqual(cand, (128, 256, 512, 1024))
        self.assertNotIn(192, cand)          # 192 未进 v7 部署表（OPT-03 只补了探针）
        self.assertNotIn(2048, cand)         # 该 batch/kv 桶下无键

    def test_segmented_model_reports_no_candidates(self):
        """非 `lookup` 模式 ⇒ 返回 `None` ⇒ 调用方沿用二分（那里 T 单调、可行域连续）。"""
        m = CostModel.load_offline(V1)
        self.assertIsNone(m.lookup_prefill_candidates(batch=1, kv_len=KV, cap=4096))

    def test_cap_filters_candidates(self):
        m = CostModel.load_offline(V7)
        self.assertEqual(m.lookup_prefill_candidates(batch=1, kv_len=KV, cap=300),
                         (128, 256))


class TestSparseLookupDoesNotCollapse(unittest.TestCase):
    """★ 反例：旧二分在稀疏表上会把**可行**预算误判成 decode-only。"""

    def _old_bisection(self, p: SLOWeavePolicy, *, n_active: int, L: int,
                       kv_len: int, budget: float) -> int:
        """**修复前**的二分逻辑（逐行复刻，含 `None ⇒ hi = mid - 1`）。

        保留它是为了让"这个 bug 真实存在过"有可执行证据 —— 它是本用例的反例对照。
        """
        lo, hi = 0, L
        while lo < hi:
            mid = (lo + hi + 1) // 2
            pred = p._predict_ms(n_active, mid, kv_len)
            if pred is None:
                hi = mid - 1
                continue
            if pred + p._margin_ms(pred) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def test_old_bisection_reproduces_the_defect(self):
        """★ 反例本体（`b1` / `kv` 桶 1024，`T` 见下表）。

        | `L` | 预算 | 旧二分 | 真值 | 症状 |
        | --- | --- | --- | --- | --- |
        | 1024 | 400 ms | **512** | 1024 | **欠选**（少一半，`c=1024` 只需 316.9 ms）|
        | 1536 | 320 ms | **0** | 1024 | **塌缩成 decode-only**（`c=512` 只需 194.4 ms）|
        | 4096 | 400 ms | 1024 | 1024 | 偶合正确（`mid` 恰好落在 1024 上）⇒ 不能靠这个自证 |

        最后一行是关键：**同一个 bug 在 L=4096 时看不出来** —— 所以"跑通了"不能当证据。
        """
        p = lookup_policy()
        self.assertEqual(self._old_bisection(p, n_active=1, L=1024, kv_len=KV,
                                             budget=400.0), 512)
        self.assertEqual(self._old_bisection(p, n_active=1, L=1536, kv_len=KV,
                                             budget=320.0), 0)
        self.assertEqual(self._old_bisection(p, n_active=1, L=4096, kv_len=KV,
                                             budget=400.0), 1024)
        # `c=1024` 在 400 ms 下确实可行（316.9 ms）⇒ 上面那个 0/512 是**误判**。
        pred = p._predict_ms(1, 1024, KV)
        self.assertLessEqual(pred + p._margin_ms(pred), 400.0)

    def test_fixed_decision_picks_the_largest_feasible_candidate(self):
        p = lookup_policy()
        d = p.decide(now_ms=0.0, deadlines=[400.0], pending_prefill=[("r", 4096)], kv_len=KV)
        self.assertEqual(d.chunk, 1024)
        self.assertFalse(d.decode_only)

    def test_fixed_decision_does_not_collapse_when_L_is_short(self):
        """旧逻辑在 `L=1536`、预算 320 ms 下塌缩成 0；修复后必须给出 1024。"""
        p = lookup_policy()
        d = p.decide(now_ms=0.0, deadlines=[320.0], pending_prefill=[("r", 1536)], kv_len=KV)
        self.assertEqual(d.chunk, 1024)
        self.assertFalse(d.decode_only)

    def test_fixed_decision_is_not_short_when_L_is_exactly_a_candidate(self):
        """旧逻辑在 `L=1024`、预算 400 ms 下只给 512（欠选一半）。"""
        p = lookup_policy()
        d = p.decide(now_ms=0.0, deadlines=[400.0], pending_prefill=[("r", 1024)], kv_len=KV)
        self.assertEqual(d.chunk, 1024)

    def test_smaller_budget_picks_smaller_candidate(self):
        """单调性现在是**在候选之间**成立（枚举天然如此），不是对连续 c 的假设。"""
        p = lookup_policy()
        chunks = [p.decide(now_ms=0.0, deadlines=[b], pending_prefill=[("r", 4096)],
                           kv_len=KV).chunk for b in (300.0, 400.0, 700.0)]
        self.assertEqual(chunks, [512, 1024, 1024], chunks)
        self.assertEqual(chunks, sorted(chunks))

    def test_missing_key_is_unknown_not_infeasible(self):
        """缺键 = 未知 ⇒ 跳过它，但**更大的可行候选仍要被选中**（这正是旧逻辑做不到的）。"""
        p = lookup_policy()
        cand = p.model.lookup_prefill_candidates(batch=1, kv_len=KV, cap=4096)
        self.assertIn(1024, cand)
        self.assertNotIn(2048, cand)         # 2048 缺键
        d = p.decide(now_ms=0.0, deadlines=[400.0], pending_prefill=[("r", 1024)], kv_len=KV)
        self.assertEqual(d.chunk, 1024)      # 越过缺键的 2048 之前就已停在 1024

    def test_explicit_candidates_override_the_model(self):
        p = lookup_policy()
        d = p.decide(now_ms=0.0, deadlines=[400.0], pending_prefill=[("r", 4096)],
                     kv_len=KV, candidates=[128, 256])
        self.assertEqual(d.chunk, 256)

    def test_out_of_table_range_still_decode_only(self):
        """kv 落在没有键的桶 ⇒ 无候选可预测 ⇒ 保守 decode-only（**不可测就不动**）。"""
        p = lookup_policy()
        d = p.decide(now_ms=0.0, deadlines=[5000.0], pending_prefill=[("r", 4096)],
                     kv_len=99999)
        self.assertEqual(d.chunk, 0)
        self.assertTrue(d.decode_only)


if __name__ == "__main__":
    unittest.main()
