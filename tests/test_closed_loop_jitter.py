"""闭环补位抖动（`B1` 草案 §5 / OPT-03R 裁定 Q4）的用例。

**要保的性质**（★ 第 3 条于 2026-10-03 **反过来了**，见下）：

1. **默认关闭**（`closure_jitter_ms=0`）⇒ 抖动恒 0 ⇒ 与既有 `closed` 逐字节相同。
2. **可复现**：同 `(seed, 客户端, 轮次)` ⇒ 同值（`hash()` 不行 —— `PYTHONHASHSEED` 会变）。
3. ★★ **与调度**无**关**：抖动**不得**依赖 `step`。派生依据只有
   `(seed, 客户端, 轮次)`，而这两者都由**负载**决定 ⇒ **到达时刻是输入**、
   跑之前就有定义。

   ⚠️ 旧版把这条写成"**与调度耦合**（`step` 变 ⇒ 值变，这是本改动的**目的**）"。
   那个"目的"是错的：`step` 是调度策略的函数 ⇒ ① 到达时刻在跑之前没有定义；
   ② 请求仍在 `step_start` 入队 ⇒ 实测 **16/16** 个请求 `enqueue_ms < arrival_ms`，
   "到达"退化成装饰性字段。**去相关**这个目标由"抖动本身"达成（值随客户端/轮次变化），
   **不需要**、也不应该让抖动去依赖调度。
4. **值域** `[0, span)`：不能出现负值或越界（到达时刻不能比放行时刻更早）。
5. **校验**：开放环用 `closure_jitter_ms` ⇒ **报错**（不静默忽略）；负值 ⇒ 报错。
6. **落盘**：`describe()` 带 `closure_jitter_ms`。

⚠️ 这里是**纯函数 + 配置**的单元测试；"跨重复方差是否真的被放大"属**真机**问题，
须由真机批次回答（dry-run 的模拟器不是真机时序）。
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.workload import WorkloadSpec  # noqa: E402
from real_backend import _closure_jitter_ms, closure_jitter_by_request  # noqa: E402


class TestJitterFunction(unittest.TestCase):
    """抖动是 `(seed, 客户端, 轮次)` 的纯函数 —— **与调度无关**。"""

    def test_zero_span_returns_zero(self):
        """默认关闭 ⇒ 恒 0（与既有 closed 一致）。"""
        for s_ in (0, 1, 7):
            self.assertEqual(
                _closure_jitter_ms(seed=s_, client=0, round_index=0, span_ms=0.0), 0.0)
        # 负 span 也当 0（不产生负抖动 —— 那会让到达早于放行）
        self.assertEqual(
            _closure_jitter_ms(seed=1, client=1, round_index=1, span_ms=-5.0), 0.0)

    def test_reproducible_same_inputs(self):
        """同输入必得同输出 —— 这条让抖动可重放。"""
        a = _closure_jitter_ms(seed=42, client=7, round_index=11, span_ms=30.0)
        b = _closure_jitter_ms(seed=42, client=7, round_index=11, span_ms=30.0)
        self.assertEqual(a, b)

    def test_not_python_hash(self):
        """实现必须与 `PYTHONHASHSEED` 无关（用 blake2b 而非 `hash()`）。

        这条是**回归锚**：若有人改成 `hash(...)`，跨进程/跨会话的重放会失配。
        """
        h = hashlib.blake2b(b"42|c7|r11", digest_size=8).digest()
        expect = (int.from_bytes(h, "big") / float(1 << 64)) * 30.0
        got = _closure_jitter_ms(seed=42, client=7, round_index=11, span_ms=30.0)
        self.assertAlmostEqual(got, expect, places=9)

    def test_signature_carries_no_step(self):
        """★★ **靶心（性质 3）**：派生依据里**不许**出现 `step`。

        结构断言 + 行为断言一起给：只要 `step` 不出现在签名里，"到达时刻是调度的
        函数"这件事在**接口层面**就不可能发生（而不是靠约定）。
        """
        import inspect

        params = set(inspect.signature(_closure_jitter_ms).parameters)
        self.assertNotIn("step", params)
        self.assertEqual(params, {"seed", "client", "round_index", "span_ms"})

    def test_client_or_round_change_changes_value(self):
        """值随 `(客户端, 轮次)` 变化 ⇒ **去相关**靠的是它，不是靠调度。"""
        base = _closure_jitter_ms(seed=1, client=3, round_index=5, span_ms=30.0)
        self.assertNotEqual(
            base, _closure_jitter_ms(seed=1, client=4, round_index=5, span_ms=30.0))
        self.assertNotEqual(
            base, _closure_jitter_ms(seed=1, client=3, round_index=6, span_ms=30.0))

    def test_range_and_uniformity(self):
        """值域 `[0, span)`，且分布大致均匀（非均匀会让批边界被系统性偏置）。"""
        span = 30.0
        vals = [_closure_jitter_ms(seed=7, client=i % 17, round_index=i // 17,
                                   span_ms=span)
                for i in range(3000)]
        self.assertGreaterEqual(min(vals), 0.0)
        self.assertLess(max(vals), span, "抖动必须 < span（否则不是 [0,span)）")
        self.assertLess(abs(statistics.mean(vals) - span / 2.0), span * 0.05)


class TestJitterTable(unittest.TestCase):
    """`closure_jitter_by_request`：把"客户端 × 轮次"绑定成一张**预先算好**的表。"""

    def test_client_and_round_binding(self):
        order = [f"r{i:05d}" for i in range(8)]
        c2 = closure_jitter_by_request(seed=1, order=order, concurrency=2, span_ms=25.0)
        c4 = closure_jitter_by_request(seed=1, order=order, concurrency=4, span_ms=25.0)
        self.assertEqual(c2["r00000"], c4["r00000"], "client0/round0 与并发无关")
        self.assertNotEqual(c2["r00002"], c4["r00002"],
                            "并发变了 ⇒ (客户端, 轮次) 绑定变了 ⇒ 抖动应随之变")

    def test_deterministic(self):
        order = [f"r{i:05d}" for i in range(12)]
        a = closure_jitter_by_request(seed=3, order=order, concurrency=4, span_ms=25.0)
        b = closure_jitter_by_request(seed=3, order=order, concurrency=4, span_ms=25.0)
        self.assertEqual(a, b)

    def test_all_distinct_for_distinct_positions(self):
        order = [f"r{i:05d}" for i in range(16)]
        tbl = closure_jitter_by_request(seed=0, order=order, concurrency=4, span_ms=25.0)
        self.assertEqual(len(set(tbl.values())), 16)

    def test_zero_span_is_all_zero(self):
        tbl = closure_jitter_by_request(seed=0, order=["a", "b"], concurrency=1, span_ms=0.0)
        self.assertEqual(set(tbl.values()), {0.0})

    def test_zero_concurrency_falls_back_to_one(self):
        """`concurrency=0` 是配置错误，但不该把取模搞崩。"""
        tbl = closure_jitter_by_request(seed=0, order=["a", "b"], concurrency=0,
                                       span_ms=10.0)
        self.assertEqual(len(tbl), 2)


class TestWorkloadSpecValidation(unittest.TestCase):
    def test_open_loop_rejects_jitter(self):
        """开放环用 `closure_jitter_ms` ⇒ 报错（不静默忽略）。"""
        for arrival in ("poisson", "burst"):
            with self.assertRaises(ValueError):
                WorkloadSpec(name="x", arrival=arrival, closure_jitter_ms=10.0)

    def test_negative_jitter_rejected(self):
        with self.assertRaises(ValueError):
            WorkloadSpec(name="x", arrival="closed", closure_jitter_ms=-1.0)

    def test_closed_jittered_accepted_and_described(self):
        s = WorkloadSpec(name="slo_reach", arrival="closed_jittered",
                         closure_jitter_ms=25.0, concurrency=4)
        d = s.describe()
        self.assertEqual(d["arrival"], "closed_jittered")
        self.assertEqual(d["closure_jitter_ms"], 25.0)
        # 生成器仍产出正确条数（真实到达时刻由执行器在补位时分配）
        self.assertEqual(len(s.build()), s.num_requests)

    def test_plain_closed_still_works(self):
        """既有 `closed` 不受影响（默认值 0）。"""
        s = WorkloadSpec(name="legacy", arrival="closed", concurrency=4)
        self.assertEqual(s.closure_jitter_ms, 0.0)
        self.assertEqual(s.describe()["closure_jitter_ms"], 0.0)

    def test_zero_jitter_on_closed_is_allowed(self):
        """显式写 0 不报错（等价于不启用）。"""
        s = WorkloadSpec(name="x", arrival="closed", closure_jitter_ms=0.0)
        self.assertEqual(s.closure_jitter_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
