"""S02 / P2a：把 `musa.empty_cache()` 从常规退役路径里分离出来。

要保的性质（每条都配反向对照）：

1. **默认不变**：`retire_empty_cache=True` ⇒ 每次真的摘掉槽位就清一次缓存
   （与既有行为逐字节相同）。
2. **关掉后只摘槽位**：slot 数归零（**无活跃引用泄漏**），但 `empty_cache` 一次不调。
3. **run 边界显式整理**：`flush_allocator_cache()` 可用；设备不可用时返回
   `False`（**不虚报**"调用过"）。
4. **高周转**：连续多轮请求退役不会让 `empty_cache` 的次数随请求数增长。

⚠️ 用假 `torch.musa` 替身来数调用次数。真机上的**耗时**收益（分配器缓存复用省下
的分配/整理开销）只能由真机配对测量回答 ⇒ `REMOTE_PENDING`。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _tiny_support as T  # noqa: E402
from real_backend import ForwardItem  # noqa: E402

import real_backend as rb  # noqa: E402


class _FakeMusa:
    """最小可用的 `torch.musa` 替身。

    ⚠️ 必须覆盖驱动**实际调用到的每一个**方法 —— 第一次写时漏了 `synchronize`，
    于是 5 个用例挂在 `AttributeError` 上，症状看起来像"实现坏了"，
    实际是替身不全（这正是"替身与被替者接口不一致"的典型表现）。
    """

    def __init__(self) -> None:
        self.empty_cache_calls = 0
        self.synchronize_calls = 0

    def is_available(self) -> bool:
        return True

    def synchronize(self) -> None:
        self.synchronize_calls += 1

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def mem_get_info(self):
        mib = 1024 * 1024
        total = 16 * 1024 * mib
        free = 8 * 1024 * mib
        return free, total


class _MusaPatcher(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = _FakeMusa()
        self._orig = rb._musa_module
        rb._musa_module = lambda torch: self.fake            # type: ignore[assignment]

    def tearDown(self) -> None:
        rb._musa_module = self._orig                          # type: ignore[assignment]


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestRetireDefaultUnchanged(_MusaPatcher):
    def test_default_clears_cache_once_per_removed_slot_batch(self):
        drv = T.make_driver(T.make_model())
        drv.load()
        drv.forward([ForwardItem(T.req("a", 4), n_new_tokens=4, need_token=True)])
        self.assertEqual(drv.live_slots(), 1)
        drv.retire(["a"])
        self.assertEqual(self.fake.empty_cache_calls, 1)
        self.assertEqual(drv.execution_optimizations["retire_empty_cache"]
                         ["n_empty_cache_calls"], 1)

    def test_retiring_unknown_id_does_not_clear(self):
        """没真摘掉槽位就不该调 `empty_cache`（白调是纯开销）。"""
        drv = T.make_driver(T.make_model())
        drv.load()
        drv.retire(["nope"])
        self.assertEqual(self.fake.empty_cache_calls, 0)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestRetireWithoutEmptyCache(_MusaPatcher):
    def test_slots_are_still_released_but_cache_is_not_cleared(self):
        drv = T.make_driver(T.make_model(), retire_empty_cache=False)
        drv.load()
        for name in ("a", "b", "c"):
            drv.forward([ForwardItem(T.req(name, 4), n_new_tokens=4, need_token=True)])
        self.assertEqual(drv.live_slots(), 3)
        drv.retire(["a", "b", "c"])
        # ★ 无活跃引用泄漏：槽位必须真的被摘掉（这是**不能**省的那一半）
        self.assertEqual(drv.live_slots(), 0)
        self.assertEqual(self.fake.empty_cache_calls, 0, "关掉后不得清缓存")
        eo = drv.execution_optimizations["retire_empty_cache"]
        self.assertFalse(eo["requested_at_retire"])
        self.assertEqual(eo["n_empty_cache_calls"], 0)
        self.assertEqual(eo["n_suppressed"], 1)

    def test_high_turnover_keeps_empty_cache_count_zero(self):
        """高周转：连续 10 轮「进 1 出 1」，`empty_cache` 次数不随请求数增长。"""
        drv = T.make_driver(T.make_model(), retire_empty_cache=False)
        drv.load()
        for i in range(10):
            name = f"r{i:02d}"
            drv.forward([ForwardItem(T.req(name, 4), n_new_tokens=4, need_token=True)])
            drv.retire([name])
            self.assertEqual(drv.live_slots(), 0, "每轮结束后不得残留槽位")
        self.assertEqual(self.fake.empty_cache_calls, 0)

    def test_flush_at_run_boundary_clears_once(self):
        drv = T.make_driver(T.make_model(), retire_empty_cache=False)
        drv.load()
        drv.forward([ForwardItem(T.req("a", 4), n_new_tokens=4, need_token=True)])
        drv.retire(["a"])
        self.assertTrue(drv.flush_allocator_cache())
        self.assertEqual(self.fake.empty_cache_calls, 1)
        self.assertEqual(drv.execution_optimizations["retire_empty_cache"]
                         ["n_empty_cache_calls"], 1)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestFlushReportsTruthfully(unittest.TestCase):
    """设备不可用时 `flush_allocator_cache()` 必须返回 `False`，不能虚报。"""

    def test_returns_false_when_device_unavailable(self):
        orig = rb._musa_module
        rb._musa_module = lambda torch: None                  # type: ignore[assignment]
        try:
            drv = T.make_driver(T.make_model(), retire_empty_cache=False)
            drv.load()
            # `_CpuDriver._device_available` 恒 True，但 `_musa_module` 返回 None
            # ⇒ 无从调用 empty_cache ⇒ 必须如实返回 False。
            self.assertFalse(drv.flush_allocator_cache())
            self.assertEqual(drv.execution_optimizations["retire_empty_cache"]
                             ["n_empty_cache_calls"], 0)
        finally:
            rb._musa_module = orig                           # type: ignore[assignment]


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestNumbersUnaffected(_MusaPatcher):
    def test_tokens_identical_with_and_without_cache_flush(self):
        """只改"何时清缓存"不得改变任何计算结果。"""
        seqs = []
        for flag in (True, False):
            drv = T.make_driver(T.make_model(), retire_empty_cache=flag)
            drv.load()
            out = drv.forward([ForwardItem(T.req("a", 6), n_new_tokens=6,
                                           need_token=True)])
            seq = [out.slots[0].token_id]
            for _ in range(2):
                out = drv.forward([ForwardItem(T.req("a", 6), n_new_tokens=1,
                                               need_token=True)])
                seq.append(out.slots[0].token_id)
            drv.retire(["a"])
            seqs.append(seq)
        self.assertEqual(seqs[0], seqs[1])


if __name__ == "__main__":
    unittest.main()
