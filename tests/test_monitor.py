"""状态观测与规范模块命名的测试（规范第 2 / 12 节）。

守两件事：
1. **不可测的量保持 ``None``**。填 0 会被读成"显存耗尽"或"NPU 空闲"，
   而控制器正是按这个状态向量做决策的——伪造一个没发生过的状态，
   比测不出来危险得多。
2. **被协调的子控制器仍然是有界且纯函数的**。控制器振荡是这类工作
   最常见的自伤，边界与无状态性是它的两道防线。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.cache_policy import CacheStats  # noqa: E402
from ascend_opt.controller import (  # noqa: E402
    AdaptiveBatchController,
    AdaptiveController,
    ChunkController,
    ControllerConfig,
)
from ascend_opt.monitor import (  # noqa: E402
    STATE_VECTOR_FIELDS,
    SystemState,
    WorkloadMonitor,
    estimate_prefix_reuse_ratio,
)
from ascend_opt.workload import Request  # noqa: E402


def _req(req_id: str, prompt: int, gen: int = 32, shared: int = 0) -> Request:
    return Request(
        req_id=req_id,
        prompt_len=prompt,
        gen_len=gen,
        shared_prefix_len=shared,
        prefix_group="g0" if shared else "",
    )


class TestSystemState(unittest.TestCase):
    def test_all_unavailable_by_default(self):
        state = SystemState()
        self.assertEqual(set(state.unavailable_fields()), set(STATE_VECTOR_FIELDS))
        self.assertFalse(state.complete)

    def test_vector_order_matches_spec_field_list(self):
        state = SystemState(input_length=1, queue_length=2, ttft_ms=3)
        vector = state.as_vector()
        self.assertEqual(len(vector), len(STATE_VECTOR_FIELDS))
        idx = {name: i for i, name in enumerate(STATE_VECTOR_FIELDS)}
        self.assertEqual(vector[idx["input_length"]], 1)
        self.assertEqual(vector[idx["queue_length"]], 2)
        self.assertEqual(vector[idx["ttft_ms"]], 3)

    def test_missing_values_stay_none_and_are_never_zero_filled(self):
        # 这是本模块最重要的不变量：0 与 None 语义完全不同。
        state = SystemState(input_length=512)
        vector = dict(zip(STATE_VECTOR_FIELDS, state.as_vector()))
        self.assertIsNone(vector["available_memory_mb"])
        self.assertIsNone(vector["npu_utilization_pct"])
        self.assertNotEqual(vector["available_memory_mb"], 0)

    def test_describe_exposes_unavailable_fields(self):
        described = SystemState(input_length=512).describe()
        self.assertIn("npu_utilization_pct", described["_unavailable_fields"])


class TestPrefixReuseRatio(unittest.TestCase):
    def test_none_when_no_requests(self):
        # 空负载的复用率未定义；写成 0 会让"低复用负载"与"空负载"看起来一样。
        self.assertIsNone(estimate_prefix_reuse_ratio([]))

    def test_zero_when_measured_but_no_sharing(self):
        ratio = estimate_prefix_reuse_ratio([_req("a", 100), _req("b", 100)])
        self.assertEqual(ratio, 0.0)

    def test_ratio_is_token_weighted_not_request_weighted(self):
        # 一个长请求占主导时，按请求数平均会算出 50%，按 token 才是真实值。
        ratio = estimate_prefix_reuse_ratio([
            _req("a", 1000, shared=1000),
            _req("b", 1000, shared=0),
        ])
        self.assertAlmostEqual(ratio, 0.5)

    def test_shared_prefix_longer_than_prompt_is_clamped(self):
        ratio = estimate_prefix_reuse_ratio([_req("a", 100, shared=100)])
        self.assertAlmostEqual(ratio, 1.0)


class TestWorkloadMonitor(unittest.TestCase):
    def test_hardware_fields_stay_unavailable_on_offline_path(self):
        monitor = WorkloadMonitor()
        state = monitor.observe(requests=[_req("a", 512)], step=0, queue_depth=1)
        self.assertEqual(state.available_memory_mb, None)
        self.assertEqual(state.npu_utilization_pct, None)
        self.assertIsNotNone(state.input_length)

    def test_available_memory_needs_both_used_and_total(self):
        monitor = WorkloadMonitor()
        only_used = monitor.observe(memory_used_mb=1024.0)
        self.assertIsNone(only_used.available_memory_mb, "只有已用、没有总量时不可推断")
        both = monitor.observe(memory_used_mb=1024.0, memory_total_mb=65536.0)
        self.assertEqual(both.available_memory_mb, 64512.0)

    def test_cache_hit_ratio_comes_from_cache_stats(self):
        monitor = WorkloadMonitor()
        stats = CacheStats(lookups=4, hits=3)
        state = monitor.observe(cache_stats=stats)
        self.assertAlmostEqual(state.cache_hit_ratio, 0.75)

    def test_cache_hit_ratio_is_none_when_nothing_was_looked_up(self):
        # 缓存关闭时 CacheStats.hit_rate 读出 0.0，读起来像"缓存开着但一次没命中"。
        # 两者的对策完全相反（前者"这个信号不存在"，后者"策略没留住前缀"），
        # 所以监测器必须把它传成 None（未测）。
        monitor = WorkloadMonitor()
        state = monitor.observe(cache_stats=CacheStats())
        self.assertIsNone(state.cache_hit_ratio)
        self.assertNotEqual(state.cache_hit_ratio, 0.0)
        self.assertIn("cache_hit_ratio", state.unavailable_fields())

    def test_missing_cache_stats_leaves_the_signal_unmeasured(self):
        monitor = WorkloadMonitor()
        self.assertIsNone(monitor.observe().cache_hit_ratio)

    def test_latency_percentiles_use_p90_not_mean(self):
        monitor = WorkloadMonitor()
        state = monitor.observe(ttft_samples=[100.0] * 9 + [900.0])
        # P90 而非均值：尾延迟才是 SLO 违约的来源。
        self.assertGreater(state.ttft_ms, 100.0)
        self.assertLessEqual(state.ttft_ms, 900.0)

    def test_history_is_bounded_and_returned_as_copy(self):
        monitor = WorkloadMonitor(history_limit=3)
        for i in range(10):
            monitor.observe(step=i, requests=[_req("a", 100)])
        self.assertEqual(len(monitor.history()), 3)
        # 返回的是副本，调用方改写不会污染监测记录。
        monitor.history().clear()
        self.assertEqual(len(monitor.history()), 3)

    def test_reset_clears_history(self):
        monitor = WorkloadMonitor()
        monitor.observe(step=0)
        monitor.reset()
        self.assertIsNone(monitor.latest())


class TestChunkController(unittest.TestCase):
    def test_grow_respects_upper_bound(self):
        ctrl = ChunkController(ControllerConfig(chunk_size_min=512,
                                                chunk_size_max=1024,
                                                chunk_size_step=512))
        value, moved = ctrl.grow(512)
        self.assertEqual((value, moved), (1024, True))
        # 已在上界：不再移动，且必须报告 changed=False 让调用方计 clamp
        value, moved = ctrl.grow(1024)
        self.assertEqual((value, moved), (1024, False))

    def test_shrink_respects_lower_bound(self):
        ctrl = ChunkController(ControllerConfig(chunk_size_min=512,
                                                chunk_size_max=1024,
                                                chunk_size_step=256))
        value, moved = ctrl.shrink(1024)
        self.assertEqual((value, moved), (768, True))
        value, moved = ctrl.shrink(512)
        self.assertEqual((value, moved), (512, False))

    def test_controller_is_stateless(self):
        # 同一个实例连续调用不得因历史而改变结果。
        ctrl = ChunkController(ControllerConfig())
        self.assertEqual(ctrl.shrink(4096), ctrl.shrink(4096))

    def test_clamp_keeps_value_inside_bounds(self):
        ctrl = ChunkController(ControllerConfig(chunk_size_min=512,
                                                chunk_size_max=2048))
        self.assertEqual(ctrl.clamp(1), 512)
        self.assertEqual(ctrl.clamp(999999), 2048)


class TestAdaptiveBatchController(unittest.TestCase):
    def test_batch_size_control_is_off_by_default(self):
        # 默认不控制并发上限：改它会改变准入语义，是需要显式开启的一档干预。
        cfg = ControllerConfig()
        self.assertFalse(cfg.batch_size_controlled)
        ctrl = AdaptiveBatchController(cfg)
        self.assertFalse(ctrl.seq_bound.active)
        self.assertEqual(ctrl.grow_seqs(32), (32, False))

    def test_batch_size_control_works_when_enabled(self):
        cfg = ControllerConfig(batch_size_min=8, batch_size_max=32, batch_size_step=8)
        ctrl = AdaptiveBatchController(cfg)
        self.assertTrue(cfg.batch_size_controlled)
        self.assertEqual(ctrl.grow_seqs(8), (16, True))
        self.assertEqual(ctrl.shrink_seqs(8), (8, False))

    def test_invalid_batch_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            ControllerConfig(batch_size_min=64, batch_size_max=32)


class TestAdaptiveControllerCoordination(unittest.TestCase):
    def test_state_starts_from_config_initials(self):
        cfg = ControllerConfig(chunk_size_init=2048, max_batched_tokens_init=4096)
        ctrl = AdaptiveController(cfg)
        self.assertEqual(ctrl.state.chunk_size, 2048)
        self.assertEqual(ctrl.state.max_batched_tokens, 4096)

    def test_batch_controller_is_wired_into_runtime_state(self):
        cfg = ControllerConfig(batch_size_init=16, batch_size_min=8, batch_size_max=32)
        ctrl = AdaptiveController(cfg)
        self.assertEqual(ctrl.state.max_num_seqs, 16)

    def test_plan_describe_lists_both_components(self):
        described = AdaptiveController().plan_describe()
        names = {c["controller"] for c in described["components"]}
        self.assertEqual(names, {"chunk_controller", "adaptive_batch_controller"})

    def test_tpot_pressure_reduces_per_step_work(self):
        # decode 超目标时，控制器要给 decode 让出预算：chunk 与 token 预算都要降。
        cfg = ControllerConfig(
            chunk_size_init=4096, chunk_size_min=512, chunk_size_step=512,
            max_batched_tokens_init=8192, max_batched_tokens_min=2048,
            max_batched_tokens_step=1024, target_tpot_ms=50.0, hysteresis=0.0,
        )
        ctrl = AdaptiveController(cfg)
        action = ctrl.observe(step=100, ttft_samples=[100.0], tpot_samples=[200.0])
        self.assertTrue(action.changed)
        self.assertEqual(action.bottleneck, "tpot")
        self.assertEqual(ctrl.state.chunk_size, 3584)
        self.assertEqual(ctrl.state.max_batched_tokens, 7168)

    def test_ttft_pressure_without_tpot_headroom_is_refused(self):
        # 这是取舍而非免费收益：TPOT 已贴近目标时不该为了 TTFT 放大 chunk。
        cfg = ControllerConfig(target_tpot_ms=50.0, target_ttft_ms=100.0,
                               hysteresis=0.10, cooldown_cycles=0)
        ctrl = AdaptiveController(cfg)
        before = ctrl.state.chunk_size
        action = ctrl.observe(step=100, ttft_samples=[500.0], tpot_samples=[49.0])
        self.assertFalse(action.changed)
        self.assertEqual(action.bottleneck, "ttft")
        self.assertEqual(ctrl.state.chunk_size, before)


if __name__ == "__main__":
    unittest.main()
