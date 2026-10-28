"""策略层单元测试：调度 / 前缀缓存 / 分块 / 控制器。

这些测试的共同目的不是"证明实现正确"，而是**锁住策略的边界条件**：
预算不能被突破、缓存不能复用未完成的前缀、控制器不能在阈值附近振荡。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.cache_policy import (  # noqa: E402
    CacheStats,
    PrefixCacheConfig,
    PrefixCachePolicy,
)
from ascend_opt.chunk_policy import (  # noqa: E402
    ChunkConfig,
    ChunkLimits,
    DecodePriorityChunkPolicy,
    FixedChunkPolicy,
    QueueAwareChunkPolicy,
    build_chunk_policy,
)
from ascend_opt.controller import (  # noqa: E402
    AdaptiveController,
    ControllerConfig,
    replay_decisions,
)
from ascend_opt.scheduler import (  # noqa: E402
    ContinuousBatchingScheduler,
    FIFOScheduler,
    StaticBatchScheduler,
    TokenBudget,
    build_scheduler,
)
from ascend_opt.workload import Request, WorkloadSpec, build_requests  # noqa: E402


def make_requests(n: int, prompt_len: int = 100, gen_len: int = 8,
                  shared: int = 0, group: str = "") -> list[Request]:
    return [
        Request(
            req_id=f"r{i:03d}",
            prompt_len=prompt_len,
            gen_len=gen_len,
            shared_prefix_len=shared,
            prefix_group=group,
        )
        for i in range(n)
    ]


class TestSchedulerBudget(unittest.TestCase):
    def test_all_policies_respect_token_budget(self):
        waiting = make_requests(20, prompt_len=600)
        budget = TokenBudget(max_num_batched_tokens=2048, max_num_seqs=32)
        for policy in (FIFOScheduler(), StaticBatchScheduler(), ContinuousBatchingScheduler()):
            plan = policy.plan(waiting, [], budget)
            self.assertLessEqual(
                plan.scheduled_tokens,
                budget.max_num_batched_tokens,
                msg=f"{policy.name} 突破了 token 预算",
            )

    def test_all_policies_respect_seq_budget(self):
        waiting = make_requests(50, prompt_len=10)
        budget = TokenBudget(max_num_batched_tokens=100000, max_num_seqs=4)
        for policy in (FIFOScheduler(), StaticBatchScheduler(), ContinuousBatchingScheduler()):
            plan = policy.plan(waiting, [], budget)
            self.assertLessEqual(plan.num_seqs, 4, msg=f"{policy.name} 突破了序列数上限")

    def test_continuous_batching_prioritises_decode(self):
        # decode 被饿死是 TPOT 尾延迟的主要来源，这条测试守的就是这个机制
        running = make_requests(3, prompt_len=100)
        waiting = make_requests(10, prompt_len=2000)
        budget = TokenBudget(max_num_batched_tokens=2048, max_num_seqs=32)
        plan = ContinuousBatchingScheduler().plan(waiting, running, budget)
        self.assertEqual(len(plan.decode), 3)
        self.assertLess(plan.scheduled_tokens, budget.max_num_batched_tokens * 2)

    def test_static_batch_does_not_admit_while_busy(self):
        running = make_requests(2)
        waiting = make_requests(5)
        plan = StaticBatchScheduler().plan(waiting, running, TokenBudget())
        self.assertEqual(len(plan.prefill), 0)
        self.assertEqual(len(plan.decode), 2)

    def test_occupied_seqs_reduces_admission(self):
        # 正在 prefill 的请求也占名额。忽视它会导致超额准入，
        # 等这些请求转入 decode 时序列数突破上限，尾部请求被永久饿死。
        waiting = make_requests(20, prompt_len=10)
        budget = TokenBudget(max_num_batched_tokens=100000, max_num_seqs=8,
                             occupied_seqs=3)
        for policy in (FIFOScheduler(), StaticBatchScheduler(),
                       ContinuousBatchingScheduler()):
            with self.subTest(policy=policy.name):
                plan = policy.plan(waiting, [], budget)
                self.assertLessEqual(plan.num_seqs, 8 - 3)

    def test_continuous_batching_advances_all_decoding_requests(self):
        # decode 侧不做截断：被截掉的请求再也拿不到 token，
        # 症状是「永远不完成」而不是报错，属于最难查的一类问题。
        running = make_requests(12, prompt_len=100)
        budget = TokenBudget(max_num_batched_tokens=64, max_num_seqs=4)
        plan = ContinuousBatchingScheduler().plan([], running, budget)
        self.assertEqual(len(plan.decode), 12)
        self.assertIn("invariant violated", plan.note)

    def test_plans_are_reproducible(self):
        waiting = make_requests(12, prompt_len=300)
        policy = ContinuousBatchingScheduler()
        first = policy.plan(waiting, [], TokenBudget())
        second = policy.plan(waiting, [], TokenBudget())
        self.assertEqual(
            [r.req_id for r in first.prefill], [r.req_id for r in second.prefill]
        )

    def test_unknown_policy_name_raises(self):
        with self.assertRaises(ValueError):
            build_scheduler("teleport")


class TestPrefixCache(unittest.TestCase):
    def test_shared_prefix_hits_after_warmup(self):
        cfg = PrefixCacheConfig(enabled=True, block_size=16, max_blocks=1024)
        cache = PrefixCachePolicy(cfg)
        reqs = make_requests(8, prompt_len=512, shared=512, group="g0")
        cache.warmup(reqs)
        cache.reset_stats()

        reused = cache.lookup(reqs[0])
        self.assertEqual(reused, 512)
        self.assertAlmostEqual(cache.stats.hit_rate, 1.0)
        self.assertAlmostEqual(cache.stats.token_reuse_rate, 1.0)

    def test_cold_cache_reports_no_reuse(self):
        cache = PrefixCachePolicy(PrefixCacheConfig(enabled=True))
        req = make_requests(1, prompt_len=512, shared=512, group="g0")[0]
        self.assertEqual(cache.lookup(req), 0)
        self.assertEqual(cache.stats.hits, 0)

    def test_disabled_cache_never_reports_reuse(self):
        cache = PrefixCachePolicy(PrefixCacheConfig(enabled=False))
        reqs = make_requests(4, prompt_len=512, shared=512, group="g0")
        cache.warmup(reqs)
        self.assertEqual(cache.lookup(reqs[0]), 0)

    def test_reuse_is_block_granular(self):
        # 共享 100 token、块长 16 -> 只有 6 个完整块可复用（96 token）
        cfg = PrefixCacheConfig(enabled=True, block_size=16, max_blocks=1024)
        cache = PrefixCachePolicy(cfg)
        reqs = make_requests(2, prompt_len=200, shared=100, group="g0")
        cache.warmup(reqs)
        self.assertEqual(cache.lookup(reqs[0]), 96)

    def test_reuse_never_exceeds_shared_prefix(self):
        cfg = PrefixCacheConfig(enabled=True, block_size=16, max_blocks=1024)
        cache = PrefixCachePolicy(cfg)
        reqs = make_requests(2, prompt_len=256, shared=64, group="g0")
        cache.warmup(reqs)
        reused = cache.lookup(reqs[0])
        self.assertLessEqual(reused, 64)

    def test_capacity_limit_causes_eviction_and_reduces_hits(self):
        # 容量不足必须体现为命中率下降，而不是"缓存无限大"的假象
        cfg = PrefixCacheConfig(enabled=True, block_size=16, max_blocks=4,
                                eviction="lru", min_reuse_blocks=1)
        cache = PrefixCachePolicy(cfg)
        # 每个请求 16 个块，容量只有 4 块，逐一带入后早期的组会被淘汰
        for g in range(8):
            req = Request(req_id=f"r{g}", prompt_len=256, gen_len=4,
                          shared_prefix_len=256, prefix_group=f"g{g}")
            cache.admit(req)
        self.assertEqual(cache.resident_blocks, 4)
        self.assertGreater(cache.stats.evictions, 0)

        first = Request(req_id="x", prompt_len=256, gen_len=4,
                        shared_prefix_len=256, prefix_group="g0")
        self.assertEqual(cache.lookup(first), 0)  # 已被淘汰

    def test_no_eviction_mode_respects_capacity_by_growing(self):
        cfg = PrefixCacheConfig(enabled=True, block_size=16, max_blocks=2,
                                eviction="none")
        cache = PrefixCachePolicy(cfg)
        cache.admit(make_requests(1, prompt_len=256, shared=256, group="g0")[0])
        self.assertEqual(cache.stats.evictions, 0)
        self.assertEqual(cache.resident_blocks, 16)

    def test_min_reuse_blocks_gate(self):
        # 复用收益太小时放弃走缓存路径，避免得不偿失
        cfg = PrefixCacheConfig(enabled=True, block_size=16, max_blocks=64,
                                min_reuse_blocks=4)
        cache = PrefixCachePolicy(cfg)
        reqs = make_requests(2, prompt_len=256, shared=32, group="g0")
        cache.warmup(reqs)
        self.assertEqual(cache.lookup(reqs[0]), 0)
        self.assertEqual(cache.stats.rejected_small, 1)

    def test_hit_rate_or_none_distinguishes_no_lookup_from_zero_hits(self):
        # 决策用的命中率必须能区分这两种情形。合并成一个 0.0 之后，
        # "缓存关闭"会被读成"缓存开着但一次没命中"，进而触发一次基于假信号的决策。
        cold = CacheStats()                      # 一次查询都没有 -> 未测
        self.assertIsNone(cold.hit_rate_or_none)
        self.assertEqual(cold.hit_rate, 0.0)     # 报表口径仍然是 0.0

        missed = CacheStats(lookups=4, hits=0)   # 确实查过、确实没命中
        self.assertEqual(missed.hit_rate_or_none, 0.0)
        self.assertEqual(missed.hit_rate, 0.0)

        hit = CacheStats(lookups=4, hits=3)
        self.assertAlmostEqual(hit.hit_rate_or_none, 0.75)


class TestChunkPolicies(unittest.TestCase):
    def test_fixed_chunk_is_capped_by_chunk_size(self):
        policy = FixedChunkPolicy(ChunkConfig(enabled=True, chunk_size=1024))
        budget = TokenBudget(max_num_batched_tokens=8192)
        self.assertEqual(
            policy.prefill_budget(pending_prefill_tokens=9999,
                                  running_decodes=0, budget=budget),
            1024,
        )

    def test_disabled_chunking_uses_full_budget(self):
        policy = FixedChunkPolicy(ChunkConfig(enabled=False, chunk_size=1024))
        budget = TokenBudget(max_num_batched_tokens=8192)
        self.assertEqual(
            policy.prefill_budget(pending_prefill_tokens=9999,
                                  running_decodes=0, budget=budget),
            8192,
        )

    def test_no_pending_prefill_returns_zero(self):
        policy = FixedChunkPolicy(ChunkConfig(enabled=True, chunk_size=1024))
        self.assertEqual(
            policy.prefill_budget(pending_prefill_tokens=0,
                                  running_decodes=5,
                                  budget=TokenBudget()),
            0,
        )

    def test_decode_priority_reserves_budget(self):
        policy = DecodePriorityChunkPolicy(
            ChunkConfig(enabled=True, chunk_size=8192,
                        max_num_batched_tokens=8192, min_decode_seqs=4)
        )
        budget = TokenBudget(max_num_batched_tokens=8192)
        got = policy.prefill_budget(pending_prefill_tokens=99999,
                                    running_decodes=4, budget=budget)
        self.assertLessEqual(got, 8192 - 4)

    def test_small_remainder_is_not_split_again(self):
        # 剩余量小于一个 chunk 时直接给完整剩余量，避免无意义的碎切
        policy = DecodePriorityChunkPolicy(ChunkConfig(enabled=True, chunk_size=4096))
        got = policy.prefill_budget(pending_prefill_tokens=100,
                                    running_decodes=0, budget=TokenBudget())
        self.assertEqual(got, 100)

    def test_queue_aware_shrinks_under_congestion(self):
        policy = QueueAwareChunkPolicy(
            ChunkConfig(enabled=True, chunk_size=4096, max_num_batched_tokens=16384),
            long_queue_threshold=4,
        )
        budget = TokenBudget(max_num_batched_tokens=16384)
        idle = policy.prefill_budget(pending_prefill_tokens=99999,
                                     running_decodes=0, budget=budget)
        busy = policy.prefill_budget(pending_prefill_tokens=99999,
                                     running_decodes=8, budget=budget)
        self.assertLess(busy, idle)

    def test_invalid_ratio_rejected(self):
        with self.assertRaises(ValueError):
            ChunkConfig(decode_reserve_ratio=1.0)

    def test_unknown_policy_name_raises(self):
        with self.assertRaises(ValueError):
            build_chunk_policy("vibes")


class TestRuntimeChunkLimits(unittest.TestCase):
    """运行期上限必须压过 frozen 的 ``ChunkConfig``。

    这一组守的是一个**静默失效**：策略内部用 ``min(config.chunk_size, ...)`` 封顶，
    控制器把 ``chunk_size`` 调大的动作就会被压回原值——决策日志显示"已调大"，
    实际额度一动不动，不报错、账目也自洽。三条策略都必须认运行期上限，
    漏掉任何一条，那条链路上的自适应就是摆设。
    """

    def _budget(self, tokens: int = 8192):
        return TokenBudget(max_num_batched_tokens=tokens)

    def test_fixed_chunk_honours_runtime_limit_above_static_value(self):
        policy = FixedChunkPolicy(ChunkConfig(enabled=True, chunk_size=1024))
        got = policy.prefill_budget(
            pending_prefill_tokens=9999, running_decodes=0,
            budget=self._budget(),
            limits=ChunkLimits(chunk_size=4096),
        )
        self.assertEqual(got, 4096, "控制器调大的分块上限被策略的静态值压回了")

    def test_fixed_chunk_honours_runtime_limit_below_static_value(self):
        # 反向也要成立：控制器收紧时生效，不能只认"放大"。
        policy = FixedChunkPolicy(ChunkConfig(enabled=True, chunk_size=4096))
        got = policy.prefill_budget(
            pending_prefill_tokens=9999, running_decodes=0,
            budget=self._budget(),
            limits=ChunkLimits(chunk_size=1024),
        )
        self.assertEqual(got, 1024)

    def test_limits_none_is_byte_identical_to_static_behaviour(self):
        # 无控制器（limits=None）时，行为必须与引入 ChunkLimits 之前逐字一致。
        policy = FixedChunkPolicy(ChunkConfig(enabled=True, chunk_size=1024))
        self.assertEqual(
            policy.prefill_budget(pending_prefill_tokens=9999, running_decodes=0,
                                  budget=self._budget()),
            1024,
        )
        self.assertEqual(
            policy.prefill_budget(pending_prefill_tokens=9999, running_decodes=0,
                                  budget=self._budget(), limits=ChunkLimits()),
            1024,
        )

    def test_runtime_limit_never_exceeds_step_token_budget(self):
        # 大于单步 token 预算的上限永远兑现不了：策略必须自己夹住，
        # 否则 tokens_prefilled + n_decode 会突破 max_num_batched_tokens。
        policy = FixedChunkPolicy(ChunkConfig(enabled=True, chunk_size=1024))
        got = policy.prefill_budget(
            pending_prefill_tokens=99999, running_decodes=0,
            budget=self._budget(2048),
            limits=ChunkLimits(chunk_size=16384, max_num_batched_tokens=2048),
        )
        self.assertEqual(got, 2048)

    def test_decode_priority_honours_runtime_token_limit(self):
        # cfg.max_num_batched_tokens=8192 是静态值；控制器的 token 预算更大时
        # decode 保留量要按**生效**的预算算，否则保留比例会被静默打折。
        policy = DecodePriorityChunkPolicy(
            ChunkConfig(enabled=True, chunk_size=8192,
                        max_num_batched_tokens=8192, min_decode_seqs=1)
        )
        got = policy.prefill_budget(
            pending_prefill_tokens=99999, running_decodes=0,
            budget=self._budget(16384),
            limits=ChunkLimits(max_num_batched_tokens=16384),
        )
        self.assertEqual(got, 8192, "token 上限没有跟着控制器的建议值走")

    def test_queue_aware_halves_the_runtime_limit_under_congestion(self):
        policy = QueueAwareChunkPolicy(
            ChunkConfig(enabled=True, chunk_size=1024, max_num_batched_tokens=65536),
            long_queue_threshold=4,
        )
        idle = policy.prefill_budget(
            pending_prefill_tokens=99999, running_decodes=0,
            budget=self._budget(65536), limits=ChunkLimits(chunk_size=8192),
        )
        busy = policy.prefill_budget(
            pending_prefill_tokens=99999, running_decodes=8,
            budget=self._budget(65536), limits=ChunkLimits(chunk_size=8192),
        )
        self.assertEqual(idle, 8192)
        self.assertEqual(busy, 4096)

    def test_non_positive_limits_are_rejected(self):
        # 0 不是"上限"，是"本步一个 prefill token 都不给"——那是死锁的配方。
        with self.assertRaises(ValueError):
            ChunkLimits(chunk_size=0)
        with self.assertRaises(ValueError):
            ChunkLimits(max_num_batched_tokens=-1)


class TestController(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(enabled=True, target_tpot_ms=50.0, target_ttft_ms=2000.0,
                    hysteresis=0.10, cooldown_cycles=3,
                    chunk_size_init=4096, chunk_size_min=512, chunk_size_max=8192,
                    chunk_size_step=512,
                    max_batched_tokens_init=8192, max_batched_tokens_min=2048,
                    max_batched_tokens_max=16384, max_batched_tokens_step=1024)
        base.update(kw)
        return ControllerConfig(**base)

    def test_no_change_inside_hysteresis_band(self):
        ctrl = AdaptiveController(self._cfg())
        action = ctrl.observe(step=0, ttft_samples=[1000.0] * 4,
                              tpot_samples=[50.0] * 4)
        self.assertFalse(action.changed)
        self.assertEqual(action.bottleneck, "none")
        self.assertEqual(ctrl.state.chunk_size, 4096)

    def test_shrink_when_tpot_exceeds_target(self):
        ctrl = AdaptiveController(self._cfg())
        action = ctrl.observe(step=0, ttft_samples=[500.0] * 4,
                              tpot_samples=[200.0] * 4)
        self.assertTrue(action.changed)
        self.assertEqual(action.bottleneck, "tpot")
        self.assertLess(ctrl.state.chunk_size, 4096)
        self.assertLess(ctrl.state.max_batched_tokens, 8192)

    def test_grow_when_ttft_exceeds_target_with_headroom(self):
        ctrl = AdaptiveController(self._cfg())
        action = ctrl.observe(step=0, ttft_samples=[5000.0] * 4,
                              tpot_samples=[20.0] * 4)
        self.assertTrue(action.changed)
        self.assertEqual(action.bottleneck, "ttft")
        self.assertGreater(ctrl.state.chunk_size, 4096)

    def test_refuses_to_grow_without_tpot_headroom(self):
        # TTFT 超目标但 TPOT 已贴线时，放大 chunk 会把 TPOT 推过线。
        # 这条测试锁住"不做免费午餐式优化"的取舍判断。
        ctrl = AdaptiveController(self._cfg())
        action = ctrl.observe(step=0, ttft_samples=[5000.0] * 4,
                              tpot_samples=[50.0] * 4)
        self.assertFalse(action.changed)
        self.assertEqual(action.bottleneck, "ttft")
        self.assertIn("余量", action.reason)

    def test_cooldown_prevents_consecutive_changes(self):
        ctrl = AdaptiveController(self._cfg(cooldown_cycles=5))
        first = ctrl.observe(step=0, ttft_samples=[500.0] * 4, tpot_samples=[200.0] * 4)
        self.assertTrue(first.changed)
        second = ctrl.observe(step=1, ttft_samples=[500.0] * 4, tpot_samples=[200.0] * 4)
        self.assertFalse(second.changed)
        self.assertEqual(ctrl.state.n_skipped_cooldown, 1)

    def test_state_stays_within_bounds(self):
        ctrl = AdaptiveController(self._cfg(chunk_size_step=4096))
        for step in range(0, 200, 10):
            ctrl.observe(step=step, ttft_samples=[500.0] * 4,
                         tpot_samples=[900.0] * 4)
        self.assertGreaterEqual(ctrl.state.chunk_size, 512)
        self.assertGreaterEqual(ctrl.state.max_batched_tokens, 2048)
        self.assertGreater(ctrl.state.n_clamped, 0)

    def test_disabled_controller_never_moves(self):
        ctrl = AdaptiveController(self._cfg(enabled=False))
        action = ctrl.observe(step=0, ttft_samples=[1e6], tpot_samples=[1e6])
        self.assertFalse(action.changed)
        self.assertEqual(ctrl.state.chunk_size, 4096)

    def test_empty_observation_is_ignored(self):
        ctrl = AdaptiveController(self._cfg())
        action = ctrl.observe(step=0, ttft_samples=[], tpot_samples=[])
        self.assertFalse(action.changed)

    def test_replay_is_deterministic(self):
        # 离线复放必须可重放，否则"在历史 telemetry 上验证控制器"这件事就不成立
        cfg = self._cfg()
        obs = [
            {"step": i, "ttft_samples": [3000.0] * 4, "tpot_samples": [80.0] * 4}
            for i in range(0, 40, 4)
        ]
        a = replay_decisions(AdaptiveController(cfg), obs)
        b = replay_decisions(AdaptiveController(cfg), obs)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
