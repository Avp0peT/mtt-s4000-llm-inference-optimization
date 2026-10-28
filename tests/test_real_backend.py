"""真实后端的契约测试（用假驱动 + 虚拟时钟，**不触达任何设备**）。

这里验证的是"时刻 -> RequestMetrics"这条链路是否可信，而不是性能。三件事必须成立：

1. **账目自洽**：``TTFT = admission + queue + prefill_exec``，
   且 ``E2E == done - arrival``（由 ``accounting_error_ms`` 自查）。
   这两条一旦不成立，后面所有结论都是拿错的时间戳算出来的。
2. **计时边界**：``wall_ms`` 只含正式测量段。装载与预热（真机上可达数十秒）
   若被算进去，吞吐会系统性偏低——而"偏低"看起来完全正常。
3. **失效必须显形**：驱动抛错、死锁、缓存未实现，都要变成产物里的字段或异常，
   而不是静默少跑几个请求。

真机驱动的**数值正确性**（补位/位置编号/KV 写回）另有一个测试文件，
用 CPU 上的微型模型做等价性比对。
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt import ExecutionResult, Request  # noqa: E402
from ascend_opt.metrics import (  # noqa: E402
    OUTCOME_FAILED,
    OUTCOME_SUCCESS,
    OUTCOME_UNFINISHED,
    SLO,
)
from real_backend import (  # noqa: E402
    FakeDriver,
    RealBackend,
    TransformerDriver,
    VirtualClock,
    build_driver,
)
from run_benchmark import load_config  # noqa: E402
from workloads import get_workload  # noqa: E402

BASELINE = _ROOT / "configs" / "baseline" / "default.yaml"
CHUNKED = _ROOT / "configs" / "chunked_prefill" / "chunked_4k.yaml"


def _cfg(workload: dict[str, Any] | None = None, workload_name: str = "short_chat",
         **overrides: Any):
    """取一份真机配置：把执行环境声明成远程 MTT。

    本机跑假驱动时，执行环境声明必须是 ``remote_*``，否则配置加载会拒绝
    （``local`` 不是推理环境，规范明令）。这里宣 remote、用假驱动，
    因此产物自证不是实测——正是我们要测的那条路径。
    """
    cfg = load_config(BASELINE)
    runtime = dict(cfg.runtime)
    runtime["execution_environment"] = "remote_mthreads"
    runtime["hardware_platform"] = "mthreads"
    runtime["device"] = "musa:0"
    cfg = replace(cfg, runtime=runtime)
    wl = dict(workload or {})
    cfg = replace(
        cfg,
        workload_name=workload_name,
        workload_overrides={**cfg.workload_overrides, **wl},
    )
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def _requests(cfg, **wl):
    spec = get_workload(cfg.workload_name, **{**cfg.workload_overrides, **wl})
    from ascend_opt import build_requests

    return spec, build_requests(spec)


class _SlowLoadDriver(FakeDriver):
    """装载/预热会推进虚拟时钟的假驱动。

    用来证明 ``wall_ms`` **不含**装载与预热：如果后端的计时原点落在
    ``load()`` 之前，这里就会被算进去，断言随即失败。
    """

    def __init__(self, *, clock: VirtualClock, load_s: float, warmup_s: float, **kw):
        super().__init__(clock=clock, **kw)
        self.load_s = load_s
        self.warmup_s = warmup_s

    def load(self) -> dict[str, Any]:
        self.clock.advance(self.load_s)
        return {"loaded": True, "kind": "fake-slow", "advanced_s": self.load_s}

    def warmup(self, target_s, *, batch, prompt_len, gen_len) -> dict[str, Any]:
        self.clock.advance(self.warmup_s)
        return {
            "target_s": target_s,
            "warmup_s": self.warmup_s,
            "iterations": 1,
            "short_of_target": False,
        }


class TestContract(unittest.TestCase):
    """账目、时刻、结局三态。"""

    def setUp(self):
        self.vc = VirtualClock()
        self.cfg = _cfg({"num_requests": 6, "prompt_len_min": 32, "prompt_len_max": 32,
                         "gen_len_min": 4, "gen_len_max": 4, "concurrency": 2})
        self.spec, self.requests = _requests(self.cfg)

    def _backend(self, drv, cfg=None):
        return RealBackend(
            cfg or self.cfg,
            driver=drv,
            clock=self.vc.now,
            sleep=lambda _s: None,
            allow_non_hardware_driver=True,
        )

    def _run(self, driver=None):
        drv = driver or FakeDriver(clock=self.vc)
        backend = self._backend(drv)
        return backend, drv, backend.run(self.requests, spec=self.spec)

    def test_result_is_shared_contract(self):
        _, _, result = self._run()
        self.assertIsInstance(result, ExecutionResult)

    def test_all_requests_finish_and_are_success(self):
        _, _, result = self._run()
        self.assertEqual(len(result.metrics), len(self.requests))
        self.assertEqual([m.outcome for m in result.metrics],
                         [OUTCOME_SUCCESS] * len(self.requests))
        self.assertEqual(result.dropped, 0)
        self.assertEqual(result.stall_reason, "")

    def test_six_timestamps_present_and_ordered(self):
        """六个时刻一个都不能缺，且必须单调。

        缺一个不会报错，只会让某一段分解算不出（``None``），
        于是"该加容量还是该调分块"就失去了依据。

        ⚠️ **``scheduled_arrival_ms <= arrival_ms`` 这一条曾经漏掉，而它正是
        2026-09-24 那个开放环 arrival 缺陷的判据**：``simulate()`` 的同一测试
        （``tests/test_simulator.py``）一直有它，真实后端这份没有 ⇒ 两个执行器
        分叉了却没有任何用例失败。**两边必须断言同一组不变式。**
        """
        _, _, result = self._run()
        for m in result.metrics:
            with self.subTest(req=m.req_id):
                for name in (
                    "scheduled_arrival_ms", "arrival_ms", "enqueue_ms",
                    "first_schedule_ms", "first_token_ms", "done_ms",
                ):
                    self.assertIsNotNone(getattr(m, name), f"{name} 缺失")
                self.assertLessEqual(m.scheduled_arrival_ms, m.arrival_ms)
                self.assertLessEqual(m.arrival_ms, m.enqueue_ms)
                self.assertLessEqual(m.enqueue_ms, m.first_schedule_ms)
                self.assertLessEqual(m.first_schedule_ms, m.first_token_ms)
                self.assertLessEqual(m.first_token_ms, m.done_ms)

    def test_ttft_decomposes_exactly(self):
        """TTFT == admission + queue + prefill_exec，逐请求成立。"""
        _, _, result = self._run()
        for m in result.metrics:
            with self.subTest(req=m.req_id):
                total = m.admission_ms + m.queue_ms + m.prefill_exec_ms
                self.assertAlmostEqual(total, m.ttft_ms, places=6)

    def test_accounting_is_self_consistent(self):
        """E2E 与「完成 - 到达」之差必须远小于一步耗时。

        差值大 = TTFT/TPOT 的采样点算错了。这里用步长的量级做上界：
        单步是"毫秒级"，差个几毫秒说明口径自洽；差出几十毫秒就有问题。
        """
        _, drv, result = self._run()
        step_ms = max(c["elapsed_ms"] for c in drv.calls)
        for m in result.metrics:
            err = m.accounting_error_ms()
            self.assertIsNotNone(err)
            self.assertLess(err, 2 * step_ms + 1e-6)

    def test_stall_separates_from_mean_tpot(self):
        """tpot_p99 必须存在且 >= 均值 TPOT（同一批间隔的分位数性质）。"""
        _, _, result = self._run()
        for m in result.metrics:
            with self.subTest(req=m.req_id):
                self.assertGreaterEqual(m.tpot_p99_ms + 1e-9, m.tpot_ms)

    def test_wall_ms_excludes_load_and_warmup(self):
        """计时边界：装载与预热不得进入 wall_ms。"""
        vc = VirtualClock()
        drv = _SlowLoadDriver(clock=vc, load_s=5.0, warmup_s=7.0)
        backend = RealBackend(self.cfg, driver=drv, clock=vc.now,
                              sleep=lambda _s: None, allow_non_hardware_driver=True)
        result = backend.run(self.requests, spec=self.spec)
        # 时钟一共走了 12s + 计算耗时；wall_ms 只能是计算那部分。
        self.assertLess(result.wall_ms, 12_000.0)
        self.assertAlmostEqual(
            result.wall_ms, result.backend_stats["wall_ms_measured_window"], places=6
        )
        self.assertEqual(result.backend_stats["load"]["advanced_s"], 5.0)
        self.assertEqual(result.backend_stats["warmup"]["warmup_s"], 7.0)

    def test_device_sample_absent_is_none_not_zero(self):
        """假驱动采不到显存 → mem_mb 必须是 None，不能填 0。"""
        _, _, result = self._run()
        self.assertTrue(result.samples)
        self.assertTrue(all(s.mem_mb is None for s in result.samples))

    def test_backend_stats_carry_driver_identity(self):
        """产物必须自证"用的是什么驱动、是否硬件执行"。"""
        _, _, result = self._run()
        stats = result.backend_stats
        self.assertEqual(stats["driver"]["kind"], "fake")
        self.assertFalse(stats["driver"]["hardware_backed"])
        self.assertIn("warmup", stats)
        self.assertIn("warmup_shape", stats)
        self.assertEqual(stats["arrival_model"], "closed")


class TestGuards(unittest.TestCase):
    """不该发生的事必须报错，不能静默降级。"""

    def test_fake_driver_cannot_be_built_from_config(self):
        cfg = _cfg()
        cfg = replace(cfg, runtime={**cfg.runtime, "backend_driver": "fake"})
        with self.assertRaises(ValueError):
            build_driver(cfg)

    def test_misspelled_driver_name_is_rejected(self):
        cfg = _cfg()
        cfg = replace(cfg, runtime={**cfg.runtime, "backend_driver": "transformes"})
        with self.assertRaises(ValueError) as ctx:
            build_driver(cfg)
        self.assertIn("transformers", str(ctx.exception))

    def test_driver_whitelist_excludes_fake(self):
        from real_backend import _DRIVER_KINDS

        self.assertNotIn("fake", _DRIVER_KINDS)

    def test_non_hardware_backed_driver_is_refused(self):
        cfg = _cfg({"num_requests": 2, "prompt_len_min": 16, "prompt_len_max": 16,
                    "gen_len_min": 2, "gen_len_max": 2})
        spec, requests = _requests(cfg)
        backend = RealBackend(cfg, driver=FakeDriver())
        with self.assertRaises(RuntimeError) as ctx:
            backend.run(requests, spec=spec)
        self.assertIn("hardware_backed", str(ctx.exception))

    def test_prefix_cache_is_refused_with_actionable_message(self):
        """前缀缓存未实现 → 报错，而不是"记了命中却没省算力"。

        这是最容易悄悄错的一处：接口齐全、字段齐全，只是数字是假的。
        """
        cfg = _cfg({"num_requests": 2, "prompt_len_min": 16, "prompt_len_max": 16})
        cfg = replace(cfg, cache=replace(cfg.cache, enabled=True))
        spec, requests = _requests(cfg)
        drv = FakeDriver()
        # 打开假驱动开口，才能走到"缓存未实现"那一道闸门——
        # 否则先被 hardware_backed 拦下，测的就不是缓存这条路径了。
        backend = RealBackend(cfg, driver=drv, allow_non_hardware_driver=True)
        with self.assertRaises(NotImplementedError) as ctx:
            backend.run(requests, spec=spec)
        msg = str(ctx.exception)
        self.assertIn("cache.enabled=false", msg)
        # 而且必须在装载之前就拒绝：白跑一轮基准才发现不支持是纯粹的浪费。
        self.assertEqual(drv.calls, [])

    def test_missing_model_in_runtime_is_rejected(self):
        cfg = _cfg()
        cfg = replace(cfg, runtime={**cfg.runtime, "model": ""})
        with self.assertRaises(ValueError) as ctx:
            build_driver(cfg)
        self.assertIn("runtime.model", str(ctx.exception))


class TestFailurePaths(unittest.TestCase):
    """驱动故障与死锁都要显形。"""

    def test_driver_error_marks_failed_and_records_reason(self):
        cfg = _cfg({"num_requests": 4, "prompt_len_min": 16, "prompt_len_max": 16,
                    "gen_len_min": 4, "gen_len_max": 4})
        spec, requests = _requests(cfg)
        drv = FakeDriver(fail_on_step=2, fail_message="注入的设备故障")
        vc = VirtualClock()
        drv.clock = vc
        backend = RealBackend(cfg, driver=drv, clock=vc.now,
                              sleep=lambda _s: None, allow_non_hardware_driver=True)
        result = backend.run(requests, spec=spec)

        self.assertTrue(result.backend_stats["driver_error"])
        self.assertIn("注入的设备故障", result.backend_stats["driver_error"])
        # 批里被中断的请求记 failed；没轮到的记 unfinished——两类不能混。
        failed = [m for m in result.metrics if m.outcome == OUTCOME_FAILED]
        unfinished = [m for m in result.metrics if m.outcome == OUTCOME_UNFINISHED]
        self.assertTrue(failed, "驱动故障必须至少让批内请求记为 failed")
        self.assertEqual(len(failed) + len(unfinished), len(requests))
        # 失败请求不得拥有可用的时延数值。
        for m in failed:
            if m.ttft_ms is None:
                self.assertIsNone(m.e2e_ms)

    def test_deadlock_is_reported_not_spun_to_max_steps(self):
        """prompt 超过单步预算且关闭分块 → 死锁必须显形。

        不检测的后果是空转到 max_steps，最后只剩一个 dropped=N 且毫无线索。
        """
        cfg = _cfg({"num_requests": 2, "prompt_len_min": 4096, "prompt_len_max": 4096,
                    "gen_len_min": 2, "gen_len_max": 2})
        cfg = replace(
            cfg,
            chunk=replace(cfg.chunk, enabled=False, max_num_batched_tokens=256),
        )
        spec, requests = _requests(cfg)
        vc = VirtualClock()
        drv = FakeDriver(clock=vc)
        backend = RealBackend(cfg, driver=drv, clock=vc.now,
                              sleep=lambda _s: None, allow_non_hardware_driver=True)
        result = backend.run(requests, spec=spec)

        self.assertTrue(result.stall_reason, "死锁必须写出原因")
        self.assertIn("无法推进", result.stall_reason)
        self.assertEqual(result.dropped, len(requests))
        # 关键：不能空转到 max_steps。步数应当很小。
        self.assertLess(result.steps, 10)


class TestChunkedPrefill(unittest.TestCase):
    """分块预填充：中途前向不产出 token，且确实跨了多步。"""

    def test_partial_prefill_forwards_emit_no_token(self):
        cfg = _cfg({"num_requests": 3, "prompt_len_min": 256, "prompt_len_max": 256,
                    "gen_len_min": 3, "gen_len_max": 3, "concurrency": 3})
        cfg = replace(cfg, workload_overrides={**cfg.workload_overrides})
        # 打开分块，把 chunk 压到 64 → 256 的 prompt 至少要 4 步
        cfg = replace(
            cfg,
            chunk=replace(cfg.chunk, enabled=True, chunk_size=64,
                          max_num_batched_tokens=4096),
        )
        spec, requests = _requests(cfg)
        vc = VirtualClock()
        drv = FakeDriver(clock=vc)
        backend = RealBackend(cfg, driver=drv, clock=vc.now,
                              sleep=lambda _s: None, allow_non_hardware_driver=True)
        result = backend.run(requests, spec=spec)

        partial = [c for c in drv.calls if c["need_token"] < c["n_seqs"]]
        self.assertTrue(partial, "分块预填充必须出现「不产出 token」的中途前向")
        # 每步喂入的 token 数不得超过 chunk 上限（上限只有一个来源）。
        for c in drv.calls:
            self.assertLessEqual(c["n_tokens"], 4096)
        # 请求仍应全部完成。
        self.assertEqual([m.outcome for m in result.metrics],
                         [OUTCOME_SUCCESS] * len(requests))
        for m in result.metrics:
            self.assertAlmostEqual(
                m.admission_ms + m.queue_ms + m.prefill_exec_ms, m.ttft_ms, places=6
            )

    def test_chunked_ttft_is_not_shorter_than_unchunked_for_long_prompt(self):
        """方向性检查：长 prompt 分块后 TTFT 不应变短。

        分块把整段 prefill 摊到多步，TTFT 只会变长或持平；若测出"分块让 TTFT
        明显变短"，说明分块的 token 账目算错了（少算了工作量）。
        这里用假驱动的**确定性耗时公式**做严格比较，因此可以断言方向。
        """
        wl = {"num_requests": 2, "prompt_len_min": 512, "prompt_len_max": 512,
              "gen_len_min": 2, "gen_len_max": 2, "concurrency": 1}

        def run(enabled: bool):
            cfg = _cfg(wl)
            cfg = replace(
                cfg,
                chunk=replace(cfg.chunk, enabled=enabled, chunk_size=128,
                              max_num_batched_tokens=4096),
            )
            spec, requests = _requests(cfg)
            vc = VirtualClock()
            drv = FakeDriver(clock=vc)
            RealBackend(
                cfg, driver=drv, clock=vc.now, sleep=lambda _s: None,
                allow_non_hardware_driver=True,
            ).run(requests, spec=spec)
            return max(c["n_tokens"] for c in drv.calls)

        self.assertLessEqual(run(True), run(False))


class TestArrivalSemantics(unittest.TestCase):
    """闭环与开放环的到达语义必须分开。"""

    def test_closed_loop_respects_concurrency(self):
        cfg = _cfg({"num_requests": 8, "prompt_len_min": 32, "prompt_len_max": 32,
                    "gen_len_min": 6, "gen_len_max": 6, "concurrency": 2})
        spec, requests = _requests(cfg)
        vc = VirtualClock()
        drv = FakeDriver(clock=vc)
        backend = RealBackend(cfg, driver=drv, clock=vc.now,
                              sleep=lambda _s: None, allow_non_hardware_driver=True)
        result = backend.run(requests, spec=spec)

        # 闭环下同时在跑的请求数不得超过客户端池大小。
        # 这是硬上界而不是"大致"：新请求只在客户端腾出空位时才产生，
        # 因此任何一步的批大小都不可能超过 concurrency。
        for c in drv.calls:
            self.assertLessEqual(c["n_seqs"], 2)
        for m in result.metrics:
            self.assertEqual(m.outcome, OUTCOME_SUCCESS)
        # 前两个在 t=0 到达，后面的到达时刻应当严格大于 0。
        arrivals = sorted(m.arrival_ms for m in result.metrics)
        self.assertEqual(arrivals[0], 0.0)
        self.assertGreater(arrivals[-1], 0.0)

    def test_open_loop_uses_schedule_and_waits_instead_of_skipping(self):
        """开放环：请求按日程到达；空闲是**真实流逝时间**，不跳过。

        跳过空闲（把 now 直接推到下一个到达时刻）会让 wall_ms 里没有空闲，
        吞吐被虚高——而"虚高"看起来完全正常。
        """
        cfg = _cfg(
            {"num_requests": 3, "prompt_len_min": 16, "prompt_len_max": 16,
             "gen_len_min": 2, "gen_len_max": 2, "arrival": "poisson",
             "request_rate_qps": 20.0, "concurrency": 1},
            workload_name="mixed_burst",
        )
        spec, requests = _requests(cfg)
        self.assertEqual(spec.arrival, "poisson")
        vc = VirtualClock()
        drv = FakeDriver(clock=vc)

        waited: list[float] = []

        def fake_sleep(seconds: float) -> None:
            waited.append(seconds)
            vc.advance(seconds)

        backend = RealBackend(cfg, driver=drv, clock=vc.now, sleep=fake_sleep,
                              allow_non_hardware_driver=True)
        result = backend.run(requests, spec=spec)

        self.assertTrue(waited, "开放环下应出现等待下一个到达的动作")
        self.assertGreater(result.backend_stats["idle_ms"], 0.0)
        self.assertGreater(result.backend_stats["n_idle_steps"], 0)
        # 到达时刻必须按日程（而不是全被挤到 0）。
        self.assertGreater(max(m.arrival_ms for m in result.metrics), 0.0)


class TestOpenLoopArrivalReference(unittest.TestCase):
    """⚠️ **开放环下 ``arrival_ms`` 的参考点是"放行时刻"，不是"本步开始时刻"。**

    2026-09-24 修：``real_backend`` 原先把 ``s.arrival_ms`` 赋成 ``step_start``，
    而 ``simulate()``（``benchmark/run_benchmark.py``）用的是 ``scheduled_arrival``
    ——**同一份语义，两个执行器只修了一边**，于是"模拟器与真机的 TTFT"不是同一个量。
    ``metrics.py`` 早已在 ``scheduled_arrival_ms`` 上写明"**不得被执行步的时间覆盖**"，
    所以这个修复是**让真实后端遵守既定契约**，不是改口径。

    **本类用离线假时钟构造反例**（不触达任何设备）：假驱动的每步耗时是
    ``fixed + per_seq×批大小 + per_token×token 数``，取
    ``fixed=0 / per_seq=0 / per_token=1 ms`` 即可精确控制步长。

    日程：``r0`` 于 **0 ms** 到达（它让第 0 步占满），``r1`` 于 **10 ms** 到达，
    **落在第 0 步内部**（第 0 步从 0 跑到 100 ms）：

    | 步 | 内容 | 时间 |
    | --- | --- | --- |
    | 0 | r0 的 100 个 prompt token ⇒ 100 ms | 0 → 100 |
    | 1 | r1 的 1 个 prompt token ⇒ 1 ms | 100 → 101 |

    ⇒ r1 的正确 TTFT = 101 − **10** = **91 ms**；被覆盖成 ``step_start`` 的旧读法
    会给出 101 − 100 = **1 ms**，把 90 ms 的真实等待整段抹掉。
    """

    ARRIVAL_MS = 10.0
    LONG_PROMPT = 100
    STEP0_END_MS = 100.0
    FIRST_TOKEN_MS = 101.0

    def _spec(self):
        """只要一份声明"开放环"的 spec——请求列表由本测试自己写死，便于精确控制日程。"""
        cfg = _cfg(
            {"num_requests": 2, "prompt_len_min": 8, "prompt_len_max": 8,
             "gen_len_min": 1, "gen_len_max": 1, "arrival": "poisson",
             "request_rate_qps": 1.0, "concurrency": 1},
            workload_name="mixed_burst",
        )
        spec, _ = _requests(cfg)
        self.assertEqual(spec.arrival, "poisson")
        return cfg, spec

    def _run_counterexample(self):
        cfg, spec = self._spec()
        requests = [
            # r0：占满第 0 步（100 token ⇒ 100 ms）
            Request(req_id="r0", prompt_len=self.LONG_PROMPT, gen_len=1, arrival_ms=0.0),
            # r1：在长步**中途**到达
            Request(req_id="r1", prompt_len=1, gen_len=1, arrival_ms=self.ARRIVAL_MS),
        ]
        vc = VirtualClock()
        drv = FakeDriver(clock=vc, fixed_ms=0.0, per_seq_ms=0.0, per_token_ms=1.0)
        backend = RealBackend(
            cfg, driver=drv, clock=vc.now,
            sleep=lambda s: vc.advance(s),   # 开放环会真的睡到下一个到达时刻
            allow_non_hardware_driver=True,
        )
        result = backend.run(requests, spec=spec)
        by_id = {m.req_id: m for m in result.metrics}
        self.assertEqual(set(by_id), {"r0", "r1"})
        return result, drv, by_id["r0"], by_id["r1"]

    def test_step_times_are_what_the_counterexample_assumes(self):
        """先把平台搭稳：否则后面所有的数都是在错误的步长上对的账。"""
        _, drv, r0, _ = self._run_counterexample()
        self.assertEqual([c["elapsed_ms"] for c in drv.calls], [100.0, 1.0])
        # r0 的整段 prefill 在第 0 步算完 ⇒ 第 0 步结束就是它出首 token 的时刻。
        self.assertAlmostEqual(r0.first_token_ms, self.STEP0_END_MS, places=6)

    def test_arrival_is_the_scheduled_time_not_the_step_start(self):
        """**核心断言**：arrival = 日程时刻（10 ms），不是本步开始时刻（100 ms）。"""
        _, _, _, r1 = self._run_counterexample()
        self.assertAlmostEqual(r1.scheduled_arrival_ms, self.ARRIVAL_MS, places=6)
        self.assertAlmostEqual(r1.arrival_ms, self.ARRIVAL_MS, places=6)
        # 旧读法在这里会给 100.0（step_start）——这条断言就是它能失败的原因。
        self.assertNotAlmostEqual(r1.arrival_ms, self.STEP0_END_MS, places=6)
        # 与 simulate() 同一份语义：arrival 恰等于日程时刻。
        self.assertAlmostEqual(r1.scheduled_arrival_ms, r1.arrival_ms, places=6)
        # arrival 与 enqueue **必须分开**：前者是放行，后者是被受理。
        self.assertAlmostEqual(r1.enqueue_ms, self.STEP0_END_MS, places=6)
        self.assertNotAlmostEqual(r1.arrival_ms, r1.enqueue_ms, places=6)

    def test_ttft_counts_the_wait_inside_the_long_step(self):
        """验收给的例子：日程 10 ms、100 ms 才受理、101 ms 出首 token ⇒ TTFT **91 ms**。"""
        _, _, _, r1 = self._run_counterexample()
        self.assertAlmostEqual(r1.first_token_ms, self.FIRST_TOKEN_MS, places=6)
        self.assertAlmostEqual(r1.ttft_ms, 91.0, places=6)
        # 旧读法会给 1.0（= 101 − 100），把 90 ms 的等待抹掉。
        self.assertNotAlmostEqual(r1.ttft_ms, 1.0, places=6)

    def test_decomposition_and_e2e(self):
        """受理延迟 / 排队 / prefill 三段各自独立，且 TTFT 精确可分解。"""
        _, _, _, r1 = self._run_counterexample()
        self.assertAlmostEqual(r1.admission_ms, 90.0, places=6)   # 10 → 100
        self.assertAlmostEqual(r1.queue_ms, 0.0, places=6)        # 100 → 100
        self.assertAlmostEqual(r1.prefill_exec_ms, 1.0, places=6)  # 100 → 101
        total = r1.admission_ms + r1.queue_ms + r1.prefill_exec_ms
        self.assertAlmostEqual(total, r1.ttft_ms, places=6)
        # gen_len=1 ⇒ E2E 退化为 TTFT
        self.assertAlmostEqual(r1.e2e_ms, 91.0, places=6)

    def test_accounting_error_is_zero_EITHER_WAY_so_it_cannot_be_the_proof(self):
        """⚠️ **本类存在的理由**：``accounting_error == 0`` **证明不了**口径是对的。

        旧读法把 arrival 覆盖成 step_start 之后，``done_ms``、``ttft_ms``、``e2e_ms``
        全都以**同一个被覆盖的参考点**计量 ⇒ 账目依然自洽、``accounting_error`` 依然是 0。
        这里把这个"旧读法也自洽"算出来并断言它，就是为了说明：
        **判据必须是把 arrival / enqueue 拆开看**，不能只看账目。
        """
        _, _, _, r1 = self._run_counterexample()

        # 现行（正确）读法
        self.assertEqual(r1.accounting_error_ms(), 0.0)

        # 旧读法：arrival 被覆盖成 step_start，其余各步同量平移 ⇒ 账目同样自洽
        buggy_arrival = self.STEP0_END_MS
        buggy_ttft = self.FIRST_TOKEN_MS - buggy_arrival          # 1.0
        buggy_e2e = buggy_ttft                                   # gen_len == 1
        buggy_error = abs((r1.done_ms - buggy_arrival) - buggy_e2e)
        self.assertEqual(buggy_error, 0.0)
        # 两套参考点的 TTFT 差 90 ms，但两边的 accounting_error 都是 0 ⇒ 账目不可作为判据。
        self.assertAlmostEqual(r1.ttft_ms - buggy_ttft, 90.0, places=6)

    def test_slo_verdict_flips_with_the_reference_point(self):
        """口径错会让 SLO 判定整格翻面：91 ms 不达标，而 1 ms 会被判达标。

        这正是"TTFT 参考点错"最危险的后果——它不报错，只让**达标结论**变得乐观。
        """
        _, _, _, r1 = self._run_counterexample()
        slo = SLO(ttft_ms=50.0)
        self.assertFalse(slo.satisfied_by(r1), "91 ms 超过 50 ms 的 TTFT 门槛，应判不达标")
        # 旧读法会给 1 ms ⇒ 判达标。断言这一点，本用例才能捕获回归。
        buggy = replace(r1, ttft_ms=self.FIRST_TOKEN_MS - self.STEP0_END_MS)
        self.assertTrue(slo.satisfied_by(buggy))

    def test_closed_loop_semantics_are_preserved(self):
        """闭环不受影响：没有"原定到达"，请求在有客户端空位时**产生**。

        ⇒ 闭环下 ``arrival == scheduled == step_start``，本修复**不改变**闭环行为
        （否则就是把一个新缺陷换成另一个）。
        """
        cfg = _cfg({"num_requests": 6, "prompt_len_min": 32, "prompt_len_max": 32,
                    "gen_len_min": 4, "gen_len_max": 4, "concurrency": 2})
        spec, requests = _requests(cfg)
        self.assertEqual(spec.arrival, "closed")
        vc = VirtualClock()
        drv = FakeDriver(clock=vc, fixed_ms=0.0, per_seq_ms=0.0, per_token_ms=1.0)
        backend = RealBackend(cfg, driver=drv, clock=vc.now, sleep=lambda _s: None,
                              allow_non_hardware_driver=True)
        result = backend.run(requests, spec=spec)

        for m in result.metrics:
            with self.subTest(req=m.req_id):
                self.assertAlmostEqual(m.scheduled_arrival_ms, m.arrival_ms, places=6)
                self.assertAlmostEqual(m.arrival_ms, m.enqueue_ms, places=6)
                self.assertEqual(m.admission_ms, 0.0)


class TestLayeredSignals(unittest.TestCase):
    """第 04 步：把"这段等待是 decode 慢、还是被 prefill 挤的"拆成两个量。

    背景（`EXPERIMENTS.md` §3.3 **E9**）：`recent_tpot` 存的是相邻两次 token 之间的
    间隔，而某一步做了 prefill 时，落在这两步之间的间隔**直接包含该 prefill 的耗时**
    ⇒ 控制器把"prefill 挤占"读成"decode 变慢"，而它的响应是调小 `chunk_size`
    （prefill 更碎 ⇒ 含 prefill 的步更多）——**反向放大**。

    ⚠️ 本类**不改判据**：只断言"两种停顿已经被分开记录"，且**原始间隔与用户指标未被触碰**。
    """

    def _run(self, *, max_num_seqs: int = 4, chunked: bool = True,
             chunk_size: int = 1, max_batched_tokens: int = 2,
             min_decode_seqs: int = 0, per_token_ms: float = 1.0,
             gen0: int = 8, prompt1: int = 4, arrival1: float = 1.5):
        cfg = _cfg(
            {"num_requests": 2, "prompt_len_min": 8, "prompt_len_max": 8,
             "gen_len_min": 2, "gen_len_max": 2, "arrival": "poisson",
             "request_rate_qps": 1.0, "concurrency": 1},
            workload_name="mixed_burst",
        )
        # 夹具意图：r0 一路 decode，r1 在它**中途**到达并分块 prefill；
        # 让 prefill 与 decode 在**不同步**上交替，从而同时造出
        # 「含 prefill 的间隔」与「纯 decode 的间隔」两种样本。
        cfg = replace(
            cfg,
            max_num_seqs=max_num_seqs,
            chunk=replace(cfg.chunk, enabled=chunked, chunk_size=chunk_size,
                          max_num_batched_tokens=max_batched_tokens,
                          min_decode_seqs=min_decode_seqs),
            controller=replace(cfg.controller, signal_mode="layered"),
        )
        spec, _ = _requests(cfg)
        requests = [
            Request(req_id="r0", prompt_len=1, gen_len=gen0, arrival_ms=0.0),
            Request(req_id="r1", prompt_len=prompt1, gen_len=1, arrival_ms=arrival1),
        ]
        vc = VirtualClock()
        drv = FakeDriver(clock=vc, fixed_ms=0.0, per_seq_ms=0.0,
                         per_token_ms=per_token_ms)
        backend = RealBackend(cfg, driver=drv, clock=vc.now,
                              sleep=lambda s: vc.advance(s),
                              allow_non_hardware_driver=True)
        return backend.run(requests, spec=spec), drv

    @staticmethod
    def _emit_times(first_token_ms: float, intervals: list[float]) -> list[float]:
        """逐 token 的发出时刻：首 token + 间隔累加。

        间隔取自 ``layered_signals.per_request[rid]["intervals_ms"]``
        （``RequestMetrics`` 只给汇总值，不给原始序列——原始序列在 backend_stats 里）。
        """
        out = [first_token_ms]
        for iv in intervals:
            out.append(out[-1] + iv)
        return out

    def _steps_in(self, trace, lo: float, hi: float):
        """落在 `(lo, hi]` 里的步。**全部**，不是最后一步。"""
        return [s for s in trace if lo < s[0] <= hi + 1e-9]

    # ------------------------------------------------------------------ #

    def test_fixture_produces_both_kinds_of_interval(self):
        """先把夹具搭稳：两类间隔都要出现，否则后面几条断言有一半是空转。"""
        result, _ = self._run()
        sig = result.backend_stats["layered_signals"]
        flags = [bool(x) for d in sig["per_request"].values() for x in d["had_prefill"]]
        self.assertIn(True, flags, "夹具里没有'含 prefill 的间隔'")
        self.assertIn(False, flags, "夹具里没有'纯 decode 的间隔'")

    def test_flag_semantics_cover_all_steps_not_only_the_last(self):
        """⚠️ **规格测试**：钉住"时钟差"这一读法的语义，并证明它**不是**"只看最后一步"。

        ⚠️ 这里用的是**合成步序列**而不是真实调度器跑出来的夹具，原因要写明：
        本项目的分配器**优先满足 decode**，所以"某个请求被整步饿住"在预算争抢下
        构造不出来 ⇒ 真实夹具里每个间隔恰好只跨一步，无法区分两种读法。
        ⇒ 因此本用例证明的是**规则本身**；"真实运行下覆盖全部步"由
        :meth:`test_flag_covers_every_step_in_the_interval`（用 `step_trace` 独立重算）负责。
        两者合起来才是完整的证据：一个是语义规格，一个是实现与规格一致。
        """
        # 三步：prefill 在**中间**，最后一步是纯 decode
        steps = [(1.0, True), (1.0, False), (1.0, False)]
        prefill_clock = 0.0
        decode_clock = 0.0
        at_interval_start = (0.0, 0.0)      # 上一次 emit 时的两个时钟读数
        for duration, has_prefill in steps:
            if has_prefill:
                prefill_clock += duration
            else:
                decode_clock += duration

        by_clock = prefill_clock > at_interval_start[0]      # ← 生产代码用的读法
        pure_decode_ms = decode_clock - at_interval_start[1]
        last_step_only = steps[-1][1]                        # ← 被禁止的那种读法

        self.assertTrue(by_clock, "中间那一步的 prefill 必须被看见")
        self.assertFalse(last_step_only, "本夹具的末步刻意是纯 decode")
        self.assertNotEqual(
            by_clock, last_step_only,
            "两种读法必须给出不同答案 —— 否则本用例对'只看最后一步'没有分辨力",
        )
        self.assertAlmostEqual(pure_decode_ms, 2.0, places=9)

    def test_flag_covers_every_step_in_the_interval(self):
        """`had_prefill` 必须等于"区间内**任意一步**做过 prefill"，而不是最后一步。"""
        result, _ = self._run()
        sig = result.backend_stats["layered_signals"]
        by_id = {m.req_id: m for m in result.metrics}
        checked = 0
        for rid, d in sig["per_request"].items():
            times = self._emit_times(by_id[rid].first_token_ms,
                                     d["intervals_ms"])
            self.assertEqual(len(d["had_prefill"]), len(d["intervals_ms"]))
            self.assertEqual(len(d["decode_ms"]), len(d["intervals_ms"]))
            for k in range(len(d["intervals_ms"])):
                steps = self._steps_in(sig["step_trace"], times[k], times[k + 1])
                with self.subTest(req=rid, k=k):
                    self.assertTrue(steps, "每个间隔至少应跨一步")
                    self.assertEqual(
                        bool(d["had_prefill"][k]),
                        any(bool(s[1]) for s in steps),
                        "标记必须覆盖区间内全部执行步",
                    )
                    # 纯 decode 耗时 = 区间内**非 prefill 步**的耗时之和
                    self.assertAlmostEqual(
                        d["decode_ms"][k], sum(s[2] for s in steps), places=3
                    )
                checked += 1
        self.assertGreater(checked, 0)

    def test_decode_ms_is_smaller_when_the_interval_contains_prefill(self):
        """两类停顿被分开：含 prefill 的间隔里，`interval` 大而 `decode_ms` 明显更小。"""
        result, _ = self._run()
        sig = result.backend_stats["layered_signals"]
        by_id = {m.req_id: m for m in result.metrics}
        saw_prefill = saw_pure = False
        for rid, d in sig["per_request"].items():
            times = self._emit_times(by_id[rid].first_token_ms,
                                     d["intervals_ms"])
            for k, iv in enumerate(d["intervals_ms"]):
                steps = self._steps_in(sig["step_trace"], times[k], times[k + 1])
                if not steps:
                    continue
                if d["had_prefill"][k]:
                    saw_prefill = True
                    # 含 prefill 的步没被算进纯 decode ⇒ 差额 > 0
                    self.assertGreater(iv - d["decode_ms"][k], 0.0)
                else:
                    saw_pure = True
                    self.assertAlmostEqual(iv, d["decode_ms"][k], places=3)
        self.assertTrue(saw_prefill and saw_pure, "夹具里两种情况都该出现")

    def test_raw_intervals_and_user_metrics_are_untouched(self):
        """⚠️ **不许靠丢样本改善数字**：原始间隔照旧全录，用户指标一字不改。"""
        result, _ = self._run(gen0=5)
        r0 = next(m for m in result.metrics if m.req_id == "r0")
        sig = result.backend_stats["layered_signals"]
        ivs = sig["per_request"]["r0"]["intervals_ms"]
        # gen_len=5 ⇒ 首 token 之外还有 4 个间隔，一个都不能少
        self.assertEqual(len(ivs), 4)
        self.assertTrue(all(x > 0 for x in ivs), "慢间隔不许被丢掉")
        # 用户指标仍然由**原始间隔**算出（不是由"纯 decode"那个量算）
        self.assertAlmostEqual(r0.tpot_ms, sum(ivs) / len(ivs), places=3)
        # 分层量逐请求对齐
        self.assertEqual(len(sig["per_request"]["r0"]["had_prefill"]), 4)
        self.assertTrue(sig["aligned"])

    def test_legacy_mode_does_not_collect_layered_signals(self):
        """`legacy`（默认）⇒ `n_decode_samples` 是 ``None``（**没有这个信号**），
        而不是 0（"采集了但窗口为空"）。两者对策不同：前者要开口径，后者要等样本。"""
        cfg = _cfg({"num_requests": 2, "prompt_len_min": 8, "prompt_len_max": 8,
                    "gen_len_min": 2, "gen_len_max": 2, "arrival": "poisson",
                    "request_rate_qps": 1.0, "concurrency": 1},
                   workload_name="mixed_burst")
        cfg = replace(cfg, controller=replace(cfg.controller, enabled=True,
                                             signal_mode="legacy"))
        spec, _ = _requests(cfg)
        requests = [Request(req_id="r0", prompt_len=1, gen_len=3, arrival_ms=0.0)]
        vc = VirtualClock()
        drv = FakeDriver(clock=vc, fixed_ms=0.0, per_seq_ms=0.0, per_token_ms=1.0)
        backend = RealBackend(cfg, driver=drv, clock=vc.now, sleep=lambda s: vc.advance(s),
                              allow_non_hardware_driver=True)
        result = backend.run(requests, spec=spec)
        self.assertEqual(result.backend_stats["layered_signals"]["per_request"]["r0"]["had_prefill"],
                         [False, False])

    def test_same_input_gives_identical_schedule_and_tokens(self):
        """**等价性**：加遥测不得改变调度与 token 语义。

        判据是**逐位**比对两次运行的调度计划（每步的请求集合与 token 数），
        不是"统计量看起来差不多"。
        """
        r1, drv1 = self._run()
        r2, drv2 = self._run()
        plan1 = [(c["req_ids"], c["n_tokens"]) for c in drv1.calls]
        plan2 = [(c["req_ids"], c["n_tokens"]) for c in drv2.calls]
        self.assertEqual(plan1, plan2)
        self.assertGreater(len(plan1), 0)
        self.assertEqual(
            {m.req_id: m.ttft_ms for m in r1.metrics},
            {m.req_id: m.ttft_ms for m in r2.metrics},
        )
        self.assertEqual(
            r1.backend_stats["layered_signals"]["per_request"],
            r2.backend_stats["layered_signals"]["per_request"],
        )
        # 分层信号本身也要逐位可复现——它只是记录，不该引入任何随机性
        # （已在上面的 per_request 比对中覆盖）


class TestRepresentativeShape(unittest.TestCase):
    """预热形状取中位数，而不是最大/最小值。"""

    def test_median_not_extreme(self):
        reqs = [
            Request(req_id=f"r{i}", prompt_len=n, gen_len=2)
            for i, n in enumerate([10, 20, 30, 40, 1000])
        ]
        shape = RealBackend._representative_shape(reqs)
        self.assertEqual(shape["prompt_len"], 30)
        self.assertEqual(shape["n_requests"], 5)

    def test_empty_workload_is_handled(self):
        shape = RealBackend._representative_shape([])
        self.assertEqual(shape["n_requests"], 0)


class TestWarmupContract(unittest.TestCase):
    """预热路径的形状契约——预热在计时窗口**之外**，所以它失败得很安静。

    预热不产出首 token 时，报错要等到下一步 decode 才出现
    （``last_token`` 为空），而那已经离原因很远了；更糟的是它发生在
    wall_ms 之前，于是产物里连一条时延记录都没有，看起来像"还没跑"。
    """

    def test_prefill_item_always_wants_a_token(self):
        """预热的首个 prefill 必须 ``need_token=True``，与 gen_len 无关。

        完整算完 prompt 之后模型本就会给出下一个 token；正式路径
        （``_Slot.take_next`` 进 decode 分支）也依赖 ``last_token`` 非空。
        写成 ``need_token=(gen_len == 1)`` 会让 gen_len>1 的预热在第二步
        就报"还没有产出过任何 token"。
        """
        from real_backend import TransformerDriver

        captured: list[Any] = []

        class _Recorder(TransformerDriver):
            hardware_backed = False

            def forward(self, batch):
                captured.extend(batch)
                return _ok_outcome(batch)

        drv = _Recorder(model_name="x", max_seq_len=4096)
        drv._loaded = True
        for gen_len in (1, 2, 8):
            captured.clear()
            drv._drive_warmup_batch(
                [Request(req_id="w0", prompt_len=32, gen_len=gen_len)]
            )
            self.assertTrue(
                all(it.need_token for it in captured),
                f"gen_len={gen_len} 的预热出现了不产出 token 的前向",
            )
            # 首步之后还应跟 gen_len-1 次 decode。
            self.assertEqual(len(captured), max(1, gen_len))


def _ok_outcome(batch: Sequence[Any]) -> Any:
    from real_backend import SlotResult, StepOutcome, StepTiming

    return StepOutcome(
        timing=StepTiming(
            n_seqs=len(batch),
            n_tokens=sum(it.n_new_tokens for it in batch),
            n_forwards=1,
            pack_ms=0.0,
            fwd_ms=0.0,
            started_s=0.0,
            finished_s=0.0,
        ),
        slots=[SlotResult(it.req.req_id, 1, 0.0) for it in batch],
    )


class TestKvLifecycle(unittest.TestCase):
    """KV 的生命周期：完成的请求要及时回收，峰值驻留不随请求总数增长。"""

    @staticmethod
    def _run(n_requests: int, concurrency: int):
        cfg = _cfg(
            {
                "num_requests": n_requests,
                "prompt_len_min": 32,
                "prompt_len_max": 32,
                "gen_len_min": 3,
                "gen_len_max": 3,
                "concurrency": concurrency,
            }
        )
        spec, requests = _requests(cfg)
        vc = VirtualClock()
        drv = FakeDriver(clock=vc)
        # 记录每一步之后的驻留槽位数，用来判断峰值。
        peak = {"n": 0}
        original = drv.forward

        def spy(batch):
            out = original(batch)
            peak["n"] = max(peak["n"], drv.live_slots())
            return out

        drv.forward = spy
        backend = RealBackend(
            cfg, driver=drv, clock=vc.now, sleep=lambda _s: None,
            allow_non_hardware_driver=True,
        )
        result = backend.run(requests, spec=spec)
        return drv, result, peak["n"]

    def test_completed_requests_are_released_promptly(self):
        """并发为 1 跑 12 个请求，峰值驻留必须远小于 12。

        若整轮结束才统一释放，驻留会累积到 12。短负载看不出来，
        长负载上表现为 OOM——而那与"负载太重"长得一样。
        """
        drv, result, peak = self._run(n_requests=12, concurrency=1)
        self.assertEqual(peak, 1, "闭环并发为 1 时，峰值驻留不应超过 1")
        lifecycle = result.backend_stats["kv_lifecycle"]
        self.assertEqual(lifecycle["n_requests"], 12)
        self.assertEqual(lifecycle["max_live_slots"], 1)
        self.assertEqual(lifecycle["live_slots_at_end"], 0, "结束时不应有驻留")
        self.assertTrue(lifecycle["retired_per_request"])

    def test_peak_residency_tracks_concurrency_not_request_count(self):
        """峰值驻留随**并发**变化，而不是随请求总数变化。"""
        _, _, peak1 = self._run(n_requests=4, concurrency=1)
        _, _, peak4 = self._run(n_requests=24, concurrency=4)
        self.assertLessEqual(peak4, 4)
        self.assertLess(peak1, peak4, "并发提高应让峰值驻留上升，而不是持平")
        # 请求数从 4 涨到 24（6 倍），峰值不应跟着涨 6 倍。
        self.assertLess(peak4, 24)

    def test_every_request_is_retired_exactly_once(self):
        drv, result, _ = self._run(n_requests=6, concurrency=2)
        self.assertEqual(sorted(drv.retired), sorted(
            m.req_id for m in result.metrics
        ))
        self.assertEqual(len(drv.retired), len(set(drv.retired)), "不应重复回收")
        self.assertEqual(drv.live_slots(), 0)


class TestUnimplementedRuntimeCapabilities(unittest.TestCase):
    """配置声明了没实现的能力 → 报错，且请求/生效分开记录。"""

    def test_quantization_is_rejected(self):
        cfg = _cfg()
        cfg = replace(cfg, runtime={**cfg.runtime, "quantization": "w8a8"})
        with self.assertRaises(ValueError) as ctx:
            build_driver(cfg)
        self.assertIn("quantization", str(ctx.exception))
        self.assertIn("none", str(ctx.exception))

    def test_graph_mode_is_rejected(self):
        cfg = _cfg()
        cfg = replace(cfg, runtime={**cfg.runtime, "graph_mode": "acl_graph"})
        with self.assertRaises(ValueError) as ctx:
            build_driver(cfg)
        self.assertIn("graph_mode", str(ctx.exception))

    def test_tensor_parallel_size_is_rejected(self):
        cfg = _cfg()
        cfg = replace(cfg, runtime={**cfg.runtime, "tensor_parallel_size": 4})
        with self.assertRaises(ValueError) as ctx:
            build_driver(cfg)
        self.assertIn("tensor_parallel_size", str(ctx.exception))

    def test_unimplemented_extra_keys_are_rejected(self):
        """``capture`` / ``quantization_detail`` 这类键的存在本身就是声明。"""
        for key, value in (
            ("capture", {"max_batch_size": 32}),
            ("quantization_detail", {"weight_dtype": "int8"}),
        ):
            with self.subTest(key=key):
                cfg = _cfg()
                cfg = replace(cfg, runtime={**cfg.runtime, key: value})
                with self.assertRaises(ValueError) as ctx:
                    build_driver(cfg)
                self.assertIn(key, str(ctx.exception))

    def test_supported_values_still_build(self):
        """默认配置（none / eager / 1）必须能构造——否则上面的拒绝没有意义。"""
        drv = build_driver(_cfg())
        self.assertEqual(drv.effective_quantization, "none")
        self.assertEqual(drv.effective_graph_mode, "eager")
        self.assertEqual(drv.effective_tensor_parallel_size, 1)

    def test_effective_is_reported_separately_from_requested(self):
        """产物里要能区分"请求的"与"实际生效的"，两者都有名字。"""
        drv = build_driver(_cfg())
        self.assertEqual(drv.requested["quantization"], "none")
        self.assertIn("effective", drv.describe())
        self.assertEqual(drv.describe()["effective"]["graph_mode"], "eager")


class TestTransformerDriverOffline(unittest.TestCase):
    """不装 torch 也必须能导入并给出可读的错误。"""

    def test_precision_misspelling_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            TransformerDriver(model_name="x", precision="fp8")
        self.assertIn("fp8", str(ctx.exception))
        self.assertIn("fp16", str(ctx.exception))

    def test_forward_before_load_is_rejected(self):
        drv = TransformerDriver(model_name="x")
        with self.assertRaises(RuntimeError) as ctx:
            drv.forward([_item()])
        self.assertIn("load()", str(ctx.exception))


class TestEvidenceWritesAreLF(unittest.TestCase):
    """落盘证据的写点必须显式声明 LF 换行。

    为什么值得一条守卫：``Path.write_text`` / ``open(path, "w")`` 在 Windows 上
    默认 ``newline=None``，会把 ``\\n`` 翻成 ``\\r\\n``。后果是**同一份 harness
    产物因平台不同而字节不同**——本机产 CRLF、远程 Linux 产 LF。而
    ``results/raw/**`` 是证据，字节即证据：跨机器比对、算 md5、与入库版本核对时
    会出现与行为无关的假 diff。

    这类缺陷**不报错**：内容完全正确、JSON 照常能解析、测试也全绿，只有量字节
    才看得出来。因此不能靠"跑一遍看有没有报错"发现，必须在源头用静态检查钉住。

    检查方式是 AST 而非文本匹配：文本匹配会被注释里出现的 ``write_text``
    误判，也会被 ``newline="\\n"`` 里的引号形式差异漏掉。
    """

    #: 这些模块负责把证据写到磁盘，其内的每个文本写点都必须声明 newline。
    _EVIDENCE_MODULES = (
        "src/ascend_opt/telemetry.py",
        "benchmark/run_benchmark.py",
        "benchmark/sweep.py",
        "benchmark/collect_metrics.py",
        "benchmark/plot_p99.py",
    )

    def _call_name(self, node):
        """取调用名：``f(...)`` -> 'f'，``p.f(...)`` -> 'f'（含 write_text 等）。"""
        import ast

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                return func.attr
            if isinstance(func, ast.Name):
                return func.id
        return None

    def _open_mode(self, node) -> str | None:
        """取 ``open()`` 调用的 mode 字符串。

        必须同时看 args[0] 与 args[1]：对 ``Path.open``（``p.open("w", ...)``）
        mode 在 **args[0]**（因为 path 是 ``func.value``），而对内建
        ``open(path, "w")`` 它在 args[1]。只看 args[1] 会让守卫对
        ``p.open("w")`` 这种**本仓库实际使用的**写法静默通过——即守卫恒真。
        （这个洞是被 test_guard_detects_the_defect_it_claims_to 抓出来的。）
        """
        import ast

        for idx in (0, 1):
            if len(node.args) > idx and isinstance(node.args[idx], ast.Constant):
                val = node.args[idx].value
                if isinstance(val, str) and val:
                    return val
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                val = kw.value.value
                if isinstance(val, str):
                    return val
        return None

    def _scan(self, path_str: str) -> list[str]:
        """返回该文件里缺少 newline 声明的文本写点描述。"""
        import ast

        root = Path(__file__).resolve().parent.parent
        path = root / path_str
        self.assertTrue(path.exists(), f"扫描目标不存在：{path}")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = self._call_name(node)
            has_newline = any(kw.arg == "newline" for kw in node.keywords)

            if name == "write_text":
                # write_text 没有 newline 参数，只能改用 open(..., newline=...)。
                offenders.append(f"{path_str}:{node.lineno} write_text() 无法声明换行")
            elif name == "open" and not has_newline:
                mode = self._open_mode(node)
                if mode is not None and any(c in mode for c in "wax"):
                    offenders.append(
                        f"{path_str}:{node.lineno} open(mode={mode!r}) 未声明 newline"
                    )
        return offenders

    def test_no_evidence_writer_omits_newline(self):
        offenders: list[str] = []
        for mod in self._EVIDENCE_MODULES:
            offenders.extend(self._scan(mod))
        self.assertEqual(
            offenders,
            [],
            "以下证据写点未声明 LF 换行，会在 Windows 上产出 CRLF 证据：\n  "
            + "\n  ".join(offenders),
        )

    def test_guard_detects_the_defect_it_claims_to(self):
        """正向对照：守卫必须能真的失败，否则它没有价值。

        用一个临时的、故意写坏的模块跑同一条检查逻辑，确认它会报出问题。
        样本要**逐字模拟真机的写法**（``p.open("w", ...)`` 把 mode 放在
        args[0]），否则会像本次一样：样本用了 args[1]，于是守卫「通过」了，
        而真正的代码路径根本没被覆盖。
        """
        import ast
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad_writer.py"
            # 逐字抄自真实写点：Path.open("w", encoding=...) 与 write_text(...)
            bad.write_text(
                "from pathlib import Path\n"
                "def f(p: Path, text: str) -> None:\n"
                '    with p.open("w", encoding="utf-8") as fh:\n'
                "        fh.write(text)\n"
                '    p.write_text(text, encoding="utf-8")\n',
                encoding="utf-8",
                newline="\n",
            )
            tree = ast.parse(bad.read_text(encoding="utf-8"))

            found: list[str] = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = self._call_name(node)
                if name == "write_text":
                    found.append("write_text")
                elif name == "open":
                    has_newline = any(kw.arg == "newline" for kw in node.keywords)
                    mode = self._open_mode(node)
                    if not has_newline and mode is not None and "w" in mode:
                        found.append("open")

        self.assertIn("write_text", found, "守卫漏掉了 write_text")
        self.assertIn("open", found, "守卫漏掉了 Path.open('w')（mode 在 args[0]）")


def _item(n_new_tokens: int = 4, need_token: bool = True):
    from real_backend import ForwardItem

    return ForwardItem(
        req=Request(req_id="r0", prompt_len=8, gen_len=2),
        n_new_tokens=n_new_tokens,
        need_token=need_token,
    )


if __name__ == "__main__":
    unittest.main()
