"""第 13 步要求的**正确性检查**：混合 KV 长度 / 分块后多步 decode / 请求退出与新请求补入。

为什么单独一组：这三个形状是**新执行路径**才走到的（`chunk.enabled=True` + 长短不一的 prompt +
边跑边来新请求）。它们出问题的方式是**账目仍然自洽、只是取错了数**（本仓库已经吃过同类亏），
所以断言要落在**可核对的守恒量**上，而不是"跑通了没报错"。

每条断言都注明它对应的验收条目。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.workload import Request  # noqa: E402
from run_benchmark import load_config, simulate  # noqa: E402

BASELINE = _ROOT / "configs" / "baseline" / "default.yaml"


def cfg_with(*, chunk_enabled: bool, chunk_size: int, max_seqs: int = 8):
    cfg = load_config(BASELINE)
    chunk = type(cfg.chunk)(**{**cfg.chunk.__dict__,
                               "enabled": chunk_enabled, "chunk_size": chunk_size})
    return type(cfg)(**{**cfg.__dict__, "chunk": chunk,
                        "max_num_seqs": max_seqs})


def mixed_requests() -> list[Request]:
    """**混合 KV 长度**：三种 prompt 长度 × 两种 gen 长度，到达时刻错开。"""
    out: list[Request] = []
    i = 0
    for prompt_len in (128, 512, 2048):
        for gen_len in (4, 16):
            for k in range(3):
                out.append(Request(req_id=f"m{i:03d}", prompt_len=prompt_len,
                                   gen_len=gen_len, arrival_ms=float(20 * i)))
                i += 1
    return out


class TestMixedKvLengths(unittest.TestCase):
    """验收：'混合 KV 长度 …… 通过正确性检查'。"""

    def setUp(self):
        self.reqs = mixed_requests()
        self.r = simulate(cfg_with(chunk_enabled=True, chunk_size=512), self.reqs)

    def test_no_request_is_dropped(self):
        self.assertEqual(self.r.dropped, 0, "有请求被丢弃 ⇒ 混合长度下有准入缺陷")

    def test_every_request_completes_with_exact_token_counts(self):
        by_id = {m.req_id: m for m in self.r.metrics}
        self.assertEqual(len(by_id), len(self.reqs), "有请求没出现在明细里（分母被静默改小）")
        for req in self.reqs:
            m = by_id[req.req_id]
            self.assertTrue(m.success, f"{req.req_id} 未成功完成")
            self.assertEqual(m.prompt_len, req.prompt_len)
            self.assertEqual(m.gen_len, req.gen_len)
            self.assertGreaterEqual(m.cached_tokens, 0)
            self.assertLessEqual(m.cached_tokens, req.prompt_len, "缓存命中数不可能超过 prompt")

    def test_per_request_timestamps_are_monotone(self):
        """★ 时间线必须自洽 —— 这是"取错数"最先暴露的地方。"""
        for m in self.r.metrics:
            chain = [("scheduled_arrival_ms", m.scheduled_arrival_ms),
                     ("arrival_ms", m.arrival_ms),
                     ("enqueue_ms", m.enqueue_ms),
                     ("first_schedule_ms", m.first_schedule_ms),
                     ("first_token_ms", m.first_token_ms),
                     ("done_ms", m.done_ms)]
            seen = [(n, v) for n, v in chain if v is not None]
            for (n1, v1), (n2, v2) in zip(seen, seen[1:]):
                self.assertLessEqual(
                    v1, v2 + 1e-9,
                    msg=f"{m.req_id}: {n1}={v1} 晚于 {n2}={v2} ⇒ 时刻记错了")

    def test_accounting_is_self_consistent(self):
        errs = [e for m in self.r.metrics if (e := m.accounting_error_ms()) is not None]
        self.assertTrue(errs)
        self.assertLess(max(errs), 1.0, "E2E 与完成时刻不一致 ⇒ 采样点算错")


class TestChunkedMultiStepDecode(unittest.TestCase):
    """验收：'分块后多步 decode …… 通过正确性检查'。"""

    def test_long_prompt_spans_multiple_prefill_steps(self):
        """分块开启 + 长 prompt ⇒ 首 token 与首次调度之间应有**真实的时间间隔**。"""
        reqs = [Request(req_id="c0", prompt_len=4096, gen_len=8, arrival_ms=0.0)]
        r = simulate(cfg_with(chunk_enabled=True, chunk_size=512), reqs)
        m = r.metrics[0]
        self.assertTrue(m.success)
        self.assertIsNotNone(m.first_schedule_ms)
        self.assertIsNotNone(m.first_token_ms)
        self.assertGreater(m.first_token_ms, m.first_schedule_ms,
                           "4096 的 prompt 用 512 分块不可能一步算完")

    def test_chunked_and_unchunked_both_conserve_tokens(self):
        """★ 开/关分块都必须**token 守恒**（不丢、不重复）—— 这是多步 decode 的核心不变量。"""
        reqs = [Request(req_id=f"c{i}", prompt_len=2048, gen_len=8, arrival_ms=float(i * 30))
                for i in range(4)]
        for enabled in (True, False):
            with self.subTest(chunk_enabled=enabled):
                r = simulate(cfg_with(chunk_enabled=enabled, chunk_size=512), reqs)
                self.assertEqual(r.dropped, 0)
                by_id = {m.req_id: m for m in r.metrics}
                self.assertEqual(len(by_id), len(reqs))
                for req in reqs:
                    m = by_id[req.req_id]
                    self.assertTrue(m.success)
                    self.assertEqual(m.gen_len, req.gen_len)
                    self.assertIsNotNone(m.ttft_ms)
                    # gen_len>1 ⇒ TPOT 必须为正（多步 decode 真的发生了）
                    self.assertGreater(m.tpot_ms, 0.0)


class TestExitsAndArrivals(unittest.TestCase):
    """验收：'请求退出和新请求补入通过正确性检查'。"""

    def setUp(self):
        # 并发上限卡紧 ⇒ 被迫出现"退出腾位 → 新请求补入"
        self.reqs = [Request(req_id=f"e{i}", prompt_len=512, gen_len=4,
                             arrival_ms=float(i * 10))
                     for i in range(20)]
        self.max_seqs = 4
        self.r = simulate(cfg_with(chunk_enabled=True, chunk_size=512,
                                   max_seqs=self.max_seqs), self.reqs)

    def test_slot_limit_never_exceeded(self):
        """★ 序列名额：任何一步的在解码+在 prefill 数都不超过 `max_num_seqs`。"""
        self.assertTrue(self.r.samples)
        for s in self.r.samples:
            self.assertLessEqual(s.running, self.max_seqs,
                                 msg=f"step {s.step}: running={s.running} > {self.max_seqs}")

    def test_the_slot_bound_was_actually_binding(self):
        """★ **非平凡性**：上面那条断言不能是空的。

        若 `running` 从未到过 `max_num_seqs`，那条"不超上限"就是**空断言**
        —— 它会在任何输入下通过，却什么也没证明（本仓库踩过同族坑：
        空输入伪装成合格结果）。
        """
        peak = max(s.running for s in self.r.samples)
        self.assertEqual(peak, self.max_seqs,
                         f"名额上限从未被顶到（峰值 {peak} < {self.max_seqs}）⇒ "
                         "本用例没有真的验证到名额闸门")
        self.assertGreater(max(s.queue_depth for s in self.r.samples), 0,
                           "队列从未排过 ⇒ 退出/补入路径没被走到")

    def test_token_budget_never_exceeded(self):
        """★ 单步 token 预算：`scheduled_tokens` 不得超过生效的 `max_batched_tokens`。"""
        for s in self.r.samples:
            if s.max_batched_tokens:
                self.assertLessEqual(s.scheduled_tokens, s.max_batched_tokens,
                                     msg=f"step {s.step}: {s.scheduled_tokens} > "
                                         f"{s.max_batched_tokens}")

    def test_no_request_is_served_twice(self):
        """每个请求在明细里**恰好出现一次**（重复出现 = 退出后又被当新请求补入）。"""
        ids = [m.req_id for m in self.r.metrics]
        self.assertEqual(len(ids), len(set(ids)), "有请求出现多次")
        self.assertEqual(set(ids), {r.req_id for r in self.reqs})

    def test_all_requests_reach_a_terminal_state(self):
        """退出/补入的账目必须闭合：完成 + 未完成 = 提交总数。"""
        done = sum(1 for m in self.r.metrics if m.success)
        self.assertEqual(done + self.r.dropped, len(self.reqs),
                         "完成数 + 丢弃数 != 提交数 ⇒ 有请求既没完成也没被记为丢弃")

    def test_completion_order_is_not_required_but_start_order_is_fifo(self):
        """退出腾位后，**先到的先起步**（FIFO 队列语义）。"""
        scheduled = [(m.first_schedule_ms, m.scheduled_arrival_ms, m.req_id)
                     for m in self.r.metrics if m.first_schedule_ms is not None]
        self.assertTrue(scheduled)
        # 按首次调度时刻排序后，原定到达时刻**不应递减**（允许相等：同一步起步）
        by_sched = sorted(scheduled)
        arrivals = [a for _, a, _ in by_sched]
        self.assertEqual(arrivals, sorted(arrivals),
                         "先调度了后来才到达的请求 ⇒ 队列语义被破坏")


if __name__ == "__main__":
    unittest.main()
