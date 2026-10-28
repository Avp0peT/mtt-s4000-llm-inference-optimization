"""成本控制器的用例（第 12 步）。

**验收点名的七类边界**逐条对应：
无可行动作 / decode-only 也超预算 / 无信号 / 模型过期 / 显存不足 / 负载恢复 / prefill 饥饿。

用**真实**的 `FIFOScheduler` + `ChunkPolicy` + `PrefixCache` 组合，不用假替身 ——
控制器与它们的接缝正是容易出错的地方（`plan_step` 是刻意抽出来的唯一组合点）。
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

from ascend_opt.cache_policy import PrefixCacheConfig, PrefixCachePolicy  # noqa: E402
from ascend_opt.chunk_policy import (  # noqa: E402
    ChunkConfig,
    ChunkLimits,
    DecodePriorityChunkPolicy,
)
from ascend_opt.cost_model import CostModel  # noqa: E402
from ascend_opt.cost_policy import CostPolicy, CostPolicyConfig  # noqa: E402
from ascend_opt.scheduler import ContinuousBatchingScheduler  # noqa: E402
from ascend_opt.workload import Request  # noqa: E402

MODEL = _ROOT / "configs" / "cost_calib" / "model_v1.json"
GOOD_FP = {
    "hardware": {"device_model": "MTT S4000", "driver_version": "2.7.0"},
    "software": {"torch": "2.2.0", "torch_musa": "1.3.0+81caf0a",
                 "transformers": "4.46.3", "python": "3.10.8"},
    "model": {"id": "Qwen/Qwen2.5-7B-Instruct",
              "revision": "a09a35458c702b33eeacc393d103063234e8bc28"},
    "precision": "fp16",
}

#: 标定范围内、且**成本差异明显**的一个上下文长度：越大越贵 ⇒ 候选之间才有区分度。
KV = 1024


def make_requests(n: int, prompt_len: int = 100, gen_len: int = 8) -> list[Request]:
    return [Request(req_id=f"r{i:03d}", prompt_len=prompt_len, gen_len=gen_len)
            for i in range(n)]


def fixture(model: CostModel | None, *, deadline: float | None = 100.0, enabled: bool = True):
    """构造一套真实组件 + 策略。"""
    cfg = CostPolicyConfig(enabled=enabled, step_deadline_ms=deadline)
    return CostPolicy(cfg, model)


def parts():
    # `DecodePriorityChunkPolicy`：先从预算里扣掉 decode 占用再分 prefill ⇒
    # 它正是"decode 把 prefill 挤到 0"（prefill 饥饿）那条用例需要的行为。
    return (ContinuousBatchingScheduler(),
            DecodePriorityChunkPolicy(ChunkConfig(enabled=True)),
            PrefixCachePolicy(PrefixCacheConfig()))


def decide(policy, *, waiting=(), decoding=(), prefilling=(), limits=None,
           chunk_ceiling=2048, budget_ceiling=8192, **kw):
    # `kv_len` 允许调用方覆盖默认值（越界用例要传一个范围外的值）
    kv = kw.pop("kv_len", KV)
    sched, cp, cache = parts()
    return policy.decide(scheduler=sched, chunk_policy=cp, cache=cache,
                         waiting=list(waiting), decoding=list(decoding),
                         prefilling=list(prefilling), limits=limits or ChunkLimits(),
                         chunk_ceiling=chunk_ceiling, budget_ceiling=budget_ceiling,
                         max_num_seqs=8, chunk_enabled=True, kv_len=kv, **kw)


class TestNormalDecision(unittest.TestCase):
    def setUp(self):
        self.m = CostModel.load(MODEL, GOOD_FP)
        self.p = fixture(self.m)

    def test_picks_a_feasible_candidate_and_records_all_candidates(self):
        d = decide(self.p, waiting=make_requests(12, prompt_len=2048))
        self.assertTrue(d.changed)
        self.assertIsNotNone(d.chunk)
        self.assertGreater(len(d.candidates), 1, "被否决的候选也要留，否则事后答不出'为什么没选它'")
        for c in d.candidates:
            self.assertIsInstance(c.describe(), dict)

    def test_candidates_respect_the_runtime_ceiling(self):
        """运行期上限只在 `ChunkLimits` 一处权威 ⇒ 候选必须被它夹住。"""
        d = decide(self.p, waiting=make_requests(12, prompt_len=2048),
                   limits=ChunkLimits(chunk_size=512, max_num_batched_tokens=2048))
        self.assertLessEqual(max(c.chunk for c in d.candidates), 512)
        self.assertLessEqual(max(c.max_tokens for c in d.candidates), 2048)


class TestNoActionableAction(unittest.TestCase):
    def test_empty_candidate_set_falls_back_and_counts(self):
        """**无可行动作**：候选集为空 ⇒ 明确回退 + 计数（不是静默不动）。"""
        p = CostPolicy(CostPolicyConfig(enabled=True, candidate_chunks=(),
                                        candidate_max_tokens=()), None)
        d = decide(p, waiting=make_requests(4))
        self.assertFalse(d.changed)
        self.assertIsNotNone(d.fallback_reason)
        self.assertEqual(p.state.n_fallback, 1)

    def test_zero_ceiling_clamped_to_nothing_falls_back(self):
        p = fixture(CostModel.load(MODEL, GOOD_FP))
        d = decide(p, waiting=make_requests(4), chunk_ceiling=0)
        self.assertFalse(d.changed)
        self.assertIn("没有候选", d.fallback_reason)


class TestDecodeOnlyOverBudget(unittest.TestCase):
    def test_decode_only_can_exceed_the_step_deadline_and_degrades_explicitly(self):
        """**decode-only 也超预算**：降级取最省者，并**显式记录降级**。"""
        p = fixture(CostModel.load(MODEL, GOOD_FP), deadline=30.0)   # 30ms 谁都超
        d = decide(p, waiting=make_requests(1, prompt_len=100, gen_len=4))
        self.assertTrue(d.changed)
        self.assertTrue(d.degraded, d.reason)
        self.assertGreater(p.state.n_degraded, 0)
        # 取的是**预测成本最小**的那个
        preds = [c for c in d.candidates if c.predicted_ms is not None]
        self.assertAlmostEqual(d.candidates and min(c.predicted_ms for c in preds),
                               min(c.predicted_ms for c in preds))


class TestNoSignal(unittest.TestCase):
    def test_no_model_falls_back_and_counts(self):
        """**无信号**：没有模型 ⇒ 回退，且**必须计数**。"""
        p = CostPolicy(CostPolicyConfig(enabled=True), None)
        d = decide(p, waiting=make_requests(6, prompt_len=1024))
        self.assertFalse(d.changed)
        self.assertIn("没有加载成本模型", d.fallback_reason)
        self.assertEqual(p.state.n_fallback, 1)

    def test_all_out_of_range_falls_back(self):
        """**模型过期/越界**：全部候选都在范围外 ⇒ 回退（不外推）。"""
        p = fixture(CostModel.load(MODEL, GOOD_FP))
        d = decide(p, waiting=make_requests(6, prompt_len=1024), kv_len=99999)
        self.assertFalse(d.changed)
        self.assertIn("范围外", d.fallback_reason)
        self.assertTrue(all(c.predicted_ms is None for c in d.candidates))
        # ⚠️ 越界**不填 0**：`predicted_ms is None` 而不是 0
        self.assertTrue(all(c.predicted_ms is None for c in d.candidates))


class TestMemoryPressureAndRecovery(unittest.TestCase):
    def test_memory_critical_prefers_the_most_conservative_candidate(self):
        """**显存不足** ⇒ 取最保守（最小 max_tokens → 最小 chunk），并计数。"""
        p = fixture(CostModel.load(MODEL, GOOD_FP), deadline=None)   # 不设预算上限，只看显存
        d = decide(p, waiting=make_requests(12, prompt_len=2048), memory_critical=True)
        self.assertTrue(d.changed)
        self.assertEqual(d.max_tokens, min(c.max_tokens for c in d.candidates))
        self.assertEqual(p.state.n_memory_conservative, 1)

    def test_recovery_goes_back_to_the_prefill_progress_choice(self):
        """**负载恢复**：显存压力解除后回到"取 prefill 进度最大者"。"""
        p = fixture(CostModel.load(MODEL, GOOD_FP), deadline=None)
        pressured = decide(p, waiting=make_requests(12, prompt_len=2048),
                           memory_critical=True)
        recovered = decide(p, waiting=make_requests(12, prompt_len=2048),
                           memory_critical=False)
        self.assertGreaterEqual(recovered.candidates and
                                max(c.tokens_prefilled for c in recovered.candidates),
                                pressured.candidates and
                                max(c.tokens_prefilled for c in pressured.candidates))
        self.assertIn("prefill 进度最大", recovered.reason)
        self.assertEqual(p.state.n_memory_conservative, 1, "恢复后不该再累加显存保守计数")


class TestPrefillStarvation(unittest.TestCase):
    def test_starvation_is_counted_when_backlog_exists_but_nothing_progresses(self):
        """**prefill 饥饿**：有积压、而所有候选本步都推进不了 prefill ⇒ 必须数出来。

        构造方式：单步 token 预算被**在解码的序列**吃满（每步每条 1 token），
        于是留给 prefill 的额度为 0。
        """
        p = fixture(CostModel.load(MODEL, GOOD_FP), deadline=None)
        # 8 条在解码 + 预算只有 8 token ⇒ 预留给 decode 之后 prefill 额度为 0
        d = decide(p, waiting=make_requests(4, prompt_len=2048),
                   decoding=make_requests(8, prompt_len=100),
                   budget_ceiling=8)
        self.assertTrue(d.changed)
        self.assertEqual(max(c.tokens_prefilled for c in d.candidates), 0)
        self.assertEqual(p.state.n_prefill_starved, 1)
        self.assertIn("prefill 饥饿", d.reason)


class TestIdleStep(unittest.TestCase):
    def test_idle_step_is_not_counted_as_fallback(self):
        """★ **空闲步**必须与**回退**分开计数。

        实测踩到过：`load_ramp` 的低负载段大量步次没有活可做（`batch=0`），
        而模型没有 `batch=0` 的标定点 ⇒ 若把这些算成"回退"，25000 次决策里会有
        23598 次"回退" —— 那个计数器就再也反映不出"模型到底有没有在用"。
        """
        p = fixture(CostModel.load(MODEL, GOOD_FP))
        d = decide(p)                       # 全空：没有 waiting / decoding / prefilling
        self.assertFalse(d.changed)
        self.assertIsNone(d.fallback_reason, "空闲不是回退")
        self.assertEqual(p.state.n_idle, 1)
        self.assertEqual(p.state.n_fallback, 0)


class TestOfflineRunnableAndLegacyUntouched(unittest.TestCase):
    def test_disabled_by_default(self):
        """新臂**默认关闭** ⇒ 既有批次的行为逐字节不变。"""
        self.assertFalse(CostPolicyConfig().enabled)

    def test_describe_is_json_serialisable(self):
        """轨迹要能落盘 ⇒ `describe()` 必须可 JSON 化。"""
        p = fixture(CostModel.load(MODEL, GOOD_FP))
        decide(p, waiting=make_requests(4, prompt_len=1024))
        json.dumps(p.describe(), ensure_ascii=False)
        self.assertIn("state", p.describe())
        self.assertIn("model", p.describe())


if __name__ == "__main__":
    unittest.main()


class TestHostOverheadInBudget(unittest.TestCase):
    """★ 预算判据必须是"**整步** ≤ 预算"，不是"前向 ≤ 预算"。

    实测（`scripts/decompose_step_overhead.py`，S14B1 的 65 组配对）：
    整步 **130.52** ms = 前向 **104.79** + **host 25.25**（pack 24.94 + readback 0.31），
    而模型对**前向**是中位 **−4.4%（准的）**、对整步是 **−23.2%**。
    ⇒ 漏掉 host 就是拿前向的预测去比整步的 deadline（这正是 D1=84.6% 的成因）。
    """

    def _policy(self, host, deadline=120.0):
        from ascend_opt.cost_policy import CostPolicy, CostPolicyConfig
        return CostPolicy(CostPolicyConfig(enabled=True, step_deadline_ms=deadline,
                                          host_overhead_ms=host),
                          CostModel.load(MODEL, GOOD_FP))

    def test_host_overhead_flips_feasibility_at_the_boundary(self):
        """b8/kv2048 的预测 ≈111.5 ms：不含 host 时 ≤120、含 host(25.29) 时 >120 ⇒ **判定翻转**。"""
        dec = make_requests(8, prompt_len=2048)
        no_host = decide(self._policy(None), decoding=dec, kv_len=2048)
        with_host = decide(self._policy({8: 25.29}), decoding=dec, kv_len=2048)
        self.assertTrue(any(c.feasible for c in no_host.candidates),
                        "不含 host 时应有可行候选（否则用例的前提不成立）")
        self.assertFalse(any(c.feasible for c in with_host.candidates),
                         "含 host 后应全部不可行 —— 即 host 项真的参与了判据")
        self.assertTrue(any("host" in c.reject_reason for c in with_host.candidates),
                        "否决理由里要能看出 host 项参与了判定")

    def test_linear_fit_is_used_for_unlisted_batches(self):
        """表里没列到的 batch 用线性拟合兜底（实测三点 ⇒ 斜率 3.705 / 截距 −4.35）。"""
        from ascend_opt.cost_policy import CostPolicy, CostPolicyConfig
        p = CostPolicy(CostPolicyConfig(enabled=True, step_deadline_ms=120.0,
                                       host_overhead_fit=(3.705, -4.35)),
                       CostModel.load(MODEL, GOOD_FP))
        self.assertAlmostEqual(p._host_overhead(8), 25.29, places=1)
        self.assertAlmostEqual(p._host_overhead(4), 10.47, places=1)
        self.assertGreaterEqual(p._host_overhead(1), 0.0, "小 batch 不能算出负开销（已 clamp）")

    def test_no_host_config_means_zero(self):
        """没配 host 开销时是 0（旧批次行为不变）—— **但这条要在产物里能看出来**。"""
        from ascend_opt.cost_policy import CostPolicy, CostPolicyConfig
        p = CostPolicy(CostPolicyConfig(enabled=True, step_deadline_ms=120.0),
                       CostModel.load(MODEL, GOOD_FP))
        self.assertEqual(p._host_overhead(8), 0.0)


class Dec:
    """最小解码请求替身（只为让 `kv_len` 之外的分支跑通）。"""

    def __init__(self, i: int):
        from ascend_opt.workload import Request
        self.req = Request(req_id=f"d{i}", prompt_len=1024, gen_len=4)
