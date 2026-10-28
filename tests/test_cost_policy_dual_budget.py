"""双预算 deadline 的用例（`B1` 草案 §1.4 方案 A / OPT-03R 裁定 Q1）。

**要保的性质**（每条都配反向对照，避免"恒真"式假通过）：

1. **默认不启用**（`prefill_deadline_ms=None`）⇒ 行为与引入双预算**逐字节相同**：
   prefill 步也用 decode 预算。这是"既有批次不变"的硬保证。
2. **启用后**：`tokens_prefilled > 0` 的候选用 `prefill_deadline_ms`；
   纯 decode 候选仍用 `step_deadline_ms`。**归类按候选自己的计划**，无手调权重。
3. **落盘可复算**：每条候选带 `budget_ms` / `budget_kind` ⇒ "双预算真被用上"能**数出来**。
4. **无预测的候选** `budget_ms` 必须是 `None`（**不是 0** —— 0 会被读成"预算为 0，必超"）。
5. ★ **判据确实换档**：同一批候选，在"启用双预算且 prefill 预算更大"时
   应有候选从 `feasible=False` 翻成 `True`（这是本改动的**收益机制**，
   若它不出现，说明双预算只是记了个字段而没改变任何决策）。

用**真实**的 `FIFOScheduler` + `ChunkPolicy` + `PrefixCache` 组合（与
`tests/test_cost_policy.py` 同款），不用假替身 —— 控制器与它们的接缝正是易错处。
"""

from __future__ import annotations

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
from ascend_opt.cost_policy import (  # noqa: E402
    CandidateEvaluation,
    CostPolicy,
    CostPolicyConfig,
)
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

KV = 1024


def make_requests(n: int, prompt_len: int = 100, gen_len: int = 8) -> list[Request]:
    return [Request(req_id=f"r{i:03d}", prompt_len=prompt_len, gen_len=gen_len)
            for i in range(n)]


def make_policy(*, deadline: float | None = 100.0,
                prefill_deadline: float | None = None,
                model: CostModel | None = None,
                load_model: bool = True) -> CostPolicy:
    cfg = CostPolicyConfig(enabled=True, step_deadline_ms=deadline,
                           prefill_deadline_ms=prefill_deadline)
    m = model if model is not None else (CostModel.load(MODEL, GOOD_FP) if load_model else None)
    return CostPolicy(cfg, m)


def parts():
    return (ContinuousBatchingScheduler(),
            DecodePriorityChunkPolicy(ChunkConfig(enabled=True)),
            PrefixCachePolicy(PrefixCacheConfig()))


def decide(policy, *, waiting=(), decoding=(), prefilling=(), limits=None,
           chunk_ceiling=2048, budget_ceiling=8192, kv_len: int = KV):
    sched, cp, cache = parts()
    return policy.decide(scheduler=sched, chunk_policy=cp, cache=cache,
                         waiting=list(waiting), decoding=list(decoding),
                         prefilling=list(prefilling),
                         limits=limits or ChunkLimits(),
                         chunk_ceiling=chunk_ceiling, budget_ceiling=budget_ceiling,
                         max_num_seqs=8, chunk_enabled=True, kv_len=kv_len)


class TestBudgetForUnit(unittest.TestCase):
    """`_budget_for` 的**直接**单测：不经过调度器，把归类规则钉死。"""

    def test_default_none_always_uses_decode_budget(self):
        """默认不启用 ⇒ 无论有没有 prefill，都走 decode 预算（§默认不变）。"""
        p = make_policy(deadline=100.0, prefill_deadline=None, load_model=False)
        self.assertEqual(p._budget_for(tokens_prefilled=0), ("decode", 100.0))
        self.assertEqual(p._budget_for(tokens_prefilled=512), ("decode", 100.0))

    def test_enabled_prefill_uses_prefill_budget_only_when_no_decode_at_risk(self):
        """★ 启用双预算：prefill 预算**只在没有 decode 处于风险中**时适用。

        ⚠️ 2026-10-03 修（缺口 §5.1）：旧用例只写 `_budget_for(tokens_prefilled=512)`
        就断言 `("prefill", 2000.0)` —— 它默认了"有 prefill 成分 ⇒ 用 prefill 预算"，
        而那条默认正是缺口本身（本步还有 decode 在跑时，逐 token 护栏被旁路）。
        现在必须显式声明 `decode_at_risk`：
          * `False` ⇒ 纯 prefill 步 ⇒ prefill 预算（服务 TTFT，设计如此）；
          * `True`  ⇒ 有 decode 被推迟 ⇒ **decode 预算**（逐 token 护栏）；
          * 不传参  ⇒ 默认 `True`（保守方向，见 `_budget_for` 的 docstring）。
        """
        p = make_policy(deadline=100.0, prefill_deadline=2000.0, load_model=False)
        self.assertEqual(p._budget_for(tokens_prefilled=0, decode_at_risk=False),
                         ("decode", 100.0))
        self.assertEqual(p._budget_for(tokens_prefilled=512, decode_at_risk=False),
                         ("prefill", 2000.0))
        self.assertEqual(p._budget_for(tokens_prefilled=512, decode_at_risk=True),
                         ("decode", 100.0))
        # 默认值必须落在保守侧
        self.assertEqual(p._budget_for(tokens_prefilled=512), ("decode", 100.0))

    def test_inf_deadline_becomes_inf(self):
        """`step_deadline_ms=None` ⇒ `inf`（无上限），与既有语义一致。"""
        p = make_policy(deadline=None, prefill_deadline=None, load_model=False)
        _kind, b = p._budget_for(tokens_prefilled=0)
        self.assertEqual(_kind, "decode")
        self.assertEqual(b, float("inf"))


class TestRecordedFields(unittest.TestCase):
    """落盘字段的**形状**（不依赖模型，用构造的对象）。"""

    def test_missing_prediction_records_none_not_zero(self):
        """★ 无预测 ⇒ `budget_ms` 是 `None`，**不是 0**。"""
        e = CandidateEvaluation(chunk=512, max_tokens=2048, predicted_ms=None,
                                tokens_prefilled=0, batch=0)
        d = e.describe()
        self.assertIsNone(d["budget_ms"])
        self.assertEqual(d["budget_kind"], "")

    def test_declared_budget_roundtrips(self):
        e = CandidateEvaluation(chunk=256, max_tokens=2048, predicted_ms=120.0,
                                tokens_prefilled=256, batch=4,
                                budget_ms=2000.0, budget_kind="prefill")
        d = e.describe()
        self.assertEqual(d["budget_ms"], 2000.0)
        self.assertEqual(d["budget_kind"], "prefill")


class TestDecisionPathUsesDualBudget(unittest.TestCase):
    """端到端：真实调度器 + 真实模型，看**决策本身**是否改变。"""

    def setUp(self):
        self.m = CostModel.load(MODEL, GOOD_FP)

    def _evals(self, policy):
        # 造一个"必须做 prefill"的局面：一堆 prompt 在等待。
        d = decide(policy, waiting=make_requests(12, prompt_len=2048))
        return d

    def test_default_config_records_decode_kind_only(self):
        """默认（不启用）⇒ 所有候选的 `budget_kind` 都是 `decode`。"""
        p = make_policy(deadline=100.0, prefill_deadline=None, model=self.m)
        d = self._evals(p)
        kinds = {c.budget_kind for c in d.candidates if c.predicted_ms is not None}
        self.assertTrue(kinds, "至少要有一个带预测的候选，否则本用例没验到东西")
        self.assertEqual(kinds, {"decode"},
                         "默认不启用双预算 ⇒ 不应出现 prefill 归类")
        for c in d.candidates:
            if c.predicted_ms is not None:
                self.assertEqual(c.budget_ms, 100.0)

    def test_enabled_dual_budget_labels_prefill_candidates(self):
        """启用 ⇒ 推进了 prefill 的候选标 `prefill`，且预算 = prefill 值。"""
        p = make_policy(deadline=100.0, prefill_deadline=2000.0, model=self.m)
        d = self._evals(p)
        pf = [c for c in d.candidates if c.predicted_ms is not None
              and c.tokens_prefilled > 0]
        self.assertTrue(pf, "本局面应有候选推进 prefill（否则用例前提不成立）")
        for c in pf:
            self.assertEqual(c.budget_kind, "prefill")
            self.assertEqual(c.budget_ms, 2000.0)

    def test_larger_prefill_budget_flips_feasibility(self):
        """★ **收益机制**：prefill 预算更大 ⇒ 至少一个候选从不可行翻成可行。

        这是反向对照 —— 若双预算只记字段而不改变任何决策，本用例会失败。
        """
        p_single = make_policy(deadline=100.0, prefill_deadline=None, model=self.m)
        d_single = self._evals(p_single)
        n_feas_single = sum(1 for c in d_single.candidates if c.feasible)

        p_dual = make_policy(deadline=100.0, prefill_deadline=2000.0, model=self.m)
        d_dual = self._evals(p_dual)
        n_feas_dual = sum(1 for c in d_dual.candidates if c.feasible)

        self.assertGreater(
            n_feas_dual, n_feas_single,
            "prefill 预算放大后应有更多候选可行；若相等，说明双预算没改变决策")

    def test_degraded_reason_mentions_prefill_budget(self):
        """全部超预算时，降级原因里的预算口径必须与**被选候选**一致。"""
        # prefill 预算设得极小 ⇒ 所有 prefill 候选都不可行 ⇒ 走降级分支
        p = make_policy(deadline=100.0, prefill_deadline=1.0, model=self.m)
        d = self._evals(p)
        if d.degraded:
            self.assertIn("预算", d.reason)
            # 不得硬写"每步预算"当被选的是 prefill 步
            chosen = [c for c in d.candidates
                      if c.chunk == d.chunk and c.max_tokens == d.max_tokens]
            if chosen and chosen[0].budget_kind == "prefill":
                self.assertIn("prefill", d.reason)


class TestRegressionDefaultsUnchanged(unittest.TestCase):
    """**回归锚**：默认配置下，双预算改动不得改变任何既有可观测行为。"""

    def test_describe_exposes_both_budgets(self):
        p = make_policy(deadline=100.0, prefill_deadline=None, load_model=False)
        desc = p.describe()
        self.assertEqual(desc["step_deadline_ms"], 100.0)
        self.assertIsNone(desc["prefill_deadline_ms"])
        self.assertIn("candidate_chunks", desc)

    def test_describe_after_enabling(self):
        p = make_policy(deadline=100.0, prefill_deadline=1500.0, load_model=False)
        self.assertEqual(p.describe()["prefill_deadline_ms"], 1500.0)


if __name__ == "__main__":
    unittest.main()
