"""S00 **已知缺口**的特征化用例（`01_PROJECT_PLAN.md` §5.1 / §5.2）。

⚠️ 读法（重要，别误读成"通过 = 没问题"）
----------------------------------------
本文件的用例**断言的是当前（有缺陷的）行为**，作用是：

1. 把"问题确实存在"变成**可执行、可复算**的证据（而不是文档里的一句话）；
2. 当缺口被真正修掉时，这些用例会**失败** ⇒ 强制修复者同时更新它们（不会留下一条
   自称还在缺口状态、而实际已修复的第二处真相）。

⇒ 它们**不是**"正确性通过"。真正的正确性断言在
`tests/test_slo_weave_sparse_lookup.py`（§5.3，已修）等处。

缺口清单与作用范围（阻断面 = **新策略对比**，即 `OPT-04`/S07；**不**阻断 S01–S03 的
输出层/allocator/线性融合这些独立纯逻辑改动）：

| 缺口 | 现象（实测） | 为什么 S00 只登记不改执行路径 |
| --- | --- | --- |
| §5.1 双预算 | `_budget_for` 只看候选**自己**的 `tokens_prefilled>0`，不看本步仍在等 token 的 decode 请求 ⇒ 实测 12 条 2048-token 等待局面下有 **6** 个候选被 prefill 预算放行、而按 decode 预算会不可行 | 修它 = 定义"逐 token 护栏"这一**新策略语义**，属 P7/S07 预注册内容（草案已裁定"逐 token 护栏与请求平均 TPOT 须分开定义"） |
| §5.2 闭环抖动 | `closed_jittered` 把 `scheduled_arrival` 推到未来，请求却在 `step_start` 就入队 ⇒ 实测 **16/16** 请求 `enqueue_ms < arrival_ms` | 真机路径上"实际等待"= 真的空转 GPU；而 `step` 相关抖动使"绝对放行时刻"无定义（相对量 + 移动的 `step_start`）。⇒ 语义必须先在预注册里裁定（Q4 明写"改执行路径须重核 digest"） |
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.cache_policy import PrefixCacheConfig, PrefixCachePolicy  # noqa: E402
from ascend_opt.chunk_policy import (  # noqa: E402
    ChunkConfig, ChunkLimits, DecodePriorityChunkPolicy)
from ascend_opt.cost_model import CostModel  # noqa: E402
from ascend_opt.cost_policy import CostPolicy, CostPolicyConfig  # noqa: E402
from ascend_opt import plan_step  # noqa: E402
from ascend_opt.scheduler import ContinuousBatchingScheduler  # noqa: E402
#: ★ 用**真实的**状态机类型（不是自造替身）：`apply_arm` 读的是它的字段
#: （`.req` / `.prefill_remaining` / `.decode_remaining` / `.last_emit_ms`）。
#: 替身必须把这些字段复刻一遍，而复刻件与真件的漂移**不会报错**——
#: 只会让"建议动作能落实"这条验收验在一个不存在的形状上。
from run_benchmark import _SimRequest  # noqa: E402
from ascend_opt.workload import Request, build_requests  # noqa: E402
from run_benchmark import load_config, simulate  # noqa: E402
from workloads import get_workload  # noqa: E402

BASELINE = _ROOT / "configs" / "baseline" / "default.yaml"
MODEL_V7 = _ROOT / "configs" / "cost_calib" / "model_v7.json"
MODEL_V1 = _ROOT / "configs" / "cost_calib" / "model_v1.json"


class TestDualBudgetDecodeGuardIsNotBypassed(unittest.TestCase):
    """§5.1 **已修**（2026-10-03，用户裁定后）：prefill 预算**只在纯 prefill 步**适用。

    ★ 原缺口的形状（实测，`results/processed/s00_audit.json`）
    ----------------------------------------------------------
    旧 `_budget_for` 只看候选**自己**的 `tokens_prefilled > 0` ⇒ 本步**还有 decode
    在跑（或没被服务）**时，那些 decode 也会被按 **prefill 预算**（2000 ms，而
    decode 只有 100 ms）放行。方向是**放行过松**：逐 token 护栏在这些步上**根本不生效**，
    而产物里 `budget_kind="prefill"` 看上去完全正常。

    ⚠️ 读法变了：本类现在断言的是**修好之后**的行为。
    旧版本断言"缺口仍在"（`relaxed` 非空即通过），于是**修好了它也不会失败** ——
    那种"特征化用例"在缺口被修掉后不会报警，等于把第二处真相留在测试里。
    现在拆成两条：纯 prefill 步按设计放宽（**并对 decode 无风险**），
    混合步**必须**回到 decode 预算。
    """

    MODEL = MODEL_V7
    D_MS = 100.0
    PF_MS = 2000.0

    def _policy(self, **over):
        cfg = dict(enabled=True, step_deadline_ms=self.D_MS, prefill_deadline_ms=self.PF_MS)
        cfg.update(over)
        return CostPolicy(CostPolicyConfig(**cfg), CostModel.load_offline(self.MODEL))

    def _decide(self, pol, *, waiting, decoding):
        return pol.decide(
            scheduler=ContinuousBatchingScheduler(),
            chunk_policy=DecodePriorityChunkPolicy(ChunkConfig(enabled=True)),
            cache=PrefixCachePolicy(PrefixCacheConfig()),
            waiting=waiting, decoding=decoding, prefilling=[],
            limits=ChunkLimits(), chunk_ceiling=2048, budget_ceiling=8192,
            max_num_seqs=8, chunk_enabled=True, kv_len=1024,
        )

    @staticmethod
    def _waiting(n=12):
        return [Request(req_id=f"w{i:03d}", prompt_len=2048, gen_len=8) for i in range(n)]

    @staticmethod
    def _decoding(n=1):
        return [Request(req_id=f"d{i:03d}", prompt_len=64, gen_len=64) for i in range(n)]

    @staticmethod
    def _kinds(d):
        return {c.budget_kind for c in d.candidates if c.predicted_ms is not None}

    # ---------------- 按设计放宽：纯 prefill 步 ----------------

    def test_pure_prefill_step_uses_the_prefill_budget(self):
        """**没有**任何 decode 在跑 ⇒ prefill 预算适用（这是设计，不是缺口）。

        它服务的承诺是 TTFT；此时没有任何 decode token 被推迟。
        """
        d = self._decide(self._policy(), waiting=self._waiting(), decoding=[])
        self.assertEqual(self._kinds(d), {"prefill"})
        relaxed = [c for c in d.candidates if c.predicted_ms and c.feasible
                   and c.budget_kind == "prefill" and c.predicted_ms > self.D_MS]
        self.assertTrue(relaxed, "纯 prefill 步按设计允许超过 decode 预算（服务 TTFT）")
        self.assertTrue(all(c.budget_ms == self.PF_MS for c in relaxed))

    # ---------------- ★ 靶心：混合步不得被 prefill 预算放行 ----------------

    def test_mixed_step_falls_back_to_the_decode_budget(self):
        """★★ **缺口靶心**：本步同时有 decode 在跑 ⇒ **一个候选都不许**按 prefill 预算放行。

        （`decode_at_risk = decode_served or decode_blocked` ⇒ True
          ⇒ 逐 token 护栏必须适用。）
        """
        d = self._decide(self._policy(), waiting=self._waiting(), decoding=self._decoding())
        self.assertEqual(self._kinds(d), {"decode"},
                         "混合步必须只用 decode 预算")
        leaked = [c for c in d.candidates
                  if c.predicted_ms and c.feasible and c.budget_kind == "prefill"]
        self.assertEqual(leaked, [], "混合步里仍有候选被 prefill 预算放行 ⇒ 缺口回来了")

    def test_blocked_decode_also_forces_the_decode_budget(self):
        """★ **受阻 decode**（本步有 decode 请求**没拿到** token）同样必须用 decode 预算。

        ⚠️ 为什么这条走**单元**级而不是场景级：当前主用的
        `DecodePriorityChunkPolicy` **不做 decode 裁剪**（decode 优先占位，全部服务），
        所以"场景级造出受阻 decode"要换一个会裁剪的调度策略才行；
        而**判据本身**（`decode_at_risk` 的第二个来源）必须能独立验证 ——
        它存在的意义正是"**将来**某个会裁剪 decode 的调度器接上来时，
        这个洞不会重新打开"。只测场景级会把这条规则的保护面说小了。
        """
        pol = self._policy()
        pf = 512
        # 有 decode 处于风险中（无论是"被服务"还是"没被服务"）⇒ 必须用 decode 预算
        for label in ("decode_served", "decode_blocked"):
            kind, budget = pol._budget_for(tokens_prefilled=pf, decode_at_risk=True)
            self.assertEqual(kind, "decode", f"{label} ⇒ 必须用 decode 预算")
            self.assertEqual(budget, self.D_MS, "decode 预算必须等于 step_deadline_ms 原值")
        # 没有任何 decode 处于风险中 ⇒ prefill 预算适用
        kind, budget = pol._budget_for(tokens_prefilled=pf, decode_at_risk=False)
        self.assertEqual((kind, budget), ("prefill", self.PF_MS))
        # 没有 prefill 成分 ⇒ 无论风险与否都是 decode 预算
        for risk in (True, False):
            kind, budget = pol._budget_for(tokens_prefilled=0, decode_at_risk=risk)
            self.assertEqual((kind, budget), ("decode", self.D_MS))

    def test_blocked_decode_defaults_to_safe_side(self):
        """`decode_at_risk` 的**默认值**必须是"有风险"（保守），而不是"没风险"。

        默认 `False` 会让任何忘记传参的调用点**静默回到旧缺口**；
        默认 `True` 则最坏只是"该放宽时没放宽"（保守，方向安全）。
        """
        pol = self._policy()
        kind, _ = pol._budget_for(tokens_prefilled=512)      # 不传 decode_at_risk
        self.assertEqual(kind, "decode", "默认必须按保守方向（decode 预算）")

    def test_mixed_step_records_degradation_when_all_candidates_exceed(self):
        """混合步全部超 decode 预算 ⇒ 必须 `degraded=True`（显式降级，不静默放行）。"""
        d = self._decide(self._policy(), waiting=self._waiting(), decoding=self._decoding())
        predicted = [c for c in d.candidates if c.predicted_ms is not None]
        self.assertTrue(predicted)
        if not any(c.feasible for c in predicted):
            self.assertTrue(d.degraded, "全超预算时必须显式标 degraded")
            self.assertIn("每步", d.reason, "报错口径须写清用的是 decode 预算")

    # ---------------- 默认关闭时逐字节不变 ----------------

    def test_default_none_keeps_single_budget(self):
        """默认（`prefill_deadline_ms=None`）⇒ 不存在双预算；既有批次不受影响。"""
        pol = self._policy(prefill_deadline_ms=None)
        d = self._decide(pol, waiting=self._waiting(), decoding=[])
        self.assertEqual(self._kinds(d), {"decode"})

    def test_default_none_still_decode_only_when_mixed(self):
        pol = self._policy(prefill_deadline_ms=None)
        d = self._decide(pol, waiting=self._waiting(), decoding=self._decoding())
        self.assertEqual(self._kinds(d), {"decode"})

    # ---------------- 语义：逐 token 护栏 ≠ 请求平均 TPOT ----------------

    def test_budget_is_the_per_token_guard_not_a_request_average(self):
        """`step_deadline_ms` 是**逐 token 上界**（整步 ≤ 它），不是"请求平均 TPOT"。

        这条把语义钉住：外部 SLO 的取值一个都没动（100/2000 还是那两个数），
        改的只是"哪一步该用哪一个"。若有人日后把它改成"平均"口径，
        那要单独裁定，不能靠改这一行悄悄发生。
        """
        d = self._decide(self._policy(), waiting=[], decoding=self._decoding())
        for c in d.candidates:
            if c.budget_kind == "decode" and c.budget_ms is not None:
                self.assertEqual(c.budget_ms, self.D_MS,
                                 "decode 预算必须就是 step_deadline_ms 原值")


class TestClosedJitteredArrivalSemantics(unittest.TestCase):
    """§5.2 **已修**（2026-10-03，用户裁定后）：到达是**绝对时刻**，请求**不得提前放行**。

    ★ 原缺口的形状（实测，`results/processed/s00_audit.json`）
    ----------------------------------------------------------
    * 抖动由 `(seed, req_id, **step**)` 派生 ⇒ 到达时刻是**调度的函数**，
      跑之前没有定义；
    * 请求仍在 `step_start` 入队 ⇒ 实测 **16/16** 个请求 `enqueue_ms < arrival_ms`
      （最大倒置 24.78 ms ≈ 抖动上限），"到达时刻"只是装饰性字段；
      在真机上它的字面含义是"**请求还没到就已经在队列里等了**"。

    ★ 修法：抖动改由 `(seed, **客户端**, **轮次**)` 派生（两者都是闭模型负载自身的
    属性，与策略无关）⇒ 到达成为**输入**；执行器**等到点才放行**。

    ⚠️ 两条测试分别守住"两个方向"，缺一不可：
      * `test_no_inversion_*` —— 不许提前放行；
      * `test_arrivals_are_still_distinct` —— **也不许**用"把抖动抹平成 0"来让倒置消失。
        （第一版修法就正好犯了这个错：倒置没了，但 16 个 arrival 全变成 0。)
    """

    SPAN = 25.0
    CONC = 4
    N = 16

    @classmethod
    def setUpClass(cls):
        cfg = load_config(BASELINE)
        spec = dataclasses.replace(get_workload("long_context_qa"),
                                   arrival="closed_jittered",
                                   closure_jitter_ms=cls.SPAN,
                                   concurrency=cls.CONC, num_requests=cls.N)
        cls.spec = spec
        cls.result = simulate(cfg, build_requests(spec), spec=spec)
        cls.rows = [m for m in cls.result.metrics
                    if m.arrival_ms is not None and m.enqueue_ms is not None]

    def test_evidence_exists(self):
        self.assertEqual(len(self.rows), self.N, "必须每个请求都有可读时刻，否则证据不成立")

    def test_no_inversion_enqueue_never_precedes_arrival(self):
        """★★ 靶心：**没有任何**请求在到达之前被受理。"""
        inverted = [m for m in self.rows if m.enqueue_ms < m.arrival_ms - 1e-9]
        self.assertEqual(inverted, [],
                         f"仍有 {len(inverted)} 个请求提前入队："
                         f"{[(m.req_id, m.arrival_ms, m.enqueue_ms) for m in inverted[:3]]}")

    def test_arrivals_are_still_distinct(self):
        """★ **反向守卫**：抖动必须仍然生效（不许靠"抹平成 0"来消除倒置）。

        抖动原本的**目的**是把调度差异映射成到达差异（`closed` 下跨重复
        `arrival_ms` 逐位相同 ⇒ 主指标零方差 ⇒ 配对判据失效）。修倒置时若把抖动
        丢掉，就同时废掉了那个目的 —— 而倒置检查**照样通过**。
        """
        uniq = {round(m.arrival_ms, 6) for m in self.rows}
        self.assertEqual(len(uniq), self.N, "16 个请求应有 16 个不同的到达时刻")
        span = max(m.arrival_ms for m in self.rows) - min(m.arrival_ms for m in self.rows)
        self.assertGreater(span, 0.0, "到达跨度必须 > 0（否则抖动没生效）")

    def test_inversion_is_gone_without_flattening(self):
        """把两条判据合成一句话：**没有倒置，且时刻确实被推开了**。"""
        self.assertFalse([m for m in self.rows if m.enqueue_ms < m.arrival_ms - 1e-9])
        self.assertGreater(
            len({round(m.arrival_ms, 6) for m in self.rows}), 1)

    def test_enqueue_delay_is_non_negative_and_bounded_by_the_step(self):
        """`enqueue - arrival` 是**受理延迟**（步粒度）⇒ 必须 ≥ 0，且不该失控。"""
        delays = [m.enqueue_ms - m.arrival_ms for m in self.rows]
        self.assertGreaterEqual(min(delays), 0.0)
        self.assertLess(max(delays), 60_000.0, "受理延迟不该到分钟量级（那说明放行逻辑卡住）")

    def test_plain_closed_still_has_enqueue_at_arrival(self):
        """`closed`（未抖动）不受影响：`enqueue == arrival`（无倒置、无延迟）。"""
        cfg = load_config(BASELINE)
        spec = dataclasses.replace(get_workload("long_context_qa"), num_requests=8)
        result = simulate(cfg, build_requests(spec), spec=spec)
        checked = 0
        for m in result.metrics:
            if m.arrival_ms is None:
                continue
            checked += 1
            self.assertAlmostEqual(m.enqueue_ms, m.arrival_ms, places=6)
        self.assertTrue(checked, "closed 场景必须有可读时刻，否则这条没验到东西")

    def test_zero_span_is_byte_identical_to_closed(self):
        """`closure_jitter_ms=0` ⇒ 与 `closed` **逐字节相同**（默认关闭）。"""
        cfg = load_config(BASELINE)
        base = dataclasses.replace(get_workload("long_context_qa"), num_requests=8)
        z = dataclasses.replace(base, arrival="closed_jittered", closure_jitter_ms=0.0)
        a = simulate(cfg, build_requests(base), spec=base).metrics
        b = simulate(cfg, build_requests(z), spec=z).metrics
        self.assertEqual([(m.req_id, m.arrival_ms, m.enqueue_ms) for m in a],
                         [(m.req_id, m.arrival_ms, m.enqueue_ms) for m in b])


class TestClosureJitterIsScheduleIndependent(unittest.TestCase):
    """★ 抖动的**派生依据**里不许出现 `step`（那正是"到达=调度的函数"的成因）。"""

    def test_signature_has_no_step_parameter(self):
        import inspect

        from real_backend import _closure_jitter_ms as jit

        params = set(inspect.signature(jit).parameters)
        self.assertNotIn("step", params, "抖动**不得**依赖 step")
        self.assertIn("client", params)
        self.assertIn("round_index", params)

    def test_table_is_deterministic(self):
        from real_backend import closure_jitter_by_request

        order = [f"r{i:05d}" for i in range(12)]
        a = closure_jitter_by_request(seed=7, order=order, concurrency=4, span_ms=25.0)
        b = closure_jitter_by_request(seed=7, order=order, concurrency=4, span_ms=25.0)
        self.assertEqual(a, b, "同输入必须给出同一张表（可复现）")

    def test_values_are_in_range_and_decorrelated(self):
        from real_backend import closure_jitter_by_request

        order = [f"r{i:05d}" for i in range(16)]
        tbl = closure_jitter_by_request(seed=0, order=order, concurrency=4, span_ms=25.0)
        self.assertTrue(all(0.0 <= v < 25.0 for v in tbl.values()),
                        "抖动必须落在 [0, span)")
        self.assertEqual(len(set(tbl.values())), 16, "不同 (client, round) 应给出不同抖动")

    def test_client_and_round_binding(self):
        """`client = k % concurrency`、`round = k // concurrency` —— 只由**负载**决定。"""
        from real_backend import closure_jitter_by_request

        order = [f"r{i:05d}" for i in range(8)]
        c2 = closure_jitter_by_request(seed=1, order=order, concurrency=2, span_ms=25.0)
        c4 = closure_jitter_by_request(seed=1, order=order, concurrency=4, span_ms=25.0)
        # r00000 = client 0 / round 0 在两种并发下都一样；r00002 在 c=2 时是
        # (client 0, round 1)，在 c=4 时是 (client 2, round 0) ⇒ 应当不同
        self.assertEqual(c2["r00000"], c4["r00000"])
        self.assertNotEqual(c2["r00002"], c4["r00002"],
                            "并发变了 ⇒ (client, round) 绑定变了 ⇒ 抖动应随之变")

    def test_zero_span_is_all_zero(self):
        from real_backend import closure_jitter_by_request

        tbl = closure_jitter_by_request(seed=0, order=["a", "b"], concurrency=1, span_ms=0.0)
        self.assertEqual(set(tbl.values()), {0.0})

    def test_zero_concurrency_does_not_crash(self):
        """`concurrency=0` 是配置错误，但不该把取模搞崩（按 1 处理并如实返回）。"""
        from real_backend import closure_jitter_by_request

        tbl = closure_jitter_by_request(seed=0, order=["a", "b"], concurrency=0, span_ms=10.0)
        self.assertEqual(len(tbl), 2)

    def test_both_execution_paths_share_one_implementation(self):
        """红线：模拟器与真机必须是**同一个量** ⇒ 共用同一份抖动实现。"""
        import run_benchmark as RB
        import real_backend as RBE

        self.assertIs(RB.closure_jitter_by_request, RBE.closure_jitter_by_request)


class TestRecommendedActionTakesEffect(unittest.TestCase):
    """★ 第三条验收：**建议动作能落实**（离线真实执行入口，不跑真机）。

    前两条验收回答"不许做什么"（不提前放行、不遗漏受阻 decode）；
    这一条回答**"该做的做了没有"** —— 控制器给出 `(chunk, max_tokens)` 建议之后，
    它必须真的进入**下一步的规划**。否则产物里会留下一条"调了参数"的轨迹，
    而调度行为一点没变（本仓踩过同族：`apply_arm` 只接进模拟器 ⇒ 真机上该臂
    一次都没被调用，两臂产物逐项相同却看起来完全正常）。
    """

    @staticmethod
    def _policy():
        return CostPolicy(
            CostPolicyConfig(enabled=True, step_deadline_ms=100.0,
                             prefill_deadline_ms=2000.0),
            CostModel.load_offline(MODEL_V7))

    @staticmethod
    def _waiting(n=12):
        return [_SimRequest(req=Request(req_id=f"w{i:03d}", prompt_len=2048, gen_len=8),
                            decode_remaining=7)
                for i in range(n)]

    def _apply(self, *, shadow: bool):
        from ascend_opt.arm import apply_arm

        return apply_arm(
            kind="cost", policy=self._policy(), shadow=shadow, step=1, now_ms=0.0,
            cur_chunk=2048, cur_max_tokens=8192,
            waiting=self._waiting(), decoding=[], prefilling=[],
            scheduler=ContinuousBatchingScheduler(),
            chunk_policy=DecodePriorityChunkPolicy(ChunkConfig(enabled=True)),
            cache=PrefixCachePolicy(PrefixCacheConfig()),
            limits=ChunkLimits(), max_num_seqs=8, chunk_enabled=True,
        )

    def test_closed_loop_recommendation_is_returned(self):
        chunk, mt, entry = self._apply(shadow=False)
        self.assertIsNotNone(entry, "必须留下轨迹条目")
        d = entry["cost_policy"]
        self.assertTrue(d["changed"], "该场景下策略应当给出一个不同的建议")
        # ★ 闭环：返回的就是建议值（下一步会拿它去规划）
        self.assertEqual((chunk, mt), (d["chunk"], d["max_tokens"]))
        self.assertNotEqual((chunk, mt), (2048, 8192), "确实改了参数")

    def test_shadow_records_but_does_not_apply(self):
        """`shadow=True` ⇒ **只记录、不生效**（与闭环严格区分）。"""
        chunk, mt, entry = self._apply(shadow=True)
        self.assertEqual((chunk, mt), (2048, 8192), "shadow 不得改动当前参数")
        self.assertIsNotNone(entry, "shadow 仍要留下轨迹（否则无法事后核对预测）")

    def test_recommendation_actually_reaches_the_planner(self):
        """★★ 靶心：把建议喂给 `plan_step` ⇒ **规划里确实按新额度分配**。

        这是"落实"的可执行证据：不只看返回值，而是看**规划结果**。
        """
        chunk, mt, _ = self._apply(shadow=False)
        waiting = [st.req for st in self._waiting()]
        plan = plan_step(
            scheduler=ContinuousBatchingScheduler(),
            chunk_policy=DecodePriorityChunkPolicy(ChunkConfig(enabled=True)),
            cache=PrefixCachePolicy(PrefixCacheConfig()),
            waiting=waiting, decoding=[], prefilling=[],
            cur_chunk=chunk, cur_max_tokens=mt,
            max_num_seqs=8, chunk_enabled=True, step=1,
        )
        total = sum(plan.prefill_alloc.values())
        self.assertTrue(plan.prefill_alloc, "应当有请求被分到 prefill 额度")
        self.assertLessEqual(max(plan.prefill_alloc.values()), chunk,
                            "单个请求的 prefill 不得超**新** chunk")
        self.assertLessEqual(total, mt, "总 prefill 不得超**新** max_tokens")
        self.assertEqual(total, plan.tokens_prefilled,
                         "`tokens_prefilled` 必须等于分配之和（账目自洽）")

    def test_recommendation_is_observable_in_the_trace(self):
        """轨迹条目要**可读地**记录建议，且字段能被下游直接取用。"""
        _c, _m, entry = self._apply(shadow=False)
        d = entry["cost_policy"]
        for key in ("changed", "chunk", "max_tokens", "reason"):
            self.assertIn(key, d, f"轨迹里缺 {key}")


if __name__ == "__main__":
    unittest.main()
