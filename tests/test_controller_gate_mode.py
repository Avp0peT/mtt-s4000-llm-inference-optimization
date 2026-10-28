"""OPT-02 / 路线 B2：**门限形式**与**判据覆盖率**。

为什么单独一组测试
------------------
`AdaptiveController` 的 A/B 两个分支都建立在"判据读数**可信**"这个前提上。
S14B2 的现读产物显示这个前提在两处被悄悄破坏过，两处都**不报错**：

1. ★ **纯 decode 覆盖为 0 时判据读出 `0.0` 而不是 `None`**。采集侧只在
   "本步**完全没有** prefill"时才推进 decode 时钟（`benchmark/real_backend.py`），
   于是在持续过载段（几乎每步都带 prefill）该时钟根本不动 ⇒ 窗口里 32 个样本
   全为 `0.0` ⇒ `decode_p90_ms = 0.0`，**穿过了** `_decide` 里那道 `is None` 的守卫，
   被读成"TPOT = 0 ms ⇒ 余量 100%"。实测后果：S14B2 三个自适应臂共 **64 次**决策
   在 TTFT = 13.8–190 s 时说"TPOT 尚有 100.0% 余量"并**放大 chunk**。
2. ★ **门限是固定绝对值，而工作点跨多个档位**。滞回带宽 `2h·target` ≈ 19.2 ms
   是常数，工作点却随负载档位变一个数量级（S14B2 控制器**实际读到的**
   `criterion_ms` 在一次 run 内为 63.0 → 147.1 ms）。某个档位的工作点落在带内时，
   决策会随重复之间的小扰动翻转（`EXPERIMENTS.md` §4.3.4：三次重复 TTFT
   481.4 / 481.6 / **652.1**，+43%），而门限本身**看不出来**。

断言一律看**状态、轨迹与决策标签**，不看时间——这些是策略层性质，与硬件无关。
两个"反向对照"（`TestCoverageGuard` 与 `TestGateScaling` 里各一条）专门证明
这两条守卫**有能力**区分对错，而不是恒真。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.controller import (  # noqa: E402
    AdaptiveController,
    ControllerConfig,
    replay_decisions,
)

#: 与 S14B2 实际生效的门限同源（`target_tpot_ms=120` / `hysteresis=0.08`
#: ⇒ 放大 110.4 / 收缩 129.6）；工作点则取 S14B2 控制器**实测读到**的两档
#: （低载 63、过载 146）。用真实数字，避免"挑一组好算的数"把结论调出来。
S14B2 = dict(
    enabled=True,
    kind="threshold",
    signal_mode="layered",
    signal_criterion="decode_p90",
    target_tpot_ms=120.0,
    target_ttft_ms=2000.0,
    hysteresis=0.08,
    cooldown_cycles=0,          # 逐步驱动时冷却只会掩盖方向
    chunk_size_init=1024,
    chunk_size_min=512,
    chunk_size_max=2048,
    chunk_size_step=512,
    max_batched_tokens_init=8192,
    max_batched_tokens_min=2048,
    max_batched_tokens_max=32768,
    max_batched_tokens_step=1024,
)

LOW_WORKPOINT_MS = 63.0
HIGH_WORKPOINT_MS = 146.0


def cfg(**overrides) -> ControllerConfig:
    merged = dict(S14B2)
    merged.update(overrides)
    return ControllerConfig(**merged)


def obs(*, crit: float, ttft: float = 300.0, n: int = 8) -> dict:
    """一条观测：纯 decode 判据 = `crit`，TTFT = `ttft`。

    `ttft=300` 是"健康"（远低于 2160），`ttft=9999` 才让放大分支可达。
    """
    return dict(
        ttft_samples=[ttft] * 4,
        tpot_samples=[max(crit, 1.0)] * 4,   # tpot_p90 与判据同量级，但只在 legacy 下被读
        decode_samples=[crit] * n,
        had_prefill_samples=[False] * n,
    )


def labels(trace: list[dict]) -> list[str]:
    """决策标签序列：changed ⇒ 动作，未改 ⇒ 瓶颈（含 `none`）。"""
    return [f"{t['bottleneck']}:{'CHANGED' if t['changed'] else '-'}" for t in trace]


class TestCoverageGuard(unittest.TestCase):
    """★ 覆盖为 0 ⇒ **不可测**（None），不是 0 ms 的延迟。"""

    def test_all_zero_window_is_unmeasurable_and_does_not_act(self):
        ctrl = AdaptiveController(cfg())
        # TTFT 严重超限（放大分支本来可达），但窗口里**一个纯 decode 覆盖都没有**。
        a = ctrl.observe(step=0, **obs(crit=9999.0, ttft=9999.0, n=4))
        # 把 decode 样本改成全 0 —— 这才是实际踩到的形状（n_decode_samples=32）
        a_zero = ctrl.observe(
            step=8,
            ttft_samples=[9999.0] * 4,
            tpot_samples=[307.0] * 4,
            decode_samples=[0.0] * 32,
            had_prefill_samples=[True] * 32,
        )
        self.assertFalse(a_zero.changed, "覆盖为 0 时不得动作（旧实现会在此放大 chunk）")
        self.assertIn("不可测", a_zero.reason)
        self.assertEqual(ctrl.state.n_decode_samples, 32, "窗口确实有 32 个样本")
        self.assertEqual(ctrl.state.n_decode_covered, 0, "但其中 0 个有纯 decode 覆盖")
        self.assertIsNone(ctrl.state.decode_p90_ms, "★ 必须是 None，不是 0.0")
        self.assertIsNotNone(a, "上一条观测只是为了确认配置可用")

    def test_zero_coverage_is_not_silently_read_as_max_headroom(self):
        """反向对照：同样的 TTFT 超限，**有**覆盖时会放大 ⇒ 守卫不是恒真。"""
        ctrl = AdaptiveController(cfg())
        a = ctrl.observe(step=0, **obs(crit=LOW_WORKPOINT_MS, ttft=9999.0, n=8))
        self.assertTrue(a.changed, "有可信的低 TPOT 读数时，放大分支应当执行")
        self.assertEqual(a.bottleneck, "ttft")
        self.assertIn("余量", a.reason)
        self.assertIsNotNone(ctrl.state.decode_p90_ms)

    def test_zero_coverage_samples_are_excluded_from_the_percentile(self):
        """0.0 = "这段等待没有纯 decode 覆盖"，把它当"0 ms 延迟"会把 P90 往下拽。"""
        ctrl = AdaptiveController(cfg())
        ctrl.observe(
            step=0,
            ttft_samples=[300.0] * 4,
            tpot_samples=[120.0] * 4,
            decode_samples=[0.0] * 10 + [100.0] * 2,
            had_prefill_samples=[True] * 10 + [False] * 2,
        )
        self.assertEqual(ctrl.state.n_decode_samples, 12)
        self.assertEqual(ctrl.state.n_decode_covered, 2)
        self.assertEqual(ctrl.state.decode_p90_ms, 100.0,
                         "★ 必须是 100.0（只看有覆盖的样本），不是被 10 个 0 拽低的值")

    def test_legacy_mode_still_reports_no_such_signal(self):
        """legacy 下字段是 None（**没有这个信号**），与 0（采集了但没覆盖）分开。"""
        ctrl = AdaptiveController(cfg(signal_mode="legacy", signal_criterion="tpot_p90"))
        ctrl.observe(step=0, **obs(crit=63.0, n=8))
        self.assertIsNone(ctrl.state.n_decode_samples)
        self.assertIsNone(ctrl.state.n_decode_covered)
        self.assertIsNone(ctrl.state.decode_p90_ms)


class TestGateScaling(unittest.TestCase):
    """★ 门限按工作点归一化 ⇒ 决策对工作点的**整体缩放**不变。"""

    REL_SEQ = (1.0, 1.4, 2.3, 1.4, 1.0)

    def _run(self, mode: str, workpoint: float) -> tuple[list[str], list[dict]]:
        ctrl = AdaptiveController(cfg(
            gate_mode=mode,
            workload_scale_seed_ms=workpoint,
        ))
        trace = replay_decisions(ctrl, [
            dict(step=i * 8, **obs(crit=workpoint * r, ttft=9999.0, n=8))
            for i, r in enumerate(self.REL_SEQ)
        ])
        return labels(trace), trace

    def test_normalized_gate_is_scale_invariant(self):
        """同一相对轨迹在 63 ms 与 146 ms 两档工作点上必须给出**同一串决策**。"""
        low, _ = self._run("normalized", LOW_WORKPOINT_MS)
        high, _ = self._run("normalized", HIGH_WORKPOINT_MS)
        self.assertEqual(low, high, "归一化门限下决策应与工作点的绝对量级无关")

    def test_absolute_gate_is_not_scale_invariant(self):
        """★ 反向对照：绝对门限**没有**这个性质 —— 证明上一条不是自动成立。

        同一个相对轨迹：146 ms 档上全部越过绝对上限 129.6 ⇒ 整串收缩；
        63 ms 档上大部分低于绝对下限 110.4 ⇒ 以放大为主。
        ⇒ 两串决策不同，而区别**只**来自工作点的绝对量级。这正是要修的现象。
        """
        low, _ = self._run("absolute", LOW_WORKPOINT_MS)
        high, _ = self._run("absolute", HIGH_WORKPOINT_MS)
        self.assertNotEqual(low, high, "绝对门限与工作点档位强耦合")
        # 方向也要钉住：档位越高越偏收缩、越低越偏放大。只断言"不同"是不够的
        # ——两种形式都可能在别的意义上不同，而这里要证明的是**耦合方向**。
        self.assertGreater(high.count("tpot:CHANGED"), low.count("tpot:CHANGED"),
                           "高档位应出现更多 TPOT 收缩")
        self.assertGreater(low.count("ttft:CHANGED"), high.count("ttft:CHANGED"),
                           "低档位应出现更多放大")

    def test_gate_base_is_recorded_so_the_threshold_is_auditable(self):
        _, tr_norm = self._run("normalized", LOW_WORKPOINT_MS)
        _, tr_abs = self._run("absolute", LOW_WORKPOINT_MS)
        self.assertIn("gate_base_ms", tr_norm[0]["signals"],
                      "归一化下必须记下门限基准，否则门限实际取值无从复核")
        self.assertNotIn("gate_base_ms", tr_abs[0]["signals"],
                         "绝对口径不得新增键（否则既有批次轨迹不再逐字节可比）")

    def test_scale_tracks_the_working_point_and_seed_is_respected(self):
        ctrl = AdaptiveController(cfg(
            gate_mode="normalized",
            workload_scale_seed_ms=LOW_WORKPOINT_MS,
            workload_scale_alpha=0.5,
        ))
        self.assertEqual(ctrl.state.workload_scale_ms, LOW_WORKPOINT_MS)
        ctrl.observe(step=0, **obs(crit=HIGH_WORKPOINT_MS, ttft=300.0, n=8))
        # EWMA: 63 + 0.5·(146−63) = 104.5
        self.assertAlmostEqual(ctrl.state.workload_scale_ms, 104.5, places=6)
        self.assertEqual(ctrl.state.n_scale_updates, 1)

    def test_without_a_seed_the_first_measurable_observation_seeds_it(self):
        ctrl = AdaptiveController(cfg(gate_mode="normalized"))
        self.assertIsNone(ctrl.state.workload_scale_ms)
        ctrl.observe(step=0, **obs(crit=LOW_WORKPOINT_MS, ttft=300.0, n=8))
        self.assertEqual(ctrl.state.workload_scale_ms, LOW_WORKPOINT_MS)
        self.assertEqual(ctrl.state.n_scale_updates, 1)

    def test_default_mode_is_absolute_and_is_recorded(self):
        c = cfg()
        self.assertEqual(c.gate_mode, "absolute")
        d = c.describe()
        self.assertEqual(d["gate_mode"], "absolute")
        for key in ("workload_scale_alpha", "workload_scale_seed_ms", "slo_hard_guard"):
            self.assertIn(key, d, "门限形式的组成部分必须进产物，否则事后分不清用的哪种")


class TestGateValidation(unittest.TestCase):
    """拼错必须当场报错：静默退回绝对门限会让"我用了归一化"与"其实没用"长得一样。"""

    def test_illegal_gate_mode_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            cfg(gate_mode="normalised")      # 英式拼写：正是会被静默接受的那类输入
        self.assertIn("gate_mode", str(cm.exception))

    def test_alpha_out_of_range_is_rejected(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                cfg(workload_scale_alpha=bad)

    def test_normalized_requires_layered(self):
        with self.assertRaises(ValueError) as cm:
            cfg(gate_mode="normalized", signal_mode="legacy", signal_criterion="tpot_p90")
        self.assertIn("layered", str(cm.exception))

    def test_seed_must_be_positive(self):
        with self.assertRaises(ValueError):
            cfg(workload_scale_seed_ms=0.0)


class TestSloHardGuard(unittest.TestCase):
    """可选护栏：相对余量够 **不** 等于绝对 SLO 没超。默认关。"""

    def test_guard_off_still_grows_when_relative_headroom_exists(self):
        ctrl = AdaptiveController(cfg(gate_mode="normalized",
                                      workload_scale_seed_ms=HIGH_WORKPOINT_MS))
        # crit 稍低于工作点 ⇒ 相对余量足；但它仍高于绝对上限 129.6。
        a = ctrl.observe(step=0, **obs(crit=120.0, ttft=9999.0, n=8))
        self.assertTrue(a.changed, "护栏默认关 ⇒ 只看相对余量")
        self.assertEqual(ctrl.state.n_vetoed_slo_guard, 0)

    def test_guard_on_vetoes_growth_and_counts_it(self):
        ctrl = AdaptiveController(cfg(gate_mode="normalized",
                                      workload_scale_seed_ms=HIGH_WORKPOINT_MS,
                                      slo_hard_guard=True))
        # 选 crit 落在**两条约束之间**的窄区间：绝对上限 129.6 < crit < 相对上限 134.3。
        # 只有这个区间才能把"相对余量够"与"绝对没超"分开——crit 取 120 时两者都满足，
        # 护栏本就不该触发（那正是它默认关的理由：叠在一起会看不出谁在起作用）。
        a = ctrl.observe(step=0, **obs(crit=132.0, ttft=9999.0, n=8))
        self.assertFalse(a.changed)
        self.assertIn("护栏", a.reason)
        self.assertEqual(ctrl.state.n_vetoed_slo_guard, 1)
        self.assertEqual(ctrl.state.chunk_size, ctrl.state.init_chunk_size)


if __name__ == "__main__":
    unittest.main()
