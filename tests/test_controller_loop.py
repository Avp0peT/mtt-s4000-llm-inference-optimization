"""控制器的闭环与恢复能力。

这一组测试对应三类必须被固定住的行为，它们都是"不会报错但会让自适应名不副实"的失效：

1. **可逆**（不是棘轮）：压力消退后，此前被收缩的维度要能恢复到配置初值。
   只有收缩没有恢复的控制器在负载回落后会把系统永久按在过保守的配置上——
   而那正是"自适应优于静态配置"这个主张最该赢下来的场景。
2. **可行**：控制器不得宣告一个兑现不了的运行点（chunk 大于单步 token 预算）。
   宣告了就会被下游静默夹一次，于是"建议值 vs 生效值"这个检查本身就失效了。
3. **信号要么可信、要么显式为 None**：显存压迫要能否决放大、临界时要能触发收缩；
   缓存命中率高时要能拒绝"为 TTFT 放大分块"；而**未测**的信号不得改变任何决策，
   但必须出现在决策记录里。

断言一律看**状态与轨迹**，不看时间：这些是策略层的性质，与硬件无关。
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
    ControllerState,
    replay_decisions,
)
from ascend_opt.monitor import SystemState  # noqa: E402

# 一份便于推理的配置：初值 4096 / 8192，步长 512 / 1024，上下界都留足空间。
BASE = dict(
    enabled=True,
    target_tpot_ms=50.0,
    target_ttft_ms=2000.0,
    hysteresis=0.10,
    cooldown_cycles=0,         # 默认关冷却：逐步驱动时冷却只会掩盖方向
    chunk_size_init=4096,
    chunk_size_min=512,
    chunk_size_max=16384,
    chunk_size_step=512,
    max_batched_tokens_init=8192,
    max_batched_tokens_min=2048,
    max_batched_tokens_max=32768,
    max_batched_tokens_step=1024,
)


def cfg(**overrides) -> ControllerConfig:
    merged = dict(BASE)
    merged.update(overrides)
    return ControllerConfig(**merged)


# 观测的三档：健康 / TPOT 超目标（decode 瓶颈）/ TTFT 超目标（prefill 瓶颈）
HEALTHY = dict(ttft_samples=[100.0] * 4, tpot_samples=[10.0] * 4)
TPOT_OVER = dict(ttft_samples=[100.0] * 4, tpot_samples=[500.0] * 4)
TTFT_OVER = dict(ttft_samples=[9999.0] * 4, tpot_samples=[5.0] * 4)


class TestNotARatchet(unittest.TestCase):
    """压力消退后必须能恢复——这是本次改动要修的核心缺陷。"""

    def test_tpot_shrink_is_undone_after_pressure_recedes(self):
        ctrl = AdaptiveController(cfg())
        init_chunk, init_tokens = 4096, 8192

        for i in range(5):
            ctrl.observe(step=i, **TPOT_OVER)
        self.assertLess(ctrl.state.chunk_size, init_chunk, "过载时应当收缩")
        self.assertLess(ctrl.state.max_batched_tokens, init_tokens, "过载时应当收缩")
        shrunk_chunk = ctrl.state.chunk_size
        shrunk_tokens = ctrl.state.max_batched_tokens

        for i in range(10, 30):
            ctrl.observe(step=i, **HEALTHY)

        self.assertEqual(ctrl.state.chunk_size, init_chunk)
        self.assertEqual(ctrl.state.max_batched_tokens, init_tokens)
        self.assertGreater(ctrl.state.n_recovered, 0, "恢复分支从未触发")
        self.assertLess(shrunk_chunk, ctrl.state.chunk_size)

    def test_token_budget_is_not_a_one_way_ratchet(self):
        # 单独把 token 预算这一维拿出来锁一遍：它是"只在 TPOT 超标时收缩、
        # 没有放大路径"的那一维，也是本次改动前唯一完全没有恢复路径的参数。
        ctrl = AdaptiveController(cfg())
        for i in range(3):
            ctrl.observe(step=i, **TPOT_OVER)
        floor = ctrl.state.max_batched_tokens
        self.assertLess(floor, 8192)

        for i in range(10, 25):
            ctrl.observe(step=i, **HEALTHY)
        self.assertEqual(
            ctrl.state.max_batched_tokens, 8192,
            "token 预算回不到初值 = 仍然是棘轮",
        )

    def test_recovery_stops_at_the_configured_init_not_at_the_upper_bound(self):
        # 恢复不是探索：越过初值继续放大只会让下一个突发到来时系统更脆弱。
        ctrl = AdaptiveController(cfg())
        for i in range(3):
            ctrl.observe(step=i, **TPOT_OVER)
        for i in range(10, 60):
            ctrl.observe(step=i, **HEALTHY)
        self.assertEqual(ctrl.state.chunk_size, 4096)
        self.assertEqual(ctrl.state.max_batched_tokens, 8192)
        self.assertLess(ctrl.state.chunk_size, ctrl.config.chunk_size_max)

    def test_recovery_is_incremental(self):
        # 一步跳回初值会让"恢复"本身变成扰动源，与冷却机制的目的正好相反。
        ctrl = AdaptiveController(cfg())
        ctrl.observe(step=0, **TPOT_OVER)
        before = ctrl.state.chunk_size
        action = ctrl.observe(step=10, **HEALTHY)
        self.assertTrue(action.changed)
        self.assertEqual(action.bottleneck, "recover")
        self.assertLessEqual(ctrl.state.chunk_size - before, 512)

    def test_healthy_controller_that_never_shrank_stays_put(self):
        # 安全区里不动：没有"此前收缩"可恢复时，健康状态不该触发任何调整。
        ctrl = AdaptiveController(cfg())
        for i in range(20):
            action = ctrl.observe(step=i, **HEALTHY)
        self.assertFalse(action.changed)
        self.assertEqual(action.bottleneck, "none")
        self.assertEqual(ctrl.state.n_recovered, 0)

    def test_recovery_is_reported_with_its_own_bottleneck_label(self):
        # "无压力所以不动"与"无压力所以恢复"必须能分开：
        # 两者都写 none 的话，轨迹看上去会自相矛盾（chunk 变了但瓶颈是 none）。
        ctrl = AdaptiveController(cfg())
        ctrl.observe(step=0, **TPOT_OVER)
        trace = replay_decisions(
            ctrl, [{"step": 10, **HEALTHY}]
        )
        self.assertEqual(trace[0]["bottleneck"], "recover")
        self.assertTrue(trace[0]["changed"])


class TestWorkloadSwitch(unittest.TestCase):
    """负载切换：判据方向必须跟着瓶颈翻面。"""

    def test_ttft_pressure_then_tpot_pressure_reverses_direction(self):
        ctrl = AdaptiveController(cfg())

        for i in range(4):
            ctrl.observe(step=i, **TTFT_OVER)
        grown = ctrl.state.chunk_size
        self.assertGreater(grown, 4096, "TTFT 超目标且有 TPOT 余量时应当放大 chunk")

        action = ctrl.observe(step=10, **TPOT_OVER)
        self.assertEqual(action.bottleneck, "tpot")
        self.assertLess(ctrl.state.chunk_size, grown, "瓶颈翻面后必须反向下调")

    def test_switch_back_to_ttft_pressure_grows_again(self):
        ctrl = AdaptiveController(cfg())
        ctrl.observe(step=0, **TPOT_OVER)
        sunk = ctrl.state.chunk_size
        ctrl.observe(step=5, **TTFT_OVER)
        self.assertGreater(ctrl.state.chunk_size, sunk)

    def test_chunk_and_token_budget_stay_feasible_across_the_switch(self):
        # 两个维度的步长不同（512 / 1024），同时收缩时 chunk 会相对抬起来；
        # 可行性约束必须每一步都成立，否则控制器会长期宣告一个兑现不了的运行点。
        ctrl = AdaptiveController(cfg())
        for step, obs in enumerate(
            [TPOT_OVER] * 12 + [TTFT_OVER] * 12 + [TPOT_OVER] * 12
        ):
            ctrl.observe(step=step * 8, **obs)
            self.assertLessEqual(
                ctrl.state.chunk_size, ctrl.state.max_batched_tokens,
                f"step {step}: chunk 超过单步 token 预算，该运行点无法兑现",
            )

    def test_shrink_keeps_chunk_within_token_budget(self):
        # 一次收缩把 token 预算直接打到下界，chunk 必须跟着降下来。
        ctrl = AdaptiveController(cfg(max_batched_tokens_step=8192))
        self.assertEqual(ctrl.state.max_batched_tokens, 8192)
        ctrl.observe(step=0, **TPOT_OVER)
        self.assertEqual(ctrl.state.max_batched_tokens, 2048)
        self.assertLessEqual(ctrl.state.chunk_size, 2048)
        self.assertEqual(ctrl.state.n_infeasible_repaired, 1)


class TestFeedbackSignals(unittest.TestCase):
    """``s_t`` 的非时延分量：显存压力与缓存命中率。"""

    SIGNALS = dict(
        cache_hit_rate_high=0.5,
        min_available_memory_mb=8192.0,
        critical_available_memory_mb=4096.0,
    )

    def _ctrl(self, **overrides):
        merged = dict(self.SIGNALS)
        merged.update(overrides)
        return AdaptiveController(cfg(**merged))

    @staticmethod
    def _state(**kwargs) -> SystemState:
        return SystemState(step=0, **kwargs)

    def test_memory_pressure_vetoes_growth(self):
        # 可用 6000MB：高于收缩线 4096（不该收缩），低于否决线 8192（不该放大）。
        ctrl = self._ctrl()
        action = ctrl.observe(
            step=0, system_state=self._state(available_memory_mb=6000.0), **TTFT_OVER
        )
        self.assertFalse(action.changed)
        self.assertEqual(action.bottleneck, "ttft")
        self.assertEqual(ctrl.state.n_vetoed_memory, 1)
        self.assertEqual(ctrl.state.chunk_size, 4096)
        self.assertIn("显存", action.reason)

    def test_critical_memory_shrinks_even_when_latency_is_fine(self):
        # 显存临界不是"慢了"，是"快跑不动了"：压在一切时延判据之前。
        ctrl = self._ctrl()
        action = ctrl.observe(
            step=0, system_state=self._state(available_memory_mb=1000.0), **HEALTHY
        )
        self.assertTrue(action.changed)
        self.assertEqual(action.bottleneck, "memory")
        self.assertLess(ctrl.state.chunk_size, 4096)
        self.assertLess(ctrl.state.max_batched_tokens, 8192)

    def test_high_cache_hit_rate_blocks_chunk_growth_for_ttft(self):
        # 缓存已经把未命中部分省掉了，再放大分块是拿 TPOT 换一个不存在的收益。
        ctrl = self._ctrl()
        action = ctrl.observe(
            step=0, system_state=self._state(cache_hit_ratio=0.9), **TTFT_OVER
        )
        self.assertFalse(action.changed)
        self.assertEqual(ctrl.state.n_vetoed_cache, 1)
        self.assertIn("缓存命中率", action.reason)

    def test_low_cache_hit_rate_does_not_block_growth(self):
        ctrl = self._ctrl()
        action = ctrl.observe(
            step=0, system_state=self._state(cache_hit_ratio=0.05), **TTFT_OVER
        )
        self.assertTrue(action.changed)
        self.assertGreater(ctrl.state.chunk_size, 4096)

    def test_cache_gate_does_not_block_token_recovery(self):
        # 高命中率只否决"放大分块"这一件事；撤销此前的 token 收缩与缓存无关。
        ctrl = self._ctrl()
        ctrl.observe(step=0, system_state=self._state(available_memory_mb=60000.0),
                     **TPOT_OVER)
        sunk = ctrl.state.max_batched_tokens
        ctrl.observe(step=10, system_state=self._state(cache_hit_ratio=0.95,
                                                       available_memory_mb=60000.0),
                     **HEALTHY)
        self.assertGreater(ctrl.state.max_batched_tokens, sunk)

    def test_memory_veto_also_blocks_recovery(self):
        ctrl = self._ctrl()
        ctrl.observe(step=0, system_state=self._state(available_memory_mb=60000.0),
                     **TPOT_OVER)
        sunk = ctrl.state.chunk_size
        action = ctrl.observe(
            step=10, system_state=self._state(available_memory_mb=1000.0), **HEALTHY
        )
        # 1000MB 低于临界线，走的是收缩分支而不是恢复分支。
        self.assertEqual(action.bottleneck, "memory")
        self.assertLess(ctrl.state.chunk_size, sunk)

    def test_unmeasured_signals_do_not_change_decisions_but_are_recorded(self):
        # None = 未测。它既不触发收缩也不否决放大——但它必须显形，
        # 否则事后无法区分"显存没压力"与"根本没量显存"。
        ctrl = self._ctrl()
        action = ctrl.observe(step=0, **TTFT_OVER)   # 完全不给 system_state
        self.assertTrue(action.changed, "未测信号不该否决放大")
        self.assertEqual(ctrl.state.n_vetoed_memory, 0)
        self.assertIsNone(action.signals["available_memory_mb"])
        self.assertIsNone(action.signals["cache_hit_rate"])
        self.assertIn("available_memory_mb", action.describe()["_unavailable_signals"])

    def test_thresholds_absent_means_signals_are_inert(self):
        # 不配阈值 = 该信号不参与决策。给一个 1MB 的读数也不该触发任何动作：
        # 否则"配置里没写"会被读成"已经按显存做过决策"。
        ctrl = AdaptiveController(cfg())      # 三个阈值都是 None
        action = ctrl.observe(
            step=0, system_state=self._state(available_memory_mb=1.0), **HEALTHY
        )
        self.assertFalse(action.changed)
        self.assertEqual(action.bottleneck, "none")
        self.assertFalse(ctrl.config.signal_thresholds_configured)

    def test_signals_are_carried_into_the_decision_record(self):
        ctrl = self._ctrl()
        action = ctrl.observe(
            step=0,
            system_state=self._state(cache_hit_ratio=0.25, available_memory_mb=50000.0,
                                     npu_utilization_pct=72.0),
            **HEALTHY,
        )
        described = action.describe()
        self.assertAlmostEqual(described["signals"]["cache_hit_rate"], 0.25)
        self.assertAlmostEqual(described["signals"]["available_memory_mb"], 50000.0)
        self.assertEqual(described["_unavailable_signals"], [])

    def test_replay_can_carry_signals_from_recorded_observations(self):
        # 离线复放必须能带上信号，否则"在真机 telemetry 上验证信号闸门"做不到。
        ctrl = self._ctrl()
        trace = replay_decisions(
            ctrl,
            [
                {"step": i * 8, "cache_hit_rate": 0.95, "available_memory_mb": 50000.0,
                 **TTFT_OVER}
                for i in range(4)
            ],
        )
        self.assertFalse(any(d["changed"] for d in trace))
        self.assertAlmostEqual(trace[0]["signals"]["cache_hit_rate"], 0.95)
        self.assertIn("缓存命中率", trace[0]["reason"])

    def test_replay_without_signal_fields_leaves_them_unmeasured(self):
        ctrl = self._ctrl()
        trace = replay_decisions(ctrl, [{"step": 0, **TTFT_OVER}])
        self.assertIsNone(trace[0]["signals"]["cache_hit_rate"])
        self.assertIsNone(trace[0]["signals"]["available_memory_mb"])
        self.assertEqual(
            set(trace[0]["_unavailable_signals"]),
            {"cache_hit_rate", "available_memory_mb", "npu_utilization_pct"},
        )


class TestSignalThresholdValidation(unittest.TestCase):
    def test_critical_above_veto_threshold_is_rejected(self):
        # 反过来会造出自相矛盾的区间：显存已低于收缩线却又"高于"否决线，
        # 于是既不放大也不收缩——一个永远空转的控制器。
        with self.assertRaises(ValueError):
            ControllerConfig(min_available_memory_mb=2048.0,
                             critical_available_memory_mb=4096.0)

    def test_cache_threshold_out_of_range_is_rejected(self):
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    ControllerConfig(cache_hit_rate_high=bad)

    def test_signal_thresholds_reach_describe(self):
        # 漏进 describe() 的后果不是报错，而是产物里这几项静默变成 NOT_MEASURED：
        # 看起来像"没配"，实际是"没记"。
        described = cfg(**TestFeedbackSignals.SIGNALS).describe()
        self.assertEqual(described["cache_hit_rate_high"], 0.5)
        self.assertEqual(described["min_available_memory_mb"], 8192.0)
        self.assertEqual(described["critical_available_memory_mb"], 4096.0)


class TestRecoveryTargetsComeFromConfig(unittest.TestCase):
    def test_init_targets_are_synced_from_config_even_with_an_explicit_state(self):
        # state 可能来自磁盘上的历史轨迹；恢复目标属于**配置**，
        # 不该被那个 state 的 dataclass 默认值悄悄改写。
        state = ControllerState()          # 默认 chunk_size=4096 / tokens=8192
        ctrl = AdaptiveController(
            cfg(chunk_size_init=2048, max_batched_tokens_init=3072), state=state
        )
        self.assertEqual(ctrl.state.init_chunk_size, 2048)
        self.assertEqual(ctrl.state.init_max_batched_tokens, 3072)
        # 当前值不被改写：那个是运行期状态，不是配置。
        self.assertEqual(ctrl.state.chunk_size, 4096)

    def test_state_describe_exposes_the_recovery_target(self):
        described = AdaptiveController(cfg()).state.describe()
        self.assertEqual(described["init_chunk_size"], 4096)
        self.assertEqual(described["init_max_batched_tokens"], 8192)
        self.assertEqual(described["n_recovered"], 0)


class TestCooldownUnitIsObservationCycles(unittest.TestCase):
    """冷却的单位必须是**观测周期**，不能是推理步。

    这是本轮修掉的一个死参数：字段原叫 ``cooldown_steps``、按推理步比较，
    而决策节拍是每 8 个推理步一次——两次决策之间天然已隔 ≥8 步，
    于是 `step - last_change < 3` **永远为假**，冷却从不触发、
    ``n_skipped_cooldown`` 恒为 0。它不报错，只是让配置注释里那句
    "最重要的抗振荡参数，扫参时优先调它"变成一句空话：扫参只会扫出一条直线。

    所以下面的用例刻意按**真实节拍**（每 8 个推理步一次决策）驱动。
    用相邻步驱动是不行的——旧实现那样也会通过，那正是它把问题藏起来的方式：
    单元测试的步距恰好比冷却窗口更近，于是"通过"掩盖了"生产路径上永不生效"。
    """

    #: 与 configs/*.yaml 里的 run.control_interval_steps 一致。
    INTERVAL = 8

    def _drive(self, ctrl, cycles):
        return [ctrl.observe(step=i * self.INTERVAL, **TPOT_OVER) for i in range(cycles)]

    def test_cooldown_binds_at_realistic_decision_spacing(self):
        ctrl = AdaptiveController(cfg(cooldown_cycles=3))
        actions = self._drive(ctrl, 5)

        self.assertTrue(actions[0].changed, "第 1 个周期应当调整")
        # 距上次调整不足 3 个周期 → 第 2、3 个周期必须被冷却挡住
        self.assertFalse(actions[1].changed, "冷却应当在真实节拍下生效")
        self.assertFalse(actions[2].changed)
        self.assertIn("冷却中", actions[1].reason)
        # 挡住的仍然要说明"卡在哪一维"，否则轨迹看不出本来要做什么
        self.assertEqual(actions[1].bottleneck, "tpot")
        # 第 4 个周期距上次调整已满 3 个周期 → 重新允许调整
        self.assertTrue(actions[3].changed, "冷却期满后必须能继续调整")
        # 第 5 个周期又落入新一轮冷却（距第 4 周期只有 1 个周期）
        self.assertFalse(actions[4].changed)
        self.assertEqual(ctrl.state.n_skipped_cooldown, 3)

    def test_cooldown_zero_never_blocks(self):
        ctrl = AdaptiveController(cfg(cooldown_cycles=0))
        actions = self._drive(ctrl, 3)
        self.assertTrue(all(a.changed for a in actions))
        self.assertEqual(ctrl.state.n_skipped_cooldown, 0)

    def test_cycle_advances_once_per_call_not_per_step(self):
        """周期由"调用"定义，不由步号定义——否则单位就不是周期了。"""
        ctrl = AdaptiveController(cfg())
        self.assertEqual(ctrl.observe(step=0, **HEALTHY).cycle_index, 1)
        self.assertEqual(ctrl.observe(step=8, **HEALTHY).cycle_index, 2)
        # 同一步号上再调一次仍是新周期；步号不参与周期计数。
        self.assertEqual(ctrl.observe(step=8, **HEALTHY).cycle_index, 3)
        self.assertEqual(ctrl.state.n_cycles, 3)

    def test_early_returns_do_not_consume_a_cycle(self):
        """做不出判断的调用不该消耗周期。

        否则"冷却距离"会取决于调用方有多勤快——那不是控制器的属性，
        而且会让同一份配置在不同调用点下表现出不同的抗振荡能力。
        """
        ctrl = AdaptiveController(cfg())
        ctrl.observe(step=0, ttft_samples=[], tpot_samples=[])
        self.assertEqual(ctrl.state.n_cycles, 0)

        disabled = AdaptiveController(cfg(enabled=False))
        disabled.observe(step=0, **HEALTHY)
        self.assertEqual(disabled.state.n_cycles, 0)

    def test_action_exposes_real_call_interval(self):
        """实际调用间隔必须落进动作里——它是"节拍是否等于配置"的唯一凭据。"""
        ctrl = AdaptiveController(cfg())
        first = ctrl.observe(step=16, **HEALTHY)
        self.assertIsNone(first.steps_since_last_decision, "首次决策没有前序")
        second = ctrl.observe(step=24, **HEALTHY)
        self.assertEqual(second.steps_since_last_decision, 8)
        third = ctrl.observe(step=40, **HEALTHY)
        self.assertEqual(third.steps_since_last_decision, 16)

    def test_negative_cooldown_is_rejected(self):
        # 单位换了，负值依旧没有意义；报错信息里点明单位，避免又被按步理解。
        with self.assertRaises(ValueError) as ctx:
            cfg(cooldown_cycles=-1)
        self.assertIn("观测周期", str(ctx.exception))

    def test_describe_states_the_unit(self):
        """产物里记的必须是新字段名与周期数。

        沿用旧字段名会让读产物的人按"步"理解这个数——而两者差着 8 倍。
        """
        described = cfg(cooldown_cycles=3).describe()
        self.assertIn("cooldown_cycles", described)
        self.assertNotIn("cooldown_steps", described)
        state = AdaptiveController(cfg()).state.describe()
        for key in ("n_cycles", "last_change_cycle", "last_decision_step"):
            self.assertIn(key, state)


class TestLayeredSignalWiring(unittest.TestCase):
    """第 04 步：分层信号的**接线**（记录口径、窗口过期、模式区分）。

    ⚠️ 本类不断言"分层信号改变了决策"——本步**刻意不改判据**：
    "纯 decode 特征有效"不等于"应把所有决策交给它"，那是第 09 步的问题。
    """

    HIGH_PRESSURE = dict(ttft_samples=[10.0], tpot_samples=[500.0])   # 远超 target=50

    def test_signal_mode_is_validated_at_load(self):
        """拼错必须**立刻报错**：静默降级成 legacy 会让"我开了分层"与"其实没开"长得一样。"""
        with self.assertRaises(ValueError):
            cfg(signal_mode="layred")
        with self.assertRaises(ValueError):
            cfg(signal_window_expiry_ms=0.0)
        # 合法取值不报错
        cfg(signal_mode="legacy")
        cfg(signal_mode="layered")

    def test_legacy_reports_none_not_zero_for_uncollected_signals(self):
        """legacy ⇒ 分层字段是 ``None``（**没有这个信号**），不是 0（"采集了但窗口为空"）。"""
        ctrl = AdaptiveController(cfg(signal_mode="legacy"))
        ctrl.observe(step=0, observed_at_ms=0.0, **self.HIGH_PRESSURE)
        st = ctrl.state
        self.assertIsNone(st.decode_p90_ms)
        self.assertIsNone(st.n_decode_samples)
        self.assertIsNone(st.window_has_prefill)
        # legacy 下 signals 里**不出现**分层键（旧批次轨迹逐字节不变）
        trace = ctrl.observe(step=8, observed_at_ms=8.0, **self.HIGH_PRESSURE)
        self.assertNotIn("decode_p90_ms", trace.describe()["signals"])

    def test_layered_populates_and_exposes_the_signals(self):
        """layered ⇒ 分层量被算出来，并出现在决策轨迹里（第 09 步要靠它做干预实验）。"""
        ctrl = AdaptiveController(cfg(signal_mode="layered"))
        action = ctrl.observe(
            step=0, observed_at_ms=0.0,
            ttft_samples=[10.0], tpot_samples=[500.0, 500.0],
            decode_samples=[60.0, 70.0], had_prefill_samples=[True, False],
        )
        st = ctrl.state
        self.assertEqual(st.signal_mode, "layered")
        self.assertEqual(st.n_decode_samples, 2)
        # 注意 percentile 是**带插值**的：[60, 70] 的 P90 = 69.0（不是 70）
        self.assertAlmostEqual(st.decode_p90_ms, 69.0, places=9)
        self.assertTrue(st.window_has_prefill)
        self.assertIn("decode_p90_ms", action.describe()["signals"])
        self.assertIn("window_age_ms", action.describe()["signals"])

    def test_empty_window_gives_none_not_false(self):
        """空窗口 ⇒ ``None``（未知）。**不是 False**——"没测"与"测到没有"对策不同。"""
        ctrl = AdaptiveController(cfg(signal_mode="layered"))
        ctrl.observe(step=0, observed_at_ms=0.0, ttft_samples=[10.0],
                     tpot_samples=[500.0], decode_samples=[], had_prefill_samples=[])
        self.assertEqual(ctrl.state.n_decode_samples, 0)   # 采集了，但窗口为空
        self.assertIsNone(ctrl.state.decode_p90_ms)
        self.assertIsNone(ctrl.state.window_has_prefill)

    def test_expired_window_blocks_action_and_does_not_consume_a_cycle(self):
        """**窗口过期 ⇒ 保守（不做动作）**，且不消耗观察周期。

        为什么必须有一条过期规则：此前没有任何过期机制 ⇒ 负载停顿时窗口里残留的
        **旧**样本仍会驱动收缩/放大，而从产物里看不出那些样本已经过时。
        """
        ctrl = AdaptiveController(cfg(signal_window_expiry_ms=1000.0))
        first = ctrl.observe(step=0, observed_at_ms=0.0, **self.HIGH_PRESSURE)
        cycles_after_first = ctrl.state.n_cycles
        self.assertIsNotNone(first.cycle_index, "窗口新鲜时应正常做出判断")

        # 样本数没变（没有新样本）+ 时刻推进 ⇒ 窗口过期
        stale = ctrl.observe(step=8, observed_at_ms=100_000.0, **self.HIGH_PRESSURE)
        self.assertFalse(stale.changed)
        self.assertIn("过期", stale.reason)
        self.assertIsNone(stale.cycle_index, "过期不消耗周期：'做不出判断'不是决策")
        self.assertEqual(ctrl.state.n_cycles, cycles_after_first)
        self.assertAlmostEqual(ctrl.state.window_age_ms, 100_000.0, places=3)

    def test_without_expiry_config_the_stale_window_still_acts(self):
        """反向对照：**不配**上限时同一调用照常动作 ⇒ 证明上一条的效果来自过期规则本身。

        注意"记年龄"与"按年龄设限"是两件事：年龄**总是**被记下来（它是可核对的事实），
        而是否因此拒绝动作由 ``signal_window_expiry_ms`` 决定。所以这里断言的是
        "年龄照记、动作照做"，而不是"年龄为 None"。
        """
        ctrl = AdaptiveController(cfg())          # signal_window_expiry_ms 默认 None
        ctrl.observe(step=0, observed_at_ms=0.0, **self.HIGH_PRESSURE)
        later = ctrl.observe(step=8, observed_at_ms=100_000.0, **self.HIGH_PRESSURE)
        self.assertIsNotNone(later.cycle_index)
        self.assertAlmostEqual(ctrl.state.window_age_ms, 100_000.0, places=3)


class TestSignalCriterion(unittest.TestCase):
    """第 09 步：**判据用哪一套口径**（`tpot_p90` 含 prefill 污染 / `decode_p90` 纯 decode）。

    背景（`EXPERIMENTS.md` §3.4 **E10**）：控制器读的 `tpot_p90` 在含 prefill 的窗口里
    被污染到 **337 ms**，而同一窗口的纯 decode 只有 **61 ms**（**5.5 倍**）；
    它据此收缩，又把纯 decode 从 61 ms 推到 **118 ms**。
    ⇒ 这里要证的是：**同一份输入、只换判据，决策就相反**——那是干预实验的机制。
    """

    #: 门限用**已标定**的值（`EXPERIMENTS` §4.3.4）：120 × 1.08 = 129.6。
    GATE = dict(target_tpot_ms=120.0, hysteresis=0.08)
    #: E10 的实际读数：污染值 337、纯 decode 61。
    CONTAMINATED = 337.2
    PURE_DECODE = 61.4

    def test_decode_criterion_requires_layered_sampling(self):
        """选 `decode_p90` 却不采集分层样本 ⇒ **加载即报错**（不静默退回旧口径）。"""
        with self.assertRaises(ValueError):
            cfg(signal_criterion="decode_p90", signal_mode="legacy", **self.GATE)
        with self.assertRaises(ValueError):
            cfg(signal_criterion="decode_p90x", **self.GATE)      # 拼错
        cfg(signal_criterion="decode_p90", signal_mode="layered", **self.GATE)

    def _observe(self, criterion: str):
        ctrl = AdaptiveController(cfg(signal_criterion=criterion,
                                      signal_mode="layered", **self.GATE))
        return ctrl, ctrl.observe(
            step=0, observed_at_ms=0.0,
            ttft_samples=[500.0], tpot_samples=[self.CONTAMINATED] * 32,
            decode_samples=[self.PURE_DECODE] * 32,
            had_prefill_samples=[False] * 28 + [True] * 4,
        )

    def test_same_inputs_opposite_decisions(self):
        """★ 本类存在的理由：**只换判据**，一个收缩、一个不动。

        若这条不成立，第 09 步的干预实验就没有机制可言。
        """
        _, old = self._observe("tpot_p90")
        self.assertTrue(old.changed, "含 prefill 污染的读数 337ms 超过 129.6 ⇒ 应当收缩")

        _, new = self._observe("decode_p90")
        self.assertFalse(new.changed, "纯 decode 61ms 低于 129.6 ⇒ 不应当收缩")
        self.assertIn("滞回带内", new.reason)

    def test_decode_criterion_is_recorded_and_marked(self):
        _, old = self._observe("tpot_p90")
        _, new = self._observe("decode_p90")
        # 默认口径**不得**多出这两个键（轨迹逐字节不变）
        self.assertNotIn("criterion", old.describe()["signals"])
        # 非默认口径必须写明用了哪一套 + 读数是多少
        sig = new.describe()["signals"]
        self.assertEqual(sig["criterion"], "decode_p90")
        self.assertAlmostEqual(sig["criterion_ms"], self.PURE_DECODE, places=6)

    def test_unmeasurable_criterion_is_conservative_not_a_fallback(self):
        """⚠️ 纯 decode 不可测 ⇒ **保守不动**，且**不回退**到 tpot_p90。

        回退等于把本步要消除的污染又放回来，而且"看起来在工作"。
        """
        ctrl = AdaptiveController(cfg(signal_criterion="decode_p90",
                                      signal_mode="layered", **self.GATE))
        a = ctrl.observe(step=0, observed_at_ms=0.0,
                         ttft_samples=[500.0], tpot_samples=[self.CONTAMINATED] * 32,
                         decode_samples=[], had_prefill_samples=[])
        self.assertFalse(a.changed)
        self.assertIn("不可测", a.reason)
        self.assertIsNone(a.describe()["signals"]["criterion_ms"])

    def test_reason_names_the_criterion_it_used(self):
        """轨迹里的人话必须说清是哪一套口径 —— 否则事后分不出两种 337 与 61。"""
        _, old = self._observe("tpot_p90")
        _, new = self._observe("decode_p90")
        self.assertIn("TPOT P90=", old.reason)
        # decode 口径下 61ms 触发不了；把门限压低让它触发，再检查措辞
        ctrl = AdaptiveController(cfg(signal_criterion="decode_p90",
                                      signal_mode="layered",
                                      target_tpot_ms=10.0, hysteresis=0.08))
        a = ctrl.observe(step=0, observed_at_ms=0.0,
                         ttft_samples=[10.0], tpot_samples=[999.0] * 32,
                         decode_samples=[self.PURE_DECODE] * 32,
                         had_prefill_samples=[False] * 32)
        self.assertTrue(a.changed)
        self.assertIn("decode P90=", a.reason)
        self.assertNotIn("TPOT P90=", a.reason)


if __name__ == "__main__":
    unittest.main()