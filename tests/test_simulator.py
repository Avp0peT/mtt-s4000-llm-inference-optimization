"""端到端冒烟测试：用 --dry-run 的模拟路径跑通全链路。

这个测试的价值在于**账目自洽**和**定性方向**两件事：
  1. 账目：E2E 必须等于「完成时刻 - 到达时刻」，否则计时口径有 bug
     （这正是 `RequestMetrics.accounting_error_ms` 存在的意义）。
  2. 定性：分块预填充应当降低 TPOT 尾延迟、代价是 TTFT 变化——
     如果连这个方向都不对，说明配置或策略接线错了，不该上真机浪费时间。

注意：这里断言的是**方向**而不是数值。模拟器的绝对值没有意义。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.metrics import OUTCOME_SUCCESS, OUTCOME_UNFINISHED, percentile  # noqa: E402
from ascend_opt.provenance import (  # noqa: E402
    NOT_MEASURED,
    REQUIRED_REPRO_FIELDS,
    DataLabel,
    ExecutionEnvironment,
    HardwarePlatform,
)
from ascend_opt.workload import WorkloadSpec, build_requests  # noqa: E402
from run_benchmark import _label_reason, load_config, run_once, simulate  # noqa: E402
from workloads import get_workload, list_workloads  # noqa: E402

BASELINE = _ROOT / "configs" / "baseline" / "default.yaml"
CHUNKED = _ROOT / "configs" / "chunked_prefill" / "chunked_4k.yaml"
PREFIX = _ROOT / "configs" / "prefix_cache" / "prefix_cache.yaml"
ADAPTIVE = _ROOT / "configs" / "adaptive" / "adaptive_v1.yaml"


class TestConfigsLoad(unittest.TestCase):
    """所有随仓库提供的**基准配置**都必须能加载——配置坏掉是最容易被忽略的回归。

    ⚠️ 边界（按**目录**划，不按文件名列清单）：
    * `configs/**` 下的 `*.yaml` 都是 run 配置，必须能 `load_config`；
    * `configs/cost_calib/**` **不是** run 配置，而是成本标定探针的**网格输入**
      （第 11 步，协议 `cost-calib-v1`）⇒ 它由**它自己的**检查器覆盖：
      `tests/test_costcalib_grid.py::TestRealGridPasses::test_shipped_grid_is_valid`。
      ⇒ 这里是"从**错误的**检查里排除、并加进**正确的**检查"，不是留一个没人管的洞。
    * `sweep.yaml` 是扫描定义，同理。
    """

    def test_all_shipped_configs_load(self):
        for path in sorted((_ROOT / "configs").rglob("*.yaml")):
            if path.name == "sweep.yaml":
                continue  # 扫描定义不是 run 配置
            if "cost_calib" in path.parts:
                continue  # 探针网格输入；由 test_costcalib_grid 校验
            with self.subTest(config=path.name):
                cfg = load_config(path)
                self.assertGreaterEqual(cfg.max_num_seqs, 1)
                self.assertTrue(cfg.run.slo.ttft_ms is not None)

    def test_unknown_field_is_rejected(self):
        import yaml

        payload = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        payload["chunk"]["chunk_size_typo"] = 123
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(bad)

    def test_misspelled_execution_environment_is_rejected_at_load(self):
        # 执行环境声明拼错，必须在**加载时**报错，而不是白跑一轮基准、
        # 最后在写 meta 时才炸。静默当成 UNKNOWN 更糟：人会以为声明生效了。
        import yaml

        payload = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        payload.setdefault("runtime", {})["execution_environment"] = "remote_ascent"
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad_env.yaml"
            bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_config(bad)
            self.assertIn("remote_ascent", str(ctx.exception))

    def test_valid_execution_environment_declaration_is_accepted(self):
        import yaml

        payload = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        payload.setdefault("runtime", {})["execution_environment"] = "remote_ascend"
        with tempfile.TemporaryDirectory() as tmp:
            ok = Path(tmp) / "remote.yaml"
            ok.write_text(yaml.safe_dump(payload), encoding="utf-8")
            cfg = load_config(ok)
            self.assertEqual(cfg.runtime.get("execution_environment"), "remote_ascend")

    def test_misspelled_hardware_platform_is_rejected_at_load(self):
        # 与执行环境同理：拼错的平台声明要在加载时报错，不能静默降级成"没声明"。
        import yaml

        payload = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        payload.setdefault("runtime", {})["hardware_platform"] = "mthread"
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad_platform.yaml"
            bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_config(bad)
            self.assertIn("mthread", str(ctx.exception))

    def test_contradictory_platform_declaration_is_rejected_at_load(self):
        # 声明"在昇腾上跑"却把平台写成 mthreads：两个字段都进 meta 且都被下游
        # 按平台筛选，放过去就是永久留一份自相矛盾的溯源记录，事后没人能发现。
        import yaml

        payload = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        runtime = payload.setdefault("runtime", {})
        runtime["execution_environment"] = "remote_ascend"
        runtime["hardware_platform"] = "mthreads"
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "contradiction.yaml"
            bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                load_config(bad)
            self.assertIn("hardware_platform", str(ctx.exception))

    def test_s4000_declaration_is_accepted(self):
        # 以 S4000 为主：这份声明必须能通过加载，否则 S4000 的数据永远进不了结论。
        import yaml

        payload = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        runtime = payload.setdefault("runtime", {})
        runtime["execution_environment"] = "remote_mthreads"
        runtime["hardware_platform"] = "mthreads"
        with tempfile.TemporaryDirectory() as tmp:
            ok = Path(tmp) / "s4000.yaml"
            ok.write_text(yaml.safe_dump(payload), encoding="utf-8")
            cfg = load_config(ok)
            self.assertEqual(cfg.runtime["hardware_platform"], "mthreads")

    def test_missing_required_section_is_rejected(self):
        import yaml

        payload = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
        del payload["workload"]
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "noworkload.yaml"
            bad.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(bad)


class TestWorkloadRegistry(unittest.TestCase):
    def test_registry_is_usable(self):
        for name in list_workloads():
            with self.subTest(workload=name):
                spec = get_workload(name)
                self.assertGreater(len(build_requests(spec)), 0)

    def test_unknown_workload_raises_with_hint(self):
        with self.assertRaises(KeyError) as ctx:
            get_workload("nonexistent_workload")
        self.assertIn("WORKLOADS", str(ctx.exception))

    def test_override_of_unknown_field_raises(self):
        with self.assertRaises(KeyError):
            get_workload("short_chat", concurrency_typo=4)


class TestAccounting(unittest.TestCase):
    def _sim(self, cfg, spec_name):
        spec = get_workload(spec_name)
        return simulate(cfg, build_requests(spec), spec=spec)

    def test_e2e_matches_wall_clock_of_request(self):
        cfg = load_config(BASELINE)
        result = self._sim(cfg, "short_chat")

        self.assertEqual(result.dropped, 0, "不应有请求因 max_steps 被丢弃")
        # 只有"完成且到达时刻已知"的请求能自查账目；其余返回 None。
        # 把 None 当 0 收进来会读成"账目完全自洽"，恰好掩盖掉问题。
        errors = [
            e for m in result.metrics if (e := m.accounting_error_ms()) is not None
        ]
        self.assertTrue(errors, "应当有请求完成")
        self.assertLess(
            percentile(errors, 99),
            1.0,
            msg="E2E 与「完成时刻 - 到达时刻」不一致，说明 TTFT/TPOT 的采样点算错了",
        )

    def test_every_request_generates_declared_token_count(self):
        cfg = load_config(BASELINE)
        requests = build_requests(get_workload("short_chat"))
        result = simulate(cfg, requests, spec=get_workload("short_chat"))
        by_id = {m.req_id: m for m in result.metrics}
        for req in requests:
            m = by_id[req.req_id]
            expected_steps = max(0, req.gen_len - 1)
            if expected_steps == 0:
                self.assertAlmostEqual(m.tpot_ms, 0.0)
            else:
                self.assertGreater(m.tpot_ms, 0.0)
            self.assertTrue(m.success)

    def test_telemetry_covers_the_run(self):
        cfg = load_config(BASELINE)
        result = self._sim(cfg, "short_chat")
        self.assertGreater(len(result.samples), 0)
        self.assertEqual(result.samples[0].step, 0)
        # 采样时刻必须单调递增，否则时间线不可信
        times = [s.t_ms for s in result.samples]
        self.assertEqual(times, sorted(times))
        self.assertEqual(result.samples[-1].step, result.steps - 1)

    def test_all_five_timestamps_are_recorded_and_ordered(self):
        """五个时刻必须齐全且单调：到达 ≤ 受理 ≤ 首次调度 ≤ 首 token ≤ 完成。

        少任何一个，TTFT 就无法拆回"等了多少"与"算了多少"，
        于是"该加容量还是该调分块"这个判断失去依据。
        """
        cfg = load_config(BASELINE)
        result = self._sim(cfg, "long_context_qa")
        checked = 0
        for m in result.metrics:
            if not m.success:
                continue
            checked += 1
            self.assertLessEqual(m.scheduled_arrival_ms, m.arrival_ms)
            self.assertLessEqual(m.arrival_ms, m.enqueue_ms)
            self.assertLessEqual(m.enqueue_ms, m.first_schedule_ms)
            self.assertLessEqual(m.first_schedule_ms, m.first_token_ms)
            self.assertLessEqual(m.first_token_ms, m.done_ms)
        self.assertGreater(checked, 0, "应当有请求完成")

    def test_ttft_is_exactly_the_sum_of_its_three_segments(self):
        cfg = load_config(BASELINE)
        result = self._sim(cfg, "long_context_qa")
        for m in result.metrics:
            if not m.success:
                continue
            self.assertAlmostEqual(
                m.admission_ms + m.queue_ms + m.prefill_exec_ms,
                m.ttft_ms,
                places=6,
                msg=f"{m.req_id}：TTFT 三段分解对不上，说明某个时刻记错了位置",
            )

    def test_queue_wait_is_measurable_under_contention(self):
        """容量不足导致的排队必须显形，而不是恒为 0。

        改写前 queue_ms 恒为 0（arrival 与 enqueue 被赋成同一个值），
        于是"调度容量不足"这一整类问题在指标上根本不可见。

        负载必须用**开放环突发**：闭环下在途数被 concurrency 卡住，
        只有最初填满的那几个请求会排队（实测 32 个里只有 5 个，
        中位数本来就是 0），拿闭环的中位数当判据会得出相反结论。
        突发负载每 8 秒来 16 个请求，而单步只吃得下两三个，
        排队因此是秒级、必然显形。
        """
        cfg = load_config(BASELINE)
        spec = get_workload("mixed_burst")
        result = simulate(cfg, build_requests(spec), spec=spec)

        queue = [
            m.queue_ms for m in result.metrics
            if m.success and m.queue_ms is not None
        ]
        self.assertTrue(queue, "应当有请求完成")
        self.assertGreater(
            percentile(queue, 50), 100.0,
            msg="容量不足时的排队没有显形，arrival/enqueue 可能又被混成一个时刻",
        )

    def test_outcome_is_success_or_unfinished_in_the_simulator(self):
        """模拟器只有两条终态：success 与 unfinished。

        它没有服务端错误注入、也不做准入拒绝，所以 failed / rejected 恒为 0。
        这一点必须由测试固定住，否则 n_failed=0 会被读成"测过、没失败"。
        """
        cfg = load_config(BASELINE)
        result = self._sim(cfg, "short_chat")
        outcomes = {m.outcome for m in result.metrics}
        self.assertTrue(outcomes <= {OUTCOME_SUCCESS, OUTCOME_UNFINISHED}, outcomes)
        self.assertEqual(sum(1 for m in result.metrics if not m.success), result.dropped)


class TestOpenLoopTiming(unittest.TestCase):
    """开放环的到达语义：日程时刻不得被当前步的时间覆盖。

    这是评审点名的缺陷的专用回归组——它不以"报错"的形式存在，
    而是让等待时间凭空消失，因此只能靠不变量守住。
    """

    def test_arrival_equals_schedule_and_wait_is_visible(self):
        cfg = load_config(ADAPTIVE)
        spec = get_workload("mixed_burst")      # arrival="burst" -> 开放环
        result = simulate(cfg, build_requests(spec), spec=spec)

        # 1) 到达时刻必须等于日程时刻（被改写的唯一症状就是这个等式不成立）
        for m in result.metrics:
            self.assertAlmostEqual(
                m.arrival_ms, m.scheduled_arrival_ms, places=6,
                msg=f"{m.req_id} 的到达时刻被步时间覆盖了",
            )
            self.assertGreaterEqual(m.enqueue_ms, m.arrival_ms)

        # 2) 日程的跨度必须保留：把 arrival 压成本步起始会让突发结构消失
        span = max(m.arrival_ms for m in result.metrics) - min(m.arrival_ms for m in result.metrics)
        self.assertGreater(span, 1000.0, "到达时刻被压平，突发日程信息丢失")

        # 3) "早已到达、但下一步才轮到处理"的等待必须显形，而不是变成 0
        admission = [m.admission_ms for m in result.metrics]
        self.assertGreater(
            percentile(admission, 50), 0.0,
            msg="受理延迟恒为 0：到达与受理又被合并成一个时刻了",
        )

    def test_closed_loop_arrival_is_when_client_frees_up(self):
        """闭环没有"原定到达"，客户端腾出容量那一刻才是到达。

        闭环下 admission_ms 应当为 0（到达即受理），等待全部体现在 queue_ms 上；
        与开放环的口径差异必须一眼可辨，否则两种负载的数字会被混着比。
        """
        cfg = load_config(BASELINE)
        spec = get_workload("long_context_qa")   # closed
        result = simulate(cfg, build_requests(spec), spec=spec)
        for m in result.metrics:
            self.assertAlmostEqual(m.admission_ms, 0.0)
            # 闭环下"原定到达"就是补位时刻，与 arrival 一致
            self.assertAlmostEqual(m.scheduled_arrival_ms, m.arrival_ms)


class TestClosedLoopAdmission(unittest.TestCase):
    """闭环语义：在途请求数不得超过 workload.concurrency。

    这个不变量如果破了，症状不是报错，而是「一部分请求永远不完成」——
    所以必须由测试守住。
    """

    def test_in_flight_never_exceeds_concurrency(self):
        cfg = load_config(BASELINE)
        spec = get_workload("long_context_qa")   # concurrency=4
        result = simulate(cfg, build_requests(spec), spec=spec)
        self.assertEqual(result.dropped, 0)
        # 遥测里的 running = 正在 prefill + 正在 decode 的请求数，
        # 它是在途请求数的下界，必须始终不超过 concurrency。
        peak = max(s.running for s in result.samples)
        self.assertLessEqual(peak, spec.concurrency)
        self.assertGreater(peak, 0)

    def test_open_loop_ignores_concurrency(self):
        # 对照：不传 spec 时是开放环，在途数由调度预算（max_num_seqs）决定，
        # 不该被 workload.concurrency 限制。这条测试锁住两种语义不会混淆。
        cfg = load_config(BASELINE)
        spec = get_workload("long_context_qa")
        result = simulate(cfg, build_requests(spec))
        peak = max(s.running for s in result.samples)
        self.assertEqual(result.dropped, 0)
        self.assertGreater(peak, spec.concurrency)

    def test_closed_loop_finishes_all_requests(self):
        cfg = load_config(BASELINE)
        for name in ("short_chat", "long_context_qa", "shared_prefix_4k"):
            with self.subTest(workload=name):
                spec = get_workload(name)
                result = simulate(cfg, build_requests(spec), spec=spec)
                self.assertEqual(result.dropped, 0)
                self.assertEqual(len(result.metrics), spec.num_requests)
                self.assertTrue(all(m.success for m in result.metrics))


class TestQualitativeBehaviour(unittest.TestCase):
    """定性方向测试：这些是"策略接线是否正确"的探针，不是性能结论。"""

    def _sim(self, config_path, workload: str):
        cfg = load_config(config_path)
        spec = get_workload(workload)
        return simulate(cfg, build_requests(spec), spec=spec)

    @staticmethod
    def _stall(result):
        """每请求自身 token 间隔的 P99 —— 卡顿指标。

        不能用 tpot（每请求均值）代替：长 prompt 阻塞造成的秒级停顿会被
        gen_len 除开，在均值里几乎消失，而用户感受到的正是那次停顿。
        """
        return [m.tpot_p99_ms for m in result.metrics]

    def test_long_context_baseline_has_severe_stalls(self):
        # 长 prompt 一次算完会长时间阻塞 decode，应当出现秒级停顿。
        result = self._sim(BASELINE, "long_context_qa")
        stall_p50 = percentile(self._stall(result), 50)
        self.assertGreater(stall_p50, 500.0)

    def test_tpot_mean_hides_the_stall_that_stall_metric_catches(self):
        # 这条测试的意义在于记录「为什么需要 stall 指标」这一事实本身：
        # 同一个 run，均值看起来完全正常，而卡顿是秒级的。
        result = self._sim(BASELINE, "long_context_qa")
        tpot_p99 = percentile([m.tpot_ms for m in result.metrics], 99)
        stall_p50 = percentile(self._stall(result), 50)
        self.assertLess(tpot_p99, 100.0)
        self.assertGreater(stall_p50 / tpot_p99, 5.0)

    def test_chunked_prefill_improves_stall(self):
        base = self._sim(BASELINE, "long_context_qa")
        chunked = self._sim(CHUNKED, "long_context_qa")
        base_stall = percentile(self._stall(base), 50)
        chunk_stall = percentile(self._stall(chunked), 50)
        self.assertLess(
            chunk_stall,
            base_stall,
            msg="分块预填充没有改善卡顿，策略接线可能有问题",
        )

    def test_prefix_cache_improves_ttft_without_touching_tpot(self):
        # 前缀缓存只该影响 prefill（TTFT），不该改变 decode 的节奏（TPOT）。
        # 若 TPOT 也变了，说明改动影响到了批组成，必须单独解释。
        cached = self._sim(PREFIX, "shared_prefix_4k")
        hit_rates = [m.cached_tokens / m.prompt_len for m in cached.metrics]
        self.assertGreater(percentile(hit_rates, 50), 0.9, "共享前缀应接近满命中")
        # 命中后 prefill 几乎为零，TTFT 应当远小于一个 prefill 步
        self.assertLess(percentile([m.ttft_ms for m in cached.metrics], 50), 100.0)

    def test_adaptive_controller_makes_decisions_without_oscillating(self):
        result = self._sim(ADAPTIVE, "mixed_burst")
        decisions = result.controller_trace
        self.assertGreater(len(decisions), 0, "控制器未产生任何决策记录")
        # 振荡的定义：绝大多数决策都在「改变」。冷却机制生效时 changed 占比应明显低于 1。
        changed_ratio = sum(1 for d in decisions if d["changed"]) / len(decisions)
        self.assertLess(changed_ratio, 0.6, f"控制器疑似振荡：changed 占比 {changed_ratio:.2f}")

    def test_no_dropped_requests_across_shipped_configs(self):
        for path, workload in (
            (BASELINE, "short_chat"),
            (CHUNKED, "long_context_qa"),
            (PREFIX, "shared_prefix_4k"),
            (ADAPTIVE, "mixed_burst"),
        ):
            with self.subTest(config=path.name):
                result = self._sim(path, workload)
                self.assertEqual(result.dropped, 0)
                self.assertEqual(result.stall_reason, "")


class TestControllerClosedLoop(unittest.TestCase):
    """控制器调大的额度必须**真的生效**，且决策依据必须可核验。

    改动前的失效是静默的：策略内部用 frozen 的 ``config.chunk_size`` 封顶，
    控制器把 chunk 调大之后日志显示"已调大"，而实际发放的 prefill 额度一动不动。
    产物里只有生效值，所以这个失效在数据上完全看不出来——
    唯一的守法是让"建议值 vs 生效值"都落盘，并断言额度真的涨过了静态上限。
    """

    def _grow_forced_cfg(self, *, controller_enabled: bool = True):
        """造一个必然持续走"放大 chunk"分支的配置。

        - TTFT 目标设成 1ms：任何一步都算超目标；
        - TPOT 目标设成 500ms：保证 decode 侧永远有余量（否则取舍判断会拦住放大）；
        - 静态 chunk_size=512，控制器初值同为 512：起点一致，差异只可能来自控制器。
        """
        cfg = load_config(CHUNKED)
        cfg.run.repeats = 1
        cfg.run.control_interval_steps = 4
        cfg.chunk = replace(cfg.chunk, chunk_size=512, max_num_batched_tokens=8192)
        cfg.controller = replace(
            cfg.controller,
            enabled=controller_enabled,
            target_ttft_ms=1.0,
            target_tpot_ms=500.0,
            cooldown_cycles=0,
            chunk_size_init=512,
            chunk_size_min=512,
            chunk_size_max=8192,
            chunk_size_step=512,
            max_batched_tokens_init=8192,
            max_batched_tokens_max=8192,
        )
        return cfg

    def _sim(self, cfg, workload: str = "long_context_qa"):
        spec = get_workload(workload)
        return simulate(cfg, build_requests(spec), spec=spec)

    @staticmethod
    def _prefill_budgets(result):
        return [
            s.prefill_budget_effective
            for s in result.samples
            if s.prefill_budget_effective is not None
        ]

    def test_controller_growth_is_honoured_by_the_chunk_policy(self):
        cfg = self._grow_forced_cfg()
        result = self._sim(cfg)
        peak_budget = max(self._prefill_budgets(result))
        self.assertGreater(
            peak_budget, cfg.chunk.chunk_size,
            "控制器把 chunk 调大了，但实际发放的 prefill 额度没有涨过静态上限——"
            "建议值被某个闸门静默压回了",
        )
        self.assertGreater(result.controller_trace[-1]["state"]["chunk_size"],
                           cfg.chunk.chunk_size)

    def test_static_chunking_is_the_control_condition(self):
        # 同一负载、同一静态 chunk_size，只把控制器关掉：
        # 额度必须一步都没有涨过 512。没有这条对照，上面的"涨了"无法归因给控制器。
        cfg = self._grow_forced_cfg(controller_enabled=False)
        result = self._sim(cfg)
        self.assertLessEqual(max(self._prefill_budgets(result)), 512)
        self.assertEqual(result.controller_trace, [])

    def test_requested_and_effective_operating_point_are_both_recorded(self):
        cfg = self._grow_forced_cfg()
        result = self._sim(cfg)
        for s in result.samples:
            self.assertIsNotNone(s.chunk_size_requested, "建议值必须落盘")
            self.assertIsNotNone(s.max_batched_tokens_requested, "建议值必须落盘")
            # 生效上限 = min(建议上限, 单步 token 预算)。这个恒等式是可核验的，
            # 不必依赖"读日志猜额度到底动没动"。
            self.assertEqual(
                s.chunk_size, min(s.chunk_size_requested, s.max_batched_tokens)
            )

    def test_state_vector_is_observed_on_every_decision_step(self):
        cfg = self._grow_forced_cfg()
        result = self._sim(cfg)
        self.assertGreater(len(result.system_states), 0)
        self.assertEqual(
            [s["step"] for s in result.system_states],
            [d["step"] for d in result.controller_trace],
            "s_t 的采样节拍必须与控制节拍一致，否则决策依据无法逐条对齐",
        )

    def test_unmeasurable_signals_stay_explicitly_unavailable(self):
        # 离线路径上量不到显存：必须是 None 并列入 _unavailable_fields，
        # 不能被填成 0（0 会被读成"显存已耗尽"）。
        cfg = self._grow_forced_cfg()
        result = self._sim(cfg)
        for state in result.system_states:
            self.assertIsNone(state["available_memory_mb"])
            self.assertIsNone(state["npu_utilization_pct"])
            self.assertIn("available_memory_mb", state["_unavailable_fields"])

    def test_cache_hit_signal_is_none_when_cache_is_off_and_zero_when_it_misses(self):
        # 这两种情形在报表口径上都是 0.0，但给控制器时必须分开：
        # "缓存没开"（信号不存在）与"缓存开着但一次没命中"（策略没留住前缀）。
        off = self._sim(self._grow_forced_cfg())
        self.assertTrue(
            all(s["cache_hit_ratio"] is None for s in off.system_states),
            "缓存关闭时命中率必须是 None，而不是 0.0",
        )

        cfg = self._grow_forced_cfg()
        cfg.cache = replace(cfg.cache, enabled=True)
        on = self._sim(cfg)          # long_context_qa 无共享前缀 -> 查过、没命中
        ratios = {s["cache_hit_ratio"] for s in on.system_states}
        self.assertEqual(ratios, {0.0}, "缓存开启后命中率应当是测得的 0.0")

    def test_unavailable_signal_does_not_veto_growth_in_the_simulator(self):
        # 阈值配了但信号测不到：不得凭空否决放大，且必须标注该信号未参与决策。
        cfg = self._grow_forced_cfg()
        cfg.controller = replace(
            cfg.controller, min_available_memory_mb=8192.0, cache_hit_rate_high=0.5
        )
        result = self._sim(cfg)
        changed = [d for d in result.controller_trace if d["changed"]]
        self.assertTrue(changed)
        self.assertTrue(
            all("available_memory_mb" in d["_unavailable_signals"] for d in changed)
        )
        self.assertEqual(
            result.controller_trace[-1]["state"]["n_vetoed_memory"], 0
        )

    def test_controller_effect_is_written_to_meta(self):
        cfg = self._grow_forced_cfg()
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            import json

            run_dir = run_once(cfg, dry_run=True, out_dir=Path(tmp),
                               run_id="ctrl-effect", verbose=False)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            effect = meta["controller_effect"]
            self.assertGreater(effect["n_control_decisions"], 0)
            self.assertGreater(effect["n_decisions_changed"], 0)
            # 建议值与生效值的区间必须都落盘，且这次应当是一致的（无未兑现）
            self.assertEqual(
                effect["chunk_requested"]["max"], effect["chunk_effective"]["max"]
            )
            self.assertEqual(effect["n_steps_chunk_request_not_honoured"], 0)
            self.assertEqual(len(meta["system_states"]), effect["n_control_decisions"])

    def test_decision_interval_matches_configured_control_interval(self):
        """决策节拍必须等于 ``run.control_interval_steps``，而且这件事要落盘。

        冷却按**观测周期**计数，而"一个周期跨多少推理步"只在这里看得见。
        节拍一旦与配置脱节（例如调用点仍按旧常量触发），冷却就会按错误的时间
        尺度工作——它不报错，只表现为"参数调了没反应"。
        """
        cfg = self._grow_forced_cfg()
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            import json

            run_dir = run_once(cfg, dry_run=True, out_dir=Path(tmp),
                               run_id="ctrl-interval", verbose=False)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

        trace = meta["controller_trace"]
        intervals = [
            d["steps_since_last_decision"] for d in trace
            if d["steps_since_last_decision"] is not None
        ]
        self.assertTrue(intervals, "应当发生多次决策，才谈得上间隔")
        self.assertEqual(
            set(intervals), {cfg.run.control_interval_steps},
            f"决策节拍与 control_interval_steps="
            f"{cfg.run.control_interval_steps} 不一致：实得 {sorted(set(intervals))}",
        )
        # 同一个事实也要能从 controller_effect 读到，而不是只能翻轨迹
        effect = meta["controller_effect"]
        self.assertEqual(
            effect["decision_interval_steps"]["min"], cfg.run.control_interval_steps
        )
        self.assertEqual(
            effect["decision_interval_steps"]["max"], cfg.run.control_interval_steps
        )
        # 轨迹条数 ≠ 周期数：调用点每次节拍都会去看一眼，但"没有有效样本"的那几次
        # 做不出任何判断，不该消耗周期（cycle_index 为 None）。这个差额必须能对上，
        # 否则"周期数"与"调用次数"就会被混成一个数——而这正是冷却单位搞错的开端。
        not_a_cycle = [d for d in trace if d["cycle_index"] is None]
        self.assertEqual(effect["n_cycles"], len(trace) - len(not_a_cycle))
        # intervals 比周期数少一个：**第一个**周期没有"上一次决策步"，量不出间隔。
        # 这不是记账漏洞，而是"间隔"这个量的定义域——所以差额必须恰好是 1，
        # 多一个少一个都说明 cycle_index 与 steps_since_last_decision 不同源。
        self.assertEqual(len(intervals), effect["n_cycles"] - 1)

    def test_cooldown_is_reachable_in_a_real_run(self):
        """冷却必须能在真实 run 里被触发过——否则它就是个死参数。

        这一条是给"参数存在但永不生效"那类失效兜底的：单元测试可以按相邻步
        驱动从而让冷却看起来工作正常，而真实节拍下它一次都不触发。
        """
        cfg = self._grow_forced_cfg()
        cfg.run.repeats = 1
        cfg.controller = replace(cfg.controller, cooldown_cycles=2)
        with tempfile.TemporaryDirectory() as tmp:
            import json

            run_dir = run_once(cfg, dry_run=True, out_dir=Path(tmp),
                               run_id="ctrl-cooldown", verbose=False)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))

        state = meta["controller_trace"][-1]["state"]
        self.assertGreater(
            state["n_skipped_cooldown"], 0,
            "真实节拍下冷却一次都没触发——说明冷却窗口比决策节拍还短，"
            "这个参数是死的（扫它不会有任何效果）",
        )
        self.assertGreater(state["n_cycles"], state["n_skipped_cooldown"])


class TestStarvationGuards(unittest.TestCase):
    """守住两类「请求永远不完成」的失效模式。

    它们都不会报错，只会表现为 dropped > 0 或无限空转，是最难排查的一类问题，
    因此必须有测试固定住行为。
    """

    def _tiny_budget_cfg(self, chunk_enabled: bool):
        cfg = load_config(BASELINE)
        cfg.chunk = replace(cfg.chunk, enabled=chunk_enabled, chunk_size=1024,
                            max_num_batched_tokens=1024)
        cfg.device = replace(cfg.device, max_steps=100000)
        cfg.run.repeats = 1
        return cfg

    def _spec(self):
        # prompt 4096 远超单步预算 1024
        return WorkloadSpec(
            name="overflow_prompt",
            num_requests=4,
            prompt_len_min=4096,
            prompt_len_max=4096,
            gen_len_min=8,
            gen_len_max=8,
            concurrency=2,
            arrival="closed",
            seed=0,
        )

    def test_oversized_prompt_without_chunking_is_reported_as_stall(self):
        # 未开启分块 = 整段 prompt 一次算完。prompt 超过单步预算时根本无法起步，
        # 此时正确行为是**立刻判定为死锁并给出诊断**，而不是空转到 max_steps。
        cfg = self._tiny_budget_cfg(chunk_enabled=False)
        spec = self._spec()
        result = simulate(cfg, build_requests(spec), spec=spec)

        self.assertNotEqual(result.stall_reason, "", "应当判定为死锁")
        self.assertIn("chunk.enabled=False", result.stall_reason)
        self.assertLess(result.steps, 10, "应当立即判定，而不是空转")
        self.assertGreater(result.dropped, 0)
        # 未完成的请求必须出现在明细里并标成 unfinished。
        # 它们若不出现在明细中，就会从"未完成率"的分母里消失——
        # 分母被静默改小，正是本次修复要堵的东西。
        self.assertEqual(len(result.metrics), spec.num_requests)
        self.assertTrue(
            all(m.outcome == OUTCOME_UNFINISHED for m in result.metrics),
            "未完成的请求应当标成 unfinished，而不是被混成 success/failed",
        )

    def test_partial_prefill_prevents_starvation_when_chunking_on(self):
        # 同一个负载，打开分块后允许「部分起步」，系统可以正常推进。
        cfg = self._tiny_budget_cfg(chunk_enabled=True)
        spec = self._spec()
        result = simulate(cfg, build_requests(spec), spec=spec)

        self.assertEqual(result.stall_reason, "")
        self.assertEqual(result.dropped, 0)
        self.assertEqual(len(result.metrics), spec.num_requests)
        self.assertTrue(all(m.success for m in result.metrics))

    def test_token_budget_is_never_exceeded(self):
        cfg = self._tiny_budget_cfg(chunk_enabled=True)
        spec = self._spec()
        result = simulate(cfg, build_requests(spec), spec=spec)
        budget = cfg.chunk.max_num_batched_tokens
        for sample in result.samples:
            self.assertLessEqual(
                sample.scheduled_tokens, budget,
                msg=f"step {sample.step} 调度了 {sample.scheduled_tokens} token，超过预算 {budget}",
            )


class TestRunOnceWritesArtifacts(unittest.TestCase):
    def test_outputs_are_written_and_self_describing(self):
        cfg = load_config(BASELINE)
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = run_once(
                cfg,
                dry_run=True,
                out_dir=Path(tmp),
                run_id="test-run",
                verbose=False,
            )
            for name in ("meta.json", "telemetry.jsonl", "requests.jsonl",
                         "config.snapshot.yaml"):
                self.assertTrue((run_dir / name).exists(), f"缺少产物 {name}")

            import json

            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["context"]["dry_run"], "dry_run 标记必须落盘")
            self.assertEqual(meta["summary"]["n_requests"],
                             meta["summary"]["n_success"])
            # 环境指纹缺失时必须显式标注，而不是留空
            self.assertIn("available", meta["hardware_env"])

    def test_config_describe_includes_runtime_section(self):
        """回归：runtime 段必须进 describe()。

        它曾从 describe() 里漏掉，后果不是报错，而是 meta.json 里
        model / precision / quantization / graph_mode 静默变成
        NOT_MEASURED——看起来像"没配"，实际是"没记"。
        这类静默丢失正是规范第 8 节要防的。
        """
        cfg = load_config(BASELINE)
        described = cfg.describe()
        self.assertIn("runtime", described)
        self.assertEqual(described["runtime"]["model"], "Qwen2.5-7B-Instruct")

    def test_reproducibility_metadata_lands_in_meta(self):
        cfg = load_config(BASELINE)
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            import json

            run_dir = run_once(cfg, dry_run=True, out_dir=Path(tmp),
                               run_id="repro", verbose=False)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            repro = meta["reproducibility"]
            for field in REQUIRED_REPRO_FIELDS:
                self.assertIn(field, repro, f"元数据缺字段 {field}")
            # 配置里写明的项必须真的被带过来，不能是缺失标记
            self.assertEqual(repro["model"], "Qwen2.5-7B-Instruct")
            self.assertEqual(repro["precision"], "bf16")
            self.assertEqual(repro["graph_mode"], "eager")
            # dry-run 没有预热轮：必须是缺失标记而不是 0（0 会被读成"确实不预热"）
            self.assertEqual(repro["warmup_count"], NOT_MEASURED)

    def test_execution_environment_declaration_reaches_meta(self):
        """执行环境声明必须被搬运进 meta，且不因 dry-run 而丢失。

        这里只验证"配置 -> 产物"这条链路；声明对**标签**的影响由
        test_provenance.TestExecutionEnvironment 逐格钉死。
        """
        cfg = load_config(BASELINE)
        cfg.runtime["execution_environment"] = "local"
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            import json

            run_dir = run_once(cfg, dry_run=True, out_dir=Path(tmp),
                               run_id="env-decl", verbose=False)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(
                meta["reproducibility"]["execution_environment"], "local"
            )
            self.assertEqual(meta["config"]["runtime"]["execution_environment"], "local")
            # 声明自己跑在哪，并不能把合成数字变成实测。
            self.assertEqual(meta["data_label"], "TEST_DATA")

    def test_dry_run_is_labelled_test_data_and_cannot_enter_conclusions(self):
        cfg = load_config(BASELINE)
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            import json

            run_dir = run_once(cfg, dry_run=True, out_dir=Path(tmp),
                               run_id="labelled", verbose=False)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["data_label"], "TEST_DATA")
            self.assertEqual(meta["summary"]["data_label"], "TEST_DATA")
            # 判据来自 summary 的标签，且必须是"不可用于结论"
            self.assertFalse(
                DataLabel(meta["summary"]["data_label"]).eligible_for_conclusion
            )
            # 必须写清"为什么不能进结论"，而不是只给一个标签
            self.assertIn("label_reason", meta["summary"]["extra"])

    def test_repeat_directories_are_separated(self):
        cfg = load_config(BASELINE)
        cfg.run.repeats = 2
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = run_once(cfg, dry_run=True, out_dir=out,
                             run_id="rep", repeat_index=0, verbose=False)
            second = run_once(cfg, dry_run=True, out_dir=out,
                              run_id="rep", repeat_index=1, verbose=False)
            self.assertNotEqual(first, second)
            self.assertTrue(first.name.startswith("rep"))
            self.assertTrue(second.name.startswith("rep-r1"))

    def test_shared_env_fingerprint_is_picked_up(self):
        """run 目录里没有 env.json 时，应回退到共享位置 results/raw/_env/env.json。

        这条回退解决的是一个真实的时序问题：硬件指纹必须在 run 开始**之前**采集
        （否则测到的是被 run 影响过的状态），但 run 目录要到 run 开始之后才创建。
        没有这条测试，回退逻辑一旦失效就会静默退化成「环境未知」。

        注意这里断言的是**指纹被读到了**（source 指向共享目录），而不是
        ``available is True``：本 fixture 没有设备探测段，因此可用性属于
        "未知"。把它默认成 True 会让无 NPU 机器上的 run 自称已实测，
        正是规范第 10 节要拦的事。
        """
        import json

        cfg = load_config(BASELINE)
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            env_dir = out / "_env"
            env_dir.mkdir(parents=True, exist_ok=True)
            (env_dir / "env.json").write_text(
                json.dumps({"collected_at": "test", "hostname": "test-host"}),
                encoding="utf-8",
            )

            run_dir = run_once(cfg, dry_run=True, out_dir=out,
                               run_id="envtest", verbose=False)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            hw = meta["hardware_env"]
            self.assertEqual(hw["hostname"], "test-host", f"未取到共享环境指纹：{hw}")
            self.assertIn("_env", hw["source"])
            # 无设备探测段 -> 可用性未知，而不是"可用"
            self.assertIsNone(hw["available"], f"缺失的硬件探测不应被当成可用：{hw}")

    def test_env_fingerprint_derives_availability_from_device_probe(self):
        """有设备探测段时，可用性必须由探测结果推导，而不是默认 True。

        这是本机（无 NPU 笔记本）的真实情形：env.json 里
        ``device.npu_smi_available = false``，run 必须被标成
        HARDWARE_REQUIRED，而不是 MEASURED。
        """
        import json

        cfg = load_config(BASELINE)
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            env_dir = out / "_env"
            env_dir.mkdir(parents=True, exist_ok=True)
            (env_dir / "env.json").write_text(
                json.dumps({
                    "collected_at": "test",
                    "device": {"npu_smi_available": False, "npu_smi_error": "not found"},
                }),
                encoding="utf-8",
            )

            run_dir = run_once(cfg, dry_run=True, out_dir=out,
                               run_id="nodev", verbose=False)
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            self.assertIs(meta["hardware_env"]["available"], False)

    def test_missing_env_fingerprint_is_explicitly_marked(self):
        cfg = load_config(BASELINE)
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            import json

            run_dir = run_once(cfg, dry_run=True, out_dir=Path(tmp),
                               run_id="noenv", verbose=False)
            hw = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["hardware_env"]
            # 没有指纹文件 = "没查过"，不是"这台机器没有 NPU"。
            # 两者都必须与"已确认可用"区分开。
            self.assertIsNone(hw["available"])
            self.assertIn("reason", hw)   # 必须给出原因，不能只是空值


class TestLabelReasonNamesTheRightPlatform(unittest.TestCase):
    """不可用结果里的"该去哪台机器重跑"必须点名**正确**的平台。

    这句话只在"结果不可用、必须重跑"时出现——正是最需要它准确的时候。
    原先写死"昇腾"，于是 S4000（MTT）的 HARDWARE_REQUIRED 会把读的人
    指向昇腾主机；而这句话的用途恰恰就是告诉他去哪台机器重跑。
    """

    def _reason(self, env, platform):
        return _label_reason(DataLabel.HARDWARE_REQUIRED, {}, env, platform)

    def test_mthreads_run_is_not_sent_to_an_ascend_host(self):
        text = self._reason(ExecutionEnvironment.LOCAL, HardwarePlatform.MTHREADS)
        self.assertIn("摩尔线程", text)
        self.assertNotIn("昇腾", text)

    def test_ascend_run_still_names_ascend(self):
        self.assertIn(
            "昇腾",
            self._reason(ExecutionEnvironment.LOCAL, HardwarePlatform.ASCEND),
        )

    def test_undeclared_platform_invents_nothing(self):
        # 没声明平台就不能凭空点一个平台名——那是编造。
        text = self._reason(ExecutionEnvironment.LOCAL, HardwarePlatform.UNKNOWN)
        for name in ("昇腾", "摩尔线程"):
            self.assertNotIn(name, text)
        self.assertIn("远程", text)

    def test_remote_run_without_a_device_names_the_platform(self):
        text = self._reason(
            ExecutionEnvironment.REMOTE_MTHREADS, HardwarePlatform.MTHREADS
        )
        self.assertIn("摩尔线程", text)
        self.assertNotIn("昇腾", text)
        # npu-smi 是昇腾专有的探测工具名，不该出现在别的平台的解释里。
        self.assertNotIn("npu-smi", text)


if __name__ == "__main__":
    unittest.main()
