"""第⑥步静态扫描配置与运行器的验收用例。**全部离线**。

要证的六件事：
1. **每个格都把 `scheduler.max_num_seqs` 显式覆盖成 8**（模板是 32 ⇒ 这条覆盖是**承重的**，
   守卫会同时检查"模板里的值不是 8"，否则断言会退化成恒真）；
2. **单变量**：除锚点外，每格与它声明的基线**只差一个点分路径**；
3. 验收点名的轴线**一条不少**（臂 / 预算 / 分块 / 并发 / 门限 / 轨迹）；
4. **piecewise 格不覆盖 `workload.num_requests`**（覆盖会把轨迹截断，而截断不报错）；
5. 生成物能**真的被 `load_config` 加载**（"生成器自检通过"不等于"配置能被读"）；
6. **反向对照**：把缺陷注回去，守卫必须报警——否则"守卫通过"不含信息。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gen_phase1_step6_configs as G  # noqa: E402
from workloads import get_workload  # noqa: E402
from run_benchmark import load_config  # noqa: E402


class TestCellTable(unittest.TestCase):
    def setUp(self):
        self.cfgs = G.build()

    def test_every_cell_pins_max_num_seqs_to_eight(self):
        self.assertTrue(self.cfgs)
        for name, cfg in self.cfgs.items():
            with self.subTest(cell=name):
                self.assertEqual(G._get_path(cfg, "scheduler.max_num_seqs"),
                                 G.MAX_NUM_SEQS)

    def test_the_override_is_load_bearing(self):
        """⚠️ 模板里的值**必须不是** 8，否则"显式覆盖成 8"这条断言恒真。

        模板哪天被改成 8，"第⑥步把并发从 32 收到 8"这个前提就再没人守。
        """
        tmpl_val = G._get_path(G._template(), "scheduler.max_num_seqs")
        self.assertNotEqual(tmpl_val, G.MAX_NUM_SEQS,
                            "模板已是 8 ⇒ 覆盖不再承重，守卫失去分辨力")
        self.assertEqual(G._assert_max_num_seqs_explicit(self.cfgs), [])

    def test_each_cell_differs_from_its_baseline_by_exactly_one_path(self):
        pairs = [c for c in G.CELLS if c.baseline]
        self.assertGreater(len(pairs), 0)
        for cell in pairs:
            with self.subTest(cell=cell.name, baseline=cell.baseline):
                diffs = G._assert_single_variable(self.cfgs, cell.baseline, cell.name)
                self.assertEqual(
                    diffs, G.CELLS and diffs,  # 保持原值，便于失败时看到
                    f"{cell.name} 与 {cell.baseline} 应只差一处，实为 {diffs}",
                )
                self.assertEqual(len(diffs), 1, f"{cell.name}: {diffs}")

    def test_all_required_axes_are_present(self):
        self.assertEqual(G._assert_axes_present(self.cfgs), [])
        axes = {c.axis for c in G.CELLS}
        for key in ("arm", "budget", "chunk_size", "concurrency", "gate", "ramp"):
            with self.subTest(axis=key):
                self.assertIn(key, axes)

    def test_ramp_cells_keep_the_piecewise_request_count(self):
        """piecewise 的请求数由 λ(t)×时长决定；写小了会把轨迹**截断**且不报错。"""
        ramp_cells = [c for c in G.CELLS if c.overrides.get("workload.name") == G.RAMP_WORKLOAD]
        self.assertGreaterEqual(len(ramp_cells), 2, "轨迹扫描需要 static/adaptive 两臂")
        self.assertEqual(G._assert_ramp_window_is_consistent(self.cfgs), [])
        derived = sum(round(d * q) for d, q, _ in G._RAMP_SEGMENTS)
        self.assertGreater(derived, G.NUM_REQUESTS)
        for cell in ramp_cells:
            with self.subTest(cell=cell.name):
                # ⚠️ 契约是"**与轨迹一致**"，不是"不写"：piecewise 的负载会自己校验，
                # 不一致会在加载时报错（或把轨迹截断）。两个值都由 _common 一起给出。
                self.assertEqual(G._common(G.RAMP_WORKLOAD)["workload.num_requests"], derived)
                self.assertEqual(G._get_path(self.cfgs[cell.name], "workload.num_requests"),
                                 derived)

    def test_recovery_window_has_enough_margin(self):
        """恢复窗口必须留足余量，否则积压清不完 ⇒ 恢复时间**删失**（不是"慢"）。

        pilot 实测：初值 recover=120 s 的余量只有 **1.05×**（drain 尾段仅 8.4 s）
        ⇒ 任何一次服务能力略低就清不完。⇒ 按 1.5× 反解 T ≥ 197 s，取 240 s（1.70×）。
        """
        margin = G._recovery_margin()
        self.assertGreaterEqual(margin, G.RECOVERY_MARGIN_MIN,
                                f"余量 {margin:.3f}× 不足")
        self.assertAlmostEqual(margin, 1.702, places=2)
        # 反向对照：把 recover 缩回 120 s，守卫必须报警
        short = ((120.0, 0.08, "low"), (90.0, 0.70, "high"), (120.0, 0.08, "recover"))
        self.assertLess(G._recovery_margin(short), G.RECOVERY_MARGIN_MIN,
                        "120 s 的余量应当被判为不足——否则守卫没有分辨力")

    def test_every_cell_carries_the_fields_the_scan_report_reads(self):
        """扫描表要读的字段必须逐格存在——缺了会静默变成 n/a。"""
        for name, cfg in self.cfgs.items():
            with self.subTest(cell=name):
                for path in ("scheduler.max_num_seqs", "chunk.enabled", "chunk.chunk_size",
                             "chunk.max_num_batched_tokens", "controller.enabled",
                             "controller.target_tpot_ms", "workload.concurrency"):
                    self.assertIsNotNone(G._get_path(cfg, path),
                                         f"{path} 缺失（∈ {name}）")


class TestGuardsHaveTeeth(unittest.TestCase):
    """**反向对照**：把缺陷注回去，守卫必须报警。"""

    def test_max_num_seqs_guard_fires_on_a_wrong_value(self):
        cfgs = G.build()
        G._set_path(cfgs["static_c8"], "scheduler.max_num_seqs", 32)
        problems = G._assert_max_num_seqs_explicit(cfgs)
        self.assertTrue(any("static_c8" in p for p in problems), problems)

    def test_max_num_seqs_guard_fires_when_the_template_would_make_it_vacuous(self):
        """模板被改成 8 时，守卫必须**报警**（这条覆盖已不再承重），而不是默默通过。"""
        cfgs = G.build()
        real = G._template
        try:
            G._template = lambda: {"scheduler": {"max_num_seqs": G.MAX_NUM_SEQS}}
            problems = G._assert_max_num_seqs_explicit(cfgs)
        finally:
            G._template = real
        self.assertTrue(any("恒真" in p or "分辨力" in p for p in problems), problems)

    def test_single_variable_guard_fires_on_a_two_field_change(self):
        cfgs = G.build()
        G._set_path(cfgs["adaptive_c8"], "chunk.chunk_size", 4096)   # 除开关外再加一处
        diffs = G._assert_single_variable(cfgs, "static_chunk_c8", "adaptive_c8")
        self.assertEqual(len(diffs), 2, diffs)

    def test_axis_guard_fires_when_a_required_axis_is_missing(self):
        """把预算格删掉 ⇒ 必须报"缺少「预算扫描」"，而不是静默通过。"""
        cfgs = {k: v for k, v in G.build().items()
                if not k.startswith("static_chunk_c8_bt")}
        problems = G._assert_axes_present(cfgs)
        self.assertTrue(any("预算扫描" in p for p in problems), problems)


class TestGeneratedFilesLoad(unittest.TestCase):
    """生成器自检通过 ≠ 配置能被读。必须真的走一遍 `load_config`。"""

    def test_each_generated_yaml_loads_and_pins_eight(self):
        d = G.OUT_DIR
        if not d.exists():
            self.skipTest(f"尚未生成：{d}（先跑 gen_phase1_step6_configs.py）")
        files = sorted(d.glob("*.yaml"))
        self.assertEqual(len(files), len(G.CELLS))
        for path in files:
            with self.subTest(cfg=path.name):
                cfg = load_config(path)
                self.assertEqual(cfg.max_num_seqs, G.MAX_NUM_SEQS)
                self.assertEqual(cfg.workload_name,
                                 "load_ramp" if path.name.startswith("ramp_") else G.WORKLOAD)

    def test_piecewise_config_keeps_the_full_trajectory(self):
        path = G.OUT_DIR / "ramp_static_c8.yaml"
        if not path.exists():
            self.skipTest("尚未生成")
        cfg = load_config(path)
        spec = get_workload(cfg.workload_name, **cfg.workload_overrides)
        self.assertEqual(spec.arrival, "piecewise")
        # 三段齐全（low/high/recover）——被截断的轨迹根本走不到恢复段
        self.assertEqual([s.label for s in spec.rate_segments], ["low", "high", "recover"])
        self.assertGreater(spec.num_requests, G.NUM_REQUESTS)


if __name__ == "__main__":
    unittest.main()
