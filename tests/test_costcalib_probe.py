"""成本标定探针的**纯逻辑**用例（`scripts/fwd_costcalib_probe.py`，协议 `cost-calib-v1`）。

为什么只测纯逻辑：真机部分（MUSA）在测试机上根本 import 不进来。
所以探针被刻意切成两半 —— 汇总与产物组装是纯函数，设备栈**延迟到运行期**加载。
本文件正是那半分界的证据：**没有 torch_musa 也要能 import 并跑通这些断言**。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fwd_costcalib_probe as P  # noqa: E402
from costcalib_grid import load_grid  # noqa: E402


class TestImportsWithoutDevice(unittest.TestCase):
    def test_module_imports_without_musa(self):
        """★ 设备栈必须**延迟到运行期**加载。

        模块级 `import torch_musa` 会让本文件在测试机上直接 collect 失败 ⇒
        纯逻辑再也测不到（v5 就是模块级 import）。这条断言把那个设计决定钉住。
        """
        self.assertIsNone(P.musa)
        self.assertIsNone(P.torch)
        self.assertIsNone(P.torch_musa)


class TestBlockA(unittest.TestCase):
    def test_takes_block_min_not_mean(self):
        """取**块最小值**：占空比 ~100% 时最小值就是快态平台；均值会被偶发慢档污染。"""
        r = P.summarize_block_a([[62.0, 62.5, 99.0], [61.8, 62.0, 98.0]])
        self.assertEqual(r["platform_min_ms"], 61.8)

    def test_reports_block_spread_as_the_pinning_evidence(self):
        """块间极差是"这条协议是否真钉住了快态"的判据，必须报出来。"""
        r = P.summarize_block_a([[60.0], [66.0]])
        self.assertEqual(r["block_spread_ms"], 6.0)

    def test_role_is_lower_bound_only(self):
        """★ 验收第 2 条：块 A **不得**被当成拟合对象。"""
        self.assertEqual(P.summarize_block_a([[60.0]])["role"], "lower_bound_only")

    def test_empty_is_none_not_zero(self):
        r = P.summarize_block_a([])
        self.assertIsNone(r["platform_min_ms"])
        self.assertEqual(r["n_blocks"], 0)

    def test_empty_inner_blocks_are_skipped(self):
        r = P.summarize_block_a([[], [61.0]])
        self.assertEqual(r["platform_min_ms"], 61.0)
        self.assertEqual(r["n_blocks"], 1)


class TestBlockB(unittest.TestCase):
    def test_reports_quantiles_and_keeps_raw_calls(self):
        calls = [{"i": i, "ms": v} for i, v in
                 enumerate([60.0, 80.0, 90.0, 70.0])]
        r = P.summarize_block_b(calls)
        self.assertEqual(r["n_calls"], 4)
        self.assertEqual(r["p50_ms"], 75.0)
        self.assertEqual(r["max_ms"], 90.0)
        # 原始序列必须留着（不只存汇总）⇒ 事后可重算、可换分位口径
        self.assertEqual([c["ms"] for c in r["calls"]], [60.0, 80.0, 90.0, 70.0])

    def test_role_is_fitted(self):
        self.assertEqual(P.summarize_block_b([{"ms": 1.0}])["role"], "fitted")

    def test_empty_is_none_not_zero(self):
        r = P.summarize_block_b([])
        self.assertIsNone(r["p50_ms"])
        self.assertEqual(r["n_calls"], 0)


class TestArtifactRolesAreNotRewritable(unittest.TestCase):
    def test_assembler_does_not_touch_the_role_labels(self):
        """★ 组装层**不得**改写两个块的 `role`。

        一旦"哪个块是拟合对象"能在组装时被覆盖，验收第 2 条
        （不得只拟合快态最小值）就守不住了 —— 所以这条从产物端再验一次。
        """
        grid = load_grid()
        point = {"key": "b1_pf0_kv1024_nf1",
                 "A": P.summarize_block_a([[60.0]]),
                 "B": P.summarize_block_b([{"ms": 70.0}])}
        art = P.build_artifact(session_id="s", t0_monotonic_s=0.0, grid=grid,
                               grid_path="g.yaml", points=[point],
                               runtime_env={"device_name": "MTT S4000"}, complete=True)
        self.assertEqual(art["points"][0]["A"]["role"], "lower_bound_only")
        self.assertEqual(art["points"][0]["B"]["role"], "fitted")

    def test_artifact_carries_protocol_version_and_session(self):
        art = P.build_artifact(session_id="sess-1", t0_monotonic_s=12.0,
                               grid=load_grid(), grid_path="g.yaml", points=[],
                               runtime_env={}, complete=False)
        self.assertEqual(art["protocol_version"], "cost-calib-v1")
        self.assertEqual(art["session_id"], "sess-1")
        self.assertIn("RESEARCH_LOG", art["note"])   # 纪律写进产物，不只写在文档里
        self.assertNotIn("EXPERIMENTS.md 结论行", art["note"].split("不得进")[0])

    def test_artifact_fingerprint_keeps_both_expected_and_observed(self):
        """只看实测值没法判断"环境是否还是标定时那个" ⇒ 两者都要留。"""
        grid = load_grid()
        art = P.build_artifact(session_id="s", t0_monotonic_s=0.0, grid=grid,
                               grid_path="g.yaml", points=[],
                               runtime_env={"torch": "9.9"}, complete=False)
        fp = art["fingerprint"]
        self.assertEqual(fp["observed"]["torch"], "9.9")
        self.assertEqual(fp["expected"]["software"]["torch"], "2.2.0")

    def test_grid_hash_changes_with_grid(self):
        g1, g2 = load_grid(), load_grid()
        g2["protocol_version"] = "cost-calib-v2"
        a1 = P.build_artifact(session_id="s", t0_monotonic_s=0.0, grid=g1,
                              grid_path="g", points=[], runtime_env={}, complete=False)
        a2 = P.build_artifact(session_id="s", t0_monotonic_s=0.0, grid=g2,
                              grid_path="g", points=[], runtime_env={}, complete=False)
        self.assertNotEqual(a1["grid_sha256"], a2["grid_sha256"])


if __name__ == "__main__":
    unittest.main()
