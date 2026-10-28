"""第 15 步**消融**配置的验收：单变量、有据、且"刻意冻结"要显式豁免。

为什么必须有这一组用例
----------------------
消融的全部价值在于**单变量**。6 份配置是生成的，但生成器只在**生成那一刻**对；
之后任何人手改一份，单变量就悄悄破坏了 —— 而产出的表看起来完全正常，
结论却归因到了错的维度（实测踩过：`analyze_step14.py` 的假警、
`test_step14_configs.py` 与它口径不一致，都是"两处各写一套"的变体）。

三条硬约束
----------
1. **reference 就是被部署的那份**：`fixed_cost_full.yaml` 逐字节等于
   `phase1_step14b/fixed_cost.yaml`（否则"消融"对照的不是系统本身）。
2. **每份与 reference 的差异恰好是声明的那一处**（按配置字段名逐一算出来，不靠人读 diff）。
3. **`chunk_only` / `budget_only` 刻意让一维不可动**（`min == max == init`），
   这是**有意的**——它们要回答的正是"把这一维拿掉会怎样"。
   ⇒ 用例显式断言"只有这两条冻结了维度，且冻结的就是声明的那一维"，
   这样任何**可动性自检**都能按 `_FROZEN_DIM` 精确豁免，而**不必**放宽检查本身。
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_benchmark import load_config  # noqa: E402

import gen_phase1_step15_configs as G  # noqa: E402

STEP15 = G.OUT_DIR
REF = G.REFERENCE

#: 每条消融**声明**改变的字段（必须与生成器一致；对不上就是生成器或文件漂了）
EXPECTED_DELTA: dict[str, set[str]] = {
    "full": set(),
    "no_host": {"host_overhead_ms", "host_overhead_fit"},
    "no_margin": {"apply_safety_margin"},
    "signal_tpot": {"signal_criterion"},
    "chunk_only": {"max_batched_tokens_min", "max_batched_tokens_max"},
    "budget_only": {"chunk_size_min", "chunk_size_max"},
}

#: 除 controller 外**必须逐字段相同**的顶层段
_OTHER_SECTIONS = ("run", "workload", "scheduler", "chunk", "cache", "runtime")


def _ctrl_delta(a, b) -> set[str]:
    return {f.name for f in dataclasses.fields(a.controller)
            if getattr(a.controller, f.name) != getattr(b.controller, f.name)}


class TestReferenceIsTheDeployedSystem(unittest.TestCase):
    def test_full_is_byte_identical_to_deployed_config(self):
        a = (STEP15 / "fixed_cost_full.yaml").read_text(encoding="utf-8")
        b = REF.read_text(encoding="utf-8")
        self.assertEqual(a, b, "reference 与 phase1_step14b/fixed_cost.yaml 不一致 ⇒ "
                              "消融对照的不是被部署的系统")

    def test_all_conditions_exist(self):
        for name in G.CONDITIONS:
            self.assertTrue((STEP15 / f"fixed_cost_{name}.yaml").is_file(),
                            f"缺少 fixed_cost_{name}.yaml")


class TestGeneratorStaysInSync(unittest.TestCase):
    def test_check_mode_passes(self):
        """文件必须与生成器输出一致 —— 手改一份就会在这里变红。"""
        rc = G.main(["--check"])
        self.assertEqual(rc, 0, "配置与生成器不一致（有人手改过？）")


class TestSingleVariable(unittest.TestCase):
    def test_each_condition_differs_in_exactly_the_declared_place(self):
        ref = load_config(REF)
        for name, want in EXPECTED_DELTA.items():
            if name == "full":
                continue
            with self.subTest(condition=name):
                cfg = load_config(STEP15 / f"fixed_cost_{name}.yaml")
                got = _ctrl_delta(cfg, ref)
                self.assertEqual(got, want,
                                 f"{name} 的 controller 差异是 {sorted(got)}，声明的是 {sorted(want)}")

    def test_non_controller_sections_are_untouched(self):
        """控制器之外的段必须**逐字段相同**（消融只动控制器）。

        ⚠️ 实现上比的是**原始 yaml 字典去掉 `controller`**，而不是枚举字段名 ——
        枚举会漂（第一版就写错了字段名，把不存在的 `.workload` 当成属性）。
        比字典则是"除了 controller 之外的一切"，覆盖面自动完整。
        """
        import copy
        ref_raw = copy.deepcopy(load_config(REF).raw)
        ref_raw.pop("controller", None)
        for name in EXPECTED_DELTA:
            cfg = load_config(STEP15 / f"fixed_cost_{name}.yaml")
            got = copy.deepcopy(cfg.raw)
            got.pop("controller", None)
            self.assertEqual(got, ref_raw,
                             f"{name}: 除 controller 之外还有字段变了 ⇒ 不只改了一个变量")

    def test_slo_and_workload_pinned(self):
        """SLO 与负载必须钉死（否则"收益"里混进了判据/负载的变化）。"""
        ref = load_config(REF)
        for name in EXPECTED_DELTA:
            cfg = load_config(STEP15 / f"fixed_cost_{name}.yaml")
            self.assertEqual(cfg.run.slo.tpot_ms, ref.run.slo.tpot_ms, name)
            self.assertEqual(cfg.run.slo.ttft_ms, ref.run.slo.ttft_ms, name)
            self.assertEqual(cfg.workload_name, ref.workload_name, name)
            self.assertEqual(cfg.workload_overrides, ref.workload_overrides, name)
            self.assertEqual(cfg.max_num_seqs, ref.max_num_seqs, name)


class TestIntentOfEachAblation(unittest.TestCase):
    def test_no_host_really_has_no_host_term(self):
        c = load_config(STEP15 / "fixed_cost_no_host.yaml").controller
        self.assertIsNone(c.host_overhead_ms)
        self.assertIsNone(c.host_overhead_fit)

    def test_reference_really_has_host_term(self):
        """反面对照：reference **必须**有 host 项，否则 `no_host` 消融的是空气。"""
        c = load_config(REF).controller
        self.assertTrue(c.host_overhead_ms, "reference 没有 host_overhead_ms ⇒ no_host 无意义")

    def test_no_margin_only_flips_the_switch(self):
        c = load_config(STEP15 / "fixed_cost_no_margin.yaml").controller
        self.assertFalse(c.apply_safety_margin)
        self.assertTrue(load_config(REF).controller.apply_safety_margin,
                        "reference 默认应**乘**余量")

    def test_signal_tpot_switches_criterion_only(self):
        c = load_config(STEP15 / "fixed_cost_signal_tpot.yaml").controller
        self.assertEqual(c.signal_criterion, "tpot_p90")
        self.assertEqual(load_config(REF).controller.signal_criterion, "decode_p90")


class TestFrozenDimensionsAreExplicit(unittest.TestCase):
    """★ `chunk_only` / `budget_only` 刻意让一维不可动 —— 必须**显式**且**可豁免**。"""

    def _movable(self, c, lo: str, hi: str, step: str) -> bool:
        return getattr(c, lo) < getattr(c, hi) and getattr(c, step) > 0

    def test_only_the_declared_two_conditions_freeze_a_dimension(self):
        frozen: dict[str, str] = {}
        for name in G.CONDITIONS:
            c = load_config(STEP15 / f"fixed_cost_{name}.yaml").controller
            if not self._movable(c, "chunk_size_min", "chunk_size_max", "chunk_size_step"):
                frozen[name] = "chunk_size"
            if not self._movable(c, "max_batched_tokens_min", "max_batched_tokens_max",
                                 "max_batched_tokens_step"):
                frozen[name] = frozen.get(name, "") + " max_batched_tokens"
        self.assertEqual(set(frozen), set(G._FROZEN_DIM),
                         f"冻结维度的条件集合是 {sorted(frozen)}，声明的豁免是 "
                         f"{sorted(G._FROZEN_DIM)} ⇒ 要么漏声明、要么有意外冻结")
        for name, dim in G._FROZEN_DIM.items():
            self.assertIn(dim, frozen[name], f"{name} 声明冻结 {dim}，实际冻结的是 {frozen[name]}")

    def test_frozen_conditions_keep_the_other_dimension_movable(self):
        """冻结一维 ≠ 两维都冻（否则这条消融等于 static，测不出"只动这一维"）。"""
        c = load_config(STEP15 / "fixed_cost_chunk_only.yaml").controller
        self.assertTrue(self._movable(c, "chunk_size_min", "chunk_size_max", "chunk_size_step"))
        c = load_config(STEP15 / "fixed_cost_budget_only.yaml").controller
        self.assertTrue(self._movable(c, "max_batched_tokens_min", "max_batched_tokens_max",
                                      "max_batched_tokens_step"))

    def test_full_system_has_both_dimensions_movable(self):
        c = load_config(STEP15 / "fixed_cost_full.yaml").controller
        self.assertTrue(self._movable(c, "chunk_size_min", "chunk_size_max", "chunk_size_step"))
        self.assertTrue(self._movable(c, "max_batched_tokens_min", "max_batched_tokens_max",
                                      "max_batched_tokens_step"))

if __name__ == "__main__":
    unittest.main()
