"""S01 独立复核的**开跑前置**：参照臂的单变量性 + AB/BA 顺序表。

★ 两件事必须开跑前就钉住
------------------------
**① 参照臂**：S01 的对照必须是"**同一条执行路径**、只差两个开关"。
`configs/mthreads/s4000_fp16.yaml`（平台基线）的 `model_revision` 是 `main`（引用），
而 `s01_both.yaml` 里是**钉死的 commit** ⇒ 直接拿它们比会多一个差异。
⇒ 新增 `s01_reference.yaml` = 同基线 + **同一个钉死 revision** + 两个开关都不开。

**② 顺序表**：AB/BA 交错要求"计划 / 实际执行 / 续跑"用**同一张**表。
此前它们**各写了一遍**（`build_plan` 与执行循环里的 `for name in cells`）
⇒ 只改计划会让"计划写 ABBA、实际跑 ABAB"，而且**两边都不报错**。
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("scripts", "benchmark", "src"):
    _p = str(_ROOT / _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exec_opt_plan import ORDER_MODES, build_plan, order_for_pass  # noqa: E402
from run_benchmark import load_config  # noqa: E402

PINNED = "a09a35458c702b33eeacc393d103063234e8bc28"
EXEC_OPT = _ROOT / "configs" / "mthreads" / "exec_opt"
PLATFORM_BASELINE = _ROOT / "configs" / "mthreads" / "s4000_fp16.yaml"
REFERENCE = EXEC_OPT / "s01_reference.yaml"
S01_BOTH = EXEC_OPT / "s01_both.yaml"

S01_SWITCHES = {"runtime.logits_last_only", "runtime.sample_needed_only"}


def _flat(o, p: str = "") -> dict:
    out: dict = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flat(v, f"{p}.{k}" if p else k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(_flat(v, f"{p}[{i}]"))
    else:
        out[p] = o
    return out


class TestS01ReferenceArmIsSingleVariable(unittest.TestCase):
    def test_reference_arm_exists(self):
        self.assertTrue(REFERENCE.is_file(),
                        "必须有 s01_reference.yaml（S01 复核的对照臂）")

    def test_diff_vs_s01_both_is_exactly_the_two_switches(self):
        """★★ 靶心：参照臂与 s01_both 的**有效差异只有两个开关**。"""
        a = _flat(load_config(REFERENCE).raw)
        b = _flat(load_config(S01_BOTH).raw)
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        self.assertEqual(diff, S01_SWITCHES,
                         f"差异集必须是两个 S01 开关，实际 {sorted(diff)}")

    def test_revision_is_pinned_and_matches_s01_both(self):
        self.assertEqual(load_config(REFERENCE).raw["runtime"]["model_revision"], PINNED)
        self.assertEqual(load_config(S01_BOTH).raw["runtime"]["model_revision"], PINNED)

    def test_reference_derives_from_the_platform_baseline_with_only_the_pin(self):
        """它必须是**原 baseline 的派生**，且只把 revision 钉死（不改基线本身）。"""
        a = _flat(load_config(REFERENCE).raw)
        c = _flat(load_config(PLATFORM_BASELINE).raw)
        diff = {k for k in set(a) | set(c) if a.get(k) != c.get(k)}
        self.assertEqual(diff, {"runtime.model_revision"},
                         f"与平台基线只许差 revision，实际 {sorted(diff)}")

    def test_platform_baseline_is_not_modified(self):
        """★ 历史基线**逐字节不变**（生成器只读它）。"""
        self.assertEqual(load_config(PLATFORM_BASELINE).raw["runtime"]["model_revision"],
                         "main", "基线仍应是 `main`（引用），说明没被就地改掉")

    def test_switches_are_off_in_the_reference_arm(self):
        rt = load_config(REFERENCE).raw["runtime"]
        for key in ("logits_last_only", "sample_needed_only"):
            self.assertFalse(bool(rt.get(key)), f"{key} 在参照臂里必须关闭")

    def test_switches_are_on_in_s01_both(self):
        rt = load_config(S01_BOTH).raw["runtime"]
        for key in ("logits_last_only", "sample_needed_only"):
            self.assertTrue(bool(rt.get(key)), f"{key} 在 s01_both 里必须开启")

    def test_all_derived_configs_reproducible(self):
        """派生文件必须能被生成器**逐字节复现**（有人手改过就会不一致）。"""
        proc = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "gen_exec_opt_configs.py"), "--check"],
            capture_output=True, text=True, cwd=str(_ROOT))
        self.assertEqual(proc.returncode, 0,
                         f"派生配置与生成器不一致：\n{proc.stdout}\n{proc.stderr}")

    def test_reference_is_lf_only(self):
        self.assertEqual(REFERENCE.read_bytes().count(b"\r"), 0, "必须 LF")


class TestAbbaOrderTable(unittest.TestCase):
    """★ AB/BA 顺序表：**一张表**，计划与执行都用它。"""

    CELLS = ["baseline", "s01_both"]

    def test_abba_alternates_the_within_round_order(self):
        self.assertEqual(order_for_pass(self.CELLS, 0, order="abba"), ["baseline", "s01_both"])
        self.assertEqual(order_for_pass(self.CELLS, 1, order="abba"), ["s01_both", "baseline"])
        self.assertEqual(order_for_pass(self.CELLS, 2, order="abba"), ["baseline", "s01_both"])
        self.assertEqual(order_for_pass(self.CELLS, 3, order="abba"), ["s01_both", "baseline"])

    def test_block_is_the_historical_behaviour(self):
        """默认 `block` ⇒ 每轮同序 ⇒ 与引入本功能之前**逐字节相同**。"""
        for p in range(4):
            self.assertEqual(order_for_pass(self.CELLS, p), ["baseline", "s01_both"])

    def test_plan_for_this_batch_is_the_frozen_ten_item_table(self):
        """★★ 本批**冻结的 10 项顺序表**（计划/执行/续跑共用）。"""
        plan = [it.key for it in build_plan(self.CELLS, 5, order="abba")]
        self.assertEqual(plan, [
            "baseline#r0", "s01_both#r0",
            "s01_both#r1", "baseline#r1",
            "baseline#r2", "s01_both#r2",
            "s01_both#r3", "baseline#r3",
            "baseline#r4", "s01_both#r4",
        ])
        self.assertEqual(len(plan), 10, "两臂 × 五轮 = 10 项")

    def test_each_cell_gets_exactly_five_rounds(self):
        plan = [it.key for it in build_plan(self.CELLS, 5, order="abba")]
        for cell in self.CELLS:
            self.assertEqual(sum(1 for k in plan if k.startswith(cell + "#")), 5)

    def test_every_round_contains_every_cell_exactly_once(self):
        plan = [it.key for it in build_plan(self.CELLS, 5, order="abba")]
        for r in range(5):
            in_round = [k for k in plan if k.endswith(f"#r{r}")]
            self.assertEqual(sorted(in_round),
                             sorted(f"{c}#r{r}" for c in self.CELLS))

    def test_abba_is_not_identical_to_block(self):
        """★ **反向守卫**：若 ABBA 退化成 BLOCK，这条会红。"""
        abba = [it.key for it in build_plan(self.CELLS, 5, order="abba")]
        block = [it.key for it in build_plan(self.CELLS, 5, order="block")]
        self.assertNotEqual(abba, block, "AB/BA 必须与 BLOCK 不同（否则没起到抵消作用）")

    def test_order_mode_affects_only_odd_rounds(self):
        """★ 只反转**奇数轮**：偶数轮保持原序 ⇒ 位置效应在一半轮里反向。"""
        for p in (0, 2, 4):
            self.assertEqual(order_for_pass(self.CELLS, p, order="abba"), self.CELLS)
        for p in (1, 3):
            self.assertEqual(order_for_pass(self.CELLS, p, order="abba"),
                             list(reversed(self.CELLS)))

    def test_unknown_order_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            order_for_pass(self.CELLS, 0, order="random")

    def test_modes_are_declared(self):
        self.assertEqual(set(ORDER_MODES), {"block", "abba"})


class TestPlanAndExecutionShareOneOrderTable(unittest.TestCase):
    """★★ 用户明确要求：**计划、实际执行和续跑必须使用同一顺序表**。

    此前它们是**两处实现**：`build_plan`（计划）与执行循环里的
    `[(name, i) for i in (pass_index,) for name in cells]`。
    只改一处 ⇒ "计划写 ABBA、实际跑 ABAB"，而两边都不报错。
    """

    def test_runner_has_no_second_order_derivation(self):
        """★ 结构断言：runner 里**不许**再出现自己拼格序的写法。"""
        src = (_ROOT / "scripts" / "run_exec_opt_matrix.py").read_text(encoding="utf-8")
        self.assertNotIn("for i in (pass_index,) for name in cells", src,
                         "执行循环又自己拼了一份格序 ⇒ 计划与执行会分叉")
        self.assertIn("order_for_pass(", src,
                      "执行循环必须用 `order_for_pass`（与计划同一个函数）")

    def test_runner_passes_the_same_order_to_plan_and_execution(self):
        src = (_ROOT / "scripts" / "run_exec_opt_matrix.py").read_text(encoding="utf-8")
        self.assertIn("build_plan(cells, repeats, order=args.order)", src,
                      "快照/计划要用同一个 `--order`")
        self.assertIn("order=args.order", src,
                      "执行侧也要用同一个 `--order`（不许有一处漏传 = 默认 block）")

    def test_resume_classifies_by_item_key_not_by_position(self):
        """续跑按 `run_id`/`pass` 判已存在，因此**与顺序无关** —— 顺序变了也能正确跳过。

        （这条是"顺序表统一"能成立的前提：若续跑依赖位置，改顺序就会重复跑。）
        """
        src = (_ROOT / "scripts" / "run_exec_opt_matrix.py").read_text(encoding="utf-8")
        self.assertIn("classify_existing(", src)
        self.assertIn("key = f\"{name}#r{p}\"", src,
                      "已存在判据必须按 (name, pass) 这个**键**，不是按位置")

    def test_cli_exposes_order_with_history_preserving_default(self):
        src = (_ROOT / "scripts" / "run_exec_opt_matrix.py").read_text(encoding="utf-8")
        self.assertIn('"--order"', src)
        self.assertIn('default="block"', src,
                      "默认必须是 block（既有批次的顺序不变）")


if __name__ == "__main__":
    unittest.main()
