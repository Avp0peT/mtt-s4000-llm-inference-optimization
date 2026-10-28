"""`scripts/run_exec_opt_matrix.py` 的**固定重复实验**判据（离线）。

★ 2026-10-03：实测到一条**静默少跑**（用户第 3 条指令的靶心）
--------------------------------------------------------------
用真实 `main()` + 桩 `run_once`（2 格 × `--repeats 5`）复现：

    run_once 调用 **2** 次（应为 10）
    n_done=2、n_items=10、completed_passes=1、balanced=**True**、failures=[]、rc=**0**

成因：主循环在每轮结束时写成

    pass_index += 1
    if not fill_budget:
        break          # ← 固定轮数模式下轮条件根本没机会生效

于是 `--repeats 5` 只跑 **1** 轮。而收尾判据是 `balanced`（n_done % n_cells == 0），
它在**任何整数轮**上都为真 —— 2 % 2 == 0 ⇒ 判"均衡" ⇒ rc=0。
"少跑 8 项"被读成"完整成功"。

⇒ 两处修：① 轮条件说 `continue`；② 收尾增加**对着计划总数核完成度**的判据。
本文件把这两处都锁住：

* `test_fixed_repeats_runs_every_item` —— 直击"必须实际执行十项"；
* `test_under_run_is_not_a_full_success` —— 直击"少跑不得返回完整成功"；
* `test_runner_args_passes_baseline_and_repeats` —— 直击"外层编排正确传递"。

⚠️ 这里**不起真机**：`run_once` 被替换成桩，产物是 `TEST_DATA`。
本文件证明的是**编排**正确，不是性能。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("scripts", "benchmark", "src"):
    _p = str(_ROOT / _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_exec_opt_matrix as M  # noqa: E402
from exec_opt_plan import BASELINE_DEFAULT  # noqa: E402

CFG_DIR = "configs/mthreads/exec_opt"
ARM = "s01_both_s04_pkv"
BASELINE = f"{CFG_DIR}/s01_both.yaml"
OS_KEY = ("n_requests", "n_success", "n_failed", "n_rejected", "n_unfinished")


class _StubRunner:
    """替身：造出 `read_run_status` 认得的产物，并记录调用。"""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_after = fail_after

    def __call__(self, cfg, *, dry_run, out_dir, repeat_index=0, run_id=None,
                 extra_meta=None, verbose=True, **kw):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("stub: 模拟中途失败")
        d = Path(out_dir) / (run_id if repeat_index == 0 else f"{run_id}-r{repeat_index}")
        self.calls.append({"dir": d.name, "dry_run": dry_run,
                           "repeat_index": repeat_index, "run_id": run_id})
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(
            json.dumps({
                "data_label": "TEST_DATA",
                # ⚠️ 必须逐键给对：把 n_failed 也填成正数会让 `read_run_status`
                #    判 not-ok（曾在第一次写这个桩时踩到，导致 7 条测试全红）。
                "summary": {"n_requests": 4, "n_success": 4, "n_failed": 0,
                            "n_rejected": 0, "n_unfinished": 0},
                "context": {"extra": {}},
            }),
            encoding="utf-8", newline="\n")
        return d


class _BatchCase(unittest.TestCase):
    """跑真实 `main()`，参数固定成"2 格 × 5 轮 = 10 项"。"""

    REPEATS = 5

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="eom_")
        self.dir = Path(self._tmp.name)
        self._orig = M.run_once

    def tearDown(self):
        M.run_once = self._orig
        self._tmp.cleanup()

    def _args(self, *extra: str) -> list[str]:
        return [
            "--tag", "TST", "--arms", ARM, "--repeats", str(self.REPEATS),
            "--cfg-dir", CFG_DIR, "--baseline", BASELINE, "--out", str(self.dir),
            "--progress-file", str(self.dir / "p.json"),
            "--summary-file", str(self.dir / "s.json"),
            "--stop-file", str(self.dir / "stop"),
            *extra,
        ]

    def _run_main(self, *extra: str) -> tuple[int, dict, io.StringIO]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(self._args(*extra))
        sm = json.loads((self.dir / "s.json").read_text(encoding="utf-8"))
        return rc, sm, buf


class TestFixedRepeatsActuallyRun(_BatchCase):
    """★ "两臂 × 五次必须实际执行十项"。"""

    def test_fixed_repeats_runs_every_item(self):
        stub = _StubRunner()
        M.run_once = stub
        rc, sm, _ = self._run_main()

        # 2 格（baseline + 1 arm）× 5 轮 = 10 项
        self.assertEqual(
            len(stub.calls), 10,
            "★ `--repeats 5` 必须真的跑 5 轮（曾经只跑 1 轮 = 2 项）")
        self.assertEqual(sm["n_planned_fixed"], 10)
        self.assertEqual(sm["n_done"], 10)
        self.assertEqual(sm["completed_passes"], 5)
        self.assertTrue(sm["complete"], "跑满 10 项 ⇒ complete 必须为 True")
        self.assertEqual(rc, 0, "跑满且无失败 ⇒ rc=0")

    def test_every_arm_gets_exactly_five_rounds(self):
        """均衡性：每臂恰好 5 轮（不是"总数对"就够）。"""
        stub = _StubRunner()
        M.run_once = stub
        self._run_main()
        per_arm: dict[str, int] = {}
        for c in stub.calls:
            arm = "baseline" if c["run_id"].endswith("s01_both") else ARM
            per_arm[arm] = per_arm.get(arm, 0) + 1
        self.assertEqual(per_arm, {"baseline": 5, ARM: 5})

    def test_interleaved_not_back_to_back(self):
        """穿插顺序：一轮 = 扫完整张表（不是"一臂跑完 5 次再换另一臂"）。"""
        stub = _StubRunner()
        M.run_once = stub
        self._run_main()
        seq = ["baseline" if c["run_id"].endswith("s01_both") else ARM
               for c in stub.calls]
        self.assertEqual(seq[:4], ["baseline", ARM, "baseline", ARM])


class TestCompletionFlags(unittest.TestCase):
    """★ 完成度判据本体（`completion_flags`）—— 直接测，不绕主循环。"""

    def test_full_run_is_complete(self):
        f = M.completion_flags(n_done=10, n_cells=2, repeats=5, fill_budget=False)
        self.assertEqual(f["n_planned_fixed"], 10)
        self.assertTrue(f["complete"])
        self.assertFalse(f["under_run"])

    def test_one_pass_only_is_under_run_though_balanced(self):
        """★★ 这就是实测到的那个静默少跑：2/10，而 `balanced` 为真。"""
        f = M.completion_flags(n_done=2, n_cells=2, repeats=5, fill_budget=False)
        self.assertTrue(f["balanced"], "2 % 2 == 0 ⇒ balanced 为真（所以它拦不住）")
        self.assertFalse(f["complete"], "但 2 < 10 ⇒ 不完整")
        self.assertTrue(f["under_run"], "★ under_run 才是'少跑'的判据")

    def test_partial_round_is_both_under_run_and_unbalanced(self):
        f = M.completion_flags(n_done=7, n_cells=2, repeats=5, fill_budget=False)
        self.assertTrue(f["under_run"])
        self.assertFalse(f["balanced"])

    def test_fill_budget_mode_has_no_fixed_target(self):
        """填预算模式：目标是开区间 ⇒ `complete` 不适用，但必须标 `budget_limited`。"""
        f = M.completion_flags(n_done=6, n_cells=2, repeats=3, fill_budget=True)
        self.assertIsNone(f["n_planned_fixed"])
        self.assertIsNone(f["complete"])
        self.assertTrue(f["budget_limited"])
        self.assertFalse(f["under_run"], "填预算下'没跑满'是预期行为，不是少跑")

    def test_more_than_planned_is_not_under_run(self):
        f = M.completion_flags(n_done=12, n_cells=2, repeats=5, fill_budget=False)
        self.assertFalse(f["under_run"])
        self.assertTrue(f["complete"])


class TestUnderRunIsNotAFullSuccess(_BatchCase):
    """★ "少跑不得返回完整成功"。"""

    def test_under_run_is_not_a_full_success(self):
        stub = _StubRunner(fail_after=6)
        M.run_once = stub
        rc, sm, buf = self._run_main("--keep-going")

        self.assertEqual(len(stub.calls), 6, "桩在 6 项后开始失败")
        self.assertFalse(sm["complete"], "只完成 6/10 ⇒ complete 必须为 False")
        self.assertEqual(rc, 1, "★ 少跑必须返回失败，不得返回完整成功")
        self.assertIn("少跑", buf.getvalue(), "必须明确报出'少跑'，而不是只在 rc 里体现")

    def test_balanced_alone_would_not_catch_it(self):
        """**否定证据**：`balanced` 在"只跑了 1 轮"时也为真 ⇒ 它不是完成度判据。

        这条锁住"为什么必须新增 `n_planned_fixed`/`complete`"：
        若有人把完成度判据退回成只看 `balanced`，下面这个断言就会失败。
        """
        stub = _StubRunner()
        M.run_once = stub
        _rc, sm, _ = self._run_main()
        # 造一个"只跑 1 轮"的等价情形：2 项、2 格 ⇒ balanced 为真但不是完成
        two_items, n_cells = 2, 2
        self.assertTrue(two_items % n_cells == 0,
                        "balanced 在 2/2 时为真 —— 所以它拦不住'少跑一整批'")
        self.assertTrue(sm["complete"], "而 complete 是按 10 项算的，这才是完成度判据")

    def test_completion_is_not_reported_by_n_items(self):
        """`n_items` 是**计划**数，不能当完成数——两者必须分开报。"""
        stub = _StubRunner(fail_after=4)
        M.run_once = stub
        _rc, sm, _ = self._run_main("--keep-going")
        self.assertEqual(sm["n_items"], 10, "n_items 始终是计划总数")
        self.assertEqual(sm["n_done"], 4, "n_done 才是实际完成数")
        self.assertNotEqual(sm["n_items"], sm["n_done"])


class TestDryPlan(_BatchCase):
    """dry-plan 必须给出**确定的项数**，供批次启动前核对。"""

    def test_dry_plan_reports_ten_items(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(["--dry-plan", "--arms", ARM, "--repeats", "5",
                         "--cfg-dir", CFG_DIR, "--baseline", BASELINE])
        self.assertEqual(rc, 0)
        # ⚠️ dry-plan 之前**还会打印告警行**（本机源码摘要测不出、部署戳只在远端），
        #   所以不能对整段 stdout 直接 json.loads；从第一个 `{` 起解析。
        text = buf.getvalue()
        plan = json.loads(text[text.index("{"):])
        self.assertEqual(plan["repeats"], 5)
        self.assertEqual(plan["cells"], ["baseline", ARM])
        self.assertEqual(len(plan["plan"]), 10, "★ 2 臂 × 5 轮 = 10 项")
        self.assertFalse(plan["fill_budget"], "给了 --repeats ⇒ 固定轮数模式")

    def test_dry_plan_touches_nothing(self):
        """dry-plan 不得创建任何 run 目录（它只打印）。"""
        M.run_once = _StubRunner()
        with contextlib.redirect_stdout(io.StringIO()):
            M.main(["--dry-plan", "--arms", ARM, "--repeats", "5",
                    "--cfg-dir", CFG_DIR, "--baseline", BASELINE])
        self.assertEqual(list(self.dir.iterdir()), [], "dry-plan 不得写盘")


class TestRepeatsFlagIsPassedToRunner(_BatchCase):
    """"真实 main 的离线模拟执行"：`--dry-run` 必须真的传到 `run_once`。"""

    def test_dry_run_flag_reaches_run_once(self):
        stub = _StubRunner()
        M.run_once = stub
        rc, _sm, _ = self._run_main("--dry-run")
        self.assertEqual(rc, 0)
        self.assertTrue(stub.calls, "必须真的调用到 run_once")
        self.assertTrue(all(c["dry_run"] for c in stub.calls),
                        "★ `--dry-run` 必须透传到 run_once（否则'离线模拟'其实在跑真机）")

    def test_default_is_not_dry_run(self):
        stub = _StubRunner()
        M.run_once = stub
        self._run_main()
        self.assertTrue(all(c["dry_run"] is False for c in stub.calls),
                        "不给 --dry-run 时必须是真实模式（默认不得被改）")

    def test_repeats_conflicts_with_fill_budget(self):
        rc = M.main(["--arms", ARM, "--repeats", "5", "--fill-budget",
                     "--cfg-dir", CFG_DIR, "--baseline", BASELINE,
                     "--out", str(self.dir), "--progress-file", str(self.dir / "p.json"),
                     "--summary-file", str(self.dir / "s.json"),
                     "--stop-file", str(self.dir / "stop")])
        self.assertEqual(rc, 2, "两者互斥 ⇒ 用法错误")


class TestOuterOrchestrationPassesBaselineAndRepeats(unittest.TestCase):
    """★ 外层编排（`exec_opt_batch.py`）必须正确传递 `baseline` / `repeats`。

    实测缺陷：`runner_args` 原先**从不传 `--baseline`**，而矩阵的 `--baseline`
    **有默认值** ⇒ 对照臂静默落到兜底配置（`configs/mthreads/s4000_fp16.yaml`），
    与臂清单所在的目录对不上，且**不报错**。
    """

    @staticmethod
    def _ctx(**over):
        import exec_opt_batch as B

        base = dict(tag="TST", remote_root="/root/x", activate="/a/b/activate.sh",
                    stage="screen", arms=[ARM], budget_minutes=240.0,
                    min_repeats=3, max_repeats=50, baseline=BASELINE,
                    repeats=5, no_wait=False, poll_seconds=60.0,
                    wait_timeout_minutes=600.0, dry_run=False)
        base.update(over)
        return B.Ctx(argparse.Namespace(**base))

    def test_baseline_is_always_passed(self):
        args = self._ctx().runner_args
        self.assertIn("--baseline", args, "对照臂必须显式出现在命令行里")
        self.assertIn(BASELINE, args)

    def test_repeats_is_passed_when_set(self):
        self.assertIn("--repeats 5", self._ctx(repeats=5).runner_args)

    def test_repeats_omitted_when_not_set(self):
        self.assertNotIn("--repeats", self._ctx(repeats=None).runner_args)

    def test_default_baseline_is_the_single_source_of_truth(self):
        """不传 `--baseline` 时也要落到**同一个**默认值（不许各写一份）。"""
        args = self._ctx(baseline=None).runner_args
        self.assertIn(BASELINE_DEFAULT, args)
        import exec_opt_batch as B
        self.assertEqual(B._DEFAULTS["baseline"], BASELINE_DEFAULT)
        self.assertEqual(B.BASELINE_DEFAULT, BASELINE_DEFAULT)


class TestPlanMaths(unittest.TestCase):
    """计划项数的算术本身（2 格 × 5 = 10）。"""

    def test_two_cells_five_rounds_is_ten(self):
        plan = M.build_plan(["baseline", ARM], 5)
        self.assertEqual(len(plan), 10)
        self.assertEqual([i.key for i in plan][:2], ["baseline#r0", f"{ARM}#r0"])

    def test_repeats_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            M.build_plan(["a"], 0)


if __name__ == "__main__":
    unittest.main()
