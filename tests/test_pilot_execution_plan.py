#!/usr/bin/env python
"""pilot 三臂执行清单的回归：**跑之前**就能查住任务数。

★ 失效形态（本文件钉住的东西）
--------------------------------
`run_exec_opt_matrix` 的两个**默认值**会让人以为跑 3 项、实际跑了别的：
  · `--repeats` 默认 `None` ⇒ 退到 `--fill-budget`（3~50 轮）
  · `--baseline` 默认 `s4000_fp16.yaml` ⇒ cells = `["baseline"] + arms`
    ⇒ 若三臂全塞进 `--arms`，会变成 **4 格**
⇒ 因此这里断言：**显式 repeats=1**、**恰好 3 格**、**P1 映射到 baseline 格**，
   并且**与矩阵的 dry-plan 独立对齐**（静态断言证明不了"矩阵真会跑 3 项"）。
"""

from __future__ import annotations

import json
import importlib.util
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ROOT = Path(__file__).resolve().parents[1]
PLAN = _ROOT / "scripts" / "pilot_execution_plan.py"
MATRIX = _ROOT / "scripts" / "run_exec_opt_matrix.py"
ART = _ROOT / "results" / "processed" / "pilot_plan.json"
_SPEC = importlib.util.spec_from_file_location("pilot_plan_under_test", PLAN)
P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P)


def _run_plan(*args: str):
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(PLAN), *args], cwd=str(_ROOT),
                          capture_output=True, text=True, encoding="utf-8", env=env,
                          timeout=300)


class TestPilotPlanIsThreeRuns(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.r = _run_plan()

    def test_exit_zero(self):
        self.assertEqual(self.r.returncode, 0, self.r.stdout[-1500:] + self.r.stderr[-800:])

    def test_explicit_repeats_one(self):
        self.assertIn("repeats       : 1", self.r.stdout)

    def test_total_is_three(self):
        self.assertIn("计划任务      : 3 项", self.r.stdout)

    def test_fill_budget_is_not_used(self):
        """★ 必须显式 repeats ⇒ 不得落到 fill-budget（默认值陷阱）。

        ⚠️ 判据只看**真机命令块**，不是全文 —— 全文里有"**不**用 `--fill-budget`"
        这类**说明**，对它做 `assertNotIn` 会把写对的文档判红
        （同族：对**原文**做 `assertNotIn` 会误伤解释性注释）。
        """
        cmd_block = self.r.stdout.split("真机命令", 1)[-1].split("矩阵 dry-plan", 1)[0]
        self.assertNotIn("--fill-budget", cmd_block,
                         "真机命令里出现了 --fill-budget ⇒ 会退到 3~50 轮")
        self.assertIn("--repeats 1", cmd_block, "真机命令必须显式 --repeats 1")
        self.assertIn("fixed-repeats", self.r.stdout, "必须声明走固定轮数模式")

    def test_three_distinct_arms_listed(self):
        for arm in ("static_small", "cost_single", "cost_dual"):
            self.assertIn(arm, self.r.stdout)
        self.assertIn("baseline#r0, cost_single#r0, cost_dual#r0", self.r.stdout)

    def test_baseline_points_at_p1_not_default(self):
        """★ P1 必须走 `--baseline`，否则会多出一格（4 项）。"""
        self.assertIn("--baseline configs/mthreads/prereg_batch/static_small.yaml",
                      self.r.stdout)
        self.assertNotIn("--baseline configs/mthreads/s4000_fp16.yaml", self.r.stdout)


class TestPilotPlanKeepsFzStateConsistent(unittest.TestCase):
    """配置、文档与 FZ A/A 状态必须一致。"""

    @classmethod
    def setUpClass(cls):
        cls.r = _run_plan()

    def test_declares_fz_a_a(self):
        self.assertIn("FZ-1: A（逐步择一）", self.r.stdout)
        self.assertIn("FZ-2: A（本步没被服务）", self.r.stdout)

    def test_does_not_claim_a_is_the_final_choice(self):
        """★ A 只是**当前实现**，不得写成"已选定最优语义"。"""
        self.assertIn("不得", self.r.stdout)
        for bad in ("已选定最优", "已裁定为最优"):
            for i, ln in enumerate(self.r.stdout.splitlines(), 1):
                if bad not in ln:
                    continue
                self.assertTrue(any(k in ln for k in ("不得", "不是", "⚠️")),
                                f"L{i}: 把 A 写成了最终选择：{ln.strip()[:110]}")

    def test_pilot_is_not_for_performance_claims(self):
        self.assertIn("不用于宣布性能优势", self.r.stdout)

    def test_mechanism_verdict_only_from_cost_arms(self):
        self.assertIn("只看 P2/P3", self.r.stdout)


class TestPilotPlanCrossChecksWithMatrix(unittest.TestCase):
    """★★ 真做一遍：与 `run_exec_opt_matrix --dry-plan` 独立对齐。"""

    def test_matrix_dryplan_agrees_on_three(self):
        cmd = [sys.executable, str(MATRIX), "--tag", "PILOT-TEST", "--cfg-dir",
               "configs/mthreads/prereg_batch", "--baseline",
               "configs/mthreads/prereg_batch/static_small.yaml",
               "--arms", "cost_single", "cost_dual", "--repeats", "1", "--dry-plan"]
        r = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True,
                           encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1",
                           "PYTHONIOENCODING": "utf-8"}, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr)
        body = r.stdout[r.stdout.find("{"):]
        d = json.loads(body)
        self.assertEqual(len(d["plan"]), 3, f"矩阵计划 {d['plan']} != 3 项")
        self.assertEqual(d["cells"], ["baseline", "cost_single", "cost_dual"])
        self.assertEqual(d["repeats"], 1)

    def test_script_reports_the_cross_check(self):
        r = _run_plan()
        self.assertIn("矩阵 dry-plan 交叉核对：✓", r.stdout)


class TestPilotPlanArtifact(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.art = Path(cls.tmp.name) / "pilot_plan.json"
        result = _run_plan("--json", str(cls.art))
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.d = json.loads(cls.art.read_text(encoding="utf-8"))

    def test_lf_only_and_plan_only(self):
        self.assertNotIn(b"\r", self.art.read_bytes())
        self.assertIn("PLAN_ONLY", self.d["evidence_class"])

    def test_ok_and_three_runs(self):
        self.assertTrue(self.d["ok"], self.d.get("validation_problems"))
        self.assertEqual(self.d["n_planned_runs"], 3)
        self.assertEqual(self.d["expected_total_runs"], 3)

    def test_matrix_crosscheck_recorded(self):
        self.assertTrue(self.d["matrix_dryplan_ok"])
        self.assertIn("3", self.d["matrix_dryplan_note"])

    def test_unverified_remote_is_not_presented_as_ready(self):
        self.assertEqual(self.d["remote_identity_status"], "REQUIRED_NOT_PROVIDED")
        self.assertIsNone(self.d["remote_command"])
        self.assertIsNone(self.d["budget_minutes_expected"])
        self.assertNotIn("budget_minutes_hard_cap", self.d)
        self.assertIn("不是硬超时", self.d["budget_check_boundary"])


class TestSafeRemotePlan(unittest.TestCase):
    def test_generated_tags_are_real_safe_and_distinct(self):
        a, b = P.build_plan(), P.build_plan()
        self.assertNotEqual(a["tag"], b["tag"])
        self.assertRegex(a["tag"], r"^PILOT_\d{8}T\d{6}Z_[0-9a-f]{12}$")

    def test_tag_rejects_placeholders_and_shell_syntax(self):
        for tag in ("PILOT-<日期>", "x;echo", "../x", "x\ny", "$(echo x)"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                P.build_plan(tag=tag)

    def test_remote_command_requires_explicit_absolute_paths(self):
        with self.assertRaises(ValueError):
            P.remote_command(P.build_plan())
        for root, py in (("relative", "/env/python"), ("/repo", "python"),
                         ("/repo", None), ("/repo/../other", "/env/python")):
            with self.subTest(root=root, py=py), self.assertRaises(ValueError):
                P.build_plan(remote_root=root, remote_python=py)

    def test_remote_command_quotes_paths_and_namespaces_outputs(self):
        plan = P.build_plan(tag="PILOT_TEST_unique", remote_root="/root/project space",
                            remote_python="/root/env space/bin/python")
        cmd = P.remote_command(plan)
        args = shlex.split(cmd.replace("\\\n", " "))
        self.assertEqual(args[:4], ["cd", "--", "/root/project space", "&&"])
        self.assertEqual(args[4], "/root/env space/bin/python")
        self.assertEqual(args[args.index("--tag") + 1], plan["tag"])
        self.assertEqual(args[args.index("--repeats") + 1], "1")
        for flag in ("--progress-file", "--summary-file", "--stop-file"):
            self.assertIn(plan["tag"], args[args.index(flag) + 1])
        self.assertNotIn("--fill-budget", args)

    def test_print_command_without_paths_refuses(self):
        p = _run_plan("--print-command")
        self.assertEqual(p.returncode, 2)
        self.assertIn("remote-python", p.stderr)

    def test_print_command_and_matrix_share_actual_tag(self):
        plan = P.build_plan(tag="PILOT_exact_001", remote_root="/checked/repo",
                            remote_python="/checked/env/bin/python")
        ok, note = P.cross_check_with_matrix(plan)
        self.assertTrue(ok, note)
        p = _run_plan("--print-command", "--tag", plan["tag"],
                      "--remote-root", plan["remote_root"],
                      "--remote-python", plan["remote_python"])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--tag PILOT_exact_001", p.stdout)
        self.assertIn("不是硬超时", p.stdout)

    def test_crosscheck_rejects_wrong_cells_even_when_total_is_three(self):
        plan = P.build_plan()
        wrong = {"tag": plan["tag"], "plan": plan["planned_runs"], "repeats": 1,
                 "fill_budget": False, "budget_minutes": plan["budget_minutes"],
                 "cells": ["baseline", "cost_single", "unrelated"]}
        proc = subprocess.CompletedProcess([], 0, json.dumps(wrong), "")
        with patch.object(P.subprocess, "run", return_value=proc):
            ok, note = P.cross_check_with_matrix(plan)
        self.assertFalse(ok)
        self.assertIn("cells", note)


if __name__ == "__main__":
    unittest.main(verbosity=2)
