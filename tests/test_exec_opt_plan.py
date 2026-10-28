"""执行优化线批次编排纯逻辑的用例（S01–S03 真机 A/B 的前置）。

这些判据**必须能在本机验证**：它们决定"跑什么、跑多久、什么时候停、从哪续"。
只在远端跑过而不在本地验过的判据，出问题时只能靠读日志猜。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exec_opt_plan import (  # noqa: E402
    build_plan,
    decide_stop,
    decide_wait_exit,
    plan_summary,
    progress_payload,
    read_json,
    resolve_arms,
    run_dir_name,
    write_json_atomic,
)

# 编排层（`exec_opt_batch.py`）没有 `__all__`，直接取需要的符号。
# ⚠️ 它 import 时不触网（子命令才触网）⇒ 可以在单测里安全导入。
from exec_opt_batch import (  # noqa: E402
    STAMP_OK_MARKER,
    _DEFAULTS,
    Ctx,
    find_bogus_todo,
    parse_push_summary,
    parse_push_todo,
    push_summary_complaint,
    stamp_complaint,
)

# 续跑的身份核验在矩阵运行器里（也是纯函数）。
from run_exec_opt_matrix import identity_complaint  # noqa: E402

# 矩阵运行器里的心跳（`Heartbeat` 也是纯逻辑，不触网）。
# ⚠️ 它 import `run_benchmark`，后者需要在 sys.path 里有 benchmark/（本文件已加）。
from run_exec_opt_matrix import Heartbeat  # noqa: E402


class TestInterleavedOrder(unittest.TestCase):
    """★ 判据 1：穿插，不背靠背。"""

    def test_order_is_pass_major(self):
        plan = build_plan(["base", "a", "b"], repeats=3)
        self.assertEqual(
            [it.key for it in plan],
            ["base#r0", "a#r0", "b#r0",
             "base#r1", "a#r1", "b#r1",
             "base#r2", "a#r2", "b#r2"],
        )

    def test_not_back_to_back(self):
        """反例：任一格的相邻两次不得紧挨着（那正是背靠背）。"""
        plan = build_plan(["a", "b", "c"], repeats=4)
        keys = [it.key for it in plan]
        for i in range(1, len(keys)):
            self.assertNotEqual(keys[i - 1].split("#")[0], keys[i].split("#")[0],
                                "同一格出现了相邻两次 ⇒ 退化成背靠背")

    def test_every_cell_gets_equal_repeats(self):
        plan = build_plan(["a", "b", "c"], repeats=5)
        counts = {}
        for it in plan:
            counts[it.cell] = counts.get(it.cell, 0) + 1
        self.assertEqual(set(counts.values()), {5})

    def test_rejects_zero_repeats(self):
        with self.assertRaises(ValueError):
            build_plan(["a"], repeats=0)

    def test_plan_summary_is_serializable(self):
        json.dumps(plan_summary(["a", "b"], 2), ensure_ascii=False)


class TestRunDirNaming(unittest.TestCase):
    """断点续跑靠它 ⇒ 必须与 `run_once` 的落盘规则逐字一致。"""

    def test_first_pass_has_no_suffix(self):
        self.assertEqual(run_dir_name("T-short_chat-a", 0), "T-short_chat-a")

    def test_later_passes_use_rN(self):
        self.assertEqual(run_dir_name("T-short_chat-a", 3), "T-short_chat-a-r3")


class TestStopDecision(unittest.TestCase):
    """★ 判据 2：预算不足时**宁可少一整轮**，也不跑半轮。"""

    def test_stop_file_wins(self):
        d = decide_stop(elapsed_s=0, budget_s=None, items_done=0, items_total=9,
                        avg_item_s=None, remaining_items_this_pass=3,
                        stop_file_exists=True)
        self.assertTrue(d.stop)
        self.assertIn("STOP", d.reason)

    def test_all_done_stops(self):
        d = decide_stop(elapsed_s=10, budget_s=None, items_done=9, items_total=9,
                        avg_item_s=1.0, remaining_items_this_pass=0,
                        stop_file_exists=False)
        self.assertTrue(d.stop)

    def test_no_budget_never_stops_on_time(self):
        d = decide_stop(elapsed_s=1e6, budget_s=None, items_done=1, items_total=9,
                        avg_item_s=600.0, remaining_items_this_pass=3,
                        stop_file_exists=False)
        self.assertFalse(d.stop)

    def test_stops_when_current_pass_cannot_finish(self):
        # 已用 60 min，预算 90 min，本轮还需 3 项 × 20 min = 60 min ⇒ 停
        d = decide_stop(elapsed_s=3600, budget_s=5400, items_done=3, items_total=9,
                        avg_item_s=1200.0, remaining_items_this_pass=3,
                        stop_file_exists=False)
        self.assertTrue(d.stop)
        self.assertIn("均衡", d.reason)

    def test_keeps_going_when_pass_fits(self):
        # 已用 60 min，预算 300 min，本轮还需 3 × 20 = 60 min ⇒ 继续
        d = decide_stop(elapsed_s=3600, budget_s=18000, items_done=3, items_total=9,
                        avg_item_s=1200.0, remaining_items_this_pass=3,
                        stop_file_exists=False)
        self.assertFalse(d.stop)

    def test_no_estimate_does_not_stop(self):
        """还没跑完任何一项（没有用时估计）⇒ **不**因预算停，否则一次都跑不了。"""
        d = decide_stop(elapsed_s=10, budget_s=60, items_done=0, items_total=9,
                        avg_item_s=None, remaining_items_this_pass=3,
                        stop_file_exists=False)
        self.assertFalse(d.stop)

    def test_margin_is_applied(self):
        """边界：刚好等于预算 ⇒ 不算够（留 15% 余量），必须停。"""
        d = decide_stop(elapsed_s=0.0, budget_s=3600.0, items_done=1, items_total=9,
                        avg_item_s=1200.0, remaining_items_this_pass=3,
                        stop_file_exists=False)
        self.assertTrue(d.stop)   # 3×1200×1.15 = 4140 > 3600


class TestMultiPassProgression(unittest.TestCase):
    """★ 回归：**多轮批次不得被"已完成项数"误判为提前结束**。

    历史 bug（`scripts/run_exec_opt_matrix.py`，真机 EXECOPT2 复现）：
    轮末报告行写成 `total_planned = len(cells) * (pass_index + 1)`，
    **复用**了判据要用的 "计划总数" 变量。于是
      r0 结束（pass_index→1）后 `total_planned` 从 `4*3=12` 变成 `4*1=4`，
      r1 结束（pass_index→2）时 `decide_stop(items_done=8, items_total=4)`
      ⇒ `8 >= 4` ⇒ 返回 **"全部项已完成"** ⇒ 3 轮只跑 2 轮。
    症状是**静默**的：`failures=[]`、日志无错、summary 里
    `n_done=8 / n_items=12 / completed_passes=2`。
    ⇒ 这里把"正确的判据序列"钉死：只要传**计划总数**，任何中途轮次都不该停。
    """

    CELLS = ["baseline", "s01_both", "s02_no_empty_cache", "s03_fuse_gate_up"]

    def _decide_after_pass(self, pass_index_done: int, *, items_total: int,
                           budget_s: float = 4 * 3600.0):
        """模拟"跑完 pass_index_done 轮之后"的那次轮末判据（预算极充裕）。"""
        done = pass_index_done * len(self.CELLS)
        return decide_stop(
            elapsed_s=done * 176.6,            # ≈2.94 min/项
            budget_s=budget_s,
            items_done=done,
            items_total=items_total,
            avg_item_s=176.6,
            remaining_items_this_pass=len(self.CELLS),
            stop_file_exists=False,
        )

    def test_three_pass_plan_never_stops_early(self):
        """3 轮（12 项）：跑完第 1、2 轮后都**必须继续**。"""
        for done_passes in (1, 2):
            with self.subTest(done_passes=done_passes):
                d = self._decide_after_pass(done_passes, items_total=12)
                self.assertFalse(
                    d.stop,
                    f"跑完 {done_passes} 轮后不该停（共 3 轮）：{d.reason}",
                )

    def test_completed_passes_count_is_the_bug_shape(self):
        """把 bug 的**错误输入**也钉住：若把 items_total 传成"已完成轮数×臂数"，
        第 2 轮后必然被误判为完成 —— 这条用例存在的意义是证明
        `test_three_pass_plan_never_stops_early` **不是恒真**。"""
        # r1 结束：pass_index=2 ⇒ 旧代码把 items_total 写成 4*2=8，而 items_done=8
        d = self._decide_after_pass(2, items_total=8)
        self.assertTrue(d.stop, "错误输入下应当（且确实）误停 —— 这是 bug 的形状")
        self.assertIn("完成", d.reason)

    def test_final_pass_does_stop(self):
        """跑满 3 轮后必须停（否则会无限跑）。"""
        d = self._decide_after_pass(3, items_total=12)
        self.assertTrue(d.stop)

    def test_budget_still_governs_when_plan_is_long(self):
        """计划总数传对了，预算仍是硬约束：预算不够跑完下一轮 ⇒ 停（判据 2 不被削弱）。"""
        d = decide_stop(
            elapsed_s=0.0, budget_s=800.0,          # 13.3 min 预算
            items_done=4, items_total=1000,          # 计划很长
            avg_item_s=176.6, remaining_items_this_pass=4,
            stop_file_exists=False,
        )
        self.assertTrue(d.stop)
        self.assertIn("均衡", d.reason)


class TestProgressPayload(unittest.TestCase):
    def test_fields_are_stable_and_serializable(self):
        plan = build_plan(["a", "b"], 2)
        p = progress_payload(tag="T", plan=plan, done_keys=["a#r0"], current="b#r0",
                             started_epoch=100.0, now_epoch=160.0)
        self.assertEqual(p["n_total"], 4)
        self.assertEqual(p["n_done"], 1)
        self.assertEqual(p["elapsed_s"], 60.0)
        self.assertEqual(p["heartbeat_epoch"], 160.0)
        json.dumps(p, ensure_ascii=False)

    def test_counts_only_done_keys(self):
        """进度只数"产物判定完成"的项 —— 传进来的 done_keys 就是那批。"""
        plan = build_plan(["a"], 3)
        p = progress_payload(tag="T", plan=plan, done_keys=["a#r0", "a#r1"],
                             current="a#r2", started_epoch=0.0, now_epoch=1.0)
        self.assertEqual(p["n_done"], 2)


class TestWaitExitDecision(unittest.TestCase):
    """★ 本机等待侧的退出判据：**三条通道都必须识别**。

    实测失效（EXECOPT3，2026-09-30）：只查 `progress.stop.stop` ⇒
    预算收尾（不写 `stop`）时条件永假 ⇒ 本机轮询到 `--wait-timeout-minutes`
    （默认 600 min）为止，而远端 35 min 就写完了 summary。
    """

    def test_summary_is_the_authoritative_exit(self):
        """★ 预算收尾：无 `stop`、有 summary ⇒ **必须**退出（这就是踩过的形态）。"""
        d = decide_wait_exit(
            progress={"n_done": 12, "n_total": 12, "stop": None},
            summary={"n_done": 12, "n_items": 12, "completed_passes": 3,
                     "balanced": True},
        )
        self.assertTrue(d.done)
        self.assertEqual(d.channel, "summary")

    def test_old_logic_would_hang(self):
        """**否定测试**：把判据退回"只认 stop.stop" ⇒ 同一输入**不**退出。

        这条用例的作用是证明上面的用例**确实在测新通道**；
        若有人把实现改回只查 `stop`，本用例会通过而 `test_summary_*` 会失败 ——
        两个一起，说明"新通道是必需的"。
        """
        progress = {"n_done": 12, "n_total": 12, "stop": None}
        summary = {"n_done": 12, "n_items": 12}
        old_logic = bool((progress.get("stop") or {}).get("stop"))
        self.assertFalse(old_logic, "旧判据在预算收尾时不会退出 ⇒ 会空转")
        self.assertTrue(decide_wait_exit(progress=progress, summary=summary).done,
                        "新判据必须能退出")

    def test_all_items_is_redundant_fallback(self):
        """summary 读不到时，`n_done>=n_total` 兜底。"""
        d = decide_wait_exit(progress={"n_done": 12, "n_total": 12}, summary=None)
        self.assertTrue(d.done)
        self.assertEqual(d.channel, "all_items")

    def test_cooperative_stop_still_wins(self):
        d = decide_wait_exit(
            progress={"n_done": 3, "n_total": 12,
                      "stop": {"stop": True, "reason": "发现 STOP 文件"}},
            summary=None,
        )
        self.assertTrue(d.done)
        self.assertEqual(d.channel, "stopped")

    def test_still_running_does_not_exit(self):
        """未跑满、无 summary、无 stop ⇒ **不**退出（否则会提前收尾）。"""
        d = decide_wait_exit(progress={"n_done": 4, "n_total": 12}, summary=None)
        self.assertFalse(d.done)
        self.assertEqual(d.channel, "")

    def test_nothing_readable_does_not_exit(self):
        """两个文件都读不到（刚起跑 / 控制面抖动）⇒ 不退出。"""
        self.assertFalse(decide_wait_exit(progress=None, summary=None).done)

    def test_partial_n_done_does_not_exit(self):
        """★ 边界：n_done 恰好差一项（例如最后一项在跑）⇒ 不退出。"""
        self.assertFalse(
            decide_wait_exit(progress={"n_done": 11, "n_total": 12}, summary=None).done
        )

    def test_zero_total_is_not_treated_as_complete(self):
        """`n_total=0` 不许被读成"全完成"（恒定值假通过）。"""
        self.assertFalse(
            decide_wait_exit(progress={"n_done": 0, "n_total": 0}, summary=None).done
        )


class TestAtomicWriteAndRead(unittest.TestCase):
    def test_write_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sub" / "progress.json"
            write_json_atomic(p, {"a": 1})
            self.assertEqual(read_json(p), {"a": 1})

    def test_no_leftover_tmp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "progress.json"
            write_json_atomic(p, {"x": 1})
            write_json_atomic(p, {"x": 2})
            self.assertEqual(sorted(x.name for x in Path(tmp).iterdir()),
                             ["progress.json"])

    def test_missing_file_returns_default_not_empty_dict(self):
        """★ "读不到"与"读到空对象"必须可分。"""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(read_json(Path(tmp) / "nope.json"))

    def test_corrupt_file_returns_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{ 这不是 json", encoding="utf-8")
            self.assertIsNone(read_json(p))


class TestResolveArms(unittest.TestCase):
    def _mk(self, tmp: str, names) -> Path:
        d = Path(tmp) / "cfg"
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / f"{n}.yaml").write_text("run: {}\n", encoding="utf-8")
        return d

    def test_sorted_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._mk(tmp, ["b", "a", "c"])
            self.assertEqual(resolve_arms(d), ["a", "b", "c"])
            self.assertEqual(resolve_arms(d), ["a", "b", "c"])

    def test_explicit_include_keeps_given_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._mk(tmp, ["a", "b", "c"])
            self.assertEqual(resolve_arms(d, include=["c", "a"]), ["c", "a"])

    def test_exclude(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._mk(tmp, ["a", "b", "c"])
            self.assertEqual(resolve_arms(d, exclude=["b"]), ["a", "c"])

    def test_unknown_arm_is_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._mk(tmp, ["a"])
            with self.assertRaises(ValueError) as ctx:
                resolve_arms(d, include=["zzz"])
            self.assertIn("zzz", str(ctx.exception))

    def test_empty_dir_is_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "empty"
            d.mkdir()
            with self.assertRaises(FileNotFoundError):
                resolve_arms(d)


class TestBatchArmsPassthrough(unittest.TestCase):
    """★ `exec_opt_batch.py --arms` 必须**真的进到**矩阵脚本的命令行里。

    背景（2026-09-30）：`--stage screen` 是**写死的 S01–S03 三臂**，
    而 S04 的臂（`s04_pkv` / `s01_both_s04_pkv`）不在预设内。
    若编排层不把 `--arms` 透传下去，起跑命令会**静默**跑成"上一批的三臂"，
    产物照样成功（failures=[]）——正是"报告成功但没测到东西"的形状。
    """

    @staticmethod
    def _ctx(**over):
        a = argparse.Namespace(
            tag="T", remote_root="/root/x", activate="/root/x/venv/activate.sh",
            stage="screen", arms=None, budget_minutes=240.0,
            min_repeats=3, max_repeats=50, no_wait=False,
            poll_seconds=60.0, wait_timeout_minutes=600.0, dry_run=False,
        )
        for k, v in over.items():
            setattr(a, k, v)
        return Ctx(a)

    def test_arms_appear_in_runner_args(self):
        args = self._ctx(arms=["s01_both", "s01_both_s04_pkv", "s04_pkv"]).runner_args
        self.assertIn("--arms s01_both s01_both_s04_pkv s04_pkv", args)

    def test_arms_reach_the_real_command_string(self):
        args = self._ctx(arms=["s04_pkv"]).runner_inner
        self.assertIn("--arms s04_pkv", args)

    def test_no_arms_omits_the_flag(self):
        """**否定测试**：不给 `--arms` 时**不得**凭空出现这个选项。

        没有这条，上面的用例可能因为默认值恰好等于期望而恒真。
        """
        args = self._ctx(arms=None).runner_args
        self.assertNotIn("--arms", args)

    def test_does_not_reuse_the_catch_all_key_name(self):
        """★ 回归：默认表 / argparse 的键名必须是 `arms`。

        踩过：默认表写成了 `active_arms`，而 `Ctx` 读 `a.arms` ⇒
        两者永不一致，且**不报错**（默认表只是兜底）。
        """
        self.assertIn("arms", _DEFAULTS)
        self.assertNotIn("active_arms", _DEFAULTS)


class TestMatrixSwitchManifest(unittest.TestCase):
    """★ summary 里的 `switches` 必须覆盖 S04 的新开关。

    踩过的坑的镜像：字段清单不覆盖新形状 ⇒ 键恒缺失，而"缺失"与"没开"
    在报告里看不出区别（`--dry-run` 也验不出来）。
    """

    def test_switch_key_list_includes_s04(self):
        src = (_ROOT / "scripts" / "run_exec_opt_matrix.py").read_text(
            encoding="utf-8")
        self.assertIn("persistent_kv_slots", src)
        self.assertIn("persistent_kv_capacity", src)

    def test_s04_configs_carry_the_switches(self):
        cfg_dir = _ROOT / "configs" / "mthreads" / "exec_opt"
        for name in ("s04_pkv", "s01_both_s04_pkv"):
            text = (cfg_dir / f"{name}.yaml").read_text(encoding="utf-8")
            self.assertIn("persistent_kv_slots", text, name)

    def test_combination_arm_only_adds_pkv_keys(self):
        """单变量：`s01_both_s04_pkv` 相对 `s01_both` 只多 **2 个配置键**。

        这不是格式美化——组合臂若顺带改了别的开关，对照就不再是单变量。
        ⚠️ 只比**键行**（跳过注释与空行）：注释说明"这条臂是什么"，
           本来就该不同；把它算进来会让断言变成在测注释。
        """
        cfg_dir = _ROOT / "configs" / "mthreads" / "exec_opt"

        def keys(name: str) -> set[str]:
            out = set()
            for ln in (cfg_dir / f"{name}.yaml").read_text(
                    encoding="utf-8").splitlines():
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                out.add(s.split(":")[0])
            return out

        added = keys("s01_both_s04_pkv") - keys("s01_both")
        self.assertEqual(sorted(added),
                         ["persistent_kv_capacity", "persistent_kv_slots"])
        # 反向：不许**悄悄删掉**基线的键（那也是"单变量"的破坏）。
        removed = keys("s01_both") - keys("s01_both_s04_pkv")
        self.assertEqual(sorted(removed), [])


class TestHeartbeatDoesNotClobberProgress(unittest.TestCase):
    """★ 心跳**不得**把 run 边界写的进度覆盖掉。

    实测失效（EXECOPT4，2026-09-30）：`Heartbeat` 持有一份**构造时**的
    `snapshot(None)`，每 30 s 原样写回 ⇒ `n_done`/`current`/`elapsed_s`
    被打回初始值，而 run 边界刚写的真实值被覆盖。
    后果：本机轮询看到的进度**时真时假**，`decide_wait_exit` 的
    `n_done>=n_total` 这条冗余通道因此不可靠。
    """

    def test_provider_is_called_each_tick(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "progress.json"
            state = {"current": None}
            seen: list = []

            def provider():
                seen.append(state["current"])
                return {"current": state["current"], "n_done": 0}

            hb = Heartbeat(provider, p, interval=0.02)
            hb.start()
            try:
                state["current"] = "a#r0"      # ⇐ run 边界"写入"的值
                time.sleep(0.15)
            finally:
                hb.stop()
            self.assertGreaterEqual(len(seen), 2, "心跳没有跑够次数 ⇒ 用例无效")
            self.assertEqual(seen[-1], "a#r0", "心跳写出了过期值")

    def test_written_file_carries_latest_value(self):
        import json as _json
        import time
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "progress.json"
            state = {"current": None}
            hb = Heartbeat(lambda: {"current": state["current"]}, p, interval=0.02)
            hb.start()
            try:
                state["current"] = "b#r1"
                time.sleep(0.15)
            finally:
                hb.stop()
            self.assertEqual(_json.loads(p.read_text(encoding="utf-8"))["current"],
                             "b#r1")

    def test_frozen_snapshot_style_would_clobber(self):
        """**否定测试**：内联一份"冻结快照"式的旧实现，证明它确实会覆盖成旧值。

        没有这条，上面两条可能只是"看似测到了"，无法说明改动解决了什么。
        """
        import json as _json
        import time
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "progress.json"
            frozen = {"current": None}          # 构造时的快照（旧写法）
            hb = Heartbeat(lambda: dict(frozen), p, interval=0.02)
            hb.start()
            try:
                write = {"current": "a#r0"}     # run 边界写真实值
                p.write_text(_json.dumps(write), encoding="utf-8")
                time.sleep(0.15)
            finally:
                hb.stop()
            after = _json.loads(p.read_text(encoding="utf-8"))
            self.assertIsNone(after["current"],
                              "旧写法本应把真实值覆盖成 None —— 这是它的形状")


class TestPushTodoParsing(unittest.TestCase):
    """★ 回归：待推送清单**不得**从人类报告里"取缩进行"来解析。

    实测事故（2026-09-30，EXECOPT4）：`step_push` 用「以两个空格开头的整行」当路径，
    而 `batch_sha_diff.py` 的人类报告里
      * `sha 不同` 的行带后缀 `local=<sha> remote=<sha>`；
      * 末尾还有「建议命令」区块，其续行也以两个空格开头。
    ⇒ 解析出的是**伪路径** ⇒ `push_files.sh` 判「跳过（不存在）」⇒
      `N=0 / ok=0 / skip=0 / fail=0` ⇒ **退出码 0** ⇒ 编排照常起跑，
      **远端一个文件都没更新**（症状：整批 `switches` 缺新机制的键，而 `failures=[]`）。
    """

    #: 逐字取自真机输出（不要凭记忆手写）。
    REAL_REPORT = """远端根：/root/ascend_probe_d1
本地命中 134 个文件（扩展名 ['py', 'sh', 'yaml']）
远端返回 133 个 sha

一致(skip)=130  远端缺失=1  sha不同=3

--- 远端缺失 ---
  scripts/analyze_exec_opt_batch.py

--- sha 不同 ---
  scripts/exec_opt_batch.py  local=24838c0e539c remote=1ef3bba2c6a5
  scripts/run_exec_opt_matrix.py  local=5ed75268ae49 remote=9cab54bf0561

=== 建议命令（只推差异）===
bash scripts/push_files.sh --include py,sh,yaml,yml /root/ascend_probe_d1 \\
  scripts/analyze_exec_opt_batch.py \\
  scripts/exec_opt_batch.py \\
  # end
"""

    #: `--emit-paths` 的真实形状（每行一个路径，无后缀、无区块）。
    REAL_EMIT = """scripts/analyze_exec_opt_batch.py
scripts/exec_opt_batch.py
scripts/run_exec_opt_matrix.py
"""

    def test_emit_paths_yields_only_real_paths(self):
        todo = parse_push_todo(self.REAL_EMIT)
        self.assertEqual(todo, ["scripts/analyze_exec_opt_batch.py",
                                "scripts/exec_opt_batch.py",
                                "scripts/run_exec_opt_matrix.py"])

    def test_old_parsing_is_provably_wrong(self):
        """**否定测试**：把人类报告喂给同一个解析器 ⇒ 必然产出伪路径。

        这条用例的存在是为了证明上面的用例**确实在测新格式**；
        若有人把 `--emit-paths` 去掉、退回解析报告，本用例仍绿，
        而 `test_emit_paths_yields_only_real_paths` 会因为拿到后缀串而失败。
        """
        grep_like = [ln.strip() for ln in self.REAL_REPORT.splitlines()
                     if ln.startswith("  ") and ln.strip()
                     and not ln.strip().startswith("#")]
        # 确实能挑出"看起来像路径"的行……
        self.assertTrue(any(l.startswith("scripts/") for l in grep_like))
        # ……但其中含伪路径（带 sha 后缀）与命令续行。
        fake = [l for l in grep_like
                if not (Path(".") / l).is_file()]
        self.assertGreaterEqual(len(fake), 2, "旧解析法应当产出伪路径")

    def test_bogus_detection_catches_suffixed_and_continuation(self):
        todo = parse_push_todo(self.REAL_REPORT)
        bogus = find_bogus_todo(todo, root=_ROOT)
        joined = "\n".join(bogus)
        self.assertIn("local=", joined)
        self.assertIn("\\", joined)

    def test_crlf_input_is_normalised(self):
        """★ CRLF 容错：Windows 上 Python stdout 会把 `\\n` 翻成 `\\r\\n`。

        实测（2026-09-30）：`--emit-paths` 的输出行尾曾是 `^M$`，
        shell 侧 `$(...)` 展开不会剥掉 `\\r` ⇒ 每个路径变成
        `scripts/x.py\\r` ⇒ push 判「不存在」⇒ 又是静默不推。
        （源头已改为 LF 输出；这里再保一层：解析器必须剥掉 `\\r`。）
        """
        todo = parse_push_todo("scripts/a.py\r\nscripts/b.py\r\n")
        self.assertEqual(todo, ["scripts/a.py", "scripts/b.py"])
        self.assertFalse(any("\r" in t for t in todo))


class TestPushSummaryGuard(unittest.TestCase):
    """★ 守卫 2：`push_files.sh` 说"完成"时，必须真覆盖了清单。"""

    def test_parses_real_summary_line(self):
        out = ("[1/5] OK    scripts/a.py  (123 B, try 1)\n"
               "----\npush_files: ok=5 skip=0 fail=0 (共 5)\n")
        self.assertEqual(parse_push_summary(out), (5, 0, 0, 5))

    def test_unreadable_summary_is_a_complaint(self):
        self.assertIsNotNone(push_summary_complaint(None, 5))

    def test_full_coverage_is_clean(self):
        self.assertIsNone(push_summary_complaint((5, 0, 0, 5), 5))

    def test_skip_counts_as_coverage(self):
        """diff 与 push 之间文件可能变得一致 ⇒ skip 是合法覆盖。"""
        self.assertIsNone(push_summary_complaint((3, 2, 0, 5), 5))

    def test_n_less_than_todo_is_loud(self):
        """★ 核心：清单 5 个但只展开 0 个（全被判「跳过（不存在）」）⇒ 必须报错。

        这正是事故形态：退出码 0、`fail=0`，但**什么都没推**。
        """
        msg = push_summary_complaint((0, 0, 0, 0), 5)
        self.assertIsNotNone(msg)
        self.assertIn("未覆盖", msg)

    def test_fail_is_loud(self):
        self.assertIsNotNone(push_summary_complaint((4, 0, 1, 5), 5))

    def test_inconsistent_stats_are_loud(self):
        self.assertIsNotNone(push_summary_complaint((3, 0, 0, 5), 5))


class TestStampGuard(unittest.TestCase):
    """★ 守卫 3：溯源戳不过 ⇒ **禁止起跑**。

    实测（EXECOPT4）：退出码被 `check=False` 丢掉 ⇒ 12 份产物全是
    `data_label=MEASURED` + `git_commit=NOT_MEASURED` + `git_source=stamp_stale`
    （"有测量标签、没有源码身份"），而编排照常走到起跑。
    """

    OK_OUT = ("投放 SOURCE_PROVENANCE.json → /root/x\n"
              "投放并校验通过：远端以 /root/musavenv/bin/python3 重算摘要相符，"
              "读回 commit=a09a35458c70\n")

    def test_success_marker_and_zero_rc_is_clean(self):
        self.assertIsNone(stamp_complaint(0, self.OK_OUT))

    def test_nonzero_rc_is_loud(self):
        for rc in (4, 6):
            with self.subTest(rc=rc):
                self.assertIsNotNone(stamp_complaint(rc, ""))

    def test_zero_rc_without_marker_is_loud(self):
        """**否定测试**：退出码 0 但没有成功标记 ⇒ 仍必须拦下来。

        没有这条，"只看退出码"就会放过"没验过"的情形。
        """
        self.assertIsNotNone(stamp_complaint(0, "投放 SOURCE_PROVENANCE.json → /root/x\n"))
        self.assertIn("成功标记", stamp_complaint(0, "x"))

    def test_marker_constant_matches_remote_run(self):
        """标记常量必须与 `remote_run.py` 真正打印的那串一致（防改名漂移）。"""
        src = (_ROOT / "scripts" / "remote_run.py").read_text(encoding="utf-8")
        self.assertIn(STAMP_OK_MARKER, src)


class TestResumeIdentityCheck(unittest.TestCase):
    """★ 断点续跑必须核**身份**（源码摘要/配置/模型），不能只数"请求跑完没"。

    ★★ 第二轮修订（2026-10-02）把三处"缺席即通过"改成了"缺席即拒绝"：
      * 旧 `if src is not None and src not in _TRUSTED...` —— 没字段就放行；
      * 旧 `if snap.is_file() and cfg_file.is_file()` —— 少一个就不比；
      * 旧 `if model_revision and got and ...` —— 任一为空就不比。
    三处都把"测不出"读成了"没问题"。下面每条都带**反向对照**（缺席必须红）。
    """

    _DIGEST = "d7a47c803c9d47d6e8182211a1db1541514f3aab8a504946e8a5c63ddb247c77"

    def _mk_run(self, tmp: Path, *, git_source=None, git_commit=None,
                model_revision=None, snapshot: bytes | None = None,
                tree_digest="__default__", name: str = "T-short_chat-arm") -> Path:
        d = tmp / name
        d.mkdir(parents=True, exist_ok=False)
        repro = {"git_source": git_source, "git_commit": git_commit,
                 "model_revision": model_revision}
        if tree_digest != "__omit__":
            repro["tree_digest"] = self._DIGEST if tree_digest == "__default__" else tree_digest
        (d / "meta.json").write_text(
            json.dumps({"reproducibility": repro}, ensure_ascii=False), encoding="utf-8")
        if snapshot is not None:
            (d / "config.snapshot.yaml").write_bytes(snapshot)
        return d

    def _cfg(self, tmp: Path, body: bytes = b"run: {}\n") -> Path:
        p = tmp / "arm.yaml"
        p.write_bytes(body)
        return p

    def _ok(self, d, cfg, *, model_revision="rev1", source_digest=_DIGEST):
        return identity_complaint(d, arm="arm", cfg_file=cfg,
                                  model_revision=model_revision,
                                  source_digest=source_digest)

    # -------- 正向：全都对上才放行 --------

    def test_trusted_source_and_matching_config_is_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            d = self._mk_run(t, git_source="stamp", git_commit="abc123",
                             model_revision="rev1", snapshot=body)
            self.assertIsNone(self._ok(d, self._cfg(t, body)))

    # -------- 源码身份（摘要级）--------

    def test_stamp_stale_is_rejected(self):
        """★ 这正是 EXECOPT4 那 12 份产物的形态。"""
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            d = self._mk_run(t, git_source="stamp_stale", git_commit="NOT_MEASURED",
                             model_revision="rev1", snapshot=body)
            why = self._ok(d, self._cfg(t, body))
            self.assertIsNotNone(why)
            self.assertIn("stamp_stale", why)

    def test_source_digest_drift_is_rejected(self):
        """★ 核心反向对照：**commit 相同、摘要不同** ⇒ 必须拒。

        旧判据只比 commit，这条会**静默通过**——那正是"88 项未提交改动"下的
        常规形态：两支代码共用同一个分支点，产物却出自另一棵树。
        """
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            d = self._mk_run(t, git_source="stamp", git_commit="abc123",
                             model_revision="rev1", snapshot=body,
                             tree_digest="0" * 64)
            why = self._ok(d, self._cfg(t, body))
            self.assertIsNotNone(why)
            self.assertIn("源码摘要不同", why)

    def test_missing_tree_digest_is_rejected_not_skipped(self):
        """**缺席即拒绝**：产物没有 `tree_digest`（旧产物/戳没带摘要）⇒ 拒。

        旧判据在这里会因为"没得比"而放行——把"测不出"读成"没问题"。
        """
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            d = self._mk_run(t, git_source="stamp", git_commit="abc123",
                             model_revision="rev1", snapshot=body,
                             tree_digest="__omit__")
            why = self._ok(d, self._cfg(t, body))
            self.assertIsNotNone(why)
            self.assertIn("tree_digest", why)

    def test_missing_source_digest_side_is_rejected(self):
        """本机求不出摘要（没有可用部署戳）⇒ 拒**任何**既有产物。"""
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            d = self._mk_run(t, git_source="stamp", git_commit="abc123",
                             model_revision="rev1", snapshot=body)
            why = self._ok(d, self._cfg(t, body), source_digest=None)
            self.assertIsNotNone(why)
            self.assertIn("无法", why)

    def test_git_source_without_digest_is_conservatively_rejected(self):
        """★ `git_source="git"` 在**摘要**这一关必然拒绝（有意为之，不是漏洞）。

        `detect_source_identity` 对真 git 树**刻意不给摘要**，于是产物里
        `tree_digest=None` ⇒ 命中 C 那条 ⇒ 拒。冻结这个行为：真 git 树上做续跑
        本就没有可比的内容摘要，此时"核不了"必须读成"不许复用"。
        """
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            d = self._mk_run(t, git_source="git", git_commit="abc123",
                             model_revision="rev1", snapshot=body,
                             tree_digest="__omit__")
            why = self._ok(d, self._cfg(t, body))
            self.assertIsNotNone(why)
            self.assertIn("tree_digest", why)

    def test_missing_git_source_is_rejected(self):
        """**缺席即拒绝**：产物连 `git_source` 都没有（旧产物）⇒ 拒。"""
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            d = self._mk_run(t, model_revision="rev1", snapshot=body)
            why = self._ok(d, self._cfg(t, body))
            self.assertIsNotNone(why)
            self.assertIn("源码身份**缺失**", why)

    # -------- 配置身份 --------

    def test_config_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            d = self._mk_run(t, git_source="stamp", git_commit="abc",
                             model_revision="rev1", snapshot=b"run: {}\n")
            why = self._ok(d, self._cfg(t, b"run: {a: 1}\n"))
            self.assertIsNotNone(why)
            self.assertIn("配置已变", why)

    def test_missing_snapshot_is_rejected(self):
        """产物没有 `config.snapshot.yaml` ⇒ 拒（旧判据 `snap.is_file()` 为假就跳过）。"""
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            d = self._mk_run(t, git_source="stamp", git_commit="abc",
                             model_revision="rev1", snapshot=None)
            why = self._ok(d, self._cfg(t, b"run: {}\n"))
            self.assertIsNotNone(why)
            self.assertIn("config.snapshot.yaml", why)

    def test_missing_current_cfg_is_rejected(self):
        """当前配置**不存在** ⇒ 拒（旧判据 `cfg_file.is_file()` 为假就跳过）。

        ★ 这正对应真实的 bug：baseline 臂的 `cfg_file` 曾被写成
        `cfg_dir/baseline.yaml`（该文件不存在），于是配置身份**从来不比**。
        """
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            d = self._mk_run(t, git_source="stamp", git_commit="abc",
                             model_revision="rev1", snapshot=body)
            why = self._ok(d, t / "does_not_exist.yaml")
            self.assertIsNotNone(why)
            self.assertIn("不存在", why)

    # -------- 模型身份 --------

    def test_model_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            d = self._mk_run(t, git_source="stamp", git_commit="abc",
                             model_revision="OLD", snapshot=body)
            why = self._ok(d, self._cfg(t, body), model_revision="NEW")
            self.assertIsNotNone(why)
            self.assertIn("revision", why)

    def test_missing_model_revision_either_side_is_rejected(self):
        """两侧任一为空 ⇒ 拒（旧判据里 `model_revision and got` 会短路跳过）。"""
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            body = b"run: {}\n"
            # 当前配置没给 revision
            d = self._mk_run(t, git_source="stamp", git_commit="abc",
                             model_revision="rev1", snapshot=body)
            why = self._ok(d, self._cfg(t, body), model_revision=None)
            self.assertIsNotNone(why)
            self.assertIn("model_revision", why)
            # 产物没给 revision
            d2 = self._mk_run(t, git_source="stamp", git_commit="abc",
                              model_revision=None, snapshot=body, name="T2-arm")
            why2 = self._ok(d2, self._cfg(t, body), model_revision="rev1")
            self.assertIsNotNone(why2)
            self.assertIn("model_revision", why2)

    # -------- 缺失 meta --------

    def test_missing_meta_is_not_a_complaint(self):
        """`meta.json` 根本不在 ⇒ 不在这里判死（交给 `incomplete` 分支，成因不同）。

        ★ 注意与上面那些"缺席即拒绝"的区别：**整个 meta 都不在** = "这次 run 没落盘"，
        是 `incomplete`；而 **meta 在、身份字段不在** = "落盘了但证不出身份"，必须拒。
        两者修法不同，不能合并成一个失败态。
        """
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            d = t / "T-short_chat-arm"
            d.mkdir()
            self.assertIsNone(self._ok(d, self._cfg(t)))


class TestResumeIdentityRegressionOnRealArtifact(unittest.TestCase):
    """用**真实历史产物**做一次反向对照。

    EXECOPT4 的 12 份产物 `git_source=stamp_stale`、`git_commit=NOT_MEASURED`、
    且没有 `tree_digest`——新判据必须把它们**全部**判为不可复用。
    这是"新守卫真的会拦住旧产物"的证据，而不是我对旧产物的印象。
    """

    def test_execopt4_artifact_is_rejected(self):
        import glob
        cands = sorted(glob.glob(str(_ROOT / "results" / "raw" / "EXECOPT4-*" / "meta.json")))
        if not cands:
            self.skipTest("本机没有 EXECOPT4 产物可作对照")
        rejected = 0
        for mp in cands:
            d = Path(mp).parent
            snap = d / "config.snapshot.yaml"
            why = identity_complaint(
                d, arm="x", cfg_file=snap, model_revision="a09a35458c702b33eeacc393d103063234e8bc28",
                source_digest="b" * 64)
            self.assertIsNotNone(why, f"{d.name} 竟被判为可复用")
            rejected += 1
        self.assertGreaterEqual(rejected, 1)


class TestPushIncludePositionGuard(unittest.TestCase):
    """★ `--include` 放错位置必须**报错退出**，不得静默退回默认扩展名。

    2026-10-02 实测（真实代价）：把 `--include py,sh,yaml,yml,json` 写在位置参数**之后**
    ⇒ 脚本第 28 行的 while 循环在第一个位置参数处 `break` ⇒ `--include` 留在 `"$@"` 里
    **被当成路径** ⇒ 打出「跳过（不存在）：--include」⇒ `INCLUDE` **静默退回 `py,sh`**
    ⇒ 整批 `.yaml/.yml/.json` **一个都没进清单**，而输出形如
    `ok=87 skip=120 fail=0`、**退出码 0** ⇒ 看起来完全成功。
    后果：远端配置落后（`configs/mthreads/phase1_step15_v5/` 整目录缺失，
    configs 只 138 vs 本地 144），而**没有任何告警**。

    ⚠️ 这就是 PLAYBOOK §1 记录的「代码是新的、配置是旧的」的**实际成因**，
    也是「静默降级成默认值」这一失效模式的一个具体形态。
    ⇒ 判据：位置参数里出现 `--include` / `--include=…` 形态 ⇒ **退出码 2 + 明确提示**。
    """

    SCRIPT = (
        Path(__file__).resolve().parent.parent / "scripts" / "push_files.sh"
    )

    @staticmethod
    def _bash() -> str | None:
        """找一个**可用的 Git Bash**；找不到就跳过（本类只验参数解析，不验远端）。

        ⚠️⚠️ **绝不要把裸 `"bash"` 放进候选列表。**
        在 Windows 上它可能解析到 `C:\\Program Files\\WSL\\wsl.exe`，而本机安全策略
        把 wsl.exe 列进了程序黑名单 —— 后果**不是"这条探测失败"**，而是
        **整个进程被 SIGTERM 掉**（实测：整轮 pytest 直接死掉，连汇总行都没有）。
        `except (OSError, SubprocessError)` **接不住**这种死法。
        ⇒ 只探**明确的 Git Bash 路径**（PortableGit 版本目录会随升级变，故用 glob，
          不硬编码版本号；与 `scripts/exec_opt_batch._find_git_bash()` 同一策略）。
        """
        cands: list[str] = []
        base = Path.home() / ".workbuddy" / "binaries" / "PortableGit"
        if base.is_dir():
            for ver in sorted(base.iterdir(), reverse=True):
                for sub in ("usr/bin/bash.exe", "bin/bash.exe"):
                    cands.append(str(ver / sub))
        cands += [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
        for cand in cands:
            if not Path(cand).is_file():
                continue
            try:
                p = subprocess.run(
                    [cand, "-c", "echo ok"],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=20,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if p.returncode == 0 and "ok" in p.stdout:
                return cand
        return None

    def _bash_run(self, args: list[str], *, timeout: int = 60):
        """跑脚本，返回 ``CompletedProcess``。**调用方负责确保参数里带 `--dry-run`**。"""
        sh = self._bash()
        if sh is None:
            self.skipTest("本机没有可用的 bash（或其在安全黑名单中）")
        return subprocess.run(
            [sh, str(self.SCRIPT), *args],
            capture_output=True, text=True,
            # ★ 脚本 stderr 在中文 Windows 上可能是 GBK；用 errors="replace"
            #   避免"编码"伪装成"脚本坏了"（2026-10-02 实测踩到）。
            errors="replace", timeout=timeout,
        )

    def _dry_run(self, *args: str) -> tuple[int, str]:
        """★ **真正离线**跑脚本：参数里带 `--dry-run`。

        ⚠️ 旧做法是"`Popen` 读首行后 `terminate()`"，并据此声称"不联网"——
        那是**证不出来的**：它只说明"进程在那之后没有继续输出"，
        而连接**可能已经建过**（甚至在首行打印之前）。
        ⇒ 现在改成：给脚本 `--dry-run`，它在**第一个 `remote_run.py` 调用之前**
        `exit 0` ⇒ 后面的行**读不到执行**，"不联网"是**构造上**成立的。
        （该性质另有 `test_dry_run_never_touches_network` 用不可达主机做**外部**核验。）
        """
        proc = self._bash_run(["--dry-run", *args])
        return proc.returncode, proc.stdout + proc.stderr

    def _dry_run_placed_last(self, *args: str) -> tuple[int, str]:
        """把 `--dry-run` **故意放在最后**（错序），返回 ``rc, out``。"""
        proc = self._bash_run(list(args))
        return proc.returncode, proc.stdout + proc.stderr

    def test_include_after_positionals_is_rejected(self):
        # 这是 2026-10-02 踩到的**真实错序**，逐字复刻。
        rc, out = self._dry_run(
            "/root/ascend_probe_d1",
            "configs/mthreads/phase1_step15_v5",
            "--include",
            "py,sh,yaml,yml,json",
        )
        self.assertEqual(rc, 2, f"错序必须退出码 2，实际 {rc}；输出：\n{out}")
        self.assertIn("--include 必须放在", out)
        # ★ 否定断言：不得再出现"静默降级"的旧症状。
        self.assertNotIn("跳过（不存在）：--include", out)
        self.assertNotIn("待推送", out)

    def test_include_eq_form_after_positionals_is_rejected(self):
        # `--include=…` 单参数形态同样必须被拦（不能只拦分开写的形态）。
        rc, out = self._dry_run(
            "/root/ascend_probe_d1", "configs", "--include=yaml"
        )
        self.assertEqual(rc, 2, f"错序（=形态）必须退出码 2，实际 {rc}；输出：\n{out}")
        self.assertIn("--include 必须放在", out)

    def test_missing_include_falls_back_to_default_silently(self):
        """正向对照：**不写** `--include` 时，退回默认 `py,sh` 是**设计行为**，不拦。

        ⚠️ 这条同时是上面两条的**判别力证明**：如果守卫写成"见到任何
        `--include` 字样就拒"，它会在**合法调用**上也触发 —— 本条会红。
        """
        rc, out = self._dry_run("/root/ascend_probe_d1", "configs/mthreads/phase1_step15_v5")
        self.assertEqual(rc, 0, f"合法调用应 rc=0，实际 {rc}；输出：\n{out}")
        self.assertNotIn("--include 必须放在", out)
        # 默认扩展名是 `py,sh` ⇒ 该目录下没有 .py/.sh ⇒ 清单为 0，但**流程要跑完**。
        self.assertIn("push_files(dry-run): 共", out, f"dry-run 应完整跑完；输出：\n{out}")

    def test_correct_order_passes_argument_validation(self):
        """正向对照：`--include` 在前 ⇒ **不得**触发守卫，且扩展名**真的被应用**。

        ★ 判据不是"看到了某行"，而是 **N 等于该目录下匹配扩展名的真实文件数** ——
        这才证明 `--include` **真的被应用**（而不是退回默认 `py,sh` 后的数字）。
        """
        target_dir = (
            Path(__file__).resolve().parent.parent
            / "configs" / "mthreads" / "phase1_step15_v5"
        )
        if not target_dir.is_dir():
            self.skipTest(f"用于正向对照的目录不存在：{target_dir}")
        want = len([p for p in target_dir.iterdir()
                    if p.is_file() and p.suffix in (".yaml", ".yml")])
        self.assertGreater(want, 0, "对照目录里没有 .yaml/.yml，本用例将失去判别力")

        rc, out = self._dry_run(
            "--include", "py,sh,yaml,yml,json",
            "/root/ascend_probe_d1", "configs/mthreads/phase1_step15_v5",
        )
        self.assertEqual(rc, 0, f"dry-run 应 rc=0，实际 {rc}；输出：\n{out}")
        self.assertNotIn("--include 必须放在", out)
        # ★ 关键断言：N 必须是**真实匹配数** ⇒ 证明扩展名被应用了。
        m = re.search(r"待推送\s+(\d+)\s+个文件", out)
        self.assertIsNotNone(m, f"无法从输出解析待推送数量：\n{out}")
        self.assertEqual(
            int(m.group(1)), want,
            f"`--include` 未生效：报告 {m.group(1)} 个，目录里实有 {want} 个 "
            f".yaml/.yml ⇒ 扩展名被退回默认 `py,sh` 了",
        )
        # dry-run 必须**列全清单**（不是只报个数），且不建连接。
        self.assertIn("dry-run 清单：", out)
        self.assertIn("未建立任何连接", out)
        listed = [ln for ln in out.splitlines() if ln.startswith("configs/")]
        self.assertEqual(len(listed), want, f"清单行数 {len(listed)} ≠ {want}")

    def test_dry_run_after_positionals_is_rejected(self):
        """★ `--dry-run` 放错位置必须报错 —— 否则它**静默失效**（比 `--include` 更危险）。

        `--include` 失效的后果是"少推了配置"；`--dry-run` 失效的后果是
        **"你以为在干跑，其实在真推"** —— 调用方据此以为"已核对过清单"，
        而实际已经动了远端。
        """
        rc, out = self._dry_run_placed_last(
            "/root/ascend_probe_d1", "configs", "--dry-run"
        )
        self.assertEqual(rc, 2, f"错序必须退出码 2，实际 {rc}；输出：\n{out}")
        self.assertIn("--dry-run 必须放在", out)

    def test_dry_run_never_touches_network(self):
        """★ **证明 dry-run 真的不联网**（而不是"停得早"）。

        判据取**可观测的外部事实**，不靠推断：
        ① 把 `PY` 指向一个**不存在**的解释器路径 —— 真联网版本**必然**失败
           （它要靠 `remote_run.py` 建连），dry-run 版本**照样跑完**；
        ② 用**错误的远端主机**（不可解析的域名）—— 同理；
        ③ 设一个很短的 `timeout`，dry-run 必须**远快于**它。
        三条都过 ⇒ "没建连接"是被证过的，不是被假设的。
        """
        sh = self._bash()
        if sh is None:
            self.skipTest("本机没有可用的 bash（或其在安全黑名单中）")
        env = dict(os.environ)
        # ① 让 `PY` 指向不存在的位置：dry-run 不该用到它。
        #    脚本里 PY 是硬编码常量，所以改为**整体替换**解释器路径不可行；
        #    这里改用"不可达主机 + 极短超时"来证明（见下）。
        start = time.monotonic()
        proc = subprocess.run(
            [sh, str(self.SCRIPT), "--dry-run", "--include", "yaml",
             "//192.0.2.1:1", "configs/mthreads/phase1_step15_v5"],
            capture_output=True, text=True, errors="replace",
            env=env, timeout=60,
        )
        elapsed = time.monotonic() - start
        out = proc.stdout + proc.stderr
        # ② 不可达主机（TEST-NET-1，RFC 5737）**不得**让它失败或变慢。
        self.assertEqual(
            proc.returncode, 0,
            f"dry-run 对不可达主机也应 rc=0（说明没建连接）；实际 {proc.returncode}：\n{out}",
        )
        self.assertIn("未建立任何连接", out)
        # ③ 真联网版本对不可达主机会卡在 TCP 超时（脚本里 timeout 45/120 s）。
        #    dry-run 必须在**数量级**上更快。
        self.assertLess(
            elapsed, 20.0,
            f"dry-run 耗时 {elapsed:.1f}s ⇒ 疑似真的在等网络（应远小于 20 s）",
        )


if __name__ == "__main__":
    unittest.main()
