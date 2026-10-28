"""``scripts/calibrate_controller_gate.py`` 的口径守卫。

这个脚本回答的是"控制器门限与实测信号**分得开吗**"。它自己必须先对：

1. **用控制器实际读到的信号**（``controller_trace[*].signals.*_p90_ms``），
   不是报表里的逐请求分位——两者不是同一个量（本例 ≈98.9 vs 110.15），
   用错了会得到相反结论；
2. **分支顺序与 ``_decide`` 一致**（``memory`` ⇒ ``tpot`` ⇒ ``ttft`` ⇒ ``recover`` ⇒ ``none``），
   写反了会把"该收缩"算成"该放大"；
3. **读不到信号必须显形并返回非零**——一格没有决策点，与"所有决策点都在带内"
   在表里长得一样；
4. **判据本身要能被回答**（2026-09-24 补）。旧版的 ``headroom min/max`` 把
   **收缩点**也算进来，而收缩点的 headroom 是很大的负数 ⇒ ``min < h < max`` 恒成立
   ⇒「跨过门槛」**恒报**，并且会**与它自己的「带内」列自相矛盾**
   （对 120/0.08 的 c4：合格点 90 个全部过门槛、带内 0，却仍报"跨越门限"）。
   旧版还把 ``n_band`` 当成一种原因，实际它是**两个相反原因**的和：
   「ttft 条件不满足 ⇒ 放大分支从未被评估」与「过了 ttft 条件但被 headroom 门槛拒」。
   ⇒ 本文件的 ``test_headroom_span_is_reported``（旧）**恰好用了收缩点**来证明"区间跨门槛"，
   等于把缺陷当成了判据 —— 它已改写为 ``test_headroom_ignores_shrink_points``
   等一组用例（含反向控制）。
5. **它不能充当"可复现性"的阻断门**（2026-09-24 补）。旧版只读**首轮**，
   而"可复现"本身要跨重复才谈得上——排除掉后续重复，等于把能回答这个问题的
   数据先扔掉了。旧版还把**一次运行内**信号落在门限两侧，直接写成
   "跨重复不可复现"（判定名就叫「抛硬币」），这是一次**未被证实的跳步**。
   ⇒ 现在读全部重复、判定改名「样本跨越门限」，并**另设** ``cross_repeat_verdict``
   专门比较"同一条件下不同重复的决策分歧"。两者的区别由
   :class:`TestCrossRepeatVerdict` 与 ``test_verdict_name_does_not_claim_reproducibility``
   钉住。

P1S3B 的 ``adaptive_c4`` 就是栽在第 1、2 条的边界上：门限 99.0 落在工作点
97.8–99.2 内部 ⇒ 放大决策的 headroom 余量只有 0.0011（门槛 0.1000）
⇒ 跨重复不可复现 ⇒ 0.2% 的观测差异造出 43% 的时延后果。
修好判据后这一点可被直接复现：``--targets 110 --hysteresis 0.1`` 报 c4
「样本跨越门限」，且离放大门限最近的**点**相距 **0.0 ms**。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    if not path.exists():
        raise unittest.SkipTest(f"缺少脚本 {path}")
    modname = f"_gate_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


class TestClassifyMatchesDecideLogic(unittest.TestCase):
    """``classify`` 必须复刻 ``AdaptiveController._decide`` 的判据与顺序。

    ``(ttft_p90, tpot_p90)``，target=100、ttft_target=500、h=0.1
    ⇒ 收缩门限 110、放大要求 tpot_p90 < 90 且 ttft_p90 > 550。
    """

    def setUp(self):
        self.mod = _load_script("calibrate_controller_gate")
        self.kw = dict(target_tpot_ms=100.0, target_ttft_ms=500.0, hysteresis=0.1)

    def test_shrink_when_tpot_over_upper_limit(self):
        r = self.mod.classify([(100.0, 200.0)], **self.kw)  # 200 > 110
        self.assertEqual((r["n_shrink"], r["n_grow"], r["n_band"]), (1, 0, 0))

    def test_grow_requires_both_ttft_over_and_tpot_headroom(self):
        # ttft 超 550 且 tpot < 90 ⇒ 放大
        r = self.mod.classify([(900.0, 80.0)], **self.kw)
        self.assertEqual((r["n_shrink"], r["n_grow"], r["n_band"]), (0, 1, 0))

    def test_ttft_over_but_no_tpot_headroom_is_band(self):
        """TTFT 超目标但 TPOT 已无余量 ⇒ **拒绝放大**，停在带内，不是放大。"""
        r = self.mod.classify([(900.0, 95.0)], **self.kw)  # headroom=0.05 <= 0.1
        self.assertEqual((r["n_shrink"], r["n_grow"], r["n_band"]), (0, 0, 1),
                         "headroom 不足时仍判为放大 —— 判据写漏了 headroom 门槛")

    def test_ttft_within_limit_does_not_grow(self):
        """TTFT 没超目标就不动，哪怕 TPOT 余量很大。"""
        r = self.mod.classify([(100.0, 50.0)], **self.kw)
        self.assertEqual((r["n_shrink"], r["n_grow"], r["n_band"]), (0, 0, 1))

    def test_shrink_takes_precedence_over_grow(self):
        """两者同时成立（tpot 爆表但 ttft 也爆）时，先走收缩——顺序不能反。"""
        r = self.mod.classify([(9999.0, 9999.0)], **self.kw)
        self.assertEqual(r["n_shrink"], 1)
        self.assertEqual(r["n_grow"], 0,
                         "顺序反了会把「该收缩」算成「该放大」")

    def test_gate_values_are_reported(self):
        r = self.mod.classify([(1.0, 1.0)], **self.kw)
        self.assertAlmostEqual(r["tpot_limit"], 110.0)      # 收缩门限 = t(1+h)
        self.assertAlmostEqual(r["grow_tpot_gate"], 90.0)   # 放大要求 < t(1-h)
        self.assertAlmostEqual(r["ttft_limit"], 550.0)

    def test_headroom_ignores_shrink_points(self):
        """**反向控制**：收缩点的 headroom 不得进入区间统计。

        ``(900, 120)`` 里 tpot=120 > 收缩门限 110 ⇒ 它是**收缩点**，根本没走到放大分支，
        它的 headroom(−0.2) 与"门槛在不在区间内部"无关。
        旧版把它算进来 ⇒ ``min < h < max`` 恒成立 ⇒「跨过门槛」**恒报**。
        """
        r = self.mod.classify([(900.0, 80.0), (900.0, 120.0)], **self.kw)
        self.assertEqual(r["n_shrink"], 1, "120 应先被收缩分支吃掉")
        self.assertEqual(r["n_eligible"], 1, "收缩点不算合格点")
        self.assertAlmostEqual(r["headroom_min"], 0.2)
        self.assertAlmostEqual(r["headroom_max"], 0.2)
        self.assertEqual(r["verdict"], "必放大",
                         "一个合格的放大点 + 一个收缩点，不是抛硬币")

    def test_verdict_spanning_needs_both_outcomes_among_eligible(self):
        """真正的「样本跨越门限」：**同为合格点**，一部分过门槛、一部分被拒。"""
        r = self.mod.classify([(900.0, 80.0), (900.0, 95.0)], **self.kw)  # 0.2 / 0.05
        self.assertEqual((r["n_grow"], r["n_gate_rejected"]), (1, 1))
        self.assertEqual(r["verdict"], "样本跨越门限")
        self.assertLess(r["headroom_min"], 0.1)
        self.assertGreater(r["headroom_max"], 0.1)

    def test_verdict_name_does_not_claim_reproducibility(self):
        """**反向控制**：判定名与说明里不得把"样本内不决断"说成"不可复现"。

        旧名「抛硬币」读起来就等于"跨重复不可复现"——而它只说明**这条合并
        序列上**门限不决断。同一次运行内负载本来就可能在工作点之间移动，
        两种结果并存完全正常。可复现性要拿同一条件下的不同重复去比
        （``cross_repeat_verdict``）。

        断言分两步：① 判定名里不出现"可复现"；② **说明文字必须显式划清界限**
        （含"不等于"）——只删掉一句话很容易，难的是让读者知道它们不是一回事。
        """
        r = self.mod.classify([(900.0, 80.0), (900.0, 95.0)], **self.kw)
        self.assertNotIn("可复现", r["verdict"])
        for bad in ("抛硬币", "不可复现"):
            with self.subTest(bad=bad):
                self.assertNotIn(bad, r["verdict"])
        doc = self.mod._VERDICTS["样本跨越门限"]
        self.assertIn("样本内部", doc)
        self.assertIn("不等于", doc, "说明里没有划清「样本内不决断」与「不可复现」")

    def test_verdict_unanimous_pass_is_decisive(self):
        r = self.mod.classify([(900.0, 80.0), (900.0, 85.0)], **self.kw)
        self.assertEqual(r["verdict"], "必放大")
        self.assertEqual(r["n_gate_rejected"], 0)

    def test_verdict_unanimous_reject_is_decisive(self):
        """合格但全部被门槛拒 ⇒ 决断（放大门限在分布之外），**不是**抛硬币。"""
        r = self.mod.classify([(900.0, 95.0), (900.0, 100.0)], **self.kw)
        self.assertEqual(r["verdict"], "必不放大")
        self.assertEqual(r["n_grow"], 0)

    def test_unreachable_is_not_a_gate_verdict(self):
        """ttft 从未超限 ⇒ **不可达**，不得与「被门槛拒」混为一谈。

        这两件事的结论相反：不可达是**负载形状**让它轮不到（P1S3 的
        ``target_ttft_ms=2000`` 就是这一类），被拒是**判据形式**把点挡在门外。
        旧口径把两者都记进 ``n_band`` ⇒ 在表里同形。
        """
        r = self.mod.classify([(100.0, 50.0)], **self.kw)
        self.assertEqual(r["verdict"], "不可达")
        self.assertEqual((r["n_not_eligible"], r["n_gate_rejected"]), (1, 0))
        self.assertEqual(r["gap_grow_ms"], None, "没有合格点 ⇒ 放大侧无点可测")

    def test_n_band_is_kept_only_as_the_sum_of_two_reasons(self):
        """``n_band`` 保留为兼容字段，且必须恒等于两个**不同原因**之和。"""
        r = self.mod.classify([(100.0, 50.0), (900.0, 95.0)], **self.kw)
        self.assertEqual((r["n_not_eligible"], r["n_gate_rejected"]), (1, 1))
        self.assertEqual(r["n_band"], r["n_not_eligible"] + r["n_gate_rejected"])

    def test_nearest_gate_gaps_are_reported(self):
        """离门限最近的**点**（不是中位数）—— 抛硬币的成因，
        也是"判定决断、但离翻转只有零点几毫秒"的唯一指示。"""
        r = self.mod.classify([(900.0, 89.7), (900.0, 110.4)], **self.kw)
        self.assertAlmostEqual(r["gap_grow_ms"], 0.3, places=6)    # |89.7 − 90|
        self.assertAlmostEqual(r["gap_shrink_ms"], 0.4, places=6)  # |110.4 − 110|

    def test_every_verdict_is_documented(self):
        cases = [[(100.0, 50.0)], [(900.0, 80.0)],
                 [(900.0, 95.0)], [(900.0, 80.0), (900.0, 95.0)]]
        for sig in cases:
            v = self.mod.classify(sig, **self.kw)["verdict"]
            self.assertIn(v, self.mod._VERDICTS,
                          f"verdict {v!r} 没在 _VERDICTS 里登记说明")


class TestEmptyInputFailsLoudly(unittest.TestCase):
    def setUp(self):
        self.mod = _load_script("calibrate_controller_gate")

    def test_missing_raw_dir_is_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertNotEqual(
                self.mod.main(["--raw", str(Path(td) / "nope"), "--prefix", "X"]), 0
            )

    def test_unknown_prefix_is_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertNotEqual(
                self.mod.main(["--raw", td, "--prefix", "NOSUCH"]), 0
            )

    def test_only_static_arms_is_nonzero(self):
        """门限只对 adaptive 臂起作用；只有 static 臂时必须显形，不能静默出空表。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "B-static_c1"
            d.mkdir()
            (d / "meta.json").write_text("{}", encoding="utf-8")
            self.assertNotEqual(
                self.mod.main(["--raw", td, "--prefix", "B"]), 0,
                "没有 adaptive 臂时仍然返回 0 —— 空表会被当成「无异常」",
            )

    def test_adaptive_run_without_signals_is_nonzero(self):
        """有 adaptive 臂但轨迹里没有可用信号 ⇒ 非零（本格未检验）。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "B-adaptive_c1"
            d.mkdir()
            (d / "meta.json").write_text(
                '{"controller_trace": []}', encoding="utf-8"
            )
            self.assertNotEqual(
                self.mod.main(["--raw", td, "--prefix", "B"]), 0
            )

    def test_all_repeats_are_read(self):
        """**全部重复**都要读，并带上重复序号与（剥掉 ``-rN`` 的）格名。

        旧版只取首轮——首轮并不更权威，砍掉 2/3 只是白白降低分辨率；
        更要紧的是，**可复现性需要跨重复比较，只看一轮根本无从比**。
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("B-adaptive_c1", "B-adaptive_c1-r1", "B-adaptive_c1-r2"):
                d = root / name
                d.mkdir()
                (d / "meta.json").write_text("{}", encoding="utf-8")
            got = [
                (cell, idx, meta.parent.name)
                for _batch, cell, idx, meta in self.mod._iter_adaptive_runs(root, ["B"])
            ]
        self.assertEqual(
            got,
            [
                ("adaptive_c1", 0, "B-adaptive_c1"),
                ("adaptive_c1", 1, "B-adaptive_c1-r1"),
                ("adaptive_c1", 2, "B-adaptive_c1-r2"),
            ],
            "三份产物都必须被读到，且格名要相同（否则归不到一组）",
        )

    def test_static_arms_are_still_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            d = root / "B-static_c1"
            d.mkdir()
            (d / "meta.json").write_text("{}", encoding="utf-8")
            self.assertEqual(self.mod._iter_adaptive_runs(root, ["B"]), [])


class TestCrossRepeatVerdict(unittest.TestCase):
    """**这才是可复现性的问法**：同一条件、不同重复的判定一致吗。

    :func:`classify` 的 ``verdict`` 说的是"**合并后的样本里**门限分不分得开"，
    它答不了可复现性——一条序列里两种结果并存，完全可能是同一次运行内负载
    本来就在两个工作点之间移动。拿它当可复现性的阻断门是一次未证实的跳步。
    """

    def setUp(self):
        self.mod = _load_script("calibrate_controller_gate")

    def test_identical_verdicts_are_consistent(self):
        label, why = self.mod.cross_repeat_verdict(["必放大", "必放大", "必放大"])
        self.assertEqual(label, "重复间一致")
        self.assertIn("3", why)

    def test_differing_verdicts_are_divergent(self):
        label, why = self.mod.cross_repeat_verdict(["必放大", "必不放大", "必放大"])
        self.assertEqual(label, "重复间分歧")
        self.assertIn("必不放大", why)

    def test_single_repeat_cannot_claim_reproducibility(self):
        """**反向控制**：一次重复无从比较 ⇒ 必须显形，不得据此说"可复现"。"""
        label, why = self.mod.cross_repeat_verdict(["必放大"])
        self.assertEqual(label, "单一重复")
        self.assertIn("不得", why)

    def test_spanning_in_one_repeat_and_decisive_in_another_is_divergent(self):
        """真实形状：一轮里样本跨越门限、另一轮里决断 ⇒ **判定不同** ⇒ 分歧。

        这正说明为什么"跨过门限的样本"与"跨重复不可复现"不能混为一谈：
        前者是**样本属性**，后者要这么比才比得出来。
        """
        label, _ = self.mod.cross_repeat_verdict(["样本跨越门限", "必放大"])
        self.assertEqual(label, "重复间分歧")


class TestLoadTraceFacts(unittest.TestCase):
    """跨重复比较必须同时看到**阶段**与**控制状态**，否则"判定相同"会被读成
    "行为相同"——一轮收缩了 4 次、另一轮一次没动，是两回事。
    """

    def setUp(self):
        self.mod = _load_script("calibrate_controller_gate")

    def _write(self, td, trace):
        import json as _json

        d = Path(td) / "B-adaptive_c1"
        d.mkdir()
        (d / "meta.json").write_text(
            _json.dumps({"controller_trace": trace}), encoding="utf-8"
        )
        return d / "meta.json"

    def test_extracts_steps_bottlenecks_and_chunk_range(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._write(td, [
                {"step": 0, "bottleneck": "none", "state": {"chunk_size": 1024}},
                {"step": 8, "bottleneck": "tpot", "state": {"chunk_size": 512}},
                {"step": 16, "bottleneck": "tpot", "state": {"chunk_size": 512}},
            ])
            facts = self.mod.load_trace_facts(meta)
        self.assertEqual(facts["run"], "B-adaptive_c1")
        self.assertEqual(facts["steps"], [0, 8, 16])
        self.assertEqual(facts["bottlenecks"], ["none", "tpot", "tpot"])
        self.assertEqual(facts["chunk_range"], (512, 1024))

    def test_missing_fields_are_not_filled_with_zero(self):
        """缺字段不猜：``chunk_size`` 不是 int 就不进范围；一个都没有则 ``None``。

        （0 是"测出来是 0"，与"没测到"相反——填 0 会让"控制器没动"与
        "没记录"在表里长得一样。）
        """
        with tempfile.TemporaryDirectory() as td:
            meta = self._write(td, [{"step": 0, "bottleneck": None, "state": {}}])
            facts = self.mod.load_trace_facts(meta)
        self.assertIsNone(facts["chunk_range"])
        self.assertEqual(facts["bottlenecks"], [])
        self.assertEqual(facts["steps"], [0])


class TestChunkMotionIsExtracted(unittest.TestCase):
    """``chunk_range`` 只是 min..max —— **不是轨迹**。

    两轮可以有**相同**的 range 而在不同**时刻**走到不同的值，这是同一 range 下
    TTFT 481 与 652 并存的成因。所以还要取出「变了几次」与终值。
    """

    def setUp(self):
        self.mod = _load_script("calibrate_controller_gate")

    def _write(self, root, name, trace, config=None):
        d = root / name
        d.mkdir()
        payload: dict = {"controller_trace": trace}
        if config is not None:
            payload["config"] = {"controller": config}
        (d / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
        return d / "meta.json"

    @staticmethod
    def _trace(chunk_sizes, bottleneck="none"):
        return [
            {"step": i * 8, "bottleneck": bottleneck,
             "signals": {"ttft_p90_ms": 100.0, "tpot_p90_ms": 50.0},
             "state": {"chunk_size": cs}}
            for i, cs in enumerate(chunk_sizes)
        ]

    def test_counts_chunk_changes_and_final(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._write(Path(td), "B-adaptive_c1",
                               self._trace([1024, 512, 512, 1024]))
            facts = self.mod.load_trace_facts(meta)
        self.assertEqual(facts["chunk_range"], (512, 1024))
        self.assertEqual(facts["chunk_changes"], 2)
        self.assertEqual(facts["chunk_final"], 1024)

    def test_same_range_can_hide_different_motion(self):
        """**反向控制**：相同 range、不同动作次数 —— 只看 range 会看成一个样。"""
        with tempfile.TemporaryDirectory() as td:
            one = self._write(Path(td), "B-adaptive_c1", self._trace([512, 1024]))
            two = self._write(Path(td), "B-adaptive_c2",
                              self._trace([512, 1024, 512]))
            a = self.mod.load_trace_facts(one)
            b = self.mod.load_trace_facts(two)
        self.assertEqual(a["chunk_range"], b["chunk_range"], "两轮的 range 应当相同")
        self.assertNotEqual(a["chunk_changes"], b["chunk_changes"],
                            "动作次数不同 ⇒ 必须能被区分出来")
        self.assertNotEqual(a["chunk_final"], b["chunk_final"])

    def test_own_thresholds_come_from_the_run_config(self):
        """核对必须用该 run **自己**的阈值，而不是 ``--targets``。"""
        with tempfile.TemporaryDirectory() as td:
            meta = self._write(
                Path(td), "B-adaptive_c1", self._trace([1024]),
                config={"target_tpot_ms": 120.0, "target_ttft_ms": 500.0,
                        "hysteresis": 0.08},
            )
            facts = self.mod.load_trace_facts(meta)
        self.assertEqual(facts["own_thresholds"], (120.0, 500.0, 0.08))

    def test_missing_thresholds_stay_none_not_defaulted(self):
        """三态：读不到阈值记 ``None``（**没查** ≠ 通过），不得用默认值顶替。

        顶替的后果很隐蔽：用 ``--targets`` 去对，两边本来就**应该**不同，
        核对会恒失败 ⇒ 守卫退化成噪声，然后被人忽略。
        """
        with tempfile.TemporaryDirectory() as td:
            meta = self._write(Path(td), "B-adaptive_c1", self._trace([1024]))
            facts = self.mod.load_trace_facts(meta)
        self.assertIsNone(facts["own_thresholds"])


class TestCheckBranchConsistency(unittest.TestCase):
    """把"``classify`` 与 ``_decide`` 一一对应"从**断言**变成**不变量**。

    ``classify`` 的 docstring 写着一一对应，而断言不被检查就会漂移：一旦
    ``_decide`` 改了分支、或本脚本抄漏一个门槛，表里的「门槛拒 42/192」就会与
    轨迹里的「``ttft×30``」各说各话，而**两边看起来都很正常**。
    """

    def setUp(self):
        self.mod = _load_script("calibrate_controller_gate")
        self.kw = dict(target_tpot_ms=100.0, target_ttft_ms=500.0, hysteresis=0.1)

    def test_gate_rejected_is_recorded_as_ttft_branch(self):
        """**核心不变量**：被 headroom 门槛拒时，轨迹记的仍是 ``ttft``。

        所以能与轨迹对上的是 ``n_grow + n_gate_rejected``，**不是** ``n_grow``。
        实测坐实：P1S3D 的 ``c8`` 一轮 ``放大=0 / 门槛拒=1``，轨迹正好 ``ttft×1``。
        """
        sigs = [(900.0, 80.0), (900.0, 95.0)]      # 1 个过门槛 + 1 个被拒
        rc = self.mod.classify(sigs, **self.kw)
        self.assertEqual((rc["n_grow"], rc["n_gate_rejected"]), (1, 1))
        ok = self.mod.check_branch_consistency(
            rc, bottlenecks=["ttft", "ttft"], n_signals=2, n_trace=2
        )
        self.assertTrue(ok["ok"], ok)

    def test_trace_with_only_one_ttft_is_drift(self):
        """若轨迹只记了一次 ``ttft``，说明实现与工具**真的**不同 ⇒ 必须显形。"""
        rc = self.mod.classify([(900.0, 80.0), (900.0, 95.0)], **self.kw)
        bad = self.mod.check_branch_consistency(
            rc, bottlenecks=["ttft"], n_signals=2, n_trace=2
        )
        self.assertFalse(bad["ok"])
        self.assertFalse(bad["checks"]["ttft_branch"])
        self.assertFalse(bad["checks"]["trace_complete"])

    def test_skipped_entry_without_signal_is_explained(self):
        """轨迹条数 > 有信号的决策点数：差额由"带内且无信号"解释掉。

        真实形状：step 0 没有有效样本（``load_signal_sequence`` 跳过），
        而它仍记了一条 ``bottleneck="none"``。这是**正常**的，不该报漂移。
        """
        rc = self.mod.classify([(100.0, 50.0)], **self.kw)
        r = self.mod.check_branch_consistency(
            rc, bottlenecks=["none", "none"], n_signals=1, n_trace=2
        )
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["checks"]["skipped_explained"])

    def test_unexplained_extra_entry_is_drift(self):
        """多了条目却解释不出来源 ⇒ 漂移（可能是"带信号却被跳过"）。"""
        rc = self.mod.classify([(100.0, 50.0)], **self.kw)
        r = self.mod.check_branch_consistency(
            rc, bottlenecks=["none"], n_signals=1, n_trace=3
        )
        self.assertFalse(r["ok"])
        self.assertFalse(r["checks"]["skipped_explained"])

    def test_shrink_count_must_match_tpot_branch(self):
        rc = self.mod.classify([(100.0, 200.0)], **self.kw)   # 200 > 110 ⇒ 收缩
        self.assertEqual(rc["n_shrink"], 1)
        r = self.mod.check_branch_consistency(
            rc, bottlenecks=["none"], n_signals=1, n_trace=1
        )
        self.assertFalse(r["checks"]["shrink"])

    def test_memory_branch_is_flagged_not_absorbed_into_band(self):
        """``memory`` 本工具判不了 ⇒ 出现即不成立，**不得**混进「带内」凑平。"""
        rc = self.mod.classify([(100.0, 50.0), (100.0, 50.0)], **self.kw)
        r = self.mod.check_branch_consistency(
            rc, bottlenecks=["memory", "none"], n_signals=2, n_trace=2
        )
        self.assertFalse(r["checks"]["unmodelled_memory"])
        self.assertFalse(r["ok"])

    def test_classify_counts_are_complete(self):
        """分类四类之和必须等于有信号的决策点数（否则是漏算一类）。"""
        rc = self.mod.classify([(100.0, 50.0), (900.0, 80.0), (100.0, 200.0)],
                               **self.kw)
        r = self.mod.check_branch_consistency(
            rc, bottlenecks=["none", "ttft", "tpot"], n_signals=3, n_trace=3
        )
        self.assertTrue(r["ok"], r)


class TestReportBranchConsistency(unittest.TestCase):
    """自检表：漂移要能被调用方拿到（并让进程非零）；读不到阈值记「不可判定」。"""

    def setUp(self):
        self.mod = _load_script("calibrate_controller_gate")

    @staticmethod
    def _rep(sigs, bottlenecks, *, own, n_trace=None):
        return {
            "repeat_index": 0,
            "batch": "B",
            "sigs": sigs,
            "bottlenecks": bottlenecks,
            "chunk_range": (512, 1024),
            "chunk_changes": 1,
            "chunk_final": 1024,
            "n_trace_entries": len(bottlenecks) if n_trace is None else n_trace,
            "own_thresholds": own,
        }

    def _run(self, rep):
        cells = [("B", "c4")]
        per_cell = {("B", "c4"): {"per_repeat": [rep]}}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            drift = self.mod.report_branch_consistency(cells, per_cell)
        return drift, out.getvalue()

    def test_consistent_run_reports_no_drift(self):
        drift, text = self._run(self._rep(
            [(900.0, 80.0)], ["none", "ttft"], own=(100.0, 500.0, 0.1)))
        self.assertEqual(drift, [])
        self.assertIn("✓ 一致", text)

    def test_drift_names_the_failing_check(self):
        """静态分类说该收缩，轨迹却记 ``none`` ⇒ 漂移，且要点名是哪条不变量。"""
        drift, _ = self._run(self._rep(
            [(100.0, 200.0)], ["none"], own=(100.0, 500.0, 0.1)))
        self.assertEqual(len(drift), 1)
        self.assertIn("shrink", drift[0])

    def test_missing_thresholds_are_undecidable_not_pass(self):
        """三态：读不到阈值 ⇒「不可判定」，**不是**「✓ 一致」，也不是漂移。"""
        drift, text = self._run(self._rep(
            [(100.0, 50.0)], ["none", "none"], own=None, n_trace=2))
        self.assertEqual(drift, [], "不可判定不该被算成漂移")
        self.assertIn("不可判定", text)
        self.assertNotIn("✓ 一致", text)


class TestMainFailsOnCaliberDrift(unittest.TestCase):
    """漂移必须让 ``main`` 返回非零 —— 否则"守卫失败"看起来像"跑完了"。"""

    CONFIG = {"controller": {"target_tpot_ms": 100.0, "target_ttft_ms": 500.0,
                             "hysteresis": 0.1}}

    def setUp(self):
        self.mod = _load_script("calibrate_controller_gate")

    def _mk(self, root, name, trace, config=None):
        d = root / name
        d.mkdir()
        payload = {"controller_trace": trace}
        if config is not None:
            payload["config"] = {"controller": config}
        (d / "meta.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_drift_makes_main_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            # tpot_p90=200 > 收缩门限 110 ⇒ 静态分类判「收缩」，
            # 而轨迹把它记成 "none" ⇒ 人为造出一处口径漂移。
            self._mk(Path(td), "B-adaptive_c4", [
                {"step": 0, "bottleneck": "none",
                 "signals": {"ttft_p90_ms": 100.0, "tpot_p90_ms": 200.0},
                 "state": {"chunk_size": 1024}},
            ], config=self.CONFIG["controller"])
            out = io.StringIO()
            err = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = self.mod.main(["--raw", td, "--prefix", "B",
                                    "--targets", "100", "--hysteresis", "0.1"])
        self.assertEqual(rc, 6, f"口径漂移必须非零；stdout={out.getvalue()}")
        self.assertIn("口径漂移", out.getvalue() + err.getvalue())

    def test_matching_trace_is_zero(self):
        """**正对照**：同样的信号、轨迹记对了分支 ⇒ 0。"""
        with tempfile.TemporaryDirectory() as td:
            self._mk(Path(td), "B-adaptive_c4", [
                {"step": 0, "bottleneck": "tpot",
                 "signals": {"ttft_p90_ms": 100.0, "tpot_p90_ms": 200.0},
                 "state": {"chunk_size": 1024}},
            ], config=self.CONFIG["controller"])
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = self.mod.main(["--raw", td, "--prefix", "B",
                                    "--targets", "100", "--hysteresis", "0.1"])
        self.assertEqual(rc, 0)


class TestCellsAreKeyedByBatch(unittest.TestCase):
    """格的身份是 ``(批次, 格名)`` —— 跨批次的同名 ``r0`` **不是**同一条件。

    只按格名分组，P1S3B 的首轮与 P2 的首轮会被并成"同一条件的两次重复"
    （实测症状：跨重复表里 c4 出现**两行 r0**，静态分类还不同，而
    ``chunk 范围`` 与瓶颈串逐字相同 ⇒ 看起来像"同条件重复却判定不同"，
    实际是**两个条件被混成一组**）。

    这里用"两批判定相反"把它变成**可见**的：一旦合并，就会造出一个
    假的「重复间分歧」。
    """

    CONFIG = {"controller": {"target_tpot_ms": 100.0, "target_ttft_ms": 500.0,
                             "hysteresis": 0.1}}

    def setUp(self):
        self.mod = _load_script("calibrate_controller_gate")

    def _mk_run(self, root, batch, cell, rep, ttft):
        name = f"{batch}-{cell}" + ("" if rep is None else f"-r{rep}")
        d = root / name
        d.mkdir()
        trace = [
            {"step": i * 8,
             "bottleneck": "ttft" if ttft > 550 else "none",
             "signals": {"ttft_p90_ms": ttft, "tpot_p90_ms": 80.0},
             "state": {"chunk_size": 1024}}
            for i in range(2)
        ]
        (d / "meta.json").write_text(
            json.dumps({"controller_trace": trace, "config": self.CONFIG}),
            encoding="utf-8",
        )

    def test_same_cell_name_in_two_batches_stays_separate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rep in (None, 1, 2):
                self._mk_run(root, "P1S3B", "adaptive_c4", rep, 900.0)  # 必放大
                self._mk_run(root, "P2", "adaptive_c4", rep, 100.0)     # 必不放大
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = self.mod.main(["--raw", td, "--prefix", "P1S3B,P2",
                                    "--targets", "100", "--hysteresis", "0.1"])
        text = out.getvalue()
        self.assertEqual(rc, 0, text)
        self.assertIn("✓ P1S3B/adaptive_c4: 重复间一致", text)
        self.assertIn("✓ P2/adaptive_c4: 重复间一致", text)
        self.assertNotIn(
            "重复间分歧", text,
            "两个批次被判成一组了 —— 跨批次同名重复不是同一条件",
        )


if __name__ == "__main__":
    unittest.main()
