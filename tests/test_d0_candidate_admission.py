"""`diagnose_scheduling.py` 的 **D0 判决层**用例（候选有没有被"用上"）。

为什么要有这一层
----------------
项目纪律："**没被用上 ≠ 用了没用**"。`fixed` 格上 cost 臂的回退率曾被解释成
"空闲步回退"——**错了**（空闲步 = 0；94.4% 是"batch 不在标定网格"）。
若候选**几乎全被判据拒掉**，那么"cost 臂与 threshold 表现接近"说明的是
**判据太严 / 候选没被用上**，**不是**"成本模型没用"。⇒ 判决层必须先于服务层被报出来。

用例要钉住三点：
① 三态**不合并**（可行 / 被拒 / 模型没给预测）；
② 拒因按**归一化文本**分组，但**保留逐字样本**（量级本身是结论：
   `943 ms` 与 `108 ms` 是两件不同的事，不能因为它们都被归一成同一个 key 就丢掉）；
③ 归一化后的分组必须**真的合起来** —— 数字不同的同因文本不能各占一行（那会看不出主因）。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import diagnose_scheduling as D  # noqa: E402


def _run(traces: list[dict]) -> dict:
    return {"trace": traces}


def _dec(cands: list[dict]) -> dict:
    return {"cost_policy": {"candidates": cands}}


class TestThreeStatesAreNotMerged(unittest.TestCase):
    def test_counts_feasible_rejected_and_no_prediction_separately(self):
        r = _run([
            _dec([{"feasible": True}, {"feasible": False, "reject_reason": "超每步预算 120.0ms"},
                  {"feasible": False, "reject_reason": "模型范围外（不可外推）"}]),
            _dec([{"feasible": True}]),
        ])
        d = D.d0_candidate_admission(r)
        self.assertEqual(d["n_decisions"], 2)
        self.assertEqual(d["n_candidates"], 4)
        self.assertEqual(d["n_feasible"], 2)
        self.assertAlmostEqual(d["feasible_rate"], 0.5)

    def test_no_candidates_gives_none_not_zero(self):
        """没有候选 ⇒ 可行率 `None`（不可判），**不是** 0（0 会被读成"全被拒"）。"""
        d = D.d0_candidate_admission(_run([_dec([])]))
        self.assertEqual(d["n_candidates"], 0)
        self.assertIsNone(d["feasible_rate"])

    def test_trace_without_arm_entry_is_skipped(self):
        d = D.d0_candidate_admission(_run([{"step": 1}, _dec([{"feasible": True}])]))
        self.assertEqual(d["n_decisions"], 1)


class TestReasonsAreGroupedButExamplesKept(unittest.TestCase):
    def test_different_numbers_group_together_and_keep_a_verbatim_sample(self):
        """★ 主因必须能合并显示，同时保留量级（943 与 108 不能混成一个无名数字）。"""
        r = _run([
            _dec([{"feasible": False, "reject_reason": "预测 108.6 + host 25.3 = 133.9ms 超每步预算 120.0ms"}]),
            _dec([{"feasible": False, "reject_reason": "预测 106.4 + host 21.6 = 128.0ms 超每步预算 120.0ms"}]),
            _dec([{"feasible": False, "reject_reason": "预测 943.1 + host 21.6 = 964.7ms 超每步预算 120.0ms"}]),
        ])
        d = D.d0_candidate_admission(r)
        self.assertEqual(len(d["top_reject_reasons"]), 1, "同因不同数字应合成一组")
        g = d["top_reject_reasons"][0]
        self.assertEqual(g["n"], 3)
        self.assertIn("108.6", g["example"], "必须保留**逐字**样本（量级是结论的一部分）")
        # 归一化 key 本身不该还把数字留在里面（否则分组会碎掉）
        self.assertNotIn("108.6", D.re.sub(r"\d+(?:\.\d+)?", "N", g["example"]))

    def test_distinct_reasons_stay_separate(self):
        r = _run([
            _dec([{"feasible": False, "reject_reason": "模型范围外（不可外推）"}]),
            _dec([{"feasible": False, "reject_reason": "没有加载成本模型"}]),
        ])
        d = D.d0_candidate_admission(r)
        self.assertEqual(len(d["top_reject_reasons"]), 2)

    def test_sorted_by_count_desc(self):
        r = _run([
            _dec([{"feasible": False, "reject_reason": "A"}]),
            _dec([{"feasible": False, "reject_reason": "B"}, {"feasible": False, "reject_reason": "B"}]),
        ])
        d = D.d0_candidate_admission(r)
        self.assertEqual([g["n"] for g in d["top_reject_reasons"]], [2, 1])


class TestSloWeaveArmIsAlsoScanned(unittest.TestCase):
    def test_slo_weave_entries_are_counted(self):
        """两条并列臂共用 `plan_step`；D0 必须两条都看（只看 cost 会漏掉另一条）。"""
        r = _run([{"slo_weave": {"candidates": [{"feasible": True}, {"feasible": False,
                                                                     "reject_reason": "x"}]}}])
        d = D.d0_candidate_admission(r)
        self.assertEqual(d["n_candidates"], 2)
        self.assertEqual(d["n_feasible"], 1)


class TestTruncatedProductsAreSkippedLoudly(unittest.TestCase):
    """**截断的产物必须"响亮跳过"，而不是让整脚本崩掉、也不是静默当空。**

    为什么要钉住：回拉是分文件的，中断会在本地留下**截断**的 `meta.json`
    （实测 32768 B）。`json.loads` 在那种文件上抛 `JSONDecodeError`——旧实现
    会让 `load()` 直接冒泡 ⇒ **整个分析一个数都出不来**，看起来像"脚本坏了"，
    而真相是"少了个文件"。两种错法都要避免：
    * 崩掉 ⇒ 你会去查脚本，白费时间；
    * 静默跳过且不留痕 ⇒ 分母悄悄变小，被读成"这批就这么点"（更危险）。
    正确行为：**打印 + 记录 + 返回非零**。
    """

    def _make_dir(self, root: Path, name: str, meta_text: str) -> Path:
        """`load()` 的 glob 是 `<root>/results/raw/<tag>-*` ⇒ 临时树下要造出 `results/raw/`。"""
        d = root / "results" / "raw" / name
        d.mkdir(parents=True)
        (d / "meta.json").write_text(meta_text, encoding="utf-8", newline="\n")
        return d

    def test_truncated_meta_is_skipped_and_recorded(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # 一个**可解析**的产物（用完整 JSON）+ 一个**截断**的产物
            good = json.dumps({"summary": {"run_id": "S14ZZ-load_ramp-ramp_static"}})
            self._make_dir(root, "S14ZZ-load_ramp-ramp_static", good)
            bad = good[: len(good) // 2]  # 截断：JSON 不完整
            self._make_dir(root, "S14ZZ-load_ramp-ramp_cost", bad)

            orig = D._ROOT
            try:
                D._ROOT = root
                runs = D.load("S14ZZ")
            finally:
                D._ROOT = orig

            self.assertEqual(len(runs), 1, "截断的那个不应进入 runs")
            skipped = getattr(D.load, "skipped", [])
            self.assertEqual(len(skipped), 1, "截断的必须被记录，不能静默")
            self.assertIn("cost", skipped[0][0])
            self.assertIn("meta.json", skipped[0][1])

    def test_truncated_telemetry_is_skipped_not_read_as_empty(self):
        """⚠️ 截断的 `telemetry.jsonl` **不得**被当"没有 telemetry"（那会把
        D1/D2 的分母静默改小）。必须同样跳过并记录。"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            name = "S14ZZ-load_ramp-ramp_static"
            d = self._make_dir(root, name, json.dumps({"summary": {"run_id": name}}))
            # telemetry 第一行合法、第二行截断 ⇒ 解析必定失败
            (d / "telemetry.jsonl").write_text(
                '{"step": 1, "t_ms": 10.0}\n{"step": 2, "t_ms":', encoding="utf-8", newline="\n")

            orig = D._ROOT
            try:
                D._ROOT = root
                runs = D.load("S14ZZ")
            finally:
                D._ROOT = orig

            self.assertEqual(len(runs), 0, "telemetry 截断 ⇒ 该 run 不得进入分析")
            self.assertIn("telemetry.jsonl", getattr(D.load, "skipped", [])[0][1])

    def test_no_skip_when_all_products_are_intact(self):
        """反向对照：产物都完好时必须**不**产生 skip（否则守卫会误报）。"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            name = "S14ZZ-load_ramp-ramp_static"
            d = self._make_dir(root, name, json.dumps({"summary": {"run_id": name}}))
            (d / "telemetry.jsonl").write_text('{"step": 1, "t_ms": 10.0}\n',
                                               encoding="utf-8", newline="\n")

            orig = D._ROOT
            try:
                D._ROOT = root
                runs = D.load("S14ZZ")
            finally:
                D._ROOT = orig

            self.assertEqual(len(runs), 1)
            self.assertEqual(getattr(D.load, "skipped", []), [])


if __name__ == "__main__":
    unittest.main()
