"""执行优化批次分析器（`scripts/analyze_exec_opt_batch.py`）的用例。

★ 本文件的核心不是"能不能出表"，而是**读不到与读到 0 必须可分**。
踩过的形状：字段路径按记忆写成 `meta["execution_optimizations"]`（不存在）
⇒ 恒取到 `None` ⇒ 表里全是空值而脚本不报错，看起来像"这些臂没有该开销"。
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_exec_opt_batch import (  # noqa: E402
    DRIVER_STATS,
    coverage_of,
    decompose_delta,
    load_runs,
    summarize,
    tail_single_decode,
)


def _meta(*, pack=1000.0, fwd=5000.0, pkv=None, runtime=None) -> dict:
    """造一份最小 meta（形状逐字对齐真机产物）。"""
    eo: dict = {}
    if pkv is not None:
        eo["persistent_kv_slots"] = pkv
    return {
        "config": {"runtime": runtime or {}},
        "context": {"extra": {"backend_stats": {
            "driver_stats": {
                "pack_ms_total": pack,
                "fwd_ms_total": fwd,
                "n_forwards": 10,
                "execution_optimizations": eo,
            }
        }}},
    }


def _write(dirpath: Path, meta: dict) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")


class TestLoadRunsArmGrouping(unittest.TestCase):
    def test_strips_repeat_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            for name in ("T-short_chat-a", "T-short_chat-a-r1", "T-short_chat-a-r2"):
                _write(raw / name, _meta())
            runs = load_runs(raw, "T")
            self.assertEqual(list(runs), ["a"])
            self.assertEqual(len(runs["a"]), 3)
            # r0 必须排在 r1/r2 之前（自然序）
            self.assertTrue(runs["a"][0].name.endswith("chat-a"))

    def test_arm_with_underscore_and_dash_is_one_arm(self):
        """臂名自带 `_` 与 `-`（如 `s02_no_empty_cache`）不得被拆成两臂。"""
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            _write(raw / "T-short_chat-s02_no_empty_cache", _meta())
            _write(raw / "T-short_chat-s02_no_empty_cache-r1", _meta())
            runs = load_runs(raw, "T")
            self.assertEqual(list(runs), ["s02_no_empty_cache"])
            self.assertEqual(len(runs["s02_no_empty_cache"]), 2)

    def test_ignores_dirs_without_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            (raw / "T-short_chat-x").mkdir(parents=True)
            self.assertEqual(load_runs(raw, "T"), {})


class TestMissingVsZero(unittest.TestCase):
    """★ 本文件存在的理由：把"没有该机制"与"该机制 0 次回退"分开。"""

    def test_absent_pkv_block_reports_none_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "T-short_chat-a"
            _write(d, _meta(pkv=None))
            row = summarize("a", [d])["rows"][0]
            self.assertFalse(row["has_pkv"])
            self.assertIsNone(row["n_fastpath"], "缺失不得被读成 0")
            self.assertIsNone(row["n_fallback"], "缺失不得被读成 0")
            self.assertIsNone(row["reasons_sum"], "缺失不得被读成 0")

    def test_present_but_zero_pkv_reports_zero(self):
        """反向对照：**有**该块且计数为 0 ⇒ 必须报 0（否则上面的用例恒真）。"""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "T-short_chat-b"
            _write(d, _meta(pkv={"requested": True, "effective": True,
                                 "n_fastpath_steps": 0, "n_fallback_steps": 0,
                                 "fallback_reasons": {}, "bytes_allocated": 0}))
            row = summarize("b", [d])["rows"][0]
            self.assertTrue(row["has_pkv"])
            self.assertEqual(row["n_fastpath"], 0)
            self.assertEqual(row["n_fallback"], 0)
            self.assertEqual(row["reasons_sum"], 0)

    def test_wrong_path_yields_none(self):
        """**否定测试**：把 driver_stats 挪走 ⇒ pack 读出必须是 `None`。

        这条证明"读得到"不是因为脚本到处瞎找，而是因为路径写对了；
        若有人把路径改回 `meta["execution_optimizations"]`，这里会红。
        """
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "T-short_chat-c"
            d.mkdir(parents=True)
            meta = _meta()
            meta["execution_optimizations"] = {"persistent_kv_slots": {"x": 1}}
            del meta["context"]                       # 真路径没了
            (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            row = summarize("c", [d])["rows"][0]
            self.assertIsNone(row["pack_ms_total"])
            self.assertFalse(row["has_pkv"])


class TestFallbackBucketClosure(unittest.TestCase):
    def test_reason_sum_matches_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "T-short_chat-a"
            _write(d, _meta(pkv={
                "requested": True, "effective": True,
                "n_fastpath_steps": 90, "n_fallback_steps": 10,
                "fallback_reasons": {"unequal_kv_len": 7, "new_sequence": 3},
                "bytes_allocated": 123,
            }))
            row = summarize("a", [d])["rows"][0]
            self.assertEqual(row["reasons_sum"], row["n_fallback"])


class TestPathConstant(unittest.TestCase):
    def test_path_is_the_one_observed_in_artifacts(self):
        """路径常量必须与真机产物一致（EXECOPT3 实测）。"""
        self.assertEqual(DRIVER_STATS,
                         ("context", "extra", "backend_stats", "driver_stats"))


class TestCoverageDenominator(unittest.TestCase):
    """★ 覆盖率的分母与"扣预热"必须写死——用全窗口当分母会把它虚高一倍。

    实测（EXECOPT4）：`n_fastpath_steps=193` 是**含预热**的总数；预热每轮会走
    `gen_len-1 = 116` 步纯 decode，那些步同样命中快路径 ⇒ 正式窗口只有
    `193-116 = 77`，分母是 telemetry 的 `forwards` 求和 `1143` ⇒ **6.74%**。
    若拿 `77/1260`（不扣预热分母）或 `193/1260`，都会读出错的数。
    """

    def test_derives_and_subtracts_warmup(self):
        cov = coverage_of({}, {"iterations": 1, "gen_len": 117}, 193,
                          tel={"forwards": 1143, "n_steps": 1080})
        self.assertEqual(cov["fast"], 77)
        self.assertEqual(cov["n_forward_groups"], 1143)
        self.assertEqual(cov["n_logical_steps"], 1080)
        self.assertAlmostEqual(cov["cov_of_groups"], 77 / 1143, places=6)
        self.assertIn("derived", cov["source"])
        self.assertIn("telemetry", cov["denominator_source"])

    def test_prefers_harness_measured_window(self):
        """有 harness 自报值时不得去派生（派生依赖"预热每步 1 组"的假设）。"""
        ds = {"measured_window": {"pkv_fastpath_measured": 80,
                                  "n_forwards_measured": 1143,
                                  "n_logical_steps_measured": 1080}}
        cov = coverage_of(ds, {"iterations": 2, "gen_len": 117}, 400,
                          tel={"forwards": 1143, "n_steps": 1080})
        self.assertEqual(cov["fast"], 80)
        self.assertEqual(cov["source"], "harness")

    def test_not_half_reported_when_warmup_shape_missing(self):
        cov = coverage_of({}, {}, 193, tel={"forwards": 1143, "n_steps": 1080})
        self.assertIsNone(cov["fast"], "缺预热形状时不许猜，只报「不可得」")


class TestTailSegment(unittest.TestCase):
    """★ 有效样本 = 末尾**连续** `decode==1` 的步（快路径唯一的覆盖区）。"""

    def test_finds_contiguous_single_decode_tail(self):
        rows = ([{"decode": 8}] * 5
                + [{"decode": 2}] * 1
                + [{"decode": 1}] * 4)
        seg = tail_single_decode(rows)
        self.assertEqual(len(seg), 4)
        self.assertTrue(all(r["decode"] == 1 for r in seg))

    def test_no_single_decode_returns_empty(self):
        self.assertEqual(tail_single_decode([{"decode": 8}, {"decode": 3}]), [])

    def test_all_single_decode_is_whole_series(self):
        rows = [{"decode": 1}] * 6
        self.assertEqual(len(tail_single_decode(rows)), 6)

    def test_stops_at_first_non_single_from_the_end(self):
        """中间出现过的 decode==1 不算——只取**末尾连续**段（否则会混入别处）。"""
        rows = [{"decode": 1}, {"decode": 8}, {"decode": 1}, {"decode": 1}]
        seg = tail_single_decode(rows)
        self.assertEqual(len(seg), 2)


def _summary(arm: str, *, whole: list[float], tail: list[float]) -> dict:
    """造一份 `summarize()` 形态的最小 summary：逐轮 whole/tail 两列。"""
    rows = []
    for i, (w, t) in enumerate(zip(whole, tail)):
        rows.append({
            "dir": f"{arm}" if i == 0 else f"{arm}-r{i}",
            "tel": {"pack_ms": w, "n_steps": 1080},
            "tail_pack": t,
            "tail_n": 77,
            "switches": {"a": True} if arm == "base" else {"a": True, "b": True},
        })
    return {"arm": arm, "n": len(rows), "rows": rows}


class TestDecomposeDelta(unittest.TestCase):
    """★ 用时分解（尾段 / 非尾段 / 全窗）+ 两个统计口径必须分开给。

    为什么这条判据值得单独测：**报错口径**是本项目反复踩的坑 ——
    "配对比值的中位数"与"中位数之比"是两个不同的量，混用会让别人复算对不上。
    """

    def _pair(self):
        """**逐轮比值不相等**的数据 —— 这样"配对比值中位"与"中位数之比"才会分开。

        ⚠️ 构造要点：`median(x_i/y_i)` 与 `median(x)/median(y)` **经常恰好相等**
        （前几版随手编的数据次次撞上），必须刻意造一个**离群比值**才分得开：
        这里 r0 的比值是 100，其余四轮都是 9.5 ⇒
        配对比值中位 = **9.5**，而中位数之比 = 380/30 = **12.67**。
        """
        base = _summary("base", whole=[10000.0, 10100.0, 9900.0, 10050.0, 10020.0],
                        tail=[10.0, 20.0, 30.0, 40.0, 100.0])
        opt = _summary("opt", whole=[10040.0, 10140.0, 9940.0, 10090.0, 10060.0],
                       tail=[1000.0, 190.0, 285.0, 380.0, 950.0])
        return [base, opt]

    def _additive_pair(self):
        """**每轮常量** ⇒ 中位数=常量 ⇒ 可加性是**恒等式**（残差 0）。"""
        base = _summary("base", whole=[10000.0] * 5, tail=[100.0] * 5)
        opt = _summary("opt", whole=[10040.0] * 5, tail=[170.0] * 5)
        return [base, opt]

    def test_tail_ratio_is_large_and_whole_is_small(self):
        d = decompose_delta(self._pair(), "base", "opt")
        self.assertGreater(d["tail_ratio_median"], 1.5, "尾段效应应显著")
        self.assertLess(d["whole_ratio_median"], 1.02, "全窗应被时间权重摊薄")

    def test_three_segments_are_all_reported(self):
        d = decompose_delta(self._pair(), "base", "opt")
        for seg in ("whole", "tail", "non_tail"):
            self.assertIn(f"{seg}_delta_ms", d)
            self.assertIn(f"{seg}_ratio_median", d)
            self.assertIn(f"{seg}_ratio_of_medians", d)

    def test_non_tail_is_median_of_per_round_difference(self):
        """`non_tail` = 逐轮 `(whole_i - tail_i)` 的**中位数**。

        ⚠️ 它**不等于** `median(whole) - median(tail)` —— 中位数对差**不可加**。
        第一版测试就是按"可加"写的，当场被这条打红。
        """
        d = decompose_delta(self._pair(), "base", "opt")
        expected_base = statistics.median([10000.0 - 10.0, 10100.0 - 20.0,
                                           9900.0 - 30.0, 10050.0 - 40.0,
                                           10020.0 - 100.0])
        self.assertAlmostEqual(d["non_tail_base_ms"], expected_base, places=6)

    def test_additivity_holds_exactly_on_additive_data(self):
        """常量数据上残差应为 0（说明分解本身写对了，不是靠巧合）。"""
        d = decompose_delta(self._additive_pair(), "base", "opt")
        self.assertAlmostEqual(d["additivity_residual_ms"], 0.0, places=9)
        self.assertAlmostEqual(d["tail_delta_ms"], 70.0, places=9)
        self.assertAlmostEqual(d["non_tail_delta_ms"], -30.0, places=9)
        self.assertAlmostEqual(d["whole_delta_ms"], 40.0, places=9)

    def test_additivity_residual_is_reported_when_medians_do_not_add(self):
        """★ 非可加数据上残差**非零** ⇒ 它是个**诊断量**，不是恒等式的一部分。

        真机残差（EXECOPT5 = +7.11 ms / 0.034%）就该这么读：中位数不可加造成的，
        不是"分解错了"。
        """
        d = decompose_delta(self._pair(), "base", "opt")
        self.assertIn("additivity_residual_ms", d)
        self.assertNotAlmostEqual(d["additivity_residual_ms"], 0.0, places=6)
        self.assertAlmostEqual(
            d["additivity_residual_ms"],
            d["whole_delta_ms"] - d["tail_delta_ms"] - d["non_tail_delta_ms"],
            places=9)

    def test_two_statistics_differ_and_are_both_given(self):
        """★ 两个口径**不相等** ⇒ 必须都给出、都不许替对方作证。"""
        d = decompose_delta(self._pair(), "base", "opt")
        self.assertNotEqual(d["tail_ratio_median"], d["tail_ratio_of_medians"],
                            "逐轮比值不等 ⇒ 两个口径就该有差")
        self.assertAlmostEqual(d["tail_ratio_of_medians"],
                               d["tail_opt_ms"] / d["tail_base_ms"], places=9)

    def test_sign_consistency_is_reported(self):
        d = decompose_delta(self._pair(), "base", "opt")
        self.assertEqual(d["tail_n_same_sign"], 5)
        self.assertEqual(d["tail_n_paired"], 5)

    def test_unpaired_rounds_are_skipped_not_zero_filled(self):
        """缺一轮 ⇒ 该轮**不参与**（不是用 0 顶替）。用 0 会造出一个假的大效应。"""
        base = _summary("base", whole=[10000.0, 10000.0], tail=[100.0, 100.0])
        opt = _summary("opt", whole=[10040.0, 10040.0], tail=[170.0, 170.0])
        opt["rows"][1]["tail_pack"] = None
        d = decompose_delta([base, opt], "base", "opt")
        self.assertEqual(d["tail_n_paired"], 1, "只有 1 轮可配对")
        self.assertAlmostEqual(d["tail_ratio_median"], 1.70, places=2)

    def test_uncountable_pairs_returns_empty(self):
        base = _summary("base", whole=[1.0], tail=[1.0])
        opt = _summary("opt", whole=[2.0], tail=[2.0])
        for r in base["rows"] + opt["rows"]:
            r["tel"]["pack_ms"] = None
        self.assertEqual(decompose_delta([base, opt], "base", "opt"), {})


if __name__ == "__main__":
    unittest.main()
