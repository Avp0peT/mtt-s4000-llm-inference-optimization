"""``scripts/calibrate_slo.py`` 的口径守卫（P3）。

这个脚本的作用是"把 SLO 判据从够不着标定到能分辨"，所以它自己的三件事必须先对：

1. **逐请求、且是「与」** —— 与 ``SLO.satisfied_by()`` 同口径。写成"比汇总分位"或
   写成"或"关系，标定出的阈值就是错的，而且不会有任何报错。
2. **分母是全部提交**（含失败/未完成）—— 用成功数当分母会让大批失败的 run
   显示出很高的达标率。
3. **读不到东西必须显形并返回非零** —— 一格读不到请求，在表里与"该格全达标"
   长得一样（都缺行）。这是本项目反复出现的"空输入伪装成合格结果"。

与其它守卫测试一样，每一条都配一个**反向对照**：注入缺陷 → 必须报警。
"""

from __future__ import annotations

import importlib.util
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
    modname = f"_calib_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod  # dataclasses 要按 __module__ 回查
    spec.loader.exec_module(mod)
    return mod


def _req(*, ttft: float | None, tpot: float | None, success: bool = True) -> dict:
    return {"ttft_ms": ttft, "tpot_ms": tpot, "success": success}


class TestSatisfactionRateMatchesHarnessSemantics(unittest.TestCase):
    """重算的达标率必须与 ``ascend_opt.metrics`` 的判定一致。"""

    def setUp(self):
        self.mod = _load_script("calibrate_slo")

    def test_denominator_is_all_submitted(self):
        """分母含失败/未完成，不可被稀释。"""
        recs = [
            _req(ttft=100, tpot=10),
            _req(ttft=100, tpot=10, success=False),   # 失败也算分母
            _req(ttft=None, tpot=None, success=False),
        ]
        r = self.mod.satisfaction_rate(recs, ttft_ms=2000, tpot_ms=100)
        self.assertEqual(r["n_total"], 3, "分母必须是全部提交数")
        self.assertEqual(r["n_sat"], 1)
        self.assertAlmostEqual(r["rate"], 1 / 3)

    def test_is_and_not_or(self):
        """同时配了 ttft 与 tpot 时，一个不满足就不达标。"""
        recs = [
            _req(ttft=100, tpot=10),      # 两个都过
            _req(ttft=99999, tpot=10),    # 只 TTFT 不过
            _req(ttft=100, tpot=99999),   # 只 TPOT 不过
        ]
        r = self.mod.satisfaction_rate(recs, ttft_ms=2000, tpot_ms=100)
        self.assertEqual(r["n_sat"], 1, "「或」关系会算出 3 —— 口径错了")

    def test_unmeasured_latency_is_not_satisfied(self):
        """不可测 ≠ 达标：所以一个连首 token 都没产出的请求不能被算作达标。"""
        r = self.mod.satisfaction_rate(
            [_req(ttft=None, tpot=None)], ttft_ms=2000, tpot_ms=100
        )
        self.assertEqual(r["n_sat"], 0)

    def test_attribution_splits_by_binding_criterion(self):
        """归因要把"哪个判据卡住"分开——合并的数会把依据认错。"""
        recs = [
            _req(ttft=100, tpot=10),         # 达标
            _req(ttft=99999, tpot=10),       # 仅 TTFT 超
            _req(ttft=100, tpot=99999),      # 仅 TPOT 超
            _req(ttft=99999, tpot=99999),    # 两者都超
            _req(ttft=1, tpot=1, success=False),  # 未成功（与阈值无关）
        ]
        a = self.mod.attribution(recs, ttft_ms=2000, tpot_ms=100)
        self.assertEqual(a["sat"], 1)
        self.assertEqual(a["fail_ttft_only"], 1)
        self.assertEqual(a["fail_tpot_only"], 1)
        self.assertEqual(a["fail_both"], 1)
        self.assertEqual(a["fail_outcome"], 1)

    def test_attribution_totals_equal_bucket_count(self):
        """五桶之和必须等于请求总数——漏一桶会让归因表静默少数。"""
        recs = [
            _req(ttft=100, tpot=10),
            _req(ttft=99999, tpot=10),
            _req(ttft=100, tpot=99999),
            _req(ttft=99999, tpot=99999),
            _req(ttft=1, tpot=1, success=False),
        ]
        a = self.mod.attribution(recs, ttft_ms=2000, tpot_ms=100)
        self.assertEqual(sum(a.values()), len(recs))


class TestCellLabelsAreReadable(unittest.TestCase):
    """格名要能一眼看懂；早期版本按 ``c[-14:]`` 截断，列头全是 ``or-adaptive_c1``。"""

    def setUp(self):
        self.mod = _load_script("calibrate_slo")

    def test_labels_are_short_and_unique(self):
        cells = [
            "phase1_anchor-static_c1",
            "phase1_anchor-static_chunk_c4",
            "phase1_anchor-adaptive_c8",
        ]
        labels = [self.mod._label(c) for c in cells]
        self.assertEqual(labels, ["stat_c1", "schu_c4", "adap_c8"])
        self.assertEqual(len(set(labels)), len(labels), "标签必须互不相同")

    def test_sort_key_puts_concurrency_first(self):
        """按并发优先排序，表里才看得出"达标率随并发下滑"的梯度。"""
        cells = [
            "phase1_anchor-adaptive_c4",
            "phase1_anchor-static_c1",
            "phase1_anchor-static_chunk_c2",
        ]
        order = [self.mod._label(c) for c in sorted(cells, key=self.mod._split_cell)]
        self.assertEqual(order, ["stat_c1", "schu_c2", "adap_c4"])


class TestEmptyInputFailsLoudly(unittest.TestCase):
    """读不到东西必须返回非零。空输入伪装成合格结果，这是本项目的老毛病。"""

    def setUp(self):
        self.mod = _load_script("calibrate_slo")

    def test_missing_raw_dir_is_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nope"
            self.assertNotEqual(
                self.mod.main(["--raw", str(missing), "--prefix", "X"]), 0
            )

    def test_unknown_prefix_is_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "OTHER-static_c1").mkdir()
            (root / "OTHER-static_c1" / "requests.jsonl").write_text(
                json.dumps(_req(ttft=1, tpot=1)) + "\n", encoding="utf-8"
            )
            self.assertNotEqual(
                self.mod.main(["--raw", str(root), "--prefix", "NOSUCH"]), 0
            )

    def test_one_empty_cell_makes_the_whole_run_nonzero(self):
        """**块级失败必须显形**：一格读不到请求，表里缺一行，看起来像"该格全达标"。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            good = root / "B-static_c1"
            good.mkdir()
            (good / "requests.jsonl").write_text(
                json.dumps(_req(ttft=100, tpot=10)) + "\n", encoding="utf-8"
            )
            empty = root / "B-static_c2"
            empty.mkdir()
            (empty / "requests.jsonl").write_text("", encoding="utf-8")
            rc = self.mod.main([
                "--raw", str(root), "--prefix", "B", "--candidates", "100",
            ])
            self.assertNotEqual(
                rc, 0,
                "有格读不到请求时退出码仍为 0 —— 残缺的表会被当成完整的",
            )

    def test_all_cells_present_is_success(self):
        """正向对照：数据齐了就该返回 0，否则守卫是个恒真报警器。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in (1, 2):
                d = root / f"B-static_c{i}"
                d.mkdir()
                (d / "requests.jsonl").write_text(
                    json.dumps(_req(ttft=100, tpot=10)) + "\n", encoding="utf-8"
                )
            rc = self.mod.main([
                "--raw", str(root), "--prefix", "B", "--candidates", "100",
            ])
            self.assertEqual(rc, 0)

    def test_bad_json_line_raises_instead_of_being_skipped(self):
        """坏行必须报错。静默跳过会让"少读了几个请求"看起来像"那几个都达标"。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "B-static_c1"
            d.mkdir()
            (d / "requests.jsonl").write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                self.mod.load_requests(d)

    def test_repeats_are_excluded(self):
        """只取首轮：三次重复的逐请求值几乎逐字相同，多取只是重复打印同一张表。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("B-static_c1", "B-static_c1-r1", "B-static_c1-r2"):
                d = root / name
                d.mkdir()
                (d / "requests.jsonl").write_text(
                    json.dumps(_req(ttft=1, tpot=1)) + "\n", encoding="utf-8"
                )
            got = [d.name for _, d in self.mod._iter_run_dirs(root, ["B"])]
            self.assertEqual(got, ["B-static_c1"])


if __name__ == "__main__":
    unittest.main()
