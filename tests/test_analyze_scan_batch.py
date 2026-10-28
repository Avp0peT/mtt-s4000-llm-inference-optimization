"""`analyze_scan_batch.py` 的三情形判定必须**条件为真才打印**（不是静态文案）。

为什么单独钉这个
----------------
2026-09-27 读 §7d 分析时发现：脚本的 `else` 分支**只印静态文案**，
不检查格级条件。于是**看不到"情形二其实为真"**：

* §7d 第 5 节的情形二是**格级**的（`ttft.p50 ≤ 2000` 且 `tpot.p50 ≤ 1.5×SLO`）；
* 实测 `c4_m4` / `c4_m8`：TTFT p50 = 1487/1488 ms（≤2000 ✅）、
  TPOT p50 = 118.0/117.9 ms（≤150 ✅）⇒ **条件为真**；
* 但它们的达标率是 **1/64** ⇒ 若把"达标率"也并进情形二，两格会落进情形三。
  §7d 的分档**只看时延**，所以它们**属情形二**。

⇒ 这两格当时**被静默丢掉**了。风险不在于少打一行字，而在于：
将来若有人在别处看到"情形二为真 ⇒ 应换更轻负载"，
会与"本批选 `c2_m4`"的结论**读起来矛盾** —— 而矛盾的原因只是**文案没说清不并存**。

⚠️ 本文件钉的是**判定机制**（条件真值 + 不覆盖情形一），
不是"这 8 格的数字"（那是 `EXPERIMENTS.md` §S14S 的事，判据不改）。
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analyze_scan_batch as A  # noqa: E402


def make_run(root: Path, stem: str, *, repeat: int,
             n_req: int = 64, n_slo: int = 64,
             ttft: float | None = 1155.0, tpot: float | None = 86.7,
             tpot_cv: float | None = 0.02) -> None:
    """造一个**结构真实**的格目录（键名逐字照抄真产物）。

    ⚠️ 分位块是**嵌套**的：`summary.ttft.p50`。写成 `summary.ttft_ms_p50`
    会让脚本**静默取到 None**，于是"全部不可达"与"读错键"无法区分
    —— 那正是本脚本 docstring 警告过的坑，夹具必须同形。
    """
    suffix = "" if repeat == 0 else f"-r{repeat}"
    d = root / f"S14S-short_chat-{stem}{suffix}"
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_label": "MEASURED",
        "summary": {
            "n_requests": n_req,
            "n_slo_satisfied": n_slo,
            "ttft": {"p50": ttft},
            "tpot": {"p50": tpot},
            "queue": {"p50": 127.2},
            "prefill": {"p50": 24.1},
            "stability": {"ttft_robust_cv": 0.3, "tpot_robust_cv": tpot_cv},
            "wall_ms": 98000.0,
        },
    }
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8", newline="\n")


def run_analysis(root: Path) -> tuple[str, int]:
    """跑 `main()`，把 stdout 抓回来（**不改判据**，只读输出）。"""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = A.main([str(root), "--tag", "S14S"])
    return buf.getvalue(), rc


class TestCellCm(unittest.TestCase):
    """`_cell_cm` 的契约收紧：要么正确抽出，要么**抛错**，不许静默取错。"""

    def test_extracts_known_cells(self):
        for stem, want in (("fixed_cost_scan_c2_m4", (2, 4)),
                           ("fixed_cost_scan_c2_m8", (2, 8)),
                           ("fixed_cost_scan_c8_m8", (8, 8))):
            with self.subTest(stem=stem):
                self.assertEqual(A._cell_cm(stem), want)

    def test_rejects_non_cell_shapes(self):
        """不合 `..._c<c>_m<m>$` 就必须 `ValueError`，且信息里**带格名**。

        ⚠️ 这里**不主张**"旧写法会取错" —— 实测**没能构造出分歧样本**
        （见 `_cell_cm` docstring 的诚实标注）。本用例钉的是
        "不合形状时**拒绝**且报错可读"，不是"修了一个已发生的错"。
        """
        for stem in ("fixed_cost_scan", "service_benefit_c2_m4_cost",
                     "fixed_cost_scan_m4", "scan_c2_m4_mp8"):
            with self.subTest(stem=stem):
                with self.assertRaises(ValueError) as ctx:
                    A._cell_cm(stem)
                self.assertIn(stem, str(ctx.exception),
                              "报错里必须出现格名，否则不知道是哪个格子不对")


class TestThreeCasesAreConditionDriven(unittest.TestCase):
    """三情形必须**逐条打印条件真值**，不是只留一段静态的 `else` 文案。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_case_one_selected_when_reachable(self):
        for i in range(3):
            make_run(self.root, "fixed_cost_scan_c2_m4", repeat=i)
        out, rc = run_analysis(self.root)
        self.assertEqual(rc, 0)
        self.assertIn("【情形一】", out)
        self.assertIn("选用 fixed_cost_scan_c2_m4", out)

    def test_case_two_is_evaluated_even_when_case_one_is_nonempty(self):
        """★ 本文件存在的理由：情形一非空时，情形二**仍须被求值并显式报出**。

        场景复刻真数据：`c2_m4` 可达（情形一），`c4_m4` 时延达标但达标率 1/64
        ⇒ 情形二**为真**。旧脚本会**只印静态 else 文案**，读者看不到这一条。
        """
        for i in range(3):
            make_run(self.root, "fixed_cost_scan_c2_m4", repeat=i,
                     n_slo=64, ttft=1155.0, tpot=86.7)
            make_run(self.root, "fixed_cost_scan_c4_m4", repeat=i,
                     n_slo=1, n_req=64, ttft=1487.5, tpot=118.0)
        out, rc = run_analysis(self.root)
        self.assertEqual(rc, 0)
        # 情形一照旧命中，且**不被**情形二覆盖
        self.assertIn("选用 fixed_cost_scan_c2_m4", out)
        # 情形二**为真**且被报出，含那个"率不足但时延达标"的格
        self.assertIn("【情形二】", out)
        self.assertIn("fixed_cost_scan_c4_m4", out.split("【情形二】")[1].split("【情形三】")[0])
        # 且文案必须**显式说明**它与情形一不并存，否则读起来会像矛盾结论
        self.assertIn("触发情形二 ≠ 情形一为空", out)

    def test_case_two_false_when_no_cell_meets_latency(self):
        """全部格时延都超 ⇒ 情形二为**假**，且情形三不适用（因为情形一非空）。

        ⚠️ 夹具设计注意：**不能**把一个"时延达标"的格留在里面当陪衬 ——
        情形二只看时延，任何时延达标的格都会让它为真。所以陪衬格必须是
        **时延超标但达标率 1.0**（即"情形一可达、情形二不成立"的那种）。
        """
        for i in range(3):
            # 时延达标 ⇒ 情形一可达（陪衬）
            make_run(self.root, "fixed_cost_scan_c2_m4", repeat=i,
                     n_slo=64, ttft=1155.0, tpot=86.7)
            # 时延超标 ⇒ 情形一不可达，**且**不满足情形二
            make_run(self.root, "fixed_cost_scan_c8_m8", repeat=i,
                     n_slo=0, ttft=2342.0, tpot=218.6)
        out, rc = run_analysis(self.root)
        self.assertEqual(rc, 0)
        self.assertIn("选用 fixed_cost_scan_c2_m4", out)
        sec2 = out.split("【情形二】")[1].split("【情形三】")[0]
        self.assertNotIn("fixed_cost_scan_c8_m8", sec2,
                         "TPOT 218.6 > 150 ⇒ 该格不属情形二")

    def test_case_two_false_when_every_cell_exceeds_latency(self):
        """**没有任何**时延达标的格 ⇒ 情形二为空集 ∅，情形三不适用（情形一也空）。

        这是 §7d 的否定形态：题目要求"全部 8 格都不可达 ⇒ 先重新归因"。
        此时必须报 ∅，**不得**因空集而崩或静默跳过。
        """
        for i in range(3):
            make_run(self.root, "fixed_cost_scan_c8_m4", repeat=i,
                     n_slo=0, ttft=6682.0, tpot=115.7)
            make_run(self.root, "fixed_cost_scan_c8_m8", repeat=i,
                     n_slo=0, ttft=2342.0, tpot=218.6)
        out, rc = run_analysis(self.root)
        self.assertEqual(rc, 0)
        self.assertIn("∅", out.split("【情形一】")[1].split("【情形二】")[0])
        sec2 = out.split("【情形二】")[1].split("【情形三】")[0]
        self.assertIn("∅", sec2, "没有任何时延达标的格 ⇒ 情形二应为空集")

    def test_case_three_fires_only_when_all_are_bad(self):
        """情形一为空 **且** 情形二为空 ⇒ 情形三为真，且必须说"未判/不得改 SLO"。"""
        for i in range(3):
            make_run(self.root, "fixed_cost_scan_c8_m8", repeat=i,
                     n_slo=0, ttft=2342.0, tpot=218.6)
        out, rc = run_analysis(self.root)
        self.assertEqual(rc, 0)
        self.assertIn("【情形三】", out)
        sec3 = out.split("【情形三】")[1]
        self.assertIn("True", sec3)
        self.assertIn("不得改 SLO", sec3)

    def test_case_two_ignores_slo_rate(self):
        """★ 情形二**只看时延**：达标率 0/64 但时延达标的格也算情形二。

        这条是 §7d 的**分档语义**：情形二判的是"再降一档负载值不值得试"，
        不是"这一格好不好"。把达标率并进来会让所有格都掉进情形三。
        """
        for i in range(3):
            make_run(self.root, "fixed_cost_scan_c4_m4", repeat=i,
                     n_slo=0, n_req=64, ttft=1999.0, tpot=149.0)
        out, rc = run_analysis(self.root)
        self.assertEqual(rc, 0)
        sec2 = out.split("【情形二】")[1].split("【情形三】")[0]
        self.assertIn("fixed_cost_scan_c4_m4", sec2,
                      "达标率 0/64 但时延达标 ⇒ 仍属情形二（分档只看时延）")

    def test_cv_over_limit_is_not_reachable(self):
        """CV 超限 ⇒ 不进可达集（第 4 节第 3 条），但**不影响**情形二的时延判定。"""
        for i in range(3):
            make_run(self.root, "fixed_cost_scan_c2_m4", repeat=i,
                     n_slo=64, ttft=1155.0, tpot=86.7, tpot_cv=0.55)
        out, rc = run_analysis(self.root)
        self.assertEqual(rc, 0)
        self.assertIn("∅", out.split("【情形一】")[1].split("【情形二】")[0])
        # 时延达标 ⇒ 情形二仍为真
        self.assertIn("fixed_cost_scan_c2_m4",
                      out.split("【情形二】")[1].split("【情形三】")[0])

    def test_single_repeat_does_not_enter_reachable_set(self):
        """<3 轮不得进可达集（项目纪律：重复 <3 不出结论）。"""
        make_run(self.root, "fixed_cost_scan_c2_m4", repeat=0)
        out, rc = run_analysis(self.root)
        self.assertEqual(rc, 0)
        self.assertIn("<3 不出结论", out)
        self.assertIn("∅", out.split("【情形一】")[1].split("【情形二】")[0])

    def test_missing_artifact_dir_returns_nonzero(self):
        """没有任何产物 ⇒ rc≠0（不能静默当"全部不可达"）。"""
        out, rc = run_analysis(self.root)
        self.assertNotEqual(rc, 0)


class TestKeyNamesMatchRealArtifacts(unittest.TestCase):
    """★ 键名必须与**真产物**同形 —— 写错键会静默取 `None`。"""

    def test_nested_percentile_blocks_are_used(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_run(root, "fixed_cost_scan_c2_m4", repeat=0)
            runs = A.load_runs("S14S", root)
            self.assertEqual(list(runs), ["fixed_cost_scan_c2_m4"])
            r = runs["fixed_cost_scan_c2_m4"][0]
            # 若脚本改用 `ttft_ms_p50` 这种扁平键，这里会变成 None
            self.assertEqual(r["ttft_p50"], 1155.0)
            self.assertEqual(r["tpot_p50"], 86.7)
            self.assertEqual(r["queue_p50"], 127.2)

    def test_repeat_suffix_groups_into_one_cell(self):
        """`-r1` / `-r2` 必须归到**同一格**，否则 3 轮凑不齐。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(3):
                make_run(root, "fixed_cost_scan_c2_m4", repeat=i)
            runs = A.load_runs("S14S", root)
            self.assertEqual(list(runs), ["fixed_cost_scan_c2_m4"])
            self.assertEqual(len(runs["fixed_cost_scan_c2_m4"]), 3)


if __name__ == "__main__":
    unittest.main()
