"""探针的**选点**逻辑（`--only-group`）：不能靠 `--limit` 代替。

实测踩到（2026-09-26）
--------------------
我要"只采 v7 新增的小 chunk 组（16 点）"，第一版做法是**把新组写在网格文件的第一位**，
然后用 `--limit 16`。它**不成立**：`costcalib_grid.expand()` 返回的是
`sorted(out.values(), key=lambda s: s.as_key())` —— **按 key 排序**，不是文件顺序
⇒ `--limit 16` 取到的永远是排序最前的点（`pure_decode`），**当不了分组过滤**。
（校验时才发现：前 16 点全是 `pure_decode`，而我以为采的是新组。）

⇒ 正确的做法是真正的按组过滤，并且**子集必须显形**：`complete=True` 的含义是
"**被选中的点**都测完了"，不是"整个网格都测完了"。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fwd_costcalib_probe as P  # noqa: E402
from costcalib_grid import Shape, expand, load_grid  # noqa: E402

GRID_V7 = _ROOT / "configs" / "cost_calib" / "shape_grid_v7.yaml"


class TestLimitCannotBeUsedAsGroupFilter(unittest.TestCase):
    def test_limit_does_not_isolate_a_group(self):
        """★ 钉住这个坑：`--limit` 取的是**按 key 排序**的前 N，与网格文件顺序无关。

        断言的是**性质**（而不是某一个具体的组集合）：排序后前 N 点会**跨组**，
        所以它**不能**被当成"只采某一组"的手段。
        （我第一版断言"前 16 点全是 pure_decode"——错了：`as_key` 里 `prefill_tok`
        排在 `kv_len` 前面，于是 `pure_prefill(pf=512,kv=0)` 与 `mixed_small_chunk(pf=128)`
        会交错进前 16 点。教训：断言要钉**性质**，不要钉我当时看到的那个快照。）
        """
        shapes = list(expand(load_grid(GRID_V7)))
        head = P.select_shapes(shapes, None, 16)
        self.assertEqual(len(head), 16)
        groups = {s.group for s in head}
        self.assertGreater(len(groups), 1,
                           f"前 16 点竟只来自一个组 {groups} ⇒ 那么 --limit 也许能用，"
                           "本用例的前提要重审")
        self.assertNotEqual(groups, {"mixed_small_chunk"},
                            "`--limit 16` 恰好隔离出了新组 ⇒ 要么网格变了、要么排序变了，重审")
        self.assertNotEqual(
            {(s.batch, s.prefill_tok, s.kv_len) for s in head},
            {(s.batch, s.prefill_tok, s.kv_len)
             for s in P.select_shapes(shapes, ["mixed_small_chunk"], None)},
            "`--limit 16` 的选点与 `--only-group mixed_small_chunk` 相同 ⇒ 本用例失去意义")


class TestOnlyGroup(unittest.TestCase):
    def test_selects_exactly_the_new_group(self):
        shapes = list(expand(load_grid(GRID_V7)))
        got = P.select_shapes(shapes, ["mixed_small_chunk"], None)
        self.assertEqual(len(got), 16)
        self.assertTrue(all(s.group == "mixed_small_chunk" for s in got))
        self.assertTrue(all(s.n_new_mode == "real_mixed" for s in got))
        self.assertTrue(all(s.prefill_tok in (128, 256) for s in got))

    def test_multiple_groups_union(self):
        shapes = list(expand(load_grid(GRID_V7)))
        got = P.select_shapes(shapes, ["mixed_small_chunk", "pure_prefill"], None)
        self.assertEqual({s.group for s in got}, {"mixed_small_chunk", "pure_prefill"})

    def test_order_is_group_filter_then_limit(self):
        shapes = [Shape(group=g, batch=1, prefill_tok=0, kv_len=512, n_forwards=1)
                  for g in ("a", "b", "a", "b")]
        got = P.select_shapes(shapes, ["a"], 1)
        self.assertEqual([s.group for s in got], ["a"], "应先按组过滤再取前 N")

    def test_unknown_group_raises_with_candidates(self):
        shapes = list(expand(load_grid(GRID_V7)))
        with self.assertRaises(ValueError) as cm:
            P.select_shapes(shapes, ["no_such_group"], None)
        msg = str(cm.exception)
        self.assertIn("no_such_group", msg)
        self.assertIn("mixed_small_chunk", msg, "报错要列出可选组名，否则只能猜")

    def test_empty_selection_is_possible_and_caller_handles_it(self):
        """选空不该抛异常（由调用方报 rc=2）—— 抛异常会让"过滤器写错"看起来像崩溃。"""
        self.assertEqual(P.select_shapes([], None, None), [])

    def test_no_filter_returns_everything(self):
        shapes = list(expand(load_grid(GRID_V7)))
        self.assertEqual(len(P.select_shapes(shapes, None, None)), len(shapes))


class TestSubsetIsRecordedInArtifact(unittest.TestCase):
    def test_probe_writes_subset_block(self):
        """产物必须能自证"这轮采的是不是整个网格"。"""
        src = (_ROOT / "scripts" / "fwd_costcalib_probe.py").read_text(encoding="utf-8")
        self.assertIn('art["subset"]', src)
        for key in ("only_group", "limit", "n_points_selected", "n_points_in_grid",
                    "is_full_grid"):
            self.assertIn(key, src, f"subset 缺 {key}")


if __name__ == "__main__":
    unittest.main()
