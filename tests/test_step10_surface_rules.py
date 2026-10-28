"""第⑩步**预注册规则**的用例。

为什么给这三条函数写用例：它们决定论文的基线（`Best Static`）。
一度写成 `main` 里的闭包（没法测），而"决定基线的规则没有用例"是本仓库反复警惕的风险
（见 `experiment-harness-integrity`：**守卫要能在缺陷上失败**）。

规则原文在 `NEXT_SESSION_TASKS.md` 第 10 步 §二；本文件是它的**可失败**形式。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_s10_surface import (  # noqa: E402
    dominates,
    noise_floor,
    pareto_front,
    rank_by_tie_rule,
    sole_dominator,
)

SEGS = ("low", "high", "recover")


def _median_of(table: dict[str, dict[str, float | None]]):
    return lambda cell, seg: table[cell][seg]


class TestDominance(unittest.TestCase):
    def test_same_config_never_dominates_itself(self):
        m = _median_of({"A": dict.fromkeys(SEGS, 1.0)})
        self.assertEqual(dominates("A", "A", m, dict.fromkeys(SEGS, 0.0), SEGS)[0], False)

    def test_dominates_when_all_segments_geq_and_one_strictly_exceeds_floor(self):
        m = _median_of({
            "A": {"low": 5.0, "high": 5.0, "recover": 5.0},
            "B": {"low": 5.0, "high": 3.0, "recover": 5.0},
        })
        ok, why = dominates("A", "B", m, dict.fromkeys(SEGS, 0.1), SEGS)
        self.assertTrue(ok)
        self.assertIn("high", why)

    def test_does_not_dominate_when_a_segment_is_worse(self):
        """只要有一段更差 ⇒ 不支配（这是"支配"与"平均更好"的分界）。

        ⚠️ 首版这条用例的数据写错了（B 在三段**都**更差 ⇒ 函数判支配是对的）。
        正确形状：A 在两段明显更好、**但在 `high` 段更差** ⇒ 不支配。
        """
        m = _median_of({
            "A": {"low": 9.0, "high": 2.0, "recover": 9.0},
            "B": {"low": 1.0, "high": 5.0, "recover": 1.0},
        })
        self.assertFalse(dominates("A", "B", m, dict.fromkeys(SEGS, 0.1), SEGS)[0])
        # 反向同样不支配（互为非支配 ⇒ 两个都进 Pareto 前沿）
        self.assertFalse(dominates("B", "A", m, dict.fromkeys(SEGS, 0.1), SEGS)[0])

    def test_does_not_dominate_when_all_gaps_are_within_the_noise_floor(self):
        """★ 本条是噪声底存在的理由：全面"略好"但都不过噪声底 ⇒ **不算支配**。

        少了这条，任何一点点系统性偏差都会被当成"确实更好"，Best Static 就变成
        在噪声上挑赢家。
        """
        m = _median_of({
            "A": {"low": 5.05, "high": 5.05, "recover": 5.05},
            "B": {"low": 5.00, "high": 5.00, "recover": 5.00},
        })
        floor = dict.fromkeys(SEGS, 0.10)      # 极差上限 0.10 > 0.05 的差
        ok, why = dominates("A", "B", m, floor, SEGS)
        self.assertFalse(ok, f"差 0.05 未超过噪声底 0.10，不该判支配（why={why}）")

    def test_missing_data_is_not_a_win(self):
        """数据缺失时**返回不支配**——宁可判不出，不可判错。"""
        m = _median_of({
            "A": {"low": 9.0, "high": 9.0, "recover": 9.0},
            "B": {"low": 1.0, "high": None, "recover": 1.0},
        })
        self.assertFalse(dominates("A", "B", m, dict.fromkeys(SEGS, 0.1), SEGS)[0])


class TestNoiseFloor(unittest.TestCase):
    def test_takes_the_upper_bound_not_the_median(self):
        """取**上限**：只要有一个候选抖得厉害，那段就该被承认不可分辨。"""
        spreads = {"A": {"low": 0.01, "high": 0.02, "recover": 0.01},
                   "B": {"low": 0.50, "high": 0.02, "recover": 0.01}}
        floor = noise_floor(spreads, SEGS)
        self.assertAlmostEqual(floor["low"], 0.50)
        self.assertAlmostEqual(floor["high"], 0.02)


class TestParetoAndTieRule(unittest.TestCase):
    def test_pareto_front_excludes_dominated(self):
        dom = {"A": [], "B": ["C"], "C": [], "D": ["C"]}
        self.assertEqual(pareto_front(["A", "B", "C", "D"], dom), ["A", "B", "D"])

    def test_tie_rule_priority_high_then_chunk_off_then_smaller_budget(self):
        high = {"A": 5.0, "B": 5.0, "C": 9.0}
        chunk = {"A": True, "B": False, "C": True}
        budget = {"A": 8192, "B": 8192, "C": 8192}
        # C 的 high 最高 ⇒ 第一优先，压过 B 的 "chunk 关"
        self.assertEqual(rank_by_tie_rule(["A", "B", "C"], high, chunk, budget)[0], "C")

        high2 = {"A": 5.0, "B": 5.0}
        # high 相同 ⇒ 看 chunk：B 关着 ⇒ B 优先
        self.assertEqual(rank_by_tie_rule(["A", "B"], high2, chunk, budget)[0], "B")

        high3 = {"A": 5.0, "B": 5.0}
        chunk3 = {"A": True, "B": True}
        budget3 = {"A": 8192, "B": 2048}
        # high 与 chunk 都相同 ⇒ 看 budget：小的优先
        self.assertEqual(rank_by_tie_rule(["A", "B"], high3, chunk3, budget3)[0], "B")

    def test_tie_rule_treats_missing_high_as_worst_not_zero(self):
        """`high` 取不到 ⇒ 按最差处理（**不是 0**）：0 是一个合法读数。"""
        high = {"A": None, "B": 0.0}
        chunk = {"A": False, "B": True}       # A 在第二优先上占优
        budget = {"A": 8192, "B": 8192}
        # 若把 None 当 0，A 会因"chunk 关"胜出；按最差处理则 B 胜
        self.assertEqual(rank_by_tie_rule(["A", "B"], high, chunk, budget)[0], "B")

    def test_sole_dominator_triggers_s5(self):
        dom = {"A": ["B", "C", "D"], "B": [], "C": [], "D": []}
        self.assertEqual(sole_dominator(["A", "B", "C", "D"], dom), ["A"])
        dom2 = {"A": ["B"], "B": [], "C": []}
        self.assertEqual(sole_dominator(["A", "B", "C"], dom2), [])


if __name__ == "__main__":
    unittest.main()
