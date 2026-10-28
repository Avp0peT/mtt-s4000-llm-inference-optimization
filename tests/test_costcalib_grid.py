"""成本标定形状网格的校验/展开用例（第 11 步，协议 `cost-calib-v1`）。

**每条校验都必须能失败** —— 否则"校验通过"不含信息。
本文件对每条规则都构造一个"故意违反它"的输入，断言守卫会报出来。
（这正是 `experiment-harness-integrity` 里那条：守卫要在缺陷上失败。）
"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from costcalib_grid import DEFAULT_GRID, expand, load_grid, validate  # noqa: E402


def _grid():
    return copy.deepcopy(load_grid(DEFAULT_GRID))


class TestRealGridPasses(unittest.TestCase):
    def test_shipped_grid_is_valid(self):
        self.assertEqual(validate(_grid()), [])

    def test_protocol_version_is_registered(self):
        self.assertEqual(_grid()["protocol_version"], "cost-calib-v1")


class TestGuardsCanFail(unittest.TestCase):
    """每条守卫配一个"故意违反"的输入。"""

    def test_guard_fires_if_block_A_is_declared_fittable(self):
        """★ 验收第 2 条：快态平台**只能作下界**，不得被当成拟合对象。

        若有人把 `A.role` 改成 `fitted`，守卫必须报出来 ——
        否则"SLO 预测只拟合了快态最小值"这件事会**无声发生**。
        """
        g = _grid()
        g["blocks"]["A"]["role"] = "fitted"
        problems = validate(g)
        self.assertTrue(any("lower_bound_only" in x for x in problems), problems)

    def test_guard_fires_if_block_B_is_not_the_fitted_one(self):
        g = _grid()
        g["blocks"]["B"]["role"] = "lower_bound_only"
        self.assertTrue(any("blocks.B.role" in x for x in validate(g)))

    def test_guard_fires_on_string_instead_of_list_for_forbidden_split(self):
        """★ 字符串会被 `set()` **逐字符拆开** ⇒ 交集恒空 ⇒ 守卫静默失效。

        这条用例存在的理由就是让那种"看起来登记了、其实一条没匹配上"无法发生。
        """
        g = _grid()
        g["session"]["forbidden_split"] = "stepwise_random"      # 故意退化成字符串
        problems = validate(g)
        self.assertTrue(any("必须是列表" in x for x in problems), problems)

    def test_guard_fires_when_forbidden_split_is_unknown(self):
        g = _grid()
        g["session"]["forbidden_split"] = ["something_else"]
        self.assertTrue(any("没有登记被禁止的划分方式" in x for x in validate(g)))

    def test_guard_fires_when_cross_session_merge_is_allowed(self):
        """COST_PROFILE §6：绝对值跨会话漂移 1.3–4% ⇒ 不得并表。"""
        g = _grid()
        g["session"]["cross_session_merge"] = "allowed"
        self.assertTrue(any("cross_session_merge" in x for x in validate(g)))

    def test_guard_fires_when_a_group_has_no_why(self):
        g = _grid()
        g["groups"][0]["why"] = "   "
        self.assertTrue(any("缺 why" in x for x in validate(g)))

    def test_guard_fires_when_a_point_is_outside_the_axes_domain(self):
        g = _grid()
        g["groups"][0]["batch"] = [3]          # 3 不在 axes.batch=[1,2,4,8] 里
        self.assertTrue(any("不在 axes.batch" in x for x in validate(g)))

    def test_guard_fires_when_model_revision_is_main(self):
        """写 `main` 等于没钉住模型版本 —— 模型换了也看不出来。"""
        g = _grid()
        g["fingerprint"]["model"]["revision"] = "main"
        self.assertTrue(any("具体 commit" in x for x in validate(g)))

    def test_guard_fires_when_fingerprint_category_missing(self):
        g = _grid()
        del g["fingerprint"]["precision"]
        self.assertTrue(any("fingerprint.precision" in x for x in validate(g)))

    def test_guard_fires_when_split_unit_is_forbidden(self):
        g = _grid()
        g["session"]["split_unit"] = ["stepwise_random"]
        problems = validate(g)
        self.assertTrue(any("不允许的单位" in x for x in problems), problems)


class TestExpand(unittest.TestCase):
    def test_dedups_across_groups(self):
        """跨组重复的点只留一个（否则"每个形状采几次"不可数、白花机时）。"""
        shapes = expand(_grid())
        keys = [s.as_key() for s in shapes]
        self.assertEqual(len(keys), len(set(keys)))

    def test_expansion_is_deterministic(self):
        self.assertEqual([s.as_key() for s in expand(_grid())],
                         [s.as_key() for s in expand(_grid())])

    def test_serial_group_loses_the_duplicate_point(self):
        """`serial_forward_group` 的 `n_forwards=1` 与 `mixed_serving` 的点重合 ⇒ 少一个。"""
        g = _grid()
        serial = [s for s in expand(g) if s.group == "serial_forward_group"]
        self.assertEqual(len(serial), 3)          # 声明了 4 个，去重后 3 个
        self.assertEqual(sorted(s.n_forwards for s in serial), [1, 2, 3, 4][1:])

    def test_expand_empty_grid_is_empty_not_an_error(self):
        self.assertEqual(expand({}), [])


if __name__ == "__main__":
    unittest.main()
