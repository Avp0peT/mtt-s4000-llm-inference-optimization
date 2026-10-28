"""登记判定的回归用例（**逐字相等 vs markdown 内联装饰**）。

实测踩到（2026-09-26）
--------------------
`registered_in_table()` 逐字比较表格单元，而 `scripts/emit_experiment_rows.py`
生成的 run_id 单元写作 `` `S14B1-short_chat-fixed_cost` ``（**带反引号**）
⇒ 判定恒为 False。

后果不是"漏报"，而是**恒假**：`benchmark/preflight.py` 的检查 4 会把**已登记**的批次
判成"未登记"，`export_results.sh` 因此**永远导不出去**。而恒假的守卫与恒真的守卫一样
**没有信息量** —— 这正是它存活至今没人察觉的原因（同一批 run 在闸门处被拦、
在日常自查里被当成"就是没登记"）。

⚠️ 反向的坑也在这一条附近：修的时候**不能顺手把下划线也去掉** ——
run_id 自己就含 `_`（`fixed_cost`），去掉会变成 `fixedcost`，是另一种静默错配。
本文件对这一条单独设了用例。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preflight  # noqa: E402
from preflight import registered_in_table  # noqa: E402


def _table(run_id_cell: str) -> str:
    return ("| 编号 | run_id | 变量 |\n| --- | --- | --- |\n"
            f"| EXP-X-001 | {run_id_cell} | controller.kind=cost |\n")


class TestCellDecorationMustNotMatter(unittest.TestCase):
    def test_bare_cell_matches(self):
        self.assertTrue(registered_in_table(_table("T1-wl-fixed_cost"), "T1-wl-fixed_cost"))

    def test_backticked_cell_matches(self):
        """★ 本次回归：带反引号的单元必须匹配（生成器产出的就是这个形状）。"""
        self.assertTrue(registered_in_table(_table("`T1-wl-fixed_cost`"), "T1-wl-fixed_cost"))

    def test_bold_cell_matches(self):
        self.assertTrue(registered_in_table(_table("**T1-wl-fixed_cost**"), "T1-wl-fixed_cost"))

    def test_underscore_inside_run_id_is_preserved(self):
        """⚠️ 不许把下划线也当装饰去掉 —— 去了 `fixed_cost` 就变成 `fixedcost`。"""
        self.assertTrue(registered_in_table(_table("`T1-wl-fixed_cost`"), "T1-wl-fixed_cost"))
        # 反向：不能因为"去了装饰"就让一个**不同**的 id 命中
        self.assertFalse(registered_in_table(_table("`T1-wl-fixedcost`"), "T1-wl-fixed_cost"))


class TestStrictnessMustBeKept(unittest.TestCase):
    """"去装饰" ≠ "放宽到全文包含"。规范要的是**表格行**（含重复次数、离散度）。"""

    def test_prose_mention_does_not_count(self):
        prose = "本次实验 T1-wl-fixed_cost 跑得不错。\n"
        self.assertFalse(registered_in_table(prose, "T1-wl-fixed_cost"))

    def test_backticked_prose_still_does_not_count(self):
        """散文里的行内代码同样不算 —— 否则"去装饰"就把严格性一起丢了。"""
        prose = "见 `T1-wl-fixed_cost` 那一批。\n"
        self.assertFalse(registered_in_table(prose, "T1-wl-fixed_cost"))

    def test_non_table_line_does_not_count(self):
        self.assertFalse(registered_in_table("- T1-wl-fixed_cost\n", "T1-wl-fixed_cost"))

    def test_empty_text(self):
        self.assertFalse(registered_in_table("", "T1-wl-fixed_cost"))


class TestOldCaliberWouldHaveFailed(unittest.TestCase):
    def test_old_logic_fails_on_backticked_cell(self):
        """把**旧口径**（不去装饰）也算一遍并断言它失败。

        这样"为什么这一条必须存在"就不是注释，而是一条会失败的断言：
        谁把 `_cell_text` 去掉，本用例立刻变红。
        """
        def old_logic(text: str, run_id: str) -> bool:
            for line in text.splitlines():
                s = line.strip()
                if not s.startswith("|"):
                    continue
                if run_id in [c.strip() for c in s.strip("|").split("|")]:
                    return True
            return False

        text = _table("`T1-wl-fixed_cost`")
        self.assertFalse(old_logic(text, "T1-wl-fixed_cost"),
                         "旧口径在这个样本上应当失败（否则本用例没在测回归）")
        self.assertTrue(registered_in_table(text, "T1-wl-fixed_cost"))


class TestSingleImplementation(unittest.TestCase):
    def test_traceability_reuses_the_same_function_object(self):
        """★ 结构守卫：日常自查与导出闸门必须是**同一个函数对象**。

        若哪天有人在 `check_traceability.py` 里另写一套（比如"包含即可"的宽松版），
        两套口径就会分叉：导出闸门拦住、自查却绿灯 —— 而那正是本项目最怕的形态。
        """
        import check_traceability
        self.assertIs(check_traceability.registered_in_table, preflight.registered_in_table,
                      "登记判定出现了第二份实现 ⇒ 两处口径会分叉")


if __name__ == "__main__":
    unittest.main()
