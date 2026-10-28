"""论文占位清点器：**"还剩多少坑"必须是可数的，且空输入不得伪装成通过**。

三条判据（`scripts/list_paper_placeholders.py`）
------------------------------------------------
1. 有占位 ⇒ 非零退出（那是"还不能投稿"的**机器可读信号**，不是报错）；
2. 占位 ID **全局唯一**（重复会让"补第 P03 个"指到两处）；
3. **空输入 ⇒ rc=2**，不得当成"全部通过"。

第 3 条是本项目反复吃亏的那个形态：路径写错 ⇒ 扫到 0 个文件 ⇒ 打印"✓ 无占位"，
而它实际验的是**空集**。所以这里专门有一条用例拿"空目录"去要求它**失败**。
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "scripts"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import list_paper_placeholders as L  # noqa: E402


def _run(argv: list[str]) -> tuple[int, str]:
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        rc = L.main(argv)
    return rc, so.getvalue() + se.getvalue()


class TestEmptyInputMustNotPass(unittest.TestCase):
    def test_empty_dir_is_not_a_pass(self):
        """★ 空输入最可能意味着"路径写错"，绝不能读成"无占位"。"""
        with tempfile.TemporaryDirectory() as td:
            rc, out = _run(["--dir", td])
        self.assertEqual(rc, 2, "空目录被判成通过 ⇒ 就是那个'空输入伪装成合格结果'的坑")
        self.assertIn("未判", out)

    def test_missing_dir_is_an_error(self):
        rc, _ = _run(["--dir", str(_ROOT / "no" / "such" / "dir")])
        self.assertEqual(rc, 2)


class TestPlaceholdersAreCounted(unittest.TestCase):
    def test_placeholders_force_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "01.md").write_text(
                "正文。\n【待补·P01】覆盖率的正式表述（来源：x；待跑）\n"
                "【待补·P02】另一处\n", encoding="utf-8", newline="\n")
            rc, out = _run(["--dir", td])
        self.assertEqual(rc, 1)
        self.assertIn("P01", out)
        self.assertIn("占位 2 处", out)

    def test_clean_dir_passes(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "01.md").write_text("正文，没有占位。\n", encoding="utf-8", newline="\n")
            rc, out = _run(["--dir", td])
        self.assertEqual(rc, 0)
        self.assertIn("可以进入下一轮评审", out)

    def test_duplicate_ids_are_reported(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "01.md").write_text("【待补·P01】a\n", encoding="utf-8", newline="\n")
            (Path(td) / "02.md").write_text("【待补·P01】b\n", encoding="utf-8", newline="\n")
            rc, out = _run(["--dir", td])
        self.assertEqual(rc, 1)
        self.assertIn("ID 重复", out)

    def test_lazy_tokens_are_flagged(self):
        """`TBD`/`待填` 是**缺失与测量值的混淆源** ⇒ 必须被点名。"""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "01.md").write_text("这里 TBD，那里待填。\n", encoding="utf-8",
                                            newline="\n")
            rc, out = _run(["--dir", td])
        self.assertEqual(rc, 1)
        self.assertIn("偷懒写法", out)


class TestJsonOutput(unittest.TestCase):
    def test_json_records_total_and_groups(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "03.md").write_text("【待补·P09】x\n", encoding="utf-8", newline="\n")
            jf = Path(td) / "out.json"
            rc, _ = _run(["--dir", td, "--json", str(jf)])
            d = json.loads(jf.read_text(encoding="utf-8"))
        self.assertEqual(rc, 1)
        self.assertEqual(d["total"], 1)
        self.assertIn("03.md", d["by_file"])


if __name__ == "__main__":
    unittest.main()
