"""`scripts/build_manuscript.py` 的用例：装配必须**要么完整、要么报错**。

为什么要测它
------------
装配脚本的失效形态都是**静默**的：少拼一节、把 §4 拼成半截、把换行符写成 CRLF ——
产出的 `manuscript.md` 依旧是一份**看上去完整**的文章，读者不会发现少了哪一节。
所以这里逐条钉住"可核对的结构事实"，并对**两个真会发生的输入缺陷**做反向验证
（故意把源文件改坏，确认脚本**报错而不是照样产出**）。

纪律：**双向验证**——只测"正常输入能过"是不够的，
必须证明"坏输入会失败"，否则守卫可能恒真。
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "build_manuscript.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_manuscript", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_manuscript"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestManuscriptBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load()

    def test_inputs_complete(self):
        """所有分节文件与 §4 来源都在盘上。"""
        missing = [n for n, _ in self.mod._ORDER if not (self.mod._SECTIONS / n).is_file()]
        self.assertEqual(missing, [], f"缺分节文件：{missing}")
        self.assertTrue(self.mod._FAILURE.is_file())

    def test_order_covers_all_phases(self):
        """装配顺序必须覆盖 00→09 全部十节，且 §4 的正文来自 failure_analysis。"""
        names = [n for n, _ in self.mod._ORDER]
        self.assertEqual(len(names), 10)
        self.assertEqual(names[0], "00_abstract.md")
        self.assertEqual(names[-1], "09_conclusion.md")
        self.assertEqual(names[4], "04_failure_analysis.md")

    def test_failure_body_starts_at_4_1(self):
        """§4 正文必须从 `## 4.1` 开始（丢掉工作文档头），且带 4.7 缺陷清单。"""
        body = self.mod._failure_body()
        self.assertTrue(body.lstrip().startswith("## 4.1"), body[:60])
        self.assertIn("## 4.7", body)
        self.assertNotIn("工作文档，三层标注", body)  # 文档头被丢掉

    def test_failure_body_missing_anchor_raises(self):
        """★ 反向验证：源文件里没有 `## 4.1` 时必须**报错**，不得静默拼半截。"""
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "failure_analysis.md"
            fake.write_text("# 空壳\n\n什么都没有。\n", encoding="utf-8", newline="\n")
            orig = self.mod._FAILURE
            self.mod._FAILURE = fake
            try:
                with self.assertRaises(SystemExit):
                    self.mod._failure_body()
            finally:
                self.mod._FAILURE = orig

    def test_missing_section_returns_2(self):
        """★ 反向验证：把某一节"搬走"后必须 rc=2（空/缺输入不得读成"通过"）。"""
        with tempfile.TemporaryDirectory() as td:
            fake_dir = Path(td)
            for n, _ in self.mod._ORDER[:-1]:  # 故意少最后一节
                (fake_dir / n).write_text("# x\n", encoding="utf-8", newline="\n")
            orig = self.mod._SECTIONS
            self.mod._SECTIONS = fake_dir
            try:
                rc = self.mod.main(["--check"])
            finally:
                self.mod._SECTIONS = orig
            self.assertEqual(rc, 2)

    def test_title_override_replaces_h1(self):
        """`title_override` 必须**替换**而不是追加一个标题。"""
        body = self.mod._section_body(self.mod._SECTIONS / "01_introduction.md", "9 假标题")
        lines = [ln for ln in body.splitlines() if ln.startswith("# ")]
        self.assertEqual(lines, ["# 9 假标题"])

    def test_no_override_keeps_h1(self):
        body = self.mod._section_body(self.mod._SECTIONS / "01_introduction.md", None)
        self.assertTrue(body.startswith("# 1 Introduction"), body[:40])

    def test_build_writes_lf_and_is_reproducible(self):
        """装配产物必须是 LF，且**两次装配逐字节相同**（否则就不是可重复构建）。"""
        with tempfile.TemporaryDirectory() as td:
            out1 = Path(td) / "a.md"
            out2 = Path(td) / "b.md"
            rc1 = self.mod.main(["--out", str(out1)])
            rc2 = self.mod.main(["--out", str(out2)])
            self.assertEqual(rc1, rc2)
            raw = out1.read_bytes()
            self.assertNotIn(b"\r", raw, "写出的文件里含 CR ⇒ 换行符被静默改写")
            self.assertEqual(raw, out2.read_bytes())

    def test_placeholder_count_matches_lister(self):
        """附录 B 的占位数必须与占位清点器一致（两处口径不得分叉）。"""
        sys.path.insert(0, str(_ROOT / "scripts"))
        import list_paper_placeholders as lpp  # noqa: PLC0415

        found, ids, lazy = lpp.scan(self.mod._SECTIONS)
        expected = sum(len(v) for v in found.values())
        _text, n = self.mod._placeholder_block()
        self.assertEqual(n, expected)
        self.assertEqual({k for k, v in ids.items() if len(v) > 1}, set(), "占位 ID 必须全局唯一")
        self.assertEqual(lazy, {}, f"不得有 TBD/待填类偷懒写法：{lazy}")


if __name__ == "__main__":
    unittest.main()
