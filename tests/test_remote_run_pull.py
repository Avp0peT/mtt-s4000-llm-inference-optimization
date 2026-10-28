"""``remote_run.py`` 取回（pull）落点语义的守卫测试。

要钉住的是一个**静默通过**的缺陷：解包目标选错层级。

远端 tar 内成员的顶层名是 ``<弧名>/``（弧名 = 远端目录的 basename）。若把它解进
**落点目录本身**，数据实际落在 ``<落点>/<弧名>/...``——多套一层。而当时的校验按
**tar 内部路径**去查，于是"每个文件都在、大小都对"全部通过：检查过了，数据却不在
预期位置。真踩过，而且是在拉取 36 个 run 的证据之后才发现的。

这里直接测 ``_extract_archive`` 的两条性质：

* 布局判定**会失败**（tar 顶层不是弧名时必须报出来，且**不写任何文件**）；
* 落点语义要求传入**父目录**——传落点目录本身就会得到嵌套，
  所以 ``do_pull`` 必须传 ``parent``。最后一条是刻意的"文档即测试"：
  若有人把前缀剥离挪进 ``_extract_archive`` 而 ``do_pull`` 仍传 ``parent``，
  就会双重剥离，这条会失败。
"""

from __future__ import annotations

import importlib.util
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_remote_run():
    """按路径加载 scripts/remote_run.py。

    paramiko 是在函数内延迟导入的，所以没有 paramiko 的解释器也能导入本模块
    并测试其中的纯逻辑——这正是把 ``_extract_archive`` 抽出来的好处之一。
    """
    path = _ROOT / "scripts" / "remote_run.py"
    if not path.exists():
        raise unittest.SkipTest(f"缺少脚本 {path}")
    spec = importlib.util.spec_from_file_location("_remote_run_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_tar(tar_path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, data in members.items():
            src = tar_path.parent / f"_{name.replace('/', '__')}.src"
            src.write_bytes(data)
            tar.add(str(src), arcname=name)
            src.unlink()


class TestExtractArchiveLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_remote_run()

    def test_correct_layout_extracts_under_dest_parent(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tar = root / "_a.tar"
            _make_tar(tar, {"run1/meta.json": b"{}", "run1/t.jsonl": b"x\n"})
            dest_parent = root / "out"
            dest_parent.mkdir()
            stripped, off = self.mod._extract_archive(tar, "run1", dest_parent)
            self.assertEqual(off, [])
            self.assertEqual(sorted(stripped), ["meta.json", "t.jsonl"])
            # 落点语义：dest_parent/run1/<内容>
            self.assertTrue((dest_parent / "run1" / "meta.json").is_file())
            self.assertTrue((dest_parent / "run1" / "t.jsonl").is_file())
            # 剥掉前缀后不应再含 "run1/"
            self.assertNotIn("run1/", "".join(stripped))

    def test_wrong_top_level_is_reported_and_nothing_is_written(self):
        """正向对照：tar 顶层与弧名不符时，判据必须失败。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tar = root / "_b.tar"
            _make_tar(tar, {"other/meta.json": b"{}"})
            dest_parent = root / "out"
            dest_parent.mkdir()
            stripped, off = self.mod._extract_archive(tar, "run1", dest_parent)
            leftovers = sorted(p.name for p in dest_parent.iterdir())
            self.assertEqual(stripped, {})
            self.assertEqual(off, ["other/meta.json"])
            # 布局不符时**不得**写任何东西，避免落点旁留下"半对"的目录
            self.assertEqual(leftovers, [])

    def test_passing_the_landing_dir_itself_nests_one_level(self):
        """文档即测试：落点语义要求传父目录；传落点目录本身会多套一层。

        这条固定住 ``do_pull`` 必须传 ``local_root.parent``，也防止有人把
        "剥前缀"挪进 ``_extract_archive`` 之后造成双重剥离。
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tar = root / "_c.tar"
            _make_tar(tar, {"run1/meta.json": b"{}"})
            landing = root / "raw" / "run1"
            landing.mkdir(parents=True)
            self.mod._extract_archive(tar, "run1", landing)
            self.assertTrue((landing / "run1" / "meta.json").is_file())
            self.assertFalse((landing / "meta.json").is_file())

    def test_size_mismatch_is_visible_to_the_caller_check(self):
        """落点存在但大小不符时，调用方的逐文件校验必须能发现。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tar = root / "_d.tar"
            _make_tar(tar, {"run1/meta.json": b"12345678"})
            landing = root / "raw" / "run1"
            landing.mkdir(parents=True)
            stripped, off = self.mod._extract_archive(tar, "run1", landing.parent)
            self.assertEqual(off, [])
            # 模拟"落了但被截断"
            (landing / "meta.json").write_bytes(b"123")
            bad = [
                rel for rel, size in stripped.items()
                if not (landing / rel).is_file()
                or (landing / rel).stat().st_size != size
            ]
        self.assertEqual(bad, ["meta.json"])


class TestPullCliWiring(unittest.TestCase):
    """pull 必须真的接到 CLI 上——"有实现但用不到"和没实现一样。

    注意断言要能**区分**两种情况：``pull`` 缺参数报的是缺 ``remote_dir``，
    而未知子命令报的是 ``invalid choice``。两者退出码都是 2，所以只看退出码
    区分不了——必须看报错文本，否则这条测试连"子命令根本没注册"都发现不了。
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_remote_run()

    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            with self.assertRaises(SystemExit) as cm:
                self.mod.main(argv)
        return int(cm.exception.code or 0), buf.getvalue()

    def test_pull_subcommand_requires_two_args(self):
        code, err = self._run_main(["pull"])
        self.assertEqual(code, 2)
        self.assertIn("remote_dir", err)
        self.assertIn("local_dir", err)

    def test_unknown_subcommand_is_a_different_error(self):
        """反向对照：未知子命令的报错文本与上面不同，证明上一条测的是真分支。"""
        code, err = self._run_main(["definitely-not-a-subcommand"])
        self.assertEqual(code, 2)
        self.assertIn("invalid choice", err)
        self.assertNotIn("remote_dir", err)


class TestPullLandingNameGuard(unittest.TestCase):
    """落点名与远端目录名不一致时，必须**先报错**且不做任何远端动作。

    为什么单列一条：这个不一致原先要等到"打包取回 → 解开 → 逐文件核对"
    之后才被发现，而那时数据其实已经被写到 ``<落点父目录>/<远端目录名>/``
    （多套一层的地方根本不是落点）。报错只说"文件缺失或大小不符"，
    完全看不出真正原因是"名字不一致"——踩过：把远端 `_pull_b` 拉到本地
    `results/_incoming_b`，症状像"拉取不完整"，排查方向整个是错的。

    本测试用**一碰就炸的假 client**：若实现走到了远端步骤，第一条断言就失败。
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_remote_run()

    class _ExplodingClient:
        def __getattr__(self, name):  # pragma: no cover - 只在实现跑偏时触发
            raise AssertionError(f"落点名不一致时不该碰 client（访问了 .{name}）")

    def test_mismatched_landing_name_fails_before_touching_remote(self):
        with tempfile.TemporaryDirectory() as td:
            landing = Path(td) / "_incoming_b"
            code = self.mod.do_pull(
                self._ExplodingClient(), "/root/probe/_pull_b", str(landing)
            )
        self.assertEqual(code, 5, "必须返回非零，不能静默继续")
        self.assertFalse(landing.exists(), "拒绝时不得创建落点")

    def test_error_message_names_the_real_constraint(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                code = self.mod.do_pull(
                    self._ExplodingClient(), "/root/probe/_pull_b",
                    str(Path(td) / "_incoming_b"),
                )
            err = buf.getvalue()
        self.assertEqual(code, 5)
        self.assertIn("_pull_b", err)
        self.assertIn("_incoming_b", err)
        self.assertIn("同名", err)

    def test_matching_name_passes_the_guard(self):
        """正向对照：同名时**不会**被这条守卫拦下（否则它就是个永远拒绝的假守卫）。

        用一个一碰就炸的 client 证明它确实走到了远端步骤——即守卫放行了。
        """
        with tempfile.TemporaryDirectory() as td:
            landing = Path(td) / "_pull_b"
            with self.assertRaises(AssertionError):
                self.mod.do_pull(
                    self._ExplodingClient(), "/root/probe/_pull_b", str(landing)
                )


if __name__ == "__main__":
    unittest.main()
