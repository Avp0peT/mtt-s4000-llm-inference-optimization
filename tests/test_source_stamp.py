"""部署戳（``SOURCE_PROVENANCE.json``）的守卫测试。

守的是"**版本追溯到底成不成立**"这条线：远端源码树不带 ``.git`` 是常态，
所以 ``git_commit`` 只能靠部署戳。而戳有两种**不报错**的坏法，本文件逐条钉死：

  1. **从投下去就过期**（写端/读端的作用域或算法分家）——现场只表现为
     "``git_commit`` 又是 ``NOT_MEASURED``"，与"从来没投过"完全无法区分；
  2. **戳与树不符**（推了一半、远端残留旧文件）——它给出一个**错的** commit。
     这比缺值更危险：不报错、不留痕，还会变成下游结论的依据，
     并且看起来像一次成功的采集。

因此这里的断言不是"函数能跑"，而是"**它在该失败时确实失败**"。每个关键判据
都配了反向对照：把输入改成本不该被拦的形态，同一段代码必须放行——否则人会去
关掉这个检查，而一个总在误报的守卫最终等于没有守卫（典型例：把 ``__pycache__``
算进摘要，于是每次跑完测试戳都"过期"）。

另有一条端到端测试会**真的执行**投放校验用的那段脚本（``_VERIFY_SCRIPT``），
用真实输出喂给解析器。理由：那段脚本是字符串形式的，解析常量与它之间是
两处独立的真相，只有真跑一次才能证明它们还对得上。
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.provenance import (  # noqa: E402
    SOURCE_DIGEST_SCOPE,
    SOURCE_STAMP_NAME,
    SourceStampError,
    build_source_stamp,
    compute_tree_digest,
    detect_git_commit,
    detect_git_dirty,
    detect_source_commit,
    detect_source_identity,
    iter_scope_files,
    read_source_stamp,
)

#: 三个作用域目录各放一个文件，形状与真实源码树一致。
_SCOPE_FILES = (
    "src/ascend_opt/a.py",
    "benchmark/b.py",
    "scripts/c.py",
)

_FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _write(path: Path, text: str) -> None:
    """写文件。**显式 ``newline="\\n"``**：Windows 上默认会把 ``\\n`` 翻成 ``\\r\\n``，
    于是同一份内容在两个平台上的字节不同，而摘要比的就是字节。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git"] + args, cwd=str(cwd), capture_output=True, text=True
    )
    if check and proc.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} 失败：{proc.stderr.strip()}")
    return proc


def _init_git_repo(root: Path) -> bool:
    """初始化一个**配置自足**的 git 仓库；环境里没有 git 时返回 False。

    ``core.autocrlf=false`` 是必需的：本机全局配置是 ``true``，而测试写的文件是
    LF。若沿用全局配置，git 会认为工作区里的 LF 文件"应该"是 CRLF，于是把每个
    文件都报成已修改 —— 测试断言"干净树"时就会看到一个恒为真的"脏"，
    属于**用错误的原因通过/失败**。
    """
    if shutil.which("git") is None:
        return False
    try:
        _git(["init", "-q"], root)
        _git(["config", "user.email", "stamp-test@example.invalid"], root)
        _git(["config", "user.name", "stamp-test"], root)
        _git(["config", "core.autocrlf", "false"], root)
    except AssertionError:
        return False
    return True


def _load_remote_run():
    """按路径加载 ``scripts/remote_run.py``（paramiko 是延迟导入的，加载不需要它）。"""
    path = _ROOT / "scripts" / "remote_run.py"
    if not path.exists():
        raise unittest.SkipTest(f"缺少脚本 {path}")
    spec = importlib.util.spec_from_file_location("_remote_run_under_test_stamp", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 摘要本身
# --------------------------------------------------------------------------- #


class TestTreeDigest(unittest.TestCase):
    def test_empty_tree_returns_none_not_empty_string(self):
        """取不到摘要必须是 ``None``：空串会被读成"算出来了，且是个空值"。"""
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(compute_tree_digest(td))

    def test_scope_dirs_missing_returns_none(self):
        """作用域目录一个都不存在 ⇒ 没算成，不是"算出一个空的"。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write(root / "results" / "x.json", "{}")
            self.assertIsNone(compute_tree_digest(root))

    def test_missing_some_scope_dirs_is_tolerated(self):
        """只缺部分作用域目录时仍要能算出摘要，且等于"只有这些目录"的那棵树。"""
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as td2:
            a, b = Path(td), Path(td2)
            _write(a / "src/ascend_opt/a.py", "# a\n")
            _write(a / "scripts/c.py", "# c\n")
            _write(b / "src/ascend_opt/a.py", "# a\n")
            _write(b / "scripts/c.py", "# c\n")
            self.assertIsNotNone(compute_tree_digest(a))
            self.assertEqual(compute_tree_digest(a), compute_tree_digest(b))

    def test_creation_order_does_not_matter(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as td2:
            a, b = Path(td), Path(td2)
            for rel in _SCOPE_FILES:
                _write(a / rel, "# v1\n")
            for rel in reversed(_SCOPE_FILES):
                _write(b / rel, "# v1\n")
            self.assertEqual(compute_tree_digest(a), compute_tree_digest(b))

    def test_content_change_changes_digest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in _SCOPE_FILES:
                _write(root / rel, "# v1\n")
            before = compute_tree_digest(root)
            _write(root / "benchmark" / "b.py", "# v2\n")
            self.assertNotEqual(before, compute_tree_digest(root))

    def test_filename_change_changes_digest(self):
        """改名必须被发现：只对内容求和在"删一个+加一个同内容文件"时会漏。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in _SCOPE_FILES:
                _write(root / rel, "# v1\n")
            before = compute_tree_digest(root)
            (root / "scripts" / "c.py").rename(root / "scripts" / "d.py")
            self.assertNotEqual(before, compute_tree_digest(root))

    def test_pycache_is_excluded(self):
        """``__pycache__``/``.pyc`` 不算进去。

        这不是洁癖：若算进去，那么"远端跑过一次"就会改摘要 ⇒ 戳每次都被判过期
        ⇒ 人只好不再投放戳。一个总在误报的守卫等于没有守卫。
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in _SCOPE_FILES:
                _write(root / rel, "# v1\n")
            before = compute_tree_digest(root)
            _write(root / "scripts" / "__pycache__" / "c.cpython-310.pyc", "junk")
            _write(root / "src/ascend_opt" / "a.pyc", "junk")
            self.assertEqual(before, compute_tree_digest(root))

    def test_iter_scope_files_is_sorted_and_covers_scope(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in _SCOPE_FILES:
                _write(root / rel, "# v1\n")
            files = iter_scope_files(root)
            self.assertEqual([r for r, _ in files], sorted(r for r, _ in files))
            self.assertEqual({r for r, _ in files}, set(_SCOPE_FILES))


# --------------------------------------------------------------------------- #
# 写入端
# --------------------------------------------------------------------------- #


class TestBuildSourceStamp(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        if not _init_git_repo(self.root):
            self._td.cleanup()
            self.skipTest("环境里没有可用的 git")
        for rel in _SCOPE_FILES:
            _write(self.root / rel, "# v1\n")
        _git(["add", "-A"], self.root)
        _git(["commit", "-qm", "init"], self.root)

    def tearDown(self):
        self._td.cleanup()

    def test_clean_tree_stamps_with_head(self):
        stamp = build_source_stamp(self.root)
        self.assertEqual(stamp["git_commit"], detect_git_commit(self.root))
        self.assertIs(stamp["git_dirty"], False)
        self.assertEqual(stamp["digest_scope"], list(SOURCE_DIGEST_SCOPE))
        self.assertEqual(stamp["tree_digest"], compute_tree_digest(self.root))
        self.assertTrue(stamp["deployed_at"])
        self.assertTrue(stamp["deployed_by"])

    def test_not_a_git_repo_is_refused(self):
        """非 git 仓库 ⇒ 没有 commit 可记 ⇒ 必须抛错，不能产出一个空戳。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in _SCOPE_FILES:
                _write(root / rel, "# v1\n")
            with self.assertRaises(SourceStampError) as ctx:
                build_source_stamp(root)
            self.assertIn("git", str(ctx.exception))

    def test_dirty_scope_is_refused_by_default(self):
        _write(self.root / "benchmark" / "b.py", "# edited, not committed\n")
        with self.assertRaises(SourceStampError) as ctx:
            build_source_stamp(self.root)
        msg = str(ctx.exception)
        self.assertIn("未提交", msg)
        self.assertIn("--allow-dirty", msg)

    def test_untracked_file_inside_scope_is_also_dirty(self):
        """未跟踪文件也算脏。

        它是"部署的代码无法从任何 commit 复现"的典型形态：文件在工作区里、
        会被推上远端、却不在 HEAD 里。``git diff --quiet`` 看不见它，
        所以这里用的是 ``git status --porcelain``。
        """
        _write(self.root / "scripts" / "brand_new.py", "# never committed\n")
        self.assertIs(detect_git_dirty(self.root), True)
        with self.assertRaises(SourceStampError):
            build_source_stamp(self.root)

    def test_allow_dirty_records_the_flag(self):
        _write(self.root / "benchmark" / "b.py", "# edited\n")
        stamp = build_source_stamp(self.root, allow_dirty=True)
        self.assertIs(stamp["git_dirty"], True)

    def test_changes_outside_scope_do_not_refuse(self):
        """作用域外的改动（新配置、新产物）**不得**让投放失败。

        反向对照：若把 ``configs/``、``results/`` 也算脏，那么每生成一份配置、
        每跑一个 run 都会让树变脏，人会直接把检查关掉。
        """
        _write(self.root / "results" / "raw" / "run1" / "meta.json", "{}")
        _write(self.root / "configs" / "mthreads" / "new.yaml", "a: 1\n")
        self.assertIs(detect_git_dirty(self.root), False)
        stamp = build_source_stamp(self.root)
        self.assertIs(stamp["git_dirty"], False)

    def test_empty_scope_is_refused(self):
        """作用域里一个文件都没有 ⇒ 摘要为 None ⇒ 必须拒绝，而不是投一个"永不失效"的戳。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            if not _init_git_repo(root):
                self.skipTest("环境里没有可用的 git")
            _write(root / "results" / "x.json", "{}")
            _git(["add", "-A"], root)
            _git(["commit", "-qm", "init"], root)
            with self.assertRaises(SourceStampError) as ctx:
                build_source_stamp(root)
            self.assertIn("摘要", str(ctx.exception))

    def test_root_missing_is_refused(self):
        with self.assertRaises(SourceStampError):
            build_source_stamp(self.root / "no_such_dir")

    def test_clean_git_repo_reports_not_dirty(self):
        """git 路径下 ``git_dirty`` 必须是 ``False``（查过、干净），不能是 ``None``。

        返回 ``None`` 看着无害，实际把唯一的警告信号关掉了：有未提交改动时，
        HEAD 同样不描述被测的那份代码，而记录里却只有一个光秃秃的 commit。
        """
        commit, source, dirty = detect_source_commit(self.root)
        self.assertEqual(source, "git")
        self.assertEqual(commit, detect_git_commit(self.root))
        self.assertIs(dirty, False)

    def test_dirty_git_repo_reports_the_flag(self):
        """反向对照：同一个 git 仓库，改一个作用域内的文件，同一个返回值要翻成 True。"""
        _write(self.root / "benchmark" / "b.py", "# uncommitted\n")
        commit, source, dirty = detect_source_commit(self.root)
        self.assertEqual(source, "git")
        self.assertIs(dirty, True)
        # 作用域外的改动不得让它变脏（否则守卫会被误报淹没）。
        _git(["checkout", "--", "benchmark/b.py"], self.root)
        _write(self.root / "results" / "raw" / "x.json", "{}")
        self.assertIs(detect_source_commit(self.root)[2], False)

    def test_round_trip_on_a_non_git_copy_yields_a_usable_commit(self):
        """**核心正向对照**：戳能不能在"非 git 的树副本"上救回 ``git_commit``。

        这正是在复刻真实远端：本机是 git 仓库（能出 commit），推过去的那棵树
        不带 ``.git``（读不到 commit），中间的桥就是那份戳。
        """
        local_commit = detect_git_commit(self.root)
        stamp = build_source_stamp(self.root)

        with tempfile.TemporaryDirectory() as td:
            remote = Path(td) / "ascend_probe"
            for rel in _SCOPE_FILES + ("results/raw/run1/x.json",):
                src = self.root / rel
                dst = remote / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    _write(dst, "{}\n")
            _write(remote / SOURCE_STAMP_NAME, json.dumps(stamp, indent=2) + "\n")

            # 前置条件：这棵树**真的**不是 git 仓库。否则下面会经由外层仓库
            # 取到 commit，测试就变成"用错误的原因通过"。
            self.assertEqual(detect_git_commit(remote), "")

            commit, source, dirty = detect_source_commit(remote)
            self.assertEqual(commit, local_commit)
            self.assertEqual(source, "stamp")
            self.assertIs(dirty, False)


# --------------------------------------------------------------------------- #
# 读取端
# --------------------------------------------------------------------------- #


def _stamp_dict(root: Path, **over) -> dict:
    stamp = {
        "stamp_version": 1,
        "git_commit": _FAKE_COMMIT,
        "git_dirty": False,
        "tree_digest": compute_tree_digest(root),
        "digest_scope": list(SOURCE_DIGEST_SCOPE),
        "deployed_at": "2026-09-23T10:00:00+0800",
        "deployed_by": "tester@host",
    }
    stamp.update(over)
    return stamp


class TestReadSourceStamp(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        for rel in _SCOPE_FILES:
            _write(self.root / rel, "# v1\n")

    def tearDown(self):
        self._td.cleanup()

    def _put_stamp(self, **over) -> None:
        _write(
            self.root / SOURCE_STAMP_NAME,
            json.dumps(_stamp_dict(self.root, **over), ensure_ascii=False) + "\n",
        )

    def test_no_stamp_is_none_not_false(self):
        """从来没有戳 ⇒ ``validated is None``（"没查"≠"查了不行"）。"""
        out = read_source_stamp(self.root)
        self.assertIsNone(out["validated"])
        self.assertIn(SOURCE_STAMP_NAME, out["reason"])
        self.assertEqual(detect_source_commit(self.root), ("", "none", None))

    def test_valid_stamp_is_used(self):
        self._put_stamp()
        out = read_source_stamp(self.root)
        self.assertIs(out["validated"], True)
        self.assertEqual(out["git_commit"], _FAKE_COMMIT)
        self.assertEqual(out["deployed_by"], "tester@host")
        self.assertEqual(detect_source_commit(self.root), (_FAKE_COMMIT, "stamp", False))

    def test_tampered_tree_makes_the_stamp_stale(self):
        """**核心反向对照**：部署之后改了源码 ⇒ 戳作废，commit 不得再被采信。"""
        self._put_stamp()
        _write(self.root / "benchmark" / "b.py", "# edited after deploy\n")
        out = read_source_stamp(self.root)
        self.assertIs(out["validated"], False)
        self.assertIsNone(out["git_commit"])
        self.assertIn("摘要不符", out["reason"])
        self.assertEqual(detect_source_commit(self.root), ("", "stamp_stale", None))

    def test_deleted_file_makes_the_stamp_stale(self):
        self._put_stamp()
        (self.root / "scripts" / "c.py").unlink()
        self.assertIs(read_source_stamp(self.root)["validated"], False)

    def test_scope_mismatch_is_reported_separately_from_content_change(self):
        """作用域变了与内容变了**必须能分开**：两者修法不同（重投 vs 对树）。

        混成一句"摘要不符"的话，人会先按"源码被改过"去查，查不出来，
        最后把戳当成坏的不用了。
        """
        self._put_stamp(digest_scope=["src/ascend_opt"])
        out = read_source_stamp(self.root)
        self.assertIs(out["validated"], False)
        self.assertIn("作用域", out["reason"])
        self.assertNotIn("摘要不符", out["reason"])

    def test_missing_digest_is_not_trusted(self):
        """没有摘要的戳无法判断是否过期 ⇒ 不得当作有效。"""
        self._put_stamp(tree_digest="")
        out = read_source_stamp(self.root)
        self.assertIsNot(out["validated"], True)
        self.assertIsNone(out["git_commit"])

    def test_missing_commit_is_not_trusted(self):
        self._put_stamp(git_commit="")
        out = read_source_stamp(self.root)
        self.assertIsNot(out["validated"], True)

    def test_unparsable_stamp_is_not_trusted(self):
        _write(self.root / SOURCE_STAMP_NAME, "{ this is not json")
        out = read_source_stamp(self.root)
        self.assertIsNot(out["validated"], True)
        self.assertIn("无法解析", out["reason"])

    def test_oversized_scope_list_is_reported(self):
        """作用域"少一项"与"多一项"都要被判出，不能只判其一。"""
        self._put_stamp(digest_scope=list(SOURCE_DIGEST_SCOPE) + ["tests"])
        self.assertIs(read_source_stamp(self.root)["validated"], False)

    def test_deployed_at_survives_a_failed_validation(self):
        """戳作废时仍要报出"谁在什么时候投的"——否则"该重新投放"这条结论没有落点。"""
        self._put_stamp()
        _write(self.root / "benchmark" / "b.py", "# edited\n")
        out = read_source_stamp(self.root)
        self.assertIs(out["validated"], False)
        self.assertEqual(out["deployed_at"], "2026-09-23T10:00:00+0800")
        self.assertEqual(out["deployed_by"], "tester@host")

    def test_stamp_file_itself_does_not_change_the_digest(self):
        """戳写在树根、不在作用域里 ⇒ 投放动作本身不得把摘要改掉。

        否则会出现"投一次、过期一次"的死循环：每次投放都让下一次校验失败。
        """
        before = compute_tree_digest(self.root)
        self._put_stamp()
        self.assertEqual(before, compute_tree_digest(self.root))
        self.assertIs(read_source_stamp(self.root)["validated"], True)

    def test_git_dirty_true_is_preserved(self):
        self._put_stamp(git_dirty=True)
        out = read_source_stamp(self.root)
        self.assertIs(out["git_dirty"], True)
        self.assertIs(detect_source_commit(self.root)[2], True)

    def test_real_git_repo_wins_over_stamp(self):
        """真 git 仓库优先于戳：git 是**当前**状态，戳最多只说明部署那一刻。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            if not _init_git_repo(root):
                self.skipTest("环境里没有可用的 git")
            for rel in _SCOPE_FILES:
                _write(root / rel, "# v1\n")
            _write(
                root / SOURCE_STAMP_NAME,
                json.dumps(_stamp_dict(root), ensure_ascii=False) + "\n",
            )
            _git(["add", "-A"], root)
            _git(["commit", "-qm", "init"], root)
            head = detect_git_commit(root)
            commit, source, dirty = detect_source_commit(root)
            self.assertEqual(source, "git")
            self.assertEqual(commit, head)
            self.assertNotEqual(commit, _FAKE_COMMIT)

    def test_root_none_does_not_crash(self):
        out = read_source_stamp(None)
        self.assertIsNone(out["validated"])
        self.assertEqual(detect_source_commit(None), ("", "none", None))


class TestDetectSourceIdentity(unittest.TestCase):
    """``detect_source_identity`` = ``detect_source_commit`` + 内容摘要。

    ★ 为什么需要它（2026-10-02）：本机工作区常年是脏的（实测 83 项未提交改动），
    此时 ``git_commit`` 只标出"分支点"，**不描述被测的那份代码**。两份 commit 相同、
    代码却不同的产物，在只比 commit 的续跑判据下**一模一样**，于是"复用了不同代码的
    产物"这件事不会报错。摘要级判据才能问出"这是不是当前这棵树跑出来的"。
    """

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        for rel in _SCOPE_FILES:
            _write(self.root / rel, "# v1\n")

    def tearDown(self):
        self._td.cleanup()

    def _put_stamp(self, **over) -> None:
        _write(
            self.root / SOURCE_STAMP_NAME,
            json.dumps(_stamp_dict(self.root, **over), ensure_ascii=False) + "\n",
        )

    def test_no_stamp_gives_none_digest(self):
        """没有戳 ⇒ 摘要 ``None``（**不可用**），不是空串。"""
        commit, source, dirty, digest = detect_source_identity(self.root)
        self.assertEqual((commit, source, dirty), ("", "none", None))
        self.assertIsNone(digest)

    def test_valid_stamp_carries_digest(self):
        """有效戳 ⇒ commit 与摘要**一起**给出（这正是 commit 单值给不了的信息）。"""
        self._put_stamp()
        commit, source, dirty, digest = detect_source_identity(self.root)
        self.assertEqual(commit, _FAKE_COMMIT)
        self.assertEqual(source, "stamp")
        self.assertIs(dirty, False)
        self.assertEqual(digest, compute_tree_digest(self.root))
        self.assertTrue(digest)

    def test_stamp_stale_gives_no_digest(self):
        """戳过期 ⇒ 摘要与 commit 一起作废。**不得**返回一个过期摘要。"""
        self._put_stamp()
        _write(self.root / "benchmark" / "b.py", "# edited after deploy\n")
        commit, source, dirty, digest = detect_source_identity(self.root)
        self.assertEqual(source, "stamp_stale")
        self.assertEqual(commit, "")
        self.assertIsNone(digest)

    def test_git_branch_has_no_digest(self):
        """真 git 树上不给摘要——本机开发树的语义是 HEAD + 未提交改动，
        一个纯 HEAD 摘要反而不描述它。这里只断言"不给"，不假装它给了。"""
        real = Path(__file__).resolve().parents[1]
        commit, source, _dirty, digest = detect_source_identity(real)
        if source != "git":
            self.skipTest("本仓库不是 git 仓库，无法构造这条对照")
        self.assertTrue(commit)
        self.assertIsNone(digest)

    def test_digest_changes_when_content_changes(self):
        """**核心反向对照**：换了内容 ⇒ 摘要必须变。

        否则续跑判据就是恒真的——两份不同代码的产物会被判成"同一棵树"。
        """
        self._put_stamp()
        before = detect_source_identity(self.root)[3]
        # 改内容后重新投放戳（模拟"同一 commit、不同代码又被部署了一次"）
        _write(self.root / "benchmark" / "b.py", "# v2 different\n")
        self._put_stamp(git_commit=_FAKE_COMMIT)
        after = detect_source_identity(self.root)[3]
        self.assertIsNotNone(before)
        self.assertIsNotNone(after)
        self.assertNotEqual(before, after)

    def test_empty_digest_in_stamp_is_treated_as_unavailable(self):
        """戳里摘要为空串 ⇒ ``None``。空串是"不可用"的另一种写法，
        判据必须把它当**拒绝复用**，不能当"这一项不用比"。"""
        self._put_stamp(tree_digest="")
        _commit, source, _dirty, digest = detect_source_identity(self.root)
        # 空摘要的戳本就读不过校验 ⇒ 走的是 stamp_stale 分支
        self.assertNotEqual(source, "stamp")
        self.assertIsNone(digest)


class TestDetectGitDirty(unittest.TestCase):
    def test_not_a_repo_is_none_not_false(self):
        """非 git 仓库要返回 ``None``（"这个检查在这里不成立"）而不是 ``False``
        （"查过了，是干净的"）—— 两者含义相反。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for rel in _SCOPE_FILES:
                _write(root / rel, "# v1\n")
            self.assertIsNone(detect_git_dirty(root))

    def test_missing_root_is_none(self):
        self.assertIsNone(detect_git_dirty("/definitely/not/here"))


class TestNoAccidentalAttribution(unittest.TestCase):
    """``root=None`` 必须是"没查"，**不得**回落到进程当前目录。

    这条是被一个测试逼出来的真缺陷（原先 ``subprocess.run(cwd=None)`` 会继承
    调用者的 cwd）：``detect_source_commit(None)`` 曾经返回**本开发仓库**的 HEAD。
    在开发机上跑远端实验时，那意味着产物里会出现一个与远端毫无关系的 commit，
    且不报错——缺值可以看出来，错值看不出来。

    反向对照写在同一条测试里：``detect_git_commit(_ROOT)`` 必须**非空**，
    证明"空结果"不是因为这里没有 git 仓库，而是因为**没查**。
    """

    def test_none_root_does_not_fall_back_to_cwd(self):
        head = detect_git_commit(_ROOT)
        if not head:
            self.skipTest("本仓库不是 git 仓库，无法构造这条对照")
        self.assertEqual(detect_git_commit(None), "")
        self.assertEqual(detect_source_commit(None), ("", "none", None))
        self.assertIsNone(detect_git_dirty(None))

    def test_a_record_without_repo_root_has_no_commit(self):
        """端到端：不传 ``repo_root`` 时 ``git_commit`` 必须是缺失标记，而不是某个哈希。"""
        from ascend_opt.provenance import DataLabel, ReproInputs, build_reproducibility_record

        if not detect_git_commit(_ROOT):
            self.skipTest("本仓库不是 git 仓库，无法构造这条对照")
        record = build_reproducibility_record(
            ReproInputs(run_id="t", config={}), label=DataLabel.TEST_DATA
        )
        self.assertEqual(record.git_commit, "")
        self.assertIn("git_commit", record.missing_fields())
        self.assertIn("git_commit", record.describe())
        self.assertEqual(record.describe()["git_commit"], "NOT_MEASURED")
        self.assertEqual(record.git_source, "none")


class TestRecordCarriesTreeDigest(unittest.TestCase):
    """写端必须把摘要落盘——否则续跑判据拿不到要比的东西，只能退回比 commit。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        for rel in _SCOPE_FILES:
            _write(self.root / rel, "# v1\n")

    def tearDown(self):
        self._td.cleanup()

    def _record(self):
        from ascend_opt.provenance import DataLabel, ReproInputs, build_reproducibility_record

        return build_reproducibility_record(
            ReproInputs(run_id="t", config={}, repo_root=self.root),
            label=DataLabel.TEST_DATA,
        )

    def test_digest_written_when_stamp_valid(self):
        _write(
            self.root / SOURCE_STAMP_NAME,
            json.dumps(_stamp_dict(self.root), ensure_ascii=False) + "\n",
        )
        record = self._record()
        want = compute_tree_digest(self.root)
        self.assertEqual(record.tree_digest, want)
        self.assertEqual(record.describe()["tree_digest"], want)
        self.assertEqual(record.git_source, "stamp")

    def test_digest_is_none_not_empty_string_without_stamp(self):
        """没戳 ⇒ 摘要写 JSON ``null``（== 不可用），**不**写空串。

        空串在 CSV 里会被读成"这一列没数据"，而这里想说的是"本次 run 的部署戳
        没带摘要"。续跑判据把 ``None`` 当**拒绝复用**。
        """
        record = self._record()
        self.assertIsNone(record.tree_digest)
        self.assertIsNone(record.describe()["tree_digest"])

    def test_digest_not_in_missing_fields(self):
        """摘要是**补充维度**，不进 REQUIRED_REPRO_FIELDS——否则所有历史 run
        一夜之间变成"元数据不完整"，preflight 全红。"""
        record = self._record()
        self.assertNotIn("tree_digest", record.missing_fields())


# --------------------------------------------------------------------------- #
# remote_run.py 里的投放/校验纯逻辑
# --------------------------------------------------------------------------- #


class TestRemoteRunStampHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_remote_run()

    def test_marker_matches_what_the_script_prints(self):
        """常量与脚本必须对得上。

        它们是两处独立的真相：改了脚本里的标记而没改解析常量，症状是
        "远端明明校验过了，客户端却说拿不到结果"——一个纯粹的假失败。
        """
        self.assertIn(f"{self.mod.VERIFY_MARKER}'", self.mod._VERIFY_SCRIPT)
        self.assertIn(f"{self.mod.VERIFY_PY_MARKER}'", self.mod._VERIFY_SCRIPT)
        self.assertIn(f"{self.mod.MANIFEST_MARKER}'", self.mod._MANIFEST_SCRIPT)

    def test_verify_script_reuses_the_deployed_module(self):
        """校验必须 import **远端那棵树**里的实现，不得另写一份摘要算法。

        两份实现迟早分家，而分家的表现是"戳从投下去就是过期的"——最难查的形态。
        """
        script = self.mod._VERIFY_SCRIPT
        self.assertIn("read_source_stamp", script)
        self.assertIn("sys.path.insert(0, 'src')", script)
        self.assertNotIn("hashlib", script)
        self.assertNotIn("rglob", script)

    def test_manifest_script_reuses_the_deployed_module(self):
        script = self.mod._MANIFEST_SCRIPT
        self.assertIn("iter_scope_files", script)
        self.assertNotIn("hashlib", script)

    def test_verification_requires_the_marker(self):
        """没有标记 ⇒ ``None``，绝不是"默认通过"。"""
        self.assertIsNone(self.mod.parse_stamp_verification("一切正常，真的\n"))
        self.assertIsNone(self.mod.parse_stamp_verification(""))

    def test_verification_rejects_malformed_payload(self):
        bad = self.mod.VERIFY_MARKER + "{oops"
        self.assertIsNone(self.mod.parse_stamp_verification(bad))

    def test_verification_rejects_non_object_payload(self):
        self.assertIsNone(self.mod.parse_stamp_verification(self.mod.VERIFY_MARKER + "[1,2]"))

    def test_verification_reads_the_dict_among_noise(self):
        payload = json.dumps({"validated": True, "git_commit": _FAKE_COMMIT})
        out = "noise\n" + self.mod.VERIFY_MARKER + payload + "\ntrailing\n"
        got = self.mod.parse_stamp_verification(out)
        self.assertEqual(got, {"validated": True, "git_commit": _FAKE_COMMIT})

    def test_verification_parses_the_real_script_output(self):
        """**端到端**：真跑一次 ``_VERIFY_SCRIPT``，用真实输出喂给解析器。

        这能同时抓住两类失效：脚本语法/导入坏了，以及输出格式与解析常量脱节。
        这里用**本仓库**当那棵树：它没有戳，所以结果是"没有戳"而不是崩溃。
        """
        if not sys.executable:
            self.skipTest("拿不到当前解释器路径")
        proc = subprocess.run(
            [sys.executable, "-c", self.mod._VERIFY_SCRIPT],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertTrue(self.mod.parse_verify_python(proc.stdout))
        got = self.mod.parse_stamp_verification(proc.stdout)
        self.assertIsNotNone(got, msg=f"输出里没有可解析的结果：{proc.stdout!r}")
        # 本仓库没有投放戳 ⇒ 必须是 None（没查）而不是 True/False。
        self.assertIsNone(got["validated"])
        self.assertTrue(got["reason"])

    def test_manifest_parses_real_script_output(self):
        if not sys.executable:
            self.skipTest("拿不到当前解释器路径")
        proc = subprocess.run(
            [sys.executable, "-c", self.mod._MANIFEST_SCRIPT],
            cwd=str(_ROOT), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        rows = self.mod.parse_manifest(proc.stdout)
        self.assertIsNotNone(rows)
        # 与本地那份 iter_scope_files 逐条相同（复用同一实现，因此必须一致）。
        expect = {rel: sha for rel, sha in iter_scope_files(_ROOT)}
        self.assertEqual(rows, expect)
        self.assertIn("src/ascend_opt/provenance.py", rows)

    def test_manifest_returns_none_when_absent(self):
        self.assertIsNone(self.mod.parse_manifest("no manifest here\n"))

    def test_manifest_ignores_a_line_without_a_hash(self):
        out = self.mod.MANIFEST_MARKER + "src/a.py\n" + self.mod.MANIFEST_MARKER + "src/b.py deadbeef\n"
        self.assertEqual(self.mod.parse_manifest(out), {"src/b.py": "deadbeef"})

    def test_coverage_exact_and_by_ancestor(self):
        scope = ("src/ascend_opt", "benchmark", "scripts")
        self.assertEqual(self.mod._pushed_covers_scope(["src/ascend_opt", "benchmark", "scripts"], scope), [])
        # 推了更上层目录 ⇒ 覆盖
        self.assertEqual(self.mod._pushed_covers_scope(["src", "benchmark", "scripts"], scope), [])
        # 尾随斜杠不应影响判定
        self.assertEqual(self.mod._pushed_covers_scope(["src/ascend_opt/", "benchmark/", "scripts/"], scope), [])

    def test_coverage_reports_uncovered_entries(self):
        scope = ("src/ascend_opt", "benchmark", "scripts")
        self.assertEqual(self.mod._pushed_covers_scope(["scripts"], scope), ["src/ascend_opt", "benchmark"])

    def test_coverage_prefix_does_not_match_partial_name(self):
        """``src/ascend`` **不能**被当成 ``src/ascend_opt`` 的祖先（前缀必须落在 / 上）。"""
        scope = ("src/ascend_opt",)
        self.assertEqual(self.mod._pushed_covers_scope(["src/ascend"], scope), ["src/ascend_opt"])

    def test_pushed_covers_scope_is_exposed_as_pure_function(self):
        """它是纯函数：不碰网络、不碰文件系统。"""
        self.assertEqual(self.mod._pushed_covers_scope([], []), [])


if __name__ == "__main__":
    unittest.main()
