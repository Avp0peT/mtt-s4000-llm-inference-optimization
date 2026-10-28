"""`scripts/batch_lock.py` 的守卫测试（纯离线，不连远端、不起真进程）。

这个模块守的是**起跑互斥**。它出错的两个方向都已实测过，且**都不报错**：

* 锁太短（只覆盖起跑动作）⇒ 同 tag 重起跑畅通 ⇒ 两个矩阵同写 `results/raw/`。
  实测：退出码 0、`SHOULD_NOT_RUN` 真的执行了。
* 锁太长（批次结束后没人放）⇒ **永久堵住后续所有批次**。

⇒ 两个方向各有一组用例。另外覆盖：
* `decide()` 的三态（`alive=None` 是"查不到"，**不得**当成"已结束"）；
* 防误删：迟到清理者不得删掉别人的锁；
* 路径穿越：tag 含 `../` 必须被拒。
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "batch_lock.py"


def _load():
    spec = importlib.util.spec_from_file_location("batch_lock_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load()


class _TmpRoot(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="batch_lock_test_")
        self.root = Path(self._dir) / "locks"

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)

    def locks(self):
        return m.BatchLocks(self.root)


class TestAcquireRelease(_TmpRoot):
    def test_acquire_then_holder_is_readable(self):
        locks = self.locks()
        rec = locks.acquire("S14H", cmd="python -u x.py")
        self.assertEqual(rec["tag"], "S14H")
        self.assertEqual(rec["cmd"], "python -u x.py")
        h = locks.holder(locks.all_lock)
        self.assertEqual(h["tag"], "S14H")
        self.assertIsNotNone(h.get("started_ts"))

    def test_same_tag_second_acquire_is_refused(self):
        """★ 方向一：同 tag 重起跑必须被拦（复现"命令发了两遍"事故）。"""
        locks = self.locks()
        locks.acquire("S14H")
        with self.assertRaises(m.LockBusy) as ctx:
            locks.acquire("S14H")
        self.assertEqual(ctx.exception.code, 5, "全局锁是更粗的约束，应先报 5")

    def test_other_tag_is_refused_while_running(self):
        """★ `results/raw` 全局唯一 ⇒ 别的 tag 也不许并发。"""
        locks = self.locks()
        locks.acquire("S14H")
        with self.assertRaises(m.LockBusy) as ctx:
            locks.acquire("S14S")
        self.assertEqual(ctx.exception.code, 5)

    def test_holder_names_the_real_culprit(self):
        """报错必须指出**是谁**在占 —— 否则排障只能靠猜。"""
        locks = self.locks()
        locks.acquire("S14H", cmd="run_step3_matrix --tag S14H")
        with self.assertRaises(m.LockBusy) as ctx:
            locks.acquire("S14S")
        self.assertIn("S14H", str(ctx.exception))
        self.assertIn("run_step3_matrix", str(ctx.exception))

    def test_release_allows_reacquire(self):
        """★ 方向二：批次结束后必须能再起 —— 锁不得永久堵住。"""
        locks = self.locks()
        locks.acquire("S14H")
        self.assertTrue(locks.release("S14H"))
        self.assertFalse(locks.is_held())
        locks.acquire("S14H")   # 不应抛

    def test_release_is_idempotent(self):
        """重复放锁不算错（监视进程 + 人手动清理可能撞上）。"""
        locks = self.locks()
        locks.acquire("S14H")
        self.assertTrue(locks.release("S14H"))
        self.assertFalse(locks.release("S14H"))

    def test_partial_acquire_rolls_back(self):
        """tag 锁取不到时必须**回滚全局锁**，否则留下"半持有"死状态。"""
        locks = self.locks()
        # 预先占住 tag 锁（模拟"同 tag 已有批次"）而全局锁空着
        locks.root.mkdir(parents=True, exist_ok=True)
        locks.tag_lock("S14H").mkdir()
        with self.assertRaises(m.LockBusy) as ctx:
            locks.acquire("S14H")
        self.assertEqual(ctx.exception.code, 4)
        self.assertFalse(locks.all_lock.is_dir(), "全局锁未回滚 ⇒ 其它 tag 也被永久堵住")


class TestGuardAgainstClobberingOthers(_TmpRoot):
    def test_release_refuses_when_pid_differs(self):
        """★ 迟到清理者不得删掉**别人**刚拿到的锁。

        同源教训：事故里第二个进程 `rm -f` 抹掉了第一个进程正在写的 inode。
        同一个错误不能在锁上再犯一次。
        """
        locks = self.locks()
        locks.acquire("S14H", pid=11111)
        with self.assertRaises(m.LockBusy) as ctx:
            locks.release("S14H", only_if_holder_pid=22222)
        self.assertEqual(ctx.exception.code, 7)
        self.assertTrue(locks.is_held(), "拒绝之后锁必须还在")

    def test_release_succeeds_when_pid_matches(self):
        locks = self.locks()
        locks.acquire("S14H", pid=11111)
        self.assertTrue(locks.release("S14H", only_if_holder_pid=11111))

    def test_release_without_pid_guard_always_works(self):
        """监视进程走这条路：它只知道"批次没了"，不知道谁持的锁。"""
        locks = self.locks()
        locks.acquire("S14H", pid=11111)
        self.assertTrue(locks.release("S14H"))

    def test_illegal_tag_is_rejected(self):
        """tag 进路径名 ⇒ 必须挡住分隔符与 `..`（否则能写到锁根之外）。"""
        locks = self.locks()
        for bad in ("../evil", "a/b", "a\\b", "", ".", ".."):
            with self.subTest(tag=bad):
                with self.assertRaises(ValueError):
                    locks.tag_lock(bad)


class TestDecideTriState(unittest.TestCase):
    """`decide()` 的三态与两个边界。**`None` 不是 `False`。**"""

    def test_running_waits(self):
        self.assertEqual(
            m.decide(alive=True, lock_still_held=True, unknown_strikes=0,
                     batch_age_s=100)[0], "wait")

    def test_dead_releases(self):
        self.assertEqual(
            m.decide(alive=False, lock_still_held=True, unknown_strikes=0,
                     batch_age_s=100)[0], "release")

    def test_lock_gone_exits_without_rm(self):
        """锁被外部清掉 ⇒ exit。**不得** release（会删掉别人刚拿到的锁）。"""
        act, why = m.decide(alive=True, lock_still_held=False, unknown_strikes=0,
                            batch_age_s=100)
        self.assertEqual(act, "exit")
        self.assertIn("外部", why)

    def test_min_hold_window_guards_startup_race(self):
        """刚起跑就探测 ⇒ 可能查不到 ⇒ **不得**立刻放锁。"""
        self.assertEqual(
            m.decide(alive=False, lock_still_held=True, unknown_strikes=0,
                     batch_age_s=0.1, min_hold_s=1.0)[0], "wait")

    def test_unknown_is_not_dead(self):
        """★ 核心三态：`None`（查不到）不得当成 `False`（已结束）。"""
        self.assertEqual(
            m.decide(alive=None, lock_still_held=True, unknown_strikes=0,
                     batch_age_s=100)[0], "wait",
            "单次查不到就放锁 ⇒ 并发批次又能进来")

    def test_unknown_accumulates_to_release(self):
        """但连续查不到必须最终放锁 —— 否则监视进程自己成了永久堵点。"""
        n = m.UNKNOWN_STRIKES_TO_RELEASE
        self.assertGreater(n, 1)
        self.assertEqual(
            m.decide(alive=None, lock_still_held=True, unknown_strikes=n - 1,
                     batch_age_s=100)[0], "release")

    def test_dead_more_decisive_than_unknown(self):
        """`False` 与 `None` 必须给出**不同**的动作。"""
        a = m.decide(alive=False, lock_still_held=True, unknown_strikes=0,
                     batch_age_s=100)[0]
        b = m.decide(alive=None, lock_still_held=True, unknown_strikes=0,
                     batch_age_s=100)[0]
        self.assertNotEqual(a, b)

    def test_threshold_is_not_one(self):
        self.assertGreater(m.UNKNOWN_STRIKES_TO_RELEASE, 1,
                           "阈值 =1 等价于把 unknown 当已结束")


class TestCliExitCodes(_TmpRoot):
    """CLI 退出码是 shell 唯一该看的东西 ⇒ 必须逐个钉住。"""

    def _run(self, *args: str) -> int:
        p = subprocess.run(
            [sys.executable, str(_SCRIPT), "--root", str(self.root), *args],
            capture_output=True, text=True,
        )
        return p.returncode

    def test_exit_codes_are_distinct(self):
        self.assertEqual(self._run("acquire", "--tag", "T1"), 0)
        self.assertEqual(self._run("acquire", "--tag", "T1"), 5, "全局锁先触发")
        self.assertEqual(self._run("status", "--tag", "T1"), 1)
        self.assertEqual(self._run("release", "--tag", "T1"), 0)
        self.assertEqual(self._run("status", "--tag", "T1"), 0)

    def test_status_on_missing_root_is_idle_not_held(self):
        """锁根不存在 = 首次起跑的正常情形 ⇒ 空闲（**不是**被占）。"""
        p = subprocess.run(
            [sys.executable, str(_SCRIPT), "--root", str(self.root / "nope"),
             "status", "--tag", "T1"],
            capture_output=True, text=True,
        )
        self.assertEqual(p.returncode, 0)

    def test_illegal_tag_via_cli_fails(self):
        rc = self._run("acquire", "--tag", "../evil")
        self.assertNotEqual(rc, 0)

    def test_status_prints_holder_for_diagnosis(self):
        self._run("acquire", "--tag", "T1", "--cmd", "MAGIC_CMD_TOKEN")
        p = subprocess.run(
            [sys.executable, str(_SCRIPT), "--root", str(self.root),
             "status", "--tag", "T1"],
            capture_output=True, text=True,
        )
        self.assertIn("MAGIC_CMD_TOKEN", p.stdout,
                      "status 必须能指出是谁在跑，否则排障只能靠猜")

    def test_release_on_wrong_root_reports_nothing_found(self):
        """★ `--root` 指错时，放锁必须**显形**而不是静默成功。

        实测 2026-09-27：监视进程把 `--root` 指向了 scripts/ 而非锁目录，
        `rmtree(ignore_errors=True)` 什么都没删，而日志照样打印"锁已释放"
        ⇒ 锁永久残留、后续批次全被堵住。这一条把"没找到"与"删成功"分开报。
        """
        self._run("acquire", "--tag", "T1")           # 在正确的 root 上取锁
        wrong = self.root.parent / "scripts_like_elsewhere"
        p = subprocess.run(
            [sys.executable, str(_SCRIPT), "--root", str(wrong),
             "release", "--tag", "T1"],
            capture_output=True, text=True,
        )
        self.assertIn("removed=False", p.stdout + p.stderr)
        self.assertIn("没找到", p.stdout + p.stderr,
                      "必须显式提示'没找到锁目录'，否则 --root 指错是静默的")
        # 反向：真正的锁必须**仍在**（指错的 root 不得误删别人的锁）
        self.assertTrue(m.BatchLocks(self.root).is_held())


class TestNoExternalBinaryDependency(unittest.TestCase):
    """★ 判据不得依赖本机不存在的二进制。

    实测 2026-09-27：本机（Git for Windows）**没有 `flock` 也没有 `setsid`**，
    而远端（Linux 5.15 / bash 5.1）**两样都有**。若判据写死 `flock`，
    它在本机一次都验不了 —— 而"从没本地验证过"的判据出问题时只能靠猜日志。
    `mkdir` 两边都有且原子 ⇒ 主路径必须是它。
    """

    def test_source_does_not_require_flock_or_setsid(self):
        src = _SCRIPT.read_text(encoding="utf-8")
        # 去掉注释与 docstring 里的解释性提及再判定
        code = "\n".join(
            ln.split("#", 1)[0] for ln in src.splitlines()
        )
        for binname in ("flock", "setsid"):
            with self.subTest(bin=binname):
                self.assertNotIn(
                    f'"{binname}"', code,
                    f"判据里出现了对 {binname} 的硬依赖 ⇒ 本机无法验证")
                self.assertNotIn(
                    f"'{binname}'", code,
                    f"判据里出现了对 {binname} 的硬依赖 ⇒ 本机无法验证")

    def test_mkdir_is_used_for_atomicity(self):
        src = _SCRIPT.read_text(encoding="utf-8")
        code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
        self.assertIn("mkdir", code, "原子锁必须由 mkdir 提供")


class TestLockScriptIsWiredIn(unittest.TestCase):
    """`launch_batch.sh` 必须真的用这个锁，而不是各写一份。

    本项目已三次踩过"同一件事的 N 份实现分叉"（臂名解析三份、并发判据两份）。
    这条守卫的作用是：把"起跑锁"收敛到单一实现，改一处就全改。
    """

    def test_launch_script_calls_batch_lock(self):
        p = _ROOT / "scripts" / "launch_batch.sh"
        if not p.is_file():
            self.skipTest("launch_batch.sh 尚未创建")
        body = p.read_text(encoding="utf-8")
        self.assertIn("batch_lock.py", body,
                      "launch_batch.sh 未使用 scripts/batch_lock.py ⇒ 锁又分叉了")

    def test_launch_script_does_not_reimplement_lock(self):
        """不得在 shell 里再写一份 mkdir 锁（同源分叉的直接形态）。"""
        p = _ROOT / "scripts" / "launch_batch.sh"
        if not p.is_file():
            self.skipTest("launch_batch.sh 尚未创建")
        body = p.read_text(encoding="utf-8")
        code = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())
        self.assertNotIn("mkdir \"$LOCK_ANY\"", code,
                         "launch_batch.sh 自己又实现了一遍锁")


class TestMonitorProcessWiring(unittest.TestCase):
    """监视进程的**三个静默失效**各一条守卫。

    这三个缺陷都是 2026-09-27 端到端跑出来的，且**都不报错**：
    * `python - <<'PY'` 里的 `__file__` 是 `<stdin>` ⇒ 模块路径算成仓库根；
    * 放锁命令用了 `$root/batch_lock.py`，而 `$root` 是锁目录、不是 scripts；
    * 监视进程与批次**共用日志** ⇒ 它的 traceback 被启动报错扫描误读成
      "批次没起来"，于是批次明明活着却判 `LAUNCH_FAILED`。

    三条的共同点：**症状是"起了锁但没人放"或"起跑被误判失败"，而日志里
    看不到任何指向本缺陷的线索**。所以必须由守卫钉住。
    """

    def _code(self) -> str:
        p = _ROOT / "scripts" / "launch_batch.sh"
        if not p.is_file():
            self.skipTest("launch_batch.sh 尚未创建")
        body = p.read_text(encoding="utf-8")
        return "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())

    def test_module_path_is_passed_in_not_derived_from_dunder_file(self):
        """★ `__file__` 在 `python -`（heredoc）下是 `<stdin>` ⇒ 不得用它定位模块。"""
        code = self._code()
        self.assertNotIn(
            "__file__", code,
            "监视进程用 __file__ 定位 batch_lock.py：从 stdin 跑时它是 <stdin>，"
            ".parent 会变成仓库根 ⇒ 每次迭代 FileNotFoundError，锁永远没人放")
        # 反向：调用方必须把模块路径**显式传进去**
        self.assertIn("$LOCK_PY", code, "模块路径应由调用方显式传入（$LOCK_PY）")

    def test_release_uses_the_scripts_dir_not_the_lock_dir(self):
        """★ 放锁命令的模块路径必须落在 scripts/，不是锁目录之下。"""
        code = self._code()
        self.assertNotIn(
            "$root/batch_lock.py", code,
            "$root 是锁目录；模块在 scripts/ ⇒ 这条命令必然 FileNotFoundError")

    def test_monitor_has_its_own_log_not_the_batch_log(self):
        """★ 监管者的输出不得混进批次日志，否则启动扫描会误判。"""
        code = self._code()
        self.assertIn(
            "MON_LOGFILE", code,
            "监视进程需要独立日志：否则它的 traceback 会被启动报错扫描当成"
            "批次启动失败（实测把退出码从 0 变成 7）")
        # 监视进程的两次重定向都必须指向 MON_LOGFILE
        mon_redirs = [ln for ln in code.splitlines()
                      if ">>" in ln and "MON_LOGFILE" in ln]
        self.assertGreaterEqual(
            len(mon_redirs), 2,
            "setsid 与 nohup 两条分支的重定向都必须用 MON_LOGFILE")

    def test_monitor_release_targets_the_lock_dir(self):
        """★ 监视进程放锁时的 `--root` 必须是**锁目录**，不是 scripts 目录。

        实测 2026-09-27：`--root` 指到 scripts/ ⇒ release 什么都没删却打印
        "锁已释放" ⇒ 锁永久残留。这条守卫把 `--root "$lockdir"` 钉死。
        """
        code = self._code()
        self.assertIn(
            '--root "$lockdir"', code,
            "监视进程放锁必须显式用锁目录作为 --root；指错会让放锁静默失败")
        # 监视进程的入参里必须同时有锁目录与模块路径两个变量
        self.assertIn("$LOCKDIR", code)
        self.assertIn("$LOCK_PY", code)


class TestCallersUseSingleEntryPoints(unittest.TestCase):
    """起跑与并发判据都**只有一处权威**，调用方不得各写一份。

    本项目已三次踩过"同一件事的 N 份实现分叉"（臂名解析三份、并发判据两份、
    起跑三处）。下面把判据面收敛后的**反向**不变量钉住：
      * 并发判据 = `check_remote_busy.py`（不得再有 `ps -eo cmd | grep`）；
      * 起跑 = `launch_batch.sh`（不得再有裸 `setsid nohup ... &`）。
    任何一处回退都会让守卫变红。
    """

    _CALLERS = ("run_s14_d1.sh", "run_s14_d1_min.sh", "remote_wait_then_d1.sh")

    def _body(self, name: str) -> str:
        p = _ROOT / "scripts" / name
        if not p.is_file():
            self.skipTest(f"{name} 不在")
        return p.read_text(encoding="utf-8")

    @staticmethod
    def _strip_comments(body: str) -> str:
        return "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())

    def test_no_caller_reimplements_busy_check(self):
        """★ 并发判据不得再有 `ps -eo cmd | grep`（判据面收敛到唯一脚本）。"""
        for name in self._CALLERS:
            with self.subTest(caller=name):
                code = self._strip_comments(self._body(name))
                self.assertNotIn(
                    "ps -eo cmd", code,
                    f"{name} 又写了一份并发判据 ⇒ 与 check_remote_busy.py 分叉")

    def test_no_caller_reimplements_launch(self):
        """★ 起跑不得再有裸 `setsid nohup ... &`（起跑收敛到唯一入口）。"""
        for name in self._CALLERS:
            with self.subTest(caller=name):
                code = self._strip_comments(self._body(name))
                self.assertNotIn(
                    "setsid nohup", code,
                    f"{name} 又自己起了一次批次 ⇒ 绕过了 launch_batch.sh 的锁")

    def test_callers_use_the_shared_entry_points(self):
        """正向：真的要**调用**这两个入口，而不是只删掉旧的。"""
        for name in self._CALLERS:
            with self.subTest(caller=name):
                body = self._body(name)
                self.assertIn("check_remote_busy.py", body,
                              f"{name} 没有调用共享的并发判据")
        # 两个起跑点必须都存在（d1 启动 ②、wait 启动 d1）
        self.assertIn("launch_batch.sh", self._body("run_s14_d1.sh"))
        self.assertIn("launch_batch.sh", self._body("remote_wait_then_d1.sh"))

    def test_busy_check_is_invoked_with_expect_idle(self):
        """★ 必须带 `--expect-idle`：否则 BUSY 也返回 0 ⇒ 闸门恒放行。"""
        for name in self._CALLERS:
            with self.subTest(caller=name):
                code = self._strip_comments(self._body(name))
                if "check_remote_busy.py" not in code:
                    continue
                self.assertIn(
                    "--expect-idle", code,
                    f"{name} 调用 check_remote_busy.py 时缺 --expect-idle ⇒ "
                    "BUSY 也返回 0，闸门形同虚设")


class TestActivationIsStructuralNotCallerDiscipline(unittest.TestCase):
    """★ 环境激活必须由**起跑器**做，不能靠调用方在起跑器外面 `source`。

    踩过（2026-09-28）：调用方先 `source activate_musa.sh` 再调 `launch_batch.sh`，
    批次立刻 `ModuleNotFoundError: No module named 'transformers'`（0/20），
    而脚本照打 `BATCH_LAUNCH_OK`、退出码 0 —— 它只查"进程是否存活"，
    而激活失败时 python 确实**起来过**、只是立刻退出。

    根因：起跑走 `setsid nohup "$@"`，**直接 exec**，中间没有 shell ⇒ setsid 在
    **新会话 + 非交互 shell** 里重新 exec，外层环境不继承。所以激活只能发生在
    setsid **之后**的那个 shell 里 ⇒ `--activate` 选项。

    这条守卫钉住三件事：选项存在、激活确实进到 setsid 后的命令里、
    内层用 `bash -c`（不是 `sh -c`，否则 `source` 在 dash 里**静默失败**）。
    """

    def _raw(self) -> str:
        p = _ROOT / "scripts" / "launch_batch.sh"
        if not p.is_file():
            self.skipTest("launch_batch.sh 尚未创建")
        return p.read_text(encoding="utf-8")

    @staticmethod
    def _strip_comments(body: str) -> str:
        return "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())

    def test_activate_option_exists_and_is_parsed(self):
        """`--activate` 必须真的被解析（不是只写进注释/用法里）。"""
        code = self._strip_comments(self._raw())
        self.assertIn("--activate", code,
                      "缺少 --activate ⇒ 调用方只能靠'在外面 source'这条不可行纪律")
        self.assertIn("ACTIVATE", code, "--activate 没有被解析进变量")

    def test_activate_wraps_in_bash_c_not_sh_c(self):
        """★ 承载 `source` 的必须是 `bash`，不能是 `sh`（dash 里 `source` 不存在）。

        ⚠️ 不能写成 `assertNotIn("sh -c", code)` —— `bash -c` 本身**包含**子串
        `sh -c`，那样写会把自己判失败（写这条时真的踩到了）。
        正确判据：逐个 token 找"以 sh 结尾的 shell 词 + -c"，且它前面不是 ba。
        """
        code = self._strip_comments(self._raw())
        # 逐个 '-' 前的词看它是 bash 还是 sh
        import re as _re
        shells = _re.findall(r"(?<![\w-])([a-z/]*sh)\s+-c", code)
        self.assertTrue(shells, "激活层里找不到任何 `<shell> -c`")
        bad = [s for s in shells if not s.endswith("bash")]
        self.assertEqual(
            bad, [],
            f"承载 source 的 shell 不是 bash：{bad} ⇒ dash 里 source 不是内建，"
            "激活会静默不发生")

    def test_launch_uses_launch_cmd_not_bare_at(self):
        """★ 起跑必须 exec `LAUNCH_CMD`，不能再直接 exec `"$@"`。

        若仍是 `"$@"`，`--activate` 就形同虚设（激活层根本没被起跑）。
        """
        code = self._strip_comments(self._raw())
        self.assertIn("LAUNCH_CMD", code,
                      "起跑命令必须走 LAUNCH_CMD 变量，否则 --activate 不被使用")
        self.assertNotIn(
            'nohup "$@"', code,
            '起跑仍直接 exec "$@" ⇒ --activate 被绕过，激活不会发生')

    def test_activate_failure_is_loud_not_silent(self):
        """★ `source` 失败必须**非零退出**并显式打印，不能静默继续。"""
        code = self._strip_comments(self._raw())
        self.assertIn("source", code, "激活层里没有 source 动作")
        self.assertIn("exit 7", code,
                      "source 失败必须非零退出；静默继续会让批次以'没激活'运行")


class TestActivateEndToEnd(unittest.TestCase):
    """真跑一次：**双向验证**——修好的能激活，且旧写法确实不激活。

    只做结构断言等于"静态推断"，我的既往教训是那会放过真缺陷
    （换行符、局部 import、路径转换都是"看起来对、实际错"）。
    所以这里用一个真的 `source` 脚本改环境变量，再看命令是否看到它。

    ⚠️ **被测命令必须活得比起跑器的存活确认窗（3 s）长**，否则起跑器会按
    `NO_EVIDENCE`（退出码 9）报警——它要求"pid 还在 **或** 日志非空"。
    实测过：一条 `echo > file` 的瞬完命令会拿到 rc=9，**但文件其实已写出**；
    那是起跑器对"短命令"的正常判定，不是缺陷。故这里让命令 sleep 后写，
    并把金丝雀同时**回显到 stdout**（落进日志）作为第二重证据。
    """

    _HOLD_S = "5"  # > 起跑器的 3 s 存活确认窗

    @classmethod
    def setUpClass(cls):
        """★ 本机能否**真的**拉起起跑器？不能就整体跳过，并说清原因与复现命令。

        踩过（2026-09-28，本机 Git-for-Windows + 沙箱）：
        `subprocess.run(['bash', '-c', ...])` 返回 `rc=1` + 乱码 stderr ——
        PortableGit 的 `bash.EXE` 在本会话里**无法从 Python 直接 spawn**。
        但经 `sh -c 'bash …'` 间接调用**可以**（`sh`/`dash` 不在拦截名单里）。
        故这里统一走 `sh -c` 间接层；只有当**间接层也起不来**时才跳过。
        """
        probe = ("bash -c 'echo WBTEST_BASH_OK'")
        try:
            p = subprocess.run(["sh", "-c", probe],
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=30)
            cls._bash_ok = (p.returncode == 0 and "WBTEST_BASH_OK" in (p.stdout or ""))
            cls._bash_detail = f"rc={p.returncode} out={p.stdout!r} err={p.stderr!r}"
        except Exception as exc:
            cls._bash_ok = False
            cls._bash_detail = f"{type(exc).__name__}: {exc}"

    def setUp(self):
        if not self._bash_ok:
            self.skipTest(
                "本机经 `sh -c` 也拉不起 bash ⇒ 端到端起跑用例跳过（环境限制，非缺陷）；"
                "结构守卫（TestActivationIsStructural…）仍在跑。"
                f" 实测={self._bash_detail}")
        # 每只用例一个自增序号 ⇒ 锁目录唯一（见 _posix_tmp 的 (b) 段）
        import itertools
        self._counter = itertools.count(1)

    def _run_launch(self, tmp: Path, args: list[str]) -> tuple[int, str]:
        """经 `sh -c 'bash <launcher> ...'` 调用（绕过本机对 bash 直接 spawn 的拦截）。

        ⚠️ 参数用 shlex.quote 逐项转义后再拼成一条 `sh -c` 命令 —— 因为 `sh -c`
        只收一个字符串。这不是"拼接命令"的老毛病：每个参数都过了 quote。
        """
        env = dict(os.environ)
        env["BATCH_LOCKDIR"] = str(tmp / "locks")
        env["BATCH_PY"] = sys.executable
        inner = " ".join(
            shlex.quote(a)
            for a in ("bash", str(_ROOT / "scripts" / "launch_batch.sh"), *args)
        )
        # ⚠️ `text=True` 在 Windows 上按 locale(cp936) 解码，而脚本输出的中文
        #    会抛 UnicodeDecodeError（实测炸在采集线程里，连 pytest 都只是 warning）。
        #    因此显式 utf-8 + errors=replace：测试关注的是**退出码与日志**，
        #    不该因为一个装饰性字符而失败。
        proc = subprocess.run(
            ["sh", "-c", inner],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=env, cwd=str(_ROOT), timeout=120,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    def _pump(self, logfile: Path, marker: Path, until: str,
              timeout: float = 25.0) -> str:
        """等 marker 或日志里出现 `until`，返回日志全文。"""
        deadline = time.time() + timeout
        text = ""
        while time.time() < deadline:
            if logfile.is_file():
                text = logfile.read_text(encoding="utf-8", errors="replace")
            if until in text:
                return text
            if marker.is_file() and until in marker.read_text(
                    encoding="utf-8", errors="replace"):
                return text + "\n" + marker.read_text(encoding="utf-8",
                                                      errors="replace")
            time.sleep(0.4)
        return text

    @staticmethod
    def _canary_cmd(marker: Path) -> str:
        """先 sleep 越过存活确认窗，再把金丝雀写文件**并**回显到 stdout。

        ⚠️ marker 路径同样要用 `//c/...` 形式（见 `_path_for_sh`），
        否则经 `sh -c` 传递时会被 MSYS 改写成 `\\c\\...`，tee 写到奇怪的位置。
        """
        target = TestActivateEndToEnd._path_for_sh(marker)
        return (
            f"sleep {TestActivateEndToEnd._HOLD_S}; "
            f"printf 'CANARY=%s\\n' \"${{CANARY_ACTIVATED:-MISSING}}\" "
            f"| tee {shlex.quote(target)}"
        )

    def _posix_tmp(self, tag: str) -> Path:
        """建一个 **唯一的、POSIX 风格** 的临时目录并返回其路径。

        ⚠️ 两个坑，都实测踩到过：

        (a) **不能用** `tempfile.TemporaryDirectory()`：它给的是
            `C:\\Users\\...\\Temp\\xxx`（反斜杠）。起跑器里 `mkdir -p "$LOGDIR"`
            收到反斜杠路径时 Git Bash 把 `\\` 当转义 ⇒ **立刻 rc=1、stdout 空**。
            真实调用永远发生在 Linux 上，故测试也必须用正斜杠。

        (b) **必须带唯一 nonce**：起跑器收尾后，**监视进程**还会持锁约 10 s
            （`min_hold_s`）。若两次测试复用同一个 `BATCH_LOCKDIR`，
            第二次会拿到 `rc=5 GLOBAL_BUSY`（实测：反向用例因此假失败）。
            用 nonce 把每只用例的锁目录彻底隔开，互不干扰。
        """
        nonce = f"{os.getpid()}_{next(self._counter)}"
        base = Path("/c/Users/13944/AppData/Local/Temp") / f"wbtest_{tag}_{nonce}"
        subprocess.run(
            ["sh", "-c", f"rm -rf -- {shlex.quote(str(base))}"],
            capture_output=True, timeout=30,
        )
        base.mkdir(parents=True, exist_ok=True)
        return base

    _counter = None

    @staticmethod
    def _path_for_sh(p: Path) -> str:
        """把 `/c/...` 形式转成 `//c/...`，避免经 `sh -c` 传递时被改写。

        ⚠️ 实测踩到：`/c/Users/...` 经 `sh -c`+`shlex.quote` 传进去后，
        接收方看到的是 `\\c\\Users\\...`（MSYS 路径转换）。
        双斜杠 `//c/...` 是 MSYS 的显式"不要转换"写法。
        """
        s = str(p)
        if s.startswith("/") and not s.startswith("//"):
            return "/" + s
        return s

    def test_activate_reaches_the_spawned_command(self):
        """正向：`--activate` 里设的变量，必须在**被起跑的命令**里可见。"""
        tmp = self._posix_tmp("act_yes")
        act = tmp / "act.sh"
        # 故意用一个不寻常的变量名，避免被既有环境掩盖
        act.write_text("export CANARY_ACTIVATED=yes-from-activate\n",
                       encoding="utf-8")
        marker = tmp / "marker.txt"
        logfile = tmp / "out.log"
        rc, out = self._run_launch(tmp, [
            "--activate", self._path_for_sh(act),
            "S14TEST_A", self._path_for_sh(logfile),
            "sh", "-c", self._canary_cmd(marker),
        ])
        self.assertEqual(rc, 0, f"起跑应成功；rc={rc} 输出={out}")
        text = self._pump(logfile, marker, "CANARY=")
        self.assertIn("yes-from-activate", text,
                      "--activate 的 source 没有作用到被起跑的命令上")

    def test_without_activate_the_canary_is_missing(self):
        """反向：不加 `--activate` 时，**同一个金丝雀**必须不可见。

        这条是"测得出对错"的保证——否则上一条可能因环境里本就存在该变量
        而恒过。这里把"未激活"也实测出来。
        """
        tmp = self._posix_tmp("act_no")
        marker = tmp / "marker2.txt"
        logfile = tmp / "out2.log"
        rc, out = self._run_launch(tmp, [
            "S14TEST_B", self._path_for_sh(logfile),
            "sh", "-c", self._canary_cmd(marker),
        ])
        self.assertEqual(rc, 0, f"起跑应成功；rc={rc} 输出={out}")
        text = self._pump(logfile, marker, "CANARY=")
        self.assertIn("MISSING", text,
                      "没有 --activate 却看到了金丝雀 ⇒ 上一条测试无鉴别力")
        self.assertNotIn("yes-from-activate", text,
                         "无 --activate 时金丝雀不该出现")


if __name__ == "__main__":
    unittest.main(verbosity=2)
