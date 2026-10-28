"""陈旧批次锁清理的用例（`scripts/clear_stale_batch_lock.py`）。

为什么这些判据必须本地可验：它们是**删除操作**的准入条件。
在无人值守场景里，判据错了的代价是双向的 ——
删多了（把正在跑的批次的锁清掉）会让两个批次同时写 `results/raw/`；
删少了（陈旧锁永不清）会让**之后所有批次都起不来**，且不报错。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clear_stale_batch_lock as C  # noqa: E402

#: 一个几乎不可能存在的 pid。用**实测过的**判据（见下）确认它"确实不存在"，
#: 不靠"这个数很大所以肯定没有"这种想象。
DEAD_PID = 999999


def _pid_is_really_dead(pid: int) -> bool:
    entry = C.LockEntry(Path("/nonexistent"), "t", pid, 0.0)
    return entry.pid_alive() is False


class TestPidAlive(unittest.TestCase):
    def test_own_pid_is_alive(self):
        e = C.LockEntry(Path("/x"), "t", os.getpid(), 0.0)
        self.assertTrue(e.pid_alive())

    def test_dead_pid_is_false_not_none(self):
        """★ 平台差异：Windows 上不存在的 pid 抛 `OSError(winerror=87)`，
        只捕 `ProcessLookupError` 会让这里变成 `None` ⇒ 清理永不触发。"""
        self.assertTrue(_pid_is_really_dead(DEAD_PID),
                        "探测不到'已死'的 pid ⇒ 清理逻辑在本地等于没验")

    def test_missing_pid_is_none(self):
        e = C.LockEntry(Path("/x"), "t", None, 0.0)
        self.assertIsNone(e.pid_alive(), "没有 pid 记录是'无法判定'，不是'已死'")


class TestScanAndDecide(unittest.TestCase):
    def _mk_lock(self, root: Path, name: str, *, tag: str, pid: int | None,
                 age_s: float) -> Path:
        """按**真实布局**造锁：记录写在 `holder` 里。

        ⚠️ 第一版这里造的是 `{"tag":...}.json` —— 与 `batch_lock.py` 的实际布局
        **不一样**（它把记录写在 `holder`，见 `batch_lock.py:169`）。
        假样本让一个错解析器通过了测试：真实扫描恒得 `pid=None` ⇒ 清理永不触发，
        而报告看起来像"没有陈旧锁"。⇒ 夹具必须逐字照抄真实产物。
        """
        d = root / name
        d.mkdir(parents=True)
        rec = {"tag": tag, "pid": pid, "started_ts": time.time() - age_s,
               "started_at": "x", "cmd": "x"}
        (d / "holder").write_text(json.dumps(rec), encoding="utf-8", newline="\n")
        return d

    def test_scan_reads_the_real_layout(self):
        """★ 用**真实** `batch_lock.py` 取一次锁，验证扫描器读得到 pid/tag。

        这条是"解析器与真产物一致"的可执行证据：夹具再怎么造都不如让**真实现**
        写一份出来。
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bl = C._lock_module()
            locks = bl.BatchLocks(str(root))
            rec = locks.acquire("REALTAG", cmd="demo")
            entries = C.scan_locks(root)
            self.assertTrue(entries, "真实锁目录没被扫到")
            all_lock = [e for e in entries if "all.lock.d" in e.path.name]
            self.assertEqual(len(all_lock), 1)
            self.assertEqual(all_lock[0].tag, "REALTAG")
            self.assertEqual(all_lock[0].pid, rec["pid"])
            self.assertIsNotNone(all_lock[0].started_ts)
            locks.release("REALTAG")

    def test_scan_reads_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._mk_lock(root, "tag.A.lock.d", tag="A", pid=os.getpid(), age_s=100)
            entries = C.scan_locks(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].tag, "A")
            self.assertEqual(entries[0].pid, os.getpid())

    def test_scan_ignores_non_lock_dirs_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notalock").mkdir()
            (root / "stray.txt").write_text("x", encoding="utf-8")
            self.assertEqual(C.scan_locks(root), [])

    def test_old_lock_with_dead_pid_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._mk_lock(root, "tag.A.lock.d", tag="A", pid=DEAD_PID, age_s=3600)
            stale, keep = C.decide(C.scan_locks(root), now=time.time(), min_age_s=600)
            self.assertEqual([e.tag for e in stale], ["A"])
            self.assertEqual(keep, [])

    def test_old_lock_with_live_pid_is_kept(self):
        """★ 年龄**不**能压过存活性：批次跑 10 小时也还在跑。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._mk_lock(root, "tag.A.lock.d", tag="A", pid=os.getpid(), age_s=36000)
            stale, keep = C.decide(C.scan_locks(root), now=time.time(), min_age_s=600)
            self.assertEqual(stale, [])
            self.assertEqual(len(keep), 1)

    def test_fresh_lock_with_dead_pid_is_kept(self):
        """★ 年龄下限防的是"另一个批次刚拿到锁、pid 还没写全"的竞态。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._mk_lock(root, "tag.A.lock.d", tag="A", pid=DEAD_PID, age_s=5)
            stale, keep = C.decide(C.scan_locks(root), now=time.time(), min_age_s=600)
            self.assertEqual(stale, [])
            self.assertEqual(len(keep), 1)

    def test_lock_without_pid_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._mk_lock(root, "tag.A.lock.d", tag="A", pid=None, age_s=36000)
            stale, keep = C.decide(C.scan_locks(root), now=time.time(), min_age_s=600)
            self.assertEqual(stale, [])
            self.assertEqual(len(keep), 1)


class TestCliSafety(unittest.TestCase):
    """CLI 层：默认**只报告不删**，且只删 `*.lock.d`。"""

    def _run(self, args) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(_ROOT / "scripts"
                                                   / "clear_stale_batch_lock.py"), *args],
                              capture_output=True, text=True, encoding="utf-8")

    def _mk(self, root: Path, *, pid: int | None, age_s: float) -> Path:
        d = root / "tag.A.lock.d"
        d.mkdir(parents=True)
        (d / "holder").write_text(json.dumps(
            {"tag": "A", "pid": pid, "started_ts": time.time() - age_s}),
            encoding="utf-8", newline="\n")
        return d

    def test_default_does_not_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = self._mk(root, pid=DEAD_PID, age_s=3600)
            p = self._run(["--root", str(root)])
            self.assertEqual(p.returncode, 1, "有陈旧锁但未 --apply ⇒ 必须非零且不删")
            self.assertTrue(d.exists(), "默认绝不能不删而返回成功")
            self.assertIn("不删", p.stdout)

    def test_apply_deletes_only_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = self._mk(root, pid=DEAD_PID, age_s=3600)
            keep = root / "tag.B.lock.d"
            keep.mkdir()
            (keep / "holder").write_text(json.dumps(
                {"tag": "B", "pid": os.getpid(), "started_ts": time.time() - 3600}),
                encoding="utf-8", newline="\n")
            p = self._run(["--root", str(root), "--apply", "--min-age-minutes", "10"])
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertFalse(stale.exists(), "陈旧锁应被删除")
            self.assertTrue(keep.exists(), "★ 活着的锁绝不能被删")
            self.assertIn("已删除陈旧锁", p.stdout)

    def test_missing_root_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._run(["--root", str(Path(tmp) / "nope")])
            self.assertEqual(p.returncode, 0)
            self.assertIn("无锁可清", p.stdout)

    def test_no_locks_is_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self._run(["--root", tmp])
            self.assertEqual(p.returncode, 0)
            self.assertIn("没有锁", p.stdout)


if __name__ == "__main__":
    unittest.main()
