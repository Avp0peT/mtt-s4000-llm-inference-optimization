#!/usr/bin/env python3
"""清理**陈旧**批次锁（`batch_lock.py` 的补位安全阀）。

为什么需要它
------------
`launch_batch.sh` 的锁有一个**刻意的**生命周期设计：锁由**监视进程**在批次结束后释放
（不能由起跑的父 shell 放 —— 它 `setsid` 后立刻退出；也不能不放 —— 那会永久堵住）。
于是有一个失效模式：**监视进程自己死了**（被 OOM 杀、被整组 kill、机器重启），
锁就永远留着，之后**所有**批次都被 rc=4/5 挡在门外。

在无人值守场景下这个代价最大：不是"跑错"，而是"什么都不跑"，而且不报错。
⇒ 起跑前先做一次**有条件的**清理。

删除必须同时满足四条（缺一条都不删）
------------------------------------
1. 只删**锁根目录下**匹配 `*.lock.d` 的目录（不做任何其它 glob，不递归删别的东西）；
2. 锁里记录的 `pid` **已不存在**（`os.kill(pid, 0)` 抛 `ProcessLookupError`）；
3. 锁的年龄 > `--min-age-minutes`（默认 10）—— 避免把**另一个正在起跑**的批次
   刚拿到的锁清掉（那种情况下 pid 可能还没写全）；
4. 调用方另行确认过"远端没有基准在跑"（见 `scripts/check_remote_busy.py`）。
   ⚠️ 本脚本**不**替调用方做这件事：它只能看到锁文件，看不到进程表全貌。

默认**只报告不删**（`--apply` 才真的删）。要删的东西**逐个打印**，
不做静默批量删除。

用法::

    python3 scripts/clear_stale_batch_lock.py                      # 只报告
    python3 scripts/clear_stale_batch_lock.py --apply --min-age-minutes 10

退出码：0 = 没有陈旧锁（或已清理）；1 = 有陈旧锁但未清理（未给 --apply）；
        2 = 参数/根目录错。
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_LOCK_ROOT = "/root/batch_locks"

__all__ = ["LockEntry", "scan_locks", "decide", "main"]


class LockEntry:
    def __init__(self, path: Path, tag: str | None, pid: int | None,
                 started_ts: float | None) -> None:
        self.path = path
        self.tag = tag
        self.pid = pid
        self.started_ts = started_ts

    def age_s(self, now: float) -> float | None:
        return None if self.started_ts is None else max(0.0, now - self.started_ts)

    def pid_alive(self) -> bool | None:
        """`True`/`False`；`None` = **无法判定**（没有 pid，或环境不支持该探测）。

        ⚠️ `None` 不是 `False`：没有 pid 记录不代表进程不在，
        只代表我们不知道 —— 那种锁一律**不删**（交给人工看一眼）。

        ⚠️ **平台差异（实测，不是推断）**：不存在的 pid 在 Linux 上抛
        `ProcessLookupError`，在 Windows 上抛 `OSError(winerror=87, errno=22)`
        （"参数错误"）。只捕 `ProcessLookupError` 会让"清除陈旧锁"在 Windows 上
        **永不触发** —— 症状是"跑了但一条都没清"，看起来像"没有陈旧锁"。
        """
        if not self.pid:
            return None
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # 进程存在但不是我们的（同机不同用户）⇒ 视为活着
            return True
        except OSError as exc:
            if getattr(exc, "winerror", None) == 87 or getattr(exc, "errno", None) == errno.ESRCH:
                return False
            return None
        return True

    def describe(self, now: float) -> dict[str, Any]:
        age = self.age_s(now)
        return {
            "path": str(self.path),
            "tag": self.tag,
            "pid": self.pid,
            "age_minutes": None if age is None else round(age / 60.0, 2),
            "pid_alive": self.pid_alive(),
        }


def _lock_module() -> Any:
    """拿到 `batch_lock` 模块 —— **复用它的 `holder()` 解析**，不自己再写一份。

    ⚠️ 我第一版自己写了解析，并把 `holder` 当成"标记文件"跳过 —— 而**真实布局**
    是记录就写在 `holder` 里（`batch_lock.py:169`）。于是扫描结果恒为
    `tag=None, pid=None` ⇒ 清理**永不触发**，而报告看起来像"没有陈旧锁"。
    教训：解析外部产物前先看**真实产物**，别按想象写。
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import batch_lock  # noqa: PLC0415
    return batch_lock


def scan_locks(root: Path) -> list[LockEntry]:
    """扫描 `*.lock.d` 目录并读出元数据。**只读**，不改任何东西。

    解析走 `batch_lock.BatchLocks.holder()`（唯一实现），因此与 `acquire`/`release`
    对同一份文件的读法**必然一致**。
    """
    root = Path(root)
    out: list[LockEntry] = []
    if not root.is_dir():
        return out
    bl = _lock_module()
    locks = bl.BatchLocks(str(root))
    for d in sorted(root.glob("*.lock.d")):
        if not d.is_dir():
            continue
        rec = locks.holder(d) or {}
        tag = rec.get("tag")
        pid: int | None = None
        try:
            pid = int(rec["pid"]) if rec.get("pid") else None
        except (TypeError, ValueError):
            pid = None
        started: float | None = None
        try:
            started = float(rec["started_ts"]) if rec.get("started_ts") is not None else None
        except (TypeError, ValueError):
            started = None
        out.append(LockEntry(d, tag, pid, started))
    return out


def decide(entries: Iterable[LockEntry], *, now: float,
           min_age_s: float) -> tuple[list[LockEntry], list[LockEntry]]:
    """返回 `(可删, 保留)`。四条条件在 `LockEntry` 层面能查的三条都在这。"""
    stale: list[LockEntry] = []
    keep: list[LockEntry] = []
    for e in entries:
        age = e.age_s(now)
        if e.pid_alive() is False and age is not None and age > min_age_s:
            stale.append(e)
        else:
            keep.append(e)
    return stale, keep


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_LOCK_ROOT)
    ap.add_argument("--min-age-minutes", type=float, default=10.0)
    ap.add_argument("--apply", action="store_true", help="真的删除（默认只报告）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"[stale-lock] 锁根目录不存在：{root} ⇒ 无锁可清（不是错误）")
        return 0
    now = time.time()
    entries = scan_locks(root)
    if not entries:
        print("[stale-lock] 没有锁 ⇒ 无陈旧锁")
        return 0
    stale, keep = decide(entries, now=now, min_age_s=args.min_age_minutes * 60.0)
    report = {
        "root": str(root),
        "n_total": len(entries),
        "stale": [e.describe(now) for e in stale],
        "kept": [e.describe(now) for e in keep],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else
          "\n".join(
              f"[stale-lock] {'STALE' if x in stale else 'keep '} {x.describe(now)}"
              for x in list(stale) + list(keep)))
    if not stale:
        return 0
    if not args.apply:
        print("[stale-lock] 有陈旧锁但未给 --apply ⇒ **不删**（默认只报告）")
        return 1
    for e in stale:
        # 只删这一个 `*.lock.d` 目录本身；不递归到锁根之外，不做通配批量删。
        shutil.rmtree(e.path)
        print(f"[stale-lock] 已删除陈旧锁：{e.path}（pid={e.pid} 已不存在）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
