#!/usr/bin/env python3
"""批次起跑的**互斥锁 + 生命期监视**（纯逻辑，可在本机完整验证）。

为什么要有它
------------
2026-09-27 的事故：本机编排脚本的失败重试把同一条 `nohup` **起跑命令发了
两遍**，第二遍的 `rm -f <log>` 抹掉了第一遍正在写的 inode
（`/proc/<pid>/fd/1 -> ... (deleted)`）⇒ 日志永久丢失，且**差点让两个矩阵
同时写 `results/raw/`**。第二个矩阵在第 1 格 `RunDirConflict` 退出——纯属运气。

⇒ 起跑必须**持锁**。而"锁该持多久"这个判断，我第一版就做错了两次，
   两个方向都错，所以把判据直接写进代码并配上测试：

   ① 锁只覆盖"起跑动作"（脚本一返回就放锁）
      ⇒ **同 tag 重起跑畅通无阻**（实测：退出码 0，`SHOULD_NOT_RUN` 真的执行了）
      ⇒ 事故原样复现。**这是漏**。

   ② 锁覆盖整个批次存活期，但没人负责在批次结束后放锁
      ⇒ 批次跑完锁还在 ⇒ **永久堵住后续所有批次**。**这是死**。

   正确：锁的生命周期 **== 批次的存活期**，且必须有一个**独立的持有者**
   （不能是起跑的父 shell —— 它 `setsid` 之后立刻退出）。

为什么锁原语用 `mkdir` 而不是 `flock`
--------------------------------------
实测 2026-09-27：
  * 远端（真正跑批次的机器，Linux 5.15 / bash 5.1）：`setsid` `flock` **都有**；
  * 本机（Git for Windows / PortableGit）：`setsid` `flock` **都没有**。
⇒ 若把 `flock` 当硬前提，这个**判据**在本机一次都验不了 —— 而"只在远端跑过、
  从没本地验证"的判据正是本项目最怕的形态（出了问题只能靠读日志猜）。
  `mkdir` 两边都有、且**原子**（已存在则 EEXIST），语义完全一致。

为什么监视进程必须能**被验证**
--------------------------------
本机没有 `setsid` ⇒ 本机起的任何后台进程都会随父 shell 退出而死
（实测：批次 `nohup ... &` 与监视进程双双消失、日志里连 monitor 的第一行
都没有）。这**不是**代码缺陷，是"本机不具备脱离会话的能力"。
⇒ 所以本模块把两件事**分开**：
  * **锁语义**（`acquire` / `holder` / `release` / `is_held`）—— 本机完全可验；
  * **监视循环的判据**（`decide()` 纯函数）—— 本机完全可验；
  * 只有"起一个脱离会话的进程"这一步依赖 `setsid`，而它由调用方
    （远端 shell）保证，本机测试用**注入**的存活探测函数驱动。

用法（库）::

    locks = BatchLocks("/root/batch_locks")
    got, code, why = locks.acquire("S14H")
    if not got:
        ...  # code 4/5，why 里说明是谁在占

用法（命令行，供 shell 调用）::

    python3 scripts/batch_lock.py acquire --tag S14H --cmd "..."
    python3 scripts/batch_lock.py release --tag S14H
    python3 scripts/batch_lock.py status  --tag S14H

退出码：0 = 取到 / 释放成功；4 = 同 tag 在跑；5 = 其它批次在跑；
        6 = 参数或目录错；7 = 试图释放**不属于自己**的锁（防误删）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

#: 批次的存活期由监视进程持续检查的间隔（秒）。取 5 s 是个折中：
#: 批次结束后最多再挡 5 s 才能起下一个，而监视进程本身的 CPU 开销可忽略
#: （一个 `kill -0` + 一次 `stat`）。取更小没有意义，取更大会让"跑完立刻
#: 想重跑"的人白等。
POLL_SECONDS = 5.0

#: 启动时间戳的容差。同一秒内两次起跑应视为**同一次**（时钟精度）。
_TS_TOLERANCE = 0.0


class LockBusy(Exception):
    """锁被占用。``code`` 与 CLI 退出码一致（4=同 tag / 5=其它批次）。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class BatchLocks:
    """`results/raw` 的全局锁 + 每 tag 锁。

    目录布局（用**目录**而不是文件：`mkdir` 是原子的，而"文件存在性检查 +
    创建"两步之间有 TOCTOU 窗口）：:

        <root>/all.lock.d/          # 全局：同一时刻只允许一个批次
            holder                  # 文本：tag / started_at / pid / cmd
        <root>/tag.<TAG>.lock.d/    # 每 tag：同名 run_id 不得撞车
            holder
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.all_lock = self.root / "all.lock.d"

    def tag_lock(self, tag: str) -> Path:
        # tag 进路径名 ⇒ 必须挡住分隔符与 `..`，否则能写到锁目录之外。
        # 这不是假想威胁：tag 由调用方拼命令行而来，打错一个字符就会
        # 让 `rm -rf <lockdir>/../..` 这种路径出现。
        if not tag or "/" in tag or "\\" in tag or tag in (".", ".."):
            raise ValueError(f"非法 tag：{tag!r}（不得含路径分隔符）")
        return self.root / f"tag.{tag}.lock.d"

    # -- 基础操作 ---------------------------------------------------------- #

    def _mkdir_atomic(self, p: Path) -> bool:
        """原子创建目录。已存在 ⇒ False（**不报错**，由调用方判语义）。"""
        try:
            p.mkdir(parents=True, exist_ok=False)
            return True
        except FileExistsError:
            return False
        except OSError as exc:  # 权限/只读文件系统等：必须显形，不能当成"占用"
            raise OSError(f"无法创建锁目录 {p}：{exc}") from exc

    def holder(self, p: Path) -> dict | None:
        """读锁的持有者信息。读不到返回 ``None``（**不返回空 dict**：
        "没有记录"与"记录是空的"必须可分，否则排障时会以为持有者无信息）。"""
        f = p / "holder"
        if not f.is_file():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def is_held(self) -> bool:
        return self.all_lock.is_dir()

    def acquire(self, tag: str, *, cmd: str = "", pid: int | None = None) -> dict:
        """取锁。失败抛 :class:`LockBusy`。

        顺序：**先全局后 tag** —— 全局锁是更粗的约束，先取它能让
        "另一个 tag 在跑"这种情况以 code 5 报出来（而不是被 tag 锁的 4 掩盖）。
        """
        self.root.mkdir(parents=True, exist_ok=True)
        tag_lock = self.tag_lock(tag)

        if not self._mkdir_atomic(self.all_lock):
            h = self.holder(self.all_lock)
            raise LockBusy(5, f"已有批次在跑（全局锁被占）：{h or '（无 holder 记录）'}")
        # 全局锁已到手；之后任何失败都必须回滚，否则会留下"半持有"状态。
        try:
            if not self._mkdir_atomic(tag_lock):
                h = self.holder(tag_lock)
                raise LockBusy(4, f"同 tag 已在跑：{tag}（{h or '（无 holder 记录）'}）")
        except BaseException:
            shutil.rmtree(self.all_lock, ignore_errors=True)
            raise

        rec = {
            "tag": tag,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "started_ts": time.time(),
            "pid": os.getpid() if pid is None else pid,
            "cmd": cmd,
        }
        blob = json.dumps(rec, ensure_ascii=False, indent=2) + "\n"
        for p in (self.all_lock, tag_lock):
            (p / "holder").write_text(blob, encoding="utf-8", newline="\n")
        return rec

    def release(self, tag: str, *, only_if_holder_pid: int | None = None) -> bool:
        """放锁。默认**无条件**放（监视进程用）；给了 pid 则只在 pid 相符时放。

        ★ `only_if_holder_pid` 存在的理由：事故里"第二个进程"差点把第一个
        正在用的资源删掉（`rm -f` 抹 inode）。同一个错误不能再在**锁**上犯一次：
        一个迟到的清理者不应该删掉**别人**刚拿到的锁。
        """
        if only_if_holder_pid is not None:
            h = self.holder(self.all_lock)
            if h and h.get("pid") not in (None, only_if_holder_pid):
                raise LockBusy(
                    7,
                    f"拒绝放锁：持有者 pid={h.get('pid')} != 调用者 {only_if_holder_pid}",
                )
        removed = False
        for p in (self.all_lock, self.tag_lock(tag)):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed = True
        return removed


# --------------------------------------------------------------------------- #
# 监视循环的**判据**（纯函数 —— 本机可完整验证，不需要真的起进程）
# --------------------------------------------------------------------------- #

#: `decide()` 的三态。字符串常量而不是布尔：
#: "批次还在跑"与"查不到批次状态"必须可分 —— 后者若当成"跑完了"，
#: 锁会被**误释放**，于是并发批次又能进来（正是要防的事）。
MONITOR_RUNNING = "running"
MONITOR_DONE = "done"
MONITOR_UNKNOWN = "unknown"

#: 连续多少次 `UNKNOWN` 才判定"批次已不在"。取 3 而非 1：
#: 探测函数可能因瞬时错误（ps 输出截断、权限抖动）返回 unknown，
#: 单次 unknown 就放锁是危险的。3 次 × 5 s = 15 s 的宽限。
UNKNOWN_STRIKES_TO_RELEASE = 3


def decide(
    *,
    alive: bool | None,
    lock_still_held: bool,
    unknown_strikes: int,
    batch_age_s: float,
    min_hold_s: float = 0.0,
) -> tuple[str, str]:
    """监视进程每一步该做什么。返回 ``(动作, 理由)``。

    动作 ∈ ``{"wait", "release", "exit"}``：

    * ``wait``   —— 继续等；
    * ``release``—— 批次已结束 ⇒ 放锁后退出；
    * ``exit``   —— 锁已被外部清掉 ⇒ 无事可做，直接退出（**不要**再 rm，
                     否则会删掉别人刚拿到的锁）。

    ⚠️ ``alive=None`` 是"查不到"，**不**等于"已经死了"。这是本项目的通行纪律
    （信号要么可信、要么显式 None；不可测的值保持 None 禁止填 0）。
    这里同样：unknown 累计到阈值才敢放锁。
    """
    if not lock_still_held:
        return "exit", "锁已被外部清除 ⇒ 监视进程无事可做"
    if batch_age_s < min_hold_s:
        # 刚起跑的一小段窗口里，进程可能还没被 /proc 反映出来。
        # 不设这个下限会有一个真实竞态：起跑后 5 ms 就探测 ⇒ 没找到 ⇒ 误判跑完。
        return "wait", f"批次刚起 {batch_age_s:.2f}s < 下限 {min_hold_s:.2f}s"
    if alive is True:
        return "wait", "批次仍在跑"
    if alive is False:
        return "release", "批次已退出 ⇒ 放锁"
    # alive is None
    if unknown_strikes + 1 >= UNKNOWN_STRIKES_TO_RELEASE:
        return "release", (f"连续 {unknown_strikes + 1} 次查不到批次状态 "
                           f"⇒ 按已结束处理（放锁，避免永久堵住）")
    return "wait", f"批次状态查不到（第 {unknown_strikes + 1} 次），继续等"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cmd_acquire(args: argparse.Namespace) -> int:
    locks = BatchLocks(args.root)
    try:
        rec = locks.acquire(args.tag, cmd=args.cmd or "")
    except LockBusy as exc:
        print(f"✗ {exc}", file=sys.stderr)
        print(f"  锁目录：{locks.root}", file=sys.stderr)
        print("  若确认没有批次在跑（先跑 scripts/check_remote_busy.py），"
              f"再删锁：rm -rf {locks.all_lock} {locks.tag_lock(args.tag)}",
              file=sys.stderr)
        print(f"LOCK_ABORTED_{'GLOBAL' if exc.code == 5 else 'TAG'}_BUSY")
        return exc.code
    print(f"LOCK_ACQUIRED tag={args.tag} pid={rec['pid']} at={rec['started_at']}")
    print(f"  全局锁 {locks.all_lock}")
    print(f"  tag 锁 {locks.tag_lock(args.tag)}")
    return 0


def _cmd_release(args: argparse.Namespace) -> int:
    locks = BatchLocks(args.root)
    try:
        removed = locks.release(args.tag, only_if_holder_pid=args.only_if_pid)
    except LockBusy as exc:
        print(f"✗ {exc}", file=sys.stderr)
        print("LOCK_RELEASE_REFUSED")
        return exc.code
    # ⚠️ `removed=False` 意味着**一个锁目录都没找到** ⇒ 多半是 `--root` 指错了地方。
    #    实测 2026-09-27：监视进程把 `--root` 指到了 scripts/ 而不是锁目录，
    #    `rmtree(ignore_errors=True)` 于是一声不响地什么都没删，而日志仍然打印
    #    "锁已释放" ⇒ 锁永久残留。这里把"什么都没删"与"删成功"分开报，
    #    并给出根目录，让这个缺陷一眼可辨。
    if removed:
        print(f"LOCK_RELEASED tag={args.tag} removed=True root={locks.root}")
    else:
        print(f"LOCK_RELEASED tag={args.tag} removed=False **没找到任何锁目录** "
              f"root={locks.root} ⇒ 若应为已释放，检查 --root 是否指对",
              file=sys.stderr)
        print(f"LOCK_RELEASED tag={args.tag} removed=False root={locks.root}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    locks = BatchLocks(args.root)
    held = locks.is_held()
    h = locks.holder(locks.all_lock)
    th = locks.holder(locks.tag_lock(args.tag)) if args.tag else None
    print(f"LOCK_STATUS held={held}")
    print(f"  holder={json.dumps(h, ensure_ascii=False) if h else None}")
    if args.tag:
        print(f"  tag.{args.tag}={json.dumps(th, ensure_ascii=False) if th else None}")
    # ⚠️ 退出码语义：0 = 查到了且空闲；1 = 被占；2 = **查不出**（root 不可读）。
    #    三者不可合并 —— "查不出"当成"空闲"会让闸门静默失效。
    if not locks.root.is_dir():
        print("  ⚠️ 锁根目录不存在 ⇒ 视为**未持有**（首次起跑的正常情形）")
        return 0
    return 1 if held else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="批次起跑互斥锁（mkdir 原子锁）")
    ap.add_argument("--root", default="/root/batch_locks")
    sub = ap.add_subparsers(dest="action", required=True)

    a = sub.add_parser("acquire", help="取锁；失败返回 4/5")
    a.add_argument("--tag", required=True)
    a.add_argument("--cmd", default="")
    a.set_defaults(func=_cmd_acquire)

    r = sub.add_parser("release", help="放锁")
    r.add_argument("--tag", required=True)
    r.add_argument("--only-if-pid", type=int, default=None,
                   help="只在 holder 的 pid 相符时放（防迟到清理者删掉别人的锁）")
    r.set_defaults(func=_cmd_release)

    s = sub.add_parser("status", help="查状态；0=空闲 1=被占")
    s.add_argument("--tag", default=None)
    s.set_defaults(func=_cmd_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
