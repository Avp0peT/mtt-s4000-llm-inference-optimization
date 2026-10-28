#!/usr/bin/env python3
"""S14B2 回拉补全——**走 exec 通道 + 远端分块 base64**。

为什么不用 `sftp.get` / `pull`（都是实测，不是推测）：
  * `sftp.get` 在批次满负载时对 ≥100 KB 的文件会**无限期挂住**：
    本地文件被创建但恒为 0 字节，13 分钟不返回、不报错。
    「重试」在这种形态下无效——每次重试只是新开一条连接去撞同一堵墙。
  * `pull` 走的是同一个 sftp 数据通道（远端 `tar czf` 后 `sftp.get` tar），
    没有证据表明它更快，且远端打包本身也要时间。

所以改用**已经证明会返回**的 `exec_command` 通道：
  * 远端把文件切成 N 块，逐块 `base64 -w0` 打到 stdout；
  * 每块的 base64 字符串用**哨兵行**包起来，便于在混有告警的输出里精确定位；
  * 本地解码并**逐块累积**，最后核对总字节数 == 远端 `stat` 大小。

分块而非一次全取，是因为**单条 exec 的输出越大越容易被拖慢/截断**；
64 KB 的块能把"一次要传多少"压到与已知能成的小文件同量级。

用法::

    python scripts/pull_s14b2_exec.py                  # 补不完整的目录
    python scripts/pull_s14b2_exec.py --all            # 强制全部重拉
    python scripts/pull_s14b2_exec.py --register       # 齐了自动生成登记行
    python scripts/pull_s14b2_exec.py --only <dir> ... # 只处理指定目录

安全：只写 `results/raw/S14B2-*` 下的目标文件；用一个**临时文件 + 原子替换**，
所以中途失败不会留下"半截的 meta.json"冒充完整产物。
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
RAW = ROOT / "results" / "raw"
MEM = ROOT / ".workbuddy" / "memory"
RROOT = "/root/ascend_probe_d1/results/raw"
TAG = "S14B2"

# 文件正常大小的**保守下界**：用来把 0 / 32768 这类截断判成不完整。
MIN_BYTES = {
    "config.snapshot.yaml": 500,
    "meta.json": 100_000,
    "requests.jsonl": 20_000,
    "telemetry.jsonl": 100_000,
}
FILES = list(MIN_BYTES)

CHUNK = 48 * 1024          # 每块原始字节数（base64 后约 64 KB）
LOCK = MEM / ".pull_s14b2_exec.lock"
STOP = MEM / ".pull_s14b2.stop"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    with open(MEM / "pull_s14b2_exec.log", "a", encoding="utf-8", newline="\n") as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")


def connect(tries: int = 8):
    """建立连接。复用 remote_run 的连接层（它自己也会重试）。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import remote_run as rr  # noqa: PLC0415

    return rr._connect(rr._load_credentials(), attempts=tries, backoff=3.0)


def exec_lines(client, cmd: str, timeout: float = 240) -> tuple[int, str, str]:
    """跑一条远端命令并取回文本输出。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    import remote_run as rr  # noqa: PLC0415

    return rr._exec_capture(client, cmd, timeout)


def remote_size(client, rd: str, f: str) -> int | None:
    """取远端原始字节数。取不到返回 None（**不是 0**——0 会被误当成"文件是空的"）。"""
    marker = "RSZ="
    for attempt in range(5):
        try:
            code, out, _ = exec_lines(
                client, f"printf '{marker}\\n'; stat -c %s {rd}/{f} 2>/dev/null", 120
            )
        except Exception as exc:  # noqa: BLE001
            log(f"    [{f}] stat 异常: {type(exc).__name__}")
            time.sleep(3)
            continue
        for line in out.splitlines():
            if line.strip().isdigit():
                return int(line.strip())
        log(f"    [{f}] stat 无数字输出（rc={code}），重试 {attempt + 1}")
        time.sleep(3)
    return None


def fetch_file(client, rd: str, f: str, dest: Path) -> bool:
    """分块 base64 取回一个文件；成功返回 True（并已原子替换落盘）。"""
    rsz = remote_size(client, rd, f)
    if rsz is None:
        log(f"    [{f}] ✗ 取不到远端大小 ⇒ 判为**不可判**（不是文件不存在）")
        return False
    if rsz < MIN_BYTES[f]:
        log(f"    [{f}] ✗ 远端原始仅 {rsz} B < 门槛 {MIN_BYTES[f]} B ⇒ 远端也没写全")
        return False

    nchunks = (rsz + CHUNK - 1) // CHUNK
    tmp = dest.with_suffix(dest.suffix + ".partial")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()

    total = 0
    for i in range(nchunks):
        off = i * CHUNK
        # 用 `dd` 精确切块，再 base64；哨兵行让输出可被无歧义地切开。
        # 注意 `skip=` 用字节偏移、`bs=1`（小文件够用；大文件也不慢到不可接受）。
        cmd = (
            f"echo B64BEGIN; dd if={rd}/{f} bs=1 skip={off} count={CHUNK} 2>/dev/null "
            f"| base64 -w0; echo; echo B64END"
        )
        ok = False
        for attempt in range(6):
            try:
                code, out, err = exec_lines(client, cmd, 300)
            except Exception as exc:  # noqa: BLE001
                log(f"    [{f}] chunk {i+1}/{nchunks} 异常: {type(exc).__name__}")
                time.sleep(4)
                continue
            body = None
            lines = out.splitlines()
            for j, line in enumerate(lines):
                if line.strip() == "B64BEGIN":
                    acc = []
                    for k in range(j + 1, len(lines)):
                        if lines[k].strip() == "B64END":
                            body = "".join(acc)
                            break
                        acc.append(lines[k].strip())
                    break
            if body is None:
                log(f"    [{f}] chunk {i+1}/{nchunks} 没找到哨兵（rc={code}），重试 {attempt+1}")
                time.sleep(4)
                continue
            try:
                raw = base64.b64decode(body, validate=False)
            except Exception as exc:  # noqa: BLE001
                log(f"    [{f}] chunk {i+1}/{nchunks} base64 解码失败: {exc}")
                time.sleep(4)
                continue
            expect = min(CHUNK, rsz - off)
            if len(raw) != expect:
                log(f"    [{f}] chunk {i+1}/{nchunks} 长度 {len(raw)} != 期望 {expect}，重试 {attempt+1}")
                time.sleep(4)
                continue
            with open(tmp, "ab") as fh:
                fh.write(raw)
            total += len(raw)
            ok = True
            break
        if not ok:
            log(f"    [{f}] ✗ chunk {i+1}/{nchunks} 6 次都没成 ⇒ 放弃该文件（临时文件保留，不冒充完整）")
            return False

    if total != rsz:
        log(f"    [{f}] ✗ 累积 {total} B != 远端 {rsz} B ⇒ 不落盘")
        return False
    os.replace(tmp, dest)          # 原子替换：不会出现"半截冒充完整"
    log(f"    [{f}] ✓ {total} B（{nchunks} 块 via exec+base64）")
    return True


def incomplete() -> list[str]:
    out: list[str] = []
    for d in sorted(p for p in RAW.glob(f"{TAG}-*") if p.is_dir()):
        for f, mb in MIN_BYTES.items():
            p = d / f
            if (p.stat().st_size if p.exists() else 0) < mb:
                out.append(d.name)
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    MEM.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text().strip())
        except Exception:  # noqa: BLE001
            pid = -1
        log(f"❌ 锁已存在 (pid={pid}) ⇒ 拒绝启动第二个写者")
        return 3
    LOCK.write_text(str(os.getpid()))

    try:
        dirs = args.only or (sorted(p.name for p in RAW.glob(f"{TAG}-*") if p.is_dir())
                             if args.all else incomplete())
        log(f"=== 目标 {len(dirs)} 个目录（chunk={CHUNK} B, lock={os.getpid()}）===")
        for b in dirs:
            log(f"  - {b}")

        for b in dirs:
            if STOP.exists():
                log(f"⏹ 检测到 {STOP} ⇒ 主动退出")
                STOP.unlink(missing_ok=True)
                break
            rd = f"{RROOT}/{b}"
            ld = RAW / b
            log(f"=== {b} ===")
            for f in FILES:
                if STOP.exists():
                    log(f"⏹ 检测到停止标记 ⇒ 退出")
                    STOP.unlink(missing_ok=True)
                    return 0
                mb = MIN_BYTES[f]
                p = ld / f
                sz = p.stat().st_size if p.exists() else 0
                if sz >= mb and not args.all:
                    log(f"    [{f}] 已有 {sz} B，跳过")
                    continue
                # 连接逐文件重建：连接层已被证明会抖，长连接更不可靠
                client = None
                for attempt in range(4):
                    try:
                        client = connect()
                        break
                    except Exception as exc:  # noqa: BLE001
                        log(f"    [{f}] 连接失败({attempt+1}): {type(exc).__name__}")
                        time.sleep(5)
                if client is None:
                    log(f"    [{f}] ✗ 连不上 ⇒ 跳过（下一轮再来；**不是**产物缺失）")
                    continue
                try:
                    fetch_file(client, rd, f, p)
                finally:
                    try:
                        client.close()
                    except Exception:  # noqa: BLE001
                        pass

        ok, bad = 0, []
        for d in sorted(p for p in RAW.glob(f"{TAG}-*") if p.is_dir()):
            good = all(
                (d / f).exists() and (d / f).stat().st_size >= mb
                for f, mb in MIN_BYTES.items()
            )
            if good:
                ok += 1
            else:
                bad.append(d.name)
        log(f"=== 结算：完整 {ok}/20；仍不完整 {len(bad)} 个 ===")
        for b in bad:
            log(f"    {b}")

        if args.register:
            if ok == 20 and not bad:
                log("=== 20/20 齐 ⇒ 生成登记行（**不自动 append**，交人工复核）===")
                r = subprocess.run(
                    [PY, str(ROOT / "scripts" / "emit_experiment_rows.py"),
                     "--tag", TAG],
                    cwd=str(ROOT), capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                outp = ROOT / "results" / "processed" / f"emit_{TAG}.txt"
                outp.parent.mkdir(parents=True, exist_ok=True)
                outp.write_text((r.stdout or "") + (r.stderr or ""),
                                encoding="utf-8", newline="\n")
                log(f"    已写到 {outp}（rc={r.returncode}）")
            else:
                log(f"=== 未达 20/20（{ok}/20）⇒ **不生成登记行**（编号按 20 个预留）===")
        return 0 if ok == 20 else 1
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
