"""分块回拉远端大文件。

为什么需要它（2026-09-28 §64 实测）：
  * 远端 S14B3 只有**裸文件**（无 `.gz` 边车），单个 `meta.json` 3.9–9.0 MB；
  * 裸 `sftp.get` 一个 4.3 MB 的文件在**当前链路**上 **>300 s 未完成**（rc=124），
    本地只留 0 字节 ⇒ **整文件拉不可行**；
  * 而 1.6 KB 的小文件**秒成** ⇒ 链路是"能通但吞吐低"。

结论：把大文件切成小块，**每块独立重试**，最后拼接 + 用**远端 sha256/size**校验。
每块独立成一次 SFTP 请求，单块超时不会毁掉整文件进度，天然可续传。

设计要点：
  * 块大小默认 512 KiB（4.3 MB ≈ 9 块）；单块超时 120 s。
  * 校验：拼接后 **字节数 == 远端 stat 尺寸**，且（可选）sha256 一致。
    校验不过 ⇒ **保留 .partial 不覆盖目标**，返回非零。
  * 本地路径一律 **Windows 原生**（`C:/...`），规避 `MSYS_NO_PATHCONV` 把
    `/c/...` 写成 `C:\\c\\...`（§63）。

用法：
    python scripts/pull_chunked.py <remote_file> <local_file> [--chunk 524288] [--sha]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
REMOTE = ROOT / "scripts" / "remote_run.py"


def _run(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, timeout=timeout, cwd=str(ROOT),
    )


def remote_size(remote: str) -> int | None:
    r = _run([PY, str(REMOTE), "exec", f"stat -c %s '{remote}' 2>/dev/null"], 120)
    for ln in (r.stdout or b"").decode("utf-8", "replace").replace("\r", "").splitlines():
        if ln.strip().isdigit():
            return int(ln.strip())
    return None


def remote_sha(remote: str) -> str | None:
    r = _run([PY, str(REMOTE), "exec", f"sha256sum '{remote}' 2>/dev/null"], 180)
    for ln in (r.stdout or b"").decode("utf-8", "replace").replace("\r", "").splitlines():
        parts = ln.split()
        if parts and len(parts[0]) == 64:
            return parts[0]
    return None


def pull_chunk(remote: str, skip: int, length: int, dest: Path, tries: int) -> bool:
    """用 `exec` + base64 取 [skip, skip+length) 这一段。

    走 exec 而不是 sftp：sftp.get 没有 offset 参数，而 `dd`/`tail`+base64
    可以精确取任意区间，且每段是**一次独立的远端命令**。
    """
    import base64

    for i in range(tries):
        cmd = (
            f"dd if='{remote}' bs=1 skip={skip} count={length} 2>/dev/null | base64 -w0"
        )
        try:
            r = _run([PY, str(REMOTE), "exec", cmd], 300)
        except subprocess.TimeoutExpired:
            print(f"      chunk@{skip} try{i+1}: 超时", flush=True)
            continue
        out = (r.stdout or b"").decode("ascii", "replace").replace("\r", "").replace("\n", "")
        if not out:
            print(f"      chunk@{skip} try{i+1}: 空输出 (rc={r.returncode})", flush=True)
            continue
        try:
            blob = base64.b64decode(out, validate=True)
        except Exception as e:  # noqa: BLE001
            print(f"      chunk@{skip} try{i+1}: base64 解码失败 {type(e).__name__}", flush=True)
            continue
        if len(blob) == 0:
            print(f"      chunk@{skip} try{i+1}: 0 字节", flush=True)
            continue
        with open(dest, "ab") as fh:
            fh.write(blob)
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("remote")
    ap.add_argument("local")
    ap.add_argument("--chunk", type=int, default=512 * 1024)
    ap.add_argument("--sha", action="store_true", help="额外做 sha256 校验（较慢）")
    ap.add_argument("--tries", type=int, default=4)
    args = ap.parse_args()

    dest = Path(args.local)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")

    total = remote_size(args.remote)
    if total is None:
        print(f"✗ 远端取不到尺寸：{args.remote}")
        return 2
    if dest.exists() and dest.stat().st_size == total:
        print(f"✓ 已是完整文件（{total} B），跳过")
        return 0

    if partial.exists():
        partial.unlink()
    done = 0
    while done < total:
        n = min(args.chunk, total - done)
        if not pull_chunk(args.remote, done, n, partial, args.tries):
            print(f"✗ 块 @{done} 失败；保留 {partial.name}（{partial.stat().st_size if partial.exists() else 0} B）")
            return 3
        done += n
        print(f"    {done}/{total} B ({done*100//total}%)", flush=True)

    got = partial.stat().st_size
    if got != total:
        print(f"✗ 拼接后 {got} B ≠ 远端 {total} B；目标文件未写")
        return 4
    if args.sha:
        want = remote_sha(args.remote)
        h = hashlib.sha256()
        with open(partial, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        have = h.hexdigest()
        if want and have != want:
            print(f"✗ sha256 不一致 remote={want[:12]} local={have[:12]}；目标文件未写")
            return 5
    os.replace(partial, dest)
    print(f"✓ {dest.name} {total} B 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
