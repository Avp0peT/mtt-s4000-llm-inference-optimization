"""批量核对本地 ↔ 远端 sha256：**一条**远端命令拿回所有哈希，本地比对。

存在理由：`push_files.sh` 是逐文件 2 次往返（取远端 sha + put）。在本项目这条
高 RTT/高抖动链路上，128 个文件 ≈ 18 分钟，而且任何一次抖动都会拖长。
本工具把「查远端现状」压成 **1 次往返**，只对**真正不同**的文件调用 push_files.sh。

用法：
    python scripts/batch_sha_diff.py --remote-root /root/ascend_probe_d1 \
        --include py,sh,yaml src benchmark scripts configs/mthreads/exec_opt

输出：
    - 差异清单（本地有、远端缺 / sha 不同 / 远端多出）
    - 可直接复制的 `push_files.sh ... <差异文件...>` 命令

⚠️ 只做**只读**比对，不推送（推送仍由 push_files.sh 负责，保持单一职责）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY_VENV = Path("C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe")
REMOTE_RUN = REPO / "scripts" / "remote_run.py"


def local_files(roots: list[str], exts: set[str]) -> list[str]:
    """展开成仓库相对路径（正斜杠）。"""
    out: list[str] = []
    for r in roots:
        p = REPO / r
        if p.is_file():
            if p.suffix.lstrip(".") in exts:
                out.append(r)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if (f.is_file() and f.suffix.lstrip(".") in exts
                        and "__pycache__" not in f.parts):
                    out.append(f.relative_to(REPO).as_posix())
        else:
            print(f"⚠️ 跳过（不存在）：{r}", file=sys.stderr)
    return out


def local_sha(rel: str) -> str:
    h = hashlib.sha256()
    with open(REPO / rel, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_shas(root: str, rels: list[str], chunk: int = 200) -> dict[str, str]:
    """一次取回远端 sha（分块防命令行过长）。

    ⚠️ **必须设 `MSYS_NO_PATHCONV=1`**：本机是 Git Bash，它会把参数里形如
    `/root/...` 的**远端路径**改写成 `C:/.../PortableGit/versions/x/root/...`
    再交给 Windows 原生 `python.exe` ⇒ 远端 `cd` 直接失败、返回 0 个 sha
    （踩过：结果表现为"所有文件都缺失"，与"远端真的没有"无法区分）。
    ⚠️ 本地路径不受影响：本文件里本地一律用 `Path`（Windows 绝对路径）。
    """
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    got: dict[str, str] = {}
    for i in range(0, len(rels), chunk):
        part = rels[i:i + chunk]
        quoted = " ".join(f"'{r}'" for r in part)
        cmd = f"cd {root} && sha256sum {quoted} 2>/dev/null"
        proc = subprocess.run(
            [str(PY_VENV), str(REMOTE_RUN), "exec", cmd],
            capture_output=True, timeout=600, env=env,
        )
        text = ""
        for enc in ("utf-8", "gbk"):
            try:
                text = proc.stdout.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = proc.stdout.decode("utf-8", errors="replace")
        for line in text.splitlines():
            bits = line.split()
            if len(bits) >= 2 and len(bits[0]) == 64:
                got[bits[-1].lstrip("./")] = bits[0]
    return got


def normalize_remote_root(root: str) -> str:
    """把被 Git Bash 改写过的远端路径**还原**成 POSIX 绝对路径。

    ⚠️ 本机是 Git Bash：命令行参数里形如 `/root/ascend_probe_d1` 的**远端路径**
    会被改写成 `C:/Users/.../PortableGit/versions/1.2.0/root/ascend_probe_d1`
    再交给 Windows 原生 `python.exe` ⇒ 远端 `cd` 失败、返回 0 个 sha，
    而症状是「所有文件都缺失」——与「远端真的是空的」**长得一模一样**（踩过）。
    这里按已知的 Git 安装前缀做还原；识别不出就原样返回（并让上层报错）。
    """
    if not root.startswith("/") and "PortableGit" in root.replace("\\", "/"):
        # 形如 <gitroot>/root/ascend_probe_d1 ⇒ 取第一个 "root/" 之后的部分
        p = root.replace("\\", "/")
        idx = p.find("/root/")
        if idx != -1:
            return p[idx:]
        idx = p.find("root/")
        if idx != -1:
            return "/" + p[idx:]
    return root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote-root", required=True)
    ap.add_argument("--include", default="py,sh,yaml")
    ap.add_argument("--emit-paths", action="store_true",
                    help="只输出**待推送的路径清单**（每行一个，= 远端缺失 + sha 不同），"
                         "供编排脚本消费；不打印人类报告。")
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()

    a.remote_root = normalize_remote_root(a.remote_root)
    if not a.remote_root.startswith("/"):
        print(f"⚠️ remote-root 不像 POSIX 绝对路径：{a.remote_root!r}\n"
              "   （若被 Git Bash 改写，远端 cd 会失败并表现为「全部缺失」）",
              file=sys.stderr)
    # `--emit-paths` 时**只**吐路径（供编排层消费）⇒ 人类报告全部静音。
    # 为什么不让编排层去解析人类报告：报告里的 `sha 不同` 行带
    # `local=<sha> remote=<sha>` 后缀，而"取缩进行"的解析器会把整行当路径
    # ⇒ `push_files.sh` 判「跳过（不存在）」⇒ **静默不推**（2026-09-30 实测）。
    #
    # ⚠️ 机器可读输出**必须是 LF**：Windows 上 stdout 是文本模式，会把 `\n`
    #    翻成 `\r\n`；消费方若是 shell（`$(...)` 展开）只会按空白切分，
    #    于是每个路径尾部带一个 `\r` ⇒ 判「不存在」⇒ 又变成**静默不推**。
    #    （Python 侧 `str.strip()` 恰好会剥掉 `\r`，所以只坑 shell 调用方——
    #     正是"只在一条路径上正确"的形状。）
    if a.emit_paths:
        try:
            sys.stdout.reconfigure(newline="\n")
        except (AttributeError, ValueError):  # pragma: no cover - 老解释器/被重定向
            pass
    say = (lambda *a_, **k_: None) if a.emit_paths else print
    say(f"远端根：{a.remote_root}")
    if not a.remote_root.startswith("/"):
        print(f"⚠️ remote-root 不像 POSIX 绝对路径：{a.remote_root!r}\n"
              "   （若被 Git Bash 改写，远端 cd 会失败并表现为「全部缺失」）",
              file=sys.stderr)

    exts = {e.strip().lstrip(".") for e in a.include.split(",") if e.strip()}
    rels = local_files(a.paths, exts)
    say(f"本地命中 {len(rels)} 个文件（扩展名 {sorted(exts)}）")

    rshas = remote_shas(a.remote_root, rels)
    say(f"远端返回 {len(rshas)} 个 sha")

    missing, diff, same = [], [], []
    for r in rels:
        ls = local_sha(r)
        rs = rshas.get(r)
        if rs is None:
            missing.append(r)
        elif rs == ls:
            same.append(r)
        else:
            diff.append(r)

    say(f"\n一致(skip)={len(same)}  远端缺失={len(missing)}  sha不同={len(diff)}")
    todo = missing + diff
    if a.emit_paths:
        # 机器可读：每行一个路径，**不带任何后缀**。
        for r in todo:
            print(r)
        return 1 if todo else 0
    if missing:
        print("\n--- 远端缺失 ---")
        for r in missing:
            print(f"  {r}")
    if diff:
        print("\n--- sha 不同 ---")
        for r in diff:
            print(f"  {r}  local={local_sha(r)[:12]} remote={rshas[r][:12]}")

    if todo:
        print("\n=== 建议命令（只推差异）===")
        print(f"bash scripts/push_files.sh --include {a.include} {a.remote_root} \\")
        for r in todo:
            print(f"  {r} \\")
        print("  # end")
        return 1
    print("\n✓ 全部一致，无需推送")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
