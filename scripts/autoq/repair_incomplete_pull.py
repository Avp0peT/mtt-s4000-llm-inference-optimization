#!/usr/bin/env python3
"""修补"本地目录存在但内容不完整"的回拉缺口。

为什么需要它
------------
`pull_incremental.py` 的差集是**按目录名**算的：

    have = {d.name for d in local_root.glob(f"{tag}-*") if d.is_dir()}
    missing = [n for n in remote if n not in have]

于是"目录在、但里面是半截产物"**永远不会进入待取清单**——
它报 `待取 0 个`，返回码却是 1（完整性核对列 6 个 `✗`）。
`local_finish.py` 的循环因此空转 40 次、每次都停在 15/20。

实测（2026-09-28，S14B2）：6 个目录本地不完整而**远端完整**
（e.g. `ramp_static-r1` 远端 meta.json 1241426 B，本地 0 B）。
⇒ 成因是**中断的回拉留下截断文件**，不是数据丢失。
⇒ 修法是**对这些目录绕过名称差集、直接重传一遍**。

纪律
----
* 只读远端；本地只写**判定为不完整**的目录（其余一律不碰）。
* 逐个目录重传，每个目录限时；失败只报告并继续下一个（不中断整批）。
* 不改产物内容、不改标签、不动 `results/raw` 之外的任何东西。

用法::

    python scripts/autoq/repair_incomplete_pull.py --tag S14B2
    python scripts/autoq/repair_incomplete_pull.py --tag S14B2 --expect 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_REMOTE = _ROOT / "scripts" / "remote_run.py"
_REMOTE_ROOT = "/root/ascend_probe_d1/results/raw"

#: paramiko 只装在托管 venv 里；本脚本常被别的解释器调用 ⇒ 显式挑一个。
_VENV_PY = (Path.home() / ".workbuddy" / "binaries" / "python" / "envs"
            / "default" / "Scripts" / "python.exe")

_REQUIRED = ("meta.json", "requests.jsonl", "telemetry.jsonl", "config.snapshot.yaml")


def _pick_interpreter() -> str:
    for p in (str(_VENV_PY), sys.executable):
        if not p or not Path(p).exists():
            continue
        try:
            r = subprocess.run([p, "-c", "import paramiko"],
                               capture_output=True, timeout=30)
        except Exception:                                    # noqa: BLE001
            continue
        if r.returncode == 0:
            return p
    return sys.executable


def is_incomplete(d: Path) -> list[str]:
    """返回该目录的问题清单（空 = 完整）。与 pull_incremental._verify 同一口径。"""
    bad: list[str] = []
    for f in _REQUIRED:
        p = d / f
        if not p.exists() or p.stat().st_size == 0:
            bad.append(f"缺/空 {f}")
    meta = d / "meta.json"
    if meta.exists() and meta.stat().st_size > 0:
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            if not m.get("data_label"):
                bad.append("meta.data_label 为空")
        except Exception as e:                               # noqa: BLE001
            bad.append(f"meta.json 不可解析：{type(e).__name__}")
    return bad


def _pull_one(name: str, py: str, timeout: int) -> tuple[int, str]:
    """直接调 remote_run.py pull（绕过名称差集）。返回 (rc, 输出)。"""
    local = _ROOT / "results" / "raw" / name
    # ⚠️ 远端绝对路径必须用 '//root/...' 形态：本机 MSYS 会把 '/root/...'
    #    改写成 'C:/.../PortableGit/.../root/...'（remote_run.py 会拒绝，rc=5）。
    cmd = [py, str(_REMOTE), "pull", f"//{_REMOTE_ROOT.lstrip('/')}/{name}", str(local)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, cwd=str(_ROOT))
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"超时（{timeout}s）——传输停滞"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=480,
                    help="单个目录的重传限时（秒）；停滞时不要无限等")
    args = ap.parse_args(argv)

    py = _pick_interpreter()
    raw = _ROOT / "results" / "raw"
    dirs = sorted(d for d in raw.glob(f"{args.tag}-*") if d.is_dir())

    incomplete = [(d.name, is_incomplete(d)) for d in dirs]
    bad = [(n, why) for n, why in incomplete if why]

    print(f"[scan] {len(dirs)} 个 {args.tag} 目录；不完整 {len(bad)} 个")
    for n, why in bad:
        print(f"   ✗ {n}: {'；'.join(why)}")
    if not bad:
        print("[scan] 无需修补")
        return 0

    fixed, still = [], []
    for n, why in bad:
        print(f"\n[pull] 重传 {n} …")
        rc, out = _pull_one(n, py, args.timeout)
        tail = out.strip().splitlines()[-3:] if out.strip() else []
        for t in tail:
            print(f"   {t}")
        left = is_incomplete(raw / n)
        if rc == 0 and not left:
            print(f"   ✓ {n} 已完整")
            fixed.append(n)
        else:
            print(f"   ✗ {n} 仍不完整（rc={rc}）：{'；'.join(left) or 'rc 非零'}")
            still.append(n)

    print(f"\n=== 修补结果：修复 {len(fixed)} / 仍缺 {len(still)} / 共 {len(bad)} ===")
    if args.expect is not None:
        ok = sum(1 for d in dirs if d.is_dir() and not is_incomplete(d))
        print(f"[count] 本地完整目录 {ok}/{args.expect}")
    if still:
        print("  仍缺：" + ", ".join(still))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
