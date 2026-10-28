#!/usr/bin/env python3
"""第 14 步收尾：**本地**侧串联动作（回拉 → 分析 → 登记 → 报告）。

为什么需要它
------------
远端队列（`/root/autoq/run_auto_queue.sh`）负责**产生数据**；本脚本负责
**把数据变成证据**。两边分开的理由：远端实例重启会把系统盘回滚到快照，
所以"边跑边取"必须由本地主动做，且**每次动作都要能单独重跑**（幂等）。

三个纪律
--------
1. **回拉幂等**：`pull_incremental.py` 已"跳过已有的"，本脚本只是**重试**
   到成功（SSH 链路抖动是已知的，重试**不会**造成重复或覆盖）。
2. **分析只读**：`analyze_step14.py` 不改产物。**不允许**它去改标签或删数据。
3. **登记用生成器**：`emit_experiment_rows.py` 生成 Markdown 行，本脚本
   只负责**把它接在正确位置**并核对行数——不手写数字。

用法::

    python scripts/autoq/local_finish.py --tag S14B2 --expect 20 --analyze
    python scripts/autoq/local_finish.py --tag S14B3 --expect 20 --analyze
    python scripts/autoq/local_finish.py --tag S14B2 --expect 20 --wait-remote 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PY = sys.executable
_REMOTE = _ROOT / "scripts" / "remote_run.py"


def _run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", cwd=str(_ROOT), timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def remote_raw_count(tag: str, retries: int = 6) -> int | None:
    """远端本批产物数（**只看有非空 meta.json 的目录**）。取不到返回 None。"""
    probe = (
        "python3 -c \"import glob,os;"
        f"ds=[d for d in glob.glob('/root/ascend_probe_d1/results/raw/{tag}-*') "
        "if os.path.isdir(d) and os.path.getsize(os.path.join(d,'meta.json'))>0 "
        "if os.path.exists(os.path.join(d,'meta.json'))];print(len(ds))\""
    )
    for _ in range(retries):
        rc, out = _run([_PY, str(_REMOTE), "exec", probe], timeout=120)
        if rc == 0:
            for line in reversed(out.strip().splitlines()):
                if line.strip().isdigit():
                    return int(line.strip())
        time.sleep(5)
    return None


def local_count(tag: str) -> int:
    n = 0
    for d in sorted((_ROOT / "results" / "raw").glob(f"{tag}-*")):
        if d.is_dir() and (d / "meta.json").exists() and (d / "meta.json").stat().st_size > 0:
            n += 1
    return n


def pull_until_complete(tag: str, expect: int, max_attempts: int = 40) -> int:
    """重试回拉直到本地数 == expect（或 attempt 用尽）。返回最终本地数。"""
    for attempt in range(1, max_attempts + 1):
        rc, out = _run([_PY, str(_ROOT / "scripts" / "pull_incremental.py"),
                        "--tag", tag, "--expect", str(expect)], timeout=1800)
        n = local_count(tag)
        print(f"[pull] attempt={attempt} rc={rc} local={n}/{expect}")
        if rc == 0 and n >= expect:
            return n
        # 回拉失败或未满：等一会再来（增量语义 ⇒ 安全）
        time.sleep(10)
    return local_count(tag)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--expect", type=int, required=True)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--wait-remote", type=int, default=1,
                    help="1=先等远端产物达标再回拉；0=直接回拉当前已有的")
    args = ap.parse_args(argv)

    out_dir = _ROOT / "results" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.wait_remote:
        print(f"[wait] 等远端 {args.tag} 产物达到 {args.expect} …")
        deadline = time.time() + 12 * 3600
        while time.time() < deadline:
            n = remote_raw_count(args.tag)
            print(f"[wait] 远端 {args.tag} = {n}")
            if n is not None and n >= args.expect:
                break
            time.sleep(180)

    print(f"[pull] 开始回拉 {args.tag}（期望 {args.expect}）")
    n = pull_until_complete(args.tag, args.expect)
    status = {
        "tag": args.tag,
        "expect": args.expect,
        "local_count": n,
        "complete": n >= args.expect,
    }
    print(f"[pull] 最终 local={n}/{args.expect}")

    if args.analyze:
        rc, out = _run([_PY, str(_ROOT / "scripts" / "analyze_step14.py"),
                        "--tag", args.tag,
                        "--high-start-s", "120", "--high-end-s", "210",
                        "--json", str(out_dir / f"step14_{args.tag}.json")],
                       timeout=1800)
        print(f"[analyze] rc={rc}")
        print(out[-4000:])
        status["analyze_rc"] = rc

    (out_dir / f"finish_{args.tag}.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"[done] 状态写入 {out_dir / f'finish_{args.tag}.json'}")
    return 0 if status["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
