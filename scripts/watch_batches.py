#!/usr/bin/env python3
"""双批次看护：批次2 回拉补全 + 批次3 进度/到点自动分析。

设计约束（都是踩过的坑）：
  * **回拉只能有一个写者**。本脚本用锁文件保证；已有写者时**只报告不启动**。
  * 完整性判据用**尺寸门槛**，不用"文件存在"（`meta.json` 的 0/32768 都是截断）。
  * 远端控制面在负载下会抖（banner/auth timeout）⇒ 每个远端调用**重试到拿到真输出**；
    控制面失败 **≠** 远端状态变化。
  * 阶段转换（批次3 收尾）**只看 S14B3 的目录与 label**，不依赖历史 PASS。

用法：
    python scripts/watch_batches.py            # 一次检查（适合放进每小时看护）
    python scripts/watch_batches.py --daemon   # 常驻，每 900 s 一轮
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = r"C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
RAW = ROOT / "results" / "raw"
MEM = ROOT / ".workbuddy" / "memory"
LOG = MEM / "watch_batches.log"
LOCK = MEM / ".pull_s14b2.lock"

# 文件正常大小的**保守下界**（用来把 0 / 32768 这类截断判成不完整）
MIN_BYTES = {
    "config.snapshot.yaml": 500,
    "meta.json": 100_000,
    "requests.jsonl": 20_000,
    "telemetry.jsonl": 100_000,
}
FILES = list(MIN_BYTES)

TAG2 = "S14B2"
TAG3 = "S14B3"
B3_PLANNED = 20

#: 控制面往返判据（2026-09-28 §54）。
#: 正常 `pwd` 约 100–300 ms；满载实测 **18 213 ms**（慢 60~180×）。
#: 取 2000 ms 作闸门：既容正常的网络抖动，又能在"控制面被拖死"时拦住回拉。
RTT_OK_MS = 2000
#: 单次 RTT 探测的硬超时（秒）。超过即视为"控制面不可用"。
RTT_TIMEOUT_S = 60


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    MEM.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def remote(script: str, marker: str = "NOW=", tries: int = 6, timeout: int = 300) -> str | None:
    """跑远端命令；**重试到拿到带 marker 的真输出**。控制面抖动不算远端状态变化。"""
    last = ""
    for i in range(tries):
        try:
            r = subprocess.run(
                [PY, str(ROOT / "scripts" / "remote_run.py"), "exec", script],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=timeout, cwd=str(ROOT),
            )
            out = ((r.stdout or "") + (r.stderr or "")).strip()
            if marker in out:
                return out
            last = out
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(4)
    log(f"  ⚠️ 远端调用重试 {tries} 次未拿到 '{marker}'，最后输出: {last[:160]}")
    return None


def _ps_text() -> str:
    """`ps -ef` 输出用 errors='replace' 读。⚠️ 本机 bash 的进程命令行里可能出现
    非 UTF-8 字节（emoji 之类），默认 utf-8 严格解码会**在 reader 线程里抛
    UnicodeDecodeError**、且异常发生在子线程 ⇒ 主流程只看到"结果为空"，
    表现为"写者明明在跑却说没在跑"。（实测踩到过一次。）"""
    return subprocess.run(
        ["ps", "-ef"], capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    ).stdout or ""


# --------------------------------------------------------------------------- 批次2
def local_status(tag: str) -> tuple[int, list[str]]:
    """按尺寸门槛统计完整度。返回 (完整数, 不完整目录名列表)。"""
    dirs = sorted(p for p in RAW.glob(f"{tag}-*") if p.is_dir())
    bad: list[str] = []
    for d in dirs:
        for f, mb in MIN_BYTES.items():
            p = d / f
            sz = p.stat().st_size if p.exists() else 0
            if sz < mb:
                bad.append(d.name)
                break
    return len(dirs) - len(bad), bad


def pull_writer_alive() -> int | None:
    if not LOCK.exists():
        return None
    try:
        pid = int(LOCK.read_text().strip())
    except Exception:  # noqa: BLE001
        return None
    for line in _ps_text().splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[1] == str(pid) and "pull_s14b2_robust" in line:
            return pid
    return None


def start_pull(dry_run: bool = False) -> bool:
    """启动回拉写者。已在跑 ⇒ **不启动第二个**（并发写会互相删文件）。

    ⚠️ `dry_run=True` 时**只打印将要执行的命令，不真的启动**。
    为什么需要它：验证"互斥逻辑拦不拦"这件事，**必须**在不产生副作用的前提下做——
    第一次跑时确实没有写者，于是"验证"本身就把写者启动了（实测踩到：12:16 的一次
    测试跑出了一个整夜重试循环，见 `2026-09-28.md` §42）。这类失误是**静默的**：
    日志只多一行"启动回拉写者"，看起来像正常操作。
    """
    pid = pull_writer_alive()
    if pid:
        log(f"  回拉写者已在运行 (pid={pid})，**不启动第二个**")
        return False
    cmd = ["bash", "scripts/pull_s14b2_robust.sh", "--register"]
    if dry_run:
        log(f"  [dry-run] 本应启动: {' '.join(cmd)}（**未真正启动**）")
        return False
    log("  启动回拉写者（唯一写者）…")
    subprocess.Popen(
        cmd,
        cwd=str(ROOT), stdout=open(MEM / "pull_s14b2.log", "a", encoding="utf-8"),
        stderr=subprocess.STDOUT, env={**__import__("os").environ, "MAXTRY": "18"},
    )
    return True


def remote_b2_identity(tag: str) -> str | None:
    """远端核验批次2：目录数 + 每个目录的 data_label 分布。"""
    s = (
        f"cd /root/ascend_probe_d1/results/raw; "
        f"echo 'NOW='$(date '+%H:%M:%S'); "
        f"echo 'B2DIRS='$(ls -d {tag}-* 2>/dev/null | wc -l); "
        f"for d in {tag}-*; do "
        f"  lab=$(python3 -c \"import json,sys;print(json.load(open('$d/meta.json'))['summary'].get('data_label','?'))\" 2>/dev/null || echo ERR); "
        f"  echo \"  $d $lab\"; done"
    )
    return remote(s)


# --------------------------------------------------------------------------- 批次3
def b3_progress() -> tuple[int, list[str]] | None:
    s = (
        "echo 'NOW='$(date '+%H:%M:%S'); "
        f"echo 'B3DIRS='$(ls -d /root/ascend_probe_d1/results/raw/{TAG3}-* 2>/dev/null | wc -l); "
        f"ls -d /root/ascend_probe_d1/results/raw/{TAG3}-* 2>/dev/null | sed 's#.*/##'"
    )
    out = remote(s)
    if out is None:
        return None
    dirs = [ln.strip() for ln in out.splitlines() if ln.startswith(TAG3)]
    return len(dirs), dirs


def b3_queue_alive() -> bool:
    out = remote("echo 'NOW='$(date '+%H:%M:%S'); ps -eo pid,etime,cmd | grep run_auto_queue | grep -v grep")
    return out is not None and "run_auto_queue" in out


def b3_all_measured() -> tuple[bool, str]:
    """批次3 全 20 且全部 MEASURED 才算收尾（标签即判据，不看历史）。"""
    s = (
        "echo 'NOW='$(date '+%H:%M:%S'); "
        f"cd /root/ascend_probe_d1/results/raw; n=0; ok=0; "
        f"for d in {TAG3}-*; do n=$((n+1)); "
        f"lab=$(python3 -c \"import json;print(json.load(open('$d/meta.json'))['summary'].get('data_label','?'))\" 2>/dev/null || echo ERR); "
        f"[ \"$lab\" = MEASURED ] && ok=$((ok+1)); echo \"  $d $lab\"; done; "
        f"echo \"B3N=$n B3OK=$ok\""
    )
    out = remote(s)
    if out is None:
        return False, "(远端不可达)"
    tail = [ln for ln in out.splitlines() if "B3N=" in ln]
    if not tail:
        return False, "(没拿到 B3N)"
    body = out
    n = int(tail[0].split("B3N=")[1].split()[0])
    ok = int(tail[0].split("B3OK=")[1].split()[0])
    return (n >= B3_PLANNED and ok == n), f"{ok}/{n} MEASURED\n{body}"


# --------------------------------------------------------------------------- 轮次
def control_plane_rtt_ms() -> int | None:
    """**控制面往返延迟**（ms）——回拉可行性的**唯一前置判据**。

    2026-09-28 §54 定级：回拉挂起的根因**不是带宽、不是文件大小**，
    而是**SSH 控制面被 CPU/调度争用拖死**。实测（`LOAD≈10.25`）：

    =========  ==========================  ========  ==========  ========
    级别       载荷                        rc        用时        输出
    =========  ==========================  ========  ==========  ========
    L1         ``pwd``（纯控制面，无数据）  0         18 213 ms   7 B
    L2         4 KB 读                     124(超时)  61 433 ms   0 B
    =========  ==========================  ========  ==========  ========

    ``pwd`` 是最纯的控制面往返（无数据、无磁盘），正常 ~0.1–0.3 s，
    被拖到 **18.2 s（慢 60~180×）** ⇒ 任何附带数据的操作都必然挂。

    ⚠️ **为什么不能直接测"2.6 MB `get`"**：那个判据太钝，
    会把"控制面慢"与"带宽慢"混成一件事，于是**即使控制面已恢复也看不出**，
    反之亦然。**先量 RTT，再谈大数据**。

    :return: 往返毫秒数；远端不可达返回 ``None``。
    """
    import subprocess as _sp
    import time as _t
    t0 = _t.monotonic()
    try:
        p = _sp.run(
            [PY, str(ROOT / "scripts" / "remote_run.py"), "exec", "pwd"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", cwd=str(ROOT), timeout=RTT_TIMEOUT_S,
        )
    except _sp.TimeoutExpired:
        return RTT_TIMEOUT_S * 1000
    ms = int((_t.monotonic() - t0) * 1000)
    if p.returncode != 0:
        return None
    return ms


def pull_ready() -> tuple[bool, str]:
    """回拉前置判据：**控制面须已恢复**。

    ⭐ 2026-09-28 §54 升级：原来是"批次结束就测 2.6 MB `get`"，
    这会把两种完全不同的病因混在一起。改为**先量 `pwd` 的往返**：
    RTT 回到 ``RTT_OK_MS`` 以内才允许开始批量回拉。
    """
    ms = control_plane_rtt_ms()
    if ms is None:
        return False, "远端不可达"
    if ms <= RTT_OK_MS:
        return True, f"控制面 RTT={ms} ms（≤{RTT_OK_MS}）⇒ 允许回拉"
    return False, (f"控制面 RTT={ms} ms（>{RTT_OK_MS}）⇒ **负载未降，禁止开始回拉**"
                   f"（每次重试都要先付 {ms/1000:.1f}s 握手成本，多半到不了数据阶段）")


def one_round(dry_run: bool = False) -> None:
    log("=" * 72)
    # ⭐ §54：**控制面 RTT 放最前面**。
    # 理由：满载时每个远端调用要几十秒，若把 RTT 探测放在最后，
    # 前面的慢调用会把整轮拖到被 SIGTERM，于是**最关键的判据反而永远测不到**
    # （实测 14:46 那轮就是这样被截断的）。
    rtt_ms = control_plane_rtt_ms()
    log(f"控制面: RTT={rtt_ms} ms" if rtt_ms is not None else "控制面: ⚠️ 不可达")

    # --- 批次 3 ---
    prog = b3_progress()
    if prog is None:
        log("批次3: ⚠️ 远端不可达，本轮跳过（**不等于**批次停了）")
    else:
        n, dirs = prog
        alive = b3_queue_alive()
        log(f"批次3: 远端 {n}/{B3_PLANNED} 目录；队列进程{'存活' if alive else '**不在**'}")
        log(f"        最近 3 个: {', '.join(dirs[-3:]) if dirs else '(无)'}")
        if n >= B3_PLANNED:
            done, detail = b3_all_measured()
            if done:
                log("批次3: ★ 20/20 全部 MEASURED ⇒ 可做覆盖率/成因分解（**不得**归因 cost 更差）")
                log(detail)
            else:
                log(f"批次3: 未达收尾判据 —— {detail.splitlines()[-1] if detail else '?'}")
        elif not alive:
            log("批次3: ⚠️ 队列进程不在且未满 20 ⇒ **两条线都要查**（勿只当成'跑完了'）")

    # --- 批次 2 ---
    ok, bad = local_status(TAG2)
    log(f"批次2: 本地完整 {ok}/20（判据=尺寸门槛）；不完整 {len(bad)} 个: {', '.join(bad) if bad else '无'}")
    if ok < 20:
        # ⭐ §54：RTT 已在轮首测得，这里直接判，不再重复探测。
        if rtt_ms is not None and rtt_ms <= RTT_OK_MS:
            log(f"批次2: 回拉前置 —— 控制面 RTT={rtt_ms} ms（≤{RTT_OK_MS}）⇒ 允许回拉")
            start_pull(dry_run=dry_run)
        elif rtt_ms is None:
            log("批次2: ⏸ 控制面不可达 ⇒ **本轮不启动回拉**")
        else:
            log(f"批次2: ⏸ 控制面 RTT={rtt_ms} ms（>{RTT_OK_MS}）⇒ **负载未降，禁止回拉**"
                f"（每次重试先付 {rtt_ms/1000:.1f}s 握手成本，多半到不了数据阶段）")
    else:
        log("批次2: 20/20 齐；登记行已由 --register 生成（见 /tmp/emit_S14B2.txt），待人工复核后 --append")
    log("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daemon", action="store_true", help="常驻，每 900 s 一轮")
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument(
        "--no-start", dest="dry_run", action="store_true",
        help="**不真的启动回拉写者**，只打印本应执行的命令（验证互斥逻辑时用这个！）",
    )
    args = ap.parse_args()
    if not args.daemon:
        one_round(dry_run=args.dry_run)
        return 0
    log(f"看护常驻启动，间隔 {args.interval}s（dry_run={args.dry_run}）")
    while True:
        try:
            one_round(dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            log(f"⚠️ 本轮异常（已吞掉，不中断看护）: {type(exc).__name__}: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
