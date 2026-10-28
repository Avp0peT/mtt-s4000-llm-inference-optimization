#!/usr/bin/env python3
"""CPU 稳定性探针：**直接测**"这个容器会不会被别的负载抢 CPU 而污染计时"。

为什么不能只看 loadavg
----------------------
``/proc/loadavg`` 在**没有 lxcfs** 的容器里是**宿主机**的负载，不是本容器的。
本机实测：loadavg ≈ 12.85 而宿主机有 128 核（≈10%），容器配额只有 15 CPU
（``cpu.max = 1500000 100000``）。⇒ 拿宿主 loadavg 去判断"我会不会被抢"是**跨主体比较**，
两个方向都可能错：宿主很忙而本容器配额充裕（误报），或宿主看着闲而本容器
被同宿主的邻居抢（漏报）。

正确做法两层：
1. **容器级 PSI**（``/sys/fs/cgroup/cpu.pressure``）——cgroup v2 下它是**本 cgroup**
   的压力，直接回答"有没有被饿住"；
2. **直接测**：跑一段固定量的纯 CPU 计算若干次，看耗时离散度。
   这才是"计时会不会被污染"的**地面真值**——仪器本身也被污染时，只有实测能看出来。

判据（**事前固定**，不事后放宽）
--------------------------------
* ``cpu.pressure some avg10``：> 5% ⇒ 有压力，正式计时前先查。
* 计时探针的**稳健 CV**（1.4826·MAD/中位数，见 `hw-benchmark-stability`）：
  ≤ 1% 视为稳定；> 5% 视为不稳定 ⇒ **不得开始正式计时**。

⚠️ 它要跑在**推理主机**上（计时受污染的正是那台）：用 ``scripts/remote_run.py`` 投上去跑，
脚本已入库而在远端副本不必保留（见 PLAYBOOK §5 的投放方式）。

用法::

    python3 check_cpu_stability.py [重复次数] [每次秒数]     # 默认 7 × 1.0s

退出码：``0`` 稳定 / ``1`` 边缘或**不稳定**（⇒ 调用方应当**拒绝开始正式计时**）。
把"记得先看一眼负载"变成"跑不过就不许开始"——这类纪律靠人记必然会漏。
"""

from __future__ import annotations

import statistics
import sys
import time


def read_cpu_pressure() -> dict[str, str]:
    """容器级 PSI。取不到就如实说取不到（**不返回 0**，那会被读成"毫无压力"）。"""
    for path in ("/sys/fs/cgroup/cpu.pressure", "/proc/pressure/cpu"):
        try:
            with open(path, encoding="utf-8") as fh:
                return {"path": path, "lines": fh.read().strip()}
        except OSError:
            continue
    return {"path": "", "lines": "（读不到）"}


def _read(path: str) -> str | None:
    """读一个 /proc 或 /sys 文件；**取不到返回 None**（不是空串、不是 0）。

    为什么必须区分：本脚本要在**推理主机**上跑，而它也会在本机（Windows）被误跑。
    Windows 上没有 ``/proc/loadavg`` ⇒ 若让它抛异常，"跑不了"会被读成"测出问题了"；
    若返回空串，又会被读成"负载为空"。两者都错。⇒ 返回 ``None``，由调用方判"无法判定"。
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return None


def read_cpu_max() -> str:
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "（读不到）"


def busy(seconds: float) -> float:
    """一段固定量的纯 CPU 计算，返回实际耗时（秒）。

    刻意**不用** ``sleep``：sleep 不消耗 CPU，测不出争抢。
    也刻意不用多进程：并行度会掩盖单核争抢。
    """
    start = time.perf_counter()
    x = 0
    while time.perf_counter() - start < seconds:
        for i in range(1000):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    return time.perf_counter() - start


def robust_cv(values: list[float]) -> float | None:
    """稳健 CV = 1.4826·MAD/中位数。n < 3 时**不给**——两个点算出的离散度会误导人。"""
    if len(values) < 3:
        return None
    med = statistics.median(values)
    if med == 0:
        return None
    mad = statistics.median([abs(v - med) for v in values])
    return 1.4826 * mad / med


def main(argv: list[str]) -> int:
    reps = int(argv[1]) if len(argv) > 1 else 7
    span = float(argv[2]) if len(argv) > 2 else 1.0

    loadavg = _read("/proc/loadavg")
    pressure = read_cpu_pressure()
    is_linux = loadavg is not None
    print(f"负载      : {loadavg or 'n/a（本机不是 Linux）'}"
          + ("   （⚠️ 无 lxcfs ⇒ 这是**宿主机**的负载）" if is_linux else ""))
    print(f"cpu.max   : {read_cpu_max()}   （容器真正能用的 CPU 数）")
    print(f"PSI       : {pressure['path']}")
    for line in pressure["lines"].splitlines():
        print(f"            {line}")

    print(f"\n计时探针：{reps} 次 × 每次约 {span}s 纯 CPU 计算")
    durs: list[float] = []
    for k in range(reps):
        d = busy(span)
        durs.append(d)
        print(f"  #{k}: {d:.6f} s")
    cv = robust_cv(durs)
    print(f"\n中位数    : {statistics.median(durs):.6f} s")
    print(f"极差      : {max(durs) - min(durs):.6f} s")
    if cv is None:
        # n < 3 时**不给**离散度（两个点算出的 CV 会误导人），也**不放行**：
        # "没测出离散度"不是"离散度合格"。
        print(f"稳健 CV   : n/a（需要 ≥3 次，现 n={len(durs)}）⇒ 拒绝放行")
        return 1
    print(f"稳健 CV   : {cv * 100:.3f}%")
    if not is_linux:
        # ⚠️ 本机跑不出结论：这个闸门要挡的是**推理主机**的计时环境。
        # 三态：0 = 稳定 / 1 = 不稳定 / 2 = **无法判定**（不可测不能冒充合格）。
        print("？ 无法判定：本机不是 Linux（读不到 /proc/loadavg 与 cgroup）——")
        print("   请在**推理主机**上跑本脚本；用 scripts/remote_run.py put 投上去再 exec。")
        return 2
    if cv <= 0.01:
        print("✓ 稳定（≤1%）⇒ 计时环境可用")
        return 0
    if cv <= 0.05:
        print("⚠️ 边缘（1%~5%）⇒ 可以做正确性验证，**不要**做性能对比")
        return 1
    print("✗ 不稳定（>5%）⇒ **不得开始正式计时**：差值会被噪声淹没")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
