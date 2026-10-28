"""分析 fwd_stability_probe.json：冷→暖归因 + 标定表对照。只读，不改任何产物。"""

from __future__ import annotations

import json
import statistics
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "C:/tmp/remote_scripts/fwd_stability_probe.json"

data = json.load(open(PATH, encoding="utf-8"))
phases = data["phases"]
clock = data["clock_samples"]
print(f"probe complete={data.get('complete')}  t_load_done={data.get('t_load_done_s')}s")
if data.get("clock_errors"):
    print("clock_errors:", data["clock_errors"])
print()

TRUTH = {"b1": 59.94, "b2": 75.53, "b4": 85.19, "b8": 113.75}


def med(vals):
    return statistics.median(vals) if vals else float("nan")


def win_med(ms, w=5):
    return [round(med(ms[i:i + w]), 2) for i in range(0, len(ms) - w + 1)]


# ---------- 1. P1 ramp ----------
p1 = phases.get("P1_ramp", {})
if "samples" in p1:
    s = p1["samples"]
    ms = [x["ms"] for x in s]
    t = [x["t"] for x in s]
    print("== P1 ramp (b=2 kv=2048, no warmup) ==")
    print(f"  n={len(ms)}  first={ms[0]:.2f}  min={min(ms):.2f}  "
          f"last10med={med(ms[-10:]):.2f}")
    wm = win_med(ms, 5)
    # 最大相邻跳变（5 次窗中位数）
    jumps = [(abs(wm[i + 1] - wm[i]), i) for i in range(len(wm) - 1)]
    jumps.sort(reverse=True)
    j, i = jumps[0]
    print(f"  rolling-5-med: {wm}")
    print(f"  最大跳变 {j:.2f} ms 在窗口 {i}->{i+1} "
          f"(约第 {i*5+5}..{i*5+15} 次调用, t≈{t[min(len(t)-1, i*5+10)]:.1f}s)")
    print(f"  invariants_ok={p1.get('invariants_ok')}")
    print()

# ---------- 2. 时钟对照 ----------
if clock:
    gfx = [c["gfx_mhz"] for c in clock if c.get("gfx_mhz") is not None]
    print("== clock samples ==")
    print(f"  n={len(clock)}  gfx min/max = {min(gfx)}/{max(gfx)} MHz")
    # 转变点前后各 3 个采样
    if "samples" in p1:
        s = p1["samples"]
        ms = [x["ms"] for x in s]
        wm = win_med(ms, 5)
        jumps = [(abs(wm[i + 1] - wm[i]), i) for i in range(len(wm) - 1)]
        jumps.sort(reverse=True)
        _, i = jumps[0]
        t_trans = s[min(len(s) - 1, i * 5 + 10)]["t"]
        near = [c for c in clock if abs(c["t"] - t_trans) <= 6]
        print(f"  拐点 t≈{t_trans}s 附近的时钟采样:")
        for c in near:
            print(f"    t={c['t']:7.1f}  gfx={c['gfx_mhz']}MHz  util={c['gpu_util']}%")
    print()

# ---------- 3. empty_cache / 重建 / 换形状 是否重置冷态 ----------
print("== 重置试验（看 first10 med vs last10 med）==")
for name in ("P2a_empty_cache", "P3_shape_switch", "P2b_rebuild_after_empty"):
    p = phases.get(name, {})
    if "samples" not in p:
        print(f"  {name}: MISSING/ERROR {p.get('error')}")
        continue
    ms = [x["ms"] for x in p["samples"]]
    f, l = med(ms[:10]), med(ms[-10:])
    verdict = "重置回冷态" if f > l * 1.10 else "未重置（暖态保持）"
    print(f"  {name:24s} first10={f:7.2f}  last10={l:7.2f}  -> {verdict}")
print()

# ---------- 4. P4 分解 ----------
print("== P4 分解（b=2 暖态）==")
p4a = phases.get("P4a_steady", {}).get("summary", {})
p4c = phases.get("P4c_build_only", {}).get("summary", {})
p4b = phases.get("P4b_build_plus_fwd", {}).get("summary", {})
if p4a and p4b:
    print(f"  steady fwd          = {p4a.get('med')}")
    print(f"  build_only          = {p4c.get('med')}")
    print(f"  build+fwd           = {p4b.get('med')}")
    print(f"  (build+fwd) - fwd   = "
          f"{round((p4b['med'] - p4a['med']), 3)}  "
          f"(对照 build_only={p4c.get('med')} ⇒ 有无交互项)")
print()

# ---------- 5. P5 标定表 ----------
p5 = phases.get("P5_calibration", {})
if "cells" in p5:
    print("== P5 标定表（med ms；括号 = 该格三趟 pass 的中位数漂移）==")
    cells = p5["cells"]
    hdr = "b\\kv    " + "".join(f"{k:>14}" for k in (2048, 2112, 2176))
    print(hdr)
    for b in (1, 2, 4, 8):
        row = f"b={b}     "
        for kv in (2048, 2112, 2176):
            c = cells.get(f"b{b}_kv{kv}")
            if not c:
                row += f"{'--':>14}"
                continue
            m = c["summary"]["med"]
            drift = max(c["pass_meds"]) - min(c["pass_meds"])
            row += f"{m:>8.2f}({drift:4.1f})"
        print(row)
    print()
    print("  对照 P1S3D 真值（kv≈2112 p50）:")
    for b in (1, 2, 4, 8):
        c = cells.get(f"b{b}_kv2112")
        if c:
            m = c["summary"]["med"]
            tr = TRUTH[f"b{b}"]
            print(f"    b={b}: probe {m:8.2f}  truth {tr:8.2f}  "
                  f"diff {(m - tr) / tr * 100:+6.1f}%")
    print()
    print("  kv 斜率 (ms/token, 由 2048->2176):")
    for b in (1, 2, 4, 8):
        lo = cells.get(f"b{b}_kv2048")
        hi = cells.get(f"b{b}_kv2176")
        if lo and hi:
            slope = (hi["summary"]["med"] - lo["summary"]["med"]) / 128
            print(f"    b={b}: {slope:.4f} ms/tok  "
                  f"(128 tok ≈ {slope * 128:.2f} ms)")
    print()

# ---------- 6. P6 KV 扫描 ----------
p6 = phases.get("P6_kv_sweep_warm", {})
if "cells" in p6:
    print("== P6 暖态 KV 扫描（b=1）==")
    for kv in (512, 1024, 2048, 4096, 8192):
        c = p6["cells"].get(f"kv{kv}")
        if c:
            print(f"  kv={kv:5d}: med={c['summary']['med']:8.2f}  "
                  f"pass_meds={c['pass_meds']}")
    print(f"  monotonic={p6.get('monotonic')}")

# ---------- 7. phase 错误汇总 ----------
errs = [k for k, v in phases.items() if "error" in v]
if errs:
    print("\n!! 失败 phase:", errs)
