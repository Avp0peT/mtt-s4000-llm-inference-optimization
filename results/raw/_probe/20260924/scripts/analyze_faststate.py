"""分析 fwd_faststate_probe.json：快态标定表 + 归因。只读。"""

from __future__ import annotations

import json
import statistics
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "C:/tmp/remote_scripts/fwd_faststate_probe.json"
data = json.load(open(PATH, encoding="utf-8"))
ph = data["phases"]
TRUTH = {"b1": 59.94, "b2": 75.53, "b4": 85.19, "b8": 113.75}

print(f"complete={data.get('complete')}  t_load={data.get('t_load_done_s')}s")
if data.get("clock_errors"):
    print("clock_errors:", data["clock_errors"])
temps = [c["temp_c"] for c in data.get("clock_samples", []) if c.get("temp_c") is not None]
utils = [c["gpu_util"] for c in data.get("clock_samples", []) if c.get("gpu_util") is not None]
if temps:
    print(f"temp: min/max = {min(temps)}/{max(temps)} C   "
          f"util min/med/max = {min(utils)}/{statistics.median(utils)}/{max(utils)}%")
print()

# ---- S1 ----
s1 = ph.get("S1_submit_cost", {})
if "submit_ms" in s1:
    print("== S1 主机提交成本（每次 fn() 的 host 耗时）==")
    for k, ts in s1["submit_ms"].items():
        print(f"  {k}: med={statistics.median(ts):.2f}  "
              f"min={min(ts):.2f}  max={max(ts):.2f}  raw={ts}")
    print()

# ---- S2 标定表 ----
s2 = ph.get("S2_cal_async", {})
if "cells" in s2:
    cells = s2["cells"]
    print("== S2 async 标定表（per-call ms）==")
    print("  b\\kv      " + "".join(f"{k:>22}" for k in (2048, 2112, 2176)))
    for b in (1, 2, 4, 8):
        row = f"  b={b}      "
        for kv in (2048, 2112, 2176):
            c = cells.get(f"b{b}_kv{kv}")
            if c:
                s = c["summary"]
                row += f"{s['min']:>8.2f}/{s['med']:>7.2f}/{s['max']:>6.2f}"
        print(row)
    print("  （min/med/max，min=快态平台）")
    print()
    print("  对照 P1S3D 真值（kv≈2112 p50）:")
    for b in (1, 2, 4, 8):
        c = cells.get(f"b{b}_kv2112")
        if c:
            mn = c["summary"]["min"]
            md = c["summary"]["med"]
            tr = TRUTH[f"b{b}"]
            print(f"    b={b}: min {mn:8.2f} ({(mn-tr)/tr*100:+5.1f}%)  "
                  f"med {md:8.2f} ({(md-tr)/tr*100:+5.1f}%)  truth {tr}")
    print()
    print("  kv 斜率（2048→2176, 用 min）:")
    for b in (1, 2, 4, 8):
        lo = cells.get(f"b{b}_kv2048")
        hi = cells.get(f"b{b}_kv2176")
        if lo and hi:
            slope = (hi["summary"]["min"] - lo["summary"]["min"]) / 128
            print(f"    b={b}: {slope:.4f} ms/tok (128 tok ≈ {slope*128:.2f} ms)")
    print()
    print("  每格 chunk 序列（快态平台是否复现）:")
    for b in (1, 2, 4, 8):
        for kv in (2048, 2112, 2176):
            c = cells[f"b{b}_kv{kv}"]
            seq = " ".join(f"{x['per_call_ms']:.1f}" for x in c["chunks"])
            print(f"    b{b}_kv{kv}: {seq}")
    print()

# ---- S3 KV 扫描 ----
s3 = ph.get("S3_kv_async", {})
if "cells" in s3:
    print("== S3 b=1 KV 扫描（async，min/med）==")
    for kv in (512, 1024, 4096, 8192):
        c = s3["cells"].get(f"kv{kv}")
        if c:
            s = c["summary"]
            seq = " ".join(f"{x['per_call_ms']:.1f}" for x in c["chunks"])
            print(f"  kv{kv:5d}: min={s['min']:7.2f} med={s['med']:7.2f}  [{seq}]")
    s2c = ph.get("S2_cal_async", {}).get("cells", {})
    if s2c:
        print(f"  (对照 kv2048: min={s2c['b1_kv2048']['summary']['min']:.2f})")
    print()

# ---- S4 sync vs async ----
s4 = ph.get("S4_sync_vs_async", {})
if "series" in s4:
    print("== S4 sync vs async（b=1 kv=2048）==")
    for k in ("sync_sampler_on", "sync_sampler_off"):
        ms = [s["ms"] for s in s4["series"][k]]
        srt = sorted(ms)
        print(f"  {k:18s} med={statistics.median(ms):7.2f}  "
              f"p10={srt[int(0.1*(len(srt)-1))]:7.2f}  "
              f"p90={srt[int(0.9*(len(srt)-1))]:7.2f}  "
              f"min={min(ms):7.2f}  max={max(ms):7.2f}")
    for s in s4["series"]["async_sampler_on"]:
        print(f"  async chunk        per_call={s['per_call_ms']:7.2f}")
    print()

# ---- S5 MIMIC ----
s5 = ph.get("S5_mimic_decode", {})
if "records" in s5:
    recs = s5["records"]
    print("== S5 MIMIC（b=1 解码步进，逐步 sync）==")
    s = s5["summary"]
    print(f"  全部 {len(recs)} 步: med={s['med']:.2f} p10={s['p10']:.2f} "
          f"p90={s['p90']:.2f} min={s['min']:.2f} max={s['max']:.2f}  "
          f"truth=59.94 ({(s['med']-59.94)/59.94*100:+.1f}%)")
    # 逐 kv 中位数（两轮合并）
    by_kv: dict[int, list[float]] = {}
    for r in recs:
        by_kv.setdefault(r["kv"], []).append(r["ms"])
    kvs = sorted(by_kv)
    print("  逐 kv med（每 32 tok 打点）:")
    for kv in kvs[::32]:
        vals = by_kv[kv]
        print(f"    kv={kv}: med={statistics.median(vals):7.2f}  n={len(vals)}")
    # 轨迹斜率
    if len(kvs) > 2:
        lo = statistics.median(by_kv[kvs[0]])
        hi = statistics.median(by_kv[kvs[-1]])
        print(f"    kv {kvs[0]}→{kvs[-1]}: "
              f"{lo:.2f}→{hi:.2f}  斜率 {(hi-lo)/(kvs[-1]-kvs[0]):.4f} ms/tok")
    # 轮内稳定性
    for rnd in (0, 1):
        ms = [r["ms"] for r in recs if r["round"] == rnd]
        srt = sorted(ms)
        print(f"    round {rnd}: med={statistics.median(ms):.2f} "
              f"p10={srt[int(0.1*(len(srt)-1))]:.2f} "
              f"p90={srt[int(0.9*(len(srt)-1))]:.2f}")

errs = [k for k, v in ph.items() if "error" in v]
if errs:
    print("\n!! 失败 phase:", errs)
