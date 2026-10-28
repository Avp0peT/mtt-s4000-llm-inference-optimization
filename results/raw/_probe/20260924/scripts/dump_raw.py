import json

d = json.load(open("C:/tmp/remote_scripts/fwd_stability_probe.json", encoding="utf-8"))
ph = d["phases"]

print("== P5 per-pass medians ==")
cells = ph["P5_calibration"]["cells"]
for b in (1, 2, 4, 8):
    for kv in (2048, 2112, 2176):
        c = cells[f"b{b}_kv{kv}"]
        print(f"  b{b}_kv{kv}: pass_meds={c['pass_meds']}")

print("\n== P5 b1 cells raw sequences (t, ms, pass) ==")
for kv in (2048, 2112, 2176):
    c = cells[f"b1_kv{kv}"]
    print(f"  b1_kv{kv}:")
    for s in c["samples"]:
        print(f"    t={s['t']:7.1f}  ms={s['ms']:7.2f}  pass={s['pass']}")

print("\n== P6 raw sequences ==")
for kv in (512, 1024, 2048, 4096, 8192):
    c = ph["P6_kv_sweep_warm"]["cells"][f"kv{kv}"]
    line = " ".join(f"p{s['pass']}:{s['ms']:.1f}" for s in c["samples"])
    print(f"  kv{kv}: {line}")

print("\n== clock util/mem over time (every 3rd) ==")
for c in d["clock_samples"][::3]:
    print(f"  t={c['t']:7.1f} gfx={c['gfx_mhz']} mem={c['mem_mhz']} util={c['gpu_util']}%")

print("\n== P2a/P2b/P4a raw (t, ms) first 12 + last 6 ==")
for name in ("P2a_empty_cache", "P2b_rebuild_after_empty", "P4a_steady", "P4b_build_plus_fwd"):
    ss = ph[name]["samples"]
    head = " ".join(f"{s['ms']:.1f}" for s in ss[:12])
    tail = " ".join(f"{s['ms']:.1f}" for s in ss[-6:])
    print(f"  {name}: head[{head}] tail[{tail}]")
