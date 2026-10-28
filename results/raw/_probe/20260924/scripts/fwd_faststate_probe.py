"""设备侧探针 v5：用「满占空比」钉住快态，产出可信标定表 + 归因证据。

v4（fwd_stability_probe.json）已确认的事实
----
- 设备存在（至少）三档性能态：b=1 快态 ~59 ms / 慢态 ~78.5 ms（比值 1.327），
  b=2 快态 ~75 / 慢态 ~81（比值 1.079 ≈ 1791/1660）。
- 态的翻转与**负载占空比**相关：harness 连续步进（~85% 占空比）全程稳在快态
  （P1S3D c1 的 p10-p90 = 59.77-60.05，1016 步 × 3 重复！）；本探针的
  sync-per-call 模式（~65% 占空比，每次调用间 ~40ms python 间隙）在态间振荡，
  相位边界（flush/换格的 1-2s 空闲）会把态打回慢档。
- mthreads-gmi 的 CLOCK 读数恒 1660MHz，**反映不了**这个分档（静态值）。
- 每个新形状有一次性编译（首调 ~2.7s，落进 warmup 即可）+ ~20-30 次调用的
  autotune 微调（+5-10ms）。

本探针的设计
----
- **async 块**：连续提交 N 次前向、只在块尾 sync 一次 ⇒ 占空比 ~100%，
  应能钉住快态。每块的 per-call = 块总时长/N（设备受限时 = 每步设备成本）。
- S2 CAL：b∈{1,2,4,8} × kv∈{2048,2112,2176}，两趟（升/降序）交错。
  取**块最小值**为该格快态值（多块复现 ⇒ 平台验证）。
- S3 KV：b=1 的 kv∈{512,1024,4096,8192}（2048-2176 由 CAL 覆盖）。
- S4 SA：同一形状 sync-per-call ×40（采样器开）vs ×40（采样器停）vs async×3块
  —— 占空比归因的直接对照（顺带排除 gmi 子进程的 CPU 争抢）。
- S5 MIMIC：复刻 harness 的 b=1 解码步进——逐步 sync、每步重建 mask/pos、
  past 由上一步输出真实生长（kv 2048→2175，2 轮）——应落在 P1S3D c1 真值
  59.94 附近；若成立，探针协议与 harness 测量条件等价性得证。
- 采样器加温度（GPU Current Temp），间隔 1.5s；首条存原始输出备查。

**这不是 harness 产物**（无 meta.json、无标签盖章）⇒ 落 adhoc/，
结论不得进 EXPERIMENTS.md 结论行，只能进 RESEARCH_LOG。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

T0 = time.monotonic()  # 进程时间线原点：在任何重活之前

import torch
import torch_musa  # noqa: F401  —— 必须先 import，否则 torch.musa 不存在

# ⚠️ 不能用 `import torch.musa as musa`（模块加载期解析会失败），用属性访问。
musa = torch.musa


# ---------------------------------------------------------------- 时钟采样 --

class ClockSampler(threading.Thread):
    """每 interval 秒采 Graphics/Memory 时钟、利用率、温度。

    失败必须显形（errors 列表）；探针结论不依赖采样成功。
    支持 pause()/resume()（S4 的对照用）。
    """

    def __init__(self, interval: float = 1.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.pause_evt = threading.Event()
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.raw_first: str | None = None
        self.cmd = ["mthreads-gmi", "-q",
                    "--display=CLOCK,UTILIZATION,TEMPERATURE", "-i", "0"]

    def pause(self) -> None:
        self.pause_evt.set()

    def resume(self) -> None:
        self.pause_evt.clear()

    def run(self) -> None:
        while not self.stop_evt.is_set():
            if self.pause_evt.is_set():
                self.stop_evt.wait(0.2)
                continue
            t = round(time.monotonic() - T0, 2)
            try:
                p = subprocess.run(self.cmd, capture_output=True, text=True,
                                   timeout=6)
                out = p.stdout
                if self.raw_first is None:
                    self.raw_first = out[:800]
                gfx = mem = util = temp = None
                m = re.search(r"Graphics\s*:\s*(\d+)\s*MHz", out)
                if m:
                    gfx = int(m.group(1))
                m = re.search(r"Memory\s*:\s*(\d+)\s*MHz", out)
                if m:
                    mem = int(m.group(1))
                m = re.search(r"Gpu\s*:\s*(\d+)\s*%", out)
                if m:
                    util = int(m.group(1))
                m = re.search(r"GPU Current Temp\s*:\s*(\d+)\s*C", out)
                if m:
                    temp = int(m.group(1))
                if gfx is None and len(self.errors) < 3:
                    self.errors.append(
                        f"parse-fail rc={p.returncode} head={out[:150]!r}")
                self.samples.append({"t": t, "gfx_mhz": gfx, "mem_mhz": mem,
                                     "gpu_util": util, "temp_c": temp})
            except Exception as e:
                if len(self.errors) < 3:
                    self.errors.append(repr(e))
            self.stop_evt.wait(self.interval)


# ------------------------------------------------------- 复刻自 v3/v4 的构件 --

def cache_needs_object() -> bool:
    try:
        from transformers import cache_utils
    except Exception:
        return False
    dyn = getattr(cache_utils, "DynamicCache", None)
    return dyn is not None and hasattr(dyn, "from_legacy_cache")


def build_past_for_hf(kv):
    if not kv:
        return ()
    legacy = tuple((k, v) for k, v in kv)
    if cache_needs_object():
        from transformers.cache_utils import DynamicCache
        return DynamicCache.from_legacy_cache(legacy)
    return legacy


def build_past(model, batch: int, kv_lens: list[int], dtype, device):
    """复刻 _pack_past 的形状：逐层 (key, value)，左补位到 max(kv_lens)。"""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    past_max = max(kv_lens)
    layers = []
    for _ in range(n_layers):
        rows_k, rows_v = [], []
        for L in kv_lens:
            k = torch.zeros(1, n_kv, L, head_dim, dtype=dtype, device=device)
            v = torch.zeros(1, n_kv, L, head_dim, dtype=dtype, device=device)
            if L < past_max:
                pad = past_max - L
                k = torch.nn.functional.pad(k, (0, 0, pad, 0))
                v = torch.nn.functional.pad(v, (0, 0, pad, 0))
            rows_k.append(k)
            rows_v.append(v)
        layers.append((torch.cat(rows_k, dim=0), torch.cat(rows_v, dim=0)))
        del rows_k, rows_v
    return layers, past_max


def make_batch(model, batch: int, kv_lens: list[int], n_new: int, dtype, device):
    ids = torch.randint(1, 1000, (batch, n_new), dtype=torch.long, device=device)
    past_max = max(kv_lens)
    mask = torch.ones((batch, past_max + n_new), dtype=torch.long, device=device)
    for i, L in enumerate(kv_lens):
        pad = past_max - L
        if pad:
            mask[i, :pad] = 0
    pos = torch.stack([
        torch.arange(L, L + n_new, dtype=torch.long, device=device)
        for L in kv_lens
    ])
    return ids, mask, pos


def make_fwd(model, ids, mask, pos, past):
    def fn():
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask,
                         position_ids=pos,
                         past_key_values=build_past_for_hf(past),
                         use_cache=True)
    return fn


# ----------------------------------------------------------------- 计时 --

def async_chunk(fn, n: int) -> float:
    """连续提交 n 次前向、块尾 sync 一次 ⇒ 占空比 ~100%。
    返回 per-call ms（设备受限时 = 每步设备成本）。"""
    musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    musa.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def sync_series(fn, n: int) -> list[dict]:
    """对照用：v3/v4 的 sync-per-call 模式（含 python 间隙）。"""
    musa.synchronize()
    out = []
    for i in range(n):
        wall = round(time.monotonic() - T0, 2)
        t0 = time.perf_counter()
        fn()
        musa.synchronize()
        out.append({"i": i, "t": wall,
                    "ms": round((time.perf_counter() - t0) * 1000.0, 3)})
    return out


def submit_cost(fn, n: int = 10) -> list[float]:
    """主机侧提交成本：每次 fn() 的 host 耗时（不同步，设备滞后执行）。"""
    musa.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(round((time.perf_counter() - t0) * 1000.0, 3))
    musa.synchronize()
    return ts


# ------------------------------------------------------------------ main --

# P1S3D static 臂纯 decode 步 fwd_ms p50（本地 telemetry 提取，kv≈2112）。
TRUTH_P1S3D = {"b1": 59.94, "b2": 75.53, "b4": 85.19, "b8": 113.75}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out",
                    default="/root/ascend_probe/results/fwd_faststate_probe.json")
    args = ap.parse_args()

    device = torch.device("musa:0")
    dtype = torch.float16
    print(f"[load] {args.model} ...", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    musa.synchronize()
    t_load_done = round(time.monotonic() - T0, 1)
    print(f"[load] done at t={t_load_done}s", flush=True)

    sampler = ClockSampler()
    sampler.start()

    phases: dict[str, dict] = {}
    had_error = False
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def flush(complete: bool) -> None:
        payload = {
            "probe": "fwd_faststate_probe",
            "model": args.model,
            "precision": "fp16",
            "device_name": musa.get_device_name(0),
            "torch": torch.__version__,
            "torch_musa": getattr(torch_musa, "__version__", None),
            "t_load_done_s": t_load_done,
            "complete": complete,
            "note": "设备侧探针，无 harness 盖章；不得进 EXPERIMENTS 结论行",
            "reference_p1s3d": TRUTH_P1S3D,
            "clock_samples": list(sampler.samples),
            "clock_errors": sampler.errors,
            "clock_raw_first": sampler.raw_first,
            "phases": phases,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)

    def do_phase(name: str, fn_build):
        nonlocal had_error
        gc.collect()
        print(f"\n[{name}] t={time.monotonic()-T0:.1f}s", flush=True)
        try:
            phases[name] = fn_build()
            flush(complete=False)
        except Exception as e:
            import traceback
            phases[name] = {"error": repr(e),
                            "traceback": traceback.format_exc()}
            had_error = True
            print(f"  !! {name} FAILED: {e!r}", flush=True)
            flush(complete=False)

    def build_cell(b: int, kv: int):
        kv_lens = [kv] * b
        past, _ = build_past(model, b, kv_lens, dtype, device)
        ids, mask, pos = make_batch(model, b, kv_lens, 1, dtype, device)
        fn = make_fwd(model, ids, mask, pos, past)
        return {"b": b, "kv": kv, "past": past, "fn": fn}

    # ---- S1：主机提交成本（async 解释的锚点）----
    def s1():
        out = {}
        for b in (1, 2):
            cell = build_cell(b, 2048)
            ts = submit_cost(cell["fn"], 10)
            out[f"b{b}"] = ts
            print(f"  submit b={b}: med={statistics.median(ts):.2f} ms "
                  f"(min {min(ts):.2f} / max {max(ts):.2f})", flush=True)
            s1.cells = s1.__dict__.get("cells", {})
            s1.cells[b] = cell
        return {"spec": "每次 fn() 的 host 耗时（无 sync）×10",
                "submit_ms": out}

    do_phase("S1_submit_cost", s1)

    # ---- S2：async 标定表（两趟升/降序交错）----
    CHUNKS = {1: (4, 20), 2: (4, 20), 4: (3, 20), 8: (3, 20)}

    def s2():
        cal = [(b, kv) for b in (1, 2, 4, 8) for kv in (2048, 2112, 2176)]
        orders = [cal, list(reversed(cal))]
        built: dict[tuple, dict] = {}
        per_cell: dict[str, dict] = {}
        for (b, kv) in cal:
            per_cell[f"b{b}_kv{kv}"] = {"b": b, "kv": kv, "chunks": []}
        for r, order in enumerate(orders):
            for (b, kv) in order:
                if (b, kv) not in built:
                    built[(b, kv)] = build_cell(b, kv)
                cell = built[(b, kv)]
                if r == 0:
                    for _ in range(3):  # 吸收一次性编译（首调 ~2.7s）
                        cell["fn"]()
                    musa.synchronize()
                nch, ncalls = CHUNKS[b]
                for ci in range(nch):
                    wall = round(time.monotonic() - T0, 2)
                    pc = async_chunk(cell["fn"], ncalls)
                    per_cell[f"b{b}_kv{kv}"]["chunks"].append(
                        {"round": r, "chunk": ci, "t": wall,
                         "per_call_ms": round(pc, 3)})
            print(f"  round {r} done t={time.monotonic()-T0:.1f}s", flush=True)
        for key, d in per_cell.items():
            vals = [c["per_call_ms"] for c in d["chunks"]]
            r0 = [c["per_call_ms"] for c in d["chunks"] if c["round"] == 0]
            r1 = [c["per_call_ms"] for c in d["chunks"] if c["round"] == 1]
            d["summary"] = {
                "min": round(min(vals), 3),
                "med": round(statistics.median(vals), 3),
                "max": round(max(vals), 3),
                "round0_min": round(min(r0), 3),
                "round1_min": round(min(r1), 3),
            }
            print(f"  {key:12s} min={d['summary']['min']:8.2f}  "
                  f"med={d['summary']['med']:8.2f}  "
                  f"max={d['summary']['max']:8.2f}", flush=True)
        print("  -- 对照 P1S3D 真值（kv≈2112 p50）--", flush=True)
        for b in (1, 2, 4, 8):
            d = per_cell[f"b{b}_kv2112"]["summary"]
            tr = TRUTH_P1S3D[f"b{b}"]
            print(f"    b={b}: async-min {d['min']:8.2f} vs truth {tr:8.2f} "
                  f"({(d['min']-tr)/tr*100:+.1f}%)", flush=True)
        return {"spec": "async 块标定；每块 per-call=块时长/N",
                "chunk_plan": {str(k): v for k, v in CHUNKS.items()},
                "cells": per_cell}

    do_phase("S2_cal_async", s2)

    # ---- S3：b=1 的 KV 扫描（async）----
    def s3():
        kvs = [512, 1024, 4096, 8192]
        built: dict[int, dict] = {}
        per_cell: dict[str, dict] = {}
        for kv in kvs:
            per_cell[f"kv{kv}"] = {"kv": kv, "chunks": []}
        for r, order in enumerate([kvs, list(reversed(kvs))]):
            for kv in order:
                if kv not in built:
                    built[kv] = build_cell(1, kv)
                    for _ in range(3):
                        built[kv]["fn"]()
                    musa.synchronize()
                for ci in range(3):
                    wall = round(time.monotonic() - T0, 2)
                    pc = async_chunk(built[kv]["fn"], 20)
                    per_cell[f"kv{kv}"]["chunks"].append(
                        {"round": r, "chunk": ci, "t": wall,
                         "per_call_ms": round(pc, 3)})
        for key, d in per_cell.items():
            vals = [c["per_call_ms"] for c in d["chunks"]]
            d["summary"] = {"min": round(min(vals), 3),
                            "med": round(statistics.median(vals), 3),
                            "max": round(max(vals), 3)}
            print(f"  {key:8s} min={d['summary']['min']:8.2f}  "
                  f"med={d['summary']['med']:8.2f}  "
                  f"max={d['summary']['max']:8.2f}", flush=True)
        return {"spec": "b=1 KV 扫描（async 块 ×3/轮 ×2 轮）",
                "cells": per_cell}

    do_phase("S3_kv_async", s3)

    # ---- S4：sync vs async 对照（同形状 b=1 kv=2048）----
    def s4():
        cell = build_cell(1, 2048)
        for _ in range(3):
            cell["fn"]()
        musa.synchronize()
        out: dict[str, list] = {}
        out["sync_sampler_on"] = sync_series(cell["fn"], 40)
        sampler.pause()
        out["sync_sampler_off"] = sync_series(cell["fn"], 40)
        sampler.resume()
        out["async_sampler_on"] = [
            {"t": round(time.monotonic() - T0, 2),
             "per_call_ms": round(async_chunk(cell["fn"], 20), 3)}
            for _ in range(3)
        ]
        for k in ("sync_sampler_on", "sync_sampler_off"):
            ms = [s["ms"] for s in out[k]]
            print(f"  {k:18s} med={statistics.median(ms):8.2f}  "
                  f"min={min(ms):8.2f}  max={max(ms):8.2f}", flush=True)
        for s in out["async_sampler_on"]:
            print(f"  async chunk        per_call={s['per_call_ms']:8.2f}",
                  flush=True)
        return {"spec": "b=1 kv=2048：sync40(采样开) / sync40(采样停) / async×3块",
                "series": out}

    do_phase("S4_sync_vs_async", s4)

    # ---- S5：MIMIC——复刻 harness 的 b=1 解码步进 ----
    def s5():
        all_recs: list[dict] = []
        for rnd in range(2):
            past, _ = build_past(model, 1, [2048], dtype, device)
            musa.synchronize()
            for step in range(128):
                kv = 2048 + step
                ids = torch.randint(1, 1000, (1, 1), dtype=torch.long,
                                    device=device)
                mask = torch.ones((1, kv + 1), dtype=torch.long, device=device)
                pos = torch.tensor([[kv]], dtype=torch.long, device=device)
                wall = round(time.monotonic() - T0, 2)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(input_ids=ids, attention_mask=mask,
                                position_ids=pos,
                                past_key_values=build_past_for_hf(past),
                                use_cache=True)
                musa.synchronize()
                ms = round((time.perf_counter() - t0) * 1000.0, 3)
                all_recs.append({"round": rnd, "step": step, "kv": kv,
                                 "t": wall, "ms": ms})
                cache = out.past_key_values
                if hasattr(cache, "to_legacy_cache"):
                    past = cache.to_legacy_cache()
                else:
                    past = cache
            ms = [r["ms"] for r in all_recs if r["round"] == rnd]
            print(f"  round {rnd}: med={statistics.median(ms):.2f} "
                  f"min={min(ms):.2f} max={max(ms):.2f}", flush=True)
        ms_all = [r["ms"] for r in all_recs]
        srt = sorted(ms_all)
        p10 = srt[int(0.10 * (len(srt) - 1))]
        p90 = srt[int(0.90 * (len(srt) - 1))]
        med = statistics.median(ms_all)
        print(f"  全部 256 步: med={med:.2f} p10={p10:.2f} p90={p90:.2f}  "
              f"truth(c1)=59.94", flush=True)
        return {"spec": "b=1 解码步进 mimic：逐步步进 kv 2048→2175，逐次 sync，"
                        "past 由上一步输出真实生长；2 轮 × 128 步",
                "records": all_recs,
                "summary": {"med": round(med, 3), "p10": round(p10, 3),
                            "p90": round(p90, 3), "min": round(min(ms_all), 3),
                            "max": round(max(ms_all), 3)}}

    do_phase("S5_mimic_decode", s5)

    sampler.stop_evt.set()
    sampler.join(timeout=3)
    flush(complete=True)
    print(f"\n[done] phases={len(phases)} clock_samples={len(sampler.samples)} "
          f"-> {args.out}", flush=True)
    if had_error:
        print("[done] 有 phase 失败，退出码=1", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
