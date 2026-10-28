"""设备侧探针：定位 fwd_cost_profile v3 里 decode 计时的冷→暖漂移，并产出可信的暖态标定表。

背景（v3 已观测，/root/ascend_probe/results/fwd_cost_profile.json）
----
- B 组 kv=1024 的 5 次 repeat 内出现**阶跃**：78.5/79.4/77.7 → 59.1/55.9 ms（-27%）。
- 阶跃前后各自都很稳：kv=512 稳在 78（"稳但冷"），kv=4096 稳在 66.2 ±0.3%。
- prefill D 组（阶跃之后测的）与 harness 遥测吻合到 0.01%
  （D b=2/chunk=2048 = 864.286 vs telemetry `prefill_tok=4096 fwd_ms=864.37`）
  ⇒ 探针结构与 harness 同构，只是 decode A/B 两组在"冷态"里测的。
- idle 时 Graphics clock 1660MHz vs max 1791MHz（+7.9%，**解释不了 27%**）
  ⇒ 时钟只是候选之一，须并发采样对照。

要回答的问题
----
Q1 冷→暖的形态：连续计时（无 warmup），拐点在哪、几级台阶、多快收敛？
Q2 empty_cache 是否重置回冷态？（v3 每组之间都 empty_cache——嫌疑最大）
Q3 换新形状（**不** empty_cache）是否重新触发冷态？（autotune 假说；D 组证据说不会）
Q4 固定 past 张量 vs 每次重造 past（分配器扰动）差多少？
Q5 暖态标定表：b∈{1,2,4,8} × kv∈{2048,2112,2176}，对照 P1S3D 真值
   （纯 decode 步 fwd_ms p50：59.94 / 75.53 / 85.19 / 113.75，kv≈2112）。
Q6 B 组的非单调（kv512:78 → kv2048:61）在暖态下是否消失（变单调）？
Q7 时钟采样与拐点是否对齐（DVFS 归因：1660→1791 只值 ~8%，缺口另有来源）。

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
    """每 interval 秒采一次 Graphics/Memory 时钟与 GPU 利用率。

    失败必须**显形**（errors 列表），且探针结论不依赖采样成功。
    """

    def __init__(self, interval: float = 0.7):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.cmd = ["mthreads-gmi", "-q", "--display=CLOCK,UTILIZATION", "-i", "0"]

    def run(self) -> None:
        while not self.stop_evt.is_set():
            t = round(time.monotonic() - T0, 2)
            try:
                p = subprocess.run(self.cmd, capture_output=True, text=True,
                                   timeout=6)
                out = p.stdout
                gfx = mem = util = None
                m = re.search(r"Graphics\s*:\s*(\d+)\s*MHz", out)
                if m:
                    gfx = int(m.group(1))
                m = re.search(r"Memory\s*:\s*(\d+)\s*MHz", out)
                if m:
                    mem = int(m.group(1))
                m = re.search(r"Gpu\s*:\s*(\d+)\s*%", out)
                if m:
                    util = int(m.group(1))
                if gfx is None and len(self.errors) < 3:
                    self.errors.append(
                        f"parse-fail rc={p.returncode} head={out[:150]!r}")
                self.samples.append({"t": t, "gfx_mhz": gfx,
                                     "mem_mhz": mem, "gpu_util": util})
            except Exception as e:  # 采样失败不拖垮探针
                if len(self.errors) < 3:
                    self.errors.append(repr(e))
            self.stop_evt.wait(self.interval)


# ------------------------------------------------------- 复刻自 v3 的构件 --

def cache_needs_object() -> bool:
    try:
        from transformers import cache_utils
    except Exception:
        return False
    dyn = getattr(cache_utils, "DynamicCache", None)
    return dyn is not None and hasattr(dyn, "from_legacy_cache")


def build_past_for_hf(kv):
    """每次前向重建 HF 缓存对象（等价 real_backend._build_past_for_hf；
    4.46.3 上 = DynamicCache.from_legacy_cache，只包引用不拷贝）。"""
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
    """与 harness 对齐的前向：no_grad + use_cache=True + 缓存对象每次重建。"""
    def fn():
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask,
                         position_ids=pos,
                         past_key_values=build_past_for_hf(past),
                         use_cache=True)
    return fn


# ----------------------------------------------------------------- 统计 --

def pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    if not s:
        return float("nan")
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def summarize(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0}
    n = len(ms)
    half = max(1, n // 2)
    return {
        "n": n,
        "med": round(statistics.median(ms), 3),
        "p10": round(pct(ms, 0.10), 3),
        "p90": round(pct(ms, 0.90), 3),
        "min": round(min(ms), 3),
        "max": round(max(ms), 3),
        "first_half_med": round(statistics.median(ms[:half]), 3),
        "last_half_med": round(statistics.median(ms[-half:]), 3),
    }


def one_line(tag: str, s: dict) -> str:
    if s.get("n", 0) == 0:
        return f"  {tag:24s} (no data)"
    return (f"  {tag:24s} n={s['n']:3d}  med={s['med']:8.2f}  "
            f"[p10 {s['p10']:8.2f} .. p90 {s['p90']:8.2f}]  "
            f"first/last-half med {s['first_half_med']:8.2f}/{s['last_half_med']:8.2f}")


# ------------------------------------------------------------- 计时序列 --

def run_series(fn, n: int, *, warmup: int = 0) -> list[dict]:
    """n 次 sync 边界计时；每次记 wall(相对 T0) 与 ms。"""
    for _ in range(warmup):
        fn()
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


def run_ramp(fn, n_min: int = 60, n_max: int = 150) -> list[dict]:
    """无 warmup 连续计时；n_min 之后若最近两个 10 次的中位数相差 <1% 则停。"""
    musa.synchronize()
    out = []
    for i in range(n_max):
        wall = round(time.monotonic() - T0, 2)
        t0 = time.perf_counter()
        fn()
        musa.synchronize()
        out.append({"i": i, "t": wall,
                    "ms": round((time.perf_counter() - t0) * 1000.0, 3)})
        if i + 1 >= n_min and len(out) >= 20:
            last10 = [s["ms"] for s in out[-10:]]
            prev10 = [s["ms"] for s in out[-20:-10]]
            m1, m0 = statistics.median(last10), statistics.median(prev10)
            if m0 > 0 and abs(m1 - m0) / m0 < 0.01:
                break
    return out


# ------------------------------------------------------------------ main --

# P1S3D static 臂纯 decode 步 fwd_ms p50（本地 telemetry 提取，kv≈2112）。
# 只作对照打印，不参与任何计算。
TRUTH_P1S3D = {"b1": 59.94, "b2": 75.53, "b4": 85.19, "b8": 113.75}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out",
                    default="/root/ascend_probe/results/fwd_stability_probe.json")
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
    free, total = musa.mem_get_info()
    print(f"[load] done at t={t_load_done}s  free={free/1024**3:.2f} GiB",
          flush=True)

    sampler = ClockSampler()
    sampler.start()

    phases: dict[str, dict] = {}
    had_error = False
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def flush(complete: bool) -> None:
        payload = {
            "probe": "fwd_stability_probe",
            "model": args.model,
            "precision": "fp16",
            "device": "musa:0",
            "device_name": musa.get_device_name(0),
            "torch": torch.__version__,
            "torch_musa": getattr(torch_musa, "__version__", None),
            "t_load_done_s": t_load_done,
            "complete": complete,
            "note": "设备侧探针，无 harness 盖章；不得进 EXPERIMENTS 结论行",
            "reference_p1s3d": TRUTH_P1S3D,
            "clock_samples": list(sampler.samples),
            "clock_errors": sampler.errors,
            "phases": phases,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)

    def do_phase(name: str, fn_build):
        """每个 phase 独立 try/except：失败显形 + 退出码非零，但不拖垮后续。"""
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

    # ---- 张量集（P1/P2/P4 用的 b=2、P3 用的 b=4）----
    def build_cell(b: int, kv: int):
        kv_lens = [kv] * b
        past, _ = build_past(model, b, kv_lens, dtype, device)
        ids, mask, pos = make_batch(model, b, kv_lens, 1, dtype, device)
        fn = make_fwd(model, ids, mask, pos, past)
        return {"b": b, "kv": kv, "past": past, "ids": ids, "mask": mask,
                "pos": pos, "fn": fn}

    cell_b2 = build_cell(2, 2048)

    # ---- P1：冷→暖 ramp（无 warmup，自适应停）----
    def p1():
        samples = run_ramp(cell_b2["fn"])
        ms = [s["ms"] for s in samples]
        inv = (cell_b2["past"][0][0].shape[2] == 2048
               and cell_b2["past"][0][1].shape[2] == 2048)
        print(one_line("ramp_b2_kv2048", summarize(ms)), flush=True)
        print(f"  invariants(past 未被原地污染)={inv}", flush=True)
        return {"spec": "b=2 kv=2048 无 warmup 连续计时",
                "invariants_ok": inv, "samples": samples,
                "summary": summarize(ms)}

    do_phase("P1_ramp", p1)

    # ---- P2a：empty_cache 后同一批张量继续测（是否重置回冷态）----
    def p2a():
        musa.empty_cache()
        free2, _ = musa.mem_get_info()
        samples = run_series(cell_b2["fn"], 40)
        ms = [s["ms"] for s in samples]
        print(f"  after empty_cache free={free2/1024**3:.2f} GiB", flush=True)
        print(one_line("same_tensors", summarize(ms)), flush=True)
        return {"spec": "empty_cache 后沿用旧张量 40 次",
                "samples": samples, "summary": summarize(ms)}

    do_phase("P2a_empty_cache", p2a)

    # ---- P3：换新形状 b=4（不 empty_cache，张量全新）----
    def p3():
        cell_b4 = build_cell(4, 2048)
        samples = run_series(cell_b4["fn"], 40)
        ms = [s["ms"] for s in samples]
        print(one_line("fresh_b4_kv2048", summarize(ms)), flush=True)
        # b=4 张量留给后面（保持引用，避免 free-list 噪声）
        p3.cell = cell_b4
        return {"spec": "新形状 b=4 kv=2048（无 empty_cache）40 次",
                "samples": samples, "summary": summarize(ms)}

    do_phase("P3_shape_switch", p3)

    # ---- P2b：del + empty_cache + 重建 b=2 张量（复刻 v3 的组切换）----
    def p2b():
        nonlocal cell_b2
        del cell_b2
        gc.collect()
        musa.empty_cache()
        cell_b2 = build_cell(2, 2048)
        samples = run_series(cell_b2["fn"], 40)
        ms = [s["ms"] for s in samples]
        print(one_line("rebuilt_b2_kv2048", summarize(ms)), flush=True)
        return {"spec": "del 张量 + empty_cache + 重建 b=2 后 40 次",
                "samples": samples, "summary": summarize(ms)}

    do_phase("P2b_rebuild_after_empty", p2b)

    # ---- P4a：暖态基准（b=2 固定张量）----
    def p4a():
        samples = run_series(cell_b2["fn"], 40)
        ms = [s["ms"] for s in samples]
        print(one_line("steady_b2", summarize(ms)), flush=True)
        return {"spec": "暖态基准：b=2 固定张量 40 次",
                "samples": samples, "summary": summarize(ms)}

    do_phase("P4a_steady", p4a)

    # ---- P4c：只测"重造张量"本身（build_past + make_batch，无前向）----
    def p4c():
        def build_only():
            build_past(model, 2, [2048, 2048], dtype, device)
            make_batch(model, 2, [2048, 2048], 1, dtype, device)
        samples = run_series(build_only, 30)
        ms = [s["ms"] for s in samples]
        print(one_line("build_only", summarize(ms)), flush=True)
        return {"spec": "只重造 b=2 张量（无前向）30 次",
                "samples": samples, "summary": summarize(ms)}

    do_phase("P4c_build_only", p4c)

    # ---- P4b：每次计时内重造张量 + 前向（分配器扰动上界）----
    def p4b():
        def build_and_fwd():
            past, _ = build_past(model, 2, [2048, 2048], dtype, device)
            ids, mask, pos = make_batch(model, 2, [2048, 2048], 1,
                                        dtype, device)
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      past_key_values=build_past_for_hf(past), use_cache=True)
        samples = run_series(build_and_fwd, 30)
        ms = [s["ms"] for s in samples]
        print(one_line("build_plus_fwd", summarize(ms)), flush=True)
        return {"spec": "计时区内重造张量+前向 30 次",
                "samples": samples, "summary": summarize(ms)}

    do_phase("P4b_build_plus_fwd", p4b)

    # ---- P5：暖态标定表 b×kv，三趟交错（升/降/洗牌）----
    def p5():
        cells = [(b, kv) for b in (1, 2, 4, 8) for kv in (2048, 2112, 2176)]
        orders = [
            cells,
            list(reversed(cells)),
            [cells[i] for i in (5, 0, 11, 3, 8, 1, 7, 10, 2, 6, 4, 9)],
        ]
        built: dict[tuple, dict] = {}
        per_cell: dict[str, dict] = {}
        for (b, kv) in cells:
            key = f"b{b}_kv{kv}"
            per_cell[key] = {"b": b, "kv": kv, "samples": [], "pass_meds": []}
        for pi, order in enumerate(orders):
            for (b, kv) in order:
                if (b, kv) not in built:
                    built[(b, kv)] = build_cell(b, kv)
                cell = built[(b, kv)]
                samples = run_series(cell["fn"], 5, warmup=7)
                key = f"b{b}_kv{kv}"
                ms = [s["ms"] for s in samples]
                for s in samples:
                    s["pass"] = pi
                per_cell[key]["samples"].extend(samples)
                per_cell[key]["pass_meds"].append(
                    round(statistics.median(ms), 3))
            print(f"  pass {pi} done t={time.monotonic()-T0:.1f}s", flush=True)
        for key, d in per_cell.items():
            ms = [s["ms"] for s in d["samples"]]
            d["summary"] = summarize(ms)
            print(one_line(key, d["summary"]), flush=True)
        print("  -- 对照 P1S3D 真值（kv≈2112 的 p50）--", flush=True)
        for b in (1, 2, 4, 8):
            key = f"b{b}_kv2112"
            if key in per_cell and per_cell[key]["summary"].get("n"):
                med = per_cell[key]["summary"]["med"]
                truth = TRUTH_P1S3D[f"b{b}"]
                print(f"    b={b}: probe {med:8.2f} vs truth {truth:8.2f} "
                      f"({(med-truth)/truth*100:+.1f}%)", flush=True)
        return {"spec": "标定表 b∈{1,2,4,8}×kv∈{2048,2112,2176}，"
                        "每趟 7 warmup + 5 计时",
                "cells": per_cell}

    do_phase("P5_calibration", p5)

    # ---- P6：暖态 KV 扫描（b=1，两趟升/降序）----
    def p6():
        kvs = [512, 1024, 2048, 4096, 8192]
        orders = [kvs, list(reversed(kvs))]
        built: dict[int, dict] = {}
        per_cell: dict[str, dict] = {}
        for kv in kvs:
            per_cell[f"kv{kv}"] = {"kv": kv, "samples": [], "pass_meds": []}
        for pi, order in enumerate(orders):
            for kv in order:
                if kv not in built:
                    built[kv] = build_cell(1, kv)
                samples = run_series(built[kv]["fn"], 7, warmup=7)
                ms = [s["ms"] for s in samples]
                for s in samples:
                    s["pass"] = pi
                per_cell[f"kv{kv}"]["samples"].extend(samples)
                per_cell[f"kv{kv}"]["pass_meds"].append(
                    round(statistics.median(ms), 3))
        for key, d in per_cell.items():
            ms = [s["ms"] for s in d["samples"]]
            d["summary"] = summarize(ms)
            print(one_line(key, d["summary"]), flush=True)
        meds = [per_cell[f"kv{k}"]["summary"]["med"] for k in kvs]
        mono = all(meds[i] <= meds[i + 1] + 0.5 for i in range(len(meds) - 1))
        print(f"  单调性(kv 512→8192): {'PASS' if mono else 'FAIL'} {meds}",
              flush=True)
        return {"spec": "b=1 KV 扫描，两趟（升/降序），每趟 7 warmup + 7 计时",
                "cells": per_cell, "monotonic": mono}

    do_phase("P6_kv_sweep_warm", p6)

    sampler.stop_evt.set()
    sampler.join(timeout=3)
    flush(complete=True)
    n_clk = len(sampler.samples)
    print(f"\n[done] phases={len(phases)} clock_samples={n_clk} "
          f"errors={len(sampler.errors)} -> {args.out}", flush=True)
    if had_error:
        print("[done] 有 phase 失败，退出码=1", flush=True)
        return 1
    if n_clk == 0:
        print("[done] 时钟采样完全失败（探针数据仍有效）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
