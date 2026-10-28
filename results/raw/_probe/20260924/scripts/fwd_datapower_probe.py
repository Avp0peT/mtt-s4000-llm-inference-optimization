"""设备侧探针 v6：b=8 那 -4% 的缺口是谁的锅？+ 快态协议下的 harness 复刻。

要回答
------
v5 的 async 块最小值：b1 59.88(-0.1%) / b2 74.28(-1.7%) / b4 83.28(-2.2%) /
b8 109.17(-4.0%) vs P1S3D 真值。缺口**随 batch 单调增大** ⇒ 候选：

(a) **数据内容导致功耗差异**：探针 past 是 torch.zeros，harness 是真实 KV。
    零值的 HBM/SRAM 翻转率最低 ⇒ 动态功耗最低 ⇒ 同频有效性能最好。
    batch 越大算力越密 ⇒ 功耗越接近上限 ⇒ 差异越明显。预测：randn > ones > zeros。
(b) **harness 每步重建 past**（_pack_past 的 pad+cat），探针固定地址。
(c) **态差**（那天 harness 落在稍慢的一档）——若 (a)(b) 都不成立则归此项。

设计
----
A 组（async 块，占空比 100%）：b8 kv2112 的 zeros/randn/zeros 三轮回环
  （zeros 测两遍 ⇒ 用重复包围 randn，把"态漂移"和"数据效应"分开）；
  b1 kv2112 的 randn/zeros 对照（b=1 的缺口只有 -0.1%，预期数据效应很小）。
B 组：MIMIC（harness 协议：逐步 sync、每步重建 mask/pos/由真实输出生长 past、
  每步 cat 成批）——b=8 与 b=1 各 2 轮 × 128 步，past 用 randn 初始化。
  b=8 MIMIC 若落在 ~113.75 ⇒ (a)(b) 合起来解释缺口；若仍是 ~109 ⇒ 归 (c)。

顺带：v5 已排除"占空比/温度/其他进程"三个候选（sync 65% 占空比也能快态；
温度恒 42C；gmi 无其他进程）。v6 不再追"态为什么翻"，只把标定值钉死。
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

T0 = time.monotonic()

import torch
import torch_musa  # noqa: F401

musa = torch.musa


class ClockSampler(threading.Thread):
    def __init__(self, interval: float = 1.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_evt = threading.Event()
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.cmd = ["mthreads-gmi", "-q",
                    "--display=CLOCK,UTILIZATION,TEMPERATURE", "-i", "0"]

    def run(self) -> None:
        while not self.stop_evt.is_set():
            t = round(time.monotonic() - T0, 2)
            try:
                p = subprocess.run(self.cmd, capture_output=True, text=True,
                                   timeout=6)
                out = p.stdout
                gfx = util = temp = None
                m = re.search(r"Graphics\s*:\s*(\d+)\s*MHz", out)
                if m:
                    gfx = int(m.group(1))
                m = re.search(r"Gpu\s*:\s*(\d+)\s*%", out)
                if m:
                    util = int(m.group(1))
                m = re.search(r"GPU Current Temp\s*:\s*(\d+)\s*C", out)
                if m:
                    temp = int(m.group(1))
                if gfx is None and len(self.errors) < 3:
                    self.errors.append(f"parse-fail head={out[:120]!r}")
                self.samples.append({"t": t, "gfx_mhz": gfx,
                                     "gpu_util": util, "temp_c": temp})
            except Exception as e:
                if len(self.errors) < 3:
                    self.errors.append(repr(e))
            self.stop_evt.wait(self.interval)


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


def _fill(shape, kind: str, dtype, device):
    if kind == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    if kind == "ones":
        return torch.ones(shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


def build_past(model, batch: int, kv: int, kind: str, dtype, device):
    """逐层 (key, value)，batch 行等长（uniform ⇒ 无需补位）。"""
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    layers = []
    for _ in range(cfg.num_hidden_layers):
        k = _fill((batch, n_kv, kv, head_dim), kind, dtype, device)
        v = _fill((batch, n_kv, kv, head_dim), kind, dtype, device)
        layers.append((k, v))
    return layers


def make_fwd(model, ids, mask, pos, past):
    def fn():
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask,
                         position_ids=pos,
                         past_key_values=build_past_for_hf(past),
                         use_cache=True)
    return fn


def make_inputs(batch: int, kv: int, dtype, device):
    ids = torch.randint(1, 1000, (batch, 1), dtype=torch.long, device=device)
    mask = torch.ones((batch, kv + 1), dtype=torch.long, device=device)
    pos = torch.full((batch, 1), kv, dtype=torch.long, device=device)
    return ids, mask, pos


def async_chunk(fn, n: int) -> float:
    musa.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    musa.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def sync_series(fn, n: int) -> list[float]:
    musa.synchronize()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        musa.synchronize()
        out.append(round((time.perf_counter() - t0) * 1000.0, 3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out",
                    default="/root/ascend_probe/results/fwd_datapower_probe.json")
    args = ap.parse_args()

    device = torch.device("musa:0")
    dtype = torch.float16
    print(f"[load] {args.model} ...", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    musa.synchronize()
    print(f"[load] done t={time.monotonic()-T0:.1f}s", flush=True)

    sampler = ClockSampler()
    sampler.start()
    phases: dict[str, dict] = {}
    had_error = False
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def flush(complete: bool) -> None:
        payload = {"probe": "fwd_datapower_probe", "model": args.model,
                   "precision": "fp16",
                   "device_name": musa.get_device_name(0),
                   "t_load_done_s": round(time.monotonic() - T0, 1),
                   "complete": complete,
                   "note": "设备侧探针，无 harness 盖章；不得进 EXPERIMENTS 结论行",
                   "clock_samples": list(sampler.samples),
                   "clock_errors": sampler.errors,
                   "phases": phases}
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)

    def do_phase(name, fn_build):
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

    def cell_async(b: int, kv: int, kind: str, nchunk: int = 3,
                   ncalls: int = 20):
        past = build_past(model, b, kv, kind, dtype, device)
        ids, mask, pos = make_inputs(b, kv, dtype, device)
        fn = make_fwd(model, ids, mask, pos, past)
        for _ in range(3):
            fn()
        musa.synchronize()
        chunks = []
        for ci in range(nchunk):
            wall = round(time.monotonic() - T0, 2)
            pc = async_chunk(fn, ncalls)
            chunks.append({"t": wall, "per_call_ms": round(pc, 3)})
            print(f"    chunk {ci}: {pc:.2f} ms/call", flush=True)
        vals = [c["per_call_ms"] for c in chunks]
        out = {"b": b, "kv": kv, "fill": kind, "chunks": chunks,
               "min": round(min(vals), 3), "med": round(statistics.median(vals), 3)}
        print(f"  -> b={b} kv={kv} fill={kind}: min={out['min']:.2f} "
              f"med={out['med']:.2f}", flush=True)
        del past, ids, mask, pos, fn
        return out

    # ---- A：数据内容对照（zeros / randn，b8 与 b1）----
    def a():
        res = {}
        res["b8_zeros_A1"] = cell_async(8, 2112, "zeros")
        res["b8_randn_A2"] = cell_async(8, 2112, "randn")
        res["b8_ones_A3"] = cell_async(8, 2112, "ones")
        res["b8_zeros_A4"] = cell_async(8, 2112, "zeros")
        res["b1_randn_A5"] = cell_async(1, 2112, "randn")
        res["b1_zeros_A6"] = cell_async(1, 2112, "zeros")
        return res

    do_phase("A_data_content", a)

    # ---- B：MIMIC（harness 协议：逐步 sync + 每步重建 + 真实输出生长 past）----
    def mimic(b: int, kind: str, rounds: int = 2, steps: int = 128,
              kv0: int = 2048):
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        recs = []
        for rnd in range(rounds):
            past = build_past(model, b, kv0, kind, dtype, device)
            musa.synchronize()
            for step in range(steps):
                kv = kv0 + step
                ids, mask, pos = make_inputs(b, kv, dtype, device)
                wall = round(time.monotonic() - T0, 2)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(input_ids=ids, attention_mask=mask,
                                position_ids=pos,
                                past_key_values=build_past_for_hf(past),
                                use_cache=True)
                musa.synchronize()
                ms = round((time.perf_counter() - t0) * 1000.0, 3)
                recs.append({"round": rnd, "step": step, "kv": kv,
                             "t": wall, "ms": ms})
                cache = out.past_key_values
                legacy = (cache.to_legacy_cache()
                          if hasattr(cache, "to_legacy_cache") else cache)
                # 逐行切回（等价 harness 的 per-request past）：past[i][l] = (k, v)
                past = [
                    tuple((legacy[l][0][i:i + 1].contiguous(),
                           legacy[l][1][i:i + 1].contiguous())
                          for l in range(n_layers))
                    for i in range(b)
                ]
                # cat 成批张量（等价 _pack_past 的 cat，uniform ⇒ 无 pad）
                packed = []
                for l in range(n_layers):
                    k = torch.cat([p[l][0] for p in past], dim=0)
                    v = torch.cat([p[l][1] for p in past], dim=0)
                    packed.append((k, v))
                past = packed
            ms = [r["ms"] for r in recs if r["round"] == rnd]
            srt = sorted(ms)
            print(f"    round {rnd}: med={statistics.median(ms):.2f} "
                  f"p10={srt[int(0.1*(len(srt)-1))]:.2f} "
                  f"p90={srt[int(0.9*(len(srt)-1))]:.2f}", flush=True)
        ms_all = [r["ms"] for r in recs]
        srt = sorted(ms_all)
        return {"b": b, "fill": kind, "records": recs,
                "summary": {"med": round(statistics.median(ms_all), 3),
                            "p10": round(srt[int(0.1 * (len(srt) - 1))], 3),
                            "p90": round(srt[int(0.9 * (len(srt) - 1))], 3),
                            "min": round(min(ms_all), 3),
                            "max": round(max(ms_all), 3)}}

    def b1_phase():
        r = mimic(8, "randn")
        print(f"  -> b=8 MIMIC(randn): med={r['summary']['med']:.2f} "
              f"p10={r['summary']['p10']:.2f} p90={r['summary']['p90']:.2f} "
              f"(truth 113.75)", flush=True)
        return r

    do_phase("B_mimic_b8_randn", b1_phase)

    def b2_phase():
        r = mimic(1, "randn")
        print(f"  -> b=1 MIMIC(randn): med={r['summary']['med']:.2f} "
              f"p10={r['summary']['p10']:.2f} p90={r['summary']['p90']:.2f} "
              f"(truth 59.94)", flush=True)
        return r

    do_phase("B_mimic_b1_randn", b2_phase)

    sampler.stop_evt.set()
    sampler.join(timeout=3)
    flush(complete=True)
    print(f"\n[done] phases={len(phases)} -> {args.out}", flush=True)
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
