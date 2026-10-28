"""设备侧探针：拆解 S4000 上 Qwen2.5-7B fp16 的每步前向成本结构。

**这不是 harness 产物**（没有 meta.json、没有标签盖章）⇒ 落在 adhoc/，
结论**不得**进 EXPERIMENTS.md 的结论行，只能作为 RESEARCH_LOG 的记录。

要回答的问题
------------
P1S3D 里 `fwd_ms/n_forwards` 从 c1 的 **79.9 ms** 单调升到 c8 的 **168.1 ms**
（**2.1x**）。这 2.1x 是什么组成的？候选：

(a) **batch 维度本身**：decode 时 batch 从 1 到 8 ⇒ 计算量线性增长？
(b) **KV 长度**：attention 是 O(len)。
(c) **长度差异导致的 padding 浪费**：补位区不参与注意力但仍要走满张量。
(d) **形状导致的 MUSA 内核效率**：小矩阵 / 非规整形状上达不到峰值。

与 harness 的对齐（对齐了才能跟 79.9/168.1 对照）
----
- 前向调用：``no_grad`` + ``use_cache=True`` + 显式 ``past_key_values``
  + 逐行自己的 ``position_ids`` + 左补位 ``attention_mask``。
- **HF 缓存对象每次前向重建**（real_backend.py 每次都调 _build_past_for_hf）。
  ⚠️ 踩过：把 DynamicCache 建在循环外复用，``use_cache=True`` 会**原地追加**
  KV ⇒ 重复计时中 cache 长度越滚越长，与固定长度的 2D mask 失配 ⇒
  均匀行（全 1 mask）侥幸不炸、**带左补位的行在 MuDNN SDPA 内核处炸**
  （``MuDNNFlashSDPAFwd MUDNN failed in: Run SDPA``）。单独复跑同形状却通过——
  所以第一反应"内核不支持补位"是**错的**，那是跨调用状态污染。
- 混合步的形态：real_backend **按 n_new 分组、组间串行各自前向**
  （一次"步"= decode 组前向 + prefill 组前向，而非一个变长 n_new 的批）。
  实验 E 按这个形态测"加和性"。

计时口径：``torch.musa.synchronize()`` 后打点 = 纯设备段，与 harness 的
``fwd_ms`` 同义（都不含 pack / sample / readback）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch
import torch_musa  # noqa: F401  —— 必须先 import，否则 torch.musa 不存在

# ⚠️ 不能用 `import torch.musa as musa`：import 语句在**模块加载时**解析，
# 而 `torch.musa` 是 torch_musa 注册出来的**子模块**；即使上一行已 import torch_musa，
# `import torch.musa` 仍会走 importlib 去找文件并在某些布局下失败。
# 用属性访问才稳定（踩过一次，报 ModuleNotFoundError: No module named 'torch.musa'）。
musa = torch.musa


def cache_needs_object() -> bool:
    """复刻 real_backend._cache_needs_object 的能力探测。"""
    try:
        from transformers import cache_utils
    except Exception:
        return False
    dyn = getattr(cache_utils, "DynamicCache", None)
    return dyn is not None and hasattr(dyn, "from_legacy_cache")


def build_past_for_hf(kv):
    """**每次前向都要重建**（等价 real_backend._build_past_for_hf）。

    from_legacy_cache 只包一层引用，不拷贝数据，重建代价可忽略；
    而**复用**会让 use_cache=True 的原地追加污染下一次计时（见模块 docstring）。
    """
    if not kv:
        return ()
    legacy = tuple((k, v) for k, v in kv)
    if cache_needs_object():
        from transformers.cache_utils import DynamicCache
        return DynamicCache.from_legacy_cache(legacy)
    return legacy


def dev_sync() -> None:
    musa.synchronize()


def timed(fn, *, warmup: int, repeats: int):
    """warmup 不计入（内核 autotune / 首次分配），计时取 repeats 的中位数。"""
    for _ in range(warmup):
        fn()
    dev_sync()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dev_sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


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
            if L == 0:
                z = torch.zeros(1, n_kv, 0, head_dim, dtype=dtype, device=device)
                rows_k.append(z)
                rows_v.append(z)
                continue
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
    """造一批前向输入。n_new = 本次喂入的新 token 数（decode=1，prefill=chunk）。"""
    ids = torch.randint(1, 1000, (batch, n_new), dtype=torch.long, device=device)
    past_max = max(kv_lens)
    mask = torch.ones((batch, past_max + n_new), dtype=torch.long, device=device)
    for i, L in enumerate(kv_lens):
        pad = past_max - L
        if pad:
            mask[i, :pad] = 0
    pos = torch.stack([
        torch.arange(L, L + n_new, dtype=torch.long, device=device) for L in kv_lens
    ])
    return ids, mask, pos


def make_fwd(model, ids, mask, pos, past):
    """包一个与 harness 对齐的前向：no_grad + use_cache=True + **缓存每次重建**。"""
    def fn():
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=mask,
                         position_ids=pos,
                         past_key_values=build_past_for_hf(past),
                         use_cache=True)
    return fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="/root/ascend_probe/results/fwd_cost_profile.json")
    args = ap.parse_args()

    device = torch.device("musa:0")
    dtype = torch.float16
    print(f"[load] {args.model} ...", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    dev_sync()
    cfg = model.config
    print(f"[load] layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
          f"kv_heads={getattr(cfg,'num_key_value_heads',cfg.num_attention_heads)} "
          f"hidden={cfg.hidden_size}", flush=True)

    results = []
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def flush_results(complete: bool) -> None:
        """每步增量落盘：中断不丢已测数据（踩过：只在末尾写 ⇒ OOM 后全丢）。"""
        payload = {
            "probe": "fwd_cost_profile",
            "model": args.model,
            "precision": "fp16",
            "device": "musa:0",
            "repeats": args.repeats,
            "warmup": args.warmup,
            "complete": complete,
            "note": "设备侧探针，无 harness 盖章；不得进 EXPERIMENTS 结论行",
            "torch": torch.__version__,
            "torch_musa": getattr(torch_musa, "__version__", None),
            "device_name": musa.get_device_name(0),
            "results": results,
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def reclaim(tag: str) -> None:
        musa.empty_cache()
        free, total = musa.mem_get_info()
        print(f"    [mem] {tag}: free={free/1024**3:.2f} GiB / {total/1024**3:.2f} GiB",
              flush=True)

    def rec(tag, batch, kv_lens, n_new, med, allt, extra=None):
        row = {
            "tag": tag, "batch": batch, "n_new": n_new,
            "kv_max": max(kv_lens), "kv_min": min(kv_lens), "kv_lens": kv_lens,
            "ms_median": round(med * 1000, 3),
            "ms_all": [round(t * 1000, 3) for t in allt],
        }
        if extra:
            row.update(extra)
        results.append(row)
        spread = (max(kv_lens) - min(kv_lens)) if len(set(kv_lens)) > 1 else 0
        print(f"  {tag:30s} b={batch} n_new={n_new:5d} "
              f"kv={min(kv_lens)}..{max(kv_lens)} (spread {spread:4d})  "
              f"{med*1000:8.2f} ms", flush=True)
        flush_results(complete=False)

    # ---------- A：pure decode，batch 1/2/4/8，固定等长 KV=2048 ----------
    # 回答 (a)：batch 维度本身的成本（无 padding、无长度差异）。
    print("\n[A] pure decode：batch 扫描（等长 KV=2048）", flush=True)
    for b in (1, 2, 4, 8):
        kv = [2048] * b
        past, _ = build_past(model, b, kv, dtype, device)
        ids, mask, pos = make_batch(model, b, kv, 1, dtype, device)
        fn = make_fwd(model, ids, mask, pos, past)
        med, allt = timed(fn, warmup=args.warmup, repeats=args.repeats)
        rec("A_decode_batch", b, kv, 1, med, allt)
        del past, ids, mask, pos, fn
        reclaim("A")

    # ---------- B：pure decode，batch=1，KV 长度扫描 ----------
    # 回答 (b)：attention O(len)；与 (a) 分离后能看每步固定开销的地板。
    print("\n[B] pure decode：KV 长度扫描（batch=1）", flush=True)
    for L in (512, 1024, 2048, 4096, 8192):
        kv = [L]
        past, _ = build_past(model, 1, kv, dtype, device)
        ids, mask, pos = make_batch(model, 1, kv, 1, dtype, device)
        fn = make_fwd(model, ids, mask, pos, past)
        med, allt = timed(fn, warmup=args.warmup, repeats=args.repeats)
        rec("B_decode_kvlen", 1, kv, 1, med, allt)
        del past, ids, mask, pos, fn
        reclaim("B")

    # ---------- C：pure decode，batch=8，长度差异（padding 浪费）----------
    # 回答 (c)：同 kv_max=2048，行内有短行 ⇒ 补位区浪费多少。
    # kv_sum 保持 2048*8=16384 的只有 uniform；其余按"真实行数×真实长度"计。
    print("\n[C] pure decode：长度差异（batch=8, kv_max=2048）", flush=True)
    for spread, kv in (
        ("uniform", [2048] * 8),
        ("half1024", [2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024]),
        ("half512", [2048, 2048, 2048, 2048, 512, 512, 512, 512]),
        ("one512", [2048] * 7 + [512]),
    ):
        past, _ = build_past(model, 8, kv, dtype, device)
        ids, mask, pos = make_batch(model, 8, kv, 1, dtype, device)
        fn = make_fwd(model, ids, mask, pos, past)
        med, allt = timed(fn, warmup=args.warmup, repeats=args.repeats)
        rec(f"C_spread_{spread}", 8, kv, 1, med, allt,
            extra={"kv_sum": sum(kv)})
        del past, ids, mask, pos, fn
        reclaim("C")

    # ---------- D：prefill（无历史），batch×chunk 扫描 ----------
    # 回答 prefill 的成本面：f(batch, chunk)。真实步里 prefill 组可能带多行。
    print("\n[D] prefill（无历史）：batch × chunk", flush=True)
    for b in (1, 2, 4):
        for chunk in (512, 1024, 2048):
            ids = torch.randint(1, 1000, (b, chunk), dtype=torch.long, device=device)
            mask = torch.ones((b, chunk), dtype=torch.long, device=device)
            pos = torch.arange(chunk, dtype=torch.long, device=device)
            pos = pos.unsqueeze(0).expand(b, -1).contiguous()
            fn = make_fwd(model, ids, mask, pos, None)
            med, allt = timed(fn, warmup=args.warmup, repeats=args.repeats)
            rec("D_prefill", b, [0], chunk, med, allt,
                extra={"prefill_tokens": b * chunk})
            del ids, mask, pos, fn
            reclaim("D")

    # ---------- E：混合步的加和性（decode 组 + prefill 组串行）----------
    # real_backend 按 n_new 分组、组间串行 ⇒ 一步的成本 = Σ(组成本)？
    # 若成立，成本模型只需 per-组 函数 f(行数, n_new, kv 分布)，不需要"混合项"。
    print("\n[E] 混合步加和性：decode 组 + prefill 组（串行，两次前向）", flush=True)
    for tag, dec_b, dec_kv, pre_b, chunk in (
        ("E_c4like_chunk1024", 4, [2048, 2048, 1024, 1024], 1, 1024),
        ("E_c4like_chunk512", 4, [2048, 2048, 1024, 1024], 1, 512),
        ("E_c8like_chunk512", 7, [2048] * 3 + [1024] * 4, 1, 512),
    ):
        # decode 组
        past_d, _ = build_past(model, dec_b, dec_kv, dtype, device)
        ids_d, mask_d, pos_d = make_batch(model, dec_b, dec_kv, 1, dtype, device)
        # prefill 组（无历史）
        ids_p = torch.randint(1, 1000, (pre_b, chunk), dtype=torch.long, device=device)
        mask_p = torch.ones((pre_b, chunk), dtype=torch.long, device=device)
        pos_p = torch.arange(chunk, dtype=torch.long, device=device)
        pos_p = pos_p.unsqueeze(0).expand(pre_b, -1).contiguous()

        fn_d = make_fwd(model, ids_d, mask_d, pos_d, past_d)
        fn_p = make_fwd(model, ids_p, mask_p, pos_p, None)

        def step(fn_d=fn_d, fn_p=fn_p):
            fn_d()
            fn_p()

        med, allt = timed(step, warmup=args.warmup, repeats=args.repeats)
        # 分量也测一遍（加和性检查的直接对照）
        med_d, _ = timed(fn_d, warmup=1, repeats=args.repeats)
        med_p, _ = timed(fn_p, warmup=1, repeats=args.repeats)
        rec(tag, dec_b, dec_kv, chunk, med, allt, extra={
            "decode_ms": round(med_d * 1000, 3),
            "prefill_ms": round(med_p * 1000, 3),
            "sum_ms": round((med_d + med_p) * 1000, 3),
        })
        del past_d, ids_d, mask_d, pos_d, ids_p, mask_p, pos_p, fn_d, fn_p
        reclaim("E")

    flush_results(complete=True)
    print(f"\n[done] {len(results)} 组 → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
