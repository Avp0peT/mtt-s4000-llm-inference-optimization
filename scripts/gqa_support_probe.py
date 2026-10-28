#!/usr/bin/env python3
"""S05 / P4：GQA 支持度能力探针（**真机，MUSA**）。

目的（逐字对应 S05 任务书）
---------------------------
1. 记录**实际** attention 类与内核（配置名或 `enable_gqa` 选项**不算证据**）；
2. 检查 `repeat_kv` / reshape 后的 **storage、分配、拷贝、耗时**；
3. 若不存在显著复制/热点 ⇒ 记录结果结束；若存在 ⇒ 探测当前版本内核的
   **FP16、q_len/kv_len、因果 mask、左 padding** 支持度；
4. 从 decode 最小单元 `Q[B,28,1,128]` / `K,V[B,4,L,128]` 开始，测 B∈{1,2,4} 与多 KV 长度。

⚠️ 纪律（继承既有探针）
----------------------
**这不是 harness 产物**（无 `meta.json`、无标签盖章）⇒ 产物进
`results/raw/_probe/<日期>/`，**不盖四标签、不得进 `EXPERIMENTS.md` 结论表**；
结论只能进 `RESEARCH_LOG.md`。

本地已得的静态事实（本探针要**在真机上核实或推翻**）
--------------------------------------------------
- transformers **4.46.3**（远端）的 `qwen2` 模块有 `repeat_kv`；
- `eager_attention_forward` 里对 K 与 V **各调一次** `repeat_kv`，
  `num_key_value_groups = 28/4 = 7`；
- `repeat_kv` 用 `expand` + `reshape`，`reshape` 在非连续张量上会**真实拷贝**
  ⇒ 理论量 `B×28×L×128×2B×L层×2(KV)`，`short_chat` decode 步（B=8, L≈600）≈ **1.9 GB/步**。
- ⚠️ 本机 transformers 是 4.50.0、远端是 4.46.3 ⇒ **本地版本不能替代远端核实**。

用法（在远端跑）::

    python3 scripts/gqa_support_probe.py --out results/raw/_probe/<日期>/gqa_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
N_Q_HEADS = 28
N_KV_HEADS = 4
HEAD_DIM = 128


def _sync() -> None:
    """设备同步。MUSA 走 torch.musa；拿不到就退回 cuda/cpu（并如实报告）。"""
    try:
        import torch_musa  # noqa: F401

        torch.musa.synchronize()
        return
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _device() -> torch.device:
    try:
        import torch_musa  # noqa: F401

        if torch.musa.is_available():
            return torch.device("musa:0")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _timed(fn, *, warmup: int = 3, iters: int = 10) -> dict:
    """同步计时：warmup 后取 iters 次的中位/最小（设备时间）。"""
    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {"min_ms": ts[0], "median_ms": ts[len(ts) // 2], "max_ms": ts[-1], "n": len(ts)}


def _env() -> dict:
    info = {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception as exc:
        info["transformers"] = f"ERR {type(exc).__name__}"
    try:
        import torch_musa

        info["torch_musa"] = getattr(torch_musa, "__version__", "unknown")
    except Exception as exc:
        info["torch_musa"] = f"ERR {type(exc).__name__}"
    try:
        info["musa_driver"] = torch.musa.get_device_properties(0).name
    except Exception:
        pass
    info["device_used"] = str(_device())
    return info


def probe_attention_class() -> dict:
    """① 记录**实际** attention 类与函数（不是配置名）。"""
    out: dict = {}
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq

        out["module_file"] = getattr(mq, "__file__", None)
        out["has_repeat_kv"] = hasattr(mq, "repeat_kv")
        out["has_eager_attention_forward"] = hasattr(mq, "eager_attention_forward")
        out["attention_class"] = [
            n for n in dir(mq) if n.endswith("Attention") and not n.startswith("_")
        ]
        # 源码证据：eager_attention_forward 里是否**真的**调了 repeat_kv。
        import inspect

        if out["has_eager_attention_forward"]:
            src = inspect.getsource(mq.eager_attention_forward)
            out["eager_calls_repeat_kv"] = src.count("repeat_kv(")
            out["eager_src_sha_head"] = __import__("hashlib").sha256(
                src.encode("utf-8")
            ).hexdigest()[:16]
        if out["has_repeat_kv"]:
            out["repeat_kv_src"] = inspect.getsource(mq.repeat_kv)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def probe_attention_backend(model) -> dict:
    """② 模型**实际**用的 attention 实现（`_attn_implementation`）。"""
    info: dict = {}
    try:
        cfg = model.config
        info["config_attn_impl"] = getattr(cfg, "_attn_implementation", None)
        info["use_sliding_window"] = getattr(cfg, "use_sliding_window", None)
        info["num_key_value_groups"] = N_Q_HEADS // N_KV_HEADS
        # 逐层记录 attention 模块的类名（实际实例化的那个）。
        layers = getattr(getattr(model, "model", None), "layers", [])
        info["n_layers"] = len(layers)
        if layers:
            attn = layers[0].self_attn
            info["attn_module_class"] = type(attn).__module__ + "." + type(attn).__name__
            info["attn_has_num_kv_groups"] = hasattr(attn, "num_key_value_groups")
            info["attn_num_kv_groups"] = getattr(attn, "num_key_value_groups", None)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_repeat_kv_cost(device: torch.device) -> dict:
    """③ `repeat_kv` 的复制量与时延（用真实形状，直接调它）。

    这里**直接调用** transformers 的 `repeat_kv`（若存在），而不是自己实现：
    自己写一份就测不到真实实现是否为 view / copy。
    """
    rows = []
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    if not hasattr(mq, "repeat_kv"):
        return {"error": "transformers 里没有 qwen2.repeat_kv"}

    repeat_kv = mq.repeat_kv
    dtype = torch.float16
    for B in (1, 2, 4, 8):
        for L in (1, 64, 512, 1024):
            K = torch.randn(B, N_KV_HEADS, L, HEAD_DIM, dtype=dtype, device=device)
            V = torch.randn(B, N_KV_HEADS, L, HEAD_DIM, dtype=dtype, device=device)
            n_rep = N_Q_HEADS // N_KV_HEADS

            def run():
                a = repeat_kv(K, n_rep)
                b = repeat_kv(V, n_rep)
                return a, b

            a, b = run()
            # ★ storage 证据：reshape 后是否真的分配了新存储（而非共享视图）。
            same_storage_k = a.data_ptr() == K.data_ptr()
            contiguous_k = a.is_contiguous()
            copied_bytes = a.numel() * a.element_size() * 2  # K+V
            t = _timed(run)
            rows.append(
                {
                    "B": B,
                    "L": L,
                    "n_rep": n_rep,
                    "out_shape": list(a.shape),
                    "shares_storage_with_input": same_storage_k,
                    "out_is_contiguous": contiguous_k,
                    "copied_bytes_kv": copied_bytes,
                    "copy_per_step_mb": round(copied_bytes / 1e6, 3),
                    "min_ms": round(t["min_ms"], 4),
                    "median_ms": round(t["median_ms"], 4),
                    "eff_gbs": round((copied_bytes / 1e9) / (t["min_ms"] / 1e3), 2)
                    if t["min_ms"] > 0
                    else None,
                }
            )
            del K, V, a, b
    return {"rows": rows}


def probe_sdpa_gqa(device: torch.device) -> dict:
    """④ MUSA 原生 GQA：`F.scaled_dot_product_attention(..., enable_gqa=True)` 是否可用。

    判定"可用"必须**同时**满足：不抛异常 **且** 与手工 repeat_kv 的结果数值一致。
    只看"不抛异常"会把"静默忽略 enable_gqa"当成支持。
    """
    res: dict = {"tested": True}
    dtype = torch.float16
    B, L, qlen = 4, 512, 1
    torch.manual_seed(0)
    Q = torch.randn(B, N_Q_HEADS, qlen, HEAD_DIM, dtype=dtype, device=device)
    K = torch.randn(B, N_KV_HEADS, L, HEAD_DIM, dtype=dtype, device=device)
    V = torch.randn(B, N_KV_HEADS, L, HEAD_DIM, dtype=dtype, device=device)
    scale = HEAD_DIM ** -0.5

    # 参照：手工 repeat_kv + SDPA（enable_gqa=False）
    try:
        from transformers.models.qwen2 import modeling_qwen2 as mq

        Kr = mq.repeat_kv(K, N_Q_HEADS // N_KV_HEADS)
        Vr = mq.repeat_kv(V, N_Q_HEADS // N_KV_HEADS)
    except Exception:
        Kr = K.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=1)
        Vr = V.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=1)
    try:
        ref = F.scaled_dot_product_attention(Q, Kr, Vr, scale=scale)
        res["ref_ok"] = True
    except Exception as exc:
        res["ref_ok"] = False
        res["ref_error"] = f"{type(exc).__name__}: {exc}"
        ref = None

    # 被测：enable_gqa=True
    try:
        got = F.scaled_dot_product_attention(Q, K, V, scale=scale, enable_gqa=True)
        res["enable_gqa_accepted"] = True
        if ref is not None:
            diff = (got.float() - ref.float()).abs().max().item()
            res["max_abs_diff_vs_repeat_kv"] = diff
            res["numerically_equal"] = diff < 1e-2  # fp16 容差
    except Exception as exc:
        res["enable_gqa_accepted"] = False
        res["enable_gqa_error"] = f"{type(exc).__name__}: {exc}"
        # 退一步：位置参数形式（部分版本 enable_gqa 不是 kwarg）
        try:
            got2 = F.scaled_dot_product_attention(Q, K, V, None, 0.0, False, scale, True)
            res["enable_gqa_positional_accepted"] = True
            if ref is not None:
                res["pos_max_abs_diff"] = (got2.float() - ref.float()).abs().max().item()
        except Exception as exc2:
            res["enable_gqa_positional_error"] = f"{type(exc2).__name__}: {exc2}"

    # 时延对照（若 enable_gqa 可用）
    if res.get("enable_gqa_accepted"):

        def run_gqa():
            F.scaled_dot_product_attention(Q, K, V, scale=scale, enable_gqa=True)

        def run_rep():
            F.scaled_dot_product_attention(
                Q, mq.repeat_kv(K, 7), mq.repeat_kv(V, 7), scale=scale
            )

        res["timing_gqa"] = _timed(run_gqa)
        try:
            res["timing_repeat_kv"] = _timed(run_rep)
        except Exception as exc:
            res["timing_repeat_kv_error"] = f"{type(exc).__name__}: {exc}"
    return res


def probe_attention_variants(device: torch.device) -> dict:
    """⑤ 完整 attention 的变体支持度：FP16 / q_len / kv_len / 因果 mask / 左 padding。"""
    out: dict = {"cases": []}
    dtype = torch.float16
    B = 2
    for qlen, kvlen, use_causal, use_leftpad in [
        (1, 512, False, False),   # 纯 decode
        (1, 512, True, False),
        (1, 512, False, True),    # 左 padding（mask 加在左侧）
        (8, 512, True, False),    # chunk prefill
        (128, 512, True, False),
        (1, 2048, False, False),  # 长 KV
    ]:
        case: dict = {
            "batch": B,
            "q_len": qlen,
            "kv_len": kvlen,
            "causal": use_causal,
            "left_pad": use_leftpad,
            "dtype": "fp16",
        }
        try:
            Q = torch.randn(B, N_Q_HEADS, qlen, HEAD_DIM, dtype=dtype, device=device)
            K = torch.randn(B, N_KV_HEADS, kvlen, HEAD_DIM, dtype=dtype, device=device)
            V = torch.randn(B, N_KV_HEADS, kvlen, HEAD_DIM, dtype=dtype, device=device)
            mask = None
            if use_causal or use_leftpad:
                mask = torch.zeros(B, 1, qlen, kvlen, dtype=dtype, device=device)
                if use_causal:
                    for i in range(qlen):
                        mask[:, :, i, : kvlen - qlen + i + 1] = 1.0
                if use_leftpad:
                    mask[:, :, :, :64] = 0.0   # 左 64 列视为 padding
                    if not use_causal:
                        mask[:, :, :, 64:] = 1.0
            got = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=mask, scale=HEAD_DIM ** -0.5, enable_gqa=True
            )
            case["enable_gqa_ok"] = True
            case["out_shape"] = list(got.shape)
        except Exception as exc:
            case["enable_gqa_ok"] = False
            case["enable_gqa_error"] = f"{type(exc).__name__}: {exc}"
        # 对照：走 repeat_kv 的手工路径
        try:
            Kr = K.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=1)
            Vr = V.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=1)
            got2 = F.scaled_dot_product_attention(Q, Kr, Vr, attn_mask=mask,
                                                  scale=HEAD_DIM ** -0.5)
            case["repeat_kv_ok"] = True
            case["out_shape_repeat_kv"] = list(got2.shape)
        except Exception as exc:
            case["repeat_kv_ok"] = False
            case["repeat_kv_error"] = f"{type(exc).__name__}: {exc}"
        out["cases"].append(case)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="产物路径（JSON）")
    ap.add_argument("--skip-model", action="store_true",
                    help="不加载 7B 模型（只做算子级探针，快得多）")
    args = ap.parse_args()

    device = _device()
    result: dict = {
        "protocol": "gqa-support-probe-v1",
        "session": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "env": _env(),
        "model_revision_expected": MODEL_REVISION,
        "shapes": {"n_q_heads": N_Q_HEADS, "n_kv_heads": N_KV_HEADS, "head_dim": HEAD_DIM},
        "label": "PROBE_NOT_HARNESS",
        "note": "探针产物：不盖四标签、不进 EXPERIMENTS.md 结论表",
    }
    print("[1/5] attention 类与函数 ...")
    result["attention_class"] = probe_attention_class()
    print("[2/5] repeat_kv 复制量与时延 ...")
    result["repeat_kv_cost"] = probe_repeat_kv_cost(device)
    print("[3/5] SDPA 原生 GQA（enable_gqa）...")
    result["sdpa_gqa"] = probe_sdpa_gqa(device)
    print("[4/5] attention 变体支持度 ...")
    result["variants"] = probe_attention_variants(device)

    if not args.skip_model:
        print("[5/5] 装载模型并记录实际 attention 后端 ...")
        try:
            from transformers import AutoModelForCausalLM

            t0 = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, revision=MODEL_REVISION, torch_dtype=torch.float16
            ).to(device).eval()
            result["model_load_s"] = round(time.perf_counter() - t0, 2)
            result["attention_backend"] = probe_attention_backend(model)
            del model
        except Exception as exc:
            result["attention_backend"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        result["attention_backend"] = {"skipped": True}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"写出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
