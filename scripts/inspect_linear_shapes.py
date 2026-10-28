#!/usr/bin/env python
"""S03/P3 前置：**从实际模型**读出线性投影的形状 / bias / dtype / stride。

要回答的问题（`REFERENCE_8_PLANS.md` §P3「实现任务 1」）：
本模型里 Q/K/V 与 gate/up 到底长什么样？能不能拼、拼成多宽？切出来的片
需不需要 contiguous？——**都必须从模型现读**，不能按配置名推断。

用法::

    # 本地自检（微型随机模型；不需要权重、不需要联网）
    python scripts/inspect_linear_shapes.py --tiny

    # 真机（MTT S4000）：读实际 Qwen2.5-7B-Instruct
    python scripts/inspect_linear_shapes.py --model Qwen/Qwen2.5-7B-Instruct \
        --revision a09a35458c702b33eeacc393d103063234e8bc28 --device cpu

⚠️ 本地没有 7B 权重 ⇒ 真机那几个数字是 `REMOTE_PENDING`；`--tiny` 只证明
**脚本本身可用**，不代表真机形状。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: 交接包记录的 Qwen2.5-7B-Instruct 期望形状（`REFERENCE_8_PLANS.md` §3）。
DECLARED_QWEN25_7B = {
    "q_out": 3584, "k_out": 512, "v_out": 512, "qkv_fused": 4608,
    "gate_out": 18944, "up_out": 18944, "gate_up_fused": 37888,
    "hidden": 3584, "n_layers": 28,
}


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: PLC0415
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=48,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    m = Qwen2ForCausalLM(cfg)
    m.eval()
    return m


def inspect(model) -> dict:
    import linear_fusion as LF  # noqa: PLC0415

    layers = LF.iter_layers(model)
    rows: list[dict] = []
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        mlp = getattr(layer, "mlp", None)
        row: dict = {"layer": i}
        for tag, owner, names in (("qkv", attn, ("q_proj", "k_proj", "v_proj")),
                                  ("gate_up", mlp, ("gate_proj", "up_proj"))):
            if owner is None or not all(hasattr(owner, n) for n in names):
                row[tag] = None
                continue
            mods = [getattr(owner, n) for n in names]
            row[tag] = {
                "modules": list(names),
                "out_features": [int(m.weight.shape[0]) for m in mods],
                "in_features": int(mods[0].weight.shape[1]),
                "fused_out_features": int(sum(m.weight.shape[0] for m in mods)),
                "dtype": str(mods[0].weight.dtype),
                "stride": list(mods[0].weight.stride()),
                "has_bias": [m.bias is not None for m in mods],
                "contiguous": [bool(m.weight.is_contiguous()) for m in mods],
            }
        rows.append(row)

    # 真正装一次，拿**融合后**的权重形状/stride（这是最直接的证据）。
    manifest = LF.install_fusions(model, fuse_qkv=True, fuse_gate_up=True)
    try:
        desc = manifest.describe()
    finally:
        LF.uninstall_fusions(manifest)
    return {"n_layers": len(layers), "per_layer": rows[:3], "fusion": desc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tiny", action="store_true", help="用微型随机模型自检")
    ap.add_argument("--model", default=None, help="HF repo id 或本地路径")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.tiny:
        model = _tiny_model()
    elif args.model:
        from transformers import AutoModelForCausalLM  # noqa: PLC0415
        kwargs: dict = {"torch_dtype": "auto"}
        if args.revision:
            kwargs["revision"] = args.revision
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        model.to(args.device)
        model.eval()
    else:
        ap.error("给出 --tiny 或 --model")

    out = inspect(model)
    first = out["per_layer"][0] if out["per_layer"] else {}
    qkv = (first.get("qkv") or {})
    gu = (first.get("gate_up") or {})
    summary = {
        "n_layers": out["n_layers"],
        "q_out": (qkv.get("out_features") or [None])[0],
        "q_in": qkv.get("in_features"),
        "fused_qkv_out": qkv.get("fused_out_features"),
        "gate_out": (gu.get("out_features") or [None])[0],
        "fused_gate_up_out": gu.get("fused_out_features"),
        "dtype": qkv.get("dtype"),
        "stride": qkv.get("stride"),
        "has_bias": qkv.get("has_bias"),
        "installed_groups": out["fusion"]["installed_groups"],
    }
    if not args.tiny and out["n_layers"] == DECLARED_QWEN25_7B["n_layers"]:
        # 只比对**同名量**：`fused_qkv_out` vs 声明的 4608、`fused_gate_up_out` vs 37888。
        checks = {
            "hidden": summary.get("q_in"),
            "fused_qkv_out": summary.get("fused_qkv_out"),
            "fused_gate_up_out": summary.get("fused_gate_up_out"),
        }
        summary["matches_declared_qwen25_7b"] = {
            k: {"got": checks.get(k), "expected": v, "ok": checks.get(k) == v}
            for k, v in DECLARED_QWEN25_7B.items() if k in checks
        }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"写出 {args.json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
