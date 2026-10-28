#!/usr/bin/env python3
"""把标定结果**固化成模型文件**（第 11→12 步的交界，协议 `cost-calib-v1`）。

为什么要固化、而不是每次现拟合
------------------------------
控制器的选型要在**毫秒级**完成，现拟合既慢又不可复核；更要紧的是：**现拟合的模型没法比对指纹**
—— 而验收要求"模型带硬件/软件/模型版本/精度和标定范围指纹，**失配不能静默使用**"。
⇒ 固化成一个带范围的 JSON，加载时逐项比对，不符即拒。

三件必须在文件里写死的东西
--------------------------
1. **标定范围**（`calibrated_range`）：每个 `batch` 下 `prefill_tok` / `kv_len` 的**实测边界**。
   控制器**越界即回退**（不得外推 —— 第 11 步实测外推到 `kv=2048` 时低估 p95 达 23.11%）。
2. **安全余量**（`safety_margin`）：取**留出形状上的低估误差 p95**（第 11 步实测 1.77%）。
   余量必须从"低估尾部"推，不能拍脑袋 —— 低估会让控制器以为来得及。
3. **指纹**（`fingerprint`）：与产物里 `expected` 同形；加载时与 `observed` 逐项比对。

关于 `n_forwards`：它**不是**范围的一部分，而是一个**乘子**。
依据：实测 `n_forwards` 1→4 的**单前向成本完全相同**（262.7 ms 四个点逐位一致）⇒ 加和性成立。
⚠️ 这条是**有实据的**假设（不是"看起来像"），但也要在文件里注明依据。

用法::

    python scripts/export_cost_model.py --eval results/processed/costcalib_eval.json \\
        --artifacts "results/raw/_costcalib/*.json" --out configs/cost_calib/model_v1.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from costcalib_fit import _ols  # noqa: E402  —— 复用同一份最小二乘，别再写一遍

MODEL_VERSION = "cost-model-v1"


def fit_per_batch(records: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    """按 `batch` 分段，每段拟合 `[1, prefill_tok, kv_len] → per-forward ms`。

    ⚠️ 用**全部**标定记录拟合（生产模型该用上全部数据）；留出集只用于**误差估计**，
    那件事已经在 `costcalib_fit.py` 里做过、报过了。**不得**拿留出误差去反过来调系数。
    """
    by_b: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_b.setdefault(str(int(r["batch"] or 1)), []).append(r)
    return {
        b: _ols([[1.0, float(r["prefill_tok"] or 0), float(r["kv_len"] or 0)] for r in rows],
                [float(r["actual_ms"]) for r in rows])
        for b, rows in by_b.items()
    }


def calibrated_range(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """逐 `batch` 记录 `prefill_tok` / `kv_len` 的**实测**边界，以及出现过的组合数。"""
    out: dict[str, Any] = {}
    for r in records:
        b = str(int(r["batch"] or 1))
        e = out.setdefault(b, {"prefill_tok": [None, None], "kv_len": [None, None], "n_points": 0})
        e["n_points"] += 1
        for axis, val in (("prefill_tok", int(r["prefill_tok"] or 0)),
                          ("kv_len", int(r["kv_len"] or 0))):
            lo, hi = e[axis]
            e[axis] = [val if lo is None else min(lo, val),
                       val if hi is None else max(hi, val)]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="导出固化的成本模型文件")
    ap.add_argument("--eval", required=True, help="costcalib_fit.py --json 的产物（取误差与指纹）")
    ap.add_argument("--artifacts", nargs="+", required=True, help="标定产物（取指纹与记录）")
    ap.add_argument("--out", required=True)
    # ⚠️ 为什么这几件事必须由**导出器**写，而不是事后用脚本改 JSON：
    # `cost_model.py` 解析模式只认 `payload["default_mode"]`（配置里**没有**这个键），
    # 所以"这批实验到底用了 segmented 还是 lookup"**只**取决于模型文件里的那一个字段。
    # 实测踩过：链里用 heredoc 事后改 `default_mode`，改失败（键名写错/文件没写进去）
    # **不会报错** —— 模型静默按 `segmented` 跑，而覆盖率闸门显式传 `--replay-mode lookup`，
    # **照样通过** ⇒ 闸门与实跑口径脱节，产物看起来完全正常。
    # ⇒ 把选择变成**导出时的入参**：值随文件一起冻结、可复核、有默认（= 旧行为）。
    ap.add_argument("--default-mode", choices=("segmented", "lookup"), default="segmented",
                    help="写进模型的 default_mode；控制器只读这个键（默认=旧行为）")
    ap.add_argument("--lookup-bucket-pf-step", type=int, default=0,
                    help="查表分桶：prefill_tok 方向的桶宽（0=精确键，默认）")
    ap.add_argument("--lookup-bucket-kv-step", type=int, default=0,
                    help="查表分桶：kv_len 方向的桶宽（0=精确键，默认）")
    ap.add_argument("--mixed-shape-mode", default=None,
                    help="记录 mixed 形状的**语义口径**（如 real_mixed）；不传则不写这个键")
    args = ap.parse_args(argv)
    if args.lookup_bucket_pf_step < 0 or args.lookup_bucket_kv_step < 0:
        print("✗ 桶宽不得为负", file=sys.stderr)
        return 2

    ev = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    records = ev.get("records") or []
    if not records:
        print("✗ 评估产物里没有 records ⇒ 无法拟合", file=sys.stderr)
        return 2

    paths: list[str] = []
    for a in args.artifacts:
        paths.extend(sorted(glob.glob(a)))
    arts = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    if not arts:
        print("✗ 找不到标定产物 ⇒ 指纹没法写进模型（失配就无从判定）", file=sys.stderr)
        return 2

    protocols = {a.get("protocol_version") for a in arts}
    grid_hashes = {a.get("grid_sha256") for a in arts}
    if len(protocols) != 1 or len(grid_hashes) != 1:
        print(f"✗ 产物的协议/网格不一致：protocols={protocols} grid_hashes={grid_hashes}",
              file=sys.stderr)
        return 2

    # 安全余量 = **留出形状上的低估 p95**（从评估产物取，不重算 —— 免得换个口径）
    under_p95 = None
    for title, ctx in (ev.get("contexts") or {}).items():
        if title.startswith("①"):
            under_p95 = (ctx.get("分段模型(按batch)") or {}).get("underest_p95")
    if under_p95 is None:
        print("✗ 取不到'留出形状上的低估 p95' ⇒ 无法定安全余量（不得拍脑袋）", file=sys.stderr)
        return 2

    # ---- 查表（"查表 + 保守回退"分支的载体）----------------------------------
    # 键 = (batch, prefill_tok, kv_len) 的**精确**形状；值取该键实测的**最大值**
    # —— 保守方向：宁可高估（控制器更保守），**绝不低估**（那会错误接纳过大的 prefill）。
    # ⚠️ 查表**不插值**：未命中的形状返回 None（回退），代价是**覆盖率**
    # ⇒ 覆盖率必须与误差**并排报**（工具 `scripts/check_model_coverage.py`）。
    def build_lookup(recs):
        by: dict[str, list[float]] = {}
        for r in recs:
            k = f"b{int(r['batch'])}_pf{int(r['prefill_tok'])}_kv{int(r['kv_len'])}"
            by.setdefault(k, []).append(float(r["actual_ms"]))
        import statistics as _st
        return {k: {"median": round(_st.median(v), 4), "max": round(max(v), 4), "n": len(v)}
                for k, v in sorted(by.items())}

    fp = (arts[0].get("fingerprint") or {}).get("expected") or {}
    model = {
        "model_version": MODEL_VERSION,
        "protocol_version": protocols.pop(),
        "grid_sha256": grid_hashes.pop(),
        "fit_on_sessions": sorted({r["session"] for r in records}),
        "n_records": len(records),
        "coefficients": fit_per_batch(records),      # batch -> [c0, c1, c2]
        "basis": ["1", "prefill_tok", "kv_len"],
        #: 两种**可选**的预测方式（第 14 步预注册分支）：
        #: * ``segmented``（默认）：分段线性插值 —— 覆盖率高，但插值误差**不可忽略**
        #:   （秩保证划分下实测低估 p95 = 6.12%）；
        #: * ``lookup``：**只用实测点**，未命中即回退 —— 无插值误差，代价是**覆盖率**。
        "modes": ["segmented", "lookup"],
        "default_mode": args.default_mode,
        "lookup_table": build_lookup(records),
        "lookup_value": "max",       # 保守：取该键实测最大值（永不低估）
        "lookup_note": ("键是**精确形状**（batch/prefill_tok/kv_len 三元）；不插值、不外推，"
                        "未命中返回 None ⇒ 由调用方回退。覆盖率必须与误差并排报。"),
        "calibrated_range": calibrated_range(records),
        "safety_margin": under_p95,
        "safety_margin_basis": "留出形状上的低估误差 p95（预测值<实际值的那些样本的 p95 缺口）",
        "n_forwards_is_multiplier": True,
        "n_forwards_basis": ("实测 n_forwards 1→4 的单前向成本逐位相同（262.7 ms）"
                            "⇒ 加和性成立；故它作乘子，不参与范围检查"),
        "fingerprint": fp,
        # ⚠️ **必须如实记录拟合是否通过了预注册判据** —— 否则下游会把
        # "文件存在"当成"模型可用"。第 14 步实测：v4 在 1-in-3 下 5.04%、
        # 在秩保证划分下 6.12%，**两次都没过** ⇒ 查表是主路径（它不依赖拟合）。
        "validation": {
            "model_holds": bool((ev.get("verdict") or {}).get("model_holds")),
            "verdict_reasons": (ev.get("verdict") or {}).get("reasons") or [],
            "mode_intended_for_use": "lookup",
            "why": ("分段拟合在**秩保证划分**下低估 p95 = 6.12% > 5% ⇒ 触发预注册止损点："
                    "改走'查表 + 保守回退'，把覆盖率作头部指标。"
                    "查表值是**直接实测**、不依赖拟合；其代价是覆盖率（未命中即回退）。"),
        },
        "notes": ("由 scripts/export_cost_model.py 从 cost-calib-v1 的标定产物导出。"
                  "越界必须回退、不得外推；指纹不符不得静默使用。"),
    }
    # ---- 可选键：**不传就不写** ⇒ 默认路径与旧产物逐字节相同（可复核）----------
    if args.lookup_bucket_pf_step or args.lookup_bucket_kv_step:
        model["lookup_bucket"] = {"pf_step": int(args.lookup_bucket_pf_step),
                                  "kv_step": int(args.lookup_bucket_kv_step)}
    if args.mixed_shape_mode:
        model["mixed_shape_mode"] = args.mixed_shape_mode

    # ★ **语义自证守卫**：声称 mixed 点是 `real_mixed`，就必须能在**产物里**找到这个标记。
    # 为什么非要这样：键 `_kb()` **不含** `n_new_mode`（这样 v5/v6 同形状才同键、才可比），
    # ⇒ 单看键**分不出**"每条序列各做 prefill_tok 个"与"1 条做、其余各 1 个"。
    # 若不查产物，模型文件里那一个字符串就成了语义的**唯一**证据 ——
    # 而 v5 正是这么错的（mixed 点实际是 uniform），且**没有任何东西能发现**。
    # ⚠️ v4 及更早的产物根本没有这个字段（读出 `None`）⇒ 用它们声称 real_mixed 会被挡下，
    #    这正是期望行为（那些点确实是旧口径）。
    if args.mixed_shape_mode:
        modes: set[str | None] = set()
        for a in arts:
            for p in (a.get("points") or []):
                modes.add((p.get("shape") or {}).get("n_new_mode"))
        if args.mixed_shape_mode not in modes:
            shown = sorted("（无此字段=旧口径）" if m is None else m for m in modes)
            print(f"✗ --mixed-shape-mode {args.mixed_shape_mode!r} 在产物里找不到："
                  f"产物实际记录的 n_new_mode = {shown} ⇒ 该语义声明**无据**（拒绝导出）",
                  file=sys.stderr)
            return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False, indent=1),
                   encoding="utf-8", newline="\n")
    print(f"已写出 {out}")
    print(f"  协议 {model['protocol_version']} | 网格 {model['grid_sha256']} | "
          f"{model['n_records']} 条 / {len(model['fit_on_sessions'])} 会话")
    print(f"  分段 batch: {sorted(model['coefficients'])}")
    print(f"  标定范围: {json.dumps(model['calibrated_range'], ensure_ascii=False)}")
    print(f"  安全余量 {model['safety_margin']:.4%}（= 留出形状低估 p95）")
    print(f"  ★ 生效的预测模式 default_mode={model['default_mode']}"
          f" | 查表 {len(model['lookup_table'])} 键"
          f" | 分桶 {model.get('lookup_bucket') or '精确键'}"
          f" | mixed 口径 {model.get('mixed_shape_mode') or '未记录'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
