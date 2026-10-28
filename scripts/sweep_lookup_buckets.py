#!/usr/bin/env python3
"""**分桶查表的取舍曲线**：桶宽 ↔ 覆盖率 ↔ 保守代价（**纯本地，用已有数据**）。

为什么要有它
------------
第 14 步按预注册止损点转"查表 + 保守回退"后，实测 **精确查表在真实负载上的覆盖率 = 0.00%**
（280 个有效工作候选一个都没命中）—— 因为真实 `kv`/`prefill` 是**连续取值**，
不匹配任何标定点。⇒ 纯精确表在这个负载上是**死路**。

⇒ 于是分支的正确形态是**分桶查表**：把 `(prefill_tok, kv_len)` 量化到桶，
桶值取**桶内实测最大值**（保守：绝不低估），未命中才回退。
本脚本把这条**取舍**量出来：

* **覆盖率**：在**真实负载记录的形状**上，有多大比例的候选能命中某个桶；
* **保守代价**：用**逐点留一**（leave-one-out，桶内其余点的最大值）去预测该点，
  看会**高估多少** —— 高估是"保守"的代价（控制器更犹豫、更可能不推进 prefill）。
  ⚠️ 按构造**低估恒为 0**（因为取的是桶内最大值），所以这里只报高估侧。

口径写死在这里，**不许事后换**。用法::

    python scripts/sweep_lookup_buckets.py \
        --calib "results/raw/_costcalib/costcalib_v4_s*.json" \
        --shapes-tag S14B1 --json results/processed/lookup_bucket_sweep.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

#: 桶宽的候选值（`prefill_tok` 与 `kv_len` 各自）
PF_STEPS = (0, 128, 256, 512)      # 0 = 不量化（精确）
KV_STEPS = (0, 128, 256, 512, 1024)


def load_records(pattern: str) -> list[dict[str, Any]]:
    """读**评估 JSON**（`costcalib_fit.py --json` 的产物）里的 `records`。

    ⚠️ **原始探针产物里没有 `records`**（那里是逐点的 `points`）—— 这个坑今天踩过两次。
    所以这里只认 `records`，并在取不到时**明确报错**（而不是静默给 0 条）。
    """
    files = sorted(glob.glob(str(_ROOT / pattern)))
    if not files:
        raise SystemExit(f"✗ 没匹配到文件：{pattern}")
    recs: list[dict[str, Any]] = []
    for f in files:
        payload = json.loads(Path(f).read_text(encoding="utf-8"))
        if "records" not in payload:
            raise SystemExit(
                f"✗ {Path(f).name} 里没有 `records`（只有 {sorted(payload)[:6]}…）⇒ "
                "请指向评估 JSON（`scripts/costcalib_fit.py --json` 的产物），"
                "原始探针产物里是逐点的 `points`。")
        recs.extend(payload["records"])
    return recs


def load_shapes(tag: str, assume_kv: int | None = None) -> list[tuple[int, int, int]]:
    """真实负载记录到的**有效工作**候选形状（batch>0）。

    ⚠️ `kv_len` 是**后加的**字段：旧产物里没有（当时没记 ⇒ 无法重放，这本身是缺陷，
    已修）。对旧产物只能按 `assume_kv` 给一个**近似值**，并在输出里标明。
    """
    out: list[tuple[int, int, int]] = []
    for f in sorted(glob.glob(str(_ROOT / "results" / "raw" / f"{tag}-*cost*" / "meta.json"))):
        meta = json.loads(Path(f).read_text(encoding="utf-8"))
        for e in meta.get("controller_trace") or []:
            d = e.get("cost_policy") or {}
            for c in (d.get("candidates") or []):
                b = int(c.get("batch") or 0)
                if b > 0:
                    kv = int(c.get("kv_len") or 0)
                    out.append((b, int(c.get("tokens_prefilled") or 0),
                                kv if kv > 0 else int(assume_kv or 0)))
    return out


def bucket_of(rec: dict[str, Any], pf_step: int, kv_step: int) -> tuple[int, int, int]:
    pf, kv = int(rec["prefill_tok"]), int(rec["kv_len"])
    if pf_step:
        pf = (pf // pf_step) * pf_step
    if kv_step:
        kv = (kv // kv_step) * kv_step
    return (int(rec["batch"] or 1), pf, kv)


def build_buckets(recs: list[dict], pf_step: int, kv_step: int) -> dict[tuple, list[float]]:
    by: dict[tuple, list[float]] = {}
    for r in recs:
        by.setdefault(bucket_of(r, pf_step, kv_step), []).append(float(r["actual_ms"]))
    return by


def loo_conservative_error(recs: list[dict], pf_step: int, kv_step: int) -> dict[str, Any]:
    """**逐点留一**：用"桶内其余点的最大值"预测该点 ⇒ 报**高估**的分布。"""
    buckets = build_buckets(recs, pf_step, kv_step)
    over: list[float] = []
    n_used = 0
    for r in recs:
        b = bucket_of(r, pf_step, kv_step)
        others = [v for v in buckets[b] if v != float(r["actual_ms"])] or (
            [] if len(buckets[b]) < 2 else buckets[b][:-1])
        if not others:
            continue                       # 落单的桶无法留一 ⇒ 不进分母（并如实报个数）
        n_used += 1
        pred = max(others)
        act = float(r["actual_ms"])
        over.append((pred - act) / act)    # 按构造 ≥ 0
    q = lambda v, f: (sorted(v)[min(len(v) - 1, int(f * len(v)))] if v else None)  # noqa: E731
    return {"n_used": n_used, "n_skipped_singleton": len(recs) - n_used,
            "n_buckets": len(buckets),
            "over_p50": q(over, 0.50), "over_p95": q(over, 0.95), "over_max": (max(over) if over else None)}


def coverage(buckets: dict[tuple, list[float]],
             shapes: list[tuple[int, int, int]], pf_step: int, kv_step: int) -> float | None:
    if not shapes:
        return None
    hit = 0
    for b, pf, kv in shapes:
        if pf_step:
            pf = (pf // pf_step) * pf_step
        if kv_step:
            kv = (kv // kv_step) * kv_step
        hit += int((b, pf, kv) in buckets)
    return hit / len(shapes)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="分桶查表的取舍曲线（只读）")
    ap.add_argument("--calib", required=True, help="标定产物 glob（相对仓库根）")
    ap.add_argument("--shapes-tag", required=True, help="真实负载 tag（从它的 cost 臂轨迹取形状）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--assume-kv", type=int, default=None,
                    help="旧产物没记 kv_len ⇒ 用一个近似值（结果必须标明是近似的）")
    args = ap.parse_args(argv)

    recs = load_records(args.calib)
    shapes = load_shapes(args.shapes_tag, args.assume_kv)
    if not recs or not shapes:
        print(f"✗ 数据不足：标定 {len(recs)} 条 / 形状 {len(shapes)} 个", file=sys.stderr)
        return 2
    print(f"标定 {len(recs)} 条｜真实负载形状 {len(shapes)} 个（tag={args.shapes_tag}）")
    if args.assume_kv:
        print(f"  ⚠️ 旧产物没记 `kv_len` ⇒ 形状里的 kv 用**近似值 {args.assume_kv}**"
              "（结论要标明是近似的；新产物已修）")
    print()
    header = (f"{'pf 桶宽':>8}{'kv 桶宽':>9}{'桶数':>7}{'覆盖率':>9}"
              f"{'保守高估 p50':>13}{'p95':>9}{'最大':>9}{'留一可用点':>11}{'落单桶跳过':>11}")
    print(header)
    print('-' * len(header))
    out: dict[str, Any] = {"calib": args.calib, "shapes_tag": args.shapes_tag,
                           "n_records": len(recs), "n_shapes": len(shapes), "cells": []}
    for pf_step in PF_STEPS:
        for kv_step in KV_STEPS:
            buckets = build_buckets(recs, pf_step, kv_step)
            e = loo_conservative_error(recs, pf_step, kv_step)
            cov = coverage(buckets, shapes, pf_step, kv_step)
            cell = {"pf_step": pf_step, "kv_step": kv_step, "n_buckets": e["n_buckets"],
                    "coverage": cov, "over_p50": e["over_p50"], "over_p95": e["over_p95"],
                    "over_max": e["over_max"], "loo_n": e["n_used"],
                    "loo_skipped": e["n_skipped_singleton"]}
            out["cells"].append(cell)
            f = lambda x, p='{:.2%}': ('—' if x is None else p.format(x))  # noqa: E731
            print(f"{pf_step:>8}{kv_step:>9}{e['n_buckets']:>7}{f(cov):>9}"
                  f"{f(e['over_p50']):>13}{f(e['over_p95']):>9}{f(e['over_max']):>9}"
                  f"{e['n_used']:>11}{e['n_skipped_singleton']:>11}")
    print()
    print("读法：**桶越宽 ⇒ 覆盖率越高、保守代价越大**（高估更多 ⇒ 控制器更犹豫）。")
    print("      ⚠️ 低估**恒为 0**（取桶内最大值），所以只报高估侧 —— 这是这条分支的设计取舍。")
    print("      ⚠️ '落单桶跳过' 多的格说明桶太细：那些点**无法被留一验证**，其数字参考性弱。")
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"\n已写出：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
