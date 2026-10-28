#!/usr/bin/env python3
"""**把"逐步缺失开销"分解开**：用遥测 `note` 里的 `fwd_ms / pack_ms / readback_ms`。

为什么要有它
------------
实测发现：模型对**纯 decode 步**预测 100.2 ms 而实际 130.4 ms（中位 **−23%**），
且绝对差随 batch 增长（b2 −10.5 / b4 +15.4 / **b8 +30.4 ms**）⇒ 指向"**每序列**的一笔开销"。
但"拟合一个 `a + b×batch`"是**盲猜形状**；而遥测的 `note` 已经把一步拆开了：

    prefill_tok=512 decode=7 forwards=2 fwd_ms=322.17 pack_ms=4.30 readback_ms=0.63

⇒ `fwd_ms` 是**前向**（模型要预测的就是它），`pack_ms`/`readback_ms` 是**host 侧**。
本脚本把"实际步时长"分解成这几块，从而**判定**缺失的那笔到底是哪一块、
以及它是否随 `batch` 线性 —— 而不是先假定它长什么样。

⚠️ 归因范围：这是 **S4000 上当前软件栈与执行路径**的开销，不能据此推断芯片架构。

用法::

    python scripts/decompose_step_overhead.py --tag S14B1 --json results/processed/step_overhead.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

#: `note` 里的键值对（逐字按真机产物写的模式；**不要**用一个交替正则去撞运气 ——
#: 多标签要按名取，取错会静默给出另一个组件的值）。
_NOTE_RE = re.compile(r"(\w+)=([-\d.]+)")


def parse_note(note: str | None) -> dict[str, float]:
    """把 `note` 解析成字典。取不到的键**不出现**（而不是填 0）。"""
    if not note:
        return {}
    return {k: float(v) for k, v in _NOTE_RE.findall(note)}


def load(tag: str) -> list[dict[str, Any]]:
    out = []
    for d in sorted(glob.glob(str(_ROOT / "results" / "raw" / f"{tag}-*cost*"))):
        mp = Path(d) / "meta.json"
        tp = Path(d) / "telemetry.jsonl"
        if not (mp.exists() and tp.exists()):
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        tel = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()]
        out.append({"dir": Path(d).name,
                    "trace": meta.get("controller_trace") or [], "tel": tel})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="分解逐步开销（只读产物）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    runs = load(args.tag)
    if not runs:
        print(f"✗ 没找到 tag={args.tag} 的 cost 臂产物", file=sys.stderr)
        return 2

    # 先看 note 覆盖情况：没有 note 就没法分解，必须**明说**而不是给个 0
    n_note = n_rows = 0
    missing = 0
    rows: list[dict[str, float]] = []
    for r in runs:
        pts = sorted((int(t["step"]), float(t["t_ms"]))
                     for t in r["tel"] if t.get("t_ms") is not None)
        dur = {pts[i][0]: pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)}
        note_by_step = {int(t["step"]): parse_note(t.get("note")) for t in r["tel"]}
        for e in r["trace"]:
            dd = e.get("cost_policy")
            if not dd or dd.get("fallback_reason"):
                continue
            cs = [c for c in (dd.get("candidates") or []) if c.get("feasible")]
            if not cs:
                continue
            pm = cs[0].get("predicted_ms")
            stp = int(e["step"])
            if pm is None or (stp + 1) not in dur:
                continue
            n_rows += 1
            n = note_by_step.get(stp + 1) or {}
            if not n:
                missing += 1
                continue
            n_note += 1
            rows.append({
                "batch": float(cs[0].get("batch") or 0),
                "predicted_ms": float(pm),
                "actual_step_ms": dur[stp + 1],
                "fwd_ms": n.get("fwd_ms", float("nan")),
                "pack_ms": n.get("pack_ms", float("nan")),
                "readback_ms": n.get("readback_ms", float("nan")),
                "forwards": n.get("forwards", float("nan")),
                "decode": n.get("decode", float("nan")),
                "prefill_tok": n.get("prefill_tok", float("nan")),
                "scheduled": n.get("scheduled_tokens", float("nan")),
            })

    print(f"tag={args.tag}｜run={len(runs)}｜可用配对 {n_rows}｜**note 可解析 {n_note}**"
          f"｜note 缺失 {missing}")
    if not rows:
        print("✗ 没有可分解的配对（note 缺失或没有未回退的决策）"
              " ⇒ 这条分析**做不了**，不假装有结论", file=sys.stderr)
        return 2

    def med(key: str) -> float | None:
        v = [r[key] for r in rows if r[key] == r[key]]      # NaN 过滤
        return st.median(v) if v else None

    def fmt(v: float | None, p="{:.2f}") -> str:
        return "—" if v is None else p.format(v)

    print()
    print("=== 每步各分块的中位（ms）===")
    print(f"  实际步时长          {fmt(med('actual_step_ms'))}")
    print(f"  其中 fwd_ms（前向）  {fmt(med('fwd_ms'))}   ← 模型要预测的就是它")
    print(f"    pack_ms           {fmt(med('pack_ms'))}")
    print(f"    readback_ms       {fmt(med('readback_ms'))}")
    host = [r["pack_ms"] + r["readback_ms"] for r in rows]
    host_med = st.median(host)
    print(f"  host 部分合计        {fmt(host_med)}")
    print(f"  其余（未被 note 覆盖）{fmt((med('actual_step_ms') or 0) - (med('fwd_ms') or 0) - host_med)}")
    print()
    print("=== 关键对比：模型预测 vs **只比前向** vs 整步 ===")
    print(f"  预测中位             {fmt(med('predicted_ms'))}")
    rel_fwd = [(r["predicted_ms"] - r["fwd_ms"]) / r["fwd_ms"] for r in rows if r["fwd_ms"] == r["fwd_ms"]]
    rel_step = [(r["predicted_ms"] - r["actual_step_ms"]) / r["actual_step_ms"] for r in rows]
    print(f"  预测 vs fwd_ms   中位 {fmt(st.median(rel_fwd), '{:+.1%}')}"
          f"   ← **这才是模型该被评的**（同口径）")
    print(f"  预测 vs 整步      中位 {fmt(st.median(rel_step), '{:+.1%}')}"
          f"   ← 之前一直在比这个（口径不同）")
    print()
    print("=== host 部分是否随 batch 线性 ===")
    for b in sorted({int(r['batch']) for r in rows}):
        v = [r["pack_ms"] + r["readback_ms"] for r in rows if int(r["batch"]) == b]
        f = [r["fwd_ms"] for r in rows if int(r["batch"]) == b]
        if v:
            print(f"  batch={b:<2} n={len(v):<3} host 中位 {fmt(st.median(v)):>7} ms"
                  f"｜fwd 中位 {fmt(st.median(f)):>8} ms"
                  f"｜host/序列 {fmt(st.median(v) / max(1, b)):>6} ms")
    print()
    print("读法：若 **预测 vs fwd_ms 已经很小**，说明模型对'前向'是准的，"
          "缺的是 **host 部分**（它不在模型的特征里）⇒ 该加的是 host 项，"
          "而不是去改前向的拟合。")
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "tag": args.tag, "n_pairs": n_rows, "n_note_parsed": n_note, "n_note_missing": missing,
            "median": {k: med(k) for k in
                       ("actual_step_ms", "fwd_ms", "pack_ms", "readback_ms", "predicted_ms")},
            "median_rel_pred_vs_fwd": (st.median(rel_fwd) if rel_fwd else None),
            "median_rel_pred_vs_step": st.median(rel_step),
            "boundary": "归因限定 S4000 上当前软件栈与执行路径，不推断芯片架构",
        }, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"\n已写出：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
