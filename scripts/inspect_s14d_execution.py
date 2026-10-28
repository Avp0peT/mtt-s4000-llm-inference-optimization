#!/usr/bin/env python3
"""**D0 判决层 vs 实际执行** 并排：候选被拒 ≠ 不发 prefill。

为什么要有它
------------
`diagnose_scheduling.py` 报的是 **D0 可行率**（"控制器接纳了多少候选"）。
但 `cost_policy.py` 有**两条降级路径**，两条都**仍然在发 prefill**：

* `predicted` 为空（候选全在模型范围外）⇒ `_fallback()` **回退到阈值控制器**；
* 有预测但全超每步预算 ⇒ **取最省的那个**（`n_degraded`）。

⇒ 从低 D0 推"请求不推进 / 不发 prefill"是**错的**。要知道 prefill 究竟发没发，
只能从遥测 `note` 里的 `prefill_tok=…` **实测**。

⚠️ 遥测里**没有** `tokens_prefilled` 这一列；`note` 取不到 `prefill_tok` 的行
   不计入"没发"，而是由 `count_unparsed` 单独显形（缺字段 ≠ 值为 0）。

用法::

    python scripts/inspect_s14d_execution.py --tag S14D --json results/processed/exec_S14D.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from ascend_opt.metrics import percentile  # noqa: E402
from inspect_step3_runs import (  # noqa: E402
    count_unparsed,
    expected_prefill_tokens,
    load_telemetry,
    parse_note,
    prefill_events,
)


def _arm_of(name: str) -> str:
    """``S14D-short_chat-fixed_cost-r2`` → ``cost``。"""
    parts = name.split("-")
    return parts[2] if len(parts) > 2 else name


def _stat(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def scan(tag: str, raw_root: Path) -> list[dict]:
    out: list[dict] = []
    for d in sorted(p for p in raw_root.glob(f"{tag}-*") if p.is_dir()):
        rows, bad_lines = load_telemetry(d)
        if not rows:
            continue
        ev = prefill_events(rows)
        n_unparsed = count_unparsed(rows)
        fwd: list[float] = []
        host: list[float] = []
        for r in rows:
            kv = parse_note(r.get("note"))
            if "fwd_ms" not in kv:
                continue
            fwd.append(kv["fwd_ms"])
            if "pack_ms" in kv and "readback_ms" in kv:
                host.append(kv["pack_ms"] + kv["readback_ms"])
        tot = sum(e["prefill_tok"] for e in ev)
        exp = expected_prefill_tokens(d)
        out.append({
            "run": d.name,
            "arm": _arm_of(d.name),
            "_dir": d,          # 仅供本脚本内部使用（落 JSON 时剔除）
            "_tel": rows,       # 同上
            "steps": len(rows),
            "bad_lines": bad_lines,
            "unparsed_prefill": n_unparsed,
            "prefill_steps": len(ev),
            "prefill_frac": (len(ev) / len(rows)) if rows else None,
            "prefill_max_step": max((e["prefill_tok"] for e in ev), default=None),
            "prefill_total": tot,
            "prefill_expected": exp,
            "prefill_ratio": (tot / exp) if (exp not in (None, 0)) else None,
            "fwd_ms": _stat(fwd),
            "host_ms": _stat(host),
        })
    return out


def by_arm(runs: list[dict]) -> dict[str, dict]:
    arms: dict[str, dict] = {}
    for a in sorted({r["arm"] for r in runs}):
        rs = [r for r in runs if r["arm"] == a]
        steps = sum(r["steps"] for r in rs)
        psteps = sum(r["prefill_steps"] for r in rs)
        tot = sum(r["prefill_total"] for r in rs)
        arms[a] = {
            "n_runs": len(rs),
            "steps": steps,
            "prefill_steps": psteps,
            "prefill_frac": (psteps / steps) if steps else None,
            "prefill_max_step": max(
                (r["prefill_max_step"] for r in rs if r["prefill_max_step"] is not None),
                default=None,
            ),
            "prefill_total": tot,
            "unparsed_prefill": sum(r["unparsed_prefill"] for r in rs),
            "bad_lines": sum(r["bad_lines"] for r in rs),
        }
    return arms


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D0 判决层 vs 实际执行 prefill（只读产物）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--raw-root", default=str(_ROOT / "results" / "raw"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    runs = scan(args.tag, Path(args.raw_root))
    if not runs:
        print(f"✗ 没有读到任何 {args.tag}-* run —— 本项**未检验**（不要当成 0）")
        return 1

    arms = by_arm(runs)

    print(f"tag={args.tag}｜run={len(runs)}｜臂={sorted(arms)}")
    print()
    print("=== 实际执行：每个 run 的 prefill ===")
    print(f"{'run':<44} {'步数':>5} {'发prefill步':>10} {'占比':>7} "
          f"{'最大单步':>8} {'Σprefill':>9} {'待发总量':>9} {'缺字段':>6}")
    for r in runs:
        mx = r["prefill_max_step"]
        ratio = r["prefill_ratio"]
        print(f"{r['run']:<44} {r['steps']:>5} {r['prefill_steps']:>10} "
              f"{(r['prefill_frac'] * 100 if r['prefill_frac'] is not None else float('nan')):>6.1f}% "
              f"{(f'{mx:.0f}' if mx is not None else '—'):>8} {r['prefill_total']:>9.0f} "
              f"{(f'{ratio * 100:.1f}%' if ratio is not None else '—'):>9} "
              f"{r['unparsed_prefill']:>6}")

    print()
    print("=== 按臂汇总 ===")
    print(f"{'臂':<12} {'run':>4} {'总步数':>7} {'发prefill步':>10} {'占比':>7} "
          f"{'最大单步':>8} {'Σprefill':>9}")
    for a, s in arms.items():
        mx = s["prefill_max_step"]
        print(f"{a:<12} {s['n_runs']:>4} {s['steps']:>7} {s['prefill_steps']:>10} "
              f"{(s['prefill_frac'] * 100 if s['prefill_frac'] is not None else float('nan')):>6.1f}% "
              f"{(f'{mx:.0f}' if mx is not None else '—'):>8} {s['prefill_total']:>9.0f}")

    print()
    print("=== fwd_ms（模型该预测的）与 host=pack_ms+readback_ms 分位（按臂）===")
    print("    ⚠️ 分位跨 run 池化（把同一臂各 repeat 的步堆在一起）；"
          "伪重复不影响量级，但不做显著性判断")
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        print(f"  [{a}]")
        for label, key in (("fwd_ms", "fwd_ms"), ("host_ms", "host_ms")):
            vals: list[float] = []
            for r in rs:
                for row in r["_tel"]:
                    kv = parse_note(row.get("note"))
                    if "fwd_ms" not in kv:
                        continue
                    if key == "fwd_ms":
                        vals.append(kv["fwd_ms"])
                    elif "pack_ms" in kv and "readback_ms" in kv:
                        vals.append(kv["pack_ms"] + kv["readback_ms"])
            if vals:
                print(f"    {label:<9} n={len(vals):>5}  p50={percentile(vals, 50):8.2f}  "
                      f"p90={percentile(vals, 90):8.2f}  p99={percentile(vals, 99):8.2f}  "
                      f"max={max(vals):9.2f}")
            else:
                print(f"    {label:<9} n=0（无样本）")

    print()
    print("=== 口径对照：'全体步' vs '已接纳可行候选的步'（decompose_step_overhead 口径）===")
    print("    ⚠️ 两者是**不同子样本**，数值不可混用；差距本身说明"
          "被接纳的步是 prefill 更轻的步")
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        sub = _accepted_subsample(rs)
        if not sub:
            print(f"  [{a}] 已接纳子样本 n=0（该臂无未回退的可行候选）")
            continue
        fv = sorted(s["fwd"] for s in sub)
        hv = sorted(s["host"] for s in sub)
        pv = sorted(s["prefill_tok"] for s in sub)
        print(f"  [{a}] 已接纳子样本 n={len(sub)}  "
              f"fwd 中位={percentile(fv, 50):8.2f}  "
              f"host 中位={percentile(hv, 50):7.2f}  "
              f"prefill_tok 中位={percentile(pv, 50):7.0f} "
              f"max={max(pv):7.0f}")

    print()
    print("=== 决策路径：回退 / 降级 / 正常（从 trace 数，**不是**汇总字段）===")
    print("    ⚠️ `controller_effect` 里**没有** n_degraded / n_prefill_starved"
          " ⇒ 那两个**数不出来（字段未入产物）**；")
    print("       下面是从 `controller_trace[*].cost_policy` 逐条数出来的，口径更细但可核对。")
    for a in arms:
        rs = [r for r in runs if r["arm"] == a]
        p = _decision_paths(rs)
        if not p["with_cost_policy"]:
            print(f"  [{a}] 决策 {p['decisions_total']} 条，**无一条**带 cost_policy"
                  f"（该臂不加载成本模型）")
            continue
        print(f"  [{a}] 决策 {p['decisions_total']} 条，带 cost_policy {p['with_cost_policy']} 条："
              f" 回退={p['n_fallback']}  降级取最省={p['n_degraded']}  正常选中={p['n_normal']}")
        if p["fallback_reasons"]:
            print(f"        回退原因：{p['fallback_reasons']}")
        for ex in p["degraded_examples"]:
            print(f"        降级样本 step={ex['step']} chunk={ex['chunk']} "
                  f"max_tokens={ex['max_tokens']}  {ex['reason']}")

    tot_unparsed = sum(r["unparsed_prefill"] for r in runs)
    if tot_unparsed:
        print()
        print(f"⚠️ 有 {tot_unparsed} 行的 note 里**没有** prefill_tok 字段 ⇒ "
              f"上面的'发 prefill 步数'是**下界**，不是 0 也不是完整值")
    else:
        print()
        print("✓ 所有行的 note 都含 prefill_tok 字段 ⇒ '发 prefill 步数'口径完整")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"tag": args.tag,
                       "runs": [{k: v for k, v in r.items() if not k.startswith("_")}
                                for r in runs],
                       "by_arm": arms},
                      f, ensure_ascii=False, indent=1)
        print(f"\n已写出：{out}")
    return 0


def _decision_paths(rs: list[dict]) -> dict:
    """从 `controller_trace[*].cost_policy` 数**每条决策走了哪条路径**。

    ⚠️ 汇总字段 `controller_effect` 里**没有** `n_degraded` / `n_prefill_starved`
    —— 那两个数**数不出来**（未入产物）。但逐决策的 `degraded` 布尔标记**在 trace 里**，
    所以"降级"这类**可以**数（数与数不出来必须分开写，不得用 0 代替）。

    三条路径（按 `cost_policy.py` 的语义）：
    * `fallback_reason` 非空 ⇒ `_fallback()` 回退到阈值控制器（**仍在发 prefill**）；
    * `degraded == True` ⇒ 有预测但全超预算 ⇒ 取最省者（**仍在发 prefill**）；
    * 其余 ⇒ 正常选中可行候选。
    """
    out = {
        "decisions_total": 0, "with_cost_policy": 0,
        "n_fallback": 0, "n_degraded": 0, "n_normal": 0,
        "fallback_reasons": {}, "degraded_examples": [],
    }
    for r in rs:
        mp = Path(r["_dir"]) / "meta.json"
        if not mp.exists():
            continue
        trace = json.loads(mp.read_text(encoding="utf-8")).get("controller_trace") or []
        out["decisions_total"] += len(trace)
        for e in trace:
            cp = e.get("cost_policy")
            if cp is None:
                continue
            out["with_cost_policy"] += 1
            fr = cp.get("fallback_reason")
            if fr:
                out["n_fallback"] += 1
                out["fallback_reasons"][fr] = out["fallback_reasons"].get(fr, 0) + 1
            elif cp.get("degraded"):
                out["n_degraded"] += 1
                if len(out["degraded_examples"]) < 3:
                    out["degraded_examples"].append(
                        {"step": e.get("step"), "chunk": cp.get("chunk"),
                         "max_tokens": cp.get("max_tokens"), "reason": cp.get("reason")})
            else:
                out["n_normal"] += 1
    return out


def _accepted_subsample(rs: list[dict]) -> list[dict]:
    """`decompose_step_overhead.py` 的口径：只取**未回退且被接纳**的决策所对的 step。

    ⚠️ 这是**有偏子样本**（控制器只在它认为可行时才记 `predicted_ms`），
    与"全体步"的分位差距很大 —— 两个数**不要混用**，各自标明口径。
    """
    out: list[dict] = []
    for r in rs:
        d = Path(r["_dir"])
        mp, tp = d / "meta.json", d / "telemetry.jsonl"
        if not (mp.exists() and tp.exists()):
            continue
        trace = (json.loads(mp.read_text(encoding="utf-8")).get("controller_trace") or [])
        note_by_step = {int(t["step"]): parse_note(t.get("note"))
                        for t in r["_tel"] if t.get("step") is not None}
        for e in trace:
            dd = e.get("cost_policy")
            if not dd or dd.get("fallback_reason"):
                continue
            cs = [c for c in (dd.get("candidates") or []) if c.get("feasible")]
            if not cs:
                continue
            stp = int(e["step"])
            # 与 decompose_step_overhead.py 保持**逐字一致**的配对：决策在 step stp，
            # 其产生的遥测在 step stp+1（改配对数会造出与既有脚本不可比的数字）。
            n = note_by_step.get(stp + 1)
            if n and "fwd_ms" in n:
                out.append({"fwd": n["fwd_ms"],
                            "host": n.get("pack_ms", 0.0) + n.get("readback_ms", 0.0),
                            "prefill_tok": n.get("prefill_tok", 0.0)})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
