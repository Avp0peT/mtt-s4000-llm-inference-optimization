#!/usr/bin/env python3
"""第⑨步干预实验（判据口径 A/B）的**结果对照**。

把"跑完回来对照那 4 条预测"变成一条命令，而不是靠肉眼翻产物。

设计要点
--------
* **预测先登记、脚本后写**：4 条预测写在 `NEXT_SESSION_TASKS.md` 第 09 步 §二/§五，
  本脚本把它们**程序化**成 PASS/FAIL 判定——否则"事后挑一个能成立的读法"太容易。
* **三态判定**：每条预测是 `pass` / `fail` / **`unknown`（数据不可判）**。
  "测不出"不能算通过（`smoke-harness-trustworthy` 的同一条纪律）。
* **可比性前置检查**：两臂必须同标签、同溯源戳、同请求数、`dropped=0`；
  不通过就**不报差值**（先做检查再报数，不把不可比的数字放进同一张表）。

用法::

    python scripts/analyze_s9_ab.py --out-root results/raw --tag S9AB
    python scripts/analyze_s9_ab.py --run-dir results/raw --a <dir> --b <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: 期望值来自 pilot 实测（`EXPERIMENTS.md` §3.4 E10）与 static 参考。
PILOT_FIRST_ACTION_T_S = 123.0      # A 臂首次动作时刻
PILOT_PURE_DECODE_MS = 61.4         # A 臂首动作窗内纯 decode p50
PILOT_POST_SHRINK_MS = 118.0        # 收缩后纯 decode p50
STATIC_TPOT_P50_MS = 175.9          # 上一批 static 的 tpot p50
PILOT_ADAPTIVE_TPOT_P50_MS = 182.4  # 上一批 adaptive（干预前）的 tpot p50 —— ③ 的第二参照
STATIC_STALL_P99_MS = 412.4         # 上一批 static 的 stall p99
ADAPTIVE_STALL_P99_MS = 330.8       # 上一批 adaptive 的 stall p99

_NOTE = re.compile(r"prefill_tok=(\d+) decode=(\d+).*fwd_ms=([\d.]+)")


def _load(run_dir: Path) -> dict[str, Any]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    tel = [json.loads(l) for l in (run_dir / "telemetry.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    steps: dict[int, dict[str, Any]] = {}
    for t in tel:
        m = _NOTE.search(t.get("note") or "")
        if m:
            steps[t["step"]] = dict(
                t_ms=t["t_ms"], pf=int(m.group(1)), fwd=float(m.group(3)),
                qd=t.get("queue_depth"), budget=t.get("max_batched_tokens"))
    rep = meta.get("reproducibility") or {}
    s = meta.get("summary") or {}
    trace = meta.get("controller_trace") or []
    return dict(
        path=run_dir, meta=meta, tel=tel, steps=steps,
        label=meta.get("data_label"), commit=(rep.get("git_commit") or "")[:12],
        dirty=rep.get("git_dirty"),
        n=s.get("n_requests"), ok=s.get("n_success"),
        dropped=(s.get("extra") or {}).get("dropped_requests"),
        tpot_p50=(s.get("tpot") or {}).get("p50"),
        stall_p99=(s.get("stall") or {}).get("p99"),
        slo_rate=s.get("slo_satisfaction_rate"),
        actions=[a for a in trace if a.get("changed")],
        criterion=((meta.get("config") or {}).get("controller") or {}).get("signal_criterion"),
    )


def _check_first_action(firsts_t: list[float], firsts_qd: list[Any],
                        b_arms: list[dict[str, Any]]) -> bool | None:
    """预测①：B 臂要么**不动**，要么动了但**不在 pilot 那个时刻/那批队深为 0 的位置**。

    两个分支都算 pass，因为这条预测要排除的是"**在没人排队时、由污染读数触发的收缩**"，
    而不是"必须一次都不动"。
    """
    if not firsts_t:
        # 三个 run 都没有动作 ⇒ 满足"干脆不动"这一支
        return None if any(r["actions"] for r in b_arms) else True
    if all(abs(t - PILOT_FIRST_ACTION_T_S) < 1.0 for t in firsts_t):
        return False                      # 与 pilot 同一时刻触发 ⇒ 污染还在
    if all(q is not None and q > 0 for q in firsts_qd):
        return True                       # 动过，但不是在"队深 0"时
    # 动过、且在队深 0 时 ⇒ 按**登记原文**（"且触发时队列不空"）判 fail。
    # ⚠️ 这条登记写得不好：队深 0 不等于"没有真实压力"（8 个槽全满、批内 8 条序列时，
    # 纯 decode 本来就会变慢）。但**不事后改判据**——改判据另列为 post-hoc 观测，
    # 见本次分析记录里"触发值的性质"那张表。
    return False


def _action_shape(r: dict[str, Any]) -> dict[str, Any]:
    acts = r["actions"]
    steps = r["steps"]
    out: dict[str, Any] = {"n_actions": len(acts), "steps": [a["step"] for a in acts]}
    if not acts:
        out.update(first_t=None, first_qd=None, pure_p50_at_first=None,
                   pure_p50_after=None, reported=None, budget_range=None)
        return out
    first = acts[0]["step"]
    S = sorted(steps)
    win = [steps[s] for s in S if s <= first][-32:]
    pd = [x["fwd"] for x in win if x["pf"] == 0]
    after = [x["fwd"] for s, x in steps.items() if first < s <= first + 160 and x["pf"] == 0]
    budgets = [x["budget"] for x in steps.values() if x.get("budget") is not None]
    out.update(
        first_step=first,
        first_t=steps[first]["t_ms"] / 1000.0,
        first_qd=steps[first]["qd"],
        pure_p50_at_first=(statistics.median(pd) if pd else None),
        pure_p50_after=(statistics.median(after) if after else None),
        reported=(acts[0].get("signals") or {}).get("criterion_ms")
        or (acts[0].get("signals") or {}).get("tpot_p90_ms"),
        budget_range=(min(budgets), max(budgets)) if budgets else None,
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="第⑨步 A/B 干预结果对照")
    ap.add_argument("--out-root", default="results/raw", help="产物根目录")
    ap.add_argument("--tag", default="S9AB", help="run_id 前缀")
    ap.add_argument("--a", default="ramp_adaptive_c8", help="A 臂配置名（判据 tpot_p90）")
    ap.add_argument("--b", default="ramp_adaptive_decode", help="B 臂配置名（判据 decode_p90）")
    args = ap.parse_args(argv)

    root = Path(args.out_root)
    arms: dict[str, list[dict[str, Any]]] = {}
    for tag, cell in (("A", args.a), ("B", args.b)):
        dirs = sorted(d for d in root.glob(f"{args.tag}-*{cell}*") if d.is_dir())
        if not dirs:
            print(f"✗ 找不到 {tag} 臂的产物（{root}/{args.tag}-*{cell}*）", file=sys.stderr)
            return 2
        arms[tag] = [_load(d) for d in dirs]
        print(f"{tag} 臂（{cell}）：{len(dirs)} 个 run")
        for r in arms[tag]:
            print(f"   {r['path'].name}  label={r['label']} commit={r['commit']} "
                  f"dirty={r['dirty']} n={r['n']} dropped={r['dropped']} "
                  f"criterion={r['criterion']}")

    # ---- 可比性前置检查（不通过就不报差值）--------------------------------
    allr = arms["A"] + arms["B"]
    problems: list[str] = []
    if {r["label"] for r in allr} != {"MEASURED"}:
        problems.append(f"标签不全是 MEASURED：{sorted({r['label'] for r in allr})}")
    if len({(r["commit"], r["dirty"]) for r in allr}) != 1:
        problems.append("两臂**溯源戳不同** ⇒ 不是同版本对照")
    if {r["n"] for r in allr} != {92}:
        problems.append(f"请求数不一致：{sorted({r['n'] for r in allr})}")
    if any(r["ok"] != r["n"] for r in allr):
        problems.append("有 run 未全部成功")
    if {r["dropped"] for r in allr} != {0}:
        problems.append(f"有 dropped：{sorted({r['dropped'] for r in allr})}")
    print("\n=== 可比性前置检查 ===")
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
        print("  ⇒ **不报差值**（先把可比性修好）")
        return 1
    print("  ✓ 通过（同标签 / 同戳 / 同请求数 / 全成功 / dropped=0）⇒ 可以报差值")

    shapes = {t: [_action_shape(r) for r in arms[t]] for t in ("A", "B")}
    print("\n=== 逐 run：动作形态 ===")
    hdr = (f"{'臂':<3}{'run':<34}{'动作数':>6}{'首动作step':>10}{'首动作t(s)':>11}"
           f"{'队深':>6}{'报告值':>9}{'窗内纯decode':>13}{'之后纯decode':>13}{'budget 范围':>16}")
    print(hdr)
    def _f(v, nd=1):
        """None ⇒ 'n/a'（**不填 0**）。"""
        return "n/a" if v is None else f"{v:.{nd}f}"

    for t in ("A", "B"):
        for r, sh in zip(arms[t], shapes[t]):
            print(f"{t:<3}{r['path'].name[-33:]:<34}"
                  f"{sh['n_actions']:>6}"
                  f"{str(sh.get('first_step')):>10}"
                  f"{_f(sh.get('first_t')):>11}"
                  f"{str(sh.get('first_qd')):>6}"
                  f"{_f(sh.get('reported')):>9}"
                  f"{_f(sh.get('pure_p50_at_first')):>13}"
                  f"{_f(sh.get('pure_p50_after')):>13}"
                  f"{str(sh.get('budget_range')):>16}")

    print("\n=== ★ 触发值的性质：报告的判据值 vs 同窗纯 decode p50 ===")
    print("（比值 ≈5 说明读数被 prefill 污染；比值 ≈1 说明报告的就是纯 decode）")
    print(f"{'臂':<3}{'run':<34}{'报告判据':>10}{'窗内纯decode p50':>18}{'比值':>8}")
    for t in ("A", "B"):
        for r, sh in zip(arms[t], shapes[t]):
            rep, pd = sh.get("reported"), sh.get("pure_p50_at_first")
            ratio = (rep / pd) if (rep and pd) else None
            print(f"{t:<3}{r['path'].name[-33:]:<34}{_f(rep,1):>10}{_f(pd,1):>18}"
                  f"{(f'{ratio:.2f}' if ratio else 'n/a'):>8}")

    print("\n=== 两臂用户指标（3 次重复）===")
    def med(vals):
        v = [x for x in vals if x is not None]
        return (statistics.median(v) if v else None), (max(v) - min(v) if len(v) > 1 else None)
    for key, name, unit in (("tpot_p50", "tpot p50", "ms"), ("stall_p99", "stall p99", "ms"),
                            ("slo_rate", "达标率", "")):
        line = f"  {name:<11}"
        for t in ("A", "B"):
            m, spread = med([r[key] for r in arms[t]])
            line += f" {t}: {'n/a' if m is None else f'{m:.4f}'} (极差 {spread if spread is None else round(spread, 4)})  "
        print(line)

    # ---- 4 条已登记的预测 ---------------------------------------------------
    print("\n=== 对照已登记的 4 条预测（pass / fail / unknown）===")
    b = shapes["B"]
    b_arms = arms["B"]
    def verdict(ok: bool | None, why: str) -> str:
        return "unknown" if ok is None else ("pass" if ok else "fail")

    firsts_t = [x["first_t"] for x in b if x["first_t"] is not None]
    firsts_qd = [x["first_qd"] for x in b if x["first_qd"] is not None]
    checks: list[tuple[str, bool | None, str]] = [
        ("① B 臂首次动作不在 t≈123s、且触发时队列不空（或干脆不动）",
         _check_first_action(firsts_t, firsts_qd, b_arms),
         f"B 臂首动作 t = {firsts_t}（A 臂/pilot 是 {PILOT_FIRST_ACTION_T_S}s）；"
         f"B 臂首动作队深 = {firsts_qd}"),
        ("② B 臂纯 decode 全程维持 ≈62ms（而非升到 118ms）",
         None if not any(x["pure_p50_after"] is not None for x in b) else
         all((x["pure_p50_after"] is None) or (x["pure_p50_after"] < 90.0) for x in b),
         f"B 臂动作后纯 decode p50 = {[x['pure_p50_after'] for x in b]}（阈值 <90ms；A/pilot 是 {PILOT_POST_SHRINK_MS}）"),
        # ⚠️ 判据**自校准**：要求 B 离 static 比离"干预前的 adaptive"更近。
        # 起初写成 |B − static| < 8ms —— 那**在"干预完全无效"的数据上也会 pass**
        # （|182.4 − 175.9| = 6.5 < 8.0），是个不会失败的判据。演练时才发现。
        ("③ B 臂 tpot p50 落到 static 一侧（离 static 比离干预前更近）",
         None if any(r["tpot_p50"] is None for r in b_arms) else
         (abs(med([r["tpot_p50"] for r in b_arms])[0] - STATIC_TPOT_P50_MS)
          < abs(med([r["tpot_p50"] for r in b_arms])[0] - PILOT_ADAPTIVE_TPOT_P50_MS)),
         f"B 臂 tpot p50 = {[r['tpot_p50'] for r in b_arms]}"
         f"（对照：static {STATIC_TPOT_P50_MS} / 干预前 adaptive {PILOT_ADAPTIVE_TPOT_P50_MS}）"),
        ("④ B 臂 stall p99 不再比 static 低 20%",
         None if any(r["stall_p99"] is None for r in b_arms) else
         (med([r["stall_p99"] for r in b_arms])[0] > ADAPTIVE_STALL_P99_MS + 20.0),
         f"B 臂 stall p99 = {[r['stall_p99'] for r in b_arms]}"
         f"（对照 static {STATIC_STALL_P99_MS} / adaptive {ADAPTIVE_STALL_P99_MS}）"),
    ]
    n_pass = n_fail = 0
    for name, ok, why in checks:
        v = verdict(ok, why)
        n_pass += v == "pass"; n_fail += v == "fail"
        print(f"  [{v:>7}] {name}\n            {why}")
    print(f"\n  小结：pass {n_pass} / fail {n_fail} / unknown {len(checks) - n_pass - n_fail}")
    print("  ⚠️ 若 ①–③ 全部 fail ⇒ E10 的因果链不成立，须回头重查；"
          "**不得**用「控制器没动」当结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
