#!/usr/bin/env python3
"""第⑩步：**轨迹上的静态响应面** → 按**预先登记**的规则冻结 `Best Static`。

⚠️ 本脚本是**规则的可执行形式**，不是"跑完再看怎么选"。
规则原文在 `NEXT_SESSION_TASKS.md` 第 10 步 §一/§二，**必须与那里逐字一致**；
若发现规则本身有缺陷，要**同时**改规则与理由，并标明"规则改动发生在看到数据之前还是之后"。

规则（复述，便于与原文对照）
--------------------------
* 主指标 `slo_goodput_tok_s`，按**段**（`low`/`high`/`recover`）取，
  用分阶段报告的 **window 口径**（分母 = 段墙钟长度）；跨 3 次重复取**中位**。
* 噪声底 `spread(s)` = **各候选在该段重复极差的上限**（保守取法）。
* **X 支配 Y** ⟺ 三段都有 `G(X,s) ≥ G(Y,s)`，且**至少一段** `G(X,s) − G(Y,s) > spread(s)`。
* `Best Static` = **不被任何其它候选支配**的候选；
  若 Pareto 前沿多于一个，平局规则按序：**high 段更高 → `chunk.enabled=False` 优先 →
  `max_batched_tokens` 更小优先**。
* ⚠️ 按段事后各选最优 = **oracle 参考**，单独打印并标注，**不是** Best Static。
* 若存在单一候选**支配其余全部** ⇒ 打印 **S5 触发**（回第 01 步改假设，不得直接扩大成本控制器）。

用法::

    python scripts/analyze_s10_surface.py --tag S10
    python scripts/analyze_s10_surface.py --tag S10 --json results/processed/s10_surface.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.stage_report import build_stage_report, build_stage_specs  # noqa: E402

# ⚠️ 复用 `report_stages` 的这两个函数，**不重写**：
# `requests.jsonl` **没有 `slo_satisfied` 字段**（第 05 步就发现过），必须按**该 run 自己的**
# `config.run.slo` 逐请求调权威的 `SLO.satisfied_by` 把它补上。
# 我第一版忘了这一步、把原始 dict 直接喂给 `build_stage_report` ⇒ 它认为没人达标
# ⇒ **逐段达标吞吐全是 0.000**，而且**不报错**（一个看起来像测量值的数）。
# 同一条教训本仓库已记过一次："加工具函数时先看有没有现成的"。
from report_stages import _requests_with_slo, _slo_of  # noqa: E402

#: 预注册的 5 个静态候选（顺序 = 报告顺序，也是平局规则里"配置序号"的依据）。
CELLS: tuple[str, ...] = (
    "ramp_static_c8",
    "ramp_static_c8_ck512",
    "ramp_static_c8_ck2048",
    "ramp_static_off_c8",
    "ramp_static_c8_bt2048",
)
SEGMENTS: tuple[str, ...] = ("low", "high", "recover")


# --------------------------------------------------------------------------- #
# 预注册规则（**模块级纯函数**，便于用合成数据测）
# --------------------------------------------------------------------------- #
# 为什么抽出来：这三条函数决定论文的基线（Best Static），却一度写成 `main` 里的闭包
# ——闭包**没法测**，而"决定基线的规则没有用例"正是本仓库反复警惕的那类风险。


def noise_floor(spreads: Mapping[str, Mapping[str, float]],
                segments: Sequence[str]) -> dict[str, float]:
    """噪声底 `spread(s)` = **各候选在该段重复极差的上限**（保守取法）。

    取上限而不是中位：只要有一个候选在该段抖得厉害，比较就该承认那段不可分辨。
    """
    return {s: max(spreads[c][s] for c in spreads) for s in segments}


def dominates(x: str, y: str, median: Callable[[str, str], float | None],
              floor: Mapping[str, float], segments: Sequence[str]) -> tuple[bool, str]:
    """X 支配 Y ⟺ 三段都有 `G(X,s) ≥ G(Y,s)`，且**至少一段**的优势 > `spread(s)`。

    ⚠️ "至少一段严格超过噪声底" 这一条不能省：否则任何一点点系统性偏差都算支配，
    而 `spread` 正是用来把"看起来更好"与"确实更好"分开的。
    数据缺失时**返回不支配**（宁可判不出，不可判错）。
    """
    if x == y:
        return False, ""
    strict: list[str] = []
    for s in segments:
        mx, my = median(x, s), median(y, s)
        if mx is None or my is None:
            return False, ""
        if mx < my:
            return False, ""
        if mx - my > floor[s]:
            strict.append(f"{s}(+{mx - my:.3f} > {floor[s]:.4f})")
    return (True, "、".join(strict)) if strict else (False, "")


def pareto_front(cells: Sequence[str], dominated_by: Mapping[str, Sequence[str]]) -> list[str]:
    """不被任何其它候选支配的候选（保持原顺序）。"""
    return [c for c in cells if not any(c in dominated_by[o] for o in cells)]


def rank_by_tie_rule(cells: Sequence[str],
                     high_goodput: Mapping[str, float | None],
                     chunk_enabled: Mapping[str, bool | None],
                     budget: Mapping[str, int | None]) -> list[str]:
    """平局规则（**顺序固定，不因数据而调**）：

    1. `high` 段的达标吞吐更高优先 —— 过载段是本研究的主战场；
    2. 仍平 ⇒ `chunk.enabled=False` 优先（更简单、更接近"不做任何配置"的基线）；
    3. 仍平 ⇒ `max_batched_tokens` 更小优先（更保守、更易部署）。

    取不到的值按"最差"处理（`-inf` / 不优先），**不填 0**。
    """
    return sorted(
        cells,
        key=lambda c: (
            -(high_goodput[c] if high_goodput[c] is not None else float("-inf")),
            0 if chunk_enabled[c] is False else 1,
            budget[c] if budget[c] is not None else 10**9,
        ),
    )


def sole_dominator(cells: Sequence[str], dominated_by: Mapping[str, Sequence[str]]) -> list[str]:
    """支配**其余全部**候选的那些候选（非空 ⇒ 触发 S5）。"""
    return [c for c in cells if len(dominated_by[c]) == len(cells) - 1]


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load(run_dir: Path) -> dict[str, Any]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    reqs = _read_jsonl(run_dir / "requests.jsonl")
    tel = _read_jsonl(run_dir / "telemetry.jsonl")
    fp = ((meta.get("context") or {}).get("extra") or {}).get("workload_fingerprint") or {}
    rep = meta.get("reproducibility") or {}
    s = meta.get("summary") or {}
    cfg = meta.get("config") or {}
    return dict(
        dir=run_dir, meta=meta, cfg=cfg, s=s,
        label=meta.get("data_label"), commit=(rep.get("git_commit") or "")[:12],
        dirty=rep.get("git_dirty"), n=s.get("n_requests"), ok=s.get("n_success"),
        dropped=(s.get("extra") or {}).get("dropped_requests"),
        #: 汇总里说有多少个达标 —— 用来做"逐段全 0"的**矛盾检查**（见 main 的 sanity 闸门）。
        n_slo_satisfied=s.get("n_slo_satisfied"),
        stages=build_stage_specs(fp.get("arrival_segments") or []),
        measured_window_ms=s.get("wall_ms"),
        # ⚠️ **必须**带上 `slo_satisfied`（raw 里没有）—— 否则达标吞吐恒为 0 且不报错。
        reqs=_requests_with_slo(reqs, _slo_of(meta)),
        tel=tel,
    )


def _segment_metrics(r: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """逐段指标（window 口径）。

    ⚠️ **不自己写一遍阈值判断**：直接调 `build_stage_report`，它的达标判定走的是
    权威的 `SLO.satisfied_by`（并已核对过与 `summary.slo_goodput_tok_s` 同分母）。
    """
    if not r["stages"]:
        return {}
    rep = build_stage_report(
        requests=r["reqs"], stages=r["stages"], telemetry=r["tel"],
        actions=(r["meta"].get("controller_trace") or []),
        measured_window_ms=r["measured_window_ms"],
    )
    out: dict[str, dict[str, Any]] = {}
    # ⚠️ **两套口径分开取，并且标出来**（第 05 步的核心纪律）：
    #   * 吞吐/达标 → **window** 口径（按事件发生时刻）——"这段时间系统产出多少"；
    #   * 延迟 → **cohort** 口径（按请求**到达**在哪一段）——"这一段进来的请求等了多久"。
    # 混成一个口径的后果在第 05 步已写过：既漏又重（high 段末尾到达的请求在 recover 完成）。
    # 出图时每个 panel 都必须写明用的是哪一套。
    cohorts = {c.label: c for c in rep.cohorts}
    for w in rep.windows:
        if w.label not in SEGMENTS:
            continue
        c = cohorts.get(w.label)
        out[w.label] = {
            "goodput_tok_s": w.slo_goodput_tok_s,
            "satisfaction_rate": w.slo_satisfaction_rate,
            "n_completed": w.n_completed,
            "queue_depth_max": w.queue_depth_max,
            "n_submitted": (c.n_submitted if c else None),
            "ttft_p50_ms": (c.ttft_ms_p50 if c else None),
            "ttft_p95_ms": (c.ttft_ms_p95 if c else None),
            "tpot_p50_ms": (c.tpot_ms_p50 if c else None),
            "tpot_p95_ms": (c.tpot_ms_p95 if c else None),
            "stall_p99_ms": (c.stall_ms_p99 if c else None),
            "insufficient_samples": (c.insufficient_samples if c else None),
        }
    return out


def _agg(rows: list[dict[str, Any]], cell: str, seg: str, key: str) -> dict[str, Any]:
    """把逐 run 的逐段某个量聚成中位/极差/n。**取不到就是 None**（不填 0）。"""
    vals = []
    for r in rows:
        if r["cell"] != cell:
            continue
        m = (r.get("segments") or {}).get(seg) or {}
        v = m.get(key)
        if v is not None:
            vals.append(v)
    if not vals:
        return {"median": None, "spread": None, "n": 0}
    return {"median": statistics.median(vals),
            "spread": (max(vals) - min(vals)) if len(vals) > 1 else 0.0,
            "n": len(vals)}


def _describe(cfg: dict[str, Any]) -> dict[str, Any]:
    c, ctl = cfg.get("chunk") or {}, cfg.get("controller") or {}
    return {
        "chunk_enabled": c.get("enabled"),
        "chunk_size": c.get("chunk_size"),
        "max_batched_tokens": c.get("max_num_batched_tokens"),
        "controller_enabled": ctl.get("enabled"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="第⑩步静态响应面 + Best Static 冻结")
    ap.add_argument("--out-root", default="results/raw")
    ap.add_argument("--tag", default="S10")
    ap.add_argument("--json", default=None, help="把响应面写成 JSON（建议放 results/processed/）")
    args = ap.parse_args(argv)

    root = Path(args.out_root)

    # ---- 载入 -------------------------------------------------------------
    per_run: dict[str, list[dict[str, Any]]] = {}
    problems: list[str] = []
    for cell in CELLS:
        dirs = sorted(d for d in root.glob(f"{args.tag}-*{cell}*") if d.is_dir())
        # ⚠️ 前缀匹配会同时命中 ck512 与其它的名字，用精确后缀过滤
        dirs = [d for d in dirs if d.name.endswith(cell) or d.name.endswith(f"{cell}-r1")
                or d.name.endswith(f"{cell}-r2")]
        if len(dirs) != 3:
            problems.append(f"{cell}: 找到 {len(dirs)} 个 run（应为 3）⇒ 该批还没跑完？")
        per_run[cell] = [_load(d) for d in dirs]

    missing = [p for p in problems]
    if missing:
        print("⚠️ 产物不全，先不判定：")
        for p in missing:
            print(f"   - {p}")
        print("   （跑完再来：本脚本**不会**用不完整的数据给出 Best Static）")
        return 2

    # ---- 可比性前置检查 ---------------------------------------------------
    allr = [r for rs in per_run.values() for r in rs]
    if {r["label"] for r in allr} != {"MEASURED"}:
        problems.append(f"标签不全是 MEASURED：{sorted({r['label'] for r in allr})}")
    if len({(r["commit"], r["dirty"]) for r in allr}) != 1:
        problems.append("**溯源戳不同** ⇒ 不是同版本对照")
    if {r["n"] for r in allr} != {92}:
        problems.append(f"请求数不一致：{sorted({r['n'] for r in allr})}")
    if any(r["ok"] != r["n"] for r in allr):
        problems.append("有 run 未全部成功")
    if {r["dropped"] for r in allr} != {0}:
        problems.append(f"有 dropped：{sorted({r['dropped'] for r in allr})}")
    print("=== 可比性前置检查 ===")
    if problems:
        for p in problems:
            print(f"  ✗ {p}")
        print("  ⇒ **不报差值、不判定**")
        return 1
    print("  ✓ 通过（同标签 / 同戳 / 同请求数 / 全成功 / dropped=0）")

    # ---- 逐 run 的逐段指标 → 逐格中位 + 极差 ------------------------------
    seg_vals: dict[str, dict[str, list[float]]] = {c: {s: [] for s in SEGMENTS} for c in CELLS}
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        for r in per_run[cell]:
            m = _segment_metrics(r)
            for s in SEGMENTS:
                if s in m and m[s]["goodput_tok_s"] is not None:
                    seg_vals[cell][s].append(m[s]["goodput_tok_s"])
            rows.append({"cell": cell, "run": r["dir"].name, **_describe(r["cfg"]),
                         "segments": m})

    def med(cell: str, s: str) -> float | None:
        v = seg_vals[cell][s]
        return statistics.median(v) if v else None

    def spread(cell: str, s: str) -> float:
        v = seg_vals[cell][s]
        return (max(v) - min(v)) if len(v) > 1 else 0.0

    #: 参与聚合的量：`(键, 显示名, 口径)`。口径必须显式携带——出图与结论都要写它。
    METRICS: tuple[tuple[str, str, str], ...] = (
        ("goodput_tok_s", "goodput tok/s", "window"),
        ("ttft_p50_ms", "TTFT p50 ms", "cohort"),
        ("tpot_p50_ms", "TPOT p50 ms", "cohort"),
        ("stall_p99_ms", "stall p99 ms", "cohort"),
    )

    print("\n=== 响应面：逐格逐段的达标吞吐（window 口径，3 次重复中位）===")
    print(f"{'格':<26}{'ck':>5}{'budget':>7}" + "".join(f"{s:>12}" for s in SEGMENTS)
          + "   组内极差(三段)")
    for cell in CELLS:
        d = _describe(per_run[cell][0]["cfg"])
        line = (f"{cell:<26}{str(d['chunk_size'] if d['chunk_enabled'] else 'off'):>5}"
                f"{str(d['max_batched_tokens']):>7}")
        for s in SEGMENTS:
            v = med(cell, s)
            line += f"{(f'{v:.3f}' if v is not None else 'n/a'):>12}"
        line += "   " + "/".join(f"{spread(cell, s):.4f}" for s in SEGMENTS)
        print(line)

    # ---- sanity 闸门：不可调和的矛盾 --------------------------------------------------
    # 为什么要有它：本文件第一版漏了"给请求补 `slo_satisfied`"，于是逐段达标吞吐**全是 0.000**。
    # 0 看起来像一个**测量值**（"这一段确实一个都没达标"），不报错、不留痕 ——
    # 比缺值危险得多。所以这里把"汇总说有人达标"与"逐段全 0"的冲突显形化。
    # ⇒ 守卫必须**能失败**：故意抽掉 `_requests_with_slo` 就会在这里被拦下。
    any_slo = any((r["n_slo_satisfied"] or 0) > 0 for r in allr)
    all_zero = all((med(c, s) or 0.0) == 0.0 for c in CELLS for s in SEGMENTS)
    if any_slo and all_zero:
        print("\n✗ sanity 闸门未通过：summary 里有 `n_slo_satisfied > 0`，"
              "而**逐段达标吞吐全为 0.000** —— 这是不可调和的矛盾。")
        print("  常见成因：请求没带 `slo_satisfied` 就喂给了 `build_stage_report`"
              "（raw 的 requests.jsonl 里没有这个字段）。")
        print("  ⇒ **拒绝继续**：0.000 看起来像测量值，比缺值更危险。")
        return 1

    # ---- 噪声底（各候选极差的上限，保守取法）------------------------------
    floor = {s: max(spread(c, s) for c in CELLS) for s in SEGMENTS}
    print("\n=== 噪声底 spread(s)（各候选重复极差的**上限**，保守）===")
    for s in SEGMENTS:
        print(f"  {s:<9} {floor[s]:.4f}")

    # ---- 支配关系 ---------------------------------------------------------
    def dominates(x: str, y: str) -> tuple[bool, str]:
        """X 支配 Y ⟺ 三段都 ≥，且至少一段的优势 > spread(s)。"""
        if x == y:
            return False, ""
        ge_all, strict = True, []
        for s in SEGMENTS:
            mx, my = med(x, s), med(y, s)
            if mx is None or my is None:
                return False, "数据缺失"
            if mx < my:
                ge_all = False
                break
            if mx - my > floor[s]:
                strict.append(f"{s}(+{mx - my:.3f} > {floor[s]:.4f})")
        if ge_all and strict:
            return True, "、".join(strict)
        return False, ""

    dom: dict[str, list[str]] = {c: [] for c in CELLS}
    rels: list[str] = []
    for x in CELLS:
        for y in CELLS:
            ok, why = dominates(x, y)
            if ok:
                dom[x].append(y)
                rels.append(f"  {x}  **支配**  {y}   （{why}）")
    print("\n=== 支配关系 ===")
    print("\n".join(rels) if rels else "  （没有任何支配关系 ⇒ 全部落在 Pareto 前沿）")

    nondominated = [c for c in CELLS if not any(c in dom[o] for o in CELLS)]
    print(f"\n=== Pareto 前沿（不被任何候选支配）===\n  {nondominated}")

    # ---- 平局规则（顺序固定）---------------------------------------------
    def tie_key(c: str) -> tuple:
        d = _describe(per_run[c][0]["cfg"])
        g_high = med(c, "high") or float("-inf")
        chunk_off = 0 if d["chunk_enabled"] is False else 1     # 越小越优先
        budget = d["max_batched_tokens"] or 0                    # 越小越优先
        return (-g_high, chunk_off, budget)

    ordered = sorted(nondominated, key=tie_key)
    best = ordered[0] if ordered else None
    print("\n=== 按预注册平局规则排序（high 段更高 → chunk 关优先 → budget 更小优先）===")
    for i, c in enumerate(ordered):
        mark = "  ← **Best Static**" if i == 0 else ""
        print(f"  {i + 1}. {c:<26} high={med(c, 'high'):.3f}{mark}")

    # ---- oracle 参考（**不是** Best Static）-------------------------------
    print("\n=== oracle 参考（**按段事后各选最优** —— 只作上界参考，不可部署）===")
    for s in SEGMENTS:
        cand = max(CELLS, key=lambda c: med(c, s) or float("-inf"))
        print(f"  {s:<9} 最优 = {cand:<26} {med(cand, s):.3f}")

    # ---- S5 触发判据 ------------------------------------------------------
    sole = [c for c in CELLS if len(dom[c]) == len(CELLS) - 1]
    if sole:
        print(f"\n⚠️⚠️ **S5 触发**：{sole[0]} **支配其余全部候选** ⇒ "
              "按 `RESEARCH_PLAN` §7，须**回第 01 步缩小/修改研究假设**，"
              "**不得**直接扩大成本控制器实现。")
    else:
        print("\n（未出现单一候选支配其余全部 ⇒ S5 未触发）")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "tag": args.tag, "cells": list(CELLS), "segments": list(SEGMENTS),
            "per_cell": {c: {s: {"median": med(c, s), "spread": spread(c, s),
                                 "values": seg_vals[c][s]} for s in SEGMENTS}
                         for c in CELLS},
            # 逐段延迟（cohort 口径）与吞吐（window 口径）分开存，**并带口径标签**
            "metrics": [{"key": k, "label": lb, "lens": lens} for k, lb, lens in METRICS],
            "per_cell_metrics": {
                c: {s: {k: _agg(rows, c, s, k) for k, _, _ in METRICS}
                    for s in SEGMENTS}
                for c in CELLS
            },
            "noise_floor": floor, "dominates": dom,
            "pareto": nondominated, "best_static": best,
            "tie_order": ordered, "s5_sole_dominator": sole or None,
            "per_run": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        print(f"\n已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
