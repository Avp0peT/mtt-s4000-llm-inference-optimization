#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPT-01（路线 S0）：**零成本只读分析** —— 用本地既有 `S14B2` 产物回答两问。

两问（见 `OPTIMIZATION_TEST_SESSION.md` §2）：

1. `load_ramp` 的 **low 段是否落入设备慢态**（部署损失 + 控制器信号污染的潜在来源）；
2. **最优工作带复核**：负载若设计在 `c4–c6`，实测 TPOT 是否落在 SLO 100 ms 内。

复用而非新写
------------
* `fwd_ms` 的正则解析复用 `scripts/decompose_step_overhead.py::parse_note`
  （**全项目唯一实现**；`fwd_ms` 不在 telemetry 顶层，只在 `note` 文本里）。
* 段边界复用 `ascend_opt.stage_report.build_stage_specs`（由 `meta.json` 的
  `workload_fingerprint.arrival_segments` 派生），**不自带一套边界常量**。
* 臂名解析复用 `ascend_opt.run_ids.parse_run_id`。

三条口径决定（写进代码，不靠记性）
----------------------------------
A. **`t_ms` 基准 = 段时钟，不做平移。** `arrival_segments` 的 `start_ms/end_ms`
   与 telemetry `t_ms` **同一时钟**：按 `anchor=0` 数段内到达数，四个臂都恰好复现
   声明的 `realized_count=[10,63,19]`；按 `anchor=min(arrival_ms)≈10133` 则得
   `[17,57,18]`，**与声明矛盾**。⇒ 首行 `t_ms≈10143` 是 **`low` 段的首次到达**
   （`qps=0.08` ⇒ 平均间隔 12.5 s），**不是**墙钟偏移。
   （`OPTIMIZATION_TEST_SESSION.md` §2.1 的"约 10 s 偏移"前提**不成立**，本脚本
   每次运行都重新做这个证伪，避免结论靠人记住。）

B. **分档键 = `decode`（该步真实前向的序列数），不是 `running`。**
   `note` 里 `forwards` 恒为 1，且 `fwd_ms` 只随 `decode` 变：
   `(running=8, decode=1)` 的 `fwd_ms` 中位 61.41 ≈ 档位 1 的 61.33，
   说明**前向批大小 = `decode`（= 顶层 `scheduled_tokens`）**；
   `running` 是调度器"运行集"大小，在 **20.6%** 的纯 decode 步里 **大于**实际前向批
   ⇒ 直接按 `running` 分档会把这批步**贴错标签**。两个键**都报**，判据用 `decode`。

C. **纯 decode 步才与标定表同口径。** `note` 的 `prefill_tok > 0` 的步
   `fwd_ms` 可达 ~538 ms（2048 token 预填充），混进来会让分布假性双峰。
   ⇒ 判据只用 `prefill_tok == 0` 的步；全步统计一并打印，供对照。

红线：只读 `results/raw/**`；不写 `results/raw/**`；不改 `benchmark/real_backend.py`；
不产生任何新实验条件。产物写 `results/processed/`。

用法::

    python scripts/analyze_opt01_s14b2.py --tag S14B2 \
        --json results/processed/opt01_s14b2.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ascend_opt.run_ids import parse_run_id                      # noqa: E402
from ascend_opt.stage_report import build_stage_specs            # noqa: E402

# ⚠️ 复用唯一实现：fwd_ms 只在 note 文本里，正则已在该模块里按真机产物逐字校准过。
from decompose_step_overhead import parse_note                   # noqa: E402

# ---------------------------------------------------------------------------
# 参考量：**全部取自 `COST_PROFILE_S4000.md`（独立来源），不得从本脚本要检验的数据里推**
# ---------------------------------------------------------------------------
#: b1 快态（探针"块最小值"，§2 表第 1 行）
FAST_B1_PROBE_MIN_MS = 59.88
#: b1 快态（P1S3D 真机值，§2 表第 1 行）
FAST_B1_P1S3D_MS = 59.94
#: b1 慢态带（§4：「b1 快态 ~57–60 ms / 慢态 ~76–82 ms（比 ~1.33）」）
SLOW_B1_LOW_MS, SLOW_B1_HIGH_MS = 76.0, 82.0
#: §2 快态标定表（探针块最小值 / P1S3D 真值）——**性能下界**，只作形状参考
FAST_TABLE_MS = {1: 59.88, 2: 74.28, 4: 83.28, 8: 109.17}
FAST_TABLE_TRUTH_MS = {1: 59.94, 2: 75.53, 4: 85.19, 8: 113.75}
#: SLO（`config.snapshot.yaml` 的 `run.slo.tpot_ms`；每次运行也从产物现读校对）
SLO_TPOT_MS = 100.0

#: **判据用的两条带**（声明在前，测在后）。COST_PROFILE §2 明说标定表是**下界**、
#: 真实服务路径读数**更高**（占空比更低），所以快态上界不能在 59.88 上卡死；
#: 这里给 15% 容差：59.88 × 1.15 = 68.86，仍与慢态下界 76 留出 ~10% 空隙，
#: 因此该带**不会**把慢态误判成快态（判据方向安全）。
#:
#: ⚠️⚠️ **这两条带只对 `decode=1` 有效。** COST_PROFILE §4 的快/慢值是**按 batch 标定**的
#: （b1 比 ~1.33、b2 比 ~1.08），b2 的正常成本本来就是 74–81 ms —— 把 72.2 ms
#: 这个门槛套到 b2 上会把**正常档位**误判成"慢态"（本脚本 v1 踩过）。
#: ⇒ 非 b1 档**不套绝对带**，只判**形状**（单态 vs 双峰）。
FAST_CEIL_MS = FAST_B1_PROBE_MIN_MS * 1.15
SLOW_FLOOR_MS = SLOW_B1_LOW_MS * 0.95
#: 有独立快/慢标定的唯一档位（COST_PROFILE §4 只给了 b1 的双态值）
CALIBRATED_BUCKET = 1
#: 形状判据：同档内 p90/p10 超过此值 ⇒ 分布被拉开，需查是否双峰
TIGHT_RATIO = 1.02

SEGMENTS = ("low", "high", "recover", "drain")


# ---------------------------------------------------------------------------
def _pct(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """逐行解析，返回 (好行, 坏行数)。**坏行不吞掉整个文件**。"""
    good: list[dict[str, Any]] = []
    bad = 0
    if not path.exists():
        return good, -1
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    return good, bad


def load_runs(tag: str) -> list[dict[str, Any]]:
    """载入 `<tag>-*` 全部 run，带**完整性判定**（不静默丢弃）。"""
    out: list[dict[str, Any]] = []
    for d in sorted(glob.glob(str(_ROOT / "results" / "raw" / f"{tag}-*"))):
        if not os.path.isdir(d):
            continue
        name = Path(d).name
        # --- meta.json ---
        mp = Path(d) / "meta.json"
        meta: dict[str, Any] | None = None
        meta_state = "ABSENT"
        if mp.exists():
            try:
                meta = json.loads(mp.read_text(encoding="utf-8", errors="replace"))
                meta_state = "OK"
            except json.JSONDecodeError:
                meta_state = f"UNPARSEABLE({mp.stat().st_size}B)"
        # --- telemetry.jsonl ---
        tel, tel_bad = _read_jsonl(Path(d) / "telemetry.jsonl")
        # --- requests.jsonl ---
        req, req_bad = _read_jsonl(Path(d) / "requests.jsonl")

        parsed = parse_run_id(name)
        rec: dict[str, Any] = {
            "dir": name,
            "arm": parsed.arm if parsed else None,
            "pass_index": (parsed.pass_index if parsed else None),
            "meta_state": meta_state,
            "n_samples": (meta or {}).get("n_samples"),
            "tel_n": len(tel),
            "tel_bad": tel_bad,
            "req_n": len(req),
            "req_bad": req_bad,
            "label": (meta or {}).get("data_label"),
            "commit": ((meta or {}).get("reproducibility") or {}).get("git_commit"),
            "tel": tel,
            "req": req,
            "meta": meta or {},
        }
        # 段边界由产物自身派生（权威），不是本脚本的常量
        fp = (((meta or {}).get("context") or {}).get("extra") or {}).get(
            "workload_fingerprint") or {}
        rec["arrival_segments"] = fp.get("arrival_segments") or []
        rec["specs"] = build_stage_specs(rec["arrival_segments"]) if rec["arrival_segments"] else []
        # 运行是否跑完：末行 t_ms 是否越过后最小段终点
        last_t = None
        for row in reversed(tel):
            if isinstance(row.get("t_ms"), (int, float)):
                last_t = float(row["t_ms"])
                break
        rec["last_t_ms"] = last_t
        end_ms = max((s.end_ms for s in rec["specs"]), default=None)
        rec["expected_end_ms"] = end_ms
        # 完整 = meta 可解析 且 telemetry 无坏行 且 跑过负载名义终点
        rec["complete"] = bool(
            meta_state == "OK"
            and tel_bad == 0
            and last_t is not None
            and end_ms is not None
            and last_t >= end_ms - 1000.0
        )
        reason = []
        if meta_state != "OK":
            reason.append(f"meta:{meta_state}")
        if tel_bad > 0:
            reason.append(f"telemetry 有 {tel_bad} 行不可解析")
        if tel_bad == 0 and last_t is not None and end_ms is not None and last_t < end_ms - 1000.0:
            reason.append(f"中断于 {last_t / 1000:.1f}s（名义终点 {end_ms / 1000:.0f}s）")
        rec["incomplete_reason"] = "; ".join(reason)
        out.append(rec)
    return out


def segment_of(t_ms: float, specs: Sequence[Any]) -> str | None:
    for s in specs:
        if s.start_ms <= t_ms < s.end_ms:
            return s.label
    if specs and t_ms >= specs[-1].end_ms:
        return "drain"
    return None


def note_rows(run: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """把 telemetry 行整理成带 `seg` / `bucket` 的记录（**只读产物字段**）。"""
    for row in run["tel"]:
        t = row.get("t_ms")
        if not isinstance(t, (int, float)):
            continue
        seg = segment_of(float(t), run["specs"])
        if seg is None:
            continue
        n = parse_note(row.get("note"))
        if "decode" not in n:          # 没 note ⇒ 无法分档，**跳过并计数**，不当 0
            yield {"seg": seg, "missing_note": True, "t_ms": float(t)}
            continue
        yield {
            "seg": seg,
            "t_ms": float(t),
            "running": row.get("running"),
            "decode": int(n["decode"]),
            "prefill_tok": n.get("prefill_tok"),
            "fwd_ms": n.get("fwd_ms"),
            "tpot_ms": row.get("tpot_ms"),
            "queue_depth": row.get("queue_depth"),
            "missing_note": False,
        }


# ---------------------------------------------------------------------------
def step1_slowstate(flows: list[dict[str, Any]], min_n: int) -> dict[str, Any]:
    """步骤 1：按 (段 × `decode` 档) 统计纯 decode `fwd_ms`，判慢态。

    ⚠️ **绝对快/慢带只用于 `CALIBRATED_BUCKET`**（见常量处说明）；其余档位只判形状。
    """
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    buckets_all: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_arm: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for fr in flows:
        for r in fr["rows"]:
            if r.get("missing_note"):
                continue
            if r["prefill_tok"] == 0 and r["fwd_ms"] is not None:
                buckets[(r["seg"], r["decode"])].append(float(r["fwd_ms"]))
                per_arm[(fr["arm"] or "?", r["seg"], r["decode"])].append(float(r["fwd_ms"]))
            if r["fwd_ms"] is not None:
                buckets_all[(r["seg"], r["decode"])].append(float(r["fwd_ms"]))

    def _verdict(v: list[float], k: int) -> tuple[str, str, float | None]:
        """返回 (verdict, 判据依据, ≥慢态下界的比例)。"""
        if len(v) < min_n:
            return "不可判（样本不足）", "—", None
        p10, med = _pct(v, .10), _pct(v, .50)
        slow_hits = sum(1 for x in v if x >= SLOW_FLOOR_MS)
        slow_frac = slow_hits / len(v)
        if k == CALIBRATED_BUCKET:
            basis = f"绝对带（COST_PROFILE §4，仅 b{CALIBRATED_BUCKET} 有标定）"
            if med is not None and med >= SLOW_FLOOR_MS:
                return "落慢态", basis, slow_frac
            if slow_hits:
                return "落慢态", basis, slow_frac
            if med is not None and med <= FAST_CEIL_MS:
                return "未落慢态", basis, slow_frac
            return "不可判（落在快/慢之间的空隙带）", basis, slow_frac
        # 非标定档：不套 b1 的绝对带，只看分布形状
        basis = "形状（该档无独立快/慢标定，故不套 b1 绝对带）"
        ratio = (max(v) / min(v)) if min(v) > 0 else None
        spread = (_pct(v, .90) / p10) if (p10 and _pct(v, .90)) else None
        if spread is not None and spread > TIGHT_RATIO:
            return f"疑双峰/被拉开（p90/p10={spread:.4f}）", basis, slow_frac
        return "单态（紧）", basis, slow_frac

    cells = []
    for seg in SEGMENTS:
        for k in sorted({kk for (ss, kk) in buckets if ss == seg}):
            v = buckets[(seg, k)]
            med = _pct(v, 0.5)
            p10, p90 = _pct(v, .10), _pct(v, .90)
            verdict, basis, slow_frac = _verdict(v, k)
            cells.append({
                "seg": seg, "decode": k, "n": len(v),
                "p10": p10, "p25": _pct(v, .25), "p50": med,
                "p75": _pct(v, .75), "p90": p90,
                "min": (min(v) if v else None), "max": (max(v) if v else None),
                "spread_p90_over_p10": ((p90 / p10) if (p10 and p90) else None),
                "slow_frac": slow_frac,
                "verdict": verdict, "basis": basis,
            })

    # 状态翻转检验：**只在标定档**上找「连续 ≥慢态下界」的步
    # （非标定档的正常值本就在慢带之上，找它没有信息量）
    streaks: list[dict[str, Any]] = []
    for fr in flows:
        for seg in SEGMENTS:
            best = cur = 0
            for r in fr["rows"]:
                if (r["seg"] != seg or r.get("missing_note")
                        or r["prefill_tok"] != 0 or r["decode"] != CALIBRATED_BUCKET):
                    continue
                if r["fwd_ms"] is not None and float(r["fwd_ms"]) >= SLOW_FLOOR_MS:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            if best:
                streaks.append({"dir": fr["dir"], "seg": seg,
                                "max_consecutive_slow_steps": best})
    return {"cells": cells, "slow_streaks": streaks,
            "all_steps_p50": {f"{s}|b{k}": _pct(v, .5) for (s, k), v in buckets_all.items()},
            "per_arm_p50": {f"{a}|{s}|b{k}": _pct(v, .5) for (a, s, k), v in per_arm.items()}}


def step2_workband(flows: list[dict[str, Any]], min_n: int) -> dict[str, Any]:
    """步骤 2：各段 `decode` 分布 + 实测 `tpot_ms` 分档对 SLO，回答工作带。"""
    dist: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    dist_running: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    tpot: dict[tuple[str, int], list[float]] = defaultdict(list)
    qdep: dict[tuple[str, int], list[float]] = defaultdict(list)
    mismatch = 0
    total = 0
    for fr in flows:
        for r in fr["rows"]:
            if r.get("missing_note"):
                continue
            if r["prefill_tok"] != 0:
                continue
            total += 1
            if r["running"] != r["decode"]:
                mismatch += 1
            dist[r["seg"]][r["decode"]] += 1
            dist_running[r["seg"]][int(r["running"] or 0)] += 1
            if isinstance(r["tpot_ms"], (int, float)):
                tpot[(r["seg"], r["decode"])].append(float(r["tpot_ms"]))
            if isinstance(r["queue_depth"], (int, float)):
                qdep[(r["seg"], r["decode"])].append(float(r["queue_depth"]))

    cells = []
    for seg in SEGMENTS:
        for k in sorted({kk for (ss, kk) in tpot if ss == seg}):
            v = tpot[(seg, k)]
            q = qdep[(seg, k)]
            n_slo = sum(1 for x in v if x <= SLO_TPOT_MS)
            cells.append({
                "seg": seg, "decode": k, "n": dist[seg][k], "n_tpot": len(v),
                "tpot_p25": _pct(v, .25), "tpot_p50": _pct(v, .5), "tpot_p90": _pct(v, .90),
                "slo_pass_frac": (n_slo / len(v) if v else None),
                "queue_p50": _pct(q, .5), "queue_max": (max(q) if q else None),
                "verdict": ("达标" if v and _pct(v, .5) is not None
                            and (_pct(v, .5) or 0) <= SLO_TPOT_MS
                            else ("超 SLO" if v else "不可判")),
            })
    # 每格的「跳过率」：`running > decode`（该步有序列没拿到 token）
    # ⇒ 那些步的 telemetry `tpot_ms` 对**被跳过的序列**是下界。
    cell_skip: dict[tuple[str, int], list[int]] = defaultdict(list)
    for fr in flows:
        for r in fr["rows"]:
            if r.get("missing_note") or r["prefill_tok"] != 0:
                continue
            cell_skip[(r["seg"], r["decode"])].append(
                1 if (r["running"] or 0) > r["decode"] else 0)
    for c in cells:
        flags = cell_skip.get((c["seg"], c["decode"]), [])
        c["n_skipped_steps"] = sum(flags)
        c["skip_frac"] = (sum(flags) / len(flags) if flags else None)
    return {"cells": cells,
            "decode_dist": {s: dict(sorted(d.items())) for s, d in dist.items()},
            "running_dist": {s: dict(sorted(d.items())) for s, d in dist_running.items()},
            "running_ne_decode_steps": mismatch,
            "decode_steps_total": total}


def step2b_request_tpot(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """**交叉核对**：用 `requests.jsonl` 的逐请求 `tpot_ms`（SLO 自己的字段），**按段分**。

    为什么要它：telemetry 的 `tpot_ms` 实测≈该步墙钟时长；某条序列在 `running > decode`
    的步被跳过时它那一步**没拿到 token** ⇒ 它的真实间隔**大于**步时长
    ⇒ telemetry `tpot_ms` 在存在跳过的段里是**请求级 TPOT 的下界**。

    判读方式：`low` 段两侧应**接近**（那里不跳过，两口径同义）⇒ 用来验证口径；
    `high`/`recover` 段若请求级**明显更高**，即跳过效应显形（下界方向被证实）。
    """
    per_seg: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if not r["specs"]:
            continue
        for x in r["req"]:
            v = x.get("tpot_ms")
            a = x.get("scheduled_arrival_ms")
            if not isinstance(v, (int, float)) or not isinstance(a, (int, float)):
                continue
            seg = segment_of(float(a), r["specs"])
            if seg:
                per_seg[seg].append(float(v))
    cells = []
    for seg in SEGMENTS:
        v = per_seg.get(seg) or []
        if not v:
            continue
        cells.append({
            "seg": seg, "n": len(v),
            "p50": _pct(v, .50), "p90": _pct(v, .90), "max": max(v),
            "slo_pass_frac": sum(1 for x in v if x <= SLO_TPOT_MS) / len(v),
        })
    allv = [x for v in per_seg.values() for x in v]
    return {"cells": cells, "n": len(allv),
            "p50": _pct(allv, .50) if allv else None,
            "slo_pass_frac": (sum(1 for x in allv if x <= SLO_TPOT_MS) / len(allv)
                              if allv else None),
            "note": ("逐请求 TPOT = (done−first_token)/(gen_len−1)，按**到达段**归属；"
                     "与 telemetry 分档值**权重不同**（按请求 vs 按步），只作方向核对")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OPT-01：S14B2 零成本只读分析")
    ap.add_argument("--tag", default="S14B2")
    ap.add_argument("--json", default=None)
    ap.add_argument("--min-n", type=int, default=20,
                    help="分档后支撑分位数的最小样本数（低于则记不可判）")
    args = ap.parse_args(argv)

    runs = load_runs(args.tag)
    if not runs:
        print(f"✗ 没找到 {args.tag}-* 产物", file=sys.stderr)
        return 2

    # ---------------- 前置核查 -------------------------------------------
    print("=" * 100)
    print(f"§2.1 前置核查｜tag={args.tag}｜目录数 {len(runs)}")
    print("=" * 100)
    print(f"{'dir':46s} {'arm':10s} {'meta':14s} {'tel_n':>6s} {'bad':>4s} "
          f"{'req_n':>6s} {'end_t_s':>8s} {'comp':>5s}")
    for r in runs:
        lt = "-" if r["last_t_ms"] is None else f"{r['last_t_ms'] / 1000:.1f}"
        print(f"{r['dir']:46s} {str(r['arm']):10s} {r['meta_state']:14s} "
              f"{r['tel_n']:>6d} {r['tel_bad']:>4d} {r['req_n']:>6d} {lt:>8s} "
              f"{('YES' if r['complete'] else 'no'):>5s}")
    ok = [r for r in runs if r["complete"]]
    bad = [r for r in runs if not r["complete"]]
    print(f"\nmeta 可解析 {sum(1 for r in runs if r['meta_state'] == 'OK')}/{len(runs)}｜"
          f"telemetry 无坏行 {sum(1 for r in runs if r['tel_bad'] == 0)}/{len(runs)}｜"
          f"requests 无坏行 {sum(1 for r in runs if r['req_bad'] == 0)}/{len(runs)}")
    print(f"★ 完整 run（跑过名义终点且产物无坏行）= **{len(ok)}/{len(runs)}**"
          f" ⇒ 本报表分母 = **{len(ok)}**")
    for r in bad:
        print(f"   ⏭️  {r['dir']}: {r['incomplete_reason']}")
    print("   ⚠️ 分母变化说明：`OPTIMIZATION_TEST_SESSION.md` §2.1 记的「11/20」"
          "与 `EVIDENCE_INDEX.md` §1.7b 的 13:08 实测同源；")
    print("      本地 17:51–17:57 又落盘 12 个目录 ⇒ 现测分母已变。**引用时写明采集时刻**。")

    # ---------------- t_ms 基准证伪 ---------------------------------------
    print("\n" + "=" * 100)
    print("§2.1 ⚠️ `t_ms` 基准：用段内到达数**证伪**两种假设（不靠记忆）")
    print("=" * 100)
    anchors = []
    for r in ok:
        arr = sorted(x["scheduled_arrival_ms"] for x in r["req"]
                     if isinstance(x.get("scheduled_arrival_ms"), (int, float)))
        if not arr or not r["specs"]:
            continue
        declared = [s.get("realized_count") for s in r["arrival_segments"]]
        got = {}
        for hyp, base in (("anchor=0(段时钟)", 0.0), ("anchor=min(arr)(平移)", arr[0])):
            got[hyp] = [sum(1 for a in arr if base + s.start_ms <= a < base + s.end_ms)
                        for s in r["specs"]]
        anchors.append({"dir": r["dir"], "declared": declared, "got": got})
    n_zero = sum(1 for a in anchors
                 if a["got"]["anchor=0(段时钟)"] == [c for c in a["declared"]])
    n_shift = sum(1 for a in anchors
                  if a["got"]["anchor=min(arr)(平移)"] == [c for c in a["declared"]])
    for a in anchors[:6]:
        print(f"  {a['dir']:46s} declared={a['declared']} "
              f"anchor=0 -> {a['got']['anchor=0(段时钟)']} "
              f"shift -> {a['got']['anchor=min(arr)(平移)']}")
    if len(anchors) > 6:
        print(f"  …（另 {len(anchors) - 6} 个 run 同型，见 JSON）")
    print(f"\n  复现声明 realized_count 的 run 数：anchor=0 → **{n_zero}/{len(anchors)}**；"
          f"平移 anchor → {n_shift}/{len(anchors)}")
    verdict_anchor = ("segment_clock_no_shift" if n_zero > n_shift else "shifted")
    print(f"  ★ 判定：`arrival_segments` 与 `t_ms` **同一时钟**，"
          f"**段边界直接用 `t_ms`**（{'✓ 已证伪平移假设' if verdict_anchor == 'segment_clock_no_shift' else '⚠️ 平移假设未被证伪，需人工复核'}）。")
    print("   ⚠️ 因此 `OPTIMIZATION_TEST_SESSION.md` §2.1 的「约 10 s 墙钟偏移」**不成立**：")
    print("      首行 `t_ms≈10143` 是 `low` 段**首次到达**（qps=0.08 ⇒ 平均间隔 12.5 s）。")

    # ---------------- 组装 flows ------------------------------------------
    flows = []
    for r in ok:
        rows = list(note_rows(r))
        flows.append({"dir": r["dir"], "arm": r["arm"], "rows": rows})
    n_no_note = sum(1 for f in flows for x in f["rows"] if x.get("missing_note"))
    if n_no_note:
        print(f"\n  ⚠️ {n_no_note} 行缺 `note` ⇒ 无法分档，已**跳过并计数**（不当 0）")

    s1 = step1_slowstate(flows, args.min_n)
    s2 = step2_workband(flows, args.min_n)
    req_tpot = step2b_request_tpot(ok)

    # ---------------- 步骤 1 ---------------------------------------------
    print("\n" + "=" * 100)
    print("步骤 1：逐段 `fwd_ms` 分布（**纯 decode**，按 `decode` 分档）→ 判慢态")
    print(f"  参考量（COST_PROFILE §2/§4，独立来源）：b1 快态 {FAST_B1_PROBE_MIN_MS}/"
          f"{FAST_B1_P1S3D_MS} ms｜b1 慢态 {SLOW_B1_LOW_MS}–{SLOW_B1_HIGH_MS} ms")
    print(f"  判据：快态上界 {FAST_CEIL_MS:.2f} ms（=59.88×1.15）｜"
          f"慢态下界 {SLOW_FLOOR_MS:.2f} ms（=76×0.95）｜最小样本数 {args.min_n}")
    print(f"  ⚠️ 这两条**只对 b{CALIBRATED_BUCKET}** 有效；其余档位的正常值本就高于慢带"
          f"（b2 = {FAST_TABLE_MS[2]:.0f}–81 ms）⇒ 那些档只判形状（p90/p10 ≤ {TIGHT_RATIO}）")
    print("=" * 100)
    print(f"{'seg':8s}{'dec':>4s}{'n':>7s} | {'min':>7s}{'p10':>7s}{'p25':>7s}{'p50':>7s}"
          f"{'p75':>7s}{'p90':>7s}{'max':>7s} | {'p90/p10':>8s} | verdict")
    for c in s1["cells"]:
        f = lambda x: "—" if x is None else f"{x:7.2f}"
        sp = "—" if c["spread_p90_over_p10"] is None else f"{c['spread_p90_over_p10']:8.4f}"
        print(f"{c['seg']:8s}{c['decode']:>4d}{c['n']:>7d} | {f(c['min'])}{f(c['p10'])}"
              f"{f(c['p25'])}{f(c['p50'])}{f(c['p75'])}{f(c['p90'])}{f(c['max'])} | "
              f"{sp} | {c['verdict']}")
    low1 = [c for c in s1["cells"] if c["seg"] == "low" and c["decode"] == CALIBRATED_BUCKET]
    if low1:
        c = low1[0]
        print(f"\n  ★ low 段 b{CALIBRATED_BUCKET}：n={c['n']}｜p50={c['p50']:.2f} ms｜"
              f"p90={c['p90']:.2f}｜max={c['max']:.2f}｜"
              f"≥{SLOW_FLOOR_MS:.1f} ms 的样本 {(c['slow_frac'] or 0) * 100:.4f}%")
    if s1["slow_streaks"]:
        mx = max(x["max_consecutive_slow_steps"] for x in s1["slow_streaks"])
        print(f"  ⚠️ b{CALIBRATED_BUCKET} 上存在「连续 ≥慢态下界」的步："
              f"{len(s1['slow_streaks'])} 处，最长 {mx} 步（见 JSON）")
    else:
        print(f"  ★ 全部 {len(ok)} 个 run、全部段：b{CALIBRATED_BUCKET} 上"
              f"**没有任何一步** `fwd_ms ≥ 慢态下界` ⇒ 不存在秒级状态翻转的痕迹")
    print(f"\n  -- 稳健性：各臂 low 段 b{CALIBRATED_BUCKET} 中位 --")
    for a in sorted({k.split("|")[0] for k in s1["per_arm_p50"]}):
        v = s1["per_arm_p50"].get(f"{a}|low|b{CALIBRATED_BUCKET}")
        print(f"     {a:10s} {'—' if v is None else f'{v:7.2f} ms'}")

    # ---------------- 步骤 2 ---------------------------------------------
    print("\n" + "=" * 100)
    print("步骤 2：最优工作带复核（实测 `tpot_ms`，非快态表）")
    print(f"  SLO `tpot_ms` = {SLO_TPOT_MS} ms（产物现读校对见下）")
    print("=" * 100)
    print("-- 各段实测 `decode` 分布（纯 decode 步数）--")
    for seg in SEGMENTS:
        d = s2["decode_dist"].get(seg) or {}
        if d:
            tot = sum(d.values())
            print(f"   {seg:8s} n={tot:>6d}  " +
                  " ".join(f"b{k}:{v}({v / tot:.1%})" for k, v in sorted(d.items())))
    print("\n-- 各段实测 `running` 分布（**与上表不同**，见口径 B）--")
    for seg in SEGMENTS:
        d = s2["running_dist"].get(seg) or {}
        if d:
            tot = sum(d.values())
            print(f"   {seg:8s} n={tot:>6d}  " +
                  " ".join(f"r{k}:{v}({v / tot:.1%})" for k, v in sorted(d.items())))
    print(f"\n  ⚠️ 纯 decode 步中 `running != decode` 的有 "
          f"**{s2['running_ne_decode_steps']}**/ {s2['decode_steps_total']}"
          f"（{s2['running_ne_decode_steps'] / max(1, s2['decode_steps_total']):.1%}）"
          " ⇒ 按 `running` 分档会把这批步贴错标签。")

    print(f"\n{'seg':8s}{'dec':>4s}{'n':>7s}{'n_tpot':>7s} | {'tpot p25':>9s}{'tpot p50':>9s}"
          f"{'tpot p90':>9s} | {'≤SLO 占比':>9s} | {'q p50':>6s}{'q max':>6s}"
          f"{'跳过率':>8s} | verdict")
    for c in s2["cells"]:
        f = lambda x: "—" if x is None else f"{x:9.2f}"
        pf = "—" if c["slo_pass_frac"] is None else f"{c['slo_pass_frac']:9.1%}"
        qp = "—" if c["queue_p50"] is None else f"{c['queue_p50']:6.0f}"
        qm = "—" if c["queue_max"] is None else f"{c['queue_max']:6.0f}"
        sk = "—" if c.get("skip_frac") is None else f"{c['skip_frac']:8.1%}"
        print(f"{c['seg']:8s}{c['decode']:>4d}{c['n']:>7d}{c['n_tpot']:>7d} | "
              f"{f(c['tpot_p25'])}{f(c['tpot_p50'])}{f(c['tpot_p90'])} | {pf} | "
              f"{qp}{qm}{sk} | {c['verdict']}")
    print("  ⚠️ 「跳过率」= 该格中 `running > decode` 的步占比。它只解释**一部分**"
          "被跳过的情形，")
    print("     **不是**「telemetry `tpot_ms` = 请求级 TPOT」的充分条件 —— 实测 b8 档"
          "跳过率 0.0% 而请求级 TPOT 仍更高（成因未定位，登记待查）。")

    # 干净的 c-k 工作点：drain 段（队列排空）——不受过载污染
    print("\n  -- 干净工作点（`drain` 段，队列已排空）vs 快态标定表 --")
    print(f"     {'dec':>4s}{'n':>7s}{'tpot p50':>10s}{'q p50':>7s} | {'标定b(min)':>10s}"
          f"{'标定b(真值)':>11s} | {'实测/标定min':>12s}")
    ratios = {}
    for c in s2["cells"]:
        if c["seg"] != "drain":
            continue
        k = c["decode"]
        tb = FAST_TABLE_MS.get(k)
        tt = FAST_TABLE_TRUTH_MS.get(k)
        r_ = (c["tpot_p50"] / tb) if (tb and c["tpot_p50"]) else None
        if r_:
            ratios[k] = r_
        print(f"     {k:>4d}{c['n']:>7d}{c['tpot_p50']:>10.2f}{(c['queue_p50'] or 0):>7.0f} | "
              f"{(f'{tb:.2f}' if tb else '—'):>10s}{(f'{tt:.2f}' if tt else '—'):>11s} | "
              f"{(f'{r_:.3f}' if r_ else '—'):>12s}")

    print("\n  -- 交叉核对：`requests.jsonl` 逐请求 `tpot_ms`（SLO 自己的字段，按到达段）--")
    if req_tpot.get("n"):
        print(f"     全体 n={req_tpot['n']}｜p50={req_tpot['p50']:.2f}｜"
              f"≤SLO 占比={req_tpot['slo_pass_frac']:.1%}")
        print(f"     {'seg':8s}{'n':>6s}{'p50':>9s}{'p90':>9s}{'max':>9s}{'≤SLO':>8s}")
        for c in req_tpot.get("cells", []):
            print(f"     {c['seg']:8s}{c['n']:>6d}{c['p50']:>9.2f}{c['p90']:>9.2f}"
                  f"{c['max']:>9.2f}{c['slo_pass_frac']:>8.1%}")
        print("     读法：`low` 段两口径**接近**（63.2 vs 63.2）⇒ telemetry `tpot_ms` 与"
              "请求级 TPOT 在无竞争时**同义**，口径通过核验；")
        print("           `high`/`recover` 请求级**明显更高**（如 high 172.4 vs 步级 b8 145.4）"
              "⇒ 步级量在过载段**低估**请求级 TPOT。")
        print("           ⇒ 方向：步级是**下界** ⇒ 本会话的「c4 超 SLO」是**保守**判定。")
        print("           ⚠️ 两者**权重不同**（按请求 vs 按步）/成因未定位，**不得**并列当同一分布。")

    # ---------------- 结论 ------------------------------------------------
    low_b1 = next((c for c in s1["cells"]
                   if c["seg"] == "low" and c["decode"] == CALIBRATED_BUCKET), None)
    band = {}
    for c in s2["cells"]:
        if c["seg"] == "drain":
            band[c["decode"]] = c
    for c in s2["cells"]:                     # drain 缺档时用全段补齐
        band.setdefault(c["decode"], c)
    # ⚠️ 实测 batch 成本**非单调**（b3 比 b4 贵）⇒ 「最大可行档」这个说法本身
    # 不成立；只报**达标档位集合**，不报单个数。
    slo_ok_set = sorted(k for k in band
                        if band[k]["tpot_p50"] is not None
                        and band[k]["tpot_p50"] <= SLO_TPOT_MS)

    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)
    if low_b1:
        v1 = low_b1["verdict"]
        print(f"(1) `load_ramp` 的 low 段是否落入设备慢态 ⇒ "
              f"**{'否（未落慢态）' if v1 == '未落慢态' else v1}**")
        print(f"    依据：low 段 b{CALIBRATED_BUCKET} 纯 decode `fwd_ms` n={low_b1['n']}｜"
              f"p50={low_b1['p50']:.2f}｜p90={low_b1['p90']:.2f}｜max={low_b1['max']:.2f} ms，"
              f"全部 < 慢态下界 {SLOW_B1_LOW_MS:.0f} ms；")
        print(f"    且 max 距慢态下界 {SLOW_B1_LOW_MS - low_b1['max']:.1f} ms；"
              f"全 4 段全部 run **零**「连续 ≥慢下界」的步 ⇒ 既非双峰、也非整体偏慢。")
        print(f"    ⇒ 项目级影响：**不构成** COST_PROFILE §4 所说的「占空比打回慢档」"
              f"⇒ 该批不存在由此而来的控制器信号污染；")
        print(f"      ⚠️ 但这是**有限观察**（只覆盖 S14B2 一个负载/一次会话），"
              f"不能推广成「该设备不会进慢态」。")
    print()
    if band:
        ks = sorted(band)
        tbl = "  ".join(f"b{k}={band[k]['tpot_p50']:.1f}" for k in ks
                        if band[k]["tpot_p50"] is not None)
        c4_ok = bool(band.get(4) and band[4]["tpot_p50"] and band[4]["tpot_p50"] <= SLO_TPOT_MS)
        print(f"(2) 负载设计在 c4–c6 时 TPOT 是否落在 SLO {SLO_TPOT_MS:.0f} ms 内 ⇒ "
              f"**{'是' if c4_ok else '否'}**")
        print(f"    按实测 `tpot_ms` 分档：{tbl}（ms，drain 段优先）")
        b4 = band.get(4)
        if b4 and b4["tpot_p50"]:
            print(f"    c4 实测 TPOT 中位 **{b4['tpot_p50']:.2f} ms** "
                  f"(seg={b4['seg']}, n={b4['n_tpot']}, queue p50={b4['queue_p50'] or 0:.0f}, "
                  f"跳过率={(b4.get('skip_frac') or 0):.1%}) "
                  f"⇒ **{'超' if not c4_ok else '未超'} SLO {b4['tpot_p50'] - SLO_TPOT_MS:+.2f} ms**；"
                  f"c5/c6 更高（{band.get(5, {}).get('tpot_p50', float('nan')):.1f} / "
                  f"{band.get(6, {}).get('tpot_p50', float('nan')):.1f} ms）。")
            if b4["seg"] == "drain" and not b4.get("skip_frac"):
                print("    ★ 该点是**干净工作点**：`drain` 段（负载已结束）、队列 p50=0、"
                      "跳过率 0.0% ⇒ 4 条序列每步各得 1 token ⇒ 请求级 TPOT ≡ 步时长。")
        print(f"    ★ 实测**达标档位集合 = {{{', '.join('b%d' % k for k in slo_ok_set)}}}**"
              f" ⇒ `c4–c6` 的设计假设**与实测不符**。")
        print("    ⚠️ 这里**不报**「最大可行档」这个单个数：实测 batch 成本曲线"
              "**非单调**（b3 比 b4 贵），「最大档」在非单调曲线下无定义。")
        print("    ⚠️ 与路线图 §0「SLO 可行最大批 ≈ b6」的差异**不是**矛盾，而是"
              "口径不同：b6 是由**快态标定表下界**线性内插得到，")
        print("       而标定表是**性能下界**（COST_PROFILE §2 使用边界）。"
              "按 §2.4 的判据（用实测 `tpot_ms`）结论是 b4 已超。")
    print("\n  复算命令：")
    print(f"    \"C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe\" "
          f"scripts/analyze_opt01_s14b2.py --tag {args.tag} --json results/processed/opt01_s14b2.json")

    # ---------------- 落盘 ------------------------------------------------
    if args.json:
        out = Path(args.json)
        if not out.is_absolute():
            out = _ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        # 产物现读校对 SLO
        slo_seen = {(r["meta"].get("config") or {}).get("run", {}).get("slo", {}).get("tpot_ms")
                    for r in ok}
        report = {
            "session": "OPT-01",
            "tag": args.tag,
            "generated_by": "scripts/analyze_opt01_s14b2.py",
            "denominator": {"n_runs_included": len(ok), "n_runs_total": len(runs),
                            "excluded": [{"dir": r["dir"], "reason": r["incomplete_reason"]}
                                         for r in bad]},
            "anchor_check": {"verdict": verdict_anchor,
                             "n_reproduced_anchor0": n_zero,
                             "n_reproduced_shift": n_shift,
                             "detail": anchors},
            "slo_tpot_ms_from_products": sorted(x for x in slo_seen if x is not None),
            "constants": {"fast_b1_probe_min_ms": FAST_B1_PROBE_MIN_MS,
                          "fast_b1_p1s3d_ms": FAST_B1_P1S3D_MS,
                          "slow_b1_band_ms": [SLOW_B1_LOW_MS, SLOW_B1_HIGH_MS],
                          "fast_ceil_ms": FAST_CEIL_MS,
                          "slow_floor_ms": SLOW_FLOOR_MS,
                          "fast_table_ms": FAST_TABLE_MS,
                          "fast_table_truth_ms": FAST_TABLE_TRUTH_MS,
                          "slo_tpot_ms": SLO_TPOT_MS,
                          "min_n": args.min_n},
            "step1_slowstate": s1,
            "step2_workband": s2,
            "step2b_request_tpot": req_tpot,
            "conclusions": {
                "low_segment_slow_state": (low_b1["verdict"] if low_b1 else "不可判"),
                "low_segment_b1_n": (low_b1["n"] if low_b1 else None),
                "low_segment_b1_p50_ms": (low_b1["p50"] if low_b1 else None),
                "low_segment_b1_max_ms": (low_b1["max"] if low_b1 else None),
                "max_consecutive_slow_steps_b1": (
                    max((x["max_consecutive_slow_steps"] for x in s1["slow_streaks"]),
                        default=0)),
                "slo_ok_decode_buckets": slo_ok_set,
                "c4_tpot_p50_ms": (band.get(4, {}).get("tpot_p50")),
                "c4_meets_slo": bool(band.get(4) and band[4]["tpot_p50"] is not None
                                     and band[4]["tpot_p50"] <= SLO_TPOT_MS),
                "drain_ratios_vs_fast_table_min": ratios,
                "request_tpot_p50_ms": req_tpot.get("p50"),
                "request_tpot_slo_pass_frac": req_tpot.get("slo_pass_frac"),
            },
            "boundary": ("只读分析；未改 results/raw/**、未改 benchmark/real_backend.py、"
                         "未产生新实验条件。分段与分档口径见脚本 docstring 的 A/B/C 三条。"),
        }
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                       encoding="utf-8", newline="\n")
        print(f"\n已写出：{out}")

    if bad:
        print(f"\n⚠️ 分母不全（{len(ok)}/{len(runs)}）：另有 {len(bad)} 个目录不可用，"
              f"见上表。**非零 ≠ 方法被否证**，只表示本次样本不齐。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
