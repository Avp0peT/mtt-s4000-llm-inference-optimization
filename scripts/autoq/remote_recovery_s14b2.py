#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S14B2 恢复段只读分析（**复用项目自己的 stage_report**，不另写解析器）。

用法（远端）：
    AQ_TAG=S14B2 AQ_ROOT=/root/ascend_probe_d1 \
      python3 /root/autoq/remote_recovery_s14b2.py

纪律：
* 只读；跳过 .tar 暂存包与缺 meta.json 的目录。
* 恢复判据**显式给定**（RecoveryRule: metric=queue_depth, threshold, hold_ms, until_ms）。
* ``not_recovered`` 与 ``censored`` 分开报，**绝不填 0**。
* 只覆盖 ``ramp``（过载区）；不与 S6（v7 / c2_m4）混算。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("AQ_ROOT", "/root/ascend_probe_d1"))
TAG = os.environ.get("AQ_TAG", "S14B2")
RAW = ROOT / "results" / "raw"

# 让项目自身的模块可被 import（远端 repo 根 = ROOT）
sys.path.insert(0, str(ROOT / "src"))

try:
    from ascend_opt.stage_report import (  # type: ignore
        RecoveryRule,
        StageSpec,
        build_stage_specs,
        build_stage_report,
    )
    from ascend_opt.metrics import RequestMetrics, SLO  # type: ignore
except Exception as exc:  # noqa: BLE001
    print(f"FATAL: 无法 import ascend_opt: {exc!r}", file=sys.stderr)
    sys.exit(2)

_STAGING = (".tar", ".tar.gz", ".tgz")


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _dirs() -> list[pathlib.Path]:
    got = []
    for p in sorted(RAW.glob(f"{TAG}-*")):
        if not p.is_dir():
            continue
        if p.name.lower().endswith(_STAGING):
            continue
        if not (p / "meta.json").is_file():
            continue
        got.append(p)
    return got


def analyze(d: pathlib.Path) -> dict:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    fp = meta["context"]["extra"]["workload_fingerprint"]
    segs = fp["arrival_segments"]
    stages = build_stage_specs(segs)

    reqs = _load_jsonl(d / "requests.jsonl")
    telem = _load_jsonl(d / "telemetry.jsonl")

    # ★ 必须现场**派生** slo_satisfied：requests.jsonl 由 RequestMetrics.describe() 写出，
    #   而 describe() **不含** slo_satisfied 字段。stage_report 的判据是
    #   ``r.get("slo_satisfied") is True``，字段缺失 ⇒ 全 False ⇒ 逐段达标率静默变 0
    #   （与 summary.n_slo_satisfied 矛盾：0 vs 18）。
    #   这里复用项目自身的 SLO.satisfied_by，**不另写判据**。
    slo_cfg = ((meta.get("config") or {}).get("run") or {}).get("slo") or {}
    slo = SLO(ttft_ms=slo_cfg.get("ttft_ms"), tpot_ms=slo_cfg.get("tpot_ms"),
              e2e_ms=slo_cfg.get("e2e_ms"))
    fields = set(getattr(RequestMetrics, "__dataclass_fields__", {}))
    derived_n = 0

    def _derive(row: dict) -> bool:
        nonlocal derived_n
        rm = RequestMetrics(**{k: v for k, v in row.items() if k in fields})
        ok = bool(slo.satisfied_by(rm))
        if ok:
            derived_n += 1
        return ok

    reqs = [dict(r, slo_satisfied=_derive(r)) for r in reqs]
    declared = ((meta.get("summary") or {}).get("n_slo_satisfied"))
    if declared is not None and declared != derived_n:
        print(f"[!] {d.name}: 派生达标 {derived_n} ≠ summary {declared}"
              f"（口径不一致 ⇒ 不要采信逐段达标率）", file=sys.stderr)

    # 恢复段起点 = 负载回落时刻；观察窗右端 = 恢复段结束。
    rec = [s for s in segs if s.get("label") == "recover"]
    fail_seg = [s for s in segs if s.get("label") == "high"]
    from_ms = rec[0]["start_ms"] if rec else None
    until_ms = rec[0]["end_ms"] if rec else None
    thr = float(os.environ.get("AQ_REC_THR", "1"))
    hold = float(os.environ.get("AQ_REC_HOLD", "5000"))
    rule = RecoveryRule(metric="queue_depth", threshold=thr, hold_ms=hold,
                        from_ms=from_ms, until_ms=until_ms)

    rep = build_stage_report(
        requests=reqs, stages=stages, telemetry=telem,
        recovery_rule=rule, counts_toward_drain=True,
    )
    rv = rep.recovery
    # 控制器侧"同一含义"的量：收缩是否回到配置初值（配置属性，非观测量）
    ct = meta.get("controller_trace") or []
    # ⚠️ chunk_size 越过 init **不构成违规**：控制器是双向的，
    #    案例 B（TTFT 超目标且 TPOT 有余量）= "为 prefill 加码"，**无 init 上限**；
    #    案例 C（双目标都达标且此前收缩过）= "恢复到初值"，**有 init 上限**（controller.py:1079）。
    #    所以这里把两类动作分开统计，而不是简单比较 last==init。
    n_grow_prefill = 0   # case B：TTFT 超目标 ⇒ 放大（可越过 init）
    n_recover = 0        # case C：恢复（上限 = init）
    n_shrink = 0         # case A/M：收缩
    for s in ct:
        if not isinstance(s, dict) or not s.get("changed"):
            continue
        rsn = str(s.get("reason") or "")
        cur = s.get("state") or {}
        nxt = None
        if "TTFT" in rsn and ("放大" in rsn or "余" in rsn):
            n_grow_prefill += 1
        elif "恢复" in rsn:
            n_recover += 1
        elif "decode P90" in rsn or "收缩" in rsn:
            n_shrink += 1
    chunk_vals, token_vals, init_chunk, init_tokens = [], [], None, None
    for s in ct:
        st = s.get("state") if isinstance(s, dict) else None
        if not isinstance(st, dict):
            continue
        if isinstance(st.get("chunk_size"), int):
            chunk_vals.append(st["chunk_size"])
        if isinstance(st.get("max_batched_tokens"), int):
            token_vals.append(st["max_batched_tokens"])
        if init_chunk is None and st.get("init_chunk_size") is not None:
            init_chunk = st["init_chunk_size"]
        if init_tokens is None and st.get("init_max_batched_tokens") is not None:
            init_tokens = st["init_max_batched_tokens"]
    return {
        "dir": d.name,
        "label": meta.get("data_label"),
        "cell": meta["config"].get("config_stem"),
        "criterion": meta["config"].get("controller", {}).get("signal_criterion"),
        "from_ms": from_ms,
        "until_ms": until_ms,
        "rec_status": rv.status,
        "rec_at_ms": rv.at_ms,
        "rec_duration_ms": rv.duration_ms,
        "rec_rule": rv.rule,
        "init_chunk": init_chunk,
        "init_tokens": init_tokens,
        "min_chunk": min(chunk_vals) if chunk_vals else None,
        "max_chunk": max(chunk_vals) if chunk_vals else None,
        "last_chunk": chunk_vals[-1] if chunk_vals else None,
        "min_tokens": min(token_vals) if token_vals else None,
        "max_tokens": max(token_vals) if token_vals else None,
        "n_changed": sum(1 for s in ct if isinstance(s, dict) and s.get("changed")),
        "n_grow_prefill": n_grow_prefill,
        "n_recover": n_recover,
        "n_shrink": n_shrink,
        # 收缩过又恢复到初值？(case C 的存在性证据)
        "had_case_C": n_recover > 0,
        # cohorts / windows 都是 list（不是 dict）
        "cohorts": [{"label": c.label, "n_submitted": c.n_submitted,
                     "n_success": c.n_success, "n_rejected": c.n_rejected,
                     "n_unfinished": c.n_unfinished,
                     "n_slo_satisfied": c.n_slo_satisfied,
                     "n_cross_stage": c.n_cross_stage,
                     "insufficient_samples": c.insufficient_samples}
                    for c in rep.cohorts],
        "windows": [{"label": w.label, "start_ms": w.start_ms, "end_ms": w.end_ms,
                     "n_completed": w.n_completed, "n_admitted": w.n_admitted,
                     "sat_rate": w.slo_satisfaction_rate,
                     "goodput_tok_s": w.slo_goodput_tok_s,
                     "queue_max": w.queue_depth_max, "queue_p50": w.queue_depth_p50,
                     "n_control_actions": w.n_control_actions,
                     "chunk_min": w.chunk_size_min, "chunk_max": w.chunk_size_max}
                    for w in rep.windows],
        "reconciliation": rep.reconciliation,
        "notes": list(rep.notes),
        # 派生达标 vs summary 声明是否一致（False ⇒ 该 run 的逐段达标率不可信）
        "sat_reconciled": (declared is None) or (declared == derived_n),
        "n_sat_derived": derived_n,
        "n_sat_declared": declared,
    }


def main() -> int:
    ds = _dirs()
    print(f"=== {TAG}: 可用产物 {len(ds)} 个（含 meta.json）===")
    rows = []
    for d in ds:
        try:
            rows.append(analyze(d))
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {d.name}: {exc!r}")
    print(f"\n{'dir':46s} {'label':9s} {'rec_status':13s} {'rec_ms':>9s} "
          f"{'chunk[min,max]':>15s} {'tok[min,max]':>15s} {'grw':>4s} {'rec':>4s} {'shr':>4s}")
    for r in rows:
        dur = "-" if r["rec_duration_ms"] is None else f"{r['rec_duration_ms']:.0f}"
        ck = f"[{r['min_chunk']},{r['max_chunk']}]"
        tk = f"[{r['min_tokens']},{r['max_tokens']}]"
        print(f"{r['dir']:46s} {str(r['label']):9s} {r['rec_status']:13s} {dur:>9s} "
              f"{ck:>15s} {tk:>15s} {r['n_grow_prefill']:>4d} {r['n_recover']:>4d} {r['n_shrink']:>4d}")
    # 分 cell 汇总
    print("\n=== 按 cell 汇总（恢复状态计数；缺 = 不出现，不填 0）===")
    by: dict[str, dict[str, int]] = {}
    for r in rows:
        b = by.setdefault(str(r["cell"]), {})
        b[r["rec_status"]] = b.get(r["rec_status"], 0) + 1
    for cell, b in sorted(by.items()):
        print(f"{cell:16s} {b}")
    # 控制器"恢复"语义分层：case B（加码，可越 init）vs case C（恢复到 init，封顶）
    print("\n=== 控制器动作分层（case B 加码 / case C 恢复 / 收缩）===")
    for cell in sorted(by):
        sub = [r for r in rows if str(r["cell"]) == cell]
        g = sum(r["n_grow_prefill"] for r in sub)
        c = sum(r["n_recover"] for r in sub)
        s = sum(r["n_shrink"] for r in sub)
        inits = {r["init_chunk"] for r in sub}
        over = sum(1 for r in sub
                   if r["max_chunk"] is not None and r["init_chunk"] is not None
                   and r["max_chunk"] > r["init_chunk"])
        print(f"{cell:16s} init_chunk={sorted(x for x in inits if x is not None)} "
              f"加码={g} 恢复={c} 收缩={s} | 峰值>init 的 run 数={over}/{len(sub)}")
    # controller_effect 的降级/饿死计数（缺 = None，不填 0）
    print("\n=== controller_effect（缺 = None，不填 0）===")
    for r in rows:
        m = json.loads((RAW / r["dir"] / "meta.json").read_text(encoding="utf-8"))
        eff = (m.get("controller_effect") or {}).get("arm_state")
        if eff is None:
            print(f"{r['dir']:46s} arm_state=None")
        else:
            print(f"{r['dir']:46s} fallback={eff.get('n_fallback')} "
                  f"degraded={eff.get('n_degraded')} starved={eff.get('n_prefill_starved')} "
                  f"idle={eff.get('n_idle')}")
    # 分窗证据（区域判定 + chunk 实际区间）
    print("\n=== 分窗证据（wall-clock 窗口）===")
    for r in rows:
        print(f"-- {r['dir']} ({r['cell']}) rec={r['rec_status']}")
        for w in r["windows"]:
            sr = w["sat_rate"]
            sr_s = "-" if sr is None else f"{sr:.3f}"
            gp = w["goodput_tok_s"]
            gp_s = "-" if gp is None else f"{gp:.3f}"
            print(f"   {w['label']:9s} [{w['start_ms']:.0f},{w['end_ms']:.0f}]ms "
                  f"adm={w['n_admitted']} done={w['n_completed']} sat={sr_s} "
                  f"goodput={gp_s} qmax={w['queue_max']} q50={w['queue_p50']} "
                  f"act={w['n_control_actions']} chunk=[{w['chunk_min']},{w['chunk_max']}]")
        if r["reconciliation"]:
            print(f"   reconciliation.ok={r['reconciliation'].get('ok')}")
    # 对账失败必须显形（不静默）
    bad = [r["dir"] for r in rows
           if r["reconciliation"] and r["reconciliation"].get("ok") is False]
    if bad:
        print(f"\n[!] 对账失败 {len(bad)} 个：{bad}")
    # 恢复"测不出"必须显形
    nc = sum(1 for r in rows if r["rec_status"] != "recovered")
    if nc:
        print(f"[!] 非 recovered 的 run：{nc} 个（censored/not_recovered ≠ 0 ms）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
