#!/usr/bin/env python3
"""按**到达阶段**与**墙钟窗口**两套口径，报告一个 run 的动态负载表现。

为什么需要它
------------
研究问题的自变量是负载**轨迹**（低压 → 高压 → 恢复）。整 run 一个汇总值回答不了：
低压段与高压段的读数混进同一个中位数；恢复段有没有回落、多久回落看不见；
高压段末尾到达的请求在恢复段完成——算哪一段？本脚本把这两套口径**分开报**。

**本脚本只读产物、不跑任何东西、不写 results/**。产物由 :mod:`ascend_opt.stage_report`
（纯逻辑）分析；这里只负责把产物读进来、把结论打出来。

三条口径（与 `stage_report` 模块一致，**预先定义**而非事后解释）
--------------------------------------------------------------
1. **达标判定**：直接调用权威的 :meth:`SLO.satisfied_by`（成功 + TTFT/TPOT/E2E 同时满足），
   ⚠️ **不在本脚本里另写一套阈值判断**——两处各写一套的下场是粗的那套放行。
   请求的 SLO 取自**该 run 自己的** `meta.config.run.slo`，不是当前默认值。
2. **分母**：达标吞吐统一用测量分母（分窗口 = 窗口墙钟长度）。
   ⚠️ 用"最后一个完成时刻"当分母会把排空尾段从分母里抹掉，吞吐虚高。
   ⚠️ 口径来源：优先用产物里的 **``summary.wall_ms``**（正式测量窗，权威值）；
   只有它缺失时才退回 **telemetry 的 ``t_ms`` 跨度**（``max − min``）作代理。
   实测代理与权威值差 **~0.1%**（P1S3D `adaptive_c1`：跨度 66962.554 vs
   ``wall_ms`` 67029.045 ⇒ 代理**略小**），量级上可用但**不逐位相等**。
   ⇒ 一律优先读 ``wall_ms``；退回代理时会在报告里标明。
3. **恢复**：阈值与持续时间由命令行**显式给定**（``--recovery-threshold`` /
   ``--recovery-hold-ms``）；不给就不判。没观察到 ⇒ 记未恢复/删失，**不填 0**。

用法::

    python scripts/report_stages.py --run-dir results/raw/<run_id>
    python scripts/report_stages.py --run-dir <dir> --json out.json
    python scripts/report_stages.py --run-dir <dir> \\
        --recovery-threshold 1 --recovery-hold-ms 1000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.metrics import SLO, RequestMetrics  # noqa: E402
from ascend_opt.stage_report import (  # noqa: E402
    DRAIN_LABEL,
    RecoveryRule,
    StageSpec,
    build_stage_report,
)

_FIELDS = {f.name for f in dataclasses.fields(RequestMetrics)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load(run_dir: Path) -> tuple[dict[str, Any], list[dict], list[dict]]:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    reqs = _read_jsonl(run_dir / "requests.jsonl")
    tel_path = run_dir / "telemetry.jsonl"
    tel = _read_jsonl(tel_path) if tel_path.exists() else []
    return meta, reqs, tel


def _slo_of(meta: dict[str, Any]) -> SLO:
    """用**该 run 自己的** SLO 判达标。缺了就不判（返回空 SLO ⇒ 全部达标）。

    ⚠️ 绝不能拿"当前配置里的阈值"去重算旧 run：阈值改过之后，
    旧批的达标率与新批不是同一个判据，必须按该 run 落盘的阈值算。
    """
    raw = ((meta.get("config") or {}).get("run") or {}).get("slo") or {}
    return SLO(
        ttft_ms=raw.get("ttft_ms"),
        tpot_ms=raw.get("tpot_ms"),
        e2e_ms=raw.get("e2e_ms"),
    )


def _requests_with_slo(reqs: list[dict], slo: SLO) -> list[dict]:
    """逐请求构造 :class:`RequestMetrics` 并调用权威判据，把结果并回一行。"""
    out: list[dict] = []
    for row in reqs:
        kwargs = {k: v for k, v in row.items() if k in _FIELDS}
        try:
            m = RequestMetrics(**kwargs)
            satisfied = slo.satisfied_by(m)
        except Exception:            # 产物缺字段就退回"不算达标"，绝不默认达标
            satisfied = False
        out.append({**row, "slo_satisfied": bool(satisfied)})
    return out


def _stages(meta: dict[str, Any], reqs: list[dict]) -> tuple[list[StageSpec], str]:
    """取阶段边界。**优先用产物里落盘的 ``arrival_segments``**（那是实际负载），
    没有就退回"整 run 一段"并明确声明——不凭空造阶段。"""
    fp = ((meta.get("context") or {}).get("extra") or {}).get("workload_fingerprint") or {}
    segs = fp.get("arrival_segments")
    if segs:
        return [StageSpec(label=str(s["label"]), start_ms=float(s["start_ms"]),
                          end_ms=float(s["end_ms"]),
                          nominal_qps=s.get("nominal_qps"),
                          nominal_count=s.get("nominal_count"),
                          realized_count=s.get("realized_count"))
                for s in segs], "来自产物 arrival_segments"
    span = max((r.get("arrival_ms") or 0.0) for r in reqs) if reqs else 0.0
    end = max(span, max((r.get("done_ms") or 0.0) for r in reqs) if reqs else 0.0)
    return ([StageSpec(label="all", start_ms=0.0, end_ms=end + 1.0)],
            "⚠️ 该 run 没有 arrival_segments（非 piecewise 负载）⇒ 退化成整 run 一段；"
            "分阶段结论**不适用**")


def _actions_with_time(meta: dict[str, Any], tel: list[dict]) -> list[dict]:
    """控制动作只有 ``step``，没有时刻 ⇒ 用 telemetry 的 step→t_ms 映射补上。

    ⚠️ 不补就会**静默地全落到窗口之外**（``n_control_actions`` 恒为 0），
    而 0 看起来像"这段时间控制器没动"——那是另一个结论。
    """
    step_to_ms = {t.get("step"): t.get("t_ms") for t in tel if t.get("step") is not None}
    out: list[dict] = []
    for a in (meta.get("controller_trace") or []):
        t_ms = a.get("t_ms")
        if t_ms is None and a.get("step") is not None:
            t_ms = step_to_ms.get(a["step"])
        out.append({**a, "t_ms": t_ms})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="动态负载的分阶段报告")
    ap.add_argument("--run-dir", required=True, help="run 目录（含 meta.json）")
    ap.add_argument("--json", default=None, help="把报告同时写成 JSON")
    ap.add_argument("--min-n", type=int, default=20,
                    help="分位数所需的最小样本量（低于则不给 P95/P99）")
    ap.add_argument("--recovery-threshold", type=float, default=None)
    ap.add_argument("--recovery-hold-ms", type=float, default=1000.0)
    ap.add_argument("--recovery-from-ms", type=float, default=None)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not (run_dir / "meta.json").exists():
        print(f"✗ {run_dir} 里没有 meta.json", file=sys.stderr)
        return 2

    meta, reqs, tel = _load(run_dir)
    slo = _slo_of(meta)
    requests = _requests_with_slo(reqs, slo)
    stages, stage_src = _stages(meta, requests)

    # 分母：优先用产物里落盘的正式测量窗；缺失才退回 telemetry 跨度（并标明）
    window_ms = (meta.get("summary") or {}).get("wall_ms")
    window_src = "summary.wall_ms（正式测量窗）"
    if window_ms is None and tel:
        ts = [t["t_ms"] for t in tel if t.get("t_ms") is not None]
        if ts:
            window_ms = max(ts) - min(ts)
            window_src = "⚠️ 代理：telemetry 的 t_ms 跨度（summary.wall_ms 缺失）"

    rule = None
    if args.recovery_threshold is not None:
        rule = RecoveryRule(metric="queue_depth", threshold=args.recovery_threshold,
                            hold_ms=args.recovery_hold_ms, from_ms=args.recovery_from_ms)

    rep = build_stage_report(
        requests=requests, stages=stages, telemetry=tel,
        actions=_actions_with_time(meta, tel),
        measured_window_ms=window_ms, recovery_rule=rule,
        min_n_for_percentiles=args.min_n,
    )

    print(f"=== {run_dir.name} ===")
    print(f"数据标签 : {meta.get('data_label')}")
    print(f"SLO      : ttft_ms={slo.ttft_ms}  tpot_ms={slo.tpot_ms}  e2e_ms={slo.e2e_ms}"
          "   （取自该 run 落盘的配置，不是当前默认值）")
    print(f"阶段来源 : {stage_src}")
    print(f"测量窗口 : {window_ms if window_ms is None else round(window_ms, 3)} ms"
          f"   （{window_src}）")

    print("\n--- cohort：按**到达阶段**看这批请求后来表现如何 ---")
    hdr = f"{'段':<10}{'提交':>6}{'成功':>6}{'失败':>6}{'拒':>5}{'未完':>6}{'达标':>6}{'跨段':>6}{'TTFT p50':>11}{'TTFT p95':>11}{'n':>6}"
    print(hdr)
    for c in rep.cohorts:
        def f(v):
            return "  n/a" if v is None else f"{v:8.1f}"
        print(f"{c.label:<10}{c.n_submitted:>6}{c.n_success:>6}{c.n_failed:>6}"
              f"{c.n_rejected:>5}{c.n_unfinished:>6}{c.n_slo_satisfied:>6}"
              f"{c.n_cross_stage:>6}{f(c.ttft_ms_p50):>11}{f(c.ttft_ms_p95):>11}"
              f"{c.n_submitted:>6}")
        if c.insufficient_samples:
            print(f"{'':<10}  ⚠️ 样本不足（n={c.n_submitted} < {args.min_n}）"
                  "⇒ 不给 P95/P99：十几个样本的 P99 就是最大值")

    print("\n--- window：按**墙钟**看这段时间系统在处理什么 ---")
    hdr2 = (f"{'段':<10}{'窗口ms':>9}{'受理':>6}{'完成':>6}{'失败':>6}{'达标':>6}"
            f"{'达标率':>9}{'tok/s':>10}{'队深max':>9}{'活跃max':>9}{'动作':>6}")
    print(hdr2)
    for w in rep.windows:
        def g(v, nd=2):
            return "  n/a" if v is None else f"{v:.{nd}f}"
        print(f"{w.label:<10}{w.duration_ms:>9.0f}{w.n_admitted:>6}{w.n_completed:>6}"
              f"{w.n_failed:>6}{w.n_slo_satisfied:>6}"
              f"{g(w.slo_satisfaction_rate, 3):>9}{g(w.slo_goodput_tok_s):>10}"
              f"{str(w.queue_depth_max):>9}{str(w.active_max):>9}{w.n_control_actions:>6}")

    r = rep.recovery
    print(f"\n--- 恢复 ---\n状态: {r.status}"
          + ("" if r.at_ms is None else f"  时刻: {r.at_ms:.0f} ms  用时: {r.duration_ms:.0f} ms"))
    print(f"判据: {r.rule}")
    if r.status in ("not_recovered", "censored") and r.duration_ms is None:
        print("⚠️ 未观察到恢复 ⇒ 用时记 None（**不是 0**）："
              "「0 ms 恢复」是好得不可能的读数，比「未恢复」更容易误导。")

    rec = rep.reconciliation
    print(f"\n--- 对账 ---\n{'✅ 通过' if rec['ok'] else '❌ 不通过'}"
          f"  请求 {rec['n_requests']} / cohort 认领 {rec['n_cohort_submitted']}"
          f" / 窗口完成 {rec['n_completed_in_windows']}（成功 {rec['n_success']}）")
    for p in rec["problems"]:
        print(f"  ✗ {p}")
    for n in rep.notes:
        print(f"  ⚠️ {n}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(rep.describe(), ensure_ascii=False, indent=2),
            encoding="utf-8", newline="\n",
        )
        print(f"\n已写出 JSON: {args.json}")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
