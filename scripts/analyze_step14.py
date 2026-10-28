#!/usr/bin/env python3
"""第 14 步的四臂对比分析（**只读产物，不改任何东西**）。

验收要求同时报：达标吞吐 · 达标率 · TTFT · TPOT · stall · **恢复时间** ·
**控制开销** · 失败/拒绝/未完成率。本脚本逐条落地，并对**每条**注明它来自哪个字段。

三条纪律（写进代码，不靠记性）
------------------------------
1. **报全部重复，不挑最好一次**：每组都给 `all` + 中位 + 极差，不给"最好那次"。
2. **配对**：矩阵按 pass 交错（`-r1..-rN`），所以按 **pass 序号配对**比较各臂，
   而不是拿两组的独立分布硬比（同会话内的漂移被配对消掉）。
3. **"没被用上" ≠ "用了没用"**：`cost` 臂超出模型范围会回退 ⇒ 必须报**回退率**，
   且回退率高时**不得**据"与 threshold 接近"下"成本模型没用"的结论。

用法::

    python scripts/analyze_step14.py --tag S14B1 --json results/processed/step14_S14B1.json
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
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

#: 从 run_id 里拆出 cell 与 arm：`<TAG>-<workload>-<cell>_<arm>[-rN]`
#: ⚠️ 2026-09-27：解析收归唯一实现 `ascend_opt.run_ids`（此前本文件与
#: `diagnose_scheduling.py` 各写一份，D1① 闸门又写第三份 ⇒ 三份分叉，详见该模块 docstring）。
from ascend_opt.run_ids import parse_run_id  # noqa: E402

#: 要报的 summary 字段 → 报表列名。**只列有据可查的**，缺的写"未测量"。
SUMMARY_METRICS: tuple[tuple[str, str], ...] = (
    ("slo_goodput_tok_s", "达标吞吐 tok/s"),
    ("slo_satisfaction_rate", "达标率"),
    ("throughput_tok_s", "吞吐 tok/s"),
    ("failure_rate", "失败率"),
    ("rejection_rate", "拒绝率"),
    ("unfinished_rate", "未完成率"),
    ("accounting_error_p99_ms", "账目误差 P99 ms"),
    ("wall_ms", "单次墙钟 ms"),
)
#: 分位字段（summary 里是 {p50,p90,p95,p99} 子字典）
QUANTILE_METRICS: tuple[tuple[str, str], ...] = (
    ("ttft", "TTFT"), ("tpot", "TPOT"), ("stall", "stall"),
)


def load_runs(tag: str, skipped: list[tuple[str, str]] | None = None
              ) -> list[dict[str, Any]]:
    """加载某标签下全部可解析的 run。

    ⚠️ **不完整产物必须"响亮跳过"，不许静默丢弃，更不许整脚本崩掉**
    （2026-09-28 §46 踩过：一个 32 768 B 的**截断** `meta.json` 让整个报表
    `JSONDecodeError` 中断，已经加载好的 9 个 run 全部白跑）。
    与登记侧 §37 的处理保持同一口径：**列出坏文件 → 跳过 → 由调用方返回非零**。

    坏文件的三种形态要分开报，因为**处置不同**：
    - `ABSENT`    文件不存在（多半是拉取还没轮到）
    - `EMPTY`     0 字节（拉取建了文件但没落字节）
    - `UNPARSEABLE` 有字节但不是合法 JSON（**截断**，多为 32 KiB 整数倍）
      ⇒ 这一种最危险：**按尺寸门槛判定会把它当"完好"**，但它的数字一个都不能用。

    :param skipped: 若传入列表，把跳过的 `(目录名, 原因)` 追加进去。
    """
    out: list[dict[str, Any]] = []
    for d in sorted(glob.glob(str(_ROOT / "results" / "raw" / f"{tag}-*"))):
        meta = Path(d) / "meta.json"
        name = Path(d).name
        if not meta.exists():
            if skipped is not None:
                skipped.append((name, "ABSENT"))
            continue
        raw = meta.read_text(encoding="utf-8", errors="replace")
        if not raw.strip():
            if skipped is not None:
                skipped.append((name, "EMPTY"))
            print(f"  ⏭️  {name}: meta.json 为 0 字节，跳过（未测量）", file=sys.stderr)
            continue
        try:
            m = json.loads(raw)
        except json.JSONDecodeError as exc:
            if skipped is not None:
                skipped.append((name, f"UNPARSEABLE({meta.stat().st_size}B)"))
            print(f"  ⏭️  {name}: meta.json 截断/非法 JSON（{meta.stat().st_size} B，"
                  f"{exc.msg} @ char {exc.pos}），跳过——**其数字不可用**",
                  file=sys.stderr)
            continue
        rid = (m.get("summary") or {}).get("run_id") or Path(d).name
        parsed = parse_run_id(rid)
        if parsed is None:
            print(f"  ⚠️ run_id 不合模式，跳过：{rid}", file=sys.stderr)
            continue
        tel_path = Path(d) / "telemetry.jsonl"
        # ⚠️ `telemetry.jsonl` **逐行**解析，且**末行被截断是常态**（JSONL 按行落地，
        # 断在最后一行 ⇒ 前面所有行都是好的）。所以这里**不能**整体 json.loads，
        # 更不能因一行坏就丢掉整个 run：**能读多少读多少，但要把坏行数显式带出去**
        # （2026-09-28 §46：此前无守卫，一行截断就让整个报表崩，9 个 run 白跑）。
        tel: list[dict[str, Any]] = []
        tel_bad = 0
        if tel_path.exists():
            for ln, line in enumerate(tel_path.read_text(encoding="utf-8",
                                                        errors="replace").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    tel.append(json.loads(line))
                except json.JSONDecodeError:
                    tel_bad += 1
            if tel_bad:
                print(f"  ⚠️ {Path(d).name}: telemetry.jsonl 有 {tel_bad} 行不可解析"
                      f"（已读入 {len(tel)} 行）——缺的只是**该 run 的尾部**，"
                      f"分段统计仍可用，但**必须报出这个缺口**", file=sys.stderr)
        out.append({
            "dir": Path(d).name, "run_id": rid,
            "cell": parsed.cell, "arm": parsed.arm,
            "pass": int(parsed.pass_index or 0),
            "label": m.get("data_label"),
            "commit": (m.get("reproducibility") or {}).get("git_commit"),
            "dirty": (m.get("reproducibility") or {}).get("git_dirty"),
            # ⚠️ **实际生效的判据**：第 09 步 §⑥ 冻结的是 `decode_p90`，但第 14 步的配置
            # 漏了这个字段 ⇒ 加载成默认 `tpot_p90`。所以必须从产物里读、并与冻结值比对，
            # 不能凭"我们冻结过"就当它生效了。
            "criterion": ((m.get("config") or {}).get("controller") or {}).get("signal_criterion"),
            "kind_used": ((m.get("config") or {}).get("controller") or {}).get("kind"),
            "ctrl_enabled": ((m.get("config") or {}).get("controller") or {}).get("enabled"),
            "shadow_used": ((m.get("config") or {}).get("controller") or {}).get("shadow"),
            "summary": m.get("summary") or {},
            "effect": m.get("controller_effect") or {},
            "trace": m.get("controller_trace") or [],
            "telemetry": tel,
        })
    return out


def _vals(runs: list[dict[str, Any]], key: str) -> list[float]:
    out = []
    for r in runs:
        v = r["summary"].get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _q(runs: list[dict[str, Any]], grp: str, q: str) -> list[float]:
    out = []
    for r in runs:
        v = (r["summary"].get(grp) or {}).get(q)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def fallback_rate(runs: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    """`cost` / `slo_weave` 臂的**回退率**：回退次数 / 决策次数。

    ⚠️ 这是"**没被用上**"的度量，不是"用了没用"。必须与性能指标**并排**出现。
    """
    key = "cost_policy" if kind == "cost" else "slo_weave"
    tot = fb = dec = 0
    reasons: dict[str, int] = {}
    for r in runs:
        for e in r["trace"]:
            d = e.get(key)
            if not d:
                continue
            tot += 1
            if d.get("fallback_reason"):
                fb += 1
                reasons[str(d["fallback_reason"])[:60]] = \
                    reasons.get(str(d["fallback_reason"])[:60], 0) + 1
            if d.get("decode_only"):
                dec += 1
    return {"decisions": tot, "fallbacks": fb,
            "fallback_rate": (fb / tot if tot else None),
            "decode_only": dec, "top_reasons": sorted(reasons.items(),
                                                      key=lambda kv: -kv[1])[:3]}


def recovery_time_ms(run: dict[str, Any], *, high_end_ms: float,
                     baseline: float, tol: float = 0.10) -> float | None:
    """**恢复时间**（定义写在代码里，避免事后换口径）。

    定义：高负载段结束后（`high_end_ms`），**首次**出现"连续 3 个采样点的
    `tpot_ms` 都回到 `baseline × (1 + tol)` 以内"的时刻，减去 `high_end_ms`。
    `baseline` = 该 run 在**低负载段**（`t < high_start`）的 `tpot_ms` 中位。

    ⚠️ 只用 telemetry 的**已记录**字段（`t_ms` / `tpot_ms`），不引入新口径。
    """
    pts = [(t["t_ms"], t.get("tpot_ms")) for t in run["telemetry"]
           if t.get("t_ms") is not None and t.get("tpot_ms") is not None]
    if not pts or baseline is None or baseline <= 0:
        return None
    thr = baseline * (1.0 + tol)
    streak = 0
    for t_ms, tp in pts:
        if t_ms < high_end_ms:
            continue
        if tp <= thr:
            streak += 1
            if streak >= 3:
                return t_ms - high_end_ms
        else:
            streak = 0
    return None


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_runs": len(runs)}
    for key, _ in SUMMARY_METRICS:
        v = _vals(runs, key)
        out[key] = ({"all": v, "median": st.median(v), "range": max(v) - min(v)}
                    if v else {"all": [], "median": None, "range": None})
    for grp, _ in QUANTILE_METRICS:
        for q in ("p50", "p99"):
            v = _q(runs, grp, q)
            out[f"{grp}_{q}"] = ({"all": v, "median": st.median(v), "range": max(v) - min(v)}
                                 if v else {"all": [], "median": None, "range": None})
    # 控制器的"请求值 vs 实际值"未兑现步数（既有字段，直接用）
    unh = [r["effect"].get("n_steps_chunk_request_not_honoured") for r in runs]
    unh = [float(x) for x in unh if isinstance(x, (int, float))]
    out["chunk_request_not_honoured"] = ({"all": unh, "median": st.median(unh)}
                                         if unh else {"all": [], "median": None})
    out["n_control_decisions"] = [r["effect"].get("n_control_decisions") for r in runs]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="第 14 步四臂对比分析（只读）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--high-start-s", type=float, default=120.0,
                    help="ramp 高负载段起点（秒），用于恢复时间")
    ap.add_argument("--high-end-s", type=float, default=210.0,
                    help="ramp 高负载段终点（秒），恢复时间从这里算起")
    args = ap.parse_args(argv)

    skipped: list[tuple[str, str]] = []
    runs = load_runs(args.tag, skipped=skipped)
    if skipped:
        # ⚠️ 与登记侧 §37 同一纪律：**列表 + 非零退出**，不静默、不崩。
        # 但**不是**方法被否证，只是**样本不全** ⇒ 报表照出，状态标"不完整"。
        unparsable = [s for s in skipped if s[1].startswith("UNPARSEABLE")]
        print(f"\n⚠️  {len(skipped)} 个目录未纳入（共 "
              f"{len(skipped) + len(runs)} 个）：")
        for n, why in skipped:
            print(f"     - {n}: {why}")
        if unparsable:
            print(f"   ★ 其中 {len(unparsable)} 个是**截断产物**（尺寸像完好、内容不可解析）"
                  f" ⇒ **按尺寸门槛判定会误判为完整**；处置只能是重拉，不可修。")
        print(f"   ⇒ 本报表分母 = **{len(runs)}**，不是全部重复数。\n")
    if not runs:
        print(f"✗ 没找到 tag={args.tag} 的可解析 run", file=sys.stderr)
        return 2

    # ---- 一致性闸门（验收：不混合不同源码/环境）---------------------------
    commits = {r["commit"] for r in runs}
    dirties = {r["dirty"] for r in runs}
    labels = {r["label"] for r in runs}
    print(f"载入 {len(runs)} 个 run｜tags={args.tag}")
    print(f"  源码提交: {commits}  dirty: {dirties}  标签: {labels}")
    if len(commits) > 1 or True in dirties:
        print("  ⚠️ **混了不同源码状态** ⇒ 按验收第 5 条不得合并解释")
    if labels != {"MEASURED"}:
        print(f"  ⚠️ 存在非 MEASURED 的 run：{labels - {'MEASURED'}} ⇒ 不得进入结论")
    # ⚠️ **豁免控制器关闭的臂**（如 Best Static）：它们的 `signal_criterion` 恒为默认值，
    # 不参与判据选择 ⇒ 把它们算进来会让任何含 static 臂的批次**必出假警**
    # （"与冻结的 decode_p90 不一致"）。口径必须与 `tests/test_step14_configs.py` 一致
    # —— 那里是按 `controller.enabled` 豁免的。实测踩到：两处口径不一致 ⇒ 假警。
    tries = [r for r in runs if r.get("ctrl_enabled") is not False]
    crit = {r["criterion"] for r in tries}
    # `None` = 产物里**没有这个字段** ⇒ 那一刻的 `describe()` 还没记它 ⇒ 实际走的是默认值
    shown = {("未声明（⇒ 默认 tpot_p90）" if c is None else c) for c in crit}
    print(f"  **实际生效的判据** signal_criterion: {shown}"
          f"（已豁免控制器关闭的臂 {len(runs) - len(tries)} 个）")
    if crit != {"decode_p90"}:
        print("  ⚠️ **与第 09 步 §⑥ 冻结的 decode_p90 不一致** ⇒ 这批结果的判据是 "
              f"{shown}，**不得**与冻结方案下的批次混合汇总；"
              "结论里必须写明它实际用的是什么")
    if len({r["shadow_used"] for r in runs}) > 1:
        print("  ⚠️ 混了 shadow 与闭环的 run ⇒ 两者不得合并")

    cells = sorted({r["cell"] for r in runs})
    arms = sorted({r["arm"] for r in runs})
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault((r["cell"], r["arm"]), []).append(r)

    report: dict[str, Any] = {"tag": args.tag, "commits": sorted(commits),
                              "dirty": sorted(map(str, dirties)), "cells": {}}

    for cell in cells:
        print(f"\n{'=' * 78}\n# cell = {cell}\n{'=' * 78}")
        cell_out: dict[str, Any] = {}
        for arm in arms:
            rs = groups.get((cell, arm)) or []
            if not rs:
                continue
            s = summarise(rs)
            extra = {}
            if arm in ("cost", "slo_weave"):
                extra = fallback_rate(rs, arm)
            # 恢复时间（只有 ramp 类负载有分段；其它 cell 不报）
            if cell == "ramp":
                base = []
                for r in rs:
                    v = [t["tpot_ms"] for t in r["telemetry"]
                         if t.get("tpot_ms") is not None
                         and (t.get("t_ms") or 0) < args.high_start_s * 1000]
                    if v:
                        base.append(st.median(v))
                if base:
                    b = st.median(base)
                    rec = [recovery_time_ms(r, high_end_ms=args.high_end_s * 1000, baseline=b)
                           for r in rs]
                    rec = [x for x in rec if x is not None]
                    extra["recovery_ms"] = ({"all": rec, "median": st.median(rec)}
                                            if rec else {"all": [], "median": None})
                    extra["low_segment_tpot_baseline_ms"] = b
            cell_out[arm] = {"summary": s, "extra": extra, "runs": [r["run_id"] for r in rs]}

            print(f"\n-- arm = {arm} （n={len(rs)}）"
                  f"  kind={rs[0]['kind_used']} shadow={rs[0]['shadow_used']}"
                  f" criterion={rs[0]['criterion']}")
            print(f"   达标吞吐 tok/s : 中位 {_fmt(s['slo_goodput_tok_s']['median'])}"
                  f"  极差 {_fmt(s['slo_goodput_tok_s']['range'])}"
                  f"  all={_fmt_list(s['slo_goodput_tok_s']['all'])}")
            print(f"   达标率         : 中位 {_fmt(s['slo_satisfaction_rate']['median'])}"
                  f"  all={_fmt_list(s['slo_satisfaction_rate']['all'])}")
            for g, name in (("ttft", "TTFT"), ("tpot", "TPOT"), ("stall", "stall")):
                print(f"   {name:<5} p50/p99 ms: 中位 {_fmt(s[f'{g}_p50']['median'])} / "
                      f"{_fmt(s[f'{g}_p99']['median'])}")
            print(f"   吞吐 tok/s     : 中位 {_fmt(s['throughput_tok_s']['median'])}"
                  f"  极差 {_fmt(s['throughput_tok_s']['range'])}")
            print(f"   失败/拒绝/未完成: 中位 {_fmt(s['failure_rate']['median'])} / "
                  f"{_fmt(s['rejection_rate']['median'])} / {_fmt(s['unfinished_rate']['median'])}")
            print(f"   账目误差 P99 ms: 中位 {_fmt(s['accounting_error_p99_ms']['median'])}")
            print(f"   单次墙钟 ms    : 中位 {_fmt(s['wall_ms']['median'])}")
            print(f"   请求值未兑现步数: 中位 {_fmt(s['chunk_request_not_honoured']['median'])}")
            print(f"   控制开销       : **未测量**（本版未埋点；见报告说明）")
            if extra:
                print(f"   回退率         : {_fmt(extra.get('fallback_rate'))}"
                      f"  （{extra.get('fallbacks')}/{extra.get('decisions')} 次决策）")
                print(f"     decode-only  : {extra.get('decode_only')}")
                for rsn, n in extra.get("top_reasons") or []:
                    print(f"     理由×{n}: {rsn}")
                if "recovery_ms" in extra:
                    print(f"   恢复时间 ms    : 中位 {_fmt(extra['recovery_ms']['median'])}"
                          f"  all={_fmt_list(extra['recovery_ms']['all'])}")
                    print(f"     （低负载段 TPOT 基线 {_fmt(extra.get('low_segment_tpot_baseline_ms'))} ms）")
        report["cells"][cell] = cell_out

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                       encoding="utf-8", newline="\n")
        print(f"\n已写出：{out}")
    if skipped:
        # ⚠️ **样本不全 ⇒ 非零退出**，让调用方（编排/自动化）不可能把
        # "这次报表绿了"误读成"批次齐了"。这与登记侧 §37 的 rc=1 是同一纪律。
        # 强调：**非零 ≠ 方法被否证**，只表示"本次分母 < 全部重复数"。
        print(f"⚠️ 退出码 1：本次仅纳入 {len(runs)} 个 run，"
              f"另有 {len(skipped)} 个目录不可用（见上）。"
              f"**在批次齐备前，不得用本报表更新结论中的分母。**")
        return 1
    return 0


def _fmt(v: Any) -> str:
    return "—" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def _fmt_list(v: list[float]) -> str:
    return "[" + ", ".join(f"{x:.4g}" for x in v) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
