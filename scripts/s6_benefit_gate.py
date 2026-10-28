#!/usr/bin/env python3
"""S6 服务收益档闸门：**机械执行** `EXPERIMENTS.md` §S6 预注册的判据。

为什么要有这个脚本
------------------
§S6 的规则（机械前提 / 主预期 / 否证形态）是**预注册**的，必须**机械执行**，
不能事后再"看着数据商量"。把规则写进代码，是让它**无法被事后调整**的唯一办法。
（与 `analyze_scan_batch.py`（S14S 档）、`d1_gate.py`（机制诊断档）并列，
三档判据**不得互相替代**：见 §7d 第 6 节「三档分离」。）

本闸门判什么
------------
1. **机械前提**（最先看，不过则整批不判服务收益 —— §5）：
   * `cost` 臂轨迹里**必须**有 `cost_policy` 条目（>0）；
   * `threshold` 臂轨迹里**不得**有 `cost_policy` 条目；
   * `static` 臂轨迹**应为空**（控制器整组跳过）。
   任一不满足 ⇒ 该臂没被跑到 ⇒ 判「**接线失败**」，**不是**「无收益」。
2. **主预期**（§5）：`cost` 臂的**达标吞吐**与**达标率**不低于 `threshold`，
   且其中**至少一项严格高于** `static`。
3. **否证形态**（§7）：
   * 三臂在 3 重复稳健 CV 带内**全部重合** ⇒ 「**不可区分**」，不得声称任一臂更优；
   * `cost` 严格劣于 `static` ⇒ 「**负收益**」。

三态返回（退出码）
------------------
  0 = PASS（cost 有服务收益，按主预期成立）
  1 = FAIL（cost 严格劣于 static ⇒ 负收益）
  2 = INCONCLUSIVE（不可区分 / 接线失败 / 证据不足 —— **不是**「方法没用」）
  3 = 输入错

★ 键名全部对**真产物**核对过（`results/raw/*/meta.json`）——
  分位块是**嵌套的**：`summary.tpot.p90`，不是 `summary.tpot_ms_p90`。
  凭记忆写会静默取到 None，"不可区分"与"读错键"就无法区分。

用法::

    python scripts/s6_benefit_gate.py --tag S6 --root results/raw [--repeats 3]
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

#: 三臂（§2 定义）
ARMS = ("static", "threshold", "cost")
#: 稳健 CV 上限（与 §7d / AGENTS.md A5 同口径：1.4826·MAD/median）
CV_LIMIT = 0.10
#: 每臂要求的重复数（§7：`--repeats ≥ 3`；判据 = 3 轮全部满足，取最差一次）
MIN_REPEATS = 3

PASS, FAIL, INCONCLUSIVE, BAD_INPUT = 0, 1, 2, 3


def robust_cv(values: list[float]) -> float | None:
    """1.4826·MAD / median —— 对 outlier 有抵抗力的离散度（AGENTS.md A5）。"""
    if len(values) < 2:
        return None
    med = statistics.median(values)
    if med == 0:
        return None
    mad = statistics.median([abs(v - med) for v in values])
    return 1.4826 * mad / abs(med)


def _arm_of(m: dict, dirname: str) -> str | None:
    """从产物里取**臂名**。

    ★ 不靠目录名猜：run_id 形如 `S6-short_chat-service_benefit_c2_m4_cost[-rN]`，
      而 cell 名里既有 `-` 又有 `_` ⇒ 按 `-` 切会拿到 `service_benefit_c2_m4_cost`
      （整个 cell 名），**不是**臂 —— 实测踩到（第一版 `stem.split('-')[-1]` 让三臂
      全部匹配失败、`run 0 次`，而 9 个目录明明在）。
      这与 `analyze_scan_batch.py` 面对的是**两种不同形状**的 run-id。

    ⇒ **唯一可信来源 = 产物里记录的 config 名**：
      `context.config_path`（绝对路径）/ `config.config_stem` / `summary.config`。
      臂名 = config stem 的**最后一段下划线后缀**（`service_benefit_c2_m4_<arm>`）。
    """
    cand = None
    ctx = m.get("context") or {}
    cp = ctx.get("config_path")
    if isinstance(cp, str) and cp:
        cand = Path(cp).stem
    if not cand:
        cs = (m.get("config") or {}).get("config_stem")
        if isinstance(cs, str) and cs:
            cand = cs
    if not cand:
        sc = (m.get("summary") or {}).get("config")
        if isinstance(sc, str) and sc:
            cand = sc
    if not cand:
        return None
    # 臂名 = 末尾下划线后缀，且必须在已知臂集合里
    tail = cand.rsplit("_", 1)[-1]
    return tail if tail in ARMS else None


def load_runs(tag: str, root: Path) -> dict[str, list[dict]]:
    """按臂聚合：{arm: [每次重复的 meta.json 内容, ...]}。臂名取自产物记录的 config。"""
    per_arm: dict[str, list[dict]] = {a: [] for a in ARMS}
    for d in sorted(glob.glob(str(root / f"{tag}-*"))):
        p = Path(d)
        mp = p / "meta.json"
        if not mp.is_file():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        arm = _arm_of(m, p.name)
        if arm in per_arm:
            m["_dir"] = p.name
            per_arm[arm].append(m)
    return per_arm


def _trace_entries(m: dict) -> list[dict]:
    t = m.get("controller_trace")
    return t if isinstance(t, list) else []


def n_cost_policy_entries(m: dict) -> int:
    """该运行轨迹里带 `cost_policy` 的条目数（机械前提的核心量）。"""
    return sum(1 for e in _trace_entries(m) if e.get("cost_policy"))


def arm_state_of(m: dict) -> dict | None:
    """取最后一次决策的 `arm_state`（若无则为 None）—— 取自 trace 末条 `state` 或顶层。"""
    ce = m.get("controller_effect") or {}
    if "arm_state" in ce and isinstance(ce["arm_state"], dict):
        return ce["arm_state"]
    for e in reversed(_trace_entries(m)):
        st = e.get("state")
        if isinstance(st, dict) and ("n_degraded" in st or "n_decisions" in st):
            return st
    return None


def arm_facts(runs: list[dict]) -> dict:
    """把一个臂的多次重复汇成机械事实（连报分母）。"""
    n_cost = [n_cost_policy_entries(m) for m in runs]
    n_trace = [len(_trace_entries(m)) for m in runs]
    goodput, sat = [], []
    for m in runs:
        s = m.get("summary") or {}
        gp = s.get("slo_goodput_req_s")
        st = s.get("slo_satisfaction_rate")
        if isinstance(gp, (int, float)):
            goodput.append(float(gp))
        if isinstance(st, (int, float)):
            sat.append(float(st))
    return {
        "n_runs": len(runs),
        "dirs": [m.get("_dir") for m in runs],
        "n_cost_policy_entries": sum(n_cost),
        "per_run_cost_policy": n_cost,
        "n_trace_entries": sum(n_trace),
        "per_run_trace_len": n_trace,
        "goodput_req_s": goodput,
        "satisfaction": sat,
        "goodput_min": min(goodput) if goodput else None,   # 取最差一次（§7 口径）
        "sat_min": min(sat) if sat else None,
        "goodput_cv": robust_cv(goodput),
        "sat_cv": robust_cv(sat),
    }


def mechanical_premise(facts: dict) -> tuple[list[str], list[str]]:
    """§5 机械前提。返回 (fails, notes)。"""
    fails, notes = [], []
    # cost 臂：必须有 cost_policy 条目
    f = facts["cost"]
    if f["n_cost_policy_entries"] <= 0:
        fails.append(
            f"接线：cost 臂轨迹里**没有** `cost_policy` 条目"
            f"（轨迹 {f['n_trace_entries']} 条 / {f['n_runs']} 次重复）"
            " ⇒ cost 臂根本没被跑到（判「接线失败」，不是「无收益」）")
    else:
        notes.append(f"接线过：cost 臂有 `cost_policy` 条目 {f['n_cost_policy_entries']} 条"
                     f"（逐次 {f['per_run_cost_policy']}）")
    # threshold 臂：不得有 cost_policy 条目
    f = facts["threshold"]
    if f["n_cost_policy_entries"] != 0:
        fails.append(
            f"接线：threshold 臂轨迹里**出现了** `cost_policy` 条目"
            f"（{f['n_cost_policy_entries']} 条）⇒ 臂没被正确区分（cost 逻辑串了）")
    elif f["n_trace_entries"] <= 0:
        fails.append("接线：threshold 臂**轨迹为空**（控制器应运行但没记录）⇒ 不可判")
    else:
        notes.append(f"接线过：threshold 臂 `cost_policy` 条目 0，轨迹 {f['n_trace_entries']} 条")
    # static 臂：轨迹应为空（控制器整组跳过）
    f = facts["static"]
    if f["n_trace_entries"] > 0:
        notes.append(f"⚠️ static 臂轨迹非空（{f['n_trace_entries']} 条）"
                     " —— §5 预期「应为空」；请人工确认控制器是否真的整组跳过")
    else:
        notes.append("接线过：static 臂轨迹为空（控制器整组跳过，符合 §5）")
    return fails, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S6 服务收益档闸门（机械执行 §S6 预注册判据）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default="results/raw", help="产物根（默认 results/raw）")
    ap.add_argument("--repeats", type=int, default=MIN_REPEATS)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"✗ root 不存在：{root}", file=sys.stderr)
        return BAD_INPUT

    per_arm = load_runs(args.tag, root)
    facts = {a: arm_facts(per_arm[a]) for a in ARMS}

    print(f"=== S6 服务收益闸门｜tag={args.tag}｜每臂要求重复 {args.repeats} ===")
    for a in ARMS:
        f = facts[a]
        print(f"[{a:<9}] run {f['n_runs']} 次；cost_policy 条目 {f['n_cost_policy_entries']}"
              f"；轨迹 {f['n_trace_entries']} 条"
              f"；达标吞吐(最差) {_fmt(f['goodput_min'])}"
              f"；达标率(最差) {_fmt(f['sat_min'])}")
    print()

    # --- 重复数是否够（§7）----------------------------------------------------
    short = [a for a in ARMS if facts[a]["n_runs"] < args.repeats]
    if short:
        print(f"⚠️ 以下臂重复数 < {args.repeats}：{short} ⇒ INCONCLUSIVE（证据不足）")

    # --- 机械前提 -------------------------------------------------------------
    fails, notes = mechanical_premise(facts)
    for n in notes:
        print("  ✓ " + n)
    for f in fails:
        print("  ✗ " + f)
    print()

    verdict, reasons = _judge(facts, fails, short, args.repeats)
    print("=" * 60)
    print(f"判定：{verdict}")
    for r in reasons:
        print("  - " + r)
    print("=" * 60)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tag": args.tag,
            "repeats_required": args.repeats,
            "verdict": verdict,
            "reasons": reasons,
            "mechanical_fails": fails,
            "mechanical_notes": notes,
            "arms": facts,
        }
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print(f"已写出：{p}")

    return {"PASS": PASS, "FAIL": FAIL, "INCONCLUSIVE": INCONCLUSIVE}[verdict]


def _judge(facts: dict, mech_fails: list[str], short: list[str],
           repeats: int) -> tuple[str, list[str]]:
    """按 §5/§7 机械判定。返回 (verdict, reasons)。"""
    reasons: list[str] = []
    # 机械前提不过 ⇒ 接线失败（INCONCLUSIVE，**不是** FAIL/负收益）
    if mech_fails:
        reasons.append("机械前提不过 ⇒ 判「接线失败」。**不得**把整批读成「没收益」。")
        return "INCONCLUSIVE", reasons
    if short:
        reasons.append(f"重复数不足（{short} < {repeats}）⇒ 证据不足，不判收益。")
        return "INCONCLUSIVE", reasons
    if any(facts[a]["n_runs"] == 0 for a in ARMS):
        reasons.append("有臂没有产物 ⇒ 证据不足。")
        return "INCONCLUSIVE", reasons

    c, t, s = facts["cost"], facts["threshold"], facts["static"]

    # --- 主预期（§5）：cost 不低于 threshold，且至少一项严格高于 static ------
    def _ge(a, b):
        return a is not None and b is not None and a >= b

    def _gt(a, b):
        return a is not None and b is not None and a > b

    cost_ge_thr_gp = _ge(c["goodput_min"], t["goodput_min"])
    cost_ge_thr_sat = _ge(c["sat_min"], t["sat_min"])
    cost_gt_static = _gt(c["goodput_min"], s["goodput_min"]) or _gt(c["sat_min"], s["sat_min"])

    reasons.append(f"cost vs threshold：达标吞吐 {_fmt(c['goodput_min'])} vs {_fmt(t['goodput_min'])}"
                   f"（{'≥' if cost_ge_thr_gp else '<'}）；"
                   f"达标率 {_fmt(c['sat_min'])} vs {_fmt(t['sat_min'])}"
                   f"（{'≥' if cost_ge_thr_sat else '<'}）")
    reasons.append(f"cost vs static：达标吞吐 {_fmt(c['goodput_min'])} vs {_fmt(s['goodput_min'])}；"
                   f"达标率 {_fmt(c['sat_min'])} vs {_fmt(s['sat_min'])}"
                   f" ⇒ 至少一项严格高于 static = {cost_gt_static}")

    # --- 否证形态（§7）：三臂全部落在稳健 CV 带内 ⇒ 不可区分 -----------------
    cvs = [facts[a]["goodput_cv"] for a in ARMS]
    cv_known = [v for v in cvs if v is not None]
    if cv_known and max(cv_known) <= CV_LIMIT:
        # 三者读数差异是否都在各自 CV 带内
        gps = [facts[a]["goodput_min"] for a in ARMS]
        sats = [facts[a]["sat_min"] for a in ARMS]
        gp_spread = (max(gps) - min(gps)) if all(v is not None for v in gps) else None
        sat_spread = (max(sats) - min(sats)) if all(v is not None for v in sats) else None
        # 用最差读数之差与各自 CV 带比较：若差异 < 读数×CV，视为重合
        def _within(spread, vals, cvs_):
            if spread is None or any(v is None for v in vals):
                return False
            ref = max(abs(v) for v in vals) or 1.0
            band = ref * max(v for v in cvs_ if v is not None)
            return spread <= band
        if _within(gp_spread, gps, cvs) and _within(sat_spread, sats, cvs):
            reasons.append(
                f"否证形态命中：三臂达标吞吐极差 {_fmt(gp_spread)}、达标率极差 {_fmt(sat_spread)}"
                f" 均 ≤ 各自稳健 CV 带（上限 {CV_LIMIT:.0%}）⇒ 判「**在本负载上三臂不可区分**」，"
                "**不得**声称任一臂更优。")
            return "INCONCLUSIVE", reasons

    # --- cost 严格劣于 static ⇒ 负收益 ---------------------------------------
    cost_lt_static = (_ge(s["goodput_min"], c["goodput_min"]) and s["goodput_min"] != c["goodput_min"]) \
        or (_ge(s["sat_min"], c["sat_min"]) and s["sat_min"] != c["sat_min"])
    if cost_lt_static and not cost_gt_static:
        reasons.append("cost 严格劣于 static ⇒ 判「成本模型自适应在本负载上为**负收益**」（§7）。")
        return "FAIL", reasons

    # --- 主预期成立 -----------------------------------------------------------
    if cost_ge_thr_gp and cost_ge_thr_sat and cost_gt_static:
        reasons.append("主预期成立：cost ≥ threshold（两项），且至少一项严格 > static ⇒ PASS。")
        return "PASS", reasons

    reasons.append("证据既不构成「不可区分」，也不构成「负收益」，也未满足主预期"
                   " ⇒ INCONCLUSIVE（需人工看 §6 五组量再定）。")
    return "INCONCLUSIVE", reasons


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
