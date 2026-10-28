#!/usr/bin/env python3
"""批次 3（`lenmix`）的**覆盖率与回退成因分解**分析（**只读产物**）。

受命口径（必须逐条遵守，写进代码而不是靠记性）
------------------------------------------------
1. **只报覆盖率与回退成因分解**。本批的负载是 `lenmix`（混合长度），
   其**服务收益**受 §"三档分离"约束：`S14B3` 不承担"谁更好"的结论。
2. **不得**据回退写「`cost` 更差」。回退率高只说明"**没被用上**"，
   不等于"**用了没用**"（见 `analyze_step14.py` 的三条纪律）。
3. `S14B3-lenmix_threshold` 的 `git_source` 须标 `stamp_stale`
   —— 本脚本**只负责报出它的实际值**，标注由登记侧完成。
4. 分母必须**显式报出**，且不完整产物要**响亮跳过**（与 `analyze_step14.py` 同口径）。

用法::

    python scripts/analyze_s14b3.py --json results/processed/s14b3.json
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

TAG = "S14B3"

#: `arm_state` 里各计数器的含义（来自产物自带的 `_note` 与字段名）
ARM_STATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("n_decisions", "总决策数"),
    ("n_fallback", "回退次数"),
    ("n_degraded", "降级次数"),
    ("n_no_candidates", "无候选次数"),
    ("n_memory_conservative", "因显存保守"),
    ("n_prefill_starved", "prefill 饥饿"),
)


def load_b3(skipped: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """加载 `S14B3` 全部**可解析**的 run。不完整产物响亮跳过（不崩、不静默丢）。"""
    out: list[dict[str, Any]] = []
    for d in sorted(glob.glob(str(_ROOT / "results" / "raw" / f"{TAG}-*"))):
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
            print(f"  ⏭️  {name}: meta.json 为 0 字节", file=sys.stderr)
            continue
        try:
            m = json.loads(raw)
        except json.JSONDecodeError as exc:
            if skipped is not None:
                skipped.append((name, f"UNPARSEABLE({meta.stat().st_size}B)"))
            print(f"  ⏭️  {name}: meta.json 截断（{meta.stat().st_size} B，"
                  f"{exc.msg} @ {exc.pos}）—— 数字不可用", file=sys.stderr)
            continue
        ce = m.get("controller_effect") or {}
        rep = m.get("reproducibility") or {}
        cfg = (m.get("config") or {}).get("controller") or {}
        out.append({
            "dir": name,
            "arm": name.split("lenmix_")[-1].split("-r")[0],
            "label": m.get("data_label"),
            "git_source": rep.get("git_source"),
            "criterion": cfg.get("signal_criterion"),
            "ctrl_enabled": cfg.get("enabled"),
            "effect": ce,
        })
    return out


def rate(num: Any, den: Any) -> float | None:
    """比率；**任一侧不可用就返回 None**（禁止填 0）。"""
    if not isinstance(num, (int, float)) or not isinstance(den, (int, float)):
        return None
    if not den:
        return None
    return num / den


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="S14B3 覆盖率与回退成因分解（只读）")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    skipped: list[tuple[str, str]] = []
    runs = load_b3(skipped)
    if skipped:
        print(f"\n⚠️  {len(skipped)} 个目录未纳入（共 {len(skipped)+len(runs)} 个）：")
        for n, why in skipped:
            print(f"     - {n}: {why}")
        print(f"   ⇒ 本报表分母 = **{len(runs)}**（不是全部重复数）\n")
    if not runs:
        print("✗ 没有可解析的 S14B3 run（可能尚未回拉）", file=sys.stderr)
        return 2

    print(f"载入 {len(runs)} 个 run｜tag={args.tag}")
    labs = {r["label"] for r in runs}
    srcs = {r["git_source"] for r in runs}
    print(f"  数据标签: {labs}")
    print(f"  git_source: {srcs}")
    if "stamp_stale" in srcs:
        print("  ⚠️ 本批含 stamp_stale 的 run（登记时须逐条标注，**不得**混算）")

    report: dict[str, Any] = {"tag": args.tag, "n": len(runs),
                              "labels": sorted(x for x in labs if x),
                              "git_sources": sorted(x for x in srcs if x),
                              "arms": {}}

    arms = sorted({r["arm"] for r in runs})
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        print(f"\n{'='*70}\n-- arm = {arm} （n={len(rs)}）\n{'='*70}")
        # ---- 回退成因分解 ----
        print(f"  {'计数器':<24s} {'逐 run':<28s} {'中位':>10s} {'占比中位':>9s}")
        arm_out: dict[str, Any] = {"n": len(rs), "counters": {}, "coverage": {}}
        for key, cn in ARM_STATE_FIELDS:
            vals = [(r["effect"].get("arm_state") or {}).get(key) for r in rs]
            nums = [v for v in vals if isinstance(v, (int, float))]
            dens = [(r["effect"].get("arm_state") or {}).get("n_decisions") for r in rs]
            ratios = [rate(n, d) for n, d in zip(vals, dens)]
            ratios = [x for x in ratios if x is not None]
            med = st.median(nums) if nums else None
            rmed = st.median(ratios) if ratios else None
            arm_out["counters"][key] = {"all": nums, "median": med,
                                        "rate_median": rmed}
            shown = "[" + ", ".join(str(v) for v in vals) + "]"
            print(f"  {cn+' ('+key+')':<24s} {shown:<28s} "
                  f"{('—' if med is None else f'{med:.0f}'):>10s} "
                  f"{('—' if rmed is None else f'{rmed:.4f}'):>9s}")

        # ---- 覆盖率：建议值 vs 实际生效值 ----
        print(f"\n  覆盖率（**建议被兑现的比例**；`_note` 明示差额=`建议未兑现`）:")
        cov_out: dict[str, Any] = {}
        for pair in (("chunk_requested", "chunk_effective"),
                     ("max_batched_tokens_requested", "max_batched_tokens_effective")):
            kq, ke = pair
            qr = [(r["effect"].get(kq) or {}) for r in rs]
            er = [(r["effect"].get(ke) or {}) for r in rs]
            qm = [q.get("max") for q in qr if isinstance(q.get("max"), (int, float))]
            em = [e.get("max") for e in er if isinstance(e.get("max"), (int, float))]
            cov = None
            if qm and em:
                covs = [e / q for q, e in zip(qm, em) if q]
                cov = st.median(covs) if covs else None
            print(f"    {qk if False else kq+'.max 中位':<42s} "
                  f"{('—' if not qm else f'{st.median(qm):.0f}'):>10s}")
            print(f"    {ke+'.max 中位':<42s} "
                  f"{('—' if not em else f'{st.median(em):.0f}'):>10s}")
            print(f"    {'⇒ 覆盖率(生效/建议) 中位':<42s} "
                  f"{('—' if cov is None else f'{cov:.4f}'):>10s}")
            cov_out[kq] = {"requested_max_median": st.median(qm) if qm else None,
                           "effective_max_median": st.median(em) if em else None,
                           "coverage_median": cov}
        arm_out["coverage"] = cov_out

        # ---- 步级未兑现（受命要求的口径，与 requests 级区分）----
        ns = [(r["effect"].get("n_control_decisions"),
               r["effect"].get("n_steps"),
               r["effect"].get("n_steps_chunk_request_not_honoured"),
               r["effect"].get("n_steps_prefill_below_chunk_request")) for r in rs]
        print(f"\n  步级口径（**不是**请求级达标率）:")
        for i, nm in enumerate(("n_control_decisions", "n_steps",
                                "n_steps_chunk_request_not_honoured",
                                "n_steps_prefill_below_chunk_request")):
            vals = [x[i] for x in ns if isinstance(x[i], (int, float))]
            print(f"    {nm:<42s} {'['+', '.join(str(v) for v in vals)+']':<26s} "
                  f"中位 {('—' if not vals else f'{st.median(vals):.0f}')}")
            arm_out.setdefault("step_level", {})[nm] = {
                "all": vals, "median": st.median(vals) if vals else None}
        report["arms"][arm] = arm_out

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                       encoding="utf-8", newline="\n")
        print(f"\n已写出：{out}")
    if skipped:
        print(f"⚠️ 退出码 1：仅纳入 {len(runs)} 个 run，另有 {len(skipped)} 个不可用。"
              f"**非零 ≠ 方法被否证**，只表示分母不全。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
