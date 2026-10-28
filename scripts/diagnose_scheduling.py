#!/usr/bin/env python3
"""**证据链三诊断**：把"预测误差"与"最终服务收益"连起来（第 14 步验收 + 评审第 3 步）。

为什么要有它
------------
"误差下降了"本身**不是**结论。要有说服力，必须能说：
**误差下降 → 错误调度减少 → 达标吞吐提高**。
所以除误差以外要盯三个量：

| 诊断 | 回答的问题 | 本脚本的定义（**写死在这里，避免事后换口径**） |
| --- | --- | --- |
| **D1 预测可行但实际超时** | 模型是否错误接纳了过大的 prefill？ | 决策里**选中了可行候选**（有 `predicted_ms` 且未回退）的那些步中，**紧随其后的那一步**的**实际时长**超过 deadline 的比例 |
| **D2 过度保守造成的等待** | 安全余量/回退是否让请求不必要地排队？ | `prefill_budget_effective == 0` **且** `queue_depth > 0` 的步数，以及这些步上的 `Σqueue_depth`（"等在队列里的请求·步"） |
| **D3 最终 SLO goodput** | 更准确的模型是否改善了实际服务？ | 直接取 `summary.slo_goodput_tok_s` 与 `slo_satisfaction_rate` |

⚠️ **口径边界（必须与数字一起给）**
* D1 的 deadline 取**配置的 `target_tpot_ms`**（不是从产物里读的——轨迹里没记 `budget_ms`）；
  而实际步时长 = 遥测相邻两点的 `t_ms` 之差（**含** host 提交与调度间隙）。
  两者口径**不完全同源**，所以 D1 只当**相对比较**用（同批次内横向比各臂），不当绝对值。
* D2 是**代理量**：`prefill 额度=0 且队列非空`不等于"本可以推进 prefill"，
  但它精确对应"**有活在等、这一步却没有给 prefill 预算**"⇒ 对"保守是否造成排队"是合适的指示器。

用法::

    python scripts/diagnose_scheduling.py --tag S14B1
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

#: 从 run_id 里拆出 cell 与**臂名**：`<TAG>-<workload>-<cell>_<arm>[-rN]`
#: ⚠️ **不能用配置里的 `controller.kind` 分组**：`static` 臂的 kind 也是 `threshold`
#: （它只是把控制器关掉了）⇒ 按 kind 分组会把 `static` 与 `threshold` 合并，
#: 得出一个"n=10、混了两套处理"的假组。实测踩到过。
#:
#: ⚠️ 2026-09-27：解析**收归唯一实现** `ascend_opt.run_ids`。此前本文件与
#: `analyze_step14.py` 各写一份同样的正则，而 D1① 闸门又用"字符串替换"解析出了
#: **另一个**臂名（`fixed_cost_pilot` vs `cost_pilot`）⇒ 闸门在自己字典里查不到数据，
#: 报出两条**假失败**。同一件事有两份实现，迟早分叉，且分叉的表现是"查不到"。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.run_ids import parse_run_id  # noqa: E402


def load(tag: str, arm_glob: str = "*") -> list[dict[str, Any]]:
    """载入某 tag 的产物。

    ⚠️ **残缺产物必须"响亮地跳过"，不能整脚本崩掉，也不能静默当空**：
    回拉是分文件的，中断会在本地留下**截断**的 `meta.json`/`telemetry.jsonl`
    （实测踩到：`meta.json` 32768 B 截断 ⇒ `json.loads` 在第 1604 行抛
    `JSONDecodeError`，整个分析**一个数都出不来**，看起来像"脚本坏了"）。
    两条纪律：
    * 截断/解析失败 ⇒ **记进 `skipped` 并在末尾打印**，**进程返回非零**
      （否则"少了几格"会被读成"这批就这么点"）；
    * 截断 ≠ 数据为空 ⇒ **不得**把解析失败的文件当"没有 telemetry"处理。
    """
    out = []
    skipped: list[tuple[str, str]] = []
    for d in sorted(glob.glob(str(_ROOT / "results" / "raw" / f"{tag}-*{arm_glob}*"))):
        mp = Path(d) / "meta.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            skipped.append((Path(d).name, f"meta.json 解析失败（疑截断）：{exc}"))
            continue
        tel_path = Path(d) / "telemetry.jsonl"
        tel = []
        if tel_path.exists():
            try:
                tel = [json.loads(l) for l in tel_path.read_text(encoding="utf-8").splitlines()
                       if l.strip()]
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                skipped.append((Path(d).name, f"telemetry.jsonl 解析失败（疑截断）：{exc}"))
                continue
        rid = (m.get("summary") or {}).get("run_id") or Path(d).name
        parsed = parse_run_id(rid)
        out.append({
            "run": rid,
            "arm": (parsed.arm if parsed is not None
                    else str(((m.get("config") or {}).get("controller") or {}).get("kind"))),
            "dir": Path(d).name,
            "label": m.get("data_label"),
            "kind": ((m.get("config") or {}).get("controller") or {}).get("kind"),
            "deadline_ms": ((m.get("config") or {}).get("controller") or {}).get("target_tpot_ms"),
            "trace": m.get("controller_trace") or [],
            "summary": m.get("summary") or {},
            "tel": tel,
        })
    if skipped:
        print(f"⚠️  跳过 {len(skipped)} 个残缺产物（**不是**'没有这批数据'，是文件被截断）：")
        for name, why in skipped:
            print(f"     - {name}: {why}")
        print("   ⇒ 先按 scripts/pull_gz.sh 补齐再复算；结论里的分母须排除这些格。")
    load.skipped = skipped  # type: ignore[attr-defined]
    return out


def step_durations_ms(tel: list[dict]) -> dict[int, float]:
    """`{step: 实际时长}`，用相邻 `t_ms` 之差。最后一步没有下一个点 ⇒ 不给值。"""
    pts = [(int(t["step"]), float(t["t_ms"])) for t in tel if t.get("t_ms") is not None]
    pts.sort()
    return {pts[i][0]: pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)}


def d1_over_deadline(run: dict) -> dict[str, Any]:
    """预测可行、实际超时。"""
    deadline = run["deadline_ms"]
    if not deadline:
        return {"n": 0, "n_over": 0, "rate": None}
    dur = step_durations_ms(run["tel"])
    n = over = 0
    examples = []
    for e in run["trace"]:
        d = e.get("cost_policy")
        if not d or d.get("fallback_reason"):
            continue                       # 只看"模型给了预测并采纳"的决策
        cs = [c for c in (d.get("candidates") or []) if c.get("feasible")]
        if not cs or d.get("chunk") in (None, 0):
            continue
        nxt = int(e["step"]) + 1
        if nxt not in dur:
            continue
        n += 1
        if dur[nxt] > deadline:
            over += 1
            if len(examples) < 3:
                examples.append({"step": int(e["step"]), "actual_next_ms": round(dur[nxt], 1),
                                 "deadline_ms": deadline,
                                 "predicted_ms": cs[0].get("predicted_ms")})
    return {"n": n, "n_over": over, "rate": (over / n if n else None), "examples": examples}


def d2_conservative_wait(run: dict) -> dict[str, Any]:
    """过度保守造成的等待：prefill 额度=0 且队列非空。"""
    steps = [t for t in run["tel"]
             if (t.get("prefill_budget_effective") or 0) == 0 and (t.get("queue_depth") or 0) > 0]
    q = [int(t.get("queue_depth") or 0) for t in steps]
    tot_steps = len(run["tel"])
    return {"blocked_steps": len(steps), "total_steps": tot_steps,
            "blocked_share": (len(steps) / tot_steps if tot_steps else None),
            "queue_step_sum": sum(q), "queue_depth_median": (st.median(q) if q else None)}


def d0_candidate_admission(run: dict[str, Any]) -> dict[str, Any]:
    """**判决层**（D0）：控制器真的**挑得出**候选吗？挑不出时是为什么？

    为什么必须有这一层：`fixed` 格上 cost 臂的回退率曾被解释成"空闲步回退"——**错了**
    （空闲步 = 0；94.4% 是"batch 不在标定网格"）。项目纪律写明
    **"没被用上" ≠ "用了没用"**：若候选**几乎全被判据拒掉**，那么"cost 臂与 threshold 表现接近"
    说明的是**判据太严 / 候选没被用上**，**不是**"成本模型没用"。

    实测（S14D `fixed_cost`，37 次决策、~141 次候选评估）：**只有 8 个可行候选**，
    拒因分两类 ——
    * `预测 108.6 + host 25.3 = 133.9ms 超预算`（host 项把边缘候选挡掉，**本轮修复的严化效应**）；
    * `预测 943.1 + host 21.6 = 964.7ms 超预算`（v5 的 `uniform` 口径把 mixed 点放大 ~b 倍）。

    ⇒ 三态**不许合并**：可行 / 被拒 / **模型没给预测**（范围外）。
    拒因按**归一化文本**分组（数字换成 `N`），但每组保留**一个逐字样本** ——
    量级本身就是结论（943 与 108 是两件不同的事）。
    """
    n_dec = n_cand = n_feas = 0
    reasons: dict[str, dict[str, Any]] = {}
    for e in run.get("trace") or []:
        d = e.get("cost_policy") or e.get("slo_weave")
        if not d:
            continue
        n_dec += 1
        for c in (d.get("candidates") or []):
            n_cand += 1
            if c.get("feasible"):
                n_feas += 1
            raw = str(c.get("reject_reason") or "")
            if not raw:
                continue
            key = re.sub(r"\d+(?:\.\d+)?", "N", raw)[:60]
            slot = reasons.setdefault(key, {"n": 0, "example": raw})
            slot["n"] += 1
    return {"n_decisions": n_dec, "n_candidates": n_cand, "n_feasible": n_feas,
            "feasible_rate": (n_feas / n_cand if n_cand else None),
            "top_reject_reasons": sorted(reasons.values(),
                                         key=lambda v: -v["n"])[:6]}


def coverage_of_model(run: dict[str, Any]) -> dict[str, Any]:
    """**模型覆盖率**：本次运行里，模型是否**给出过预测**（未被"范围外"全拒）。

    为什么它与 D0 分开
    ------------------
    D0 问的是"候选**被接纳**的比例"（还叠了预算判据）；本量只问
    "模型**有没有话可说**"——它是查表/拟合路径的**头部健康指标**：
    覆盖率低 ⇒ 大量决策走了回退（回到阈值控制器），此时"cost 臂与 threshold 表现接近"
    说明的是**模型没被用上**，**不是**"模型没用"。
    ⇒ 与 `scripts/check_model_coverage.py` 的口径**互补**：
      那边按 run 级汇"有效工作决策"（分母剔空闲，**部署视角**）；
      这边按**决策级**数"至少有一个候选拿到预测"（分母含所有决策，**模型视角**）。
    两者**都要报**，且**不得互相替代**（分母不同）。

    判定规则：一个决策里**至少有一个候选**的 `reject_reason` **不是**"模型范围外"
    ⇒ 该决策被模型覆盖。用"是否有候选**没有**范围外拒因"近似"模型给了值"，
    因为轨迹里 `predicted_ms` 只在可行候选上非空（被预算拒的候选也带 `predicted_ms`）。
    """
    n_dec = n_covered = 0
    for e in run.get("trace") or []:
        d = e.get("cost_policy") or e.get("slo_weave")
        if not d:
            continue
        cands = d.get("candidates") or []
        if not cands:
            continue
        n_dec += 1
        # 有任一候选**拿到了预测**（predicted_ms 非空）或拒因不是"范围外" ⇒ 被覆盖
        if any((c.get("predicted_ms") is not None)
               or ("范围外" not in str(c.get("reject_reason") or ""))
               for c in cands):
            n_covered += 1
    return {"n_decisions": n_dec, "n_covered": n_covered,
            "coverage": (n_covered / n_dec if n_dec else None)}


def _coverage_by_arm(rs: list[dict[str, Any]]) -> dict[str, Any]:
    """把一个臂的多次重复的**决策级覆盖率**并起来（分子分母各求和，不是取中位数）。

    ⚠️ 并法必须是"求和再相除"：对**率**取中位数会得到一个"谁也不满足"的假值 ——
    3 次重复各 0.6/0.4/0.5，中位数 0.5 并不对应任何一次真实运行，
    而"总覆盖 / 总决策"才是可解释的。
    """
    per = [coverage_of_model(r) for r in rs]
    nd = sum(x["n_decisions"] for x in per)
    nc = sum(x["n_covered"] for x in per)
    return {"n_decisions": nd, "n_covered": nc,
            "coverage": (nc / nd if nd else None),
            "per_run": [x["coverage"] for x in per]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="证据链三诊断（只读产物）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    runs = load(args.tag)
    if not runs:
        print(f"✗ 没找到 tag={args.tag}", file=sys.stderr)
        return 2

    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(str(r["arm"]), []).append(r)   # 按**臂名**，不是 kind

    print(f"tag={args.tag}｜run={len(runs)}｜臂={sorted(groups)}")
    if len(groups) < 2:
        print("  ⚠️ 只识别出 1 个臂 —— 检查 run_id 是否符合 `<TAG>-<workload>-<cell>_<arm>`")
    print()
    # ---- D0 判决层：候选被接纳的比例与拒因（**先于 D1/D2/D3**）----------------
    print("== D0 判决层（候选被接纳的比例）==")
    print(f"{'臂':<12}{'决策数':>7}{'候选数':>7}{'可行数':>7}{'可行率':>9}")
    d0_by_arm: dict[str, Any] = {}
    for arm, rs in sorted(groups.items()):
        ds = [d0_candidate_admission(r) for r in rs]
        nc = sum(x["n_candidates"] for x in ds)
        nf = sum(x["n_feasible"] for x in ds)
        nd = sum(x["n_decisions"] for x in ds)
        rate = (nf / nc) if nc else None
        print(f"{arm:<12}{nd:>7}{nc:>7}{nf:>7}"
              f"{('—' if rate is None else f'{rate:.1%}'):>9}")
        merged: dict[str, dict[str, Any]] = {}
        for x in ds:
            for r in x["top_reject_reasons"]:
                key = re.sub(r"\d+(?:\.\d+)?", "N", r["example"])[:60]
                slot = merged.setdefault(key, {"n": 0, "example": r["example"]})
                slot["n"] += r["n"]
        d0_by_arm[arm] = {"n_decisions": nd, "n_candidates": nc, "n_feasible": nf,
                          "feasible_rate": rate,
                          "top_reject_reasons": sorted(merged.values(),
                                                       key=lambda v: -v["n"])[:6]}
    for arm, v in sorted(d0_by_arm.items()):
        if not v["top_reject_reasons"]:
            continue
        print(f"  -- {arm} 的拒因（归一化分组，附逐字样本）:")
        for r in v["top_reject_reasons"]:
            print(f"     ×{r['n']:<5} {r['example'][:78]}")
    print("  ⚠️ **可行率低时不得下「这个方法没用」的结论** —— 说明的是候选被拒/没被用上；")
    print("     要区分四种成因：判据口径（如 host 项）、标定语义（mixed 形状口径）、")
    print("     候选集本身（chunk 下限是否高于该平台上**可行的 prefill 长度**）、以及模型范围外。")
    print()

    header = (f"{'臂':<12}{'n':>3}{'D1 可行超时率':>14}{'D2 阻塞步占比':>14}"
              f"{'D2 Σ队列·步':>13}{'D3 达标吞吐':>12}{'D3 达标率':>10}{'TTFT排队中位':>13}")
    print(header)
    print('-' * len(header))
    report: dict[str, Any] = {"tag": args.tag, "arms": {}}
    for kind, rs in sorted(groups.items()):
        d1 = [d1_over_deadline(r) for r in rs]
        d2 = [d2_conservative_wait(r) for r in rs]
        gp = [float((r["summary"].get("slo_goodput_tok_s") or 0.0)) for r in rs]
        sat = [float((r["summary"].get("slo_satisfaction_rate") or 0.0)) for r in rs]
        qm = [float(((r["summary"].get("queue") or {}).get("p50") or 0.0)) for r in rs]
        r1 = [x["rate"] for x in d1 if x["rate"] is not None]
        r2 = [x["blocked_share"] for x in d2 if x["blocked_share"] is not None]
        qs = [x["queue_step_sum"] for x in d2]
        med = lambda v: (st.median(v) if v else None)  # noqa: E731
        f = lambda v, p='{:.2%}': ('—' if v is None else p.format(v))  # noqa: E731
        print(f"{kind:<12}{len(rs):>3}{f(med(r1)):>14}{f(med(r2)):>14}"
              f"{(str(med(qs)) if qs else '—'):>13}{med(gp):>12.2f}{f(med(sat), '{:.3f}'):>10}"
              f"{med(qm):>13.0f}")
        report["arms"][kind] = {
            "n": len(rs), "d1_rate_median": med(r1), "d2_blocked_share_median": med(r2),
            "d2_queue_step_sum_median": med(qs), "d3_goodput_median": med(gp),
            "d3_satisfaction_median": med(sat), "ttft_queue_ms_median": med(qm),
            "d1_examples": next((x["examples"] for x in d1 if x.get("examples")), []),
            "d1_n_median": med([x["n"] for x in d1]),
            "d0": d0_by_arm.get(kind),
            # ★ 模型覆盖率（决策级）—— 与 check_model_coverage.py 的 run 级口径互补，
            #   分母不同、不得互相替代。v2 判据的**准入条件**用它。
            "coverage": _coverage_by_arm(rs),
        }
    print()
    print("读法：**D0 可行率**先看（候选有没有被用上）→ **D1 越低越好**（少错误接纳大的 prefill）；")
    print("      **D2 越低越好**（少让请求干等）；**D3 越高越好**。")
    print("      四者与「预测误差」一起看，才构成「误差 → 错误决策 → 服务收益」的链。")
    print("⚠️ D1 的 deadline 取自配置（轨迹没记 budget_ms）、实际时长含 host 间隙 ⇒ 只作**臂间横向比较**。")
    skipped = getattr(load, "skipped", []) or []
    if skipped:
        report["skipped_truncated_products"] = [
            {"dir": n, "why": w} for n, w in skipped]
        print(f"⚠️ 本报告**未包含** {len(skipped)} 个残缺产物（文件被截断，非'无数据'）：")
        for n, w in skipped:
            print(f"     - {n}: {w}")
        print("   ⇒ 分母变了；补齐后须复算。")
    if args.json:
        q = Path(args.json)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                     encoding="utf-8", newline="\n")
        print(f"\n已写出：{q}")
    # ⚠️ 有残缺产物 ⇒ **返回非零**：否则"少了几格"会被下游读成"这批就这么点"。
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
