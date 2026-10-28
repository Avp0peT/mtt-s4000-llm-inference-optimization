#!/usr/bin/env python3
"""远端只读探针：对 S14B2 逐 run 汇总关键量 + 跨重复一致性 + 身份校验。

用途：批次还在跑时，**远端**已有全部产物而本地只有少数。
本探针在远端跑，**只读**（不写、不删、不改标签），把结果打到 stdout。

为什么要它（而不是等 pull 完再跑 analyze_step14）
------------------------------------------------
* pull 在批次跑完前**不可能**凑够 20（远端也不够）⇒ 干等没有信息增量；
* 该探针只回答"数据可用性 + 身份 + 跨重复离散"三问，
  **不**替代 `analyze_step14.py` 的口径（后者仍是登记的唯一来源）。

三条纪律（与 analyze_step14 一致）
* 报**全部**重复，不给"最好一次"；
* 跨重复离散用**稳健 CV**（1.4826·MAD/中位），抗单次离群；
* 缺字段显形为 None，**不填 0**。
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st

#: 远端默认路径；可用 ``AQ_ROOT`` 覆盖以便**离线试跑**（本地指向仓库根）。
ROOT = os.environ.get("AQ_ROOT", "/root/ascend_probe_d1")
RAW = os.path.join(ROOT, "results", "raw")
TAG = os.environ.get("AQ_TAG", "S14B2")
_STAGING = (".tar", ".tar.gz", ".tgz")


def rob_cv(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    med = st.median(xs)
    if med == 0:
        return None
    mad = st.median([abs(x - med) for x in xs])
    return 1.4826 * mad / med


def q(sub, key):
    if not isinstance(sub, dict):
        return None
    return sub.get(key)


rows = []
for d in sorted(glob.glob(f"{RAW}/{TAG}-*")):
    if not os.path.isdir(d) or d.lower().endswith(_STAGING):
        continue
    mp = os.path.join(d, "meta.json")
    if not os.path.exists(mp) or os.path.getsize(mp) == 0:
        continue                      # 目录在写 / 空 ⇒ 不算产物
    try:
        m = json.load(open(mp, encoding="utf-8"))
    except Exception as e:            # noqa: BLE001
        rows.append({"dir": os.path.basename(d), "error": f"meta 不可解析: {e}"})
        continue
    s = m.get("summary") or {}
    cfg = (m.get("config") or {})
    ctl = cfg.get("controller") or {}
    rep = m.get("reproducibility") or {}
    rows.append({
        "dir": os.path.basename(d),
        "run_id": (m.get("context") or {}).get("run_id") or s.get("run_id"),
        "pass": (m.get("context") or {}).get("repeat_index"),
        "cell": cfg.get("config_stem"),
        "cfg_path": cfg.get("config_path"),
        "workload": cfg.get("workload"),
        "label": m.get("data_label"),
        # ⚠️ 控制器**停用**时（enabled=false），meta 里的 kind/signal_criterion 是**默认值，
        #    不生效**，且与 config.snapshot.yaml 口径不一致（快照正确地不写它们）。
        #    直接展示会把人引向"判据错配"的误判（曾发生过，见 2026-09-28.md §21）。
        #    ⇒ 一律随 enabled 一起展示；停用时把值折成 "(off)"。
        "enabled": ctl.get("enabled"),
        "criterion": (ctl.get("signal_criterion") if ctl.get("enabled") else "(off)"),
        "kind": (ctl.get("kind") if ctl.get("enabled") else "(off)"),
        "cost_model": ctl.get("cost_model_path"),
        "commit": rep.get("git_commit"),
        "dirty": rep.get("git_dirty"),
        "env": rep.get("execution_environment"),
        "model": rep.get("model"),
        "device": rep.get("device_model"),
        "ttft_p50": q(s.get("ttft"), "p50"),
        "ttft_p95": q(s.get("ttft"), "p95"),
        "ttft_p99": q(s.get("ttft"), "p99"),
        "tpot_p50": q(s.get("tpot"), "p50"),
        "tpot_p99": q(s.get("tpot"), "p99"),
        "stall_p99": q(s.get("stall"), "p99"),
        "goodput_tok_s": s.get("slo_goodput_tok_s"),
        "sat_rate": s.get("slo_satisfaction_rate"),
        "throughput": s.get("throughput_tok_s"),
        "unfinished": s.get("unfinished_rate"),
        "failure": s.get("failure_rate"),
        "rob_cv_ttft": q(s.get("stability") or {}, "ttft_robust_cv"),
        "arm_state": (m.get("controller_effect") or {}).get("arm_state"),
    })

print(f"=== {TAG}: 可用产物 {len(rows)} 个 ===")
print()
hdr = ("dir", "pass", "label", "cell", "criterion", "ttft_p50", "tpot_p50",
       "stall_p99", "goodput", "sat", "unfin")
print(" | ".join(f"{h:>26}" if h == "dir" else f"{h:>10}" for h in hdr))
for r in rows:
    if "error" in r:
        print(f"{r['dir']:>26} | ERROR {r['error']}")
        continue
    def f(v, nd=2):
        return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
    print(" | ".join([
        f"{r['dir']:>26}", f"{f(r['pass'],0):>10}", f"{str(r['label']):>10}",
        f"{str(r['cell']):>10}", f"{str(r['criterion']):>10}",
        f"{f(r['ttft_p50'],1):>10}", f"{f(r['tpot_p50'],1):>10}",
        f"{f(r['stall_p99'],1):>10}", f"{f(r['goodput_tok_s'],3):>10}",
        f"{f(r['sat_rate'],3):>10}", f"{f(r['unfinished'],3):>10}",
    ]))

# ---- 身份一致性：同 cell 内 commit / env / model / cost_model / criterion 必须唯一 ----
print()
print("=== 身份一致性（同 cell 内应各自唯一）===")
by_cell: dict[str, list[dict]] = {}
for r in rows:
    if "error" in r:
        continue
    by_cell.setdefault(str(r["cell"]), []).append(r)
for cell, rs in sorted(by_cell.items()):
    for field in ("commit", "env", "model", "device", "cost_model", "criterion", "kind"):
        vals = {str(r[field]) for r in rs}
        mark = "OK " if len(vals) == 1 else "⚠️ "
        if len(vals) != 1:
            print(f"{mark}{cell}.{field} 有 {len(vals)} 种: {sorted(vals)}")
    n_lab = sum(1 for r in rs if r["label"] == "MEASURED")
    print(f"OK  {cell}: n={len(rs)} MEASURED={n_lab} "
          f"commit={ {str(r['commit'])[:8] for r in rs} } "
          f"cost_model={ {str(r['cost_model']).split('/')[-1] for r in rs} }")

print()
print("=== 跨重复离散（稳健 CV = 1.4826·MAD/中位；同 cell 内）===")
for cell, rs in sorted(by_cell.items()):
    for key in ("ttft_p50", "tpot_p50", "stall_p99", "goodput_tok_s", "sat_rate"):
        vals = [r[key] for r in rs]
        if any(v is None for v in vals):
            continue
        cv = rob_cv(vals)
        med = st.median(vals)
        flag = ""
        if cv is not None and cv > 0.10:
            flag = "  ⚠️ >10%（§7.3 需先修测量）"
        print(f"{cell:<12} {key:<14} n={len(vals)} 中位={med:>10.3f} "
              f"极差={max(vals)-min(vals):>9.3f} 稳健CV={('%.4f' % cv) if cv is not None else '-'}{flag}")

# ---- 回退/降级计数（cost 臂：读 arm_state；缺则显形 None，不填 0）----
print()
print("=== 回退/降级计数（arm_state；缺 = None，不填 0）===")
for r in rows:
    if "error" in r:
        continue
    stt = r["arm_state"]
    if stt:
        print(f"{r['dir']:<32} kind={r['kind']} n_fallback={stt.get('n_fallback')} "
              f"n_degraded={stt.get('n_degraded')} n_starved={stt.get('n_prefill_starved')} "
              f"n_idle={stt.get('n_idle')}")
    else:
        print(f"{r['dir']:<32} kind={r['kind']} arm_state=None")
