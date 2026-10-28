#!/usr/bin/env python3
"""**OPT-03 / B4**：小 chunk（prefill 128/192/256）能否被可靠预测？

要回答的问题（逐字来自 `OPTIMIZATION_ROADMAP_SESSIONS.md` §5「B4」）
------------------------------------------------------------------
> 回答：小 chunk 能否被可靠预测（lookup 有键且留出误差达标）？

为什么要有这个脚本（它不是"再跑一遍评估"）
--------------------------------------------
`results/processed/lookup_deploy_eval_v7.json` 的 verdict 是**全形状混算**的
（覆盖 6.67% ⇒ `lookup_holds=false`）。但那个 6.67% 是**被大 prefill 的留出形状拖下去的**：
v7 网格里 `mixed_serving`/`pure_prefill` 的 pf 有 512/1024/2048 共 106 个点，而
`mixed_small_chunk`（pf 128/256）只有 16 个形状。⇒ 混算的覆盖率**不代表小 chunk 的覆盖率**。

本脚本把口径**收到 `mixed_small_chunk` 子集**（v7 网格里唯一服务小 chunk 的组），
分三层回答：

1. **重复性**（会话内）——同一形状在 s1/s2 两次会话的实测差；
   这是"可预测"的**必要条件**：形状测不准就谈不上预测得准。
2. **查表命中**——按部署路径（分桶键）在**留出小 chunk 形状**上是否有键。
3. **留出误差**——命中样本的低估 p95 / |rel| p90（预注册阈值 5% / 10%）。

⚠️ 结构性事实（必须与小 chunk 结论并排读）
--------------------------------------------
v7 的两个会话（s1/s2）是**同一张网格的两次遍历** ⇒ 小 chunk 的 16 个形状
**在两个会话里都有**（键交集 = 全等）⇒ ② 跨会话留出对小 chunk **不构成泛化考验**
（留出的形状训练里有）。因此判据必须以 **① 按形状留出**为准 —— 与小 chunk 无关的
通用结论一致（见 `eval_lookup_deployment_path.py` 的 `_cross_session_shape_overlap`）。

口径（刻意**复用**而不是重写）
-------------------------------
* 键 / 分桶 / 取值规则：import 自 `eval_lookup_deployment_path`
  （`build_lookup_key` / `refit_lookup_table` / `make_predictor` / `measure`）；
* 划分：`split_rank_guaranteed`（同源）；
* 预测器：`CostModel.predict(mode="lookup")`（**部署的那一个**）。
  本脚本**不重抄**查表分支 —— 两份易碎的实现迟早不一致。

产物：`results/processed/b4_small_chunk_v7.json`（**探针分析，不盖 MEASURED**）。

用法::

    python scripts/analyze_b4_small_chunk.py \\
        --eval results/processed/costcalib_eval_v7.json \\
        --model configs/cost_calib/model_v7.json \\
        --json results/processed/b4_small_chunk_v7.json

⚠️ 定位：这是**离线切片分析**，不产生新的真机测量。真机小 chunk 数据来自
`results/raw/_costcalib/costcalib_v7_s{1,2}.json`（v7 网格 `mixed_small_chunk` 组，
已采）。本脚本只回答"这批数据支持什么结论"。
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
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.cost_model import CostModel  # noqa: E402
from eval_lookup_deployment_path import (  # noqa: E402
    ABS_REL_P90_MAX,
    UNDEREST_P95_MAX,
    _fmt,
    build_lookup_key,
    make_predictor,
    measure,
    refit_lookup_table,
    split_rank_guaranteed,
)

#: 小 chunk 的定义 —— 与 v7 网格 `mixed_small_chunk` 组、B4 任务书一致。
SMALL_PREFILL = (128, 192, 256)


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2%}"


#: 小 chunk 键的解析（与 `build_lookup_key` 的拼法同源、同顺序）。
_KEY_RE = re.compile(r"^b(\d+)_pf(\d+)_kv(\d+)$")


def nearest_key_predict(train: list[dict[str, Any]], bucket: dict[str, Any] | None,
                        rec: dict[str, Any]) -> tuple[float | None, str]:
    """未命中时的**最近邻补键**预测：同 batch+pf，取 kv 最近的已建键。

    ⚠️ 这只用于**诊断"无键是不是致命"**，**不是**部署路径（部署路径无键即回退）。
    理由：小 chunk 的 16 个形状在同一 batch+pf 下只有 kv 一个自由度，
    而实测该自由度几乎不影响成本（见 `kv_sensitivity`）⇒ 用它量化"最坏情况下
    补一个键能有多准"。
    返回 (预测值, 说明)；无任何可用键返回 (None, 原因)。
    """
    tab = refit_lookup_table(train, bucket)
    k = build_lookup_key(rec, bucket)
    if k in tab:
        return tab[k]["max"], "exact"
    want_b = int(rec["batch"])
    want_pf = int(rec["prefill_tok"])
    best: float | None = None
    best_d: int | None = None
    for kk, vv in tab.items():
        m = _KEY_RE.match(kk)
        if not m:
            continue
        if int(m.group(1)) == want_b and int(m.group(2)) == want_pf:
            d = abs(int(m.group(3)) - int(rec["kv_len"]))
            if best_d is None or d < best_d:
                best_d, best = d, vv["max"]
    if best is None:
        return None, "no_same_batch_pf_key"
    return best, f"nearest_kv_d{best_d}"


def kv_sensitivity(recs: list[dict[str, Any]], bucket: dict[str, Any] | None) -> dict[str, Any]:
    """同一 batch+pf 下、kv 由 1024→2048 的实测成本变化（小 chunk 是否对 kv 敏感）。

    这决定"最近邻补键"是否可信：若 kv 的影响 ≤ 会话内漂移，补键就是安全的。
    """
    by: dict[tuple[int, int], dict[int, float]] = {}
    for r in recs:
        by.setdefault((int(r["batch"]), int(r["prefill_tok"])), {})[int(r["kv_len"])] = \
            float(r["actual_ms"])
    rows: list[dict[str, Any]] = []
    rels: list[float] = []
    for (b, pf), d in sorted(by.items()):
        kvs = sorted(d)
        if len(kvs) < 2:
            continue
        a, z = d[kvs[0]], d[kvs[-1]]
        rel = abs(z - a) / a
        rels.append(rel)
        rows.append({"batch": b, "prefill_tok": pf, "kv_lo": kvs[0], "kv_hi": kvs[-1],
                     "ms_lo": a, "ms_hi": z, "abs_rel": rel})
    return {
        "n_pairs": len(rows),
        "max_abs_rel": (max(rels) if rels else None),
        "median_abs_rel": (statistics.median(rels) if rels else None),
        "per_pair": rows,
        "note": "同一 batch+pf 下 kv 从最小到最大的实测相对变化。",
    }


def repeatability(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """同一形状跨会话的实测差（相对 s1）。

    这是"可预测"的**必要**条件：若同一形状两次会话就能差几十个百分点，
    查表（取 max）**必然**系统性高估，掉头讨论"留出误差"就没有意义。
    ⚠️ 两会话是同一网格的两次遍历 ⇒ 这里量的是**会话间漂移**，不是泛化。
    """
    by: dict[str, dict[str, float]] = {}
    for r in recs:
        by.setdefault(str(r["key"]), {})[str(r["session"])] = float(r["actual_ms"])
    both = {k: v for k, v in by.items() if len(v) >= 2}
    rels: list[float] = []
    per: list[dict[str, Any]] = []
    for k, v in sorted(both.items()):
        sess = sorted(v)
        a, b = v[sess[0]], v[sess[1]]
        rel = (b - a) / a if a else None
        if rel is not None:
            rels.append(abs(rel))
        per.append({"key": k, "s1_ms": a, "s2_ms": b, "abs_rel": rel,
                    "s1": sess[0], "s2": sess[1]})
    return {
        "n_shapes_both_sessions": len(both),
        "max_abs_rel": (max(rels) if rels else None),
        "median_abs_rel": (statistics.median(rels) if rels else None),
        "per_shape": per,
        "note": ("同一形状跨会话的实测相对差；**不是**泛化（两会话是同网格两次遍历）。"
                 "口径同 COST_PROFILE §6 的会话间漂移。"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OPT-03/B4：小 chunk 可预测性切片分析")
    ap.add_argument("--eval", default="results/processed/costcalib_eval_v7.json")
    ap.add_argument("--model", default="configs/cost_calib/model_v7.json")
    ap.add_argument("--json", default="results/processed/b4_small_chunk_v7.json")
    ap.add_argument("--artifacts", default=None,
                    help="原始标定产物（逗号分隔或 glob）；**血缘字段的权威来源**。"
                         "缺省时只报 None，不拿模型的版本号顶替。")
    args = ap.parse_args(argv)

    ev = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    all_recs = ev.get("records") or []
    if not all_recs:
        print("✗ 评估产物里没有 records", file=sys.stderr)
        return 2

    small = [r for r in all_recs if int(r["prefill_tok"] or 0) in SMALL_PREFILL]
    if not small:
        print("✗ 没有小 chunk 记录", file=sys.stderr)
        return 2

    model_path = Path(args.model)
    cm = CostModel.load_offline(model_path)
    bucket = cm.payload.get("lookup_bucket") or None

    print(f"模型   {model_path.name} | protocol={cm.payload.get('protocol_version')} "
          f"| default_mode={cm.payload.get('default_mode')}")
    print(f"小 chunk 定义 prefill ∈ {SMALL_PREFILL}")
    print(f"总记录 {len(all_recs)} | 小 chunk 记录 {len(small)}"
          f"（{len({r['key'] for r in small})} 形状 × "
          f"{len({r['session'] for r in small})} 会话）")
    print(f"lookup_bucket={bucket} | lookup_value={cm.payload.get('lookup_value')}")

    # ---- 0. 会话内重复性（先说数据可信度）----------------------------------
    rep = repeatability(small)
    print("\n=== 0. 小 chunk 同一形状的跨会话重复性 ===")
    print(f"  两会话都有的形状：{rep['n_shapes_both_sessions']}")
    for p in rep["per_shape"]:
        print(f"    {p['key']:22s} s1={p['s1_ms']:9.3f}  s2={p['s2_ms']:9.3f}  "
              f"|Δ|={_pct(p['abs_rel'])}")
    print(f"  中位 |Δ| {_pct(rep['median_abs_rel'])} / 最大 |Δ| {_pct(rep['max_abs_rel'])}")

    # ---- 0b. kv 敏感度（决定"最近邻补键"是否可信）---------------------------
    kvs = kv_sensitivity(small, bucket)
    print("\n=== 0b. 小 chunk 对 kv 的敏感度（同 batch+pf，kv 由 1024→2048）===")
    for p in kvs["per_pair"]:
        print(f"    b{p['batch']} pf{p['prefill_tok']}: kv{p['kv_lo']}={p['ms_lo']:.2f} "
              f"→ kv{p['kv_hi']}={p['ms_hi']:.2f}  |Δ|={_pct(p['abs_rel'])}")
    print(f"  中位 |Δ| {_pct(kvs['median_abs_rel'])} / 最大 |Δ| {_pct(kvs['max_abs_rel'])}")
    print("  ⇒ kv 影响若 ≤ 会话内漂移，则『同 batch+pf 补一个键』是安全的外推。")

    # ---- 1. 按形状留出（判据主口径）---------------------------------------
    # ⚠️ 划分必须在**全记录**上做（与小 chunk 无关的形状也会占位），
    #    否则"留出比例"与全局不一致；这里沿用 eval 脚本的全记录秩保证划分。
    tr_all, ho_all = split_rank_guaranteed(all_recs, holdout_modulus=3)
    tr_small = [r for r in tr_all if int(r["prefill_tok"] or 0) in SMALL_PREFILL]
    ho_small = [r for r in ho_all if int(r["prefill_tok"] or 0) in SMALL_PREFILL]

    print("\n=== 1. ① 按形状留出（秩保证划分 1/3，全记录上划分）===")
    print(f"  小 chunk：训练 {len(tr_small)} / 留出 {len(ho_small)}")
    ctx: dict[str, Any] = {
        "n_train_small": len(tr_small),
        "n_holdout_small": len(ho_small),
        "holdout_shapes": sorted({r["key"] for r in ho_small}),
        "train_shapes": sorted({r["key"] for r in tr_small}),
    }
    if ho_small:
        # 键**只建在小 chunk 的训练划分**上 ⇒ 这样测的是"小 chunk 自己的键够不够"
        pred_m = make_predictor(cm, tr_small, bucket, apply_margin=True)
        pred_nm = make_predictor(cm, tr_small, bucket, apply_margin=False)
        m_m = measure(ho_small, pred_m)
        m_nm = measure(ho_small, pred_nm)
        keys = refit_lookup_table(tr_small, bucket)
        ctx["holdout_with_margin"] = m_m
        ctx["holdout_no_margin"] = m_nm
        ctx["n_keys_on_small_train"] = len(keys)
        ctx["keys_on_small_train"] = sorted(keys)
        print(f"  小 chunk 训练划分建了 {len(keys)} 个键：{sorted(keys)}")
        print(f"  留出·带余量  {_fmt(m_m)}")
        print(f"  留出·无余量  {_fmt(m_nm)}")
        # 留出里有哪些形状、其键在不在训练里
        hit_keys = {build_lookup_key(r, bucket) for r in ho_small}
        missed = sorted(hit_keys - set(keys))
        ctx["holdout_keys_missing_on_train"] = missed
        print(f"  留出形状的键 {len(hit_keys)} 个；训练里**没有**的 {len(missed)} 个"
              f"：{missed}")
        # ★ 诊断：无键是不是"致命"？用同 batch+pf 的最近邻键补一手，量最坏误差。
        # ⚠️ 这不是部署路径（部署无键即回退）；它回答的是"若给这一族补一个键，够不够准"。
        nn_errs: list[float] = []
        for r in ho_small:
            p, why = nearest_key_predict(tr_small, bucket, r)
            if p is None or float(r["actual_ms"]) <= 0:
                continue
            nn_errs.append((p - float(r["actual_ms"])) / float(r["actual_ms"]))
        if nn_errs:
            def _q95(vals):
                v = sorted(vals)
                return v[min(len(v) - 1, int(0.95 * len(v)))] if v else None
            nn_under = [abs(x) for x in nn_errs if x < 0]
            nn_over = [abs(x) for x in nn_errs if x >= 0]
            ctx["nearest_key_diagnosis"] = {
                "n": len(nn_errs),
                "underest_p95": _q95(nn_under),
                "overest_p95": _q95(nn_over),
                "abs_rel_p90": _q95([abs(x) for x in nn_errs]),
                "note": ("非部署路径：同 batch+pf 取 kv 最近的已建键。"
                         "用于判定『无键是不是致命』，不替代部署判据。"),
            }
            d = ctx["nearest_key_diagnosis"]
            print(f"  [诊断] 最近邻补键（非部署路径）：n={d['n']} "
                  f"低估p95 {_pct(d['underest_p95'])} / 高估p95 {_pct(d['overest_p95'])} "
                  f"/ |rel|p90 {_pct(d['abs_rel_p90'])}")
    else:
        print("  （小 chunk 无留出样本 ⇒ 跳过）")
        ctx["skipped"] = "no small-chunk holdout"

    # ---- 1b. ★ 跨会话留出（同一网格的两次遍历）-----------------------------
    # 这是**小 chunk 最严的泛化考验**：在 s1 上建表、到 s2 上评。
    # ⚠️ 它**不考验"新形状"**（两会话形状相同 ⇒ 键全等），但它是**唯一**能把
    #    "键存在 + 精度"两件事同时验到 100% 覆盖的口径；且它的漂移量正是
    #    COST_PROFILE §6 要报的会话间漂移。**两种口径必须并排读**。
    sessions = sorted({str(r["session"]) for r in small})
    print("\n=== 1b. ★ 跨会话留出（小 chunk：在 s1 建表、评 s2）===")
    xs: dict[str, Any] = {"n_sessions": len(sessions), "sessions": sessions}
    if len(sessions) >= 2:
        s_tr = [r for r in small if str(r["session"]) == sessions[0]]
        s_ho = [r for r in small if str(r["session"]) != sessions[0]]
        pred_x = make_predictor(cm, s_tr, bucket, apply_margin=False)
        m_x = measure(s_ho, pred_x)
        xs["holdout_no_margin"] = m_x
        xs["n_train"] = len(s_tr)
        xs["n_holdout"] = len(s_ho)
        print(f"  {sessions[0]} 建表 {len(s_tr)} 条 → 评 {len(s_ho)} 条")
        print(f"  跨会话留出·无余量  {_fmt(m_x)}")
        # 按 pf 分开报（B4 的核心：192 与 128/256 是否同级）
        per_pf: dict[str, Any] = {}
        for pf in sorted({int(r["prefill_tok"]) for r in s_ho}):
            sub = [r for r in s_ho if int(r["prefill_tok"] or 0) == pf]
            mm = measure(sub, pred_x)
            per_pf[str(pf)] = mm
            print(f"    pf{pf:4d}: {_fmt(mm)}")
        xs["per_prefill"] = per_pf
        xs["keys_on_train"] = sorted(refit_lookup_table(s_tr, bucket))
    else:
        print(f"  （只有 {len(sessions)} 个会话 ⇒ 无法做跨会话留出）")
    ctx["cross_session"] = xs

    # ---- 2. ★ 部署路径 vs 小 chunk 的实际判据 ------------------------------
    # 判据口径：小 chunk 形状**必须**在部署表（全部记录建表）里有键，
    # 否则运行时恒回退 —— 那正是"查表没键就等于不能预测"。
    #
    # ⚠️ 两种表都要看，且要**分清**：
    #   ① `model.payload.lookup_table` = **当前部署模型**的表（可能还是旧协议，
    #      例如 v8 数据采了、但模型还没导出 ⇒ 它当然缺 192 的键）；
    #   ② **本产物数据自己建的表** = 若这批数据被导出成模型，部署表会是什么。
    #      只有 ② 能回答"这批新采样点支持不支持预测"。
    deployed_keys = cm.payload.get("lookup_table") or {}
    own_keys = refit_lookup_table(all_recs, bucket)
    small_deployed_keys = sorted(k for k in own_keys
                                 if "_pf128_" in k or "_pf192_" in k or "_pf256_" in k)
    print("\n=== 2. 小 chunk 键（两种表分开看）===")
    print(f"  ① 当前部署模型 {cm.payload.get('protocol_version')} 的表："
          f"总 {len(deployed_keys)} 键；"
          f"其中小 chunk {len([k for k in deployed_keys if '_pf128_' in k or '_pf192_' in k or '_pf256_' in k])} 键")
    print(f"  ② **本批数据自己建的表**：总 {len(own_keys)} 键；小 chunk {len(small_deployed_keys)} 键")
    for k in small_deployed_keys:
        v = own_keys[k]
        print(f"    {k:22s} max={v.get('max'):9.3f} median={v.get('median'):9.3f} n={v.get('n')}")
    ctx["deployed_table_n_keys_total"] = len(deployed_keys)
    ctx["deployed_table_n_keys_small"] = len([k for k in deployed_keys
                                              if "_pf128_" in k or "_pf192_" in k or "_pf256_" in k])
    ctx["own_table_n_keys_total"] = len(own_keys)
    ctx["own_table_n_keys_small"] = len(small_deployed_keys)
    ctx["own_table_keys_small"] = small_deployed_keys

    # 192 的缺口（任务书要 128/192/256）
    pf_axis = sorted({int(r["prefill_tok"] or 0) for r in all_recs})
    n192_own = len([k for k in own_keys if "_pf192_" in k])
    has192 = n192_own > 0
    ctx["prefill_axis_in_data"] = pf_axis
    ctx["pf192_present"] = has192
    ctx["pf192_n_keys_own_table"] = n192_own
    print(f"  数据里的 prefill 轴 {pf_axis}；"
          f"本批数据建表后 pf192 键数 = **{n192_own}**"
          f"{'' if has192 else '  ⇒ 192 没有样本 ⇒ 无法预测（不是查表未命中，是没测）'}")

    # ---- 3. 结论 ----------------------------------------------------------
    print("\n=== 3. 结论（三层，分别给）===")
    concl: dict[str, Any] = {}
    # 层 1：重复性
    med = rep["median_abs_rel"]
    concl["repeatability_ok"] = (med is not None and med <= 0.05)
    print(f"  层1 重复性（会话内）：中位 |Δ| {_pct(med)} ⇒ "
          f"{'合格（≤5%）' if concl['repeatability_ok'] else '不合格（>5%）'}")
    # 层 2：128/192/256 的键存在性 —— 用**本批数据自己建的表**（那才是"这批采样点
    #       被导出成模型后"的部署表）。⚠️ 不去看旧部署模型：它可能还没重导出。
    n128 = len([k for k in own_keys if "_pf128_" in k])
    n256 = len([k for k in own_keys if "_pf256_" in k])
    concl["keys_pf128"] = n128
    concl["keys_pf256"] = n256
    concl["keys_pf192"] = n192_own
    print(f"  层2 键存在性（本批数据建表）：pf128 {n128} 键 / pf256 {n256} 键 / "
          f"pf192 {n192_own} 键")
    # 层 2b：本批表对**已采**小 chunk 形状的命中
    own_cov = None
    if small:
        n_hit_own = sum(1 for r in small if build_lookup_key(r, bucket) in own_keys)
        own_cov = n_hit_own / len(small)
        concl["own_coverage_on_sampled_small"] = own_cov
        print(f"  层2b 本批表对**已采**小 chunk 形状的命中：{n_hit_own}/{len(small)} "
              f"= {_pct(own_cov)}")
    # 层 3：按形状留出（部署路径 = 无键即回退）
    if ho_small:
        m_m = ctx["holdout_with_margin"]
        cvg = m_m["coverage"]
        up = m_m["underest_p95"]
        ar = m_m["abs_rel_p90"]
        concl["holdout_small_coverage"] = cvg
        concl["holdout_small_underest_p95"] = up
        concl["holdout_small_abs_rel_p90"] = ar
        ok_err = (up is not None and up <= UNDEREST_P95_MAX
                  and ar is not None and ar <= ABS_REL_P90_MAX)
        print(f"  层3a 按形状留出（部署路径=无键回退）：覆盖 {_pct(cvg)} / "
              f"低估p95 {_pct(up)} / |rel|p90 {_pct(ar)} ⇒ "
              f"{'达标' if ok_err else '不达标'}")
        print(f"      ⚠️ 这 {_pct(cvg)} 是**秩保证划分按 hash 切出的『新形状』**造成的，"
              f"不等于『小 chunk 不可预测』——")
        print(f"      同 batch+pf 补一个键后（见上方诊断），误差降到 "
              f"{_pct((ctx.get('nearest_key_diagnosis') or {}).get('abs_rel_p90'))}。")
        concl["holdout_error_ok"] = ok_err
    # 层 3b：★ 跨会话留出（严格口径：键全等 ⇒ 100% 覆盖；报精度与会话漂移）
    xs = ctx.get("cross_session") or {}
    if xs.get("holdout_no_margin"):
        mx = xs["holdout_no_margin"]
        xs_cov, xs_up, xs_ar = mx["coverage"], mx["underest_p95"], mx["abs_rel_p90"]
        xs_ok = (xs_up is not None and xs_up <= UNDEREST_P95_MAX
                 and xs_ar is not None and xs_ar <= ABS_REL_P90_MAX)
        concl["xsession_coverage"] = xs_cov
        concl["xsession_underest_p95"] = xs_up
        concl["xsession_abs_rel_p90"] = xs_ar
        concl["xsession_ok"] = xs_ok
        print(f"  层3b ★跨会话留出（键全等 ⇒ 100% 覆盖）：覆盖 {_pct(xs_cov)} / "
              f"低估p95 {_pct(xs_up)} / |rel|p90 {_pct(xs_ar)} ⇒ "
              f"{'达标' if xs_ok else '不达标'}")
        for pf, mm in (xs.get("per_prefill") or {}).items():
            print(f"      pf{pf:>4s}: 覆盖 {_pct(mm['coverage'])} / "
                  f"低估p95 {_pct(mm['underest_p95'])} / |rel|p90 {_pct(mm['abs_rel_p90'])}")
    # 综合判定（三态：能 / 部分 / 不能）
    # 判据（**三件都要成立**）：① 形状内可重复；② 本批表 100% 命中已采形状；
    #                            ③ 最严的留出误差达标（优先跨会话口径）
    nn_ok = None
    if ctx.get("nearest_key_diagnosis"):
        nn_ok = (ctx["nearest_key_diagnosis"]["abs_rel_p90"] is not None
                 and ctx["nearest_key_diagnosis"]["abs_rel_p90"] <= ABS_REL_P90_MAX)
    concl["nearest_key_ok"] = nn_ok
    pf_complete = (concl["keys_pf128"] > 0 and concl["keys_pf192"] > 0
                   and concl["keys_pf256"] > 0)
    err_ok = concl.get("xsession_ok") or concl.get("holdout_error_ok")
    if concl["repeatability_ok"] and (own_cov or 0) >= 1.0 and pf_complete and err_ok:
        verdict = ("能（128/192/256 齐、本批表 100% 命中、会话内重复性 "
                   f"{_pct(rep['median_abs_rel'])}，最严留出低估 p95 "
                   f"{_pct(concl.get('xsession_underest_p95'))}）")
    elif concl["repeatability_ok"] and (own_cov or 0) >= 1.0:
        missing = [p for p, n in (("128", n128), ("192", n192_own), ("256", n256))
                   if n == 0]
        verdict = (f"部分（已采形状 100% 命中且可重复，但 pf{'/'.join(missing)} 无样本 "
                   "⇒ 那几档不可预测）")
    else:
        verdict = "不能（见三层明细）"
    concl["verdict"] = verdict
    print(f"\n  ★ 综合判定：**{verdict}**")
    print("  ⚠️ 口径提醒：全文的『覆盖率』有三种 —— ① 本批表对**已采形状**的命中"
          f"（{_pct(own_cov)}），② **按形状留出**的新形状命中"
          f"（{_pct((ctx or {}).get('holdout_with_margin', {}).get('coverage'))} + 最近邻诊断），"
          f"③ **跨会话留出**的命中（{_pct(concl.get('xsession_coverage'))}）。"
          "\n     全形状混算的 6.67%（`lookup_deploy_eval_v7.json`）**三者都不是**"
          "（被大 prefill 留出形状拖低），四者不可互换。")

    # ⚠️ 血缘字段必须来自**原始标定产物**，不是模型文件、也不是 eval 产物 ——
    #   实测：`costcalib_eval_*.json` **不含** protocol_version / grid_sha256，
    #   而模型文件的版本可能比数据旧（v8 数据配 v7 模型）。
    #   ⇒ 从 --artifacts 现读；不给就如实为 None，**绝不拿模型的值顶替**。
    import glob as _glob
    raw_protocols: set[str] = set()
    raw_grids: set[str] = set()
    raw_sessions: set[str] = set()
    raw_paths: list[str] = []
    if args.artifacts:
        pats = [x.strip() for x in args.artifacts.split(",") if x.strip()]
        files: list[str] = []
        for pat in pats:
            files.extend(sorted(_glob.glob(pat)) or [pat])
        for f in files:
            try:
                a = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception as exc:      # 缺文件/坏 JSON 都要显式说，不能静默跳过
                print(f"⚠️ 读不到原始产物 {f}: {exc}", file=sys.stderr)
                continue
            raw_paths.append(f)
            if a.get("protocol_version"):
                raw_protocols.add(str(a["protocol_version"]))
            if a.get("grid_sha256"):
                raw_grids.add(str(a["grid_sha256"]))
            if a.get("session_id"):
                raw_sessions.add(str(a["session_id"]))
    data_protocol = "/".join(sorted(raw_protocols)) if raw_protocols else None
    data_grid = "/".join(sorted(raw_grids)) if raw_grids else None
    print(f"\n血缘（来自原始标定产物 {len(raw_paths)} 份）：")
    print(f"  protocol_version = {data_protocol}")
    print(f"  grid_sha256      = {data_grid}")
    print(f"  sessions         = {sorted(raw_sessions) or None}")

    out = {
        "script": "analyze_b4_small_chunk.py",
        "session": "OPT-03 / B4",
        "eval_source": args.eval,
        "model": str(model_path),
        # --- 数据血缘（权威：来自 --artifacts 的原始标定产物；缺则 None）---
        "data_protocol_version": data_protocol,
        "data_grid_sha256": data_grid,
        "data_sessions": sorted(raw_sessions) or None,
        "data_artifact_paths": raw_paths,
        "data_sessions_in_eval": sorted({str(r["session"]) for r in all_recs}),
        # --- 被借用来做"部署路径"实现的模型（其 protocol 可能比数据旧）---
        "predictor_model_protocol_version": cm.payload.get("protocol_version"),
        "predictor_model_grid_sha256": cm.payload.get("grid_sha256"),
        "n_records_total": len(all_recs),
        "n_records_small": len(small),
        "small_prefill": list(SMALL_PREFILL),
        "lookup_bucket": bucket,
        "repeatability": rep,
        "shape_holdout": ctx,
        "conclusion": concl,
        "label": "NOT_MEASURED（探针切片分析；不盖四标签、不进 EXPERIMENTS.md 结论表）",
        "note": (f"数据源 = {args.eval}（探针切片分析；真机标定产物见 results/raw/_costcalib/）；"
                 "本脚本只做切片与判据计算，不产生新测量。"
                 "覆盖率与误差**并排报**；未命中一律计入回退、不填 0。"
                 "⚠️ `data_*` 字段描述**这批数据**，`predictor_model_*` 描述**借来做预测实现的"
                 "那份模型**（两者协议版本可以不同）。"),
    }
    p = Path(args.json)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(f"\n已写出 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
