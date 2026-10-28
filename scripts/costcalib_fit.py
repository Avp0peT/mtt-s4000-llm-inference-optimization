#!/usr/bin/env python3
"""成本模型的**拟合与评估**（第 11 步，协议 `cost-calib-v1`）。

它做三件事，顺序不能换
----------------------
1. **划分**（划分规则必须与数据值无关 —— 见下）；
2. 在训练划分上**拟合**两个候选模型（简单查表 / 分段线性）；
3. 按**预注册的误差口径**在留出划分上报，并按 §五 的判据回答"模型成立与否"。

关于划分：为什么用"按 key 的哈希取 1/4"而不是"挑几个形状"
----------------------------------------------------------
在**已经看到数据**之后再挑留出形状，等于挑最容易/最难的那几个 ⇒ 留出误差可以被随意做大做小。
所以这里用一个**与数据值完全无关**的确定性规则：`sha256(key) % 4 == 0` 的进入 `hash` 留出集。
另外再留一个**外推**留出集：所有 `kv_len == 2048` 的点（测越出训练范围时的行为）。
两层都报，且**分开报**。

⚠️ 被拟合的量是**块 B**（服务协议）；块 A 只作下界参考（预注册 §二 + 验收第 2 条）。

用法::

    python scripts/costcalib_fit.py --artifacts results/raw/_costcalib/*.json --json results/processed/costcalib_eval.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: 预注册 §五 的"模型不成立"判据（**看到数据之前定的**，这里只引用，不改）。
UNDEREST_P95_MAX = 0.05      # 低估误差的相对 p95 上限
CROSS_SESSION_DRIFT_MAX = 0.10   # 跨会话漂移上限（预注册 §五）
#: ⚠️ **事后新增**（不在预注册里）：整体精度上限。
#:
#: 为什么必须补：预注册的两条只约束了"不要低估"与"跨会话别漂"。实测发现一个
#: **总是高估 400%** 的模型能同时满足那两条 —— 它从不低估、也不漂。
#: 那不是"成立"，那是"不可用"。⇒ 补一条 |相对误差| p90 的约束。
#: 取值理由（现在写下）：SLO 是逐请求 TPOT ≤ 100 ms，而单步成本 60–90 ms；
#: ±10% 的步成本误差 ≈ ±6–9 ms，再大就会让"选哪个 chunk"的决策翻转。
ABS_REL_P90_MAX = 0.10


# --------------------------------------------------------------------------- #
# 纯逻辑：记录装配、划分、拟合、评估
# --------------------------------------------------------------------------- #


def record_of(point: dict[str, Any], session: str) -> dict[str, Any] | None:
    """把产物里的一个点摊平成一条记录。**只取块 B**（被拟合的量）。"""
    if "error" in point:
        return None
    b = (point.get("B") or {})
    if b.get("p50_ms") is None:
        return None
    sh = point.get("shape") or {}
    return {
        "session": session, "key": point.get("key"),
        "batch": sh.get("batch"), "prefill_tok": sh.get("prefill_tok"),
        "kv_len": sh.get("kv_len"), "n_forwards": sh.get("n_forwards"),
        "actual_ms": b["p50_ms"],                     # 服务协议下的每步成本（p50）
        "actual_p90_ms": b.get("p90_ms"),
        "block_a_ms": (point.get("A") or {}).get("platform_min_ms"),
    }


def load_records(paths: Sequence[str]) -> tuple[list[dict[str, Any]], list[str]]:
    recs: list[dict[str, Any]] = []
    problems: list[str] = []
    for p in paths:
        art = json.loads(Path(p).read_text(encoding="utf-8"))
        sid = art.get("session_id") or Path(p).stem
        if not art.get("complete"):
            problems.append(f"{Path(p).name}: complete=False（采集被打断）⇒ 仍收，但报告里注明")
        for pt in art.get("points") or []:
            r = record_of(pt, sid)
            if r:
                recs.append(r)
    if not recs:
        problems.append("没有任何可用记录")
    # 同一 (session,key) 重复 ⇒ 只留第一条（产物是增量写的，可能被读过两次）
    seen = set()
    uniq = []
    for r in recs:
        k = (r["session"], r["key"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    return uniq, problems


def split_rank_guaranteed(recs: list[dict[str, Any]], holdout_modulus: int = 3
                          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """**秩保证划分**（第 14 步归因审计的产物）。

    为什么要有它：哈希划分对**训练覆盖**是一次抽签。实测踩到过 ——
    `1-in-2` 下 `batch=2` 的训练集只剩纯 decode（`prefill_tok` 全 0）⇒ 秩降到 1 ⇒
    **prefill 系数无法学习** ⇒ 一个实测 **1857.9 ms** 的混合形状被预测成 **88.1 ms**（低估 95.26%）。
    换成"每段先凑满秩、其余再留出"后，**同等规模下 95.26% → 11.77%**。

    规则（**必须在看误差之前固定**）：
    每段按 `sha256(key)` 稳定排序，**贪心取到设计矩阵秩满**为止放进训练集，
    其余按 `sha256(key) % holdout_modulus == 0` 留出。
    """
    tr: list[dict[str, Any]] = []
    ho: list[dict[str, Any]] = []
    for b in sorted({int(r["batch"] or 1) for r in recs}):
        rows = [r for r in recs if int(r["batch"] or 1) == b]
        ordered = sorted(rows, key=lambda r: hashlib.sha256(str(r["key"]).encode()).hexdigest())
        core: list[dict[str, Any]] = []
        rest = list(ordered)
        while rest:
            cand = core + [rest[0]]
            core.append(rest.pop(0))
            if _rank(cand) >= 3:                      # 3 = 该段模型的参数个数
                break
        for r in rest:
            v = int(hashlib.sha256(str(r["key"]).encode()).hexdigest()[:8], 16) % holdout_modulus
            (ho if v == 0 else tr).append(r)
        tr.extend(core)
    return tr, ho


def _rank(rows: list[dict[str, Any]]) -> int:
    """该段训练集设计矩阵的秩（`[1, prefill_tok, kv_len]`）。"""
    if not rows:
        return 0
    import numpy as np          # 本文件的既有风格：numpy 在函数内局部导入
    A = np.asarray([[1.0, float(r["prefill_tok"]), float(r["kv_len"])] for r in rows], float)
    return int(np.linalg.matrix_rank(A))


def diagnose_coverage(rows: list[dict[str, Any]], label: str) -> list[str]:
    """逐段报**秩 / prefill 取值数 / kv 取值数**，并把"秩不足"的段点名。

    ⚠️ 没有这条诊断，就还会再抽到一次 95% 而不自知 —— 误差本身不告诉你为什么。
    """
    problems: list[str] = []
    for b in sorted({int(r["batch"] or 1) for r in rows}):
        seg = [r for r in rows if int(r["batch"] or 1) == b]
        rk = _rank(seg)
        pfs = sorted({int(r["prefill_tok"]) for r in seg})
        kvs = sorted({int(r["kv_len"]) for r in seg})
        # ⚠️ 训练集与留出集的"秩不足"**含义不同**，不能报成同一件事：
        #   * train 秩不足 ⇒ **模型学不到那个方向**（本次 95% 误差的成因）；
        #   * holdout 秩不足 ⇒ **评估覆盖不足**（留出集探不到那些方向 ⇒ 误差数字偏乐观）。
        if rk >= 3:
            flag = ""
        elif label.startswith("train"):
            flag = "   ⚠️ **秩不足 ⇒ 该段不可辨识（模型学不到）**"
        else:
            flag = "   ℹ️ 留出集秩不足 ⇒ **评估覆盖不足**（不是模型的问题）"
        print(f"    [{label}] batch={b:<2} n={len(seg):<3} 秩={rk} "
              f"prefill={pfs} kv={kvs}{flag}")
        if rk < 3:
            kind = "训练集不可辨识" if label.startswith("train") else "留出评估覆盖不足"
            problems.append(f"{kind}：{label} batch={b} 秩={rk}（参数需 3）prefill={pfs} kv={kvs}")
    return problems


def split_hash(rec: dict[str, Any], modulus: int = 4) -> str:
    """**与数据值无关**的确定性划分：按 key 的哈希取 1/modulus 留出。"""
    h = hashlib.sha256(str(rec["key"]).encode("utf-8")).hexdigest()
    return "hash_holdout" if int(h[:8], 16) % modulus == 0 else "train"


def _design(rec: dict[str, Any]) -> list[float]:
    """分段模型的基函数。

    选这几项的理由（**看数据之前**按结构定的，不是为了拟合好看）：
    * `prefill_tok` —— 纯 prefill 的成本主体（实测近线性）；
    * `batch` —— 每条序列的固定开销；
    * `batch * (kv_len + 1)` —— attention 的工作量随 batch 与 KV 同时增长；
    * `(n_forwards - 1)` —— 串行前向组的加和项（若加和性成立，这一项应当**不显著**，
      它是那条结论的一个**可检验**形式）。
    """
    b = int(rec["batch"] or 1)
    pf = int(rec["prefill_tok"] or 0)
    kv = int(rec["kv_len"] or 0)
    nf = int(rec["n_forwards"] or 1)
    return [1.0, float(pf), float(b), float(b) * (kv + 1), float(nf - 1)]


def _ols(xs: list[list[float]], ys: list[float]) -> list[float]:
    """最小二乘（用 numpy 若可用，否则手写高斯消元）。"""
    try:
        import numpy as np
        A = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        return [float(c) for c in coef]
    except Exception:
        n = len(xs[0])
        A = [[sum(xs[k][i] * xs[k][j] for k in range(len(xs))) for j in range(n)]
             + [sum(xs[k][i] * ys[k] for k in range(len(xs)))] for i in range(n)]
        for c in range(n):                                   # 高斯消元
            piv = max(range(c, n), key=lambda r: abs(A[r][c]))
            A[c], A[piv] = A[piv], A[c]
            if abs(A[c][c]) < 1e-12:
                continue
            for r in range(n):
                if r != c and A[r][c]:
                    f = A[r][c] / A[c][c]
                    for j in range(c, n + 1):
                        A[r][j] -= f * A[c][j]
        return [(A[i][n] / A[i][i]) if abs(A[i][i]) > 1e-12 else 0.0 for i in range(n)]


def fit_segmented(train: list[dict[str, Any]]) -> list[float]:
    return _ols([_design(r) for r in train], [r["actual_ms"] for r in train])


def fit_piecewise(train: list[dict[str, Any]]) -> dict[int, list[float]]:
    """**真正的"分段模型"**：按 `batch` 分段，每段内对 `(prefill_tok, kv_len)` 线性拟合。

    ⚠️ 分段变量（`batch`）是**看到数据之后**选的，登记为 **post-hoc**：
    实测 b1→b2 有 +29% 的跳变，而 b2→b4 在 kv512 上几乎不动（66.12→66.24），
    单一超平面在结构上拟合不了这种形状。**不得**把这个选择说成"预先登记的"。
    """
    by_b: dict[int, list[dict[str, Any]]] = {}
    for r in train:
        by_b.setdefault(int(r["batch"] or 1), []).append(r)
    return {
        b: _ols([[1.0, float(r["prefill_tok"]), float(r["kv_len"])] for r in rows],
                [r["actual_ms"] for r in rows])
        for b, rows in by_b.items()
    }


def predict_piecewise(models: dict[int, list[float]],
                      rec: dict[str, Any]) -> float | None:
    """没见过的 `batch` **不猜**（返回 `None`）—— 它就是"范围外"。"""
    m = models.get(int(rec["batch"] or 1))
    if m is None:
        return None
    return m[0] + m[1] * float(rec["prefill_tok"] or 0) + m[2] * float(rec["kv_len"] or 0)


def predict_segmented(coef: list[float], rec: dict[str, Any]) -> float:
    return sum(c * x for c, x in zip(coef, _design(rec)))


def fit_lookup(train: list[dict[str, Any]]) -> dict[tuple, float]:
    per: dict[tuple, list[float]] = {}
    for r in train:
        per.setdefault((r["batch"], r["prefill_tok"], r["kv_len"], r["n_forwards"]), []).append(
            r["actual_ms"])
    return {k: statistics.median(v) for k, v in per.items()}


def predict_lookup(table: dict[tuple, float], rec: dict[str, Any]) -> float | None:
    """查表：**精确命中才有值**；未命中返回 `None`（不猜、不外推）。

    这是"简单查表"模型的诚实形式：它的**覆盖率**是一条要报的结论，
    而不是靠最近邻把没测过的形状"补"上一个数。
    """
    return table.get((rec["batch"], rec["prefill_tok"], rec["kv_len"], rec["n_forwards"]))


def predict_hybrid(table: dict[tuple, float], coef: list[float],
                   rec: dict[str, Any]) -> float:
    """**部署形态**：查表覆盖内用查表（近乎无误差），未覆盖退回分段模型。

    为什么要合成一个"混合"而不是把两个模型各报一行就完事：真正会被部署的是**一个**
    模型，而它的误差才是"模型成立与否"该问的东西。分开报会掩盖"未覆盖时退回了什么"。
    """
    v = predict_lookup(table, rec)
    return v if v is not None else predict_segmented(coef, rec)


def errors(recs: list[dict[str, Any]], predictor) -> dict[str, Any]:
    """按预注册口径算误差。**低估与高估分开报**（合并成"平均误差"会盖住危险侧）。"""
    rel: list[float] = []          # (pred - actual)/actual，带符号
    under: list[float] = []        # 只留 pred < actual 的相对缺口（正数）
    n_covered = 0
    for r in recs:
        p = predictor(r)
        if p is None:
            continue
        n_covered += 1
        a = r["actual_ms"]
        d = (p - a) / a
        rel.append(d)
        if d < 0:
            under.append(-d)
    if not rel:
        return {"n": 0, "n_covered": 0, "coverage": 0.0, "rel_p50": None, "rel_p90": None,
                "rel_p99": None, "underest_rate": None, "underest_p95": None}

    def q(v, frac):
        s = sorted(v)
        return s[min(len(s) - 1, int(frac * len(s)))]

    return {
        "n": len(recs), "n_covered": n_covered, "coverage": n_covered / len(recs),
        "abs_rel_p50": q([abs(x) for x in rel], 0.5),
        "abs_rel_p90": q([abs(x) for x in rel], 0.9),
        "abs_rel_p99": q([abs(x) for x in rel], 0.99),
        "underest_rate": len(under) / len(rel),
        "underest_p95": (q(under, 0.95) if under else 0.0),
        "worst_underest": (max(under) if under else 0.0),
    }


def cross_session_drift(recs: list[dict[str, Any]], predictor) -> dict[str, Any]:
    """跨会话漂移：**同一形状**在不同会话上的实测差 / 会话内极差。"""
    by_key: dict[tuple, list[float]] = {}
    for r in recs:
        by_key.setdefault((r["key"],), []).append(r["actual_ms"])
    rel = []
    for (_, ), vals in by_key.items():
        if len(vals) >= 2:
            m = statistics.median(vals)
            if m:
                rel.append((max(vals) - min(vals)) / m)
    if not rel:
        return {"n_shapes": 0, "drift_median": None, "drift_p90": None, "drift_max": None}
    return {"n_shapes": len(rel),
            "drift_median": statistics.median(rel),
            "drift_p90": sorted(rel)[min(len(rel) - 1, int(0.9 * len(rel)))],
            "drift_max": max(rel)}


def verdict(under_p95: float | None, drift: float | None,
            abs_rel_p90: float | None = None) -> dict[str, Any]:
    """预注册 §五 的判据 —— 这里**只引用阈值，不改**。"""
    reasons = []
    holds = True
    if under_p95 is None:
        reasons.append("低估误差不可算 ⇒ 无法判定")
        holds = False
    elif under_p95 > UNDEREST_P95_MAX:
        reasons.append(f"低估相对 p95 {under_p95:.2%} > {UNDEREST_P95_MAX:.0%}")
        holds = False
    if drift is not None and drift > CROSS_SESSION_DRIFT_MAX:
        reasons.append(f"跨会话漂移 {drift:.2%} > {CROSS_SESSION_DRIFT_MAX:.0%}")
        holds = False
    # ⚠️ 事后新增那一条（理由见常量注释）：没有它，"总是高估 400%"会被判成立。
    if abs_rel_p90 is None:
        reasons.append("整体精度不可算 ⇒ 无法判定")
        holds = False
    elif abs_rel_p90 > ABS_REL_P90_MAX:
        reasons.append(f"整体精度 |相对误差| p90 {abs_rel_p90:.2%} > {ABS_REL_P90_MAX:.0%}"
                       "（事后新增的判据：预注册的两条会被一个总是高估的模型满足）")
        holds = False
    return {"model_holds": holds, "reasons": reasons or ["全部判据均未触发"],
            "thresholds": {"underest_p95_max": UNDEREST_P95_MAX,
                           "cross_session_drift_max": CROSS_SESSION_DRIFT_MAX,
                           "abs_rel_p90_max_posthoc": ABS_REL_P90_MAX},
            "note": "abs_rel_p90_max 是事后新增的：预注册只约束低估与漂移，"
                    "而一个总是高估的模型会满足那两条"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="成本模型拟合与评估（cost-calib-v1）")
    ap.add_argument("--artifacts", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--split", choices=("hash", "rank_guaranteed"), default="hash",
                    help="留出划分方式。hash=纯哈希（v1..v4 协议，对**训练覆盖**是抽签）；"
                         "rank_guaranteed=每段先凑满秩再留出（归因审计证明它把 95.26%% 降到 11.77%%）")
    ap.add_argument("--hash-modulus", type=int, default=4,
                    help="留出规则：sha256(key)%%modulus==0 进留出。默认 4（v1..v3 协议）；"
                         "v4 协议在网格里登记为 3")
    ap.add_argument("--no-drift", action="store_true", help="只有一个会话时跳过漂移分析")
    args = ap.parse_args(argv)

    paths = []
    for a in args.artifacts:
        paths.extend(sorted(glob.glob(a)))
    recs, problems = load_records(paths)
    if not recs:
        print("✗ 没有可用记录：", problems)
        return 2
    sessions = sorted({r["session"] for r in recs})
    print(f"载入 {len(recs)} 条记录 / {len(sessions)} 个会话")
    for p in problems:
        print(f"  ⚠️ {p}")

    # ---- 三层划分（各自独立报，**不合成一个数**）----------------------------
    if args.split == "rank_guaranteed":
        hash_tr, hash_ho = split_rank_guaranteed(recs, holdout_modulus=args.hash_modulus)
    else:
        hash_tr = [r for r in recs if split_hash(r, args.hash_modulus) == "train"]
        hash_ho = [r for r in recs if split_hash(r, args.hash_modulus) == "hash_holdout"]
    # 会话留出：最后一个会话整体留出
    sess_ho_name = sessions[-1] if len(sessions) > 1 else None
    sess_tr = [r for r in recs if r["session"] != sess_ho_name]
    sess_ho = [r for r in recs if r["session"] == sess_ho_name]
    # 外推留出：所有 kv==2048 的点（测超出训练范围时的行为）
    kv_ho = [r for r in recs if int(r["kv_len"] or 0) == 2048]
    kv_tr = [r for r in recs if int(r["kv_len"] or 0) != 2048]

    print(f"\n=== 划分 ===")
    print(f"  划分方式 = {args.split}（1/{args.hash_modulus}）："
          f"train {len(hash_tr)} / holdout {len(hash_ho)}")
    # ★ **覆盖诊断**：秩不足的段必须点名 —— 误差本身不告诉你为什么。
    problems = diagnose_coverage(hash_tr, "train") + diagnose_coverage(hash_ho, "holdout")
    if problems:
        print(f"  ⚠️ **覆盖诊断发现 {len(problems)} 处秩不足**"
              "（**训练集**不足 ⇒ 模型学不到那个方向；**留出集**不足 ⇒ 评估数字偏乐观）：")
        for p in problems[:12]:
            print(f"     - {p}")
    else:
        print("  ✓ 覆盖诊断：每段训练集满秩（3）")
    print(f"  按会话整块留出：train {len(sess_tr)}（{len(sessions)-1 if sess_ho_name else 0} 会话）"
          f" / holdout {len(sess_ho)}（{sess_ho_name}）")
    print(f"  外推（全部 kv=2048）：{len(kv_ho)} 条")

    # ---- 评估：**每个上下文独立拟合**（否则会泄漏）--------------------------
    # ⚠️ 第一版把"查表"只建在 `hash_tr` 上，却拿它去评 `kv=2048` 外推集 ——
    # 而 `hash_tr` **含有** kv2048 的点 ⇒ 外推集里大半是查表见过的键 ⇒
    # 报出"0% 误差"这种漂亮的假结果。**每个评估上下文必须在自己的训练集上拟合**。
    contexts: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = [
        ("① 按形状留出（内插；规则与数据值无关）", hash_tr, hash_ho),
        (f"② 跨会话留出（整块留出会话 {sess_ho_name}）", sess_tr, sess_ho),
        ("③ 外推：全部 kv=2048（训练集里**没有** kv=2048）", kv_tr, kv_ho),
    ]

    results: dict[str, Any] = {}
    for title, tr, te in contexts:
        print(f"\n=== {title} ===")
        if not te:
            print("  （该上下文没有留出样本 ⇒ 跳过）")
            continue
        if not tr:
            print("  （训练集为空 ⇒ 跳过）")
            continue
        coef = fit_segmented(tr)
        tbl = fit_lookup(tr)
        pw = fit_piecewise(tr)
        ctx: dict[str, Any] = {"n_train": len(tr), "n_holdout": len(te)}
        for name, f in (
            ("分段模型(单超平面)", lambda r, c=coef: predict_segmented(c, r)),
            ("分段模型(按batch)", lambda r, m=pw: predict_piecewise(m, r)),
            ("查表", lambda r, t=tbl: predict_lookup(t, r)),
            ("**混合（部署形态）**", lambda r, c=coef, t=tbl: predict_hybrid(t, c, r)),
        ):
            e = errors(te, f)
            ctx[name] = e
            if not e["n"]:
                print(f"  {name:<20} 无记录")
                continue
            tail = (f"覆盖率 {e['coverage']:.0%} | 相对误差 p50 {e['abs_rel_p50']:.2%} "
                    f"p90 {e['abs_rel_p90']:.2%} p99 {e['abs_rel_p99']:.2%} | "
                    f"低估率 {e['underest_rate']:.0%} | **低估 p95 {e['underest_p95']:.2%}** "
                    f"| 最差低估 {e['worst_underest']:.2%}")
            print(f"  {name:<20} {tail}")
        results[title] = ctx
        print(f"    （该上下文的分段模型系数 = {[round(c, 4) for c in coef]}）")

    print(f"\n=== ④ 跨会话漂移（同形状、不同会话）===")
    drift = (cross_session_drift(recs, None) if not args.no_drift
             else {"drift_p90": None, "n_shapes": 0})
    if drift.get("n_shapes"):
        print(f"  覆盖 {drift['n_shapes']} 个形状 | 漂移 中位 {drift['drift_median']:.2%} "
              f"p90 {drift['drift_p90']:.2%} max {drift['drift_max']:.2%}")
    else:
        print("  （数据不足）")

    # ⚠️ 判据用**分段（按 batch）模型** —— 那才是任务点名的"分段模型"，
    # 也是唯一在本数据上站得住的候选。但**分段变量是事后选的**（见 fit_piecewise 的注释），
    # 所以下面把这件事显式写出来，不假装它是预注册的一部分。
    _pw_main = fit_piecewise(hash_tr)
    _tbl_main = fit_lookup(hash_tr)
    e_main = errors(hash_ho, lambda r: predict_piecewise(_pw_main, r))
    v = verdict(e_main.get("underest_p95"), drift.get("drift_p90"),
                e_main.get("abs_rel_p90"))
    v["model_used"] = "分段模型（按 batch 分段）"
    v["posthoc_note"] = ("分段变量(batch)是**看到数据之后**选的：实测 b1→b2 跳 +29% 而 b2→b4 几乎不动，"
                         "单一超平面在结构上拟合不了。预注册只点名了'分段模型'这一类，"
                         "**没有**指定分段方式 ⇒ 这一选择登记为 post-hoc。")
    print(f"\n=== ⑤ 判据 ===")
    print(f"  评估的模型：{v['model_used']}")
    print(f"  ⚠️ {v['posthoc_note']}")
    print(f"  模型成立？**{'是' if v['model_holds'] else '否'}**"
          f"  —— {('、'.join(v['reasons']))}")
    print(f"  阈值：低估 p95 ≤ {UNDEREST_P95_MAX:.0%}、跨会话漂移 ≤ "
          f"{CROSS_SESSION_DRIFT_MAX:.0%}（这两条**预注册**）；"
          f"|相对误差| p90 ≤ {ABS_REL_P90_MAX:.0%}（**事后新增**）")
    if not v["model_holds"]:
        print("  ⇒ 按 S4：第三臂降级或放弃，**不得**用失效模型产生结论。")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "n_records": len(recs), "sessions": sessions,
            "split": {"hash_train": len(hash_tr), "hash_holdout": len(hash_ho),
                      "session_holdout": sess_ho_name, "kv2048": len(kv_ho)},
            "contexts": results,
            "verdict_model": e_main,
            "cross_session_drift": drift,
            "verdict": v,
            "records": recs,
        }, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        print(f"\n已写出：{out}")
    return 0 if v["model_holds"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
