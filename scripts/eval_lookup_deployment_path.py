#!/usr/bin/env python3
"""**按部署路径**重新评估查表模型（第 3 步）。

为什么必须有这个脚本（它不是"再写一遍评估"）
--------------------------------------------
`scripts/costcalib_fit.py` 报的 `查表` 行**不是**部署路径。三处实质差异：

| | `costcalib_fit.py` 的 `查表` 行 | **部署路径**（`CostModel.predict(mode="lookup")`）|
| --- | --- | --- |
| 键 | `(batch, prefill_tok, kv_len, n_forwards)` 四元 | `b{batch}_pf{pf}_kv{kv}` 三元（**不含** `n_forwards`）|
| 分桶 | 无（精确键）| **`lookup_bucket`**（v7：`pf_step=0`, `kv_step=256`）|
| 取值 | `median` | **`max`**（保守，绝不低估）|

⇒ 用前者的数字回答"lookup 通过了吗"是**指代错误**：它评的是一个从未被部署的模型。
实测（本脚本 `--diagnose`）：四元精确键 + median 在 v7 上给的覆盖率是 **3.16%**
（① 按形状留出上下文），而部署路径的覆盖率是另一回事 —— 两者不可互相替代。

⚠️ **一个必须避开的陷阱（本地实测踩到）**：模型文件里的 `lookup_table` 是用**全部**标定
记录建的（生产模型该用上全部数据）。若拿它去评**同一批**记录，覆盖率恒 = 100%、
低估恒 = 0% —— 那是**自匹配**，不是泛化。第一版脚本就掉进去了（四处上下文全 100%，
连"外推"上下文都是 100%，这个"太好"本身就是告警）。
⇒ 正确做法：**按留出划分重新拟合一份查表**（规则逐字沿用 `export_cost_model.build_lookup`
+ `cost_model.predict(mode="lookup")` 的分桶），只在**训练划分**上建键，再评留出。

本脚本的口径
------------
* **预测器 = 部署的那一个**：重拟合后用 `CostModel.predict(mode="lookup")` 出值
  —— **不重抄一份查表分支**（两份易碎的实现迟早不一致）。
  唯一被重做的环节是"这份表建在哪些点上"（必须是训练划分），而**键/分桶/取值规则**
  逐字来自部署实现。
* **四组量并排报**（用户指令第 3 步逐字要求）：覆盖率 / 低估 / 高估 / 回退。
  * 覆盖率 = 命中 / 总数（命中 = 模型给出预测，即未回退）；
  * 低估 p95 = 命中样本里 `pred < actual` 的相对缺口 p95（**正数**）；
  * 高估 p95 = 命中样本里 `pred > actual` 的相对超出 p95（**正数**）；
  * 回退率 = 未命中 / 总数。
* ⚠️ **安全余量**：部署时 `predict(apply_margin=True)` 会乘 `safety_margin`。
  本脚本**两个都报**（`with_margin` / `no_margin`），因为判据用的是乘余量后的值，
  而"查表本身准不准"要看乘余量前 —— 混在一起会说不清余量贡献了多少。

用法::

    python scripts/eval_lookup_deployment_path.py --diagnose \\
        --eval results/processed/costcalib_eval_v7.json \\
        --model configs/cost_calib/model_v7.json \\
        --json results/processed/lookup_deploy_eval_v7.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.cost_model import CostModel  # noqa: E402

#: 预注册阈值（与 `costcalib_fit.py` **同源**，这里只引用、不改）。
UNDEREST_P95_MAX = 0.05
ABS_REL_P90_MAX = 0.10


def build_lookup_key(rec: dict[str, Any], bucket: dict[str, Any] | None) -> str:
    """**与部署实现逐字同构**的键（`cost_model.predict` 的 lookup 分支 + 导出器的 `build_lookup`）。

    ⚠️ 这两处的键格式**必须**一致：导出器写 `f"b{...}_pf{...}_kv{...}"`，
    `predict` 读同一个拼法。任何一处改了，另一处不改 ⇒ 查表静默恒未命中（回退率 100%），
    而**不会报错**。
    """
    bk = bucket or {}
    pf_step = int(bk.get("pf_step") or 0)
    kv_step = int(bk.get("kv_step") or 0)
    pf = int(rec["prefill_tok"])
    kv = int(rec["kv_len"])
    _pf = (pf // pf_step) * pf_step if pf_step else pf
    _kv = (kv // kv_step) * kv_step if kv_step else kv
    return f"b{int(rec['batch'])}_pf{_pf}_kv{_kv}"


def refit_lookup_table(train: list[dict[str, Any]], bucket: dict[str, Any] | None
                       ) -> dict[str, dict[str, Any]]:
    """在**训练划分**上重拟合查表 —— 规则逐字沿用 `export_cost_model.build_lookup`
    （键 = 分桶后的三元；值 = 该键实测的 **max**，保守、绝不低估）。"""
    by: dict[str, list[float]] = {}
    for r in train:
        by.setdefault(build_lookup_key(r, bucket), []).append(float(r["actual_ms"]))
    return {k: {"median": float(__import__("statistics").median(v)),
                "max": float(max(v)), "n": len(v)}
            for k, v in sorted(by.items())}


def make_predictor(base_model: CostModel, train: list[dict[str, Any]],
                   bucket: dict[str, Any] | None, *, apply_margin: bool):
    """造一个**只在 `train` 上建过键**的部署路径预测器。

    做法：深拷贝模型 payload、把 `lookup_table` 换成训练划分重拟合的那份，
    其余（`lookup_bucket` / `safety_margin` / `lookup_value`）**逐字保留**，
    再走 `CostModel.predict(mode='lookup')`。⇒ 键、分桶、取值、余量四条与部署完全同源。
    """
    import copy
    payload = copy.deepcopy(base_model.payload)
    payload["lookup_table"] = refit_lookup_table(train, bucket)
    payload["default_mode"] = "lookup"
    m = CostModel(payload=payload)

    def predict(r: dict[str, Any]) -> float | None:
        p = m.predict(batch=int(r["batch"]), prefill_tok=int(r["prefill_tok"]),
                      kv_len=int(r["kv_len"]), n_forwards=int(r.get("n_forwards") or 1),
                      apply_margin=apply_margin, mode="lookup")
        return None if p is None else p.total_ms

    return predict


# --------------------------------------------------------------------------- 划分
def split_rank_guaranteed(recs: list[dict[str, Any]], holdout_modulus: int = 3
                          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """**秩保证划分** —— 逐字复用 `costcalib_fit.split_rank_guaranteed` 的规则。

    为什么不 import：那份实现依赖 `numpy`，而本脚本要在无 numpy 的解释器上也能跑。
    规则本身很短（每段先凑满秩进训练、其余按 hash 留出），照抄规则比照抄代码更稳。
    ⚠️ 若 `costcalib_fit` 的规则变了，这里必须同步 —— 判据口径只能有一个。
    """
    def _rank(rows: list[dict[str, Any]]) -> int:
        pts = [(float(r["prefill_tok"] or 0), float(r["kv_len"] or 0)) for r in rows]
        if not pts:
            return 0
        # 秩 = 独立方向数：常量项恒有 ⇒ 起点 1；再看 pf / kv 是否各有变化。
        rank = 1
        if len({p[0] for p in pts}) > 1:
            rank += 1
        if len({p[1] for p in pts}) > 1:
            rank += 1
        return rank

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
            if _rank(cand) >= 3:
                break
        for r in rest:
            v = int(hashlib.sha256(str(r["key"]).encode()).hexdigest()[:8], 16) % holdout_modulus
            (ho if v == 0 else tr).append(r)
        tr.extend(core)
    return tr, ho


def split_hash(rec: dict[str, Any], modulus: int = 4) -> str:
    h = hashlib.sha256(str(rec["key"]).encode("utf-8")).hexdigest()
    return "hash_holdout" if int(h[:8], 16) % modulus == 0 else "train"


# --------------------------------------------------------------------------- 度量
def _q(vals: list[float], frac: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(frac * len(s)))]


def measure(recs: list[dict[str, Any]], predict: Callable[[dict], float | None]) -> dict[str, Any]:
    """一组预测的四组量：覆盖率 / 低估 / 高估 / 回退。**未命中不填 0，计数并跳过。**"""
    n = len(recs)
    n_miss = 0
    rel: list[float] = []
    under: list[float] = []
    over: list[float] = []
    for r in recs:
        p = predict(r)
        if p is None:
            n_miss += 1
            continue
        a = float(r["actual_ms"])
        if a <= 0:
            continue
        d = (p - a) / a
        rel.append(d)
        (under if d < 0 else over).append(abs(d))
    n_hit = len(rel)
    return {
        "n": n,
        "n_hit": n_hit,
        "n_miss_fallback": n_miss,
        "coverage": (n_hit / n) if n else 0.0,
        "fallback_rate": (n_miss / n) if n else 0.0,
        "underest_rate": (len(under) / n_hit) if n_hit else None,
        "underest_p95": _q(under, 0.95),
        "worst_underest": (max(under) if under else None),
        "overest_rate": (len(over) / n_hit) if n_hit else None,
        "overest_p95": _q(over, 0.95),
        "abs_rel_p50": _q([abs(x) for x in rel], 0.5),
        "abs_rel_p90": _q([abs(x) for x in rel], 0.9),
    }


def _fmt(m: dict[str, Any]) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x:.2%}"
    return (f"覆盖 {pct(m['coverage'])}（命中 {m['n_hit']}/{m['n']}）| "
            f"回退 {pct(m['fallback_rate'])} | "
            f"低估率 {pct(m['underest_rate'])} 低估p95 {pct(m['underest_p95'])} | "
            f"高估率 {pct(m['overest_rate'])} 高估p95 {pct(m['overest_p95'])} | "
            f"|rel| p50 {pct(m['abs_rel_p50'])} p90 {pct(m['abs_rel_p90'])}")


# --------------------------------------------------------------------------- 主
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按部署路径评估查表模型（第 3 步）")
    ap.add_argument("--eval", required=True, help="costcalib_fit.py 的评估产物（含 records）")
    ap.add_argument("--model", required=True, help="部署用的模型文件（default_mode=lookup）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--diagnose", action="store_true",
                    help="同时报『旧口径（四元精确键+median）』以显示指代差异")
    args = ap.parse_args(argv)

    ev = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    recs = ev.get("records") or []
    if not recs:
        print("✗ 评估产物里没有 records", file=sys.stderr)
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"✗ 找不到模型文件 {model_path}", file=sys.stderr)
        return 2
    # ⚠️ 离线加载：标定评估没有真实运行环境可观测 ⇒ **明说未做指纹校验**
    #（`load_offline` 会把这件事记进 relaxations，随 describe() 进产物）。
    cm = CostModel.load_offline(model_path)
    mode = str(cm.payload.get("default_mode") or "segmented")
    bucket = cm.payload.get("lookup_bucket") or None
    print(f"模型 {model_path.name} | default_mode={mode} | "
          f"lookup_bucket={bucket or '精确键'} | "
          f"lookup_value={cm.payload.get('lookup_value')}")
    print("⚠️ 模型文件里的 lookup_table 建在**全部**记录上 ⇒ 本脚本**不直接用它**评估"
          "（那是自匹配）。改为按每个上下文的训练划分**重拟合键**，再走部署的 predict。")
    if mode != "lookup":
        print(f"⚠️ 部署模式不是 lookup（={mode}）—— 本脚本评的是部署实际用的那一种。",
              file=sys.stderr)

    # ---- 上下文划分（与 costcalib_fit 的三个上下文对齐）----------------------
    sessions = sorted({r["session"] for r in recs})
    sess_ho_name = sessions[-1] if len(sessions) > 1 else None
    hash_tr, hash_ho = split_rank_guaranteed(recs, holdout_modulus=3)
    sess_ho = [r for r in recs if r["session"] == sess_ho_name] if sess_ho_name else []
    sess_tr = [r for r in recs if r["session"] != sess_ho_name] if sess_ho_name else []
    kv_ho = [r for r in recs if int(r["kv_len"] or 0) == 2048]
    kv_tr = [r for r in recs if int(r["kv_len"] or 0) != 2048]

    contexts: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = [
        ("① 按形状留出（秩保证划分 1/3）", hash_tr, hash_ho),
        (f"② 跨会话留出（整块留出会话 {sess_ho_name}）", sess_tr, sess_ho),
        ("③ 外推：全部 kv=2048", kv_tr, kv_ho),
    ]

    out: dict[str, Any] = {
        "model": str(model_path),
        "model_describe": cm.describe(),
        "default_mode": mode,
        "lookup_bucket": cm.payload.get("lookup_bucket") or None,
        "lookup_value": cm.payload.get("lookup_value"),
        "safety_margin": cm.payload.get("safety_margin"),
        "n_records": len(recs),
        "sessions": sessions,
        "split": {"rank_train": len(hash_tr), "rank_holdout": len(hash_ho),
                  "session_holdout": sess_ho_name},
        "contexts": {},
        "note": ("预测器 = 每个上下文在**自己的训练划分**上重拟合键，再走 "
                 "CostModel.predict(mode='lookup')（与部署同一份预测实现）。"
                 "with_margin 是判据实际用的值；no_margin 是查表本身。"
                 "未命中一律计入回退，**不填 0**。"
                 "⚠️ 模型文件自带的 lookup_table 建在全部记录上 ⇒ **不用于评估**（自匹配）。"),
    }

    for title, tr, te in contexts:
        print(f"\n=== {title} ===")
        if not te:
            print("  （无留出样本 ⇒ 跳过）")
            out["contexts"][title] = {"skipped": "no holdout"}
            continue
        ctx: dict[str, Any] = {"n_train": len(tr), "n_holdout": len(te)}
        # ★ 每个上下文**在自己的训练划分上**重拟合键 —— 否则就是自匹配（见模块 docstring）。
        pred_hold_m = make_predictor(cm, tr, bucket, apply_margin=True)
        pred_hold_nm = make_predictor(cm, tr, bucket, apply_margin=False)
        m_hold_m = measure(te, pred_hold_m)
        m_hold_nm = measure(te, pred_hold_nm)
        # 训练集自身命中率（诊断用：区分"桶太粗"与"根本没见过"）
        m_train_m = measure(tr, pred_hold_m)
        ctx["holdout_with_margin"] = m_hold_m
        ctx["holdout_no_margin"] = m_hold_nm
        ctx["train_with_margin"] = m_train_m
        ctx["n_lookup_keys_on_train"] = len(refit_lookup_table(tr, bucket))
        print(f"  留出·带余量   {_fmt(m_hold_m)}")
        print(f"  留出·无余量   {_fmt(m_hold_nm)}")
        print(f"  （训练划分建了 {ctx['n_lookup_keys_on_train']} 个键；"
              f"训练集自身命中 {m_train_m['n_hit']}/{m_train_m['n']} "
              f"= {m_train_m['coverage']:.2%}）")
        if args.diagnose:
            ctx["legacy_exact_median"] = _legacy_exact_median(tr, te)
            lm = ctx["legacy_exact_median"]
            print(f"  [诊断] 旧口径(四元精确键+median，键也只建在训练划分上) "
                  f"覆盖 {lm['coverage']:.2%}（命中 {lm['n_hit']}/{lm['n']}）"
                  f" —— **这不是部署路径**（键不含 n_forwards、无分桶、取 median）")
        out["contexts"][title] = ctx

    # ---- 判据（**只引预注册阈值，不改**）------------------------------------
    print(f"\n=== 判据（按部署路径，① 上下文）===")
    c1 = out["contexts"].get("① 按形状留出（秩保证划分 1/3）") or {}
    hm = c1.get("holdout_with_margin") or {}
    verdict: dict[str, Any] = {}
    if hm:
        reasons: list[str] = []
        holds = True
        up95 = hm.get("underest_p95")
        if up95 is None:
            reasons.append("低估 p95 不可算（无命中样本）⇒ 无法判定")
            holds = False
        elif up95 > UNDEREST_P95_MAX:
            reasons.append(f"低估相对 p95 {up95:.2%} > {UNDEREST_P95_MAX:.0%}")
            holds = False
        ar90 = hm.get("abs_rel_p90")
        if ar90 is None:
            reasons.append("|相对误差| p90 不可算 ⇒ 无法判定")
            holds = False
        elif ar90 > ABS_REL_P90_MAX:
            reasons.append(f"整体精度 |相对误差| p90 {ar90:.2%} > {ABS_REL_P90_MAX:.0%}")
            holds = False
        cvg = hm.get("coverage") or 0.0
        if cvg < 1.0:
            reasons.append(f"覆盖率 {cvg:.2%} < 100% ⇒ 回退 {hm.get('fallback_rate'):.2%}"
                           "（查表的代价，必须与误差并排报）")
        verdict = {"lookup_holds": holds, "reasons": reasons or ["全部判据均未触发"],
                   "coverage": cvg, "fallback_rate": hm.get("fallback_rate"),
                   "basis": "部署路径（预测器 = CostModel.predict(mode='lookup')）",
                   "thresholds": {"underest_p95_max": UNDEREST_P95_MAX,
                                  "abs_rel_p90_max": ABS_REL_P90_MAX}}
        print(f"  查表（部署路径）成立？**{'是' if holds else '否'}**"
              f"  —— {('、'.join(reasons)) or '全部判据均未触发'}")
        print(f"  覆盖率 {cvg:.2%} / 回退 {hm.get('fallback_rate'):.2%}"
              f" —— 覆盖率是查表的头部指标，必须与误差并排读。")
    else:
        print("  ⚠️ ① 上下文无留出样本 ⇒ 无法判定")
        verdict = {"lookup_holds": None, "reasons": ["① 上下文无留出样本"]}
    out["verdict"] = verdict

    # ---- ★ 结构性诊断：跨会话留出**不是**泛化考验 ---------------------------
    # 实测踩到：② 覆盖率 100%、① 只有 6.67%，两者"看起来"矛盾。真因是
    # **两个会话是同一张网格的两次遍历**（v7 s1/s2 的键交集 = 107/107 全等）
    # ⇒ 留出会话的形状，训练会话里**全都有** ⇒ ② 的 100% 是"**换了会话、没换形状**"，
    # 不是泛化。若不点名这件事，读者会拿 100% 当"查表泛化很好"的证据。
    diag = _cross_session_shape_overlap(recs, bucket)
    out["cross_session_shape_overlap"] = diag
    print("\n=== ★ 结构性诊断：跨会话留出的形状重叠 ===")
    if diag.get("applicable"):
        print(f"  会话 {diag['s1']} 键 {diag['n_s1']} / {diag['s2']} 键 {diag['n_s2']}"
              f" / 交集 {diag['n_intersect']}（占 {diag['intersect_frac']:.1%}）")
        print("  ⇒ 跨会话留出**不构成泛化考验**：留出的形状在训练会话里全有。")
        print("  ⇒ 判据必须以 **① 按形状留出** 为准（只有它真的留出了形状）。")
    else:
        print(f"  （不适用：{diag.get('why')}）")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding="utf-8", newline="\n")
        print(f"\n已写出 {p}")
    return 0


def _cross_session_shape_overlap(recs: list[dict[str, Any]],
                                 bucket: dict[str, Any] | None) -> dict[str, Any]:
    """两个会话各自的**形状键集合**是否重叠 —— 判断"跨会话留出"有没有意义。

    ★ 为什么这必须是一个显式诊断：若两会话是**同一张网格**的两次遍历，
    则"留出一个会话"退化成"留出一批重复测量"：形状一个没少，只是换了次运行。
    此时跨会话覆盖率必然接近 100%，而它**不能**被读成"查表泛化好"。
    """
    sessions = sorted({r["session"] for r in recs})
    if len(sessions) < 2:
        return {"applicable": False, "why": f"只有 {len(sessions)} 个会话"}
    a, b = sessions[0], sessions[-1]
    ka = {build_lookup_key(r, bucket) for r in recs if r["session"] == a}
    kb = {build_lookup_key(r, bucket) for r in recs if r["session"] == b}
    inter = ka & kb
    denom = min(len(ka), len(kb)) or 1
    return {"applicable": True, "s1": a, "s2": b, "n_s1": len(ka), "n_s2": len(kb),
            "n_intersect": len(inter), "intersect_frac": len(inter) / denom}


def _legacy_exact_median(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    """**仅诊断用**：`costcalib_fit.py` 的旧口径（四元精确键 + median）。

    它评的**不是**部署路径，列出来只为显示指代差异有多大。
    键按旧口径建在**训练划分**上（与正确做法一致），这样差异只来自"键/取值口径"，
    不掺入泄漏。
    """
    import statistics as st
    per: dict[tuple, list[float]] = {}
    for r in train:
        per.setdefault((int(r["batch"]), int(r["prefill_tok"]), int(r["kv_len"]),
                        int(r.get("n_forwards") or 1)), []).append(float(r["actual_ms"]))
    tbl = {k: st.median(v) for k, v in per.items()}
    return measure(test, lambda r: tbl.get((int(r["batch"]), int(r["prefill_tok"]),
                                            int(r["kv_len"]),
                                            int(r.get("n_forwards") or 1))))


if __name__ == "__main__":
    raise SystemExit(main())
