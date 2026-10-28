#!/usr/bin/env python3
"""**两个标定版本之间的语义核验**：把"口径改了，真的生效了"变成可测的对照实验。

为什么要有它（而不是在链里写一段 heredoc 现算）
----------------------------------------------
第 14 步的实锤：标定探针的 `prefill_tok` 在 `uniform` 口径下是**每条序列各做** `prefill_tok` 个
新 token，而 harness 的 `tokens_prefilled` 是**全步总量** ⇒ 同形状的成本差约 `batch` 倍。
`real_mixed` 口径（1 条做全部 + 其余各 1 个）才是 harness 的真实结构。

这件事必须**可证伪地**验收，否则"改对了"只是一句自我声明。本脚本把它拆成三问，
并且**带一个对照组**：

| 问 | 联的是 | 期望 | 若不符 |
| --- | --- | --- | --- |
| **C1 对照** | `pure_decode`（`pf=0`）同键比值 | ≈ 1（±tol） | 两会话本身不可比（机器漂移/重标定）⇒ 后两问**作废** |
| **C2 处理** | `mixed`（`pf>0, batch>1`）同键比值 | ≪ 1（约 `1/batch`） | 口径没生效，或没改对 |
| **C3 自证** | 产物里 `shape.n_new_mode` 的分布 | 新=`real_mixed`／旧=不含 | 语义声明**无据**（键不含该字段，看不出来） |

⚠️ **为什么要 C1**：没有对照时，"新版本所有键都变小了"可以被解释成"改对了"，
也可以是"这台机器换了/发热了/重标定了"。`pure_decode` 点在两种口径下**算的是一模一样的东西**
（`n_new=1`）⇒ 它必须**不动**。这是"先怀疑对比方法引入的差异，再怀疑被测对象"的落地。
⚠️ 若 C1 不通过，C2/C3 的结论**一律记作未判**，不得单独解释。

三态（不把"测不出"混进"通过"）
------------------------------
退出码：`0` = 全部通过；`1` = 有明确不符；`2` = 输入错；`3` = **未判**（同键太少，
统计量不可信 —— 与"通过"分开报）。

用法::

    python scripts/verify_calib_semantics.py \
        --old-model configs/cost_calib/model_v5.json \
        --new-model configs/cost_calib/model_v6.json \
        --old-artifacts "results/raw/_costcalib/costcalib_v5_s*.json" \
        --new-artifacts "results/raw/_costcalib/costcalib_v6_s*.json" \
        --json results/processed/verify_v5_v6.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]

#: 查表键的形状：`build_lookup()` 里的 `f"b{b}_pf{pf}_kv{kv}"`。
#: ⚠️ **不含 `n_new_mode`** —— 这正是 v5/v6 能同形状同键、从而可比的前提；
#: 也正是"光看键分不出两种口径"的原因（故需要 C3 回产物自证）。
_KEY_RE = re.compile(r"^b(?P<b>\d+)_pf(?P<pf>\d+)_kv(?P<kv>\d+)$")

_MIN_CONTROL_N = 5      #: 对照组同键数下限（少了就不判，避免高方差统计量当结论）
_MIN_TREAT_N = 3        #: 处理组同键数下限


def _parse_key(k: str) -> dict[str, int] | None:
    m = _KEY_RE.match(k)
    return None if not m else {g: int(m.group(g)) for g in ("b", "pf", "kv")}


def load_table(path: str | Path) -> dict[str, dict[str, Any]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    t = d.get("lookup_table") or {}
    if not t:
        raise ValueError(f"{path}: 没有 lookup_table（模型不是'查表+保守回退'分支导出的？）")
    return t


def same_key_rows(old: dict[str, dict[str, Any]],
                  new: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """同键对比，取 `max`（保守值）—— 判据用哪个值就比哪个值，别混 `median`。"""
    rows: list[dict[str, Any]] = []
    for k in sorted(set(old) & set(new)):
        p = _parse_key(k)
        if p is None:
            continue
        o = float(old[k].get("max") or 0.0)
        n = float(new[k].get("max") or 0.0)
        if o <= 0:
            continue                       # 比值无定义 ⇒ 不入统计（并让 n 显形）
        rows.append({**p, "key": k, "old_max": o, "new_max": n, "ratio": n / o})
    return rows


def _bucket(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"control": [], "treat": [], "mixed_b1": []}
    for r in rows:
        if r["pf"] == 0:
            out["control"].append(r)       # 两种口径算的是同一件事 ⇒ 必须不动
        elif r["b"] == 1:
            out["mixed_b1"].append(r)      # batch=1 时两种口径**也**等价（无"其余序列"）
        else:
            out["treat"].append(r)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "median": None, "min": None, "max": None,
                "max_abs_dev_from_1": None, "per_batch": {}}
    rs = [r["ratio"] for r in rows]
    pb: dict[int, list[float]] = {}
    for r in rows:
        pb.setdefault(r["b"], []).append(r["ratio"])
    return {"n": len(rows),
            "median": st.median(rs), "min": min(rs), "max": max(rs),
            "max_abs_dev_from_1": max(abs(x - 1.0) for x in rs),
            "per_batch": {str(b): round(st.median(v), 4) for b, v in sorted(pb.items())}}


def mode_distribution(paths: list[str]) -> dict[str, Any]:
    """产物里 `shape.n_new_mode` 的分布（按"是否 mixed"分开数）。

    `None` = 该产物**没有这个字段**（v4 及更早的采集代码）⇒ 与"记录了 uniform"不同：
    前者是"当时根本没记口径"，后者是"记了、是旧的错口径"。
    """
    overall: Counter[str] = Counter()
    mixed: Counter[str] = Counter()
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for pt in d.get("points") or []:
            sh = pt.get("shape") or {}
            m = sh.get("n_new_mode")
            tag = "（无此字段=旧口径）" if m is None else str(m)
            overall[tag] += 1
            if int(sh.get("prefill_tok") or 0) > 0 and int(sh.get("batch") or 0) > 1:
                mixed[tag] += 1
    return {"n_points": sum(overall.values()),
            "overall": dict(overall), "mixed_points": dict(mixed)}


def _expand(pats: list[str]) -> list[str]:
    out: list[str] = []
    for a in pats:
        out.extend(sorted(glob.glob(str(_ROOT / a))) or sorted(glob.glob(a)))
    # 去重保序
    seen: set[str] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="两个标定版本之间的语义核验（只读）")
    ap.add_argument("--old-model", required=True)
    ap.add_argument("--new-model", required=True)
    ap.add_argument("--old-artifacts", nargs="*", default=[])
    ap.add_argument("--new-artifacts", nargs="*", default=[])
    ap.add_argument("--control-tol", type=float, default=0.10,
                    help="对照组的允许相对偏移（默认 ±10%%）")
    ap.add_argument("--mixed-ratio-max", type=float, default=0.60,
                    help="处理组同键比值上限（默认 0.60；理想约 1/batch）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    try:
        old_tbl = load_table(args.old_model)
        new_tbl = load_table(args.new_model)
    except (OSError, ValueError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    rows = same_key_rows(old_tbl, new_tbl)
    bk = _bucket(rows)
    con = summarize(bk["control"])
    trt = summarize(bk["treat"])
    b1 = summarize(bk["mixed_b1"])

    print(f"旧模型 {Path(args.old_model).name}（查表 {len(old_tbl)} 键）"
          f"  vs  新模型 {Path(args.new_model).name}（查表 {len(new_tbl)} 键）")
    print(f"  同键 {len(rows)} 个 ⇒ 对照组(pf=0) {con['n']}"
          f"｜处理组(pf>0,b>1) {trt['n']}｜batch=1 混合 {b1['n']}")

    verdict: dict[str, Any] = {"checks": {}, "verdict": "pass"}
    rc = 0

    def record(name: str, ok: bool | None, detail: dict[str, Any]) -> None:
        nonlocal rc
        verdict["checks"][name] = {"ok": ok, **detail}
        if ok is False and rc == 0:
            rc = 1
        if ok is None and rc == 0:
            rc = 3

    # ---- C1 对照组 --------------------------------------------------------
    print(f"\n-- C1 对照组（pure_decode，pf=0；两种口径算的是**同一件事** ⇒ 必须≈1）")
    if con["n"] < _MIN_CONTROL_N:
        print(f"  ⚠️ 同键只有 {con['n']} 个（< {_MIN_CONTROL_N}）⇒ **未判**")
        record("C1_control_ratio", None, {"reason": "同键太少", **con})
    else:
        dev = con["max_abs_dev_from_1"]
        ok = dev <= args.control_tol
        print(f"  比值 中位 {con['median']:.4f}｜范围 [{con['min']:.4f}, {con['max']:.4f}]"
              f"｜最大|比值−1| {dev:.2%}（阈 {args.control_tol:.0%}）")
        print(f"  按 batch: {con['per_batch']}")
        print(f"  ⇒ {'✓ 可比' if ok else '✗ **两会话本身不可比 ⇒ C2/C3 一律未判**'}")
        record("C1_control_ratio", ok, {"tol": args.control_tol, **con})
        if not ok:
            print("  ⚠️ 先查机器状态/是否重标定/是否跨了不同 grid；**不要**把 C2 的下降当口径功劳")
            verdict["verdict"] = "fail"

    control_broken = verdict["checks"]["C1_control_ratio"]["ok"] is False

    # ---- C2 处理组 --------------------------------------------------------
    print(f"\n-- C2 处理组（mixed，pf>0 且 batch>1；应为旧口径的 ~1/batch）")
    if trt["n"] < _MIN_TREAT_N:
        print(f"  ⚠️ 同键只有 {trt['n']} 个（< {_MIN_TREAT_N}）⇒ **未判**")
        record("C2_mixed_ratio", None, {"reason": "同键太少", **trt})
    else:
        ok = trt["median"] <= args.mixed_ratio_max
        print(f"  比值 中位 {trt['median']:.4f}｜范围 [{trt['min']:.4f}, {trt['max']:.4f}]"
              f"（阈 ≤ {args.mixed_ratio_max}）")
        print(f"  按 batch（横比 1/batch，越大的 batch 降得越多才对）: {trt['per_batch']}")
        imp = [{"batch": int(b), "ratio": r, "1/b": round(1 / int(b), 4)}
               for b, r in trt["per_batch"].items()]
        print(f"  参考 1/batch: {[x['1/b'] for x in imp]}")
        print(f"  ⇒ {'✓ 口径生效' if ok else '✗ **没有生效或没改对**（比值接近 1 就是要找的信号）'}")
        record("C2_mixed_ratio", ok, {"mixed_ratio_max": args.mixed_ratio_max,
                                      "implied_1_over_b": imp, **trt})
        if not ok:
            verdict["verdict"] = "fail"
        if control_broken:
            print("  ⚠️ C1 未通过 ⇒ 本项**不得**单独解释")
            record("C2_mixed_ratio", None, {"reason": "C1 未通过", **trt})

    # ---- C3 自证（回产物）--------------------------------------------------
    print(f"\n-- C3 自证：产物里 `shape.n_new_mode` 的分布")
    n_old = len(_expand(args.old_artifacts))
    n_new = len(_expand(args.new_artifacts))
    if not (args.old_artifacts or args.new_artifacts):
        print("  （未给 --old-artifacts/--new-artifacts ⇒ 跳过；本项**不能**由模型文件自证）")
        record("C3_semantics_self_evidence", None, {"reason": "未给产物"})
    else:
        dist_old = mode_distribution(_expand(args.old_artifacts)) if n_old else None
        dist_new = mode_distribution(_expand(args.new_artifacts)) if n_new else None
        for label, dist, nf in (("旧", dist_old, n_old), ("新", dist_new, n_new)):
            if dist is None:
                print(f"  {label}: 文件数 0 ⇒ 未判")
                continue
            print(f"  {label}: 文件 {nf}｜点 {dist['n_points']}｜全部 {dist['overall']}"
                  f"｜其中 mixed 点 {dist['mixed_points']}")
        if dist_new is None:
            record("C3_semantics_self_evidence", None, {"reason": "新产物文件数 0"})
        else:
            has_new = "real_mixed" in (dist_new.get("mixed_points") or {})
            old_has = bool(dist_old and "real_mixed" in (dist_old.get("mixed_points") or {}))
            ok = has_new and not old_has
            print(f"  新产物 mixed 点含 real_mixed: {has_new}｜旧产物 mixed 点含 real_mixed: {old_has}")
            if old_has:
                print("  ⚠️ **旧产物也标了 real_mixed** ⇒ 'v5 口径错'这个前提不成立，"
                      "整条链的叙事要重查（不要硬套预期）")
            elif not has_new:
                print("  ✗ 新产物里找不到 real_mixed ⇒ 口径修复**无据**"
                      "（键不含该字段 ⇒ 光比键看不出两种口径）")
            print(f"  ⇒ {'✓ 语义有据' if ok else '✗ 语义无据或不成立'}")
            record("C3_semantics_self_evidence", ok,
                   {"new_has_real_mixed": has_new, "old_has_real_mixed": old_has,
                    "old": dist_old, "new": dist_new})
            if not ok:
                verdict["verdict"] = "fail"

    # ---- 结论 -------------------------------------------------------------
    label = {0: "pass", 1: "fail", 3: "undetermined"}[rc]
    print(f"\n★ 结论: **{label}**"
          + ("  （含未判项：同键太少，不把'测不出'当'通过'）" if rc == 3 else ""))
    print("  提醒：本脚本只核**内部一致性**（两版本之间 + 产物自证）；"
          "**不**证明新口径更接近真机 —— 那需要与遥测实测步时比较。")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        verdict.update({"old_model": args.old_model, "new_model": args.new_model,
                        "n_common_keys": len(rows),
                        "control": con, "treatment": trt, "mixed_b1": b1})
        out.write_text(json.dumps(verdict, ensure_ascii=False, indent=1),
                       encoding="utf-8", newline="\n")
        print(f"  已写出：{out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
