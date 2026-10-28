#!/usr/bin/env python3
"""**2×2 解耦**：{旧网格, 新网格} × {哈希划分, 秩保证划分}，并把"五个数"一起报出来。

为什么必须解耦
--------------
v4 **同时**改了采样网格（`kv` 加密）与划分规则 ⇒ **即使 v4 通过，也不能单独归功于"加密 kv"**。
要分清两者各贡献多少，就得把这个 2×2 表跑出来。

为什么必须报**五个数**
----------------------
`train n` / `holdout n` / **训练集最小秩** / **留出集最小秩** / 低估 p95。
没有它们，"误差"是不可解释的：实测踩到过——同一个模型，哈希 1-in-4 下是 **1.77%**、
1-in-2 下是 **95.25%**，差别**全在训练集的秩**（某段训练集只剩纯 decode ⇒ 秩 2 < 参数 3）。

判据规则（**提前固定，不许事后挑**）
------------------------------------
* 低估 p95 ≤ 5%、跨会话漂移 ≤ 10%、|rel| p90 ≤ 10%；
* ⚠️ **`holdout < 15` ⇒ 该格只能记「未判」**（`int(0.95×3)=2` 时那个"p95"就是最大值本身）。

实现纪律：**复用** `costcalib_fit.py` 的划分与指标函数（**不另写一份**——
两份实现迟早不一致，且这种不一致不会报错）。

用法::

    python scripts/decouple_grid_split.py \
        --old "results/raw/_costcalib/costcalib_v3_s*.json" \
        --new "results/raw/_costcalib/costcalib_v4_s*.json" \
        --holdout-modulus 2 --json results/processed/decouple_v3_v4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import costcalib_fit as F  # noqa: E402  —— 复用，不复制

#: 判据（预注册值；改这里等于改协议）
UNDEREST_P95_MAX = 0.05
ABS_REL_P90_MAX = 0.10
MIN_HOLDOUT_FOR_VERDICT = 15      # 留出少于它 ⇒ 判据只能记"未判"


def _min_rank(rows: list[dict[str, Any]]) -> int | None:
    """逐段设计矩阵秩的**最小值**（`None` = 没有段）。

    ⚠️ 秩是**方向数**：`prefill=0` + 多个 `kv` 的秩是 2（常数项 + kv），
    而模型有 3 个参数 ⇒ 秩 2 仍**不可辨识**。判据是"秩 < 参数个数"。
    """
    bs = sorted({int(r["batch"] or 1) for r in rows})
    if not bs:
        return None
    return min(F._rank([r for r in rows if int(r["batch"] or 1) == b]) for b in bs)


def _fit_once(artifact_glob: str, split: str, modulus: int, tmp: Path) -> dict[str, Any]:
    """**调用拟合脚本**跑一格，返回它的 `--json` 产物。

    ⚠️ 为什么不在这里自己解析/自己划分：那会把划分与指标**再实现一份**，
    而两份实现迟早不一致 —— 而且这种不一致**不会报错**（本仓库为此吃过亏）。
    拟合脚本是唯一实现，本驱动只负责编排与制表。
    """
    import subprocess
    tmp.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(_ROOT / "scripts" / "costcalib_fit.py"),
           "--artifacts", str(_ROOT / artifact_glob),
           "--split", split, "--hash-modulus", str(modulus), "--json", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    # ⚠️ **退出码语义混淆的陷阱**：拟合脚本在"**模型不成立**"时**也返回 1**
    #（那是有效结论，eval JSON 照样写出）。实测踩到过：本驱动把它当"拟合失败"，
    # 于是 4 个格全被跳过。⇒ 判据是 **JSON 有没有写出来**，不是 rc。
    #（夜间队列的闸门当时就是读 JSON 的 `verdict.model_holds`，所以没踩到。）
    if not tmp.exists():
        raise RuntimeError(
            f"拟合没有产出 JSON（rc={r.returncode}）：{' '.join(cmd)}\n"
            f"stdout 尾：{(r.stdout or '')[-400:]}\nstderr 尾：{(r.stderr or '')[-400:]}")
    return json.loads(tmp.read_text(encoding="utf-8"))


def evaluate(artifact_glob: str, split: str, modulus: int, tmp: Path) -> dict[str, Any]:
    """跑一格，取出**五个数**与指标。

    五个数：`train n` / `holdout n` / 训练集最小秩 / 留出集最小秩 / 低估 p95
    —— 没有它们，"误差"不可解释。

    ⚠️ 产物里 `split` 段的键名仍叫 `hash_train`/`hash_holdout`（即使实际用的是秩保证划分）
    ⇒ 只把它们当作"训练/留出的大小"读，别被键名误导。
    """
    ev = _fit_once(artifact_glob, split, modulus, tmp)
    recs = ev.get("records") or []
    if not recs:
        raise RuntimeError(f"{artifact_glob} 的拟合产物里没有 records")
    if split == "rank_guaranteed":
        tr, ho = F.split_rank_guaranteed(recs, holdout_modulus=modulus)
    else:
        tr = [r for r in recs if F.split_hash(r, modulus) == "train"]
        ho = [r for r in recs if F.split_hash(r, modulus) == "hash_holdout"]
    ctx = ev.get("contexts") or {}
    key = next((k for k in ctx if k.startswith("①")), None)
    m = ((ctx.get(key) or {}).get("分段模型(按batch)") or {}) if key else {}
    return {
        "split": split, "modulus": modulus,
        "train_n": len(tr), "holdout_n": len(ho),
        "train_min_rank": _min_rank(tr), "holdout_min_rank": _min_rank(ho),
        "underest_p95": m.get("underest_p95"), "abs_rel_p90": m.get("abs_rel_p90"),
        "eval_json": str(tmp.relative_to(_ROOT)),
        "model_holds": ((ev.get("verdict") or {}).get("model_holds")),
    }


def verdict(r: dict[str, Any]) -> str:
    """**按提前固定的规则**给结论（不许事后挑）。"""
    if r["holdout_n"] < MIN_HOLDOUT_FOR_VERDICT:
        return f"未判（留出只有 {r['holdout_n']} 点 < {MIN_HOLDOUT_FOR_VERDICT}）"
    if r["train_min_rank"] is not None and r["train_min_rank"] < 3:
        return f"不成立（训练集最小秩 {r['train_min_rank']} < 3 ⇒ 该段不可辨识）"
    bad = []
    if r["underest_p95"] is None or r["underest_p95"] > UNDEREST_P95_MAX:
        bad.append(f"低估p95 {r['underest_p95']:.2%} > {UNDEREST_P95_MAX:.0%}")
    if r["abs_rel_p90"] is None or r["abs_rel_p90"] > ABS_REL_P90_MAX:
        bad.append(f"|rel|p90 {r['abs_rel_p90']:.2%} > {ABS_REL_P90_MAX:.0%}")
    return "成立" if not bad else "不成立（" + "；".join(bad) + "）"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="2×2 解耦（网格 × 划分）")
    ap.add_argument("--old", required=True, help="旧网格产物 glob（相对仓库根）")
    ap.add_argument("--new", required=True, help="新网格产物 glob")
    ap.add_argument("--holdout-modulus", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    grids = {"旧网格": args.old, "新网格": args.new}
    tmpdir = _ROOT / "results" / "processed" / "_decouple"

    print(f"2×2 解耦｜holdout-modulus={args.holdout_modulus}｜"
          f"判据：低估p95≤{UNDEREST_P95_MAX:.0%}、|rel|p90≤{ABS_REL_P90_MAX:.0%}、"
          f"留出≥{MIN_HOLDOUT_FOR_VERDICT} 点")
    header = (f"{'网格':<8}{'划分':<18}{'train':>6}{'holdout':>8}{'训练最小秩':>11}"
              f"{'留出最小秩':>11}{'低估p95':>10}{'|rel|p90':>10}   结论")
    print(header)
    print('-' * len(header))
    out: dict[str, Any] = {"holdout_modulus": args.holdout_modulus, "cells": {}}
    for gname, glob_pat in grids.items():
        for split in ("hash", "rank_guaranteed"):
            tmp = tmpdir / f"{gname}_{split}.json"
            try:
                r = evaluate(glob_pat, split, args.holdout_modulus, tmp)
            except Exception as exc:                       # noqa: BLE001
                print(f"{gname:<8}{split:<18} —— 跳过：{exc}")
                continue
            v = verdict(r)
            r["verdict"] = v
            out["cells"][f"{gname}|{split}"] = r
            f = lambda x, p='{:.2%}': ('—' if x is None else p.format(x))  # noqa: E731
            print(f"{gname:<8}{split:<18}{r['train_n']:>6}{r['holdout_n']:>8}"
                  f"{str(r['train_min_rank']):>11}{str(r['holdout_min_rank']):>11}"
                  f"{f(r['underest_p95']):>10}{f(r['abs_rel_p90']):>10}   {v}")
    print()
    print("读法：**同一列内比划分**（哈希 vs 秩保证）⇒ 划分规则贡献多少；")
    print("      **同一行内比网格**（旧 vs 新）⇒ 采样密度贡献多少。两者不能混为一谈。")
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding="utf-8", newline="\n")
        print(f"\n已写出：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
