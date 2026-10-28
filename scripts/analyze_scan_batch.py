#!/usr/bin/env python3
"""读 `S14S` 扫描批的产物，**机械地**套用 `EXPERIMENTS.md` §7d 第 4/5 节的冻结判据。

为什么要有这个脚本：§7d 的判据（① 达标率 == 1.0；② ttft.p50 ≤ 2000 且 tpot.p50 ≤ 100；
③ 3 重复稳健 CV ≤ 10%）是**预注册**的，必须**机械执行**，不能事后再"看着数据商量"。
把规则写进代码是让它**无法被事后调整**的唯一办法。

★ 键名全部对真产物核对过（`results/raw/S14GV-.../meta.json`）：
  分位块是**嵌套的** ⇒ `summary.ttft.p50`，不是 `summary.ttft_ms_p50`。
   凭记忆写会静默取到 None，"全部不可达"与"读错键"就无法区分。

用法::

    python scripts/analyze_scan_batch.py <产物目录> [--tag S14S]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

#: §7d 冻结的量（逐字取自真产物键名）
SLO_TTFT = 2000.0
SLO_TPOT = 100.0
CV_LIMIT = 0.10

#: 格名尾部形状 `..._c<c>_m<m>`（见 `_cell_cm`）。
#: ⚠️ 放在**模块级**而不是函数内 —— 函数内 import 是"出了那个函数就失效"的老坑
#: （本项目踩过：局部 import 被 `try` 吞掉 ⇒ 退出码仍为 0，看起来像跑完了）。
_RE_CELL_CM = re.compile(r"_c(\d+)_m(\d+)$")


def robust_cv(values: list[float]) -> float | None:
    """1.4826·MAD / median —— 对 outlier 有抵抗力的离散度（AGENTS.md A5）。"""
    if len(values) < 2:
        return None
    med = statistics.median(values)
    if med == 0:
        return None
    mad = statistics.median([abs(v - med) for v in values])
    return 1.4826 * mad / abs(med)


def _cell_cm(stem: str) -> tuple[int, int]:
    """从格名抽出 `(concurrency, max_num_seqs)`。

    旧写法是 `int(stem.split("_c")[-1].split("_")[0])` 配 `int(stem.split("_m")[-1])`。

    ⚠️ **诚实标注（2026-09-28 复核）**：我**没能**构造出让两者结果不同的格名
    （试了 `_mp8` 后缀、中缀 `_m2`、`c2_m4m8` 等），**本批 8 个格名两者也一致**
    ⇒ 这**不是**一个已发生的 bug 修复，而是**契约收紧 + 报错可读性**：
    `split("_m")[-1]` 的语义是"取最后一段"，它**依赖一个未声明的假设**
    （`_m` 只在末尾出现一次）；今天不触发不代表假设成立。
    这里改成显式锚定尾部 `..._c<c>_m<m>$`，取不到就**抛错**而不是静默取一个数
    —— 判据分档的依据不允许"悄悄地取错值"。
    """
    mo = _RE_CELL_CM.search(stem)
    if mo is None:
        raise ValueError(
            f"格名 {stem!r} 不是 `..._c<c>_m<m>` 形状 ⇒ 无法抽出 (concurrency,"
            f" max_num_seqs)；**不猜**，请核对配置生成器的命名")
    return int(mo.group(1)), int(mo.group(2))


def load_runs(tag: str, root: Path) -> dict[str, list[dict]]:
    """按格名聚合：{cell: [每次重复的读数, ...]}。格名从 run_id 去掉 tag 与 -rN 得到。"""
    runs: dict[str, list[dict]] = {}
    for d in sorted(glob.glob(str(root / f"{tag}-*"))):
        p = Path(d)
        mp = p / "meta.json"
        if not mp.is_file():
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        s = m.get("summary") or {}
        name = p.name
        # 去掉 tag 前缀与 -rN 后缀
        cell = name[len(tag) + 1:] if name.startswith(tag + "-") else name
        cell = cell.split("-")[-1]
        if cell.endswith(("-r1", "-r2", "-r3")):
            cell = cell.rsplit("-r", 1)[0]
        # run_id 形如 S14S-short_chat-fixed_cost_scan_c8_m8[-rN]
        # 取 config_stem（= 最后一段去掉 -rN）
        parts = name.split("-")
        stem = parts[-1]
        if stem in ("r1", "r2", "r3", "r4") and len(parts) >= 2:
            stem = parts[-2]
        runs.setdefault(stem, []).append({
            "dir": name,
            "label": m.get("data_label"),
            "n_requests": s.get("n_requests"),
            "n_slo_satisfied": s.get("n_slo_satisfied"),
            "ttft_p50": (s.get("ttft") or {}).get("p50"),
            "tpot_p50": (s.get("tpot") or {}).get("p50"),
            "queue_p50": (s.get("queue") or {}).get("p50"),
            "prefill_p50": (s.get("prefill") or {}).get("p50"),
            "ttft_cv": (s.get("stability") or {}).get("ttft_robust_cv"),
            "tpot_cv": (s.get("stability") or {}).get("tpot_robust_cv"),
            "wall_ms": s.get("wall_ms"),
        })
    return runs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按 §7d 冻结判据分析扫描批")
    ap.add_argument("root", nargs="?", default="results/raw")
    ap.add_argument("--tag", default="S14S")
    args = ap.parse_args(argv)

    runs = load_runs(args.tag, Path(args.root))
    if not runs:
        print(f"没有找到 {args.tag}-* 产物（在 {args.root}）", file=sys.stderr)
        return 1

    print("=" * 100)
    print(f"{args.tag} 扫描批 · 按 §7d 冻结判据（SLO ttft≤{SLO_TTFT} / tpot≤{SLO_TPOT}；CV≤{CV_LIMIT}）")
    print("=" * 100)

    header = (f"{'格':<34}{'轮':>3}{'label':>12}{'n':>5}{'slo':>5}"
              f"{'ttft_p50':>11}{'tpot_p50':>10}{'queue_p50':>11}{'tpot_cv':>9}  判定")
    print(header)
    print("-" * 100)

    reachable: list[tuple[str, int, int]] = []   # (stem, c, m)
    #: 情形二候选：(stem, c, m) —— `ttft.p50 ≤ SLO` 且 `tpot.p50 ≤ 1.5×SLO`，
    #: **格级**（不是"3 轮全过"）。§7d 第 5 节的情形二按**格**说，
    #: 这里逐次重复记录，最后按格聚合 —— 见下面的 `case2_cells`。
    case2_hits: list[tuple[str, int, int]] = []
    any_partial = False
    for stem in sorted(runs):
        rs = runs[stem]
        ok_all = True
        for r in sorted(rs, key=lambda x: x["dir"]):
            n = r["n_requests"] or 0
            slo = r["n_slo_satisfied"]
            ttft = r["ttft_p50"]
            tpot = r["tpot_p50"]
            cv = r["tpot_cv"]
            rate_ok = (n > 0 and slo == n)
            lat_ok = (ttft is not None and ttft <= SLO_TTFT
                      and tpot is not None and tpot <= SLO_TPOT)
            cv_ok = (cv is not None and cv <= CV_LIMIT)
            ok = rate_ok and lat_ok and cv_ok
            ok_all = ok_all and ok
            verdict = "✅ 可达" if ok else (
                "· 率不足" if not rate_ok else
                "· 时延超" if not lat_ok else "· CV超")
            ttft_s = f"{ttft:.0f}" if ttft is not None else "-"
            tpot_s = f"{tpot:.1f}" if tpot is not None else "-"
            q_s = f"{r['queue_p50']:.0f}" if r["queue_p50"] is not None else "-"
            cv_s = f"{cv:.4f}" if cv is not None else "-"
            slo_s = slo if slo is not None else "-"
            print(f"{stem:<34}{len(rs):>3}{str(r['label']):>12}{n:>5}{slo_s:>5}"
                  f"{ttft_s:>11}{tpot_s:>10}{q_s:>11}{cv_s:>9}  {verdict}")
            # 情形二判据（格级）：只看**时延**两条，**与达标率无关**。
            # ⚠️ 这一点必须显式写出来：`c4_*` 的达标率是 1/64，若把达标率也并进
            #    情形二，它就会落进情形三 —— 而 §7d 的分档只看时延。
            if ttft is not None and ttft <= SLO_TTFT \
                    and tpot is not None and tpot <= 1.5 * SLO_TPOT:
                case2_hits.append((stem, *_cell_cm(stem)))
        # 是否需要 3 轮才算
        n_rounds = len(rs)
        note = "" if n_rounds >= 3 else f"  ⚠️ 只有 {n_rounds} 轮（<3 不出结论）"
        if ok_all and n_rounds >= 3:
            reachable.append((stem, *_cell_cm(stem)))
        elif not ok_all:
            any_partial = True
        print(f"{'':<34}{'':>3}{'':>12}{'':>5}{'':>5}{'':>11}{'':>10}{'':>11}{'':>9}  {'':>6}{note}")

    print("-" * 100)
    print()
    print("=== §7d 第 5 节：机械套用选取规则 ===")
    #: 情形二按**格**聚合（同格 3 次重复都命中才算"该格属情形二"）
    case2_cells = sorted({h[0] for h in case2_hits
                          if sum(1 for x in case2_hits if x[0] == h[0]) >= 3},
                         key=lambda s: (-_cell_cm(s)[0], _cell_cm(s)[1]))

    print(f"【情形一】A（可达集合，率=1.0 且时延达标 且 3 轮 CV≤10%）= "
          f"{[t[0] for t in reachable] or '∅'}")
    if reachable:
        # A 非空 ⇒ 取 concurrency 最大者；并列取 max_num_seqs 更小者
        best = sorted(reachable, key=lambda t: (-t[1], t[2]))[0]
        print(f"  ⇒ **选用 {best[0]}**（concurrency={best[1]} 最大；并列取 m 更小）")
    else:
        print("  （不适用：A = ∅）")

    print()
    print(f"【情形二】格级 `ttft.p50 ≤ {SLO_TTFT:.0f}` 且 "
          f"`tpot.p50 ≤ {1.5 * SLO_TPOT:.0f}`（= 1.5×SLO）的格 = "
          f"{case2_cells or '∅'}")
    if case2_cells:
        print("  ⚠️ 条件为真 ⇒ 按 §7d 应「换更轻负载再试一轮」，"
              "**不得**把超标组合直接当可达用。")
        print("  ⚠️ 但**触发情形二 ≠ 情形一为空**：本批情形一已非空、已按情形一选定；")
        print("     情形二在这里是**并存的信息**，不得据此覆盖情形一的结论。")
    else:
        print("  （条件为假）")

    print()
    print(f"【情形三】全部格 `tpot.p50 > {1.5 * SLO_TPOT:.0f}` = "
          f"{'True' if not case2_cells and not reachable else 'False'}")
    if not case2_cells and not reachable:
        print("  ⇒ 报「SLO 在此负载下不可达」+ **未判**（**不得改 SLO**）")
    else:
        print("  ⇒ 不适用（见情形一/二）")

    print()
    print("⚠️ 边界：本批只判**可达性**，不判服务收益（§7d「三档分离」）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
