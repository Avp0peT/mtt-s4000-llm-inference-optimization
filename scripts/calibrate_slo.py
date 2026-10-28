#!/usr/bin/env python3
"""把 ``slo.tpot_ms`` 从"够不着"标定到"能分辨"。

为什么需要它
------------
``run.slo.tpot_ms`` 从 P1S3 起一直是 **50 ms**，而 7B fp16 在 MTT S4000 上的
逐请求 TPOT 实测落在 60~160 ms（随并发变化）⇒ **没有任何请求达标** ⇒ 表格里的
``slo_satisfaction_rate`` / ``slo_goodput_*`` 三列恒为 ``0.000``。

这三列的问题**不是"达标率低"，是"判据无效"**：一个对所有配置都给 0 的指标不携带
任何信息，把它放进对照表会让人误以为"两种配置都很差"。P1S3B 只把
``controller.target_tpot_ms`` 改成了 110，**那是控制目标，不是 SLO 判据**——
两者是不同的字段、不同的用途（见 README §6）。

本脚本做什么、不做什么
----------------------
**做**：从已登记的产物里读出**逐请求**的 TPOT/TTFT，给出（a）各格分布，
（b）一组候选阈值下各格的达标率，用来挑一个"落在实测区间内部、能分辨配置"的值。

**不做**：不写任何配置，不给"推荐值"这种需要人来拍板的东西。阈值一旦定下来就是
**跨实验约定**（README：SLO 判定口径跨实验必须保持一致），改它是一次单变量改动，
应当被登记，而不是由脚本自动落盘。

三条容易搞错的口径
------------------
1. **逐请求，不是汇总分位**。``SLO.satisfied_by``（``src/ascend_opt/metrics.py``）
   对每个请求单独判 ``m.tpot_ms > 阈值``。所以标定要看**逐请求**分布；用
   ``summary.tpot.p50`` 代替会得到不同的数（那 8 个请求的 p50 不等于逐请求分布）。
2. **SLO 是"与"**。同时配了 ttft 与 tpot 时，**两个都满足**才算达标。
   所以 ``ttft_ms`` 会参与：某格 TPOT 合格但 TTFT 不合格，仍然不达标。
3. **分母是全部提交**（含失败/未完成），不是成功数（``metrics.py`` 注释）。

用法::

    python scripts/calibrate_slo.py --raw results/raw --prefix P1S3B
    python scripts/calibrate_slo.py --raw results/raw --prefix P1S3B,P2 \\
        --candidates 80,90,100,110,120
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.metrics import percentile  # noqa: E402

DEFAULT_RAW = _ROOT / "results" / "raw"

#: 当前两种配置里的 ttft 判据。标定 tpot 时把 ttft 一并代入，
#: 因为达标是"与"的关系——只调 tpot 而 ttft 卡着，达标率照样可能是 0。
DEFAULT_TTFT_SLO_MS = 2000.0

#: 候选阈值默认取实测区间（约 60~160）内部与外缘若干点。**含区间外点**是有意的：
#: 区间外点会让达标率退化成全 0 或全 1，正好用来说明"为什么不能取它"。
DEFAULT_CANDIDATES = (50.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0, 160.0)


def _iter_run_dirs(raw: Path, prefixes: Sequence[str]) -> list[tuple[str, Path]]:
    """按 ``(批次前缀, run 目录)`` 产出。

    只取**首轮**（目录名不以 ``-rN`` 结尾）：三次重复的逐请求值几乎逐字相同
    （负载与静态配置都固定），多取只是把同一组数重复打印。跨重复的差异
    由 ``repeats.csv`` 的离散度负责，不属于本脚本的口径。
    """
    out: list[tuple[str, Path]] = []
    for d in sorted(p for p in raw.iterdir() if p.is_dir()):
        name = d.name
        batch, _, rest = name.partition("-")
        if batch not in prefixes:
            continue
        # 排除重复轮次：形如 ...-c4-r1
        tail = rest.rsplit("-", 1)[-1]
        if len(tail) == 2 and tail[0] == "r" and tail[1].isdigit():
            continue
        if not (d / "requests.jsonl").exists():
            continue
        out.append((batch, d))
    return out


def load_requests(run_dir: Path) -> list[dict[str, Any]]:
    """读一个 run 的逐请求记录。空文件/坏行**显式报错**，不静默跳过。

    静默跳过会让"这一格一个请求都没读到"变成"这一格没有不达标的请求"——
    正是本项目反复出现的"空输入伪装成合格结果"。
    """
    recs: list[dict[str, Any]] = []
    path = run_dir / "requests.jsonl"
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    return recs


def _stat(values: Sequence[float]) -> dict[str, float]:
    vs = sorted(values)
    return {
        "n": float(len(vs)),
        "min": vs[0],
        "p50": percentile(vs, 50),
        "p90": percentile(vs, 90),
        "p95": percentile(vs, 95),
        "p99": percentile(vs, 99),
        "max": vs[-1],
    }


def _fmt(s: dict[str, float]) -> str:
    return (
        f"{s['min']:.1f}/{s['p50']:.1f}/{s['p90']:.1f}/{s['p99']:.1f}/{s['max']:.1f}"
    )


_ARM_ORDER = {"adaptive": 0, "static": 1, "static_chunk": 2}


def _split_cell(cell: str) -> tuple[int, str]:
    """把 ``phase1_anchor-static_chunk_c4`` 拆成 ``(4, 'static_chunk')``。

    排序键用**并发**优先：这样表里是按并发递增排的，"达标率随并发下滑"这条
    梯度一眼可见；按字母排会把 adaptive / static / static_chunk 各自成块，
    反而看不出梯度。
    """
    import re

    m = re.search(r"_c(\d+)$", cell)
    conc = int(m.group(1)) if m else -1
    rest = cell[: m.start()] if m else cell
    # 去掉批次内的工作负载前缀（phase1_anchor-），只留臂名
    rest = rest.split("-", 1)[1] if "-" in rest else rest
    return conc, rest


def _label(cell: str) -> str:
    """短标签：``phase1_anchor-static_chunk_c4`` → ``stchunk_c4``。"""
    conc, arm = _split_cell(cell)
    short = {"adaptive": "adap", "static": "stat", "static_chunk": "schu"}.get(arm, arm[:4])
    return f"{short}_c{conc}"


def satisfaction_rate(
    recs: Sequence[dict[str, Any]], *, ttft_ms: float | None, tpot_ms: float | None
) -> dict[str, Any]:
    """按 ``SLO.satisfied_by`` 的口径重算达标率。

    分子 = 成功 **且** 所有已配置阈值都满足的请求数；分母 = **全部提交**请求数。
    时延字段为 ``None``（不可测）同样不算达标——**不可测 ≠ 达标**。
    """
    n_total = len(recs)
    n_sat = 0
    for r in recs:
        if not r.get("success"):
            continue
        ok = True
        if ttft_ms is not None:
            v = r.get("ttft_ms")
            if v is None or v > ttft_ms:
                ok = False
        if ok and tpot_ms is not None:
            v = r.get("tpot_ms")
            if v is None or v > tpot_ms:
                ok = False
        if ok:
            n_sat += 1
    return {
        "n_total": n_total,
        "n_sat": n_sat,
        "rate": (n_sat / n_total) if n_total else None,
    }


def attribution(
    recs: Sequence[dict[str, Any]], *, ttft_ms: float | None, tpot_ms: float | None
) -> dict[str, int]:
    """达标率**为什么**是那个数：把不达标的请求按"哪个判据卡住"分堆。

    为什么必须拆开：SLO 是**与**关系，一个 4/8 既可能是"一半请求 TPOT 超标"，
    也可能是"TTFT 超标"（``adap_c4`` 的 TTFT p99 就 > 2000）。两者的研究含义完全不同
    ——前者指向 decode 调度，后者指向 prefill/分块。只看一个合计数会把依据认错。
    """
    c = {"sat": 0, "fail_ttft_only": 0, "fail_tpot_only": 0, "fail_both": 0, "fail_outcome": 0}
    for r in recs:
        if not r.get("success"):
            c["fail_outcome"] += 1
            continue
        bad_t = False
        if ttft_ms is not None:
            v = r.get("ttft_ms")
            bad_t = v is None or v > ttft_ms
        bad_p = False
        if tpot_ms is not None:
            v = r.get("tpot_ms")
            bad_p = v is None or v > tpot_ms
        if bad_t and bad_p:
            c["fail_both"] += 1
        elif bad_t:
            c["fail_ttft_only"] += 1
        elif bad_p:
            c["fail_tpot_only"] += 1
        else:
            c["sat"] += 1
    return c


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="slo.tpot_ms 标定：逐请求分布 + 候选阈值达标率")
    ap.add_argument("--raw", default=str(DEFAULT_RAW), help="产物根目录（results/raw）")
    ap.add_argument(
        "--prefix",
        default="P1S3B",
        help="批次前缀，逗号分隔（例：P1S3B,P2）。默认只取 P1S3B",
    )
    ap.add_argument(
        "--ttft-slo",
        type=float,
        default=DEFAULT_TTFT_SLO_MS,
        help=f"代入的 ttft 判据（默认 {DEFAULT_TTFT_SLO_MS:g}）；达标是「与」的关系",
    )
    ap.add_argument(
        "--candidates",
        default=",".join(f"{c:g}" for c in DEFAULT_CANDIDATES),
        help="候选 tpot 阈值（毫秒），逗号分隔",
    )
    ap.add_argument(
        "--explain",
        type=float,
        default=None,
        help="在该 tpot 阈值下打印逐格失败归因（哪个判据卡住）",
    )
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    if not raw.is_dir():
        print(f"标定失败：产物根目录不存在 {raw}", file=sys.stderr)
        return 2

    prefixes = [p.strip() for p in args.prefix.split(",") if p.strip()]
    candidates = [float(c) for c in args.candidates.split(",") if c.strip()]

    runs = _iter_run_dirs(raw, prefixes)
    if not runs:
        # 空输入必须显形：**不是**"全部达标"，是"没读到东西、本项未检验"。
        print(
            f"标定失败：{raw} 下找不到批次 {prefixes} 的 run（需含 requests.jsonl）。",
            file=sys.stderr,
        )
        return 3

    # ---- 逐请求分布 ----
    per_cell: dict[str, dict[str, Any]] = {}
    all_tpot: list[float] = []
    all_ttft: list[float] = []
    #: 块级失败的格。**必须显形并让进程非零退出**：一格读不到请求，
    #: 在表里与"这一格的请求全部达标"长得一样（都缺行/缺列）。
    #: 局部 catch 只是为了让其余格照常出数，**不代表这一格算通过**。
    empty_cells: list[str] = []
    for batch, d in runs:
        recs = load_requests(d)
        if not recs:
            print(f"  ✗ {d.name}: requests.jsonl 为空 —— 本格未检验", file=sys.stderr)
            empty_cells.append(d.name)
            continue
        cell = d.name[len(batch) + 1 :]
        tpot = [float(r["tpot_ms"]) for r in recs if r.get("tpot_ms") is not None]
        ttft = [float(r["ttft_ms"]) for r in recs if r.get("ttft_ms") is not None]
        per_cell[cell] = {"batch": batch, "dir": d.name, "recs": recs, "tpot": tpot, "ttft": ttft}
        all_tpot.extend(tpot)
        all_ttft.extend(ttft)

    if not all_tpot:
        print("标定失败：所有 run 都没有可用的 tpot_ms 值。", file=sys.stderr)
        return 4

    print("=" * 100)
    print(f"逐请求 TPOT / TTFT 分布（批次：{','.join(prefixes)}；首轮；仅含成功请求的时延）")
    print("=" * 100)
    print(f"{'格':<12}{'n':>4}  {'TPOT min/p50/p90/p99/max':>30}  {'TTFT min/p50/p90/p99/max':>30}")
    for cell in sorted(per_cell, key=_split_cell):
        e = per_cell[cell]
        print(
            f"{_label(cell):<12}{len(e['tpot']):>4}  {_fmt(_stat(e['tpot'])):>30}  "
            f"{_fmt(_stat(e['ttft'])):>30}"
        )
    pooled_t = _stat(all_tpot)
    print("-" * 100)
    print(f"{'合并（所有格）':<24}{len(all_tpot):>4}  {_fmt(pooled_t):>30}")
    print(
        f"  合并逐请求 TPOT 区间：[{pooled_t['min']:.1f}, {pooled_t['max']:.1f}] ms；"
        f"p50={pooled_t['p50']:.1f}"
    )
    print(
        f"  当前 slo.tpot_ms = 50.0 ⇒ "
        f"{'低于实测下界，恒不可达' if pooled_t['min'] > 50 else '落在区间内'}"
    )

    # ---- 候选阈值 → 达标率 ----
    print()
    print("=" * 100)
    print(
        f"候选阈值下的达标率（分母 = 全部提交；ttft 判据固定为 {args.ttft_slo:g} ms；"
        f"'—' = 该格无数据）"
    )
    print("=" * 100)
    cells = sorted(per_cell, key=_split_cell)
    header = f"{'tpot阈值':>9} | " + " ".join(f"{_label(c):>11}" for c in cells) + f" | {'全格':>14}"
    print(header)
    print("-" * len(header))
    for cand in candidates:
        parts = []
        for c in cells:
            recs = per_cell[c]["recs"]
            r = satisfaction_rate(recs, ttft_ms=args.ttft_slo, tpot_ms=cand)
            parts.append(f"{r['n_sat']}/{r['n_total']}={r['rate']:.3f}".rjust(11))
        all_recs = [rr for c in cells for rr in per_cell[c]["recs"]]
        rt = satisfaction_rate(all_recs, ttft_ms=args.ttft_slo, tpot_ms=cand)
        print(f"{cand:>9.1f} | " + " ".join(parts) + f" | {rt['n_sat']}/{rt['n_total']}={rt['rate']:.3f}")

    print()
    print("读法：候选阈值要落在**逐请求 TPOT 区间内部**，各格达标率才会出现 0 与 1 之间的")
    print("      差异（可分辨）；取区间以外的值会让整列退化成全 0 或全 1（零信息量）。")
    print("      '全格'列是把所有格的请求混在一起 —— 它随阈值单调，不代表分辨能力，")
    print("      分辨能力看各格列是否分档。")
    if args.explain is not None:
        print()
        print("=" * 100)
        print(
            f"失败归因（tpot 阈值 = {args.explain:g} ms，ttft 判据 = {args.ttft_slo:g} ms）"
            f"—— SLO 是「与」，下面拆开「是哪个判据卡住」"
        )
        print("=" * 100)
        hdr = (
            f"{'格':<12}{'达标':>5}{'仅TTFT超':>9}{'仅TPOT超':>9}{'两者都超':>9}{'未成功':>8}"
        )
        print(hdr)
        print("-" * len(hdr))
        tot = {"sat": 0, "fail_ttft_only": 0, "fail_tpot_only": 0, "fail_both": 0, "fail_outcome": 0}
        for c in cells:
            a = attribution(per_cell[c]["recs"], ttft_ms=args.ttft_slo, tpot_ms=args.explain)
            for k in tot:
                tot[k] += a[k]
            print(
                f"{_label(c):<12}{a['sat']:>5}{a['fail_ttft_only']:>9}"
                f"{a['fail_tpot_only']:>9}{a['fail_both']:>9}{a['fail_outcome']:>8}"
            )
        print("-" * len(hdr))
        print(
            f"{'合计':<12}{tot['sat']:>5}{tot['fail_ttft_only']:>9}"
            f"{tot['fail_tpot_only']:>9}{tot['fail_both']:>9}{tot['fail_outcome']:>8}"
        )
        print()
        print("读法：'仅TTFT超' 指向 prefill/分块，'仅TPOT超' 指向 decode 调度。")
        print("      两者都超 ⇒ 单一判据的改动救不回来。未成功 ⇒ 与阈值无关，属于结局账目。")

    print()
    print("⚠️ 本脚本只报分布与候选结果，**不写配置**。slo.tpot_ms 是跨实验约定，")
    print("   一旦改动就是一次单变量改动，须登记进 EXPERIMENTS.md。")
    if empty_cells:
        # 有格没数 ⇒ 上面的表是**残缺**的，不能当作"该格全达标"。
        print()
        print(
            f"标定不完整：{len(empty_cells)} 个 run 读不到请求（{', '.join(empty_cells)}）——"
            f"上表缺这些格，**不是**它们全达标。",
            file=sys.stderr,
        )
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
