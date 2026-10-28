#!/usr/bin/env python3
"""P2 —— token 时间戳口径修复的**前后对照**分析。

背景
----
``benchmark/real_backend.py`` 原先在**采样与设备→主机取回之前**就记 ``ready_s``，
而 ``ready_s`` 被下游用来算 ``ttft_ms`` / ``e2e_ms`` / 逐步间隔（ITL、stall）。
于是这三个量**系统性偏早**，偏早量 = 采样 + 拷回，且不报错、不留痕。

修复后同处记两个时刻：

* ``logits_ready_s`` —— 设备算完（同步返回）那一刻；
* ``ready_s`` —— **token 值已在主机侧可得**那一刻（采样与 D2H 都完成）。

差值即 ``readback_ms``，直接写进 telemetry 的 ``note``。

同一提交还捆绑了第二项改动：写回循环里的 ``token = int(next_ids[i])``
（**逐元素** device sync，发生在 ``pack_ms`` 计时窗内）换成一次
``next_ids.detach().to("cpu").tolist()``。所以本脚本**必须把两件事分开报**，
否则会把"少做 N−1 次同步"的真实收益误记成"口径造成的差异"。

两批产物
--------
* ``P1S3B-*`` 修复**前**口径（note 止于 ``pack_ms=``）
* ``P2-*``    修复**后**口径（note 追加 ``readback_ms=``）

纪律：两批数字**不得并表**（EXPERIMENTS.md §4.3 P2）。本脚本只报差值与量级。

用法::

    python scripts/inspect_timing_scope.py
    python scripts/inspect_timing_scope.py --cells static_c1 static_c8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from ascend_opt.metrics import percentile  # noqa: E402
from inspect_step3_runs import load_telemetry, parse_note  # noqa: E402

DEFAULT_BEFORE = "P1S3B"
DEFAULT_AFTER = "P2"
DEFAULT_CELLS = ("static_c1", "static_c4", "static_c8", "adaptive_c4")


def runs_of(raw: Path, prefix: str, cell: str) -> list[Path]:
    """``<prefix>-phase1_anchor-<cell>`` 及其 ``-r1`` / ``-r2``，按名字排序。"""
    return sorted(p for p in raw.glob(f"{prefix}-phase1_anchor-{cell}*") if p.is_dir())


def stat_line(values: list[float]) -> str:
    if not values:
        return "n=0（无样本）"
    return (
        f"n={len(values):4d}  "
        f"p50={percentile(values, 50):7.3f}  p90={percentile(values, 90):7.3f}  "
        f"p99={percentile(values, 99):7.3f}  max={max(values):8.3f}  "
        f"mean={statistics.fmean(values):7.3f}"
    )


def _readback_rows(run_dir: Path) -> list[tuple[float, float, float, float]]:
    """``[(readback_ms, prefill_tok, decode, fwd_ms)]``；缺键的行**不返回**。"""
    rows, bad = load_telemetry(run_dir)
    if bad:
        print(f"  ⚠ {run_dir.name}: telemetry 有 {bad} 行解析失败（不静默丢弃）")
    out: list[tuple[float, float, float, float]] = []
    for r in rows:
        kv = parse_note(r.get("note"))
        if "readback_ms" not in kv:
            continue
        out.append(
            (
                kv["readback_ms"],
                kv.get("prefill_tok", 0.0),
                kv.get("decode", 0.0),
                kv.get("fwd_ms", 0.0),
            )
        )
    return out


def _seg(rb: float, pre: float, dec: float) -> str:
    if pre > 0 and dec == 0:
        return "prefill_only"
    if pre == 0 and dec > 0:
        return "decode_only"
    if pre > 0 and dec > 0:
        return "mixed"
    return "idle"


# --------------------------------------------------------------------------- #
# A / A2. readback_ms 分布
# --------------------------------------------------------------------------- #

def section_a(raw: Path, after: str, cells: tuple[str, ...]) -> None:
    print("=" * 92)
    print("A. readback_ms —— 修复前被记成 0 的那一段有多长（直接测量，只有修复后能量到）")
    print("=" * 92)
    print("   注意读法：修复后的 TTFT 取的就是 token 可取回那一刻，口径**已经正确**；")
    print("   所以这不是「残余偏差」，而是「此前被错记的区间长度」。")
    for cell in cells:
        dirs = runs_of(raw, after, cell)
        if not dirs:
            print(f"\n[{cell}] 暂无修复后产物")
            continue
        segs: dict[str, list[float]] = {}
        n_rows = n_no_key = n_zero = 0
        for d in dirs:
            rows, _ = load_telemetry(d)
            for r in rows:
                n_rows += 1
                kv = parse_note(r.get("note"))
                if "readback_ms" not in kv:
                    n_no_key += 1
                    continue
                v = kv["readback_ms"]
                if v == 0.0:
                    n_zero += 1
                segs.setdefault(
                    _seg(v, kv.get("prefill_tok", 0.0), kv.get("decode", 0.0)), []
                ).append(v)
                segs.setdefault("all", []).append(v)
        print(f"\n[{cell}]  {len(dirs)} 个 run，telemetry {n_rows} 行；"
              f"缺 readback_ms 键 {n_no_key} 行；readback_ms==0 的 {n_zero} 行")
        if n_no_key:
            print("  ⚠ 缺键说明这批里混进了**修复前**口径的产物，下面的统计不可用")
        for name in ("prefill_only", "decode_only", "mixed", "all"):
            if name in segs:
                print(f"    {name:<13} {stat_line(segs[name])}")


def section_a2(raw: Path, after: str, cells: tuple[str, ...]) -> None:
    print()
    print("=" * 92)
    print("A2. readback_ms 随批内序列数（该步 decode 计数，作 batch size 代理）的变化")
    print("=" * 92)
    print("   这一节回答：批量变大时，一次性 D2H 会不会变贵？")
    for cell in cells:
        by_n: dict[int, list[float]] = {}
        for d in runs_of(raw, after, cell):
            for rb, pre, dec, _ in _readback_rows(d):
                if pre > 0:
                    continue  # 只有纯 decode 步，N 才正好等于 decode 计数
                by_n.setdefault(int(dec), []).append(rb)
        if not by_n:
            continue
        print(f"\n[{cell}]")
        for n in sorted(by_n):
            v = by_n[n]
            print(f"    N={n:<3} n={len(v):4d}  p50={percentile(v, 50):7.3f}  "
                  f"max={max(v):7.3f}")


# --------------------------------------------------------------------------- #
# B. readback 相对设备计算的占比
# --------------------------------------------------------------------------- #

def section_b(raw: Path, after: str, cells: tuple[str, ...]) -> None:
    print()
    print("=" * 92)
    print("B. readback 相对设备计算（fwd_ms）的占比 —— 它是不是「可忽略的小量」")
    print("=" * 92)
    for cell in cells:
        ratios: list[float] = []
        rb_dec: list[float] = []
        fwd_dec: list[float] = []
        for d in runs_of(raw, after, cell):
            for rb, pre, dec, fwd in _readback_rows(d):
                if fwd <= 0:
                    continue
                ratios.append(rb / fwd * 100.0)
                if pre == 0 and dec > 0:
                    rb_dec.append(rb)
                    fwd_dec.append(fwd)
        if not ratios:
            print(f"\n[{cell}] 无可用样本，**本项未检验**")
            continue
        print(f"\n[{cell}] readback/fwd 占比（全部 step，单位 %）")
        print(f"    {stat_line(ratios)}")
        if rb_dec:
            m_rb, m_fwd = percentile(rb_dec, 50), percentile(fwd_dec, 50)
            print(f"    纯 decode step：readback 中位 {m_rb:.3f} ms，"
                  f"fwd 中位 {m_fwd:.3f} ms，占比 {m_rb / m_fwd * 100:.2f}%")


# --------------------------------------------------------------------------- #
# C. 请求级指标的前后对比
# --------------------------------------------------------------------------- #

def req_metrics(run_dir: Path, key: str) -> list[float]:
    p = run_dir / "requests.jsonl"
    if not p.exists():
        return []
    out: list[float] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = d.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def section_c(raw: Path, before: str, after: str, cells: tuple[str, ...]) -> None:
    print()
    print("=" * 92)
    print("C. 请求级 TTFT / E2E / TPOT 前后对比（**只报差值，不并表**）")
    print("=" * 92)
    print("   两批相隔约 11 小时，环境漂移与口径偏差混在差值里；口径量级以 A 节为准。")
    for cell in cells:
        b_dirs, a_dirs = runs_of(raw, before, cell), runs_of(raw, after, cell)
        if not b_dirs or not a_dirs:
            print(f"\n[{cell}] 两批未齐，**本项未检验**")
            continue
        print(f"\n[{cell}]  前 {len(b_dirs)} 个 run / 后 {len(a_dirs)} 个 run")
        for key in ("ttft_ms", "prefill_exec_ms", "e2e_ms", "tpot_ms"):
            b: list[float] = []
            a: list[float] = []
            for d in b_dirs:
                b += req_metrics(d, key)
            for d in a_dirs:
                a += req_metrics(d, key)
            if not b or not a:
                continue
            print(f"    {key:<17} 前 {stat_line(b)}")
            print(f"    {'':<17} 后 {stat_line(a)}")
            d50 = percentile(a, 50) - percentile(b, 50)
            d90 = percentile(a, 90) - percentile(b, 90)
            print(f"    {'':<17} Δp50={d50:+.3f} ms "
                  f"({d50 / percentile(b, 50) * 100:+.3f}%)   Δp90={d90:+.3f} ms")


# --------------------------------------------------------------------------- #
# D. 版本溯源
# --------------------------------------------------------------------------- #

def section_d(raw: Path, before: str, after: str, cells: tuple[str, ...]) -> None:
    print()
    print("=" * 92)
    print("D. 版本溯源：修复前批次 git_commit=NOT_MEASURED，修复后批次应钉到部署戳")
    print("=" * 92)
    for prefix in (before, after):
        seen: dict[str, str] = {}
        for cell in cells:
            dirs = runs_of(raw, prefix, cell)
            if not dirs:
                continue
            r = json.loads((dirs[0] / "meta.json").read_text(encoding="utf-8")).get(
                "reproducibility", {}
            )
            seen[cell] = (
                f"git_commit={r.get('git_commit')!r} source={r.get('git_source')!r} "
                f"dirty={r.get('git_dirty')!r}\n"
                f"                  model_rev={r.get('model_revision')!r} "
                f"rev_resolved={r.get('model_revision_resolved')!r}"
            )
        if seen:
            print(f"\n[{prefix}]")
            for cell, s in seen.items():
                print(f"    {cell:<13} {s}")

    print()
    # 三态检查。**先确认输入非空**：一个 run 都没读到时"零个坏样本"与"全好"
    # 打印出来一模一样——空输入会伪装成合格结果。
    unresolved: list[str] = []
    resolved_false: list[str] = []
    n = 0
    for cell in cells:
        for d in runs_of(raw, after, cell):
            n += 1
            v = json.loads((d / "meta.json").read_text(encoding="utf-8")).get(
                "reproducibility", {}
            ).get("model_revision_resolved")
            if v is None:
                unresolved.append(d.name)
            elif v is False:
                resolved_false.append(d.name)
    if n == 0:
        print("    ✗ 没读到任何修复后的 run —— 本项**未检验**（不是'通过'）")
    elif unresolved:
        print(f"    ✗ model_revision_resolved 为 None（没查）的 run：{unresolved}")
    elif resolved_false:
        print(f"    ⚠ 检查了 {n} 个 run；其中为 False（只是引用而非权重 commit）："
              f"{resolved_false}")
    else:
        print(f"    ✓ 检查了 {n} 个 run，model_revision_resolved 均为 True"
              f"（解析到权重 commit）")


# --------------------------------------------------------------------------- #
# E. 守恒量自校验
# --------------------------------------------------------------------------- #

def section_e(raw: Path, before: str, after: str, cells: tuple[str, ...]) -> None:
    print()
    print("=" * 92)
    print("E. 守恒量自校验：每个 run 实际 prefill 的 token 总数（负载锚点应为 16384）")
    print("=" * 92)
    n = 0
    bad = 0
    for prefix in (before, after):
        for cell in cells:
            for d in runs_of(raw, prefix, cell):
                rows, _ = load_telemetry(d)
                tot = sum(parse_note(r.get("note")).get("prefill_tok", 0.0) for r in rows)
                n += 1
                if abs(tot - 16384.0) > 1e-6:
                    bad += 1
                    print(f"    ⚠ {d.name:<46} prefill_total={tot:9.1f}（≠16384）")
    if n == 0:
        print("    ✗ 没有读到任何 run —— 本项**未检验**")
    else:
        print(f"    ✓ 检查了 {n} 个 run，全部 prefill_total=16384"
              if bad == 0 else f"    ✗ {n} 个 run 中 {bad} 个不满足守恒")


# --------------------------------------------------------------------------- #
# F. 修复对报告值的总作用 + 被错记区间的直接测量
# --------------------------------------------------------------------------- #
#
# 先把三个量的关系摆清，避免把两个不同的东西混成一个：
#
#   ``readback_ms`` = t(token 可取回) − t(logits 算完) = 采样 + D2H。
#
#   **它不是"修复后的残余偏差"**。修复后的 TTFT 取的就是 token 可取回那一刻，
#   口径已经正确；``readback_ms`` 只是**此前被记错的区间有多长**。
#
#   ``Δ报告`` = 报告值(后) − 报告值(前)。它同时含两项：
#     (i)  口径修正 —— 前批把"logits 算完"当成了 token 可得；
#     (ii) 真实提速 —— 前批的逐元素 ``int(next_ids[i])`` 被换成一次批量 D2H。
#   两批代码在**同一步做的工作量不同**，所以二者**不可分**；把 Δ报告 当成
#   纯口径量会高估口径修正。故本节只报 Δ报告 与本批的重复漂移，判"是否可分辨"。

def _per_run_p50(dirs: list[Path], key: str) -> list[float]:
    out = []
    for d in dirs:
        v = req_metrics(d, key)
        if v:
            out.append(percentile(v, 50))
    return out


def section_f(raw: Path, before: str, after: str, cells: tuple[str, ...]) -> None:
    print()
    print("=" * 92)
    print("F. 修复对报告值的总作用（含口径修正 + 真实提速，**不可分**）")
    print("=" * 92)
    print("   判读规则：|Δ报告| 若不大于本批重复漂移，则该格的效果**不可分辨**——")
    print("   此时拒绝给数，而不是给一个数。")
    for cell in cells:
        b_dirs, a_dirs = runs_of(raw, before, cell), runs_of(raw, after, cell)
        if not b_dirs or not a_dirs:
            print(f"\n[{cell}] 两批未齐，**本项未检验**")
            continue
        rbs: list[float] = []
        for d in a_dirs:
            rbs += [rb for rb, _, _, _ in _readback_rows(d)]
        if not rbs:
            print(f"\n[{cell}] 修复后批次无 readback_ms 键，**本项未检验**")
            continue
        b_post = percentile(rbs, 50)

        print(f"\n[{cell}]")
        print(f"    被错记区间的直接测量：readback_ms p50 = {b_post:.3f} ms"
              f"（修复前这一整段都被记成了 0）")
        for key in ("ttft_ms", "e2e_ms", "tpot_ms"):
            prev = _per_run_p50(b_dirs, key)
            post = _per_run_p50(a_dirs, key)
            if not prev or not post:
                continue
            t_pre, t_post = percentile(prev, 50), percentile(post, 50)
            d_rep = t_post - t_pre
            drift = max(max(prev) - min(prev), max(post) - min(post))
            rel = d_rep / t_pre * 100.0
            if abs(d_rep) <= drift:
                verdict = f"不可分辨（≤ 批内漂移 {drift:.3f} ms）"
            else:
                verdict = f"可分辨（> 批内漂移 {drift:.3f} ms）"
            print(f"    {key:<9} 前 {t_pre:9.3f} → 后 {t_post:9.3f} ms   "
                  f"Δ报告 {d_rep:+7.3f} ms ({rel:+.3f}%)   {verdict}")


# --------------------------------------------------------------------------- #
# G. pack_ms 前后对比：逐元素同步原本记在哪里
# --------------------------------------------------------------------------- #

def section_g(raw: Path, before: str, after: str, cells: tuple[str, ...]) -> None:
    print()
    print("=" * 92)
    print("G. pack_ms 前后对比 —— 修复把逐元素 `int(next_ids[i])` 从写回段里拿走了")
    print("=" * 92)
    print("   修复前：逐元素 device sync 在写回循环内 ⇒ 计入 pack_ms；")
    print("           而 ready_s 记在采样**之前**，所以这段代价不体现在 TTFT 上。")
    print("   修复后：一次批量 .to('cpu') 落在 logits_ready 与 ready 之间 ⇒ 计入")
    print("           readback_ms，并**真的**计入 TTFT/E2E。")
    print("   所以这里的 Δpack 是「少做 N−1 次同步」的真实收益（与口径无关），")
    print("   而被移到 readback_ms 的那部分 D2H 是真被计入时延的代价。二者要分开读。")
    for cell in cells:
        row: list[float | None] = []
        for prefix in (before, after):
            vals: list[float] = []
            for d in runs_of(raw, prefix, cell):
                rows, _ = load_telemetry(d)
                for r in rows:
                    kv = parse_note(r.get("note"))
                    if "pack_ms" in kv and kv.get("prefill_tok", 0.0) == 0:
                        vals.append(kv["pack_ms"])
            row.append(percentile(vals, 50) if vals else None)
        b, a = row
        if b is None or a is None:
            print(f"\n[{cell}] 两批未齐，**本项未检验**")
            continue
        print(f"\n[{cell}] 纯 decode step 的 pack_ms p50：前 {b:.3f} ms  后 {a:.3f} ms"
              f"  Δ={a - b:+.3f} ms")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=None, help="产物根（默认 results/raw）")
    ap.add_argument("--before", default=DEFAULT_BEFORE, help=f"修复前 run-id 前缀（默认 {DEFAULT_BEFORE}）")
    ap.add_argument("--after", default=DEFAULT_AFTER, help=f"修复后 run-id 前缀（默认 {DEFAULT_AFTER}）")
    ap.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS),
                    help=f"要比较的格（默认 {' '.join(DEFAULT_CELLS)}）")
    args = ap.parse_args(argv)

    raw = Path(args.raw) if args.raw else _ROOT / "results" / "raw"
    if not raw.exists():
        print(f"找不到产物根：{raw}", file=sys.stderr)
        return 2
    cells = tuple(args.cells)
    section_a(raw, args.after, cells)
    section_a2(raw, args.after, cells)
    section_b(raw, args.after, cells)
    section_c(raw, args.before, args.after, cells)
    section_d(raw, args.before, args.after, cells)
    section_e(raw, args.before, args.after, cells)
    section_f(raw, args.before, args.after, cells)
    section_g(raw, args.before, args.after, cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
