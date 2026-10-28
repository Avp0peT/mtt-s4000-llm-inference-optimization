#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""执行优化批次产物分析（S01–S04）：把 `pack_ms` / 快路径计数 / 单变量关系摆成一张表。

为什么单独有这个脚本
--------------------
S04 的判据**不是** TTFT/TPOT，而是「`pack_ms` 是否真的降」+「快路径是否真被走到」。
这两件事都只能从产物的 **`driver_stats`** 里读，而该字段**不在 `meta.json` 顶层**
（顶层没有 `execution_optimizations`；真实路径是
`meta["context"]["extra"]["backend_stats"]["driver_stats"]`）。
踩过的坑：按记忆写 `meta["execution_optimizations"]` ⇒ 恒取到 `None` ⇒
输出的表**全是空值而脚本不报错**（"恒定值假通过"）。
⇒ 本脚本**所有字段路径都从产物现读**（`--dump-paths` 可打印实际形状），
   并在读不到时**显式报 `(缺)`**，不用空值冒充"没效果"。

判据（写进代码，不靠记性）
--------------------------
* **主判据**：`pack_ms_total`（单位 ms，整个 run 的补位/切片搬运总量）。
  `benchmark/real_backend.py` 的文件头定义了它 = "补位与切片的显存搬运，
  **不含任何设备计算**"⇒ 它是 S04 唯一对准的量。
* **"真走到了"判据**：`n_fastpath_steps > 0` 且 `n_fallback_steps` 的
  `fallback_reasons` 之和 == `n_fallback_steps`（分桶完备）。
  ⚠️ 只看 `requested=true` **不算**"走到了"——那是"配了"，不是"生效"。
* **单变量判据**：`switches` 里组合臂相对其基线臂**只多** pkv 两个键。
  脚本自己算差集，不靠人写死。

用法::

    python scripts/analyze_exec_opt_batch.py --tag EXECOPT4
    python scripts/analyze_exec_opt_batch.py --tag EXECOPT4 --dump-paths
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ★ 复用既有解析器（**全项目唯一实现**）：`note` 文本里的 pack/fwd 只能由它解。
from decompose_step_overhead import parse_note  # noqa: E402

#: `driver_stats` 的星号路径。**从产物现读得到**（EXECOPT3 实测），
#: 不要替换成 `meta["execution_optimizations"]`（顶层没有这个键）。
DRIVER_STATS = ("context", "extra", "backend_stats", "driver_stats")

MISSING = "(缺)"


def _dig(obj, path):
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _fmt(v, nd=1):
    if v is None:
        return MISSING
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def load_runs(raw: Path, tag: str) -> dict[str, list[Path]]:
    """把 `<tag>-<workload>-<arm>[-rN]` 按臂名归组（自然序：r0 在前）。

    ⚠️ 臂名可能自带 `-rN` 后缀（`s01_both-r1`）⇒ 必须**先剥 `tag-workload`
    前缀**再解析轮次，否则会把 `s01_both` 与 `s01_both-r1` 当成两个臂。
    """
    pat = re.compile(rf"^{re.escape(tag)}-(?P<wl>.+?)-(?P<arm>.+?)(?:-r(?P<rep>\d+))?$")
    out: dict[str, list[tuple[int, Path]]] = {}
    for d in sorted(raw.glob(f"{tag}-*")):
        if not d.is_dir() or not (d / "meta.json").is_file():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        # 臂名里可能含 `-`（如 `s02_no_empty_cache`）⇒ 用「是否以 -r<digits> 结尾」
        # 判断轮次，而不是贪婪匹配。
        rep = int(m.group("rep")) if m.group("rep") is not None else 0
        out.setdefault(m.group("arm"), []).append((rep, d))
    return {k: [p for _, p in sorted(v)] for k, v in out.items()}


def read_meta(p: Path) -> dict:
    return json.loads((p / "meta.json").read_text(encoding="utf-8"))


def switches_of(meta: dict) -> dict:
    """从 `config.runtime` 里取**已知的执行优化开关**（可选键，缺 = 未设）。

    与 `run_exec_opt_matrix.py::_SWITCH_KEYS` 同源；这里独立列出是为了
    让"产物里到底记了什么"可被本脚本单独核对（不是第二处真相源——
    真相源仍是脚本写盘的那一刻）。
    """
    rt = (meta.get("config") or {}).get("runtime") or {}
    keys = ("logits_last_only", "sample_needed_only", "retire_empty_cache",
            "fuse_qkv", "fuse_gate_up", "linear_fuse_shapes",
            "persistent_kv_slots", "persistent_kv_capacity")
    return {k: rt[k] for k in keys if k in rt}


def telemetry_rows(run_dir: Path) -> list[dict]:
    """逐 step 的 telemetry 解析结果：`{step, prefill_tok, decode, forwards, fwd_ms, pack_ms}`。

    取不到的键**不出现**（而不是填 0）——"没记录"与"记了 0"必须可分。
    """
    tp = run_dir / "telemetry.jsonl"
    if not tp.is_file():
        return []
    out = []
    for line in tp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        d = parse_note(row.get("note"))
        rec = {"step": row.get("step")}
        for k in ("prefill_tok", "decode", "forwards", "fwd_ms", "pack_ms"):
            if k in d:
                rec[k] = d[k]
        out.append(rec)
    return out


def tail_single_decode(rows: list[dict]) -> list[dict]:
    """末尾**连续** `decode == 1` 的步 = 单请求纯 decode 尾段。

    ★ 为什么它才是 S04 的"有效样本"：快路径要求"历史等长 + 纯 decode"，
    连续批处理下只有**只剩一个请求**时才天然满足。EXECOPT4 实测：
    正式窗口 1143 个前向组里只有 **77** 个落在这一段（= 快路径的全部覆盖）。
    ⇒ 用全窗口总量去判 S04，等于把 93% 没走快路径的步混进来**稀释**成本。
    """
    decs = [r.get("decode") for r in rows]
    i = len(decs)
    while i > 0 and decs[i - 1] == 1:
        i -= 1
    return rows[i:]


def coverage_of(driver_stats: dict, warm: dict, total_fastpath: int,
                tel: dict | None = None) -> dict:
    """快路径覆盖率 + **分母口径**。

    `measured_window.pkv_fastpath_measured` 是 harness 自报值（权威）；
    老产物没这个字段时只能**派生**：预热每轮会走 `gen_len - 1` 步纯 decode
    （等长、单组），这些步同样计入快路径 ⇒ 减去 `iterations * (gen_len - 1)`。
    ⚠️ 派生值依赖"预热每步恰好 1 个前向组"这个假设，**必须标注来源**。

    两个分母分别回退：
      * 前向组数 ← `measured_window.n_forwards_measured`，缺则 telemetry 的
        `forwards` 求和（正式段，可靠）；
      * 逻辑步数 ← `measured_window.n_logical_steps_measured`，缺则 telemetry
        的行数（一行 = 一个逻辑步）。
    两者都标在 `denominator_source` 里 —— 报覆盖率必须知道分母是哪来的。
    """
    mw = driver_stats.get("measured_window") or {}
    tel = tel or {}
    fwd_m = mw.get("n_forwards_measured")
    log_m = mw.get("n_logical_steps_measured")
    den_src = "harness"
    if fwd_m is None and tel.get("forwards"):
        fwd_m, den_src = tel["forwards"], "telemetry(forwards 求和)"
    if log_m is None and tel.get("n_steps"):
        log_m = tel["n_steps"]
        if den_src == "harness":
            den_src = "telemetry(行数=逻辑步)"
        else:
            den_src += " + telemetry(行数)"
    if mw.get("pkv_fastpath_measured") is not None:
        fast, source = int(mw["pkv_fastpath_measured"]), "harness"
    else:
        it = warm.get("iterations")
        gl = warm.get("gen_len")
        if it is None or gl is None:
            return {"fast": None, "source": "unavailable"}
        fast = int(total_fastpath) - int(it) * max(0, int(gl) - 1)
        source = "derived(warmup iters*(gen_len-1))"
    return {
        "fast": fast,
        "source": source,
        "denominator_source": den_src,
        "n_forward_groups": fwd_m,
        "n_logical_steps": log_m,
        "cov_of_groups": (fast / fwd_m) if (fast is not None and fwd_m) else None,
        "cov_of_logical": (fast / log_m) if (fast is not None and log_m) else None,
    }


def telemetry_totals(run_dir: Path) -> dict:
    """把逐 step 的 telemetry 求和 ⇒ **只含正式测量段**的 pack/fwd/forwards。

    ★ 为什么必须有这一路：`driver_stats` 的 `pack_ms_total`/`fwd_ms_total`/
    `n_forwards`/`tokens_out` 由 `_drive_warmup_batch` 走到同一个累加器
    （它直接调 `self.forward(...)`）⇒ **包含 warmup**；而 `wall_ms` 不含。
    warmup 轮数由"跑到 target_s"决定 ⇒ **各臂可能不同**（EXECOPT4 实测 2 vs 1），
    于是总量差里混着 warmup 差（一轮 warmup = 117 次前向）。
    telemetry 只记正式段的步 ⇒ 求和即无 warmup 口径。
    """
    tp = run_dir / "telemetry.jsonl"
    if not tp.is_file():
        return {"n_steps": None, "pack_ms": None, "fwd_ms": None, "forwards": None}
    n = 0
    pack = fwd = 0.0
    fwds = 0
    saw_pack = False
    for line in tp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        d = parse_note(row.get("note"))
        if "pack_ms" in d:
            saw_pack = True
            pack += d["pack_ms"]
        if "fwd_ms" in d:
            fwd += d["fwd_ms"]
        if "forwards" in d:
            fwds += int(d["forwards"])
    return {"n_steps": n,
            "pack_ms": pack if saw_pack else None,
            "fwd_ms": fwd if fwd else None,
            "forwards": fwds if fwds else None}


def summarize(arm: str, dirs: list[Path]) -> dict:
    rows = []
    for d in dirs:
        meta = read_meta(d)
        ds = _dig(meta, DRIVER_STATS)
        eo = (ds or {}).get("execution_optimizations") or {}
        # ★ `pkv is None` ⇒ 该 run **根本没有这个机制**（旧批次/未开开关），
        #   与"开了但 0 次回退"是**两件事**。
        #   踩过的形状：把缺失读成 0 ⇒ 报告里看不出"这条臂没测"，而它看起来
        #   像"测了、回退为 0、所以很干净"。⇒ 缺失一律传 `None`，打印时显 `(缺)`。
        pkv = eo.get("persistent_kv_slots")
        has_pkv = isinstance(pkv, dict)
        reasons = (pkv.get("fallback_reasons") or {}) if has_pkv else None
        warm = (ds or {}).get("warmup") or {}
        mw = (ds or {}).get("measured_window") or {}
        trows = telemetry_rows(d)
        tail = tail_single_decode(trows)
        stall_f = max(((r.get("fwd_ms"), r.get("step")) for r in trows
                       if r.get("fwd_ms") is not None), default=(None, None))
        stall_p = max(((r.get("pack_ms"), r.get("step")) for r in trows
                       if r.get("pack_ms") is not None), default=(None, None))
        # 单步"长停顿"：偏离常规步一个数量级。阈值取 1000 ms（常规步 <100 ms）。
        over = sum(1 for r in trows if (r.get("fwd_ms") or 0) > 1000.0
                   or (r.get("pack_ms") or 0) > 1000.0)
        rows.append({
            "dir": d.name,
            "pack_ms_total": (ds or {}).get("pack_ms_total"),
            "fwd_ms_total": (ds or {}).get("fwd_ms_total"),
            "n_forwards": (ds or {}).get("n_forwards"),
            "warmup_iters": warm.get("iterations"),
            "mw": mw,
            "tel": telemetry_totals(d),
            "tail_n": len(tail),
            "tail_pack": sum(r["pack_ms"] for r in tail if "pack_ms" in r)
                         if tail and all("pack_ms" in r for r in tail) else None,
            "tail_fwd": sum(r["fwd_ms"] for r in tail if "fwd_ms" in r)
                        if tail and all("fwd_ms" in r for r in tail) else None,
            "max_fwd_ms": stall_f[0],
            "max_fwd_step": stall_f[1],
            "max_pack_ms": stall_p[0],
            "max_pack_step": stall_p[1],
            "n_long_steps": over,
            "cov": coverage_of(ds or {}, warm, (pkv.get("n_fastpath_steps") or 0)
                               if has_pkv else 0,
                               tel=telemetry_totals(d)),
            "has_pkv": has_pkv,
            "has_pkv": has_pkv,
            "pkv_requested": pkv.get("requested") if has_pkv else None,
            "pkv_effective": pkv.get("effective") if has_pkv else None,
            "n_fastpath": pkv.get("n_fastpath_steps") if has_pkv else None,
            "n_fallback": pkv.get("n_fallback_steps") if has_pkv else None,
            "fallback_reasons": reasons,
            "reasons_sum": sum(reasons.values()) if isinstance(reasons, dict) else None,
            "bytes_alloc": pkv.get("bytes_allocated") if has_pkv else None,
            "switches": switches_of(meta),
        })
    return {"arm": arm, "n": len(rows), "rows": rows}


def decompose_delta(summaries: list[dict], base: str, opt: str) -> dict:
    """把 `pack` 的跨臂差分解成 **尾段 / 非尾段 / 全窗** 三块（可加性自查）。

    ★ 为什么必须分解（2026-10-03 加）
    ---------------------------------
    只报"全窗 +0.415%"或只报"尾段 +70.9%"都**不完整**：
    前者把效应摊薄到看不出，后者不提它占总耗时多少。
    分解同时回答"效应多大""落在哪一段"，并且**可加性核对**能用来自查
    （尾段Δ + 非尾段Δ 应 ≈ 全窗Δ）。

    ⚠️ **两个统计口径都要给，且不可混用**：
      * `ratio_median`      = **逐轮配对比值的中位数** `median(pkv_i / base_i)`
        —— 配对消掉了"这一轮整体偏快/偏慢"，问的是"同一轮内两臂差多少"；
      * `ratio_of_medians`  = **两臂中位数之比** `median(pkv) / median(base)`
        —— 不受轮内配对约束，对单轮离群更敏感。
    两者本批接近（+70.86% vs +70.91%），但**报数时必须写明是哪一个**，
    否则别人复算会对不上。

    返回的 `non_tail_*` 用来回答"S04 有没有顺带影响到**没走快路径**的那些步"——
    这是判断"成本是否只落在机制真正触及的样本上"的关键一格。
    """
    by = {s["arm"]: s for s in summaries}
    if base not in by or opt not in by:
        return {}

    res: dict[str, Any] = {}

    def _series(arm: str) -> dict[str, dict[str, float | None]]:
        """逐轮取 `whole` / `tail` / `non_tail` 三个量（按 `rN` 配对）。"""
        out: dict[str, dict[str, float | None]] = {}
        for r in by[arm]["rows"]:
            tel = r.get("tel") or {}
            whole = tel.get("pack_ms")
            tail = r.get("tail_pack")
            if whole is None:
                continue
            name = r["dir"]
            key = "r" + name.rsplit("-r", 1)[1] if "-r" in name else "r0"
            out[key] = {
                "whole": float(whole),
                "tail": None if tail is None else float(tail),
                "non_tail": None if tail is None else float(whole) - float(tail),
            }
        return out

    bs, os_ = _series(base), _series(opt)
    for label in ("whole", "tail", "non_tail"):
        b = {k: v[label] for k, v in bs.items() if v[label] is not None}
        o = {k: v[label] for k, v in os_.items() if v[label] is not None}
        shared = sorted(set(b) & set(o))
        if not shared:
            continue
        res[f"{label}_base_ms"] = statistics.median([b[k] for k in shared])
        res[f"{label}_opt_ms"] = statistics.median([o[k] for k in shared])
        res[f"{label}_delta_ms"] = res[f"{label}_opt_ms"] - res[f"{label}_base_ms"]
        ratios = [o[k] / b[k] for k in shared if b[k]]
        if ratios:
            res[f"{label}_ratio_median"] = statistics.median(ratios)
            res[f"{label}_ratio_span"] = max(ratios) - min(ratios)
            res[f"{label}_n_paired"] = len(ratios)
            res[f"{label}_n_same_sign"] = sum(1 for x in ratios if x > 1)
        if res[f"{label}_base_ms"]:
            res[f"{label}_ratio_of_medians"] = (res[f"{label}_opt_ms"]
                                                / res[f"{label}_base_ms"])
    # 加性自查：尾段Δ + 非尾段Δ 应 ≈ 全窗Δ（分解是恒等式，残差说明用的是不同统计量）
    if all(f"{k}_delta_ms" in res for k in ("whole", "tail", "non_tail")):
        res["additivity_residual_ms"] = (res["whole_delta_ms"]
                                         - res["tail_delta_ms"]
                                         - res["non_tail_delta_ms"])
    return res


def print_table(summaries: list[dict], dump_paths: bool = False) -> int:
    print("=" * 100)
    print("执行优化批次分析（判据 = pack_ms_total 是否降 + 快路径是否真走到）")
    print("=" * 100)
    rc = 0

    print("\n-- ① 逐 run：driver 总量（**含 warmup**）vs telemetry 求和（**不含**）--")
    hdr = (f"{'run 目录':<44}{'wIt':>4}{'pack_drv':>11}{'pack_tel':>10}"
           f"{'fwd_drv':>11}{'fwd_tel':>10}{'fwd#drv':>8}{'fwd#tel':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        for r in s["rows"]:
            t = r["tel"]
            print(f"{r['dir']:<44}{_fmt(r['warmup_iters']):>4}"
                  f"{_fmt(r['pack_ms_total']):>11}{_fmt(t['pack_ms']):>10}"
                  f"{_fmt(r['fwd_ms_total']):>11}{_fmt(t['fwd_ms']):>10}"
                  f"{_fmt(r['n_forwards']):>8}{_fmt(t['forwards']):>8}")
            if r["n_fallback"] is not None and r["reasons_sum"] is not None \
                    and r["reasons_sum"] != r["n_fallback"]:
                print("      ⚠️ 回退分桶不闭合")
                rc = 1
    # ★ 交叉校验：harness 自报的 `measured_window` 与 telemetry 求和是**两条
    #   独立路径**得到同一个量。两者不一致 ⇒ 其中一条错了，**先查再引用**
    #   （这条纪律踩过的形态：拿"两个来源都写着同一个数"当互证，其实是同源）。
    all_rows = [r for s in summaries for r in s["rows"]]
    have_mw = [r for r in all_rows if r["mw"]]
    print("\n-- ①b 交叉校验：`measured_window`（harness 自报）vs telemetry 求和 --")
    if not have_mw:
        print("  这些产物**没有** `measured_window` 字段（该字段是 EXECOPT4 之后加的）"
              "⇒ 本次只能以 telemetry 为准。")
    else:
        bad = 0
        for r in all_rows:
            mp, tp = r["mw"].get("pack_ms_measured"), r["tel"]["pack_ms"]
            if mp is None or not tp:
                continue
            diff = abs(mp - tp) / tp * 100
            flag = "  ⚠️ 两路不一致" if diff > 2.0 else ""
            print(f"  {r['dir']:<46} mw_pack={mp:>10,.1f} tel_pack={tp:>10,.1f} "
                  f"diff={diff:5.2f}%{flag}")
            if diff > 2.0:
                bad += 1
        if bad:
            rc = 1

    # ★★ ①c 覆盖率：**必须以"正式窗口 + 前向组"为分母**。
    print("\n-- ①c 快路径覆盖率（分母 = 正式窗口；全窗口总量会把它稀释掉）--")
    for s in summaries:
        for r in s["rows"]:
            if not r["has_pkv"]:
                print(f"  {r['dir']:<46} (无该机制：未开 persistent_kv_slots)")
                continue
            c = r["cov"]
            if c.get("fast") is None or c["fast"] < 0:
                print(f"  {r['dir']:<46} fast={MISSING} [{c.get('source')}]")
                rc = 1
                continue
            covg = c["cov_of_groups"]
            print(f"  {r['dir']:<46} fast={c['fast']:>5}  "
                  f"/groups={_fmt(c['n_forward_groups'])}  "
                  f"/logical={_fmt(c['n_logical_steps'])}  "
                  f"cov_groups={('%6.2f%%' % (100 * covg)) if covg else MISSING}  "
                  f"[{c['source']}; 分母={c['denominator_source']}]")
    print("  ⚠️ 分母**必须**是正式窗口（扣预热）：预热每轮会走 `gen_len-1` 步纯 decode，\n"
          "     那些步同样命中快路径 —— 用全窗口当分母会把覆盖率**虚高一倍**。")

    # ★★ ①d 有效样本（单请求纯 decode 尾段）才是判 S04 的地方。
    print("\n-- ①d 有效样本对比：末尾连续 decode==1 的尾段（快路径唯一的覆盖区）--")
    by_arm2 = {s["arm"]: s for s in summaries}
    for s in summaries:
        tails = [r for r in s["rows"] if r["tail_n"]]
        if not tails:
            continue
        ns = {r["tail_n"] for r in tails}
        ps = [r["tail_pack"] for r in tails if r["tail_pack"] is not None]
        fs = [r["tail_fwd"] for r in tails if r["tail_fwd"] is not None]
        pm = statistics.median(ps) if ps else None
        fm = statistics.median(fs) if fs else None
        print(f"  {s['arm']:<26} n_tail={sorted(ns)}  "
              f"tail_packΣ中位={_fmt(pm, 2)} ms  tail_fwdΣ中位={_fmt(fm, 1)} ms")

    # ★★ ①f 用时分解：尾段 / 非尾段 / 全窗（可加性自查 + 两个统计口径都标出）
    #
    # ⚠️ 配对规则**复用 ③ 的那一条**（`added and not removed` = 声明了叠加关系），
    #    不另立第二套判据 —— 两处判据不一致时，读者没法知道该信哪个。
    by_arm0 = {s["arm"]: s for s in summaries}
    for s in summaries:
        if not s["rows"]:
            continue
        cur = s["rows"][0]["switches"]
        for other_name, other in by_arm0.items():
            if other_name == s["arm"] or not other["rows"]:
                continue
            base_sw = other["rows"][0]["switches"]
            added = {k: v for k, v in cur.items()
                     if k not in base_sw or base_sw[k] != v}
            removed = {k: v for k, v in base_sw.items() if k not in cur}
            if not added or removed:
                continue
            d = decompose_delta(summaries, other_name, s["arm"])
            if "whole_delta_ms" not in d:
                continue
            print(f"\n-- ①f 用时分解：{s['arm']} 相对 {other_name}"
                  f"（叠加 {sorted(added)}；尾段 / 非尾段 / 全窗）--")
            print(f"  {'段':<8}{'baseline(ms)':>15}{'opt(ms)':>14}{'Δ(ms)':>11}"
                  f"{'配对比值中位':>15}{'中位数之比':>13}{'同号':>8}")
            for label, human in (("whole", "全窗"), ("tail", "尾段"),
                                 ("non_tail", "非尾段")):
                if f"{label}_delta_ms" not in d:
                    continue
                rm = d.get(f"{label}_ratio_median")
                rom = d.get(f"{label}_ratio_of_medians")
                sign, npar = d.get(f"{label}_n_same_sign"), d.get(f"{label}_n_paired")
                print(f"  {human:<8}{d[f'{label}_base_ms']:>15,.2f}"
                      f"{d[f'{label}_opt_ms']:>14,.2f}"
                      f"{d[f'{label}_delta_ms']:>+11,.2f}"
                      f"{(f'{rm:.4f}' if rm else '—'):>15}"
                      f"{(f'{rom:.4f}' if rom else '—'):>13}"
                      f"{(f'{sign}/{npar}' if sign is not None else '—'):>8}")
            if "additivity_residual_ms" in d:
                r_ms, wb = d["additivity_residual_ms"], (d["whole_base_ms"] or 1.0)
                print(f"  ★ 可加性：尾段Δ + 非尾段Δ = "
                      f"{d['tail_delta_ms'] + d['non_tail_delta_ms']:+,.2f} ms "
                      f"vs 全窗Δ {d['whole_delta_ms']:+,.2f} ms ⇒ 残差 "
                      f"{r_ms:+,.2f} ms（{r_ms / wb * 100:+.3f}%）")
            if d.get("tail_base_ms") and d.get("whole_base_ms"):
                print(f"  ★ **时间权重**（尾段 `pack`Σ / 全窗 `pack`Σ）："
                      f"baseline {d['tail_base_ms'] / d['whole_base_ms'] * 100:.4f}%"
                      f" / opt {d['tail_opt_ms'] / d['whole_opt_ms'] * 100:.4f}%")
            print("  ⚠️ 读数纪律：`配对比值中位` = median(opt_i/base_i)（消轮间漂移）；"
                  "`中位数之比` = median(opt)/median(base)。**口径不同，报数必须写明**。\n"
                  "  ⚠️ `6.74%` 是**前向组数量**覆盖率，**不是**耗时占比；"
                  "解释全窗差要用上面那个**时间权重**。\n"
                  "  ⚠️ 非尾段的小值**不得**写成'无差异'（区间接近不是检验）——"
                  "只能说'远小于尾段'。")

    # ★★ ①e 孤立长停顿：wall 的离群**不是**持续慢态，而是**单步**停顿。
    print("\n-- ①e 孤立长停顿（单步异常）--")
    for r in sorted(all_rows, key=lambda x: -(x["max_fwd_ms"] or 0)):
        if (r["max_fwd_ms"] or 0) <= 1000 and (r["max_pack_ms"] or 0) <= 1000:
            continue
        print(f"  {r['dir']:<46} max_fwd={_fmt(r['max_fwd_ms'])}ms@{r['max_fwd_step']}"
              f"  max_pack={_fmt(r['max_pack_ms'], 2)}ms@{r['max_pack_step']}"
              f"  超 1s 步数={r['n_long_steps']}")
    print("  ⚠️ 判读：若某轮 wall 比常规轮多出的量 ≈ 它那一个停顿步的耗时，\n"
          "     则**吞吐/时延的跨轮差异由该停顿解释**，与机制归因无关；\n"
          "     成因（分配/同步/宿主机暂停/实现交互）**未定位**，不得写成"
          "「持续慢态」或「抽样假象」。")

    # ★ warmup 口径检查：轮数不一致 ⇒ driver 总量**不可跨臂比较**。
    iters = {r["warmup_iters"] for r in all_rows}
    if len(iters) > 1:
        print(f"\n  ⚠️ warmup 轮数在 run 之间**不一致**：{sorted(i for i in iters if i is not None)}"
              "\n     ⇒ `pack_drv`/`fwd_drv`/`fwd#drv` 含 warmup、**跨臂不可比**；"
              "\n     判机制效应只看 `pack_tel`/`fwd_tel`（telemetry = 正式测量段）。")

    print("\n-- ② 臂级汇总（**telemetry 口径**；跨轮极差用于判稳定性）--")
    for s in summaries:
        packs = [r["tel"]["pack_ms"] for r in s["rows"] if r["tel"]["pack_ms"] is not None]
        fwds = [r["tel"]["fwd_ms"] for r in s["rows"] if r["tel"]["fwd_ms"] is not None]
        steps = [r["tel"]["n_steps"] for r in s["rows"] if r["tel"]["n_steps"] is not None]
        fps = [r["n_fastpath"] for r in s["rows"] if r["n_fastpath"] is not None]
        fbs = [r["n_fallback"] for r in s["rows"] if r["n_fallback"] is not None]
        sw = s["rows"][0]["switches"] if s["rows"] else {}
        pack_rng = f"{min(packs):,.1f}–{max(packs):,.1f}" if packs else MISSING
        fwd_rng = f"{min(fwds):,.1f}–{max(fwds):,.1f}" if fwds else MISSING
        print(f"  {s['arm']:<26} n={s['n']} steps={sorted(set(steps))}  "
              f"pack_tel=[{pack_rng}]  fastpath={fps}  fallback={fbs}")
        print(f"      fwd_tel=[{fwd_rng}]")
        print(f"      switches={json.dumps(sw, ensure_ascii=False)}")

    print("\n-- ③ 单变量核对：每个非基线臂相对**它声明叠加的那个基线臂**的差异 --")
    # 只在能配对时比：按"去掉 pkv 键后是否等于另一臂"来判父子关系。
    by_arm = {s["arm"]: s for s in summaries}
    for s in summaries:
        cur = s["rows"][0]["switches"] if s["rows"] else {}
        for other_name, other in by_arm.items():
            if other_name == s["arm"] or not other["rows"]:
                continue
            base = other["rows"][0]["switches"]
            added = {k: v for k, v in cur.items() if k not in base or base[k] != v}
            removed = {k: v for k, v in base.items() if k not in cur}
            if added and not removed:
                print(f"  {s['arm']} 相对 {other_name}：只多 {added}")

    if dump_paths:
        print("\n-- ④ 字段路径自检（防「按记忆写路径」）--")
        any_row = next((r for s in summaries for r in s["rows"]), None)
        print(f"  driver_stats 路径 = {'.'.join(DRIVER_STATS)}")
        print(f"  首个 run = {any_row['dir'] if any_row else MISSING}")
        print(f"  pack_ms_total 读出 = {_fmt(any_row['pack_ms_total']) if any_row else MISSING}")
        if any_row and any_row["pack_ms_total"] is None:
            print("  ⚠️ 读不到 pack_ms_total ⇒ **先核对路径**，"
                  "不要把它当成「这条臂没有 pack 开销」。")
            rc = 1

    print(f"\n退出码 {rc}（0 = 分桶闭合且关键字段可读；1 = 有疑问需人看）")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--raw", default=str(_ROOT / "results" / "raw"))
    ap.add_argument("--dump-paths", action="store_true")
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    runs = load_runs(raw, args.tag)
    if not runs:
        print(f"[analyze] ✗ 在 {raw} 下找不到 tag={args.tag} 的产物", file=sys.stderr)
        return 2
    summaries = [summarize(arm, dirs) for arm, dirs in runs.items()]
    return print_table(summaries, dump_paths=args.dump_paths)


if __name__ == "__main__":
    sys.exit(main())
