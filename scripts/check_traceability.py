#!/usr/bin/env python3
"""**追溯链自查**（第 16 步准备）：把"机械可查"的部分自动化，剩下的列成人工清单。

第 16 步要求"检查原始结果、实验登记、配置、源码与图表的追溯链"。
但不是所有检查都能自动化 —— 本脚本**明确区分**两类：

**机械可查（本脚本做）**
1. 每个 `MEASURED` 产物是否带齐溯源字段（`data_label` / `git_commit` / `dirty` /
   `recorded_at` / `config` / 判据）；
2. **"配置声称的臂"与"实际跑出来的行为"是否一致** —— 实测踩到过：配置写 `kind=slo_weave`
   而 arm 根本没接进真机路径 ⇒ 产物**被误标**、却看起来完全正常。判据：
   `controller.kind != threshold` 的 run，若 `controller_trace` 里**没有对应的决策条目**，就是**误标**；
3. `TEST_DATA`（dry-run）产物是否**混进了**结论来源（`results/raw` 里带结论指针的目录不该是 TEST_DATA）；
4. 同一批（同一 tag）内是否**混了不同 `git_commit` / `dirty`**（验收第 5 条禁止）；
5. **`MEASURED` 批次是否已登记进 `EXPERIMENTS.md`**（2026-09-26 新增）。
   ⚠️ 这一条原先写在本文件的"只能人工"清单里，**但它其实机械可查**：导出闸门
   （`benchmark/preflight.py` 的检查 4）早就在验，只是**只有跑导出时才会响**
   —— 实测踩到：`S14` 全族**一条登记行都没有**，却没人发现，因为它从没被导出过。
   ⇒ 判定**复用同一份实现**（`preflight.registered_in_table`）与同一个粒度（批次 `base_run_id`）。

**只能人工（本脚本列出待办，不假装能自动判）**
* "图表能否由脚本重建"（看 `results/figures/*.source.txt` 是否存在且指向脚本）；
* "撤回的结论是否已传播到**所有**引用处"（只能靠搜关键词 + 人读）。

用法::

    python scripts/check_traceability.py --tags S14B1 S14B2 S14C1 S13COSTINB
    python scripts/check_traceability.py --all-measured
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ★ 复用导出闸门的判定与粒度，**不另写一套**：
#   两套口径的话，松的那套会静默放行（导出时被拦、自查却绿灯）。
from ascend_opt.grouping import base_run_id              # noqa: E402
from preflight import registered_in_table                # noqa: E402


def _runs(substr: str | None) -> list[dict[str, Any]]:
    out = []
    pat = str(_ROOT / "results" / "raw" / (f"*{substr}*" if substr else "*") / "meta.json")
    for f in sorted(glob.glob(pat)):
        d = Path(f).parent
        try:
            m = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception as exc:                                   # noqa: BLE001
            out.append({"dir": d.name, "_broken": str(exc)})
            continue
        rid = (m.get("summary") or {}).get("run_id") or d.name
        ctrl = (m.get("config") or {}).get("controller") or {}
        trace = m.get("controller_trace") or []
        out.append({
            "dir": d.name, "run_id": rid,
            "label": m.get("data_label"),
            "commit": (m.get("reproducibility") or {}).get("git_commit"),
            "dirty": (m.get("reproducibility") or {}).get("git_dirty"),
            "kind": ctrl.get("kind"), "shadow": ctrl.get("shadow"),
            "criterion": ctrl.get("signal_criterion"),
            "n_trace": len(trace),
            "has_kind_entry": any(("cost_policy" if ctrl.get("kind") == "cost"
                                   else "slo_weave") in e for e in trace),
            "recorded_at": m.get("recorded_at"),
            "has_config": bool(m.get("config")),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="追溯链自查")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--all-measured", action="store_true")
    ap.add_argument("--experiments", default=str(_ROOT / "EXPERIMENTS.md"),
                    help="登记表路径（默认仓库根的 EXPERIMENTS.md）")
    args = ap.parse_args(argv)

    subs = args.tags or ([None] if args.all_measured else [])
    if not subs:
        print("用法：--tags S14B1 S14C1 …  或  --all-measured", file=sys.stderr)
        return 2

    runs: list[dict[str, Any]] = []
    for s in subs:
        runs.extend(_runs(s))
    if not runs:
        print("✗ 没找到产物", file=sys.stderr)
        return 2

    problems: list[str] = []
    print(f"扫描 {len(runs)} 个 run（tags={subs}）\n")

    # ---- 1) 溯源字段齐不齐 -------------------------------------------------
    print("=== 1) 溯源字段 ===")
    miss = [r for r in runs if not (r.get("commit") and r.get("label") and r.get("recorded_at")
                                    and r.get("has_config"))]
    print(f"  字段不全的：{len(miss)}")
    for r in miss[:8]:
        problems.append(f"字段不全：{r['dir']}")

    # ---- 2) 配置声称的臂 vs 实际行为（**误标**）---------------------------
    print("\n=== 2) '配置说是什么臂' vs '实际跑出什么' ===")
    mis = [r for r in runs
           if r.get("kind") in ("cost", "slo_weave") and r.get("n_trace", 0) > 0
           and not r.get("has_kind_entry")]
    zero = [r for r in runs if r.get("kind") in ("cost", "slo_weave") and r.get("n_trace", 0) == 0]
    print(f"  轨迹里**没有该臂决策**的：{len(zero)}"
          f"{'  ← 配置写了 kind 却没跑（误标 / 未接线）' if zero else ''}")
    for r in zero[:8]:
        problems.append(f"**误标**：{r['dir']}（kind={r['kind']} 但轨迹 {r['n_trace']} 条）")
    print(f"  有轨迹但缺该臂条目的：{len(mis)}"
          f"{'  ← **误标**：配置写了该臂，轨迹里却没有它的决策（它根本没被调用）' if mis else ''}")
    for r in mis[:8]:
        problems.append(f"**误标**：{r['dir']}（kind={r['kind']}，轨迹 {r['n_trace']} 条但无该臂条目）")

    # ---- 3) TEST_DATA 混入 -------------------------------------------------
    print("\n=== 3) TEST_DATA（dry-run）混入 ===")
    td = [r for r in runs if r.get("label") == "TEST_DATA"]
    print(f"  TEST_DATA 产物：{len(td)}（它们**只能**用于链路验证，不得进结论）")
    for r in td[:6]:
        print(f"     · {r['dir']}")

    # ---- 4) 同一 tag 内混源码 ----------------------------------------------
    print("\n=== 4) 同一 tag 内是否混了不同源码状态 ===")
    by_tag: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        tag = str(r["run_id"]).split("-")[0]
        by_tag[tag].append(r)
    for tag, rs in sorted(by_tag.items()):
        commits = {r.get("commit") for r in rs}
        dirt = {bool(r.get("dirty")) for r in rs}
        flag = ""
        if len(commits) > 1 or True in dirt:
            flag = "  ← ⚠️ 混了源码状态（验收第 5 条禁止合并解释）"
            problems.append(f"{tag} 混源码：commits={commits} dirty={dirt}")
        print(f"  {tag:<16} n={len(rs):<3} commits={len(commits)} dirty={dirt}{flag}")

    # ---- 5) 判据是否被记录 -------------------------------------------------
    print("\n=== 5) 实际生效的判据（signal_criterion）===")
    crit = defaultdict(int)
    for r in runs:
        if r.get("kind") in ("cost", "slo_weave", "threshold"):
            crit[str(r.get("criterion"))] += 1
    for k, v in sorted(crit.items()):
        note = "（'None' = 产物没记该字段 ⇒ 走的是默认值）" if k == "None" else ""
        print(f"  {k:<14} {v}{note}")

    # ---- 6) MEASURED 批次是否已登记（**机械可查**；口径与导出闸门一致）------
    print("\n=== 6) 登记覆盖（EXPERIMENTS.md 里有没有该**批次**的表格单元）===")
    exp_path = Path(args.experiments)
    if not exp_path.exists():
        print(f"  ⚠️ 没找到登记表 {exp_path} ⇒ **未判**（不假装通过）")
        problems.append(f"缺登记表：{exp_path}")
    else:
        text = exp_path.read_text(encoding="utf-8")
        measured = [r for r in runs if r.get("label") == "MEASURED"]
        # 粒度取**批次 base**（与 preflight 检查 4 一致）；同批拆成多组也只登记一次。
        bases: dict[str, set[str]] = defaultdict(set)
        for r in measured:
            bases[str(r["run_id"]).split("-")[0]].add(base_run_id(str(r["run_id"])))
        n_base_unreg = 0
        # 附加信息：**逐个 run** 的登记数（只报，不参与判定 —— 判定的权威在 preflight）
        n_run_reg = sum(1 for r in measured if registered_in_table(text, str(r["run_id"])))
        print(f"  MEASURED run {len(measured)} 个（其中 {n_run_reg} 个 run_id 本身出现在表里）"
              f"｜批次 base {sum(len(v) for v in bases.values())} 个")
        for tag, bs in sorted(bases.items()):
            unreg = sorted(b for b in bs if not registered_in_table(text, b))
            n_base_unreg += len(unreg)
            flag = ""
            if unreg:
                flag = f"  ← ⚠️ **未登记 {len(unreg)}/{len(bs)}**"
                problems.append(f"{tag}: 未登记的批次 {unreg}")
            print(f"  {tag:<16} 批次 {len(bs):<3} 未登记 {len(unreg):<3}{flag}")
            for b in unreg[:4]:
                # ⚠️ 区分两种"未登记"，否则会把"旧格式登记过"误报成"漏登记"：
                #   · 表里**任何位置**都搜不到该 id ⇒ 确实没登记
                #   · 搜得到、但不是独立单元 ⇒ 多半写在证据列 / 用的是配置名而非 run_id
                #     （老批次 S9/S10/P8FULL 就是这个形状）⇒ 口径不一致，需人工核，
                #     **不得**直接判成"漏登记"。
                where = "表中**完全未出现**" if b not in text else "表中有提及但**非独立单元**"
                print(f"       · {b}  —— {where}")
        print("  ⚠️ 判定要求 run_id 作为**表格单元**出现 —— 散文里提一嘴**不算**"
              "（那给不出重复次数与离散度）。")
        print("  ⚠️ 这条原先列为'只能人工'，其实机械可查：导出闸门（`preflight.py` 检查 4）"
              "早就在验，只是**只有跑导出时才会响** ⇒ `S14` 全族没登记也没人发现。")

    # ---- 人工清单 ----------------------------------------------------------
    print("\n=== 只能人工核的两件（本脚本不假装能自动判）===")
    print("  A. 图表能否**由脚本重建**（看 results/figures/*.source.txt 是否在、且指向脚本）")
    print("  B. **撤回的结论是否已传播到所有引用处**（搜关键词 + 人读；例：'1.77%'）")

    print(f"\n=== 汇总：机械可查的问题 {len(problems)} 处 ===")
    for p in problems[:20]:
        print(f"  - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
