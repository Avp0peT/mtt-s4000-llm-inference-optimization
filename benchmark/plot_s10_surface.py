#!/usr/bin/env python
"""从 `results/processed/s10_surface.json` 生成**静态响应面**图。

约定（继承 `benchmark/plot_p99.py` 的绘图纪律）
----------------------------------------------
* **`build_plan` 是纯函数**（不需要 matplotlib）：分组与"缺哪些口径"都在它里面定，
  于是这些判断可以被单元测试覆盖。绘图逻辑内嵌在别处 ⇒ 无法测试 ⇒ 缺陷静默存活，
  这条教训在本仓库已经用真事付过学费（见 `plot_p99.py` 的开头）。
* **每个 panel 必须写明口径**：吞吐是 **window**（按事件发生时刻），
  延迟是 **cohort**（按请求**到达**在哪一段）。混成一个口径的后果见第 05 步。
* 除 PNG 外写一个 `.source.txt`：**这张图是用哪些 run 画的、哪个口径缺数据**。
  没有它，半年后没人能解释这张图。
* 标签一律 **ASCII**（matplotlib 默认字体没有中文字形 ⇒ 中文会变方块）；
  中文解释放在 `.source.txt` 与文档里。

用法::

    python scripts/analyze_s10_surface.py --tag S10 --json results/processed/s10_surface.json
    python benchmark/plot_s10_surface.py --plan results/processed/s10_surface.json \
        --out results/figures/s10_surface.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: 短名（图例用）。键 = 配置名，值 = 图例标签。
SHORT: dict[str, str] = {
    "ramp_static_c8": "ck=1024 bt=8192",
    "ramp_static_c8_ck512": "ck=512",
    "ramp_static_c8_ck2048": "ck=2048",
    "ramp_static_off_c8": "chunk OFF",
    "ramp_static_c8_bt2048": "bt=2048",
}


@dataclass
class Panel:
    key: str
    label: str
    lens: str                                  # window | cohort
    #: {cell: {segment: 中位}} —— 缺的格/段**不填 0**，直接缺席
    values: dict[str, dict[str, float]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


@dataclass
class PlotPlan:
    panels: list[Panel] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    cells: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    best_static: str | None = None
    suptitle: str = ""


def build_plan(plan: dict[str, Any]) -> PlotPlan:
    """纯函数：把 `s10_surface.json` 变成绘图计划（**不 import matplotlib**）。

    ⚠️ 缺数据一律记进 `Panel.missing` 并从 `values` 里**缺席**——不填 0。
    填 0 会在图上画出一根"看起来是好得不可能的测量值"的柱子（`plot_p99.py` 的原始缺陷）。
    """
    segments = list(plan.get("segments") or [])
    cells = [c for c in (plan.get("cells") or []) if c != "best_static"]
    metrics = plan.get("metrics") or []
    per = plan.get("per_cell_metrics") or {}

    panels: list[Panel] = []
    for m in metrics:
        p = Panel(key=m["key"], label=m["label"], lens=m["lens"])
        for c in cells:
            for s in segments:
                cell_seg = (per.get(c) or {}).get(s) or {}
                agg = cell_seg.get(m["key"]) or {}
                v = agg.get("median")
                n = agg.get("n") or 0
                if v is None:
                    p.missing.append(f"{c}/{s}（n={n}）")
                    continue
                p.values.setdefault(c, {})[s] = float(v)
        panels.append(p)

    run_ids: list[str] = []
    for r in plan.get("per_run") or []:
        rid = r.get("run") or ""
        if rid:
            run_ids.append(rid)

    best = plan.get("best_static")
    return PlotPlan(
        panels=panels, segments=segments, cells=cells, run_ids=sorted(set(run_ids)),
        best_static=best,
        suptitle=(f"Static response surface on load_ramp  (tag={plan.get('tag')}, "
                  f"repeats per cell=3)  best={best or 'n/a'}"),
    )


def write_source(plan: PlotPlan, plan_json: dict[str, Any], png: Path) -> Path:
    """写 `.source.txt`：**这张图是用哪些 run 画的、哪个口径缺数据**。"""
    p = png.with_suffix(".source.txt")
    lines = [
        "S10 static response surface",
        "=" * 40,
        f"suptitle: {plan.suptitle}",
        "",
        "口径（每个 panel 必须分开读）：",
        "  * goodput  → **window** 口径（按事件发生时刻；分母 = 段墙钟长度）",
        "  * 延迟类   → **cohort** 口径（按请求到达在哪一段）",
        "  ⚠️ 两套口径不可互换（第 05 步）：high 段末尾到达的请求会在 recover 段完成。",
        "",
        "本轮各 panel 的实际口径：",
    ]
    for panel in plan.panels:
        lines.append(f"  - {panel.label:<16} lens={panel.lens}")
    lines += ["", f"噪声底（各候选重复极差上限）：{plan_json.get('noise_floor')}",
              f"Pareto 前沿：{plan_json.get('pareto')}",
              f"Best Static：{plan_json.get('best_static')}",
              f"平局规则排序：{plan_json.get('tie_order')}",
              f"S5 单一支配者：{plan_json.get('s5_sole_dominator')}", ""]
    miss = [f"{p.key}: {', '.join(p.missing)}" for p in plan.panels if p.missing]
    lines += ["缺数据的格/段（**图上直接缺席，不填 0**）："] + (miss or ["  （无）"])
    lines += ["", f"源 run（{len(plan.run_ids)} 个）："] + [f"  {r}" for r in plan.run_ids]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p


def plot(plan: PlotPlan, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(1, len(plan.panels))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    x = list(range(len(plan.segments)))
    for ax, panel in zip(axes[0], plan.panels):
        for c in plan.cells:
            ys = [panel.values.get(c, {}).get(s) for s in plan.segments]
            if all(v is None for v in ys):
                continue        # 该格在本 panel 上完全没数据 ⇒ 不画（而不是画 0）
            xs = [xi for xi, v in zip(x, ys) if v is not None]
            vs = [v for v in ys if v is not None]
            style = {"marker": "o", "linewidth": 2}
            if c == plan.best_static:
                style = {"marker": "o", "linewidth": 3.2}
            ax.plot(xs, vs, label=SHORT.get(c, c), **style)
        ax.set_xticks(x)
        ax.set_xticklabels(plan.segments, fontsize=9)
        ax.set_title(f"{panel.label}  [{panel.lens}]", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.suptitle(plan.suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="画 S10 静态响应面")
    ap.add_argument("--plan", required=True, help="analyze_s10_surface.py --json 的产物")
    ap.add_argument("--out", default=str(_ROOT / "results" / "figures" / "s10_surface.png"))
    ap.add_argument("--plot-only", action="store_true", help="只打印计划，不画图")
    args = ap.parse_args(argv)

    plan_json = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    plan = build_plan(plan_json)

    print(f"{plan.suptitle}")
    for p in plan.panels:
        n_cells = len(p.values)
        print(f"  {p.label:<16} [{p.lens:<6}] 有数据的格 {n_cells}/{len(plan.cells)}"
              + (f"；缺 {len(p.missing)} 项" if p.missing else "；无缺项"))
    for p in plan.panels:
        for m in p.missing:
            print(f"    缺 {p.key}: {m}")
    if args.plot_only:
        return 0

    png = Path(args.out)
    plot(plan, png)
    src = write_source(plan, plan_json, png)
    print(f"\n已写出：{png}\n         {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
