#!/usr/bin/env python
"""从 ``summary.csv`` 生成 P99 对比条形图。

为什么把它从 ``scripts/export_results.sh`` 里抽出来
----------------------------------------------------
绘图逻辑原先内嵌在 bash 的 heredoc 里，**无法被单元测试覆盖**。于是两个缺陷
一路静默存活到评审阶段：

  - 中位数函数对**空分位块**兜底成 ``0.0``，图上出现一根
    "TPOT P99 = 0 ms"的柱子。0 看起来像一个好得不可能的**测量值**，
    比缺一根柱子更容易误导；
  - 分组只按 ``config``，把"同一份配置跑在两个负载上"的数字混进同一个中位数——
    那个数既不是负载 A 的结果，也不是负载 B 的结果，而是谁都没测过的东西。
  - 分组里没有**平台**，两个平台上的同名实验被并成一根柱子。已复现：
    两份明确标着 ``TEST_DATA`` 的数据（同负载、同配置）分别取值 100 与 200，
    图上出现一根 150 的柱——一个谁都没测过的值，且全程不报错。

抽成模块后，这两件事都成了可断言的：``build_plan`` 是纯函数（不需要 matplotlib），
绘图纪律由它兜住。

绘图纪律
--------
图注必须写明数据来源 run_id 与口径，否则半年后无人能解释这张图。除 PNG 外
还写一个 ``.source.txt``，记录这张图是用哪些 run 画出来的、哪些口径缺数据。
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.grouping import GROUP_DIMENSIONS, group_dimensions  # noqa: E402

#: 写进 ``.source.txt`` 的分组维度名，直接取自共用定义（见 :func:`write_source`）。
_GROUPING_DIMENSIONS = GROUP_DIMENSIONS

METRICS = ("ttft", "tpot", "stall")
_SUFFIX = "_p99"
#: 轴上标签的最大长度（超出则截断，避免互相压字）。
_MAX_LABEL = 22

_METRIC_TITLES = {
    "ttft": "TTFT P99 (ms)",
    "tpot": "TPOT P99 (ms)",
    "stall": "TPOT stall P99 (ms/token)",
}

#: 图内文字一律 ASCII。默认字体（DejaVu Sans）没有中文字形，写中文只会渲染成
#: 一串豆腐块——比不写更糟，因为读者会以为图坏了。中文解释放在 .source.txt
#: 与终端输出里：那两处是 UTF-8 文本，不经过字体渲染。
#:
#: 标题**动态生成**（见 :func:`suptitle`），不再写死平台名：写死昇腾会让一张
#: MTT 数据的图自称昇腾结果，而这句错话只在"有人照着复现"的时候才暴露。
TITLE_LEAD = "LLM inference optimization - P99 comparison (median across repeats)"


def _fnum(value: Any) -> float | None:
    """把一个格子读成 float；空值返回 ``None``。

    **不能用** ``float(value or 0)``：那样空列会变成 0，图上就出现一根
    "P99 = 0 ms"的柱子。列本身为空是合法状态：一次"全部请求都没跑完"的 run
    没有成功请求，分位块整个是空的。
    """
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def median_or_none(values: Iterable[float | None]) -> float | None:
    """中位数；**空集合返回 None**，不返回 0。

    见模块开头：这正是"缺数据被画成 0"的来源。缺失必须在图上留白，
    而不是变成一个看起来完美的测量值。
    """
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


#: 图中"环境维度"列的读法：空/缺失一律落到这个值。
#: 与 ``collect_metrics`` 用同一个默认值——两处不一致会让同一份数据在表和图上
#: 分到不同的组，而两处都不会报错。
_UNKNOWN = "unknown"


def _cell(row: dict[str, Any], key: str, default: str = _UNKNOWN) -> str:
    value = row.get(key)
    if value is None:
        return default
    return str(value).strip() or default


@dataclass
class PlotPlan:
    """一张图的全部决策结果。纯数据，不依赖 matplotlib——因此可被测试。"""

    #: 分组标签，形如 ``<workload>/<config>``，只在图上有数据的组。
    labels: list[str] = field(default_factory=list)
    #: 轴上显示的标签（可能被截断），与 ``labels`` 一一对应。
    display_labels: list[str] = field(default_factory=list)
    #: 每个面板一组中位数；缺数据的组是 ``None``（**不是** 0）。
    panels: dict[str, list[float | None]] = field(default_factory=dict)
    #: 标签 -> 该组用到的 run_id（图注溯源用）。
    run_ids: dict[str, list[str]] = field(default_factory=dict)
    #: 整组无可画指标（TTFT 全空），被略去的组。
    skipped: list[str] = field(default_factory=list)
    #: 标签 -> 该组缺失的指标名。
    omitted: dict[str, list[str]] = field(default_factory=dict)
    #: 出现过的数据标签（写进图注，判断这张图有没有资格进论文）。
    data_labels: list[str] = field(default_factory=list)
    #: 出现过的环境维度。写进标题与 .source.txt —— 图上必须能读出这张图
    #: 是哪个平台/哪个后端/哪种精度的数据，否则换个人看图只能靠猜。
    platforms: list[str] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)
    precisions: list[str] = field(default_factory=list)

    @property
    def has_data(self) -> bool:
        return bool(self.labels)

    @property
    def multi_platform(self) -> bool:
        return len(self.platforms) > 1

    @property
    def multi_backend(self) -> bool:
        """数据里真有两个以上后端时，标签才需要写出是哪一个。"""
        return len(self.backends) > 1


def _group_key(row: dict[str, Any]) -> tuple[str, ...]:
    """绘图分组键：``GROUP_DIMENSIONS`` 的取值，即
    ``(workload, config, hardware_platform, backend)``。

    平台与后端都必须进键。少了任何一个，两个来源的同名实验就会被并进一组求
    中位数，产出一个谁都没测过的数——已复现：TEST_DATA 的 100 与 200 合成了一根
    150 的柱。这个错误不报错、不出警告，只是图上多一根看起来正常的柱子。

    维度表**只有一份**，在 :mod:`ascend_opt.grouping`——这里曾经漏掉后端维度，
    于是"同配置、模拟 vs 真机"两次运行被并成一根柱，而汇总表（按后端拆开）
    与图上（没拆）对不上。两处不一致时，读的人只能靠猜该信哪个。

    刻意**不含** run_id 批次：本函数是逐行聚合分位数，把同一实验的几次重复
    当成几个采样点正是要的结果；带上批次会把它们拆成几根柱。
    """
    return group_dimensions(row)


def _display_labels(labels: list[str], limit: int = _MAX_LABEL) -> list[str]:
    """按长度截断标签；若截断后会**重名**，则放弃截断。

    重名比"字太长"严重得多：两根不同的柱子顶着同一个名字，读者会把它们
    当成同一组。加平台后缀之后标签普遍变长，截断到同一前缀的可能性大幅上升，
    所以这条兜底是必须的。
    """
    truncated = [
        label if len(label) <= limit else label[: limit - 2] + ".."
        for label in labels
    ]
    if len(set(truncated)) == len(truncated):
        return truncated
    # 宁可挤一点，也不能让读者把两根柱子当成同一组。
    return list(labels)


def build_plan(rows: list[dict[str, Any]]) -> PlotPlan:
    """把汇总行压成一张图的绘制计划。纯函数，无副作用。

    分组键是 ``GROUP_DIMENSIONS``（见 :func:`_group_key`），不是 config 单独一项。
    """
    plan = PlotPlan(panels={m: [] for m in METRICS})
    plan.data_labels = sorted(
        {(r.get("data_label") or "<无标签>").strip() for r in rows}
    )
    plan.platforms = sorted({_cell(r, "hardware_platform") for r in rows})
    plan.backends = sorted({_cell(r, "backend") for r in rows})
    # 精度缺数据时写空（"没记录"），不写 unknown —— 它与平台/后端不同，
    # 后两者是分类维度，精度是取值维度，空值在这里是可读的。
    plan.precisions = sorted(
        {_cell(r, "precision", "").strip() for r in rows} - {""}
    )

    groups = sorted({_group_key(r) for r in rows})
    for dims in groups:
        group = [r for r in rows if _group_key(r) == dims]
        workload, cfg, platform, backend = dims
        # 只在数据里**真有**多个平台/后端时才把该维度写进标签：只有一个取值时
        # 每个标签都重复同一句是噪音，而分组本身（正确性所在）不受影响。
        #
        # 标签因此**必然唯一**：两个分组若取值不同，差异维度要么本就在标签里
        # （workload/config），要么会让该维度成为"多取值"从而被写进后缀。
        label = f"{workload}/{cfg}"
        if plan.multi_platform:
            label = f"{label}@{platform}"
        if plan.multi_backend:
            label = f"{label}#{backend}"
        series = {
            m: [v for v in (_fnum(r.get(f"{m}{_SUFFIX}")) for r in group)
                if v is not None]
            for m in METRICS
        }
        # TTFT 是这张图的主口径。它整组取不到 → 这个组没有可画的东西，
        # 明确跳过并说明原因，而不是在图上留一根高度为 0 的柱子。
        if not series["ttft"]:
            plan.skipped.append(label)
            continue

        plan.labels.append(label)
        plan.run_ids[label] = [str(r.get("run_id", "")) for r in group]

        missing: list[str] = []
        for m in METRICS:
            value = median_or_none(series[m])
            plan.panels[m].append(value)
            if value is None:
                missing.append(m)
        if missing:
            plan.omitted[label] = missing

    plan.display_labels = _display_labels(plan.labels)
    return plan


def _title_dimensions() -> str:
    """标题里的分组维度串，**由共用定义派生**（见 :func:`suptitle`）。"""
    return "/".join(d.replace("hardware_", "") for d in GROUP_DIMENSIONS)


def suptitle(plan: PlotPlan) -> str:
    """图的总标题：三行，全部 ASCII，且**按实际数据**写出环境维度。

    为什么必须动态：写死平台名（或写死分组维度）的标题在数据或分组规则变化后
    依然会印着老内容。那句错话不影响图的观感，只会在有人照着它去复现时把
    人引到错误的结论上——"分组里有没有后端"正是最要紧的一处。
    """
    return (
        f"{TITLE_LEAD}\n"
        f"grouped by ({_title_dimensions()}); platforms: "
        f"{', '.join(plan.platforms) or _UNKNOWN}\n"
        f"backend: {', '.join(plan.backends) or _UNKNOWN}; "
        f"precision: {', '.join(plan.precisions) or _UNKNOWN}; "
        f"data labels: {', '.join(plan.data_labels) or 'n/a'}; "
        f"'n/a' = no usable percentile (not drawn as 0)"
    )


def describe_plan(plan: PlotPlan) -> list[str]:
    """把计划里"需要注意"的地方写成人类可读的行（打印到终端）。"""
    lines: list[str] = []
    if plan.skipped:
        lines.append(
            f"  以下分组无可用的 TTFT 分位数，已从图中略去：{', '.join(plan.skipped)}"
        )
    for label, miss in plan.omitted.items():
        lines.append(
            f"  注意：{label} 缺 {'/'.join(miss)} 分位数，该子图留空并标 n/a（不画 0）"
        )
    return lines


def render(plan: PlotPlan, png_path: Path) -> None:
    """画图并保存。matplotlib 在此处惰性导入——没装也不影响 ``build_plan`` 的测试。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    xs = list(range(len(plan.labels)))
    for ax, metric in zip(axes, METRICS):
        values = plan.panels[metric]
        # 只画有值的柱子。缺值的组在原位标 n/a——"缺"必须在图上看得见，
        # 而不是被画成 0，也不是被悄悄挤掉（挤掉会让柱子与刻度错位）。
        drawn = [(x, v) for x, v in zip(xs, values) if v is not None]
        if drawn:
            ax.bar([x for x, _ in drawn], [v for _, v in drawn],
                   width=0.7, color="#c0392b")
        for x, v in zip(xs, values):
            if v is None:
                ax.annotate("n/a", (x, 0), ha="center", va="bottom",
                            fontsize=8, color="#888888", rotation=90)
        ax.set_xticks(xs)
        ax.set_xticklabels(plan.display_labels, rotation=30, ha="right")
        ax.set_title(_METRIC_TITLES[metric])
        ax.grid(axis="y", alpha=0.3)
    # 字号调小、图加高：标题从两行变三行（多了平台/后端/精度），
    # 不腾地方就会和子图标题压在一起。
    fig.suptitle(suptitle(plan), fontsize=9)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)


def write_source(plan: PlotPlan, source_path: Path) -> None:
    """写图的数据来源说明（.source.txt）。

    分组口径那句**由 :data:`ascend_opt.grouping.GROUP_DIMENSIONS` 生成**，
    不写死：这行字是别人复核这张图的唯一入口，写死就会在分组规则变化后
    继续宣称旧规则——图是对的、说明是错的，比两者都错更难发现。
    """
    source_path.parent.mkdir(parents=True, exist_ok=True)
    dims = ", ".join(_GROUPING_DIMENSIONS)
    body = (
        "figure: p99_comparison\n"
        "data: results/processed/summary.csv\n"
        f"grouping: ({dims})\n"
        f"aggregation: median across repeats per ({dims})\n"
        "missing_metrics: 该口径无可用分位数时留空并标 n/a，**不画成 0**"
        "（空分位块 = 没有成功请求，是合法状态）\n"
        f"data_labels: {', '.join(plan.data_labels) or 'n/a'}\n"
        # 环境维度必须落盘：图缩进论文之后没人能再从标题里读出它，
        # 而"这张图是哪个平台/哪个后端/哪种精度的"正是复现的第一问。
        f"platforms: {', '.join(plan.platforms) or _UNKNOWN}\n"
        f"backends: {', '.join(plan.backends) or _UNKNOWN}\n"
        f"precisions: {', '.join(plan.precisions) or _UNKNOWN}\n"
        "caveat: 标签非 MEASURED 时本图不得进论文"
        "（AGENTS.md 正文第 10 节 / 第 18 节）\n"
        "caveat: 跨平台/跨后端的数据不得合并求中位数；分组键已含平台与执行后端，"
        "若图上出现 mixed 说明分组被绕过\n"
        + (f"omitted_groups: {', '.join(plan.skipped)}  # 无成功请求，无 TTFT 分位数\n"
           if plan.skipped else "")
        + "".join(
            f"missing_in_group: {label} -> {', '.join(miss)}\n"
            for label, miss in plan.omitted.items()
        )
        + "runs:\n"
        + "\n".join(
            f"  {label}: {', '.join(ids)}" for label, ids in plan.run_ids.items()
        )
        + "\n"
    )
    # newline="\n"：这是给人看的溯源旁注（哪些 run 进了这张图）。Windows 上
    # write_text 默认会把 \n 翻成 \r\n，跨平台 diff 时会凭空多出一堆行尾差异。
    with source_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)


def read_rows(summary_path: Path) -> list[dict[str, Any]]:
    with Path(summary_path).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="从 summary.csv 生成 P99 对比图")
    p.add_argument("--summary", required=True, help="汇总表路径（summary.csv）")
    p.add_argument("--figures-dir", default="results/figures", help="图片输出目录")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = Path(args.summary)
    if not summary.exists() or summary.stat().st_size == 0:
        print("  没有 summary.csv，跳过绘图")
        return 0

    rows = read_rows(summary)
    if not rows:
        print("  summary.csv 无数据行，跳过绘图")
        return 0

    plan = build_plan(rows)
    if not plan.has_data:
        print("  所有分组都没有可用的 TTFT 分位数（没有任何成功请求），跳过绘图")
        return 0
    for line in describe_plan(plan):
        print(line)

    export_name = summary.resolve().parent.name
    figures_dir = Path(args.figures_dir)
    png = figures_dir / f"p99_comparison_{export_name}.png"
    source = figures_dir / f"p99_comparison_{export_name}.source.txt"

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        # 图是"尽力而为"：汇总表本身仍可用，所以这里不算失败。
        # 但仍要把来源说明写出来——否则事后会以为这张图存在过。
        write_source(plan, source)
        print("  未安装 matplotlib，跳过绘图（pip install matplotlib 后可重跑本步骤）")
        print(f"  图源说明已写出：{source}")
        return 0

    render(plan, png)
    write_source(plan, source)
    print(f"  图已生成：{png}")
    print(f"  数据标签：{', '.join(plan.data_labels)}")
    print(
        f"  环境维度：平台 {', '.join(plan.platforms) or _UNKNOWN}；"
        f"后端 {', '.join(plan.backends) or _UNKNOWN}；"
        f"精度 {', '.join(plan.precisions) or _UNKNOWN}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
