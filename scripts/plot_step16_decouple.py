#!/usr/bin/env python
"""第 16 步 · P27 的两张图（**只读、可重建**）。

对应 `paper/sections/04_failure_analysis.md` 的
「【待补·P27】两张图」占位块。该占位明写：
    来源：`results/processed/decouple_*.json` 与 `lookup_bucket_sweep.json`；
    当前状态：数据齐，缺作图脚本与 `.source.txt`。
    图必须能**由脚本重建**并标注 run_id。
本脚本就是那个"缺的作图脚本"。

两张图
------
图 A `fig_p27a_decouple.{png,svg}` —— 2×2 解耦（{旧,新}网格 × {hash, 秩保证} 划分）
    面板 A1：`underest_p95`（低估 p95，主判据；红线 = 5% 阈值）
    面板 A2：`train_min_rank` / `holdout_min_rank`（解释**为什么**：秩不足）
    读法：`hash` 列两代几乎不动（≈95%），`rank_guaranteed` 列随网格变密而下降
    ⇒ **主因是划分方式，不是网格密度**。

图 B `fig_p27b_lookup_sweep.{png,svg}` —— 查表分桶扫描（20 格：`pf_step` × `kv_step`）
    面板 B1：`coverage`（覆盖率，越高越好）
    面板 B2：`over_p95`（超出 p95，越低越好，红区 = >5%）
    读法：**`pf_step` 完全不动**（同一列 5 个值逐字相同），只有 `kv_step` 在动
    ⇒ "范围外回退"的瓶颈在 **kv 桶**，不在 prefill 桶。

设计约束（刻意遵守）
--------------------
* **不产生新数字**：只读上面两个 JSON，不重算、不四舍五入成"更好看"的值；
* **缺 ≠ 0**：`None` 一律**留空/断线**，不画成 0；
* **可溯源**：每张图旁写 `.source.txt`，含输入文件路径、文件 sha256、生成命令、
  以及被引用的 run_id / tag 清单；
* 中文标注：本机字体够用（`DejaVu Sans` 无中文 ⇒ 显式指定中文字体，找不到则回退并**警告**）。

用法
----
    python scripts/plot_step16_decouple.py                 # 图写到 paper/figures/
    python scripts/plot_step16_decouple.py --out DIR
    python scripts/plot_step16_decouple.py --check         # 只校验输入，不画
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "results" / "processed"
DEFAULT_OUT = REPO / "paper" / "figures"

# 输入（顺序即图 A / 图 B 的数据来源）
DECOUPLE_FILES = ("decouple_v1_v3.json", "decouple_v3_v4.json")
LOOKUP_FILE = "lookup_bucket_sweep.json"

# 判据阈值（**照抄预注册**，不在这里新造）
UNDEREST_P95_THRESHOLD = 0.05   # 低估 p95 > 5% ⇒ model_holds=False（阶段二止损点）
COVERAGE_TARGET = 1.0


# 图内文字用的"安全"标记：Microsoft YaHei / SimHei **不含** ✅⚠️⇒ 这类符号
# （实测：matplotlib 会报 "Glyph 9888 (WARNING SIGN) missing from font"，
#  落到图上是**方框/缺字**）。图里一律用这些 ASCII/中文替代，保证可渲染。
_GLYPH_MAP = {
    "✅": "[OK]", "⚠️": "[!]", "⚠": "[!]", "⇒": "=>", "→": "->",
    "✗": "[x]", "✓": "[v]",
}


def safe_plot_text(s: str) -> str:
    """把图内文字里的非字体符号换成可渲染的替代（只用于**画到图上**的文字）。"""
    for k, v in _GLYPH_MAP.items():
        s = s.replace(k, v)
    return s


# --------------------------------------------------------------------------
# 载入
# --------------------------------------------------------------------------
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_decouple(processed: Path) -> list[dict[str, Any]]:
    """读 `decouple_*.json`；缺文件即报错（**不得**当作'没有数据'）。"""
    out: list[dict[str, Any]] = []
    for name in DECOUPLE_FILES:
        p = processed / name
        if not p.is_file():
            raise FileNotFoundError(
                f"缺输入 {p}；P27 图 A 无法重建。"
                f"（注意：`decouple_dryrun.json` 是干跑产物，**不得**替代）"
            )
        d = json.loads(p.read_text(encoding="utf-8"))
        cells = d.get("cells") or {}
        if not cells:
            raise ValueError(f"{p} 的 `cells` 为空 ⇒ 不得当作'没有数据'，应视为产物损坏")
        out.append({"name": name, "path": p, "sha256": _sha256(p),
                    "modulus": d.get("holdout_modulus"), "cells": cells})
    return out


def load_lookup(processed: Path) -> dict[str, Any]:
    p = processed / LOOKUP_FILE
    if not p.is_file():
        raise FileNotFoundError(f"缺输入 {p}；P27 图 B 无法重建")
    d = json.loads(p.read_text(encoding="utf-8"))
    cells = d.get("cells") or []
    if not cells:
        raise ValueError(f"{p} 的 `cells` 为空 ⇒ 视为产物损坏，不得画空图")
    return {"name": LOOKUP_FILE, "path": p, "sha256": _sha256(p), "doc": d}


# --------------------------------------------------------------------------
# 解析 decouple 的格子键（形如 "旧网格|rank_guaranteed"）
# --------------------------------------------------------------------------
GRID_ORDER = ("旧网格", "新网格")
SPLIT_ORDER = ("hash", "rank_guaranteed")
SPLIT_LABEL = {"hash": "哈希划分 (hash)", "rank_guaranteed": "秩保证划分 (rank_guaranteed)"}


def parse_decouple(cells: dict[str, dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, val in cells.items():
        if "|" not in key:
            raise ValueError(f"decouple 格子键 '{key}' 不含 '|'，无法解析网格|划分")
        grid, split = key.split("|", 1)
        out[(grid.strip(), split.strip())] = val
    return out


# --------------------------------------------------------------------------
# 字体（中文）
# --------------------------------------------------------------------------
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 本机可用中文字体（按优先级）
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "WenQuanYi Zen Hei", "SimSun"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = next((c for c in candidates if c in available), None)
    if picked is None:
        print("[!] 未找到中文字体 ⇒ 图内中文可能显示为方框。"
              "请安装 SimHei / Microsoft YaHei 或 Noto Sans CJK。", file=sys.stderr)
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = [picked] + candidates
    plt.rcParams["axes.unicode_minus"] = False   # 负号
    return plt, picked


# --------------------------------------------------------------------------
# 图 A
# --------------------------------------------------------------------------
def fig_a(plt, decouple: list[dict[str, Any]], outdir: Path) -> tuple[Path, Path]:
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))

    gens = [d["name"].replace("decouple_", "").replace(".json", "") for d in decouple]
    x = np.arange(len(gens), dtype=float)
    width = 0.34

    parsed = [parse_decouple(d["cells"]) for d in decouple]

    # ---- 面板 A1：低估 p95（主判据）--------------------------------------
    for j, split in enumerate(SPLIT_ORDER):
        ys, ns = [], []
        for pm in parsed:
            v = pm.get((GRID_ORDER[0], split), {})
            ys.append(v.get("underest_p95"))
            ns.append(v.get("holdout_n"))
        # 缺值一律跳过（不画 0）
        xs = np.array([xx for xx, yy in zip(x, ys) if yy is not None], dtype=float)
        yv = [yy for yy in ys if yy is not None]
        ax1.bar(xs + (j - 0.5) * width, yv, width,
                label=SPLIT_LABEL[split],
                color=("#c0392b" if split == "hash" else "#2471a3"),
                edgecolor="black", linewidth=0.6)
        for xx, yy, nn in zip(xs, yv, ns):
            if yy is not None and nn is not None:
                ax1.text(xx + (j - 0.5) * width, yy + 0.015, f"n={nn}",
                         ha="center", va="bottom", fontsize=8)

    ax1.axhline(UNDEREST_P95_THRESHOLD, color="black", linestyle="--", linewidth=1.2)
    # 阈值标签放左上，避开右侧图例与柱子标注
    ax1.text(-0.45, UNDEREST_P95_THRESHOLD + 0.03,
             f"判据阈值 {UNDEREST_P95_THRESHOLD:.0%}（超过即 model_holds=False）",
             ha="left", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"网格 {g}" for g in gens])
    ax1.set_ylabel("低估 p95（underest_p95，越低越好）")
    ax1.set_title("A1 · 误差：只是划分方式在动，网格加密几乎无效")
    ax1.set_ylim(0, 1.25)
    ax1.legend(loc="upper center", fontsize=9, framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    # ---- 面板 A2：秩（解释成因）------------------------------------------
    for j, split in enumerate(SPLIT_ORDER):
        t, h = [], []
        for pm in parsed:
            v = pm.get((GRID_ORDER[0], split), {})
            t.append(v.get("train_min_rank"))
            h.append(v.get("holdout_min_rank"))
        base = x + (j - 0.5) * width
        ax2.bar(base, t, width, label=f"{SPLIT_LABEL[split]} · 训练最小秩",
                color=("#c0392b" if split == "hash" else "#2471a3"),
                edgecolor="black", linewidth=0.6)
        # 留出最小秩用叠加深色描边表示
        ax2.bar(base, h, width, color="none", edgecolor="black",
                linewidth=1.8, linestyle=(0, (2, 1.5)))
        for xx, hh in zip(base, h):
            if hh is not None:
                ax2.text(xx, -0.28, f"H={hh}", ha="center", va="top", fontsize=8)

    # 探针需要的最小秩（3 参数 ⇒ 满秩 3）
    ax2.axhline(3, color="green", linestyle=":", linewidth=1.2)
    ax2.text(len(gens) - 0.5, 3.06, "满秩 = 3（3 个参数）",
             ha="right", va="bottom", fontsize=9, color="green")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"网格 {g}" for g in gens])
    ax2.set_ylabel("最小秩（train 实心 / holdout 虚线框）")
    ax2.set_title("A2 · 成因：hash 的秩恒为 1，秩保证划分能拿到满秩")
    ax2.set_ylim(0, 3.9)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    src = " + ".join(d["name"] for d in decouple)
    fig.suptitle(f"P27 图 A · 2×2 解耦（网格 × 划分）　数据：{src}　"
                 f"每格 holdout_modulus={decouple[0]['modulus']}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    png = outdir / "fig_p27a_decouple.png"
    svg = outdir / "fig_p27a_decouple.svg"
    _save(fig, png, svg, plt)
    return png, svg


# --------------------------------------------------------------------------
# 图 B
# --------------------------------------------------------------------------
def fig_b(plt, lookup: dict[str, Any], outdir: Path) -> tuple[Path, Path]:
    import numpy as np

    cells = lookup["doc"]["cells"]
    pf_vals = sorted({c["pf_step"] for c in cells})
    kv_vals = sorted({c["kv_step"] for c in cells})

    cov = np.full((len(pf_vals), len(kv_vals)), np.nan)
    o95 = np.full((len(pf_vals), len(kv_vals)), np.nan)
    nbk = np.full((len(pf_vals), len(kv_vals)), np.nan)
    for c in cells:
        i = pf_vals.index(c["pf_step"])
        j = kv_vals.index(c["kv_step"])
        if c.get("coverage") is not None:
            cov[i, j] = c["coverage"]
        if c.get("over_p95") is not None:
            o95[i, j] = c["over_p95"]
        if c.get("n_buckets") is not None:
            nbk[i, j] = c["n_buckets"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # B1 覆盖率
    ax = axes[0]
    im = ax.imshow(cov, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title("B1 · 覆盖率（coverage，越高越好）")
    _annot(ax, cov, pf_vals, kv_vals, fmt="{:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # B2 超出 p95（红区 = >5%）
    ax = axes[1]
    im = ax.imshow(o95, cmap="RdYlGn_r", vmin=0, vmax=0.15, aspect="auto")
    ax.set_title(f"B2 · 超出 p95（over_p95，越低越好；> {UNDEREST_P95_THRESHOLD:.0%} 为红）")
    _annot(ax, o95, pf_vals, kv_vals, fmt="{:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # B3 桶数（结构成本）
    ax = axes[2]
    im = ax.imshow(nbk, cmap="Blues", aspect="auto")
    ax.set_title("B3 · 桶数（n_buckets，查表结构成本）")
    _annot(ax, nbk, pf_vals, kv_vals, fmt="{:.0f}")
    fig.colorbar(im, ax=ax, fraction=0.046)

    src = lookup["name"]
    fig.suptitle(
        f"P27 图 B · 查表分桶扫描（pf_step × kv_step，共 {len(cells)} 格）　"
        f"数据：{src}　标定 {lookup['doc'].get('calib')}　"
        f"shapes_tag={lookup['doc'].get('shapes_tag')}（n_records={lookup['doc'].get('n_records')}）",
        fontsize=10.5)
    # ★ 把机械结论与陷阱写进**图注**（必须在 savefig 之前，否则不落图）
    pf_inert = False
    if len(pf_vals) > 1:
        pf_inert = all(np.allclose(cov[i, :], cov[0, :], equal_nan=True)
                       for i in range(len(pf_vals)))
    footnote = ("[OK] 四个 pf_step 行逐字相同 => pf_step 惰性（不动覆盖率/误差）；只有 kv_step 在动"
                if pf_inert else
                "[!] 各行不全等 => **不得**照抄「pf_step 惰性」的读法")
    fig.text(0.5, 0.045, safe_plot_text(footnote), ha="center", va="bottom", fontsize=9.5,
             color=("#1e8449" if pf_inert else "#c0392b"))
    fig.text(0.5, 0.005,
             "[!] kv_step=0 的 coverage=0.000 是「一条形状都没命中」（实测 loo_n=124，非缺测）"
             "=> 该配置不可用；其 over_p95 最低**不代表**它最好",
             ha="center", va="bottom", fontsize=8.8)

    fig.tight_layout(rect=(0, 0.075, 1, 0.93))

    png = outdir / "fig_p27b_lookup_sweep.png"
    svg = outdir / "fig_p27b_lookup_sweep.svg"
    _save(fig, png, svg, plt)

    if len(pf_vals) > 1 and not pf_inert:
        print("[!] 与「pf_step 完全不动」的读法不符：各行不全等 ⇒ "
              "**必须**重新核对图 B 的结论，不得照抄旧读法。", file=sys.stderr)
    return png, svg


def _annot(ax, mat, pf_vals, kv_vals, fmt: str) -> None:
    import numpy as np
    for i in range(len(pf_vals)):
        for j in range(len(kv_vals)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=7.5, color="white" if v > (np.nanmax(mat) * 0.6) else "black")
    ax.set_yticks(range(len(pf_vals)))
    ax.set_yticklabels([f"pf={v}" for v in pf_vals])
    ax.set_xticks(range(len(kv_vals)))
    ax.set_xticklabels([f"kv={v}" for v in kv_vals], rotation=45, ha="right")
    ax.set_xlabel("kv 分桶步长 (kv_step)")
    ax.set_ylabel("prefill 分桶步长 (pf_step)")


# --------------------------------------------------------------------------
# .source.txt
# --------------------------------------------------------------------------
def _svglf(p: Path) -> None:
    """把 matplotlib 在 Windows 上写出的 CRLF 归一为 LF。

    踩过：`git add` 会对 `.svg` 报 "CRLF will be replaced by LF"，
    而且"内容对、字节错"的改动**没有报错**（本项目的老坑）。
    生成即归一，保证仓库里的字节稳定、diff 干净。
    """
    raw = p.read_bytes()
    if b"\r\n" in raw:
        p.write_bytes(raw.replace(b"\r\n", b"\n"))


def _save(fig, png: Path, svg: Path, plt) -> None:  # noqa: ANN001
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    _svglf(svg)


def _rel(p: Path) -> str:
    """仓库内路径用 POSIX 斜杠（跨平台、可直接复制进 markdown）。"""
    try:
        r = p.relative_to(REPO)
    except ValueError:
        r = p
    return str(r).replace("\\", "/")


def write_source(outdir: Path, plots: list[tuple[Path, Path]],
                 decouple: list[dict[str, Any]], lookup: dict[str, Any],
                 cmd: str) -> Path:
    lines = [
        "# P27 图的来源与重建（.source.txt）",
        "",
        "> 本文件由 `scripts/plot_step16_decouple.py` 自动生成，**请勿手改**。",
        "> 对应 `paper/sections/04_failure_analysis.md` 的【待补·P27】占位块。",
        "",
        "## 重建命令",
        "",
        "```bash",
        cmd,
        "```",
        "",
        '## 输入（内容哈希用于确认「图来自哪一份文件」）',
        "",
    ]
    for d in decouple:
        lines.append(f"* `{_rel(d['path'])}`")
        lines.append(f"  * sha256 = `{d['sha256']}`")
        lines.append(f"  * holdout_modulus = {d['modulus']}")
        lines.append(f"  * 格子数 = {len(d['cells'])}")
    lines.append(f"* `{_rel(lookup['path'])}`")
    lines.append(f"  * sha256 = `{lookup['sha256']}`")
    lines.append(f"  * calib = `{lookup['doc'].get('calib')}`")
    lines.append(f"  * shapes_tag = `{lookup['doc'].get('shapes_tag')}`")
    lines.append(f"  * n_records = {lookup['doc'].get('n_records')}, "
                 f"n_shapes = {lookup['doc'].get('n_shapes')}, "
                 f"格子数 = {len(lookup['doc']['cells'])}")
    lines += [
        "",
        "## 溯源标签（run_id / tag）",
        "",
        "⚠️ 这两份 JSON **本身**是 `scripts/` 的离线产物，其 run_id 溯源在各自的",
        "生成脚本与登记行里；**图不引入任何新 run**。按占位块要求，此处列出其所引用的标签：",
        "",
        "* 图 A（decouple）：标定/评估族 —— 见 `results/processed/decouple_*.json` 内的",
        "  `eval_json` 字段（每格一个，指向被评估的成本模型评测产物）。",
        "* 图 B（lookup sweep）：`shapes_tag` = "
        f"`{lookup['doc'].get('shapes_tag')}`（形状普查批），`calib` = "
        f"`{lookup['doc'].get('calib')}`。",
        "",
        "## 产物",
        "",
    ]
    for png, svg in plots:
        lines.append(f"* `{png.name}` / `{svg.name}`")
    lines += [
        "",
        "## 读法（**照抄结论，不要在图里重新解释**）",
        "",
        "* 图 A：`hash` 列两代几乎不动（≈95%），`rank_guaranteed` 列随网格加密而下降",
        "  ⇒ **主因是划分方式，不是网格密度**。",
        "* 图 B：`pf_step` 的各行**逐字相同**（惰性），只有 `kv_step` 在动",
        "  ⇒ 「范围外回退」的瓶颈在 **kv 桶**，不在 prefill 桶。",
        "",
        "⚠️ **陷阱（必须一起写）**：`kv_step=0` 的四格 `coverage = 0.000`，",
        "而它们的 `over_p95` 恰好**最低**（0.0013）——这会误导成「不分桶最准」。",
        "实际是 **coverage=0 = 一条形状都没命中查表**（`loo_n=124`、`loo_skipped=0`，",
        "是**实测的 0**，不是缺测）⇒ 该配置**不可用**；其低误差只反映「被评估到的那些形状」。",
        "**不得**据 `kv_step=0` 的低 `over_p95` 说它最好。",
        "",
        "⚠️ 阈值与判据（低估 p95 > 5% ⇒ `model_holds=False`）来自**预注册**，",
        "本脚本只照抄，不新造。",
        "",
    ]
    p = outdir / "fig_p27.source.txt"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return p


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="第16步 P27 两张图（只读、可重建）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录（默认 paper/figures）")
    ap.add_argument("--processed", default=str(PROCESSED), help="results/processed 路径")
    ap.add_argument("--check", action="store_true", help="只校验输入，不画图")
    args = ap.parse_args(argv)

    processed = Path(args.processed)
    if not processed.is_dir():
        print(f"✗ 找不到 {processed}", file=sys.stderr)
        return 2

    try:
        decouple = load_decouple(processed)
        lookup = load_lookup(processed)
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    print(f"✓ 输入齐：decouple ×{len(decouple)}、lookup ×1")
    for d in decouple:
        print(f"    {d['name']}: {len(d['cells'])} 格, modulus={d['modulus']}, "
              f"sha256={d['sha256'][:12]}…")
    print(f"    {lookup['name']}: {len(lookup['doc']['cells'])} 格, "
          f"n_shapes={lookup['doc'].get('n_shapes')}, sha256={lookup['sha256'][:12]}…")

    if args.check:
        print("✓ --check 通过（未画图）")
        return 0

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    plt, font = setup_matplotlib()
    print(f"  字体：{font or '（回退 DejaVu Sans，中文可能缺字）'}")

    pa = fig_a(plt, decouple, outdir)
    pb = fig_b(plt, lookup, outdir)
    cmd = ("python scripts/plot_step16_decouple.py "
           f"--processed {_rel(processed)} --out {_rel(outdir)}")
    srcp = write_source(outdir, [pa, pb], decouple, lookup, cmd)

    for p in (*pa, *pb):
        print(f"  写出 {p}  ({p.stat().st_size} bytes)")
    print(f"  写出 {srcp}")

    # 简单 sanity：文件非空
    for p in (*pa, *pb, srcp):
        if not p.is_file() or p.stat().st_size == 0:
            print(f"[!] 产物为空：{p} —— **不得**当作成功", file=sys.stderr)
            return 1
    print("✓ P27 两张图已重建（含 .source.txt）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
