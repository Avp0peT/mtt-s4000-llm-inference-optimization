#!/usr/bin/env bash
# 导出结果包：把一次聚合结果打成一个可交付的目录，并尽力生成图。
# ---------------------------------------------------------------------------
# 为什么需要一个"导出"步骤：
#   results/raw 是按 run 平铺的原始数据，体量大且混杂（含 dry-run）。
#   论文/汇报需要的是一个**自洽快照**：某天的汇总表 + 当时的配置 + 环境指纹。
#   导出的目录一旦生成就冻结，之后 raw 再变也不影响已交付的结论。
#
# 用法：
#   bash scripts/export_results.sh                 # 用默认 tag（日期时间）
#   bash scripts/export_results.sh --tag v0.1      # 自定义 tag
#   INCLUDE_DRY=1 bash scripts/export_results.sh   # 纳入非 MEASURED 数据（仅在核对口径时）
#
# 退出码：
#   0  成功
#   3  汇总表里没有一条 MEASURED（无可交付的结论）
#   4  前置校验未通过（缺原始明细 / 环境指纹 / 有效重复 / 实验登记）
#
# 实现约定：bash 只做编排，路径拼接与文件操作一律交给 Python。
# 原因：bash 传进来的路径在不同平台形态不同（Git Bash 是 /c/...，
# Linux 是 /home/...），直接塞进 Python 会拼出错误的路径。
# Python 侧统一用 Path.cwd() 作为仓库根——脚本开头已经 cd 到根目录。
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY="python"

TAG="$(date +%Y%m%d-%H%M%S)"
if [ "${1:-}" = "--tag" ] && [ -n "${2:-}" ]; then
    TAG="$2"
fi

EXPORT_DIR="results/export/$TAG"
mkdir -p "$EXPORT_DIR"

echo "== 1/5 重新聚合（保证导出与 raw 一致）=="
if [ "${INCLUDE_DRY:-0}" = "1" ]; then
    "$PY" benchmark/collect_metrics.py --include-non-measured
else
    "$PY" benchmark/collect_metrics.py
fi

echo
echo "== 2/5 导出前置校验（原始明细 / 环境指纹 / 有效重复 / 实验登记）=="
# 这一步的存在理由：这四项缺了**都不会报错**——汇总表照样生成、图照样画出来、
# 导出包照样写着"完成"。所以必须在这里被挡住，不能留到被质疑时才发现。
# INCLUDE_DRY=1 时走 --lenient：那是诊断包不是交付物，可追溯性要求不适用，
# 但缺原始明细仍然是阻断项（数字得能对上明细）。
if [ "${INCLUDE_DRY:-0}" = "1" ]; then
    PREFLIGHT_FLAGS="--lenient"
else
    PREFLIGHT_FLAGS=""
fi
if ! "$PY" benchmark/preflight.py $PREFLIGHT_FLAGS; then
    echo >&2
    echo "前置校验未通过，导出中止（退出码 4）。" >&2
    echo "  补齐依据后重跑；若只是要一份链路口径诊断包，用：" >&2
    echo "  INCLUDE_DRY=1 bash scripts/export_results.sh --tag <tag>" >&2
    exit 4
fi

echo
echo "== 3/5 拷贝汇总表、配置快照与日志 =="
# 导出目录通过环境变量传给 Python，避免在 bash 与 Python 之间来回拼路径。
EXPORT_DIR="$EXPORT_DIR" "$PY" - <<'PYEOF'
"""校验并拷贝导出内容。

关键校验：汇总表里必须至少有**一条 MEASURED 的 run**。
只有模拟数据的导出会产出一个"看起来成功、实际没有任何可用数字"的包，
那种包比直接报错更危险。

过滤依据是 `data_label` 而不是 `dry_run`：dry_run 只是数据不可用的**一种**原因，
HARDWARE_REQUIRED（本机无 NPU）与 NOT_MEASURED（环境未知）同样不能进交付包，
但它们 dry_run=false，用 dry_run 过滤会漏掉。
"""
import csv
import os
import shutil
import sys
from pathlib import Path

root = Path.cwd()                      # 脚本已 cd 到仓库根
export_dir = Path(os.environ["EXPORT_DIR"]).resolve()
include_dry = os.environ.get("INCLUDE_DRY", "0") == "1"
export_dir.mkdir(parents=True, exist_ok=True)

summary = root / "results" / "processed" / "summary.csv"
if not summary.exists() or summary.stat().st_size == 0:
    print("错误：results/processed/summary.csv 为空或不存在 —— 没有可导出的 run。",
          file=sys.stderr)
    print("  先跑 benchmark/run_benchmark.py 生成产物。", file=sys.stderr)
    raise SystemExit(3)

with summary.open(encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))


def _label(row: dict) -> str:
    return (row.get("data_label") or "").strip()


measured = [r for r in rows if _label(r) == "MEASURED"]
unlabelled = [r for r in rows if not _label(r)]

if not measured and not include_dry:
    counts: dict[str, int] = {}
    for r in rows:
        counts[_label(r) or "<无标签>"] = counts.get(_label(r) or "<无标签>", 0) + 1
    print(f"错误：汇总表里 {len(rows)} 个 run 没有一个标签是 MEASURED。", file=sys.stderr)
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}", file=sys.stderr)
    print("  非实测数据不进交付包（AGENTS.md 正文第 10 节）。", file=sys.stderr)
    print("  仅在核对链路口径时才用：INCLUDE_DRY=1 bash scripts/export_results.sh",
          file=sys.stderr)
    raise SystemExit(3)

if unlabelled:
    # 老产物没有 data_label 列。不猜它们是不是实测，直接把风险说出来。
    print(f"  警告：{len(unlabelled)} 行没有 data_label 列（旧版本产物）。"
          f"这些行能否进结论无法由本脚本判定，请重新生成产物。",
          file=sys.stderr)

shutil.copy2(summary, export_dir / "summary.csv")
repeats = root / "results" / "processed" / "repeats.csv"
if repeats.exists():
    shutil.copy2(repeats, export_dir / "repeats.csv")

# 配置快照：没带配置的导出包无法复核实验条件
src_configs = root / "configs"
if src_configs.is_dir():
    shutil.copytree(src_configs, export_dir / "configs", dirs_exist_ok=True)

env_json = root / "results" / "raw" / "_env" / "env.json"
if env_json.exists():
    shutil.copy2(env_json, export_dir / "env.json")
else:
    print("  提示：未找到环境指纹，导出包中不会有 env.json（这本身就是信息）")

for name in ("EXPERIMENTS.md", "RESEARCH_LOG.md"):
    src = root / name
    if src.exists():
        shutil.copy2(src, export_dir / name)

print(f"  已拷入 {len(rows)} 行汇总（其中 MEASURED {len(measured)} 个）")
PYEOF

echo
echo "== 4/5 生成图（需要 matplotlib，缺失则跳过）=="
mkdir -p results/figures
# 绘图逻辑在 benchmark/plot_p99.py 里（而不是内嵌 heredoc）——因为内嵌版本
# 无法被单元测试覆盖，曾让"空分位块被画成 0"与"跨负载混进同一个中位数"
# 这两个缺陷静默存活。抽成模块后两者都有测试兜住。
"$PY" benchmark/plot_p99.py --summary "$EXPORT_DIR/summary.csv" --figures-dir results/figures \
    || echo "  绘图失败，已跳过（汇总表仍可用）"

echo
echo "== 5/5 写导出说明 =="
{
    echo "# 结果导出 $TAG"
    echo
    echo "生成时间：$(date '+%Y-%m-%d %H:%M:%S')"
    echo
    echo "内含："
    echo "- summary.csv：每个 run 一行（原始口径，含 data_label 列）"
    echo "- repeats.csv：跨重复聚合，含 sufficient_repeats 与离散度"
    echo "- configs/：本次导出时的全部配置快照"
    echo "- env.json：硬件与软件环境指纹"
    echo "- EXPERIMENTS.md / RESEARCH_LOG.md：当时的实验登记与研究日志"
    echo
    echo "使用提醒："
    echo "- **只有 data_label=MEASURED 的行才能进入结论与论文表格。**"
    echo "  TEST_DATA=模拟数据；HARDWARE_REQUIRED=本机无昇腾设备；NOT_MEASURED=未测到。"
    echo "- label_source=legacy_inferred 表示标签由 dry_run 回推，不是权威标签。"
    echo "- sufficient_repeats=false 的组不能用于结论（重复次数 < 3）。"
    echo "- 若 env.json 缺失或 npu-smi 不可用，本包中的性能数字不具备跨环境可比性。"
    echo "- 本包已通过导出前置校验（原始明细 / 环境指纹 / 有效重复 >= 3 / 实验登记四项）；"
    echo "  缺任何一项都会在导出阶段被阻断（退出码 4），不会产出「看起来成功」的包。"
    if [ "${INCLUDE_DRY:-0}" = "1" ]; then
        echo "- **本包含非 MEASURED 数据**，仅可用于链路口径核对，不得进入论文。"
    fi
} > "$EXPORT_DIR/README.md"

echo
echo "导出完成：$EXPORT_DIR"
ls -1 "$EXPORT_DIR"
