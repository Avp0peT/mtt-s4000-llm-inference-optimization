#!/usr/bin/env bash
# 跑基线对照
# ---------------------------------------------------------------------------
# 顺序很重要：先采环境指纹，再跑实验。反过来会把 run 造成的状态
# （频率、显存碎片）记成「环境」。
#
# 用法：
#   bash scripts/run_baseline.sh                 # 真机路径
#   bash scripts/run_baseline.sh --dry-run       # 离线模拟，验证链路
#   bash scripts/run_baseline.sh --config configs/baseline/default.yaml --repeats 5
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY="python"

CONFIG="configs/baseline/default.yaml"
EXTRA_ARGS=()

# 解析参数：--config 单独抽出，其余原样透传给 run_benchmark
while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "== 1/3 采集环境指纹 =="
bash scripts/inspect_env.sh

echo
echo "== 2/3 执行基准 ($CONFIG) =="
"$PY" -m benchmark.run_benchmark --config "$CONFIG" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo
echo "== 3/3 聚合指标 =="
if [ "${#EXTRA_ARGS[@]}" -gt 0 ] && [ "${EXTRA_ARGS[0]:-}" = "--dry-run" ]; then
    # dry-run 产物默认不进汇总表；这里显式提示，而不是让它静默被排除
    echo "本次为 dry-run，聚合时需显式加 --include-dry-run，否则不会有任何 run 被处理。"
    "$PY" benchmark/collect_metrics.py --include-dry-run
else
    "$PY" benchmark/collect_metrics.py
fi

echo
echo "完成。产物：results/raw/ 与 results/processed/"
