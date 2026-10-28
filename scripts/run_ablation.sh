#!/usr/bin/env bash
# 跑消融实验（baseline + 各优化维度的单变量对照）
# ---------------------------------------------------------------------------
# 消融的纪律（对应 AGENTS.md 附录 A4 与正文第 5 节）：
#   - 一次只改一个维度，因此这里逐个配置**串行**跑，不并行。
#     并行会互相抢资源，把「变量效果」和「资源竞争」混在一起。
#   - 每个配置都必须完成 >=3 次重复；少于 3 次的差异不能与噪声区分。
#   - 中途失败不中断整轮（记录失败点继续），否则一次环境抖动就要全部重跑。
#
# 用法：
#   bash scripts/run_ablation.sh                    # 真机路径
#   bash scripts/run_ablation.sh --dry-run           # 链路验证
#   ADAPTIVE=1 bash scripts/run_ablation.sh          # 额外带上自适应配置
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY="python"

DRY_FLAG=""
if [ "${1:-}" = "--dry-run" ]; then
    DRY_FLAG="--dry-run"
fi

# 每个配置单独跑；顺序固定，便于按顺序读日志
CONFIGS=(
    "configs/baseline/default.yaml"
    "configs/prefix_cache/prefix_cache.yaml"
    "configs/chunked_prefill/chunked_4k.yaml"
)

# 量化与图模式在 dry-run 下不生效（见配置头部注释），因此默认不纳入模拟消融；
# 真机路径下它们才有意义。
if [ -z "$DRY_FLAG" ]; then
    CONFIGS+=(
        "configs/quantization/w8a8.yaml"
        "configs/graph_mode/acl_graph.yaml"
    )
fi

if [ "${ADAPTIVE:-0}" = "1" ]; then
    CONFIGS+=("configs/adaptive/adaptive_v1.yaml")
fi

echo "== 采集环境指纹 =="
bash scripts/inspect_env.sh || echo "警告：环境采集失败，继续执行（产物中会标记环境未知）"

FAILED=()
for cfg in "${CONFIGS[@]}"; do
    echo
    echo "== 消融：$cfg =="
    if "$PY" -m benchmark.run_benchmark --config "$cfg" ${DRY_FLAG}; then
        echo "  OK"
    else
        echo "  失败：$cfg（已记录，继续下一个）"
        FAILED+=("$cfg")
    fi
done

echo
echo "== 聚合 =="
if [ -n "$DRY_FLAG" ]; then
    "$PY" benchmark/collect_metrics.py --include-dry-run
else
    "$PY" benchmark/collect_metrics.py
fi

echo
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "以下配置失败，必须在 EXPERIMENTS.md 中登记失败原因："
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    exit 2
fi

echo "全部完成。下一步：核对 results/processed/repeats.csv，"
echo "确认 sufficient_repeats 全为 True 且跨重复离散度小于待解释的效应量。"
