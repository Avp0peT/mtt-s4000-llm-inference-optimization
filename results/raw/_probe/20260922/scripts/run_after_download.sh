#!/bin/bash
# 等 7B 权重下载完，然后跑 Qwen2.5-7B-Instruct 单请求验证。
#
# 为什么用脚本文件而不是在 SSH 命令里内联：内联要先过一层本地 shell、再过一层远端
# shell，引号会被吃掉两层，`$!`、通配符、嵌套单引号都可能在远端被解释成别的东西。
# 踩过：报 `syntax error near unexpected token ')'`，而且看不出来是哪一层的引号错了。
set -u

MODEL="Qwen/Qwen2.5-7B-Instruct"
RUN1=/root/ascend_probe/results/qwen_7b_run1.json
RUN2=/root/ascend_probe/results/qwen_7b_run2.json

echo "[chain] 等待下载进程退出 $(date -Is)"
n=0
while pgrep -f "huggingface-cli download" >/dev/null; do
    sleep 15
    n=$((n + 1))
    if [ "$n" -ge 240 ]; then          # 60 分钟上限，避免无限等
        echo "[chain] 等待超时，放弃"
        exit 1
    fi
done
echo "[chain] 下载进程已退出 $(date -Is)"
df -h /root/autodl-tmp | tail -1

# 确认权重真的齐了，别把"下载失败"当成"下载完成"
source /root/musa-envs/qwen310/activate_musa.sh

python - <<'PYEOF'
import glob, os, sys
from huggingface_hub import snapshot_download
try:
    p = snapshot_download("Qwen/Qwen2.5-7B-Instruct",
                          allow_patterns=["*.safetensors", "*.json", "*.txt"])
except Exception as exc:
    print(f"[chain] 权重不完整：{type(exc).__name__}: {exc}")
    sys.exit(1)
shards = sorted(glob.glob(os.path.join(p, "*.safetensors")))
total = sum(os.path.getsize(s) for s in shards)
print(f"[chain] 缓存目录 {p}")
print(f"[chain] safetensors 分片 {len(shards)} 个，合计 {total / 2**30:.2f} GiB")
if len(shards) != 4 or total < 14 * 2**30:
    print("[chain] 分片数或体积异常，判定为未下载完整")
    sys.exit(1)
print("[chain] 权重校验通过")
PYEOF

if [ $? -ne 0 ]; then
    echo "[chain] 权重校验失败，不启动验证"
    exit 1
fi

cd /root/ascend_probe

# run1：权重刚下完，页缓存是冷的 —— 冷载入耗时单独留档，与 run2 对比即可分离
#       "冷盘代价"和"设备差异"。不带 --compare-cpu，先拿到干净的显存/时延数字。
echo "[chain] === run1（冷载入）开始 $(date -Is) ==="
python scripts/qwen_single_request.py --model "$MODEL" --repeats 3 --out "$RUN1"
echo "[chain] === run1 结束 rc=$? $(date -Is) ==="

# run2：页缓存已热。带 --compare-cpu 做 fp32 首前向的 MUSA-vs-CPU 对照。
#       脚本设计上先跑完 fp16 的全部测量、之后才载入 fp32 模型，故 fp16 的
#       显存/时延数字不会被这次对照污染。
echo "[chain] === run2（热载入 + fp32 正确性对照）开始 $(date -Is) ==="
python scripts/qwen_single_request.py --model "$MODEL" --repeats 3 --compare-cpu --out "$RUN2"
echo "[chain] === run2 结束 rc=$? $(date -Is) ==="

echo "[chain] 全部完成 $(date -Is)"
