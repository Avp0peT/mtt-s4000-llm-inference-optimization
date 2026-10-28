#!/bin/bash
# 预下载 Qwen2.5-7B-Instruct 到 HF 缓存。
#
# 为什么要"预下载"而不是让 from_pretrained 边下边载：
#   验证脚本要报告 load_s（载入耗时）。若权重不在本地，这个数字里混着网络传输，
#   就不是"载入"而是"下载"。先把权重落盘，计时才是纯磁盘读取。
#
# 排除项说明：
#   original/*  —— 仓库里的原始 PyTorch 权重（.bin + .safetensors 两份），体积翻倍且用不到
#   *.pth       —— 原始 transformer 格式权重
#   *.h5 *.msgpack *.ot —— TF / Flax 格式，本项目不用
# 保留 config / tokenizer / generation_config / *.safetensors 及其 index。
set -u

source /root/musa-envs/qwen310/activate_musa.sh

echo "[download] HF_ENDPOINT=$HF_ENDPOINT"
echo "[download] HF_HOME=$HF_HOME"
df -h /root/autodl-tmp | tail -1

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --exclude "original/*" "*.pth" "*.h5" "*.msgpack" "*.ot"

echo "[download] huggingface-cli 退出码=$?"

echo "=== 缓存落盘情况 ==="
find "$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct" -name "*.safetensors" -o -name "*.json" | sort
du -sh "$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct"
df -h /root/autodl-tmp | tail -1
