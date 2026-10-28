#!/usr/bin/env bash
# 唯一职责：**等上游链退出，然后立刻启动 D1 队列** —— 用来消除空转。
# ---------------------------------------------------------------------------
# 为什么需要它：链与链之间若靠**本机定时任务**衔接，就会在"上游结束"与"下一个触发点"
# 之间空转（实测差 1h40m）。让**远端自己等**，空转趋近于零。
#
# 它本身只轮询 `ps`，**不占 GPU**，因此可以与上游链同时存在。
# ⚠️ 但它**只**负责等待与衔接：不做任何测量、不改任何产物。
#
# 前置（不满足就中止，不硬启）：
#   1. 本根下有 `SOURCE_PROVENANCE.json`（否则产物的 `git_commit` 会是 `NOT_MEASURED`）；
#   2. `scripts/run_s14_d1.sh` 存在。
#
# 终止标记：D1_WAIT_ABORTED_<原因>
# ---------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$0")/.." || exit 9
LOG=/root/s14_wait.log
exec >>"$LOG" 2>&1

echo "=============================================================="
echo "[wait] 开工 $(date)｜cwd=$(pwd)"

# 判据面唯一权威 = check_remote_busy.py（此前这里写死了另一份正则 ⇒ 两份判据分叉）。
# ⚠️ 本脚本运行在**远端** ⇒ 必须 `--local`（默认模式会再去 SSH 远端，远端自连
#    缺 REMOTE_* 凭据 ⇒ UNKNOWN ⇒ 永远判"忙"）。
# ⚠️ `busy()` 是**轮询谓词**：退出码 0=空闲 / 1=忙 / 2=查不出。
#    查不出（链路抖动）时**按"忙"处理** —— 宁可多等一轮，也不要因为探测失败
#    就误判"上游跑完了"而提前启动 D1（那会造成并发污染 results/raw）。
busy() {
    python3 scripts/check_remote_busy.py --local --expect-idle >/dev/null 2>&1
    local rc=$?
    [ "$rc" -ne 0 ]
}

if [ ! -f SOURCE_PROVENANCE.json ]; then
    echo "[wait] ✗ 本根缺 SOURCE_PROVENANCE.json ⇒ 产物的 git_commit 会是 NOT_MEASURED"
    echo "D1_WAIT_ABORTED_NO_STAMP"
    exit 9
fi
if [ ! -f scripts/run_s14_d1.sh ]; then
    echo "[wait] ✗ 缺 scripts/run_s14_d1.sh"
    echo "D1_WAIT_ABORTED_NO_SCRIPT"
    exit 9
fi

# 最多等 6 小时 —— ⚠️ **用墙钟截止时间，不要用计数器**
# 踩过：我原先写 `for i in $(seq 1 60); do ... sleep 30 ...` 并注释成"6 小时"，
# 实际 60 × 30s = **30 分钟**。后果很具体：等待器 21:10 就打印"超时（>6h）"退出，
# 而 v6 链还在跑 ⇒ **D1 从未启动，服务器空转约 2 小时**。
WAIT_HOURS=${WAIT_HOURS:-6}
DEADLINE=$(( $(date +%s) + WAIT_HOURS * 3600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if busy; then
        sleep 30
        continue
    fi
    # 连续两次（间隔 30s）都空闲才算真空闲 —— 上游可能在两步之间短暂无子进程
    sleep 30
    busy || { echo "[wait] 上游已连续两次探测为空闲 $(date)"; break; }
done

if busy; then
    echo "[wait] 超时（>${WAIT_HOURS}h）上游仍在跑 ⇒ 不启动，交由本机任务处理"
    echo "D1_WAIT_ABORTED_TIMEOUT"
    exit 0
fi

echo "[wait] 仍在本根的进程（应只剩我自己）："
ps -eo pid,etime,cmd | grep -E "bash|python" | grep -v grep | grep -v "ps -eo" | head -6

python - <<'PY'
import json
p = json.load(open("SOURCE_PROVENANCE.json", encoding="utf-8"))
keys = ("git_commit", "git_dirty", "digest", "scope_digest", "recorded_at")
print("[wait] 戳:", {k: p.get(k) for k in keys if k in p})
if p.get("git_dirty"):
    print("[wait] ⚠️ 戳记的是 dirty=true ⇒ 那批产物**不得**与其它批次合并解释")
PY

python -c "import torch, torch_musa; print('[wait] torch', torch.__version__, 'musa ok')" \
    || { echo "[wait] ✗ 本 shell 没激活 MUSA 环境 ⇒ 让 run_s14_d1.sh 自己去激活"; }

echo "[wait] 启动 D1 队列 $(date)"
# 起跑走唯一入口 launch_batch.sh（取锁 + 只发一次 + 监视进程负责放锁）。
# ⚠️ 此前是裸 `setsid nohup ... &` ⇒ 无锁；本机侧的重试曾把同一条命令发两遍。
bash scripts/launch_batch.sh S14D1 /root/s14_d1.log \
    bash scripts/run_s14_d1.sh
LAUNCH_RC=$?
echo "[wait] D1 起跑退出码 = $LAUNCH_RC（0=成功 4/5=被锁拦 7=起跑失败）"
if [ "$LAUNCH_RC" -ne 0 ]; then
    echo "[wait] ⚠️ D1 未成功起跑（rc=$LAUNCH_RC）"
    echo "D1_WAIT_ABORTED_LAUNCH_FAILED"
    exit 0
fi
sleep 8
echo "[wait] D1 日志前几行："
head -12 /root/s14_d1.log 2>/dev/null || echo "（还没有日志）"
echo "[wait] 完工 $(date)"
