#!/bin/bash
# ============================================================================
# 自主续跑队列（第 14 步 · 批次 2 收尾 → 批次 3 `lenmix`）
# ----------------------------------------------------------------------------
# 与既有编排脚本的关系（**不覆盖、不修改任何旧脚本**）：
#   /root/run_s14_night.sh / run_s14_v5.sh / run_s14_v6.sh 均指向**旧根**
#   `/root/ascend_probe` 与旧 tag（v5/v6/S14E）⇒ **本脚本不复用它们**
#   （旧 v6 脚本还把 `default_mode=lookup` 硬写进 v6 模型、并跑 `S14E`）。
#   本脚本是**当前唯一**的续跑入口，落在独立路径 `/root/autoq/`。
#
# ⚠️ 四条硬约束（来自主控指令）：
#   1. **不修改正在运行的批次的源码/配置/模型/戳** —— 本脚本只读它们。
#      自身落在 `/root/autoq/`，**不在 `SOURCE_DIGEST_SCOPE`
#      (`src/ascend_opt`,`benchmark`,`scripts`) 内** ⇒ 投放它不动溯源戳。
#   2. **新运行一律用新 tag**（批次 3 = `S14B3`），绝不与 `S14B2` 混算。
#   3. **闸门绑定本次批次**：只认 `S14B2-*` 目录的**自带 `meta.json`**
#      里的 `data_label`（契约已在真机核过：`meta.data_label`，回退
#      `meta.summary.data_label`，见 `benchmark/run_benchmark.py:842-859`）。
#      **禁止**用历史日志里的 PASS 或旧 JSON 触发。
#   4. 同一故障**最多自动重试一次**；失败则停止该分支、保留现场。
#
# 启动（远端）：setsid nohup bash /root/autoq/run_auto_queue.sh >> /root/autoq/queue.log 2>&1 &
# 状态：/root/autoq/state.json（机器可读，每阶段覆写）/ state.txt（追加式人类可读）
# ============================================================================
set -u

ROOT=/root/ascend_probe_d1
AUTOQ=/root/autoq
STATE="$AUTOQ/state.json"
STATETXT="$AUTOQ/state.txt"
QLOG="$AUTOQ/queue.log"
TAG2=S14B2
TAG3=S14B3
ACT=/root/musa-envs/qwen310/activate_musa.sh

LENMIX_CELLS="lenmix_static lenmix_threshold lenmix_cost lenmix_slo_weave"
EXPECT2=20
EXPECT3=20

# ---- 起跑器能力探测（★ 必须用**远端实际部署**的版本，不是本机仓库的版本）----
# 踩过：本机 `launch_batch.sh` 已有 `--activate`（本地 commit 60f8fcc），
# 但**远端部署树**停在 `7051ea62`（源码戳口径），那份**没有** `--activate`。
# 于是队列按新版调用 ⇒ 旧脚本把 `--activate` 当成 <tag> ⇒
# `batch_lock.py acquire --tag --activate` ⇒ argparse 报错 rc=2 ⇒ 批次 3 起不来。
# 教训：**编排脚本的能力必须以部署树为准**，不能以本机仓库为准。
#
# ⚠️ 探测必须**只看可执行代码**，不能 grep 整个文件：
#    第一版写成 `grep -q -- '--activate' "$file"`，结果**匹配到了自己的注释文本**
#    （注释里写了"无 --activate"）⇒ 对旧脚本也报"支持"⇒ 探测失效且**不报错**。
#    故这里先去掉整行注释与行尾注释，再匹配。
LAUNCH_SH="$ROOT/scripts/launch_batch.sh"
supports_activate() {
    [ -f "$LAUNCH_SH" ] || return 1
    # 去掉：行首（可含空白）# 开头的整行注释；以及行内未被引号保护的 # 之后的部分。
    # 本脚本的调用形态是 `--activate` 或 `--activate=`，都出现在代码位置。
    sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$LAUNCH_SH" \
        | grep -q -- '--activate'
}

# 统一的起跑封装：无论起跑器支持哪种形式，都产出**同样的 argv 语义**。
#   - 支持 --activate  ⇒ 用原生开关
#   - 不支持          ⇒ 自己用 `bash -c 'source …; exec …'` 包起来
#     （起跑器内部是 `setsid nohup "$@"`，把 `bash -c …` 当命令传完全成立）
launch_batch() {  # launch_batch <tag> <logfile> <cmd...>
    local tag="$1" log="$2"; shift 2
    if supports_activate; then
        bash "$LAUNCH_SH" --activate "$ACT" "$tag" "$log" "$@"
    else
        bash "$LAUNCH_SH" "$tag" "$log" \
            bash -c 'source "$1" || { echo "[activate] ✗ source 失败：$1" >&2; exit 7; }; shift; exec "$@"' \
            _ "$ACT" "$@"
    fi
}

mkdir -p "$AUTOQ"
cd "$ROOT" || { echo "[autoq] ✗ 无法进入 $ROOT"; exit 6; }

st() {  # st <phase> <status> <note>
    printf '{"phase":"%s","status":"%s","note":"%s","ts":"%s"}\n' \
        "$1" "$2" "$3" "$(date -Iseconds)" > "$STATE"
    printf '[%s] %-10s %-6s %s\n' "$(date -Iseconds)" "$1" "$2" "$3" >> "$STATETXT"
    echo "[autoq] $1 $2 $3"
}

# ---- 产物计数：要求目录里**真有实质文件**（防"目录建了但写空"）----
count_num() {  # count_num <TAG>
    local tag="$1" n=0 d
    for d in "$ROOT"/results/raw/${tag}-*; do
        [ -d "$d" ] || continue
        [ -s "$d/meta.json" ] && n=$((n+1))
    done
    echo "$n"
}

# ---- 标签普查：绑定**本批**，逐目录读自带 meta.json ----
# 输出形如 "MEASURED=20 OTHER=0 DETAIL=<无>"；DETAIL 只列异常项（最多 5 个）
label_census() {  # label_census <TAG>
    local tag="$1"
    python3 - "$ROOT" "$tag" <<'PY'
import json, os, sys, glob
root, tag = sys.argv[1], sys.argv[2]
m = o = 0
bad = []
for d in sorted(glob.glob(os.path.join(root, "results/raw", f"{tag}-*"))):
    if not os.path.isdir(d):
        continue
    p = os.path.join(d, "meta.json")
    lab = None
    if os.path.exists(p):
        try:
            j = json.load(open(p, encoding="utf-8"))
            lab = j.get("data_label")
            if lab is None:
                lab = (j.get("summary") or {}).get("data_label")
        except Exception as e:
            lab = f"PARSE_ERR:{type(e).__name__}"
    if lab == "MEASURED":
        m += 1
    else:
        o += 1
        if len(bad) < 5:
            bad.append(f"{os.path.basename(d)}={lab!r}")
print(f"MEASURED={m} OTHER={o} DETAIL={';'.join(bad) if bad else '-'}")
PY
}

# ---- 等矩阵进程结束（判据 = 进程表 + 产物活性，不是纯日志）----
#
# ⚠️ 2026-09-28 §52 修正（原实现有**假超时**缺陷）：
#   原来是 `wait_proc_gone 480`（480 分钟 = 8 h 墙钟上限），
#   而 S14B3 的真实时长 = 20 次 × 中位 23.6 min = **7.87 h**，
#   再加起跑开销（06:14 起跑、06:36 才出第 1 个产物，启动段 22 min）
#   ⇒ **必然撞上限**，于是队列打印 `DONE_ABORTED_B3_TIMEOUT` 并 `exit 0`，
#   而 `run_step3_matrix.py` **仍在跑**（setsid+nohup，没被杀）。
#   后果：① 后续逻辑会把"仍在跑"误读成"已放弃"；
#         ② 批次变成**无人看护**；③ 下游"回拉慢"的因果归因会被带偏
#            （负载未降 ⇒ 与"批次结束"这个前提矛盾）。
#
# 新判据（两条同时看，比墙钟可靠）：
#   ① **进程消失** ⇒ 完成；
#   ② 进程在、但**产物在规定窗口内不再增长** ⇒ 判定挂死（真正的异常）。
#      这样"跑得慢"不再等于"超时"，只有"没进展"才算失败。
#
# :param max_stall_minutes: 允许的**无进展**上限（默认 60 min）。
#        单次运行中位 23.6 min、最坏 34 min，取 60 min 留足余量。
wait_proc_gone() {  # wait_proc_gone [max_stall_minutes]
    local maxstall="${1:-60}" i=0 stall=0 last="" cur=""
    local probe_last=$((maxstall*6))   # 每 10 s 一次
    while :; do
        if ! pgrep -f "run_step3_matrix.py" >/dev/null 2>&1; then
            return 0                    # ① 进程结束 ⇒ 正常完成
        fi
        # ② 活性探针：产物目录数 + 最近 mtime，两者都不变才算"无进展"
        cur=$(ls -d "$ROOT"/results/raw/"${TAG3}"-* 2>/dev/null | wc -l)
        cur="$cur:$(find "$ROOT"/results/raw -maxdepth 2 -name "${TAG3}-*" \
                     -newermt "-${maxstall} minutes" 2>/dev/null | wc -l)"
        if [ "$cur" = "$last" ]; then
            stall=$((stall+1))
        else
            stall=0; last="$cur"
        fi
        if [ "$stall" -ge "$probe_last" ]; then
            st B_wait STALL "进程仍在但产物 ${maxstall} min 无进展 ⇒ 判定挂死"
            return 1
        fi
        sleep 10; i=$((i+1))
    done
}

# ---- 等全局锁释放（监视进程在批次结束后放锁）----
wait_unlock() {
    local i=0
    while [ "$i" -lt 120 ]; do
        [ -d /root/batch_locks/all.lock.d ] || return 0
        sleep 5; i=$((i+1))
    done
    return 1
}

echo "=============================================================="
echo "[autoq] 启动 $(date -Iseconds)  root=$ROOT  HEAD=$(git -C $ROOT rev-parse --short HEAD 2>/dev/null || echo NO_GIT)"
echo "=============================================================="
st boot START "队列启动；存量 $(count_num $TAG2)/$EXPECT2"

# ============================ 阶段 A：等批次 2 ============================
st A_wait RUN "等 $TAG2 矩阵进程结束（当前 $(count_num $TAG2)/$EXPECT2）"
# ⚠️ 传的是**允许的无进展分钟数**，不是墙钟上限（见 wait_proc_gone 注释 §52）。
# 批次 2 单次约 24 min ⇒ 60 min 无进展即判挂死。
if ! wait_proc_gone 60; then
    st A_wait STALL "批次2 进程在但产物无进展 ⇒ 不推进批次 3，保留现场"
    echo "S14AUTOQ_ABORTED_A2_STALL"; exit 0
fi
C2=$(label_census $TAG2)
N2=$(count_num $TAG2)
st A_done DONE "N2=$N2/$EXPECT2  $C2"

# ---- 闸门 G1：齐 + 全 MEASURED（**只用本批自带 meta.json**）----
M2=$(printf '%s' "$C2" | sed -n 's/.*MEASURED=\([0-9]*\).*/\1/p')
O2=$(printf '%s' "$C2" | sed -n 's/.*OTHER=\([0-9]*\).*/\1/p')
echo "[autoq] G1: N2=$N2 EXPECT=$EXPECT2  $C2"
if [ "$N2" -lt "$EXPECT2" ]; then
    st G1 FAIL "批次2产物不足 $N2/$EXPECT2 ⇒ **不启动批次3**"
    echo "S14AUTOQ_DONE_ABORTED_B2_INCOMPLETE"; exit 0
fi
if [ "$M2" -lt "$EXPECT2" ]; then
    st G1 FAIL "批次2非全 MEASURED：MEASURED=$M2 OTHER=$O2 ⇒ **不启动批次3**（留人工判定）"
    echo "S14AUTOQ_DONE_ABORTED_B2_NOT_ALL_MEASURED"; exit 0
fi
st G1 PASS "批次2齐且全 MEASURED（$M2/$EXPECT2）⇒ 允许批次3"

if ! wait_unlock; then
    st A_unlock WARN "锁 60 次检查后仍被占（$(ls -d /root/batch_locks/*.lock.d 2>/dev/null | tr '\n' ' ')）⇒ 仍继续，由 launch_batch.sh 再判"
else
    st A_unlock DONE "锁已释放"
fi

# ============================ 阶段 B：批次 3 lenmix ============================
st B_launch RUN "启动 $TAG3（lenmix 4臂×5重复）"
SA=$(supports_activate && echo yes || echo no)
echo "[autoq] 起跑器 $LAUNCH_SH 支持 --activate: $SA"
echo "[autoq] 激活脚本: $ACT"
launch_batch "$TAG3" "$QLOG.b3" \
    python -u scripts/run_step3_matrix.py \
        --cells $LENMIX_CELLS --repeats 5 --tag "$TAG3" \
        --cfg-dir configs/mthreads/phase1_step14 \
        --out "$ROOT/results/raw" >> "$QLOG" 2>&1
LRC=$?
echo "[autoq] B launch rc=$LRC"

# ⚠️ 起跑器的退出码**不含**"批次跑完没跑好"——它只证明"起跑动作成功"。
#    真正的判据在下面（产物数 + 标签），故这里 rc!=0 才当作失败处理。
if [ "$LRC" -ne 0 ]; then
    st B_launch RETRY "launch rc=$LRC ⇒ **重试一次**（同 tag；起跑器自带锁，重试安全）"
    sleep 20
    launch_batch "$TAG3" "$QLOG.b3" \
        python -u scripts/run_step3_matrix.py \
            --cells $LENMIX_CELLS --repeats 5 --tag "$TAG3" \
            --cfg-dir configs/mthreads/phase1_step14 \
            --out "$ROOT/results/raw" >> "$QLOG" 2>&1
    LRC=$?
    echo "[autoq] B launch retry rc=$LRC"
fi
if [ "$LRC" -ne 0 ]; then
    st B_launch FAIL "批次3起跑两次均失败 rc=$LRC ⇒ **停止该分支**、保留现场"
    echo "S14AUTOQ_DONE_ABORTED_B3_LAUNCH"; exit 0
fi
st B_launch DONE "批次3已起跑（launch rc=0）"

st B_wait RUN "等批次3结束（当前 $(count_num $TAG3)/$EXPECT3）"
# ⚠️ §52：原为 `wait_proc_gone 480`（8 h 墙钟），而真实时长 7.87 h + 启动 22 min
# ⇒ **必然假超时**。改为"60 min 无进展才判挂死"，跑得慢不再等于失败。
if ! wait_proc_gone 60; then
    st B_wait STALL "批次3 进程在但产物 60 min 无进展 ⇒ 判定挂死，保留现场，不重跑"
    echo "S14AUTOQ_DONE_ABORTED_B3_STALL"; exit 0
fi
C3=$(label_census $TAG3)
N3=$(count_num $TAG3)
M3=$(printf '%s' "$C3" | sed -n 's/.*MEASURED=\([0-9]*\).*/\1/p')
st B_done DONE "N3=$N3/$EXPECT3  $C3"

echo "[autoq] G2: N3=$N3 EXPECT=$EXPECT3  $C3"
if [ "$N3" -ge "$EXPECT3" ] && [ "$M3" -ge "$EXPECT3" ]; then
    st G2 PASS "批次3齐且全 MEASURED（$M3/$EXPECT3）"
else
    st G2 FAIL "批次3不完整：N3=$N3 $C3 ⇒ 标记待人工判定（**不重跑**）"
fi

wait_unlock || true
echo "S14AUTOQ_DONE $(date -Iseconds)"
st final DONE "队列结束；产物 $ROOT/results/raw/S14B2-* 与 S14B3-*"
