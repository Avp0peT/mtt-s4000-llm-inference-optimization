#!/usr/bin/env bash
# ============================================================================
# 批次起跑的唯一入口（`run_s14_d1.sh` / `remote_wait_then_d1.sh` 共用）
# ----------------------------------------------------------------------------
# 抽成独立文件的原因：起跑有**三处会静默出错**的地方，此前在 4 个文件里
# 各写了一遍（`run_s14_d1.sh` / `run_s14_d1_min.sh` / `remote_wait_then_d1.sh` /
# 本机 `.workbuddy/tmp/scan_run.sh`）。同一件事的 N 份实现必然分叉 ——
# 本项目已踩过三次（臂名解析三份、并发判据两份）。
#
# ① **副作用型命令绝不重试**
#    踩过（2026-09-27）：本机 `remote()` 的失败重试把整条 `nohup` 起跑命令
#    发了**两遍**，两遍都真执行 ⇒ 第二个进程 `rm -f <log>` 抹掉第一个进程
#    正在写的 inode（`/proc/<pid>/fd/1 -> ... (deleted)`）⇒ **日志永久丢失**，
#    且差点让两个矩阵同时写 `results/raw/`（第二个在第 1 格 `RunDirConflict`
#    退出，纯属运气）。
#    判据：命令"是否已发出"与"输出里有没有失败标记"是**两件事** ——
#    `nohup` 立刻返回，链路抖动只影响**回读**，不影响**已发出的副作用**。
#    ⇒ 起跑只发一次；失败靠**产物**判定。
#
# ② **起跑必须持锁，且锁的生命周期 == 批次的生命周期**
#    ★ 这一条我第一版**两个方向都做错过**，所以锁本体挪进了
#      `scripts/batch_lock.py`（带测试，能在本机完整验证）：
#        · 锁只覆盖"起跑动作" ⇒ 同 tag 重起跑畅通 ⇒ 事故原样复现；
#        · 锁覆盖批次期但没人放 ⇒ **永久堵住后续所有批次**。
#      本脚本只负责：取锁 → 起跑 → 起监视进程（由监视进程在批次结束后放锁）。
#
# ③ **"报成功"必须由证据支撑**
#    踩过：本机没有 `setsid` 时子进程从未创建、日志里只有一行
#    `setsid: command not found`，而脚本打印 `BATCH_LAUNCH_OK` 且退出码 0。
#    ⇒ 起跑后做存活确认 + 日志报错扫描，二者之一失败即**非零退出**。
#
# ④ **激活环境必须由起跑器做，不能靠调用方纪律**
#    踩过（2026-09-28）：调用方在起跑器**外面** `source activate_musa.sh` 再调本脚本，
#    批次立刻 `ModuleNotFoundError: No module named 'transformers'`（0/20），
#    而本脚本照打 `BATCH_LAUNCH_OK` —— 因为它只查"进程是否存活"（激活失败时
#    python 确实起来过，只是立刻退出）。
#    根因：起跑走 `setsid nohup "$@"`，**直接 exec**，中间没有 shell ⇒
#    setsid 在**新会话 + 非交互 shell** 里重新 exec，环境变量不继承。
#    ⇒ 环境激活只能发生在 setsid **之后**的那个 shell 里 ⇒ 本脚本加 `--activate`。
#    ⚠️ 内层必须 `bash -c` 而非 `sh -c`：`source` 非 POSIX 内建，dash 里会**静默失败**。
#
# 依赖说明（**实测，不是推断**）
# -----------------------------
#   远端（真正跑批次处，Linux 5.15 / bash 5.1）：`setsid` `nohup` `flock` 都有。
#   本机（Git for Windows / PortableGit）：**`setsid` 与 `flock` 都没有**。
#   ⇒ 起跑器按可用性探测（有 setsid 最好）；锁用 `mkdir`（两边都有、原子）。
#     本机因此**能验证锁语义**，但**不能**验证"脱离会话"——那是远端的契约。
#
# 用法::
#     bash scripts/launch_batch.sh <tag> <logfile> <命令行...>
#     bash scripts/launch_batch.sh --activate <脚本> <tag> <logfile> <命令行...>
#       例：--activate /root/musa-envs/qwen310/activate_musa.sh
#
# 退出码（调用方**只认这一个**，不得回头 grep 日志）：
#     0 = 起跑成功
#     4 = 同 tag 已在跑       5 = 其它批次已在跑
#     6 = 参数/目录错         7 = 起跑失败（进程没起来 / 启动器报错）
#     9 = **未能证明**起跑成功（pid 不在且日志为空）
#
# 终止标记：BATCH_LAUNCH_OK / BATCH_LAUNCH_ABORTED_<原因>
# ----------------------------------------------------------------------------
set -uo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="${BATCH_PY:-python3}"

usage() {
    echo "用法: bash scripts/launch_batch.sh [--activate <脚本>] <tag> <logfile> <命令行...>" >&2
    echo "  --activate <脚本>  起跑前在同一个 shell 里 source 它（用来激活远端 venv）" >&2
}

# ---- 可选开关：必须在解析 tag/logfile 之前摘掉，否则会被当成位置参数 ----
# ★ 为什么需要 `--activate`：起跑用 `setsid nohup "$@"` **直接 exec**（见第 ② 段），
#   中间没有 shell ⇒ 调用方在**外面** `source activate_musa.sh` 是无效的：
#   setsid 会在一个**全新的会话 + 非交互 shell** 里重新 exec 那条命令，
#   环境变量不会被继承。踩过（2026-09-28）：这样起跑的批次立刻
#   `ModuleNotFoundError: No module named 'transformers'`（0/20），
#   而 `launch_batch.sh` 照打 `BATCH_LAUNCH_OK` —— 因为它只查进程存活。
#   ⇒ 激活必须由**本脚本**在 setsid 后的那个 shell 里做，不能靠调用方纪律。
#
#   ⚠️ 必须用 `bash -c 'source ...'` 而不是 `sh -c`：`source` 不是 POSIX 内建，
#      dash（`sh`）里没有它 ⇒ **静默失败**（激活没发生，但不会报错）。
ACTIVATE=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --activate)
            if [ -z "${2:-}" ]; then
                echo "[launch] ✗ --activate 需要一个脚本路径" >&2
                usage
                echo "BATCH_LAUNCH_ABORTED_BADARGS"
                exit 6
            fi
            ACTIVATE="$2"
            shift 2
            ;;
        --activate=*)
            ACTIVATE="${1#--activate=}"
            shift
            ;;
        *) break ;;
    esac
done

TAG="${1:-}"
LOGFILE="${2:-}"
shift 2 2>/dev/null || true

if [ -z "$TAG" ] || [ -z "$LOGFILE" ] || [ "$#" -eq 0 ]; then
    usage
    echo "BATCH_LAUNCH_ABORTED_BADARGS"
    exit 6
fi

# 把用户命令包进激活层。变量留给第 ② 段用。
# 用 `exec` 让激活 shell **被替换**成目标命令 ⇒ 少一层进程，
# 且 `kill -0 $LAUNCH_PID` 探到的就是真正跑批次的进程。
if [ -n "$ACTIVATE" ]; then
    LAUNCH_CMD=(bash -c 'source "$1" || { echo "[activate] ✗ source 失败：$1" >&2; exit 7; }; shift; exec "$@"' _ "$ACTIVATE" "$@")
else
    LAUNCH_CMD=("$@")
fi

LOCKDIR="${BATCH_LOCKDIR:-/root/batch_locks}"
LOCK_PY="$SELF_DIR/batch_lock.py"
[ -f "$LOCK_PY" ] || { echo "[launch] ✗ 缺 $LOCK_PY"; echo "BATCH_LAUNCH_ABORTED_NO_LOCK_MODULE"; exit 6; }

LOGDIR="$(dirname "$LOGFILE")"
mkdir -p "$LOGDIR" || { echo "[launch] ✗ 日志目录不可建：$LOGDIR"; echo "BATCH_LAUNCH_ABORTED_NOLOGDIR"; exit 6; }

# ⚠️ 监视进程**必须有自己的日志**，不能与批次共用 $LOGFILE。
#    踩过：共用时监视进程的 traceback（含 "No such file or directory"）落进
#    批次日志，被下面的启动报错扫描 grep 到 ⇒ **批次明明起来了却判 LAUNCH_FAILED**。
#    同一文件里既有"批次的输出"又有"监管者的输出"，判据就不再可靠。
MON_LOGFILE="${LOGFILE}.monitor"

echo "[launch] tag=$TAG"
echo "[launch] 日志=$LOGFILE"
if [ -n "$ACTIVATE" ]; then
    echo "[launch] 激活脚本=$ACTIVATE（将在 setsid 后的 shell 里 source）"
else
    echo "[launch] 激活脚本=（无）⚠️ 远端 venv **不会**被激活；跑 python 大概率 ModuleNotFoundError"
fi
echo "[launch] 命令行=$*"

# ---- ① 取锁（唯一实现 = scripts/batch_lock.py）-----------------------------
# 退出码 4/5 由模块给出，本脚本**原样透传** —— 不要在 shell 里重新解释。
set +e
"$PY_BIN" "$LOCK_PY" --root "$LOCKDIR" acquire --tag "$TAG" --cmd "$*"
LOCK_RC=$?
set -e
if [ "$LOCK_RC" -ne 0 ]; then
    echo "[launch] ✗ 取锁失败（rc=$LOCK_RC）⇒ 拒绝起跑"
    # 模块已打印 4/5 的具体原因与清锁指引，这里不再重复。
    case "$LOCK_RC" in
        4) echo "BATCH_LAUNCH_ABORTED_TAG_BUSY" ;;
        5) echo "BATCH_LAUNCH_ABORTED_GLOBAL_BUSY" ;;
        *) echo "BATCH_LAUNCH_ABORTED_LOCK_ERROR" ;;
    esac
    exit "$LOCK_RC"
fi
echo "[launch] 锁已取（$LOCKDIR）"

# 取锁后的任何失败都必须放锁，否则会堵住后续批次。
# ⚠️ 但**起跑成功之后**要取消这个 trap —— 那时锁的归属已转交监视进程。
_release_on_fail() {
    if [ "${_LAUNCH_OK:-0}" != "1" ]; then
        echo "[launch] 起跑未成功 ⇒ 归还锁"
        "$PY_BIN" "$LOCK_PY" --root "$LOCKDIR" release --tag "$TAG" || true
    fi
}
trap '_release_on_fail' EXIT

# ---- ② 起跑：**只发一次**。优先 setsid，无则退回 nohup，都无则报错 --------
if command -v setsid >/dev/null 2>&1; then
    LAUNCH_MODE="setsid+nohup"
    setsid nohup "${LAUNCH_CMD[@]}" >"$LOGFILE" 2>&1 </dev/null &
    LAUNCH_PID=$!
elif command -v nohup >/dev/null 2>&1; then
    # ⚠️ 本机（Git for Windows）走这条：**没有 setsid**。
    #    这样的子进程仍属于本 shell 的会话 ⇒ 本 shell 被整组 kill 时会一起死。
    #    这不是缺陷而是事实，故**显式打印**，让读日志的人知道边界在哪。
    LAUNCH_MODE="nohup-only（无 setsid ⇒ 未脱离会话，父 shell 被整组杀会连带）"
    nohup "${LAUNCH_CMD[@]}" >"$LOGFILE" 2>&1 </dev/null &
    LAUNCH_PID=$!
else
    echo "[launch] ✗ 既无 setsid 也无 nohup ⇒ 无法把批次放到后台"
    echo "[launch]   静默降级成前台会让本脚本永远不返回（长批次会被误杀）"
    echo "BATCH_LAUNCH_ABORTED_NO_LAUNCHER"
    exit 6
fi
echo "[launch] 起跑方式：$LAUNCH_MODE"
echo "[launch] STARTED_PID=$LAUNCH_PID"

# ---- ③ 监视进程：批次结束后放锁（锁的生命周期 == 批次存活期）---------------
# ★ 为什么必须独立进程：本脚本 `setsid` 之后立刻退出，**不能**留在这里等
#   （那会让编排方一直阻塞）。而若没人等，锁要么被 trap 立刻放掉（=没锁），
#   要么永远留着（=堵死）。监视进程是唯一能同时满足两者的形态。
# ⚠️ 监视逻辑（判定"批次是否还在"）在 `batch_lock.py::decide()` 里，
#   本机有测试覆盖；这里只做"起进程 + 调 decide"。
_monitor_loop() {
    local bpid="$1" la="$2" lt="$3" tag="$4" lockdir="$5" lockpy="$6" pypath="$7"
    local strikes=0 t0
    t0="$(date +%s)"
    while :; do
        if [ ! -d "$la" ]; then
            echo "[monitor] 锁已被外部清除 ⇒ 退出（不再 rm，避免删掉别人的锁）"
            return 0
        fi
        local alive="unknown"
        if kill -0 "$bpid" 2>/dev/null; then alive=true
        elif [ -d "/proc/$bpid" ]; then alive=true
        else alive=false
        fi
        local age=$(( $(date +%s) - t0 ))
        local verdict
        # ⚠️ **不要**用 `python - <<'PY'` + `__file__` 定位模块：
        #    从 stdin 跑时 `__file__` 是 `<stdin>`，`.parent` 会变成当前工作目录
        #    （= 仓库根，而模块在 scripts/）⇒ 每次迭代都 FileNotFoundError。
        #    模块路径由调用方**显式传入**（$lockpy）。
        verdict="$("$pypath" - "$alive" "$strikes" "$age" "$lockpy" <<'PY'
import sys, importlib.util
alive_s, strikes_s, age_s, lockpy = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
alive = {"true": True, "false": False}.get(alive_s, None)
spec = importlib.util.spec_from_file_location("bl", lockpy)
bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
act, why = bl.decide(alive=alive, lock_still_held=True,
                     unknown_strikes=int(strikes_s), batch_age_s=float(age_s),
                     min_hold_s=10.0)
print(f"{act}|{why}")
PY
)"
        case "${verdict%%|*}" in
            release)
                "$pypath" "$lockpy" --root "$lockdir" release --tag "$tag" || true
                echo "[monitor] 锁已释放（$verdict）"
                return 0
                ;;
            exit) return 0 ;;
            *)
                case "$alive" in unknown) strikes=$((strikes + 1)) ;; *) strikes=0 ;; esac
                sleep 5
                ;;
        esac
    done
}
if command -v setsid >/dev/null 2>&1; then
    setsid bash -c "$(declare -f _monitor_loop); _monitor_loop \"\$@\"" _ \
        "$LAUNCH_PID" "$LOCKDIR/all.lock.d" "$LOCKDIR/tag.$TAG.lock.d" "$TAG" "$LOCKDIR" "$LOCK_PY" "$PY_BIN" \
        >>"$MON_LOGFILE" 2>&1 </dev/null &
else
    nohup bash -c "$(declare -f _monitor_loop); _monitor_loop \"\$@\"" _ \
        "$LAUNCH_PID" "$LOCKDIR/all.lock.d" "$LOCKDIR/tag.$TAG.lock.d" "$TAG" "$LOCKDIR" "$LOCK_PY" "$PY_BIN" \
        >>"$MON_LOGFILE" 2>&1 </dev/null &
fi
echo "[monitor] 监视进程已起（pid=$!，日志=$MON_LOGFILE）"

# ---- ④ 存活确认：**必须能判失败**，不是只打印提醒 ---------------------------
sleep 3
ALIVE=0
if kill -0 "$LAUNCH_PID" 2>/dev/null; then ALIVE=1; fi
echo "[launch] 存活确认：pid=$LAUNCH_PID alive=$ALIVE"

LAUNCH_ERR=""
if [ -f "$LOGFILE" ]; then
    LAUNCH_ERR="$(grep -m1 -E "command not found|No such file or directory|Permission denied" "$LOGFILE" 2>/dev/null || true)"
fi
if [ -n "$LAUNCH_ERR" ]; then
    echo "[launch] ✗ 日志里出现启动器/命令报错 ⇒ 批次**没有真的起来**：$LAUNCH_ERR"
    head -20 "$LOGFILE"
    echo "BATCH_LAUNCH_ABORTED_LAUNCH_FAILED"
    exit 7
fi
if [ "$ALIVE" -eq 0 ] && [ ! -s "$LOGFILE" ]; then
    echo "[launch] ✗ pid 已不在**且**日志为空 ⇒ 无法证明批次起来了"
    echo "BATCH_LAUNCH_ABORTED_NO_EVIDENCE"
    exit 9
fi

# ---- ⑤ 起跑成立：锁的归属转交监视进程，取消失败回滚 trap -------------------
_LAUNCH_OK=1
trap - EXIT

if [ -s "$LOGFILE" ]; then
    echo "[launch] 日志已产生内容（$(wc -l < "$LOGFILE") 行）"
    head -8 "$LOGFILE"
else
    echo "[launch] ⚠️ 日志暂空（可能刚起、缓冲未刷）"
fi
echo "[launch] 判据：**以产物为准** —— 起跑成功 ≠ 批次成功，"
echo "[launch]      失败判定读 results/raw/<tag>-* 是否在增长，不看本命令返回码。"
echo "BATCH_LAUNCH_OK"
exit 0
