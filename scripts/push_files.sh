#!/bin/bash
# 逐文件推送源码到远端（`pull_files.sh` 的镜像）。
#
# 背景：`remote_run.py push` 把整个 scope 打成**一个** tar 经一条连接上传，
#   在本项目这条高 RTT/高抖动的链路上会中途 `Server connection dropped`
#   （实测 294912 B 截断，远端留下一个半个的 `_push.tar`，一个文件都没落地）。
#   本脚本改为逐文件 put，每个文件独立短连接 + 独立重试，
#   成功判据 = 远端 sha256 == 本机 sha256（不是"put 返回 0"）。
#
# 用法：bash scripts/push_files.sh <remote_root> <relpath> [relpath ...]
#   例：bash scripts/push_files.sh /root/ascend_probe_d1 scripts/diagnose_scheduling.py
#   若给目录，递归展开其中所有 *.py / *.sh。
#
# 可选：`--include 'py,sh,yaml'` 覆盖递归展开的扩展名（默认 `py,sh`）。
#   为什么需要它：**配置是实验条件的一部分**（`configs/**/*.yaml`），
#   只推代码会得到"代码是新的、配置是旧的"——而产物里看不出来。
#   默认值刻意不变 ⇒ 既有调用方的行为逐字节相同。
#
# 可选：`--dry-run` **只展开并打印清单，不建任何连接**。
#   用途：① 测试可以在**断网**下核对"扩展名是否真被应用"；
#   ② 起跑前自查"这次要推哪些文件"，避免再出现 `--include` 静默降级那类事故。
#   ⚠️ 它必须在**位置参数之前**（否则会被当路径、静默失效 ⇒ 你以为在干跑、其实在真推）。
#   ⚠️ "不联网"的证明方式：dry-run 在第一个 `remote_run.py` 调用**之前**就 `exit 0`，
#      所以**后面的行根本读不到**；测试**不得**用"读首行后杀进程"来声称不联网——
#      那只能证明"没继续输出"，证不了"没建过连接"。
#
# ⚠️ 与 pull_files.sh 相对的是位移对象不同：put 是**本地→远端**，
#   因此本地路径用仓库相对路径（cwd 由 REPO 固定），远端路径由 RROOT + 同一相对路径拼出。

set -u
PY="C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
REPO="C:/Users/13944/WorkBuddy/st/ascend-llm-inference-optimization"

# ---- 可选：--include（必须在读位置参数之前摘掉）-----------------------
INCLUDE="py,sh"
DRY_RUN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --include) INCLUDE="${2:?--include 需要一个逗号分隔的扩展名列表}"; shift 2 ;;
    --include=*) INCLUDE="${1#--include=}"; shift ;;
    # ★ `--dry-run`：**只展开清单、不建任何连接**（见下方说明）。
    --dry-run) DRY_RUN=1; shift ;;
    *) break ;;
  esac
done
# ⚠️ 扩展名条件用**数组**传，不经字符串拼接：字符串里嵌引号不会被二次解析
#    （变量展开后 `'` 是字面字符 ⇒ find 去找名为 `'*.yaml'` 的文件 ⇒ 恒空，
#    踩过：`--include py,sh,yaml` 报「待推送 1 个」却 ok=0/skip=0/fail=0）。
#    而字符串里**不**嵌引号时，`*.$e` 又会被调用方 shell 预先 glob 展开 ⇒ 同样错。
#    ⇒ 数组 + 引号，两条坑一起规避。
FIND_ARGS=()
IFS=',' read -r -a _EXT <<< "$INCLUDE"
for e in "${_EXT[@]}"; do
  [ -z "$e" ] && continue
  [ ${#FIND_ARGS[@]} -gt 0 ] && FIND_ARGS+=(-o)
  FIND_ARGS+=(-name "*.$e")
done

RROOT="${1:?need remote root}"; shift
MAXTRY=6

# ★ 守卫：`--include` 是**选项**，必须在位置参数之前。放错位置时上面的 while 循环
#   在第一个位置参数处 `break`，`--include` 就**留在 `"$@"` 里当路径**：
#   于是被当「跳过（不存在）」，而 INCLUDE **静默退回默认 `py,sh`**
#   ⇒ 产物看起来"成功"，但 `.yaml/.yml/.json` **一个都没进清单**。
#   踩过（2026-10-02）：首次推送因此漏掉全部配置类文件，直到核对 configs 目录摘要
#   才发现差了 6 个（`configs/mthreads/phase1_step15_v5/` 整目录）。
#   ⇒ 判据：位置参数里若出现 `--include` / `--include=…` 形态，**直接报错退出**。
#     静默降级成默认值正是本条要禁止的失效模式。
for _arg in "$@"; do
  case "$_arg" in
    --include|--include=*)
      echo "用法错误：--include 必须放在**位置参数之前**，实际收到：$*" >&2
      echo "  正确写法：bash scripts/push_files.sh --include py,sh,yaml,yml,json <remote_root> <path>..." >&2
      echo "  （否则它会退回默认 \`py,sh\`，配置类文件不会被推送，且不报错）" >&2
      exit 2
      ;;
    # `--dry-run` 同族：放错位置会被当**路径**，于是它**静默失效**
    # ⇒ 调用方以为在 dry-run，实际**真去推送了**。这比 `--include` 更危险。
    --dry-run|--dry-run=*)
      echo "用法错误：--dry-run 必须放在**位置参数之前**，实际收到：$*" >&2
      echo "  （放错位置会被当路径跳过，于是它**不生效**——你以为在干跑，其实在真推）" >&2
      exit 2
      ;;
  esac
done

if [ ! -x "$PY" ] && [ ! -f "$PY" ]; then
  echo "找不到解释器：$PY"; exit 2
fi

# 展开目标列表：目录 ⇒ 其下 *.py/*.sh；文件 ⇒ 原样
LIST=""
for item in "$@"; do
  p="$REPO/$item"
  if [ -d "$p" ]; then
    files=$(find "$item" -type f \( "${FIND_ARGS[@]}" \) -not -path '*__pycache__*' | sort)
  elif [ -f "$p" ]; then
    files="$item"
  else
    echo "跳过（不存在）：$item" >&2
    continue
  fi
  LIST="$LIST$files
"
done
LIST=$(printf '%s' "$LIST" | sed '/^$/d')
N=$(printf '%s\n' "$LIST" | wc -l)
echo "待推送 $N 个文件 -> $RROOT"

# ---- ★ `--dry-run`：**只列清单，不建任何连接** -------------------------
#  为什么必须是"脚本自己有一个早退出口"，而不是靠调用方"读首行后杀进程"：
#  后者只能证明**进程在那之后没有继续输出**，**不能**证明它没有建过连接
#  （连接可能已经在首行打印前发生，也可能在缓冲里）。判据的要害是
#  **"有没有可能联网"**，而不是"看起来停得早不早"。
#  ⇒ 这里在**清单展开完成之后、第一个 `remote_run.py` 调用之前**直接 return；
#    之后的每一行都读不到执行。测试据此可以在**断网/错误凭据**下跑完并核对清单。
if [ "$DRY_RUN" -eq 1 ]; then
  echo "----"
  echo "dry-run 清单："
  printf '%s\n' "$LIST"
  echo "----"
  echo "push_files(dry-run): 共 $N；未建立任何连接"
  exit 0
fi

total_ok=0; total_bad=0; total_skip=0; bad_list=""
i=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  i=$((i+1))
  lp="$REPO/$f"
  rp="$RROOT/$f"
  lsha=$(sha256sum "$lp" | awk '{print $1}')
  lsz=$(stat -c %s "$lp")

  # 远端现有 sha256（省掉无变更文件的重传）
  rsha=$(MSYS_NO_PATHCONV=1 timeout 45 "$PY" scripts/remote_run.py exec \
           "sha256sum '$rp' 2>/dev/null | awk '{print \$1}'" 2>/dev/null \
         | grep -oE '^[0-9a-f]{64}' | head -1)
  if [ "$lsha" = "$rsha" ]; then
    echo "[$i/$N] SKIP  $f  (sha 已一致)"
    total_skip=$((total_skip+1))
    continue
  fi

  ok=0
  for try in $(seq 1 $MAXTRY); do
    MSYS_NO_PATHCONV=1 timeout 120 "$PY" scripts/remote_run.py put "$lp" "$rp" >/dev/null 2>&1
    # 成功判据：远端能算出 sha256 且与本机一致（不看 put 的退出码）
    rsha=$(MSYS_NO_PATHCONV=1 timeout 45 "$PY" scripts/remote_run.py exec \
             "sha256sum '$rp' 2>/dev/null | awk '{print \$1}'" 2>/dev/null \
           | grep -oE '^[0-9a-f]{64}' | head -1)
    if [ "$lsha" = "$rsha" ]; then ok=1; echo "[$i/$N] OK    $f  ($lsz B, try $try)"; break; fi
    echo "[$i/$N] retry $try  $f" >&2
  done
  if [ $ok -eq 1 ]; then total_ok=$((total_ok+1)); else
    total_bad=$((total_bad+1)); bad_list="$bad_list $f"
    echo "[$i/$N] FAIL  $f  (本机 $lsha 远端 ${rsha:-<none>})" >&2
  fi
done <<< "$LIST"

echo "----"
echo "push_files: ok=$total_ok skip=$total_skip fail=$total_bad (共 $N)"
if [ -n "$bad_list" ]; then
  echo "失败清单：$bad_list"
  exit 1
fi
exit 0
