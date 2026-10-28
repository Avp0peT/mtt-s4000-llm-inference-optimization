#!/bin/bash
# S14B2 回拉补全（鲁棒版，断点续跑，**单写者**）。
#
# 为什么需要它（都是实测教训，不是推测）：
#   1) `get` 的失败面在**落地尺寸**上：< ~100 KB 几乎必成，≥ ~100 KB 反复截断，
#      截断台阶恒为 0/32768/65536（32 KiB 整数倍）⇒ 中止发生在**本地落盘段**。
#   2) 因此 `pull_gz.sh` 的 MAXTRY=6 对 ~2.6 MB 的 `meta.json` 不够。
#   3) ⚠️ **同一批目录只能有一个写者**。并发两个回拉进程会互相删/写同一文件，
#      症状是"同一文件大小来回变"。本脚本用**锁文件**把这件事变成不可能。
#
# 判据（关键）：**"文件存在"会骗人**。`meta.json` 大小 0 / 32768 都是截断，
#   正常约 2.6 MB。故完整性一律按**尺寸门槛**判定，不按存在性。
#
# 用法：
#   bash scripts/pull_s14b2_robust.sh                 # 只补不完整的目录（常规用法）
#   bash scripts/pull_s14b2_robust.sh --all           # 强制重拉全部 20 个
#   bash scripts/pull_s14b2_robust.sh --register      # 补全后若 20/20 齐 ⇒ 自动登记
#   MAXTRY=20 bash scripts/pull_s14b2_robust.sh       # 提高单文件重试上限
#
# 依赖：scripts/remote_run.py（远端 exec/get）。本地解释器必须是装了 paramiko 的那个。

set -u
cd "$(dirname "$0")/.." || exit 2

PY="C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
LROOT="C:/Users/13944/WorkBuddy/st/ascend-llm-inference-optimization/results/raw"
RROOT="/root/ascend_probe_d1/results/raw"
TAG="S14B2"
DO_ALL=0; DO_REGISTER=0
for a in "$@"; do
  case "$a" in
    --all) DO_ALL=1 ;;
    --register) DO_REGISTER=1 ;;
    *) echo "未知参数: $a" >&2; exit 2 ;;
  esac
done
MAXTRY="${MAXTRY:-20}"

# ---------- 尺寸门槛（按文件类型，取 "正常值的下界" 而非精确值） ----------
# 说明：这些门槛是**保守下界**，用来把 0 / 32768 这类截断判为不完整。
# 精确字节数由远端 `stat -c %s` 提供，落盘后**逐字节比对**，所以门槛放宽是安全的。
min_bytes() {
  case "$1" in
    meta.json)          echo 100000 ;;   # 正常 ~2.6 MB
    telemetry.jsonl)    echo 100000 ;;   # 正常 ~1.5–2.1 MB
    requests.jsonl)     echo 20000  ;;   # 正常 ~42 KB（源文件；gz 中转 7–9 KB）
    config.snapshot.yaml) echo 500  ;;   # 正常 ~1.5–1.9 KB
    *)                  echo 1      ;;
  esac
}

LOCKFILE=".workbuddy/memory/.pull_s14b2.lock"
STOPFILE=".workbuddy/memory/.pull_s14b2.stop"
if [ -e "$LOCKFILE" ]; then
  pid=$(cat "$LOCKFILE" 2>/dev/null || echo "?")
  # 只用 kill -0 判活；僵尸/已死则接管（避免永久卡锁）
  if [ "$pid" != "?" ] && kill -0 "$pid" 2>/dev/null; then
    echo "❌ 已有回拉写者在运行 (pid=$pid)。**拒绝启动第二个写者**（并发写会互相删文件）。"
    echo "   若要强制接管，先确认旧进程已死，再: rm -f $LOCKFILE"
    exit 3
  fi
  echo "⚠️  发现陈旧锁 (pid=$pid 已不存在)，接管。"
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT INT TERM
rm -f "$STOPFILE"   # 开跑时清掉上一轮的停止标记

# 软停止文件：某些平台上沙箱内的子进程**无法从外部被 kill**
# （实测：`os.kill`/`taskkill`/`TerminateProcess` 全部 PermissionError，
#  而进程其实还活着并持续 fork 新的 get 子进程）。
# ⇒ 提供一个**协作式停止点**：外部只需 `touch .workbuddy/memory/.pull_s14b2.stop`,
#   脚本在每个目录开始前检查一次并自行退出（不留下半截文件）。
check_stop() {
  if [ -e "$STOPFILE" ]; then
    echo "⏹  检测到停止标记 $STOPFILE ⇒ 主动退出（未完成的文件保持原样，不留半截）。"
    rm -f "$STOPFILE"
    exit 0
  fi
}

# ---------- 远端单文件拉取（gz 中转 + 原始字节数校验） ----------
#
# ⚠️⚠️ **绝不先删目标文件**。旧版在这里写了 `rm -f "$ld/$f.gz" "$ld/$f"`，
#   于是"重试失败"会**把已经完好的文件删掉**——实测把本地完整度从 11/20
#   恶化到 8/20（`2026-09-28.md` §43）。这不是"没拉上来"，是"自己删的"，
#   方向与目标相反、且**完全静默**（只多一行 retry 日志）。
#
#   现在的形态：**写到 .partial，校验通过后才 os.replace 式覆盖**。
#   bash 里用 `mv`（同一目录内是原子改名）。失败路径**保持原文件不动**。
fetch_one() {
  local rd="$1" f="$2" ld="$3"
  local mb; mb=$(min_bytes "$f")
  local rsz gsz lgsz lsz
  mkdir -p "$ld"
  local partial="$ld/.$f.partial"
  rm -f "$partial"
  local i
  for i in $(seq 1 "$MAXTRY"); do
    rsz=$(timeout 120 "$PY" scripts/remote_run.py exec "stat -c %s '$rd/$f' 2>/dev/null" 2>/dev/null \
          | tr -d '\r' | grep -E '^[0-9]+$' | tail -1)
    if [ -z "$rsz" ]; then echo "    [$f] try$i: 远端 stat 无输出（控制面抖动或文件不存在）"; sleep 3; continue; fi
    if [ "$rsz" -lt "$mb" ]; then echo "    [$f] 远端原始仅 $rsz B < 门槛 $mb B ⇒ 远端也没写全，等下一轮"; return 1; fi

    timeout 180 "$PY" scripts/remote_run.py exec "gzip -k -f -9 '$rd/$f' 2>/dev/null" >/dev/null 2>&1
    gsz=$(timeout 120 "$PY" scripts/remote_run.py exec "stat -c %s '$rd/$f.gz' 2>/dev/null" 2>/dev/null \
          | tr -d '\r' | grep -E '^[0-9]+$' | tail -1)
    if [ -z "$gsz" ]; then echo "    [$f] try$i: 远端 gz 大小读不到"; sleep 3; continue; fi

    # 中转文件落到 **.partial 名下**：目标文件在整个过程里**始终不被触碰**
    rm -f "$partial.gz"
    MSYS_NO_PATHCONV=1 timeout 300 "$PY" scripts/remote_run.py get "$rd/$f.gz" "$partial.gz" >/dev/null 2>&1
    lgsz=$(stat -c %s "$partial.gz" 2>/dev/null || echo 0)
    if [ "$lgsz" != "$gsz" ] || [ "$lgsz" -eq 0 ]; then
      echo "    [$f] try$i: gz 传不完整 (remote=$gsz local=$lgsz)"; sleep 3; continue
    fi
    gzip -d -c "$partial.gz" > "$partial" 2>/dev/null
    lsz=$(stat -c %s "$partial" 2>/dev/null || echo 0)
    rm -f "$partial.gz"
    if [ "$lsz" = "$rsz" ] && [ "$lsz" -ge "$mb" ]; then
      mv -f "$partial" "$ld/$f"      # ← 原子替换：只有此刻目标文件才被改写
      echo "    [$f] ✓ $lsz B (gz $lgsz B, try$i)"
      return 0
    fi
    echo "    [$f] try$i: 解压后 $lsz B ≠ 远端原始 $rsz B（目标文件未动）"; rm -f "$partial"; sleep 3
  done
  echo "    [$f] ✗ 重试 $MAXTRY 次仍失败（**原文件保持不动**）"
  rm -f "$partial" "$partial.gz"
  return 1
}

# ---------- 组装目标目录清单 ----------
DIRS=()
for d in "$LROOT/$TAG"-*/; do
  [ -d "$d" ] || continue
  b=$(basename "$d")
  if [ "$DO_ALL" -eq 1 ]; then DIRS+=("$b"); continue; fi
  need=0
  for f in config.snapshot.yaml meta.json requests.jsonl telemetry.jsonl; do
    mb=$(min_bytes "$f"); sz=$(stat -c %s "$d$f" 2>/dev/null || echo 0)
    [ "$sz" -lt "$mb" ] && need=1
  done
  [ "$need" -eq 1 ] && DIRS+=("$b")
done

echo "=== 目标目录 ${#DIRS[@]} 个（DO_ALL=$DO_ALL, MAXTRY=$MAXTRY, lock=$$） ==="
for b in "${DIRS[@]:-}"; do [ -n "$b" ] && echo "  - $b"; done

# ---------- 顺序拉取（**串行**，不并发，避免链路争用） ----------
FILES="config.snapshot.yaml meta.json requests.jsonl telemetry.jsonl"
for b in "${DIRS[@]:-}"; do
  [ -n "$b" ] || continue
  check_stop          # 协作式停止点：每个目录开始前检查一次
  rd="$RROOT/$b"; ld="$LROOT/$b"
  echo "=== $b ==="
  for f in $FILES; do
    check_stop
    mb=$(min_bytes "$f"); sz=$(stat -c %s "$ld/$f" 2>/dev/null || echo 0)
    if [ "$sz" -ge "$mb" ] && [ "$DO_ALL" -eq 0 ]; then echo "    [$f] 已有 $sz B，跳过"; continue; fi
    fetch_one "$rd" "$f" "$ld"
  done
  # 无论成败都清远端中转文件（不留现场）
  for f in $FILES; do timeout 120 "$PY" scripts/remote_run.py exec "rm -f '$rd/$f.gz'" >/dev/null 2>&1; done
done

# ---------- 结算 ----------
echo "=== 结算（判据 = 尺寸门槛，不是文件存在） ==="
ok=0; bad=0; BADDIRS=()
for d in "$LROOT/$TAG"-*/; do
  [ -d "$d" ] || continue
  done_ok=1
  for f in config.snapshot.yaml meta.json requests.jsonl telemetry.jsonl; do
    mb=$(min_bytes "$f"); sz=$(stat -c %s "$d$f" 2>/dev/null || echo 0)
    [ "$sz" -lt "$mb" ] && done_ok=0
  done
  if [ "$done_ok" -eq 1 ]; then ok=$((ok+1)); else bad=$((bad+1)); BADDIRS+=("$(basename "$d")"); fi
done
echo "完整 $ok / 共 $((ok+bad))"
for b in "${BADDIRS[@]:-}"; do [ -n "$b" ] && echo "  仍不完整: $b"; done

# ---------- 可选：自动登记（**只在 20/20 且编号无缺口时**） ----------
if [ "$DO_REGISTER" -eq 1 ]; then
  if [ "$ok" -eq 20 ] && [ "$bad" -eq 0 ]; then
    echo "=== 20/20 齐 ⇒ 登记（一次性生成全部行，编号是预留给 20 个的） ==="
    "$PY" scripts/emit_experiment_rows.py --tag "$TAG" \
      | tee "/tmp/emit_$TAG.txt" >/dev/null
    echo "（行已生成到 /tmp/emit_$TAG.txt，请人工复核后 --append）"
  else
    echo "=== 未达 20/20（$ok/20）⇒ **不登记**（编号按 20 个预留，缺目录会出编号缺口） ==="
  fi
fi
