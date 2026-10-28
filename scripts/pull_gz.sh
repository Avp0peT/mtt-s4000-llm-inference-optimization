#!/bin/bash
# 逐文件拉取远端归档目录（gzip 中转版）。
#
# 为什么不用 `remote_run.py pull` / 裸 `get meta.json`：
#   `meta.json` 有 2.6 MB（其中 controller_trace 783 KB + context 386 KB +
#   system_states 177 KB），在批次运行占满 SSH 链路时**传不完**——症状是本地
#   留下 0 字节文件、命令却"看起来在跑"。裸 tar 路径还会留 0 字节/截断的 .tar 残留。
#
# 做法：远端把文件 gzip 成**中转文件**（`meta.json.gz`，实测 2.6 MB → 107 KB，
#   压缩比 ~24×），传 .gz，本地 gunzip 回原字节，**逐文件校验"解压后字节数 == 远端原始字节数"**，
#   然后删掉远端的 .gz 中转文件（不留现场）。
#
# ⚠️ 这是**纯传输层**改造：解压回来的字节与原始文件逐字节相同，产物身份不变。
#   校验用 `stat -c %s <原文件>` 对比**解压后**大小（不是 .gz 的大小）。
#
# 用法：bash scripts/pull_gz.sh <remote_raw_dir> <subdir> [subdir ...]

set -u
PY="C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
GUNZIP="gzip -d -c"
RROOT="${1:?need remote root}"; shift
FILES="config.snapshot.yaml meta.json requests.jsonl telemetry.jsonl"
MAXTRY=6

# ⚠️ **本地目录必须是绝对 Windows 路径**：Git Bash 的 `/tmp/x` 会被 paramiko 当成
# 相对 `\tmp\x` ⇒ 实际写到 `C:\tmp\`（当前盘符根），而 `get` **照样打印 OK**
# ⇒ "报告成功但未落盘"。用过一次 `/tmp/rq_test.jsonl` 踩到，`local=0` 而远端说 OK。
LROOT="C:/Users/13944/WorkBuddy/st/ascend-llm-inference-optimization/results/raw"

total_ok=0; total_bad=0
for name in "$@"; do
  rd="$RROOT/$name"; ld="$LROOT/$name"
  mkdir -p "$ld"
  echo "=== $name ==="
  for f in $FILES; do
    ok=0
    for i in $(seq 1 $MAXTRY); do
      # 1) 远端原始字节数（成功判据的基准）
      rsz=$(timeout 120 "$PY" scripts/remote_run.py exec "stat -c %s '$rd/$f' 2>/dev/null" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | tail -1)
      if [ -z "$rsz" ]; then sleep 4; continue; fi
      # 2) 远端压成中转文件
      timeout 180 "$PY" scripts/remote_run.py exec "gzip -k -f -9 '$rd/$f' 2>/dev/null && stat -c %s '$rd/$f.gz'" >/dev/null 2>&1
      gsz=$(timeout 120 "$PY" scripts/remote_run.py exec "stat -c %s '$rd/$f.gz' 2>/dev/null" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | tail -1)
      if [ -z "$gsz" ]; then sleep 4; continue; fi
      # 3) 传 .gz
      rm -f "$ld/$f.gz"
      MSYS_NO_PATHCONV=1 timeout 300 "$PY" scripts/remote_run.py get "$rd/$f.gz" "$ld/$f.gz" >/dev/null 2>&1
      lgsz=$(stat -c %s "$ld/$f.gz" 2>/dev/null || echo 0)
      if [ "$lgsz" != "$gsz" ] || [ "$lgsz" -eq 0 ]; then
        echo "  retry $f.gz 传输 (remote=$gsz local=$lgsz) try $i"; sleep 4; continue
      fi
      # 4) 解压并校验**解压后**大小
      rm -f "$ld/$f"
      $GUNZIP "$ld/$f.gz" > "$ld/$f" 2>/dev/null
      lsz=$(stat -c %s "$ld/$f" 2>/dev/null || echo 0)
      rm -f "$ld/$f.gz"
      if [ "$lsz" = "$rsz" ] && [ "$lsz" -gt 0 ]; then
        ok=1; echo "  OK   $f ($lsz B via gz $lgsz B, try $i)"; break
      fi
      echo "  retry $f 解压 (remote_orig=$rsz local_after_gunzip=$lsz) try $i"; sleep 4
    done
    # 清远端中转文件（不论成败都清，避免留现场）
    timeout 120 "$PY" scripts/remote_run.py exec "rm -f '$rd/$f.gz'" >/dev/null 2>&1
    if [ $ok -eq 0 ]; then echo "  FAIL $f"; total_bad=$((total_bad+1)); else total_ok=$((total_ok+1)); fi
  done
done
echo "=== 汇总: files_ok=$total_ok files_fail=$total_bad ==="
n=0; bad=0
for d in "$LROOT"/S14B2-*/; do
  if [ -s "$d/meta.json" ] && [ -s "$d/requests.jsonl" ]; then n=$((n+1)); else bad=$((bad+1)); fi
done
echo "本地完整目录 = $n  不完整 = $bad"
