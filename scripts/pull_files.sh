#!/bin/bash
# 逐文件拉取远端归档目录（比 tar 路径稳健：每个文件独立重试、逐个校验非空）。
#
# 背景：remote_run.py pull <dir> 走 tar 传输，在 SSH 高竞争下会留下 0 字节/截断的
#   .tar 残留，且失败后本地目录只被建出来、内容为空（"有目录"≠"有数据"）。
#   本脚本改为逐文件 get，每个文件独立重试，成功判据 = 本地非空 且 字节数 == 远端字节数。
#
# 用法：bash scripts/pull_files.sh <remote_raw_dir> <remote_subdir> [remote_subdir ...]
#   例：bash scripts/pull_files.sh /root/ascend_probe_d1/results/raw S14B2-load_ramp-ramp_cost

set -u
PY="C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
# 输出根：绝对路径，消除对调用方 cwd 的依赖。
# 注意 MSYS_NO_PATHCONV=1 只加在 get 那行（关掉远端参数的本地转换），
# 本地目标必须是 Windows 原生形状 C:/... ，否则会被误写成 C:\c\Users\...（见 §63）。
OUTROOT="C:/Users/13944/WorkBuddy/st/ascend-llm-inference-optimization/results/raw"
REPO="C:/Users/13944/WorkBuddy/st/ascend-llm-inference-optimization"
RROOT="${1:?need remote root}"; shift
FILES="config.snapshot.yaml meta.json requests.jsonl telemetry.jsonl"
MAXTRY=6

total_ok=0; total_bad=0
for name in "$@"; do
  rd="$RROOT/$name"; ld="$OUTROOT/$name"
  mkdir -p "$ld"
  echo "=== $name ==="
  for f in $FILES; do
    ok=0
    for i in $(seq 1 $MAXTRY); do
      # 远端字节数（一次取，失败则重连）
      rsz=$(timeout 120 "$PY" scripts/remote_run.py exec "stat -c %s '$rd/$f' 2>/dev/null" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | tail -1)
      if [ -z "$rsz" ]; then sleep 4; continue; fi
      rm -f "$ld/$f"
      MSYS_NO_PATHCONV=1 timeout 300 "$PY" scripts/remote_run.py get "$rd/$f" "$ld/$f" >/dev/null 2>&1
      lsz=$(stat -c %s "$ld/$f" 2>/dev/null || echo 0)
      if [ "$lsz" = "$rsz" ] && [ "$lsz" -gt 0 ]; then ok=1; echo "  OK   $f ($lsz B, try $i)"; break; fi
      echo "  retry $f (remote=$rsz local=$lsz) try $i"
      sleep 4
    done
    if [ $ok -eq 0 ]; then echo "  FAIL $f"; total_bad=$((total_bad+1)); else total_ok=$((total_ok+1)); fi
  done
done
echo "=== 汇总: files_ok=$total_ok files_fail=$total_bad ==="
echo "成功判据：本地字节数 == 远端字节数 且 > 0；files_fail=0 才算整批可用"
