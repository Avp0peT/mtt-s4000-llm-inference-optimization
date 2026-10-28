#!/bin/bash
# 定点补齐 S14B2 本地缺失文件（只拉清单里点名的 (dir,file)，**不枚举远端**）。
#
# 为什么单独一个脚本、而不是复用 pull_gz.sh：
#   ① pull_gz.sh 按"目录"工作，会把 4 个文件全部重拉一遍——而这里只有 2 个文件真缺；
#   ② 更要紧的是**不要 ls 远端目录**：远端 raw/ 里躺着 4 个 `*.tar` 残留
#      （`cost-r3.tar` / `slo_weave-r1.tar.tar` / `static-r1.tar` / `threshold-r2.tar`），
#      按目录枚举会把它们当成"run"去取（实测 `pull_incremental.py` 就这么做过，
#      于是产出一串 rc=1/2 的假失败）。**点名制**从根上避开这个坑。
#
# meta.json 有 2.6 MB，在批次占满 SSH 链路时裸 get **传不完**（症状：本地 0 字节/
# 截断，命令却"看起来在跑"）⇒ 一律 gzip 中转（实测 2.6 MB → 107 KB），
# 逐文件校验"解压后字节数 == 远端原始字节数"。
#
# 本地目标路径必须是**绝对 Windows 路径**（Git Bash 的 /tmp/x 会被 paramiko 当相对路径）。
set -u
PY="C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
LROOT="C:/Users/13944/WorkBuddy/st/ascend-llm-inference-optimization/results/raw"
RROOT="/root/ascend_probe_d1/results/raw"
MAXTRY=14

# 形如 "<dir>::<file>"
JOBS="
S14B2-load_ramp-ramp_cost-r3::meta.json
S14B2-load_ramp-ramp_cost-r3::telemetry.jsonl
S14B2-load_ramp-ramp_threshold-r2::config.snapshot.yaml
S14B2-load_ramp-ramp_threshold-r2::meta.json
S14B2-load_ramp-ramp_threshold-r2::requests.jsonl
S14B2-load_ramp-ramp_threshold-r2::telemetry.jsonl
"

okn=0; badn=0
for job in $JOBS; do
  name="${job%%::*}"; f="${job##*::}"
  rd="$RROOT/$name"; ld="$LROOT/$name"
  mkdir -p "$ld"
  got=0
  for i in $(seq 1 $MAXTRY); do
    rsz=$(timeout 120 "$PY" scripts/remote_run.py exec "stat -c %s '$rd/$f' 2>/dev/null" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | tail -1)
    if [ -z "$rsz" ]; then echo "  [$name/$f] try$i: 读远端大小失败"; sleep 3; continue; fi
    # 远端压缩（幂等：-f 覆盖）
    timeout 180 "$PY" scripts/remote_run.py exec "gzip -k -f -9 '$rd/$f' 2>/dev/null; stat -c %s '$rd/$f.gz'" >/dev/null 2>&1
    gsz=$(timeout 120 "$PY" scripts/remote_run.py exec "stat -c %s '$rd/$f.gz' 2>/dev/null" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]+$' | tail -1)
    if [ -z "$gsz" ] || [ "$gsz" -eq 0 ]; then echo "  [$name/$f] try$i: 读远端 .gz 大小失败"; sleep 3; continue; fi
    rm -f "$ld/$f.gz"
    MSYS_NO_PATHCONV=1 timeout 300 "$PY" scripts/remote_run.py get "$rd/$f.gz" "$ld/$f.gz" >/dev/null 2>&1
    lgsz=$(stat -c %s "$ld/$f.gz" 2>/dev/null || echo 0)
    if [ "$lgsz" != "$gsz" ]; then echo "  [$name/$f] try$i: gz 传输不齐 remote=$gsz local=$lgsz"; sleep 3; continue; fi
    rm -f "$ld/$f"
    gzip -d -c "$ld/$f.gz" > "$ld/$f" 2>/dev/null
    lsz=$(stat -c %s "$ld/$f" 2>/dev/null || echo 0)
    rm -f "$ld/$f.gz"
    if [ "$lsz" = "$rsz" ] && [ "$lsz" -gt 0 ]; then
      echo "  OK  [$name/$f] $lsz B (gz $lgsz, try $i)"; got=1; break
    fi
    echo "  [$name/$f] try$i: 解压后 $lsz != remote_orig $rsz"; sleep 3
  done
  # 清远端中转文件（不论成败）
  timeout 120 "$PY" scripts/remote_run.py exec "rm -f '$rd/$f.gz'" >/dev/null 2>&1
  if [ $got -eq 1 ]; then okn=$((okn+1)); else badn=$((badn+1)); echo "  FAIL [$name/$f]"; fi
done

echo "=== 定点补齐: ok=$okn fail=$badn ==="
n=0; bad=0
for d in "$LROOT"/S14B2-*/; do
  if [ -s "$d/meta.json" ] && [ -s "$d/requests.jsonl" ] && [ -s "$d/telemetry.jsonl" ] && [ -s "$d/config.snapshot.yaml" ]; then
    n=$((n+1)); else bad=$((bad+1)); echo "  仍不完整: $(basename "$d")"; fi
done
echo "本地完整目录 = $n  不完整 = $bad"
