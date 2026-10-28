#!/usr/bin/env bash
# D1 队列：v7 标定（pilot -> 全量 2 会话）-> 导出 model_v7 -> D1① 验证批
# ---------------------------------------------------------------------------
# 前置（**不满足就不要启动**）：
#   1. 整树已同步到远端并**重新投放溯源戳**（否则产物里数不出 arm_state、
#      describe() 也不记 candidate_chunks ⇒ D1① 的核心判据无法核）；
#   2. **`results/raw/_env/env.json` 必须存在**（见 ⓪b）—— 新根默认没有它，
#      缺了整批会被标成 `NOT_MEASURED`，**跑多久都白跑**（2026-09-27 实测踩过）；
#   3. 没有别的批次在跑（v6 链、run_step3_matrix、fwd_costcalib_probe 全部退出）。
#
# 为什么 v7 要重采：部署换了源码树 ⇒ 标定必须对着**部署后的执行路径**重做；
# 且 export_cost_model.py 要求产物的 protocol 与 grid_sha256 各自唯一
# ⇒ 新旧点要进同一个模型就必须共用同一份网格文件。
#
# 为什么先跑 pilot：pilot（`--only-group mixed_small_chunk`，16 点）只用来**尽早发现
# 采集层面就失败**（点没采到 / 不是 real_mixed / 有 error）⇒ 早停，别把 3 小时跑完才发现。
# ⚠️ pilot **不**证明"小候选能被预测/选中/执行" —— 那要等 model_v7 + S14G 批。
#
# 终止标记：S14D1_DONE / S14D1_ABORTED_<原因>
# ---------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$0")/.." || exit 9
LOG=/root/s14_d1.log
exec >>"$LOG" 2>&1

echo "=============================================================="
echo "[d1] 开工 $(date)"

# 环境必须**显式**激活且失败要显形（`sh -c` 下 source 会静默失败 —— 踩过）
if [ -f /root/musa-envs/qwen310/activate_musa.sh ]; then
    # shellcheck disable=SC1091
    source /root/musa-envs/qwen310/activate_musa.sh || { echo "S14D1_ABORTED_ACTIVATE_FAILED"; exit 9; }
else
    echo "[d1] ⚠️ 未找到 activate_musa.sh，沿用当前环境"
fi
python -c "import torch, torch_musa; print('[d1] torch', torch.__version__, 'musa ok')" \
    || { echo "S14D1_ABORTED_NO_MUSA"; exit 9; }

GRID=configs/cost_calib/shape_grid_v7.yaml
mkdir -p adhoc results/raw/_costcalib results/processed

# ---- ⓪ 前置自检：没有别的批次在跑 ------------------------------------------
# ⚠️ 判据面**只有一处权威** = `scripts/check_remote_busy.py` 的 `TASK_SCRIPTS`。
#    之前这里写死一个 grep 正则 ⇒ 与检测脚本的判据面**必然分叉**
#    （实测：只认 run_step3_matrix 会漏掉 run_s14_d1.sh 这类外层队列）。
#    ★ 必须用 `--local`：本脚本**就运行在跑批次的那台机器上**，探测不能再去 SSH
#      （默认模式是给本机编排方用的，会去连远端；在远端自连会因缺 REMOTE_* 凭据
#      报 UNKNOWN ⇒ 闸门永远拒绝，第二条腿永远起不来 —— 已实测 rc=2）。
#    退出码：0=IDLE 可起跑 / 1=BUSY 拒绝并发 / 2=UNKNOWN **也必须拒绝**（查不出≠空闲）。
if ! python3 scripts/check_remote_busy.py --local --expect-idle; then
    echo "[d1] ✗ 前置闸门未通过（BUSY 或 UNKNOWN）⇒ 拒绝并发"
    echo "S14D1_ABORTED_BUSY"
    exit 0
fi

# ---- ⓪b 前置自检：环境指纹必须在（否则整批 NOT_MEASURED，白跑）-------------
# 2026-09-27 实测：新根 `/root/ascend_probe_d1` 没有 `results/raw/_env/`（该目录被
# `.gitignore` 排除 ⇒ **每次部署到新根都会缺**），于是 harness 判不出硬件可用性，
# 6 个 S14G 产物**全部**被标成 `NOT_MEASURED` —— run 正常结束、数字正常，
# 只有一行警告埋在日志里。⇒ 放在这里，让它**跑不起来**而不是跑完作废。
if [ ! -f results/raw/_env/env.json ]; then
    echo "[d1] ✗ results/raw/_env/env.json 不存在 ⇒ 产物会被标成 NOT_MEASURED（不可进结论）"
    echo "[d1]    先在本根采一次指纹：PYTHON=python3 bash scripts/inspect_env.sh"
    echo "S14D1_ABORTED_NO_ENV"
    exit 0
fi
echo "[d1] ✓ 环境指纹在位：$(python -c "
import json;d=json.load(open('results/raw/_env/env.json',encoding='utf-8'))
print(d.get('platform'), d.get('device',{}).get('model'), 'available=', d.get('available'))" 2>/dev/null || echo '(解析失败，继续但请人工确认)')"

# ---- ① pilot：只采新组（16 点）---------------------------------------------
SID_P="$(date -u +%Y%m%dT%H%M%SZ)-autodl-c-v7p"
python -u scripts/fwd_costcalib_probe.py --grid "$GRID" --only-group mixed_small_chunk \
    --out adhoc/costcalib_v7_pilot.json --session-id "$SID_P"
echo "[d1] pilot rc=$?"

# pilot 闸门：只查**采集完整性**（不查机制）
python - <<'PY' || { echo "S14D1_ABORTED_PILOT_GATE"; exit 0; }
import json, sys
d = json.load(open("adhoc/costcalib_v7_pilot.json", encoding="utf-8"))
pts = d.get("points") or []
err = [p for p in pts if p.get("error")]
sub = d.get("subset") or {}
ok = (len(pts) == 16 and not err and sub.get("is_full_grid") is False
      and all((p.get("shape") or {}).get("n_new_mode") == "real_mixed" for p in pts))
print(f"[d1] pilot: 点 {len(pts)} 错误 {len(err)} subset={sub}")
if not ok:
    print("[d1] ✗ pilot 闸门不过（点数/错误/is_full_grid/real_mixed 之一不符）")
    sys.exit(1)
print("[d1] ✓ pilot 采集完整")
PY

# ---- ② 全量 v7（2 会话）-----------------------------------------------------
for i in 1 2; do
    SID="$(date -u +%Y%m%dT%H%M%SZ)-autodl-c-v7s$i"
    echo "[d1] 全量会话 $i $SID $(date)"
    python -u scripts/fwd_costcalib_probe.py --grid "$GRID" \
        --out "adhoc/costcalib_v7_s$i.json" --session-id "$SID"
    echo "[d1] 会话 $i rc=$? $(date)"
done
cp -f adhoc/costcalib_v7_s*.json results/raw/_costcalib/
ls -la results/raw/_costcalib/costcalib_v7_*.json

# ---- ③ 拟合（秩保证划分；**可信口径**）-------------------------------------
python -u scripts/costcalib_fit.py --artifacts "results/raw/_costcalib/costcalib_v7_s*.json" \
    --split rank_guaranteed --hash-modulus 2 \
    --json results/processed/costcalib_eval_v7.json
echo "[d1] 拟合 rc=$?"

# ---- ④ 导出 model_v7（**总是导出**；模式/桶/口径由导出器入参写，不事后改 JSON）----
python -u scripts/export_cost_model.py \
    --eval results/processed/costcalib_eval_v7.json \
    --artifacts results/raw/_costcalib/costcalib_v7_s1.json results/raw/_costcalib/costcalib_v7_s2.json \
    --out configs/cost_calib/model_v7.json \
    --default-mode lookup --lookup-bucket-kv-step 256 --mixed-shape-mode real_mixed
[ -f configs/cost_calib/model_v7.json ] || { echo "S14D1_ABORTED_NO_MODEL"; exit 0; }

# ---- ⑤ 覆盖率闸门（**从配置读模型** —— 避免"验 A 跑 B"）---------------------
python -u scripts/check_model_coverage.py --tag S14B1 \
    --cfg configs/mthreads/phase1_d1/fixed_cost_min512.yaml \
    --replay-mode lookup --assume-kv 1568 --json results/processed/coverage_v7_vs_S14B1.json
COV=$(python -c "
import json
try: print(json.load(open('results/processed/coverage_v7_vs_S14B1.json'))['coverage'])
except Exception: print('ERR')")
echo "[d1] 覆盖率 = $COV（登记预期：v5 已 53.57%；v7 主要看**小 chunk 有没有键**）"

# 小 chunk 的键是否真的在模型里（**这是 D1① 的前提**）
python - <<'PY'
import json
m = json.load(open("configs/cost_calib/model_v7.json", encoding="utf-8"))
tbl = m.get("lookup_table") or {}
need = [f"b{b}_pf{pf}_kv{kv}" for b in (1, 2, 4, 8) for pf in (128, 256) for kv in (1024, 2048)]
hit = [k for k in need if k in tbl]
print(f"[d1] 小 chunk 键命中 {len(hit)}/{len(need)}；缺 {sorted(set(need) - set(hit))[:6]}")
print(f"[d1] default_mode={m.get('default_mode')} mixed_shape_mode={m.get('mixed_shape_mode')}")
PY

# ---- ⑥ D1① 验证批（S14G：**两个 cell**，同一模型/同一源码 ⇒ 可比）----------
#   `fixed_cost_pilot`  = min128 + 显式候选 [128,256,512,2048]
#   `fixed_cost_min512` = min512 + 派生候选 {512,2048}（**同批内的参照**）
#   ⚠️ 带 min512 是为了让闸门的比较**同模型同源码**（拿 S14D 当参照是跨模型，不可比）。
python3 scripts/check_cpu_stability.py 7 1.0 | tail -1
python -u scripts/run_step3_matrix.py --cells fixed_cost_pilot fixed_cost_min512 \
    --cfg-dir configs/mthreads/phase1_d1 --repeats 3 --tag S14G --out results/raw
echo "[d1] S14G rc=$?"

python -u scripts/diagnose_scheduling.py --tag S14G --json results/processed/diag_S14G.json

# ---- ⑦ ★ 三态闸门（PASS / FAIL / INCONCLUSIVE）------------------------------
#   判据本体在 `scripts/d1_gate.py`（**唯一实现**，含臂名解析与有效性前置）。
#   ⚠️⚠️ 是否启动 ② **只许看这一次运行的退出码**。
#   旧写法是 `grep -c "S14D1_GATE_PASS" "$LOG"` —— 而 `$LOG` 是 **append-only**，
#   历史里只要有过一次 PASS，之后**每一次**重跑都会误触发 ②（2026-09-27 实修）。
#   ★ 判据版本：**v2 = 机制诊断档**（2026-09-27 冻结）。v1 的旧三条件原样保留在
#     `d1_gate.py::criteria_v1`，可用 `--criteria v1` 复现旧结论（对照用）。
#     v2 的差别：① 覆盖率作准入（<50% ⇒ INCONCLUSIVE）；② 用**同批参照**而非跨模型阈值；
#     ③ 降级/回退**连报分母**；④ **不判服务收益**（过载区批次专用于机制诊断）。
python -u scripts/d1_gate.py --tag S14G --arm cost_pilot --ref-arm cost_min512 \
    --repeats 3 --criteria v2 \
    --diag results/processed/diag_S14G.json \
    --json results/processed/d1_gate_S14G.json
GATE_RC=$?
echo "[d1] 闸门退出码 = $GATE_RC（0=PASS / 1=FAIL / 2=INCONCLUSIVE / 3=输入错）"

case "$GATE_RC" in
    0)
        echo "[d1] ✓ ① PASS ⇒ 立刻启动 ②"
        # 连续作业：**不给空转留窗口**。起跑走唯一入口 `launch_batch.sh`
        # （取锁 + 起跑一次 + 监视进程在批次结束后放锁）。
        # ⚠️ 此前是裸 `setsid nohup ... &`：无锁 ⇒ 同一条命令发两遍就会
        #    两个矩阵同写 results/raw（已发生过）。退出码只认 launch_batch 的。
        bash scripts/launch_batch.sh S14D1 /root/s14_d1_min.log \
            bash scripts/run_s14_d1_min.sh
        LAUNCH_RC=$?
        echo "[d1] ② 起跑退出码 = $LAUNCH_RC（0=成功 4/5=被锁拦 7=起跑失败）"
        if [ "$LAUNCH_RC" -ne 0 ]; then
            echo "[d1] ⚠️ ② 未成功起跑（rc=$LAUNCH_RC）⇒ 不要当成功处理"
            echo "S14D1_SECOND_STAGE_LAUNCH_FAILED"
        fi
        ;;
    1)
        echo "[d1] ✗ ① FAIL（判据不满足）⇒ **不启动 ②**（按预注册的否证形态，先补标定）"
        ;;
    2)
        echo "[d1] ⚠️ ① INCONCLUSIVE（证据不足/无效 —— **不是**「方法没用」）⇒ 不启动 ②"
        echo "[d1]    先看 results/processed/d1_gate_S14G.json 的 validity_problems"
        ;;
    *)
        echo "[d1] ✗ 闸门输入错 rc=$GATE_RC ⇒ 不启动 ②"
        ;;
esac

# ---- ⑧ 与既有批并排（新根下可能没有 S14D ⇒ 容忍缺失，不当作失败）------------
for t in S14G S14H S14D; do
    if ls -d results/raw/${t}-* >/dev/null 2>&1; then
        python -u scripts/diagnose_scheduling.py --tag "$t" --json "results/processed/diag_$t.json" || true
    else
        echo "[d1] （跳过 $t：本根下没有它的产物）"
    fi
done

echo "[d1] 完工 $(date)"
echo "S14D1_DONE"
