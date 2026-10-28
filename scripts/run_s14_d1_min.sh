#!/usr/bin/env bash
# D1② 队列：**只改候选下限**的三点对照（min = 512 / 256 / 128）
# ---------------------------------------------------------------------------
# ⚠️ **前置**：D1①（`S14G`）必须**已判定通过**（小候选能被预测、被选中、被实际执行）。
#    ① 不通过就**不要**启动本脚本 —— 否则跑出来的仍是"降级/回退"，无法归因到下限。
#    ① 的判定依据见 `EXPERIMENTS.md` 的 D1 预注册节（含否证形态）。
#
# 唯一变量：`controller.chunk_size_min`（逐批一个值；`candidate_chunks` 保持 null
#           ⇒ 候选 = {min, max}，这正是旧行为）。其余逐字节相同（生成器 `--check` 保证）。
# 不变项：同一模型（model_v7）/ 同一 SLO / 同一负载 / 同源码树。
#
# 机时：3 点 × 5 重复 = 15 次运行（fixed 格 ≈1.2 min/次；小 chunk 会多几步 ⇒ 略长）。
# 终止标记：S14D2_DONE / S14D2_ABORTED_<原因>
# ---------------------------------------------------------------------------
set -uo pipefail

cd "$(dirname "$0")/.." || exit 9
LOG=/root/s14_d1_min.log
exec >>"$LOG" 2>&1

echo "=============================================================="
echo "[d2] 开工 $(date)"

if [ -f /root/musa-envs/qwen310/activate_musa.sh ]; then
    # shellcheck disable=SC1091
    source /root/musa-envs/qwen310/activate_musa.sh || { echo "S14D2_ABORTED_ACTIVATE_FAILED"; exit 9; }
fi

# ---- ⓪a 前置：环境指纹必须在（否则 ② 的产物同样会 NOT_MEASURED）------------
if [ ! -f results/raw/_env/env.json ]; then
    echo "[d2] ✗ results/raw/_env/env.json 不存在 ⇒ 产物会被标成 NOT_MEASURED，拒绝启动"
    echo "S14D2_ABORTED_NO_ENV"; exit 0
fi

# ---- ⓪b 前置：① 必须**三态判定为 PASS** ------------------------------------
#   判据与解析的**唯一权威**是 `scripts/d1_gate.py`（本文件此前自带一套内联判据，
#   与诊断工具的臂名解析口径不一致 ⇒ 2026-09-27 报出两条假失败）。
#   ⚠️ 只认**本次调用**的退出码，**不得**回头 grep 累计日志。
#   ★ 判据版本必须与主链 `run_s14_d1.sh` **一致** = v2（机制诊断档，2026-09-27 冻结）。
python -u scripts/d1_gate.py --tag S14G --arm cost_pilot --ref-arm cost_min512 \
    --repeats 3 --criteria v2 \
    --diag results/processed/diag_S14G.json \
    --json results/processed/d1_gate_S14G.json
GATE1_RC=$?
if [ "$GATE1_RC" -ne 0 ]; then
    echo "[d2] ✗ ① 的判定不是 PASS（rc=$GATE1_RC：1=FAIL / 2=INCONCLUSIVE）"
    echo "[d2]    依据在 results/processed/d1_gate_S14G.json ⇒ 拒绝启动 ②"
    echo "S14D2_ABORTED_GATE1"; exit 0
fi
echo "[d2] ✓ ① 判定为 PASS"

# 判据面唯一权威 = check_remote_busy.py；★ `--local`（本脚本就在跑批次的机器上，
# 不能再去 SSH —— 远端自连会因缺 REMOTE_* 凭据报 UNKNOWN，闸门永远拒绝）。
# 查不出（UNKNOWN=2）同样**拒绝** —— "查不出"不等于"空闲"。
if ! python3 scripts/check_remote_busy.py --local --expect-idle; then
    echo "[d2] ✗ 前置闸门未通过（BUSY 或 UNKNOWN）⇒ 拒绝并发"
    echo "S14D2_ABORTED_BUSY"; exit 0
fi

# ---- ① 三点 × 5 重复（**逐批一个值**，批间只差下限）-------------------------
python3 scripts/check_cpu_stability.py 7 1.0 | tail -1
for cell in fixed_cost_min512 fixed_cost_min256 fixed_cost_min128; do
    echo "[d2] ---- $cell $(date)"
    python -u scripts/run_step3_matrix.py --cells "$cell" \
        --cfg-dir configs/mthreads/phase1_d1 --repeats 5 --tag S14H --out results/raw
    echo "[d2] $cell rc=$?"
done

# ---- ② 机械检验 + D0/D1/D2 并排 --------------------------------------------
# ⚠️ 这里以前是一段**内联的**目录名解析（`re.sub(r"-r\d+$",...).replace("short_chat-",...)`）
#    —— 那是同一件事的**第四份**实现。D1① 的假失败就是同源：解析分叉 ⇒ 查不到数据 ⇒
#    把"数不出来"报成"没发生"。现在一律走 `scripts/d1_gate.py --all-arms`（统一解析）。
echo "[d2] == 机械检验：最小 prefill_tok 应随下限降低而降低；降级要连报分母 =="
python -u scripts/d1_gate.py --tag S14H --all-arms \
    --json results/processed/d1_arms_S14H.json || true

for t in S14H S14G S14D; do
    # 新根（独立部署目录）下可能没有 S14D 的产物 ⇒ 容忍缺失，**不**当作失败
    if ls -d results/raw/${t}-* >/dev/null 2>&1; then
        python -u scripts/diagnose_scheduling.py --tag "$t" --json "results/processed/diag_$t.json" || true
    else
        echo "[d2] （跳过 $t：本根下没有它的产物）"
    fi
done

echo "[d2] ② 完工 $(date)"
echo "S14D2_DONE"

# ---- ③ 第 15 步消融：**按预注册规则机械选点** ⇒ 不等人工判断（避免空转）--------
#   规则：`EXPERIMENTS.md` 的「D1② · 选取规则（先冻结）」，实现见 `scripts/pick_d1_min.py`。
#   ⚠️ 这条是"把判断从人手里拿走"，**不是**"绕过判断" —— 规则在**看到 ② 数据之前**就冻结了。
python -u scripts/diagnose_scheduling.py --tag S14H --json results/processed/diag_S14H.json || true
python scripts/pick_d1_min.py --diag results/processed/diag_S14H.json \
    --json results/processed/pick_d1_min.json > /tmp/pick_d1.txt 2>&1 || true
cat /tmp/pick_d1.txt
PICK=$(grep '^PICK=' /tmp/pick_d1.txt | head -1 | cut -d= -f2)
echo "[d2] 规则选出的 chunk_size_min = ${PICK:-（空）}"
if [ -z "${PICK:-}" ]; then
    echo "[d2] ✗ 选点失败 ⇒ **不启动消融**（如实登记，别硬编一个值）"
    echo "S14K_SKIPPED_PICK_FAILED"
    exit 0
fi

# 消融配置**生成**（不是手抄）：钉在选出的下限 + 最终模型上；输出到新目录，
# 不动被 `tests/test_step15_configs.py` 逐字节钉住的 `phase1_step15/`（否则弄红回归）。
if ! python scripts/gen_phase1_step15_configs.py --min "$PICK" \
        --model configs/cost_calib/model_v7.json --out-dir configs/mthreads/phase1_s14k; then
    echo "[d2] ✗ 消融配置生成失败 ⇒ 不跑消融"
    echo "S14K_SKIPPED_GEN_FAILED"
    exit 0
fi

python3 scripts/check_cpu_stability.py 7 1.0 | tail -1
python -u scripts/run_step3_matrix.py \
    --cells fixed_cost_full fixed_cost_no_host fixed_cost_no_margin \
            fixed_cost_signal_tpot fixed_cost_chunk_only fixed_cost_budget_only \
    --cfg-dir configs/mthreads/phase1_s14k --repeats 3 --tag S14K --out results/raw
echo "[d2] S14K rc=$?"
python -u scripts/diagnose_scheduling.py --tag S14K --json results/processed/diag_S14K.json || true
echo "[d2] ③ 完工 $(date)"
echo "S14K_DONE"
