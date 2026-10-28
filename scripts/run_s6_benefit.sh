#!/usr/bin/env bash
# S6 服务收益档队列：三臂（static / threshold / cost）× 同一 SLO 可达负载（c2_m4）
# ============================================================================
# 预注册：`EXPERIMENTS.md` §「第 6 步 · S6 服务收益档 · 预注册」
# 配置：  `configs/mthreads/phase1_s6/`（生成器 `scripts/gen_phase1_s6_configs.py`，带 `--check`）
# 测试：  `tests/test_s6_configs.py`（13 条，逐字段钉住"只差两个开关"）
#
# 本队列**只做一件事**：在已证明 SLO 可达的负载上，比较三臂的服务收益。
#   ✅ 回答：cost 相对 threshold / static 有没有服务收益
#   ❌ 不回答：小 chunk 能否被执行（机制诊断档 S14G）
#   ❌ 不回答：负载可达性（S14S）
# 三档分离见 §7d 第 6 节 —— **不得**与本批之外的档合并成一批。
#
# 前置（**不满足就不要启动**，全部由本脚本自检）：
#   ① 整树已同步到**权威根**且戳 validated 且 `commit == 本机 HEAD`；
#   ② `results/raw/_env/env.json` 在位（缺了整批 NOT_MEASURED，跑多久都白跑）；
#   ③ 无并发批次（`check_remote_busy.py --local --expect-idle` 返回 0）；
#   ④ 无锁（首次起跑正常；held=True 说明有别的批次或残留锁）。
#
# ⚠️ 起跑**不得**裸 `setsid nohup ... &`：走唯一入口 `launch_batch.sh`
#    （取锁 + 起跑一次 + 监视进程在批次结束后放锁）。本脚本由 `launch_batch.sh`
#    拉起，因此**本脚本内部不再取锁**（锁已在父进程持有）。
#
# 终止标记：S6_DONE / S6_ABORTED_<原因>
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.." || { echo "S6_ABORTED_CD_FAILED"; exit 9; }
LOG=/root/s6_benefit.log
exec >>"$LOG" 2>&1

TAG=S6
CFG_DIR=configs/mthreads/phase1_s6
REPEATS=3          # 预注册「--repeats ≥ 3；判据 = 3 轮全部满足（取最差一次）」
CELLS=(service_benefit_c2_m4_static service_benefit_c2_m4_threshold service_benefit_c2_m4_cost)

echo "=============================================================="
echo "[s6] 开工 $(date)"
echo "[s6] 预注册 = EXPERIMENTS.md §「第 6 步 · S6 服务收益档 · 预注册」"
echo "[s6] 载荷冻结 = c2_m4（concurrency=2 / max_num_seqs=4）；臂 = ${CELLS[*]}"

# ---- 环境必须**显式**激活且失败要显形（`sh -c` 下 source 会静默失败 —— 踩过）----
if [ -f /root/musa-envs/qwen310/activate_musa.sh ]; then
    # shellcheck disable=SC1091
    source /root/musa-envs/qwen310/activate_musa.sh || { echo "S6_ABORTED_ACTIVATE_FAILED"; exit 9; }
else
    echo "[s6] ⚠️ 未找到 activate_musa.sh，沿用当前环境"
fi
python -c "import torch, torch_musa; print('[s6] torch', torch.__version__, 'musa ok')" \
    || { echo "S6_ABORTED_NO_MUSA"; exit 9; }

mkdir -p results/raw results/processed

# ---- ⓪ 前置自检：无并发批次 -------------------------------------------------
# ⚠️ 判据面**只有一处权威** = `scripts/check_remote_busy.py` 的 TASK_SCRIPTS。
#    必须 `--local`：本脚本就运行在跑批次的那台机器上，默认模式会去 SSH 远端。
#    退出码：0=IDLE / 1=BUSY / 2=UNKNOWN ⇒ **2 也必须拒绝**（查不出≠空闲）。
if ! python3 scripts/check_remote_busy.py --local --expect-idle; then
    echo "[s6] ✗ 前置闸门未通过（BUSY 或 UNKNOWN）⇒ 拒绝并发"
    echo "S6_ABORTED_BUSY"
    exit 0
fi

# ---- ⓪b 前置自检：环境指纹必须在（否则整批 NOT_MEASURED，白跑）--------------
if [ ! -f results/raw/_env/env.json ]; then
    echo "[s6] ✗ results/raw/_env/env.json 不存在 ⇒ 产物会被标成 NOT_MEASURED（不可进结论）"
    echo "[s6]    先在本根采一次指纹：PYTHON=python3 bash scripts/inspect_env.sh"
    echo "S6_ABORTED_NO_ENV"
    exit 0
fi
echo "[s6] ✓ 环境指纹在位：$(python -c "
import json;d=json.load(open('results/raw/_env/env.json',encoding='utf-8'))
print(d.get('platform'), d.get('device',{}).get('model'), 'available=', d.get('available'))" 2>/dev/null || echo '(解析失败，继续但请人工确认)')"

# ---- ⓪c 前置自检：部署戳必须**字段齐全**（漂移会让产物无法溯源）------------
#   ★ 这一块连错两次，两个错误方向都值得记住（都是"读了记忆里的键名/形状"）：
#     (a) 曾用 `git rev-parse HEAD` 取远端 HEAD —— **远端部署树不是 git 仓库**
#         （`remote_run.py` 第 29/425 行写明：现场 `git rev-parse` 必然失败，
#          连产物的 `git_commit` 都会退成 `NOT_MEASURED`）⇒ 取到**空串**、
#          比较永远不等 ⇒ 每一个合法部署都被判漂移、批次一启动就自杀
#          （实测：S6 首跑 `S6_ABORTED_STAMP_MISMATCH`，而戳其实是好的）。
#     (b) 改用 `prov.get("commit")` / `prov.get("validated")` —— 仍是记忆里的键名。
#         真实戳（`SOURCE_PROVENANCE.json`）里根本没有这两个键，
#         字段名是 **`git_commit` / `git_dirty` / `tree_digest`**；
#         `validated` 只存在于 `stamp` 的**内存校验结果**里，**不落盘**。
#   ⇒ 结论：**戳的正确性由 `remote_run.py stamp` 的第 3 步（远端重算摘要）保证**，
#      本脚本只负责"确认戳文件字段齐全、digest 非空"，**不重复实现校验**。
#      （"同一件事不要写第二份实现"——本项目已因重复实现分叉踩过三次。）
if [ -f SOURCE_PROVENANCE.json ]; then
    S6_EXPECT_COMMIT="${S6_EXPECT_COMMIT:-}" \
    python - <<'PY' || { echo "S6_ABORTED_STAMP_INVALID"; exit 0; }
import json, os, sys
prov = json.load(open("SOURCE_PROVENANCE.json", encoding="utf-8"))
commit = prov.get("git_commit")
dirty = prov.get("git_dirty")
digest = prov.get("tree_digest")
expect = os.environ.get("S6_EXPECT_COMMIT") or ""
print(f"[s6] 戳 git_commit={commit} git_dirty={dirty} "
      f"digest={(digest or '')[:12]}… 期望={expect or '(未传入)'}")
# 字段必须齐全（真字段名，不是 commit/validated）
missing = [k for k in ("git_commit", "tree_digest") if not prov.get(k)]
if missing:
    print(f"[s6] ✗ 戳缺字段 {missing} ⇒ 部署未经有效投放，拒绝起跑")
    print("[s6]    重新投放：python3 scripts/remote_run.py stamp //root/ascend_probe_d1")
    sys.exit(1)
if digest and len(str(digest)) != 64:
    print(f"[s6] ✗ tree_digest 长度异常（{len(str(digest))} != 64）⇒ 戳可疑，拒绝起跑")
    sys.exit(1)
# ⚠️ git_dirty=True 不判失败，但要显形：那一批的 git_commit 只能当"分支点"
if dirty is True:
    print("[s6] ⚠️ 戳 git_dirty=True ⇒ 本机当时有未提交改动，git_commit 只作分支点")
# 仅当调用方显式传入期望 commit 时才比对（远端无 git 仓库，不能现场取 HEAD）
if expect and commit != expect:
    print(f"[s6] ✗ 戳 git_commit {commit} != 期望 {expect} ⇒ 部署漂移，拒绝起跑")
    sys.exit(1)
print("[s6] ✓ 部署戳字段齐全（校验由 stamp 的远端重算保证）")
PY
else
    echo "[s6] ✗ 未找到 SOURCE_PROVENANCE.json ⇒ 无法核部署溯源 ⇒ 拒绝起跑"
    echo "[s6]    先投放戳：python3 scripts/remote_run.py stamp //root/ascend_probe_d1"
    echo "S6_ABORTED_NO_STAMP"
    exit 0
fi

# ---- ⓪d 前置自检：S6 三臂配置在位 ------------------------------------------
for c in "${CELLS[@]}"; do
    [ -f "$CFG_DIR/$c.yaml" ] || { echo "[s6] ✗ 缺配置 $CFG_DIR/$c.yaml"; echo "S6_ABORTED_NO_CONFIG"; exit 0; }
done
echo "[s6] ✓ S6 三臂配置 3/3 在位"

# ---- ⓪e 前置自检：配置自检通过（含真加载器级校验）----------------------------
python3 scripts/gen_phase1_s6_configs.py --check \
    || { echo "[s6] ✗ S6 配置自检不过 ⇒ 拒绝起跑"; echo "S6_ABORTED_CONFIG_CHECK_FAILED"; exit 0; }
echo "[s6] ✓ 生成器 --check 通过"

# ---- ① 预热/稳定性检查（与 S14D1 同口径）-----------------------------------
python3 scripts/check_cpu_stability.py 7 1.0 | tail -1 || echo "[s6] ⚠️ CPU 稳定性检查非零退出（继续）"

# ---- ② 三臂矩阵（穿插执行：同格重复相隔整张表）------------------------------
echo "[s6] 起跑矩阵：$(date)"
python3 -u scripts/run_step3_matrix.py \
    --cells "${CELLS[@]}" \
    --cfg-dir "$CFG_DIR" \
    --repeats "$REPEATS" \
    --tag "$TAG" \
    --out results/raw
MATRIX_RC=$?
echo "[s6] 矩阵 rc=$MATRIX_RC $(date)"

# ---- ③ 诊断（**唯一权威**：逐臂 cost_policy 覆盖率 + 调度事实）---------------
#   ⚠️ 不另写解析器：`diagnose_scheduling.py` 已是"逐臂读轨迹"的唯一实现
#      （含 `coverage_of_model` 判别 `cost_policy` 条目），本项目有"同一件事
#      多份实现必然分叉"的教训（臂名解析三份、并发判据两份）。
python -u scripts/diagnose_scheduling.py --tag "$TAG" --json results/processed/diag_$TAG.json
echo "[s6] 诊断 rc=$?"

# ---- ④ 机械事实报表（**统一入口**，不判定）----------------------------------
#   `d1_gate.py --all-arms` 逐臂列出小 chunk 步数 / 降级分母 / 饥饿 —— 给读日志的人
#   一份结构化事实，**不含**任何"有收益/无收益"的判断（那是第 ⑤ 步的事）。
python -u scripts/d1_gate.py --tag "$TAG" --all-arms \
    --raw-root results/raw --json results/processed/all_arms_$TAG.json
echo "[s6] 机械事实报表 rc=$?"

# ---- ⑤ 服务收益判读（**预注册 §5/§6/§7 的机械执行**）------------------------
#   ⚠️ 判据本体在 `scripts/s6_benefit_gate.py`（唯一实现）。**不在此处手写判据**：
#      预注册的规则必须由代码机械执行，不能事后"看着数据商量"。
#   三态：0=PASS / 1=FAIL（cost 严格劣于 static）/ 2=INCONCLUSIVE（不可区分/接线失败）
#   ⚠️ 只传闸门**确实支持**的参数。`--diag` 是 `d1_gate.py` 的选项，不是本闸门的
#      —— 首跑实测：多传它 ⇒ argparse 报 `unrecognized arguments` ⇒ 闸门退出码 2，
#      与「INCONCLUSIVE」**同码**，日志于是把一次**参数错误**印成
#      「⚠️ INCONCLUSIVE（三臂不可区分/接线失败）⇒ 不得声称任一臂更优」。
#      ★ 这是「错误的值比缺的值危险」的又一例：判定没跑，却留下了一个像判定的结论。
python -u scripts/s6_benefit_gate.py --tag "$TAG" --root results/raw --repeats "$REPEATS" \
    --json results/processed/s6_benefit_gate.json
GATE_RC=$?
# ---- ⑤b ★ 把「参数错」与「判为 INCONCLUSIVE」拆开 ---------------------------
#   退出码 2 有两种来源：闸门**判了** INCONCLUSIVE，或 argparse**根本没跑**。
#   判据 = 结构化产物是否产出（闸门只在真正判完后才写 `--json`）。
#   没有产物 ⇒ 是调用错，**不得**读成结论。
if [ ! -s results/processed/s6_benefit_gate.json ]; then
    echo "[s6] ✗✗ 闸门**没有产出判定文件** ⇒ 这是**调用错误**，不是 INCONCLUSIVE；"
    echo "       不得读成「三臂不可区分」，也不得读成任何收益结论。请先修命令行。"
    GATE_RC=3
fi
echo "[s6] 服务收益闸门退出码 = $GATE_RC（0=PASS / 1=FAIL / 2=INCONCLUSIVE / 3=输入错）"
case "$GATE_RC" in
    0) echo "[s6] ✓ 判为 PASS（cost 有服务收益）⇒ 按预注册 §5 主预期登记" ;;
    1) echo "[s6] ✗ 判为 FAIL（cost 严格劣于 static）⇒ 记「负收益」并按 §7 登记" ;;
    2) echo "[s6] ⚠️ INCONCLUSIVE（三臂不可区分 / 接线失败）⇒ **不得**声称任一臂更优" ;;
    *) echo "[s6] ✗ 闸门输入错 rc=$GATE_RC ⇒ 不判收益" ;;
esac

echo "[s6] 完工 $(date)"
echo "S6_DONE"
