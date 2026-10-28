#!/usr/bin/env python
"""**预算函数示例**：报告候选命中率 / 预测值 / 预算分支 / 回退原因。

★ 本脚本是**预算函数示例**，**不是**本批机制的触发证据
------------------------------------------------------------------
⚠️ **用户 2026-10-03 点名**：下表不得称"本批机制已触发"。

原因：本脚本**把 `(batch, prefill_tok, kv_len)` 人手喂给** `CostModel.predict`，
再**自己**调 `CostPolicy._budget_for` 问归类。它**没有**走
`load_config → build_arm → plan_step` ⇒ 那个形状是**手选的**，不是调度器算出来的。
⇒ 它**只能**证明"**预算函数在给定输入下怎么算**"（= 预算函数示例），
  **不能**证明"控制器在真实队列状态下会怎么选"。

⇒ **要判定"本批机制是否被行使"，看** `scripts/dryplan_decision_chain.py`
  （真实决策链演练）：那个脚本用计划负载可达的队列驱动真链路，
  产物里才有"生成候选 / 逐候选预测 / 预算分支 / 所选动作 / 回退原因"。

★ 这份产物是 `TEST_DATA`，不是真机标定
------------------------------------------------
它用 `CostModel.load_offline()`（**跳过指纹校验**，并把这件事记进 `relaxations`）。

⇒ 本报告**只能**用于：① 确认预算函数在四点上会给出两种分支；② 开跑前发现
   "某个输入恒回退"这类**配置级**问题。
⇒ 本报告**不能**用于：预测真机能拿到多少 TTFT/TPOT、判定双预算"有效"、
   或替代 `NEXT_BATCHES_PREREG.md` §6 要求的**标定环境** dry-plan。

⚠️ **`--dry-run` 的已知边界**（与本报告无关但常被混淆）：正式 `--dry-run` 从不喂
`decode_samples` ⇒ 判据恒"不可测" ⇒ 控制器保守不动。本脚本**绕开**那条路径。

用法::

    python scripts/enumerate_prereg_states.py
    python scripts/enumerate_prereg_states.py --json results/processed/prereg_states.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.chunk_policy import ChunkLimits          # noqa: E402
from ascend_opt.cost_model import CostModel              # noqa: E402
from ascend_opt.cost_policy import CostPolicy, CostPolicyConfig  # noqa: E402

COST_MODEL = _ROOT / "configs" / "cost_calib" / "model_v7.json"
STEP_BUDGET_MS = 120.0        # 内部每步预算（decode 护栏）
PREFILL_BUDGET_MS = 2000.0    # prefill 预算（服务 TTFT）
LIMITS = ChunkLimits(chunk_size=1024, max_num_batched_tokens=8192)

#: host 侧逐步开销（ms，按 batch）—— 与配置里 `controller.host_overhead_ms` 同源。
#: ⚠️ 模型预测的是**前向**，预算约束的是**整步**；不加 host 就是口径错误。
HOST_OVERHEAD_MS: dict[int, float] = {2: 3.06, 4: 10.76, 8: 25.29}
#: 表里没列到的 batch 用这条线性拟合外推（与配置的 `host_overhead_fit` 同源）。
HOST_FIT: tuple[float, float] | None = (3.705, -4.35)

#: 四个**示例输入**（**预算函数示例**，不是"本批代表状态"）。
#: 每个都写清"它代表什么"，但 ⚠️ 它们是**人手选的形状**，不是调度器算出来的。
#: ⇒ 不得据此称"本批机制已触发"（那要看 dryplan_decision_chain.py）。
#: ⚠️ 形状必须落在 v7 的**标定范围内**，否则"越界"会盖住真正要看的"预算分支"信息。
STATES: list[dict] = [
    {
        "name": "pure_prefill",
        "desc": "纯 prefill：队列里只有等待起步的请求，没有 decode 在跑"
                "（**双预算 prefill 分支的唯一适用场景**）",
        "feed": "等待中 2 个（各 1536 token，尚未起步），在解码 0 个",
        "batch": 1, "prefill_tok": 1024, "kv_len": 0, "decode_at_risk": False,
    },
    {
        "name": "pure_decode",
        "desc": "纯 decode：没有等待起步的请求，全在解码",
        "feed": "等待中 0 个，在解码 4 个（各已解 1024 token）",
        "batch": 4, "prefill_tok": 0, "kv_len": 1024, "decode_at_risk": True,
    },
    {
        "name": "mixed_step",
        "desc": "混合步：既有 decode 在跑、又有新请求要起步"
                "（**判决点**：按 §5.1 修正，此处必须走 decode 预算而非 prefill）",
        "feed": "等待中 1 个（1536 token），在解码 4 个",
        "batch": 4, "prefill_tok": 512, "kv_len": 1024, "decode_at_risk": True,
    },
    {
        "name": "lookup_miss",
        "desc": "查表缺失：**在标定范围内**、但该 (batch,pf,kv) 无实测键"
                " ⇒ lookup 模式返回 None ⇒ 应回退",
        "feed": "在解码 6 个，本步 pf=128（v7 的 pf 网格无 128 键）",
        "batch": 6, "prefill_tok": 128, "kv_len": 1024, "decode_at_risk": True,
    },
]


def _probe_modes(model: CostModel, st: dict) -> dict[str, Any]:
    """同一点在 `lookup` 与 `segmented` 两种预测方式下的对照。"""
    out: dict[str, Any] = {}
    for mode in ("lookup", "segmented"):
        p = model.predict(batch=st["batch"], prefill_tok=st["prefill_tok"],
                          kv_len=st["kv_len"], n_forwards=1, mode=mode)
        out[mode] = None if p is None else round(p.total_ms, 2)
    return out


def _probe(model: CostModel, st: dict) -> dict:
    """对单个状态问三件事：模型怎么说、范围判定、两套预算各怎么归类。

    ⚠️ **口径提醒**：模型预测的是**前向**（`fwd_ms`），而预算约束的是**整步**。
    要把两者比出结论，必须**加上 host 开销**（`host_overhead_ms`）。
    本报告因此**同时**给出"前向预测"和"整步预测（含 host）"，
    并用**整步**去比预算 —— 直接拿前向比预算是本项目记录在案的口径错误。
    """
    pred = model.predict(batch=st["batch"], prefill_tok=st["prefill_tok"],
                         kv_len=st["kv_len"], n_forwards=1)   # 走模型自己的 default_mode
    in_rng = model.in_range(batch=st["batch"], prefill_tok=st["prefill_tok"],
                            kv_len=st["kv_len"])
    modes = _probe_modes(model, st)
    # ★ 预算分支：直接问 `_budget_for`（唯一权威），不自己复算规则
    pol_dual = CostPolicy(CostPolicyConfig(enabled=True,
                                           step_deadline_ms=STEP_BUDGET_MS,
                                           prefill_deadline_ms=PREFILL_BUDGET_MS))
    pol_single = CostPolicy(CostPolicyConfig(enabled=True,
                                             step_deadline_ms=STEP_BUDGET_MS,
                                             prefill_deadline_ms=None))
    kind_d, ms_d = pol_dual._budget_for(
        tokens_prefilled=st["prefill_tok"], decode_at_risk=st["decode_at_risk"])
    kind_s, ms_s = pol_single._budget_for(
        tokens_prefilled=st["prefill_tok"], decode_at_risk=st["decode_at_risk"])

    # host 开销：按 batch 查表，表里没有就用线性拟合（与配置里的规则一致）
    host = 0.0
    if st["batch"] in HOST_OVERHEAD_MS:
        host = HOST_OVERHEAD_MS[st["batch"]]
    elif HOST_FIT is not None:
        host = max(0.0, HOST_FIT[0] * st["batch"] + HOST_FIT[1])

    hit = pred is not None
    step_ms = None if pred is None else pred.total_ms + host
    return {
        "state": st["name"],
        "desc": st["desc"],
        "feed": st["feed"],
        "shape": {"batch": st["batch"], "prefill_tok": st["prefill_tok"],
                  "kv_len": st["kv_len"]},
        "in_calibrated_range": in_rng,
        "predicted_forward_ms": (None if pred is None else round(pred.total_ms, 2)),
        "predicted_by_mode": modes,
        "host_overhead_ms": round(host, 2),
        "predicted_step_ms": (None if step_ms is None else round(step_ms, 2)),
        "margin": (None if pred is None else pred.margin),
        "candidate_hit": hit,
        "fallback_reason": (None if hit else
                            ("越界：不在标定范围内" if not in_rng
                             else "该 (batch,pf,kv) 无实测键（lookup 未命中）")),
        "budget_dual": {"kind": kind_d, "ms": ms_d},
        "budget_single": {"kind": kind_s, "ms": ms_s},
        "budget_branch_differs": (kind_d != kind_s) or (ms_d != ms_s),
        "feasible_under_dual": (step_ms is not None and step_ms <= ms_d),
        "feasible_under_single": (step_ms is not None and step_ms <= ms_s),
        "would_degrade": (step_ms is not None and step_ms > ms_d),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None)
    ap.add_argument("--cost-model", default=str(COST_MODEL))
    args = ap.parse_args()

    model = CostModel.load_offline(args.cost_model)
    print("=" * 100)
    print("预算函数示例（**TEST_DATA** —— 离线演练，不代替真机标定）")
    print("⚠️ 本表**不是**本批机制的触发证据（形状是手选的）；机制是否被行使请看")
    print("   scripts/dryplan_decision_chain.py（真实决策链演练）")
    print("=" * 100)
    d = model.describe()
    print(f"成本模型 : {Path(args.cost_model).name}  ver={d.get('model_version')} "
          f"grid={d.get('grid_sha256')} default_mode={d.get('default_mode')}")
    print(f"预算     : 内部每步 {STEP_BUDGET_MS:g} ms（decode 护栏） / "
          f"prefill {PREFILL_BUDGET_MS:g} ms / 外部 SLO tpot 100 ms")
    print(f"relaxations: {model.relaxations}")
    print()

    rows = [_probe(model, st) for st in STATES]

    print(f"{'状态':<14}{'形状(b,pf,kv)':>16}{'范围':>6}{'lookup':>9}{'segm':>9}"
          f"{'host':>7}{'整步ms':>9}{'双预算':>14}{'单预算':>14}{'可行':>6}{'回退原因':>26}")
    for r in rows:
        sh = r["shape"]
        shape_s = f"({sh['batch']},{sh['prefill_tok']},{sh['kv_len']})"
        inr = "✓" if r["in_calibrated_range"] else "✗"
        lk = "—" if r["predicted_by_mode"]["lookup"] is None else str(r["predicted_by_mode"]["lookup"])
        sg = "—" if r["predicted_by_mode"]["segmented"] is None else str(r["predicted_by_mode"]["segmented"])
        stp = "—" if r["predicted_step_ms"] is None else str(r["predicted_step_ms"])
        bd = f"{r['budget_dual']['kind']}/{r['budget_dual']['ms']:g}"
        bs = f"{r['budget_single']['kind']}/{r['budget_single']['ms']:g}"
        ok = "✓" if r["feasible_under_dual"] else "✗"
        why = r["fallback_reason"] or "—"
        print(f"{r['state']:<14}{shape_s:>16}{inr:>6}{lk:>9}{sg:>9}"
              f"{r['host_overhead_ms']:>7}{stp:>9}{bd:>14}{bs:>14}{ok:>6}{why:>26}")

    n_hit = sum(1 for r in rows if r["candidate_hit"])
    n_branch = sum(1 for r in rows if r["budget_branch_differs"])
    n_feas_d = sum(1 for r in rows if r["feasible_under_dual"])
    n_feas_s = sum(1 for r in rows if r["feasible_under_single"])
    print()
    print(f"候选命中率       : {n_hit}/{len(rows)}")
    print(f"预算分支有差异   : {n_branch}/{len(rows)}"
          f"  ⇒ {'双预算在这四点改变了归类' if n_branch else '⚠️ 双预算在这四点完全等价'}")
    print(f"双预算下可行     : {n_feas_d}/{len(rows)}")
    print(f"单预算下可行     : {n_feas_s}/{len(rows)}")
    print()
    print("⚠️ 限定：本表是**预算函数示例**（手选形状）⇒ 上两行**不能**读成")
    print("   「两套预算给出不同可行集 ⇒ 本批**不是**『机制被行使』」。")
    print("   ★ 机制是否被行使要看 dryplan_decision_chain.py 的真实决策链结果。")

    payload = {
        "evidence_class": "TEST_DATA",
        "artifact_role": "budget_function_example",
        "not_evidence_of": "本批机制被行使 —— 形状是手选的，未走 plan_step（见 dryplan_decision_chain.py）",
        "note": "离线演练；不代替真机标定，不得据此宣布双预算有效/无效（FZ-1/FZ-2 未裁定）",
        "cost_model": {"path": str(args.cost_model), **{k: d.get(k) for k in
                       ("model_version", "protocol_version", "grid_sha256",
                        "default_mode", "safety_margin")}},
        "cost_model_relaxations": model.relaxations,
        "budgets": {"internal_step_ms": STEP_BUDGET_MS,
                    "prefill_ms": PREFILL_BUDGET_MS, "external_slo_tpot_ms": 100.0},
        "states": rows,
    }
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"\n写出 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
