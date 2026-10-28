#!/usr/bin/env python
"""S00 阻断项证据脚本（只读产物；**不改** `results/raw/**`、不改执行路径）。

对应 `handoff/optimization_20260929/01_PROJECT_PLAN.md` §5 的阻断项：

* **§5.1 双预算语义**：`CostPolicy._budget_for` 只按**候选自己的** `tokens_prefilled > 0`
  选较宽的 prefill 预算，**不**考虑"本步仍在等 token 的 decode 请求"的逐 token 约束。
  本脚本把这一分类**数出来**（每条候选的 `budget_kind` / `budget_ms`），并给出
  "混合步里 decode 更紧、却被 prefill 预算放行"的可复算证据。
* **§5.2 闭环抖动到达语义**：`closed_jittered` 把 `scheduled_arrival` 推到
  `step_start + jitter`（未来），但请求**仍在 `step_start` 就入队** ⇒
  `enqueue_ms < arrival_ms`（入队早于到达）。本脚本把倒置**数出来**。
* （§5.3 的两条已在代码层修复，见 `tests/test_slo_weave_sparse_lookup.py`；
  本脚本同时复核其修复后的状态。）

用法::

    python scripts/audit_s00_blockers.py --json results/processed/s00_audit.json

退出码：0 = 证据齐；2 = 证据不齐（例如闭环抖动场景构造失败）。
**非零 ≠ 方法被否证** —— 它只表示"这一项的证据没拿到"。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.cost_model import CostModel  # noqa: E402
from ascend_opt.cost_policy import CostPolicy, CostPolicyConfig  # noqa: E402
from ascend_opt.cache_policy import PrefixCacheConfig, PrefixCachePolicy  # noqa: E402
from ascend_opt.chunk_policy import (  # noqa: E402
    ChunkConfig, ChunkLimits, DecodePriorityChunkPolicy)
from ascend_opt.scheduler import ContinuousBatchingScheduler  # noqa: E402
from ascend_opt.workload import Request, build_requests  # noqa: E402
from run_benchmark import load_config, simulate  # noqa: E402
from workloads import get_workload  # noqa: E402

BASELINE = _ROOT / "configs" / "baseline" / "default.yaml"
MODEL_V7 = _ROOT / "configs" / "cost_calib" / "model_v7.json"
JITTER_SPAN_MS = 25.0


# --------------------------------------------------------------------------- #
# §5.1 双预算 decode 保护语义
# --------------------------------------------------------------------------- #

def _dual_budget_case(*, waiting, decoding, deadline_ms: float,
                      prefill_deadline_ms: float, label: str) -> dict[str, Any]:
    """跑一个双预算场景，统计"被 prefill 预算放行、而按 decode 预算不可行"的候选数。

    ★ 为什么要**分场景**（2026-10-03）：旧探针只跑 `decoding=[]`（纯 prefill 步），
    于是它数出来的 `n_relaxed_by_prefill_budget=6` 其实是**按设计**的行为
    （纯 prefill 步服务 TTFT，此时没有任何 decode token 被推迟）。
    ⇒ 真正要看的那个数在**混合步**（有 decode 在跑）上：
      修好之后它必须是 **0**。
    """
    model = CostModel.load_offline(MODEL_V7)
    pol = CostPolicy(CostPolicyConfig(enabled=True, step_deadline_ms=deadline_ms,
                                      prefill_deadline_ms=prefill_deadline_ms), model)
    d = pol.decide(
        scheduler=ContinuousBatchingScheduler(),
        chunk_policy=DecodePriorityChunkPolicy(ChunkConfig(enabled=True)),
        cache=PrefixCachePolicy(PrefixCacheConfig()),
        waiting=waiting, decoding=decoding, prefilling=[],
        limits=ChunkLimits(), chunk_ceiling=2048, budget_ceiling=8192,
        max_num_seqs=8, chunk_enabled=True, kv_len=1024,
    )
    cands = []
    for c in d.candidates:
        if c.predicted_ms is None:
            cands.append({"chunk": c.chunk, "predicted_ms": None,
                          "budget_kind": c.budget_kind, "budget_ms": c.budget_ms,
                          "feasible": c.feasible})
            continue
        cands.append({
            "chunk": c.chunk, "predicted_ms": round(c.predicted_ms, 4),
            "tokens_prefilled": c.tokens_prefilled, "batch": c.batch,
            "budget_kind": c.budget_kind, "budget_ms": c.budget_ms,
            "feasible": c.feasible,
            # 同一候选若改按 **decode** 预算判，是否仍然可行？
            "feasible_if_decode_budget": bool(c.predicted_ms <= deadline_ms),
        })
    relaxed = [c for c in cands
               if c.get("predicted_ms") is not None
               and c["budget_kind"] == "prefill"
               and c["feasible"]
               and not c["feasible_if_decode_budget"]]
    kinds = sorted({c["budget_kind"] for c in cands if c.get("predicted_ms") is not None})
    return {
        "label": label,
        "n_waiting": len(waiting), "n_decoding": len(decoding),
        "chosen_chunk": d.chunk, "chosen_max_tokens": d.max_tokens,
        "degraded": bool(d.degraded),
        "n_candidates": len(d.candidates),
        "budget_kinds": kinds,
        # ★ 混合步上 > 0 ⇒ §5.1 的"受阻 decode 被忽略"仍然存在
        "n_relaxed_by_prefill_budget": len(relaxed),
        "relaxed_examples": relaxed[:5],
        "candidates": cands,
    }


def probe_dual_budget_gap(*, deadline_ms: float = 100.0,
                          prefill_deadline_ms: float = 2000.0) -> dict[str, Any]:
    """§5.1 双预算：**逐 token 护栏（decode）不得被 prefill 预算旁路**。

    ★ 三个场景一起报，缺一个都读不出结论（2026-10-03）：

    | 场景 | 应有行为 | 为什么需要它 |
    | --- | --- | --- |
    | **纯 prefill 步**（`decoding=[]`） | 允许按 prefill 预算放宽 | 这是**设计**（服务 TTFT）；没有它，"放宽"与"旁路"分不开 |
    | **混合步**（有 decode 在跑） | **一个都不许**按 prefill 预算放行 | ★ 这才是 §5.1 的靶心：这些步上有 decode token 被推迟，逐 token 护栏必须适用 |
    | **纯 decode 步** | 一律 decode 预算 | 对照，确认判据没反向出错 |

    ⚠️ 旧版只跑第一个场景，于是 `n_relaxed=6` 被当成缺口证据 ——
    而它其实是**按设计**的行为。缺口只在**混合步**上出现，
    修好后混合步的 `n_relaxed_by_prefill_budget` 必须是 **0**。
    """
    waiting = [Request(req_id=f"w{i:03d}", prompt_len=2048, gen_len=8) for i in range(12)]
    one_decode = [Request(req_id="d000", prompt_len=64, gen_len=64)]
    pure_pf = _dual_budget_case(waiting=waiting, decoding=[],
                               deadline_ms=deadline_ms,
                               prefill_deadline_ms=prefill_deadline_ms,
                               label="pure_prefill_step")
    mixed = _dual_budget_case(waiting=waiting, decoding=one_decode,
                              deadline_ms=deadline_ms,
                              prefill_deadline_ms=prefill_deadline_ms,
                              label="mixed_step")
    pure_dec = _dual_budget_case(waiting=[], decoding=one_decode,
                                 deadline_ms=deadline_ms,
                                 prefill_deadline_ms=prefill_deadline_ms,
                                 label="pure_decode_step")
    return {
        "step_deadline_ms": deadline_ms,
        "prefill_deadline_ms": prefill_deadline_ms,
        "pure_prefill_step": pure_pf,
        "mixed_step": mixed,
        "pure_decode_step": pure_dec,
        # 顶层保留旧字段（兼容既有读者），但语义已明确：这是**纯 prefill 步**的数
        "n_candidates": pure_pf["n_candidates"],
        "n_relaxed_by_prefill_budget": pure_pf["n_relaxed_by_prefill_budget"],
        "relaxed_examples": pure_pf["relaxed_examples"],
        # ★ **验收判据**（§5.1）：混合步上不得有候选被 prefill 预算旁路；
        #   纯 prefill 步**应当**有（按设计放宽），否则说明放宽本身失效了。
        "acceptance_pass": bool(
            mixed["n_relaxed_by_prefill_budget"] == 0
            and mixed["budget_kinds"] == ["decode"]
        ),
        "note": ("混合步按 decode 预算判 = 逐 token 护栏生效；"
                 "纯 prefill 步按 prefill 预算判 = 服务 TTFT（设计如此）。"
                 "本探针不修改任何 SLO 取值。"),
    }

def probe_closed_jittered_arrival(*, span_ms: float = JITTER_SPAN_MS,
                                  concurrency: int = 4) -> dict[str, Any]:
    """跑一次 `closed_jittered` 的 dry-run，数 `enqueue_ms < arrival_ms` 的倒置。"""
    cfg = load_config(BASELINE)
    base = get_workload("long_context_qa")          # arrival="closed"
    spec = dataclasses.replace(base, arrival="closed_jittered",
                               closure_jitter_ms=span_ms, concurrency=concurrency,
                               num_requests=16)
    result = simulate(cfg, build_requests(spec), spec=spec)
    rows, n_inverted, max_inv = [], 0, 0.0
    span_lo, span_hi = None, None
    for m in result.metrics:
        arr, enq = m.arrival_ms, m.enqueue_ms
        if arr is None or enq is None:
            continue
        inv = arr - enq
        if inv > 1e-9:
            n_inverted += 1
            max_inv = max(max_inv, inv)
        rows.append({
            "req_id": m.req_id,
            "scheduled_arrival_ms": m.scheduled_arrival_ms,
            "arrival_ms": arr,
            "enqueue_ms": enq,
            "first_schedule_ms": m.first_schedule_ms,
            "inversion_ms": round(inv, 4),
        })
        lo = arr
        span_lo = lo if span_lo is None else min(span_lo, lo)
        span_hi = lo if span_hi is None else max(span_hi, lo)
    arrivals = [r["arrival_ms"] for r in rows]
    delays = [r["enqueue_ms"] - r["arrival_ms"] for r in rows]
    return {
        "arrival_kind": "closed_jittered",
        "closure_jitter_ms": span_ms,
        "concurrency": concurrency,
        "n_requests": len(rows),
        # ★ > 0 ⇒ "入队早于到达"（禁止的状态；§5.2 要求"实际等待后入队"）
        #   修好后**必须是 0** —— 这是 §5.2 的**验收判据**。
        "n_enqueue_before_arrival": n_inverted,
        "max_inversion_ms": round(max_inv, 4),
        "arrival_span_ms": round(max(arrivals) - min(arrivals), 4) if arrivals else 0.0,
        "scheduled_equals_arrival": all(
            abs((r["scheduled_arrival_ms"] or 0.0) - r["arrival_ms"]) < 1e-6 for r in rows),
        # ★ **反向守卫**：不许靠"把抖动抹平成 0"来消除倒置。
        #   `n_distinct_arrivals` 必须 == n_requests；span 必须 > 0。
        "n_distinct_arrivals": len({round(x, 6) for x in arrivals}),
        # ★ 受理延迟（步粒度）必须 ≥ 0：它是 `arrival -> enqueue` 的**合法**间隔，
        #   而倒置（负值）才是禁止的。
        "min_enqueue_minus_arrival_ms": round(min(delays), 4) if delays else None,
        #: ★ 验收（三条同时成立才算 §5.2 通过）
        "acceptance_pass": bool(
            n_inverted == 0 and rows
            and len({round(x, 6) for x in arrivals}) == len(rows)
            and (min(delays) >= -1e-9 if delays else False)
        ),
        "sample_rows": rows[:6],
    }


# --------------------------------------------------------------------------- #
# §5.3（复核）SLOWeave 修复后状态
# --------------------------------------------------------------------------- #

def probe_slo_weave_fixed() -> dict[str, Any]:
    from ascend_opt.slo_weave import SLOWeaveConfig, SLOWeavePolicy
    m = CostModel.load_offline(MODEL_V7)
    sm = float(m.payload["safety_margin"])
    p = SLOWeavePolicy(SLOWeaveConfig(enabled=True, c_max=4096), m)
    raw = m.predict(batch=1, prefill_tok=256, kv_len=1024, apply_margin=False).total_ms
    t = p._predict_ms(1, 256, 1024)
    pred_ratio = (t + p._margin_ms(t)) / raw
    cand = m.lookup_prefill_candidates(batch=1, kv_len=1024, cap=4096)
    d = p.decide(now_ms=0.0, deadlines=[400.0], pending_prefill=[("r", 1024)], kv_len=1024)
    d_short = p.decide(now_ms=0.0, deadlines=[320.0], pending_prefill=[("r", 1536)],
                       kv_len=1024)
    return {
        "safety_margin": sm,
        "predicate_ratio": round(pred_ratio, 9),
        "predicate_ratio_expected": round(1.0 + sm, 9),
        "margin_applied_once": abs(pred_ratio - (1.0 + sm)) < 1e-9,
        "lookup_candidates_b1_kv1024": list(cand),
        "decide_budget400_L1024_chunk": d.chunk,          # 旧逻辑=512（欠选）
        "decide_budget320_L1536_chunk": d_short.chunk,    # 旧逻辑=0（塌缩）
        "decode_only_collapse_gone": d_short.chunk == 1024,
    }


# --------------------------------------------------------------------------- #
# §5（模型导出）v8 采样表 vs 部署模型
# --------------------------------------------------------------------------- #

def probe_model_export() -> dict[str, Any]:
    d = _ROOT / "configs" / "cost_calib"
    models = sorted(x.name for x in d.glob("model_v*.json"))
    grids = sorted(x.name for x in d.glob("shape_grid_v*.yaml"))
    payload = json.loads(MODEL_V7.read_text(encoding="utf-8"))
    tbl = payload.get("lookup_table") or {}
    pfs: dict[int, int] = {}
    for k in tbl:
        parts = k.split("_")
        if len(parts) == 3 and parts[1].startswith("pf"):
            pfs[int(parts[1][2:])] = pfs.get(int(parts[1][2:]), 0) + 1
    return {
        "model_files": models,
        "grid_files": grids,
        "model_v8_json_exists": (d / "model_v8.json").exists(),
        "grid_v8_exists_not_model": (d / "shape_grid_v8.yaml").exists()
        and not (d / "model_v8.json").exists(),
        "deployed_model": "model_v7.json",
        "deployed_default_mode": payload.get("default_mode"),
        "deployed_protocol_version": payload.get("protocol_version"),
        "deployed_grid_sha256": payload.get("grid_sha256"),
        "deployed_lookup_keys": len(tbl),
        "deployed_pf_histogram": dict(sorted(pfs.items())),
        "pf192_in_deployed_table": 192 in pfs,
        "note": ("v8 只在探针层（_probe/_costcalib 的 24 点，protocol cost-calib-v8）"
                 "有数据；**未导出**为部署模型 ⇒ 192 不在部署查表内。"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="S00 阻断项证据（只读）")
    ap.add_argument("--json", type=Path, default=None, help="把结果写到该 JSON 路径")
    args = ap.parse_args()

    out: dict[str, Any] = {
        "schema": "s00-audit-v1",
        "generated_by": "scripts/audit_s00_blockers.py",
        "git_head": _git_head(),
        "items": {},
    }
    rc = 0
    out["items"]["dual_budget_gap"] = probe_dual_budget_gap()
    try:
        out["items"]["closed_jittered_arrival"] = probe_closed_jittered_arrival()
    except Exception as exc:                                    # noqa: BLE001
        out["items"]["closed_jittered_arrival"] = {"error": repr(exc)}
        rc = 2
    out["items"]["slo_weave_fixed"] = probe_slo_weave_fixed()
    out["items"]["model_export"] = probe_model_export()

    summary = {
        # ---- §5.1 双预算（2026-10-03 修）----
        # 顶层那一项仍是"纯 prefill 步"的数（**按设计**放宽，非缺口）；
        # 判缺口的那个数是 `mixed_step.n_relaxed_by_prefill_budget`，修好后为 0。
        "dual_budget.relaxed_by_prefill_budget(PURE_PREFILL, BY DESIGN)":
            out["items"]["dual_budget_gap"]["n_relaxed_by_prefill_budget"],
        "dual_budget.mixed_step.relaxed(MUST_BE_0)":
            out["items"]["dual_budget_gap"]["mixed_step"]["n_relaxed_by_prefill_budget"],
        "dual_budget.mixed_step.budget_kinds(MUST_BE_decode)":
            out["items"]["dual_budget_gap"]["mixed_step"]["budget_kinds"],
        "dual_budget.acceptance_pass":
            out["items"]["dual_budget_gap"]["acceptance_pass"],
        # ---- §5.2 闭环抖动（2026-10-03 修）----
        "jitter.enqueue_before_arrival(MUST_BE_0)":
            out["items"]["closed_jittered_arrival"].get("n_enqueue_before_arrival"),
        "jitter.arrival_span_ms(MUST_BE_>0)":
            out["items"]["closed_jittered_arrival"].get("arrival_span_ms"),
        "jitter.n_distinct_arrivals(MUST_BE_n_requests)":
            out["items"]["closed_jittered_arrival"].get("n_distinct_arrivals"),
        "jitter.acceptance_pass":
            out["items"]["closed_jittered_arrival"].get("acceptance_pass"),
        "slo_weave.margin_once": out["items"]["slo_weave_fixed"]["margin_applied_once"],
        "slo_weave.collapse_gone":
            out["items"]["slo_weave_fixed"]["decode_only_collapse_gone"],
        "model.pf192_in_deployed": out["items"]["model_export"]["pf192_in_deployed_table"],
    }
    out["summary"] = summary

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"写出 {args.json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return rc


def _git_head() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_ROOT),
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                           # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
