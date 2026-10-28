#!/usr/bin/env python
"""**离线**把代表状态接到**真实决策链**：`load_config → build_arm → plan_step / decide`。

★ 与 `enumerate_prereg_states.py` 的区别（**这一点决定本产物的价值**）
--------------------------------------------------------------------------
`enumerate_prereg_states.py` 是**预算函数示例**：它**直接把**
`(batch, prefill_tok, kv_len)` 喂给 `CostModel.predict`，再**自己**调
`CostPolicy._budget_for` 问归类。它证明的是"**预算函数在给定输入下怎么算**"，
**不能**证明"控制器在真实队列状态下会怎么选" —— 因为那个 `(batch, pf, kv)` 是
**人手选的**，不是调度器算出来的。

本脚本走**真链路**：

    load_config(真实派生 YAML)
        → build_arm(cfg)                  # 造出**真实的** CostPolicy（含模型/候选/host 开销）
        → 用校验过的独立合成队列状态（不声称生产路径可达）
        → plan_step(...)                  # 调度器 + chunk_policy 算出 (batch, pf, kv)
        → CostPolicy.decide(...)          # 控制器按真实 plan 选动作

⇒ 它记录的是**真实决策过程**：生成候选、逐候选预测、预算分支、所选动作、回退原因。
⇒ 它**不要求**两臂结论必须不同（那会把"没差异"当成失败）；要求的是**过程真实、结果如实**。

⚠️ 全部产物是 **TEST_DATA**（离线演练，`CostModel.load_offline()` ⇒ **未做指纹校验**，
记进产物 `relaxations`）。**不代替真机标定**，**不得**据此宣布双预算有效/无效
（采用 FZ-1 A / FZ-2 A 当前实现；不证明此语义最优）。

用法::

    python scripts/dryplan_decision_chain.py                     # 两臂并排（cost_single / cost_dual）
    python scripts/dryplan_decision_chain.py --json results/processed/dryplan_execution_shape_20261003.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "benchmark"))

from ascend_opt.cache_policy import PrefixCachePolicy   # noqa: E402
from ascend_opt.chunk_policy import (ChunkConfig, ChunkLimits,  # noqa: E402
                                     FixedChunkPolicy)
from ascend_opt.cost_model import CostModel             # noqa: E402
from ascend_opt.execution_shape import execution_shape_key, execution_signature  # noqa: E402
from ascend_opt.scheduler import ContinuousBatchingScheduler  # noqa: E402
from ascend_opt.workload import Request                 # noqa: E402

from arm_factory import build_arm                       # noqa: E402
from run_benchmark import load_config                   # noqa: E402

CFG_DIR = _ROOT / "configs" / "mthreads" / "prereg_batch"
DEFAULT_ARMS = ("cost_single", "cost_dual")

#: 本批计划的 cell（`NEXT_BATCHES_PREREG.md` §5.2）：prompt 1536 / gen 32 / conc 16。
#: ⚠️ 这些数与 exec_opt 的 `short_chat` **不是同一组** ⇒ 不得照搬 162.2 s/项。
PLAN = {"prompt_len": 1536, "gen_len": 32, "concurrency": 16, "max_num_seqs": 8}

#: Independent TEST_DATA fixtures. Legal partitions do not establish that this
#: state occurs in the deployed workload; no production replay is available here.
STATES: list[dict] = [
    {
        "name": "batch_start_prefill",
        "desc": "批起步：队列里全是等待起步的新请求，还没有 decode 在跑"
                "（双预算 prefill 分支的一个合成场景）",
        "fixture_reason": "全等待边界；没有用生产到达/完成事件重放验证",
        "arrival_upto": 8,          # 前 8 个已到达（= max_num_seqs）
        "decoding": 0,
        "prefilling": [],
    },
    {
        "name": "steady_decode",
        "desc": "稳态纯 decode：队列空、全部在解码",
        "fixture_reason": "满额 decode 边界；concurrency 不构成可达性证据",
        "arrival_upto": 8,
        "decoding": 8,
        "prefilling": [],
    },
    {
        "name": "mixed_admit",
        "desc": "混合步：既有 decode 在跑、又有新请求要起步"
                "（**判决点**：按 §5.1 修正此处必须走 decode 预算而非 prefill）",
        "fixture_reason": "满额 decode 加 waiting；是否准入由真实 plan_step 决定",
        "arrival_upto": 16,
        "decoding": 8,
        "prefilling": [],
    },
    {
        "name": "contended_prefill",
        "desc": "争抢步：已开工的请求**没跑完**、新请求又要进来"
                "（prefilling 非空 ⇒ 续传优先，且序列名额被占）",
        "fixture_reason": "4 条 decode 加 2 条部分 prefill；独立构造，不声称负载必然产生",
        "arrival_upto": 16,
        "decoding": 4,
        "prefilling": [4, 5],       # 已开工的请求下标（剩余量由 _PF_DONE 派生）
    },
]

#: 队列里预留多少请求（模拟"后 8 个已到达但还没被接纳"）。
POOL = 16


def _make_requests() -> list[Request]:
    """按计划 cell 造 POOL 个请求。**全部同一形状**（本 cell 是固定长度负载）。"""
    return [Request(req_id=f"r{i:02d}", prompt_len=PLAN["prompt_len"],
                    gen_len=PLAN["gen_len"], arrival_ms=float(i))
            for i in range(POOL)]


def _queue_for(st: dict, reqs: list[Request]) -> tuple[list, list, list[tuple]]:
    """Compatibility tuple backed by the validated snapshot constructor."""
    s = _snapshot_for(st, reqs, max_num_seqs=PLAN["max_num_seqs"])
    return s["waiting"], s["decoding"], s["prefilling"]


def _snapshot_for(st: dict, reqs: list[Request], *, max_num_seqs: int) -> dict:
    """Construct one self-consistent fixture; invalid states fail before decide.

    Decode KV contains the full prompt plus generated tokens except the most
    recently sampled token, which enters the next forward. Completed requests
    remain in the arrived partition but consume no active sequence slots.
    """
    if len({r.req_id for r in reqs}) != len(reqs):
        raise ValueError("duplicate request IDs")
    upto = int(st["arrival_upto"])
    if not 0 <= upto <= len(reqs):
        raise ValueError("arrival_upto outside request pool")
    now_ms = float(st.get("now_ms", reqs[upto - 1].arrival_ms if upto else -1))
    arrived = [r for r in reqs if r.arrival_ms <= now_ms]
    if {r.req_id for r in arrived} != {r.req_id for r in reqs[:upto]}:
        raise ValueError("arrived set disagrees with arrival timestamps")
    indices = {
        "decoding": list(st.get("decoding_indices", range(int(st["decoding"])))),
        "prefilling": list(st["prefilling"]),
        "completed": list(st.get("completed", [])),
    }
    occupied = [i for xs in indices.values() for i in xs]
    if len(set(occupied)) != len(occupied):
        raise ValueError("request phases overlap or contain duplicate IDs")
    if any(not 0 <= i < upto for i in occupied):
        raise ValueError("active/completed request has not arrived")
    if len(indices["decoding"]) + len(indices["prefilling"]) > max_num_seqs:
        raise ValueError("active sequences exceed max_num_seqs")
    waiting = [r for i, r in enumerate(reqs[:upto]) if i not in set(occupied)]
    decoding = [reqs[i] for i in indices["decoding"]]
    completed = [reqs[i] for i in indices["completed"]]
    remaining = st.get("prefill_remaining", {})
    pf = [(reqs[i], int(remaining.get(i, reqs[i].prompt_len - _PF_DONE)))
          for i in indices["prefilling"]]
    if any(not 0 < rem < r.prompt_len for r, rem in pf):
        raise ValueError("partial prefill remaining outside (0,prompt_len)")
    generated = {r.req_id: int(st.get("generated_tokens", {}).get(r.req_id, 4))
                 for r in decoding}
    if any(not 1 <= generated[r.req_id] < r.gen_len for r in decoding):
        raise ValueError("decode generated tokens outside [1,gen_len)")
    progress = {r.req_id: {"phase": "waiting", "prefill_remaining": r.prompt_len,
                           "generated_tokens": 0, "kv_len": 0} for r in waiting}
    progress.update({r.req_id: {"phase": "prefilling", "prefill_remaining": rem,
                               "generated_tokens": 0, "kv_len": r.prompt_len - rem}
                     for r, rem in pf})
    progress.update({r.req_id: {"phase": "decoding", "prefill_remaining": 0,
                               "generated_tokens": generated[r.req_id],
                               "kv_len": r.prompt_len + generated[r.req_id] - 1}
                     for r in decoding})
    progress.update({r.req_id: {"phase": "completed", "prefill_remaining": 0,
                               "generated_tokens": r.gen_len, "kv_len": None}
                     for r in completed})
    return {"waiting": waiting, "decoding": decoding, "prefilling": pf,
            "completed": completed, "arrived": arrived, "now_ms": now_ms,
            "progress": progress,
            "kv_lengths": {rid: p["kv_len"] for rid, p in progress.items()
                           if p["kv_len"] is not None}}


#: prefill 中的请求"已经算过多少 token"（用于派生剩余量）。
#: 取 `prompt_len` 的一半 ⇒ `remaining = 768`，只是假设的部分进度。
_PF_DONE = PLAN["prompt_len"] // 2


def _decision_bounds(cfg) -> tuple[int, int]:
    """从**真实配置**取 chunk / budget 上限（运行期上限的唯一权威仍是 ChunkLimits）。"""
    c = cfg.controller
    return int(c.max_batched_tokens_max), int(c.max_batched_tokens_max)


def _validate_plan_cell(cfg) -> None:
    """This fixture family is scoped to the declared cell; fail on config drift."""
    workload = cfg.raw.get("workload") or {}
    expected = {"prompt_len_min": PLAN["prompt_len"], "prompt_len_max": PLAN["prompt_len"],
                "gen_len_min": PLAN["gen_len"], "gen_len_max": PLAN["gen_len"],
                "concurrency": PLAN["concurrency"]}
    mismatches = {key: (workload.get(key), value) for key, value in expected.items()
                  if workload.get(key) != value}
    if mismatches:
        raise ValueError(f"fixture cell differs from config workload: {mismatches}")


def run_arm(cfg_path: Path, *, state_names: list[str] | None = None) -> dict:
    """把一条臂跑完所有代表状态。**走真实决策链**。

    ★ 指纹来源：派生配置写的是 `observed`（真机才可校验）。**离线**没有真实运行环境
    ⇒ 本脚本把它**临时改成 `declared`**，让 `build_arm` 走 `CostModel.load_offline()`
    （**明说"没做指纹校验"**并记进产物 `relaxations`）。
    ⚠️ 这不是"绕过校验"：离线**本来就不可能**做指纹校验；假装校验过才是问题。
    真机路径仍用 `observed` ⇒ 指纹不符时**直接抛**。
    """
    cfg = load_config(cfg_path)
    _validate_plan_cell(cfg)
    declared = cfg.controller.cost_model_fingerprint_source
    if declared == "observed":
        # `ControllerConfig` 是 **frozen** ⇒ 不能原地赋值，必须 replace 出新对象。
        cfg = replace(cfg, controller=replace(cfg.controller,
                                              cost_model_fingerprint_source="declared"))
    arm = build_arm(cfg)
    if arm is None:
        return {"config": str(cfg_path), "arm": None,
                "error": "build_arm 返回 None（controller.enabled=False 或 kind=threshold）"
                         " ⇒ 该臂没有成本决策链，不适用本演练"}

    reqs = _make_requests()
    # ★ chunk policy 也从**真实配置**取（`chunk.policy` / `chunk.chunk_size`），
    #   不是脚本自己编一个 —— 否则"真实决策链"就断在这里了。
    ck = cfg.raw.get("chunk") or {}
    chunk_policy = FixedChunkPolicy(ChunkConfig(
        enabled=bool(ck.get("enabled", True)),
        chunk_size=int(ck.get("chunk_size", cfg.controller.chunk_size_init)),
        max_num_batched_tokens=int(cfg.controller.max_batched_tokens_max),
    ))
    limits = ChunkLimits(chunk_size=cfg.controller.chunk_size_max,
                         max_num_batched_tokens=cfg.controller.max_batched_tokens_max)
    chunk_ceiling, budget_ceiling = _decision_bounds(cfg)
    max_num_seqs = (cfg.raw.get("scheduler") or {}).get("max_num_seqs") or cfg.max_num_seqs
    chunk_enabled = bool(cfg.raw.get("chunked_prefill", True))

    rows: list[dict] = []
    for st in STATES:
        if state_names and st["name"] not in state_names:
            continue
        # Independent fixtures must not inherit cooldown, actions or cache state.
        arm = build_arm(cfg)
        sched = ContinuousBatchingScheduler()
        cache = PrefixCachePolicy()
        snapshot = _snapshot_for(st, reqs, max_num_seqs=max_num_seqs)
        waiting, decoding, prefilling = (snapshot[k] for k in
                                         ("waiting", "decoding", "prefilling"))
        # ★ 真实调用：控制器内部会对**每个候选**跑一遍 plan_step 再 predict
        dec = arm.decide(
            scheduler=sched, chunk_policy=chunk_policy, cache=cache,
            waiting=waiting, decoding=decoding, prefilling=prefilling,
            limits=limits, chunk_ceiling=chunk_ceiling, budget_ceiling=budget_ceiling,
            max_num_seqs=max_num_seqs, chunk_enabled=chunk_enabled,
            kv_len=_kv_len_from(decoding, snapshot["progress"]),
            per_request_kv_lengths=snapshot["kv_lengths"], step=0,
        )
        evals = list(dec.candidates or [])
        rows.append({
            "state": st["name"],
            "desc": st["desc"],
            "evidence_class": "TEST_DATA",
            "state_source": "independent_fixture",
            "reachable_from_production": None,
            "fixture_reason": st["fixture_reason"],
            "initialization": "fresh arm, scheduler and cache for each fixture",
            "now_ms": snapshot["now_ms"],
            "queue": {"n_waiting": len(waiting), "n_decoding": len(decoding),
                      "n_prefilling": len(prefilling),
                      "prefilling_remaining": [rem for _r, rem in prefilling]},
            "queued_ids": [r.req_id for r in waiting],
            "decoding_ids": [r.req_id for r in decoding],
            "prefilling_ids": [r.req_id for r, _ in prefilling],
            "completed_ids": [r.req_id for r in snapshot["completed"]],
            "arrived_ids": [r.req_id for r in snapshot["arrived"]],
            "request_progress": snapshot["progress"],
            "n_candidates_generated": len(evals),
            "candidates": [_eval_row(e) for e in evals],
            "action_taken": bool(dec.changed),
            "chosen": (None if not dec.changed else
                       {"chunk": dec.chunk, "max_tokens": dec.max_tokens}),
            "action_reason": dec.reason,
            "fallback_reason": dec.fallback_reason,
            "n_predicted": sum(1 for e in evals if e.predicted_ms is not None),
            "n_feasible": sum(1 for e in evals if e.feasible),
            "prediction_unavailable_all": (bool(evals) and
                                all(e.predicted_ms is None for e in evals)),
        })
    return {
        "config": str(cfg_path),
        "fingerprint_source_declared": declared,
        "fingerprint_source_used": ("declared(离线，未校验)" if declared == "observed"
                                    else declared),
        "model_relaxations": list(getattr(arm.model, "relaxations", []) or []),
        "controller": {"kind": cfg.controller.kind, "enabled": bool(cfg.controller.enabled),
                       "cost_model_path": getattr(cfg.controller, "cost_model_path", None),
                       "step_deadline_ms": arm.config.step_deadline_ms,
                       "prefill_deadline_ms": arm.config.prefill_deadline_ms,
                       "chunk_size_max": cfg.controller.chunk_size_max,
                       "max_batched_tokens_max": cfg.controller.max_batched_tokens_max},
        "max_num_seqs": max_num_seqs,
        "chunk_enabled": chunk_enabled,
        "execution_semantics": {
            "S01": {key: cfg.raw.get("runtime", {}).get(key) for key in
                    ("logits_last_only", "sample_needed_only")},
            "model_revision": cfg.raw.get("runtime", {}).get("model_revision"),
            "precision": cfg.raw.get("runtime", {}).get("precision"),
            "model_file_sha256": hashlib.sha256(
                (_ROOT / cfg.controller.cost_model_path).read_bytes()).hexdigest(),
            "grouping": "production execution-shape-v1; groups submitted in order",
        },
        "execution_signature": execution_signature(cfg.runtime or {}),
        "states": rows,
    }


def _kv_len_from(decoding: list, progress: dict) -> int:
    """Legacy representative KV; per-request values are retained separately."""
    if not decoding:
        return 0
    return max(int(progress[r.req_id]["kv_len"]) for r in decoding)


def _eval_row(e) -> dict:
    """一个候选的完整记录：**预测 / 预算分支 / 可行性 / 拒绝原因**。"""
    return {
        **e.describe(),
        "chunk": e.chunk,
        "max_tokens": e.max_tokens,
        "batch": e.batch,
        "tokens_prefilled": e.tokens_prefilled,
        "kv_len": e.kv_len,
        "predicted_forward_ms": (None if e.predicted_ms is None else round(e.predicted_ms, 2)),
        "budget_kind": e.budget_kind,
        "budget_ms": e.budget_ms,
        "feasible": bool(e.feasible),
        "reject_reason": (e.reject_reason or None),
    }


def _print_report(res: dict) -> None:
    if res.get("arm") is None and res.get("error"):
        print(f"\n[{Path(res['config']).name}] ⚠️ {res['error']}")
        return
    cfgname = Path(res["config"]).name
    c = res["controller"]
    pf_ms = c["prefill_deadline_ms"]
    pf_s = "未启用(None)" if pf_ms is None else f"{pf_ms:g} ms"
    print()
    print("=" * 108)
    print(f"臂 {cfgname}   kind={c['kind']}  enabled={c['enabled']}  "
          f"内部步预算={c['step_deadline_ms']:g} ms  prefill 预算={pf_s}")
    print(f"    成本模型={Path(c['cost_model_path'] or '—').name}  "
          f"候选上限 chunk≤{c['chunk_size_max']} / tokens≤{c['max_batched_tokens_max']}  "
          f"max_num_seqs={res['max_num_seqs']}  chunk_enabled={res['chunk_enabled']}")
    print("=" * 108)
    for r in res["states"]:
        q = r["queue"]
        print()
        print(f"── 状态 {r['state']} ──")
        print(f"   合成边界   : {r['fixture_reason']}；生产可达性未验证")
        print(f"   校验后队列 : waiting={q['n_waiting']} decoding={q['n_decoding']} "
              f"prefilling={q['n_prefilling']}{q['prefilling_remaining'] or ''}")
        print(f"   生成候选   : {r['n_candidates_generated']} 个"
              f"（已预测 {r['n_predicted']} / 可行 {r['n_feasible']}）")
        for e in r["candidates"]:
            pred = "—" if e["predicted_forward_ms"] is None else f"{e['predicted_forward_ms']:g}"
            bk = e["budget_kind"] or "—"
            bms = "—" if e["budget_ms"] is None else f"{e['budget_ms']:g}"
            flag = ("不可判" if e["predicted_ms"] is None
                    else "可行" if e["feasible"] else "不可行")
            print(f"     · chunk={e['chunk']:<6} max_tok={e['max_tokens']:<6} "
                  f"batch={e['batch']:<3} pf={e['tokens_prefilled']:<6} "
                  f"pred={pred:>8} 预算={bk}/{bms:<7} {flag}"
                  f"{'' if e['feasible'] else '  ← ' + (e['reject_reason'] or '')}")
        act = ("选取 " + json.dumps(r["chosen"], ensure_ascii=False)) if r["action_taken"] else "**未动作**"
        print(f"   所选动作   : {act}")
        print(f"   动作理由   : {r['action_reason']}")
        if r["fallback_reason"]:
            print(f"   回退原因   : {r['fallback_reason']}")


def _shape_without_ids(shape: dict) -> list[dict]:
    """Whole-step key: preserve ordering, lengths, roles and sampling mask."""
    return [{"n_new_tokens": group["n_new_tokens"], "batch": group["batch"],
             "past_max": group["past_max"],
             "items": [{k: item[k] for k in
                        ("n_new_tokens", "kv_len", "role", "need_token")}
                       for item in group["items"]]}
            for group in shape["groups"]]


def _coverage_gap_analysis(results: list[dict]) -> dict:
    """Derive a measurement manifest from actual candidate executions.

    Legacy scalar lookup diagnostics are not semantic coverage. The old probe
    uses one prefill row, clamps KV to one, and does not enable S01. No inference
    about accuracy or a fixed Cartesian sampling grid follows from those keys.
    """
    unique: dict[str, dict] = {}
    corrections: list[dict] = []
    for arm in results:
        if arm.get("error"):
            continue
        model_path = _ROOT / arm["controller"]["cost_model_path"]
        model = CostModel.load_offline(model_path)
        semantics = arm["execution_semantics"]
        for state in arm["states"]:
            for candidate_index, candidate in enumerate(state["candidates"]):
                shape = candidate["execution_shape"]
                groups = _shape_without_ids(shape)
                source = {"arm": arm["arm_name"], "config": arm["config"],
                          "state": state["state"], "candidate_index": candidate_index,
                          "chunk": candidate["chunk"], "max_tokens": candidate["max_tokens"],
                          "state_source": state["state_source"],
                          "reachable_from_production": state["reachable_from_production"]}
                legacy_batch = candidate["legacy_batch"]
                if legacy_batch != shape["batch"]:
                    corrections.append({**source, "category": "input_bug",
                                        "legacy_batch": legacy_batch,
                                        "actual_items": shape["batch"],
                                        "execution_groups": groups})
                if not groups:
                    continue
                # Diagnostic only: show what the superseded scalar input did.
                old_shape = {"batch": legacy_batch,
                             "prefill_tok": candidate["tokens_prefilled"],
                             "kv_len": candidate["kv_len"]}
                old_hit = model.predict(**old_shape, mode="lookup") is not None
                old_in_range = model.in_range(**old_shape)
                legacy_diagnostic = {
                    "input": old_shape, "key_hit": old_hit, "in_range": old_in_range,
                    "category": ("anchor_revalidation" if old_hit else
                                 "key_miss" if old_in_range else "range_miss"),
                    "scope": "superseded scalar input only; not actual-shape coverage",
                }
                key = json.dumps({"shape_key": execution_shape_key(shape),
                                  "signature": arm["execution_signature"]},
                                 sort_keys=True, separators=(",", ":"))
                if key not in unique:
                    point_id = "shape-" + hashlib.sha256(key.encode()).hexdigest()[:16]
                    unique[key] = {
                        "point_id": point_id, "evidence_class": "NOT_MEASURED",
                        "origin_evidence_class": "TEST_DATA",
                        "config_path": Path(arm["config"]).relative_to(_ROOT).as_posix(),
                        "execution_signature": arm["execution_signature"],
                        "execution_shape": shape,
                        "execution_shape_key": execution_shape_key(shape),
                        "model_revision": semantics["model_revision"],
                        "prime_requests": {r.req_id: {"prompt_len": r.prompt_len, "gen_len": r.gen_len}
                                           for r in _make_requests()
                                           if r.req_id in {item["req_id"] for item in shape["items"]}},
                        "repeats": 3, "validation_repeats": 3, "warmup_s": 30,
                        "execution_groups": groups, "n_forwards": len(groups),
                        "execution_semantics": semantics,
                        "sources": [], "legacy_scalar_diagnostics": [],
                        "production_coverage": [],
                        "measurement_status": "HARDWARE_REQUIRED",
                        "measurement_unit": "one complete ordered step including all groups",
                        "timing": {
                            "forward": "synchronized device forward sum observed within the actual step",
                            "host": "driver logical forward elapsed_ms - fwd_ms; pack/readback retained as diagnostics",
                            "step": "record end-to-end step; never multiply scalar forward by group count",
                        },
                        "warmup": "at least 30 seconds outside measurement, retain stability diagnostic",
                        "training": {"independent_processes": 3, "phase": "calibration",
                                     "records_per_shape_per_process": 30},
                        "independent_validation": {
                            "independent_processes": 3, "phase": "held_out_validation",
                            "records_per_shape_per_process": 30,
                            "rule": "freeze estimator after training; validation records never enter fit/table",
                        },
                        "old_anchor_rule": "same identity and execution semantics required before pooling; "
                                           "otherwise keep old data separate",
                    }
                entry = unique[key]
                entry["sources"].append(source)
                if legacy_diagnostic not in entry["legacy_scalar_diagnostics"]:
                    entry["legacy_scalar_diagnostics"].append(legacy_diagnostic)
                coverage = {"status": candidate["coverage_status"],
                            "reason": candidate["coverage_reason"]}
                if coverage not in entry["production_coverage"]:
                    entry["production_coverage"].append(coverage)
    entries = sorted(unique.values(), key=lambda row: row["point_id"])
    for entry in entries:
        statuses = {coverage["status"] for coverage in entry["production_coverage"]}
        entry["requires_new_calibration"] = statuses != {"exact_execution_key"}
        entry["sampling_purpose"] = ("new_execution_semantics" if statuses & {
            "execution_contract_unvalidated", "execution_signature_mismatch",
            "timing_contract_mismatch"} else "missing_execution_key" if
            "missing_execution_key" in statuses else "optional_identity_anchor" if
            statuses == {"exact_execution_key"} else "resolve_invalid_or_unknown_input_first")
    return {
        "schema": "execution_shape_manifest_v1",
        "origin": "actual_candidate_execution_groups",
        "evidence_class": "TEST_DATA",
        "reachable_workload_coverage": "NOT_VERIFIED: independent legal fixtures, not a production replay",
        "unique_execution_shapes": len(entries),
        "input_bug_corrections": corrections,
        "entries": entries,
        "calibrated_grid_from_raw": _grid_from_raw(),
        "categories": {
            "input_bug": "legacy admitted-count differs from actual executing items",
            "key_miss": "legacy scalar key absent inside its calibrated range; diagnostic only",
            "range_miss": "legacy scalar input outside its calibrated range; diagnostic only",
            "anchor_revalidation": "legacy scalar key exists but semantic/identity revalidation is required",
            "semantic_mismatch": "legacy real_mixed probe/S01/KV treatment cannot establish actual-step coverage",
        },
        "required_before_sampling": [
            "Use production driver with exact per-request vectors and S01 flags from this manifest",
            "Old fwd_costcalib_probe is not a compatible consumer: it forces KV>=1, one prefill row and no S01",
            "Verify source/environment/model identity and short numerical acceptance first",
            "Measure whole ordered steps; do not assume costs of separate groups are additive",
            "This fixture-derived list is diagnostic; replay/trace is required to claim workload coverage",
        ],
        "consequence_for_expansion": "First verify semantics and necessary coverage; "
                                     "no fixed nine-point grid and no automatic performance-matrix expansion",
        "duration_estimate": None,
        "duration_note": "No hardware timing for this protocol; do not borrow another workload's seconds/run",
    }


def _grid_from_raw() -> dict:
    """从**原始标定产物** `results/raw/_costcalib/*.json` 读真实网格。

    ⚠️ 血缘字段的权威来源是**原始产物**，不在 `costcalib_eval_*`、也不在模型文件。
    这里只读不写（`results/raw/**` 只读）。
    """
    raw_dir = _ROOT / "results" / "raw" / "_costcalib"
    pts: list[dict] = []
    used: list[str] = []
    if raw_dir.is_dir():
        for p in sorted(raw_dir.glob("*v7*.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            pts += [q for q in (d.get("points") or []) if isinstance(q, dict)]
            used.append(p.name)
    if not pts:
        return {"available": False, "note": "本机无 v7 原始标定产物（只读路径）"}
    kv0_batches = sorted({q["shape"]["batch"] for q in pts
                          if q["shape"].get("kv_len") == 0})
    b2plus_kv = sorted({q["shape"]["kv_len"] for q in pts
                        if (q["shape"].get("batch") or 0) >= 2})
    return {
        "available": True,
        "source_files": used,
        "n_points": len(pts),
        "kv_len_0_batches": kv0_batches,
        "batch_ge2_kv_min": (b2plus_kv[0] if b2plus_kv else None),
        "kv_grid": sorted({q["shape"]["kv_len"] for q in pts}),
        "batch_grid": sorted({q["shape"]["batch"] for q in pts}),
        "pf_grid": sorted({q["shape"]["prefill_tok"] for q in pts}),
        "the_hole": f"kv_len=0 只覆盖 batch={kv0_batches}；"
                    f"batch≥2 的 kv 维从 {b2plus_kv[0] if b2plus_kv else '?'} 起 "
                    f"⇒ (batch≥2, kv=0) **无任何点**",
    }


def _write_json(path: str, payload: dict) -> None:
    out = Path(path)
    if out.resolve() == (_ROOT / "results/processed/dryplan_decision_chain.json").resolve():
        raise ValueError("legacy dryplan output is reserved; use a new versioned path")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"写出 {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS))
    ap.add_argument("--cfg-dir", type=Path, default=CFG_DIR,
                    help="派生配置目录，位于当前仓库内；可用于新模型的离线重放")
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--manifest-json", default=None,
                    help="write production-shape measurement manifest (not measured data)")
    args = ap.parse_args()
    cfg_dir = (args.cfg_dir if args.cfg_dir.is_absolute() else _ROOT / args.cfg_dir).resolve()
    if not cfg_dir.is_relative_to(_ROOT.resolve()):
        ap.error("cfg-dir must be inside the repository")
    names = {state["name"] for state in STATES}
    if args.states and set(args.states) - names:
        ap.error(f"unknown states: {sorted(set(args.states) - names)}")
    print("load_config → build_arm → plan_step / CostPolicy.decide")
    print("TEST_DATA — 独立合成状态；未验证生产可达性；未验证硬件/模型指纹")
    print(f"计划 cell: {PLAN}; 不沿用其他负载的运行时长")
    results = []
    for name in args.arms:
        path = cfg_dir / f"{name}.yaml"
        if not path.is_file():
            ap.error(f"missing arm config: {path}")
        result = run_arm(path, state_names=args.states)
        result["arm_name"] = name
        results.append(result)
        _print_report(result)
    gap = _coverage_gap_analysis(results)
    print()
    print("候选的实际执行形状清单（未采样，不是固定 9 点网格）")
    print(f"旧 batch 输入修正: {len(gap['input_bug_corrections'])} 个候选")
    print(f"去重整步形状: {gap['unique_execution_shapes']} 个")
    for entry in gap["entries"]:
        states = sorted({source["state"] for source in entry["sources"]})
        statuses = sorted({coverage["status"] for coverage in entry["production_coverage"]})
        print(f"  {entry['point_id']} groups={entry['n_forwards']} "
              f"states={states} coverage={statuses}")
    print("旧 scalar 键命中只作诊断，不证明 S01/真实 KV/分组语义一致。")
    payload = {
        "evidence_class": "TEST_DATA",
        "note": "独立合法合成队列；FZ-1 A/FZ-2 A；不代表生产可达性或服务收益",
        "plan_cell": PLAN,
        "chain": "load_config → build_arm → plan_step / CostPolicy.decide",
        "states_declared": [{"name": state["name"], "fixture_reason": state["fixture_reason"],
                             "state_source": "independent_fixture",
                             "reachable_from_production": None} for state in STATES
                            if not args.states or state["name"] in args.states],
        "coverage_gap": gap,
        "arms": [{key: value for key, value in result.items() if key != "arm"}
                 for result in results],
    }
    if args.json:
        _write_json(args.json, payload)
    if args.manifest_json:
        _write_json(args.manifest_json, gap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
