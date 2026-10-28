"""Pure production plan/forward shape contract; no tensors or hardware imports.

KV lengths are lengths already in the cache, before this call.  A generated
token has not entered the cache until the following decode call consumes it.
Request IDs identify telemetry only and are excluded from calibration keys.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeVar

from .step_planner import StepPlan
from .workload import Request

T = TypeVar("T")


def group_by_new_tokens(items: Sequence[T]) -> list[tuple[int, list[T]]]:
    """The Transformers driver's exact grouping and stable within-group order."""
    buckets: dict[int, list[T]] = {}
    for item in items:
        n = int(item.n_new_tokens)
        if n <= 0:
            raise ValueError("n_new_tokens must be positive")
        buckets.setdefault(n, []).append(item)
    return sorted(buckets.items(), reverse=True)


@dataclass(frozen=True)
class ForwardSpec:
    req: Request
    n_new_tokens: int
    need_token: bool
    role: str


def plan_forward_specs(plan: StepPlan, *, waiting: Sequence[Request],
                       decoding: Sequence[Request],
                       prefilling: Sequence[tuple[Request, int]]) -> list[ForwardSpec]:
    """Translate a plan once, shared by policy and real backend."""
    all_requests = list(waiting) + list(decoding) + [r for r, _ in prefilling]
    ids = [r.req_id for r in all_requests]
    if len(ids) != len(set(ids)):
        raise ValueError("waiting/prefilling/decoding request IDs must be disjoint")
    by_id = {r.req_id: r for r in all_requests}
    order = list(prefilling) + [
        (by_id[rid], by_id[rid].prompt_len - plan.cached_tokens.get(rid, 0))
        for rid in plan.new_start_ids if rid not in plan.immediate_ids]
    out: list[ForwardSpec] = []
    for req, remaining in order:
        take = int(plan.prefill_alloc.get(req.req_id, 0))
        if take <= 0:
            continue
        if take > remaining:
            raise ValueError("prefill allocation exceeds remaining prompt")
        out.append(ForwardSpec(req, take, take == remaining, "prefill"))
    out.extend(ForwardSpec(by_id[rid], 1, True, "decode") for rid in plan.decode_ids)
    return out


@dataclass(frozen=True)
class ExecutionItem:
    req_id: str
    n_new_tokens: int
    kv_len: int | None
    role: str
    need_token: bool

    def describe(self) -> dict[str, Any]:
        return dict(req_id=self.req_id, n_new_tokens=self.n_new_tokens,
                    kv_len=self.kv_len, role=self.role, need_token=self.need_token)


@dataclass(frozen=True)
class ExecutionShape:
    items: tuple[ExecutionItem, ...]

    def describe(self) -> dict[str, Any]:
        groups = []
        for n, items in group_by_new_tokens(self.items):
            kv = [i.kv_len for i in items]
            groups.append(dict(n_new_tokens=n, items=[i.describe() for i in items],
                               batch=len(items),
                               past_max=None if None in kv else max(kv)))
        return dict(schema="execution-shape-v1", items=[i.describe() for i in self.items],
                    groups=groups, n_forwards=len(groups), batch=len(self.items))

    def key(self) -> str:
        return execution_shape_key(self.describe())


def execution_shape_key(shape: Mapping[str, Any]) -> str:
    """Exact whole-step key: no bucketing, padding approximation or group sum."""
    vectors = [[i["n_new_tokens"], i["kv_len"], i["role"], i["need_token"]]
               for i in shape["items"]]
    return json.dumps(vectors, separators=(",", ":"), allow_nan=False)


def plan_execution_shape(plan: StepPlan, *, waiting: Sequence[Request],
                         decoding: Sequence[Request],
                         prefilling: Sequence[tuple[Request, int]],
                         per_request_kv_lengths: Mapping[str, int] | None = None) -> ExecutionShape:
    specs = plan_forward_specs(plan, waiting=waiting, decoding=decoding, prefilling=prefilling)
    known = {r.req_id: int(r.prompt_len - remaining) for r, remaining in prefilling}
    known.update({rid: int(plan.cached_tokens.get(rid, 0)) for rid in plan.new_start_ids})
    if per_request_kv_lengths is not None:
        for rid, kv in per_request_kv_lengths.items():
            if isinstance(kv, bool) or not isinstance(kv, int) or kv < 0:
                raise ValueError(f"invalid KV length for {rid}: {kv!r}")
            if rid in known and known[rid] != kv:
                raise ValueError(f"KV length disagrees with prefill progress: {rid}")
            known[rid] = kv
    return ExecutionShape(tuple(ExecutionItem(s.req.req_id, s.n_new_tokens,
                           known.get(s.req.req_id), s.role, s.need_token) for s in specs))


def execution_signature(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Execution choices needed to interpret a shape's measured cost."""
    return {"contract": "execution-shape-v1", "grouping": "descending_n_new_tokens",
            **{k: runtime.get(k) for k in ("model", "model_revision", "precision")},
            **{k: bool(runtime.get(k, False)) for k in
               ("logits_last_only", "sample_needed_only", "fuse_qkv", "do_sample",
                "fuse_gate_up", "persistent_kv_slots", "capture_logits_digest")},
            "retire_empty_cache": bool(runtime.get("retire_empty_cache", True)),
            "persistent_kv_capacity": runtime.get("persistent_kv_capacity"),
            "linear_fuse_shapes": runtime.get("linear_fuse_shapes"),
            "temperature": float(runtime.get("temperature") or 1.0),
            "top_p": float(runtime.get("top_p") or 1.0)}
