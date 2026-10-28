#!/usr/bin/env python3
"""控制开销的**离线微基准**（补第 14 步验收里的"控制开销"一项）。

为什么是离线
------------
第 14 步的产物里**没有控制耗时的埋点**。而批次已在**冻结版**上跑，
现在加埋点会把 run 混在不同版本之间（验收第 5 条禁止）⇒ 只能离线补。

⚠️ 边界（必须与数字一起给）：这是**本机**（开发机）上对**纯逻辑**的计时，
**不是**真机上的控制开销；它证明的是"选一次动作需要多少 CPU 时间"这个**量级**，
用来判断"控制器本身会不会成为瓶颈"。真机上的开销还包括 I/O 与调度器交互，
本基准**不覆盖**那一部分。

用法::

    python scripts/bench_control_overhead.py --reps 20000
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt import apply_arm  # noqa: E402
from ascend_opt.chunk_policy import ChunkConfig, ChunkLimits, DecodePriorityChunkPolicy  # noqa: E402
from ascend_opt.cache_policy import PrefixCacheConfig, PrefixCachePolicy  # noqa: E402
from ascend_opt.cost_model import CostModel  # noqa: E402
from ascend_opt.cost_policy import CostPolicy, CostPolicyConfig  # noqa: E402
from ascend_opt.scheduler import ContinuousBatchingScheduler  # noqa: E402
from ascend_opt.slo_weave import SLOWeaveConfig, SLOWeavePolicy  # noqa: E402
from ascend_opt.workload import Request  # noqa: E402

MODEL = _ROOT / "configs" / "cost_calib" / "model_v1.json"


class _S:
    """最小的状态替身：只提供 `apply_arm` 用到的字段（与两条路径的契约一致）。"""

    def __init__(self, i: int, prompt: int, gen: int, decoded: int, phase: str):
        self.req = Request(req_id=f"r{i}", prompt_len=prompt, gen_len=gen)
        self.prefill_remaining = 0 if phase == "decode" else prompt
        self.decode_remaining = gen - decoded
        self.last_emit_ms = 1000.0 + i


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="控制开销离线微基准")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    model = CostModel.load_offline(MODEL)
    sched = ContinuousBatchingScheduler()
    chunk_policy = DecodePriorityChunkPolicy(ChunkConfig(enabled=True))
    cache = PrefixCachePolicy(PrefixCacheConfig())
    limits = ChunkLimits(chunk_size=2048, max_num_batched_tokens=8192)

    decoding = [_S(i, 1536, 32, 16, "decode") for i in range(6)]
    waiting = [_S(i, 1536, 32, 0, "waiting") for i in range(10, 14)]
    prefilling = [_S(i, 1536, 32, 0, "prefill") for i in range(20, 23)]

    cases = {
        "slo_weave": SLOWeavePolicy(SLOWeaveConfig(enabled=True, tpot_target_ms=120.0), model),
        "cost": CostPolicy(CostPolicyConfig(enabled=True, step_deadline_ms=120.0), model),
    }

    print(f"复算 {args.reps} 次／臂（本机纯逻辑计时；**不是**真机开销）")
    out: dict[str, dict[str, float]] = {}
    for kind, policy in cases.items():
        ts: list[float] = []
        for i in range(args.reps):
            t0 = time.perf_counter()
            apply_arm(kind=kind, policy=policy, shadow=False, step=i,
                      now_ms=1000.0 + i * 6.0, cur_chunk=1024, cur_max_tokens=8192,
                      waiting=waiting, decoding=decoding, prefilling=prefilling,
                      scheduler=sched, chunk_policy=chunk_policy, cache=cache,
                      limits=limits, max_num_seqs=8, chunk_enabled=True)
            ts.append((time.perf_counter() - t0) * 1000.0)     # ms
        ts.sort()
        out[kind] = {
            "p50_ms": st.median(ts), "p99_ms": ts[int(0.99 * len(ts))],
            "max_ms": ts[-1], "mean_ms": sum(ts) / len(ts),
        }
        print(f"  {kind:<10} p50 {out[kind]['p50_ms'] * 1000:8.1f} µs | "
              f"p99 {out[kind]['p99_ms'] * 1000:8.1f} µs | "
              f"max {out[kind]['max_ms'] * 1000:8.1f} µs")
    # 与一步的实测步成本对比：这是"控制开销可忽略吗"的判据
    step_ms = 240.0
    worst = max(v["p99_ms"] for v in out.values())
    print(f"\n  参照：该 cell 的实测步成本 ≈ {step_ms:.0f} ms")
    print(f"  ⇒ 控制开销 p99 占一步的 {worst / step_ms:.4%}"
          f"（{'可忽略' if worst / step_ms < 0.01 else '⚠️ 不可忽略'}）")
    out["_reference_step_ms"] = step_ms
    out["_overhead_p99_share_of_step"] = worst / step_ms
    out["_boundary"] = ("本机（开发机）纯逻辑计时；**不是**真机开销，也不含 I/O 与调度器交互")
    if args.json:
        q = Path(args.json)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding="utf-8", newline="\n")
        print(f"\n已写出：{q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
