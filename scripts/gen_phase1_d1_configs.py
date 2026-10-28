#!/usr/bin/env python3
"""生成 D1 的配置集（`configs/mthreads/phase1_d1/`）——**候选下限**的两步走。

背景（`NEXT_SESSION_TASKS.md` 的「D1 已裁（离线部分）」）
------------------------------------------------------
判据是「**预测整步 ≤ 每步预算**」。在 120 ms 预算下，512 的 prefill 步预测已超限
⇒ 若候选集里没有更小的块，控制器只能在「降级取最省」与「回退阈值」之间走。
⇒ 要做"小候选能不能被预测、被选中、被实际执行"的验证，必须**显式**给出更小的候选。

⚠️ 两个**不是**同一件事的参数（搞混会让实验白跑）
* `chunk_size_min/max` = **可动范围**（运行期上限的唯一权威仍是 `ChunkLimits`，由它派生）；
* `candidate_chunks` = **在范围内挑哪几个点去评估**（`None` ⇒ 旧行为，只取 `{min, max}`）。
⇒ 想让 128 真的可动，必须**同时**把 `chunk_size_min` 降到 128。

四份配置
--------
| 文件 | 用途 | `chunk_size_min` | `candidate_chunks` |
| --- | --- | --- | --- |
| `fixed_cost_min512.yaml` | **参照点**（= 部署后的当前行为） | 512 | `null`（⇒ `{512, 2048}`） |
| `fixed_cost_pilot.yaml` | **D1① 验证**：小候选能否被预测/选中/执行 | 128 | `[128, 256, 512, 2048]` |
| `fixed_cost_min256.yaml` | **D1② 对照**：只改下限 | 256 | `null`（⇒ `{256, 2048}`） |
| `fixed_cost_min128.yaml` | **D1② 对照**：只改下限 | 128 | `null`（⇒ `{128, 2048}`） |

⚠️ ① 与 ② 的唯一变量**不同**（① 是候选列表，② 是下限），**不得混算**；
且**都不得**与 `S14B1/S14B2/S14C1/S14C2` 混算。

用法::

    python scripts/gen_phase1_d1_configs.py            # 写盘
    python scripts/gen_phase1_d1_configs.py --check     # 只校验、不写
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = _ROOT / "configs" / "mthreads" / "phase1_step14b" / "fixed_cost.yaml"
OUT_DIR = _ROOT / "configs" / "mthreads" / "phase1_d1"

#: 所有 D1 配置都指向**新模型版本**（v7），**不覆盖**旧模型。
_MODEL_OLD = "cost_model_path: configs/cost_calib/model_v6.json"
_MODEL_NEW = "cost_model_path: configs/cost_calib/model_v7.json"

_MIN_OLD = "  chunk_size_min: 512\n"


def _spec() -> dict[str, dict]:
    """文件名 → 该份要施加的**唯一**变换（`None` 值 = 不施加）。"""
    return {
        "fixed_cost_min512.yaml": {"why": "参照点（部署后的当前行为）"},
        "fixed_cost_pilot.yaml": {
            "why": "D1① 验证：显式小候选 ⇒ 只回答「能否被预测/选中/执行」",
            "min": 128,
            "candidates": [128, 256, 512, 2048],
        },
        "fixed_cost_min256.yaml": {"why": "D1② 对照（只改下限）", "min": 256},
        "fixed_cost_min128.yaml": {"why": "D1② 对照（只改下限）", "min": 128},
    }


def build() -> dict[str, str]:
    if not REFERENCE.exists():
        raise SystemExit(f"✗ 找不到参照配置 {REFERENCE}")
    base = REFERENCE.read_text(encoding="utf-8")
    if _MODEL_OLD not in base:
        raise SystemExit(f"✗ 参照配置里没有 {_MODEL_OLD!r}（已改过？请改本脚本）")
    if _MIN_OLD not in base:
        raise SystemExit(f"✗ 参照配置里没有 {_MIN_OLD!r}（已改过？请改本脚本）")
    out: dict[str, str] = {}
    for name, spec in _spec().items():
        t = base.replace(_MODEL_OLD, _MODEL_NEW)
        if spec.get("min") is not None:
            t = t.replace(_MIN_OLD, f"  chunk_size_min: {int(spec['min'])}\n")
        cands = spec.get("candidates")
        if cands:
            anchor = "  signal_criterion: decode_p90\n"
            if anchor not in t:
                raise SystemExit("✗ 锚点 `signal_criterion` 不在（配置结构变了？）")
            t = t.replace(anchor, anchor + f"  candidate_chunks: {list(cands)}\n")
        out[name] = t
    return out


def verify() -> list[str]:
    """把"每份只差声明的那一处"与"分工正确"做成断言。"""
    for p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from run_benchmark import load_config  # noqa: E402

    bad: list[str] = []
    ref = load_config(REFERENCE)
    for name, spec in _spec().items():
        f = OUT_DIR / name
        if not f.exists():
            bad.append(f"缺 {name}")
            continue
        c = load_config(f)
        ctrl = c.controller
        if str(ctrl.cost_model_path) != "configs/cost_calib/model_v7.json":
            bad.append(f"{name}: cost_model_path 应为 model_v7.json，实际 {ctrl.cost_model_path}")
        want_min = spec.get("min", ref.controller.chunk_size_min)
        if int(ctrl.chunk_size_min) != int(want_min):
            bad.append(f"{name}: chunk_size_min 应为 {want_min}，实际 {ctrl.chunk_size_min}")
        want_c = spec.get("candidates")
        got_c = list(ctrl.candidate_chunks) if ctrl.candidate_chunks else None
        if got_c != want_c:
            bad.append(f"{name}: candidate_chunks 应为 {want_c}，实际 {got_c}")
        # 除声明项外必须逐字段相同（SLO / 负载 / 判据 / 宿主开销 / 余量）
        for k in ("target_tpot_ms", "hysteresis", "signal_criterion", "signal_mode",
                  "apply_safety_margin", "host_overhead_ms", "host_overhead_fit",
                  "chunk_size_max", "chunk_size_step"):
            if getattr(ctrl, k) != getattr(ref.controller, k):
                bad.append(f"{name}: controller.{k} 与参照不同（不止改了声明的那一处）")
        if c.run.slo.tpot_ms != ref.run.slo.tpot_ms or c.run.slo.ttft_ms != ref.run.slo.ttft_ms:
            bad.append(f"{name}: SLO 与参照不同")
        if c.workload_name != ref.workload_name or c.workload_overrides != ref.workload_overrides:
            bad.append(f"{name}: 负载与参照不同")
        if c.max_num_seqs != ref.max_num_seqs:
            bad.append(f"{name}: max_num_seqs 与参照不同")
        # 小候选必须**真的可动**：候选不得小于下限（否则被 ChunkLimits 夹回去）
        if ctrl.candidate_chunks:
            oob = [int(x) for x in ctrl.candidate_chunks if not (
                ctrl.chunk_size_min <= int(x) <= ctrl.chunk_size_max)]
            if oob:
                bad.append(f"{name}: candidate_chunks 越界 {oob} ⇒ 会被 ChunkLimits 夹回，白写")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 D1 的候选下限配置")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    if args.check:
        bad = verify()
        if bad:
            print("✗ " + "；".join(bad), file=sys.stderr)
            return 1
        print(f"✓ {len(_spec())} 个 D1 配置校验通过")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in build().items():
        with open(OUT_DIR / name, "w", encoding="utf-8", newline="\n") as fh:  # ⚠️ 显式 LF
            fh.write(text)
    bad = verify()
    if bad:
        print("✗ 写出后校验失败：" + "；".join(bad), file=sys.stderr)
        return 1
    print(f"已写出 {len(_spec())} 个配置到 {OUT_DIR}")
    for name, spec in _spec().items():
        print(f"  {name:<28} min={spec.get('min', '512(默认)'):<10} "
              f"candidates={spec.get('candidates') or 'null(⇒{min,max})'}  — {spec['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
