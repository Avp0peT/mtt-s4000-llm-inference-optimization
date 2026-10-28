#!/usr/bin/env python3
"""**SLO 可达边界**：在给定模型与判据口径下，哪些 `(batch, prefill_tok, kv_len)` 真的够得着 SLO。

为什么要有它
------------
`paper/experiments.md` §3 要一个 `slo_reach` cell —— 理由是**让"达标吞吐"这个头条量有区分度**：
`fixed` 格上四臂达标率**全为 0**（连 `static` 的 TPOT p50 都 156 ms > SLO 100 ms），
于是 goodput 那一列**不携带信息**（不是"都很差"，是"这个格测不出差异"）。

但"选个更轻的负载"不能靠感觉：**判据是 `预测整步 ≤ deadline`**，而整步 = 前向 + host。
⇒ 本脚本直接把该判据在 `(batch, kv_len)` 平面上算出来，给出**边界**，让 cell 的设计
（并发、长度、分块）有据可依。

口径（刻意**复用**而不是重写）
------------------------------
* 前向预测：`CostModel.predict()`（与控制器同一条路径，含 `apply_safety_margin` 开关）；
* host 开销：`CostPolicy.host_overhead()`（**同一个实现** —— 它含"表命中否则线性拟合"的规则，
  另写一套简化版会与真实判据不一致）；
* deadline：`configs/.../controller.target_tpot_ms`（成本臂的每步预算就是它）。
执行 `arm_factory.build_arm()` 之外的一切，脚本**不引入**任何新口径。

⚠️ **边界性质**：这是**预测**的可行性，不是实测。
前向预测的实测误差中位 **−4.4%（准）**、host 是**实测表** ⇒ 本表适合当**设计依据**；
但它**不是**结论（"该格 goodput 会 > 0"必须真机跑出来才算）。
⚠️ host 项是**本机 + 当前执行路径**的量，换路径/换 batch 上限要重测。

用法::

    python scripts/slo_reach_frontier.py --model configs/cost_calib/model_v4.json \
        --cfg configs/mthreads/phase1_step14b/fixed_cost.yaml --prefill 0 512 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.cost_model import CostModel                    # noqa: E402
from ascend_opt.cost_policy import CostPolicy, CostPolicyConfig  # noqa: E402
from run_benchmark import load_config                          # noqa: E402


def _policy(cfg_path: Path, model_path: Path, slo: float) -> CostPolicy:
    cfg = load_config(cfg_path)
    c = cfg.controller
    return CostPolicy(
        CostPolicyConfig(
            enabled=True,
            step_deadline_ms=slo,
            host_overhead_ms=getattr(c, "host_overhead_ms", None),
            host_overhead_fit=(tuple(c.host_overhead_fit)
                               if getattr(c, "host_overhead_fit", None) else None),
            apply_safety_margin=bool(getattr(c, "apply_safety_margin", True)),
        ),
        CostModel.load_offline(model_path),
    )


def frontier(policy: CostPolicy, *, batch: int, prefill_tok: int,
             kvs: list[int], slo: float, mode: str) -> dict:
    """该 `(batch, prefill_tok)` 下，`kv_len` 最大能到多少仍然可行性成立。

    ⚠️ **三态，不许合并**（实测踩到：把"未命中"印成"不可行"，于是得出了
    "任何含 prefill 的步都不可行"这个**错的**结论——真实原因是 `lookup` 模式下
    那个形状**根本没有键**）：
    * `feasible`：有预测且整步 ≤ 预算；
    * `infeasible`：**有预测**但超预算（这才是真的"够不着"）；
    * `no_prediction`：模型**没给预测**（查表未命中 / 越界 / 系数退化）⇒ **不可判**。
    """
    host = policy.host_overhead(batch)
    ok: list[int] = []
    detail: list[dict] = []
    counts = {"feasible": 0, "infeasible": 0, "no_prediction": 0}
    # ⚠️ **第四态：口径存疑**。`model_v4` 及更早的 `mixed_serving` 标定点用的是
    # `uniform` 口径（**每条序列各做** `prefill_tok` 个新 token），而 harness 的真实结构是
    # `real_mixed`（1 条做全部 + 其余各 1 个）⇒ 这些点的预测**偏大约 batch 倍**。
    # 于是它们算出来的"不可行"**不是**真机事实，**不得**当结论用。
    # 判据用模型自己记的 `mixed_shape_mode`（v6 才有），**不靠猜**。
    caliber_ok = str(policy.model.payload.get("mixed_shape_mode") or "") == "real_mixed"
    suspect = prefill_tok > 0 and batch > 1 and not caliber_ok
    for kv in kvs:
        pred = policy.model.predict(batch=batch, prefill_tok=prefill_tok, kv_len=kv,
                                    mode=mode,
                                    apply_margin=policy.config.apply_safety_margin)
        if pred is None:
            state, step = "no_prediction", None
        else:
            step = pred.total_ms + host
            state = "feasible" if step <= slo else "infeasible"
            if state == "feasible":
                ok.append(kv)
        counts[state] += 1
        detail.append({"kv_len": kv,
                       "pred_fwd_ms": None if pred is None else pred.total_ms,
                       "host_ms": host, "pred_step_ms": step, "state": state})
    return {"batch": batch, "prefill_tok": prefill_tok, "host_ms": host, "mode": mode,
            "max_feasible_kv": max(ok) if ok else None, "counts": counts,
            "caliber_suspect": suspect, "detail": detail}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SLO 可达边界（预测口径，只读）")
    ap.add_argument("--model", default=str(_ROOT / "configs" / "cost_calib" / "model_v4.json"))
    ap.add_argument("--cfg", default=str(_ROOT / "configs" / "mthreads" / "phase1_step14b" /
                                        "fixed_cost.yaml"))
    ap.add_argument("--slo", type=float, default=None, help="缺省取配置里的 target_tpot_ms")
    ap.add_argument("--mode", choices=("segmented", "lookup"), default="segmented",
                    help="预测方式。**默认 segmented**（要的是连续边界）；"
                         "`lookup` 下未命中的形状没有预测，会大量落在'不可判'里")
    ap.add_argument("--prefill", nargs="*", type=int, default=[0, 512, 1024])
    ap.add_argument("--batch", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--kv", nargs="*", type=int, default=[512, 1024, 1536, 2048, 3072, 4096])
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.cfg))
    slo = float(args.slo if args.slo is not None else cfg.controller.target_tpot_ms)
    policy = _policy(Path(args.cfg), Path(args.model), slo)
    model_default = str(policy.model.payload.get("default_mode") or "segmented")

    print(f"模型 {Path(args.model).name}｜配置 {Path(args.cfg).name}｜"
          f"deadline(每步预算) = {slo:.1f} ms｜余量 {'开' if policy.config.apply_safety_margin else '关'}")
    print(f"预测方式 = **{args.mode}**（模型自带的 default_mode = {model_default}）")
    mixed_ok = str(policy.model.payload.get("mixed_shape_mode") or "") == "real_mixed"
    print(f"mixed 形状口径 = {policy.model.payload.get('mixed_shape_mode') or '未记录（=旧口径）'}"
          f" ⇒ {'' if mixed_ok else '⚠️ **pf>0 且 batch>1 的格子预测偏大约 batch 倍，不可当结论**'}")
    if args.mode != model_default:
        print("⚠️ 与模型自带的 default_mode **不同** ⇒ 本表**不**代表控制器实际会用的口径，"
              "只用于看趋势（要复现控制器的判定，请传 --mode " + model_default + "）")
    print("⚠️ 预测口径，不是实测：前向预测实测中位 −4.4%（准），host 为实测表；"
          "**不得**把本表当结论（goodput 是否 > 0 必须真机跑）\n")

    out: dict = {"model": Path(args.model).name, "cfg": Path(args.cfg).name,
                 "deadline_ms": slo, "mode": args.mode,
                 "model_default_mode": model_default, "rows": []}
    for pf in args.prefill:
        print(f"-- prefill_tok = {pf}")
        print("   batch | host ms | 最大可行 kv | 可行/超预算/不可判")
        for b in args.batch:
            f = frontier(policy, batch=b, prefill_tok=pf, kvs=sorted(args.kv), slo=slo,
                         mode=args.mode)
            mv, c = f["max_feasible_kv"], f["counts"]
            mark = "  ⚠️ 口径存疑" if f["caliber_suspect"] else ""
            print(f"   {b:>5} | {f['host_ms']:>7.1f} | "
                  f"{'—' if mv is None else str(mv):>11} | "
                  f"{c['feasible']:>3}/{c['infeasible']:>3}/{c['no_prediction']:>3}{mark}")
            out["rows"].append({k: v for k, v in f.items() if k != "detail"})
        print()

    print("读法：'不可判'不是'不可行' —— 前者是**模型没给预测**（查表未命中/越界），"
          "后者是**有预测且超预算**。把两者混在一起会得出错的结论（实测踩过）。")
    print("设计 `slo_reach` 时，把并发/长度选到落在这张表**可行**区内的点，才能让 goodput 有区分度。")
    print("⚠️ 反过来也成立：若负载落在可行区**之外**，四臂 goodput 仍会全为 0 ——"
          "那是**过载边界**，不是「这个方法没用」（§1 阶段五）。")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding="utf-8", newline="\n")
        print(f"已写出：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
