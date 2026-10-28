#!/usr/bin/env python3
"""**成本模型的覆盖率**：一个头部指标，不是一个事后解释。

为什么要有它
------------
第 14 步实测：`cost` 臂在 `fixed` 格上的回退率 48.65%，我一开始把它解释成"空闲步回退"
—— **错了**。按真实候选形状重算后：**空闲 = 0**，185 个决策**全是有效工作**，
回退成因 **94.4% 是"batch 不在标定网格 {1,2,4,8}"**（真实服务 batch 连续取 1..8）。

⇒ 结论：**"kv 不超限" ≠ "执行形状在范围内"**，而且
**回退率高时"与阈值臂表现接近"不是"成本模型没用"，是"成本模型没被用上"**。
⇒ 所以覆盖率必须**和性能指标并排报**，并且**按有效工作**算（把空闲剔出去，
否则空闲步会把覆盖率稀释成一个好看但没意义的数）。

口径（写死在这里，避免事后换算法）
----------------------------------
* **有效工作决策** = 该次决策里至少有一个候选在真的做活
  （`tokens_prefilled > 0` 或 `batch > 0`）；
* **覆盖** = 该决策**没有回退**（模型给出了预测）；
* 覆盖率 = 覆盖 / 有效工作决策（**分母剔除空闲**）。

用法::

    python scripts/check_model_coverage.py --tag S14C1 \
        --model configs/cost_calib/model_v3.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: 兜底模型常量**刻意删除**：原先默认 `model_v3.json`，而 v3 被判否、故意未导出
#: ⇒ 默认值指向不存在的文件，闸门会抛异常（看起来像坏了）。现在要求显式指定来源。
#: ⚠️ 教训：**"有个默认值"本身不是 bug，但一个指向"被否掉的版本"的默认值等于静默指错。**


def _is_work(cands: list[dict]) -> bool:
    return any((int(c.get("tokens_prefilled") or 0) > 0) or (int(c.get("batch") or 0) > 0)
               for c in cands)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="成本模型覆盖率（只读产物）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--glob", default=None, help="缺省 <tag>-*cost*/meta.json")
    ap.add_argument("--model", default=None,
                    help="模型文件（与 --cfg 互斥；**不给默认值**，理由见 main 里的说明）")
    #: ★ **推荐用法**：从配置里读 `cost_model_path`。
    #: 为什么：闸门验的模型与批实际加载的模型是**两个指代**时，会变成"验 A 跑 B"。
    #: 实测踩过（2026-09-26）：v6 链的闸门显式传 `--model model_v6.json`，而紧接着的批
    #: `--cfg-dir phase1_step14` 读**配置**、配置里指向 `model_v5.json` ⇒ 两者指纹相同
    #: ⇒ **不报错**、产物完全正常，但那一批**没有**检验它声称要检验的东西。
    #: ⇒ 让闸门**从配置读**，两者在结构上就只能是同一个。
    ap.add_argument("--cfg", default=None,
                    help="从该配置的 `controller.cost_model_path` 取模型（与 --model 互斥）")
    ap.add_argument("--json", default=None)
    ap.add_argument("--assume-kv", type=int, default=None,
                    help="旧产物的轨迹里**没有 `kv_len`**（当时没记，已修）⇒ 重放时用一个近似值；"
                         "结果必须标明是近似的")
    ap.add_argument("--replay-mode", default=None,
                    help="把记录到的候选形状**重放**一遍，用指定的预测方式重算覆盖率 "
                         "（segmented / lookup）。产物里的回退标志是**运行当时**的口径，"
                         "换方式必须重放才能比。")
    args = ap.parse_args(argv)

    # ---- 模型从哪来：配置优先，且**两个来源同时给就报错** --------------------
    if args.cfg and args.model:
        print("✗ `--cfg` 与 `--model` 只能给一个 —— 给两个正是'验 A 跑 B'的温床"
              "（闸门验一个、实跑读配置里的另一个）", file=sys.stderr)
        return 2
    if args.cfg:
        from run_benchmark import load_config
        cfg = load_config(args.cfg)
        rel = cfg.controller.cost_model_path
        if not rel:
            print(f"✗ {args.cfg} 的 controller.cost_model_path 为空（该臂不加载模型）",
                  file=sys.stderr)
            return 2
        model_path = Path(rel)
        if not model_path.is_absolute():
            model_path = _ROOT / model_path
        if not model_path.exists():
            print(f"✗ 配置指向的模型不存在：{model_path}", file=sys.stderr)
            return 2
        args.model = str(model_path)
        print(f"模型来源 = **配置** {args.cfg} → controller.cost_model_path = {rel}")
    else:
        if not args.model:
            # ⚠️ **不给静默默认**：原先默认 `model_v3.json`，而 v3 是被判否、
            # **故意没有导出**的那一版（见 `results/raw/_costcalib/README.md`）⇒
            # 不带参数跑这个闸门会**抛 FileNotFoundError**，看起来像"闸门坏了"，
            # 实际是"默认值指向一个不存在的文件"。而任何静默默认都是"验 A 跑 B"的温床。
            print("✗ 必须显式指定模型来源：`--cfg <配置>`（**推荐**，与实跑同源）"
                  "或 `--model <模型文件>`。\n"
                  "  刻意不设默认值——原先默认指向 `model_v3.json`，而 v3 被判否、"
                  "**故意未导出** ⇒ 默认值本身就是坏的。", file=sys.stderr)
            return 2
        args.model = str(args.model)

    from ascend_opt.cost_model import CostModel
    try:
        model = CostModel.load_offline(args.model)
    except (OSError, ValueError) as e:
        # ⚠️ 闸门崩栈与闸门判决是两件事：崩栈会被读成"工具坏了"，
        # 从而**绕过**这次判定。凡是加载失败都转成干净的 rc=2 并说明原因。
        print(f"✗ 模型加载失败（{args.model}）：{e}", file=sys.stderr)
        return 2
    coef = model.payload["coefficients"]
    rng = model.payload["calibrated_range"]

    pattern = args.glob or f"results/raw/{args.tag}-*cost*/meta.json"
    files = sorted(glob.glob(str(_ROOT / pattern)))
    if not files:
        print(f"✗ 没找到 {pattern}", file=sys.stderr)
        return 2

    # ---- 可选：按指定方式**重放**候选形状，重算覆盖率 ----------------------
    if args.replay_mode:
        n_w = n_hit = 0
        b_hist: dict[int, list[int]] = {}
        for f in files:
            meta = json.loads(Path(f).read_text(encoding="utf-8"))
            for e in meta.get("controller_trace") or []:
                d = e.get("cost_policy") or e.get("slo_weave")
                if not d:
                    continue
                for c in (d.get("candidates") or []):
                    b = int(c.get("batch") or 0)
                    pf = int(c.get("tokens_prefilled") or 0)
                    kv = int(c.get("kv_len") or 0) or int(args.assume_kv or 0)
                    if b <= 0:
                        continue                     # 空闲候选不进分母
                    n_w += 1
                    hit = model.predict(batch=b, prefill_tok=pf, kv_len=kv,
                                        mode=args.replay_mode) is not None
                    n_hit += int(hit)
                    b_hist.setdefault(b, []).append(int(hit))
        print(f"tag={args.tag}｜run={len(files)}｜模型={Path(args.model).name}"
              f"｜**重放方式 = {args.replay_mode}**")
        print(f"  有效工作候选 {n_w}｜命中 {n_hit}｜"
              f"★ **覆盖率 = {n_hit / max(1, n_w):.2%}**")
        if args.assume_kv:
            print(f"  ⚠️ 旧产物没记 `kv_len` ⇒ 重放用**近似值 {args.assume_kv}**"
                  "（结论标明是近似的；新产物已修）")
        bk = model.payload.get("lookup_bucket") or {}
        print(f"  分桶: pf_step={bk.get('pf_step')} kv_step={bk.get('kv_step')}")
        print("  ⚠️ 未命中的候选 = 该形状**没有实测点（含桶内）** ⇒ 只能回退（不插值、不外推）")
        for b in sorted(b_hist):
            v = b_hist[b]
            print(f"     batch={b:<2} n={len(v):<4} 命中率 {sum(v) / len(v):.1%}")
        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(
                {"tag": args.tag, "replay_mode": args.replay_mode,
                 "n_work_candidates": n_w, "n_hit": n_hit,
                 "coverage": n_hit / max(1, n_w),
                 "per_batch": {str(b): sum(v) / len(v) for b, v in b_hist.items()}},
                ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
            print(f"  已写出：{out}")
        return 0

    n_dec = n_work = n_cov = 0
    why = Counter()
    shapes = Counter()
    for f in files:
        meta = json.loads(Path(f).read_text(encoding="utf-8"))
        for e in meta.get("controller_trace") or []:
            d = e.get("cost_policy")
            if not d:
                continue
            n_dec += 1
            cands = d.get("candidates") or []
            if not _is_work(cands):
                continue                       # 空闲 ⇒ **不进分母**
            n_work += 1
            if not d.get("fallback_reason"):
                n_cov += 1
                continue
            b = int(cands[0].get("batch") or 0)
            pf = int(cands[0].get("tokens_prefilled") or 0)
            shapes[(b, pf)] += 1
            if str(b) not in coef:
                why["batch 不在标定网格"] += 1
            else:
                lo, hi = rng[str(b)]["prefill_tok"]
                why["batch 在网格内、prefill 超出该 batch 的范围" if not lo <= pf <= hi
                    else "kv 越界（本脚本无法区分，需看模型范围）"] += 1

    out = {
        "tag": args.tag,
        "model": args.model,
        "n_runs": len(files),
        "n_decisions": n_dec,
        "n_work_decisions": n_work,
        "n_covered": n_cov,
        "coverage_on_work": (n_cov / n_work if n_work else None),
        "coverage_on_all": (n_cov / n_dec if n_dec else None),
        "fallback_reasons": dict(why),
        "fallback_shapes_top": [{"batch": b, "prefill_tok": p, "n": n}
                                for (b, p), n in shapes.most_common(8)],
        "calibrated_batches": sorted(coef, key=int),
    }
    print(f"tag={args.tag}｜run={len(files)}｜模型={Path(args.model).name}")
    print(f"  标定网格里的 batch: {sorted(coef, key=int)}")
    print(f"  决策 {n_dec}｜**有效工作** {n_work}｜其中覆盖 {n_cov}")
    cov_w = out["coverage_on_work"]
    cov_a = out["coverage_on_all"]
    s_w = "—" if cov_w is None else f"{cov_w:.2%}"
    s_a = "—" if cov_a is None else f"{cov_a:.2%}"
    print(f"  ★ **有效工作上的覆盖率** = {s_w}（全决策口径 {s_a} —— "
          "两者差得多就说明空闲步很多，**分母必须用有效工作**）")
    if why:
        print("  回退成因:")
        for k, v in why.most_common():
            print(f"     {k}: {v}")
        print("  回退形状 top:", out["fallback_shapes_top"][:5])
    if args.json:
        q = Path(args.json)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                     encoding="utf-8", newline="\n")
        print(f"  已写出：{q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
