#!/usr/bin/env python3
"""从**产物**生成 `EXPERIMENTS.md` 的登记行（数据驱动，不手抄）。

为什么要有它
------------
项目的纪律是"**每个结论都要指向一条登记行**"。而 2026-09-26 的复核发现：
`EXPERIMENTS.md` 里**一条 `S14*` 都没有**（只有两行占位的格式示例）——
于是下午那些结论**在登记表里无迹可查**。

手抄这些数字既慢又会错（而且错法通常是"看起来对"）。⇒ 从产物自动生成。

⚠️ 生成的行**保留空值**（缺的字段写 `-` 而不是 0）：本仓库一贯的口径是
"不可测的值保持 None，禁止填 0"，登记表也一样。

用法::

    python scripts/emit_experiment_rows.py --tag S14B1
    python scripts/emit_experiment_rows.py --tag S14B1 --append EXPERIMENTS.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

HEADER = ("| 编号 | run_id | 工作负载 | 配置 | 变量 | 重复 | 数据标签 | "
          "TTFT P50/P95/P99 (ms) | TPOT P50/P99 (ms) | stall P99 (ms) | 吞吐 (tok/s) | "
          "达标率 | 达标吞吐 (req/s) | 达标吞吐 (tok/s) | 未完成率 | 离散度(稳健CV) | 状态 |")


def _f(v, spec="{:.2f}", dash="-"):
    return dash if v is None else spec.format(v)


def _q(d: dict, k: str):
    v = (d or {}).get(k)
    return None if v is None else float(v)


def rows_for_tag(tag: str) -> tuple[list[str], list[str]]:
    rows, notes = [], []
    dirs = sorted(glob.glob(str(_ROOT / "results" / "raw" / f"{tag}-*")))
    for i, d in enumerate(dirs, 1):
        mp = Path(d) / "meta.json"
        if not mp.exists():
            notes.append(f"{Path(d).name}: 缺 meta.json ⇒ 不进登记表")
            continue
        m = json.loads(mp.read_text(encoding="utf-8"))
        s = m.get("summary") or {}
        cfg = m.get("config") or {}
        ctrl = cfg.get("controller") or {}
        rid = s.get("run_id") or Path(d).name
        rep = m.get("reproducibility") or {}
        st = s.get("stability") or {}
        # 变量：这一批里"变的是什么"由 config 与 run_id 决定；这里如实列出关键几项
        var = (f"controller.kind={ctrl.get('kind')}, shadow={cfg.get('shadow', ctrl.get('shadow'))}, "
               f"criterion={ctrl.get('signal_criterion')}")
        cv = None
        if st.get("ttft_robust_cv") is not None:
            cv = max(float(st.get("ttft_robust_cv") or 0), float(st.get("tpot_robust_cv") or 0))
        rows.append("| EXP-{tag}-{i:03d} | `{rid}` | {wl} | `{cfgpath}` | {var} | {rep} | {lab} | "
                    "{ttft} | {tpot} | {stall} | {thr} | {sat} | {greq} | {gtok} | {unf} | {cv} | "
                    "{status} |".format(
                        tag=tag, i=i, rid=rid, wl=s.get("workload") or "-",
                        # 配置名在 `summary.config` 里（产物里没有 config.path）
                        cfgpath=(s.get("config") or "-"),
                        var=var, rep="-", lab=m.get("data_label") or "-",
                        ttft="/".join(_f(_q(s.get("ttft") or {}, q)) for q in ("p50", "p95", "p99")),
                        tpot="/".join(_f(_q(s.get("tpot") or {}, q)) for q in ("p50", "p99")),
                        stall=_f(_q(s.get("stall") or {}, "p99")),
                        thr=_f(s.get("throughput_tok_s")),
                        sat=_f(s.get("slo_satisfaction_rate"), "{:.3f}"),
                        greq=_f(s.get("slo_goodput_req_s")),
                        gtok=_f(s.get("slo_goodput_tok_s")),
                        unf=_f(s.get("unfinished_rate"), "{:.3f}"),
                        cv=_f(cv, "{:.4f}"),
                        status="registered" if m.get("data_label") == "MEASURED"
                        else "TEST_DATA（仅链路验证，不得进结论）"))
    if not dirs:
        notes.append(f"{tag}: 没找到产物目录")
    return rows, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从产物生成 EXPERIMENTS.md 的登记行")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--append", default=None, help="直接追加到该 markdown 文件")
    args = ap.parse_args(argv)

    rows, notes = rows_for_tag(args.tag)
    for n in notes:
        print(f"  ⚠️ {n}")
    if not rows:
        print("✗ 没有可登记的行", flush=True)
        return 2
    block = [f"", f"<!-- {args.tag}：由 scripts/emit_experiment_rows.py 从产物生成 -->",
             HEADER, "| " + " | ".join(["---"] * 17) + " |", *rows]
    text = "\n".join(block) + "\n"
    if args.append:
        p = Path(args.append)
        with p.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"已追加 {len(rows)} 行到 {p}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
