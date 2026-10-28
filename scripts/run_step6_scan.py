#!/usr/bin/env python3
"""PHASE 1 第⑥步的**静态扫描**运行器：逐格 dry-run，并汇总一张可核对的扫描表。

⚠️⚠️ **本脚本产出的所有数字都是合成的，只能用于"链路是否跑通"**
------------------------------------------------------------------------
它跑的是 ``--dry-run``，走的是**一阶设备模型**（`configs/*/device_model`），
不是真机。因此：

* **不能**据此评估过载 —— 模型里没有真实排队行为；
* **不能**据此评估 SLO —— 达标率由合成时延算出；
* **不能**据此评估性能 —— 吞吐与尾延迟都是模型给的。

它能证明的只有三件事：配置能加载、链路能跑、产物字段齐全。
真机数据只能来自第 08 步的 pilot（本脚本会把产物标成 ``TEST_DATA``，
聚合器默认拒收它们，这道隔离由标签保证、不靠人记得）。

为什么产物写到 ``results/scan_step6/`` 而不是 ``results/raw/``
------------------------------------------------------------
`results/raw/` 是**证据**目录，`run_benchmark` 的覆盖守卫规定：即使加了
``--force`` 也**永远不可覆盖**证据目录下的产物。而扫描要能反复重跑（改一格再跑），
所以它的落点必须在证据目录之外——``--force`` 也才会放行
（放行条件：非证据目录 + 本次 dry-run + 已有产物为 TEST_DATA，三条同时满足）。

用法::

    python scripts/run_step6_scan.py --only dry-run          # 冒烟：逐格 dry-run，退出码 0
    python scripts/run_step6_scan.py                          # dry-run + 扫描表
    python scripts/run_step6_scan.py --cells static_c8 adaptive_c8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.stage_report import build_stage_report, build_stage_specs  # noqa: E402
from gen_phase1_step6_configs import CELLS, OUT_DIR as CFG_DIR  # noqa: E402

DEFAULT_OUT = _ROOT / "results" / "scan_step6"


def _cells() -> dict[str, Any]:
    return {c.name: c for c in CELLS}


def _run_one(cfg_path: Path, run_id: str, out_dir: Path) -> tuple[int, str]:
    """跑一格 dry-run。返回 (退出码, 说明)。

    ⚠️ ``--force`` 在这里是**合法的**：本目录不是证据目录、本次是 dry-run。
    换到 ``results/raw`` 下同样的命令会被守卫拒掉——那是有意为之。
    """
    cmd = [
        sys.executable, "-m", "benchmark.run_benchmark",
        "--config", str(cfg_path),
        "--dry-run",
        "--force",
        "--run-id", run_id,
        "--out", str(out_dir),
        "--quiet",
    ]
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()
    msg = tail[-1] if tail else (proc.stderr or "").strip().splitlines()[-1:] or ""
    return proc.returncode, (msg if isinstance(msg, str) else " ".join(msg))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _summarize(run_dir: Path, cell: Any) -> dict[str, Any]:
    """从产物里抽出扫描表要的字段。**取不到就是 None**，不填 0。"""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {"cell": cell.name, "error": "meta.json 缺失"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    s = meta.get("summary") or {}
    cfg = meta.get("config") or {}
    tel = _read_jsonl(run_dir / "telemetry.jsonl")
    # ⚠️ `meta.config.workload` 是**字符串**（负载名），不是字典 —— 把它当字典读会
    # `AttributeError: 'str' object has no attribute 'get'`（本轮就是这么崩的）。
    # 并发要从这两处取：负载指纹优先，其次 workload_overrides。
    fp0 = ((meta.get("context") or {}).get("extra") or {}).get("workload_fingerprint") or {}
    overrides = cfg.get("workload_overrides") or {}

    qd = [t["queue_depth"] for t in tel if t.get("queue_depth") is not None]
    act = [t["running"] for t in tel if t.get("running") is not None]
    row: dict[str, Any] = {
        "cell": cell.name,
        "axis": cell.axis,
        "data_label": meta.get("data_label"),
        "dry_run": (meta.get("context") or {}).get("dry_run"),
        # ---- 覆盖项：验收点名要有 ----
        "max_num_seqs": (cfg.get("scheduler") or {}).get("max_num_seqs"),
        "chunk_enabled": (cfg.get("chunk") or {}).get("enabled"),
        "chunk_size": (cfg.get("chunk") or {}).get("chunk_size"),
        "max_num_batched_tokens": (cfg.get("chunk") or {}).get("max_num_batched_tokens"),
        "controller_enabled": (cfg.get("controller") or {}).get("enabled"),
        "target_tpot_ms": (cfg.get("controller") or {}).get("target_tpot_ms"),
        "workload": cfg.get("workload") if isinstance(cfg.get("workload"), str) else None,
        "concurrency": fp0.get("concurrency", overrides.get("concurrency")),
        "num_requests": fp0.get("num_requests", overrides.get("num_requests")),
        "arrival_model": fp0.get("arrival"),
        # ---- 固定批次吞吐与尾延迟（⚠️ 合成值）----
        # ⚠️ 键名是**查过产物**的，不是猜的：分布在 _dist() 生成的子字典里
        # （``summary.ttft.p99`` 等）。凭直觉写 ``summary.ttft_p99_ms`` 会静默拿到
        # ``None``——扫描表整列 n/a，看起来像"这一格没测"。
        "n_success": s.get("n_success"),
        "throughput_tok_s": s.get("throughput_tok_s"),
        "ttft_p99_ms": (s.get("ttft") or {}).get("p99"),
        "tpot_p99_ms": (s.get("tpot") or {}).get("p99"),
        "stall_p99_ms": (s.get("stall") or {}).get("p99"),
        "slo_satisfaction_rate": s.get("slo_satisfaction_rate"),
        # ---- 队列深度 / 活跃数 ----
        "queue_depth_max": max(qd) if qd else None,
        "active_max": max(act) if act else None,
        "n_steps": (s.get("extra") or {}).get("steps"),
    }

    # ---- 分阶段报告字段：只有 piecewise 的格才有 ----
    fp = ((meta.get("context") or {}).get("extra") or {}).get("workload_fingerprint") or {}
    segs = fp.get("arrival_segments")
    if not segs:
        row["stage_report"] = "n/a（非 piecewise）"
        return row
    reqs = _read_jsonl(run_dir / "requests.jsonl")
    # dry-run 的逐请求 SLO 判定：这里用**产物自报**的 success/时延粗判即可——
    # 扫描表的用途是"字段齐不齐"，不是"达标率是多少"。真值由 report_stages.py 给。
    rep = build_stage_report(
        requests=reqs,
        # ⚠️ 用现成的 build_stage_specs，**不要**自己拼 dict：
        # build_stage_report 要的是 StageSpec（它读 .end_ms 这类属性），
        # 传 dict 会 AttributeError: 'dict' object has no attribute 'end_ms'。
        stages=build_stage_specs(segs),
        telemetry=tel,
        actions=(meta.get("controller_trace") or []),
        measured_window_ms=(max(t["t_ms"] for t in tel) - min(t["t_ms"] for t in tel))
        if tel else None,
    )
    d = rep.describe()
    row["stage_report"] = {
        "n_cohorts": len(d["cohorts"]),
        "n_windows": len(d["windows"]),
        "window_labels": [w["label"] for w in d["windows"]],
        "recovery_status": d["recovery"]["status"],
        "reconciliation_ok": d["reconciliation"]["ok"],
        "reconciliation_problems": d["reconciliation"]["problems"],
    }
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="第⑥步静态扫描（dry-run / 扫描表）")
    ap.add_argument("--only", choices=("dry-run", "report"), default=None,
                    help="只跑其中一个阶段（默认两个都跑）")
    ap.add_argument("--cells", nargs="+", default=None, help="只跑这些格")
    ap.add_argument("--out", default=None, help=f"产物根目录（默认 {DEFAULT_OUT}）")
    ap.add_argument("--cfg-dir", default=None, help=f"配置目录（默认 {CFG_DIR}）")
    ap.add_argument("--tag", default=None, help="run-id 前缀（默认按时间戳）")
    ap.add_argument("--json", default=None, help="把扫描表写成 JSON")
    args = ap.parse_args(argv)

    known = _cells()
    names = args.cells or list(known)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"未知的格：{unknown}；可用：{sorted(known)}")

    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    cfg_dir = Path(args.cfg_dir) if args.cfg_dir else CFG_DIR
    stamp = args.tag or time.strftime("%Y%m%d-%H%M%S")

    methods = ("dry-run", "report") if args.only is None else (args.only,)
    rc_total = 0
    rows: list[dict[str, Any]] = []

    if "dry-run" in methods:
        print(f"=== dry-run {len(names)} 格 -> {out_dir} ===")
        for name in names:
            cfg_path = cfg_dir / f"{name}.yaml"
            if not cfg_path.exists():
                print(f"  ✗ {name}: 配置不存在 {cfg_path}（先生成：gen_phase1_step6_configs.py）")
                rc_total = 1
                continue
            run_id = f"SCAN6-{stamp}-{name}"
            rc, msg = _run_one(cfg_path, run_id, out_dir)
            artifact = (out_dir / run_id / "meta.json").exists()
            mark = "✓" if (rc == 0 and artifact) else "✗"
            if rc != 0 or not artifact:
                rc_total = 1
            print(f"  {mark} {name:<26} rc={rc} artifact={'yes' if artifact else 'NO'}  {msg}")
        if rc_total:
            print("✗ dry-run 阶段有失败", file=sys.stderr)
            return rc_total

    if "report" in methods:
        print(f"\n=== 扫描表（⚠️ 全部为**合成**数字，只能证明链路跑通）===")
        hdr = (f"{'格':<26}{'轴':<12}{'负载':<16}{'到达':<10}{'seqs':>5}{'ck':>6}"
               f"{'budget':>7}{'ctrl':>5}{'门限':>6}{'并发':>5}{'请求':>5}"
               f"{'吞吐tok/s':>11}{'TTFT p99':>10}{'TPOT p99':>10}{'STALL p99':>10}"
               f"{'队深max':>8}{'标签':>10}")
        print(hdr)
        for name in names:
            # 找到该格最近一次产物（按 run_id 前缀匹配）
            cands = sorted(p for p in out_dir.glob(f"SCAN6-*-{name}") if p.is_dir())
            if not cands:
                print(f"{name:<26} （没有产物，先跑 --only dry-run）")
                continue
            row = _summarize(cands[-1], known[name])
            rows.append(row)

            def g(key: str, nd: int = 1) -> str:
                v = row.get(key)
                return "n/a" if v is None else f"{v:.{nd}f}"

            print(f"{row['cell']:<26}{row['axis']:<12}"
                  f"{str(row.get('workload')):<16}{str(row.get('arrival_model')):<10}"
                  f"{str(row.get('max_num_seqs')):>5}"
                  f"{str(row.get('chunk_size')):>6}"
                  f"{str(row.get('max_num_batched_tokens')):>7}"
                  f"{str(row.get('controller_enabled')):>5}"
                  f"{str(row.get('target_tpot_ms')):>6}"
                  f"{str(row.get('concurrency')):>5}"
                  f"{str(row.get('num_requests')):>5}"
                  f"{g('throughput_tok_s'):>11}{g('ttft_p99_ms'):>10}"
                  f"{g('tpot_p99_ms'):>10}{g('stall_p99_ms'):>10}"
                  f"{str(row.get('queue_depth_max')):>8}"
                  f"{str(row.get('data_label')):>10}")

        print("\n--- 验收点名要覆盖的字段（逐格核对，缺了就显形）---")
        problems: list[str] = []
        for row in rows:
            if row.get("max_num_seqs") != 8:
                problems.append(f"{row['cell']}: max_num_seqs={row.get('max_num_seqs')}（应 8）")
            if row.get("data_label") != "TEST_DATA":
                problems.append(
                    f"{row['cell']}: data_label={row.get('data_label')!r}（dry-run 应为 TEST_DATA）"
                )
            for k in ("chunk_enabled", "controller_enabled", "concurrency",
                      "queue_depth_max", "n_steps"):
                if row.get(k) is None:
                    problems.append(f"{row['cell']}: 字段 {k} 缺失")
        for row in rows:
            sr = row.get("stage_report")
            if isinstance(sr, dict):
                if not sr["reconciliation_ok"]:
                    problems.append(
                        f"{row['cell']}: 分阶段报告对账不通过 {sr['reconciliation_problems']}"
                    )
                if sr["n_cohorts"] < 3:
                    problems.append(
                        f"{row['cell']}: 分阶段报告只认出 {sr['n_cohorts']} 段（piecewise 应 ≥3）"
                    )
        if problems:
            print("✗ 覆盖项有问题：")
            for p in problems:
                print(f"  - {p}")
            rc_total = 1
        else:
            print("✓ 覆盖项齐全（max_num_seqs / 分块 / 预算 / 控制器 / 门限 / 并发 / 队深 / 分阶段字段）")

        piecewise = [r for r in rows if isinstance(r.get("stage_report"), dict)]
        if piecewise:
            print("\n--- 负载轨迹格的分阶段报告 ---")
            for r in piecewise:
                sr = r["stage_report"]
                print(f"  {r['cell']:<26} 窗口={sr['window_labels']} "
                      f"恢复={sr['recovery_status']} 对账={'✓' if sr['reconciliation_ok'] else '✗'}")

        if args.json:
            Path(args.json).write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8", newline="\n",
            )
            print(f"\n已写出扫描表 JSON: {args.json}")

    print("\n⚠️ 以上全部为 **--dry-run / 合成** 数字：只能证明配置能加载、链路能跑、"
          "产物字段齐全。\n   **不得**据此评估过载、SLO 或性能——那些只能来自第 08 步的真机 pilot。")
    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
