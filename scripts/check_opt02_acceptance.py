#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPT-02 的**可复算验收**：B2（门限形式）× B3（候选集中间点）。

用法::

    "C:/Users/13944/.workbuddy/binaries/python/envs/default/Scripts/python.exe" \
        scripts/check_opt02_acceptance.py

为什么要有它
------------
OPT-02 的两条验收都要求"**产物里看得见**"，而不是"代码里写了"：
* B3：dry-run 产物里 `candidate_chunks` 含**中间值**，且中间值真的进入**评估集**；
* B2：`--dry-run` **链路可跑**。

⚠️ 第二条极易被读成"dry-run 验证了新门限" —— 那是**假的**。实测（本脚本每次都复测）：
dry-run 路径**从不喂** `decode_samples` ⇒ `n_decode_samples=0` ⇒ 判据恒"不可测"
⇒ 控制器全程"保守不动"、`signals.gate_base_ms` **一次都不落盘**。
⇒ dry-run 能证明的只有"装配/加载/落盘没崩"，**不能**证明任何与判据有关的逻辑。
判据逻辑的离线证据在 `tests/test_controller_gate_mode.py`（含两条反向对照）。

产物落在 `--workdir`（默认系统临时目录）——**刻意不写 `results/raw/**`**
（OPT-02 红线）。全部为 `TEST_DATA`，不得进入结论。

退出码：0 = 全部通过；1 = 有断言失败（逐条打印）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: 基准配置。**只在这里改一处**，三个变体都由它派生 ⇒ 变体之间的差异可逐行核对。
BASE_CFG = _ROOT / "configs" / "mthreads" / "phase1_step14b" / "fixed_cost.yaml"

#: 基准配置里要替换的锚点（改基准配置会让本脚本**响亮失败**，而不是静默错位）。
_ANCHORS = {
    "prompt_len_min": ("  prompt_len_min: 1536\n", "  prompt_len_min: 1024\n"),
    "prompt_len_max": ("  prompt_len_max: 1536\n", "  prompt_len_max: 1024\n"),
    "concurrency": ("  concurrency: 16\n", "  concurrency: 4\n"),
    "max_num_seqs": ("  max_num_seqs: 8\n", "  max_num_seqs: 4\n"),
    "chunk_size": ("  chunk_size: 1024\n", "  chunk_size: 512\n"),
    "chunk_size_min": ("  chunk_size_min: 512\n", "  chunk_size_min: 128\n"),
    "fingerprint": ("  cost_model_fingerprint_source: observed\n",
                    "  cost_model_fingerprint_source: declared\n"),
}
_CAND_BLOCK = (
    "  candidate_chunks:\n"
    "  - 128\n  - 256\n  - 512\n  - 1024\n  - 2048\n"
)
_CRITERION_ANCHOR = "  signal_criterion: decode_p90\n"
_GATE_BLOCK = (
    _CRITERION_ANCHOR
    + "  gate_mode: normalized\n"
    + "  workload_scale_seed_ms: 63.0\n"
    + "  workload_scale_alpha: 0.1\n"
)


def _derive(base: str, *, with_candidates: bool, with_gate: bool) -> str:
    src = base
    for name, (old, new) in _ANCHORS.items():
        if old not in src:
            raise SystemExit(
                f"✗ 锚点 {name!r} 在基准配置里找不到 ⇒ 基准配置变了；"
                f"本脚本拒绝带着错位的锚点继续跑（那会静默改错东西）"
            )
        src = src.replace(old, new, 1)
    if _CRITERION_ANCHOR not in src:
        raise SystemExit("✗ 锚点 signal_criterion 找不到")
    if with_candidates:
        src = src.replace(_CRITERION_ANCHOR, _GATE_BLOCK if with_gate else
                          _CRITERION_ANCHOR + _CAND_BLOCK, 1)
    return src


def _run_dry(config: Path, out: Path, run_id: str, notes: str) -> Path:
    cmd = [
        sys.executable, str(_ROOT / "benchmark" / "run_benchmark.py"),
        "--config", str(config), "--dry-run", "--out", str(out),
        "--run-id", run_id, "--notes", notes, "--quiet",
    ]
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"✗ dry-run 失败（rc={proc.returncode}）：{run_id}")
    return out / run_id


def _load(d: Path) -> tuple[dict, list[dict], list[dict]]:
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    tel = [json.loads(l) for l in (d / "telemetry.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    return meta, meta.get("controller_trace") or [], tel


def _candidate_chunks(trace: list[dict]) -> set[int]:
    out: set[int] = set()
    for e in trace:
        for c in ((e.get("cost_policy") or {}).get("candidates") or []):
            if c.get("chunk") is not None:
                out.add(int(c["chunk"]))
    return out


def _action_entries(trace: list[dict]) -> list[dict]:
    """只取阈值控制器的动作条目（成本臂的条目是 `{step, cost_policy}`）。"""
    return [e for e in trace if "reason" in e and "cost_policy" not in e]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OPT-02 可复算验收（B2 × B3）")
    ap.add_argument("--workdir", default=None,
                    help="工作目录（默认系统临时目录）。**不要指向 results/raw**")
    ap.add_argument("--keep", action="store_true", help="保留工作目录（默认保留）")
    args = ap.parse_args(argv)

    wd = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix="opt02_accept_"))
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "out"
    out.mkdir(exist_ok=True)
    if "results" in wd.parts and "raw" in wd.parts:
        raise SystemExit("✗ --workdir 不得落在 results/raw 下（OPT-02 红线）")

    base = BASE_CFG.read_text(encoding="utf-8")
    cfgs = {
        "cand": (True, False),
        "nocand": (False, False),
        "normgate": (True, True),
    }
    for name, (wc, wg) in cfgs.items():
        (wd / f"{name}.yaml").write_text(
            _derive(base, with_candidates=wc, with_gate=wg),
            encoding="utf-8", newline="\n")

    print(f"工作目录：{wd}")
    runs = {
        name: _run_dry(wd / f"{name}.yaml", out, f"OPT02-ACC-{name}",
                       f"OPT-02 验收（{name}）；TEST_DATA")
        for name in cfgs
    }

    fails: list[str] = []

    def check(cond: bool, label: str, detail: str = "") -> None:
        print(f"  {'✓' if cond else '✗'} {label}" + (f"   {detail}" if detail else ""))
        if not cond:
            fails.append(label)

    # ---------------- B3 ----------------
    print("\n【B3】候选集补中间点：dry-run 产物必须自证")
    meta_c, trace_c, tel_c = _load(runs["cand"])
    meta_n, trace_n, _ = _load(runs["nocand"])
    declared = (meta_c.get("config") or {}).get("controller", {}).get("candidate_chunks")
    check(declared == [128, 256, 512, 1024, 2048],
          "meta 记录显式候选列表（含中间值）", f"= {declared}")
    cand_c = _candidate_chunks(trace_c)
    check({256, 512}.issubset(cand_c),
          "trace 的评估集含中间值 256/512", f"= {sorted(cand_c)}")
    cand_n = _candidate_chunks(trace_n)
    check(not {256, 512}.issubset(cand_n),
          "★反向对照：不给 candidate_chunks 时中间值**不出现**",
          f"= {sorted(cand_n)}（只剩 min/max）")
    eff_c = sorted({r["chunk_size"] for r in tel_c if r.get("chunk_size") is not None})
    print(f"  · telemetry 生效 chunk_size（参考）= {eff_c}")

    # ---------------- B2 ----------------
    print("\n【B2】门限形式：dry-run 链路可跑（**及其能证明的边界**）")
    meta_g, trace_g, _ = _load(runs["normgate"])
    ctl = (meta_g.get("config") or {}).get("controller", {})
    check(ctl.get("gate_mode") == "normalized",
          "配置与产物记录了 gate_mode", f"= {ctl.get('gate_mode')}")
    check(ctl.get("workload_scale_seed_ms") == 63.0,
          "seed 进产物（可核验参考从哪起步）", f"= {ctl.get('workload_scale_seed_ms')}")
    acts = _action_entries(trace_g)
    check(bool(acts), "控制器动作条目已落盘", f"n={len(acts)}")
    unmeasurable = sum(1 for a in acts if "不可测" in a.get("reason", ""))
    print(f"  · 其中「判据不可测 ⇒ 保守不动」{unmeasurable}/{len(acts)}"
          f"（余下 {len(acts) - unmeasurable} 条是更早的早退：窗口为空）")
    # ★ 判据**是否被评估过**的权威判据 = 轨迹里有没有一个非 None 的 `criterion_ms`。
    # 用 reason 文本计数只是参考——文本会随措辞改动而漂，而字段不会。
    evaluated = sum(1 for a in acts
                    if (a.get("signals") or {}).get("criterion_ms") is not None)
    check(evaluated == 0,
          "★ dry-run 下判据**一次都没被评估** ⇒ 控制器全程不动",
          f"非 None 的 criterion_ms 条数 = {evaluated}")
    no_gate_base = sum(1 for a in acts if "gate_base_ms" not in (a.get("signals") or {}))
    check(no_gate_base == len(acts),
          "★ 因此 gate_base_ms **一次都没落盘**",
          "⇒ dry-run 不能验证任何与判据有关的逻辑（见脚本 docstring）")
    cov = {int((a.get("state") or {}).get("n_decode_samples") or 0) for a in acts}
    check(cov == {0}, "根因：dry-run 从**不喂** decode_samples ⇒ 窗口为空",
          f"n_decode_samples={sorted(cov)}")

    print("\n" + "=" * 74)
    if fails:
        print(f"✗ 未通过 {len(fails)} 条：")
        for f in fails:
            print(f"    - {f}")
        return 1
    print("✓ 全部通过。产物目录：")
    for name, d in runs.items():
        print(f"    {name:9s} {d}")
    print("   ⚠️ 全部 TEST_DATA；dry-run 数字不得进入结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
