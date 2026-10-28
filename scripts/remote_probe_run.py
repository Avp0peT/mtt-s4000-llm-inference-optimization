#!/usr/bin/env python3
"""只读：打印一个 run 目录的关键身份与汇总字段（用于远端逐目录核验）。

用法：python3 scripts/remote_probe_run.py <run_dir> [more_dirs...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def probe(d: Path) -> None:
    meta_p = d / "meta.json"
    if not meta_p.exists():
        print(f"!! {d.name}: 缺 meta.json")
        return
    m = json.loads(meta_p.read_text(encoding="utf-8"))
    s = m.get("summary") or {}
    c = m.get("config") or {}
    w = c.get("workload") or {}
    ctr = c.get("controller") or {}
    rep = m.get("reproducibility") or {}
    cm = c.get("cost_model")
    if isinstance(cm, dict):
        cm = cm.get("path") or cm.get("name") or json.dumps(cm, ensure_ascii=False)[:80]
    print(f"=== {d.name} ===")
    print(f"  run_id     : {s.get('run_id')}")
    print(f"  label      : {m.get('data_label')}   commit={rep.get('git_commit')}  dirty={rep.get('git_dirty')}")
    print(f"  controller : kind={ctr.get('kind')} enabled={ctr.get('enabled')} shadow={ctr.get('shadow')} crit={ctr.get('signal_criterion')}")
    print(f"  cost_model : {cm}")
    wt = json.dumps(w, ensure_ascii=False)
    print(f"  workload   : {wt[:240]}")
    print(f"  goodput    : {s.get('slo_goodput_tok_s')}   slo_sat={s.get('slo_satisfaction_rate')}")
    print(f"  ttft p50/99: {(s.get('ttft') or {}).get('p50')} / {(s.get('ttft') or {}).get('p99')}")
    print(f"  tpot p50/99: {(s.get('tpot') or {}).get('p50')} / {(s.get('tpot') or {}).get('p99')}")
    print(f"  stall p50/99: {(s.get('stall') or {}).get('p50')} / {(s.get('stall') or {}).get('p99')}")
    print(f"  fail/rej/unf: {s.get('failure_rate')} / {s.get('rejection_rate')} / {s.get('unfinished_rate')}")
    print(f"  summary keys: {sorted(s.keys())}")
    eff = m.get("controller_effect") or {}
    if eff:
        print(f"  effect keys : {sorted(eff.keys())}")


def main() -> int:
    for a in sys.argv[1:]:
        probe(Path(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
