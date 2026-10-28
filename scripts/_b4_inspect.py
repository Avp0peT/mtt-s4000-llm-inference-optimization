#!/usr/bin/env python3
"""打印一个 costcalib 探针产物的结构性摘要（远端用，避免多层引号）。

用法::  python3 _b4_inspect.py <artifact.json>
"""
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
print("keys:", sorted(d.keys()))
for k in ("protocol_version", "session_id", "grid_sha256", "complete", "subset",
          "grid_path", "elapsed_s", "note"):
    print(f"{k}: {json.dumps(d.get(k), ensure_ascii=False)[:300]}")
pts = d.get("points") or []
print("n_points:", len(pts))
if pts:
    p = pts[0]
    print("point keys:", sorted(p.keys()))
    print("point0:", json.dumps({k: v for k, v in p.items()
                                 if k not in ("block_a", "block_b", "raw")},
                                ensure_ascii=False))
    print("point0.block_a:", json.dumps(p.get("block_a"), ensure_ascii=False)[:300])
    print("point0.block_b:", json.dumps(p.get("block_b"), ensure_ascii=False)[:300])
print("fingerprint:", json.dumps(d.get("fingerprint"), ensure_ascii=False)[:500])
print("session_policy:", json.dumps(d.get("session_policy"), ensure_ascii=False)[:200])
