#!/usr/bin/env python
"""参数扫描：在一维或多维参数网格上批量起 run。

定位说明
--------
扫描是**探索**，不是结论。因此：
- 每个扫描点默认只跑 1 次（repeats=1），它的用途是圈定「值得深挖的区间」；
- 扫描点被固定下来之后，用 ``run_benchmark.py`` 对它跑 >=3 次重复，
  再登记进 EXPERIMENTS.md —— 只有那一步的产物才能进结论。

扫描定义（YAML）
----------------
    name: chunk_size_sweep
    base_config: configs/baseline/default.yaml
    dry_run: true
    grid:
      chunk.policy: [fixed_chunk, decode_priority_chunk]
      chunk.chunk_size: [1024, 2048, 4096]
      workload.name: [long_context_qa]      # 也可扫工作负载本身

``grid`` 的键是相对配置根节点的点分路径。多键时取笛卡尔积。
每个组合会被写成一份独立的配置文件并保存在产物目录中——扫描结果必须
可追溯到当时的完整配置，不能只有一份「基配置 + 我记得改了什么」。
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_benchmark import load_config, run_once, _write_text_lf  # noqa: E402


def set_dotted(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    """按点分路径写入嵌套字典，路径不存在时创建（并在注释中说明这是刻意的）。"""
    parts = dotted_key.split(".")
    node = payload
    for part in parts[:-1]:
        if part not in node:
            node[part] = {}
        if not isinstance(node[part], dict):
            raise ValueError(
                f"路径 {dotted_key!r} 中途遇到非映射节点 {part!r}，无法写入"
            )
        node = node[part]
    node[parts[-1]] = value


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"扫描维度 {key!r} 必须是非空列表，收到 {values!r}")
    combos = [
        dict(zip(keys, combo))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]
    return combos


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="参数网格扫描")
    p.add_argument("--sweep", required=True, help="扫描定义 YAML 路径")
    p.add_argument("--out", default=None, help="产物根目录，默认 results/raw")
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 个组合（调试用）")
    p.add_argument("--only-dry-run", action="store_true", help="强制全部走 dry-run（安全开关）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 PyYAML：pip install pyyaml") from exc

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        raise SystemExit(f"扫描定义不存在：{sweep_path}")

    sweep = yaml.safe_load(sweep_path.read_text(encoding="utf-8")) or {}
    name = sweep.get("name") or sweep_path.stem
    base_config = sweep.get("base_config")
    if not base_config:
        raise SystemExit("扫描定义缺少 base_config")

    # dry_run 默认 True：扫描通常在没有独占机器的环境下做，
    # 默认走模拟可以避免误占用真机；要上真机必须显式写 dry_run: false。
    dry_run = bool(sweep.get("dry_run", True)) or args.only_dry_run
    grid = sweep.get("grid") or {}
    if not grid:
        raise SystemExit("扫描定义缺少 grid")

    combos = expand_grid(grid)
    if args.limit is not None:
        combos = combos[: args.limit]

    out_root = Path(args.out) if args.out else _ROOT / "results" / "raw"
    sweep_id = f"{time.strftime('%Y%m%d-%H%M%S')}-sweep-{name}"
    sweep_dir = out_root / sweep_id
    gen_dir = sweep_dir / "generated_configs"
    gen_dir.mkdir(parents=True, exist_ok=True)

    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = _ROOT / base_path
    base_payload = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

    print(f"扫描 {name}：{len(combos)} 个组合，dry_run={dry_run}")
    print(f"  基配置：{base_path}")
    print(f"  产物目录：{sweep_dir}\n")

    entries: list[dict[str, Any]] = []
    for idx, combo in enumerate(combos):
        payload = json.loads(json.dumps(base_payload))  # 深拷贝，避免组合间串味
        for key, value in combo.items():
            set_dotted(payload, key, value)

        combo_stem = f"{idx:03d}-" + "-".join(
            f"{k.split('.')[-1]}_{_slug(v)}" for k, v in sorted(combo.items())
        )
        cfg_path = gen_dir / f"{combo_stem}.yaml"
        _write_text_lf(
            cfg_path,
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        )

        cfg = load_config(cfg_path)
        cfg.run.repeats = 1   # 扫描点不重复，重复留给后续的确认实验
        entry: dict[str, Any] = {"index": idx, "combo": combo, "config": str(cfg_path)}
        try:
            run_dir = run_once(
                cfg,
                dry_run=dry_run,
                out_dir=out_root,
                repeat_index=0,
                run_id=f"{sweep_id}-{combo_stem}",
                extra_meta={"sweep_id": sweep_id, "sweep_combo": combo},
                verbose=True,
            )
            entry["status"] = "done"
            entry["run_dir"] = str(run_dir)
        except Exception as exc:  # 单个点失败不该让整轮扫描白跑
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [{idx}] 失败：{entry['error']}")
        entries.append(entry)

    manifest = {
        "sweep_id": sweep_id,
        "name": name,
        "base_config": str(base_path),
        "dry_run": dry_run,
        "grid": grid,
        "n_combos": len(combos),
        "n_done": sum(1 for e in entries if e["status"] == "done"),
        "n_failed": sum(1 for e in entries if e["status"] == "failed"),
        "entries": entries,
        "warning": (
            "扫描点各只运行 1 次，仅用于圈定区间，不得作为结论；"
            "确认结论需对选定配置跑 >=3 次重复并登记 EXPERIMENTS.md"
        ),
    }
    manifest_path = sweep_dir / "sweep_manifest.json"
    _write_text_lf(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"\n完成：成功 {manifest['n_done']} / 失败 {manifest['n_failed']}")
    print(f"清单：{manifest_path}")
    print(f"聚合：python benchmark/collect_metrics.py --raw {out_root}"
          + ("  --include-dry-run" if dry_run else ""))
    return 0 if manifest["n_failed"] == 0 else 2


def _slug(value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in text)


if __name__ == "__main__":
    raise SystemExit(main())
