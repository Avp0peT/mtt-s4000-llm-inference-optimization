#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S14B2 逐运行身份核验（只读）。

覆盖用户要求的六项：**数据标签 / 源码 / 环境 / 配置 / 模型 / 重复覆盖**。
用法（远端）：
    AQ_TAG=S14B2 AQ_ROOT=/root/ascend_probe_d1 python3 /root/autoq/remote_identity_s14b2.py

纪律：
* 只读；剔除 `.tar` 暂存包；缺 `meta.json` 的目录不计入。
* 任何"应为唯一"的维度出现多值 ⇒ **显著报错**（不静默）。
* 缺失字段报 `None`，**不填 0**。
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(os.environ.get("AQ_ROOT", "/root/ascend_probe_d1"))
TAG = os.environ.get("AQ_TAG", "S14B2")
RAW = ROOT / "results" / "raw"
EXPECT_REPEATS = int(os.environ.get("AQ_REPEATS", "5"))
_STAGING = (".tar", ".tar.gz", ".tgz")

#: 期望唯一的维度：键 -> meta 里的取值函数
UNIQUE_DIMS = {
    "git_commit": lambda m: (m.get("reproducibility") or {}).get("git_commit"),
    "git_dirty": lambda m: (m.get("reproducibility") or {}).get("git_dirty"),
    "model": lambda m: (m.get("reproducibility") or {}).get("model"),
    "model_revision": lambda m: (m.get("reproducibility") or {}).get("model_revision"),
    "precision": lambda m: (m.get("reproducibility") or {}).get("precision"),
    "quantization": lambda m: (m.get("reproducibility") or {}).get("quantization"),
    "execution_environment": lambda m: (m.get("reproducibility") or {}).get("execution_environment"),
    "hardware_platform": lambda m: (m.get("reproducibility") or {}).get("hardware_platform"),
    "device_model": lambda m: (m.get("reproducibility") or {}).get("device_model"),
    "backend": lambda m: (m.get("reproducibility") or {}).get("backend"),
    "git_source": lambda m: (m.get("reproducibility") or {}).get("git_source"),
    "input_length": lambda m: (m.get("reproducibility") or {}).get("input_length"),
    "output_length": lambda m: (m.get("reproducibility") or {}).get("output_length"),
    "concurrency": lambda m: (m.get("reproducibility") or {}).get("concurrency"),
    "batch_configuration": lambda m: (m.get("reproducibility") or {}).get("batch_configuration"),
    "prefix_cache_enabled": lambda m: (m.get("reproducibility") or {}).get("prefix_cache_enabled"),
    "random_seed": lambda m: (m.get("reproducibility") or {}).get("random_seed"),
    # 环境（软件栈）
    "python": lambda m: (m.get("software_env") or {}).get("python"),
    "torch_version": lambda m: (m.get("software_env") or {}).get("torch_version"),
    "transformers_version": lambda m: (m.get("software_env") or {}).get("transformers_version"),
    "vllm_version": lambda m: (m.get("software_env") or {}).get("vllm_version"),
    "mthreads_driver_version": lambda m: (m.get("software_env") or {}).get("mthreads_driver_version"),
    "musa_version": lambda m: (m.get("software_env") or {}).get("musa_version"),
    # SLO（必须冻结）
    "slo_ttft_ms": lambda m: (((m.get("config") or {}).get("run") or {}).get("slo") or {}).get("ttft_ms"),
    "slo_tpot_ms": lambda m: (((m.get("config") or {}).get("run") or {}).get("slo") or {}).get("tpot_ms"),
    # 控制器冻结量
    "target_tpot_ms": lambda m: ((m.get("config") or {}).get("controller") or {}).get("target_tpot_ms"),
    "hysteresis": lambda m: ((m.get("config") or {}).get("controller") or {}).get("hysteresis"),
    # 硬件探针可用性
    "hardware_env_available": lambda m: (m.get("hardware_env") or {}).get("available"),
}


def _dirs() -> list[pathlib.Path]:
    out = []
    for p in sorted(RAW.glob(f"{TAG}-*")):
        if not p.is_dir() or p.name.lower().endswith(_STAGING):
            continue
        if not (p / "meta.json").is_file():
            continue
        out.append(p)
    return out


def main() -> int:
    ds = _dirs()
    print(f"=== {TAG} 身份核验：{len(ds)} 个真 run（已剔暂存包）===")
    per_meta = []
    for d in ds:
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        label = m.get("data_label")
        stem = (m.get("config") or {}).get("config_stem")
        rid = (m.get("context") or {}).get("run_id")
        rep = (m.get("context") or {}).get("repeat_index")
        wl = (m.get("context") or {}).get("workload")
        per_meta.append((d, m, label, stem, rid, rep, wl))

    # ---- 1. 数据标签 ----
    print("\n[1] 数据标签分布（缺 = None，不填 0）")
    print("   ", dict(Counter(x[2] for x in per_meta)))

    # ---- 2. 重复覆盖：4 臂 × EXPECT_REPEATS ----
    print(f"\n[2] 重复覆盖（期望每臂 {EXPECT_REPEATS} 个）")
    by_arm: dict[str, list[int]] = defaultdict(list)
    for x in per_meta:
        if x[5] is not None:
            by_arm[str(x[3])].append(int(x[5]))
    total_expected = 0
    coverage_ok = True
    for arm in sorted(by_arm):
        reps = sorted(by_arm[arm])
        missing = [r for r in range(EXPECT_REPEATS) if r not in reps]
        expect_arm = EXPECT_REPEATS
        total_expected += expect_arm
        flag = "OK " if not missing else "MISS"
        if missing:
            coverage_ok = False
        print(f"   [{flag}] {arm:16s} n={len(reps)}/{expect_arm} reps={reps} 缺={missing or '-'}")
    print(f"   合计 {sum(len(v) for v in by_arm.values())}/{total_expected}"
          f"  覆盖完整={coverage_ok}")

    # ---- 3. 唯一性维度 ----
    print("\n[3] 应为唯一的维度（多值 = 血缘污染，必须显形）")
    problems = []
    for name, fn in sorted(UNIQUE_DIMS.items()):
        vals = Counter()
        for _, m, *_ in per_meta:
            try:
                vals[repr(fn(m))] += 1
            except Exception as exc:  # noqa: BLE001
                vals[f"<ERR {exc!r}>"] += 1
        keys = list(vals)
        if len(keys) == 1:
            print(f"   [OK ] {name:24s} = {keys[0]}  (n={sum(vals.values())})")
        else:
            print(f"   [!!!] {name:24s} 多值 {dict(vals)}")
            problems.append((name, dict(vals)))

    # ---- 4. 每臂内配置是否一致（只差声明变量） ----
    print("\n[4] 臂内配置一致性（臂 = config_stem；臂内除声明变量外应逐字相同）")
    for arm in sorted(by_arm):
        sub = [x for x in per_meta if str(x[3]) == arm]
        ctrl = Counter()
        for _, m, *_ in sub:
            c = (m.get("config") or {}).get("controller") or {}
            ctrl[(c.get("kind"), c.get("signal_criterion"), c.get("cost_model_path"))] += 1
        if len(ctrl) == 1:
            print(f"   [OK ] {arm:16s} (kind, criterion, model) = {list(ctrl)[0]}")
        else:
            print(f"   [!!!] {arm:16s} 臂内不一致: {dict(ctrl)}")
            problems.append((f"arm-internal:{arm}", dict(ctrl)))

    # ---- 5. 结论 ----
    print("\n=== 结论 ===")
    if problems:
        print(f"[!] 身份核验发现 {len(problems)} 类问题：")
        for name, detail in problems:
            print(f"    - {name}: {detail}")
    else:
        print("[OK] 所有唯一性维度一致，臂内配置一致")
    if not coverage_ok:
        print("[!] 重复覆盖**未满** ⇒ 该批次尚不可作为完整矩阵出结论")
    # 非零退出表示"身份有问题"（覆盖不满不算身份问题）
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
