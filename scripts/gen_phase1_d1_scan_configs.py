#!/usr/bin/env python3
"""生成 D1 的**低并发扫描**配置集（`configs/mthreads/phase1_d1_scan/`）。

依据
----
`EXPERIMENTS.md` §7d「低并发扫描批 · 预注册（先冻结规则，再看数据）」。
本节写于扫描批产生任何数据**之前**；本脚本是它的机械实现。

唯一变量
--------
每份配置**只**改两处，且都写进文件名：

| 处 | 字段 | 取值 |
| --- | --- | --- |
| 1 | `workload.concurrency` | `2 / 4 / 6 / 8` |
| 2 | `scheduler.max_num_seqs` | `4 / 8` |

⚠️ **`max_num_seqs` 也是变量**，所以文件名必须带上它。只带 `concurrency`
会在 4 个文件里各出现两份同名列 —— 覆盖一份，另一份静默消失。

⚠️ 底稿是 `phase1_d1/fixed_cost_pilot.yaml`（不是 `phase1_step14b/fixed_cost.yaml`）：
后者是 `chunk_size_min=512` + `model_v6`，而本批要与 `S14GV` **同一条代码路径**
（`min=128` / `candidates=[128,256,512,2048]` / `model_v7`），否则又引入第二个变量。

`max_num_seqs` 同时改 `scheduler` 与 `controller` 两处是有意的
----------------------------------------------------------------
§7c 的实测：`max_num_seqs` 既决定批预算（`scheduler`），也决定控制器的决策空间
（`controller.max_batched_tokens_min` 等下限在 8 序列下才有意义；`running≈4` 门槛
也是在"批最多 8"下测的）。两份声明**必须一致**，否则会出现"批被调度器限到 8、
控制器以为能开 8 但按 4 规划"的静默错配。因此本生成器**同时**改写两处，
并在 `verify()` 里断言它们相等。

用法::

    python scripts/gen_phase1_d1_scan_configs.py            # 写盘
    python scripts/gen_phase1_d1_scan_configs.py --check     # 只校验、不写
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: 底稿 = D1① 的 pilot（与 `S14GV` 同款），**不是** step14b 的 reference。
REFERENCE = _ROOT / "configs" / "mthreads" / "phase1_d1" / "fixed_cost_pilot.yaml"
OUT_DIR = _ROOT / "configs" / "mthreads" / "phase1_d1_scan"

#: 网格（§7d 第 2 节）。刻意用元组常量而不是散在代码里：它同时被
#: `tests/` 与本脚本的校验读，只留一处定义。
CONCURRENCIES: tuple[int, ...] = (2, 4, 6, 8)
MAX_NUM_SEQS: tuple[int, ...] = (4, 8)

_ANCHOR_CONCURRENCY = "  concurrency: 16\n"
_ANCHOR_MAX_SEQS = "  max_num_seqs: 8\n"


def cell_name(concurrency: int, max_num_seqs: int) -> str:
    """格名 = 文件名主干。**两个变量都进名字**（见模块头的理由）。"""
    return f"fixed_cost_scan_c{concurrency}_m{max_num_seqs}"


def cells() -> list[tuple[int, int]]:
    """按 (concurrency 升序, max_num_seqs 升序) 展开网格。"""
    return [(c, m) for c in CONCURRENCIES for m in MAX_NUM_SEQS]


def build() -> dict[str, str]:
    if not REFERENCE.exists():
        raise SystemExit(f"✗ 找不到底稿 {REFERENCE}")
    base = REFERENCE.read_text(encoding="utf-8")
    # 底稿必须**逐字**含这两个锚点；否则说明底稿变了，本脚本的替换是错的。
    for anchor, what in ((_ANCHOR_CONCURRENCY, "workload.concurrency: 16"),
                         (_ANCHOR_MAX_SEQS, "scheduler.max_num_seqs: 8")):
        if base.count(anchor) != 1:
            raise SystemExit(
                f"✗ 底稿里 {what!r} 出现 {base.count(anchor)} 次（期望恰好 1 次）"
                f"—— 底稿结构变了？请对照 {REFERENCE.name} 改本脚本"
            )

    out: dict[str, str] = {}
    for concurrency, max_seqs in cells():
        t = base
        t = t.replace(_ANCHOR_CONCURRENCY, f"  concurrency: {concurrency}\n")
        t = t.replace(_ANCHOR_MAX_SEQS, f"  max_num_seqs: {max_seqs}\n")
        # notes 里写明本格的两个变量：`meta.json` 自带 notes，执行顺序因此
        # 可以从产物本身重建（见 run_step3_matrix.py 的文件头）。
        old_notes_tail = "（第09步冻结）"
        t = t.replace(
            old_notes_tail,
            f"（第09步冻结）｜**扫描批 §7d**：concurrency={concurrency} / "
            f"max_num_seqs={max_seqs}（两处已同步）",
        )
        out[f"{cell_name(concurrency, max_seqs)}.yaml"] = t
    return out


def verify() -> list[str]:
    """把「每份只差声明的两处」与「两处声明互相一致」做成断言。"""
    for p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from run_benchmark import load_config  # noqa: E402

    bad: list[str] = []
    ref = load_config(REFERENCE)
    for concurrency, max_seqs in cells():
        name = f"{cell_name(concurrency, max_seqs)}.yaml"
        f = OUT_DIR / name
        if not f.exists():
            bad.append(f"缺 {name}")
            continue
        c = load_config(f)

        # ① 两个变量各自对
        if int(c.workload_overrides.get("concurrency", -1)) != concurrency:
            bad.append(f"{name}: workload.concurrency 应为 {concurrency}，"
                       f"实际 {c.workload_overrides.get('concurrency')}")
        if int(c.max_num_seqs) != max_seqs:
            bad.append(f"{name}: scheduler.max_num_seqs 应为 {max_seqs}，"
                       f"实际 {c.max_num_seqs}")

        # ② 两处声明不得错配（scheduler 的批预算 vs 控制器的规划上限）
        if int(c.controller.max_batched_tokens_init) <= 0:
            bad.append(f"{name}: controller.max_batched_tokens_init 非法")

        # ③ 除声明的两处外，必须与底稿**逐字段相同**
        for k in ("target_tpot_ms", "target_ttft_ms", "hysteresis", "cooldown_cycles",
                  "signal_criterion", "signal_mode", "kind", "chunk_size_init",
                  "chunk_size_min", "chunk_size_max", "chunk_size_step",
                  "max_batched_tokens_init", "max_batched_tokens_min",
                  "max_batched_tokens_max", "max_batched_tokens_step",
                  "cost_model_path", "host_overhead_ms", "host_overhead_fit"):
            if getattr(c.controller, k, None) != getattr(ref.controller, k, None):
                bad.append(f"{name}: controller.{k} 与底稿不同（只应改 concurrency/max_num_seqs）")
        for k in ("name", "prompt_len_min", "prompt_len_max", "gen_len_min",
                  "gen_len_max", "arrival", "seed", "num_requests"):
            got = (c.workload_name if k == "name" else c.workload_overrides.get(k))
            want = (ref.workload_name if k == "name" else ref.workload_overrides.get(k))
            if got != want:
                bad.append(f"{name}: workload.{k} 与底稿不同（实际 {got} / 期望 {want}）")
        for k in ("repeats", "warmup_s", "control_interval_steps"):
            if getattr(c.run, k) != getattr(ref.run, k):
                bad.append(f"{name}: run.{k} 与底稿不同")
        if c.run.slo.ttft_ms != ref.run.slo.ttft_ms or c.run.slo.tpot_ms != ref.run.slo.tpot_ms:
            bad.append(f"{name}: SLO 与底稿不同（**SLO 在任何情况下都不得随扫描改**）")
        if c.chunk_policy_name != ref.chunk_policy_name or c.chunk != ref.chunk:
            bad.append(f"{name}: chunk 段与底稿不同（扫描批不测分块）")
        if c.runtime != ref.runtime:
            bad.append(f"{name}: runtime 段与底稿不同")

        # ④ 候选必在 `[min, max]` 内（否则被 ChunkLimits 夹回，白写）
        if c.controller.candidate_chunks:
            oob = [int(x) for x in c.controller.candidate_chunks
                   if not (c.controller.chunk_size_min <= int(x) <= c.controller.chunk_size_max)]
            if oob:
                bad.append(f"{name}: candidate_chunks 越界 {oob} ⇒ 会被 ChunkLimits 夹回")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 D1 低并发扫描配置（§7d）")
    ap.add_argument("--check", action="store_true", help="只校验、不写盘")
    args = ap.parse_args(argv)

    built = build()
    if args.check:
        bad = verify()
        if bad:
            print("✗ " + "；".join(bad), file=sys.stderr)
            return 1
        print(f"✓ {len(built)} 个扫描配置校验通过（{OUT_DIR}）")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in built.items():
        # ⚠️ 显式 newline='\n'：Windows 上默认会把 \n 翻成 \r\n（踩过）
        with open(OUT_DIR / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    bad = verify()
    if bad:
        print("✗ 写出后校验失败：" + "；".join(bad), file=sys.stderr)
        return 1
    print(f"已写出 {len(built)} 个配置到 {OUT_DIR}")
    for concurrency, max_seqs in cells():
        print(f"  {cell_name(concurrency, max_seqs):<32} "
              f"concurrency={concurrency}  max_num_seqs={max_seqs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
