#!/usr/bin/env python
"""派生执行优化线的 A/B 配置（`configs/mthreads/exec_opt/`）。

为什么**派生**而不是手写：单变量对照的前提是"只有 runtime 那几行不同"。
手写 9 份配置，改一次基线就要改 9 处 —— 漏一处就悄悄变成两变量对比，
而产物里看不出来。这里以 `configs/mthreads/s4000_fp16.yaml` 为**唯一**来源，
只在 `runtime:` 段插入开关，其余**逐字节保留**（含全部注释）。

同时把 `model_revision` **钉死**成实测解析到的 commit：
`main` 只是一个引用（今天指 A、明天可能指 B），不钉死就无法把两批评数
归到同一权重上。⚠️ 钉死之后**必须**在真机上确认该 snapshot 已在缓存里
（否则会走网络或直接失败）——这一点写进每个文件头，不靠人记。

用法::

    python scripts/gen_exec_opt_configs.py            # 生成/覆盖
    python scripts/gen_exec_opt_configs.py --check     # 只校验现有文件是否与派生物一致
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
BASE = _ROOT / "configs" / "mthreads" / "s4000_fp16.yaml"
OUTDIR = _ROOT / "configs" / "mthreads" / "exec_opt"

#: 项目记录并已实测解析到的权重 commit（`handoff/.../01_PROJECT_PLAN.md` §2）。
PINNED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

#: `名称 -> (开关 dict, 说明)`。开关值为 YAML 字面量（布尔/数字/列表）。
VARIANTS: dict[str, tuple[dict[str, str], str]] = {
    # ★★ S01 复核的**参照臂**（2026-10-03 加）。
    # 为什么需要它：S01 的对照必须是"**同一条执行路径**、只差两个开关"。
    # `s4000_fp16.yaml` 是平台基线，它的 `model_revision` 是 `main`（引用），
    # 而 `s01_both.yaml` 里是**钉死的 commit** ⇒ 直接拿它们比会多出一个差异
    # （`main` 今天指向哪里），那就不是单变量了。
    # ⇒ 本臂 = 同一个基线 + **同一个钉死 revision** + **两个开关都不开**。
    #   于是与 `s01_both` 的有效差异**恰好**是两个开关（有程序化核对）。
    # ⚠️ 它**不改** `s4000_fp16.yaml`（历史基线逐字节不变）。
    "s01_reference": (
        {},
        "P0：S01 **参照臂**（两个开关全关）；只把 model_revision 钉死成与 s01_both 同一个 commit"
        " ⇒ 与 s01_both 的有效差异恰好是两个 S01 开关"),
    "s01_last_only": (
        {"logits_last_only": "true"},
        "P1-a：只算最后位置的 logits（`num_logits_to_keep=1`）"),
    "s01_sample_needed": (
        {"sample_needed_only": "true"},
        "P1-b：只对 need_token=True 的行采样/回传"),
    "s01_both": (
        {"logits_last_only": "true", "sample_needed_only": "true"},
        "P1-a+b：两开关组合"),
    "s02_no_empty_cache": (
        {"retire_empty_cache": "false"},
        "P2a：退役时不清 allocator 缓存（改为 run 边界显式 flush）"),
    "s03_fuse_qkv": (
        {"fuse_qkv": "true"},
        "P3-a：QKV 3→1 融合"),
    "s03_fuse_gate_up": (
        {"fuse_gate_up": "true"},
        "P3-b：gate/up 2→1 融合"),
    "s03_fuse_both": (
        {"fuse_qkv": "true", "fuse_gate_up": "true"},
        "P3-a+b：两个融合组合"),
    "s03_fuse_both_shapes": (
        {"fuse_qkv": "true", "fuse_gate_up": "true",
         "linear_fuse_shapes": "[1, 2, 4, 8]"},
        "P3-a+b + 形状选择：只在 M ∈ {1,2,4,8} 时融合，prefill 走原路径"),
    # ---- S04/P2b-c：常驻 KV 缓冲 ----
    # ⚠️ **必须显式给 `persistent_kv_capacity`**：默认值是 `max_seq_len`（32768），
    #    每槽位要 28 层 × 2(K/V) × 4 head × 32768 × 128 × 2B ≈ 1.9 GB，
    #    并发 8 就是 15 GB —— 能跑，但那是把"预分配容量"这个代价放到最大，
    #    与"测 fast-path 本身是否省搬运"混在一起。这里按负载实需取 1024
    #    （`short_chat` 实测 `prompt_len+gen_len ≤ 950`），每槽位 ≈117 MB。
    "s04_pkv": (
        {"persistent_kv_slots": "true", "persistent_kv_capacity": "1024"},
        "P2b-c：常驻 KV 缓冲（纯 decode + 历史等长走快路径，容量 1024）"),
    # ★ 组合臂：**在已冻结的 `s01_both` 基线上**叠加 S04。
    # 为什么必须有它：`s01_both` 已被裁定为后续批次的默认基线（TTFT −4.17%，
    # 36× 噪声底），因此"S04 的净收益"必须在**该基线上**测量，而不是在裸基线上。
    # 只跑 `s04_pkv` vs `s4000_fp16` 会把 S01 的效果混进来（两变量）。
    "s01_both_s04_pkv": (
        {"logits_last_only": "true", "sample_needed_only": "true",
         "persistent_kv_slots": "true", "persistent_kv_capacity": "1024"},
        "P1-a+b + P2b-c：冻结基线上叠加常驻 KV 缓冲"),
}

_HEADER = """# ⚠️ **派生文件，请勿手改** —— 由 scripts/gen_exec_opt_configs.py 生成。
# 来源：configs/mthreads/s4000_fp16.yaml（唯一来源；改基线请改它再重新生成）
# 开关：{desc}
# 模型 revision 已**钉死**为 {rev}
#   （`main` 是引用不是版本；不钉死就无法把两批评数归到同一权重上）
#   ⚠️ 真机前置：确认该 snapshot 已在本地 HF 缓存内，否则会走网络或直接失败。
# 用法：python -m benchmark.run_benchmark --config {rel} --dry-run
"""


def _derive(base_text: str, overrides: dict[str, str], *, desc: str, rel: str) -> str:
    lines = base_text.splitlines()
    # 1) 钉死 revision：只改 `model_revision:` 那一行的值，保留其余行与注释。
    out: list[str] = []
    pinned = False
    for ln in lines:
        if ln.strip().startswith("model_revision:") and not pinned:
            indent = ln[: len(ln) - len(ln.lstrip())]
            out.append(f'{indent}model_revision: "{PINNED_REVISION}"')
            pinned = True
            continue
        out.append(ln)
    if not pinned:
        raise SystemExit("基线里找不到 `model_revision:` 行 —— 派生前提不成立")

    # 2) 在 `runtime:` 段头之后插入开关。插在**段头紧后面**而不是段尾，
    #    是为了让"这一行是本次派生的差异"一眼可见。
    idx = next((i for i, ln in enumerate(out) if ln.startswith("runtime:")), None)
    if idx is None:
        raise SystemExit("基线里找不到顶层 `runtime:` 段 —— 派生前提不成立")
    injected = [f"  {k}: {v}" for k, v in overrides.items()]
    merged = out[: idx + 1] + injected + out[idx + 1:]
    body = "\n".join(merged).rstrip("\n") + "\n"
    return _HEADER.format(desc=desc, rev=PINNED_REVISION, rel=rel) + body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="只校验：现有文件必须与派生物逐字节一致")
    args = ap.parse_args()

    base_text = BASE.read_text(encoding="utf-8")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    for name, (ov, desc) in VARIANTS.items():
        rel = f"configs/mthreads/exec_opt/{name}.yaml"
        text = _derive(base_text, ov, desc=desc, rel=rel)
        target = OUTDIR / f"{name}.yaml"
        if args.check:
            if not target.exists():
                problems.append(f"{target.name}: 缺失")
            elif target.read_text(encoding="utf-8") != text:
                problems.append(f"{target.name}: 与派生物不一致（有人手改过？）")
            continue
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"写出 {target.relative_to(_ROOT)}")

    if args.check:
        if problems:
            print("不一致：")
            for p in problems:
                print("  -", p)
            return 1
        print(f"通过：{len(VARIANTS)} 份派生配置与来源一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
