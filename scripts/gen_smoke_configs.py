#!/usr/bin/env python
"""派生**短启动/数值验收**用的极小配置（`configs/mthreads/_smoke/`）。

★ 为什么需要它（2026-10-03，S01 复核的开跑前置）
------------------------------------------------
用户要求"**先完成短启动和必要的数值正确性验收**；失败则保留证据、停止依赖实验，
不带病运行十项批次"。

对 S01 而言，**必要的数值验收**是什么？S01 的两个开关
（`logits_last_only` / `sample_needed_only`）动的正是**取 logits / 采样**这条路
⇒ 它们**可能改变输出 token**。所以验收判据是：

    同一负载、两个臂，**产出的 token 序列必须逐位相同**。

⚠️ 为什么不能用 `--dry-run`：dry-run 走模拟器，**根本不碰这两个开关**
（配置头里就写着"这些开关在 --dry-run 下不生效"）⇒ 用模拟路径验收等于没验。
必须**真机短跑**。

⚠️ 本目录的配置**不是批次配置**：负载被改到极小（只为快），
**不得**用于任何对照实验；生成器每次覆盖，手工改动会被 `--check` 抓出。

用法::

    python scripts/gen_smoke_configs.py            # 生成/覆盖
    python scripts/gen_smoke_configs.py --check     # 只校验一致性
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = _ROOT / "configs" / "mthreads" / "exec_opt"
OUTDIR = _ROOT / "configs" / "mthreads" / "_smoke"

#: 极小负载：够走到两个开关的代码路径，又能在秒级跑完。
SMOKE_WORKLOAD = [
    "workload:",
    "  name: short_chat",
    "  # ⚠️ 极小负载（短启动/数值验收专用）；**不是**批次负载。",
    "  num_requests: 4",
    "  concurrency: 2",
    "  prompt_len_min: 64",
    "  prompt_len_max: 64",
    "  gen_len_min: 8",
    "  gen_len_max: 8",
]

_HEADER = """# ⚠️ **派生文件，请勿手改** —— 由 scripts/gen_smoke_configs.py 生成。
# 来源：configs/mthreads/exec_opt/{src}
# 用途：**短启动 + 数值正确性验收**（不是批次配置！负载被改到极小）。
# 判据：两臂产出的 **token 序列逐位相同**（S01 的开关动的正是取 logits / 采样）。
# ⚠️ 这些开关在 `--dry-run` 下**不生效** ⇒ 必须真机短跑，模拟路径验不出来。
"""


def _derive(src_name: str) -> str:
    text = (SRC_DIR / src_name).read_text(encoding="utf-8")
    lines = text.splitlines()
    # 找到 `workload:` 段并用极小负载整体替换（该段只有两行，替换最不容易出错）
    try:
        start = lines.index("workload:")
    except ValueError:
        raise SystemExit(f"{src_name} 里找不到顶层 `workload:` 段 —— 派生前提不成立")
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t")) or not lines[end]):
        end += 1
    body = "\n".join(lines[:start] + SMOKE_WORKLOAD + lines[end:]).rstrip("\n") + "\n"
    return _HEADER.format(src=src_name) + body


VARIANTS = ("s01_reference.yaml", "s01_both.yaml")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    for name in VARIANTS:
        text = _derive(name)
        target = OUTDIR / name.replace(".yaml", "_smoke.yaml")
        if args.check:
            if not target.exists():
                problems.append(f"{target.name}: 缺失")
            elif target.read_text(encoding="utf-8") != text:
                problems.append(f"{target.name}: 与派生物不一致（有人手改过？）")
            continue
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"写出 {target.relative_to(_ROOT)}（来源 {name}）")
    if args.check:
        if problems:
            print("不一致：")
            for p in problems:
                print("  -", p)
            return 1
        print(f"通过：{len(VARIANTS)} 份短启动配置与来源一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
