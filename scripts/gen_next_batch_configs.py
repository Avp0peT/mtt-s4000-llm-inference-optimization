#!/usr/bin/env python
"""派生**下一批（机制小批）**所需的配置（`configs/mthreads/next_batch/`）。

★ 为什么只派生**两份**（2026-10-03，任务 E）
--------------------------------------------
机制小批要的 5 条臂里，**3 条已经存在**，再去复制一份就等于制造第二处真相：

| 需要的臂 | 从哪来 | 是否需要新文件 |
| --- | --- | --- |
| `static_nochunk` | `phase1_step14/fixed_static.yaml`（`chunk.enabled=false` + 控制器关） | ❌ **直接复用** |
| `threshold` | `phase1_step14/fixed_threshold.yaml` | ❌ **直接复用** |
| `cost_single`（单预算） | `phase1_step14/fixed_cost.yaml`（无 `prefill_deadline_ms` ⇒ 单预算） | ❌ **直接复用** |
| `static_small`（静态小 chunk） | `fixed_static.yaml` + 开分块、`chunk_size=1024` | ✅ **新建** |
| `cost_dual`（修正后的双预算） | `fixed_cost.yaml` + `prefill_deadline_ms` | ✅ **新建** |

⚠️ **为什么 `static_small` 的 `chunk_size` 取 1024**：这是三条自适应臂的
`chunk_size_init`（`fixed_cost/fixed_threshold` 都是 1024）。取同一个值才能问出
"控制器的收益"而不是"换了个 chunk 大小"——**静态基线必须与自适应臂同起点**。

⚠️ **为什么 `prefill_deadline_ms` 取 2000**：它与 `run.slo.ttft_ms` **同源**
（服务的是同一个承诺：TTFT）。不一致就变成"偷偷改了一个 SLO 口径"。
⚠️ `step_deadline_ms` **不显式写**：它走 `CostPolicyConfig` 的默认 `100.0`，
与 `fixed_cost.yaml` 完全一致 ⇒ 两个 cost 臂**只差 `prefill_deadline_ms` 一项**。

⚠️ **Best Static 只能在校准数据上选**：本文件不选、也不建议选哪个静态基线；
`static_nochunk` 与 `static_small` 是**两个候选**，选谁必须由校准数据（不是测试集）决定，
并把选择过程落盘。派生出这两份**不等于**已经选好。

用法::

    python scripts/gen_next_batch_configs.py            # 生成/覆盖
    python scripts/gen_next_batch_configs.py --check     # 只校验一致性
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = _ROOT / "configs" / "mthreads" / "phase1_step14"
OUTDIR = _ROOT / "configs" / "mthreads" / "next_batch"

#: 与自适应臂同起点（见模块 docstring）。
STATIC_SMALL_CHUNK = 1024
#: 与 `run.slo.ttft_ms` 同源。
PREFILL_DEADLINE_MS = 2000.0

_HEADER = """# ⚠️ **派生文件，请勿手改** —— 由 scripts/gen_next_batch_configs.py 生成。
# 来源：configs/mthreads/phase1_step14/{src}（**冷冻的**第 14 步配置；
#       派生只改下面写明的那一项，其余逐字节保留）
# 差异：{desc}
# 用法：python -m benchmark.run_benchmark --config {rel} --dry-run
# ⚠️ 本文件**只**为机制小批准备；它不是"已经跑过"的证据，也不含任何测量值。
"""


def _set_scalar(lines: list[str], key: str, value: str, *,
                section: str) -> list[str]:
    """在 `section:` 段内把 `key:` 的值改成 `value`（保留缩进与行尾注释之外的内容）。

    ⚠️ 找不到就在**段头之后**插入：静默不生效会让"派生配置看起来改了、其实没改"，
    而那正是本仓最怕的形态（产物看不出区别）。
    """
    out: list[str] = []
    in_sec = False
    sec_indent = 0
    done = False
    for ln in lines:
        stripped = ln.strip()
        if ln.startswith(f"{section}:"):
            in_sec = True
            sec_indent = 0
            out.append(ln)
            continue
        if in_sec and stripped and not ln.startswith((" ", "\t")):
            # 离开该段
            if not done:
                out.append(f"  {key}: {value}")
                done = True
            in_sec = False
            out.append(ln)
            continue
        if in_sec and stripped.startswith(f"{key}:"):
            indent = ln[: len(ln) - len(ln.lstrip())]
            out.append(f"{indent}{key}: {value}")
            done = True
            continue
        out.append(ln)
    if in_sec and not done:            # 该段是文件最后一段
        out.append(f"  {key}: {value}")
        done = True
    if not done:
        raise SystemExit(f"段 `{section}:` 不存在 ⇒ 派生前提不成立")
    return out


def _derive(src_name: str, *, desc: str, rel: str,
            edits: list[tuple[str, str, str]]) -> str:
    """`edits` = [(段, 键, 值)]，按顺序应用。"""
    text = (SRC_DIR / src_name).read_text(encoding="utf-8")
    lines = text.splitlines()
    for section, key, value in edits:
        lines = _set_scalar(lines, key, value, section=section)
    body = "\n".join(lines).rstrip("\n") + "\n"
    return _HEADER.format(src=src_name, desc=desc, rel=rel) + body


VARIANTS: dict[str, tuple[str, str, list[tuple[str, str, str]]]] = {
    # 名称 -> (来源文件, 差异说明, 编辑)
    "static_small": (
        "fixed_static.yaml",
        f"开分块 + `chunk.chunk_size: {STATIC_SMALL_CHUNK}`（其余同 frozen static：控制器关闭）",
        [("chunk", "enabled", "true"),
         ("chunk", "chunk_size", str(STATIC_SMALL_CHUNK))],
    ),
    "cost_dual": (
        "fixed_cost.yaml",
        f"新增 `controller.prefill_deadline_ms: {PREFILL_DEADLINE_MS:g}`"
        "（= 与 `run.slo.ttft_ms` 同源）⇒ 启用**修正后**的双预算；"
        "`step_deadline_ms` 不写，走默认 100.0 ⇒ 与 cost_single 只差这一项",
        [("controller", "prefill_deadline_ms", f"{PREFILL_DEADLINE_MS:g}")],
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="只校验：现有文件必须与派生物逐字节一致")
    args = ap.parse_args()

    if not SRC_DIR.is_dir():
        print(f"[gen-next-batch] ✗ 来源目录不存在：{SRC_DIR}", file=sys.stderr)
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    for name, (src, desc, edits) in VARIANTS.items():
        rel = f"configs/mthreads/next_batch/{name}.yaml"
        text = _derive(src, desc=desc, rel=rel, edits=edits)
        target = OUTDIR / f"{name}.yaml"
        if args.check:
            if not target.exists():
                problems.append(f"{target.name}: 缺失")
            elif target.read_text(encoding="utf-8") != text:
                problems.append(f"{target.name}: 与派生物不一致（有人手改过？）")
            continue
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"写出 {target.relative_to(_ROOT)}（来源 {src}）")

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
