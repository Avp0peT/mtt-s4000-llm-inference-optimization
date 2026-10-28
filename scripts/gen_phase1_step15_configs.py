#!/usr/bin/env python3
"""生成**第 15 步消融**的配置集（`configs/mthreads/phase1_step15/`）。

为什么消融必须是**生成**的、而且从"完整系统"派生
--------------------------------------------------
消融的整个价值在于**单变量**：每条只改一个维度，其余逐字节相同。
手抄 6 份配置一定会漂（实测踩过的形态：抄的时候顺手改了别的字段，
于是"消融的差异"里混进了无关改动，结论就归因错了）。

⇒ 本生成器把"完整系统"（`reference`）当作**唯一来源**，逐条只施加**一处**声明的变换，
并由 `tests/test_step15_configs.py` 断言"与 reference 的差异恰好是声明的那一处"。

消融清单与依据（`paper/experiments.md` §5 第 7 条 + 本文件补的第 5 条）
---------------------------------------------------------------------
| 条件 | 改什么 | 它回答的问题 |
| --- | --- | --- |
| `full` | —（reference） | 第 14 步落地后的完整系统 |
| `no_host` | 去掉 `host_overhead_ms` / `host_overhead_fit` | "判据从'前向 ≤ 预算'改成'整步 ≤ 预算'"**这一项**值多少 D1？ |
| `no_margin` | `apply_safety_margin: false` | 误差裕量是否真的买到东西（还是只让控制器更保守）？ |
| `signal_tpot` | `signal_criterion: tpot_p90` | 换回**会被 prefill 污染**的读数会怎样（第 09 步冻结 `decode_p90` 的反面对照） |
| `chunk_only` | 冻结单步 token 预算（min=max=init） | 只动分块这一维够不够 |
| `budget_only` | 冻结分块（min=max=init） | 只动预算这一维够不够 |

⚠️ **`no_host` 的读法**：`host_overhead_ms` 是**本机 + 当前执行路径**的量（含 python 侧
打包/构造/mask 组装），**不是**芯片或模型的性质。所以这一条消融测的是"判据口径"，
**不能**被写成"host 开销是芯片特性"。

⚠️ **`chunk_only` / `budget_only` 刻意让一维不可动**（`min == max == init`）。
这与"控制器维度必须可动"的自检**方向相反**，是**有意的**：这两条要回答的正是
"把这一维拿掉会怎样"。它们**不得**被任何"可动性自检"当成坏配置——
自检若覆盖本目录，必须按 `_FROZEN_DIM` 显式豁免，而不是把检查放宽。

用法::

    python scripts/gen_phase1_step15_configs.py            # 写盘
    python scripts/gen_phase1_step15_configs.py --check     # 只校验、不写
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

REFERENCE = _ROOT / "configs" / "mthreads" / "phase1_step14b" / "fixed_cost.yaml"
OUT_DIR = _ROOT / "configs" / "mthreads" / "phase1_step15"

#: 每条消融：名字 → (说明, "只施加一处变换"的函数)
#: ⚠️ 变换一律**按行**做，保持未触及的行逐字节不变（含缩进与注释）。


def _drop_block(lines: list[str], key: str) -> list[str]:
    """删掉 `  key:` 这一行及其后续更深的缩进行（含列表项 `  - x`）。"""
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.rstrip("\n").strip().startswith(f"{key}:"):
            # 本行是一级键（2 空格缩进）⇒ 吞掉它与其后缩进更深的行
            base = len(ln) - len(ln.lstrip())
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    break
                ind = len(nxt) - len(nxt.lstrip())
                if ind <= base and not nxt.lstrip().startswith("-"):
                    break
                i += 1
            continue
        out.append(ln)
        i += 1
    return out


def _set_scalar(lines: list[str], key: str, value: str) -> list[str]:
    out: list[str] = []
    hit = 0
    for ln in lines:
        if ln.rstrip("\n").strip().startswith(f"{key}:"):
            indent = ln[: len(ln) - len(ln.lstrip())]
            out.append(f"{indent}{key}: {value}\n")
            hit += 1
        else:
            out.append(ln)
    if hit != 1:
        raise SystemExit(f"✗ 期望 {key} 恰好出现 1 次，实际 {hit} 次（配置结构变了？）")
    return out


def _insert_under_controller(lines: list[str], text: str) -> list[str]:
    """在 `controller:` 段的**末尾**插入一行（缩进 2）。"""
    start = next(i for i, ln in enumerate(lines) if ln.rstrip("\n") == "controller:")
    end = start + 1
    while end < len(lines) and (not lines[end].strip()
                                or len(lines[end]) - len(lines[end].lstrip()) > 0):
        end += 1
    return lines[:end] + [f"  {text}\n"] + lines[end:]


def _identity(lines: list[str]) -> list[str]:
    return list(lines)


def _no_host(lines: list[str]) -> list[str]:
    return _drop_block(_drop_block(lines, "host_overhead_ms"), "host_overhead_fit")


def _no_margin(lines: list[str]) -> list[str]:
    return _insert_under_controller(lines, "apply_safety_margin: false")


def _signal_tpot(lines: list[str]) -> list[str]:
    return _set_scalar(lines, "signal_criterion", "tpot_p90")


def _chunk_only(lines: list[str]) -> list[str]:
    # 冻结**预算**维 ⇒ 只留分块可动
    return _set_scalar(_set_scalar(lines, "max_batched_tokens_min", "8192"),
                       "max_batched_tokens_max", "8192")


def _budget_only(lines: list[str]) -> list[str]:
    # 冻结**分块**维 ⇒ 只留预算可动
    return _set_scalar(_set_scalar(lines, "chunk_size_min", "1024"),
                       "chunk_size_max", "1024")


CONDITIONS: dict[str, tuple[str, object]] = {
    "full": ("完整系统（reference，逐字节等于 phase1_step14b/fixed_cost.yaml）", _identity),
    "no_host": ("消融：去掉逐步 host 开销项（判据退回'前向 ≤ 预算'）", _no_host),
    "no_margin": ("消融：不乘安全余量（apply_safety_margin=false）", _no_margin),
    "signal_tpot": ("消融：判据换回会被 prefill 污染的 tpot_p90", _signal_tpot),
    "chunk_only": ("消融：冻结单步 token 预算 ⇒ 只动分块", _chunk_only),
    "budget_only": ("消融：冻结分块 ⇒ 只动预算", _budget_only),
}

#: 刻意让一维不可动的条件 → 该维名字。可动性自检必须按**这个**豁免，而不是放宽检查。
_FROZEN_DIM: dict[str, str] = {
    "chunk_only": "max_batched_tokens",
    "budget_only": "chunk_size",
}


_MIN_OLD = "  chunk_size_min: 512\n"


#: CLI 覆盖项（`main()` 里设置）。默认**全为 None** ⇒ 生成结果与既有 `phase1_step15/`
#: **逐字节一致**（被 `tests/test_step15_configs.py` 钉住）。带覆盖时**必须**换输出目录，
#: 否则会改到被用例断言的目录上 ⇒ 弄红回归。
_OVERRIDES: dict[str, object] = {"model": None, "min": None, "out_dir": None}


def _apply_overrides(text: str) -> str:
    """把 `--model` / `--min` 施加到**底稿**上（先于按条件的变换）。"""
    m = _OVERRIDES.get("model")
    if m:
        old = "cost_model_path: configs/cost_calib/model_v6.json"
        if old not in text:
            raise SystemExit("✗ 底稿里没有 model_v6.json 行（已改过？）")
        text = text.replace(old, f"cost_model_path: {m}")
    mn = _OVERRIDES.get("min")
    if mn is not None:
        if _MIN_OLD not in text:
            raise SystemExit("✗ 底稿里没有 chunk_size_min 行（已改过？）")
        text = text.replace(_MIN_OLD, f"  chunk_size_min: {int(mn)}\n")
    return text


def build() -> dict[str, str]:
    if not REFERENCE.exists():
        raise SystemExit(f"✗ 找不到 reference：{REFERENCE}")
    src = _apply_overrides(REFERENCE.read_text(encoding="utf-8"))
    lines = src.splitlines(keepends=True)
    out: dict[str, str] = {}
    for name, (_why, fn) in CONDITIONS.items():
        out[f"fixed_cost_{name}.yaml"] = "".join(fn(lines))          # type: ignore[operator]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成第 15 步消融配置")
    ap.add_argument("--check", action="store_true", help="只校验、不写盘")
    # ---- 参数化（D1② 之后要把消融钉在**选出的下限 + 最终模型**上）------------
    # ⚠️ 带覆盖时**必须**同时给 `--out-dir`，否则会改到被 `tests/test_step15_configs.py`
    # 逐字节断言的目录上 ⇒ 弄红回归。默认（都不给）时生成结果与既有文件一致。
    ap.add_argument("--model", default=None,
                    help="替换 cost_model_path（如 configs/cost_calib/model_v7.json）")
    ap.add_argument("--min", type=int, default=None, dest="min_",
                    help="替换 chunk_size_min（D1② 按预注册规则选出的值）")
    ap.add_argument("--out-dir", default=None, help="输出目录（带覆盖时必填）")
    args = ap.parse_args(argv)

    if (args.model or args.min_ is not None or args.out_dir) and not args.out_dir:
        print("✗ 给了 --model/--min 就必须同时给 --out-dir"
              "（否则会覆盖被用例逐字节钉住的 phase1_step15/）", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    if args.out_dir and (args.model is None and args.min_ is None):
        print("✗ 只给 --out-dir 而没有 --model/--min ⇒ 那只是复制一份，容易造成第二处真相",
              file=sys.stderr)
        return 2
    _OVERRIDES.update({"model": args.model, "min": args.min_, "out_dir": args.out_dir})

    built = build()
    if args.check:
        bad = []
        for name, text in built.items():
            p = out_dir / name
            if not p.exists():
                bad.append(f"缺文件 {name}")
            elif p.read_text(encoding="utf-8") != text:
                bad.append(f"内容不符 {name}（跑一次不带 --check 重新生成）")
        if bad:
            print("✗ " + "；".join(bad), file=sys.stderr)
            return 1
        print(f"✓ {len(built)} 个第 15 步消融配置与 reference 一致（{out_dir}）")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in built.items():
        # ⚠️ 显式 newline='\n'：Windows 上默认会把 \n 翻成 \r\n（踩过）
        with open(out_dir / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print(f"  写出 {name:<28} {CONDITIONS[name.split('fixed_cost_')[1][:-5]][0]}")
    print(f"\n已写出 {len(built)} 个配置到 {out_dir}")
    if args.model or args.min_ is not None:
        print(f"  覆盖：model={args.model or '(不变)'} min={args.min_ if args.min_ is not None else '(不变)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
