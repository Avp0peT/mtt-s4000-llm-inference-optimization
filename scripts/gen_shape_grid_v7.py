#!/usr/bin/env python3
"""生成 `configs/cost_calib/shape_grid_v7.yaml` = **v6 + 小 chunk 组**。

为什么 v7 要**重采**（而不是只补新点）
--------------------------------------
① `export_cost_model.py` 要求所有产物的 `protocol_version` 与 `grid_sha256` **各自唯一**
   ⇒ 想让"新点"与"旧点"进同一个模型，就必须共用**同一份网格文件**；
② 更根本的：**部署换了源码树**（D2）⇒ 标定必须**对着部署后的执行路径**重做，
   否则"标定时的执行路径"与"实验时的执行路径"不是同一个 —— 那正是本项目反复吃亏的那类口径错。
   ⇒ 所以重采不是浪费，是**有效性要求**；顺带还多一次跨会话漂移的测量。

⚠️ **不要**靠"把新组放到文件第一位 + `--limit`"来采子集 —— 那条路是错的
--------------------------------------------------------------------
`costcalib_grid.expand()` 返回的是 **按 `as_key` 排序**的列表，
**不是**网格文件里的顺序（`sorted(out.values(), key=lambda s: s.as_key())`）。
⇒ `--limit N` 取到的永远是排序最前的那些点（`pure_decode`），**当不了分组过滤**。
（实测踩到：我先按"放第一位"改了一版，校验时发现前 16 点全是 `pure_decode`。）
⇒ 采子集请用 `scripts/fwd_costcalib_probe.py --only-group <id>`；
   子集信息会写进产物的 `subset` 字段（`is_full_grid: false`），避免被当成一次完整会话。

用法::

    python scripts/gen_shape_grid_v7.py            # 写盘
    python scripts/gen_shape_grid_v7.py --check     # 只校验、不写
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SRC = _ROOT / "configs" / "cost_calib" / "shape_grid_v6.yaml"
DST = _ROOT / "configs" / "cost_calib" / "shape_grid_v7.yaml"

_OLD_PROTOCOL = "protocol_version: cost-calib-v6"
_NEW_PROTOCOL = "protocol_version: cost-calib-v7"

_OLD_AXIS = ("  prefill_tok: [0, 512, 1024, 2048]   # 0 = 纯 decode 步；"
             "512/1024/2048 与 chunk_size 取值对齐")
_NEW_AXIS = ("  prefill_tok: [0, 128, 256, 512, 1024, 2048]   # 0 = 纯 decode；"
             "v7 新增 128/256（D1 的小 chunk 验证要求它们在 lookup 下有键）")

_NEW_GROUP_ID = "mixed_small_chunk"
#: 该组的点数（4 batch × 2 prefill × 2 kv）—— `verify()` 会核对，防止悄悄改大改小。
_GROUP_N = 16

_NEW_GROUP = f"""  - id: {_NEW_GROUP_ID}
    why: "**D1 的小 chunk 验证**：判据是「预测整步 ≤ 每步预算」，而 512 的 prefill 步在该预算下
      预测超限；若候选集里没有更小的块，控制器只能在「降级取最省」与「回退阈值」之间走。
      本组为「小候选能不能被预测」提供**实测键** —— `lookup` 下没有键就等于不能预测。
      ⚠️ 采它请用 `fwd_costcalib_probe.py --only-group {_NEW_GROUP_ID}`
      （**不要**用 `--limit`：`expand()` 是按 key 排序的，`--limit` 取不到这一组）。"
    batch: [1, 2, 4, 8]
    prefill_tok: [128, 256]
    kv_len: [1024, 2048]
    n_forwards: [1]
    n_new_mode: real_mixed

"""


def build() -> str:
    src = SRC.read_text(encoding="utf-8")
    if _OLD_PROTOCOL not in src:
        raise SystemExit(f"✗ {SRC.name} 里没有 {_OLD_PROTOCOL!r}（已改过？请改本脚本）")
    out = src.replace(_OLD_PROTOCOL, _NEW_PROTOCOL)
    if _OLD_AXIS not in out:
        raise SystemExit("✗ axes.prefill_tok 行与预期不符（已改过？请改本脚本）")
    out = out.replace(_OLD_AXIS, _NEW_AXIS)
    marker = "\ngroups:\n"
    if marker not in out:
        raise SystemExit("✗ 找不到 `groups:` 段头")
    i = out.index(marker) + len(marker)
    return out[:i] + _NEW_GROUP + out[i:]


def verify(path: Path) -> list[str]:
    """校验：protocol、新组存在且形状正确、点数与 `_GROUP_N` 一致。"""
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT / "scripts"))
    from costcalib_grid import expand, load_grid  # noqa: E402

    bad: list[str] = []
    grid = load_grid(path)
    shapes = list(expand(grid))
    if grid.get("protocol_version") != "cost-calib-v7":
        bad.append(f"protocol_version 应为 cost-calib-v7，实际 {grid.get('protocol_version')!r}")
    new = [s for s in shapes if s.group == _NEW_GROUP_ID]
    if len(new) != _GROUP_N:
        bad.append(f"`{_NEW_GROUP_ID}` 有 {len(new)} 点，期望 {_GROUP_N}（改了组就要同步 `_GROUP_N`）")
    if not all(s.batch in (1, 2, 4, 8) and s.prefill_tok in (128, 256)
               and s.kv_len in (1024, 2048) for s in new):
        bad.append("新组里有超出声明范围的形状")
    if not all(s.n_new_mode == "real_mixed" for s in new):
        bad.append("新组必须全是 real_mixed —— 否则小 chunk 的成本仍按旧口径被放大 ~b 倍")
    if not any(s.prefill_tok in (128, 256) and s.batch > 1 for s in shapes):
        bad.append("网格里没有 (pf∈{128,256}, batch>1) 的点 ⇒ 采了也验证不了小 chunk 的 mixed 形状")
    if not (SRC.exists() and "v6" in SRC.name):
        bad.append("参照网格不是 v6 —— 本脚本假定以 v6 为底")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成/校验 v7 标定网格")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    if args.check:
        if not DST.exists():
            print(f"✗ 缺 {DST}", file=sys.stderr)
            return 1
        bad = verify(DST)
        if bad:
            print("✗ " + "；".join(bad), file=sys.stderr)
            return 1
        print(f"✓ {DST.name} 校验通过（含 `{_NEW_GROUP_ID}` {_GROUP_N} 点、全 real_mixed）")
        return 0

    text = build()
    with open(DST, "w", encoding="utf-8", newline="\n") as fh:   # ⚠️ 显式 LF
        fh.write(text)
    bad = verify(DST)
    if bad:
        print("✗ 写出后校验失败：" + "；".join(bad), file=sys.stderr)
        return 1
    print(f"已写出 {DST}")
    print(f"  采子集请用：--only-group {_NEW_GROUP_ID}（{_GROUP_N} 点）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
