#!/usr/bin/env python3
"""生成 `configs/cost_calib/shape_grid_v8.yaml` = **v7 + pf192**（小 chunk 的第三个点）。

为什么要有 v8（而不是原地改 v7）
---------------------------------
`OPTIMIZATION_ROADMAP_SESSIONS.md` §5「B4」（= 会话 OPT-03）点名要补 **128/192/256**
三个小 chunk 采样点。OPT-03 前置核查实测：

    model_v7.json 的 `lookup_table` 里  pf128 **8** 键 / pf256 **8** 键 / **pf192 = 0** 键
    （v7 数据的 `prefill_tok` 轴 = [0,128,256,512,1024,2048]，**192 从未出现**）

⇒ 128/256 在 v7 已可预测（B4 已证：两会话 32 条、重复性中位 0.07%、部署表 100% 命中）；
   缺的**只有** 192。v8 只做这一件事。

⚠️ **不能原地改 v7**，两条硬理由：
① 网格自己的纪律（`shape_grid_v1.yaml` 文件头）：
   "看到结果之后再加密采样点，就会把'事后挑形状'混进来 ⇒ 要加密必须另开一轮（v2）并说明"。
② `export_cost_model.py` 要求所有产物的 `protocol_version` 与 `grid_sha256` **各自唯一**
   ⇒ 原地改 v7 会让 `model_v7.json` 记的 `grid_sha256` 与网格文件失配 ⇒ 血缘断。

⚠️ 与 `gen_shape_grid_v7.py` 的关系：本脚本以 **v7** 为底（那一个以 v6 为底）。
   两者都刻意**只做加法制**改动，且都用 `--check` 把不变量写成可执行断言。

用法::

    python scripts/gen_shape_grid_v8.py            # 写盘 + 校验
    python scripts/gen_shape_grid_v8.py --check     # 只校验、不写
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SRC = _ROOT / "configs" / "cost_calib" / "shape_grid_v7.yaml"
DST = _ROOT / "configs" / "cost_calib" / "shape_grid_v8.yaml"

_OLD_PROTOCOL = "protocol_version: cost-calib-v7"
_NEW_PROTOCOL = "protocol_version: cost-calib-v8"

_OLD_AXIS = ("  prefill_tok: [0, 128, 256, 512, 1024, 2048]   # 0 = 纯 decode；"
             "v7 新增 128/256（D1 的小 chunk 验证要求它们在 lookup 下有键）")
_NEW_AXIS = ("  prefill_tok: [0, 128, 192, 256, 512, 1024, 2048]   # 0 = 纯 decode；"
             "v7 新增 128/256；**v8 新增 192**（B4 要求 128/192/256 齐，v7 缺 192）")

#: 小 chunk 组里 `prefill_tok` 的**旧值与新值**（只改这一行的一处）。
_OLD_GROUP_PF = "    prefill_tok: [128, 256]\n    kv_len: [1024, 2048]"
_NEW_GROUP_PF = "    prefill_tok: [128, 192, 256]\n    kv_len: [1024, 2048]"

_GROUP_ID = "mixed_small_chunk"
#: 该组的点数（4 batch × 3 prefill × 2 kv）—— `verify()` 会核对，防止悄悄改大改小。
_GROUP_N = 24

#: v8 变更头（采数**之前**登记）——插到文件最前。
_HEADER = """# 成本模型标定的形状网格（**协议 v8**）—— **补小 chunk 的第三个点 pf192**
#
# ⚠️ **v8 的变更理由**（采数**之前**登记，由 OPT-03/B4 的缺口指出、不是我猜的）：
#
# `OPTIMIZATION_ROADMAP_SESSIONS.md` §5「B4」点名要补 **128/192/256** 三个小 chunk
# 采样点（EVIDENCE_INDEX 缺口 3）。B4 前置核查（OPT-03）读 `model_v7.json` 发现：
#
#     pf128 有 **8** 个键、pf256 有 **8** 个键、**pf192 = 0 个键**（数据里 prefill 轴
#     是 [0,128,256,512,1024,2048]，**从没出现过 192**）。
#
# ⇒ 128/256 在 v7 就已经可预测（B4 已证：两会话 32 条、会话内重复性中位 0.07%、
#    部署表对已采形状 **100%** 命中）；**缺的只有 192**。
#
# ⇒ **v8 只改一处**：把 `mixed_small_chunk.prefill_tok` 从 `[128, 256]` 扩到
#    `[128, 192, 256]`。**不动**别的（batch 仍 {1,2,4,8}、kv 仍 {1024,2048}、
#    其余三组逐字沿用 v7）。
#    ⇒ 新增 4 batch × 1 pf × 2 kv = **8 个点**。
#
# ⚠️ 为什么**另开 v8** 而不是原地改 v7（这条是网格自己的纪律）：
#    "看到结果之后再加密采样点，就会把'事后挑形状'混进来 ⇒ 要加密必须另开一轮（v2）并说明"。
#    且原地改 v7 会让 `grid_sha256` 变 ⇒ 已采的 `model_v7.json` 记录的
#    `grid_sha256: 5c3b66f75c89522c` 与之失配 ⇒ 血缘断。
#
# ⚠️ 可证伪的预期（采完必须核，不达标要如实记）：
#    ① pf192 的实测值应落在 **pf128 与 pf256 之间**（同 batch、同 kv）；
#       若落到区间外，说明小 chunk 在 pf 维**非单调** —— 那正是 COST_PROFILE 记录
#       过的 batch 非单调（b3>b4）的同类风险，会直接影响"能否内插"的结论。
#    ② pf192 的会话内重复性应 ≤ pf128/pf256 的水平（≤0.23%）。
#
# ⚠️ 与 v7 的关系：v8 是 v7 的**超集**（多 8 个点）⇒ v7 的 220 条记录**全部**仍是
#    v8 的合法子集。**但不得跨会话并表**（COST_PROFILE §6）⇒ v8 的键仍只从
#    **v8 自己的会话**建；B4 的"能"结论也只对 v7 已采的 128/256 成立。
#
"""


def build() -> str:
    src = SRC.read_text(encoding="utf-8")
    if _OLD_PROTOCOL not in src:
        raise SystemExit(f"✗ {SRC.name} 里没有 {_OLD_PROTOCOL!r}（已改过？请改本脚本）")
    out = src.replace(_OLD_PROTOCOL, _NEW_PROTOCOL, 1)
    if _OLD_AXIS not in out:
        raise SystemExit("✗ axes.prefill_tok 行与预期不符（已改过？请改本脚本）")
    out = out.replace(_OLD_AXIS, _NEW_AXIS, 1)
    if _OLD_GROUP_PF not in out:
        raise SystemExit("✗ mixed_small_chunk 的 prefill_tok/kv_len 两行与预期不符")
    out = out.replace(_OLD_GROUP_PF, _NEW_GROUP_PF, 1)
    # 该组的 `why` 里也提到点数含义，追加一句 v8 说明（不改结构）。
    _why_anchor = ("本组为「小候选能不能被预测」提供**实测键** —— `lookup` 下没有键就等于不能预测。\n")
    if _why_anchor not in out:
        raise SystemExit("✗ 找不到 mixed_small_chunk 的 why 锚点")
    out = out.replace(
        _why_anchor,
        _why_anchor + "      **v8 补齐 192**：B4 核查发现 v7 只有 128/256（192 键数 = 0），"
        "而任务书点名 128/192/256。\n", 1)
    return _HEADER + out


def verify(path: Path) -> list[str]:
    """校验：protocol、三组形状、点数、real_mixed、以及**只多 pf192** 这一不变量。"""
    sys.path.insert(0, str(_ROOT / "src"))
    sys.path.insert(0, str(_ROOT / "scripts"))
    from costcalib_grid import expand, load_grid  # noqa: E402

    bad: list[str] = []
    grid = load_grid(path)
    shapes = list(expand(grid))
    if grid.get("protocol_version") != "cost-calib-v8":
        bad.append(f"protocol_version 应为 cost-calib-v8，实际 {grid.get('protocol_version')!r}")
    new = [s for s in shapes if s.group == _GROUP_ID]
    if len(new) != _GROUP_N:
        bad.append(f"`{_GROUP_ID}` 有 {len(new)} 点，期望 {_GROUP_N}（改了组就要同步 `_GROUP_N`）")
    if not all(s.batch in (1, 2, 4, 8) and s.prefill_tok in (128, 192, 256)
               and s.kv_len in (1024, 2048) for s in new):
        bad.append("小 chunk 组里有超出声明范围的形状")
    if not all(s.n_new_mode == "real_mixed" for s in new):
        bad.append("小 chunk 组必须全是 real_mixed —— 否则成本仍按旧口径被放大 ~b 倍")
    # ★ v8 的核心不变量：**192 必须出现**，且只在 mixed_small_chunk 里出现
    pf192 = [s for s in shapes if s.prefill_tok == 192]
    if not pf192:
        bad.append("网格里没有 pf192 的点 ⇒ v8 的唯一改动没生效")
    if any(s.group != _GROUP_ID for s in pf192):
        bad.append(f"pf192 只应出现在 `{_GROUP_ID}`，实际也出现在 "
                   f"{sorted({s.group for s in pf192 if s.group != _GROUP_ID})}")
    if len(pf192) != 8:
        bad.append(f"pf192 应有 8 点（4 batch × 2 kv），实际 {len(pf192)}")
    # 与 v7 对比：v8 应恰好多 8 点
    v7 = _ROOT / "configs" / "cost_calib" / "shape_grid_v7.yaml"
    if v7.exists():
        n7 = len(list(expand(load_grid(v7))))
        if len(shapes) != n7 + 8:
            bad.append(f"v8 应比 v7 多 8 点（v7={n7}、v8={len(shapes)}）")
    if not (SRC.exists() and "v7" in SRC.name):
        bad.append("参照网格不是 v7 —— 本脚本假定以 v7 为底")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成/校验 v8 标定网格（v7 + pf192）")
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
        print(f"✓ {DST.name} 校验通过（含 `{_GROUP_ID}` {_GROUP_N} 点、全 real_mixed、"
              f"pf192 恰 8 点）")
        return 0

    text = build()
    with open(DST, "w", encoding="utf-8", newline="\n") as fh:   # ⚠️ 显式 LF
        fh.write(text)
    bad = verify(DST)
    if bad:
        print("✗ 写出后校验失败：" + "；".join(bad), file=sys.stderr)
        return 1
    print(f"已写出 {DST}")
    print(f"  采子集请用：--only-group {_GROUP_ID}（{_GROUP_N} 点）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
