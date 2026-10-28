#!/usr/bin/env python3
"""按**预注册规则**机械选出 D3/D7 要用的 `chunk_size_min`（把判断从人手里拿走）。

规则写在 `EXPERIMENTS.md` 的「D1② · 选取规则（先冻结，写在看到 ② 的数据之前）」。
本脚本**只实现它**，不含任何自由裁量 —— 否则"规则"就退化成"我看了数据以后再说"。

| 条件（同系列内部：同模型/同源码/同 SLO/同负载） | 采用 |
| --- | --- |
| `D1(128) ≤ D1(512) − 20pp` **且** `D1(512) ≥ D1(256) ≥ D1(128)` | 128 |
| 同上但**非单调**（128 破单调） | 256 |
| 都不满足（含不降/反向） | 512（放弃小候选）⇒ 必须重新归因 |
| **三点 D1 全为 0（无区分度）** | 转用 D2：`D2(512) ≥ D2(256) ≥ D2(128)` 且严格更小 ⇒ 256；<br>再不行 ⇒ **512** 且记「未判」 |

⚠️ 三条纪律（写进输出，避免被读大）：
1. **「无区分度」≠「都没改善」≠「都一样」**，三者结论不同；
2. 本脚本选出的下限**只是把 D3/D7 的配置钉住**，**不构成"改善"的结论**；
3. 任何一项**数不出来**（`None`）⇒ 走保守分支并**显式标注"不可判"**，不得当 0。

用法::

    python scripts/pick_d1_min.py --diag results/processed/diag_S14H.json
    # 输出（给 shell 用）：最后一行形如  PICK=128
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: 主判据的阈值：`D1(128)` 要比 `D1(512)` 低这么多 **个百分点对应的比例**。
MARGIN = 0.20

_DEFAULT_MIN = 512


def pick_min(d1: dict[str, float | None], d2: dict[str, float | None]) -> tuple[int, str]:
    """返回 `(选出的下限, 理由)`。纯函数，便于离线测试。"""
    v1_512, v1_256, v1_128 = d1.get("512"), d1.get("256"), d1.get("128")

    if any(v is None for v in (v1_512, v1_256, v1_128)):
        return _DEFAULT_MIN, ("不可判：D1 有三点缺失（数不出来）⇒ 保守维持 512；"
                             "不得把缺项当 0")

    # 三点全 0 ⇒ 该格 D1 无区分度 ⇒ 不能据此选点，转 D2
    if max(v1_512, v1_256, v1_128) == 0:
        v2_512, v2_256, v2_128 = d2.get("512"), d2.get("256"), d2.get("128")
        if any(v is None for v in (v2_512, v2_256, v2_128)):
            return _DEFAULT_MIN, ("未判：D1 三点全为 0（**无区分度**，不是'都没改善'，也不是"
                                  "'都一样'），且 D2 数不出来 ⇒ 维持 512")
        if v2_512 >= v2_256 >= v2_128 and v2_128 < v2_512:
            return 256, (f"未判→转 D2：D1 无区分度；D2 单调下降 "
                         f"({v2_512:.3f}≥{v2_256:.3f}≥{v2_128:.3f}) ⇒ 取 256"
                         f"（不取 128：无 D1 支撑，只按 D2 退一档保守）")
        return _DEFAULT_MIN, ("未判：D1 无区分度，D2 也无区分度（或不单调）⇒ 维持 512；"
                              "「未判」登记，不得写成'小候选无用'")

    if v1_128 <= v1_512 - MARGIN and v1_512 >= v1_256 >= v1_128:
        return 128, (f"主规则：D1(128)={v1_128:.2%} ≤ D1(512)={v1_512:.2%} − 20pp 且单调不增 "
                     f"(512≥256≥128) ⇒ 取 128")
    if v1_128 <= v1_512 - MARGIN:
        return 256, (f"非单调分支：D1(128)={v1_128:.2%} 降幅达标但破了单调 "
                     f"(512={v1_512:.2%}, 256={v1_256:.2%}, 128={v1_128:.2%}) ⇒ 取 256")
    return _DEFAULT_MIN, (f"否证：D1(128)={v1_128:.2%} 相对 D1(512)={v1_512:.2%} "
                          f"降幅不足 20pp（或反向）⇒ 维持 512 并**重新归因**："
                          f"候选集不是限制项，不得继续加小块，也不得把 D1 变化归给候选集")


def _from_diag(diag: dict, arm: str) -> tuple[float | None, float | None]:
    a = (diag.get("arms") or {}).get(arm) or {}
    return a.get("d1_rate_median"), a.get("d2_blocked_share_median")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按预注册规则选 chunk_size_min")
    ap.add_argument("--diag", required=True, help="diagnose_scheduling.py --tag S14H 的 --json 产物")
    ap.add_argument("--json", default=None, help="把结论写成 JSON（可选）")
    args = ap.parse_args(argv)

    p = Path(args.diag)
    if not p.exists():
        print(f"✗ 找不到 {p}", file=sys.stderr)
        return 2
    diag = json.loads(p.read_text(encoding="utf-8"))

    d1, d2 = {}, {}
    for mn in ("512", "256", "128"):
        d1[mn], d2[mn] = _from_diag(diag, f"cost_min{mn}")

    print(f"读入 {p.name}｜D1 = {d1}｜D2 = {d2}")
    chosen, why = pick_min(d1, d2)
    for i, line in enumerate(("⚠️ 本结论只是把 D3/D7 的配置钉住，**不构成'改善'的结论**；"
                              "改善与否仍按 §6 的证据链逐层报。",
                              "⚠️ 「无区分度」≠「都没改善」≠「都一样」——三者写法不同。",
                              "⚠️ 任何缺项都按'不可判'走保守分支，不得当 0。"), 1):
        print(f"  {i}. {line}")
    print(f"理由：{why}")
    print(f"PICK={chosen}")

    if args.json:
        q = Path(args.json)
        q.parent.mkdir(parents=True, exist_ok=True)
        with open(q, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"d1": d1, "d2": d2, "pick": chosen, "reason": why},
                                ensure_ascii=False, indent=1))
        print(f"已写出：{q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
