#!/usr/bin/env python3
"""清点论文里的**占位标记**：还剩多少坑、分别在哪个文件、ID 有没有重复。

为什么需要它
------------
写作会话的纪律是「**未收到的数据用占位标记，不许留空、不许猜**」
（见 `paper/WRITING_BRIEF.md` §3）。但"还有多少没补"如果只能靠人读，就会：
① 补漏（改了一处忘了另一处）；② 把"没查"当成"没有"。

三条判据（都刻意做成会失败的）
------------------------------
1. **有占位就非零退出** —— 那是"还不能投稿"的机器可读信号，不是错误；
2. **ID 必须全局唯一** —— 重复 ID 会让"补第 P03 个"指到两处；
3. **输入为空 ⇒ 非零（rc=2）**，**不得**当成"全部通过" ——
   空输入会伪装成合格结果（实测踩过的形态：路径写错 ⇒ 读出 0 ⇒ 显示"全部通过"，
   实际验的是空集）。

用法::

    python scripts/list_paper_placeholders.py                 # 人读
    python scripts/list_paper_placeholders.py --json x.json    # 机器读
退出码：0 = 无占位；1 = 有占位；2 = 输入错（目录不存在 / 一个文件都没扫到）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = _ROOT / "paper" / "sections"

#: 占位标记：`【待补·ID】说明…`。ID 允许 `P01` / `p1` / `D3.2` 这类写法。
_MARK = re.compile(r"【待补[·・]([A-Za-z0-9_.\-]+)】")
#: 被**禁止**的偷懒写法（它们是"缺失"与"测量值"的混淆源）。
#: ⚠️ 只在**正文目录**里查，避免对规划类文档误报（那些文件里"待填"是正常用词）。
_LAZY = re.compile(r"\bTBD\b|\bXXX+\b|待填|待定")


def scan(dirpath: Path) -> tuple[dict[str, list[dict]], dict[str, list[str]], list[str]]:
    """返回 (按文件分组的占位, 重复 ID → 文件列表, 偷懒写法命中)。"""
    found: dict[str, list[dict]] = defaultdict(list)
    ids: dict[str, list[str]] = defaultdict(list)
    lazy: dict[str, list[str]] = defaultdict(list)
    files = sorted(dirpath.glob("*.md"))
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in _MARK.finditer(line):
                pid = m.group(1)
                found[f.name].append({"id": pid, "line": i, "text": line.strip()})
                ids[pid].append(f.name)
            if _LAZY.search(line):
                lazy[f.name].append(f"L{i}: {line.strip()[:70]}")
    return dict(found), dict(ids), dict(lazy)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="清点论文占位标记")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    d = Path(args.dir)
    if not d.is_dir():
        print(f"✗ 目录不存在：{d}", file=sys.stderr)
        return 2
    if not list(d.glob("*.md")):
        # ★ 空输入不得当成"通过"——它最可能意味着路径写错
        print(f"✗ {d} 里一个 .md 都没有 ⇒ **未判**（不是'无占位'）", file=sys.stderr)
        return 2

    found, ids, lazy = scan(d)
    n = sum(len(v) for v in found.values())
    dup = {k: v for k, v in ids.items() if len(v) > 1}

    print(f"扫描 {d}｜文件 {len(list(d.glob('*.md')))} 个｜占位 {n} 处")
    for name in sorted(found):
        print(f"\n  {name}（{len(found[name])} 处）")
        for item in found[name]:
            print(f"    · 【待补·{item['id']}】L{item['line']}  {item['text'][:96]}")
    if dup:
        print("\n⚠️ **ID 重复**（补数时会指错地方）：")
        for k, v in sorted(dup.items()):
            print(f"    · {k} → {v}")
    if lazy:
        print("\n⚠️ **偷懒写法**（禁止：缺失≠0，必须写成 `【待补·ID】…`）：")
        for name in sorted(lazy):
            print(f"    {name}: {len(lazy[name])} 行")
            for x in lazy[name][:4]:
                print(f"      {x}")
    else:
        print("\n✓ 没有 TBD/XXX/待填/待定 这类偷懒写法")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"total": n, "by_file": found, "duplicate_ids": dup,
                                 "lazy": lazy}, ensure_ascii=False, indent=1),
                     encoding="utf-8", newline="\n")
        print(f"\n已写出：{p}")

    bad = n or dup or lazy
    print(f"\n=== {'✗ 还不能投稿' if bad else '✓ 无占位，可以进入下一轮评审'} ===")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
