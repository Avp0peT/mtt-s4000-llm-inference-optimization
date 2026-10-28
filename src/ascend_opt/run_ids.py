"""`run_id` 的**唯一**解析实现 —— 项目里只允许有这一份。

为什么必须统一
--------------
2026-09-27 实测一次**假失败**：D1① 的闸门把 `S14G-short_chat-fixed_cost_pilot` 解析成臂名
`fixed_cost_pilot`（目录名只做 `-r\\d+$` 与 `short_chat-` 两处字符串替换），
而 `diagnose_scheduling.py` 用正则解析成 **`cost_pilot`** ⇒ 闸门在自己的
`per` 字典里查 `cost_pilot` 查到空列表，于是打印

    n_runs: 0, small_steps: 0, n_degraded: [], has_arm_state: False

并据此报出**两条**"失败"：

1. `条件1：小 chunk(128/256) 一步都没被实际执行` —— 其实日志里同一批的
   `fixed_cost_pilot` 打印 `chunk 生效=128..1024`，**执行过**；
2. `条件2：arm_state/n_degraded 数不出来（源码未部署到位）` —— 其实
   `meta.json` 的 `controller_effect.arm_state` **就在产物里**
   （`n_degraded: 34`、`n_prefill_starved: 17`）。

⇒ 教训：**同一件事有两份实现时，它们迟早分叉，而分叉的表现是"另一个模块查不到数据"**
—— 不报错、只是空。所以解析收在这里，`scripts/` 与 `benchmark/` 都从这里 import。

格式
----
`<TAG>-<workload>-<cell>_<arm>[-r<pass>]`

* `TAG` 与 `workload` 不含 `-`；`cell` 是纯小写字母；`arm` 可含 `_` 与 `-`；
* 重复跑的同一格用 `-r0` / `-r1` … 区分（无后缀 = 第 0 次）。

⚠️ **解析失败一律返回 `None`，不得回退成猜测值**。调用方必须把"解析不出来"
当成**独立的状态**处理（本项目的做法是记为 `INCONCLUSIVE` 而不是当成某个默认臂名）——
把它们合并，正是上面那次假失败的形态。

⚠️ 已知边界：`20260925-225848-short_chat-inB` 这类**旧命名**（时间戳当 tag）会被解析成
`tag=20260925, workload=225848, ...`，得到一个"语法上成立、语义上无意义"的臂名。
因此调用方还须做**交叉核对**（解析出的臂名是否出现在同名批次的产物里），
不能只看"有没有解析出来"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: 与 `scripts/diagnose_scheduling.py` 原先的内联正则**逐字一致** —— 统一时
#: 刻意不改语义，只把它挪到唯一一处。行为变化会让历史结论对不上。
RUN_ID_RE = re.compile(
    r"^(?P<tag>[^-]+)-(?P<workload>[^-]+)-(?P<cell>[a-z]+)_(?P<arm>.+?)"
    r"(?:-r(?P<pass>\d+))?$"
)


@dataclass(frozen=True)
class RunId:
    """解析结果。`pass_index` 为 `None` 表示名字里没有 `-rN` 后缀。"""

    raw: str
    tag: str
    workload: str
    cell: str
    arm: str
    pass_index: int | None = None

    @property
    def group_key(self) -> str:
        """同一格的所有重复共用的键（=`TAG-workload-cell_arm`）。"""
        return f"{self.tag}-{self.workload}-{self.cell}_{self.arm}"


def parse_run_id(run_id: str) -> RunId | None:
    """解析；**失败返回 `None`**（不猜、不回退）。"""
    if not run_id or not isinstance(run_id, str):
        return None
    m = RUN_ID_RE.match(run_id.strip())
    if not m:
        return None
    p = m.group("pass")
    return RunId(raw=run_id.strip(), tag=m.group("tag"), workload=m.group("workload"),
                 cell=m.group("cell"), arm=m.group("arm"),
                 pass_index=(int(p) if p is not None else None))


def arm_of(run_id: str) -> str | None:
    """只要臂名；解析不出来返回 `None`。"""
    r = parse_run_id(run_id)
    return r.arm if r else None


def normalize_arm(name: str) -> str:
    """把**目录名**（可能带 `-rN` 后缀、可能带 workload 前缀）归一成臂名。

    这是**旧代码那条路径的替代品**：它以前写的是
    `re.sub(r"-r\\d+$", "", stem).replace("short_chat-", "")`，
    得到的是 **cell+arm**（如 `fixed_cost_pilot`）而不是臂名 ⇒ 与 `diagnose_scheduling`
    的臂名（`cost_pilot`）不一致。现在唯一权威是 `parse_run_id`，本函数只是
    "拿不到合法 run_id 时的兜底"，**并且会显式告诉调用方它走的是兜底**。
    """
    r = parse_run_id(name)
    if r is not None:
        return r.arm
    stripped = re.sub(r"-r\d+$", "", name)
    # 兜底：去掉 `<workload>-` 前缀（workload 无 `-`）。
    parts = stripped.split("-", 1)
    return parts[1] if len(parts) == 2 else stripped
