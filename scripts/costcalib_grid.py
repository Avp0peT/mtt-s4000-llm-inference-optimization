#!/usr/bin/env python3
"""成本标定**形状网格**的加载 / 校验 / 展开（第 11 步，协议 `cost-calib-v1`）。

为什么单独一个模块、且**不 import 任何设备库**
------------------------------------------------
这一层全是**纯逻辑**：读网格 → 校验它是否守住了预注册的规则 → 展开成具体采样点。
把它与"真机上跑 MUSA"分开，是为了让这些判断**可以被单元测试覆盖**
（`benchmark/plot_s10_surface.py` 的 `build_plan` 已经用过同一条路子）。
采集脚本再薄，也只要"照着展开出来的点去跑"。

校验项都**直接对应预注册里的规则**（`NEXT_SESSION_TASKS.md` 第 11 步）：
这些规则如果只写在文档里，改配置的人不会知道；写成守卫才会**当场失败**。
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GRID = _ROOT / "configs" / "cost_calib" / "shape_grid_v1.yaml"

#: 预注册要求的划分单位。任何"逐步随机"式的划分都在禁止之列。
_ALLOWED_SPLIT_UNITS = {"shape", "session", "trajectory"}
#: 明确禁止的划分方式（写出来是为了让报错信息能点出"你用了被禁的那个"）。
_FORBIDDEN_SPLIT_UNITS = {"stepwise_random", "step", "adjacent_step", "random"}


@dataclass(frozen=True)
class Shape:
    """一个具体采样点。字段名与 `COST_PROFILE_S4000.md` 的口径一致。"""

    group: str
    batch: int
    prefill_tok: int
    kv_len: int
    n_forwards: int
    #: **新 token 的分配方式**（第 14 步新增）：
    #:
    #: * ``uniform``（默认，= v1..v5 口径）：**每条序列都做 `prefill_tok` 个**新 token；
    #: * ``real_mixed``：**一条序列承担 `prefill_tok` 个、其余各只做 1 个**
    #:   —— 这才是 harness 的真实结构（实测：遥测 `note` 里 `prefill_tok=512 decode=7`
    #:   ⇒ 512 是**全步总量**、给**一条**请求，而旧口径把它当成"每条都做 512" ⇒ 差约 `batch` 倍）。
    #:
    #: ⚠️ **必须进 `as_key`**：否则同 `(batch, prefill, kv, nf)` 的两种口径会**撞键**，
    #: 产物里就分不出这一格是哪种形状。
    n_new_mode: str = "uniform"

    def as_key(self) -> tuple:
        return (self.batch, self.prefill_tok, self.kv_len, self.n_forwards, self.n_new_mode)


def load_grid(path: str | Path = DEFAULT_GRID) -> dict[str, Any]:
    import yaml  # 局部 import：本模块的纯逻辑部分不依赖 yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate(grid: dict[str, Any]) -> list[str]:
    """返回问题清单（空 = 通过）。**每条都对应一条预注册规则**，不是泛泛的 schema 检查。"""
    p: list[str] = []

    if not grid.get("protocol_version"):
        p.append("缺 protocol_version —— 验收要求'预先登记新的采集协议与**版本**'")

    axes = grid.get("axes") or {}
    for ax in ("batch", "prefill_tok", "kv_len"):
        if not axes.get(ax):
            p.append(f"axes.{ax} 为空 ⇒ 网格边界没定，点就没法校验是否越界")

    # ① 每个采样组必须写明"为什么要有它"（否则没人敢删也没人敢加）
    groups = grid.get("groups") or []
    if not groups:
        p.append("groups 为空 ⇒ 没有采样点")
    for g in groups:
        if not (g.get("why") or "").strip():
            p.append(f"group {g.get('id')!r} 缺 why（说不出为什么要有它，就不该有它）")
        p.extend(_check_n_new_mode(g))          # ④ 新 token 的分配方式只能两种

    # ② 每个采样点必须在 axes 的取值域内 —— 主要抓拼写/手滑
    for g in groups:
        for ax in ("batch", "prefill_tok", "kv_len", "n_forwards"):
            vals = g.get(ax)
            if vals is None:
                continue
            domain = axes.get(ax)
            if domain and ax != "n_forwards":
                bad = [v for v in vals if v not in domain]
                if bad:
                    p.append(f"group {g.get('id')!r} 的 {ax}={bad} 不在 axes.{ax}={domain} 内")

    # ③ ★ 块 A 只能作下界参考、**不得用于拟合**（验收第 2 条）
    blocks = grid.get("blocks") or {}
    if (blocks.get("A") or {}).get("role") != "lower_bound_only":
        p.append("blocks.A.role 必须是 'lower_bound_only' —— "
                 "快态平台只是性能下界，验收明确禁止'只拟合快态最小值'")
    if (blocks.get("B") or {}).get("role") != "fitted":
        p.append("blocks.B.role 必须是 'fitted' —— 被拟合的应是**服务协议**下的每步成本")
    for b in ("A", "B"):
        if not (blocks.get(b) or {}).get("method"):
            p.append(f"blocks.{b}.method 为空 ⇒ 采集方法没登记")

    # ④ 会话与划分（验收第 3 条）
    sess = grid.get("session") or {}
    if sess.get("cross_session_merge") != "forbidden":
        p.append("session.cross_session_merge 必须是 'forbidden' —— "
                 "COST_PROFILE §6：绝对值跨会话漂移 1.3–4%，不得并表")

    # ⚠️ 两个列表字段都要先挡"写成字符串"：`set("stepwise_random")` 会把它**逐字符拆开**，
    # 与 `_FORBIDDEN_SPLIT_UNITS` 的交集恒为空 ⇒ 下面那条守卫会**静默失效**
    # （看起来"登记了"，其实一个都没匹配上）。这个坑在本仓库踩过同族的，所以显式变错误。
    raw_units = sess.get("split_unit")
    if isinstance(raw_units, str):
        p.append("session.split_unit 是字符串；**必须是列表** —— set(字符串) 会被逐字符拆开")
        raw_units = []
    raw_forbidden = sess.get("forbidden_split")
    if isinstance(raw_forbidden, str):
        p.append("session.forbidden_split 是字符串；**必须是列表**（同上）")
        raw_forbidden = []

    units = set(raw_units or [])
    if not units:
        p.append("session.split_unit 为空 ⇒ 验收要求'按独立形状/轨迹/会话'划分")
    bad_units = units - _ALLOWED_SPLIT_UNITS
    if bad_units:
        p.append(f"session.split_unit 含不允许的单位 {sorted(bad_units)}；"
                 f"只允许 {sorted(_ALLOWED_SPLIT_UNITS)}")
    forbidden_used = set(raw_forbidden or []) & _FORBIDDEN_SPLIT_UNITS
    if not forbidden_used:
        p.append("session.forbidden_split 没有登记被禁止的划分方式 ⇒ "
                 "相邻步随机切分这个泄漏源没被写下来")

    # ⑤ 指纹（验收第 5 条）：四类都要有
    fp = grid.get("fingerprint") or {}
    for k in ("hardware", "software", "model", "precision"):
        if not fp.get(k):
            p.append(f"fingerprint.{k} 缺失 ⇒ 失配时无法比对（验收：失配不能静默使用）")
    if (fp.get("model") or {}).get("revision") in (None, "", "main"):
        p.append("fingerprint.model.revision 必须是**具体 commit**——"
                 "写 'main' 等于没钉住（模型换了也看不出来）")

    # ⑥ 必须一起记录的量
    if not grid.get("record"):
        p.append("record 为空 ⇒ 没登记'必须一起记录哪些量'，事后无法重算")

    return p


def expand(grid: dict[str, Any]) -> list[Shape]:
    """把采样组展开成具体采样点（笛卡尔积），并按 key 去重。

    去重是必要的：`n_forwards` 为空的组与 `[1]` 的组会展开出同一批点，
    重复采会白白多花机时，也会让"每个形状采几次"这个数变得不可数。
    """
    out: dict[tuple[int, int, int, int], Shape] = {}
    for g in grid.get("groups") or []:
        batches = g.get("batch") or []
        pf = g.get("prefill_tok") or []
        kv = g.get("kv_len") or []
        nf = g.get("n_forwards") or [1]
        for b, t, k, n in itertools.product(batches, pf, kv, nf):
            s = Shape(group=g["id"], batch=b, prefill_tok=t, kv_len=k, n_forwards=n,
                      n_new_mode=str(g.get("n_new_mode") or "uniform"))
            out.setdefault(s.as_key(), s)
    return sorted(out.values(), key=lambda s: s.as_key())


_N_NEW_MODES = ("uniform", "real_mixed")


def _check_n_new_mode(g: dict) -> list[str]:
    m = str(g.get("n_new_mode") or "uniform")
    if m not in _N_NEW_MODES:
        return [f"group {g.get('id')!r} 的 n_new_mode={m!r} 非法；只能 {'/'.join(_N_NEW_MODES)}"]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="成本标定形状网格：校验与展开")
    ap.add_argument("--grid", default=str(DEFAULT_GRID))
    args = ap.parse_args(argv)

    grid = load_grid(args.grid)
    problems = validate(grid)
    print(f"协议版本：{grid.get('protocol_version')}")
    print(f"\n=== 校验（{len(problems)} 个问题）===")
    if problems:
        for x in problems:
            print(f"  ✗ {x}")
        return 1
    print("  ✓ 通过（每条校验都对应一条预注册规则）")

    shapes = expand(grid)
    print(f"\n=== 展开：{len(shapes)} 个**去重后**的采样点 ===")
    by_group: dict[str, int] = {}
    for s in shapes:
        by_group[s.group] = by_group.get(s.group, 0) + 1
    for g, n in by_group.items():
        print(f"  {g:<24} {n} 点")
    blocks = grid.get("blocks") or {}
    a, b = blocks.get("A") or {}, blocks.get("B") or {}
    n_a = (a.get("n_blocks") or 0) * (a.get("block_n") or 0)
    n_b = b.get("n_calls") or 0
    print(f"\n=== 采集量估算 ===")
    print(f"  块 A（{a.get('role')}）：{len(shapes)} 点 × {n_a} 次调用 = {len(shapes) * n_a} 次前向")
    print(f"  块 B（{b.get('role')}）：{len(shapes)} 点 × {n_b} 次调用 = {len(shapes) * n_b} 次前向")
    print(f"  合计 ≈ {len(shapes) * (n_a + n_b)} 次前向"
          f"（按每步 60–90 ms 粗估 ≈ {len(shapes) * (n_a + n_b) * 0.075 / 60:.0f} 分钟设备时间）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
