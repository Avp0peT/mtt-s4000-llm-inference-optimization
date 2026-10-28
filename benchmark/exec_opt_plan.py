"""执行优化线批次编排的**纯逻辑**（S01–S03 真机 A/B）。

为什么单独成模块：批次的"跑什么、跑多久、什么时候停、下次从哪续"
这些**判据**必须能在本机验证。把它们塞进需要 MUSA/transformers 的运行器里，
就等于"只在远端跑过、从没本地验证"——那正是本项目最怕的形态
（出问题只能靠读日志猜）。

本模块**不导入 torch、不碰网络、不读写 `results/raw/`**，只做纯函数与数据结构。

三条设计判据（都有实测理由）
----------------------------
1. **穿插，不背靠背**。同一配置连续跑 N 次会把一次偶发停顿整段归因成
   "这个配置更差"。已实测过同类现象（同配置重复两次：tpot 只差 1.23×，
   而 stall_p99 差 **4.35×**）。⇒ 顺序必须是"扫完整张表 = 一轮"。
2. **均衡重复**。"跑满"不等于"能塞多少塞多少"：所有臂的**完成重复数必须相等**，
   否则比较退化成"谁的 n 大"。⇒ 预算不足时**宁可少跑一整轮，也不跑半轮**。
3. **成功由产物判定**。"跑完了但活没干完"（失败/拒绝/未完成 > 0）**不算成功**，
   且这类产物必须**留在证据目录**里，不能删。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "PlanItem",
    "build_plan",
    "plan_summary",
    "StopDecision",
    "decide_stop",
    "WaitExit",
    "decide_wait_exit",
    "write_json_atomic",
    "read_json",
    "resolve_arms",
    "run_dir_name",
    "utc_stamp",
    "BASELINE_DEFAULT",
    "CFG_DIR_DEFAULT",
    "ORDER_MODES",
    "order_for_pass",
    "completion_flags",
]

#: 对照臂配置的**兜底路径**（相对仓库根）。
#:
#: ★ 放在这个纯逻辑模块里，是为了让它成为**唯一真相源**：矩阵运行器与它的
#: 外层编排器都要用它。2026-10-03 实测的缺陷正是"两处各自默认"：
#: 外层编排**从不传** `--baseline`，于是对照臂静默落到这里的兜底值，
#: 而外层给出的臂清单来自另一个目录 —— 产物里两臂的配置来源根本对不上，
#: 且**不报错**（`--baseline` 有默认值，缺省完全合法）。
#: ⇒ 判据：要么显式传，要么**在计划里打印出实际采用的那一个**，不许静默。
BASELINE_DEFAULT = "configs/mthreads/s4000_fp16.yaml"

#: 对照臂**配置目录**的兜底值（相对仓库根）。
#: ⚠️ 与 `BASELINE_DEFAULT` 同理：矩阵与外层编排必须用**同一个**默认值，
#: 否则"这一批用的是哪一目录的臂"就取决于谁在跑，而产物看不出区别。
CFG_DIR_DEFAULT = "configs/mthreads/exec_opt"

#: 一条待跑项。`pass_index` 从 0 起，与 `run_once(repeat_index=...)` 对齐。
@dataclass(frozen=True)
class PlanItem:
    pass_index: int
    cell: str                     # 臂名（配置 stem，基线固定为 "baseline"）

    @property
    def key(self) -> str:
        return f"{self.cell}#r{self.pass_index}"


def run_dir_name(run_id: str, pass_index: int) -> str:
    """与 `run_once` 的落盘规则**逐字一致**（`<run_id>` / `<run_id>-rN`）。

    必须在**同一条**规则上做"已完成"判定：若这里与 `run_once` 的命名规则分叉，
    断点续跑会重复跑已经跑过的格（写新目录），或跳过没跑的格（漏数据）——
    两种都静默。
    """
    return run_id if pass_index == 0 else f"{run_id}-r{pass_index}"


#: 顺序模式。
#: * ``block`` —— 每轮都按 `cells` 的原序（历史行为，**逐字节不变**）；
#: * ``abba``  —— 每轮**反转**格序：第 0 轮 A,B；第 1 轮 B,A；第 2 轮 A,B …
#:   ⇒ 同一格不会永远排在"这一轮的第一位"。
#: ⚠️ 为什么要 ``abba``：轮内位置本身可能是一个**系统性因素**（先跑的那一格享受
#: 更"冷"的机器状态、或更晚的机器时段）。固定 AB 会把"格效应"与"位置效应"
#: **混在一个方向差里**，而产物看不出区别。AB/BA 让两者在一半的轮里反向，
#: 于是配对差的中位数对位置不敏感。
ORDER_MODES: tuple[str, ...] = ("block", "abba")


def order_for_pass(cells: Sequence[str], pass_index: int, *,
                   order: str = "block") -> list[str]:
    """**唯一**的一处"某一轮按什么顺序跑"。★★ 计划、实际执行、续跑都必须用它。

    ⚠️ 为什么必须只有一处（2026-10-03 修）：此前**两处**各写了一遍 ——
    计划用 :func:`build_plan`，执行循环里另写了一个
    `[(name, i) for i in (pass_index,) for name in cells]`。
    于是"计划写着 ABBA、实际跑的是 ABAB"，而**两边都不会报错**：
    进度文件里 `plan` 字段与实际执行顺序不符，事后核对时会以为是执行错了。
    """
    if order not in ORDER_MODES:
        raise ValueError(f"未知顺序模式 {order!r}；可用：{list(ORDER_MODES)}")
    seq = list(cells)
    if order == "abba" and pass_index % 2 == 1:
        seq = list(reversed(seq))
    return seq


def build_plan(cells: Sequence[str], repeats: int, *,
               order: str = "block") -> list[PlanItem]:
    """穿插顺序：``for p in range(repeats): for cell in order_for_pass(...)``。

    ⚠️ 不是"每格跑完 N 次再下一格"。见模块 docstring 判据 1。
    ⚠️ 默认 `block` ⇒ 与引入 `order` 之前**逐字节相同**（既有批次不变）。
    """
    if repeats < 1:
        raise ValueError("repeats 必须 >= 1")
    return [PlanItem(pass_index=p, cell=c)
            for p in range(repeats)
            for c in order_for_pass(cells, p, order=order)]


def plan_summary(cells: Sequence[str], repeats: int, *,
                 order: str = "block") -> dict[str, Any]:
    plan = build_plan(cells, repeats, order=order)
    return {
        "cells": list(cells),
        "repeats": repeats,
        "order_mode": order,
        "n_items": len(plan),
        "order": [it.key for it in plan],
        "note": ("穿插顺序 = 扫完整张表为一轮；每格的重复在时间上被拉开"
                 + ("；AB/BA：奇数轮反转格序（抵消「轮内位置」这一系统性因素）"
                    if order == "abba" else "")),
    }


# --------------------------------------------------------------------------- #
# 停止判据
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str = ""

    def __bool__(self) -> bool:                                  # pragma: no cover
        return self.stop


def decide_stop(
    *,
    elapsed_s: float,
    budget_s: float | None,
    items_done: int,
    items_total: int,
    avg_item_s: float | None,
    remaining_items_this_pass: int,
    stop_file_exists: bool,
) -> StopDecision:
    """是否应当**干净收尾**（而不是被杀）。

    ⚠️ **判据 2（均衡重复）在这里落地**：预算不够跑完当前这一轮时，
    **不开始**这一轮 —— 宁可少一轮，也不要出现"有的臂 5 次、有的臂 4 次"。
    `remaining_items_this_pass` = 本轮**尚未开始**的项数（不含正在跑的）。

    `avg_item_s is None`（还没跑完任何一项、没有用时估计）⇒ **不因预算停**
    （否则一次都跑不了）；由调用方先用 `--min-items-before-budget-check` 兜底。
    """
    if stop_file_exists:
        return StopDecision(True, "发现 STOP 文件 ⇒ 协作式停止（当前 run 跑完后退出）")
    if items_done >= items_total and items_total > 0:
        return StopDecision(True, "全部项已完成")
    if budget_s is None or avg_item_s is None or remaining_items_this_pass <= 0:
        return StopDecision(False)
    need = avg_item_s * remaining_items_this_pass * 1.15   # 15% 余量：别卡在边界上
    if elapsed_s + need > budget_s:
        return StopDecision(
            True,
            f"预算不足跑完当前这一轮（已用 {elapsed_s / 60:.1f} min，"
            f"本轮还需 ≈{need / 60:.1f} min，预算 {budget_s / 60:.1f} min）"
            "⇒ 干净收尾，避免出现不均衡重复",
        )
    return StopDecision(False)


# --------------------------------------------------------------------------- #
# 本机等待侧的退出判据
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WaitExit:
    """本机轮询**是否应当停止等待**。`channel` 说明是**哪条通道**触发的。"""

    done: bool
    channel: str = ""      # "stopped" | "summary" | "all_items" | ""
    reason: str = ""

    def __bool__(self) -> bool:                                   # pragma: no cover
        return self.done


def decide_wait_exit(
    *,
    progress: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> WaitExit:
    """★ 本机等待的退出判据 —— **三条通道都必须识别**。

    | 通道 | 谁写 | 何时 |
    | --- | --- | --- |
    | `stopped` | runner | 仅**协作式停止**（STOP 文件）|
    | `summary` | runner 收尾 | **总是**（预算收尾 / 全部完成 / 被中断都写）|
    | `all_items` | runner | 项数跑满（进程可能**仍在**收尾）|

    ⚠️ **实测失效（EXECOPT3，2026-09-30）**：远端 35 min 跑完 12/12 并写了 summary，
    而本机**只查** `progress.stop.stop` —— 预算收尾**不写** `stop` ⇒ 条件永假 ⇒
    本机轮询到 `--wait-timeout-minutes`（默认 600 min）为止。
    症状是**远端早已空闲、本机显示"在跑"**，与"卡住"不可区分。
    ⇒ 这就是"**只认一条退出通道**"的代价：进程的结束方式**不止一种**。

    `summary` 是**权威**信号（runner 的**最后一步**写入）；`all_items` 是**冗余**
    （summary 读不到时兜底）。返回 `channel` 便于调用方与日志区分。
    """
    st = (progress or {}).get("stop") or {}
    if st.get("stop"):
        return WaitExit(True, "stopped", str(st.get("reason") or ""))
    if summary:
        return WaitExit(
            True, "summary",
            f"汇总已生成：n_done={summary.get('n_done')}/{summary.get('n_items')}"
            f"｜轮数={summary.get('completed_passes')}"
            f"｜均衡={summary.get('balanced')}",
        )
    if progress:
        total = progress.get("n_total")
        done = progress.get("n_done")
        if total and done is not None and int(done) >= int(total):
            return WaitExit(True, "all_items", f"已完成 {done}/{total}")
    return WaitExit(False)


# --------------------------------------------------------------------------- #
# 进度 / 心跳 载荷（本机侧解析它，故字段名必须稳定）
# --------------------------------------------------------------------------- #

def progress_payload(
    *,
    tag: str,
    plan: Sequence[PlanItem],
    done_keys: Iterable[str],
    current: str | None,
    started_epoch: float,
    now_epoch: float,
    stopped: StopDecision | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """进度 + 心跳。**同时是心跳**：`heartbeat_epoch` 定期更新，
    这样单次 run 跑 20 分钟时，本机也能区分「在跑」与「卡住」。

    `done_keys` 只数**产物判定为完成**的项（不是"调用没抛异常"）。
    """
    done = list(done_keys)
    total = len(plan)
    payload: dict[str, Any] = {
        "schema": "exec-opt-progress-v1",
        "tag": tag,
        "n_total": total,
        "n_done": len(done),
        "done_keys": done,
        "current": current,
        "started_epoch": started_epoch,
        "heartbeat_epoch": now_epoch,
        "elapsed_s": round(now_epoch - started_epoch, 1),
        "items": [{"pass_index": it.pass_index, "cell": it.cell, "key": it.key}
                  for it in plan],
    }
    if stopped is not None:
        payload["stop"] = {"stop": stopped.stop, "reason": stopped.reason}
    if extra:
        payload["extra"] = dict(extra)
    return payload


def write_json_atomic(path: Path, payload: Any) -> None:
    """原子写（同目录临时文件 + `os.replace`）。

    为什么必须原子：本机会在批次跑的同时**轮询**这个文件读心跳；
    非原子写会让轮询读到"写了一半的 JSON" ⇒ 解析失败 ⇒ 看起来像"批次卡住了"。
    这类假故障会直接浪费一次人工介入。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path, *, default: Any = None) -> Any:
    """读 JSON；**读不到/解析失败返回 default**（调用方须把"未知"与"空"分开）。

    踩过的形态：把解析失败静默当成 `{}`，于是"进度文件还没写"与"文件坏了"
    长得一样。这里保留 `read_json(...) is None` 与 `== {}` 的区别。
    """
    p = Path(path)
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                            # noqa: BLE001
        return default


def resolve_arms(cfg_dir: Path, *, include: Sequence[str] | None = None,
                 exclude: Sequence[str] = ()) -> list[str]:
    """从配置目录解析臂名（= 配置 stem）。

    `include` 给定时按**给定顺序**返回（顺序会影响穿插次序，故允许显式指定）；
    否则按文件名排序 ⇒ **确定性**（同一目录两次解析必须给出同一顺序，
    否则"复现同一批次"会静默变成另一种穿插）。
    """
    d = Path(cfg_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"配置目录不存在：{d}")
    found = sorted(p.stem for p in d.glob("*.yaml"))
    if not found:
        raise FileNotFoundError(f"配置目录里没有 *.yaml：{d}")
    if include:
        missing = [a for a in include if a not in found]
        if missing:
            raise ValueError(f"指定的臂在 {d} 里不存在：{missing}（现有：{found}）")
        order = list(include)
    else:
        order = found
    return [a for a in order if a not in set(exclude)]


def utc_stamp(epoch: float | None = None) -> str:
    """run_id 用的时间戳（UTC，`YYYYMMDDTHHMMSSZ`）——与既有批次命名风格一致。"""
    t = time.gmtime(epoch if epoch is not None else time.time())
    return time.strftime("%Y%m%dT%H%M%SZ", t)


def completion_flags(*, n_done: int, n_cells: int, repeats: int,
                     fill_budget: bool) -> dict[str, Any]:
    """★ 收尾的**完成度**判据。

    为什么必须单独有它（2026-10-03 实测的静默少跑）
    -----------------------------------------------
    原先只看 `balanced`（= `n_done % n_cells == 0`）。而 `balanced` 在**任何**
    整数轮上都为真 —— 包括"只跑了 1 轮"。实测 `--repeats 5`（2 格）时：

        run_once 只被调用 2 次、n_done=2/10、completed_passes=1、
        balanced=**True**、failures=[]、rc=**0**

    ⇒ 少跑 8 项，却被判成"完整成功"。**均衡 ≠ 完整**：前者说"各臂一样多"，
    后者说"达到计划项数"。两个问题，两个判据。

    `fill_budget=True`（填预算模式）时目标是开区间，"没跑满"是**预期行为**，
    因此 `complete=None`（不适用）而不是 `False`；但它会带上
    `budget_limited=True`，免得"预算收尾"与"完整跑完"在产物里长得一样。
    """
    n_planned_fixed = (n_cells * repeats) if (not fill_budget and n_cells) else None
    complete = (n_done >= n_planned_fixed) if n_planned_fixed is not None else None
    return {
        "n_planned_fixed": n_planned_fixed,
        "complete": complete,
        "budget_limited": bool(fill_budget),
        "balanced": (n_done % n_cells == 0) if n_cells else True,
        #: ★ 唯一的"少跑"判据。调用方据此把 rc 置 1 —— **不看 rc 的读者**也能从
        #:   summary 里读到它，不必去比 n_done 与 n_planned_fixed。
        "under_run": bool(n_planned_fixed is not None and n_done < n_planned_fixed),
    }
