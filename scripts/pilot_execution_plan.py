#!/usr/bin/env python
"""**pilot 三臂执行清单**：只输出计划，就能在**跑之前**核对任务数。

★ 为什么需要本脚本
--------------------
`run_exec_opt_matrix.py` 的 `--baseline` **默认为真**（会自动加一个 baseline 格），
`--repeats` **默认为 None** ⇒ 若不给，会退到 `--fill-budget` 模式（可跑 3~50 轮）。
⇒ 「我以为跑 3 项、实际跑了 15 项」是**默认值**造成的，不是操作失误。

本脚本把 pilot 的三臂**写成常量**（配置路径 + 臂名 + 轮数），并**先算计划再校验**：
  · `repeats` 必须显式 = 1（本 pilot 不是性能批次 ⇒ 1 轮）
  · 臂数必须 = 3（P1/P2/P3）
  · 总任务数必须 = 臂数 × 轮数 = **3**
**任一不符 ⇒ 非零退出**（不给"跑起来再说"的机会）。

★ 与 FZ 状态一致
-----------------
三臂都用**当前实现**的语义，即 **FZ-1 = A（逐步择一）**、**FZ-2 = A（本步没被服务）**。
⇒ 结论**只能**表述为"**在（逐步择一）语义下 / 在（本步没被服务）口径下**"，
   **不得**把 A 写成"已选定最优语义"（`NEXT_SESSION_TASKS.md` §待冻结）。

⚠️ pilot **只用于**检查**机制 / 覆盖 / 节拍**，**不用于宣布性能优势**。
⚠️ 本脚本**不运行**任何真实批次；真机执行必须另跑（见 `--print-command`）。

用法::

    python scripts/pilot_execution_plan.py                    # 打印清单 + 校验
    # 真机命令需显式提供已在执行端核验的目录和 Python 绝对路径：
    python scripts/pilot_execution_plan.py --remote-root /verified/root \\
        --remote-python /verified/env/bin/python --print-command
    python scripts/pilot_execution_plan.py --json results/processed/pilot_plan.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath

_ROOT = Path(__file__).resolve().parents[1]

CFG_DIR = "configs/mthreads/prereg_batch"
BUDGET_MINUTES = 45  # 预算输入；矩阵仅在整轮边界检查，不会中断已经开始的一轮。
_SAFE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,95}\Z")


def new_tag() -> str:
    """每次计划生成一个实际标签；UTC 时间 + 随机后缀，避免日期占位和碰撞。"""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"PILOT_{stamp}_{uuid.uuid4().hex[:12]}"


def _remote_path(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if (not value or any(ord(c) < 32 for c in value) or "\\" in value
            or not value.startswith("/") or value.startswith("//")
            or ".." in PurePosixPath(value).parts or "<" in value or ">" in value):
        raise ValueError(f"{name} 必须是明确的 POSIX 绝对路径（无占位符/控制字符/..）")
    return str(PurePosixPath(value))

#: pilot 三臂。**只此一处定义**（避免文档与命令的第二处真相）。
#:
#: ★★ **P1 就是 `run_exec_opt_matrix` 的 `baseline` 格**（不是第四个 `--arms` 成员）。
#:    矩阵的 cells 是 `["baseline"] + arms` ⇒ 若把三臂全塞进 `--arms`、
#:    再让 `--baseline` 取默认值，会变成 **4 格**（`baseline` + 三臂），
#:    而用户要的是「**三臂各一次 = 3 项**」。
#:    ⇒ 正确写法：`--baseline <static_small.yaml>`（= P1）+ `--arms cost_single cost_dual`（= P2/P3）。
#:    产物里 P1 的臂名是 **`baseline`**（不是 `static_small`）—— 看 `run_ids` 即可对上。
#:
#: ⚠️ `static_small` 是**静态对照**（控制器关）⇒ 它**不是** cost 族，产物里没有
#:    `budget_kind`。判"机制是否被行使"只看 P2/P3（cost_single / cost_dual）。
PILOT_ARMS: list[dict] = [
    {"id": "P1", "arm": "static_small", "matrix_cell": "baseline",
     "role": "静态对照（同起点）", "controller": "关", "cost_family": False,
     "purpose": "测节拍 + 测该 cell 的 TPOT / 达标率基线"},
    {"id": "P2", "arm": "cost_single", "matrix_cell": "cost_single",
     "role": "单预算成本臂", "controller": "开（单预算 prefill_deadline_ms=null）",
     "cost_family": True,
     "purpose": "测控制器的步级轨迹与回退率"},
    {"id": "P3", "arm": "cost_dual", "matrix_cell": "cost_dual",
     "role": "双预算成本臂", "controller": "开（双预算 prefill_deadline_ms=2000）",
     "cost_family": True,
     "purpose": "测双预算是否改变步级轨迹（与 P2 逐 step 对比）"},
]

#: `--arms` 里**只放** P2/P3；P1 走 `--baseline`。
ARMS_ARG = [a["arm"] for a in PILOT_ARMS if a["matrix_cell"] != "baseline"]
BASELINE_ARG = f"{CFG_DIR}/static_small.yaml"

#: 本 pilot 用**固定轮数**（不是 fill-budget）⇒ 1 轮。理由：pilot 判"机制/覆盖/节拍"，
#: 不判性能 ⇒ 不需要重复；重复是**性能批次**的事（那里 >= 3 轮）。
REPEATS = 1

EXPECTED_TOTAL = len(PILOT_ARMS) * REPEATS

#: FZ 状态：三臂都用 A/A（当前实现）。**不得**写成"已选定最优"。
FZ_STATE = {
    "FZ-1": "A（逐步择一）—— 当前实现",
    "FZ-2": "A（本步没被服务）—— 当前实现",
    "utterance_ceiling": "只能说『在（逐步择一）语义下 / 在（本步没被服务）口径下』；"
                         "**不得**把 A 写成『已选定最优语义』",
}


def build_plan(*, tag: str | None = None, remote_root: str | None = None,
               remote_python: str | None = None,
               budget_minutes: float = BUDGET_MINUTES, cfg_dir: str = CFG_DIR) -> dict:
    """把 pilot 计划写成**可校验**的结构（不跑任何东西）。"""
    tag = new_tag() if tag is None else tag
    if not _SAFE_TAG.fullmatch(tag):
        raise ValueError("tag 只允许字母、数字、下划线和连字符，长度 1–96，首字符为字母或数字")
    remote_root = _remote_path(remote_root, name="remote-root")
    remote_python = _remote_path(remote_python, name="remote-python")
    if bool(remote_root) != bool(remote_python):
        raise ValueError("remote-root 与 remote-python 必须同时提供；未核验时两者都省略")
    if not math.isfinite(budget_minutes) or budget_minutes <= 0:
        raise ValueError("budget-minutes 必须是有限正数")
    config_path = PurePosixPath(cfg_dir)
    if config_path.is_absolute() or ".." in config_path.parts or "\\" in cfg_dir or ":" in cfg_dir:
        raise ValueError("cfg-dir must be a repository-relative POSIX path")
    # ★ cells 必须与 `run_exec_opt_matrix` 的 `["baseline"] + arms` 逐条对应
    cells = [a["matrix_cell"] for a in PILOT_ARMS]
    plan = [f"{c}#r{r}" for r in range(REPEATS) for c in cells]
    return {
        "tag": tag,
        "remote_root": remote_root,
        "remote_python": remote_python,
        "remote_identity_status": "PROVIDED_NOT_VERIFIED" if remote_root else "REQUIRED_NOT_PROVIDED",
        "cfg_dir": cfg_dir,
        "baseline_arg": f"{cfg_dir}/static_small.yaml",
        "arms_arg": ARMS_ARG,
        "arms": PILOT_ARMS,
        "cells": cells,
        "repeats": REPEATS,
        "mode": "fixed-repeats（**显式** --repeats 1；**不**用 --fill-budget）",
        "expected_total_runs": EXPECTED_TOTAL,
        "planned_runs": plan,
        "n_planned_runs": len(plan),
        "budget_minutes": budget_minutes,
        "budget_check_boundary": "整轮边界；不中断正在运行的任务或轮次，不是硬超时",
        "budget_minutes_expected": None,
        "budget_estimation": "由相同 cell 的启动/标定与 pilot 实测节拍估算；当前未知",
        "fz_state": FZ_STATE,
        "not_for": "宣布性能优势（pilot 只查机制/覆盖/节拍）",
    }


def validate(plan: dict) -> list[str]:
    """跑**之前**就能查的四个不变量。任一不过 ⇒ 不进真机。"""
    probs: list[str] = []
    if not _SAFE_TAG.fullmatch(str(plan.get("tag", ""))):
        probs.append("tag 含占位符或不安全字符")
    arms = plan["arms"]
    if len(arms) != 3:
        probs.append(f"臂数 {len(arms)} != 3 ⇒ pilot 必须是 P1/P2/P3 三臂")
    ids = [a["id"] for a in arms]
    if ids != ["P1", "P2", "P3"]:
        probs.append(f"臂编号 {ids} != ['P1','P2','P3']")
    if plan["repeats"] != 1:
        probs.append(f"repeats={plan['repeats']} != 1 ⇒ pilot 用固定 1 轮，"
                     "不得落到 fill-budget（那会跑 3~50 轮）")
    if plan["n_planned_runs"] != EXPECTED_TOTAL:
        probs.append(f"计划任务数 {plan['n_planned_runs']} != 预期 {EXPECTED_TOTAL}")
    # 每条臂的配置必须真的存在
    for a in arms:
        p = _ROOT / plan["cfg_dir"] / f"{a['arm']}.yaml"
        if not p.is_file():
            probs.append(f"{a['id']} 的配置不存在：{p}")
    # cost 族必须恰好 2 条（P2/P3），静态 1 条（P1）
    n_cost = sum(1 for a in arms if a["cost_family"])
    if n_cost != 2:
        probs.append(f"cost 族臂数 {n_cost} != 2 ⇒ 判『机制是否被行使』需要 P2/P3 两条")
    # ★ cells 必须**恰好**是 3 个且与 [`baseline`] + arms_arg 逐条相等
    expect_cells = ["baseline"] + list(ARMS_ARG)
    if plan["cells"] != expect_cells:
        probs.append(f"cells {plan['cells']} != ['baseline'] + {ARMS_ARG} "
                     "⇒ 与矩阵的 cells 构造不符（会多出/少掉格子）")
    if len(plan["cells"]) != 3:
        probs.append(f"格子数 {len(plan['cells'])} != 3 ⇒ 三臂各一次要求恰好 3 格")
    # ★ P1 必须走 baseline 格，且三臂的 matrix_cell 必须互不相同
    if PILOT_ARMS[0]["matrix_cell"] != "baseline":
        probs.append("P1 没有映射到 baseline 格 ⇒ 会变成 4 格")
    mcs = [a["matrix_cell"] for a in arms]
    if len(set(mcs)) != len(mcs):
        probs.append(f"matrix_cell 有重复 {mcs} ⇒ 同一格会被跑两次")
    return probs


def matrix_arguments(plan: dict) -> list[str]:
    """计划、交叉验证和远端命令共享同一组实验参数。"""
    return ["--tag", plan["tag"], "--cfg-dir", plan["cfg_dir"],
            "--baseline", plan["baseline_arg"], "--arms", *plan["arms_arg"],
            "--repeats", str(plan["repeats"]), "--order", "block",
            "--budget-minutes", str(plan["budget_minutes"])]


def cross_check_with_matrix(plan: dict | None = None) -> tuple[bool, str]:
    """★ **真做一遍**：拿 `run_exec_opt_matrix --dry-plan` 的计划与我们的清单对齐。

    静态断言只能证明"我写的数和我想的一致"；这一步才能证明
    **"矩阵真的会跑这 3 项"**（用户点名：只输出计划时就能检查次数）。
    """
    plan = build_plan() if plan is None else plan
    cmd = [sys.executable, str(_ROOT / "scripts" / "run_exec_opt_matrix.py"),
           *matrix_arguments(plan), "--dry-plan"]
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True,
                       encoding="utf-8", env=env, timeout=300)
    if r.returncode:
        return False, f"矩阵 dry-plan 退出码 {r.returncode}: {r.stderr[-600:]}"
    # dry-plan 把 JSON 打在 stdout，前面可能有一行警告
    body = r.stdout[r.stdout.find("{"):]
    try:
        d = json.loads(body)
    except Exception as exc:  # pragma: no cover
        return False, f"无法解析 dry-plan 输出：{type(exc).__name__}: {exc}"
    n = len(d.get("plan") or [])
    cells = d.get("cells") or []
    if n != EXPECTED_TOTAL:
        return False, f"矩阵计划 {n} 项 != 预期 {EXPECTED_TOTAL} 项（cells={cells}）"
    expected = {"tag": plan["tag"], "cells": plan["cells"],
                "repeats": plan["repeats"], "fill_budget": False,
                "plan": plan["planned_runs"], "budget_minutes": plan["budget_minutes"]}
    for key, value in expected.items():
        if d.get(key) != value:
            return False, f"矩阵 {key}={d.get(key)!r} != 清单 {value!r}"
    return True, f"矩阵 dry-plan 确认 **{n} 项** = 3 臂 × 1 轮（cells={cells}）"


def remote_command(plan: dict) -> str:
    """真机执行的**实际命令**。

    ★★ `--baseline` **必须**显式指向 P1（`static_small.yaml`）：矩阵的 cells 是
       `["baseline"] + arms` ⇒ 只有这样才能得到**恰好 3 格**。
       若省掉它（默认 `s4000_fp16.yaml`）并把三臂放进 `--arms`，会变成 **4 格**。

    ⚠️ **不写 `--fill-budget`、不省 `--repeats`** —— 那两个默认值正是"跑到 15 项"的原因。
    """
    if not plan.get("remote_root") or not plan.get("remote_python"):
        raise ValueError("未提供远端目录/Python：先核验，再显式传 --remote-root 和 --remote-python")
    root = _remote_path(plan["remote_root"], name="remote-root")
    python = _remote_path(plan["remote_python"], name="remote-python")
    tag = plan["tag"]
    if not _SAFE_TAG.fullmatch(tag):
        raise ValueError("不安全的 tag")
    # 使用唯一 tag 绑定进度、汇总和 stop 文件；不共享 /root/pilot_*.json。
    extra = ["--out", str(PurePosixPath(root) / "results/raw"),
             "--progress-file", str(PurePosixPath(root) / f"pilot_progress_{tag}.json"),
             "--summary-file", str(PurePosixPath(root) / f"pilot_summary_{tag}.json"),
             "--stop-file", str(PurePosixPath(root) / f"pilot_STOP_{tag}")]
    argv = [python, "scripts/run_exec_opt_matrix.py", *matrix_arguments(plan), *extra]
    return f"cd -- {shlex.quote(root)} && \\\n    {shlex.join(argv)}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print-command", action="store_true", help="只打印真机命令")
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default=None, help="省略则生成 UTC 时间+随机后缀；不得复用旧批次标签")
    ap.add_argument("--cfg-dir", default=CFG_DIR, help="仓库相对配置目录；新模型须用派生配置")
    ap.add_argument("--remote-root", default=None, help="执行端已核验的项目 POSIX 绝对路径")
    ap.add_argument("--remote-python", default=None, help="执行端已核验的 MUSA Python 绝对路径")
    ap.add_argument("--budget-minutes", type=float, default=BUDGET_MINUTES,
                    help="整轮边界预算，不是中断任务的硬超时")
    args = ap.parse_args()

    try:
        plan = build_plan(tag=args.tag, remote_root=args.remote_root,
                          remote_python=args.remote_python, budget_minutes=args.budget_minutes,
                          cfg_dir=args.cfg_dir)
        command = remote_command(plan) if plan["remote_root"] else None
    except ValueError as exc:
        ap.error(str(exc))
    probs = validate(plan)
    ok_x, msg_x = cross_check_with_matrix(plan)

    if args.print_command:
        if command is None:
            ap.error("--print-command 必须同时提供 --remote-root / --remote-python；历史路径尚未核验")
        if probs or not ok_x:
            print("\n".join(probs + ([msg_x] if not ok_x else [])), file=sys.stderr)
            return 1
        print("# PLAN_ONLY：未连接远端，路径/环境/源码/补标定仍需前置验收。")
        print("# 预算仅在整轮边界检查；45 分钟等预算值不是硬超时。")
        print(command)
        return 0

    print("=" * 96)
    print("pilot 三臂执行清单（**跑之前**就能核对任务数）")
    print("=" * 96)
    print(f"tag / cfg-dir : {plan['tag']}  /  {plan['cfg_dir']}")
    print(f"模式          : {plan['mode']}")
    print(f"预算          : {plan['budget_minutes']} min；{plan['budget_check_boundary']}")
    print(f"预计用时      : 待同 cell 节拍；不承诺在预算分钟数内结束")
    print(f"远端路径状态  : {plan['remote_identity_status']}（本脚本不会连接远端核验）")
    print()
    print(f"{'项':<4}{'臂':<14}{'矩阵格':<12}{'控制器':<40}{'用途'}")
    for a in plan["arms"]:
        print(f"{a['id']:<4}{a['arm']:<14}{a['matrix_cell']:<12}{a['controller']:<40}{a['purpose']}")
    print()
    print(f"repeats       : {plan['repeats']}")
    print(f"计划任务      : {plan['n_planned_runs']} 项  = 臂数 {len(plan['arms'])} × 轮数 {plan['repeats']}")
    print(f"明细          : {', '.join(plan['planned_runs'])}")
    print(f"★ P1 走 `--baseline`（矩阵格名 = baseline）；P2/P3 走 `--arms` ⇒ 恰好 3 格")
    print()
    print("FZ 状态（三臂一致）：")
    for k, v in FZ_STATE.items():
        print(f"  {k}: {v}")
    print()
    print("⚠️ pilot 只用于检查**机制 / 覆盖 / 节拍**，**不用于宣布性能优势**。")
    print("⚠️ 判『机制是否被行使』只看 P2/P3（它们才有 budget_kind）；P1 是静态对照。")
    print()
    print("真机命令（**显式 --repeats 1、--baseline 指向 P1**）：")
    if command:
        print(command)
    else:
        print("未生成：远端目录和 Python 待执行端核验。")
        print("计划参数（只用于核对，不是可执行真机命令）：")
        print(shlex.join(matrix_arguments(plan)))
    print()

    # ★ 真做一遍：与矩阵的 dry-plan 对齐（静态断言证明不了"矩阵真会跑 3 项"）
    print(f"矩阵 dry-plan 交叉核对：{'✓ ' if ok_x else '✗ '}{msg_x}")
    print()
    if probs or not ok_x:
        print("★ 校验 **不通过** ⇒ 不得进真机：")
        for p in probs:
            print(f"  ✗ {p}")
        if not ok_x:
            print(f"  ✗ {msg_x}")
    else:
        print(f"★ 校验通过：{plan['n_planned_runs']} 项（= 3 臂 × 1 轮），"
              f"与预期 {EXPECTED_TOTAL} 一致，且矩阵 dry-plan 独立确认。")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {**plan, "validation_problems": probs,
                   "matrix_dryplan_ok": ok_x, "matrix_dryplan_note": msg_x,
                   "ok": (not probs) and ok_x,
                   "remote_command": command,
                   "evidence_class": "PLAN_ONLY（未执行任何批次）"}
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"\n写出 {out}")
    return 0 if (not probs and ok_x) else 1


if __name__ == "__main__":
    sys.exit(main())
