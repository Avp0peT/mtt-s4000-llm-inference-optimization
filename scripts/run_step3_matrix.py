#!/usr/bin/env python3
"""按**穿插顺序**跑 PHASE 1 第③步的性能矩阵。

为什么不能用 ``run_benchmark --repeats 3``
-----------------------------------------
``run_benchmark`` 的重复是**背靠背**的：跑完 A 的 3 次，再跑 B 的 3 次。
而第③步的方案 §1.5 明确要求"配置间**穿插**对照"，理由是**已实测**过：

    同一配置、同一进程重复 2 次：tpot 只差 1.23×，但 stall_p99 差 **4.35×**
    （51.2 vs 222.9 ms），ITL 里出现 249ms/161ms 两个孤立巨停顿。

在这种噪声下，背靠背的顺序会把"某次偶发停顿正好落在配置 B 上"直接归因成
"B 比 A 差"。穿插（每轮取每格一次）让漂移均匀落到所有格上，是唯一能让
配置间比较成立的做法。

本脚本的执行顺序
----------------
``for pass in 0..repeats-1:  for cell in cells:  run(cell, repeat_index=pass)``

即"扫完整张表"为一轮，共扫 ``repeats`` 轮。每格的 3 次因此**在时间上被拉开**，
而不是挤在一起。

记录与可追溯
------------
不另写"执行顺序清单"文件：每次 run 的 ``meta.json`` 自带时间戳与 ``notes``
（notes 里写明 arm 与 concurrency），**顺序可以从产物本身重建**。
再存一份清单就多了一处可能与产物不一致的副本。

纪律
----
* 默认**首个失败即停**：这个矩阵里的失败通常是系统性的（模型名、显存、
  能力不支持），继续跑只会用同样的错烧掉几十分钟，并留下半张表。
  确实想跑完剩下的用 ``--keep-going``。
* **"失败"包含"跑完了但活没干完"。** 判定读**落盘产物**的计数
  （成功/失败/拒绝/未完成），不是看 ``run_once`` 有没有抛异常：真实后端会把
  单请求的异常接住、记进计数、正常返回。只看异常的话，一次
  "0 成功 / 1 失败 / 7 未完成"的 run 会被计成成功（**实测过**）。
  产物一律保留，失败的那一格照样留在证据目录里。
* 不覆盖已有产物：``run_once`` 在目标目录已存在时会抛 ``RunDirConflict``
  （证据保护）。想重跑请用新的 ``--tag``。
* 只跑真实后端。它显式拒绝 ``cache.enabled=true``，所以前缀缓存臂在本步
  不可能存在——这一点由 ``check_step3_configs.py`` 预先拦下。

用法::

    python scripts/run_step3_matrix.py --dry-plan      # 只打印顺序，不跑
    python scripts/run_step3_matrix.py                 # 跑满 12 格 × 3 轮
    python scripts/run_step3_matrix.py --cells static_c1 static_chunk_c1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "benchmark"))
sys.path.insert(0, str(_ROOT / "src"))

from run_benchmark import build_run_id, load_config, run_once  # noqa: E402

#: 默认配置目录 = 第③步**第一次**执行的那一版。当前要跑的版本用 ``--cfg-dir`` 指到
#: ``configs/mthreads/phase1_step3d``（2026-09-24 起；更早的修正版是 ``...3b``）。
#: 各版的参数差异见生成器的 PROFILES。
#: 刻意不改默认值：改了默认值会让"复现旧矩阵"这条命令的语义随时间漂移。
CFG_DIR = _ROOT / "configs" / "mthreads" / "phase1_step3"
ARMS: tuple[str, ...] = ("static", "static_chunk", "adaptive")
CONCURRENCIES: tuple[int, ...] = (1, 2, 4, 8)

def default_cells() -> list[str]:
    return [f"{a}_c{c}" for a in ARMS for c in CONCURRENCIES]


# --------------------------------------------------------------------------- #
# 运行状态：从**已落盘的产物**读，而不是从"函数有没有抛异常"推
# --------------------------------------------------------------------------- #
#
# 为什么必须这样：真实后端对**单请求**的失败是"接住并记录"的——显存不足之类
# 的异常会被写进该请求的 outcome，进程照常收尾、``run_once`` 照常返回。
# 于是"函数返回了"与"活干完了"是两件事。混成一件的后果实测过：
# 离线注入 OOM 后产物是 **0 成功 / 1 失败 / 7 未完成**，而矩阵报告"成功"。
#
# 失败判据必须**能失败**，也必须**不误报**：
#   * 只认"抛异常" ⇒ 上面那种 run 会被计成成功（漏报）；
#   * 把"读不到产物"当成功 ⇒ 后端崩在落盘前也会被计成成功（更漏）；
#   * 把"读不到产物"当失败 ⇒ 正确方向（说不清就停下看）。
# 因此三态：ok / 明确不良 / **不可判定**，后两者一律算失败。

#: 判定"这次 run 干完了活"所需的计数，全部来自 ``meta.json`` 的 ``summary``。
_STATUS_KEYS = ("n_requests", "n_success", "n_failed", "n_rejected", "n_unfinished")


def read_run_status(run_dir: Path) -> dict[str, Any]:
    """读一次 run 的**运行状态**，返回 ``{"ok", "reasons", "counts", ...}``。

    ``counts`` 里取不到的字段是 ``None`` 而**不是** 0（0 是"测出来是 0"）；
    ``ok=False`` 且 ``counts`` 全为 ``None`` 时，``reasons`` 说明的是"不可判定"，
    与"确实干了坏活"是两回事——但都不许当成成功。
    """
    status: dict[str, Any] = {
        "run_dir": str(run_dir),
        "ok": False,
        "reasons": [],
        "counts": dict.fromkeys(_STATUS_KEYS, None),
        "dropped_requests": None,
        "label": None,
    }
    meta_path = Path(run_dir) / "meta.json"
    if not meta_path.is_file():
        status["reasons"].append(
            f"产物缺失：{meta_path} 不存在（后端可能在落盘之前就崩了）"
        )
        return status
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001  —— 解析失败必须显形，不能默认通过
        status["reasons"].append(
            f"meta.json 不可解析：{type(exc).__name__}: {exc}"
        )
        return status

    status["label"] = meta.get("data_label")
    extra = (meta.get("context") or {}).get("extra") or {}
    status["dropped_requests"] = extra.get("dropped_requests")

    summary = meta.get("summary")
    if not isinstance(summary, dict):
        status["reasons"].append(
            "meta.json 里没有 summary 块 ⇒ 运行状态不可判定（不可判定不等于正常）"
        )
        return status

    status["counts"] = {k: summary.get(k) for k in _STATUS_KEYS}
    missing = [k for k, v in status["counts"].items() if not isinstance(v, int)]
    if missing:
        status["reasons"].append(
            f"summary 缺少/非整数计数字段 {missing} ⇒ 运行状态不可判定"
            "（不可判定不等于正常）"
        )
        return status

    c = status["counts"]
    if c["n_requests"] <= 0:
        status["reasons"].append("n_requests<=0：一个请求都没提交")
    for key, human in (
        ("n_failed", "失败"),
        ("n_rejected", "拒绝"),
        ("n_unfinished", "未完成"),
    ):
        if c[key]:
            status["reasons"].append(f"{human} {c[key]} 个")
    if c["n_success"] != c["n_requests"]:
        status["reasons"].append(
            f"成功数 {c['n_success']} != 提交数 {c['n_requests']}"
        )
    if status["dropped_requests"]:
        status["reasons"].append(
            f"后端自报 dropped_requests={status['dropped_requests']}"
            "（有请求没走到 done）"
        )
    status["ok"] = not status["reasons"]
    return status


def format_run_status(status: dict[str, Any]) -> str:
    """把状态压成一行，供失败清单使用。计数不可用时显式说"不可用"。"""
    c = status.get("counts") or {}
    if all(isinstance(v, int) for v in c.values()) and c:
        table = (
            f"结局：成功 {c['n_success']} / 失败 {c['n_failed']} / "
            f"拒绝 {c['n_rejected']} / 未完成 {c['n_unfinished']}"
            f"（共提交 {c['n_requests']}）"
        )
    else:
        table = "结局计数不可用"
    reasons = status.get("reasons") or []
    return table + ("；" + "；".join(reasons) if reasons else "")


def run_matrix(
    *,
    cells: list[str],
    cfgs: dict[str, Any],
    run_ids: dict[str, str],
    out_dir: Path,
    repeats: int,
    keep_going: bool,
    runner: Callable[..., Path] = run_once,
    status_reader: Callable[[Path], dict[str, Any]] = read_run_status,
) -> tuple[int, list[tuple[str, int, str]]]:
    """按穿插顺序执行整张矩阵，返回 ``(完成计数, 失败清单)``。

    **成功 = 没抛异常 且 落盘产物的运行状态正常。** 后者不可省，理由见文件头
    那段注释（注入 OOM 的实测）。

    产物一律**保留**：判定发生在落盘之后，失败的那一格仍完整留在证据目录里
    ——要的是"看得到失败"，不是"删掉失败"。

    ``runner`` / ``status_reader`` 可注入是为了**可测**：这条判据的错误形态
    （看异常不看产物、读不到就当通过）都只会在真机长跑里出现，
    不能只靠"跑一遍试试"验证。默认值就是真实实现。
    """
    total = len(cells) * repeats
    done = 0
    attempt = 0
    failures: list[tuple[str, int, str]] = []

    def _stop_note() -> None:
        print(
            f"已停在第 {attempt}/{total} 次运行（默认首个失败即停）。"
            f"若确认失败是局部的，可加 --keep-going 续跑。",
            file=sys.stderr,
        )

    for p in range(repeats):
        for name in cells:
            attempt += 1
            cfg = cfgs[name]
            print(f"\n{'#' * 78}\n# pass {p}  cell {name}\n{'#' * 78}")
            try:
                run_dir = runner(
                    cfg,
                    dry_run=False,
                    out_dir=out_dir,
                    repeat_index=p,
                    run_id=run_ids[name],
                    extra_meta={"notes": cfg.run.notes},  # type: ignore[attr-defined]
                    verbose=True,
                )
            except Exception as exc:  # noqa: BLE001 —— 任何异常都算这一格失败
                msg = f"{type(exc).__name__}: {exc}"
                print(f"格 {name} pass {p} 失败：{msg}", file=sys.stderr)
                failures.append((name, p, msg))
                if not keep_going:
                    _stop_note()
                    return done, failures
                continue

            status = status_reader(run_dir)
            if not status.get("ok"):
                msg = "运行状态不正常（产物已保留）：" + format_run_status(status)
                print(f"格 {name} pass {p} 失败：{msg}", file=sys.stderr)
                failures.append((name, p, msg))
                if not keep_going:
                    _stop_note()
                    return done, failures
                continue
            done += 1

    return done, failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="按穿插顺序跑第③步矩阵")
    ap.add_argument(
        "--cells", nargs="+", default=None,
        help=f"要跑的格（默认全部 {len(default_cells())} 格）",
    )
    ap.add_argument("--repeats", type=int, default=3, help="轮数（默认 3）")
    ap.add_argument("--tag", default=None, help="run-id 前缀标记，用于区分重跑")
    ap.add_argument("--keep-going", action="store_true", help="出错继续跑其余格")
    ap.add_argument("--dry-plan", action="store_true", help="只打印执行顺序")
    ap.add_argument("--out", default=None, help="产物根目录（默认 results/raw）")
    ap.add_argument(
        "--cfg-dir", default=None,
        help=f"配置目录（默认 {CFG_DIR.relative_to(_ROOT)}，即首跑那版；当前用 "
             f"configs/mthreads/phase1_step3d）",
    )
    args = ap.parse_args(argv)

    cfg_dir = Path(args.cfg_dir) if args.cfg_dir else CFG_DIR

    cells = args.cells or default_cells()
    if args.repeats < 1:
        raise SystemExit("--repeats 必须 >= 1")
    if args.repeats < 3:
        print(
            f"警告：--repeats={args.repeats} < 3，按项目纪律（AGENTS.md 附录 A4）"
            f"其结果不能作为结论。"
        )

    out_dir = Path(args.out) if args.out else _ROOT / "results" / "raw"

    # 先全部加载，避免跑到一半才发现某个配置有问题。
    cfgs: dict[str, object] = {}
    for name in cells:
        path = cfg_dir / f"{name}.yaml"
        if not path.exists():
            raise SystemExit(f"配置不存在：{path}")
        cfgs[name] = load_config(path)

    # 每格的 run_id 在**整个矩阵期间固定**：这样 3 轮的产物落在
    # <id>、<id>-r1、<id>-r2，能被聚合器认成同一配置的 3 次重复。
    # 每轮现算的话会得到 3 个不同的时间戳前缀，聚合成 3 个不同配置。
    stamp = args.tag or time.strftime("%Y%m%d-%H%M%S")
    run_ids = {name: f"{stamp}-{cfg.workload_name}-{name}" for name, cfg in cfgs.items()}

    total = len(cells) * args.repeats
    print("=" * 78)
    print(f"穿插执行顺序（{len(cells)} 格 × {args.repeats} 轮 = {total} 次运行）")
    print("=" * 78)
    n = 0
    for p in range(args.repeats):
        for name in cells:
            n += 1
            print(f"  {n:3d}. pass {p}  {name:<18} -> {run_ids[name]}"
                  + ("" if p == 0 else f"-r{p}"))
    print()
    if args.dry_plan:
        print("（--dry-plan，未执行）")
        return 0

    done, failures = run_matrix(
        cells=cells,
        cfgs=cfgs,
        run_ids=run_ids,
        out_dir=out_dir,
        repeats=args.repeats,
        keep_going=args.keep_going,
    )

    print(f"\n{'=' * 78}")
    print(f"矩阵结束：成功 {done}/{total}，失败 {len(failures)}")
    for name, p, msg in failures:
        print(f"  失败 {name} pass {p}: {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
