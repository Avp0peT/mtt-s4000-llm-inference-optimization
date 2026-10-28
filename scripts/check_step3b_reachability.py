#!/usr/bin/env python3
"""在**开跑之前**确认：修正版控制器的阈值落在"真的会被触发"的位置。

为什么需要这道门
----------------
第③步第一次执行（``p1s3``）的 ``adaptive`` 臂在这件事上是空转：``chunk_req`` 恒为
512/512、``n_recovered=0``。产物齐全、无报错、标签全绿，但那一臂没有测到它声称
要测的东西。事后复盘发现有两个相乘的成因：

* ``chunk_size_init == chunk_size_min`` ⇒ 维度结构上不可动（这条已在生成器里
  被 ``_assert_controller_movable`` 拦下）；
* ``target_tpot_ms(50) < 实测 TPOT P90 下限(61)`` ⇒ 目标不可达，
  控制器**永远**走 ``tpot`` 分支，``ttft``/grow 判据从未被评估。

第二条是**静态检查抓不到**的：它取决于实测信号落在哪里。本脚本补上这一道门——
用**已登记的** P1S3 轨迹里的真实信号，复放到修正版的控制器参数上，看它会不会动、
往哪个方向动。跑一次要几十分钟真机时间，这道门几秒钟。

这道门证明了什么、没证明什么
----------------------------
**证明了**：修正后的阈值与实测信号相交——低并发下 ``ttft`` 分支会触发（放大）、
高并发下 ``tpot`` 分支会触发（收缩）、并且收缩之后压力消退时**恢复路径真的存在**
（不是棘轮）。这三件事都是"配置能不能回答研究问题"的前提。

**没有证明**：闭环终态。复放是**开环**的——喂进去的信号取自 P1S3 的实录
（那时 chunk=512），**不随动作变化**。真实的闭环比这复杂：放大 chunk 会让 TTFT 下降，
于是放大自己就会停下来。所以本脚本只能回答"阈值在不在可触发的位置"，
不能预测各格的最终 chunk 或最终时延。任何把这里的数字当结果引用的做法都是错的。

判据（双向可达 + 恢复可达）
--------------------------
* 至少有一格：chunk 被**放大**到超过初值（``ttft`` 分支可达）；
* 至少有一格：chunk 被**收缩**到低于初值（``tpot`` 分支可达）；
* 恢复探针：人为喂一段"先超目标、后回到带内"的信号，chunk 必须能逐档回到初值。

三者任一不成立就返回非零——**在烧掉真机时间之前**。

用法::

    python scripts/check_step3b_reachability.py
    python scripts/check_step3b_reachability.py --configs configs/mthreads/phase1_step3b \\
        --runs results/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "benchmark"))
sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.controller import AdaptiveController, ControllerConfig  # noqa: E402

DEFAULT_CONFIGS = _ROOT / "configs" / "mthreads" / "phase1_step3b"
DEFAULT_RUNS = _ROOT / "results" / "raw"

#: 复现使用的源矩阵前缀。**刻意写死成 P1S3**：这份 P1S3 轨迹是"修正前"的实测信号，
#: 正是拿它来反解阈值区间的那批数据。若换成 P1S3B 自己的轨迹，就变成用结果验证结果。
SOURCE_RUN_PREFIX = "P1S3-"

#: 复放时的步距。只影响轨迹里 ``steps_since_last_decision`` 这个可读字段——
#: 冷却按**观测周期**计数，与步距无关。取与配置一致的默认值，让打印出来的
#: 轨迹与真机轨迹同形，便于对读。
DEFAULT_CONTROL_INTERVAL = 8


def load_signal_sequence(meta_path: Path) -> list[tuple[float, float]]:
    """从一个 run 的 ``meta.json`` 里按顺序取出 ``(ttft_p90, tpot_p90)``。

    这两项就是控制器**实际读到**的反馈（``_decide`` 用 P90，见 controller.py）。
    缺信号的决策点（例如 step 0 的"无有效观测样本"）**跳过而不是填 0**：
    0 会被读成"测出来是 0"，那是一条假信号，会让复放结果偏乐观。
    """
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    out: list[tuple[float, float]] = []
    for entry in data.get("controller_trace") or []:
        sig = entry.get("signals") or {}
        ttft = sig.get("ttft_p90_ms")
        tpot = sig.get("tpot_p90_ms")
        if ttft is None or tpot is None:
            continue
        out.append((float(ttft), float(tpot)))
    return out


def replay_signals(
    config: ControllerConfig,
    signals: Sequence[tuple[float, float]],
    *,
    control_interval_steps: int = DEFAULT_CONTROL_INTERVAL,
) -> dict[str, Any]:
    """把一段 ``(ttft_p90, tpot_p90)`` 序列开环喂给一个**新建**的控制器。

    单个样本用 ``[x]`` 传入即可：``percentile([x], 90) == x``（见 metrics.percentile
    对 len==1 的处理），所以复放读到的 P90 与实录逐字相同，不引入插值误差。

    返回轨迹摘要。返回值里同时给 ``chunk_series``（逐周期），因为"最终值"会把
    中间的方向反转抹掉——而"是否两个方向都动过"恰恰是本脚本要回答的问题。

    ``chunk_series`` **以初值为第 0 项**。这一点不是格式偏好：若只记 observe
    **之后**的值，"第一步就收缩"这类动作在相邻比较里不可见——实测症状是
    c8 明明收敛到 512，转移计数却报"收缩 0 次、min=max=512"，
    自相矛盾且看起来像"没动"。
    """
    ctl = AdaptiveController(config)
    init = ctl.state.chunk_size
    chunk_series: list[int] = [init]
    bottlenecks: list[str] = []
    for i, (ttft, tpot) in enumerate(signals):
        action = ctl.observe(
            step=(i + 1) * max(1, control_interval_steps),
            ttft_samples=[ttft],
            tpot_samples=[tpot],
        )
        chunk_series.append(ctl.state.chunk_size)
        bottlenecks.append(action.bottleneck)
    st = ctl.state
    up = sum(1 for a, b in zip(chunk_series, chunk_series[1:]) if b > a)
    down = sum(1 for a, b in zip(chunk_series, chunk_series[1:]) if b < a)
    return {
        "init_chunk": init,
        "n_signals": len(signals),
        "n_cycles": st.n_cycles,
        "n_updates": st.n_updates,
        "n_recovered": st.n_recovered,
        "n_clamped": st.n_clamped,
        "n_skipped_cooldown": st.n_skipped_cooldown,
        "n_vetoed_memory": st.n_vetoed_memory,
        "n_vetoed_cache": st.n_vetoed_cache,
        "chunk_min": min(chunk_series) if chunk_series else init,
        "chunk_max": max(chunk_series) if chunk_series else init,
        "chunk_final": chunk_series[-1] if chunk_series else init,
        "n_grow_steps": up,
        "n_shrink_steps": down,
        "bottlenecks": sorted({b for b in bottlenecks}),
        "chunk_series": chunk_series,
    }


#: 恢复探针用的**绝对**信号（毫秒），刻意不按目标值缩放。
#:
#: 锚在 P1S3 实测区间上：压力段取 c8 实测 TPOT P90 的峰值附近（294 → 用 300）、
#: 消退段取 c1 实测的 TPOT P90（61.5 → 用 60）与**实际可达**的 TTFT
#: （不切分时实测 448.5 → 用 450）。
#:
#: 为什么必须绝对：本函数第一版把信号写成 ``target × 3.0``。那个写法让探针
#: **不可能失败**——目标定得越高，压力信号跟着涨，永远"超目标"，于是"能逼出收缩"
#: 恒真。这与整个脚本要防的失效是同一类：一个不会失败的检查等于没有检查。
#: 探针要回答的是"给定的实测工况下控制器会不会动"，所以输入必须是实测值，
#: 不能是目标的函数。
_PROBE_STRESS_TPOT_MS = 300.0
_PROBE_STRESS_TTFT_MS = 1000.0
_PROBE_RELAX_TPOT_MS = 60.0
_PROBE_RELAX_TTFT_MS = 450.0


def recovery_probe(config: ControllerConfig) -> dict[str, Any]:
    """恢复路径的**直接**探针：先逼出一次收缩，再喂带内信号，看它是否逐档回到初值。

    为什么要单独探一次：复放实录信号的结论依赖那份负载的形状——若负载一直超目标，
    控制器收缩后就没有"压力消退"的时刻，``n_recovered`` 自然为 0，而这**不能**推出
    "恢复路径坏了"。这个探针把两件事分开：负载形状 vs 控制器是否有恢复能力。

    信号取值见上面的 ``_PROBE_*`` 常量（绝对值，理由在那里）。
    """
    ctl = AdaptiveController(config)
    init = ctl.state.chunk_size
    # 1) 压力段：TPOT 明显超目标 ⇒ 收缩（第一次之后进入冷却，符合设计）
    for i in range(4):
        ctl.observe(
            step=(i + 1) * DEFAULT_CONTROL_INTERVAL,
            ttft_samples=[_PROBE_STRESS_TTFT_MS],
            tpot_samples=[_PROBE_STRESS_TPOT_MS],
        )
    after_stress = ctl.state.chunk_size
    # 2) 消退段：双目标都回到带内 ⇒ 应逐档恢复到初值
    for i in range(8):
        ctl.observe(
            step=(5 + i) * DEFAULT_CONTROL_INTERVAL,
            ttft_samples=[_PROBE_RELAX_TTFT_MS],
            tpot_samples=[_PROBE_RELAX_TPOT_MS],
        )
    return {
        "init_chunk": init,
        "after_stress_chunk": after_stress,
        "after_relax_chunk": ctl.state.chunk_size,
        "n_recovered": ctl.state.n_recovered,
        "shrank": after_stress < init,
        "recovered_to_init": ctl.state.chunk_size == init,
        # 探针自身是否有效：压力段都没逼出收缩的话，"恢复了没有"这个问题无从谈起。
        "probe_valid": after_stress < init,
    }


def _iter_source_runs(runs_root: Path) -> Iterable[tuple[str, Path]]:
    """按格名产出 ``(cell, meta.json)``，只取 ``adaptive`` 臂、且首轮（无 ``-rN``）。

    只取首轮是因为 3 次的信号几乎逐字相同（已实测），多取只是重复打印同一张表。
    """
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        name = d.name
        if not name.startswith(SOURCE_RUN_PREFIX):
            continue
        if "-adaptive_" not in name or name.rsplit("-", 1)[-1].startswith("r"):
            continue
        meta = d / "meta.json"
        if not meta.exists():
            continue
        cell = name[len(SOURCE_RUN_PREFIX):]
        yield cell, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="修正版控制器阈值的开环可达性检查")
    ap.add_argument("--configs", default=str(DEFAULT_CONFIGS), help="修正版配置目录")
    ap.add_argument("--runs", default=str(DEFAULT_RUNS), help="P1S3 产物根目录")
    ap.add_argument("--cell", default="adaptive_c1", help="从哪个配置读控制器参数")
    args = ap.parse_args(argv)

    cfg_dir = Path(args.configs)
    cfg_path = cfg_dir / f"{args.cell}.yaml"
    if not cfg_path.exists():
        print(f"预检失败：配置不存在 {cfg_path}", file=sys.stderr)
        return 1

    from run_benchmark import load_config  # 延迟 import：只需本机路径可达

    cfg = load_config(cfg_path)
    ctl_cfg: ControllerConfig = cfg.controller
    # 配置目录可能在仓库之外（测试的临时目录、或有人把矩阵放到别处），
    # relative_to 会直接抛 ValueError —— 打印路径这种小事不该让整道门崩掉。
    try:
        shown = cfg_path.relative_to(_ROOT)
    except ValueError:
        shown = cfg_path
    print("=" * 96)
    print(f"控制器参数（取自 {shown}）")
    print("=" * 96)
    print(
        f"  target_tpot_ms={ctl_cfg.target_tpot_ms}  hysteresis={ctl_cfg.hysteresis}"
        f"  ⇒ TPOT 上限 = {ctl_cfg.target_tpot_ms * (1 + ctl_cfg.hysteresis):.1f} ms"
    )
    print(
        f"  target_ttft_ms={ctl_cfg.target_ttft_ms}"
        f"  ⇒ TTFT 上限 = {ctl_cfg.target_ttft_ms * (1 + ctl_cfg.hysteresis):.1f} ms"
    )
    print(
        f"  chunk_size: init={ctl_cfg.chunk_size_init} min={ctl_cfg.chunk_size_min}"
        f" max={ctl_cfg.chunk_size_max} step={ctl_cfg.chunk_size_step}"
        f"  ⇒ 可放大 {(ctl_cfg.chunk_size_max - ctl_cfg.chunk_size_init) // ctl_cfg.chunk_size_step} 档"
        f" / 可收缩 {(ctl_cfg.chunk_size_init - ctl_cfg.chunk_size_min) // ctl_cfg.chunk_size_step} 档"
    )
    print(
        f"  cooldown_cycles={ctl_cfg.cooldown_cycles}（单位=观测周期）"
        f"  max_batched_tokens: init={ctl_cfg.max_batched_tokens_init}"
        f" min={ctl_cfg.max_batched_tokens_min} max={ctl_cfg.max_batched_tokens_max}"
    )

    runs_root = Path(args.runs)
    rows: list[dict[str, Any]] = []
    for cell, meta in _iter_source_runs(runs_root):
        signals = load_signal_sequence(meta)
        if not signals:
            continue
        rep = replay_signals(ctl_cfg, signals)
        rep["cell"] = cell
        rep["source"] = meta.parent.name
        rows.append(rep)

    if not rows:
        print(
            f"预检失败：{runs_root} 下找不到 {SOURCE_RUN_PREFIX}*-adaptive_* 的轨迹。"
            f"复放需要**实录信号**；没有它就无从判断阈值是否可达。",
            file=sys.stderr,
        )
        return 1

    print()
    print("=" * 96)
    print("开环复放：把 P1S3 实录的 ttft_p90 / tpot_p90 喂给修正版控制器参数")
    print("=" * 96)
    print(
        f"{'格':<18}{'信号':>5}{'周期':>5}{'改动':>5}{'放大':>5}{'收缩':>5}{'恢复':>5}"
        f"{'clamp':>6}{'冷却跳过':>9}  {'chunk min→max (末)':>20}  瓶颈"
    )
    for r in rows:
        span = f"{r['chunk_min']}→{r['chunk_max']} ({r['chunk_final']})"
        print(
            f"{r['cell']:<18}{r['n_signals']:>5}{r['n_cycles']:>5}{r['n_updates']:>5}"
            f"{r['n_grow_steps']:>5}{r['n_shrink_steps']:>5}{r['n_recovered']:>5}"
            f"{r['n_clamped']:>6}{r['n_skipped_cooldown']:>9}"
            f"  {span:>20}"
            f"  {','.join(r['bottlenecks'])}"
        )

    init = rows[0]["init_chunk"]
    grew = [r for r in rows if r["chunk_max"] > init]
    shrank = [r for r in rows if r["chunk_min"] < init]
    probe = recovery_probe(ctl_cfg)

    print()
    print(f"chunk_size 初值 = {init}")
    print(
        f"  放大可达：{'是' if grew else '否'}  "
        f"（{'、'.join(r['cell'] for r in grew) or '没有任何一格被放大'}）"
    )
    print(
        f"  收缩可达：{'是' if shrank else '否'}  "
        f"（{'、'.join(r['cell'] for r in shrank) or '没有任何一格被收缩'}）"
    )
    print(
        f"  恢复探针：压力段后 {probe['init_chunk']}→{probe['after_stress_chunk']}，"
        f"消退后 →{probe['after_relax_chunk']}（n_recovered={probe['n_recovered']}）"
    )

    # 载荷顺序（低并发净放大、高并发净收缩）是研究主张本身。判据用**净方向**
    # （末值 vs 初值），不用"任一次放大/任一次收缩"——后者在开环下会**恒假**：
    # 中并发格（c2/c4）的实录信号既有超 TTFT 上限的段、也有超 TPOT 上限的段，
    # 控制器来回动，两个方向都出现过，于是"放大格的最大并发 < 收缩格的最小并发"
    # 永不成立。净方向才是"这一格最终被推到哪一侧"。
    # 开环复放只能给出提示，不足以拦下一次真机运行：闭环里放大 chunk 会让 TTFT
    # 下降、放大自己就会停；顺序最终得由闭环产生。
    import re

    def _conc(cell: str) -> int:
        m = re.search(r"_c(\d+)$", cell)
        return int(m.group(1)) if m else -1

    net_grew = [r["cell"] for r in rows if r["chunk_final"] > init]
    net_shrank = [r["cell"] for r in rows if r["chunk_final"] < init]
    both_way = [
        r["cell"]
        for r in rows
        if r["n_grow_steps"] > 0 and r["n_shrink_steps"] > 0
    ]
    ordered = (
        bool(net_grew)
        and bool(net_shrank)
        and max(_conc(c) for c in net_grew) < min(_conc(c) for c in net_shrank)
    )
    print(
        f"  净方向：放大 {sorted(_conc(c) for c in net_grew)} / 收缩 {sorted(_conc(c) for c in net_shrank)}"
        f"  ⇒ {'符合预期（低并发净放大、高并发净收缩）' if ordered else '**不符合**预期'}；"
        f"  格内双向振荡：{sorted(_conc(c) for c in both_way) or '无'}"
    )

    problems: list[str] = []
    if not grew:
        problems.append(
            "没有任何一格被放大 ⇒ ttft 分支不可达。多半是 target_ttft_ms 定得太高"
            "（实测 TTFT P90 最低 632ms，上限须低于它才可能触发）。"
        )
    if not shrank:
        problems.append(
            "没有任何一格被收缩 ⇒ tpot 分支不可达。多半是 target_tpot_ms 定得太高"
            "（实测 TPOT P90 最高 294ms，上限须低于它才可能触发）。"
        )
    if not probe["shrank"]:
        problems.append(
            "恢复探针的压力段没能逼出收缩 ⇒ tpot 上限高于探针信号，探针本身失效"
            "（不是控制器的问题，是探针参数的问题）。"
        )
    if not probe["recovered_to_init"]:
        problems.append(
            f"恢复探针没能回到初值（停在 {probe['after_relax_chunk']}，期望 {probe['init_chunk']}）"
            f" ⇒ 恢复路径可疑（棘轮）。"
        )

    print()
    print("⚠️ 本检查是**开环复放**：信号取自 P1S3 实录（当时 chunk=512），不随动作变化。")
    print("   它只能证明「阈值落在可触发的位置」，**不能**预测闭环终态——放大 chunk 会让")
    print("   TTFT 下降，于是放大自己就会停。任何把上表数字当结果引用的做法都是错的。")
    print()
    if problems:
        for p in problems:
            print(f"可达性预检失败：{p}", file=sys.stderr)
        return 1
    print("可达性预检通过：放大/收缩两个方向都可达，且恢复路径存在（非棘轮）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
