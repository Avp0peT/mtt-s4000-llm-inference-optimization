#!/usr/bin/env python3
"""把**控制器门限**标定到信号分布上（与 ``calibrate_slo.py`` 配对）。

为什么需要它
------------
``adaptive_c4`` 在 P1S3B 上跨重复不可复现（三次 TTFT = 481.4 / 481.6 / 652.1），
根因是**门限落在工作点分布内部**：

* 放大分支要求 ``ttft_p90 > target_ttft_ms·(1+h)`` **且**
  ``tpot_headroom = (target_tpot_ms − tpot_p90)/target_tpot_ms > h``，
  即 ``tpot_p90 < target_tpot_ms·(1−h)``；
* P1S3B 的 ``target_tpot_ms=110``、``h=0.1`` ⇒ 放大门限 = **99.0 ms**，
  而 c4 的 ``tpot_p90`` 实测 **97.8–99.2** ⇒ 每次放大决策的 headroom 都在
  0.1004–0.1110（门槛 0.1000），整条爬坡近似**抛硬币**。

后果不是"稍微抖一下"：负载 ``prompt_len = 2048 = chunk_size_max``，
``chunk_size`` 卡在 1536 时整个 prompt 要拆成两步 ⇒ 那 4 个请求 TTFT
``454.7 → 652.1``（**+43%**）。**0.2% 的观测差异造出 43% 的时延后果。**

所以标定时要问的不是"阈值配了没有"，而是"阈值与工作点分布**分得开吗**"。

本脚本做什么、不做什么
----------------------
**做**：从已登记的产物里读出控制器**实际读到**的
``controller_trace[*].signals.{ttft,tpot}_p90_ms``（**含全部重复**），给出
（a）各格的信号分布与重复数，（b）当前参数下各分支的可触发比例，
（c）一组候选 ``(target_tpot_ms, hysteresis)`` 的对比，
（d）**跨重复对照**——同一条件下不同重复的判定是否一致（这**才是**可复现性
的问法），并带上每次重复的**阶段**（决策点 step）与**控制状态**
（实际走的瓶颈分支、``chunk_size`` 范围 + 变化次数），因为"判定相同"不等于
"行为相同"，
（e）**口径核对**（自检）——把"``classify`` 与 ``_decide`` 一一对应"这句
**断言**变成每次运行都算一遍的**不变量**（见 :func:`check_branch_consistency`），
漂移即退出码 6。

**不做**：不预测闭环终态，不写配置。

⚠️ **本工具不是"可复现性"的阻断门。** 它能给的是"哪一格要盯着看"：
判据是**静态口径**（只按信号落不落在门限外判可触发，不含冷却与反馈），
而"是否真的跨重复一致"要由真机重复给出。把它当门会把"样本内不决断"
误读成"不可复现"——两者不是一回事。

三条必须带着读的限定
--------------------
1. **这是静态口径**：只按"信号落不落在门限外"判可触发，**不含冷却**、
   也不含"动作会反过来改变后续信号"这一反馈。所以它是**上界**，
   真正的把关仍是 ``check_step3b_reachability.py`` 的开环复放与真机重跑。
2. **信号必须是控制器读到的那个量**。``controller_trace`` 里的 ``*_p90_ms`` 是
   决策时刻"最近若干个样本"的 P90，与报表里的逐请求分位**不是同一个量**
   （本例 ≈98.9 vs ``summary.tpot.p50`` 110.15）。拿报表分位标定会得到相反结论。
3. **"走的瓶颈分支"不是"真的改了值"**。``_decide`` 在 ``ttft`` 分支里被
   headroom 门槛拒时返回的 ``bottleneck`` **仍是 ``"ttft"``**，冷却中的返回
   也照记原分支。所以 ``bottlenecks`` 里的 ``ttft`` 计数 = 放大 + 被门槛拒
   （这正是 §(e) 核对的那条不变量，也是这里曾差点被误读成"工具多了一个门"的地方）。
   实测：P1S3D 的 ``c8`` 一轮 ``tpot×17 ttft×1``，而 ``放大=0 / 门槛拒=1``
   ⇒ 那一次 ``ttft`` 一次都没放大。

⚠️ **格的身份是 ``(批次, 格名)``**。重复序号只在**同一批次内**有意义：
只按格名分组会把 P1S3B 的 ``r0`` 与 P2 的 ``r0``（**不同条件**，P2 含
``readback_ms`` 计时修复）并成"同一条件的两次重复"，在跨重复表里出现**两行 r0**，
看起来像"同条件重复却给出不同判定"。那是**未登记的混淆**。

用法::

    python scripts/calibrate_controller_gate.py --raw results/raw --prefix P1S3B,P2
    python scripts/calibrate_controller_gate.py --raw results/raw \\
        --targets 110,130,150 --hysteresis 0.1,0.05
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from ascend_opt.metrics import percentile  # noqa: E402

# 复用既有的信号读取函数（"先看有没有现成的"）：它已经处理了
# "缺信号的决策点跳过而不是填 0"这条口径，别在这里再写一份。
from check_step3b_reachability import load_signal_sequence  # noqa: E402

DEFAULT_RAW = _ROOT / "results" / "raw"
#: 只标定 ``adaptive`` 臂：门限只对它起作用，别的臂不看信号。
ADAPTIVE_MARK = "-adaptive_"


def _iter_adaptive_runs(
    raw: Path, prefixes: Sequence[str]
) -> list[tuple[str, str, int, Path]]:
    """产出 ``(批次, 格名, 重复序号, meta.json)``，**含全部重复**。

    ⚠️ 为什么不再"只取首轮"：首轮并不比其它轮更权威，排除它只是历史遗留。
    标定要回答两件事，**两件都需要全部重复**：

    * 门限与工作点分布**分不分得开** —— 样本越多越可信，砍掉 2/3 只是白白
      降低分辨率（同一个点在中位数附近，两轮之间就可能翻转）；
    * 同一条件下不同重复的**决策是否分歧** —— 这**才是**"可复现性"的定义，
      而只看一轮**根本无从比较**。

    重复序号取自目录名尾部的 ``-rN``；首轮（无后缀）记 ``0``。
    格名会**剥掉** ``-rN``，这样同一格的各轮才能归到一组。
    """
    out: list[tuple[str, str, int, Path]] = []
    for d in sorted(p for p in raw.iterdir() if p.is_dir()):
        batch, _, rest = d.name.partition("-")
        if batch not in prefixes or ADAPTIVE_MARK not in d.name:
            continue
        tail = rest.rsplit("-", 1)[-1]
        repeat_index = 0
        if len(tail) == 2 and tail[0] == "r" and tail[1].isdigit():
            repeat_index = int(tail[1])
            rest = rest.rsplit("-", 1)[0]
        meta = d / "meta.json"
        if meta.exists():
            out.append((batch, rest, repeat_index, meta))
    return out


def load_trace_facts(meta_path: Path) -> dict[str, Any]:
    """从一次 run 的 ``controller_trace`` 里取出**运行 / 阶段 / 控制状态**。

    为什么标定要带着这些：``classify`` 只看 ``(ttft_p90, tpot_p90)`` 两个数，
    是**静态口径**——它答不了"这一轮控制器实际动了没有"。同一个判定
    （例如"样本跨越门限"）背后可能是完全不同的行为：
    一轮收缩了 4 次、另一轮一次没动。跨重复比较必须同时看到行为，否则
    "判定相同"会被读成"结果相同"。

    取出的几项：
      * ``steps``        —— 各决策点所处的**阶段**（step 序列）；
      * ``bottlenecks``  —— 每个决策点**实际走的瓶颈分支**（``_decide`` 的返回）；
      * ``chunk_range``  —— 轨迹里 ``state.chunk_size`` 的 min/max；
      * ``chunk_changes``—— ``chunk_size`` **变了几次**（含首次收缩）；
      * ``chunk_final``  —— 终值；
      * ``n_trace_entries`` —— 轨迹条数（**含没有信号的条目**，见下）。
    缺字段一律不猜：对应项记 ``None`` / 空表，而不是 0。

    ⚠️ **``bottleneck`` 记的是"选中了哪个分支"，不是"真的改了值"。** 这条必须
    带着读：``_decide`` 在 ``ttft`` 分支里被 headroom 门槛拒时，返回的
    ``bottleneck`` **仍是** ``"ttft"``（controller.py 的 ``tpot_headroom <=
    hysteresis`` 那一支）；冷却中的返回也照记原分支。所以 ``bottlenecks`` 里
    出现 ``ttft`` **不等于**控制器放大过。实测例子：P1S3D 的 ``c8`` 一轮里
    ``tpot×17 ttft×1``，而静态分类给出 ``n_grow=0``、``门槛拒=1`` ⇒
    那一次 ``ttft`` **一次都没放大**（见 :func:`check_branch_consistency`）。
    "到底动没动"要看 ``chunk_changes`` / ``chunk_range``。

    ⚠️ **``chunk_range`` 只是 min..max，不是轨迹。** 两轮可以有**相同**的
    range 而在不同的**时刻**走到不同的值 —— 这正是同一 range 下 TTFT 481 与
    652 并存的原因。``chunk_changes`` 只补"动过几次"，仍不给出**什么时候动**；
    要顺序请读 ``steps`` 与原始 ``controller_trace``。

    ⚠️ ``n_trace_entries`` 会**大于**有信号的决策点数：``load_signal_sequence``
    会跳过没有有效样本的条目（例如 step 0），而这里是原样照收。差额不是缺陷，
    但必须被解释掉 —— 由 :func:`check_branch_consistency` 每次显式核对。

    ``own_thresholds`` —— **该 run 自己用的** ``(target_tpot_ms, target_ttft_ms,
    hysteresis)``（取自 ``meta["config"]["controller"]``）。为什么必须从产物里读
    而不是用 ``--targets``：口径核对问的是"**静态分类复刻得对不对**"，只有在
    阈值与该 run 实际用的那组一致时才有意义 —— 拿别的阈值去对，两边本来就
    **应该**不同，核对会恒失败，守卫立刻退化成噪声。
    取不到就记 ``None``（三态：**没查** ≠ 不可用），由
    :func:`check_branch_consistency` 报"不可判定"而不是判它过。
    """
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    entries = data.get("controller_trace") or []
    steps: list[Any] = []
    bottlenecks: list[str] = []
    chunk_sizes: list[int] = []
    for entry in entries:
        steps.append(entry.get("step"))
        if entry.get("bottleneck") is not None:
            bottlenecks.append(str(entry["bottleneck"]))
        state = entry.get("state") or {}
        cs = state.get("chunk_size")
        if isinstance(cs, int):
            chunk_sizes.append(cs)
    ctrl = (data.get("config") or {}).get("controller") or {}
    own = (
        ctrl.get("target_tpot_ms"),
        ctrl.get("target_ttft_ms"),
        ctrl.get("hysteresis"),
    )
    return {
        "run": meta_path.parent.name,
        "steps": steps,
        "bottlenecks": bottlenecks,
        "chunk_range": (
            (min(chunk_sizes), max(chunk_sizes)) if chunk_sizes else None
        ),
        "chunk_changes": sum(
            1 for a, b in zip(chunk_sizes, chunk_sizes[1:]) if a != b
        ),
        "chunk_final": chunk_sizes[-1] if chunk_sizes else None,
        "n_trace_entries": len(entries),
        "own_thresholds": own if all(v is not None for v in own) else None,
    }


def cross_repeat_verdict(verdicts: Sequence[str]) -> tuple[str, str]:
    """同一格**各次重复的判定**是否一致 —— 这才是"可复现性"的问法。

    ``classify`` 的 ``verdict`` 说的是"**样本内部**门限分不分得开"，
    它**不是**可复现性的答案：一条序列里两种结果并存，完全可能是同一次运行内
    负载本来就在两个工作点之间移动，与"换个重复就换结论"是两件事。
    真正的可复现性要拿**同一条件下不同重复**去比。

    三态（**"单一重复"必须显形**：一次重复无从比较，不得据此断言可复现）：
      ``重复间一致`` / ``重复间分歧`` / ``单一重复``。
    """
    if len(verdicts) < 2:
        return (
            "单一重复",
            f"只有 {len(verdicts)} 次重复 ⇒ 无从比较，"
            "**不得**据此断言「可复现」（项目纪律：重复 < 3 不出结论）",
        )
    uniq = sorted(set(verdicts))
    if len(uniq) == 1:
        return "重复间一致", f"{len(verdicts)} 次重复的判定均为『{uniq[0]}』"
    return (
        "重复间分歧",
        f"{len(verdicts)} 次重复出现 {len(uniq)} 种判定 {uniq}",
    )


def check_branch_consistency(
    classified: dict[str, Any],
    *,
    bottlenecks: Sequence[str],
    n_signals: int,
    n_trace: int,
) -> dict[str, Any]:
    """核对**静态分类**与**轨迹里记下的瓶颈分支**对不对得上（口径漂移守卫）。

    为什么需要它：``classify`` 的 docstring 写着"与实现一一对应"。这是个**断言**，
    而断言如果不被检查就会漂移 —— 一旦 ``_decide`` 改了分支、或本脚本抄漏了
    一个门槛，表里的"门槛拒 42/192"就会与"实际走的 ``ttft×30``"各说各话，
    而**两边看起来都很正常**。所以这里把它变成一个每次运行都算一遍的不变量。

    四条真不变量（前两条是**相等**，不是数量级相近）：

    ============================== ============================================
    ``shrink``                     ``n_shrink == #(bottleneck=="tpot")``
    ``ttft_branch``                ``n_grow + n_gate_rejected == #("ttft")``
    ``signals_complete``           四类之和 == 有信号的决策点数
    ``trace_complete``             ``#("tpot")+#("ttft")+#(带内) == n_trace``
    ============================== ============================================

    ⚠️ **``ttft_branch`` 为什么必须是"和"，不能是 ``n_grow``**：``_decide`` 在
    ``ttft`` 分支里被 headroom 门槛拒（``tpot_headroom <= hysteresis``）时
    返回的 ``bottleneck`` **仍是 ``"ttft"``**。所以轨迹里的 ``ttft`` 计数 =
    放大 + 被门槛拒。拿 ``n_grow`` 去对就会**恒不相等**，而那不是"实现有多余的
    门"，是**读法错了**。（本工具初版正是在这里差点误判成"工具多了一个门"。
    实测坐实：P1S3D 的 ``c8`` 一轮 ``门槛拒=1 / 放大=0``，轨迹正好 ``ttft×1``。）

    ``skipped_explained`` 把差额解释掉：``n_trace - n_signals`` 必须由
    "带内轨迹条目数 − ``n_not_eligible``" 解释（``load_signal_sequence``
    跳过没有有效样本的条目，而它们的分支是 ``none``/``recover``）。
    解释不了 ⇒ 说明有条目**带着信号却被跳过**，那才是真异常。

    ``unmodelled_memory``：``memory`` 分支本脚本判不了（拿不到显存信号）
    ⇒ 出现即不成立，必须显形，不能混进"带内"。
    """
    counts: dict[str, int] = {}
    for b in bottlenecks:
        counts[b] = counts.get(b, 0) + 1
    trace_shrink = counts.get("tpot", 0)
    trace_ttft = counts.get("ttft", 0)
    # ``memory`` 单独拿出来：它判不了，不能算进"带内"去把差额凑平。
    trace_band = counts.get("none", 0) + counts.get("recover", 0)
    trace_total = trace_shrink + trace_ttft + trace_band + counts.get("memory", 0)

    checks = {
        "shrink": classified["n_shrink"] == trace_shrink,
        "ttft_branch": (
            classified["n_grow"] + classified["n_gate_rejected"] == trace_ttft
        ),
        "signals_complete": n_signals == (
            classified["n_shrink"] + classified["n_grow"]
            + classified["n_gate_rejected"] + classified["n_not_eligible"]
        ),
        "trace_complete": n_trace == trace_total,
        "skipped_explained": (
            n_trace - n_signals
            == trace_band - classified["n_not_eligible"]
        ),
        "unmodelled_memory": counts.get("memory", 0) == 0,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "trace_counts": counts,
        "n_signals": n_signals,
        "n_trace": n_trace,
    }


def report_branch_consistency(
    cells: Sequence[tuple[str, str]],
    per_cell: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    """打印并返回**口径漂移**清单（空表 = 全部对上）。

    与 ``--targets`` / ``--hysteresis`` **无关**：这里用的是每个 run 自己
    ``meta.json`` 里的那组阈值，因为核对问的是"``classify`` 复刻 ``_decide``
    复刻对了没有"，而不是"换一组参数会怎样"。用候选阈值去对会**恒失败**
    （两边本来就**应该**不同），守卫立刻退化成噪声。

    返回值是漂移描述串的列表（调用方据此决定退出码）。读不到阈值的 run 记入
    "不可判定"并打印 —— 三态：**没查** ≠ 通过。
    """
    print()
    print("  口径核对（**用该 run 自己的阈值**；静态分类 vs 轨迹记下的分支）")
    print(f"    {'批次':<8}{'格':<26}{'重复':>4}  {'收缩':>4}{'放大':>4}{'门槛拒':>6}"
          f"{'未评估':>7}  {'轨迹分支计数':<26}{'核对':<14}")
    drift: list[str] = []
    undecidable: list[str] = []
    for key in cells:
        batch, cell = key
        for rep in sorted(per_cell[key]["per_repeat"],
                          key=lambda x: x["repeat_index"]):
            label = f"r{rep['repeat_index']}"
            own = rep["own_thresholds"]
            seen: dict[str, int] = {}
            for b in rep["bottlenecks"]:
                seen[b] = seen.get(b, 0) + 1
            bneck = " ".join(f"{k}×{v}" for k, v in sorted(seen.items())) or "n/a"
            if own is None:
                # 三态：读不到阈值 = **没查**，不是"通过"。
                undecidable.append(f"{batch}/{cell} {label}")
                print(f"    {batch:<8}{cell:<26}{label:>4}  "
                      f"{'—':>4}{'—':>4}{'—':>6}{'—':>7}  "
                      f"{'（产物里没有控制器阈值）':<26}{'不可判定':<14}")
                continue
            o_tpot, o_ttft, o_hys = float(own[0]), float(own[1]), float(own[2])
            rc = classify(rep["sigs"], target_tpot_ms=o_tpot,
                          target_ttft_ms=o_ttft, hysteresis=o_hys)
            chk = check_branch_consistency(
                rc,
                bottlenecks=rep["bottlenecks"],
                n_signals=len(rep["sigs"]),
                n_trace=rep["n_trace_entries"],
            )
            if chk["ok"]:
                verdict_txt = "✓ 一致"
            else:
                bad = ",".join(k for k, v in chk["checks"].items() if not v)
                verdict_txt = f"✗ 漂移:{bad}"
                drift.append(
                    f"{batch}/{cell} {label} [{bad}] "
                    f"阈值={o_tpot:g}/{o_ttft:g}/{o_hys:g}"
                )
            print(f"    {batch:<8}{cell:<26}{label:>4}  "
                  f"{rc['n_shrink']:>4}{rc['n_grow']:>4}{rc['n_gate_rejected']:>6}"
                  f"{rc['n_not_eligible']:>7}  {bneck:<26}{verdict_txt:<14}")
    if drift:
        for d in drift:
            print(f"    ✗ 口径漂移：{d}")
        print("      ⇒ 静态分类与轨迹的分支计数对不上。**先修工具再标定**：")
        print("        表里的「门槛拒/放大/未评估」在这些格上不可信。")
    elif undecidable:
        print(f"    · {len(undecidable)} 个 run 的产物里没有控制器阈值 ⇒ "
              f"这些格**未核对**（不是'通过'）。")
    else:
        print("    ⇒ 全部 run 逐条对上：`n_shrink==#tpot`、"
              "`n_grow+门槛拒==#ttft`、分类与轨迹各自自洽。")
    return drift



def _stat(vs: Sequence[float]) -> dict[str, float]:
    s = sorted(vs)
    return {"n": float(len(s)), "min": s[0], "p50": percentile(s, 50), "max": s[-1]}


def _lo_hi(v: dict[str, float]) -> str:
    return f"{v['min']:.1f}/{v['p50']:.1f}/{v['max']:.1f}"


def classify(
    signals: Sequence[tuple[float, float]],
    *,
    target_tpot_ms: float,
    target_ttft_ms: float,
    hysteresis: float,
) -> dict[str, Any]:
    """按 ``AdaptiveController._decide`` 的判据给每个决策点分类（**静态口径**）。

    与实现一一对应，顺序也一致（``memory`` ⇒ ``tpot`` ⇒ ``ttft`` ⇒ ``recover`` ⇒ ``none``）。
    其中两个分支**只看信号序列判不出来**，各自显式记在输出里，而不是默认"没发生"：

    * ``memory``：本脚本拿不到显存信号 ⇒ 恒不成立；
    * ``recover``：需要"当前值 ≠ 初值"这类**状态量**（信号序列里没有）⇒ 恒不成立。

    其余情形分成**四类互斥计数**：

    ====================== ==================================================
    ``n_shrink``           ``tpot_p90 > target·(1+h)`` ⇒ 收缩
    ``n_grow``             满足 ttft 条件 **且** ``headroom > h`` ⇒ 放大
    ``n_gate_rejected``    满足 ttft 条件 **但** ``headroom <= h`` ⇒ 停在带内
    ``n_not_eligible``     ttft 条件不满足 ⇒ **放大分支根本没被评估**
    ====================== ==================================================

    ⚠️ **为什么必须把 ``n_band`` 拆开**：把「ttft 条件不满足」与「门槛拒了」
    合成一个 ``n_band``，会让**放大分支从未被评估**与**门限把点全拦在带内**
    在表里长得一模一样——而这两件事的结论相反：前者是"这个负载形状下放大
    根本轮不到"（P1S3 的 ``target_ttft_ms=2000`` 就是这一类），后者是
    "判据形式把点挡在门外"。``n_band`` 保留为两者之和（向后兼容），
    **新代码请用拆分后的字段**。

    ⚠️ **``headroom`` 只在合格点（eligible）上统计。** 收缩点的 ``headroom``
    是很大的负数（``tpot_p90=571.7``、target=120 ⇒ **−3.76**），把它们算进来会让
    ``min < h < max`` **恒成立** ⇒ "门槛落在区间内部"**恒报**，与合格点是否一致无关。
    旧版正是这样：它对 120/0.08 的 c4（合格点 90 个**全部过门槛**、带内 0）
    照样报"抛硬币"，即**与自己的带内列自相矛盾**。
    ⇒ 判"分不分得开"要问的是：**在合格点里，门槛的判定一致吗**。

    ``verdict`` 就是对这个问题的回答（四态，见 ``_VERDICTS``）：
    决断（必放大 / 必不放大）才算"分得开"；**合格点里两种结果并存 ⇒
    「样本跨越门限」**——注意它只是"这条序列上门限不决断"，
    **不代表跨重复不可复现**（那要另比，见 ``cross_repeat_verdict``）。
    """
    tpot_limit = target_tpot_ms * (1.0 + hysteresis)   # 收缩：tpot_p90 > 此值
    ttft_limit = target_ttft_ms * (1.0 + hysteresis)   # TTFT 超此值才考虑放大
    grow_tpot_gate = target_tpot_ms * (1.0 - hysteresis)   # 放大要求 tpot_p90 < 此值
    headrooms: list[float] = []                 # **仅合格点**
    n_shrink = n_grow = n_gate_rejected = n_not_eligible = 0
    gap_grow: float | None = None               # 合格点里离放大门限最近的距离（ms）
    gap_shrink: float | None = None             # 全部点里离收缩门限最近的距离（ms）
    for ttft, tpot in signals:
        d = abs(tpot - tpot_limit)
        gap_shrink = d if gap_shrink is None else min(gap_shrink, d)
        if tpot > tpot_limit:
            n_shrink += 1
            continue
        if ttft <= ttft_limit:
            # ttft 条件不满足：**放大分支根本没被评估**，与"被门槛拒"不是一回事。
            n_not_eligible += 1
            continue
        headroom = (target_tpot_ms - tpot) / target_tpot_ms
        headrooms.append(headroom)
        d = abs(tpot - grow_tpot_gate)
        gap_grow = d if gap_grow is None else min(gap_grow, d)
        if headroom > hysteresis:
            n_grow += 1
        else:
            # 走到放大分支，但没过 headroom 门槛 ⇒ 停在带内（既不放也不缩）。
            n_gate_rejected += 1
    n_eligible = n_grow + n_gate_rejected
    h = _stat(headrooms) if headrooms else {"n": 0.0, "min": 0.0, "p50": 0.0, "max": 0.0}
    if n_eligible == 0:
        verdict = "不可达"          # ttft 条件从未满足 ⇒ 放大分支没被评估过
    elif n_gate_rejected == 0:
        verdict = "必放大"          # 合格点全部过门槛 ⇒ 判据决断
    elif n_grow == 0:
        verdict = "必不放大"        # 合格点全部被拒 ⇒ 判据决断
    else:
        # ⚠️ 名字是**样本跨越门限**，不是"抛硬币"。
        # 它描述的是"这条样本序列上门限不决断"（一部分点过、一部分点不过），
        # **不是**一条可复现性结论：同一次运行内负载本来就可能在工作点之间移动，
        # 两种结果并存完全正常。可复现性要拿**同一条件下不同重复**去比，
        # 见 ``cross_repeat_verdict``。旧名把"样本内不决断"直接说成
        # "跨重复不可复现"，是一次**未被证实的跳步**。
        verdict = "样本跨越门限"
    return {
        "n": len(signals),
        "n_shrink": n_shrink,
        "n_grow": n_grow,
        "n_gate_rejected": n_gate_rejected,
        "n_not_eligible": n_not_eligible,
        "n_eligible": n_eligible,
        # 兼容旧读数；**已混淆两种情形，不得用于判据**（见 docstring）。
        "n_band": n_gate_rejected + n_not_eligible,
        "tpot_limit": tpot_limit,
        "ttft_limit": ttft_limit,
        "grow_tpot_gate": grow_tpot_gate,
        "headroom_min": h["min"],
        "headroom_p50": h["p50"],
        "headroom_max": h["max"],
        "gap_grow_ms": gap_grow,
        "gap_shrink_ms": gap_shrink,
        "verdict": verdict,
    }


#: ``classify`` 的 ``verdict`` 取值及其含义。**只有前两个算"分得开"**。
#: 它说的是"**这条样本序列上**门限分不分得开"，**不是**可复现性结论。
_VERDICTS: dict[str, str] = {
    "必放大": "合格点全部过门槛 ⇒ 判据决断（放大分支确定触发）",
    "必不放大": "合格点全部被门槛拒 ⇒ 判据决断（放大门限在分布之外）",
    "样本跨越门限": "合格点里两种结果并存 ⇒ 门限落在**样本内部**；"
                    "**这不等于跨重复不可复现**（另见「重复间」一列）",
    "不可达": "ttft 条件从未满足 ⇒ 放大分支**根本没被评估**（不是'门限拦住了'）",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="控制器门限与实测信号分布的对齐检查")
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--prefix", default="P1S3B,P2", help="批次前缀，逗号分隔")
    ap.add_argument(
        "--targets", default="110,130,150",
        help="候选 target_tpot_ms，逗号分隔",
    )
    ap.add_argument("--hysteresis", default="0.1,0.05", help="候选 hysteresis，逗号分隔")
    ap.add_argument(
        "--target-ttft-ms", type=float, default=500.0,
        help="target_ttft_ms（放大分支还要求 ttft_p90 超它；默认 500）",
    )
    args = ap.parse_args(argv)

    raw = Path(args.raw)
    if not raw.is_dir():
        print(f"标定失败：产物根目录不存在 {raw}", file=sys.stderr)
        return 2
    prefixes = [p.strip() for p in args.prefix.split(",") if p.strip()]
    runs = _iter_adaptive_runs(raw, prefixes)
    if not runs:
        print(
            f"标定失败：{raw} 下找不到批次 {prefixes} 的 adaptive 臂 run。"
            f"门限只对 adaptive 臂起作用，没有它就无从标定。",
            file=sys.stderr,
        )
        return 3

    per_cell: dict[str, dict[str, Any]] = {}
    empty: list[str] = []
    for batch, cell, repeat_index, meta in runs:
        sigs = load_signal_sequence(meta)
        facts = load_trace_facts(meta)
        if not sigs:
            # 空输入必须显形：没有决策点与"所有决策点都在带内"长得一样。
            print(f"  ✗ {meta.parent.name}: 没有可用的 controller_trace 信号 —— 本格未检验",
                  file=sys.stderr)
            empty.append(meta.parent.name)
            continue
        # ⚠️ 格的身份 = **(批次, 格名)**，不是单独的格名。
        # 只按格名分组，P1S3B 的首轮与 P2 的首轮会被并成"同一条件的两次重复"
        # —— 而它们是**不同条件**（P2 含 `readback_ms` 计时修复）。实测症状：
        # 跨重复表里 c4 出现**两行 r0**，静态分类还不同（24/6 vs 29/1），
        # 而 `chunk 范围` 与瓶颈串**逐字相同** ⇒ 看起来像"同一条件下重复两次
        # 却给出不同判定"，实际是**两个条件被混成一组**。这正是项目纪律要禁的
        # 未登记混淆（单变量）。
        cell_rec = per_cell.setdefault((batch, cell), {"sigs": [], "per_repeat": []})
        cell_rec["sigs"].extend(sigs)
        # **逐重复**保留：标定既要用全部样本（分辨率），也要比较各重复的判定
        # （可复现性）。合并成一条序列就再也分不出"同一次运行内两种结果并存"
        # 与"换个重复换结论"——而这两件事的结论相反。
        cell_rec["per_repeat"].append(
            {
                "batch": batch,
                "repeat_index": repeat_index,
                "run": facts["run"],
                "sigs": sigs,
                "steps": facts["steps"],
                "bottlenecks": facts["bottlenecks"],
                "chunk_range": facts["chunk_range"],
                "chunk_changes": facts["chunk_changes"],
                "chunk_final": facts["chunk_final"],
                "n_trace_entries": facts["n_trace_entries"],
                "own_thresholds": facts["own_thresholds"],
            }
        )

    if not per_cell:
        print("标定失败：所有 adaptive run 都没有可用的控制器信号。", file=sys.stderr)
        return 4

    n_runs = sum(len(v["per_repeat"]) for v in per_cell.values())
    print("=" * 104)
    print(f"控制器**实际读到**的信号（决策点级；批次 {','.join(prefixes)}；"
          f"{len(per_cell)} 格 / {n_runs} 个 run，**含全部重复**）")
    print("=" * 104)
    print(f"{'批次':<8}{'格':<26}{'点数':>5}  {'tpot_p90 min/p50/max':>26}  {'ttft_p90 min/p50/max':>26}")
    for key in sorted(per_cell):
        batch, cell = key
        sigs = per_cell[key]["sigs"]
        t = _stat([s[1] for s in sigs])
        f = _stat([s[0] for s in sigs])
        print(f"{batch:<8}{cell:<26}{len(sigs):>5}  {_lo_hi(t):>26}  {_lo_hi(f):>26}")

    print()
    print("=" * 104)
    print("候选 (target_tpot_ms, hysteresis) 下各分支的可触发比例 —— **静态口径，不含冷却与反馈**")
    print("=" * 104)
    cells = sorted(per_cell)
    for target in [float(x) for x in args.targets.split(",") if x.strip()]:
        for hys in [float(x) for x in args.hysteresis.split(",") if x.strip()]:
            r0 = classify(per_cell[cells[0]]["sigs"], target_tpot_ms=target,
                          target_ttft_ms=args.target_ttft_ms, hysteresis=hys)
            print()
            print(
                f"target_tpot_ms={target:g}  hysteresis={hys:g}  ⇒  "
                f"收缩门限={r0['tpot_limit']:.1f}（tpot_p90 超它才收缩）  "
                f"放大要求 tpot_p90 < {r0['grow_tpot_gate']:.1f} 且 "
                f"ttft_p90 > {r0['ttft_limit']:.1f}"
            )
            print(
                f"  {'批次':<8}{'格':<26}{'轮数':>5}{'收缩':>7}{'放大':>7}{'门槛拒':>8}"
                f"{'未评估':>8}{'headroom(合格) min/p50/max':>30}{'判定':>16}"
            )
            got: dict[tuple[str, str], dict[str, Any]] = {}
            for key in cells:
                batch, cell = key
                r = classify(per_cell[key]["sigs"], target_tpot_ms=target,
                             target_ttft_ms=args.target_ttft_ms, hysteresis=hys)
                got[key] = r
                # 无合格点时写 n/a，**不写 0.0000**：0 是"测出来是 0"（与缺失相反）。
                hr = (f"{r['headroom_min']:.4f}/{r['headroom_p50']:.4f}/{r['headroom_max']:.4f}"
                      if r["n_eligible"] else "n/a")
                # 分数先算好再拼：**不要写嵌套 f-string**（同引号的嵌套只在本机 3.13 合法，
                # 远端的 3.10 会 SyntaxError —— 同一份工具要在两边各跑一次做交叉验证）。
                frac_shrink = f"{r['n_shrink']}/{r['n']}"
                frac_grow = f"{r['n_grow']}/{r['n']}"
                frac_rej = f"{r['n_gate_rejected']}/{r['n']}"
                frac_ne = f"{r['n_not_eligible']}/{r['n']}"
                n_rep = len(per_cell[key]["per_repeat"])
                print(
                    f"  {batch:<8}{cell:<26}{n_rep:>5}{frac_shrink:>7}{frac_grow:>7}"
                    f"{frac_rej:>8}{frac_ne:>8}{hr:>30}{r['verdict']:>16}"
                )
            n_span = sum(1 for k in cells if got[k]["verdict"] == "样本跨越门限")
            n_decisive = sum(1 for k in cells if got[k]["verdict"] in ("必放大", "必不放大"))
            n_unreach = sum(1 for k in cells if got[k]["verdict"] == "不可达")
            print(f"  ⇒ 决断 {n_decisive} 格 / **样本跨越门限 {n_span} 格** / "
                  f"不可达 {n_unreach} 格")
            print("     （「样本跨越门限」是**样本内**的说法；可复现性见下一节"
                  "「跨重复对照」，两者不是一回事）")

            # 离门限最近的点：抛硬币的成因是"点紧贴门限"，所以要把这个距离报出来
            # （**只报事实，不设阈值** —— 阈值是守卫的职责，别把它搬进权威工具）。
            print()
            print("  离门限最近的点（ms；n/a = 该侧没有点）")
            print(f"    {'批次':<8}{'格':<26}{'放大门限':>10}{'收缩门限':>10}")
            for key in cells:
                batch, cell = key
                r = got[key]
                g_grow = f"{r['gap_grow_ms']:.1f}" if r["gap_grow_ms"] is not None else "n/a"
                g_shr = f"{r['gap_shrink_ms']:.1f}" if r["gap_shrink_ms"] is not None else "n/a"
                print(f"    {batch:<8}{cell:<26}{g_grow:>10}{g_shr:>10}")
            for key in cells:
                batch, cell = key
                r = got[key]
                if r["verdict"] == "样本跨越门限":
                    print(f"    ⚠ 样本跨越门限：{batch}/{cell} —— 合格点 {r['n_eligible']} 个里 "
                          f"{r['n_grow']} 个过门槛、{r['n_gate_rejected']} 个被拒；"
                          f"离放大门限最近的点相距 {r['gap_grow_ms']:.1f} ms")
                elif r["verdict"] == "不可达":
                    print(f"    · 不可达：{batch}/{cell} —— ttft_p90 全部 ≤ {r['ttft_limit']:.1f}，"
                          f"放大分支**从未被评估**（不是'被门限拦住'）")

            # ---- 跨重复对照：相同条件下不同重复的**决策分歧** ----
            #
            # 这一节才是"可复现性"的答案。上一节的 `verdict` 只看合并后的样本，
            # 而"一条序列里两种结果并存"完全可能是**同一次运行内**负载本来就在
            # 两个工作点之间移动——与"换个重复就换结论"是两件事。
            # 这里逐重复各判一次，再看判定集合。
            print()
            print("  跨重复对照（**同一条件、不同重复**；这才是可复现性的问法）")
            print(f"    {'批次':<8}{'格':<26}{'重复':>4} {'点数':>4}{'收缩':>5}{'放大':>5}"
                  f"{'门槛拒':>6}{'未评估':>7}  {'判定':<16}{'chunk 范围(变次)':>17}  {'实际走的瓶颈分支':<26}")
            for key in cells:
                batch, cell = key
                per_rep_verdicts: list[str] = []
                for rep in sorted(per_cell[key]["per_repeat"],
                                  key=lambda x: x["repeat_index"]):
                    rr = classify(rep["sigs"], target_tpot_ms=target,
                                  target_ttft_ms=args.target_ttft_ms, hysteresis=hys)
                    per_rep_verdicts.append(rr["verdict"])
                    cr = rep["chunk_range"]
                    # `(k)` = chunk_size **变了几次**。只给 min..max 不够：
                    # 两轮可以有相同的 range 而在不同**时刻**走到不同的值
                    # （同一 range 下 TTFT 481 与 652 并存的成因）。
                    chunk_txt = f"{cr[0]}..{cr[1]} ({rep['chunk_changes']})" if cr else "n/a"
                    # 实际走的瓶颈分支：计数压成 `tpot×4 ttft×2` 形状。
                    # ⚠️ 它与静态分类**不是同一个量**，而且**不代表真的改了值**：
                    # 这里说的是"控制器那一轮**选中了哪个分支**"（含冷却、含被
                    # headroom 门槛拒——被拒时返回的 bottleneck 仍是 `ttft`）。
                    seen: dict[str, int] = {}
                    for b in rep["bottlenecks"]:
                        seen[b] = seen.get(b, 0) + 1
                    bneck = " ".join(f"{k}×{v}" for k, v in sorted(seen.items())) or "n/a"
                    label = f"r{rep['repeat_index']}"
                    print(
                        f"    {batch:<8}{cell:<26}{label:>4} {len(rep['sigs']):>4}"
                        f"{rr['n_shrink']:>5}{rr['n_grow']:>5}{rr['n_gate_rejected']:>6}"
                        f"{rr['n_not_eligible']:>7}  {rr['verdict']:<16}"
                        f"{chunk_txt:>17}  {bneck:<26}"
                    )
                agree, why = cross_repeat_verdict(per_rep_verdicts)
                mark = "✓" if agree == "重复间一致" else "⚠"
                print(f"      {mark} {batch}/{cell}: {agree} —— {why}")

    # 口径核对**只跑一次**：它与 `--targets/--hysteresis` 无关（用的是每个 run
    # 自己的阈值），放进候选循环里会把同一张表打印 N 遍。它是本工具的**自检**，
    # 不是标定结论的一部分，所以放在所有候选之后。
    drift = report_branch_consistency(cells, per_cell)

    print()
    print("读法：")
    print("  判定列（四态）：**必放大 / 必不放大** = 判据在该格上**决断**（分得开）；")
    print("                  **样本跨越门限** = **这条（合并后的）样本序列上**合格点一部分")
    print("                  过门槛、一部分被拒 ⇒ 门限落在样本内部；")
    print("                  **不可达** = ttft 条件从未满足 ⇒ 放大分支**没被评估过**。")
    print("  ⚠️⚠️ **「样本跨越门限」不是可复现性结论。** 它只看一条合并序列；")
    print("     同一次运行内负载本来就可能在工作点之间移动，于是两种结果并存——")
    print("     这与「换个重复就换结论」是两件事。可复现性要拿**同一条件下不同重复**")
    print("     去比，见「跨重复对照」一节（`cross_repeat_verdict`）。")
    print("     旧名「抛硬币」把前者直接说成「跨重复不可复现」，是一次**未被证实的跳步**；")
    print("     本工具也**不能**充当可复现性的阻断门——它能给的是「哪一格要盯着看」。")
    print("  ⚠️ 「未评估」与「被门槛拒」必须分开：前者是**负载形状**的问题（放大轮不到），")
    print("     后者是**判据形式**的问题（门限把点挡在门外）。旧版把两者合成「带内」一列，")
    print("     于是 P1S3 的 `target_ttft_ms=2000`（判据一次不触发）与 P1S3B 的")
    print("     「门限贴工作点」在表里长得一样 —— 结论却完全相反。")
    print("  ⚠️ headroom 只统计**合格点**。旧版把收缩点也算进来，而收缩点的 headroom")
    print("     是很大的负数 ⇒ `min < h < max` 恒成立 ⇒ 「跨过门槛」**恒报**：")
    print("     它对 120/0.08 的 c4（合格点全部过门槛、带内 0）照样报跨越门限，")
    print("     与自己的「门槛拒」列**自相矛盾**。")
    print("  ⚠️ 现在是**全部重复**合并后统计（旧版只取首轮）。"
          "重复数在表里单列出来：样本量不同，同一个门限的结论可信度也不同。")
    print("  ⚠️ 格的身份是 **(批次, 格名)**。「重复」只在**同一批次内**编号，"
          "所以跨批次同名的 r0 **不是**同一条件（P2 含 `readback_ms` 计时修复）。")
    print("  ⚠️ 「实际走的瓶颈分支」记的是控制器**选中了哪个分支**，"
          "**不等于真的改了值**：被 headroom 门槛拒时返回的 bottleneck 仍是 `ttft`，"
          "冷却中的返回也照记原分支。")
    print("     实测例：P1S3D 的 c8 一轮 `tpot×17 ttft×1`，而静态分类是 `放大=0、门槛拒=1`"
          " ⇒ 那一次 `ttft` **一次都没放大**。")
    print("     「到底动没动」要看 `chunk 范围(k)`（k = chunk_size 变了几次），"
          "而它仍**不给时刻**：两轮可以 range 相同却在不同时刻走到不同的值。")
    print("  ⚠️ 「口径核对」一节是本工具的**自检**，用每个 run **自己**的阈值")
    print("     （`meta.config.controller`）核对静态分类与轨迹分支是否逐条对上。"
          "漂移 ⇒ 退出码 6，表里的「门槛拒/放大/未评估」不可信。"
          "读不到阈值 ⇒ 记「不可判定」，**不是**通过。")
    print()
    print("⚠️ 严格程度：本表只判「信号落不落在门限外」，不含冷却，也不含"
          "「动作改变后续信号」这一反馈。")
    print("   `memory` 与 `recover` 两个分支本脚本**判不了**（前者无显存信号，后者需要")
    print("   状态量）⇒ 一律记为不成立，**不要**读成「这两个分支没发生」。")
    print("   它是**上界**；真正的把关仍是 scripts/check_step3b_reachability.py 的开环复放")
    print("   与真机重跑。")
    if empty:
        print()
        print(f"标定不完整：{len(empty)} 个 run 没有控制器信号（{', '.join(empty)}）"
              f"——上表缺这些格，**不是**它们全在带内。", file=sys.stderr)
        return 5
    if drift:
        # 口径漂移是**工具自身不可信**，不是"某格结论不好"：必须非零退出，
        # 否则它会被当成"跑完了"混进下游。（"内容对、退出码错"这类失效不会有
        # 任何报错，所以这里显式返回。）
        print()
        print(f"标定不可信：{len(drift)} 个 run 的静态分类与轨迹分支计数对不上"
              f"（见上表「口径核对」）——先修工具再读标定结论。", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
