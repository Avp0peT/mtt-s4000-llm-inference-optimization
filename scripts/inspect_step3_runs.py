#!/usr/bin/env python3
"""检查第③步产物的**重复独立性**与离散度——先怀疑，再相信。

为什么必须单独查这件事
----------------------
方案 §1.5 的强制披露义务第 2 条写得很直白：

    离散度为 0 或极小是**可疑信号**，不是好消息。本项目已实测跨进程尾延迟差 35%、
    单配置内孤停顿 4.35×。若某格 3 次几乎不差，先查是否采到了同一进程的缓存/热状态，
    而不是先庆祝稳定性。

所以看到"稳健 CV = 0.0002"时的正确动作是**去查**，而不是当作稳定性证据写进结论。
本脚本把要查的几件事变成一次可复现的检查：

1. **三次重复是不是独立的 run**：各自的时间戳间隔、是否各自重新装载了模型。
   如果三次挤在同一时刻、或只有第一次真正装载，那它们不是独立样本。
2. **离散度**：逐格给出 3 个原始值（不是只给中位数），并算稳健 CV。
3. **有没有孤停顿**：`stall_p99 / stall_p50` 的比值。这是 4.35× 那次波动的特征量；
   本脚本把它显式算出来，避免"看起来挺稳"掩盖掉单次巨停顿。
4. **控制器动作有没有真的落到 prefill 上**（``--prefill-effect``）。见下。
5. **这次 run 用的到底是不是那份配置**（``--verify-config DIR``）。见下。

为什么"控制器改了什么"还不够
----------------------------
``--controller`` 回答的是"控制器改过哪些参数"。但它只看 ``controller_trace`` 里的
``state``，因此**天然看不见**下面这件事：

    控制器在纯 decode 阶段改了一次 ``chunk_size``，此后直到 run 结束都没有再发生
    prefill —— 这次改动**对 prefill 的贡献是零**，可它在 ``--controller`` 的
    ``chunk_requested(min/max)`` 里照样表现为"改过 512/2048"。

反过来的假象也存在：某一档全程零动作，``adaptive`` 与 ``static_chunk`` 数值上完全
相等，而这**不是**"自适应没有收益"——是"这个负载下控制器无事可做"。两种"没差别"
的成因完全不同，所以这两件事**必须是两张表**（把"改了参数"与"改到了 prefill"
并进同一个格子，就再也分不出来了）；可以一次输出来看，但不得合并成一格。

为什么"代码是这个提交"还不够
----------------------------
部署戳（``SOURCE_PROVENANCE.json``）的 ``digest_scope`` 是
``src/ascend_opt`` / ``benchmark`` / ``scripts``，**不含 ``configs/``**。
于是"代码是新的"与"配置是新的"是两件独立的事，前者成立**推不出**后者。
本仓库踩过一次：远端留着旧模板（``slo.tpot_ms`` 还是 50）且没有新配置目录，
批次照样跑得动、照样盖 ``MEASURED`` —— 从产物表面看不出任何异常。

``--verify-config`` 逐 run 把 ``config.snapshot.yaml`` 与目录下的同名配置
**逐字段**比一遍，**有差异就返回 1**（能失败的检查才算检查）。

用法::

    python scripts/inspect_step3_runs.py results/raw --tag 20260923
    python scripts/inspect_step3_runs.py results/raw --dump   # 打印 meta 结构
    python scripts/inspect_step3_runs.py results/raw --controller     # 改了什么
    python scripts/inspect_step3_runs.py results/raw --cycles          # 补入批次 × 决策步号
    python scripts/inspect_step3_runs.py results/raw --prefill-effect # 有没有落到 prefill
    python scripts/inspect_step3_runs.py results/raw \
        --verify-config configs/mthreads/phase1_step3d    # 输入是不是这一版（失败=1）

多个视图标志可以同时给，会按 ``trace → controller → cycles → prefill-effect →
verify-config → detail → dump`` 顺序**逐个**输出（早先是静默只出第一个）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# 顶层 import，**不**放进函数里：本仓库其它地方（preflight/run_benchmark/sweep）
# 惯例是"用到时才 import yaml"，但那样一旦某个分支漏掉这句，调用方会以 NameError
# **整块**失败——已经踩过一次。PyYAML 是 pyproject 里的核心依赖，顶层引入无额外代价。
import yaml


def _load(run_dir: Path) -> dict | None:
    p = run_dir / "meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _robust_cv(values: list[float]) -> float | None:
    """稳健 CV = 1.4826·MAD/中位数（与项目口径一致，不用普通 CV）。

    普通 CV 会被单次巨停顿拉爆，而我们要判的恰恰是"有没有巨停顿"——
    用会被它污染的量去判它，等于把信号抹掉。MAD 对离群点不敏感。
    """
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    med = statistics.median(vals)
    if med == 0:
        return None
    mad = statistics.median([abs(v - med) for v in vals])
    return 1.4826 * mad / med


def _plain_cv(values: list[float]) -> float | None:
    """普通 CV = 标准差/均值。**只作对照量，不作判据。**

    它和稳健 CV 一起看才有意义：稳健 CV 小、普通 CV 大，说明"稳"是假象——
    MAD 取的是"偏离中位数"的**中位数**，而 n=3 时**一个**离群点并不改变
    这个中位数，于是离群被抹平。只报稳健 CV 会漏掉这一类，而它恰恰是
    §1.5 第 2 条要我们警惕的"看着极稳"。
    """
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    if mean == 0:
        return None
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return (var ** 0.5) / mean


# 稳健 CV 低于此值 → 按 §1.5 第 2 条列为"可疑"，先排除共享热状态。
TINY_CV = 0.005
# 普通 CV 超过稳健 CV 这么多倍 → MAD 掩盖了单点离群。
MASK_RATIO = 10.0


def _pct(block: dict | None, key: str = "p50") -> float | None:
    if not isinstance(block, dict):
        return None
    v = block.get(key)
    return float(v) if isinstance(v, (int, float)) else None


# --------------------------------------------------------------------------- #
# 控制器动作 → 后续 prefill 的关联
# --------------------------------------------------------------------------- #
#
# **为什么不能只做布尔**："动作生效 / 没生效"这个二分法会把三类完全不同的事实
# 压成同一个值，而它们对结论的影响相反：
#
#   · 改了值，但此后**一次 prefill 都没有** → 不是"没生效"，是"没有可施加的对象"。
#     下一轮把负载从"一次性 8 个请求"改成"持续补入"之后，这一类大概率会消失，
#     因此它是一个**负载形状的读数**，不是控制器缺陷。
#   · 改了值，但在**下一个 prefill 之前**又被改了一次 → 那个窗口里根本没有 prefill，
#     同样谈不上生效不生效。它最初被我并进下面那一类，结论立刻变成"c2/c4 有 5 次
#     改了却没生效"——**假的**。写这个脚本的价值就在于它把这个假结论暴露出来了。
#   · 值一直保持到下一个 prefill，而那次 prefill 用的却是别的值 → 这才是真正的
#     "改了却没落到 prefill 上"，属于机制问题，必须查。**实测这一态一次都没出现。**
#
# 再加上"改了但取值没变"（被上下界夹住、或重申原值）——它既不是"改了"也不是
# "没改"，计进任何一边都会把结论带偏。故一共五态。
#
# 真机依据（P1S3B，2026-09-23）：
#   c1：一次动作都没有（0 条 changed）
#   c2/c4：动作落到了 prefill 上，但**只有扩容那一半**（1024→2048，后续 prefill 用 2048）；
#          收缩那一次（1024→512）在下个 prefill 前已被改回，属"被覆盖"态
#   c8：step 8 的收缩落到了 prefill 上（此后 14 个 prefill 事件用 512）；
#       其后 5 次 changed=True 全部是 max_batched_tokens 在动、chunk_size 已触下界不动
#
# 所以这五态不是为了完备性写上去的，是**每一种都能在产物里指到具体某一次决策**。

# telemetry 的 note 是**给人看的字符串**，不是结构化字段。键名形如
# `prefill_tok=` / `fwd_ms=`，只取数值。
_NOTE_KV = re.compile(r"([a-z_]+)=(-?\d+(?:\.\d+)?)")

# 五态取值。
ACTION_VALUE_UNCHANGED = "value_unchanged"
ACTION_APPLIED = "applied_to_prefill"
ACTION_SUPERSEDED = "superseded_before_prefill"
ACTION_NO_PREFILL_AFTER = "no_prefill_after"
ACTION_NOT_APPLIED = "not_applied_to_prefill"

#: 每一态的读法。写成映射而不是散在 if 里，免得将来只改一处。
ACTION_KIND_NOTE = {
    ACTION_VALUE_UNCHANGED:
        "决策标记为已改，但**取值没变**——被上下界夹住或重申原值，实际动的是别的参数",
    ACTION_APPLIED:
        "改了值，且其后第一次 prefill 用的就是新值 —— 动作确实落到了 prefill 上",
    ACTION_SUPERSEDED:
        "改了值，但在**下一个 prefill 发生之前**又被下一次决策改掉了 —— "
        "窗口里没有 prefill，因此**谈不上生效不生效**，不是机制缺陷",
    ACTION_NO_PREFILL_AFTER:
        "改了值，但此后**再无 prefill** —— 不是没生效，是这一负载下没有可施加的对象",
    ACTION_NOT_APPLIED:
        "值一直保持到下一次 prefill，而那次 prefill 用的**不是**它 —— "
        "真的「改了却没生效」，需要查机制",
}


def parse_note(note: str | None) -> dict[str, float]:
    """解析 telemetry 的 ``note`` 文本 → ``{键: 数值}``；解析不出任何键则返回 ``{}``。

    必须**按键名**取值，不能"取第一个数字"：真机的一条 note 是
    ``prefill_tok=1024 decode=0 forwards=1 fwd_ms=247.50 pack_ms=0.38``，
    按位置取会静默拿到别的量（"第一个数字"恰好是 1024，掩体太浅，看不出错）。
    """
    if not note:
        return {}
    return {m.group(1): float(m.group(2)) for m in _NOTE_KV.finditer(note)}


def prefill_tokens(row: dict) -> float | None:
    """这一 step 做了多少 prefill token。**取不到返回 ``None``，不返回 0**。

    ``0`` 与 ``None`` 在这里含义相反：``0`` 是"这一 step 确实没做 prefill"
    （真实测量值），``None`` 是"note 里没有这个字段"（没测到）。把 ``None``
    当 0 会把"引擎根本没上报"读成"确实没 prefill"——正是要防的静默失效。

    与项目口径一致：不可测的值保持 ``None``；但 0 是真实测量值时保留 0。
    """
    return parse_note(row.get("note")).get("prefill_tok")


def load_telemetry(run_dir: Path) -> tuple[list[dict], int]:
    """读 ``telemetry.jsonl`` → ``(行列表, 跳过的坏行数)``。

    坏行**不静默丢弃**：返回计数由调用方显形。否则文件被截断时，"prefill 事件数"
    会变小甚至变 0，而在报表上"0 个 prefill 事件"与"这个负载没有 prefill"
    长得一模一样。
    """
    p = Path(run_dir) / "telemetry.jsonl"
    if not p.exists():
        return [], 0
    rows: list[dict] = []
    skipped = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            skipped += 1
    return rows, skipped


def prefill_events(rows: list[dict]) -> list[dict]:
    """抽出真正做过 prefill 的 step（``prefill_tok > 0``）。

    取不到 ``prefill_tok`` 的行**既不进列表、也不记作 0**——它们由调用方用
    :func:`count_unparsed` 单独计数。这样"引擎改了日志格式"表现为一个显式计数，
    而不是"prefill 事件数变成 0"。
    """
    out: list[dict] = []
    for r in rows:
        v = prefill_tokens(r)
        if v is None or v <= 0:
            continue
        out.append({
            "step": r.get("step"),
            "prefill_tok": v,
            "chunk_size": r.get("chunk_size"),
            "chunk_size_requested": r.get("chunk_size_requested"),
            "prefill_budget_effective": r.get("prefill_budget_effective"),
            "queue_depth": r.get("queue_depth"),
            "running": r.get("running"),
        })
    return out


def count_unparsed(rows: list[dict]) -> int:
    """有多少行的 note 里**没有** ``prefill_tok`` 字段。>0 即解析口径需要复核。"""
    return sum(1 for r in rows if prefill_tokens(r) is None)


def expected_prefill_tokens(run_dir: Path) -> float | None:
    """从 ``requests.jsonl`` 求和 ``prompt_len`` —— prefill 工作量的**地面真值**。

    有它才能把"控制器把 prefill 重新分配了"与"控制器弄丢/重复了 prefill 工作"
    分开。取不到（文件缺失或字段缺失）返回 ``None``，**不返回 0**：0 是个具体的
    断言（"这个 run 没有任何 prefill 工作"），与"读不到"是两回事。
    """
    p = Path(run_dir) / "requests.jsonl"
    if not p.exists():
        return None
    total = 0.0
    seen = False
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        v = r.get("prompt_len")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total += float(v)
            seen = True
    return total if seen else None


def _is_step(v: object) -> bool:
    """step 必须是**真数值**才能比大小。

    真机上它是 int；但如果将来某个引擎把它写成字符串，``"9" > 9`` 会抛
    ``TypeError`` 而不是给个错答案——把它挡在比较之前，避免整段审计因为
    一个字段类型变化而崩掉，也避免"看起来比出来是假的"。
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool)


#: 补入批次切分的时间下限（ms）。低于它的间隔一律算"同一批"，
#: 否则"全部同刻到达"会被切成一堆单请求批。
_BATCH_MIN_GAP_MS = 500.0


def load_requests(run_dir: Path) -> list[dict]:
    """读 ``requests.jsonl`` → 行列表（坏行跳过，**不抛**）。

    ``arrival_ms`` 是**执行器实际分配**的到达时刻。闭模型（``arrival: closed``）
    负载里 ``meta.json`` 的负载指纹记的是**全 0 的计划到达**，
    所以"到底是不是同时到达"**只能**从这里看。
    """
    p = Path(run_dir) / "requests.jsonl"
    if not p.is_file():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def arrival_batches(arrivals: list[float]) -> list[float]:
    """把到达时刻切成"补入批次"，返回每批的**首个**到达时刻。

    阈值 = ``max(500 ms, 0.5 × 最大间隔)``。批内间隔远小于批间间隔时，
    取最大间隔的一半即可分开；下限防止同刻到达被切碎。
    调用方应把阈值与原始间隔一并打印，**读者可以推翻这个切法**。
    """
    a = sorted(float(x) for x in arrivals if isinstance(x, (int, float)) and not isinstance(x, bool))
    if not a:
        return []
    gaps = [b - x for x, b in zip(a, a[1:])]
    if not gaps:
        return [a[0]]
    cut = max(_BATCH_MIN_GAP_MS, 0.5 * max(gaps))
    starts = [a[0]]
    for x, g in zip(a[1:], gaps):
        if g > cut:
            starts.append(x)
    return starts


def step_at_ms(rows: list[dict], t_ms: float) -> int | None:
    """把时刻映射到**最接近**的 step（同一 run 内 ``t_ms`` 单调不减）。

    取不到可用行时返回 ``None``，**不返回 0**：step 0 是"这一步"，
    与"映射不出来"不是一回事。
    """
    best: int | None = None
    best_d: float | None = None
    for r in rows:
        t = r.get("t_ms")
        s = r.get("step")
        if not isinstance(t, (int, float)) or not _is_step(s):
            continue
        d = abs(float(t) - t_ms)
        if best_d is None or d < best_d:
            best, best_d = int(s), d
    return best


def cycle_report(name: str, meta: dict, run_dir: Path) -> dict:
    """一个 run 的"总步数 × 补入批次 × 决策步号"读数。

    为什么单开一个视图：**决策步号是 run 内的局部量**，跨格引用极易接错——
    本仓库踩过：把 c2 的 ``264/288/392/416`` 当成了 c4 的，而 **c4 首轮只有 263 步**。
    把"总步数 / 批次数 / 批起点 / 决策步号"放在同一行，接错时**对不上**。
    """
    trace = meta.get("controller_trace") or []
    rows, _bad = load_telemetry(run_dir)
    reqs = load_requests(run_dir)
    arrivals = [r.get("arrival_ms") for r in reqs]
    starts = arrival_batches([x for x in arrivals if isinstance(x, (int, float))])

    ctrl = (meta.get("config") or {}).get("controller") or {}
    decisions = []
    for e in trace:
        if not e.get("changed"):
            continue
        bn = e.get("bottleneck")
        # 方向按 `_decide` 的分支名读：tpot ⇒ 收缩、ttft ⇒ 放大。
        # 其它分支（memory/recover/none）**照原样印出来**，不硬塞进两类。
        label = {"tpot": "收缩", "ttft": "放大"}.get(bn, str(bn))
        decisions.append({"step": e.get("step"), "dir": label})

    return {
        "name": name,
        "n_requests": len(reqs) if reqs else None,
        "total_steps": len(rows) if rows else None,
        "batches": [step_at_ms(rows, s) for s in starts],
        "batch_starts_ms": starts,
        "decisions": decisions,
        "controller_enabled": bool(ctrl.get("enabled")),
        "concurrency": (meta.get("config") or {}).get("workload_overrides", {}).get("concurrency"),
    }


def action_effects(trace: list[dict], prefills: list[dict]) -> list[dict]:
    """把每一次 ``changed`` 决策与**其后第一次 prefill** 关联起来。

    返回项含 ``kind``，取值为 ``ACTION_*`` 五态之一。分类顺序有意从"最容易被
    误读"的那一类开始判：

    1. 取值没变 → ``value_unchanged``（先判它，否则它会冒充"生效"或"无对象"）
    2. 此后无 prefill → ``no_prefill_after``
    3. 下一个 prefill 之前又被改过一次 → ``superseded_before_prefill``
    4. 后续 prefill 用新值 → ``applied_to_prefill``
    5. 否则 → ``not_applied_to_prefill``（**只有这一态才是机制缺陷**）

    第 3 态是看到真机输出之后补上的。最初只有四态，于是把 c2/c4 里"step 8 改成
    512、但下一个 prefill 在 step 129（那时值已被 step 80 改回 2048）"判成了
    "改了却没生效"。**这个判断是错的**：512 在整个 step 8–31 区间里一直是生效值，
    只是那个区间里一次 prefill 都没发生。把它报成机制缺陷，就等于把"负载形状的
    读数"冒充成"控制器缺陷"——两者对下一步的指示完全相反。

    ``prev_chunk`` 只在 ``changed`` 决策处推进，因此"取值是否真的变了"比的是
    **上一次动作后的值**，而不是上一条 trace 的值——后者在冷却期会被重复写入，
    比出来永远是"没变"。
    """
    idx: list[int] = [i for i, e in enumerate(trace) if e.get("changed")]
    #: 每一条 trace 的"决策后取值"，用来回答"这次决策之前的值是多少"。
    #  取 i-1 而不是上一条 changed：冷却期里会重复写入同样的 state，
    #  跳过它们会让 before 停在更早的值上。
    chunks = [(e.get("state") or {}).get("chunk_size") for e in trace]
    out: list[dict] = []
    for pos, i in enumerate(idx):
        e = trace[i]
        st = e.get("state") or {}
        cur = st.get("chunk_size")
        if i > 0:
            prev_chunk = chunks[i - 1]
        else:
            # 第一条 trace 本身就是决策：此前没有可比的 state，退回**配置初值**。
            # 宁可拿初值也不要留 None —— None 会让"取值是否变了"恒判为"变了"，
            # 于是每一次动作都被算成"改了值"，把 value_unchanged 这一态清空。
            prev_chunk = st.get("init_chunk_size")
        step = e.get("step")
        nxt = None
        if _is_step(step):
            nxt = next((p for p in prefills if _is_step(p.get("step")) and p["step"] > step), None)
        # 本次决策之后的**下一次**决策发生在哪一步（无则 None）
        later_step = None
        if pos + 1 < len(idx):
            cand = trace[idx[pos + 1]].get("step")
            if _is_step(cand):
                later_step = cand
        if prev_chunk is not None and cur == prev_chunk:
            kind = ACTION_VALUE_UNCHANGED
        elif nxt is None:
            kind = ACTION_NO_PREFILL_AFTER
        elif (later_step is not None and _is_step(nxt.get("step"))
              and later_step < nxt["step"]):
            # 值在下一次 prefill 之前就被改掉了 —— 这个窗口里没有 prefill 可用
            kind = ACTION_SUPERSEDED
        elif nxt.get("chunk_size") == cur:
            kind = ACTION_APPLIED
        else:
            kind = ACTION_NOT_APPLIED
        out.append({
            "step": step,
            "chunk_before": prev_chunk,
            "chunk_after": cur,
            "reason": e.get("reason"),
            "next_prefill_step": nxt.get("step") if nxt else None,
            "next_prefill_chunk": nxt.get("chunk_size") if nxt else None,
            "next_prefill_tok": nxt.get("prefill_tok") if nxt else None,
            "next_decision_step": later_step,
            "kind": kind,
        })
    return out


def effect_tally(effects: list[dict]) -> dict[str, int]:
    """按五态计数。五态**全部**出现在返回值里（缺的记 0），免得上游只看到
    非零的那几项，误以为"没出现过的态不存在"。"""
    tally = {k: 0 for k in ACTION_KIND_NOTE}
    for e in effects:
        tally[e["kind"]] = tally.get(e["kind"], 0) + 1
    return tally


def concurrency_of(run_name: str) -> int | None:
    """从 run 目录名里取并发档（``...-adaptive_c8-r2`` → 8）。

    ``rsplit("_c", 1)`` 而不是 ``split``：臂名里本身就含 ``_c``
    （``static_chunk``），从左切会切在 ``static`` 与 ``chunk`` 之间。
    """
    tail = run_name.rsplit("_c", 1)
    if len(tail) != 2:
        return None
    m = re.match(r"(\d+)", tail[1])
    return int(m.group(1)) if m else None


def cell_of(run_name: str) -> str | None:
    """从 run 目录名里取"格名"（``P1S3D-phase1_anchor-adaptive_c4-r2`` → ``adaptive_c4``）。

    为什么不能复用 :func:`concurrency_of`：那个只拿到并发档（``4``），拿不到臂，
    而配置文件名是 **臂+并发**（``adaptive_c4.yaml``），靠并发档定位不到唯一一份。

    取**最后一段**（按 ``-`` 切）并先剥掉尾部 ``-rN``：run 名形如
    ``<tag>-<workload>-<格>-rN``，格名恒定在倒数第一（剥掉 ``-rN`` 后）。
    取不出时返回 ``None``，由调用方按"取不出"处理，**不要**猜一个默认值。
    """
    base = re.sub(r"-r\d+$", "", run_name)
    if "-" not in base:
        return None
    cell = base.rsplit("-", 1)[1]
    return cell or None


def config_mismatches(cfg: dict, snap: dict) -> tuple[list[str], int]:
    """比对"配置声明的字段"与"run 实际加载到的快照"，返回 (问题清单, 快照多出的键数)。

    判据方向是**单向**的：配置是**意图**，快照是**这次 run 真正加载到的**。
    所以只判配置里写了的键——

    * 配置里有、快照里没有 → 这条设置根本没进这次 run；
    * 两边都有但取值不同   → 这次 run 跑的不是这份配置。

    快照里多出来的键是**默认值展开**（快照记录的是加载后的完整对象），
    不算问题，只计数报出来，避免"快照字段多"被误读成不一致。

    展平复用 ``check_step3_configs._flatten``：同一件事（点分路径展平）
    在本仓库已经有一份实现，再写一份迟早会不一致。

    这个 import 刻意留在**函数内**（与上面 ``import yaml`` 的处理相反）：
    ``check_step3_configs`` 是**兄弟脚本**，只有 ``scripts/`` 在 ``sys.path`` 上时
    才导入得到——直接跑脚本或本仓库的测试都满足，但 ``python -m scripts.xxx``
    之类调用方式不满足。放函数内可以让那种调用方式照常工作，只在真正用本视图时
    才需要 sibling 可导入。
    """
    from check_step3_configs import _flatten

    fcfg: dict[str, object] = {}
    fsnap: dict[str, object] = {}
    _flatten("", cfg, fcfg)
    _flatten("", snap, fsnap)

    problems: list[str] = []
    for key in sorted(fcfg):
        if key not in fsnap:
            problems.append(f"{key}: 配置={fcfg[key]!r}，但快照里**没有这个键**")
        elif fsnap[key] != fcfg[key]:
            problems.append(f"{key}: 配置={fcfg[key]!r} ≠ 快照={fsnap[key]!r}")
    return problems, len(set(fsnap) - set(fcfg))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="检查第③步产物的重复独立性与离散度")
    ap.add_argument("raw_dir", help="results/raw 或含 run 目录的路径")
    ap.add_argument("--tag", default=None, help="只处理名字含该前缀/子串的 run")
    ap.add_argument("--dump", action="store_true", help="打印一个 meta 的顶层结构")
    ap.add_argument("--detail", default=None, help="打印指定 run 的稳定性/装载/后端统计")
    ap.add_argument(
        "--controller", action="store_true",
        help="打印控制器**实际改了什么**（自适应臂是否真的动作了）",
    )
    ap.add_argument(
        "--prefill-effect", action="store_true",
        help="打印控制器动作**有没有真的落到 prefill 上**（与 --controller 分开看）",
    )
    ap.add_argument(
        "--verify-config", default=None, metavar="DIR",
        help="核对每个 run 的 config.snapshot 是否等于 DIR 下的那份配置（部署戳不含 configs/）",
    )
    ap.add_argument("--trace", default=None, help="打印匹配 run 的决策轨迹（前后各若干条）")
    ap.add_argument(
        "--cycles", action="store_true",
        help="打印每个 run 的总步数、补入批次（批起点 step）与决策步号——防跨格接错步号",
    )
    args = ap.parse_args(argv)

    root = Path(args.raw_dir)
    if not root.is_dir():
        print(f"目录不存在：{root}", file=sys.stderr)
        return 2

    runs: list[tuple[str, dict]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if args.tag and args.tag not in d.name:
            continue
        m = _load(d)
        if m is not None:
            runs.append((d.name, m))
    if not runs:
        print(f"{root} 下没有找到带 meta.json 的 run（--tag={args.tag}）", file=sys.stderr)
        return 2

    # 视图**可叠加**。早先每个视图各自 `return 0`，于是 `--controller
    # --prefill-effect` 会**静默**只出第一个视图——调用方以为两件事都查过了。
    # 那正是本项目最防的一类失效（"报告成功但没做"）。现在逐个跑完再统一收口，
    # 并在叠加时先把顺序声明出来，免得读的人以为只有一份输出。
    views = [
        ("trace", args.trace is not None),
        ("controller", args.controller),
        ("cycles", args.cycles),
        ("prefill-effect", args.prefill_effect),
        ("verify-config", args.verify_config is not None),
        ("detail", args.detail is not None),
        ("dump", args.dump),
    ]
    picked = [name for name, on in views if on]
    if len(picked) > 1:
        print(f"（指定了 {len(picked)} 个视图，按此顺序全部输出：{' → '.join(picked)}）\n")
    handled = False

    if args.trace:
        pick = [(n, m) for n, m in runs if args.trace in n]
        if not pick:
            print(f"没有匹配 {args.trace!r} 的 run", file=sys.stderr)
            return 2
        for name, m in pick:
            trace = m.get("controller_trace") or []
            print("=" * 110)
            print(f"{name}   决策 {len(trace)} 条")
            print("=" * 110)
            if not trace:
                print("  （空：控制器未做任何决策。若是非自适应臂，这是正常的。）")
                continue
            keys = sorted({k for e in trace for k in e})
            print(f"  字段：{keys}\n")
            show = trace[:4] + ([{"...": f"省略 {len(trace) - 8} 条"}] if len(trace) > 8 else []) + trace[-4:]
            for e in show:
                print("  " + json.dumps(e, ensure_ascii=False))
        handled = True

    if args.controller:
        # 为什么必须单独看这个：如果"自适应臂"里控制器**从未改动过任何参数**，
        # 那么 adaptive 与 static_chunk 的差异就等于零，而这个零**不是**
        # "自适应没收益"的证据——它是"这一负载下控制器无事可做"。
        # 两种"没差别"的成因完全不同，写结论前必须分清。
        print("=" * 110)
        print("控制器实际动作（自适应臂）——回答'它到底改了什么'")
        print("=" * 110)
        hdr = (f"{'run':<34}{'decisions':<11}{'changed':<9}{'recover':<9}"
               f"{'cycl':<6}{'chunk_req(min/max)':<20}{'mbt_req(min/max)':<20}{'over_mem'}")
        print(hdr)
        print("-" * 110)
        for name, m in runs:
            ce = m.get("controller_effect")
            if not isinstance(ce, dict):
                continue
            if "adaptive" not in name:
                continue

            def mm(key: str) -> str:
                v = ce.get(key)
                if not isinstance(v, dict):
                    return "-"
                return f"{v.get('min')}/{v.get('max')}"

            print(
                f"{name.split('phase1_anchor-')[-1]:<34}"
                f"{str(ce.get('n_control_decisions')):<11}"
                f"{str(ce.get('n_decisions_changed')):<9}"
                f"{str(ce.get('n_decisions_recover')):<9}"
                f"{str(ce.get('n_cycles')):<6}"
                f"{mm('chunk_requested'):<20}"
                f"{mm('max_batched_tokens_requested'):<20}"
                f"{ce.get('n_decisions_over_memory')}"
            )
        print()
        print("读法：chunk_req(min/max) 若 min == max，说明控制器**没有改过** chunk_size；")
        print("      那 adaptive 与 static_chunk 的差异就来自其它参数（如 max_batched_tokens）")
        print("      或为零。若两者皆恒定，结论只能是'本负载下控制器无事可做'，")
        print("      **不能**写成'自适应没有收益'——这两句是不同的事实。")
        handled = True

    if args.cycles:
        # 为什么需要第三个视图：**决策步号是 run 内的局部量**。
        # 本仓库踩过：把 c2 的 264/288/392/416 当成 c4 的决策链写进了结论，
        # 而 c4 首轮只有 263 步。只看一列"step"是发现不了的——
        # 必须把"总步数 / 批次数 / 批起点 / 决策步号"放在**同一行**，接错才对不上。
        # 另一层：看到"周期性调整"要先问"周期由谁定"。
        # 闭模型负载是**补位**式的（前 concurrency 个立即到达，其余随执行槽释放补入），
        # 所以批次数 = num_requests / concurrency，而"周期"就是批间隔，
        # **是负载的量，不是控制器的固有周期**。
        print("=" * 118)
        print("补入批次 × 决策步号 —— 回答'反复调整是跟着负载、还是控制器自己'")
        print("=" * 118)
        print(f"{'run':<26}{'总步数':<9}{'并发':<6}{'批次数':<8}{'批起点(step)':<36}{'决策(step:方向)'}")
        print("-" * 118)
        for name, m in runs:
            rep = cycle_report(name, m, root / name)
            bs = (
                ",".join(str(x) if x is not None else "?" for x in rep["batches"])
                if rep["batches"]
                else "-"
            )
            if not rep["controller_enabled"]:
                ds = "（控制器未启用）"
            else:
                ds = " ".join(f"{d['step']}:{d['dir']}" for d in rep["decisions"]) or "（零动作）"
            print(
                f"{name.split('phase1_anchor-')[-1]:<26}"
                f"{str(rep['total_steps']):<9}"
                f"{str(rep['concurrency']):<6}"
                f"{len(rep['batches']):<8}"
                f"{bs:<36}"
                f"{ds}"
            )
        print()
        print("读法：① 批次数 = num_requests / concurrency（闭模型补位）；批间隔 ≈ 一个请求的 e2e。")
        print("      ② 若收缩决策稳定落在每个批起点之后的第一个观测点，那是**跟随补入**，")
        print("         不是控制器自激——'周期'是负载的，不是控制器的固有周期。")
        print("      ③ 总步数用于防盗引：跨格引用步号前，先确认它 < 该格总步数。")
        print("      ④ 到达切批的阈值 = max(500 ms, 0.5×最大间隔)；批内/批间间隔同量级时切法不可靠，")
        print("         此时应改用 --detail 或直接读 requests.jsonl 的 arrival_ms。")
        handled = True

    if args.prefill_effect:
        # 与 --controller 有意**分开**成两个视图：合成一张表会让"改了参数"与
        # "改到了 prefill"变成同一个格子，而这两件事在真机产物里确实分开过。
        print("=" * 118)
        print("控制器动作是否**落到 prefill 上**——回答'这些改动有没有可施加的对象'")
        print("=" * 118)

        by_conc: dict[int, dict[str, tuple[str, dict]]] = {}
        for name, m in runs:
            c = concurrency_of(name)
            if c is None:
                continue
            arm = ("static_chunk" if "-static_chunk_" in name
                   else "adaptive" if "-adaptive_" in name
                   else "static" if "-static_" in name else None)
            if arm is None:
                continue
            # 同名多份（-r1/-r2/-r3）只保留基准那份，避免把重复当成不同档
            if arm not in by_conc.setdefault(c, {}):
                by_conc[c][arm] = (name, m)

        verdicts: list[str] = []
        for c in sorted(by_conc):
            arms = by_conc[c]
            adapt = arms.get("adaptive")
            if adapt is None:
                continue
            name, m = adapt
            run_dir = root / name
            rows, bad_lines = load_telemetry(run_dir)
            pf = prefill_events(rows)
            unparsed = count_unparsed(rows)
            need = expected_prefill_tokens(run_dir)
            got = sum(p["prefill_tok"] for p in pf)
            trace = m.get("controller_trace") or []
            effects = action_effects(trace, pf)
            tally = effect_tally(effects)

            dist = {}
            for p in pf:
                dist[p["chunk_size"]] = dist.get(p["chunk_size"], 0) + 1

            print()
            print(f"── 并发 {c} ──  {name}")
            print(f"   prefill 事件 {len(pf)} 个；实际 prefill 总 token {got:.0f}"
                  + (f"；requests.jsonl 求和 {need:.0f}"
                     f"{'（一致 ✔）' if need == got else '（**不一致 ✘**）'}" if need is not None
                     else "；requests.jsonl 求和：读不到（None）"))
            print(f"   prefill 时实际生效的 chunk_size 分布：{dist}")
            print(f"   changed 决策 {len(effects)} 次；telemetry 坏行 {bad_lines}；"
                  f"note 缺 prefill_tok 的行 {unparsed}")
            if unparsed:
                print("   ⚠️ 有行解析不出 prefill_tok：上面的事件数**不可信**，先核对日志格式。")
            if not effects:
                print("   ⇒ **零动作**：控制器全程未改动任何取值。")
                print("      此时 adaptive 与 static_chunk 的差异必然是 0，"
                      "而这个 0 是'无事可做'，**不是**'自适应没收益'。")
                verdicts.append(f"c{c}: 零动作（差异为 0 属无事可做）")
            for k, n in tally.items():
                if n:
                    print(f"     · {k:<24} {n} 次 —— {ACTION_KIND_NOTE[k]}")
            decisive = [e for e in effects
                        if e["kind"] in (ACTION_APPLIED, ACTION_NOT_APPLIED,
                                         ACTION_NO_PREFILL_AFTER, ACTION_SUPERSEDED)]
            for e in decisive:
                if e["kind"] == ACTION_APPLIED:
                    tail = (f"其后首个 prefill 在 step {e['next_prefill_step']}"
                            f"（{e['next_prefill_tok']:.0f} tok）用 {e['next_prefill_chunk']}")
                elif e["kind"] == ACTION_NO_PREFILL_AFTER:
                    tail = "此后**没有** prefill，这次改动没有施加对象"
                elif e["kind"] == ACTION_SUPERSEDED:
                    tail = (f"但 step {e['next_decision_step']} 又改了它，"
                            f"而下一个 prefill 要到 step {e['next_prefill_step']}"
                            f"（用 {e['next_prefill_chunk']}）—— 这个窗口里没有 prefill")
                else:
                    tail = (f"值一直保持到 step {e['next_prefill_step']}，"
                            f"而那次 prefill 用的却是 {e['next_prefill_chunk']} —— **需查机制**")
                print(f"     step {str(e['step']):<5} chunk {e['chunk_before']}→{e['chunk_after']}"
                      f"  {tail}")
                print(f"        理由：{e['reason']}")
            if not effects:
                pass
            elif tally[ACTION_NOT_APPLIED]:
                verdicts.append(
                    f"c{c}: {tally[ACTION_APPLIED]} 次落到 prefill 上，"
                    f"**{tally[ACTION_NOT_APPLIED]} 次改了却没生效（需查）**"
                )
            elif tally[ACTION_APPLIED]:
                extra = (f"，另有 {tally[ACTION_SUPERSEDED]} 次因窗口内无 prefill 被覆盖"
                         if tally[ACTION_SUPERSEDED] else "")
                verdicts.append(f"c{c}: {tally[ACTION_APPLIED]} 次落到 prefill 上{extra}")
            else:
                verdicts.append(f"c{c}: 有改动但**没有一次**落到 prefill 上")
            contrast = arms.get("static_chunk")
            if contrast is not None:
                crows, _ = load_telemetry(root / contrast[0])
                cpf = prefill_events(crows)
                cdist = {}
                for p in cpf:
                    cdist[p["chunk_size"]] = cdist.get(p["chunk_size"], 0) + 1
                print(f"   对照 static_chunk（{contrast[0]}）：prefill 事件 {len(cpf)} 个，"
                      f"chunk 分布 {cdist}")

        print()
        print("=" * 118)
        print("小结（每档一句话）")
        for v in verdicts:
            print(f"  · {v}")
        print()
        print("读法：'动作落到 prefill 上'的档位，控制器**确实**改变了 prefill 的节奏，")
        print("      因此该档的 TTFT 差异可以归因到控制器；'零动作'的档位差距为 0，")
        print("      **不得**把它读成'自适应无收益'——那是另一个事实。")
        print("      另注意 superseded_before_prefill 与 not_applied_to_prefill 的区别：")
        print("      前者是'窗口里没有 prefill'（负载形状问题），**不是**机制缺陷；")
        print("      只有后者才是'改了却没生效'。混为一谈会把前者误判成后者。")
        handled = True

    if args.verify_config:
        # 为什么必须单独查：部署戳（SOURCE_PROVENANCE.json）的 digest_scope 只含
        # src/ascend_opt、benchmark、scripts，**不含 configs/**。于是"代码是新的"
        # 与"配置是新的"是两件事，前者成立**推不出**后者——已经踩过一次：远端留着
        # 旧模板（slo.tpot_ms=50）且没有新目录，批次照样跑得动、照样打 MEASURED。
        # 这条是**能让批次失败的**检查（有问题返回 1），不是只打印一行提示。
        cfg_dir = Path(args.verify_config)
        if not cfg_dir.is_dir():
            print(f"配置目录不存在：{cfg_dir}", file=sys.stderr)
            return 2
        # --tag 是**子串**匹配，很容易把别的批次一并扫进来（例：`--tag adaptive_c4`
        # 会同时命中 P1S3/P1S3B/P1S3D 三个批次）。一个配置目录不可能同时是三个批次
        # 的输入，那样比出来的"不一致"全是假的。宁可直接拒绝，也不要印一堆噪音。
        prefixes = sorted({name.split("-", 1)[0] for name, _m in runs})
        if len(prefixes) > 1:
            print(f"拒绝：匹配到 {len(prefixes)} 个批次（{', '.join(prefixes)}），"
                  f"一个配置目录不可能同时是它们的输入。", file=sys.stderr)
            print("   --tag 是子串匹配；请用它精确到单个批次（例如 --tag P1S3D）后重跑。",
                  file=sys.stderr)
            return 2
        print("=" * 100)
        print(f"每个 run 的快照 vs {cfg_dir.name}/：这次跑的到底是不是这份配置")
        print("=" * 100)
        n_problems = 0
        used_cells: set[str] = set()
        for name, _m in runs:
            cell = cell_of(name)
            if cell is None:
                print(f"  ✘ {name}: 名字里取不出格名，无法定位配置文件")
                n_problems += 1
                continue
            used_cells.add(cell)
            snap_path = root / name / "config.snapshot.yaml"
            exp_path = cfg_dir / f"{cell}.yaml"
            if not snap_path.exists():
                print(f"  ✘ {cell:<22} 快照缺失：{snap_path}")
                n_problems += 1
                continue
            if not exp_path.exists():
                print(f"  ✘ {cell:<22} 配置目录里没有 {exp_path.name}")
                n_problems += 1
                continue
            try:
                cfg = yaml.safe_load(exp_path.read_text(encoding="utf-8"))
                snap = yaml.safe_load(snap_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"  ✘ {cell:<22} YAML 解析失败：{exc}")
                n_problems += 1
                continue
            if not isinstance(cfg, dict) or not isinstance(snap, dict):
                print(f"  ✘ {cell:<22} 顶层不是映射（配置/快照至少一边不是 dict）")
                n_problems += 1
                continue
            probs, extra = config_mismatches(cfg, snap)
            if probs:
                n_problems += len(probs)
                print(f"  ✘ {cell:<22} 与 {exp_path.name} 有 {len(probs)} 处不一致")
                for p in probs:
                    print(f"        {p}")
            else:
                tail = f"（快照另有 {extra} 个默认值键）" if extra else ""
                print(f"  ✔ {cell:<22} 与 {exp_path.name} 逐字段一致{tail}")
        orphans = sorted(p.stem for p in cfg_dir.glob("*.yaml")
                         if p.stem not in used_cells)
        if orphans:
            print(f"\n  仅提示｜{cfg_dir.name}/ 里有配置但没有对应 run：{', '.join(orphans)}")
        print()
        if n_problems:
            print(f"⇒ **{n_problems} 处问题**：本批不能当作 {cfg_dir.name}/ 的产物。")
            print("   部署戳不含 configs/，所以「代码对」不能代替「配置对」。")
            return 1
        print(f"⇒ {len(runs)} 个 run 的快照与 {cfg_dir.name}/ 逐字段一致，输入是新的。")
        handled = True

    if args.detail:
        pick = [(n, m) for n, m in runs if args.detail in n]
        if not pick:
            print(f"没有匹配 {args.detail!r} 的 run", file=sys.stderr)
            return 2
        for name, m in pick:
            s = m.get("summary", {}) or {}
            extra = s.get("extra", {}) if isinstance(s.get("extra"), dict) else {}
            print("=" * 90)
            print(f"{name}")
            print("=" * 90)
            print(f"  recorded_at      : {m.get('recorded_at')}")
            print(f"  repeat_index     : {(m.get('context') or {}).get('repeat_index')}")
            print(f"  data_label       : {m.get('data_label')}")
            print(f"  n_samples        : {m.get('n_samples')}")
            print(f"  wall_ms          : {s.get('wall_ms')}")
            print(f"  throughput_tok_s : {s.get('throughput_tok_s')}")
            print(f"  stability        : {json.dumps(s.get('stability'), ensure_ascii=False)}")
            print(f"  summary.extra 键 : {sorted(extra.keys())}")
            bs = extra.get("backend_stats")
            if isinstance(bs, dict):
                print(f"  backend_stats.driver : "
                      f"{json.dumps(bs.get('driver'), ensure_ascii=False)}")
                for k in sorted(bs.keys()):
                    if k == "driver":
                        continue
                    print(f"  backend_stats.{k} : "
                          f"{json.dumps(bs[k], ensure_ascii=False)[:400]}")
            r = m.get("reproducibility", {}) or {}
            for k in ("measurement_count", "warmup_count", "concurrency", "model",
                      "git_commit", "device_count", "batch_configuration",
                      "chunk_size", "max_batched_tokens", "input_length"):
                if k in r:
                    print(f"  repro.{k:<20}: {r[k]!r}")
            # 装载/预热是否真的发生在本次 run：这三项一起看才能回答
            # "三次重复是不是各自独立装了一遍模型"。
            for k in ("load_s", "warmup_s", "load_ms", "warmup_ms"):
                if k in extra:
                    print(f"  extra.{k:<19}: {extra[k]!r}")
        handled = True

    if args.dump:
        name, m = runs[0]
        print(f"=== {name} 的 meta 顶层结构 ===")
        for k, v in m.items():
            if isinstance(v, dict):
                print(f"  {k}: dict, keys={sorted(v.keys())[:18]}")
            elif isinstance(v, list):
                print(f"  {k}: list(len={len(v)})")
            else:
                print(f"  {k}: {v!r}")
        s = m.get("summary", {})
        if isinstance(s, dict):
            print(f"\n  summary 的键：{sorted(s.keys())}")
        handled = True

    if handled:
        # 任何视图跑过就不再落到"逐 run 明细"这个默认路径上。
        # `--verify-config` 走到这里时必为 0 问题（有问题在上面已经 return 1）。
        return 0

    # ---- 逐 run 明细 ----
    print("=" * 100)
    print(f"{'run':<46}{'标签':<14}{'起测':<22}{'TTFT_p50':<10}{'TPOT_p50':<10}{'stall_p99':<11}{'stall比'}")
    print("=" * 100)
    for name, m in runs:
        s = m.get("summary", {}) or {}
        label = m.get("data_label") or s.get("data_label") or "?"
        ttft = _pct(s.get("ttft"))
        tpot = _pct(s.get("tpot"))
        stall = s.get("stall") if isinstance(s.get("stall"), dict) else None
        stall_p50 = _pct(stall, "p50")
        stall_p99 = _pct(stall, "p99")
        ratio = (stall_p99 / stall_p50) if (stall_p99 and stall_p50) else None
        ctx = m.get("context", {}) or {}
        started = ctx.get("started_at") or ctx.get("start") or ""
        f = lambda v: f"{v:.2f}" if isinstance(v, float) else "-"  # noqa: E731
        print(
            f"{name:<46}{str(label):<14}{str(started)[:21]:<22}"
            f"{f(ttft):<10}{f(tpot):<10}{f(stall_p99):<11}"
            f"{(f'{ratio:.2f}' if ratio else '-'):}"
        )

    # ---- 按"格"聚合：去掉 -rN 后缀 ----
    groups: dict[str, list[dict]] = {}
    for name, m in runs:
        base = name
        for i in (1, 2, 3, 4, 5):
            base = base.replace(f"-r{i}", "")
        groups.setdefault(base, []).append(m)

    print("\n" + "=" * 100)
    print("每格的离散度（3 个**原始值**，不只给中位数）")
    print("=" * 100)
    tiny_n = 0
    total_n = 0
    masked: list[str] = []
    by_metric: dict[str, list[float]] = {"TTFT p50": [], "TPOT p50": [], "stall p99": []}
    for base, ms in sorted(groups.items()):
        ttfts = [_pct(m.get("summary", {}).get("ttft")) for m in ms]
        tpots = [_pct(m.get("summary", {}).get("tpot")) for m in ms]
        stalls = [
            _pct((m.get("summary", {}) or {}).get("stall")
                 if isinstance((m.get("summary", {}) or {}).get("stall"), dict) else None,
                 "p99")
            for m in ms
        ]
        cv_t = _robust_cv([v for v in ttfts if v is not None])
        cv_p = _robust_cv([v for v in tpots if v is not None])
        cv_s = _robust_cv([v for v in stalls if v is not None])
        n = len(ms)
        print(f"\n{base}   n={n}")

        def _line(title: str, vals: list, cv: float | None) -> None:
            # 三元表达式在这里必须**只**作用在 CV 那一段上。写成
            # f"..." f"..." if cv is not None else "..." 时，条件表达式会把
            # 两个 f-string 的拼接整体当成真分支，于是 cv 为 None 时
            # **连原始值一起吞掉**——正好丢掉最该看的东西。
            shown = " / ".join(f"{v:.2f}" if v is not None else "-" for v in vals)
            if cv is None:
                cv_txt = "稳健CV=-（样本不足）"
            else:
                cp = _plain_cv(vals)
                cv_txt = (f"稳健CV={cv:.4f}" if cp is None
                          else f"稳健CV={cv:.4f}  普通CV={cp:.4f}")
            print(f"  {title:<9}{shown:<44}{cv_txt}")

        # 三个指标**都要**过一遍判据。原版只查 TTFT，于是漏掉了矩阵里最该看的那条：
        # static_c8 的 stall p99 = 479.49/458.41/458.49，稳健CV=0.0002（看着极稳），
        # 普通CV=0.0213 —— 首点离群被 MAD 抹平了。
        for title, vals, cv in (("TTFT p50", ttfts, cv_t),
                                ("TPOT p50", tpots, cv_p),
                                ("stall p99", stalls, cv_s)):
            _line(title, vals, cv)
            if cv is None:
                continue
            total_n += 1
            by_metric[title].append(cv)
            if cv < TINY_CV:
                tiny_n += 1
            cp = _plain_cv(vals)
            if cp is not None and cp > MASK_RATIO * cv:
                raw = " / ".join(f"{v:.2f}" if v is not None else "-" for v in vals)
                masked.append(
                    f"{base} · {title}: 稳健CV={cv:.4f} 但普通CV={cp:.4f}"
                    f"（{cp / cv:.0f} 倍）—— 单点离群被 MAD 抹平，原始值 {raw}"
                )

    # ---- 整体形状 ----
    # 逐格罗列"CV 极小"在没有信息量时是反效果：本负载下几乎所有格的所有指标
    # 都极小，于是"可疑清单"变成"全部可疑"，判据就失效了。先看整体分布，
    # 再决定是个别格异常还是整体现象。
    print("\n" + "=" * 100)
    print("离散度的整体形状（后面要用的'噪声底'只能从这里来，必须整体看）")
    print("=" * 100)
    for title in ("TTFT p50", "TPOT p50", "stall p99"):
        cvs = sorted(by_metric[title])
        if not cvs:
            continue
        print(f"  {title:<10} n={len(cvs):<3} 最小={cvs[0]:.4f}  "
              f"中位={statistics.median(cvs):.4f}  最大={cvs[-1]:.4f}")
    print(f"\n  稳健CV < {TINY_CV} 的「格 × 指标」：{tiny_n}/{total_n}")
    if total_n and tiny_n / total_n > 0.5:
        print("  ⇒ 这是**整体现象**，不是个别格的问题。按 §1.5 第2条：")
        print("     要么整体排除共享热状态，要么整体承认'该负载在本平台上重复性就是好'；")
        print("     **不得**从中挑出几格当作'稳定'的证据，也不得把它当其它负载的噪声底。")
    if masked:
        print("\n" + "!" * 100)
        print("可疑｜稳健 CV 掩盖了单点离群（MAD 是'偏离中位数'的**中位数**，")
        print("      n=3 时一个离群点不改变它 ⇒ '看着极稳'可能是假象，须单独看原始值）：")
        for s in masked:
            print(f"  · {s}")
        print("!" * 100)

    labels = {m.get("data_label") or (m.get("summary", {}) or {}).get("data_label")
              for _, m in runs}
    print(f"\n标签集合：{sorted(x for x in labels if x)}")
    if labels - {"MEASURED"}:
        print("⚠️ 存在非 MEASURED 的 run，聚合时必须按标签过滤。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
