#!/usr/bin/env python3
"""生成 PHASE 1 第⑥步的**静态扫描**配置（`configs/mthreads/phase1_step6/`）。

为什么另起一份生成器而不是改第③步那份
--------------------------------------
第③步的生成器被它自己的测试**钉死**在"3 臂 × 4 并发 × 4 profile"的矩阵形状上，
并且**模板是 32 并发**。第⑥步要做的是另一件事：

* **单臂 8 并发**：把 `scheduler.max_num_seqs` 从模板的 **32 显式覆盖成 8**
  （pilot 的前提之一），负载固定在一个臂上；
* 在这一个臂上扫**开关 / 分块 / 预算 / 门限 / 并发 / 负载轨迹**六个维度。

⇒ 与其让第③步那份长出两套语义，不如单起一份、把共同的部分**导入复用**。

三条纪律（与第③步一致，这里再写一遍因为它们是这份文件的正确性条件）
------------------------------------------------------------------
1. **单变量**：除锚点外，每个格与它声明的基线**只差一个点分路径**——
   由 :func:`_assert_single_variable`（**从第③步生成器导入**）强制。
2. **显式覆盖而非继承**：`scheduler.max_num_seqs` 必须**真的被覆盖**。
   仅断言"生成物里是 8"是不够的——模板若哪天也改成 8，这条断言就恒真了。
   故 :func:`_assert_max_num_seqs_explicit` **同时**要求"模板里的值不是 8"，
   让"这条覆盖是承重的"成为可核对的事实。
3. **标定过的值照抄不重算**：门限 120/0.08、SLO 100 ms 都来自已落地的标定
   （`EXPERIMENTS.md` §4.3.3/§4.3.4），本脚本只引用，**不顺手调**。

⚠️ **dry-run 能证明什么、不能证明什么**：它能证明"配置能加载、链路能跑、产物字段齐全"；
它**不能**证明过载、SLO 或性能——dry-run 走的是**一阶设备模型**，不是真机。
本脚本生成的配置是给真机 pilot 用的，**干跑结果不得用于评估这些**。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ⚠️ 复用第③步的守卫，**不重写一份**。两处各写一套判据的结果是
# "粗的那套放行"（这一点已经在第③步的门限缺陷上真实发生过）。
from gen_phase1_step3_configs import (  # noqa: E402
    _assert_controller_movable,
    _assert_gate_separation,
    _assert_single_variable,
    _assert_slo_reachable,
    _get_path,
    _set_path,
)

TEMPLATE = _ROOT / "configs" / "mthreads" / "s4000_fp16.yaml"
OUT_DIR = _ROOT / "configs" / "mthreads" / "phase1_step6"

#: ⚠️ 本步的**承重**覆盖：模板是 32，第⑥步要求单臂 8 并发。
MAX_NUM_SEQS = 8
NUM_REQUESTS = 16
CONCURRENCY = 8
WORKLOAD = "phase1_anchor"
RAMP_WORKLOAD = "load_ramp"

# ---- 照抄已落地的标定值（不重算、不顺手调）--------------------------------
SLO_TPOT_MS = 100.0                 # EXPERIMENTS §4.3.3
TARGET_TPOT_MS = 120.0              # EXPERIMENTS §4.3.4
HYSTERESIS = 0.08                   # EXPERIMENTS §4.3.4
CHUNK_SIZE = 1024
CHUNK_SIZE_MIN = 512
CHUNK_SIZE_MAX = 2048
MAX_BATCHED_TOKENS = 8192

NOTES_TAG = "PHASE1-⑥"

# ---- 负载轨迹（第 08 步 pilot 实测后**延长 recover**）----------------------
# 为什么改：pilot 用初值（recover 120 s）跑出来"恰好排空"（drain 尾段只用 8.4 s）⇒
# 该段的**余量只有 1.05×**（见 §判据）。余量这么小的后果不是"慢一点"，
# 而是**删失**：任何一次运行的服务能力略低（0.34 vs 0.356），积压就清不完，
# 恢复时间变成"测不出"——而恢复时间恰是本研究要比较的量。
# ⇒ 按 1.5× 余量反解：T ≥ 1.5·qd /(C − 1.5·λ) = 1.5×31/(0.356−0.12) ≈ 197 s ⇒ 取 240 s（余量 1.70×）。
# ⚠️ 延长**不会稀释**恢复时间指标：它是在**穿越点**测量的，加长只消除删失。
_RAMP_SEGMENTS: tuple[tuple[float, float, str], ...] = (
    (120.0, 0.08, "low"),        # 锚点：远低于容量，不积压
    (90.0, 0.70, "high"),        # 过载：λ/C ≈ 1.96 ⇒ 必然积压（实测 qd 末 31）
    (240.0, 0.08, "recover"),    # 原 120 s → 240 s（余量 1.05× → 1.70×）
)
#: 实测的持续容量与 high 段末积压（第 08 步 pilot，`PHASE1S8-load_ramp-ramp_static_c8`）。
MEASURED_CAPACITY_REQ_S = 0.356
MEASURED_BACKLOG_AT_HIGH_END = 31
#: 恢复窗口余量的**下限**（服务能力 / 待清量）。
RECOVERY_MARGIN_MIN = 1.5

#: 第 04/09 步新增、**平台模板里还没有**的 `controller` 子键。
#: ⚠️ 为什么不往模板里加：模板是"平台基线"，第③步已执行的配置由它派生——
#: 往模板里加字段会**改动那些已执行配置的字段集合**（本仓库明确要求已执行配置逐字可复现）。
#: `_set_path` 只改**已有**路径（设计如此，防拼错静默生效），所以这几个键在这里显式写。
_NEW_CONTROLLER_KEYS = ("signal_mode", "signal_criterion")


@dataclass(frozen=True)
class Cell:
    """一个配置格。

    ``baseline`` 是它应当与之**只差一个字段**的对照格（``None`` = 该组的锚点）。
    锚点也要显式声明自己的覆盖；它不是"默认"，而是"这一组的起点"。
    """

    name: str
    baseline: str | None
    axis: str
    overrides: dict[str, Any]


def _common(workload: str) -> dict[str, Any]:
    """所有格共用的取值。

    为什么把"三个臂都用不到的值"也写进每一格（与第③步同一理由）：
    字段集合不同会让"只差一个开关"的自检失败。逐字相同、只有被测那一项不同，
    才是可归因的形状。
    """
    common: dict[str, Any] = {
        # ⚠️ 承重覆盖：模板 32 → 这里 8。见 _assert_max_num_seqs_explicit。
        "scheduler.max_num_seqs": MAX_NUM_SEQS,
        "scheduler.policy": "continuous_batching",
        "chunk.policy": "fixed_chunk",
        "chunk.chunk_size": CHUNK_SIZE,
        "chunk.max_num_batched_tokens": MAX_BATCHED_TOKENS,
        "cache.enabled": False,
        "workload.name": workload,
        "workload.concurrency": CONCURRENCY,
        # 达标率判据**逐格显式写死**（不从模板继承）：平台模板承载的是"当前约定"，
        # 而"这一格当时用的是什么判据"必须逐格固定，否则模板一改，已执行配置的
        # 达标率列会被静默改写。
        "run.slo.tpot_ms": SLO_TPOT_MS,
        "controller.target_tpot_ms": TARGET_TPOT_MS,
        "controller.hysteresis": HYSTERESIS,
        "controller.chunk_size_init": CHUNK_SIZE,
        "controller.chunk_size_min": CHUNK_SIZE_MIN,
        "controller.chunk_size_max": CHUNK_SIZE_MAX,
    }
    if workload == RAMP_WORKLOAD:
        # ⚠️ 采集口径**两臂一致**（都 layered）：否则"判据"这个自变量会同时带上
        # "采集口径"的变化 ⇒ 一次改两个变量。layered 对**决策**是等价的
        # （第 04 步已证：两种模式在决策上等价），它只是多记几个键。
        common["controller.signal_mode"] = "layered"
        # piecewise 的请求数由 λ(t)×时长**决定**，且负载自己会校验一致性。
        # ⇒ 两个值一起写：轨迹 + **由轨迹算出**的请求数（不手抄，防漂移）。
        common["workload.rate_segments"] = [
            {"duration_s": d, "qps": q, "label": lb} for d, q, lb in _RAMP_SEGMENTS
        ]
        common["workload.num_requests"] = sum(round(d * q) for d, q, _ in _RAMP_SEGMENTS)
    else:
        common["workload.num_requests"] = NUM_REQUESTS
    return common


def _recovery_margin(segments=_RAMP_SEGMENTS) -> float:
    """恢复窗口的余量比 = (C×T) / (high 末积压 + λ_recover×T)。

    < ``RECOVERY_MARGIN_MIN`` ⇒ 积压可能清不完 ⇒ 恢复时间被**删失**（不是"慢"，是"测不出"）。
    """
    T = next(d for d, _, lb in segments if lb == "recover")
    lam = next(q for _, q, lb in segments if lb == "recover")
    need = MEASURED_BACKLOG_AT_HIGH_END + lam * T
    return MEASURED_CAPACITY_REQ_S * T / need


#: 静态扫描的格表。顺序 = 报告顺序。
CELLS: tuple[Cell, ...] = (
    # --- A. 驱动对照：同一负载、同一并发，只有开关在动 -----------------------
    Cell("static_c8", None, "arm_anchor",
         {"chunk.enabled": False, "controller.enabled": False}),
    Cell("static_chunk_c8", "static_c8", "arm",
         {"chunk.enabled": True, "controller.enabled": False}),
    Cell("adaptive_c8", "static_chunk_c8", "arm",
         {"chunk.enabled": True, "controller.enabled": True}),
    # --- B. 预算扫描（基线 = 8192，只动 max_num_batched_tokens）------------
    Cell("static_chunk_c8_bt2048", "static_chunk_c8", "budget",
         {"chunk.enabled": True, "controller.enabled": False,
          "chunk.max_num_batched_tokens": 2048}),
    Cell("static_chunk_c8_bt4096", "static_chunk_c8", "budget",
         {"chunk.enabled": True, "controller.enabled": False,
          "chunk.max_num_batched_tokens": 4096}),
    # --- C. 分块粒度扫描（基线 = 1024，只动 chunk_size）---------------------
    Cell("static_chunk_c8_ck512", "static_chunk_c8", "chunk_size",
         {"chunk.enabled": True, "controller.enabled": False,
          "chunk.chunk_size": CHUNK_SIZE_MIN}),
    Cell("static_chunk_c8_ck2048", "static_chunk_c8", "chunk_size",
         {"chunk.enabled": True, "controller.enabled": False,
          "chunk.chunk_size": CHUNK_SIZE_MAX}),
    # --- D. 并发轴线（基线 8，只动 workload.concurrency）-------------------
    # 16 > max_num_seqs(8) ⇒ 这一格才**真的**产生排队（8 的那格不会）。
    Cell("static_chunk_c16", "static_chunk_c8", "concurrency",
         {"chunk.enabled": True, "controller.enabled": False,
          "workload.concurrency": NUM_REQUESTS}),
    # --- H. 第⑩步：**稳态过载点**上的 ck 对照（轨迹之外的核验）--------------
    # 为什么加：S10 的响应面在**开放环轨迹**上测到"细分块（512）把 high 段的达标吞吐打到 0"。
    # 那可能是**轨迹特有**的（到达过程、阶段边界打回慢档），也可能是一般现象。
    # 这一格把同一个 ck 问题搬到**稳态闭环过载**上（并发 16 / 只受理 8 ⇒ 真排队），
    # 与 `static_chunk_c16` **只差 `chunk.chunk_size`** ⇒ 单变量。
    # ⚠️ 它与轨迹格**不是**同一个负载 ⇒ 结论只能作"同向性核对"，不得当单变量比较。
    Cell("static_chunk_c16_ck512", "static_chunk_c16", "chunk_size",
         {"chunk.enabled": True, "controller.enabled": False,
          "workload.concurrency": NUM_REQUESTS, "chunk.chunk_size": CHUNK_SIZE_MIN}),
    # --- E. 门限轴线（基线 120，只动 target_tpot_ms）----------------------
    # 门限取值**不是随手挑的**：160 被守卫 `_assert_gate_separation` 拒掉——
    # c8 的实测工作点是 tpot_p90=142.1，而 160×(1−0.08)=147.2 距它只有 5.1 ms
    # （门槛要求 > 8.0）⇒ 门限落在工作点分布内部、该格跨重复不可复现。
    # 180 ⇒ 放大门限 180×0.92=165.6，距工作点 23.5 ms ✓，且与 120 那一组构成"松/紧"对照。
    Cell("adaptive_c8_th180", "adaptive_c8", "gate",
         {"chunk.enabled": True, "controller.enabled": True,
          "controller.target_tpot_ms": 180.0}),
    # --- F. 负载轨迹扫描（piecewise；static / adaptive 对照）--------------
    Cell("ramp_static_c8", None, "ramp_anchor",
         {"chunk.enabled": True, "controller.enabled": False,
          "workload.name": RAMP_WORKLOAD}),
    Cell("ramp_adaptive_c8", "ramp_static_c8", "ramp",
         {"chunk.enabled": True, "controller.enabled": True,
          "workload.name": RAMP_WORKLOAD}),
    # 第 09 步的**干预臂**：判据从 `tpot_p90`（含 prefill 污染）换成 `decode_p90`（纯 decode）。
    # 与 `ramp_adaptive_c8` 只差这一个字段 ⇒ 单变量。E10 说这两套口径在同窗差 5.5 倍。
    Cell("ramp_adaptive_decode", "ramp_adaptive_c8", "criterion",
         {"chunk.enabled": True, "controller.enabled": True,
          "workload.name": RAMP_WORKLOAD,
          "controller.signal_criterion": "decode_p90"}),
    # --- G. 第⑩步：**轨迹上的静态响应面**（每格与 ramp_static_c8 只差一处）----
    # 为什么在轨迹上做响应面而不是闭环分档：轨迹的三段（low/high/recover）就是
    # 三个负载水平、且**在同一次 run 内**——那正是本研究的主场景；而闭环分档要同时
    # 动 concurrency 与 num_requests 两个量，口径反而更脏。
    Cell("ramp_static_c8_ck512", "ramp_static_c8", "ramp_chunk",
         {"chunk.enabled": True, "controller.enabled": False,
          "workload.name": RAMP_WORKLOAD, "chunk.chunk_size": CHUNK_SIZE_MIN}),
    Cell("ramp_static_c8_ck2048", "ramp_static_c8", "ramp_chunk",
         {"chunk.enabled": True, "controller.enabled": False,
          "workload.name": RAMP_WORKLOAD, "chunk.chunk_size": CHUNK_SIZE_MAX}),
    Cell("ramp_static_off_c8", "ramp_static_c8", "ramp_chunk",
         {"chunk.enabled": False, "controller.enabled": False,
          "workload.name": RAMP_WORKLOAD}),
    Cell("ramp_static_c8_bt2048", "ramp_static_c8", "ramp_budget",
         {"chunk.enabled": True, "controller.enabled": False,
          "workload.name": RAMP_WORKLOAD, "chunk.max_num_batched_tokens": 2048}),
)


def _template() -> dict[str, Any]:
    if not TEMPLATE.exists():
        raise SystemExit(f"模板不存在：{TEMPLATE}")
    base = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise SystemExit(f"模板根节点不是映射：{TEMPLATE}")
    return base


def _assert_max_num_seqs_explicit(cfgs: dict[str, dict]) -> list[str]:
    """`scheduler.max_num_seqs` 必须被**真的覆盖**成 8。

    ⚠️ 只断言"生成物里是 8"是不够的：模板哪天也改成 8，那条断言就**恒真**了，
    而"第⑥步把并发从 32 收到 8"这个前提就再也没人守。
    所以**同时**要求模板里的值不是 8 —— 那样"这条覆盖是承重的"才是可核对的事实。
    """
    problems: list[str] = []
    tmpl_val = _get_path(_template(), "scheduler.max_num_seqs")
    if tmpl_val == MAX_NUM_SEQS:
        problems.append(
            f"模板里 scheduler.max_num_seqs 已经是 {MAX_NUM_SEQS} ⇒ "
            "本步的'显式覆盖'变成恒真断言，守卫失去分辨力。"
            "请确认模板是否被改过（本步要求**不改模板、只在新 profile 里覆盖**）。"
        )
    for name, cfg in cfgs.items():
        val = _get_path(cfg, "scheduler.max_num_seqs")
        if val != MAX_NUM_SEQS:
            problems.append(
                f"{name}: scheduler.max_num_seqs = {val!r}，应为 {MAX_NUM_SEQS}"
            )
    return problems


def _assert_ramp_window_is_consistent(cfgs: dict[str, dict]) -> list[str]:
    """piecewise 格的**两件**结构性条件（第 08 步升级）。

    1. **请求数与轨迹一致**：piecewise 的请求数由 λ(t)×时长决定，写小了会把整条
       轨迹**截断**——而截断后的 run 看起来只是"负载更小"，不报错、不留痕，
       且"恢复段"根本走不到。
    2. **恢复窗口有余量**：余量比 < ``RECOVERY_MARGIN_MIN`` ⇒ 积压可能清不完 ⇒
       恢复时间被**删失**。这是 pilot 实测后加的一条：初值 120 s 的余量只有 1.05×。
    """
    problems: list[str] = []
    ramp_cells = [c for c in CELLS if c.overrides.get("workload.name") == RAMP_WORKLOAD]
    derived = sum(round(d * q) for d, q, _ in _RAMP_SEGMENTS)
    for cell in ramp_cells:
        cfg = cfgs.get(cell.name)
        if cfg is None:
            continue
        got = _get_path(cfg, "workload.num_requests")
        if got != derived:
            problems.append(
                f"{cell.name}: workload.num_requests={got!r}，但轨迹推出的是 {derived}"
                "（不一致 ⇒ 轨迹会被截断或负载加载即报错）"
            )
    margin = _recovery_margin()
    if margin < RECOVERY_MARGIN_MIN:
        problems.append(
            f"恢复窗口余量 {margin:.3f}× < 下限 {RECOVERY_MARGIN_MIN}× "
            f"⇒ 积压可能清不完、恢复时间会被删失。请延长 recover 段"
            f"（按 1.5× 反解 T ≥ 1.5·qd/(C − 1.5·λ)）。"
        )
    return problems


def _assert_axes_present(cfgs: dict[str, dict]) -> list[str]:
    """验收点名的覆盖项，逐格核一遍（缺了就报，不靠人记得）。

    ⚠️ 轴线要从**实际生成的 cfgs** 核，不能只按 ``CELLS`` 这张表推。
    只按表推的话，"表里声明了预算轴"与"预算格真的被生成了"是两件事——
    ``build()`` 哪天静默漏掉一格，守卫照样绿灯（这正是"守卫读的是意图、
    而不是产物"那个老坑，见 `experiment-harness-integrity` §20.5 面孔一）。
    """
    problems: list[str] = []
    axes: dict[str, list[str]] = {}
    for c in CELLS:
        if c.name in cfgs:                       # ← 只认**真的生成了**的格
            axes.setdefault(c.axis, []).append(c.name)
    missing_cells = [c.name for c in CELLS if c.name not in cfgs]
    if missing_cells:
        problems.append(f"这些格在 CELLS 里声明了、但没有被生成：{missing_cells}")
    required = {
        "arm": "固定 / 自适应臂对照",
        "budget": "预算扫描",
        "chunk_size": "分块上下限扫描",
        "concurrency": "并发轴线",
        "gate": "控制门限轴线",
        "ramp": "负载轨迹扫描",
        "criterion": "控制判据口径（第 09 步干预）",
    }
    for key, why in required.items():
        if not axes.get(key):
            problems.append(f"缺少「{why}」（轴 {key}）的格")
    for name in ("static_c8", "adaptive_c8", "ramp_static_c8", "ramp_adaptive_c8"):
        if name not in cfgs:
            problems.append(f"缺少验收点名的格：{name}")
    return problems


def build() -> dict[str, dict]:
    """产出 {配置名 -> 配置字典}。"""
    base = _template()
    out: dict[str, dict] = {}
    for cell in CELLS:
        cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))   # 深拷贝
        # 负载名从**覆盖表里读**（而不是猜）：ramp 组的格会显式带 workload.name。
        workload = cell.overrides.get("workload.name", WORKLOAD)
        common = _common(workload)
        for k, v in common.items():
            leaf = k.split(".", 1)[1] if k.startswith("controller.") else None
            if leaf in _NEW_CONTROLLER_KEYS:
                # 新键：模板里没有，直接写进 controller 段（理由见 _NEW_CONTROLLER_KEYS）
                cfg.setdefault("controller", {})[leaf] = v
            else:
                _set_path(cfg, k, v)
        for k, v in cell.overrides.items():
            leaf = k.split(".", 1)[1] if k.startswith("controller.") else None
            if leaf in _NEW_CONTROLLER_KEYS:
                cfg.setdefault("controller", {})[leaf] = v    # 同上：新键直接写
            else:
                _set_path(cfg, k, v)
        # ⚠️ notes **逐格相同**：表头（哪个格、哪条轴）不进 cfg。
        # 它写进 cfg 会被单变量自检当成一处差异（每格都不一样），
        # 也会让"未知字段加载即抛错"的约定多一个例外。
        # 表头在写盘时作为**注释**放在文件最前（见 main）——注释不是键。
        cfg["run"]["notes"] = (
            f"{NOTES_TAG} static_scan max_num_seqs={MAX_NUM_SEQS} "
            f"slo_tpot_ms={SLO_TPOT_MS} gate={TARGET_TPOT_MS}/{HYSTERESIS}"
        )
        out[cell.name] = cfg
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 PHASE 1 第⑥步静态扫描配置")
    ap.add_argument("--check", action="store_true", help="只校验，不写盘")
    ap.add_argument("--out-dir", default=None, help=f"输出目录（默认 {OUT_DIR}）")
    args = ap.parse_args(argv)

    cfgs = build()

    # ---- 逻辑校验：任何一条不过就**拒绝生成**（不是打个警告继续） ----------
    problems: list[str] = []
    problems += _assert_max_num_seqs_explicit(cfgs)
    problems += _assert_axes_present(cfgs)
    problems += _assert_ramp_window_is_consistent(cfgs)

    for cell in CELLS:
        if cell.baseline is None:
            continue
        diffs = _assert_single_variable(cfgs, cell.baseline, cell.name)
        if len(diffs) != 1:
            problems.append(
                f"{cell.name} 与基线 {cell.baseline} 相差 {len(diffs)} 处"
                f"（应为 1）：{diffs}"
            )

    for name in ("adaptive_c8", "adaptive_c8_th180", "ramp_adaptive_c8"):
        problems += [f"{name}: {p}" for p in (
            _assert_controller_movable(cfgs[name])
            + _assert_slo_reachable(cfgs[name])
            + _assert_gate_separation(cfgs[name])
        )]

    if problems:
        print("✗ 逻辑校验未通过，拒绝生成：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"✓ 逻辑校验通过：{len(cfgs)} 个格，覆盖 "
          f"{len({c.axis for c in CELLS})} 条轴线")

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    if args.check:
        print(f"（--check：未写盘；本应写到 {out_dir}）")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    by_name = {c.name: c for c in CELLS}
    for name, cfg in cfgs.items():
        cell = by_name[name]
        path = out_dir / f"{name}.yaml"
        header = (
            f"# {NOTES_TAG} 静态扫描 // cell={cell.name} axis={cell.axis} "
            f"baseline={cell.baseline or '(anchor)'}\n"
            f"# 由 scripts/gen_phase1_step6_configs.py 生成，**请勿手改**。\n"
            f"# 表头写成注释而不是 cfg 里的键：写进 cfg 会被单变量自检当成一处差异。\n"
        )
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(header)
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    print(f"已写出 {len(cfgs)} 个配置 -> {out_dir}")
    print("⚠️ 这些配置是给**真机 pilot** 用的；dry-run 只证明链路能跑，"
          "不能证明过载、SLO 或性能。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
