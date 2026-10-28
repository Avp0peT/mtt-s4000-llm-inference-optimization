#!/usr/bin/env python3
"""生成 PHASE 1 第③步性能矩阵的配置（**派生自平台基线，不手写**）。

为什么要生成而不是手抄
----------------------
第③步要在 3 个研究臂 × 4 个并发点上跑，共 12 个配置。手抄会有两个后果：

1. **不一致**：基线里改一个字段（例如 ``warmup_s``），12 个副本里总有几个忘了跟。
   这种漂移不会报错，只会让某一格悄悄跑在不同的条件下。
2. **变量不单**：每个臂应当**只**比另一个臂多一个开关。手抄时多改一处
   （例如顺手也调了 ``chunk_size``），整张矩阵的归因就废了，而产物里看不出来。

因此本脚本以 ``configs/mthreads/s4000_fp16.yaml``（已发布的平台基线）为**唯一**
模板，只覆盖构成研究臂所需的最少字段，并在写盘前**自检单变量性**（见
:func:`_assert_single_variable`）——自检失败就拒绝生成。

三个臂的定义（依据 ``FixedChunkPolicy`` 的文档意图）
----------------------------------------------------
``src/ascend_opt/chunk_policy.py`` 里写得很直白：「'固定'指的是**不随 decode 压力
变化**……控制器启用时它会变，这正是'静态切片 vs 自适应切片'的对照所在」。
于是三个臂是：

===============  ===============  ==================  ==========================
臂               chunk.enabled    controller.enabled  这一臂回答什么
===============  ===============  ==================  ==========================
``static``       False            False               "完全不切分"的基线
``static_chunk`` True             False               "**仅仅**切片"是否已经足够
``adaptive``     True             True                切片大小随负载变化是否额外有收益
===============  ===============  ==================  ==========================

* ``static -> static_chunk``：只动 ``chunk.enabled``。
* ``static_chunk -> adaptive``：只动 ``controller.enabled``。

前缀缓存臂**不存在**：真实后端显式拒绝 ``cache.enabled=true``（HF 原生 KV 缓存
无法按前缀哈希复用块），硬跑会得到一个"看起来正常但错的" ``cached_tokens``。

四份 profile（为什么需要这么多）
-------------------------------
第③步的**第一次**执行（``p1s3``，2026-09-23）事后被查出设计缺陷：被扫描的维度
**根本不可动**。两个字段各自都合理，乘在一起才失效：

* ``chunk_size_init(512) == chunk_size_min(512)`` —— 控制器的恢复目标是**配置初值**，
  下界等于初值意味着既不能收缩（已在下界）也不能恢复（已在目标），chunk 维度死锁；
* ``target_tpot_ms(50) < 实测 TPOT P90 下限(61)`` —— 目标不可达，控制器**永远**走
  ``tpot`` 分支，于是连"该不该放大"的那条判据都没被评估过。

两者的乘积是：36/36 次运行里 ``chunk_req`` 恒为 512/512、``n_recovered=0``。
产物齐全、无报错、标签全绿，但 ``adaptive`` 臂没有测到它声称要测的东西。

``p1s3b`` 是修正版。它**不是**把参数调得"更好看"，而是把两个字段挪到使
"控制器有可动空间、目标可达"成立的位置：

* ``chunk_size_init(1024) > chunk_size_min(512)`` ⇒ 该维度可动；
* ``chunk_size_max(2048) == prompt_len(2048)`` ⇒ 上界正落在"再大就等于不切分"的
  边界上。**刻意收窄**：超出这个点的 grow 会记进轨迹却不产生任何效果，
  与旧的 ``cooldown_steps`` 是同一类"死参数"——日志说动了，实际没动；
* ``target_tpot_ms(110)``、``target_ttft_ms(500)`` 取值区间由实测信号反解，
  见 :data:`PROFILES` 里 ``p1s3b`` 的注释（每个区间都是从**已登记的** P1S3
  轨迹反推出来的，不是拍脑袋）。

``p1s3`` 留在表里只为一件事：让已执行的 36 份证据（``results/raw/P1S3-*``）
**可逐字复现**。它的缺陷是已知且已声明的，生成时会被拦下（见下）。

``p1s3c`` 是第二份修正版（2026-09-24，P3），与 ``p1s3b`` **只差**
``run.slo.tpot_ms``（50 → 100）：

* ``run.slo`` 是**达标率判据**（逐请求），与 ``controller.target_tpot_ms``
  （控制目标）是两个字段。``p1s3b`` 只改了后者，前者留在 50；
* 逐请求 TPOT 实测区间是 ``[61.7, 168.0]`` ms ⇒ 50 **低于下界、恒不可达** ⇒
  ``slo_satisfaction_rate`` / ``slo_goodput_*`` 三列恒为 0，**不携带信息**
  （是"判据无效"，不是"达标率低"）。P1S3/P1S3B 两批共 84 份产物都带着这个缺陷；
* 新值 100（= 10 tok/s）落在区间内部，且位于 c4 的"不分块 88.7 / 分块 110.4"
  之间 ⇒ 判据对 ``chunk_size`` 敏感——正是本项目要测的那个权衡。
  标定工具 ``scripts/calibrate_slo.py``，可达性由 ``_assert_slo_reachable`` 兜底。

``p1s3c`` 与原两版的区别**只影响派生指标**：``run.slo`` 不参与任何调度决策，
所以它**不需要重跑**——已入库产物的达标率可由 ``calibrate_slo.py`` 按新阈值重算。

``p1s3d`` 是第三份修正版（2026-09-24，P3 后段），与 ``p1s3c`` **只差**控制门限的
两个参数（``target_tpot_ms`` 110 → 120、``hysteresis`` 0.1 → 0.08）。同一个变量的
两个参数，所以仍算单变量改动：

* 门限贴着工作点时，控制器**不报错、不停止，而是随机地动**。放大判据要求
  ``tpot_p90 < target·(1−h)``，即同时有足够 headroom；110/0.1 把它放在 99.0，
  而 c4 的实测工作点（``controller_trace`` 里控制器**实际读到**的 ``tpot_p90``）
  中位数是 98.7 —— 相距 **0.3 ms**。标定工具算出的 headroom 余量只剩
  0.0112 对门槛 0.1000 ⇒ 抛硬币；
* 后果是跨重复不可复现：同一份 ``phase1_step3b/adaptive_c4.yaml`` 三次重复的
  TTFT 是 481.4 / 481.6 / 652.1 ms（**+43%**）。这不是"抖一下"，是"结论取决于
  一次浮点比较的符号"；
* 120/0.08 ⇒ 放大门限 110.4、收缩门限 129.6：c4 的余量升到 0.1052（≈10×），
  且 c8（142.1）仍高于收缩门限 12.5 ms（不会误触防御性收缩）。
  标定工具 ``scripts/calibrate_controller_gate.py``（**权威判定** = 四态 ``verdict``
  ＋"离门限最近的**点**"距离；2026-09-24 晚修掉了它"把收缩点也算进 headroom"的缺陷，
  该缺陷会让"跨过门槛"恒报），守卫 ``_assert_gate_separation``（**生成期静态下限**
  = 中位数到门限的距离，看不见分布宽度）。两者**不可能等价**（互相都不包含对方），
  分工与实测分歧见常量 ``_GATE_MIN_SEPARATION_FRACTION`` 的注释。
* ⚠️ **120/0.08 并未做到"干净分离"**：真正的障碍是**判据形式**（绝对阈值 + 单一
  ``(target, h)`` 与跨 9× 的多档工作点分不开）——**负载连续变化会扫过整条区间**，
  必然有档位落在滞回带内。⚠️ 本节原写"c4 的 tpot_p90 实测区间 [97.8, 571.7]
  **仍跨过**新门限"，那个读数来自标定工具一个**把收缩点也算进去的坏列**（已修：
  它使"跨过门槛"恒报，且与自己的「带内」列自相矛盾）。**结论不变，理由换成负载扫描**，
  见 ``EXPERIMENTS.md`` §4.3.4 —— 不要把这个 profile 读成"门限问题已解决"。

**门限改动与 SLO 改动不同：它会改变调度行为，所以 ``p1s3d`` 必须重跑**
（``p1s3c`` 的产物不能借用）。

已知缺陷必须显式声明
--------------------
:func:`_assert_controller_movable`、:func:`_assert_slo_reachable` 与
:func:`_assert_gate_separation` 会检查 profile 的参数是否让控制器**真的能两个方向
都动**、判据是否**可达**、门限是否**与实测工作点分得开**。``p1s3`` 三条都不满足；
``p1s3b`` 不满足后两条；``p1s3c`` 不满足最后一条（它的门限仍钉在 110/0.1，为的是
保持与 ``p1s3b`` 的单变量关系）。三者都带 ``known_defects``；生成时必须加
``--allow-known-defects`` 才会放行，且缺陷会打在 stderr 上。

只有 ``p1s3d`` 无缺陷，所以它是 ``--profile`` 的**默认值**——默认值必须是"干净的那
一版"，否则 ``--check`` 不带参数会被自己的守卫拒掉。

这三条检查存在的理由，正是三次真实踩过的坑：第③步第一次执行、P3 前段、P3 后段。
这类缺陷**不会被任何运行时检查发现**——配置合法、run 成功、产物齐全，只有结论是错的。

用法::

    python scripts/gen_phase1_step3_configs.py --profile p1s3d            # 生成 12 个配置
    python scripts/gen_phase1_step3_configs.py --profile p1s3d --check    # 只校验，不写盘
    python scripts/gen_phase1_step3_configs.py --profile p1s3 \\
        --allow-known-defects --out-dir /tmp/repro                     # 复现已执行的旧配置
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE = _ROOT / "configs" / "mthreads" / "s4000_fp16.yaml"
_DEFAULT_OUT_ROOT = _ROOT / "configs" / "mthreads"

#: 被扫描的轴：并发。取自 ``benchmark.workloads.CONCURRENCY_AXIS`` 的子集——
#: 第③步只取低并发区（1/2/4/8），因为研究对象是**小 batch 的延迟**，
#: 64 那档属于吞吐压力场景，与本步要回答的问题无关。
CONCURRENCIES: tuple[int, ...] = (1, 2, 4, 8)

#: 每个配置的请求数。三臂**共用同一个值**是刻意的：并发是被扫描的轴，
#: 请求数变了就会变成"两个变量一起动"。8 个请求在 c=8 时恰好一波，
#: 在 c=1 时是 8 波——两端都能覆盖到。
NUM_REQUESTS = 8

#: 负载名。``phase1_anchor`` 的参数在 ``benchmark/workloads.py`` 里：
#: ``input_length=2048, output_length=128``。**prompt_len 是 2048 这一点对 chunk
#: 上界的取法有直接影响**（见 ``p1s3b`` 的 ``chunk_size_max``）。
WORKLOAD = "phase1_anchor"


@dataclass(frozen=True)
class Profile:
    """一份可复现的矩阵参数集。

    ``None`` 表示**不覆盖**该字段（沿用平台基线）。这不是"留空"，
    而是刻意的：``p1s3`` 必须能逐字复现已执行的配置，而它在写入时并没有动
    ``chunk_size_min`` / ``target_*`` 这几个字段——若在这里给它们填上"基线的值"，
    生成物会多出几行与原件不同的 YAML，复现检查就失去意义了。
    """

    name: str
    out_dir_name: str
    notes_tag: str
    chunk_size: int
    chunk_size_init: int
    #: ``run.slo.tpot_ms``：**达标率判据**（逐请求），与控制器的 ``target_tpot_ms``
    #: 是两个东西——后者是控制目标，前者是评价标尺。
    #:
    #: **刻意不给默认值**：它是跨实验约定（README §6），一旦跟随模板漂移，
    #: 已执行配置的达标率列就会在重生成时被静默改写。要求每个 profile 显式写出来。
    #:
    #: 类型写成 ``float | int`` 而非只写 ``float``：整数值用 **int 字面量**
    #: （``50`` 而不是 ``50.0``）—— YAML 会把类型逐字带到配置里，而模板里是
    #: ``tpot_ms: 50``。用 ``50.0`` 重生成会让 12 个已执行配置全部出现
    #: ``50 → 50.0`` 的假 diff，逐字可复现性当场断掉。（这一步踩过。）
    slo_tpot_ms: float | int
    chunk_size_min: int | None = None
    chunk_size_max: int | None = None
    target_tpot_ms: float | None = None
    target_ttft_ms: float | None = None
    #: 滞回带半宽。``None`` = 沿用平台基线（0.1）——已执行的 profile 必须如此，
    #: 否则重生成会多出几行与原件不同的 YAML，逐字可复现性当场断掉。
    #:
    #: 它同时决定**两个门限**：放大要求 ``tpot_p90 < target·(1−h)``、
    #: 收缩要求 ``tpot_p90 > target·(1+h)``。所以它是"带有多宽"的唯一旋钮，
    #: 见 :func:`_assert_gate_separation`。
    hysteresis: float | None = None
    rationale: str = ""
    known_defects: tuple[str, ...] = field(default_factory=tuple)


PROFILES: dict[str, Profile] = {
    # ---- 第一次执行的那一版（有缺陷，保留只为复现） ----
    "p1s3": Profile(
        name="p1s3",
        out_dir_name="phase1_step3",
        notes_tag="PHASE1-③",
        chunk_size=512,
        chunk_size_init=512,
        slo_tpot_ms=50,
        rationale=(
            "2026-09-23 已执行。缺陷：chunk_size_init == chunk_size_min 使 chunk 维度"
            "不可动；target_tpot_ms=50 低于实测 TPOT P90 下限。"
        ),
        known_defects=(
            "chunk_size_init(512) == chunk_size_min(512)：收缩已在下界、恢复已在目标"
            " ⇒ chunk 维度两个方向都动不了（实测 chunk_req 恒 512/512、n_recovered=0）",
            "target_tpot_ms(50.0)*(1+hysteresis)=55.0 < 实测 TPOT P90 下限 61.5"
            " ⇒ 目标不可达，控制器永远走 tpot 分支，ttft/grow 判据从未被评估",
            "run.slo.tpot_ms(50) < 实测逐请求 TPOT 下界(61.7) ⇒ 达标率三列恒为 0，"
            "判据无效（不是'达标率低'）",
        ),
    ),
    # ---- 修正版（2026-09-23 执行）----
    "p1s3b": Profile(
        name="p1s3b",
        out_dir_name="phase1_step3b",
        notes_tag="PHASE1-③b",
        chunk_size=1024,
        chunk_size_init=1024,
        chunk_size_min=512,
        chunk_size_max=2048,
        target_tpot_ms=110.0,
        target_ttft_ms=500.0,
        # 显式钉死 50：本 profile 的产物已入库（P1S3B-/P2-），必须逐字可复现。
        # 平台模板已改为 100（P3 重标定），所以**不能**让它跟着模板走。
        slo_tpot_ms=50,
        rationale=(
            "修正 §6.3 揭示的两个缺陷。chunk_size_init(1024) > chunk_size_min(512) "
            "⇒ 可动；chunk_size_max(2048) == prompt_len ⇒ 上界落在'再大即不切分'的边界；"
            "目标由 P1S3 实录信号反解（见代码注释）。"
        ),
        known_defects=(
            "run.slo.tpot_ms(50) < 实测逐请求 TPOT 下界(61.7) ⇒ 达标率三列恒为 0"
            "（判据无效）。由 p1s3c 重标定为 100 修正；本 profile 保留 50 只为复现"
            "已入库的 P1S3B-/P2- 产物。",
            "controller.target_tpot_ms(110)×(1−0.1)=99.0 落在 c4 实测工作点"
            " tpot_p90=98.7 的邻域内 ⇒ 该格行为跨重复不可复现（实测三次 TTFT"
            " 481.4/481.6/652.1）。由 p1s3d 重标定为 120/0.08 修正；本 profile"
            " 保留原值只为复现已入库产物。",
        ),
    ),
    # ---- 修正 SLO 判据的那一版（P3，2026-09-24）----
    "p1s3c": Profile(
        name="p1s3c",
        out_dir_name="phase1_step3c",
        notes_tag="PHASE1-③c",
        chunk_size=1024,
        chunk_size_init=1024,
        chunk_size_min=512,
        chunk_size_max=2048,
        target_tpot_ms=110.0,
        target_ttft_ms=500.0,
        slo_tpot_ms=100,
        rationale=(
            "与 p1s3b 只差 run.slo.tpot_ms（50 → 100）——单变量重标定。"
            "依据：逐请求 TPOT 实测区间 [61.7, 168.0] ms（P1S3B+P2，128 请求），"
            "50 低于下界 ⇒ 达标率三列恒 0；100（=10 tok/s）落在区间内，且位于"
            "c4 的'不分块 88.7 / 分块 110.4'之间 ⇒ 判据对 chunk_size 敏感。"
            "标定工具 scripts/calibrate_slo.py。"
            "**不改调度行为**：run.slo 只影响派生指标，不影响任何决策。"
        ),
        known_defects=(
            "controller.target_tpot_ms(110)×(1−0.1)=99.0 落在 c4 实测工作点"
            " tpot_p90=98.7 的邻域内 ⇒ 该格行为跨重复不可复现（P1S3B 实测三次"
            " TTFT = 481.4/481.6/652.1，+43% 的那次即由此而来）。"
            "由 p1s3d 重标定为 120/0.08 修正；本 profile 保留 110/0.1 只为"
            "保持与 p1s3b 的单变量关系（两者只差 SLO 判据一个字段）。",
        ),
    ),
    # ---- 修正控制门限的那一版（P3 后半，2026-09-24）----
    "p1s3d": Profile(
        name="p1s3d",
        out_dir_name="phase1_step3d",
        notes_tag="PHASE1-③d",
        chunk_size=1024,
        chunk_size_init=1024,
        chunk_size_min=512,
        chunk_size_max=2048,
        target_tpot_ms=120.0,
        target_ttft_ms=500.0,
        hysteresis=0.08,
        slo_tpot_ms=100,
        rationale=(
            "与 p1s3c 只差 controller.target_tpot_ms（110→120）与 hysteresis"
            "（0.1→0.08）——同一个变量的两个参数。"
            "依据：控制器**实际读到**的 tpot_p90 工作点中位数为 c1 62.0 / c2 81.9 /"
            " c4 98.7 / c8 142.1（P1S3B+P2，adaptive 臂，"
            "scripts/calibrate_controller_gate.py）。"
            "110/0.1 ⇒ 放大门限 99.0，距 c4 中位数仅 **0.3 ms**；标定工具算出的"
            "headroom 余量 = 0.1112 − 0.1000 = **0.0112**（门槛 0.1000）⇒ 抛硬币。"
            "120/0.08 ⇒ 放大门限 110.4（距 c4 中位数 11.7 ms）、收缩门限 129.6；"
            "c4 的 headroom 余量升到 0.1852 − 0.08 = **0.1052**（≈10×），"
            "c8 中位数 142.1 仍高于收缩门限 12.5 ms。"
            "候选 130/0.1 被否：收缩门限 143.0 跨过 c8 中位数 142.1，"
            "该档收缩触发数从 17/18 掉到 5/18（把一个确定行为变成不确定行为）。"
            "⚠️ **并未做到'干净分离'**：c4 的 tpot_p90 实测区间是 [97.8, 571.7]，"
            "**仍跨过**新门限 110.4（放大 60/64、收缩 4/64、带内 0/64）。"
            "本 profile 只是把最坏余量从 0.0112 抬到 0.1052；"
            "'绝对阈值 + 单一 (target,h) 与多档工作点分不开'是判据形式的结构性障碍，"
            "见 EXPERIMENTS.md §4.3.4。"
            "标定工具 scripts/calibrate_controller_gate.py（权威口径：headroom 余量）；"
            "守卫 _assert_gate_separation（粗代理：只比中位数距离，见其 docstring）。"
        ),
    ),
}

#: 研究臂定义：臂名 -> 需要覆盖的字段（点分路径）。
#: 刻意只列出**必需的**覆盖项，其余全部继承平台基线。
ARMS: dict[str, dict[str, object]] = {
    "static": {
        "chunk.enabled": False,
        "controller.enabled": False,
    },
    "static_chunk": {
        "chunk.enabled": True,
        "controller.enabled": False,
    },
    "adaptive": {
        "chunk.enabled": True,
        "controller.enabled": True,
    },
}


def _common_overrides(profile: Profile) -> dict[str, object]:
    """所有臂共用的覆盖（不构成变量，只是把"与切分相关的取值"对齐到本次矩阵）。

    每一项都打进 ``_COMMON``（而不是只打进 chunked 臂）是刻意的：
    ``static`` 臂虽然 ``chunk.enabled=False`` 用不到这些值，但如果只有另两臂带这些
    字段，臂间的字段集合就会不同——"只差一个开关"的自检立刻失败。让三臂逐字相同、
    只有 ``enabled`` 不同，才是可归因的形状。
    """
    common: dict[str, object] = {
        "chunk.policy": "fixed_chunk",
        "chunk.chunk_size": profile.chunk_size,
        "chunk.max_num_batched_tokens": 8192,
        "controller.chunk_size_init": profile.chunk_size_init,
        "workload.name": WORKLOAD,
        "workload.num_requests": NUM_REQUESTS,
        "cache.enabled": False,
        # 达标率判据。**显式写进每个 profile**（不从模板继承）：模板承载的是
        # "当前约定"，而"这一格当时用的是什么判据"必须逐 profile 固定，
        # 否则平台模板一改，已执行配置的达标率列就会被静默改写。
        "run.slo.tpot_ms": profile.slo_tpot_ms,
    }
    # None = 沿用平台基线（复现已执行配置时必须如此，理由见 Profile 的 docstring）。
    optional = {
        "controller.chunk_size_min": profile.chunk_size_min,
        "controller.chunk_size_max": profile.chunk_size_max,
        "controller.target_tpot_ms": profile.target_tpot_ms,
        "controller.target_ttft_ms": profile.target_ttft_ms,
        "controller.hysteresis": profile.hysteresis,
    }
    for k, v in optional.items():
        if v is not None:
            common[k] = v
    return common


def _set_path(payload: dict, dotted: str, value: object) -> None:
    """按点分路径写值。

    语义刻意分两种，因为两类 key 的性质不同：

    * ``workload.*`` —— **允许新建叶子**。``workload:`` 段在模板里只写了
      ``name``，其余字段是"覆盖项"，它们必须已存在于 ``benchmark/workloads.py``
      的负载定义里（``get_workload`` 会对未知名报错），而不是必须已存在于本模板。
    * 其余路径 —— **要求叶子已存在**。这些是真实配置字段，拼错一个字母
      （``chunk_sizee``）会被下游的"未知字段加载即抛错"拦下，但在这里就报错
      能指出是哪一行生成逻辑写错了。

    父级一律必须已存在：段名拼错（``workloda.num_requests``）在两种语义下都是错。
    """
    parts = dotted.split(".")
    node = payload
    for p in parts[:-1]:
        if p not in node:
            raise KeyError(f"模板里没有 {p!r}（路径 {dotted}）——拼错了？")
        node = node[p]
    leaf = parts[-1]
    if leaf not in node and not dotted.startswith("workload."):
        raise KeyError(f"模板里没有 {leaf!r}（路径 {dotted}）——拼错了？")
    node[leaf] = value


def _get_path(payload: dict, dotted: str) -> object:
    node: object = payload
    for p in dotted.split("."):
        node = node[p]  # type: ignore[index]
    return node


def _assert_single_variable(
    configs: dict[str, dict], arm_a: str, arm_b: str
) -> list[str]:
    """列出两个臂之间**真正**不同的点分路径。

    返回差异清单；调用方据此断言"只差一个开关"。这是本脚本存在的主要理由：
    单变量性是整张矩阵可归因的前提，而它**不会**被任何运行时检查发现——
    配置合法、run 成功、产物齐全，只是结论错了。
    """
    diffs: list[str] = []

    def walk(a: object, b: object, prefix: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k)
            return
        if a != b:
            diffs.append(prefix)

    walk(configs[arm_a], configs[arm_b], "")
    return diffs


def _assert_controller_movable(cfg: dict) -> list[str]:
    """检查「被扫描的维度是否真的可动」。

    这里查的都是**静态可判**的结构性条件；"目标可达"没法静态判（它取决于实测信号），
    所以只在 ``p1s3b`` 的注释里写清反解依据，并在重跑后用
    ``scripts/check_step3b_reachability.py`` 以实录信号开环复放来把关。

    读的是**生成的配置**（``cfg``）而不是 ``Profile`` 的字段：``Profile`` 里
    ``None`` 表示"沿用基线"，若从那里读，三臂中那些"没显式覆盖"的字段会全部
    跳过检查——于是守卫对复现旧配置这条路径**恒真**。这不是假设：
    本函数第一版就是这么写的，所以 ``p1s3``（恰恰是有缺陷的那份）能一路通过。
    凡是拿 profile 当输入的结构检查，都要改成拿"最终产物"当输入。

    三类问题，每一类都对应一次真实失效：

    1. ``chunk_size_min >= chunk_size_init``：收缩已在下界、恢复已在目标 ⇒ 维度死锁。
    2. ``chunk_size_max <= chunk_size_init``：放大无空间 ⇒ 只有单向（棘轮）。
    3. ``chunk_size_max`` 超过 prompt 全长：超过的部分"grow 了但不切分"，
       轨迹显示它在动，实际什么都没发生——日志可信度被稀释，
       而检查"建议值 vs 生效值差额"的手法会因此失效。
    """
    ctl = cfg["controller"]
    init = ctl["chunk_size_init"]
    lo = ctl["chunk_size_min"]
    hi = ctl["chunk_size_max"]
    step = ctl["chunk_size_step"]
    workload = cfg["workload"]["name"]

    problems: list[str] = []
    if lo >= init:
        problems.append(
            f"chunk_size_min({lo}) >= chunk_size_init({init})：chunk 维度不可收缩"
        )
    if hi <= init:
        problems.append(
            f"chunk_size_max({hi}) <= chunk_size_init({init})：chunk 维度不可放大"
        )
    # prompt 全长取自 ``phase1_anchor`` 的 input_length。刻意**不** import 负载模块：
    # 生成器不该依赖 benchmark 层。代价是这个常量与 workloads.py 之间需要人工同步，
    # 而"需要人工同步的耦合"藏在常量表里比散在逻辑里更容易被 review 到。
    prompt_len = _PROMPT_LEN_FOR_DEADZONE.get(workload)
    if prompt_len is None:
        problems.append(
            f"负载 {workload!r} 没有登记 prompt 全长 ⇒ 取消「grow 到不切分死区」这项检查，"
            f"但死区依然存在。请在 _PROMPT_LEN_FOR_DEADZONE 里补一行。"
        )
    elif hi > prompt_len:
        problems.append(
            f"chunk_size_max({hi}) > {workload} 的 prompt_len({prompt_len})："
            f"超出部分'grow 了但不切分'，轨迹会显示动作而实际无效果"
            f"（与旧的 cooldown_steps 同类：死参数）"
        )
    n_up = (hi - init) // step
    n_dn = (init - lo) // step
    if n_up == 0 or n_dn == 0:
        problems.append(
            f"步长 {step} 下可动档数：放大 {n_up} 档 / 收缩 {n_dn} 档，有一向为 0"
        )
    return problems


#: ``phase1_anchor`` 的 prompt 全长。超出它 chunk 再大也等于"整段一次算完"。
#: 硬编码在这里是刻意的：这份常量与 ``benchmark/workloads.py`` 的负载定义
#: 是一处**需要人工同步的耦合**，藏在代码里比藏在配置里更容易被 review 到。
#:
#: 2026-09-25 补 ``load_ramp``：第⑥步把它作为**负载轨迹扫描**的臂，
#: 而守卫 `_assert_gate_separation` 会在没登记时**静默取消**死区检查
#: （只留一句"请在表里补一行"）——补上它，检查才真的跑。
#: 取值 2048 与 ``workloads.py`` 里 ``prompt_len_min == prompt_len_max == 2048`` 一致。
_PROMPT_LEN_FOR_DEADZONE: dict[str, int] = {WORKLOAD: 2048, "load_ramp": 2048}


#: S4000 fp16 / Qwen2.5-7B 上**实测的逐请求 TPOT 可达区间**（毫秒）。
#:
#: 来源：``results/raw/P1S3B-*`` + ``results/raw/P2-*``，128 个成功请求，
#: 2026-09-24 由 ``scripts/calibrate_slo.py`` 统计（min=61.7、max=168.0）。
#: 手工同步是刻意的：这份常量依赖**实测结果**，不该让生成器去 import 产物。
#: 换硬件/换模型/换负载后必须重采，否则这道门会在错误的区间上放行。
_SLO_TPOT_REACHABLE_MS: tuple[float, float] = (61.7, 168.0)


def _assert_slo_reachable(cfg: dict) -> list[str]:
    """检查 SLO 判据是否落在**实测可达区间**内（P3 的守卫）。

    为什么必须查：``run.slo`` 不参与任何调度决策，所以一个不可达的阈值
    **不会**让 run 失败、不会让产物缺字段、标签照样全绿——它只让
    ``slo_satisfaction_rate`` / ``slo_goodput_*`` 三列恒为 0。这在表里与
    "所有配置都很差"长得一模一样，属于本项目定义的**静默失效**。
    P1S3/P1S3B 的 36+48 份产物就是这么来的。

    两个方向都要查：
    * 低于下界 ⇒ 恒不可达 ⇒ 恒 0；
    * 高于上界 ⇒ 恒可达 ⇒ 恒 1；
    两种都**不携带信息**。判据要能分辨，就必须落在区间内部。

    与"目标可达"那条的关系：``target_tpot_ms`` 的可达性只能在**重跑前**用实录信号
    开环复放把关（``scripts/check_step3b_reachability.py``）；SLO 的标定/把关同理，
    由 ``scripts/calibrate_slo.py`` 负责。本函数是静态兜底，防的是"把区间外的值
    写进新 profile"这类低级复发。
    """
    slo = cfg.get("run", {}).get("slo", {})
    tpot = slo.get("tpot_ms")
    lo, hi = _SLO_TPOT_REACHABLE_MS
    if tpot is None:
        return ["run.slo.tpot_ms 为 None ⇒ 无法判定判据是否可达（不可测 ≠ 达标）"]
    if tpot < lo:
        return [
            f"run.slo.tpot_ms({tpot:g}) < 实测逐请求 TPOT 下界({lo:g})"
            f" ⇒ 恒不可达，达标率三列将恒为 0（判据无效，不是'达标率低'）"
        ]
    if tpot > hi:
        return [
            f"run.slo.tpot_ms({tpot:g}) > 实测逐请求 TPOT 上界({hi:g})"
            f" ⇒ 恒可达，达标率三列将恒为 1（同样不携带信息）"
        ]
    return []


#: ``phase1_anchor`` 各并发档位上**实测的控制器工作点**：``controller_trace`` 里
#: 控制器**实际读到**的 ``tpot_p90_ms`` 的**中位数**（毫秒）。
#:
#: 来源：``results/raw/P1S3B-*`` + ``results/raw/P2-*`` 的 adaptive 臂首轮，
#: 2026-09-24 由 ``scripts/calibrate_controller_gate.py`` 统计。
#: 手工同步是刻意的（同 ``_SLO_TPOT_REACHABLE_MS``）：这份常量依赖实测结果，
#: 生成器不该去 import 产物。**换负载/换硬件必须重采**，否则这道门会在错误的
#: 区间上放行或误拦。
#:
#: ⚠️ **只取中位数，抹掉了分布宽度**（同一次统计给出 min/max：
#: c1 61.8–65.0、c2 81.6–291.6、**c4 97.8–571.7**、c8 116.2–415.8）。
#: 分布宽正是"绝对阈值分不开"的结构性障碍的形状，但本守卫的判据读不到它——
#: 见 :data:`_GATE_MIN_SEPARATION_FRACTION`。**不要把这份表当成分布的唯一描述。**
_MEASURED_WORKING_TPOT_P90_MS: dict[int, float] = {1: 62.0, 2: 81.9, 4: 98.7, 8: 142.1}

#: 门限与工作点必须离开多远才算"分得开"，取占 ``target_tpot_ms`` 的比例。
#:
#: ⚠️ **这是粗代理（floor），与标定工具的口径不同，而且不可能等价。**
#: 2026-09-24 查清了这一点（此前只知"不等价"，不知差在哪）：
#:
#: * 本守卫判的是「**中位数**工作点到门限的距离」——回答"门限离**典型**工作点够远吗"；
#: * 标定工具判的是「**实测的每一个点**上是决断的吗、离翻转多远」——回答
#:   "门限落在**实测分布**内部了吗"。
#:
#: 两者**互相都不包含对方**，实测四个候选可见分歧方向相反：
#:
#: ============ ====================== ==========================================
#: (target, h)  本守卫                 标定工具（修正后）
#: ============ ====================== ==========================================
#: 110 / 0.1    **拦下**（c4 距 0.3）   抛硬币 2 格（c2 1.0 ms、c4 **0.0 ms**）
#: 110 / 0.08   **拦下**（c4 距 2.5）   全格决断，但 c4 最近点仅 **0.6 ms**
#: 120 / 0.1    **放行**               全格决断，但 c8 最近点仅 **1.1 ms**（工具看到、守卫看不到）
#: 120 / 0.08   **放行**               全格决断，最近点 3.5–29.0 ms
#: ============ ====================== ==========================================
#:
#: ⇒ **不能靠"换个统计量"把两者统一**：用"最近点 + 同一个 margin"去替换中位数，
#: 会把 **120/0.08 也拦下**（c8 收缩侧最近点 3.5 ms < 6.0 ms），即**回溯否掉已
#: 入库的 P1S3D**。正确的分工是：本守卫 = **生成期的静态下限**（不解析产物、几秒可跑），
#: 标定工具 = **跑之前用产物做的权威判定**（要求每格 ``verdict`` 不是"抛硬币"；
#: 它需要 artifacts，所以属于**前置门**而不是生成器）。两处口径的差异要**一直写明**，
#: 不要试图消掉。登记待办：把"verdict 全为决断"接成一道真正的前置门。
#:
#: 取 0.05 的依据是实测：门限贴着工作点时工作点的**中位数**就已经贴上去了
#: （P1S3B 的 c4 中位数 98.7 距放大门限 99.0 只有 **0.3 ms**），于是整条爬坡
#: 近似抛硬币（三次重复 TTFT = 481.4 / 481.6 / 652.1）。中位数距离取 5%，
#: 恰好把该情形拦下、把 120/0.08（c4 中位数距离 11.7 ms > 6.0 ms）放行。
_GATE_MIN_SEPARATION_FRACTION = 0.05


def _assert_gate_separation(cfg: dict) -> list[str]:
    """检查控制器门限与**实测工作点**是否分得开（P3 的第二道门）。

    为什么必须查：门限离工作点太近时，控制器**不会报错、也不会不动**——它会
    **随机地动**。P1S3B 的 ``adaptive_c4`` 就是这样：放大门限 = ``110×0.9`` = 99.0，
    而 c4 的 ``tpot_p90`` 工作点中位数是 98.7 ⇒ 门限落在工作点上。

    两个方向都要查（判据见 :func:`_assert_slo_reachable` 的同类理由）：

    * 某个工作点落在**任一**门限的邻域内 ⇒ 该格的行为跨重复不可复现
      （后果不止是"抖一下"：P1S3B 里它让 TTFT 从 481 跳到 652，**+43%**）；
    * **全部**工作点都在收缩门限之上 ⇒ 放大分支永不触发，退化成**单向棘轮**
      （P1S3 的 `target_tpot_ms=50` 就是这一类）。

    ⚠️ **本判据是粗代理，见 :data:`_GATE_MIN_SEPARATION_FRACTION` 的说明**：
    它只比**中位数**与门限的距离，**看不见工作点分布有多宽** ⇒ 通过 ≠ 充分分离。
    权威判定是 ``scripts/calibrate_controller_gate.py`` 的 ``verdict``
    （每格"决断 / 抛硬币 / 不可达"）＋"离门限最近的**点**相距多远"。
    两者**不可能等价** —— 分工不同：这里是**生成期**的下限（不解析产物、几秒可跑），
    那边是**跑之前**用产物做的判定（需要 artifacts）。分歧的实测例子
    （四个候选各自谁放行谁拦下）见该常量的注释。

    读的是**最终产物**的生效值，不是 ``Profile`` 里可能为 ``None`` 的字段——
    理由同 :func:`_assert_controller_movable`（那一次守卫对**恰恰有缺陷的那份**恒真）。
    """
    ctl = cfg.get("controller", {})
    target = ctl.get("target_tpot_ms")
    hyst = ctl.get("hysteresis")
    if target is None or hyst is None:
        return [
            "controller.target_tpot_ms / hysteresis 为 None"
            " ⇒ 无法判定门限是否与工作点分得开"
        ]
    lower = target * (1.0 - hyst)   # 放大要求 tpot_p90 < 此值
    upper = target * (1.0 + hyst)   # 收缩要求 tpot_p90 > 此值
    margin = target * _GATE_MIN_SEPARATION_FRACTION

    problems: list[str] = []
    for conc, w in sorted(_MEASURED_WORKING_TPOT_P90_MS.items()):
        for gname, gate in (("放大门限", lower), ("收缩门限", upper)):
            if abs(w - gate) <= margin:
                problems.append(
                    f"c{conc} 的实测工作点 tpot_p90={w:g} 距{gname}={gate:.1f} 仅 "
                    f"{abs(w - gate):.1f} ms（须 > {margin:.1f}）⇒ 门限落在工作点"
                    f"分布内部，该格控制器行为跨重复不可复现"
                )
    if all(w > upper for w in _MEASURED_WORKING_TPOT_P90_MS.values()):
        problems.append(
            f"全部实测工作点都高于收缩门限 {upper:.1f} ⇒ 控制器只会收缩、"
            f"放大分支永不触发（退化棘轮）"
        )
    return problems


_HEADER = """\
# ⚠️ 本文件由 scripts/gen_phase1_step3_configs.py **生成**，不要手工编辑。
# 改这里请改生成器（或它的模板 configs/mthreads/s4000_fp16.yaml）后重新生成，
# 否则下一次生成会把手工改动静默覆盖掉。
#
# 权威说明（逐字段的"为什么"）保留在模板 configs/mthreads/s4000_fp16.yaml 里，
# 包括两条容易踩的警告：cache.block_size 是**模拟粒度**而非设备参数；
# 以及不要预设推理引擎。生成配置只承载"这一格与别的格差在哪"。
#
# 臂：{arm}    并发：{conc}
# 与相邻臂的差异（单变量自检已在生成时通过）：
#   static        -> static_chunk  只差 chunk.enabled
#   static_chunk  -> adaptive      只差 controller.enabled
"""


def build(profile: Profile) -> dict[str, dict]:
    """产出 {配置名（不含扩展名） -> 配置字典}。"""
    if not _TEMPLATE.exists():
        raise SystemExit(f"模板不存在：{_TEMPLATE}")
    base = yaml.safe_load(_TEMPLATE.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise SystemExit(f"模板根节点不是映射：{_TEMPLATE}")

    common = _common_overrides(profile)
    out: dict[str, dict] = {}
    for arm, overrides in ARMS.items():
        for c in CONCURRENCIES:
            # sort_keys=False：保住模板里的字段顺序，使生成物与模板对读时
            # 不必在两套顺序之间来回换算。
            cfg = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))  # 深拷贝
            for k, v in common.items():
                _set_path(cfg, k, v)
            for k, v in overrides.items():
                _set_path(cfg, k, v)
            _set_path(cfg, "workload.concurrency", c)
            # notes 必须能一眼看出这是哪一格：产物聚合时靠它对齐矩阵。
            cfg["run"]["notes"] = (
                f"{profile.notes_tag} arm={arm} concurrency={c} "
                f"(chunk.enabled={_get_path(cfg, 'chunk.enabled')}, "
                f"controller.enabled={_get_path(cfg, 'controller.enabled')})"
            )
            # 注意：表头**不进** cfg。它是注释，不是键——写进 cfg 会被
            # 单变量自检当成一处差异（表头里含臂名与并发），
            # 也会让"未知字段加载即抛错"的约定多一个例外。
            # 表头在写盘时由文件名反推（见 main）。
            out[f"{arm}_c{c}"] = cfg
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 PHASE 1 第③步矩阵配置")
    ap.add_argument(
        "--profile", default="p1s3d", choices=sorted(PROFILES),
        help="参数集（默认 p1s3d，即当前约定：SLO 判据 100 ms、门限 120/0.08）",
    )
    ap.add_argument("--check", action="store_true", help="只校验，不写盘")
    ap.add_argument(
        "--out-dir", default=None,
        help="输出目录（默认 configs/mthreads/<该 profile 的目录名>）",
    )
    ap.add_argument(
        "--allow-known-defects", action="store_true",
        help="放行被声明的已知缺陷（仅用于逐字复现已执行的旧配置）",
    )
    args = ap.parse_args(argv)

    profile = PROFILES[args.profile]
    configs = build(profile)

    # ---- 单变量性自检（本脚本的主要价值）----
    # 并发轴内部：同一臂的不同并发点，差异必须**只有** workload.concurrency
    # 和 run.notes。前者是被扫描的轴，后者是标签。多出任何一项都说明
    # 生成逻辑把别的东西也改了。
    problems: list[str] = []
    for arm in ARMS:
        for a, b in zip(CONCURRENCIES, CONCURRENCIES[1:]):
            diffs = _assert_single_variable(
                configs, f"{arm}_c{a}", f"{arm}_c{b}"
            )
            unexpected = [d for d in diffs if d not in ("workload.concurrency", "run.notes")]
            if unexpected:
                problems.append(
                    f"{arm} c={a} vs c={b}：除并发外还差 {unexpected}"
                )

    # 臂间：static->static_chunk 只差 chunk.enabled；static_chunk->adaptive 只差 controller.enabled。
    _ARM_EDGE = {
        ("static", "static_chunk"): "chunk.enabled",
        ("static_chunk", "adaptive"): "controller.enabled",
    }
    for (a, b), expect in _ARM_EDGE.items():
        diffs = _assert_single_variable(configs, f"{a}_c1", f"{b}_c1")
        unexpected = [d for d in diffs if d not in (expect, "run.notes")]
        if unexpected:
            problems.append(f"{a} vs {b}：预期只差 {expect}，实际还差 {unexpected}")

    # ---- 被扫描的维度是否真的可动 / SLO 判据是否可达 / 门限是否离开工作点 ----
    # 三次真实失效各对应一道门：第③步第一次执行（chunk 维度死锁 + 目标不可达）、
    # P3 前段（判据不可达 ⇒ 达标率恒 0）、P3 后段（门限贴着工作点 ⇒ 行为抛硬币）。
    # 三者都是静态可判的结构性问题——运行时不会报错，只有结论是错的。
    declared = (
        _assert_controller_movable(configs["adaptive_c1"])
        + _assert_slo_reachable(configs["adaptive_c1"])
        + _assert_gate_separation(configs["adaptive_c1"])
    )
    if declared:
        if profile.known_defects:
            # 缺陷已被声明：只在加了 --allow-known-defects 时放行，且必须打在 stderr。
            print(f"⚠️ profile {profile.name!r} 带**已声明的**已知缺陷：", file=sys.stderr)
            for d in profile.known_defects:
                print(f"   · {d}", file=sys.stderr)
            for m in declared:
                print(f"   · （检查命中）{m}", file=sys.stderr)
            if not args.allow_known_defects:
                print(
                    "拒绝生成：该 profile 存在'被测维度不可动'、'SLO 判据不可达'"
                    "或'控制门限贴着工作点'，生成的矩阵无法回答它声称要回答的问题。"
                    "确实只为复现旧证据，请加 --allow-known-defects。",
                    file=sys.stderr,
                )
                return 1
            print("（--allow-known-defects 已给出，继续）", file=sys.stderr)
        else:
            problems.extend(declared)

    for line in problems:
        print(f"自检未通过：{line}", file=sys.stderr)
    if problems:
        return 1
    print(
        "单变量性自检通过：并发轴内只差并发；static->static_chunk 只差 chunk.enabled；"
        "static_chunk->adaptive 只差 controller.enabled。"
    )
    if args.out_dir is None:
        if not profile.known_defects:
            print(f"控制器可动性自检通过：{_movable_summary(profile)}")

    if args.check:
        print(f"（--check，未写盘；共 {len(configs)} 个配置）")
        return 0

    out_dir = Path(args.out_dir) if args.out_dir else _DEFAULT_OUT_ROOT / profile.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, cfg in sorted(configs.items()):
        arm, _, conc = stem.rpartition("_c")
        header = _HEADER.format(arm=arm, conc=conc)
        text = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)
        # 证据/配置一律 LF（Windows 默认会写 CRLF，跨机器比对会出假 diff）
        with open(out_dir / f"{stem}.yaml", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(header + text)
    print(f"已写出 {len(configs)} 个配置 -> {out_dir}")
    return 0


def _movable_summary(profile: Profile) -> str:
    init = profile.chunk_size_init
    lo = profile.chunk_size_min
    hi = profile.chunk_size_max
    slo_lo, slo_hi = _SLO_TPOT_REACHABLE_MS
    # 门限必须从**最终产物**算，Profile 里可能是 None（沿用基线）。
    cfg = build(profile)["adaptive_c1"]["controller"]
    t, h = cfg["target_tpot_ms"], cfg["hysteresis"]
    return (
        f"chunk_size {lo}..{hi}，初值 {init}（可放大 {hi - init}、可收缩 {init - lo}）"
        f"；target_tpot_ms={profile.target_tpot_ms}、target_ttft_ms={profile.target_ttft_ms}"
        f" 需用实录信号开环复放确认可达"
        f"；门限 放大<{t * (1 - h):.1f} / 收缩>{t * (1 + h):.1f}"
        f"（实测工作点 c1 62.0 / c2 81.9 / c4 98.7 / c8 142.1）"
        f"；run.slo.tpot_ms={profile.slo_tpot_ms:g}"
        f"（实测可达区间 [{slo_lo:g}, {slo_hi:g}]）"
    )


if __name__ == "__main__":
    raise SystemExit(main())
