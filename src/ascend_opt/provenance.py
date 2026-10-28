"""数据来源标签与可复现性元数据。

本模块是规范第 8 节（可复现性）与第 10 节（数据完整性）的代码化落点。

存在的理由
----------
规范第 10 节是**绝对要求**：不得编造实验结果；无昇腾硬件时结果必须标记为
``NOT_MEASURED`` 或 ``HARDWARE_REQUIRED``；合成数据只能用于单元测试且必须
标记 ``TEST_DATA``，且永远不得出现在论文结果表中。

把这条要求写成散文是无效的——它靠人自觉。所以这里把它变成**类型**：
每个 run 的汇总、每次聚合、每张表都必须携带一个 :class:`DataLabel`，
而 :attr:`DataLabel.eligible_for_conclusion` 是判断"能不能进结论"的唯一入口。
聚合与导出按标签过滤，而不是靠作者记得排除。

与之配套的是 :class:`ReproducibilityRecord`：规范第 8 节列出的 21 个字段。
关键设计是**缺失必须显形**——字段取不到值时写入 ``NOT_MEASURED``
（硬件本身不存在时写 ``HARDWARE_REQUIRED``），而不是留空或填 0。
留空会被读成"这项不重要"，填 0 会被读成"测出来是 0"，两者都是无声的错误。
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import time as _time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

# 缺失标记。刻意用大写下划线风格，使其在表格里一眼可见。
NOT_MEASURED = "NOT_MEASURED"
HARDWARE_REQUIRED = "HARDWARE_REQUIRED"


class DataLabel(str, Enum):
    """一次 run（或一个指标）的数据来源等级。

    继承 ``str`` 是为了让它能直接写进 JSON/CSV 而不需要额外的转换层。
    """

    #: 真实硬件上实测得到。唯一可以进入结论的标签。
    MEASURED = "MEASURED"
    #: 本应能测到但本次没测（例如未采集 token 间隔明细）。
    NOT_MEASURED = "NOT_MEASURED"
    #: 依赖昇腾硬件，当前环境不具备，无法测量。
    HARDWARE_REQUIRED = "HARDWARE_REQUIRED"
    #: 合成/模拟数据（单元测试、--dry-run 链路验证）。永远不进论文表格。
    TEST_DATA = "TEST_DATA"

    @property
    def eligible_for_conclusion(self) -> bool:
        """能否作为结论依据。只有真实实测可以。"""
        return self is DataLabel.MEASURED

    @property
    def is_synthetic(self) -> bool:
        """是否是合成了数字标签（而非"没测"）。"""
        return self is DataLabel.TEST_DATA


class HardwarePlatform(str, Enum):
    """本次实验面向的**硬件平台**。

    为什么单列一个维度：标签判定原先把"权威远程环境"写死成昇腾，于是研究主体
    一旦换到别的平台（例如摩尔线程 MTT S4000），唯一"可能拿到 MEASURED"的声明
    值就只剩 ``remote_ascend``——一份 S4000 的数据被迫在溯源元数据里自称跑在昇腾上。
    这不是措辞问题，是**元数据说谎**：后续按平台筛选、按平台复现的人会被带到
    错误的机器上，而且没有任何一处会报错。

    所以把两件事分开，并且**刻意不让它们互相推导**：

    * :class:`HardwarePlatform` 说的是**硬件是什么**——事实，可被指纹校验。
    * :class:`ExecutionEnvironment` 说的是**推理在哪执行**——声明，决定标签。

    声明 ``ascend`` 不代表推理就跑在远程主机上。把这条耦合起来，会让"本机就有一张
    昇腾卡的开发机"自动变成 MEASURED，正是规范第 10 节要拦的事。
    """

    ASCEND = "ascend"
    MTHREADS = "mthreads"
    #: 未声明。历史配置都不写这个字段，必须能落回这里而不是报错。
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: Any) -> "HardwarePlatform":
        """转成枚举。未知取值**抛错**，理由同 :meth:`ExecutionEnvironment.parse`：
        静默降级会让人以为声明生效了，而判定其实还走老逻辑。"""
        if value is None or value == "":
            return cls.UNKNOWN
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text:
                return member
        allowed = ", ".join(m.value for m in cls)
        raise ValueError(f"未知的硬件平台声明 {value!r}；允许的取值：{allowed}")

    @property
    def display_name(self) -> str:
        """人话名字，只用在给人看的解释串里，不作为机读字段。"""
        return _PLATFORM_DISPLAY_NAME.get(self.value, self.value)


#: 远程执行声明 <-> 硬件平台的对应关系。**唯一来源**：判定用的集合、报错时给出的
#: 修法、以及自洽性校验全部从这里派生，避免三处手写清单各自漂移。
#:
#: ``local`` / ``unknown`` 不在表内——它们不声称任何远程平台，因此"查不到对应项"
#: 就等于"这条声明不蕴含平台、无从矛盾"，而不是"漏配了"。
REMOTE_PLATFORM_PAIRS: tuple[tuple[str, str], ...] = (
    ("remote_ascend", "ascend"),
    ("remote_mthreads", "mthreads"),
)

#: 声明为"远程执行"的全部取值。判定逻辑只读这个集合，不再逐个比对枚举成员，
#: 于是新增平台只需在 :data:`REMOTE_PLATFORM_PAIRS` 加一行。
#:
#: 放在模块级而不是枚举内部：``str`` 混合枚举里的非字符串类属性要么被当成一个新的
#: 成员，要么被 ``str()`` 强转成一个看不出错的字符串值——两种都是静默的坑。
REMOTE_EXECUTION_VALUES: frozenset[str] = frozenset(
    env for env, _ in REMOTE_PLATFORM_PAIRS
)

_REMOTE_ENV_PLATFORM: dict[str, str] = dict(REMOTE_PLATFORM_PAIRS)
_PLATFORM_TO_REMOTE_ENV: dict[str, str] = {
    platform: env for env, platform in REMOTE_PLATFORM_PAIRS
}

_PLATFORM_DISPLAY_NAME: dict[str, str] = {
    "ascend": "昇腾 NPU",
    "mthreads": "摩尔线程 MTT",
    "unknown": "未声明平台",
}


class ExecutionEnvironment(str, Enum):
    """推理**实际执行**在哪种环境里。

    为什么需要这个维度：权威推理环境是**远程主机**（规范原先把这条写死为昇腾，
    但判定本身与平台无关），本机不是。
    于是"硬件在不在"这句话必须问清楚是"哪台机器的硬件"——本机没有 NPU，
    并不代表这次 run 的数字不是远程实测（推理可能跑在远程主机上，只是由本机
    驱动）。反过来，本机自己跑出来的数字，无论本机有没有卡，都不是远程实测。

    默认 ``UNKNOWN``：不声明就按老逻辑判（看本机指纹），行为与引入本枚举前一致。
    只有显式声明才改变判定，避免"加了个字段就把历史数据判成别的标签"。
    """

    #: 推理在权威的远程昇腾主机上执行。
    REMOTE_ASCEND = "remote_ascend"
    #: 推理在权威的远程摩尔线程（MTT）主机上执行。
    #: 单独一个取值而不是复用 ``remote_ascend``：复用会让 meta 里出现一句
    #: 与事实相反的话（S4000 的数据自称跑在昇腾上），比"没有这个声明"更糟。
    REMOTE_MTHREADS = "remote_mthreads"
    #: 推理在本机执行，明确不是权威环境。规范禁止其进入结论。
    LOCAL = "local"
    #: 未声明。按本机硬件指纹回落到原有判定。
    UNKNOWN = "unknown"

    @property
    def is_remote(self) -> bool | None:
        """推理是否在**权威远程主机**上执行。三态，与全项目的缺失约定一致。

        * ``True`` —— 声明值在 :data:`REMOTE_EXECUTION_VALUES` 里（任选远程平台）。
        * ``False`` —— 明确声明本机执行（``LOCAL``）。
        * ``None`` —— 未声明（``UNKNOWN``）。

        ``None`` **不等于 False**，调用方必须用 ``is False`` 而不是 ``not is_remote``：
        前者只命中"明确本机"，后者会把"没声明"一并扫进来，于是"环境指纹缺失"
        这条本该由指纹判定的路径被提前判成 HARDWARE_REQUIRED——
        真实测数据会被聚合成不可用，而且不报错。
        """
        if self is ExecutionEnvironment.UNKNOWN:
            return None
        return self.value in REMOTE_EXECUTION_VALUES

    @classmethod
    def parse(cls, value: Any) -> "ExecutionEnvironment":
        """把配置里的字符串转成枚举。未知取值**抛错**，不静默降级。

        静默把 ``"remote_ascent"``（拼错）当成 UNKNOWN 会让人以为声明生效了，
        而实际上判定还走的是老逻辑——这类"以为设了其实没设"的错误很难发现。
        """
        if value is None or value == "":
            return cls.UNKNOWN
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text:
                return member
        allowed = ", ".join(m.value for m in cls)
        raise ValueError(
            f"未知的执行环境声明 {value!r}；允许的取值：{allowed}"
        )


def resolve_run_label(
    *,
    dry_run: bool,
    hardware_available: bool | None,
    backend: str = "",
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.UNKNOWN,
) -> DataLabel:
    """判定一次 run 的整体标签。

    判定顺序刻意如此——先看数据是怎么产生的，再看在哪台机器上执行，
    最后才看硬件在不在：

    1. ``dry_run`` -> ``TEST_DATA``。一阶模型产出的数字是合成的，
       无论机器上有没有 NPU 都不改变这一点。
    2. 后端自称模拟 -> ``TEST_DATA``。
    3. 声明为 ``LOCAL`` -> ``HARDWARE_REQUIRED``。规范禁止本机结果被表述为
       远程硬件实测；没有这条，本机真机路径下"有卡但没声明"的历史行为就会漂移。
    4. 硬件不可用 -> ``HARDWARE_REQUIRED``。
    5. 硬件可用性**未知** -> ``NOT_MEASURED``。
       未知不等于可用：没查过环境就自称"已实测"，正是规范第 10 节要拦的事。
    6. 其余 -> ``MEASURED``。

    注意第 3、4、5 步里的 ``hardware_available`` 指的是**执行推理的那台机器**
    的硬件可用性。声明为任一远程取值（见 :data:`REMOTE_EXECUTION_VALUES`）时，
    它应当来自远程主机的指纹，而不是本机的指纹（本机指纹见
    :attr:`ExecutionEnvironment.LOCAL`）。
    """
    if dry_run:
        return DataLabel.TEST_DATA
    if backend and backend.startswith("sim"):
        # 防御性检查：后端自称是模拟的，就不许冒充实测。
        return DataLabel.TEST_DATA
    if execution_environment.is_remote is False:
        # 明确声明本机执行。规范写得很直白：绝不把本机结果表述为远程硬件实测。
        # 这里必须是 `is False` 而不是 `not is_remote`：后者会把"未声明"（None）
        # 一并扫进来，于是没声明执行环境的 run 被提前判成 HARDWARE_REQUIRED，
        # 绕过下面基于指纹的判定——一次"忘了写声明"就变成"测不了"。
        return DataLabel.HARDWARE_REQUIRED
    if hardware_available is False:
        return DataLabel.HARDWARE_REQUIRED
    if hardware_available is None:
        # 环境指纹缺失。这不是硬件的错，是流程没走完；但结果同样不能当实测用。
        return DataLabel.NOT_MEASURED
    return DataLabel.MEASURED


def validate_platform_consistency(
    execution_environment: ExecutionEnvironment,
    hardware_platform: HardwarePlatform,
) -> None:
    """校验"推理在哪执行"与"面向哪个平台"两个声明不自相矛盾；矛盾时抛 ``ValueError``。

    这是**声明之间的自洽性**检查，不是对真机的校验——真机确认要靠环境指纹，
    属于另一条路径（见 :func:`build_reproducibility_record` 与 ``inspect_env.sh``）。

    规则只有一条，而且是单向的：远程执行声明蕴含一个平台，声明的平台必须与之相符。
    其余组合一律放行，因为"没声明"推不出"说错了"：

    * 任一方为 ``unknown`` —— 放行。
    * ``local`` + 任意平台 —— 放行。"本机就是一台整机、直接在上面跑"是真实情形，
      只是按规范其结果不得进结论；能这么写，人才不会为了让流程走通而谎报平台。

    为什么必须报错而不是取其一：这两个字段一个落进 ``execution_environment``、
    一个落进 ``hardware_platform``，都会写进 meta 并被后续按平台筛选。
    让它们矛盾地并存，等于产出一份**自相矛盾的可复现性记录**，
    而下游只会读到其中一半，永远发现不了。
    """
    implied = _REMOTE_ENV_PLATFORM.get(execution_environment.value)
    if implied is None:
        # local / unknown：不声称远程平台，无从矛盾。
        return
    if hardware_platform is HardwarePlatform.UNKNOWN:
        # 未声明平台。与"声明了错的平台"不同，不报错——历史配置本来就不写这个字段。
        return
    if implied == hardware_platform.value:
        return

    suggestion = _PLATFORM_TO_REMOTE_ENV.get(hardware_platform.value)
    fix = (
        f"若推理确实跑在{hardware_platform.display_name}上，"
        f"请把 execution_environment 改成 {suggestion!r}。"
        if suggestion
        else ""
    )
    raise ValueError(
        f"execution_environment={execution_environment.value!r} 蕴含硬件平台 "
        f"{implied!r}，但 hardware_platform 写的是 {hardware_platform.value!r}，"
        f"两者矛盾。{fix}"
    )


def platform_evidence_conflict(
    declared: HardwarePlatform, detected: HardwarePlatform
) -> str | None:
    """"配置**声明**的平台"与"环境指纹**检出**的平台"冲突时返回说明；否则 ``None``。

    只在**双方都已知**且不一致时才算冲突。任一方 ``unknown`` 一律放行：
    没声明、或指纹是旧格式没写平台，都推不出"说错了"。这条边界很重要——
    把"没证据"当成"证据矛盾"会让所有历史指纹与未声明平台的配置一起被拒。

    与 :func:`validate_platform_consistency` 的分工：
    那个查的是**两个声明之间**是否自洽（只能保证不矛盾，证明不了任何事实）；
    这个查的是"说的"与"**查到的**"是否一致——后者才是能证明的那种。
    只做前者会留下一个空子：声明可以很自洽，同时与真机完全不符。
    """
    if declared is HardwarePlatform.UNKNOWN or detected is HardwarePlatform.UNKNOWN:
        return None
    if declared is detected:
        return None
    return (
        f"配置声明 hardware_platform={declared.value!r}，"
        f"但环境指纹检出的平台是 {detected.value!r}，两者矛盾。"
    )


def validate_platform_evidence(
    declared: HardwarePlatform,
    detected: HardwarePlatform,
    *,
    source: str = "",
) -> None:
    """声明与**指纹证据**冲突时抛 ``ValueError``；无从判断时放行。

    报错刻意给出**两条**可能的原因与各自的修法：这类冲突要么是指纹采错了机器，
    要么是声明写错了平台，读的人不该自己去猜是哪一种。
    """
    reason = platform_evidence_conflict(declared, detected)
    if reason is None:
        return
    where = f"（指纹来源：{source}）" if source else ""
    raise ValueError(
        f"{reason}{where}两种可能，按实际情况修："
        f"① 指纹采错了机器 —— 到{declared.display_name} 主机上重跑 "
        f"scripts/inspect_env.sh，并在配置里指明该指纹；"
        f"② 声明写错了平台 —— 把 hardware_platform 改成 {detected.value!r}，"
        f"并同步 execution_environment（两者必须自洽）。"
    )


# --------------------------------------------------------------------------- #
# 可复现性元数据（规范第 8 节）
# --------------------------------------------------------------------------- #

#: 规范第 8 节要求的全部字段，顺序照抄规范，便于逐条核对。
REQUIRED_REPRO_FIELDS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "git_commit",
    "hostname_or_machine_id",
    "device_model",
    "device_count",
    "model",
    "model_revision",
    "precision",
    "quantization",
    "tensor_parallel_size",
    "input_length",
    "output_length",
    "concurrency",
    "batch_configuration",
    "max_batched_tokens",
    "chunk_size",
    "prefix_cache_enabled",
    "graph_mode",
    "random_seed",
    "warmup_count",
    "measurement_count",
)


@dataclass
class ReproducibilityRecord:
    """一次 run 的可复现性元数据。

    所有字段都允许为 ``None``——取不到就是取不到，不做任何猜测性填充。
    序列化时 :meth:`describe` 会把空值替换成显式的缺失标记。
    """

    run_id: str = ""
    timestamp: str = ""
    git_commit: str = ""
    hostname_or_machine_id: str = ""
    device_model: str = ""
    device_count: int | None = None
    model: str = ""
    model_revision: str = ""
    precision: str = ""
    quantization: str = ""
    tensor_parallel_size: int | None = None
    input_length: int | None = None
    output_length: int | None = None
    concurrency: int | None = None
    batch_configuration: str = ""
    max_batched_tokens: int | None = None
    chunk_size: int | None = None
    prefix_cache_enabled: bool | None = None
    graph_mode: str = ""
    random_seed: int | None = None
    warmup_count: int | None = None
    measurement_count: int | None = None
    #: 本次 run 的整体数据标签（不参与缺失判定，单独输出）。
    data_label: DataLabel = DataLabel.NOT_MEASURED
    #: 是否因硬件缺失而无法测量。决定缺失标记写成哪一项。
    hardware_required: bool = False
    #: 推理**实际执行**的环境。规范要求权威环境是远程主机。
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.UNKNOWN
    #: 本次实验面向的**硬件平台**。与执行环境分开记录，且不互相推导
    #: （理由见 :class:`HardwarePlatform` 的文档）。用于下游按平台筛选与分组。
    hardware_platform: HardwarePlatform = HardwarePlatform.UNKNOWN
    #: 产出这些数字的**执行后端**（``simulate`` / ``real``）。
    #:
    #: 它原先只作为入参参与标签判定（模拟后端不得自称实测），**没有任何落盘**。
    #: 于是"同一批 run 里混了模拟与真机"这件事在汇总表上看不出来——两者的
    #: 数字口径完全不同，混在一张表里求中位数就是一个谁都没测过的数。
    backend: str = ""

    # ---------------- 补充维度（刻意不并入 REQUIRED_REPRO_FIELDS）----------------
    #
    # 理由同 describe() 里那两处注释：REQUIRED_REPRO_FIELDS 是规范第 8 节**点名**的
    # 清单，往里加会被 missing_fields() 当成必需项，于是所有历史 run 一夜之间变成
    # "元数据不完整"，preflight 全红。这三个是补充说明，单独输出。

    #: ``git_commit`` 是从哪来的：``git`` / ``stamp`` / ``stamp_stale`` / ``none``。
    #:
    #: 必需。因为"远程源码树不是 git 仓库"是常态，而"用部署戳代替 git"必须能被看出来：
    #: 只看 `git_commit` 一列，无法区分"这台机器上真是这个 commit"与
    #: "部署那一刻是这个 commit，之后有没有改过不知道"。``stamp_stale`` 尤其重要——
    #: 它表示**有过戳但不可信**，与"从来没有戳"的修法完全不同。
    git_source: str = ""
    #: 部署时本地工作区是否有未提交改动。``None`` = 没查（不是"干净"）。
    git_dirty: bool | None = None
    #: 这棵树的内容摘要（由部署戳携带），用于**摘要级**的续跑身份比对。
    #:
    #: ★ 为什么不能只靠 ``git_commit``（2026-10-02）：本机工作区常年是脏的
    #: （本次实测 83 项未提交改动），此时 ``git_commit`` 只标出"分支点"，
    #: **不描述被测的那份代码**——两份 `commit` 相同、代码却不同的产物，在只比
    #: commit 的判据下**一模一样**。有了摘要才能问出"这份产物是不是当前这棵树跑出来的"。
    #:
    #: ``None`` = **不可用**（不是"空摘要"）。只在部署戳校验通过时才有值；
    #: 真 git 树上不给摘要（本机开发树的语义是 HEAD + 未提交改动）。
    #: 续跑判定必须把 ``None`` 当成**拒绝复用**，不能当成"跳过这一项比对"。
    tree_digest: str | None = None
    #: ``model_revision`` 是**解析出的 commit** 还是**请求的引用**。
    #:
    #: 三态，必须用 `is True` / `is False` 判定，**不要**写 `if not resolved`：
    #:   * ``True``  —— 值是加载到的权重 commit（HF 在 config 上留的 `_commit_hash`），
    #:                  可以把 run 钉到具体权重上；
    #:   * ``False`` —— 值只是一个引用（如 `main`），今天指向 A、明天可能指向 B，
    #:                  **不能**用来复现；
    #:   * ``None``  —— 没查（旧产物没有这个字段）。
    #: 把 ``None`` 当 ``False`` 会把"没查"报成"查了，是引用"，两者含义相反。
    model_revision_resolved: bool | None = None

    # ---------------- 缺失检查 ----------------

    def _is_missing(self, name: str) -> bool:
        value = getattr(self, name)
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    def missing_fields(self) -> list[str]:
        """列出所有取不到值的必需字段。空列表 = 元数据完整。"""
        return [name for name in REQUIRED_REPRO_FIELDS if self._is_missing(name)]

    @property
    def complete(self) -> bool:
        return not self.missing_fields()

    def _missing_marker(self) -> str:
        return HARDWARE_REQUIRED if self.hardware_required else NOT_MEASURED

    def describe(self) -> dict[str, Any]:
        """序列化。缺失字段写显式标记，不留空字符串。"""
        marker = self._missing_marker()
        out: dict[str, Any] = {}
        for name in REQUIRED_REPRO_FIELDS:
            value = getattr(self, name)
            if self._is_missing(name):
                out[name] = marker
            elif isinstance(value, bool):
                out[name] = value
            else:
                out[name] = value
        out["data_label"] = self.data_label.value
        out["execution_environment"] = self.execution_environment.value
        # 刻意**不**并入 REQUIRED_REPRO_FIELDS：那是规范第 8 节点名的字段清单，
        # 往里加会被 missing_fields() 当成必需项，于是所有历史 run 一夜之间
        # 变成"元数据不完整"，preflight 全红。这是新增的补充维度，单独输出。
        out["hardware_platform"] = self.hardware_platform.value
        # 同理单独输出。空值写成 ``unknown`` 而不是空串：空串在 CSV 里会被读成
        # "这一列没数据"，而它实际想说的是"这次 run 没记录后端"。
        out["backend"] = self.backend.strip() or "unknown"
        # 溯源来源三态输出。`git_source` 空值写 ``unknown`` 而不是空串——
        # 空串在 CSV 里会被读成"这一列没数据"，而它实际想说的是"这次 run 没记录来源"。
        out["git_source"] = self.git_source.strip() or "unknown"
        # git_dirty / model_revision_resolved 是**三态**：True / False / None(=没查)。
        # 原样写 JSON null，不折叠成 False——折叠会把"没查"报成"查了且为否"。
        out["git_dirty"] = self.git_dirty
        out["model_revision_resolved"] = self.model_revision_resolved
        # 摘要同样是"取不到就不写一个假的"。原样写 JSON null，不折叠成空串：
        # 空串在 CSV 里会被读成"这一列没数据"，而这里想说的是"本次 run 的
        # 部署戳没带摘要（例如本机开发树）"。续跑判据把 None 当**拒绝复用**。
        out["tree_digest"] = self.tree_digest
        out["_missing_fields"] = self.missing_fields()
        out["_hardware_required"] = self.hardware_required
        return out


# --------------------------------------------------------------------------- #
# 从环境与配置采集
# --------------------------------------------------------------------------- #

def detect_git_commit(root: str | Path | None = None) -> str:
    """取 ``root`` 这棵树的当前 commit。取不到时返回空串（由调用方转成缺失标记）。

    不吞掉异常去伪造一个哈希——一个假的 commit 会让整条溯源链失去意义。

    ``root is None`` 表示**没查**，直接返回空串，**绝不回落到进程当前目录**。
    这条是踩出来的：``subprocess.run(cwd=None)`` 会继承调用者的 cwd，于是
    "调用方没传源码树根"被静默变成"把进程恰好所在的那个仓库的 commit 记上去"。
    在开发机上，那往往就是**本机开发仓库**——于是一份声称"远端实测"的产物里
    出现了一个与远端毫无关系的 commit，且不报错。缺值可以看出来，错值看不出来。

    只想记"当前目录"的调用方请显式传 ``Path(".")``。

    **只管真 git 仓库**。远程推理机上的源码树通常不是 git 仓库（是远端拷贝），
    这时要用 :func:`detect_source_commit`，它会回落到部署戳。
    """
    if root is None:
        return ""
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# 部署戳：让"远程那棵不是 git 仓库的源码树"也能被钉到具体 commit
# --------------------------------------------------------------------------- #
#
# 为什么要有这一层
# ----------------
# 权威推理环境是**远程主机**，而远程的源码树是**拷贝**过去的，不带 `.git`
# ⇒ `git rev-parse HEAD` 必然失败 ⇒ `git_commit` 恒为 `NOT_MEASURED`。
# 后果不是"少了一列"，而是 72 份产物全都**指认不到源码版本**：
# 半年后没人能回答"这批数字是哪个版本的代码跑出来的"。
#
# 但戳有个独有的危险：它**会过期**。部署之后若有人改了远端源码而没重新投放戳，
# 戳就会给出一个**错的值**——比缺的值危险得多，因为它不报错、不留痕，
# 还会变成下游结论的依据，且看起来像一次成功的采集。
# 所以戳里必须带一份**源码树摘要**，读的时候重算比对；不符就当作取不到。

#: 部署脚本在源码树根投放的溯源戳文件名。
#: 刻意**不放在 `results/` 下**：那里是"原始产物、只读"的地盘，
#: 而这是一份描述"这棵源码树是谁"的元数据，不是测量产物。
SOURCE_STAMP_NAME = "SOURCE_PROVENANCE.json"

#: 参与摘要计算的目录（相对源码树根）。
#: 选取原则是**只包含会影响测量结果的树**：
#: 少列一个目录 ⇒ 改了那个目录戳也检测不出来（校验静默失效）；
#: 多列一个目录（例如 `results/`）⇒ 产物一写进去摘要就变，戳每次都判为过期。
SOURCE_DIGEST_SCOPE: tuple[str, ...] = ("src/ascend_opt", "benchmark", "scripts")


def iter_scope_files(
    root: str | Path | None, scope: tuple[str, ...] = SOURCE_DIGEST_SCOPE
) -> list[tuple[str, str]]:
    """列出作用域内的 ``(相对路径, 文件 sha256)``，按相对路径排序。

    刻意做成**公开**函数而不是 :func:`compute_tree_digest` 的内部细节：摘要不符时
    最需要的信息是"到底是哪些文件不同"，而差异定位要重新列一遍清单。让校验端
    复用这一份实现，就不存在"两份差不多的遍历"——那两份迟早会不一致，且不一致时
    表现为"摘要对不上，但逐文件比又都一样"这种最难查的形态。
    """
    if root is None:
        return []
    base = Path(root)
    if not base.is_dir():
        return []
    pairs: list[tuple[str, str]] = []
    for sub in scope:
        d = base / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            rel = p.relative_to(base).as_posix()
            h = hashlib.sha256()
            try:
                h.update(p.read_bytes())
            except OSError:
                # 读不到的文件**跳过**而不是记一个空摘要：记空摘要会让"权限问题"
                # 变成"内容变了"，下次比对时戳被误判为过期。
                continue
            pairs.append((rel, h.hexdigest()))
    return sorted(pairs)


def compute_tree_digest(
    root: str | Path | None, scope: tuple[str, ...] = SOURCE_DIGEST_SCOPE
) -> str | None:
    """对源码树算一个确定性摘要：``sha256`` over 排序后的 ``relpath\\0文件sha256``。

    用途**只有一个**——判断部署戳是否还描述着当前这棵树。
    取不到（根目录不存在、范围内一个文件都没有）返回 ``None``，**不返回空串**：
    空串会被下游当成"算出来了，且是个空值"，而这里是"根本没算成"。

    刻意排除 ``__pycache__`` 与 ``.pyc``：它们随解释器版本变化、与源码内容无关，
    留在摘要里会让摘要"看起来变了"而源码其实没变——那时戳会被误判为过期，
    于是又在"明明能记录"的情况下退化成 `NOT_MEASURED`。

    本机与远端必须对这种树算出**同一个值**，否则部署戳永远判为过期。因此这里
    只用标准库、且不依赖任何解释器版本相关的行为（实测：本机 3.13 与远端
    /usr/bin/python3 对同一棵树算出的摘要相同）。
    """
    pairs = iter_scope_files(root, scope)
    if not pairs:
        return None
    acc = hashlib.sha256()
    for rel, digest in pairs:
        acc.update(rel.encode("utf-8"))
        acc.update(b"\0")
        acc.update(digest.encode("ascii"))
        acc.update(b"\n")
    return acc.hexdigest()


def read_source_stamp(root: str | Path | None) -> dict[str, Any]:
    """读部署戳并**验证**它是否仍与当前源码树一致。

    返回 ``{"git_commit", "git_dirty", "validated", "reason", "deployed_at"}``。

    **`validated is False` 时 `git_commit` 一律为 `None`。** 拿不到可信值就用
    不可信值填空，等于把"没记录"变成"记录了但记错了"——这条线上只允许前者。
    调用方据 ``reason`` 显形，不要静默当作"没有"。
    """
    out: dict[str, Any] = {
        "git_commit": None, "git_dirty": None, "validated": None,
        "reason": "", "deployed_at": None, "deployed_by": None,
    }
    if root is None:
        out["reason"] = "未提供源码树根目录"
        return out
    path = Path(root) / SOURCE_STAMP_NAME
    if not path.is_file():
        out["reason"] = f"无部署戳 {SOURCE_STAMP_NAME}（源码树未经部署脚本投放）"
        return out
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["reason"] = f"部署戳无法解析：{exc}"
        return out
    if not isinstance(stamp, dict):
        out["reason"] = "部署戳不是 JSON 对象"
        return out

    # 这两项只是说明性信息（谁在什么时候投的），与"可不可信"无关，
    # 因此**在判定之前**就透出：戳后来被判过期时，人仍需要知道它是什么时候投的，
    # 否则"该重新投放"这条结论就没有落点。
    out["deployed_at"] = stamp.get("deployed_at")
    out["deployed_by"] = stamp.get("deployed_by")

    commit = stamp.get("git_commit")
    commit = commit.strip() if isinstance(commit, str) else ""
    want = stamp.get("tree_digest")
    if not commit:
        out["reason"] = "部署戳里没有 git_commit"
        return out
    if not isinstance(want, str) or not want:
        out["reason"] = "部署戳里没有 tree_digest，无法判断它是否过期"
        return out

    # 摘要作用域不一致时**必须单独报**：它和"源码被改过"都会表现为摘要不符，
    # 但修法完全不同（前者要重新投放戳，后者要先把树对齐）。混在一句话里，
    # 人会先按"源码被改过"去查，查不出来，最后把戳当成坏的不用了。
    stamp_scope = stamp.get("digest_scope")
    if isinstance(stamp_scope, list) and stamp_scope:
        if [str(s) for s in stamp_scope] != list(SOURCE_DIGEST_SCOPE):
            out["validated"] = False
            out["reason"] = (
                f"部署戳的摘要作用域与当前代码不一致（戳 {stamp_scope} vs 当前 "
                f"{list(SOURCE_DIGEST_SCOPE)}）：两次摘要覆盖的文件集合不同，"
                f"本来就不可比，已按取不到处理。改动 SOURCE_DIGEST_SCOPE 之后必须重新投放戳。"
            )
            return out

    got = compute_tree_digest(root)
    if got is None:
        out["reason"] = "无法计算当前源码树摘要，不能确认部署戳是否过期"
        return out
    if got != want:
        out["validated"] = False
        out["reason"] = (
            "部署戳与当前源码树**不一致**（摘要不符）：部署之后源码被改过，"
            "因此戳里的 commit 不再描述这棵树，已按取不到处理"
        )
        return out

    out["git_commit"] = commit
    out["git_dirty"] = bool(stamp.get("git_dirty")) if "git_dirty" in stamp else None
    # 把通过校验的摘要**透出**。★ 这一行是漏出来的：`want` 在校验时被反复读到，
    # 却从没进过返回值，于是 `detect_source_identity` 读 `stamp.get("tree_digest")`
    # 恒得 `None`——而 `validated` 仍是 `True`。症状是"戳明明有效，摘要却是不可用"，
    # 续跑判据于是对**所有**已有产物一律拒绝，看起来像判据坏了。
    # 校验的意义就是把 `want` 确认为"当前这棵树"的摘要，那就应该把它交出去。
    out["tree_digest"] = want
    out["validated"] = True
    out["reason"] = "部署戳已校验（摘要相符）"
    return out


def detect_source_commit(root: str | Path | None) -> tuple[str, str, bool | None]:
    """返回 ``(commit, source, dirty)``。``source`` ∈ ``{"git","stamp","stamp_stale","none"}``。

    优先级：**真 git 仓库 > 部署戳（且摘要相符）> 取不到**。
    真 git 优先，因为它是**当前**状态，而戳最多只说明部署那一刻。
    两者一致时没有差别；不一致时只可能出现在本机开发树上——那里本就没有戳。

    ``stamp_stale`` 与 ``none`` 必须分开：前者是"有戳但不可信"（该重新投放），
    后者是"从来没有戳"（该让部署脚本开始投放）。两者的修法不同。

    ``root is None`` ⇒ ``("", "none", None)``：没指定树就是**没查**，
    不拿进程当前目录顶替（理由见 :func:`detect_git_commit`）。
    """
    commit = detect_git_commit(root)
    if commit:
        # 这里**必须**顺手查一次脏不脏，不能返回 None（"没查"）。
        # 返回 None 看着无害，实际把唯一的警告信号关掉了：有未提交改动时，
        # HEAD 同样不描述被测的这份代码，而记录里却只有一个光秃秃的 commit。
        # 与写端的不对称是有意的——写端（部署戳）**拒绝**默认的脏树，因为戳是一句
        # "这份 commit 就是被部署的代码"的断言；记录只是陈述事实，如实标 True 即可。
        return commit, "git", detect_git_dirty(root)
    stamp = read_source_stamp(root)
    if stamp["validated"] is True:
        return str(stamp["git_commit"]), "stamp", stamp["git_dirty"]
    if stamp["validated"] is False:
        return "", "stamp_stale", None
    return "", "none", None


def detect_source_identity(root: str | Path | None) -> tuple[str, str, bool | None, str | None]:
    """返回 ``(commit, source, dirty, tree_digest)``——比 :func:`detect_source_commit` 多一个摘要。

    ★ 为什么需要**摘要**而不只是 commit（2026-10-02）：本机工作区常年是脏的
    （本次实测 83 项未提交改动），此时 ``git_commit`` 只标出"分支点"，
    **不描述被测的那份代码**——两份 `commit` 相同、代码却不同的产物，
    在只比 commit 的判据下**一模一样**。有了 ``tree_digest`` 才能问出
    "这份产物是不是**当前这棵树**跑出来的"。

    ``tree_digest`` 只在 ``source == "stamp"``（戳已校验通过）时给出；
    其余情形返回 ``None``——**这是"不可用"，不是"空摘要"**，调用方必须分开处理。
    ``git`` 分支不给摘要：真 git 树上摘要要另算，而本机开发树的语义是
    "HEAD + 未提交改动"，一个纯 HEAD 摘要反而不描述它。
    """
    commit, source, dirty = detect_source_commit(root)
    if source != "stamp":
        return commit, source, dirty, None
    stamp = read_source_stamp(root)
    digest = stamp.get("tree_digest")
    return commit, source, dirty, (str(digest) if digest else None)


# --------------------------------------------------------------------------- #
# 写入端：部署时投放戳
# --------------------------------------------------------------------------- #
#
# 写在读端**旁边**，共用同一个作用域常量与同一份遍历。写作两端分家是个具体的
# 坑：写端按"我推了哪些目录"算摘要、读端按"常量里的目录"算摘要，两者一旦不同，
# 戳在投放的那一刻就是过期的，而现场表现只是"git_commit 又是 NOT_MEASURED"。

#: 戳格式版本。字段增删时递增，读端据它判断能不能按当前规则解释。
SOURCE_STAMP_VERSION = 1


class SourceStampError(RuntimeError):
    """无法生成**可信**部署戳。

    故意用异常而不是返回一个"降级版戳"：戳存在的唯一理由是让 `git_commit`
    可追溯，一个缺了 commit 或用脏树生成的戳，会让 `git_commit` 看起来**有值**
    却是错的——那比 `NOT_MEASURED` 危险得多（错的、且像一次成功的采集）。
    """


def detect_git_dirty(
    root: str | Path | None, scope: tuple[str, ...] = SOURCE_DIGEST_SCOPE
) -> bool | None:
    """作用域内是否有未提交改动（含未跟踪文件）。``None`` = 不是 git 仓库/查不到。

    检查**限定在摘要作用域内**，这不是省事，是要让"这棵树可不可信"的两个判据
    看同一组文件：摘要回答"内容还一样吗"，这条回答"这些内容进过某个提交吗"。
    若把 `configs/`、`results/` 的改动也算脏，那么每生成一份新配置、每跑一个 run
    都会把树变脏，人会去关掉这个检查——一个总在误报的守卫最终等于没有守卫。

    用 `--porcelain` 而不是 `git diff --quiet`：后者看不见未跟踪文件，而一个
    "本地有、提交里没有"的源文件恰恰是"部署的代码无法从 commit 复现"的典型形态。
    """
    if root is None:
        return None
    base = Path(root)
    if not base.is_dir():
        return None
    paths = [s for s in scope if (base / s).exists()]
    cmd = ["git", "status", "--porcelain", "--"] + (paths or ["."])
    try:
        proc = subprocess.run(
            cmd, cwd=str(base), capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    if proc.returncode != 0:
        # 非 git 仓库会走到这里。返回 None 而不是 False：False 会被读成
        # "查过了，是干净的"，而真相是"这个检查在这里不成立"。
        return None
    return bool(proc.stdout.strip())


def build_source_stamp(
    root: str | Path,
    *,
    scope: tuple[str, ...] = SOURCE_DIGEST_SCOPE,
    allow_dirty: bool = False,
    deployed_at: str | None = None,
    deployed_by: str | None = None,
) -> dict[str, Any]:
    """生成一份部署戳（**只生成、不落盘**，投放由脚本负责）。

    失败即抛 :class:`SourceStampError`。三条失败路径都有具体理由：

    * 取不到 commit —— 要记的就是那个 commit，取不到时投戳毫无意义；
    * 工作区脏（限作用域）—— HEAD 描述的已经不是被部署的代码，默认拒绝，
      `allow_dirty=True` 才写，并在戳里如实标 ``git_dirty=true``；
    * 算不出摘要 —— 没有摘要的戳**无法判断是否过期**，它会一直自称有效。

    注意第三条：一个"永远不会过期"的戳比没有戳更糟，因为它把
    "版本未知" 伪装成 "版本已确认"。
    """
    base = Path(root)
    if not base.is_dir():
        raise SourceStampError(f"源码树根目录不存在：{base}")

    commit = detect_git_commit(base)
    if not commit:
        raise SourceStampError(
            f"{base} 不是 git 仓库（或 git 不可用），取不到 commit。\n"
            f"  部署戳的全部价值就是把远端那棵**拷贝过去**的树钉到一个 commit 上；"
            f"本机也取不到时，戳里就没有可钉的东西。\n"
            f"  请在**本机**的 git 仓库根目录投放（即本仓库工作区），"
            f"而不是在远端那棵没有 .git 的树上投放。"
        )

    dirty = detect_git_dirty(base, scope)
    if dirty and not allow_dirty:
        detail = ""
        try:
            detail = subprocess.run(
                ["git", "status", "--porcelain", "--"] + [s for s in scope],
                cwd=str(base), capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            pass
        raise SourceStampError(
            f"作用域 {scope} 内有未提交改动，HEAD 描述的**不是**将要部署的这棵树：\n"
            f"{detail}\n"
            f"  两个选择：先 `git commit`（推荐，这样 commit 能真正复现出这棵树）；"
            f"或明确用 `--allow-dirty` 承担——戳里会记 `git_dirty=true`，"
            f"此时 `git_commit` 只应被当作「这批代码的分支点」，不能当作该代码的版本号。"
        )

    digest = compute_tree_digest(base, scope)
    if digest is None:
        raise SourceStampError(
            f"无法计算源码树摘要：{base} 下作用域 {scope} 内一个文件都没找到。\n"
            f"  没有摘要的戳无法判断是否过期（它会对任何树都自称有效），"
            f"因此这里直接失败，而不是投一个「永远有效」的戳。"
        )

    return {
        "stamp_version": SOURCE_STAMP_VERSION,
        "git_commit": commit,
        "git_dirty": bool(dirty),
        "tree_digest": digest,
        "digest_scope": list(scope),
        "deployed_at": deployed_at or _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "deployed_by": deployed_by or detect_machine_id(),
    }



def detect_machine_id() -> str:
    """主机标识：优先 hostname，附带用户名以便区分共享机器上的不同账号。

    不是安全标识，只是为了让"同一台机器"这件事在日志里明确。
    """
    try:
        host = socket.gethostname()
    except Exception:
        host = ""
    try:
        user = getpass.getuser()
    except Exception:
        user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    if host and user:
        return f"{host}/{user}"
    return host or user or ""


def _lookup(env: Mapping[str, Any], *paths: str) -> Any:
    """按点分路径在嵌套字典里取值，取不到返回 None。"""
    current: Any = env
    for key in paths:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


# npu-smi 设备行形如：
#   | 0     910B3               | OK            | 92.4   39   0 / 0 ...
# 只取"行首是数字 + 设备名"这一特征，不依赖列宽——列宽随版本和终端宽度变化，
# 按列切会在换一台机器后静默解析出空值。
#
# 设备名允许以**数字**开头（910B3 / 310P3 都这样），但必须至少含一个字母，
# 否则会把纯数字列误认成设备名。
_NPU_ROW = re.compile(
    r"^\|\s*(\d+)\s+(?=[A-Za-z0-9\-_]*[A-Za-z])([A-Za-z0-9][A-Za-z0-9\-_]*)\s*\|"
)


def parse_npu_smi(text: str | None) -> dict[str, Any]:
    """从天真的 ``npu-smi info`` 文本里提取设备型号与数量。

    返回 ``{"models": [...], "model": "910B3", "count": N}``；
    解析不出任何设备行时返回 ``{"models": [], "model": "", "count": None}``
    ——数量为 ``None`` 表示"没解析出来"，与"解析出 0 张卡"区分开。

    这是一个**尽力而为**的解析器：不同 CANN 版本输出格式不同。因此它
    只提取最稳定的特征（行首编号 + 名称），其余一律不猜。
    """
    empty: dict[str, Any] = {"models": [], "model": "", "count": None, "raw_parsed": False}
    if not text:
        return empty

    indices: dict[int, str] = {}
    for line in text.splitlines():
        m = _NPU_ROW.match(line.strip())
        if not m:
            continue
        idx, name = int(m.group(1)), m.group(2)
        # 排除把版本号/表头误认成设备名的情况
        if name.lower() in {"version", "npu-smi", "health", "power"}:
            continue
        indices.setdefault(idx, name)

    if not indices:
        return empty

    models = [indices[i] for i in sorted(indices)]
    # 型号取出现次数最多者（异构机器上可能出现两种卡）
    dominant = max(set(models), key=models.count)
    return {
        "models": models,
        "model": dominant,
        "count": len(models),
        "raw_parsed": True,
    }


#: MTT（摩尔线程）产品名，例如 ``MTT S4000`` / ``MTT S80``。
#: 刻意只认这一个形状：认不出就不给结论（``count=None``），而不是猜一个数字。
_MTT_MODEL = re.compile(r"\bMTT\s*S?\d{2,4}[A-Za-z]{0,4}\b", re.IGNORECASE)

#: ★ 公开别名：外部需要"这串是不是 MTT 产品名"时用**这一份**正则，
#: 不要再写一条 —— 两份解析器迟早给出不同答案（本仓的通则）。
MTT_MODEL_RE = _MTT_MODEL

#: ``mthreads-gmi`` 的设备行：行首编号，且**同一行**里出现 MTT 产品名。
#: 不按列切——列宽随驱动版本变化，按列切会在换一台机器后静默解析出空值
#: （与 :data:`_NPU_ROW` 同一个教训）。
_MTT_DEVICE_ROW = re.compile(
    r"^\s*\|?\s*(\d+)\s+.*?MTT\s*S?\d{2,4}", re.IGNORECASE | re.MULTILINE
)

#: ``mthreads-gmi`` 的**驱动**版本，锚定显式的 ``Driver Version`` 标签，取值形状为
#: "标签 + 分隔符 + 数字开头"；认不出就留 ``None``，不猜。
#:
#: ★ 这里**刻意不**接受 ``mthreads-gmi:`` 这个标签——实测踩到的坑：
#:   真实输出把两个版本印在**同一行**上
#:       mthreads-gmi:1.14.0          Driver Version:2.7.0
#:   前者是 gmi **工具**版本，后者才是驱动版本。早先的正则把两者写成一个交替
#:   ``(?:mthreads-gmi|driver\s+version)``，而交替总是**先匹配到位置靠前的那个**，
#:   于是驱动版本被静默写成了工具版本（真实值 2.7.0 被写成 1.14.0）。
#:   错的版本号比 ``None`` 危险得多：它看起来像一次成功的采集，既不报错也不留痕，
#:   而驱动版本正是"块大小为什么是 64"这类结论的依据。
#:   工具版本没有丢——``mthreads-gmi`` 的原文已逐字存进 ``device.mthreads_gmi_info``，
#:   需要时可以从那里再取，不必在这里冒充驱动版本。
_MTT_DRIVER_VERSION = re.compile(
    r"driver\s+version\s*[:=]\s*v?([0-9][0-9A-Za-z.\-]*)",
    re.IGNORECASE,
)
#: MUSA（SDK）版本。必须带分隔符，避免在说明文字里误匹配 "MUSA" 三个字母。
_MTT_MUSA_VERSION = re.compile(
    r"musa(?:\s+sdk)?(?:\s+version)?\s*[:=]\s*v?([0-9][0-9A-Za-z.\-]*)",
    re.IGNORECASE,
)


def parse_mthreads_gmi(text: str | None) -> dict[str, Any]:
    """从 ``mthreads-gmi`` 文本里提取 MTT 设备型号与数量（**尽力而为**）。

    定位与免责同 :func:`parse_npu_smi`：不同驱动版本的版面差异很大，所以只提取
    最稳的两个特征——产品名形状（``MTT Sxxxx``）与"行首编号 + 同行出现产品名"的设备行。

    ``count`` 的三种取值必须分清：

    * 整数 —— 数出了设备行；
    * ``None`` —— **没认出来**，不是"0 张卡"。

    返回 ``{"models": [...], "model": "...", "count": N|None, "raw_parsed": bool}``。
    """
    empty: dict[str, Any] = {
        "models": [], "model": "", "count": None, "raw_parsed": False
    }
    if not text:
        return empty

    models = [re.sub(r"\s+", " ", m).strip() for m in _MTT_MODEL.findall(text)]
    if not models:
        return empty

    dominant = max(set(models), key=models.count)
    rows = len(_MTT_DEVICE_ROW.findall(text))
    return {
        "models": sorted(set(models)),
        "model": dominant,
        # 数不出设备行时**不回落成 1**：那会把"只看到一个产品名"当成"有一张卡"。
        "count": rows or None,
        "raw_parsed": True,
    }


def parse_mthreads_software(text: str | None) -> dict[str, str | None]:
    """从 ``mthreads-gmi`` 文本里提取**软件版本**（驱动 / MUSA），提取不到写 ``None``。

    为什么单独一个函数、而不是塞进 :func:`parse_mthreads_gmi`：那个函数回答的是
    "有几张卡"（硬件事实），这个回答的是"驱动是哪版"（软件事实）。两者在
    ``mthreads-gmi`` 里恰好打印在同一份文本中，但**取不到时的后果完全不同**——
    卡数取不到要判"未知"，驱动版本取不到只是少一条溯源信息。混在一个返回值里，
    调用方很容易把"版本没解析出来"错当成"没有设备"。

    取不到时**不猜**：返回 ``None`` 而不是编一个"unknown"字符串——后者会被写进
    指纹文件，在需要交叉核对驱动版本时看起来像一次成功的采集。

    ``driver_version`` **只来自显式的 ``Driver Version`` 标签**（见
    :data:`_MTT_DRIVER_VERSION`）。只有 ``mthreads-gmi:`` 工具版本行、没有独立驱动
    字段时返回 ``None``：把工具版本当驱动版本是被实测证实过的错误，而"少一条信息"
    比"多一条错信息"好修。
    """
    out: dict[str, str | None] = {"driver_version": None, "musa_version": None}
    if not text:
        return out
    driver = _MTT_DRIVER_VERSION.search(text)
    if driver:
        out["driver_version"] = driver.group(1)
    musa = _MTT_MUSA_VERSION.search(text)
    if musa:
        out["musa_version"] = musa.group(1)
    return out


def parse_musa_version_file(text: str | None) -> str | None:
    """从 MUSA 的版本文件内容里取 SDK 版本；取不到返回 ``None``（不猜）。

    为什么单独一个纯函数、而不是在 ``scripts/inspect_env.sh`` 里就地写一段：脚本里
    的逻辑跑在 heredoc 里，**单测覆盖不到**；而这正是上次漏掉的地方——采集端只找了
    ``/usr/local/musa/version`` 与 ``version.txt``，实测那台主机的 MUSA 3.1.0 装的却是
    ``version.json``。文件找不到，MUSA 版本就静默变成 ``None``，看起来跟"这台机器
    没装 MUSA"一模一样。

    ``version.json`` 里同名组件很多（``mublas``/``muPP``/``mufft``…），每个都有自己的
    ``version`` 字段。**只有 ``musa_toolkits``（缺失时退到 ``musa_runtime``）才是 SDK
    版本**；随手取文件里出现的第一个数字会取到别的组件（或 schema 版本）。
    """
    if not text:
        return None
    if text.lstrip().startswith("{"):
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        for key in ("musa_toolkits", "musa_runtime"):
            block = data.get(key)
            if isinstance(block, dict):
                version = block.get("version")
                if version:
                    return str(version)
        return None
    # 纯文本版本文件（``version`` / ``version.txt``）：只有一个版本号，取第一个数字串。
    found = re.search(r"([0-9][0-9A-Za-z.\-]*)", text)
    return found.group(1) if found else None


def normalize_availability(probe_results: Sequence[bool | None]) -> bool | None:
    """把多个设备探测的结果压成一个可用性。三态，取"最保守的可判定值"。

    每个探测的取值含义：

    * ``True``  —— 这个探测确实认出了至少一张设备；
    * ``False`` —— 这个探测**跑了**，并明确没有设备（含"探测工具不存在"）；
    * ``None``  —— 跑了但**认不出来**（输出格式不认、命令报错）。

    合成规则：任一 ``True`` -> ``True``；否则任一 ``None`` -> ``None``；否则 ``False``。
    一个探测都没有 -> ``None``。

    为什么不把 ``None`` 当 ``False``：两者要修的东西不同。``False`` 是"这台机器上
    没有该硬件"，``None`` 是"探测没得出结论"。后者该去修探测或换机器重采，
    读成"没有硬件"会让一个可修的问题看起来像硬件缺失，于是永远不会被修。

    这条规则比"命令退出码为 0 就算可用"严：退出码只说明**工具能跑**，
    不说明**它认出了设备**。后者才是"这台机器能产出实测数据"的证据。
    """
    results = list(probe_results)
    if not results:
        return None
    if any(r is True for r in results):
        return True
    if any(r is None for r in results):
        return None
    return False


def device_probe_result(
    tool_present: bool,
    count: int | None,
    has_name: bool = False,
) -> bool | None:
    """单次设备探测的结论。三态：``True`` 认出设备 / ``False`` 明确没有 / ``None`` 认不出。

    判定**只认设备记录**（解析出的卡数），**不认产品名**。这条边界是必须的：
    一句 ``mthreads-gmi 2.7.0 supports MTT S4000`` 里也含产品名，但那是**说明
    文字**，不是"这台机器上有一张卡"的证据。只看产品名会把这种文本判成可用，
    于是给一台没有设备的机器背书——而这类错误不报错、也不产生任何异常，
    只是让一批数字带着"硬件实测"的标签出现。

    ``count`` 的三种取值各有含义，不能混：

    * 正整数 —— 数出了设备行 -> ``True``；
    * ``0`` —— 工具跑了、明确一张卡都没有 -> ``False``；
    * ``None`` —— **没认出来** -> ``None``。这里不是"没有卡"，是"没查清"。

    工具不存在 -> ``False``：那不是"没查清"，是"这台机器上压根没这个工具"，
    与"跑了但认不出输出格式"要修的东西完全不同。

    ``has_name`` 保留在签名里是为了让调用点的语义显式（调用方确实拿到了产品名），
    但它**不参与判定**——见上。
    """
    if not tool_present:
        return False
    if count is None:
        return None
    return count >= 1


#: 环境指纹里"检出的平台"的键名。新格式写 ``platform``；``hardware_platform``
#: 作为别名一起认，免得两个名字各写一半。
_ENV_PLATFORM_KEYS = ("platform", "hardware_platform")


def env_platform(payload: Mapping[str, Any] | None) -> HardwarePlatform:
    """从环境指纹里读**检出的**硬件平台。读不到就是 :attr:`HardwarePlatform.UNKNOWN`。

    刻意**不**从 ``device.npu_smi_available`` 之类的旧字段反推平台：旧指纹只证明了
    "那个工具能跑"，没有声明这是哪台机器。按"缺证据就保留 unknown"处理，
    让历史数据落在"无从判断"，而不是被猜一个平台——猜错会把下游按平台筛选的
    结果直接带偏，且不报错。

    但"**没有**平台字段"与"写了平台字段、值不认识"是两回事，处理方式刻意不同：

    * 没有字段 -> ``UNKNOWN``（旧指纹，缺证据，放行）；
    * 值不认识 -> **抛错**。静默降级成 ``UNKNOWN`` 会把一个新平台的数据混进"平台
      未知"的桶里：既不报错，也无法按平台筛出来，且声明与检出的矛盾从此永远检不出。

    与 :meth:`HardwarePlatform.parse` 的分工同 :class:`ExecutionEnvironment`：
    加载配置时对拼错抛错，读指纹时对未登记平台抛错——两处都宁可报错，不要猜。
    """
    if not payload:
        return HardwarePlatform.UNKNOWN
    for key in _ENV_PLATFORM_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            try:
                return HardwarePlatform.parse(value)
            except ValueError as exc:
                raise ValueError(
                    f"环境指纹里的 {key}={value!r} 不是已知平台（{exc}）。"
                    f"指纹出现没登记的平台时**不静默降级成 unknown**——那会让这批"
                    f"数据混进'平台未知'的桶里，既不报错、也无法按平台筛出，"
                    f"声明与检出的矛盾更会从此检不出来。"
                    f"若这确实是新平台，请在 provenance.REMOTE_PLATFORM_PAIRS 加一行"
                    f"（判定集合、报错修法、自洽校验三处都由它派生），"
                    f"并让 scripts/inspect_env.sh 能认出它。"
                ) from exc
    return HardwarePlatform.UNKNOWN


@dataclass
class ReproInputs:
    """构造 :class:`ReproducibilityRecord` 所需的原始输入。

    全部可选。构造器只做搬运与回落，不做任何编造。
    """

    run_id: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    workload_fingerprint: Mapping[str, Any] = field(default_factory=dict)
    hardware_env: Mapping[str, Any] = field(default_factory=dict)
    repo_root: str | Path | None = None
    warmup_count: int | None = None
    measurement_count: int | None = None
    tensor_parallel_size: int | None = None
    model_revision: str = ""
    #: 传入的 ``model_revision`` 是**解析出的 commit**（``True``）还是**请求的引用**（``False``）？
    #: 缺省 ``None`` = 调用方没说明——此时按"引用"处理（保守：不声称它是 commit）。
    model_revision_resolved: bool | None = None
    #: 推理实际执行的环境。缺省 UNKNOWN 时回落到配置里的同名声明。
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.UNKNOWN
    #: 面向的硬件平台。缺省 UNKNOWN 时回落到配置里的 ``runtime.hardware_platform``。
    hardware_platform: HardwarePlatform = HardwarePlatform.UNKNOWN
    #: 执行后端（``simulate`` / ``real``）。只在调用方知道时传——猜不出来。
    backend: str = ""


def build_reproducibility_record(
    inputs: ReproInputs,
    *,
    label: DataLabel,
    timestamp: str | None = None,
) -> ReproducibilityRecord:
    """从配置 + 环境指纹装配元数据。

    取值优先级：显式入参 > 配置/环境 > 缺失标记。
    这里**只搬运**：任何一个取不到的字段都会原样留空，由
    :meth:`ReproducibilityRecord.describe` 写成缺失标记。
    """
    cfg = inputs.config or {}
    env = inputs.hardware_env or {}
    wl = inputs.workload_fingerprint or {}

    runtime = cfg.get("runtime") or {}
    chunk = cfg.get("chunk") or {}
    cache = cfg.get("cache") or {}
    sched = cfg.get("scheduler") or {}

    npu_text = _lookup(env, "device", "npu_smi_info") or _lookup(env, "npu_smi_info")
    parsed = parse_npu_smi(npu_text if isinstance(npu_text, str) else None)

    # 硬件指纹的字段名以 scripts/inspect_env.sh 的产物为准，同时容忍几种常见写法。
    # ``device.model`` / ``device.count`` 是采集端**规范化**后的键（新格式）；
    # 旧格式没有它们，于是继续认那几种历史写法，最后才回落到解析 npu-smi 文本。
    #
    # 规范化键必须排在第一位：采集端已经替读端解决了"该看哪个平台"的问题，
    # 读端再自己判断一次，就是在两个地方维护同一条规则——而那正是本轮的
    # 第二项修复（MTT 型号/卡数已采集却进不了元数据）。
    device_model = (
        _lookup(env, "device", "model")
        or _lookup(env, "device_model")
        or _lookup(env, "npu", "model")
        or _lookup(env, "hardware", "device_model")
        or _lookup(env, "discovery", "values", "ASCEND_DEVICE_MODEL")
        or parsed["model"]
    )
    # 卡数刻意**拆开写**：``0`` 是一个真实测量值（测量到"一张卡都没有"），
    # 用 ``or`` 串起来会让它被当成"缺值"而继续回落，最终静默变成 None。
    device_count = _lookup(env, "device", "count")
    if device_count is None:
        device_count = (
            _lookup(env, "device_count")
            or _lookup(env, "npu", "count")
            or _lookup(env, "hardware", "device_count")
            or _lookup(env, "device", "device_count")
            or _lookup(env, "discovery", "values", "ASCEND_DEVICE_COUNT")
        )
    if device_count is None:
        device_count = parsed["count"]
    precision = runtime.get("precision") or runtime.get("dtype")
    graph_mode = runtime.get("graph_mode") or runtime.get("acl_graph")
    quantization = _as_text(runtime.get("quantization"))
    if not quantization and runtime.get("quantization_detail"):
        # 兼容把细节放在独立键下的配置写法。
        quantization = _as_text(runtime.get("quantization_detail"))

    # 执行环境：显式入参优先，否则读配置里的 runtime.execution_environment。
    # 未知取值由 parse 抛错——拼错的声明必须当场暴露，不能退化成"没声明"。
    exec_env = inputs.execution_environment
    if exec_env is ExecutionEnvironment.UNKNOWN and runtime.get("execution_environment"):
        exec_env = ExecutionEnvironment.parse(runtime.get("execution_environment"))

    # 硬件平台：同样先看显式入参，再看配置。
    hw_platform = inputs.hardware_platform
    if hw_platform is HardwarePlatform.UNKNOWN and runtime.get("hardware_platform"):
        hw_platform = HardwarePlatform.parse(runtime.get("hardware_platform"))

    # 装配之前先做自洽性校验。放在这里而不是只放在配置加载处：
    # 直接构造 ReproInputs 的调用方（测试、脚本）同样不该产出自相矛盾的记录，
    # 而在装配处拦下，比发现一份已落盘、字段互斥的 meta 要早得多。
    validate_platform_consistency(exec_env, hw_platform)

    # 再查"说的"与"**查到的**"是否一致。只查自洽性会留下一个空子：声明可以完全
    # 自洽，同时与真机毫无关系（"声明 MTT、指纹明确是 Ascend"就是这样通过的）。
    # 这是唯一能**证伪**声明的检查——`validate_platform_consistency` 只能保证两个
    # 声明不互相打架，证明不了任何事实。
    #
    # 放在装配处而不是只放在配置加载处：直接构造 ReproInputs 的调用方（测试、
    # 脚本）同样不该产出一份"声明与指纹矛盾"的记录。
    detected = env_platform(env)
    validate_platform_evidence(
        hw_platform, detected, source=str(env.get("source") or "未记录来源")
    )

    # 源码版本：真 git 仓库 > 部署戳（且摘要相符）> 取不到。
    # 远程推理机上的树不是 git 仓库是**常态**，所以这一步不能省成 detect_git_commit。
    #
    # 用 detect_source_identity 而不是 detect_source_commit：后者拿到戳里的 commit 后
    # **丢掉了 tree_digest**，于是产物里只剩一根光秃秃的 commit（脏树上它只是分支点）。
    # 续跑判据要的是摘要级身份，所以这里必须把摘要一起落盘。
    (git_commit, git_source, git_dirty, git_tree_digest) = detect_source_identity(
        inputs.repo_root
    )

    # model_revision 的**未解析**回落链：显式入参 > 配置。
    # 只有当调用方没有给出解析后的 commit 时才会走到这里，此时值是一个**引用**
    # （如 `main`），必须把 model_revision_resolved 标成 False——否则下游会把
    # `main` 当成一个可以复现的版本号，而它今天指向 A、明天可能指向 B。
    revision = inputs.model_revision or runtime.get("model_revision") or ""
    if inputs.model_revision_resolved is not None:
        resolved = inputs.model_revision_resolved
    else:
        # 调用方没说明值的性质：保守按"引用"处理，不替它声称是 commit。
        # 连引用都没有（revision 为空）时记 None = 没查——不是"查了且不是 commit"。
        resolved = False if revision else None

    return ReproducibilityRecord(
        run_id=inputs.run_id,
        timestamp=timestamp or _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        git_commit=git_commit,
        hostname_or_machine_id=detect_machine_id(),
        device_model=device_model or "",
        device_count=device_count,
        model=runtime.get("model") or runtime.get("model_name") or "",
        # model_revision 优先取显式入参（例如从服务端 /v1/models 读到的实际版本），
        # 否则回落到配置。两者都取不到就留空——权重版本对不上是结果不可复现的
        # 常见原因，不能用一个猜出来的值糊过去。
        model_revision=revision,
        precision=precision or "",
        quantization=runtime.get("quantization") or "",
        tensor_parallel_size=(
            inputs.tensor_parallel_size
            if inputs.tensor_parallel_size is not None
            else runtime.get("tensor_parallel_size")
        ),
        input_length=wl.get("prompt_len_mean"),
        output_length=wl.get("gen_len_mean"),
        concurrency=wl.get("concurrency"),
        batch_configuration=(
            f"policy={sched.get('policy')};max_num_seqs={sched.get('max_num_seqs')}"
            if sched
            else ""
        ),
        max_batched_tokens=chunk.get("max_num_batched_tokens"),
        chunk_size=chunk.get("chunk_size"),
        prefix_cache_enabled=cache.get("enabled"),
        graph_mode=graph_mode or "",
        random_seed=wl.get("seed"),
        warmup_count=inputs.warmup_count,
        measurement_count=inputs.measurement_count,
        data_label=label,
        # 硬件未知 -> 缺失字段标记为 HARDWARE_REQUIRED；硬件已知但字段缺失 -> NOT_MEASURED
        hardware_required=(_lookup(env, "available") is False),
        execution_environment=exec_env,
        hardware_platform=hw_platform,
        backend=inputs.backend,
        git_source=git_source,
        git_dirty=git_dirty,
        tree_digest=git_tree_digest,
        model_revision_resolved=resolved,
    )


def _as_text(value: Any) -> str:
    """把配置值压成一个字符串字段。

    非字符串（例如 w8a8 的量化细节是一个嵌套字典）会被压成紧凑 JSON，
    而不是被丢掉或原样塞进去。原因：``quantization`` 在元数据里是**标量标签**，
    下游（表格、分组、筛选）都按标量处理；混进一个 dict 会让 CSV 里出现
    一坨转义引号，且分组时每行都是唯一值。细节仍然保留，只是换个形态。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    try:
        import json as _json

        return _json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def summarize_labels(records: Sequence[DataLabel]) -> dict[str, int]:
    """统计一组标签的分布，供聚合脚本打印"这批数据里有多少是可用的"。"""
    counts: dict[str, int] = {label.value: 0 for label in DataLabel}
    for label in records:
        counts[label.value] += 1
    return counts
