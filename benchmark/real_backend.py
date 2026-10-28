"""真实硬件执行后端（进程内 Transformers）。

分层
----
本模块刻意把两件事分开：

**纯逻辑** —— :class:`RealBackend`：到达/受理/排队时刻、批调度与账目、请求终态。
它不 import torch，不含任何设备调用（**全部 lazy import**），因此可以在无加速卡的
机器上被单测完整覆盖。

**设备驱动** —— :class:`GenerationDriver` 协议：只做"喂一批 token 进去、告诉我
每个序列什么时候拿到了 token"。两个实现：

- :class:`TransformerDriver`：真机实现（HuggingFace Transformers + MUSA）。
- :class:`FakeDriver`：纯 Python + 虚拟时钟，供本地单测使用，**不触达任何设备**。

驱动是**注入**的，所以契约（时刻 -> ``RequestMetrics``、账目自洽、结局三态）
可以在本机用假驱动完整验证。真机与本地跑的是同一份 ``RealBackend``。

为什么必须复用 ``ascend_opt.step_planner.plan_step``
----------------------------------------------------
批预算 / 分块额度 / 缓存命中 / token 分配这套组合逻辑只有一份实现。
真实后端如果自己再写一遍，那么以后修了模拟器里的某一处闸门（例如"两处都夹取
导致控制器调大被静默压回"那个缺陷），真机路径不会跟着修——两条路径给出不同的
调度行为却都自称"同一套策略"，而产物里看不出任何异常。

本版支持与不支持的能力（**必须显式，不靠读者猜**）
------------------------------------------------------
支持：

- **分块预填充**：``plan_step`` 的分配结果直接变成每次前向喂入的 token 数，
  长 prompt 可以跨多步推进，decode 因此不会被整段 prefill 阻塞。
- **连续批处理（调度语义）**：每一步都用当前队列重算计划，请求可以在其它请求
  还在 decode 时获准起步，不存在"整批等齐"。同一步里 prefill（含续传）与
  decode 合成**一个**批送进设备——这正是连续批处理在真机上的样子。
- **自适应控制器**：与 dry-run 走完全相同的 ``AdaptiveController`` 与
  ``WorkloadMonitor``，决策依据的 ``s_t`` 一并落盘。
- **token 级到达时刻**：每个 token 的可得时刻在**设备同步之后**打点，
  因此 ``stall``（单请求 token 间隔 P99）可测——HTTP 流式路径拿不到这个量。

**不支持**（违反时**报错**，不静默降级）：

- **前缀缓存**：``cache.enabled=true`` 会被拒绝。HF 的原生 KV 缓存没有按前缀
  哈希复用块的能力，若照常"记命中"却并不省算力，产物里的 ``cached_tokens``
  就是一个**看起来正常但错**的数。该能力留给按功能选定的引擎。
- **非 MUSA 设备**：拿不到 ``musa`` 设备时直接报错。静默退回 CPU 会把
  "本机 CPU 时延"写成"设备实测时延"，且不会有任何迹象。

已知与真实引擎的差异（**记录而非隐藏**）
------------------------------------------
HF 的 ``past_key_values`` 是规则张量，批内各序列 history 长度不一致时需要补位
（补位区域用 attention mask 屏蔽，写回时切掉）。这带来两个真实开销，分别计时并
写进 ``backend_stats``：

- ``pack_ms``：补位与切片的显存搬运，**不含任何设备计算**；
- ``n_forwards``：一次"逻辑步"实际发起几次前向。形状一致的批（补位量为零）
  只有 1 次；形状参差的批会被按喂入长度拆开，launch 次数随之上升。

这两个量本身就是有意义的指标（自适应控制器会改变批的规整度），因此并排报告，
而不是藏进总耗时里。

``wall_ms`` 只覆盖**正式测量段**：``load()`` 与 ``warmup()`` 都在时间原点之前。
把装载算进 wall_ms 会让吞吐偏低，而"偏低"看起来完全正常——没有字段会暴露它。
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from arm_factory import build_arm  # 与模拟器路径共用
from ascend_opt import (
    AdaptiveController,
    apply_arm,
    arm_state,
    ChunkLimits,
    ExecutionResult,
    PrefixCachePolicy,
    Request,
    RequestMetrics,
    TelemetrySample,
    WorkloadMonitor,
    build_chunk_policy,
    build_scheduler,
    percentile,
    plan_step,
)
from ascend_opt.metrics import OUTCOME_FAILED, OUTCOME_SUCCESS, OUTCOME_UNFINISHED
from ascend_opt.execution_shape import (
    execution_shape_key, group_by_new_tokens, plan_execution_shape, plan_forward_specs,
)

__all__ = [
    "VirtualClock",
    "StepTiming",
    "ForwardItem",
    "SlotResult",
    "StepOutcome",
    "GenerationDriver",
    "SyntheticPromptSource",
    "FakeDriver",
    "TransformerDriver",
    "RealBackend",
    "build_driver",
]


# --------------------------------------------------------------------------- #
# 时钟
# --------------------------------------------------------------------------- #

def _default_clock() -> float:
    """单调高精度时钟（秒）。所有时刻共用一把，差值才有意义。"""
    return time.perf_counter()


def _resolved_revision(model: Any) -> str | None:
    """模型**实际加载到**的权重 commit，取不到返回 ``None``。

    HF 的 ``from_pretrained`` 成功后会把这个 commit 留在 ``config._commit_hash`` 上。
    它和配置里的 ``model_revision`` 是两件事：后者通常是引用 ``main``，
    今天指向 A、明天可能指向 B；只有这个值能把一次 run 钉到具体权重上。

    **取不到就是 ``None``，不回落成 ``main``**——把引用当成版本号写进元数据，
    就是把"没记录版本"伪装成"记录了版本"。
    """
    cfg = getattr(model, "config", None)
    value = getattr(cfg, "_commit_hash", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class VirtualClock:
    """可手动推进的时钟，供单测做确定性计时。

    真机时钟是"读一次走一点"，无法断言具体数值；假驱动配虚拟时钟之后，
    "第 3 步的时刻应当是 x" 这种断言才成立——否则单测只能断言"不为 None"，
    而那种断言抓不住"时刻记错了参考点"这类错误。
    """

    def __init__(self, start: float = 0.0) -> None:
        self.t = float(start)

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(f"时钟不能倒退，收到 {seconds}")
        self.t += float(seconds)


@dataclass(frozen=True)
class StepTiming:
    """一次"逻辑步"的耗时分解。时刻为 perf_counter 秒，由调用方换算成相对 ms。

    刻意把四个量分开：只报一个总耗时，看的人无法判断该去优化设备计算、
    优化补位搬运，还是该去减少 launch 次数——而三者的对策完全不同。
    """

    n_seqs: int
    n_tokens: int
    #: 本步实际发起的前向次数（形状参差的批会被拆开）。
    n_forwards: int
    #: 补位/切片搬运耗时（ms）。**不含设备计算**。
    pack_ms: float
    #: 设备前向耗时（ms），含同步等待。
    fwd_ms: float
    #: 本步开始时刻。
    started_s: float
    #: 本步结束时刻（**已在设备同步之后**）。
    finished_s: float
    #: 采样 + 设备→主机取回耗时（ms）。**不属于设备计算**，但属于 TTFT/E2E：
    #: logits 算完与 token 可取回主机是两件事，早先被记成同一时刻，
    #: 于是时延系统性偏早。这里单独列出来，偏早量便可直接读出。
    readback_ms: float = 0.0

    @property
    def elapsed_ms(self) -> float:
        return (self.finished_s - self.started_s) * 1000.0

    def describe(self) -> dict[str, Any]:
        return {
            "n_seqs": self.n_seqs,
            "n_tokens": self.n_tokens,
            "n_forwards": self.n_forwards,
            "pack_ms": round(self.pack_ms, 4),
            "fwd_ms": round(self.fwd_ms, 4),
            "readback_ms": round(self.readback_ms, 4),
            "elapsed_ms": round(self.elapsed_ms, 4),
        }


@dataclass(frozen=True)
class ForwardItem:
    """交给驱动的一条序列需求。

    **由后端（纯逻辑）决定要喂多少 token、这次该不该出 token**，驱动只负责执行。
    这样"哪个时刻该产出 token"这个判断只有一处（后端的状态机）；
    若让驱动自己也判一次，两份判断迟早不一致。
    """

    req: Request
    #: 本次前向要喂入的新 token 数。分块预填充下可以大于 1。
    n_new_tokens: int
    #: 本次前向结束后，该序列是否应产出一个 token。
    #: 分块预填充中途的前向不产出 token（prompt 还没算完）。
    need_token: bool


@dataclass
class SlotResult:
    """驱动返回的单序列结果。"""

    req_id: str
    #: 本次产出的 token id；``need_token=False`` 的序列为 ``None``。
    #: 用 ``None`` 而不是 0：0 是合法的 token id。
    token_id: int | None
    #: 该序列本次前向**结束**的时刻（perf_counter 秒，设备同步之后）。
    #: 产出 token 的序列：token 可得那一刻；不产出 token 的序列：本组前向结束那一刻。
    #: 后者不是 ``None``——"走完了一步"本身是有信息的，而"有没有产出 token"
    #: 由 :attr:`token_id` 单独表达，两个问题不该挤进一个字段。
    ready_s: float | None
    #: 本行的**末位置 logits 摘要**（仅驱动开了 `capture_logits_digest` 时非 `None`）。
    #: 判据是其中的 `sha256`（逐位）；`argmax`/`max`/`sum` 只为便于人读。
    #: ⚠️ 默认 `None` ⇒ 既有字段顺序与语义不变（新增字段放在**末尾**，
    #:    且 `SlotResult` 一律用关键字构造的那些点不受影响）。
    logits_digest: dict[str, Any] | None = None


@dataclass
class StepOutcome:
    timing: StepTiming
    slots: list[SlotResult] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 驱动协议
# --------------------------------------------------------------------------- #

class GenerationDriver(Protocol):
    """设备侧最小接口。

    只有四个动作：装载、预热、推进一步、回收。刻意不提供"按请求生成完整回复"
    这样的高层方法——那会把批调度（本项目的**研究变量**）藏进驱动里，
    两个驱动就会有不同的调度语义，实验之间不可比。
    """

    #: 是否真正在硬件上执行。假驱动必须为 ``False``：
    #: 它让产物能自证"这批数字不是实测"，堵住"本地假驱动跑出 MEASURED"这条路。
    hardware_backed: bool

    def describe(self) -> dict[str, Any]:
        """驱动身份（写进产物，用于溯源）。"""

    def load(self) -> dict[str, Any]:
        """装载模型与运行时。**在计时窗口之外调用**。"""

    def warmup(
        self, target_s: float, *, batch: int, prompt_len: int, gen_len: int
    ) -> dict[str, Any]:
        """按**时间**预热，直到累计耗时 >= ``target_s``。

        为什么按时间而不是按轮数：单轮耗时随模型/批大小差一个数量级，
        固定轮数在某个模型上够、在另一个上远远不够，而"够不够"看不出来。
        返回值记录**实际**预热轮数与耗时——写"预热了 30s"却只跑了 3s，
        是那种不会报错的偏差。
        """

    def forward(self, batch: Sequence[ForwardItem]) -> StepOutcome:
        """推进一批序列各若干 token。**一次逻辑步 = 一次调用**（内部可能拆多次前向）。"""

    def retire(self, req_ids: Sequence[str]) -> None:
        """释放已完成序列的设备侧状态（KV 缓存）。

        **逐请求及时调用**（在后端里是每步结算完就调），不是整轮结束才调一次：
        单个序列的 KV 占用与 ``prompt_len + gen_len`` 成正比，攒着不放到长负载上
        就是 OOM，而 OOM 与"负载太重"长得一样。幂等：重复调用不应重复付开销。
        """

    def live_slots(self) -> int:
        """当前仍持有设备侧状态的序列数。

        这是"及时释放"的**可观测量**，因此进协议而不是当作实现细节：
        没有它，产物里只能写一句"已释放"，而那句话在"整轮结束才释放"的
        实现里同样成立。
        """

    def sample_device(self) -> dict[str, Any]:
        """可选：采样设备状态（显存等）。取不到就返回空 dict，**不要编数**。"""

    def stats(self) -> dict[str, Any]:
        """累计统计（前向次数、搬运耗时、显存峰值等）。"""


# --------------------------------------------------------------------------- #
# 确定性 prompt 合成
# --------------------------------------------------------------------------- #

class SyntheticPromptSource:
    """按长度与共享前缀确定性地合成 token id。

    为什么不用真实文本：工作负载模型（``WorkloadSpec``）描述的就是**长度**，
    它没有文本字段。改用真实语料会引入两件麻烦：语料本身不可复现，
    以及"分词后的 token 数 != prompt_len"这个新的偏差源。

    合成序列保证三件事：

    1. 同一 ``req_id`` 在任何机器上得到逐位相同的 id（seed 由 req_id 派生）；
    2. 同一 ``prefix_group`` 的前 ``shared_prefix_len`` 个 token 逐位相同——
       这是前缀缓存收益的**基数**，不保证它，缓存实验就没有可解释的上限；
    3. 所有 id 落在 ``[0, vocab_size)`` 且避开给出的特殊 token，
       否则模型可能提前吐出 EOS，``gen_len`` 悄悄变短。

    **必须写明的代价**：这批数字反映的是**长度维度上的时延**，
    不含语义正确性。输出正确性由 PHASE 1 的单请求验证单独负责（已通过）。
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        seed: int = 0,
        excluded_ids: Sequence[int] = (),
    ) -> None:
        if vocab_size < 2:
            raise ValueError(f"vocab_size 至少为 2，收到 {vocab_size}")
        # 一律从 id 1 起：0 在很多词表里是 pad/unk 之类的特殊 token。
        self.lo = 1
        self.hi = int(vocab_size)
        self.seed = int(seed)
        self.excluded = frozenset(int(i) for i in excluded_ids)
        if len(self.excluded) >= (self.hi - self.lo):
            raise ValueError(
                f"被排除的 token 数（{len(self.excluded)}）不能占满整个词表区间 "
                f"[{self.lo}, {self.hi})，否则任何 prompt 都合不出来"
            )

    def _rng(self, tag: str) -> random.Random:
        digest = hashlib.blake2b(tag.encode("utf-8"), digest_size=8).digest()
        return random.Random(self.seed ^ int.from_bytes(digest, "big"))

    def _draw(self, rng: random.Random) -> int:
        x = rng.randrange(self.lo, self.hi)
        # 拒绝采样避开特殊 token。命中率极低（特殊 token 通常只有个位数），
        # 但**不能省**：省掉之后 EOS 何时出现取决于词表布局，
        # 会让"生成长度"随模型变化，而产物的 gen_len 列看不出这件事。
        while x in self.excluded:
            x = rng.randrange(self.lo, self.hi)
        return x

    def _block(self, tag: str, n: int) -> list[int]:
        rng = self._rng(tag)
        return [self._draw(rng) for _ in range(n)]

    def prompt_ids(self, req: Request) -> list[int]:
        """该请求的完整 prompt id 序列（长度 == ``req.prompt_len``）。"""
        n = int(req.prompt_len)
        if n <= 0:
            raise ValueError(f"请求 {req.req_id} 的 prompt_len 必须为正，收到 {n}")
        ids = self._block(f"prompt:{req.req_id}", n)
        shared = int(req.shared_prefix_len)
        if shared > 0 and req.prefix_group:
            if shared > n:
                raise ValueError(
                    f"请求 {req.req_id} 的 shared_prefix_len={shared} 超过 "
                    f"prompt_len={n}"
                )
            # 组内共享段按**组名**派生：同组请求逐位相同，跨组不同。
            ids[:shared] = self._block(f"prefix:{req.prefix_group}", shared)
        return ids

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "synthetic",
            "seed": self.seed,
            "vocab_size": self.hi,
            "excluded_ids": sorted(self.excluded),
            "_note": (
                "prompt 内容按长度确定性合成；"
                "测的是长度维度上的时延，不含语义正确性"
            ),
        }


# --------------------------------------------------------------------------- #
# 假驱动（本地单测用，不触达设备）
# --------------------------------------------------------------------------- #

class FakeDriver:
    """纯 Python 驱动，供本地单测覆盖后端契约。**不触达任何设备。**

    ``hardware_backed=False``，因此由它产出的任何产物都能自证不是实测，
    并且 :class:`RealBackend` 会拒绝把它当成实测结果。

    时间由注入的 :class:`VirtualClock` 推进：``forward()`` 按配置的公式
    算出这一步该花多久，然后把时钟往前推。于是单测可以断言
    "第 k 步之后 now 应当是多少"，而不是只能断言"不为 None"。
    """

    hardware_backed = False

    def __init__(
        self,
        *,
        fixed_ms: float = 1.0,
        per_seq_ms: float = 0.5,
        per_token_ms: float = 0.01,
        vocab_size: int = 1024,
        clock: VirtualClock | None = None,
        fail_on_step: int | None = None,
        fail_message: str = "注入的驱动故障",
    ) -> None:
        self.fixed_ms = float(fixed_ms)
        self.per_seq_ms = float(per_seq_ms)
        self.per_token_ms = float(per_token_ms)
        self.clock = clock or VirtualClock()
        self.prompt_source = SyntheticPromptSource(vocab_size=vocab_size)
        self.fail_on_step = fail_on_step
        self.fail_message = fail_message
        self.calls: list[dict[str, Any]] = []
        self.retired: list[str] = []
        self._n_forward_calls = 0
        self._next_token: dict[str, int] = {}
        #: 当前"持有设备侧状态"的序列（真实驱动里就是 KV 槽位）。
        #: 假驱动模拟它的生命周期，好让"及时释放"在本机可被断言。
        self._live: set[str] = set()

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "fake",
            "hardware_backed": False,
            "step_time_ms": {
                "fixed": self.fixed_ms,
                "per_seq": self.per_seq_ms,
                "per_token": self.per_token_ms,
            },
            "_note": "假驱动：时延由公式合成，未经任何设备执行",
        }

    def load(self) -> dict[str, Any]:
        return {"loaded": True, "kind": "fake"}

    def warmup(self, target_s, *, batch, prompt_len, gen_len) -> dict[str, Any]:
        # 假驱动的预热只记账、不推进虚拟时钟：真正的"按时间预热"逻辑
        # 由真机驱动负责，这里假装时钟没动，测试才好断言时间原点。
        return {
            "target_s": target_s,
            "warmup_s": 0.0,
            "iterations": 0,
            "skipped": True,
            "_note": "假驱动不预热；真机驱动的预热在时间原点之前完成",
        }

    def forward(self, batch: Sequence[ForwardItem]) -> StepOutcome:
        if not batch:
            raise ValueError("forward() 收到空批；空批不应发起前向")
        for it in batch:
            if it.n_new_tokens <= 0:
                raise ValueError(
                    f"ForwardItem({it.req.req_id}) 的 n_new_tokens="
                    f"{it.n_new_tokens}；要推进一步就必须喂至少一个 token"
                )
        self._n_forward_calls += 1
        if self.fail_on_step is not None and self._n_forward_calls >= self.fail_on_step:
            raise RuntimeError(self.fail_message)

        started = self.clock.now()
        n_tokens = sum(it.n_new_tokens for it in batch)
        elapsed_s = (
            self.fixed_ms + self.per_seq_ms * len(batch) + self.per_token_ms * n_tokens
        ) / 1000.0
        self.clock.advance(elapsed_s)
        finished = self.clock.now()
        self.calls.append(
            {
                "n_seqs": len(batch),
                "n_tokens": n_tokens,
                "need_token": sum(1 for it in batch if it.need_token),
                "req_ids": [it.req.req_id for it in batch],
                "elapsed_ms": round(elapsed_s * 1000.0, 6),
            }
        )

        slots: list[SlotResult] = []
        for it in batch:
            self._live.add(it.req.req_id)
            if not it.need_token:
                slots.append(SlotResult(it.req.req_id, None, finished))
                continue
            tok = self._next_token.get(it.req.req_id, 1)
            self._next_token[it.req.req_id] = (tok + 1) % 1024 or 1
            slots.append(SlotResult(it.req.req_id, tok, finished))
        return StepOutcome(
            timing=StepTiming(
                n_seqs=len(batch),
                n_tokens=n_tokens,
                n_forwards=1,
                pack_ms=0.0,
                fwd_ms=elapsed_s * 1000.0,
                started_s=started,
                finished_s=finished,
            ),
            slots=slots,
        )

    def retire(self, req_ids: Sequence[str]) -> None:
        self.retired.extend(req_ids)
        for rid in req_ids:
            self._live.discard(rid)

    def live_slots(self) -> int:
        """假驱动也如实报告"当前仍持状态的序列数"。

        假驱动**没有真的 KV**，但它必须参与同一个生命周期契约：
        "完成的请求被及时回收" 这条性质要在本机就能被断言，
        而不是等上了真机才发现峰值驻留随请求总数增长。
        """
        return len(self._live)

    def sample_device(self) -> dict[str, Any]:
        return {}

    def stats(self) -> dict[str, Any]:
        return {
            "n_forward_calls": self._n_forward_calls,
            "retired": list(self.retired),
            "live_slots": self.live_slots(),
        }


# --------------------------------------------------------------------------- #
# 真机驱动
# --------------------------------------------------------------------------- #

#: 精度声明 -> torch dtype 名。用**名字**而不是直接存 dtype 对象，
#: 是为了让本模块在不 import torch 的情况下也能被导入——纯逻辑层与单测依赖这一点。
_DTYPE_NAMES = {
    "fp16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}

#: 本驱动**真的会执行**的取值。表外的取值一律在构造/装配时报错，绝不静默接受。
#:
#: 为什么单列一张表而不是靠"实现里没写就算不支持"：
#: 这类字段在配置里是**元数据标签**，会直接进 meta.json 的 reproducibility 段。
#: 于是"配置写了 w8a8、实际按 bf16 跑"会产出一份自称量化过的记录——
#: 数值全对、字段齐全、没有任何异常，只有结论是错的（而且下游会拿它做
#: "量化收益"的分组）。宁可当场报错，也不要让一个不存在的条件进入元数据。
#:
#: 加新取值时必须**同时**在这里加名字并在实现里落地对应行为；
#: 只改这里不改实现，等于换了个地方静默。
_SUPPORTED_QUANTIZATION = ("none",)
_SUPPORTED_GRAPH_MODE = ("eager",)
_SUPPORTED_TP_SIZE = (1,)

#: 配置里出现这些键即视为"声明了某个未实现的能力"，做显式拒绝。
_UNIMPLEMENTED_RUNTIME_KEYS = ("capture", "quantization_detail")

#: 这些键在配置里取"未生效"的取值时不算声明，因此放行。
#: 例如 ``kv_cache_dtype: auto`` 是**默认值**，它表达的是"跟权重精度走"，
#: 而不是"我要用某种量化过的 KV 缓存"。把所有出现过的键都当成声明，
#: 会让默认配置直接跑不起来——闸门误伤默认值之后就没人会留着它。
_NOOP_RUNTIME_VALUES = {
    "kv_cache_dtype": ("", "auto", "none", "default"),
}


def _build_past_for_hf(torch: Any, kv: Sequence[tuple[Any, Any]] | None) -> Any:
    """把我们内部的 ``list[(key, value)]`` 交给当前 Transformers 能吃的形态。

    Transformer 的 KV 容器在不同版本间换过形态，而**两者都要求
    "每层 K 与 V 分开"**这一条始终成立（这正是 :class:`_Slot` 的存储格式）：

    - ≥ 4.48：``past_key_values`` 必须是 ``Cache`` 对象（``DynamicCache``）。
      传 list 会在 ``cache_position`` 计算处报
      ``'list' object has no attribute 'get_seq_length'``——报错位置在
      transformers 内部，看不出是"格式不对"。
    - ≤ 4.47：直接吃 ``tuple[tuple[key, value]]``。

    这里按能力探测而不是按版本号猜：有 ``from_legacy_cache`` 就用它，
    否则按老 tuple 格式给。**探测不到就报错**，不退化成"随手传一个"。
    """
    if kv is None:
        return None
    if not kv:
        # 空历史：等价于"没有缓存"。传空 tuple 而不是空 Cache，
        # 因为 DynamicCache 空对象在部分版本里会走另一条分支。
        return () if _cache_needs_object(torch) is False else _empty_cache(torch)
    legacy = tuple((k, v) for k, v in kv)
    if _cache_needs_object(torch):
        from transformers.cache_utils import DynamicCache  # noqa: PLC0415

        return DynamicCache.from_legacy_cache(legacy)
    return legacy


def _cache_utils_module() -> Any:
    try:
        from transformers import cache_utils  # noqa: PLC0415
    except Exception:
        return None
    return cache_utils


def _cache_needs_object(torch: Any) -> bool:
    """当前 Transformers 是否要求 ``past_key_values`` 是一个 ``Cache`` 对象。"""
    mod = _cache_utils_module()
    if mod is None:
        return False
    dyn = getattr(mod, "DynamicCache", None)
    return dyn is not None and hasattr(dyn, "from_legacy_cache")


def _empty_cache(torch: Any) -> Any:
    from transformers.cache_utils import DynamicCache  # noqa: PLC0415

    return DynamicCache()


def _kv_from_hf(torch: Any, past: Any, n_layers: int) -> list[tuple[Any, Any]]:
    """把 ``model(...)`` 返回的 ``past_key_values`` 归一到 ``list[(key, value)]``。

    两种形态都要认：``Cache`` 对象（``to_legacy_cache``）与老的 tuple。
    **认不出来时报错而不是猜**：猜错的表现是"KV 静默变成别人的历史"，
    只有数值对比能发现。
    """
    if past is None:
        raise RuntimeError("驱动拿到 past_key_values=None；use_cache=True 时应总是有值")
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    try:
        items = list(past)
    except TypeError as exc:  # pragma: no cover - 形态未知
        raise RuntimeError(
            f"无法解析 past_key_values 的形态（{type(past).__name__}）："
            "需要 Cache 对象或 tuple[tuple[key, value]]"
        ) from exc
    if len(items) != n_layers:
        raise RuntimeError(
            f"past_key_values 的层数（{len(items)}）与模型层数（{n_layers}）不一致"
        )
    out: list[tuple[Any, Any]] = []
    for layer_idx, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise RuntimeError(
                f"第 {layer_idx} 层的 KV 不是 (key, value) 二元组，"
                f"而是 {type(item).__name__}；K 与 V 必须分开取出"
            )
        out.append((item[0], item[1]))
    return out


def _musa_module(torch: Any) -> Any:
    """取 ``torch.musa``，没有就返回 ``None``。

    单独抽出来是为了让"设备相关的每一处"都走同一个判据：``torch.musa`` 是
    ``import torch_musa`` 注册进来的，不是 torch 自带的——任何一处漏判都会在
    无 MUSA 的环境里以 ``AttributeError`` 的形式炸掉，而不是干净地走"设备不可用"。
    """
    return getattr(torch, "musa", None)


@dataclass
class _Slot:
    """一个序列的设备侧状态。

    ``past`` 是**不补位**的真实 KV：``list[(key, value)]``，每层一项，
    ``key`` / ``value`` 各自是 ``[1, n_kv_heads, past_len, head_dim]``。

    ⚠️ **K 与 V 必须分开存**。HF 的 ``past_key_values`` 是
    ``list[layer] -> (key, value)`` 的嵌套结构；把每层的 ``(key, value)`` 当成
    一个张量来切片/补位，形状会凑巧"看起来"能用（元组被当成序列），
    实际拿到的是一段错位的缓存——首次前向就会失败或算错，
    而报错位置远离真正的原因。本文件里凡是碰 KV 的地方都按 ``[layer][0]``
    = K、``[layer][1]`` = V 写。

    补位只在每次前向的临时张量上做，用完即弃——这样无效 KV 不会随步数累积。
    但**每个活跃序列的 ``past`` 是常驻的**，其占用与 ``prompt_len + gen_len``
    成正比，因此请求完成时必须由 :meth:`TransformerDriver.retire` 及时摘除。
    """

    req_id: str
    prompt: list[int]
    #: 每层的 (key, value)。**分开的**两个张量，不是"打包在一起的 KV"。
    past: list[tuple[Any, Any]] = field(default_factory=list)
    past_len: int = 0
    last_token: int | None = None
    emitted: int = 0

    def take_next(self, k: int) -> list[int]:
        """取本次要喂入的 ``k`` 个 token。

        prompt 还没算完就从 prompt 里取；算完之后喂上一次产出的 token
        （decode 一步一个）。这条分界由 ``past_len`` 与 prompt 长度决定，
        而不是由调用方另行告知——**分块预填充下"prompt 算完没有"正是
        past_len 与 prompt_len 的关系**，两处判断迟早会不一致。
        """
        if k <= 0:
            raise ValueError(f"take_next 的 k 必须为正，收到 {k}")
        if self.past_len < len(self.prompt):
            seg = self.prompt[self.past_len : self.past_len + k]
            if len(seg) < k:
                raise ValueError(
                    f"序列 {self.req_id} 本次要喂 {k} 个 token，但 prompt 只剩 "
                    f"{len(seg)} 个（past_len={self.past_len}, "
                    f"prompt_len={len(self.prompt)}）；prompt 算完后应改为 decode。"
                )
            return seg
        if self.last_token is None:
            raise ValueError(
                f"序列 {self.req_id} 的 prompt 已算完（past_len={self.past_len}），"
                "但还没有产出过任何 token，无法继续 decode。"
            )
        if k != 1:
            raise ValueError(
                f"序列 {self.req_id} 处于 decode 阶段，一次只能推进 1 个 token，"
                f"收到 {k}。"
            )
        return [self.last_token]

    def record_token(self, token: int) -> None:
        self.last_token = token
        self.emitted += 1


@dataclass
class _GroupOutcome:
    """单次子前向的结果（补位、设备计算、取回三段耗时分开记）。

    **两个时刻必须分开**（``logits_ready_s`` vs ``ready_s``）：这是本轮修掉的
    一个真实偏差。设备把 logits 算完（同步返回）**不等于** token 可得——
    采样（argmax / multinomial）本身是一次设备计算，``int(device_tensor[i])``
    又是一次设备→主机的同步拷贝。早先只记一个 ``ready_s``，且记在采样**之前**，
    于是下游拿它算 TTFT / E2E / 逐步间隔（stall、ITL）时**系统性偏早**，
    偏早量 = 采样 + 拷回耗时，并且随批内行数增长（批量越大，逐元素拷回越多）。
    这个偏差不报错、不留痕，还会被读成"设备很快"。
    """

    pack_ms: float
    fwd_ms: float
    #: 设备算完（同步返回）那一刻。**不是** token 可得时刻。
    logits_ready_s: float
    #: **token 值已在主机侧可得**那一刻：采样与设备→主机拷贝都完成之后。
    #: 下游算 TTFT / E2E / 逐步间隔必须用这个，不能用 ``logits_ready_s``。
    ready_s: float
    produced: dict[str, int | None]
    #: 不产出 token 的序列所属组**结束**的时刻。
    #:
    #: 为什么不给它们一个 ``ready_s``：token 是"算完那一刻可得"的，而
    #: "算完但没有 token"对后续步骤没有意义——真正的信号是**这一步何时结束**。
    #: 多组批里若把它记成该组的 ready_s，后面的组会把时间往前推，
    #: 于是整步的结束时刻被记成"第一次子前向结束"——比实际早，
    #: 步耗时偏低，而"偏低"看起来完全正常。
    last_ready_s: float = 0.0
    #: `req_id -> logits 摘要`（仅在驱动开了 `capture_logits_digest` 时非空）。
    #:
    #: 摘要含 `argmax` / `max` / `sum` / `sha256`；**shа256 是逐位判据**，
    #: 其余三个只为便于人读。默认空 dict ⇒ 既有调用方拿到的对象行为不变。
    logits_digests: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def readback_ms(self) -> float:
        """采样 + 取回主机的耗时（ms）——即"偏早"的量级本身。"""
        return (self.ready_s - self.logits_ready_s) * 1000.0


class TransformerDriver:
    """HuggingFace Transformers + MUSA 的进程内驱动。

    torch / transformers 一律**在方法里** import：本模块会被
    ``benchmark/run_benchmark.py`` 在 --dry-run 路径上导入，
    那种场合不应该要求装 torch。

    一次 :meth:`forward` 的处理顺序：

    1. 按"本次喂入的 token 数"把批分成形状组（同组内补位量为零）；
    2. 组内按最长 history 左补位，补位区用 attention mask 置 0 屏蔽；
    3. ``model(...)`` 一次前向；
    4. ``synchronize`` —— **此刻** token 才真的可得，以此打点；
    5. 把补位切掉，写回各序列自己的 KV。
    """

    hardware_backed = True

    def __init__(
        self,
        *,
        model_name: str,
        device: str = "musa:0",
        precision: str = "fp16",
        model_revision: str | None = None,
        max_seq_len: int = 32768,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int = 0,
        quantization: str = "none",
        graph_mode: str = "eager",
        tensor_parallel_size: int = 1,
        # ---- 执行优化开关（S01/P1，默认全关 ⇒ 既有行为逐字节不变）----
        #: 只算**最后位置**的 logits（`num_logits_to_keep=1`）。服务协议只用最后位置的
        #: 下一个 token，其余位置的输出投影是纯冗余（Qwen2 词表 152064 ⇒ 输出层很宽）。
        logits_last_only: bool = False,
        #: 只对 `need_token=True` 的行做采样 / D2H。中间 chunk 不产出 token，
        #: 对它们采样再回传是白做的（且会拖长 TTFT）。
        sample_needed_only: bool = False,
        # ---- 执行优化开关（S02/P2a，默认 = 原行为）----
        #: 常规 retire 时是否清 allocator 缓存。**默认 True = 原行为**；
        #: 设 False 时改为由 `flush_allocator_cache()` 在 run 边界显式调用。
        retire_empty_cache: bool = True,
        # ---- 执行优化开关（S03/P3，默认全关）----
        #: QKV 三投影融合（3→1）。权重在装载时沿输出维拼接，**不**每 token 拼。
        fuse_qkv: bool = False,
        #: gate/up 两投影融合（2→1）。
        fuse_gate_up: bool = False,
        #: 融合的**形状选择**：只在"本步 token 数"落在集合内时走融合路径，其余回退。
        #: `None` ⇒ 所有形状都融合（不做选择）。
        linear_fuse_shapes: Sequence[int] | None = None,
        # ---- 执行优化开关（S04/P2b-c，默认关闭）----
        #: 纯 decode 且**历史等长**的批走"常驻 KV 缓冲"快路径：每槽位预分配一块
        #: 固定容量缓冲，每步用 `copy_` 原地写入新 KV，再用 `narrow` 出视图拼批，
        #: 从而免掉逐层逐行的 `F.pad` + 切片重建（`pack_ms` 的主要来源）。
        #: **默认 False = 原行为**；成员退出/补入、混长、非纯 decode 一律回退原路径。
        persistent_kv_slots: bool = False,
        #: 常驻缓冲的**预分配容量**（每槽位 token 数）。`None` ⇒ 按 `max_seq_len`。
        #: ⚠️ 该容量**计入内存预算**，产物里如实报告申请多少字节。
        persistent_kv_capacity: int | None = None,
        # ---- 数值验证开关（S04 重入验证；**默认关闭 = 原行为**）----
        #: 每步把各行的**末位置 logits** 压成一个摘要（argmax / max / sum / sha256）
        #: 挂到 `_GroupOutcome.logits_digests` 上。
        #:
        #: 为什么需要它：`_GroupOutcome` 原本只回传 `produced`（token id）。而
        #: "KV 读错了一段"这类缺陷**可以**在不改变 argmax 的情况下发生（top-1 恰好
        #: 相同、次优项已经错了）—— 只看 token 相等会漏。logits 摘要提供了
        #: **比 token 更早暴露**的对照量（见 PLAYBOOK：「逐层 KV 对照比 token 对照
        #: 更早暴露」）。
        #:
        #: 为什么不直接回传 logits 张量：Qwen2 词表 152064，fp16 下每行 ~300 KB，
        #: 逐步 × 多行 × 两臂全量留存会让产物膨胀到不可用；而**摘要**足够回答
        #: "两臂在同一步同一行的 logits 是否逐位相同"。
        #: 摘要里同时给 sha256 与几个标量：标量便于人读，sha256 是**逐位**判据。
        capture_logits_digest: bool = False,
        model: Any = None,
        tokenizer: Any = None,
        prompt_source: SyntheticPromptSource | None = None,
        clock: Callable[[], float] = _default_clock,
    ) -> None:
        key = precision.strip().lower()
        if key not in _DTYPE_NAMES:
            raise ValueError(f"未知精度 {precision!r}；可用：{sorted(_DTYPE_NAMES)}")
        # 不支持的能力**在这里就拒绝**。放在构造函数而不是 build_driver 里，
        # 是因为直接构造驱动的调用方（测试、脚本、以后的其它工厂）同样不该
        # 拿到一个"声明了但没做"的驱动。报错信息里给出可用取值，可照抄修改。
        self._requested = {
            "quantization": str(quantization or "none").strip().lower(),
            "graph_mode": str(graph_mode or "eager").strip().lower(),
            "tensor_parallel_size": int(tensor_parallel_size or 1),
        }
        self._reject_unimplemented(self._requested)
        self.model_name = model_name
        self.device = device
        self.precision = key
        #: 配置里**请求**的 revision（通常就是引用 `main`）。
        self.model_revision = model_revision
        #: 装载后**解析到**的权重 commit。装载之前一律为 ``None``。
        #:
        #: 与 `model_revision` 是两件事：前者是引用，今天指向 A、明天可能指向 B；
        #: 只有这个值能把一次 run 钉到具体权重上。取不到就是 ``None``，**不猜**。
        self.resolved_model_revision: str | None = None
        self.max_seq_len = int(max_seq_len)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self.clock = clock
        # 允许注入模型/分词器/词源：本地用微型随机模型做**数值等价性验证**时必需。
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_source = prompt_source
        self._loaded = False
        self._slots: dict[str, _Slot] = {}
        self._n_layers = 0
        self._n_forwards = 0
        self._n_logical_steps = 0
        self._pack_s = 0.0
        self._fwd_s = 0.0
        #: 采样 + 设备→主机取回的累计耗时。它**不计入 fwd_ms**（那是设备计算），
        #: 但**计入** TTFT 等时延——单独记出来是为了让"偏早量"可被观测而非隐形。
        self._readback_s = 0.0
        self._tokens_out = 0
        self._max_batch = 0
        self._bytes_per_param: int | None = None
        self._load_info: dict[str, Any] = {}
        self._warmup_info: dict[str, Any] = {}
        #: 预热结束那一刻的累加器快照 ⇒ 供 `stats()` 给出"仅正式测量段"的派生量。
        #: `None` = 尚未预热（此时 `*_total` 就已是正式段口径）。
        self._warmup_baseline: dict[str, float] | None = None

        # ---- 执行优化（S01/P1 · S02/P2a · S03/P3）的请求与生效状态 ----
        # ⚠️ 「请求」与「生效」**必须分开存**：`num_logits_to_keep` 这类参数的可用性
        # 取决于模型/transformers 版本，配置里开着而实际没生效时，产物若只写
        # 「请求值」，读者会把一次**未优化**的 run 当成优化 run 来解读。
        self._logits_last_only_requested = bool(logits_last_only)
        self._sample_needed_only = bool(sample_needed_only)
        self._retire_empty_cache = bool(retire_empty_cache)
        self._fuse_qkv_requested = bool(fuse_qkv)
        self._fuse_gate_up_requested = bool(fuse_gate_up)
        self._linear_fuse_shapes = (
            None if linear_fuse_shapes is None else tuple(int(x) for x in linear_fuse_shapes)
        )
        #: `True/False` = 已判定；`None` = 尚未判定（装载时判定一次）。
        self._num_logits_keep_support: bool | None = None
        self._logits_last_only_effective = False
        self._num_logits_keep_unsupported_reason = ""
        self._n_calls_logits_last_only = 0
        self._n_calls_logits_all_positions = 0
        self._n_sampled_rows = 0
        self._n_sampling_rows_skipped = 0
        self._n_retire_empty_cache_calls = 0
        self._n_retire_empty_cache_suppressed = 0
        #: S03 融合清单（`linear_fusion.FuseManifest`）；未启用时为 `None`。
        self._fusion: Any = None
        # ---- S04/P2b-c：常驻 KV 缓冲（默认关闭）----
        self._persistent_kv_requested = bool(persistent_kv_slots)
        #: 数值验证：是否每步采集 logits 摘要（默认关 ⇒ 既有行为逐字节不变）。
        self._capture_logits_digest = bool(capture_logits_digest)
        self._n_logits_digests = 0
        self._persistent_kv_capacity_cfg = (
            None if persistent_kv_capacity is None else int(persistent_kv_capacity)
        )
        #: 逐槽位的常驻缓冲：`{req_id: [(K_buf, V_buf), ...]}`，每层一项。
        #: 缓冲形状 `[1, n_kv, capacity, head_dim]`——**容量固定**，
        #: 有效长度由 `_Slot.past_len` 决定，两者分开记（不靠形状猜长度）。
        self._kv_buf: dict[str, list[tuple[Any, Any]]] = {}
        #: ★ 每个槽位缓冲的**已同步长度**：`_pkv_synced_len[req_id] = L` 表示
        #: "缓冲的前 L 行与 `_Slot.past` 的内容逐位一致"。
        #: 为什么必须有：**回退路径只更新 `s.past`，不碰缓冲**（回退要让通用路径
        #: 保持原样、不加额外拷贝）。于是「快路径 → 回退 → 再进快路径」时，
        #: 缓冲里是**过期内容**，而形状合法、不报错 ⇒ 数值悄悄错。
        #: 光看"缓冲是否存在"判不出来（这正是原实现 `if buf is not None: continue`
        #: 的漏洞），必须比较长度；长度只增 ⇒ `synced != past_len` 即"脏"。
        self._pkv_synced_len: dict[str, int] = {}
        #: 快路径实际生效步数 / 因不满足判据而回退步数（**都要可数出来**）。
        self._n_pkv_fastpath_steps = 0
        self._n_pkv_fallback_steps = 0
        #: ★ **重入重同步次数**（缓冲过期、被迫整段重搬）。它是快/回退交替的真实成本，
        #: 必须可数出来；否则"回退不碰缓冲"这条设计选择的代价会隐形。
        self._n_pkv_resyncs = 0
        #: 回退原因计数（按原因分桶；"回退"本身不是失败，但要能解释）。
        self._pkv_fallback_reasons: dict[str, int] = {}
        #: 缓冲申请的总字节数（**当前驻留**：`retire` 会扣减）。
        self._pkv_bytes_allocated = 0
        #: ★ 历史峰值与累计申请次数。为什么必须有：`bytes_allocated` 是**当前驻留**，
        #:   而 run 结束时所有请求都已 `retire` ⇒ 终态**恒为 0** ⇒ 它无法区分
        #:   「从未分配」与「分配后全部释放」（EXECOPT4 实测：跑了 193 步快路径，
        #:   报告里 `bytes_allocated=0`）。峰值 + 次数才回答"到底有没有分配过"。
        self._pkv_bytes_peak = 0
        self._pkv_n_allocations = 0
        #: 每个槽位的缓冲字节数（容量固定 ⇒ 逐槽相同，供 `retire` 精确扣减）。
        self._pkv_bytes_per_slot = 0
        self._pkv_capacity: int | None = None
        self._pkv_n_kv_heads: int | None = None
        self._pkv_head_dim: int | None = None
        self._pkv_dtype: str | None = None

    # ---- 能力：请求了什么 vs 什么在生效 --------------------------------
    @staticmethod
    def _reject_unimplemented(requested: dict[str, Any]) -> None:
        """拒绝本驱动未实现的运行时能力。

        单独成一个静态方法，是为了让"什么没实现"这件事只有一个地方声明：
        校验与报错文案都从这里出，不会出现"某条路径漏检一个字段"。
        """
        q = requested["quantization"]
        if q not in _SUPPORTED_QUANTIZATION:
            raise ValueError(
                f"本驱动不支持 quantization={q!r}（可用：{list(_SUPPORTED_QUANTIZATION)}）。\n"
                "  Transformers 路径不做权重量化；接受这个声明会让产物自称"
                "「已按 w8a8 运行」，而实际按未量化精度跑——数字全对、字段齐全、"
                "只有结论是错的。\n"
                "  要测量化请走对应的引擎路线，或把 runtime.quantization 改回 \"none\"。"
            )
        g = requested["graph_mode"]
        if g not in _SUPPORTED_GRAPH_MODE:
            raise ValueError(
                f"本驱动不支持 graph_mode={g!r}（可用：{list(_SUPPORTED_GRAPH_MODE)}）。\n"
                "  eager 之外的捕获模式（如 acl_graph）需要在实现里真正做捕获/重捕获，"
                "并把 capture_count 一类证据落盘；只写配置不实现等于把"
                "「图模式没生效」记成「图模式跑了」。\n"
                "  请把 runtime.graph_mode 改回 \"eager\"，或先实现捕获再放开这里。"
            )
        tp = requested["tensor_parallel_size"]
        if tp not in _SUPPORTED_TP_SIZE:
            raise ValueError(
                f"本驱动不支持 tensor_parallel_size={tp!r}"
                f"（可用：{list(_SUPPORTED_TP_SIZE)}，即单卡）。\n"
                "  多卡张量并行需要真正的权重切分与跨卡通信；当前每层前向都在单卡上，"
                "接受这个声明会让一次单卡 run 在 meta 里记成 2 卡/4 卡并行，"
                "而吞吐却只有单卡的水平。\n"
                "  请把 runtime.tensor_parallel_size 设为 1。"
            )

    @property
    def requested(self) -> dict[str, Any]:
        """配置**请求**的运行能力（原样搬运，供产物对照）。"""
        return dict(self._requested)

    @property
    def effective_quantization(self) -> str:
        return "none"

    @property
    def effective_graph_mode(self) -> str:
        return "eager"

    @property
    def effective_tensor_parallel_size(self) -> int:
        return 1

    @property
    def effective(self) -> dict[str, Any]:
        """**实际生效**的运行能力。

        与 :attr:`requested` 分开返回、分开落盘：只有"请求 = 生效"时，
        元数据里的 quantization / graph_mode / tensor_parallel_size 才可以被
        当成实验条件读。任何一天实现落地了某项能力，这里就要跟着改——
        而改动点只有这一处，不会漏。
        """
        return {
            "quantization": self.effective_quantization,
            "graph_mode": self.effective_graph_mode,
            "tensor_parallel_size": self.effective_tensor_parallel_size,
        }

    # ---- 执行优化：请求了什么 vs 什么在生效 ------------------------------
    def _detect_num_logits_to_keep(self) -> tuple[bool, str]:
        """判定当前模型是否支持 `num_logits_to_keep`。返回 `(支持, 原因/说明)`。

        **必须显式判定，不能"传了再看"**：不支持的模型会忽略或报错，而"配置里
        开着、实际没生效"如果只落盘请求值，一次未优化的 run 会被读成优化 run。

        判定顺序：
        1. 先看 `forward` 的签名是否**显式声明**该形参（生产路径的 HF 模型如此）；
        2. 签名不可得或只有 `*args, **kwargs`（测试里注入的记录器/包装器）⇒
           用一次**极小的真实前向**探测，`TypeError` 里提到该形参即判不支持。
           ⚠️ 探测发生在 `load()` 之后、计时窗口之前，且不计入 `n_forwards`。
        """
        import inspect  # noqa: PLC0415

        target = self.model
        fn = getattr(target, "forward", None)
        if fn is not None:
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                params = {}
            if "num_logits_to_keep" in params:
                return True, "forward 签名显式声明 num_logits_to_keep"
            if params and not any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            ):
                return False, "forward 签名未声明 num_logits_to_keep"

        # 走到这里：签名不可得，或含 **kwargs（无法静态判定）⇒ 真实探测一次。
        torch, _ = self._torch()
        try:
            with torch.no_grad():
                self.model(
                    input_ids=torch.zeros((1, 1), dtype=torch.long, device=self.device),
                    attention_mask=torch.ones((1, 1), dtype=torch.long, device=self.device),
                    position_ids=torch.zeros((1, 1), dtype=torch.long, device=self.device),
                    use_cache=True,
                    num_logits_to_keep=1,
                )
        except TypeError as exc:
            text = str(exc)
            if "num_logits_to_keep" in text:
                return False, f"探测前向报 TypeError：{text.splitlines()[0][:160]}"
            raise
        return True, "仅能用运行时探测判定（forward 带 **kwargs）⇒ 探测通过"

    @property
    def execution_optimizations(self) -> dict[str, Any]:
        """执行优化开关的**请求值 / 生效值 / 计数**（S01/P1 · S02/P2a · S03/P3 · S04/P2b-c）。

        落盘它是为了回答两个问题，而且**必须能被回答**：
          * 这次 run 到底开没开这些优化（请求 vs 生效）；
          * 它们**真的被执行了多少次**（计数为 0 = 配了但没走到，与"没配"必须可分）。
        """
        fusion = self._fusion.describe() if self._fusion is not None else None
        return {
            "logits_last_only": {
                "requested": self._logits_last_only_requested,
                "effective": self._logits_last_only_effective,
                "unsupported_reason": self._num_logits_keep_unsupported_reason,
                "n_calls_last_only": self._n_calls_logits_last_only,
                "n_calls_all_positions": self._n_calls_logits_all_positions,
            },
            "sample_needed_only": {
                "requested": self._sample_needed_only,
                # 它不依赖模型能力 ⇒ 请求即生效；保留同构结构便于下游统一读。
                "effective": self._sample_needed_only,
                "n_sampled_rows": self._n_sampled_rows,
                "n_sampling_rows_skipped": self._n_sampling_rows_skipped,
            },
            "retire_empty_cache": {
                "requested_at_retire": self._retire_empty_cache,
                "n_empty_cache_calls": self._n_retire_empty_cache_calls,
                "n_suppressed": self._n_retire_empty_cache_suppressed,
            },
            "linear_fusion": fusion,
            # S04/P2b-c：常驻 KV 缓冲。★ **快路径步数与回退步数必须同时非零可数**——
            # 只报"开了"会让一次"每步都回退"的 run 看上去像优化生效了。
            "persistent_kv_slots": {
                "requested": self._persistent_kv_requested,
                # 「生效」= 至少走过一次快路径；与"请求"分开。
                "effective": self._n_pkv_fastpath_steps > 0,
                # ⚠️ **这两个是"前向组"级计数**（`_forward_group` 调一次算一次），
                # 不是逻辑步级。报覆盖率时必须写明分母：见
                # `measured_window.pkv_coverage_of_forward_groups` /
                # `..._of_logical_steps`。混用会让覆盖率看起来高一截。
                "n_fastpath_steps": self._n_pkv_fastpath_steps,
                "n_fallback_steps": self._n_pkv_fallback_steps,
                "fallback_reasons": dict(sorted(self._pkv_fallback_reasons.items())),
                #: 回退造成的**重入重同步**次数（缓冲过期 ⇒ 整段重搬）。真实成本。
                "n_resyncs": self._n_pkv_resyncs,
                "capacity": self._pkv_capacity,
                "n_kv_heads": self._pkv_n_kv_heads,
                "head_dim": self._pkv_head_dim,
                "dtype": self._pkv_dtype,
                #: 预分配缓冲的**当前驻留**字节数（`retire` 会扣减 ⇒ 收尾时通常为 0）。
                "bytes_allocated": self._pkv_bytes_allocated,
                #: ★ **峰值**驻留与累计申请次数：`bytes_allocated` 终态恒 0，
                #:   单看它无法区分「从未分配」与「分配后全部释放」。判"到底分配过没有"
                #:   要看 `n_allocations > 0` 且 `bytes_peak > 0`。
                "bytes_peak": self._pkv_bytes_peak,
                "n_allocations": self._pkv_n_allocations,
            },
            # 数值验证用（默认关）。计数为 0 = 配了但没走到，与"没配"可分。
            "capture_logits_digest": {
                "requested": self._capture_logits_digest,
                "effective": self._capture_logits_digest,
                "n_logits_digests": self._n_logits_digests,
            },
        }

    # ---- 身份 -----------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "kind": "transformers",
            "hardware_backed": True,
            "model": self.model_name,
            "model_revision": self.model_revision,
            #: 解析后的权重 commit。与 `model_revision`（引用）分开记：
            #: 只记引用的话，"这批评数是哪个权重跑出来的"仍然无法回答。
            "model_revision_resolved": self.resolved_model_revision,
            "device": self.device,
            "precision": self.precision,
            "sampling": "sampled" if self.do_sample else "greedy",
            "max_seq_len": self.max_seq_len,
            # 实际生效的执行能力。**没有这三个键，产物就无法区分**
            # "按 eager 未量化跑" 与 "配置里声明了量化/图模式但没实现"——
            # 而两者在 meta 里长得一模一样（都写着 w8a8 / acl_graph）。
            "effective": {
                "quantization": self.effective_quantization,
                "graph_mode": self.effective_graph_mode,
                "tensor_parallel_size": self.effective_tensor_parallel_size,
            },
            # 执行优化开关（S01/S02/S03）的请求/生效/计数。缺了它，
            # 产物就无法区分"这次跑的是优化路径"与"配了但没生效"。
            "execution_optimizations": self.execution_optimizations,
        }

    # ---- 装载 -----------------------------------------------------------
    def load(self) -> dict[str, Any]:
        torch, _ = self._torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        if self.model is None:
            if not self.model_name:
                raise ValueError("TransformerDriver 需要 model_name 或注入 model")
            kwargs: dict[str, Any] = {"torch_dtype": self._dtype(torch)}
            if self.model_revision:
                kwargs["revision"] = self.model_revision
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, revision=self.model_revision or "main"
            )
        else:
            self._dtype(torch)  # 注入模型时也要确定字节数

        # 解析**实际加载到**的权重 commit。HF 在 `from_pretrained` 成功后会把
        # snapshot 的 commit 留在 `config._commit_hash` 上；这是唯一能把一次 run
        # 钉到具体权重的值（配置里的 `revision: main` 只是一个引用，不是版本）。
        # 本地路径加载 / 注入模型时取不到，那就是 `None`——**不猜、不回落成 `main`**。
        self.resolved_model_revision = _resolved_revision(self.model)

        # 设备必须真的存在。**不静默退回 CPU**：那会把本机 CPU 时延写成
        # "设备实测时延"，而产物里没有任何字段能暴露这件事。
        if not self._device_available(torch):
            raise RuntimeError(
                f"设备 {self.device!r} 不可用（torch.musa.is_available() 为假）。\n"
                "  真实后端不会退回 CPU——那会把 CPU 时延记成设备实测。\n"
                "  在 MTT 主机上请先确认 torch_musa 已 import"
                "（torch.musa 由它注册），并核对设备号。"
            )

        if self.prompt_source is None:
            self.prompt_source = SyntheticPromptSource(
                vocab_size=self._vocab_size(),
                seed=self.seed,
                excluded_ids=self._special_ids(),
            )

        mem_before = self._device_mem(torch)
        self.model.to(self.device)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        # 层数从模型配置上取一次并记住：写回 KV 时要按层逐个切，
        # 而 "len(out.past_key_values)" 在 Cache 对象上未必等于层数
        # （它可能只含已写入的层）。用模型自己的层数做校验，而不是相信返回值。
        self._n_layers = self._model_n_layers()
        # S04/P2b-c：常驻缓冲容量在**装载期**定一次（默认 = `max_seq_len`，
        # 与"最坏情况每请求要占多少 token"一致）。放到前向里定会让第一次前向
        # 额外分配一大块缓冲，把一次分配开销混进正式测量段。
        if self._persistent_kv_requested:
            self._pkv_capacity = (
                int(self._persistent_kv_capacity_cfg)
                if self._persistent_kv_capacity_cfg is not None
                else int(self.max_seq_len)
            )
            if self._pkv_capacity <= 0:
                raise ValueError(
                    f"persistent_kv_capacity 必须为正，收到 {self._pkv_capacity}；"
                    "容量为 0 时每一步都会判 `capacity_exceeded` 而全程回退——"
                    "这会表现为「开了优化但没有收益」，与「优化本身无用」不可分。"
                )
        # ---- 执行优化：装载期一次性判定 / 安装（都在计时窗口之外）----
        # 判定必须发生在**装载期**：放到第一步前向里会把一次探测混进预热/正式段，
        # 而"多用一次前向"这类污染在时延数字上看起来与优化无关。
        if self._logits_last_only_requested:
            ok, why = self._detect_num_logits_to_keep()
            self._num_logits_keep_support = bool(ok)
            self._logits_last_only_effective = bool(ok)
            self._num_logits_keep_unsupported_reason = "" if ok else why
        if self._fuse_qkv_requested or self._fuse_gate_up_requested:
            from linear_fusion import install_fusions  # noqa: PLC0415
            self._fusion = install_fusions(
                self.model,
                fuse_qkv=self._fuse_qkv_requested,
                fuse_gate_up=self._fuse_gate_up_requested,
                shapes=self._linear_fuse_shapes,
            )
        mem_after = self._device_mem(torch)
        self._loaded = True
        self._load_info = {
            "n_params": n_params,
            "n_params_billion": round(n_params / 1e9, 4),
            "param_bytes_theoretical_mib": round(
                n_params * (self._bytes_per_param or 0) / (1024 * 1024), 1
            ),
            "mem_before_load_mib": mem_before.get("used_mib"),
            "mem_after_load_mib": mem_after.get("used_mib"),
            "mem_total_mib": mem_after.get("total_mib"),
            "dtype": _DTYPE_NAMES[self.precision],
            "torch_version": torch.__version__,
            "prompt_source": self.prompt_source.describe(),
        }
        return self._load_info

    # ---- 预热 -----------------------------------------------------------
    def warmup(
        self, target_s: float, *, batch: int, prompt_len: int, gen_len: int
    ) -> dict[str, Any]:
        """按**时间**预热，直到累计耗时 >= ``target_s``。

        预热用的形状要贴近正式测量段（同样的批大小量级、同样的 prompt/gen 长度），
        否则预热覆盖不到正式段真正会走的 kernel 与显存尺寸，等于没预热。

        预热**在计时窗口之外**：``RealBackend`` 在预热结束后才记时间原点。
        """
        self._assert_loaded()
        batch = max(1, int(batch))
        prompt_len = max(1, int(prompt_len))
        gen_len = max(1, int(gen_len))
        t0 = self.clock()
        iters = 0
        while True:
            # 预热用独立的 req_id 命名空间，绝不与正式请求混在同一个 KV 槽位里。
            tag = f"__warmup_{iters:03d}"
            warm_reqs = [
                Request(
                    req_id=f"{tag}_{i}",
                    prompt_len=prompt_len,
                    gen_len=gen_len,
                    shared_prefix_len=0,
                )
                for i in range(batch)
            ]
            self._drive_warmup_batch(warm_reqs)
            self.retire([r.req_id for r in warm_reqs])
            iters += 1
            if self.clock() - t0 >= target_s:
                break
            if iters >= 10_000:  # 兜底：target_s 极大时不至于无限循环
                break
        elapsed = self.clock() - t0
        # ★ 预热走的是**同一个** forward 累加器（`_drive_warmup_batch` 直接调
        #   `self.forward`）⇒ `stats()` 的 `*_total` 与 `n_forwards`/`tokens_out`
        #   **包含预热**。而预热跑几轮由"累计跑到 target_s"决定 ⇒ **各次运行可能
        #   不同**（EXECOPT4 实测 2 vs 1 轮，差 117 次前向、≈3.7k tokens）。
        #   ⇒ 在此（**预热结束那一刻**）存一份"零点"，`stats()` 另给一组
        #     "仅正式测量段"的键。⚠️ 快照必须在**循环之后**：放在循环之前，
        #     相减等于什么也没扣（本地单测抓到过）。
        #   ⚠️ 既有 `*_total` 的语义**保持不变**（改了会与历史产物不可比）。
        self._warmup_baseline = self._counter_snapshot()
        self._warmup_info = {
            "target_s": target_s,
            "warmup_s": round(elapsed, 3),
            "iterations": iters,
            "batch": batch,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            # 明确标出"没跑到目标时长"：写"预热了 30s"却只跑了 3s
            # 是不会报错的偏差。
            "short_of_target": elapsed < target_s,
        }
        return self._warmup_info

    def _drive_warmup_batch(self, reqs: Sequence[Request]) -> None:
        """把一组预热请求走完 prefill + (gen_len-1) 次 decode。

        走完整流程而不是只做一次 prefill：decode 步的 kernel 形状与 prefill 不同
        （每序列 1 token），只预热 prefill 等于漏掉正式段里绝大多数步。

        **prefill 必须产出首 token**（``need_token=True``），无论 ``gen_len`` 是几：
        完整算完 prompt 之后模型本来就会给出下一个 token，正式路径也依赖这一点
        （``_Slot.take_next`` 在 decode 阶段需要 ``last_token`` 非空）。
        写成 ``need_token=(gen_len == 1)`` 会让 ``gen_len > 1`` 的预热在第一步之后
        就报"还没有产出过任何 token，无法继续 decode"——预热失败，
        而它发生在计时窗口之外，是本轮最先暴露、也最容易被误读成
        "模型有问题"的一处。
        """
        self.forward(
            [ForwardItem(req=r, n_new_tokens=r.prompt_len, need_token=True) for r in reqs]
        )
        for _ in range(max(0, reqs[0].gen_len - 1)):
            self.forward(
                [ForwardItem(req=r, n_new_tokens=1, need_token=True) for r in reqs]
            )

    # ---- 前向 -----------------------------------------------------------
    def forward(self, batch: Sequence[ForwardItem]) -> StepOutcome:
        # 先做便宜的自查，再 import torch：没装载时不该为了报一个错去付
        # 十几次秒级的 import 代价，而且这个顺序让"未装载"的报错不依赖环境。
        self._assert_loaded()
        torch, _ = self._torch()
        if not batch:
            raise ValueError("forward() 收到空批；空批不应发起前向")
        for it in batch:
            if it.n_new_tokens <= 0:
                raise ValueError(
                    f"ForwardItem({it.req.req_id}) 的 n_new_tokens="
                    f"{it.n_new_tokens}；要推进一步就必须喂至少一个 token"
                )

        started = self.clock()
        n_tokens = sum(it.n_new_tokens for it in batch)
        groups = self._shape_groups(batch)

        produced: dict[str, int | None] = {}
        ready_at: dict[str, float | None] = {}
        digests: dict[str, dict[str, Any]] = {}
        pack_ms = 0.0
        fwd_ms = 0.0
        readback_ms = 0.0
        for k, items in groups:
            gout = self._forward_group(torch, items, k)
            pack_ms += gout.pack_ms
            fwd_ms += gout.fwd_ms
            readback_ms += gout.readback_ms
            produced.update(gout.produced)
            digests.update(gout.logits_digests)
            for rid, tok in gout.produced.items():
                # 产出 token 的序列：可得时刻 = **token 值取回主机之后**那一刻
                # （采样与设备→主机拷贝都已完成），不是"设备算完 logits"那一刻。
                # 不产出 token 的序列：给"本组结束时刻"——它是**这一步走到哪了**
                # 的证据；不给 None 是因为后端已经有 need_token 这一路信息，
                # 这里的时刻只用于记账与排障（见 StepTiming.finished_s）。
                ready_at[rid] = gout.ready_s if tok is not None else gout.last_ready_s

        finished = self.clock()
        self._pack_s += pack_ms / 1000.0
        self._fwd_s += fwd_ms / 1000.0
        self._readback_s += readback_ms / 1000.0
        self._n_forwards += len(groups)
        # ★ **第二个分母**：一次 `forward()` 调用 = 一个逻辑步，它可能被
        #   `_shape_groups` 拆成多个前向组（形状不一致时）。`n_forwards` 数的是
        #   **组**，两者在混形状批次下不相等 ⇒ 报"覆盖率"时必须写明用哪个作分母。
        self._n_logical_steps += 1
        self._max_batch = max(self._max_batch, len(batch))
        return StepOutcome(
            timing=StepTiming(
                n_seqs=len(batch),
                n_tokens=n_tokens,
                n_forwards=len(groups),
                pack_ms=pack_ms,
                fwd_ms=fwd_ms,
                started_s=started,
                finished_s=finished,
                readback_ms=readback_ms,
            ),
            slots=[
                SlotResult(
                    it.req.req_id,
                    produced.get(it.req.req_id),
                    ready_at.get(it.req.req_id),
                    digests.get(it.req.req_id),
                )
                for it in batch
            ],
        )

    @staticmethod
    def _shape_groups(
        batch: Sequence[ForwardItem],
    ) -> list[tuple[int, list[ForwardItem]]]:
        """按"本次喂入的 token 数"分组。

        同组内部各序列喂入长度一致，因此不会有任何无效计算。
        形状一致的批（同批起步、或同批 decode 的序列）天然落进同一组，一次前向就够；
        参差的批会被拆成多次前向——**这是 HF 规则张量 KV 的真实代价**，
        所以被显式计数（``n_forwards``）而不是藏起来。
        """
        return group_by_new_tokens(batch)

    def _forward_group(
        self, torch: Any, items: Sequence[ForwardItem], k: int
    ) -> _GroupOutcome:
        import torch.nn.functional as F  # noqa: PLC0415

        pairs = [(self._slot(it.req), it) for it in items]

        # ---- S04/P2b-c：先判快路径能否走（判不了就原路径，且记下原因）----
        pkv_ok = False
        if self._persistent_kv_requested:
            pkv_ok, pkv_reason = self._pkv_eligible(pairs, k)
            if not pkv_ok:
                self._n_pkv_fallback_steps += 1
                self._pkv_fallback_reasons[pkv_reason] = (
                    self._pkv_fallback_reasons.get(pkv_reason, 0) + 1
                )

        # ---- 补位（这一段是显存搬运，不是设备计算，单独计时）----
        t_pack0 = self.clock()
        past_max = max(s.past_len for s, _ in pairs)
        rows_ids: list[list[int]] = []
        rows_mask: list[list[int]] = []
        past_lens: list[int] = []
        for s, _ in pairs:
            # ``rows_ids`` 只放**本次要喂的 token**，不含历史：历史走
            # ``past_key_values``。把历史补进 input_ids 会让不同历史长度的请求
            # 无法组成规则张量（每行长度不同），而且等于把同一段 token 算两遍
            # （一次写进 KV、一次再当输入重算）。
            rows_ids.append(s.take_next(k))
            pad = past_max - s.past_len
            # ``rows_mask`` 描述整段 key 的可见性，宽度**必须是统一的**
            # ``past_max + k``（补位 + 本次输入），不能写成 ``past_max + k``
            # 再各自减去 pad——那样每行宽度不同（19/23/27…），拼不成张量，
            # 直接报 "expected sequence of length N at dim 1"。
            #
            #   行内容 = [0]*pad + [1]*s.past_len + [1]*k
            #          = 补位(屏蔽) + 本行真实历史(可见) + 本次输入(可见)
            #   宽度   = pad + s.past_len + k = past_max + k  ✅ 统一
            #
            # ⚠️ 三种"看起来更自然"的写法都是错的：
            #   ① ``[1]*k``（与 input_ids 同形状）：HF 只从
            #      ``past_key_values.get_seq_length()`` 推 cache_position，
            #      它**假设批内所有行共享同一个 KV 长度**；左补位让短行的真实历史
            #      比 past_max 短，这个差异只能靠 2-D mask 转达。mask 不给全长，
            #      HF 就让补位槽（全零 K/V）也参与注意力 → 每个短行都算错，
            #      而形状全程合法。
            #   ② ``[0]*pad + [1]*past_max + [1]*k``：宽度 = past_max + k + pad，
            #      比 KV 张量宽出一个 pad，报 "size of tensor a … must match …"。
            #   ③ 每行都从 0 开始按 past_max 写（忘了左侧 pad）：真实历史被整体
            #      右移，短行的注意力窗口错了位置，形状仍然合法。
            rows_mask.append([0] * pad + [1] * s.past_len + [1] * k)
            # position_ids 必须反映各行**自己的**历史长度（不是批内最长的）。
            # 用 ``past_max + 列号`` 会让短序列的位置整体前移，
            # 于是它们以为自己的历史更长——形状合法、数值错，只有对比能发现。
            past_lens.append(s.past_len)

        past = None
        if pkv_ok:
            # 历史等长 ⇒ past_max == 各槽 past_len，无补位。
            self._pkv_ensure_buffers(torch, pairs)
            past = self._pkv_build_past(torch, pairs, k)
            # ⚠️ 快路径**不推进** `past_len`：推进必须发生在写回之后
            # （见 `_pkv_writeback`）。在这里提前推进会让写回段算出错的
            # `start`，写进相邻区段——形状合法、不报错、数值错。
        elif past_max > 0:
            past = self._pack_past(torch, F, pairs, past_max)
        pack_ms = (self.clock() - t_pack0) * 1000.0

        # ---- 设备前向（计时到同步之后为止）----
        t_fwd0 = self.clock()
        # 「仅末位置 logits」（S01/P1）：服务协议只用最后位置的下一个 token，
        # 其余位置的输出投影是纯冗余。可用性已在装载期判定并落盘
        # （`execution_optimizations.logits_last_only.effective`）——判不了就不传，
        # **不静默吞参数**。
        _use_last_only = bool(self._logits_last_only_effective)
        fwd_kwargs: dict[str, Any] = {
            "input_ids": self._input_ids(torch, rows_ids),
            "attention_mask": torch.tensor(
                rows_mask, dtype=torch.long, device=self.device
            ),
            # position_ids 与 input_ids 同形状：第 i 行、第 j 个新 token 的
            # 位置 = 该行自己的 past_len + j。**必须逐行用自己的历史长度**。
            "position_ids": self._position_ids(torch, past_lens, k),
            "past_key_values": _build_past_for_hf(torch, past),
            "use_cache": True,
        }
        if _use_last_only:
            fwd_kwargs["num_logits_to_keep"] = 1
        with torch.no_grad():
            out = self.model(**fwd_kwargs)
        if _use_last_only:
            self._n_calls_logits_last_only += 1
        else:
            self._n_calls_logits_all_positions += 1
        # 同步之后 logits 才真的算完。**打点必须在同步之后**：
        # 同步之前记的只是"任务已下发"，那是 CPU 侧时间，不是设备时间。
        self._synchronize(torch)
        logits_ready_s = self.clock()
        fwd_ms = (logits_ready_s - t_fwd0) * 1000.0

        # ⚠️ `logits` 的形状依赖于是否开了"仅末位置"：开了是 `[B, 1, V]`，
        # 不开是 `[B, k, V]`。两种情况都取 `[:, -1, :]` ⇒ 下一 token 的 logits。
        #
        # 「取回主机」的口径不变：``int(device_tensor[i])`` 是一次逐元素设备同步，
        # `_sample` 里的 argmax / multinomial 也还在设备上。所以"logits 算完"与
        # "token 可得"之间隔着采样 + 拷回，把它们记成同一时刻会让 TTFT/E2E/ITL 全部偏早。
        # 一次性把整批搬到主机：既避免逐元素同步（N 次小传输 vs 1 次），
        # 也让 ``ready_s`` 的含义唯一——"token 值可取回主机"。
        last_logits = out.logits[:, -1, :]
        # 「只采需要的行」（S01/P1）：中间 chunk 不产出 token，对它们采样再回传
        # 是白做的（且拖长 TTFT）。**按行下标显式映射**，不靠顺序假设——
        # 靠顺序会让"批里第 i 个是不是 need_token"这种假设在混批时不成立。
        if self._sample_needed_only:
            rows = [i for i, (_, it) in enumerate(pairs) if it.need_token]
            next_host: list[int | None] = [None] * len(pairs)
            if rows:
                picked = self._sample(torch, last_logits[rows, :])
                host = [int(v) for v in picked.detach().to("cpu").tolist()]
                for pos, i in enumerate(rows):
                    next_host[i] = host[pos]
            self._n_sampled_rows += len(rows)
            self._n_sampling_rows_skipped += len(pairs) - len(rows)
        else:
            next_ids = self._sample(torch, last_logits)
            next_host = [int(v) for v in next_ids.detach().to("cpu").tolist()]
        ready_s = self.clock()

        # ---- 写回（同样计入补位搬运）----
        t_pack1 = self.clock()
        produced: dict[str, int | None] = {}
        n_layers = self._n_layers
        hf_past = _kv_from_hf(torch, out.past_key_values, n_layers)
        if pkv_ok:
            # 快路径：`copy_` 原地写入常驻缓冲尾部，再让 `s.past` 指向有效前缀视图。
            # ⚠️ **仍然要维护 `s.past`**：它是 `take_next` / 资格判据 / 回退到通用路径
            # 时的唯一数据源。只写缓冲而不更新 `s.past`，一旦本槽位下一步回退到
            # 通用路径，`_pack_past` 会拿到陈旧（短一截）的KV —— 输出会悄悄错。
            self._pkv_writeback(torch, pairs, hf_past, k)
            for s, _it in pairs:
                keep = s.past_len
                buf = self._kv_buf[s.req_id]
                s.past = [
                    (buf[layer][0][:, :, :keep, :], buf[layer][1][:, :, :keep, :])
                    for layer in range(len(buf))
                ]
                # 写回已把缓冲填到 `past_len` ⇒ 记下"此刻缓冲与 `s.past` 一致"。
                self._pkv_synced_len[s.req_id] = s.past_len
            self._n_pkv_fastpath_steps += 1
        for i, (s, it) in enumerate(pairs):
            if pkv_ok:
                # 快路径已在上面统一推进 `past_len` 与 `s.past`；这里只补产 token。
                if it.need_token:
                    token = next_host[i]
                    s.record_token(token)
                    produced[s.req_id] = token
                    # ★ **快路径也必须计 `tokens_out`**：它原先只在下面的通用分支
                    #   累加 ⇒ 快路径产出的 token 全被漏记。实测（CPU 微型模型）
                    #   实际 17 个 token、`tokens_out` 只记 9 个；真机上这一整批
                    #   的快路径步都少记。计数漏记会让下游"每 token 偏早量"
                    #   之类派生量整体偏大，且**不报错**。
                    self._tokens_out += 1
                else:
                    produced[s.req_id] = None
                continue
            pad = past_max - s.past_len
            keep = s.past_len + k
            # ``hf_past[layer]`` 是 **(key, value) 二元组**，必须逐项取：
            # [0] = K，[1] = V。当成一个张量切会拿到错位的缓存（见 _Slot 文档）。
            # 行切片用 [i : i+1] 取本序列那一行；
            # 序列维切片用 [pad : pad + keep] 切掉左侧补位——
            # 写成 [:keep] 会**静默取到别人的历史**：形状对、不报错，
            # 只有数值对比能发现。数值等价性测试专门覆盖这一点。
            s.past = [
                (
                    hf_past[layer][0][i : i + 1, :, pad : pad + keep, :],
                    hf_past[layer][1][i : i + 1, :, pad : pad + keep, :],
                )
                for layer in range(n_layers)
            ]
            s.past_len += k
            if it.need_token:
                # 用**已取回主机**的值，不再在循环里逐元素同步——
                # 那会把"取回"摊进写回段，且让 ready_s 又变得说不清。
                token = next_host[i]
                s.record_token(token)
                produced[s.req_id] = token
                self._tokens_out += 1
            else:
                produced[s.req_id] = None
        pack_ms += (self.clock() - t_pack1) * 1000.0
        finished_s = self.clock()
        # ---- 数值验证：采集本步各行的 logits 摘要（默认关） ----
        # 放在返回前统一算：此时 `last_logits` 已经拿到，且采样/写回都已完成，
        # 不会影响 `ready_s` 的计时口径（摘要计算发生在"token 可得"之后）。
        digests: dict[str, dict[str, Any]] = {}
        if self._capture_logits_digest:
            for i, (s, _it) in enumerate(pairs):
                digests[s.req_id] = self._logits_digest(torch, last_logits[i])
                self._n_logits_digests += 1
        # 一律用关键字构造：位置构造会把新增字段静默错位，
        # ``last_ready_s`` 当初就是这样恒为 0.0 的（而它的文档写着"本组结束时刻"）。
        return _GroupOutcome(
            pack_ms=pack_ms,
            fwd_ms=fwd_ms,
            logits_ready_s=logits_ready_s,
            ready_s=ready_s,
            produced=produced,
            last_ready_s=finished_s,
            logits_digests=digests,
        )

    def _input_ids(
        self, torch: Any, rows_ids: Sequence[Sequence[int]]
    ) -> Any:
        """本次喂入的 token，形状 ``[batch, k]``。

        **不做补位**。补位只存在于 ``past_key_values``（左侧）与推导 position_ids
        用的 ``rows_full`` 里。往 ``input_ids`` 里补零会带来两个后果：
        那些 0 会被当成真实 token 参与注意力与 KV 写入（且 mask 要额外为它们留位），
        以及不同历史长度的请求无法组成规则张量。
        HF 约定「``input_ids`` 是本次新 token，历史走 ``past_key_values``」，
        跟着约定走，两个 mask 才各自有明确的长度。
        """
        return torch.tensor(
            [list(r) for r in rows_ids], dtype=torch.long, device=self.device
        )

    @staticmethod
    def _pack_past(
        torch: Any, F: Any, pairs: Sequence[tuple[_Slot, ForwardItem]], past_max: int
    ) -> list[tuple[Any, Any]] | None:
        """把各槽位的真实 KV 左对齐补到 ``past_max`` 并叠成一个批。

        返回值直接给 ``past_key_values``：``list[(key, value)]`` 每层一项，
        与 HF 的嵌套结构一致——**K 与 V 分别补位、分别 concat**，
        不是把 (key, value) 当成一个张量处理。

        补位区是全零 + attention mask 置 0，因此不参与注意力。
        """
        with_real = next((s for s, _ in pairs if s.past), None)
        if with_real is None:
            return None  # 整组都是新序列：没有历史，不需要 KV
        n_layers = len(with_real.past)
        layers: list[tuple[Any, Any]] = []
        for layer in range(n_layers):
            key_rows: list[Any] = []
            value_rows: list[Any] = []
            for s, _ in pairs:
                if not s.past:
                    # 新序列（past_len=0）与有历史的序列同组时，
                    # 用零张量补齐——不能直接索引空列表。
                    k0 = with_real.past[layer][0][:, :, :0, :]
                    v0 = with_real.past[layer][1][:, :, :0, :]
                    key_rows.append(F.pad(k0, (0, 0, past_max, 0)))
                    value_rows.append(F.pad(v0, (0, 0, past_max, 0)))
                    continue
                sk, sv = s.past[layer]
                if s.past_len == past_max:
                    key_rows.append(sk)
                    value_rows.append(sv)
                else:
                    n_pad = past_max - s.past_len
                    key_rows.append(F.pad(sk, (0, 0, n_pad, 0)))
                    value_rows.append(F.pad(sv, (0, 0, n_pad, 0)))
            layers.append((torch.cat(key_rows, dim=0), torch.cat(value_rows, dim=0)))
        return layers

    # ---- S04/P2b-c：常驻 KV 缓冲快路径 -------------------------------
    def _pkv_eligible(
        self, pairs: Sequence[tuple[_Slot, ForwardItem]], k: int
    ) -> tuple[bool, str]:
        """快路径判据。**返回原因**（空串 = 可走），不返回裸布尔。

        为什么必须带原因：回退是**常态**（成员退出/补入、混长、新序列都会回退），
        只记一个计数的话，"为什么 90% 的步都回退了"无法回答，
        而它看起来和"这个优化没用"一模一样。

        任务书要求的窄条件是**三条同时成立**：
          1. 纯 decode（本组每个 item 只喂 1 个 token）；
          2. 全部槽位**已有历史**（新序列没有可复用的缓冲，也无从"免补位"）；
          3. 全部槽位历史**等长**（= 无补位 ⇒ 这才是 `pack_ms` 的主要来源）。
        另加一条**容量判据**：写入 `past_len + k` 不得越过预分配容量。

        ⚠️ **顺序有意义，且 `new_sequence` 必须排在 `not_pure_decode` 之前。**
        常见的调用形状是"新请求一次性喂入整段 prompt"（`k = prompt_len > 1`），
        此时若先查 `not_pure_decode`，`new_sequence` 这个原因**永远观察不到**
        （实测：《新序列加入批》的场景只报出 `not_pure_decode`）。
        但两者是**不同的阻塞**：新序列即使 `k == 1` 也走不了快路径
        （它没有可复用的缓冲）。归因错了会让"为什么这批回退"无法回答。
        """
        if not pairs:
            return False, "empty_batch"
        # 先查"结构性阻塞"（与 k 无关），再查 k。
        for s, _it in pairs:
            if not s.past:
                return False, "new_sequence"
        if k != 1:
            return False, "not_pure_decode"
        past_lens = []
        for s, it in pairs:
            if it.n_new_tokens != 1:
                return False, "not_pure_decode"
            past_lens.append(s.past_len)
        # 历史等长：这一条排除了"成员补入/退出导致的混长批"。
        if len(set(past_lens)) != 1:
            return False, "unequal_kv_len"
        cap = self._pkv_capacity or 0
        if past_lens[0] + k > cap:
            return False, "capacity_exceeded"
        return True, ""

    def _pkv_ensure_buffers(self, torch: Any, pairs: Sequence[tuple[_Slot, ForwardItem]]) -> None:
        """为批内槽位准备常驻缓冲（只在本槽位首次进入快路径时分配）。

        缓冲形状 `[1, n_kv, capacity, head_dim]`，**每层 K/V 各一块**，
        与 `_Slot.past` 同样的嵌套结构（`[layer] -> (K, V)`）——
        直接复用 `_build_past_for_hf`，两条路径交给 HF 的形态完全一致。
        """
        sample = next(s for s, _ in pairs if s.past)
        n_layers = len(sample.past)
        k0 = sample.past[0][0]
        n_kv, head_dim = int(k0.shape[1]), int(k0.shape[3])
        capacity = self._pkv_capacity or 0
        dtype = k0.dtype
        self._pkv_n_kv_heads = n_kv
        self._pkv_head_dim = head_dim
        self._pkv_dtype = str(dtype).replace("torch.", "")
        for s, _ in pairs:
            buf = self._kv_buf.get(s.req_id)
            synced = self._pkv_synced_len.get(s.req_id)
            if buf is not None and synced == s.past_len:
                # 缓冲已在**本次进入时的长度**上与 `s.past` 一致 ⇒ 无需重搬。
                continue
            fresh = buf is None
            if fresh:
                alloc: list[tuple[Any, Any]] = []
                for _layer in range(n_layers):
                    kb = torch.zeros((1, n_kv, capacity, head_dim),
                                     dtype=dtype, device=self.device)
                    vb = torch.zeros((1, n_kv, capacity, head_dim),
                                     dtype=dtype, device=self.device)
                    alloc.append((kb, vb))
                self._kv_buf[s.req_id] = alloc
                buf = alloc
            # ★ **必须把槽位当前的 `s.past` 搬进缓冲**（新分配时是修复"零张量当历史"，
            #   重入时是修复"过期内容"）。两条都是**形状合法、不报错、数值错**。
            #   判据不看"缓冲在不在"，而是看**已同步长度**——回退路径只推进 `s.past`、
            #   不碰缓冲，长度必然落后（见 `_pkv_synced_len` 的说明）。
            if s.past:
                for layer in range(n_layers):
                    if layer >= len(s.past):
                        break
                    sk, sv = s.past[layer]
                    length = sk.shape[2]
                    buf[layer][0][:, :, :length, :].copy_(sk)
                    buf[layer][1][:, :, :length, :].copy_(sv)
            self._pkv_synced_len[s.req_id] = s.past_len
            if not fresh:
                # 重入造成的**重同步**单独计数：它是真实成本，也必须可数出来
                # （否则"覆盖面小"会被读成"机制没成本"）。
                self._n_pkv_resyncs += 1
                continue
            # 逐字节记出来：预分配容量是**内存代价**，必须计入预算而不是隐形成本。
            per_slot = n_layers * 2 * (1 * n_kv * capacity * head_dim) * self._dtype_bytes(dtype)
            self._pkv_bytes_per_slot = per_slot
            self._pkv_bytes_allocated += per_slot
            self._pkv_n_allocations += 1
            self._pkv_bytes_peak = max(self._pkv_bytes_peak, self._pkv_bytes_allocated)

    @staticmethod
    def _dtype_bytes(dtype: Any) -> int:
        try:
            return int(dtype.itemsize)
        except Exception:  # pragma: no cover - 非标准 dtype
            return 2

    def _pkv_build_past(
        self, torch: Any, pairs: Sequence[tuple[_Slot, ForwardItem]], k: int
    ) -> list[tuple[Any, Any]]:
        """把常驻缓冲的**有效前缀**拼成 `past_key_values`。

        ⚠️ 前缀长度只到 `past_len`，**不含本次要喂的 `k` 个 token**。
        多给一个 token 会让 HF 在 `past` 之上再算本次输入 ⇒ 长度双重计数
        （实测表现：`size of tensor a (10) != b (11)`）。这是本会话踩到的坑。
        HF 的约定是"`past_key_values` = 历史，`input_ids` = 本次新 token"。

        ⚠️ 这里仍然要 `cat`（把 B 个独立缓冲合成一个批张量是 HF 规则张量的硬约束，
        任务书也点名"持久 Cache 对象未必消除内部 cat"）。真正省下的是：
          * **补位**：历史等长 ⇒ 无 `F.pad`、无无效 KV；
          * **写回的逐层切片重建**：改用 `copy_` 原地写入固定地址（见 `_pkv_writeback`）。
        """
        sample = next(s for s, _ in pairs if s.past)
        n_layers = len(sample.past)
        keep = pairs[0][0].past_len
        layers: list[tuple[Any, Any]] = []
        for layer in range(n_layers):
            keys = [self._kv_buf[s.req_id][layer][0][:, :, :keep, :] for s, _ in pairs]
            vals = [self._kv_buf[s.req_id][layer][1][:, :, :keep, :] for s, _ in pairs]
            if len(keys) == 1:
                layers.append((keys[0], vals[0]))
            else:
                layers.append((torch.cat(keys, dim=0), torch.cat(vals, dim=0)))
        return layers

    def _pkv_writeback(
        self,
        torch: Any,
        pairs: Sequence[tuple[_Slot, ForwardItem]],
        hf_past: Sequence[tuple[Any, Any]],
        k: int,
    ) -> None:
        """把 HF 本步返回的 KV **原地写进**常驻缓冲，并推进 `past_len`。

        ⚠️ **HF 返回的是累积的完整 KV（长度 `past_len + k`），不是本步增量。**
        因为上面交给 HF 的 past 只到 `past_len`，返回的整段正好
        `[0 : past_len + k]` —— 与缓冲的前缀**逐位对应**，可整体 `copy_`。
        （与通用路径 `[pad:pad+keep]` 语义一致，区别是快路径无补位、无需切左侧。）

        与通用路径的关键差别在于**写到哪里**：这里是写进固定地址的常驻缓冲，
        `s.past` 随后以**视图**指过去 ⇒ 不需要每步新建一份 KV 张量。
        """
        keep = pairs[0][0].past_len + k
        for i, (s, _it) in enumerate(pairs):
            buf = self._kv_buf[s.req_id]
            for layer in range(len(buf)):
                kb, vb = buf[layer]
                kb[:, :, :keep, :].copy_(hf_past[layer][0][i : i + 1])
                vb[:, :, :keep, :].copy_(hf_past[layer][1][i : i + 1])
            s.past_len += k

    # ---- 槽位 -----------------------------------------------------------
    def _slot(self, req: Request) -> _Slot:
        slot = self._slots.get(req.req_id)
        if slot is None:
            if self.prompt_source is None:
                raise RuntimeError("prompt_source 未初始化；请先 load() 或显式注入")
            if req.prompt_len + req.gen_len > self.max_seq_len:
                raise ValueError(
                    f"请求 {req.req_id} 需要 {req.prompt_len + req.gen_len} 个位置，"
                    f"超过 max_seq_len={self.max_seq_len}；"
                    "请调大 max_seq_len 或改工作负载"
                )
            slot = _Slot(req_id=req.req_id, prompt=self.prompt_source.prompt_ids(req))
            self._slots[req.req_id] = slot
        return slot

    def retire(self, req_ids: Sequence[str]) -> None:
        """释放这些序列的设备侧状态。

        **必须逐请求及时调用**，不能等到整轮结束：
        ``_Slot.past`` 持有每个序列逐层的 K/V 张量，一个请求的 KV 占用与
        ``prompt_len + gen_len`` 成正比。并发为 1 跑 12 个请求时，
        若只在 run 末尾统一释放，驻留槽位会累积到 12——短工作负载看不出来，
        长工作负载上就是 OOM，而 OOM 的表现是"跑了一半崩掉"，
        和"工作负载太重"长得一样。

        ``empty_cache`` 只在真的摘掉了槽位时才调用：它是一次同步的显存整理，
        白调会引入与请求数无关的固定开销。

        ★ S02/P2a：**逐请求删 slot 与"清 allocator 缓存"是两件事**。
        前者是必需的（否则驻留槽位累积到 OOM），后者是**可选**的（`empty_cache`
        把已释放的显存还给驱动 ⇒ 下次分配要重新向驱动要）。
        高周转下每次请求都做一次同步整理会变成纯粹的周转税。
        ``retire_empty_cache=False`` 时本方法**只**摘槽位，把整理交给
        :meth:`flush_allocator_cache`（run 边界显式调用）。
        默认 ``True`` ⇒ 与既有行为逐字节相同。
        """
        n_before = len(self._slots)
        for rid in req_ids:
            self._slots.pop(rid, None)
            # ★ S04/P2b-c：常驻 KV 缓冲**必须与槽位同生共死**。
            # 只摘 `_slots` 而留着 `_kv_buf`，等于每完成一个请求就漏掉
            # `capacity × L × 2 × n_kv × D × 2B` 字节（默认容量下是每请求数十 MB）
            # —— 短负载看不出来，长负载上表现为"跑一半 OOM"，与"负载太重"长得一样。
            # `pop` 同时把累计字节数扣回去，使 `bytes_allocated` 是**当前驻留**值。
            dropped = self._kv_buf.pop(rid, None)
            # 同步长度记录一并摘掉，否则 `req_id` 复用时会被当成"已同步"
            # ⇒ 新槽位带着旧长度进入快路径 ⇒ 又变成读到过期/零内容。
            self._pkv_synced_len.pop(rid, None)
            if dropped is not None:
                rows = getattr(self, "_pkv_bytes_per_slot", 0)
                self._pkv_bytes_allocated = max(0, self._pkv_bytes_allocated - rows)
        if req_ids and len(self._slots) < n_before:
            if not self._retire_empty_cache:
                # 槽位已摘除（KV 引用随之释放）；只是**不**做显存整理。
                self._n_retire_empty_cache_suppressed += 1
                return
            torch, _ = self._torch()
            musa = _musa_module(torch)
            if musa is not None and self._device_available(torch):
                musa.empty_cache()
                self._n_retire_empty_cache_calls += 1

    def flush_allocator_cache(self) -> bool:
        """显式清一次 allocator 缓存。**返回是否真的调用了**（设备不可用 ⇒ `False`）。

        用途：`retire_empty_cache=False` 时，由调用方在 **run 边界**调用一次，
        既保住"长期周转不漏显存"，又不把每请求一次的同步整理压进周转路径。
        ⚠️ 返回值必须可核验：把"没调用"记成"调用过"，等于把一个没发生的动作
        写进了产物。
        """
        torch, _ = self._torch()
        musa = _musa_module(torch)
        if musa is None or not self._device_available(torch):
            return False
        musa.empty_cache()
        self._n_retire_empty_cache_calls += 1
        return True

    def live_slots(self) -> int:
        """当前仍持有设备侧状态的序列数（供后端核对"及时释放"是否成立）。"""
        return len(self._slots)

    def sample_device(self) -> dict[str, Any]:
        torch, _ = self._torch()
        if not self._loaded or not self._device_available(torch):
            return {}
        return self._device_mem(torch)

    def _counter_snapshot(self) -> dict[str, float]:
        """当前累加器快照（供"预热前/后"相减，得到仅正式测量段的口径）。"""
        return {
            "n_forwards": float(self._n_forwards),
            "n_logical_steps": float(self._n_logical_steps),
            "pack_s": float(self._pack_s),
            "fwd_s": float(self._fwd_s),
            "readback_s": float(self._readback_s),
            "tokens_out": float(self._tokens_out),
            # ★ S04 的两个分桶也要分窗口：预热里**同样会走快路径**（预热请求等长、
            #   纯 decode），若不扣掉，报出来的"覆盖率"会把预热段算成正式覆盖。
            #   实测影响很大：EXECOPT4 的 193 里约有 116 来自 1 轮预热。
            "pkv_fastpath": float(self._n_pkv_fastpath_steps),
            "pkv_fallback": float(self._n_pkv_fallback_steps),
            "pkv_resyncs": float(self._n_pkv_resyncs),
        }

    def stats(self) -> dict[str, Any]:
        # ★ 派生一组"**仅正式测量段**"的量：`*_total` 与 `n_forwards`/`tokens_out`
        #   含预热（预热与正式段共用同一累加器），而预热轮数随机器速度变
        #   ⇒ 直接跨臂比 `*_total` 会把"预热跑了几轮"当成机制效应。
        #   ⚠️ 不修改既有键：它们要与历史产物保持可比；这里只**增加**。
        base = self._warmup_baseline
        measured: dict[str, Any] = {}
        if base is None:
            # 没预热过 ⇒ 总量本身就是正式段口径（如实标注，不猜）。
            measured = {
                "n_forwards_measured": self._n_forwards,
                "n_logical_steps_measured": self._n_logical_steps,
                "pack_ms_measured": round(self._pack_s * 1000.0, 3),
                "fwd_ms_measured": round(self._fwd_s * 1000.0, 3),
                "tokens_out_measured": self._tokens_out,
                "pkv_fastpath_measured": self._n_pkv_fastpath_steps,
                "pkv_fallback_measured": self._n_pkv_fallback_steps,
                "pkv_resyncs_measured": self._n_pkv_resyncs,
                "basis": "no_warmup",
            }
        else:
            measured = {
                "n_forwards_measured": self._n_forwards - int(base["n_forwards"]),
                "n_logical_steps_measured":
                    self._n_logical_steps - int(base["n_logical_steps"]),
                "pack_ms_measured": round((self._pack_s - base["pack_s"]) * 1000.0, 3),
                "fwd_ms_measured": round((self._fwd_s - base["fwd_s"]) * 1000.0, 3),
                "tokens_out_measured": self._tokens_out - int(base["tokens_out"]),
                "pkv_fastpath_measured":
                    self._n_pkv_fastpath_steps - int(base["pkv_fastpath"]),
                "pkv_fallback_measured":
                    self._n_pkv_fallback_steps - int(base["pkv_fallback"]),
                "pkv_resyncs_measured":
                    self._n_pkv_resyncs - int(base.get("pkv_resyncs", 0.0)),
                "basis": "total_minus_warmup",
            }
        measured["pkv_coverage_of_forward_groups"] = (
            round(measured["pkv_fastpath_measured"] / measured["n_forwards_measured"], 6)
            if measured["n_forwards_measured"] else None
        )
        measured["pkv_coverage_of_logical_steps"] = (
            round(measured["pkv_fastpath_measured"] / measured["n_logical_steps_measured"], 6)
            if measured["n_logical_steps_measured"] else None
        )
        return {
            "n_forwards": self._n_forwards,
            "pack_ms_total": round(self._pack_s * 1000.0, 3),
            "fwd_ms_total": round(self._fwd_s * 1000.0, 3),
            #: 采样 + 取回主机的累计耗时。与 tokens_out 相除即"每 token 的偏早量"。
            "readback_ms_total": round(self._readback_s * 1000.0, 3),
            "tokens_out": self._tokens_out,
            "max_batch_seen": self._max_batch,
            "live_slots": len(self._slots),
            "load": dict(self._load_info),
            "warmup": dict(self._warmup_info),
            #: ★ 仅正式测量段的同口径派生量（跨臂比"机制效应"要看这一组）。
            "measured_window": measured,
            # 执行优化计数（S01/S02/S03）。计数为 0 = 配了但没走到；
            # 与"没配"必须可分，否则"优化没生效"会被读成"没有收益"。
            "execution_optimizations": self.execution_optimizations,
        }

    # ---- 设备工具 -------------------------------------------------------
    @staticmethod
    def _torch() -> tuple[Any, Any]:
        try:
            import torch  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "真机后端需要 torch。本模块在 --dry-run 路径上也会被导入，"
                "因此 torch 只在真正用到时才 import。"
            ) from exc
        try:
            import torch_musa  # noqa: F401, PLC0415
        except ImportError:
            # 不在这里报错：本地 CPU 数值验证（注入微型模型）也会走到这里，
            # 而 torch.musa 的真实可用性由 _device_available 判定。
            torch_musa = None
        return torch, torch_musa

    def _device_available(self, torch: Any) -> bool:
        musa = _musa_module(torch)
        if musa is None:
            return False
        try:
            return bool(musa.is_available())
        except Exception:  # pragma: no cover - 驱动异常一律按"不可用"处理
            return False

    def _synchronize(self, torch: Any) -> None:
        musa = _musa_module(torch)
        if musa is not None and self._device_available(torch):
            musa.synchronize()

    def _dtype(self, torch: Any) -> Any:
        dtype = getattr(torch, _DTYPE_NAMES[self.precision])
        # 逐元素字节数从真实 dtype 上取，而不是按精度名硬编码：
        # 硬编码的映射在新增精度时会静默算错显存理论值。
        self._bytes_per_param = int(torch.tensor([], dtype=dtype).element_size())
        return dtype

    def _config(self) -> Any:
        """取模型配置。

        容忍一层包装：测试里模型常被套在一个记录器外层（``_RecordingModel``），
        而规范实现里没有"外面套一层"这回事。这里只往内看**一层**——
        递归遍历属性会掩盖真正的配置缺失。
        """
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            return cfg
        inner = getattr(self.model, "inner", None)
        return getattr(inner, "config", None)

    def _model_n_layers(self) -> int:
        """模型层数。取不到时报错——层数是 KV 写回的循环上界，猜一个值
        会导致"只写回了前几层"，而形状仍然合法。"""
        cfg = self._config()
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        raise RuntimeError(
            "拿不到模型的层数（config.num_hidden_layers 等均缺失）。"
            "KV 写回需要它做循环上界；猜错会静默只写回部分层。"
        )

    def _vocab_size(self) -> int:
        val = getattr(self._config(), "vocab_size", None)
        if isinstance(val, int) and val > 1:
            return val
        raise RuntimeError(
            "拿不到词表大小（model.config.vocab_size 缺失）。"
            "合成 prompt 必须要它才能保证 token id 落在合法区间；"
            "不要猜一个值——越界的 id 会在 embedding 处报错，"
            "但猜小了会静默命中错误的 token。"
        )

    def _special_ids(self) -> list[int]:
        tok = self.tokenizer
        if tok is None:
            return []
        ids: list[int] = []
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"):
            val = getattr(tok, attr, None)
            if isinstance(val, int):
                ids.append(val)
        extra = getattr(tok, "additional_special_tokens_ids", None)
        if isinstance(extra, list):
            ids.extend(int(i) for i in extra if isinstance(i, int))
        return sorted(set(ids))

    def _device_mem(self, torch: Any) -> dict[str, Any]:
        musa = _musa_module(torch)
        if musa is None or not self._device_available(torch):
            return {}
        try:
            free_b, total_b = musa.mem_get_info()
        except Exception:  # pragma: no cover
            return {}
        # 显存有两个口径且不可混用（见 PLAYBOOK）：这里是 torch 侧口径。
        mib = 1024 * 1024
        return {
            "used_mib": round((total_b - free_b) / mib, 1),
            "total_mib": round(total_b / mib, 1),
            "source": "torch.musa.mem_get_info",
        }

    def _position_ids(self, torch: Any, past_lens: Sequence[int], k: int) -> Any:
        """本次输入各 token 的位置编号，形状 ``[batch, k]``。

        第 i 行、第 j 个新 token 的位置 = ``past_lens[i] + j``。

        为什么直接算而不用 mask 前缀和：同组内各行的 ``past_len`` 可以不同
        （补位把它们对齐到 ``past_max``），于是"整段可见性"是个**参差**的结构，
        拼不成规则张量。而位置编号其实只依赖各行的真实历史长度，
        直接由 ``past_lens`` 与列号相加即可——它同时把补位的影响彻底排除
        （补位区不出现在 ``input_ids`` 里，自然也不需要给它编位置）。

        这是左补位最容易出错的一处：若按补位后的宽度来数（或用行号乘某个宽度），
        位置会整体偏移；错了之后**形状仍然合法**，只有数值对比能发现。
        """
        rows = [
            list(range(pl, pl + k)) for pl in past_lens
        ]
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def _logits_digest(self, torch: Any, row: Any) -> dict[str, Any]:
        """把**一行** logits 压成摘要。`sha256` 是逐位判据，其余三个便于人读。

        为什么先转 `float32` 再取字节：`bfloat16` 没有 `.numpy()`（会在真机上抛
        `TypeError`），而 `fp16 → fp32` 是**精确**转换（fp32 是 fp16 的超集）⇒
        "两臂同值"这一判据不被这次转换削弱。直接对 fp16 取 `view(int16)` 也能
        哈希，但一旦以后换 dtype 就会静默改变摘要语义，不如统一到 fp32。
        """
        import hashlib

        host = row.detach().to("cpu").float().contiguous()
        raw = host.numpy().tobytes()
        return {
            "argmax": int(host.argmax().item()),
            "max": round(float(host.max().item()), 6),
            "sum": round(float(host.sum().item()), 6),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def _sample(self, torch: Any, logits: Any) -> Any:
        if not self.do_sample:
            return logits.argmax(dim=-1)
        logits = logits.float() / max(self.temperature, 1e-6)
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            remove = (cum - probs) >= self.top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter_(
                1, sorted_idx, sorted_logits
            )
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "驱动尚未 load()。装载必须在任何 forward 之前完成——"
                "它包含显存分配与 kernel 初始化，落在计时窗口内会污染全部时延。"
            )


# --------------------------------------------------------------------------- #
# 闭环补位抖动（B1 草案 §5 / OPT-03R 裁定 Q4）
# --------------------------------------------------------------------------- #

def _closure_jitter_ms(*, seed: int, client: int, round_index: int,
                       span_ms: float) -> float:
    """闭环补位时刻的**确定性**抖动（ms，∈ [0, span_ms)）。

    ★ 为什么需要它（实测依据）：`closed` 闭模型下，补位时刻恒 = `step_start`，
    而 `step_start` 只由前面各步的墙钟之和决定 ⇒ 在确定性负载 + 固定配置下，
    整个 `arrival_ms` 序列**逐位相同**。实测 `S14B2`（5 次重复）：
    `arrival_ms` **0/92 个请求不同**（sha256 逐位一致），而同时 `done_ms` **92/92 不同**。

    影响：主指标的跨重复噪声带被压到 CV ≈ **0.01–0.09%**（实测 ttft_p50 0.093% /
    tpot_p50 **0.014%**）⇒ 任何小于该量级的真实效应都被淹没（S6 的 +0.05% 正是此例），
    且配对判据"≥4/5 同号"在同号/异号上退化为**恒真或恒假**。

    本函数把"调度差异"映射到"到达差异"。

    ★★ **2026-10-03 改：抖动绑定到「客户端 + 轮次」，不再由 `step` 决定**
    ---------------------------------------------------------------------
    旧版是 `f(seed, req_id, step)`。`step` 是**调度策略的函数** ⇒
    同一个客户端在同一轮的抖动，会因为它**在第几步被补位**而不同。
    后果有两层，第二层才是要点：

      ① **不可复现为绝对时刻**：抖动值依赖当时走到第几步 ⇒ "这个请求的到达时刻"
         在跑之前**没有定义**（只能跑完才知道）。而到达语义本该是**输入**，不是输出。
      ② **它让"等待"这件事无法表达**：既然到达时刻=当时那一步 + 抖动，而请求又是在
         `step_start` 入队 ⇒ 实测 **16/16** 个请求 `enqueue_ms < arrival_ms`
         （最大倒置 ≈ 抖动上限），即"到达时刻"只是个**装饰性字段**；
         在真机上它的字面含义是"请求还没到就已经在队列里等了"。

    ⇒ 现在抖动只由 `(seed, 客户端, 轮次)` 派生：
      * **客户端**与**轮次**都是**闭模型负载本身**的属性（第 k 个补位属于
        `client = k % concurrency`、`round = k // concurrency`），**与调度策略无关**
        ⇒ 抖动是**绝对量**，跑之前就定下来；
      * 于是"到达时刻"重新成为**输入**，执行器必须**等到**它才放行
        （见 `run_benchmark` / `real_backend` 的补位循环）。

    ⚠️ **不是**真随机、**不是**调参入口：`span_ms` 由负载显式给定、须在预注册里写死。
    `span_ms <= 0` ⇒ 返回 0 ⇒ 行为与 `closed` **逐字节相同**（默认关闭）。
    """
    if span_ms <= 0.0:
        return 0.0
    # 用 blake2b（stdlib，跨平台稳定）而不是 `hash()`（PYTHONHASHSEED 会变）
    h = hashlib.blake2b(f"{seed}|c{client}|r{round_index}".encode("utf-8"),
                        digest_size=8).digest()
    frac = int.from_bytes(h, "big") / float(1 << 64)  # ∈ [0, 1)
    return frac * span_ms


def closure_jitter_by_request(*, seed: int, order: Sequence[str], concurrency: int,
                              span_ms: float) -> dict[str, float]:
    """**一次算好**每个请求的补位抖动：`req_id -> 抖动 ms`。

    ★ 为什么必须**预先算好**（2026-10-03）：抖动只依赖 `(seed, 客户端, 轮次)`，
    而这两者由**请求在负载里的位置**决定（第 k 个补位 ⇒ `client = k % concurrency`、
    `round = k // concurrency`）⇒ 完全可以、也应该在跑之前一次算定。

    这样做的三个好处：
      ① 到达时刻成为**输入**（跑之前就有定义），策略不再影响它；
      ② 不会因为"每次循环重算"而在补位等待期间悄悄改变抖动值；
      ③ 两条执行路径（模拟器 / 真机）可以各自查表，而**表是同一份**。

    ⚠️ `concurrency <= 0` 时按 1 处理（否则取模会崩；而那是配置错误，
    不是负载语义）。`span_ms <= 0` ⇒ 全 0（与 `closed` 逐字节相同）。
    """
    c = max(1, int(concurrency))
    out: dict[str, float] = {}
    for idx, rid in enumerate(order):
        out[str(rid)] = _closure_jitter_ms(
            seed=int(seed or 0), client=idx % c, round_index=idx // c,
            span_ms=float(span_ms or 0.0),
        )
    return out


# --------------------------------------------------------------------------- #
# 真实后端（纯逻辑）
# --------------------------------------------------------------------------- #

@dataclass
class _RunState:
    """真实后端的逐请求状态机。

    与 ``run_benchmark._SimRequest`` 字段一一对应，理由相同：
    三个"可能不发生"的时刻一律 ``None`` 表示未发生，**不用 0**——
    0 既是"run 的第一步"这一合法时刻，也是"确实等了 0ms"。
    """

    req: Request
    scheduled_arrival_ms: float | None = None
    arrival_ms: float | None = None
    enqueue_ms: float | None = None
    first_schedule_ms: float | None = None
    first_token_ms: float | None = None
    done_ms: float | None = None
    phase: str = "waiting"          # waiting | prefill | decode | done
    prefill_remaining: int = 0
    decode_remaining: int = 0
    cached_tokens: int = 0
    ttft_ms: float | None = None
    last_emit_ms: float = 0.0
    intervals: list[float] = field(default_factory=list)
    #: 与 ``intervals`` **逐项对齐**的两个分解量（第 04 步）：
    #: 该间隔里有没有 prefill / 纯 decode 一共花了多少。
    #: 逐请求保留（而不只留全局滚动窗口）的原因是：验收要**按请求**核对
    #: "这个间隔跨了哪些步"，全局列表在多个请求交替发 token 时无法一一对应。
    interval_had_prefill: list[bool] = field(default_factory=list)
    interval_decode_ms: list[float] = field(default_factory=list)
    outcome: str = OUTCOME_UNFINISHED
    failure: str = ""
    #: 上一次发 token 时的两个**单调时钟**读数（见 ``prefill_clock_ms`` 的说明）。
    #: 两者的差就是"这一个间隔里，系统分别有多少时间花在 prefill / 纯 decode 上"。
    #: 用时钟差而不是"看最后一步"：间隔可能跨很多步，只看向最后一步会漏掉
    #: **夹在中间的** prefill 步。
    prefill_clock_at_last_emit: float = 0.0
    decode_clock_at_last_emit: float = 0.0


def _pending_execution_prediction(entry: dict[str, Any] | None, *,
                                  decision_step: int, shadow: bool) -> dict[str, Any] | None:
    """Bind an applied, predicted cost candidate to exactly the next step."""
    decision = (entry or {}).get("cost_policy") or {}
    if shadow or not decision.get("changed") or decision.get("fallback_reason"):
        return None
    chosen = [c for c in decision.get("candidates", [])
              if (c.get("chunk"), c.get("max_tokens")) ==
                 (decision.get("chunk"), decision.get("max_tokens"))]
    if len(chosen) != 1:
        return None
    candidate = chosen[0]
    shape = candidate.get("execution_shape")
    fwd, host = candidate.get("predicted_ms"), candidate.get("host_ms")
    if (not shape or not shape.get("items") or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) for v in (fwd, host)) or fwd <= 0 or host < 0):
        return None
    return {"decision_step": decision_step, "target_step": decision_step + 1,
            "chunk": candidate["chunk"], "max_tokens": candidate["max_tokens"],
            "predicted_shape": shape, "predicted_fwd_ms": fwd, "predicted_host_ms": host,
            "prediction_value_basis": "recorded_decision_candidate"}


def _prediction_execution_observation(pending: dict[str, Any], *, actual_step: int,
                                     actual_shape: dict[str, Any] | None,
                                     timing: StepTiming | None, hardware_backed: bool,
                                     unexecuted_reason: str | None = None) -> dict[str, Any]:
    """Observe correspondence only; this function never changes scheduling.

    Match the calibration probe: host = driver.forward elapsed - forward.
    Pack/readback and remaining host work are separate diagnostics.  No
    prediction error is computed for a changed shape or nonhardware execution.
    """
    predicted = pending["predicted_shape"]
    shape_match = (execution_shape_key(predicted) == execution_shape_key(actual_shape)
                   if actual_shape is not None else None)
    n_forwards_match = (timing.n_forwards == actual_shape["n_forwards"] == predicted["n_forwards"]
                        if timing is not None and actual_shape is not None else None)
    reasons = []
    if actual_step != pending["target_step"]:
        reasons.append("not_immediate_next_step")
    if shape_match is not True:
        reasons.append("execution_shape_changed" if actual_shape is not None else "actual_shape_unavailable")
    if n_forwards_match is not True:
        reasons.append("forward_group_count_unconfirmed_or_changed")
    if not hardware_backed:
        reasons.append("nonhardware_execution")
    actual_timing = None
    if timing is None:
        reasons.append(unexecuted_reason or "forward_not_executed")
    else:
        values = (timing.fwd_ms, timing.pack_ms, timing.readback_ms, timing.elapsed_ms)
        valid = (all(isinstance(v, (int, float)) and not isinstance(v, bool)
                     and math.isfinite(v) and v >= 0 for v in values)
                 and timing.fwd_ms > 0 and timing.elapsed_ms > 0
                 and timing.elapsed_ms >= timing.fwd_ms)
        if not valid:
            reasons.append("invalid_timing")
        actual_timing = {
            "fwd_ms": timing.fwd_ms, "pack_ms": timing.pack_ms,
            "readback_ms": timing.readback_ms,
            "host_ms": timing.elapsed_ms - timing.fwd_ms,
            "driver_elapsed_ms": timing.elapsed_ms, "n_forwards": timing.n_forwards,
            "unmodeled_driver_ms": timing.elapsed_ms - sum(values[:3]),
        }
    return {**pending, "actual_step": actual_step, "actual_shape": actual_shape,
            "shape_match": shape_match, "n_forwards_match": n_forwards_match,
            "actual_timing": actual_timing, "comparison_eligible": not reasons,
            "ineligible_reasons": reasons, "hardware_backed": hardware_backed,
            "timing_contract": "whole_step_fwd_plus_host_v1",
            "host_definition": "driver.forward elapsed_ms - fwd_ms; excludes prime/warmup/external sync"}


class RealBackend:
    """把真实驱动接到与 ``simulate()`` 相同的 ``ExecutionResult`` 契约上。

    计时边界（**这是与 dry-run 最重要的差别**）
    ------------------------------------------
    ``load()`` 与 ``warmup()`` 都在时间原点**之前**完成，``wall_ms`` 只覆盖
    正式测量段。把装载算进 wall_ms 会让吞吐类指标偏低，而"偏低"看起来
    完全正常——没有任何字段会暴露它。

    与 ``simulate()`` 的关系
    ------------------------
    调度决策共用 ``plan_step``（组合逻辑唯一实现）；本类只负责"用什么时间推进"。
    到达/受理那段循环与 ``simulate()`` 逐行对应，读那段时请两处对照看。

    请求终态只有三态：``success`` / ``failed``（驱动抛错）/ ``unfinished``
    （预算内没跑完）。``rejected`` 需要准入拒绝语义，本后端不实现限流，
    因此**产生不出**这一类——这一点写在代码里，免得汇总里 ``n_rejected=0``
    被读成"测过了，没有被拒"。
    """

    def __init__(
        self,
        cfg: Any,
        *,
        driver: GenerationDriver | None = None,
        clock: Callable[[], float] = _default_clock,
        sleep: Callable[[float], None] = time.sleep,
        allow_non_hardware_driver: bool = False,
    ) -> None:
        self.cfg = cfg
        self.clock = clock
        self._sleep = sleep
        self._driver = driver
        #: 只给单测用的显式开口。名字写得很长是故意的：
        #: 它在代码里出现就意味着"这里在跑假驱动"，不可能被误读成正常用法。
        #: 即便打开，``backend_stats`` 里仍会记 ``hardware_backed=False``，
        #: 且 ``run_once`` 另有一道闸门——所以生产路径拿不到被标成实测的假数字。
        self._allow_non_hardware = bool(allow_non_hardware_driver)
        self.backend_stats: dict[str, Any] = {}

    # ---- 驱动选择 -------------------------------------------------------
    def driver(self) -> GenerationDriver:
        if self._driver is None:
            self._driver = build_driver(self.cfg)
        return self._driver

    def _assert_hardware_backed(self, drv: GenerationDriver) -> None:
        """拒绝把非硬件驱动当成实测。

        单测会注入假驱动，那种产物必须自证不是实测。这里是最后一道闸门：
        若哪天有人把它接到 CLI 路径上，报错会立刻出现，而不是产出一批
        ``MEASURED`` 的假数字。
        """
        if not getattr(drv, "hardware_backed", False) and not self._allow_non_hardware:
            raise RuntimeError(
                f"驱动 {type(drv).__name__} 声明 hardware_backed=False，"
                "不能用于产出实测结果。假驱动只应出现在单测中；"
                "单测请显式传 allow_non_hardware_driver=True。"
            )

    # ---- 主循环 ---------------------------------------------------------
    def run(
        self,
        requests: Sequence[Request],
        repeat_index: int = 0,
        *,
        spec: Any = None,
    ) -> Any:
        """跑一次。``spec`` 提供到达语义（闭环并发数）；缺省即开放环。

        ``spec`` 必须由调用方传入而不能从 ``requests`` 反推：闭环下所有
        ``arrival_ms`` 都是 0，从请求列表看不出"这是闭环"，
        会把闭环误当开放环（所有请求在 t=0 一次性涌入）。
        """
        cfg = self.cfg
        drv = self.driver()
        self._assert_hardware_backed(drv)

        if cfg.cache.enabled:
            raise NotImplementedError(
                "真实后端本版**不支持前缀缓存**：HF 的原生 KV 缓存无法按前缀哈希"
                "复用块，照常记命中却并不省算力，产物里的 cached_tokens 会是一个"
                "看起来正常但错的数。\n"
                "  请用 cache.enabled=false 跑基线；前缀缓存的实验留给按功能选定的"
                "引擎（vLLM_musa 路线，见 README §9）。"
            )

        # ---- 装载与预热：都在时间原点之前 ----
        load_info = drv.load()
        shape = self._representative_shape(requests)
        warm_info = drv.warmup(
            cfg.run.warmup_s,
            batch=max(1, min(cfg.max_num_seqs, len(requests))),
            prompt_len=shape["prompt_len"],
            gen_len=shape["gen_len"],
        )
        origin = self.clock()

        # ★ **闭环补位**（`B1` 草案 §5 / OPT-03R 裁定 Q4）：
        # `closed` = 补位时刻恒为 `step_start`（**确定性**：同一步序列 ⇒ 同一串 arrival
        #   ⇒ 跨重复零方差，实测 S14B2 的 `arrival_ms` 5 次重复逐位相同）。
        # `closed_jittered` = 同一闭环语义（在途仍受 `concurrency` 硬上限约束），
        #   但补位时刻加上**确定性抖动**（由 `seed` 与 `req_id` 派生，非真随机）
        #   ⇒ 批边界随调度改变 ⇒ 跨重复有差异。
        #   ⚠️ 语义变化：`arrival_ms` 不再恒等于 `step_start`，而是"客户端放行时刻"。
        #   在途上限**仍严格 ≤ concurrency**（与开放环不同——开放环过载会无限堆积）。
        arrival_kind = getattr(spec, "arrival", None) if spec is not None else None
        closed_loop = arrival_kind in ("closed", "closed_jittered")
        jittered_closure = arrival_kind == "closed_jittered"
        concurrency = int(getattr(spec, "concurrency", 0) or 0) if spec is not None else 0

        scheduler = build_scheduler(cfg.scheduler_name, **cfg.scheduler_kwargs)
        chunk_policy = build_chunk_policy(cfg.chunk_policy_name, config=cfg.chunk)
        cache = PrefixCachePolicy(cfg.cache)
        controller = AdaptiveController(cfg.controller)
        # 策略臂：与模拟器**共用**构造（`benchmark/arm_factory.build_arm`）。
        arm = build_arm(cfg)
        monitor = WorkloadMonitor()

        cur_chunk = cfg.chunk.chunk_size
        cur_max_tokens = cfg.chunk.max_num_batched_tokens
        max_seqs = cfg.max_num_seqs

        states = {
            r.req_id: _RunState(
                req=r,
                scheduled_arrival_ms=r.arrival_ms,
                decode_remaining=max(0, r.gen_len - 1),
            )
            for r in requests
        }
        order = sorted(requests, key=lambda r: (r.arrival_ms, r.req_id))
        not_arrived = list(order)
        # ★ 补位抖动**预先算好**（按「客户端 × 轮次」），**不依赖 `step`**。
        #   必须在循环外一次算定：① 到达时刻才算**输入**（跑之前就有定义）；
        #   ② 不会在等待期间被重算而改变；③ 与模拟器查的是同一张表的同一套规则。
        #: 抖动后的**绝对到达时刻**（第一次决定后不再变）。
        pending_arrival: dict[str, float] = {}
        jitter_by_id = closure_jitter_by_request(
            seed=int(getattr(spec, "seed", 0) or 0),
            order=[r.req_id for r in order],
            concurrency=int(concurrency or 0),
            span_ms=float(getattr(spec, "closure_jitter_ms", 0.0) or 0.0),
        ) if jittered_closure else {}
        enqueued: list[_RunState] = []

        samples: list[TelemetrySample] = []
        controller_trace: list[dict[str, Any]] = []
        pending_prediction: dict[str, Any] | None = None
        system_states: list[dict[str, Any]] = []
        recent_ttft: list[float] = []
        recent_tpot: list[float] = []
        stall_reason = ""
        driver_error = ""
        idle_ms = 0.0
        n_idle_steps = 0
        #: 已经回收过设备侧状态的请求。用集合而不是布尔量：retire 幂等，
        #: 但**重复调用会重复触发 empty_cache**（一次同步的显存整理），
        #: 白调会引入与请求数无关的固定开销。
        retired: set[str] = set()
        #: 同时驻留槽位数的峰值。它是"及时释放"这条性质的**可观测量**：
        #: 闭环并发为 k 时峰值不应显著超过 k；若它随请求总数增长，
        #: 说明释放又回到了"整轮结束才做"。
        max_live = 0

        # ---- 分层控制信号（第 04 步）-----------------------------------------
        # 为什么需要：``recent_tpot`` 存的是**相邻两次 token 发出之间的间隔**，
        # 而某一步做了 prefill 时，落在这两步之间的间隔就**直接包含该 prefill 的耗时**。
        # 控制器于是把"prefill 挤占"读成"decode 变慢"，而它的响应是调小 chunk_size
        # （prefill 更碎 ⇒ 含 prefill 的步更多）——反向放大。
        # ⇒ 把两种东西**分开记**，让控制器能看见"停顿是 decode 慢还是被别人挤的"。
        #
        # 判法用**单调时钟**而不是"看这一步/最后一步"：
        #   每步结束按"本步有没有 prefill 工作"把耗时累进两个时钟之一；
        #   两次 emit 之间的时钟差，就是该间隔里分别花在两者上的时间。
        #   这样**天然覆盖间隔内的全部执行步**，不会漏掉夹在中间的 prefill。
        prefill_clock_ms = 0.0   # 有 prefill 工作的步，耗时累进这里
        decode_clock_ms = 0.0    # **纯 decode** 步（本步完全没 prefill），累进这里
        recent_decode_ms: list[float] = []
        recent_had_prefill: list[bool] = []
        #: 每步的 (时刻, 是否含 prefill, 纯decode耗时)——**可复核性**用：
        #: 事后可以从它独立重算 recent_had_prefill / recent_decode_ms，
        #: 从而验证"加遥测没有改变调度与 token 语义"（见测试里的等价性夹具）。
        step_trace: list[tuple[float, bool, float]] = []

        def now_ms() -> float:
            return (self.clock() - origin) * 1000.0

        def all_settled() -> bool:
            return all(s.phase == "done" for s in states.values())

        def in_flight() -> int:
            """闭环下"在途"的口径：已发出但尚未完成的请求。

            必须包含 ``waiting``（客户端已发出、服务端还没排上）。只数
            prefill/decode 会让限流门形同虚设：同一时刻新入队的请求不被计入，
            一个步长内能把队列一次性填满。
            """
            return sum(1 for s in enqueued if s.phase != "done")

        step = 0
        now = 0.0
        while not all_settled() and step < cfg.device.max_steps:
            step_start = now_ms()
            # A decision is about its immediate next step, never every step up
            # to the next controller tick. Consume even if this attempt stalls.
            prediction_for_step, pending_prediction = pending_prediction, None

            # 1) 到达与受理。与 simulate() 同构：
            #    闭环看客户端容量，开放环看日程；两者的 arrival 语义不同。
            while not_arrived:
                nxt = not_arrived[0]
                if closed_loop:
                    if concurrency > 0 and in_flight() >= concurrency:
                        break
                    if jittered_closure:
                        # ★ 闭环补位（Q4）：放行时刻 = `step_start` + **确定性抖动**。
                        #
                        # 为什么需要：`closed` 下 `scheduled` 恒等于 `step_start`，
                        # 而 `step_start` 只由"前面各步的墙钟之和"决定 ⇒ 若两次运行的
                        # 步序列一致（确定性负载 + 固定配置），`arrival_ms` 就**逐位相同**
                        # ⇒ 跨重复零方差 ⇒ 配对判据（"≥4/5 同号"）恒真/恒假而失效。
                        #
                        # 抖动由 `(seed, req_id, step)` 派生 ⇒ **可复现**（不是真随机），
                        # 但**随调度改变**（`step` 与 `in_flight` 取决于控制器的决策）
                        # ⇒ 使"会调的臂"与"不会调的臂"的轨迹真正分岔。
                        #
                        # ⚠️ 幅度 `jitter_ms` 由负载显式给定（默认为 0 ⇒ 退化为 `closed`）。
                        # 幅度**不是**调参入口：它只用来把"调度差异"映射到"到达差异"，
                        # 取值须在预注册里写死（见预注册协议）。
                        # ★ 2026-10-03 改：抖动**查预先算好的表**（按 客户端×轮次），
                        #   不再用 `step` 现算（那是把"到达"变成了"调度的函数"）。
                        # ⚠️ 用本地待定表，**不用** `state.scheduled_arrival_ms`：
                        #    后者构造时已被填成 `r.arrival_ms`（闭环 = 0），
                        #    拿它当"算过没算过"的标志会让抖动**一次都不生效**。
                        jit = jitter_by_id.get(nxt.req_id, 0.0)
                        sched = pending_arrival.get(nxt.req_id)
                        if sched is None:
                            sched = step_start + jit
                            pending_arrival[nxt.req_id] = sched
                        # ★★ **等到点才放行**：`scheduled` 是绝对时刻，未到就**不放行**
                        #   （且**不 pop**，下一个 step 再看）。旧实现把 `scheduled`
                        #   记成 `step_start + jit` 却仍在 `step_start` 入队
                        #   ⇒ 实测 16/16 个请求 `enqueue_ms < arrival_ms`
                        #   （字面含义是"请求还没到就已经在队列里等"）。
                        if step_start < sched:
                            break
                        scheduled = sched
                    else:
                        scheduled = step_start
                else:
                    if nxt.arrival_ms > step_start:
                        break
                    scheduled = nxt.arrival_ms
                not_arrived.pop(0)
                s = states[nxt.req_id]
                s.scheduled_arrival_ms = scheduled
                # ⚠️ ``arrival`` 记的是**放行时刻**（= ``scheduled``），**不是**本步开始时刻。
                #
                # 为什么不能写成 ``step_start``：后端以「步」为单位推进，而一步可能很久
                # （chunk 关闭时一次算完整个长 prompt）。把 arrival 改写成 step_start，
                # "请求其实早已到达、只是本步才轮到处理"这段等待会被**整段抹掉**，
                # 突发负载下的 TTFT 被系统性低估——正是 `metrics.py` 在
                # ``scheduled_arrival_ms`` 上写明"**不得被执行步的时间覆盖**"的那件事。
                #
                # 这个缺陷的隐蔽之处在于**账目仍然自洽**：``done_ms`` 与 ``ttft_ms`` 都以
                # 同一个被覆盖的参考点计量，``accounting_error_ms`` 依旧是 0。
                # 只有把 arrival / enqueue 拆成两个时刻，才能看见它。
                #
                # 与 ``simulate()``（`benchmark/run_benchmark.py`）**同一份语义**：
                # 它写的是 ``s.arrival_ms = scheduled_arrival``。两处必须一致，
                # 否则"模拟器与真机的同一指标"根本不是同一个量。
                s.arrival_ms = scheduled
                # 受理时刻 = 本步开始（调度器在这一刻才"看见"它）。
                # arrival -> enqueue 的差是**受理延迟**（步粒度 / 链路获取），
                # enqueue -> first_schedule 的差才是**真正的排队等待**（容量不足）。
                # 两者成因与对策都不同，合并成一个数会让"加容量"与"修链路"失去依据。
                s.enqueue_ms = step_start
                s.phase = "waiting"
                enqueued.append(s)

            waiting = [s for s in enqueued if s.phase == "waiting"]
            prefilling = [s for s in enqueued if s.phase == "prefill"]
            decoding = [s for s in enqueued if s.phase == "decode"]

            # 2) 调度决策：与 dry-run 完全同一份组合逻辑。
            plan = plan_step(
                scheduler=scheduler,
                chunk_policy=chunk_policy,
                cache=cache,
                waiting=[s.req for s in waiting],
                decoding=[s.req for s in decoding],
                prefilling=[(s.req, s.prefill_remaining) for s in prefilling],
                cur_chunk=cur_chunk,
                cur_max_tokens=cur_max_tokens,
                max_num_seqs=max_seqs,
                chunk_enabled=cfg.chunk.enabled,
                step=step,
            )
            if plan.stall_reason:
                stall_reason = plan.stall_reason
                if prediction_for_step is not None:
                    controller_trace.append({"step": step, "prediction_execution_match":
                        _prediction_execution_observation(prediction_for_step, actual_step=step,
                            actual_shape=None, timing=None, hardware_backed=drv.hardware_backed,
                            unexecuted_reason="scheduler_stalled")})
                break

            by_id = states
            new_starts = [by_id[rid] for rid in plan.new_start_ids]
            decode_targets = [by_id[rid] for rid in plan.decode_ids]
            for s in new_starts:
                s.cached_tokens = plan.cached_tokens[s.req.req_id]
                s.prefill_remaining = max(0, s.req.prompt_len - s.cached_tokens)
                s.phase = "prefill"

            # 整段前缀命中：无需计算即可产出首 token。本后端禁用缓存，
            # 因此正常应为空；真出现了说明"命中却仍要算"的漏洞回来了。
            if plan.immediate_ids:
                raise RuntimeError(
                    f"出现整段前缀命中 {list(plan.immediate_ids)}，"
                    "但本后端未实现前缀缓存 → 这些请求不会真的省下计算。"
                    "要么关闭缓存，要么实现 KV 复用；不要让它静默通过。"
                )

            # 3) 把决策翻译成一次设备前向。
            #    prefill（含续传）与 decode 合成同一个批——这正是连续批处理
            #    在真机上的样子：同一时刻既有新请求在算 prompt，也有老请求在出 token。
            specs = plan_forward_specs(
                plan, waiting=[s.req for s in waiting], decoding=[s.req for s in decoding],
                prefilling=[(s.req, s.prefill_remaining) for s in prefilling])
            batch = [ForwardItem(it.req, it.n_new_tokens, it.need_token) for it in specs]
            actual_prediction_shape = None
            if prediction_for_step is not None:
                actual_prediction_shape = plan_execution_shape(
                    plan, waiting=[s.req for s in waiting], decoding=[s.req for s in decoding],
                    prefilling=[(s.req, s.prefill_remaining) for s in prefilling],
                    per_request_kv_lengths={s.req.req_id: int(s.req.prompt_len + s.req.gen_len
                                                            - s.decode_remaining - 1)
                                            for s in decoding}).describe()
            for it in batch:
                s = by_id[it.req.req_id]
                if s.first_schedule_ms is None:
                    s.first_schedule_ms = step_start

            if not batch:
                if prediction_for_step is not None:
                    controller_trace.append({"step": step, "prediction_execution_match":
                        _prediction_execution_observation(prediction_for_step, actual_step=step,
                            actual_shape=actual_prediction_shape, timing=None,
                            hardware_backed=drv.hardware_backed,
                            unexecuted_reason="empty_execution_batch")})
                # 本步无活可干。三种情形必须分开，否则会静默空转到 max_steps：
                #   (a) 还有请求没到 → 真的等到它到达（不"跳过"空闲：
                #       跳过会把空闲从 wall_ms 里抹掉，吞吐虚高）；
                #   (b) 没有待到达的请求，但仍有未完成请求 → 死锁，显形退出；
                #   (c) 全部完成 → 循环条件自会退出。
                if not_arrived:
                    target = not_arrived[0].arrival_ms
                    wait_ms = target - now_ms()
                    if closed_loop or wait_ms <= 0:
                        step += 1
                        now = now_ms()
                        continue
                    self._sleep(wait_ms / 1000.0)
                    idle_ms += wait_ms
                    n_idle_steps += 1
                    now = now_ms()
                    continue
                if not all_settled():
                    busy = [s.req.req_id for s in enqueued if s.phase != "done"]
                    stall_reason = (
                        f"step {step} 没有可推进的请求，也没有待到达的请求，"
                        f"但仍有 {len(busy)} 个请求未完成：{busy[:8]}。"
                        "这是**死锁**（不是「预算不足」）：典型原因是某个请求的 "
                        "prompt 在关闭分块时超过单步 token 预算，永远拿不到额度。"
                    )
                break

            try:
                outcome = drv.forward(batch)
            except Exception as exc:  # noqa: BLE001
                # 驱动抛错必须**显形**：把这些请求记成 failed 并终止，
                # 而不是让整次 run 崩掉或悄悄少跑几个请求。
                driver_error = f"{type(exc).__name__}: {exc}"
                if prediction_for_step is not None:
                    controller_trace.append({"step": step, "prediction_execution_match":
                        _prediction_execution_observation(prediction_for_step, actual_step=step,
                            actual_shape=actual_prediction_shape, timing=None,
                            hardware_backed=drv.hardware_backed,
                            unexecuted_reason="driver_failed")})
                for it in batch:
                    st = states[it.req.req_id]
                    st.outcome = OUTCOME_FAILED
                    st.failure = driver_error
                    st.phase = "done"
                break

            ready_by_id = {r.req_id: r for r in outcome.slots}
            step_end_ms = (outcome.timing.finished_s - origin) * 1000.0
            if prediction_for_step is not None:
                controller_trace.append({"step": step, "prediction_execution_match":
                    _prediction_execution_observation(prediction_for_step, actual_step=step,
                        actual_shape=actual_prediction_shape, timing=outcome.timing,
                        hardware_backed=drv.hardware_backed)})

            # 3b) 分层信号：本步的耗时归到 prefill 还是"纯 decode"。
            #     ⚠️ **必须在结算之前读相位**——结算会把本步算完 prefill 的请求
            #     改成 decode/done，那时再读就一律"没有 prefill"，判据恒假。
            #     "纯 decode" 的定义是本步**完全**没有 prefill 工作：
            #     同时含 prefill 与 decode 的混合步**不算**纯 decode（这正是要被减掉的东西）。
            _step_has_prefill = any(
                states[it.req.req_id].phase == "prefill" for it in batch
            )
            _step_ms = max(0.0, step_end_ms - step_start)
            if _step_has_prefill:
                prefill_clock_ms += _step_ms
            else:
                decode_clock_ms += _step_ms
            step_trace.append(
                (step_end_ms, _step_has_prefill, 0.0 if _step_has_prefill else _step_ms)
            )

            # 4) 结算：prefill 完成的请求进 decode；decode 各推进 1 个 token。
            #    token 的可得时刻取自驱动的 ``ready_s``——那是**采样与设备→主机
            #    取回都已完成之后**那一刻（"token 值在主机可取回"）。**不是**
            #    "设备把 logits 算完"那一刻：两者之间隔着采样 + D2H，用后者会把
            #    TTFT/E2E 系统性记早，并让逐步间隔（stall）归属错位。故驱动另有一个
            #    ``logits_ready_s`` 专门表示"算完"，本处不取它。
            #    因此 first_token_ms / last_emit_ms 是真实到达时刻，
            #    stall（间隔 P99）才有意义。
            for it in batch:
                s = states[it.req.req_id]
                res = ready_by_id.get(it.req.req_id)
                if res is None:
                    raise RuntimeError(
                        f"驱动未返回请求 {it.req.req_id} 的结果；"
                        "批与结果不配对会让该请求的时刻静默缺失"
                    )
                s.prefill_remaining = max(0, s.prefill_remaining - it.n_new_tokens)
                if res.token_id is None:
                    # prompt 还没算完（分块预填充中途），本步不产出 token。
                    continue
                emit_ms = (
                    (res.ready_s - origin) * 1000.0
                    if res.ready_s is not None
                    else step_end_ms
                )
                if s.first_token_ms is None:
                    s.first_token_ms = emit_ms
                    s.ttft_ms = emit_ms - (s.arrival_ms or 0.0)
                    recent_ttft.append(s.ttft_ms)
                    s.last_emit_ms = emit_ms
                    # 首 token 时把两个时钟的读数打底；第一个间隔从"首 token 之后"才成立，
                    # 因此**不为首 token 记 interval**（与 recent_tpot 的既有口径一致）。
                    s.prefill_clock_at_last_emit = prefill_clock_ms
                    s.decode_clock_at_last_emit = decode_clock_ms
                    if s.decode_remaining > 0:
                        s.phase = "decode"
                    else:
                        s.phase = "done"
                        s.done_ms = emit_ms
                else:
                    interval = emit_ms - s.last_emit_ms
                    s.last_emit_ms = emit_ms
                    s.intervals.append(interval)      # ← 用户感知的原始间隔，**必须保留**
                    recent_tpot.append(interval)
                    # 分层：同一个间隔的两个分解量。
                    #   had_prefill：这段等待里**任何一步**做过 prefill 吗
                    #                （时钟差 > 0 ⇒ 中间有 prefill，哪怕最后一步是纯 decode）
                    #   decode_ms  ：这段等待里**纯 decode 步**一共花了多少
                    #                ⇒ interval 大而 decode_ms 不变 = "被 prefill 挤的"，
                    #                  不是 decode 本身变慢。
                    recent_had_prefill.append(
                        prefill_clock_ms > s.prefill_clock_at_last_emit
                    )
                    recent_decode_ms.append(
                        decode_clock_ms - s.decode_clock_at_last_emit
                    )
                    # 同两个量**逐请求**再存一份（与 s.intervals 逐项对齐）：
                    # 验收要按请求核对"该间隔跨了哪些步"，全局窗口做不到一一对应。
                    s.interval_had_prefill.append(
                        prefill_clock_ms > s.prefill_clock_at_last_emit
                    )
                    s.interval_decode_ms.append(
                        decode_clock_ms - s.decode_clock_at_last_emit
                    )
                    s.prefill_clock_at_last_emit = prefill_clock_ms
                    s.decode_clock_at_last_emit = decode_clock_ms
                    s.decode_remaining -= 1
                    if s.decode_remaining <= 0:
                        s.phase = "done"
                        s.done_ms = emit_ms

            # 4b) **及时释放**刚完成的请求。
            #     设备侧 KV 与 prompt_len + gen_len 成正比；留到最后统一释放，
            #     并发为 1 跑 12 个请求就会同时驻留 12 份。短负载看不出来，
            #     长负载上表现为 OOM——而那与"负载太重"长得一样。
            #     必须在**没有别的东西引用这些张量**之后调：批内其余请求的
            #     历史已在写回时各自切好，不再依赖本序列的 KV。
            #
            # 采样峰值**必须在回收之前**：先回收再采，峰值永远是 0，
            # 于是"驻留随请求总数增长"这个缺陷量不出来（这正是本轮要抓的东西）。
            max_live = max(max_live, drv.live_slots())
            finished_now = [
                s.req.req_id
                for s in enqueued
                if s.phase == "done" and s.req.req_id not in retired
            ]
            if finished_now:
                drv.retire(finished_now)
                retired.update(finished_now)
                max_live = max(max_live, drv.live_slots())

            # 5) 遥测：字段与 dry-run 逐列对齐，汇总/绘图无需分支。
            if cfg.controller.enabled:
                chunk_requested = controller.state.chunk_size
                tokens_requested = controller.state.max_batched_tokens
            else:
                chunk_requested = cfg.chunk.chunk_size
                tokens_requested = cfg.chunk.max_num_batched_tokens
            # 显存采样按控制节拍做（它是一次驱动查询，不该每步都做——
            # 每步都采会把查询开销算进步耗时）。没采的步写 None，不写 0。
            dev = (
                drv.sample_device()
                if step % max(1, cfg.run.control_interval_steps) == 0
                else {}
            )
            samples.append(
                TelemetrySample(
                    step=step,
                    t_ms=round(step_start, 4),
                    queue_depth=max(0, len(waiting) - len(new_starts)),
                    running=len(decoding) + len(prefilling),
                    scheduled_tokens=outcome.timing.n_tokens,
                    ttft_ms=(
                        round(percentile(recent_ttft[-8:], 50), 3)
                        if recent_ttft
                        else None
                    ),
                    tpot_ms=(
                        round(percentile(recent_tpot[-8:], 50), 3)
                        if recent_tpot
                        else None
                    ),
                    mem_mb=dev.get("used_mib"),
                    chunk_size=plan.effective_chunk_limit,
                    chunk_size_requested=chunk_requested,
                    max_batched_tokens=cur_max_tokens,
                    max_batched_tokens_requested=tokens_requested,
                    prefill_budget_effective=plan.prefill_budget,
                    note=(
                        f"prefill_tok={plan.tokens_prefilled} "
                        f"decode={len(decode_targets)} "
                        f"forwards={outcome.timing.n_forwards} "
                        f"fwd_ms={outcome.timing.fwd_ms:.2f} "
                        f"pack_ms={outcome.timing.pack_ms:.2f} "
                        # 采样 + 取回主机：它是 TTFT/E2E 的一部分，但不属于设备计算。
                        # 记出来是为了让"token 何时可知"这个口径可被复核。
                        f"readback_ms={outcome.timing.readback_ms:.2f}"
                    ),
                )
            )

            # 6) 控制器：与 dry-run 同一节拍、同一观测口径。
            if cfg.controller.enabled and step % max(1, cfg.run.control_interval_steps) == 0:
                system_state = monitor.observe(
                    requests=[s.req for s in waiting],
                    step=step,
                    queue_depth=len(waiting),
                    running=len(decoding),
                    concurrency=len(decoding) + len(prefilling),
                    # 缓存关闭时不传 stats：关闭的缓存会读出 0.0 命中率，
                    # 那会被读成"开着但一次没命中"，与事实相反。
                    cache_stats=cache.stats if cfg.cache.enabled else None,
                    ttft_samples=recent_ttft[-32:],
                    tpot_samples=recent_tpot[-32:],
                )
                action = controller.observe(
                    step=step,
                    ttft_samples=recent_ttft[-32:],
                    tpot_samples=recent_tpot[-32:],
                    # 分层信号（第 04 步）：与上面**同一窗口**取，保证三组样本一一对应
                    # （长度相等 ⇒ 第 i 个 tpot 样本对应第 i 个 decode/flag 样本）。
                    decode_samples=recent_decode_ms[-32:],
                    had_prefill_samples=recent_had_prefill[-32:],
                    observed_at_ms=step_end_ms,
                    queue_depth=len(waiting),
                    running=len(decoding),
                    system_state=system_state,
                )
                cur_chunk = controller.state.chunk_size
                cur_max_tokens = controller.state.max_batched_tokens
                controller_trace.append({"step": step, **action.describe()})
                system_states.append(system_state.describe())
                # ★ 策略臂（第 12/13 步）：与模拟器**共用同一份决策实现**
                # （`ascend_opt.arm.apply_arm`）。实测踩到过：只接进模拟器 ⇒ 真机上
                # `slo_weave` 决策数 = 0、两臂产物逐项相同，而配置写着 kind=slo_weave。
                if arm is not None:
                    cur_chunk, cur_max_tokens, _entry = apply_arm(
                        kind=cfg.controller.kind, policy=arm,
                        shadow=cfg.controller.shadow, step=step, now_ms=step_end_ms,
                        cur_chunk=cur_chunk, cur_max_tokens=cur_max_tokens,
                        # The just-finished forward changed these phases; the
                        # old lists may still contain completed/new decode rows.
                        waiting=[s for s in enqueued if s.phase == "waiting"],
                        decoding=[s for s in enqueued if s.phase == "decode"],
                        prefilling=[s for s in enqueued if s.phase == "prefill"],
                        scheduler=scheduler, chunk_policy=chunk_policy, cache=cache,
                        limits=ChunkLimits(chunk_size=cur_chunk,
                                           max_num_batched_tokens=cur_max_tokens),
                        max_num_seqs=cfg.max_num_seqs, chunk_enabled=cfg.chunk.enabled,
                    )
                    if _entry is not None:
                        controller_trace.append(_entry)
                        pending_prediction = _pending_execution_prediction(
                            _entry, decision_step=step, shadow=cfg.controller.shadow)

            now = step_end_ms
            step += 1

        # 测量窗口到此为止。**先关窗口再回收**：retire 会释放显存、
        # 可能触发同步，把它算进 wall_ms 会让吞吐偏低。
        wall_ms = now_ms()
        # 收尾清理：只剩**没有正常走到 done** 的请求（失败/未完成/死锁退出）。
        # 正常完成的请求已经在 4b) 里逐条回收过了——这里再兜一次底，
        # 而不是把"全轮统一释放"当成主路径（那正是长负载 OOM 的来源）。
        leftover = [s.req.req_id for s in states.values() if s.req.req_id not in retired]
        if leftover:
            drv.retire(leftover)
            retired.update(leftover)
        self.backend_stats = {
            "driver": drv.describe(),
            "driver_stats": drv.stats(),
            "load": load_info,
            "warmup": warm_info,
            "warmup_shape": shape,
            "wall_ms_measured_window": round(wall_ms, 3),
            "idle_ms": round(idle_ms, 3),
            "n_idle_steps": n_idle_steps,
            # 及时释放的**可观测量**：闭环并发为 k 时，峰值驻留不应显著超过 k。
            # 只写"已释放"是不够的——那句话在"整轮结束才释放"的实现里同样成立。
            "kv_lifecycle": {
                "n_requests": len(states),
                "max_live_slots": max_live,
                "live_slots_at_end": drv.live_slots(),
                "n_retired": len(retired),
                "retired_per_request": len(retired) == len(states),
            },
            "driver_error": driver_error or None,
            "arrival_model": (
                "closed" if closed_loop else ("open_loop" if spec is None else spec.arrival)
            ),
            # ---- 分层控制信号（第 04 步）----
            # 落盘的是**原始样本**而不是汇总值：汇总口径还会变（第 09 步才定策略），
            # 而"哪一种停顿"这个事实一旦丢掉就补不回来。
            "layered_signals": {
                "n_tpot_samples": len(recent_tpot),
                "n_decode_samples": len(recent_decode_ms),
                "n_had_prefill_samples": len(recent_had_prefill),
                # 三组样本**等长且一一对应**（同一个 interval 的三个视角）
                "aligned": (
                    len(recent_tpot) == len(recent_decode_ms) == len(recent_had_prefill)
                ),
                "n_intervals_with_prefill": sum(1 for x in recent_had_prefill if x),
                "prefill_clock_ms": round(prefill_clock_ms, 3),
                "decode_clock_ms": round(decode_clock_ms, 3),
                #: 每步 (时刻, 本步是否含 prefill, 本步纯 decode 耗时)。
                #: **可复核性**：由它可独立重算 recent_had_prefill / recent_decode_ms，
                #: 因此"加遥测没有改变调度与 token 语义"是可从产物验证的，不靠声明。
                "n_steps": len(step_trace),
                "step_trace": [
                    [round(t, 3), bool(hp), round(pd, 3)] for t, hp, pd in step_trace
                ],
                #: 逐请求的三个序列（与用户感知的 ``intervals`` **逐项对齐**）。
                #: 放这里而不是只留全局滚动窗口：验收与第 09 步都要**按请求**
                #: 把"某次停顿"对回到"那段时间里发生了什么"。
                "per_request": {
                    s.req.req_id: {
                        "intervals_ms": [round(x, 3) for x in s.intervals],
                        "had_prefill": list(s.interval_had_prefill),
                        "decode_ms": [round(x, 3) for x in s.interval_decode_ms],
                    }
                    for s in states.values()
                    if s.intervals
                },
            },
            "notes": {
                "prompt_content": "synthetic_by_length",
                "prefix_cache": "not_implemented",
                "idle_ms": (
                    "空闲等待（等待下一个到达）已计入 wall_ms；"
                    "它是真实流逝时间，不是慢"
                ),
            },
        }

        metrics: list[RequestMetrics] = []
        for s in states.values():
            tpot = sum(s.intervals) / len(s.intervals) if s.intervals else 0.0
            tpot_p99 = percentile(s.intervals, 99) if s.intervals else 0.0
            # 结局在这里定：跑完了才算 success。
            if s.phase == "done" and s.outcome != OUTCOME_FAILED:
                s.outcome = OUTCOME_SUCCESS
            metrics.append(
                RequestMetrics(
                    req_id=s.req.req_id,
                    prompt_len=s.req.prompt_len,
                    gen_len=s.req.gen_len,
                    ttft_ms=s.ttft_ms,
                    tpot_ms=tpot,
                    tpot_p99_ms=tpot_p99,
                    cached_tokens=s.cached_tokens,
                    scheduled_arrival_ms=s.scheduled_arrival_ms,
                    arrival_ms=s.arrival_ms,
                    enqueue_ms=s.enqueue_ms,
                    first_schedule_ms=s.first_schedule_ms,
                    first_token_ms=s.first_token_ms,
                    done_ms=s.done_ms,
                    outcome=s.outcome,
                )
            )

        return ExecutionResult(
            metrics=metrics,
            samples=samples,
            wall_ms=wall_ms,
            steps=step,
            dropped=sum(1 for s in states.values() if s.phase != "done"),
            controller_trace=controller_trace,
            system_states=system_states,
            stall_reason=stall_reason,
            # ★ 策略臂自己的计数器（`n_degraded` / `n_prefill_starved` / `n_fallback`）。
            # ⚠️ 与模拟器路径**共用同一个函数**（`ascend_opt.arm.arm_state`）：
            # 两条路径各写一份 ⇒ 修了一边另一边不修，正是第 13 步踩过的坑。
            arm_state=arm_state(arm),
            backend_stats=self.backend_stats,
        )

    @staticmethod
    def _representative_shape(requests: Sequence[Request]) -> dict[str, Any]:
        """预热用形状：取工作负载的中位长度。

        用**中位数**而不是最大值：按最大值预热会让每次预热都跑最重的形状
        （可能几倍于正式段的平均开销），而用最小值又覆盖不到正式段里更长的
        那批请求。中位数是"代表性"与"不过度"之间最不坏的选择，
        实际用的值随结果一起落盘（``warmup_shape``），读的人可据此判断贴合度。
        """
        if not requests:
            return {"prompt_len": 1, "gen_len": 1, "n_requests": 0}
        p = sorted(r.prompt_len for r in requests)
        g = sorted(r.gen_len for r in requests)
        return {
            "prompt_len": int(p[len(p) // 2]),
            "gen_len": int(g[len(g) // 2]),
            "n_requests": len(requests),
        }


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #

#: 允许的驱动名。**白名单**：拼错或写到不存在的驱动必须当场报错，
#: 而不是静默退回某个默认实现——那会让一次实验用的驱动与配置声明不一致。
_DRIVER_KINDS = ("transformers",)


def _reject_unimplemented_runtime_keys(rt: dict[str, Any]) -> None:
    """整段配置里出现了声明"未实现能力"的键 → 报错。

    单看 ``quantization == "none"`` 还不够：``quantization_detail`` /
    ``capture`` 这些键的存在本身就说明有人在配置里登记了一项量化校准或
    图捕获设置。留着它们会让配置看起来"这部分已经配好了"，
    而实际一个字段都没被执行。报错让人要么删掉、要么去实现。

    **顺序很重要**：先查 ``_UNIMPLEMENTED_RUNTIME_KEYS``（非空即声明），
    再查 ``kv_cache_dtype`` 是否落在 ``_NOOP_RUNTIME_VALUES`` 之外的取值。
    把两者合成一个条件会让报错信息指向"某个键存在"，
    而真正的问题可能是"某个键取了不支持的**值**"——报错指错方向比不报错更费时间。
    """
    present = [k for k in _UNIMPLEMENTED_RUNTIME_KEYS if rt.get(k)]
    if present:
        raise ValueError(
            f"runtime 里出现了本驱动未实现的配置项 {present}。\n"
            "  这些键会被写进 meta 的 reproducibility 段，于是产物自称"
            "「已按这份量化/图模式配置运行」，而实际一步都没执行。\n"
            "  请删掉它们，或先在 TransformerDriver 里实现对应行为。"
        )
    for key, allowed in _NOOP_RUNTIME_VALUES.items():
        value = rt.get(key)
        if value is None:
            continue
        if str(value).strip().lower() in allowed:
            continue
        raise ValueError(
            f"本驱动不支持 runtime.{key}={value!r}"
            f"（可用的「未生效」取值：{list(allowed)}）。\n"
            "  非默认取值意味着声明了一项未实现的缓存精度设置，"
            "而它会被原样写进 meta：产物自称用了该精度，实际没有。\n"
            "  请改回默认值，或先实现对应的 KV 缓存精度支持。"
        )


def build_driver(cfg: Any) -> GenerationDriver:
    """按配置构造真机驱动。

    ``runtime.backend_driver`` 缺省为 ``transformers``。
    假驱动**不在此白名单内**（它只应出现在单测里），因此 CLI 路径不可能拿到它。

    ``quantization`` / ``graph_mode`` / ``tensor_parallel_size`` 三个字段在配置里
    是给元数据用的标签，本驱动**只实现 none / eager / 1** 这一组取值，
    其余一律在这里报错（详见 :data:`_SUPPORTED_QUANTIZATION` 的注释）。
    """
    rt = cfg.runtime or {}
    kind = str(rt.get("backend_driver") or "transformers").strip().lower()
    if kind not in _DRIVER_KINDS:
        raise ValueError(
            f"未知 backend_driver={kind!r}；可用：{list(_DRIVER_KINDS)}。"
            "（假驱动仅供单测注入，不能通过配置选择。）"
        )
    model = rt.get("model")
    if not model:
        raise ValueError(
            "runtime.model 未配置：真机后端需要知道装哪个模型。"
            "留空会让一次 run 用上默认模型而配置里看不出来。"
        )
    # 构造驱动：它会拒绝 quantization / graph_mode / tp 的不支持取值。
    # **这一步必须排在最前**，因为它的报错最具体（点名哪个能力、列出可用取值、
    # 给出可照抄的修法）；"整段键存在性"的检查只能说"某几个键在"。
    # 顺序反了会出现"配置里明明写着 w8a8，报错却说 kv_cache_dtype 未实现"。
    driver = TransformerDriver(
        model_name=str(model),
        model_revision=rt.get("model_revision"),
        device=str(rt.get("device") or "musa:0"),
        precision=str(rt.get("precision") or "fp16"),
        max_seq_len=int(rt.get("max_seq_len") or 32768),
        do_sample=bool(rt.get("do_sample", False)),
        temperature=float(rt.get("temperature") or 1.0),
        top_p=float(rt.get("top_p") or 1.0),
        seed=int(rt.get("seed") or 0),
        quantization=str(rt.get("quantization") or "none"),
        graph_mode=str(rt.get("graph_mode") or "eager"),
        tensor_parallel_size=int(rt.get("tensor_parallel_size") or 1),
        # 执行优化开关（S01/P1 · S02/P2a · S03/P3）。**默认全关** ⇒
        # 不写这些键的既有配置行为逐字节不变。
        logits_last_only=bool(rt.get("logits_last_only", False)),
        sample_needed_only=bool(rt.get("sample_needed_only", False)),
        retire_empty_cache=bool(rt.get("retire_empty_cache", True)),
        fuse_qkv=bool(rt.get("fuse_qkv", False)),
        fuse_gate_up=bool(rt.get("fuse_gate_up", False)),
        linear_fuse_shapes=(
            None if rt.get("linear_fuse_shapes") is None
            else [int(x) for x in rt["linear_fuse_shapes"]]
        ),
        # S04/P2b-c：常驻 KV 缓冲（默认关 ⇒ 既有配置逐字节不变）。
        persistent_kv_slots=bool(rt.get("persistent_kv_slots", False)),
        persistent_kv_capacity=(
            None if rt.get("persistent_kv_capacity") is None
            else int(rt["persistent_kv_capacity"])
        ),
        # 数值验证：逐行 logits 摘要（默认关 ⇒ 既有配置逐字节不变）。
        # ⚠️ 只被 `scripts/verify_s04_reenact_numerics.py` 这条**验证**路径打开；
        # 性能批次一律不开（它会往产物里加每步的摘要，且多一次 D2H）。
        capture_logits_digest=bool(rt.get("capture_logits_digest", False)),
    )
    # 再查"剩下的、驱动构造函数管不到的"键。
    _reject_unimplemented_runtime_keys(rt)
    return driver
