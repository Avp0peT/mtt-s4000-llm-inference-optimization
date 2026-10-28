"""调度策略。

职责边界：给定「等待队列 / 运行中请求 / token 预算」，产出本步的批组成。
不负责真正的 token 计算，也不感知硬件——执行器（benchmark/）负责把 BatchPlan
翻译成真实的 forward 调用。

三种递进策略，用于做单变量对照：
    FIFOScheduler               先到先服务，无批处理意识（最弱基线）
    StaticBatchScheduler        攒满一批跑到底，追求吞吐（典型 static batching）
    ContinuousBatchingScheduler decode 优先 + 空闲预算填 prefill（continuous batching）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence
from ascend_opt.workload import Request


@dataclass(frozen=True)
class TokenBudget:
    """单步能处理的 token 与序列上限。

    ``occupied_seqs`` 的存在是为了堵一个真实的漏洞：
    调用方传入的 ``running`` 只包含**正在 decode** 的请求，但此刻还有一批请求
    处于「prefill 进行中」——它们同样占着序列名额。如果调度器只按
    ``len(running)`` 计算剩余名额，就会超额准入；等这些 prefill 完成转入 decode 时，
    序列数会突破 ``max_num_seqs``，尾部请求被永久截断而饿死
    （表现为一部分请求永远不完成，而不是报错）。

    调用方负责把「不在 running 里但已占用名额」的序列数告诉调度器。
    """

    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 32
    max_prefill_tokens: int | None = None  # None 表示不额外限制 prefill 份额
    occupied_seqs: int = 0                 # running 之外已占用的序列名额
    # 是否允许「部分起步」：prompt 放不进本步预算时先算一部分。
    # 只有分块预填充开启时才应为 True。关闭时语义是「整段 prompt 一次算完」，
    # 放不下的请求无法起步——若此时队列非空且在途为零，系统就是死锁状态，
    # 模拟器会显式判定失败，而不是无限空转（见 run_benchmark.simulate）。
    allow_partial_prefill: bool = False

    def prefill_capacity(self, used_tokens: int) -> int:
        """本步还能用于 prefill 的 token 数（已计入 decode 占用）。"""
        remain = self.max_num_batched_tokens - used_tokens
        if self.max_prefill_tokens is not None:
            remain = min(remain, self.max_prefill_tokens)
        return max(0, remain)

    def seq_capacity(self, used_seqs: int) -> int:
        """还能再准入多少个序列。"""
        return max(0, self.max_num_seqs - self.occupied_seqs - used_seqs)

    def describe(self) -> dict[str, Any]:
        return {
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "max_prefill_tokens": self.max_prefill_tokens,
            "occupied_seqs": self.occupied_seqs,
            "allow_partial_prefill": self.allow_partial_prefill,
        }


@dataclass
class BatchPlan:
    """单步的批组成。"""

    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    scheduled_tokens: int = 0
    note: str = ""

    @property
    def num_seqs(self) -> int:
        return len(self.prefill) + len(self.decode)

    @property
    def is_empty(self) -> bool:
        return self.num_seqs == 0

    def describe(self) -> dict[str, Any]:
        return {
            "n_prefill": len(self.prefill),
            "n_decode": len(self.decode),
            "scheduled_tokens": self.scheduled_tokens,
            "note": self.note,
        }


class SchedulerPolicy:
    """调度策略基类。

    子类必须实现 ``plan``，并保证：
    - 不修改传入的 Request（它们是共享的不可变对象）；
    - 同一个 (waiting, running, budget) 输入产出相同的 plan（可重放）；
    - ``scheduled_tokens`` 不超过 budget.max_num_batched_tokens。
    """

    name: str = "scheduler"

    def plan(
        self,
        waiting: Sequence[Request],
        running: Sequence[Request],
        budget: TokenBudget,
    ) -> BatchPlan:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name}


class FIFOScheduler(SchedulerPolicy):
    """最弱基线：按到达顺序取满批，不做 prefill/decode 区分。

    存在的意义是提供「没有调度优化」的下界。
    """

    name = "fifo"

    def plan(self, waiting, running, budget) -> BatchPlan:
        # FIFO 的次序即到达次序：running 早于 waiting，
        # 且不区分 prefill / decode 的优先级（这正是它作为最弱基线的意义）。
        capacity = budget.seq_capacity(0)
        plan = BatchPlan()
        used = 0

        for req in running:
            if len(plan.decode) >= capacity:
                break
            if used + 1 > budget.max_num_batched_tokens:
                break
            plan.decode.append(req)
            used += 1

        for req in waiting:
            if plan.num_seqs >= capacity:
                break
            remaining = budget.max_num_batched_tokens - used
            if remaining <= 0:
                break
            if req.prompt_len > remaining:
                # 放不下整段：只有在开启分块（允许部分起步）时才部分准入；
                # 否则本步容纳不了，直接停下——交由上层判定是否死锁。
                if not budget.allow_partial_prefill:
                    break
                take = remaining
            else:
                take = req.prompt_len
            plan.prefill.append(req)
            used += take
        plan.scheduled_tokens = used
        plan.note = "fifo: 不区分 prefill/decode"
        return plan


class StaticBatchScheduler(SchedulerPolicy):
    """静态批：攒满一批后独占跑到全部完成，期间不接纳新请求。"""

    name = "static_batch"

    def plan(self, waiting, running, budget) -> BatchPlan:
        if running or budget.occupied_seqs:
            # 批未跑完（含仍在 prefill 的成员），继续推进当前批，不补位。
            plan = BatchPlan(decode=list(running))
            plan.scheduled_tokens = len(running)
            plan.note = "static: 当前批未完成，拒绝补位"
            return plan

        plan = BatchPlan()
        used = 0
        for req in waiting:
            if plan.num_seqs >= budget.seq_capacity(0):
                break
            remaining = budget.max_num_batched_tokens - used
            if remaining <= 0:
                break
            if req.prompt_len > remaining:
                if not budget.allow_partial_prefill:
                    break
                take = remaining
            else:
                take = req.prompt_len
            plan.prefill.append(req)
            used += take
        plan.scheduled_tokens = used
        plan.note = "static: 开新批"
        return plan


class ContinuousBatchingScheduler(SchedulerPolicy):
    """连续批：decode 优先占位，剩余 token 预算填 prefill。

    ``reserve_decode_slots`` 用于给「即将完成 prefill 的请求」预留 decode 位置，
    避免 decode 被长 prefill 饿死（这是 TPOT 尾延迟的主要来源之一）。
    """

    name = "continuous_batching"

    def __init__(self, reserve_decode_slots: int = 0) -> None:
        if reserve_decode_slots < 0:
            raise ValueError("reserve_decode_slots 不能为负")
        self.reserve_decode_slots = reserve_decode_slots

    def plan(self, waiting, running, budget) -> BatchPlan:
        plan = BatchPlan()
        # 1) decode 优先：**所有**运行中的请求都要推进。
        #    这里刻意不做截断：截断会让被截掉的请求再也拿不到 token，
        #    表现是「请求永远不完成」而不是报错，属于最难查的一类问题。
        #    序列数上限应当由准入阶段（seq_capacity）保证，而不是在这里丢弃。
        plan.decode = list(running)
        used = len(plan.decode)

        # 2) 剩余预算按 FIFO 填 prefill。
        #    seq_capacity 必须扣除 occupied_seqs（正在 prefill 的请求），
        #    否则超额准入会在它们转入 decode 时把序列数顶过上限。
        remaining_seqs = max(
            0, budget.seq_capacity(len(plan.decode)) - self.reserve_decode_slots
        )
        for req in waiting:
            if remaining_seqs <= 0:
                break
            capacity = budget.prefill_capacity(used)
            if capacity <= 0:
                # 预算被 decode 占满，本步无法再起步任何 prefill。
                break
            if capacity < req.prompt_len:
                if budget.allow_partial_prefill:
                    # 分块开启：先算一部分，剩下的留给后续步。
                    plan.prefill.append(req)
                    used += capacity          # 吃满剩余预算
                    remaining_seqs -= 1
                    break
                # 未开启分块：整段算不完就不能起步。
                # 跳过它继续尝试更短的请求，避免一个长 prompt 阻塞整个队列。
                continue
            plan.prefill.append(req)
            used += req.prompt_len
            remaining_seqs -= 1

        plan.scheduled_tokens = used
        note = f"continuous: decode={len(plan.decode)} prefill={len(plan.prefill)}"
        if len(running) > budget.max_num_seqs:
            note += (
                f" [invariant violated: in-flight decode {len(running)} > "
                f"max_num_seqs {budget.max_num_seqs}]"
            )
        plan.note = note
        return plan

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["reserve_decode_slots"] = self.reserve_decode_slots
        return base


_REGISTRY: dict[str, type[SchedulerPolicy]] = {
    "fifo": FIFOScheduler,
    "static_batch": StaticBatchScheduler,
    "continuous_batching": ContinuousBatchingScheduler,
}


def build_scheduler(name: str, **kwargs: Any) -> SchedulerPolicy:
    """从配置构造调度策略。未知名称直接报错，不做静默回退。"""
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"未知调度策略 {name!r}；可用：{sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](**kwargs)


def available_schedulers() -> list[str]:
    return sorted(_REGISTRY)
