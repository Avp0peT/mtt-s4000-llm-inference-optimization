"""一步的调度决策：批预算 -> 分块额度 -> 缓存命中 -> prefill token 分配。

**为什么把它单独抽出来**

这段组合逻辑原先只存在于 ``benchmark/run_benchmark.py`` 的 ``simulate()`` 里。
真实后端如果要跑同一批实验，就必须做同样的组合——而"再写一遍"会带来一种
不会报错的失效：以后修了模拟器里的某一处闸门（例如"两处都夹取导致控制器
调大被静默压回"那个缺陷），真实路径不会跟着修。两条路径于是给出不同的
调度行为，却都自称"同一套策略"，产物里看不出任何异常。

因此本模块是**唯一**的组合点，顺序与闸门都在这里：

1. 用 ``TokenBudget`` 把单步 token / 序列预算交给调度器，拿到 prefill/decode 名单；
2. 算待办 prefill 总量，问分块策略本步能切多少（上限只在 ``ChunkLimits`` 一处）；
3. 新起步的请求先查前缀缓存，命中整段则本步即可出首 token；
4. 按"续传优先"分配 prefill token，剩余额度逐条递减。

**它不做的事**（刻意）：不推进时钟、不写任何时刻、不结算请求终态。
决策与结算是两件事——结算需要真实耗时（模拟器用设备模型，真机用实测），
把两者混在一起会让"策略错了"与"计时错了"无法区分。

本模块纯逻辑，不导入任何硬件/框架库。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .chunk_policy import ChunkLimits
from .scheduler import TokenBudget
from .workload import Request

__all__ = ["StepPlan", "plan_step"]


@dataclass
class StepPlan:
    """一步的决策结果。

    名单一律用 ``req_id`` 而不是 ``Request`` 对象：调用方各自持有自己的状态机
    （模拟器用 ``_SimRequest``，真机用设备侧句柄），按 id 回查可以避免让本模块
    依赖任何一种中间状态类型。这也是它能被两条路径共用的原因。
    """

    #: 本步获准**起步**的请求（已通过序列名额与预算闸门）。
    new_start_ids: tuple[str, ...] = ()
    #: 本步推进 decode 的请求。
    decode_ids: tuple[str, ...] = ()
    #: 新起步请求的前缀缓存命中数。命中数决定它本步要真实计算多少 token。
    cached_tokens: dict[str, int] = field(default_factory=dict)
    #: 整段前缀命中的请求：无需任何计算，本步即可产出首 token。
    #: 单独列出而不是让调用方自己从 ``cached_tokens`` 推——
    #: "命中数 == prompt_len"这个判据散落两处时，某一处漏了不会报错，
    #: 只会表现为"缓存命中却没省下 prefill 时间"。
    immediate_ids: tuple[str, ...] = ()
    #: 本步分给每个请求的 prefill token 数。**不在其中的请求本步没被服务**。
    prefill_alloc: dict[str, int] = field(default_factory=dict)
    #: 本步实际发放的 prefill token 总数（= ``prefill_alloc`` 之和）。
    tokens_prefilled: int = 0
    #: 经全部闸门之后真正可用的 prefill 额度。与"请求了多少"并排落盘，
    #: "控制器说调大了、额度没涨"才看得见。
    prefill_budget: int = 0
    #: 本步生效的分块上限（= min(建议 chunk, 单步 token 预算)）。
    effective_chunk_limit: int = 0
    #: 非空表示本步无法推进任何请求，内容即诊断信息。调用方应据此终止，
    #: 而不是继续空转到步数上限——后者只剩一个 ``dropped=N`` 且毫无线索。
    stall_reason: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "new_start_ids": list(self.new_start_ids),
            "decode_ids": list(self.decode_ids),
            "cached_tokens": dict(self.cached_tokens),
            "immediate_ids": list(self.immediate_ids),
            "prefill_alloc": dict(self.prefill_alloc),
            "tokens_prefilled": self.tokens_prefilled,
            "prefill_budget": self.prefill_budget,
            "effective_chunk_limit": self.effective_chunk_limit,
            "stall_reason": self.stall_reason,
        }


def plan_step(
    *,
    scheduler: Any,
    chunk_policy: Any,
    cache: Any,
    waiting: Sequence[Request],
    decoding: Sequence[Request],
    prefilling: Sequence[tuple[Request, int]],
    cur_chunk: int,
    cur_max_tokens: int,
    max_num_seqs: int,
    chunk_enabled: bool,
    step: int = 0,
) -> StepPlan:
    """算出本步该做什么。**无副作用**（除了 ``cache.lookup`` 会更新缓存统计）。

    参数
    ----
    ``waiting``
        已到达但还没起步的请求（队列）。
    ``decoding``
        正在 decode 的请求。它们每步各占 1 个 token 预算。
    ``prefilling``
        **已在 prefill 中**的请求及其剩余 token 数，**按入队顺序**给出。
        顺序有意义：分配额度时"续传优先"，已开工的请求不该被新请求挤掉，
        而"谁更早开工"正是这个顺序。
    ``cur_chunk`` / ``cur_max_tokens``
        控制器（或静态配置）的**建议值**。它们是建议而非既成事实——
        能兑现多少取决于下面的闸门，因此两者都会被回报给调用方落盘。

    返回值里的名单按调度与分配顺序给出，调用方应保持该顺序结算。
    """
    # 1) 批预算。``occupied_seqs`` 必须包含 prefill 中的请求：
    #    它们同样占着序列名额，漏掉会让准入超额，等它们转入 decode 时
    #    序列数突破上限、尾部请求被永久截断（不报错，只是永远不完成）。
    budget = TokenBudget(
        max_num_batched_tokens=cur_max_tokens,
        max_num_seqs=max_num_seqs,
        occupied_seqs=len(prefilling),
        # 部分起步只有分块预填充才成立；关闭时"整段一次算完"是基线语义的一部分，
        # 不能为了跑通而悄悄放宽。
        allow_partial_prefill=chunk_enabled,
    )
    plan = scheduler.plan(list(waiting), list(decoding), budget)

    new_starts = list(plan.prefill)
    decode_targets = list(plan.decode)

    # 2) 分块预算：先算待办 prefill 总量，再决定这一步切多少。
    pending_prefill = sum(remaining for _, remaining in prefilling)
    pending_prefill += sum(r.prompt_len for r in new_starts)
    # 一个超过单步 token 预算的 chunk 永远兑现不了（策略会把它夹到预算）。
    # 这里夹一次并把建议值一并回报：差额本身就是"建议未兑现"的证据。
    effective_chunk_limit = min(cur_chunk, cur_max_tokens)
    prefill_budget = chunk_policy.prefill_budget(
        pending_prefill_tokens=pending_prefill,
        running_decodes=len(decode_targets),
        budget=budget,
        limits=ChunkLimits(
            chunk_size=effective_chunk_limit,
            max_num_batched_tokens=cur_max_tokens,
        ),
    )
    # 剩下的闸门只有一道：decode 也占 token 预算，本次可用的 prefill 额度
    # 必须扣除它们，否则 tokens_prefilled + n_decode 会突破 max_num_batched_tokens。
    prefill_budget = min(prefill_budget, budget.prefill_capacity(len(decode_targets)))

    # 3) 新起步的请求：先查前缀缓存，再决定真正要算多少 token。
    cached_tokens: dict[str, int] = {}
    immediate_ids: list[str] = []
    remaining_after_cache: dict[str, int] = {}
    for r in new_starts:
        cached = cache.lookup(r)
        cached_tokens[r.req_id] = cached
        remaining = max(0, r.prompt_len - cached)
        remaining_after_cache[r.req_id] = remaining
        if remaining == 0:
            # 整段前缀命中：无需计算，首 token 立即可得。
            immediate_ids.append(r.req_id)

    # 4) 分配 prefill token：续传优先。
    alloc: dict[str, int] = {}
    left = prefill_budget
    order: list[tuple[Request, int]] = list(prefilling)
    order += [
        (r, remaining_after_cache[r.req_id])
        for r in new_starts
        if r.req_id not in set(immediate_ids)
    ]
    for r, remaining in order:
        if left <= 0:
            break
        take = min(remaining, left)
        if take > 0:
            alloc[r.req_id] = take
            left -= take

    tokens_prefilled = sum(alloc.values())

    # 5) 死锁判定：本步没有任何请求能推进，但仍有已到达的未完成请求。
    #    注意要排除"请求还没到达"的合法空转——那种情况下列表全为空。
    stall_reason = ""
    if (
        not (alloc or immediate_ids or decode_targets)
        and (waiting or prefilling or decoding)
    ):
        stall_reason = (
            f"step {step} 无法推进任何请求：waiting={len(waiting)}, "
            f"prefilling={len(prefilling)}, decoding={len(decoding)}, "
            f"prefill_budget={prefill_budget}, "
            f"max_batched_tokens={cur_max_tokens}, chunk_size={effective_chunk_limit}。"
            f"典型原因：chunk.enabled=False 时队列中存在 prompt 长度超过单步预算的请求"
            f"（整段算不完就无法起步），或 token 预算被 decode 完全占满。"
        )

    return StepPlan(
        new_start_ids=tuple(r.req_id for r in new_starts),
        decode_ids=tuple(r.req_id for r in decode_targets),
        cached_tokens=cached_tokens,
        immediate_ids=tuple(immediate_ids),
        prefill_alloc=alloc,
        tokens_prefilled=tokens_prefilled,
        prefill_budget=prefill_budget,
        effective_chunk_limit=effective_chunk_limit,
        stall_reason=stall_reason,
    )
