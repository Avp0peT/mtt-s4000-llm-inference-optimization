"""分块预填充策略（chunked prefill）。

问题：长 prompt 的 prefill 会独占算力，导致同一批里的 decode 步骤被推迟数秒，
表现为 TPOT 尾延迟（P99）爆掉。把 prefill 切成小块、与 decode 交织执行，
可以把尖峰摊平，代价是长 prompt 的 TTFT 略微变长。

本模块只决定「这一步给 prefill 多少 token」，不决定具体切哪个请求
（那是 scheduler 的职责）。两者组合才构成完整的 chunked prefill。

静态配置 vs 运行期上限（为什么需要 :class:`ChunkLimits`）
--------------------------------------------------------
``ChunkConfig`` 是 **frozen 的静态配置**，同时被本模块当作硬上限使用。这本身没问题，
直到运行期控制器开始改 ``chunk_size``：策略里的 ``min(config.chunk_size, ...)``
会把控制器调大的动作**静默压回原值**——控制器日志显示"已调大到 8192"，
执行侧拿到的额度一动不动。这类失效不抛错、不让账目不自洽，
只能靠"建议值 vs 实际生效值"的对照发现，而对照的前提是两者都被记录下来。

因此把运行期上限**显式**传入（:class:`ChunkLimits`），策略不再自己假设
"配置值就是天花板"。``limits=None`` 表示无控制器，行为与引入本类之前逐字一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ascend_opt.scheduler import TokenBudget


@dataclass(frozen=True)
class ChunkConfig:
    enabled: bool = True
    chunk_size: int = 4096              # 单个 prefill 分块的 token 上限
    max_num_batched_tokens: int = 8192  # 单步 token 总预算
    decode_reserve_ratio: float = 0.0   # 为 decode 保留的预算比例 [0, 1)
    min_decode_seqs: int = 1            # 无论多忙都要保证能推进的 decode 序列数

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须为正")
        if not 0.0 <= self.decode_reserve_ratio < 1.0:
            raise ValueError("decode_reserve_ratio 必须在 [0, 1) 区间")

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "chunk_size": self.chunk_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "decode_reserve_ratio": self.decode_reserve_ratio,
            "min_decode_seqs": self.min_decode_seqs,
        }


@dataclass(frozen=True)
class ChunkLimits:
    """本步**运行期**的分块上限——即控制器给出的建议值。

    与 :class:`ChunkConfig` 的分工：

    - ``ChunkConfig``：实验的静态设定。它决定"这次实验登记的 chunk_size 是多少"，
      是溯源信息的一部分，运行期不得被改写（frozen 就是为此）。
    - ``ChunkLimits``：本步实际生效的天花板。它随负载变化，
      是 :class:`~ascend_opt.controller.ControllerState` 的输出。

    两者分开之后，"控制器想调多大"（建议值）与"策略实际允许多大"（生效值）
    才有机会被同时记录并比对。合并成一个数的后果见模块 docstring。

    ``None`` 字段 = **不覆盖**，沿用 ``ChunkConfig`` 的静态值。
    """

    chunk_size: int | None = None
    max_num_batched_tokens: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("chunk_size", self.chunk_size),
            ("max_num_batched_tokens", self.max_num_batched_tokens),
        ):
            if value is not None and value <= 0:
                # 0 或负数意味着"本步一个 prefill token 都不给"，那不是"上限"，
                # 是死锁的配方；拼错的值必须当场暴露。
                raise ValueError(f"ChunkLimits.{name} 必须为正整数或 None，收到 {value!r}")

    def chunk_ceiling(self, config: ChunkConfig) -> int:
        """本步实际生效的分块上限。"""
        return config.chunk_size if self.chunk_size is None else self.chunk_size

    def token_ceiling(self, config: ChunkConfig) -> int:
        """本步实际生效的单步 token 上限。"""
        if self.max_num_batched_tokens is None:
            return config.max_num_batched_tokens
        return self.max_num_batched_tokens

    def describe(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "_note": "None = 不覆盖 ChunkConfig 的静态值",
        }


class ChunkPolicy:
    """分块策略基类：返回本步可用于 prefill 的 token 数。"""

    name = "chunk"

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self.config = config or ChunkConfig()

    def prefill_budget(
        self,
        *,
        pending_prefill_tokens: int,
        running_decodes: int,
        budget: TokenBudget,
        limits: ChunkLimits | None = None,
    ) -> int:
        """本步可用于 prefill 的 token 数。

        ``limits`` 是运行期上限（控制器建议值）；``None`` 表示沿用静态配置。
        """
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"policy": self.name, "config": self.config.describe()}


class FixedChunkPolicy(ChunkPolicy):
    """固定分块：每步最多 chunk_size 个 prefill token，不感知 decode 压力。

    这是 chunked prefill 的对照基线——用于证明「仅仅切片」是否足够，
    还是必须让切片大小随 decode 压力变化。

    "固定"指的是**不随 decode 压力变化**，而不是"上限不可改"：
    运行期上限仍由调用方经 ``limits`` 给出。控制器启用时它会变，
    这正是"静态切片 vs 自适应切片"的对照所在。
    """

    name = "fixed_chunk"

    def prefill_budget(
        self, *, pending_prefill_tokens, running_decodes, budget, limits=None
    ) -> int:
        if not self.config.enabled:
            return budget.max_num_batched_tokens
        if pending_prefill_tokens <= 0:
            return 0
        lim = limits or ChunkLimits()
        # 固定切片：不感知 decode 压力，这是它的对照意义所在。
        # 但三个上限都要认：运行期上限、策略的 token 上限、本步 token 预算。
        return min(
            lim.chunk_ceiling(self.config),
            lim.token_ceiling(self.config),
            budget.max_num_batched_tokens,
        )


class DecodePriorityChunkPolicy(ChunkPolicy):
    """decode 优先分块：先从预算中扣除 decode 占用，再按 chunk_size 切分。

    比 FixedChunkPolicy 多的那一层是「decode 永远有位置」——即使 prefill 队列积压，
    也不会把已经在生成的请求挤停。
    """

    name = "decode_priority_chunk"

    def prefill_budget(
        self, *, pending_prefill_tokens, running_decodes, budget, limits=None
    ) -> int:
        if not self.config.enabled:
            return budget.max_num_batched_tokens
        if pending_prefill_tokens <= 0:
            return 0

        cfg = self.config
        lim = limits or ChunkLimits()
        total = min(budget.max_num_batched_tokens, lim.token_ceiling(cfg))

        # 1) decode 保留：至少 min_decode_seqs 个位置，另有比例保留。
        reserve = max(cfg.min_decode_seqs, int(total * cfg.decode_reserve_ratio))
        reserve = min(reserve, running_decodes) if running_decodes > 0 else 0

        available = max(0, total - reserve)
        chunk = min(lim.chunk_ceiling(cfg), available)

        # 2) 小步长保护：切得太碎会让每步固定开销占比过高。
        #    当剩余 prefill 不足一个 chunk 时直接给完整剩余量，不做无意义的再切分。
        if pending_prefill_tokens <= chunk:
            return pending_prefill_tokens
        return chunk


class QueueAwareChunkPolicy(ChunkPolicy):
    """队列感知分块：短队列时用大块（优先 TTFT），长队列时用小块（优先 TPOT）。

    这是「自适应」维度的最小实现，用于验证一条假设：
    最优 chunk_size 不是常数，而是队列深度的函数。
    """

    name = "queue_aware_chunk"

    def __init__(self, config: ChunkConfig | None = None, long_queue_threshold: int = 4) -> None:
        super().__init__(config)
        self.long_queue_threshold = long_queue_threshold

    def prefill_budget(
        self, *, pending_prefill_tokens, running_decodes, budget, limits=None
    ) -> int:
        if not self.config.enabled:
            return budget.max_num_batched_tokens
        if pending_prefill_tokens <= 0:
            return 0

        cfg = self.config
        lim = limits or ChunkLimits()
        total = min(budget.max_num_batched_tokens, lim.token_ceiling(cfg))
        congested = running_decodes >= self.long_queue_threshold

        # 拥塞时退化为上限的一半，给 decode 让路。
        effective_chunk = lim.chunk_ceiling(cfg) // 2 if congested else lim.chunk_ceiling(cfg)
        reserve = max(0, running_decodes)
        available = max(0, total - reserve)
        return min(effective_chunk, available, pending_prefill_tokens)

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        base["long_queue_threshold"] = self.long_queue_threshold
        return base


_REGISTRY: dict[str, type[ChunkPolicy]] = {
    "fixed_chunk": FixedChunkPolicy,
    "decode_priority_chunk": DecodePriorityChunkPolicy,
    "queue_aware_chunk": QueueAwareChunkPolicy,
}


def build_chunk_policy(name: str, **kwargs: Any) -> ChunkPolicy:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"未知分块策略 {name!r}；可用：{sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


def available_chunk_policies() -> list[str]:
    return sorted(_REGISTRY)
