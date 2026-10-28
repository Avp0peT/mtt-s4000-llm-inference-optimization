"""前缀缓存策略（KV 复用）。

模型：把 prompt 按固定 block_size 切成块，块内容以 (prefix_group, block_index) 的
确定性哈希标识。若某块已驻留在缓存的「驻留块表」中，该块对应的 prompt token 就
无需重算，直接计入 ``reused_tokens``。

这是一个**用于策略对照与容量估算的模型**，不是真实的 KV 内存管理器：
它不分配显存、不感知物理布局，只回答两个问题——
    1) 给定访问序列，命中多少 token？（收益上限）
    2) 给定显存预算，能驻留多少块？淘汰策略如何影响命中率？

真实 KV 管理由运行时（如 vllm-ascend）负责；本模块的输出用于与实测命中率对齐。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from ascend_opt.workload import Request

EvictionPolicy = Literal["lru", "fifo", "none"]


@dataclass(frozen=True)
class PrefixCacheConfig:
    enabled: bool = True
    block_size: int = 16          # 每个 KV 块的 token 数
    max_blocks: int = 8192        # 缓存可驻留的块数上限
    eviction: EvictionPolicy = "lru"
    min_reuse_blocks: int = 1     # 复用块数低于此值则不值得走缓存路径

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "block_size": self.block_size,
            "max_blocks": self.max_blocks,
            "capacity_tokens": self.max_blocks * self.block_size,
            "eviction": self.eviction,
            "min_reuse_blocks": self.min_reuse_blocks,
        }


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0                 # 至少复用 1 个块
    full_hits: int = 0            # prompt 被完全复用
    reused_tokens: int = 0
    requested_tokens: int = 0
    evictions: int = 0
    rejected_small: int = 0       # 因收益不足而放弃走缓存

    @property
    def hit_rate(self) -> float:
        """命中率。**没有查询记录时返回 0.0**——注意这与"命中率为 0"含义不同。

        要给控制器/监测器喂这个值时不要用它，用 :attr:`hit_rate_or_none`：
        0.0 会被读成"缓存开着但一次都没命中"，从而触发一次基于假信号的决策。
        """
        return self.hits / self.lookups if self.lookups else 0.0

    @property
    def hit_rate_or_none(self) -> float | None:
        """命中率；**没有任何查询时返回 ``None``（未测）**。

        为什么必须与 :attr:`hit_rate` 并存：``hit_rate`` 返回 0.0 的那个分支
        同时覆盖了两种完全不同的情形——
            1) 缓存关闭（``lookup`` 直接返回，``lookups`` 不计）
            2) 缓存开着但确实一次都没命中
        聚合与报表里把两者合并无伤大雅，但**决策**里不能合并：前者说明
        "这个信号在本路径上不存在"，后者说明"缓存策略没留住前缀"。
        """
        return self.hits / self.lookups if self.lookups else None

    @property
    def token_reuse_rate(self) -> float:
        return self.reused_tokens / self.requested_tokens if self.requested_tokens else 0.0

    def describe(self) -> dict[str, Any]:
        return {
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 4),
            "full_hits": self.full_hits,
            "reused_tokens": self.reused_tokens,
            "requested_tokens": self.requested_tokens,
            "token_reuse_rate": round(self.token_reuse_rate, 4),
            "evictions": self.evictions,
            "rejected_small": self.rejected_small,
        }


class PrefixCachePolicy:
    """块级前缀缓存模型。

    生命周期：
        lookup(req)   -> 复用 token 数（只读，不改状态）
        admit(req)    -> 把该请求计算过的块写入缓存，必要时按策略淘汰

    两个调用分开是为了让执行器在「真的算完」之后再 admit，
    避免把未完成计算的前缀提前标为可复用（真实系统里这是正确性 bug）。
    """

    name = "prefix_cache"

    def __init__(self, config: PrefixCacheConfig | None = None) -> None:
        self.config = config or PrefixCacheConfig()
        self._blocks: OrderedDict[str, int] = OrderedDict()  # block_id -> 最后访问序号
        self._clock = 0
        self.stats = CacheStats()

    # ---------------- 内部 ----------------

    def _block_ids(self, req: Request) -> list[str]:
        """按块粒度生成块标识。

        只有共享前缀部分参与缓存；请求私有部分即使长度相同也不可能命中
        （真实系统中哈希不同，这里用 req_id 参与哈希来模拟该性质）。
        """
        bs = self.config.block_size
        n_blocks = req.shared_prefix_len // bs
        if n_blocks == 0:
            return []
        group = req.prefix_group or "ungrouped"
        # 私有部分截断后不足一块的内容不构成可复用块。
        return [f"{group}:{i}" for i in range(n_blocks)]

    def _evict_if_needed(self) -> None:
        if self.config.eviction == "none":
            return
        while len(self._blocks) > self.config.max_blocks:
            if self.config.eviction == "lru":
                self._blocks.popitem(last=False)
            else:  # fifo：语义上先进先出，与 lru 的差别体现在重访是否刷新位置
                self._blocks.popitem(last=False)
            self.stats.evictions += 1

    # ---------------- 对外接口 ----------------

    def lookup(self, req: Request) -> int:
        """返回本请求可复用的 prompt token 数。"""
        if not self.config.enabled:
            return 0

        self.stats.lookups += 1
        self.stats.requested_tokens += req.prompt_len

        ids = self._block_ids(req)
        if not ids:
            return 0

        # 前缀语义：必须从第 0 块起连续命中，中途断了后面全部失效。
        matched = 0
        for block_id in ids:
            if block_id not in self._blocks:
                break
            matched += 1
            self._clock += 1
            if self.config.eviction == "lru":
                self._blocks.move_to_end(block_id)
                self._blocks[block_id] = self._clock

        if matched < self.config.min_reuse_blocks:
            if matched > 0:
                self.stats.rejected_small += 1
            return 0

        reused = matched * self.config.block_size
        # 复用不能超过实际共享前缀长度。
        reused = min(reused, req.shared_prefix_len)
        self.stats.hits += 1
        self.stats.reused_tokens += reused
        if reused >= req.prompt_len:
            self.stats.full_hits += 1
        return reused

    def admit(self, req: Request, computed_tokens: int | None = None) -> None:
        """把该请求已完成计算的块写入缓存。

        ``computed_tokens`` 给出实际算完的 token 数（默认按整个 prompt 计算）。
        只写入完整的块，半块不入缓存。
        """
        if not self.config.enabled:
            return
        bs = self.config.block_size
        total = req.prompt_len if computed_tokens is None else computed_tokens
        n_blocks = total // bs
        # 私有尾部不构成可复用前缀，最多驻留到共享前缀结束的位置。
        n_blocks = min(n_blocks, req.prompt_len // bs)

        ids = self._block_ids(req)
        for i in range(min(n_blocks, len(ids))):
            self._clock += 1
            self._blocks[ids[i]] = self._clock
            if self.config.eviction == "lru":
                self._blocks.move_to_end(ids[i])
        self._evict_if_needed()

    def warmup(self, requests: Iterable[Request]) -> None:
        """预填充缓存（模拟「缓存已有历史前缀」的起点条件）。"""
        for req in requests:
            self.admit(req)

    def reset_stats(self) -> None:
        self.stats = CacheStats()

    @property
    def resident_blocks(self) -> int:
        return len(self._blocks)

    def describe(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "config": self.config.describe(),
            "resident_blocks": self.resident_blocks,
            "stats": self.stats.describe(),
        }
