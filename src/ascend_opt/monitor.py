"""工作负载与系统状态观测（规范第 12 节的 ``WorkloadMonitor``）。

职责边界（这是本模块存在的全部意义）
------------------------------------
本模块**只观测与汇总**：把散落在各处的原始读数收敛成规范第 2 节定义的
状态向量 ``s_t``，供控制器消费。它**不做任何决策**——决策在 controller.py，
把两者混在一起会让"控制器为什么这么动"变得无法复盘。

    s_t = [
        input_length,            输入长度
        estimated_output_length, 预估输出长度
        queue_length,            队列深度
        concurrency,             并发度
        prefix_reuse_ratio,      前缀复用比例
        cache_hit_ratio,         缓存命中率
        available_memory,        可用 HBM
        NPU_utilization,         NPU 利用率
        TTFT, TPOT
    ]

两个容易被混为一谈的量，这里刻意分开
------------------------------------
- ``prefix_reuse_ratio`` 是**需求侧**：这批请求自己有多少 prompt token 是共享前缀。
  它是一个工作负载属性，与系统配置无关。
- ``cache_hit_ratio`` 是**供给侧**：当前缓存策略实际命中了多少。
  它是系统状态，同一个负载在不同缓存配置下会不同。

把两者合并成一个"复用率"是这类工作里最常见的口径错误：那样就再也分不清
"缓存没命中"是因为负载本来就没有可复用的前缀，还是因为缓存策略没留住它。

不可测字段的处理
----------------
``available_memory`` 与 ``NPU_utilization`` 依赖昇腾运行时。在无 NPU 的离线路径上
它们取不到值——此时保留 ``None`` 而**不是填 0**：0 会被读成"显存已耗尽"或
"NPU 完全空闲"，两种误读都会让控制器做出错误决策。
:meth:`SystemState.unavailable_fields` 让这种缺失显形。

``cache_hit_ratio`` 同理：**前缀缓存关闭时它也是 ``None``**（调用方不传
``cache_stats``，或缓存一次查询都没有过）。这一点容易写错——取
``CacheStats.hit_rate`` 会得到 0.0，读起来像"缓存开着但一次没命中"。
两者的对策完全相反，所以这里用 ``hit_rate_or_none``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ascend_opt.cache_policy import CacheStats
from ascend_opt.metrics import percentile
from ascend_opt.workload import Request

#: 状态向量的字段顺序，与规范第 2 节逐字对应。向量化时按此顺序。
STATE_VECTOR_FIELDS: tuple[str, ...] = (
    "input_length",
    "estimated_output_length",
    "queue_length",
    "concurrency",
    "prefix_reuse_ratio",
    "cache_hit_ratio",
    "available_memory_mb",
    "npu_utilization_pct",
    "ttft_ms",
    "tpot_ms",
)


@dataclass
class SystemState:
    """某一时刻的系统状态 ``s_t``。

    字段为 ``None`` 表示"该量在当前路径上不可测"，绝不表示 0。
    """

    input_length: float | None = None
    estimated_output_length: float | None = None
    queue_length: int | None = None
    concurrency: int | None = None
    prefix_reuse_ratio: float | None = None
    cache_hit_ratio: float | None = None
    available_memory_mb: float | None = None
    npu_utilization_pct: float | None = None
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    step: int | None = None

    def unavailable_fields(self) -> list[str]:
        """列出无法测得的字段。"""
        return [n for n in STATE_VECTOR_FIELDS if getattr(self, n) is None]

    @property
    def complete(self) -> bool:
        return not self.unavailable_fields()

    def as_vector(self) -> list[float | None]:
        """按 :data:`STATE_VECTOR_FIELDS` 顺序输出向量。

        缺失位置保持 ``None``。不填 0、也不填均值——任何插补都会伪造出一个
        没发生过的系统状态，而控制器恰恰是按这个向量做决策的。
        """
        return [getattr(self, name) for name in STATE_VECTOR_FIELDS]

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"step": self.step}
        for name in STATE_VECTOR_FIELDS:
            value = getattr(self, name)
            out[name] = None if value is None else round(value, 4)
        out["_unavailable_fields"] = self.unavailable_fields()
        return out


def estimate_prefix_reuse_ratio(requests: Sequence[Request]) -> float | None:
    """需求侧前缀复用比例 = 共享前缀 token 数 / prompt token 总数。

    返回 ``None`` 而非 0.0 的场合：没有请求可统计。空负载的复用率是未定义的，
    写成 0 会让"这是低复用负载"和"这是空负载"看起来一样。
    """
    total_prompt = sum(r.prompt_len for r in requests)
    if total_prompt <= 0:
        return None
    reusable = sum(min(r.shared_prefix_len, r.prompt_len) for r in requests)
    return reusable / total_prompt


class WorkloadMonitor:
    """把原始读数归一成 :class:`SystemState` 序列。

    刻意保持"无状态决策"：本类只保留一段有界历史供事后复盘，
    不缓存任何会影响后续观测结果的推导量。
    """

    name = "workload_monitor"

    def __init__(self, history_limit: int = 256) -> None:
        if history_limit < 1:
            raise ValueError("history_limit 必须 >= 1")
        self.history_limit = history_limit
        self._history: list[SystemState] = []

    # ---------------- 观测 ----------------

    def observe(
        self,
        *,
        requests: Sequence[Request] = (),
        step: int | None = None,
        queue_depth: int | None = None,
        running: int | None = None,
        concurrency: int | None = None,
        cache_stats: CacheStats | None = None,
        memory_used_mb: float | None = None,
        memory_total_mb: float | None = None,
        npu_utilization_pct: float | None = None,
        ttft_samples: Iterable[float] = (),
        tpot_samples: Iterable[float] = (),
    ) -> SystemState:
        """采集一个状态点。

        所有硬件侧入参都是可选的：默认值 ``None`` 一路传到
        :class:`SystemState`，最终在 :meth:`SystemState.unavailable_fields`
        里显形。这样"没接硬件"是一个可见的状态，而不是一堆 0。
        """
        ttft_list = list(ttft_samples)
        tpot_list = list(tpot_samples)

        available_memory_mb: float | None = None
        if memory_used_mb is not None and memory_total_mb is not None:
            available_memory_mb = memory_total_mb - memory_used_mb

        cache_hit_ratio: float | None = None
        if cache_stats is not None:
            # 用 hit_rate_or_none 而不是 hit_rate：后者的 0.0 同时覆盖了
            # "缓存关闭"与"缓存开着但一次没命中"两种情形——
            # 前者是"这个信号不存在"，后者才是"缓存策略没留住前缀"。
            # 决策按这个值走时两者绝不能合并。
            cache_hit_ratio = cache_stats.hit_rate_or_none

        state = SystemState(
            input_length=_mean([r.prompt_len for r in requests]) if requests else None,
            estimated_output_length=(
                _mean([r.gen_len for r in requests]) if requests else None
            ),
            queue_length=queue_depth,
            concurrency=concurrency,
            prefix_reuse_ratio=(
                estimate_prefix_reuse_ratio(requests) if requests else None
            ),
            cache_hit_ratio=cache_hit_ratio,
            available_memory_mb=available_memory_mb,
            npu_utilization_pct=npu_utilization_pct,
            ttft_ms=percentile(ttft_list, 90) if ttft_list else None,
            tpot_ms=percentile(tpot_list, 90) if tpot_list else None,
            step=step,
        )
        self._history.append(state)
        if len(self._history) > self.history_limit:
            del self._history[: len(self._history) - self.history_limit]
        return state

    # ---------------- 读取 ----------------

    def latest(self) -> SystemState | None:
        return self._history[-1] if self._history else None

    def history(self) -> list[SystemState]:
        """返回历史副本，防止调用方改写监测记录。"""
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()

    def describe(self) -> dict[str, Any]:
        return {
            "monitor": self.name,
            "history_limit": self.history_limit,
            "n_observed": len(self._history),
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)
