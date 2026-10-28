"""一次基准执行的结果契约。

**为什么单独放一个模块**：这个类型原先定义在 ``benchmark/run_benchmark.py`` 里，
名字叫 ``SimResult``。接真机之后它被**两条**执行路径共用——离线模拟器与真实
后端——于是出现两个问题：

1. ``benchmark/real_backend.py`` 需要引用它，只能反向 import 执行脚本；
   以 ``python -m benchmark.run_benchmark`` 方式启动时，``run_benchmark``
   的真名是 ``__main__``，反向 import 会**再执行一遍整个模块**（得到第二份
   ``SimResult`` 类）。两份同名类并存不会立刻报错，只会让类型检查、
   ``isinstance`` 与后续扩展在某一天悄悄失效。
2. 名字本身在误导：真机路径产出的东西并不"sim"。

放在纯逻辑层之后，两个后端都从同一处取这个契约，且该模块不依赖任何硬件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .metrics import RequestMetrics
from .telemetry import TelemetrySample

__all__ = ["ExecutionResult"]


@dataclass
class ExecutionResult:
    """一次执行（模拟或真机）的全部产出。

    ``samples`` / ``controller_trace`` / ``system_states`` 三者都必须随结果一起
    返回，而不是各自落盘：控制器"按 s_t 决策"这句话只有在 ``s_t`` 也被记录时
    才可核验；只有决策日志而没有依据，读者无法判断决策凭什么做出。
    """

    metrics: list[RequestMetrics]
    samples: list[TelemetrySample]
    #: 测量窗口长度（ms）。**只含正式测量段**，不含装载与预热——
    #: 吞吐类指标全部以它作分母。
    wall_ms: float
    #: 步数。真机路径下 = 逻辑步数（一次设备前向批算一步）。
    steps: int
    #: 未完成的请求数（预算耗尽 / 死锁 / 驱动故障）。
    dropped: int
    controller_trace: list[dict[str, Any]]
    #: 决策步上采集到的 ``s_t``（``WorkloadMonitor`` 的输出）。
    system_states: list[dict[str, Any]] = field(default_factory=list)
    #: 非空表示执行因死锁提前终止（无法推进任何请求），内容即诊断信息。
    #: 与「跑满 max_steps」区分开：前者是配置/策略缺陷，后者是预算不足。
    stall_reason: str = ""
    #: 后端自报的执行条件（驱动身份、装载/预热耗时、子前向次数、空闲等待等）。
    #: 模拟路径为空 dict。**必须落盘**的理由：实测数字的可信度取决于这些条件，
    #: 而它们不在 RequestMetrics 里——不记就只能靠记忆，事后无法核对。
    #: 尤其地，假驱动的 ``hardware_backed=False`` 是"这批数不是实测"的唯一凭据。
    backend_stats: dict[str, Any] = field(default_factory=dict)
    #: **策略臂自己的计数器**（如 `CostPolicyState.describe()`）。
    #:
    #: ⚠️ 必须落盘：`n_degraded`（"有预测但全部超预算 ⇒ 取最省那个"）与
    #: `n_prefill_starved`（"有 prefill 积压、而所有候选本步都推进不了 prefill"）
    #: 是「**动作方向全错**」的**唯一**直接度量。
    #: 实测踩到（2026-09-26）：这两个计数**没有进产物**，于是我事后只能拿 D0 可行率去**猜**
    #: ——并且猜错了：从"D0 只有 5.7% 可行"推出"结构上无法发 prefill"，而遥测
    #: `note` 显示**183/292 步仍在发 `prefill_tok=1024`**（因为全无预测时走的是
    #: `_fallback()` **回退到阈值控制器**，它照常发 prefill）。
    #: ⇒ **"数不出来"不等于"没发生"**；缺计数时只能靠猜，而猜会错。
    arm_state: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "n_requests": len(self.metrics),
            "wall_ms": round(self.wall_ms, 3),
            "steps": self.steps,
            "dropped": self.dropped,
            "stall_reason": self.stall_reason,
            "n_samples": len(self.samples),
            "controller_decisions": len(self.controller_trace),
            "backend_stats": dict(self.backend_stats),
            "arm_state": dict(self.arm_state),
        }
