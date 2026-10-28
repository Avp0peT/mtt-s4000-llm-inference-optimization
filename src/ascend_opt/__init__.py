"""ascend_opt — LLM 推理优化的可复用策略库（研究主体平台 MTT S4000；次平台昇腾）。

包名保留为 ``ascend_opt``：改名会让所有导入路径与历史产物里的引用一起失效，
而规范第 1 节的术语条款明确"标识符不因平台变更而改名"。

分层约定（见仓库根 AGENTS.md）：
    workload / metrics / scheduler / cache_policy / chunk_policy / controller / monitor
        -> 纯逻辑层，禁止调用任何硬件 SDK，必须能在无加速卡环境离线运行
    step_planner
        -> 一步的调度组合逻辑（批预算/分块/缓存/分配）的**唯一**实现，
           由离线模拟器与真实后端共用，避免两条路径各写一份而悄悄分叉
    telemetry
        -> 只做采集与落盘，不主动触达设备（硬件指纹由 scripts/inspect_env.sh 产出）
    provenance
        -> 数据来源标签与可复现性元数据；纯搬运，不做任何推断
    grouping
        -> "什么算同一个实验"的唯一定义；供汇总、绘图、导出校验三处共用，
           避免三处各写一份而悄悄分叉（纯函数，不碰文件系统）

benchmark/ 与 scripts/ 负责把本库接到真实运行时上。

规范第 12 节的模块对应关系：
    WorkloadMonitor          -> monitor.WorkloadMonitor（状态观测，s_t）
    AdaptiveBatchController  -> controller.AdaptiveBatchController（批参数）
    PrefixCachePolicy        -> cache_policy.PrefixCachePolicy
    ChunkController          -> controller.ChunkController（分块大小）
    AdaptiveController       -> controller.AdaptiveController（协调以上三者）
"""

from ascend_opt.cache_policy import (
    CacheStats,
    PrefixCacheConfig,
    PrefixCachePolicy,
)
from ascend_opt.chunk_policy import (
    ChunkConfig,
    ChunkLimits,
    ChunkPolicy,
    DecodePriorityChunkPolicy,
    FixedChunkPolicy,
    build_chunk_policy,
)
from ascend_opt.controller import (
    AdaptiveBatchController,
    AdaptiveController,
    Bound,
    ChunkController,
    ControlAction,
    ControllerConfig,
    ControllerState,
)
from ascend_opt.grouping import (
    GROUP_DIMENSIONS,
    base_run_id,
    group_dimensions,
    group_key,
)
from ascend_opt.execution import ExecutionResult
from ascend_opt.metrics import (
    RequestMetrics,
    RunSummary,
    SLO,
    percentile,
    robust_cv,
    summarize,
)
from ascend_opt.monitor import STATE_VECTOR_FIELDS, SystemState, WorkloadMonitor
from ascend_opt.arm import apply_arm, arm_state
from ascend_opt.cost_model import CostModel, CostModelError, PredictedCost
from ascend_opt.cost_policy import (
    CandidateEvaluation,
    CostDecision,
    CostPolicy,
    CostPolicyConfig,
)
from ascend_opt.slo_weave import (
    SLOWeaveConfig,
    SLOWeaveDecision,
    SLOWeavePolicy,
)
from ascend_opt.provenance import (
    REQUIRED_REPRO_FIELDS,
    REMOTE_EXECUTION_VALUES,
    REMOTE_PLATFORM_PAIRS,
    DataLabel,
    ExecutionEnvironment,
    HardwarePlatform,
    ReproInputs,
    ReproducibilityRecord,
    build_reproducibility_record,
    device_probe_result,
    env_platform,
    normalize_availability,
    parse_mthreads_gmi,
    parse_mthreads_software,
    parse_musa_version_file,
    platform_evidence_conflict,
    resolve_run_label,
    validate_platform_consistency,
    validate_platform_evidence,
)
from ascend_opt.scheduler import (
    BatchPlan,
    ContinuousBatchingScheduler,
    FIFOScheduler,
    SchedulerPolicy,
    StaticBatchScheduler,
    TokenBudget,
    build_scheduler,
)
from ascend_opt.step_planner import StepPlan, plan_step
from ascend_opt.telemetry import TelemetryCollector, TelemetrySample
from ascend_opt.workload import Request, WorkloadSpec, build_requests

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # workload
    "Request",
    "WorkloadSpec",
    "build_requests",
    # metrics
    "RequestMetrics",
    "RunSummary",
    "SLO",
    "percentile",
    "robust_cv",
    "summarize",
    # execution（两条执行路径共用的结果契约）
    "ExecutionResult",
    # scheduler
    "BatchPlan",
    "TokenBudget",
    "SchedulerPolicy",
    "FIFOScheduler",
    "StaticBatchScheduler",
    "ContinuousBatchingScheduler",
    "build_scheduler",
    # step planner（模拟器与真实后端共用的调度组合逻辑）
    "StepPlan",
    "plan_step",
    # cache
    "PrefixCacheConfig",
    "PrefixCachePolicy",
    "CacheStats",
    # chunk
    "ChunkConfig",
    "ChunkPolicy",
    "ChunkLimits",
    "FixedChunkPolicy",
    "DecodePriorityChunkPolicy",
    "build_chunk_policy",
    # controller
    "ControllerConfig",
    "ControllerState",
    "ControlAction",
    "AdaptiveController",
    "AdaptiveBatchController",
    "ChunkController",
    "Bound",
    # monitor
    "WorkloadMonitor",
    "SystemState",
    "STATE_VECTOR_FIELDS",
    # provenance
    "DataLabel",
    "ExecutionEnvironment",
    "HardwarePlatform",
    "ReproducibilityRecord",
    "ReproInputs",
    "REQUIRED_REPRO_FIELDS",
    "REMOTE_EXECUTION_VALUES",
    "REMOTE_PLATFORM_PAIRS",
    "build_reproducibility_record",
    "device_probe_result",
    "env_platform",
    "normalize_availability",
    "parse_mthreads_gmi",
    "parse_mthreads_software",
    "parse_musa_version_file",
    "platform_evidence_conflict",
    "resolve_run_label",
    "validate_platform_consistency",
    "validate_platform_evidence",
    # grouping（汇总/绘图/导出三处共用的分组定义）
    "GROUP_DIMENSIONS",
    "base_run_id",
    "group_dimensions",
    "group_key",
    # slo_weave（第 13 步的最接近方法对照）
    "SLOWeavePolicy",
    "SLOWeaveConfig",
    "SLOWeaveDecision",
    # arm（两条路径共用的臂决策）
    "apply_arm",
    "arm_state",
    # cost_policy（第 12 步的控制器）
    "CostPolicy",
    "CostPolicyConfig",
    "CostDecision",
    "CandidateEvaluation",
    # cost_model（第 12 步的模型层）
    "CostModel",
    "CostModelError",
    "PredictedCost",
    # telemetry
    "TelemetrySample",
    "TelemetryCollector",
]

