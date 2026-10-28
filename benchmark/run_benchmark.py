#!/usr/bin/env python
"""基准执行入口。

职责：把配置、工作负载与策略装起来，跑一次（或多次）基准，把原始产物落到
``results/raw/<run_id>/``。

两种执行路径
------------
``--dry-run``（离线模拟）
    用一阶设备模型驱动完整链路：调度 -> 分块 -> 缓存 -> 控制器 -> 指标。
    目的是**验证 harness 与策略本身**（能否跑通、账目是否自洽、控制器会不会振荡），
    而不是预测真实性能。模拟假设显式写在 ``DeviceModel`` 里，可被实测证伪。

真实后端
    接入 vllm-ascend / torch_npu 的适配层尚未实现（见 ``RealBackend``）。
    这是刻意的：先让 harness 与策略无硬件可验证，再接真机，否则调试时
    分不清「策略错了」还是「环境不对」。

纪律：``--dry-run`` 产出的数字**不得**写入 EXPERIMENTS 结论或 paper。
产物中 ``meta.json`` 的 ``dry_run`` 字段用于机器识别，``collect_metrics.py``
默认会排除这些 run。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

_BENCH_DIR = Path(__file__).resolve().parent
_ROOT = _BENCH_DIR.parent

#: 证据目录：写进这里的产物是"事后唯一能翻的原始记录"。
#: ``--force`` 在这棵子树下一律无效——它只服务于可丢弃的调试产物（见
#: :func:`overwrite_refusal`）。
_RAW_ROOT = _ROOT / "results" / "raw"


for _p in (str(_ROOT / "src"), str(_BENCH_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arm_factory import build_arm  # noqa: E402  —— 与真机路径共用
from ascend_opt import (  # noqa: E402
    AdaptiveController,
    apply_arm,
    arm_state,
    CostModel,
    CostPolicy,
    CostPolicyConfig,
    SLOWeaveConfig,
    SLOWeavePolicy,
    ChunkConfig,
    ChunkLimits,
    ControllerConfig,
    DataLabel,
    ExecutionResult,
    PrefixCacheConfig,
    PrefixCachePolicy,
    ReproInputs,
    Request,
    RequestMetrics,
    SLO,
    TelemetryCollector,
    TelemetrySample,
    WorkloadMonitor,
    WorkloadSpec,
    ExecutionEnvironment,
    HardwarePlatform,
    build_chunk_policy,
    build_reproducibility_record,
    build_scheduler,
    build_requests,
    env_platform,
    percentile,
    plan_step,
    resolve_run_label,
    summarize,
    validate_platform_consistency,
    validate_platform_evidence,
)
from ascend_opt.metrics import (  # noqa: E402
    OUTCOME_SUCCESS,
    OUTCOME_UNFINISHED,
)
from ascend_opt.telemetry import (  # noqa: E402
    RunContext,
    build_run_id,
    hardware_env_candidates,
    load_hardware_env,
)
from ascend_opt.workload import summarize_workload  # noqa: E402
from real_backend import RealBackend as _RealBackendImpl  # noqa: E402
# ★ 闭环补位抖动：**复用**真机路径的同一份实现。
# 另写一份会让"模拟器与真机的同一指标"在到达语义上悄悄分岔 —— 而那正是
# 本项目反复警戒的"两份易碎实现迟早不一致"。
from real_backend import closure_jitter_by_request  # noqa: E402
from workloads import get_workload  # noqa: E402


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DeviceModel:
    """一阶设备模型，**仅用于 --dry-run**。

    假设（写出来是为了能被证伪）：
      - prefill 算力受限：耗时与 token 数近似成正比，批内可加
      - decode 带宽受限：单步耗时 = 固定开销 + 每序列边际成本

    上真机后应先用实测数据校准这四个参数；若拟合残差过大，说明该模型
    不适用于此硬件，此时 dry-run 的结论价值进一步下降——这一点也要记录。
    """

    step_overhead_ms: float = 2.0        # 每步固定开销（launch / kernel 启动）
    prefill_tok_per_ms: float = 6.0      # prefill 吞吐（token/ms）
    decode_base_ms: float = 1.0          # decode 单步固定开销
    decode_per_seq_ms: float = 0.06      # decode 每序列边际开销（带宽受限）
    max_steps: int = 200000

    def step_time_ms(self, n_prefill_tokens: int, n_decode: int) -> float:
        t = self.step_overhead_ms
        t += n_prefill_tokens / max(self.prefill_tok_per_ms, 1e-9)
        if n_decode > 0:
            t += self.decode_base_ms + self.decode_per_seq_ms * n_decode
        return t

    def describe(self) -> dict[str, Any]:
        return {
            "step_overhead_ms": self.step_overhead_ms,
            "prefill_tok_per_ms": self.prefill_tok_per_ms,
            "decode_base_ms": self.decode_base_ms,
            "decode_per_seq_ms": self.decode_per_seq_ms,
            "max_steps": self.max_steps,
            "_note": "一阶模型，仅供 --dry-run 打通链路，不可用于性能结论",
        }


@dataclass
class RunOptions:
    repeats: int = 3
    warmup_s: float = 30.0               # 真机路径用；dry-run 忽略
    #: 计时窗口**之外**的预热轮数（规范第 8/9 节）。
    #: dry-run 下没有预热概念，产物中该项会被标为缺失，而不是填 0。
    warmup_count: int = 3
    control_interval_steps: int = 8      # 控制器每隔多少步做一次决策
    #: SLO 判据的**兜底默认值**。注意这只是"配置没写 run.slo 时"的兜底：
    #: 正常路径下 ``run.slo`` 是 YAML 必填段（``_require_mapping``），会覆盖这里。
    #: 兜底也不得低于实测可达区间——否则程序化构造 RunOptions 时又会静默退回
    #: "达标率恒 0"（2026-09-24 标定：S4000 fp16 逐请求 TPOT ∈ [61.7, 168.0] ms，
    #: 见 scripts/calibrate_slo.py）。
    slo: SLO = field(default_factory=lambda: SLO(ttft_ms=2000.0, tpot_ms=100.0))
    notes: str = ""

    def __post_init__(self) -> None:
        if self.repeats < 1:
            raise ValueError("repeats 必须 >= 1")
        if self.warmup_count < 0:
            raise ValueError("warmup_count 不能为负")

    def describe(self) -> dict[str, Any]:
        return {
            "repeats": self.repeats,
            "warmup_s": self.warmup_s,
            "warmup_count": self.warmup_count,
            "control_interval_steps": self.control_interval_steps,
            "slo": self.slo.describe(),
            "notes": self.notes,
        }


@dataclass
class Config:
    """一份完整实验配置（YAML 解析结果）。"""

    path: Path
    stem: str
    run: RunOptions
    workload_name: str
    workload_overrides: dict[str, Any]
    scheduler_name: str
    scheduler_kwargs: dict[str, Any]
    # max_num_seqs 属于批预算而非某个调度策略的构造参数，
    # 单独提出来是为了让三个调度策略在同一预算下可比。
    max_num_seqs: int
    chunk_policy_name: str
    chunk: ChunkConfig
    cache: PrefixCacheConfig
    controller: ControllerConfig
    device: DeviceModel
    runtime: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {
            "config_path": str(self.path),
            "config_stem": self.stem,
            "run": self.run.describe(),
            "workload": self.workload_name,
            "workload_overrides": self.workload_overrides,
            "scheduler": {
                "policy": self.scheduler_name,
                "max_num_seqs": self.max_num_seqs,
                **self.scheduler_kwargs,
            },
            "chunk": {"policy": self.chunk_policy_name, **self.chunk.describe()},
            "cache": self.cache.describe(),
            "controller": self.controller.describe(),
            "device_model": self.device.describe(),
            # runtime 必须进 describe()：规范第 8 节要求 model / precision /
            # quantization / graph_mode 可溯源，而它们只存在于这一节。
            # 漏掉它的后果不是报错，而是 meta.json 里这几项静默变成
            # NOT_MEASURED——看起来像"没配"，实际是"没记"。
            "runtime": dict(self.runtime),
        }


def _require_mapping(payload: Any, key: str) -> dict[str, Any]:
    section = payload.get(key, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"配置段 {key!r} 必须是映射，收到 {type(section).__name__}")
    return section


def _write_text_lf(path: Path, text: str) -> None:
    """写文本文件并强制 LF 换行。

    为什么不直接 ``Path.write_text``：它在 Windows 上默认 ``newline=None``，
    会把 ``\\n`` 翻成 ``\\r\\n``。``results/raw/**`` 是证据，字节即证据——
    本机产 CRLF、远程 Linux 产 LF，会让跨机器比对/算 md5/核对入库版本时出现
    与行为无关的假 diff。这类差异不报错、内容照常可解析，只有量字节才发现，
    因此在**产生侧**写对，而不是依赖入库时的转换掩盖它。
    """
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _build_dataclass(cls, section: dict[str, Any], section_name: str):
    """按字段名构造 dataclass，未知字段直接报错（配置拼错必须立刻可见）。"""
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(section) - known
    if unknown:
        raise ValueError(
            f"配置段 {section_name!r} 含未知字段 {sorted(unknown)}；可用：{sorted(known)}"
        )
    return cls(**section)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "需要 PyYAML 来读取配置：pip install pyyaml"
        ) from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置根节点必须是映射：{path}")

    run_section = _require_mapping(raw, "run")
    slo_section = _require_mapping(run_section, "slo")
    run = _build_dataclass(RunOptions, {**run_section, "slo": SLO(**slo_section)}, "run")

    workload_section = dict(_require_mapping(raw, "workload"))
    workload_name = workload_section.pop("name", None)
    if not workload_name:
        raise ValueError("配置缺少 workload.name")

    sched_section = dict(_require_mapping(raw, "scheduler"))
    scheduler_name = sched_section.pop("policy", "continuous_batching")
    # max_num_seqs 属于批预算而不是调度策略参数，先取出，
    # 否则会被当成策略构造参数传给 build_scheduler 而报 TypeError。
    max_num_seqs = int(sched_section.pop("max_num_seqs", 32))
    if max_num_seqs < 1:
        raise ValueError("scheduler.max_num_seqs 必须 >= 1")

    chunk_section = dict(_require_mapping(raw, "chunk"))
    chunk_policy_name = chunk_section.pop("policy", "fixed_chunk")
    chunk = _build_dataclass(ChunkConfig, chunk_section, "chunk")

    cache = _build_dataclass(PrefixCacheConfig, _require_mapping(raw, "cache"), "cache")
    controller = _build_dataclass(
        ControllerConfig, _require_mapping(raw, "controller"), "controller"
    )
    device = _build_dataclass(DeviceModel, _require_mapping(raw, "device_model"), "device_model")

    # runtime 段是**自由形式**的运行时开关集合（模型、精度、图模式、KV 类型等），
    # 只被记录进 meta.json 作为溯源信息，不被策略层消费。
    # 这样设计的理由：这些开关的效果发生在算子与图编译层面，无法用策略层模拟；
    # 强行给它们编一个参数，只会制造"看起来测过了"的假象。
    runtime = _require_mapping(raw, "runtime")

    # 执行环境与硬件平台声明（规范 "Remote Execution Environment"）。
    # 在这里校验而不是等到落盘：拼错的取值、以及**两个声明互相矛盾**，都要在
    # run 开始前就报错，否则会白跑一轮基准，然后才在写 meta 时抛异常。
    #
    # 矛盾为什么必须拦：这两个字段都进 meta、都会被下游按平台筛选，
    # 而下游只会读到其中一个。放过去就是永久留下一份自相矛盾的溯源记录。
    _exec_env = ExecutionEnvironment.parse(runtime.get("execution_environment"))
    _hw_platform = HardwarePlatform.parse(runtime.get("hardware_platform"))
    validate_platform_consistency(_exec_env, _hw_platform)

    return Config(
        path=path,
        stem=path.stem,
        run=run,
        workload_name=workload_name,
        workload_overrides=workload_section,
        scheduler_name=scheduler_name,
        scheduler_kwargs=sched_section,
        max_num_seqs=max_num_seqs,
        chunk_policy_name=chunk_policy_name,
        chunk=chunk,
        cache=cache,
        controller=controller,
        device=device,
        runtime=runtime,
        raw=raw,
    )


# --------------------------------------------------------------------------- #
# 配置身份（**写端与读端共用的唯一约定**）
# --------------------------------------------------------------------------- #

#: 产物里配置快照的文件名。写端与读端都从这里取，避免两处写字面量而漂移。
CONFIG_SNAPSHOT_NAME = "config.snapshot.yaml"


def config_data_digest(data: Any) -> str:
    """配置**参数内容**的规范摘要（与排版无关）。

    ★ 为什么不能用"源文件字节"当配置身份（2026-10-03 实测到的缺陷）
    ---------------------------------------------------------------
    写端（`write_run_outputs`）落快照时用的是
    ``yaml.safe_dump(cfg.raw, ...)`` —— 那是一次**重新序列化**：注释、空行、
    引号风格、键序全都被规范化掉。而读端（续跑判据）原先比的是

        snapshot.read_bytes() == cfg_file.read_bytes()      # 原始字节

    ⇒ 本仓每份配置都带大段注释，两者**永远不可能相等** ⇒ 配置身份这一项
    **恒判"配置已变"** ⇒ 续跑**永远不复用**任何既有产物。而报错文案说
    "配置已变（… 与产物里的快照不一致）"，把人指向"去查配置差异"——
    查不出任何差异，因为差异只是注释。

    ⇒ 约定（**写读统一**）：**身份 = 参数内容的规范摘要**。
      * 注释/空行/键序变了 ⇒ 摘要**不变**（它们不影响实验）⇒ 允许复用；
      * 任一参数变了 ⇒ 摘要**变** ⇒ 拒绝复用；
      * 摘要**取不到**（快照缺失/坏掉）⇒ 拒绝复用（"测不出"不得读成"没问题"）。

    规范化的两步都是必要的：`sort_keys=True` 消掉键序，
    `separators` 消掉空白差异；`json.dumps` 保证值被**类型化**地比较
    （`1` 与 `"1"` 不同），这比 YAML 文本比较更严格也更可复现。
    """
    import hashlib

    payload = json.dumps(data, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# 离线模拟后端
# --------------------------------------------------------------------------- #

@dataclass
class _SimRequest:
    """模拟循环中的逐请求状态机。

    三个"可能不发生"的时刻一律用 ``None`` 表示未发生，**不用 0**：
    0 是 run 的第一步这一合法时刻，也是"等了 0ms"。用同一个值表示两件事，
    读的人无法分辨——而这种歧义会导出相反的动作（去查为什么没调度 / 什么都不做）。
    """

    req: Request
    #: 工作负载日程里**原定**到达的时刻。初始化时写一次，之后不得改写。
    scheduled_arrival_ms: float | None = None
    #: 实际到达时刻：开放环 = 日程时刻；闭环 = 客户端腾出容量的时刻。
    arrival_ms: float | None = None
    #: 被服务端受理（进入等待队列）的时刻。
    enqueue_ms: float | None = None
    #: 首次被排进一个 batch 的时刻。
    #: ``enqueue → first_schedule`` 才是真正的排队等待（容量不足）；
    #: ``arrival → enqueue`` 是受理延迟（链路/步粒度）。两者成因不同，必须分开。
    first_schedule_ms: float | None = None
    #: 首个输出 token 产生的绝对时刻。
    first_token_ms: float | None = None
    phase: str = "waiting"        # waiting | prefill | decode | done
    prefill_remaining: int = 0
    decode_remaining: int = 0
    cached_tokens: int = 0
    #: 相对到达时刻的 TTFT。没产出过 token 就是 ``None``。
    ttft_ms: float | None = None
    last_emit_ms: float = 0.0     # 上一次产出 token 的绝对时刻（仅在首个 token 之后有意义）
    done_ms: float | None = None
    intervals: list[float] = field(default_factory=list)


#: **已弃用的旧名**，指向同一个类（不是子类，避免出现两份同名契约）。
#: 保留它只是为了让既有测试与脚本的引用不失效；新代码请直接用 ``ExecutionResult``——
#: 真机路径产出的东西并不"sim"，名字里带 ``Sim`` 会让读者以为它只属于模拟路径。
SimResult = ExecutionResult


def _outcome_for(s: _SimRequest) -> str:
    """把模拟器的终态映射成请求结局（success / failed / rejected / unfinished）。

    模拟器**只可能**产出 ``success`` 与 ``unfinished``：它没有服务端错误注入，
    也没有准入拒绝（不做限流）。这一点必须写在代码里而不是留给读者猜——
    否则汇总表里 ``n_failed`` 恒为 0 会被读成"测过了，没失败"，
    而真实语义是"这条路径根本产生不出失败"。

    ``failed`` / ``rejected`` 由真实后端与单元测试覆盖。现在就把四态口径打通
    是为了避免接真机时再改一遍聚合逻辑——那会让已有产物的列语义在某个时间点
    前后不一致，而历史产物是不能重算的。
    """
    return OUTCOME_SUCCESS if s.phase == "done" else OUTCOME_UNFINISHED


def simulate(
    cfg: Config,
    requests: Sequence[Request],
    *,
    spec: WorkloadSpec | None = None,
    repeat_index: int = 0,
) -> SimResult:
    """事件驱动地跑完整链路。

    时间推进只发生在「一步」的末尾：先按当前队列做决策，再一次性结算这一步的耗时。
    这样调度决策看到的状态与真实系统一致（不能边算边看结果）。

    ``spec`` 影响到达语义，这一点必须说清楚：

    - ``spec=None``：**开放环**。请求按 ``arrival_ms`` 单调放行，到达即入队，
      并发量由调度预算决定。
    - ``spec.arrival == "closed"``：**闭环**。同一时刻的在途请求数不超过
      ``spec.concurrency``，请求只有在有客户端空位时才「产生」。
      这是压测服务系统的标准做法（固定客户端池）。
      由此 ``arrival_ms`` 被理解为「客户端有容量的时刻」，
      因此 TTFT 反映的是服务端排队与计算，不含客户端侧的等待——
      比较不同配置时口径一致，但不要把它当作「用户感知时延」。

    时间戳口径（五个时刻，缺一不可）
    --------------------------------
    每个请求记录 ``scheduled_arrival -> arrival -> enqueue -> first_schedule
    -> first_token -> done``。它们不是冗余：TTFT 是「等待 + 计算」的合成量，
    只报一个总数，看的人无法判断该去加容量还是该去调分块——而这两件事的对策完全相反。
    由此得到恒等式：

        TTFT = admission_ms + queue_ms + prefill_exec_ms
             = (enqueue - arrival) + (first_schedule - enqueue) + (first_token - first_schedule)

    三段成因不同：受理延迟是链路/步粒度，排队是并发容量不足，prefill 执行是算力与分块策略。
    分布与汇总里三段分开报告（``RunSummary.admission/queue/prefill``）。
    """
    # ★ `closed_jittered`（B1 草案 §5 / OPT-03R 裁定 Q4）：与 `closed` **同**闭环语义
    # （在途受 `concurrency` 硬上限），但补位时刻 + 确定性抖动 ⇒ 跨重复有差异。
    closed_loop = spec is not None and spec.arrival in ("closed", "closed_jittered")
    jittered_closure = spec is not None and spec.arrival == "closed_jittered"
    concurrency = spec.concurrency if spec is not None else 0

    scheduler = build_scheduler(cfg.scheduler_name, **cfg.scheduler_kwargs)
    chunk_policy = build_chunk_policy(cfg.chunk_policy_name, config=cfg.chunk)
    cache = PrefixCachePolicy(cfg.cache)
    controller = AdaptiveController(cfg.controller)
    # 成本臂（第 12 步）：与阈值控制器**并列**可选，默认关闭 ⇒ 上面那行的行为不变。
    # ⚠️ 指纹来源见 ControllerConfig.cost_model_fingerprint_source：
    # `observed` = 真机实测指纹（严格比对）；`declared`/离线 = **明说未校验**，
    # 记录进产物。离线那一层**不是**失配防护。
    # 策略臂：构造逻辑与真机路径**共用**（`benchmark/arm_factory.build_arm`）。
    arm = build_arm(cfg)
    # 观测与决策分开（规范第 12 节）：monitor 只汇总 s_t，判据在 controller 里。
    monitor = WorkloadMonitor()

    # 控制器可调的运行期参数（不修改 frozen 配置）。
    # 这三个数是**建议值**：它们能不能兑现，取决于下面把 ChunkLimits 传进策略之后
    # 策略与容量闸门是否接受——因此遥测里必须同时记建议值与生效值。
    cur_chunk = cfg.chunk.chunk_size
    cur_max_tokens = cfg.chunk.max_num_batched_tokens
    max_seqs = cfg.max_num_seqs

    # 共享前缀预热：模拟「服务已运行一段时间，缓存里已有这些前缀」。
    # 不预热会让前缀缓存看起来毫无价值——这是评估这类策略最常见的口径错误。
    if cfg.cache.enabled:
        cache.warmup(requests)
        cache.reset_stats()

    states = {
        r.req_id: _SimRequest(
            req=r,
            scheduled_arrival_ms=r.arrival_ms,
            # arrival_ms 刻意留 None（尚未到达），由准入时填写：
            # 预先填上会让"从未被受理"的请求看起来像正常到达过。
            decode_remaining=max(0, r.gen_len - 1),
        )
        for r in requests
    }
    order = sorted(requests, key=lambda r: (r.arrival_ms, r.req_id))
    not_arrived = list(order)
    # ★ 与真机侧**同一张表、同一套规则**（红线：两条路径的同一指标必须是同一个量）。
    #: 抖动后的**绝对到达时刻**（第一次决定后不再变）。
    pending_arrival: dict[str, float] = {}
    jitter_by_id = closure_jitter_by_request(
        seed=int(getattr(spec, "seed", 0) or 0),
        order=[r.req_id for r in order],
        concurrency=int(getattr(spec, "concurrency", 0) or 0),
        span_ms=float(getattr(spec, "closure_jitter_ms", 0.0) or 0.0),
    ) if jittered_closure else {}
    enqueued: list[_SimRequest] = []

    now = 0.0
    step = 0
    stall_reason = ""
    recent_ttft: list[float] = []
    recent_tpot: list[float] = []
    samples: list[TelemetrySample] = []
    controller_trace: list[dict[str, Any]] = []
    system_states: list[dict[str, Any]] = []

    def all_settled() -> bool:
        return all(s.phase == "done" for s in states.values())

    def _finish_or_decode(s: _SimRequest, at_ms: float) -> None:
        """prefill 完成后的状态迁移：还要 decode 就进 decode，否则直接完成。

        gen_len == 1 的请求在这里被正确收尾——首 token 就是它唯一的输出 token，
        E2E 等于 TTFT，不存在也不可能再有 decode 步。
        """
        if s.decode_remaining <= 0:
            s.phase = "done"
            s.done_ms = at_ms
        else:
            s.phase = "decode"

    def in_flight() -> int:
        """闭环下「在途」的口径：已发出但尚未完成的请求。

        必须包含 ``waiting``（客户端已发出、服务端还没排上）——
        如果只数 prefill/decode，同一时刻新入队的请求不会被计入，
        限流门形同虚设，一整个步长内能把队列一次性填满。
        """
        return sum(1 for s in enqueued if s.phase != "done")

    while not all_settled() and step < cfg.device.max_steps:
        step_start = now

        # 1) 到达与受理——两个时刻必须分开记
        #    闭环：只要客户端池有空位就补一个新请求进来，"客户端腾出容量"即它的到达时刻。
        #    开放环：按 arrival_ms 的日程放行，不看并发；**到达时刻就是日程时刻**。
        #
        #    为什么到达时刻不能写成 now（= 本步开始时刻）：
        #    模拟器以「步」为单位推进，而一步可能很久（chunk 关闭时一次算完整个长 prompt）。
        #    把 arrival 改写成 now，"请求其实早已到达、只是本步才轮到处理"这段等待会被
        #    整段抹掉，突发负载下的 TTFT 被系统性低估。
        #    这个缺陷的隐蔽之处在于账目仍然自洽：done_ms 与 ttft_ms 都以同一个被覆盖的
        #    参考点计量，所以 accounting_error 依旧是 0——只有把 arrival / enqueue 拆开
        #    成两个时刻，才能看见它。
        while not_arrived:
            nxt = not_arrived[0]
            if closed_loop:
                if in_flight() >= concurrency:
                    break
                if jittered_closure:
                    # ★ 闭环补位（Q4）：抖动**查预先算好的表**（按「客户端 × 轮次」）。
                    # 与 `benchmark/real_backend.py` **同一份实现**（红线：模拟器与真机
                    # 的同一指标必须是同一个量 ⇒ 到达语义也必须同构）。
                    # ⚠️ 2026-10-03 改：旧版用 `step` 现算 ⇒ 到达时刻成了调度的函数，
                    #    且请求仍在 `step_start` 入队 ⇒ 实测 16/16 倒置。
                    # ⚠️ **不能**用 `state.scheduled_arrival_ms` 当"算过没算过"的标志：
                    #    它在构造时就被填成了 `r.arrival_ms`（闭环下 = 0），
                    #    于是 `is None` 永远不成立 ⇒ 抖动**一次都不会生效**
                    #    （第一版就是这么写的：倒置没了，但到达也全都变成 0）。
                    #    ⇒ 用一张**只在本地循环里**的待定表来记。
                    jit = jitter_by_id.get(nxt.req_id, 0.0)
                    sched = pending_arrival.get(nxt.req_id)
                    if sched is None:
                        sched = now + jit          # 客户端腾出容量的时刻 + 抖动
                        pending_arrival[nxt.req_id] = sched
                    if now < sched:
                        break        # ★ 时间未到 ⇒ **不放行**（也不 pop）
                    scheduled_arrival = sched
                else:
                    scheduled_arrival = now      # 闭环没有"原定到达"，有容量即产生
            else:
                if nxt.arrival_ms > now:
                    break
                scheduled_arrival = nxt.arrival_ms
            not_arrived.pop(0)
            s = states[nxt.req_id]
            s.scheduled_arrival_ms = scheduled_arrival
            s.arrival_ms = scheduled_arrival
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

        # 2) 调度决策（只看当前状态）
        #    批预算 / 分块额度 / 缓存命中 / token 分配的组合逻辑在
        #    ``ascend_opt.step_planner``，与真实后端**共用同一份实现**。
        #    刻意不在这里再写一遍：两条路径各写一份之后，修了模拟器里的某一处
        #    闸门而真机路径没跟着修，两边的调度行为就会悄悄分叉——
        #    产物里看不出异常，却都在自称"同一套策略"。
        plan = plan_step(
            scheduler=scheduler,
            chunk_policy=chunk_policy,
            cache=cache,
            waiting=[s.req for s in waiting],
            decoding=[s.req for s in decoding],
            # 按入队顺序给出：分配额度时"续传优先"依赖这个顺序。
            prefilling=[(s.req, s.prefill_remaining) for s in prefilling],
            cur_chunk=cur_chunk,
            cur_max_tokens=cur_max_tokens,
            max_num_seqs=max_seqs,
            chunk_enabled=cfg.chunk.enabled,
            step=step,
        )
        by_id = states
        new_starts = [by_id[rid] for rid in plan.new_start_ids]
        decode_targets = [by_id[rid] for rid in plan.decode_ids]
        immediate = [by_id[rid] for rid in plan.immediate_ids]
        alloc = dict(plan.prefill_alloc)
        tokens_prefilled = plan.tokens_prefilled
        prefill_budget = plan.prefill_budget
        effective_chunk_limit = plan.effective_chunk_limit
        # 新起步请求的缓存命中数落到状态机上。注意：**不在这里**记
        # first_schedule_ms——被列入名单不等于真的拿到了算力（额度可能被削到 0），
        # 把这个时刻记在这里会把"等待"算成"计算"。
        for s in new_starts:
            s.cached_tokens = plan.cached_tokens[s.req.req_id]
            s.prefill_remaining = max(0, s.req.prompt_len - s.cached_tokens)
            s.phase = "prefill"

        if plan.stall_reason:
            stall_reason = plan.stall_reason
            break

        step_time = cfg.device.step_time_ms(tokens_prefilled, len(decode_targets))
        step_end = step_start + step_time

        # 6) 结算：prefill 完成的请求进入 decode
        #    first_token_ms 必须单独记：TTFT 是 first_token - arrival 的合成量，
        #    只留 ttft 一个数就无法把它拆回"等了多少"与"算了多少"。
        for s in immediate:
            s.ttft_ms = step_start - s.arrival_ms
            # 命中整段缓存：本步就产出了首 token，因此本步即"被服务"的时刻。
            s.first_schedule_ms = step_start
            s.first_token_ms = step_start
            s.last_emit_ms = step_start
            cache.admit(s.req)
            recent_ttft.append(s.ttft_ms)
            _finish_or_decode(s, step_start)

        for s in prefilling + [s for s in new_starts if s not in immediate]:
            take = alloc.get(s.req.req_id, 0)
            if take <= 0:
                # 没分到 token = 本步没被服务。它的 first_schedule 仍未定，
                # 排队时间继续累计——这正是"容量不足"该有的表现。
                continue
            if s.first_schedule_ms is None:
                # 首次真的拿到算力的时刻。只在第一次写，后续步骤不得覆盖。
                s.first_schedule_ms = step_start
            s.prefill_remaining -= take
            if s.prefill_remaining <= 0:
                s.ttft_ms = step_end - s.arrival_ms
                s.first_token_ms = step_end
                s.last_emit_ms = step_end
                cache.admit(s.req)
                recent_ttft.append(s.ttft_ms)
                _finish_or_decode(s, step_end)

        # 7) 结算：decode 各推进 1 个 token
        for s in decode_targets:
            interval = step_end - s.last_emit_ms
            s.last_emit_ms = step_end
            s.intervals.append(interval)
            s.decode_remaining -= 1
            recent_tpot.append(interval)
            if s.decode_remaining <= 0:
                s.phase = "done"
                s.done_ms = step_end

        # 8) 遥测
        #    建议值（requested）与生效值（effective）必须成对落盘：
        #    只有建议值时，"调大了但没生效"与"调大且生效了"在产物里长得一样。
        if cfg.controller.enabled:
            chunk_requested = controller.state.chunk_size
            tokens_requested = controller.state.max_batched_tokens
        else:
            # 没有控制器时，"建议值"就是静态配置——这样对比列在任何配置下都有意义，
            # 不必用 None 去表示"这一栏不适用"（那会让下游多做一次分支判断）。
            chunk_requested = cfg.chunk.chunk_size
            tokens_requested = cfg.chunk.max_num_batched_tokens
        samples.append(
            TelemetrySample(
                step=step,
                t_ms=round(step_start, 4),
                queue_depth=max(0, len(waiting) - len(new_starts)),
                running=len(decoding) + len(prefilling),
                scheduled_tokens=tokens_prefilled + len(decode_targets),
                ttft_ms=round(percentile(recent_ttft[-8:], 50), 3) if recent_ttft else None,
                tpot_ms=round(percentile(recent_tpot[-8:], 50), 3) if recent_tpot else None,
                chunk_size=effective_chunk_limit,
                chunk_size_requested=chunk_requested,
                max_batched_tokens=cur_max_tokens,
                max_batched_tokens_requested=tokens_requested,
                prefill_budget_effective=prefill_budget,
                note=f"prefill_tok={tokens_prefilled} decode={len(decode_targets)}",
            )
        )

        # 9) 控制器（按固定间隔决策，避免每步都动造成抖动）
        if cfg.controller.enabled and step % max(1, cfg.run.control_interval_steps) == 0:
            # 先观测、再决策：WorkloadMonitor 只汇总 s_t，判据在控制器里。
            # 采样节拍 = 控制节拍——中间步的状态不进入任何决策，采它只会让历史里
            # 堆满没人读过的点，还会制造"一直在观测"的假象。
            system_state = monitor.observe(
                # requests 取当前排队中的请求：input_length / 预估输出长度 /
                # 需求侧前缀复用比例都是"接下来这批活"的属性。
                requests=[s.req for s in waiting],
                step=step,
                queue_depth=len(waiting),
                running=len(decoding),
                concurrency=len(decoding) + len(prefilling),
                # 缓存关闭时**不传** stats：关闭的缓存会让 hit_rate 读出 0.0，
                # 那会被控制器读成"缓存开着但一次没命中"，而事实是"缓存没开"。
                cache_stats=cache.stats if cfg.cache.enabled else None,
                ttft_samples=recent_ttft[-32:],
                tpot_samples=recent_tpot[-32:],
            )
            action = controller.observe(
                step=step,
                ttft_samples=recent_ttft[-32:],
                tpot_samples=recent_tpot[-32:],
                queue_depth=len(waiting),
                running=len(decoding),
                system_state=system_state,
            )
            cur_chunk = controller.state.chunk_size
            cur_max_tokens = controller.state.max_batched_tokens
            # 成本臂（第 12 步）：与阈值臂**并列**。默认 `kind=threshold` ⇒ 上面两行
            # 就是全部行为，既有批次逐字节不变。
            #
            # ⚠️ 调用点在**观测之后**：本步已经跑完，`waiting/decoding/prefilling`
            # 是**步后**状态 ⇒ 这个决定是给**下一步**的（与阈值臂同一套"先决定后生效"）。
            if arm is not None:
                # 运行期上限仍只在 `ChunkLimits` 一处权威。
                _limits = ChunkLimits(
                    chunk_size=controller.state.chunk_size,
                    max_num_batched_tokens=controller.state.max_batched_tokens,
                )
                # ★ 决策逻辑**只有一处实现**（`ascend_opt.arm.apply_arm`）：
                # 模拟器与真机路径共用。实测踩到过"只接进模拟器"⇒ 真机上该臂一次都没被调用、
                # 两臂产物逐项相同却看起来完全正常。
                cur_chunk, cur_max_tokens, _entry = apply_arm(
                    kind=cfg.controller.kind, policy=arm, shadow=cfg.controller.shadow,
                    step=step, now_ms=step_end, cur_chunk=cur_chunk,
                    cur_max_tokens=cur_max_tokens,
                    waiting=[s for s in enqueued if s.phase == "waiting"],
                    decoding=[s for s in enqueued if s.phase == "decode"],
                    prefilling=[s for s in enqueued if s.phase == "prefill"],
                    scheduler=scheduler, chunk_policy=chunk_policy,
                    cache=cache, limits=_limits, max_num_seqs=max_seqs,
                    chunk_enabled=cfg.chunk.enabled,
                )
                if _entry is not None:
                    controller_trace.append(_entry)
            controller_trace.append({"step": step, **action.describe()})
            system_states.append(system_state.describe())

        now = step_end
        step += 1
    else:
        # 循环因 max_steps 退出：有请求未完成，必须显式记录，不能悄悄丢掉
        pass

    dropped = sum(1 for s in states.values() if s.phase != "done")

    # 明细必须覆盖**全部提交的请求**，不能只遍历 enqueued。
    # 从来未被受理的请求（预算耗尽或死锁时留在 not_arrived 里）如果不出现在明细中，
    # 它就会从"未完成率"的分母里消失——而"分母被静默改小"正是本次修复要堵的东西。
    # 同时，这类请求的各时刻保持 0（= 未设置），而不是被填成某个看起来正常的数。
    metrics: list[RequestMetrics] = []
    for s in states.values():
        tpot = sum(s.intervals) / len(s.intervals) if s.intervals else 0.0
        # 该请求自身 token 间隔的 P99：长 prompt 阻塞造成的数秒级停顿会在这里
        # 显形，而不会被 gen_len 除开后在均值里消失。
        tpot_p99 = percentile(s.intervals, 99) if s.intervals else 0.0
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
                outcome=_outcome_for(s),
            )
        )

    return SimResult(
        metrics=metrics,
        samples=samples,
        wall_ms=now,
        steps=step,
        dropped=dropped,
        controller_trace=controller_trace,
        system_states=system_states,
        stall_reason=stall_reason,
        # 策略臂自己的计数器（降级/饥饿/回退）。⚠️ 不落盘就只能靠猜，而猜会错。
        arm_state=arm_state(arm),
    )


# --------------------------------------------------------------------------- #
# 真实后端（占位）
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 真实后端
# --------------------------------------------------------------------------- #

#: 真实后端实现落在 ``benchmark/real_backend.py``（驱动可注入，纯逻辑与设备层分开），
#: 这里只做**转出**，使 ``RealBackend(cfg).run(requests, repeat_index=...)`` 的调用点
#: 与契约保持不变。实现方法见该模块 docstring：
#:   1. 预热按时间进行（>= run.warmup_s），且预热在计时窗口之外；
#:   2. 计时窗口内不做图捕获 / 权重转换；显存分配集中在装载与预热段；
#:   3. 逐请求采样 TTFT 与每 token 间隔（设备同步之后打点），不用总时长除以 token 数；
#:   4. 设备不可用时**报错**，不静默退回 CPU。
RealBackend = _RealBackendImpl


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #

#: run 目录里的标志性产物。任意一个已存在 = 这个 run_id 已被占用。
#: 判据刻意选「产物」而不是「目录存在」：run 目录会被 mkdir(parents=True) 提前建出来，
#: 一次中断留下的**空目录**不构成冲突，否则一个 run_id 会被永久堵死。
RUN_ARTIFACT_NAMES = ("meta.json", "telemetry.jsonl", "requests.jsonl", "config.snapshot.yaml")


class RunDirConflict(RuntimeError):
    """目标 run 目录里已经有产物，拒绝覆盖。

    为什么必须拒绝：``results/raw`` 是**证据**目录，同名产物一旦被覆盖，
    事后就无法区分这一行数字是改动前还是改动后的结果——而这种错误不会报错，
    只会让两条互相矛盾的结论共用同一个 run_id（规范第 2、8 节的溯源要求）。

    ``reason`` 给出**本次**拒绝的具体理由。它刻意做成必填的口径差异：
    ``--force`` 不是万能钥匙，只在极窄的条件下放行（见 :func:`overwrite_refusal`），
    把理由讲清楚，才不会出现"加了 --force 还是失败，不知道差在哪"。
    """

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        existing: Sequence[str],
        reason: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.existing = tuple(existing)
        self.reason = reason
        head = (
            f"run_id 已被占用：{run_id}\n"
            f"  目录：{self.run_dir}\n"
            f"  已存在的产物：{', '.join(self.existing)}\n"
        )
        if reason is None:
            tail = (
                "  results/raw 是证据目录，覆盖后无法区分产物是改动前还是改动后的结果。\n"
                "  换一个 run_id（--run-id <新值>）重跑；只有可丢弃的调试产物才用 --force。"
            )
        else:
            tail = (
                f"  已给 --force，但仍然拒绝覆盖：\n"
                f"    {reason}\n"
                "  已有实测结果只能换新 run_id 重跑，不能被覆盖。"
            )
        super().__init__(head + tail)


def occupying_artifacts(run_dir: Path) -> list[str]:
    """列出该目录里已存在的关键产物名。空列表 = 这个 run 目录可用。

    这是"run_id 是否已被占用"的**唯一判据**，fail-fast 检查与落盘时的权威检查
    共用它——两处用同一函数，避免出现"预检说可以、落盘才发现冲突"的分裂。
    """
    if not run_dir.exists() or not run_dir.is_dir():
        return []
    return [name for name in RUN_ARTIFACT_NAMES if (run_dir / name).exists()]


def load_existing_label(run_dir: Path) -> str | None:
    """读已有产物的 ``data_label``。取不到返回 ``None``（未知），**不是**"可覆盖"。

    刻意不做"没有标签就当没数据"的推断：覆盖判据宁可保守。把一条其实有标签的
    记录当成无标签而覆盖掉，损失不可逆；反过来只是多让人换一个 run_id。
    因此这里的 ``None`` 在 :func:`overwrite_refusal` 里一律按"不可覆盖"处理。
    """
    meta_path = Path(run_dir) / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    label = meta.get("data_label")
    if label is None:
        label = (meta.get("summary") or {}).get("data_label")
    return label if isinstance(label, str) else None


def overwrite_refusal(
    run_dir: Path,
    existing: Sequence[str],
    *,
    dry_run: bool,
    allow_overwrite: bool,
) -> str | None:
    """判定"已占用的目录能不能被覆盖"。返回拒绝理由；``None`` = 允许。

    默认（``allow_overwrite=False``）一律拒绝。``--force`` 也不是万能钥匙，
    它只打开**可丢弃的调试产物**，必须**同时**满足三条——

      1. 目标不在 ``results/raw``（证据目录）子树下；
      2. 本次是 ``dry-run``（真机 run 产出实测数据，实测数据不可覆盖）；
      3. 已有产物的标签是 ``TEST_DATA``（其余一律不许覆盖）。

    任一条不成立即拒绝，理由写进 :class:`RunDirConflict`。

    为什么这么严：覆盖这件事**不会报错**，只会让两条互相矛盾的结论共用一个
    run_id。所以"已有实验换新 run_id"必须是默认且唯一的正路，``--force`` 只留
    给"临时目录里重跑一次 dry-run"这种无损场景。
    """
    if not existing:
        return None

    if not allow_overwrite:
        # 刻意**不**说"因为这是证据目录"：没加 --force 时目标可能是任意临时目录，
        # 把理由写成目录属性会把读的人引到错误的方向去查（"我的临时目录怎么成了证据目录？"）。
        # 这里的真实理由只有一条：没加 --force。
        return (
            "未加 --force，默认拒绝覆盖：已有产物会被原地替换，"
            "覆盖后无法区分产物是改动前还是改动后的结果。"
            "请换一个 --run-id 重跑；只有临时目录里可丢弃的调试产物才用 --force——"
            "它也只在「非证据目录 + 本次 dry-run + 已有产物为 TEST_DATA」三条同时满足时放行。"
        )

    run_dir = Path(run_dir)
    try:
        under_raw = run_dir.resolve().is_relative_to(_RAW_ROOT.resolve())
    except (OSError, ValueError):
        under_raw = True  # 解析失败时按"危险"处理：宁可拒绝
    if under_raw:
        return f"目标位于证据目录 {_RAW_ROOT} 之下，证据目录永远不可覆盖。"

    if not dry_run:
        return "本次不是 dry-run，会产出实测数据；实测数据不可覆盖。"

    label = load_existing_label(run_dir)
    if label != DataLabel.TEST_DATA.value:
        shown = label if label is not None else "取不到（meta.json 缺失或损坏）"
        return (
            f"已有产物的标签是 {shown}，只有 {DataLabel.TEST_DATA.value} "
            f"可被 --force 覆盖。"
        )
    return None


def write_run_outputs(
    run_dir: Path,
    cfg: Config,
    result: SimResult,
    *,
    run_id: str,
    repeat_index: int,
    dry_run: bool,
    workload_fingerprint: dict[str, Any],
    summary: dict[str, Any],
    controller_effect: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
    overwrite: bool = False,
    hardware_env: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """写出一次 run 的全部原始产物。``results/raw`` 一旦写入即视为只读。

    本函数同时负责**给这次 run 盖章**：判定数据标签（规范第 10 节）并装配
    可复现性元数据（第 8 节）。把盖章放在落盘这一步，是因为它必须发生在
    产物生成的那一刻——事后补标签就可能补错，而标签错了整批数据都作废。

    ``hardware_env`` 由调用方在 run 开始前读好并传入（见 ``run_one``）：
    判定用的指纹与写进 meta 的指纹必须是**同一份**。为 ``None`` 时本函数
    自己按候选路径读一份——单测与脚本直接调用时可以省掉这一步。

    覆盖保护：目录里已有任一关键产物时抛 ``RunDirConflict``，除非
    ``overwrite=True`` **且** :func:`overwrite_refusal` 放行（它还会再查三条）。
    这条检查必须在这里（而不是只在调用方）——本函数是产物写入的唯一出口，
    放在别处就会出现"另有一条路径绕过去"的漏洞。
    """
    existing = occupying_artifacts(run_dir)
    reason = overwrite_refusal(
        run_dir, existing, dry_run=dry_run, allow_overwrite=overwrite
    )
    if reason is not None:
        # 未加 --force 时用默认文案；加了 --force 仍被拒时把**具体**理由带出来。
        raise RunDirConflict(
            run_dir, run_id, existing, reason=reason if overwrite else None
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    context = RunContext(
        run_id=run_id,
        workload=cfg.workload_name,
        config_path=str(cfg.path),
        repeat_index=repeat_index,
        dry_run=dry_run,
        extra={
            "dropped_requests": result.dropped,
            "steps": result.steps,
            "controller_decisions": len(result.controller_trace),
            "stall_reason": result.stall_reason,
            "workload_fingerprint": workload_fingerprint,
            # 后端自报的执行条件。模拟路径没有可报的（空 dict → None，不写 {}，
            # 因为 {} 看起来像"报过了但什么都没报"）。
            "backend_stats": result.backend_stats or None,
            **(extra_meta or {}),
        },
    )
    collector = TelemetryCollector(run_dir, context, hardware_env=hardware_env)
    for sample in result.samples:
        collector.record(sample)

    # ---- 数据标签（规范第 10 节 + Remote Execution Environment）----
    # 看三件事：数据是怎么产生的（dry_run）、推理跑在哪台机器上（执行环境声明）、
    # 那台机器的硬件在不在。
    # hardware 为 None 表示指纹文件缺失 —— 未知不等于可用，见 provenance 模块。
    hardware_env = collector.read_hardware_env()
    exec_env = ExecutionEnvironment.parse(cfg.runtime.get("execution_environment"))
    hw_platform = HardwarePlatform.parse(cfg.runtime.get("hardware_platform"))
    # 后端只在这里判定一次，同时喂给标签判定与元数据。分开判会漂移：
    # 标签按一个值判、元数据记另一个值，而两者本该永远一致。
    backend = "simulate" if dry_run else "real"
    label = resolve_run_label(
        dry_run=dry_run,
        hardware_available=hardware_env.get("available"),
        backend=backend,
        execution_environment=exec_env,
    )
    summary["data_label"] = label.value
    extra = summary.setdefault("extra", {})
    extra["data_label"] = label.value
    if not label.eligible_for_conclusion:
        # 把"为什么不能进结论"直接写在产物里，避免下游凭印象猜测。
        extra["label_reason"] = _label_reason(
            label, hardware_env, exec_env, hw_platform
        )

    # ---- 可复现性元数据（规范第 8 节）----
    #
    # 模型版本走"解析后的 commit"而不是配置里的引用：配置写的是 `main`，
    # 那是一个**引用**，今天指向 A、明天可能指向 B，拿它复现不了任何东西。
    # 后端装载成功后会把真实的权重 commit 报在 driver describe 里；拿不到就
    # 传空串（record 侧会退化成配置里的引用，并把 `model_revision_resolved`
    # 标成 False）——**不允许**把引用冒充成版本号。
    driver_desc = (result.backend_stats or {}).get("driver") or {}
    resolved_rev = driver_desc.get("model_revision_resolved")
    repro = build_reproducibility_record(
        ReproInputs(
            run_id=run_id,
            config=cfg.describe(),
            workload_fingerprint=workload_fingerprint,
            hardware_env=hardware_env,
            repo_root=_ROOT,
            # dry-run 没有预热轮，记录成缺失而不是 0——0 会被读成"确实不预热"。
            warmup_count=None if dry_run else cfg.run.warmup_count,
            measurement_count=cfg.run.repeats,
            tensor_parallel_size=cfg.runtime.get("tensor_parallel_size"),
            model_revision=(
                resolved_rev if isinstance(resolved_rev, str) and resolved_rev else ""
            ),
            model_revision_resolved=(
                True if isinstance(resolved_rev, str) and resolved_rev else None
            ),
            execution_environment=exec_env,
            hardware_platform=hw_platform,
            backend=backend,
        ),
        label=label,
    )

    # 注意 summary 与 extra 分开传：summary 落在 meta["summary"]，
    # 让读取端有一个稳定路径；config 与 controller_trace 落在 meta 顶层。
    paths = collector.write(
        summary=summary,
        extra={
            "config": cfg.describe(),
            "controller_trace": result.controller_trace,
            # s_t 序列：没有它，"控制器按 s_t 决策"就只是一句无从核验的话。
            "system_states": result.system_states,
            "controller_effect": controller_effect if controller_effect is not None
            else _controller_effect(result),
            "reproducibility": repro.describe(),
            "data_label": label.value,
        },
    )

    # 逐请求明细：collect_metrics.py 的唯一输入
    #
    # newline="\n" 是必须的，不是风格问题：Windows 上 open(..., "w") 默认
    # newline=None，会把 "\n" 翻成 "\r\n"。后果是**同一份 harness 产物因平台不同
    # 而字节不同**——本机产 CRLF、远程 Linux 产 LF。而 results/raw/** 是证据，
    # 字节即证据：跨机器比对、算 md5、与入库版本核对时会造出与行为无关的假 diff，
    # 且这类差异不会报错、JSON 照样能解析，只有量字节才看得出来。
    # 仓库规范（.gitattributes: * text=auto eol=lf）要求一律 LF，这里在产生侧
    # 就写对，而不是依赖入库时的转换掩盖它。
    requests_path = run_dir / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8", newline="\n") as fh:
        for m in result.metrics:
            fh.write(json.dumps(m.describe(), ensure_ascii=False) + "\n")

    # 配置快照：让事后复盘不依赖 configs/ 目录当时的状态
    # ⚠️ 这是一次**重新序列化**（注释/键序会丢）⇒ 读端**不得**拿源文件字节来比。
    #    读端比的是 `config_data_digest(...)`（见该函数的 docstring）。
    snapshot_path = run_dir / CONFIG_SNAPSHOT_NAME
    try:
        import yaml

        _write_text_lf(
            snapshot_path,
            yaml.safe_dump(cfg.raw, allow_unicode=True, sort_keys=False),
        )
    except ImportError:  # pragma: no cover
        _write_text_lf(snapshot_path, json.dumps(cfg.raw, ensure_ascii=False, indent=2))

    paths.update({"requests": requests_path, "config_snapshot": snapshot_path})
    return paths


def _controller_effect(result: SimResult) -> dict[str, Any]:
    """把"建议值 vs 实际生效值"压成几个可核对的数字。

    为什么不能只看 ``controller_trace``：轨迹里记的是**建议值**——
    控制器说自己把 chunk 调到 8192，trace 里就是 8192。真正生效的额度在遥测的
    ``prefill_budget_effective`` 里。两者系统性不一致，就是"日志显示调大、
    实际额度未变"那个静默失效；只有把两边都算出来才看得见。

    三个计数各自的含义（不要合并）：
      - ``n_steps_chunk_request_not_honoured``：建议的 chunk 上限被夹小了。
        成因固定且可解释（chunk 上限超过单步 token 预算，永远兑现不了）。
      - ``n_steps_prefill_below_chunk_request``：本步实际发放的 prefill 额度
        低于建议上限。**这不是 bug**：decode 保留、容量闸门、以及"剩余量不足
        一个 chunk"都会合理压低它。它的用途是给出"额度上限是否真的被吃满"的量级，
        而不是判定失败。
      - ``max_*``：整段 run 里出现过的极值，用于确认控制器是否真的动过
        （若 max == min == 配置初值，说明控制器一次都没生效）。
    """
    samples = result.samples
    decisions = result.controller_trace

    def _ints(name: str) -> list[int]:
        return [v for s in samples if (v := getattr(s, name)) is not None]

    requested_chunks = _ints("chunk_size_requested")
    effective_chunks = _ints("chunk_size")
    effective_prefill = _ints("prefill_budget_effective")
    requested_tokens = _ints("max_batched_tokens_requested")
    effective_tokens = _ints("max_batched_tokens")

    chunk_not_honoured = sum(
        1
        for s in samples
        if s.chunk_size_requested is not None
        and s.chunk_size is not None
        and s.chunk_size_requested > s.chunk_size
    )
    prefill_below_request = sum(
        1
        for s in samples
        if s.chunk_size_requested is not None
        and s.prefill_budget_effective is not None
        and s.prefill_budget_effective < s.chunk_size_requested
    )

    # 相邻两次决策之间的推理步数（第一次没有前序，记 None 并跳过）。
    decision_intervals = [
        v
        for d in decisions
        if (v := d.get("steps_since_last_decision")) is not None
    ]

    # 建议值本身是否变过：这是"控制器是否真的在动"的最直接证据，
    # 比数 n_updates 可靠——n_updates>0 但建议值一直等于初值，说明动的是别的维度。
    return {
        "n_steps": len(samples),
        "n_control_decisions": len(decisions),
        "n_decisions_changed": sum(1 for d in decisions if d.get("changed")),
        "n_decisions_recover": sum(
            1 for d in decisions if d.get("bottleneck") == "recover"
        ),
        "n_decisions_over_memory": sum(
            1 for d in decisions if d.get("bottleneck") == "memory"
        ),
        "n_steps_chunk_request_not_honoured": chunk_not_honoured,
        "n_steps_prefill_below_chunk_request": prefill_below_request,
        "chunk_requested": _range(requested_chunks),
        "chunk_effective": _range(effective_chunks),
        "max_batched_tokens_requested": _range(requested_tokens),
        "max_batched_tokens_effective": _range(effective_tokens),
        "prefill_budget_effective": _range(effective_prefill),
        # 决策的**实际节拍**（相邻两次决策之间隔了多少推理步）。
        # 它应当等于 run.control_interval_steps；不等就说明调用点与配置脱节。
        # 必须落盘的原因：冷却按观测周期计数，而"一个周期对应多少推理步"这件事
        # 只在这里才看得见——节拍比配置更密时，冷却会以错误的时间尺度工作，
        # 而它不会报错，只会表现为"参数调了没反应"。
        "decision_interval_steps": _range(decision_intervals),
        # ★ 策略臂自己的计数器（`n_degraded` / `n_prefill_starved` / `n_fallback` / `n_idle`）。
        # ⚠️ 这三件事**不在采样里**：它们记的是"选不出来时怎么办"，而不是"选了什么"。
        # 不落盘 ⇒ 事后只能拿 D0 可行率去猜"动作方向"，而实测证明那样会猜错
        # （见 `ExecutionResult.arm_state` 的注释）。没有该状态的臂返回 `{}`，不虚构。
        "arm_state": dict(getattr(result, "arm_state", None) or {}),
        "n_cycles": (decisions[-1].get("state", {}).get("n_cycles") if decisions else None),
        "n_skipped_cooldown": (
            decisions[-1].get("state", {}).get("n_skipped_cooldown") if decisions else None
        ),
        "_note": (
            "chunk_requested / max_batched_tokens_requested 是控制器的建议值，"
            "其余是实际生效值；两者的差额即『建议未兑现』的量。"
            "prefill_budget_effective 低于 chunk 上限是正常的（decode 保留与容量闸门），"
            "不构成缺陷判定。"
            "decision_interval_steps 用于核对决策节拍是否等于 control_interval_steps；"
            "n_skipped_cooldown 与 n_cycles 用于核对冷却是否**能**触发"
            "（恒为 0 说明冷却参数是死的）。"
        ),
    }


def _range(values: Sequence[int]) -> dict[str, int] | None:
    """给一串数记录 min/max/first/last。空列表返回 ``None``，不返回全 0 的假记录。"""
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "first": values[0],
        "last": values[-1],
    }


def _assert_platform_evidence(
    hardware_env: dict[str, Any], cfg: Config
) -> None:
    """run 开始前核对「配置声明的平台」与「指纹检出的平台」，矛盾即抛错。

    与 :func:`ascend_opt.provenance.validate_platform_consistency` 的分工：
    那个只保证**两个声明之间**不打架（说 MTT 就别又说在本机跑），自洽但可以
    与真机毫无关系。只有把"说的"与"查到的"对照，才能**证伪**一个声明。

    缺证据时放行：指纹没写平台（旧格式）或没找到指纹时，检出的平台是
    ``unknown``，此时推不出"说错了"——把"没证据"当成"证据矛盾"会让所有
    未采集指纹的机器一并跑不动。
    """
    declared = HardwarePlatform.parse(cfg.runtime.get("hardware_platform"))
    validate_platform_evidence(
        declared,
        env_platform(hardware_env),
        source=str(hardware_env.get("source") or "未找到环境指纹"),
    )


def _label_reason(
    label: DataLabel,
    hardware_env: dict[str, Any],
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.UNKNOWN,
    hardware_platform: HardwarePlatform = HardwarePlatform.UNKNOWN,
) -> str:
    """给非 MEASURED 的 run 生成一句人话解释。

    解释里点明**平台**而不是一律写"昇腾"：这句话的用途就是让人知道下一步该去哪台
    机器重跑。平台写错会把人送到错误的机器上，而它只在"结果不可用、必须重跑"时
    出现——恰恰是最需要它准确的时候。未声明平台时不硬编码任何平台名。
    """
    declared = hardware_platform is not HardwarePlatform.UNKNOWN
    platform_name = hardware_platform.display_name if declared else "远程硬件"
    target = f"{hardware_platform.display_name} 远程主机" if declared else "远程推理主机"

    if label is DataLabel.TEST_DATA:
        return "本次为 --dry-run 模拟路径：数字由一阶设备模型合成，仅供链路验证，不得进入结论。"
    if label is DataLabel.HARDWARE_REQUIRED:
        if execution_environment.is_remote is False:
            # 与"远程主机上没卡"是两回事，必须分开说，否则修不出正确动作。
            # 规范性那句刻意**不带平台名**："不得表述为实测"这条规则讲的是
            # 本机与远程之分，与平台无关；平台名出现在"该去哪台机器重跑"里才有用。
            return (
                "本次 run 声明为在本地执行（runtime.execution_environment=local）。"
                "规范规定本机不是推理环境，其结果不得表述为远程硬件实测；"
                f"需在{target}上重跑后方可用于结论。"
            )
        return (
            "执行推理的主机上未检测到可用的设备"
            f"（平台：{platform_name}；设备探测工具不可用），"
            "该 run 的结果为 HARDWARE_REQUIRED，需在真机重跑后方可用于结论。"
        )
    if label is DataLabel.NOT_MEASURED:
        reason = hardware_env.get("reason")
        return f"硬件可用性未知，无法判定为实测数据。{reason or ''}".strip()
    return ""


def run_once(
    cfg: Config,
    *,
    dry_run: bool,
    out_dir: Path,
    repeat_index: int = 0,
    run_id: str | None = None,
    extra_meta: dict[str, Any] | None = None,
    verbose: bool = True,
    overwrite: bool = False,
) -> Path:
    """跑一次（一个 repeat），返回该 run 的目录。

    ``overwrite=False``（默认）时，若目标 run 目录已有产物则**在做任何计算之前**
    就报错——白白跑完一次基准再发现写不进去，是纯粹的浪费；真正的写入保护
    在 ``write_run_outputs`` 里，这里只是同一判据的提前触发。

    ``overwrite=True`` 也只是**候选**放行：具体能不能覆盖由
    :func:`overwrite_refusal` 决定（不在证据目录、是 dry-run、已有产物是
    ``TEST_DATA``，三条都要满足）。
    """
    # spec 必须单独保留并传给 simulate：闭环到达语义依赖 spec.arrival 与
    # spec.concurrency，只传 requests 会丢掉这两个信息而退化成开放环。
    spec = get_workload(cfg.workload_name, **cfg.workload_overrides)
    requests = build_requests(spec)

    run_id = run_id or build_run_id(cfg.workload_name, cfg.stem)
    dir_name = run_id if repeat_index == 0 else f"{run_id}-r{repeat_index}"
    run_dir = out_dir / dir_name

    existing = occupying_artifacts(run_dir)
    reason = overwrite_refusal(
        run_dir, existing, dry_run=dry_run, allow_overwrite=overwrite
    )
    if reason is not None:
        raise RunDirConflict(
            run_dir, dir_name, existing, reason=reason if overwrite else None
        )

    # ---- 运行前的平台证据检查（**在任何计算之前**）----
    # 为什么必须在这里：一次基准动辄数分钟。等到落盘才发现"配置声明 MTT、
    # 指纹明确是 Ascend"是纯粹的浪费，而且那份矛盾产物已经进了证据目录。
    #
    # 指纹一次性读好、并且**带着往下传**（write_run_outputs -> TelemetryCollector），
    # 让"运行前的判定"与"落盘的元数据"看到同一份快照。分成两次读就会出现
    # "判定时读到 A、落盘时读到 B"的产物——字段齐全，只是来自两个不同的文件，
    # 没有任何地方会报错。
    hardware_env = load_hardware_env(hardware_env_candidates(run_dir))
    _assert_platform_evidence(hardware_env, cfg)

    if dry_run:
        result = simulate(cfg, requests, spec=spec, repeat_index=repeat_index)
    else:
        # spec 必须传进去：闭环下所有 arrival_ms 都是 0，
        # 从 requests 反推不出"这是闭环"，会把它误当开放环（t=0 一次性涌入）。
        result = RealBackend(cfg).run(requests, repeat_index=repeat_index, spec=spec)
    # wall_ms **一律来自后端自己**，不用外面包一圈的计时。
    # 真机路径的装载与预热（可达数十秒）发生在后端内部的时间原点之前；
    # 用外面的计时会把它们算进吞吐的分母里，让吞吐系统性偏低——
    # 而"偏低"看起来完全正常，没有任何字段会暴露它。
    wall_ms = result.wall_ms

    # 非硬件驱动不得产出"实测"。CLI 路径不可能拿到假驱动（白名单里没有它），
    # 所以这条只在有人把假驱动接到正式路径上时触发——那时必须立刻失败，
    # 而不是留下一批 MEASURED 的假数字。
    if result.backend_stats.get("driver", {}).get("hardware_backed") is False:
        raise RuntimeError(
            "本次 run 的驱动声明 hardware_backed=False（假驱动），"
            "其结果不得作为实测。请检查 RealBackend 的驱动注入来源。"
        )

    summary_obj = summarize(
        result.metrics,
        run_id=dir_name,
        workload=cfg.workload_name,
        config=cfg.stem,
        wall_ms=wall_ms,
        slo=cfg.run.slo,
        extra={
            "steps": result.steps,
            "dropped_requests": result.dropped,
            "stall_reason": result.stall_reason,
            "dry_run": dry_run,
        },
    )
    summary = summary_obj.describe()
    # 账目自查：差值异常说明计时口径有问题，而不是「数据不好看」。
    # 只对"完成且到达时刻已知"的请求自查——其余请求返回 None（无从自查）。
    # 不能把 None 当 0 收进分布：0 会被读成"账目完全自洽"，恰好把问题掩盖掉。
    errors = [
        e for m in result.metrics if (e := m.accounting_error_ms()) is not None
    ]
    if errors:
        summary["accounting_error_p99_ms"] = round(percentile(errors, 99), 4)

    effect = _controller_effect(result)

    paths = write_run_outputs(
        run_dir,
        cfg,
        result,
        run_id=dir_name,
        repeat_index=repeat_index,
        dry_run=dry_run,
        # 传 spec 才会带上 concurrency / seed / arrival —— 这三个量无法从请求
        # 列表反推（闭环下 arrival_ms 全为 0），而它们是规范第 8 节的必需字段。
        workload_fingerprint=summarize_workload(requests, spec),
        summary=summary,
        controller_effect=effect,
        extra_meta=extra_meta,
        overwrite=overwrite,
        # 与运行前判定**同一份**指纹快照，见 _assert_platform_evidence。
        hardware_env=hardware_env,
    )

    if verbose:
        _print_summary(dir_name, summary, paths, dry_run, effect)
    return run_dir


def _fmt_pcts(block: dict[str, Any], digits: int) -> str:
    """把分位块格式化成一串 P50/P90/P95/P99，并带上样本量 n。

    空块必须显式写成"无数据"而不是抛 KeyError：一次"全部请求都没跑完"的 run
    恰恰最需要被看见，而异常会让它的摘要整行消失（看起来像"这个配置还没跑"）。
    """
    if not block:
        return "无数据（没有成功完成的请求）"
    fmt = f"{{:.{digits}f}}"
    pcts = " / ".join(fmt.format(block[k]) for k in ("p50", "p90", "p95", "p99"))
    return f"{pcts} ms（n={block.get('n', '?')}）"


def _fmt_range(block: dict[str, Any] | None) -> str:
    """把 ``_range`` 的结果打成 ``min..max(首/末)``。缺失写成"无数据"而不是 0。"""
    if not block:
        return "无数据"
    if block["min"] == block["max"]:
        return str(block["max"])
    return f"{block['min']}..{block['max']}"


def _print_summary(
    run_id: str,
    summary: dict[str, Any],
    paths: dict[str, Path],
    dry_run: bool,
    controller_effect: dict[str, Any] | None = None,
) -> None:
    tag = "[DRY-RUN]" if dry_run else "[REAL]"
    label = summary.get("data_label", "UNKNOWN")
    print(f"\n{tag} run_id = {run_id}   数据标签 = {label}")
    print(f"  TTFT  P50/P90/P95/P99 = {_fmt_pcts(summary.get('ttft') or {}, 1)}")
    print(f"  TPOT  P50/P90/P95/P99 = {_fmt_pcts(summary.get('tpot') or {}, 2)}")
    if summary.get("stall"):
        print(
            f"  停顿  P50/P99 = {summary['stall']['p50']:.1f} / "
            f"{summary['stall']['p99']:.1f} ms（单请求 token 间隔 P99 的分布）"
        )
    # TTFT 的三段分解必须与总数并列显示：只看总数无法判断该加容量还是该调分块。
    # 括号里的提醒是必要的——三段各自取 P50，其和不严格等于 TTFT 的 P50
    # （分位数不可加）。不加这句，读者会以为算错了。
    adm = (summary.get("admission") or {}).get("p50")
    if adm is not None:
        que = (summary.get("queue") or {}).get("p50", 0.0)
        pre = (summary.get("prefill") or {}).get("p50", 0.0)
        print(
            f"  TTFT 拆分 P50：受理 {adm:.1f} + 排队 {que:.1f} + prefill {pre:.1f} ms"
            f"（各段独立取分位，故与上面的 TTFT P50 不严格相等）"
        )
    print(
        f"  吞吐 = {summary['throughput_tok_s']:.1f} tok/s   "
        f"缓存命中 = {summary['cache_hit_rate']:.3f}"
    )
    # 结局账目与"干得多快"分开显示：失败/拒绝/未完成不是"不好的吞吐"，
    # 而是"服务没把活干完"，成因与对策都不同。
    print(
        f"  结局  成功={summary.get('n_success', 0)} 失败={summary.get('n_failed', 0)} "
        f"拒绝={summary.get('n_rejected', 0)} 未完成={summary.get('n_unfinished', 0)}"
        f"（共提交 {summary.get('n_requests', 0)}）"
    )
    satisfaction = summary.get("slo_satisfaction_rate")
    if satisfaction is None:
        # 没配 SLO 时三个量都不可计算。写 0 会被读成"一个都没达标"。
        print("  SLO 达标率 = 未配置（不可计算，故不写 0）")
    else:
        goodput_tok = summary.get("slo_goodput_tok_s")
        print(
            f"  SLO 达标率 = {satisfaction:.3f}（达标 / 全部提交）  "
            f"达标吞吐 = {goodput_tok:.1f} tok/s  "
            f"达标请求数 = {summary.get('n_slo_satisfied', 0)}"
        )
    stability = summary.get("stability") or {}
    if stability:
        print(
            f"  稳健CV  ttft={stability.get('ttft_robust_cv', 0.0):.4f}  "
            f"tpot={stability.get('tpot_robust_cv', 0.0):.4f}"
        )
    # 控制器效果必须打印"建议值 vs 生效值"两条范围，而不是只报 n_updates：
    # 只看决策次数无法分辨"调了并生效"与"调了但被压回原值"。
    if controller_effect and controller_effect.get("n_control_decisions"):
        eff = controller_effect
        print(
            f"  控制器 决策={eff['n_control_decisions']} 生效改动={eff['n_decisions_changed']}"
            f" 恢复={eff.get('n_decisions_recover', 0)} 显存收缩={eff.get('n_decisions_over_memory', 0)}"
        )
        print(
            f"    chunk 建议={_fmt_range(eff.get('chunk_requested'))} "
            f"生效={_fmt_range(eff.get('chunk_effective'))}  "
            f"prefill 额度={_fmt_range(eff.get('prefill_budget_effective'))}"
        )
        if eff.get("n_steps_chunk_request_not_honoured"):
            print(
                f"    ！有 {eff['n_steps_chunk_request_not_honoured']} 步的 chunk 建议值未被兑现"
                f"（建议上限超过单步 token 预算时必然如此）"
            )
    if "accounting_error_p99_ms" in summary:
        print(f"  账目自查误差 P99 = {summary['accounting_error_p99_ms']:.3f} ms")
    unfinished = summary.get("n_unfinished", 0)
    if unfinished:
        print(
            f"  ！有 {unfinished} 个请求未完成——时延分布已剔除它们，"
            f"尾延迟被低估，本次结果不可用于结论"
        )
        reason = summary.get("extra", {}).get("stall_reason") or ""
        if reason:
            print(f"    原因：{reason}")
    print(f"  -> {paths['meta'].parent}")
    if not DataLabel(label).eligible_for_conclusion:
        note = summary.get("extra", {}).get("label_reason", "")
        print(f"  注意：数据标签 {label}，不得写入结论。{note}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM 推理优化基准执行入口（研究主体平台 MTT S4000）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, help="配置 YAML 路径，如 configs/baseline/default.yaml")
    p.add_argument("--dry-run", action="store_true", help="离线模拟，不触达硬件")
    p.add_argument("--repeats", type=int, default=None, help="覆盖配置中的 repeats（稳定性评估需 >=3）")
    p.add_argument("--out", default=None, help="原始产物输出目录，默认 results/raw")
    p.add_argument("--run-id", default=None, help="显式指定 run_id（默认按时间戳生成）")
    p.add_argument("--notes", default=None, help="附加到 meta 的备注，建议写实验编号")
    p.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖**可丢弃的调试产物**。不是万能钥匙：只在同时满足"
             "（1）目标不在 results/raw 下、（2）本次是 --dry-run、"
             "（3）已有产物标签为 TEST_DATA 时才放行。已有实测结果一律换新 run-id 重跑。",
    )
    p.add_argument("--quiet", action="store_true", help="不打印摘要")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.repeats is not None:
        if args.repeats < 1:
            raise SystemExit("--repeats 必须 >= 1")
        cfg.run.repeats = args.repeats
    if args.notes is not None:
        cfg.run.notes = args.notes

    if cfg.run.repeats < 3 and not args.dry_run:
        # 不禁止，但必须显式提示：单次/两次运行的差异无法与噪声区分。
        print(
            f"警告：repeats={cfg.run.repeats} < 3，结果不能作为结论。"
            f"（AGENTS.md 附录 A4）"
        )

    out_dir = Path(args.out) if args.out else _ROOT / "results" / "raw"
    run_id = args.run_id or build_run_id(cfg.workload_name, cfg.stem)

    # 落盘前先把**全部重复**的目标目录查一遍。
    # 逐个重复地边跑边查会留下半批产物（r0 写了、r1 冲突退出），
    # 而半批重复最容易在下游被当成"完整的 1 次重复"用掉。
    #
    # --force 也要过这一关：它只放行"可覆盖的调试产物"，不是无条件放行。
    # 预检与落盘共用 overwrite_refusal，避免"预检说可以、落盘才拒绝"的分裂。
    for i in range(cfg.run.repeats):
        name = run_id if i == 0 else f"{run_id}-r{i}"
        existing = occupying_artifacts(out_dir / name)
        reason = overwrite_refusal(
            out_dir / name, existing,
            dry_run=args.dry_run, allow_overwrite=args.force,
        )
        if reason is not None:
            print(
                str(RunDirConflict(
                    out_dir / name, name, existing,
                    reason=reason if args.force else None,
                )),
                file=sys.stderr,
            )
            return 4

    dirs: list[Path] = []
    for i in range(cfg.run.repeats):
        dirs.append(
            run_once(
                cfg,
                dry_run=args.dry_run,
                out_dir=out_dir,
                repeat_index=i,
                run_id=run_id,
                extra_meta={"notes": cfg.run.notes} if cfg.run.notes else None,
                verbose=not args.quiet,
                overwrite=args.force,
            )
        )

    print(f"\n完成 {len(dirs)} 次运行，产物目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
