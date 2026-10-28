"""运行期自适应控制器。

输入：最近的观测（TTFT / TPOT / 队列深度 / 运行中批大小），
      以及 :class:`~ascend_opt.monitor.SystemState` 汇总出的状态向量 ``s_t``
      （规范第 2 节）中的**非时延**分量——缓存命中率、可用显存、NPU 利用率。
输出：对 chunk 与批预算的调整量

设计原则（这些原则本身就是本研究要验证的假设，实现时刻意写清楚）：
1. **双目标取舍**：TTFT 与 TPOT 是竞争关系——加大 prefill 份额降 TTFT、抬 TPOT。
   控制器必须显式选择「当前谁是瓶颈」，而不是同时优化两者。
2. **滞回 + 冷却**：没有这两个机制，控制器会在阈值附近来回抖动，
   制造出比不控制更差的尾延迟。
3. **有界**：所有输出都必须夹在配置区间内；控制器不允许把系统推到未验证的工况。
4. **可解释**：每次调整都要给出 reason，写进 telemetry，事后能复盘决策序列。
5. **纯逻辑**：只做加减与夹取，不调用任何硬件或运行时接口。
6. **双向可逆**：调小是防御，调大是恢复或为 prefill 加码。**两者都必须存在。**
   只有收缩没有恢复的控制器是一根棘轮：压过一次就再也回不去，
   负载回落后系统被永久按在过保守的配置上——而那正是"自适应"最该赢下来的场景。
   恢复的边界是配置初值，不是上界：恢复的语义是撤销防御，不是在安全区里继续探索
   （后者只会引入抖动，见第 2 条）。
7. **信号要么可信、要么显式为 None**：未测的信号（离线路径上没有显存读数、
   缓存可能是关的）**不参与决策**，但必须出现在决策记录里。
   "没测"与"测出来没压力"是两件事，合并之后控制器会给出一个无法复盘的决策——
   而本研究的核心主张恰恰是"控制器按 ``s_t`` 做决策"。

被这些原则排除掉的两种实现（都曾经真实存在过，记在这里避免回退）：
- 策略层用 frozen 配置值当硬上限，控制器调大的动作被静默吃掉
  （"日志显示调大、实际额度未变"，见 :class:`~ascend_opt.chunk_policy.ChunkLimits`）；
- token 预算只在 TPOT 超标时收缩、没有恢复路径（单向棘轮）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ascend_opt.metrics import percentile
from ascend_opt.monitor import SystemState


def _round_opt(value: float | None, ndigits: int = 3) -> float | None:
    """**取不到就保持 ``None``，不填 0**。

    0 是"测出来是 0"，缺失是"没测"——语义相反。本模块里最容易踩的是
    ``window_age_ms`` 与 ``decode_p90_ms``：它们在没有样本时是"未知"，
    写成 0 会被读成"窗口刚刷新过""纯 decode 用时为 0"，两个都恰好是最乐观的读法。
    """
    return None if value is None else round(value, ndigits)


@dataclass(frozen=True)
class ControllerConfig:
    enabled: bool = True
    target_tpot_ms: float = 50.0        # TPOT 目标（SLO）
    target_ttft_ms: float = 2000.0      # TTFT 目标（SLO）
    hysteresis: float = 0.10            # 超出目标多少比例才触发调整
    #: **控制器种类**（第 12 步新增）。两条**并列**的臂，只换"怎么挑"、不换"挑什么"
    #: （两者共用 `plan_step` 与 `ChunkLimits`）：
    #:
    #: * ``threshold``（**默认**）：阈值控制器，判据是实测信号越过目标+滞回；
    #: * ``cost``：成本模型控制器，判据是**预测的每步成本**是否超每步预算。
    #:
    #: * ``slo_weave``：第 13 步的**最接近方法对照**（论文 arXiv:2609.07883 的 Algorithm 1）。
    #:
    #: ⚠️ 默认 ``threshold`` 是为了让**既有批次的行为逐字节不变**（可比性优先）。
    kind: str = "threshold"
    #: **shadow 模式**：只**记录**策略的建议，**不生效**。
    #:
    #: 验收要求"shadow 仅证明预测/决策可记录，**闭环才验证动作实际生效**" ⇒
    #: 这两件事必须在产物里分得开：shadow 的轨迹证明"能算出建议"，
    #: 但它**不能**用来支持任何"策略有效"的主张。
    shadow: bool = False
    #: **逐步 host 侧开销**（ms，按 batch）：控制器的预算判据必须把它算进去。
    #: 模型只预测**前向**，而 deadline 约束的是**整步**（实测 b8：前向 104.8 + host 25.3 = 130.5 ms）。
    #: ⚠️ 本机 + 当前执行路径的量（`scripts/decompose_step_overhead.py` 可重测），不是芯片性质。
    host_overhead_ms: dict[int, float] | None = None
    #: 上表没列到的 batch 用这条线性拟合 `(斜率, 截距)`；实测三点 ⇒ (3.705, -4.35)。
    host_overhead_fit: list[float] | None = None
    #: 是否在成本预测上乘**安全余量**（模型文件的 `safety_margin`）。默认 ``True``。
    #: 第 15 步消融"去掉误差裕量"由它表达 —— 不要靠改模型文件（那会造出同一份标定的两个真相）。
    apply_safety_margin: bool = True
    #: **显式候选分块列表**（成本臂/SLOWeave 臂的评估点）。``None`` ⇒ 旧行为：
    #: 只取 `{chunk_size_min, chunk_size_max}` **两个**点。
    #:
    #: 为什么需要它：`{min, max}` 这个取法意味着**改 `chunk_size_min` 不会自动新增中间点**
    #: —— 把下限改成 128 只会得到 `{128, 2048}`，**不会**产生 192/256。
    #: 要做"小 chunk 能不能被选中并执行"的验证，必须**显式**给出评估点。
    #:
    #: ⚠️ **分工别搞混**：`chunk_size_min/max` 是**可动范围**（运行期上限的唯一权威是
    #: `ChunkLimits`，由它派生）；本字段只是**在范围内挑哪几个点去评估**。
    #: ⇒ 想让 128 真的可动，必须**同时**把 `chunk_size_min` 降到 128，
    #: 否则候选会被 `ChunkLimits` 夹回下限，白写。
    #: ⚠️ 越界（不在 `[chunk_size_min, chunk_size_max]` 内）一律**报错**，不静默夹取。
    candidate_chunks: list[int] | None = None
    #: SLOWeave 的 `C_max`（论文 §4.3 取 4096）。仅 `kind == "slo_weave"` 时用。
    slo_weave_c_max: int = 4096
    #: SLOWeave 的 TPOT 目标 `D`。``None`` ⇒ 用 ``target_tpot_ms``（不另立一个数）。
    slo_weave_tpot_target_ms: float | None = None
    #: ★ **prefill 步的预算**（ms，仅 ``kind == "cost"``）。``None``（默认）= **不启用双预算**
    #: ⇒ 全部候选都按 ``target_tpot_ms``（decode/逐 token 护栏）判 ⇒ **既有批次逐字节不变**。
    #:
    #: 为什么必须能在**配置**里表达（2026-10-03 补）：双预算此前只在代码与单测里可达 ——
    #: `arm_factory` 构造 `CostPolicyConfig` 时**从不传** `prefill_deadline_ms`，
    #: 而 `ControllerConfig` 也没有这个字段 ⇒ **任何批次都无法开启它**，
    #: 产物里也就永远看不到"这个臂到底开没开双预算"。
    #: （这正是本仓记过的"机制写了、没接线"的同族：功能在、实验达不到。）
    #:
    #: ⚠️ **语义**（详见 `cost_policy._budget_for`）：本预算**只在纯 prefill 步**适用；
    #: 只要本步服务了 decode token、或让某个 decode 请求没拿到 token，
    #: 就必须回到 `step_deadline_ms`（逐 token 护栏）。
    #: ⚠️ 取值应与 `run.slo.ttft_ms` **同源**（服务的是同一个承诺），
    #: 不一致时要在预注册里写明理由。
    prefill_deadline_ms: float | None = None
    #: 成本模型文件路径（``kind == "cost"`` 时用）。``None`` ⇒ 用下面的默认路径。
    cost_model_path: str | None = None
    #: 成本模型的**指纹来源**（``kind == "cost"`` 时用）：
    #:
    #: * ``observed``：取自**运行环境的实测指纹**（真机路径该用这个）；
    #: * ``declared``（**默认**）：取自**配置里的平台声明**。
    #:
    #: ⚠️ ``declared`` 只适用于**离线（模拟器）**路径：那里的"观测"就是配置本身，
    #: 于是这一层校验退化成"配置与标定一致"的检查，**不能**当作真机上的失配防护。
    #: 这一点必须显式写出来，否则离线跑通的成本臂会给人"指纹校验也过了"的错觉。
    cost_model_fingerprint_source: str = "declared"
    #: **控制信号口径**（第 04 步新增）。
    #:
    #: * ``legacy``（**默认**）：只看 ``tpot_p90_ms``——它来自"相邻两次 token 发出
    #:   之间的间隔"，而某一步做了 prefill 时，落在这两步之间的间隔**直接包含该 prefill
    #:   的耗时**。于是"prefill 挤占"被读成"decode 变慢"，而控制器对此的响应是调小
    #:   ``chunk_size``（prefill 更碎 ⇒ 含 prefill 的步更多）——**反向放大**。
    #: * ``layered``：额外收集/记录"这段等待里有没有 prefill""纯 decode 花了多少"
    #:   （见 :class:`~ascend_opt.controller.SignalView`），供第 09 步做干预实验。
    #:
    #: ⚠️ **两种模式在决策上等价**——本步只做**记录**，不改判据。
    #: "纯 decode 特征有效"不等于"应把所有决策都交给它"，那是第 09 步要检验的策略问题。
    #: 默认 ``legacy`` 是为了让既有批次与判定的语义**逐格不变**（可比性优先）；
    #: 改这个值必须显式写配置，且会随 meta 落盘。
    signal_mode: str = "legacy"
    #: **判据用哪一套口径**（第 09 步新增）。
    #:
    #: * ``tpot_p90``（**默认**）：现状——来自"相邻两次 token 之间的间隔"，
    #:   而含 prefill 的步会把 prefill 耗时算进去。实测差值 **5.5 倍**
    #:   （`EXPERIMENTS.md` §3.4 **E10**：报告 337 ms vs 纯 decode 61 ms），
    #:   而控制器据此收缩、又把纯 decode 从 61 ms 推到 118 ms。
    #: * ``decode_p90``：只用**纯 decode 步**（`signal_mode: layered` 采集的
    #:   `decode_p90_ms`），即"用户感知的稳定态延迟"。
    #:
    #: ⚠️ 默认值保证**既有批次逐字节不变**：判据没变时，`reason` 与 `signals`
    #: 都与加这个字段之前完全一致（新增键只在非默认口径下才并入）。
    #: ⚠️ 选 ``decode_p90`` 必须同时开 ``signal_mode: layered``（否则没有那组样本，
    #: 会在加载配置时报错——不静默退回旧口径，那正是本步要消除的东西）。
    signal_criterion: str = "tpot_p90"
    #: 观测窗口的**过期上限**（ms）。
    #:
    #: 默认 ``None`` = **不过期**（保持既有行为）。设了值之后，若距上一次**收到新样本**
    #: 已超过该时长，则本周期视为**无可信观测** ⇒ 直接返回不做动作（保守），
    #: 并在决策轨迹里写明原因。
    #:
    #: 为什么需要它：此前没有任何过期规则 ⇒ 负载停顿/空闲时，窗口里残留的**旧**样本
    #: 仍会继续驱动收缩或放大，而产物里看不出那些样本已经过时。
    signal_window_expiry_ms: float | None = None
    #: **门限形式**（OPT-02 / 路线 B2 新增）。
    #:
    #: * ``absolute``（**默认**）：门限 = `target_tpot_ms·(1±h)`，单位是 **ms 绝对值**。
    #:   既有批次全部用这个，默认值保证行为**逐字节不变**（可比性优先）。
    #: * ``normalized``：门限 = `workload_scale_ms·(1±h)`，其中 `workload_scale_ms`
    #:   是**同一会话内在线跟踪**的工作点（见 `workload_scale_alpha`）。
    #:
    #: **为什么需要它**（实测，不是推断）：滞回带宽度是固定绝对值 `2h·target` ≈ 19.2 ms，
    #: 而工作点的量级随负载档位变化一个数量级以上。只要某个负载档位的工作点落在带宽内，
    #: 该档行为就会随重复之间的小扰动**翻转**（`EXPERIMENTS.md` §4.3.4：同一份
    #: `adaptive_c4.yaml` 三次重复 TTFT = 481.4 / 481.6 / **652.1**，+43%）。
    #: 门限贴着工作点时控制器**不报错、不停止，只是随机地动**。
    #:
    #: OPT-02 用 S14B2 的 `controller_trace` 现测复核了这个前提，并**替换**了 §4.3.4
    #: 原来那个取自 `tpot_p90` 的读数（那个量含 prefill 污染，见 §3.4 E10）：
    #: 控制器**实际读到的** `criterion_ms`（= `decode_p90`）在同一次 run 内为
    #: **63.0 → 147.1 ms**；`adaptive` 臂有 **11/446** 次决策落在带 `[110.4, 129.6]` 内，
    #: 且在同一工作点上 `headroom` 的观测散布（≈0.094）**大于门槛本身**（0.08）
    #: ⇒ 决策确实不可复现。
    #:
    #: ⚠️ **选了 `normalized` 就不再以 SLO 为控制目标**：门限跟着工作点走，控制器
    #: 变成"检测并响应**变化**"的器件（升载收缩、卸载恢复），而不是"把 TPOT 压到
    #: 某条绝对线以下"。这是**有意的机制分离**——绝对预算交给 `kind: cost` 臂
    #: （预测式），本臂保持**无模型、反应式**。要让本臂重新带上绝对约束，
    #: 用 `slo_hard_guard`（默认关）。
    #:
    #: ⚠️ 该值必须进产物：否则事后分不清某批用的是哪种门限形式，
    #: 而两种形式的读数**不可比**（与 `signal_criterion` 同理）。
    gate_mode: str = "absolute"
    #: `gate_mode: normalized` 时工作点跟踪的 **EWMA 速率** α ∈ (0, 1]。
    #:
    #: 越小 ⇒ 参考值越"迟钝" ⇒ 对负载变化的响应越灵敏（参考值还没跟上，
    #: 判据已越带）；越大 ⇒ 越快把新工作点当"新常态"⇒ 只对**突变**响应。
    #: 默认 0.1（约 10 个观测周期收敛）。
    #: ⚠️ 该速率**决定控制器的响应时标**，必须与 `control_interval_steps` 一起读：
    #: 1 个观测周期 = `control_interval_steps` 个推理步。
    workload_scale_alpha: float = 0.1
    #: `gate_mode: normalized` 时工作点参考的**初值**（ms）。
    #:
    #: ``None``（默认）⇒ 由**第一个可测观测**播种。这有一个已知且**被登记**的后果：
    #: 若 run 一开始就在过载区，参考值会直接播种在过载工作点上 ⇒ 控制器认为
    #: "这就是常态" ⇒ **不动作**。要避免，就显式给一个本会话测得的**空载/低载**参考
    #: （即"用会话内比值"那条纪律的落地方式）。
    workload_scale_seed_ms: float | None = None
    #: `gate_mode: normalized` 时是否**额外**保留绝对 SLO 上限作为放大动作的护栏。
    #:
    #: 默认 ``False``：不加，即纯粹的相对门限（见 `gate_mode` 的机制分离说明）。
    #: 打开后，放大动作还要求 `criterion <= target_tpot_ms·(1+h)`——于是
    #: "相对门限 + 绝对 SLO 护栏"同时成立。它是**可选**而不是默认，
    #: 因为把两条约束悄悄叠在一起，会让"这条臂到底在追什么"变得不可分辨。
    slo_hard_guard: bool = False
    #: 两次调整之间至少隔多少个**观测周期**。
    #:
    #: 一个观测周期 = 一次 ``observe()`` 调用，与推理步数**无关**：决策节拍由
    #: ``run.control_interval_steps`` 决定，控制器不该再重复知道这个数——
    #: 两个地方各自持有一个"步"的概念，迟早会打架。
    #:
    #: 这个字段此前叫 ``cooldown_steps`` 且按**推理步**比较，结果就是打架了：
    #: 默认决策节拍是 8 个推理步、冷却写 3 个推理步，于是两次决策之间天然已隔 ≥8 步，
    #: `step - last_change < 3` **永远为假**——冷却从不触发，`n_skipped_cooldown` 恒为 0。
    #: 它是死参数，而配置注释还写着"冷却步数是最重要的抗振荡参数，扫参时优先调它"，
    #: 于是扫参只会得到一条平平的直线，并被误读为"这个参数不敏感"。
    cooldown_cycles: int = 3
    # chunk_size 的可行区间与步长
    chunk_size_init: int = 4096
    chunk_size_min: int = 512
    chunk_size_max: int = 16384
    chunk_size_step: int = 512
    # 单步 token 预算的可行区间与步长
    max_batched_tokens_init: int = 8192
    max_batched_tokens_min: int = 2048
    max_batched_tokens_max: int = 32768
    max_batched_tokens_step: int = 1024
    # 并发序列数（batch_size）的可行区间。
    # 三个字段同为 0 表示**不控制该维度**——这是默认值，刻意如此：
    # 运行期改并发上限会改变准入语义（进而改变"丢弃了哪些请求"），
    # 属于比 chunk/token 预算更强的一档干预，必须由实验显式开启。
    batch_size_init: int = 0
    batch_size_min: int = 0
    batch_size_max: int = 0
    batch_size_step: int = 1
    # ---- 非时延反馈信号的阈值 ----
    # 三个字段默认都是 ``None`` = **该信号不参与决策**。这是刻意的默认值：
    # 离线路径上没有显存读数、缓存也可能是关的。给一个"看起来合理"的默认阈值，
    # 等于让控制器按一个不存在的信号做决策——那比不做决策更糟，
    # 而且不会有人发现（决策日志里看不出这个信号是测出来的还是编出来的）。
    # 要启用必须在配置里显式写。
    #: 缓存命中率 ≥ 此值时，拒绝为 TTFT 放大分块。
    #: 依据：大分块的价值是缩短**未命中**部分的 prefill 计算；缓存已经省下算力时，
    #: 这个杠杆是错的（收益有限，却要付出 TPOT）。
    cache_hit_rate_high: float | None = None
    #: 可用显存 < 此值时，否决一切放大动作（chunk / token 预算 / 并发）。
    min_available_memory_mb: float | None = None
    #: 可用显存 < 此值时，**主动收缩**——显存临界不是"慢了"，是"快跑不动了"，
    #: 优先级高于一切时延判据。
    critical_available_memory_mb: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("threshold", "cost", "slo_weave"):
            # 拼错的值必须当场暴露 —— 静默退回 threshold 会让"我跑的是成本臂"
            # 变成一个错误的自我认知，而产物里看不出区别。
            raise ValueError(
                f"controller.kind 只能是 threshold / cost / slo_weave，收到 {self.kind!r}")
        if self.slo_weave_c_max <= 0:
            raise ValueError("controller.slo_weave_c_max 必须为正")
        if self.kind != "threshold" and not self.enabled:
            # ⚠️ 这两者组合起来是个**静默陷阱**：`kind: cost` 说"我在用成本臂"，
            # 而 `enabled: False` 让 run 路径**整块跳过控制器**（成本臂的调用点就在里面）
            # ⇒ 实际什么都没发生，产物里只有一条空轨迹。
            # 实测踩到过（用静态格的配置试成本臂，轨迹 0 条）。⇒ 加载期就拦住。
            raise ValueError(
                f"controller.kind == {self.kind!r} 时必须 controller.enabled == True"
                "（否则 run 路径会整块跳过控制器，该臂不会被执行，轨迹为空）")
        if self.cost_model_fingerprint_source not in ("declared", "observed"):
            raise ValueError("controller.cost_model_fingerprint_source 只能是 declared / observed")
        if self.chunk_size_min > self.chunk_size_max:
            raise ValueError("chunk_size_min 不能大于 chunk_size_max")
        if self.max_batched_tokens_min > self.max_batched_tokens_max:
            raise ValueError("max_batched_tokens_min 不能大于 max_batched_tokens_max")
        if not 0.0 <= self.hysteresis < 1.0:
            raise ValueError("hysteresis 必须在 [0, 1) 区间")
        if self.cooldown_cycles < 0:
            # 单位为观测周期；负值没有意义。（旧的 cooldown_steps 是推理步，
            # 改名后旧配置会在加载时因"未知字段"直接报错——这是刻意的：
            # 静默沿用旧单位正是这个缺陷的成因。）
            raise ValueError("cooldown_cycles 不能为负（单位是观测周期，不是推理步）")
        if self.batch_size_max and self.batch_size_min > self.batch_size_max:
            raise ValueError("batch_size_min 不能大于 batch_size_max")
        if self.cache_hit_rate_high is not None and not 0.0 < self.cache_hit_rate_high <= 1.0:
            raise ValueError("cache_hit_rate_high 必须在 (0, 1] 区间")
        if (
            self.min_available_memory_mb is not None
            and self.critical_available_memory_mb is not None
            and self.critical_available_memory_mb > self.min_available_memory_mb
        ):
            # 收缩阈值必须比否决阈值更严格。反过来会造出一个自相矛盾的区间：
            # 显存已经低于收缩线，却又"高于"否决线，于是既不放大也不收缩。
            raise ValueError(
                "critical_available_memory_mb 必须 <= min_available_memory_mb；"
                "否则会出现「否决放大、也不触发收缩」的空转区间"
            )
        # 信号口径拼错必须**立刻报错**（第 04 步）：
        # 静默降级成 legacy 会让"我开了分层信号"与"其实没开"长得一样——
        # 而它的下游后果是"这批用了哪种信号"无从分辨（那正是本步要堵的）。
        if self.signal_mode not in ("legacy", "layered"):
            raise ValueError(
                f"controller.signal_mode 取值非法：{self.signal_mode!r}；"
                "可用：'legacy'（默认，只看 tpot_p90）/ 'layered'（额外记录分层信号）"
            )
        if self.signal_window_expiry_ms is not None and self.signal_window_expiry_ms <= 0:
            raise ValueError(
                "signal_window_expiry_ms 若要启用必须为正（单位 ms）；"
                "不启用请留空（None = 不过期）"
            )
        # 判据口径（第 09 步）：拼错必须报错；且 decode_p90 需要分层样本。
        if self.signal_criterion not in ("tpot_p90", "decode_p90"):
            raise ValueError(
                f"controller.signal_criterion 取值非法：{self.signal_criterion!r}；"
                "可用：'tpot_p90'（默认）/ 'decode_p90'"
            )
        if self.signal_criterion == "decode_p90" and self.signal_mode != "layered":
            # ⚠️ 不能静默退回旧口径——那正是这一步要消除的东西。
            # 而且静默退回在生产里几乎看不出来：判据"看起来在工作"，只是用的不是你要的那套。
            raise ValueError(
                "signal_criterion='decode_p90' 需要 signal_mode='layered'（纯 decode 样本来自它）。"
                "请在配置里同时写 controller.signal_mode: layered。"
            )
        # 门限形式（OPT-02 / B2）：拼错必须报错。静默退回 absolute 会让
        # "这批用的是归一化门限"与"其实没用"长得一样，而两种形式的读数不可比。
        if self.gate_mode not in ("absolute", "normalized"):
            raise ValueError(
                f"controller.gate_mode 取值非法：{self.gate_mode!r}；"
                "可用：'absolute'（默认，门限 = target_tpot_ms·(1±h)）/ "
                "'normalized'（门限 = 在线跟踪的工作点·(1±h)）"
            )
        if not 0.0 < self.workload_scale_alpha <= 1.0:
            raise ValueError(
                "controller.workload_scale_alpha 必须在 (0, 1] 区间"
                f"（收到 {self.workload_scale_alpha!r}）"
            )
        if self.workload_scale_seed_ms is not None and self.workload_scale_seed_ms <= 0:
            raise ValueError(
                "controller.workload_scale_seed_ms 若给出必须为正（单位 ms）；"
                "不给请留空（None = 由第一个可测观测播种）"
            )
        if self.gate_mode == "normalized" and self.signal_mode != "layered":
            # 归一化门限跟踪的是**判据**本身。legacy 下没有纯 decode 样本、
            # `criterion_ms` 也不会写进轨迹 ⇒ 事后**无法核验**门限参考是怎么来的。
            # 这不是保守起见，是"产物必须自证"：不拦就会得到一批
            # 看起来用了归一化门限、但参考值无从追溯的运行。
            raise ValueError(
                "gate_mode='normalized' 需要 signal_mode='layered'"
                "（跟踪的是判据本身；legacy 下轨迹里没有 criterion_ms，参考值不可核验）。"
                "请在配置里同时写 controller.signal_mode: layered。"
            )

    @property
    def batch_size_controlled(self) -> bool:
        """是否在运行期调整并发序列数上限。"""
        return self.batch_size_max > 0

    @property
    def signal_thresholds_configured(self) -> bool:
        """是否至少启用了一个非时延信号（供产物质检用）。"""
        return any(
            v is not None
            for v in (
                self.cache_hit_rate_high,
                self.min_available_memory_mb,
                self.critical_available_memory_mb,
            )
        )

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            # ⚠️ **哪条臂**必须进产物。少了它，"这次跑的是 threshold 还是 slo_weave"
            # 在产物里无从判断 —— 而三臂对照的全部前提就是"看得出跑的是哪条"。
            # （实测踩到：加了 kind/shadow 却漏进 describe()，两臂的产物看起来一模一样。）
            "kind": self.kind,
            # shadow 也要记：验收要求"shadow 只证明可记录、闭环才验证生效"，
            # 而**产物得能分辨二者**，否则 shadow 的轨迹会被当成闭环结果读。
            "shadow": self.shadow,
            # ⚠️ **读数口径必须进产物**。实测踩到：第 14 步的配置漏了
            # `signal_criterion` ⇒ 加载成默认 `tpot_p90`，而 `describe()` 里又没有这两个字段
            # ⇒ **产物完全看不出"判据读的是哪一列"**（只有 `signal_mode=layered` 的采集痕迹）。
            # 于是"跑的不是冻结方案"这件事在产物里**不可见**。
            "signal_mode": self.signal_mode,
            "signal_criterion": self.signal_criterion,
            "cost_model_path": self.cost_model_path,
            "cost_model_fingerprint_source": self.cost_model_fingerprint_source,
            # ⚠️ **判据口径的组成部分必须进产物**，与 `signal_criterion` 同理：
            # 第 14 步的整条主线是"判据从'前向 ≤ 预算'改成'整步 ≤ 预算'"
            # （前向 104.8 + host 25.3 = 整步 130.5 @b8）—— 若不记这三项，
            # 产物就**看不出**某次运行到底有没有把 host 项算进判据、余量有没有乘。
            # 实测踩到：`S14B1`（无 host 项）与 `S14D`（有 host 项）的产物在这一点上
            # **完全一样**，只能靠仓库配置 + `git_commit` 反推（可追溯，但**不是产物自证**）。
            # ⚠️ `host_overhead_ms` 是**本机 + 当前执行路径**的量，不是芯片性质（换路径要重测）。
            "host_overhead_ms": self.host_overhead_ms,
            "host_overhead_fit": self.host_overhead_fit,
            "apply_safety_margin": self.apply_safety_margin,
            "slo_weave_c_max": self.slo_weave_c_max,
            "slo_weave_tpot_target_ms": self.slo_weave_tpot_target_ms,
            "target_tpot_ms": self.target_tpot_ms,
            "target_ttft_ms": self.target_ttft_ms,
            "hysteresis": self.hysteresis,
            # ⚠️ 门限形式必须进产物（OPT-02 / B2）：两种形式的读数**不可比**，
            # 而不记它就会让"这批用的是归一化门限"在产物里不可见
            # （与 `signal_criterion` 漏记那次同型）。
            "gate_mode": self.gate_mode,
            "workload_scale_alpha": self.workload_scale_alpha,
            "workload_scale_seed_ms": self.workload_scale_seed_ms,
            "slo_hard_guard": self.slo_hard_guard,
            # 记的是**观测周期**数，不是推理步数。单位必须写进产物：
            # 否则事后看轨迹里的间隔只会看到"每隔 8 步"，无从判断冷却是否生效。
            "cooldown_cycles": self.cooldown_cycles,
            "chunk_size": [self.chunk_size_min, self.chunk_size_max, self.chunk_size_step],
            # 显式候选列表必须进产物：否则事后无法判断"这次评估的是哪几个块大小"，
            # 而"改了候选集"正是 D1 两次对照的**唯一变量**（漏记 ⇒ 变量不可核）。
            "candidate_chunks": (list(self.candidate_chunks)
                                 if self.candidate_chunks else None),
            # ★ 双预算开关必须落盘：否则"这个臂开没开双预算"事后无从核对，
            #   而它恰恰决定了两条 cost 臂是不是同一个实验条件。
            "prefill_deadline_ms": self.prefill_deadline_ms,
            "max_batched_tokens": [
                self.max_batched_tokens_min,
                self.max_batched_tokens_max,
                self.max_batched_tokens_step,
            ],
            # 信号阈值必须进 describe()：它决定"控制器读了哪些反馈"，
            # 是实验登记的一部分。漏掉它的后果不是报错，而是产物里这几项
            # 静默变成 NOT_MEASURED——看起来像"没配"，实际是"没记"。
            "cache_hit_rate_high": self.cache_hit_rate_high,
            "min_available_memory_mb": self.min_available_memory_mb,
            "critical_available_memory_mb": self.critical_available_memory_mb,
        }


@dataclass
class ControllerState:
    """控制器的可变状态。与 run 绑定，可序列化以便复现决策序列。"""

    chunk_size: int = 4096
    max_batched_tokens: int = 8192
    #: 运行期并发上限。``None`` = 不控制，沿用配置里的静态值。
    max_num_seqs: int | None = None
    #: 收缩后的**恢复目标** = 配置初值。恢复不得越过它（见模块 docstring 第 6 条）。
    #: 放在 state 里而不是只留在 controller 上，是为了让导出的决策轨迹自带
    #: "这次放大是在恢复到哪一档"，否则事后看到一串 grow 无法判断收敛点在哪。
    init_chunk_size: int = 4096
    init_max_batched_tokens: int = 8192
    last_change_step: int = -10**9
    #: 已发生的**观测周期**数（一条 observe() 调用 = 一个周期）。
    #: 冷却按它计数，而不是按推理步——见 ``ControllerConfig.cooldown_cycles``。
    n_cycles: int = 0
    #: 上一次真正改动参数时处在第几个观测周期。``-10**9`` = 从未改动。
    last_change_cycle: int = -10**9
    #: 上一次 observe() 调用时的推理步。它与 ``n_cycles`` 一起，让"实际调用间隔"
    #: 变成可核对的事实：只记周期数看不出决策节拍是否真的等于配置值。
    last_decision_step: int | None = None
    n_updates: int = 0
    n_clamped: int = 0
    n_skipped_cooldown: int = 0
    #: 因压力消退而做的恢复次数。与 ``n_updates`` 分开计：把"恢复"和"加码"
    #: 混成一个计数，就看不出控制器是否只在一个方向上工作（棘轮）。
    n_recovered: int = 0
    #: 因信号被否决而未执行的次数。分别记，是为了回答
    #: "这次没动，是因为显存还是因为缓存"——这两个结论的后续动作完全不同。
    n_vetoed_memory: int = 0
    n_vetoed_cache: int = 0
    #: 因**绝对 SLO 护栏**（`slo_hard_guard`）而拒绝放大的次数。
    #: 单列：它与 memory/cache 两个否决是**不同性质**的拒绝（那两条是"不该放大"，
    #: 这条是"相对余量够但绝对线已破"），混算就看不出护栏是否真的在起作用。
    n_vetoed_slo_guard: int = 0
    #: 因「chunk 超过单步 token 预算」而做的可行性修复次数。
    #: 单列出来是因为它不是"调优"，而是把宣告出来的运行点拉回可行域。
    n_infeasible_repaired: int = 0

    # ---- 分层控制信号（第 04 步）-----------------------------------------
    #: 本周期生效的信号口径（``legacy`` / ``layered``）。落进轨迹的目的是
    #: 事后能分清"这批用了哪种信号"——否则同一个格的不同批次不可比而无人察觉。
    signal_mode: str = "legacy"
    #: 本次观测窗口里的样本数与**窗口年龄**（距上一次收到新样本多久）。
    #: 只报样本数是不够的：残留的旧样本数量不变，但已经过时。
    n_tpot_samples: int = 0
    #: ⚠️ ``None`` = **本口径不采集**（legacy），``0`` = 采集了但窗口为空。
    #: 两者必须分开：前者是"没有这个信号"，后者是"这个信号此刻没样本"。
    n_decode_samples: int | None = None
    #: 上一次**收到新样本**的时刻（单调 ms）。与窗口年龄一起构成"过时"判据。
    last_sample_at_ms: float | None = None
    #: 距上一次收到新样本过了多久（ms）。只报样本数不够——残留的旧样本数量不变，
    #: 但已经过时。
    window_age_ms: float | None = None
    #: 窗口内**真正给了纯 decode 覆盖**的样本数（即 `decode_ms > 0` 的那些）。
    #:
    #: ⚠️ 为什么必须单列（OPT-02 现测）：采集侧只在"本步**完全没有** prefill"时
    #: 才推进 decode 时钟（`benchmark/real_backend.py`），于是在**持续过载**段
    #: （几乎每一步都带 prefill）该时钟根本不动 ⇒ 窗口里 32 个样本**全为 0.0**
    #: ⇒ `decode_p90_ms` 读出 **0 ms**。而 0 ms 会被读成"纯 decode 几乎不花时间"，
    #: 即**最大余量**。实测后果：`S14B2` 三个自适应臂共 **64 次**决策据此在
    #: TTFT = 13.8–190 s 时说"TPOT 尚有 100.0% 余量"并**放大 chunk**。
    #: ⇒ `n_decode_samples=32` 与"其中 0 个有覆盖"是两件完全不同的事，必须分开记。
    n_decode_covered: int | None = None
    #: 纯 decode 的 P90（**与 interval 的 P90 不是同一个量**）。
    #: 无样本**或窗口内没有任何纯 decode 覆盖**时为 ``None``——**不可测 ≠ 0**。
    decode_p90_ms: float | None = None
    #: 窗口里有没有"含 prefill"的间隔。``None`` = 无样本。
    window_has_prefill: bool | None = None

    # ---- 工作点跟踪（OPT-02 / B2）----------------------------------------
    #: `gate_mode: normalized` 时的门限参考 = 在线跟踪的工作点（ms）。
    #: ``None`` = 尚未播种（此时退回绝对门限，并在轨迹里写明）。
    workload_scale_ms: float | None = None
    #: 参考值被更新的次数（可核验"跟踪是否真的在跑"，而不是恒为初值）。
    n_scale_updates: int = 0

    def describe(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "max_batched_tokens": self.max_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "init_chunk_size": self.init_chunk_size,
            "init_max_batched_tokens": self.init_max_batched_tokens,
            "last_change_step": self.last_change_step,
            "n_cycles": self.n_cycles,
            "last_change_cycle": self.last_change_cycle,
            "last_decision_step": self.last_decision_step,
            "n_updates": self.n_updates,
            "n_clamped": self.n_clamped,
            "n_skipped_cooldown": self.n_skipped_cooldown,
            "n_recovered": self.n_recovered,
            "n_vetoed_memory": self.n_vetoed_memory,
            "n_vetoed_cache": self.n_vetoed_cache,
            "n_vetoed_slo_guard": self.n_vetoed_slo_guard,
            "n_infeasible_repaired": self.n_infeasible_repaired,
            "signal_mode": self.signal_mode,
            "n_tpot_samples": self.n_tpot_samples,
            "n_decode_samples": self.n_decode_samples,
            # ⚠️ 覆盖率必须进产物：`n_decode_samples=32` 与"其中 0 个有纯 decode 覆盖"
            # 是两件完全不同的事，而后者会让 `decode_p90_ms` 变成 0（不是 None）。
            # 不记它，事后就看不出那次 0 读数到底是"真的快"还是"根本没覆盖"。
            "n_decode_covered": self.n_decode_covered,
            "last_sample_at_ms": _round_opt(self.last_sample_at_ms),
            "window_age_ms": _round_opt(self.window_age_ms),
            "decode_p90_ms": _round_opt(self.decode_p90_ms),
            "window_has_prefill": self.window_has_prefill,
            # 工作点参考是**归一化门限的锚**：不记它，门限的实际取值就无从复核。
            "workload_scale_ms": _round_opt(self.workload_scale_ms),
            "n_scale_updates": self.n_scale_updates,
        }


@dataclass
class ControlAction:
    """一次决策的结果。changed=False 时也要返回，便于记录「为什么没动」。"""

    changed: bool
    reason: str
    state: ControllerState
    #: ttft | tpot | memory | recover | none
    #: ``recover`` 是"压力消退后的恢复"，与 ``none``（安全区内不动）不是一回事：
    #: 前者动、后者不动。混用会让轨迹看起来自相矛盾。
    bottleneck: str = "none"
    #: 本次决策读到的反馈信号。值为 ``None`` = **该信号未测**，不参与决策，
    #: 但必须显形：否则事后无法区分"显存没压力"与"根本没量显存"。
    signals: dict[str, float | None] = field(default_factory=dict)
    #: 本次决策属于第几个观测周期（1 起计）。
    #: ``None`` = **这次调用没有消耗周期**（控制器关闭、或没有有效样本）。
    #: 用 ``None`` 而不是 0：0 会被读成"第 0 个周期"这个不存在的编号，
    #: 而这两类调用本来就没做出任何判断，记 0 会让"周期数"与"调用次数"混成一个数。
    cycle_index: int | None = None
    #: 距上一次**消耗了周期**的决策过了多少推理步（首次为 ``None``）。
    #: 与 ``cycle_index`` 一起落盘，"决策是否真的按配置节拍发生"才是可核对的：
    #: 只看周期序号看不出节拍，只看步间距看不出冷却有没有生效。
    steps_since_last_decision: int | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "reason": self.reason,
            "bottleneck": self.bottleneck,
            "cycle_index": self.cycle_index,
            "steps_since_last_decision": self.steps_since_last_decision,
            # ⚠️ 只对**数值**取整，其余原样透传。
            # 原先这里无条件 `round(v, 4)`，而第 04 步往 signals 里加了
            # ``signal_mode``（字符串）与 ``window_has_prefill``（bool）⇒
            # `round('legacy', 4)` 直接抛 TypeError，**18 个用例一起红**。
            # 判据放宽成"是数值才取整"，比"把非数值字段挪出 signals"好：
            # 这个字段就该出现在决策轨迹里（那正是它能被核对的原因）。
            "signals": {
                k: (
                    None
                    if v is None
                    else (round(v, 4) if isinstance(v, (int, float)) else v)
                )
                for k, v in self.signals.items()
            },
            "_unavailable_signals": sorted(
                k for k, v in self.signals.items() if v is None
            ),
            "state": self.state.describe(),
        }


# --------------------------------------------------------------------------- #
# 被协调的子控制器（规范第 12 节）
#
# 这两个类刻意设计成**无状态**：给定当前值与方向，返回新值与是否真的动了。
# 可变状态统一由 ControllerState 持有。理由：如果子控制器各自维护一份当前值，
# 就会和 ControllerState 形成两个真相来源，序列化出来的决策轨迹将无法解释
# ——这类不一致在调参阶段极难排查。
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Bound:
    """一个可调参数的区间与步长。"""

    minimum: int
    maximum: int
    step: int

    @property
    def active(self) -> bool:
        """区间退化（max<=0）表示该维度不参与控制。"""
        return self.maximum > 0 and self.maximum > self.minimum

    def grow(self, current: int) -> tuple[int, bool]:
        """上调一档。已在边界则返回原值并置 changed=False（由调用方计 clamp）。"""
        if not self.active or current >= self.maximum:
            return current, False
        return min(current + self.step, self.maximum), True

    def shrink(self, current: int) -> tuple[int, bool]:
        """下调一档。"""
        if not self.active or current <= self.minimum:
            return current, False
        return max(current - self.step, self.minimum), True

    def clamp(self, value: int) -> int:
        if not self.active:
            return value
        return max(self.minimum, min(value, self.maximum))

    def describe(self) -> list[int]:
        return [self.minimum, self.maximum, self.step]


class ChunkController:
    """动态选择 Prefill 分块大小（规范第 12 节）。

    要研究的取舍：分块越大，长 prompt 越快算完（降 TTFT），
    但单步占用越久，正在 decode 的请求会被卡住（抬 TPOT / stall）。
    因此它**只暴露 grow / shrink 两个动作**，不做判据——
    判据（谁是瓶颈）属于 AdaptiveController 的协调职责。
    """

    name = "chunk_controller"

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.bound = Bound(
            config.chunk_size_min, config.chunk_size_max, config.chunk_size_step
        )

    def grow(self, current: int) -> tuple[int, bool]:
        return self.bound.grow(current)

    def shrink(self, current: int) -> tuple[int, bool]:
        return self.bound.shrink(current)

    def clamp(self, value: int) -> int:
        return self.bound.clamp(value)

    def describe(self) -> dict[str, Any]:
        return {"controller": self.name, "bound": self.bound.describe()}


class AdaptiveBatchController:
    """选择批相关参数（规范第 12 节）。

    覆盖两个维度：
      - ``max_batched_tokens``：单步 token 预算。加大它提升吞吐，
        但会拉长单步耗时。
      - ``max_num_seqs``：并发序列上限。仅在配置显式开启时参与控制
        （见 ``ControllerConfig.batch_size_controlled``）——它改变准入语义，
        是比 token 预算更强的一档干预。
    """

    name = "adaptive_batch_controller"

    def __init__(self, config: ControllerConfig) -> None:
        self.config = config
        self.token_bound = Bound(
            config.max_batched_tokens_min,
            config.max_batched_tokens_max,
            config.max_batched_tokens_step,
        )
        self.seq_bound = Bound(
            config.batch_size_min, config.batch_size_max, config.batch_size_step
        )

    def grow(self, current: int) -> tuple[int, bool]:
        return self.token_bound.grow(current)

    def shrink(self, current: int) -> tuple[int, bool]:
        return self.token_bound.shrink(current)

    def clamp(self, value: int) -> int:
        return self.token_bound.clamp(value)

    def grow_seqs(self, current: int) -> tuple[int, bool]:
        return self.seq_bound.grow(current)

    def shrink_seqs(self, current: int) -> tuple[int, bool]:
        return self.seq_bound.shrink(current)

    def describe(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "token_bound": self.token_bound.describe(),
            "seq_bound": (
                self.seq_bound.describe() if self.seq_bound.active else None
            ),
        }


class AdaptiveController:
    """双目标自适应控制器。

    决策规则（按优先级短路）：

        M. 可用显存跌破临界线   -> 主动收缩（chunk + token 预算）。
           它不是"慢了"而是"快跑不动了"，压在一切时延判据之前。
        A. TPOT 超出目标 + 滞回 -> 当前瓶颈是 decode：缩小 chunk、同时降 token 预算
        B. TTFT 超出目标 + 滞回，且 TPOT 有余量 -> 当前瓶颈是 prefill：放大 chunk
           （若放大后的 chunk 已超过单步 token 预算，则一并抬高预算，
           否则这次放大永远兑现不了）
        C. 双目标都在目标内且此前收缩过 -> 逐档恢复到**配置初值**
        D. 双目标都在目标内且未收缩过 -> 不动（不要在安全区里「优化」，那只会引入抖动）

    放大类动作（B / C）还要过两道信号闸门：
        - 可用显存低于 ``min_available_memory_mb``  -> 否决放大
        - 缓存命中率高于 ``cache_hit_rate_high``    -> 否决"为 TTFT 放大分块"

    两个阈值默认 ``None``（不启用）：离线路径上没有显存读数，缓存也可能是关的。

    A / B 里的"目标"取哪个数由 ``gate_mode`` 决定（OPT-02 / B2）：
        - ``absolute``（默认）：`target_tpot_ms`。门限是一个固定 ms 数。
        - ``normalized``：**在线跟踪的工作点** `workload_scale_ms`。门限 = `scale·(1±h)`，
          带宽随工作点缩放 ⇒ "某个档位的工作点恰好落在带内"这件事在构造上不再发生
          （稳态时判据在带**中央**，而不是被一个外部常数切在半路上）。
          ⚠️ 代价：本臂不再以绝对 SLO 为控制目标，变成"检测并响应**变化**"的器件
          （升载收缩、卸载恢复）。要让绝对约束回来，用 ``slo_hard_guard``（默认关）。

    ⚠️ 判据 ``signal_criterion: decode_p90`` 必须过"**纯 decode 覆盖**"这道检查
    （OPT-02 补）：持续过载时窗口里可能一个纯 decode 步都没有，此时 `decode_p90_ms`
    是 ``None``（**不可测**）而不是 0 —— 写 0 会被读成"余量 100%"并**放大 chunk**。
    """

    name = "adaptive_controller"

    def __init__(
        self,
        config: ControllerConfig | None = None,
        state: ControllerState | None = None,
    ) -> None:
        self.config = config or ControllerConfig()
        self.state = state or ControllerState(
            chunk_size=self.config.chunk_size_init,
            max_batched_tokens=self.config.max_batched_tokens_init,
            max_num_seqs=(
                self.config.batch_size_init
                if self.config.batch_size_controlled
                else None
            ),
        )
        # 恢复目标始终取自配置：它是**配置属性**，不是运行期观测量。
        # state 可能来自磁盘上的历史轨迹，那种情况下它的 __init__ 语义不该被继承。
        self.state.init_chunk_size = self.config.chunk_size_init
        self.state.init_max_batched_tokens = self.config.max_batched_tokens_init
        # 工作点参考的**初值**同理是配置属性：声明了 seed 就要在 state 里立刻可见，
        # 否则产物在"第一次可测观测之前"显示 None，与配置里写着的 seed 对不上
        # （于是"这批到底用哪个参考起步"只能靠读配置去猜，而不是产物自证）。
        # seed 留空（None）则保持 None，由第一个可测观测播种。
        if self.config.gate_mode == "normalized":
            self.state.workload_scale_ms = self.config.workload_scale_seed_ms
        # 协调层持有两个子控制器；子控制器无状态，动作经这里统一落到 state 上。
        self.chunk_controller = ChunkController(self.config)
        self.batch_controller = AdaptiveBatchController(self.config)

    # ---------------- 对外 ----------------

    def observe(
        self,
        *,
        step: int,
        ttft_samples: list[float],
        tpot_samples: list[float],
        queue_depth: int = 0,
        running: int = 0,
        system_state: SystemState | None = None,
        decode_samples: list[float] | None = None,
        had_prefill_samples: list[bool] | None = None,
        observed_at_ms: float | None = None,
    ) -> ControlAction:
        """喂入一批观测，返回调整动作。

        本方法只做**周期记账**，判据主体在 :meth:`_decide`。

        一个观测周期 = 一次能做出判断的调用。提前返回的三种情况
        （控制器关闭、无有效样本、**窗口过期**）**不消耗周期**——它们本来就做不出任何判断，
        让它们推进冷却等于"冷却距离取决于调用方有多勤快"，那不是控制器的属性。
        这类调用返回的 ``cycle_index`` 是 ``None``，与"第几个周期"分得开。

        记账的两个数都会进决策轨迹：``cycle_index`` 让冷却的**单位**可核对，
        ``steps_since_last_decision`` 让决策的**实际节拍**可核对
        （应当等于 ``run.control_interval_steps``；不等就说明调用点与配置脱节）。

        ``decode_samples`` / ``had_prefill_samples`` 是**分层信号**（第 04 步）：
        与 ``tpot_samples`` 同窗口、**一一对应**。它们在本步**只被记录**，
        **不参与判据**——"纯 decode 特征有效"不等于"应把所有决策交给它"，
        那是第 09 步要检验的策略问题。
        """
        cfg = self.config
        if not cfg.enabled:
            return ControlAction(False, "controller disabled", self.state, "none")

        # ---- 窗口年龄 / 过期（第 04 步）--------------------------------------
        # 默认 ``signal_window_expiry_ms is None`` ⇒ 不判过期 ⇒ 既有行为**逐格不变**。
        # 设了值之后：距上一次**收到新样本**超过上限，本周期就不可信 ⇒ **不做动作**。
        # "保守"在这里的准确含义是：**不用过时的数据去动参数**——
        # 停在一个已知的配置上，比按几秒前的读数收缩/放大要好。
        self.state.signal_mode = cfg.signal_mode
        if observed_at_ms is not None:
            if len(tpot_samples) != self.state.n_tpot_samples:
                self.state.last_sample_at_ms = observed_at_ms   # 本周期确实来了新样本
            elif self.state.last_sample_at_ms is None:
                self.state.last_sample_at_ms = observed_at_ms
            self.state.n_tpot_samples = len(tpot_samples)
            self.state.window_age_ms = observed_at_ms - self.state.last_sample_at_ms
            expiry = cfg.signal_window_expiry_ms
            if expiry is not None and self.state.window_age_ms > expiry:
                return ControlAction(
                    False,
                    f"观测窗口过期：距上次新样本 {self.state.window_age_ms:.1f} ms "
                    f"> 上限 {expiry:.1f} ms ⇒ 按保守处理，本周期不动作",
                    self.state,
                    "none",
                )

        if not ttft_samples or not tpot_samples:
            return ControlAction(False, "无有效观测样本", self.state, "none")

        # ---- 分层信号：只记录，不参与判据 ------------------------------------
        self._record_layered(decode_samples, had_prefill_samples)

        self.state.n_cycles += 1
        interval = (
            None
            if self.state.last_decision_step is None
            else step - self.state.last_decision_step
        )
        self.state.last_decision_step = step

        action = self._decide(
            step=step,
            ttft_samples=ttft_samples,
            tpot_samples=tpot_samples,
            queue_depth=queue_depth,
            running=running,
            system_state=system_state,
        )
        return replace(
            action,
            cycle_index=self.state.n_cycles,
            steps_since_last_decision=interval,
        )

    def _record_layered(
        self,
        decode_samples: list[float] | None,
        had_prefill_samples: list[bool] | None,
    ) -> None:
        """把分层信号记进状态（第 04 步：**只记录，不参与判据**）。

        取不到时一律写 ``None``——不可测 ≠ 0。特别是 ``decode_p90_ms``：
        写成 0 会被读成"纯 decode 几乎不花时间"，那恰好是最乐观的一种读法。

        ⚠️ **OPT-02 补上的缺口**：光判"窗口空不空"不够。采集侧只在"本步**完全没有**
        prefill"时才推进 decode 时钟（`benchmark/real_backend.py`：
        ``if _step_has_prefill: prefill_clock += dt else: decode_clock += dt``），
        所以在**持续过载**段（几乎每步都带 prefill）该时钟根本不动 ⇒
        窗口里 32 个样本**全是 0.0** ⇒ 旧实现算出 `decode_p90_ms = 0.0`，
        它不是 ``None``，于是**穿过了 ``_decide`` 里那道 `is None` 的守卫**，
        被读成"TPOT = 0 ms ⇒ 余量 100%"。
        实测（`S14B2`，现读 `controller_trace`）：三个自适应臂共 **64 次**决策
        在 TTFT = 13.8–190 s 时说"TPOT 尚有 100.0% 余量"并**放大 chunk**。
        ⇒ 这里把"有没有**覆盖**"与"窗口大不大"分开：无覆盖的样本是**不可测**，
        不是"0 ms 的延迟"。它们同时要从分位数里**剔除**——混进去会把 P90 往下拽，
        给出一个比真实 decode 成本**更乐观**的读数。
        """
        if self.config.signal_mode != "layered":
            # legacy：完全不消费这两组样本 ⇒ 字段是 **None（没有这个信号）**，
            # 而不是 0（"采集了但窗口为空"）。两者对策不同：前者要开信号口径，
            # 后者要等样本。
            self.state.n_decode_samples = None
            self.state.n_decode_covered = None
            self.state.decode_p90_ms = None
            self.state.window_has_prefill = None
            return
        dec = [float(x) for x in (decode_samples or [])]
        flags = [bool(x) for x in (had_prefill_samples or [])]
        # 覆盖 = 该间隔里**确实花了纯 decode 时间**。0.0 表示这段等待没有被纯 decode
        # 步覆盖（含采样时序上的边界情形），两种情形下把它当"0 ms 延迟"都是错的。
        covered = [x for x in dec if x > 0.0]
        self.state.n_decode_samples = len(dec)
        self.state.n_decode_covered = len(covered)
        self.state.decode_p90_ms = percentile(covered, 90) if covered else None
        # 有没有"含 prefill 的间隔"：没有样本时是 None（未知），**不是 False**
        # ——"没测"与"测到了没有"给出的对策不同。
        self.state.window_has_prefill = any(flags) if flags else None

    def _decide(
        self,
        *,
        step: int,
        ttft_samples: list[float],
        tpot_samples: list[float],
        queue_depth: int = 0,
        running: int = 0,
        system_state: SystemState | None = None,
    ) -> ControlAction:
        """判据主体。

        用 P90 而不是均值做判据：尾延迟才是 SLO 违约的来源，
        而均值会被大量快请求掩盖（这正是「看均值一切正常、线上却在骂」的成因）。

        ``system_state`` 是 :class:`~ascend_opt.monitor.WorkloadMonitor` 汇总出的
        ``s_t``（规范第 2 节）。控制器只消费它的**非时延**分量（缓存命中率、
        可用显存、NPU 利用率）：时延判据必须来自样本——需要的是分布（P90），
        而不是一个已经被聚合掉的点。

        调节是**双向**的：见模块 docstring 第 6 条。
        """
        cfg = self.config

        ttft = percentile(ttft_samples, 90)
        tpot = percentile(tpot_samples, 90)
        signals = _signal_snapshot(system_state, ttft=ttft, tpot=tpot)
        # 分层信号（第 04 步）：进决策轨迹，供第 09 步做**干预实验**。
        # ⚠️ 本步**不参与任何判据**——把"这段等待是 decode 慢还是被别人挤的"
        # 变成可观测的事实，比立刻用它改判据重要：判据形式是第 09 步的问题。
        #
        # ⚠️ **只在 layered 下并入**：legacy 的轨迹必须与之前**逐字节相同**，
        # 否则"旧批次语义不变"就成了空话（轨迹本身也是被比对的产物）。
        # ``signal_mode`` 不靠这里记录——它在 ``state.describe()`` 里，两种模式都有。
        if self.state.signal_mode == "layered":
            signals.update(
                {
                    "decode_p90_ms": self.state.decode_p90_ms,
                    "window_has_prefill": self.state.window_has_prefill,
                    "window_age_ms": self.state.window_age_ms,
                    "n_tpot_samples": self.state.n_tpot_samples,
                    "n_decode_samples": self.state.n_decode_samples,
                }
            )
        cache_hit = signals["cache_hit_rate"]
        available_mb = signals["available_memory_mb"]

        # ---- 判据口径（第 09 步）---------------------------------------------
        # `tpot` 默认来自"相邻两次 token 之间的间隔" ⇒ 含 prefill 的步会把它污染
        # （E10：读数 337 ms vs 纯 decode 61 ms）。`decode_p90` 只用**纯 decode 步**。
        criterion = cfg.signal_criterion
        crit_label = "TPOT" if criterion == "tpot_p90" else "decode"
        if criterion == "decode_p90":
            tpot = self.state.decode_p90_ms
            # ⚠️ 只在**非默认口径**下并入这两个键 ⇒ 默认口径的决策轨迹与
            # 加该字段之前**逐字节一致**（同第 04 步 layered 键的做法）。
            signals["criterion"] = criterion
            signals["criterion_ms"] = tpot
            if tpot is None:
                # ⚠️ 不可测 ⇒ **保守不动**，且**不回退**到 `tpot_p90`。
                # 回退等于把这一步要消除的污染又放回来，而且"看起来在工作"。
                # ⚠️ 两种"不可测"都走这里：窗口**空**，或窗口**没有任何纯 decode 覆盖**
                # （后者 OPT-02 补上——旧实现会算出 0.0，不是 None，于是穿过这道守卫）。
                return ControlAction(
                    False,
                    "判据 decode_p90 不可测（窗口为空或窗口内无纯 decode 覆盖）⇒ 保守不动",
                    self.state,
                    "none",
                    signals,
                )

        # ---- 门限基准：绝对 vs 按工作点归一化（OPT-02 / B2）------------------
        # 绝对形式：门限 = `target_tpot_ms·(1±h)`，一个固定 ms 数。
        # 归一化形式：门限 = `workload_scale_ms·(1±h)`，基准由**同一会话内在线跟踪**。
        # 为什么要改：固定带宽 ≈19.2 ms 与多档工作点分不开（见 `gate_mode` 字段注释
        # 与 `EXPERIMENTS.md` §4.3.4）。归一化之后"带"始终**跟着工作点走**，
        # 于是"某个档位的工作点恰好落在带内"这件事在构造上不再发生——
        # 稳态时判据就在带的**中央**，而不是被一个外部常数切在半路上。
        gate_base_ms = cfg.target_tpot_ms
        if cfg.gate_mode == "normalized":
            if tpot is not None:
                if self.state.workload_scale_ms is None:
                    seed = cfg.workload_scale_seed_ms
                    # 没给 seed ⇒ 由**第一个可测观测**播种。已登记的后果：若 run 一开始
                    # 就在过载区，参考值会直接落在过载工作点上 ⇒ 控制器认为"这就是常态"
                    # ⇒ 不动作。要避免就显式给一个本会话测得的空载/低载参考。
                    self.state.workload_scale_ms = float(seed if seed is not None else tpot)
                else:
                    a = cfg.workload_scale_alpha
                    self.state.workload_scale_ms += a * (tpot - self.state.workload_scale_ms)
                self.state.n_scale_updates += 1
            if self.state.workload_scale_ms is not None:
                gate_base_ms = self.state.workload_scale_ms
            # ⚠️ 只在归一化口径下并入这个键 ⇒ 绝对口径的决策轨迹与加它之前**逐字节一致**。
            signals["gate_base_ms"] = gate_base_ms

        tpot_limit = gate_base_ms * (1.0 + cfg.hysteresis)
        ttft_limit = cfg.target_ttft_ms * (1.0 + cfg.hysteresis)
        # 门限文本必须写明**用的是哪个基准**：归一化下 `target_tpot_ms` 已不是门限，
        # 还照旧印"超目标 120.0ms"会让读者把绝对目标当成判据。
        if cfg.gate_mode == "normalized":
            limit_label = (f"超门限基准 {gate_base_ms:.1f}ms"
                           f"（工作点归一化；绝对目标 {cfg.target_tpot_ms:.1f}ms 不参与判据）")
        else:
            limit_label = f"超目标 {cfg.target_tpot_ms:.1f}ms"

        # 未测（None）的阈值不参与判断——不触发，也不否决。
        memory_critical = (
            available_mb is not None
            and cfg.critical_available_memory_mb is not None
            and available_mb < cfg.critical_available_memory_mb
        )
        memory_veto = (
            available_mb is not None
            and cfg.min_available_memory_mb is not None
            and available_mb < cfg.min_available_memory_mb
        )
        cache_veto = (
            cache_hit is not None
            and cfg.cache_hit_rate_high is not None
            and cache_hit >= cfg.cache_hit_rate_high
        )

        if memory_critical:
            bottleneck = "memory"
        elif tpot > tpot_limit:
            bottleneck = "tpot"
        elif ttft > ttft_limit:
            bottleneck = "ttft"
        elif self.recovery_pending():
            bottleneck = "recover"
        else:
            bottleneck = "none"

        if bottleneck == "none":
            return ControlAction(
                False,
                f"双目标均在滞回带内 (ttft={ttft:.1f}<={ttft_limit:.1f}, "
                f"tpot={tpot:.1f}<={tpot_limit:.1f})，且无需恢复",
                self.state,
                "none",
                signals,
            )

        # 冷却：刚调整过就再调，等于在噪声上做反馈控制。
        # 计数单位是**观测周期**（本次调用是第几个周期），不是推理步——
        # 按推理步比较时，默认节拍 8 步 / 冷却 3 步会让这个判断永远为假。
        cycles_since_change = self.state.n_cycles - self.state.last_change_cycle
        if cycles_since_change < cfg.cooldown_cycles:
            self.state.n_skipped_cooldown += 1
            return ControlAction(
                False,
                f"冷却中（距上次调整 {cycles_since_change} 个观测周期，"
                f"需 {cfg.cooldown_cycles} 个）",
                self.state,
                bottleneck,
                signals,
            )

        if bottleneck == "memory":
            changed = self._shrink(chunk=True, tokens=True)
            reason = (
                f"可用显存 {available_mb:.0f}MB 低于临界 "
                f"{cfg.critical_available_memory_mb:.0f}MB，主动收缩以先活下来"
            )
        elif bottleneck == "tpot":
            changed = self._shrink(chunk=True, tokens=True)
            reason = (
                f"{crit_label} P90={tpot:.1f}ms {limit_label}，"
                f"给 decode 让出预算"
            )
        elif bottleneck == "ttft":
            # TTFT 超目标但 TPOT 仍有较大余量时才敢放大 chunk；
            # 否则会把本来达标的 TPOT 推过线——这是取舍，不是免费收益。
            #
            # ⚠️ 余量以**门限基准**为分母（归一化下 = 跟踪的工作点，绝对下 = 绝对目标）。
            # 这样 `headroom > h` 与"判据位于带内"才是同一个条件
            # （tpot < base·(1−h) ⟺ (base−tpot)/base > h），不会出现两处各写一个口径。
            tpot_headroom = (gate_base_ms - tpot) / gate_base_ms
            if cfg.slo_hard_guard and tpot > cfg.target_tpot_ms * (1.0 + cfg.hysteresis):
                # 可选护栏：归一化门限下"相对余量够"**不**等于"绝对 SLO 没超"。
                # 默认关——把两条约束叠起来会让"这条臂在追什么"不可分辨。
                self.state.n_vetoed_slo_guard += 1
                return ControlAction(
                    False,
                    f"TTFT 超目标且相对余量足，但绝对 SLO 护栏否决："
                    f"{crit_label} P90={tpot:.1f}ms > 绝对上限 "
                    f"{cfg.target_tpot_ms * (1.0 + cfg.hysteresis):.1f}ms，拒绝放大",
                    self.state,
                    "ttft",
                    signals,
                )
            if memory_veto:
                self.state.n_vetoed_memory += 1
                return ControlAction(
                    False,
                    f"TTFT 超目标，但可用显存 {available_mb:.0f}MB 低于否决线 "
                    f"{cfg.min_available_memory_mb:.0f}MB，拒绝放大",
                    self.state,
                    "ttft",
                    signals,
                )
            if cache_veto:
                self.state.n_vetoed_cache += 1
                return ControlAction(
                    False,
                    f"TTFT 超目标，但缓存命中率 {cache_hit:.2f}>="
                    f"{cfg.cache_hit_rate_high:.2f}：未命中部分已被缓存省下，"
                    f"放大分块的收益有限",
                    self.state,
                    "ttft",
                    signals,
                )
            if tpot_headroom <= cfg.hysteresis:
                return ControlAction(
                    False,
                    f"TTFT 超目标但 TPOT 已无余量 (headroom={tpot_headroom:.3f})，"
                    f"拒绝放大 chunk",
                    self.state,
                    "ttft",
                    signals,
                )
            changed = self._grow(chunk=True, tokens=False)
            reason = f"TTFT P90={ttft:.1f}ms 超目标，且 TPOT 尚有 {tpot_headroom:.1%} 余量"
            if changed and self.state.chunk_size > self.state.max_batched_tokens:
                # 一个大于单步 token 预算的 chunk 永远兑现不了：策略侧会把它夹到预算。
                # 只调 chunk 不调预算的后果就是"日志里调大了、额度没变"，
                # 所以这里必须一起抬——这也是 chunk 与 token 两个维度之间唯一的耦合点。
                if self._grow(chunk=False, tokens=True):
                    reason += "；chunk 已超单步 token 预算，一并抬高预算使其可兑现"
                else:
                    reason += (
                        "；chunk 已超单步 token 预算且预算已达上界，"
                        "本次放大无法完全兑现"
                    )
        else:  # recover
            if memory_veto:
                self.state.n_vetoed_memory += 1
                return ControlAction(
                    False,
                    f"压力已消退，但可用显存 {available_mb:.0f}MB 仍低于否决线 "
                    f"{cfg.min_available_memory_mb:.0f}MB，暂不恢复",
                    self.state,
                    "recover",
                    signals,
                )
            if cache_veto:
                self.state.n_vetoed_cache += 1
            changed = self._recover(allow_chunk=not cache_veto)
            reason = f"双目标均在目标内，恢复此前的防御性收缩（上限=配置初值）"
            if cache_veto:
                reason += (
                    f"；缓存命中率 {cache_hit:.2f}>={cfg.cache_hit_rate_high:.2f}，"
                    f"只恢复 token 预算，不放大分块"
                )
            if changed:
                self.state.n_recovered += 1

        # 可行性修复：chunk 不得超过单步 token 预算。
        # 一个大于预算的 chunk 永远兑现不了（策略会把它夹到预算），
        # 于是"建议值 vs 生效值"的差额会变成常驻噪声——**而那个差额本该是
        # 用来发现失效的检查**。两个维度的步长不同（chunk 512 / token 1024），
        # 同时收缩时 chunk 会相对抬起来，所以这个修复不是理论上的洁癖，是必须的。
        if self._repair_infeasible_chunk():
            changed = True
            reason += (
                f"；chunk 下调到 {self.state.chunk_size}"
                f"（不得超过单步 token 预算，否则无法兑现）"
            )

        if not changed:
            return ControlAction(
                False,
                f"{reason}，但当前无可调整的维度（已达边界或被信号否决）",
                self.state,
                bottleneck,
                signals,
            )

        self.state.last_change_step = step
        # 冷却按周期计数，所以两处时间戳都要写：step 用于人读轨迹，
        # cycle 用于冷却判断。只写一个，另一个就会静默失效。
        self.state.last_change_cycle = self.state.n_cycles
        self.state.n_updates += 1
        return ControlAction(True, reason, self.state, bottleneck, signals)

    def recovery_pending(self) -> bool:
        """是否存在「此前被收缩、现在可以恢复」的维度。"""
        return (
            self.state.chunk_size < self.state.init_chunk_size
            or self.state.max_batched_tokens < self.state.init_max_batched_tokens
        )

    def _repair_infeasible_chunk(self) -> bool:
        """把 chunk 压回单步 token 预算以内；无违约时不做任何事。

        这是**可行性约束**而不是策略：``chunk_size <= max_batched_tokens``
        描述的是可行域，不是偏好。控制器宣告一个不可行的运行点，会让下游
        （策略/调度器）替它夹一次——夹是静默的，检查就失效了。
        """
        if self.state.chunk_size <= self.state.max_batched_tokens:
            return False
        target = self.chunk_controller.clamp(
            min(self.state.max_batched_tokens, self.state.chunk_size)
        )
        if target == self.state.chunk_size:
            return False
        self.state.chunk_size = target
        self.state.n_infeasible_repaired += 1
        return True

    def plan_describe(self) -> dict[str, Any]:
        return {
            "controller": self.name,
            "config": self.config.describe(),
            "state": self.state.describe(),
            "components": [
                self.chunk_controller.describe(),
                self.batch_controller.describe(),
            ],
        }

    # ---------------- 内部 ----------------

    def _grow(self, *, chunk: bool, tokens: bool) -> bool:
        """上调。chunk 与 tokens 的分支互不干扰（各自独立计一次 clamp）。"""
        changed = False
        if chunk:
            new_chunk, moved = self.chunk_controller.grow(self.state.chunk_size)
            if moved:
                self.state.chunk_size = new_chunk
                changed = True
            else:
                self.state.n_clamped += 1
        if tokens:
            new_tokens, moved = self.batch_controller.grow(self.state.max_batched_tokens)
            if moved:
                self.state.max_batched_tokens = new_tokens
                changed = True
            else:
                self.state.n_clamped += 1
        return changed

    def _shrink(self, *, chunk: bool, tokens: bool) -> bool:
        changed = False
        if chunk:
            new_chunk, moved = self.chunk_controller.shrink(self.state.chunk_size)
            if moved:
                self.state.chunk_size = new_chunk
                changed = True
            else:
                self.state.n_clamped += 1
        if tokens:
            new_tokens, moved = self.batch_controller.shrink(self.state.max_batched_tokens)
            if moved:
                self.state.max_batched_tokens = new_tokens
                changed = True
            else:
                self.state.n_clamped += 1
        return changed

    def _recover(self, *, allow_chunk: bool = True, allow_tokens: bool = True) -> bool:
        """把此前因过载而收缩的维度逐档恢复到**配置初值**。

        三个约束，每个都对应一个具体的失效：

        1. **以初值为上限**，不是以 ``*_max`` 为上限。恢复不是探索——
           在安全区里继续放大没有收益，只会让下一个突发到来时系统处在更脆弱的工况
           （见模块 docstring 第 2、6 条）。
        2. **逐档**（每次一个 step），不一步跳回初值。一步跳回会让"恢复"本身
           成为一次扰动源，与冷却机制的目的正好相反。
        3. **只恢复低于初值的维度**。已经高于初值（TTFT 压力下加过码）的维度
           不属于"恢复"，不该被这个分支碰——否则恢复分支会变成第二个放大器，
           两个分支同时放大就无法归因。
        """
        changed = False
        if allow_chunk and self.state.chunk_size < self.state.init_chunk_size:
            new_chunk, _ = self.chunk_controller.grow(self.state.chunk_size)
            new_chunk = min(new_chunk, self.state.init_chunk_size)
            if new_chunk != self.state.chunk_size:
                self.state.chunk_size = new_chunk
                changed = True
        if allow_tokens and self.state.max_batched_tokens < self.state.init_max_batched_tokens:
            new_tokens, _ = self.batch_controller.grow(self.state.max_batched_tokens)
            new_tokens = min(new_tokens, self.state.init_max_batched_tokens)
            if new_tokens != self.state.max_batched_tokens:
                self.state.max_batched_tokens = new_tokens
                changed = True
        return changed


def _signal_snapshot(
    state: SystemState | None,
    *,
    ttft: float,
    tpot: float,
) -> dict[str, Any]:
    """把本次决策读到的信号摊平成可序列化的一行。

    ``None`` 一律表示"该信号**未测**"：离线路径上没有显存读数、缓存可能是关的。
    未测的值不参与决策（既不触发收缩，也不否决放大），但必须出现在轨迹里——
    不然事后无法区分"没压力"与"没测"，而这两者对应的动作完全相反。

    ⚠️ 返回类型放宽成 ``Any``（第 04 步）：调用方会再并入**分层信号**
    （``decode_p90_ms`` 是 ``float | None``，``window_has_prefill`` 是 ``bool | None``，
    ``n_*`` 是 ``int``）。它们同样遵守"未测写 ``None``"这条铁律。
    """
    return {
        "ttft_p90_ms": ttft,
        "tpot_p90_ms": tpot,
        "cache_hit_rate": state.cache_hit_ratio if state is not None else None,
        "available_memory_mb": state.available_memory_mb if state is not None else None,
        "npu_utilization_pct": state.npu_utilization_pct if state is not None else None,
    }


def replay_decisions(
    controller: AdaptiveController,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """离线复放一段观测序列，产出决策轨迹。

    用途：调参时不必上真机——先在已有 telemetry 上验证控制器是否会振荡、
    以及在压力上升 / 消退 / 负载切换下是否表现得像预期。

    每条观测是一个 dict，字段与 ``observe`` 的入参同名，另可选带
    ``cache_hit_rate`` / ``available_memory_mb`` / ``npu_utilization_pct``
    三个**信号**字段（对应 ``s_t`` 的非时延分量）。信号字段缺省即"未测"，
    保持 ``None``——不要用 0 代替：0 会被读成"命中率为 0 / 显存已耗尽"。

    ⚠️ 可选的 ``decode_samples`` / ``had_prefill_samples`` / ``observed_at_ms``
    （OPT-02 补）：缺省**完全不传** ⇒ 与加这三个字段之前**逐字节一致**。
    补它们是为了让**新口径可离线复放**——`signal_criterion: decode_p90` 与
    `gate_mode: normalized` 都依赖分层样本，而此前 replay 根本喂不进来 ⇒
    "离线验证新判据形式"这件事在工具上**做不到**（只能上真机，与"离线优先"相悖）。
    传了 `decode_samples`` 但 `signal_mode` 仍是 ``legacy`` 时它们被忽略
    （`_record_layered` 的第一道分支），因此不会悄悄改变旧复放的结果。
    """
    trace: list[dict[str, Any]] = []
    for obs in observations:
        action = controller.observe(
            step=int(obs.get("step", len(trace))),
            ttft_samples=list(obs.get("ttft_samples", [])),
            tpot_samples=list(obs.get("tpot_samples", [])),
            queue_depth=int(obs.get("queue_depth", 0)),
            running=int(obs.get("running", 0)),
            system_state=_state_from_observation(obs),
            # 缺省不传（None）⇒ 旧复放行为逐字节不变。传了才喂分层信号。
            decode_samples=(list(obs["decode_samples"]) if "decode_samples" in obs else None),
            had_prefill_samples=(list(obs["had_prefill_samples"])
                                 if "had_prefill_samples" in obs else None),
            observed_at_ms=obs.get("observed_at_ms"),
        )
        trace.append({"step": obs.get("step", len(trace)), **action.describe()})
    return trace


def _state_from_observation(obs: dict[str, Any]) -> SystemState | None:
    """从一条复放记录里还原 ``s_t`` 的可用部分。

    一条观测里完全没有信号字段时返回 ``None``（整条 ``s_t`` 未测），
    而不是一个全 ``None`` 的 ``SystemState``——两者对决策等价，
    但前者让"这次复放根本没接信号"这件事在代码里一眼可见。
    """
    names = ("cache_hit_rate", "available_memory_mb", "npu_utilization_pct")
    if not any(k in obs for k in names):
        return None
    return SystemState(
        step=obs.get("step"),
        cache_hit_ratio=obs.get("cache_hit_rate"),
        available_memory_mb=obs.get("available_memory_mb"),
        npu_utilization_pct=obs.get("npu_utilization_pct"),
    )
