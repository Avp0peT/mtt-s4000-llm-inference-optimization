"""工作负载建模。

把「要跑什么请求」抽成可复现的纯数据对象：同一个 ``WorkloadSpec`` 在任何机器上
都应生成完全相同的请求序列（由 ``seed`` 决定），这是跨 run 可比的前提。

设计要点：
- ``Request`` 不可变（frozen），避免下游策略意外改写共享的请求对象。
- 到达时间使用相对毫秒（相对 run 起点），不用绝对时间戳，便于重放与比较。
- 共享前缀用 ``shared_prefix_len`` 显式表达，前缀缓存的收益才有可解释的基数。

本模块不涉及任何硬件或框架调用。
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Literal

ArrivalModel = Literal["closed", "closed_jittered", "poisson", "burst", "piecewise"]

#: ``piecewise`` 段内到达时刻的分布形式（两种都确定性：同 spec 同输出）。
#:   ``jittered`` 每段的请求数**精确**等于该段名义值，位置在等间隔上抖动（分层抽样）；
#:   ``poisson``  只固定总数，位置按条件泊松抽取（给定总数即 i.i.d. ~ λ(t)/Λ(T)），
#:   各段实际请求数按多项分布波动。
#: 选哪个是**方法论问题**而不是实现细节：``jittered`` 让"负载水平"这个自变量是精确的，
#: ``poisson`` 更接近真实到达，但段内请求数少时会有可观的相对噪声。
#: 两者都保证总请求数恰为 ``num_requests``，不会出现"生成数与请求数不一致"的静默截断。
RateNoise = Literal["jittered", "poisson"]


@dataclass(frozen=True)
class Request:
    """单个推理请求的静态描述。"""

    req_id: str
    prompt_len: int          # prompt token 数
    gen_len: int             # 期望生成的 token 数
    arrival_ms: float = 0.0  # 相对 run 起点的到达时刻（ms）
    shared_prefix_len: int = 0   # 与其它请求共享的前缀长度（token）
    prefix_group: str = ""       # 共享前缀的归属组，用于判定缓存可复用性

    @property
    def unique_prefix_len(self) -> int:
        """前缀缓存无法覆盖、必须真正计算的部分。"""
        return max(0, self.prompt_len - self.shared_prefix_len)

    @property
    def total_tokens(self) -> int:
        return self.prompt_len + self.gen_len

    def describe(self) -> dict[str, Any]:
        return {
            "req_id": self.req_id,
            "prompt_len": self.prompt_len,
            "gen_len": self.gen_len,
            "arrival_ms": round(self.arrival_ms, 3),
            "shared_prefix_len": self.shared_prefix_len,
            "prefix_group": self.prefix_group,
        }


@dataclass(frozen=True)
class RateSegment:
    """分段常量速率 λ(t) 的一段。

    ``duration_s`` 是本段占用的**墙钟**时长（秒），``qps`` 是本段内的恒定到达速率；
    两者相乘是本段的名义请求数。段之间首尾相接，因此负载是一条
    「低压 → 高压 → 恢复」的**水平轨迹**，而不是某个固定并发。

    为什么用"速率 × 时长"而不是"并发数"：并发是**结果**（由到达速率与服务速率
    共同决定），把它当自变量会让它与调度策略自我耦合——控制器一改批预算，
    并发就跟着变，"负载水平"这个自变量也就不再是同一个东西了。
    """

    duration_s: float
    qps: float
    label: str = ""       # 段的人类可读名字（low / high / recover），只用于报表

    @property
    def nominal_count(self) -> float:
        return self.duration_s * self.qps

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "duration_s": round(self.duration_s, 3),
            "qps": round(self.qps, 6),
            "nominal_count": round(self.nominal_count, 3),
        }


@dataclass
class WorkloadSpec:
    """工作负载的生成参数。

    长度分布用 (min, max) 区间 + 均匀采样表示，够用且便于手算期望；
    若后续需要重尾分布（真实对话流量常见），在这里加新字段而不是改语义。
    """

    name: str
    num_requests: int = 64
    prompt_len_min: int = 512
    prompt_len_max: int = 512
    gen_len_min: int = 128
    gen_len_max: int = 128
    concurrency: int = 8
    arrival: ArrivalModel = "closed"   # closed=满并发后补位；poisson / burst=按速率；piecewise=分段 λ(t)
    #: ★ **闭环补位抖动幅度**（ms）—— 仅 `arrival in {closed, closed_jittered}` 时有效。
    #:
    #: 背景（`B1` 草案 §5 / OPT-03R 裁定 Q4）：`closed` 下补位时刻恒 = `step_start`
    #: ⇒ 确定性负载 + 固定配置时 `arrival_ms` **逐位相同**（`S14B2` 实测 0/92 不同）
    #: ⇒ 主指标跨重复噪声带仅 CV 0.01–0.09% ⇒ 小效应被淹没、配对判据退化。
    #: 设 >0 时执行器把补位时刻改为 `step_start + 确定性抖动`（`closed_jittered`），
    #: 使批边界随调度改变 ⇒ 跨重复有差异（**在途上限仍受 `concurrency` 硬约束**）。
    #:
    #: ⚠️ `0`（默认）= 关闭 ⇒ 与既有 `closed` 行为**逐字节相同**。
    #: ⚠️ 幅度须在预注册里**写死**，不得为凑结论事后调。
    closure_jitter_ms: float = 0.0
    request_rate_qps: float = 1.0      # 仅 arrival in {poisson, burst} 时有效
    rate_segments: tuple[RateSegment, ...] = ()  # 仅 arrival == "piecewise" 时有效
    rate_noise: RateNoise = "jittered"           # 仅 arrival == "piecewise" 时有效
    prefix_groups: int = 0             # >0 时启用共享前缀分组
    shared_prefix_len: int = 0         # 每组共享的前缀长度
    seed: int = 0

    def __post_init__(self) -> None:
        """piecewise 下总请求数必须与段定义一致。

        放在 ``__post_init__`` 而不是只在 ``from_dict``：**直接构造**
        ``WorkloadSpec``（脚本、单元测试、注册表里的手写条目）也要受同一条约束，
        否则会出现"配置路径下报错、代码路径下静默"的双重标准。
        """
        self.rate_segments = _coerce_rate_segments(self.rate_segments)
        # ★ 闭环补位抖动只在闭环模型下有定义（`closed` / `closed_jittered`）。
        # 放在这里而不是只在校验里"忽略"：若在开放环上悄悄忽略它，负载会少了
        # 一整项而产物里完全看不出来 —— 那正是本仓库反复警戒的静默失效。
        if self.closure_jitter_ms and self.arrival not in ("closed", "closed_jittered"):
            raise ValueError(
                "closure_jitter_ms 只在 closed / closed_jittered 下有效；"
                f"当前 arrival={self.arrival!r}。"
                "开放环的到达时刻由 arrival_ms 直接给出，抖动应由该分布承担。"
            )
        if self.closure_jitter_ms < 0:
            raise ValueError("closure_jitter_ms 不能为负（它是抖动的**幅度**上界）")
        if self.arrival != "piecewise":
            if self.rate_segments:
                raise ValueError(
                    "rate_segments 只在 arrival=piecewise 下有效；"
                    f"当前 arrival={self.arrival!r}，请删掉 rate_segments 或改用 piecewise"
                )
            return
        _validate_rate_segments(self.rate_segments)
        derived = _derive_piecewise_count(self.rate_segments, self.rate_noise)
        if self.num_requests != derived:
            raise ValueError(
                f"arrival=piecewise 时 num_requests 由 rate_segments 决定（= {derived}），"
                f"与传入的 {self.num_requests} 不一致；"
                f"请把 num_requests 改成 {derived}，或调整各段的 duration_s / qps。"
            )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_requests": self.num_requests,
            "prompt_len": [self.prompt_len_min, self.prompt_len_max],
            "gen_len": [self.gen_len_min, self.gen_len_max],
            "concurrency": self.concurrency,
            "arrival": self.arrival,
            "closure_jitter_ms": self.closure_jitter_ms,
            "request_rate_qps": self.request_rate_qps,
            "rate_segments": [s.describe() for s in self.rate_segments],
            "rate_noise": self.rate_noise,
            "prefix_groups": self.prefix_groups,
            "shared_prefix_len": self.shared_prefix_len,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkloadSpec":
        """从字典构造。

        未知字段直接报错——配置里把 ``prompt_len`` 拼成 ``prompt_lens`` 这类错误
        必须立刻可见，否则会静默退化成默认值，实验条件与预期不符而无人察觉。

        **例外**：下划线开头的键被视为**注释**，直接忽略。
        这条约定让调用方可以往负载字典里附加元信息（例如扫描矩阵标注的
        ``_axes`` / ``_is_anchor``）而不必先剥掉它们，同时不削弱上面那条严格性
        ——没有任何真实字段以下划线开头。
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = {
            k for k in payload if not k.startswith("_")
        } - known
        if unknown:
            raise ValueError(
                f"workload 配置含未知字段 {sorted(unknown)}；"
                f"可用字段：{sorted(known)}"
            )
        clean = {k: v for k, v in payload.items() if not k.startswith("_")}
        clean["rate_segments"] = _coerce_rate_segments(clean.get("rate_segments"))
        noise = clean.get("rate_noise", "jittered")
        if noise not in ("jittered", "poisson"):
            raise ValueError(
                f"rate_noise 只能是 jittered / poisson，收到 {noise!r}"
            )
        if clean.get("arrival") == "piecewise":
            segs = clean["rate_segments"]
            _validate_rate_segments(segs)
            # 总请求数由段定义**决定**，不接受第二个真值：
            # 两个都能改就会出现"负载被截断/延长而没人发现"的静默失效。
            derived = _derive_piecewise_count(segs, noise)
            if "num_requests" in payload and payload["num_requests"] != derived:
                raise ValueError(
                    f"arrival=piecewise 时 num_requests 由 rate_segments 决定（= {derived}），"
                    f"与配置里写的 {payload['num_requests']} 不一致；"
                    f"请把 num_requests 改成 {derived}，或调整各段的 duration_s / qps。"
                )
            clean["num_requests"] = derived
        elif clean["rate_segments"]:
            raise ValueError(
                "rate_segments 只在 arrival=piecewise 下有效；"
                f"当前 arrival={clean.get('arrival')!r}，请删掉 rate_segments 或改用 piecewise"
            )
        return cls(**clean)

    def build(self) -> list[Request]:
        return build_requests(self)


def _coerce_rate_segments(value: Any) -> tuple[RateSegment, ...]:
    """把配置里的 ``list[dict]`` 变成 ``RateSegment`` 元组。

    段内字段拼错同样**直接报错**：写成 ``qps_s`` 会静默退化成缺少该段，
    负载轨迹少一整段而产物里看不出来。
    """
    if value is None:
        return ()
    if isinstance(value, RateSegment):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"rate_segments 必须是列表（每项含 duration_s / qps，可选 label），"
            f"收到 {type(value).__name__}"
        )
    known = {f.name for f in RateSegment.__dataclass_fields__.values()}
    out: list[RateSegment] = []
    for i, item in enumerate(value):
        if isinstance(item, RateSegment):
            # 允许字典与 RateSegment 混写：注册表里手写条目常直接构造对象
            out.append(item)
            continue
        if not isinstance(item, dict):
            raise ValueError(
                f"rate_segments[{i}] 必须是字典，收到 {type(item).__name__}"
            )
        unknown = set(item) - known
        if unknown:
            raise ValueError(
                f"rate_segments[{i}] 含未知字段 {sorted(unknown)}；可用字段：{sorted(known)}"
            )
        out.append(RateSegment(**item))
    return tuple(out)


def _validate_rate_segments(segs: tuple[RateSegment, ...]) -> None:
    if not segs:
        raise ValueError(
            "arrival=piecewise 必须给出 rate_segments（至少一段），"
            "每段形如 {duration_s: 30.0, qps: 2.0, label: low}"
        )
    for i, s in enumerate(segs):
        if not math.isfinite(s.duration_s) or s.duration_s <= 0:
            raise ValueError(
                f"rate_segments[{i}].duration_s 必须为正有限值，收到 {s.duration_s!r}"
            )
        if not math.isfinite(s.qps) or s.qps <= 0:
            raise ValueError(
                f"rate_segments[{i}].qps 必须为正有限值，收到 {s.qps!r}"
            )


def _derive_piecewise_count(segs: tuple[RateSegment, ...], noise: str) -> int:
    """piecewise 的总请求数由段定义决定（不是另一个可自由填的字段）。"""
    if noise == "jittered":
        total = 0
        for i, s in enumerate(segs):
            k = int(round(s.nominal_count))
            if k < 1:
                raise ValueError(
                    f"rate_segments[{i}]（{s.label or '未命名'}）名义请求数 "
                    f"{s.nominal_count:.2f} 不足 1；请增大 duration_s 或 qps"
                    f"（jittered 下每段至少要有 1 个请求）"
                )
            total += k
        return total
    total = int(round(sum(s.nominal_count for s in segs)))
    if total < 1:
        raise ValueError(
            "rate_segments 的名义请求总数不足 1；请增大 duration_s 或 qps"
        )
    return total


def _deterministic_group(req_id: str, groups: int) -> int:
    """用 req_id 的哈希做分组，保证同一 spec 下分组稳定且与采样顺序无关。"""
    digest = hashlib.blake2b(req_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % groups


def build_requests(spec: WorkloadSpec) -> list[Request]:
    """按 spec 生成请求序列（确定性：同 spec 同输出）。"""
    if spec.num_requests <= 0:
        raise ValueError("num_requests 必须为正整数")
    if spec.prompt_len_min > spec.prompt_len_max:
        raise ValueError("prompt_len_min 不能大于 prompt_len_max")
    if spec.gen_len_min > spec.gen_len_max:
        raise ValueError("gen_len_min 不能大于 gen_len_max")
    if spec.gen_len_min < 1:
        raise ValueError("gen_len_min 至少为 1，否则请求没有输出 token，TPOT 无定义")
    if spec.shared_prefix_len > spec.prompt_len_min:
        raise ValueError(
            "shared_prefix_len 不能超过最小 prompt 长度，否则请求将无可计算部分"
        )

    rng = random.Random(spec.seed)
    arrivals = _sample_arrivals(spec, rng)
    if len(arrivals) != spec.num_requests:
        # 不允许"截断"或"补齐"来凑数：那会让负载轨迹少跑一段（或多跑一段）
        # 而产物里完全看不出来。数量不对就是生成器的缺陷，必须修。
        raise RuntimeError(
            f"到达时刻数量 {len(arrivals)} 与 num_requests {spec.num_requests} 不一致"
            f"（arrival={spec.arrival!r}）；这是到达模型的实现缺陷，不要靠截断/补齐绕过"
        )

    requests: list[Request] = []
    for idx in range(spec.num_requests):
        req_id = f"r{idx:05d}"
        prompt_len = rng.randint(spec.prompt_len_min, spec.prompt_len_max)
        gen_len = rng.randint(spec.gen_len_min, spec.gen_len_max)

        prefix_group = ""
        shared_prefix_len = 0
        if spec.prefix_groups > 0 and spec.shared_prefix_len > 0:
            group_idx = _deterministic_group(req_id, spec.prefix_groups)
            prefix_group = f"g{group_idx:03d}"
            shared_prefix_len = spec.shared_prefix_len

        requests.append(
            Request(
                req_id=req_id,
                prompt_len=prompt_len,
                gen_len=gen_len,
                arrival_ms=arrivals[idx],
                shared_prefix_len=shared_prefix_len,
                prefix_group=prefix_group,
            )
        )
    return requests


def _sample_arrivals(spec: WorkloadSpec, rng: random.Random) -> list[float]:
    """生成到达时刻序列（ms）。"""
    n = spec.num_requests

    if spec.arrival in ("closed", "closed_jittered"):
        # 闭模型：前 concurrency 个立即到达，其余按需由执行器补位，
        # 因此这里把 arrival 全部记为 0，由执行器在补位时分配真实时刻。
        #
        # `closed_jittered`（B1 草案 §5 / Q4）：**同样**走闭模型语义（在途受
        # `concurrency` 硬上限），区别在执行器侧 —— 补位时刻 = `step_start + 抖动`
        # （`workload.closure_jitter_ms`）⇒ 批边界随调度改变、跨重复有差异。
        # ⚠️ 本函数**不**替执行器算抖动：真实时刻只能在执行时确定（依赖步序列）。
        return [0.0] * n

    if spec.arrival == "poisson":
        interval_ms = 1000.0 / max(spec.request_rate_qps, 1e-9)
        t = 0.0
        out: list[float] = []
        for _ in range(n):
            t += rng.expovariate(1.0 / interval_ms)
            out.append(t)
        return out

    if spec.arrival == "burst":
        # 突发模型：分段突发，每 burst_size 个请求几乎同时到达，段间留间隔。
        burst_size = max(1, spec.concurrency)
        gap_ms = burst_size * 1000.0 / max(spec.request_rate_qps, 1e-9)
        out = []
        for i in range(n):
            out.append((i // burst_size) * gap_ms)
        return out

    if spec.arrival == "piecewise":
        return _sample_piecewise(spec, rng)

    raise ValueError(f"未知到达模型：{spec.arrival!r}")


def _sample_piecewise(spec: WorkloadSpec, rng: random.Random) -> list[float]:
    """按**分段常量速率** λ(t) 生成到达时刻（ms）。

    段边界严格按墙钟对齐：第 i 段占用 ``[Σ_{j<i} d_j, Σ_{j≤i} d_j)``。
    这是"负载水平轨迹"的定义本身——若边界跟着实际到达漂，
    "何时升压/降压"就变成一个随机的量，跨 run 无法对照。

    两种分布都保证总数恰为 ``num_requests``，所以调用方不会遇到
    "生成数对不上请求数"的静默截断。
    """
    segs = spec.rate_segments
    n = spec.num_requests

    bounds: list[float] = []
    t = 0.0
    for s in segs:
        t += s.duration_s * 1000.0
        bounds.append(t)

    if spec.rate_noise == "jittered":
        out: list[float] = []
        start = 0.0
        for i, s in enumerate(segs):
            k = int(round(s.nominal_count))
            span = bounds[i] - start
            stride = span / k
            # 分层抽样：第 j 个落在该段第 j 个等分格内的随机位置。
            # 既让本段计数**精确**等于名义值，又不落在周期网格上——
            # 严格等间隔会与执行器的步长网格相位锁定（P1S3D E7 的教训）。
            out.extend(start + (j + rng.random()) * stride for j in range(k))
            start = bounds[i]
        out.sort()
        return out

    # poisson：总数固定，位置按 λ(t)/Λ(T) 做 i.i.d. 抽样，等价于
    # "给定 [0,T] 内恰有 N 个事件"的条件泊松过程 ⇒ 各段**实际**计数按多项分布
    # 波动。段内请求数少时相对噪声可观，事后必须从 requests.jsonl 读实际到达，
    # 不能拿名义 qps 当自变量。
    weights = [s.nominal_count for s in segs]
    total_w = float(sum(weights))
    out = []
    for _ in range(n):
        u = rng.random() * total_w
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if u < acc:
                start = bounds[i] - segs[i].duration_s * 1000.0
                out.append(start + (u - (acc - w)) / segs[i].qps * 1000.0)
                break
        else:
            # 浮点兜底：落在最后一段末尾。正常不会触发。
            out.append(bounds[-1])
    out.sort()
    return out


def summarize_workload(
    requests: list[Request], spec: WorkloadSpec | None = None
) -> dict[str, Any]:
    """给 meta.json 用的工作负载指纹，便于事后判断两次 run 是否可比。

    ``spec`` 可选：传入时会额外记录 ``concurrency`` / ``seed`` / ``arrival``。
    这三个量无法从请求列表反推（闭环下 ``arrival_ms`` 全为 0，是执行器后来
    才填的），而它们恰恰是可复现性元数据的必需字段（规范第 8 节）
    ——不传 spec 就只记录能从请求本身得到的量，缺的字段由上层标为缺失。
    """
    if not requests:
        return {"num_requests": 0}
    prompt = [r.prompt_len for r in requests]
    gen = [r.gen_len for r in requests]
    reuse = [r.shared_prefix_len for r in requests]
    out: dict[str, Any] = {
        "num_requests": len(requests),
        "prompt_len_mean": round(sum(prompt) / len(prompt), 1),
        "prompt_len_min": min(prompt),
        "prompt_len_max": max(prompt),
        "gen_len_mean": round(sum(gen) / len(gen), 1),
        "gen_len_min": min(gen),
        "gen_len_max": max(gen),
        "total_output_tokens": sum(gen),
        "total_prompt_tokens": sum(prompt),
        "reusable_prompt_tokens": sum(reuse),
        "distinct_prefix_groups": len({r.prefix_group for r in requests if r.prefix_group}),
    }
    if spec is not None:
        out.update(
            {
                "spec_name": spec.name,
                "concurrency": spec.concurrency,
                "arrival": spec.arrival,
                "seed": spec.seed,
                "shared_prefix_len": spec.shared_prefix_len,
            }
        )
        # 需求侧前缀复用比例。单独给出便于与实测 cache_hit_ratio 对照：
        # 两者差距大，说明缓存没留住本可复用的前缀（策略问题），
        # 而不是负载本身没有复用空间（需求问题）。
        total_prompt = sum(prompt)
        out["prefix_reuse_ratio"] = (
            round(sum(min(r.shared_prefix_len, r.prompt_len) for r in requests) / total_prompt, 4)
            if total_prompt
            else None
        )
        if spec.arrival == "piecewise" and spec.rate_segments:
            out["arrival_segments"] = _summarize_piecewise(requests, spec)
    return out


def _summarize_piecewise(
    requests: list[Request], spec: WorkloadSpec
) -> list[dict[str, Any]]:
    """逐段给出**名义**与**实际**的到达速率与请求数。

    两者必须同时给：名义值是自变量（实验设计时的意图），实际值是真正落到
    设备上的负载。``poisson`` 模式下两者会有可观差距（尤其段内请求数少时），
    只报名义值等于假定"想要多少就来多少"。
    """
    bounds: list[float] = []
    t = 0.0
    for s in spec.rate_segments:
        t += s.duration_s * 1000.0
        bounds.append(t)

    arrivals = sorted(r.arrival_ms for r in requests)
    out: list[dict[str, Any]] = []
    start = 0.0
    for i, s in enumerate(spec.rate_segments):
        end = bounds[i]
        realized = sum(1 for a in arrivals if start <= a < end)
        dur_s = (end - start) / 1000.0
        out.append(
            {
                "label": s.label or f"seg{i}",
                "start_ms": round(start, 3),
                "end_ms": round(end, 3),
                "nominal_qps": round(s.qps, 6),
                "nominal_count": round(s.nominal_count, 3),
                "realized_count": realized,
                "realized_qps": round(realized / dur_s, 6) if dur_s > 0 else None,
            }
        )
        start = end
    return out
