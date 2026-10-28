"""指标定义与聚合。

全仓库唯一的指标口径（与 README 第 4 节一致）：
    TTFT        Time To First Token       首 token 时延（ms）
    TPOT        Time Per Output Token     decode 每 token 间隔（ms/token）
    stall       单请求 token 间隔的 P99    用户实际感受到的卡顿（ms）
    E2E         端到端时延（ms）
    Throughput  输出 token 速率（tok/s）
    Goodput     **本仓库不再使用这个含混的名字**。它同时被读成"达标比例"与
                "达标吞吐"，而两者是不同的数，于是同一张表里的 goodput 列
                谁也不知道在说什么。已拆成三个明确字段：
                  slo_satisfaction_rate  达标请求数 / **全部提交**请求数
                  slo_goodput_req_s      达标请求数 / 测量秒数
                  slo_goodput_tok_s      达标请求的输出 token / 测量秒数

两条曾被搞混、必须分开报告的口径：

- **SLO 满足率**（``slo_satisfaction_rate``）= 达标请求数 / **全部提交**请求数。
  分母是全部提交，不是成功数。用成功数做分母会把失败请求从账上抹掉，
  于是一批大量超时/失败的 run 反而显示出很高的"达标率"。
- **达标吞吐**（``slo_goodput_req_s`` / ``slo_goodput_tok_s``）= 达标请求数（或其
  输出 token 数）/ 测量秒数。比例高但请求少，与比例低但请求多，是两种处境，
  所以比例与吞吐必须成对给出。

另需单独报告失败率、拒绝率与未完成数——它们不是"不好的吞吐"，
而是"服务没把活干完"，成因与对策都与时延优化不同。

纪律：
- 一律报分位数（P50/P90/P95/P99）并**附带样本量 n**，不把单次运行的均值当结论。
- **不可测的值保持 ``None``，禁止填 0**。0 是"测出来是 0"，缺失是"没测"，语义相反。
- 离散度用稳健 CV = 1.4826 * MAD / median，普通标准差对 outlier 无抵抗力。
- 只依赖标准库，保证在任何环境（含无 NPU 的 CI）都能算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

MAD_SCALE = 1.4826  # 正态假设下 MAD -> sigma 的一致性因子

#: 请求结局的四种取值。刻意分开而不是合成一个 bool——
#: 「服务算错了」「被限流拒了」「预算不够没跑完」的处置方式完全不同，
#: 混成一类之后失败率只剩一个数，反而查不出该修什么。
OUTCOME_SUCCESS = "success"      #: 正常完成
OUTCOME_FAILED = "failed"        #: 服务端出错（HTTP 5xx、流中断、解析失败）
OUTCOME_REJECTED = "rejected"    #: 被拒绝（限流 429、队列满、准入失败）
OUTCOME_UNFINISHED = "unfinished"  #: 未在预算/时限内完成

#: 只有这一种结局可以进入时延分布。未完成的请求没有可用的时延数值，
#: 把它按"最大时延"塞进分布会污染分位数。
OUTCOME_TERMINAL = (OUTCOME_SUCCESS, OUTCOME_FAILED, OUTCOME_REJECTED)


# --------------------------------------------------------------------------- #
# 基础统计
# --------------------------------------------------------------------------- #

def percentile(values: Sequence[float], q: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认行为一致）。

    q 取 0~100。空序列抛错而不是返回 nan —— 静默的 nan 会污染整张汇总表。
    """
    if not values:
        raise ValueError("percentile() 收到空序列")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q 必须在 [0, 100]，收到 {q}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def median(values: Sequence[float]) -> float:
    return percentile(values, 50.0)


def mad(values: Sequence[float]) -> float:
    """中位数绝对偏差。"""
    if not values:
        raise ValueError("mad() 收到空序列")
    med = median(values)
    return median([abs(v - med) for v in values])


def robust_cv(values: Sequence[float]) -> float:
    """稳健变异系数 = 1.4826 * MAD / median。

    median 为 0 时返回 0.0（对时延类指标不会发生；对差值类指标需要调用方自行判断）。
    """
    if not values:
        raise ValueError("robust_cv() 收到空序列")
    med = median(values)
    if med == 0:
        return 0.0
    return MAD_SCALE * mad(values) / abs(med)


def coeff_variation(values: Sequence[float]) -> float:
    """普通 CV，仅用于对照展示，不作为稳定性判据。"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (var ** 0.5) / abs(mean)


def _round_opt(value: float | None, digits: int = 4) -> float | None:
    """四舍五入但保留 ``None``。

    ``round(None, 4)`` 会抛 TypeError；而把 None 顺手写成 0 更糟——
    0 会被读成"测出来是 0"，而 None 表示"这次没这个数"。
    """
    return None if value is None else round(value, digits)


def _collect(values: Iterable[float | None]) -> list[float]:
    """收拢一串可能缺失的数值，**丢掉 ``None``** 而不是替换成 0。

    替换成 0 会把缺失值混进分布并拉低分位数——而且它读起来像一个正常的快值，
    不会被追查。缺失必须在进入统计前就被排除，并由 *_rate 一类的字段另行说明
    （例如 ``unfinished_rate > 0`` 就是在提示"这份分布剔掉了最慢的那批请求"）。
    """
    return [v for v in values if v is not None]


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #

@dataclass
class RequestMetrics:
    """单请求的测量结果。执行器负责填充，本模块只负责聚合。

    时间戳一律用 ``float | None``，``None`` 表示**该时刻从未发生**
    （请求没到达 / 没被受理 / 没被调度 / 没产出首 token）。
    刻意**不用 0 当哨兵**：0 既是"run 的第一步"这一合法时刻，也是"确实等了 0ms"。
    一个字段同时表示"没发生"和"发生在 0"，读的人无法分辨，而两者导出的动作
    完全相反——前者要去查"为什么没被调度"，后者什么都不用做。
    这正是本仓库一贯的纪律：不可测的值保持 ``None``，禁止填 0。
    """

    req_id: str
    prompt_len: int
    gen_len: int
    #: 首 token 时延。未产出首 token 时为 ``None``——写成 0 会被读成"飞快"。
    ttft_ms: float | None
    tpot_ms: float
    #: 该请求自身 token 间隔的 P99，即「卡顿」程度
    tpot_p99_ms: float = 0.0
    cached_tokens: int = 0
    # ---- 时间戳（相对 run 起点的绝对 ms；``None`` = 未发生）------------------
    # 为什么分开记六个时刻，而不是只留 arrival / done：
    # TTFT 是「等待 + 计算」的合成量。只报一个总数，看的人无法判断
    # 该去优化排队还是该去优化算子——而这两件事的对策完全不同。
    #     TTFT = admission_ms + queue_ms + prefill_exec_ms
    #: 工作负载日程里**原定**到达的时刻。
    #: **不得被执行步的时间覆盖**：一旦改写，"因单步粒度而多等的时间"就从
    #: TTFT 里消失了，突发负载下的 TTFT 会被系统性低估。
    scheduled_arrival_ms: float | None = None
    #: 请求实际到达系统的时刻。开放环 = 日程时刻；闭环 = 客户端腾出容量的时刻。
    #: 账目自查（E2E == 完成 − 到达）的参考点。
    arrival_ms: float | None = None
    #: 被服务端受理、进入等待队列的时刻。与 arrival 的差是**受理延迟**
    #: （真实系统里是网络与连接获取；模拟器里是"到下一步才开始处理"的步粒度）。
    enqueue_ms: float | None = None
    #: 首次被排进一个 batch 的时刻。与 enqueue 的差才是**真正的排队等待**。
    first_schedule_ms: float | None = None
    #: 首个输出 token 产生的时刻（应等于 arrival + ttft，单独记录便于交叉核对）。
    first_token_ms: float | None = None
    #: 请求完成时刻。
    done_ms: float | None = None
    #: 结局：``success`` / ``failed`` / ``rejected`` / ``unfinished``。
    #: 刻意不用一个 bool：那会把「服务出错」「被限流拒绝」「预算不足没跑完」
    #: 混成一类，而三者的处置方式完全不同（修 bug / 加容量 / 加预算）。
    #: 混类之后失败率只剩一个数，反而查不出问题。
    outcome: str = OUTCOME_SUCCESS

    @property
    def success(self) -> bool:
        """是否成功完成。只有这一种结局可以计入时延分布。"""
        return self.outcome == OUTCOME_SUCCESS

    @staticmethod
    def _span(start: float | None, end: float | None) -> float | None:
        """两个时刻之间的时长。任一缺失则整段**不可计算**（``None``）。

        不返回 0：把"没发生"报成"耗时 0ms"会让最慢的请求看起来最快，
        而且这个数字读起来完全正常，因此不会被追查。
        """
        if start is None or end is None:
            return None
        return max(0.0, end - start)

    @property
    def admission_ms(self) -> float | None:
        """受理延迟：到达 → 被服务端受理。"""
        return self._span(self.arrival_ms, self.enqueue_ms)

    @property
    def queue_ms(self) -> float | None:
        """排队等待：被受理 → 首次被调度。

        这才是「调度慢」的度量。注意它**不是** ``enqueue - arrival``：
        受理延迟与排队等待的成因不同（前者是链路/步粒度，后者是容量不足），
        合并成一个数会让"加容量"与"修链路"两种对策失去依据。
        """
        return self._span(self.enqueue_ms, self.first_schedule_ms)

    @property
    def prefill_exec_ms(self) -> float | None:
        """prefill 计算耗时：首次被调度 → 首 token。"""
        return self._span(self.first_schedule_ms, self.first_token_ms)

    @property
    def e2e_ms(self) -> float | None:
        """E2E = TTFT + TPOT * (gen_len - 1)。

        gen_len == 1 时退化为 TTFT，不加额外项。
        未产出首 token（未完成）时返回 ``None``——E2E 不可测，而不是 0。
        """
        if self.ttft_ms is None:
            return None
        steps = max(0, self.gen_len - 1)
        return self.ttft_ms + self.tpot_ms * steps

    def accounting_error_ms(self) -> float | None:
        """E2E 与「完成时刻 - 到达时刻」的差。

        真实后端记录 ``done_ms`` 后可以用它自查计时口径是否自洽；
        差值远大于一步耗时，说明 TTFT/TPOT 的采样点算错了。

        参考点是 **到达时刻**而不是排队时刻：只有在开放环下两者才相等，
        闭环下请求在获得容量时才产生，用排队时刻做参考点会算出一个恒定偏差。

        未完成 / 未到达的请求返回 ``None``（无从自查），不是 0。
        """
        e2e = self.e2e_ms
        if e2e is None or self.done_ms is None or self.arrival_ms is None:
            return None
        return abs((self.done_ms - self.arrival_ms) - e2e)

    def describe(self) -> dict[str, Any]:
        return {
            "req_id": self.req_id,
            "prompt_len": self.prompt_len,
            "gen_len": self.gen_len,
            "ttft_ms": _round_opt(self.ttft_ms),
            "tpot_ms": round(self.tpot_ms, 4),
            "tpot_p99_ms": round(self.tpot_p99_ms, 4),
            "e2e_ms": _round_opt(self.e2e_ms),
            "outcome": self.outcome,
            "success": self.success,
            "scheduled_arrival_ms": _round_opt(self.scheduled_arrival_ms),
            "arrival_ms": _round_opt(self.arrival_ms),
            "enqueue_ms": _round_opt(self.enqueue_ms),
            "first_schedule_ms": _round_opt(self.first_schedule_ms),
            "first_token_ms": _round_opt(self.first_token_ms),
            "done_ms": _round_opt(self.done_ms),
            # 三段分解：三者之和应等于 TTFT（账目自查会校验）。
            "admission_ms": _round_opt(self.admission_ms),
            "queue_ms": _round_opt(self.queue_ms),
            "prefill_exec_ms": _round_opt(self.prefill_exec_ms),
            "cached_tokens": self.cached_tokens,
        }


@dataclass(frozen=True)
class SLO:
    """服务等级目标。用于「SLO 满足率」与「达标吞吐」的判定。"""

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None

    def satisfied_by(self, m: RequestMetrics) -> bool:
        """该请求是否达标。

        未成功完成的请求**一律不算达标**——一个失败或没跑完的请求不可能
        满足时延 SLO。这一条加上"分母取全部提交数"，才使达标率不可被
        失败请求稀释。

        时延字段为 ``None``（没有可测值）时同样不算达标：**不可测 ≠ 达标**。
        否则一个"连首 token 都没产出来"的请求会因为"没超过阈值"而被算作达标。
        """
        if not m.success:
            return False
        if self.ttft_ms is not None:
            if m.ttft_ms is None or m.ttft_ms > self.ttft_ms:
                return False
        if self.tpot_ms is not None and m.tpot_ms > self.tpot_ms:
            return False
        if self.e2e_ms is not None:
            e2e = m.e2e_ms
            if e2e is None or e2e > self.e2e_ms:
                return False
        return True

    def describe(self) -> dict[str, Any]:
        return {"ttft_ms": self.ttft_ms, "tpot_ms": self.tpot_ms, "e2e_ms": self.e2e_ms}


@dataclass
class RunSummary:
    """一次 run 的聚合结果。可直接序列化进 summary.csv / meta.json。"""

    run_id: str
    workload: str
    config: str
    n_requests: int
    n_success: int
    ttft: dict[str, float] = field(default_factory=dict)
    tpot: dict[str, float] = field(default_factory=dict)
    e2e: dict[str, float] = field(default_factory=dict)
    # TTFT 的三段分解（三者之和 ≈ TTFT）。分开报的理由：这三段的成因与对策不同——
    # 受理延迟是链路/步粒度，排队是容量不足，prefill 执行是算力与分块策略。
    # 合成一个 TTFT 之后，"该加容量还是该调 chunk"就失去了依据。
    admission: dict[str, float] = field(default_factory=dict)
    queue: dict[str, float] = field(default_factory=dict)
    prefill: dict[str, float] = field(default_factory=dict)
    # stall = 每请求自身 token 间隔 P99 的分布，即「卡顿」的分布。
    # 为什么必须单独统计：tpot 是每请求的均值，长 prompt 阻塞造成的一次
    # 数秒级停顿会被 gen_len 除开、在均值里几乎看不见，而用户感受到的正是它。
    stall: dict[str, float] = field(default_factory=dict)
    output_tokens: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    #: 测量窗口长度（ms）。**只包含正式测量段**，不含初始化与预热——
    #: 详见 run_benchmark 的计时边界说明。吞吐类指标全部以它作分母。
    wall_ms: float = 0.0
    throughput_tok_s: float = 0.0
    # ---- 结局账目（分母统一为「全部提交请求数」）----
    n_failed: int = 0
    n_rejected: int = 0
    n_unfinished: int = 0
    # ---- SLO 口径：三个数各有用途，不可互相替代 ----
    # 三者都是 ``None`` 表示**本次 run 没有配置 SLO**，因此无法计算——
    # 这与"算了，结果是 0"是两回事：0 会被读成"没有任何请求达标"，
    # 于是一个没配 SLO 的 run 看起来像一次彻底失败。不可计算就写 None。
    #: 达标请求数 / **全部提交**请求数。分母含失败与未完成，因此不可被稀释。
    slo_satisfaction_rate: float | None = None
    #: 达标请求数 / 测量秒数（req/s）。
    slo_goodput_req_s: float | None = None
    #: 达标请求的输出 token / 测量秒数（tok/s）。论文中的 "goodput" 通常指这个。
    slo_goodput_tok_s: float | None = None
    #: 达标请求数本身。留一个原始计数，便于下游自行组合并核对上面两个率。
    n_slo_satisfied: int = 0
    stability: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_submitted(self) -> int:
        """全部提交请求数——一切比率的分母。"""
        return self.n_requests

    @property
    def measured_seconds(self) -> float:
        """测量窗口的秒数。吞吐类的分母。"""
        return self.wall_ms / 1000.0

    def _rate(self, count: int) -> float:
        return count / self.n_requests if self.n_requests else 0.0

    @property
    def failure_rate(self) -> float:
        """服务端出错占比。"""
        return self._rate(self.n_failed)

    @property
    def rejection_rate(self) -> float:
        """被拒绝占比（限流、队列满、准入失败）。"""
        return self._rate(self.n_rejected)

    @property
    def unfinished_rate(self) -> float:
        """未完成占比（预算/时限不足）。**这个数大于 0 时，时延分布不可信**：
        未完成的请求往往正是最慢的那些，把它们从分布里剔掉会低估尾延迟。"""
        return self._rate(self.n_unfinished)

    @property
    def cache_hit_rate(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens

    def describe(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workload": self.workload,
            "config": self.config,
            "n_requests": self.n_requests,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "n_rejected": self.n_rejected,
            "n_unfinished": self.n_unfinished,
            "n_slo_satisfied": self.n_slo_satisfied,
            "failure_rate": round(self.failure_rate, 4),
            "rejection_rate": round(self.rejection_rate, 4),
            "unfinished_rate": round(self.unfinished_rate, 4),
            "slo_satisfaction_rate": _round_opt(self.slo_satisfaction_rate),
            "slo_goodput_req_s": _round_opt(self.slo_goodput_req_s, 4),
            "slo_goodput_tok_s": _round_opt(self.slo_goodput_tok_s, 3),
            "ttft": self.ttft,
            "tpot": self.tpot,
            "stall": self.stall,
            "e2e": self.e2e,
            "admission": self.admission,
            "queue": self.queue,
            "prefill": self.prefill,
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "wall_ms": round(self.wall_ms, 3),
            "throughput_tok_s": round(self.throughput_tok_s, 3),
            "stability": self.stability,
            "extra": self.extra,
        }

    def flat_row(self) -> dict[str, Any]:
        """拍平成一行，供 summary.csv 使用。"""
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "workload": self.workload,
            "config": self.config,
            "n_requests": self.n_requests,
            "n_success": self.n_success,
            "n_failed": self.n_failed,
            "n_rejected": self.n_rejected,
            "n_unfinished": self.n_unfinished,
            "failure_rate": round(self.failure_rate, 4),
            "rejection_rate": round(self.rejection_rate, 4),
            "unfinished_rate": round(self.unfinished_rate, 4),
            "n_slo_satisfied": self.n_slo_satisfied,
            "slo_satisfaction_rate": _round_opt(self.slo_satisfaction_rate),
            "slo_goodput_req_s": _round_opt(self.slo_goodput_req_s),
            "slo_goodput_tok_s": _round_opt(self.slo_goodput_tok_s, 3),
            "output_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "wall_ms": round(self.wall_ms, 3),
            "throughput_tok_s": round(self.throughput_tok_s, 3),
        }
        for prefix, block in (
            ("ttft", self.ttft),
            ("tpot", self.tpot),
            ("stall", self.stall),
            ("e2e", self.e2e),
            ("admission", self.admission),
            ("queue", self.queue),
            ("prefill", self.prefill),
            ("cv", self.stability),
        ):
            for key, value in block.items():
                row[f"{prefix}_{key}"] = value
        return row


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #

def _dist(values: Sequence[float]) -> dict[str, Any]:
    """分位数口径（规范第 7 节）。

    ``median`` 与 ``p50`` 数值上恒等，这里刻意都输出：
    规范正文用的是 "median"，论文表格与工程日志用的是 "P50"，
    两张表混排时读者最容易被"到底是不是同一个数"绊住。
    显式并列能让口径自证，代价只是一个冗余字段。

    ``n`` 必须一起输出：P99 的可靠性与样本量强相关，十几个样本的 P99
    本质上就是最大值。只给分位不给 n，读者无法判断该不该信它。

    空序列返回 ``{}`` 而不是抛错——一次"全部请求都失败"的 run 仍然要出一行
    汇总。抛异常会让那一行整个消失，而汇总表里少一行看起来像"这个配置还没跑"，
    恰恰把最该被看见的失败藏了起来。

    不输出 std：普通标准差对 outlier 毫无抵抗力，本仓库一律用
    ``robust_cv``（见 stability 块）。
    """
    if not values:
        return {}
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 4),
        "median": round(percentile(values, 50), 4),
        "p90": round(percentile(values, 90), 4),
        "p95": round(percentile(values, 95), 4),
        "p99": round(percentile(values, 99), 4),
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarize(
    metrics: Iterable[RequestMetrics],
    *,
    run_id: str,
    workload: str,
    config: str,
    wall_ms: float,
    slo: SLO | None = None,
    extra: dict[str, Any] | None = None,
) -> RunSummary:
    """把逐请求指标聚合成一次 run 的汇总。

    ``wall_ms`` 必须由调用方给出**正式测量窗口**的长度——不能用「最后请求完成
    时刻」代替，否则闭环下的补位等待会被算进吞吐；更不能把初始化与预热包进来，
    否则吞吐会被摊薄（详见 run_benchmark 的计时边界说明）。

    「全部请求都失败」不再抛错：那种 run 恰恰最需要出一行汇总。抛异常会让它
    直接从表里消失，而表里少一行看起来像"这个配置还没跑过"。
    """
    samples = list(metrics)
    if not samples:
        raise ValueError("summarize() 收到空的 metrics 序列")

    n_submitted = len(samples)
    ok = [m for m in samples if m.success]
    n_failed = sum(1 for m in samples if m.outcome == OUTCOME_FAILED)
    n_rejected = sum(1 for m in samples if m.outcome == OUTCOME_REJECTED)
    n_unfinished = sum(1 for m in samples if m.outcome == OUTCOME_UNFINISHED)

    # 时延分布只统计**成功**请求：失败与被拒的请求没有可用的时延数值，
    # 未完成的更是"跑都没跑完"。把它们按最大值填进分布会污染分位数。
    # 代价是尾延迟被低估——所以 unfinished_rate 必须与分布一起看：
    # 它大于 0 就是在提示"这份分布剔掉了最慢的那批请求"。
    #
    # 各段还要再过滤一次 None：一段"从未发生"的时长是不可计算的，
    # 混进分布（哪怕按 0 计）都会凭空制造出一批"极快"的样本。
    ttft = _collect(m.ttft_ms for m in ok)
    tpot = [m.tpot_ms for m in ok]
    e2e = _collect(m.e2e_ms for m in ok)
    admission = _collect(m.admission_ms for m in ok)
    queue = _collect(m.queue_ms for m in ok)
    prefill_exec = _collect(m.prefill_exec_ms for m in ok)
    # 只有记录过 token 间隔明细的执行器才有 stall 数据；没有就输出空字典，
    # 而不是一串 0——0 会被读成「完全没有卡顿」。
    stall_values = [m.tpot_p99_ms for m in ok if m.tpot_p99_ms > 0]

    output_tokens = sum(m.gen_len for m in ok)
    prompt_tokens = sum(m.prompt_len for m in ok)
    cached_tokens = sum(m.cached_tokens for m in ok)

    seconds = wall_ms / 1000.0
    throughput = output_tokens / seconds if seconds > 0 else 0.0

    # ---- SLO 口径：分母一律是**全部提交请求数**，不是成功数 ----
    # 用成功数做分母会把失败与未完成的请求从账上抹掉，于是一批大量超时/失败的
    # run 反而显示出很高的"达标比例"——这正是被高估的来源。
    if slo is None:
        # 没配 SLO：这三个量**不可计算**，写 None。
        # 写 0 会被读成"一个都没达标"，把"没配"伪装成"全失败"。
        n_satisfied = 0
        satisfaction_rate: float | None = None
        goodput_req_s: float | None = None
        goodput_tok_s: float | None = None
    else:
        satisfied = [m for m in ok if slo.satisfied_by(m)]
        n_satisfied = len(satisfied)
        satisfaction_rate = n_satisfied / n_submitted
        goodput_req_s = n_satisfied / seconds if seconds > 0 else 0.0
        goodput_tok_s = (
            sum(m.gen_len for m in satisfied) / seconds if seconds > 0 else 0.0
        )

    stability: dict[str, float] = {}
    if ttft:
        stability = {
            "ttft_robust_cv": round(robust_cv(ttft), 4),
            "tpot_robust_cv": round(robust_cv(tpot), 4),
            "ttft_cv": round(coeff_variation(ttft), 4),
            "tpot_cv": round(coeff_variation(tpot), 4),
            "ttft_mad": round(mad(ttft), 4),
            "tpot_mad": round(mad(tpot), 4),
        }

    return RunSummary(
        run_id=run_id,
        workload=workload,
        config=config,
        n_requests=n_submitted,
        n_success=len(ok),
        n_failed=n_failed,
        n_rejected=n_rejected,
        n_unfinished=n_unfinished,
        n_slo_satisfied=n_satisfied,
        slo_satisfaction_rate=satisfaction_rate,
        slo_goodput_req_s=goodput_req_s,
        slo_goodput_tok_s=goodput_tok_s,
        ttft=_dist(ttft),
        tpot=_dist(tpot),
        e2e=_dist(e2e),
        admission=_dist(admission),
        queue=_dist(queue),
        prefill=_dist(prefill_exec),
        stall=_dist(stall_values),
        output_tokens=output_tokens,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        wall_ms=wall_ms,
        throughput_tok_s=throughput,
        stability=stability,
        extra=extra or {},
    )


def compare(baseline: RunSummary, candidate: RunSummary) -> dict[str, Any]:
    """安全地比较两次 run。

    先做可比性检查再算差值——把不可比的数字放上同一张表是最常见的错误来源。
    """
    warnings: list[str] = []
    if baseline.workload != candidate.workload:
        warnings.append(f"工作负载不同：{baseline.workload} vs {candidate.workload}")
    if baseline.n_success != candidate.n_success:
        warnings.append(
            f"成功请求数不同：{baseline.n_success} vs {candidate.n_success}"
        )
    # 未完成请求会把时延分布向"快"的一侧偏：跑不完的往往正是最慢的那些请求，
    # 它们不进分布，于是 P99 看起来变好了。这种"改善"是口径造成的，不是优化。
    for name, summary in (("baseline", baseline), ("candidate", candidate)):
        if summary.n_unfinished > 0:
            warnings.append(
                f"{name} 有 {summary.n_unfinished} 个未完成请求"
                f"（{summary.unfinished_rate:.1%}）：时延分布已剔除这批请求，"
                f"尾延迟会被低估，差值不可直接采信"
            )
    # stall 缺失时明确标注，而不是当成 0 参与比较
    if not baseline.stall or not candidate.stall:
        warnings.append(
            "至少一侧缺少 stall 数据（tpot_p99_ms 全为 0）："
            "无法判断卡顿是否改善，结论需注明该口径缺失"
        )
    if baseline.slo_satisfaction_rate is None or candidate.slo_satisfaction_rate is None:
        warnings.append(
            "至少一侧未配置 SLO：达标率与达标吞吐不可比（已跳过），"
            "不能因为字段为空就当成 0"
        )

    blocks = {
        "ttft": (baseline.ttft, candidate.ttft),
        "tpot": (baseline.tpot, candidate.tpot),
        "stall": (baseline.stall, candidate.stall),
        # TTFT 的分段也能比：排队变好、计算变差时，只看 TTFT 总数会把两者抵消。
        "queue": (baseline.queue, candidate.queue),
        "prefill": (baseline.prefill, candidate.prefill),
    }

    def delta(base_block: dict[str, float], cand_block: dict[str, float]) -> dict[str, float]:
        if not base_block or not cand_block:
            return {}
        return {
            "p50_base": base_block.get("p50", 0.0),
            "p50_cand": cand_block.get("p50", 0.0),
            "p50_delta_pct": _pct(base_block.get("p50", 0.0), cand_block.get("p50", 0.0)),
            "p99_base": base_block.get("p99", 0.0),
            "p99_cand": cand_block.get("p99", 0.0),
            "p99_delta_pct": _pct(base_block.get("p99", 0.0), cand_block.get("p99", 0.0)),
            # P99 的可靠性依赖样本量，一并带出来，避免"小样本 P99"被当结论用。
            "n_base": base_block.get("n", 0),
            "n_cand": cand_block.get("n", 0),
        }

    result: dict[str, Any] = {
        "baseline": baseline.run_id,
        "candidate": candidate.run_id,
        "throughput_delta_pct": _pct(baseline.throughput_tok_s, candidate.throughput_tok_s),
        # 达标口径成对给出：比例说明"用户满意的比例"，吞吐说明"实际服务能力"。
        # 只给一个都会漏信息——比例高但请求少，和比例低但请求多，是两种处境。
        "slo_satisfaction_rate_base": baseline.slo_satisfaction_rate,
        "slo_satisfaction_rate_cand": candidate.slo_satisfaction_rate,
        "slo_goodput_tok_s_base": baseline.slo_goodput_tok_s,
        "slo_goodput_tok_s_cand": candidate.slo_goodput_tok_s,
        "unfinished_rate_base": round(baseline.unfinished_rate, 4),
        "unfinished_rate_cand": round(candidate.unfinished_rate, 4),
        "cache_hit_rate_base": round(baseline.cache_hit_rate, 4),
        "cache_hit_rate_cand": round(candidate.cache_hit_rate, 4),
        "warnings": warnings,
    }
    for key, (base_block, cand_block) in blocks.items():
        result[key] = delta(base_block, cand_block)

    # 收益必须显著大于测量噪声，否则不构成结论。
    noise_keys = ("ttft_robust_cv", "tpot_robust_cv")
    noise = max(
        [baseline.stability.get(k, 0.0) for k in noise_keys]
        + [candidate.stability.get(k, 0.0) for k in noise_keys]
    )
    result["noise_floor_pct"] = round(noise * 100, 3)

    result["verdict"] = _verdict(result, noise)
    return result


def _pct(base: float, cand: float) -> float:
    if base == 0:
        return 0.0
    return round((cand - base) / base * 100.0, 3)


def _verdict(cmp_result: dict[str, Any], noise: float) -> str:
    """把差值翻译成人话，且不掩盖「测不出结论」这一状态。

    判据取 ttft / tpot / stall 三个 P99 的差值。加入 stall 是刻意的：
    只看 tpot（每请求均值）会把「个别请求被长 prefill 卡住数秒」这种
    最需要修的尾延迟判成 inconclusive。
    """
    keys = ("ttft", "tpot", "stall")
    available = [cmp_result[k] for k in keys if cmp_result.get(k)]
    if not available:
        return "inconclusive: 缺少可比较的指标块"

    deltas = [block["p99_delta_pct"] for block in available]
    tolerance = noise * 100.0
    if all(abs(d) <= tolerance for d in deltas):
        return "inconclusive: 变化幅度未超过测量噪声，需增加重复次数"
    improved = [d for d in deltas if d <= -tolerance]
    worsened = [d for d in deltas if d >= tolerance]
    if len(improved) == len(deltas):
        return "positive: 所有观测指标的 P99 均改善且超过噪声"
    if len(worsened) == len(deltas):
        return "negative: 所有观测指标的 P99 均恶化"
    return "mixed: 存在指标间取舍，需按目标函数加权后再判断"
