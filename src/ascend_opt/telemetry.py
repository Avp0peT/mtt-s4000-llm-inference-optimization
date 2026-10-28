"""遥测采集与落盘。

职责：把执行过程中产生的时间序采样写成 jsonl，供事后复盘控制器的决策序列。

边界（见根 AGENTS.md 附录 A3 代码约定）：
本模块**不主动触达硬件**。硬件侧指纹（芯片型号、驱动、CANN 版本、显存、
设备拓扑）由 ``scripts/inspect_env.sh`` 采集并写入
``results/raw/<run_id>/env.json``，本模块只负责读取与转存。

这样做的原因：策略与采集逻辑必须能在无 NPU 的开发机上跑通，
否则离线验证（--dry-run）和 CI 都做不了。
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from ascend_opt.provenance import normalize_availability, parse_mthreads_software

ENV_FILENAME = "env.json"
TELEMETRY_FILENAME = "telemetry.jsonl"

#: 硬件可用性的三态。
#: ``None`` = 没查到，不是"没有"。把"没查"当成"没有"会让补齐环境的责任
#: 被静默推给硬件；反过来把"没有"当成"有"更糟——那会声称测过没测过的东西。
UNKNOWN_AVAILABILITY: None = None

#: 环境指纹里"可用性是谁给的"。
#: ``declared`` = 采集脚本显式写了 ``available``（新格式，导出闸门认它）；
#: ``derived``  = 指纹里没有，本模块在内存里按探测结果推的（旧格式）；
#: ``missing``  = 根本没找到指纹文件；``unreadable`` = 找到了但解析不了。
#:
#: 为什么要记这个：``derived`` 意味着"运行时认为可用、导出时会被拒"——
#: 因为导出闸门只认文件里**显式声明**的可用性。不把这个来源写进 meta，
#: 这个分裂就只能等导出失败时才发现，而且看不出原因。
AVAILABILITY_SOURCE_DECLARED = "declared"
AVAILABILITY_SOURCE_DERIVED = "derived"
AVAILABILITY_SOURCE_MISSING = "missing"
AVAILABILITY_SOURCE_UNREADABLE = "unreadable"


def _derive_availability(payload: dict[str, Any]) -> bool | None:
    """从环境指纹**推断**昇腾硬件是否可用（旧格式的兼容路径）。

    优先读显式字段；其次读规范化后的 ``probes`` 块（与采集脚本用同一套合成规则）；
    最后回落到设备探测结果（``device.npu_smi_available``）。
    三者都没有则返回 ``None``（未知），**不返回 True**——
    默认成 True 会让无 NPU 的机器上跑出的 run 自称"已实测"，
    这是规范第 10 节明令禁止的情形。
    """
    explicit = payload.get("available")
    if explicit is not None:
        return bool(explicit)

    probes = payload.get("probes")
    if isinstance(probes, dict):
        results = [
            p.get("result") for p in probes.values() if isinstance(p, dict)
        ]
        if results:
            return normalize_availability(results)

    device = payload.get("device")
    if isinstance(device, dict) and device.get("npu_smi_available") is not None:
        # 注意这条回落的强度：它只说明 **npu-smi 能跑**，不说明它认出了设备。
        # 旧格式没有更好的证据，所以保留它；新格式一律走上面两条。
        return bool(device["npu_smi_available"])

    return UNKNOWN_AVAILABILITY


def hardware_env_candidates(run_dir: str | Path) -> list[Path]:
    """环境指纹的候选路径，按优先级排列。

    run 目录优先，其次共享位置。共享位置解决了一个真实的时序问题——
    硬件指纹必须在 run 开始**之前**采集（状态不能被 run 影响），
    但 run 目录要到 run 开始之后才创建。

    单独抽成函数是为了让"读指纹"这件事只有一个入口：
    run 期判定与落盘、以及事后核对，三处必须看到**同一份**指纹，
    否则就会出现"判定时可用、落盘时不可用"这种自相矛盾的产物。
    """
    run_dir = Path(run_dir)
    return [run_dir / ENV_FILENAME, run_dir.parent / "_env" / ENV_FILENAME]


def load_hardware_env(candidates: Sequence[str | Path]) -> dict[str, Any]:
    """按候选路径依次查找并读取环境指纹。

    全部缺失时返回带说明的结构而**不抛错**——「环境未知」是可接受的状态，
    但必须显式标注出来，绝不能让一次环境未知的 run 看起来和另一次环境已知的一样可信。

    读到的指纹一律补上 ``available_source``：调用方据此判断可用性是"文件声明的"
    还是"本模块推出来的"。两者的下游命运不同（后者会被导出闸门拒绝），
    必须可区分。
    """
    for path in candidates:
        path = Path(path)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "available": False,
                "available_source": AVAILABILITY_SOURCE_UNREADABLE,
                "source": str(path),
                "reason": f"{path} 解析失败：{exc}",
            }
        declared = "available" in payload
        if not declared:
            payload["available"] = _derive_availability(payload)
        payload["available_source"] = (
            AVAILABILITY_SOURCE_DECLARED if declared else AVAILABILITY_SOURCE_DERIVED
        )
        payload["source"] = str(path)
        return payload

    return {
        # 刻意是 None 而不是 False：找不到指纹文件说明"没查过"，
        # 不说明"这台机器没有 NPU"。把两者混同会让环境采集这件事
        # 看起来已经做完了，从而永远不会被补上。
        "available": UNKNOWN_AVAILABILITY,
        "available_source": AVAILABILITY_SOURCE_MISSING,
        "reason": (
            f"未找到 {ENV_FILENAME}（查找过："
            f"{[str(p) for p in candidates]}）；"
            f"硬件可用性未知，请先运行 scripts/inspect_env.sh"
        ),
    }


@dataclass
class TelemetrySample:
    """一个采样点的状态快照。字段命名与 summary.csv 的列保持可对齐。"""

    step: int
    t_ms: float                      # 相对 run 起点的毫秒数
    queue_depth: int = 0             # 等待中的请求数
    running: int = 0                 # 运行中的请求数
    scheduled_tokens: int = 0        # 本步实际调度的 token 数
    ttft_ms: float | None = None     # 本步内观测到的 TTFT（可为空）
    tpot_ms: float | None = None     # 本步内观测到的 TPOT（可为空）
    throughput_tok_s: float = 0.0
    mem_mb: float | None = None      # 设备显存占用（由执行器传入，本模块不查询）
    util_pct: float | None = None    # 设备利用率（同上）
    #: **实际生效**的分块上限（本步真正交给分块策略的那个数）。
    chunk_size: int | None = None
    #: **实际生效**的单步 token 预算。
    max_batched_tokens: int | None = None
    #: 控制器本步**建议**的分块上限。与 ``chunk_size`` 并排看：
    #: 两者长期不等就说明建议值兑现不了（策略上限、token 预算、容量闸门都可能拦它）。
    #: 只记建议值（或只记生效值）都无法发现这类失效——它不抛错、账目也自洽。
    chunk_size_requested: int | None = None
    #: 控制器本步**建议**的单步 token 预算。
    max_batched_tokens_requested: int | None = None
    #: 本步**实际发放**给 prefill 的 token 数（经策略与容量闸门之后）。
    #: 这是"额度到底动没动"的最终凭据：控制器日志说调大了，若这个数没涨，
    #: 那就是没生效。
    prefill_budget_effective: int | None = None
    note: str = ""

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunContext:
    """一次 run 的身份信息，写进 meta.json。"""

    run_id: str
    workload: str
    config_path: str
    repeat_index: int = 0
    dry_run: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryCollector:
    """按 run 目录收集遥测与元数据。

    落盘时机：``record`` 只进内存缓冲，``flush`` 才写文件。这样即使中途异常，
    也能由外层决定「丢弃这一次」还是「保留已采到的部分」——不要用增量写换来
    半截损坏的 jsonl。
    """

    def __init__(
        self,
        run_dir: str | Path,
        context: RunContext,
        env_candidates: Sequence[str | Path] | None = None,
        hardware_env: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.context = context
        self.samples: list[TelemetrySample] = []
        self._t0 = time.perf_counter()
        if env_candidates is None:
            env_candidates = hardware_env_candidates(self.run_dir)
        self._env_candidates = [Path(p) for p in env_candidates]
        # 已绑定的指纹快照。调用方在 run **开始前**就为判定读了一份，落盘时必须
        # 用同一份——否则"判定时读到 A、落盘时读到 B"会产出一份自相矛盾的产物，
        # 而它看起来完全正常（字段都在，只是来自两个不同的文件）。
        self._bound_hardware_env = hardware_env

    # ---------------- 采集 ----------------

    def record(self, sample: TelemetrySample) -> None:
        self.samples.append(sample)

    def record_dict(self, **kwargs: Any) -> TelemetrySample:
        """便捷入口：自动补齐 step 与相对时间。"""
        kwargs.setdefault("step", len(self.samples))
        kwargs.setdefault("t_ms", self.elapsed_ms)
        sample = TelemetrySample(**kwargs)
        self.samples.append(sample)
        return sample

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    # ---------------- 环境指纹 ----------------

    def software_env(self) -> dict[str, Any]:
        """软件侧指纹（不含硬件信息）。

        ⚠️ 末两项（`mthreads_driver_version` / `musa_version`）**是从 ``hardware_env`` 里
        那段 ``mthreads-gmi`` 原始文本解析出来的**，用现成的
        :func:`~ascend_opt.provenance.parse_mthreads_software`。

        为什么必须解析成字段：驱动版本原先只以**自由文本**形式躺在
        ``hardware_env.device.mthreads_gmi_info`` 里 ⇒ **没法程序化比对**，
        而"这一批和上一批之间驱动换没换"正是跨批次可比性的前提（`RESEARCH_LOG` 的
        E4 已经吃过一次同类亏：戳看着有、其实是别的东西的版本）。
        ⇒ 解析器早就写好了（连"只有工具版本行、没有独立驱动字段时返回 None"这种
        边界都考虑过），**只是此前没人调用它**。

        取不到就是 ``None``：**不猜**，也不写 ``"unknown"``
        —— 后者会被写进指纹，在需要交叉核对时看起来像一次成功的采集。
        """
        env: dict[str, Any] = {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        for mod in ("torch", "torch_npu", "vllm", "transformers", "ascend_opt"):
            env[f"{mod}_version"] = _module_version(mod)

        # `read_hardware_env()` 按设计**不抛错**（缺文件时返回带说明的结构）⇒ 这里不需要
        # try/except 去吞异常；真吞了才会把"指纹坏了"变成"指纹看起来正常"。
        device = (self.read_hardware_env() or {}).get("device") or {}
        parsed = parse_mthreads_software(
            device.get("mthreads_gmi_info") if isinstance(device, dict) else None
        )
        env["mthreads_driver_version"] = parsed["driver_version"]
        env["musa_version"] = parsed["musa_version"]
        return env

    def read_hardware_env(self) -> dict[str, Any]:
        """读取 ``scripts/inspect_env.sh`` 预先写好的硬件指纹。

        按候选路径依次查找，全部缺失时返回带说明的结构而不抛错（见
        :func:`load_hardware_env`）。返回的字典一定带 ``available_source``，
        调用方可以据此区分"文件声明的可用性"与"这里推出来的可用性"。

        构造时传入了 ``hardware_env`` 则原样返回它：run 开始前的判定与落盘
        必须看到同一份指纹，见 :meth:`__init__`。
        """
        if self._bound_hardware_env is not None:
            return self._bound_hardware_env
        return load_hardware_env(self._env_candidates)

    # ---------------- 落盘 ----------------

    def write(
        self,
        summary: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """写出 meta.json / telemetry.jsonl，返回产物路径。

        ``summary`` 落在 ``meta["summary"]``，``extra`` 的键直接并入 meta 顶层
        （用于放 config、controller_trace 这类同级信息）。
        分开是为了让读取端有一个稳定的路径：``meta["summary"]["ttft"]["p99"]``
        永远指向同一次 run 的汇总，而不会被额外字段挤歪。
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        meta_path = self.run_dir / "meta.json"
        telemetry_path = self.run_dir / TELEMETRY_FILENAME

        meta: dict[str, Any] = {
            "context": self.context.describe(),
            "software_env": self.software_env(),
            "hardware_env": self.read_hardware_env(),
            "n_samples": len(self.samples),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if summary is not None:
            meta["summary"] = summary
        if extra:
            collision = set(extra) & set(meta)
            if collision:
                raise ValueError(
                    f"extra 与 meta 顶层字段冲突：{sorted(collision)}；"
                    f"换一个键名，避免覆盖溯源信息"
                )
            meta.update(extra)
        # ⚠️ 落盘前把**非有限浮点**标准化：`inf`/`nan` 会被 Python 的 json 写成
        # `Infinity`/`NaN` —— 那是**非标准 JSON**，严格解析器读我们自己的证据会挂。
        # 实测踩到过：`S14B1-*fixed_slo_weave*/meta.json` 里就有 `Infinity`
        # （来自 `budget_ms = inf`："无活跃 decode ⇒ 预算无上限"）。
        # 统一映射为 `None`；需要区分"无限"与"未测量"的地方，**另立显式布尔字段**
        # （见 `SLOWeaveDecision.describe()` 的 `budget_unbounded`）。
        _write_text_lf(meta_path, json.dumps(_json_safe(meta), ensure_ascii=False, indent=2))

        with telemetry_path.open("w", encoding="utf-8", newline="\n") as fh:
            for sample in self.samples:
                fh.write(json.dumps(_json_safe(sample.describe()), ensure_ascii=False) + "\n")

        return {"meta": meta_path, "telemetry": telemetry_path}


def _json_safe(obj: Any) -> Any:
    """把**非有限浮点**换成 `None`，其余原样（递归）。

    为什么必须做：Python 的 `json.dumps` 默认把 `inf`/`nan` 写成 `Infinity`/`NaN`，
    这是**非标准 JSON** —— 严格解析器（以及很多下游工具）会直接报错。
    而这里落盘的是**证据**：证据必须能被任何标准 JSON 读者打开。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _write_text_lf(path: Path, text: str) -> None:
    """写文本文件并强制 LF 换行（与 benchmark/_write_text_lf 同一约定）。

    为什么不直接 ``Path.write_text``：它在 Windows 上默认 ``newline=None``，
    会把 ``\\n`` 翻成 ``\\r\\n``。这里落盘的是**证据**（meta/telemetry），
    字节即证据：本机产 CRLF、远程 Linux 产 LF，会让跨机器比对、算 md5、
    与入库版本核对时出现与行为无关的假 diff，且不报错、内容照常可解析，
    只有量字节才发现。仓库规范 (.gitattributes: eol=lf) 要求一律 LF，
    故在产生侧写对，而不是依赖入库转换掩盖它。
    """
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _module_version(name: str) -> str | None:
    """尽力获取模块版本；缺失或导入报错都返回 None，不干扰主流程。"""
    try:
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


# --------------------------------------------------------------------------- #
# 读取端：供 collect_metrics.py 与离线复放使用
# --------------------------------------------------------------------------- #

def iter_telemetry(run_dir: str | Path) -> Iterator[TelemetrySample]:
    """逐行读取遥测文件。坏行跳过并计数不在这里处理，由调用方决定。"""
    path = Path(run_dir) / TELEMETRY_FILENAME
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield TelemetrySample(**json.loads(line))


def load_meta(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 meta.json：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_run_id(workload: str, config_stem: str, when: time.struct_time | None = None) -> str:
    """生成统一的 run_id：YYYYMMDD-HHMMSS-<workload>-<config_stem>。"""
    stamp = time.strftime("%Y%m%d-%H%M%S", when or time.localtime())
    safe = lambda s: "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
    return f"{stamp}-{safe(workload)}-{safe(config_stem)}"
