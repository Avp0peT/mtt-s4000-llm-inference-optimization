"""工作负载注册表。

把常用的负载定义集中在这里，用名字引用（配置里写 ``workload: short_chat``），
避免每个实验各自手写长度分布——那会让「同负载对照」这个前提悄悄失效。

每条负载都要写清它代表什么真实场景，否则半年后没人知道该拿它证明什么。

两层内容
--------
1. **扫描轴**（规范第 6 节）：``INPUT_LENGTH_AXIS`` 等四个常量，
   外加由它们生成的 ``spec_*`` 系列负载。这些是规范点名的取值，
   不允许随手改——改了就等于偷偷换了实验的横轴。
2. **项目本地负载**：``short_chat`` / ``long_context_qa`` / ``shared_prefix_4k`` /
   ``mixed_burst``。它们用**区间**而非单点描述长度，用于贴近真实流量的异质性。
   与规范原型的关系：``long_context_qa`` 是 ``long_in_short_out`` 的区间化版本，
   ``mixed_burst`` 是 ``mixed`` 的区间化版本。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator

# 直接以脚本方式运行时（python benchmark/workloads.py），sys.path[0] 是
# benchmark/ 而不是仓库根，src-layout 下的 ascend_opt 就不可见。
# 这里显式补齐，使本模块既能被 import 也能被直接执行。
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.workload import WorkloadSpec  # noqa: E402

# --------------------------------------------------------------------------- #
# 扫描轴（规范第 6 节原文取值）
# --------------------------------------------------------------------------- #

#: 输入长度轴。16384 是本轮的上下文上限，再长需要重新评估显存与分块策略。
INPUT_LENGTH_AXIS: tuple[int, ...] = (128, 512, 2048, 8192, 16384)
#: 输出长度轴。只取两个点：decode 成本随输出长度线性增长，
#: 扫太多点只会拉长实验时间而不增加信息量。
OUTPUT_LENGTH_AXIS: tuple[int, ...] = (128, 512)
#: 并发轴。1 是延迟敏感场景，64 是吞吐压力场景。
CONCURRENCY_AXIS: tuple[int, ...] = (1, 8, 16, 32, 64)
#: 前缀复用比例轴（共享前缀 token / prompt token）。
PREFIX_REUSE_AXIS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)

#: 前缀缓存按组驻留；4 组既能让缓存有明显收益，又不至于一组就全命中。
DEFAULT_PREFIX_GROUPS = 4
#: 扫描用的默认请求数。**四个轴共用同一个值**是刻意选择：
#: 每次扫描只让横轴变、请求数不变，否则两个变量一起动，收益无法归因。
DEFAULT_AXIS_REQUESTS = 64


def build_axis_workload(
    *,
    input_length: int,
    output_length: int,
    concurrency: int,
    prefix_reuse_ratio: float = 0.0,
    num_requests: int = DEFAULT_AXIS_REQUESTS,
    prefix_groups: int = DEFAULT_PREFIX_GROUPS,
    arrival: str = "closed",
    request_rate_qps: float = 1.0,
    seed: int = 0,
    name: str | None = None,
) -> WorkloadSpec:
    """按规范第 6 节的四个轴构造一个**单点**负载。

    与项目本地负载的区别：这里长度区间退化为单点（min == max）。
    单点负载的意义是让横轴是确定的——区间负载在扫长度时，每个点实际上
    是"均值为此的分布"，两个相邻点可能因为采样重叠而看不出差异。

    ``prefix_reuse_ratio`` 转成共享前缀长度的规则：``shared_prefix_len =
    round(ratio * input_length)``。0 表示关闭分组（``prefix_groups=0``），
    这样"没有共享前缀"和"有共享前缀但长度为 0"在元数据里可区分。
    """
    if input_length <= 0 or output_length <= 0:
        raise ValueError("input_length 与 output_length 必须为正")
    if concurrency <= 0:
        raise ValueError("concurrency 必须为正")
    if not 0.0 <= prefix_reuse_ratio <= 1.0:
        raise ValueError(f"prefix_reuse_ratio 必须在 [0, 1]，收到 {prefix_reuse_ratio}")
    if num_requests <= 0:
        raise ValueError("num_requests 必须为正")

    shared = int(round(prefix_reuse_ratio * input_length))
    enabled_groups = prefix_groups if shared > 0 else 0

    return WorkloadSpec(
        name=name or f"axis_in{input_length}_out{output_length}_c{concurrency}_r{prefix_reuse_ratio:.2f}",
        num_requests=num_requests,
        prompt_len_min=input_length,
        prompt_len_max=input_length,
        gen_len_min=output_length,
        gen_len_max=output_length,
        concurrency=concurrency,
        arrival=arrival,  # type: ignore[arg-type]
        request_rate_qps=request_rate_qps,
        prefix_groups=enabled_groups,
        shared_prefix_len=shared,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# 规范第 6 节点名的七类负载分布
#
# 这些条目**由 build_axis_workload 生成**而不是手写常量。
# 手写会产生一种隐蔽的漂移：轴上改了 16384，这里还留着旧的 8192，
# 于是"规范化实验"和"轴常量"对不上却没人发现。
# --------------------------------------------------------------------------- #

def _spec_archetypes() -> dict[str, dict[str, Any]]:
    specs: dict[str, WorkloadSpec] = {
        # 短输入 / 长输出：decode 为主，考察持续生成速率与批效率。
        "short_in_long_out": build_axis_workload(
            input_length=128, output_length=512, concurrency=16
        ),
        # 长输入 / 短输出：prefill 为绝对瓶颈，最能暴露"长 prefill 阻塞 decode"。
        "long_in_short_out": build_axis_workload(
            input_length=8192, output_length=128, concurrency=4
        ),
        # 高前缀复用：RAG / 系统提示词场景，前缀缓存收益的上限区。
        "high_prefix_reuse": build_axis_workload(
            input_length=4096, output_length=128, concurrency=16,
            prefix_reuse_ratio=0.75,
        ),
        # 低前缀复用：与高复用组对照，用于证明收益确实来自复用而非缓存本身。
        "low_prefix_reuse": build_axis_workload(
            input_length=4096, output_length=128, concurrency=16,
            prefix_reuse_ratio=0.25,
        ),
        # 低并发：单请求延迟敏感，调度开销占比高，批处理收益有限。
        "low_concurrency": build_axis_workload(
            input_length=2048, output_length=128, concurrency=1
        ),
        # 高并发：吞吐压力场景，显存与 KV 容量成为约束。
        "high_concurrency": build_axis_workload(
            input_length=2048, output_length=128, concurrency=64
        ),
        # PHASE 1 第③步的并发扫描锚点：长度取规范锚点（prompt 2048 / gen 128），
        # **并发由各配置覆盖**，因此名称里刻意不含并发数——并发是被扫描的轴，
        # 而 ``get_workload`` 会强制 ``payload["name"] = name``，
        # 把并发写进名字会在第一次覆盖之后就变成一句谎话。
        # 请求数同样由配置覆盖：它决定"跑几波"，不改变被扫描的并发。
        "phase1_anchor": build_axis_workload(
            input_length=2048, output_length=128, concurrency=8, num_requests=8
        ),
    }
    return {name: _to_payload(spec) for name, spec in specs.items()}


def _to_payload(spec: WorkloadSpec) -> dict[str, Any]:
    """把 WorkloadSpec 摊成 WORKLOADS 表用的字典（与手写条目同构）。"""
    payload: dict[str, Any] = {
        "name": spec.name,
        "num_requests": spec.num_requests,
        "prompt_len_min": spec.prompt_len_min,
        "prompt_len_max": spec.prompt_len_max,
        "gen_len_min": spec.gen_len_min,
        "gen_len_max": spec.gen_len_max,
        "concurrency": spec.concurrency,
        "arrival": spec.arrival,
        "request_rate_qps": spec.request_rate_qps,
        "prefix_groups": spec.prefix_groups,
        "shared_prefix_len": spec.shared_prefix_len,
        "seed": spec.seed,
    }
    # 只在 piecewise 下带上这两项：非 piecewise 的负载条目保持原有形状，
    # 免得"多出两个空字段"污染配置快照的逐字节比对。
    if spec.arrival == "piecewise":
        payload["rate_segments"] = [
            {"duration_s": s.duration_s, "qps": s.qps, "label": s.label}
            for s in spec.rate_segments
        ]
        payload["rate_noise"] = spec.rate_noise
    return payload


# 命名负载：全部参数显式写出，不做隐式继承。
WORKLOADS: dict[str, dict[str, Any]] = {
    # 短对话：prompt 与输出都短，考察调度开销与固定 launch 成本的占比。
    "short_chat": {
        "name": "short_chat",
        "num_requests": 64,
        "prompt_len_min": 256,
        "prompt_len_max": 768,
        "gen_len_min": 64,
        "gen_len_max": 192,
        "concurrency": 8,
        "arrival": "closed",
        "seed": 0,
    },
    # 长文档问答：prompt 很长、输出很短，prefill 是绝对瓶颈，最能暴露
    # 「长 prefill 阻塞 decode」的问题——分块预填充的主战场。
    "long_context_qa": {
        "name": "long_context_qa",
        "num_requests": 32,
        "prompt_len_min": 4096,
        "prompt_len_max": 8192,
        "gen_len_min": 32,
        "gen_len_max": 128,
        "concurrency": 4,
        "arrival": "closed",
        "seed": 0,
    },
    # 共享前缀：多请求共用同一段系统提示词，用于量化前缀缓存的收益上限。
    "shared_prefix_4k": {
        "name": "shared_prefix_4k",
        "num_requests": 48,
        "prompt_len_min": 4096,
        "prompt_len_max": 4352,
        "gen_len_min": 64,
        "gen_len_max": 256,
        "concurrency": 8,
        "arrival": "closed",
        "prefix_groups": 1,
        "shared_prefix_len": 4096,
        "seed": 0,
    },
    # 突发混合负载（项目本地）：与 mixed 同样的异质长度，但到达模型是突发。
    # 用于观察"瞬时过载下自适应控制器的反应"，是闭环版本测不出来的场景。
    "mixed_burst": {
        "name": "mixed_burst",
        "num_requests": 96,
        "prompt_len_min": 2048,
        "prompt_len_max": 4096,
        "gen_len_min": 64,
        "gen_len_max": 512,
        "concurrency": 16,
        "arrival": "burst",
        "request_rate_qps": 2.0,
        "seed": 0,
    },
    # 负载爬坡（项目本地，**首轮取值，须经 pilot 标定**）：
    # 到达速率按墙钟分段恒定 λ(t) —— 低压 → 高压 → 恢复。
    #
    # 存在的理由：研究问题是"负载**变化**时控制器能否优于静态配置"，而
    # closed / poisson / burst 三种到达模型都表达不出"负载水平的轨迹"——
    # closed 全程填满并发，poisson 是恒定速率，burst 只有瞬时尖峰。
    #
    # ⚠️ 速率取值的依据（**是量级估计，不是已标定的容量**——第 01 步 2026-09-24 加注）：
    #   探针实测 batch=8 的 **快态** decode 步 = 109.17 ms ⇒ 128 步 = 14.0 s
    #   ⇒ 满批服务速率 ≈ 8/14.0 = 0.57 req/s；P1S3D c8 实测墙钟
    #   20.4 s / 8 请求 = 0.391 req/s（含 prefill 与起停）。
    #   ⇒ **假设** C ≈ 0.5 req/s：low = 0.08（≈0.16C，批为 1~2，TPOT ~60~75 ms，达标）；
    #     high = 0.70（≈1.4C，队积压、批打满 8，TPOT ~110~140 ms，超标）。
    #   ⚠️⚠️ 这组值**尚未经真机 pilot 验证**，且**不能**由下面两条推出持续容量：
    #     ① 8 个请求的短批闭环完成率（P1S3D 每格只有 8 请求，`max_num_seqs` 还是 32）；
    #     ② 纯 decode 的快态最小值（那是性能**下界**，不是服务路径上的吞吐）。
    #   持续容量必须专门标定（见 NEXT_SESSION_TASKS.md 第 08 步）。若 pilot 显示 high 段
    #   没有积压（队深不涨）或积压过深（尾部请求 TTFT 爆炸），按实测调整 qps 后重跑，
    #   并把改动作为独立变更登记。
    #
    # ⚠️ 配套的**两个**前提（缺一则该负载测不到想测的东西）：
    #   ① scheduler.max_num_seqs 必须设为 8 —— 容量 C 是在"批上限 8"下测出来的，
    #      批上限一变，λ 相对容量的含义就变了；
    #   ② 报告里必须按 arrival_segments 逐段读（名义 vs 实际到达都给了），
    #      不能拿整 run 的汇总值当"高压下的表现"。
    "load_ramp": {
        "name": "load_ramp",
        "num_requests": 83,
        "prompt_len_min": 2048,
        "prompt_len_max": 2048,
        "gen_len_min": 128,
        "gen_len_max": 128,
        "concurrency": 8,
        "arrival": "piecewise",
        "rate_noise": "jittered",
        "rate_segments": [
            {"duration_s": 120.0, "qps": 0.08, "label": "low"},
            {"duration_s": 90.0, "qps": 0.70, "label": "high"},
            {"duration_s": 120.0, "qps": 0.08, "label": "recover"},
        ],
        "seed": 0,
    },
    # 混合负载（规范原型）：长短 prompt 混在一起，考察调度在异质请求下的取舍能力。
    # 与 mixed_burst 的区别只在到达模型：这里是闭环（固定客户端池），
    # mixed_burst 是突发。选哪个取决于要回答的问题——
    # 比"稳定性"用闭环（消除到达随机性），比"过载行为"用突发。
    #
    # prompt 跨度 4 倍（2048~8192）而不是 128~8192：共享前缀是**每请求的绝对值**，
    # 一个 128 token 的请求装不下 2048 token 的共享前缀，会被 build_requests 拒绝。
    # 想要真正"短请求 + 长共享前缀"的混合，需要让共享前缀按请求长度自适应，
    # 那是本模型当前不支持的形状（见模块末尾"已知限制"）。
    # 带 ~40% 复用是刻意的：真实混合流量几乎总有一部分共享上下文，
    # 完全无复用的混合负载会低估前缀缓存在这类场景下的作用。
    "mixed": {
        "name": "mixed",
        "num_requests": 96,
        "prompt_len_min": 2048,
        "prompt_len_max": 8192,
        "gen_len_min": 128,
        "gen_len_max": 512,
        "concurrency": 16,
        "arrival": "closed",
        "prefix_groups": 4,
        "shared_prefix_len": 2048,
        "seed": 0,
    },
}


# 规范第 6 节的原型并入注册表。
# 重名直接抛错而不是静默覆盖——静默覆盖会让某个实验悄悄用到另一条负载，
# 而 run_id 上完全看不出来。
_ARCHETYPES = _spec_archetypes()
_dup = sorted(set(_ARCHETYPES) & set(WORKLOADS))
if _dup:
    raise RuntimeError(
        f"规范原型与项目本地负载重名：{_dup}；请删掉其中一个，不要靠覆盖顺序决定语义"
    )
WORKLOADS.update(_ARCHETYPES)


#: 扫描锚点：四个轴共用同一个"其余项"取值。
#: 共用是刻意的——它让四个轴的结果可以互相参照（同一个基线负载）。
#: 代价是锚点在未去重的清单里会出现四次，见 matrix_specs()。
ANCHOR_INPUT_LENGTH = 2048
ANCHOR_OUTPUT_LENGTH = 128
ANCHOR_CONCURRENCY = 8
ANCHOR_PREFIX_INPUT_LENGTH = 4096


def _axis_plan(requests: int) -> Iterator[tuple[str, WorkloadSpec]]:
    """按轴产出 (轴名, 负载) 序列。**不去重**——同一负载可服务多个轴。"""
    for n in INPUT_LENGTH_AXIS:
        yield "input_length", build_axis_workload(
            input_length=n, output_length=ANCHOR_OUTPUT_LENGTH,
            concurrency=ANCHOR_CONCURRENCY, num_requests=requests,
        )
    for n in OUTPUT_LENGTH_AXIS:
        yield "output_length", build_axis_workload(
            input_length=ANCHOR_INPUT_LENGTH, output_length=n,
            concurrency=ANCHOR_CONCURRENCY, num_requests=requests,
        )
    for c in CONCURRENCY_AXIS:
        yield "concurrency", build_axis_workload(
            input_length=ANCHOR_INPUT_LENGTH, output_length=ANCHOR_OUTPUT_LENGTH,
            concurrency=c, num_requests=requests,
        )
    for r in PREFIX_REUSE_AXIS:
        yield "prefix_reuse", build_axis_workload(
            input_length=ANCHOR_PREFIX_INPUT_LENGTH, output_length=ANCHOR_OUTPUT_LENGTH,
            concurrency=ANCHOR_CONCURRENCY, prefix_reuse_ratio=r, num_requests=requests,
        )


def matrix_specs(requests: int = DEFAULT_AXIS_REQUESTS) -> Iterator[dict[str, Any]]:
    """展开规范第 6 节的四个单变量扫描序列，**按负载去重**。

    每次只让一个轴变化、其余取锚点值——这是单变量原则在负载层面的落实。

    为什么必须去重：四个轴共用同一个锚点 (in=2048, out=128, c=8)，
    不去重会把**同一条负载**排出三次。后果不只是浪费三次运行时间：
    run_id 是「负载名 + 配置名」，同一条负载配上不同轴的配置文件时
    人眼会以为它们是三个不同的实验条件，而实际上完全一样。

    去重后每条负载带两个标注：
        ``axes``      它服务于哪些轴（多个 = 它被用作锚点）
        ``is_anchor`` 是否被多个轴共用
    """
    merged: dict[str, dict[str, Any]] = {}
    for axis, spec in _axis_plan(requests):
        payload = _to_payload(spec)
        key = payload["name"]
        if key in merged:
            merged[key]["_axes"].append(axis)
            merged[key]["_is_anchor"] = True
        else:
            payload["_axes"] = [axis]
            payload["_is_anchor"] = False
            merged[key] = payload
    yield from merged.values()


def list_workloads() -> list[str]:
    return sorted(WORKLOADS)


def get_workload(name: str, **overrides: Any) -> WorkloadSpec:
    """按名取负载定义，允许单点覆盖（覆盖项必须已在定义中存在，防止拼写错误静默生效）。"""
    if name not in WORKLOADS:
        raise KeyError(
            f"未知负载 {name!r}；可用：{list_workloads()}。"
            f"新增负载请写入 benchmark/workloads.py 的 WORKLOADS 表"
        )
    payload = dict(WORKLOADS[name])
    unknown = set(overrides) - set(payload)
    if unknown:
        raise KeyError(f"负载 {name!r} 不支持覆盖字段 {sorted(unknown)}")
    payload.update(overrides)
    payload["name"] = name
    return WorkloadSpec.from_dict(payload)


def describe_all() -> dict[str, Any]:
    return {name: WORKLOADS[name] for name in list_workloads()}


# --------------------------------------------------------------------------- #
# CLI：查看注册表与扫描序列
#
# 存在的理由：扫描轴的取值只有一处定义（本模块的常量），但要用它的人
# （准备配置、对表、写论文横轴）不该被迫写一段 sys.path hack 的 Python 片段。
# --------------------------------------------------------------------------- #

def build_parser() -> Any:
    import argparse

    p = argparse.ArgumentParser(
        description="工作负载注册表与规范第 6 节扫描轴",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python benchmark/workloads.py --axes       # 只看四个扫描轴的取值\n"
            "  python benchmark/workloads.py --matrix     # 展开待跑清单\n"
            "  python benchmark/workloads.py --list       # 列出注册表全部负载\n"
            "  python benchmark/workloads.py --show mixed # 看某条负载的完整参数\n"
        ),
    )
    p.add_argument("--axes", action="store_true", help="打印四个扫描轴的取值")
    p.add_argument("--matrix", action="store_true", help="展开四个单变量扫描序列")
    p.add_argument("--list", action="store_true", help="列出已注册的负载名")
    p.add_argument("--show", metavar="NAME", default=None, help="打印指定负载的参数")
    p.add_argument("--requests", type=int, default=DEFAULT_AXIS_REQUESTS,
                   help=f"扫描序列的请求数（默认 {DEFAULT_AXIS_REQUESTS}，恒定以保证单变量）")
    return p


def main(argv: list[str] | None = None) -> int:
    import json

    args = build_parser().parse_args(argv)

    if args.axes:
        print("输入长度轴 :", INPUT_LENGTH_AXIS)
        print("输出长度轴 :", OUTPUT_LENGTH_AXIS)
        print("并发轴     :", CONCURRENCY_AXIS)
        print("前缀复用轴 :", PREFIX_REUSE_AXIS)
        print(f"\n规范第 6 节点名分布：{list(_spec_archetypes())}")

    if args.matrix:
        entries = list(matrix_specs(requests=args.requests))
        print(f"\n扫描清单：{len(entries)} 条去重后的负载（请求数固定为 {args.requests}）")
        for entry in entries:
            tag = "锚点" if entry["_is_anchor"] else "    "
            print(
                f"  [{tag}] {entry['name']:<42} "
                f"in={entry['prompt_len_min']:<6} out={entry['gen_len_min']:<4} "
                f"c={entry['concurrency']:<3} reuse={entry['shared_prefix_len']:<5} "
                f"轴={','.join(entry['_axes'])}"
            )

    if args.list:
        print("\n已注册负载：")
        for name in list_workloads():
            origin = "规范原型" if name in _spec_archetypes() else "项目本地"
            print(f"  {name:<20} [{origin}]")

    if args.show:
        if args.show not in WORKLOADS:
            print(f"未知负载 {args.show!r}；可用：{list_workloads()}")
            return 2
        print(json.dumps(WORKLOADS[args.show], ensure_ascii=False, indent=2))

    if not (args.axes or args.matrix or args.list or args.show):
        build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
# 已知限制（写在这里而不是口头流传）
# --------------------------------------------------------------------------- #
# 1. 共享前缀是**每请求的绝对值**，且必须 <= prompt_len_min。
#    后果：无法表达"短请求 + 长共享前缀"的混合负载——一个 128 token 的请求
#    装不下 4096 token 的系统提示词。真实 RAG 流量恰恰是这个形状。
#    要支持它，需要让 shared_prefix_len 随请求长度自适应（例如取
#    min(shared_prefix_len, prompt_len * ratio)），并同步修改缓存策略
#    对"部分共享前缀"的判定。这是一次语义变更，必须作为独立实验登记。
# 2. 长度分布只有均匀采样，没有重尾。真实对话流量的长度分布明显右偏，
#    用均匀分布会低估 P99——而 P99 恰恰是本项目的主要判据。
#    需要时在 WorkloadSpec 加新字段（如分布类型），不要改现有字段语义。
# 3. 到达模型：closed / poisson / burst / piecewise 四种。
#    piecewise（分段常量速率 λ(t)，见 ``load_ramp``）能表达"低压→高压→恢复"的
#    **负载水平轨迹**，另外三种都不能。⚠️ 但它仍不是真实流量：段内是分层抽样
#    （或条件泊松），没有会话相关性、没有重尾突发；段边界是**墙钟上确定的**，
#    这是刻意选择（否则跨 run 无法对照），代价是"何时升压"这一步是理想化的。

