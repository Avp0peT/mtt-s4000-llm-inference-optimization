#!/usr/bin/env python
"""指标聚合：把 results/raw 的原始产物收敛成 results/processed 的分析表。

输出三个文件：
    results/processed/<run_id>.csv     每请求一行（来自 requests.jsonl）
    results/processed/summary.csv      每个 run 一行（喂给绘图的唯一数据源）
    results/processed/repeats.csv      同一 run_id 的多次重复聚合（含跨重复离散度）

纪律
----
1. **不修改 results/raw**：本脚本只读。
2. **默认只收 MEASURED**：数据标签（规范第 10 节）不是装饰；
   它是本脚本的过滤条件。``TEST_DATA``（--dry-run 合成数据）、
   ``HARDWARE_REQUIRED``（本机无 NPU）、``NOT_MEASURED``（硬件状态未知）
   一律不进汇总表。需要核对链路口径时用 ``--include-non-measured``，
   此时产物会带上 ``data_label`` 与 ``label_source`` 两列，一眼可辨。
3. 重复次数 < 3 的 run 会在 repeats.csv 中标注 ``insufficient_repeats``，
   而不是假装它的离散度可信。

关于 ``label_source``
---------------------
标签有两条来源：
    ``explicit``        产物里写了 ``data_label``（新版本 harness 的行为）
    ``legacy_inferred`` 产物里没有标签，只能按 ``dry_run`` 字段回推

第二条是**兼容老产物**用的，不是正常路径，而且它有一条硬规矩：

    **回推只允许降级，永不升级。**

从 ``dry_run`` 能可靠推出的只有一件事——``dry_run=True`` 说明数字是合成的，
所以可以降级成 ``TEST_DATA``。反过来推不出"实测"：在没有 NPU 的机器上跑真机路径
同样是 ``dry_run=False``，结果既非实测也不可复现；旧产物里又没有执行环境声明，
连"跑在哪台机器上"都无从知道。因此缺标签的旧产物一律判 ``NOT_MEASURED``。

这条规矩是在堵一个具体的错：早先的实现对 ``dry_run=False`` 的旧产物直接回推
``MEASURED``，于是一批"本来只是没打标签"的数据被静默升格成可进结论的实测数据。
它不抛错、不告警，只在最后产出一张看起来完全正常的汇总表。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.grouping import group_key as _canonical_group_key  # noqa: E402
from ascend_opt.metrics import percentile, robust_cv  # noqa: E402
from ascend_opt.provenance import (  # noqa: E402
    DataLabel,
    ExecutionEnvironment,
    summarize_labels,
)

MIN_REPEATS_FOR_CONCLUSION = 3

#: summary/repeats 里展开成分位数列的指标块。
#: admission / queue / prefill 是 TTFT 的三段分解——分开展开的用处是：
#: 「该加容量还是该调分块」这个判断只能从分段上读出来，从 TTFT 总数上读不出来
#: （排队变差而计算变好时，总数甚至可能是持平的）。
DIST_BLOCKS = ("ttft", "tpot", "stall", "e2e", "admission", "queue", "prefill")
#: 展开的分位数与统计量。与 metrics._dist 的输出键保持一致。
DIST_KEYS = ("p50", "median", "p90", "p95", "p99", "mean", "min", "max")


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #

def discover_runs(raw_dir: Path) -> list[Path]:
    """找出所有含 meta.json 的 run 目录。"""
    if not raw_dir.exists():
        return []
    return sorted(p for p in raw_dir.iterdir() if (p / "meta.json").exists())


#: 由本脚本自己产出的聚合表。它们不是"某个 run 的明细"，不参与孤儿回收。
_AGGREGATE_TABLES = ("summary.csv", "repeats.csv")


def prune_orphan_derived(out_dir: Path, raw_dir: Path) -> list[Path]:
    """回收"原始 run 已不存在"的逐 run 派生 CSV，返回被删除的文件列表。

    为什么必须回收：派生文件（``results/processed/<run_id>.csv``）的列结构随
    schema 演进而变。raw 目录被清理后，旧列结构的 CSV 会**留在原地**，
    下游（绘图、脚本、人工翻看）读到它时无法判断它是哪个版本的产物——
    它看起来就是一份正常数据。本仓库已实际出现过这种遗留文件：
    列里是 ``success`` 而没有 ``outcome``，正是本项要堵的那类静默错误。

    判据刻意选「raw 里有没有同名目录」（结构性），而不是「本次有没有被选中」：
    依赖"本次选中"的话，``--run-id X`` 这种过滤运行会把其它 run 的明细一起删掉。
    只删没有 raw 兜底的，才不会把可重算数据误当垃圾。
    """
    if not out_dir.exists():
        return []
    known = {p.name for p in raw_dir.iterdir() if p.is_dir()} if raw_dir.exists() else set()
    removed: list[Path] = []
    for path in sorted(out_dir.glob("*.csv")):
        if path.name in _AGGREGATE_TABLES or path.stem in known:
            continue
        path.unlink()
        removed.append(path)
    return removed


def load_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))


def effective_label(meta: dict[str, Any]) -> tuple[DataLabel, str]:
    """判定一个 run 的数据标签，并说明标签的来源。

    返回 ``(label, source)``，``source`` ∈ ``{"explicit", "legacy_inferred"}``。

    显式标签优先。缺失时**只允许降级**：
    ``dry_run=True`` 的旧产物 → ``TEST_DATA``（能确知数字是合成的）；
    其余旧产物 → ``NOT_MEASURED``。理由见模块 docstring——
    ``dry_run=False`` 推不出"实测"，无 NPU 机器上跑真机路径同样是 False。
    凡 ``label_source`` 不是 ``explicit`` 的 run，进结论前必须重新生成产物。
    """
    explicit = (meta.get("summary") or {}).get("data_label") or meta.get("data_label")
    if explicit:
        try:
            return DataLabel(explicit), "explicit"
        except ValueError:
            # 未知标签值：不猜，按最保守的 NOT_MEASURED 处理。
            return DataLabel.NOT_MEASURED, "explicit"

    ctx = meta.get("context") or {}
    if bool(ctx.get("dry_run", False)):
        # 唯一的合法升级方向之外的情形：dry_run=True 一定是合成数据，
        # 这是**降级**推断（从"无标签"到"测试数据"），不是升格。
        return DataLabel.TEST_DATA, "legacy_inferred"
    # 缺标签 + dry_run=False：既不能断言实测，也无法回推执行环境。
    # 早先这里返回 MEASURED，等于把"没打标签"静默读成"测过了"。
    return DataLabel.NOT_MEASURED, "legacy_inferred"


def iter_requests(run_dir: Path) -> Iterable[dict[str, Any]]:
    path = run_dir / "requests.jsonl"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _repro_value(meta: dict[str, Any], key: str, default: str) -> str:
    """从 ``reproducibility`` 段读一个标量字段，段里没有就回落到 meta 顶层。

    回落是必要的：正式位置是 ``reproducibility.<key>``（规范第 8 节），但 meta 里
    同时存在若干平铺写法。只认一处，就会出现"字段明明写了、汇总表里却是空的"——
    这类漏读不会报错，只会让一列静默空掉。
    """
    repro = meta.get("reproducibility") or {}
    value = repro.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        value = meta.get(key)
    if value is None:
        return default
    return str(value).strip() or default


def _column(row: dict[str, Any], key: str, default: str) -> str:
    """读一个汇总列的字符串值；空/缺失落到 ``default``。

    分组与合并都要用同一套归一化，否则同一个空值会在两处得到不同结论。
    """
    value = row.get(key)
    if value is None:
        return default
    return str(value).strip() or default


def flatten_summary(run_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """把一个 run 的 meta 拍平成一行。字段名与 RunSummary.flat_row 保持一致。"""
    summary = meta.get("summary") or {}
    ctx = meta.get("context") or {}
    cfg = meta.get("config") or {}
    label, label_source = effective_label(meta)
    row: dict[str, Any] = {
        "run_id": summary.get("run_id") or run_dir.name,
        "workload": summary.get("workload") or ctx.get("workload", ""),
        "config": summary.get("config") or cfg.get("config_stem", ""),
        "config_path": ctx.get("config_path", ""),
        "repeat_index": ctx.get("repeat_index", 0),
        "dry_run": bool(ctx.get("dry_run", False)),
        # 数据来源标签：判断这一行能不能进结论的唯一依据。
        "data_label": label.value,
        "label_source": label_source,
        "eligible_for_conclusion": label.eligible_for_conclusion,
        # 推理**实际执行**在哪台机器上（规范 "Remote Execution Environment"）。
        # 必须显形：一张混着本机 run 与远程 run 的表，光看 data_label 是看不出来的
        # （两者都可能标 MEASURED），而它们的可比性完全不同。
        "execution_environment": _repro_value(
            meta, "execution_environment", ExecutionEnvironment.UNKNOWN.value
        ),
        # ---- 环境与运行时维度 ----
        # 这三列原先**都不在表里**。后果不是"少了几列"，而是"两个平台/两个后端
        # 的数字混在一张表里，既看不出来、也筛不出来"。它们同时是分组键的成分
        # （见 group_key）——只记录、不参与分组的字段等于没记（本仓库踩过）。
        #
        # hardware_platform 取不到时写 unknown：历史 run 本来就没这个字段，
        # 保留 unknown 是为了让它们落在"无从判断"，而不是被塞进某个平台。
        "hardware_platform": _repro_value(meta, "hardware_platform", "unknown"),
        # simulate / real。同一批 run 里混了两者，光看 dry_run 列看不出来
        # （dry_run 是布尔，而"哪个后端产出的"是它背后的语义）。
        "backend": _repro_value(meta, "backend", "unknown"),
        "precision": _repro_value(meta, "precision", ""),
        "quantization": _repro_value(meta, "quantization", ""),
        "n_requests": summary.get("n_requests", 0),
        "n_success": summary.get("n_success", 0),
        # ---- 结局账目：失败 / 拒绝 / 未完成分开记 ----
        # 默认值刻意用 ""（缺失）而不是 0：旧产物没有这些字段，填 0 等于宣称
        # "没有失败、没有未完成"，而真实语义是"当时根本没记"。
        # 把"没数据"伪装成"数据是 0"正是本项要堵的静默错误。
        "n_failed": summary.get("n_failed", ""),
        "n_rejected": summary.get("n_rejected", ""),
        "n_unfinished": summary.get("n_unfinished", ""),
        "failure_rate": summary.get("failure_rate", ""),
        "rejection_rate": summary.get("rejection_rate", ""),
        "unfinished_rate": summary.get("unfinished_rate", ""),
        # ---- SLO 三口径 ----
        # 比例（达标/全部提交）、达标请求吞吐、达标 token 吞吐，三者不可互相替代：
        # 比例高但请求少，与比例低但请求多，是两种完全不同的处境。
        # 未配置 SLO 时这三项是 None（不可计算），因此同样不能用 0 兜底。
        "n_slo_satisfied": summary.get("n_slo_satisfied", ""),
        "slo_satisfaction_rate": summary.get("slo_satisfaction_rate", ""),
        "slo_goodput_req_s": summary.get("slo_goodput_req_s", ""),
        "slo_goodput_tok_s": summary.get("slo_goodput_tok_s", ""),
        "output_tokens": summary.get("output_tokens", 0),
        "prompt_tokens": summary.get("prompt_tokens", 0),
        "cached_tokens": summary.get("cached_tokens", 0),
        "cache_hit_rate": summary.get("cache_hit_rate", 0.0),
        "wall_ms": summary.get("wall_ms", 0.0),
        "throughput_tok_s": summary.get("throughput_tok_s", 0.0),
    }
    for prefix in DIST_BLOCKS:
        block = summary.get(prefix) or {}
        for key in DIST_KEYS:
            row[f"{prefix}_{key}"] = block.get(key, "")
    for key, value in (summary.get("stability") or {}).items():
        row[f"cv_{key}"] = value
    extra = summary.get("extra") or {}
    row["steps"] = extra.get("steps", 0)
    row["dropped_requests"] = extra.get("dropped_requests", 0)
    row["accounting_error_p99_ms"] = summary.get("accounting_error_p99_ms", "")
    return row


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #

def group_key(row: dict[str, Any]) -> str:
    """把同一实验的多次重复归到一组。

    重复运行共享同一个 run_id（后缀 -rN），因此按
    **(运行批次, 负载, 配置, 硬件平台, 执行后端)** 分组。

    后两维必须进键，理由是同一个失效模式：合并**不会报错**，只会安静地产出一个
    "谁都没测过"的中位数。这一点已被复现过——同负载同配置下，两个平台上的
    TEST_DATA（100 与 200）被合成了一根值为 150 的柱。后端同理：模拟与真机的
    数字口径本就不同（前者来自一阶设备模型），合起来算离散度毫无意义。

    方向刻意选"宁可拆开、不要合并"：拆错会让一组落到 ``insufficient_repeats``
    而被看见；并错只会得到一张看起来完全正常的表。

    实现**只有一份**，在 :mod:`ascend_opt.grouping`：绘图与导出校验必须与这里
    得出同一答案。曾经三处各写一份，结果绘图漏了后端维度、导出校验又只认
    run_id 前缀，两边都不报错，只是安静地给出错误的图与假的"重复次数不一致"。
    """
    return _canonical_group_key(row)


def aggregate_repeats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """跨重复聚合，并给出跨重复离散度。

    跨重复离散度回答的问题是「换一次运行，数字会飘多少」，
    它是判断 A/B 差异是否真实的第一道门槛——组内离散度不覆盖这一维度。
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[group_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items()):
        n = len(group)
        # 跨重复聚合的指标集。含 P95 是因为规范第 7 节要求记录 P95，
        # 且论文图（TTFT/TPOT vs 并发）需要用同一分位口径贯穿全文；
        # 只给 P50/P99 会逼着绘图层另起一套口径。
        series = {
            name: [float(g[name]) for g in group if g.get(name) not in ("", None)]
            for name in (
                "ttft_p50", "ttft_p95", "ttft_p99",
                "tpot_p50", "tpot_p95", "tpot_p99",
                "stall_p99", "throughput_tok_s",
                # TTFT 的排队段：跨重复看它，才能判断"调度慢"是不是稳定的系统性质，
                # 而不是某一次运行的偶然。只看 TTFT 总数会被 prefill 执行段抵消。
                "queue_p99",
                # 结局与达标口径也必须跨重复聚合：一次性的未完成率不能代表配置，
                # 但"每次都出现未完成"就是配置的性质了——后者才是必须上报的。
                "unfinished_rate",
                "slo_satisfaction_rate",
            )
        }

        record: dict[str, Any] = {
            "group": key,
            "run_id_base": key.split("|")[0],
            # **本组的明确成员**，分号分隔并排序。导出校验据此核对重复次数，
            # 不再自己去按前缀重新数——那是两套规则，且前缀法无法表达
            # "同一批次被拆成两组"。成员落盘后，"这一组到底有哪几次运行"
            # 成为一条可 diff、可复核的记录，而不是每次重新推导的结论。
            "member_run_ids": ";".join(
                sorted(str(g.get("run_id", "")) for g in group)
            ),
            "workload": group[0]["workload"],
            "config": group[0]["config"],
            "dry_run": group[0]["dry_run"],
            # 一组的标签取组内最保守者：只要有一次不是实测，整组就不能当结论。
            "data_label": _strictest_label(group),
            "label_source": _merge_label_sources(group),
            "execution_environment": _merge_execution_environments(group),
            # 分组键的成分也照样输出一份。它们此时必然是单值（进了键就拆开了），
            # 但仍然显式写出来：一是让 repeats.csv 自解释（不必反解 group 串），
            # 二是万一有人改了 group_key 而忘了这里，列里会出现 mixed，
            # 那是一个不会静默的警报。
            "hardware_platform": _merge_scalar(group, "hardware_platform"),
            "backend": _merge_scalar(group, "backend"),
            "precision": _merge_scalar(group, "precision", ""),
            "quantization": _merge_scalar(group, "quantization", ""),
            "n_repeats": n,
            "sufficient_repeats": n >= MIN_REPEATS_FOR_CONCLUSION,
        }
        for name, values in series.items():
            if not values:
                # 该指标在所有重复里都缺失（例如真实后端没采 token 间隔明细）。
                # 明确写空而不是填 0——0 会被读成「完全没有卡顿」。
                record[f"{name}_median"] = ""
                record[f"{name}_min"] = ""
                record[f"{name}_max"] = ""
                record[f"{name}_spread_pct"] = ""
                record[f"{name}_robust_cv"] = ""
                record[f"{name}_cv"] = ""
                continue
            record[f"{name}_median"] = round(percentile(values, 50), 4)
            record[f"{name}_min"] = round(min(values), 4)
            record[f"{name}_max"] = round(max(values), 4)
            record[f"{name}_spread_pct"] = _spread_pct(values)
            # 离散度只在重复次数达标时才给。
            # 用 2 个点算出来的「稳健 CV」会诱导读者相信一个没有统计意义的数字，
            # 所以这里宁可留空——空值会逼人去补重复次数。
            enough = n >= MIN_REPEATS_FOR_CONCLUSION
            record[f"{name}_robust_cv"] = round(robust_cv(values), 4) if enough else ""
            record[f"{name}_cv"] = round(_plain_cv(values), 4) if enough else ""
        out.append(record)
    return out


def _spread_pct(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    med = percentile(values, 50)
    if med == 0:
        return 0.0
    return round((max(values) - min(values)) / med * 100.0, 4)


def _strictest_label(group: list[dict[str, Any]]) -> str:
    """取组内最保守的标签。

    不可用于结论的标签排在前面，只要有一个就取它。理由：把一次模拟 run
    和两次实测 run 混在一组里求出的"稳健 CV"是没有意义的数字，
    取最保守标签能让这一组直接落进"不可用于结论"。
    """
    order = [
        DataLabel.TEST_DATA,
        DataLabel.HARDWARE_REQUIRED,
        DataLabel.NOT_MEASURED,
        DataLabel.MEASURED,
    ]
    present = {DataLabel(g["data_label"]) for g in group}
    for label in order:
        if label in present:
            return label.value
    return DataLabel.NOT_MEASURED.value


def _merge_label_sources(group: list[dict[str, Any]]) -> str:
    sources = sorted({g.get("label_source", "explicit") for g in group})
    if len(sources) == 1:
        return sources[0]
    # 混合来源：只要有一个是回推出来的，整个标签就不算权威。
    return "mixed"


def _merge_execution_environments(group: list[dict[str, Any]]) -> str:
    """一组重复的执行环境。

    混了不同环境时返回 ``mixed``，而不是取第一个——取第一个会把
    「一半在本机跑、一半在远程跑」这种最该被看见的情况掩盖掉，
    而这两批数字的可比性完全不同。
    """
    return _merge_scalar(group, "execution_environment",
                         ExecutionEnvironment.UNKNOWN.value)


def _merge_scalar(group: list[dict[str, Any]], key: str, default: str = "unknown") -> str:
    """一组重复在某一维上的取值；不一致返回 ``mixed``。

    与执行环境同一条规则：不取第一个。取第一个会把"这一组其实不是同一个实验"
    这件事从表里抹掉，而它恰恰是读者最需要知道的一条。
    """
    values = {_column(g, key, default) for g in group}
    if len(values) == 1:
        return values.pop()
    return "mixed"


def _plain_cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return (var ** 0.5) / abs(mean)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # newline=""：这里只写空串，但显式声明以免日后有人往里加行时踩到
        # Windows 的 \n -> \r\n 转换（见下方 with 块的说明）。
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return
    # 列顺序取并集并排序，保证多次运行产出的表结构稳定、可 diff。
    columns = sorted({k for row in rows for k in row})
    # newline="" 是 csv 模块的硬要求（它自己负责写行尾），同时也保证在 Windows 上
    # 不会把 \n 翻成 \r\n——results/ 是证据，本机产 CRLF、远程产 LF 会造出与行为
    # 无关的假 diff，且不报错、只有量字节才发现。
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="聚合 results/raw -> results/processed")
    p.add_argument("--raw", default=None, help="原始数据目录，默认 results/raw")
    p.add_argument("--out", default=None, help="输出目录，默认 results/processed")
    p.add_argument(
        "--include-non-measured",
        action="store_true",
        help="把非 MEASURED 的 run（TEST_DATA / HARDWARE_REQUIRED / NOT_MEASURED）"
             "也纳入汇总；产物会带 data_label 与 label_source 列（默认排除）",
    )
    p.add_argument(
        "--include-dry-run",
        action="store_true",
        help="[兼容旧用法] 等价于 --include-non-measured",
    )
    p.add_argument("--run-id", default=None, help="只处理指定 run_id（含重复后缀）")
    p.add_argument(
        "--keep-orphans",
        action="store_true",
        help="不回收 raw 中已无对应 run 的逐 run 派生 CSV（默认回收；"
             "残留的旧列结构文件会被下游当成正常数据）",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    include_all = args.include_non_measured or args.include_dry_run
    raw_dir = Path(args.raw) if args.raw else _ROOT / "results" / "raw"
    out_dir = Path(args.out) if args.out else _ROOT / "results" / "processed"

    # 派生数据卫生：raw 已不存在的逐 run CSV 必须先清掉。放在最前面，
    # 是因为它的目的是让"本次产出"与"盘上残留"不混在一起——若放在最后，
    # 中途 return 的分支（例如没有任何可用 run）就会漏掉这次回收。
    if not args.keep_orphans:
        for stale in prune_orphan_derived(out_dir, raw_dir):
            print(f"  已回收孤儿派生文件（raw 中已无对应 run）：{stale.name}")

    run_dirs = discover_runs(raw_dir)
    if not run_dirs:
        # 归零不算错误：盘上没有真实 run 是一个合法状态（例如刚克隆仓库）。
        # 用非零退出码会让上层脚本中断，反而掩盖了真正的问题。
        print(f"未在 {raw_dir} 找到任何含 meta.json 的 run。")
        print("这不是错误——先跑 benchmark/run_benchmark.py 生成产物。")
        return 0

    rows: list[dict[str, Any]] = []
    skipped: list[DataLabel] = []
    per_run_files: list[Path] = []

    for run_dir in run_dirs:
        meta = load_meta(run_dir)
        if args.run_id and args.run_id not in run_dir.name:
            continue

        label, _source = effective_label(meta)
        if not label.eligible_for_conclusion and not include_all:
            # 过滤的依据是数据标签本身，而不是 dry_run 这个更弱的代理。
            skipped.append(label)
            continue

        rows.append(flatten_summary(run_dir, meta))

        req_rows = list(iter_requests(run_dir))
        if req_rows:
            target = out_dir / f"{run_dir.name}.csv"
            write_csv(target, req_rows)
            per_run_files.append(target)

    if not rows:
        counts = summarize_labels(skipped)
        # 必须清掉上一次留下的汇总表。否则下游（绘图、导出）会读到**上一次**
        # 的数字却以为是本次的——派生数据被静默复用是这类流水线最危险的一种错。
        # 置空而不是删除：文件存在但为空，会让 export_results.sh 的
        # size==0 检查给出准确的报错，而不是"找不到文件"。
        for stale in (out_dir / "summary.csv", out_dir / "repeats.csv"):
            if stale.exists():
                print(f"  已清空过期汇总表：{stale}")
                write_csv(stale, [])
        print(f"没有可用于结论的 run（本目录共 {len(run_dirs)} 个 run，均被标签过滤）。")
        for name, count in counts.items():
            if count:
                print(f"  跳过 {name}: {count}")
        print("  这是预期行为：规范第 10 节要求只有 MEASURED 能进汇总表。")
        print("  需要核对链路口径时用 --include-non-measured，")
        print("  并注意产物中的 data_label / label_source 两列。")
        return 0

    write_csv(out_dir / "summary.csv", rows)
    repeats = aggregate_repeats(rows)
    write_csv(out_dir / "repeats.csv", repeats)

    print(f"已聚合 {len(rows)} 个 run（按标签跳过 {len(skipped)} 个）")
    if skipped:
        for name, count in summarize_labels(skipped).items():
            if count:
                print(f"  跳过 {name}: {count}")
    print(f"  每请求明细：{len(per_run_files)} 个文件 -> {out_dir}")
    print(f"  {out_dir / 'summary.csv'}")
    print(f"  {out_dir / 'repeats.csv'}")

    # 回推标签必须显形：它无法区分"真机实测"与"无 NPU 机器上跑的真机路径"。
    legacy = [r["run_id_base"] for r in repeats if r["label_source"] != "explicit"]
    if legacy:
        print(
            f"\n警告：{len(legacy)} 组的标签来自 dry_run 回推（label_source != explicit），"
            f"不是权威标签，进结论前需重新生成产物："
        )
        for name in legacy[:5]:
            print(f"  - {name}")

    insufficient = [r["run_id_base"] for r in repeats if not r["sufficient_repeats"]]
    if insufficient:
        print(
            f"\n提示：{len(insufficient)} 组重复次数 < {MIN_REPEATS_FOR_CONCLUSION}，"
            f"离散度不可信，不能用于结论："
        )
        for name in insufficient[:10]:
            print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
