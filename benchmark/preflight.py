#!/usr/bin/env python
"""导出前的前置校验：交付包里每一个数字都必须有据可查。

为什么要有这一步
----------------
``results/processed/summary.csv`` 是一张**看起来总是对**的表：列名齐全、
数字成形、放大到论文里也像模像样。它和"可复核的结果"之间的差别，全在表外的
四件事上——原始明细还在不在、环境指纹是谁的、重复够不够、有没有登记。

这四件事的共同点是：缺了它们**不会报错**。表照样生成、图照样画出来、
导出的包照样"成功"。所以必须由一道独立的前置校验把它们变成**阻断性错误**，
否则它们就只会在答辩或被质疑时才被发现。

四项检查
--------
1. **原始明细**   ``results/raw/<run_id>/`` 三重内容校验：
   - ``requests.jsonl`` 存在、非空、且**逐行**可解析（一个半截写入的文件
     "字节数看着对"，下游却只会读到 0 条记录，把"有数据"变成"没数据"）；
   - ``meta.json`` 存在、可解析、必需键齐全、``run_id`` 与目录名自洽、
     ``data_label`` 与汇总行自洽；
   - ``config.snapshot.yaml`` 存在且可解析成映射。
   没有它，汇总里的每个分位数都成了不可追溯的断言。
2. **环境指纹**   ``results/raw/_env/env.json``（或 run 目录内的）存在、是合法
   JSON 对象、且**显式声明**了硬件可用性。性能数字只在采集它的那台机器上可解释
   （规范 "Remote Execution Environment"），而 ``results/raw/_env/`` 是
   .gitignore 的——克隆到新机器后它**一定**不在，这条检查就是为那个时刻准备的。
   可用性是**三态**：字段缺失（旧格式，该重采）、``null``（探测没得出结论，
   该修探测后重采）、``false``（确实没设备，该换机器）。三者的修法不同，
   所以本检查分三条报，不合并成一句"未声明可用性"。
3. **有效重复**   **从汇总行重新推导**每组成员数（不信任 ``repeats.csv`` 自报的
   ``sufficient_repeats``——一个过期的旗标会让 2 次重复看起来像 3 次），
   并核对它与 ``repeats.csv`` 是否同步。少于 3 次的离散度不具统计意义。
4. **实验登记**   组的 ``run_id`` 必须作为**表格单元**出现在 ``EXPERIMENTS.md``
   的某一行里。"文中出现过"不算——规范要求登记行含 run_id、重复次数、离散度，
   而散文给不出后两者。

严重级别
--------
``block`` 阻断导出；``warn`` 只提示。

- **缺原始明细在任何模式下都是 ``block``**：数字连自己都对不上，
  诊断包也一样读不出东西。
- 其余三项对 **MEASURED 行**是 ``block``，对非 MEASURED 行是 ``warn``。
- ``--lenient`` 用于 ``INCLUDE_DRY=1`` 这类只做链路口径核对的诊断包：
  此时没有行被当成 MEASURED，于是后三项全部降为 ``warn``，
  但它们**仍然会被打印出来**——降级不等于消失。

退出码：``0`` 无阻断项；``4`` 有阻断项。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.grouping import base_run_id  # noqa: E402
from ascend_opt.provenance import (  # noqa: E402
    HardwarePlatform,
    env_platform,
    platform_evidence_conflict,
)
from ascend_opt.telemetry import hardware_env_candidates  # noqa: E402

BLOCK = "block"
WARN = "warn"

# 环境指纹的文件名**不再由本模块定义**：候选路径一律问
# ``ascend_opt.telemetry.hardware_env_candidates``（见 :func:`_fingerprint_candidates`）。
# 在这里再写一份常量，就等于在第二处维护"指纹在哪"这条规则——两边一旦不同，
# 运行时判定与导出校验会读不同的文件，而两者的结论都"看起来正常"。
#: 与 collect_metrics.MIN_REPEATS_FOR_CONCLUSION 同一口径。
MIN_REPEATS_FOR_CONCLUSION = 3

DETAIL_FILENAME = "requests.jsonl"
META_FILENAME = "meta.json"
SNAPSHOT_FILENAME = "config.snapshot.yaml"

#: 一个 run 记录里必须具备的顶层键。缺了任何一项，这个 run 就无法被复核：
#:
#:   ``data_label``      这条数字属于哪一类（能否进结论的判据）
#:   ``summary``         汇总数字本体
#:   ``reproducibility`` 21 字段可复现性元数据（规范第 8 节）
#:   ``config``          当时的配置——``configs/`` 会随代码演进而变，快照不会
#:   ``context``         run_id / workload / 重复序号 / dry_run
REQUIRED_META_KEYS = ("data_label", "summary", "reproducibility", "config", "context")


@dataclass(frozen=True)
class Problem:
    """一条校验发现。``code`` 用于机器识别，``detail`` 必须写明**具体哪个 run**。"""

    code: str
    level: str
    detail: str

    def render(self) -> str:
        tag = "阻断" if self.level == BLOCK else "提示"
        return f"[{tag}] {self.code}: {self.detail}"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _is_measured(row: dict[str, Any]) -> bool:
    return (row.get("data_label") or "").strip() == "MEASURED"


def _platform_evidence_problems(
    env_path: Path,
    env_payload: dict[str, Any],
    raw_dir: Path,
    rows: list[dict[str, Any]],
    *,
    lenient: bool,
) -> list[Problem]:
    """逐 run 核对「**声明**的平台」与「指纹**检出**的平台」。

    为什么必须有这一项：另外两项检查都只能保证**声明之间**自洽——配置说
    ``hardware_platform=mthreads`` 且 ``execution_environment=remote_mthreads``，
    自我一致，却可以在指纹明确检出 Ascend 的情况下照样通过。自洽证明不了任何
    事实，只有"说的"与"查到的"对照才能证伪。

    缺证据时放行（任一方 unknown 即跳过），理由同
    :func:`ascend_opt.provenance.platform_evidence_conflict`：把"没证据"当成
    "证据矛盾"会让所有历史指纹与未声明平台的配置一起被拒。
    """
    try:
        detected = env_platform(env_payload)
    except ValueError as exc:
        return [Problem(
            "env_platform_unrecognized", BLOCK if not lenient else WARN,
            f"环境指纹 {env_path} 的 platform 字段无法识别：{exc}",
        )]

    out: list[Problem] = []
    for row in rows:
        rid = (row.get("run_id") or "").strip()
        if not rid:
            continue
        declared = _declared_platform(raw_dir, rid)
        if declared is None:
            continue
        reason = platform_evidence_conflict(declared, detected)
        if reason is None:
            continue
        level = WARN if (lenient or not _is_measured(row)) else BLOCK
        out.append(Problem(
            "platform_evidence_conflict", level,
            f"run {rid}：{reason}（指纹来源：{env_path}）"
            f"两种可能，按实际情况修：① 指纹采错了机器 —— 到"
            f"{declared.display_name} 主机上重跑 scripts/inspect_env.sh；"
            f"② 声明写错了平台 —— 把 hardware_platform 改成 "
            f"{detected.value!r} 并同步 execution_environment。"
            f"在此之前，这一行的数字不能按现在声明的平台解释",
        ))
    return out


def _declared_platform(raw_dir: Path, rid: str) -> HardwarePlatform | None:
    """从 run 的 ``meta.json`` 里读**声明的**平台。读不到/不认识返回 ``None``。

    不认识的值返回 ``None`` 而不是抛错：拼错的声明由 :mod:`ascend_opt.provenance`
    在装配元数据时就已经拦住，根本落不了盘；这里再报一遍只会让同一个问题
    出现两次，反而模糊了真正的第一现场。
    """
    meta, err = _load_json_object(raw_dir / rid / META_FILENAME)
    if err is not None or meta is None:
        return None
    repro = meta.get("reproducibility")
    if not isinstance(repro, dict):
        return None
    value = repro.get("hardware_platform")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return HardwarePlatform.parse(value)
    except ValueError:
        return None


def _fingerprint_candidates(raw_dir: Path, run_ids: Iterable[str]) -> list[Path]:
    """环境指纹的候选位置，**顺序与运行时一致**：run 目录优先，其次共享位置。

    两处都认，是因为采集时点与 run 目录的创建时点不同——指纹必须在 run
    开始**之前**采集（状态不能被 run 影响），而 run 目录要等到 run 开始才建。

    顺序刻意由 :func:`ascend_opt.telemetry.hardware_env_candidates` 派生，
    而不是在这里另写一遍：运行时判定、落盘与事后校验必须看到**同一份**指纹。
    两边顺序不同就会出现"判定时读到 A、导出时读到 B"的产物，而这种不一致
    只会在有人拿两处结论对照时才暴露。
    """
    candidates: list[Path] = []
    for rid in run_ids:
        if not rid:
            continue
        for path in hardware_env_candidates(raw_dir / rid):
            if path not in candidates:
                candidates.append(path)
    return candidates


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """把一个 JSON 文件解析成字典。返回 ``(对象, 错误说明)``。

    解析失败与"顶层不是对象"合并成同一条错误：对调用方来说两者都是
    "读不出结构"，区分开只会多一个没人用的分支。
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"顶层不是 JSON 对象（是 {type(payload).__name__}）"
    return payload, None


def _jsonl_scan(path: Path) -> tuple[int, int, str | None]:
    """逐行解析 JSONL。返回 ``(有效行数, 首个坏行号, 错误说明)``。

    为什么必须逐行解析而不是只看文件大小：``requests.jsonl`` 是汇总表里
    **每一个分位数的唯一来源**。一次中断完全可能留下"字节数非零、内容却截断"
    的文件——大小检查放行，下游读到 0 条记录，于是"有数据"悄悄变成"没数据"。
    """
    good = 0
    try:
        with path.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    return good, lineno, f"第 {lineno} 行不是合法 JSON：{exc}"
                good += 1
    except (OSError, UnicodeDecodeError) as exc:
        return good, 0, f"读取失败：{type(exc).__name__}: {exc}"
    return good, 0, None


def _snapshot_error(path: Path) -> str | None:
    """配置快照必须能解析成映射。返回错误说明，``None`` = 合格。

    快照正常是 YAML，但在没有 PyYAML 的机器上会退化成 JSON
    （见 ``write_run_outputs``）。统一走 YAML 解析即可——JSON 是 YAML 的子集。
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    try:
        import yaml

        payload = yaml.safe_load(text)
    except ImportError:  # pragma: no cover - 取决于运行环境是否有 PyYAML
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            return f"JSON 解析失败：{exc}"
    except Exception as exc:  # yaml.YAMLError 及其它
        return f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return f"顶层不是映射（是 {type(payload).__name__}）"
    return None


def _meta_problems(
    meta_path: Path, rid: str, row: dict[str, Any], *, lenient: bool
) -> list[Problem]:
    """校验一个 run 的 ``meta.json`` **内容**：可解析、键齐、身份自洽。"""
    meta, err = _load_json_object(meta_path)
    if err is not None:
        return [Problem(
            "meta_unreadable", BLOCK,
            f"run {rid} 的 {META_FILENAME} 无法解析：{err}",
        )]

    out: list[Problem] = []

    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        out.append(Problem(
            "meta_incomplete", BLOCK,
            f"run {rid} 的 {META_FILENAME} 缺少必需键：{', '.join(missing)}",
        ))

    # 身份自洽：目录名 / 汇总行 / meta.context.run_id 必须指同一个 run。
    # 一致才谈得上"这一行数字来自这个 run 目录"。
    ctx = meta.get("context") if isinstance(meta.get("context"), dict) else {}
    in_meta = str(ctx.get("run_id") or meta.get("run_id") or "").strip()
    if in_meta and in_meta != rid:
        out.append(Problem(
            "meta_run_id_mismatch", BLOCK,
            f"run 目录名是 {rid}，但其中的 {META_FILENAME} 记着 "
            f"run_id={in_meta}——目录被改名或复制过，追溯链已断",
        ))

    # 标签自洽：汇总行自报的标签必须与产物自报的标签一致。
    # 不一致意味着同一行数字有两套身份——正是"看起来总是对"的典型来源。
    row_label = str(row.get("data_label") or "").strip()
    meta_label = str(meta.get("data_label") or "").strip()
    if row_label and meta_label and row_label != meta_label:
        level = WARN if (lenient or not _is_measured(row)) else BLOCK
        out.append(Problem(
            "meta_label_mismatch", level,
            f"run {rid} 的汇总行标签是 {row_label}，但 {META_FILENAME} 自报 "
            f"{meta_label}——同一行数字有两套身份，必须查清以哪个为准",
        ))

    return out


def _cell_text(cell: str) -> str:
    """去掉 markdown **内联装饰**（反引号 / 强调星号）后的单元文本。

    为什么必须去：`scripts/emit_experiment_rows.py` 生成的 run_id 单元写作
    `` `run-id` ``（带反引号），而判定是**逐字相等** ⇒ **已登记的批次被判成"未登记"**。
    实测：`S14B1` 的 32 行都在表里，检查仍报"4/4 未登记"。

    ⚠️ **不要**去下划线 —— run_id 自己就含 `_`（如 `fixed_cost`），去了会把
    `fixed_cost` 变成 `fixedcost`，那是另一种静默错配。
    """
    return cell.replace("`", "").replace("*", "").strip()


def registered_in_table(text: str, run_id: str) -> bool:
    """``run_id`` 是否作为**表格单元**出现在 markdown 表里。

    刻意不用 ``run_id in text``：那会把散文里提一嘴也算成"已登记"。
    规范要求的是每个进结论的数字都能指向一条登记**行**（含 run_id、重复次数、
    离散度），而"文中出现过"给不出后两者。

    单元比较前先过 `_cell_text()` 去内联装饰 —— 但**仍然要求它落在表格里**：
    "去装饰"不等于"放宽到全文包含"。

    ★ 公开名：`scripts/check_traceability.py` 也用它做**登记覆盖**的自查。
    刻意**只有一份实现** —— 若那边另写一套"包含即可"的判定，就会出现两套口径，
    而松的那套会静默放行（同一批 run 在导出闸门处被拦、在日常自查里却绿灯）。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [_cell_text(c) for c in stripped.strip("|").split("|")]
        if run_id in cells:
            return True
    return False


#: 旧名（私有）保留为别名：外部脚本/笔记里可能还按旧名引用。
#: 新代码请用 `registered_in_table`。
_registered_in_table = registered_in_table


def _env_content_problems(
    env_path: Path,
    *,
    level: str,
    require_available: bool,
    payload: dict[str, Any] | None = None,
) -> list[Problem]:
    """校验一份环境指纹的**内容**：可解析、且显式声明了硬件可用性。

    ``payload`` 允许调用方把已读到的内容传进来，让"同一份指纹"在一次检查里
    只被读一次：读两遍就可能出现"检查可用性时读到 A、检查平台时读到 B"
    （文件在两读之间被替换），而这种不一致没有任何地方会报错。

    刻意不替它推导 ``available``：推导是采集端的职责（``telemetry``）。
    导出闸门只认**显式声明**——一份没写可用性的指纹，无法证明跑它的那台机器上
    真有 NPU，也就不该给 MEASURED 行背书。

    "没有可用性"有三种，报错必须分开，否则读的人会去改错的地方：

    * **字段不存在** —— 旧版采集脚本（``schema_version`` 缺失或 < 2）不写这个
      字段。修法是重跑采集脚本覆盖旧文件。
    * **``available=null``** —— 三态里的"探测跑过但没得出结论"。这**不是**漏写
      字段，而是有意义的取值；修法是修好探测工具后重采，绝不能改成 ``False``
      蒙过去（那就把"没查出来"谎报成"查过、没有"）。
    * **``available=false``** —— 确实没有可用设备。修法是换机器或降级标签。

    判定用 ``is not True`` 而不是真值测试：``available`` 只接受 JSON ``true``。
    用真值测试会让 ``"yes"``、``1`` 这类值静默通过——一份**没有任何字段是布尔
    可用性**的指纹，就这样给 MEASURED 行背了书。
    """
    payload, err = _load_json_object(env_path) if payload is None else (payload, None)
    if err is not None:
        return [Problem(
            "env_unreadable", level,
            f"环境指纹 {env_path} 无法解析：{err}",
        )]

    if "available" not in payload:
        schema = payload.get("schema_version")
        provenance = (
            f"该文件声明 schema_version={schema}，是旧格式"
            if isinstance(schema, int)
            else "该文件未声明 schema_version，视为旧格式"
        )
        return [Problem(
            "env_does_not_declare_availability", level,
            f"环境指纹 {env_path} 未显式声明 available（{provenance}；"
            f"现有键：{sorted(payload)[:8]}）；无法据此确认取数机器上硬件可用。"
            f"修法：在**执行推理的那台机器**上重新运行 scripts/inspect_env.sh，"
            f"让它直接写出平台与可用性",
        )]

    available = payload.get("available")
    if require_available and available is not True:
        where = _platform_hint(payload)
        if available is None:
            # 三态里的 None。与 False 的修法完全不同，必须分开说。
            why = (
                "available=null 表示探测**跑过但没得出结论**"
                "（命令报错，或输出格式不认识），不是'确认没有硬件'。"
                "修法：修好设备探测工具后重采；不要把它改成 false——"
                "那是把'没查出来'谎报成'查过、没有'"
            )
        else:
            why = (
                f"该机器明确声明没有可用加速设备。修法：换到有设备的机器上重跑，"
                f"或按规范第 10 节降级数据标签（None/False 都不构成硬件实测证据）"
            )
        return [Problem(
            "env_declares_unavailable", level,
            f"环境指纹 {env_path}{where} 声明 available={available!r}，"
            f"但汇总表里存在 MEASURED 行——这些数字不能作为硬件实测结果。{why}",
        )]
    return []


def _platform_hint(payload: dict[str, Any]) -> str:
    """把指纹里声明的平台拼成一句可嵌进报错的话；读不到就返回空串。

    报错里点名平台是**必要**的：同一句"取数机器未声明硬件可用"，
    对应的机器可能是昇腾也可能是 MTT，而修法（去哪个主机上重采）完全不同。
    未声明平台时**不点名**——凭空挑一个平台会把读的人引到错误的机器上。
    """
    value = payload.get("platform")
    if isinstance(value, str) and value.strip():
        return f"（platform={value.strip()}）"
    return ""


def check_export(
    *,
    raw_dir: Path,
    processed_dir: Path,
    experiments_md: Path,
    lenient: bool = False,
) -> list[Problem]:
    """跑完四项检查，返回发现列表（空列表 = 可以导出）。"""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    experiments_md = Path(experiments_md)

    problems: list[Problem] = []
    rows = _read_csv(processed_dir / "summary.csv")
    repeats = _read_csv(processed_dir / "repeats.csv")

    if not rows:
        # 汇总表为空时不在这里报错：export_results.sh 已经用更准确的措辞报过
        # "没有可导出的 run"（那里知道是标签过滤导致的）。重复报错只会让
        # 真正有用的第一条信息被淹没。
        return problems

    all_ids = [(r.get("run_id") or "").strip() for r in rows]
    # 需要可追溯的 run：严格模式下 = MEASURED 行；--lenient 下没有行需要，
    # 于是后三项全部落到 warn 一级，但仍会逐条打印。
    trace_ids = [] if lenient else [rid for rid, r in zip(all_ids, rows) if _is_measured(r)]

    # ---- 检查 1：原始明细的内容（任何模式下都阻断）----
    # 三样东西缺一不可：逐请求明细（数字的来源）、meta（身份与标签）、
    # 配置快照（当时的配置）。三者都是"缺了不会报错"的典型——
    # 表照样生成、图照样画出来，所以必须逐 run 校验**内容**，而不只是存在性。
    for rid, row in zip(all_ids, rows):
        if not rid:
            problems.append(Problem(
                "raw_detail_missing", BLOCK,
                "汇总表里有一行没有 run_id，无法定位原始明细",
            ))
            continue
        run_path = raw_dir / rid
        if not run_path.is_dir():
            problems.append(Problem(
                "raw_detail_missing", BLOCK,
                f"run {rid} 的原始目录不存在：{run_path}",
            ))
            continue

        detail = run_path / DETAIL_FILENAME
        if not detail.exists() or detail.stat().st_size == 0:
            problems.append(Problem(
                "raw_detail_missing", BLOCK,
                f"run {rid} 缺少逐请求明细（{detail} 不存在或为空）",
            ))
        else:
            good, bad_line, err = _jsonl_scan(detail)
            if err is not None:
                problems.append(Problem(
                    "detail_unreadable", BLOCK,
                    f"run {rid} 的逐请求明细无法解析：{err}",
                ))
            elif good == 0:
                problems.append(Problem(
                    "detail_unreadable", BLOCK,
                    f"run {rid} 的逐请求明细没有任何有效记录（全是空白行）",
                ))

        meta = run_path / META_FILENAME
        if not meta.exists():
            problems.append(Problem(
                "raw_detail_missing", BLOCK,
                f"run {rid} 缺少 {META_FILENAME}，无法复核配置与数据标签",
            ))
        else:
            problems.extend(_meta_problems(meta, rid, row, lenient=lenient))

        snapshot = run_path / SNAPSHOT_FILENAME
        if not snapshot.exists():
            problems.append(Problem(
                "config_snapshot_missing", BLOCK,
                f"run {rid} 缺少 {SNAPSHOT_FILENAME}，事后无法复核当时的配置"
                f"（configs/ 会随代码演进而变）",
            ))
        else:
            snap_err = _snapshot_error(snapshot)
            if snap_err is not None:
                problems.append(Problem(
                    "config_snapshot_unreadable", BLOCK,
                    f"run {rid} 的 {SNAPSHOT_FILENAME} 无法解析：{snap_err}",
                ))

    # ---- 检查 2：环境指纹（存在 + 内容 + **声明与检出的平台是否一致**）----
    candidates = _fingerprint_candidates(raw_dir, all_ids)
    found_env = next((p for p in candidates if p.exists()), None)
    if found_env is None:
        problems.append(Problem(
            "env_fingerprint_missing", BLOCK if trace_ids else WARN,
            "未找到环境指纹（查过："
            + ", ".join(str(p) for p in candidates)
            + "）；性能数字只在采集它的那台机器上可解释，"
              "请先运行 scripts/inspect_env.sh",
        ))
    else:
        payload, env_err = _load_json_object(found_env)
        problems.extend(_env_content_problems(
            found_env,
            level=BLOCK if trace_ids else WARN,
            require_available=bool(trace_ids),
            payload=payload,
        ))
        # 指纹读不出内容时不再往下查平台证据：`_env_content_problems` 已经报了
        # "无法解析"，此处再报一遍会把第一现场埋在第二条同样的话下面。
        if env_err is None and payload is not None:
            problems.extend(_platform_evidence_problems(
                found_env, payload, raw_dir, rows, lenient=lenient,
            ))

    # ---- 检查 3：有效重复（按**落盘的成员清单**核对）----
    # 成员从 repeats.csv 的 member_run_ids 读，而不是在这里按 run_id 前缀重推。
    # 分组定义（含平台与执行后端）**只有一份**，在 ascend_opt.grouping；
    # 导出侧再推一遍就是在两处维护同一条规则。已经踩过：同一批次被拆成两组时，
    # 前缀推法把两组的成员一起数进来（3 + 3 = 6），于是报出"自报 3、实为 6"的
    # 假不一致——它不仅本身是错的，还会把真正的问题（同一批次出现两组）盖住。
    #
    # 但"不信自报旗标"这条纪律仍然成立，落地方式改成：**不读** sufficient_repeats，
    # 而是拿成员清单与汇总表里实际存在的 run_id 求交集来重新数。
    groups: list[dict[str, Any]] = []
    if not repeats:
        problems.append(Problem(
            "repeats_missing", BLOCK if trace_ids else WARN,
            f"{processed_dir / 'repeats.csv'} 不存在或为空，无法核对重复次数",
        ))
    else:
        present_ids = set(all_ids)
        seen_groups: set[str] = set()
        for group in repeats:
            key = (group.get("group") or "").strip()
            base = (group.get("run_id_base") or "").strip()
            if not key:
                problems.append(Problem(
                    "repeats_group_missing", BLOCK if trace_ids else WARN,
                    "repeats.csv 有一行没有 group 列，无法判断它属于哪一组"
                    "（分组键已含平台与执行后端，仅凭 run_id_base 不足以定位）",
                ))
                continue
            if key in seen_groups:
                # 同一组出现两次是聚合产物内部重复，不是"两次重复运行"。
                continue
            seen_groups.add(key)

            raw_members = (group.get("member_run_ids") or "").strip()
            if not raw_members:
                problems.append(Problem(
                    "repeats_members_missing", BLOCK if trace_ids else WARN,
                    f"组 {key} 没有 member_run_ids 列（旧版聚合产物）。"
                    f"导出侧不再按 run_id 前缀自行推导成员——那会重复一遍分组规则，"
                    f"并可能把被拆开的两组当成一组。请重新运行 "
                    f"benchmark/collect_metrics.py 生成带成员清单的 repeats.csv",
                ))
                continue

            claimed_ids = [m.strip() for m in raw_members.split(";") if m.strip()]
            level = BLOCK if any(m in trace_ids for m in claimed_ids) else WARN
            missing_members = [m for m in claimed_ids if m not in present_ids]
            if missing_members:
                problems.append(Problem(
                    "repeats_member_unknown", level,
                    f"组 {key} 声明的成员 {missing_members} 不在汇总表里——"
                    f"派生文件滞后于原始明细，或 raw 已被清理",
                ))

            derived = sum(1 for m in claimed_ids if m in present_ids)
            groups.append({"key": key, "base": base, "members": claimed_ids})

            if derived < MIN_REPEATS_FOR_CONCLUSION:
                problems.append(Problem(
                    "insufficient_repeats", level,
                    f"组 {key} 的成员清单里只有 {derived} 个 run 出现在汇总表里"
                    f"（清单共 {len(claimed_ids)} 个，自报 n_repeats="
                    f"{group.get('n_repeats', '?')}，< {MIN_REPEATS_FOR_CONCLUSION}），"
                    f"离散度不具统计意义，不得用于结论",
                ))

            declared_count = str(group.get("n_repeats", "")).strip()
            if declared_count and declared_count != str(len(claimed_ids)):
                problems.append(Problem(
                    "repeats_count_mismatch", level,
                    f"组 {key} 自报 n_repeats={declared_count}，"
                    f"但同一行的成员清单里有 {len(claimed_ids)} 个 run——"
                    f"派生文件自身就不自洽",
                ))

        # 反方向的滞后：某个 run 属于已登记的批次，却没有任何一组认领它。
        # 只数"已认领的成员"发现不了漏掉的那些——而漏掉一个 run 会让重复次数
        # 看起来更少（更保守）或更多（按批次数的旧实现），两种都错。
        accounted = {m for g in groups for m in g["members"]}
        bases_with_groups = {g["base"] for g in groups if g["base"]}
        for rid in all_ids:
            if not rid or rid in accounted:
                continue
            base = base_run_id(rid)
            if base and base in bases_with_groups:
                problems.append(Problem(
                    "repeats_member_unaccounted", WARN,
                    f"run {rid} 属于批次 {base}，却没有被该批次的任何一组列入"
                    f"成员清单（{sorted(g['key'] for g in groups if g['base'] == base)}）"
                    f"——聚合可能漏掉了它，重新运行 benchmark/collect_metrics.py 核对",
                ))

    # ---- 检查 4：实验登记（必须是真实表格行）----
    if not experiments_md.exists() or experiments_md.stat().st_size == 0:
        problems.append(Problem(
            "experiments_table_missing", BLOCK if trace_ids else WARN,
            f"未找到实验登记表：{experiments_md}；没有它就无法核对哪些数字已登记",
        ))
    else:
        text = experiments_md.read_text(encoding="utf-8")
        bases = sorted({g["base"] for g in groups if g["base"]})
        for base in bases:
            if registered_in_table(text, base):
                continue
            # 登记是按**批次**的：同批次被拆成两组（不同平台/后端）也只登记一次。
            # 只要任一分组的成员里有需要追溯的行，这一条就升级为阻断。
            member_needs_trace = any(
                rid in trace_ids
                for g in groups if g["base"] == base
                for rid in g["members"]
            )
            problems.append(Problem(
                "experiment_not_registered", BLOCK if member_needs_trace else WARN,
                f"run_id {base} 未作为登记表的**表格单元**出现；"
                f"规范第 2 条要求进结论的数字必须有一条登记行"
                f"（含 run_id、重复次数、离散度）——"
                f"只在正文里提到不算登记",
            ))

    return problems


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="导出前置校验：原始明细 / 环境指纹 / 有效重复 / 实验登记",
    )
    p.add_argument("--raw", default=None, help="原始数据目录，默认 results/raw")
    p.add_argument("--processed", default=None, help="派生数据目录，默认 results/processed")
    p.add_argument("--experiments", default=None, help="实验登记表，默认 EXPERIMENTS.md")
    p.add_argument(
        "--lenient",
        action="store_true",
        help="不把任何行当成需要可追溯（用于只做链路口径核对的诊断包）；"
             "可追溯性检查降为提示，缺原始明细仍为阻断",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_dir = Path(args.raw) if args.raw else _ROOT / "results" / "raw"
    processed_dir = Path(args.processed) if args.processed else _ROOT / "results" / "processed"
    experiments = Path(args.experiments) if args.experiments else _ROOT / "EXPERIMENTS.md"

    problems = check_export(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        experiments_md=experiments,
        lenient=args.lenient,
    )

    blocking = [p for p in problems if p.level == BLOCK]
    warning = [p for p in problems if p.level == WARN]

    for p in warning:
        print(p.render())
    for p in blocking:
        print(p.render(), file=sys.stderr)

    if blocking:
        print(
            f"\n前置校验未通过：{len(blocking)} 项阻断，导出中止。\n"
            f"这些项的共同点是「缺了不会报错」——表照样生成、图照样画出来、\n"
            f"导出包照样写着「完成」。所以必须在这里被挡住，"
            f"而不是等到被质疑时才发现。",
            file=sys.stderr,
        )
        return 4

    print(f"前置校验通过（{len(warning)} 项提示）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
