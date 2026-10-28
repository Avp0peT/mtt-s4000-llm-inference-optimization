#!/usr/bin/env python3
"""D1 可行性闸门 —— **三态**判定（`PASS` / `FAIL` / `INCONCLUSIVE`）。

为什么要有这一版（2026-09-27 的实修）
--------------------------------------
旧闸门（`run_s14_d1.sh` 里的一段内联 heredoc）在 `S14G` 上报了两条**假失败**，
根因是**臂名解析有第三份实现**：

* 它把目录名用"字符串替换"解析成 **`fixed_cost_pilot`**；
* 而 `diagnose_scheduling.py` 用正则解析成 **`cost_pilot`**。

⇒ 闸门在自己的 `per` 字典里查 `cost_pilot` 查到**空列表**，于是打印
`n_runs: 0, small_steps: 0, n_degraded: [], has_arm_state: False`，并据此报出：

1. `条件1：小 chunk(128/256) 一步都没被实际执行`（**假**：同批日志里
   `fixed_cost_pilot` 打印 `chunk 生效=128..1024`）；
2. `条件2：arm_state/n_degraded 数不出来（源码未部署到位）`（**假**：
   `meta.json.controller_effect.arm_state` 就在产物里，`n_degraded: 34`）。

**三条设计原则**（也就是这次要修的东西）：

1. **解析只有一处**：`ascend_opt.run_ids.parse_run_id`。解析失败 ⇒ `INCONCLUSIVE`，
   **不得回退成猜出来的臂名**。
2. **先验有效性，再判成败**：每臂的**重复数 / 数据标签 / 指纹 / 必要字段**先查；
   缺一样就 `INCONCLUSIVE`。**只有有效证据才能判 `PASS`/`FAIL`。**
3. **触发只能来自本次运行的结构化结果**：本脚本把判定写进 `--json`，并用**退出码**表达状态；
   调用方（shell）**只许看退出码**，**禁止**去 grep 累计日志里的 `*_GATE_PASS`
   —— 那是 append-only 日志，历史一次 PASS 会把后续每一次都误触发。

用法::

    python scripts/d1_gate.py --tag S14G --arm cost_pilot --ref-arm cost_min512 \\
        --criteria v1 --json results/processed/d1_gate_S14G.json

退出码：`0` PASS｜`1` FAIL｜`2` INCONCLUSIVE｜`3` 输入错（参数/路径/产物不可读）。
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.run_ids import parse_run_id  # noqa: E402

PASS, FAIL, INCONCLUSIVE, BAD_INPUT = 0, 1, 2, 3

#: `meta.json` 里**缺一即不可判**的字段（"缺指纹"就是后四条）。
#: ⚠️⚠️ **位置必须按真实产物来，不能凭想象**（2026-09-27 用真产物核对后更正）：
#:   `execution_environment` / `hardware_platform` 在 `meta.json` 里位于
#:   **`reproducibility.*` 与 `config.runtime.*`（两处都有）**，
#:   **不在顶层**；`label_source` 则**根本不在 `meta.json` 里**，它是聚合层
#:   （`summary.csv`）的派生列 —— 其取值规则是"产物自带 `data_label` ⇒ `explicit`"。
#: 第一版把这三个当顶层键读，实测会对**每一个合法 `MEASURED` 产物**都判 `INCONCLUSIVE`
#: —— 那是我自己造出来的**假阴性**，与要修的那个假失败同源（都源于"按想象而不是按产物写解析"）。
REQUIRED_META_NOTE = ("execution_environment / hardware_platform 读 `reproducibility` 优先、"
                      "回落 `config.runtime`；label_source 由 data_label 是否自带推出")

#: 旧判据（v1）的三个阈值。**保留原文，不改数字** —— 修订走 v2，且必须写清理由。
V1_D0_FEASIBLE_MIN = 0.057

#: ---- v2 判据（**机制诊断档**，2026-09-27 冻结）--------------------------------
#: 定位：v2 **不判服务收益**。第 3 步查明 S14G 落在**过载区**
#: （模型预测步成本 ~650ms（无余量）vs 每步预算 120ms ⇒ 5.4× 超支，
#:  且**不是**余量造成的，余量只贡献 16%）⇒ 按用户预案：
#: "如果新批次仍处于所有方案 goodput 为零的过载区域，将它用于机制诊断；
#:  服务收益另用预先登记的 SLO 可达负载验证。"
#: ⇒ v2 只判**机制是否按设计工作**：模型有没有被用上、动作有没有执行、
#:   降级与回退是否被如实计数、预测误差是否可算。
#:
#: 与 v1 的差别（**修订理由，逐条**）：
#: 1. v1 条件3 的阈值 `5.7%` 取自 **S14D（另一个模型）**⇒ 跨模型借用阈值。
#:    v2 改为**同批参照臂**比较（用户指令第 4 步逐字要求"以同批参照比较"）。
#: 2. v1 只看降级**次数**（`t >= r`）⇒ 忽略分母。v2 连报 `n_degraded/n_decisions`
#:    与 `n_fallback/n_decisions`，并要求**两个分母都非零**才判。
#: 3. v1 没有**覆盖率准入**。v2 加：决策级覆盖率 < 50% ⇒ `INCONCLUSIVE`
#:    （模型没被用上时，任何"收益"数字都不是在验这个机制）。
#:    ⚠️ 阈值的依据：第 3 步实测真实服务形状覆盖 **53.57%**（`b4=100%`，
#:    而 `b5=b6=0%`）⇒ 50% 是"与已知部署点同级"的最低线，非拍脑袋。
V2_MIN_DECISION_COVERAGE = 0.50


def criteria_v2(per_arm: dict[str, dict[str, Any]], arm: str, ref: str,
                d0: dict[str, Any], cov: dict[str, Any]) -> tuple[list[str], list[str]]:
    """**机制诊断档判据**（2026-09-27 冻结）。返回 (fails, notes)。

    判什么、不判什么
    ----------------
    ✅ 判：候选**被执行**（有可行候选且选中）、降级/回退**被计数**（分母非零）、
          预测误差可算、覆盖率过准入线。
    ❌ 不判：D3 服务收益（本批在过载区，goodput 恒 0 ⇒ 判它等于判"负载不可达"，
          不是判机制）。
    """
    fails: list[str] = []
    notes: list[str] = []
    t, r = per_arm.get(arm) or {}, per_arm.get(ref) or {}

    # --- 准入：模型必须真的被用上（决策级覆盖率）---------------------------
    c = (cov or {}).get("coverage")
    if c is None:
        fails.append("准入：模型覆盖率数不出来（轨迹里没有 cost_policy 候选）⇒ 不可判")
    elif c < V2_MIN_DECISION_COVERAGE:
        fails.append(f"准入：决策级模型覆盖率 {c:.2%} < {V2_MIN_DECISION_COVERAGE:.0%}"
                     f"（覆盖 {cov.get('n_covered')}/{cov.get('n_decisions')}）"
                     " ⇒ 大量决策走回退，此时任何收益数字都不是在验这个机制")
    else:
        notes.append(f"准入过：决策级模型覆盖率 {c:.2%}"
                     f"（{cov.get('n_covered')}/{cov.get('n_decisions')}）"
                     f" ≥ {V2_MIN_DECISION_COVERAGE:.0%}")

    # --- 机制1：模型的动作**真的被执行**（v1 条件1 的加强版：不要求"小 chunk"，
    #     只要求"有被选中的可行候选" —— 因为机制诊断档不预设 chunk 大小）--------
    if (t.get("small_steps") or 0) <= 0 and (t.get("n_pf_steps") or 0) <= 0:
        fails.append(f"机制1：{arm} 一步 prefill 都没被实际执行"
                     f"（n_steps={t.get('n_steps')}，min_pf={t.get('min_pf_seen')}）")
    else:
        notes.append(f"机制1 过：{arm} 实执行 prefill {t.get('n_pf_steps')} 步"
                     f"（其中小 chunk(128/256) {t.get('small_steps')} 步，"
                     f"min_pf={t.get('min_pf_seen')}）")

    # --- 机制2：选中的动作**确实是被采纳的**（D0 可行数 > 0）----------------
    nf = (d0 or {}).get("n_feasible")
    if not nf:
        fails.append(f"机制2：{arm} 没有任何可行候选被采纳（n_feasible={nf}）"
                     " ⇒ 判据把全部候选拒掉了，机制无从体现")
    else:
        notes.append(f"机制2 过：采纳可行候选 {nf} 个"
                     f"（拒因 top1 = {(d0.get('top_reject_reasons') or [{}])[0].get('example', '—')[:60]}）")

    # --- 机制3：降级与回退**被如实计数**（连报分母，v1 的"只比次数"已废弃）----
    for label, key in (("降级", "n_degraded"), ("回退", "n_fallback")):
        v = t.get(key)
        if v is None:
            fails.append(f"机制3：{arm} 的 `arm_state.{key}` 数不出来 ⇒ 不可判")
    if t.get("n_decisions"):
        deg = f"{t.get('n_degraded')}/{t.get('n_decisions')}"
        fb = f"{t.get('n_fallback')}/{t.get('n_decisions')}"
        notes.append(f"机制3 过：降级 {deg}={_pct(t.get('degraded_rate'))}、"
                     f"回退 {fb}（**连报分母**；不与参照比大小 —— "
                     "过载区里两臂的分母都接近满载，比大小无意义）")
    else:
        fails.append(f"机制3：{arm} 的 `arm_state.n_decisions` 为 0/缺失 ⇒ 分母不可得")

    # --- 机制4：预测误差**可算**（否则"机制成立"无从量化）-------------------
    if (d0 or {}).get("feasible_rate") is None:
        fails.append("机制4：D0 可行率数不出来 ⇒ 预测误差不可评估")
    else:
        notes.append(f"机制4 过：D0 可行率 {d0['feasible_rate']:.2%}"
                     f"（{d0.get('n_feasible')}/{d0.get('n_candidates')} 候选）"
                     "—— 该值低说明**落在过载区**，不判为失败（见 v2 定位）")

    notes.append("⚠️ v2 = **机制诊断档**：本判据**不判服务收益**。"
                 "D3 goodput 恒 0 的批次必须另用预注册的 SLO 可达负载验证收益。")
    return fails, notes



@dataclass
class RunFacts:
    dir: str
    run_id: str
    arm: str | None
    pass_index: int | None
    parse_ok: bool
    meta_ok: bool
    #: 产物的 `config.config_stem` —— **臂名的唯一权威来源**（见 `load_runs` 内注释）。
    config_stem: str | None = None
    meta_error: str | None = None
    telemetry_ok: bool = False
    n_steps: int = 0
    n_pf_steps: int = 0
    n_small_steps: int = 0
    min_pf: int | None = None
    arm_state: dict[str, Any] | None = None
    data_label: str | None = None
    label_source: str | None = None
    execution_environment: str | None = None
    hardware_platform: str | None = None
    git_commit: str | None = None
    git_source: str | None = None
    git_dirty: bool | None = None
    problems: list[str] = field(default_factory=list)


def _dirs(raw_root: Path, tag: str) -> list[Path]:
    return [Path(d) for d in sorted(glob.glob(str(raw_root / f"{tag}-*")))]


def _small_steps_from_telemetry(tel_path: Path) -> dict[str, Any]:
    """从遥测的 `note` 里数 prefill 事件。

    ⚠️ prefill 的实际执行**只能从 `note` 读**（汇总表里没有这一列）。读不到就报
    `telemetry_ok=False`，**不得填 0** —— 0 是"测到一步都没发"，缺失是"没测"。
    """
    if not tel_path.exists():
        return {"ok": False, "n_steps": 0, "n_pf": 0, "n_small": 0, "min_pf": None}
    try:
        tel = [json.loads(l) for l in tel_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    except Exception:
        return {"ok": False, "n_steps": 0, "n_pf": 0, "n_small": 0, "min_pf": None}
    pf: list[int] = []
    for t in tel:
        m = re.search(r"prefill_tok=(\d+)", str(t.get("note") or ""))
        if m:
            pf.append(int(m.group(1)))
    return {"ok": bool(tel), "n_steps": len(tel),
            "n_pf": sum(1 for v in pf if v > 0),
            "n_small": sum(1 for v in pf if v in (128, 256)),
            "min_pf": (min(pf) if pf else None)}


def load_runs(raw_root: Path, tag: str) -> list[RunFacts]:
    """读一个批次的产物，**逐 run 收集事实**（不做任何判定）。"""
    out: list[RunFacts] = []
    for d in _dirs(raw_root, tag):
        meta_path = d / "meta.json"
        facts = RunFacts(dir=d.name, run_id=d.name, arm=None, pass_index=None, parse_ok=False,
                         meta_ok=False)
        if not meta_path.exists():
            facts.problems.append("缺 meta.json")
            out.append(facts)
            continue
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            facts.meta_error = f"{type(exc).__name__}: {exc}"
            facts.problems.append("meta.json 不可解析")
            out.append(facts)
            continue
        facts.meta_ok = True
        rid = str(((m.get("summary") or {}).get("run_id")) or d.name)
        facts.run_id = rid

        # ---- 臂名：★ **唯一权威 = 产物记录的 `config_stem`** ------------------
        # 契约（2026-09-28 用真产物 + 真源码核实，不是推断）：
        #   `scripts/run_step3_matrix.py:300` 写的是
        #       run_ids = {name: f"{stamp}-{cfg.workload_name}-{name}"}
        #   其中 `name` 是 **cell 名**，而 cell 名**就是 config stem**
        #   （`benchmark/run_benchmark.py:1282` 的 `build_run_id(workload, cfg.stem)` 同理）。
        #   ⇒ run_id 的尾部是 **`<config_stem>` 整体**，**不是** `<cell>_<arm>`。
        #
        # ⚠️ 而 `run_ids.py::RUN_ID_RE` 把尾部按 `<cell>_<arm>` 拆（`cell` 段还是
        #    `[a-z]+`）⇒ 对 `service_benefit_c2_m4_cost` 得到 arm=`benefit_c2_m4_cost`
        #    （真臂名 `cost`）—— **用了一个不存在的分隔契约**。
        #    实测：`parse_run_id` 在 6 个批次 42 个目录上与 `config_stem` 不一致。
        #
        # ⇒ 因此本脚本**不再用 `parse_run_id` 定臂名**，改读产物的 `config_stem`；
        #   `parse_run_id` 的结果**保留但降级**为「名字形状的自洽性核对」（见下）。
        cs = (m.get("config") or {}).get("config_stem")
        facts.config_stem = cs if isinstance(cs, str) and cs else None

        parsed = parse_run_id(rid)
        if parsed is None:
            facts.problems.append(f"run_id 解析失败（不合名字形状约定）：{rid}")
        else:
            facts.pass_index, facts.parse_ok = parsed.pass_index, True
            if facts.config_stem:
                # 交叉核对：名字尾部应当**逐字等于** config_stem。
                # ⚠️ **必须先剥掉 `-rN` 重复后缀** —— 否则每个重复目录都会误报
                #    "名字尾部 'r1' ≠ config_stem"（我第一版就踩了，9 个目录里 6 个假阳性）。
                tail = re.sub(r"-r\d+$", "", rid).rsplit("-", 1)[-1]
                if tail != facts.config_stem:
                    facts.problems.append(
                        f"名字尾部 {tail!r} ≠ 产物的 config_stem "
                        f"{facts.config_stem!r} ⇒ 命名契约分叉（以 config_stem 为准）")

        facts.data_label = m.get("data_label")

        # ⚠️ 三个"身世"字段的**真实位置**（用真产物核对过，见文件头注释）：
        repo = m.get("reproducibility") or {}
        rt = (m.get("config") or {}).get("runtime") or {}
        facts.execution_environment = (repo.get("execution_environment")
                                       or rt.get("execution_environment"))
        facts.hardware_platform = (repo.get("hardware_platform")
                                   or rt.get("hardware_platform"))
        # `label_source` 不在 meta 里：产物自带 `data_label` ⇒ `explicit`；
        # 缺标签的旧产物才走 `legacy_inferred`（而那种产物本就不该进判定）。
        facts.label_source = ("explicit" if m.get("data_label") is not None else None)
        facts.git_commit = repo.get("git_commit") or None
        facts.git_source = repo.get("git_source") or None
        facts.git_dirty = repo.get("git_dirty")

        tel = (d / "telemetry.jsonl")
        st = _small_steps_from_telemetry(tel)
        facts.telemetry_ok = st["ok"]
        facts.n_steps, facts.n_pf_steps = st["n_steps"], st["n_pf"]
        facts.n_small_steps, facts.min_pf = st["n_small"], st["min_pf"]

        ce = m.get("controller_effect") or {}
        facts.arm_state = ce.get("arm_state") if isinstance(ce.get("arm_state"), dict) else None
        if facts.arm_state is None:
            facts.problems.append("controller_effect.arm_state 缺失（降级/饥饿计数数不出来）")

        # ★ 臂名**最后**定（依赖上面已填好的 `config_stem`）。
        #   优先级：产物记录的 config_stem > 名字解析（降级为兜底）。
        #   取不到就**留 `None`**（`_per_arm` 会归到 `<未解析>`），**不填猜测值**。
        facts.arm = derive_arm_name(facts.config_stem)
        out.append(facts)
    return out


#: 本批次类型里出现过的"臂"后缀全集 —— **只用于从 `config_stem` 里认出臂后缀**。
#: ⚠️ 这是"认识名字"，不是"猜名字"：认不出就返回整个 stem（**不截断**）。
#: 新增批次若要新的臂名，**往这里加**（与批次的配置生成器同步）。
KNOWN_ARM_SUFFIXES = (
    "static", "threshold", "cost",          # S6 服务收益档三臂
    "cost_pilot", "cost_min512",            # S14G 机制诊断档两臂
)


def derive_arm_name(config_stem: str | None) -> str | None:
    """从**产物的 `config_stem`** 定臂名。

    ⚠️ **不按分隔符拆、不截断** —— 只做"整个 stem 是否就是某个已知臂名"或
    "stem 是否以 `_<已知臂名>` 结尾"两种判断；都不中就**原样返回 stem**。
    这条克制的理由：2026-09-28 实测 `parse_run_id` 因为**擅自按 `_` 拆**，
    把 `service_benefit_c2_m4_cost` 拆成 `benefit_c2_m4_cost`，
    在 6 个批次 42 个目录上给出了错的臂名 —— **错的臂名比长的臂名危险**
    （它不报错，只是让下游"查不到数据"）。
    """
    if not config_stem:
        return None
    for suffix in KNOWN_ARM_SUFFIXES:
        if config_stem == suffix or config_stem.endswith("_" + suffix):
            return suffix
    return config_stem


def _per_arm(runs: list[RunFacts]) -> dict[str, list[RunFacts]]:
    g: dict[str, list[RunFacts]] = {}
    for r in runs:
        g.setdefault(r.arm or "<未解析>", []).append(r)
    return g


def validity_problems(runs: list[RunFacts], arms: list[str], min_repeats: int,
                      diag_arms: set[str] | None = None) -> list[str]:
    """**判定之前**的有效性检查。返回空列表才允许进入判据。

    ⚠️ 两处都收：① `load_runs` 逐 run 记下的问题（缺 meta / 解析失败 / 缺 arm_state /
    缺 telemetry）；② 这里按字段与分组再查一遍。
    **只查②不查①**是踩过的坑：问题被记下来却没被用上，于是"解析失败"照样进了判据。
    """
    p: list[str] = []
    if not runs:
        return ["该 tag 下没有任何 run 目录 ⇒ 无证据"]

    for r in runs:
        for prob in r.problems:
            p.append(f"{r.dir}: {prob}")
        if not r.meta_ok:
            continue
        if not r.parse_ok:
            p.append(f"{r.dir}: run_id 解析异常（回退猜测被禁止）")
        if not r.telemetry_ok:
            p.append(f"{r.dir}: telemetry 缺失或不可解析 ⇒ prefill 实况数不出来")
        if r.data_label != "MEASURED":
            p.append(f"{r.dir}: data_label={r.data_label!r}（判定只接受 MEASURED）")
        if r.label_source != "explicit":
            p.append(f"{r.dir}: label_source={r.label_source!r}（必须是 explicit；"
                     "legacy_inferred 只允许降级）")
        for k in ("execution_environment", "hardware_platform"):
            v = getattr(r, k)
            if v in (None, "", "unknown"):
                p.append(f"{r.dir}: {k}={v!r} ⇒ 缺指纹，无法确认推理跑在哪台机器/哪个平台")
        if not r.git_commit or r.git_commit == "NOT_MEASURED":
            p.append(f"{r.dir}: git_commit={r.git_commit!r} ⇒ 无源码版本，不可追溯")
        if r.git_dirty is True:
            p.append(f"{r.dir}: git_dirty=True ⇒ 跑的不是一个提交，"
                     "该类批次不得与其它批次合并解释（也就不能进判定）")

    by_arm = _per_arm(runs)
    for a in arms:
        rs = by_arm.get(a) or []
        if not rs:
            p.append(f"臂 {a!r} 在产物里不存在（现有臂：{sorted(by_arm)}）⇒ "
                     "要么这个臂没跑，要么**臂名解析与另一处不一致**")
        elif len(rs) < min_repeats:
            p.append(f"臂 {a!r} 只有 {len(rs)} 次重复 < 要求的 {min_repeats} ⇒ 不出结论")
    if diag_arms is not None:
        for a in arms:
            if a not in diag_arms:
                p.append(f"臂 {a!r} 出现在产物里，但诊断产物（D0/D1/D2/D3）没有它 ⇒ "
                         f"解析口径不一致（诊断里的臂：{sorted(diag_arms)}）")
    return p


def arm_metrics(runs: list[RunFacts], arm: str) -> dict[str, Any]:
    """把某臂的事实汇成**可读的量**（含分母）。判定与报告都用这一份。"""
    rs = _per_arm(runs).get(arm) or []
    deg = [(r.arm_state or {}).get("n_degraded") for r in rs]
    dec = [(r.arm_state or {}).get("n_decisions") for r in rs]
    starved = [(r.arm_state or {}).get("n_prefill_starved") for r in rs]
    fallback = [(r.arm_state or {}).get("n_fallback") for r in rs]

    def _s(vals):
        v = [x for x in vals if isinstance(x, (int, float))]
        return sum(v) if v else None
    d_sum, dec_sum = _s(deg), _s(dec)
    return {
        "n_runs": len(rs),
        "runs": [r.run_id for r in rs],
        "small_steps": sum(r.n_small_steps for r in rs),
        "min_pf_seen": sorted({r.min_pf for r in rs if r.min_pf is not None}),
        "n_pf_steps": sum(r.n_pf_steps for r in rs),
        "n_steps": sum(r.n_steps for r in rs),
        # 降级次数**必须连报分母**（"34 次降级"离开"37 次决策"没有意义）
        "n_degraded": d_sum,
        "n_decisions": dec_sum,
        "degraded_rate": (d_sum / dec_sum) if (d_sum is not None and dec_sum) else None,
        "n_prefill_starved": _s(starved),
        "n_fallback": _s(fallback),
    }


def criteria_v1(per_arm: dict[str, dict[str, Any]], arm: str, ref: str,
                d0: dict[str, Any]) -> tuple[list[str], list[str]]:
    """**旧判据（原样保留）**。返回 (fails, notes)。

    ⚠️ 保留原数字与原来的比较方式，**不借修订之名改口径**。修订走 `criteria_v2`，
    并在 `EXPERIMENTS.md` 里登记理由。
    """
    fails: list[str] = []
    notes: list[str] = []
    t, r = per_arm.get(arm) or {}, per_arm.get(ref) or {}

    if (t.get("small_steps") or 0) <= 0:
        fails.append(f"条件1：{arm} 的小 chunk(128/256) 一步都没被实际执行"
                     f"（该臂 n_pf_steps={t.get('n_pf_steps')}，min_pf={t.get('min_pf_seen')}）")
    else:
        notes.append(f"条件1 过：{arm} 实执行小 chunk 步数 = {t['small_steps']}"
                     f"（min_pf={t.get('min_pf_seen')}）")

    if t.get("n_degraded") is None or not t.get("n_decisions"):
        fails.append(f"条件2：{arm} 的 arm_state/n_degraded 数不出来 ⇒ 不可判")
    elif r.get("n_degraded") is None or not r.get("n_decisions"):
        fails.append(f"条件2：参照臂 {ref} 的 n_degraded 数不出来 ⇒ 不可比")
    elif t["n_degraded"] >= r["n_degraded"]:
        fails.append(f"条件2：{arm} 降级 {t['n_degraded']}/{t['n_decisions']}"
                     f"={_pct(t.get('degraded_rate'))} 未低于参照 {ref} "
                     f"{r['n_degraded']}/{r['n_decisions']}={_pct(r.get('degraded_rate'))}")
    else:
        notes.append(f"条件2 过：降级 {t['n_degraded']}/{t['n_decisions']}"
                     f"={_pct(t.get('degraded_rate'))} < 参照 {r['n_degraded']}"
                     f"/{r['n_decisions']}={_pct(r.get('degraded_rate'))}")

    rate = (d0 or {}).get("feasible_rate")
    if rate is None:
        fails.append("条件3：D0 可行率数不出来")
    elif rate <= V1_D0_FEASIBLE_MIN:
        fails.append(f"条件3：D0 可行率 {rate:.2%} 未超过 v1 阈值 {V1_D0_FEASIBLE_MIN:.1%}"
                     f"（该阈值取自 **S14D（另一个模型）**；同批参照见下）")
    else:
        notes.append(f"条件3 过：D0 可行率 {rate:.2%} > {V1_D0_FEASIBLE_MIN:.1%}")
    return fails, notes


def _pct(v: Any) -> str:
    return "—" if v is None else f"{v:.1%}"


def judge(raw_root: Path, tag: str, arm: str, ref: str, min_repeats: int,
          criteria: str, diag_path: Path | None) -> dict[str, Any]:
    runs = load_runs(raw_root, tag)
    diag: dict[str, Any] = {}
    if diag_path and diag_path.exists():
        try:
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
        except Exception:
            diag = {}
    diag_arms = set((diag.get("arms") or {}).keys()) if diag else None

    probs = validity_problems(runs, [arm, ref], min_repeats, diag_arms)
    per = {a: arm_metrics(runs, a) for a in (arm, ref)}
    d0 = ((diag.get("arms") or {}).get(arm) or {}).get("d0") or {}
    # 决策级覆盖率（v2 的准入条件）—— 来自诊断产物的 `coverage` 块。
    # ⚠️ 旧诊断产物（2026-09-27 之前）**没有**这个键 ⇒ 取到 `None` ⇒ v2 会判
    #    "覆盖率数不出来"。这是期望行为（不得用缺失当通过）。
    cov = ((diag.get("arms") or {}).get(arm) or {}).get("coverage") or {}

    report: dict[str, Any] = {
        "tag": tag,
        "criteria_version": criteria,
        "arm": arm,
        "ref_arm": ref,
        "min_repeats": min_repeats,
        "n_run_dirs": len(runs),
        "arms_present": sorted(_per_arm(runs)),
        "diag_arms": (sorted(diag_arms) if diag_arms is not None else None),
        "validity_problems": probs,
        "metrics": per,
        "coverage": cov,
        "d0_treatment": d0,
        "runs": [asdict(r) for r in runs],
    }

    if probs:
        report["state"] = "INCONCLUSIVE"
        report["reasons"] = probs
        report["criteria_evaluated"] = False
        report["note"] = ("有效性前置未过 ⇒ **不判 PASS/FAIL**。下面的 metrics 只是事实记录，"
                          "不是判定。修好产物/口径后重跑本脚本。")
        # 仍然给出"若有效性成立时会怎么判"，但显式标注它**不是**本次判定
        fails_v1, notes_v1 = criteria_v1(per, arm, ref, d0)
        report["criteria_if_valid_v1"] = {"would_fail": fails_v1, "notes": notes_v1}
        fails_v2, notes_v2 = criteria_v2(per, arm, ref, d0, cov)
        report["criteria_if_valid_v2"] = {"would_fail": fails_v2, "notes": notes_v2}
        return report

    if criteria == "v1":
        fails, notes = criteria_v1(per, arm, ref, d0)
        report["criteria_evaluated"] = True
        report["criteria_v1"] = {"fails": fails, "notes": notes}
    elif criteria == "v2":
        fails, notes = criteria_v2(per, arm, ref, d0, cov)
        report["criteria_evaluated"] = True
        report["criteria_v2"] = {"fails": fails, "notes": notes,
                                 "tier": "mechanism_diagnosis",
                                 "frozen_at": "2026-09-27",
                                 "does_not_judge": "D3 服务收益（本档专用于过载区批次）"}
    else:
        report["state"] = "INCONCLUSIVE"
        report["reasons"] = [f"判据版本 {criteria!r} 未实现"]
        report["criteria_evaluated"] = False
        return report

    report["state"] = "FAIL" if fails else "PASS"
    report["reasons"] = fails if fails else notes
    return report


def all_arms_report(raw_root: Path, tag: str) -> dict[str, Any]:
    """把该批次里**所有**臂的机械事实列出来（不判定）。

    用途：给队列脚本一个**统一解析**的报表入口，替掉 shell 里那份手写的
    `re.sub(r"-r\\d+$", ...).replace("short_chat-", ...)` —— 那是同一件事的第四份实现，
    而它正是 2026-09-27 那次假失败的同类隐患。
    """
    runs = load_runs(raw_root, tag)
    arms = sorted(_per_arm(runs))
    return {
        "tag": tag,
        "n_run_dirs": len(runs),
        "arms": {a: arm_metrics(runs, a) for a in arms},
        "problems": sorted({p for r in runs for p in r.problems}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="D1 三态闸门（PASS / FAIL / INCONCLUSIVE）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--arm", default="cost_pilot", help="处理臂（D1① 的 pilot）")
    ap.add_argument("--ref-arm", default="cost_min512", help="同批参照臂")
    ap.add_argument("--raw-root", default=str(_ROOT / "results" / "raw"))
    ap.add_argument("--repeats", type=int, default=3, help="每臂要求的重复数")
    ap.add_argument("--criteria", default="v2", choices=("v1", "v2"),
                    help="v1=旧三条件（原样保留）；v2=机制诊断档（默认）")
    ap.add_argument("--diag", default=None, help="diagnose_scheduling.py 的产物 JSON")
    ap.add_argument("--json", default=None, help="把结构化判定写到这个路径")
    ap.add_argument("--all-arms", action="store_true",
                    help="只列**所有臂**的机械事实（不判定）—— 给队列脚本用的统一报表入口")
    args = ap.parse_args(argv)

    raw = Path(args.raw_root)
    if not raw.is_dir():
        print(f"✗ raw-root 不存在：{raw}", file=sys.stderr)
        return BAD_INPUT
    diag = Path(args.diag) if args.diag else None

    if args.all_arms:
        rep = all_arms_report(raw, args.tag)
        print(f"=== 臂机械事实｜tag={args.tag}｜run 目录 {rep['n_run_dirs']} 个 ===")
        print(f"{'臂':<16}{'重复':>4}{'小chunk步':>10}{'min_pf':>18}{'降级/分母':>16}{'饥饿':>6}")
        for a, m in rep["arms"].items():
            deg = f"{m['n_degraded']}/{m['n_decisions']}"
            print(f"{a:<16}{m['n_runs']:>4}{m['small_steps']:>10}"
                  f"{str(m['min_pf_seen']):>18}{deg:>16}"
                  f"{str(m['n_prefill_starved']):>6}")
        if rep["problems"]:
            print("\n逐 run 问题（前 10 条）：")
            for p in rep["problems"][:10]:
                print("   - " + p)
        if args.json:
            p = Path(args.json)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(rep, fh, ensure_ascii=False, indent=1)
            print(f"\n已写出：{p}")
        return PASS

    rep = judge(raw, args.tag, args.arm, args.ref_arm, args.repeats, args.criteria, diag)

    print(f"=== D1 闸门｜tag={args.tag}｜处理臂={args.arm}｜参照={args.ref_arm}"
          f"｜判据={args.criteria} ===")
    print(f"run 目录 {rep['n_run_dirs']} 个；产物里的臂 {rep['arms_present']}；"
          f"诊断里的臂 {rep['diag_arms']}")
    for a in (args.arm, args.ref_arm):
        m = rep["metrics"][a]
        print(f"  [{a}] 重复 {m['n_runs']}｜小chunk步 {m['small_steps']}｜"
              f"min_pf {m['min_pf_seen']}｜降级 {m['n_degraded']}/{m['n_decisions']}"
              f"={_pct(m['degraded_rate'])}｜饥饿 {m['n_prefill_starved']}")
    rate = (rep["d0_treatment"] or {}).get("feasible_rate")
    print(f"  D0 可行率（处理臂）= {_pct(rate)}")
    covv = (rep.get("coverage") or {}).get("coverage")
    print(f"  决策级模型覆盖率（处理臂）= {_pct(covv)}"
          f"（覆盖 {(rep.get('coverage') or {}).get('n_covered')}"
          f"/{(rep.get('coverage') or {}).get('n_decisions')}）")
    print(f"\n【判定】{rep['state']}")
    for r in rep.get("reasons", []):
        print("   - " + r)
    if rep.get("criteria_v2"):
        c2 = rep["criteria_v2"]
        print(f"  （判据档 = {c2.get('tier')}，冻结于 {c2.get('frozen_at')}；"
              f"**不判**：{c2.get('does_not_judge')}）")
    if not rep.get("criteria_evaluated"):
        print("   ⚠️ 有效性前置未过 ⇒ 本次**不产生** PASS/FAIL；上面 metrics 仅供修产物用。")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        # 一律 LF（Windows 上 write_text 默认会把 \n 翻成 \r\n）
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
        print(f"\n已写出：{p}")

    return {"PASS": PASS, "FAIL": FAIL, "INCONCLUSIVE": INCONCLUSIVE}[rep["state"]]


if __name__ == "__main__":
    raise SystemExit(main())
