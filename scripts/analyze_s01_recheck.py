#!/usr/bin/env python
"""S01 独立复核的**逐轮配对**分析（只读 `results/raw/<TAG>-*`）。

★ 本脚本的判据在**看数据之前**写好（见 `PREREG`）—— 这是本项目的一贯纪律：
先冻结规则，再读数字。规则写在代码里而不是事后写进报告，是为了让"改规则"变成
一次**可见的代码改动**。

★★ 2026-10-03 修正（**上一版有一个把"比值"读成"绝对率"的错误**）：
    上一版把 `slo_satisfaction_rate` 的**配对比值**（两臂都是 3.125% ⇒ 比值恒
    1.0000）直接当成了"SLO 达标率 = 1.0000"写进报告。**比值不是绝对率**：
    比值 1.0 的含义是"两臂一样"，与"达标率是 100%"毫无关系。
    ⇒ 本版对**每个**指标一律同时输出**四元**：
        对照绝对值 / 处理绝对值 / 配对变化（比值中位数）/ 分母
    并且 `slo_satisfaction_rate` 的分母单独从 `n_slo_satisfied`/`n_requests`
    重算，**不**直接采信任何一个标量字段。

逐轮配对报的指标（用户指定，一项不能少）：
    TTFT p50/p95/p99、TPOT p50/p90、吞吐、达标吞吐、SLO 达标率、
    失败/未完成数、正式窗口耗时、每轮 >1 秒的步数。

⚠️ **正式窗口耗时用 `summary.wall_ms`**（用户指定口径）；`fwd_ms+pack_ms`
   是**另一个**口径（前向/打包累计），本版把它**单独命名**为 `fwd_pack_ms`，
   两者不得互相替代。
⚠️ **原始服务指标保留起手步**：主表直接用 `summary.*`（不剔除任何轮、
   不剔除任何请求）。尾部/分段一类的**分解**只作为**补充分析**另列。
⚠️ **不要求结果一定正向**：判定有三态（支持 / 不明确 / 退化）。

★★ 2026-10-03 第二处修正（**产物必须能自证"是哪次调用写的"**）：
    我跑否定对照（同配置两臂）时控制台**正确**打出 `UNKNOWN`，随后读
    `results/processed/s01_recheck_analysis.json` 却看到 `SUPPORTED` ⇒
    **一度判定"闸门没传播"**。真相：**不加 `--json` 时本脚本不写文件**，
    磁盘上那份是**上一次**（好输入）的**旧产物** ⇒ 不是闸门失效，是**旧文件冒充新结果**。
    ⇒ 本版：① 产物带 `written_at` + `argv`（可自证身份）；② 不传 `--json` 时
    **显式提示**磁盘上的是旧产物及其 mtime。回归见
    `tests/test_s01_recheck_analysis.py::TestJsonWriteIsSelfIdentifying`（含 3 轮破坏性对照）。

⚠️ **`PREREG` 的"事前性"（如实）**：判据写在**本脚本内**并随脚本版本化；
    仓库内**没有**独立的事前封存副本（如单独 commit 的 `PREREG` 文件或当时的日志快照）。
    ⇒ 只能说"判据在看数据前随脚本写定"，**不得**说成"已预注册"。

用法::

    python scripts/analyze_s01_recheck.py --tag S01RECHECK --control s01_reference
    python scripts/analyze_s01_recheck.py --tag S01RECHECK --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

#: ★★ **预注册判据**（冻结于 2026-10-03，读数据之前）。
#:
#: 主指标取 **TTFT p50**：S01 的机制（只算最后位置 logits + 只对需要的行采样）
#: 减少的是**每步的固定开销**，因此在既有证据里它先体现在 TTFT（EXECOPT3 实测 −4.2%）。
#: 阈值取 **2%**：EXECOPT3 的基线噪声底是 `ttft_p50` **0.117%**（同协议自比），
#: 2% ≈ **17×** 噪声底 ⇒ 小到不可能被轮间抖动解释。
#: 并要求 **5/5 同向**（配对符号一致性）—— 单看幅度不看方向，会被一轮离群带偏。
#:
#: ⚠️ **本判据的"事前性"证据边界（如实声明）**：`PREREG` 是随本脚本首次提交
#: 写入的，仓库里**没有**一份"实验开始之前独立留存"的副本可供比对（EXPERIMENTS.md
#: §1.8 登记时引用的就是本文件的这一份）。因此它的地位是
#: **"随分析脚本一同版本化的判据"**，而**不是**"有独立封存证据的预注册"。
#: 任何引用必须照此表述，**不得**升级为"已预注册"。
PREREG: dict[str, Any] = {
    "primary_metric": "ttft.p50",
    "threshold_rel": 0.02,
    "require_sign_consistency": "5/5",
    "supported": "配对比值中位数 <= -2% 且 5/5 同向（更快）",
    "regressed": "配对比值中位数 >= +2% 且 5/5 同向（更慢）",
    "otherwise": "差异尚不明确（**不得**说成「没有差异」）",
    "noise_floor_ref": "EXECOPT3 同协议基线 ttft_p50 跨轮极差 0.117%（S01RECHECK 不重算它）",
    "provenance": "随本脚本版本化；仓库内**无**独立事前封存副本（详见脚本 docstring）",
}

#: 越大越好的指标（"改善"= 处理 > 对照）。其余默认为越小越好（"更快"）。
#: ⚠️ 方向搞反会**静默地把退化报成改善**（旧版对吞吐就是如此：
#: 处理吞吐更高时它数进"更快 1/5"，读起来像"处理更慢"）。
HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "throughput_tok_s", "slo_goodput_tok_s", "slo_satisfaction_rate",
    "n_success", "n_slo_satisfied",
})

#: 计数类（0 与 0 相等**是有信息的结果**，不是"没测到"）。
COUNT_METRICS: frozenset[str] = frozenset({
    "n_failed", "n_rejected", "n_unfinished", "n_long_fwd", "n_long_pack",
})


def _finite(x: Any) -> bool:
    """**有限数值**判据：`int`/`float` 且**不是** NaN / ±Inf。

    ⚠️ 为什么不能只用 `isinstance(x, (int, float))`：
      * `float("nan")` 是 float，但 `nan != nan` ⇒ 任何比较都 False；
      * `float("inf")` 是 float，且比较**全部为真** ⇒ 比值会算出 inf 或 0，
        中位数被彻底污染，而**不会报错**。
    Python 允许把 `Infinity` / `NaN` 直接写进 JSON（非标准但被接受）
    ⇒ 一条坏记录足以让整批的判定翻面。所以入口就按有限数过滤。
    """
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _finite_positive(x: Any) -> bool:
    """**有限且 > 0** —— 时延量的合法域。

    ⚠️ 为什么 `_finite` 不够（用户 2026-10-03 点名）：时延**两侧**都必须有限，
    而且必须 **> 0**。`0` 与 `-1` 都能通过 `_finite`，但它们在时延上**不可能发生**：
      * `ttft_ms = 0` 会被读成"首 token 零延迟"（比任何合法值都好）⇒ 直接把
        `ratio` 拉向 0，把一次"没测到"伪装成"大幅改善"；
      * 负值则说明**记号或时间戳取反**（例如用了 `t_first - t_arrival` 而符号写反）
        ⇒ 它会**压低**中位数，让处理臂看起来更快。
    ⇒ 时延类字段用本判据；**计数类**（`n_*`）仍用 `_finite`（0 在那里是合法结果）。
    """
    return _finite(x) and x > 0


def sanitize_json(obj: Any, *, path: str = "$") -> tuple[Any, list[str]]:
    """把 **非有限浮点**（NaN / ±Inf）换成 `None`，并**逐条记下原因**。

    返回 `(清洗后的对象, 说明列表)`。

    ⚠️ 为什么不能直接 `json.dumps(..., allow_nan=True)`（默认，也是本仓的隐患）：
      * Python 默认会把 `NaN`/`Infinity` 写成 **非标准 JSON 记号**
        （`NaN` / `Infinity` / `-Infinity`），文件看起来"写成功"了，
        但**任何严格 JSON 解析器都会拒绝它**（RFC 8259 不允许这几个记号）；
      * 更糟的是 `allow_nan=False` 会**抛异常**，于是"产物写不出来"——
        而那正是最需要产物的时候（数据已经坏了）。
    ⇒ 本函数的选择：**保留结构、把坏值置 `null`、把成因写进 `_json_notes`**。
      这样产物**永远是合法 JSON**，且"哪里坏过"**留下痕迹**，
      而不是 ① 悄悄写出非法 JSON，或 ② 整个文件写不出来。
    """
    notes: list[str] = []

    def walk(o: Any, p: str) -> Any:
        if isinstance(o, bool) or o is None:
            return o
        if isinstance(o, float):
            if math.isfinite(o):
                return o
            notes.append(f"{p} = {o!r} 不是有限数 ⇒ 已置 `null`（非有限值不可作诊断依据）")
            return None
        if isinstance(o, dict):
            return {k: walk(v, f"{p}.{k}") for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v, f"{p}[{i}]") for i, v in enumerate(o)]
        return o

    return walk(obj, path), notes


def check_valid_numeric(summary: dict[str, Any], *, expect_pairs: int) -> tuple[bool, list[str]]:
    """★★ **有效数值门**：主指标必须恰好有 `expect_pairs` 对**有限且合法**的值。

    这是"闸门"而不是"提示"：不过 ⇒ `verdict()` 一律返回 UNKNOWN、
    `main()` 返回非零退出码。

    逐条要求（对应本轮用户点名）：
      ① **恰好 N 对**（默认 5）——少于 N 说明有轮次的主指标不可用；
      ② 对照与处理两侧**都**必须是有限数值（`None`/NaN/±Inf 都算不可用）；
      ③ 每一对的分母（对照绝对值）必须**非零** ⇒ 比值有定义；
      ④ 每对必须产出**合法比值**（有限、非 NaN）——`0/0`、`±Inf/0` 都拦在这；
      ⑤ ★ **TTFT 两侧都必须有限且 > 0**（用户 2026-10-03 点名）——
         `0` / 负值在时延上不可能发生，却都能穿过 `_finite`。
         `ttft = 0` 会被读成"零延迟"，比任何合法值都好 ⇒ 把"没测到"伪装成"大幅改善"。
    """
    m = PREREG["primary_metric"].replace(".", "_")
    s = summary.get(m) or {}
    problems: list[str] = []
    n_valid = s.get("n_valid_pairs")
    n_total = s.get("n_total_rows")
    if n_total is None:
        problems.append("主指标汇总缺 `n_total_rows` ⇒ 无法核对有效性")
    if n_valid is None:
        problems.append("主指标汇总缺 `n_valid_pairs` ⇒ 无法核对有效性")
        return False, problems
    if n_valid != expect_pairs:
        problems.append(
            f"主指标 `{PREREG['primary_metric']}` 的**有效配对 = {n_valid}**，"
            f"要求恰好 {expect_pairs}（总行数 {n_total}）"
            f" ⇒ 有 {expect_pairs - n_valid} 对不可用（缺值/NaN/±Inf/零分母）")
    if s.get("n_ratio_invalid"):
        problems.append(
            f"主指标有 {s['n_ratio_invalid']} 对**比值非法**（对照值为 0 或结果非有限）"
            f"：{s.get('ratio_invalid_detail')}")
    # ⑤ TTFT 两侧有限且 > 0（**独立于主指标**）——见 docstring 第 ⑤ 条。
    problems.extend(_ttft_positivity_problems(summary))
    return (not problems), problems


#: 要求"两侧有限且 > 0"的时延字段。计数类**不在此列**（0 是合法结果）。
POSITIVE_LATENCY_METRICS: tuple[str, ...] = ("ttft_p50", "ttft_p95", "ttft_p99")


def _ttft_positivity_problems(summary: dict[str, Any]) -> list[str]:
    """TTFT 各分位数**两侧**都必须有限且 > 0；否则逐条报出（含具体轮次）。

    ⚠️ 只报"非法"的轮次与数值，**不**把它当成"不可用而对" ——
    若此处有病态值，说明**数据本身有问题**，必须先查成因，不许直接下结论。
    """
    out: list[str] = []
    for metric in POSITIVE_LATENCY_METRICS:
        s = summary.get(metric) or {}
        bad_c = s.get("nonpositive_control") or []
        bad_t = s.get("nonpositive_treatment") or []
        for who, bad in (("对照", bad_c), ("处理", bad_t)):
            if bad:
                out.append(
                    f"TTFT `{metric}` 在**{who}**侧出现 {len(bad)} 个**非正/非有限**值"
                    f"（时延必须 > 0）：{bad} ⇒ 这些不是『很快』，是**不可用**；"
                    "先查时间戳口径/符号，不得据此宣称改善")
    return out


#: 计数类（0 与 0 相等**是有信息的结果**，不是"没测到"）。

ALL_METRICS: tuple[str, ...] = (
    "ttft_p50", "ttft_p95", "ttft_p99", "tpot_p50", "tpot_p90",
    "throughput_tok_s", "slo_goodput_tok_s", "slo_satisfaction_rate",
    "n_failed", "n_unfinished", "wall_ms", "fwd_pack_ms",
    "n_long_fwd", "n_long_pack",
)


def load_runs(root: Path, tag: str) -> dict[str, list[Path]]:
    """按臂归组。`<TAG>-<workload>-<arm>[-rN]`；基线不带 `-rN`。"""
    out: dict[str, list[Path]] = {}
    for d in sorted(root.glob(f"{tag}-*")):
        if not d.is_dir():
            continue
        arm = arm_of(d)
        if arm is None:
            continue
        out.setdefault(arm, []).append(d)
    return out


def arm_of(d: Path) -> str | None:
    """`<TAG>-<workload>-<arm>[-rN]` → `<arm>`。"""
    name = d.name
    tail = name.rsplit("-", 1)[-1]
    if tail.startswith("r") and tail[1:].isdigit():
        stem = name[: -(len(tail) + 1)]
    else:
        stem = name
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    return "-".join(parts[2:])


def round_of(d: Path) -> int:
    name = d.name
    tail = name.rsplit("-", 1)[-1]
    return int(tail[1:]) if tail.startswith("r") and tail[1:].isdigit() else 0


def _snapshot_digest(run_dir: Path) -> str | None:
    """生效配置的身份 = `config.snapshot.yaml` 的 sha256（**文件本身**，不是某个字段）。"""
    p = run_dir / "config.snapshot.yaml"
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pct(d: dict[str, Any] | None, key: str) -> float | None:
    v = (d or {}).get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def step_stalls(run_dir: Path, *, threshold_ms: float = 1000.0) -> dict[str, int]:
    """每轮 **>1 秒的步数**（按 `fwd_ms` / `pack_ms` 分别数）。

    ⚠️ 单位是**步**，不是毫秒：一个 3 秒的停顿与三个 1 秒的停顿在"墙钟"上等价，
    但在**成因**上完全不同（前者像一次暂停，后者像持续慢态）。
    """
    tp = run_dir / "telemetry.jsonl"
    if not tp.is_file():
        return {"n_long_fwd": -1, "n_long_pack": -1, "n_steps": -1}
    n_long_fwd = n_long_pack = n = 0
    for line in tp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        n += 1
        note = str(json.loads(line).get("note") or "")
        fwd = pack = None
        for tok in note.replace(",", " ").split():
            if tok.startswith("fwd_ms="):
                try:
                    fwd = float(tok.split("=", 1)[1])
                except ValueError:
                    pass
            elif tok.startswith("pack_ms="):
                try:
                    pack = float(tok.split("=", 1)[1])
                except ValueError:
                    pass
        if fwd is not None and fwd > threshold_ms:
            n_long_fwd += 1
        if pack is not None and pack > threshold_ms:
            n_long_pack += 1
    return {"n_long_fwd": n_long_fwd, "n_long_pack": n_long_pack, "n_steps": n}


def read_round(run_dir: Path) -> dict[str, Any]:
    """一轮的**原始服务指标**（不剔除任何东西）。"""
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    s = meta.get("summary") or {}
    repro = meta.get("reproducibility") or {}
    ds = (((meta.get("context") or {}).get("extra") or {})
          .get("backend_stats") or {}).get("driver_stats") or {}
    mw = ds.get("measured_window") or {}
    cfg = meta.get("config") or {}
    stalls = step_stalls(run_dir)
    fwd_m, pack_m = mw.get("fwd_ms_measured"), mw.get("pack_ms_measured")
    #: ⚠️ `fwd_pack_ms` 与 `wall_ms` 是**两个口径**，不得互换（见模块 docstring）。
    fwd_pack = (float(fwd_m) + float(pack_m)
                if isinstance(fwd_m, (int, float)) and isinstance(pack_m, (int, float))
                else None)
    n_req = _num(s.get("n_requests"))
    n_sat = _num(s.get("n_slo_satisfied"))
    return {
        "dir": run_dir.name,
        "round": round_of(run_dir),
        # —— 身份 ——
        "data_label": meta.get("data_label"),
        "tree_digest": repro.get("tree_digest"),
        "git_commit": repro.get("git_commit"),
        "git_source": repro.get("git_source"),
        "machine": str(repro.get("hostname_or_machine_id") or "").split("/", 1)[0],
        "config_digest": _snapshot_digest(run_dir),
        "model_revision": repro.get("model_revision")
                           or (cfg.get("runtime") or {}).get("model_revision"),
        # —— 服务指标（原始，保留起手步）——
        "ttft_p50": _pct(s.get("ttft"), "p50"),
        "ttft_p95": _pct(s.get("ttft"), "p95"),
        "ttft_p99": _pct(s.get("ttft"), "p99"),
        "tpot_p50": _pct(s.get("tpot"), "p50"),
        "tpot_p90": _pct(s.get("tpot"), "p90"),
        "throughput_tok_s": _num(s.get("throughput_tok_s")),
        "slo_goodput_tok_s": _num(s.get("slo_goodput_tok_s")),
        # ★ 达标率**从整数重算**，不采信标量字段（分母显式可查）。
        "slo_satisfaction_rate": (n_sat / n_req) if (n_sat is not None and n_req) else None,
        "n_slo_satisfied": n_sat,
        "n_requests": n_req,
        "n_success": _num(s.get("n_success")),
        "n_failed": _num(s.get("n_failed")),
        "n_rejected": _num(s.get("n_rejected")),
        "n_unfinished": _num(s.get("n_unfinished")),
        # —— 窗口口径：正式窗口 = wall_ms；fwd+pack 另名 ——
        "wall_ms": _num(s.get("wall_ms")),
        "fwd_pack_ms": fwd_pack,
        "fwd_ms": _num(fwd_m),
        "pack_ms": _num(pack_m),
        "n_steps": mw.get("n_logical_steps_measured"),
        "n_long_fwd": stalls["n_long_fwd"],
        "n_long_pack": stalls["n_long_pack"],
    }


REQUIRED_FILES = ("meta.json", "requests.jsonl", "telemetry.jsonl", "config.snapshot.yaml")


def check_completeness(runs: dict[str, list[Path]], *, expect_rounds: int,
                       expect_arms: int) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if len(runs) != expect_arms:
        problems.append(f"臂数 = {len(runs)}（期望 {expect_arms}）：{sorted(runs)}")
    for arm, dirs in sorted(runs.items()):
        if len(dirs) != expect_rounds:
            problems.append(f"{arm}: {len(dirs)} 轮（期望 {expect_rounds}）")
        for f in REQUIRED_FILES:
            for d in dirs:
                if not (d / f).is_file():
                    problems.append(f"{d.name}: 缺 {f}")
    total = sum(len(v) for v in runs.values())
    if total != expect_arms * expect_rounds:
        problems.append(f"总项数 = {total}（期望 {expect_arms * expect_rounds}）")
    return (not problems), problems


def check_pairing(rows_c: list[dict], rows_t: list[dict], *,
                  expect_rounds: int) -> tuple[bool, list[str]]:
    """★★ 配对的**硬闸门**：必须恰好 N 对、轮次唯一、且两侧轮次集合完全相同。

    任何一条不满足 ⇒ 返回 False，调用方**不得**给出 SUPPORTED/REGRESSED。
    这一层专门拦三类静默错误：
      ① 臂被反转（对照里有 `s01_both` 的目录）；
      ② 缺一轮（并集多于交集）；
      ③ 同一轮重复（`-r1` 出现两次 ⇒ 用 dict 去重会**静默丢掉一份产物**）。
    """
    problems: list[str] = []
    rc = [r["round"] for r in rows_c]
    rt = [r["round"] for r in rows_t]
    if len(set(rc)) != len(rc):
        dup = sorted({x for x in rc if rc.count(x) > 1})
        problems.append(f"对照臂轮次重复：{dup}")
    if len(set(rt)) != len(rt):
        dup = sorted({x for x in rt if rt.count(x) > 1})
        problems.append(f"处理臂轮次重复：{dup}")
    sc, stt = set(rc), set(rt)
    if sc != stt:
        only_c, only_t = sorted(sc - stt), sorted(stt - sc)
        problems.append(f"两侧轮次集合不同：仅对照有 {only_c}；仅处理有 {only_t}")
    if len(sc & stt) != expect_rounds:
        problems.append(f"可比轮次数 = {len(sc & stt)}（期望恰好 {expect_rounds}）")
    return (not problems), problems


def check_identity(rows_c: list[dict], rows_t: list[dict]) -> tuple[bool, list[str]]:
    """数据标签 / 身份 / 配置差异 / 有限数值 必须全部通过，否则不许下结论。"""
    problems: list[str] = []
    for label, rows in (("对照", rows_c), ("处理", rows_t)):
        for k in ("data_label", "tree_digest", "machine", "model_revision"):
            vals = {r.get(k) for r in rows}
            if len(vals) != 1 or None in vals or "" in vals:
                problems.append(f"{label}：{k} 不唯一/缺失 → {sorted(map(str, vals))}")
        for r in rows:
            if r.get("data_label") != "MEASURED":
                problems.append(f"{label} {r['dir']}：data_label="
                                f"{r.get('data_label')!r}（非 MEASURED）")
    # 两臂必须各跑**不同的一份**配置，且各自轮内一致
    for label, rows in (("对照", rows_c), ("处理", rows_t)):
        digs = {r.get("config_digest") for r in rows}
        if len(digs) != 1 or None in digs:
            problems.append(f"{label}：config_digest 轮内不一致/缺失 → {sorted(map(str, digs))}")
    dc = {r.get("config_digest") for r in rows_c}
    dt = {r.get("config_digest") for r in rows_t}
    if dc == dt:
        problems.append("两臂 config_digest **相同** ⇒ 单变量前提不成立（跑的是一份配置）")
    # 跨臂身份必须一致
    for k in ("tree_digest", "machine", "model_revision"):
        vc = {r.get(k) for r in rows_c}
        vt = {r.get(k) for r in rows_t}
        if vc != vt:
            problems.append(f"跨臂 {k} 不一致：对照 {sorted(map(str, vc))} vs "
                            f"处理 {sorted(map(str, vt))}")
    return (not problems), problems


def paired_rows(rows_c: list[dict], rows_t: list[dict],
                metrics: list[str]) -> list[dict]:
    """按轮次配对，并算出每对的比值/差值。

    ⚠️ 比值**只在两侧都有限且分母非零**时才给数；否则留 `None`
    （`None` 与"比值恰好是 0/1"必须区分开 —— 后者是真实结果）。
    """
    by_c = {r["round"]: r for r in rows_c}
    by_t = {r["round"]: r for r in rows_t}
    rows: list[dict] = []
    for r in sorted(set(by_c) & set(by_t)):
        row: dict[str, Any] = {"round": r}
        for m in metrics:
            vc, vt = by_c[r].get(m), by_t[r].get(m)
            row[f"{m}__c"] = vc
            row[f"{m}__t"] = vt
            if _finite(vc) and _finite(vt):
                row[f"{m}__ratio"] = (vt / vc) if vc != 0 else None
                row[f"{m}__delta"] = vt - vc
            else:
                row[f"{m}__ratio"] = None
                row[f"{m}__delta"] = None
        rows.append(row)
    return rows


def summarize_pair(rows: list[dict], metric: str) -> dict[str, Any]:
    """四元口径：对照绝对值 / 处理绝对值 / 配对变化 / 分母。

    ★★ 2026-10-03：另记**有效性**三个计数（供 `check_valid_numeric` 用）：
      * `n_total_rows` —— 参与比较的轮次数（配对行数）；
      * `n_valid_pairs` —— **两侧都有限**且**比值合法**的对数（只有这些才进统计）；
      * `n_ratio_invalid` + `ratio_invalid_detail` —— 分母为 0 或比值非有限的对。
    ⚠️ 不能只看 `n`：`n` 数的是"对照侧有值"的行数，处理侧是 NaN 时它照样计入，
      于是"5 对"看起来成立、实际只有 4 对可用。这正是本门要拦的形态。
    """
    c_vals = [r[f"{metric}__c"] for r in rows if _finite(r.get(f"{metric}__c"))]
    t_vals = [r[f"{metric}__t"] for r in rows if _finite(r.get(f"{metric}__t"))]
    ratios = [r[f"{metric}__ratio"] for r in rows if _finite(r.get(f"{metric}__ratio"))]
    deltas = [r[f"{metric}__delta"] for r in rows if _finite(r.get(f"{metric}__delta"))]

    # ---- 有效性计数（本门专用）----
    n_total = len(rows)
    ratio_invalid: list[str] = []
    n_valid = 0
    for r in rows:
        vc, vt = r.get(f"{metric}__c"), r.get(f"{metric}__t")
        if not (_finite(vc) and _finite(vt)):
            continue
        rat = r.get(f"{metric}__ratio")
        if not _finite(rat):
            ratio_invalid.append(
                f"r{r.get('round')}: 对照 {vc!r} / 处理 {vt!r} ⇒ 比值 {rat!r}")
            continue
        n_valid += 1

    higher_better = metric in HIGHER_IS_BETTER
    # ★ 时延类：另记"非正/非有限"的轮次（要求 ⑤）。计数类不看这一项。
    nonpositive_c: list[Any] = []
    nonpositive_t: list[Any] = []
    if metric in POSITIVE_LATENCY_METRICS:
        for r in rows:
            vc, vt = r.get(f"{metric}__c"), r.get(f"{metric}__t")
            if vc is not None and not _finite_positive(vc):
                nonpositive_c.append({"round": r.get("round"), "value": vc})
            if vt is not None and not _finite_positive(vt):
                nonpositive_t.append({"round": r.get("round"), "value": vt})
    out: dict[str, Any] = {
        "n": len(c_vals),
        "n_total_rows": n_total,
        "n_valid_pairs": n_valid,
        "n_ratio_invalid": len(ratio_invalid),
        "ratio_invalid_detail": ratio_invalid,
        "nonpositive_control": nonpositive_c,
        "nonpositive_treatment": nonpositive_t,
        "control_abs_median": st.median(c_vals) if c_vals else None,
        "treatment_abs_median": st.median(t_vals) if t_vals else None,
        "higher_is_better": higher_better,
    }
    if not ratios:
        if deltas and all(d == 0 for d in deltas):
            out.update({"ratio_median": None, "all_zero_delta": True,
                        "n_better": 0, "n_worse": 0, "n_faster": 0, "n_slower": 0})
        else:
            out.update({"ratio_median": None, "all_zero_delta": False,
                        "n_better": 0, "n_worse": 0, "n_faster": 0, "n_slower": 0})
        return out
    med = st.median(ratios)
    # "更好"= 处理在有利方向上（吞吐类越大越好；时延类越小越好）。
    n_better = sum(1 for x in ratios if (x > 1.0 if higher_better else x < 1.0))
    n_worse = sum(1 for x in ratios if (x < 1.0 if higher_better else x > 1.0))
    out.update({
        "ratio_median": med,
        "rel_pct": (med - 1.0) * 100.0,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "delta_median": st.median(deltas) if deltas else None,
        "n_better": n_better,
        "n_worse": n_worse,
        # 兼容旧字段名（时延语义下 better=faster）
        "n_faster": n_better if not higher_better else 0,
        "n_slower": n_worse if not higher_better else 0,
        "all_zero_delta": False,
    })
    return out


def verdict(summary: dict[str, Any], *, gates_ok: bool,
            valid_ok: bool = True, valid_problems: list[str] | None = None
            ) -> dict[str, Any]:
    """按**预注册**规则判定（三态）。**闸门不过就绝不给 SUPPORTED/REGRESSED。**
    ⚠️ `gates_ok=False` 与 `valid_ok=False` **分开报**：前者是"产物不完整/配对不成立/身份
    不一致"，后者是"对数够、但值本身不可用（缺值/NaN/Inf/零分母）"。合成一个 why
    会让人分不清该去补产物还是该去查数值，所以两条原因各自成句。
    """
    m = PREREG["primary_metric"].replace(".", "_")
    s = summary.get(m) or {}
    if not gates_ok:
        return {"state": "UNKNOWN",
                "why": "完整性/配对/身份闸门未通过 ⇒ 不得据本批下任何性能结论"
                       "（已完成的项保留，未完成的登记）"}
    if not valid_ok:
        return {"state": "UNKNOWN",
                "why": "**有效数值门未通过** ⇒ 主指标不足 5 对有限且合法的配对数"
                       " ⇒ 不得据本批下任何性能结论：" + "；".join(valid_problems or [])}
    if not s.get("n"):
        return {"state": "UNKNOWN", "why": "主指标无可用配对"}
    rel = (s.get("rel_pct") or 0.0) / 100.0
    n = s["n"]
    thr = PREREG["threshold_rel"]
    if rel <= -thr and s["n_faster"] == n:
        return {"state": "SUPPORTED",
                "why": f"主指标 {PREREG['primary_metric']} 配对中位 {rel * 100:+.2f}%"
                       f"（≤ -{thr * 100:.0f}%）且 {s['n_faster']}/{n} 同向（更快）"}
    if rel >= thr and s["n_slower"] == n:
        return {"state": "REGRESSED",
                "why": f"主指标 {PREREG['primary_metric']} 配对中位 {rel * 100:+.2f}%"
                       f"（≥ +{thr * 100:.0f}%）且 {s['n_slower']}/{n} 同向（更慢）"}
    return {"state": "UNCLEAR",
            "why": f"主指标 {PREREG['primary_metric']} 配对中位 {rel * 100:+.2f}%，"
                   f"同向 {max(s['n_faster'], s['n_slower'])}/{n}，未同时满足幅度与一致性"
                   f"⇒ **差异尚不明确**（不得表述为「无差异」）"}


def _fmt(v: Any, nd: int = 1) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def _iso_now() -> str:
    """本地时区的 ISO 时间戳。**不**用 `time.time()` 手算 —— 见全局约定。"""
    import datetime as _dt

    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _iso_now_mtime(p: Path) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(
        timespec="seconds")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", default=str(_ROOT / "results" / "raw"))
    ap.add_argument("--control", default=None, help="对照臂名（必须显式指定，防臂反转）")
    ap.add_argument("--treatment", default=None, help="处理臂名（默认取另一臂）")
    ap.add_argument("--expect-rounds", type=int, default=5)
    ap.add_argument("--expect-arms", type=int, default=2)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    runs = load_runs(Path(args.root), args.tag)
    if not runs:
        print(f"✗ 找不到 tag={args.tag} 的产物", file=sys.stderr)
        return 2
    arms = sorted(runs)
    if args.control is None:
        print("✗ 必须用 --control 显式指定对照臂（防臂反转）", file=sys.stderr)
        print(f"  可选：{arms}", file=sys.stderr)
        return 2
    control = args.control
    if control not in runs:
        print(f"✗ 对照臂 {control!r} 不在产物里；可选：{arms}", file=sys.stderr)
        return 2
    treat = args.treatment or next((a for a in arms if a != control), None)
    if treat is None or treat not in runs:
        print("✗ 处理臂无效（只有一个臂或名字不存在）", file=sys.stderr)
        return 2

    rows_c = sorted((read_round(d) for d in runs[control]), key=lambda r: r["round"])
    rows_t = sorted((read_round(d) for d in runs[treat]), key=lambda r: r["round"])

    ok_c, prob_c = check_completeness(runs, expect_rounds=args.expect_rounds,
                                      expect_arms=args.expect_arms)
    ok_p, prob_p = check_pairing(rows_c, rows_t, expect_rounds=args.expect_rounds)
    ok_i, prob_i = check_identity(rows_c, rows_t)
    base_gates_ok = ok_c and ok_p and ok_i

    # ★ 先算配对与汇总，好让 ③b 的"有效数值门"能在打印区里就地取值。
    paired = paired_rows(rows_c, rows_t, list(ALL_METRICS))
    summary: dict[str, Any] = {m: summarize_pair(paired, m) for m in ALL_METRICS}

    # ★★ 有效数值门：**主指标**必须恰好 expect_rounds 对有限且合法。
    #    它是"能不能下结论"的第三道闸，必须与 ①②③ 并列检查、共同决定 verdict。
    ok_v, prob_v = check_valid_numeric(summary, expect_pairs=args.expect_rounds)
    gates_ok = base_gates_ok and ok_v

    print("=" * 104)
    print(f"S01 独立复核 · tag={args.tag} · **对照={control} → 处理={treat}**")
    print("=" * 104)
    print(f"\n-- ① 完整性：{'✓ 通过' if ok_c else '✗ 不通过'} --")
    for p in prob_c:
        print(f"  ✗ {p}")
    print(f"-- ② 配对闸门（恰好 {args.expect_rounds} 对·唯一·完整）："
          f"{'✓ 通过' if ok_p else '✗ 不通过'} --")
    for p in prob_p:
        print(f"  ✗ {p}")
    print(f"-- ③ 身份/标签/配置差异：{'✓ 通过' if ok_i else '✗ 不通过'} --")
    for p in prob_i:
        print(f"  ✗ {p}")
    if not base_gates_ok:
        print("  ⇒ **基础闸门未全过 ⇒ 不得给 SUPPORTED/REGRESSED**（已完成的项保留）")

    print(f"-- ③b 有效数值门（主指标必须恰好 {args.expect_rounds} 对有限且合法）："
          f"{'✓ 通过' if ok_v else '✗ 不通过'} --")
    _msum = summary.get(PREREG["primary_metric"].replace(".", "_")) or {}
    print(f"  主指标 = {PREREG['primary_metric']}；"
          f"总行数 {_msum.get('n_total_rows')} / 有效配对 {_msum.get('n_valid_pairs')} / "
          f"比值非法 {_msum.get('n_ratio_invalid')}")
    for p in prob_v:
        print(f"  ✗ {p}")
    if not ok_v:
        print("  ⇒ **有效数值门未过 ⇒ 不得给 SUPPORTED/REGRESSED**（三态一律 UNKNOWN）")

    # ★ 要求 ⑤：TTFT 两侧有限且 > 0（独立于主指标；`0` 会被读成"零延迟"）
    print("-- ③c TTFT 两侧有限且 > 0（要求 ⑤）--")
    for metric in POSITIVE_LATENCY_METRICS:
        _s = summary.get(metric) or {}
        _bc, _bt = _s.get("nonpositive_control") or [], _s.get("nonpositive_treatment") or []
        _flag = "✓ 通过" if not (_bc or _bt) else "✗ 有非正值"
        print(f"  {metric:<10} {_flag}"
              f"（对照侧非正值 {len(_bc)}，处理侧非正值 {len(_bt)}）")

    print("\n-- ④ 身份（逐臂）--")
    for label, rows in ((control, rows_c), (treat, rows_t)):
        dug = sorted({str(r["tree_digest"])[:12] for r in rows})
        cfg = sorted({str(r["config_digest"])[:12] for r in rows})
        rev = sorted({str(r["model_revision"]) for r in rows})
        mac = sorted({r["machine"] for r in rows})
        lab = sorted({str(r["data_label"]) for r in rows})
        print(f"  {label:<22} tree={dug} cfg={cfg} rev={rev} machine={mac} label={lab}")

    print("\n-- ⑤ 逐轮配对（四元口径：对照绝对 / 处理绝对 / 配对变化；保留起手步）--")
    print(f"  {'round':<6}{'ttft_p50 对照':>16}{'处理':>10}{'比值':>9}"
          f"{'tpot_p50 对照':>16}{'处理':>10}{'比值':>9}")
    for r in paired:
        print(f"  {r['round']:<6}{_fmt(r['ttft_p50__c']):>16}{_fmt(r['ttft_p50__t']):>10}"
              f"{_fmt(r['ttft_p50__ratio'], 4):>9}"
              f"{_fmt(r['tpot_p50__c']):>16}{_fmt(r['tpot_p50__t']):>10}"
              f"{_fmt(r['tpot_p50__ratio'], 4):>9}")

    print("\n-- ⑥ 汇总（四元口径；「更好」按各自方向；分母显式）--")
    for m in ALL_METRICS:
        s = summary[m]
        if s["control_abs_median"] is None:
            print(f"  {m:<24} （无数据）")
            continue
        tag = "越大越好" if s["higher_is_better"] else "越小越好"
        # 按量级选小数位：达标率是 0.03 量级，用 1 位小数会显示成 "0.0"
        def _scal(v: float) -> str:
            if abs(v) < 10:
                return f"{v:.4f}"
            return f"{v:.1f}"
        if m == "slo_satisfaction_rate":
            c_sat, t_sat = rows_c[0].get("n_slo_satisfied"), rows_t[0].get("n_slo_satisfied")
            n_req = rows_c[0].get("n_requests")
            denom = (f"（分母 = n_requests = {_fmt(n_req, 0)}；"
                     f"原语整数：对照 {_fmt(c_sat, 0)}/{_fmt(n_req, 0)}、"
                     f"处理 {_fmt(t_sat, 0)}/{_fmt(n_req, 0)}）")
        else:
            denom = ""
        if s.get("all_zero_delta"):
            print(f"  {m:<24} 对照 {_fmt(s['control_abs_median'])} / "
                  f"处理 {_fmt(s['treatment_abs_median'])}；**两臂全等 0**"
                  "—— 这是结果，不是「没测到」")
            continue
        if s.get("ratio_median") is None:
            print(f"  {m:<24} 对照 {_fmt(s['control_abs_median'])} / "
                  f"处理 {_fmt(s['treatment_abs_median'])}；比值无定义{denom}")
            continue
        print(f"  {m:<24} 对照 {_scal(s['control_abs_median'])} → "
              f"处理 {_scal(s['treatment_abs_median'])}；"
              f"比值中位 {s['ratio_median']:.4f}（{s['rel_pct']:+.2f}%，{tag}）"
              f"  更好 {s['n_better']}/{s['n']}  更差 {s['n_worse']}/{s['n']}{denom}")

    print("\n-- ⑦ 每轮 >1 秒的步数（单步 `fwd_ms` / `pack_ms`）--")
    for r in paired:
        print(f"  r{r['round']}: 对照 fwd={_fmt(r['n_long_fwd__c'], 0)} "
              f"pack={_fmt(r['n_long_pack__c'], 0)}  "
              f"处理 fwd={_fmt(r['n_long_fwd__t'], 0)} "
              f"pack={_fmt(r['n_long_pack__t'], 0)}")

    v = verdict(summary, gates_ok=base_gates_ok, valid_ok=ok_v,
                valid_problems=prob_v)
    print("\n-- ⑧ 判定（**脚本内版本化判据**，非事后）--")
    print(f"  规则：主指标 {PREREG['primary_metric']}；阈值 ±{PREREG['threshold_rel'] * 100:.0f}%；"
          f"要求 {PREREG['require_sign_consistency']} 同向")
    print(f"  闸门：基础（完整性/配对/身份）={'过' if base_gates_ok else '不过'}；"
          f"有效数值={'过' if ok_v else '不过'}")
    print(f"  ★ 结论 = **{v['state']}**")
    print(f"    依据：{v['why']}")

    payload = {
        # ★ 自识别字段（防"旧文件冒充新结果"）：
        #   2026-10-03 踩到 —— 不加 `--json` 时脚本**不写文件**，于是磁盘上的旧 JSON
        #   在与本次**不同**的臂/参数下依然是 SUPPORTED。我据此误判"闸门没生效"。
        #   ⇒ 任何读 JSON 的人必须能判出"这份文件是哪次调用写的"。
        "written_at": _iso_now(),
        "argv": list(sys.argv[1:]),
        "tag": args.tag, "control": control, "treatment": treat,
        "complete": ok_c, "problems": prob_c,
        "pairing_ok": ok_p, "pairing_problems": prob_p,
        "identity_ok": ok_i, "identity_problems": prob_i,
        "valid_numeric_ok": ok_v, "valid_numeric_problems": prob_v,
        "base_gates_ok": base_gates_ok,
        "gates_ok": gates_ok,
        "prereg": PREREG, "verdict": v,
        "rounds_control": rows_c, "rounds_treatment": rows_t,
        "paired": paired, "summary": summary,
    }
    if args.json:
        # ★★ 非有限诊断值 → `null` + 错误说明，并**严格**输出 JSON（用户 2026-10-03 点名）。
        #     `allow_nan=False` 让"还有漏网的 NaN/Inf"变成**显式报错**而不是静默写出非法 JSON。
        clean, jnotes = sanitize_json(payload)
        clean["_json_notes"] = jnotes
        text = json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        Path(args.json).write_text(text, encoding="utf-8", newline="\n")
        print(f"\n写出 {args.json}（written_at={payload['written_at']}）")
        if jnotes:
            print(f"  ⚠️ 有 {len(jnotes)} 个非有限诊断值已置 `null`（详见产物 `_json_notes`）")
            for n in jnotes[:5]:
                print(f"      · {n}")
    else:
        # ★ 防"旧文件冒充新结果"：不加 --json 时磁盘上可能躺着**上一次**的 JSON，
        #   它的 verdict 与本次运行无关。2026-10-03 我正是被它误导（把旧的 SUPPORTED
        #   当成"闸门没生效"）。⇒ 这里**显式提示**，让人不会误读。
        stale = _ROOT / "results" / "processed" / "s01_recheck_analysis.json"
        if stale.is_file():
            print(f"\n⚠️ 未加 --json ⇒ **本次未写任何文件**；磁盘上的 {stale} "
                  f"是**旧**产物（mtime={_iso_now_mtime(stale)}），与本次运行无关。")
    return 0 if gates_ok else 1


if __name__ == "__main__":
    sys.exit(main())
