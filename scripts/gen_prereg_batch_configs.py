#!/usr/bin/env python
"""生成本批（机制小批 / pilot）的**可执行派生配置**。

★ 为什么另起一份生成器，而不是扩 `gen_next_batch_configs.py`
----------------------------------------------------------------
上一份（第 E 任务的产物）只解决了"**臂**从哪来"，留下三个**必须由配置回答**、
当时却靠**默认值 / 文档描述**在承担的问题：

| 遗留问题 | 当时的实际状态 | 现在怎么做 |
| --- | --- | --- |
| 成本模型版本 v5/v7 不一致 | `fixed_cost.yaml` 写 **v5**，而 v7 才是当前最大一次拟合 | 显式写 `cost_model_path`，并把 v5/v6/v7 **血缘**记进文件头 |
| `prefill_deadline_ms` = 2000 | 与 `run.slo.ttft_ms` 同源，但**不等于**"内部每步预算" | 显式区分 **120 ms 内部预算** 与 **100 ms 外部 SLO** |
| S01 开关 | 配置里**完全没有**（`fixed_cost.yaml` 无 `runtime.logits_last_only`） | ★ 显式写 `runtime.logits_last_only` / `sample_needed_only` |

⚠️ **`--dry-run` 验不了 S01、也验不了过载/SLO**：dry-run 路径从不喂 `decode_samples`
⇒ 判据恒"不可测" ⇒ 控制器保守不动。所以本文件里的 S01 开关**只能**在真机被行使。

★ **本批派生 config 的铁律**：只统一**本批**的派生配置，
**不碰** `configs/mthreads/phase1_step14|15/**`（已执行的冷冻配置）。
任何"顺手改一下老配置"都会让既有产物与其配置快照对不上。

用法::

    python scripts/gen_prereg_batch_configs.py             # 生成/覆盖
    python scripts/gen_prereg_batch_configs.py --check      # 逐字节校验
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = _ROOT / "configs" / "mthreads" / "phase1_step14"
OUTDIR = _ROOT / "configs" / "mthreads" / "prereg_batch"

# ---------------------------------------------------------------------------
# ★★ 本批的**冻结取值**（改这里 = 改研究协议，必须同步改 NEXT_BATCHES_PREREG.md）
# ---------------------------------------------------------------------------

#: **外部 SLO**：逐请求判据（`run.slo.*`）。达标率/达标吞吐按它算。
EXTERNAL_TPOT_SLO_MS = 100.0
EXTERNAL_TTFT_SLO_MS = 2000.0

#: **内部每步预算**：`step_deadline_ms` = decode 护栏（服务 TPOT）。
#: ⚠️ 它**故意大于**外部 SLO：`controller.target_tpot_ms` 是**控制目标**，
#: 不是判据阈值 —— 当年正是"100 ms 打不到"才把控制目标放宽到 120 ms。
#: ⚠️ **不得**把这两者写成一个数（那会把"控制目标"与"验收判据"混同）。
INTERNAL_STEP_BUDGET_MS = 120.0

#: **prefill 预算**：服务 TTFT 的承诺，与 `run.slo.ttft_ms` **同源**。
PREFILL_DEADLINE_MS = EXTERNAL_TTFT_SLO_MS

#: 本批统一使用的成本模型（v7 = 当前最大一次拟合）。
COST_MODEL_PATH = "configs/cost_calib/model_v7.json"

#: 冻结的模型 revision（`main` 是引用不是版本；不钉死就无法把两批归到同一权重）。
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

#: ★ S01 开关（**唯一**在真机上被行使的执行优化）。两条同开 = "P1-a+b"。
S01_KNOBS = {"logits_last_only": True, "sample_needed_only": True}

#: 与三条自适应臂同起点（静态基线若换起点，量的是"换了 chunk 大小"而非控制器）。
STATIC_SMALL_CHUNK = 1024

#: ★ 本批的 `max_num_seqs`：pilot 的前提是**先**把它改成 8（现 `p1s3d` 是 32）。
MAX_NUM_SEQS = 8


def _cost_model_lineage() -> dict[str, dict]:
    """三版成本模型的**血缘**（写进文件头 + 离线报告用）。

    ⚠️ 血缘字段的权威来源是**模型文件自己**（`grid_sha256`/`n_records`/
    `fit_on_sessions`），**不是**分析产物、也不是记忆。
    """
    out: dict[str, dict] = {}
    for v in ("v5", "v6", "v7"):
        p = _ROOT / "configs" / "cost_calib" / f"model_{v}.json"
        if not p.is_file():
            out[v] = {"present": False}
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        out[v] = {
            "present": True,
            "model_version": d.get("model_version"),
            "grid_sha256": d.get("grid_sha256"),
            "n_records": d.get("n_records"),
            "fit_on_sessions": d.get("fit_on_sessions"),
            "n_modes": len(d.get("modes") or []),
        }
    return out


_HEADER = """# ⚠️ **派生文件，请勿手改** —— 由 scripts/gen_prereg_batch_configs.py 生成。
# 来源：configs/mthreads/phase1_step14/{src}（**已执行的冷冻配置**；只派生、不改它）
# 本批统一项（显式写死，**不靠默认值**）：
#   · S01 开关      : logits_last_only={s01a}  sample_needed_only={s01b}
#   · 成本模型      : {cmp_}   （v7 = {v7grid} / {v7n} records）
#   · 内部每步预算  : {step_ms:g} ms —— 由 `controller.target_tpot_ms` **派生**
#                     （`arm_factory` 把它映射成 `CostPolicyConfig.step_deadline_ms`）
#   · prefill 预算  : prefill_deadline_ms = {pf_ms:g} ms（服务 TTFT，与 run.slo.ttft_ms 同源）
#   · 外部 SLO      : tpot_ms = {slo_tpot:g} ms  ttft_ms = {slo_ttft:g} ms（**判据**，非控制目标）
#   · 模型 revision : {rev}（钉死）
#   · max_num_seqs  : {mns}（pilot 前提）
# 差异：{desc}
# 用法：python -m benchmark.run_benchmark --config {rel} --dry-run
# ⚠️ 本文件**只**为机制小批/pilot 准备；它**不是**"已经跑过"的证据，也不含任何测量值。
# ⚠️ `--dry-run` **验不了** S01、过载与 SLO：dry-run 从不喂 decode_samples ⇒ 判据恒"不可测"。
"""


def _set_path(lines: list[str], path: tuple[str, ...], value: str) -> list[str]:
    """按键路径赋值，**逐层缩进**；目标键存在就改值，不存在就按其父段缩进插入。

    ⚠️ 只支持**已存在**的段（`section:` 首列）；不新建顶层段（避免悄悄造配置）。
    找不到父段 ⇒ 直接报错，**绝不静默跳过**。
    """
    out = list(lines)
    indent_unit = "  "
    idx: int | None = None
    depth = 0
    for i, ln in enumerate(out):
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        stripped = ln.strip()
        cur_depth = (len(ln) - len(ln.lstrip())) // len(indent_unit)
        key = stripped.split(":", 1)[0].strip()
        if cur_depth == depth and key == path[depth]:
            if depth == len(path) - 1:
                idx = i
                break
            depth += 1
    if idx is None:
        # 允许"父段存在、键不存在" ⇒ 在父段末尾插入（用父段缩进）
        return _insert_under(lines, path, value)
    indent = out[idx][: len(out[idx]) - len(out[idx].lstrip())]
    out[idx] = f"{indent}{path[-1]}: {value}"
    return out


def _insert_under(lines: list[str], path: tuple[str, ...], value: str) -> list[str]:
    """把 `path[-1]: value` 插到父段（`path[:-1]`）的最后一行之后。"""
    parent = path[:-1]
    depth = 0
    parent_line: int | None = None
    for i, ln in enumerate(lines):
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        cur_depth = (len(ln) - len(ln.lstrip())) // 2
        key = ln.strip().split(":", 1)[0].strip()
        if cur_depth == depth and key == parent[depth]:
            if depth == len(parent) - 1:
                parent_line = i
                break
            depth += 1
    if parent_line is None:
        raise SystemExit(f"父段 {'.'.join(parent)} 不存在 ⇒ 派生前提不成立")
    # 父段结束 = 后续第一行缩进 <= 父段缩进
    p_indent = len(lines[parent_line]) - len(lines[parent_line].lstrip())
    end = parent_line + 1
    while end < len(lines):
        ln = lines[end]
        if not ln.strip():
            end += 1
            continue
        ind = len(ln) - len(ln.lstrip())
        if ind <= p_indent and ln.strip() and not ln.lstrip().startswith("#"):
            break
        end += 1
    child = " " * (p_indent + 2)
    out = lines[:end] + [f"{child}{path[-1]}: {value}"] + lines[end:]
    return out


def _derive(src_name: str, *, desc: str, rel: str,
            edits: list[tuple[tuple[str, ...], str]]) -> str:
    text = (SRC_DIR / src_name).read_text(encoding="utf-8")
    lines = text.splitlines()
    for path, value in edits:
        lines = _set_path(lines, path, value)
    body = "\n".join(lines).rstrip("\n") + "\n"
    v7 = _cost_model_lineage()["v7"]
    return _HEADER.format(
        src=src_name, desc=desc, rel=rel,
        s01a=str(S01_KNOBS["logits_last_only"]).lower(),
        s01b=str(S01_KNOBS["sample_needed_only"]).lower(),
        cmp_=COST_MODEL_PATH, v7grid=v7.get("grid_sha256"), v7n=v7.get("n_records"),
        step_ms=INTERNAL_STEP_BUDGET_MS, pf_ms=PREFILL_DEADLINE_MS,
        slo_tpot=EXTERNAL_TPOT_SLO_MS, slo_ttft=EXTERNAL_TTFT_SLO_MS,
        rev=MODEL_REVISION, mns=MAX_NUM_SEQS,
    ) + body


def _common() -> list[tuple[tuple[str, ...], str]]:
    """三条 cost 族配置的**共同**显式项（每一条都不靠默认值）。

    ⚠️ **不写 `controller.step_deadline_ms`**：它不是 `ControllerConfig` 的字段
    ⇒ 写进去会让整个配置**加载失败**（`_build_dataclass` 对未知字段直接报错）。
    它由 `arm_factory` 从 `controller.target_tpot_ms` **派生**
    （`benchmark/arm_factory.py:99`）⇒ 要让"内部每步预算 = 120 ms"成立，
    **唯一**的做法是把 `target_tpot_ms` 写成 120。这正是"实际配置"与
    "内部预算"之间那一步容易断掉的地方。
    """
    return [
        # —— 外部 SLO（判据）——
        (("run", "slo", "tpot_ms"), f"{EXTERNAL_TPOT_SLO_MS:g}"),
        (("run", "slo", "ttft_ms"), f"{EXTERNAL_TTFT_SLO_MS:g}"),
        # —— 内部预算（控制目标；`step_deadline_ms` 由它派生）——
        (("controller", "target_tpot_ms"), f"{INTERNAL_STEP_BUDGET_MS:g}"),
        (("controller", "target_ttft_ms"), f"{EXTERNAL_TTFT_SLO_MS:g}"),
        (("controller", "cost_model_path"), COST_MODEL_PATH),
        # —— S01（真机才行使）——
        (("runtime", "logits_last_only"), str(S01_KNOBS["logits_last_only"]).lower()),
        (("runtime", "sample_needed_only"), str(S01_KNOBS["sample_needed_only"]).lower()),
        # —— 身份 ——
        (("runtime", "model_revision"), MODEL_REVISION),
        # —— 本批 cell ——
        (("scheduler", "max_num_seqs"), str(MAX_NUM_SEQS)),
    ]


VARIANTS: dict[str, tuple[str, str, list[tuple[tuple[str, ...], str]]]] = {
    "cost_single": (
        "fixed_cost.yaml",
        "**单预算**：`prefill_deadline_ms: null`（⇒ 与引入双预算前逐字节同行为）"
        "；其余 = 本批统一项。"
        "⚠️ 与 `cost_dual` **只差这一项**（单变量）。",
        [*_common(), (("controller", "prefill_deadline_ms"), "null")],
    ),
    "cost_dual": (
        "fixed_cost.yaml",
        f"**双预算**：`prefill_deadline_ms: {PREFILL_DEADLINE_MS:g}`"
        "（= 与 `run.slo.ttft_ms` 同源）"
        "；其余 = 本批统一项。"
        "⚠️ 与 `cost_single` **只差这一项**（单变量）。",
        [*_common(), (("controller", "prefill_deadline_ms"),
                      f"{PREFILL_DEADLINE_MS:g}")],
    ),
    "static_small": (
        "fixed_static.yaml",
        f"**静态小 chunk**：开分块 + `chunk.chunk_size: {STATIC_SMALL_CHUNK}`"
        f"（= 自适应臂的 `chunk_size_init`，同起点）；控制器关闭。"
        "⚠️ S01 仍显式打开（本批统一），但它**不是**被检验的变量。",
        [
            (("chunk", "enabled"), "true"),
            (("chunk", "chunk_size"), str(STATIC_SMALL_CHUNK)),
            (("controller", "enabled"), "false"),
            (("scheduler", "max_num_seqs"), str(MAX_NUM_SEQS)),
            (("run", "slo", "tpot_ms"), f"{EXTERNAL_TPOT_SLO_MS:g}"),
            (("run", "slo", "ttft_ms"), f"{EXTERNAL_TTFT_SLO_MS:g}"),
            (("runtime", "logits_last_only"), str(S01_KNOBS["logits_last_only"]).lower()),
            (("runtime", "sample_needed_only"),
             str(S01_KNOBS["sample_needed_only"]).lower()),
            (("runtime", "model_revision"), MODEL_REVISION),
        ],
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="只校验：现有文件必须与派生物逐字节一致")
    args = ap.parse_args()

    if not SRC_DIR.is_dir():
        print(f"[gen-prereg-batch] ✗ 来源目录不存在：{SRC_DIR}", file=sys.stderr)
        return 2
    OUTDIR.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    for name, (src, desc, edits) in VARIANTS.items():
        rel = f"configs/mthreads/prereg_batch/{name}.yaml"
        text = _derive(src, desc=desc, rel=rel, edits=edits)
        target = OUTDIR / f"{name}.yaml"
        if args.check:
            if not target.exists():
                problems.append(f"{target.name}: 缺失")
            elif target.read_text(encoding="utf-8") != text:
                problems.append(f"{target.name}: 与派生物不一致（有人手改过？）")
            continue
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"写出 {target.relative_to(_ROOT)}（来源 {src}）")

    if args.check:
        if problems:
            print("不一致：")
            for p in problems:
                print("  -", p)
            return 1
        print(f"通过：{len(VARIANTS)} 份派生配置与来源一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
