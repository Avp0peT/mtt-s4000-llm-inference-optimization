#!/usr/bin/env python3
"""把 `paper/sections/*.md` 装配成一份**可整篇阅读的论文工作稿** `paper/manuscript.md`。

为什么需要它
------------
论文正文按节写在 `paper/sections/*.md`（10 个文件），这是**权威源**；
但审阅 / 交给导师时需要**一份连续的文章**。手工拼两份必然分叉，
所以这里把它做成**可重复执行的构建**：每次运行都从各节重新生成，
输出文件头部明确声明"**本文档是构建产物，勿手改**"。

§4 的处理
---------
`paper/sections/04_failure_analysis.md` **只是一个指针**，§4 的正文在
`paper/failure_analysis.md`（用【事实】/【解释】/【假设】三层标注写）。
装配时把该文件从 `## 4.1` 起的正文内联进来，**不复制**其工作文档头
⇒ 事实仍然只有一处（`failure_analysis.md`）。

三条纪律（与项目其余工具一致）
------------------------------
1. **空输入 ⇒ 非零退出**：少一节就报错，不允许"少一节也照样装配"；
2. **必须报出占位总数并据此返回码**：`rc=1` 表示"还有待补"（那是投稿前正常状态，
   不是错误）；`rc=0` 才表示可以进入投稿评审；
3. **不静默改写内容**：只做拼接与必要的小标题前插，不改动任何一节里的字句。

用法::

    python scripts/build_manuscript.py            # 装配并打印摘要
    python scripts/build_manuscript.py --check    # 只检查，不写文件

退出码：0 = 已装配且无占位；1 = 已装配但有占位；2 = 输入错（缺文件 / 缺节）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SECTIONS = _ROOT / "paper" / "sections"
_FAILURE = _ROOT / "paper" / "failure_analysis.md"
_OUT = _ROOT / "paper" / "manuscript.md"

#: 装配顺序：文件名 → 章标题（`None` = 用文件里已有的 `# ` 标题）。
_ORDER: list[tuple[str, str | None]] = [
    ("00_abstract.md", None),
    ("01_introduction.md", None),
    ("02_related_work.md", None),
    ("03_platform.md", None),
    ("04_failure_analysis.md", "4 成本模型失效分析（阶段二）"),  # 正文来自 failure_analysis.md
    ("05_improvements.md", None),
    ("06_scheduling.md", None),
    ("07_benefits.md", None),
    ("08_limitations.md", None),
    ("09_conclusion.md", None),
]

_HEADER = """\
# 有限标定数据下的 LLM 推理成本预测可靠性：失效归因、改进与调度代价
# —— 基于 MTT S4000 的实证研究

*Cost-Model Reliability for LLM Inference under Limited Calibration Data:
Failure Attribution, Improvements, and Scheduling Cost — An Empirical Study on MTT S4000*

> ⚠️ **本文档是构建产物，请勿手工编辑。**
> 它由 `scripts/build_manuscript.py` 从 `paper/sections/*.md`（各节正文，**权威源**）
> 与 `paper/failure_analysis.md`（§4 正文）装配而成。要改内容，请改那些文件后重新装配。
>
> **阅读说明**：各节开头的 `> …` 引用块是**写作注记**（证据入口、口径纪律、占位来源），
> 供内部复核用；**投稿版本应删除**（保留会导致正文夹带工程注记）。
> 正文里的 `【待补·Pxx】` 是**占位标记**，表示该量级尚未产出（**不是 0、不是估计**）；
> 清点：`python scripts/list_paper_placeholders.py`。
>
> **证据纪律**：进入正文的每个数字都必须能指回 `EXPERIMENTS.md` 的一条**登记行**
> 且 `data_label = MEASURED`；`--dry-run`（`TEST_DATA`）产物一律不得出现。
> 无法达到该强度的量（如设备侧探针的成本画像）在文中**显式标注**为非 `MEASURED`，
> 且只作平台特征或下界参考，**不作**性能结论。
>
> **标题是暂定**：按本项目纪律，标题里的主张必须与结论一致；若最终收益侧证据不成立，
> 标题须退回"失效归因 + 适用边界"的表述，不得写成"自适应调度优化"。
"""

_REFERENCES = """\
## References

本节只列 `SOURCES.md` 中状态为 **`verified`** 的条目；`unverified` 的一律不引
（项目纪律：未核验的条目不得进入正文）。⚠️ "状态 = verified"指**该条目本身的信息已核对**
（标题 / 作者 / venue / 年份 / 机制 / 结论量级），**不**表示我们复现了它，
也**不**表示它的结论在本平台成立。

1. **SLOWeave** — *Deadline-Aware Adaptive Prefill Chunking for Efficient Large Language
   Model Serving*, arXiv:2609.07883（v1, 2026-09-07）. 与本文第三臂最接近（成本预测 +
   next-token deadline + 自适应 chunk）。本文实现了其算法作为**对照臂**，
   故**不**把该机制作为贡献。⚠️ 论文声称随文提供 artifact，但本次检索**未找到可访问仓库**
   ⇒ 我们的实现是**按论文写的 policy 参考版本**，**不得**冒称完整复现。
2. **Sarathi-Serve** — Agrawal 等, *Taming Throughput-Latency Tradeoff in LLM Inference
   with Sarathi-Serve*, OSDI '24. 分块预填充 + stall-free 调度；本文**固定分块静态基线**
   的来源。⚠️ 其开源实现自述为研究原型（功能不完整）⇒ 本文只借用其**调度规则**。
3. **DistServe** — Zhong 等, *Disaggregating Prefill and Decoding for Goodput-optimized
   Large Language Model Serving*, OSDI '24. 提供 **goodput** 的定义与讨论口径；
   其多设备分离架构不在本文范围内。
4. **DRLServe** — Sun, Bao, Wang, *Adaptive Prefill Chunking with Deep Reinforcement Learning
   for LLM Inference*, ECAI 2025（DOI 10.3233/FAIA250867）. 自适应 chunk 的**学习型**路线；
   本文不作强化学习，仅列为同族另一条路。

> 完整的"已有机制 — 本项目现有能力 — 拟研究差异 — 验证方法"对照表见 `SOURCES.md` §6。
> 【待补·P02】相关工作普查的完备性声明（覆盖到何时、检索方式）——见 §2 末。
"""

_APPENDIX_A = """\
## 附录 A · 证据索引（主张 ↔ 批次）

本表把正文里的**每一类事实**指回它所在批次；批次与 `run_id` 的登记在 `EXPERIMENTS.md`。
未列入本表的量一律视为**未建立**，不得引用。

| 正文出处 | 主张 | 批次 / 登记 | 强度 |
| --- | --- | --- | --- |
| §3.3 / §9.1 | 块 A（快态）与块 B（服务协议）差值随 prefill 增大（156.03 vs 166.85 ms） | `results/raw/_probe*`、`COST_PROFILE_S4000.md` | ⚠️ **非 `MEASURED`**（设备侧探针）⇒ 只作平台特征 |
| §3.4 / §6.6(a) | 控信号污染：读数 ≈ 5.5× 纯 decode；收缩把纯 decode 从 61→118 ms | `P8FULL`（E10） | `MEASURED`，3 轮 |
| §3.4 / §6.6(b) | 换判据把污染比从 5.47× 压到 1.33× | `S9AB`（E12/E13） | `MEASURED`，2 臂 × 3 轮（单变量） |
| §3.5 / §6.5 | 整步 130.52 = 前向 104.79 + host 25.25 ms；D1 实例 | `S14B1`（`decompose_step_overhead.py`、`diag_S14B1.json`） | `MEASURED`，5 重复 |
| §4.2 | 同一模型 1.77% ↔ 95.26%；秩保证划分 → 11.77%；2×2 解耦 ≈89pp / ≈5.7pp；满秩 6.12% | `results/processed/decouple_v1_v3.json`、`decouple_v3_v4.json`、`costcalib_eval_v4/v5.json` | 离线复算（可重建） |
| §4.2 | 覆盖率 53.57%→100% 与高估 p95 0.13%→9.76% 的取舍 | `lookup_bucket_sweep.json`、`coverage_lookup*_S14B1.json` | 离线复算（可重建） |
| §6.5 | `fixed` 格四臂 D0/D1/D2/D3 与达标吞吐全为 0 | `S14B1`（`step14_S14B1.json`、`diag_S14B1.json`） | `MEASURED`，5 重复 |
| §6.6(d) / §8.6 | 静态响应面；`Best Static` = 不分块；"最优不随段迁移"被否证 | `S10`（E14/E15） | `MEASURED`，5 候选 × 3 轮 |
| §6.2 | SLO 阈值标定（逐请求 TPOT 区间 [61.7, 168.0] ms）与门限标定 | `scripts/calibrate_slo.py`、`calibrate_controller_gate.py` | 量（重算，非重跑） |
| §5.1 / §5.2 / §5.3 / §5.4 | 各项机制的**设计**与**实现约束** | 源码 + `tests/`（用例强制单变量性） | L1 实现完成 |
| §7 | 服务收益 | 尚无 | ⬜ **未建立**（【待补·P29】） |

## 附录 B · 占位清单（尚未产出的量）

由 `scripts/list_paper_placeholders.py` 生成；**ID 全局唯一**，补数时按 ID 替换整行。
⚠️ 占位**不是 0**：0 是测量值，缺失是"没测"，两者语义相反，不得互相替代。
"""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section_body(path: Path, title_override: str | None) -> str:
    """返回该节正文；`title_override` 给定时用它替换文件里第一行的 `# ` 标题。"""
    text = _read(path).strip()
    if title_override is None:
        return text + "\n"
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body = "\n".join(lines).lstrip("\n")
    return f"# {title_override}\n\n{body}\n"


def _failure_body() -> str:
    """把 `failure_analysis.md` 从 `## 4.1` 起的正文内联进来（丢掉工作文档头）。"""
    text = _read(_FAILURE)
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## 4.1"):
            start = i
            break
    if start is None:
        raise SystemExit(f"✗ {_FAILURE} 里找不到 `## 4.1` 起的小节 ⇒ 未判（不静默拼半截）")
    return "\n".join(lines[start:]).strip() + "\n"


def _placeholder_block() -> tuple[str, int]:
    """复用占位清点器的扫描逻辑，生成附录 B 的清单；返回 (文本, 占位总数)。"""
    sys.path.insert(0, str(_ROOT / "scripts"))
    import list_paper_placeholders as lpp  # noqa: PLC0415

    found, _ids, lazy = lpp.scan(_SECTIONS)
    total = sum(len(v) for v in found.values())
    out: list[str] = []
    for name in sorted(found):
        out.append(f"**{name}**（{len(found[name])} 处）\n")
        for item in found[name]:
            out.append(f"- `【待补·{item['id']}】` L{item['line']} — {item['text'][:110]}")
        out.append("")
    if lazy:
        out.append("⚠️ 另有**偷懒写法**（禁止）：")
        for name in sorted(lazy):
            out.append(f"- {name}: {len(lazy[name])} 行")
    out.append(f"\n**合计 {total} 处**（rc=1 ⇒ 尚不能进入投稿评审）。")
    return "\n".join(out) + "\n", total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="装配 paper/manuscript.md")
    ap.add_argument("--check", action="store_true", help="只检查输入完整性，不写文件")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args(argv)

    missing = [n for n, _ in _ORDER if not (_SECTIONS / n).is_file()]
    if missing:
        print(f"✗ 缺少分节文件：{missing} ⇒ 未判（不装配半篇文章）", file=sys.stderr)
        return 2
    if not _FAILURE.is_file():
        print(f"✗ 缺少 {_FAILURE}（§4 正文的来源）", file=sys.stderr)
        return 2

    parts = [_HEADER.strip() + "\n"]
    for name, override in _ORDER:
        if name == "04_failure_analysis.md":
            parts.append("# 4 成本模型失效分析（阶段二）\n\n"
                         "> 本节正文由 `paper/failure_analysis.md` 装配（该文件是 §4 事实的唯一出处）。\n\n"
                         + _failure_body())
        else:
            parts.append(_section_body(_SECTIONS / name, override))
    parts.append(_REFERENCES.strip() + "\n")
    parts.append(_APPENDIX_A.strip() + "\n")
    ph_text, n_ph = _placeholder_block()
    parts.append(ph_text)

    doc = "\n\n---\n\n".join(p.strip() for p in parts) + "\n"

    if args.check:
        print(f"✓ 输入完整（{len(_ORDER)} 节 + References + 附录 A/B）；占位 {n_ph} 处")
        return 0

    out = Path(args.out)
    # 一律写入 LF：Windows 上 write_text 默认会把 \n 翻成 \r\n，而仓库要求 LF
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    n_lines = doc.count("\n") + 1
    print(f"✓ 已写出 {out}（{len(doc)} 字符 / {n_lines} 行）")
    print(f"  §4 正文内联自 {_FAILURE.name}；占位 {n_ph} 处")
    print(f"=== {'✓ 无占位' if n_ph == 0 else '✗ 尚有占位（rc=1，属正常）'} ===")
    return 1 if n_ph else 0


if __name__ == "__main__":
    raise SystemExit(main())
