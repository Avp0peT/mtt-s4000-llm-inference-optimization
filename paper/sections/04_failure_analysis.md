# 4 成本模型失效分析（阶段二）

**本节正文见 [`paper/failure_analysis.md`](../failure_analysis.md)** —— 那里用
**【事实】/【解释】/【假设】** 三层标注写，并显式列出"刻意不写"的四项。

本文件是**指针**，不是第二节真相（本项目最忌两处各写一份同一段事实）。

## 本节在全文中的位置

* 上游：§3 给出的字段定义与判据口径（`prefill_tok` 是全步总量、判据是整步 ≤ 预算）。
* 下游：§5 的改进项由本节的失效归因**逐条对应**；§6 的 D0/D1/D2/D3 是本节的失效
  在调度层与服务层的表现。

## 写作时必须连带的三个约束（否则本节会被读大）

1. **三处成因并列，但不给"各自造成多少损失"** —— 那是【假设】。
2. **【事实】与【解释】必须分栏**：本项目已有一次教训（"−23% 不是模型缺特征，而是把 host 侧开销
   算进了模型误差"），正是因为把解释写成了事实。
3. **"没测出"与"测出坏的"分开**：例如"该格判据只能记「未判」"不等于"该格方法无效"。

## 两张图（P27）

（来源：`results/processed/decouple_*.json` 与 `lookup_bucket_sweep.json`。）

**状态更新（2026-09-28）**：**图已产出且可由脚本重建**，正文已写入本节所指向的
[`paper/failure_analysis.md`](../failure_analysis.md) §4.2 的「两张图（P27）」小节。

* 脚本：`scripts/plot_step16_decouple.py`（只读；`--check` 只校验）
* 产物：`paper/figures/fig_p27a_decouple.{png,svg}`、`fig_p27b_lookup_sweep.{png,svg}`
* 溯源：`paper/figures/fig_p27.source.txt`（两个输入的 `sha256` + 重建命令 + 读法）
* 用例：`tests/test_plot_step16_decouple.py`（13 passed；含缺字/CRLF/空输入三守卫）
* **run_id / tag**：图**不引入任何新 run**；图 B 的 `shapes_tag = S14B1`、
  `calib = results/processed/costcalib_eval_v4.json`；图 A 的逐格 `eval_json` 见
  `decouple_*.json`。

⇒ **已并入正文（2026-09-28）**：`paper/manuscript.md` **§4.8「两张图：划分方式而非网格密度；
瓶颈在 `kv` 桶」**（commit `31f0cc2`）。正文内容是本节「两张图」小节的**逐条复核版**：
图 A 的四个 `underest_p95` 值与图 B 的 20 格 `coverage`/`over_p95` 均**逐格核过 JSON**
——复核时发现初稿把跨代的 `0.1163`（v1→v3 旧网格）与 v3→v4 的 `0.0612` 混写成一次变化，
**已改正**并补上"改善非单调"的限定；`kv_step=0` 的实测 0 也按"不可用"标注保留。

⇒ **本占位块至此可关闭**（P27 全部完成：脚本 + 图 + `.source.txt` + 用例 + 正文并入）。
