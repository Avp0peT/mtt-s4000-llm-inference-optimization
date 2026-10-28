# 设备侧探针产物（2026-09-22，MTT S4000）

**这不是 harness 产物。** 这里的 JSON 由**设备侧探针脚本**直接生成，没有经过
`benchmark/run_benchmark.py`，因此**没有被 harness 盖章**、也没有 `meta.json`。
按仓库口径它们是「测量条件已记录、但未走 harness 标签体系」的原始记录，
**不得写进 `EXPERIMENTS.md` 的结论行，也不得当作 `MEASURED` 数据使用**。

## 为什么放在 `results/raw/` 下

`results/raw/` 的定位是**证据**：「丢了它，论文里的每个数字都变成不可复核的断言」
（见 `.gitignore` 的注释）。本轮所有设备数字都引在 `RESEARCH_LOG.md` 的 2026-09-22 条目里，
而它们原本只存在于**远程容器**的 `/root/ascend_probe/`——容器是临时的，
不落盘就等于把这些数字变成无法复核的断言。故按同一套逻辑落进证据目录。

`_` 前缀沿用 `_env/` 的约定，表示「不是一次 run」。`benchmark/collect_metrics.py`
的 `discover_runs()` 只认**含 `meta.json`** 的目录，因此这个目录不会被当成 run 聚合。

**本目录不含环境指纹**：`results/raw/_env/` 被 `.gitignore` 有意排除（指纹是某台机器
某一刻的本地状态，随仓库分发会被误读成「本项目的运行环境」）。主机事实的持久记录
只在 `README.md` §9 与 `RESEARCH_LOG.md`。

## 文件清单

| 文件 | 它证明了什么 |
| --- | --- |
| `qwen_0.5b.json` | 探针 **v1** 的 0.5B 结果。**保留是为了留痕一个会误导的口径**：v1 只有一个 `load_s = 215.74`，看起来像「0.5B 载入要 215 秒」；v2 拆开阶段后同一台机器是 7.21 s。这 215 s 是页缓存冷，不是设备慢。 |
| `qwen_0.5b_v2.json` | 探针 v2 的 0.5B 结果：分阶段载入、预热移出计时窗口、逐 token 时间戳。TTFT 48.6 ms / TPOT 47.3 ms。 |
| `qwen_0.5b_v2_correctness.json` | 0.5B 的 fp32 首前向 MUSA-vs-CPU 对照（`max_abs_diff = 2.03e-4`、argmax 一致）。与 v1 的同项**逐位相同**，佐证测量可复现。 |
| `qwen_7b_run1.json` | 7B 第 1 次（权重刚下完，页缓存冷）：载入 8.48 s、显存增量 14737.3 MiB、峰值 14577.0 MiB、TTFT 90.06 ms、TPOT 55.3 ms、P99 61.3 ms。 |
| `qwen_7b_run2.json` | 7B 第 2 次。**这份里 `correctness_fp32_first_forward` 是 `{"error": "NameError: ..."}`、而进程退出码是 0** —— 探针缺陷的原样证据，保留不删（成因见下）。P99 82.9 ms。 |
| `qwen_7b_run3.json` | 7B 第 3 次，探针修好后带 `--compare-cpu`：`failed_blocks: []`、0 退出码、fp32 对照 `max_abs_diff = 8.77e-4` / argmax 一致。P99 65.5 ms。 |
| `mem_bandwidth.json` | 设备搬运带宽实测，**本目录是这两个数字的权威来源**：256 MiB copy **0.772 ms / 695.4 GB/s**、2048 MiB copy **6.6272 ms / 648.1 GB/s**。用来**否掉**「固定开销 + 参数流」两点拟合（它会要求 ~1.4 TB/s）。⚠️ `RESEARCH_LOG.md` 初稿曾把 2048 MiB 档写成 `8.4374 ms / 509.0 GB/s`——那是**另一次测量**，与入库值相差 27% 且无入库产物可复核；正文已按本文件更正。正文引用带宽时一律引本路径，不再另存第二份（成因见 RESEARCH_LOG「评审更正」更正 2）。 |
| `scripts/` | 生成上述结果的探针脚本，以及环境诊断脚本（GLIBCXX 装载顺序、依赖来源核对等）。 |

## 与 `qwen_7b_run2.json` 那个 `error` 字段有关的两件事

1. **它是探针自身的缺陷，不是设备的问题。** v2 把 transformers 的 import 收进局部函数
   `_timed_phases()`，而 fp32 对照块在 `main()` 里用 `AutoModelForCausalLM` ⇒ 整块
   `NameError`。它被 `try/except` 吞成一个 `error` 字段，**进程退出码仍是 0**，
   于是链式脚本打印「全部完成」—— 一次「输出正确性已验证」差点没有任何依据。
2. **修法不是只补那句 import**：顶层加 `failed_blocks` 显形字段，有失败则**返回退出码 3**；
   并做**双向验证**（故意删掉那句 import 复现原缺陷、确认退出码 0→3；再跑修好的版本）。
   `qwen_7b_run2.json` 与 `qwen_7b_run3.json` 的对照就是这组 A/B 的留档。

## 复现方式

```bash
# 在**执行推理的那台机器**上（本机不是推理环境）
source /root/musa-envs/qwen310/activate_musa.sh   # 必须：设 LD_PRELOAD 等，见 RESEARCH_LOG
python scripts/qwen_single_request.py --model Qwen/Qwen2.5-7B-Instruct --repeats 3 --compare-cpu
python scripts/mem_bandwidth.py
```

退出码约定：**0 = 所有请求的测量都产出了数**；**3 = 有测量环节没有产出**（见
`failed_blocks` 字段）。**不能用退出码 0 之外的任何结果去说「验证通过」。**
