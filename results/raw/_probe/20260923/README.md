# 设备侧探针脚本（2026-09-23，MTT S4000）—— V3/V4 分块预填充调查链

**这不是 harness 产物。** 这些脚本不经过 `benchmark/run_benchmark.py`，没有
`meta.json`、没有标签盖章。按仓库口径它们属于「测量条件部分记录、未走 harness
标签体系」的记录，**不得写进 `EXPERIMENTS.md` 的结论行，也不得当作 `MEASURED`
数据使用**。

## 为什么这批脚本会出现在这里

它们原先只存在于**远程容器**的 `/root/ascend_probe/scripts/` 里（在远端创建、
从未提交）。2026-09-23 补版本追溯时，部署戳（`SOURCE_PROVENANCE.json`）要求
「远端在摘要作用域（`src/ascend_opt` / `benchmark` / `scripts`）内与本机
**逐字节**相同」——这 13 个文件不在本机 `scripts/` 里，校验必然失败，
且失败原因正是"两个 `scripts/` 目录不是同一棵树"。

处置方式（用户 2026-09-23 决定）：

* **取回**本目录留档（沿用 `results/raw/_probe/20260922/` 的既有做法）；
* 在远端把它们移到 `/root/ascend_probe/adhoc/`——**不删除**，仍可复跑，
  只是移出「被部署源码」的作用域。

`_` 前缀沿用 `_env/` 的约定，表示「不是一次 run」。`benchmark/collect_metrics.py`
的 `discover_runs()` 只认**含 `meta.json`** 的目录，因此本目录不会被当成 run 聚合。

## ⚠️ 本批**没有**对应的 JSON 产物

这是与 20260922 批的一个实质差别，如实记下：这些脚本把读数 `print` 到 stdout，
**没有落盘产物**（实测：13 个脚本里只有 `mem_bandwidth.py` 与
`qwen_single_request.py` 会写 JSON，而那两个属于 20260922 批）。因此本批次里
每一个中间读数（top1−top2 差值、KV 长度、逐档批规模的通过/失败）**都不可复核**，
只有 `RESEARCH_LOG.md` 里的叙述性记录。要补上这个缺口，只能重跑探针并把输出落盘；
在此之前，**不要**把本批提到过的任何具体数字当作可直接引用的证据。

## 文件清单（描述取自各文件自身的 docstring）

| 文件 | 它要回答什么 |
| --- | --- |
| `_chk_tie.py` | prefill 分布逐位相同（差 0.0）但 token 序列分岔——分岔步是不是近平局？ |
| `_probe_gap.py` | 在分岔步上量 top1 与 top2 的差距，看是否与数值扰动（~0.03）同量级 |
| `_probe_logits.py` | **决定性判据**：分块与整段的最后一位 logits 差多少——是 fp16 舍入量级（~1e-2）还是真错（>1） |
| `_probe_repeat.py` | 同一配置重复跑，区分「确定性缺陷」与「抖动」 |
| `_probe_trace.py` | 拦 `model.forward`，跟踪分块 prefill 每次的 KV 长度与 `position_ids`（对照 [20,20] 过 / [16,16,8] 挂） |
| `_dbg_v3.py` | 诊断新版 V3 判据为何给出 1e3 量级差异 |
| `_probe_v3.py` | 隔离失败来源：驱动单跑 vs 放进变长批（A 参考 / B 驱动单跑 / C 同批带 padding） |
| `_probe_v3b.py` | 批规模是不是变量：2 行过、4 行挂，逐档加行定位 |
| `_probe_v3c.py` | 扫两两组合定位精确触发条件（16+48 过 vs 16+24 挂），并打印每行 prompt 末 token |
| `_probe_v3d.py` | 失败是否确定性（5/5）；观察到 `drv` 比 `ref` 多一个 token、从下标 4 起错位 |
| `_probe_v3e.py` | **决定性**：是驱动错，还是**参考实现本身**在长生成下不稳定（用 fp32 参考重跑同一 prompt） |
| `_probe_v4.py` | 分块 prefill 与整段 prefill 是否自洽。docstring 明确记着上一版探针**把 API 用错了**（末块 `need_token=False` 后无法 decode）、驱动以明确报错拒绝，本版按正确用法重写（末块 `need_token=True`，拿到第 1 个 token 后再逐 token decode） |
| `_t_single_variable_guard.py` | **正向对照**：故意在 adaptive 臂上多改一个字段（`chunk.max_num_batched_tokens`），使 `static_chunk → adaptive` 的差异从 1 处变 2 处，自检必须**以非零退出码**报出来 |

两点值得单独指出（都与本轮的"验收"主线有关）：

1. `_probe_v4.py` 的 docstring 是本轮**修复 #1**（V4 在分块 prefill 后立即释放请求、
   未继续 decode）的直接前史：问题被记下来了，正确用法也被写进了脚本，
   但**结论没有落成产物**——这正是"记录在叙述里、而不在产物里"的形态。
2. `_t_single_variable_guard.py` 是本仓库"守卫必须能失败"这条纪律的一次具体实践：
   它不是检查配置，而是**证明检查会红**。

## 另外 10 个脚本不在这里（已另有存档）

`check_versions.py`、`mem_bandwidth.py`、`qwen_single_request.py`、
`libstdcxx_order_probe.py`、`which_libstdcxx.py`、`probe_deps.py`、`verify_env.py`、
`import_check.py`、`download_7b.sh`、`run_after_download.sh` 同样只存在于远端
`scripts/` 里，但它们**已经**在 `results/raw/_probe/20260922/scripts/` 有存档
（内容逐字节相同），故本次不重复取回，只在远端一并移入 `adhoc/`。
逐文件说明见那份 README。

汇总：远端 `/root/ascend_probe/adhoc/` 现在共有 **23** 个脚本（13 本批 + 10 上一批）。

## 复现方式

```bash
# 必须在**执行推理的那台机器**上（本机不是推理环境），且 cwd 为远端源码树根：
#   脚本里 sys.path.insert 用的是相对路径 "src" / "benchmark" / "scripts"
source /root/musa-envs/qwen310/activate_musa.sh   # 必须：设 LD_PRELOAD 等，见 RESEARCH_LOG
cd /root/ascend_probe
python3 adhoc/_probe_v4.py      # 例：分块 prefill 自洽性
python3 adhoc/_t_single_variable_guard.py   # 例：证明单变量自检会失败
```

⚠️ 这些脚本是**一次性调查脚本**（含 v3/v3b/v3c/v3d/v3e 这样逐步演进留下的多份），
不是可维护的工具。重跑前先看 docstring 里它当时要回答的问题是否还成立。
