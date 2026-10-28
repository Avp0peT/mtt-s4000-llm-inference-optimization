# 前向执行成本探针（2026-09-24，MTT S4000）—— 第 1 步「执行成本画像」

**这不是 harness 产物。** 这些脚本不经过 `benchmark/run_benchmark.py`，没有
`meta.json`、没有标签盖章 ⇒ **不得写进 `EXPERIMENTS.md` 的结论行，也不得当作
`MEASURED` 数据使用**。它们的用途只有一个：为「S4000 执行成本驱动的控制器」
提供**标定常量**与**可预测量的形式**。

汇总报告见仓库根 `COST_PROFILE_S4000.md`。

`_` 前缀沿用 `_env/` 的约定，表示「不是一次 run」。`benchmark/collect_metrics.py`
的 `discover_runs()` 只认**含 `meta.json`** 的目录，因此本目录不会被当成 run 聚合。

## 产物

| 文件 | 版本 | 内容 |
| --- | --- | --- |
| `fwd_cost_profile.json` | v3 | 25 组成本分解：padding 浪费、分组加和性、prefill 单价、KV 长度、batch 规模 |
| `fwd_stability_probe.json` | v4 | 冷→暖漂移定位（ramp / empty_cache / 换形状 / 稳态 / build-only / KV 扫描）+ gmi 时钟与温度采样 |
| `fwd_faststate_probe.json` | v5 | async 块标定表、sync-vs-async 对照、MIMIC（复刻 harness 协议） |
| `fwd_datapower_probe.json` | v6 | 否证两个假设：数据内容/功耗、每步重建 past + 真实 KV 生长 |

`scripts/` 里是对应的探针源码（远端 `/root/ascend_probe/adhoc/` 的同名文件）与
三个本机分析脚本（`analyze_stability.py` / `analyze_faststate.py` / `dump_raw.py`，
只读，不重写产物）。

## 两条"探针可信"的验证

1. **结构上与 harness 同构**：`benchmark/real_backend.py` 的 `t_fwd0` 在 `:1208`、
   `_build_past_for_hf` 在 `:1218` ⇒ **HF 缓存重建被计入 `fwd_ms`**。探针 v3+ 照此
   把"每次前向重建缓存对象"放进计时段，两边测的是同一个东西。
2. **数值上吻合 0.01%**：探针 prefill（b=2 / chunk=2048）= **864.286 ms**，
   harness telemetry 里 `prefill_tok=4096` 那步的 `fwd_ms = 864.37`。

## ⚠️ 最重要的一个测量陷阱：设备有多性能态，由占空比决定

- b1 快态 ~57–60 ms / 慢态 ~76–82 ms；b2 快 ~74.3 / 慢 ~81。**态在秒级双向翻转。**
- harness 连续步进（~85% 占空比）**全程锁快态**（p10–p90 = 59.77–60.05 ms）；
  探针 sync-per-call（~65% 占空比、~40 ms python 间隙）在两态间振荡；
  阶段边界（1–2 s 空闲）打回慢档。
- **三个候选根因全被排除**：gfx clock 恒 1660 MHz（min=max，上限 1791）、
  温度恒 42 C、`mthreads-gmi` 无其他 GPU 进程 ⇒ 不是 DVFS、不是热、不是争抢。
  **机制未完全解释，已决定不再追。**

⇒ 可信协议的形状：**async 块**（连续提交 N 次、块尾 sync 一次 ⇒ 占空比 ~100%）
+ warmup 吸收一次性编译（首调 2694 ms）+ 取块最小值。快态下重复性 **±0.3%**。

## ⚠️ 标定纪律（长期有效）

> **形状比值可信（会话内 ±0.3%）；绝对值跨会话漂移 1.3–4%。**
> **成本模型须用同会话内对照，不得跨会话并表。**

## 复跑方法（远端）

```bash
source /root/musa-envs/qwen310/activate_musa.sh      # 必须用 bash -c，不能 sh -c
cd /root/ascend_probe/adhoc
python -u fwd_faststate_probe.py > /root/fwd_faststate.log 2>&1
```

⚠️ `import torch.musa as musa` 会失败，必须 `import torch_musa` 后用属性访问
`musa = torch.musa`；⚠️ 跨计时调用**不得**复用同一个 `use_cache=True` 缓存对象
（会原地追加 KV 并在 MuDNN SDPA 崩溃）；⚠️ 没有 `torch.no_grad()` 时 batch=8 会 OOM。
