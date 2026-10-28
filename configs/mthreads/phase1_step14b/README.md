# `phase1_step14b` —— 第 14 步配置的**派生副本**（只差"用哪个成本模型"）

## 为什么会有这个目录

第 14 步的配置（`../phase1_step14/`）里 `cost_model_path` 指向 **`model_v5.json`**，
而 v5 的 `mixed_serving` 标定点是**旧口径**（探针按"每条序列各做 `prefill_tok` 个"采，
而 harness 的 `tokens_prefilled` 是**全步总量** ⇒ 差约 `batch` 倍）。
v6 把该口径修成 `real_mixed`（1 条做全部 + 其余各 1 个）⇒ 这一族成本**只能**用 v6 模型。

⚠️ 实测踩到的**闸门/实跑脱节**：v6 链的覆盖率闸门显式传 `--model model_v6.json`，
但紧接着的批（`run_step3_matrix.py --cfg-dir configs/mthreads/phase1_step14`）读的是**配置**，
于是变成 **"用 v6 过闸、用 v5 开跑"**，而两者指纹相同 ⇒ **不会报错**。
⇒ 结论：`S14E`（tag）**没有**检验语义修复；要检验必须让**配置**指向 v6。

## 这个目录的唯一区别

**除 `controller.cost_model_path` 一行外，逐字节等于 `../phase1_step14/` 的对应文件。**
（`static` / `threshold` 两臂**本来就不加载模型** ⇒ 它们与第 14 步的对应文件完全相同，这是预期。）

这条"只差一行"由 `tests/test_step14b_configs.py` **强制**：任何其它改动都会让用例失败 ——
免得派生副本慢慢漂成"第二份真相"，而那正是本项目最不想要的东西。

## 冻结方案在哪

**权威源只有一处：`../phase1_step14/FREEZE.md`**（本目录不复制它，避免第二处真相）。
本目录不引入任何新决定；`signal_criterion` / 门限 / SLO / 形状等一律沿用第 14 步的冻结值。

## 用法

```bash
python scripts/run_step3_matrix.py \
    --cells fixed_static fixed_threshold fixed_cost fixed_slo_weave \
    --cfg-dir configs/mthreads/phase1_step14b --repeats 5 --tag S14F
```

⚠️ 前置条件：`configs/cost_calib/model_v6.json` **必须已存在**（由 v6 链导出）。
本目录出现在它之前是刻意的 —— 这样"配置齐了"不会成为"模型好了但没配置"的等待项。
