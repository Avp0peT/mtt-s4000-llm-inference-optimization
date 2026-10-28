# `phase1_step15_v5` —— 第 15 步消融的 **v5 血缘**副本

## 为什么有这个目录（而不是直接用 `../phase1_step15/`）

`../phase1_step15/` 的 `cost_model_path` 指向 **`model_v6.json`**。
但第 14 步的三臂（`ramp` / `lenmix` / `fixed`）跑的是 **`model_v5.json`**
（见 `../phase1_step14/*.yaml`），而本项目**禁止跨版本混算**。

三模型的有效性证据（实测，见 `.workbuddy/memory/2026-09-28.md` §8）：

| model | n_records | `mixed_shape_mode` | verdict | 判定 |
| --- | --- | --- | --- | --- |
| **v5** | **188** | *(无)* = 旧 `mixed_serving` | 低估 p95 9.02% | **第 14 步在用** |
| v6 | 104 | `real_mixed` | 低估 p95 5.02% | **无效分支**：`real_mixed`+`batch>1` **84/84 全报错** |
| v7 | 220 | `real_mixed` | 低估 16.02% + 精度 p90 15.17% | `real_mixed` 的正当继任 |

⇒ **v6 不是"更好的 v5"**，而是一个**没有任何有效测点**的未验证分支
（`2026-09-26.md` 原话："把标定口径改成 real_mixed 这条修复，至今没有任何一个有效测点"）。
所以第 15 步消融的基线**只能取 v5**；要研究 `real_mixed` 口径请另起 **v7** 血缘。

## 生成方式（可重复，不是手抄）

```bash
python scripts/gen_phase1_step15_configs.py \
    --model configs/cost_calib/model_v5.json \
    --out-dir configs/mthreads/phase1_step15_v5
```

本目录**不得**手改：改了就跑 `--check` 复核（见下），或用同一条命令重新生成。

## 与 `../phase1_step15/` 的唯一区别

**每个文件的 `controller.cost_model_path` 一行**：
`model_v6.json` → `model_v5.json`。
其余（含 6 个消融条件本身、门限、SLO、负载、`host_overhead_ms` 等）**逐字节相同**。

★ **附带的一个有利性质**：因为 reference `../phase1_step14b/fixed_cost.yaml`
与本目录的 `fixed_cost_full.yaml` 也只差这一行，而在 v5 下二者**本该完全相同**
⇒ **`fixed_cost_full.yaml` 逐字节等于 `../phase1_step14/fixed_cost.yaml`**
（即第 14 步三臂在用的那份）。于是"消融对照的是被部署的系统本身"在这里成立得
**比原设计更强**：不仅同源，而且**就在同一版本血缘内**。

## ⚠️ v5 的局限（不得掩盖）

v5 的 `mixed_serving` **形状口径是错的**（探针按"每条序列各做 `prefill_tok` 个"采，
harness 的 `tokens_prefilled` 是**全步总量** ⇒ 差约 `batch` 倍）。
**该缺陷只影响 mixed 族**；本目录的消融负载是 `short_chat`（`fixed`）⇒ v5 可用。
**任何**关于 mixed 形状的结论**不允许**用 v5 推。

## `--min` 为什么没给（`chunk_only` / `budget_only`）

`chunk_only` / `budget_only` 两格要"冻结一维"，其"下限选到多少"依赖第 10 步
响应面的**预注册选择规则**（`NEXT_SESSION_TASKS.md` §三）。该规则**尚无有效输入**
⇒ 按纪律**不擅自填下限、不绕过依赖**，本目录沿用生成器默认（`chunk_size_min=512`），
并在结论里**显式标注**该分支未定标。

## 校验

```bash
# 与生成器一致（防手改）
python scripts/gen_phase1_step15_configs.py --check \
    --model configs/cost_calib/model_v5.json --out-dir configs/mthreads/phase1_step15_v5

# 单变量 + 非 controller 段逐字段相同（对照 ../phase1_step14/fixed_cost.yaml）
python -m pytest tests/test_step15_configs.py -q     # 针对 ../phase1_step15/ 的既有用例
```

## 冻结方案在哪

**权威源只有一处：`../phase1_step14/FREEZE.md`**（本目录不复制它，避免第二处真相）。
本目录不引入任何新决定。
