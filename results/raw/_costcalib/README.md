# `_costcalib/`：成本标定探针产物（协议 `cost-calib-v1`）

⚠️ **这些不是 harness 产物**：没有 `meta.json`、没有 `data_label` 盖章。
按既有惯例（与 `_probe/`、`_env/` 同族）放在 `results/raw/` 下，但**读取者必须知道它们的身份不同**：

* **不得**把它们当作 `EXPERIMENTS.md` 结论行的证据；
* 结论只能进 `RESEARCH_LOG.md` 与第 11 步的完成记录；
* 它们的溯源来自产物内的 `protocol_version` / `session_id` / `grid_sha256` / `fingerprint`，
  而不是 harness 的戳。

## 文件

| 文件 | 会话 | 内容 |
| --- | --- | --- |
| `costcalib_v1_s1.json` | `20260925T084017Z-autodl-c-s1` | **v1 网格**（24 点，kv ≤ 2048） |
| `costcalib_v1_s2.json` | `20260925T090446Z-autodl-c-s2` | 同上 |
| `costcalib_v1_s3.json` | `20260925T091845Z-autodl-c-s3` | 同上 |
| `costcalib_v2_s1.json` | `20260925T140455Z-autodl-c-v2s1` | **v2 网格**（34 点，kv 扩到 4096） |
| `costcalib_v2_s2.json` | `20260925T142749Z-autodl-c-v2s2` | 同上 |
| `costcalib_v3_s1.json` | `20260925T173544Z-autodl-c-v3s1` | **v3 网格**（46 点，batch 展开 1..8）—— **被否**，见下 |
| `costcalib_v3_s2.json` | `20260925T180509Z-autodl-c-v3s2` | 同上 |

### ⚠️ v1 与 v2 的关系：**v1 是被采纳的那个，v2 只作"越界不可用"的证据**

* **`model_v1.json`（由 v1 拟合）是唯一有效模型**，其标定范围 `kv ≤ 2048` 即**适用边界**；
* **v2 的数据是有效的**（真实测量、0 错误），但它拟合出的模型**没通过自己的验收判据**：
  形状留出的**低估 p95 从 1.77%（v1）恶化到 18.04%（v2）** ⇒ 按预注册 §五 判**不成立**；
* 原因（定性）：模型 basis 对 `kv` **线性**，而 512→4096 上成本**不是线性的** ⇒
  v1 的窄范围掩盖了这一点；
* ⇒ **不导出 `model_v2.json`**。v2 的用途是**给"越界必须回退"提供实测证据**：
  如果不回退、硬用线性外推，误差就是这个量级。
* ⚠️ 要真正扩范围，必须**改 basis**（加 `kv²` 项 / 按 `kv` 分段）**并重新走预注册与验证**。

**读这两组文件的人请注意**：`v2` 的名字容易让人以为"更新、覆盖更大范围"，
实际上它是**被否掉的一轮**。别拿它当可用模型。

三个会话的网格 `grid_sha256` 相同 ⇒ 同一份取值点；**会话不同** ⇒ 可做跨会话留出与漂移。

### ⚠️ v3：**同样被 §五 判据否掉**（网格扩大 ≠ 精度提高）

* v3 的动机是修 **batch 维覆盖**：v1/v2 的 batch 只有 `{1,2,4,8}` 离散值，而真实服务
  连续取 1..8（`max_num_seqs=8`）⇒ v1 模型下 `cost` 臂的有效工作覆盖率只有 51.35%，
  回退成因 94.4% 是"batch 不在标定网格"。v3 把 `axes.batch` 展开为 1..8，
  并给 `batch≥2` 补上 prefill 点（原文见 `configs/cost_calib/shape_grid_v3.yaml`）。
* **形状范围确实修好了**：v3 实际记录到的范围为
  `b1: prefill[0,2048] kv[0,2048]`、`b2..b8: prefill[0,1024] kv[512,2048]`；
  把 `S14B1` 实测的 90 个回退形状逐条对照，**90/90 全部落入范围**
  ⇒ "覆盖率 51.35% → 100%" 在**形状范围**意义上成立。
* **但模型精度反而变差**（`scripts/costcalib_fit.py`；2 会话 / 92 条记录）：

  | §五 判据 | 阈值 | v1 | **v3** | 结果 |
  | --- | --- | --- | --- | --- |
  | 低估 p95（① 形状留出） | ≤ 5% | 1.77% | **5.15%** | ❌ **超阈** |
  | 跨会话漂移 p90 | ≤ 10% | 0.53%（max 0.62%） | 0.69%（max 0.77%） | ✅ 通过 |
  | \|相对误差\| p90（① 形状留出） | ≤ 10% | 2.41% | **5.69%** | ✅ 通过，但精度降 2.4× |

* ⇒ `verdict.model_holds = **False**`（理由只有一条："低估相对 p95 5.15% > 5%"）
  ⇒ **不导出 `model_v3.json`**。这两个文件是**被否掉的一轮**的原始测量证据，
  与 v2 同族：**别拿它们当可用模型**。
* ⚠️ **不要把它写成"覆盖率提升得不偿失"之外的因果**：本次只测到
  "网格 24→46 点后 ① 与 ③ 的低估尾部同时变差"，**成因未定位**，
  登记为**待查**（候选：`batch` 分段内 `[1, prefill_tok, kv_len]` 的线性 basis 对新点不足）。
  不得把它当成已确认结论。
* 与 v2 的否掉理由**不同**：v2 是**越界**（kv 扩到 4096，线性 basis 失效）；
  v3 是**在范围内加密**。两者都不导出模型。
* ⚠️ 复算与口径：`results/processed/costcalib_eval_v3.json` 是**本机**用
  `python scripts/costcalib_fit.py --artifacts "results/raw/_costcalib/costcalib_v3_*.json" --json …`
  跑的（与 v1 同一流程）。远端夜间队列的同一命令**路径写错**（见 `RESEARCH_LOG.md`），
  没有产出这个文件。**唯一权威 = 本机这次复算**。

## 两个块的角色（**不要混用**）

* `points[*].A` = `role: lower_bound_only` —— 占空比 ~100% 的**快态平台值**，
  只作**性能下界参考**，**不得用于拟合**；
* `points[*].B` = `role: fitted` —— **逐步 sync + KV 真实生长**（服务协议）下的每步成本，
  **被拟合的是这个**。

验收明确要求"不能只拟合快态最小值"；实测两者的差随 prefill 增大（例如 b1/pf512：
A 156.03 vs B 166.85 ms）。角色由 `summarize_block_a/_b` 带上，组装层不得改写。

## 复算

```bash
python scripts/costcalib_fit.py --artifacts "results/raw/_costcalib/*.json" \
    --json results/processed/costcalib_eval.json
```

采集协议与判据的原文在 `NEXT_SESSION_TASKS.md` 第 11 步（含"模型不成立"的阈值与
范围外回退规则）。
