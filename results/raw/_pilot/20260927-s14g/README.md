# S14G（D1① 可行性闸门）— **闸门判失败 + 全批 `NOT_MEASURED`** · 2026-09-27 03:06 取回

> ⚠️ **本目录里的产物：① 不构成 `MEASURED`；② 不能用于任何结论；③ 不得后补标签。**
> 它在这里的唯一用途是**保留现场**，供事后复核"闸门为什么这么判"。
> 收下它的动作本身也在提醒一件事：**"跑完了"与"可用"是两件事**。

## 1. 这是什么

远端 `/root/ascend_probe_d1`（2026-09-26 23:55 部署，戳 `git_commit=9df0aefa0c8818d9e584bafe628f43cf856b3748`、
`dirty=false`、`tree_digest=17da2762e3aa3f6a0c4fcf040683e855043fb1c908ea65738525bbb37c9e8455`）
上 D1 链跑到 **03:01:18** 结束，标记 `S14D1_DONE` / `S14D1_GATE_FAIL`。

本目录是那次运行的**逐字快照**（`tar -czf` 后 `get`，tarball md5
`5b2e7df555e536ef77b5e007c5d19472`，两端一致；保留了远端的相对路径结构）：

```
results/raw/S14G-short_chat-fixed_cost_{pilot,min512}{,-r1,-r2}/   # 6 个 run
results/processed/diag_S14G.json        # D0/D1/D2/D3 诊断
results/processed/costcalib_eval_v7.json # v7 模型五数 + verdict
adhoc/costcalib_v7_{pilot,s1,s2}.json   # v7 标定原始记录（16 / 111 / 111 点）
configs/cost_calib/model_v7.json        # v7 模型（已另行取回到 configs/cost_calib/）
```

## 2. 两个必须分开说的问题

### 2.1 全批 `data_label = NOT_MEASURED`（部署缺陷）

6 个 `meta.json` 一致：`data_label=NOT_MEASURED`、`label_source=None`、
`execution_environment=None`、`hardware_platform=None`。

**成因**：新根 `/root/ascend_probe_d1/results/raw/_env/` **不存在**
（`results/raw/_env/` 被 `.gitignore` 排除 ⇒ **每次部署到新根都会缺**；
旧根 `/root/ascend_probe/` 那份是 2026-09-24 22:43 单独采的）。
harness 原文：`未找到 env.json（查找过：[…]）；硬件可用性未知，无法判定为实测数据`。
**对照**：旧根的 `S14D-*` 同条件为 `MEASURED`。

⇒ 处置已记入 `PLAYBOOK.md` §5.0（新根部署的固定动作：部署完立刻在新根采一次指纹）。
⚠️ **不得**直接拷旧根的 `env.json`（容器已换，`hostname_or_machine_id` 不一致）；
⚠️ **不得**给本目录的产物后补 `MEASURED`（红线：回推**只允许降级、永不升格**）⇒ **只能整批重跑**。

### 2.2 闸门判失败，且**其中一条判据不可信**

链的原文：

```
[d1] pilot  = {'n_runs': 0, 'small_steps': 0, 'min_pf': [], 'n_degraded': [], 'd0': 0.0476, 'has_arm_state': False}
[d1] min512 = {'n_runs': 0, 'small_steps': 0, 'min_pf': [], 'n_degraded': [], 'd0': 0.0541, 'has_arm_state': False}
   - 条件1：小 chunk(128/256) 一步都没被实际执行
   - 条件2：arm_state/n_degraded **数不出来**（源码未部署到位）⇒ 不可判
   - 条件3：D0 可行率 4.8% 未超过 5.7%
[d1] ⇒ **不启动 ②**（按预注册的否证形态，先补标定）
```

★ **条件1 的措辞与实测冲突，必须打上问号**：`n_runs: 0` 说明闸门**一个 run 都没找到**
（最可能就是按 `data_label == MEASURED` 过滤，而 6 个 run 全是 `NOT_MEASURED`）
⇒ "小 chunk 一步都没被实际执行"是**"数不出来"**，**不是"没有发生"**。
**反证（同一次运行的日志）**：`fixed_cost_pilot` 打印 `chunk 建议=1024..2048 生效=128..1024`
⇒ **128 这个尺寸确实被下发并执行过**。
⇒ 这是"把缺失读成否定"的又一例：**闸门报了一个它没有证据支持的结论**。

⇒ 结论：**闸门该不该失败，目前不可判定**；但 `NOT_MEASURED` 是**确定**的，所以无论如何都要重跑。

## 3. v7 模型本身：`model_holds = false`（且比 v4/v5 更差）

`results/processed/costcalib_eval_v7.json` 的 `verdict`：

```
"model_holds": false,
"reasons": ["低估相对 p95 16.02% > 5%",
            "整体精度 |相对误差| p90 15.17% > 10%（事后新增的判据：预注册的两条会被一个总是高估的模型满足）"],
"thresholds": {"underest_p95_max": 0.05, "cross_session_drift_max": 0.1, ...}
```

| 模型 | 判据（秩保证划分下的低估 p95） | 出处 |
| --- | --- | --- |
| v4 | 6.12% | `decouple_v3_v4.json` |
| v5 | 9.02% | `costcalib_eval_v5.json` |
| **v7** | **16.02%** | 本目录 `costcalib_eval_v7.json` |

⇒ **加"小 chunk 组"后精度回退**（6.12% → 9.02% → 16.02%，单调变差）。
⚠️ 三个模型的口径必须逐条核（v4 是 `--split rank_guaranteed --hash-modulus 2`，
v5/v7 用的是 `costcalib_eval` 的 contexts 口径）⇒ **上表只作方向参考，不得当"同一判据下的三点"**。
⚠️ 注意 v7 的"查表"低估 p95 只有 **0.11%**（`① 按形状留出`）/ **0.23%**（`② 跨会话留出`）
—— 但那伴随**覆盖率**问题，与 §4.2 的"补覆盖 ≠ 提精度"是同一个权衡，不能只看精度列。

## 4. D0/D1/D2/D3（**只作现场记录，不作结论**——见 §2.1）

| 臂 | D0 可行率（可行 / 候选） | D1（分母 `d1_n`） | D2 阻塞步占比 | D2 `Σ队列·步` | D3 达标吞吐 / 达标率 |
| --- | --- | --- | --- | --- | --- |
| `cost_min512` | **5.41%**（24 / 444，111 次决策） | 0.00%（`d1_n` = **2**） | 26.71% | 624 | 0.0 / 0.000 |
| `cost_pilot` | **4.76%**（48 / 1008，126 次决策） | 0.00%（`d1_n` = **2**） | 23.35% | 624 | 0.0 / 0.000 |

* 拒因（两臂同型）：`预测 752.4 + host 14.2 = 766.5ms 超每步预算 120.0ms`（主导）
  + `模型范围外（不可外推）`。
* ⚠️ **D1 = 0.00% 必须连报分母 `d1_n` = 2** —— 这与"从 84.62%（`d1_n`=13）降到 0.00%"是
  同一类陷阱（分母塌缩），**不得**读成"接纳质量提高"。
* ⚠️ pilot（min128 + 显式 4 元候选）的 D0 **比 min512 更低**（4.76% vs 5.41%），
  两次都**没超过** 5.7% 的参照 ⇒ **"只降下限"没有提高可行率**，这是本轮的实质负面信息。

## 5. 取回清单（供核对）

* 来源：`/root/ascend_probe_d1`，打包含 36 个成员 / 30 个文件，tarball md5 `5b2e7df555e536ef77b5e007c5d19472`（本机 ≡ 远端）。
* 本目录**不进聚合**：`collect_metrics.discover_runs()` 只认顶层含 `meta.json` 的目录，
  而本目录的 run 在 `results/raw/` 之下再深两层 ⇒ 不会被当成 run。
* 远端 `/tmp/s14g_20260927.tgz` 留着（下次若需复核可直接再取）。
