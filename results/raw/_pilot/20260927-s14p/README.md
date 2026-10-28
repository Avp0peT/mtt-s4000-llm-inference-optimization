# S14P —— **环境指纹修复的验证运行**（1 格 × 1 重复）· 2026-09-27

> ⚠️ 这**不是**一个实验批次，**不进任何结论**。它存在的唯一理由是回答一个问题：
> **"在新根补采指纹之后，产物还会不会被标成 `NOT_MEASURED`？"**
> 保留它是为了让下一个人不必再问"这个只有 1 个 run 的东西是哪来的"。

## 1. 背景

`/root/ascend_probe_d1`（2026-09-26 23:55 部署）**缺 `results/raw/_env/`**（该目录被
`.gitignore` 排除 ⇒ 每次部署到新根都会缺），导致 `S14G` 的 6 个产物**全部**被标成
`NOT_MEASURED`，D1① 的闸门因此没有分辨力（详见 `results/raw/_pilot/20260927-s14g/README.md`）。

## 2. 做了什么

1. 在新根补采指纹：`cd /root/ascend_probe_d1 && PYTHON=python3 bash scripts/inspect_env.sh`
   ⇒ `results/raw/_env/{env.json,env.txt}`；
   `platform=mthreads` / `available=True` / `mthreads-gmi True` / `MTT S4000 ×1` / `torch 2.2.0`。
2. 与旧根 `/root/ascend_probe/results/raw/_env/env.json`（产过 `MEASURED` 批次的那份）
   **逐字段对比**：89 个字段**只差 5 处**，且全是非结构性项
   （`collected_at` / `mthreads_gmi_info` 里的时间戳 / `hostname`（容器已换）/
   `discovery.hostname` / `memory.free_m`）。
   ⚠️ `MTT_DRIVER_VERSION` / `MUSA_VERSION` 在**两边都是 `None`** ⇒ 不是这次才缺，
   也不是回归。
3. 跑 1 格 × 1 重复（`fixed_cost_min512`，tag `S14P`）作验证。

## 3. 结果（判据）

| 项 | 值 | 判 |
| --- | --- | --- |
| `data_label` | **`MEASURED`** | ✅ 修复生效（此前是 `NOT_MEASURED`） |
| `git_commit` / `git_source` / `git_dirty` | `9df0aefa0c8818d9e584bafe628f43cf856b3748` / `stamp` / `False` | ✅ 可追溯 |
| `model_revision` | `a09a35458c702b33eeacc393d103063234e8bc28`（`resolved=True`） | ✅ |
| `_missing_fields` | **`[]`** | ✅ 无缺失 |
| `controller_effect.arm_state` | 在位（`n_degraded=34`、`n_prefill_starved=17`） | ✅ |

**附带用途**：这批真实产物同时被用来**复核新闸门**的字段位置读法 ——
`scripts/d1_gate.py` 对它给出 `validity_problems = []`、`criteria_evaluated = True`。
（闸门第一版把 `execution_environment` / `hardware_platform` / `label_source` 当 `meta.json`
顶层键读，会对**每一个合法 `MEASURED` 产物**判 `INCONCLUSIVE`；已修，见
`tests/test_d1_gate.py::TestFingerprintLocationMatchesRealArtifacts`。）

## 4. 内容

```
results/raw/S14P-short_chat-fixed_cost_min512/   # meta.json / telemetry.jsonl / requests.jsonl /
                                                 # config.snapshot.yaml
```

* 远端来源：`/root/ascend_probe_d1`，`tar -czf` 后 `get`，tarball md5 `cc88c3685982d6318cdd6652152a8bda`。
* 本目录**不进聚合**（`collect_metrics.discover_runs()` 非递归，run 在 `results/raw/` 之下更深一层）。
* ⚠️ **不登记进 `EXPERIMENTS.md` 的结论行**：它是单次验证运行（n=1，
  `results ≥3` 的门槛不成立），且用途是"证明标签对了"，不是测性能。
