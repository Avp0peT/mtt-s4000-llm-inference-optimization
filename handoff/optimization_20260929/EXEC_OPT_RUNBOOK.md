# 执行优化线真机运行清单（S01–S03 A/B）

> 生成/核对：2026-09-29。**远端当前不可达**（`SSHException: Error reading protocol banner`）
> ⇒ 本清单是"环境一恢复就能一条命令跑起来"的待命状态，不是已跑出的结果。
>
> 前置权威：`01_PROJECT_PLAN.md` §4/§5/§6/§7、`REFERENCE_8_PLANS.md` §2、
> 项目红线见 `PROJECT_MAP.md` §7。

---

## 0. 三条命令（本机执行）

```bash
# 1) 只想看清楚会发生什么（离线，不触网）
python scripts/exec_opt_batch.py plan --tag EXECOPT1

# 2) 只做前置核验（可达性/环境/模型快照/空闲/陈旧锁/CPU 稳定性）
python scripts/exec_opt_batch.py preflight --tag EXECOPT1

# 3) 全流程：核验 → 推送 → 起跑 → 轮询 → 回拉
python scripts/exec_opt_batch.py run --tag EXECOPT1 --stage screen --budget-minutes 240
```

辅助（随时可用，都不影响正在跑的批次）：

```bash
python scripts/exec_opt_batch.py status --tag EXECOPT1    # 心跳 + 已完成项
python scripts/exec_opt_batch.py stop   --tag EXECOPT1    # 当前 run 跑完后干净退出
python scripts/exec_opt_batch.py pull   --tag EXECOPT1    # 增量回拉（可反复跑）
```

`run` 会在**前置任一项不通过时立刻停在原地**并打印原因；
**起跑只发一次，绝不重试**（重试会把同一条 `nohup` 发两遍、抹掉正在写的日志）。

---

## 1. 它替你做了什么（所以你不必盯着）

| 环节 | 交给谁 | 依据 |
| --- | --- | --- |
| 可达性三态判别 | `diag_remote_reachability.py` | 空 banner / EOF / timeout 是**三种**不同状态 |
| 环境与设备 | 远端激活 venv 后打印 torch / torch_musa / transformers / `torch.musa.is_available()` | 不满足即停 |
| **模型 revision 快照在位** | 检查 `$HF_HOME/hub/.../snapshots/<钉死的 commit>` | 配置已把 `main` 钉成 commit；快照不在 ⇒ 第一格才炸 |
| 远端是否在跑别的批次 | `check_remote_busy.py --expect-idle` | `pgrep -f` 在此调用方式下**结构性地不可用**（自匹配） |
| **陈旧锁清理** | `clear_stale_batch_lock.py`（在"确认空闲"**之后**） | 监视进程若自己死了，锁会永远堵住后续批次且不报错 |
| CPU 稳定性 | 远端 `check_cpu_stability.py 7 1.0` | 计时被污染时只有实测能看出来；rc≠0 即拒跑 |
| 推送与 provenance | `push_files.sh --include py,sh,yaml` + `remote_run.py stamp` | 逐文件 sha256 校验；配置也是**实验条件**，必须一起推 |
| 起跑/锁/监视/激活 | `launch_batch.sh --activate …` | 激活必须在 setsid **之后**的 shell 里（`bash -c`，不是 `sh -c`） |
| 穿插顺序/均衡重复/预算 | 远端 `run_exec_opt_matrix.py` | 见 §3 |
| 进度与心跳 | 远端进度文件（30 s 心跳，原子写） | 单次 run 可能 20+ min，只看 run 边界无法区分「在跑」与「卡住」 |
| 回拉 | `pull_incremental.py --tag …` | 只取本地没有的 + 逐目录完整性核对；可反复跑 |

---

## 2. 远端需要什么（环境一恢复先确认这四条）

1. **实例开机且 SSH 可达**；`~/.workbuddy/remote.env` 里的 host/port 与当前实例一致
   （重启换镜像会改端口 ⇒ 症状是"EOF"，容易被读成"关机"）。
2. `source /root/musa-envs/qwen310/activate_musa.sh` 可用；
   `torch.musa.is_available()` 为真。
3. HF 缓存里有 **`a09a35458c702b33eeacc393d103063234e8bc28`** 的 snapshot
   （`$HF_HOME=/root/autodl-tmp/hf`）。不在 ⇒ 先下载；**不要把配置改回 `main`**
   （那会让"声明与事实不一致"，而这个不一致**不会报错**）。
4. **剩余机时额度**（交接包 `STATE.json` 里是 `null` = 未知，**不按 0 也不按无限**）。

---

## 3. 批次怎么编（远端 `run_exec_opt_matrix.py`）

* **穿插**：`for pass in range(repeats): for cell in cells`。背靠背重复会把一次偶发停顿
  整段归因成"这个配置更差"（同配置重复两次实测：tpot 差 1.23×，而 stall_p99 差 **4.35×**）。
* **均衡重复**：预算不够跑完**当前这一轮**时干净收尾 —— 宁可少一整轮，
  也不出现"有的臂 5 次、有的臂 4 次"（那会让比较退化成"谁的 n 大"）。
  收尾时会显式检查 `balanced`，不均衡即非零退出。
* **阶段**：`--stage screen`（默认，4 臂：baseline + s01_both + s02_no_empty_cache +
  s03_fuse_gate_up）→ 有信号再 `--stage full`（9 臂）。
  ⚠️ 筛选阶段先测 **gate/up** 而不是 QKV：QKV 的窄片必须 `contiguous()`，
  每层每次 3 次拷贝（见 `benchmark/linear_fusion.py`），机制上更可能被拷贝抵掉。
* **跑满**：默认 `--fill-budget`，整轮整轮跑到预算不够为止；`--min-repeats 3`
  是不低于 3 次的下限（<3 次不能作结论）。
* **成功判据是产物**：`read_run_status`（失败/拒绝/未完成任一 > 0 即不通过，
  `dropped_requests` 也算）。"跑完了但活没干完"**不算成功**，产物留在证据目录不删。
* **断点续跑**：已存在且判定通过的项**跳过**；已存在但**不通过**的项会报错并**不覆盖**
  （要重跑请换新的 `--tag`）。

---

## 4. 结果怎么判

* 只认 `results/raw/<tag>-short_chat-<arm>[-rN]/` 下的产物；`meta.json` 里
  `execution_optimizations` 记录每个开关的 **requested vs effective 与调用计数**。
  **计数为 0 = 配了但没走到**，与"没配"必须分开读。
* `--stage screen` 的目的是**筛掉**明显无收益的方向（含"无收益"这一有效结论），
  不是产生论文数字。正式对照要 `full` + 冻结规则。
* ⚠️ 本批全部落在**新实验条件**下（新增 6 个执行优化开关 + 钉死 revision）
  ⇒ **不得与任何既有批次并表**。

---

## 5. 失败判据与对策

| 现象 | 含义 | 对策 |
| --- | --- | --- |
| `preflight` 停在可达性 | 主机不在 / 端口变了 / 包被丢 | 三种状态分开看；**不代表方案无效** |
| `preflight` 停在模型快照 | revision 不在缓存 | 下载该 snapshot；不要退回 `main` |
| `check_remote_busy` rc=2 | **探测本身失败**（UNKNOWN） | 不得当作空闲；先修连接 |
| 起跑 rc=4/5 | 已有锁（可能是陈旧锁） | 先 `check_remote_busy`；确认空闲后清陈旧锁再起 |
| 起跑 rc=7/9 | 启动器报错 / 无法证明起来了 | **不要重试**；读 `logs/<tag>.log` 与 `.log.monitor` |
| 某个 run 判 not_ok | 跑完了但活没干完 | 产物保留；按 `failures[].reason` 定位；换 tag 重跑该格 |
| 心跳 >10 min 不动 | 可能卡住或进程已死 | `status` → 远端日志 →`stop`（只影响当前 run 之后） |
| 回拉 rc≠0 | **传输**问题 | 不是方法失败；重跑 `pull` 续取（只取本地没有的） |

---

## 6. 已知边界（不掩盖）

1. **本地无法验证"批次结束自动放锁"**：本机 Git for Windows 没有 `setsid`，
   监视进程随父 shell 一起被杀 ⇒ 锁会留下。远端有 `setsid`，行为不同。
   ⇒ 这条只能在远端确认；本机已用 `clear_stale_batch_lock.py` 兜住后果（见 §1）。
2. 本清单**未在远端实跑过**（远端不可达）。所有命令序列在本机做了**离线路演**
   （`exec_opt_batch.py plan`）与**组件级本地实跑**：
   `launch_batch.sh` 全链路（含激活层、锁、监视）实跑通过；
   `clear_stale_batch_lock.py` 对**真实** `batch_lock.py` 产物验证通过。
3. `--dry-run` 的模拟路径**不产生** `MEASURED`，也验不了过载/SLO/判据。
4. 预算数字（`--budget-minutes 240`）是**默认值**，不是测量结果：
   真实节拍要靠第一批的实测；矩阵会自己按观测节拍决定何时收尾。

---

## 7. 不该做的事（红线）

* 不要对同一条起跑命令重试（会抹掉正在写的日志、并可能两个批次同写 `results/raw/`）。
* 不要覆盖任何既有 `results/raw/**` 目录；重跑一律换新 `--tag`。
* 不要为了"让它跑起来"把 `model_revision` 改回 `main` —— 那会让溯源声明与事实不一致。
* 不要把 CPU 稳定性 rc≠0 当成"可以忽略"。
* 不要用 `--stage full` 的臂数去猜正式结论；先把筛选阶段的信号看清楚。
