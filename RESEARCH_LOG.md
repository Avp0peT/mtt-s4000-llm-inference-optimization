# RESEARCH_LOG

> ## 📖 怎么读本文件（**248 KB，不要整读**）
>
> | 我要 | 怎么做 |
> | --- | --- |
> | 知道**最近**发生了什么 | `tail -200 RESEARCH_LOG.md` |
> | 找某个主题 / 某个日期 | `grep -n "<关键字或日期>" RESEARCH_LOG.md` |
> | 列出全部条目 | `grep -n "^## " RESEARCH_LOG.md` |
> | 项目总入口 / 一切索引 | → **`PROJECT_MAP.md`** |
>
> ⚠️ 本文件**含所有失败与否定结果的成因** —— 负面条目是它的主要价值，**不要跳过**。
> ⚠️ 它是**日志**，**不是**结论来源；结论以 `EXPERIMENTS.md` 的 `MEASURED` 登记行为准。

时间序研究日志，**append-only**。记录决策、环境变更、意外发现与返工原因。

不记录：临时路径、工具报错原文、可随手重跑的命令输出。这些属于 `results/` 与 `EXPERIMENTS.md`。

## 记录格式

```
## YYYY-MM-DD — <一句话标题>
- 背景：
- 决策 / 发现：
- 影响：
- 下一步：
- 关联：<run_id 或 EXPERIMENTS 编号>
```

---

## 2026-09-18 — 建立仓库骨架与实验规范

- 背景：需要把「昇腾推理优化」从零散试跑收敛为可复现的研究流程，先立流程再堆实验。
- 决策 / 发现：
  1. 采用 src-layout（`src/ascend_opt/`），策略层与硬件层物理隔离，保证策略能在无 NPU 环境下离线验证。
  2. 确立「raw 只读 + processed 可重算」的两级结果目录，杜绝原始数据被就地覆盖。
  3. 优化维度切分为六族：baseline / quantization / prefix_cache / chunked_prefill / graph_mode / adaptive，每族一个配置目录，便于做单变量对照。
  4. 指标口径统一为 TTFT / TPOT / E2E / Throughput / Goodput，一律报 P50·P90·P99。
- 影响：所有后续实验必须先在 `EXPERIMENTS.md` 登记再执行；结论准入需要 ≥3 次重复与离散度。
- 下一步：
  1. 跑 `scripts/inspect_env.sh`，把硬件与环境指纹固化下来。
  2. 用 `--dry-run` 打通 harness 全链路，确认指标计算与落盘格式正确。
  3. 选择第一个正式对照：建议先做 baseline vs chunked_prefill（对 TTFT/P99 的影响最直接、最容易观测）。
- 关联：EXPERIMENTS 编号待补

---

## 2026-09-18 — harness 全链路打通，暴露三个会导致错误结论的缺陷

- 背景：骨架建好后先做「不采数据、只验流程」的打通。这一步的目标是让后续每次
  调参都不必怀疑工具链，否则无法区分「策略错了」和「环境不对」。
- 决策 / 发现：
  1. **确定性模拟路径（`--dry-run`）先行**。用一阶设备模型（prefill 算力受限、
     decode 带宽受限）驱动调度 → 分块 → 缓存 → 控制器 → 指标的完整链路。
     它的价值不在数值，而在让「策略接线是否正确」「账目是否自洽」「控制器是否振荡」
     这几件事可以脱离硬件反复验证。数值一律不得进入结论。
  2. **发现并修掉三个真实缺陷**（都表现为「数字看起来正常」，因此必须靠不变量守住）：
     - *序列名额泄漏*：正在 prefill 的请求不被调度器计入名额，超额准入会在它们
       转入 decode 时把在途序列数顶过 `max_num_seqs`，尾部请求被静默截断而永不完成。
       修法：`TokenBudget` 增加 `occupied_seqs`，由调用方显式告知。
     - *长 prompt 饥饿*：单步 token 预算被 decode 占满时，放不进预算的 prompt
       永远起不了步。修法：分块开启时允许「部分起步」（`allow_partial_prefill`）；
       分块关闭时保留「整段一次算完」的语义，并把无法推进的状态**显式判定为死锁**
       并给出诊断——原来的行为是在完全相同的状态下空转到 `max_steps`，
       最后只留下一个 `dropped=N` 而毫无线索。
     - *尾延迟被均值稀释*：TPOT 是每请求均值，一次秒级停顿会被 `gen_len` 除开。
       实测同一 run 内 TPOT P99 = 74 ms 而单请求间隔 P50 = 968 ms，相差一个数量级。
       修法：新增 **TPOT stall** 指标（单请求自身 token 间隔的 P99），
       并把它纳入 A/B 判定与 `summary.csv` / `repeats.csv` 的字段。
  3. 确立了「一请求一账目」的自查机制：`E2E` 必须等于「完成时刻 − 到达时刻」，
     偏差以 `accounting_error_ms` 落到产物里（当前 P99 = 0.000 ms）。
     参考点定为**到达时刻**——闭环负载下请求在获得客户端容量时才产生，
     用排队时刻做参考点会引入恒定偏差。
  4. 负载的到达语义明确为两态：闭环（在途数 ≤ `concurrency`，压测服务系统的标准做法）
     与开放环（按到达日程放行）。两者混用会让 TTFT 的口径在实验之间悄悄变化。
- 影响：
  - 后续任何「某请求永远不完成」「小改动看不出效果」的现象，先查上述三条不变量，
    再怀疑策略本身。
  - 指标口径新增 stall 与 accounting error，论文表格与 methodology 已同步。
- 下一步：
  1. 接真实后端（`benchmark/` 下新增 backend 模块，实现与 `simulate()` 相同的
     `SimResult` 契约），先用 `short_chat` 单跑 3 次核对 `accounting_error_ms`
     是否接近一步耗时，再与 dry-run 的**定性**结论对齐（不要让数值对齐）。
  2. 用实测数据校准 `device_model` 的四个参数；若拟合残差过大，
     说明一阶模型不适用，需在 Threats to Validity 中记录。
- 关联：EXPERIMENTS 编号待补；验证产物为 14 个 dry-run run（已从 raw 清除，
  可由 `scripts/run_baseline.sh --dry-run` 秒级重放）

---

## 2026-09-18 — 采纳权威研究规范（21 节），并按规范回填缺口

- 背景：仓库原先的 `AGENTS.md` 只是一份**协作契约**（怎么提交、怎么登记实验），
  缺少研究层的规定：研究问题、目标技术栈、模型范围、负载维度、可复现性字段、
  数据完整性规则、阶段流程。现以一份完整研究规范作为权威文本纳入，合并后的
  `AGENTS.md` 分两部分：正文（第 1–21 节，研究规范）+ 附录 A（A1–A7，操作契约）。
- 决策 / 发现：
  1. **研究对象收窄为「请求感知的协调机制」**，而不是"我们启用了前缀缓存/量化/
     大 batch"。规范第 17 节点明这三种都不是贡献——它们已经存在。因此本项目的
     贡献目标定为：batching × Prefix Cache × Prefill 分块在**负载变化下**如何配置。
     这一点直接决定实验设计：对照组必须是"静态配置 vs 自适应"，而不是"开/关某优化"。
  2. **数据完整性从"纪律"升级为"类型"**。规范第 10 节是绝对要求（不得编造结果；
     无硬件时标 `NOT_MEASURED` / `HARDWARE_REQUIRED`；合成数据只能进单测且标
     `TEST_DATA`，永不得进论文表）。把它写成散文是无效的——靠人自觉。现在新增
     `src/ascend_opt/provenance.py`，每个 run 落盘时强制盖章，聚合默认只收
     `MEASURED`。**`--dry-run` 产出一律是 `TEST_DATA`，即使机器上真有 NPU 也一样**，
     因为数字是合成的，与机器无关。
  3. **发现一个会系统性高估硬件可用性的缺陷**：`TelemetryCollector.read_hardware_env`
     在指纹缺字段时 `setdefault("available", True)`。在本机（无 NPU，
     `device.npu_smi_available = false`）会得到 `available=True`，
     于是真机路径的 run 会自称"已实测"。改为从设备探测结果推导，并且引入
     三态：`True` / `False` / `None`（**未知**）。未知既不等于可用也不等于没有——
     把"没查过环境"当成"测过了"正是规范第 10 节要拦的情形。
  4. **发现 `Config.describe()` 漏掉 `runtime` 段**。后果不是报错，而是
     `meta.json` 里 `model` / `precision` / `quantization` / `graph_mode`
     静默变成 `NOT_MEASURED`——看起来像"没配"，实际是"没记"。规范第 8 节要求的
     21 个字段里，有 4 个正好只存在于这一段。已补上并加回归测试。
  5. **缺失必须显形**：可复现性元数据取不到值时写 `NOT_MEASURED`
     （硬件本身不存在时写 `HARDWARE_REQUIRED`），不留空、不填 0。
     留空会被读成"这项不重要"，填 0 会被读成"测出来是 0"，两者都是无声的错误。
     例外：`0` 是真实测量值（如 `warmup_count=0` 表示确实不预热）时必须保留 0。
  6. **负载维度矩阵化**：四个扫描轴的取值按规范原文钉进代码常量
     （输入 128/512/2048/8192/16384；输出 128/512；并发 1/8/16/32/64；
     前缀复用 0/25/50/75%），七类原型由轴函数生成而非手写——手写会产生
     "轴改了、原型没改"的隐蔽漂移。扫描时请求数恒定，保证横轴唯一变化。
  7. **模块命名对齐规范第 12 节**，并借此做了一次真实拆分：
     `WorkloadMonitor`（新，负责把观测归一成状态向量 s_t，**不做决策**）、
     `AdaptiveBatchController` 与 `ChunkController`（新，**无状态**的有界策略对象）、
     `AdaptiveController`（改为协调者）。子控制器刻意无状态：若各自维护当前值，
     就会与 `ControllerState` 形成两个真相来源，序列化出的决策轨迹将无法解释。
  8. **指标补 P95 与 median**（规范第 7 节）。`median` 与 `p50` 数值恒等但都输出，
     免得"median"与"P50"两张表混排时读者被绊住。
  9. **仓库未初始化 git**，导致规范第 8 节要求的 `git_commit` 永远只能是缺失值。
     已 `git init` 并建立首个提交。这是本地操作，未添加远端。
- 影响：
  - `paper/` 与 `README` 不得再出现"启用某项优化"式表述，只能表述协调机制的效果。
  - 任何非 `MEASURED` 的 run 进不了 `summary.csv`；`label_source != explicit`
    （旧产物靠 `dry_run` 回推）会在聚合时被点名警告。
  - 后端的不可测量字段一律保持 `None`，控制器不得消费 0 填充的伪状态。
- 已知限制（登记在案，未修）：
  1. `shared_prefix_len` 是**每请求绝对值**且必须 ≤ `prompt_len_min`，
     因此无法表达"短请求 + 长共享前缀"（真实 RAG 流量的形状）。
     修它需要让共享前缀随请求长度自适应，属语义变更，须作为独立实验登记。
  2. 长度分布只有均匀采样、无重尾。真实对话流量右偏，均匀分布会**低估 P99**，
     而 P99 恰是本项目主要判据。
- 下一步：
  1. 本机无 NPU（`npu-smi` 不可用），**PHASE 1 处于阻塞状态**，见 README 第 8 节。
  2. 阻塞解除前可做且有意义的事：按规范第 6 节把基线负载扫描的**配置**先行备好，
     等真机就位即可开跑；以及把一阶设备模型从"定性可用"做到"有实测校准依据"。
  3. 需在真机上重新采集环境指纹——当前指纹来自 Windows 笔记本，与目标技术栈
     （Linux → CANN → torch_npu → vLLM → vLLM-Ascend）完全不同，性能数字无跨环境
     可比性。
- 关联：EXPERIMENTS 编号待补；本次改动 commit `0bee1b5`（含规范合并与缺口回填）

---

## 2026-09-18 — 同步权威规范末节：权威推理环境是远程昇腾主机

- 背景：重新比对 `Desktop/AGENTS.md`（735 行）与已合并进仓库的版本，发现仓库正文
  **整段缺失**规范末节 `Remote Ascend Execution Environment`（约 48 行）。
  比对时先被一个假象误导：整文件 diff 全红，看起来"全文都变了"。实际原因是
  Desktop 文件是 CRLF、仓库是 LF——**全文件差异 ≠ 全文改动**。
  归一化换行符后，真实差异只有 5 处、60 行，其中 4 处是排版。
- 决策 / 发现：
  1. 正文补齐该节，并逐行核对到"除有意保留的排版规范化外，正文与权威文本逐字一致"。
     同时清理了因缺节而残留的孤立 `---`。
  2. 该节把工作方式钉死了：**本机不是推理环境**，硬件相关操作（设备探测、CANN 检查、
     模型加载、推理、基准、NPU 监控、显存测量、profiling）必须落在远程昇腾主机上。
     这解释了为什么本机 `--dry-run` 之外的任何数字都不具备证据资格。
  3. **`inspect_env.sh` 一项都没覆盖规范点名的 12 项环境事实**，已补齐
     （`ASCEND_DEVICE_MODEL` … `MODEL_PRECISION`），逐项记 `source`；
     取不到写 `null` 并注明原因——不猜一个模型路径，规范明文禁止猜。
  4. **发现一个反向的误杀缺陷**：判定"硬件在不在"时问的是**本机**有没有卡。
     一旦按规范改成"推理部署在远程、本机驱动"，真实测数据会被判成
     `HARDWARE_REQUIRED` 而被聚合器整批拒收，且**没有任何报错**。
     已引入执行环境声明 `runtime.execution_environment`（`remote_ascend` /
     `local` / `unknown`），缺省 `unknown` 时判定与改动前逐格一致（有测试钉死）。
     `local` 永不产生 `MEASURED`；`remote_ascend` 也要有远程指纹确认才算实测。
     拼错的声明在加载配置时直接报错，不静默降级成"没声明"。
  5. **`inspect_env.sh` 采集失败后仍然打印"完成。"并继续落盘**——
     这本身是一例静默失败：一份没被生成的指纹会被当成"已采集"，下游要么读到旧文件、
     要么读到缺失，两种情况都会让 run 的环境归属出错而无报错。已改为失败即退出。
     顺带把 `env.json` 改成先写临时文件再原子替换，避免中途异常留下**能解析但内容不全**
     的指纹。
  6. 执行环境进了 `meta.json`、`summary.csv` 与 `repeats.csv`。理由是：本机 run 与
     远程 run 都可能标 `MEASURED`，光看标签分不出来，而两者可比性完全不同；
     一组重复若混了不同环境，聚合写 `mixed` 而不是取第一个值。
- 影响：
  - 权威推理环境 = 远程昇腾主机。本机只跑离线路径（`--dry-run`、测试、聚合导出干跑）。
  - 环境指纹必须**在执行推理的那台机器上**采集；本地采一份远程指纹不成立。
  - PHASE 1 的表述由"本机阻塞"改为"需在远程昇腾主机上执行"。
  - 测试 176 → 188 项。
- 下一步：
  1. 取得远程昇腾主机访问权后，在**该主机上**跑 `scripts/inspect_env.sh`，
     采集权威环境指纹；确认 `discovery.looks_like_ascend_host = true`。
  2. 若采用"推理在远程、本机驱动"的形态，需在配置里声明
     `runtime.execution_environment: remote_ascend`，否则真实测数据会被误杀。
  3. 仍待办：`RealBackend` 实现（含预热、图捕获与计时窗分离）。
- 关联：EXPERIMENTS 编号待补；本次改动 commit `db0cbeb`（规范同步）、`38c14bc`（.gitattributes）

### 追加发现：换行符隐患（.gitattributes）

补完规范后，`git commit` 输出了一串 "LF will be replaced by CRLF" 警告。追下去发现一个
真实隐患：仓库 `core.autocrlf = true` 且**没有 `.gitattributes`**，意味着下一次检出会把
`scripts/*.sh` 写成 CRLF，bash 随即以 `$'\r': command not found` 失败。

本地工作副本是 LF，所以这个坑**在本地完全不会暴露**；但首次 clone 到远程昇腾主机时
必然触发——而环境搭建恰恰是本项目最不能出岔子的一步（规范末节要求所有硬件相关操作
都在那台机器上执行）。

这与本轮开头那个假象同源：`Desktop/AGENTS.md` 就是 CRLF，直接把文本比对搞成"全文件差异"。

已加 `.gitattributes`：`* text=auto eol=lf`，并显式钉死 `*.sh` / `*.py` / `*.yaml` / `*.md`
等类型。**验证方式是全新 clone 后量字节、再实跑**：4 个脚本 CR 均为 0，
在 clone 里直接跑 `scripts/inspect_env.sh` 得 EXIT=0，clone 内 188 项测试全过。

验证过程本身也踩了一次，值得记住：首次验证用了 Git Bash 的 `/tmp` 路径，clone 实际
没有成功，那一批 `CR=0` 是**读空文件**读出来的——验的是空集却显示"全通过"。
量字节类的检查必须同时确认"文件确实存在且非空"，否则空输入会伪装成合格结果。

（另记一条操作教训：不要用 `printf "...\`cmd\`..."` 往文件里写内容——反引号会被 shell
当命令替换执行。本次因此误跑了 `export_results.sh`，并把它的输出混进了本文件。
往文件写正文一律用编辑工具，不用 shell 拼接。）

---

## 2026-09-18 — 评审整改 (1)：修正 TTFT 归因与 SLO 账目

- 背景：对 harness 做了一轮逐点复现的评审，确认两个会直接污染结论的缺陷：
  (a) 开放环模拟把 `arrival_ms` 覆盖成"当前步开始时刻"，`enqueue_ms` 又赋同一个值。
  后果不是报错，而是**突发负载下 TTFT 被系统性低估**：一步可能很久（关闭分块时
  一次算完整个长 prompt），"请求早已到达、只是本步才轮到处理"这段等待被整段抹掉。
  更隐蔽的是账目仍然自洽——`done_ms` 与 `ttft_ms` 以同一个被覆盖的参考点计量，
  `accounting_error` 依旧是 0。只有把 arrival 与 enqueue 拆成两个时刻才能看见。
  (b) `goodput` 的分母只取成功请求，失败与拒绝被排除在外，服务质量被高估。

- 决策：把"一个 TTFT"拆成可加的**三段**，并给每个请求记录五个时刻
  (`scheduled_arrival / arrival / enqueue / first_schedule / first_token / done`)，
  得到恒等式 `TTFT = admission + queue + prefill_exec`。
  三段成因不同（链路受理 / 容量不足 / 算力与分块），对策也完全不同——只报总数，
  看的人无法判断该去加容量还是该去调分块。
  `first_schedule` 只在**真的分到 token 额度**时才写：被调度器列入 plan 不等于被服务，
  分块预算与容量闸门都可能把额度削到 0。
  结局从 `bool success` 扩为四态 `success / failed / rejected / unfinished`；
  废弃含混的 `goodput`，改为三口径：达标率（分母=全部提交）、达标吞吐（req/s、tok/s）、
  以及失败率/拒绝率/未完成数。
  时间戳一律 `float | None`，`None` = 该时刻从未发生。**禁止用 0 当哨兵**：
  0 既是合法时刻 t=0，又会被读成"未设置"，于是"没测"伪装成"极快"。

- 影响：`summary.csv` 列语义变更（`goodput` 列消失，新增三段分位与结局账目列）。
  已有 dry-run 产物按新口径重跑，静态配置的数字随之变化。203 项测试通过，提交 `77fdc2f`。

- 下一步：控制器闭环与恢复（评审整改 2）。

- 关联：提交 `77fdc2f`；指标口径已同步进 README 第 6 节。

---

## 2026-09-18 — 评审整改 (2)：控制器闭环与恢复能力

- 背景：复现出四个同类缺陷，共同点是**不报错、账目自洽、产物看起来健康**：

  1. 运行期上限的来源搞错了。分块策略内部用 frozen 的 `config.chunk_size` 封顶，
     调用方又用 `cur_chunk` 夹一次。控制器把 chunk 调大，只改了决策日志。
  2. token 预算是一根**单向棘轮**：只在 TPOT 超标时收缩，完全没有放大/恢复路径。
  3. `WorkloadMonitor` 是死代码——`simulate()` 从未调用它，
     `cache_hit_ratio` / `available_memory` 从未进入任何决策。
  4. 控制器可以宣告一个**不可行**的运行点（chunk 大于单步 token 预算），
     那永远兑现不了，下游会静默夹一次。

- 决策：逐条堵住，并**把不可见的东西做成可见的**。

  - 运行期上限经 `chunk_policy.ChunkLimits` 显式传入策略，上限只有一个来源；
    调用方不再重复夹取。`limits=None` 时行为与改动前逐字一致。
  - 新增恢复分支：把此前收缩过的维度逐档恢复到**配置初值**（不是参数上界——
    恢复的语义是撤销防御，不是在安全区里继续放大，那只会引入抖动）。
    轨迹里用 `bottleneck="recover"` 与"健康所以不动"区分开。
  - `observe()` 开始消费 `s_t` 的非时延分量：可用显存低于否决线时否决放大、
    低于临界线时**主动收缩**（显存临界不是"慢了"而是"快跑不动了"，压在一切时延判据之前）；
    缓存命中率高时拒绝"为 TTFT 放大分块"（缓存已经省下算力，这个杠杆就是错的）。
    阈值默认 `None` = 该信号不参与决策。未测信号不改变任何决策，但记录在
    `signals` / `_unavailable_signals` 里——"没测"与"测出来没压力"必须分得开。
    配套加了 `CacheStats.hit_rate_or_none`：缓存**关闭**时命中率是 `None`，
    缓存开着但没命中是 `0.0`，两者对策相反，报表口径可以合并、决策口径不能。
  - 加可行性约束 `chunk_size <= max_batched_tokens`，违反时控制器自己修复并计入
    `n_infeasible_repaired`。**这一条是实测发现的**：两个维度步长不同（512 / 1024），
    同时收缩时 chunk 会相对抬起来。不修的话，"建议值 vs 生效值"的差额会变成常驻噪声，
    而那个差额本该是用来发现失效的检查。
  - 落盘：telemetry 同时记录建议值与生效值，meta 新增 `system_states`（s_t 序列）与
    `controller_effect`（建议/生效区间、恢复次数、未兑现步数）。

- 影响：**先证明测试能失败，再相信它**。把旧行为复原后跑新测试：
  (`a`) 策略忽略运行期上限时，决策日志里 chunk 已到 8192，而实际发放的 prefill 额度
  峰值仍是 512 —— 新断言确实失败；
  (`b`) 去掉恢复分支后，压力消退时状态停在 (1536, 3072) 不回初值 —— 新断言确实失败。
  另用 `git worktree` 检出改动前的提交跑同一批配置做对照：**三个静态配置逐字未变**
  （这正是"单变量对照"的前提），只有 `adaptive_v1` 变了，而且**应该**变。

  自适应变体的变化本身是个有用的观察：TTFT P90/P99 改善（9777→9694、11377→11089 ms），
  同时 **stall P50 恶化（341→648 ms）**。这是加大分块应付的代价——prefill 步更长、
  decode 被更久地挡住。而 TPOT 均值只从 50.54 动到 50.62，**完全看不出来**。
  又一次印证了"必须把 stall 与 TPOT 并列报告"。此前控制器的这个动作是空转的，
  所以这组取舍根本没机会显形。

  测试 203 → 247。全部为 `--dry-run`（`TEST_DATA`），不得进入结论。

- 下一步（记为待办，本轮未做）：
  1. 控制器的 TPOT 判据是"最近 32 个步级间隔的 P90"，连续短 decode 步会把它压下去，
     于是判据可能**看不见自己造成的 prefill 干扰**；而报表里的 `stall` 是每请求间隔 P99，
     恰好看得见。两个量不是同一个东西，需要决定控制器该以哪个为目标（会改变实验结论，
     不能顺手改）。
  2. 信号阈值 (`cache_hit_rate_high` / `min_available_memory_mb` /
     `critical_available_memory_mb`) 目前是待标定占位值；离线路径无显存读数，
     这些闸门只在真机上才会真正生效。真机上可用 `replay_decisions` 在 telemetry 上复放验证。

- 关联：提交 `9324f64`；`tests/test_controller_loop.py`（新增）；README 第 4 节"闭环契约"。

## 2026-09-20 — 评审整改 (3)：把数据完整性从纪律变成程序化约束

- 背景：复现出三类缺口，共同点是**都不会报错**：

  1. `run_benchmark.py` 用 `run_dir.mkdir(parents=True, exist_ok=True)` 落盘，
     同一个 run_id 再跑一次就把上一批产物**原地覆盖**。而 A1 第 1 条明文要求
     `results/raw/**` 一旦落盘即只读。产物被覆盖后，两条互相矛盾的结论会共用
     同一个 run_id，事后无法区分哪一行数字是改动前的结果。
  2. `collect_metrics.effective_label` 对缺 `data_label` 的旧产物按 `dry_run` 回推，
     `dry_run=False` 直接回推 `MEASURED`。于是一批"只是没打标签"的数据被静默
     升格成可进结论的实测数据——不抛错、不告警，只在最后产出一张看起来完全
     正常的汇总表。
  3. 导出脚本只校验"至少有一条 MEASURED"，不管这些行的**依据**还在不在：
     原始明细可能已被清理；环境指纹本来就不入库（`.gitignore` 里有
     `results/raw/_env/`，克隆到新机器后它一定不在）；重复次数可能不足 3；
     数字可能从未登记进 `EXPERIMENTS.md`。这四项缺了都不会让导出失败。

- 决策：

  - **写入层拒绝覆盖。** 新增 `RUN_ARTIFACT_NAMES` / `RunDirConflict` /
    `occupying_artifacts`；`write_run_outputs` 与 `run_once` **共用同一判据**：
    目录里已有任一标志性产物（`meta.json` / `telemetry.jsonl` / `requests.jsonl` /
    `config.snapshot.yaml`）即抛错，除非显式 `--force`（CLI 退出码 4）。
    判据刻意选「产物」而不是「目录存在」：`mkdir(parents=True)` 会提前建目录，
    一次中断留下的空目录若也算冲突，这个 run_id 就被永久堵死了。
    CLI 还会在开跑前把**全部重复**的目标目录查一遍——逐个重复地边跑边查会留下
    "半批"产物，而半批重复最容易被下游当成完整的一次重复用掉。
  - **回推只允许降级。** `effective_label` 的旧产物分支收紧：`dry_run=True`
    → `TEST_DATA`（能确知数字是合成的，这是可靠推断）；其余 → `NOT_MEASURED`。
    理由：`dry_run=False` 推不出"实测"——无 NPU 机器上跑真机路径同样是 `False`，
    而旧产物里又没有执行环境声明，连跑在哪台机器上都无从知道。
  - **导出前置校验。** 新增 `benchmark/preflight.py`，四项检查并**逐条指名**是
    哪个 run 出的问题：原始明细（`requests.jsonl` 非空，且 `meta.json` 在）、
    环境指纹、有效重复（≥3）、`EXPERIMENTS.md` 登记。`export_results.sh` 在拷贝前
    调用它，不通过即退出码 4。严重级别设计：**缺原始明细在任何模式下都是阻断**
    （诊断包也要能对上明细）；其余三项对 `MEASURED` 行阻断、对非 `MEASURED` 行
    降为提示——降级不等于消失，缺陷仍然逐条打印。
  - **派生数据卫生。** `collect_metrics.py` 新增 `prune_orphan_derived`：回收
    `results/processed/` 里 raw 已无对应 run 的逐 run CSV。判据是"raw 里有没有
    同名目录"这种结构性事实，而不是"本次有没有被选中"——后者会让
    `--run-id X` 这类过滤运行把其它 run 的明细一起删掉。

- 影响：**先证明测试能失败，再相信它**。把旧实现复原后跑新断言：
  (`a`) 缺标签 + `dry_run=False` 的旧产物在原实现下回推为 `MEASURED` 并进入汇总表；
  (`b`) 不做占用检测时，重复 run_id 静默覆盖；
  (`c`) 没有前置校验时，四项缺陷全部静默通过；
  (`d`) 不回收孤儿派生文件时，旧列结构残留继续留在盘上。
  四次证伪共 11 条断言失败，无一条是"怎么写都通过"。
  其中 3a 有一条断言在复原后**仍然通过**——"空目录不构成冲突"，
  它正是防止这道闸门过宽的负对照。

  端到端实跑一遍导出链路，暴露了一个只有真跑才会出现的问题：绘图步骤
  `float(r["ttft_p99"])` 在分位列**为空**时崩溃，而"没有成功请求所以分位块为空"
  在本项目是**合法状态**（`unfinished_rate > 0` 时就是它）。原实现若用
  `float(x or 0)` 兜底，会画出"P99 = 0 ms"的柱子——比缺图更误导。已改为显式
  `_fnum()` 返回 `None`、跳过并写明 `omitted_configs`；同时修掉"标签取自全部配置、
  柱子只画可画的配置"这个数量错位（那会让每根柱子顶着别人的名字）。
  本机没装 matplotlib，这段分支是用桩模块抽出来单独跑过的——没执行过的分支
  最容易藏 bug。

  清理了 `results/processed/` 下 3 个旧 schema 遗留派生文件（列里是 `success`
  而没有 `outcome`，且 raw 已不存在），并让此后每次聚合自动回收同类残留。

  测试 247 → 271（`tests/test_data_integrity.py` 新增 18 项；`test_collect_metrics`
  由 11 增至 17）。全部为 `--dry-run`（`TEST_DATA`），不得进入结论。

- 下一步（item 4–6，本轮未做）：

  1. **最小真实后端与计时边界**：`RealBackend` 需写清初始化 / 预热 / 测量三段计时
     与必需的环境快照；配置层加版本兼容矩阵与 `model_revision` 锁定。
  2. **重组对照实验**：每种负载配 4 种对照；调参与评测用不同轨迹；
     export 按 `(workload, config)` 分组并拒绝不可比组合出图。
  3. **收紧论文创新点**。

  另：item 2 遗留的两项待决仍未处理——控制器 TPOT 判据（步级间隔 P90）与报表
  `stall`（每请求间隔 P99）不是同一个量；三个信号阈值是待标定占位值，离线不生效。

- 关联：`benchmark/preflight.py`（新增）；`tests/test_data_integrity.py`（新增）；
  README 第 5、7、8 节。本项是 A1 第 1、2 条的程序化实现——把"记得别覆盖、
  记得可溯源"变成"覆盖会被挡住、依据不全会挡住"。


---

## 2026-09-21 — 可靠性收尾四项：冷却死参数、覆盖闸门、校验只看存在、绘图兜底 0

- 背景：item 1–3 之后，重心转向"可信的真实推理基线"。但在接远程昇腾后端之前，
  先把四个**离线反例就能证明**的缺口收尾——它们都不会报错，只会让基线看起来
  比实际更可信。本项未触达硬件，全部为 `--dry-run`（`TEST_DATA`）。

- 决策 / 发现：

  1. **冷却参数是个死参数（静默失效）。** `cooldown_steps`（默认 3）按**推理步**
     比较，而决策节拍是 `control_interval_steps`（默认 8）步一次，于是
     "距上次调整 < 3 步"永远为假——冷却从未生效，扫这个参数的消融实验不会有
     任何效果。修法：单位改为**观测周期**并更名 `cooldown_cycles`
     （一次可判定的 `observe()` 调用 = 一个周期；控制器关闭或无有效样本的调用
     **不消耗周期**，否则"冷却距离"会取决于调用方有多勤快）。
     并把"冷却到底有没有在动"变成可核对的事实：轨迹里落 `cycle_index` /
     `steps_since_last_decision`，汇总里落 `n_cycles` / `n_skipped_cooldown` /
     `decision_interval_steps`。改前 `n_decisions_changed=55`；改后
     `35`、`n_skipped_cooldown=70`——冷却这次真的拦住了决策。
     旧配置字段现已**直接抛错**（未知识别字段），避免单位混用被静默接受。

  2. **`--force` 收窄为"只放行可丢弃的调试产物"。** 原实现允许 `--force`
     无条件覆盖任何已有产物，与"`results/raw` 是证据目录"直接冲突。
     新规则三条件同时满足才放行：目标不在 `results/raw` 之下、本次是 `dry-run`、
     已有产物标签是 `TEST_DATA`；否则拒绝并**说明是哪一条不满足**。
     已有实测结果只能换新 `run_id`。`load_existing_label` 取不到标签时返回
     `None` 并按**不可覆盖**处理——保守方向，因为误覆盖不可逆。

  3. **导出前置校验从"查存在"升级为"查内容"。** 原先一个非法 JSON 的
     `env.json`、一个键不全的 `meta.json`、一个把 2 次重复自报成
     `sufficient_repeats=true` 的 `repeats.csv` 都能通过。现在逐 run 解析并校验：
     `requests.jsonl` 逐行 JSON 且至少一条记录；`meta.json` 必需键齐全、
     `run_id` 与目录名自洽、`data_label` 与汇总行自洽；
     `config.snapshot.yaml` 可解析；环境指纹是合法 JSON 且**显式声明**可用性
     （不替它推导——推导是采集端的职责）；重复成员数**从汇总行重新数**
     而不信自报旗标，并核对两者同步；登记必须落在 `EXPERIMENTS.md` 的
     **表格单元**里（正文提一嘴不算登记）。

  4. **绘图：分组漏了 workload，且缺数据被兜底成 0。** 中位数函数对空集合返回
     `0.0`，于是"该口径没有成功请求"会画成一根 `P99 = 0 ms` 的柱子——
     0 看起来是一个好得不可能的**测量值**。同时分组只按 `config`，
     把同一份配置在两个负载上的数字混进同一个中位数。
     修法：分组键改为 `(workload, config)`；缺失留空并在原位标 `n/a`；
     略去的组与缺失口径写进 `.source.txt`。**并把绘图逻辑从 shell heredoc
     抽成 `benchmark/plot_p99.py`**——内嵌版本无法被单元测试覆盖，正是这两个
     缺陷能活到评审的原因。图内文字一律 ASCII（默认字体无中文字形，
     中文只会渲染成豆腐块）。

  5. **文档口径同步。** `paper/methodology.md` 的 `Goodput` 行是已废弃的单一名
     口径，改为三个独立口径（`slo_satisfaction_rate` / `slo_goodput_req_s` /
     `slo_goodput_tok_s`），并补两条早已在代码里生效、文档却没写的规则：
     缺失写 `None` 不写 0；时延分位只统计成功请求，必须与未完成率并看。
     `EXPERIMENTS.md` 与 `paper/results.md` 的同名列一并拆开。

- 验证（本项目的固定手法，全部真做了一遍）：

  - **先证明测试能失败**：把旧实现换回去跑新测试——覆盖闸门 6/6 失败
    （正向对照通过）、内容校验 15/17 失败、绘图 9/13 失败。三条主断言分别是
    `--force` 覆盖 MEASURED 被拒、非法 `env.json` 被拒、空分位块画成 0。
  - **静态路径逐字未变**：`git worktree` 检出 `1bc0778`，三个静态配置
    （baseline / chunked_4k / prefix_cache）的 `summary` 逐字段 diff **完全一致**；
    只有 adaptive 变，且变得正是预期的那一项（冷却开始生效）。
  - 测试 271 → 319（`test_plot_p99.py` 新增 15；`test_data_integrity` 18 → 42；
    `test_controller_loop` / `test_simulator` 新增节拍与冷却可达性用例）。

- 影响：`configs/**` 的 `cooldown_steps` → `cooldown_cycles`（旧字段现在报错）；
  `meta.controller_effect` 新增 3 个键；导出闸门对**既有产物**更严格
  （缺配置快照等此前能过的情形现在会阻断）；新增 `benchmark/plot_p99.py`。

- 下一步：本轮只做了离线收尾，未执行任何昇腾硬件实验。下一轮进入真实推理基线：
  `RealBackend` 的计时边界（初始化 / 预热 / 测量三段，后端自报测量时长作为吞吐
  分母）、固定基线的配对静态对照、以及统一控制器反馈口径——控制器用的
  "最近 32 个 token 间隔"与报表的每请求均值 TPOT 不是同一个量，
  标定阈值前必须先统一。

- 关联：`src/ascend_opt/controller.py`；`benchmark/run_benchmark.py`；
  `benchmark/preflight.py`；`benchmark/plot_p99.py`（新增）；
  `tests/test_plot_p99.py`（新增）；`tests/test_data_integrity.py`；
  README 第 8 节；`paper/methodology.md`。


## 2026-09-21 — 平台维度独立：执行环境与硬件平台分开，矛盾声明在加载时报错

- 背景：评审指出研究主体要转到摩尔线程 MTT S4000。查代码后发现真正的缺口不是
  "加个配置"，而是**元数据会说谎**：`ExecutionEnvironment` 只有
  `remote_ascend` 一个远程取值，于是 S4000 的数据要么无法声明、要么被迫在
  meta 里自称"跑在昇腾上"。后者更糟——它不报错，只是把后来按平台筛选/复现的人
  引到错误的机器上。

- 决策 / 发现：
  1. **拆成两个维度，且刻意不互相推导。** `HardwarePlatform`（ascend / mthreads /
     unknown）说的是"硬件是什么"（事实，可被指纹校验）；`ExecutionEnvironment`
     新增 `REMOTE_MTHREADS`，说的是"推理在哪跑"（声明，决定标签）。声明平台
     **不**代表推理就在远程——否则"本机有卡的昇腾开发机"会自动变成 MEASURED。
  2. **判定对平台逐格一视同仁。** `resolve_run_label` 本来就不该知道平台名；
     本轮的职责是让 `remote_mthreads` 与 `remote_ascend` 结果**完全相同**，
     并把这条性质做成测试（遍历 `REMOTE_EXECUTION_VALUES` 的全矩阵）。
  3. **`is_remote` 是三态，不是布尔。** `True`/`False`(本地)/`None`(未声明)。
     判定里写 `is False` 而不是 `not is_remote`——后者会把"没声明"扫进来，
     让所有没写 `execution_environment` 的配置从"按指纹判定"掉成
     `HARDWARE_REQUIRED`，真实测数据被静默降级。
  4. **矛盾声明在加载配置时报错并给可照抄的修法。** 两个字段都会进 meta 且都会被
     下游读，放过去就是永久留一份自相矛盾的溯源记录。任一方 `unknown`、
     或执行环境为 `local` 则放行（"没声明"推不出"说错了"；"本机就是整机"是真实
     情形，允许它写，人才不会为了走通流程而谎报平台）。
  5. **`hardware_platform` 不并入第 8 节字段清单。** 它是补充维度；并进去会让
     `missing_fields()` 多一项，所有历史 run 一夜之间变成"元数据不完整"、
     preflight 全红。

- 顺手修掉的缺陷：`overwrite_refusal` 在"没加 `--force`"时一律说
  "results/raw 是证据目录"。目标在 `C:/tmp` 时这句话是错的，会把读的人引去查
  "我的临时目录怎么成了证据目录"。改为只陈述真实理由（没加 `--force`）并给出
  完整的三条放行条件。`_label_reason` 也一样：原先写死"昇腾"与"npu-smi"，
  会把 S4000 的 `HARDWARE_REQUIRED` 指向昇腾主机——而这句话只在必须重跑时出现。

- 验证（固定手法，全部真做了一遍）：
  - **先证明测试能失败**：注入五个真实退化写法（`not is_remote`、昇腾特判成唯一
    远程、自洽校验变空操作、`load_config` 摘掉校验、`is_remote` 退化成布尔），
    五条全部被测试抓住；正向对照全过。再把旧的"昇腾 + npu-smi"措辞换回去，
    措辞测试也变红。
  - **静态路径数字逐字未变**：`git worktree` 检出 `9568413`，**六个**配置
    （baseline / chunked_4k / prefix_cache / adaptive / acl_graph / w8a8）
    两个树各跑一次 dry-run，`summary` 32 个字段逐字段 diff 完全一致，
    `data_label` 一致；meta 的唯一差异是预期的
    `reproducibility.hardware_platform` 新增。
  - 测试 319 → 347。

- 影响：`runtime.hardware_platform` 成为可写字段；`remote_mthreads` 可声明；
  矛盾声明会在加载配置时抛错（退出前不跑基准）；`meta.reproducibility` 与
  聚合产物新增 `hardware_platform`。

- 下一步（重要）：**规范本身还没改。** `AGENTS.md` 第 7 行与末节
  `Remote Ascend Execution Environment` 仍把权威推理环境写成"远程昇腾主机"。
  代码已平台无关，但"以哪个平台为研究主体"是规范级改动，需显式修订该节。
  在此之前仓库的规范口径仍是昇腾，README 已就地标注"待决"。
  其后：MTT 设备探测（`mthreads-gmi`，目前 `_derive_availability` 只认
  `npu_smi_available`）、聚合/绘图按平台隔离（目前 `hardware_platform`
  只记录、**无消费者**，混合平台的数据仍会被合到一张表里）、
  `configs/mthreads/s4000_fp16.yaml`、`RealBackend` HTTP 流式适配。

- 关联：`src/ascend_opt/provenance.py`；`src/ascend_opt/__init__.py`；
  `benchmark/run_benchmark.py`；`configs/baseline/default.yaml`（注释）；
  `tests/test_provenance.py`；`tests/test_simulator.py`；README 第 7/8 节。


## 2026-09-21 — 平台证据链：声明必须能被指纹证伪（补记）

- 背景：`ExecutionEnvironment` 拆出 `remote_mthreads`、`HardwarePlatform` 独立成维度之后，
  留下三个"声明了但没人核对"的缺口。本轮把它们接上证据链。
  **补记**：这轮改动于 09-21 深夜（21:41 提交之后）完成，当时既未提交也未登记，
  于 09-22 整理入账——正是下一条"未提交改动会选错对照基线"的由来。

- 决策 / 发现：
  1. **声明与指纹冲突要报错，而不是只记录。** 新增
     `provenance.validate_platform_evidence` / `platform_evidence_conflict`：
     配置声明 `mthreads` 而环境指纹检出昇腾（或反之）时抛错。原实现只校验
     "两个**声明字段之间**"自洽，于是"声明 MTT、指纹昇腾"这种组合被静默接受——
     一份自称 MTT 的数据带着昇腾的指纹进入结论。任一方无从判断
     （`unknown` / 指纹缺失）时**放行**：没有证据推不出"说错了"。
  2. **MTT 侧解析器沿用"尽力而为、绝不编造"的写法。** 新增 `parse_mthreads_gmi`：
     型号取出现次数最多者；**设备行数量不出时不回落成 1**——回落会把
     "只看到一个产品名"当成"有一张卡"。与之配套的 `normalize_availability`
     把多个探测结果压成一个三态可用性，取"最保守的可判定值"。
  3. **可用性来源分四态**（`telemetry`）：`declared`（配置声明）/ `derived`（由指纹
     推导）/ `missing`（没有指纹文件）/ `unreadable`（文件坏了）。过去前三者都会掉进
     同一个 `False`，于是一份"没采过指纹"的 run 看起来像"采过、结论是没卡"。
     找不到指纹写 `None` 而不是 `False`；候选路径按优先级依次查找。
  4. **`hardware_platform` 补上消费者**（对应上一条日志里"只记录、无消费者"的缺口）：
     `collect_metrics.group_key` 与 `plot_p99` 的分组键纳入
     `(base, workload, config, hardware_platform, backend)`。方向刻意选"宁可拆开、
     不要合并"——拆错会落到 `insufficient_repeats` 而被看见，并错只会得到一根
     谁都没测过的假柱子（100 与 200 合成 150 那例）。
  5. **`preflight` 也核对指纹里的平台与可用性声明**，不再只查"文件在不在、是不是
     合法 JSON"。
  6. **`inspect_env.sh` 加 `mthreads-gmi` 探测**，与 `npu-smi` 对称，但**刻意不复用
     `npu_*` 键名**——复用会让读端把 MTT 的结论当成昇腾的结论。

- 验证：
  - 测试 347 → **399 全绿**。
  - 冒烟（`configs/mthreads/s4000_fp16.yaml`，见下一条）：声明 `remote_mthreads`
    而本机无设备时**仍判 `TEST_DATA`**（dry-run 压过声明），且 `_missing_fields`
    如实列出 `device_model` / `device_count` / `warmup_count`——缺失显形，未填值。

- 影响：`meta` 与聚合产物新增平台证据字段；跨平台/跨后端的数据不再可能被并成一组。

- 关联：`src/ascend_opt/provenance.py`；`src/ascend_opt/telemetry.py`；
  `benchmark/collect_metrics.py`；`benchmark/plot_p99.py`；`benchmark/preflight.py`；
  `scripts/inspect_env.sh`；`tests/test_provenance.py`；`tests/test_collect_metrics.py`；
  `tests/test_plot_p99.py`；`tests/test_data_integrity.py`。


## 2026-09-22 — 规范修订：研究主体平台改为 MTT S4000

- 背景：评审提出研究主体转为摩尔线程 MTT S4000。此前代码侧已做成平台无关，但规范末节
  仍把权威推理环境写成"远程昇腾主机"，README 就地标注"待决"。本轮**显式修订规范**，
  把这项待决落定。

- 决策：
  1. **末节改名 `Remote Ascend Execution Environment` → `Remote Execution Environment`。**
     理由不是措辞好看：写死平台名的节标题正是"两平台并存时把读者指到错误机器上"的那个
     失效模式，而它已在 `_label_reason` 上真实复现过一次。新标题中性，正文点名
     "研究主体平台是 MTT S4000，次平台昇腾主机不是权威环境"。
  2. **研究主体与次平台显式分层。** 第 1 节 purpose 与第 3 节目标技术栈改为：主体
     `Linux (Ubuntu 22.04) → MTT driver (mtml / sgpu-dkms) → MUSA SDK → torch_musa →
     PyTorch → vLLM → vLLM-MUSA`；昇腾栈**保留为次平台**。并写明两条栈
     **不可互换、不可比**——呼应第 10 节的平台维度，避免有人把两平台数字并表。
  3. **新增术语条款（第 1 节）：研究主体换了，但标识符不改名。** `ascend_opt` 包名、
     `ASCEND_*` 键名、`Ascend` 枚举值一律保留；正文里"NPU / HBM / 显存"按
     "目标加速器 / 其设备显存"读，除非该句明确在讲昇腾。改名会让所有导入路径、
     历史产物引用与老读端静默失效，换来的只是字面整齐。
  4. **末节把"12 项发现"的契约从"键名"改回"项目清单"。** 具体键名是实现细节
     （`inspect_env.sh` 里目前仍是 `ASCEND_*` 拼写），**契约是那 12 项本身**。
     同时写明：某个项测不到必须报缺失，**不得**用另一个平台的值顶上——带着错误平台
     工具名的指纹比没有指纹更糟，因为它读起来像证据。
  5. **新增 `configs/mthreads/s4000_fp16.yaml`**：主体平台基线，显式声明
     `execution_environment: remote_mthreads` + `hardware_platform: mthreads`。
     注释写明它**不是** `baseline/default.yaml` 的单变量变体，而是另一参照系的平台基线
     （故平台与精度同时不同）；以它起步的优化实验仍须守单变量。

- 验证：
  - 全量 **399 项测试仍全绿**（新配置被 `test_simulator` 的 `rglob` 自动纳入加载与
    无丢弃校验，不需要手工登记）。
  - `--dry-run` 冒烟：产物 `data_label=TEST_DATA`、`execution_environment=remote_mthreads`、
    `hardware_platform=mthreads`、`precision=fp16`——同时证明两件事：**声明不会把标签
    升格**，且新配置的溯源字段真的落盘了（不是"只写了注释"）。
  - 旧节名引用逐处更新（代码注释、脚本注释、配置注释、测试 docstring、README）；
    仅 `RESEARCH_LOG.md` 的历史条目保留旧名——日志是历史记录，不改写。

- 影响：规范口径（正文 + README）与代码口径对齐；`label_reason`、CLI 说明串等面向人的
  字符串不再写死昇腾。**本轮不改变任何判定结果**：平台判定本就不认平台名。

- 已知缺口（下一步）：
  1. `inspect_env.sh` 的 `DISCOVERY_KEYS` 仍是 `ASCEND_*` 拼写，且只采集昇腾侧的
     CANN / torch_npu / vLLM-Ascend 版本；MTT 侧的 **MUSA SDK / torch_musa /
     vLLM-MUSA 版本尚未采集**。本机（无卡）表现为恒 `null`，真机上会是真的缺口
     ——那 12 项是"从执行推理的机器上发现"，主体平台缺一半采集项并不满足契约。
  2. `_PLATFORM_DISPLAY_NAME["mthreads"]` 仍是厂商级的"摩尔线程 MTT"，未细到机型
     ——刻意不写死机型，以免换卡后把读者指到错误的机器。
  3. `precision: fp16` 是按 vLLM-MUSA 常规部署写的**声明**，首轮真机运行前必须在远程
     主机上核对；不一致时改配置，而不是让 meta 与事实相反。
  4. 仓库目录名仍为 `ascend-llm-inference-optimization`；改名牵动所有本地路径与历史
     产物引用，是否改由用户决定。

- 关联：`AGENTS.md`（第 1 节 purpose / 第 3 节技术栈 / 末节 / 附录 A3）；
  `README.md`（标题 / §2 / §6 / §7 / §9）；`configs/mthreads/s4000_fp16.yaml`（新增）；
  `configs/baseline/default.yaml`；`benchmark/run_benchmark.py`；
  `benchmark/collect_metrics.py`；`benchmark/preflight.py`；`src/ascend_opt/__init__.py`；
  `scripts/inspect_env.sh`；`tests/test_provenance.py`。


## 2026-09-22 — 模块接线四项修复：平台证据、设备字段、可用性判据、分组唯一化

- 背景：上一轮（规范修订）通过的是**单模块**测试。这四项缺陷的共同点是
  「函数存在、字段存在、判据存在，但**没有接到主流程上**」——单模块测试全绿，
  产物也看起来完全正常。故本轮一律从**产物那一端**反向验证。

- 发现（四项，均已复现）：
  1. `provenance.validate_platform_evidence` 有实现、有单测，但**主流程从未调用**：
     元数据构造只查"两个声明是否自洽"，导出检查只读第一份指纹的可用性。
     于是"声明 MTT、指纹明确是 Ascend"的组合被一路放行到落盘与导出。
  2. 采集端规范化后的 `device.model` / `device.count` **读端不读**：读端只认
     `device_model` / `npu.model` 等历史写法，于是"采集成功了、meta 里却是空的"
     ——两端各记一半。
  3. `scripts/inspect_env.sh` 的 `_probe_result` 与解析器是**两套判据**：
     同一条 `mthreads-gmi 2.7.0 supports MTT S4000`，解析器返回"卡数未知"，
     采集判定却给出 `available=True`。
  4. 分组规则**三处各写一份**且已分叉：聚合按 (批次, 负载, 配置, 平台, 后端)，
     绘图漏了后端，导出校验又只认 `run_id` 前缀（把被拆开的两组 3 + 3 合成 6，
     报出并不存在的 `repeats_count_mismatch`）。

- 决策：
  1. 判定收敛到 `provenance.device_probe_result`，`inspect_env.sh` 改为调用它
     （**一份实现**）。取值表：工具不存在→False；数出 ≥ 1 张→True；
     数出 0 张→False；**数不出（含只有产品名）→None**。
  2. `device.model` / `device.count` 是**唯一权威**键名，读端优先读它，再回落历史
     写法。卡数的回落链刻意不写成 `a or b or c`：`0` 是**测量值**，假值会被静默
     跳过并最终变成 `None`——"测到没有卡"与"没查到卡数"的修法完全不同。
  3. 同一判据接在**三处**入口，且共用同一份指纹快照：`run_once`（**跑基准之前**）
     → `build_reproducibility_record`（装配元数据）→ `preflight.check_export`
     （逐 run 导出）。运行前读到的指纹**带着往下传**（`run_benchmark` →
     `TelemetryCollector`），使"判定用的"与"写进 meta 的"是**同一个对象**，
     而不是同一路径读两次。任一方 `unknown` 一律放行：没证据 ≠ 证据矛盾。
  4. 新增 `src/ascend_opt/grouping.py` 作为分组的**唯一定义**，汇总/绘图/导出三处
     从它派生。`repeats.csv` 新增 `member_run_ids` 列，导出侧改为读落盘清单，
     并把两种滞后分开报：清单里列了汇总表中不存在的 run（`repeats_member_unknown`）
     与有 run 没被任何一组认领（`repeats_member_unaccounted`）。
     分组维度含 `backend`：绘图过去漏掉它，会把"模拟 100"与"真机 200"并成一根
     150 的柱。图注与图标题里的分组口径改为**由维度定义生成**——写死会在规则
     变化后继续向复核的人宣称旧规则。

- 验证（三件事，缺一不可）：
  1. **全量 423 项测试全绿**（399 → 400（拆分重复次数检查为两条）+ 新增 23 项
     跨模块契约测试）。
  2. **证明新测试会失败**：把五处实现分别换回旧行为再跑对应测试，全部如期变红
     （Fix2 读端 / Fix3 判据 / Fix1 元数据入口 / Fix1 运行前入口 / Fix4 绘图分组），
     且每个场景还原后文件哈希与改前一致。
  3. **静态路径逐字段未变**：`git worktree` 检出 HEAD 旧代码，同一批 6 个配置各跑
     一次 `--dry-run`，两棵树用**同一个 `--out` 路径**（否则"未找到指纹"提示串里的
     路径会造出与行为无关的 diff——第一次就踩到了）。屏蔽时间戳/commit/主机名等
     易变字段后**非易变差异 = 0**，且 `telemetry.jsonl` / `requests.jsonl` /
     `config.snapshot.yaml` 逐字节一致。结论：四项修复**不改变任何既有判定结果**。

- 影响：声明与指纹的矛盾在**浪费一次基准之前**就被拦下；MTT 型号/卡数进入 meta；
  无设备记录的指纹不再被判为可用；表、图与导出对"同一实验"的理解一致。
  既有 18 个 `cmp-*`（TEST_DATA）重跑聚合后只多出 `member_run_ids` 一列，
  预检在真实数据上给出 7 项提示、无误报。

- 下一步（第二轮，用户已排定）：
  1. `configs/mthreads/s4000_fp16.yaml` 的 `cache.block_size` 仍是 16；对
     driver 2.7.0 的旧版 MTT 后端应写明引擎块大小 **64**、FP16、转换后权重路径
     与转换参数，并说清"块 16（仅模拟）"与引擎参数的关系。
  2. `RealBackend`（`run_benchmark.py`）仍未实现：当前测试通过证明的是**离线框架
     行为**，尚不能证明 S4000 部署可用。验收：单卡跑通、真实请求与失败可追溯、
     预热不计入测量、基线重复 3 次有效。
  3. 采集侧仍有缺口：`DISCOVERY_KEYS` 的 12 项仍是 `ASCEND_*` 拼写（契约是那 12 项
     本身，键名是细节）；MTT 侧已补 `MTT_DRIVER_VERSION` / `MUSA_VERSION` /
     `TORCH_MUSA_VERSION` / `VLLM_MUSA_VERSION`，但 **vLLM-MUSA 的 python 包名
     无法从本项目证实**，故只走环境变量或模块导入，取不到写 `None` 而不猜。
  4. 本机（Windows，无 `npu-smi` / `mthreads-gmi`）无法端到端验证采集脚本对
     "只有产品名"的判定：Windows 上 `subprocess` 不解析 `PATHEXT`，假工具跑不起来。
     已在**函数接缝**（`provenance.device_probe_result`）上验证，并把旧判据作为
     可执行断言留在测试里（同一条输入，旧实现说"可用"、新实现说"未知"）。

- 关联：`src/ascend_opt/grouping.py`（新增）；`tests/test_cross_module_contracts.py`（新增）；
  `src/ascend_opt/provenance.py`；`src/ascend_opt/telemetry.py`；
  `benchmark/run_benchmark.py`；`benchmark/collect_metrics.py`；
  `benchmark/plot_p99.py`；`benchmark/preflight.py`；`scripts/inspect_env.sh`；
  `README.md`（绘图纪律 / 派生数据卫生两节）；`tests/test_data_integrity.py`；
  `tests/test_plot_p99.py`。

---

## 2026-09-22 — 远程 MTT S4000 指纹采集完成，并修掉两个"静默给错"的采集缺陷

- 背景：PHASE 0 的最后一个阻塞项是"远程 MTT S4000 主机指纹待采"。拿到该主机的 SSH 凭证
  后，把 `scripts/inspect_env.sh` 与 `src/ascend_opt/` 打包送上远程执行，指纹取回本地。
  采集过程本身暴露了两个缺陷：**产物看起来完全正常，但两个版本号是错的**。

- 决策 / 发现：
  1. **远程主机事实**（指纹采集于 2026-09-22）：`platform = mthreads`、`available = true`；
     `device.model = MTT S4000`、`device.count = 1`、显存 49152 MiB；driver **2.7.0**、
     MUSA SDK **3.1.0**（`/usr/local/musa` → `musa-3.1.0`）、`mthreads-gmi 1.14.0`；
     `torch 2.2.0` + `torch_musa 1.3.0+81caf0a`；`npu-smi` 不存在——与"研究主体是 MTT、
     昇腾为次平台"一致。**该机没有 `vllm`、没有 `transformers`、也没有模型权重**，
     故 PHASE 1 的阻塞点从"连不上主机"变成"主机上缺软件"。
  2. **缺陷一：驱动版本被写成了 gmi 工具版本。** 真实输出把两者印在**同一行**：
     `mthreads-gmi:1.14.0          Driver Version:2.7.0`。旧正则写成
     `(?:mthreads-gmi|driver\s+version)` 的**交替**，而交替总是先匹配到位置靠前的
     `mthreads-gmi`，于是 `MTT_DRIVER_VERSION` 被静默写成 `1.14.0`。现改为只认显式的
     `Driver Version` 标签；只有工具版本行时写 `None`——错版本比没版本危险，且工具版本
     的原文已逐字存进 `device.mthreads_gmi_info`，信息不会丢。
  3. **缺陷二：MUSA 版本恒为 `None`。** 候选文件只列了 `/usr/local/musa/version` 与
     `version.txt`，而该机装的是 **`version.json`**。文件找不到 ⇒ `MUSA_VERSION` 为
     `None`，在产物里与"这台机器没装 MUSA"**完全无法区分**。现补上候选，并把解析收进
     `provenance.parse_musa_version_file`（纯函数、可单测）：`version.json` 里
     `mublas`/`muPP`/`mufft`… 各有自己的 `version`，**只有 `musa_toolkits`（缺失时退到
     `musa_runtime`）才是 SDK 版本**。
  4. **两个缺陷都逃过了单测，原因比缺陷本身更值得记**：当时的测试样本是**手写的想象
     输出**（把两个版本分成了两行），不是真实产物。故本轮新用例**逐字抄自真机输出**，
     并要求 MUSA 样本把别的组件排在 `musa_toolkits` 之前——否则"取文件里第一个数字"
     那种实现也能通过，用例就分不出对错。
  5. **既有闸门对新指纹的端到端验证**（用 `--out` 指向临时目录，不写进 `results/raw`）：
     指纹与配置声明一致 → 通过；把指纹平台改成 `ascend`（与 `s4000_fp16.yaml` 声明的
     `mthreads` 矛盾）→ **在跑基准之前**被拒，并给出两选一的可照抄改法；完全不放指纹
     → 通过（`unknown` 不是矛盾）。未声明平台的 `default.yaml` 配真实 mthreads 指纹
     也不被误杀。

- 影响：
  - `MTT_DRIVER_VERSION` 由 `1.14.0` 修正为 **`2.7.0`**；`MUSA_VERSION` 由 `None` 修正为
    **`3.1.0`**（来源 `file:/usr/local/musa/version.json`）；`mtt_missing` 由 2 项降到 1 项
    （只剩 `VLLM_MUSA_VERSION`，因未装 vLLM）。**同一台机器、同一份脚本，修复前后各采一次**
    即构成 A/B 对照。
  - 驱动版本 2.7.0 获得真机确认——它正是"旧版 MTT 后端引擎块大小 64"那条待办的前提，
    不再是假设。
  - 指纹落入 meta：`hardware_env.device.model/count`、`MTT_DRIVER_VERSION`、`MUSA_VERSION`
    均可在 `meta.json` 中查到；`data_label` 仍是 `TEST_DATA`——真实指纹**不会**把一次
    `--dry-run` 的数字升级成 `MEASURED`。
  - 指纹文件按 `.gitignore` 的既定口径**不入库**（`results/raw/_env/` 被明确排除），
    故主机事实的持久记录只在**本文件与 `README.md` §9**里。

- 下一步：
  1. PHASE 1：在远程主机装 vLLM-MUSA（先确认包名，不要猜）并放置 Qwen2.5-7B-Instruct
     权重，随后补齐 `MODEL_PATH/NAME/REVISION/PRECISION` 与 `VLLM_MUSA_VERSION` 声明。
  2. 核对 `s4000_fp16.yaml` 的 `precision: fp16` 与该机实际部署精度是否一致；落实
     `cache.block_size` 16→64（driver 2.7.0 的引擎块大小）。
  3. `RealBackend` 仍待实现：HTTP 流式拿不到 token 级到达时刻 ⇒ stall 测不出来。

- 关联：`results/raw/_env/env.json`（按口径不入库）；`scripts/inspect_env.sh`；
  `src/ascend_opt/provenance.py`（`_MTT_DRIVER_VERSION`、`parse_musa_version_file`）；
  `src/ascend_opt/__init__.py`；`tests/test_cross_module_contracts.py`；
  `README.md` §8 第 8 条、§9。

---

## 2026-09-22 — 口径更正：块大小 64 / 权重转换只属于旧版 MTT-vLLM 栈；附 S4000 设备自检

- 背景：上一轮把「`cache.block_size` 16→64 + 写明 FP16 / 转换后权重路径 / 转换参数」列为
  `s4000_fp16.yaml` 的**下一轮验收口径**。用户指出这是**错的**——那是**旧版
  MTT-vLLM / MTTransformer** 栈的要求，**不适用于所有 S4000 推理方式**；既然先走
  Transformers 验证路线，就不该机械照搬。

- 决策 / 发现：
  1. **撤回上述口径。** 更正后的规则写进 `README.md` §2 的引用块与 §9、以及
     `configs/mthreads/s4000_fp16.yaml` 的注释：
     - `cache.block_size` 是**本处前缀缓存的模拟粒度**（`cache_policy` 消费），
       **不是设备/引擎参数**；把它当"MTT 引擎块大小"去调是概念混淆。
     - **权重转换**与**引擎块大小 64** 只属于旧版 MTT-vLLM / MTTransformer 栈。
     - **推理引擎尚未定案**：先 Transformers 单请求验证 → 接 `RealBackend` → 做连续批处理 /
       前缀缓存 / 分块预填充实验 → **最后**按功能支持度选引擎。引擎选定后，它自己的依赖与
       参数要求再**重新核对**，并按单变量纪律单独登记。
     本条**取代**本日志同日「模块接线四项修复」与「远程指纹采集」两条末尾"下一步"中
     关于 block_size / 权重转换的表述（日志 append-only，故以新条更正，不回改旧条）。
  2. **设备自检已通过（真机实测）**：`torch.musa.is_available() = True`、
     `device_count = 1`、`get_device_name(0) = MTT S4000`；在 `musa` 上跑通 matmul
     **与 autograd**（只"导入成功"不算数，必须真的执行算子并读回结果）。
     `torch.musa.mem_get_info()` = free 49015 / total **49062** MiB——注意它与
     `mthreads-gmi` 报的 49152 MiB **不一致**（差 90 MiB）。两个数来自不同层，
     报显存时必须写明口径，不能混用。
  3. **网络可达性**（决定装包与取模型的方式）：`pypi.org` / 清华 / 阿里云 PyPI 均可达；
     **`huggingface.co` 不可达（000），`hf-mirror.com` 可达（200）** ⇒ 取模型必须设
     `HF_ENDPOINT=https://hf-mirror.com`，否则会卡在连接上而不是干净地报错。

- 影响：
  - 不会再有会话照着一个已作废的口径去改 `block_size` 或做权重转换。
  - 远程主机确认具备跑 Transformers 推理的完整设备栈（torch 2.2.0 + torch_musa 1.3.0 +
    MUSA 3.1.0）；PHASE 1 的剩余前置只有**装模型依赖**与**取权重**两件事。

- 下一步：建立独立环境（保留 base 的 torch/torch_musa/MUSA）→ 版本核对后装模型依赖 →
  Qwen 单请求生成验证（先用小模型打通链路，再上 7B 取显存与耗时数据）。

- 关联：`README.md` §2 / §9；`configs/mthreads/s4000_fp16.yaml`。

---

## 2026-09-22 — 独立环境建成、Qwen 单请求验证通过（0.5B + 7B）、推理引擎摸底

### 1. 独立 Python 环境（保留 base 的 torch / torch_musa / MUSA）

按用户排定的第一步：不换 base 环境，用 `venv --system-site-packages` 建
`/root/musa-envs/qwen310`，从而**保留下层 conda 里已装好并可用的**
`torch 2.2.0` + `torch_musa 1.3.0+81caf0a` + MUSA SDK 3.1.0，只在本层加装模型依赖。
版本核对后装入：`transformers 4.46.3`、`tokenizers 0.20.3`、`accelerate 1.1.1`、
`safetensors 0.4.5`；python 3.10.8。取模型走 `HF_ENDPOINT=https://hf-mirror.com`
（`huggingface.co` 在本机不可达，实测 curl 000）。

### 2. 装完后才暴露的失败：`GLIBCXX_3.4.30 not found`

现象：`import tokenizers` 之后 `import torch` 崩，而**单独** `import torch` 正常。
**这是导入顺序相关的**，所以极易被误判成"装包把环境装坏了"。

成因（`/proc/self/maps` 实测，不是推断）：

| 做法 | 进程里实际载入的 libstdc++ | 结果 |
| --- | --- | --- |
| 只 `import torch` | `/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30` | OK |
| 先 `tokenizers` 再 `torch` | `/root/miniconda3/lib/libstdc++.so.6.0.29` | FAIL |
| 两者 + `LD_PRELOAD` 系统版 | `/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30` | OK |

conda 自带的那份是 **6.0.29**，不含 `GLIBCXX_3.4.30`。`tokenizers` 的扩展 `.so`
**没有 RUNPATH**，于是继承 conda python 的 RPATH `$ORIGIN/../lib`，把旧的那份先载进
进程；**同名 soname 一旦载入即被复用**，之后 torch 再 `dlopen` 只能拿到旧的。
torch 自己的 `.so` 有 RUNPATH（`$ORIGIN:/usr/local/musa/lib`），那两个目录里没有
libstdc++，所以它单独跑时落到系统那份（正确）。

修法：`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`，与导入顺序无关。
已固化进 `/root/musa-envs/qwen310/activate_musa.sh`（同时设 `LD_LIBRARY_PATH`、
`HF_ENDPOINT`、`HF_HOME`）。**这不是装包的副作用**，别去回滚包版本。

### 3. Qwen 单请求生成验证（真机实测）

设备：MTT S4000 ×1，`musa:0`，fp16，贪心解码（`do_sample=False`，seed 0），
prompt 39 tokens，重复 3 次取中位数。脚本 `qwen_single_request.py`（v2）。

| 项 | Qwen2.5-0.5B-Instruct | Qwen2.5-7B-Instruct |
| --- | --- | --- |
| 参数量 | 494,032,768 | 7,615,616,512 |
| 参数所在设备 / 精度 | `musa:0` / `torch.float16` | `musa:0` / `torch.float16` |
| 参数全部在设备上 | ✅ | ✅ |
| 载入耗时（tokenizer+权重+搬运） | 7.21 s | 8.48 / 9.46 / 8.62 s |
| 显存增量（载入后） | 1002.8 MiB | 14737.3 MiB |
| 设备侧峰值分配 | 955.9 MiB | 14577.0 MiB |
| TTFT（中位） | 48.6 ms | 90.0–90.2 ms |
| TPOT（中位） | 47.3 ms | 55.3–58.4 ms |
| 单请求自身间隔 P99 | 52.5–53.1 ms | 61.3–82.9 ms |
| 贪心 3 次逐 token 相同 | ✅ | ✅ |

- **设备执行**：`torch.musa.is_available()=True`、`device_count=1`、
  `get_device_name(0)="MTT S4000"`，参数全部落在 `musa` 上。
- **输出正确性**：① 3 次贪心解码**逐 token id 完全相同**（不是"文本看起来一样"）；
  ② 强校验——同样 fp32 权重下 MUSA 与 CPU 的**首个前向 logits** 对照：
  0.5B `max_abs_diff=2.03e-4` / 7B `max_abs_diff=8.77e-4`、`mean_abs_diff` 2.4e-5、
  **argmax 一致**（7B 末位 argmax 均为 17），shape `[1,39,152064]`。只看第一步是为了
  避免生成误差累积把"数值偏差"和"采样路径分叉"混在一起。
- **显存**：7B 实测增量 14737.3 MiB，理论 fp16 权重 = 7.6156e9 × 2 B = 14.19 GiB
  = 14532 MiB，差 ~205 MiB 是 KV cache + 激活 + 对齐 → **数量级与账目自洽**。
- **口径**：上述显存数一律来自 `torch.musa.mem_get_info()` / `max_memory_allocated()`，
  与 `mthreads-gmi` 的 49152 MiB 是**两个不同层**，不得混用（见同日「口径更正」条目）。

### 4. 探针自身的一个静默缺陷（第三次同形：产物正常、内容是错的）

v2 把 transformers 的 import 收进 `_timed_phases()` 的**局部作用域**，而 fp32 对照块在
`main()` 里用 `AutoModelForCausalLM`。于是对照整块以
`NameError: name 'AutoModelForCausalLM' is not defined` 失败——但它被 `try/except`
包着，产物里只多一个 `{"error": ...}` 字段，**进程退出码仍是 0**，链式脚本照样打印
"全部完成"。**一次"输出正确性已验证"差点没有任何依据。**

修法做成两条，而不是只补那句 import：
1. 顶层新增 `failed_blocks` 字段，失败环节显形；有失败则**返回退出码 3**。
   局部 catch 保留（已跑完的 fp16 数据不能因对照崩了就丢），但**不代表可以算成功**。
2. **双向验证**：故意把那句 import 删掉复现原缺陷 → 退出码由 0 变 **3**、
   `failed_blocks=['correctness_fp32_first_forward']`；再跑修好的版本 → 退出码 0、
   `failed_blocks=[]`、产出真数字。仅证明"修好后能跑"是不够的。

顺带：0.5B 的 `max_abs_diff` 在探针 v1 与修好的 v2 下**逐位相同**
（`0.00020313262939453125`），说明测量本身可复现、且两次是同一个量。

### 5. 时延口径发现：**每步固定开销主导**，且两点拟合被自己的实测否掉

0.5B 与 7B 参数量差 **15.4 倍**，但每步耗时只从 47.3 ms 变成 55.3–58.4 ms（+17~23%），
TTFT 从 48.6 变成 90.1 ms（+85%）。一个 7B 在数据中心级卡上 57 ms/token 意味着
**设备在每步里绝大部分时间是闲的**。

为了不把"参数多 15 倍只多花 10 ms"讲成玄学，实测了这块卡的搬运带宽上限
（大张量 `copy_`，读+写各一遍，预热后取 20 次均值）。**数值一律引自入库原始数据**
`results/raw/_probe/20260922/mem_bandwidth.json`：

| 张量 | 单次 copy | 有效带宽 |
| --- | --- | --- |
| 256 MiB | 0.772 ms | **695.4 GB/s** |
| 2048 MiB | 6.6272 ms | **648.1 GB/s** |

（初稿此处 2048 MiB 档写的是 `8.4374 ms` / `509.0 GB/s`。那是**另一次测量**，
与入库值相差 27%，且**没有入库产物可复核**——已按入库值更正，
两者的来源区分见下方「评审更正」的更正 2。256 MiB 档两处一致。）

- 7B 每步至少要读 14.19 GiB 权重。按上表把 695.4 GB/s 当作**带宽上界**算，
  访存**至少**需要 21.9 ms；按 2048 MiB 那档 648.1 GB/s 算至少 23.5 ms。
  即在 57.6 ms 的 TPOT 里，**访存至少占 38%~41%**。

  > **⚠️ 本条目已被更正（2026-09-22，见下方「口径更正」）。**
  > 原稿由此推出"访存**最多**占 38%~52%，剩下大半不是访存"，**上下界方向反了**：
  > 38%~52% 是访存耗时的**下限**（因为分子用的是带宽上界），
  > 对应的比例也是**下限**。"下限 38%" 只能说明"访存不可能比这更少"，
  > **不能**说明"访存不可能更多"，因此**不能**据此认定"剩下大半不是访存"，
  > 更不能据此认定"固定开销主导 / GPU 大部分时间闲置"。
  > 正确的读法是：访存占用 **≥ 38%~52%**，上界未知。
  > 要得到上界，必须实测到访存时间的**下界**——即用带宽**实测值**（而非上界）
  > 估出的耗时，或直接做访存/计算的时间分片测量。

- **反面结论（仍然成立）**：若用 0.5B→7B 两点做"固定开销 + 参数流"拟合，会推出需要
  ~1.4 TB/s 的带宽才解释得通——**高于本机实测上限，所以该两段模型是错的**。
  固定开销的大小不能跨模型外推。要分解每步成本，必须专门扫 batch size / 模型规模
  做回归（`per_step = a + b·batch`），不能用两个点硬拟合。
- 对论文的直接含义：**"每步固定成本主导"这一条目前没有证据支持**（见上面的更正），
  唯一稳的是"参数量差 15.4 倍而每步只差 17~23%"，说明**每步里存在一大块
  与参数量无关的时间**——它是否属于"固定开销"、是否属于"闲置"，都要靠回归分解。
  连续批处理的门槛分析同样要等回归结果，不能用本轮的两点观测直接外推。
  而被均值稀释的 ITL 抖动（P99 61~83 ms）是要攻的指标，这一条不受本更正影响。
- 方法学副作用：三次独立进程的 TTFT 很稳（90.06/90.01/90.21 ms，<0.3%），
  TPOT 跨进程差 5.5%，但**P99 跨进程差 35%**（61.3/82.9/65.5 ms）。
  ⇒ 同一进程内 3 次重复的离散度**不能代表**跨进程离散度；配置之间必须**穿插对照**，
  不能一个配置一个进程地跑完再比。

### 6. 推理引擎摸底（为 PHASE 1 第三步定候选，均据一手来源）

**⚠️ 本表的"能力"列只表示 `参数存在于源码`，不表示 `MUSA 路径可用`。**
判据强度见本节末的"待验证"说明——不要按这张表直接启动完整实验。

| 候选 | 与本机栈是否匹配 | 连续批处理 | 前缀缓存 | 分块预填充 |
| --- | --- | --- | --- | --- |
| MooreThreads/vLLM_musa（GitHub，从源码构建） | ✅ 文档要求 `musa_toolkit>=dev3.0.0` / `pytorch>=2.2.0` / `torch_musa>=1.3.0` / `vllm v0.4.2`，本机 3.1.0 / 2.2.0 / 1.3.0 全部满足 | ✅ 参数存在（核心特性） | ⚠️ **待验证** `--enable-prefix-caching`（v0.4.2 默认 **False**） | ⚠️ **待验证** `--enable-chunked-prefill`（默认 False，依赖 `--max-num-batched-tokens`） |
| PyPI `vllm-musa`（第三方插件） | ❌ 依赖 `torch==2.7.1`，装它等于换掉 torch → 与"保留现有 torch/torch_musa/MUSA"冲突 | — | — | — |
| Transformers（已跑通） | ✅ | ❌ | ❌ | ❌ |

- **为什么后两项是"待验证"而不是"✅"**（2026-09-22 评审更正，详见文末「评审更正」更正 3）：
  - **参数存在 ≠ MUSA 路径可用。** 这两个开关是上游 vLLM v0.4.2 的参数；
    MooreThreads 分支的 MUSA 适配是否覆盖了这两条路径的算子，**没有验证过**。
  - **已发现的具体风险**：该分支的 attention 实现里存在把整数 `sliding_window`
    当作**可下标对象**使用的问题。这类缺陷只在特定配置下触发，
    而"参数能传进去、启动不报错"完全不能排除它。
  - **处理**：必须先做**最小可用性验证**（三项各跑一次小规模，确认数值与行为都对），
    再据此回升本表。在那之前这两行**不计入选型依据**。
  - **连续批处理**标"参数存在"而非"✅ 已验证"：它是核心特性、与本次缺陷所在的
    attention 优化路径耦合较弱，但证据强度同样只是"源码里有"，验证时一并跑。
- 上游 vLLM **v0.4.2** 源码里 `--block-size` 的允许值只有 `[8, 16, 32]`——**不含 64**；
  且 MTT 的 MUSA 适配文档通篇未提块大小或权重转换。
  ⇒ **旁证了用户那条更正**：64 是那套旧栈自己的引擎常量，不是 S4000 的通用要求。
  （本条只作旁证，不改变已作废口径，见同日「口径更正」条目。）
- 引擎选定前的边界：Transformers 路线用于**设备执行/正确性/显存**验证与逐 token 时间戳，
  不承担连续批处理/前缀缓存/分块预填充；这三项要等引擎定了再做，并按单变量纪律单独登记。

### 7. 环境坑（本轮新增）

- **`MSYS_NO_PATHCONV=1`**：Git Bash 会把传给原生 `python.exe` 的"看起来像 POSIX 路径"的
  参数改写成 Windows 路径。现象是 `sftp get /root/x.json` 报 `FileNotFoundError`，
  而**同一路径**在远端 `ls`/`md5sum` 都正常。修法：加 `MSYS_NO_PATHCONV=1` 或用 `//root/...`。
  判据是"远端说有、客户端说没有"——**不要据此怀疑远端或改远端文件**。
  注意同一 helper 的 `put` 当时没复现，**不能用"上次没事"推断这次没事**。
- **`torch.musa` 不是 torch 自带的**：必须先 `import torch_musa` 才注册出来，否则报
  `AttributeError: module 'torch' has no attribute 'musa'`（报错指向 torch，不指向缺失的导入）。
  别依赖"别的库顺手导入过"——那是靠求值顺序成立的，顺序一变就崩。
- 远程一键链式任务的判据不能用会**自我匹配**的模式：`pgrep -f run_after_download.sh`
  会命中它自己所在的命令行，于是永远判为"仍在运行"。改用查日志里的终态标记或查产物文件存在。

### 8. 影响 / 下一步

- PHASE 1 的前置只剩"接 `RealBackend`"一件事；**尚无任何 `MEASURED` 数据**——
  本轮全部数字来自设备侧探针，不是 harness 产物，故**不得**写进 EXPERIMENTS 结论。
- 下一步（用户排定的第三步）：把 `RealBackend` 接成 Transformers 后端，落在 `benchmark/` 下，
  与 `simulate()` 同一份 `SimResult` 契约。**它顺带解决了 `stall` 测不出来的阻塞**：
  在进程内用 logits processor 逐 token 打时间戳即可拿到 token 级到达时刻
  （HTTP 流式那条路拿不到）。随后才是引擎落地与三组功能实验（连续批处理 / 前缀缓存 /
  分块预填充），最后按功能支持度选引擎。

- 关联：**`results/raw/_probe/20260922/`**（本轮全部设备侧探针产物与脚本，已入库；
  该目录不是 harness 产物、不得进 EXPERIMENTS 结论，说明见其 README）；`README.md` §9。

---

## 2026-09-22 — 评审更正（同上日）：访存上/下界方向反了，且带宽两个数字的来源要说清

评审发现上一条目（同日第 5 节）有两处必须更正。两处都属于**结论方向**问题，
不是笔误——错了会让后续实验的门槛判断建立在反过来的不等式上，因此单独成节。

### 更正 1：从"访存**至少**需要多少时间"推不出"访存**最多**占多少比例"

原文用的是**带宽上界** 696 GB/s 算出"访存至少 21.9 ms"，得到的 38%~52% 因此是
**访存耗时占比的下界**（分子取的是最乐观的带宽 → 耗时最小 → 占比最小）。
把它读成"访存最多占 38%~52%"等于把下界当上界，方向正好相反。

| | 原文写法 | 正确的读法 |
| --- | --- | --- |
| 已知量 | 带宽**上界** 695.4 GB/s → 访存耗时**至少** 21.9 ms | 同 |
| 可得结论 | 访存占比 **≥** 38%~41% | 同 |
| **不可得结论** | 访存占比 **≤** 38%~41%；"剩下大半不是访存" | ✗ 反了 |
| **不可得结论** | "固定开销主导"、"GPU 大部分时间闲置" | ✗ 无证据 |

- **更正后的表述**：访存占用 **≥ 38%~41%**，**上界未知**。
- 要得到上界，必须拿到访存耗时的**下界**——即用带宽**实测值**（而非上界）估耗时，
  或直接做访存/计算的时间分片测量。本轮两者都没有，因此**不能**给出上界。
- 仍然成立的部分：参数量差 15.4 倍而每步只差 17~23%，说明每步里存在**一大块与
  参数量无关的时间**。它是否叫"固定开销"、是否等于"闲置"，都要靠
  `per_step = a + b·batch` 回归分解，不能用本轮两点观测直接命名。
- 仍然成立的反面结论：两点拟合会要求 ~1.4 TB/s，**高于本机实测上限** ⇒ 该两段模型是错的。

### 更正 2：2048 MiB copy 的 509 GB/s 与 648.1 GB/s 不是矛盾，是两次测量

入库 JSON `results/raw/_probe/20260922/mem_bandwidth.json` 里 2048 MiB 档是
**`per_copy_ms=6.6272` / `effective_gbps=648.1`**；而日志第 5 节的表格写的是
**`8.4374 ms` / `509.0 GB/s`**。两者都是自洽的（算术已逐项复核）：

| 来源 | per_copy_ms | 有效带宽 | 说明 |
| --- | --- | --- | --- |
| 入库 JSON（`mem_bandwidth.json`） | 6.6272 | 648.1 GB/s | **入库的那次测量**，权威 |
| 日志第 5 节表格 | 8.4374 | 509.0 GB/s | 另一次测量（或被 256 MiB 档数字替换过），**无入库产物** |

- 复核依据：`effective_gbps = 2 × 2048 MiB ÷ per_copy_ms`。
  6.6272 ms → 648.1 GB/s；8.4374 ms → 509.0 GB/s。两个数各自自洽，
  所以**不是**算错，而是**两份不同的原始记录**。
- **处理**：以入库 JSON 为准（它是唯一有原始数据可复核的），日志表格**已按入库值更正**；
  同时把 509 GB/s 标注为"无入库依据的另一次观测"。
  256 MiB 档两边一致（0.772 ms / 695.4 vs 日志初稿的 0.7713 / 696.0，属四舍五入差异），不受影响。
- **纪律含义**：两次测量相差 27% 而**没有任何字段能区分它们**——这正是
  "原始数据必须入库、正文只引用入库路径"的理由。以后带宽类数字一律引用
  `results/raw/_probe/<date>/mem_bandwidth.json`，不在正文另存第二份。
- 对结论的影响：取哪一档作"带宽上界"都不影响下界方向——上界取大值只会让
  "访存至少需要的时间"更小、占比下界更低。因此**更正后的结论（≥ 38%）不变**，
  变更的是"不能读成 ≤"。

### 更正 3：旧 `vLLM_musa` 的前缀缓存 / 分块预填充标为**待验证**

第 6 节的引擎能力表里，这两项从"✅ 参数存在"改为 **⚠️ 待验证**：

- **参数存在不等于 MUSA 路径可用。** `--enable-prefix-caching` 与
  `--enable-chunked-prefill` 是上游 vLLM v0.4.2 的参数；MooreThreads 分支的
  MUSA 适配是否覆盖了这两条路径的算子，**没有验证过**。
- **已发现的具体风险**：该分支的 attention 实现里存在把整数 `sliding_window`
  当作可下标对象使用的问题（`sliding_window` 在某些配置下是 `int`，
  代码却按可下标/有属性来用）。这类缺陷会让相关路径在**特定配置下**才炸，
  而"参数能传进去、启动不报错"完全不能排除它。
- **处理**：不能按能力表直接启动完整实验。要先把引擎装起来、跑一个**最小可用性
  验证**（连续批处理、前缀缓存、分块预填充各跑一次小规模，确认数值与行为都对），
  再据此更新本表。在那之前这两行写"待验证"，不计入选型依据。
- 连续批处理仍标 ✅：它是该分支的核心特性，且与本次缺陷所在的 attention
  优化路径耦合较弱——但同样属于"参数存在"级别的证据，验证时一并跑。

- 关联：`README.md` §9；`results/raw/_probe/20260922/mem_bandwidth.json`。

## 2026-09-23 — PHASE 1 第 ② 步通过：FP16 路径四场景数值等价 + 资源生命周期（0.5B / 7B）

- 背景：`VALIDATION_PLAN_PHASE1.md` 已审核通过（阈值预先声明），第 ② 步要在远程
  S4000 上验证 `RealBackend` 与原生 Transformers 的数值等价。此前一轮把四个实现缺陷
  （K/V 未分开存储、warmup 不产首 token、完成请求 KV 不释放、量化/图模式配置被接受但
  不执行）修完后，首次执行。
- 决策 / 发现：
  1. **两个模型全部通过**，产物 `passed=true`、`failures=[]`：
     `results/raw/phase1_validation/validate_0.5b.json`（md5 `de870ad0…`）、
     `results/raw/phase1_validation/validate_7b.json`（md5 `2437a135…`）。
  2. **原判据在 fp16 下不成立，已修订。** 原 §4 要求 batch/chunk 场景 token **逐位相同**；
     7B 的 V3 行 `v3_0` 在第 4 步分岔，但该步两侧 `max|Δ| = 0.0000` 且
     `top1=16 / top2=15 / gap=0.0000`——**恰为平局**，fp16 舍入即可翻转 `argmax`。
     实测动态范围：同输入重复 `0.0`；改一个输入 token `9.05`(0.5B)/`21.71`(7B)；
     切分/batch 扰动 `0.023~0.031`。据此改为 **logits 等价**，`FP16_LOGIT_TOL = 0.25`
     （约 8 倍观测扰动，远低于"改一个输入 token"的量级，两侧都有余量）。
     0.5B 无此现象（`gap=3.27`），佐证是平局特有而非普遍。
     **口径要点**：放宽的是**数值等价**，不是语义等价；首 token 与非平局 token 仍须一致，
     且平局证据必须显形（写入 `notes` 而非 `failures`）。
  3. **"捕获最后一次前向的 logits"在按形状分组时取错行。** 驱动把变长 batch 按
     `n_new_tokens` **从大到小**分组依次前向，故"最后一次"是**最小**那组；据此做逐行比对
     会拿 48 长的参考去比 16 长的前向，造出 `max|Δ|=12.6` 的**假失败**（自己引入的）。
     同类陷阱：说明分支用了循环残留变量（`for r in v3_reqs` 结束后 `r` 停在 `v3_3`），
     把 `v3_0` 的平局报成"需复核"。两处均已修，并写入方案 §5 首查表。
  4. 其余实测：V1 fp32 `max_abs_diff=0.0`（**该判据不能证伪"驱动被旁路"**——参考与驱动
     共用同一权重实例，相等是预期值；产物已写 `limitation`，旁路由 V2/V3/V4 的 token
     逐位比对证伪）；V1 标定确认判据对真实差异敏感；V2 token 逐位相同；V3 padding 前后
     各行 logits 差 `0.0`；V4 `whole_vs_chunked` = `0.0234`/`0.0313`，三种替代切分全等价，
     `reference_vs_whole = 0.0`；全部 driver 的 `live_slots_at_end = 0`。
- 影响：第 ② 步可作为 `RealBackend` 的数值正确性依据。**但注意：本步产物不是 harness
  产物**（直接调驱动、自带计时/判据），且性能数字明确不作基线（方案 §1.4/§1.5）——
  本步**不产生任何 `MEASURED` 结论**，只关闭"实现是否有语义错误"这一问题。
- 下一步：进第 ③ 步性能矩阵（并发 1/2/4/8，固定 3 次，与基线穿插，履行 §1.5 四项强制
  披露义务）。执行前先确认设备空闲（本轮实测 `mthreads-gmi` 0%、无进程；但**主机 CPU
  负载 ~13 属共享租户**，须在 §1.5 披露中体现）。
- 关联：`VALIDATION_PLAN_PHASE1.md` §7；`results/raw/phase1_validation/`；
  同日 824 行条目（Qwen 单请求验证）。

---

## 2026-09-23 — PHASE 1 第 ③ 步性能矩阵执行完毕（36/36，首批 `MEASURED` 数据）

**一句话**：矩阵跑通、证据链完整、得到两条可下的结论（C1/C2）与三条"测不出结论"（N1~N3）；
但**本轮的 `adaptive` 臂没有测到它声称要测的东西**——这是配置参数选择造成的设计缺陷，
不是实现缺陷，必须修后重跑。

- **范围**：主体平台 MTT S4000（`remote_mthreads`），Qwen2.5-7B-Instruct / fp16 /
  TP=1；负载 `phase1_anchor`（prompt 2048 / gen 128 / 8 请求，**并发是被扫描的轴**）；
  3 臂（不分块 / 分块 `chunk_size=512` / 自适应）× 并发 1/2/4/8 × 3 轮，
  **穿插执行**（`for pass: for cell:`），共 **36 次**。
- **产物**：`results/raw/P1S3-phase1_anchor-<arm>_c<N>[-rN]/`，36 个目录 / 156 个文件，
  全部 `MEASURED`，已入库。聚合走 `benchmark/collect_metrics.py` → `results/processed/`
  （按 `.gitignore` 不入库）。`benchmark/preflight.py` 本地跑通（0 项提示，RC=0），
  登记见 `EXPERIMENTS.md` §1.1。
- **执行事实**：36 次全部成功（`n_success=8/8`、`n_failed/n_rejected/n_unfinished` 全 0）；
  每次 `warmup_count=3`、实测预热 `warmup_s=48.9`（目标 30 s）；
  同格 3 次重复**相隔 19.3 分钟、跨度 38.6 分钟**——这是穿插执行真的生效的直接证据
  （若背靠背，这个间隔会是秒级）。
- **判据与噪声底（§1.5 四项义务的履行）**：噪声底必须**逐指标 × 逐分位**给，不是全负载一个数。
  12 格上 `spread_pct=(max−min)/中位` 的分布：

  | 指标·分位 | 中位 | 最大 |
  |---|---|---|
  | TTFT p50 | 0.069% | 0.415%（adaptive_c8） |
  | TTFT p90/p95/p99 | 0.187% / 0.176% / 0.166% | **4.81% / 10.40% / 14.87%**（static_c1） |
  | TPOT p50 | 0.134% | 0.720%（static_chunk_c2） |
  | stall p50 | 0.349% | 3.555%（static_c8） |
  | stall p90/p95/p99 | 0.651% / 0.636% / 0.627% | 4.60% / 4.60% / 5.10% |

  三条要点：① 噪声底随分位升高而变大，且**尾部事件只影响分位、不影响中位数**
  （`static_c1` 某轮 TTFT p99 跳到 515 ms，而**同一轮 p50 只到 448.47**）
  ⇒ 只看 p50 会完全看不见；② **stall p99 最不可重复**（中位 0.63%，约 TTFT p50 的 9 倍），
  方向与 §1.5 一致，但 §1.5 的 4.35× 来自**另一条测量链路**
  （`qwen_0.5b_v2_correctness.json`），**两者不得互相引用**；
  ③ **稳健 CV 会掩盖 n=3 的单点离群**（`static_c8` stall p99 = 479.49/458.41/458.49，
  稳健 CV 0.0002 而普通 CV 0.0213，相差 88 倍）⇒ 引用离散度必须同时给原始值。
- **可下的结论（`MEASURED`）**：
  - **C1** 分块预填充与不分块的 TTFT P50 差异**随并发换向**：c1 **+40%**（448.50→630.00）、
    c2 −5%、c4 **−42%**（1655.40→960.72）、c8 **+72%**（2278.44→3929.58）。
    差异 40%~72%，是 TTFT p50 噪声底**最大值**的 100 倍以上 ⇒ 不可能是测量噪声。
  - **C2** 分块预填充的实质是**用 TTFT 换 stall**，且放大的是 **TTFT 的尾部**：
    关分块时 TTFT 分位几乎重合（`static_c4` = 1655.40/1655.52/1655.52，prefill 一次做完
    ⇒ 各请求 TTFT 近乎常数），开分块后尾部拉开（`static_chunk_c4` = 960.72/3054.29/3344.61）。
    ⚠️ **收益侧在此更正（2026-09-23）**：原文写"并发 ≥2 时 stall P99 明显改善
    （c8：458.49→293.57，−36%）"，**与原始数据不符**——那只挑了唯一改善的一格。
    按每格 3 次重复取中位逐格重算：c1 62.54→62.17（−0.6%，低于噪声底 ⇒ 无变化）、
    **c2 78.51→222.21（+183.0%，恶化）**、**c4 88.47→271.31（+206.7%，恶化）**、
    **c8 458.49→293.57（−36.0%，改善）**。⇒ **只有 c8 改善**，且伴随 **−18.2% 吞吐**
    （50.69→41.46 tok/s）与 `wall_ms` +22.3%。正确表述是"收益**随并发变化**，
    取决于 prefill 与 decode 是否重叠"，**不得**概括为普遍改善。
    `wall_ms`：c8 20.20 s→24.70 s；c2/c4 则**两头都亏**（42.65→46.88、25.63→31.26）。
    **c=1 是例外**：无 decode 竞争，分块只带来额外调度轮次 ⇒ TTFT +40.5%、
    吞吐略降、stall 无改善（差异低于噪声底）。
- **测不出结论（与"负面结论"分开登记）**：
  - **N1** `adaptive` vs `static_chunk`：TTFT P50 差 0.02%~0.33%、TPOT P50 差 ≤0.8 ms，
    **全部低于噪声底**，也落在同格三次重复的区间内（c8 的 0.33% < 该格自身 spread 0.415%）。
    **这既不是"自适应无收益"也不是"自适应有害"**——原因见下条。
  - **N2** 失败/拒绝/未完成行为：36 格全 0，是**产生不出**这些结局，不是"测过且没有"。
  - **N3** 前缀缓存收益：`cache.enabled=false`，缓存**从未开启过**，不得据此说其无收益。
- ⚠️ **本轮最重要的负面发现（设计缺陷，我的）**：`adaptive` 臂**没有测到它声称要测的东西**，
  两个**配置**原因叠加（不是实现错误，两个字段都已定位）：
  1. `chunk_size_init = chunk_size_min = 512` ⇒ 被扫描维度**一步也动不了**。
     控制器在 `tpot` 瓶颈下走 `_shrink(chunk=True, tokens=True)`，但 `chunk_req(min/max)`
     全程恒为 `512/512`（12/12 个自适应 run 都是），只有 `max_batched_tokens` 从 8192
     收到 2048（撞下界，`n_clamped` 每步 +1）；而该参数在这些负载下**不是约束**
     （prefill 上限由 `chunk_size` 决定）。
  2. `target_tpot_ms = 50` 对 7B **不可达**（实测 TPOT 下界约 61 ms）⇒ 每次决策都读到
     "超目标"⇒ **只收不放**，`n_recovered = 0` 出现在全部 12 个自适应 run 上——
     正是规范点名的**棘轮**。
  ⇒ 后果：`adaptive` 与 `static_chunk` 在数值上就是同一件事，两者读数逐格重合。
  **结论边界**：本矩阵**没有**测到"自适应切片 vs 静态切片"的对照。
  修法：`chunk_size_init` 必须明显大于 `chunk_size_min`（例 1536/512/512 ⇒ 有可动空间）；
  `target_tpot_ms` 必须可达，使控制器既能收缩也能恢复。**修好之前，不得用本矩阵的
  adaptive 数据回答任何关于自适应收益的问题。**
- **另一项发现（登记，未改）**：本矩阵 36 格 `slo_satisfaction_rate` **恒为 0.000**，
  因为 `target_tpot_ms=50` 对 7B 不可达（§6.2 的既有发现）。⇒ **达标率这一列在本矩阵中
  不携带信息**（三值恒等 ⇒ 稳健 CV 恒为 0，那是"无信息"而非"稳定"）。
  重标定 `slo.tpot_ms` 应在本矩阵结束后作为**单变量**单独登记。
- **两项追溯缺口（均登记为发现，本轮不修）**：
  1. **36/36 份产物的 `reproducibility.git_commit` 都是 `"NOT_MEASURED"`** ——
     这批 `MEASURED` 证据**无法钉到代码版本**。不影响 C1/C2（40% 量级不可能由版本漂移造出），
     但会影响后续接近噪声底的比较（如 N1 重跑）。另：写字符串 `"NOT_MEASURED"`
     不如写 `None`——前者看起来像一个正常记录值。
  2. **报表层把"缓存未启用"渲染成 `cache_hit_rate = 0.0`**：定位到
     `src/ascend_opt/metrics.py:389`（用 `cached_tokens/prompt_tokens`，不知道缓存是否启用）
     与 `benchmark/collect_metrics.py:244`（`summary.get(..., 0.0)`，缺键静默填 0）。
     **决策层是对的**（`src/ascend_opt/monitor.py:176` 用 `CacheStats.hit_rate_or_none`，
     `tests/test_policies.py:205` 已钉住"没查 ≠ 命中 0"）⇒ 问题只在报表层。
     危险在于：读的人会得到"前缀缓存命中率 0%"这个**看起来像测量结果**的数字。
- **本轮修掉的工具缺陷（三处，都会"报成功却没生效"或"检查过了但数据不对"）**：
  1. **`push` 用 basename 作 arcname** ⇒ 推单个文件 `scripts/x.py` 会落到远端**根下**
     `x.py`，而命令执行的是 `scripts/x.py`（旧文件）。**推送打印成功、跑的却是旧代码**。
     已改为：相对路径保留整条相对路径（`as_posix()`，避免 Windows 反斜杠被当成字面字符），
     并用 sftp `stat` 逐项确认落点存在。
  2. **`pull` 解包目标选错层级** ⇒ 数据落在 `<落点>/<弧名>/...`（多套一层），
     而校验按 **tar 内部路径**去查，于是"每个文件都在、大小都对"全部通过——
     **检查过了，数据却不在预期位置**。真踩到：拉完 36 个 run 后检查器报
     "没有找到带 meta.json 的 run"。已改：解到落点**父目录**，校验改为核对**预期落点布局**，
     并把"tar 顶层不是弧名"直接判为失败；判定抽成 `_extract_archive` 以便测试
     （`tests/test_remote_run_pull.py`，含反向对照）。
  3. **`pull` 失败路径不清理远端临时 tar** ⇒ CRLF 污染的那批路径在远端留下
     **37 个体积为零、名字含 `\r` 的 `.tar`**，还被通配符统计进去，使"远端文件数"
     比"本地文件数"多出正好一个 run 数，**看起来像拉取不完整**。已改：清理放进 `finally`，
     并用"不存在的远端路径"做了正向对照（rc=2 且残留 0）。
- **检查器判据缺陷（已修 + 补测试）**：原版"可疑离散度"判据**只查 TTFT、且只看稳健 CV**，
  于是漏掉本矩阵最该看的那条（`static_c8 · stall p99` 被 MAD 抹平）。
  已改为：三个指标都过、同时报稳健 CV 与普通 CV、并以"整体形状"（逐指标 × 逐分位的分布）
  代替逐格罗列（逐格罗列在整体都极小时会退化成"全部可疑"，等于没有判据）。
  测试见 `tests/test_step3_runs_inspector.py`（13 项，含"抹掉离群点则判据必须安静"的反向对照）。
- **方法学副作用（新踩，值得记住）**：把远端命令输出重定向到本地文件时会**引入 CRLF**
  （37 行 37 个 CR），批量循环里 `read -r d` 拿到的路径带尾随 `\r` ⇒ 37 次全失败；
  而当时把 stderr 丢进了 `/dev/null`，于是**看不见原因**。教训：批量失败先留首个失败的原因，
  再谈重试。
- 影响：PHASE 1 的"性能矩阵"这一子项**已执行但有结论边界**——它产出了 C1/C2（真实可用）
  与 N1~N3（其中 N1 必须重跑才可重新表述），**没有**产出项目核心问题
  （"负载变化时自适应是否优于静态"）的答案。
- 下一步：① 按 §6.3 修 `chunk_size_init` / `target_tpot_ms` 后**重跑矩阵**（仍 3 次、仍穿插）；
  ② 单变量重标定 `slo.tpot_ms`；③ 补上 `git_commit` 采集；④ 修报表层 `cache_hit_rate` 口径；
  ⑤ 之后再进连续批处理 / 前缀缓存 / 分块预填充实验与引擎选型。
- 关联：`PERFORMANCE_MATRIX_PHASE1.md`（设计与完整读数，§6 缺陷、§7 结果）；
  `EXPERIMENTS.md` §1.1 登记行、§3 结论、§4 负面结果；
  `results/raw/P1S3-*`（36 个 run）；`results/raw/_pilot/`（试点 3 次，对聚合不可见）；
  同日 1078 行条目（第 ② 步通过）。

## 2026-09-23 — 第 ③ 步修正版矩阵（P1S3B）：设计、两道前置门、启动

承上一条的"下一步 ①"。本条的产出是**修正方案本身**（配置 + 两道可阻断的预检门 +
设计记录），真机运行已启动但**尚未出结果**。设计全文见 `PERFORMANCE_MATRIX_PHASE1B.md`。

### 修正方案**本身**漏了一处（本轮最重要的发现）

上一条记的修正方案是"改 `chunk_size_init` + `target_tpot_ms`"。真去配的时候发现
**不够**：基线的 `target_ttft_ms=2000`（上限 2200）高于四档实测 TTFT P90 的**全部**取值
（632 / 1257 / 2742 / 6959 —— 只有 c4/c8 超过，而它们会先被 `tpot` 分支短路），
于是 `ttft`/grow 分支**一次都不会触发**，控制器**仍然**只会收缩。
即：只照原方案改，`adaptive` 臂会从"维度结构上不可动"变成"结构上可动、但负载形状让它永远只往一个方向动"——
**同样是棘轮，只是成因换了一层**。已把 `target_ttft_ms` 一并压到可达区间内。

### 取值由**已登记的 P1S3 实录信号**反解，不是拍的

反解输入是控制器**实际读到**的 `controller_trace[*].signals.ttft_p90_ms / tpot_p90_ms`
（`_decide` 用 P90；与报表里的逐请求分位不是同一个量）：TPOT P90 中位 61 / 81 / 98 / 141 ms，
TTFT P90 中位 632 / 1257 / 2742 / 6959 ms（并发 1/2/4/8）。

| 目标 | 区间来源 | 取值 |
|------|---------|------|
| `target_tpot_ms` | 低并发必须**不**触发 `tpot`（否则 grow 判据轮不到被评估）：`61.5 < 1.10·t` 且 `1−61.5/t > 0.10` ⇒ `t > 68.3`；高并发必须**触发**（那里正确动作是收缩）：`140 > 1.10·t` ⇒ `t < 127.3` | **110** |
| `target_ttft_ms` | 必须能被初始状态违反：`632 > 1.10·t` ⇒ `t < 574`；必须**可达**（否则一直放大到边界）：2048 全长时不切分的实测 TTFT P50/P99 = 448.5/515 ms ⇒ `515 < 1.10·t` ⇒ `t > 468` | **500** |
| `chunk_size_max` | `phase1_anchor` 的 `input_length=2048`，超过即等同不切分 ⇒ 留基线 16384 会造出"grow 了但不切分"的**死区**（与旧的 `cooldown_steps` 同类：日志记动作、实际无效果），并且**削弱**"建议值 vs 生效值差额"这套排障手法 | **2048** |

`chunk.chunk_size` 必须**等于** `chunk_size_init`（=1024）：恢复目标是**配置初值**，
两者不等会让 `adaptive` 从第一步起就与 `static_chunk` 跑在不同切片上，第一条差异
来自"初值不同"而不是"是否自适应"。`hysteresis`/`cooldown_cycles`/`control_interval_steps`/
`max_batched_tokens_*` 一律不动。

### 两道前置门（都在本机、都不花真机时间）

1. **结构性可动**（`gen_phase1_step3_configs.py::_assert_controller_movable`）：
   下界严格小于初值、上界严格大于初值、上界不超过负载 prompt 全长；任一不成立拒绝写盘。
   生成器同时改成 profile 驱动（`p1s3` / `p1s3b`），**旧配置可逐字复现**（测试钉住），
   `results/raw/P1S3-*` 的追溯链不断。
2. **阈值可达**（新增 `scripts/check_step3b_reachability.py`）：静态检查抓不到"目标可达性"。
   把 P1S3 实录的 P90 信号**开环复放**到修正版参数上：放大可达 c1/c2/c4、
   收缩可达 c2/c4/c8、**净方向**"放大 [1,2,4] / 收缩 [8]"符合"低并发净放大、高并发净收缩"、
   恢复探针 `1024→512→1024`（`n_recovered=2`）。
   开环复放**只证明阈值落在可触发位置**，不预测闭环终态（放大 chunk 会压 TTFT、放大自己会停）——
   c1 那 124 次 `clamp` 就是开环假象，随表一起声明。

### 本轮新踩的坑（都属"看起来正常、实际没生效"类）

1. **`sh -c` 在 Linux 上是 dash，`source` 不是它的内建命令。**
   启动矩阵时用 `sh -c 'source .../activate_musa.sh >/dev/null 2>&1; python ...'`，
   activation **静默失败**（错误被 `>/dev/null 2>&1` 吞掉），`python` 落到没有
   `transformers` 的解释器上，第一格即以 `ModuleNotFoundError: No module named 'transformers'` 失败。
   现象像"依赖没装"，实际是"激活从未执行"。同一串命令经 paramiko 默认 shell（bash）跑是好的
   ——**环境相关的失效**，别用"上次没事"推断这次没事。改用 `bash -c`（或 POSIX 的 `.`）。
2. **重定向到文件时 stdout 块缓冲、stderr 无缓冲 ⇒ 日志的行序不是执行顺序。**
   上述错误信息出现在日志**第 1 行**，而穿插执行计划在后面；我先 `tail` 日志，于是"看不到错误"。
   已在启动命令里加 `python -u`。教训：日志要按**内容**检索（`grep -n`），不要按位置读。
3. **`&` 绑定整条 `&&` 链**：`cd X && rm f && cmd > log &` 会把整条链放进后台子 shell，
   于是后续前台命令仍在**原 cwd** 执行。症状是 `tail: cannot open 'log'`
   （而文件其实已生成）——差一点让我把"已成功启动"误判成"根本没启动"。
4. **守卫读 `Profile` 的占位符导致恒真**：`_assert_controller_movable` 第一版读 profile 字段，
   而 `p1s3` 的 `chunk_size_min` 是 `None`（含义"沿用基线"），于是每条检查都被
   `if ... is not None` 跳过——守卫对**恰恰有缺陷的那份**恒真。
   已改成读**最终生成配置**的生效值，并有测试钉住"改回读 profile 就会失败"。
5. **恢复探针把信号写成"目标的函数"⇒ 不可能失败**：初版 `tpot = target_tpot·3`，
   目标定得越高压力信号越大，于是"能逼出收缩"恒真。已改为锚定实测区间的**绝对值**
   （压力段 300 ms / 消退段 60 ms，TTFT 450 ms）。
6. **轨迹不含初值时"第一步就动"不可见**：`chunk_series` 从第一次 `observe()` **之后**记起，
   于是 c8 明明收敛到 512，表里却是"收缩 0 次、min=max=512"，自相矛盾且看起来像"没动"。
   已改为以初值为第 0 项。
7. **`relative_to(_ROOT)` 在仓库外的配置路径上抛 `ValueError`**，把整道门带崩。已加兜底。
8. **测试里用 `spec_from_file_location` 加载脚本但没注册 `sys.modules`**：脚本带
   `from __future__ import annotations` 时 `@dataclass` 的注解是字符串，dataclasses 解析时
   要拿 `sys.modules[cls.__module__]`，未注册即以
   `AttributeError: 'NoneType' object has no attribute '__dict__'` 整块失败。已在 `_load_script` 里注册。

### 已验证 / 未验证

- 已验证：`p1s3` 逐字复现已提交的 12 份配置（本机 `diff -r` 无输出）；`p1s3b` 通过单变量
  自检与可动性自检；配置预检在本机与远程**输出一致**；可达性门在本机与远程**输出一致**；
  修正版与旧版的逐字段 `diff` 只差设计的 4 个字段 + 标签；全量测试 **528 passed**（+17）。
- 未验证：闭环行为。开环复放不能替代它，真机运行已启动（tag `P1S3B`，36 次 = 12 格 × 3 轮穿插）。
  结果已回填：`PERFORMANCE_MATRIX_PHASE1B.md` §8，逐格原始值与结论见本文件 2026-09-23 末条
  （"P1S3B 收齐与审查"）。**开环门的预言被真机逐项证实**（c8 的 6 次改动 = 1 次动 chunk + 5 次动
  token 预算），但真机同时给出了开环门**给不出**的东西：动作是否真的落到 prefill 上。
- 远程执行前确认：主机可达、`MTT S4000` 空闲（0% GPU / 4MiB 占用 / 无残留推理进程）。

---

## 2026-09-23 — P1S3B 收齐与审查：动作**确实**落到了 prefill 上；但自适应臂的重复性低于静态臂

### 收齐与独立复核

36/36 成功、0 失败。收下来后**重新独立核对**，而不是相信拉取动作本身：
144 个文件全部非空、tar 成员大小与磁盘逐一相符、网格完整（3 臂 × 4 并发 × 3 轮，
无缺格无重复）、36/36 `data_label=MEASURED`、36/36 `meta.json.run_id` 与目录名一致、
每目录恰好 4 个预期文件。

### 新增的永久审计：`--prefill-effect`（回答"动作有没有可施加的对象"）

原来的 `--controller` 只回答"控制器改了什么"，它**结构上就看不见**下面这件事：
控制器在纯 decode 阶段改了一次 `chunk_size`，此后再无 prefill ——
这次改动对 prefill 的贡献是零，却在 `chunk_requested(min/max)` 里照样显示"改过 512/2048"。
两个视图**刻意分开**，不合成一张表：合并之后，"改了参数"与"改到了 prefill"
会变成同一个格子，而这两件事在真机产物里确实分开过。

分类是**五态**而非布尔（"生效/未生效"会丢掉成因，而不同成因对下一步的指示相反）：

| 态 | P1S3B 实测出现次数 |
| --- | --- |
| `value_unchanged`（标记已改但取值没变；被夹住或重申） | c8 5 次 |
| `applied_to_prefill`（改的值就是后续 prefill 用的值） | c2 3 次、c4 1 次、c8 1 次 |
| `superseded_before_prefill`（下个 prefill 前又被改掉） | c2 5 次、c4 3 次 |
| `no_prefill_after`（此后再无 prefill） | c2 2 次、c4 2 次 |
| `not_applied_to_prefill`（值保持到下个 prefill 却没用它） | **0 次** |

**第 3 态是我写错之后补上的，值得记下来**：初版只有四态，把 c2/c4 里
"step 8 改成 512、但下个 prefill 在 step 129（那时已被 step 80 改回 2048）"判成了
"改了却没生效"。**这个判断是错的**——512 在整个 step 8–31 区间一直是生效值，
只是那个区间里一次 prefill 都没发生。它把**负载形状的读数**冒充成**控制器缺陷**，
两者对下一步的指示完全相反（前者是"下一轮改持续补入"，后者是"控制器有 bug"）。

结论：**c1 零动作**（16 个 prefill 事件全用 1024，故 c1 的差异恒为 0 = 无事可做，
不是无收益）；**c2/c4 落到 prefill 上的是"扩容"那一半**（1024→2048，后续 prefill 用 2048）；
**c8 的收缩落到了 prefill 上**（1024→512，其后 14 个 prefill 事件用 512），
它后面 5 次 `changed=True` 其实都在动 `max_batched_tokens`，`chunk_size` 已触下界 512 不动。

**独立自校验**：两批 12 个基准 run 的实际 prefill 总 token **恒为 16384**（= 8 请求 × 2048），
且等于 `requests.jsonl` 求和。⇒ 控制器是**重新分配**了 prefill 工作，未丢失也未重复。
这一条同时是审计器的自校验（解析器丢行或重复计数会让总和对不上）。

### 两个新发现

**发现 1（本批最重要）：自适应臂的重复性系统性地低于静态臂，且成因在控制器本身。**

`adaptive_c4` 三轮 TTFT P50 = `481.37 / 481.57 / **652.15**`；同批 `static_chunk_c4` =
`619.77 / 619.88 / 619.28`。这不是机器抖动——因为那一轮的 `stall P99` 反而**更低**
（489.13 vs 570.07/569.49），说明控制器走了**另一条决策路径**：
第 3 轮 step 80 没产生动作、step 136 收缩到 1024、直到 step 184 才升到 2048；
另两轮是 step 80 即达 2048、step 136 退到 1536。`n_clamped` 三轮 = 8 / 6 / 0。
根源是控制器以**带噪的实测信号**（TTFT P90 = 2217.9 / 2218.3 / 2220.0 ms）＋冷却计数为输入，
路径本身不可复现。

⇒ 后果必须明说：c4 的 TTFT 优势按中位数是 −22.3%，但**最差一轮是 +5.2%（劣于基线）**，
**符号在重复之间翻转**。因此"自适应更优"这句话在本批证据下**只对 c2 成立**。

**发现 2：`stall` 对"预填充阶段的孤立停顿"是盲的。**

`static_c1-r1` 里 `r00003` 的 `prefill_exec_ms` = **11136.2 ms**，而同 run 其余 7 个请求的中位数是
455.0 ms（**24.5×**），`queue_ms`/`admission_ms` 均为 0、`outcome=success`。
该事件把 TTFT 的 p90/p95/p99 抬到 3661.8/7399.0/10388.8 ms，而 **`stall P99` 只有 64.68 ms**——
因为它是 prefill 事件、不是 token 间隔。两批 72 个 run 系统排查（判据 `prefill_exec > 10× 同 run 中位数`）
只有这 1 例。⇒ 只监控 stall 会漏掉整类事件；必须与 TTFT 分位（附 n）及
`admission/queue/prefill_exec` 分解并列看——这也是当初把 TTFT 拆成三段的价值所在。

### 三处配置缺陷均已确认修复

`chunk_size_init` 512→1024（解维度死锁）、`target_tpot_ms` 50→110（可达）、
**以及最初漏掉的第四处** `target_ttft_ms` 2000→500（原值高于四档实测 TTFT P90 全部取值
⇒ grow 判据一次都不触发）。真机旁证：`adaptive_c2` 已出现 grow 至 2048 的动作、
`adaptive_c8` 出现收缩并持续生效，`n_recovered` 不再恒为 0。

另记一条**必须显形**的事实：**`static_chunk` 的 `chunk.chunk_size` 同时从 512 改成了 1024**，
所以 P1S3B 的 `static_chunk` **不是** P1S3 那一格的复现；跨批比较分块基线**不是单变量**。
两批**内部**才是单变量：`static` vs `static_chunk`（只改 `chunk.enabled`）、
`static_chunk` vs `adaptive`（只改 `controller.enabled`）。

### 未验证 / 缺口

- **计时口径是修复前的**：本批 `note` 形如 `prefill_tok=… decode=… forwards=… fwd_ms=… pack_ms=…`，
  **不含 `readback_ms`**——而修复后的 `benchmark/real_backend.py`（第 1941 行）必然追加该字段。
  故本批 TTFT/E2E 记的是 **logits 就绪时刻**，未含采样与设备→主机取回，**系统性偏早**。
  批内相对比较成立；**绝对 TTFT 不得与修复后口径的运行并表**。
- **版本溯源未满足**：`git_commit` 36/36 = `NOT_MEASURED`（注意：字段**存在**且值显式为"未测"，
  并列入 `_missing_fields`——不是字段缺失），`model_revision` 36/36 = `'main'`。
- 静态基线只扫了 1024 **一个点**；"自适应优于精心选择的静态配置"**尚未**回答。
- 每格 8 请求、并发恒定 ⇒ 主要反映启动与排空；发现 1 的"路径不可复现"是否在持续补入负载下仍在，
  尚不可知。
- `run.slo.tpot_ms` 仍为 50 ⇒ 达标率三列恒为 0、零信息量（只改了 `controller.target_tpot_ms`）。

### 工具侧的一个方法学结论（变异测试）

给审计器写的 36 项测试做了 10 个变异，**8 个被抓、2 个是等价变异体**（把
`chunks[i-1]` 换成 `chunks[idx[pos-1]]`、以及 `if i > 0` 换成 `if pos > 0`）：
二者在真机上**恒等**（36 份产物里 pos>0 的 117 个样本全部相等；首个改动之前不可能有改动，
故那时 `chunk_size` 必等于 `init_chunk_size`）。
⇒ `rc == 0` 有两种成因：**守卫太弱**，或**变异体根本不改变行为**。
要分清就得换一个能改变行为的变异（去掉初值回退 → 8 个失败；before 恒取首个值 → 13 个失败）。
防的是"看到 rc=0 就去加断言"——那只会加上一堆同样抓不住东西的断言。

---

## 2026-09-23 — 版本追溯打通：部署戳、投放现场校验，以及"远端那棵树到底是哪些版本"

### 起因

`EXPERIMENTS.md` §4.3 的 **P1（版本溯源）**：`results/raw/P1S3B-*/meta.json` 的
`reproducibility.git_commit` 是 `NOT_MEASURED`（36/36），P1S3 同样。根因不是 harness
偷懒——**远端源码树是拷贝过去的、不带 `.git`**，`git rev-parse HEAD` 必然失败。
后果是那批数字**指认不到源码版本**：半年后没人能回答"这批结果是哪个版本的代码跑出来的"。

### 机制：戳必须能被证伪

一句话：**一个永远不会过期的戳比没有戳更糟**，因为它把"版本未知"伪装成"版本已确认"。
所以设计上有三处刻意：

1. `build_source_stamp()` 在**本机**生成 `SOURCE_PROVENANCE.json`（`git_commit` /
   `git_dirty`（限作用域）/ `tree_digest` / `digest_scope` / `deployed_at` / `deployed_by`）。
   取不到 commit、树脏、算不出摘要 —— 三种情况**都抛错**，不产出"凑合的戳"。
2. `read_source_stamp()` 读时**重算摘要比对**；不符即作废并把 `git_commit` 置空。
   并区分两种"对不上"：**作用域变了**（该重投）与**源码被改过**（该对树）——
   混成一句"摘要不符"，人会按第一种去查，查不出，最后把戳当成坏的不用了。
3. `detect_source_commit()` 三级优先：真 git 仓库 > 有效戳 > 取不到；
   `git_source ∈ {git, stamp, stamp_stale, none}`，其中 `stamp_stale`（有戳但不可信）
   与 `none`（从来没有戳）**不合并**——修法不同。

**最关键的一步不是"写文件"，而是"当场判定"**：`scripts/remote_run.py stamp <远端根>`
投完之后，**立刻在远端用那棵树里的 `read_source_stamp` 重算摘要**，不符返回非零并
逐文件列出差异（复用远端 `iter_scope_files`，不另写一份遍历）。没有这一步，
"戳投好了"只是一个未经证实的断言。`push` 默认在推完源码后自动做这件事。

### 真机证据

```
投放并校验通过：远端以 /root/miniconda3/bin/python3 3.10.8 重算摘要相符，
                读回 commit=10632452af1c（来源 stamp）
DETECT    ('10632452af1c…', 'stamp', False)
VALIDATED True | 部署戳已校验（摘要相符）
DEPLOYED  2026-09-23T16:59:06+0800 by LAPTOP-1LQ6BQ1P/13944
DIGEST    1b7b929575d9021551ffa669dfcea86f5942737cd9e7322177ca01aa2ea72910
RECORD    git_commit='1063245…' git_source='stamp' git_dirty=False
```

* 本机 3.13 与远端 3.10.8 对同一棵树算出**同一个摘要** ⇒ 摘要算法不依赖解释器版本。
* 远端 `python3` 是 `/root/miniconda3/bin/python3 3.10.8`，不是推理用的 venv——校验只依赖
  标准库，因此不必激活 `activate_musa.sh`，也就绕开了那套 `LD_PRELOAD` 的坑。
* 作用域 = `src/ascend_opt` / `benchmark` / `scripts`。**不含 `configs/`**：配置内容已经在
  每个 run 的 `meta.json.config` 里逐份落盘，把它收进作用域只会让摘要随每次生成配置而失效。
* 比对是**逐字节**的，故 CRLF 与 LF 之差不放过。这是有意的：本仓 `.gitattributes` 要求
  仓内一律 LF，且 CRLF 曾让远端 shell 脚本报 `$'\r': command not found`。

### 一个新方法：把远端每个文件回查到本仓库历史

远端树不是 git 仓库，"它是哪个版本"一直只能靠日志叙述。但**每个文件的内容**在 git 里
是有 blob 哈希的：在远端跑 `git hash-object <file>`（不需要仓库），拿回的 sha1 在本机
`git cat-file -e` / `git rev-list --all --objects` 里能查出它属于哪个提交、或是否**任何提交
都没有**。远端 54 个文件的结果：

| 分类 | 个数 | 含义 |
| --- | --- | --- |
| `= HEAD` 的版本 | 24 | 无缺口 |
| 同路径的老版本 | 6 | HEAD 已前进（`real_backend` / `run_benchmark` / `inspect_step3_runs` / `remote_run` / `validate_fp16_path` / `provenance`） |
| 内容在仓库、**别的路径** | 10 | 存档在 `results/raw/_probe/20260922/scripts/`——**不是"从未入库"** |
| 内容不在仓库任何地方 | 14 | 13 个远端自造探针 + **`benchmark/workloads.py`** |

两条方法论要点：

* 我最初只查了 `scripts/qwen_single_request.py` 这个路径，于是把 10 个文件误判成"从未入库"。
  **路径查不到 ≠ 内容不存在**——必须按 blob 内容查，而不是按路径查。
  这个错误如果没被抓到，就会得出"远端有 23 个文件只此一份"的错结论。
* `benchmark/workloads.py` 一开始被列为"内容不在仓库"，进一步比对后确认它**与 HEAD
  逐字节只差行尾**（440 CR ↔ 440 LF，`tr -d '\r'` 后 diff 为空）。它不是"被改过"，
  是更早一次推送留下的 CRLF 副本。⇒ 重新推送即自动对齐。
  **先归一化换行再 diff** 这一步省下了一次错误的"远端被改过"结论。

### 两个被新测试逼出来的真缺陷

1. **`detect_git_commit(root=None)` 会继承进程当前目录**（`subprocess.run(cwd=None)` 的语义）。
   于是"调用方没传源码树根"被静默变成"把进程恰好所在的那个仓库的 commit 记上去"——
   在开发机上就是**本机开发仓库**。这意味着一份声称"远端实测"的产物里可能出现一个
   与远端毫无关系的 commit，**且不报错**。触发它的是一条断言 `None → ("", "none", None)`
   的测试（写测试时以为在测一个平凡性质）。现在 `None` = 没查，绝不回落。
2. **git 路径下 `git_dirty` 恒为 `None`（"没查"）**。看着无害，实际把唯一的警告信号关掉了：
   工作区有未提交改动时，HEAD 同样不描述被测的那份代码，而记录里只有一个光秃秃的 commit。
   已改为走 `detect_git_dirty`（限作用域）。实测：改 `src/ascend_opt/provenance.py` →
   `git_dirty=True`；提交后 → `False`；只改 `tests/`（作用域外）→ 仍 `False`。
   > 与写端的不对称是**有意**的：写端（部署戳）默认**拒绝**脏树，因为一份戳是一句
   > "这个 commit 就是被部署的代码"的断言；记录只是陈述事实，如实标 True 即可。

### 守卫抓住了我自己的疏漏

第一次 `push`（含投放）**以退出码 6 被拦下**，逐文件差异指出远端 `scripts/` 里还有
10 个本机没有的文件。我上一轮只移走了 13 个 `_*.py`，把另外 10 个按"内容在仓库里有存档"
当成"没问题"——**"能找回"不等于"两棵树相等"**。这正是本次要建的那个判据的价值：
它拦下的第一个对象是写它的人。移走那 10 个之后重新投放，校验通过。

### 远端 `adhoc/` 与 `results/raw/_probe/20260923/`

部署要求"作用域内两棵树逐字节相同"，而远端 `scripts/` 里累积了 **23 个只在那台机器上
存在的脚本**（V3/V4 分块预填充调查链、环境诊断、Qwen 探针等）。处置（用户决定）：

* 13 个（`_probe_v3*.py`、`_chk_tie.py`、`_probe_logits.py`、`_t_single_variable_guard.py` …）
  取回 `results/raw/_probe/20260923/scripts/`，沿用 20260922 的约定（`_` 前缀 = 不是一次 run）；
* 另外 10 个已在 `_probe/20260922/scripts/` 有逐字节存档，不重复取回；
* 远端一律 `mv` 到 `/root/ascend_probe/adhoc/`——**不删除**，仍可复跑，只是移出作用域。

⚠️ **本批的一个缺口（已写进该目录 README）**：这 13 个脚本把读数 `print` 到 stdout，
**不写 JSON 产物**（实测只有属于 20260922 批的 `mem_bandwidth.py` / `qwen_single_request.py`
会写）。因此它们报过的每个中间读数（top1−top2 差值、KV 长度、各档批规模的通过与否）
**都不可复核**，只有本日志的叙述。补法是重跑并把输出落盘；在此之前不得引用其数字。

与本轮主线相关的两点：`_probe_v4.py` 的 docstring 是**修复 #1**（V4 在分块 prefill 后
未继续 decode）的直接前史——问题被记下来了、正确用法也写进了脚本，但**结论没落成产物**；
`_t_single_variable_guard.py` 是"守卫必须能失败"的一次具体实践：它不检查配置，
而是**证明检查会红**。

### 顺带发现

* `C:\c\tmp\meta_probe\probe\`（含 `meta.json` / `requests.jsonl`）在**仓库之外**：
  更早某次会话把 `/c/...` 或 `/tmp/...` 形式的路径传给了本机原生程序，MSYS 关掉转换后
  被解释成盘根下的相对路径。本轮我自己又复现了两次（`sftp get` 的本机落点、`run_benchmark --out`）。
  **规则：给本机原生程序传本机路径要用 `C:/...`，不要用 `/c/...`；`MSYS_NO_PATHCONV=1` 只解决
  远端路径方向的问题，反方向要自己写对。** 那两个目录已移回 `%TEMP%`，`C:\c\tmp\meta_probe\`
  是更早会话的产物，**未动**（不是我的东西，也不在仓库里）。

### 仍未验证（下一步）

* ~~**模型侧** `model_revision_resolved` 还只在本地 dry-run 验证过落盘~~ →
  **已于 2026-09-24 在真机验证通过**，见下一条（P2 批次 12/12 取到权重 commit）。
* P1 只对**新**产物成立。既有 72 个 run 保持 `NOT_MEASURED`，**不回填**。

---

## 2026-09-24 — P2 计时口径前后对照：修复的作用 ≤ 0.25%，顺带撞出一个控制器临界点

### 做了什么

真机跑 P2 批次：`static_c1` / `static_c4` / `static_c8` / `adaptive_c4` 各 3 轮，
穿插执行，**12/12 成功、0 失败**（`MATRIX_RC=0`）。四格的配置与 P1S3B **逐字节相同**，
所以两批之差＝两项代码改动的总作用。产物 `results/raw/P2-*`（12 目录 / 48 文件），
分析器 `scripts/inspect_timing_scope.py`（新工具）。登记见 EXPERIMENTS.md §1.3 / §4.3.2。

### 一、被错记的区间有多长（直接测量）

修复后的 `telemetry` 新增 `readback_ms` = "logits 算完" → "token 可取回主机"，
**正是修复前被记成 0 的那一段**：p50 = **0.240 ms**，且 **N=1 / 4 / 8 取同一值** ——
成本由 kernel 启动 + 同步开销主导，不是传输量。占设备计算 0.21–0.40%。

修复前是**逐元素** `int(next_ids[i])`（N 次小同步，且计在 `pack_ms` 里），
修复后是 1 次批量 D2H。所以这次提交其实捆绑了两件事，**不可分解**：
① 时间戳从"算完"移到"可取回"（口径修正）；② 去掉 N−1 次同步（真实提速）。

### 二、对报告值的总作用（前 P1S3B → 后 P2，p50）

| 指标 | c1 | c4 | c8 |
| --- | --- | --- | --- |
| TTFT | +0.103 ms (+0.023%) | +0.453 (+0.027%) | +0.484 (+0.021%) |
| TPOT | +0.032 (+0.051%) | −0.139 (−0.156%) | −0.294 (−0.207%) |
| E2E | +4.12 (+0.050%) | −17.17 (−0.133%) | −37.08 (−0.182%) |
| `pack_ms`（纯 decode） | −0.170 | −0.290 | −0.500 |

方向自洽：TTFT 变晚（口径修正），TPOT/E2E 变快（少做同步）。**全部 ≤ 0.25%**，
**没有既有结论会翻转**。但两批**不得并表**。
**反面也记下**：c1/c4 的 TTFT 增量**小于各自批内漂移**（0.246 / 0.539 ms）⇒
这两格"修复对 TTFT 的影响"**测不出来**，只能说量级 ≤ 0.24 ms。只有 c8 可分辨。

### 三、意外发现：控制器门限贴着工作点 ⇒ 0.2% 的观测差异造出 43% 的时延差

`EXP-adap-114`（P1S3B `adaptive_c4`）记 `TTFT P50 = 481.57`，那是 3 次重复的**中位数**，
掩盖了三次实际是 `481.4 / 481.6 / 652.1`。差的只是**第二批到达的 4 个请求**。

`controller_trace` 直接读出根因：放大 chunk 的门限是
`tpot_p90 ≤ 110 × (1 − 0.1) = 99.0 ms`，而实测工作点 `tpot_p90` 是 **97.8 – 99.2 ms**。
门限**落在工作点分布内部** ⇒ 每次"放大"决策的 headroom 都在 0.1004–0.1110（门限 0.1000），
整条爬坡近似抛硬币。`r2` 在 step 80 以 `tpot_p90 = 99.23`（**差 0.23 ms**）被拒，
`chunk_size` 卡在 1536 直到 step 185；而**负载 `prompt_len = 2048` 正好等于
`chunk_size_max`** ⇒ 1536 装不下整个 prompt，要拆成 1536+512 两步。
后果：那 4 个请求 TTFT `454.7 → 652.1`（**+43%**），而第二批在 step 129 就到了。

⇒ 该格在 P1S3B 上**跨重复不可复现**，任何拿它做的"自适应 vs 静态"比较**无效**。
这与 P1S3 那版的缺陷同源、方向相反：那版 `target_ttft_ms=2000` 高于全部实测值
（判据一次不触发），本版门限落在工作点内部（判据随机触发）。
**共同修法：阈值按控制器实际读到的 `controller_trace[*].signals.*_p90_ms` 标定，
不按报表的逐请求分位**——本例判据用 ≈98.9，而 `summary.tpot.p50` 是 110.15。这条直接进 P3。

### 四、顺带修掉一处错指针（"错的值比缺的值危险"）

EXPERIMENTS.md §1.2（P1S3B）的 12 行"配置"列原先写 `configs/mthreads/phase1_step3/…`，
那是 **P1S3 那一版**。产物自证本批用的是 `phase1_step3b`
（`meta.json.config`：`chunk_size=1024` / `target_tpot_ms=110` / `target_ttft_ms=500`；
`phase1_step3` 是 `512` / `50` / `2000`）。**照错指针复现会加载到那版缺陷配置**，已修正（§1.1 未动）。
另修 `benchmark/real_backend.py` 一处**由 0c72a6a 遗留的过时注释**：结算处仍写
"token 的可得时刻取自驱动在**设备同步之后**打的点"，而修复的要义恰是区分"算完"与"可取回"。

### 五、这次的两个方法学收获

1. **差点写下一个错误的恒等式。** 我一度把 `readback_ms` 当成"修复后的残余偏差"，
   并据此写 `bias_pre = readback_ms + Δ报告`。其实修复后 TTFT 取的就是 token 可取回那一刻，
   **口径已经正确**；`readback_ms` 只是"此前被记错的区间有多长"。
   而且 Δ报告 里混着口径修正与真实提速，**两者不可分**（两批代码在同一步做的工作量不同）。
   ⇒ 分析脚本的 F 节改成只报 Δ报告 + 一个自我否证条件：
   **`|Δ报告|` 不大于本批批内漂移时，该格"测不出"，拒绝给数。**
   这个条件真的拦住了东西——`adaptive_c4` 代进去会算出 `−169 ms / −26%` 的假结论。
2. **空输入会伪装成合格结果（又犯一次）。** 新写脚本的三态检查（`model_revision_resolved`
   必须非 `None`）在"一个 run 都没读到"时照样打印"✓ 全部为 True"。
   已改成先判 `n_checked == 0` 并显式输出"本项**未检验**（不是通过）"。

## 2026-09-24 — P3：把 `slo.tpot_ms` 从"够不着"标定到"能分辨"（50 → 100）

### 起因：一个恒为 0 的指标列

`run.slo.tpot_ms` 从第③步起一直是 **50 ms**。逐请求 TPOT 在 S4000 上是 60–170 ms
⇒ 没有任何请求达标 ⇒ `slo_satisfaction_rate` / `slo_goodput_req_s` /
`slo_goodput_tok_s` 三列**恒为 0.000**。P1S3B 只把 `controller.target_tpot_ms` 改成了
110 —— **那是控制目标，不是达标判据**，两个字段、两种用途，改了一个另一个不会跟着动。

这不是"达标率低"，是**判据无效**：一个对所有配置都给 0 的指标不携带任何信息，
放进对照表会让人读成"两种配置都很差"。属于本项目反复强调的**静默失效**
——产物齐全、标签全绿、没有任何报错。

### 一、口径先分清（两处最容易混）

* `SLO.satisfied_by()` 是**逐请求**判定（`m.tpot_ms > 阈值`），不是比汇总分位；
* 同时配了 `ttft_ms` 与 `tpot_ms` 时是**与**关系；分母是**全部提交**（含失败/未完成）。

⇒ 标定必须看**逐请求**分布。新工具 `scripts/calibrate_slo.py`（只读，不写配置）。

### 二、实测分布与三个候选（`P1S3B`，96 个成功请求）

| 层 | 逐请求 TPOT (ms) |
| --- | --- |
| c1（三臂） | 61.7 – 62.0 |
| c2 | 77.9 – 86.1 |
| c4 `static`（不分块） | 88.7 |
| c4 `static_chunk` / `adaptive` | 98.7 – 112.0 |
| c8（三臂） | 133.5 – 168.0 |

合并区间 **[61.7, 168.0]**；旧值 50 **低于下界**。（`P2` 并入后 n=128，区间不变。）

候选 90 / 100 / 110 给出**完全相同的定性分档**（c1/c2 通过、c4 按臂分裂、c8 全失败），
三者都落在 **[88.7, 110.4]**——即 c4 的"不分块 / 分块"代价区间——之内 ⇒ **结论对该选择稳健**，
差异只在 c4 的粒度。取 **100**：它同时等于 **10 tok/s**，是一个外部可辩护的
交互式流式边界，不需要靠"哪个值让结论更好看"来辩护。

### 三、一个意外口径发现：TTFT 判据当前从不单独起作用

在 `ttft_ms = 2000` 下，**没有任何请求只**因 TTFT 不达标（0 个"仅TTFT超"；
16 个是"两者都超"）。也就是说达标与否**实际由 TPOT 决定**，
`ttft_ms` 只在最差的那批请求上与 TPOT 共同起作用。
这不改变结论，但引用"TTFT 达标率"时必须知道它目前不是一个独立的信息维度。

### 四、`--dry-run` 验不了这一项（重要）

顺手跑了一次平台模板的 dry-run 冒烟：一阶设备模型的 TPOT 只有 **7.7 – 10.25 ms**，
比真机低约一个数量级 ⇒ **任何**阈值都会读成 `1.000`。
这条再次印证"静默失效不会被任何运行时检查发现"——它只能靠**真机口径的逐请求数据**标定。
（本次 dry-run 产的 3 个目录已删，未留在 `results/raw`。）

### 五、改动清单与守卫（单变量：只有 `run.slo.tpot_ms`）

* 新 profile `p1s3c` → `configs/mthreads/phase1_step3c/`，与 `p1s3b` **只差**
  `run.slo.tpot_ms`（50→100）与标签 `run.notes`（③b→③c）。已用 `diff -r` 逐键核对：
  全部差异只有 `tpot_ms` 与 `notes` 两处 × 12 文件。
* `p1s3` / `p1s3b` **逐字不变**：重生成到临时目录后 `diff -r --brief` 报 0 差异。
  这是红线——已执行证据的配置不能因为"模板改了"而漂移。
* 平台模板 `configs/mthreads/s4000_fp16.yaml` 与 `RunOptions.slo` 兜底默认同步为 100
  （同约定的另两处定义，不跟着改就会各自漂移）。
* 生成器新增 **`_assert_slo_reachable`**：阈值低于实测下界或高于上界都拦下
  （两者都不携带信息）。`p1s3`/`p1s3b` 的旧值转为**已声明的已知缺陷**，
  需 `--allow-known-defects` 才放行——与 chunk 维度死锁那条走同一个逃生口。
* 既有产物**不重跑**：`run.slo` 不参与任何调度决策，只影响派生指标 ⇒
  达标率按新阈值**重算**（100 下：c1/c2 1.000、c4 `stat` 1.000、c4 分块臂 0.125、c8 0.000、
  全格 0.604）。

### 六、这次踩到的一个"字面量类型"坑（无报错、只有假 diff）

`Profile.slo_tpot_ms` 我第一版写成 `50.0`，重生成后 12 个已执行配置**全部**出现
`tpot_ms: 50` → `50.0` 的差异。**YAML 会把 Python 字面量的类型逐字带到配置里**，
而模板里是 `50`。这类改动"内容对、字节错"，不会有任何报错，只有量字节或看 diff 才发现。
修法：整数值用 **int 字面量**，并把这条写进字段注释。



## 2026-09-24 — P4：控制器门限从"贴着工作点"抬到"分得开"（110/0.1 → 120/0.08），并暴露一个判据形式的结构性障碍

### 起因：同一份配置，三次重复的 TTFT 差 43%

P1S3B 收齐时就看到 `adaptive_c4` 的三次重复 TTFT = 481.4 / 481.6 / **652.1**。
当时的记法是"重复性低于静态臂"。P2 做完计时口径前后对照后，那个"临界点"被
定位到具体字段：`controller.target_tpot_ms(110) × (1 − hysteresis(0.1)) = 99.0`，
而 c4 的工作点中位数是 98.7 —— **相距 0.3 ms**。这不是"抖动"，是**一次浮点比较的
符号决定整格行为**。

### 一、控制器判据的结构（两个方向、各自的门限）

`AdaptiveController._decide`（`src/ascend_opt/controller.py`）：

* **收缩**：`tpot_p90 > target·(1+h)`
* **放大**：`ttft_p90 > target_ttft·(1+h)` **且** headroom `(target − tpot_p90)/target > h`
  ——即同时要求 `tpot_p90 < target·(1−h)`。**两个条件都要满足**，只满足 TTFT
  不会放大。
* 分支顺序：`memory` ⇒ `tpot` ⇒ `ttft` ⇒ `recover` ⇒ `none`。

所以 `(target, h)` 一对参数同时决定**两个门限**：放大门限 `target(1−h)` 与
收缩门限 `target(1+h)`，带的总宽是 `2h·target`。**带宽是绝对量**（不以负载归一化），
这一点后面会变成关键。

### 二、控制器**实际读到**的信号 ≠ 报表里的分位（本次最容易搞错的一处）

标定必须用 `controller_trace[*].signals.tpot_p90_ms`——控制器**实际读到**的那个量。
它和报表里的 `summary.tpot.p50` 不是同一个东西（c4 那格：98.7 vs 110.15，差 11.5 ms）。
拿报表分位数去标门限，会把门限摆到错的位置，而且不报错。

实测（`scripts/calibrate_controller_gate.py --prefix P1S3B,P2`，adaptive 臂**首轮**）：

| 格 | 决策点数 | tpot_p90 min / **p50** / max (ms) |
| --- | --- | --- |
| `adaptive_c1` | 128 | 61.8 / **62.0** / 65.0 |
| `adaptive_c2` | 64 | 81.6 / **81.9** / 291.6 |
| `adaptive_c4` | 64 | 97.8 / **98.7** / **571.7** |
| `adaptive_c8` | 18 | 116.2 / **142.1** / 415.8 |

⚠️ 注意 c4 的 max = **571.7**。单格内就跨 **5.8×**，四档合起来 **61.8 → 571.7（9.3×）**。
这条后面成了本节最重要的结论。

### 三、c4 的余量只剩 0.0112，以及候选对比

放大判据的 headroom 上限（即 110/0.1 下 c4 那 64 个决策点的最大 headroom）
= `(110 − 97.8)/110` = **0.1112**，对门槛 **0.1000** ⇒ 余量 **0.0112**。
也就是说：c4 里"能放大"的那些点，只有 **1.1 个百分点**的余量就掉到门槛下。
重复之间的微小扰动足以把它们推过去。

候选对比（工具原始输出）：

| target/h | 放大门限 | 收缩门限 | 滞回带宽 | c4 headroom max | **c4 余量**（=max−h） | c4 放大触发 | c8 收缩触发 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 110/0.1（旧） | 99.0 | 121.0 | 22.0 | 0.1112 | **0.0112** | 53/64 | 17/18 | ❌ 门限压在 c4 工作点上 |
| 110/0.08 | 101.2 | 118.8 | 17.6 | 0.1112 | 0.0312 | 60/64 | 17/18 | ❌ 只抬了余量，中位数仍贴门限 |
| 120/0.1 | 108.0 | 132.0 | 24.0 | 0.1852 | 0.0852 | 60/64 | 17/18 | ⚠️ 可用 |
| **120/0.08（采纳）** | **110.4** | **129.6** | **19.2** | 0.1852 | **0.1052** | 60/64 | 17/18 | ✅ 余量 ≈10× 于旧值 |
| 130/0.1 | 117.0 | **143.0** | 26.0 | 0.2479 | 0.1479 | 60/64 | **5/18** | ❌ 收缩门限跨过 c8 中位数 142.1 |

⚠️ **本表第一版有两处错，都已修**（写下来因为它是"错的值比缺的值危险"的现成例子）：

1. **`headroom max` 与 `余量` 混填在一列**：前者是实测 `max((target−tpot_p90)/target)`，
   **只随 `target` 变、与 `h` 无关**；后者才是判据关心的"离门槛多远"（= 前者 − `h`）。
   混填的后果是把 120/0.1（0.1852）排在了 120/0.08（0.1052）前面——**恰好相反**。
   同一列里"有的行是余量、有的行是原始值"不会报错、没人会多看一眼，
   但它能让结论翻向。
2. **原始表头漏了 `h` 也是带宽旋钮**：`h` 变小既抬高余量、也**收窄**带宽
   （`2h·target`：24.0 → 19.2 ms），而带宽窄 ⇒ 决策更容易来回翻。
   所以 120/0.08 不是"更优"，是**折中**。

**130/0.1 被否的理由值得记下来**：它的 c4 余量最大（0.1479），但收缩门限
143.0 跨过了 c8 的中位数 142.1，把 c8 从**确定收缩（17/18）**变成**不确定（5/18）**。
即"拿一个格的不确定性去换另一个格"——**不是改进**。

### 四、结构性障碍：绝对阈值 + 单一 `(target, h)` 与多档工作点分不开

这一条是本次最重要的产出，而且**不是好消息**：

* 工作点的**分布**跨 **61.8 → 571.7 ms（9.3×）**（单格 c4 自己就跨 5.8×）；
* 滞回带宽度是**固定绝对值** `2h·target` = `2 × 0.08 × 120` = **19.2 ms**。

⇒ 一个固定的绝对阈值 + 一对固定的 `(target, h)`，**无法同时与跨 9× 的多个负载档位
工作点分开**。总有一档落在带里。

**120/0.08 并没有解决它**：c4 的区间 [97.8, 571.7] **仍跨过**新放大门限 110.4
（放大 60/64、收缩 4/64、带内 0/64）。采纳它的理由是"最坏余量从 0.0112 抬到
0.1052"，是**定量改善**，不是定性解决。

而对**核心实验**来说这一条恰好致命：核心实验要"持续补入负载"，负载是**连续变化**的，
滞回带**必然**被非确定地穿过。⇒ 要解决它需要改**判据形式**（例如把门限按工作点
归一化、或让 `(target, h)` 随负载档位调度），**不是再调一个数**。登记为待办，
在核心实验设计时处理。**不要把 `p1s3d` 读成"门限问题已解决"。**

### 五、改动清单（单变量：门限两参数 + 标签）

* 新增 profile `p1s3d` → `configs/mthreads/phase1_step3d/`，与 `p1s3c` **只差**
  `controller.target_tpot_ms`（110 → 120）、`controller.hysteresis`（0.1 → 0.08）、
  标签 `run.notes`（③c → ③d）——`diff -r` 逐行核对过，只有这三处；
* `p1s3` / `p1s3b` / `p1s3c` **逐字不变**：重生成 + `diff -r --brief` 零差异；
* 生成器新增守卫 `_assert_gate_separation`（读**最终产物**的生效值），
  `p1s3b`/`p1s3c` 的门限改为**已声明的已知缺陷**，需 `--allow-known-defects` 放行；
* `--profile` 默认值 3c → **3d**。理由是机械的：默认值必须是"干净的那一版"，
  否则无参数的 `--check` 会被自己的守卫拒掉（真的发生了，rc=1）；
* 标定工具 `scripts/calibrate_controller_gate.py`（**只读，不写配置**——门限取值是
  跨实验约定，必须登记后另行单变量改动）；
* 测试：`TestGateSeparationGuard` 新增 9 项（含 4 条反向对照）、`p1s3c`/`p1s3d`
  逐字复现各 1 项、`c → d` 单变量 1 项 ⇒ 该文件 37 项通过；
* **守卫的突变测试**：把 `_assert_gate_separation` 的 `return problems` 改成
  `return []`（= 无守卫），5 条反向对照**全部失败** ⇒ 守卫断言真的会咬；
  恢复后 md5 与突变前一致、`CR=0`、无残留。
* 全量回归：**686 passed / 462 subtests passed**（干净跑，234 s）。

### 六、开环复放的门槛判定**变了**（本轮的实测折中，不是纯改进）

重跑前跑第一道前置门（`scripts/check_step3b_reachability.py`，不花硬件时间），
把 P1S3 的实录信号分别复放到两套参数上：

| 配置 | 放大可达 | 收缩可达 | 恢复路径 | 净方向 | 格内双向振荡 | 脚本判定 |
| --- | --- | --- | --- | --- | --- | --- |
| `phase1_step3b`（110/0.1） | 是 | 是 | 是（n_recovered=2） | 放大 [1,2,4] / 收缩 **[8]** | [2, 4] | 通过（净方向**符合**预期） |
| `phase1_step3d`（120/0.08） | 是 | 是 | 是（n_recovered=2） | 放大 [1,2,4] / 收缩 **[]** | [2, 4, **8**] | 通过（净方向**不符合**预期） |

两者都**通过**（判据 = 双向可达 + 恢复可达，不是净方向），但细节变了：

* **c8 的末值从 512 回到 1024（= 初值）**。不是收缩变弱，而是**放大变强**：
  放大门限由 `110×0.9 = 99.0` 抬到 `120×0.92 = 110.4`，c8 也够到了放大分支
  （1 次放大 + 1 次收缩 ⇒ 净 0）。c8 的收缩计数两边都是 1。
* **`h` 从 0.1 降到 0.08 把带宽从 22.0 收窄到 19.2 ms**，格内双向振荡的格数 +1。

⇒ 这是"修 c4 的抛硬币"与"c8 的决断性 / 带宽"之间的**实测折中**。**本回合不回退**：
判定仍通过，且回退要再走一轮完整验证；但设计核心实验时必须带上——负载**连续变化**时
带宽变窄会让穿越更频繁（正是"结构性障碍"的直接后果）。

⚠️ 净方向**只是开环复放的注记**，不能当闭环预测（脚本自己的免责声明）。闭环里
chunk=1024 会让 `tpot_p90` 高于复放所喂的值（P1S3B：c4 不分块 88.7 / 分块 98.7–112.0），
收缩反而更容易触发。**"是否真的决断"只能靠真机重跑回答**——这正是本次要跑它的理由。

### 七、自查发现：守卫与标定工具的判据口径**不一致**（登记待办）

写完之后核对了一遍两处的判据，发现它们**不是同一个口径**：

* 标定工具（**权威**）：判 `max(headroom) − h` 的余量 ⇒ 直接对"放大分支还能不能
  可靠触发"负责；
* 生成器守卫（**粗代理**）：判"**中位数**到门限的距离 ≥ 5%·target" ⇒ 只读一个分位数，
  **看不见分布宽度**。

**不等价的证据就在本次**：守卫**放行**了 120/0.08，而 c4 的区间仍跨过门限
（上面第四节）。也就是说守卫在这种情况下会放行一个"未充分分离"的配置。

⇒ 定位：守卫只能当**回归下限（floor）**，用来挡"110/0.1 这类明显贴着的值复发"；
**不得**用它宣称"门限已充分分离"。两处口径应统一（或让守卫也读分布区间），
登记为待办。这一条已写进 `_GATE_MIN_SEPARATION_FRACTION` 的注释与
`_assert_gate_separation` 的 docstring，避免后来者误读。

> **[2026-09-24 晚 更正]** 上面这个处方——"两处口径**应统一**（或让守卫也读分布区间）"
> ——**已否决**。查清后：把守卫换成"最近点 + 同一个 5%·target margin"会把 **120/0.08
> 也拦下**（c8 收缩侧最近点 3.5 ms < 6.0 ms）⇒ **回溯否掉已入库的 `P1S3D`**，
> 而它的合格点是全格决断的。两处口径的正确关系是**分工**而非统一：守卫 = 生成期
> 静态下限（不解析产物），工具 = 跑之前用产物做的权威判定。
> 另外，本节把标定工具的口径写成"判 `max(headroom) − h`"——**那个量本身是坏的**：
> 它把**收缩点**的 headroom（很大的负数）也算进区间 ⇒ `min < h < max` 恒成立
> ⇒"跨过门槛"**恒报**，并与它自己的「带内」列**自相矛盾**（见第四节）。
> 工具已修：改为四态 `verdict`（必放大/必不放大/抛硬币/不可达）+ "离门限最近的**点**"，
> 并把「ttft 条件不满足（放大分支未被评估）」与「被 headroom 门槛拒」**拆成两列**。
> 详见下方同日第二条目与 `EXPERIMENTS.md` §4.3.4。

另：`p1s3d` **必须重跑**（门限改动会改变调度行为）——与 P3 相反，
`run.slo` 不参与决策所以只重算。这一点在 `EXPERIMENTS.md` §4.3.4 与本条都已写明。

## 2026-09-24 — P1S3D 真机重跑（36/36）：门限改动的目的达到了，但**归因比预期弱**，同时补上"输入新鲜度"这道缺了的口子

- 背景：§4.3.4 末尾留了一句"要回答'是否真的决断'，只能靠真机重跑"。本批就是那次重跑
  （`configs/mthreads/phase1_step3d/`，36 run，穿插执行）。
- 决策 / 发现：见下面四节。**最重要的是一条自我否证**——我原本准备把"c4 的跨重复问题
  已修好"写成结论，核对 `P2` 之后发现**站不住**（第三节）。
- 影响：`EXPERIMENTS.md` §1.4（登记）与 §3.3 **E1–E8**（结论）；D2 与新加的 E8 互指。
- 下一步：核心实验设计时处理"判据形式"障碍；若要真正断言 c4 可复现，需要**多于 3 次的重复**。
- 关联：`P1S3D-*`（36 份 `MEASURED`）、`EXP-base-13x`/`EXP-cp-13x`/`EXP-adap-13x`、§4.3.4、§3.3 E1/E8

### 一、落地流程按纪律走完，并且第一次把"输入新鲜度"也查了

拉回 → 检查器 → 噪声底 → 登记 → commit。取回这一步做了三重校验，全部通过：
远端整树摘要 `268ae7eb…` 与本地归一化后 `7f7e8942…` **同源**（直接比 md5 串会因为
GNU 与 Git Bash 的格式差——`hash  path` vs `hash *path`——**假报不一致**）；
文件字节和 9 464 185 两机相同；`du -sb` 差的 8272 字节经查是**远端 ext4 的目录项**
（本地 NTFS 目录计 0），不是内容差。另跑了一次跨机读数比对：同一份检查器在远端
python 3.10 与本机 3.13 **输出逐字相同**（`diff` 为空）。

**新发现的一道口子**：部署戳 `SOURCE_PROVENANCE.json` 的 `digest_scope` 只含
`src/ascend_opt` / `benchmark` / `scripts`，**不含 `configs/`**。也就是说
"代码是这个提交"**推不出**"配置是这一版"——而这正是之前那次事故的形状
（远端留旧模板 `slo.tpot_ms=50`、且没有新目录，批次照样跑、照样盖 `MEASURED`）。
本次逐 run 比对了快照与 `phase1_step3d/`：**36/36 逐字段一致**；
反向对照用 `phase1_step3c/` 返回 **1** 并列出 3 处差异 ⇒ 这条检查**能失败**。
顺手加了跨批次拒绝：`--tag` 是**子串**匹配，`--tag adaptive_c4` 会把
P1S3/P1S3B/P1S3D/P2 四批一起扫进来，一个配置目录不可能同时是它们的输入，
此时直接拒绝而不是印一堆假差异。

### 二、门限改动的目的达到了，而且作用面**很窄**

* 目的（静态判据）：放大分支要 `tpot_p90 < target·(1−h)`。旧门限 `110×0.9` = **99.0**，
  而工作点区间是 **97.8–99.2** ⇒ 跨过（`P1S3B` 的 r2 就是在 step 80 以
  `tpot_p90=99.23`、**差 0.23 ms** 被拒，爬坡卡在 1536，后 4 个请求 TTFT 652.1）。
  新门限 `120×0.92` = **110.4** ⇒ 同一区间全部通过，上界距门限 **11.2 ms**（余量 ≈10×）。
* 作用面：逐格比对两批，**c1/c2/c8 的决策计数、prefill 生效分布完全相同**
  （c1 `129/0/0/128`、c2 `65/10/0/64`、c8 `19/6/0/18`；分布分别为
  c2 `{1024:4, 2048:6}`、c8 `{1024:9, 512:14}`），信号差 ≤0.5 ms。
  ⇒ 这不是"整体重新调参"，把它读成整体改动是过度解读。
* 达标率（阈值 100）**首次携带信息**：c1/c2 = 1.000、c4 = **0.125**、c8 = **0.000**；
  按同一阈值重算 `P1S3B` 得到**逐格相同**的结果（全格同为 58/96 = 0.604）
  ⇒ 门限改动与达标率无关，达标率只由**分块**决定。附带一条新证据：
  **c4 上开分块把达标率从 8/8 打到 1/8**（TPOT p50 88.6 → 110.3 跨过 100）。

### 三、自我否证：**"c4 的跨重复问题已修好"这句话不能说**

写完上两节后，我准备把"`P4` 修好了 c4 的跨重复不可复现"写成头条。核对才发现：

| 批次 | 门限 | 计时修复 | c4 三次重复的 prefill `chunk_size` 分布 |
| --- | --- | --- | --- |
| `P1S3B` | 110/0.1（旧） | 无 | `{1024:8, 2048:4}` / `{1024:8, 2048:4}` / **`{1024:8, 1536:8}`** |
| `P2` | **110/0.1（旧）** | 有 | 三轮**全部** `{1024:8, 2048:4}` |
| `P1S3D` | 120/0.08（新） | 有 | 三轮**全部** `{1024:8, 2048:4}` |

**`P2` 用的是旧门限，却已经 3/3 收敛。** 合计分叉率 1/9。这既符合"门限贴工作点 ⇒
低概率触发"，也符合"`P2` 只是没抽中"——**n=3 分不开这两者**。

⇒ 能说的：*已知的那个拒绝机制在构造上不再可能*（第一节的算术）。
**不能说的**：*c4 的跨重复可复现性已建立*。**`§3.2` 的 D2 限制只是被削弱，未被解除**；
按 1/9 的经验频率，n=3 的"0 次分叉"给不出任何统计保证。

**连带更正**：`EXPERIMENTS.md` §1.3 注 2 原写"根因见 §4.3.2，**与计时修复无关**"。
`P2` 的 3/3 与 `P1S3B` 的 1/3 并列时，这个归因**在 n=3 下不可判定**，
已改写为"归因未建立"并在 §3.3 E8 记明。

这一条是本次最有价值的部分：**"机制上不该再发生"与"实测没再发生"是两个证据等级**，
前者是推演，后者在 n=3 下几乎不携带信息。差一点就把推演当成了观测。

### 四、形式性障碍没被 P4 触及（与预期一致，但要留下证据）

c4 的 `tpot_p90` 区间 [97.8, **571.2**] 里既有合格点也有收缩点（放大 30/32、收缩 2/32）。
⚠️ **本节原写"仍然跨过新门限"并引"带内 0/32"——那是标定工具一个被收缩点污染的
输出列的产物，与它自己的"带内 0"自相矛盾；已于 2026-09-24（晚）更正**（见下一节）。
闭环里出现了**周期性调整**：c4 首轮 **263 步**、**6 次决策**（step 8 收缩
`TPOT P90=380.3` → 32/56/80 连续放大 → **136 收缩 `TPOT P90=571.2`** → 160 放大，到此结束）。
⚠️ **原把 `264 收缩 → 288 放大 → 392/416 收缩、放大` 接在 c4 之后——那四步属 c2**
（c2 首轮 **515 步**、**10 次决策**：8 收缩 → 32/56/80 放大 → 136 收缩 → 160 放大 →
**264 收缩 → 288 放大 → 392 收缩 → 416 放大**）。
⚠️ **"周期约 128 step 的极限环"这一说法也已更正**：周期 = **请求补入批间隔**
（闭模型补位，批次数 = `num_requests / concurrency`），**由负载决定、不是控制器的固有周期**；
触发机制见 `2026-09-24（补）` 条目。

两条附带的口径提醒（都会让数字被误读）：

1. **`recover` 分支全程一次都没触发**（所有格、所有重复 `recover=0`）。
   "恢复路径可达"（开环复放的结论）≠"闭环里恢复真的发生过"。
2. **`n_decisions_changed` ≠ "取值变了"**：c8 的 6 次"已改"里有 **5 次 `value_unchanged`**
   （`chunk_size` 已触下界 512，实际动的是 `max_batched_tokens`）。
   **不得把它当"动作次数"引用。** 生效比例还很低：c2 10 次里 3 次落到 prefill、
   c4 6 次里 **1** 次、c8 6 次里 **1** 次。

### 五、顺手修掉检查器的一处"静默吞视图"

`inspect_step3_runs.py` 的每个视图各自 `return 0`，于是
`--controller --prefill-effect` **只输出第一个**、不报错。我自己就这么调用过一次，
差点把"prefill 生效情况"记成"无发现"——而实际上那里有 c4 只有 1/6 生效这类关键信息。
现已改为多视图逐个输出并先声明顺序；新增 18 项测试，并用 4 组突变（把 `return 1`
改回 `0`、把多视图改回静默 `return 0`、把取值比较改成 `pass`、给格名加默认值兜底）
**逐条确认它们真的会失败**——突变的还原用 md5 校验过（`bed9e05e…`，CR=0）

## 2026-09-24（晚）— 修 `calibrate_controller_gate.py` 的两处混淆：**那个"权威口径"其实从没被真正算出来过**

起因：处理登记在案的待办"统一两处门限判据口径"。先读工具实现，发现它
**根本没有在算一个判定**，只是打印两个数让读者自己看——**而这两个数是坏的**。

### 两处混淆（都不报错）

1. **`headroom min/max` 把收缩点也算进来。** headroom 只对"放大分支"有意义
   （`_decide` 里它只出现在放大判据里），但旧版对**每个**决策点都算。收缩点的 tpot
   远大于 target ⇒ headroom 是很大的负数（c4 的 571.7 在 target=120 下是 **−3.76**）
   ⇒ `min < h < max` **恒成立** ⇒ "门槛落在区间内部"**恒报**。
   实测自相矛盾：对 **120/0.08 的 c4**，同一行里「带内 **0**/96」（=合格点**全部**过门槛）
   与「跨过门槛」**同时成立**。也正因如此，"c4 仍跨过新门限"这句被写进了
   §4.3.4 与 E7——**那是这个坏列的产物，已更正**。
2. **`n_band` 是两个相反原因的和**：`ttft 条件不满足（放大分支根本没被评估）`
   ＋ `过了 ttft 条件但被 headroom 门槛拒`。于是 P1S3 的
   `target_ttft_ms=2000`（判据一次不触发）与 P1S3B 的"门限贴工作点"（抛硬币）
   **在表里长得一样**——而这两件事的结论相反。
   ⇒ 旁证：`test_headroom_span_is_reported`（旧）**正是用一个收缩点**去证明
   "区间跨门槛"，等于把这个缺陷当成判据。**测试会守住错误的口径**。

### 修法

`classify` 改为四类互斥计数（`n_shrink` / `n_grow` / `n_gate_rejected` / `n_not_eligible`）
＋四态 `verdict`（**必放大 / 必不放大 / 抛硬币 / 不可达**）＋"离门限最近的**点**"距离
（放大侧只统计合格点，收缩侧统计全部点）。`headroom` 只在合格点上统计；
`n_band` 保留为兼容字段并注明**已混淆、不得用于判据**。无合格点时 headroom 写 `n/a`
（**不写 0.0000**——0 是"测出来是 0"）。

### 修完立刻复现了已知事实（这是它现在对的证据）

| 候选 | 修正后读数 |
| --- | --- |
| **110 / 0.1（P1S3B）** | c2、c4 **抛硬币**；c4 离放大门限最近的**点相距 0.0 ms** |
| 110 / 0.08 | 全格决断，但 c4 最近点只有 **0.6 ms**（决断但极脆） |
| 120 / 0.1 | 全格决断，但 **c8 最近点 1.1 ms** |
| **120 / 0.08（P1S3D）** | **全格决断、抛硬币 0 格**，最近点 3.5–29.0 ms |

第一行与真机完全一致（P1S3B 三次 TTFT = 481.4 / 481.6 / **652.1**）；c4 的
"最近点 **0.0 ms**"就是"门限压在工作点上"的精确图像。**这是首次能从工具端复现那次故障**
（此前只有真机读数与手算的 0.0112 余量）。⇒ 结论：**"抛硬币"是工具**现在**能判定的东西，
而"决断性"与"离翻转多远"是**两个**量，必须一起读（110/0.08 决断但只有 0.6 ms）。

### 顺带查清"统一两处口径"为什么做不到

守卫（中位数距离 ≥ 5%·target）与工具（决断性 + 最近点）**互相都不包含对方**，
实测分歧方向相反：守卫**拦下**了工具判决断的 110/0.08（c4 中位数 98.7 距门限 101.2 只 2.5 ms），
却**放行**了工具已看到 c8 最近点仅 1.1 ms 的 120/0.1。
⇒ 若把守卫换成"最近点 + 同一个 margin"，**120/0.08 会被自己拦下**
（c8 收缩侧 3.5 ms < 6.0 ms）⇒ **回溯否掉已入库的 `P1S3D`**。
所以正确关系是**分工**而非统一：守卫 = **生成期静态下限**（不解析产物、几秒可跑），
工具 = **跑之前用产物做的权威判定**。**"统一"这个处方已否决**（更正 P4 条目第七节、
§4.3.4、README §6）。

### 验证

- 三条突变各对应一条新用例（收缩点混入 headroom / 未评估记成被拒 / 取消决断支），
  **全部如期变红**；还原用**字节级拷贝**（不走文本读写）＋ md5 与 CR 双重核对。
  单文件 **19 passed**。
- ⚠️ **驱动第一版用裸用例名当 pytest node id**（用例在类里，`node_id` 要带类名）⇒
  三个"基线"全报失败，**差点被读成"突变没咬住"**。改用 `-k` 才正确。
  ⇒ 「基线不绿」要先怀疑**自己的调用方式**，不是被测代码。
- ⚠️ **差点写下同引号嵌套 f-string**（本机 3.13 合法、**远端 3.10 是 SyntaxError**）——
  这份工具要在两边各跑一次做交叉验证，所以分数先算好再拼。已改为显式变量。

### 待办（登记）

1. 把"每格 `verdict` 不为抛硬币"接成一道**真正会失败的**前置门（现在只是读数，没人强制跑）；
2. 两处口径的**分工**要一直写明，不要试图消掉；
3. "判据形式"（**负载扫描**层面，不是"某一格跨过门限"）仍待核心实验设计时处理。

## 2026-09-24（补）— 更正 E7 的步号归属；把"极限环"的成因定到 **`TPOT` 口径混入 prefill 步**

起因：用户评审指 §3.3 E7 把 `264/288/392/416` 四个决策步接在了 **c4** 后面，
而 **c4 首轮只有 263 步**；并指出"出现反复调整"这个观察要**进一步区分
「请求补入周期」与「控制器自身振荡」**。两件事都从产物直接取证，**没有靠记忆**。

### 一、步号归属：用户是对的

从 `meta.json` 的 `controller_trace` 逐条读 `step`，并与 `telemetry.jsonl` 的行数核对：

| 格 | 总步数（`telemetry.jsonl` 行数） | 决策数 | 决策步号 |
| --- | --- | --- | --- |
| c1 | 1032 | **0** | —（双目标全程在带内） |
| c2 | **515** | **10** | 8↓ 32↑ 56↑ 80↑ 136↓ 160↑ **264↓ 288↑ 392↓ 416↑** |
| c4 | **263** | **6** | 8↓ 32↑ 56↑ 80↑ 136↓ 160↑（**到此结束**） |
| c8 | 150 | 6 | 8↓ 32↓ 56↓ 80↓ 104↓ 128↓（全部收缩） |

⇒ E7 里那句"c4 … → 264 收缩 → 288 放大 → 392/416 收缩、放大"**把 c2 的决策链接到了 c4**。
c4 的 `136 收缩（TPOT P90=571.2）→ 160 放大` 是对的，之后**没有**决策（run 在 step 252 就结束了）。
c2 的首轮与 c4 的前半段**结构相同**（8/32/56/80/136/160 逐个一致）——这正是容易接错的原因。

### 二、"全部同时到达"是错的，而且错得系统

§1.4 限制 4、§3.2 缺口 3、§3.3 缺口 2 都写着"每格 8 请求、**全部同时到达**"。
从 `requests.jsonl` 的 `arrival_ms` 看，**只有 c8 是**：

    c2: r00000/01 到达 0.28 ms；r00002/03 到达 11371.5/11975.5 ms；… 4 批
    c4: r00000–03 到达 0.28 ms；r00004–07 到达 14461.5…16469.9 ms；2 批

机制在 `src/ascend_opt/workload.py` 与 `benchmark/real_backend.py`：
负载是 `arrival: closed`（闭模型）——`_sample_arrivals` 对闭模型返回**全 0 的计划到达**
（注释写明"由执行器在补位时分配真实时刻"），后端在主循环里按
`in_flight() >= concurrency` 门限**补位**。
⇒ **补入批次数 = `num_requests / concurrency`**（c1/c2/c4/c8 = **8/4/2/1**），
3 次重复的 `arrival_ms` 逐个一致。**批间隔 ≈ 一个请求的 e2e**（c2 ≈11.37 s、
c1 ≈8.37 s），换算成步数 ≈ **129–132 个 decode 步**。

**这就是"周期约 128 step 的极限环"的真身**：周期不是控制器的固有量，
而是**负载的补入批间隔**（≈ `gen_len` 步 + 该批 prefill 的摊还）。

⚠️ 追源头：`meta.json` 的负载指纹里 `arrival: closed`，而**计划到达恒为 0**，
从指纹看就像"全部同时到达"。**要看实际到达必须读 `requests.jsonl` 的 `arrival_ms`。**

### 三、真正的触发机制：收缩是 `TPOT` 口径把 prefill 步读了进来

控制器读的 `tpot_p90_ms` 来自 `recent_tpot[-32:]`，而 `real_backend.py` 往
`recent_tpot` 里放的是**相邻两次 token 发出之间的间隔**（`recent_tpot.append(interval)`）。
⇒ **某一步做了 prefill，跨过该步的 token 间隔就包含这段 prefill 时间。**
观测窗 = 最近 **32 个** token 间隔，在 c2/c4 上 ≈ 决策点前 **8–16 步**。

实测（`telemetry.jsonl` 每步 `note` 给出 `prefill_tok` 与 `fwd_ms`）：

| 格 | 纯 decode 步 | 窗内无 prefill 时的 `tpot_p90` | 批边界窗（内含 prefill 步） | `tpot_p90` | 收缩门限 |
| --- | --- | --- | --- | --- | --- |
| c2 | `fwd_ms=77.2` | **81.5–82.7** | step 129/131（`fwd_ms=512.5`） | **137.7–139.0** | `120×1.08`=**129.6** |
| c4 | `fwd_ms=77.3` | **97.8–99.4** | step 129/131/133/135 | **570.8–571.2** | 同上 |

**纯 decode 的 81.8 在门限下、含 prefill 的 138.0 在门限上。** 逐条核验
（12 个 adaptive 运行 × 3 次重复）：**每条由 `tpot` 分支触发的收缩，观测窗内都含 prefill 步；
每条由 `ttft` 分支触发的放大，窗内都没有 prefill。**
**唯一例外**是 c8 的 step 32/56/80/104/128——窗内无 prefill，`tpot_p90` ≈140，
那是 8 并发下**真的**稳态违约。

⇒ 这条把长期登记的"控制器反馈口径待统一（token 级 ITL ≠ 请求级 TPOT）"
从**疑点**变成了**已测定的成因**：控制器把"prefill 挤占"读成了"decode 变慢"。
而它的响应是**调小 `chunk_size`**（prefill 更碎 ⇒ 含 prefill 的步更多）。
⚠️ 这个"反向放大"目前**只是读数的推论，本批没做干预实验**。

### 四、所以"反复调整"要按成因拆成三件

1. **补入驱动的周期性跟随**（c2 的 4 批、c4 的 2 批）：批一到，prefill 突发进窗 ⇒
   **该批后第一个观测点收缩一次**（c2 批起点 step 0/129/257/385，收缩在 8/136/264/392
   ⇒ 偏移 +8/+7/+7/+7，正好 1 个 `control_interval_steps`）；突发过后 `TPOT` 回落 ⇒
   按最小节拍（3 周期 × 8 步 = 24 步）连续放大 3 次回上限。**周期由负载定。**
2. **稳态违约下的单向棘轮**（c8）：只有 1 批，`TPOT P90` 持续 ≈140 > 门限 ⇒
   每 24 步再收缩一次，共 6 次、**0 次放大**，值单调 7168→2048，**没有往复**。
3. **零动作**（c1）：双目标全程在带内，0 决策。

⇒ 本批能支持的是"**周期性调整跟随补入批**"。
**不能**说"控制器自激振荡"（无补入的 c1 零动作、c8 单向）；
要检验自激需要**负载水平不变**的稳态段——而那正是本批缺的（补位是"始终填满到 `concurrency`"，
没有低压→高压→恢复的轨迹）。

### 五、方法学教训（三条，都会静默）

1. **"负载形状"必须回到 `requests.jsonl` 的 `arrival_ms` 核实。**
   闭模型里计划到达恒为 0，`meta.json` 的负载指纹看起来就是"同时到达"——
   三处文档照着这个假象写了同一句错话，**没有任何检查会报错**。
2. **步号是 run 内的局部量，跨格引用必须带格名。** E7 那句错了整整一批都没被发现，
   因为"263 步"与"264 步"只差 1，而表里只有一列"步号"、没有"哪一格的步号"。
   ⇒ 已在检查器里加 `--cycles` 视图：**逐 run 打印总步数、补入批次与批起点步号、决策步号与方向**。
3. **看到"周期"先问"周期由谁定"。** 128 这个数**同时**是 `gen_len`、
   也是批间隔在步数上的长度——把负载的量当成了控制器的量，就会得出"控制器在自激"这个错误结论。

### 六、验证

- `EXPERIMENTS.md` §3.3 表列数扫描：11 行全为 4 个管道符（3 列）一致；E9 已登记。
- 全量离线测试通过（见提交信息）；改动文件 CR 字节数为 0。

### 待办（登记）

1. **控制器信号口径**：`tpot_p90` 需排除/标记含 prefill 的步（或改用"纯 decode 窗"），
   再做一次前后对照；这是"判据形式"之外的另一类障碍。
2. **补一次负载水平变化的实验**（低压→高压→恢复），才能把"跟随补入批"与"稳态下是否自激"分开。
3. E9 提出的"调小 `chunk_size` 反而加重窗口污染"需要**干预实验**验证。

---

## 2026-09-24（夜）— 三件事：矩阵的"跑完但没干完活"计成了成功、标定工具三处口径、验收产物接上可核验的溯源戳

按用户评审的优先序"**先补验收和失败退出**"推进。四条都已落地。

> 📌 **术语更正（本条目起生效）**：标定工具的判定名由「**抛硬币**」改为
> 「**样本跨越门限**」。**本日志此前的条目里出现的「抛硬币」一律读作「样本跨越门限」**
> ——日志是时间序记录，不回改历史条目；改名理由见第三节。
> 另外：把"每格 `verdict` 不为抛硬币"接成**会失败的前置门**这个旧待办**已被裁定作废**，
> 见第三节末尾。

### 一、矩阵把"跑完但没干完活"计成了成功（`af134e7`）

`run_step3_matrix.py` 只要 `run_once` **正常返回**就 `done += 1`。而真实后端会把单请求的
异常**接住**、记进 outcome、**正常返回** —— 离线注入 OOM 后产物是 `0 成功 / 1 失败 / 7 未完成`，
矩阵却报"成功"。

**为什么这类失效必须优先修**：它把"活没干完"渲染成"跑完了"，而**退出码是 0**。
下游（聚合、登记、结论）看不到任何异常。

修法：跑完 → **保留产物** → 读落盘产物里的运行状态 → 不 ok 计失败，并按 `--keep-going`
决定续跑。状态读取是三态（ok / 明确不良 / **不可判定**），取不到填 `None` 而**不填 0**
（0 是"测出来是 0"）。`runner` / `status_reader` 可注入，便于用**用户实测的那个形状**
（0/1/7）直接断言。

**核对既有结论**：逐格核对 P1S3D 36/36（`n_success==n_requests`、`dropped==0`）、
P1S3/P1S3B/P2 共 120 格 —— **全部干净**。⇒ 这个漏洞**没有**影响已入库的结论；
但扩大实验前必须先修好（本轮已修）。

### 二、验收判据的两处漏检（`958252d`）与它自身的两处缺陷（`c7783b7`）

| 类 | 问题 | 后果 |
|---|---|---|
| 漏检 | `V4b` 只验**首个**分岔步 | 首步若是近平局（可解释），后面某步 gap 很大（与"分布等价"直接矛盾）被**完全掩盖** |
| 漏检 | `V3`"补位检查"拆成了多个**单请求** | **没有真正覆盖不同历史长度的合批 decode** —— 而 padding 只在那时发生 |
| 自身缺陷 | V4 槽位判据在 `retire` **之前**读 `live_slots()` | 恒读到 1 ⇒ "收尾应为 0"**恒假** —— 却**看起来像真的抓到了 KV 泄漏** |
| 自身缺陷 | `V3` 逐个分岔步判态 | 贪心自喂下第 k 步分岔后历史即不同，后续分岔是同一条因果链的下游 ⇒ 一次平局被放大成 **10 条"需复核"** |

**修法（按"这个判据在什么口径下才成立"来分）**：

- "能不能逐个分岔步判态"取决于**比较口径**，抽成纯函数
  `judged_divergence_steps(..., shared_history)`：**共享参考历史**（teacher forcing）⇒
  逐步都判（每步输入都来自参考，都是独立验收点）；**贪心自喂** ⇒ 只判首个。
  **全部分岔步仍记入产物**（`divergence_steps`）供核对，并把"重跑参考到分岔步的 gap"
  接成取证交叉核对。
- `V3` 新增 `mixed_history_batched_decode`：先让各行 KV 长度**参差**，再合批 decode
  逐行按内容比分布；**各行 KV 长度相同、或没有在同一次 forward 里一起算 ⇒ 进 `errors`**
  （"没合批"不得静默当成"通过"）。
- 槽位判据移到 `retire` **之后**读，并把"retire 之前应为 1"记成**反向对照**。

**⚠️ 判据改了就**不能继承旧产物的 `passed`** —— 见第四节。

### 三、标定工具：读全部重复 + 跨重复对照 + 改名 + 口径核对（`73dc2d7`）

用户裁定的两句话是本节的纲：**"读取全部重复……比较「相同条件下不同重复的决策分歧」"**；
**"当前分类先改称「样本跨越门限」"**。

**（1）为什么改名**：旧名「抛硬币」说的是"**一条合并后的样本序列上**门限不决断"
（一部分点过、一部分不过），而它读起来等于"**跨重复不可复现**"。同一次运行内负载
本来就可能在工作点之间移动 ⇒ 两种结果并存完全正常。**这是一次未被证实的跳步。**
⇒ 改名「样本跨越门限」，并**另设** `cross_repeat_verdict` 专门答可复现性（三态，
**"单一重复"必须显形**：一次重复无从比较，不得据此断言可复现）。

**（2）实测坐实这个区分**（P1S3B + P2，110/0.1）：同一格 c4 的 `门槛拒` 在**同一条**件内
是 **6 / 9 / 15**（P2：1 / 6 / 5）——波动很大；而 `chunk 范围(变次)` 三次**逐字相同**
（`512..2048 (6)`），瓶颈串也逐字相同。⇒ **"门槛拒计数"不携带可复现性信息**，
把它的波动读成"不可复现"正是旧口径的错误。

**（3）修一处静默的混淆：批次没进条件**。`per_cell` 原先只按**格名**分组 ⇒ P1S3B 的 `r0`
与 P2 的 `r0`（**不同条件**，P2 含 `readback_ms` 计时修复）被并成"同一条件的两次重复"
——跨重复表里 c4 出现**两行 `r0`**，静态分类还不同（24/6 vs 29/1）而 `chunk 范围` 与瓶颈串
**逐字相同**。看起来像"同条件重复却判定不同"，实际是**两个条件被混成一组**。
⇒ 格的身份改为 **`(批次, 格名)`**；新增构造级测试：让两批判定相反，断言输出里出现
两条各自的「重复间一致」而**没有**「重复间分歧」（一旦合并就会造出一个假的分歧）。

**（4）一次"差点误判工具多了一个门"的经历（值得记）**：静态分类把 c4 报出若干「门槛拒」，
而轨迹里只有 `ttft` —— 看起来像工具多了一个实现里没有的门。**读 `_decide` 才发现**：
`ttft` 分支里被 headroom 门槛拒时（`tpot_headroom <= hysteresis`）返回的 `bottleneck`
**仍是 `"ttft"`**。⇒ 轨迹里的 `ttft` 计数 = **放大 + 被门槛拒**，工具**没有**多门，
是**读法错了**。实测坐实：P1S3D 的 `c8` 一轮 `放大=0 / 门槛拒=1`，轨迹正好 `ttft×1`。

**（5）把"一一对应"这句断言变成不变量**：新增 `check_branch_consistency` +
`report_branch_consistency`，每次运行自检四条等式（`n_shrink==#tpot`、
`n_grow+门槛拒==#ttft`、分类四类之和 == 有信号的决策点数、轨迹各分支之和 == 轨迹条数）
并把"无信号条目"的差额解释掉（`load_signal_sequence` 跳过没有有效样本的条目，例如 step 0，
而它仍记一条 `none`）。**用每个 run 自己 `meta.config.controller` 里的阈值核对**——
拿 `--targets` 去对会**恒失败**（两边本来就**应该**不同），守卫会立刻退化成噪声。
漂移 ⇒ **退出码 6**；读不到阈值 ⇒ 记「不可判定」而**不是**通过。

**（6）`chunk_range` 只是 min..max，不是轨迹**：两轮可以 range 相同而在不同**时刻**走到
不同的值——这正是同一 range 下 TTFT 481 与 652 并存的成因。新增 `chunk_changes` /
`chunk_final`。实测 c4 选了 **30 次 `ttft` 分支而 chunk 只变了 6 次**（其余被冷却/边界拦住）
⇒ 又一次印证"**选中分支 ≠ 真的改了值**"。

**（7）旧待办作废**：用户明确 **"标定工具不能直接作为可复现性的阻断门"**。它区分不了
"样本内不决断"与"跨重复不一致"。若将来要接成会失败的门，判据必须是「**重复间一致**」。

### 四、验收产物接上溯源戳 —— 并据此发现"戳**没被兑现**"

**事情的形状**：记录里写着"溯源字段已完成"，但两份产物**都没有** `source_provenance`。
（**别用"记录里说完成了"推断"产物里有"** —— 这是本轮最值得记的一条。）

**查根因**（三条证据拼起来）：

1. 远端 `SOURCE_PROVENANCE.json` 记的是 **`af134e7`**，时间 `04:44`；
2. `af134e7` **早于**加戳的 `49b7759`（`af134e7` 只改了矩阵脚本）⇒ 那一刻的
   `scripts/validate_fp16_path.py` 有修好的判据、但**没有**写戳的那段代码，与产物缺 `source_provenance` 一致；
3. 而远端 `scripts/` 里的那份**后来被单独更新过**（现在含 2 处 `source_provenance`）
   ⇒ **戳与树已经不一致，戳不再能证明产物出自哪棵树**。

**一个现成的错法入口**：远端 `/root/ascend_probe/` **根下**还有一份**同名旧副本**
`validate_fp16_path.py`（无 provenance、md5 `f7b8c458…`，与 `scripts/` 那份 `6056de08…` 不同）。
已移入 `/root/ascend_probe/_stale/` —— 留着它，"跑错脚本"随时可以重演。

**修法（不靠推断，靠重跑）**：把当前**干净**的提交 `73dc2d7` 重推到远端（戳 `dirty=false`，
远端**重算树摘要相符** —— 这一步是戳有效的前提），重跑 0.5B + 7B，拉回后**逐文件 md5 比对**。

**新产物**（`passed=true`、`needs_review=[]`、CR=0）：

| 模型 | 文件 | md5 | 戳 |
|---|---|---|---|
| 0.5B | `results/raw/phase1_validation_b/validate_0.5b.json` | `580fcf66…` | `commit=73dc2d7` / `dirty=false` / `validated=true` |
| 7B | `results/raw/phase1_validation_b/validate_7b.json` | `bc33a499…` | 同上 |

⇒ **"这是哪一版判据跑出来的"现在能从产物自己回答**，不必再靠外部记录。

**顺带确证了判据版本**（而不是只看 `passed`）：旧产物里**确实**含新判据的证据键
（`mixed_history_batched_decode` / `grouped_prefill` / `resource_lifecycle`）⇒
**判据版本没跑错，缺的只是戳**。这个区分要紧：**"缺的值"与"错的值"是两回事**，
而后者更危险。判据仍带缺陷那版的首跑（`passed=false`、`needs_review=10`）留在
`results/raw/_pilot/phase1_validation_precheck/` 作证据，不进聚合。

**第一代产物降级**：`results/raw/phase1_validation/` 保留但**降级为"判据更弱的一代"**
（没有 V4b 多步、没有变长合批 decode、槽位判据还没有正反对照）——**不得**据其宣称
"完整路径已验收"。

### 五、验证

- 全量离线回归 **785 passed / 464 subtests**（本轮 +25；改动文件 CR 字节数均为 0）。
- 标定工具实跑：P1S3D `120/0.08` ⇒ 12/12 run 口径核对 ✓、4 格 3 重复全「重复间一致」；
  P1S3B,P2 `110/0.1` ⇒ 批次分组生效、口径核对 ✓。
- 验收实跑（远端 MTT S4000）：0.5B / 7B `RC=0`，`passed=true`；关键取证
  `mixed_history` `rows_per_call=[4,4,4]`、`kv_lens=[16,25,35,51]`（真合批真补位）、
  `V4b` 16 步逐步 `max|Δ|` 0.0223–0.0391（容差 0.25）、
  `v4_drivers_live_before_retire` 全 1 与 `live_at_end` 全 0（正反两侧都在）。

### 待办（登记）

1. **控制器信号口径**：`tpot_p90` 需排除/标记含 prefill 的步（或改用"纯 decode 窗"），
   再做一次前后对照（承上一条目）。
2. **补一次负载水平变化的实验**（低压→高压→恢复）。
3. E9 提出的"调小 `chunk_size` 反而加重窗口污染"需要**干预实验**验证。
4. **给矩阵加"路径不带 tag 的开关"**，以便离线复现"注入 OOM"那条判据
   （现在已验证一次，但没有可复现的夹具）。
5. 若把标定工具接成前置门：判据用 **「重复间一致」**，**不得**用
   `verdict != 样本跨越门限`（用户裁定）。

---

## 2026-09-24（续）— 新增到达模型 `piecewise`：分段常量速率 λ(t)，把"负载水平的轨迹"变成可表达的自变量

### 一、为什么要加（不是锦上添花，是缺了就没法回答研究问题）

研究问题是「负载**变化**时自适应控制器能否优于固定静态配置」。现有三种到达模型
**都表达不出"负载水平的轨迹"**：

- `closed` 全程把在途数填满到 `concurrency`（P1S3D 已确认：实际是
  `num_requests/concurrency` 次"填满→预填充→排空"的**重复瞬变**，水平始终不变）；
- `poisson` 是**恒定**速率；
- `burst` 只有瞬时尖峰，且段间间隔是确定性的。

⇒ 核心问题此前**没有自变量可用**。这是待办第 2 条的前置基础设施。

### 二、设计（三条刻意的取舍）

1. **用"速率 × 时长"而不是"并发数"**。并发是**结果**（到达速率与服务速率共同决定），
   拿它当自变量会让它与调度策略自我耦合——控制器一改批预算，并发就跟着变，
   "负载水平"就不再是同一个东西。
2. **段边界按墙钟对齐**（第 i 段占 `[Σ_{j<i} d_j, Σ_{j≤i} d_j)`）。否则"何时升压/降压"
   变成随机量，跨 run 无法对照。
3. **总请求数由段定义决定，不接受第二个真值**：`num_requests != Σ(duration_s × qps)`
   即报错并给出可照抄的修法。两个都能改就会出现"负载被截断/延长而没人发现"。
   约束放在 `__post_init__` 而不只在 `from_dict`——避免"配置路径报错、代码路径静默"。

**段内分布 `rate_noise` 是方法论选择，不是实现细节**：
- `jittered`（默认）：每段计数**精确**等于名义值，位置在等分格内抖动（分层抽样）。
  刻意**不**用严格等间隔——P1S3D E7 的教训是决策会与批起点**相位锁定**。
- `poisson`：只固定总数，位置按条件泊松（给定 N 时 i.i.d. ~ λ(t)/Λ(T)）⇒
  各段实际计数按多项分布波动。

两者都保证总数恰为 `num_requests`，**不会**出现"生成数对不上请求数"的静默截断
（`build_requests` 里那条不等即抛错的检查就是为此）。

### 三、落点

- `src/ascend_opt/workload.py`：`RateSegment` / `ArrivalModel` 加 `piecewise` /
  `rate_segments` / `rate_noise` / `__post_init__` 一致性约束 / `_sample_piecewise` /
  `summarize_workload` 新增 `arrival_segments`（**名义与实际同时给**）。
- `benchmark/workloads.py`：注册 `load_ramp`（低压→高压→恢复）；`_to_payload` 只在
  piecewise 下带新字段，避免污染配置快照的逐字节比对。
- `README.md` §3：四种到达模型的对照表 + 两条口径警告。
- `tests/test_workload.py`：新增 `TestPiecewiseArrival` 13 项。

### 四、验证

- 全量离线回归 **792 tests / 8 errors / 25 skipped**；8 个 error 全部是
  `test_validate_fp16_path.py` 里 **`torch` 未安装**（`'NoneType' object has no attribute
  'tensor'`），改动**前**的基线同样是 8 ⇒ **不是本轮引入**。
- 端到端 dry-run 冒烟（`load_ramp`，`--dry-run`，`TEST_DATA`）：83 请求全部成功，
  指纹 `arrival_segments` 逐段报出 `low 名义 0.08/实际 0.0833（10 个）`、
  `high 名义 0.70/实际 0.7000（63 个）`、`recover 0.08/0.0833（10 个）`，
  墙钟 325 s。⇒ 开放环执行路径（到达 → 受理 → 排队）在新模型下跑通。

### 五、`load_ramp` 的速率取值依据与**未验证**之处（`MEASURED` 之外，慎引）

取值不是猜的，来源是两条实测：

- 成本探针：batch=8 的 decode 步 = **109.17 ms** ⇒ 128 步 = 14.0 s ⇒ 满批服务速率
  ≈ **0.57 req/s**；
- P1S3D c8 实测墙钟 20.4 s / 8 请求 = **0.391 req/s**（含 prefill 与起停）。

⇒ 取 **C ≈ 0.5 req/s**；`low = 0.08`（≈0.16C，批 1~2，TPOT ~60~75 ms，达标）、
`high = 0.70`（≈1.4C，队积压、批打满 8，TPOT ~110~140 ms，超标）。

⚠️ **三个前提 / 未验证项**（违反则该负载测不到想测的东西）：

1. **`scheduler.max_num_seqs` 必须为 8**——容量 C 是"批上限 8"下测出来的，
   批上限一变，λ 相对容量的含义就变了（现有 `phase1_step3d` 配置是 32，不改就不能用）。
2. **速率尚未经真机 pilot 验证**：若 pilot 显示 high 段队深不涨（没过载）或积压过深
   （尾部 TTFT 爆炸），按实测调 qps 后重跑，并把改动作为**独立变更**登记。
3. **报告必须逐段读**，不能拿整 run 汇总值当"高压下的表现"。
   ⚠️ dry-run 验不了过载（模拟器 TPOT 低一个数量级）——与 P3 的教训同形。

### 六、由此产生的新待办

- 真机 pilot：`load_ramp` 跑 1 次（静态臂），判据 = high 段**队深确实在涨**
  且 recover 段**回落**；据此定稿速率。
- 静态基线补齐 512/1024/1536/2048（现有只扫 1024 一个点）。
- 成本模型控制器：已定与阈值控制器**并列为第三个臂**（用户裁定），对照 =
  静态 / 阈值自适应 / 成本模型自适应。

---

## 2026-09-24（夜）— 换镜像后重建远端环境 + 真机正确性验收（0.5B）通过；并更正一处存活了 24 小时的错误归因

### 一、起点：新镜像只带 base 栈，推理环境与模型缓存都没了

用户给了新的连接信息（`ssh -p 14293`，**端口从 14671 变了**——这条以前只是
"重启后端口会变"的**推断**，现在有实例为证）。实测盘点（逐项读原始输出）：

| 项 | 新镜像实测 | vs 2026-09-22 |
| --- | --- | --- |
| 容器 | `autodl-container-73d6479980-71280c16` | **变了** |
| 驱动 / `mthreads-gmi` | 2.7.0 / 1.14.0 | 相同 |
| MUSA | **3.1.0**（`musa_toolkits` = `musa_runtime` = 3.1.0） | 相同 |
| 设备 | MTT S4000 ×1 / 49152 MiB / 0% util / 41 C / 无其他 GPU 进程 | 相同 |
| Python | 3.10.8 @ `/root/miniconda3` | 相同 |
| `torch` / `torch_musa` | **2.2.0**（`2.2.0a0+git8ac9b20`）/ **1.3.0+81caf0a** | 相同 |
| 磁盘 | `/` overlay 30 G（用 53 M）；`/root/autodl-tmp` 50 G（**空**） | — |
| 远端出网 | `hf-mirror.com` → HTTP 200（1.9 s） | 可用 |

**三样丢了**：`/root/musa-envs/qwen310`（故"必须先 `source activate_musa.sh`"当时无法执行）、
`transformers`（旧 4.46.3）、模型缓存 `/root/autodl-tmp/hf/hub`。
⇒ **base 栈与原记录一致 ⇒ 不需要也不允许重装/升级驱动与 torch**，只把上层补回来。

### 二、连通性诊断：**根因未分离**（如实标注）

关掉本机代理后目标端口立刻回 banner（`SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.11`，1.6 s），
而此前 8 分钟一直是"静默超时"。⚠️ **但这一分钟内两个变量同时变了**：① 代理被关闭；
② 实例可能刚好完成引导。**没做"只改一个"的对照 ⇒ 不能断言根因是代理。**
可复用的一条（不依赖根因判定）：**代理开着时对国内主机的可达性判断一律按"可疑"处理，
连不上先关代理复测**。判据与两种对照见 `PLAYBOOK.md` §5.1.1 / §5.1.2。

### 三、重建环境：写成脚本，并**逐字照抄**既有 `activate_musa.sh`

新增 `scripts/setup_remote_musa_env.sh`（幂等）。**做法上有一处值得记**：

> 第一版我按文档**描述**合成了一份 `activate_musa.sh`（LD_PRELOAD / LD_LIBRARY_PATH /
> HF_ENDPOINT / HF_HOME / PATH）。跑通之后才在本机暂存目录
> `C:\tmp\remote_scripts\` 里发现**原文**（2202 B），一比有实质差异：
> `MODELS_DIR` 没设、`LD_LIBRARY_PATH` 的拼法会多出尾冒号、少了三行 echo。
> ⇒ **"按文档描述恢复" ≠ "恢复成原来那份"。有原文就照抄原文**，并把这一段原文
> 内联进脚本（`cat > "$ACT" <<'EOF'`），使"恢复"变成可重复的动作而不是转述。
> 改完取回远端文件与本机原文 `diff` ⇒ **逐字相同（2202 B，CR=0）**。

**脚本不做**的事也写进注释：不动 base 环境、不装/升级 torch 与驱动、不下模型。
**并且复核了 `LD_PRELOAD` 的前提而不是盲抄**：
`strings /root/miniconda3/lib/libstdc++.so.6 | grep -c GLIBCXX_3.4.30` = **0**
⇒ conda 那份仍缺 `GLIBCXX_3.4.30` ⇒ 这个 workaround 仍然必要。

依赖版本**照抄** 2026-09-22 记录（不"顺手升级"）：`transformers 4.46.3` /
`tokenizers 0.20.3` / `accelerate 1.1.1` / `safetensors 0.4.5`。装完实测：

```
python 3.10.8 @ /root/musa-envs/qwen310/bin/python
tokenizers 0.20.3 → torch 2.2.0 → torch_musa 1.3.0+81caf0a → transformers 4.46.3
torch.musa=True  musa 可用=True  设备数=1
设备上做了一次 64x64 fp16 矩阵乘 ✓
实际载入 libstdc++ = /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30（LD_PRELOAD 生效）
```

⚠️ 验证**故意按 `tokenizers → torch` 的顺序导入**——那个坑只在"tokenizers 先载入"时出现，
按 `torch` 先导的顺序测是**测不出**它的。

### 四、部署与溯源：源码戳有效，且补上了"戳不覆盖 configs"的缺口

先提交本会话工作（`2bdf760`，含第 01/02 步），再 `push`：

```
push 传输完成：1269760 bytes, 98 个文件 -> /root/ascend_probe
部署戳：commit=2bdf7607c784  dirty=False  digest=edac17061b20…
投放并校验通过：远端以 python3 3.10.8 重算摘要相符
```

⚠️ **`configs/` 不在摘要作用域里**（PLAYBOOK §5 已记）：源码推完、戳通过，
**不代表远端配置是新的**。本批**另外做了 md5 逐文件比对**：
`configs/mthreads/s4000_fp16.yaml` + `phase1_step3d/` 12 个 —— **13/13 全部一致**。

### 五、真机正确性验收：0.5B 在新镜像上通过

`python -u scripts/validate_fp16_path.py --model Qwen/Qwen2.5-0.5B-Instruct --device musa:0 --gen-len 16`
（模型先预下载到 `HF_HOME`，954 MB，revision `7ae557604adf67be50417f59c2c2f167def9a775`）。

| 项 | 结果 |
| --- | --- |
| `passed` / `needs_review` | **True / []** |
| `failures` | `[]` |
| 阶段 | **6/6**（V1 单请求 / V2 多步 KV decode / V3 变长 batch / V4 分块 prefill / V4b 分块后多步 decode / 资源生命周期） |
| V1 | 16 个 token id **逐位一致**；阈值标定 `sensitive=True`（改一个 token 的分布距离 9.05 ≫ 阈值 0.001） |
| V3 | 4 行变长（16/24/33/48）**逐行一致**；`grouped_prefill` 每行 `max_abs_diff=0.0`；`mixed_history_batched_decode` 的 `rows_per_call=[4,4,4]`、`batch_calls=[10,11,12]`、`kv_len@first_batch_step=16/25/35/51` ⇒ **真合批、真补位**；该段 `max_abs_diff` ≤ 0.03125（容差 0.25） |
| V4b | 16 步逐步比对，**最大 `max_abs_diff` = 0.125**（最差第 12 步，容差 0.25）；`n_divergence_steps=0`、`any_blocking_divergence=False` |
| V4 其它切分 | `[20,20]` / `[39,1]` / `[1]x40` 全部等价 |
| 资源生命周期 | `live_before_retire` 全 1、`live_at_end` 全 0 ⇒ **正反两侧都在**（恒 0 说明计数器坏了） |
| 溯源戳 | `git_commit=2bdf7607c7843c977491c50fe239a7b9ac7f9c14`、`git_dirty=False`、`validated=True`、`reason=部署戳已校验（摘要相符）` |
| 产物 | `results/raw/phase1_validation_c/validate_0.5b.json`（17484 B，md5 `1254d13aafe7eedb6417ecb39eb61290`，**远端与本机 md5 相同**） |

⚠️ 这是**新一代目录**（`phase1_validation_c`），与 `_b` 的区别是**环境变了**（新镜像），
不是判据变了。按第 07 步验收："若环境改变，建立新实验条件"——**旧批次的标定不自动适用**。

⚠️ **本次不是"生成冒烟"**：跑的是项目自带的**数值等价性验收**（比一次生成强：
它逐位比对 token id、逐步比对分布、并检查资源生命周期）。生成冒烟留待需要时另做。

### 六、环境指纹已重采

`cd /root/ascend_probe && PYTHON=python3 bash scripts/inspect_env.sh` ⇒

* `platform=mthreads`、`available=True`、`device.model=MTT S4000`、`count=1`
* `probes.mthreads.result=true`（`npu-smi` 不存在 ⇒ `ascend.result=false`，与次平台定位一致）
* `MTT_DRIVER_VERSION=2.7.0`（源 `mthreads-gmi:version-line`）、
  `MUSA_VERSION=3.1.0`（源 `file:/usr/local/musa/version.json`）、
  `TORCH_VERSION=2.2.0`、`TORCH_MUSA_VERSION=1.3.0+81caf0a`
* `missing` 9 项 = 昇腾侧 5 项 + 4 项 `MODEL_*`（需在远端显式声明环境变量）
* ⚠️ `software.transformers_version = null`：指纹用的是 **base 的 `python3`**，
  而 transformers 装在**venv** 里 ⇒ 这是"用哪个解释器采指纹"的后果，与原记录一致（非缺陷，但要知情）

指纹已取回 `results/raw/_env/`（该目录被 `.gitignore` 排除 ⇒ **关于主机的持久记录仍在
本文件与 `README.md` §9**）。

### 七、⚠️ 新发现的测量风险：所在主机接近满载

实测 **load average = 13.85 / 14.77 / 14.40，而 `nproc` = 15**（验收期间复测 13.81 / 14.36 / 14.42，
量级稳定）。`/proc/loadavg` 的第三段还显示 `7/6080` 与 `12228` 这类量级的线程总数
⇒ 强烈提示这是**主机视角**而非容器视角。

为什么这条在本项目里特别要紧：成本探针已测定"**主机提交成本与设备工作量同量级**"
（`COST_PROFILE_S4000.md`），⇒ **CPU 争抢会污染计时**。
⚠️ **未核验该 loadavg 的视角（容器 or 主机，即是否 lxcfs）**，故**不得**当作本容器的负载。
⇒ **动作**：任何 `MEASURED` 批次开始前先记录 loadavg；若在波动，**不得**开始正式计时。

### 八、验证

* 环境重建脚本幂等复跑：`RC=0`；`activate_musa.sh` 与既有原文 **diff 逐字相同**（2202 B / CR=0）。
* 依赖导入（含会踩坑的 `tokenizers → torch` 顺序）通过；设备上完成一次 fp16 矩阵乘。
* 部署戳：远端重算摘要**相符**；配置 13/13 md5 一致。
* 真机验收：`passed=True` / `needs_review=[]` / 6 阶段全过；产物 md5 两端一致。
* 跑完复查：**无遗留 GPU 进程**（`No running processes found`）。
* 本机离线回归不受影响（本步只加脚本与文档；`scripts/setup_remote_musa_env.sh` 已在提交内）。

### 九、待办（登记）

1. ~~**7B 模型未下载**~~ ✅ **当晚补完**：15 GB 落到 `/root/autodl-tmp/hf`（50 G 卷用到 31%）。
2. ~~**7B 的数值验收未重跑**~~ ✅ **当晚补完**，见 §十。**结论：新镜像复现了与上一代完全相同的数值行为。**
3. **loadavg 视角未核验**；正式计时前必须有稳定性对照（见 §七）。
4. **容器名/端口会变**：`PLAYBOOK.md` §5 的远端工作目录与端口说明需要在下一次换机时同步
   （本次已记：端口 14671 → 14293，容器 `…73d6479980-71280c16`）。
5. 第 03–06 步（本地准备）仍未做，**正式 pilot 依赖它们**。

### 十、7B 侧验收（补完）与**跨镜像一致性**

先重推源码并投放新戳（`commit=654914ee40bf` / `dirty=False` / 远端重算摘要相符），再跑 7B。

| 项 | 值 |
| --- | --- |
| 产物 | `results/raw/phase1_validation_c/validate_7b.json`（19014 B，md5 `dc38bbefab4392a450b0113ce54d0dac`，**远端与本机相同**） |
| 判定 | `passed=True` / `needs_review=[]` / `failures=[]` / **6/6 阶段** |
| V1 / V2 | token 逐位一致；V1 阈值敏感（改一个 token 的分布距离 **21.72** ≫ 阈值 0.001） |
| V3 | 变长 16/24/33/48；`grouped_prefill_preserves_rows=True`；合批 `rows_per_call=[4,4,4]`、`kv_len@首批=16/25/35/51` ⇒ **真合批真补位** |
| V4 / V4b | 分块与整段首 token 相同（13/13）；V4b 16 步最大 `max_abs_diff=**0.125**`（容差 0.25）、`any_blocking_divergence=False` |
| 生命周期 | `live_before_retire` 全 1、`live_at_end` 全 0 |

**⭐ 本步最有价值的一条**：与上一代 `results/raw/phase1_validation_b/validate_7b.json` 逐项对照 ⇒
`per_row_match` 完全相同、`logits_equivalent=True`、`token_ids_exact_match=False`，
**且首次分岔都在 V3 的 `v3_0` 第 4 个 token（ref=15 / drv=16）**。
⇒ 换镜像后**复现了完全相同的数值行为**（连分岔步号都一样），这比"两端都 passed"强得多。

⚠️ 那个分岔**不是缺陷**，但值得记：该行 logits **完全相同**（`max_abs_diff=0.0`、argmax 198/198），
产物 `notes` 写明该步是**近平局**（top1=16 top2=15 **gap=0.0000 < 0.05**，fp16 舍入即可翻转 argmax）。
按项目自己的**三态分流**规则（`near_tie` 记 note；`forensics_failed` / `not_near_tie` **阻断通过**），
`passed=True` 是**合规**的。V4b 也有同构的一条（第 3 步，top1=17 top2=16，gap=0.0000）。

⚠️ **顺带发现一处报告不对称（登记，不改）**：`V4b` 有结构化裁决字段
（`divergence_verdicts` / `n_divergence_steps` / `any_blocking_divergence`），
而 `V3` **只有**裸 bool `per_row_match`，理由只写在自由文本 `notes` 里。
判定逻辑没问题（`needs_review` 为空），但**读者与下游脚本无法从字段直接分辨**
"合规的近平局"与"阻断性分岔"——前者要 grep 中文散文，后者应当让机器报警。
⇒ 建议给 V3 补上与 V4b 同形的结构化字段；**属独立变更**（改产物 schema 要同步 preflight/聚合），
故本轮只登记。

---

## 2026-09-24（夜，续）— 第 03 步：开放环 `arrival` 参考点修复 —— "两个执行器只修了一边"

### 一、缺陷形状：契约一直都在，是真实后端没守

`benchmark/real_backend.py` 与 `benchmark/run_benchmark.py::simulate()` 是**同一份调度语义的
两份实现**（一份接设备、一份走一阶模型）。模拟器早已是：

```python
s.arrival_ms = scheduled_arrival      # 放行时刻
s.enqueue_ms = step_start             # 受理时刻
```

而真实后端写的是 `s.arrival_ms = step_start`。**模拟器里还留着解释这段的注释**
（"为什么到达时刻不能写成 now：一步可能很久……这段等待会被整段抹掉"），真实后端没有。

`metrics.py` 对 `scheduled_arrival_ms` 的注释更是直接把这条写死了：
"**不得被执行步的时间覆盖**"。⇒ **契约没变，是本步之前的真实后端不符合它**——
所以这不是"改口径"，是**对齐口径**。

### 二、最隐蔽的一点：`accounting_error == 0` 证明不了口径正确

旧读法把 `arrival` 覆盖成 `step_start` 之后，`done_ms` / `ttft_ms` / `e2e_ms`
全部以**同一个被覆盖的参考点**计量 ⇒ 账目依然自洽、`accounting_error` **依然是 0**。

⇒ 反例里**把旧读法也算出来并断言它是 0**，让"不能只看账目"成为一条会失败的断言：
两套参考点的 TTFT 差 **90 ms**，而两边的 `accounting_error` 都是 0。

### 三、反例（离线假时钟，验收给的那个例子）

假驱动每步耗时 = `fixed + per_seq×批大小 + per_token×token 数`；取
`fixed=0 / per_seq=0 / per_token=1 ms`。日程 `r0@0 ms`（占满第 0 步：100 token ⇒ 100 ms）、
`r1@10 ms`（落在第 0 步内部）：

| 量 | 正确 | 旧读法 |
| --- | --- | --- |
| `arrival_ms` | **10** | 100 |
| `enqueue_ms` | 100 | 100 |
| `first_token_ms` | 101 | 101 |
| **`ttft_ms`** | **91** | **1** |
| `admission_ms` | 90 | 0 |
| `queue_ms` / `prefill_exec_ms` | 0 / 1 | 0 / 1 |
| `accounting_error_ms` | **0** | **0** ← 判据失效 |
| `SLO(ttft_ms=50).satisfied_by` | **False** | **True** ← 结论翻面 |

⇒ 与验收给的形式完全一致（"日程 10 ms，100 ms 才受理，101 ms 出首 token ⇒ TTFT 91"）。

### 四、为什么两个执行器分叉了却没有用例失败

**两边的"时刻单调性"断言不是同一组**：

* `tests/test_simulator.py`：`assertLessEqual(scheduled_arrival_ms, arrival_ms)` ✓
* `tests/test_real_backend.py`：只有 `arrival <= enqueue <= first_schedule <= ...` ✗

⇒ 分叉正好落在那条**只有一边有**的不变式上。已把缺的那条补上，
并在注释里写明"**两边必须断言同一组不变式**"。

### 五、影响面：既有数据**零影响**（用产物核实，不是推断）

全量扫描 `results/raw/**/meta.json` 共 141 个：

| 到达模型 | 标签 | 个数 |
| --- | --- | --- |
| `closed` | `MEASURED` | **123** |
| `closed` | `TEST_DATA` | 15 |
| `burst` | `TEST_DATA` | 3（`cmp-adaptive_v1*`，且走的是本就正确的模拟器） |

⇒ **123 个 `MEASURED` 全是闭环**。闭环没有"原定到达"，`scheduled ≡ arrival ≡ step_start`
⇒ **本修复不改变闭环行为** ⇒ **既有结论一条都不受影响、不需要重跑**。

⇒ 本修复的真实意义是**即将进行的开放环实验的前置条件**：`load_ramp` / `piecewise`
在修复前会让 pilot 的 TTFT **系统性低估**（这正是第 03 步被排为"真机 pilot 的前置项"的原因）。

### 六、验证

* 新用例：`Ran 7 tests — OK`。
* **反向对照（证明用例会失败）**：把那一行临时改回 `step_start` ⇒
  `Ran 7 tests — FAILED (failures=5)`，报错正是 `1.0 != 91.0`；还原后
  md5 与备份逐位相同、`CR=0`。
* 整模块：`Ran 44 tests — OK`（原 37 → +7）。
* **全量离线回归：807 用例，真正执行 807，跳过 0，失败 0，错误 0**（原 800 → +7）。
* 换行符：4 个被改文件 `CR=0`。

### 七、登记

`EXPERIMENTS.md` 新增 **§4.3 P5**（前置条件）+ **§4.3.5**（缺陷形状 / 判据表 / 反向对照 / 影响面）；
`README.md` §6 写明参考点与**从产物即可区分的判据**：
**开放环下一整批的 `admission_ms` 恒为 0 ⇒ 旧口径**；新口径下必有 `admission_ms > 0` 的请求。

### 八、待办

1. **真机验证未做**：本步是纯离线修复；真机侧开放环第一次跑就是第 08 步 pilot，
   那里应顺带核对产物里 `admission_ms > 0`。
2. **远端源码树仍是修复前的 `2bdf760`**，跑 pilot 前需重新 `push` 并投放新戳。
3. ⚠️ **`simulate()` 与 `real_backend` 的其它不变式仍未逐条对齐**：
   本步只补了 `scheduled <= arrival`（它是本次分叉的判据）。**两处应做一次不变式对齐审查**
   ——两份实现共享语义却不共享断言，这类分叉还会再发生。

---

## 2026-09-24（深夜）— 第 04 步：把"两种停顿"分开记录（**只记录，不改判据**）

### 一、要解决的问题（E9 已定因）

控制器读的 `tpot_p90_ms` 来自**相邻两次 token 之间的间隔**，而某一步做了 prefill 时，
落在这两步之间的间隔**直接包含该 prefill 的耗时** ⇒ "prefill 挤占"被读成"decode 变慢"，
而控制器的响应是**调小 `chunk_size`**（prefill 更碎 ⇒ 含 prefill 的步更多）——**反向放大**。

### 二、做法：两个单调时钟

每步结束按"本步有没有 prefill 工作"把耗时累进 `prefill_clock_ms` 或 `decode_clock_ms`，
两次 emit 之间的时钟差就是该间隔的两个分解量：

| 新增量 | 含义 |
| --- | --- |
| `interval_had_prefill` | 该间隔里**任意一步**做过 prefill（与 `intervals` 逐项对齐） |
| `interval_decode_ms` | 该区间里**纯 decode 步**的耗时之和 |
| `step_trace` | 每步 `(时刻, 是否含 prefill, 纯 decode 耗时)` ⇒ **可从产物独立重算** |

**为什么用时钟差而不是"看最后一步"**：一个间隔可能跨很多步，"只看发 token 的最后一步"
会漏掉**夹在中间的** prefill。时钟差天然覆盖区间内**全部执行步**。
⚠️ 另外必须在**结算之前**读相位——结算会把算完 prefill 的请求改成 decode/done，那时判据恒假。

### 三、接线：默认不开，且轨迹逐字节不变

* `controller.signal_mode`：**默认 `legacy`**；`layered` 才收集并写进决策轨迹。
* `controller.signal_window_expiry_ms`：**默认 `None`** = 不过期。
  设了值后，"距上次新样本超过上限" ⇒ 本周期**不动作**且**不消耗观察周期**，
  理由写进轨迹。此前**完全没有过期规则** ⇒ 负载停顿时旧样本仍会驱动收缩/放大。

### 四、⚠️ 过程中被测试抓到的 3 处

1. **`ControlAction.describe()` 无条件 `round(v, 4)`**，而分层信号里混进了 `str`/`bool`
   ⇒ `TypeError`，**18 个用例一起红**。修法：判据放宽成"是数值才取整"，
   而不是把非数值字段挪出 `signals`——**它就该出现在轨迹里**（那正是它能被核对的原因）。
2. **legacy 下轨迹被改**：初版无条件把分层键并入 `signals` ⇒ `_unavailable_signals`
   在 legacy 下也冒出 4 个键 ⇒ 2 个既有用例红。改成"只在 `layered` 下并入"。
   ⇒ **"旧批次语义不变"必须落到轨迹的字节上，不能只落在参数上。**
3. 我自己写错两处**期望**：`percentile([60,70],90)=69.0`（**带插值**，不是 70）；
   以及"记年龄"与"按年龄设限"是两件事（年龄**总是**记录）。

### 五、⚠️ 一项能力边界（如实登记）

验收要求"标记覆盖区间内**全部**执行步"。在**真实夹具**里这条只能用
"与 `step_trace` 独立重算一致"来证——因为试了 4 组预算参数
（`max_num_seqs` 1/4 × `max_batched_tokens` 1/2/3 × `min_decode_seqs` 0/1 × `chunk_size` 1/2）
都构造不出"某个请求被整步饿住"：**本项目的分配器优先满足 decode**，
预算紧张时被饿的是 prefill，不是 decode ⇒ 真实夹具里每个间隔恰好只跨一步，
两种读法给不出不同答案。

⇒ 另加一条**规格测试**（合成步序列：prefill 在中间、末步纯 decode）钉住语义，
并在 docstring 里写明它证的是**规则本身**，真实侧由重算一致性负责。
**两者合起来才是完整证据。**

### 六、验证

* 新增 13 项：后端 7（`TestLayeredSignals`）+ 控制器 6（`TestLayeredSignalWiring`），全过。
* **全量离线回归 820 用例 / 真正执行 820 / 跳过 0 / 失败 0 / 错误 0**（807 → +13）。
* 既有 `configs/` **一个都没动**（默认 legacy ⇒ 生成器逐字复现测试不受影响）。

### 七、待办

1. **分层信号尚无消费者**（本步只记录）⇒ 第 09 步之前必须**显式保留这个中间态**。
2. **真机未验证**：全部离线；真机侧第一次产生分层信号是第 08/09 步的 pilot。
3. **过期上限未标定**：要启用 `signal_window_expiry_ms` 必须先有实测的窗口刷新间隔。
4. ⚠️ **不得**顺手改 `recent_tpot` 的既有语义（把 prefill 从间隔里减掉）——
   那是"分层"的另一半，属独立变更，否则又是一次"一次改两个变量"。

---

## 2026-09-24（深夜，续）— 第 05 步：动态负载的分阶段报告（纯增量）

### 一、为什么需要

`piecewise` 的自变量是负载**轨迹**，整 run 一个汇总值回答不了它：低压段与高压段的读数
混进同一个中位数（那个数谁都没测过）；恢复段有没有回落、多久回落看不见；
高压段末尾到达的请求在恢复段完成——算哪一段？

### 二、两套口径，**不能互相替代**

| 口径 | 归属依据 | 回答什么 |
| --- | --- | --- |
| **cohort** | 请求的**到达/放行时刻** | "这一段进来的请求，后来表现如何" |
| **window** | 事件**发生**时刻 | "这段时间系统在处理什么、积压多少" |

跨阶段请求**同时出现在两处**——两个问题的两个答案，不是重复计数。

### 三、三条预先定死的规则

1. **达标判定 = 请求完成时**；未完成/失败/被拒不算达标。
   实现上**直接调用权威的 `SLO.satisfied_by`**，且 SLO 取自**该 run 自己**落盘的
   `meta.config.run.slo` ⇒ **报告层不另写一套阈值判断**（两处各写一套 = 粗的那套放行），
   也不拿当前默认阈值重算旧 run（阈值改过后旧批会被用新判据算）。
2. **分母 = 测量分母**（分窗口 = 窗口墙钟长度）。不用"最后一个完成时刻"——
   那会把排空尾段从分母里抹掉、吞吐虚高。
3. **恢复 = 显式阈值 + 持续时间**；没观察到 ⇒ `not_recovered`/`censored`，
   **用时 `None`，不填 0**。（`queue_depth` 里有 0，阈值取 0.5 会误判"已恢复"⇒
   要构造"从未回落"必须让阈值高于全部观测值。）

另：**排空尾段单独成段**（`drain`，不是负载的一段）；**低样本报 `n` 且不给 P95/P99**；
**对账**（cohort 之和 == 请求总数、窗口完成之和 == 成功数，对不上则退出码 1）。

### 四、★ 分母口径的交叉核对（本步最要紧的验证）

我算的窗口达标吞吐 vs 产物既有的 `summary.slo_goodput_tok_s`：

| run | 注册值 | 本脚本 |
| --- | --- | --- |
| `adaptive_c1` / `-r1` / `-r2` | 15.277 / 15.292 / 15.283 | 15.29 / 15.31 / 15.30 |
| `adaptive_c2` | 22.474 | 22.51 |

⇒ 差 ~0.1% ⇒ **同一个分母**。差的来源已查清并写进脚本 docstring：
`wall_ms` **没有落进 meta**（它在 `summary` 里被用来算过、但本身未持久化，`backend_stats`
也不在 meta 里），故用 **telemetry 的 `t_ms` 跨度**作代理；代理**略偏大**。
要让两者逐位相等须先持久化 `wall_ms`——**属独立变更**（登记 R23）。

### 五、过程中被测试抓到的问题（3 处）

1. **★ 局部字典没挂回报告**：`build_stage_report` 里 `cohorts` 建好了却**忘了
   `rep.cohorts = ...`** ⇒ 报告里 cohort 恒为空，**8 个用例一起红**（`KeyError: 'low'`）。
   这正是本仓库最警惕的那类——"算出来了但没接上去"，不报错、只是字段空着。
2. **"未成功却自报达标"**：夹具里失败/被拒的请求也带着 `slo_satisfied=True`。
   两边都修：① 模块加最后一道闸（达标 = `slo_satisfied` **且** `outcome == success`，
   与 `SLO.satisfied_by` 一致）；② 夹具按事实造。⇒ 只信自报字段会把达标率整体抬高。
3. 我自己一处期望写错（`queue_depth` 里的 0 与阈值 0.5）。

### 六、验证

* 新增 22 项（`tests/test_stage_report.py`）：跨阶段、失败/被拒/未完成、对账不通过、
  排空尾段存在与不存在、恢复四态（recovered / not_recovered / censored / not_applicable）、
  抖动不算恢复、低样本不给尾分位、边界不重复计数 —— 全过。
* **全量离线回归 842 用例 / 真正执行 842 / 跳过 0 / 失败 0 / 错误 0**（820 → +22）。
* 在**真实 MEASURED 产物**上实跑：对账 ✅ 8/8/8，且**非 piecewise 时脚本明确声明
  "分阶段结论不适用"**（不凭空造阶段）。

### 七、待办

1. **真机 piecewise 尚未跑过**：全库搜过，现有产物**没有一个**含 `arrival_segments`
   ⇒ 分阶段报告只在合成夹具上验过；真机验证要等第 08 步 pilot（登记 R24）。
2. **`wall_ms` 落进产物**（R23）——让分母逐位相等。
3. 恢复判据的默认指标固定为 `queue_depth`；按其它量判需要扩参数（R25）。

⚠️ **本节的 §四 有一处判断错误，2026-09-25 更正**：当时写"`wall_ms` 没有落进 meta"是**错的**
—— 它就在 **`summary.wall_ms`**（P1S3D `adaptive_c1` = 67029.045）。错因是
**把 `summary` 的键列表截断在 25 项**（按字母序 `t > s` 的键全被截掉）就下了结论。
⇒ `report_stages.py` 已改为**优先读 `summary.wall_ms`**，telemetry 跨度只作兜底并标明来源。
**教训：清单被截断与字段不存在，看输出时长得一样**——数条目前要先确认没被截断。

---

## 2026-09-25 — 第 09 步离线半：E9 的机制在开放环 piecewise 上复现，并量化了收缩的代价

### 一、为什么先做离线半

第 09 步的验收第一条要"动作时刻 / 实际生效 chunk·budget / prefill 分布 / 真实 ITL 与队深的
对应证据"——**pilot 的 adaptive 臂产物里全都有**（`controller_trace` 443 决策 + `telemetry`
每步 `note` 的 `prefill_tok`/`decode`/`fwd_ms`）。⇒ 不碰真机就能拿到结论级证据。

### 二、★ E10：污染在负载连续变化下同样出现，且可复现到步号

三个 adaptive run 的**动作步号逐位相同** `[1216,1240,1264,1288,1312,1336,2432,2456,2480]`，
首次动作 `t = 123.0/123.1/123.2 s`，**`queue_depth` 三个都是 0**。

| | run1 | run2 | run3 |
| --- | --- | --- | --- |
| 报告 `tpot_p90` | 337.2 | 337.9 | 342.2 ms |
| 同窗**纯 decode** `fwd_ms` p50 | **61.4** | 61.6 | 62.5 ms |
| 同窗**含 prefill** p50 | **329.3** | 329.0 | 331.5 ms |

⇒ 读数 ≈ 真实 decode 的 **5.5 倍**，**在没人在排队时收缩**。

**代价**：5 次收缩把 `max_batched_tokens` 8192→**2048**，**纯 decode p50 61→118 ms（+92%）**。
⇒ **动作把真实 decode 弄慢了近一倍**，而它要修的"问题"在纯 decode 口径下不存在。
这解释了同批 adaptive 的 `tpot` p50 高于 static（182.4 vs 175.9 ms）。

**顺带**：这回答了 `EXPERIMENTS` §3.3 限制 2 登记的"该机制在负载水平变化下是否仍出现——
本批测不到"。

### 三、★ E11：否掉"反向放大"

9 次动作前后各 64 步的含 prefill 步占比变化 `+31/−2/−42/−14/+25/+50/−12/−6/−3` pp
（**3 升 5 降 1 平**）⇒ 不是单向机制。E9 末尾那句话已就地收窄：
**成立的是"污染存在且能触发动作"，"收缩加重污染"证据不足、不得引用。**

### 四、下一步（第 09 步真机半）

新增 `controller.signal_criterion: tpot_p90 | decode_p90`（默认 `tpot_p90` ⇒ 既有批次不受影响），
跑单变量干预 A/B，并已先登记 **4 条可证伪预测**（B 臂首动作不在 t=123s/队深不空、
纯 decode 全程 ≈62 ms、`tpot` p50 降到 ≈175.9、`stall` 不再低 20%）。
⚠️ 若 1–3 都不成立须回头重查，**不得**用"控制器没动"当结论。

---

## 2026-09-25 — 第 09 步真机半：判据口径 A/B 干预结果（E12/E13）

### 一、先说差点丢掉的经历与一次判断更正

1. 凌晨那次 A/B（02:33 启动、03:23 完成）**未及时取回**；上午实例重启后系统盘回滚 ⇒ 一度找不到。
2. 用户在 13:4x **克隆了实例** ⇒ 克隆保留了关机时的磁盘 ⇒ **6 份产物完好**，已取回入库。
   ⇒ 我先前写的"白跑一次"是**错误判断**，已就地更正。**但教训不变：这次是运气，不是纪律。**

### 二、结果：机制成立，净收益很小（E12）

**触发值的性质**（报告判据 ÷ 同窗纯 decode `fwd_ms` p50）：

| 臂 | 比值 |
| --- | --- |
| A（`tpot_p90`） | **5.47 / 5.47 / 5.73**（被 prefill 污染 5.5 倍） |
| B（`decode_p90`） | **1.33 / 1.33 / 1.32**（报的就是纯 decode） |

⇒ 换判据后，那批"**在纯 decode 还只有 61 ms 时收缩**"的动作消失：
首次动作由 `t≈123.0 s` 推迟到 **`t≈131.2 s`**，且触发值 138.0 > 门限 129.6 ⇒ **合法触发**。

**用户指标**：达标率 `0.1957 → 0.2065`（**+1 个请求** / 92，组内极差都是 0）；
`tpot` p50 `182.53 → 180.79`；**`stall` p99 `330.5 → 410.3`（+24%，回到 static 水平）**。

⇒ **是权衡不是"换判据就更好"**：B 放弃了 A 靠过度收缩换来的 stall 改善。
⇒ E10 的因果链：**前半（污染触发）被消除；后半仍在** —— B 仍会合法收缩，
之后纯 decode 同样到 118 ms。**"改判据"只能消除不该有的动作，消不掉收缩本身的代价。**

### 三、4 条预测 1 pass / 3 fail，其中两条是**预测自己写错**（E13）

- ② **设计缺陷**：把"机制成立"与"动作消失"混为一谈 —— 它假设"不再收缩"，
  但 B 会**合法地**收缩。
- ① **代理不可靠**：用"`queue_depth` 不空"当"有真实压力"，
  而 8 槽全满、批内 8 条序列时纯 decode 本来就会变慢。
- ⇒ 两条的修正**只能作为 post-hoc 登记**，**不回头改写预测**（那正是"事后挑读法"）。

### 四、阈值方案冻结

自适应臂在正式比较中用 **`decode_p90`**；门限 `120 / 0.08` **不变**（在 `decode_p90` 下未被误触）。
⚠️ 一并冻结代价：放弃 A 那"低 24% 的 stall"；若第 14 步主指标改成 stall，这个选择要**重新审**。

### 五、工具教训（第三次踩，这次记进 PLAYBOOK）

用 heredoc 写**含中文引号**的 Python 字符串会出 SyntaxError，而 `&&` 链后面的 `git commit`
**照样执行** ⇒ "代码提交了、文档没写"。**中文长文本一律用编辑/写入工具，不要走 heredoc。**

---

## 2026-09-25 — 第 11 步完成：成本模型**在标定范围内成立**（协议 cost-calib-v1）

⚠️ 本节结论**不进** `EXPERIMENTS.md` 结论行：探针产物无 harness 盖章（与 `_probe/` 同族）。

### 采集
3 个会话 × 24 个形状点 × 两个块，**0 错误**（`results/raw/_costcalib/`）。
块 A = 快态平台（占空比 ~100%，`lower_bound_only`）；块 B = 服务协议（逐步 sync + KV 真实生长，
`fitted`）。两者差值随 prefill 增大（b1/pf512：A 156.03 vs B 166.85 ms）——
这给"不能只拟合快态最小值"提供了实据。

### 结果
| 上下文 | 单超平面 | **按 batch 分段** | 查表 |
| --- | --- | --- | --- |
| ① 形状留出（内插） | 低估 p95 **110.83%** | **1.77%** | 6 键全未命中 |
| ② 跨会话留出 | 292.11% | 11.56% | **0.29%** |
| ③ 外推 kv=2048 | 0%（**总是高估 908%**） | **23.11%**（92% 低估） | 无覆盖 |

**④ 跨会话漂移**：中位 **0.45%** / p90 0.53% / max 0.62%（远小于 §6 的 1.3–4%，
因为本次三个会话是**同一天同一机器**；§6 是跨天观察 ⇒ 反而印证了"同会话内对照"的纪律）。

**⑤ 判据**（预注册 §五，对分段模型在①上）：**模型成立** ✓。

### 结论的准确表述
> 标定范围内，按 batch 分段的分段模型把**未参与拟合形状**的相对误差控制在
> 中位 **1.74%** / p90 **2.41%**，**低估尾部 p95 = 1.77%** ⇒ 足以支撑"选哪个 chunk"。
> **超出范围（kv=2048）不成立**（低估 p95 23.11%），须**回退**不得外推。
> 单一超平面线性模型在这套形状面上**结构上不成立**（低估 p95 110.83%），不是调参问题。

> ⚠️ **2026-09-26 更正（上面那些精度数字的适用范围被收窄）**：
> 用**同一份数据换更严的留出**重算后发现，`1.77%` **不是泛化能力，是那个特定划分的运气**：
>
> | 数据集 | 训练点/段 | **1-in-4（预注册）** | **1-in-2** | **1-in-3** |
> | --- | --- | --- | --- | --- |
> | v1（本步，72 条/4 段） | 13.5 | **1.77%** | **95.25%** | **68.78%** |
> | v3（46 点，8 段） | ~7 | 5.15% | **95.26%** | — |
>
> ⇒ 真因：**`kv` 维不可辨识** —— 每段超平面 3 个参数，而网格里 `kv_len` **只有 3 个取值**
> （512/1024/2048）⇒ 留出一半后每段在每个 kv 值上只剩 ~1.5 个训练点 ⇒ 斜率乱跑。
> ⇒ **因此**："低估 p95 = 1.77%" **只能**表述为"**在 1-in-4 这个划分下**的留出误差"，
> **不得**当作泛化精度或"足以支撑选 chunk"的依据。
> ⇒ 且 `safety_margin` 是由这个数推出来的（见 `model_v1.json` 的 `safety_margin`）
> ⇒ **余量的量级同样不可信**。修法见网格 v4（kv 加密到 5 值 + 划分改 1-in-3）。

### ⚠️ 三件如实登记的事
1. **预注册判据不完整**：原两条会被"总是高估 400%"的模型满足 ⇒ **事后**补
   `|相对误差| p90 ≤ 10%`（理由写进常量注释），已标明非预注册。
2. **分段变量（batch）是看到数据后选的**（b1→b2 跳 +29% 而 b2→b4 几乎不动）
   ⇒ 登记为 post-hoc；预注册只点名"分段模型"这一类、未指定分段方式。
3. **评估里修掉过一处泄漏**：查表只建在哈希训练集却评 kv2048 外推集，而训练集含 kv2048
   ⇒ 曾报出"0% 误差"的假结果；已改为每个上下文独立拟合。

---

## 2026-09-26 — 第 14 步夜间队列：**闸门中止了，但中止理由写错了**（v3 网格被否 + 一处路径缺陷）

- 背景：夜间队列 `/root/run_s14_night.sh` 的设计是"等 v3 采集 → 拟合 v3 → §五 闸门
  → 导出 `model_v3.json` → 跑 fixed 格（tag `S14C1`，4 臂 × 5 重复）→ 覆盖率核验"。
  v3 网格的唯一动机：v1/v2 的 `batch` 只有离散的 `{1,2,4,8}`，而真实服务连续取 1..8
  （`max_num_seqs=8`）⇒ `cost` 臂在真实格上的有效工作覆盖率只有 **51.35%**，
  回退成因 94.4% 是"batch 不在标定网格"。目标是把覆盖率提到 ~100%。

- 决策 / 发现：
  1. **队列以 `S14NIGHT_ABORTED_AT_FIT` 结束**：`S14C1` 一格未跑、`model_v3.json` 未导出。
     但闸门打印的"**模型不成立**"**不是** §五 判据的结论 —— 它的取值行是
     `ERR:[Errno 2] … costcalib_eval_v3.json`，即**评估文件根本没被生成**。
  2. **成因：一处路径缺陷（静默型）**。采集脚本写的是 `adhoc/costcalib_v3_s*.json`，
     队列的拟合步骤 glob 的却是 `results/raw/_costcalib/costcalib_v3_*.json`；
     而 `results/raw/_costcalib/` 在**远端从来不存在**（v1/v2 的产物也一直在 `adhoc/`）。
     v1 那轮之所以没暴露：本机 `results/raw/_costcalib/` 里 v1 副本的取回时间（17:33）
     早于 `model_v1.json`（17:45），且远端从未有过该目录 ⇒ **那一轮的拟合是在本机跑的**，
     路径不一致因此被绕过。这次把拟合搬到远端，缺陷立刻显形。
     ⇒ 拟合拿到 0 条记录（`rc=2 / 没有任何可用记录`）⇒ 闸门读到不存在的文件 ⇒ 中止。
  3. **把 v3 产物取回本机、用与 v1 完全相同的流程复算后：闸门结论确实是"不成立"**，
     但**理由与队列打印的不同**：真因是 `低估相对 p95 5.15% > 5%`（阈值 5%）。
     ⇒ **中止这个动作是对的，写下的理由是错的**，二者靠巧合一致。
     危险处在于方向：若路径是对的而模型恰好成立，同一个缺陷就会让一次**该继续**的队列
     被读成"正常跑完"，或反之 —— 判据没被评估，而产物看起来有了结论。
  4. **v3 的取舍已见分晓：形状范围修好了，精度换亏了。**
     范围侧：v3 记录到 `b1: prefill[0,2048] kv[0,2048]`、`b2..b8: prefill[0,1024] kv[512,2048]`，
     `S14B1` 实测的 **90/90** 个回退形状全部落入 ⇒ "51.35% → 100%" 在形状范围意义上成立。
     精度侧（2 会话 / 92 条记录，① 形状留出，按 batch 分段模型）：

     | §五 判据 | 阈值 | v1 | v3 |
     | --- | --- | --- | --- |
     | 低估 p95 | ≤ 5% | 1.77% | **5.15%** ❌ |
     | 跨会话漂移 p90 | ≤ 10% | 0.53%（max 0.62%） | 0.69%（max 0.77%） ✅ |
     | \|相对误差\| p90 | ≤ 10% | 2.41% | **5.69%** ✅（降 2.4×） |

     ⇒ `verdict.model_holds = False` ⇒ **不导出 `model_v3.json`**。
     ⚠️ 成因**未定位**：只测到"加密采样点后 ① 与 ③ 的低估尾部同时变差"，
     登记为待查，**不得**写成"线性 basis 不足"这类已确认结论。
  5. **另一处静默缺陷（尚未触发，已定位）**：`analyze_step14.py` 的一致性闸门
     `crit != {"decode_p90"}` **没有豁免 `controller.enabled == false` 的臂**，
     而 `tests/test_step14_configs.py`（第 46–48 行）**显式豁免**了它 ⇒ 两个守卫口径不一致。
     `fixed_static.yaml` 不声明 `signal_criterion`，`describe()` 便把 dataclass 默认值
     `tpot_p90` 写进产物 ⇒ 任何含 static 臂的批次都会让分析器打出
     "⚠️ 与冻结的 decode_p90 不一致"。这是**假警**：它把"static 臂没有控制器"
     与"自适应臂真漏了字段"混成同一条告警，而后者正是第 14 步踩过的真回归。
     （本轮 3 个自适应配置 `fixed_threshold/cost/slo_weave` 都已正确声明 `decode_p90`。）

- 影响：
  * 第三臂（`cost`）**本轮没有可部署模型**：v1 模型只覆盖 batch `{1,2,4,8}`，
    v3 模型被 §五 判否 ⇒ 现在跑 `fixed` 格，`cost` 臂仍会以高回退率运行，
    四臂对比**无法归因到成本模型**（"回退率高时与 threshold 接近"≠"成本模型没用"）。
  * `S14B1`/`S14B2` 与后续任何批次**不得混算**（那两批是默认 `tpot_p90` + `model_v1`）。
  * 未动冻结源码/配置：本轮只取回与复算产物、更新文档。

- 下一步：
  1. 修队列路径（二选一：采集直接写 `results/raw/_costcalib/`，或拟合 glob 改指 `adhoc/`），
     并把闸门的 `ERR:` 与"模型不成立"**分开报**（现在 `case` 把两者并成同一句话）。
  2. 定位"加密 batch 采样点为何抬高低估尾部"；在此之前**不要**再扩网格。
  3. 给 `analyze_step14.py` 的一致性闸门补上 `enabled == false` 豁免，与配置测试对齐。
  4. `S14C1` 的启动取决于 1–3；**不得**在 v1 模型下跑 fixed 格并称之为第三臂对照。

- 关联：`results/processed/costcalib_eval_v3.json`、`results/raw/_costcalib/README.md`、
  `configs/cost_calib/shape_grid_v3.yaml`、`NEXT_SESSION_TASKS.md` 第 14 步。

---

## 2026-09-28 — OPT-01：零成本只读分析（S14B2 性能态 + 最优工作带）

- 背景：优化测试会话 `OPT-01`（路线 S0），**只读** `S14B2` 既有产物，回答两问：
  (1) `load_ramp` 的 low 段是否落入**设备慢态**（`COST_PROFILE` §4 的部署损失与
  控制器信号污染来源）；(2) 负载若设计在 `c4–c6`，实测 TPOT 是否落 SLO 100 ms 内。
  不产生任何新实验条件。脚本 `scripts/analyze_opt01_s14b2.py`
  （复用 `decompose_step_overhead.parse_note`、`stage_report.build_stage_specs`、
  `run_ids.parse_run_id`，**未新写**重复解析器）。

### 一、前置核查：分母从 11 变成 19，且**没有第 12 个坏 run**

- 2026-09-28 20:16 实测（`meta.json` 可解析 **20/20**、`telemetry.jsonl` 无坏行
  **19/20**、`requests.jsonl` 无坏行 **20/20**）⇒ **报表分母 = 19**。
- 唯一不可用：`ramp_threshold-r1` —— `telemetry.jsonl` 尾部截断（78 行 / 1 坏行 /
  停在 **15.3 s**，名义终点 450 s）。**按尺寸门槛会误判为完好**，只可重拉不可修。
- ⚠️ `OPT-01` 任务书 §2.1 与 `EVIDENCE_INDEX.md` §1.7b 记的「11/20」是 **13:08 的快照**；
  本地 **17:51–17:57 又落盘 12 个目录** ⇒ 现测数字已变。**以后引用一律现测并写明时刻**，
  不再写死分母。

### 二、★ 任务书里那条 `t_ms` 前提是**错的**（已证伪，勿再按它处理）

- 任务书 §2.1 断言：首行 `step=0, t_ms=10143.23` ⇒ 墙钟起点带约 10 s 偏移，
  **不能直接把 `t_ms` 当段边界**。
- 证伪方式（可复算，脚本每次运行都重做）：用 `requests.jsonl` 的
  `scheduled_arrival_ms` 数**各段内到达数**，与 `meta.json`
  `arrival_segments` 声明的 `realized_count` 比：

  | 假设 | 逐段到达数 | 复现声明的 run 数 |
  | --- | --- | --- |
  | `anchor=0`（段时钟，不平移） | **[10, 63, 19]** ✅ | **19/19** |
  | `anchor=min(arrival)≈10133`（平移） | [17, 57, 18] ❌ | 0/19 |

- ⇒ `arrival_segments` 与 telemetry `t_ms` **同一时钟**；首行 10143 是 `low` 段
  **首次到达**（`qps=0.08` ⇒ 平均间隔 12.5 s），**不是**偏移。段边界直接用 `t_ms`
  （`stage_report` 本来就是这么做的）。**任务书该条已就地划掉并附判据。**

### 三、★ 分档键不是 `running`，而是 `note.decode`

- `note` 里 `forwards` 在纯 decode 步**恒为 1**；`fwd_ms` **只随 `decode` 变**：
  交叉表 `(running, decode)` 上，`running` 从 1 到 8 的**同一 `decode` 列取值相同**
  （如 `(running=8, decode=1)` 中位 **61.41** ≈ 档位 1 的 **61.33**）。
- 但 `running ≠ decode` 的纯 decode 步占 **20.6%（14185 / 68811）**，
  绝大多数是 `running=8, decode<8` ⇒ **按 `running` 分档会把这批步贴错标签**。
- ⇒ 脚本**两个键都报**，判据用 `decode`（= 顶层 `scheduled_tokens`）。
  ⚠️ 这与 `EVIDENCE_INDEX.md` §1.1「纯 decode 门槛在 `running ≤ 4`」的**分档键不同**，
  两句不可直接互引。

### 四、结论 1：low 段**未落慢态**（否）

| 段 | 档 | n | p10 | p50 | p90 | max | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low | b1 | **21817** | 61.17 | **61.29** | 61.47 | **65.64** | **未落慢态** |
| high | b1 | 1699 | 61.27 | 61.43 | 61.70 | 64.77 | 未落慢态 |
| recover | b1 | 16295 | 61.17 | 61.32 | 61.47 | 65.43 | 未落慢态 |
| drain | b1 | 1885 | 61.22 | 61.40 | 61.51 | 62.13 | 未落慢态 |

- 判据（**只用 b1**，因为 `COST_PROFILE` §4 只给 b1 的双态标定）：快态上界
  68.86 ms（=59.88×1.15，容差理由写在脚本常量注释里）｜慢态下界 72.20 ms（=76×0.95）。
  low 段 max **65.64** 距慢态下界还差 **10.4 ms**；`fwd_ms ≥ 72.2` 的样本 **0.0000%**。
- 更强的一条：**全部 19 run × 4 段，b1 上「连续 ≥ 慢态下界」的步 = 0 处**
  ⇒ 既非双峰、也非整体偏慢，**连秒级翻转的痕迹都没有**。
- 跨臂稳健：low 段 b1 中位 cost 61.31 / slo_weave 61.28 / static 61.29 / threshold 61.29
  ⇒ 与控制器状态无关，是设备/负载属性。
- ⇒ 项目级影响：**不构成** `COST_PROFILE` §4 所说的「阶段边界打回慢档」
  ⇒ **该批不存在由此而来的控制器信号污染**。
- ⚠️ 这是**有限观察**（一个负载 × 一次会话 × 一台机器）。§4 的慢态来自**另一种协议**
  （sync-per-call、占空比 ~65%）；本次未复现**不等于**该现象不存在。

### 五、★ 结论 2：`c4–c6` **不能**达标 —— 实测达标档位集合 = {b1, b2}

干净工作点取 **`drain` 段**（负载已结束、队列 p50 ≈ 0、**跳过率 0%**）：

| `decode` 档 | n | 实测 `tpot_ms` p50 | 队列 p50 | 跳过率 | 对 SLO 100 | 快态标定表(下界) | 实测/标定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b1 | 1885 | **63.34** | 0 | 50.9% | 达标 | 59.88 | 1.058 |
| b2 | 484 | **83.96** | 0 | 99.2% | 达标 | 74.28 | 1.130 |
| b3 | 488 | **113.74** | 2 | 98.4% | **超 SLO** | — | — |
| b4 | 536 | **101.26** | 0 | **0.0%** | **超 SLO +1.26 ms** | 83.28 | 1.216 |

- c5 = 118.4 / c6 = 123.9 / c7 = 137.9 / c8 = 145.4 ms（drain 优先）⇒
  **实测可达的档位只有 b1、b2**。`c4–c6` 的设计假设**与实测不符**。
- b4 那一格是**干净工作点**：`drain` 段、队列 p50 = 0、**跳过率 0.0%**
  ⇒ 4 条序列每步各得 1 token ⇒ 请求级 TPOT ≡ 步时长 ⇒ 101.26 ms 可直接读作 c4 的 TPOT。
- ⚠️ **不得报「最大可行批」这个单个数**：实测 batch 成本曲线**非单调**
  （b3 105.8 ms **>** b4 89.6 ms `fwd_ms`）⇒ "最大档"无定义。只报**集合**。
- ⚠️ 与路线图 §0「SLO 可行最大批 ≈ b6」**不是矛盾，是口径不同**：b6 由**快态标定表
  下界线性内插**得到，而标定表是**性能下界**（`COST_PROFILE` §2 使用边界）；
  按任务书 §2.4 的判据（用**实测** `tpot_ms`）结论是 **b4 已超**。
- 交叉核对（`requests.jsonl` 逐请求 `tpot_ms`，SLO 自己的字段，按到达段）：
  `low` n=190 p50 **63.23**（vs 步级 63.15 ⇒ **两口径同义，口径通过核验**）；
  `high` n=1197 p50 **172.37**（vs 步级 b8 145.4）；`recover` n=361 p50 106.48。
  ⇒ 过载段**请求级 > 步级** ⇒ 步级量是**下界** ⇒ 结论 2 是**保守**判定（方向安全）。
  ⚠️ 两者**权重不同**（按请求 vs 按步），**不得**并列当同一分布。

### 六、⚠️ 两项未定位的现象（登记待查，不写成结论）

1. **batch 成本曲线非单调**：b3 `fwd_ms` 105.8 > b4 89.6（跨臂、跨段、跨队列状态
   **一致复现**，n 分别为 2209 / 2188）。成因**未定位**。后果：所有"按 batch 单调
   内插"的算术（含路线图 §0 的 b6）**不可用**，必须实测。
2. **步级 `tpot_ms` 偏离请求级 TPOT**，且**不能用 `running > decode` 解释**：
   `high` 段 b8 的跳过率 **0.0%**，但该段请求级 p50 172.4 vs 步级 145.4。
   抽查 `r00010`（static）：跨度 19914 ms / 127 个间隔 = **156.80** ms（= 其报告 TPOT），
   而窗口内步级 `tpot_ms` 均值仅 **148.26** ms ⇒ 有约 **1.1 s** 未被步级量覆盖。
   成因未定位。⇒ 已在脚本里写明「跳过率**不是**下界的充分条件」。

### 七、变更与验证

- **新增** `scripts/analyze_opt01_s14b2.py`；**新增产物**
  `results/processed/opt01_s14b2.json`（该目录在 `.gitignore` 内，不入库）。
- **更新** `OPTIMIZATION_TEST_SESSION.md`：§0 登记表；§2.1（划掉并证伪 10 s 偏移前提、
  分母改为现测、新增 `drain` 段与 `decode` 分档键）；§2.3/§2.4（判据修正）；
  新增 §2A 执行记录（变更/验证命令/剩余风险）。
- **未改** `results/raw/**`、**未改** `benchmark/real_backend.py`、
  **未产生**新实验条件 —— `git status --short -- results/raw benchmark/real_backend.py`
  **输出为空**（已实测）。
- `scripts/run_tests.py`：**1357 用例 / 真正执行 1323 / 跳过 34 / 失败 0 / 错误 0**。
- 脚本 rc=**1**（分母 19/20）——**非零 ≠ 方法被否证**，只表示样本不齐。

### 八、剩余风险与下一步

1. 结论 2 只覆盖「瞬时前向批 = c」这一读法，**不是**「并发 c 的稳态」的完整等价物：
   本批没有 c4–c6 的**稳态负载格**（b≥3 只出现在 `high` 与 `recover` 积压段；
   `drain` 是唯一干净点，但 c4 仅 536 步）。要写成论文级结论须靠 B4/收益实验的真机稳态格。
2. 结论 1 是有限观察，不得外推成「该设备不会进慢态」。
3. `slo_weave-r4` 属**已知控制轨迹分岔**（步数 4201 vs 4886），本会话**计入分母**
   但未单列；跨重复引用须沿用 `EVIDENCE_INDEX.md` §1.6 纪律（不得剔除、不得读作收益）。
4. **下一步 = `OPT-02`**（B1+B2+B3 离线准备，见 `OPTIMIZATION_TEST_SESSION.md` §3）。
   本会话**未**开始 OPT-02。
   ★ **OPT-02 必须吸收本条**：B1 的收益负载设计若要沿用「c4–c6 区间」，
   须先解释第 6.1 条（非单调）与本节 1（无稳态格），否则会把一个**未验证的
   工作带假设**写进预注册草案。
5. 关联：`OPTIMIZATION_TEST_SESSION.md` §2A、`OPTIMIZATION_ROADMAP_SESSIONS.md` §0/§5-S0、
   `COST_PROFILE_S4000.md` §2/§4、`EVIDENCE_INDEX.md` §1.1/§1.5/§1.6/§1.7b。

---

## 2026-09-28 — OPT-02：门限形式改造（B2）+ 候选集中间点（B3）+ 收益负载设计（B1）

- 背景：优化测试会话 `OPT-02`（路线图 §5「B1+B2+B3」）。**本会话只做离线**
  （纯逻辑层 + 单测 + dry-run）；未跑真机、未改 `benchmark/real_backend.py`、
  未动 `results/raw/**`。完整记录（变更/验证命令/风险/下一步）见
  `NEXT_SESSION_TASKS.md` 的「OPT-02 完成记录」；`B1 预注册草案` 同文件文末。
  本条目只留**结论与更正**，不复述细节。

### 一、★★ 一个 P0：判据在**最过载的时候**读出 `0.0`，于是控制器放大 chunk

- 机制（只读定位，`benchmark/real_backend.py:1934`）：采集侧只在**本步完全没有 prefill**
  时才推进 `decode_clock_ms`。持续过载段几乎每步都带 prefill ⇒ 该时钟不动 ⇒
  请求相邻两 token 间的 `decode_ms` 恒为 `0.0` ⇒ 窗口 32 个样本**全 0** ⇒
  `decode_p90_ms = 0.0`。
- **它不是 `None`** ⇒ **穿过**了 `controller.py` 里 `if tpot is None: 保守不动` 那道守卫。
  该守卫的 docstring 恰好写着"写成 0 会被读成'纯 decode 几乎不花时间'，那恰好是最乐观的一种读法"
  —— **意图写对了，守卫只盖住了"窗口为空"那一半**。
- 实测影响面（`S14B2` 19 run 现读 `controller_trace`，**三个自适应臂都中**）：
  `criterion_ms == 0.0` 共 **118/6979** 次决策（1.1%–2.2%），其中 **64 次 `changed=True`**；
  这 64 次的 `bottleneck` 全是 `ttft`，`reason` 全是
  "TTFT P90=… 超目标，且 **TPOT 尚有 100.0% 余量**"，TTFT 实测 **13.8 s – 190 s**。
  同一步号在 5 个重复里逐个一致 ⇒ **可复现缺陷，不是噪声**。
- 修法（`src/`）：把"窗口有没有样本"与"窗口**有没有纯 decode 覆盖**"分开
  （新增 `n_decode_covered`）；无覆盖的样本**从分位数里剔除**；无覆盖 ⇒ 判据 `None` ⇒ 保守不动。
- ⚠️ **修法的代价已登记**：深过载段判据不可测 ⇒ 控制器**实质退出**（不动）。
  这比"按 0 ms 放大 chunk"好，但该区间因此**没有控制** —— 必须与收益负载设计一起读。

### 二、B2 门限形式：选「按工作点归一化」，并**先复核了前提**

- ★ **复核结论**：`EXPERIMENTS.md` §4.3.4 那句「工作点跨 9.3×」用的是 `tpot_p90`
  —— 那一列含 **E10 记录的 prefill 污染**。用控制器**实际读到的** `criterion_ms`
  （= `decode_p90`）现测：单 run 内是 **63.0 → 147.1（2.3×）**，不是 9.3×。
- **但结论方向不变**：门限确实被跨过 —— `threshold` 11/446、`cost` 8/446、
  **`slo_weave` 153/610（25.1%）** 次决策落在带 `[110.4, 129.6]` 内。
  更直接的一条：`threshold` 在 step 2400–2448 的 7 次决策上 `headroom` = −0.043…+0.051，
  **散布 0.094 > 门槛 0.08** ⇒ 决策对重复间小扰动**不可复现**。
- 改造：新增 `gate_mode`（`absolute` 默认 ⇒ 既有行为逐字节不变 / `normalized`
  ⇒ 门限 = **同会话在线跟踪的工作点** × (1±h)）。**选归一化而非"随档位调度"**，理由：
  后者需要一张按档位的标定表 ⇒ 换硬件/模型/负载都要重采，且**在信息上与 `cost` 臂重叠**
  ⇒ 削弱三臂的机制区分度。归一化只需**一个标量**、无离线标定依赖、保持"反应式 vs 预测式"的界限。
- ⚠️ **代价（改变"这条臂是什么"）**：`normalized` 下本臂**不再以绝对 SLO 为目标**，
  变成"检测并响应**变化**"的器件（升载收缩、卸载恢复）。要给回绝对约束须显式开
  `slo_hard_guard`（默认关，单列 `n_vetoed_slo_guard`）。
- 已用单测钉住**缩放不变性**（同一相对轨迹在 63/146 ms 两档给出同一串决策），
  并用 `absolute` 模式作**反向对照**（不同串）。

### 三、B3：机制在，但验收项**部分**满足（不得夸大）

- `candidate_chunks` 键**已在 `129f042` 落地**（复用，未重写）。本会话补端到端验收：
  显式 `[128,256,512,1024,2048]` ⇒ `meta` 与 `controller_trace` 里都出现中间值
  `{128,256,512,1024}`；**反向对照**（删掉该键、其余逐字不变）⇒ 只剩 `{128,1024}`。
  ⇒ "候选列表含中间值且被记录"**满足**，且该键是**必要条件**。
- ⚠️ **"中间点被选中并执行"未证明**：两组（有/无候选列表）的 `telemetry` 生效
  `chunk_size` **完全相同**（`{128,512,1024}`），决策里的 `cost_policy.chunk` **恒为 128**。
  原因不在候选集而在**选择规则**（取 prefill 进度最大者；进度相同时取步时最小者）。
  ⇒ **不得**用"补了候选集"宣称"小 chunk 能被选中并执行"——那是 B4 的门。

### 四、B1：三条前提里两条失效，草案**明确不可冻结**

| 原依据 | 现状 |
| --- | --- |
| 「SLO 可行最大批 ≈ **b6**」 | ❌ 由**快态标定表下界线性内插**得到；实测干净工作点 **b4 = 101.26 > SLO 100** ⇒ **c4–c6 不能沿用**（OPT-01） |
| 「往高批挪能改善摊销」 | ✅ 量级仍成立，但**没有可用的目标档位**（高批在实测上已越 SLO） |
| 「TPOT 随 batch 单调」 | ❌ 见 §五 |

- ★ **工作带的问法要改**：可行性与步的**成分**强相关（不只是 batch）。
  `scripts/slo_reach_frontier.py --mode lookup`（**必须** lookup，脚本默认 `segmented`
  与模型自带 `default_mode` 不同）实测：
  * `prefill_tok = 0`（纯 decode）：b1–b5 在 kv ≤ 2048 可行，b6/b8 仅 kv ≤ 1024；
  * `prefill_tok ∈ {128, 256, 384}`：**全档全 kv "不可判"** ⇒ 模型**无这些小 prefill 的键**
    ⇒ 这就是 `EVIDENCE_INDEX` §3 **缺口 3**；
  * `prefill_tok ≥ 512`：**全档超预算** ⇒ **任何 ≥512-token 的 prefill 步都超 120 ms 每步预算**。
- ★ **两个构造性发现**（草案 §1.4）：
  ① **单一 per-step deadline 与"整块 prefill 步"在构造上不相容** ⇒ cost 臂在这些步上**必然**
     degraded/fallback（这解释了 S14B2 `cost` 臂 `n_degraded≈150`）。候选方案 A =
     **把每步预算拆成"decode 步 / prefill 步"两个**（前者服务 TPOT、后者服务 TTFT）；
  ② **小 chunk 路线被缺口 3 卡住**（模型无预测 ⇒ 该臂全程 fallback ⇒"四臂"退化成
     "三臂 + 一条没被评估的臂"）⇒ **B1 冻结依赖 B4 + 缺口 3 关闭**。
- 草案刻意**不填任何未验证数值**（工作带待裁定后推），并列出 5 个待用户裁定的评审问题。

### 五、★ 两条更正（主动传播到引用处，均已就地改）

1. **OPT-01 登记的「batch 成本曲线非单调，成因未定位」⇒ 更正为"形状已确认、机制未定位"。**
   第二条独立来源：成本模型 `model_v6` **独立复现同一凸起**
   （模型 b1..b4 = 62.34 / 79.25 / **105.93** / **88.62**；实测 63.34 / 83.96 / **113.74** / **101.26**）。
   ⇒ 不是采样噪声，也不是报表口径错误。后果不变且更硬：**单调内插不成立**。
   （已改 `OPTIMIZATION_TEST_SESSION.md` §2A 剩余风险第 1 条。）
2. **我在 OPT-01 里写的「成本模型低估 12.6 ms」是错的 —— 那是我自己犯的口径错误**
   （把**前向预测**与**整步实测**并表）。口径一致后：

   | b | 模型(前向) | 实测 `pack_ms` | 和 | 实测 `tpot` | 差 |
   | --- | --- | --- | --- | --- | --- |
   | 1 | 62.34 | 1.41 | 63.75 | 63.34 | −0.41（−0.6%） |
   | 2 | 79.25 | 4.19 | 83.44 | 83.96 | +0.52（+0.6%） |
   | 3 | 105.93 | 7.54 | 113.47 | 113.74 | +0.27（+0.2%） |
   | 4 | 88.62 | 11.20 | 99.82 | 101.26 | +1.44（+1.4%） |

   ⇒ ★ **模型在纯 decode 档上准确到 ≤1.5%**（这同时是第 1 条的第二条证据）。

### 六、变更与验证

- **主产出**：`src/ascend_opt/controller.py`（`gate_mode` 等 4 个配置项 + 4 个状态项 +
  覆盖率守卫 + 门限基准 + `replay_decisions` 接受分层样本）；**默认 `absolute`/`legacy`
  ⇒ 既有批次行为逐字节不变**。
- **新增**：`tests/test_controller_gate_mode.py`（16 用例）、
  `scripts/check_opt02_acceptance.py`（可复算验收，含反向对照）。
- **登记**：`NEXT_SESSION_TASKS.md`（完成记录 + B1 草案 + 表加 `OPT-02` 行）、
  `OPTIMIZATION_TEST_SESSION.md` §0 与 §2A、本条目。
- 验证：`scripts/check_opt02_acceptance.py` **rc=0，11 条断言全过**；
  `pytest`（5 个相关文件）**102 passed**；**全量** `scripts/run_tests.py`
  **用例 1373 / 真正执行 1339 / 跳过 34 / 失败 0 / 错误 0**（改动前 1357/1323/34/0/0）；
  **两条反向控制各自按预期失败后已还原**（`grep REVERSE-CONTROL` 无残留）；
  `git status --short -- results/raw benchmark/real_backend.py` **输出为空**。
- ⚠️ **`dry-run` 不能验证判据逻辑**：实测 dry-run 路径**从不喂** `decode_samples`
  ⇒ 判据恒不可测（72/73 次决策"不可测 ⇒ 保守不动"）、`signals.gate_base_ms` **一次都不落盘**。
  ⇒ "dry-run 链路可跑"只能当"装配没崩"，**不得**当作 B2 的验证证据（验收脚本已把这条钉成断言）。
- ⚠️ 本会话改了 `src/` ⇒ **产生新源码戳**；后续真机批次必须重核 `git_commit`/`digest`，
  既有批次**不得**并表。

### 七、下一步

- ⛔ **不得**开始 `OPT-04`（收益实验，依赖 B1–B4 全部就绪，而 B1 尚不可冻结）。
- **建议下一步 = `OPT-03`（B4 小 chunk 标定，真机）** —— 它是 B1 的前置门。
- 关联：`NEXT_SESSION_TASKS.md`「OPT-02 完成记录」与「B1 预注册草案」、
  `OPTIMIZATION_TEST_SESSION.md` §0/§2A、`EVIDENCE_INDEX.md` §3 缺口 3/4、
  `EXPERIMENTS.md` §4.3.4、`COST_PROFILE_S4000.md` §2。

---

## 2026-09-28 — OPT-03：小 chunk 可预测性标定（B4，真机）

- 背景：优化测试会话 `OPT-03`（路线图 §5「B4」）。任务书问题**逐字**：
  「**小 chunk 能否被可靠预测（lookup 有键且留出误差达标）？**」
  前置：远程 S4000 可达 + `check_cpu_stability.py` 三态判定。
  探针路径 ⇒ **不盖四标签、不进 `EXPERIMENTS.md` 结论表**（只进本日志）。

### 一、前置核查（先查产物再动手）

- ★ **远端可达性**：`autodl-container-cf99469eb6-b877706e`，MTT S4000 空闲
  （0% 占用 / 2 MiB 显存 / 无进程）。
- ★ **三态 CPU 判定 = `0 稳定`**：稳健 CV **0.007%**（≤1%）、PSI `some avg10=0.00`、
  容器配额 15 CPU（`cpu.max = 1500000 100000`）⇒ 计时环境可用。**未触发停止条件。**
- ★ **任务书前提的一个更正**：任务书写「补 **v6** 网格的 128/192/256」，
  但实测 v6 之后已有 **v7**，其 `mixed_small_chunk` 组**已经采过 128/256**
  （`results/raw/_costcalib/costcalib_v7_s{1,2}.json`，2 会话 × 16 形状）。
  ⇒ **128/256 无需重采**；**唯一缺口是 192**（v7 的 `prefill_tok` 轴 = [0,128,256,512,1024,2048]，
  从无 192；`model_v7.json` 的 `lookup_table` 里 pf192 键数 = **0**）。

### 二、做了什么

1. **新开 v8 网格**（`configs/cost_calib/shape_grid_v8.yaml`，`protocol_version: cost-calib-v8`）
   —— **不原地改 v7**，两条硬理由：① 网格自身纪律「看到结果之后再加密采样点，就会把
   『事后挑形状』混进来 ⇒ 要加密必须另开一轮」；② 原地改会让 `grid_sha256` 变 ⇒
   `model_v7.json` 记的 `5c3b66f75c89522c` 失配 ⇒ 血缘断。
   **v8 只改一处**：`mixed_small_chunk.prefill_tok` `[128,256]` → `[128,192,256]`（+8 点）。
   变更理由**采数前**写入文件头，并用 `scripts/gen_shape_grid_v8.py --check`
   把不变量钉成可执行断言（含"比 v7 恰好多 8 点"）。
2. **真机标定两会话**（快态协议 + 会话内对照，`COST_PROFILE` §6）：
   `20260928T221000Z-autodl-c-v8s1`（575.5 s）与 `20260928T222500Z-autodl-c-v8s2`（~524 s），
   各 **24/24 点**、`complete: true`、`grid_sha256 = 3473d023e269706f`。
   产物落盘 `results/raw/_probe/20260928/`（**探针区**），回拉后**远端/本地 md5 逐文件一致**。
3. **切片分析**（`scripts/analyze_b4_small_chunk.py`，**复用** `eval_lookup_deployment_path`
   的键/分桶/取值/度量，不重写）⇒ 产物 `results/processed/b4_small_chunk_v8.json`
   （及 v7 对照 `b4_small_chunk_v7.json`）。

### 三、★ 结论：**能**（小 chunk 可被可靠预测）

| 判据 | v8 实测 | 阈值 | 判定 |
| --- | --- | --- | --- |
| 键完整性 | pf128 / pf192 / pf256 各 **8** 键 | 三档都要有 | ✅ |
| 本批表对**已采** 48 个形状的命中 | **100%**（48/48） | 100% | ✅ |
| 会话内重复性（同形状跨会话） | 中位 **0.01%** / 最大 0.07% | ≤5% | ✅ |
| ★ 跨会话留出·低估 p95 | **0.07%** | ≤5% | ✅ |
| ★ 跨会话留出·\|rel\| p90 | **0.06%** | ≤10% | ✅ |
| pf192 单独 | 低估 p95 **0.06%** / \|rel\| p90 0.06% | 同上 | ✅ |

- ★ **192 与 128/256 同级**（0.06% vs 0.07%/0.05%）⇒ **三个小 chunk 档位全部可预测。**
- ★ **两条可证伪预期都成立**（采数前登记，采完核对）：
  ① **pf 维单调** —— 192 的实测**落**在 128 与 256 之间：
  b4 = 183.52 / **229.36** / 277.15 ms；b8 = 303.20 / **431.56** / 507.76 ms；
  ② **kv 不敏感** —— 同 batch+pf 下 kv 1024→2048 中位 \|Δ\| **0.02%** / 最大 0.06%
  （故"同 batch+pf 补一个键"是安全外推）。
- ⚠️ **必须并排读的第三口径**：**按形状留出**（秩保证 1/3）下覆盖率是 **0.00%**
  —— 那是"被 hash 切出的**新形状**在训练里没键"的产物，**不等于"小 chunk 不可预测"**：
  同 batch+pf 补一个键后误差降到 **0.09%**。**四套覆盖率不可互换**
  （本批表对已采形状 100% / 按形状留出 0.00% / 跨会话留出 100% / 全形状混算 6.67%）。
- ⚠️ **口径边界（不得外推）**：本结论**只对 pf ∈ {128,192,256}**、
  且**只对已采的这 24 个形状**成立；**384 仍未采样**（B1 草案 §1.4 点名的
  `prefill_tok ∈ {128,256,384}` 里的 384 **仍是缺口**）。跨硬件/模型/负载须重采。

### 四、★ 对 B1 的影响（任务书附带问题：若不能则调整候选）

- 小 chunk 答案是**「能」** ⇒ **不触发**「B1 静态公平基线须调整小 chunk 候选」这一条款。
- 用户裁定：**保持现状**，B1 候选集是否纳入 small chunk 由 B1 自己的 5 个待裁定问题决定，
  **不在本会话越权决定**。

### 五、变更与验证

- **新增**：`configs/cost_calib/shape_grid_v8.yaml`、`scripts/gen_shape_grid_v8.py`（含 `--check`）、
  `scripts/analyze_b4_small_chunk.py`、`scripts/_b4_inspect.py`；
  `results/raw/_probe/20260928/costcalib_v8_s{1,2}.json`、`results/raw/_costcalib/costcalib_v8_s{1,2}.json`；
  `results/processed/costcalib_eval_v8.json`、`b4_small_chunk_v8.json`、`b4_small_chunk_v7.json`。
- **验证命令与结果**：
  * `python scripts/costcalib_grid.py --grid configs/cost_calib/shape_grid_v8.yaml` → **0 个问题**，
    `mixed_small_chunk` **24 点**（v7 是 16）；
  * `python scripts/gen_shape_grid_v8.py --check` → 通过（pf192 恰 8 点、比 v7 多 8 点）；
  * 远端 `check_cpu_stability.py 7 1.0` → **rc=0（0 稳定）**，稳健 CV 0.007%；
  * 远端两会话 `complete: true`、24/24 点、`grid_sha256 3473d023e269706f`；
  * 回拉后 **md5 远端==本地**（`1aeba54e…`/`9964dc2d…`），CR=0；
  * `python scripts/analyze_b4_small_chunk.py --eval ..._v8 --artifacts "results/raw/_costcalib/costcalib_v8_s*.json"` → 综合判定"**能**"；
  * **否定测试**：① 删光小 chunk 记录 → **rc=2**「没有小 chunk 记录」（不伪装成"能"）；
    ② `--artifacts` 指向不存在路径 → **显式告警**且血缘字段为 `None`（**不拿模型版本号顶替**）；
  * 全量 `pytest tests/ -q` → **1355 passed / 39 skipped / 732 subtests passed / 0 failed**；
  * `git diff --stat -- results/raw/` **为空** ⇒ **只读区未被触碰**（v8 文件是新增，非改动）。

### 六、本会话自查到的两个问题（已就地修）

1. **产物血缘字段一度取自模型文件而非数据** —— v8 数据配 v7 模型，产物初次落盘时
   自称 `protocol_version=cost-calib-v7` / `grid_sha256=5c3b66f75c89522c`，
   **会让"这份结论是对哪批采样说的"无法自证**。
   实测还发现 `costcalib_eval_*.json` **根本不含**这两个字段 ⇒ 权威来源只能是
   **原始标定产物**。已改为 `--artifacts` 现读 + 缺则 `None`，并加否定测试钉住。
2. **`pull` 报 OK 后仍须复核落盘** —— 按既定纪律量了 md5 与 CR，未只信"OK"字样。

### 七、下一步

- **B1 仍未可冻结**：其 §10 的 5 个评审问题待裁定；且草案 §1.4 的 **384 仍是缺口**
  （本会话只关闭了 128/192/256）。
- ⛔ **不得**开始 `OPT-04`（收益实验，依赖 B1–B4 全部就绪）。
- 关联：`OPTIMIZATION_TEST_SESSION.md` §0/§3、`OPTIMIZATION_ROADMAP_SESSIONS.md` §5「B4」、
  `EVIDENCE_INDEX.md` §3 缺口 3、`COST_PROFILE_S4000.md` §6、`NEXT_SESSION_TASKS.md`「B1 预注册草案」§1.4/§9.1。

## 2026-09-28 — OPT-03R：B1 预注册草案 5 个评审问题裁定

- 会话：`OPT-03R`（承接 OPT-03；`OPTIMIZATION_TEST_SESSION.md` §2C）。
- 动作：裁定 `NEXT_SESSION_TASKS.md` 文末「B1 预注册草案」§10 的 5 个评审问题，
  并把 OPT-03（B4）的结果**回灌**草案中因此过时的三处。
  **未跑真机、未改 `results/raw/**`、未改 `src/`。**

### 一、★ 裁定前先把三处过期信息更正（否则按错信息裁定）

| 位置 | 原文 | 更正为 |
| --- | --- | --- |
| 草案 §1.4 结论 2 | 方案 B **做不了**（模型对小 prefill 无预测） | ★ **可行**——128/192/256 已可预测；但 `384` 与 §9.5 未闭 |
| 草案 §9.1 | 缺口 3 **未闭合** | ★ **部分闭合**——`pf ∈ {128,192,256}` 有键（`384` 仍缺） |
| 草案 §9.5 | "中间点被选中"未证明 | ★ 补注：候选 192 已可预测 ⇒ 堵点只剩**选择规则** |

### 二、★ 五条裁定

1. **方案 A（双预算）+ 同批跑单预算版本归因**。
   理由：单预算与"整块 prefill 步"在构造上不相容（≥512-token prefill ≈135 ms 必然超
   120 ms），这是**四臂共有**的缺陷（S14B2 实测四臂 `n_degraded≈150`）；同批归因是
   因为"改 deadline 语义"属单变量改动，须有同批对照才能归因。
   ⚠️ 代价：机时翻倍（40 run ≈16 h）。
2. **0.3–0.7 达标率是"目标"，不是判据**。只写死"可达性前置检查"（不饱和、不全达标）；
   不达标照跑照记。理由：写成硬判据会诱使朝数字调负载 = 制造 oracle（违反 §6 规则 6）。
3. **两条静态基线足够，不加 `static_chunk@1024`**。理由：基线设计目标是覆盖自适应臂的
   **动作范围**（`static_nochunk` + `static_small`），`@1024` 不扩范围、且是 S14B2 的
   **异血缘**。要历史锚就**引用**并注明不可并表，不重跑。
4. **用闭环补位**处理"确定性负载 ⇒ 无 CI"。理由：只报逐重复值时，§6 规则 3②
   "≥4/5 同号"会**恒真或恒假**而失效；闭环补位让批边界随调度改变 ⇒ CI 有信息量。
   ⚠️ 改执行路径 = 红线 ⇒ 预注册须写明并重核 `git_commit`/`digest`。
5. **先跑便宜替代**（1.5–2.5 h：「双预算 × {cost, threshold}」小批），再决定是否花整夜。
   理由：本格 20 run≈8 h **零余量**（一次中断报废整夜），A+归因后 40 run 单夜放不下，
   而 §9.5 未闭 ⇒ 现在起跑可能重演 S6 的"测不出差异"。该替代的产出**正是 9.2/9.5 的闭合证据**。

### 三、★ 自查出的机时冲突（未掩盖）

草案 §7 写「20 run ≈8.0 h 刚好一夜、无余量」；Q1 裁定 A+归因后 **40 run ≈16 h**。
⇒ **必须显式二选一**：**(a) 重复降到 3**（24 run ≈9.6 h，仍略超）或
**(b) 分两夜、明确两夜不得并表**。**不得**不写就按 16 h 起跑。
⇒ 留待便宜替代出结果后与 9.2/9.5 一起再裁。

### 四、变更与验证

- 变更：`NEXT_SESSION_TASKS.md`（§1.4 更正框 / §9 表 4 行 / §10 加 10.1–10.2）；
  `OPTIMIZATION_TEST_SESSION.md`（§0 加 `OPT-03R` 行 / §2C 执行记录）。
- 验证：两份文件 `tr -cd '\r' | wc -c` = **0**（LF）；五条裁定 grep 各命中；
  §10.1/§10.2 锚点行 5566/5576 在位。**未动 `results/raw/**`、未跑真机。**

### 五、结论与下一步

- **B1 仍不可冻结**：`384`（§9.1）+ §9.5 两处未闭；机时安排（16 h）待定。
- **可立即执行的下一步** = 便宜替代的**离线前置**：`slo_reach` 负载定义 +
  双预算 deadline 的纯逻辑层实现与单测（不碰真机），再起 1.5–2.5 h 小批。
- ⛔ **不得**开始 `OPT-04`。
- 关联：`EVIDENCE_INDEX.md` §3 缺口 3（已改判"部分关闭"）、`OPTIMIZATION_ROADMAP_SESSIONS.md` §5「B4」。

## 2026-09-29 — 执行优化线 **S00**：状态核验、阻断项证据与基线冻结

- 会话：`S00`（`handoff/optimization_20260929` 的第一会话；执行优化线 S00–S11，
  **与论文线 OPT-01..07 编号互不干扰**）。
- 起点：`STATE.json` = 全部 `NOT_STARTED`、无已验收恢复点 ⇒ 按 `00_START_HERE.txt` 规则从 S00 开始。
- 性质：**只做本地静态审查 / 纯逻辑修复 / CPU 测试**。真机项 = `REMOTE_PENDING`。
  **未跑真机、未改 `results/raw/**`、未产生新实验条件、未动执行路径的到达语义。**

### 一、仓库与快照核验

- HEAD = `16e82d2ea117ee4017adf3d0a716e81df12bf0e4`（分支 `master`），
  与 `SOURCE_SNAPSHOT.json` 的 `16e82d2` 一致。
- 工作树**不干净**：15 个已跟踪文件被改 + 18 个未跟踪
  （双预算 / 闭环抖动 / `gate_mode` / v8 网格 / 三份新文档 / `handoff/`）。
  **全部按原样保留，未回滚、未重置**（共同要求 1）。
- 唯一写入者：改前实测各目标文档 mtime 最新为 2026-09-29 16:49（前一会话），
  本会话 19:46–19:48 才写 `src/ascend_opt/{cost_model,slo_weave}.py` ⇒
  **本会话是唯一写入者**，无并发写者。

### 二、四列状态（执行优化线；来源 = 源码 + `EXPERIMENTS.md` + `EVIDENCE_INDEX.md`）

| 机制 | 已实现 | 离线已验 | 真机已测 | 可作结论 |
| --- | --- | --- | --- | --- |
| **P1** 末位置 logits / 按需采样（S01） | ❌ 无 | ❌ | ❌ | ❌ |
| **P2a** retire 与 allocator 清理分离（S02） | ❌ `retire()` 仍**每次摘槽即 `musa.empty_cache()`**（`benchmark/real_backend.py:1374-1378`）| ❌ | ❌ | ❌ |
| **P3** QKV / gate-up 线性融合（S03） | ❌ | ❌ | ❌ | ❌ |
| **P4/P5/P6/P8**（S04–S09） | ❌ | ❌ | ❌ | ❌ |
| **P7** 执行形状/分块成本感知调度（S07） | 🟡 已有 `cost`/`slo_weave` 臂与双预算开关（未提交） | 🟡 单测覆盖；机制未在真机验证 | 🟡 `S14B2/S14B3` 有四臂产物，但 `adaptive` 臂不可用 | ❌ |
| 成本模型控制器（论文线既有） | ✅ | ✅ | ✅ | ✅ 仅 C1/C2（见 `EXPERIMENTS.md`）|

⇒ 执行优化线**三项优先方向（P1/P2a/P3）目前都只是"未实现"**，
不是"测了没收益"。S00 未开始它们的实现。

### 三、★ 阻断项核验结果（`01_PROJECT_PLAN.md` §5）

证据脚本：`scripts/audit_s00_blockers.py`（只读）→ `results/processed/s00_audit.json`。

| # | 项 | 实测结论 | 处理 |
| --- | --- | --- | --- |
| §5.1 | 双预算 decode 保护语义 | **仍存在**：`_budget_for` 只按候选自己的 `tokens_prefilled>0` 归类。实测 12×2048-token 等待局面下 **6** 个候选被 prefill 预算放行、而按 decode 预算不可行 | **只登记不改** ⇒ 修它 = 定义"逐 token 护栏"**新策略语义**，属 S07 预注册（特征化用例 `tests/test_s00_known_gaps.py`）|
| §5.2 | `closed_jittered` 未来 release | **仍存在**：`scheduled_arrival = step_start + jitter` 是未来时刻，但请求**仍在 `step_start` 入队** ⇒ 实测 **16/16** 请求 `enqueue_ms < arrival_ms`（入队早于到达）| **只登记不改** ⇒ ① 真机路径上"实际等待" = 真的空转 GPU；② 抖动按 `step` 派生而 `step_start` 在移动 ⇒ "绝对放行时刻"无定义。语义须先在 S07 预注册裁定（Q4 明写"改执行路径须重核 digest"）|
| §5.3a | SLOWeave **重复应用** `safety_margin` | ★ **确认存在并已修**：`_predict_ms` 未传 `apply_margin=False`，而 `CostModel.predict` 默认 `True` ⇒ 判据乘 `(1+m)²`。实测 `model_v7` `m=16.02%`：**1.1602 → 1.3461** | **已修**（`slo_weave.py`）+ 反例用例 |
| §5.3b | 稀疏 lookup 上做连续二分 | ★ **确认存在并已修**：缺键返回 `None`（=未知）被当成"不可行且更大者全不可行"。实测 `b1/kv1024`：`L=1024/预算400ms` ⇒ 旧二分给 **512**（真值 1024，欠选一半）；`L=1536/预算320ms` ⇒ 旧二分给 **0（塌缩 decode-only）**，而 `c=512` 只需 194.4 ms | **已修**（枚举"有键候选"取最大可行者）+ 反例用例 |

- 阻断范围（按 `01_PROJECT_PLAN.md` §5 末段）：§5.1/§5.2 属**策略语义** ⇒ 阻断
  **新策略对比**（S07/OPT-04），**不**阻断 S01–S03 这些独立纯逻辑实现。
- ⚠️ **§5.3 修复改变了 `slo_weave` 臂是什么** ⇒ 既有含该臂的产物
  （`EXP-S14B3-006..010`，`lenmix_slo_weave`）**是修复前口径** ⇒
  **不得与修复后的 `slo_weave` 并表**（改执行路径 = 新实验条件）。

### 四、模型导出与候选覆盖（§5 之外，S00 要求）

- `configs/cost_calib/` 下**只有 `model_v1..v7.json`**，**无 `model_v8.json`**
  ⇒ v8 探针数据（24 点 `mixed_small_chunk`，`protocol cost-calib-v8`）**未导出为部署模型**。
- 部署模型 `model_v7.json`：`default_mode = lookup`、`protocol = cost-calib-v7`、
  `grid_sha256 = 5c3b66f75c89522c`、107 键、`safety_margin = 16.02%`；
  `prefill_tok` 直方图 = `{0:40, 128:8, 256:8, 512:25, 1024:25, 2048:1}`
  ⇒ ★ **`pf192` 不在部署表内**（与 `EVIDENCE_INDEX.md` §3 缺口 3 的注一致）。
- ⇒ "v8 已采样" ≠ "192 可被部署查表"；要让 192 进部署表须补采 v8 网格其余 95 点后导出。

### 五、基线协议（冻结）

- 权威平台 = 远程 MTT S4000（MUSA）；本机 `local` **永不**产生 `MEASURED`。
- 平台基线 `configs/mthreads/s4000_fp16.yaml`：FP16、单卡、`graph_mode: eager`、
  `tensor_parallel_size: 1`、`max_num_seqs: 32`、分块/缓存/自适应**全关**。
- 外部 SLO：`ttft_ms = 2000`、`tpot_ms = 100`（逐请求判据）。
  ⚠️ 配置里的 `controller.target_tpot_ms: 50.0` 是**控制器内部目标**，**不是**外部 SLO。
- ⚠️ **发现的冻结缺口**：随仓库的基线配置写 `runtime.model_revision: "main"`，
  **不是**交接记录的 pinned revision `a09a35458c702b33eeacc393d103063234e8bc28`
  ⇒ 新执行优化线须用**新配置**钉死 revision，**不**原地改历史配置
  （改配置 = 新实验条件）。
- 提示：`--dry-run` **不能**验过载、SLO、判据（不喂 `decode_samples`）⇒ dry-run 只证链路可跑。

### 六、验证命令与结果

```bash
python -m pytest tests/test_slo_weave.py tests/test_slo_weave_sparse_lookup.py \
                 tests/test_s00_known_gaps.py tests/test_cost_policy.py \
                 tests/test_cost_policy_dual_budget.py tests/test_controller_gate_mode.py -q
python scripts/audit_s00_blockers.py --json results/processed/s00_audit.json
python -m pytest tests/ -q            # 全量
```

- 定向：`test_slo_weave*.py` = **27 passed**；`test_s00_known_gaps.py` = **4 passed**。
- **破坏验证**：临时还原旧行为（`apply_margin` 缺省 + 禁用枚举）⇒ 新用例
  **5 failed / 8 passed**，恢复后 **27 passed** ⇒ 反例有效（不是恒真）。
- **全量回归**：`PYTHONPATH= python -m pytest tests/ -q`
  ⇒ **1433 passed / 1 skipped / 746 subtests passed / 0 failed / EXIT=0**（11:35）。
  ⚠️ **必须清 `PYTHONPATH`**：否则 WorkBuddy 的 safe-delete 垫片会让汇总行消失、退出码变 1
  （**假回归**，判据见 `st/.workbuddy/memory/2026-09-29.md`）。
  ⚠️ **可解释性核对（不得只看"通过"）**：前一会话记录 = 1378 passed / 39 skipped，
  但那次用的是**没有 torch 的 venv 解释器**；本次用 `versions/3.13.12`（有 torch）⇒
  38 例此前 `skip` 的用例转为**实跑**。`1378 + 38 = 1416`（同解释器基线）
  `+ 17`（本会话新增 17 例：13 + 4）`= 1433` ✅ 与实测吻合 ⇒ **无既有用例增减/回归**。
- `audit_s00_blockers.py` `rc=0`，摘要：
  `dual_budget.relaxed_by_prefill_budget=6`、`jitter.enqueue_before_arrival=16`、
  `jitter.arrival_span_ms=15705.27`、`slo_weave.margin_once=true`、
  `slo_weave.collapse_gone=true`、`model.pf192_in_deployed=false`。
- **未触碰**：`benchmark/real_backend.py` / `benchmark/run_benchmark.py` mtime 均为
  2026-09-28 23:2x（**早于**本会话 19:43 开始）⇒ 工树里它们的 ` M` 是**前一会话**的未提交改动；
  `git status --untracked-files=no -- results/` **为空** ⇒ 已跟踪的只读产物未被修改。

### 七、远程

- 本轮**最小远端可达性核验失败**：`remote_run.py exec` 返回
  `SSHException: Error reading SSH protocol banner`（重试一次同样失败）。
  ⇒ 本会话全部真机项标 **`REMOTE_PENDING`**。
- 需要的最小外部条件（一次列清）：① 实例开机/控制台确认运行；② SSH 可达；
  ③ 剩余机时额度（交接包 `STATE.json` 里为 `null` ⇒ **未知，不按 0 也不按无限**）；
  ④ 真机前按既有前置跑 `check_cpu_stability.py`（三态）。
- **不把缺硬件判为方案无效**：本地实现与测试不受影响。

### 八、限制与未决（不掩盖）

1. §5.1/§5.2 两项**只有证据、没有修复**：它们的修复属**新策略语义**，
   须在 S07 预注册里裁定（含"绝对放行时刻"如何定义、是否真的让 GPU 空转）。
2. §5.3 的修复**未在真机验证**；它改变了 `slo_weave` 臂的判据**倍率**
   （1.3461 → 1.1602）⇒ 该臂的**真机行为会变**，须在新实验条件下重跑才能引用。
3. S00 的 `closed_jittered` 证据来自 **dry-run 模拟器**，不是真机时序
   ⇒ 它证明的是**代码语义**（入队早于到达），不是真机现象。
4. 执行优化线 P1/P2a/P3 **尚未开始实现** ⇒ 本会话对"能否降低推理成本"**不给任何结论**。
5. `model_revision` 未 pin 的缺口若要动，须新建配置（新实验条件），不在 S00 内改历史配置。

### 九、下一步

- 进入 **S01（P1 输出层冗余：末位置 logits + 按需采样）**：先核对
  `benchmark/real_backend.py` 的前向入口（约 1214 行）与已验证的 `_sample` 契约，
  两个开关各自独立、默认关闭、可单独消融；本地用 CPU 微型模型验数值，
  真机项标 `REMOTE_PENDING`。
- 关联：`OPTIMIZATION_TEST_SESSION.md` §5、`NEXT_SESSION_TASKS.md` 文末
  「执行优化线 S00 完成记录」、`EVIDENCE_INDEX.md` §3（新增两行缺口）、
  `handoff/optimization_20260929/STATE.json`。

## 2026-09-29（续）— 执行优化线 **S01 / S02 / S03**：本地实现与正确性验证

- 会话：`S01`（P1 输出层）→ `S02`（P2a allocator 生命周期）→ `S03`（P3 线性融合）。
- 性质：**纯本地**（Windows，CPU 微型模型）。**未跑真机、未改 `results/raw/**`**、
  未产生新实验条件（所有开关**默认关闭** ⇒ 既有行为逐字节不变）。
- 权威记录：`OPTIMIZATION_TEST_SESSION.md` §5.2/§5.3/§5.4。

### 一、四列状态更新

| 机制 | 已实现 | 离线已验 | 真机已测 | 可作结论 |
| --- | --- | --- | --- | --- |
| **P1** 末位置 logits / 按需采样（S01） | ✅ | ✅（12 用例） | ❌ `REMOTE_PENDING` | ❌ |
| **P2a** retire 与 allocator 清理分离（S02） | ✅ | ✅（7 用例） | ❌ | ❌ |
| **P3** QKV / gate-up 融合（S03） | ✅ | ✅（16 用例） | ❌ | ❌ |

⚠️ **"已实现 + 离线已验" ≠ "有收益"**。三项的**收益结论一律待真机**；
本条线的下一步不是"再写代码"，而是"环境可用后按 §5 的 A/B 配置测"。

### 二、★ S03 实测到的两条机制事实（不预设正收益，但这条影响判断方向）

1. **两种融合的切分代价不对称**：
   - **QKV 必须 `contiguous()`**：HF 的 attention 对 q/k/v 做 `.view(...)`，
     而"从融合输出切出的窄片"**不是连续张量**（行 stride 仍是融合宽度）。
     实测每层每次融合产生 **3** 次拷贝（字节数落盘在 `bytes_copied`）。
     ⇒ 省下的 2 次 launch 要**与 3 次拷贝相抵**，净收益**不保证**。
   - **gate/up 不需要 `contiguous()`**：下游是 SiLU 与逐元素乘法，非连续张量天然可用
     ⇒ `split_copies = 0` / `bytes_copied = 0`。
   - ⇒ 若要在 S03 里挑一个先测，**gate/up 的前景明显更好**（机制上无附加拷贝）。
2. **复用必须靠"双条件"守卫**：QKV 融合靠"q 先算、k/v 复用"省下两次前向。
   若只判"有没有桩"，一旦 k/v 收到**不同**的输入张量就会**静默返回陈旧结果**
   （数值错、形状对）。故复用同时校验 `(data_ptr, shape)` 与**剩余消费计数**，
   任一条不成立即回退原 `F.linear`。破坏验证：去掉守卫 ⇒ 反例用例**立刻失败**。

### 三、S01 的一条实现纪律（可复用）

**"带默认开着的优化参数，其可用性必须在装载期判定，并落盘 `requested` vs `effective`。"**
`num_logits_to_keep` 在旧 transformers 上不存在；只传不判的后果是：
配置写着"已开末位置优化"、产物也这么记，而实际一步都没走到 —— 一次**未优化**的 run
会被读成优化 run。本实现三态处理（支持 / 不支持 / 判不了就真探测一次），
并把 `unsupported_reason` 写进产物。

### 四、变更与验证

- **变更**：`benchmark/real_backend.py`（6 个开关 + 装载期判定 + `flush_allocator_cache()`
  + `execution_optimizations` 落盘）；**新增** `benchmark/linear_fusion.py`、
  `scripts/gen_exec_opt_configs.py`、`scripts/inspect_linear_shapes.py`、
  `configs/mthreads/exec_opt/*.yaml`（8 份，派生、revision 钉死）、
  `tests/_tiny_support.py` 与三个 `tests/test_optim_s0{1,2,3}_*.py`（35 用例）。
- **验证命令与结果**：
  * 定向：`pytest tests/test_optim_s0{1,2,3}_*.py tests/test_real_backend*.py -q`
    ⇒ **102 passed / 49 subtests passed / 0 failed**。
  * **破坏验证（三次，逐项）**：S01 开关失效 ⇒ **5 failed**；S02 无视开关 ⇒ **3 failed**；
    S03 去掉复用守卫 ⇒ **1 failed**（报"静默算错"）。恢复后均全绿。
  * 8 份派生配置：`gen_exec_opt_configs.py --check` **通过**；随仓库配置加载闸门
    （`test_all_shipped_configs_load`）**通过**；`build_driver` 接受全部新键。
  * `scripts/inspect_linear_shapes.py --tiny` 自检通过。
- **未触碰**：`results/raw/**`；`benchmark/real_backend.py` 的既有行为（默认开关全关，
  既有 67 个后端用例全绿）。

### 五、限制与未决（不掩盖）

1. **全部真机项 `REMOTE_PENDING`**（远端不可达）：S01 的 TTFT / prefill 时间 / 显存；
   S02 的 `allocated`/`reserved` / 分配重试 / OOM / ITL / 端到端吞吐；
   S03 的 M=1/2/4/8/128/512 算子耗时与净收益、真实 7B 的
   `3584/512/512 → 4608` 与 `18944×2 → 37888` 实测核对。
2. ⚠️ **不得**把"输出位置数减少 k 倍"写成"端到端快 k 倍"：减少的是**输出层重复投影**，
   不是整个 Transformer 的计算量。
3. ⚠️ S02 的 **`reserved` 变高不算失败**（缓存复用的预期表现）；判据是"活跃内存不无限增长 + 无 OOM"。
4. ⚠️ 本仓库**没有**内存压力控制器 ⇒ "如何区分 allocated 与可复用 reserved"
   目前**无自动判据**，只能靠 `sample_device()` 直接读数。记为待办。
5. **执行路径已被改动**（新增 6 个开关）⇒ 按红线这是**新实验条件**：
   既有含 `slo_weave` 臂的批次本就不能与修复后并表（S00 已记），
   而**任何**执行优化线的真机批次都必须与 S00 冻结的基线同批对照。
6. S03 的 `state_dict` 逐键相同只能证明"权重未被改"，**不能**证明融合在真机上数值一致
   —— 那需要 MUSA 上的前向对比（`REMOTE_PENDING`）。

## 2026-09-29（续二）— 执行优化线 **真机运行脚本**（自动化 / 跑满 / 少干预）

- 会话：`EXEC-RUN`（跨会话交付；权威记录 `OPTIMIZATION_TEST_SESSION.md` §5.5，
  运行清单 `handoff/optimization_20260929/EXEC_OPT_RUNBOOK.md`）。
- 性质：**纯本地**编码与验证。**未跑真机**（远端仍不可达）、未改 `results/raw/**`。

### 一、交付

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 纯逻辑 | `benchmark/exec_opt_plan.py` | 穿插顺序、均衡重复停机判据、进度/心跳载荷、原子写、臂解析 |
| 远端自驱动 | `scripts/run_exec_opt_matrix.py` | 无人值守跑臂×重复；30 s 心跳；STOP 协作停止；预算填充；断点续跑；按**产物**判成功 |
| 本机一键 | `scripts/exec_opt_batch.py` | `plan/preflight/run/status/stop/pull`；核验→推送→单次起跑→轮询→回拉 |
| 安全阀 | `scripts/clear_stale_batch_lock.py` | 清理**陈旧**批次锁（四条条件，默认只报告） |
| 清单 | `handoff/.../EXEC_OPT_RUNBOOK.md` | 三条命令 + 失败判据表 + 红线 + 已知边界 |
| 小改 | `scripts/push_files.sh` | 新增 `--include`（默认 `py,sh` 行为不变）⇒ 配置也能被推送 |

一条命令：`python scripts/exec_opt_batch.py run --tag <T> --stage screen --budget-minutes 240`。

**复用而非重写**：`launch_batch.sh`（锁/setsid/监视/激活）、`push_files.sh`（逐文件 sha）、
`pull_incremental.py`（增量+完整性）、`check_remote_busy.py`、`check_cpu_stability.py`、
`diag_remote_reachability.py`、`run_step3_matrix.read_run_status`（产物判定）。

### 二、"跑满"的定义（不是"能塞多少塞多少"）

默认 `--fill-budget`：**整轮整轮**跑到预算不够为止；预算不足跑完当前一轮即**干净收尾**
（宁可少一整轮，也不出现"有的臂 5 次、有的臂 4 次"）。收尾显式检查 `balanced`。
`--min-repeats 3` 是不低于 3 次的下限；实测节拍由第一批自动测出并用于后续停机判据。

### 三、★ 本会话发现的一个**新失效**：陈旧批次锁

本地实跑 `launch_batch.sh` **全链路**（含 `--activate` 激活层）时发现：
批次正常结束（`BATCH_LAUNCH_OK`）后**锁没有被释放**。
根因是**本机 Git for Windows 没有 `setsid`** ⇒ 监视进程随父 shell 被整组杀掉
（`launch_batch.sh` 的文档已写明这条边界）。远端有 `setsid`，行为不同；
但它证明了一条真实路径：**监视进程死掉 ⇒ 锁永留 ⇒ 之后所有批次都起不来，且不报错**。
无人值守场景下这是最贵的失效 —— 不是跑错，是"什么都不跑"。

⇒ 新增 `clear_stale_batch_lock.py`，删除须**同时**满足：① 只删锁根下 `*.lock.d`；
② 记录的 pid **确已不存在**；③ 年龄 > 10 min（防"另一批次刚拿到锁"的竞态）；
④ 调用方已确认远端空闲（编排器把它排在 `check_remote_busy` **之后**）。默认只报告。

### 四、★ 自查到的一处**真错**（与"想象的夹具"同类）

清理器第一版把锁目录里的 `holder` 当成"标记文件"跳过，而**真实布局**是记录就写在
`holder`（`batch_lock.py:169`）⇒ 扫描恒得 `tag=None, pid=None` ⇒ 清理**永不触发**，
而报告长得像"没有陈旧锁"。
根因：**夹具按想象造**（`{"tag":…}.json`），假样本让错解析器通过了测试。

修法三条：① **复用** `batch_lock.BatchLocks.holder()`（唯一实现，不再自己写解析）；
② 夹具按真实布局写 `holder`；③ 加一条**用真实 `batch_lock.py` 取一次锁再读它**的用例。
同处还修了平台差异：不存在的 pid 在 Linux 抛 `ProcessLookupError`、
在 **Windows 抛 `OSError(winerror=87)`**（实测）⇒ 只捕前者会让清理在 Windows 上永不触发。

### 五、验证与结果

- `tests/test_exec_opt_plan.py` **25 passed**；`tests/test_clear_stale_batch_lock.py` **14 passed**。
- **`launch_batch.sh` 全链路本地实跑**：激活生效（`ACT=yes`）、锁已取、`BATCH_LAUNCH_OK`、rc=0。
- **清理器对真实 `batch_lock.py` 产物**：识别并删除两个卡死锁（pid 已不存在）。
- **破坏验证**：还原错解析器 ⇒ **5 failed**，恢复 ⇒ 14 passed。
- **全量回归**：`PYTHONPATH= pytest tests/ -q` ⇒
  **1507 passed / 1 skipped / 756 subtests / 0 failed / EXIT=0**。
  可解释性：1468（上次同解释器）+ **39**（本会话新增 25+14）= 1507 ✅；skip 与 subtests 不变。
  ⚠️ 本轮耗时 51:47（上轮 12:25）——**成因是自己并发起了第二个 pytest** 抢资源，
  不是用例变慢。教训：同一台机上**不要**并发跑两套全量测试。

### 六、限制与未决

1. **远端不可达** ⇒ 运行脚本**未在远端实跑过**；以上是本地组件级实跑 + 离线路演。
2. **"批次结束自动放锁"本地验不了**（缺 `setsid`）⇒ 只能到远端确认；已用安全阀兜住后果。
3. `--budget-minutes 240` 是默认值，**不是测量结果**；真实节拍靠第一批实测。
4. 交接包 `STATE.json` 的额度字段仍为 `null`（未知）。

### 七、下一步

- 环境恢复后：`exec_opt_batch.py preflight` → `run --stage screen`；
  有信号再 `--stage full`。**收益结论仍一律待真机**。

---

## 2026-09-30 — 执行优化线：远端恢复 + EXECOPT2 静默少跑一轮（纯逻辑 bug）+ EXECOPT3 起跑

### 一、远端恢复与排障（七个坑，逐一记成因）

用户给出 SSH 凭据（`SSH_HOST_FROM_PRIVATE_CONFIG:28080`，与 `remote.env` 既有值一致）。
连上后**连环踩了 7 个坑，每个都"看起来成功"或"看起来是别的问题"**。完整清单见
`st/.workbuddy/memory/2026-09-30.md`；此处只记**成因可复用的三类**：

1. **间歇性 SSH 握手**：`paramiko` 报 `Error reading SSH protocol banner`。
   裸 TCP 探测 1/5 成功、4/5 连上后**零字节**，但原生 socket 连 8 次**全成功**
   ⇒ 数据面抖动，**不是链路死**。`remote_run.py::_connect` 的既有 4 次重试**正确吸收**了它。
   ⚠️ 不要据"banner 失败"改远端或换凭据 —— 先重试。

2. **`bash -c` 会重置 PATH（★ 最贵的一个）**：`launch_batch.sh --activate` 确实执行了
   （日志有 `[activate_musa] venv=...`），但**承载真命令的内层 `bash -c` 重新加载 profile**，
   把 venv 的 `bin` 从 PATH 头部挤掉（→ `/root/miniconda3/bin`）⇒
   真机报 `ModuleNotFoundError: No module named 'transformers'`（EXECOPT1 批次即死于此）。
   ⇒ 修：`Ctx.venv_python` 用**绝对路径** `/root/musa-envs/qwen310/bin/python3`；
   ★ 并加**起跑路径守卫** —— 核验**必须用与起跑同形态的调用**，否则"核验通过、起跑炸"。

3. **"远端 stat"被当成"本地文件大小"**：日志 `OK real_backend.py (113287 B)` 里的 113287
   是**远端旧文件**的 stat（本地新版 130125 B）⇒ 假校验通过，远端停在无开关的旧版。
   ⚠️ 校验"两边一致"时两个数**必须同口径**；别把回显当基线。

另有 4 个坑属"引号/编码/转义"家族（heredoc 经 `bash -c` 被压成一行、
手写引号层级被吃、`--include py,sh,yaml` 恒匹配 0 文件、裸 `bash` 解析到 WSL 被安全策略拦），
修法一律为**机制化**：`shlex.quote` 生成载荷 / `FIND_ARGS` 数组化 /
`_find_git_bash()` 显式定位 PortableGit / `_decode()` 加 UTF-16 分支。

### 二、★★ 新失效：EXECOPT2 **静默少跑一轮**（8/12）

症状：`failures=[]`、无任何报错、`n_done=8`、`completed_passes=2`（应 3）。
根因在 `scripts/run_exec_opt_matrix.py`：**同一个 `total_planned` 既做报告量、又做停止判据，
且每轮被覆写**。

| 位置 | 代码 | 语义 |
| --- | --- | --- |
| L264 | `total_planned = len(cells) * repeats` | ✅ 正确的计划总数（=12） |
| L372（已删） | `total_planned = len(cells) * (pass_index + 1)` | ❌ 每轮被覆写：r0 后→4、r1 后→8 |

于是 r1 结束 `decide_stop(items_done=8, items_total=4)` ⇒ `8 >= 4` ⇒ 判「全部项已完成」⇒ **停**。
⇒ 这**与硬件无关，是纯逻辑 bug**，且**不报错** —— 是"报告量污染判据"这一类失效。

**修**：删 L372 赋值（留注释"计划顺序不可变"）。
**回归**：新增 `TestMultiPassProgression`（4 用例 / 含 2 subtests）。
**破坏验证**：`items_done >= items_total` → `>` ⇒ **3 failed**（含新增 2 条）⇒ 测试真能抓；
恢复后 **29 passed**；三文件 CR 计数 = 0。

★ **可复用规则**：**报告用的派生量与判据用的阈值必须来自不同来源**。
同一个变量一旦既描述"计划"又随进度被改写，停止判据就会随进度**自毁**，且静默。

### 三、EXECOPT2 产物处置（已裁定）

EXECOPT2 只跑 2 轮，跨轮次**条件漂移** ⇒ **不得**用"2 轮 + 续跑 1 轮"拼成 3 轮并表
（每轮 3 min、GPU 热态不同）。⇒ 用**新 tag `EXECOPT3`** 重跑**完整 12 项**。
**EXECOPT1（失败批次）/ EXECOPT2（2 轮）产物一律不作结论**，仅留反例。

### 四、EXECOPT3 起跑

`preflight` 第一次因间歇握手失败，**重试后全绿**：
```
[ok] 起跑路径守卫：经 bash -c 用 /root/musa-envs/qwen310/bin/python3 仍可 import（与起跑同形态）
[ok] revision 快照在位：a09a35458c702b33eeacc393d103063234e8bc28
IDLE：远端无基准进程 / [stale-lock] 没有锁 ⇒ 无陈旧锁
稳健 CV : 0.005%  ✓ 稳定（≤1%）⇒ 计时环境可用 ⇒ 可以起跑
```
命令：`python scripts/exec_opt_batch.py run --stage screen --tag EXECOPT3 --budget-minutes 240`

**中途抽样（r0 完成）**：完成行见
`baseline#r0 3.0 min` / `s01_both#r0 2.8` / `s02_no_empty_cache#r0 2.9` / `s03_fuse_gate_up#r0 3.0`
⇒ r0 四项全成功、**三个开关均可执行**；随后进入 `baseline#r1`。4 项均 ~3.0 min ⇒ 12 项预计 ~36 min。
⚠️ 进度文件 `n_done` 在轮间刷新有滞后（抽样时显示 0），**判断进度以完成行为准**。

### 五、限制与未决

1. **EXECOPT3 尚未跑完** ⇒ 本条目只记**机制能否执行**（已见三开关均执行）；
   **收益结论一律待 12/12 完成 + 回拉 + `EXPERIMENTS.md` 登记**。
2. **EXECOPT1/EXECOPT2 产物不作结论**（前者环境失败、后者少一轮）。
3. `--budget-minutes 240` 仍是默认值，**不是测量结果**；真实节拍由 EXECOPT3 实测（~3.0 min/项）。
4. `exec_opt_batch.py` 的推送快速路径与引号修复，本机全量回归**本轮补跑**（见下）。

### 六、下一步

- EXECOPT3 → 12/12 ⇒ `pull` ⇒ `EXPERIMENTS.md` 登记 4 臂对照（**仍不写收益结论**，
  只登记"四臂在同一协议下各跑 3 轮"）。
- S04/S05/S06 需先**冻结执行路径**；S07 待预注册裁定 §5.1/§5.2。
- `OPTIMIZATION_TEST_SESSION.md` 的 S0x 条目与本轮的 `exec_opt_batch.py` 变更待补。

---

## 2026-09-30（续） — 执行路径冻结 + S04 实现（本地）+ S05 探针（真机，结论为负）

### 一、执行路径冻结（依 EXECOPT3 现读复算，不引用记忆值）

轮询复算四臂 × 3 轮（从 `results/raw/EXECOPT3-short_chat-*/meta.json` **现读**）：

| 臂 | ttft_p50 mean | tpot_p50 mean | vs 基线 |
| --- | --- | --- | --- |
| `s4000_fp16` | 196.988 | 118.181 | — |
| `s01_both` | 188.769 | 118.150 | TTFT **−4.17%**、tpot −0.03% |
| `s02_no_empty_cache` | 196.810 | 117.883 | TTFT −0.09%、tpot −0.25% |
| `s03_fuse_gate_up` | 196.323 | 119.382 | TTFT −0.34%、tpot **+1.02%** |

噪声底（基线跨轮极差）：`ttft_p50 0.117%` / `tpot_p50 0.444%`。

**冻结裁定**：`s01_both` 保留并定为后续批次默认基线（36× 噪声底）；`s02` 记为
**测不出**（低于噪声底，≠ 无收益）；`s03` 记为**有效负结果**、不启用。
★ 冻结的是"本批单协议下支持的选择"，**不是**论文级收益结论
（`short_chat`/并发 8/`eager`/`git_commit=NOT_MEASURED`）。

**受控性复核（现读）**：四臂的 `execution_optimizations` 计数与 `requested` 逐臂一致
（`s01_both`：`n_calls_last_only=1377`、`n_sampled_rows=15353`；`s02`：`n_suppressed=63`
且 calls=0 —— 抑制真的发生过，不是"没生效"；`s03`：`gate_up` 28 组 / fused 38556 /
fallback 0）⇒ **无"配了但没走到"**。配置差异经 `docs` 核对只有 `runtime:` 段那两行。

### 二、★ S04 的 profile 证据（决定"进入 S04"的关键）

S04 任务书前置是"**只有 profile/分配证据支持才进入**"。证据（现读 + 代码核对）：

1. `benchmark/real_backend.py:55` 明确定义 `pack_ms` = "补位与切片的显存搬运，
   **不含任何设备计算**"；
2. `pack_ms_total` 在**四臂间极差仅 0.17%**（26725.5 / 26764.2 / 26750.4 / 26771.3 ms），
   而 `fwd_ms_total` 随开关变化（141670 / 136257 / 142755 / 149849）⇒
   **pack 是与 S01/S02/S03 无关的、未被触碰的独立热点**；
3. 从 `telemetry.jsonl` 的 `note` 字段分解步级成本：**纯 decode 步 1025/1080（94.9%）**，
   其 `pack_ms/fwd_ms = 23.59%`（19.67 / 83.36 ms）；纯 prefill 只 1 步（pack≈0）。
   1025 个纯 decode 步**全部** `forwards==1`，其中 885 步并发=8（86%）。
4. 代码层面：每 step 每槽位 **≥2 次全量 KV 分配 + 搬运** ——
   入口 `_pack_past`（`real_backend.py:1489-1509`，逐层 `F.pad` + `torch.cat`）
   与出口写回段（`1426-1432`，逐层逐行 `[i:i+1,:,pad:pad+keep,:]` 切片重建）。

**⇒ 裁定：满足前置，进入 S04。**

### 三、S04 实现（`persistent_kv_slots`，默认关）

新增开关 `runtime.persistent_kv_slots` + `runtime.persistent_kv_capacity`
（默认 `False`/`None` ⇒ **既有配置逐字节不变**）。

**快路径判据**（`_pkv_eligible`，返回**原因**而非裸布尔）：
纯 decode（k==1）+ 全部槽位已有历史 + 历史**等长** + 不超预分配容量。
⚠️ **判据顺序有意义**：`new_sequence` 必须在 `not_pure_decode` **之前**——
新请求首步喂整段 prompt（k=prompt_len>1）时若先查 k，`new_sequence` 永远观察不到
（实测：测试初版只报出 `not_pure_decode`）。两者是不同阻塞（新序列即使 k==1 也走不了）。

**快路径做什么**：给每槽位预分配常驻缓冲 `[1, n_kv, capacity, head_dim]`（每层 K/V 各一块），
每步把 HF 返回的 KV **原地 `copy_` 进缓冲**，`s.past` 以**视图**指过去
⇒ 免掉逐层逐行切片重建；判据保证无补位 ⇒ 免掉 `F.pad`。
任务书点名的"持久 Cache 对象未必消除内部 `cat`"**确实是这样**：`_pkv_build_past`
仍要 `cat`（HF 规则张量的硬约束），省的是补位与写回切片重建。

### 四、★★ S04 实现中实测踩到的两个坑（都有测试兜住）

1. **HF 返回的 `past_key_values` 是累积完整 KV（长 `past_len+k`），不是本步增量。**
   当成增量写 `[prev:prev+k]` ⇒ `RuntimeError: output with shape [1,2,1,8] doesn't
   match the broadcast shape [1,2,10,8]`。修：按整段写 `[:keep]`。
2. **喂给 HF 的 past 只能到 `past_len`，不能到 `past_len+k`。**
   多给一个 token ⇒ HF 在它之上再算本次输入 ⇒ **长度双重计数**
   （`size of tensor a (10) != b (11)`）。修：`_pkv_build_past` 的 `keep = past_len`。
3. ★ **缓冲首次分配时必须把槽位既有 `s.past` 搬进去。** 缓冲新分配是全零，
   而该槽位已经走过 prefill（通用路径）持有真实 KV ⇒ 不搬就第一次进快路径
   送**零张量**当历史 ⇒ 输出从第 1 步就分叉（实测前 2 token 相同、第 3 个起错开），
   形状全程合法、不报错。**这条是被"逐层 KV 比对"用例抓出来的**，token 比对也能抓。

### 五、S04 验证结果

- `tests/test_optim_s04_persistent_kv.py` **12 passed**（含逐层 KV 等价、快路径→混长
  回退→继续 的**接续**用例、`retire` 回收、判据原因否定测试）。
- **破坏验证 4 组**（每组都确认测试**能**抓住）：
  ① 不搬既有历史进缓冲 ⇒ **4 failed**；
  ② 快路径写回后不维护 `s.past` ⇒ **3 failed**（含接续用例，报 `cat` 尺寸不匹配）；
  ③ `retire` 不清缓冲 ⇒ **1 failed**（`2 != 1`）；
  ④ 判据去掉"历史等长"检查（最危险的静默失效）⇒ **2 failed**（数值错开）。
  还原后 12 passed、与备份 `diff` **逐字节一致**、**CR=0**。
- 全量回归：**本轮跑**（见 `OPTIMIZATION_TEST_SESSION.md` §5.9）。

### 六、★ S05 真机探针（结论为负：MUSA 无原生 GQA）

探针 `scripts/gqa_support_probe.py`（产物 `results/raw/_probe/20260930/gqa_probe_ops.json`，
**探针路径 ⇒ 不盖四标签、不进结论表**）。真机环境核实：
`python 3.10.8 / torch 2.2.0 / torch_musa 1.3.0+81caf0a / transformers 4.46.3 / MTT S4000`。

**静态+真机一致的事实**：
- `transformers.models.qwen2.modeling_qwen2` **有 `repeat_kv`**；
  `eager_attention_forward` 里对 K 与 V **各调一次**（`num_key_value_groups = 7`）；
- `repeat_kv` 用 `expand`+`reshape`，真机实测 **`shares_storage_with_input = False`**
  ⇒ **真的分配新存储**（非视图），7× 物化**真实发生**。

**真机耗时**（K+V 各一次，min_ms；`sync` 后计时）：

| B | L=1 | L=64 | L=512 | L=1024 |
| --- | --- | --- | --- | --- |
| 8 | 0.140 | 0.161 | 0.296 | 0.438 |
| 4 | 0.139 | 0.157 | 0.222 | 0.295 |

⇒ B=8/L=512 单层 ≈0.30 ms × 28 层 ≈ **8.3 ms/步**，占纯 decode 步 `fwd_ms`（83.36 ms）
的 **≈10%**。⚠️ 但**几乎不随 L 增长**（L=1→0.14、L=1024→0.44）⇒ 是**固定 launch
开销主导**，不是带宽主导（最高仅 268 GB/s，远未打满 HBM）。

**★ 决定性否定证据**：MUSA 上**没有可用的原生 GQA 路径**——
- `F.scaled_dot_product_attention(..., enable_gqa=True)` ⇒
  `TypeError: got an unexpected keyword argument 'enable_gqa'`；
- 位置参数形式（8 个参数）⇒ `TypeError: takes from 3 to 6 positional arguments`；
- 且 `torch_musa` 对 SDPA 另有约束：
  `RuntimeError: Now torch_musa only allows explicit value of 'scale' parameter,
  whose value is equal to 1/sqrt(query.size(-1))`
  ⇒ 连"手工 repeat_kv + SDPA"的基准参照都受限。
- 六个形状（q∈{1,8,128} × kv∈{512,2048} × causal × 左 padding）**全部不可用**。

**⇒ S05 裁定：按任务书"不支持即结束该分支"终止**。机制层面复制真实存在且约占
10% fwd_ms，但**当前 MUSA 栈没有替代路径**；要动它须换 torch_musa 版本或自写
attention kernel —— 均超出本阶段范围，登记为**环境限制**而非"我们没做"。

### 七、限制与未决

1. **S04 快路径的真机收益尚未测**（本地只验正确性，性能属 REMOTE_PENDING）。
   理论机会明确：纯 decode 占 94.9% 的步、其 `pack_ms` 占 23.59%。
2. S04 的 `persistent_kv_capacity` 必须显式给（默认 `max_seq_len=32768` ⇒ 每槽位
   ≈1.9 GB，8 并发 ≈15 GB）。配置里取 **1024**（`short_chat` 实测
   `prompt_len+gen_len ≤ 950`）⇒ 每槽位 ≈117 MB。
3. S05 的负结论**限定在当前版本**（torch_musa 1.3.0 + transformers 4.46.3）；
   升级后须重测。
4. `--dry-run` **验不了快路径**（dry-run 走模拟路径，不实例化 `TransformerDriver`）。

### 八、下一步

- S04 真机批次（4 臂：baseline + `s01_both` + `s01_both+s04_pkv` + `s04_pkv`）
  ⇒ 判"快路径是否真的降 `pack_ms`"。
- S06 待 S04 交付"稳定地址/容量/slot 映射"后评估（快路径已给出固定地址缓冲，
  这正是 S06 图捕获的前置）。
- S07 预注册 §5.1/§5.2 仍待裁定。

## 2026-09-30（续二）— EXECOPT4 运行中发现：心跳线程**周期性覆盖**进度文件（编排层，非测量路径）
- 背景：EXECOPT4（S04 真机批次，4 臂 × 3 轮）起跑后，我用两种方式查进度，**得到互相矛盾的答案**：
  `python3 -c "json.load(open('exec_opt_progress_EXECOPT4.json'))"` 读出
  `n_done=0 / n_total=12 / current=None / elapsed_s=0.0`，而同一次查询里
  `ls -d results/raw/EXECOPT4-*` 列出 **4 个已完成目录**、日志有 4 行 `✓ ...#r0`。
- 发现（成因）：`benchmark/exec_opt_plan.py::progress_payload` 把 `current` 与
  `n_done` 都按**值**放进 payload；而 `scripts/run_exec_opt_matrix.py` 构造心跳时传的是
  **一份构造时算好的快照** ——
  ```python
  hb = Heartbeat(snapshot(None), progress_file)      # current=None、n_done=0 被冻结
  ...
  write_json_atomic(progress_file, snapshot(key))    # run 边界写的才是真实值
  ```
  `Heartbeat._loop` 每 30 s 把**那份冻结的 dict** 原样写回 ⇒ 真实进度被周期性地
  打回初始值。★ 这是「**同一文件两个写者**」的形态，不是"偶发不一致"。
- 影响：① 本机侧轮询看到的 `current`/`n_done` **时真时假**（取决于查询落在 30 s
  窗口的哪一段）；② `exec_opt_plan.decide_wait_exit` 的三条退出通道里，
  **`n_done>=n_total` 这条冗余通道因此不可靠** —— 与
  「等待侧必须枚举全部退出通道」（第九坑）是**同一族**：通道枚举齐了，却让其中
  一条的内容被别的写入者周期性破坏。本次批次**未被拖住**（`summary` 文件通道正常，
  它由收尾逻辑单独写、不受心跳影响），但那是**运气**，不是设计保证。
- 修复（**只动编排层**）：`Heartbeat` 改持 **provider 回调**，每次心跳**重新计算**
  payload ⇒ 与 run 边界写**同源**，不再互相覆盖；另在 run **开跑前**就落一次
  `current`（单次 run 可能 20+ min，否则整段期间显示的仍是上一项）。
  `interval` 做成可注入参数（默认 30 s，测试传 0.02 s）。单测
  `tests/test_exec_opt_plan.py::TestHeartbeatDoesNotClobberProgress`（3 例，含
  **否定测试**：内联一份"冻结快照"式旧实现，证明它**确实**会把真实值覆盖成 `None`）。
  破坏验证：把 `self._provider` 改成构造时冻结 ⇒ 2 failed（正是那两条"不得写过期值"），
  从**同一份**备份恢复后 47 passed、`CR=0`。
- ⚠️ **与 EXECOPT4 产物的关系**：本修复**不改测量路径**（`benchmark/real_backend.py`、
  `run_benchmark.py` 均未动），只改**编排层**的进度文件写法 ⇒ EXECOPT4 的**测量有效性
  不受影响**。但本机 `scripts/run_exec_opt_matrix.py` 已被修改而**尚未推远端**
  ⇒ 报告该批次时须注明"编排脚本在批次运行期间有本机改动（未推送）"，避免与
  部署戳口径冲突（§65 踩过同类：部署戳与远端树实际不一致）。
- 关联：EXECOPT4；`PLAYBOOK.md` §1 速查表已加该现象行；§2 第 8 条（破坏验证的备份纪律）。

## 2026-09-30（续三）— EXECOPT4 结果 + 三处新缺陷（推送静默失败 / warmup 口径 / 环境双模态）

> ⚠️ 本条**不重复**「续二」的心跳覆盖问题。续二记录的修复（`Heartbeat` 改持
> **provider 回调** + `TestHeartbeatDoesNotClobberProgress`）**已在树里**（本次复核
> `scripts/run_exec_opt_matrix.py:137`、`tests/test_exec_opt_plan.py:460`），
> 但**当时未推远端** ⇒ EXECOPT4 跑的是**冻结快照版** ⇒ 本回合观察到的
> 「`n_done=0` 与 `n_done=12` 在同一文件里共存」正是该缺陷的**现场表现**，
> 与续二描述完全一致（不是新发现）。本次已把该文件推到远端（当前 0 差异）。

**【A】★ 新缺陷①：`exec_opt_batch.py` 的推送步骤**静默不推**（已修 + 双守卫）**

- 现象：EXECOPT4 的 summary 里 `switches` 为
  `s04_pkv = {}`、`s01_both_s04_pkv = {logits_last_only, sample_needed_only}`
  —— **缺 pkv 两个键**，看起来像"S04 没开"。
- 排查：`batch_sha_diff.py` 报 `benchmark/real_backend.py` 与全部 S04 配置
  **一致（130 个文件）** ⇒ S04 实现**在远端**，实验**有效**；落后的只有
  `scripts/` 下 4 个文件（含 `run_exec_opt_matrix.py`）。故那只属**报告缺口**。
- **根因**：`step_push` 用「取以两个空格开头的整行」解析 `batch_sha_diff.py` 的
  **人类报告**，而报告里 `sha 不同` 的行形如
  `  scripts/x.py  local=<sha12> remote=<sha12>`，末尾还有「建议命令」区块
  （其续行也以两空格开头）⇒ 解析出的是**伪路径** ⇒ `push_files.sh` 判
  「跳过（不存在）」⇒ `N=0 / ok=0 / skip=0 / fail=0` ⇒ **退出码 0**、
  **不报错** ⇒ 编排打印"推送源码与配置"后照常起跑，**远端一个文件都没更新**。
- **实测确证**（复刻旧解析器喂真机输出）：8 条 `todo`，其中**只有 1 条**是真实文件，
  其余含 `local=... remote=...` 后缀或命令续行 `\`。
- **修法**：`batch_sha_diff.py` 增 `--emit-paths`（只吐路径，无后缀、无区块）；
  `step_push` 改消费它；**守卫 1**（清单每项必须是本机真实文件，否则报错停跑）；
  **守卫 2**（解析 `push_files: ok/skip/fail (共 N)`，要求 `fail=0`、`N==清单长度`、
  `ok+skip==N`）。单测 12 例（`TestPushTodoParsing` / `TestPushSummaryGuard`），
  破坏验证 2 组（退回旧解析 ⇒ 2 failed；守卫 2 恒 None ⇒ 4 failed），从**同一份**备份
  恢复后 57 passed、`CR=0`。
- ⚠️ **同族第二处**：`--emit-paths` 的输出行尾曾是 **CRLF**（Windows 上 Python stdout
  做换行翻译）⇒ shell 侧 `$(...)` 展开**不会**剥掉 `\r` ⇒ 每个路径带 `\r` ⇒ 又判「不存在」。
  Python 侧 `str.strip()` 恰好会剥掉 ⇒ **只在 shell 这条路径上错**（"只坑一个调用方"的形状）。
  已在源头改成 LF 输出，并加 `parse_push_todo` 的 CRLF 容错用例。

**【B】★ 新缺陷②：`driver_stats` 的 `*_total` 含 warmup，且各臂 warmup 轮数不同（已加新键）**

- 证据链：`_drive_warmup_batch` 直接调 `self.forward(...)` ⇒ 预热与正式段
  **共用同一累加器**（`_n_forwards` / `_pack_s` / `_fwd_s` / `_tokens_out`），
  而 `wall_ms` 明确排除预热。EXECOPT4 实测 `warmup.iterations` = **2 vs 1**，
  `n_forwards` **1377 vs 1260**，差 **117 = 恰好一轮预热的 1 prefill + 116 decode**。
- 后果：直接比 `pack_ms_total` 读出 **"+31%"** 的**假效应**；用 telemetry（正式段）求和
  后真值是 **+0.19% / +0.52%**（≤ 噪声）。★ 这是"**口径决定结论**"的又一例。
- 修法：**不改既有键语义**（否则与历史产物不可比），在 `warmup()` **循环之后**记零点，
  `stats()` **新增** `measured_window.*`。⚠️ 零点必须在循环**之后**——放循环之前等于
  什么也没扣（本地单测当场抓到）。单测 `TestMeasuredWindowScope`，破坏验证 1 组。
- ⚠️ EXECOPT4 产物**早于该字段** ⇒ 该批只能用 telemetry 求和替代；
  `scripts/analyze_exec_opt_batch.py` 已实现**两路口径并列 + 交叉校验**。

**【C】★ 新缺陷③：`persistent_kv_slots.bytes_allocated` 终态恒 0（已修）**

- `retire()` 把驻留字节扣回去（设计成"当前驻留"），而 run 收尾时所有请求都已 retire
  ⇒ 报告恒为 `0`——**跑了 193 步快路径却显示 0 字节**，无法区分"没分配"与"分配后释放"。
- 修法：新增 `bytes_peak`（峰值）与 `n_allocations`（累计次数），不因释放归零。
  单测 `TestAllocationVisibleAfterRetire`（含"未开机制 ⇒ 峰值为 0"的反向对照），破坏验证 1 组。
- ⚠️ **破坏验证方法本身的教训**：首次注入的破坏行落在随后的真赋值**之前** ⇒
  被覆盖成死代码 ⇒ 用例照常通过。**"破坏后没红"首先要怀疑注入无效**，不要据此
  判定"用例不敏感"。（本轮破坏验证共 5 组，其中 1 组是无效注入、已重做。）

**【D】EXECOPT4 结果（S04 真机，12/12 `MEASURED`）**

- 协议与 EXECOPT3 相同；4 臂 × 3 轮；`balanced=true`、`failures=[]`、35.0 min。
- **机制确实生效**：快路径 **193**、回退 **1067**（`unequal_kv_len=1002`、
  `new_sequence=65`），**3/3 轮逐位相同**、分桶闭合。
  ★ **阻塞项是"历史不等长"而不是容量**（`capacity_exceeded` 一次未现）：
  连续批处理下各序列 `past_len` 天然相差常数偏移 ⇒"严格等长"是**稀有时刻** ⇒
  覆盖面只有 **193/1260 = 15.3%**。
- **判据（telemetry 口径）**：`pack_tel` 中位 基线 **21,269.6** vs `s04_pkv` **21,310.4**
  ⇒ **+0.19%**；组合臂相对 `s01_both` **+0.52%**。臂内极差 0.17% ⇒
  **差异 ≤ 噪声 ⇒ 测不出收益**。（详见 `EXPERIMENTS.md` §1.6 / §3.6 E20 / §4.4。）
- **`s01_both` 交叉复现**：TTFT p50 188.80 vs 197.13（**−4.2%**，3/3 同号），
  与 EXECOPT3（188.78 vs 196.98）一致 ⇒ E16 升格为**两批一致**（§3.6 E19）。

**【E】★ 环境双模态：服务指标本批不可用（新，且会影响后续所有批次）**

- 12 个 run 按 `recorded_at` 排序，`wall_ms` 分两簇：
  **快 123.2–123.6 s / 吞吐 63.6–63.8**（8 个）vs **慢 137.1–143.0 s / 吞吐 55.0–57.4**（4 个）。
- 慢模态落在 `s4000_fp16` r0、`s01_both` r2、`s01_both_s04_pkv` r0/r2 ——
  **跨 3 个不同臂**，且时间上成簇（≈19:54 与 ≈20:20–20:26 两个窗口）
  ⇒ **机器/环境瞬变，不是机制效应**。
- ⇒ `s01_both_s04_pkv` 表面的「吞吐 −12.4%、`tpot90` 212→224 ms」是**抽样假象**
  （它 2/3 轮落慢模态，而慢模态率约 33%，n=3 完全在偶然范围内）。
- ★ **分层影响**：慢模态**会**抬高 `fwd_tel`（基线慢轮 118,161 vs 快轮 ≈101,400，+16%）
  ⇒ `fwd_tel` 本批**不可比**；但**不**系统抬高 `pack_tel`（基线慢轮 r0 的 `pack_tel`
  是三轮最低）⇒ `pack_tel` 结论不受影响。
- **待办**：批次内需**探测并标记慢模态**（如每轮 `wall_ms` 离群）或按模态分层报；
  且 **3 轮不够**。

**【F】本次变更与验证（可复核）**

- 改：`benchmark/real_backend.py`（`measured_window` + `bytes_peak`/`n_allocations`，
  **默认行为不变**，S04 仍默认关）、`scripts/exec_opt_batch.py`（推送守卫 + `--arms` 透传
  + `PUSH_INCLUDE` 单一常量）、`scripts/batch_sha_diff.py`（`--emit-paths` + LF 输出）、
  `scripts/run_exec_opt_matrix.py`（`_SWITCH_KEYS` 覆盖 S04 两键）、
  新增 `scripts/analyze_exec_opt_batch.py`、新增 `tests/test_analyze_exec_opt_batch.py`。
- 测试：受影响集 `tests/test_exec_opt_plan.py` + `test_optim_s04_persistent_kv.py` +
  `test_analyze_exec_opt_batch.py` = **81 passed / 2 subtests**；
  ★ **改动后全量回归 = 1516 passed / 1 skipped / 758 subtests / 10m33s**
  （`--ignore=tests/test_validate_fp16_path.py`；该文件因本机未装 torch 而忽略）。
- 破坏验证 5 组（4 组有效、1 组无效注入已重做），每次从**同一份**备份恢复，
  收尾逐文件量 `CR=0`。
- 远端同步：`batch_sha_diff --emit-paths` 复核 **0 差异**（本地与远端同一份代码）。

**【G】下一步（须先裁定，已在会话文档 §5.9.8 列出）**

1. S04 是否继续：现判据覆盖面仅 15.3%；要可行使须改为**按槽位独立偏移/长度**
   （设计扩张），且**须先证明**"每步整段 `copy_` 已累积 KV"不是净增成本
   （本版实现整段写回，与实测 +0.2%~0.5% 一致）。
2. 服务指标协议：先探测/标记慢模态（或分层），否则 3 轮易造假效应。
3. S06 已具备设计前提（固定地址 + 定容量 + 槽位映射），但收益前提受第 1 条制约。
4. S07 预注册 §5.1/§5.2 仍待裁定。

## 2026-10-01 — ★ 独立复核后的更正：S04 带**数值错**、覆盖率是 6.74%、`pack` 尾段 **+72%**

- **来源**：用户独立复核（读源码 + 12 份原始产物 + 另做 CPU 微型模型验证），
  指出我方 2026-09-30 报告的三处误判。**我逐条复现确认**，并据此执行了一轮
  "正确性与证据链修复"（**本轮未启动任何远程实验**）。5 条指示见会话文档 §5.9.9。

**【1】★★ S04 有已复现的数值错（比"有没有收益"更优先）**

- 路径：`fast → fallback → fast`。**回退只更新 `_Slot.past`，不更新 `_kv_buf`**，
  而重入时 `_pkv_ensure_buffers` 原判据是 `if buf is not None: continue`
  ⇒ 读到**过期缓冲** ⇒ 形状合法、不报错、数值错。
- 独立复现（CPU 微型 Qwen）：重入后同一请求输出 token **60 → 36**、下一步 **62 → 4**；
  逐层 KV 与基线最大差 **2.102e-01**。
- ★ **为什么我方用例漏了**：既有 `_fallback_scenario` 补入混长请求后**从不移除它**
  ⇒ 批一直混长 ⇒ 快路径**再没被重新进入** ⇒ 重入路径零覆盖。
  ⇒ **教训：用例要覆盖"状态迁移的回路"（进入→离开→再进入），只测单向接续不够。**
- 修：按 `_pkv_synced_len`（已同步长度）判脏并重同步；`retire` 一并清。
  新增 `TestReattachAfterFallback`（token + 逐层 KV 双对照）。
  破坏验证：退回原判据 ⇒ 报 "`a` 第 0 层 K 最大差 2.102e-01"。
  （token 在小场景里恰好还没分叉 ⇒ **加 KV 对照是对的**。）
- **不得**据此说"整批都算错了"：本批快路径只覆盖 6.74% 且集中在单请求尾段，
  是否有"回退后重入"未逐轮核 ⇒ 这批 S04 的**数值正确性未获证明**。

**【2】★ 快路径漏计 `tokens_out`**

- `self._tokens_out += 1` 原**只在通用分支**；快路径分支只 `record_token`。
  CPU 微型实测：实际 **17** 个 token、只记 **9** 个。`measured_window` 救不了（相减照样漏）。
- 修 + 用例 `TestTokensOutCountsFastpath`；破坏验证（删计数行 ⇒ 失败）。

**【3】★★ 覆盖率不是 15.3%，是 6.74%；`pack` 也不是"不可分辨"，是 +72%**

- `193` **含预热**（预热 1 轮 = `gen_len-1` = **116** 步纯 decode，同样命中快路径），
  分母 `1260` 也没扣。正解：`(193-116) / (1260-117)` = **77 / 1143 = 6.74%**。
  这 77 步 = **末尾连续 `decode==1` 的单请求尾段**（telemetry **第 1004 行**起），
  与用户指出的一致。
- 尾段 `pack` **总量**（每轮 77 步求和 → 3 轮中位数）：
  `s4000_fp16` **109.02** → `s04_pkv` **187.52 ms** = **+72.0%**；
  `s01_both` **108.98** → 组合臂 **187.31** = **+71.9%**。
  逐轮 187.13/187.52/187.92（极差 **0.4%**）⇒ **可复现的真实负效应**，
  约为臂内噪声底的 **175×**。
- ⇒ 我方原结论"+0.19%、差异 ≤ 噪声、测不出收益"是**被 6.74% 覆盖率稀释**的产物，
  **已作废**。机制上也自洽：**原路径遇到等长历史本来就不 padding**，写回只是**切视图**；
  快路径每步多做一次**整段 `copy_`**（写回 HF 返回的累积 KV）⇒ 净增搬运。
- ⚠️ 这是**主机计时边界下的 `pack` 字段**，不能读成"GPU 搬运时间 +72%"。

**【4】★ "环境双模态"不成立：那是**孤立单步长停顿**，且机制归因暂不可判**

- 逐 step 实测：`s4000_fp16` r0 `fwd_ms` **16,839 ms**@779；`s01_both` r2 **13,865 ms**@993；
  组合臂 r0 `pack` **16,961 ms**@360、r2 `fwd_ms` **19,288 ms**@456。
  常规轮单步上限仅 ~1.34 s（step 0 预填充）。
- **wall 差额逐轮对上**：123.4+13.9=137.3≈137.1；123.6+16.8=140.4≈140.2；
  123.5+17.0=140.5≈140.9；123.2+19.3=142.5≈143.0
  ⇒ 差异是**一次停顿**的算术后果，**不是持续慢态**。
- 改写为：**吞吐/时延差异受未解释长停顿影响，机制归因暂不可判**；
  保留**全部**有效运行，**不得**事后删除"慢轮"。
  （我方原文"双模态已证实""吞吐下降纯抽样假象"两句**撤回**。）

**【5】★ 部署校验没闭环：12/12 产物没有源码身份（已修）**

- 实测：`data_label=MEASURED`，但 `git_commit=NOT_MEASURED`、`git_source=stamp_stale`（12/12）。
- 成因：`exec_opt_batch.py::step_push` 把 `remote_run.py stamp` 的**退出码丢掉了**
  ⇒ `do_stamp` 返回 4/6 也照样进 `step_launch`。
- 修：`stamp_complaint(rc, out)` —— `rc != 0` 拒；`rc == 0` 但缺正标记
  `投放并校验通过` 也拒（"没报错"≠"验过"）。
  **续跑加身份核验** `identity_complaint()`：源码来源可信 + `config.snapshot.yaml`
  逐字节一致 + `model_revision` 一致，否则拒绝复用（不再只看"请求完成数"）。
  ⚠️ 历史产物**保留原样、不补写标签**。
- 反例：`TestStampGuard`（含"rc=0 无标记必须拦"）+ `TestResumeIdentityCheck`
  （含 `stamp_stale` 被拒 —— **正是本批 12 份的形态**）；破坏验证各 3 failed。

**【6】本轮变更 / 验证 / 同步**

- 改：`benchmark/real_backend.py`（重同步 + `tokens_out` + 分桶/双分母 + `n_resyncs`）、
  `scripts/exec_opt_batch.py`（stamp 守卫）、`scripts/run_exec_opt_matrix.py`
  （`identity_complaint` + 续跑接上）、`scripts/analyze_exec_opt_batch.py`
  （覆盖率/尾段/长停顿三节）、`tests/*` 扩充。
- 测试：受影响集 **101 passed / 4 subtests**；★ **改动后全量回归 = 1535 passed / 1 skipped / 760 subtests / 9m34s**。
- **未跑真机**（按指示）。**未推送远端**（不改测量路径的编排层之外，`real_backend.py`
  变了 ⇒ 下次开跑前必须先推 + 先过 stamp）。
- 四状态见 `OPTIMIZATION_TEST_SESSION.md` §5.9.9(6)；最小真机验证计划见 §5.9.9(7)。

**【7】仍未解决**

- 长停顿**成因未定位**（分配/同步/宿主机暂停/实现交互四选一都没排除）。
- EXECOPT4 的 S04 数值正确性**未证明**（需修复后重跑）。
- 覆盖率为何只有 6.74% —— 判据"严格等长"在连续批处理下结构性罕见，
  扩到"按槽位独立偏移/长度"是**设计扩张**，须先裁定（D1）。
- B1 仍不可冻结（`384` 未采样 + §9.5 未证 + §5.1/§5.2 待裁）。


## 2026-10-02 — 远端连接恢复 + 推送链路的一个静默失效

### 【1】远端不可达的分诊：是端口变了，不是网络问题

`Error reading SSH protocol banner` 第一眼像网络故障。`scripts/diag_remote_reachability.py`
给出**否定证据**：**目标端口 EOF（对端主动关闭）**，而同线路对照 SSH **能拿到 banner**
⇒ 主机在线、**该端口无服务**。⇒ 判定为 **AutoDL 重启/迁移改了映射端口**。

这与 2026-09-24（换镜像 `14671 → 14293`）是同族现象的**第二次实证**
⇒ "重启/迁移会改端口"**从推断升级为规律**。处置：用户取到新端口 **23576**，
只改 **`~/.workbuddy/remote.env`（仓库外）**；两步验证都做了：
① banner `SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.11`；
② **认证**（banner ≠ 认证通过）`exec "hostname; whoami"` ⇒
`autodl-container-05264bbf0d-3b9c3488` / `root`。
凭据卫生已核：`git ls-files | grep -i env` 为空、`.gitignore` 已设防、仓库内无副本；
`MEMORY.md`/`PROJECT_MAP.md` **只指向 `remote.env`、不复述端口** ⇒ 端口变更**无需改文档**。

### 【2】★ `push_files.sh` 的 `--include` 会**静默降级**成默认扩展名

**成因**：`--include` 的解析循环在**第一个位置参数**处 `break`。把 `--include`
写在位置参数**之后** ⇒ 它留在 `"$@"` 里**被当路径** ⇒ 打「跳过（不存在）：--include」⇒
`INCLUDE` **静默退回 `py,sh`** ⇒ **全部 `.yaml/.yml/.json` 不进清单**，
而输出形如 `ok=87 skip=120 fail=0`、**退出码 0** ⇒ 看起来完全成功。

**发现路径（这次的幸运之处）**：没有只看退出码，而是核 **configs 整目录的文件数**
⇒ `远端 138 vs 本地 144`，差 6 个全在 `configs/mthreads/phase1_step15_v5/`。
按错序把 `--include` 放在后面的写法**复现**了它：`待推送 1 个文件` + 两条「跳过（不存在）」。

**意义**：这**正是** PLAYBOOK §1 那条「代码是新的、配置是旧的」的**实际成因**。
它把"我调用了带 `--include` 的命令"与"扩展名真的被应用了"两件事分开 —— 前者不蕴含后者。

**修法**：位置参数里出现 `--include`/`--include=…` 形态 ⇒ **报错 + `exit 2`**，
不再静默降级成默认值。**回归** `TestPushIncludePositionGuard`（4 例，含两条正向对照
证明守卫有判别力：不写 `--include` 时静默取默认是**设计行为**、不得被误拦）。
**破坏验证**：删守卫 ⇒ 2 条错序变红、2 条正向对照保持绿；从**同一份备份**还原，
sha 逐字节一致、LF 干净。

### 【3】⚠️ 我自己的操作失误：脚本运行期间改了脚本（并发写者）

后台 `push_files.sh` 正在跑（149 个 configs）时，我**同时**用 Python 改写了
`scripts/push_files.sh` 做破坏验证 ⇒ bash **逐行读脚本**，读到被改动的文件 ⇒
`line 116: syntax error near unexpected token '('`。

**这是用户级约定里"长跑型多步骤传输只能有一个写者"的直接命中**
（我当时以为"改文件不冲突"，忘了 bash 把脚本文件本身当输入流）。

**后果已核（不靠推断）**：149 项**均在报错前处理完**（全部 `SKIP`，且报错出现在
收尾汇总阶段）；事后 `configs` 文件数 `144/144`、逐文件 sha **diff=0**、三目录终验
**diff=0** ⇒ **未损坏任何产物**。⇒ 登记为"操作失误 + 影响已界定"，不写成"无影响"。

### 【4】一个测量小坑：整目录合并摘要会被**列宽**造出假差异

用 `xargs sha256sum | sha256sum` 比两个目录时，摘要**不一致**，但逐文件 `diff=0`。
原因：本地 `sha256sum` 单文件输出用 **2 空格**、远端经管道/xargs 用 **1 空格**
（GNU coreutils 的终端宽度判定）⇒ 路径列宽不同 ⇒ 合并摘要必然不同。
**归一化（`sed 's/  */ /'`）后完全一致。**
⇒ 判据用"整目录摘要"时**必须先归一化格式**，否则造出与内容无关的假差异
（同族：用户级约定里"对比两棵树时先怀疑对比方法引入的差异"）。

### 【5】收尾与状态

* 推送链路终验：`src/ascend_opt` 19/19、`benchmark` 11/11、`scripts` 77/77、`configs` 144/144，
  全部**逐文件 sha 一致**。
* `stamp`：`投放并校验通过`、`rc=0`、`tree_digest=2b79ff65f275…`、`dirty=True`（A6 口径限制）。
* **真机批次仍未启动** —— 须待用户批准 §5.9.9(7) 的最小验证计划。
* 未解除：EXECOPT4 的 S04 数值正确性仍**未证明**；长停顿成因未定位；
  B1 仍不可冻结；`--include` 守卫已修但**其它调用方是否也用过错序**未逐一回溯
  （已核：本轮配置已全部补齐，历史批次若有同类漏推早已体现在产物里，不追溯改写）。

## 2026-10-02（续）— 用户 6 条修订指令：测试运行器 / 离线 dry-run / 摘要级续跑 / 真机数值脚本

### 【1】★★ 更正我上一轮的一个错误数字：遗漏用例是 **13**，不是 74

我上一轮把"pytest 补跑 **74** 项（`test_plot_step16_decouple`=13、
`test_source_stamp`=61）"写进了汇总。**74 是错的。**

成因是我自己的触发条件写成 `pytest 收集数 > unittest 收集数`：
pytest 会把 `@pytest.mark.parametrize` **展开成多条 nodeid**，而 unittest 只当
**一个**测试方法 ⇒ `n_py` **天然偏大**，即使该文件里根本没有 unittest 收不到的用例。
两类假阳性实测：
`test_source_stamp`（unittest 48 / pytest 61）、`test_validate_fp16_path`（23 / 48）——
它们的**差集是空的**，却被算进了"补跑"。

**真值（直接核过，不是印象）**：

```
tests.test_plot_step16_decouple   unittest_have = 0    pytest_collect = 13
```

⇒ 真正被静默漏掉的只有这 **13** 个，正是用户第 1 条点名的数字。

**教益**：判据必须是**差集非空**，不能是**计数更大**。凡是
"两个集合大小不同 ⇒ 有问题"的判据，都要先问一句"**计数为什么不同**"——
参数化、重复、继承都会让计数天然不等。

### 【2】★★ 退出码**不是**无条件权威：必须区分"收尾噪音"与"收集期崩溃"

补跑 `test_plot_step16_decouple` 时输出 `13 passed`，而 `returncode` 非零
⇒ 旧规则 `if rc != 0 and not (failed or errored): errors = max(errors,1)`
把它强行记成 **`errors=1`** —— 一个**假失败**。

逐层查下去，真凶是**本机沙箱的 `safe-delete` 垫片**：
pytest 结束时清理 `Temp/pytest-of-<user>/garbage-*`，而 `rm` 被替换成
"送入回收站"的实现，拿到 `\\?\C:\...` 扩展路径后 Windows 报
`Errno 53 找不到网络路径`，垫片 fail-closed、把一段 JSON 打到 **stdout**、
并**返回非零退出码**。⇒ 这是**环境垫片**行为，**不是**被测代码有问题。

**但守卫不能因此删掉**（它本意是拦"收集期就崩了，压根没有结果"）。
两者的**可分辨差异**：
* 收集期崩溃 ⇒ **没有** `N passed` 汇总行；
* 垫片噪音 ⇒ **有**汇总行，且 `N` 与待跑 nodeid 数**一致**。

⇒ 新判据：**有汇总行 且 `run == len(todo)`** ⇒ 不记 error，但**必须留痕**
（写进 `skip_reasons`，说明"退出码非零来自收尾阶段"），否则事后有人问
"为什么退出码是 1"会无从查起。否则**照旧**记 error。
回归：`tests/test_run_tests_harness.py`（7 例），含两条**否定测试**
（无汇总行 ⇒ 仍必须 error；结果数对不上 ⇒ 仍必须 error）。

### 【3】★★ 续跑身份：升级到摘要级，并把三处"缺席即通过"改成"缺席即拒绝"

**(a) 先修掉一个真缺陷。** `read_source_stamp` 内部用
`want = stamp["tree_digest"]` 反复读它做校验，**却从没把它写进返回值**
⇒ 新加的 `detect_source_identity` 读 `stamp.get("tree_digest")` **恒得 `None`**，
而 `validated` 仍是 `True`。症状 = "戳明明有效，摘要却永远不可用"。
⇒ 已补 `out["tree_digest"] = want`。**校验的意义就是把 `want` 确认为当前这棵树
的摘要；不把它交出去，等于白校验。**

**(b) `ReproducibilityRecord` 新增 `tree_digest`**（落盘为 JSON `null` 表示"不可用"，
**不写空串**；**不进** `REQUIRED_REPRO_FIELDS`，否则所有历史 run 一夜之间
变成"元数据不完整"，preflight 全红）。

**(c) `identity_complaint` 三处"缺席即通过"**（旧写法在任一侧为空时**短路跳过比对**）：

| 旧写法 | 错在哪 |
| --- | --- |
| `if src is not None and src not in TRUSTED` | 产物没这个字段就放行 |
| `if snap.is_file() and cfg_file.is_file()` | 少一个就不比 |
| `if model_revision and got and got != model_revision` | 任一侧为空就不比 |

三者都把"**测不出**"读成了"**没问题**"—— 方向恰好反了。
新语义：**核不了 ⇒ 不能说它可信 ⇒ 拒绝复用。**

**(d) 锚定真实 baseline 路径。** 旧写法对 baseline 臂取
`cfg_dir / "baseline.yaml"`，**该文件不存在** ⇒ 那个 `cfg_file.is_file()` 恒假
⇒ **配置身份从来不比**。改为 `cfg_path_of["baseline"] = baseline_path`。

**(e) 破坏验证（7 条守卫逐条短路）**：源码身份缺 / 本机摘要测不出 / 产物无摘要 /
摘要漂移 / 缺快照 / 当前配置不存在 / 模型 revision 缺失 ——
**7 条全部从 REJECT 变 PASS**，对得上。

**(f) 边界（有意为之）**：`git_source == "git"`（真 git 树）在**摘要**这一关
**必然拒绝**（`detect_source_identity` 对真 git 树刻意不给摘要——本机开发树的
语义是"HEAD + 未提交改动"）。真 git 树上做续跑本就没有可比的内容摘要，
此时"核不了"必须读成"不许复用"。已用测试冻结该行为。

### 【4】第 4 项交付物：真机强制重入数值验证脚本（⏳ 待执行）

新增 `scripts/verify_s04_reenact_numerics.py` —— **独立**于单测，自己驱动
`RealBackend` 造"快路径 → 混长回退 → 成员退出 → **重入**"序列，
**7 条判据预先冻结**（快路径走过 / 混长回退且 `unequal_kv_len` / **`n_resyncs > 0`**
/ token 逐一相同 / **逐层** KV 逐位相同 / `tokens_out` 对账 / `measured_window` 可读），
并显式声明 `does_not_assert`（**不含**性能、**不含**过载/SLO、**不含**非重入尾部）。

**本地微型模型已验证"场景本身"能造出要求的形态**：
`fastpath=7 / fallback=5 / resyncs=2`、`fallback_reasons={new_sequence:3, unequal_kv_len:2}`、
`tokens equal=True`、逐层 KV `same=True`、`tokens_out 23==23`。
⇒ **场景是有效的**（这一步必须先做，否则真机上"全相等"可能只是"重入压根没发生"）。

### 【5】修掉自己引入的性能回退

`_pytest_collect_count` 与 `_run_pytest_subset` 原先**各起一次** `--collect-only`
⇒ 每模块 **2 次** pytest 启动。40 个模块下 `--group logic` 从 ~7 min 涨到 **>20 min**。
已合并为一次（`_pytest_collect_nodeids` 同时返回收集数与 nodeid 列表）。

### 【6】状态

* 第 1、2、3 项**已完成**并有回归 + 破坏对照。
* 第 4 项**脚本就绪、待用户批准占机时**；第 5 项**未开始**（须第 4 项通过）。
* 第 6 项全程遵守：S04 默认关闭、未扩混长、未进图捕获。
* ⚠️ 本轮**改了源码文件** ⇒ 远端副本已过期，**下次真机批次前必须重推 + 重打戳**
  （`tree_digest` 会变）。

---

## 2026-10-02（续·核实轮）重推 + 重打戳 + 破坏性对照

### 【1】远端副本过期这件事，实测并闭环了

上一轮末尾我记的是"**改了源码 ⇒ 远端已过期，必须重推 + 重打戳**"。本轮实做：

* 逐文件按 **sha256** 比对（`iter_scope_files()` 给 `(rel, sha)`；**不是**比文件名、
  也不是比目录清单），得**精确差集 = 5**：新增 `scripts/verify_s04_reenact_numerics.py`；
  内容不同 = `push_files.sh` / `run_exec_opt_matrix.py` / `run_tests.py` / `provenance.py`；
  远端没有多出来的文件。
* 漂移量：远端旧戳 `2b79ff65f27534a9…` ≠ 本机 `af3ec669d0678b4e…`。
* 处置：`push_files.sh --dry-run`（rc=0、未建连接）核清单 → 真推 **5/5 OK**
  （判据 = 远端 sha256 == 本机 sha256）→ `remote_run.py stamp /root/ascend_probe_d1 --allow-dirty`
  ⇒ `投放并校验通过`，读回 `tree_digest = af3ec669d0678b4ed34ad5cf97b61c205812dbedf998bb7d9df50000fbd13af9`
  = **本机值**。

### 【2】`read_source_stamp` 的修复在真机上得到证实

修复前：它内部反复读 `want = stamp["tree_digest"]` 做校验，**却从没写进返回值**
⇒ `detect_source_identity` 读 `stamp.get("tree_digest")` **恒得 `None`**，
而 `validated` 仍是 `True`（"戳有效、摘要却永远不可用"）。

真机证据（远端解释器 + 远端真实的戳）：

```
detect_source_identity('/root/ascend_probe_d1')
= ('16e82d2ea117ee4017adf3d0a716e81df12bf0e4', 'stamp', True,
   'af3ec669d0678b4ed34ad5cf97b61c205812dbedf998bb7d9df50000fbd13af9')
```

⇒ `source='stamp'` 且**第 4 位真的带上了摘要**。这条**不是**靠本机单测推的。

### 【3】真实 EXECOPT4 产物按新判据仍被拒绝（设计目标达成）

`results/raw/EXECOPT4-short_chat-s01_both{,_s04_pkv}` 的 `meta.json`：
`git_source='stamp_stale'`、`git_commit='NOT_MEASURED'`、`tree_digest=None`。
**两种**输入（`source_digest=None`、`source_digest=活摘要`）**都**返回
`源码身份不可信（git_source='stamp_stale'…）` ⇒ 这批产物**不得复用**，
第 4、5 项必须跑**新 tag**。

⚠️ **必须写明的边界**：`resolve_source_digest(_ROOT)` 读**本机**戳；本机无戳
（戳只投远端、不入库）⇒ `source_digest=None` ⇒ **在本机**跑 `run_exec_opt_matrix`
会对**一切**既有产物拒绝。这是**设计边界不是缺陷**（其用法就是远端），
但不明写就会被下一轮读成"判据坏了"。

### 【4】第 2 项的"不联网"是**证过**的 —— 靠破坏性对照

`test_dry_run_never_touches_network` 取**外部可观测事实**：远端给成
`//192.0.2.1:1`（TEST-NET-1，RFC 5737，**保证不可路由**），要求 ①`rc=0`
②输出含"未建立任何连接" ③墙钟**远小于** 20 s。与正向对照同批实测 **7.9 s 内全过**。

**破坏性对照（本轮新做）**：把 `push_files.sh` 的 dry-run 早退条件
`[ "$DRY_RUN" -eq 1 ]` 改成 `-eq 2`（**永不成立**）⇒ 脚本会真的走下去建连
⇒ 该测试**变红**，`subprocess.TimeoutExpired`（**60 s，卡在真实 TCP connect**）。
⇒ 证明这条测试**真的在区分**"没建连接"与"停得早"。
还原后 `diff` 为空、`TestPushIncludePositionGuard` **6/6 绿**。

### 【5】`test_batch_lock` 的 `errors=1` 定性 = **负载敏感，非缺陷**（且是我自己造成的）

* `_regr3`（`--group all`）里它报 `通过 42 / 失败 0 / 错误 1`；隔离跑
  `42 passed, 20 subtests passed`。
* **真因**：我较早用 `nohup … &` 起过一个 `--group all` 回归，**它没随工具调用结束而死**，
  成了孤儿进程（Windows PID **57048**，内存 228 MB），**又跑了一整轮**；
  此后我并发起了第二个回归 + 一个 `test_exec_opt_plan`，三者抢 CPU。
* 症状：`_regr4.txt` mtime **40 min 不动**、`test_batch_lock` **71 s → 1343 s（19×）**、
  `test_exec_opt_plan` 82 例跑 **32 min**；而**结果全对**（`RC=0`）。
  ⇒ 判据：**同结果、耗时差一个数量级 ⇒ 先查并发，别查代码**。
* 工具纪律：`nohup cmd &` 在工具调用返回时**保不住**子进程；长跑一律
  `run_in_background=true`。杀进程要用 **Windows PID**（`tasklist` 取），
  `ps -W` 在 Git Bash 下会显示**陈旧条目**（本次 PID 126 就是假的映射）。
* ⚠️ **不构成 flake 证据的两条假线索**：① 回归命令**漏写脚本路径** ⇒ 退回默认 S14 扫描
  （输出"没有找到 S14S-* 产物"，像环境坏了）；② `--verbose | grep` 被 SIGTERM
  （长跑输出先重定向到文件）。已登记 PLAYBOOK §2 第 20 条。

### 【6】第 4/5 项前置逐条核实

* `scripts/verify_s04_reenact_numerics.py`：AST + 实际 import 均 OK；
* `configs/mthreads/exec_opt/s01_both_s04_pkv.yaml` 存在；与 `s01_both.yaml` 的差集
  = **仅** `runtime.persistent_kv_slots: true`（后者**没有**这个键）⇒ **单变量成立**；
* **模型就绪**：远端 `snapshots/a09a35458c702b33eeacc393d103063234e8bc28` 存在、
  4 个 `safetensors` 分片齐全；两份配置钉死的 `model_revision` 与之**逐字符相同**。
  ⚠️ 我曾把缓存 revision **抄错**成 `…d103…` vs 配置 `…d043…` 并据此判 `REV_MISSING`；
  从 yaml 精确读出后确认**两者都是 `d103063`** ⇒ **我的转录错误**（同 PLAYBOOK"取值必须现读"）。

### 【7】本机测试结果（本轮实跑）

| 目标 | 结果 |
| --- | --- |
| `tests/test_exec_opt_plan.py` | **82 passed, 4 subtests passed, RC=0** |
| `tests/test_run_tests_harness.py` | **7 passed** |
| `TestPushIncludePositionGuard`（单独） | **6 passed** |
| `tests/test_batch_lock.py`（隔离） | **42 passed, 20 subtests passed, RC=0** |

### 【8】状态

* 第 1、2、3 项**已完成**，且第 2、3 项的验收证据是**真机/破坏性对照**级别的。
* 第 4 项：**脚本 + 前置全部就绪，待用户批准占机时**。
* 第 5 项：**未开始**（须第 4 项通过后）。
* 第 6 项全程遵守：S04 默认关闭、未扩混长、未进图捕获。⛔ 不得开始 OPT-04。

---

## 2026-10-02（续·第五轮）退出码判据的第三形态 → 换证据源；第 3 项重推闭环

### 【1】★ 第 1 项的判据连续被打穿**两次**，第三次我换了证据来源

`_regr3` 第一形态：`13 passed` + `rc≠0` ⇒ 判据"**有** `N passed` 汇总行"挡住。
补了"进度点"作第二判据后，`_regr6`（干净负载全量）**又**报同一模块错，
而这次**连汇总行都没有**，只剩 `.............`（13 点）——但**点也被截断**。

成因链：沙箱 `safe-delete` 垫片清理 pytest 的 `Temp/pytest-of-*/garbage-*` 时，
Windows 拒绝 `\\?\C:\...` 扩展路径（`[Errno 53] 找不到网络路径`），
垫片 **fail-closed**：把 JSON 打到 **stdout** 并**返回非零**。
而 `-q` 下汇总行与收尾**同帧输出** ⇒ 收尾一崩，汇总行整段丢；
它的 JSON 还会**挤掉/打断**进度点那一行。

⇒ 教训（已入 PLAYBOOK §2 第 21 条并写进会话文档 §5.9.11(i)）：
**判据所依赖的证据，不能与该故障同源**。汇总行与进度点**都走 stdout**，
而故障恰在收尾阶段 ⇒ 继续加正则只会得到"第四形态"。

⇒ **终局修法：换证据源。** 新增 `scripts/_pytest_outcome_plugin.py`，
用 pytest 自己的 hook 把逐结局计数写进 `PYTEST_OUTCOME_JSON` 文件。
它与 stdout 无关，且在 `pytest_sessionfinish` **立刻落盘**。
三层判据：**①旁路 JSON（主）** → ②汇总行 → ③进度点。
旁路自身也有否定测试（`collected` 与 todo 不符 ⇒ 判 `error`），
避免"文件存在"变成新的"缺席即通过"。

### 【2】第 1 项验收（`_regr8`，正确解释器）

```
用例 1620｜真正执行 1619｜跳过 1｜失败 0｜错误 0
✅ OK：0 failure / 0 error；纯逻辑模块全部真的执行了
EXIT=0
```
跳过原因行**自己写着判据来源**：
`1 × pytest 退出码 1，但 13 个用例全部有结果（判据：旁路 JSON（collected=13，合计=13））`
⇒ 旁路文件是这次判定**实际用到的依据**。

三轮同机对照（唯一变量 = 代码版本 + 解释器）：

| 运行 | 代码 | 解释器 | 跳过 | 失败 | 错误 | `EXIT` |
| --- | --- | --- | --- | --- | --- | --- |
| `_regr6` | 修复前 | 3.13.12 | 1 | 0 | **1** | 1 |
| `_regr7` | 修复后 | venv（无 torch） | **88** | 0 | 0 | 1 |
| `_regr8` | 修复后 | 3.13.12 | 1 | **0** | **0** | **0** |

### 【3】⚠️ `_regr7` 的 4 个新 `✗` = **我自己选错解释器**（如实登记）

`test_optim_s01/s02/s03/s04` 从 `通过 12/7/16/19、跳过 0` 翻成 `通过 0、跳过 12/7/16/19`：
**四组计数逐个不变、只翻状态** ⇒ 系统性信号。
根因：这四个模块 `@unittest.skipIf(T.torch is None, …)` 门控，
我用 **venv**（`ModuleNotFoundError: No module named 'torch'`）跑 ⇒ 合法全跳过。

⇒ **本机两解释器分工**：
`versions/3.13.12/python.exe`（torch 2.10.0、无 pytest）**跑回归**；
`envs/default/Scripts/python.exe`（无 torch、有 pytest/paramiko）**跑远程工具**。
⇒ 收尾纪律：**汇报"回归通过/失败"必须同时报解释器**。
`run_tests.py` 第 91–97 行**早就写明**了这一点。

### 【4】第 3 项本轮重推闭环

内容 sha256 逐文件比对（114 项清单）⇒ **精确 2 个文件差**：
`scripts/_pytest_outcome_plugin.py`（新增）、`scripts/run_tests.py`（内容变）。
`tests/test_run_tests_harness.py` **不在** delta —— `tests/` 不在 `digest_scope`，本就不部署（预期）。

推送 2/2 OK（判据 = 远端 sha == 本机 sha）→ `stamp --allow-dirty` →
`投放并校验通过`，`tree_digest = 17997a944cb4…` = 本机重算值（125 个 scope 文件）。
远端 `detect_source_identity('/root/ascend_probe_d1')` =
`('16e82d2e…', 'stamp', True, '17997a944cb4…')` ⇒ **`read_source_stamp` 修复二次真机复现**。

反向对照（全部实测）：`no_reproducibility` / `empty_git_source` ⇒ 「缺失…拒绝复用」；
`unknown_git_source='guess'` ⇒ 「不可信」；真实 `EXECOPT4-*` 三份 ⇒ `stamp_stale` ⇒ **拒绝**。

### 【5】第 4/5 项前置再核（全绿，待批准）

> ⚠️⚠️ **2026-10-03 更正：本节那句「全绿」是过度声明，已撤回（原文保留）。**
> 逐条核查确实成立，但**都属"文件/sha/配置"层面**——
> **没有一条能证明脚本跑得起来**。实测第一次真调用即
> `AttributeError: 'dict' object has no attribute 'runtime'`。
> ⇒ 教训：**AST/import OK ≠ 能跑**；"前置已就绪"必须有**执行过**的证据，
> 而不是"检查过文件都在"。更正后的口径见 `OPTIMIZATION_TEST_SESSION.md` §5.9.11(n)。

* 验证脚本本机/远端 sha **逐字节相同**（`1d15804a9edd7abe…`）；远端 AST OK；
  MUSA 激活正常（`torch 2.2.0`、`MODEL_REVISION=a09a35458c70…`）。
* 单变量性再确认：`s01_both_s04_pkv.yaml` 相对基线**只多两行**
  （`persistent_kv_slots: true` + `persistent_kv_capacity: 1024`），
  后者在 `slots=False` 时**不生效**（`real_backend.py` 读取点已核）
  ⇒ 两行 = **一个逻辑变量**。

### 【6】本轮总判定

**通过：第 1、2、3、6 项。不可判（未执行）：第 4 项。阻塞：第 5 项**（依赖第 4）。
**无失败项。** ⛔ 不得开始 OPT-04。

---

## 2026-10-03 用户第 6 条指令：按当前源码复核并修复 6 项

用户指令的一句话概括：**"未执行或未完成不得写成通过"**。
本轮因此先做**缺陷定位**（每个都先用实测复现），再改，再用**破坏性对照**证明判据有效。

### 【1】定位到 5 个真实缺陷（逐一实测复现）

**(a) ★★★ 旁路插件漏记 `teardown` 相位 ⇒ 真实缺陷被读成通过。**
上一轮我为了对抗"stdout 被沙箱垫片摧毁"而加了 pytest 旁路插件（确实解决了那个问题）。
但它只处理 `when == "call"` 与 `when == "setup"`，**完全没有 `teardown` 分支**。实测：

    真实 pytest: 1 passed, 1 error     （夹具收尾抛异常）
    旧版旁路   : passed=1 error=0 failed=0, collected=1, 合计=1

而运行器那条"**计数齐全 ⇒ 豁免非零退出码**"的规则正好拿它当依据
⇒ **我加的"解药"变成了新的遮蔽机制**。

**(b) `verify_s04_reenact_numerics.py` 根本跑不起来。**
实测第一次真调用即 `AttributeError: 'dict' object has no attribute 'runtime'`
（把 `yaml.safe_load` 的 dict 递给 `build_driver`，而真驱动读 `cfg.runtime`）。
⇒ 上一轮那条"脚本已就绪"**是错的**：当时只有 `AST/import OK`，
而 **AST 通过不能证明能跑**。

⚠️ 我中途还**误判过一次**（已自行更正）：先入为主以为 `forward()` 返回内部
`_GroupOutcome`，据此把脚本原来的 `out.slots` 判成"照假驱动写的"。核对公开签名：

    TransformerDriver.forward(batch) -> StepOutcome{slots: list[SlotResult]}   # 真驱动公开接口

⇒ **原来的写法是对的**，是我的判断错了。**把"对的东西"改掉会引入新的缺陷**。

**(c) 固定轮数 `--repeats` 只跑 1 轮，且 rc=0。**
主循环 `pass_index += 1; if not fill_budget: break` ⇒ 轮条件根本没机会生效。
实测（2 格 × `--repeats 5`）：`run_once` 调用 **2** 次、`n_done=2/10`、
`completed_passes=1`、`balanced=**True**`、`failures=[]`、**`rc=0`**。
⇒ 少跑 8 项被读成完整成功；而收尾判据 `balanced` 在**任何整数轮**上都为真。

**(d) 配置身份判据恒判不一致 ⇒ 续跑永远不复用。**
写端 `yaml.safe_dump(cfg.raw, ...)` 是重新序列化（注释/键序被规范化掉），
读端比的却是 `snapshot.read_bytes() == cfg_file.read_bytes()`（原始字节）。
本仓每份配置都带大段注释 ⇒ **永远不可能相等**。报错还说"配置已变"，指向错方向。

**(e) 环境级：测试里的裸 `"bash"` 探测会杀掉整个 pytest 进程。**
`tests/test_exec_opt_plan._bash()` 把裸 `"bash"` 放在候选首位 ⇒ Windows 上解析到
被安全策略拉黑的 `wsl.exe` ⇒ **不是探测失败，而是整个进程被 SIGTERM**
（实测整轮回归直接死掉、连汇总行都没有），`except (OSError, SubprocessError)` 接不住。

### 【2】修法与判据（都做了破坏性对照）

| 缺陷 | 修法 | 破坏性对照（改回旧行为） | 结果 |
| --- | --- | --- | --- |
| (a) teardown | 插件逐相位 + 会话异常 + `expected`/`reported` 集合；判据抽成纯函数 `_classify_pytest_run`（**集合双向对账** + 逐相位 + 会话异常；非零退出码**只在命中已知外部签名**时降级） | 改回"只看 setup" | teardown 用例 **变红**（`0 != 1`） |
| (a) 豁免 | 删除"计数齐全即豁免" | 改回 `notes.append` | 无法归因的非零退出码用例 **变红** |
| (b) 跑不起来 | 改用项目 `load_config` + `dataclasses.replace`；补 logits 摘要 / 重入时刻 KV / token 三方对账 / 预热口径 / 唯一输出路径 / 身份记录 / `--plan-only` | —（原为崩溃，无对照可分） | 新增 23 例 CLI 集成与判据正反例 |
| (c) 少跑 | `break`→`continue`；新增 `completion_flags`（`n_planned_fixed`/`complete`/`under_run`）并据此置 rc；外层编排补 `--baseline`/`--repeats` 透传 | `continue` 改回 `break` | 调用数 `2 != 10`、`6 != 2` ⇒ **2 条变红** |
| (d) 配置 | 统一为参数内容的规范摘要 `config_data_digest()`，两侧**同一解析器** | 退回"比原始字节" | 真实写端快照用例 **变红** |
| (e) wsl | 只探明确的 PortableGit 路径 | —（环境级） | 全量回归不再被 SIGTERM |

### 【3】离线验收：统一一次全量回归（`_regr9`）

```
用例 1698｜真正执行 1697｜跳过 1｜失败 0｜错误 0
✅ OK：0 failure / 0 error；纯逻辑模块全部真的执行了
EXIT=0
```

较 `_regr8`（1620）新增 **78** 例 = harness 12→21（+9）、`test_verify_s04_cli`（+23）、
`test_exec_opt_matrix_main`（+22）、`test_config_identity_contract`（+20）、
`TestLogitsDigest`（+4）。
并且**本轮不再出现**沙箱垫片造成的非零退出码（`_regr9` 无该跳过条目）。

### 【4】真机数值验证：**PASS**（10/10）

重推 8 个文件（125 项逐文件 sha **零差异**）→ `stamp --allow-dirty` ⇒
`投放并校验通过`，`tree_digest = bff7f7080b67914ea05911f9825ee2e181232928d665ede07dc866bf5654fc9f`
= **本机重算值**；远端 `detect_source_identity` 第 4 位带同一摘要。

真机执行 `verify_s04_reenact_numerics.py`（`s01_both.yaml` 派生两臂，`--plan-only` 先验过）：

    telemetry: fastpath=7 fallback=5 resyncs=2 reasons={'new_sequence': 3, 'unequal_kv_len': 2}
    logits 采集点=23
    ✓ 快路径被走过 / ✓ 混长回退发生 / ✓ ★ 重入发生
    ✓ token 逐一相同（23 个）/ ✓ logits 摘要逐点相同（23 点）
    ✓ ★ 重入时刻逐层 KV 逐位相同 / ✓ 结束时刻逐层 KV 逐位相同
    ✓ tokens_out 对账（baseline=23 pkv=23 实观察=23）
    ✓ measured_window 可读 / ✓ 预热口径自洽（no_warmup）
    === PASS ===

⚠️ **必须随结论传播的口径**：`warmup_deduction_verified = false`——
本次**未预热**，故"预热扣除"**未被验证**（`basis=no_warmup`）。
脚本把这一点写成产物字段与 `does_not_assert`，**不得**据此声称已扣除预热。

证据已回拉：`results/raw/_s04_verify/s04_reenact_verify_20261002T194537Z.json`
（本机/远端 sha256 一致 `e14c75e03a7eb8e3…`）。

### 【5】性能实验：**已执行完成**（`EXECOPT5`，2 臂 × 5 轮 = 10/10）

两门都已满足：① 数值门 PASS；② "两臂 × 五次 = 10 项"的完整性已验
（远端 `--dry-plan` = 10 项、真实 `main` 离线模拟 = 10 项、少跑 ⇒ rc=1）。
批次 summary（**第 3 项那套完成度判据在生产里第一次生效**）：

    n_items=10 / n_done=10 / n_planned_fixed=10
    complete=true / under_run=false / budget_limited=false
    completed_passes=5 / balanced=true / n_skipped_existing=0
    dry_run=false / failures=[] / elapsed 27.0 min / avg_item_s=162.2

回拉 10/10 个目录，完整性核对通过（4 文件齐）。

**结果（口径 = §4.4 A4）**

    ★ 尾段 77 步 packΣ（有效样本）: 108.80 → 185.95 ms = +70.9%
      （= **5 轮逐轮配对比值的中位数**；极差 16.3%；**5/5 同号**）
      全窗口 1,080 步 pack_tel      : 20,968.8 → 21,055.9 ms = +0.366%
      **时间权重（尾段/全窗）**      : baseline **0.5189%** / pkv **0.8831%**

⇒ **独立复现 §3.6 E20（+72.0%，3 轮）**，本批 5 轮给 **+70.9%**。
**S04 = 限定实现与负载的局部负结果**（当前实现：每步整段 `copy_`；
当前负载：`short_chat`/并发 8/64 请求/`eager`/`max_num_seqs=32`）。**不得外推。**

★ **换算必须用时间权重，不是数量覆盖率**：`6.74%`（77/1143）是**数量**份额；
尾段的**耗时**份额只有 **~0.52%**（baseline）—— 尾段是单请求纯 decode，
单位步搬运量远低于前段批量步。**加性核对**：`20,968.8 + 77.15 = 21,046.0`
vs `21,055.9` ⇒ 残差 **9.9 ms（0.047%）** ⇒ 全窗 87.1 ms 的差里 **88.6% 来自尾段**。
⇒ 正确表述：**成本真实存在、但只占总耗时的约 0.9%**（不是"被稀释成没有"）。

⚠️ **撤回过一条表述**：原先写的"`fwd_tel` 两臂区间**重叠** ⇒ 只影响 `pack`"**已删除**
—— **区间重叠不是检验**，n=5 下重叠推不出"无差异"；本批只能记"**不足以判定 `fwd`**"。

⚠️ 引用限制：`warmup` 轮数两臂**不同**（2 vs 1）⇒ driver 口径**不可跨臂比**；
每个 run 恰有 1 个 >1 s 的**起手步**（两臂皆有，10/10）⇒ 服务指标需专门口径；
**环境证据不完整**（§4.4 A7：产物里的 `hardware_env` 来自另一个容器）⇒ 论文引用前先解决。
详见 `EXPERIMENTS.md` §1.7 / §3.8 / §4.4 A4·A7。

### 【6】本轮 6 项总判定

**第 1–6 项全部完成**（第 6 项含真机数值验证 PASS 与性能实验 EXECOPT5 已完成）。
**无失败项。** ⛔ OPT-04 仍禁止开始——S04 现已确认为**净成本**（§3.8），
"是否继续 S04"需要新裁定（现判据只覆盖 6.74%）。

---

## 2026-10-03 离线会话：任务 D「已裁定的调度语义修正」

用户裁定：**修两个已裁定的语义缺口**（双预算忽略受阻 decode、`closed_jittered`
提前入队），要求"离线真实执行入口能证明请求不提前放行、受阻 decode 不被遗漏、
建议动作能落实"，并**标记"离线验证通过、真机待验"、不宣称服务收益**。

### 【1】§5.1 双预算：逐 token 护栏被 prefill 预算旁路

**缺口的确切形状**（先把旧证据读清楚，才发现旧探针**没测到**它）：
`scripts/audit_s00_blockers.py` 的双预算探针只跑 `decoding=[]`
（纯 prefill 步），于是它数出的 `n_relaxed_by_prefill_budget=6` 其实是
**按设计**的行为（纯 prefill 步服务 TTFT，此时没有任何 decode token 被推迟）。
⇒ **真正要看的那个数在"混合步"上**，而旧探针从来没跑过混合步。
（教训：**特征化用例的场景选错，会让人以为缺口在 A 处、实际在 B 处**。）

**修法**：`_budget_for` 增加 `decode_at_risk`：
`decode_at_risk = decode_served or decode_blocked`
（`decode_served` = 本步服务了 decode token；`decode_blocked` = 有的 decode 请求
本步**没拿到** token）。为真 ⇒ **必须**用 decode 预算。

⚠️ **默认值取 `True`（保守）**：默认 `False` 会让任何忘记传参的调用点**静默回到
旧缺口**；默认 `True` 最坏只是"该放宽时没放宽"。
⚠️ **没有改任何外部 SLO 取值**（`step_deadline_ms=100` / `prefill_deadline_ms=2000`
原样），改的是"哪一步该用哪一个"。

### 【2】§5.2 `closed_jittered`：抖动依赖 `step` ⇒ 到达时刻是"调度的函数"

**缺口的确切形状**：抖动由 `(seed, req_id, **step**)` 派生；`step` 是**策略**的函数
⇒ ① 到达时刻在跑之前**没有定义**；② 请求仍在 `step_start` 入队 ⇒ 实测 **16/16**
个请求 `enqueue_ms < arrival_ms`（最大倒置 24.78 ms ≈ 抖动上限）。
在真机上它的字面含义是"**请求还没到就已经在队列里等了**"。

**修法**：抖动改由 `(seed, **客户端**, **轮次**)` 派生 —— 两者都是**闭模型负载自身**
的属性（第 k 个补位 ⇒ `client = k % concurrency`、`round = k // concurrency`），
**与调度策略无关** ⇒ 到达成为**输入**。并 `closure_jitter_by_request` **预先算成一张表**
（跑之前算定，不会在等待期间被重算而改变），执行器**等到点才放行**（未到则不 pop）。
两条执行路径（`run_benchmark.simulate` 与 `real_backend`）**共用同一份实现**。

⚠️ **修的过程中自己踩了一个坑（已记 PLAYBOOK）**：第一版用
`state.scheduled_arrival_ms is None` 当"算过没算过"的标志，而它在构造时就被填成了
`r.arrival_ms`（闭环 = 0）⇒ 抖动**一次都没生效** —— 倒置是没了，但 16 个 arrival
全变成 0。**"倒置消失"与"抖动生效"是两个方向，只测前一个会把后一个悄悄弄坏。**
⇒ 于是加了**反向守卫** `n_distinct_arrivals == n_requests` 与 `arrival_span > 0`。
（这正是本项目反复记的"恒定值假通过"的同族：用"抹平"来让一个检查通过。）

### 【3】离线验收（`scripts/audit_s00_blockers.py`，新证据 `results/processed/s00_audit_20261003.json`）

| 验收项 | 判据 | 结果 |
| --- | --- | --- |
| **请求不提前放行** | `jitter.n_enqueue_before_arrival == 0` | **0**（原 16） |
| （反向守卫）抖动仍生效 | `n_distinct_arrivals == n_requests` 且 `arrival_span > 0` | **16/16**，**15,743.2 ms** |
| **受阻 decode 不被遗漏** | `dual_budget.mixed_step.*relaxed == 0` 且 `budget_kinds == ["decode"]` | **0** / `["decode"]` |
| （对照）纯 prefill 步按设计放宽 | `pure_prefill_step.n_relaxed == 6` | **6**（未变，说明放宽本身没坏） |
| **建议动作能落实** | `apply_arm(shadow=False)` 的返回值进入下一步 `plan_step` | ✅（闭环返回值 == 建议值；`shadow=True` 不改参数） |

⚠️ 两条旧的特征化用例（**断言"缺口仍在"**）已按它们自己的约定改成**正确性断言**：
`tests/test_s00_known_gaps.py`（§5.1）与 `tests/test_closed_loop_jitter.py`
（其中"与调度耦合"这条性质**反了过来**，改为"**与调度无关**"）。

### 【4】影响面与"不作废历史数据"

`closed_jittered` **从未用于任何配置或产物**（`configs/` 与 `results/raw/` 均 0 命中）
⇒ 本次语义变更**不作废任何历史数据**。
`prefill_deadline_ms` 默认 `None` ⇒ 双预算路径默认关闭 ⇒ 既有批次行为不变。

### 【5】状态：**离线验证通过、真机待验**（本项不宣称任何服务收益）

同时**登记两处待冻结语义**（`NEXT_SESSION_TASKS.md` 的 FZ-1 / FZ-2）：
① "双预算"是"逐步择一"（当前）还是"步内拆分两块"（需成本模型给分量预测）；
② "受阻 decode"的判定口径是"本步没被服务"（当前）还是"逐请求间隔将超标"。
两者都会**改变研究问题**，在裁定前不得据此改判据或重跑批次。

---

## 2026-10-03 离线会话：任务 E / F（后续实验准备 + 论文与交接收尾）

### 【E】后续实验的离线准备 —— 清点与三类划分

**成本模型现状（从产物现读）**：部署表 `model_v7.json` = **107 键**、
`grid_sha256=5c3b66f75c89522c`、batch 1–8 / pf ∈ {0,128,256,512,1024,2048} /
kv ∈ {0,512,768,1024,1536,2048}。**`model_v8.json` 不存在**（v8 只在探针层）。
v8 网格的 `axes` 比 v7 多一个 **`prefill_tok=192`**。

| 类 | 内容 |
| --- | --- |
| **可复用** | v7 的 107 键（既有全部批次用的就是它） |
| **需锚点核验** | v8 的 24 点 = `{b1,b2,b4,b8} × {pf128,192,256} × {kv1024,2048}`；其中 **16 点落在 v7 已采形状上**（跨会话 ⇒ 天然锚点，判据：中位差 ≤ 5%）；**8 点净新增**（pf192）。⚠️ 探针产物自注"**无 harness 盖章、不得进结论行**" |
| **真正缺失** | **`pf=384` 两处都没有**（v7/v8 的 pf 列表均不含）；batch > 8；kv > 2048 ⇒ **本次不补**（补 = 扩网格 = 另开一轮），**不伪造标定** |

**只派生真正缺的两份**：`configs/mthreads/next_batch/{static_small,cost_dual}.yaml`
（生成器 `scripts/gen_next_batch_configs.py`，带 `--check`）。
`static_nochunk` / `threshold` / `cost_single` **复用**冷冻的第 14 步配置 ——
复制一份就是第二处真相。

### ★ 发现并修掉一个真实接线缺口：双预算**无法从配置开启**

`prefill_deadline_ms` 此前**在配置层不存在**：`ControllerConfig` 没这个字段，
`arm_factory` 也从不传它 ⇒ 双预算**只在代码与单测里可达**，
**任何批次都开不了它**，产物里也就永远看不到"这个臂到底开没开双预算"。
（本仓"机制写了、没接线"的同族。）

⇒ 已接通：`ControllerConfig.prefill_deadline_ms`（默认 `None`）+ `describe()` 落盘
+ `arm_factory` 透传。**默认关 ⇒ 既有批次逐字节不变。**

### 两处**程序化**核对（不靠目测）

1. **S01 复核对实测差 3 个键**（`logits_last_only` / `sample_needed_only` /
   **`model_revision`**）—— 第三个不是第二个变量，而是 `main` → 钉死 commit 的
   **写法差异**；成立前提是"`main` 仍指向钉死的 commit" ⇒ 列为**开跑前置**
   （离线无法验证，**不写成恒真的测试**）。
2. `fixed_cost` vs `cost_dual` 差集**恰好**是 `controller.prefill_deadline_ms`
   （已实现为测试）。

### 离线 dry-plan 的**已验证边界**（含否定对照）

* `static_small.yaml` **本机可 dry-run**，且**把 socket 关掉再跑一遍仍成功**
  ⇒ 该路径**不连远端**；产物 `data_label=TEST_DATA`、`backend=simulate`。
* `cost` 臂**本机跑不了**：成本模型加载时**指纹比对正确地拒绝**
  （本机 torch 2.10.0+cu130 vs 标定 2.2.0、transformers 4.50.0 vs 4.46.3、
  python 3.13.14 vs 3.10.8）⇒ **cost 臂的 dry-plan 与正式运行都必须在标定环境做**。
  ⚠️ 这**不是"环境坏了"**，而是"本机不是标定环境"。

**预算**：S01 复核 ≈ **27 min**（10 项 × EXECOPT5 实测 162.2 s/项，**同 workload 同配置族**）；
机制小批 **25 项、节拍「待估」**（该 cell 的 prompt/gen/concurrency 与 exec_opt **不同**
⇒ 不照搬，须先测 1 项 pilot）。

### 【F】论文与交接收尾

* **限制与失效分析（不依赖新数据）已补入权威源** `paper/sections/08_limitations.md`：
  新增 §8.1 的**环境绑定**边界、§8.6 的 **S04 限定局部负结果**与
  **"自证文件变成新的遮蔽机制"**这条方法学失效、§8.8 **统计表述纪律**、
  §8.9 **证据资格分层（T1/T2/T3）**。
* `paper/sections/06_scheduling.md` 新增 §6.8「已冻结的设计（尚未开跑）」——
  只记录**设计**，不含任何测量值；并写明"**不以新增算法 / S04 扩展 / 图捕获为前置**"。
* `RESEARCH_PLAN.md` 新增 §0 状态更新：**研究问题与贡献边界不变**，
  更新的是**边界被收紧**（环境绑定、证据分层、统计表述）与**执行优先级**；
  成本模型失效与适用边界这条**主线保留**。
* `PROJECT_MAP.md` 状态块更新到 2026-10-03（含 T1/T2 事实与 S04 归档定性）。
* **`STATE.json` 已压缩为"恢复指针"**（23.0 KB → **12.3 KB**）：不再复述实验事实，
  只留 `next_action` + `decisions_2026_10_03` + `recovery_pointers`；
  压缩前全文备份为 `STATE_FULL_20261003.json`（同目录）。
* 论文正文已**按项目流程重建**（`scripts/build_manuscript.py`）：`paper/manuscript.md`
  1278 → **1468 行**，占位 **18 处**（全部为"待跑实验"，**不能**用现有数据填）。

---

## 2026-10-03 真机会话：**S01 独立复核**（只推这一批，完成后停止）

用户裁定：本会话**只推进 S01 复核**，完成后停止；**不链式**启动调度机制批次 / S04 /
图捕获。FZ-1/FZ-2 只影响调度研究，**不阻塞**这次控制器关闭的 S01 复核。

### 【1】开跑前的三处小修正（都是"看起来在查、其实没查"）

| # | 病灶 | 修法 |
| --- | --- | --- |
| ① | 环境指纹读取：读的是 `d.get('device_count')` 与 `software.mthreads_driver_version`，**这两个键在真实 `env.json` 里不存在** ⇒ 恒 `None` ⇒ **"设备数为正"与"驱动已确认"两条判据一次都没执行过** | 新增 `read_env_fingerprint()` 按**真实路径**读（`device.count` / `device.model` / `discovery.values.MTT_DRIVER_VERSION` / `MUSA_VERSION` / `platform`）；新增 `env_fingerprint_acceptance()` **六项无条件核对**（设备可用 / 设备数>0 / 平台与型号 / hostname 与实时实例一致 / 采集时间可解析 / 必填字段）；**首次采集同样查** —— 判据签名里**没有 `prev`**，所以"首次放宽"在接口层面不可能发生 |
| ② | S01 复核缺一个**同起点**的对照臂（原基线 `model_revision=main` 是引用，与 `s01_both` 的钉死 commit 不同 ⇒ 不是单变量） | 新增 `s01_reference.yaml` = 平台基线 + **同一个钉死 revision** + 两开关全关 ⇒ 与 `s01_both` 差异**恰好是两个开关**（程序化核对）。**平台基线未改**（git clean） |
| ③ | AB/BA 顺序表**两处各写了一遍**（`build_plan` 与执行循环）⇒ 只改一处会让"计划 ABBA、实际 ABAB"**且两边都不报错** | 提取 `order_for_pass()` 为唯一来源；新增 `--order {block,abba}`（默认 block = 历史行为），外层 `exec_opt_batch` **显式透传** |

**指纹夹具**：用**远端真实产物**存成 `tests/fixtures/env_fingerprint_real_mthreads.json`
（不是手写的想象结构）；34 条集成正反例；**破坏性对照**：短路接受判据 ⇒ **22 条变红**。

### 【2】部署与验收

* 连通性：端口 **23576 未变**，单次 `exec` 通过（未反复重试、未猜凭据）。
* 远端**无运行中实验**（无 python 进程 / GPU "No running processes" / 4MiB 空闲）。
* 推送：整推成功并当场校验；**第二次整推 SFTP 中途断连**，留下**截断的 `_push.tar`
  （1.28 MB / 3.33 MB，从未解包）** ⇒ 诊断后**清残留 + 分段推送**成功。
  ⚠️ 定式：**推送中断后先查"落点是否半新半旧"**，不要直接重试。
* **四类身份**：源码 `tree_digest=472697170d43…`（`git_source=stamp`、
  `git_commit=16e82d2ea117`、`dirty=true` 如实）；配置 = 两臂各自快照摘要；
  模型 `a09a35458c70…`；环境 = 本批前置**重采**的指纹（`05264bbf0d`、driver `2.7.0`、
  MUSA `3.1.0`、torch `2.2.0`、torch_musa `1.3.0+81caf0a`、device_count `1`、`MTT S4000`）。
  **摘要三方一致**；无 `PENDING_BATCH_STAMP`。
* 短启动：远端导入自检 IMPORT_OK + 负对照正确。
* **数值正确性验收 PASS**：真机短跑两臂，**3 条请求 token 序列逐位相同**
  （摘要 `1c4ecd6016654de0…`）。⚠️ 两项如实标注：`--dry-run` **验不了**这两个开关；
  logits 摘要两臂都没采集 ⇒ 记"**未验证**"。
  ⚠️ 短启动产物标 `NOT_MEASURED`（`--out results/raw/_smoke` 会让它去 `_smoke/_env/` 找指纹）
  ⇒ **真实批次必须用默认 `--out results/raw`**。
* ★★ **2026-10-03 数值验收脚本补强（上一版缺四项判据）**：
  ① 强制**预期请求集合**（旧版只比"两边都有"的 id ⇒ 一条整轮丢失也能过）；
  ② **每请求 token 长度与总数对账**（旧版只看"两臂相等" ⇒ 两臂一起错看不出）；
  ③ **中间 chunk 的 `need_token=False`** 与 ④ **混批 True/False** 场景；
  ⑤ **跳过计数 `n_sampling_rows_skipped` 必须 > 0**（为 0 说明这条路根本没走到）；
  ⑥ logits 未采集保持"未验证"。
  补强过程由**离线集成测试**（假驱动，无需 GPU）驱动，并**真抓出脚本自身三个缺陷**：
  假阳性 "slot 缺失"、假阳性 "位置被污染"、以及"请求集合判据恒真"。
  破坏性对照：注入 4 处破坏 → 4 红，还原 → 10 绿。
  数值产物改用**唯一编号**（`--run-id`，默认时间戳）并保存命令/退出码/源码/模型/环境身份。

### 【3】批次结果（`S01RECHECK`，2 臂 × 5 轮 AB/BA，10/10）

`complete=true` / `under_run=false` / `failures=[]` / `budget_limited=false` /
`completed_passes=5` / `balanced=true`；**29.0 min**（`avg_item_s=173.9`，与 162.2 s 的历史估计同量级）。

**判定（判据随 `scripts/analyze_s01_recheck.py` 版本化，非独立封存预注册：
主指标 TTFT p50，≤−2% 且 5/5 同向）= SUPPORTED**：

* **TTFT p50 配对中位 −4.31%（5/5 更快）**；p95 **−4.13%**、p99 **−4.04%**（均 5/5）。
* TPOT p50 −0.06%（3/5 vs 2/5，区间跨 1.0）⇒ **不足以判定**（机制上自洽：S01 减每步固定开销）。
* 吞吐 / 达标吞吐 +0.21% ⇒ **改善 4/5**（吞吐越大越好）。
* **达标率两臂同为 2/64 = 3.125%** ⇒ 本负载**不区分**两臂质量。
  ⚠️ **2026-10-03 更正**：初版此处写"达标率两臂同为 1.000"——那是把**配对比值**
  （0.0312/0.0312 = 1.0000）读成了**绝对率**；初版还把吞吐方向写成"4/5 更慢"，
  那是按时延方向（越小越好）判吞吐所致。两处均已按四元口径（对照绝对 /
  处理绝对 / 配对变化 / 分母）修正，**原始数据未改**。
  ⚠️ 并且**绝对达标率只有 3.125%** 这件事本身很重要：TPOT p50 ≈ 117 ms
  **多数超过 SLO 的 100 ms**。
* `n_failed = n_unfinished = 0`（两臂 5/5 轮）；每轮 >1 s 步数 **1/1（起手步，两臂相同）**。
* **噪声底（本批同协议自比）**：对照臂跨 5 轮极差 `ttft_p50` **0.656%** ⇒ 效应 ≈ **6.6×** 噪声底；
  且 5 轮配对比值**全部 < 1**（上界 0.9630）。
* 与 `EXECOPT3` 的 **−4.2%** **独立复现且高度一致**；而本批四类身份全闭合 ⇒ **T1**。
* ★★ **2026-10-03 分析器补强 + 一个真实误判（必须记）**：
  * **误判经过**：我跑了同配置的否定对照（`--control s01_both --treatment s01_both`），
    控制台**正确**打出 `UNKNOWN`，随后读 `results/processed/s01_recheck_analysis.json`
    却看到 `SUPPORTED` / `gates_ok=true` ⇒ **一度判定"闸门没传播到产物"**。
  * **真相**：脚本**只在传了 `--json` 时写文件**（`if args.json:`）。我的否定对照
    **没传 `--json`** ⇒ 磁盘上那份是**上一次**（好输入）留下的**旧产物**。
    `stat` 证实 mtime 未变。⇒ **不是闸门失效，是旧文件冒充新结果。**
  * **危险方向恰好相反于直觉**：人会去"修一个没坏的闸门"，而真正的问题是
    "**产物无法自证它是哪次调用写的**"。与用户长期记录同族（判定"文件没生成/没更新"前
    先怀疑它是旧的）。
  * **修法（两处）**：① 产物新增 `written_at` + `argv` ⇒ 任何人可自证它是哪次调用写的；
    ② 未传 `--json` 时**显式打印**"本次未写任何文件；磁盘上的是旧产物（含 mtime），
    与本次运行无关"。
  * **回归测试**：新增 `TestJsonWriteIsSelfIdentifying`（5 条，真跑子进程 + 真产物）。
    **破坏性对照 3 轮**：注入"总是写文件"⇒ 2 红；删 `written_at`/`argv` ⇒ 2 红；
    把 payload 写死 `gates_ok=True` ⇒ 1 红（正是我误判的那个形态）。还原 ⇒ 26 绿。
  * ⚠️ 顺带**二次踩到**用户已记过的坑：用 Windows 原生 Python `shutil.copy('/tmp/...')`
    恢复备份 ⇒ `FileNotFoundError`（`/tmp` 被解析到 `C:\tmp`）。改用 **Git Bash `cp`** 成功。
    ⇒ 规则复述：**本地路径一律用 Windows 原生绝对路径**；`/tmp` 只对 MSYS 工具有效。

### 【4】顺带发现的一处字段误命名

`exec_opt_summary` 的 `stopped_by_budget` 原写作 `budget_s is not None`
—— 那说的是"**有没有配**预算"，不是"**是不是被它停下**"。
实测本批 `budget_limited=false`（10/10 跑满）却同时 `stopped_by_budget=true`
⇒ 读汇总的人会以为这一批被截断了。**已改为如实取 `budget_limited`**，
并另存 `budget_configured` 表达"是否配了预算"。
⚠️ 该改动**发生在批次之后**：本批数字以产物内的 `tree_digest=472697170d43…` 为准。

### 【5】状态

**S01 的实际部署收益得到独立支持**（限定范围见 `EXPERIMENTS.md` §3.9）。
按指令**到此停止**：本会话**未**启动 25 项机制批次、**未**动 S04/图捕获。

## 2026-10-03 离线会话：任务 A/B/C/D（数值验收 + 有效数值门 + 可执行配置 + pilot 流程）

**范围**：只做离线修复与可执行协议，**未启动 GPU 性能矩阵**，**未重跑** S01RECHECK
的 10 份性能原始产物（`results/raw/**` 只读）。四项全部完成；组合回归 **149 passed / 3 skipped**
（3 个 skip 是 GPU 依赖项，离线预期）。

### 【1】任务 A：把数值验收从"能跑"改成"有意义的排程"

**问题**：旧验收的请求序列不构成**合法**排程（无法在一次前向组里同时满足
"哪些 slot 来自 prefill、哪些来自 decode"），于是用 `FakeDriver` 也能"通过" ⇒
**验的是 harness 自己编的序列，不是真实执行器**。

**做法**：写死一套**固定、合法**的 11 步排程，并用**真实 CPU 微型驱动**跑（不是 FakeDriver）：
① s0 完整 prefill 64(True)、s2 完整 prefill 80(True)；② s1 先 prefill 70/72(False)；
③ 同一前向组内 s0 decode1(True)、s1 prompt1(False)、s2 decode1(True)；
④ s1 最后 prompt1(True)；⑤ 固定 6 轮三请求 decode(True)，末轮 s1 再 decode1(True)。

**逐步断言**（每步都查，不是只看最终总数）：返回 slot 的**数量**、**唯一性**、
**请求集合**、**输出位置**。对账结果：**每请求 8 token / 总 24 / False 行 2 个 /
真实混合组 1 个** —— 与预注册期望**逐项相等**。
产物 `results/processed/s01num-cpu-20261003T092417Z.json`，`data_label=LOCAL_CPU_MICRO`，
**已存在即拒绝覆盖**（防"新跑失败、旧文件冒充"）。
**破坏性对照 6 处注入全红**（含"把 False 行改成 True"、"篡改输出位置"两类）。

### 【2】任务 B：分析器加"有效数值门"（恰好 5 对有限且合法）

**问题**：旧 `verdict()` 只看身份/配置闸门；主指标若出现 **NaN / Infinity / 零分母 / 缺轮次**，
会被静默算进"配对数" ⇒ 可能用**4 对**甚至**含 Inf 的比值**宣布 SUPPORTED。

**做法**（`scripts/analyze_s01_recheck.py`，671 行）：
* `_finite(x)` + `check_valid_numeric(summary, *, expect_pairs)`：要求
  `n_valid_pairs == expect_pairs` **且** `n_ratio_invalid == 0`；
* `paired_rows` 重写为 `_finite(vc) and _finite(vt)`，分母 0 ⇒ **显式 `None`**（不填 0）；
* `verdict(..., valid_ok, valid_problems)`：数值门不过 ⇒ 返回 **`UNKNOWN`** +
  **逐条原因**；`main()` 里 `gates_ok = base_gates_ok and ok_v`（**拆成两个变量**，
  避免自引用）；
* 新增打印块 `-- ③b 有效数值门`，payload 增 `valid_numeric_ok`/`valid_numeric_problems`/`base_gates_ok`。

**反例用真实 `main` 入口跑**（子进程、拷贝真产物、只改一份 `meta.json`），四类各一条：
**四轮缺失 / Infinity / NaN / 零分母** ⇒ 三者都断言：**exit≠0 + 输出 `UNKNOWN` + 不是 SUPPORTED**。
**保留** S01RECHECK 性能产物、**不重跑**：真产物的结论仍是 **TTFT −4.31%（5/5）SUPPORTED**。

**破坏性对照两轮**：① 放宽比较 ⇒ **7 红**（含 main 入口项）；②
`gates_ok = base_gates_ok`（**接好了但没通电**）⇒ **4 红**（含零分母那条）。
第 ② 轮是本轮**最有价值的一次注入** —— 它证明"闸门存在"与"闸门被使用"是两件事。

### 【3】任务 C：把下一批草案变成**可执行配置**（不是文档描述）

新增 `scripts/gen_prereg_batch_configs.py`，冻结并**写入实际 YAML**（不靠默认值）：
`EXTERNAL_TPOT_SLO_MS=100`、`EXTERNAL_TTFT_SLO_MS=2000`、`INTERNAL_STEP_BUDGET_MS=120`、
`prefill_deadline_ms=EXTERNAL_TTFT_SLO_MS`、`COST_MODEL_PATH=configs/cost_calib/model_v7.json`、
`MODEL_REVISION=a09a3545…bc28`、`S01_KNOBS={logits_last_only, sample_needed_only}`、
`max_num_seqs=8`、`STATIC_SMALL_CHUNK=1024`。生成三份：
`configs/mthreads/prereg_batch/{cost_single,cost_dual,static_small}.yaml`。
**单变量已程序化证明**：`cost_single` vs `cost_dual` 逐行 diff **恰好 1 行**
（`prefill_deadline_ms: null` ↔ `2000`）。

★★ **本轮最有价值的一个真实发现（我自己的测试抓到了我自己的 bug）**：
我最初把 `controller.step_deadline_ms` 写进 YAML ⇒ **7 个测试全红**，
报 `ValueError: 配置段 'controller' 含未知字段 ['step_deadline_ms']`。
**真相**：`step_deadline_ms` **不是** `ControllerConfig` 的字段，它是在
**`benchmark/arm_factory.py:99` 派生**出来的（`step_deadline_ms=cfg.controller.target_tpot_ms`）。
⇒ 定式：**"派生量"不许当"配置项"写**；文档里写"120 ms 内部预算"，
**不等于**配置里能有一个叫这个名字的键。已把该派生关系写进生成器注释与测试。

**离线代表状态枚举**（`scripts/enumerate_prereg_states.py` → `results/processed/prereg_states.json`，
`evidence_class=TEST_DATA`）：

| 状态 | 候选 | 预算分支 | 可行 | 回退原因 |
| --- | --- | --- | --- | --- |
| 纯 prefill | 命中 | prefill / 2000 ms | ✅ | — |
| 纯 decode | 命中 | decode / 120 ms（预测 95.64） | ✅ | — |
| 混合步 | 命中 | decode / 120 ms（预测 **591.07**） | ❌ | 超预算（预测远大于 120） |
| 查表缺失 | **未命中** | — | ❌ | **lookup 未命中**（segmented 外推 280.16，仅参考） |

⇒ **候选命中率 3/4**；**预算分支有差异 1/4**；**双预算可行 2/4 vs 单预算 1/4**
⇒ **两套预算给出不同可行集 ⇒ 双预算机制在本批**确实**被行使**（不是"配了没用"）。
⚠️ 这是**离线演练（`TEST_DATA`）**，**不代替真机标定**。
按建议**仅采用现有 FZ-1 A / FZ-2 A 语义**，**未扩展**步内拆分或逐请求 deadline 算法。

**破坏性对照三轮**：C1 内部预算 120→100 ⇒ 2 红；C2 删 S01 开关 ⇒ 1 红；
C3 把 `pure_prefill` 标成 `decode_at_risk=True` ⇒ 3 红（**改的是分类前提，不是阈值**）。
还原后 24/24 绿、无 `INJECTED BREAK` 残留。

### 【4】任务 D：pilot 流程拆成"开跑前检查"与"扩大条件"

`NEXT_BATCHES_PREREG.md` §10 重写为 10.1–10.4：
* §10.1 **pilot 目的**＝检查**机制 / 覆盖 / 节拍**，**不用于宣布性能优势**；
* §10.2 **开跑前检查** A1–A8（全属**前置**：远端可达 + 指纹 + 配置显式性等）；
* §10.3 **扩大条件** E1–E4：**废止"仅因静态对照达标率为 0 或 1 就停"**，
  改为 **(0,1) 开区间替代表** —— `0.0` ⇒ 先分辨成因；`1.0` ⇒ 改 cell 后**另开 pilot**；
  **两者都不是"停"**（附 🚫 废止规则引用块，保留原文以备溯源）；
* §10.4 **预算统一**：**45 min 上限 / 30 min 预期**，单点定义；
  `--budget-minutes` 是**干净退出守门**而**不是** timeout。§11 表格改为**只引用 §10.4**，
  消除第二处数字（破坏性对照 D2 正是"在 §11 重新插一个预算数"⇒ 1 红）。

**破坏性对照两轮**：D1 复活旧停止规则 ⇒ 2 红；D2 §11 重插预算数 ⇒ 1 红。还原后 20/20 绿。

### 【5】两处更正与一个静默陷阱（必须记）

① **`-r5` 不存在 —— 6 个端到端测试静默 skip**。
我的 `_has_artifacts` 探测 `-r1..-r5`，但真实布局是 **`-`（第 0 轮）+ `-r1..-r4` ＝ 5 轮**，
**没有 `-r5`**。`skipTest` 让"**没跑**"看起来像"**通过**"。已修为
`RUNS = ("", "-r1", "-r2", "-r3", "-r4")`。⚠️ 同族：本文件此前已记过一次
"指标路径写错（`summary.json` vs `meta.json`）"⇒ **先 `sorted(keys)` 看一眼**。

② **破坏性对照的注入本身可能非法**。第一次注入写成
`if False:  # INJECTED BREAK` + 缩进体 ⇒ `IndentationError`，
"红"是因为**语法错**而不是因为**断言被触发** ⇒ **假对照**。
改为**合法但错**的变异（`if n_valid < 1:`）后才有意义。
⇒ 定式：**注入必须是"合法的错"，不能是"跑不起来的错"。**

③ ⚠️ **端口根因措辞更正（文档层）**。`STATE.json` 此前把
端口"**已变**"写成了事实，与"**根因保持待确认**"相冲突。已改写
`remote.reachable_2026_10_03` / `reachability_error_2026_10_03_late`，
并新增 `port_2026_10_03_late_note`；`NEXT_BATCHES_PREREG.md` §12 同步加
"**诊断签名只是假设（不是结论）**"块。⇒ **本轮未定位根因 ⇒ 不写成"端口已经变化"。**

### 【6】状态

A/B/C/D **四项均完成**，全部离线。**GPU 性能矩阵未启动**，S01RECHECK 的 10 份性能原始产物
**未重跑**。**远端仍不可达**（TCP 可连、recv 无 banner），根因**待确认** ⇒
**下一批无法开跑**（G5 为唯一硬阻塞）。尚需真机完成的最小任务见 `NEXT_BATCHES_PREREG.md` §10.2/§12。

---

## 2026-10-03 离线会话：真实决策链演练 + pilot 执行清单 + 真机验收入口 + 四项小修

**范围**：用户明令「**保留已通过的固定数值排程及 S01RECHECK 性能产物**，本轮只完成离线收尾，
**不启动 GPU 性能矩阵**」。⇒ 全程**未碰 GPU / 未 SSH / 未重跑**任何性能产物
（`results/raw/**` 只读）。

### 【1】把代表状态演练接到真实配置与真实决策链

**旧做法的问题**（本仓已犯过）：`enumerate_prereg_states.py` 把 `(batch, pf, kv)` **手喂**给
`CostModel.predict` ⇒ 它只证明**预算函数**会怎么算，**完全不能推出**控制器在真实队列上会选什么。
用户要求接上真实链路：`load_config → build_arm → plan_step/CostPolicy.decide`。

新脚本 `scripts/dryplan_decision_chain.py`：
* 四个状态都写成**计划负载下实际可达**的队列形状（`PLAN = prompt 1536 / gen 32 / conc 16 /
  max_num_seqs 8`）：`batch_start_prefill`（只有 waiting）/ `steady_decode` /
  `mixed_admit`（conc 16 > max_num_seqs 8 ⇒ 有排队）/ `contended_prefill`（prefilling 非空）。
* 状态描述 → 真实 `waiting/decoding/prefilling` 由 `_queue_for()` 翻译；`(batch, tokens_prefilled)`
  是 `plan_step` **派生**的，**不是脚本挑的**。
* `ControllerConfig` 是 **frozen** ⇒ 离线改 `cost_model_fingerprint_source`（`observed`→`declared`，
  否则 `load_offline` 会正确地拒绝）必须用 `dataclasses.replace`，并把这次放松**记进产物**。

**结果（如实，且不美化）**：
* ★★ **两臂在四个状态上动作完全相同** —— 本批**不要求**两臂必须不同，只要求"决策过程真实、结果如实"。
* `batch_start_prefill`：**查表缺失**（见下）。
* `steady_decode` / `mixed_admit`：退化为**最经济候选**（`115.5 + host 25.3 = 140.8 > 120 ms`）。
* `contended_prefill`：**退化严重**（`833.6 + 17.9 = 851.5 > 120 ms`）。

**★ 新发现：`model_v7` 网格存在结构性空洞。**
批次**起点步**的形状是 `(batch ≥ 2, kv_len = 0)`；而 v7 只在 `batch = 1` 采过 `kv_len = 0`
（`batch ≥ 2` 的 `kv_min = 512`）。⇒ **起点步构造上不可预测**，不是"模型不准"。
已对着原始标定（`results/raw/_costcalib/costcalib_v7_s1/s2.json`，222 点）与完整 batch×kv
交叉表核实。**保留缺失结果，未用手选命中点替代。**

**缺失形状表 + 最小补标定方案**（产物 `results/processed/dryplan_decision_chain.json`）：
**另开一轮**（原地改会让 `grid_sha256` 变 ⇒ 血缘断），在 `kv_len = 0` × `batch ∈ {2,4,8}` ×
`pf ∈ {512,1024,1536}` ≈ **9 点**补采，绑定到计划 cell。
⇒ 结论：**先补标定，不要直接扩大为性能收益矩阵。**

**旧四点表降级**：`enumerate_prereg_states.py` 改称**预算函数示例**，
`artifact_role = budget_function_example`、`not_evidence_of = 本批机制被行使`；
输出不再声称"机制被行使"（`tests/test_prereg_batch_configs.py` 同步改写）。

### 【2】pilot 三臂各一次（计划期即可核对，避免默认跑 15 次）

新脚本 `scripts/pilot_execution_plan.py`：三臂**只此一处定义**，`repeats = 1`，
`EXPECTED_TOTAL = 3`。**真的**跑 `run_exec_opt_matrix --dry-plan` 并解析 JSON 交叉核对
（`cells == ["baseline","cost_single","cost_dual"]`、3 项、`repeats == 1`）。

**两个默认值陷阱**（"以为 3 项、实际 15 项"的成因，均已显式规避）：
| 陷阱 | 默认 | 后果 | 正确写法 |
| --- | --- | --- | --- |
| `--repeats` 不给 | 退到 `--fill-budget`（3~50 轮） | 3 → 15 | 显式 `--repeats 1` |
| `--baseline` 用默认 | `cells = ["baseline"] + arms` | 3 臂 → **4** 格 | P1 走 `--baseline static_small.yaml`，`--arms` 只放 cost 两臂 |

FZ 状态：三臂都用**当前实现**语义（**FZ-1 = A 逐步择一** / **FZ-2 = A 本步没被服务**）
⇒ 结论**只能**表述为"在（逐步择一）语义下"，**不得**写成"已选定最优语义"。

### 【3】真机数值验收入口

`scripts/verify_s01_numeric_acceptance.py`：
* 真机命令**必须显式** `--driver auto`（`--driver` 默认是 `cpu` ⇒ 漏写会**静默**跑 CPU 微型
  驱动、产物标 `LOCAL_CPU_MICRO` = **把本机证据当 S4000 证据**）。新增 `--require-real-hw`：
  非 `auto` ⇒ **exit 2**，并在报错里引用那个 `cpu` 默认值。
* 新增 `_hardware_problems()`：**拒绝式**核验 ① `hardware_backed` 为真 ② 设备含 `musa`
  ③ `kind == transformers` ④ `model_revision_resolved` 存在且 == 请求值 ⑤ `n_sampling_rows_skipped > 0`
  （优化真的生效）。任一条不成立 ⇒ 不得 PASS。
* 产物新增**四类身份绑定**（source / config（含 sha256）/ model（requested + resolved）/ environment）
  + `evidence_scope`：**明禁**把 `LOCAL_CPU_MICRO` 升级为 S4000 证据。

已用系统 Python（torch 2.10.0+cu130）跑通 CPU PASS 路径：`identity 四类`齐、
`model.revision_requested = a09a35458c702b33`、`environment.device = cpu`、`hardware_backed = False`、
**产物 CR = 0**。

### 【4】四项小修

**(a) TTFT 两侧有限且 > 0**：`analyze_s01_recheck.py` 新增 `_finite_positive` +
第 ⑤ 条检查（`POSITIVE_LATENCY_METRICS = ttft_p50/p95/p99`，含 **③c** 打印块）。
`0` 与负值都能穿过 `_finite`，但在时延上不可能发生 —— `ttft = 0` 会被读成"零延迟"
（比任何真值都好）⇒ 把"没测到"伪装成"大幅改善"。**计数类不受此约束**（`n_long_fwd = 0` 是合法结果）。

**(b) 非有限诊断值 → `null` + 错误说明 + 严格 JSON**：两个脚本都加 `sanitize_json`
（NaN/±Inf → `None`，逐条记进 `_json_notes`），写出改用 `allow_nan=False`。
理由：默认 `allow_nan=True` 会写出 `NaN`/`Infinity` —— **非标准 JSON，任何严格解析器都拒收**，
而写的一方**看不到异常**；`allow_nan=False` 单独用又会**抛异常** ⇒ 产物写不出来（而那正是最需要它的时刻）。

**(c) 本机必测失败不得转 skip**：全仓 13 处裸 `skipTest` 逐条分类（仓库内输入 vs 外部依赖）。
新增 `tests/_tiny_support.require_local_file`（缺 ⇒ **AssertionError**）与
`ExternalArtifactMissing`（外部依赖专用 skip 类型）。
仓库内输入的（`configs/**`、`results/processed/prereg_states.json`）改为**硬失败**；
`results/raw/**` 派生的端到端用例改用带理由的 `ExternalArtifactMissing`。
新增 `tests/test_must_run_not_skipped.py`（含"产物在本机 ⇒ 端到端**必须 0 skip**"的**实跑**断言）。
**破坏性对照**：把 `prereg_states.json` 临时改名 ⇒ **2 failed**（不是 skip）；改回后恢复。

**(d) 删除「P1 必须落在 (0,1)」规则**：上一轮把「达标率 ∈ {0,1} ⇒ 停」换成「必须落在**开区间
(0,1)**」—— **同族错误**：仍用一个数值区间决定"要不要扩大"。终裁 = 扩大与否看
**E1（机制是否被行使）+ E4（身份一致）**；达标率的三个取值**只用于解释**，
**不单独构成"停"或"续"**。两条旧规则都进 blockquote **登记为废止**（撤回要留痕）。

**(e) 预算说明与实际停止逻辑一致**：§10.4 加入 `benchmark/exec_opt_plan.py:157 decide_stop`
的**真实语义表**（5 种情形），并写清 ① **只在整轮边界**决定要不要开下一轮
② **不打断**正在跑的 run ③ 真实动机是**保持重复均衡**（宁可少一整轮，也不要"有的臂 5 次、
有的臂 4 次"）④ `avg_item_s is None` 时不因预算停。§11 同步。

### 【5】测试与结论

改动文件全部 CR = 0（LF-only）。逐文件现跑见交付报告。
**S01RECHECK 分析器在新 ③c 门下仍 PASS**：TTFT p50 **−4.31%**（5/5 同向）、
`ttft_p95 −4.13%`、`ttft_p99 −4.04%`，身份/配对/完整性三闸门全过，两臂 TTFT 非正值 **0/0**。

### 【6】状态

四项均完成，全部离线。**GPU 性能矩阵未启动**，S01RECHECK 的 10 份性能原始产物**未重跑**。
**远端仍不可达**（TCP 可连、recv 无 banner），根因**待确认**。
⇒ **下一步**：先补 v7 网格的**起点步空洞**（另开一轮标定，≈9 点）再谈扩大；
pilot 三臂清单已就绪、待远端可达（G5 是唯一硬阻塞）。

## 2026-10-03 · 执行形状离线修复与集中真机准备

用户要求先做完离线工作，真机部分后续一起做。本轮未 SSH、未使用 GPU、未检查远端状态。
前段“先补9点”“只剩连接”“四态已证明可达”口径撤回：预测把接纳请求数当执行数，
且部分队列重叠；合法合成状态仍须单独证明生产可达性。

生产新增共享 `execution_shape`：只有有计算配额的请求参与，包含续prefill，保留逐请求
token/KV/need_token与实际分组。复现8接纳→1执行/两条1536+512，修复步后队列传递、
decode KV减1、waiting缓存KV推导，以及成本排序漏host。生产工厂强制新整步契约，
旧标量入口明确仅历史诊断；旧v7返回 `execution_contract_unvalidated`。
32次候选评估修正14处旧batch，自动去重6个整步形状（TEST_DATA），不声称覆盖整个工作负载。

数值守卫改为对照skip0、处理skip2；本地CPU仍24token及摘要一致。新增实际下一步形状
对应记录，形状变化/过期/非硬件不能用于预测误差。新探针、训练冻结/独立验证导出器、
派生配置工具、三项pilot计划与新配置重放CLI已完成离线验证。真机模型和性能结论未产生。

联合回归21模块：375 passed、87 subtests，0失败/错误/跳过；后续重放CLI定向23 passed。
两组有重叠不累加。日志/XML：`results/processed/offline_execution_shape_tests_final_20261003.*`
与 `offline_execution_shape_replay_cli_20261003.*`。未跑整个历史仓库。
早期测试曾误写不存在路径；另有未设置UTF8导致的子进程打印失败，均显形更正并重新验证。

**操作失误保留**：生成新版时覆盖了旧未跟踪的processed演练JSON，未备份原字节，未恢复。
旧 `dryplan_decision_chain.json` 已改为明确 INVALIDATED 占位，无指标并指向新版；写入器拒旧路径。
raw、历史模型和冻结配置未改，不用伪造重建掩盖失误。

新协议与下一步：`NEXT_BATCHES_PREREG.md` §13；交付入口
`handoff/optimization_20260929/OFFLINE_EXECUTION_SHAPE_20261003.txt`；
集中真机清单为同目录 `REMOTE_EXECUTION_PLAN_20261003.txt`。
