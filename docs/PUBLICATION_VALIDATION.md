# 发布快照离线验证

执行日期：2026-10-28。位置为独立发布副本，Windows / Python 3.13.7 / pytest 9.1.1。
没有连接 MTT 实例，没有进行 GPU 性能或新真机数值测量。

| 检查 | 当前实际结果 | 证据 |
| --- | --- | --- |
| 发布前初始五模块检查（含一个 CPU 数值用例） | 99 passed，3 subtests passed，退出 0 | [core.log](validation/20261028/core.log) |
| 修正后的 README/CI 基础范围，屏蔽 ML 运行时导入 | 98 passed，1 deselected，3 subtests passed，退出 0 | [core_without_ml.log](validation/20261028/core_without_ml.log) / [复现脚本](validation/20261028/check_core_without_ml.py) |
| 执行形状相关 21 模块 | 376 passed，87 subtests passed，0 fail/error/skip，退出 0 | [日志](validation/20261028/execution_shape_regression.log) / [JUnit](validation/20261028/execution_shape_regression.xml) |
| 整步标定探针 dry-plan | 退出 0；只计划，不采样 | [shape_dryplan.log](validation/20261028/shape_dryplan.log) |
| 三项 pilot 计划及实际矩阵 dry-plan 交叉校验 | 退出 0，恰好 baseline/cost_single/cost_dual 各一次 | [pilot_dryplan.log](validation/20261028/pilot_dryplan.log) |

两组测试有重叠，不能相加为独立测试总数。历史离线报告中的 375/87 保持原记录；上表是本次解释器和测试工具下的实际输出。此次没有重新执行整个历史仓库的全部测试，也没有把未执行的 MUSA 验证算作通过。

公开前复核发现，最初的 CI 基础命令包含 `TestExecutionShapeRealCpu`，但 `.[dev]` 不安装其 PyTorch/Transformers 依赖。README 与 CI 现明确用 `-k "not TestExecutionShapeRealCpu"` 排除这个数值用例；它保留在测试文件和完整数值回归中。上表的 1 deselected 不计为通过。修正后在主动拒绝 torch/transformers/torch_musa 导入的本地进程中验证通过；这不是 Linux 云端 CI 结果。

实际命令与退出码见 [verification.json](validation/20261028/verification.json)。其中 Python 与 JUnit 输出路径仅为可移植表达；日志来自本次真实执行。

发布辅助报告脚本首次在打印 pilot 输出时因 Windows 控制台编码退出 1；当时四个子命令退出码均为 0。修正辅助脚本输出编码后，仅重跑无硬件的 pilot 计划，退出 0。这个报告问题没有被隐藏成第一次整条脚本通过，也没有改项目算法或重跑 GPU 实验。

## 完整性与凭据检查

- `src/` 和 `benchmark/` 共 31 文件，与原研究工作区逐字节一致。
- 发布的 `results/raw/` 共 1,526 文件、497,618,218 字节，与原文件逐字节一致；没有修改实验标签或环境身份。
- 暂存区不含本地 `.workbuddy/`、真实凭据配置、私钥、旧 ZIP 和回拉中转压缩包。已检查常见 token/私钥模式以及当前连接密码的精确匹配，未发现匹配；这不是对所有未知秘密的无限保证。
- 原连接主机名已在六份文档或状态副本中替换为占位符。原研究文件保持不变。
- 非 raw 文本按项目 LF 规则归一化；发布 SHA-256 清单基于 Git 暂存字节，适用于正常克隆后的内容。

GitHub Actions 工作流只运行基础离线检查与 dry-plan；本地通过不等于云端 CI 已运行。云端状态须以实际发布后的 Actions 结果为准。
