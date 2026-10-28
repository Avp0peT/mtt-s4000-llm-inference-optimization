#!/usr/bin/env python3
"""执行优化线（S01–S03）真机批次的**本机一键编排**。

目标：一条命令完成 诊断 → 前置核验 → 推送 → 起跑 → 轮询 → 回拉 → 核验，
把需要人工盯的环节压到"只在失败时看一眼"。

子命令
------
* ``plan``      只打印**将执行的命令序列**（离线，不触网）—— 验证编排本身
* ``preflight`` 只做前置核验（可达性 / 环境 / 模型 revision / 空闲 / CPU 稳定性）
* ``run``       完整流程（默认）
* ``status``    读一次远端进度（心跳 + 已完成项）
* ``stop``      投放 STOP 文件（远端在当前 run 跑完后干净退出）
* ``pull``      增量回拉产物

为什么坚持复用既有脚本
----------------------
`launch_batch.sh`（锁 / setsid / 监视进程 / 激活）、`push_files.sh`（逐文件 sha 校验）、
`pull_incremental.py`（只取本地没有的 + 完整性核对）、`check_remote_busy.py`、
`check_cpu_stability.py`、`diag_remote_reachability.py` —— 这些各自都对应一次**踩过的坑**。
本脚本只做**编排**，不重写其中任何一条判据。

本地路径纪律（本项目已踩过两次）
--------------------------------
* 远端绝对路径交给本机原生程序前要关 MSYS 路径转换 ⇒ 这里**不经过 shell**，
  用 argv 列表调 `subprocess`，因此不存在路径被改写的问题；
* 本地目标一律用 **Windows 原生绝对路径**（`C:/...`），不用 `/c/...` 或 `/tmp/...`。

用法::

    python scripts/exec_opt_batch.py plan
    python scripts/exec_opt_batch.py preflight
    python scripts/exec_opt_batch.py run --tag EXECOPT1 --stage screen --budget-minutes 240
    python scripts/exec_opt_batch.py status --tag EXECOPT1
    python scripts/exec_opt_batch.py stop   --tag EXECOPT1
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
PY_VENV = Path.home() / ".workbuddy" / "binaries" / "python" / "envs" / "default" / "Scripts" / "python.exe"

#: 纯逻辑层（`benchmark/`）：等待退出判据的**唯一实现**，与本脚本的 I/O 分离。
sys.path.insert(0, str(ROOT / "benchmark"))
from exec_opt_plan import (  # noqa: E402
    BASELINE_DEFAULT,
    ORDER_MODES,
    CFG_DIR_DEFAULT,
    decide_wait_exit,
)
# ⚠️ 复用矩阵那一套**配置身份**实现（`config_data_digest`），不在这里再写一份：
#    两份"配置摘要"一旦漂移，"同一份配置"在两个脚本里会得出不同结论。
from run_benchmark import config_data_digest, load_config  # noqa: E402
# ⚠️ 复用 `provenance` 里**已有**的 MTT 产品名正则（不另写一条）。
from ascend_opt.provenance import MTT_MODEL_RE  # noqa: E402


def _find_git_bash() -> str:
    """定位 **Git Bash** 的绝对路径。

    ⚠️ **不要用裸 `"bash"`**：Windows 上 `subprocess.run(["bash", ...])` 会按 PATH
    解析到 `C:\\Program Files\\WSL\\bash.exe`（WSL 启动器）。本机安全策略把它列进
    程序黑名单 ⇒ 进程被拦，Python 侧只看到 `rc=1` + 一段 **UTF-16LE** 的错误文本
    （按 utf-8/gbk 解都是乱码 `\\ufffd...`）——症状是"推送步骤瞬间失败且无输出"，
    而**同一条命令在 Git Bash 里手工跑完全正常**（踩过，定位了很久）。
    ⇒ 显式给出 PortableGit 的绝对路径；顺带排除 WSL/System32 下的假 bash。
    """
    cands: list[Path] = []
    base = Path.home() / ".workbuddy" / "binaries" / "PortableGit"
    if base.is_dir():
        # 版本目录可能变（升级）⇒ 不硬编码版本号
        for ver in sorted(base.iterdir(), reverse=True):
            for sub in ("usr/bin/bash.exe", "bin/bash.exe"):
                cands.append(ver / sub)
    for p in cands:
        if p.is_file():
            return str(p)
    # 兜底：PATH 上第一个**不是** WSL 的 bash
    import shutil as _sh
    found = _sh.which("bash")
    if found and "WSL" not in found.upper():
        return found
    raise SystemExit(
        "找不到可用的 Git Bash（PortableGit）。裸 \"bash\" 会解析到被拦截的 "
        "WSL bash.exe ⇒ 不要回退到它。"
    )


BASH = _find_git_bash()
DIAG = ROOT / "scripts" / "diag_remote_reachability.py"
REMOTE_RUN = ROOT / "scripts" / "remote_run.py"
PUSH_SH = ROOT / "scripts" / "push_files.sh"
SHA_DIFF = ROOT / "scripts" / "batch_sha_diff.py"
PULL_INC = ROOT / "scripts" / "pull_incremental.py"
BUSY = ROOT / "scripts" / "check_remote_busy.py"
CPU_STAB = ROOT / "scripts" / "check_cpu_stability.py"
LAUNCH_SH = ROOT / "scripts" / "launch_batch.sh"
CLEAR_LOCK = ROOT / "scripts" / "clear_stale_batch_lock.py"

DEFAULT_REMOTE_ROOT = "/root/ascend_probe_d1"
DEFAULT_ACTIVATE = "/root/musa-envs/qwen310/activate_musa.sh"
#: 批次锁根目录（`launch_batch.sh` 的 `BATCH_LOCKDIR` 默认值）。
LOCK_ROOT = "/root/batch_locks"
#: 期望的权重 commit（`handoff/.../01_PROJECT_PLAN.md` §2；配置里也已钉死）。
EXPECTED_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
#: 需要推送到远端的**相对路径**（代码 + 本次实验的配置 + 基线）。
#: ⚠️ 配置是**实验条件的一部分**：只推代码会得到"代码新、配置旧"，
#: 而产物里看不出来。这里**不**推整个 `configs/`（142 个 yaml，
#: 逐文件推送在低 RTT 抖动的链路上要 142 次往返），只推用得到的那几份。
PUSH_PATHS = (
    "src",
    "benchmark",
    "scripts",
    "configs/mthreads/exec_opt",
    "configs/mthreads/s4000_fp16.yaml",
)
FAIL_MARKERS = ("连接失败", "SSHException", "EOFError", "No existing session",
                "banner", "Authentication failed", "paramiko")


# --------------------------------------------------------------------------- #
# 极薄的本机命令封装
# --------------------------------------------------------------------------- #

#: 推送时递归展开的扩展名。**必须是唯一真相源**（曾硬编码 4 处，
#: 改一处漏三处的风险与"加了机制但配置没推上去"等价）。
#: `yml` 与 `yaml` 都列：当前仓库只有 `.yaml`，但漏掉 `yml` 时是**静默**
#: 少推文件（远端用的是旧配置，产物看不出来）。
PUSH_INCLUDE = "py,sh,yaml,yml"

#: `remote_run.py stamp` 成功时打印的标记（见 `do_stamp` 的最后一个 `print`）。
#: 用它做**正证据**：退出码 0 只说明没报错，不说明校验真的跑过。
STAMP_OK_MARKER = "投放并校验通过"


def parse_push_todo(out: str) -> list[str]:
    """解析 `batch_sha_diff.py --emit-paths` 的输出 ⇒ 待推送路径清单。

    ⚠️ **只接受"整行就是一个路径"**。历史上这里解析的是**人类报告**，
    而报告里 `sha 不同` 的行形如 `  scripts/x.py  local=ab.. remote=cd..`
    ⇒ 整行被当路径 ⇒ `push_files.sh` 判「跳过（不存在）」⇒ 静默不推。
    常见于"报告里既有缩进路径、又有缩进命令续行"的格式。
    """
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


def find_bogus_todo(todo: Sequence[str], root: Path = ROOT) -> list[str]:
    """清单里**不是本机真实文件**的条目（含空格后缀、命令续行等伪路径）。"""
    return [t for t in todo if not (root / t).is_file()]


_PUSH_SUMMARY_RE = re.compile(
    r"push_files:\s*ok=(\d+)\s+skip=(\d+)\s+fail=(\d+)\s*\(共\s*(\d+)\)")


def parse_push_summary(out: str) -> tuple[int, int, int, int] | None:
    """读 `push_files.sh` 的收尾统计 ⇒ `(ok, skip, fail, N)`；读不到返回 `None`。"""
    m = _PUSH_SUMMARY_RE.search(out)
    return tuple(int(g) for g in m.groups()) if m else None


def push_summary_complaint(summary: tuple[int, int, int, int] | None,
                           n_todo: int) -> str | None:
    """推送是否**真的覆盖了清单**；有问题时返回说明，否则 `None`。

    为什么要单独判：`push_files.sh` 对不存在的路径只印一行
    「跳过（不存在）」到 stderr 然后 `continue`，**退出码仍是 0**。
    于是"清单 5 个、实际展开 0 个"与"清单 5 个、全部成功"退出码相同。
    """
    if summary is None:
        return ("读不到 `push_files.sh` 的收尾统计行 ⇒ 无法证明推送完成"
                "（不把「没报错」当成功）。")
    ok, skip, fail, n = summary
    if fail != 0:
        return f"有 {fail} 个文件未通过 sha 校验。"
    if n != n_todo:
        return (f"推送**未覆盖全部差异**：清单 {n_todo} 个，实际只展开 {n} 个"
                f"（ok={ok} skip={skip} fail={fail}）⇒ 有路径被「跳过（不存在）」，"
                "推送并未真正发生。")
    if ok + skip != n:
        return f"统计不自洽：ok({ok}) + skip({skip}) != N({n})。"
    return None


def stamp_complaint(rc: int, out: str) -> str | None:
    """投放溯源戳是否**真的通过**；有问题时返回说明，否则 `None`。

    两个判据都要看：
      * **退出码**是权威（4 = 生成不了可信戳；6 = 远端重算摘要不符/拿不到结果）；
      * **正标记**防"退出码 0 但什么也没验"——成功路径必然打印
        `投放并校验通过：…`（见 `remote_run.py::do_stamp`）。
    实测（EXECOPT4）：退出码曾被直接丢弃 ⇒ 12 份产物全落
    `data_label=MEASURED` + `git_commit=NOT_MEASURED` + `git_source=stamp_stale`，
    即"有测量标签、没有源码身份"，而编排一路正常起跑。
    """
    if rc != 0:
        return ("溯源戳**未通过**（4=生成不了可信戳；6=远端重算摘要不符/拿不到结果）"
                "⇒ 这批产物的 `git_commit` 会是 `NOT_MEASURED`。")
    if STAMP_OK_MARKER not in out:
        return (f"退出码为 0，但输出里**没有**成功标记 {STAMP_OK_MARKER!r} ⇒ "
                "无法证明戳真的被校验过（「没报错」不等于「校验通过」）。")
    return None


class StepFailed(RuntimeError):
    def __init__(self, step: str, rc: int, out: str) -> None:
        super().__init__(f"{step} 失败（rc={rc}）：{out.strip()[:600]}")
        self.step, self.rc, self.out = step, rc, out


def sh(argv: Sequence[str], *, step: str, timeout: float | None = None,
       check: bool = True) -> tuple[int, str]:
    """跑一条本机命令。**不经过 shell**（argv 列表）⇒ 无路径改写问题。

    ⚠️ 编码不能写死 utf-8：`push_files.sh`（Git Bash 跑的 bash 脚本）在中文 Windows
    上输出 **GBK/CP936**，按 utf-8 解会得到乱码 —— 而失败清单恰好在最后一行，
    乱码后**看不出是哪个文件失败**（踩过：批次在"推送源码/配置"步骤失败，
    日志里推送段只有 `\\ufffd...` 乱码，定位不了）。
    ⇒ 先拿 bytes，再按 utf-8 → gbk 顺序试解，都失败才 replace。
    """
    try:
        proc = subprocess.run([str(a) for a in argv], capture_output=True,
                              timeout=timeout)  # 不收 text：自己解码
    except subprocess.TimeoutExpired as exc:
        # ⚠️ 超时**不能**直接冒泡成"未知失败"：本机推 128 个文件在抖动链路上确实
        #    可能超过给定 timeout（踩过）。这里转成带上下文的 StepFailed，
        #    并说清楚"是超时，不是 sha 校验失败"——两者的修法完全不同。
        partial = _decode(exc.stdout or b"") if exc.stdout else ""
        raise StepFailed(
            step, 124,
            f"本机命令超时（>{timeout}s）。**这是超时，与 sha 校验无关**。"
            f"已捕获的部分输出：\n{partial[-1500:]}",
        ) from exc
    out = _decode(proc.stdout) if proc.stdout else ""
    if proc.stderr:
        out += "\n" + _decode(proc.stderr)
    if check and proc.returncode != 0:
        raise StepFailed(step, proc.returncode, out)
    return proc.returncode, out


def _decode(raw: bytes) -> str:
    """bytes → str：utf-8 优先，GBK 兜底，**UTF-16 也要能读**（中文 Windows 的 bash 输出）。

    ⚠️ UTF-16 这一支不是凑数：被安全策略拦下的 WSL `bash.exe` 回的是 **UTF-16LE**
    文本，按 utf-8/gbk 解全是 `\\ufffd` 乱码 ⇒ 失败原因完全不可读（踩过）。
    """
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):     # BOM → 确定是 UTF-16
        return raw.decode("utf-16", errors="replace")
    # ⚠️ UTF-16 **无 BOM** 的判定必须在 utf-8/gbk 之前：ASCII 文本按 UTF-16LE 编码后
    #    每个字符带一个 `\x00`，而这串字节在 utf-8 里**也是合法的** ⇒ 先试 utf-8 会
    #    "成功"但读出一堆 `b\x00l\x00o\x00...`（踩过）。用空字节密度来判别。
    if raw.count(b"\x00") > max(2, len(raw) // 4):
        return raw.decode("utf-16-le", errors="replace")
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _remote(argv: Sequence[str], *, step: str, timeout: float | None = None,
            check: bool = True) -> tuple[int, str]:
    return sh([PY_VENV, REMOTE_RUN, *argv], step=step, timeout=timeout, check=check)


def remote_exec(cmd: str, *, step: str, timeout: float | None = 120.0,
                check: bool = True) -> tuple[int, str]:
    return _remote(["exec", cmd], step=step, timeout=timeout, check=check)


def remote_exec_checked_output(cmd: str, *, step: str,
                               timeout: float | None = 120.0) -> str:
    """执行远端命令并返回输出；**输出里出现连接类失败特征即抛**。

    为什么不能只看退出码：`launch_batch.sh` 的注释里记过同一类事故 ——
    编排侧的失败重试会把输出回读问题读成"远端状态变了"。
    这里反过来：把"回读失败"与"远端真的报错"分开，**只把前者当失败**。
    """
    rc, out = remote_exec(cmd, step=step, timeout=timeout, check=False)
    if any(m in out for m in FAIL_MARKERS):
        raise StepFailed(step, rc, f"远端调用本身失败（不是远端命令的结论）：{out}")
    return out


# --------------------------------------------------------------------------- #
# 批次上下文
# --------------------------------------------------------------------------- #

class Ctx:
    def __init__(self, a: argparse.Namespace) -> None:
        self.tag: str = a.tag
        self.remote_root: str = a.remote_root.rstrip("/")
        self.activate: str = a.activate
        self.stage: str = a.stage
        #: 显式臂集合；给定时**覆盖** `--stage`（透传给 `run_exec_opt_matrix.py`）。
        #: ⚠️ 为什么必须有它：`--stage screen` 是**写死的 S01–S03 三臂**，
        #: 而后续机制（如 S04 `s04_pkv` / `s01_both_s04_pkv`）不在预设里
        #: ⇒ 没有这个透传就只能跑"上一批的臂"，产物看起来成功、实际没测新机制。
        self.arms: list[str] | None = list(a.arms) if getattr(a, "arms", None) else None
        #: ★ 对照臂配置路径，**显式透传**给矩阵（矩阵自己也有同名选项）。
        #: ⚠️ 为什么必须透传：矩阵的 `--baseline` **有默认值**，所以"不传"是合法的，
        #: 于是外层编排以前从不传 ⇒ 对照臂静默落到兜底配置，而臂清单来自另一个目录。
        #: 两臂的配置来源对不上，产物里看不出来，也不报错（2026-10-03 修）。
        self.baseline: str = str(getattr(a, "baseline", None) or BASELINE_DEFAULT)
        #: 臂**配置目录**，显式透传给矩阵（矩阵自己也有同名选项）。
        #: 与 `BASELINE_DEFAULT` 同理：不传时矩阵会用它的默认值，而外层的臂清单可能
        #: 来自另一个目录 ⇒ 两者的配置来源对不上，产物看不出来也不报错。
        self.cfg_dir: str = str(getattr(a, "cfg_dir", None) or CFG_DIR_DEFAULT)
        #: 轮内格序（透传给矩阵）。默认 `block`；本批用 `abba`。
        self.order: str = str(getattr(a, "order", None) or "block")
        #: ★ **固定轮数**（None = 保持"填预算"模式）。
        #: 为什么需要：`--min-repeats/--max-repeats` 描述的是**填预算**模式，
        #: 它给不出"恰好 2 臂 × 5 轮 = 10 项"这种**确定项数**的批次；
        #: 而"项数确定"正是"少跑必须报失败"的前提（见矩阵的完成度判据）。
        self.repeats: int | None = (
            int(a.repeats) if getattr(a, "repeats", None) else None
        )
        self.budget_minutes: float = a.budget_minutes
        self.min_repeats: int = a.min_repeats
        self.max_repeats: int = a.max_repeats
        self.no_wait: bool = a.no_wait
        self.poll_seconds: float = a.poll_seconds
        self.wait_timeout_minutes: float = a.wait_timeout_minutes
        #: 远端文件一律放在 remote_root 下（**不写 /tmp**：实例重启会回滚系统盘，
        #: 而 /root 与 remote_root 同盘 —— 至少与证据同生命周期；产物本身边跑边回拉）。
        self.progress = f"{self.remote_root}/exec_opt_progress_{self.tag}.json"
        self.summary = f"{self.remote_root}/exec_opt_summary_{self.tag}.json"
        self.stop_file = f"{self.remote_root}/exec_opt_STOP_{self.tag}"
        self.logfile = f"{self.remote_root}/logs/{self.tag}.log"

    @property
    def runner_args(self) -> str:
        """矩阵运行器的**参数串**（不含 `python3` 与脚本名）。"""
        # ⚠️ `--arms` 必须**显式透传**（不能用 `--stage` 代替）：阶段预设是写死的
        #    三臂，给不出 S04 这类新机制；且 `--arms` 与 `--stage` 在矩阵脚本里
        #    是"后者覆盖前者"的关系，这里统一走 `--arms` 更不易错。
        arms = f"--arms {' '.join(self.arms)} " if self.arms else ""
        # ★ 对照臂**总是显式传**：让"这一批拿什么当对照"出现在命令行里，
        #   而不是藏在矩阵的默认值里（默认值不会出现在任何日志里）。
        baseline = f"--baseline {shlex.quote(self.baseline)} "
        cfg_dir = f"--cfg-dir {shlex.quote(self.cfg_dir)} "
        # ★ 固定轮数：给了 `--repeats` 就走固定模式（矩阵据此关闭 fill-budget）。
        #   此时 `--min-repeats/--max-repeats` 不再决定轮数，但**保留**
        #   `--budget-minutes` 作为跑飞的兜底；一旦它真的提前收尾，
        #   矩阵的完成度判据会把这一批判成**失败**（不会静默少跑）。
        repeats = f"--repeats {self.repeats} " if self.repeats else ""
        # ★ 轮内顺序**显式透传**：不传的话矩阵用它的默认（block），
        #   于是「外层以为跑的是 ABBA、实际跑的是 ABAB」，而两边都不报错。
        order_arg = f"--order {shlex.quote(self.order)} "
        return (
            f"--tag {self.tag} --stage {self.stage} {arms}{baseline}{cfg_dir}{order_arg}{repeats}"
            f"--min-repeats {self.min_repeats} --max-repeats {self.max_repeats} "
            f"--budget-minutes {self.budget_minutes} "
            f"--progress-file {self.progress} "
            f"--summary-file {self.summary} "
            f"--stop-file {self.stop_file}"
        )

    @property
    def venv_python(self) -> str:
        """激活脚本所在 venv 的 **python 绝对路径**。

        ⚠️ **不要用裸 `python3`**：`launch_batch.sh` 的激活是在
        `bash -c 'source <activate>; exec "$@"'` 里做的，而**再往里**那层
        `bash -c`（承载真正命令的那层）会重新加载 profile（`/etc/profile` /
        miniconda 的 `BASH_ENV`），把 venv 的 `bin` 从 PATH 头部挤掉 ⇒
        `python3` 解析到 `/root/miniconda3/bin/python3` ⇒
        `ModuleNotFoundError: No module named 'transformers'`
        （实测：激活后直接跑 PATH 头 = venv；经 `bash -c` 跑 PATH 头 = miniconda）。
        ⇒ 把解释器写成**绝对路径**，PATH 怎么变都无所谓。
        """
        # activate 形如 /root/musa-envs/qwen310/activate_musa.sh ⇒ venv 根是其父目录
        root = self.activate.rsplit("/", 1)[0] if "/" in self.activate else ""
        return f"{root}/bin/python3" if root else "python3"

    @property
    def runner_cmd(self) -> str:
        return f"{self.venv_python} scripts/run_exec_opt_matrix.py {self.runner_args}"

    @property
    def runner_inner(self) -> str:
        """要交给 `launch_batch.sh` 的**单条**命令（激活之后 exec 它）。

        ⚠️ 必须是**一条** `bash -c`：`launch_batch.sh` 用 `setsid nohup "$@"` 直接
        exec，把 `cd X && python3 ...` 当成多个 argv 传过去会让它去 exec 一个名叫
        `cd` 的程序。同理，`cd` 必须先于脚本路径解析（脚本路径是相对的）。
        ⚠️ 里面**不得出现单引号**：外层用 `'...'` 包裹它。
        ⚠️ 解释器用**绝对路径**（见 `venv_python`）：内层 `bash -c` 会重置 PATH。
        """
        return (f"cd {self.remote_root} && "
                f"exec {self.venv_python} scripts/run_exec_opt_matrix.py "
                f"{self.runner_args}")

    @property
    def launch_cmd(self) -> list[str]:
        """**起跑命令的 argv**（⚠️ 只发一次，绝不重试）。"""
        return [
            "bash", "scripts/launch_batch.sh", "--activate", self.activate,
            self.tag, self.logfile, "bash", "-c", self.runner_inner,
        ]


def _launch_shell_string(ctx: Ctx) -> str:
    """把起跑整条串成**一条**远端 shell 命令。

    ⚠️ 必须整条一次发出：`launch_batch.sh` 内部做"取锁 → setsid 起跑 → 起监视
    → 存活确认"，分两次调用会让锁与监视进程的归属错位。
    ⚠️ 最后一个 argv 是 `bash -c` 的**命令体**（含空格）⇒ 必须引号包裹，
    否则远端 shell 会把它拆成多个词，`launch_batch.sh` 拿到的就不是"一条命令"。
    """
    parts = list(ctx.launch_cmd)
    inner = parts.pop()
    assert "'" not in inner, "命令体里不得出现单引号（外层用单引号包裹）"
    return f"cd {ctx.remote_root} && " + " ".join(parts) + f" '{inner}'"


# --------------------------------------------------------------------------- #
# 步骤
# --------------------------------------------------------------------------- #

def step_reachability(*, dry: bool) -> None:
    if dry:
        print(f"[plan] {PY_VENV} {DIAG}")
        return
    rc, out = sh([PY_VENV, DIAG], step="可达性诊断", timeout=120, check=False)
    print(out.strip())
    if rc != 0:
        raise StepFailed("可达性诊断", rc,
                         "远端不可达 ⇒ **停止**（这不代表方案无效，只代表现在跑不了）")


def step_env(*, ctx: Ctx, dry: bool) -> str:
    """核对远端环境与**模型 revision 快照是否真的在本地缓存里**。"""
    # ⚠️ 这里**不能用 heredoc**：整条命令要经 `bash -c '...'` 传递，而 Python 源码里的
    #    字面 `\n` 不会被解释 ⇒ heredoc 会被压成一行 ⇒ bash 语法错（曾把「语法错」
    #    误判成「torch.musa 不可用」）。
    # ⚠️ 也**不要手写转义引号**：`bash -c '...'`（单引号）里再套 `\"...\"` 时，
    #    远端 shell 会二次解析，Python 里的 `'PY'` 会脱离字符串 ⇒ `NameError: name 'PY'
    #    is not defined`（踩过）。⇒ 一律用 `shlex.quote` 生成载荷，不靠人脑推引号层级。
    inner = (
        "import os, sys; "
        "import torch, torch_musa, transformers; "
        "print('PY', sys.version.split()[0]); "
        "print('TORCH', torch.__version__, 'MUSA', torch_musa.__version__); "
        "print('TF', transformers.__version__); "
        "avail = hasattr(torch, 'musa') and torch.musa.is_available(); "
        "print('TORCH_MUSA_AVAIL', avail); "
        "print('DEVS', torch.musa.device_count() if avail else -1); "
        "print('REV_ENV', os.environ.get('MODEL_REVISION')); "
        "print('HF_HOME', os.environ.get('HF_HOME'))"
    )
    # 整条远端 shell 脚本：激活（用配置的脚本路径，本身不含空格假设）后跑 python。
    # ⚠️ 解释器用 `ctx.venv_python`（**绝对路径**）而不是裸 `python3`：
    #    起跑时真正的命令会再包一层 `bash -c`，那一层会重置 PATH 把 venv 挤掉
    #    （实测 PATH 头从 `<venv>/bin` 变成 `/root/miniconda3/bin`）⇒ 裸 `python3`
    #    会解析到 miniconda 的解释器 ⇒ `ModuleNotFoundError: No module named
    #    'transformers'`（真机上栽过一次整轮）。核验用的解释器必须与起跑用的一致，
    #    否则"核验通过、起跑炸"——核验本身就失去了意义。
    script = (
        f"cd {shlex.quote(ctx.remote_root)} && "
        f"source {shlex.quote(ctx.activate)} >/dev/null 2>&1; "
        f"{shlex.quote(ctx.venv_python)} -c {shlex.quote(inner)}"
    )
    cmd = script
    if dry:
        print("[plan] 远端环境核对（激活 venv 后打印 torch/torch_musa/transformers/设备/REV）")
        return ""
    out = remote_exec_checked_output(cmd, step="环境核对", timeout=180)
    print(out.strip())
    if "TORCH_MUSA_AVAIL True" not in out:
        raise StepFailed("环境核对", 1, "torch.musa 不可用 ⇒ 真机推理跑不了；**停止**")

    # ★ 起跑路径守卫：验证**在 `bash -c` 里**用同一个绝对路径解释器也能 import。
    #   这一条专治"裸 python3 被 PATH 抢走"——上面那条用的是直接调用，不会暴露该问题。
    #   （判据：同一解释器在两种调用形态下都要能 import transformers。）
    guard = (
        f"cd {shlex.quote(ctx.remote_root)} && "
        f"bash -c {shlex.quote(ctx.venv_python + ' -c ' + shlex.quote(inner))}"
    )
    gout = remote_exec_checked_output(guard, step="起跑路径守卫", timeout=240)
    if "TORCH_MUSA_AVAIL True" not in gout:
        print(gout.strip())
        raise StepFailed(
            "起跑路径守卫", 1,
            f"经 `bash -c` 调用时 `{ctx.venv_python}` 无法 import transformers"
            " ⇒ 起跑会以 ModuleNotFoundError 失败。**停止**（不要浪费一整轮）。"
        )
    print(f"[ok] 起跑路径守卫：经 bash -c 用 {ctx.venv_python} 仍可 import（与起跑同形态）")
    return out


def _parse_stamp_time(text: str) -> datetime | None:
    """解析指纹里的 `collected_at`（形如 `2026-09-28T00:16:44+0800`）。

    容忍几种写法：带/不带冒号的时区、带/不带时区、`Z` 结尾。
    解析不出来就返回 `None` ——**不要**回退成"现在"或"零时刻"：
    那会让"时间未知"伪装成"时间没问题"。
    """
    s = str(text or "").strip()
    if not s:
        return None
    cands = [s, s.replace("Z", "+00:00")]
    m = re.match(r"^(.*[+-]\d{2})(\d{2})$", s)      # +0800 -> +08:00
    if m:
        cands.append(f"{m.group(1)}:{m.group(2)}")
    for c in cands:
        try:
            dt = datetime.fromisoformat(c)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def read_env_fingerprint(doc: dict[str, Any] | None) -> dict[str, Any]:
    """按 `inspect_env.sh` 的**真实结构**把指纹读成规范化的几个量。

    ★★ 为什么要单独一层（2026-09-03 实测的读取缺陷）
    ------------------------------------------------
    旧代码读的是 `d.get('device_count')` 与 `sw.get('mthreads_driver_version')`
    —— **这两个键在真实产物里根本不存在** ⇒ 恒为 `None` ⇒
    依赖它们的判据（"设备数为正"、"驱动版本已确认"）**一次都没有执行过**。
    而它们的失败态是**静默**的：产物里那两个字段看起来只是"没查到"。

    真实路径（用**远端实际采集的 `env.json`** 逐字核对过）：

    | 量 | 真实路径 | 旧代码读的 |
    | --- | --- | --- |
    | 设备可用 | `available`（顶层）/ `probes.mthreads.ok` | `available` ✅ |
    | 设备数 | **`device.count`** / `probes.mthreads.device_count` | `device_count` ❌ 不存在 |
    | 设备型号 | **`device.model`** / `probes.mthreads.model` | 未读 |
    | 驱动版本 | **`discovery.values.MTT_DRIVER_VERSION`** | `software.mthreads_driver_version` ❌ 不存在 |
    | MUSA 版本 | `discovery.values.MUSA_VERSION` | 未读 |
    | 平台 | 顶层 `platform` | 未读 |
    | 模型 revision | `discovery.values.MODEL_REVISION` | 未读 |
    | hostname / collected_at | 顶层同名字段 | ✅ |

    ⚠️ **只读不判**：本函数不做任何通过/失败判断，只把"读到了什么、缺了什么"
    如实摆出来 ⇒ 判据集中在 `env_fingerprint_acceptance`，便于逐条验证正反例。
    """
    d = doc or {}
    dev = d.get("device") or {}
    disc = d.get("discovery") or {}
    vals = disc.get("values") or {}
    probes = (d.get("probes") or {}).get("mthreads") or {}
    soft = d.get("software") or {}

    def _first(*cands: Any) -> Any:
        """按优先级取第一个**非空**值；全空返回 `None`（不填 0、不填 ""）。"""
        for c in cands:
            if c is not None and c != "" and c != []:
                return c
        return None

    return {
        # —— 顶层 ——
        "hostname": _first(d.get("hostname"), disc.get("hostname")),
        "collected_at": d.get("collected_at"),
        "platform": d.get("platform"),
        "available": d.get("available"),
        "available_reason": d.get("available_reason"),
        # —— 设备 ——
        "device_count": _first(dev.get("count"), probes.get("device_count")),
        "device_model": _first(dev.get("model"), probes.get("model")),
        "device_ok": _first(probes.get("ok"), dev.get("mthreads_gmi_available")),
        "looks_like_mthreads_host": _first(disc.get("looks_like_mthreads_host"),
                                          probes.get("result")),
        # —— 版本 ——
        "driver": vals.get("MTT_DRIVER_VERSION"),
        "musa_version": vals.get("MUSA_VERSION"),
        "torch": soft.get("torch_version"),
        "torch_musa": soft.get("torch_musa_version"),
        "transformers": soft.get("transformers_version"),
        "python": soft.get("python"),
        "model_revision": vals.get("MODEL_REVISION"),
        # —— 缺口（如实带出，不吞掉）——
        "discovery_missing": list(disc.get("missing") or []),
        "mtt_missing": list(disc.get("mtt_missing") or []),
    }


#: 接受一条指纹时**必须**非空的字段（缺任何一个 ⇒ 停）。
#: ⚠️ 不含 `driver` 的**取值**要求（老指纹可能没有），但含"它必须**被读到**"这一层
#: —— 见 `env_fingerprint_acceptance` 的 §③：读不到要**显式**说明，
#: 而不是让一个 `None` 悄悄穿过判据。
REQUIRED_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "hostname", "collected_at", "platform",
    "device_count", "device_model", "device_ok",
)


def env_fingerprint_acceptance(*, fp: dict[str, Any], live_hostname: str,
                               declared_platform: str | None,
                               declared_model: str | None = None,
                               expect_driver: str | None = None) -> dict[str, Any]:
    """**接受一条环境指纹**所需的全部核对（首次采集**同样**执行）。

    ★ 为什么"首次采集"也必须全查（2026-10-03 用户明确要求）：
    旧逻辑把若干检查挂在"有上一份指纹可比"的条件下（例如新鲜度）。
    那条豁免的本意是"没得比就没法比"，但它顺带**放行了所有缺失字段**
    —— 于是"第一次采集"成了一个**没有守卫的入口**：
    缺 `device_count`、缺 `driv器` 都照样过，而产物看起来完全正常。

    ⇒ 本函数**不区分首次/非首次**：逐条查，缺就停。
    只有"**必须比上一份**才能回答"的那一条（新鲜度）才允许在首次时**记 note**，
    且它由 `fingerprint_identity_verdict` 单独处理，不在这里放宽。

    六项核对（全部无条件执行）：
      ① **设备可用**：`available` 必须为真（`False` ⇒ 停；`None` ⇒ 停，因为"读不到"不等于"可用"）；
      ② **设备数为正**：`device_count` 必须是 ≥1 的整数；
      ③ **平台/型号匹配**：指纹的 `platform` 与 `device_model` 必须与配置声明一致；
      ④ **hostname 与实时实例一致**：指纹里的 hostname 必须等于**当前**机器标识
         （这条是 A7 的靶心：指纹来自别的容器时，它描述的机器不是实跑机器）；
      ⑤ **采集时间可解析**：`collected_at` 必须解析得出（解析不出 ⇒ 停，不许当成"刚采的"）；
      ⑥ **必填字段非空**：`REQUIRED_FINGERPRINT_FIELDS` 缺任何一个 ⇒ 停。
    """
    complaints: list[str] = []
    notes: list[str] = []

    # ---- ① 设备可用 ----
    avail = fp.get("available")
    if avail is None:
        complaints.append("指纹里读不到 `available` ⇒ **无法确认设备可用**"
                          "（读不到 ≠ 可用）；停。")
    elif not bool(avail):
        complaints.append(f"指纹判定**设备不可用**（`available={avail!r}`，"
                          f"理由：{fp.get('available_reason')}）⇒ 停。")

    # ---- ② 设备数为正 ----
    n = fp.get("device_count")
    if n is None:
        complaints.append("指纹里读不到**设备数**（`device.count`）⇒ 停。")
    else:
        try:
            ni = int(n)
        except (TypeError, ValueError):
            complaints.append(f"设备数 `{n!r}` 不是整数 ⇒ 停。")
        else:
            if ni <= 0:
                complaints.append(f"设备数 = **{ni}**（必须为正）⇒ 停。")

    # ---- ③ 平台 / 型号匹配 ----
    if declared_platform:
        got_platform = str(fp.get("platform") or "")
        if not got_platform:
            complaints.append("指纹里读不到 `platform` ⇒ 无法核对平台声明；停。")
        elif got_platform != declared_platform:
            complaints.append(
                f"**平台不符**：配置声明 `{declared_platform}`，指纹是 `{got_platform}`；停。")
        # 「像不像 MTT 主机」是**启发式**；它说不是时，必须拦（与声明矛盾）
        looks = fp.get("looks_like_mthreads_host")
        if declared_platform == "mthreads" and looks is False:
            complaints.append("指纹 `looks_like_mthreads_host=false` 与声明 `mthreads` "
                              "矛盾 ⇒ 停。")
        elif looks is None:
            notes.append("⚠️ 指纹没有 `looks_like_mthreads_host` ⇒ 该启发式未参与判断")
    if declared_model:
        got_model = str(fp.get("device_model") or "")
        if not got_model:
            complaints.append("指纹里读不到**设备型号** ⇒ 无法核对型号；停。")
        elif declared_model.lower() not in got_model.lower():
            complaints.append(
                f"**型号不符**：声明含 `{declared_model}`，指纹是 `{got_model}`；停。")
    elif declared_platform == "mthreads":
        # 配置里**没有**声明硬件型号（`device_model:` 段是模拟器成本参数，不是硬件名）。
        # ⇒ 退一步做**形状核对**：型号必须看起来是 MTT 产品名。
        # ⚠️ 用 `provenance` 里那份**已有**的正则，不另写一条（两份解析器迟早不一致）。
        got_model = str(fp.get("device_model") or "")
        if not got_model:
            complaints.append("指纹里读不到**设备型号** ⇒ 无法核对型号；停。")
        elif not MTT_MODEL_RE.search(got_model):
            complaints.append(
                f"设备型号 `{got_model}` **不匹配 MTT 产品名形状**"
                f"（`{MTT_MODEL_RE.pattern}`）⇒ 停。")

    # ---- ④ hostname 与实时实例一致 ----
    fp_host = str(fp.get("hostname") or "").strip()
    live = str(live_hostname or "").strip()
    if not fp_host:
        complaints.append("指纹里读不到 `hostname` ⇒ 无法确认它描述的就是实跑机器；停。")
    elif not live:
        complaints.append("**实时机器标识取不到** ⇒ 无法与指纹对账；停。")
    else:
        # 实时标识形如 `<hostname>/<user>`；只比 hostname 部分
        live_host = live.split("/", 1)[0]
        if fp_host != live_host:
            complaints.append(
                f"**指纹与实时实例不一致**：指纹 `{fp_host}` vs 实时 `{live_host}` "
                "⇒ 它描述的不是实跑机器（A7 的形态）；停。")

    # ---- ⑤ 采集时间可解析 ----
    ct_raw = fp.get("collected_at")
    ct = _parse_stamp_time(str(ct_raw or ""))
    if ct is None:
        complaints.append(f"`collected_at={ct_raw!r}` **解析不出时间** ⇒ 停。")

    # ---- ⑥ 必填字段 ----
    missing = [k for k in REQUIRED_FINGERPRINT_FIELDS if fp.get(k) in (None, "", [])]
    if missing:
        complaints.append(f"必填字段缺失：{missing} ⇒ 停（缺 ≠ 空，不许当默认值用）。")

    # ---- 附加：驱动版本（读到就报，读不到要显式说明）----
    drv = fp.get("driver")
    if drv is None:
        notes.append("⚠️ 指纹里没有 `discovery.values.MTT_DRIVER_VERSION` ⇒ "
                     "驱动版本**未确认**（不得据此声称驱动可比）")
    elif expect_driver and str(drv) != str(expect_driver):
        complaints.append(f"**驱动版本不符**：期望 `{expect_driver}`，指纹 `{drv}` ⇒ 停。")
    else:
        notes.append(f"驱动版本 = {drv}（来自 `discovery.values.MTT_DRIVER_VERSION`）")

    return {
        "ok": not complaints,
        "complaints": complaints,
        "notes": notes,
        "normalized": fp,
        "live_hostname": live,
        "collected_at_epoch": (ct.timestamp() if ct else None),
    }


def fingerprint_identity_verdict(*, prev: dict[str, Any] | None,
                                 cur: dict[str, Any] | None,
                                 live_machine_id: str,
                                 declared_platform: str | None = None) -> dict[str, Any]:
    """**纯函数**：环境指纹是否真的描述"本次要跑的那台机器"。

    ★★ 为什么需要它（2026-10-03 实测事故）
    --------------------------------------
    EXECOPT5（2026-10-03）的 10 份产物里，`hardware_env` 写着

        collected_at = 2026-09-28T00:16:44+0800
        hostname     = autodl-container-cf99469eb6-b877706e

    而它实际跑在 `autodl-container-05264bbf0d-3b9c3488`。
    成因：`load_hardware_env` 只按**路径**找 `results/raw/_env/env.json`，
    **没有任何新鲜度或容器身份校验** ⇒ 一份 5 天前、**另一个容器**采的指纹
    被原样写进了全部产物。

    为什么这是硬问题而不是"差一点"：那份指纹里承载着 `mthreads_driver_version` /
    OS / CPU / 甚至 `device_model` 这类**跨批次可比性的前提**。指纹来自 B 容器、
    数字来自 A 容器时，"两批同环境"这句话**没有证据**，而产物看上去完全正常。

    判定（返回 `complaints` 非空 ⇒ 调用方**必须**停止本批）：
      ① 指纹取不到 ⇒ 停（"测不到"不得读成"没问题"）；
      ② 指纹声明 `available=False`，或设备数 < 1 ⇒ 停（这种批跑了没有意义）；
      ③ 指纹的 hostname 与**实时** `detect_machine_id()` 的 host 部分不一致
         ⇒ **停**。这一条最关键：它直接判"这份指纹不是本机的"。

    `container_changed`（相对上一份指纹换没换容器）**只记不拦**：
    换容器本身是合法事件（AutoDL 重启/迁移），但它**改变了跨批可比性的前提**，
    所以必须显式记下来，而不是让读者以为两批在同一台机器上。
    """
    complaints: list[str] = []
    notes: list[str] = []

    if cur is None:
        complaints.append(
            "环境指纹**取不到**（刷新后仍没有 results/raw/_env/env.json）"
            "⇒ 本批产物将无法自证环境；停。")
        return {"ok": False, "complaints": complaints, "notes": notes,
                "container_changed": None}

    host = str(cur.get("hostname") or "").strip()
    collected = str(cur.get("collected_at") or "").strip() or "(未记时间)"

    avail = cur.get("available")
    if avail is False:
        complaints.append(
            f"指纹声明 available=False（{collected}，host={host or '?'}）"
            "⇒ 本机没有可用加速设备；停。")
    devs = cur.get("device_count")
    if isinstance(devs, int) and devs < 1:
        complaints.append(f"指纹里设备数 device_count={devs} ⇒ 没有可跑的设备；停。")

    # ★ 身份对账：指纹描述的机器 == 实时这台机器？
    #   `detect_machine_id()` 形如 `host/user`，指纹的 hostname 只有 host ⇒ 取 host 比。
    live_host = str(live_machine_id or "").split("/", 1)[0].strip()
    if not host:
        complaints.append("指纹里**没有 hostname** ⇒ 无法核对它是不是本机的指纹；停。")
    elif not live_host:
        complaints.append("实时机器标识**取不到** ⇒ 无法核对指纹归属；停。")
    elif host != live_host:
        complaints.append(
            f"★ 指纹描述的机器与实跑机器**不是同一台**："
            f"指纹 hostname={host}（采集于 {collected}），"
            f"实时机器={live_host} ⇒ 这份指纹**不是本批的**；停。")

    prev_host = str((prev or {}).get("hostname") or "").strip()
    if prev_host and host and prev_host != host:
        notes.append(
            f"⚠️ **容器已变**：上一份指纹 hostname={prev_host}"
            f"（{(prev or {}).get('collected_at') or '?'}），"
            f"本批={host}。换容器**改变跨批可比性的前提** ⇒ 与更早批次比数字时，"
            "必须把这一条一起写进口径；不要把两批说成「同一台机器上跑的」。")

    # ---------------- ④ ★ 新鲜度：重采到底生效了没有 ----------------
    #
    # 为什么还要判它（2026-10-03）：上面的"主机对账"只能发现**指纹来自别的机器**；
    # 若 `inspect_env.sh` 因任何原因**没能真正覆盖** `env.json`（写去了别处、
    # 中途失败、权限问题），指纹就会**仍是旧的**——而它描述的机器**恰好还是本机**
    # （例如同一容器里跑第二天的批次）⇒ 主机对账**照样通过**，只有日期是旧的。
    #
    # ⚠️ 判据是「新指纹的时间**必须晚于**上一份」，**不是**"晚于此刻"：
    #    后者要拿**远端时钟**和**本机时钟**比，而两台机器的时钟本来就可能偏
    #    ⇒ 正确的刷新会被判成失败（**假阻塞**，比漏判更烦人）。
    #    两份指纹都是**同一台远端机器**生成的 ⇒ 一比就是同钟比较，没有这个问题。
    prev_collected = str((prev or {}).get("collected_at") or "").strip()
    if prev_collected and collected:
        pc, cc = _parse_stamp_time(prev_collected), _parse_stamp_time(collected)
        if cc is None:
            complaints.append(
                f"指纹的 `collected_at={collected!r}` **解析不出时间** ⇒ 无法确认"
                "它是本次采集的；停。")
        elif pc is not None and cc <= pc:
            complaints.append(
                f"★ 指纹**没有被刷新**：采集时间 {cc.isoformat()} **不晚于**"
                f"上一份的 {pc.isoformat()} ⇒ `inspect_env.sh` 的写入没生效，"
                "产物会带着一份旧指纹；停。"
                "（这正是 §4.4 A7 的形态：机器没变、但指纹是几天前的。）")
    elif not prev_collected:
        notes.append("本次是**首次**采集（没有上一份可比）⇒ 新鲜度只能用时间戳本身"
                     "判断，无法用「必须变新」来核")

    # ---------------- ⑤ 设备记录：与配置声明的平台是否一致 ----------------
    if declared_platform:
        looks = cur.get("looks_like_mthreads_host")
        if declared_platform == "mthreads" and looks is False:
            complaints.append(
                "★ 配置声明 `hardware_platform=mthreads`，但指纹里 "
                "`looks_like_mthreads_host=false` ⇒ **设备记录与声明不符**；停。"
                "（这类不符会让「平台一致」这句话失去依据，而产物看不出区别。）")
        elif looks is None:
            notes.append("⚠️ 指纹没有 `looks_like_mthreads_host` ⇒ "
                         "无法核对设备记录与平台声明（**不是通过**）")

    return {"ok": not complaints, "complaints": complaints, "notes": notes,
            "container_changed": (prev_host != host) if (prev_host and host) else None,
            "fingerprint": {"hostname": host, "collected_at": collected,
                            "device_count": devs, "available": avail,
                            "looks_like_mthreads_host": cur.get(
                                "looks_like_mthreads_host")},
            "live_machine_id": live_machine_id}


def four_class_identity(*, root: Path, cfg_dir: str, baseline: str,
                        arms: Sequence[str]) -> dict[str, Any]:
    """★ **四类身份**（源码 / 配置 / 模型 / 环境）—— 一次记全，缺哪一类都显式写明。

    为什么四类都要（2026-10-03）：
    一个批次"可不可比"取决于**四件事同时没变**，而它们各自会独立地悄悄变：

    | 类 | 变了会怎样 | 从哪取 |
    | --- | --- | --- |
    | **源码** | 数字绑不到代码版本（EXECOPT3/4 就缺这一类） | `compute_tree_digest` |
    | **配置** | 参数变了却与旧产物并表 | `config_data_digest(raw)` 逐臂 |
    | **模型** | 换了个 revision 还当同一模型比 | 配置的 `model_revision` |
    | **环境** | 数字来自另一台机器（EXECOPT5 缺这一类） | 环境指纹 + 容器对账 |

    ⚠️ **任一取不到都写 `None`，不写占位值**——"缺"与"值"必须能分辨，
    否则下游会把缺失读成"这类身份没问题"。
    """
    out: dict[str, Any] = {}

    # ---- ① 源码 ----
    src: dict[str, Any] = {"local_tree_digest": None, "digest_scope": None,
                           "deployed_digest": "PENDING_BATCH_STAMP"}
    try:
        _p = str(root / "src")
        if _p not in sys.path:
            sys.path.insert(0, _p)
        from ascend_opt.provenance import SOURCE_DIGEST_SCOPE, compute_tree_digest

        src["local_tree_digest"] = compute_tree_digest(root)
        src["digest_scope"] = list(SOURCE_DIGEST_SCOPE)
    except Exception as exc:                                       # noqa: BLE001
        src["error"] = f"{type(exc).__name__}: {exc}"
    out["source"] = src

    # ---- ② 配置 + ③ 模型（逐臂）----
    per_arm: dict[str, Any] = {}
    d = Path(cfg_dir)
    if not d.is_absolute():
        d = root / d
    bp = Path(baseline)
    if not bp.is_absolute():
        bp = root / bp
    targets = {"baseline": bp}
    for a in arms:
        targets[str(a)] = d / f"{a}.yaml"
    for name, path in targets.items():
        entry: dict[str, Any] = {"path": str(path), "exists": path.is_file(),
                                 "config_digest": None, "model_revision": None,
                                 "error": None}
        try:
            cfg = load_config(path)
            raw = cfg.raw or {}
            rt = raw.get("runtime") or {}
            entry["config_digest"] = config_data_digest(raw)
            entry["model_revision"] = rt.get("model_revision")
            # ⚠️ 路径现读：`hardware_platform` 在 **`runtime`** 段下，不在顶层。
            #    （按记忆写顶层会**静默取到 None** ⇒ 身份校验里那条"设备记录与声明
            #    是否一致"就**一次都不会执行**，而它看起来一直在跑。）
            entry["hardware_platform"] = rt.get("hardware_platform")
            entry["execution_environment"] = rt.get("execution_environment")
        except Exception as exc:                                   # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        per_arm[name] = entry
    out["config_and_model"] = per_arm
    #: 声明的平台（从配置现读，**不新增一个命令行开关**——那会变成第二处真相）。
    out["declared_platform"] = (per_arm.get("baseline") or {}).get(
        "hardware_platform")
    return out


def step_env_fingerprint(*, ctx: Ctx, dry: bool) -> dict[str, Any]:
    """★ 批次启动前：**重采当前环境指纹 + 校验容器/设备身份**。

    两步各解决一半问题：
      * **重采**（`inspect_env.sh`）⇒ 产物里的指纹从此是"本次采集"的，
        而不是几天前从别的容器带过来的；
      * **校验** ⇒ 指纹描述的机器必须就是实跑机器，否则当场停。

    为什么不做"只在缺失时补采"：指纹**存在但陈旧**正是本次事故的形态——
    按"存在即用"的逻辑它永远会被复用，而且**看起来完全正常**。
    """
    if dry:
        print("[plan] 重采环境指纹（bash scripts/inspect_env.sh）并校验容器/设备身份")
        return {}
    root = shlex.quote(ctx.remote_root)
    venv = shlex.quote(ctx.venv_python)

    def _py(payload: str, *, step: str, timeout: float = 180.0) -> str:
        inner = (f"cd {root} && source {shlex.quote(ctx.activate)} >/dev/null 2>&1; "
                 f"{venv} -c {shlex.quote(payload)}")
        return remote_exec_checked_output(inner, step=step, timeout=timeout)

    # ---- ① 先读**上一份**指纹（刷新会覆盖它，必须先取） ----
    prev_raw = _py(
        "import json,pathlib;"
        "p=pathlib.Path('results/raw/_env/env.json');"
        "d=json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None;"
        "print('PREV', json.dumps({'hostname': (d or {}).get('hostname'),"
        " 'collected_at': (d or {}).get('collected_at')} if d else None))",
        step="读上一份环境指纹")
    prev: dict[str, Any] | None = None
    for line in prev_raw.splitlines():
        if line.startswith("PREV "):
            try:
                prev = json.loads(line[5:])
            except ValueError:
                prev = None
    if prev:
        print(f"[env-fp] 上一份指纹：hostname={prev.get('hostname')} "
              f"collected_at={prev.get('collected_at')}")
    else:
        print("[env-fp] 上一份指纹：**不存在**（首次采集）")

    # ---- ② 重采（★ 这一步是修复的核心：让产物携带"本次"的指纹） ----
    refresh = (f"cd {root} && PYTHON={venv} bash scripts/inspect_env.sh")
    out = remote_exec_checked_output(refresh, step="重采环境指纹", timeout=420)
    print(out.strip()[-800:])

    # ---- ③ 读回**整份**新指纹 + 实时机器标识，落一份机器可读记录 ----
    #
    # ⚠️ 这里读的是 `env.json` 的**全文**（约 5 KB），而不是几个"我以为存在的键"。
    #    旧版在远端拼了一张"要读的字段表"，其中 `d.get('device_count')` 与
    #    `software.mthreads_driver_version` **在真实产物里不存在** ⇒ 恒为 None
    #    ⇒ 依赖它们的判据一次都没执行（读取缺陷是静默的）。
    #    ⇒ 现在把原文带回来，由本地的 `read_env_fingerprint` 按**真实结构**解析，
    #      并有一条用**真实采集格式**做的集成用例盯着（`tests/test_env_fingerprint_*.py`）。
    ident_raw = _py(
        "import json,pathlib,sys;sys.path.insert(0,'src');"
        "from ascend_opt.provenance import detect_machine_id;"
        "d=json.loads(pathlib.Path('results/raw/_env/env.json').read_text(encoding='utf-8'));"
        "print('ENVJSON', json.dumps(d, ensure_ascii=False));"
        "print('MACHINE', json.dumps(detect_machine_id()))",
        step="读回环境指纹")
    doc: dict[str, Any] | None = None
    machine_id = ""
    for line in ident_raw.splitlines():
        if line.startswith("ENVJSON "):
            try:
                doc = json.loads(line[8:])
            except ValueError:
                doc = None
        elif line.startswith("MACHINE "):
            try:
                machine_id = str(json.loads(line[8:]) or "")
            except ValueError:
                machine_id = ""
    if doc is None:
        raise StepFailed("读回环境指纹", 1,
                         "刷新后仍读不到 results/raw/_env/env.json ⇒ 停")
    if not machine_id:
        raise StepFailed("读回环境指纹", 1,
                         "取不到**实时机器标识** ⇒ 无法与指纹对账（A7 的判据需要它）⇒ 停")

    fp = read_env_fingerprint(doc)
    print(f"[env-fp] 本批指纹：hostname={fp.get('hostname')} "
          f"collected_at={fp.get('collected_at')} device_count={fp.get('device_count')} "
          f"model={fp.get('device_model')} driver={fp.get('driver')} "
          f"torch={fp.get('torch')} torch_musa={fp.get('torch_musa')}")

    # ★ **四类身份**一起落盘（源码/配置/模型/环境）—— 只记环境的话，
    #   "这一批到底跑的是哪份代码、哪份配置、哪个模型"仍然无从核对。
    #   位置在 verdict 之前：声明的平台要从配置里现读，再交给身份校验用。
    identities = four_class_identity(
        root=ROOT, cfg_dir=ctx.cfg_dir, baseline=ctx.baseline,
        arms=ctx.arms or [],
    )

    # ★★ **指纹接受判据**（无条件执行，首次采集同样查）——
    #    设备可用 / 设备数为正 / 平台与型号匹配 / hostname 与实时实例一致 /
    #    采集时间可解析 / 必填字段非空。任一条不过 ⇒ **停**，
    #    绝不让一批数字带着"环境已核验"的样子跑出来。
    accept = env_fingerprint_acceptance(
        fp=fp, live_hostname=machine_id,
        declared_platform=identities.get("declared_platform"))
    for n in accept["notes"]:
        print(f"[env-fp] {n}")
    if not accept["ok"]:
        raise StepFailed(
            "环境指纹验收", 1,
            "指纹未通过接受判据（**不许「先跑起来再说」**）：\n  - "
            + "\n  - ".join(accept["complaints"]))
    print("[env-fp] ✓ 指纹接受判据全部通过"
          "（设备可用/设备数>0/平台与型号匹配/hostname 与实时实例一致/采集时间可解析）")

    # 新鲜度与容器变更：**必须比上一份**才能回答的那一条，单独走
    # `fingerprint_identity_verdict`（首次采集时它只记 note，不放松上面那些检查）。
    verdict = fingerprint_identity_verdict(
        prev=prev, cur=fp, live_machine_id=machine_id,
        declared_platform=identities.get("declared_platform"))
    for n in verdict["notes"]:
        print(f"[env-fp] {n}")
    if not verdict["ok"]:
        raise StepFailed("环境指纹验收", 1,
                         "指纹与上一份/容器身份对账不过：\n  - "
                         + "\n  - ".join(verdict["complaints"]))

    det = (identities.get("source") or {}).get("local_tree_digest")
    print(f"[env-fp] 本机 tree_digest={str(det)[:16]}…（部署戳由本批 launch 前刷新）")
    for name, e in (identities.get("config_and_model") or {}).items():
        print(f"[env-fp]   {name}: config={str(e.get('config_digest'))[:12]}… "
              f"model_revision={e.get('model_revision')} "
              f"platform={e.get('hardware_platform')}")
    # ⚠️ **不得**把"旧远端部署戳"当成当前源码的身份：本机摘要变了就说明源码变了，
    #    而远端戳是**上一次**部署留下的 ⇒ 恒记为 `PENDING_BATCH_STAMP`。
    identities["source"]["note"] = (
        "旧远端部署戳**不代表当前源码**（源码摘要随本地改动而变）；本批 launch 前重打戳。"
        "历史产物仍带旧摘要 ⇒ 不得用**当前**代码去'复核'它们的数字。")

    # 机器可读记录：与进度/汇总同处，便于事后核对"这批到底跑在哪台机器上"。
    record_path = f"{ctx.remote_root}/env_identity_{ctx.tag}.json"
    body = json.dumps({**verdict, "identities": identities, "tag": ctx.tag},
                      ensure_ascii=False, indent=2)
    write_remote = (f"cd {root} && {venv} -c "
                    f"{shlex.quote('import pathlib,sys;'
                                   f'pathlib.Path(sys.argv[1]).write_text(sys.argv[2],'
                                   f'encoding="utf-8",newline="\\n")')} "
                    f"{shlex.quote(record_path)} {shlex.quote(body)}")
    remote_exec_checked_output(write_remote, step="写环境身份记录", timeout=120)
    print(f"[env-fp] 身份记录 -> {record_path}")

    if not verdict["ok"]:
        raise StepFailed("环境身份校验", 1,
                         "；".join(verdict["complaints"]) +
                         "（**不要**带着陈旧/异机的指纹起跑：产物会自称一个它没跑过的环境）")
    print("[ok] 环境指纹已刷新，且描述的就是实跑机器（容器/设备身份对账通过）")
    return verdict


def step_model_snapshot(*, ctx: Ctx, dry: bool) -> None:
    """钉死的 revision 快照必须**已在**远端 HF 缓存里。

    ⚠️ 配置里把 `model_revision` 从 `main` 改成了具体 commit。若该 snapshot
    不在缓存里，`from_pretrained` 会走网络（`HF_ENDPOINT` 是镜像）或直接失败
    —— 两种都会让整个批次在**第一格**才炸，浪费一次起跑。⇒ 前置核。
    """
    cmd = (
        f"test -d ${{HF_HOME:-/root/autodl-tmp/hf}}/hub/models--Qwen--Qwen2.5-7B-Instruct"
        f"/snapshots/{EXPECTED_REVISION} && echo SNAP_OK || echo SNAP_MISSING"
    )
    if dry:
        print(f"[plan] 远端核对 revision 快照是否存在：{EXPECTED_REVISION}")
        return
    out = remote_exec_checked_output(cmd, step="模型快照核对", timeout=90)
    if "SNAP_OK" not in out:
        raise StepFailed("模型快照核对", 1,
                         f"远端 HF 缓存里没有 snapshot {EXPECTED_REVISION} ⇒ 先下载或改回 main"
                         "（改回 main 必须同步改配置与溯源声明，不得两边不一致）")
    print(f"[ok] revision 快照在位：{EXPECTED_REVISION}")


def step_idle(*, ctx: Ctx, dry: bool) -> None:
    rc_cmd = f"python3 {BUSY.name} --root {ctx.remote_root} --expect-idle"
    if dry:
        print(f"[plan] {PY_VENV} {ROOT/'scripts'/BUSY.name} --root {ctx.remote_root} --expect-idle")
        return
    rc, out = sh([PY_VENV, BUSY, "--root", ctx.remote_root, "--expect-idle"],
                 step="远端并发检测", timeout=180, check=False)
    print(out.strip())
    if rc != 0:
        raise StepFailed("远端并发检测", rc,
                         "远端有任务在跑或**探测本身失败** ⇒ 拒绝起跑"
                         "（探测失败是 UNKNOWN，不得当作空闲）")


def step_cpu_stability(*, ctx: Ctx, dry: bool) -> None:
    """CPU 稳定性必须**在推理主机上**测（计时受污染的正是那台）。"""
    cmd = (
        f"cd {ctx.remote_root} && bash -c 'source {ctx.activate} >/dev/null 2>&1; "
        "python3 scripts/check_cpu_stability.py 7 1.0'"
    )
    if dry:
        print("[plan] 远端跑 check_cpu_stability.py 7 1.0（rc!=0 ⇒ 拒绝开始正式计时）")
        return
    rc, out = _remote(["exec", cmd], step="CPU 稳定性", timeout=300, check=False)
    print(out.strip())
    if rc != 0:
        raise StepFailed("CPU 稳定性", rc, "不稳定 ⇒ **不得开始正式计时**（纪律，不是建议）")


def step_stale_lock(*, ctx: Ctx, dry: bool) -> None:
    """清理**陈旧**批次锁。⚠️ 必须在 `step_idle` **之后**调用。

    为什么需要：`launch_batch.sh` 的锁由**监视进程**在批次结束后释放。监视进程
    自己死掉（被 OOM 杀 / 整组 kill / 机器重启）时锁会永远留着 ⇒ 之后**所有**
    批次都被 rc=4/5 挡在门外，而且**不报错**。无人值守场景下这是最贵的失效：
    不是跑错，是"什么都不跑"。

    ⚠️ 安全性完全依赖调用顺序：只有"远端确认空闲"通过之后才允许删锁。
    本步骤与 `clear_stale_batch_lock.py` 内的三条条件（只删 `*.lock.d`、
    pid 确已不存在、年龄 > 10 min）合起来才是完整的四条。
    """
    cmd = (f"cd {ctx.remote_root} && python3 scripts/clear_stale_batch_lock.py "
           f"--root {LOCK_ROOT} --apply --min-age-minutes 10")
    if dry:
        print(f"[plan] 远端（**在确认空闲之后**）：{cmd}")
        return
    rc, out = _remote(["exec", cmd], step="清理陈旧锁", timeout=180, check=False)
    print(out.strip()[-1500:] or "[stale-lock] （无输出）")
    if rc != 0:
        # 非 0 可能是"有陈旧锁但清理后仍有保留项"，也可能是脚本自身报错。
        # **都不阻断**：真正能否起跑由 launch_batch.sh 的锁判定，这里只报告。
        print(f"[stale-lock] ⚠️ 退出码 {rc}（不阻断；起跑时锁判定为准）")


def step_push(*, ctx: Ctx, dry: bool) -> None:
    """推送代码/配置到远端。

    ⚠️ **先比对、只推差异**：`push_files.sh` 对每个文件要 2 次远端往返（取 sha +
    put）。本项目 PUSH_PATHS 展开后 **128 个文件**，在抖动链路上 ≈ 19 分钟，
    而这 128 个里通常只有个位数真的不同 —— 白等 19 分钟（实测踩过）。
    `batch_sha_diff.py` 把"远端现状"压成 **1 次往返**，于是这里变成：
    1 次比对 + N 次 put（N = 真实差异数）。
    """
    if dry:
        print(f"[plan] {PY_VENV.name} {SHA_DIFF.name} --remote-root {ctx.remote_root} "
              + " ".join(PUSH_PATHS))
        print(f"[plan] bash {PUSH_SH.name} --include {PUSH_INCLUDE} {ctx.remote_root} <差异文件>")
        print(f"[plan] {PY_VENV} {REMOTE_RUN.name} stamp {ctx.remote_root} --allow-dirty")
        return

    # ---- ① 快速比对（1 次往返）----
    # ⚠️ **必须消费机器可读输出**（`--emit-paths`），不要去解析人类报告：
    #    报告里 `sha 不同` 的行形如 `  scripts/x.py  local=ab.. remote=cd..`，
    #    而"取缩进行"的解析器会把**整行**当路径 ⇒ `push_files.sh` 判
    #    「跳过（不存在）」⇒ `N=0 / ok=0 / skip=0 / fail=0` ⇒ **rc=0、不报错**
    #    ⇒ 编排打印"推送源码与配置"后照常起跑，**远端其实一个文件都没更新**。
    #    实测（2026-09-30，EXECOPT4）：远端 `run_exec_opt_matrix.py` 落后本机，
    #    整批 `switches` 缺失 S04 键 —— 而 `failures=[]`，看不出来。
    rc, out = sh([str(PY_VENV), str(SHA_DIFF), "--remote-root", ctx.remote_root,
                  "--include", PUSH_INCLUDE, "--emit-paths", *PUSH_PATHS],
                 step="比对远端 sha", timeout=600, check=False)
    todo: list[str] = parse_push_todo(out)
    if rc not in (0, 1):
        # 0=全一致；1=有差异。其他 ⇒ 比对本身失败（网络/脚本错）。
        print(out, file=sys.stderr)
        raise StepFailed("比对远端 sha", rc,
                         "比对本身失败（不是「有差异」）⇒ 不要继续；"
                         "重跑 `python scripts/batch_sha_diff.py ...` 看原因。")
    # ★ 守卫 1：清单里的每个路径**必须真的存在**。
    #   伪路径（含空格后缀 / 命令续行）在这里就暴露，而不是被 push 静默跳过。
    bogus = find_bogus_todo(todo)
    if bogus:
        raise StepFailed(
            "比对远端 sha", 2,
            "待推送清单里有**不是文件**的条目（解析或比对输出格式出问题）：\n  "
            + "\n  ".join(bogus[:10])
            + f"\n（共 {len(bogus)} 条）⇒ **不要起跑**：远端与本地不是同一份代码。",
        )
    if not todo:
        print("[push] 比对结果：本地与远端**完全一致** ⇒ 跳过推送（0 次 put）")
    else:
        print(f"[push] 比对结果：{len(todo)} 个文件需要推送")
        for t in todo:
            print(f"       - {t}")
        # ---- ② 只推差异 ----
        rc, out = sh([BASH, str(PUSH_SH), "--include", PUSH_INCLUDE,
                      ctx.remote_root, *todo],
                     step="推送源码/配置", timeout=1800, check=False)
        if rc != 0:
            print(out.strip(), file=sys.stderr)
            bad = [ln for ln in out.splitlines() if "FAIL" in ln or "MISMATCH" in ln]
            raise StepFailed(
                "推送源码/配置", rc,
                "有文件未通过 sha 校验 ⇒ **不要起跑**（远端与本地不是同一份代码）。\n"
                + ("可疑行：\n  " + "\n  ".join(bad) if bad else
                   "（输出里未见 FAIL 行，见上方完整输出）\n")
                + "修法：重跑 `bash scripts/push_files.sh --include {PUSH_INCLUDE} "
                  f"{ctx.remote_root} {' '.join(todo)}`。")
        print(out.strip()[-1500:])
        # ★ 守卫 2：收尾统计必须与清单对得上（`push_files.sh` 对不存在路径
        #   只印 stderr 后 continue，退出码仍是 0 ⇒ 只看退出码会漏掉）。
        complaint = push_summary_complaint(parse_push_summary(out), len(todo))
        if complaint:
            raise StepFailed(
                "推送源码/配置", 3,
                complaint + "\n⇒ **不要起跑**。输出尾部：\n" + out.strip()[-800:],
            )
    # 溯源戳：让后续 run 的 git_commit 有意义（否则是 NOT_MEASURED）。
    # `--allow-dirty`：本工作树确实有未提交改动（S00–S03 的实现），
    # 戳里会记 git_dirty=true —— **如实记录**比"假装干净"重要。
    rc, out = _remote(["stamp", ctx.remote_root, "--allow-dirty"],
                      step="投放溯源戳", timeout=300, check=False)
    print(out.strip()[-1500:])
    # ★ 守卫 3：**戳不通过就不许起跑**（判据在 `stamp_complaint` 里，可单测）。
    complaint3 = stamp_complaint(rc, out)
    if complaint3:
        raise StepFailed(
            "投放溯源戳", rc or 3,
            complaint3 + "\n⇒ **不要起跑**。\n"
            "修法：先 `python scripts/remote_run.py stamp  <远端根> --allow-dirty` 单独跑到通过。\n"
            f"输出尾部：\n{out.strip()[-800:]}",
        )


def step_launch(*, ctx: Ctx, dry: bool) -> None:
    """**只发一次**。失败靠产物判定，绝不重试起跑。"""
    cmd = _launch_shell_string(ctx)
    if dry:
        print(f"[plan] 远端执行一次（不重试）：\n       {cmd}")
        return
    rc, out = remote_exec(cmd, step="起跑", timeout=300, check=False)
    print(out.strip())
    # 退出码语义由 launch_batch.sh 定义：0 成功；4/5 忙；6/7/9 失败。
    if rc != 0:
        raise StepFailed("起跑", rc,
                         "起跑未成功（4/5=已有批次在跑；6/7/9=起跑失败）。**不要重试**，"
                         "先看原因：重试可能把同一条 nohup 发两遍抹掉正在写的日志。")


def _read_remote_json(path: str) -> dict[str, Any] | None:
    """读远端一个 JSON 文件；读不到/坏掉/控制面抖动 ⇒ `None`（**不**抛）。"""
    try:
        out = remote_exec_checked_output(f"cat {path} 2>/dev/null || true",
                                         step="读状态", timeout=90)
    except StepFailed as exc:
        print(f"[wait] 状态回读失败（继续等待）：{type(exc).__name__}")
        return None
    out = out.strip()
    if not out.startswith("{"):
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # 原子写理论上不会读到半截；真读到就是异常，继续等下一轮。
        return None


def step_wait(*, ctx: Ctx, dry: bool) -> dict[str, Any] | None:
    """轮询进度直到**批次进程已退出**（不是"进度看起来完成了"）。

    ★ 退出通道有**三条**，必须全部识别 —— 只认一条会让本机在远端正跑完时**空转**：

    | 通道 | 谁写 | 何时写 |
    | --- | --- | --- |
    | `progress.stop.stop` | runner | 仅**协作式停止**（STOP 文件）时 |
    | `summary` 文件出现 | runner 收尾 | **总是**（预算收尾 / 全部完成 / 被中断都写）|
    | `n_done >= n_total` | runner | 项数跑满时（**但进程可能还在收尾**）|

    ⚠️ **实测失效（EXECOPT3，2026-09-30）**：远端 **35 min 就跑完 12/12 并写了 summary**，
    而本机这里只查 `stop.stop` —— 预算收尾**不写** `stop` ⇒ 条件永假 ⇒
    本机一直轮询到 `--wait-timeout-minutes`（默认 **600 min**）为止。
    症状：**远端早已空闲、本机显示"在跑"** —— 与"卡住"不可区分。

    ⇒ 现在以 **summary 文件为权威退出信号**（它是 runner 的**最后一步写入**），
    并用 `n_done >= n_total` 作为**冗余**判据（summary 写失败/读不到时兜底）。
    """
    if dry:
        print("[plan] 轮询远端进度文件直到「summary 出现」或「n_done>=n_total」或 超时")
        return None
    if ctx.no_wait:
        print("[skip] --no-wait：不轮询。稍后用 `status` 查看。")
        return None
    deadline = time.time() + ctx.wait_timeout_minutes * 60.0
    last_line = ""
    payload: dict[str, Any] | None = None
    while True:
        payload = _read_remote_json(ctx.progress)
        summary = _read_remote_json(ctx.summary)
        if payload:
            age = time.time() - float(payload.get("heartbeat_epoch") or 0)
            line = (f"[wait] {payload.get('n_done')}/{payload.get('n_total')} 完成"
                    f" | 当前={payload.get('current')} | 心跳 {age:.0f}s 前"
                    f" | 已用 {payload.get('elapsed_s', 0) / 60:.1f} min")
            if line != last_line:
                print(line, flush=True)
                last_line = line
        # ★ 三条退出通道的判据在**纯逻辑层**（唯一实现 + 可单测）；这里只做 I/O。
        decision = decide_wait_exit(progress=payload, summary=summary)
        if decision.done:
            print(f"[wait] ✓ 退出等待（通道={decision.channel}）：{decision.reason}")
            return summary or payload
        if payload:
            age = time.time() - float(payload.get("heartbeat_epoch") or 0)
            if age > max(600.0, ctx.poll_seconds * 20):
                print(f"[wait] ⚠️ 心跳已 {age / 60:.1f} min 未更新 ⇒ 可能卡住或进程已死；"
                      "详见远端日志。")
        if time.time() > deadline:
            print("[wait] ⚠️ 本地等待超时（远端批次可能仍在跑）。用 `status` 继续查看。")
            return payload
        time.sleep(ctx.poll_seconds)


def step_pull(*, ctx: Ctx, dry: bool) -> None:
    if dry:
        print(f"[plan] {PY_VENV} {PULL_INC.name} --tag {ctx.tag} "
              f"--remote-root {ctx.remote_root}/results/raw")
        return
    rc, out = sh([PY_VENV, PULL_INC, "--tag", ctx.tag,
                  "--remote-root", f"{ctx.remote_root}/results/raw"],
                 step="增量回拉", timeout=3600, check=False)
    print(out.strip()[-3000:])
    if rc != 0:
        # 回拉不完整**不是**方法失败：它是传输问题。如实报告，不掩盖、也不阻断。
        print("[pull] ⚠️ 回拉未全部完成（rc≠0）。原因可能是传输瓶颈，"
              "**不是**实验结论有问题；重跑本命令即可续取（只取本地没有的）。")


def _preflight_sequence(ctx: Ctx, *, dry: bool) -> None:
    """前置步骤的**唯一顺序表**。`plan` 与 `run` 都走这一个函数。

    ⚠️ 为什么要抽出来（2026-10-03 实测）：原先 `step_preflight` 与 `cmd_plan`
    **各写了一遍步骤序列**。我给 `run` 加了「重采环境指纹并校验身份」之后，
    `plan` 里**没有**这一步 —— 于是 `plan` 的输出少一行，而"计划里没写"
    极容易被读成"不会做"。两处各写一遍的代价不是重复劳动，是**两处真相会漂移**。
    """
    step_reachability(dry=dry)
    step_env(ctx=ctx, dry=dry)
    # ★ 环境指纹必须**重采**并**校验身份**（起因见 `fingerprint_identity_verdict`：
    #   EXECOPT5 的产物曾带着 5 天前**另一个容器**的指纹）。
    #   位置在 `step_env` 之后、`step_push` 之前：必须在产物落盘前把指纹改成
    #   "本次的"，且身份不符就**当场停**。
    step_env_fingerprint(ctx=ctx, dry=dry)
    step_model_snapshot(ctx=ctx, dry=dry)
    step_idle(ctx=ctx, dry=dry)
    # ⚠️ 顺序不可换：先确认空闲，才允许清陈旧锁。
    step_stale_lock(ctx=ctx, dry=dry)
    step_cpu_stability(ctx=ctx, dry=dry)


def step_preflight(*, ctx: Ctx, dry: bool) -> None:
    _preflight_sequence(ctx, dry=dry)


# --------------------------------------------------------------------------- #
# 命令
# --------------------------------------------------------------------------- #

def cmd_plan(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    print("=" * 72)
    print(f"批次 {ctx.tag}｜阶段 {ctx.stage}｜预算 {ctx.budget_minutes} min"
          f"｜远端 {ctx.remote_root}")
    print("=" * 72)
    # ★ 前置步骤走**同一张顺序表**（与 `run` 共用）⇒ 计划里不会漏掉新加的步骤。
    _preflight_sequence(ctx, dry=True)
    step_push(ctx=ctx, dry=True)
    step_launch(ctx=ctx, dry=True)
    step_wait(ctx=ctx, dry=True)
    step_pull(ctx=ctx, dry=True)
    print("\n[plan] 远端矩阵计划（需在远端跑 --dry-plan 才能看到确切顺序）：")
    print(f"       cd {ctx.remote_root} && bash -c 'source {ctx.activate} >/dev/null 2>&1; "
          f"{ctx.runner_cmd} --dry-plan'")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    try:
        step_preflight(ctx=Ctx(args), dry=False)
    except StepFailed as exc:
        print(f"\n[preflight] ✗ {exc}")
        return 1
    print("\n[preflight] ✓ 全部前置通过 ⇒ 可以起跑")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    try:
        print(f"===== 前置核验（批次 {ctx.tag}）=====")
        step_preflight(ctx=ctx, dry=False)
        print("\n===== 推送源码与配置 =====")
        step_push(ctx=ctx, dry=False)
        print("\n===== 起跑（只发一次）=====")
        step_launch(ctx=ctx, dry=False)
        print("\n===== 等待 =====")
        step_wait(ctx=ctx, dry=False)
    except StepFailed as exc:
        print(f"\n[run] ✗ {exc}")
        print("[run] 已停在失败点；**不要重复起跑**。修好原因后："
              "`status` 看远端状态 →`stop` 干净停止 →  换新 --tag 重跑。")
        return 1
    print("\n===== 回拉 =====")
    step_pull(ctx=ctx, dry=False)
    print(f"\n[run] 完成。远端汇总：{ctx.summary}（可用 status 查看）")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    for name, path in (("进度", ctx.progress), ("汇总", ctx.summary)):
        try:
            out = remote_exec_checked_output(f"cat {path} 2>/dev/null || echo MISSING",
                                             step=f"读{name}", timeout=90)
        except StepFailed as exc:
            print(f"[status] {name} 读取失败：{exc}")
            continue
        print(f"----- {name}：{path} -----")
        print(out.strip()[:4000])
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    ctx = Ctx(args)
    if args.dry_run:
        print(f"[plan] 远端 touch {ctx.stop_file}")
        return 0
    remote_exec(f"touch {ctx.stop_file} && echo STOPPED", step="投放 STOP", timeout=90)
    print(f"[stop] 已投放 {ctx.stop_file} ⇒ 远端会在**当前 run 跑完后**干净退出"
          "（不会截断正在写证据的那个 run）")
    return 0


def cmd_pull(args: argparse.Namespace) -> int:
    step_pull(ctx=Ctx(args), dry=args.dry_run)
    return 0


def _add_common(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """公共选项。`suppress=True` 时不给默认值 ——

    为什么：这些选项同时挂在主解析器与各子命令上，让"放在子命令前/后"都能用。
    但若子解析器带默认值，它会**覆盖**用户在子命令**之前**给的值
    （argparse 的经典陷阱：子解析器的默认值后写）。⇒ 子解析器用 `SUPPRESS`，
    由 `_val()` 兜底。
    """
    d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)   # noqa: E731
    parser.add_argument("--tag", default=d("EXECOPT-" + time.strftime("%Y%m%dT%H%M%SZ",
                                                                    time.gmtime())))
    parser.add_argument("--remote-root", default=d(DEFAULT_REMOTE_ROOT))
    parser.add_argument("--activate", default=d(DEFAULT_ACTIVATE))
    parser.add_argument("--stage", choices=("screen", "full"), default=d("screen"))
    parser.add_argument("--budget-minutes", type=float, default=d(240.0),
                        help="远端矩阵的时间预算（不含回拉与核验）")
    parser.add_argument("--min-repeats", type=int, default=d(3))
    parser.add_argument("--max-repeats", type=int, default=d(50))
    parser.add_argument("--cfg-dir", default=d(CFG_DIR_DEFAULT),
                        help="★ 臂配置目录，**显式透传**给矩阵（默认与矩阵同一个值）。")
    parser.add_argument("--baseline", default=d(BASELINE_DEFAULT),
                        help="★ 对照臂配置路径（相对仓库根或绝对路径），**显式透传**给矩阵。"
                             "不传时矩阵会用它自己的默认值，而那个默认值不会出现在任何日志里。")
    parser.add_argument("--order", default=d("block"), choices=list(ORDER_MODES),
                        help="★ 轮内格序，**显式透传**给矩阵（默认 block=历史行为）；"
                             "abba = 奇数轮反转 ⇒ 抵消「轮内位置」这一系统性因素。")
    parser.add_argument("--repeats", type=int, default=d(None),
                        help="★ **固定轮数**（如 5）。给了它就走固定模式，"
                             "总项数 = (臂数+1) × repeats 是**确定的**，"
                             "少跑即失败。不给则沿用填预算模式（--min/max-repeats）。")
    parser.add_argument("--arms", nargs="*", default=None,
                        help="显式臂集合（覆盖 --stage）；用于跑 --stage 预设之外的新机制，"
                             "如 --arms s01_both s01_both_s04_pkv s04_pkv")
    parser.add_argument("--no-wait", action="store_true", default=d(False),
                        help="起跑后立即返回（不本地轮询）")
    parser.add_argument("--poll-seconds", type=float, default=d(60.0))
    parser.add_argument("--wait-timeout-minutes", type=float, default=d(600.0))
    parser.add_argument("--dry-run", action="store_true", default=d(False),
                        help="对 stop/pull 只打印计划")


_DEFAULTS: dict[str, Any] = {
    "tag": None, "remote_root": DEFAULT_REMOTE_ROOT, "activate": DEFAULT_ACTIVATE,
    "stage": "screen", "arms": None, "budget_minutes": 240.0, "min_repeats": 3, "max_repeats": 50,
    "baseline": BASELINE_DEFAULT, "repeats": None, "cfg_dir": CFG_DIR_DEFAULT,
    "order": "block",
    "no_wait": False, "poll_seconds": 60.0, "wait_timeout_minutes": 600.0,
    "dry_run": False,
}


def _val(args: argparse.Namespace, name: str) -> Any:
    v = getattr(args, name, None)
    if v is None:
        v = _DEFAULTS[name]
    if name == "tag" and not v:
        v = "EXECOPT-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return v


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common(ap, suppress=False)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("plan", "preflight", "run", "status", "stop", "pull"):
        sub.add_parser(name, parents=[_common_parent()])
    args = ap.parse_args(argv)
    for k in _DEFAULTS:
        if args.cmd:                       # 让子命令解析后的值也能被 Ctx 读到
            setattr(args, k, _val(args, k))
    if args.cmd is None:
        args.cmd = "plan"
    return {"plan": cmd_plan, "preflight": cmd_preflight, "run": cmd_run,
            "status": cmd_status, "stop": cmd_stop, "pull": cmd_pull}[args.cmd](args)


def _common_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    _add_common(p, suppress=True)
    return p


if __name__ == "__main__":
    sys.exit(main())
