#!/usr/bin/env python3
"""远端并发检测：判**是否有真正在跑的基准任务**。

★ 为什么不能用 `pgrep -f run_step3_matrix`
----------------------------------------
实测（2026-09-27）：`pgrep -f run_step3_matrix | wc -l` 返回 **2**，但当时
**没有任何基准任务在跑**。原因是 `pgrep -f` 匹配的是**整条命令行**，而我们
经 `remote_run.py exec "..."` 调用时，**探测命令自身的文本里就含
`run_step3_matrix`** ⇒ 它匹配到了承载它的 `bash -c` 外壳（外层 + 内层子 shell
恰好 = 2）。三条不同的 `pgrep` 变体（含 "只看 python3 -u scripts/..."）**全都自匹配**。

⇒ `pgrep -f` 在此调用方式下**结构性地不可用**：它的命中数等于"外壳层级数"，
与"有没有任务"无关。假警报的危害是双向的——既会拦下正常起跑，
也会让人在**真有并发**时以为是自匹配而放行。

正确判据
--------
读 `/proc/<pid>/cmdline`（NUL 分隔），只认**解释器进程**：
命令行的**第 0 段**（argv[0]）必须是一个 python 解释器路径，**且**其后某段以
`run_step3_matrix` 结尾。这样：
  * 承载探测命令的 `bash -c` ⇒ argv[0] 是 `bash` ⇒ 排除；
  * 探测命令里出现的 pattern 文本 ⇒ 只出现在 argv[0] 之后的**参数里**，
    而 argv[0] 不是 python ⇒ 排除；
  * 真正在跑的 `python3 -u scripts/run_step3_matrix.py ...` ⇒ argv[0]=python ⇒ **命中**。

用法（本机）::

    python scripts/check_remote_busy.py [--root /root/ascend_probe_d1]
                                        [--tag S14S] [--expect-idle]

退出码：0 = 空闲/未在跑（--expect-idle 时即通过）；1 = 有任务在跑（意外）；
2 = **探测本身失败**（UNKNOWN，不得当作空闲）。

``--tag`` 用于额外检查**目标 tag 的产物目录是否在增长**（第二证据源）：
进程判据可能因权限/命名空间而漏看，产物 mtime 是独立的一道。

⚠️ **判据面是前缀表 :data:`TASK_SCRIPTS`，不是单一脚本名。**
只认 ``run_step3_matrix`` 会漏掉 ``run_s14_d1.sh`` 这类**外层队列**：
它们自己不起 python 进程、内层才调矩阵，于是"外层已起、内层未到"的窗口里
闸门报 IDLE 而实际已有批次在跑 —— **假 IDLE 会放行一个污染 results/raw 的并发批次**。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

#: 判定"这是基准任务"的脚本名（不含目录与扩展名）。
#: ⚠️ 保留为**单一字符串**是为了兼容既有测试与调用方；真正的判据是
#: :data:`TASK_SCRIPTS`，本常量必须是它的成员（模块导入时断言，见下）。
TASK_SCRIPT = "run_step3_matrix"

# ============================================================================
# ★ 判据面必须覆盖**全部会占设备的入口**，不能只认 run_step3_matrix
# ----------------------------------------------------------------------------
# 踩过的情形（2026-09-27）：`--tag S14S` 的前置闸门只查 `run_step3_matrix`。
# 而同一台机器上真正在跑的是 `scripts/run_s14_d1.sh`（它**内层**才调
# run_step3_matrix）。于是当外层 shell 已经起来、矩阵还没进到 python 阶段时，
# 闸门会报 IDLE —— **真有两个批次在跑，闸门说空闲**。
# 这比"假 BUSY"更危险：假 BUSY 只是拦下起跑（看得见），
# 假 IDLE 会**放行一个会污染证据目录的并发批次**（看不见，事后才发现）。
#
# ⇒ 判据 = "argv[0] 是 python 且 argv 含下列**任一**脚本名"。
#   注意 `run_s14_` 用**前缀**匹配（脚本名带 _d1 / _d1_min 等后缀）；
#   `fwd_costcalib_probe` 是标定链，同样占设备，必须一起拦。
# ============================================================================

#: 命中的脚本名片段（不含目录与 .py）。前缀匹配由 ``_matches_task`` 实现。
TASK_SCRIPTS: tuple[str, ...] = (
    "run_step3_matrix",
    "fwd_costcalib_probe",
    "run_s14_",              # 覆盖 run_s14_d1.sh / run_s14_d1_min.sh / run_s14_v5.sh…
    "run_ablation",
    "run_baseline",
)

# 常量与判据面不得脱钩：若有人改了 TASK_SCRIPT 却忘了这里，导入即失败。
assert TASK_SCRIPT in TASK_SCRIPTS, "TASK_SCRIPT 必须是 TASK_SCRIPTS 的成员"


def _matches_task(arg: str) -> bool:
    """某个 argv 段是否指向一个**会占设备的任务脚本**。

    参数 ``arg`` 是命令行里的一段（可能是 ``scripts/run_s14_d1.sh`` 或
    ``/root/x/scripts/run_step3_matrix.py``）。只看 basename，去掉 ``.py`` /
    ``.sh`` 后缀，再与前缀表比对。

    ⚠️ 匹配必须带**脚本形态**的前缀，不能是裸关键字：``run_s14_`` 若写成
    ``run_s14`` 会连带命中 ``--tag S14S`` 之外的东西（例如日志路径
    ``/root/s14_run.log`` 之类）。这里要求 "run_s14_" 整段出现在 basename 里，
    已经足够窄；同时**不做**子串全串匹配，避免把 `results/raw/S14S-xxx`
    这类路径当成任务。
    """
    base = arg.rsplit("/", 1)[-1]          # 去掉目录（含 Windows 反斜杠情形）
    if base.endswith(".py"):
        base = base[:-3]
    elif base.endswith(".sh"):
        base = base[:-3]
    return any(base.startswith(t) for t in TASK_SCRIPTS)

# ============================================================================
# ★ 远端探测脚本的**传输方式**（踩过的坑，务必不要"简化"回去）
# ----------------------------------------------------------------------------
# 最初写成 ``python3 -c {json.dumps(_PROBE)}``：``json.dumps`` 把多行脚本编成
# 带 ``\n`` 转义的 JSON，塞进 shell 的双引号里。远端 shell 不做 JSON 解码 ⇒
# Python 收到的是**字面反斜杠 + n**，于是：
#     SyntaxError: unexpected character after line continuation character
# 危害在于它**伪装成链路故障**：``probe_busy`` 拿不到 ``PROBE_JSON`` ⇒ 返回
# ``None`` ⇒ main 打印 "UNKNOWN：链路/权限"。真因（引号转义）被那句
# "疑似链路" 完全掩盖——而链路上一次其实只差一个成功的探测（推送/打戳都过了）。
#
# ⇒ 正确做法：**base64 编码**后再投递，远端先解码再交给 python3 执行。
#   base64 只含 ``[A-Za-z0-9+/=]``，与 shell 的引号/转义/换行**完全无交互**，
#   任何 shell（sh/bash，任意 locale）都能逐字还原脚本字节。
# ============================================================================

#: paramiko 只装在托管 venv 里；裸解释器（如 C:/Python313）会 ModuleNotFoundError。
#: 本机事实（跨项目复用）：`~/.workbuddy/binaries/python/envs/default`。
_VENV_PY = Path.home() / ".workbuddy" / "binaries" / "python" / "envs" / "default" / "Scripts" / "python.exe"


def _pick_interpreter() -> str:
    """选一个**能 import paramiko** 的解释器来驱动 remote_run.py。

    ★ 不能用 ``sys.executable``：本脚本常被系统解释器（``C:/Python313/python.exe``，
    带 torch 但**不带 paramiko**）调用，于是 ``remote_run.py`` 会以
    ``ModuleNotFoundError: No module named 'paramiko'`` 整块失败。危险之处在于
    **报错文本被吞进返回串、外层只看到"探测失败"** ⇒ 于是三态判据把
    **用错解释器**误报成 **链路故障**（都是 UNKNOWN），真因被掩盖。
    """
    cand = [str(_VENV_PY), sys.executable]
    for p in cand:
        if not p or not Path(p).exists():
            continue
        try:
            r = subprocess.run([p, "-c", "import paramiko"], capture_output=True, timeout=30)
        except Exception:  # noqa: BLE001
            continue
        if r.returncode == 0:
            return p
    # 都不可用：退回 sys.executable，让错误显式暴露（而不是静默用错的）
    return sys.executable


#: 模块级解析一次；后续所有远端调用共用。
_INTERPRETER = _pick_interpreter()

#: 远端探测脚本：读 /proc 逐个判定 argv[0] 是否为 python 解释器。
#: ``tasks`` 由 :data:`TASK_SCRIPTS` 注入 —— 判据面与调用方**共用一份**，
#: 避免"本机闸门认 5 个入口、远端只认 1 个"这种同源分叉（本项目已踩过三次）。
_PROBE = r"""
import json, os, sys
tasks = %(tasks)r
hits = []
for entry in os.listdir('/proc'):
    if not entry.isdigit():
        continue
    try:
        with open(f'/proc/{entry}/cmdline', 'rb') as fh:
            raw = fh.read()
    except OSError:
        continue                      # 进程已退出 / 权限不足 —— 跳过，不报错
    if not raw:
        continue                      # 内核线程，无 cmdline
    argv = [a.decode('utf-8', 'replace') for a in raw.split(b'\0') if a]
    if not argv:
        continue
    exe = os.path.basename(argv[0])
    # 判据①：argv[0] 必须是 python 解释器（排除 bash -c 外壳的自匹配）
    if not (exe.startswith('python')):
        continue
    # 判据②：其后任一段的 basename 必须以前缀表里的某一项开头
    def _hit(a):
        b = a.rsplit('/', 1)[-1]
        if b.endswith('.py') or b.endswith('.sh'):
            b = b[:-3]
        return any(b.startswith(t) for t in tasks)
    if not any(_hit(a) for a in argv[1:]):
        continue
    hits.append({'pid': int(entry), 'argv': argv[:6]})
print('PROBE_JSON ' + json.dumps(hits, ensure_ascii=False))
""" % {"tasks": list(TASK_SCRIPTS)}


def _remote(args: list[str], timeout: float = 120.0) -> tuple[int, str]:
    """经 remote_run.py 在远端执行。返回 (rc, stdout)。"""
    env = dict(os.environ)             # ★ 必须继承完整父环境（收窄会丢 HOME/USERPROFILE）
    env["MSYS_NO_PATHCONV"] = "1"
    py = _INTERPRETER                   # ★ 必须是带 paramiko 的解释器，不是 sys.executable
    script = str(_ROOT / "scripts" / "remote_run.py")
    proc = subprocess.run(
        [py, script, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, timeout=timeout, cwd=str(_ROOT),
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def _b64_py(source: str, py: str = "python3") -> str:
    """把一段 Python 源码用 base64 投递到远端执行。

    返回可直接交给 shell 的**单行**命令，形如::

        echo <b64> | base64 -d | python3 -

    ★ 用 base64 而不是 ``python3 -c "<src>"``：后者的安全性依赖于"源码里没有
    shell 元字符"，而这是一条**没人会去每次核对的隐含前提**。base64 的字母表
    与引号/反斜杠/换行全不相干，把隐含前提消除掉。
    """
    blob = base64.b64encode(source.encode("utf-8")).decode("ascii")
    return f"echo {blob} | base64 -d | {py} -"


def _run_probe_locally() -> list[dict] | None:
    """在**本进程所在主机**上直接跑同一份 `_PROBE`（不经 SSH）。

    存在的理由：本脚本被**两种场景**调用，且它们的"当前主机"不同：
      * 本机编排方（默认模式）—— 当前主机是 Windows，批次在**远端** ⇒ 必须 SSH；
      * 远端脚本（`run_s14_d1.sh` 等，`--local`）—— 当前主机**就是**跑批次的
        那台机器 ⇒ 不能再去 SSH。

    踩过的坑（2026-09-27）：把默认模式接到远端脚本里 ⇒ 远端自连去 `remote_run.py`，
    而远端没有 `REMOTE_*` 凭据 ⇒ 探测报 UNKNOWN(2) ⇒ 前置闸门**永远拒绝**，
    第二条腿永远起不来。症状是"批次静默只跑了一半"。
    """
    try:
        ns: dict = {}
        # ⚠️ `_PROBE` 末尾有 `print('PROBE_JSON ...')`（远端路径靠它回传）。
        #    本地执行时这行会**污染 stdout**，而调用方要拿 stdout 做判断
        #    （例如 shell 里 `$(python3 ... --local)`）。所以本地跑时临时吞掉它。
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(_PROBE, ns)                 # noqa: S102 —— 源码与远端逐字同一份
    except Exception:  # noqa: BLE001
        globals()["_LAST_PROBE_ERROR"] = f"本地执行 _PROBE 失败：{repr(sys.exc_info()[1])}"
        return None
    return ns.get("hits")


def probe_busy(root: str, tries: int = 6) -> list[dict] | None:
    """在远端探测基准进程。返回命中列表；探测本身失败返回 ``None``。

    ``None`` 与 ``[]`` **必须分开**：前者是"没查到"，后者是"确实没有"。
    混成一个"空"会让链路故障伪装成"系统空闲"——正是本项目最忌讳的静默失效。
    """
    last = ""
    for _ in range(tries):
        rc, out = _remote(["exec", f"cd {root} && {_b64_py(_PROBE)}"])
        last = out
        if rc == 0 and "PROBE_JSON" in out:
            payload = out.split("PROBE_JSON", 1)[1].strip().splitlines()[0]
            try:
                return json.loads(payload)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(3)
    # ★ 失败时把**最后一次**的原始输出留痕：区分"链路不通"与"引号/解释器选错"
    globals()["_LAST_PROBE_ERROR"] = last
    return None


def newest_artifact_mtime(root: str, tag: str, tries: int = 6) -> float | None:
    """目标 tag 的产物目录里最新文件的 **mtime（秒，相对现在）**；查不到返回 ``None``。

    第二证据源：进程可能漏看，但"产物在长"是独立的一道。
    """
    src = (
        "import glob, os, json\n"
        "fs = [f for d in glob.glob('results/raw/%s-*') for f in glob.glob(d + '/*')]\n"
        "print('MT_JSON ' + json.dumps(max((os.path.getmtime(f) for f in fs), default=-1.0)))\n"
    ) % tag
    cmd = f"cd {root} && {_b64_py(src)}"
    for _ in range(tries):
        rc, out = _remote(["exec", cmd])
        if rc == 0 and "MT_JSON" in out:
            try:
                v = float(out.split("MT_JSON", 1)[1].strip().splitlines()[0])
                return (time.time() - v) if v > 0 else None
            except Exception:  # noqa: BLE001
                pass
        time.sleep(3)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="远端并发检测（不用 pgrep -f）")
    ap.add_argument("--root", default="/root/ascend_probe_d1")
    ap.add_argument("--tag", default=None, help="额外检查该 tag 产物是否在增长")
    ap.add_argument("--expect-idle", action="store_true",
                    help="期望空闲；有任务在跑即退出码 1")
    ap.add_argument("--local", action="store_true",
                    help="在**本机**跑探测（远端脚本用：它们就在跑批次的机器上，"
                         "不能再 SSH；默认模式是给本机编排方用的）")
    args = ap.parse_args(argv)

    hits = _run_probe_locally() if args.local else probe_busy(args.root)
    if hits is None:
        where = "本地" if args.local else "远端"
        print(f"UNKNOWN：{where}探测本身失败（链路/权限）——**不得**当作'空闲'",
              file=sys.stderr)
        if not args.local:
            print(f"  驱动解释器：{_INTERPRETER}", file=sys.stderr)
        print(f"  最后一次原始输出：\n{globals().get('_LAST_PROBE_ERROR', '')[:1500]}", file=sys.stderr)
        return 2
    if hits:
        print(f"BUSY：{len(hits)} 个基准进程在跑")
        for h in hits:
            print(f"  pid={h['pid']} :: {' '.join(h['argv'])}")
    else:
        # 打印**判据面本身**（不是其中一项）：上次只查 run_step3_matrix，
        # 于是 run_s14_d1.sh 在跑时报 IDLE —— 闸门说空闲，批次已在占设备。
        where = "本地" if args.local else "远端"
        print(f"IDLE：{where}无基准进程（判据：argv[0] 为 python 且 argv 的 basename "
              f"以前缀之一开头 {list(TASK_SCRIPTS)}）")

    if args.tag and not args.local:
        age = newest_artifact_mtime(args.root, args.tag)
        if age is None:
            print(f"  产物 mtime：不可得（{args.tag} 无产物，或查询失败）")
        else:
            print(f"  产物 mtime：最新文件 {age:.0f}s 前被写过"
                  + ("  ← 仍在增长，疑似在跑" if age < 120 else ""))

    if args.expect_idle and hits:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
