#!/usr/bin/env python3
"""两级测试入口：**纯逻辑**与**含 CPU 微型模型的数值**。

为什么需要它
------------
2026-09-24 实测到一件**看起来像代码坏了、其实是选错解释器**的事：

| 解释器 | 结果 |
| --- | --- |
| `envs/default/Scripts/python.exe`（装了 paramiko、**没装 torch**） | `Ran 792 — FAILED (errors=8, skipped=25)` |
| `versions/3.13.12/python.exe`（装了 torch/transformers） | `Ran 800 — OK` |

⇒ 同一个仓库、同一份代码，两个解释器给出**结论相反**的结果。
更糟的是那 8 个 ERROR 的成因被写成了"本机未装 torch"——**本机装了**，
只是**那一个解释器**没装。于是"环境缺依赖"被记成了长期背景噪音，
而它本该是一次一秒钟就能查清的解释器选择问题。

本脚本做什么、不做什么
----------------------
**做**：把测试分成两个**有不同保证的入口**，并强制把"跳过"与"通过"分开：

* ``--group logic`` —— 全量测试，用**任意**解释器都要能跑。
  硬保证：**0 failure / 0 error**；允许跳过（缺依赖时），但**跳过不计入通过**，
  且**任何一个非数值模块被执行数 = 0 即判失败**（防"整体被跳过还报 OK"）。
* ``--group numeric`` —— 只跑含 CPU 微型模型的模块。
  硬保证：**相关用例真的执行了**（缺 torch 直接退出 2 而不是"跳过并通过"）。
* ``--group all`` —— 一次跑完（= 原来的 `unittest discover` 行为）。

**不做**：不改配置、不写产物、不联网、不碰 ``results/``。
它只读测试代码并报告结果。

为什么"跳过不算通过"要写进退出码
--------------------------------
`unittest` 的默认行为里，``skipped`` 与 ``ok`` 在**汇总行之外**长得一样
（都是点）。缺依赖时整类被跳过，汇总仍会打印 ``OK``——
这正是"把没跑过的当成跑过了"的入口。所以本脚本：

1. 逐模块报告 **executed / skipped** 两个数（``executed = testsRun − skipped``）；
2. 任一**非数值**模块 ``executed == 0`` ⇒ **判失败**（不是警告）；
3. numeric 组缺依赖 ⇒ **退出码 2**，并直接给出该用哪个解释器。

用法::

    # 先看环境（解释器、torch、transformers、能跑哪一组）
    python scripts/run_tests.py --env

    # 纯逻辑入口：任意解释器都该跑通（缺 torch 时数值项跳过，不算通过）
    python scripts/run_tests.py --group logic

    # 数值入口：必须用装了 torch/transformers 的解释器
    python scripts/run_tests.py --group numeric

    # 一次跑完（回归基线）
    python scripts/run_tests.py --group all

退出码：``0`` 通过；``1`` 有 failure/error 或纯逻辑模块未被执行；
``2`` 环境不满足（如 numeric 组缺 torch）；``3`` 用法或内部错误。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:  # `Any` 只用于注解，运行期可有可无；但仍按项目习惯显式导入
    from typing import Any
except ImportError:  # pragma: no cover
    Any = object  # type: ignore[assignment,misc]

ROOT = Path(__file__).resolve().parents[1]

#: 含 **CPU 微型模型**（torch + transformers）的测试模块。
#:
#: 声明式列表，不是靠命名约定猜。判据 = "这个模块里有没有会调用 torch 的类"。
#: ⚠️ 列表腐烂（改名/新增数值模块忘了登记）会被下面两道检查抓到：
#: ① ``--group numeric`` 时若发现 0 个用例 ⇒ 退出 3；
#: ② 有 torch 的机器上跑到与直觉不符的 skip ⇒ 汇总行里显形。
NUMERIC_MODULES: tuple[str, ...] = (
    "tests.test_validate_fp16_path",
    "tests.test_real_backend_numerics",
)

#: 本机装了 torch/transformers 的解释器（**跑 numeric 组用它**）。
#: 另一个常用解释器 ``envs/default/Scripts/python.exe`` 只装了 paramiko
#: （它是给 ``scripts/remote_run.py`` 用的）——用它跑测试会得到
#: ``8 errors / 25 skipped``，那是**选错解释器**，不是代码问题。
RECOMMENDED_INTERPRETER = (
    r"C:\Users\13944\.workbuddy\binaries\python\versions\3.13.12\python.exe"
)

#: 缺依赖类跳过的**原因子串**。用来把"环境缺依赖"与"环境里没有 git / 缺产物"
#: 这类**真实条件**分开——后者是合法跳过，前者在合适的解释器下不该出现。
#:
#: ⚠️ 两个子串都要有：同类跳过有**两种写法**，理由串不一样——
#: ① 类装饰器 ``@unittest.skipIf(torch is None, f"环境缺少 torch/transformers：…")``；
#: ② ``setUpClass`` 里 ``raise unittest.SkipTest(TORCH_MISSING_REASON)``
#:    ⇒ 理由串是**裸的异常消息** ``ModuleNotFoundError: No module named 'torch'``。
#: 只匹配 ① 会把 ② 的一批计入"其它原因"，于是"缺依赖"被少报。
_DEP_MISSING_MARKERS = ("环境缺少 torch/transformers", "No module named 'torch'")


def _probe_deps() -> tuple[bool, str]:
    """真的去 import，而不是 ``find_spec``。

    ``find_spec`` 只说明"找得到模块文件"，不说明"导得进来"（缺 DLL、
    版本不兼容的 .pyd 都会让它为真而 import 失败）。判据必须是能 import。
    """
    try:
        import torch  # noqa: F401
        from torch import nn  # noqa: F401
        from transformers import Qwen2Config  # noqa: F401
    except Exception as exc:  # pragma: no cover - 环境相关
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _module_names() -> list[str]:
    """返回**点号全名**（``tests.test_xxx``）。

    用点号全名而不是裸 stem：``NUMERIC_MODULES`` 用点号名声明，
    两处形状不一致会让"列表腐烂"检查在**功能正常时**也报错——
    那是把判据写错，而不是真有腐烂。
    """
    return sorted(f"tests.{p.stem}" for p in (ROOT / "tests").glob("test_*.py"))


def _load(names: list[str]) -> unittest.TestSuite:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in names:
        suite.addTests(loader.loadTestsFromName(name))
    return suite


# ---------------------------------------------------------------------------
# ★ pytest 补跑阶段（2026-10-02 新增）
# ---------------------------------------------------------------------------
#
# 为什么需要它
# ------------
# 本脚本用 ``unittest.TestLoader`` 收集用例，而 **unittest 只认 ``TestCase`` 子类**。
# 于是**纯 pytest 风格**的模块（裸 ``def test_xxx()`` / ``@pytest.fixture`` /
# ``@pytest.mark.parametrize``）会被**静默收集到 0 个**——**不报错、不留痕**，
# 只是从总数里少掉。实测（2026-10-02）：
#
#   pytest  收集 **1588** 个；unittest 收集 **1575** 个；差 **13**，
#   全部在 ``tests/test_plot_step16_decouple.py``（纯 pytest 风格，零 ``TestCase``）。
#
# 危害与"跳过"同族但**更隐蔽**：跳过至少在汇总行的 ``skipped`` 里有个数，
# 而"收集不到"连个数都没有——`testsRun` 直接少 13，汇总照样 `OK`。
# 这正是 README §7「标签与结论准入」要挡的那类事：**"用例存在" ≠ "用例被跑过"**。
#
# 做法
# ----
# 1. **先探测 pytest 是否可用**（在解释器里真实 import，不看 find_spec）；
# 2. 对每个模块分别数 **unittest 收集数**与 **pytest 收集数**；
# 3. ``pytest_only = pytest 数 − unittest 数``；>0 的模块走 **pytest 子进程补跑**；
# 4. 补跑的结果并入总数与逐模块行（标 ``[pytest]``），**失败同样计入硬失败**；
# 5. 若 pytest **不可用**且有模块 unittest 零收集 ⇒ **硬失败**（不许静默漏），
#    并明说"这 13 个没有被执行"——缺工具是**可解释的拒签**，不是"没问题"。
#
# ⚠️ 不变量保持不变：**非数值模块被执行数为 0 ⇒ 硬失败**（现在 executed 含补跑）。


def _probe_pytest() -> tuple[bool, str]:
    """真实 import pytest —— 与 ``_probe_deps`` 同一判据风格。"""
    try:
        import pytest  # noqa: F401
    except Exception as exc:  # pragma: no cover - 环境相关
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _name_to_path(name: str) -> str:
    """``tests.test_foo`` -> ``tests/test_foo.py``（pytest 要路径，unittest 要点号名）。"""
    return name.replace(".", "/") + ".py"


_NODEID_RE = re.compile(r"^(?P<node>tests/\S+?\.py::\S+)$", re.MULTILINE)


def _pytest_collect_nodeids(name: str) -> tuple[int | None, list[str]]:
    """一次 ``--collect-only`` 同时给出**收集数**与 **nodeid 列表**。

    ``count is None`` = **测不出来**（``--collect-only`` 非 0/5 退出）；与 ``0``
    （确实一个都没有）**必须分开**——混为一谈会把"pytest 起不来"读成"模块是空的"。

    ★ 合并的理由（2026-10-02）：原先 `_pytest_collect_count` 与 `_run_pytest_subset`
    各起一次子进程做同一件事 ⇒ **每模块 2 次** pytest 启动。40 个模块下 `--group logic`
    从 ~7 min 涨到 >20 min。收集只需一次。
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", _name_to_path(name), "--collect-only", "-q"],
        cwd=str(ROOT), capture_output=True, text=True, errors="replace", timeout=300,
    )
    if proc.returncode not in (0, 5):  # 5 = 没收集到（pytest 的约定）
        return None, []
    return len(_NODEID_RE.findall(proc.stdout)), _NODEID_RE.findall(proc.stdout)


#: 已知的**外部**噪声签名（不是被测代码的失败）。
#:
#: 目前只有一条：沙箱把 `rm` 换成"送入回收站"的垫片，它在清理 pytest 临时目录时
#: 拿到 `\\?\C:\...` 扩展路径，Windows 报 `Errno 53 找不到网络路径` ⇒ 垫片
#: fail-closed、**返回非零退出码**（同时把一段 JSON 打到 stdout）。
#:
#: ⚠️ **它不构成任何豁免**（2026-10-03 起）：仅用于在失败信息里补一句排查方向。
#: 历史教训：旧版把它当作"命中即豁免"的依据；更早的一版甚至以"通过数/进度点/
#: 旁路计数**看起来齐全**"为依据 —— 那条口径漏了 teardown，于是一个真实的夹具
#: 收尾缺陷被读成了通过（见 `_classify_pytest_run` 的注释）。两种都属于
#: "看起来像已知问题 ⇒ 放行"，**都已被删除**。
EXTERNAL_NOISE_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("沙箱 safe-delete 垫片清理临时目录失败", ("Errno 53", "找不到网络路径")),
)


def _match_external_noise_hint(text: str) -> str | None:
    """在文本里找**已知外部噪声**的线索；命中返回标签，否则 `None`。

    ⚠️⚠️ **它只用于在失败信息里补一句提示，绝不改变判定。**
    2026-10-03 之前这里是"签名命中 ⇒ 自动豁免非零退出码"，已**删除**：
    以输出里的**字符串**为豁免依据，等于让**任何**能打印那句话的路径都能
    把自己的失败变成通过（测试自己打印、日志回显、路径名里带该串……），
    而且它与本仓反复记的"看起来像已知问题 ⇒ 放行"是同一个坑。
    ⇒ 现在的口径：**非零退出码一律判失败**；提示只告诉你"往哪个方向查"。
    """
    for label, markers in EXTERNAL_NOISE_HINTS:
        if all(m in text for m in markers):
            return label
    return None


class _PytestVerdict:
    """一次 pytest 补跑的判定结果（供调用方记账，也供测试直接断言）。

    ★ **两件事分开记**（2026-10-03 加）：
      * `assertion_failures` —— **用例断言**层面出的问题（某相位的 failed/error、
        或某个 nodeid 根本没跑到）；
      * `session_failures`  —— **测试会话**层面出的问题（退出码、两处退出码不一致、
        收集失败/internal error、相位记录缺失、nodeid 集合对不上）。

    为什么必须分开：两者**修法完全不同**。
    "断言失败"要去改被测代码或期望；"会话没正常完成"要去查**环境和运行器**
    （收集崩了、进程被信号杀掉、插件版本不同步）。
    把它们混在一个 `hard_failures` 列表里时，报告只能给出一句"有 N 处问题"，
    接的人得自己一条条读才能知道该查哪边 —— 而实测里这两种情况**都出现过**。
    """

    __slots__ = ("passed", "failed", "error", "skipped",
                 "assertion_failures", "session_failures")

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.error = 0
        self.skipped = 0
        self.assertion_failures: list[str] = []
        self.session_failures: list[str] = []

    @property
    def hard_failures(self) -> list[str]:
        """两者的**只读合并**（保留旧名字，便于既有调用方与测试逐个迁移）。"""
        return self.assertion_failures + self.session_failures

    @property
    def defects(self) -> int:
        return self.failed + self.error

    @property
    def assertions_clean(self) -> bool:
        """**用例断言**是否全过（不看会话是否正常完成）。"""
        return not self.assertion_failures and self.failed == 0 and self.error == 0

    @property
    def session_completed(self) -> bool:
        """**测试会话**是否正常完成（与断言结果无关）。"""
        return not self.session_failures


def _classify_pytest_run(
    *, returncode: int, todo: list[str], outcome: dict[str, Any] | None, text: str
) -> _PytestVerdict:
    """把一次 pytest 补跑的**原始证据**判成结论。**纯函数**，便于逐形态实测。

    证据有四份：① 退出码；② 我们要求跑的 ``todo``（预期 nodeid）；
    ③ 旁路 JSON；④ 合并后的 stdout/stderr 文本（仅用于**识别外部噪声签名**）。

    ★★ 判据基底（2026-10-03 重写，起因是一次真实遮蔽）
    ---------------------------------------------------
    旧版允许"通过数/进度点/旁路计数**齐全** ⇒ 豁免非零退出码"。那条豁免的
    口径**漏了 teardown 相位**，于是实测出现：

        pytest 真实输出 : 1 passed, 1 error      （夹具收尾抛异常，退出码 1）
        旁路 JSON       : passed=1 error=0 failed=0，collected=1，合计=1
        ⇒ 旧规则判"收尾阶段噪声，不计为 error" ⇒ **真实缺陷被读成通过**

    ⇒ 新版**不再以"计数齐全"作任何豁免依据**，改成两条硬条件 + 一条归因条件：

    1. **集合对账（双向）**：预期 nodeid 集合 == 旁路里 `reported` 集合。
       少一个（漏跑/中途死）或多一个（跑错东西）都判失败。
       只比个数不行：个数相等也可能是"少跑一个、多跑一个"。
    2. **逐相位核对**：每个 nodeid 的 setup/call/teardown 都不得为 `failed`，
       且相位记录不缺（setup 被 skip 时才允许没有 teardown）。
    3. **非零退出码一律判失败** —— 不再有任何"按输出字符串豁免"的路径，
       也**包括**"集合与相位看起来全都正常"的那种情况：退出码非零本身
       就是一个**未解释的信号**，正确处置是查环境，不是让判定器学会忽略它。
       并且**两处退出码**（进程 `returncode` 与 pytest 自报 `exitstatus`）
       必须一致，否则同样判失败（见下）。
    """
    v = _PytestVerdict()

    # ---------- 旁路缺失：不接受 stdout 兜底做出的"通过" ----------
    if outcome is None:
        v.session_failures.append(
            "旁路 JSON **缺失** ⇒ 无法核对 nodeid 与逐相位结局。"
            "不接受「用 stdout 兜底」做出的通过判定（那正是 2026-10-02 的错法）。"
        )
        # 计数仍从 stdout 取，只为报告用；结论已经是失败。
        for key, pat in (("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"),
                         ("error", r"(\d+) error"), ("skipped", r"(\d+) skipped")):
            m = re.search(pat, text)
            setattr(v, key, int(m.group(1)) if m else 0)
        if returncode != 0:
            v.session_failures.append(f"pytest 退出码 {returncode}（旁路缺失，无法归因）")
        return v

    v.passed = int(outcome.get("passed", 0))
    v.failed = int(outcome.get("failed", 0))
    v.error = int(outcome.get("error", 0))
    v.skipped = int(outcome.get("skipped", 0))

    # ---------- 0) ★ 两处退出码必须一致（2026-10-03 补） ----------
    #
    # `returncode`（进程退出码）与旁路文件里的 `exitstatus`（**pytest 自报**）
    # 是**两个独立来源**。旧实现只读前者，把后者**写出来却从不比较** ⇒
    # "pytest 自己认为该以非零退出、而进程恰好退了 0"（或反过来）会被静默放过。
    # ⇒ 不一致就判失败：两处打架时**不许挑对自己有利的那一个**。
    # ⚠️ 字段**缺失**同样判失败：它意味着旁路插件与本运行器**版本不同步**，
    #    而那种漂移本身就会让其余字段的口径变得不可信。
    recorded_exit = outcome.get("exitstatus")
    if recorded_exit is None:
        v.session_failures.append(
            "旁路文件里**没有** `exitstatus` 字段 ⇒ 无法与进程退出码对账"
            "（插件与本运行器版本不同步？）⇒ 判定依据不完整，判失败。"
        )
    elif int(recorded_exit) != int(returncode):
        v.session_failures.append(
            f"**退出码两处不一致**：进程 returncode={returncode}，"
            f"而 pytest 自报 exitstatus={recorded_exit} ⇒ 至少一处不可信，判失败。"
        )

    per_nodeid: dict[str, dict[str, str]] = outcome.get("per_nodeid") or {}
    reported = set(outcome.get("reported") or [])
    expected = set(outcome.get("expected") or [])
    want = set(todo)

    # ---------- 1) 集合对账（双向） ----------
    missing = sorted(want - reported)
    extra = sorted(reported - want)
    if missing:
        v.assertion_failures.append(
            f"{len(missing)} 个**预期** nodeid 没有任何结局（漏跑/中途退出）："
            f"{missing[:3]}{' …' if len(missing) > 3 else ''}"
        )
    if extra:
        v.session_failures.append(
            f"{len(extra)} 个 nodeid 报了结局但**不在**预期内："
            f"{extra[:3]}{' …' if len(extra) > 3 else ''}"
        )
    if expected != want:
        only_side = sorted(expected - want)
        only_want = sorted(want - expected)
        v.session_failures.append(
            f"旁路记录的 `expected` 与调用方的 todo 不一致"
            f"（旁路多 {len(only_side)}、少 {len(only_want)}）"
            f"：旁路多 {only_side[:2]}／todo 多 {only_want[:2]}"
        )

    # ---------- 2) 逐相位核对（不只数总数） ----------
    bad_phases: list[str] = []
    incomplete: list[str] = []
    for node, phases in sorted(per_nodeid.items()):
        for phase in ("setup", "call", "teardown"):
            if phases.get(phase) == "failed":
                bad_phases.append(f"{node}[{phase}]")
        # 相位记录完整性 —— 判据必须贴合 pytest 的**实际发报规则**
        # （2026-10-03 实测三态，别照直觉写）：
        #   * setup 被 skip  ⇒ **没有** call、**没有** teardown；
        #   * setup 失败     ⇒ **没有** call，但**有** teardown（夹具收尾仍会跑）；
        #   * setup 通过     ⇒ call 与 teardown 都有。
        # ⚠️ 把"setup 失败时没有 call"当成"记录不完整"会造出**假失败**
        #    （第一版就踩了，实测 test_setup.py 被多报一条）。
        setup_status = phases.get("setup")
        if "setup" not in phases:
            incomplete.append(f"{node}（缺 setup 记录）")
        else:
            if setup_status not in ("skipped", "failed") and "call" not in phases:
                incomplete.append(f"{node}（setup 通过却缺 call 记录）")
            if setup_status != "skipped" and "teardown" not in phases:
                incomplete.append(f"{node}（缺 teardown 记录）")
    if bad_phases:
        v.assertion_failures.append(
            f"{len(bad_phases)} 个**相位**失败（含 teardown，旧口径漏计）："
            f"{bad_phases[:3]}{' …' if len(bad_phases) > 3 else ''}"
        )
    if incomplete:
        v.session_failures.append(
            f"{len(incomplete)} 个 nodeid 的相位记录**不完整**（收尾丢报？）："
            f"{incomplete[:3]}{' …' if len(incomplete) > 3 else ''}"
        )

    # ---------- 3) 计数与会话级异常 ----------
    if v.defects:
        v.assertion_failures.append(
            f"用例失败/错误：failed={v.failed} error={v.error}"
            f"（error 含 setup 与 teardown 两个相位）"
        )
    session_errors = list(outcome.get("session_errors") or [])
    if session_errors:
        v.session_failures.append(
            f"{len(session_errors)} 条**会话级异常**（收集失败 / internal error）："
            f"{session_errors[:2]}"
        )

    # ---------- 4) 退出码：**一律判失败**（2026-10-03 取消签名豁免） ----------
    #
    # 这里原先写着"若集合双向一致、相位无 failed、无会话异常、failed=error=0，
    # 且命中 KNOWN_EXTERNAL_NOISE 签名 ⇒ 归因为外部噪声、**不计为失败**"。
    # 那条规则**已删除**，理由有三条，且都不依赖后来发生了什么：
    #   ① **依据不稳**：豁免靠的是输出文本里的**字符串**，而任何路径都可能打印它
    #      （测试自己 print、日志回显、临时目录名里带该串……）⇒ 它随时可能被"蹭到"；
    #   ② **方向反了**：外部噪声是**环境**的问题，正确处置是修环境或让批次隔离它，
    #      而不是让**判定器**学会忽略它 —— 判定器一旦学会忽略，就再也分不清
    #      "又是那个噪声"与"真的是这个噪声"；
    #   ③ 它与本仓反复记的"看起来像已知问题 ⇒ 放行"是**同一个坑**（2026-10-02 的
    #      teardown 遮蔽就是这么来的）。
    # ⇒ 现在：非零退出码**一律**判失败；已知噪声线索只作为**排查提示**附在失败信息里。
    if returncode != 0:
        msg = (f"pytest 退出码 {returncode} ⇒ **一律判失败**"
               "（自 2026-10-03 起，不再有「命中已知错误字符串即豁免」这条规则）")
        if v.hard_failures:
            msg += f"；上面已列出 {len(v.hard_failures)} 条具体缺陷"
        else:
            msg += ("；集合已双向核对、三相位无 failed、无会话级异常、failed=error=0，"
                    "但退出码非零本身就是一个**未解释的信号**")
        hint = _match_external_noise_hint(text)
        if hint is not None:
            msg += (f"。⚠️ 文本里出现已知外部噪声的线索『{hint}』——"
                    f"**这只是排查方向，不构成豁免**")
        msg += f"。原文尾部：{text.strip()[-300:]}"
        v.session_failures.append(msg)
    return v


def _run_pytest_subset(name: str, *, verbose: bool,
                       precollected: tuple[int | None, list[str]] | None = None) -> _ModuleStat:
    """只跑 ``name`` 下**unittest 收不到**的那些 nodeid，返回一条 ``[pytest]`` 统计。

    为什么要精确到 nodeid 而不是整文件跑：整文件跑会把该文件里**已经被
    unittest 跑过**的用例再跑一遍 ⇒ 总数虚高（同一个用例被记两次）。
    上游 ``collect-only`` 给出的 nodeid 与 unittest 的收集结果做差集，
    才是真正需要补的部分。
    """
    stat = _ModuleStat(f"{name} [pytest]")
    _n_collected, nodeids = (precollected if precollected is not None
                             else _pytest_collect_nodeids(name))
    if not nodeids:
        return stat

    # 与 unittest 已收集的类名/方法名做差集：unittest 的 id 形如
    # `tests.test_x.SomeClass.test_m`；pytest 的 nodeid 形如 `tests/test_x.py::SomeClass::test_m`。
    have = set()
    for case in _load([name]):
        for t in _iter_cases(case):
            have.add(f"{type(t).__name__}::{t._testMethodName}")

    todo = [n for n in nodeids if n.split("::", 1)[1] not in have]
    if not todo:
        return stat

    args = [sys.executable, "-m", "pytest", *todo, "-q", "-p", "no:cacheprovider"]
    if verbose:
        args.append("-v")
    # ★★ 主证据：让 pytest 把**逐用例结局**写到旁路文件（2026-10-02 续）。
    #   为什么不继续解析 stdout：`-q` 的 `N passed` 汇总行与收尾**同帧输出**，
    #   沙箱 `safe-delete` 垫片一崩就整段丢；而进度点也可能被那段 JSON 挤掉。
    #   两次都落进"数不出结果 ⇒ 记 error"⇒ 把 13 个通过报成 1 个错误。
    #   根因是**判据依赖了会被同一故障摧毁的证据**。旁路文件由 pytest 自己的
    #   hook 在 `sessionfinish` 里写出，与 stdout 无关，收尾再崩也不影响。
    outcome_path = Path(tempfile.mkdtemp(prefix="pytestout_")) / "outcome.json"
    env = dict(os.environ)
    env["PYTEST_OUTCOME_JSON"] = str(outcome_path)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "scripts")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    args += ["-p", "_pytest_outcome_plugin"]
    proc = subprocess.run(
        args, cwd=str(ROOT), capture_output=True, text=True,
        errors="replace", timeout=3600, env=env,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    outcome: dict[str, Any] | None = None
    try:
        if outcome_path.is_file():
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        outcome = None

    # ★★★ 判定**全部**交给纯函数 `_classify_pytest_run`（2026-10-03 重写）。
    #
    #   老版在这里内联了"计数齐全 ⇒ 豁免非零退出码"的逻辑，而它的口径漏了
    #   teardown，于是一个真实的夹具收尾缺陷被判成通过。现在：
    #   * 集合对账（预期 nodeid ↔ 实际有结局的 nodeid，**双向**）；
    #   * 逐相位核对（setup/call/teardown 缺一不可，setup 被 skip 时例外）；
    #   * 会话级异常（收集失败 / internal error）；
    #   * 非零退出码**一律**判失败（已知噪声线索只作提示，见 `_match_external_noise_hint`）。
    #   ⇒ 想改动判据，只改那一个函数即可；测试也直接调它，不必抄一份。
    verdict = _classify_pytest_run(
        returncode=proc.returncode, todo=todo, outcome=outcome, text=out
    )
    stat.run = max(
        len(todo), verdict.passed + verdict.failed + verdict.error + verdict.skipped
    )
    stat.skipped = verdict.skipped
    stat.failures = verdict.failed
    stat.errors = verdict.error
    if verdict.hard_failures:
        # 至少记 1 个 error，保证上游的硬失败分支一定会被触发——
        # 即使某条硬失败不是"用例失败"（例如 nodeid 集合不符），
        # 也**不能**让它退化成一个没人看的告警。
        stat.errors = max(stat.errors, 1)
        # ★ **两类问题分开列**：接的人据此判断该"改代码/改期望"还是"查环境/查运行器"。
        reasons: list[str] = []
        if verdict.assertion_failures:
            reasons.append("【用例断言】" + "；".join(verdict.assertion_failures))
        if verdict.session_failures:
            reasons.append("【测试会话】" + "；".join(verdict.session_failures))
        stat.skip_reasons = reasons
    if verbose:
        print(out, file=sys.stderr)
    return stat


def _iter_cases(suite: unittest.TestSuite | unittest.TestCase):
    """递归展开 suite，产出 ``TestCase`` 实例。"""
    if isinstance(suite, unittest.TestCase):
        yield suite
        return
    for item in suite:  # type: ignore[union-attr]
        yield from _iter_cases(item)


class _ModuleStat:
    __slots__ = ("name", "run", "skipped", "failures", "errors", "skip_reasons")

    def __init__(self, name: str) -> None:
        self.name = name
        self.run = 0
        self.skipped = 0
        self.failures = 0
        self.errors = 0
        self.skip_reasons: list[str] = []

    @property
    def executed(self) -> int:
        """**真正跑到断言的**用例数。skipped 是从 ``testsRun`` 里扣掉的。"""
        return self.run - self.skipped

    @property
    def dep_skipped(self) -> int:
        return sum(
            1 for r in self.skip_reasons if any(m in r for m in _DEP_MISSING_MARKERS)
        )


def _run_one(name: str, verbose: bool) -> tuple[_ModuleStat, unittest.TestResult]:
    """逐模块跑，是为了拿到**每模块**的 executed/skipped——整体跑只有总数，
    而"某个模块整个被跳过"这件事只在分模块视图里才看得见。"""
    suite = _load([name])
    stream = sys.stderr if verbose else io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2 if verbose else 0)
    result = runner.run(suite)
    stat = _ModuleStat(name)
    stat.run = result.testsRun
    stat.skipped = len(result.skipped)
    stat.failures = len(result.failures)
    stat.errors = len(result.errors)
    stat.skip_reasons = [r for _, r in result.skipped]
    return stat, result


def _fmt_row(stat: _ModuleStat) -> str:
    tag = "数值" if stat.name in NUMERIC_MODULES else "逻辑"
    if stat.name.endswith(" [pytest]"):
        tag = "pytest"
    flag = ""
    if stat.errors or stat.failures:
        flag = "  <== 失败"
    elif stat.executed == 0 and not stat.name.endswith(" [pytest]"):
        flag = "  <== 未执行任何用例"
    return (
        f"  [{tag}] {stat.name:44s} 通过 {stat.executed:3d}  跳过 {stat.skipped:3d}"
        f"  失败 {stat.failures}  错误 {stat.errors}{flag}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="两级测试入口（纯逻辑 / 含 CPU 微型模型的数值）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--group", choices=("logic", "numeric", "all"), default="all")
    ap.add_argument("--env", action="store_true", help="只做环境体检，不跑测试")
    ap.add_argument("--verbose", "-v", action="store_true", help="把每个用例打印出来")
    args = ap.parse_args(argv)

    has_torch, torch_exc = _probe_deps()
    has_pytest, pytest_exc = _probe_pytest()

    print("=== 测试环境 ===")
    print(f"  解释器     : {sys.executable}")
    print(f"  Python     : {sys.version.split()[0]}")
    print(f"  torch/transformers : {'可用' if has_torch else '不可用 — ' + torch_exc}")
    print(f"  pytest     : {'可用' if has_pytest else '不可用 — ' + pytest_exc}")
    print(f"  仓库根     : {ROOT}")
    if not has_torch:
        print(f"  ⚠️ 跑 --group numeric 需要：{RECOMMENDED_INTERPRETER}")
        print("     （scripts/remote_run.py 用的 envs/default 解释器没有 torch）")

    if args.env:
        return 0 if has_torch else 2

    all_names = _module_names()
    if args.group == "numeric":
        names = [n for n in all_names if n in NUMERIC_MODULES]
        missing = [n for n in NUMERIC_MODULES if n not in all_names]
        if missing:
            print(f"✗ 声明的数值模块不存在（列表腐烂）：{missing}", file=sys.stderr)
            return 3
        if not has_torch:
            print(
                "\n✗ numeric 组要求 torch/transformers 真实可用；"
                "缺依赖时**跳过不等于通过**，因此这里直接失败而不是放过。\n"
                f"  请改用：{RECOMMENDED_INTERPRETER} -m scripts.run_tests --group numeric",
                file=sys.stderr,
            )
            return 2
    elif args.group == "logic":
        names = all_names
    else:
        names = all_names

    if not names:
        print("✗ 没有发现任何测试模块", file=sys.stderr)
        return 3

    print(f"\n=== 运行（group={args.group}，{len(names)} 个模块）===")
    stats: list[_ModuleStat] = []
    total = _ModuleStat("<总计>")
    hard_failures: list[str] = []
    #: ``name -> pytest 应收数``（仅当 pytest 可用且确有多余用例时才记）。
    pytest_only: dict[str, int] = {}

    for name in names:
        stat, _ = _run_one(name, args.verbose)
        stats.append(stat)
        total.run += stat.run
        total.skipped += stat.skipped
        total.failures += stat.failures
        total.errors += stat.errors

        if stat.errors or stat.failures:
            hard_failures.append(f"{name}: failures={stat.failures} errors={stat.errors}")

        # ---- ★ pytest 补跑（unittest 收不到的用例）------------------------
        #
        # ⚠️ 触发条件**不能**只看 `n_py > stat.run`：pytest 会把
        # `@pytest.mark.parametrize` **展开成多条 nodeid**，而 unittest 只把它当
        # **一个**测试方法 ⇒ `n_py` 天然大于 `stat.run`，即使该文件里根本没有
        # unittest 收不到的用例。2026-10-02 实测就是这个假阳性：
        # `test_validate_fp16_path` unittest 收 23、pytest 收 48（参数化展开），
        # 于是进了补跑分支，**差集却是空的** ⇒ 打出一条 `[pytest] 通过 0` 的空行，
        # 看上去像"补跑跑了 0 个"，实际是"没有需要补的"。
        #
        # 所以判据改成：**差集非空**才补跑（`_run_pytest_subset` 返回 `run>0`）。
        if has_pytest:
            # 一次 collect-only 同时拿"收集数"与"nodeid 列表"（见函数注释的性能理由）。
            n_py, nodeids = _pytest_collect_nodeids(name)
            if n_py is None:
                hard_failures.append(
                    f"{name}: pytest 收集数**测不出来**（`--collect-only` 非 0/5 退出）——"
                    "无法判断是否存在 unittest 收不到的用例，按未核实处理"
                )
            else:
                # ⚠️ 只在"pytest 收集数 > 0"时才值得起第二次子进程；否则差集必空。
                extra = (_run_pytest_subset(name, verbose=args.verbose,
                                            precollected=(n_py, nodeids))
                         if n_py > 0 else _ModuleStat(f"{name} [pytest]"))
                if extra.run or extra.failures or extra.errors:
                    pytest_only[name] = extra.run
                    stats.append(extra)
                    total.run += extra.run
                    total.skipped += extra.skipped
                    total.failures += extra.failures
                    total.errors += extra.errors
                    if extra.errors or extra.failures:
                        hard_failures.append(
                            f"{name} [pytest]: failures={extra.failures} "
                            f"errors={extra.errors}"
                        )
                    # 补跑后把本模块的"被执行"计入主行，供下面零收集判据使用。
                    stat.run += extra.run
                    stat.skipped += extra.skipped
        else:
            # pytest 不可用 ⇒ 只能靠"unittest 是否收到东西"判断。
            # 若某模块 unittest 零收集，则**无法排除**它全是 pytest 风格 ⇒ 硬失败。
            if stat.run == 0:
                hard_failures.append(
                    f"{name}: unittest 收集数为 0，且 **pytest 不可用**"
                    f"（{pytest_exc}）⇒ 无法排除「该模块的用例全部未被执行」。"
                    "请用装了 pytest 的解释器重跑（判据不允许静默漏用例）"
                )

        # 不变量：**纯逻辑模块不得被整体跳过**。
        # 只对 logic/all 组生效——numeric 组本来就只跑数值模块。
        if args.group in ("logic", "all") and name not in NUMERIC_MODULES:
            if stat.executed == 0:
                hard_failures.append(
                    f"{name}: 非数值模块被执行数为 0（整体跳过）——"
                    "纯逻辑测试不得因为缺依赖被整块跳过"
                )

    print("\n=== 逐模块 ===")
    for stat in stats:
        print(_fmt_row(stat))

    print("\n=== 汇总 ===")
    print(
        f"  用例 {total.run}｜**真正执行 {total.executed}**｜跳过 {total.skipped}"
        f"｜失败 {total.failures}｜错误 {total.errors}"
    )
    if pytest_only:
        n_extra = sum(pytest_only.values())
        detail = ", ".join(f"{k.split('tests.')[-1]}={v}" for k, v in pytest_only.items())
        print(
            f"  ★ 其中由 **pytest 补跑**的 {n_extra} 项（unittest 收集不到）：{detail}"
        )
        print(
            "     ⚠️ 这些用例此前**从未进入回归总数**（unittest 只认 TestCase 子类，"
            "纯 pytest 风格的裸函数会被静默收 0 个）。"
        )
    if total.skipped:
        print("  ⚠️ 跳过项**不计入通过**（AGENTS 附录 A5：跳过 ≠ 通过）。原因分布：")
        buckets: dict[str, int] = {}
        for stat in stats:
            for reason in stat.skip_reasons:
                key = next(
                    (m for m in _DEP_MISSING_MARKERS if m in reason), reason[:60]
                )
                buckets[key] = buckets.get(key, 0) + 1
        for key, cnt in sorted(buckets.items(), key=lambda kv: -kv[1]):
            print(f"    {cnt:3d} × {key}")

    dep_skips = sum(s.dep_skipped for s in stats)
    if args.group == "numeric" and dep_skips:
        hard_failures.append(
            f"numeric 组出现 {dep_skips} 项依赖缺失造成的跳过 ⇒ 数值验收未成立"
        )

    if hard_failures:
        print("\n=== 失败 ===")
        for line in hard_failures:
            print(f"  ✗ {line}")
        return 1

    print("\n✅ OK：0 failure / 0 error；纯逻辑模块全部真的执行了")
    if total.skipped:
        print(f"   （仍有 {total.skipped} 项跳过，它们**不**构成验收证据）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
