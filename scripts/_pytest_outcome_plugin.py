"""把 pytest 的**逐用例结局**写成一行机器可读的 JSON。

为什么需要它（2026-10-02）
--------------------------
`run_tests.py` 的 pytest 补跑原先靠**解析人看的输出**判断"跑没跑完"：

1. 数 `N passed` 汇总行 —— 但 `-q` 下汇总行与收尾**同帧输出**，
   沙箱 `safe-delete` 垫片一崩就**整段丢**；
2. 退而数进度点 `.....` —— 但那个垫片也可能把 JSON 直接打到 stdout，
   把进度行挤掉/打断。

两次都落进"数不出结果 ⇒ 记 error"的分支，把 **13 个通过报成 1 个错误**。
根因是**判据依赖了一份会被同一故障摧毁的证据**。

本插件用的是**独立**证据：每个用例的结局由 pytest 自己的 hook 交给我们，
写进一个**旁路文件**（与 stdout 无关）。收尾阶段再怎么崩，这个文件已经落盘。

★★ 2026-10-03 修正一处**同源缺陷**：旧版只记 setup/call，**漏掉 teardown**
------------------------------------------------------------------------
旧版只在 ``when == "call"`` 与 ``when == "setup"`` 上记数，**完全没有处理
``when == "teardown"``**。后果实测（夹具收尾抛异常）::

    真实 pytest:  1 passed, 1 error        （退出码 1）
    旧版旁路文件: {"passed": 1, "error": 0, "failed": 0, ..., "exitstatus": 1}

两个来源都"看起来齐全"（collected == 用例数、四类之和 == 用例数），
于是 `run_tests.py` 里那条"非零退出码来自收尾阶段 ⇒ 不计为 error"的豁免
把这个**真实缺陷**读成了通过。⇒ 旁路文件的**口径缺口**变成了新的遮蔽机制。

修法：**逐相位**（setup / call / teardown）记录结局，并按 pytest 的同一口径汇总：

* ``passed``  = ``call`` 相位通过的条数；
* ``failed``  = ``call`` 相位失败的条数；
* ``error``   = ``setup`` **或** ``teardown`` 相位失败的条数；
  （夹具收尾失败时 pytest 会**同时**计入 "1 passed" 与 "1 error"，本文件照此口径）
* ``skipped`` = 任一相位被跳过的**用例数**（按 nodeid 去重，避免重复计）。

另外记录**会话级异常**（收集失败 / internal error）与**实际报了结局的
nodeid 集合**——后者供调用方与"预期 nodeid"做**差集**核对。
只数个数是不够的：个数相等也可能是"少跑一个、多跑一个"。

用法（由 `run_tests.py` 自动加参数，不需要手工调）::

    python -m pytest <nodeids> -q -p no:cacheprovider -p _pytest_outcome_plugin

输出：环境变量 `PYTEST_OUTCOME_JSON` 指定的路径，内容形如::

    {
      "passed": 13, "failed": 0, "error": 0, "skipped": 0, "collected": 13,
      "exitstatus": 0,
      "expected":  ["tests/t.py::test_a", ...],
      "reported":  ["tests/t.py::test_a", ...],
      "per_nodeid": {"tests/t.py::test_a": {"setup": "passed", "call": "passed",
                                            "teardown": "passed"}},
      "session_errors": []
    }

顶层四个计数与 ``collected`` **保留原有形状**（旧读取方不受影响）；
新字段是**追加**的。

⚠️ 本文件**不是**测试，放在 `scripts/` 下是为了让 pytest 能 `-p` 导入它
（`-p` 按模块名导入，因此要把 `scripts/` 放进 `PYTHONPATH`）。
"""

from __future__ import annotations

import json
import os
from typing import Any

#: 逐相位结局：``nodeid -> {phase: status}``，phase ∈ {setup, call, teardown}。
_PER_NODEID: dict[str, dict[str, str]] = {}
#: 收集阶段实际收集到的 nodeid（调用方眼里的"预期"）。
_EXPECTED: list[str] = []
#: 会话级异常（收集失败 / internal error）的人话描述。
_SESSION_ERRORS: list[str] = []
_COLLECTED = 0


def _status_of(report: Any) -> str:
    """把一条 report 归一成 ``passed`` / ``failed`` / ``skipped``。

    pytest 的 report 上 ``passed``/``failed``/``skipped`` 是互斥的布尔位，
    但**不同相位**能出现的组合不同（call 不会 skipped，setup 可能 skipped）。
    这里不去猜，直接读 report 自己报的位。
    """
    if getattr(report, "skipped", False):
        return "skipped"
    if getattr(report, "failed", False):
        return "failed"
    if getattr(report, "passed", False):
        return "passed"
    return "unknown"


def pytest_collection_finish(session: Any) -> None:
    """收集结束：记下**预期要跑哪些**（nodeid 列表，不只是个数）。"""
    global _COLLECTED, _EXPECTED
    try:
        _EXPECTED = [str(getattr(item, "nodeid", "")) for item in session.items]
        _COLLECTED = len(_EXPECTED)
    except Exception as exc:  # pragma: no cover - 极端情况下不让插件本身炸掉整轮
        _COLLECTED = -1
        _SESSION_ERRORS.append(
            f"collection_finish 读取 items 失败：{type(exc).__name__}: {exc}"
        )


def pytest_runtest_logreport(report: Any) -> None:
    """逐相位记录 —— **三个相位都要记**（旧版漏了 teardown）。"""
    when = getattr(report, "when", "")
    nodeid = str(getattr(report, "nodeid", ""))
    if not when or not nodeid:
        return
    _PER_NODEID.setdefault(nodeid, {})[when] = _status_of(report)


def pytest_collectreport(report: Any) -> None:
    """收集阶段的失败 —— 这类失败**不会**出现在 runtest 的 report 里，
    只靠数 passed/failed 看不见（它表现为"少了一个 nodeid"）。"""
    if getattr(report, "failed", False):
        nodeid = str(getattr(report, "nodeid", "")) or "<collect>"
        _SESSION_ERRORS.append(f"收集失败：{nodeid}")


def pytest_internalerror(excrepr: Any, excinfo: Any) -> None:
    """pytest 自身的内部错误（不是被测代码的失败）。"""
    _SESSION_ERRORS.append(f"pytest internal error：{str(excrepr)[:400]}")


def _aggregate() -> dict[str, int]:
    """按 pytest 的**同一口径**从逐相位表汇总（映射见模块 docstring）。"""
    passed = failed = error = 0
    skipped_nodes: set[str] = set()
    for nodeid, phases in _PER_NODEID.items():
        if phases.get("call") == "passed":
            passed += 1
        elif phases.get("call") == "failed":
            failed += 1
        if phases.get("setup") == "failed" or phases.get("teardown") == "failed":
            error += 1
        if "skipped" in phases.values():
            skipped_nodes.add(nodeid)
    return {
        "passed": passed,
        "failed": failed,
        "error": error,
        "skipped": len(skipped_nodes),
    }


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    """会话结束：**立刻落盘**。这是关键——收尾清理再崩也不影响已写出的文件。"""
    path = os.environ.get("PYTEST_OUTCOME_JSON")
    if not path:
        return
    payload: dict[str, Any] = dict(_aggregate())
    payload["collected"] = _COLLECTED
    payload["exitstatus"] = int(getattr(exitstatus, "value", exitstatus) or 0)
    # 机器可读的两个集合：调用方拿 `reported` 与自己的 `todo` 做**差集**。
    payload["expected"] = sorted(_EXPECTED)
    payload["reported"] = sorted(_PER_NODEID)
    payload["per_nodeid"] = {k: dict(v) for k, v in sorted(_PER_NODEID.items())}
    payload["session_errors"] = list(_SESSION_ERRORS)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=False, sort_keys=True)
            fh.write("\n")
    except OSError:
        pass  # 落盘失败不该把整轮拖垮；调用方按"文件不存在"处理
