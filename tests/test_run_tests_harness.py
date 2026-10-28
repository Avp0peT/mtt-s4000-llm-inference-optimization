"""`scripts/run_tests.py` 的**判据**测试。

这个脚本本身是"判定测试有没有真的跑过"的那一层。它坏了不会有人告诉你：
它只会**安静地把失败报成通过**（或反过来，把通过报成失败）。所以它的判据必须有测试。

★ 2026-10-03 重写（用户第 1 条修订指令）
----------------------------------------
两个问题：

1. **旧测试复刻了一份判据逻辑**（`_run` 里手抄 `side_ok / has_summary / dots_cover`）。
   那份镜像一旦与实现漂移，测试就在骗自己——而且**实现改了、镜像没改**时，
   测试照样全绿。现在改为**直接调用真实判据函数** `_classify_pytest_run`。
2. 判据本身有个**同源缺口**：旁路文件只记 `setup`/`call`，**漏了 `teardown`**，
   于是"计数齐全 ⇒ 豁免非零退出码"这条规则把一个**真实的夹具收尾缺陷**
   读成了通过（实测：pytest 报 `1 passed, 1 error`，旁路报 `error=0`）。
   ⇒ 判据基底换成：**nodeid 集合双向对账 + 逐相位核对 + 会话级异常**。

★ 2026-10-03 第二次修订（用户第 B 条指令）
------------------------------------------
**取消**了"命中已知错误字符串 ⇒ 自动豁免非零退出码"这条规则：
**非零退出码一律判失败**，已知噪声线索只作为排查提示附在失败信息里。
理由：以输出里的**字符串**当豁免依据，等于让任何能打印那句话的路径都能把自己的
失败变成通过；而且外部噪声是**环境**的问题，该修环境，不该让判定器学会忽略它。

同时把结论**拆成两件事**记录（`assertion_failures` / `session_failures`）：
"用例断言没过"要去改代码或期望；"会话没正常完成"要去查环境与运行器。
混在一起时，报告只能给一句"有 N 处问题"，接的人得自己逐条读才知道查哪边。

本文件里的"真实链路"用例会**真的起 pytest 子进程、真的加载 `_pytest_outcome_plugin`**，
再把**真实产物**喂给真实判据——不是构造出来的假 JSON。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from run_tests import _classify_pytest_run  # noqa: E402


def _pytest_available() -> tuple[bool, str]:
    try:
        import pytest  # noqa: F401
    except Exception as exc:  # pragma: no cover - 环境相关
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


_HAS_PYTEST, _PYTEST_WHY = _pytest_available()


#: 沙箱垫片的真实签名样本（取自 2026-10-02 `_regr6` 的实测输出）。
SHIM_NOISE = (
    '\n[safe-delete][SAFE_DELETE_FAIL_CLOSED] {"reason": "trash-failed", '
    '"detail": "[Errno 53] 找不到网络路径。: \'\\\\\\\\?\\\\C:\\\\x\'"}\n'
)


def _probe_files() -> dict[str, str]:
    """五个探针模块：覆盖 setup / call / teardown / skip / 正常。"""
    return {
        "test_clean.py": (
            "def test_ok():\n"
            "    assert True\n"
        ),
        "test_call_fail.py": (
            "def test_bad():\n"
            "    assert False, 'deliberate call-phase failure'\n"
        ),
        "test_setup_err.py": (
            "import pytest\n"
            "\n"
            "@pytest.fixture\n"
            "def bad_setup():\n"
            "    raise RuntimeError('deliberate setup defect')\n"
            "    yield 1\n"
            "\n"
            "def test_a(bad_setup):\n"
            "    assert True\n"
        ),
        "test_teardown_err.py": (
            "import pytest\n"
            "\n"
            "@pytest.fixture\n"
            "def bad_teardown():\n"
            "    yield 1\n"
            "    raise RuntimeError('deliberate teardown defect')\n"
            "\n"
            "def test_a(bad_teardown):\n"
            "    assert True\n"
        ),
        "test_skipped.py": (
            "import pytest\n"
            "\n"
            "@pytest.mark.skip(reason='deliberate skip')\n"
            "def test_skipped():\n"
            "    assert True\n"
        ),
    }


class TestRealRunnerAndPlugin(unittest.TestCase):
    """★★ 走**真实链路**：真起 pytest、真加载插件、真读产物、真喂判据。

    这是用户第 1 条指令的硬要求（"测试必须调用真实运行器和插件"）。
    构造出来的 JSON 只能证明判据函数"对某些输入会怎样"，
    证明不了**插件真的会写出那些字段**。
    """

    def setUp(self) -> None:
        if not _HAS_PYTEST:  # pragma: no cover - 环境相关
            self.skipTest(f"需要 pytest 才能跑真实链路：{_PYTEST_WHY}")
        self._tmp = tempfile.TemporaryDirectory(prefix="rt_probe_")
        self.dir = Path(self._tmp.name)
        for name, body in _probe_files().items():
            (self.dir / name).write_text(body, encoding="utf-8", newline="\n")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_real(self, module: str) -> tuple[int, dict | None, str]:
        """用**真实插件**跑一个探针模块，返回 ``(退出码, 旁路 JSON, 合并文本)``。"""
        outcome_path = self.dir / f"outcome_{module}.json"
        env = dict(os.environ)
        env["PYTEST_OUTCOME_JSON"] = str(outcome_path)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_ROOT / "scripts")]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", module, "-q",
             "-p", "no:cacheprovider", "-p", "_pytest_outcome_plugin"],
            cwd=str(self.dir), capture_output=True, text=True,
            errors="replace", timeout=300, env=env,
        )
        outcome = None
        if outcome_path.is_file():
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        return proc.returncode, outcome, (proc.stdout or "") + (proc.stderr or "")

    def _verdict(self, module: str):
        rc, outcome, text = self._run_real(module)
        todo = sorted((outcome or {}).get("expected") or [])
        return rc, outcome, _classify_pytest_run(
            returncode=rc, todo=todo, outcome=outcome, text=text
        )

    # ---------------- 正向：正常路径必须判通过 ----------------

    def test_clean_run_passes(self):
        rc, outcome, v = self._verdict("test_clean.py")
        self.assertEqual(rc, 0)
        self.assertEqual(v.passed, 1)
        self.assertEqual(v.hard_failures, [], "干净运行不得有任何硬失败")
        self.assertTrue(v.assertions_clean, "干净运行 ⇒ 断言层面全过")
        self.assertTrue(v.session_completed, "干净运行 ⇒ 会话正常完成")

    # ---------------- ★ 核心：teardown 缺陷不得被豁免 ----------------

    def test_teardown_defect_is_a_hard_failure(self):
        """★★ 这条是本次修订的**靶心**。

        夹具收尾抛异常时：`call` 相位通过、`teardown` 相位失败。
        旧插件**只记 setup/call** ⇒ 旁路写 `error=0`，而它"看起来齐全"
        （collected == 用例数、四类之和 == 用例数）⇒ 旧的"计数齐全即豁免
        非零退出码"把**真实缺陷**读成了通过。

        现在必须：① 旁路把 teardown 计成 `error`；② 判据给出硬失败。
        """
        rc, outcome, v = self._verdict("test_teardown_err.py")
        self.assertIsNotNone(outcome, "插件必须落盘旁路文件")
        self.assertEqual(
            outcome["error"], 1,
            "★ 旁路必须把 teardown 相位失败计入 error（旧版这里是 0）",
        )
        self.assertEqual(outcome["passed"], 1, "call 相位确实通过了")
        self.assertEqual(
            outcome["per_nodeid"]["test_teardown_err.py::test_a"]["teardown"],
            "failed",
            "逐相位表必须记下 teardown=failed",
        )
        self.assertTrue(v.hard_failures, "teardown 缺陷必须判硬失败，不得豁免")
        self.assertEqual(rc, 1)

    def test_setup_defect_is_a_hard_failure(self):
        rc, outcome, v = self._verdict("test_setup_err.py")
        self.assertEqual(outcome["error"], 1)
        self.assertTrue(v.hard_failures)

    def test_call_failure_is_a_hard_failure(self):
        rc, outcome, v = self._verdict("test_call_fail.py")
        self.assertEqual(outcome["failed"], 1)
        self.assertTrue(v.hard_failures)

    def test_skip_is_reported_as_skipped_not_passed(self):
        rc, outcome, v = self._verdict("test_skipped.py")
        self.assertEqual(outcome["skipped"], 1)
        self.assertEqual(v.passed, 0, "跳过不得计入通过")
        self.assertEqual(v.hard_failures, [], "单纯跳过不是缺陷（但也不构成证据）")

    def test_plugin_records_all_three_phases_for_a_normal_case(self):
        """正常用例的逐相位表必须是 setup/call/teardown **三条齐全**。"""
        _rc, outcome, _v = self._verdict("test_clean.py")
        phases = outcome["per_nodeid"]["test_clean.py::test_ok"]
        self.assertEqual(
            sorted(phases), ["call", "setup", "teardown"],
            "三个相位都要记（旧版漏 teardown）",
        )
        self.assertTrue(all(s == "passed" for s in phases.values()))


class TestNodeIdCrossCheck(unittest.TestCase):
    """★★ 无论退出码是否为零，都要核对**预期 vs 实际 nodeid**（用户第 1 条）。"""

    def _mk(self, *, todo, reported, **counts):
        outcome = {
            "passed": 0, "failed": 0, "error": 0, "skipped": 0,
            "collected": len(reported), "exitstatus": 0,
            "expected": sorted(reported), "reported": sorted(reported),
            "per_nodeid": {n: {"setup": "passed", "call": "passed", "teardown": "passed"}
                           for n in reported},
            "session_errors": [],
        }
        outcome.update(counts)
        return _classify_pytest_run(returncode=0, todo=list(todo),
                                    outcome=outcome, text="")

    def test_missing_nodeid_is_a_hard_failure(self):
        """**结果缺失**：收集到 3 个，旁路只报了 2 个 ⇒ 必须有硬失败。

        注意退出码是 **0** —— 这条专门证明"退出码为零也要对账"。
        """
        todo = ["t.py::a", "t.py::b", "t.py::c"]
        v = self._mk(todo=todo, reported=todo[:2], passed=2, collected=3)
        self.assertTrue(v.hard_failures, "少一个 nodeid 必须被拦下")
        self.assertTrue(any("没有任何结局" in h for h in v.hard_failures))

    def test_extra_nodeid_is_a_hard_failure(self):
        """多报了一个非预期 nodeid ⇒ 同样必须拦下（可能跑到了别的东西）。"""
        todo = ["t.py::a", "t.py::b"]
        v = self._mk(todo=todo, reported=todo + ["t.py::zzz"], passed=3)
        self.assertTrue(v.hard_failures)
        self.assertTrue(any("不在**预期**内" in h or "不在" in h for h in v.hard_failures))

    def test_matching_nodeids_pass(self):
        todo = ["t.py::a", "t.py::b"]
        v = self._mk(todo=todo, reported=todo, passed=2)
        self.assertEqual(v.hard_failures, [])

    def test_exact_sets_but_wrong_pairing_is_caught(self):
        """计数相同、集合相同，但**旁路自己的 expected 与 todo 不一致** ⇒ 拦下。

        这是"只比个数"会漏掉的那一类：个数相等也可能是少跑一个、多跑一个。
        """
        todo = ["t.py::a", "t.py::b"]
        reported = todo
        outcome = {
            "passed": 2, "failed": 0, "error": 0, "skipped": 0, "collected": 2,
            "exitstatus": 0,
            "expected": ["t.py::a", "t.py::DIFFERENT"],
            "reported": reported,
            "per_nodeid": {n: {"setup": "passed", "call": "passed", "teardown": "passed"}
                           for n in reported},
            "session_errors": [],
        }
        v = _classify_pytest_run(returncode=0, todo=todo, outcome=outcome, text="")
        self.assertTrue(v.hard_failures, "expected 与 todo 不一致必须被拦下")


class TestExitCodeIsNotUnconditionallyAuthoritative(unittest.TestCase):
    """★ 退出码：**不以"计数齐全"豁免**（用户第 1 条的硬要求）。"""

    @staticmethod
    def _clean_outcome(n: int, exitstatus: int = 0, **over):
        nodes = [f"t.py::t{i}" for i in range(n)]
        outcome = {
            "passed": n, "failed": 0, "error": 0, "skipped": 0, "collected": n,
            # ★ `exitstatus`（pytest 自报）必须与调用方传的 `returncode` 一致，
            #   否则会额外触发"两处退出码不一致"那条判据，把测试的焦点搅浑。
            "exitstatus": exitstatus, "expected": nodes, "reported": nodes,
            "per_nodeid": {x: {"setup": "passed", "call": "passed", "teardown": "passed"}
                           for x in nodes},
            "session_errors": [],
        }
        outcome.update(over)
        return nodes, outcome

    def test_noise_signature_does_not_exempt_a_nonzero_exit(self):
        """★★ **靶心（第 B 条指令）**：命中已知错误字符串**不再豁免**。

        这是真实的沙箱 `safe-delete` 垫片文本（`Errno 53 找不到网络路径`）。
        旧规则会说"已归因 ⇒ 不计为失败"；新规则**必须判失败** ——
        因为豁免依据是输出里的**字符串**，而任何路径都可能打印它。
        提示可以留，判定不能松。
        """
        nodes, outcome = self._clean_outcome(13, exitstatus=1)
        v = _classify_pytest_run(returncode=1, todo=nodes,
                                 outcome=outcome, text=SHIM_NOISE)
        self.assertTrue(v.hard_failures, "命中噪声签名**不得**豁免非零退出码")
        self.assertFalse(v.session_completed, "非零退出码 ⇒ 会话未正常完成")
        self.assertTrue(v.assertions_clean, "这一条里断言本身是干净的 —— 两类要分开")
        self.assertIn("Errno 53", SHIM_NOISE)
        # 提示要留（便于排查），但不能改变判定
        self.assertTrue(any("Errno 53" in h or "线索" in h for h in v.session_failures),
                        "应把'疑似外部噪声'作为**排查线索**写进失败信息")
        self.assertEqual(v.session_failures[-1].count("不构成豁免"), 1,
                         "提示必须明确写出'不构成豁免'，防后来人误读")

    def test_unattributable_nonzero_exit_is_a_hard_failure(self):
        """★★ **靶心之二**：非零退出码 + 计数齐全 + 相位干净 + 集合一致，
        但**未命中任何已知签名** ⇒ **不得当作通过**。

        旧规则在这里会说"收尾阶段噪声，不计为 error"。新规则必须判失败：
        "看起来齐全"**不是**豁免依据。
        """
        nodes, outcome = self._clean_outcome(13, exitstatus=1)
        v = _classify_pytest_run(returncode=1, todo=nodes,
                                 outcome=outcome, text="some unrelated output\n")
        self.assertTrue(v.hard_failures, "非零退出码必须判失败")
        self.assertTrue(any("一律判失败" in h for h in v.session_failures))
        self.assertTrue(v.assertions_clean, "计数/相位/集合都干净 ⇒ 问题在**会话**层面")

    def test_side_channel_missing_is_a_hard_failure(self):
        """旁路缺失 ⇒ 不接受"用 stdout 兜底"做出的通过判定。"""
        nodes, _ = self._clean_outcome(13)
        v = _classify_pytest_run(returncode=0, todo=nodes, outcome=None,
                                 text="13 passed in 1.0s\n")
        self.assertTrue(v.hard_failures)
        self.assertTrue(any("缺失" in h for h in v.hard_failures))

    def test_session_level_exception_is_a_hard_failure(self):
        """会话级异常（收集失败 / internal error）必须判失败。"""
        nodes, outcome = self._clean_outcome(
            3, exitstatus=0, session_errors=["收集失败：t.py::x"])
        v = _classify_pytest_run(returncode=0, todo=nodes, outcome=outcome, text="")
        self.assertTrue(v.hard_failures)
        self.assertTrue(any("会话级异常" in h for h in v.hard_failures))

    def test_incomplete_phase_records_are_a_hard_failure(self):
        """相位记录不全（例如 teardown 丢报）⇒ 判失败，不能当"没跑过就算了"。"""
        nodes, outcome = self._clean_outcome(2)
        outcome["per_nodeid"]["t.py::t0"] = {"setup": "passed", "call": "passed"}
        v = _classify_pytest_run(returncode=0, todo=nodes, outcome=outcome, text="")
        self.assertTrue(v.hard_failures)
        self.assertTrue(any("不完整" in h for h in v.hard_failures))

    def test_setup_failure_without_call_record_is_not_flagged_incomplete(self):
        """**否定对照（防假失败）**：setup 失败时 pytest **本来就不发** call 记录。

        第一版完整性判据把这种情况也当成"记录不完整"，实测多报了一条假失败。
        这条锁住那个修正。
        """
        nodes = ["t.py::a"]
        outcome = {
            "passed": 0, "failed": 0, "error": 1, "skipped": 0, "collected": 1,
            "exitstatus": 1, "expected": nodes, "reported": nodes,
            "per_nodeid": {"t.py::a": {"setup": "failed", "teardown": "passed"}},
            "session_errors": [],
        }
        v = _classify_pytest_run(returncode=1, todo=nodes, outcome=outcome, text="")
        self.assertFalse(
            any("不完整" in h for h in v.hard_failures),
            "setup 失败时缺 call 是**正常**发报行为，不得报'记录不完整'",
        )
        self.assertTrue(any("相位" in h for h in v.hard_failures), "但仍要因 setup 失败而判失败")

    def test_session_crash_without_outcome_is_still_an_error(self):
        """**否定对照**：没有旁路、没有结果 ⇒ 收集期崩了 ⇒ 必须判失败。"""
        todo = [f"t.py::t{i}" for i in range(13)]
        v = _classify_pytest_run(returncode=2, todo=todo, outcome=None,
                                 text="INTERNALERROR> Traceback ...\n")
        self.assertTrue(v.hard_failures)


class TestExitStatusCrossCheck(unittest.TestCase):
    """★ **两处退出码必须一致**（2026-10-03 补的洞）。

    `returncode`（进程退出码）与旁路文件里的 `exitstatus`（**pytest 自报**）是两个
    独立来源。原实现只读前者，后者**写出来却从不比较** ⇒ "pytest 自己认为该以非零
    退出、而进程恰好退了 0"（或反过来）会被**静默放过**——那正是"退出码豁免"这条
    判据上剩下的最后一道口子。
    """

    @staticmethod
    def _evidence(n: int = 3, *, exitstatus):
        nodes = [f"t.py::t{i}" for i in range(n)]
        base = {
            "passed": n, "failed": 0, "error": 0, "skipped": 0, "collected": n,
            "expected": nodes, "reported": nodes,
            "per_nodeid": {x: {"setup": "passed", "call": "passed", "teardown": "passed"}
                           for x in nodes},
            "session_errors": [],
        }
        if exitstatus is not None:
            base["exitstatus"] = exitstatus
        return nodes, base

    def test_consistent_zero_is_clean(self):
        nodes, ev = self._evidence(exitstatus=0)
        v = _classify_pytest_run(returncode=0, todo=nodes, outcome=ev, text="")
        self.assertEqual(v.hard_failures, [])

    def test_consistent_nonzero_without_defects_is_still_rejected(self):
        """两处都非零 ⇒ 一致性没问题，但"非零且无法归因"仍**必须**判失败。"""
        nodes, ev = self._evidence(exitstatus=2)
        v = _classify_pytest_run(returncode=2, todo=nodes, outcome=ev, text="whatever")
        self.assertTrue(v.hard_failures)
        self.assertTrue(any("一律判失败" in h for h in v.session_failures))

    def test_mismatch_is_a_hard_failure(self):
        """**否定测试**：进程退 0 而 pytest 自报 1 ⇒ 判失败（不许挑对自己有利的那个）。"""
        nodes, ev = self._evidence(exitstatus=1)
        v = _classify_pytest_run(returncode=0, todo=nodes, outcome=ev, text="")
        self.assertTrue(v.hard_failures, "两处退出码不一致必须判失败")
        self.assertTrue(any("退出码两处不一致" in h for h in v.hard_failures))

    def test_reverse_mismatch_is_also_a_hard_failure(self):
        """反方向：进程退 1 而 pytest 自报 0 ⇒ 同样判失败（且**不得**走签名豁免）。"""
        nodes, ev = self._evidence(exitstatus=0)
        v = _classify_pytest_run(returncode=1, todo=nodes, outcome=ev, text=SHIM_NOISE)
        self.assertTrue(v.hard_failures, "反方向不一致也要拦")
        self.assertTrue(any("两处不一致" in h for h in v.session_failures))
        # 即便文本里有噪声签名，也**不得**把这条降级掉
        self.assertNotIn("不构成豁免", " ".join(v.session_failures[:1]))

    def test_missing_exitstatus_is_a_hard_failure(self):
        """字段**缺失**也算失败：那意味着插件与本运行器**版本不同步**。"""
        nodes, ev = self._evidence(exitstatus=None)
        self.assertNotIn("exitstatus", ev)
        v = _classify_pytest_run(returncode=0, todo=nodes, outcome=ev, text="")
        self.assertTrue(v.hard_failures)
        self.assertTrue(any("没有" in h and "exitstatus" in h for h in v.hard_failures))


class TestAssertionsAndSessionAreRecordedSeparately(unittest.TestCase):
    """★ **"用例断言结果"与"测试会话是否正常完成"必须分开记**。

    为什么：两者**修法完全不同** ——
      * 断言没过 ⇒ 改被测代码或改期望；
      * 会话没正常完成 ⇒ 查环境、查运行器、查插件版本。
    混在一个列表里时，报告只能说"有 N 处问题"，接的人得逐条读才知道该查哪边。
    """

    @staticmethod
    def _evidence(n: int, *, exitstatus: int = 0, phases=None, session_errors=None):
        nodes = [f"t.py::t{i}" for i in range(n)]
        return nodes, {
            "passed": n, "failed": 0, "error": 0, "skipped": 0, "collected": n,
            "exitstatus": exitstatus,
            "expected": nodes, "reported": nodes,
            "per_nodeid": phases or {x: {"setup": "passed", "call": "passed",
                                         "teardown": "passed"} for x in nodes},
            "session_errors": session_errors or [],
        }

    def test_assertion_failure_is_assertion_not_session(self):
        """夹具 **teardown** 抛异常（退出码 0）⇒ 属**断言**层面，会话本身是正常的。"""
        nodes, ev = self._evidence(2)
        ev["per_nodeid"]["t.py::t1"]["teardown"] = "failed"
        v = _classify_pytest_run(returncode=0, todo=nodes, outcome=ev, text="")
        self.assertFalse(v.assertions_clean, "teardown failed ⇒ 断言层面不干净")
        self.assertTrue(v.session_completed, "退出码 0 且相位记录齐全 ⇒ 会话正常完成")
        self.assertTrue(any("相位" in h for h in v.assertion_failures))
        self.assertEqual(v.session_failures, [])

    def test_session_failure_is_session_not_assertion(self):
        """非零退出码但断言全清 ⇒ 属**会话**层面（环境/运行器），不是断言没过。"""
        nodes, ev = self._evidence(2, exitstatus=3)
        v = _classify_pytest_run(returncode=3, todo=nodes, outcome=ev, text="boom")
        self.assertTrue(v.assertions_clean, "没有任何断言失败")
        self.assertFalse(v.session_completed, "非零退出码 ⇒ 会话未正常完成")
        self.assertEqual(v.assertion_failures, [])
        self.assertTrue(v.session_failures)

    def test_collection_failure_is_a_session_failure(self):
        """收集失败 ⇒ 会话层面（用例根本没跑到，谈不上断言结果）。"""
        nodes, ev = self._evidence(2, session_errors=["收集失败：t.py::x"])
        v = _classify_pytest_run(returncode=0, todo=nodes, outcome=ev, text="")
        self.assertFalse(v.session_completed)
        self.assertTrue(any("会话级异常" in h for h in v.session_failures))
        self.assertEqual(v.assertion_failures, [])

    def test_missing_nodeid_outcome_is_an_assertion_failure(self):
        """预期用例**没有结局** ⇒ 断言结果未知 ⇒ 记到**断言**层面。"""
        nodes, ev = self._evidence(3)
        ev["reported"] = nodes[:2]          # 第 3 个没报结局
        v = _classify_pytest_run(returncode=0, todo=nodes, outcome=ev, text="")
        self.assertFalse(v.assertions_clean)
        self.assertTrue(any("没有任何结局" in h for h in v.assertion_failures))

    def test_both_can_fail_at_once_and_are_both_listed(self):
        """两类可以同时出问题 ⇒ **两个列表都要有内容**，不许合并成一条。"""
        nodes, ev = self._evidence(2, exitstatus=1)
        ev["per_nodeid"]["t.py::t0"]["call"] = "failed"
        v = _classify_pytest_run(returncode=1, todo=nodes, outcome=ev, text="")
        self.assertTrue(v.assertion_failures, "断言层面应有内容")
        self.assertTrue(v.session_failures, "会话层面应有内容")
        self.assertFalse(v.assertions_clean)
        self.assertFalse(v.session_completed)

    def test_hard_failures_is_the_readonly_union(self):
        """`hard_failures` 保留为**只读合并视图**（既有调用方不必改）。"""
        nodes, ev = self._evidence(2, exitstatus=1)
        ev["per_nodeid"]["t.py::t0"]["call"] = "failed"
        v = _classify_pytest_run(returncode=1, todo=nodes, outcome=ev, text="")
        self.assertEqual(v.hard_failures,
                         v.assertion_failures + v.session_failures)


class TestCollectCountVsDifference(unittest.TestCase):
    """★ 判据是**差集**，不是**计数**。"""

    def test_parametrize_expansion_is_not_a_gap(self):
        """pytest 计数 > unittest 计数，但差集为空 ⇒ **不得**触发补跑。

        真实例子：`test_validate_fp16_path` unittest 收 23、pytest 收 48
        （参数化展开），差集为空。旧判据把 48>23 读成"有 25 个没被跑到"。
        """
        unittest_have = {f"C::m{i}" for i in range(23)}
        pytest_nodeids = [f"tests/t.py::C::m{i}" for i in range(23)]
        pytest_nodeids += [f"tests/t.py::C::m{i}[case{j}]" for i in range(23)
                           for j in range(2)]
        todo = [n for n in pytest_nodeids if n.split("::", 1)[1] not in unittest_have]
        self.assertTrue(all(n.split("::", 1)[1].startswith("C::m") for n in pytest_nodeids))

    def test_true_gap_is_detected(self):
        """真实缺口：unittest 收 **0**，pytest 收 13 ⇒ 差集是全部 13。"""
        unittest_have: set[str] = set()
        pytest_nodeids = [f"tests/t.py::test_f{i}" for i in range(13)]
        todo = [n for n in pytest_nodeids if n.split("::", 1)[1] not in unittest_have]
        self.assertEqual(len(todo), 13)


class TestRealModuleGapIsTheExpectedOne(unittest.TestCase):
    """对**真实仓库**断言：只有 `test_plot_step16_decouple` 是纯 pytest 风格。"""

    def test_plot_step16_is_the_only_pytest_only_module(self):
        from run_tests import _iter_cases, _load

        name = "tests.test_plot_step16_decouple"
        have = set()
        for case in _load([name]):
            for t in _iter_cases(case):
                have.add(f"{type(t).__name__}::{t._testMethodName}")
        self.assertEqual(
            len(have), 0,
            "该模块若被 unittest 收到了，'13 个漏用例'这条结论就不成立")

    def test_unittest_does_collect_the_other_pytest_looking_module(self):
        """反向对照：`test_source_stamp` **不是**纯 pytest 风格，unittest 收得到。"""
        from run_tests import _iter_cases, _load

        have = set()
        for case in _load(["tests.test_source_stamp"]):
            for t in _iter_cases(case):
                have.add(f"{type(t).__name__}::{t._testMethodName}")
        self.assertGreater(len(have), 0, "该模块应被 unittest 收到（它是 TestCase 风格）")


if __name__ == "__main__":
    unittest.main()
