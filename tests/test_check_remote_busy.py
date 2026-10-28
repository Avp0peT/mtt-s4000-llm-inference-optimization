"""`scripts/check_remote_busy.py` 的守卫测试（纯离线，不打网）。

这个脚本是**批次起跑的前置闸门**：它说 IDLE 才允许起跑。闸门判错的后果是
"整批数据无效"，且**不报错**——所以判据本身必须有测试。

覆盖三类：
1. 判据正确性：`_PROBE` 的 argv 过滤逻辑（用合成 /proc 内容驱动）；
2. 传输安全性：base64 投递不得与 shell 引号/转义交互（已踩过这个坑）；
3. 三态语义：`None`（没查到）与 `[]`（确实没有）必须可分。
"""

from __future__ import annotations

import base64
import importlib.util
import re
import shlex
import textwrap
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_remote_busy", _ROOT / "scripts" / "check_remote_busy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load()


class TestProbeDetectionLogic(unittest.TestCase):
    """判据：argv[0] 必须是 python 解释器，且 argv 其余段含目标脚本名。"""

    def _run_probe_logic(self, entries: dict[str, list[bytes]]) -> list[dict]:
        """把 `_PROBE` 的过滤逻辑抽出来，用**合成 cmdline** 驱动。

        不连远端：只验证"哪些 argv 会被判为命中"，这正是出错会静默的地方。
        """
        hits = []
        for pid, argv_bytes in entries.items():
            argv = [a.decode("utf-8", "replace") for a in argv_bytes if a]
            if not argv:
                continue
            exe = Path(argv[0]).name
            if not exe.startswith("python"):
                continue
            if not any(m.TASK_SCRIPT in a for a in argv[1:]):
                continue
            hits.append({"pid": int(pid), "argv": argv[:6]})
        return hits

    def _b(self, *parts: str) -> list[bytes]:
        return [p.encode() for p in parts] + [b""]

    def test_real_task_is_detected(self):
        """真在跑的任务：python3 -u scripts/run_step3_matrix.py ... ⇒ 命中。"""
        entries = {
            "1234": self._b(
                "/root/miniconda3/bin/python3", "-u",
                "scripts/run_step3_matrix.py", "--cells", "fixed_cost_scan_c4_m4",
            ),
        }
        hits = self._run_probe_logic(entries)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["pid"], 1234)

    def test_bash_wrapper_is_not_detected(self):
        """★ 这是 pgrep -f 陷阱的核心：承载探测命令的 bash -c 必须被排除。

        它的 argv 里**确实含** run_step3_matrix（因为那是被传的参数文本），
        但 argv[0] 是 bash ⇒ 不命中。旧判据会在这里假报 BUSY。
        """
        entries = {
            "2001": self._b(
                "bash", "-c",
                "cd /root/ascend_probe_d1 && python3 -c \"...run_step3_matrix...\"",
            ),
            "2002": self._b("/bin/sh", "-c", "pgrep -f run_step3_matrix"),
        }
        hits = self._run_probe_logic(entries)
        self.assertEqual(hits, [], "bash/sh 外壳被误判为任务进程 ⇒ 假 BUSY")

    def test_unrelated_python_is_not_detected(self):
        """argv[0] 是 python 但跑的是别的脚本 ⇒ 不命中（防过度命中）。"""
        entries = {
            "3001": self._b("/usr/bin/python3", "scripts/collect_metrics.py"),
            "3002": self._b("/usr/bin/python3", "-m", "pip", "list"),
        }
        self.assertEqual(self._run_probe_logic(entries), [])

    def test_probe_text_itself_in_args_does_not_selfmatch(self):
        """探测命令里的 pattern 只出现在**参数**里 ⇒ argv[0] 不是 python ⇒ 排除。"""
        entries = {
            "4001": self._b(
                "/bin/bash", "-c",
                f"python3 -c \"import ...; task='{m.TASK_SCRIPT}'\"",
            ),
        }
        self.assertEqual(self._run_probe_logic(entries), [])

    def test_empty_cmdline_skipped(self):
        """内核线程无 cmdline（空 bytes）⇒ 跳过，不得抛异常。"""
        entries = {"5001": [b""], "5002": []}
        self.assertEqual(self._run_probe_logic(entries), [])

    def test_two_real_tasks_both_detected(self):
        entries = {
            "6001": self._b("/usr/bin/python3", "-u", "scripts/run_step3_matrix.py"),
            "6002": self._b("python3", "scripts/run_step3_matrix.py", "--tag", "S14S"),
        }
        self.assertEqual(len(self._run_probe_logic(entries)), 2)


class TestProbeTransport(unittest.TestCase):
    """传输：base64 投递（旧写法 `python3 -c "<src>"` 已在真机上静默失败过一次）。"""

    def test_b64_payload_has_no_shell_meta(self):
        blob = base64.b64encode(m._PROBE.encode("utf-8")).decode("ascii")
        meta = set('"\'$`\\\n\r!&|;<>()')
        self.assertFalse(meta & set(blob), f"base64 产物含 shell 元字符：{meta & set(blob)}")
        self.assertRegex(blob, r"^[A-Za-z0-9+/=]+$")

    def test_b64_roundtrip_is_byte_exact(self):
        blob = base64.b64encode(m._PROBE.encode("utf-8")).decode("ascii")
        self.assertEqual(base64.b64decode(blob).decode("utf-8"), m._PROBE)

    def test_command_shape(self):
        cmd = m._b64_py("print('x')", py="python3")
        self.assertTrue(cmd.startswith("echo "), cmd)
        self.assertIn("| base64 -d | python3 -", cmd)
        # 绝不能再出现把源码塞进 shell 双引号的形式
        self.assertNotIn('-c "', cmd)

    def test_no_word_splitting_on_payload(self):
        """b64 段必须是一个词——含空格会被 shell 拆成多个参数。"""
        cmd = m._b64_py(m._PROBE, py="python3")
        payload = cmd.split("echo ", 1)[1].split(" | ", 1)[0]
        self.assertNotIn(" ", payload)
        argv = shlex.split(cmd)
        self.assertEqual(argv[0], "echo")
        self.assertEqual(
            argv[1], base64.b64encode(m._PROBE.encode("utf-8")).decode("ascii")
        )

    def test_probe_source_is_valid_python(self):
        """`_PROBE` 本身必须能编译——否则远端只会报 SyntaxError。"""
        self.assertTrue(m._PROBE.strip(), "_PROBE 为空（空输入会伪装成通过）")
        compile(m._PROBE, "<probe>", "exec")
        self.assertIn("PROBE_JSON", m._PROBE)
        self.assertIn("/proc", m._PROBE)


class TestTriStateSemantics(unittest.TestCase):
    """`None` = 没查到；`[]` = 确实没有。二者绝不可混。"""

    def test_probe_busy_returns_none_on_link_failure(self):
        orig = m._remote

        def fake(args, timeout=120.0):
            return 1, "连接失败：boom"

        m._remote = fake
        try:
            self.assertIsNone(m.probe_busy("/root/x", tries=1))
        finally:
            m._remote = orig
        # 失败必须留痕（诊断用），且留的是原始输出
        self.assertIn("boom", globals()["m"].__dict__.get("_LAST_PROBE_ERROR", "") or "")

    def test_probe_busy_returns_empty_list_when_clean(self):
        orig = m._remote

        def fake(args, timeout=120.0):
            return 0, "PROBE_JSON []"

        m._remote = fake
        try:
            self.assertEqual(m.probe_busy("/root/x", tries=1), [])
        finally:
            m._remote = orig

    def test_main_exit_codes(self):
        """UNKNOWN ⇒ 2（**不得**退出码 0，否则闸门形同虚设）。"""
        orig = m.probe_busy
        m.probe_busy = lambda root, tries=6: None
        try:
            self.assertEqual(m.main(["--expect-idle"]), 2)
        finally:
            m.probe_busy = orig

        m.probe_busy = lambda root, tries=6: []
        try:
            self.assertEqual(m.main(["--expect-idle"]), 0)
        finally:
            m.probe_busy = orig

        m.probe_busy = lambda root, tries=6: [{"pid": 1, "argv": ["python3", "x"]}]
        try:
            self.assertEqual(m.main(["--expect-idle"]), 1)
        finally:
            m.probe_busy = orig


class TestLocalMode(unittest.TestCase):
    """★ `--local` 的存在理由：本脚本被**两种**场景调用，当前主机不同。

    * 本机编排方（默认）—— 当前主机是 Windows，批次在**远端** ⇒ 必须 SSH；
    * 远端脚本（`run_s14_d1.sh` 等，`--local`）—— 当前主机**就是**跑批次那台 ⇒ 不能 SSH。

    踩过（2026-09-27）：把默认模式接进远端脚本 ⇒ 远端自连去 `remote_run.py`，
    而远端没有 `REMOTE_*` 凭据 ⇒ 探测 UNKNOWN(2) ⇒ 前置闸门**永远拒绝**，
    第二条腿永远起不来（症状是"批次静默只跑了一半"）。实测 rc=2。
    """

    def test_local_does_not_touch_remote(self):
        """`--local` 绝不能调用 `_remote`（那会去 SSH）。"""
        called = []
        orig_remote, orig_probe = m._remote, m.probe_busy

        def boom(*a, **k):
            called.append(a)
            raise AssertionError("--local 不得走 SSH 路径")

        m._remote = boom
        m.probe_busy = lambda root, tries=6: boom()
        try:
            m.main(["--local", "--expect-idle"])
        finally:
            m._remote, m.probe_busy = orig_remote, orig_probe
        self.assertEqual(called, [], "--local 触发了 SSH ⇒ 远端自连必然失败")

    def test_local_runs_the_shared_probe_source(self):
        """`--local` 必须执行**同一份** `_PROBE`（判据面不得分叉）。"""
        src = (_ROOT / "scripts" / "check_remote_busy.py").read_text(encoding="utf-8")
        self.assertIn("_run_probe_locally", src)
        # 本地路径必须直接 exec 同一个 _PROBE 常量
        seg = src.split("def _run_probe_locally", 1)[1].split("def probe_busy", 1)[0]
        self.assertIn("exec(_PROBE", seg,
                      "本地模式没有复用 _PROBE ⇒ 判据面分叉")

    def test_local_probe_does_not_leak_probe_json_to_stdout(self):
        """★ 本地执行时不得把 `PROBE_JSON ...` 喷到 stdout。

        调用方（shell）会用 `$(python3 ... --local)` 取输出判断；
        泄漏的 PROBE_JSON 行会被当成有效输出，干扰判断。
        """
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m._run_probe_locally()
        self.assertNotIn("PROBE_JSON", buf.getvalue(),
                         "_PROBE 的 print 泄漏到了 stdout")


class TestWiringUsesLocalMode(unittest.TestCase):
    """远端脚本调用本脚本时必须带 `--local`（否则永远 UNKNOWN ⇒ 永远拒绝）。"""

    _CALLERS = ("run_s14_d1.sh", "run_s14_d1_min.sh", "remote_wait_then_d1.sh")

    def test_callers_pass_local(self):
        for name in self._CALLERS:
            p = _ROOT / "scripts" / name
            if not p.is_file():
                self.skipTest(f"{name} 不在")
            with self.subTest(caller=name):
                body = p.read_text(encoding="utf-8")
                code = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())
                if "check_remote_busy.py" not in code:
                    continue
                self.assertIn(
                    "--local", code,
                    f"{name} 在远端调用 check_remote_busy.py 却缺 --local ⇒ "
                    "远端自连缺凭据 ⇒ UNKNOWN ⇒ 闸门永远拒绝（实测 rc=2）")


class TestInterpreterSelection(unittest.TestCase):
    """必须选带 paramiko 的解释器，否则错误会伪装成"链路故障"。"""

    def test_picks_venv_when_available(self):
        py = m._pick_interpreter()
        self.assertTrue(Path(py).exists(), f"选出的解释器不存在：{py}")
        # 选出的必须真的能 import paramiko
        import subprocess
        r = subprocess.run([py, "-c", "import paramiko"], capture_output=True, timeout=30)
        self.assertEqual(r.returncode, 0, f"{py} 不能 import paramiko：{r.stderr[:200]}")

    def test_source_does_not_use_bare_sys_executable_for_remote(self):
        """回归守卫：`_remote` 里不得再写 `py = sys.executable`。

        ⚠️ 只查**赋值**（`py = sys.executable`），不查任意提及——注释里出现
        `sys.executable` 是**在解释为什么不用它**，把它算作违规会让守卫
        "因为注释而红"，从而被人当成噪声而删掉。
        """
        src = (_ROOT / "scripts" / "check_remote_busy.py").read_text(encoding="utf-8")
        body = src.split("def _remote(", 1)[1].split("def ", 1)[0]
        # 去掉注释行再判定，避免"解释性提及"被误判
        code_lines = [
            ln.split("#", 1)[0] for ln in body.splitlines()
        ]
        code = "\n".join(code_lines)
        self.assertNotIn("sys.executable", code,
                         "_remote 里又用回 sys.executable ⇒ paramiko 可能缺失")
        self.assertIn("_INTERPRETER", code, "_remote 未使用 _INTERPRETER")


class TestJudgementSurfaceCoversAllDeviceEntries(unittest.TestCase):
    """★ 判据面必须覆盖**全部会占设备的入口**，不能只认 run_step3_matrix。

    踩过（2026-09-27）：`--tag S14S` 的闸门只查 `run_step3_matrix`，而真在跑的
    是 `scripts/run_s14_d1.sh`（**内层**才调矩阵）。于是"外层已起、内层未到"
    的窗口里闸门报 IDLE ⇒ **假 IDLE**。它比假 BUSY 危险：假 BUSY 只是拦下起跑
    （看得见），假 IDLE 会**放行一个并发批次去污染 `results/raw/`**（看不见）。
    """

    def test_task_script_is_member_of_surface(self):
        """两个常量不得脱钩 —— 模块导入时的 assert 是第二道，这里是第一道。"""
        self.assertIn(m.TASK_SCRIPT, m.TASK_SCRIPTS)
        self.assertGreaterEqual(len(m.TASK_SCRIPTS), 2,
                                "判据面只剩一项 ⇒ 又回到只认单一脚本的老问题")

    def test_outer_queue_scripts_are_covered(self):
        """外层队列脚本（`run_s14_d1.sh` 等）必须命中，否则假 IDLE 会回来。"""
        for a in ("scripts/run_s14_d1.sh", "scripts/run_s14_d1_min.sh",
                  "run_s14_v5.sh", "scripts/run_s14_d1"):
            with self.subTest(arg=a):
                self.assertTrue(m._matches_task(a), f"{a} 未被判据面覆盖 ⇒ 假 IDLE 风险")

    def test_calibration_probe_is_covered(self):
        """标定链同样占设备，必须一起拦。"""
        self.assertTrue(m._matches_task("scripts/fwd_costcalib_probe.py"))

    def test_matrix_is_still_covered(self):
        """改动判据面时不得把原来的那一项弄丢。"""
        self.assertTrue(m._matches_task("scripts/run_step3_matrix.py"))
        self.assertTrue(m._matches_task("/root/x/scripts/run_step3_matrix.py"))

    def test_artifact_paths_are_not_mistaken_for_tasks(self):
        """`results/raw/S14S-*` 这类**路径**不得被判成任务（会把产物当进程）。"""
        for a in ("results/raw/S14S-short_chat-fixed_cost_scan_c2_m4",
                  "--tag", "S14S", "S14S", "results/raw", "/root/ascend_probe_d1"):
            with self.subTest(arg=a):
                self.assertFalse(m._matches_task(a), f"{a} 被误判成任务脚本")

    def test_unrelated_scripts_are_not_matched(self):
        """不能宽到把无关脚本也当成任务（否则每次都假 BUSY，闸门被当噪声删掉）。"""
        for a in ("run_benchmark.py", "scripts/diagnose_scheduling.py",
                  "scripts/emit_experiment_rows.py", "analyze_scan_batch.py"):
            with self.subTest(arg=a):
                self.assertFalse(m._matches_task(a), f"{a} 被误判成任务脚本")

    def test_probe_uses_the_shared_surface(self):
        """★ 远端探针与本机 `_matches_task` 必须**共用一份**判据面。

        本项目已三次踩过"同一件事的两份实现分叉"（臂名解析三份、并发判据两份）。
        这里的要求是：`_PROBE` 里的前缀表由 `TASK_SCRIPTS` **注入**，
        而不是在探针源码里硬写一遍。
        """
        self.assertIn("tasks = ", m._PROBE, "_PROBE 未接收注入的 tasks")
        for t in m.TASK_SCRIPTS:
            with self.subTest(task=t):
                self.assertIn(repr(t), m._PROBE, f"{t} 未出现在 _PROBE 的注入表里")

    def test_probe_matches_agrees_with_local_matcher(self):
        """本机匹配器与远端探针逻辑必须**给出一致判定**（用合成 cmdline 驱动）。"""
        def probe_hit(argv: list[str]) -> bool:
            """把 `_PROBE` 的判据②抽出来（与它的实现逐行对应）。"""
            if not argv:
                return False
            exe = Path(argv[0]).name
            if not exe.startswith("python"):
                return False
            return any(m._matches_task(a) for a in argv[1:])

        cases = {
            ("/root/miniconda3/bin/python3", "-u", "scripts/run_step3_matrix.py"): True,
            ("python3", "scripts/run_s14_d1.sh"): True,
            ("/bin/bash", "-c", "python3 -c \"task='run_step3_matrix'\""): False,
            ("python3", "scripts/diagnose_scheduling.py", "--tag", "S14S"): False,
        }
        for argv, want in cases.items():
            with self.subTest(argv=argv):
                self.assertEqual(probe_hit(list(argv)), want)


if __name__ == "__main__":
    unittest.main(verbosity=2)
