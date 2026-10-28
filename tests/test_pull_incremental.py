"""`scripts/pull_incremental.py` 的守卫测试（离线，不打网）。

为什么需要它
------------
这个脚本是**每批产物的入口**：远端才丢得起（系统盘会回滚快照），本地是唯一持久副本。
它一旦"静默地说没有产物"，整批证据就停在会上消失的地方。

2026-09-27 实测踩到三个**同源**缺陷，全部是"把失败读成正常"：

1. **远端根目录过时**：默认值是迁移前的 `/root/ascend_probe/results/raw`，
   而 D1 平台一切在 `/root/ascend_probe_d1` ⇒ 枚举恒为空 ⇒ 报
   "远端没有产物（或远端不可达）" ⇒ 被读成"批次还没跑"。
   ★ 这是本项目最忌讳的形状：**一个配置漂移伪装成一次真实的状态观测**。
2. **枚举失败与"确实为空"混成一个空**：链路抖动/权限/解释器错都会得到 `[]`，
   与"这个 tag 真的没产物"不可分。
3. **逻辑不一致**：枚举用 `--python`，而 `_pull` 用 `sys.executable` ——
   同一个脚本里两套解释器选择，且 `sys.executable` 常常没有 paramiko。

本文件把这三条都钉住，并且**先证明守卫会失败**（test_stale_root_is_detected
用旧根目录值断言"它不会等于当前根"），再证明当前值是好的。
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "pull_incremental.py"


def _load():
    spec = importlib.util.spec_from_file_location("_pull_inc_under_test", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load()


class TestRemoteRootIsCurrent(unittest.TestCase):
    """① 远端根必须指向**当前平台**（D1），不得留迁移前的旧值。"""

    def test_default_root_points_at_d1(self):
        self.assertEqual(m.DEFAULT_REMOTE_ROOT, "/root/ascend_probe_d1/results/raw")

    def test_stale_root_is_detected(self):
        """★ 反向对照：证明这条守卫**能失败**。

        旧值（`/root/ascend_probe`，无 `_d1`）正是把"批次还没跑"这个假象
        送出来的那个值。这里断言"当前值不等于旧值"——
        若有人把默认值改回去，这条会红。
        """
        stale = "/root/ascend_probe/results/raw"
        self.assertNotEqual(m.DEFAULT_REMOTE_ROOT, stale,
                            "远端根又变回迁移前的值 ⇒ 枚举会恒为空而报'没有产物'")
        # 与仓库内其它脚本用的根保持一致（单一事实来源）
        self.assertIn("ascend_probe_d1", m.DEFAULT_REMOTE_ROOT)

    def test_argparse_default_matches_constant(self):
        """argparse 的默认值必须**引用**常量，不得各写一份字面量。

        ⚠️ 只查**代码行**，不查注释/文档串：常量旁边的注释会提到"旧值是多少"
        （那是在解释为什么不用它），把它算作违规会让守卫"因为注释而红"。
        """
        src = _SCRIPT.read_text(encoding="utf-8")
        self.assertIn("default=DEFAULT_REMOTE_ROOT", src,
                      "--remote-root 又写回字面量 ⇒ 两处会漂移")
        # 剥掉注释行与三引号文档串后再查旧根字面量
        code_only = re.sub(r'"""[\s\S]*?"""', "", src)
        code_only = "\n".join(ln.split("#", 1)[0] for ln in code_only.splitlines())
        self.assertNotIn("/root/ascend_probe/results/raw", code_only,
                         "代码里仍残留迁移前的远端根字面量")


class TestTriStateEnumeration(unittest.TestCase):
    """② 枚举必须三态：`None`(没查到) vs `[]`(确实没有) 可分。"""

    def test_signature_returns_optional_and_raw(self):
        sig = inspect.signature(m._remote_dirs)
        self.assertEqual(len(sig.parameters), 3)
        ann = str(sig.return_annotation)
        self.assertIn("None", ann, "返回注解必须体现'可为 None'（三态）")

    def test_none_on_link_failure(self):
        """链路失败 ⇒ None（**不是** []）。"""
        orig = m.subprocess.run

        class R:
            returncode = 1
            stdout = ""
            stderr = "连接失败：SSHException: boom"

        m.subprocess.run = lambda *a, **k: R()
        try:
            got, raw = m._remote_dirs("/root/x", "S14S", sys.executable)
        finally:
            m.subprocess.run = orig
        self.assertIsNone(got, "链路失败被读成空列表 ⇒ '没查到'会伪装成'确实没有'")
        self.assertIn("boom", raw)

    def test_match_markers_cover_paramiko_missing(self):
        """★ 用错解释器的症状（paramiko 缺失）必须被认作失败。"""
        class R:
            returncode = 1
            stdout = ""
            stderr = "连接失败：ModuleNotFoundError: No module named 'paramiko'"

        orig = m.subprocess.run
        m.subprocess.run = lambda *a, **k: R()
        try:
            got, _ = m._remote_dirs("/root/x", "S14S", sys.executable)
        finally:
            m.subprocess.run = orig
        self.assertIsNone(got, "缺 paramiko 未被识别为失败 ⇒ 会误报'没有产物'")

    def test_empty_list_when_genuinely_empty(self):
        """命令成功执行且真没有 ⇒ []（与 None 必须不同）。"""
        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        orig = m.subprocess.run
        m.subprocess.run = lambda *a, **k: R()
        try:
            got, _ = m._remote_dirs("/root/x", "S14S", sys.executable)
        finally:
            m.subprocess.run = orig
        self.assertEqual(got, [], "'确实没有'必须是 []，不能也返回 None")

    def test_parses_names_on_success(self):
        class R:
            returncode = 0
            stdout = "S14S-a\nS14S-b\n\n"
            stderr = ""

        orig = m.subprocess.run
        m.subprocess.run = lambda *a, **k: R()
        try:
            got, _ = m._remote_dirs("/root/x", "S14S", sys.executable)
        finally:
            m.subprocess.run = orig
        self.assertEqual(got, ["S14S-a", "S14S-b"])

    def test_main_exit_code_2_on_unknown(self):
        """UNKNOWN ⇒ 退出码 2；且**不得**给出"确实没有产物"的肯定判断。

        ⚠️ 判据要落在**肯定句**上（如 `下没有 … 的产物`），不能只查"没有"二字——
        告警文案本身就写着"**不得**当作'没有产物'"，查二字会因这条告警而红。
        """
        orig = m._remote_dirs
        m._remote_dirs = lambda *a, **k: (None, "连接失败：boom")
        import io
        from contextlib import redirect_stderr, redirect_stdout
        so, se = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(so), redirect_stderr(se):
                rc = m.main(["--tag", "S14S"])
        finally:
            m._remote_dirs = orig
        both = so.getvalue() + se.getvalue()
        self.assertEqual(rc, 2)
        self.assertIn("UNKNOWN", se.getvalue())
        # 不得出现"确认没有"的句式（那是把失败读成状态观测）
        self.assertNotIn("下没有", both, "UNKNOWN 时给出了'确实没有'的肯定判断")
        self.assertNotIn("确实是空的", both)


class TestInterpreterSelection(unittest.TestCase):
    """③ 解释器必须选带 paramiko 的那个，且**只有一处**选择逻辑。"""

    def test_no_bare_sys_executable_in_remote_calls(self):
        """`_pull` 不得再用 `sys.executable` 直接驱动远端。"""
        src = _SCRIPT.read_text(encoding="utf-8")
        body = src.split("def _pull(", 1)[1].split("\ndef ", 1)[0]
        code = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())
        self.assertNotIn("sys.executable", code,
                         "_pull 又用回 sys.executable ⇒ 可能没有 paramiko")

    def test_argparse_default_is_none_so_autopick_applies(self):
        src = _SCRIPT.read_text(encoding="utf-8")
        self.assertIn('ap.add_argument("--python", default=None', src,
                      "--python 默认值必须是 None，否则自动选解释器不会生效")

    def test_pick_interpreter_returns_existing_path(self):
        py = m._pick_interpreter()
        self.assertTrue(Path(py).exists(), f"选出的解释器不存在：{py}")

    def test_pick_interpreter_prefers_paramiko_capable(self):
        """若 venv 在，必须选到能 import paramiko 的那个。"""
        if not m._VENV_PY.exists():
            self.skipTest("本机没有托管 venv")
        py = m._pick_interpreter()
        import subprocess
        r = subprocess.run([py, "-c", "import paramiko"], capture_output=True, timeout=30)
        self.assertEqual(r.returncode, 0,
                         f"选出的解释器不能 import paramiko：{py}")


class TestRequiredFilesGuard(unittest.TestCase):
    """④ `_verify` 必须把"缺文件"和"文件在但内容不合格"都报出来。"""

    def test_missing_file_detected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            bad = m._verify(d)
            self.assertEqual(len(bad), len(m._REQUIRED))

    def test_unparsable_meta_detected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for f in m._REQUIRED:
                (d / f).write_text("{}" if f.endswith(".json") else "", encoding="utf-8")
            (d / "meta.json").write_text("{not json", encoding="utf-8")
            bad = m._verify(d)
            self.assertTrue(any("不可解析" in b for b in bad), bad)

    def test_blank_data_label_detected(self):
        import json as _json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for f in m._REQUIRED:
                (d / f).write_text("", encoding="utf-8")
            (d / "meta.json").write_text(
                _json.dumps({"data_label": "", "summary": {"n_requests": 8}}),
                encoding="utf-8")
            bad = m._verify(d)
            self.assertTrue(any("data_label" in b for b in bad), bad)


class TestStagingTarsAreNotRuns(unittest.TestCase):
    """★ 回拉自己的暂存包**不得**被当成 run。

    2026-09-28 实测踩到：`remote_run.py pull` 把暂存包建在**与 run 同级的父目录**
    （`do_pull` 行 715：`remote_tar = posixpath.join(parent, f"{arc}.tar")`），
    于是 `ls -d <root>/<tag>-*` 会把 `S14B2-...-r1.tar` 也列成一个"run"。实测远端
    证据目录里出现过 `...-r1.tar` **与**其倍乘 `...-r1.tar.tar`（一次失败回拉的残留）。

    后果（三件，全是静默的）：
      * 进"待取"清单 ⇒ 白传一遍，本地多出 `_...-r1.tar.tar` 的怪名字；
      * **永远过不了 `_verify`**（没有 `meta.json`）⇒ 完整性核对报**假失败**；
      * 把 `--expect N` 的计数抬高 ⇒ "批次还没跑完"长得像"跑完了但坏了"。

    这个用例**先证明守卫会失败**（拿混合样本断言"剔除前的集合里**有** tar"），
    再证明现在会剔除 —— 否则一个恒真的断言证明不了任何事。
    """

    def test_split_separates_staging_from_runs(self):
        m = _load()
        sample = [
            "S14B2-load_ramp-ramp_cost",
            "S14B2-load_ramp-ramp_cost-r1",
            "S14B2-load_ramp-ramp_slo_weave-r1.tar",       # 实测踩到的那个
            "S14B2-load_ramp-ramp_slo_weave-r1.tar.tar",   # 其倍乘（.tar.tar）
            "S14B2-load_ramp-ramp_static",
            "X-thing.tgz",
            "Y-TGZ.TAR",                                    # 大小写不敏感
        ]
        runs, staging = m._split_staging(sample)
        # 先证明"剔除前的原始集合里确实混着 tar"——否则本用例证明不了什么
        self.assertTrue(any(n.lower().endswith(".tar") for n in sample),
                        "样本里必须混着 .tar，否则测不到剔除")
        # 再证明剔除后只剩真 run
        self.assertEqual(runs, ["S14B2-load_ramp-ramp_cost",
                                "S14B2-load_ramp-ramp_cost-r1",
                                "S14B2-load_ramp-ramp_static"])
        self.assertEqual(set(staging), {
            "S14B2-load_ramp-ramp_slo_weave-r1.tar",
            "S14B2-load_ramp-ramp_slo_weave-r1.tar.tar",
            "X-thing.tgz", "Y-TGZ.TAR"})

    def test_clean_input_is_untouched(self):
        """反面对照：没有暂存包时**一个都不许少**（别把剔除写成乱剔）。"""
        m = _load()
        runs, staging = m._split_staging(["A", "B", "C"])
        self.assertEqual(runs, ["A", "B", "C"])
        self.assertEqual(staging, [])
        self.assertEqual(m._split_staging([]), ([], []))

    def test_remote_dirs_excludes_staging(self):
        """`_remote_dirs` 必须在**调用方看不到** tar 的前提下返回 runs，
        且把剔除情况附在原始输出里（否则"剔掉了什么"没人知道 = 静默吞）。"""
        import subprocess as sp
        from unittest import mock

        m = _load()
        fake = sp.CompletedProcess(
            args=[], returncode=0,
            stdout=("S14B2-a\nS14B2-b\nS14B2-b.tar\nS14B2-b.tar.tar\n"),
            stderr="")
        with mock.patch.object(m.subprocess, "run", return_value=fake):
            runs, raw = m._remote_dirs("/root/x", "S14B2", "py")
        self.assertEqual(runs, ["S14B2-a", "S14B2-b"])
        self.assertIn("[staging]", raw,
                      "剔除必须留痕，否则计数变化无从解释")
        self.assertIn("S14B2-b.tar", raw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
