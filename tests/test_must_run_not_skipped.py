#!/usr/bin/env python
"""**本机必测**不得转 skip —— 纪律的回归（用户 2026-10-03 点名）。

★ 钉住的失效形态
------------------
`unittest.skipTest` 让"**没跑**"在报告里显示成 **skip**，而 skip 与 pass 在很多
摘要里只差一个字符（`pytest -q` 的 `s` vs `.`）⇒ **静默的覆盖丢失**。

本仓已实吃过一次：端到端反例按 `r1..r5` 去探轮次目录，真实布局是
`-` + `-r1..-r4` ⇒ **6 条反例全跳过**，而当时的报告看起来是"全绿"。

⇒ 规矩分两种，**必须分开**：

| 输入 | 性质 | 缺失时 |
| --- | --- | --- |
| `configs/**`、`scripts/**`、已提交的 `results/processed/*.json` | **仓库内** | **失败**（`AssertionError`） |
| `results/raw/**`（远端跑出、需回拉） | **外部依赖** | 合法 skip，但用 `ExternalArtifactMissing`（带理由） |

⚠️ 本文件**自己也可能因此失败** —— 那正是重点：它证明"缺输入会红"，
   而不是"缺输入会静静变绿"。
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tests"))
import _tiny_support as _T  # noqa: E402

#: 允许出现**裸** `skipTest` 的文件白名单（须逐条有正当理由）。
#: 每一条都必须能用一句话解释"为什么这里不是本机必测"。
_ALLOWED_RAW_SKIP: dict[str, str] = {
    "tests/test_batch_evidence_eligibility.py": "远端批次产物未回拉 ⇒ 外部依赖",
    "tests/test_batch_lock.py": "launch_batch.sh 由部署步骤生成 ⇒ 部署态依赖",
    "tests/test_check_remote_busy.py": "远端进程表 ⇒ 外部依赖",
    "tests/test_deploy_status.py": "部署摘要由远端写入 ⇒ 外部依赖",
    "tests/test_env_fingerprint_identity.py": "远端环境指纹产物 ⇒ 外部依赖",
    "tests/test_eval_lookup_deployment_path.py": "真实评测产物 ⇒ 外部依赖",
    "tests/test_exec_opt_plan.py": "本机 bash / 既有批次产物 ⇒ 环境依赖",
    "tests/test_pull_incremental.py": "托管 venv 是可选的 ⇒ 环境依赖",
    "tests/test_run_tests_harness.py": "需要 pytest 才跑真实链路 ⇒ 环境依赖",
    "tests/test_source_stamp.py": "需要可用 git ⇒ 环境依赖",
    "tests/test_stage_report.py": "登记在案的已知缺陷 ⇒ 显式登记",
    "tests/test_step6_scan_configs.py": "step6 配置由生成器产出 ⇒ 生成态依赖",
    "tests/test_s01_recheck_analysis.py": "已改用 ExternalArtifactMissing（本条只是注释）",
}

#: **禁止**出现裸 `skipTest` 的文件：它们的输入全在仓库内。
_FORBIDDEN_RAW_SKIP = (
    "tests/test_prereg_batch_configs.py",
    "tests/test_verify_s01_numeric_offline.py",
)


class TestMustRunFilesDoNotBareSkipTest(unittest.TestCase):
    """仓库内输入缺失时必须**失败**，不得 `skipTest`。"""

    def test_forbidden_files_have_no_bare_skip_test(self):
        bad: list[str] = []
        for rel in _FORBIDDEN_RAW_SKIP:
            text = (_ROOT / rel).read_text(encoding="utf-8")
            for i, ln in enumerate(text.splitlines(), 1):
                if "skipTest(" not in ln:
                    continue
                if ln.lstrip().startswith("#"):
                    continue          # 注释里提到名字是可以的
                bad.append(f"{rel}:{i}: {ln.strip()}")
        self.assertEqual(
            bad, [],
            "下列位置仍用裸 skipTest ⇒ 缺输入会**静默变 skip**（应改 `require_local_file`）：\n  "
            + "\n  ".join(bad))

    def test_require_local_file_raises_on_missing(self):
        """★ 核心判据：缺文件必须 **raise AssertionError**，且**不是** SkipTest。"""
        missing = _ROOT / "results" / "processed" / "__definitely_absent__.json"
        self.assertFalse(missing.exists())
        with self.assertRaises(AssertionError):
            _T.require_local_file(missing, what="测试用不存在文件")
        # 明确：**不是** skip（SkipTest 是 AssertionError 的近亲，必须区分）
        try:
            _T.require_local_file(missing, what="x")
        except _T.ExternalArtifactMissing:  # pragma: no cover
            self.fail("require_local_file 不得抛 ExternalArtifactMissing（那是**外部依赖**用的）")
        except AssertionError:
            pass

    def test_require_local_file_passes_on_existing(self):
        """正向对照：存在的文件不该报错（否则判据恒假）。"""
        _T.require_local_file(_ROOT / "AGENTS.md", what="AGENTS.md")

    def test_external_artifact_missing_is_a_skiptest(self):
        """外部依赖用的类型**必须**是 `SkipTest` 的子类（才能被 unittest 识别为 skip）。"""
        self.assertTrue(issubclass(_T.ExternalArtifactMissing, unittest.SkipTest))


class TestEndToEndSuiteActuallyRunsWhenArtifactsPresent(unittest.TestCase):
    """★ 产物在本机 ⇒ 端到端用例**必须真的跑到**（不允许静默跳过）。

    做法：直接跑那几个端到端用例，断言 **skipped == 0**。
    —— 这比"看代码觉得它会跑"强，因为它**真的执行了一遍**。
    """

    ARTIFACT = _ROOT / "results" / "raw" / "S01RECHECK-short_chat-s01_both"

    def test_no_skips_when_artifacts_present(self):
        if not self.ARTIFACT.is_dir():
            raise _T.ExternalArtifactMissing(
                f"远端产物未回拉：{self.ARTIFACT} ⇒ 本项无法核验")
        p = subprocess.run(
            [sys.executable, "-m", "pytest",
             "tests/test_s01_recheck_analysis.py::TestValidNumericGateAtMainEntry",
             "tests/test_s01_recheck_analysis.py::TestJsonWriteIsSelfIdentifying",
             "-q", "-rs"],
            capture_output=True, text=True, cwd=str(_ROOT))
        self.assertEqual(p.returncode, 0, p.stdout[-1500:] + p.stderr[-500:])
        out = p.stdout
        self.assertNotIn(" skipped", out,
                         "产物在本机却出现 skip ⇒ 端到端覆盖静默丢失：\n" + out[-800:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
