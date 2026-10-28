"""批次证据资格（`EVIDENCE_INDEX.md` §5）的**机器核对**。

★ 为什么要有这个文件
--------------------
§5 那张"可正式引用／仅作诊断／证据待补"的表，逐行都是**从产物现读**的事实
（`git_source` / `tree_digest` / `hostname_or_machine_id` / `hardware_env.collected_at`）。
但文档里的表格**会过期**：产物被移动、被重新回拉、或有人补采了环境指纹之后，
表里的判定就可能不再成立，而**没有任何东西会提醒**。

⇒ 把这个判定做成测试：表里每一行的关键事实都asserts 一遍。
  这样"资格判定"与"产物现状"绑在一起，**改产物就会打红**，强迫更新表格。

⚠️ 本文件**只读** `results/raw/**`，不写、不改任何产物。
⚠️ 产物不在本机（未回拉）时**跳过并说明**——跳过不计入通过（AGENTS 附录 A5）。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RAW = _ROOT / "results" / "raw"


def _first_meta(tag: str) -> dict | None:
    """取该 tag 下第一个可解析的 `meta.json`（同批次内字段一致，取一份即可）。"""
    for d in sorted(RAW.glob(f"{tag}-*")):
        mp = d / "meta.json"
        if not mp.is_file():
            continue
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
    return None


def _facts(meta: dict) -> dict:
    repro = meta.get("reproducibility") or {}
    env = meta.get("hardware_env") or {}
    host = str(repro.get("hostname_or_machine_id") or "")
    return {
        "git_source": repro.get("git_source"),
        "tree_digest": repro.get("tree_digest"),
        "label": meta.get("data_label"),
        "fp_host": str(env.get("hostname") or ""),
        "fp_collected_at": str(env.get("collected_at") or ""),
        "live_host": host.split("/", 1)[0],
    }


class TestOlderPerfBatchesHaveEnvironmentButNoSource(unittest.TestCase):
    """`EXECOPT3/4`：**环境与容器同源**，但**没有源码身份** ⇒ T2 仅作诊断。"""

    def _check(self, tag: str):
        meta = _first_meta(tag)
        if meta is None:
            self.skipTest(f"{tag} 产物不在本机（未回拉）⇒ 无法核验")
        f = _facts(meta)
        self.assertEqual(f["git_source"], "stamp_stale",
                         f"{tag} 的 git_source 应仍是 stamp_stale（若变了，§5 要重判）")
        self.assertFalse(str(f["tree_digest"] or "").strip(),
                         f"{tag} 不该有 tree_digest（有的话它就能进 T1 了）")
        self.assertEqual(f["label"], "MEASURED",
                         "标签是 MEASURED —— 这正是'MEASURED ≠ T1'的例子")
        self.assertTrue(f["fp_host"] and f["fp_host"] == f["live_host"],
                        f"{tag} 的指纹与实跑容器应**同源**"
                        f"（fp={f['fp_host']} live={f['live_host']}）")

    def test_execopt3(self):
        self._check("EXECOPT3")

    def test_execopt4(self):
        self._check("EXECOPT4")


class TestExecopt5HasSourceButMismatchedEnvironment(unittest.TestCase):
    """`EXECOPT5`：**源码身份闭合**，但**环境异机** ⇒ 仍只能 T2（§4.4 A7）。"""

    def setUp(self):
        self.meta = _first_meta("EXECOPT5")
        if self.meta is None:
            self.skipTest("EXECOPT5 产物不在本机（未回拉）⇒ 无法核验")
        self.f = _facts(self.meta)

    def test_source_identity_is_closed(self):
        self.assertEqual(self.f["git_source"], "stamp")
        self.assertEqual(len(str(self.f["tree_digest"])), 64, "tree_digest 应是 64 位 sha256")

    def test_environment_fingerprint_is_from_a_different_container(self):
        """★ 靶心：指纹的 hostname 与实时容器**不同** ⇒ 环境绑定未闭合。"""
        self.assertTrue(self.f["fp_host"], "指纹应有 hostname")
        self.assertTrue(self.f["live_host"], "应有实时机器标识")
        self.assertNotEqual(self.f["fp_host"], self.f["live_host"],
                            "A7 的事实：指纹来自另一个容器；"
                            "若相等说明产物被改过或已重采，§5 要重判")

    def test_fingerprint_predates_the_run(self):
        """指纹采集时间**早于**该次运行（陈旧 5 天）—— 这是"存在即用"的直接后果。"""
        self.assertTrue(self.f["fp_collected_at"].startswith("2026-09-28"),
                        f"指纹应仍是 09-28 的那份，实际 {self.f['fp_collected_at']!r}")


class TestS04VerifyArtifactIsTheClosestToCitable(unittest.TestCase):
    """S04 数值验证产物：源码 + 容器身份**都闭合**，但缺 MUSA 驱动版本（残留缺口）。"""

    ART = (RAW / "_s04_verify"
           / "s04_reenact_verify_20261002T194537Z.json")

    def setUp(self):
        if not self.ART.is_file():
            self.skipTest("S04 验证产物不在本机 ⇒ 无法核验")

    def _doc(self) -> dict:
        return json.loads(self.ART.read_text(encoding="utf-8"))

    def test_source_identity_is_closed_and_validated(self):
        si = self._doc().get("source_identity") or {}
        self.assertEqual(len(str(si.get("tree_digest") or "")), 64)
        self.assertTrue((si.get("stamp") or {}).get("validated"),
                        "部署戳须为**已校验**状态")

    def test_environment_was_collected_at_run_time(self):
        """★ 它**自己现采**环境（不读共享指纹）⇒ 容器身份是**实时**的。"""
        env = self._doc().get("environment") or {}
        self.assertTrue(str(env.get("hostname") or "").startswith("autodl-container-"),
                        "应记录实时 hostname")
        self.assertIn("platform", env, "应记录 OS/kernel")
        self.assertIn("torch_musa", env, "应记录 MUSA 运行时版本")

    def test_musa_driver_version_is_the_named_residual_gap(self):
        """残留缺口：**MUSA 驱动版本**没有被采集 ⇒ 跨驱动的可比性不可主张。"""
        blob = json.dumps(self._doc().get("environment") or {}, ensure_ascii=False).lower()
        self.assertNotIn("mthreads_driver", blob)
        self.assertNotIn("musa_version", blob)

    def test_it_does_not_assert_performance(self):
        """产物自己声明**不主张**性能 —— 引用时不得越界。"""
        dna = " ".join(self._doc().get("does_not_assert") or [])
        self.assertIn("性能", dna)
        self.assertIn("预热", dna)

    def test_warmup_deduction_is_recorded_as_unverified(self):
        self.assertIs(self._doc().get("warmup_deduction_verified"), False)


if __name__ == "__main__":
    unittest.main()
