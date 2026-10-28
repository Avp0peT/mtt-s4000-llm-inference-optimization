"""`scripts/deploy_status.py` —— 本地源码相对**已部署版本**的状态。

★ 要防的那件事（2026-10-03）
----------------------------
代码在**本机**改、戳打在**远端** ⇒ 存在一个安静的危险窗口：
改了代码之后，本机摘要变了，而远端戳还是旧的；若有人拿那个戳当"当前源码的身份"，
就会把"这批数字来自我现在这份代码"这句话说成事实 —— 而它其实来自**改动之前**那份。

⇒ 本文件锁住三件事：
  1. `IN_SYNC` 只在摘要**真的相等**时出现；
  2. 缺记录 / 缺本机摘要**都不许**被读成"同步"；
  3. 状态文案里必须**明确否定**"旧戳代表当前源码"。

⚠️ 本文件**不连远端**，只测纯函数与本地记录读写。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("scripts", "benchmark", "src"):
    _p = str(_ROOT / _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import deploy_status as D  # noqa: E402

DEPLOYED = "bff7f7080b67914ea05911f9825ee2e181232928d665ede07dc866bf5654fc9f"
LOCAL = "6b67af55b4c543dd000000000000000000000000000000000000000000000000"


class TestDeploymentState(unittest.TestCase):
    def _rec(self, digest=None, at="2026-10-03T03:45:07+0800"):
        return {"schema": D.SCHEMA, "deployed_tree_digest": digest, "recorded_at": at}

    def test_in_sync_only_when_digests_are_equal(self):
        st = D.deployment_state(local_digest=DEPLOYED, record=self._rec(DEPLOYED))
        self.assertEqual(st["state"], "IN_SYNC")
        self.assertFalse(st["redeploy_needed"])

    def test_local_ahead_is_flagged_as_not_deployed(self):
        st = D.deployment_state(local_digest=LOCAL, record=self._rec(DEPLOYED))
        self.assertEqual(st["state"], "LOCAL_AHEAD")
        self.assertTrue(st["redeploy_needed"])

    def test_local_ahead_message_denies_the_old_stamp_represents_current_source(self):
        """★ 靶心：文案必须**明确否定**"旧戳代表当前源码"。"""
        st = D.deployment_state(local_digest=LOCAL, record=self._rec(DEPLOYED))
        msg = st["message"]
        self.assertIn("尚未部署", msg)
        self.assertIn("不代表当前源码", msg)
        self.assertIn("不得", msg)
        # 两个摘要都要出现，便于人工比对
        self.assertIn(LOCAL[:12], msg)
        self.assertIn(DEPLOYED[:12], msg)

    def test_missing_record_is_never_deployed_not_in_sync(self):
        """没有记录 ⇒ `NEVER_DEPLOYED`（**不是** `IN_SYNC`）。"""
        st = D.deployment_state(local_digest=LOCAL, record=None)
        self.assertEqual(st["state"], "NEVER_DEPLOYED")
        self.assertTrue(st["redeploy_needed"])

    def test_record_without_digest_is_never_deployed(self):
        st = D.deployment_state(local_digest=LOCAL, record=self._rec(None))
        self.assertEqual(st["state"], "NEVER_DEPLOYED")

    def test_unreadable_local_digest_is_unknown_not_in_sync(self):
        """★ 本机摘要**取不到** ⇒ `UNKNOWN`，且**不得**被读成"同步"。"""
        st = D.deployment_state(local_digest=None, record=self._rec(DEPLOYED))
        self.assertEqual(st["state"], "UNKNOWN")
        self.assertTrue(st["redeploy_needed"])
        self.assertIn("不得", st["message"])

    def test_all_states_carry_the_schema(self):
        for st in (
            D.deployment_state(local_digest=LOCAL, record=self._rec(DEPLOYED)),
            D.deployment_state(local_digest=DEPLOYED, record=self._rec(DEPLOYED)),
            D.deployment_state(local_digest=LOCAL, record=None),
            D.deployment_state(local_digest=None, record=self._rec(DEPLOYED)),
        ):
            self.assertEqual(st["schema"], D.SCHEMA)


class TestRecordReadWrite(unittest.TestCase):
    def test_read_record_missing_file_returns_none_not_empty_dict(self):
        """★ 读不到 ⇒ `None`。返回 `{}` 会被读成"记录存在但没内容"，两者不同。"""
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(D.read_record(Path(td) / "nope.json"))

    def test_read_record_corrupt_file_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text("{not json", encoding="utf-8", newline="\n")
            self.assertIsNone(D.read_record(p))

    def test_write_then_read_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "D.json"
            original = D.RECORD_PATH
            try:
                D.RECORD_PATH = p
                D.write_record(digest=DEPLOYED, note="t")
                rec = D.read_record(p)
            finally:
                D.RECORD_PATH = original
            self.assertEqual(rec["deployed_tree_digest"], DEPLOYED)
            self.assertEqual(rec["schema"], D.SCHEMA)
            self.assertIn("recorded_at", rec)
            self.assertEqual(p.read_bytes().count(b"\r"), 0, "记录必须是 LF")


class TestRepoDeploymentRecordIsHonest(unittest.TestCase):
    """★ 仓库里那份记录必须与**本机现状**一致（否则它自己在说谎）。"""

    def test_record_exists_and_has_a_digest(self):
        rec = D.read_record()
        if rec is None:
            self.skipTest("尚未记录部署摘要（本轮之前没有这份记录）⇒ 无法核验")
        self.assertEqual(len(str(rec.get("deployed_tree_digest") or "")), 64)

    def test_current_state_matches_the_measured_digest(self):
        """把真实本机摘要算出来，状态必须是可解释的三态之一（不许抛错/空）。"""
        st = D.deployment_state(local_digest=D.local_tree_digest(), record=D.read_record())
        self.assertIn(st["state"], ("IN_SYNC", "LOCAL_AHEAD", "NEVER_DEPLOYED",
                                    "UNKNOWN"))
        # 本轮改过代码 ⇒ 现应是 LOCAL_AHEAD（若哪天变成 IN_SYNC，说明真部署过了）
        self.assertIn(st["state"], ("LOCAL_AHEAD", "IN_SYNC"))

    def test_json_is_a_superset_of_what_the_doc_claims(self):
        """状态对象的键要与取用方约定一致（防有人删键后下游静默拿到 None）。"""
        st = D.deployment_state(local_digest=LOCAL, record=None)
        for key in ("state", "redeploy_needed", "local_tree_digest",
                    "deployed_tree_digest", "message", "schema", "recorded_at"):
            self.assertIn(key, st)
        json.dumps(st, ensure_ascii=False)   # 必须可序列化


if __name__ == "__main__":
    unittest.main()
