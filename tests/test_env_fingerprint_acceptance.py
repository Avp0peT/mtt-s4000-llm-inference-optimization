"""环境指纹的**读取 + 接受判据**（2026-10-03 修，S01 复核的开跑前置）。

★ 要防的两件事
--------------
**① 读取缺陷（静默）**：旧代码读 `d.get('device_count')` 与
`software.mthreads_driver_version` —— 这两个键在**真实产物里不存在** ⇒ 恒为 `None`
⇒ 依赖它们的判据（"设备数为正"、"驱动已确认"）**一次都没执行过**，
而产物里它们看起来只是"没查到"。

**② 首次采集是"没有守卫的入口"**：旧逻辑把若干检查挂在"有上一份指纹可比"上，
顺带**放行了所有缺失字段** ⇒ 第一次采集缺什么都不拦。

⇒ 本文件的用例全部走**真实采集格式**（`tests/fixtures/env_fingerprint_real_mthreads.json`
是远端 `results/raw/_env/env.json` 的逐字副本，只把换行归一化成 LF）：
`真实格式 → read_env_fingerprint → env_fingerprint_acceptance → 判定`。
⚠️ 夹具**不是手写的想象结构** —— 这一点必须有，否则测的是我记忆里的字段名。
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("scripts", "benchmark", "src"):
    _p = str(_ROOT / _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exec_opt_batch as B  # noqa: E402

FIXTURE = _ROOT / "tests" / "fixtures" / "env_fingerprint_real_mthreads.json"

#: 夹具里那台机器的**实时**标识形态（`<hostname>/<user>`）。
LIVE = "autodl-container-cf99469eb6-b877706e/root"


def _real_doc() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class TestFixtureIsTheRealFormat(unittest.TestCase):
    """夹具必须真的是采集器产出的形状（否则下面所有用例都在验想象）。"""

    def test_fixture_parses_and_has_the_real_top_level_keys(self):
        d = _real_doc()
        for k in ("available", "collected_at", "device", "discovery", "hostname",
                  "platform", "probes", "software"):
            self.assertIn(k, d, f"真实 `env.json` 应有顶层键 {k}")

    def test_old_reader_paths_really_do_not_exist(self):
        """★ **回归锚**：把"旧代码读的两个键不存在"这件事钉住。

        若哪天有人把这两个键**加进**产物（那是好事），这条会红 —— 那时应当
        重新评估"读取路径"而不是默默继续读旧键。
        """
        d = _real_doc()
        self.assertNotIn("device_count", d,
                         "顶层没有 `device_count`（旧代码读的正是它）")
        self.assertNotIn("mthreads_driver_version", d.get("software") or {},
                         "`software` 里没有 `mthreads_driver_version`（旧代码读的正是它）")
        # 真实位置在这里：
        self.assertIn("count", d["device"])
        self.assertIn("MTT_DRIVER_VERSION", d["discovery"]["values"])


class TestReaderUsesTheRealPaths(unittest.TestCase):
    def setUp(self):
        self.fp = B.read_env_fingerprint(_real_doc())

    def test_the_two_fields_that_used_to_be_none_are_now_read(self):
        """★ 靶心之一：`device_count` 与 `driver` 过去恒为 `None`，现在读得到。"""
        self.assertEqual(self.fp["device_count"], 1)
        self.assertEqual(self.fp["driver"], "2.7.0")

    def test_other_fields(self):
        self.assertEqual(self.fp["hostname"], "autodl-container-cf99469eb6-b877706e")
        self.assertEqual(self.fp["collected_at"], "2026-09-28T00:16:44+0800")
        self.assertEqual(self.fp["platform"], "mthreads")
        self.assertEqual(self.fp["device_model"], "MTT S4000")
        self.assertTrue(self.fp["available"])
        self.assertEqual(self.fp["torch"], "2.2.0")
        self.assertEqual(self.fp["torch_musa"], "1.3.0+81caf0a")
        self.assertEqual(self.fp["transformers"], "4.46.3")
        self.assertEqual(self.fp["musa_version"], "3.1.0")
        self.assertEqual(self.fp["model_revision"],
                         "a09a35458c702b33eeacc393d103063234e8bc28")

    def test_falls_back_to_probes_when_device_block_lacks_a_field(self):
        """`device.count` 缺了就退到 `probes.mthreads.device_count`（两处同源，非重复实现）。"""
        d = _real_doc()
        d["device"] = {k: v for k, v in d["device"].items() if k != "count"}
        fp = B.read_env_fingerprint(d)
        self.assertEqual(fp["device_count"], 1, "应退到 probes 块")

    def test_missing_gives_none_not_zero_and_not_empty_string(self):
        """★ 缺 ⇒ `None`。**不许**填 0（"0 张卡"与"没读到"是两件事）。"""
        fp = B.read_env_fingerprint({})
        self.assertIsNone(fp["device_count"])
        self.assertIsNone(fp["driver"])
        self.assertIsNone(fp["device_model"])
        self.assertIsNotNone(fp["discovery_missing"], "缺口列表要有（哪怕是空表）")

    def test_none_document_is_tolerated(self):
        fp = B.read_env_fingerprint(None)
        self.assertIsNone(fp["hostname"])
        self.assertIsNone(fp["device_count"])


class TestAcceptancePositive(unittest.TestCase):
    def setUp(self):
        self.fp = B.read_env_fingerprint(_real_doc())

    def test_real_fingerprint_is_accepted(self):
        v = B.env_fingerprint_acceptance(fp=self.fp, live_hostname=LIVE,
                                        declared_platform="mthreads")
        self.assertTrue(v["ok"], f"真实指纹应被接受：{v['complaints']}")
        self.assertEqual(v["complaints"], [])
        self.assertIsNotNone(v["collected_at_epoch"], "必须给出可比的采集时刻")

    def test_driver_mismatch_is_rejected_when_expected_driver_given(self):
        v = B.env_fingerprint_acceptance(fp=self.fp, live_hostname=LIVE,
                                        declared_platform="mthreads",
                                        expect_driver="9.9.9")
        self.assertFalse(v["ok"])
        self.assertTrue(any("驱动版本不符" in c for c in v["complaints"]))

    def test_driver_absent_is_a_note_not_a_pass_silencer(self):
        fp = dict(self.fp, driver=None)
        v = B.env_fingerprint_acceptance(fp=fp, live_hostname=LIVE,
                                        declared_platform="mthreads")
        self.assertTrue(v["ok"], "驱动取值缺失**不单独**判死（老指纹可能没有）")
        self.assertTrue(any("驱动版本**未确认**" in n for n in v["notes"]),
                        "但必须显式记 note（不许静默）")


class TestAcceptanceNegatives(unittest.TestCase):
    """★ 逐字段反例：**每一个**都必须被拒，且拒的理由要对。"""

    def setUp(self):
        self.fp = B.read_env_fingerprint(_real_doc())

    def _v(self, **over):
        fp = dict(self.fp)
        fp.update(over)
        return B.env_fingerprint_acceptance(fp=fp, live_hostname=LIVE,
                                           declared_platform="mthreads")

    def _rejects(self, v, needle):
        self.assertFalse(v["ok"], f"应被拒，实际通过（needle={needle}）")
        self.assertTrue(any(needle in c for c in v["complaints"]),
                        f"拒的理由应含 `{needle}`，实际：{v['complaints']}")

    # ① 设备可用
    def test_device_unavailable_is_rejected(self):
        self._rejects(self._v(available=False), "设备不可用")

    def test_device_availability_missing_is_rejected(self):
        """★ 读不到 ≠ 可用（旧代码就是在这里静默放行的）。"""
        self._rejects(self._v(available=None), "无法确认设备可用")

    # ② 设备数为正
    def test_zero_devices_is_rejected(self):
        self._rejects(self._v(device_count=0), "必须为正")

    def test_negative_devices_is_rejected(self):
        self._rejects(self._v(device_count=-1), "必须为正")

    def test_missing_device_count_is_rejected(self):
        """★ 靶心之二：旧代码这里恒为 `None`，于是这条判据从未执行。"""
        self._rejects(self._v(device_count=None), "读不到**设备数**")

    def test_non_integer_device_count_is_rejected(self):
        self._rejects(self._v(device_count="many"), "不是整数")

    # ③ 平台 / 型号
    def test_platform_mismatch_is_rejected(self):
        self._rejects(self._v(platform="ascend"), "平台不符")

    def test_missing_platform_is_rejected(self):
        self._rejects(self._v(platform=None), "读不到 `platform`")

    def test_model_not_looking_like_mtt_is_rejected(self):
        self._rejects(self._v(device_model="SomeOtherCard 9000"), "不匹配 MTT 产品名形状")

    def test_missing_model_is_rejected(self):
        self._rejects(self._v(device_model=None), "读不到**设备型号**")

    def test_explicit_model_mismatch_is_rejected(self):
        v = B.env_fingerprint_acceptance(fp=self.fp, live_hostname=LIVE,
                                        declared_platform="mthreads",
                                        declared_model="S80")
        self.assertFalse(v["ok"])
        self.assertTrue(any("型号不符" in c for c in v["complaints"]))

    def test_looks_like_false_contradicts_declared_platform(self):
        self._rejects(self._v(looks_like_mthreads_host=False), "矛盾")

    # ④ hostname 与实时实例一致
    def test_hostname_mismatch_is_rejected(self):
        """★ A7 的靶心：指纹来自别的容器。"""
        self._rejects(self._v(hostname="autodl-container-OTHER-123"),
                      "指纹与实时实例不一致")

    def test_missing_fingerprint_hostname_is_rejected(self):
        self._rejects(self._v(hostname=None), "读不到 `hostname`")

    def test_missing_live_hostname_is_rejected(self):
        v = B.env_fingerprint_acceptance(fp=self.fp, live_hostname="",
                                        declared_platform="mthreads")
        self._rejects(v, "实时机器标识取不到")

    # ⑤ 采集时间
    def test_unparsable_collected_at_is_rejected(self):
        self._rejects(self._v(collected_at="sometime"), "解析不出时间")

    def test_missing_collected_at_is_rejected(self):
        self._rejects(self._v(collected_at=None), "解析不出时间")

    # ⑥ 必填字段
    def test_required_field_list_is_enforced(self):
        fp = dict(self.fp)
        fp["device_ok"] = None
        v = B.env_fingerprint_acceptance(fp=fp, live_hostname=LIVE,
                                         declared_platform="mthreads")
        self.assertFalse(v["ok"])
        self.assertTrue(any("必填字段缺失" in c for c in v["complaints"]))

    def test_required_fields_constant_covers_the_six_checks(self):
        self.assertEqual(set(B.REQUIRED_FINGERPRINT_FIELDS),
                         {"hostname", "collected_at", "platform",
                          "device_count", "device_model", "device_ok"})


class TestFirstCollectionIsFullyChecked(unittest.TestCase):
    """★★ 用户明确要求：**首次采集同样检查**，不能因没有上一份就放行缺失字段。

    实现上这条靠"接受判据**不接受** `prev` 参数"来保证 —— 它没有"首次"这个概念，
    因此不可能存在"首次就跳过"的分支。下面用**结构断言 + 行为断言**把它钉住。
    """

    def test_acceptance_signature_takes_no_prev(self):
        """★ 只要签名里没有 `prev`，「首次放宽」在**接口层面**就不可能发生。"""
        import inspect

        params = set(inspect.signature(B.env_fingerprint_acceptance).parameters)
        self.assertNotIn("prev", params,
                         "接受判据不得依赖「有没有上一份指纹」—— 那正是旧缺陷的入口")
        self.assertNotIn("is_first", params)

    def test_missing_fields_are_rejected_regardless_of_any_previous_state(self):
        """行为断言：缺字段一律拒 —— 调用方无从表达「这是第一次」。"""
        fp = B.read_env_fingerprint(_real_doc())
        for broken in ({"device_count": None}, {"platform": None},
                       {"device_model": None}, {"collected_at": None},
                       {"hostname": None}):
            bad = dict(fp, **broken)
            v = B.env_fingerprint_acceptance(fp=bad, live_hostname=LIVE,
                                            declared_platform="mthreads")
            self.assertFalse(v["ok"], f"{broken} 应被拒（与是否首次无关）")


class TestFreshnessStaysConditional(unittest.TestCase):
    """新鲜度**必须**比上一份才能回答 ⇒ 它单独走 `fingerprint_identity_verdict`。"""

    def setUp(self):
        self.fp = B.read_env_fingerprint(_real_doc())

    def test_first_collection_records_a_note_not_a_pass(self):
        v = B.fingerprint_identity_verdict(prev=None, cur=self.fp, live_machine_id=LIVE)
        self.assertTrue(v["ok"])
        self.assertTrue(any("首次" in n for n in v["notes"]))

    def test_stale_fingerprint_is_rejected(self):
        prev = {"hostname": self.fp["hostname"],
                "collected_at": "2026-10-03T00:00:00+0800"}   # 比新指纹**更新**
        v = B.fingerprint_identity_verdict(prev=prev, cur=self.fp, live_machine_id=LIVE)
        self.assertFalse(v["ok"])
        self.assertTrue(any("没有被刷新" in c for c in v["complaints"]))

    def test_refreshed_fingerprint_passes(self):
        prev = {"hostname": self.fp["hostname"],
                "collected_at": "2026-09-27T00:00:00+0800"}   # 比新指纹旧
        v = B.fingerprint_identity_verdict(prev=prev, cur=self.fp, live_machine_id=LIVE)
        self.assertTrue(v["ok"], v["complaints"])


if __name__ == "__main__":
    unittest.main()
