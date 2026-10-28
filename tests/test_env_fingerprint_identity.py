"""批次启动前的**环境指纹身份**判据（`fingerprint_identity_verdict`）。

★ 2026-10-03 实测事故（本文件的由来）
--------------------------------------
EXECOPT5（2026-10-03）的 10 份产物里嵌着 `hardware_env`：

    collected_at = 2026-09-28T00:16:44+0800
    hostname     = autodl-container-cf99469eb6-b877706e

而它实际跑在 `autodl-container-05264bbf0d-3b9c3488`。
成因：`load_hardware_env` 只按**路径**找 `results/raw/_env/env.json`，
**没有任何新鲜度或容器身份校验** ⇒ 一份 5 天前、**另一个容器**采的指纹
被原样写进了全部产物，而产物看起来完全正常。

为什么这不能"算了"：那份指纹承载着 `mthreads_driver_version` / OS / CPU /
`device_model` —— 这些正是**跨批次可比性的前提**。指纹来自 B 容器、数字来自
A 容器时，"两批同环境"这句话**没有证据**。

本文件把两个方向都锁住：
* 正向：指纹描述的机器 == 实跑机器（含"换了容器但指纹是新的"这一合法情形）；
* **反向**：指纹陈旧/异机/缺失/设备数为 0 时**必须拒绝**（不能"看起来正常"就放过）。
"""

from __future__ import annotations

import io
import json
import contextlib
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("scripts", "benchmark", "src"):
    _p = str(_ROOT / _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exec_opt_batch as B  # noqa: E402

#: 实测里出现过的两个容器（EXECOPT5 那一次）。
OLD_CONTAINER = "autodl-container-cf99469eb6-b877706e"
NEW_CONTAINER = "autodl-container-05264bbf0d-3b9c3488"
OLD_COLLECTED = "2026-09-28T00:16:44+0800"
NEW_COLLECTED = "2026-10-03T03:47:12+0800"


def _fp(hostname: str, collected_at: str, *, available=True, device_count=1):
    return {"hostname": hostname, "collected_at": collected_at,
            "available": available, "device_count": device_count}


class TestFingerprintIdentityVerdict(unittest.TestCase):
    """判据本体：**纯函数**，直接喂构造好的指纹。"""

    def _v(self, *, prev, cur, live=NEW_CONTAINER + "/root"):
        return B.fingerprint_identity_verdict(prev=prev, cur=cur,
                                              live_machine_id=live)

    # ---------------- 正向 ----------------

    def test_identical_container_is_clean(self):
        v = self._v(prev=_fp(NEW_CONTAINER, OLD_COLLECTED),
                    cur=_fp(NEW_CONTAINER, NEW_COLLECTED))
        self.assertTrue(v["ok"])
        self.assertEqual(v["complaints"], [])
        self.assertFalse(v["container_changed"], "同一容器 ⇒ 未变")

    def test_container_change_but_fresh_fingerprint_is_allowed_with_a_note(self):
        """★ 合法情形：换了容器，但指纹是**本次重采**的 ⇒ 允许，只记不改判。"""
        v = self._v(prev=_fp(OLD_CONTAINER, OLD_COLLECTED),
                    cur=_fp(NEW_CONTAINER, NEW_COLLECTED))
        self.assertTrue(v["ok"], "指纹是新的且描述实跑机器 ⇒ 不该拦")
        self.assertEqual(v["complaints"], [])
        self.assertTrue(v["container_changed"])
        self.assertTrue(v["notes"], "换容器必须留下显著提示（跨批可比性前提变了）")
        self.assertIn("容器已变", v["notes"][0])

    def test_first_ever_collection_is_clean(self):
        v = self._v(prev=None, cur=_fp(NEW_CONTAINER, NEW_COLLECTED))
        self.assertTrue(v["ok"])
        self.assertIsNone(v["container_changed"], "没有上一份 ⇒ 无从比较，记 None")

    # ---------------- 反向：都必须拦住 ----------------

    def test_stale_fingerprint_from_another_container_is_rejected(self):
        """★★ 靶心：**就是 EXECOPT5 那次的形态**——指纹来自另一个容器。

        注意 `prev` 与 `cur` 在这里**相同**（都是旧容器）⇒ `container_changed=False`，
        也就是说"换没换容器"这个信息**看不出来**问题。能被看出来的只有
        "指纹的 host 与实时机器不一致"。这条正是判据存在的理由。
        """
        v = self._v(prev=_fp(OLD_CONTAINER, OLD_COLLECTED),
                    cur=_fp(OLD_CONTAINER, OLD_COLLECTED))
        self.assertFalse(v["ok"], "★ 指纹不是本机的 ⇒ 必须停")
        self.assertFalse(v["container_changed"], "两者相同 ⇒ 变更标志为 False（看不出来）")
        self.assertTrue(any("不是同一台" in c for c in v["complaints"]))
        self.assertTrue(any(NEW_CONTAINER in c for c in v["complaints"]))

    def test_missing_fingerprint_is_rejected(self):
        v = self._v(prev=None, cur=None)
        self.assertFalse(v["ok"])
        self.assertTrue(any("取不到" in c for c in v["complaints"]))

    def test_unavailable_device_is_rejected(self):
        v = self._v(prev=None, cur=_fp(NEW_CONTAINER, NEW_COLLECTED, available=False))
        self.assertFalse(v["ok"])
        self.assertTrue(any("available=False" in c for c in v["complaints"]))

    def test_zero_devices_is_rejected(self):
        v = self._v(prev=None, cur=_fp(NEW_CONTAINER, NEW_COLLECTED, device_count=0))
        self.assertFalse(v["ok"])
        self.assertTrue(any("device_count=0" in c for c in v["complaints"]))

    def test_fingerprint_without_hostname_is_rejected(self):
        v = self._v(prev=None, cur=_fp("", NEW_COLLECTED))
        self.assertFalse(v["ok"])
        self.assertTrue(any("没有 hostname" in c for c in v["complaints"]))

    def test_unreadable_live_identity_is_rejected(self):
        """实跑机器标识**测不出** ⇒ 拒绝。

        这一条挡的是"两头都空所以相等"那类**恒定值假通过**：
        `live_machine_id` 为空时**不能**判通过。
        """
        v = self._v(prev=None, cur=_fp(NEW_CONTAINER, NEW_COLLECTED), live="")
        self.assertFalse(v["ok"])
        self.assertTrue(any("取不到" in c for c in v["complaints"]))

    def test_hostname_comparison_ignores_the_user_suffix(self):
        """`detect_machine_id()` 形如 `host/user`；指纹只存 host ⇒ 取 host 比。"""
        v = self._v(prev=None, cur=_fp(NEW_CONTAINER, NEW_COLLECTED),
                    live=NEW_CONTAINER + "/someotheruser")
        self.assertTrue(v["ok"], "用户名不同不该判为异机")


class TestFingerprintFreshnessAndDeviceRecord(unittest.TestCase):
    """★ 两条"看起来通过了、其实没查"的补充判据。

    * **新鲜度**：主机对账只发现"指纹来自别的机器"。若重采没生效（写去了别处、
      中途失败），指纹仍是旧的 —— 而它描述的机器**恰好还是本机**（同容器跑第二天
      的批次）⇒ 主机对账**照样通过**，只有日期是旧的。这正是 §4.4 A7 的形态。
    * **设备记录**：配置声明 `mthreads` 而指纹说"看起来不是 MTT 主机" ⇒ 必须拦。
    """

    NEW = "autodl-container-05264bbf0d-3b9c3488"
    OLD = "autodl-container-cf99469eb6-b877706e"

    def _v(self, **over):
        kw = dict(
            prev=_fp(self.NEW, "2026-10-03T03:40:00+0800"),
            cur={**_fp(self.NEW, "2026-10-03T03:47:12+0800"),
                 "looks_like_mthreads_host": True},
            live_machine_id=self.NEW + "/root",
        )
        kw.update(over)
        return B.fingerprint_identity_verdict(**kw)

    def test_refresh_that_did_not_refresh_is_blocked(self):
        """★ 靶心：`collected_at` 与上一份**一样** ⇒ 重采没生效 ⇒ 停。"""
        v = self._v(cur={**_fp(self.NEW, "2026-10-03T03:40:00+0800"),
                         "looks_like_mthreads_host": True})
        self.assertFalse(v["ok"], "指纹没被刷新必须拦（机器对账发现不了这一种）")
        self.assertTrue(any("没有被刷新" in c for c in v["complaints"]))

    def test_older_fingerprint_is_blocked(self):
        """比上一份**更旧**（时间倒流）同样拦。"""
        v = self._v(cur={**_fp(self.NEW, "2026-09-28T00:16:44+0800"),
                         "looks_like_mthreads_host": True})
        self.assertFalse(v["ok"])

    def test_newer_fingerprint_passes(self):
        self.assertTrue(self._v()["ok"])

    def test_unparsable_time_is_blocked(self):
        v = self._v(cur={**_fp(self.NEW, "not-a-time"),
                         "looks_like_mthreads_host": True})
        self.assertFalse(v["ok"])
        self.assertTrue(any("解析不出时间" in c for c in v["complaints"]))

    def test_first_ever_collection_notes_that_freshness_cannot_be_checked(self):
        """首次采集没有上一份可比 ⇒ 记 note（**不是**记通过）。"""
        v = self._v(prev=None, cur={**_fp(self.NEW, "2026-10-03T03:47:12+0800"),
                                    "looks_like_mthreads_host": True})
        self.assertTrue(v["ok"])
        self.assertTrue(any("首次" in n for n in v["notes"]))

    def test_device_record_contradicting_declared_platform_is_blocked(self):
        v = self._v(cur={**_fp(self.NEW, "2026-10-03T03:47:12+0800"),
                         "looks_like_mthreads_host": False},
                    declared_platform="mthreads")
        self.assertFalse(v["ok"])
        self.assertTrue(any("设备记录与声明不符" in c for c in v["complaints"]))

    def test_missing_device_record_is_a_note_not_a_pass(self):
        """没有 `looks_like_mthreads_host` ⇒ 记 note（**不是**通过）。"""
        cur = _fp(self.NEW, "2026-10-03T03:47:12+0800")   # ⚠️ **不含** 该键
        self.assertNotIn("looks_like_mthreads_host", cur, "这条测试就是要它缺")
        v = self._v(cur=cur, declared_platform="mthreads")
        self.assertTrue(v["ok"])
        self.assertTrue(any("无法核对设备记录" in n for n in v["notes"]),
                        f"应记 note，实际 notes={v['notes']}")

    def test_no_declared_platform_skips_that_axis(self):
        v = self._v()
        self.assertTrue(v["ok"])
        self.assertEqual([n for n in v["notes"] if "设备记录" in n], [])


class TestFourClassIdentity(unittest.TestCase):
    """★ 源码 / 配置 / 模型 / 环境**四类身份**都要记（缺哪类都要看得见）。"""

    CFG_DIR = "configs/mthreads/exec_opt"
    BASELINE = f"{CFG_DIR}/s01_both.yaml"
    ARM = "s01_both_s04_pkv"

    @classmethod
    def setUpClass(cls):
        cls.ident = B.four_class_identity(
            root=_ROOT, cfg_dir=cls.CFG_DIR, baseline=cls.BASELINE, arms=[cls.ARM])

    def test_all_four_classes_present(self):
        self.assertIn("source", self.ident)
        self.assertIn("config_and_model", self.ident)
        src = self.ident["source"]
        self.assertEqual(len(str(src["local_tree_digest"])), 64, "① 源码摘要")
        self.assertEqual(src["digest_scope"], ["src/ascend_opt", "benchmark", "scripts"])
        for arm in ("baseline", self.ARM):
            e = self.ident["config_and_model"][arm]
            self.assertEqual(len(str(e["config_digest"])), 64, f"② 配置摘要（{arm}）")
            self.assertTrue(e["model_revision"], f"③ 模型 revision（{arm}）")
            self.assertEqual(e["hardware_platform"], "mthreads", f"环境声明（{arm}）")

    def test_deployed_digest_is_not_claimed_to_be_current(self):
        """★ 旧远端部署戳**不得**被当成当前源码的身份。"""
        self.assertEqual(self.ident["source"]["deployed_digest"],
                         "PENDING_BATCH_STAMP")

    def test_platform_is_read_from_the_real_nested_path(self):
        """`hardware_platform` 在 `runtime` 段下。按顶层读会**静默取到 None**
        ⇒ 那条"设备记录与声明是否一致"的判据会一次都不执行。"""
        self.assertEqual(self.ident["declared_platform"], "mthreads")

    def test_missing_config_is_recorded_not_silently_zero(self):
        """配置不存在 ⇒ `exists=False` 且 `config_digest=None`（**不是**空串/0）。"""
        ident = B.four_class_identity(
            root=_ROOT, cfg_dir=self.CFG_DIR, baseline=self.BASELINE,
            arms=["definitely_not_an_arm"])
        e = ident["config_and_model"]["definitely_not_an_arm"]
        self.assertFalse(e["exists"])
        self.assertIsNone(e["config_digest"])
        self.assertTrue(e["error"], "加载失败必须留下原因")


class TestPreflightSequenceHasNoDrift(unittest.TestCase):
    """★ `plan` 与 `run` 必须走**同一张顺序表**（否则新步骤只出现在一边）。"""

    def test_plan_and_run_share_the_sequence(self):
        """实测过的漂移：给 `run` 加了指纹重采步骤后，`plan` 里**没有**它
        ⇒ "计划里没写"被读成"不会做"。⇒ 两者现在都调用 `_preflight_sequence`。"""
        src = (_ROOT / "scripts" / "exec_opt_batch.py").read_text(encoding="utf-8")
        self.assertIn("def _preflight_sequence(", src)
        # step_preflight 与 cmd_plan 都必须经由它
        self.assertIn("_preflight_sequence(ctx, dry=dry)", src)
        self.assertIn("_preflight_sequence(ctx, dry=True)", src)

    def test_plan_output_lists_the_fingerprint_step(self):
        """端到端：`plan` 的输出里**必须**能看见指纹步骤。"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = B.main(["plan", "--tag", "ENVCHK",
                         "--arms", "s01_both_s04_pkv",
                         "--baseline", "configs/mthreads/exec_opt/s01_both.yaml",
                         "--repeats", "5"])
        self.assertEqual(rc, 0)
        self.assertIn("重采环境指纹", buf.getvalue(),
                      "计划里必须出现指纹重采步骤（否则读者以为不会做）")


class TestExecopt5ArtifactsWereNotEdited(unittest.TestCase):
    """★ 登记事实：EXECOPT5 的原始产物**保持原样**（不改、不回填标签）。

    这条作为**回归守卫**存在：`hardware_env` 是旧的、`hostname_or_machine_id`
    是新的 —— 这个"矛盾"是**真实发生过的事**，必须留在产物里当证据。
    若有人事后"顺手改一致"，这条会变红，提醒他原始产物不得修改。
    """

    RUN = _ROOT / "results" / "raw" / "EXECOPT5-short_chat-s01_both"

    def setUp(self):
        if not (self.RUN / "meta.json").is_file():
            self.skipTest("EXECOPT5 产物不在本机（未回拉）⇒ 无法核验")
        import json

        self.meta = json.loads((self.RUN / "meta.json").read_text(encoding="utf-8"))

    def test_fingerprint_kept_as_is(self):  # noqa: D401
        hw = self.meta.get("hardware_env") or {}
        self.assertEqual(hw.get("collected_at"), OLD_COLLECTED,
                         "原始产物里的指纹时间戳必须保持 09-28（不改）")
        self.assertEqual(hw.get("hostname"), OLD_CONTAINER,
                         "原始指纹的 hostname 必须保持旧容器（不回填）")

    def test_live_machine_id_records_the_new_container(self):
        """同一份产物里，**实时**机器标识是**新**容器 ⇒ 矛盾被保留为证据。"""
        rid = (self.meta.get("reproducibility") or {}).get("hostname_or_machine_id")
        self.assertTrue(str(rid).startswith(NEW_CONTAINER),
                        f"hostname_or_machine_id 应是新容器，实际 {rid!r}")


if __name__ == "__main__":
    unittest.main()
