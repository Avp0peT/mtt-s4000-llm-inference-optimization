#!/usr/bin/env python
"""真机数值验收入口（`--driver auto`）的**身份与设备守卫**回归。

★ 钉住的失效形态
------------------
`verify_s01_numeric_acceptance.py` 的 `--driver` **默认值是 `cpu`**。
⇒ 在真机上**漏写** `--driver auto` 会**静默**跑 CPU 微型驱动，产物标成
   `LOCAL_CPU_MICRO` —— 也就是**把本机证据当成 S4000 证据**。
这是本项目最危险的一类错（不报错、不留痕，还会变成下游结论的依据）。

⇒ 本文件断言：
  ① `--require-real-hw` 在非 auto 时**必须拒绝**（防漏写）；
  ② 产物的**证据等级**在 CPU 路径上**不得**被读成真机证据（有显式 scope 字段）；
  ③ 身份绑定覆盖**源码 / 配置 / 模型 / 环境**四类（配置用 sha256 字节级）；
  ④ 真机三项核验（MUSA 设备 / resolved revision / 优化生效）的判据是**拒绝式**的
     —— 用一个**假**的 arm 事实字典喂进去，必须**报出问题**（阴性对照）。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = _ROOT / "scripts" / "verify_s01_numeric_acceptance.py"
CFG_A = _ROOT / "configs" / "mthreads" / "exec_opt" / "s01_reference.yaml"
CFG_B = _ROOT / "configs" / "mthreads" / "exec_opt" / "s01_both.yaml"


def _load_module():
    """按路径加载被测算脚本（它的 `_hardware_problems` 需要直接单测）。"""
    spec = importlib.util.spec_from_file_location("_vsa", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_vsa"] = m
    spec.loader.exec_module(m)
    return m


class TestRequireRealHwGuard(unittest.TestCase):
    """★ 防"漏写 `--driver auto`"⇒ 把本机证据当 S4000 证据。"""

    def test_rejects_non_auto(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--require-real-hw",
                            "--config-a", str(CFG_A), "--config-b", str(CFG_B)],
                           cwd=str(_ROOT), capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 2, "非 auto 时没有拒绝 ⇒ 守卫失效")
        self.assertIn("--driver auto", r.stderr)
        self.assertIn("cpu", r.stderr, "必须点明当前默认值是 cpu")

    def test_default_driver_is_cpu(self):
        """默认值是 cpu —— 这正是危险之处，必须被文档与守卫同时覆盖。"""
        r = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                           cwd=str(_ROOT), capture_output=True, text=True, timeout=120)
        self.assertIn("默认", r.stdout)
        self.assertIn("cpu", r.stdout)

    def test_real_hw_requires_unique_output_path(self):
        for extra in ([], ["--json", "unused.json", "--allow-overwrite"]):
            with self.subTest(extra=extra):
                r = subprocess.run([sys.executable, str(SCRIPT), "--require-real-hw",
                                    "--driver", "auto", "--config-a", str(CFG_A),
                                    "--config-b", str(CFG_B), *extra], cwd=str(_ROOT),
                                   capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 2)
                self.assertIn("唯一 --json", r.stderr)


class TestEvidenceLevelNotUpgraded(unittest.TestCase):
    """★ CPU 证据等级**不得**被升级为 S4000 证据。"""

    def test_docstring_says_scope_is_interface_only(self):
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("LOCAL_CPU_MICRO", src)
        self.assertIn("不得", src)
        self.assertIn("接口与判据语义", src)

    def test_help_text_declares_default_is_cpu_and_auto_is_real(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                           cwd=str(_ROOT), capture_output=True, text=True, timeout=120)
        self.assertIn("auto=真机", r.stdout)
        self.assertIn("cpu=真实 CPU 微型驱动", r.stdout)


class TestHardwareProblemsIsRejective(unittest.TestCase):
    """★ 真机三项核验必须是**拒绝式**的（阴性对照：喂假的 arm 事实必须报错）。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _load_module()
        from benchmark.run_benchmark import load_config      # noqa: E402
        cls.cfg = load_config(CFG_B)

    def _arm(self, *, hw=True, device="musa:0", rev="a09a35458c702b33eeacc393d103063234e8bc28",
             kind="transformers", skipped=2, enabled=True):
        return {"config": "x.yaml", "hardware_backed": hw,
                "driver_identity": {"kind": kind, "device": device,
                                    "model_revision": rev, "model_revision_resolved": rev},
                "execution_optimizations": {"sample_needed_only": {
                    "requested": enabled, "effective": enabled,
                    "n_sampling_rows_skipped": skipped}, "logits_last_only": {
                    "requested": enabled, "effective": enabled,
                    "n_calls_last_only": 12 if enabled else 0}}}

    def _problems(self, treatment=None, baseline=None):
        return self.m._hardware_problems(
            None, self.cfg, [baseline or self._arm(enabled=False, skipped=0),
                            treatment or self._arm()], driver="auto")

    def test_cpu_driver_is_out_of_scope_not_pass(self):
        """非真机驱动 ⇒ 本项**不适用**（返回空 ⇒ 不产生问题，但也未核验）。"""
        probs = self.m._hardware_problems(None, self.cfg, [self._arm()], driver="cpu")
        self.assertEqual(probs, [], "CPU 路径不该报真机问题（本项不适用）")

    def test_clean_real_hw_passes(self):
        probs = self._problems()
        self.assertEqual(probs, [], f"干净真机事实却报错：{probs}")

    def test_baseline_unexpected_optimization_is_rejected(self):
        probs = self._problems(baseline=self._arm())
        self.assertTrue(any("对照" in p and "角色" in p for p in probs), probs)

    def test_treatment_disabled_switch_is_rejected(self):
        for key in ("logits_last_only", "sample_needed_only"):
            arm = self._arm()
            arm["execution_optimizations"][key]["effective"] = False
            probs = self._problems(treatment=arm)
            self.assertTrue(any(key in p and "未真正生效" in p for p in probs), probs)

    def test_treatment_requires_exact_false_row_count(self):
        for skipped in (0, 1, 3, None, True):
            probs = self._problems(treatment=self._arm(skipped=skipped))
            self.assertTrue(any("n_sampling_rows_skipped" in p for p in probs), probs)

    def test_treatment_last_only_must_execute(self):
        arm = self._arm()
        arm["execution_optimizations"]["logits_last_only"]["n_calls_last_only"] = 0
        probs = self._problems(treatment=arm)
        self.assertTrue(any("n_calls_last_only" in p for p in probs), probs)

    def test_two_musa_device_ids_are_not_matched(self):
        probs = self._problems(treatment=self._arm(device="musa:1"))
        self.assertTrue(any("设备不一致" in p for p in probs), probs)

    def test_not_hardware_backed_is_caught(self):
        probs = self.m._hardware_problems(None, self.cfg, [self._arm(hw=False)],
                                          driver="auto")
        self.assertTrue(any("hardware_backed" in p for p in probs))

    def test_non_musa_device_is_caught(self):
        """★ 配置写 musa:0 ≠ 真的跑在 MUSA 上。"""
        probs = self.m._hardware_problems(None, self.cfg, [self._arm(device="cpu")],
                                          driver="auto")
        self.assertTrue(any("musa" in p.lower() for p in probs),
                        f"非 MUSA 设备没被抓到：{probs}")

    def test_missing_resolved_revision_is_caught(self):
        a = self._arm()
        a["driver_identity"]["model_revision_resolved"] = None
        probs = self.m._hardware_problems(None, self.cfg, [a], driver="auto")
        self.assertTrue(any("resolved" in p for p in probs))

    def test_mismatched_revision_is_caught(self):
        """★ 配置**请求**的 revision 与**实际 resolved** 不等 ⇒ 跑的不是同一份权重。"""
        a = self._arm()
        a["driver_identity"]["model_revision_resolved"] = "0" * 40
        probs = self.m._hardware_problems(None, self.cfg, [a], driver="auto")
        self.assertTrue(any("不是同一份权重" in p for p in probs))

    def test_zero_skipped_means_optimization_did_not_take_effect(self):
        """★ 配置开了 ≠ 走了。跳过计数为 0 ⇒ 优化未真正生效。"""
        probs = self._problems(treatment=self._arm(skipped=0))
        self.assertTrue(any("未真正生效" in p for p in probs))

    def test_wrong_driver_kind_is_caught(self):
        probs = self.m._hardware_problems(None, self.cfg, [self._arm(kind="fake")],
                                          driver="auto")
        self.assertTrue(any("kind" in p for p in probs))


class TestIdentityBindingPresent(unittest.TestCase):
    """★ 产物必须绑定**源码 / 配置 / 模型 / 环境**四类身份。"""

    def test_payload_has_all_four_identity_classes(self):
        src = SCRIPT.read_text(encoding="utf-8")
        for k in ('"source":', '"config":', '"model":', '"environment":'):
            self.assertIn(k, src, f"payload 缺身份类 {k}")

    def test_config_identity_is_byte_level(self):
        """配置身份必须是 sha256（字节级），不是"看着一样"。"""
        self.assertIn("_file_sha256", SCRIPT.read_text(encoding="utf-8"))

    def test_overwrite_is_refused_by_default(self):
        src = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("拒绝覆盖", src)
        self.assertIn("--allow-overwrite", src)

    def test_source_identity_preserves_unknown_dirty_and_stamp(self):
        m = _load_module()
        stamp = {"validated": None, "reason": "TEST_DATA no deployment"}
        with patch.object(m, "detect_source_identity", return_value=("", "none", None, None)), \
                patch.object(m, "compute_tree_digest", return_value="abc"), \
                patch.object(m, "read_source_stamp", return_value=stamp):
            out = m._git_identity()
        self.assertIsNone(out["git_dirty"])
        self.assertIsNone(out["git_commit"])
        self.assertEqual(out["tree_digest"], "abc")
        self.assertEqual(out["stamp"], stamp)

    def test_environment_retains_hash_but_never_imports_other_host_versions(self):
        m = _load_module()
        data = {"hostname": "TEST_REMOTE", "available": True,
                "collected_at": "2026-10-03T00:00:00+0800",
                "discovery": {"values": {"MTT_DRIVER_VERSION": "test-driver",
                                          "MUSA_VERSION": "test-musa"}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with patch.object(m.platform, "node", return_value="TEST_LOCAL"):
                out = m._environment_snapshot(path)
            self.assertFalse(out["fingerprint_matches_host"])
            self.assertIsNone(out["driver_version"])
            self.assertIsNone(out["musa_version"])
            self.assertEqual(out["fingerprint_sha256"], m._file_sha256(path))
            self.assertEqual(out["fingerprint"]["collected_at"], data["collected_at"])
            with patch.object(m.platform, "node", return_value="TEST_REMOTE"):
                same_host = m._environment_snapshot(path)
            self.assertEqual(same_host["driver_version"], "test-driver")
            self.assertEqual(same_host["musa_version"], "test-musa")
            self.assertEqual(same_host["fingerprint_freshness"], "not_verified")


class TestNumericAcceptanceJsonIsStrict(unittest.TestCase):
    """★★ 要求 ④（用户 2026-10-03 点名）：非有限诊断值 → `null` + 错误说明，**严格** JSON。

    ⚠️ 为什么"严格"是独立要求：Python 默认 `json.dumps(allow_nan=True)` 会写出
       `NaN` / `Infinity` / `-Infinity` —— **非标准 JSON**（RFC 8259 不允许）。
       写的一方看不到任何异常，读的一方（任何严格解析器）全部拒收。
    """

    @classmethod
    def setUpClass(cls):
        cls.m = _load_module()

    def test_sanitizer_nulls_and_records(self):
        clean, notes = self.m.sanitize_json(
            {"a": float("nan"), "b": [float("inf")], "c": {"d": float("-inf")}, "keep": 3.0})
        self.assertIsNone(clean["a"])
        self.assertIsNone(clean["b"][0])
        self.assertIsNone(clean["c"]["d"])
        self.assertEqual(clean["keep"], 3.0)
        self.assertEqual(len(notes), 3, notes)

    def test_output_is_strict_json(self):
        import json as _json
        clean, _ = self.m.sanitize_json({"x": float("inf"), "y": float("nan")})
        text = _json.dumps(clean, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("Infinity", text)
        self.assertNotIn("NaN", text)
        self.assertEqual(_json.loads(text), {"x": None, "y": None})

    def test_negative_control_without_sanitizer_raises(self):
        """破坏性对照：不清洗 ⇒ `allow_nan=False` 必须抛 ⇒ 证明清洗是必需的。"""
        import json as _json
        with self.assertRaises(ValueError):
            _json.dumps({"x": float("nan")}, allow_nan=False)

    def test_script_writes_with_allow_nan_false(self):
        """静态核对：入口**必须**用 `allow_nan=False`（否则清洗可被绕过）。"""
        self.assertIn("allow_nan=False", SCRIPT.read_text(encoding="utf-8"))

    def test_bool_is_not_treated_as_number(self):
        """`True` 是 `int` 的子类 ⇒ 若按 `isinstance(x, float)` 之外的写法会误判。"""
        clean, notes = self.m.sanitize_json({"flag": True, "s": None})
        self.assertIs(clean["flag"], True)
        self.assertIsNone(clean["s"])
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
