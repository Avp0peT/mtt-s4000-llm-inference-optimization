"""软件指纹里**驱动 / MUSA 版本**的解析用例（R14）。

背景：这两个值原先只以**自由文本**形式躺在 `hardware_env.device.mthreads_gmi_info` 里，
没法程序化比对——而"这一批和上一批之间驱动换没换"正是跨批次可比性的前提。
仓库里 `parse_mthreads_software` 早就写好了（连边界都考虑过），**只是此前没人调用它**；
本文件把"接上了、而且接对"钉住。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.telemetry import RunContext, TelemetryCollector  # noqa: E402

#: 真机 `mthreads-gmi` 输出的形状（**逐字抄自**真机产物，不手写想象值）。
REAL_GMI = (
    "Fri Sep 25 13:26:04 2026\n"
    "---------------------------------------------------------------\n"
    "    mthreads-gmi:1.14.0          Driver Version:2.7.0\n"
    "---------------------------------------------------------------\n"
)


def _env(gmi: str | None):
    hw = {"platform": "mthreads", "device": {"mthreads_gmi_info": gmi}} if gmi is not None else {
        "platform": "mthreads", "device": {}}
    c = TelemetryCollector(
        run_dir="/tmp/_fp_probe",
        context=RunContext(run_id="probe", workload="w", config_path="c.yaml"),
        hardware_env=hw,
    )
    return c.software_env()


class TestSoftwareFingerprint(unittest.TestCase):
    def test_driver_version_is_parsed_from_the_raw_text(self):
        env = _env(REAL_GMI)
        self.assertEqual(env["mthreads_driver_version"], "2.7.0")

    def test_tool_version_is_not_mistaken_for_driver_version(self):
        """★ 反向对照：**只有工具版本行**时，驱动版本必须是 `None`。

        把 `mthreads-gmi:1.14.0` 当成驱动版本是**实测证实过**的错误
        （见 `parse_mthreads_software` 的 docstring）。少了这条用例，
        "少一条信息"就会变成"多一条错信息"，而且看起来像一次成功的采集。
        """
        env = _env("    mthreads-gmi:1.14.0\n")
        self.assertIsNone(env["mthreads_driver_version"])

    def test_musa_version_is_none_when_the_text_has_no_such_field(self):
        """真机文本里**没有**独立的 MUSA SDK 行 ⇒ 必须是 `None`（不猜、不写 "unknown"）。"""
        env = _env(REAL_GMI)
        self.assertIsNone(env["musa_version"])

    def test_missing_device_does_not_raise_and_stays_none(self):
        """没有设备信息时**不抛错**，两个值都是 `None` ——
        "不可测"要保持 `None`，不能变成 `"unknown"` 那种看起来成功的值。"""
        env = _env(None)
        self.assertIsNone(env["mthreads_driver_version"])
        self.assertIsNone(env["musa_version"])

    def test_existing_keys_survive(self):
        """加这两项**不得**弄丢原有键（它们已进了既跑产物的 schema）。"""
        env = _env(REAL_GMI)
        for k in ("python", "torch_version", "transformers_version", "ascend_opt_version"):
            self.assertIn(k, env)


if __name__ == "__main__":
    unittest.main()
