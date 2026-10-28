"""环境指纹读取的测试：**可用性是"文件声明的"还是"这里推出来的"，必须可区分**。

这组测试守的是一个具体的分裂（评审 #1）：

    运行期认为硬件可用（遥测在内存里按探测结果推出 ``available``）
    导出时却拒绝   （闸门只认文件里**显式声明**的 ``available``）

两份判断读的是两份东西，所以能长期并存而不报错——直到导出失败，而且失败信息
指不出原因。修法不是"让闸门也去推"，而是把**来源**记下来：``available_source``
为 ``derived`` 就说明"这次 run 的可用性没有文件背书"。

因此这里不测"能不能读出 True"，测的是**来源标注是否正确**——那才是修复点。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.telemetry import (  # noqa: E402
    AVAILABILITY_SOURCE_DECLARED,
    AVAILABILITY_SOURCE_DERIVED,
    AVAILABILITY_SOURCE_MISSING,
    AVAILABILITY_SOURCE_UNREADABLE,
    ENV_FILENAME,
    hardware_env_candidates,
    load_hardware_env,
)


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


class TestHardwareEnvCandidates(unittest.TestCase):
    """候选路径只有一处定义——判定、落盘、事后核对必须看到**同一份**指纹。"""

    def test_run_dir_first_then_shared(self):
        run_dir = Path("C:/tmp/raw/20260101-x")
        candidates = hardware_env_candidates(run_dir)
        self.assertEqual(
            candidates,
            [
                run_dir / ENV_FILENAME,
                run_dir.parent / "_env" / ENV_FILENAME,
            ],
        )


class TestLoadHardwareEnv(unittest.TestCase):
    def test_missing_fingerprint_is_unknown_not_unavailable(self):
        """找不到指纹 = "没查过"，**不是** "这台机器没有 NPU"。

        两者混同会让"环境采集"看起来已经做完，于是永远不会被补上；
        而 ``available=None`` 会一路传到标签判定，把 run 判成 NOT_MEASURED ——
        这个结果是对的，但原因必须显式写出来。
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIsNone(payload["available"])
        self.assertEqual(payload["available_source"], AVAILABILITY_SOURCE_MISSING)
        self.assertIn("inspect_env.sh", payload["reason"])

    def test_new_format_is_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            _write(run_dir / ENV_FILENAME, {
                "schema_version": 2,
                "platform": "ascend",
                "available": True,
                "probes": {"ascend": {"result": True}},
            })
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIs(payload["available"], True)
        self.assertEqual(payload["available_source"], AVAILABILITY_SOURCE_DECLARED)

    def test_new_format_null_stays_null(self):
        """新格式里 ``available=null``（探测没结论）不得被推成 True 或 False。"""
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            _write(run_dir / ENV_FILENAME, {
                "schema_version": 2, "platform": "unknown",
                "available": None,
                "probes": {"ascend": {"result": None}, "mthreads": {"result": None}},
            })
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIsNone(payload["available"])
        # 关键：来源是 declared。字段写了就是声明，哪怕值是 null ——
        # 这条区别决定了导出闸门该报"没结论"还是"旧格式该重采"。
        self.assertEqual(payload["available_source"], AVAILABILITY_SOURCE_DECLARED)

    def test_old_format_is_derived_and_marked_as_such(self):
        """旧指纹没有 ``available``：这里推出 True，但**必须标注是推的**。

        这就是那个分裂的来源。标成 ``derived`` 之后，下游能一眼看出
        "这次 run 的可用性没有文件背书"，而不是等导出时才被打回。
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            _write(run_dir / ENV_FILENAME, {
                "schema_version": 1,
                "device": {"npu_smi_available": True, "npu_smi_info": "npu-smi 23.0.3"},
            })
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIs(payload["available"], True)
        self.assertEqual(payload["available_source"], AVAILABILITY_SOURCE_DERIVED)

    def test_old_format_probes_block_is_used_when_there_is_no_available_key(self):
        """中间格式：有 ``probes`` 但没有顶层 ``available``，按同一套规则合成。

        与 ``inspect_env.sh`` 用同一个 :func:`normalize_availability`，
        否则"脚本合成的可用性"与"遥测合成的可用性"会在同一份文件上给出不同答案。
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            _write(run_dir / ENV_FILENAME, {
                "probes": {
                    "ascend": {"result": True},
                    "mthreads": {"result": False},
                },
            })
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIs(payload["available"], True)
        self.assertEqual(payload["available_source"], AVAILABILITY_SOURCE_DERIVED)

    def test_nothing_to_derive_from_is_unknown_not_false(self):
        """旧指纹里连设备探测都没有 -> ``None``。默认成 False 也错。

        ``False`` 是"查了、没有"，``None`` 是"没有可查的东西"。写成 False
        会让"该去采指纹"变成"这台机器不能用"。
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            _write(run_dir / ENV_FILENAME, {"hostname": "somewhere"})
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIsNone(payload["available"])
        self.assertEqual(payload["available_source"], AVAILABILITY_SOURCE_DERIVED)

    def test_run_dir_wins_over_the_shared_fingerprint(self):
        """run 目录里的指纹优先——每次运行绑定自己那一刻的环境。

        共享位置是时序妥协（指纹必须在 run 开始前采），一旦这个 run 留下了
        自己的快照，就必须以它为准，否则同一台机器上先后两次运行会共用
        一份可能与本次不符的指纹。
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            _write(run_dir.parent / "_env" / ENV_FILENAME, {
                "schema_version": 2, "platform": "ascend", "available": True,
            })
            _write(run_dir / ENV_FILENAME, {
                "schema_version": 2, "platform": "mthreads", "available": False,
            })
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIs(payload["available"], False)
        self.assertIn("mthreads", payload["platform"])

    def test_shared_fingerprint_is_the_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            _write(run_dir.parent / "_env" / ENV_FILENAME, {
                "schema_version": 2, "platform": "ascend", "available": True,
            })
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIs(payload["available"], True)
        self.assertEqual(payload["available_source"], AVAILABILITY_SOURCE_DECLARED)

    def test_unreadable_fingerprint_is_flagged_not_silently_missing(self):
        """文件在、但解析不了 -> ``unreadable``（False + 来源标注）。

        与"文件不在"分开：前者说明采集这一步跑了但产物坏了（该重采），
        后者说明采集压根没跑（该去跑）。报同一句话会让人去修错的地方。
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            _write(run_dir / ENV_FILENAME, "{ this is not json")
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIs(payload["available"], False)
        self.assertEqual(payload["available_source"], AVAILABILITY_SOURCE_UNREADABLE)
        self.assertIn("解析失败", payload["reason"])

    def test_declared_availability_is_not_overridden_by_derivation(self):
        """文件显式声明了就以文件为准，不拿探测结果去"纠正"它。

        反过来的话，闸门读文件、运行期读推导，两者会在同一份文件上得出不同结论 ——
        正是要修的那个分裂。
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "raw" / "20260101-x"
            run_dir.mkdir(parents=True)
            _write(run_dir / ENV_FILENAME, {
                "available": True,
                # 探测结果说全是 False —— 显式声明仍然优先。
                "device": {"npu_smi_available": False},
                "probes": {"ascend": {"result": False}},
            })
            payload = load_hardware_env(hardware_env_candidates(run_dir))

        self.assertIs(payload["available"], True)
        self.assertEqual(payload["available_source"], AVAILABILITY_SOURCE_DECLARED)


if __name__ == "__main__":
    unittest.main()
