"""两条执行路径**必须共用同一份臂实现**（第 13 步的教训）。

为什么需要这条用例
------------------
第 13 步实测踩到：策略臂只被接进模拟器（`simulate()`），真机路径（`real_backend`）
没有 ⇒ 真机上 `slo_weave` 决策数 = **0**、两臂产物**逐项相同**，
而配置里明明写着 `controller.kind: slo_weave`。
那批产物**看起来完全正常**（`MEASURED`、戳齐全、账目自洽）——
正是 `step_planner` 的 docstring 警告过的"不会报错的失效"。

⇒ 决策逻辑只有一处（`ascend_opt.arm.apply_arm`）、构造也只有一处
（`benchmark/arm_factory.build_arm`）。本文件把这两件事钉住。
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arm_factory  # noqa: E402
import real_backend  # noqa: E402
import run_benchmark  # noqa: E402
from ascend_opt import apply_arm  # noqa: E402


class TestSharedArmImplementation(unittest.TestCase):
    def test_both_paths_call_the_shared_decider(self):
        """★ 模拟器与真机都必须调用 `apply_arm`。"""
        for mod in (run_benchmark, real_backend):
            src = inspect.getsource(mod)
            self.assertIn("apply_arm(", src,
                          f"{mod.__name__} 没有调用共享的 apply_arm ⇒ 两条路径会分叉")
            self.assertIn("from ascend_opt import (", src)
            self.assertIn("    apply_arm,", src,
                          f"{mod.__name__} 导入了但没在 from-import 里列出 apply_arm")

    def test_both_paths_build_the_arm_through_the_shared_factory(self):
        """★ 构造也必须共用（否则一边改了口径、另一边不会跟着改）。"""
        for mod in (run_benchmark, real_backend):
            src = inspect.getsource(mod)
            self.assertIn("build_arm(cfg)", src,
                          f"{mod.__name__} 没有用共享工厂构造策略臂")

    def test_only_one_implementation_of_the_slo_weave_semantics(self):
        """★ 语义只能有一份。

        这条抓的是"在调用方里再写一遍语义"——例如把 decode-only 写成
        `cur_max_tokens = ...`、把"设值"写成 `min(...)`。
        两条路径若各写一遍，改了 A 不改 B 时**不会报错**，只会让两臂产物看起来相同。
        """
        for mod in (run_benchmark, real_backend):
            src = inspect.getsource(mod)
            self.assertNotIn("decode_only", src,
                             f"{mod.__name__} 里出现了 decode-only 的语义 ⇒ 应移进 apply_arm")
            self.assertNotIn("slo_weave\"", src.replace('"slo_weave"', "", 1),
                             f"{mod.__name__} 里出现了 SLOWeave 的分支语义")

    def test_apply_arm_is_exported(self):
        """两条路径都从 `ascend_opt` 取它 ⇒ 必须真的导出。"""
        self.assertTrue(callable(apply_arm))
        import ascend_opt
        self.assertIn("apply_arm", ascend_opt.__all__)


class TestFactoryContracts(unittest.TestCase):
    def test_threshold_kind_builds_no_arm(self):
        """`threshold` ⇒ 返回 `None`（既有行为逐字节不变）。"""
        class C:
            class controller:
                kind = "threshold"
                enabled = True
        self.assertIsNone(arm_factory.build_arm(C()))

    def test_disabled_controller_builds_no_arm(self):
        class C:
            class controller:
                kind = "slo_weave"
                enabled = False
        self.assertIsNone(arm_factory.build_arm(C()))

    def test_offline_fingerprint_source_does_not_fake_a_check(self):
        """`declared` ⇒ 走 `load_offline`，并在产物里**明说"没做指纹校验"**。

        本机（torch 2.10）与标定环境（torch 2.2）必然不符 ⇒ 若它走了严格路径，
        这里就会抛异常。反过来，`declared` 下能构造成功、且 `relaxations` 里有那句话，
        说明"明说未校验"这条纪律生效。
        """
        class R:
            model = "Qwen/Qwen2.5-7B-Instruct"
            model_revision = "a09a35458c702b33eeacc393d103063234e8bc28"
            precision = "fp16"

        class C:
            runtime = R()

            class controller:
                kind = "slo_weave"
                enabled = True
                cost_model_path = str(_ROOT / "configs" / "cost_calib" / "model_v1.json")
                cost_model_fingerprint_source = "declared"
                slo_weave_c_max = 4096
                slo_weave_tpot_target_ms = None
                target_tpot_ms = 120.0

        arm = arm_factory.build_arm(C())
        self.assertIsNotNone(arm)
        self.assertTrue(any("未做指纹校验" in r for r in arm.model.relaxations),
                        arm.model.relaxations)


if __name__ == "__main__":
    unittest.main()
