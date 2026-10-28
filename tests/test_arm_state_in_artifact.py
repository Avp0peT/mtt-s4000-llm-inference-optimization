"""`arm_state` 必须落盘：**"数不出来"不等于"没发生"**。

背景（实测踩到，2026-09-26）
---------------------------
`n_degraded`（"有预测但全部超预算 ⇒ 取最省那个"）与 `n_prefill_starved`
（"有 prefill 积压、而所有候选本步都推进不了 prefill"）是**动作方向全错**的唯一直接度量，
但它们原先**没有进产物**（`CostPolicyState.describe()` 有这些键，只是没人把它并进去）。

⇒ 后果不是"少一列"，而是**只能靠猜**：我从 `S14D` 的 D0 可行率 5.7% 推出
"结构上无法发 prefill"，而 `S14B1` 的遥测 `note` 显示 **183/292 步仍在发
`prefill_tok=1024`**（全无预测时走 `_fallback()` 回退到阈值控制器，它照常发）。
**猜错了，而且错得方向相反。**

本文件钉住四件事：
① 没有 `state` 的臂返回 `{}`（**不虚构**键）；
② 有 `state` 的臂把 `describe()` 的内容带出来；
③ `describe()` 抛错时错误**显形**（不是静默吞掉 —— 静默吞掉与"没这一项"无法区分）；
④ 两条执行路径**共用同一个函数**（否则又是一边修一边不修）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.arm import arm_state  # noqa: E402
from ascend_opt.execution import ExecutionResult  # noqa: E402
from ascend_opt.cost_policy import CostPolicyState  # noqa: E402


class _NoState:
    """静态臂：没有 `state`。"""


class _WithState:
    def __init__(self, payload=None, boom=False):
        self.state = _S(payload, boom)


class _S:
    def __init__(self, payload, boom):
        self._p, self._boom = payload, boom

    def describe(self):
        if self._boom:
            raise RuntimeError("计数器坏了")
        return self._p


class TestArmStateExtraction(unittest.TestCase):
    def test_arm_without_state_returns_empty_not_invented_keys(self):
        self.assertEqual(arm_state(_NoState()), {},
                         "没有 state 的臂不得虚构键（虚构会让'没测'看起来像'测到了 0'）")

    def test_arm_with_state_exposes_all_counters(self):
        st = CostPolicyState()
        st.n_decisions, st.n_degraded, st.n_prefill_starved = 10, 3, 2
        got = arm_state(_WithState(st.describe()))
        for k in ("n_decisions", "n_degraded", "n_prefill_starved", "n_fallback", "n_idle"):
            self.assertIn(k, got, f"缺 {k} ⇒ 又回到'靠猜动作方向'")

    def test_broken_describe_is_visible_not_swallowed(self):
        got = arm_state(_WithState(boom=True))
        self.assertIn("_error", got, "度量自己坏掉必须显形（否则与'没这一项'无法区分）")
        self.assertIn("计数器坏了", got["_error"])

    def test_non_dict_result_is_flagged(self):
        got = arm_state(_WithState(payload=["not", "a", "dict"]))
        self.assertIn("_error", got)


class TestExecutionResultCarriesArmState(unittest.TestCase):
    def test_describe_includes_arm_state(self):
        r = ExecutionResult(metrics=[], samples=[], wall_ms=1.0, steps=1, dropped=0,
                            controller_trace=[], arm_state={"n_degraded": 7})
        self.assertEqual(r.describe()["arm_state"], {"n_degraded": 7})

    def test_default_is_empty_dict(self):
        r = ExecutionResult(metrics=[], samples=[], wall_ms=1.0, steps=1, dropped=0,
                            controller_trace=[])
        self.assertEqual(r.describe()["arm_state"], {},
                         "默认应为空字典（不是 None）：下游按 dict 读，None 会当场炸")


class TestControllerEffectMergesArmState(unittest.TestCase):
    def test_effect_contains_arm_state(self):
        from run_benchmark import _controller_effect

        class _R:
            samples: list = []
            controller_trace: list = []
            arm_state = {"n_degraded": 5, "n_prefill_starved": 1}

        eff = _controller_effect(_R())
        self.assertEqual(eff["arm_state"], {"n_degraded": 5, "n_prefill_starved": 1})

    def test_effect_tolerates_missing_attribute(self):
        """旧结果对象（没有 `arm_state` 属性）不得让汇总炸掉。"""
        from run_benchmark import _controller_effect

        class _R:
            samples: list = []
            controller_trace: list = []

        self.assertEqual(_controller_effect(_R())["arm_state"], {})


class TestBothPathsShareOneImplementation(unittest.TestCase):
    def test_simulator_and_real_backend_use_the_same_function(self):
        """★ 两条路径必须是**同一个函数对象**（第 13 步踩过：只接一边 ⇒ 产物看起来正常）。"""
        import real_backend
        import run_benchmark
        self.assertIs(run_benchmark.arm_state, real_backend.arm_state,
                      "两条路径各拿了一份 ⇒ 以后修一边另一边不会跟着修")


if __name__ == "__main__":
    unittest.main()
