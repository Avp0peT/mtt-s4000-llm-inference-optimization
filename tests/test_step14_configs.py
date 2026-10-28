"""第 14 步的**配置验收**：冻结决定必须真的进了配置（第 09 步 §⑥ / 第 14 步）。

为什么要有这一条
----------------
实测踩到：第 09 步 §⑥ **白纸黑字冻结**了
「自适应臂在正式比较中的判据用 `controller.signal_criterion: decode_p90`」，
但第 14 步的 **9 个自适应配置全都漏了这个字段** ⇒ 加载器解析成**默认 `tpot_p90`**
⇒ 正式批次实际上用的**不是**冻结的判据。

⚠️ 这个失效**不会报错**：`signal_mode: layered` 仍然照常采集分层信号、轨迹里照样有
`decode_p90_ms`，只有"判据读哪一列"这一件事悄悄换掉了。产物完全正常（`MEASURED`、戳齐全、
账目误差 0），**看不出跑的不是冻结方案**。

⇒ 这类"冻结决定没进配置"只能靠**显式验收**来抓：本文件把它变成一条会失败的用例。
"""

from __future__ import annotations

import glob
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_benchmark import load_config  # noqa: E402

STEP14 = _ROOT / "configs" / "mthreads" / "phase1_step14"
#: 第 09 步 §⑥ 冻结值。改这里 = 改冻结方案 ⇒ 必须重新登记版本并重跑受影响的对照。
FROZEN_CRITERION = "decode_p90"


class TestStep14FrozenCriterion(unittest.TestCase):
    def test_dir_exists(self):
        self.assertTrue(STEP14.is_dir(), f"缺少 {STEP14}")

    def test_every_enabled_controller_declares_the_frozen_criterion(self):
        """★ 冻结决定必须**显式**落在配置里（靠默认值等于没冻结）。"""
        bad: list[str] = []
        for f in sorted(glob.glob(str(STEP14 / "*.yaml"))):
            cfg = load_config(f)
            name = Path(f).name
            if not cfg.controller.enabled:
                # 控制器关闭的臂（如 Best Static）不参与判据选择 ⇒ 豁免，但要**说明**
                continue
            if cfg.controller.signal_criterion != FROZEN_CRITERION:
                bad.append(f"{name}: {cfg.controller.signal_criterion!r}")
        self.assertEqual(bad, [], f"以下配置没有声明冻结判据 {FROZEN_CRITERION}：{bad}")

    def test_controller_arms_also_need_layered_mode(self):
        """`decode_p90` 依赖分层采集（否则那一列没有样本）⇒ 必须同时开 `layered`。"""
        for f in sorted(glob.glob(str(STEP14 / "*.yaml"))):
            cfg = load_config(f)
            if not cfg.controller.enabled:
                continue
            if cfg.controller.signal_criterion == "decode_p90":
                self.assertEqual(cfg.controller.signal_mode, "layered",
                                 f"{Path(f).name}: decode_p90 需要 signal_mode=layered")

    def test_only_one_variable_within_a_cell(self):
        """同一个 cell 内的三条**自适应**臂只差 `controller.kind`（单变量）。

        把"运行级参数"与"控制器参数"一起钉住：只盯 `kind` 会漏掉
        "顺手把 `target_tpot_ms` 也改了一点点"这类**单变量被破坏**的情况。

        ⚠️ **模型相关的键只在用模型的两臂之间比**（`cost` / `slo_weave`）：
        `threshold` 臂**根本不加载模型** ⇒ 它的 `cost_model_path` 是 `None` 不代表差异。
        （这条是实测校准出来的：加上 `cost_model_path` 后本用例先报了假警。）
        """
        ctrl_keys = ("signal_criterion", "signal_mode", "target_tpot_ms", "hysteresis",
                     "chunk_size_min", "chunk_size_max", "slo_weave_c_max",
                     "min_available_memory_mb", "critical_available_memory_mb")
        model_keys = ("cost_model_path", "cost_model_fingerprint_source")
        for cell in ("fixed", "ramp", "lenmix"):
            cfgs = {a: load_config(STEP14 / f"{cell}_{a}.yaml")
                    for a in ("threshold", "cost", "slo_weave")}
            base = cfgs["threshold"]
            for arm, c in cfgs.items():
                self.assertEqual(c.controller.kind, arm)
                for k in ctrl_keys:
                    self.assertEqual(getattr(c.controller, k), getattr(base.controller, k),
                                     f"{cell}_{arm}: controller.{k} 与 threshold 不一致")
                self.assertEqual(c.run.slo.tpot_ms, base.run.slo.tpot_ms)
                self.assertEqual(c.run.slo.ttft_ms, base.run.slo.ttft_ms)
                self.assertEqual(c.max_num_seqs, base.max_num_seqs)
                self.assertEqual(c.workload_name, base.workload_name)
                self.assertEqual(c.chunk.enabled, base.chunk.enabled)
                if arm == "threshold":
                    self.assertIsNone(c.controller.cost_model_path,
                                      f"{cell}_threshold 不加载模型，不应有 cost_model_path")
            # ★ 用模型的两臂必须指向**同一个**模型文件（否则四臂里混了两个模型）
            for k in model_keys:
                self.assertEqual(getattr(cfgs["cost"].controller, k),
                                 getattr(cfgs["slo_weave"].controller, k),
                                 f"{cell}: cost 与 slo_weave 的 {k} 不一致")


if __name__ == "__main__":
    unittest.main()
