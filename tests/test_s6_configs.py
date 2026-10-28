"""§S6 服务收益档配置的验收：**三臂只差两个开关**、**组合合法**、**负载已冻结**。

为什么必须有这一组用例
----------------------
S6 的全部价值在于「三臂在同一负载、同一源码、同一 SLO 下只差
`controller.enabled` / `controller.kind`」—— 否则"哪条臂更好"无法归因。
3 份配置是生成的，但生成器只在**生成那一刻**对；之后任何人手改一份，
"只差两个开关"就悄悄破了，而产出的对比表看起来完全正常。

三条硬约束
----------
1. **负载已冻结**：三臂都必须是 `concurrency=2` / `max_num_seqs=4`
   （§7d 第 5 节机械选出的组合）。任一份漂回别的值 ⇒ 服务收益结论作废。
2. **三臂只差两个开关**：`static` 与另两臂逐字段相同，只有
   `controller.enabled` / `controller.kind`（+ `notes`）不同。
3. **组合必须能被真加载器接受**：`ControllerConfig.__post_init__` 规定
   `kind != "threshold"` 时 `enabled` 必须为 True
   ⇒ `static` 臂必须写 `kind: threshold`（**不是**留底稿的 `cost`）。
   ★ 这条是**实测踩到的**：只做文本级校验（数 `enabled: true` 出现次数）
   放过了 `kind: cost` + `enabled: false`，真加载器一构造就抛错。
   ⇒ 因此本文件**必须**过一遍 `load_config` / `ControllerConfig`，
   不能只比文本。

与 `test_d1_scan_configs.py` 的关系：那组钉的是**扫描批**（phase1_d1_scan/），
本组钉的是**服务收益档**（phase1_s6/）。reference 不同，**故意分开**。
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_benchmark import load_config  # noqa: E402

import gen_phase1_s6_configs as G  # noqa: E402

S6 = G.OUT_DIR
BASE = G.REFERENCE

#: §7d 第 5 节机械选出的冻结组合。**写死**：这是预注册的一部分，
#: 不该由生成器常量"顺手"决定（否则改网格会静默改掉 S6 的负载）。
FROZEN_CONCURRENCY = 2
FROZEN_MAX_NUM_SEQS = 4


def _load(arm: str):
    return load_config(S6 / f"{G.cell_name(arm)}.yaml")


class TestArmsExistAndAreNamed(unittest.TestCase):
    def test_three_arms(self):
        self.assertEqual([a for a, *_ in G.ARMS], ["static", "threshold", "cost"])

    def test_all_files_exist(self):
        for arm, *_ in G.ARMS:
            f = S6 / f"{G.cell_name(arm)}.yaml"
            self.assertTrue(f.is_file(), f"缺少 {f.name}")

    def test_arm_name_in_filename(self):
        names = [f"{G.cell_name(a)}.yaml" for a, *_ in G.ARMS]
        self.assertEqual(len(names), len(set(names)), f"文件名有重复：{names}")
        for arm, name in zip([a for a, *_ in G.ARMS], names):
            self.assertIn(arm, name)


class TestFrozenWorkload(unittest.TestCase):
    """★ 三臂的负载必须都是 §7d 冻结组合；漂了就没有可比性。"""

    def test_concurrency_and_max_seqs_frozen(self):
        for arm, *_ in G.ARMS:
            cfg = _load(arm)
            with self.subTest(arm=arm):
                self.assertEqual(
                    int(cfg.workload_overrides.get("concurrency", -1)),
                    FROZEN_CONCURRENCY,
                    f"{arm}: concurrency 应为 {FROZEN_CONCURRENCY}")
                self.assertEqual(int(cfg.max_num_seqs), FROZEN_MAX_NUM_SEQS,
                                 f"{arm}: max_num_seqs 应为 {FROZEN_MAX_NUM_SEQS}")

    def test_prompt_and_gen_frozen(self):
        for arm, *_ in G.ARMS:
            cfg = _load(arm)
            with self.subTest(arm=arm):
                for k, v in (("prompt_len_min", 1536), ("prompt_len_max", 1536),
                             ("gen_len_min", 32), ("gen_len_max", 32)):
                    self.assertEqual(int(cfg.workload_overrides.get(k, -1)), v,
                                     f"{arm}: workload.{k} 应为 {v}")
                self.assertEqual(cfg.workload_name, "short_chat")

    def test_slo_frozen(self):
        for arm, *_ in G.ARMS:
            cfg = _load(arm)
            with self.subTest(arm=arm):
                self.assertEqual(cfg.run.slo.ttft_ms, 2000)
                self.assertEqual(cfg.run.slo.tpot_ms, 100.0)


class TestOnlyTwoSwitchesDiffer(unittest.TestCase):
    """三臂两两对比：除 `controller` 的两个开关外，逐字段相同。"""

    def test_reference_workload_unchanged(self):
        """底稿（§7d 的 c2_m4）除 controller 两个开关外一律不动。"""
        ref = _load("static")   # 以最简臂当基准
        for arm, _, _, _ in G.ARMS:
            cfg = _load(arm)
            with self.subTest(arm=arm):
                self.assertEqual(cfg.workload_name, ref.workload_name)
                self.assertEqual(cfg.chunk, ref.chunk, f"{arm}: chunk 段被改了")
                self.assertEqual(cfg.chunk_policy_name, ref.chunk_policy_name)
                self.assertEqual(cfg.scheduler_name, ref.scheduler_name)
                self.assertEqual(cfg.runtime, ref.runtime, f"{arm}: runtime 段被改了")
                for k in ("repeats", "warmup_s", "control_interval_steps"):
                    self.assertEqual(getattr(cfg.run, k), getattr(ref.run, k),
                                     f"{arm}: run.{k} 被改了")
                # 负载其余字段
                for k, v in ref.workload_overrides.items():
                    self.assertEqual(cfg.workload_overrides.get(k), v,
                                     f"{arm}: workload.{k} 被改了")

    def test_controller_differs_only_in_enabled_and_kind(self):
        """★ 核心不变式：controller 段除 enabled/kind 外逐字段相同。"""
        ref = _load("static")
        switch_fields = {"enabled", "kind"}
        for arm, _, _, _ in G.ARMS:
            cfg = _load(arm)
            with self.subTest(arm=arm):
                all_fields = {f.name for f in dataclasses.fields(cfg.controller)}
                for name in all_fields - switch_fields:
                    self.assertEqual(
                        getattr(cfg.controller, name),
                        getattr(ref.controller, name),
                        f"{arm}: controller.{name} 被改了（只允许 enabled/kind 不同）")

    def test_switch_values_match_declaration(self):
        for arm, enabled, kind, _ in G.ARMS:
            cfg = _load(arm)
            with self.subTest(arm=arm):
                self.assertEqual(bool(cfg.controller.enabled), enabled,
                                 f"{arm}: controller.enabled 应为 {enabled}")
                self.assertEqual(cfg.controller.kind, kind,
                                 f"{arm}: controller.kind 应为 {kind}")


class TestRealLoaderAcceptsAllArms(unittest.TestCase):
    """★ 真加载器必须接受三臂 —— 上一版文本校验放过非法组合，这条守住。"""

    def test_load_config_does_not_raise(self):
        for arm, *_ in G.ARMS:
            with self.subTest(arm=arm):
                try:
                    _load(arm)
                except Exception as e:  # noqa: BLE001
                    self.fail(f"{arm}: 真加载器拒绝 —— {type(e).__name__}: {e}")

    def test_static_arm_uses_legal_kind(self):
        """static 臂必须是 `kind: threshold`（不是留底稿的 cost）。

        反向对照：若改成 cost，`ControllerConfig` 会因
        `kind != threshold 且 enabled == False` 抛错。
        """
        cfg = _load("static")
        self.assertFalse(cfg.controller.enabled)
        self.assertEqual(cfg.controller.kind, "threshold")

        # 反向对照：手工构造那个非法组合必须抛错
        from ascend_opt.controller import ControllerConfig  # noqa: E402
        with self.assertRaises(ValueError):
            ControllerConfig(kind="cost", enabled=False)


class TestGeneratorStaysInSync(unittest.TestCase):
    def test_verify_passes(self):
        """文件必须与生成器输出一致 —— 手改一份就会在这里变红。"""
        bad = G.verify()
        self.assertEqual(bad, [], "配置与生成器不一致（有人手改过？）")

    def test_verify_catches_injected_defects(self):
        """★ 反向对照：注入 3 类缺陷，`verify` 必须**全部**变红。

        为什么要有这条：守卫本身也会失效，而失效的守卫**看起来是绿的**。
        """
        # ① 两臂同 kind
        f = G.build()
        f[f"{G.cell_name('threshold')}.yaml"] = f[f"{G.cell_name('threshold')}.yaml"].replace(
            "  kind: threshold\n", "  kind: cost\n")
        self.assertNotEqual(G.verify(f), [], "① 两臂同 kind 未被抓出")

        # ② cost 臂 controller.enabled 关掉（非法组合）
        f = G.build()
        fn = f"{G.cell_name('cost')}.yaml"
        f[fn] = G._replace_in_ctrl(f[fn], "  enabled: true\n", "  enabled: false\n")
        self.assertNotEqual(G.verify(f), [], "② cost 臂被关（非法组合）未被抓出")

        # ③ static 臂 kind 写成 cost（真加载器才抓到的那类）
        f = G.build()
        fn = f"{G.cell_name('static')}.yaml"
        f[fn] = G._replace_in_ctrl(f[fn], "  kind: threshold\n", "  kind: cost\n")
        self.assertNotEqual(G.verify(f), [], "③ static 臂 kind=cost 未被抓出")

        # ④ 引入第二个变量
        f = G.build()
        fn = f"{G.cell_name('threshold')}.yaml"
        f[fn] = f[fn].replace("  max_num_seqs: 4\n", "  max_num_seqs: 8\n")
        self.assertNotEqual(G.verify(f), [], "④ 第二个变量未被抓出")


if __name__ == "__main__":
    unittest.main()
