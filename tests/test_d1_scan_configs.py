"""§7d 低并发扫描配置的验收：**单变量**、**两处声明一致**、且"扫描批不改别的"。

为什么必须有这一组用例
----------------------
扫描的全部价值在于「只改 `concurrency` 与 `max_num_seqs`，其余逐字段不动」。
8 份配置是生成的，但生成器只在**生成那一刻**对；之后任何人手改一份，
"只改两个变量"就悄悄破坏了 —— 而产出的表看起来完全正常，结论却归因到了错的维度。

两条硬约束
----------
1. **两个变量都进文件名**：`fixed_cost_scan_c<c>_m<m>`.yaml。
   若只写 `c<c>`，4 个并发度 × 2 个批预算会撞名 ⇒ 后写的覆盖先写的，
   **静默少掉 4 格**（这类"内容对、条目数错"的失败不报错）。
2. **`max_num_seqs` 与控制器规划上限不得错配**：扫描批要回答的是
   "改批预算会怎样"，而 `controller.max_batched_tokens_*` 是同一维的另一处声明。
   只在 `scheduler` 改、不在 `controller` 同步 ⇒ 产物里两个数互相矛盾。

与 `test_step15_configs.py` 的关系：那组用例钉的是**消融**（phase1_step15/），
本组钉的是**扫描**（phase1_d1_scan/）。两者的 reference 不同，
**故意分成两个文件**——合并成一个会让"哪份配置属于哪条结论"变模糊。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_benchmark import load_config  # noqa: E402

import gen_phase1_d1_scan_configs as G  # noqa: E402

SCAN = G.OUT_DIR
BASE = G.REFERENCE


def _load(p: Path):
    return load_config(p)


class TestScanGridShape(unittest.TestCase):
    def test_grid_is_cartesian(self):
        """8 格 = 4 并发度 × 2 批预算，一格不少。"""
        self.assertEqual(len(G.cells()), 8)
        self.assertEqual(set(G.CONCURRENCIES), {2, 4, 6, 8})
        self.assertEqual(set(G.MAX_NUM_SEQS), {4, 8})

    def test_names_encode_both_variables(self):
        """★ 名字必须同时含 c 与 m，否则会撞名并静默丢掉半张表。"""
        names = [f"{G.cell_name(c, m)}.yaml" for c, m in G.cells()]
        self.assertEqual(len(names), len(set(names)), f"格名有重复：{names}")
        for (c, m), name in zip(G.cells(), names):
            self.assertIn(f"c{c}", name.replace("_cost_", "_"))
            self.assertIn(f"m{m}", name)

    def test_all_files_exist(self):
        for c, m in G.cells():
            f = SCAN / f"{G.cell_name(c, m)}.yaml"
            self.assertTrue(f.is_file(), f"缺少 {f.name}")


class TestOnlyTwoVariables(unittest.TestCase):
    """每份与底稿的差异**恰好**是声明的两处。"""

    def test_concurrency_and_max_seqs_are_the_only_deltas(self):
        ref = _load(BASE)
        for c, m in G.cells():
            name = f"{G.cell_name(c, m)}.yaml"
            cfg = _load(SCAN / name)

            with self.subTest(name=name):
                self.assertEqual(int(cfg.workload_overrides.get("concurrency", -1)), c,
                                 f"{name}: concurrency 应为 {c}")
                self.assertEqual(int(cfg.max_num_seqs), m,
                                 f"{name}: max_num_seqs 应为 {m}")

                # 负载的**其余**字段一个都不许动
                for k, v in ref.workload_overrides.items():
                    if k == "concurrency":
                        continue
                    self.assertEqual(cfg.workload_overrides.get(k), v,
                                     f"{name}: workload.{k} 被改了（实际 "
                                     f"{cfg.workload_overrides.get(k)}）")
                self.assertEqual(cfg.workload_name, ref.workload_name)

                # 控制器逐字段相同（扫描批不动策略）
                import dataclasses
                for f_ in dataclasses.fields(ref.controller):
                    self.assertEqual(getattr(cfg.controller, f_.name),
                                     getattr(ref.controller, f_.name),
                                     f"{name}: controller.{f_.name} 被改了")

                # 分块 / SLO / runtime 一字不动
                self.assertEqual(cfg.chunk, ref.chunk, f"{name}: chunk 段被改了")
                self.assertEqual(cfg.chunk_policy_name, ref.chunk_policy_name)
                self.assertEqual(cfg.run.slo.ttft_ms, ref.run.slo.ttft_ms)
                self.assertEqual(cfg.run.slo.tpot_ms, ref.run.slo.tpot_ms)
                self.assertEqual(cfg.runtime, ref.runtime, f"{name}: runtime 段被改了")
                for k in ("repeats", "warmup_s", "control_interval_steps"):
                    self.assertEqual(getattr(cfg.run, k), getattr(ref.run, k),
                                     f"{name}: run.{k} 被改了")

    def test_scheduler_section_only_differs_in_max_num_seqs(self):
        ref = _load(BASE)
        for c, m in G.cells():
            cfg = _load(SCAN / f"{G.cell_name(c, m)}.yaml")
            with self.subTest(name=cfg.stem):
                self.assertEqual(cfg.scheduler_name, ref.scheduler_name)
                self.assertEqual(cfg.scheduler_kwargs, ref.scheduler_kwargs,
                                 "scheduler 除 max_num_seqs 外的参数不应变")


class TestGeneratorStaysInSync(unittest.TestCase):
    def test_verify_passes(self):
        """文件必须与生成器输出一致 —— 手改一份就会在这里变红。"""
        bad = G.verify()
        self.assertEqual(bad, [], "配置与生成器不一致（有人手改过？）")


class TestCoverageGapIsDocumented(unittest.TestCase):
    """§7d 声明的"查表覆盖缺口"必须与模型文件的实际形状一致。

    为什么测这个：结论表要**并排**报命中率缺口；若模型换了却没更新文档，
    缺口说明就成了假话，而数字看起来照常。
    """

    def test_b3_5_6_7_lack_small_prefill_keys(self):
        import json
        model = json.loads(
            (_ROOT / "configs" / "cost_calib" / "model_v7.json").read_text(encoding="utf-8")
        )
        table = model["lookup_table"]
        have: dict[int, set[int]] = {}
        for k in table:
            b, pf, _kv = k.split("_")
            have.setdefault(int(b[1:]), set()).add(int(pf[2:]))
        for b in (3, 5, 6, 7):
            self.assertNotIn(128, have.get(b, set()),
                             f"b={b} 现在有 pf=128 键了 ⇒ §7d 的覆盖缺口说明已过期，须更新")
            self.assertNotIn(256, have.get(b, set()),
                             f"b={b} 现在有 pf=256 键了 ⇒ §7d 的覆盖缺口说明已过期")
        for b in (1, 2, 4, 8):
            self.assertIn(128, have.get(b, set()), f"b={b} 应有 pf=128 键")


if __name__ == "__main__":
    unittest.main()
