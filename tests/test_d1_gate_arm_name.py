"""臂名的**唯一权威**是产物的 `config.config_stem`，不是 run_id 的命名解析。

为什么单开一个文件
------------------
2026-09-28 实测：`d1_gate.py` 用 `parse_run_id` 定臂名，而 `RUN_ID_RE` 把 run_id 尾部
按 **`<cell>_<arm>`** 拆（`cell` 段还是 `[a-z]+`）。但真实的写入契约是
**尾部 = `<config_stem>` 整体**：

* `scripts/run_step3_matrix.py:300` → `f"{stamp}-{cfg.workload_name}-{name}"`（`name` = cell 名）
* `benchmark/run_benchmark.py:1282` → `build_run_id(cfg.workload_name, cfg.stem)`

⇒ `parse_run_id` 用了**一个不存在的分隔契约**去拆一个整体名。后果（实测，非推断）：

| run_id | `parse_run_id` 给 | 产物 `config_stem` | 谁对 |
| --- | --- | --- | --- |
| `S6-…-service_benefit_c2_m4_cost` | `benefit_c2_m4_cost` | `service_benefit_c2_m4_cost` | 都不对（真臂名 `cost`）|
| `S14G-…-fixed_cost_min512` | `cost_min512` | `fixed_cost_min512` | **stem** |
| `S14S-…-fixed_cost_scan_c2_m4` | `cost_scan_c2_m4` | `fixed_cost_scan_c2_m4` | **stem** |

实测范围：6 个批次 / 42 个目录。**这条缺陷最危险之处是它不报错** ——
`--arm cost_min512` 在 S14G 上之所以能取到数据，纯属"旧解析器错得恰好一致"。

⚠️ 本文件**只钉住事实与派生规则**，不改 `run_ids.py`（它的改动属第 14 步 P0
"臂名归一"的范畴，须先预注册）。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import d1_gate as G  # noqa: E402
from ascend_opt.run_ids import parse_run_id  # noqa: E402


class TestDeriveArmName(unittest.TestCase):
    """`derive_arm_name` 的规则：认识名字就归名，不认识就**原样返回**（不截断）。"""

    def test_known_arm_suffixes_are_recognized(self):
        cases = {
            "service_benefit_c2_m4_cost": "cost",
            "service_benefit_c2_m4_static": "static",
            "service_benefit_c2_m4_threshold": "threshold",
            "fixed_cost_pilot": "cost_pilot",
            "fixed_cost_min512": "cost_min512",
            "cost_pilot": "cost_pilot",
        }
        for stem, want in cases.items():
            self.assertEqual(G.derive_arm_name(stem), want, f"stem={stem}")

    def test_unknown_stem_is_returned_whole_not_truncated(self):
        """★ 扫描批的"臂"其实是**负载格** —— 不得被拆成 `cost_scan_c2_m4`。"""
        for stem in ("fixed_cost_scan_c2_m4", "fixed_cost_scan_c8_m8"):
            self.assertEqual(G.derive_arm_name(stem), stem,
                             "认不出臂后缀时**原样返回**，绝不按分隔符拆")

    def test_none_stays_none(self):
        self.assertIsNone(G.derive_arm_name(None), "取不到就留 None，禁止填猜测值")


class TestParseRunIdVsConfigStem(unittest.TestCase):
    """`parse_run_id` 与**产物真实 `config_stem`** 的对照（三个真实形状）。"""

    def test_s6_shape_shows_the_fork(self):
        rid = "S6-short_chat-service_benefit_c2_m4_cost"
        parsed = parse_run_id(rid)
        self.assertIsNotNone(parsed)
        # 名字解析给的 arm 是 cell 的尾巴
        self.assertEqual(parsed.arm, "benefit_c2_m4_cost")
        # 而产物记录的 stem 是整体 —— 两者本就不同名
        stem = "service_benefit_c2_m4_cost"
        self.assertNotEqual(parsed.arm, stem)
        # 只有 config_stem 能定出真臂名
        self.assertEqual(G.derive_arm_name(stem), "cost")
        self.assertNotEqual(parsed.arm, G.derive_arm_name(stem),
                            "★ 名字解析与 config_stem 在 S6 形状上必然分叉")

    def test_s14g_shape_parse_gives_a_name_that_is_not_in_any_artifact(self):
        """★ 决定性证据：名字解析给出 **`cost_min512`**，而产物里只有 `fixed_cost_min512`。"""
        rid = "S14G-short_chat-fixed_cost_min512"
        parsed = parse_run_id(rid)
        self.assertEqual(parsed.arm, "cost_min512")
        # 真产物（results/raw/S14G-short_chat-fixed_cost_min512/meta.json）的 config_stem
        self.assertEqual(G.derive_arm_name("fixed_cost_min512"), "cost_min512",
                         "本批的已知臂后缀里含 cost_min512 ⇒ 归名后与名字解析**恰好一致**")
        # ⇒ 这个"恰好一致"正是旧行为能工作的原因，也就是它脆弱的原因

    def test_s14s_scan_shape_parse_truncates_a_load_cell(self):
        rid = "S14S-short_chat-fixed_cost_scan_c2_m4"
        parsed = parse_run_id(rid)
        self.assertEqual(parsed.arm, "cost_scan_c2_m4",
                         "旧解析把负载格截断（去掉 fixed_）")
        self.assertEqual(G.derive_arm_name("fixed_cost_scan_c2_m4"),
                         "fixed_cost_scan_c2_m4",
                         "★ 新规则不截断 ⇒ 扫描批的负载格名保持完整")


class TestLoadRunsUsesConfigStem(unittest.TestCase):
    """端到端：`load_runs` 的 `arm` 来自 `config_stem`，且与名字分叉时**有话说**。"""

    def _mk(self, root: Path, run_id: str, stem: str | None) -> None:
        d = root / run_id
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "summary": {"run_id": run_id},
            "config": ({"config_stem": stem} if stem else {}),
            "reproducibility": {"git_commit": "a" * 40, "git_dirty": False},
            "controller_effect": {"arm_state": {"n_degraded": 1, "n_decisions": 2,
                                                "n_fallback": 0}},
            "data_label": "MEASURED",
        }
        with open(d / "meta.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, ensure_ascii=False)
        with open(d / "telemetry.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"step": 0, "note": "prefill_tok=128"}) + "\n")

    def test_arm_comes_from_config_stem(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._mk(root, "S6-short_chat-service_benefit_c2_m4_cost",
                     "service_benefit_c2_m4_cost")
            runs = G.load_runs(root, "S6")
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].arm, "cost", "臂名取自 config_stem")
            self.assertEqual(runs[0].config_stem, "service_benefit_c2_m4_cost")

    def test_missing_stem_leaves_arm_none_not_guessed(self):
        """★ 缺 `config_stem` ⇒ `arm` 留 `None`（归到 `<未解析>`），**不猜**。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._mk(root, "S6-short_chat-service_benefit_c2_m4_cost", None)
            runs = G.load_runs(root, "S6")
            self.assertIsNone(runs[0].arm, "取不到就 None —— 不填猜测值")
            self.assertEqual(sorted(G._per_arm(runs)), ["<未解析>"])

    def test_name_stem_mismatch_is_reported(self):
        """名字尾部与 `config_stem` 不一致时**必须留痕**（不静默选一个）。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # 目录/名字说是 A，产物说是 B ⇒ 以 B 为准，但报告分叉
            self._mk(root, "S6-short_chat-something_else", "service_benefit_c2_m4_cost")
            runs = G.load_runs(root, "S6")
            self.assertEqual(runs[0].arm, "cost", "以产物为准")
            self.assertTrue(any("命名契约分叉" in p for p in runs[0].problems),
                            f"应报告分叉，实得 {runs[0].problems}")

    def test_repeat_suffix_is_not_a_mismatch(self):
        """★ `-rN` 重复后缀**不得**被误报成分叉（我第一版踩过：9 目录 6 个假阳性）。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._mk(root, "S6-short_chat-service_benefit_c2_m4_cost-r2",
                     "service_benefit_c2_m4_cost")
            runs = G.load_runs(root, "S6")
            self.assertEqual(runs[0].arm, "cost")
            self.assertEqual([p for p in runs[0].problems if "分叉" in p], [],
                             "带 -rN 的同名目录不该报分叉")


if __name__ == "__main__":
    unittest.main()
