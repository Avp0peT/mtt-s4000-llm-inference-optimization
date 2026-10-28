"""`costcalib_fit.py` 的两件新东西的用例（第 14 步归因审计的产物）。

1. **秩保证划分**：每段先凑满秩、其余按哈希留出 —— 它把 95.26% 的低估降到 11.77%；
2. **覆盖诊断**：逐段报秩并点名不足的段 —— 没有它，误差本身不告诉你为什么。

⚠️ 这两件都必须**能失败**：故意破坏输入时用例要红。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import costcalib_fit as F  # noqa: E402


def rec(batch: int, pf: int, kv: int, key: str | None = None) -> dict:
    return {"key": key or f"b{batch}_pf{pf}_kv{kv}_nf1", "batch": batch,
            "prefill_tok": pf, "kv_len": kv, "n_forwards": 1, "actual_ms": 100.0}


def full_grid(batch: int) -> list[dict]:
    """某段的一个**满秩**最小集：三种 prefill × 两种 kv。"""
    return [rec(batch, 0, 512), rec(batch, 512, 512), rec(batch, 1024, 1024)]


class TestRankGuaranteedSplit(unittest.TestCase):
    def test_every_segment_of_the_training_set_is_full_rank(self):
        """★ 核心保证：**每一段**的训练集都满秩（3）。"""
        recs = []
        for b in (1, 2, 8):
            recs += full_grid(b)
            recs += [rec(b, 0, 1024), rec(b, 512, 2048), rec(b, 0, 2048)]
        tr, ho = F.split_rank_guaranteed(recs, holdout_modulus=2)
        self.assertTrue(tr, "训练集不应为空")
        for b in sorted({int(r["batch"]) for r in tr}):
            seg = [r for r in tr if int(r["batch"]) == b]
            self.assertGreaterEqual(F._rank(seg), 3,
                                    f"batch={b} 训练集秩 {F._rank(seg)} < 3（该段不可辨识）")
        self.assertEqual(len(tr) + len(ho), len(recs), "划分不能丢点")

    def test_hash_split_can_leave_a_segment_rank_deficient(self):
        """对照组：**纯哈希**划分确实会留下秩不足的训练段（这就是要修的东西）。

        ⚠️ 必须用**确定性**构造（key 的 `%4` 已核对过），否则这条用例会因为
        "哈希恰好不是那样"而失败 —— 那是**因错误的原因红**，比不写还糟。
        构造：把该段 `prefill>0` 的点放在 `%4==0`（留出）的 key 上，
        `prefill=0` 的点放在 `%4!=0`（训练）的 key 上
        ⇒ 哈希划分下该段的训练集**全是纯 decode** ⇒ 秩 2 < 3（参数需 3）。
        """
        recs = [rec(8, 0, 512, "k0"),      # %4=2 → train
                rec(8, 0, 1024, "k1"),     # %4=3 → train
                rec(8, 0, 2048, "k2"),     # %4=3 → train
                rec(8, 512, 1024, "k10"),  # %4=0 → holdout
                rec(8, 1024, 1024, "k18")] # %4=0 → holdout
        tr = [r for r in recs if F.split_hash(r, 4) == "train"]
        self.assertEqual(len(tr), 3, "构造失效：训练侧应有 3 个点")
        self.assertTrue(all(int(r["prefill_tok"]) == 0 for r in tr),
                        "构造失效：训练侧应全是纯 decode")
        self.assertEqual(F._rank(tr), 2, "纯 decode + 多个 kv ⇒ 秩 2（常数项 + kv，缺 prefill 方向）")
        self.assertLess(F._rank(tr), 3, "★★ 秩 2 < 参数个数 3 ⇒ 该段**不可辨识**")

    def test_rank_helper_counts_directions_not_points(self):
        """秩是**方向数**，不是点数。

        ⚠️ 注意 `prefill=0` + 多个 `kv` 的秩是 **2**（常数项 + `kv` 两个方向），
        而模型有 **3 个参数** ⇒ 仍然**不可辨识**（`prefill` 系数由 lstsq 的最小范数解给出，
        于是"纯 decode 的 88 ms"会被用来预测一个 prefill 主导的 1858 ms 形状）。
        ⇒ 判据必须是 `秩 < 参数个数`，**不能**用"有没有点"或"秩 > 0"。
        """
        self.assertEqual(F._rank([rec(1, 0, 512), rec(1, 0, 1024), rec(1, 0, 2048)]), 2)
        self.assertEqual(F._rank([rec(1, 512, 512)]), 1)     # 单点（一行）⇒ 秩 1
        self.assertEqual(F._rank([]), 0)
        self.assertEqual(F._rank(full_grid(1)), 3)
        self.assertEqual(F._rank([rec(1, 0, 512), rec(1, 0, 1024)]), 2)


class TestCoverageDiagnosis(unittest.TestCase):
    def test_flags_rank_deficient_training_segment(self):
        """★ 秩不足的**训练**段必须被点名（它是 95% 低估的成因）。"""
        rows = [rec(2, 0, 512), rec(2, 0, 2048)]            # 纯 decode ⇒ 秩 1
        problems = F.diagnose_coverage(rows, "train")
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("训练集不可辨识", problems[0])

    def test_holdout_rank_deficiency_is_reported_differently(self):
        """留出集秩不足 ⇒ 是**评估覆盖不足**，不是模型的问题 —— 两者不能混报。"""
        rows = [rec(3, 0, 1024)]
        problems = F.diagnose_coverage(rows, "holdout")
        self.assertEqual(len(problems), 1)
        self.assertIn("留出评估覆盖不足", problems[0])

    def test_full_rank_segment_is_not_flagged(self):
        self.assertEqual(F.diagnose_coverage(full_grid(4), "train"), [])


if __name__ == "__main__":
    unittest.main()
