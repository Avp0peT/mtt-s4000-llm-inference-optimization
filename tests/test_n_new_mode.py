"""`n_new_mode`：**新 token 的分配方式**（第 14 步的核心口径修复）。

背景（实测）：同一列 `prefill_tok` 两侧含义不同 ——
* 标定探针（v1..v5 的 `mixed_serving`）：`ids=(batch, n_new)` ⇒ **每条序列各做 `prefill_tok`**
  ⇒ 全步 `batch × prefill_tok`；
* harness：`tokens_prefilled` 是**全步总量**（遥测 `note` 逐字如 `prefill_tok=512 decode=7`）。
⇒ 差约 `batch` 倍，实证：模型算出 `b2/pf512 = 286.8 ms`、`b8/pf512 = 1004.8 ms` 明显偏高。
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from costcalib_grid import Shape, expand, load_grid  # noqa: E402


def _batch_layout():
    """从源码里抠出纯函数（`import` 探针会触发 torch，本机管理解释器没有）。"""
    with io.open(_ROOT / "scripts" / "fwd_costcalib_probe.py", encoding="utf-8") as fh:
        src = fh.read()
    i, j = src.index("def batch_layout"), src.index("def make_batch")
    ns: dict = {}
    exec(src[i:j].replace("-> tuple[int, int, list[tuple[int, int]]]", ""), ns)
    return ns["batch_layout"]


class TestBatchLayout(unittest.TestCase):
    def test_uniform_mode_is_every_sequence_the_same(self):
        f = _batch_layout()
        max_n, total, rows = f([1024] * 4, 512)
        self.assertEqual(max_n, 512)
        self.assertEqual(total, 1536)
        self.assertEqual(rows, [(0, 1536)] * 4, "uniform：每条序列都做 512 个新 token")

    def test_real_mixed_mode_is_one_sequence_plus_ones(self):
        f = _batch_layout()
        max_n, total, rows = f([1024] * 4, [512, 1, 1, 1])
        self.assertEqual(max_n, 512)
        self.assertEqual(total, 1536)
        self.assertEqual(rows[0], (0, 1536), "第一条：KV 1024 + 512 个新 token")
        self.assertEqual(rows[1:], [(0, 1025)] * 3, "其余各：KV 1024 + **1** 个新 token")

    def test_pure_decode_is_one_token_each(self):
        f = _batch_layout()
        max_n, total, rows = f([1024] * 4, 1)
        self.assertEqual((max_n, total), (1, 1025))
        self.assertEqual(rows, [(0, 1025)] * 4)

    def test_rejects_bad_inputs(self):
        f = _batch_layout()
        with self.assertRaises(ValueError):
            f([1024, 1024], [512, 1, 1])          # 长度不匹配
        with self.assertRaises(ValueError):
            f([1024], [0])                        # 每条至少 1 个新 token


class TestShapeKeyCarriesMode(unittest.TestCase):
    def test_two_modes_do_not_collide(self):
        """★ 两种口径**必须**在 key 上分得开，否则产物里分不出这一格是什么形状。"""
        a = Shape(group="g", batch=4, prefill_tok=512, kv_len=1024, n_forwards=1)
        b = Shape(group="g", batch=4, prefill_tok=512, kv_len=1024, n_forwards=1,
                  n_new_mode="real_mixed")
        self.assertNotEqual(a.as_key(), b.as_key())
        self.assertIn("uniform", a.as_key()[-1])

    def test_v6_mixed_serving_uses_real_mixed(self):
        pts = expand(load_grid(_ROOT / "configs" / "cost_calib" / "shape_grid_v6.yaml"))
        mixed = [s for s in pts if s.group == "mixed_serving"]
        self.assertTrue(mixed)
        self.assertTrue(all(s.n_new_mode == "real_mixed" for s in mixed),
                        "v6 的 mixed_serving 必须是真实结构")
        self.assertTrue(all(s.n_new_mode == "uniform"
                            for s in pts if s.group == "pure_decode"),
                        "pure_decode 不受影响（两边都是 1 个新 token）")

    def test_older_grids_default_to_uniform(self):
        pts = expand(load_grid(_ROOT / "configs" / "cost_calib" / "shape_grid_v4.yaml"))
        self.assertTrue(all(s.n_new_mode == "uniform" for s in pts),
                        "旧网格没有该字段 ⇒ 必须默认 uniform（行为逐字节不变）")

    def test_v6_recovers_the_point_that_v5_silently_dropped(self):
        """v5 里 serial 组的 `nf=1` 点与 mixed 的同形状点**撞键被吞掉**；v6 恢复它。"""
        v5 = expand(load_grid(_ROOT / "configs" / "cost_calib" / "shape_grid_v5.yaml"))
        v6 = expand(load_grid(_ROOT / "configs" / "cost_calib" / "shape_grid_v6.yaml"))
        s5 = [s for s in v5 if s.group == "serial_forward_group"]
        s6 = [s for s in v6 if s.group == "serial_forward_group"]
        self.assertEqual(len(s5), 3, "v5 里少一个（被撞键吞掉）")
        self.assertEqual(len(s6), 4, "v6 恢复为声明的 4 个")


if __name__ == "__main__":
    unittest.main()
