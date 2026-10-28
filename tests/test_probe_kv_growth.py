"""`grown_kv_lens`：修复 `real_mixed` **从未跑通**的那个 bug（真实事故）。

事故（2026-09-26 实测）
--------------------
`fwd_costcalib_probe.measure_point` 的「服务协议」块里原本写的是
`L = min(kv_lens[0] + i * n_new, KV_CAP)` 再 `[L] * b`。
`uniform` 口径下 `n_new` 是 `int` ⇒ 正确；但 **`real_mixed` 把它变成 `list`**
（一条做 `prefill_tok`、其余各 1 个）⇒ `i * n_new` 成了 list、再 `int + <list>`
⇒ **`TypeError: unsupported operand type(s) for +: 'int' and 'list'`**。

**后果**：`real_mixed` 路径**从未产生过一个有效测点** ——
v6 标定 96 个 `real_mixed` 点**失败 84 个**；D1 pilot 16 个小 chunk 点**失败 12 个**。
而脚本**照常写产物**、闸门只报"覆盖率不足" ⇒ 表现成"模型没建好"，不是"代码有 bug"。

本文件钉三件事：
① **list 不再抛错**（直接回归）；
② `uniform` 下与旧实现的**逐值相同**（否则会顺手改掉 v1..v5 的可比性）；
③ `real_mixed` 下**按序列各自增长**（各序列 KV 增速本来就不同）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fwd_costcalib_probe import grown_kv_lens  # noqa: E402

_CAP = 4096


def _old_impl(kv: int, step: int, n: int, b: int, cap: int = _CAP) -> list[int]:
    """**旧实现**（只在 `n` 为 int 时可用）—— 用作反面参照。"""
    return [max(int(min(kv + step * n, cap)), 1)] * b


class TestTheBugItself(unittest.TestCase):
    def test_old_expression_really_raised(self):
        """证明本用例测的是**真缺陷**：把旧的 `int + i*list` 原样写出来必须抛错。"""
        n_list = [512, 1, 1]
        with self.assertRaises(TypeError):
            _ = 1024 + 1 * n_list          # noqa: B018  —— 旧写法

    def test_new_helper_accepts_a_list(self):
        """★ 回归：list 输入不再抛错。"""
        got = grown_kv_lens(1024, 1, [512, 1, 1], _CAP)
        self.assertEqual(len(got), 3)
        self.assertTrue(all(isinstance(x, int) for x in got))


class TestUniformIsByteCompatible(unittest.TestCase):
    """★ 修 `real_mixed` **不得**顺手改掉 `uniform` 的口径（那会让 v1..v5 不可比）。"""

    def test_matches_old_impl_for_several_points(self):
        for kv, n, b in ((1024, 512, 2), (1024, 0, 4), (2048, 128, 8), (1, 1, 3)):
            n_eff = n if n > 0 else 1         # 与 measure_point 一致：pure decode 取 1
            with self.subTest(kv=kv, n=n, b=b):
                self.assertEqual(grown_kv_lens(kv, 3, [n_eff] * b, _CAP),
                                 _old_impl(kv, 3, n_eff, b))

    def test_zero_step_is_identity(self):
        self.assertEqual(grown_kv_lens(1024, 0, [1] * 4, _CAP), [1024] * 4)


class TestRealMixedGrowsPerSequence(unittest.TestCase):
    def test_first_sequence_grows_fast_others_slow(self):
        got = grown_kv_lens(1024, 2, [512, 1, 1, 1], _CAP)
        self.assertEqual(got, [1024 + 1024, 1026, 1026, 1026])

    def test_length_follows_batch(self):
        self.assertEqual(len(grown_kv_lens(0, 1, [512] + [1] * 7, _CAP)), 8)

    def test_sequences_are_not_uniform(self):
        got = grown_kv_lens(1024, 3, [512, 1, 1], _CAP)
        self.assertNotEqual(len(set(got)), 1, "real_mixed 下各序列 KV 增速不同，不该是同一个值")


class TestCapAndFloor(unittest.TestCase):
    def test_cap_clamps_each_sequence(self):
        got = grown_kv_lens(4090, 5, [512, 1], 4096)
        self.assertEqual(got, [4096, 4095], "各自夹到 cap，不是一起夹")

    def test_floor_is_one(self):
        self.assertEqual(grown_kv_lens(0, 0, [1, 1], _CAP), [1, 1])


if __name__ == "__main__":
    unittest.main()
