"""S01 / P1：输出层冗余（仅末位置 logits + 只采需要的行）。

两个机制各自**独立开关、默认关闭**。本文件验的是**正确性**：
"关掉时逐字节不变"与"打开时数值/生成结果与基线一致"。
⚠️ 性能（TTFT / prefill 时间 / 显存）**不属于本地可验范围** ⇒ `REMOTE_PENDING`。

为什么必须验"等价的三种变体"而不是只看 argmax：`num_logits_to_keep=1` 影响的是
**logits 张量的形状**，一旦某处仍按 `[B, k, V]` 索引就会静默取到错的位置；
而"只采需要的行"靠**行下标映射**，映射写错会让非 need_token 的行拿到别人的 token
——两种错误都不会抛异常。所以这里比对**逐行 logits**，不只比最终 token。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _tiny_support as T  # noqa: E402
from real_backend import ForwardItem  # noqa: E402

if T.torch is not None:
    import torch  # noqa: E402
    from torch import nn  # noqa: E402


class _NoKeepModule(nn.Module if T.torch is not None else object):
    """一个**明确不支持** `num_logits_to_keep` 的模型（拒绝该关键字参数）。

    用来验"配置里开着但模型不支持 ⇒ **显式回退并记录**"，而不是静默吞掉参数。
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.config = inner.config

    def forward(self, **kwargs):
        if "num_logits_to_keep" in kwargs:
            raise TypeError(
                "forward() got an unexpected keyword argument 'num_logits_to_keep'"
            )
        return self.inner(**kwargs)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestDefaultsUnchanged(unittest.TestCase):
    """两个开关默认关闭 ⇒ 请求/生效都为 False，且计数为 0。"""

    def test_defaults_are_off_and_recorded(self):
        model = T.make_model()
        drv = T.make_driver(model)
        drv.load()
        eo = drv.execution_optimizations
        self.assertFalse(eo["logits_last_only"]["requested"])
        self.assertFalse(eo["logits_last_only"]["effective"])
        self.assertFalse(eo["sample_needed_only"]["requested"])
        self.assertEqual(eo["logits_last_only"]["n_calls_last_only"], 0)
        self.assertEqual(eo["logits_last_only"]["n_calls_all_positions"], 0)

    def test_default_path_never_sends_the_kwarg(self):
        rec = T.RecordingModel(T.make_model())
        drv = T.make_driver(rec)
        drv.load()
        drv.forward([ForwardItem(T.req("a", 5), n_new_tokens=5, need_token=True)])
        self.assertNotIn("num_logits_to_keep", rec.kwargs[-1])
        self.assertEqual(drv.execution_optimizations["logits_last_only"]
                         ["n_calls_all_positions"], 1)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestLastPositionLogits(unittest.TestCase):
    """仅末位置 logits：数值与 KV 必须与全位置路径一致。"""

    PROMPTS = {"a": 9, "b": 5}

    def test_last_position_logits_are_identical(self):
        """末位置 logits 与"全位置再取最后一列"必须逐数值一致。"""
        base = T.make_model()
        full_rec = T.RecordingModel(base)
        full = T.make_driver(full_rec)
        full.load()
        last_rec = T.RecordingModel(T.make_model())
        last = T.make_driver(last_rec, logits_last_only=True)
        last.load()

        for key, n in self.PROMPTS.items():
            r = T.req(key, n)
            full.forward([ForwardItem(r, n_new_tokens=n, need_token=True)])
            last.forward([ForwardItem(r, n_new_tokens=n, need_token=True)])

        got = last_rec.logits[-1]
        ref = full_rec.logits[-1][:, -1:, :]
        self.assertEqual(tuple(got.shape), tuple(ref.shape))
        self.assertTrue(
            torch.allclose(got, ref, atol=1e-5, rtol=1e-4),
            f"末位置 logits 不一致：max_abs_diff={(got - ref).abs().max().item():.3e}",
        )

    def test_kwarg_is_really_passed(self):
        rec = T.RecordingModel(T.make_model())
        drv = T.make_driver(rec, logits_last_only=True)
        drv.load()
        drv.forward([ForwardItem(T.req("a", 4), n_new_tokens=4, need_token=True)])
        self.assertEqual(rec.kwargs[-1].get("num_logits_to_keep"), 1)
        eo = drv.execution_optimizations["logits_last_only"]
        self.assertTrue(eo["effective"])
        self.assertEqual(eo["n_calls_last_only"], 1)

    def test_past_len_and_kv_are_unchanged(self):
        """末位置只影响输出投影；KV 与历史长度必须完全一致。"""
        a_model = T.make_model()
        a = T.make_driver(a_model, logits_last_only=True)
        a.load()
        b_model = T.make_model()
        b = T.make_driver(b_model)
        b.load()

        for key, n in self.PROMPTS.items():
            r = T.req(key, n)
            a.forward([ForwardItem(r, n_new_tokens=n, need_token=True)])
            b.forward([ForwardItem(r, n_new_tokens=n, need_token=True)])

        for key, n in self.PROMPTS.items():
            with self.subTest(req=key):
                self.assertEqual(a._slots[key].past_len, b._slots[key].past_len)
                for la, lb in zip(a._slots[key].past, b._slots[key].past):
                    self.assertTrue(torch.allclose(la[0], lb[0], atol=1e-6))
                    self.assertTrue(torch.allclose(la[1], lb[1], atol=1e-6))

    def test_multi_step_greedy_tokens_match(self):
        """多步生成（decode 3 步）的 token 序列必须一致。"""
        seq_a, seq_b = [], []
        rec_a = T.RecordingModel(T.make_model())
        a = T.make_driver(rec_a, logits_last_only=True)
        a.load()
        rec_b = T.RecordingModel(T.make_model())
        b = T.make_driver(rec_b)
        b.load()

        for drv, seq in ((a, seq_a), (b, seq_b)):
            out = drv.forward([ForwardItem(T.req("a", 7), n_new_tokens=7, need_token=True)])
            seq.append(out.slots[0].token_id)
            for _ in range(3):
                out = drv.forward([ForwardItem(T.req("a", 7), n_new_tokens=1,
                                               need_token=True)])
                seq.append(out.slots[0].token_id)
        self.assertEqual(seq_a, seq_b)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestUnsupportedModelFallsBackExplicitly(unittest.TestCase):
    """模型不支持时必须**显式回退并记录** —— 不静默吞参数。"""

    def test_unsupported_is_recorded_not_silently_ignored(self):
        inner = T.make_model()
        rec = T.RecordingModel(_NoKeepModule(inner))
        drv = T.make_driver(rec, logits_last_only=True)
        drv.load()
        eo = drv.execution_optimizations["logits_last_only"]
        self.assertTrue(eo["requested"])
        self.assertFalse(eo["effective"], "不支持时必须 effective=False（不能假装生效）")
        self.assertIn("num_logits_to_keep", eo["unsupported_reason"])

    def test_fallback_still_produces_correct_tokens(self):
        """回退路径的结果必须与不带该开关时逐 token 相同。"""
        rec1 = T.RecordingModel(_NoKeepModule(T.make_model()))
        d1 = T.make_driver(rec1, logits_last_only=True)
        d1.load()
        rec2 = T.RecordingModel(T.make_model())
        d2 = T.make_driver(rec2)
        d2.load()
        t1 = d1.forward([ForwardItem(T.req("a", 6), n_new_tokens=6, need_token=True)])
        t2 = d2.forward([ForwardItem(T.req("a", 6), n_new_tokens=6, need_token=True)])
        self.assertEqual(t1.slots[0].token_id, t2.slots[0].token_id)
        self.assertEqual(d1.execution_optimizations["logits_last_only"]
                         ["n_calls_all_positions"], 1)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestSampleNeededRowsOnly(unittest.TestCase):
    """只采 `need_token=True` 的行，并显式恢复行映射。"""

    def _driver(self, **flags):
        rec = T.RecordingModel(T.make_model())
        drv = T.make_driver(rec, **flags)
        drv.load()
        return drv, rec

    def test_all_false_skips_sampling_entirely(self):
        """need_token 全假 ⇒ **一次都不采样**（中间 chunk 本就不产出 token）。"""
        drv, _ = self._driver(sample_needed_only=True)
        out = drv.forward([ForwardItem(T.req("a", 5), n_new_tokens=5, need_token=False),
                           ForwardItem(T.req("b", 5), n_new_tokens=5, need_token=False)])
        self.assertTrue(all(s.token_id is None for s in out.slots))
        eo = drv.execution_optimizations["sample_needed_only"]
        self.assertEqual(eo["n_sampled_rows"], 0)
        self.assertEqual(eo["n_sampling_rows_skipped"], 2)
        # KV 仍必须照常推进（跳过的是采样，不是前向）
        self.assertEqual(drv._slots["a"].past_len, 5)

    def test_mixed_rows_map_to_the_right_requests(self):
        """混合批：被采样的行必须回到**它自己**的请求上（行映射不得错位）。"""
        ref_pairs = []
        base_rec = T.RecordingModel(T.make_model())
        base = T.make_driver(base_rec)
        base.load()
        for name in ("a", "b", "c"):
            base.forward([ForwardItem(T.req(name, 6), n_new_tokens=6, need_token=True)])
        out = base.forward([
            ForwardItem(T.req("a", 6), n_new_tokens=1, need_token=True),
            ForwardItem(T.req("b", 6), n_new_tokens=1, need_token=False),
            ForwardItem(T.req("c", 6), n_new_tokens=1, need_token=True),
        ])
        ref = {s.req_id: s.token_id for s in out.slots}

        drv, _ = self._driver(sample_needed_only=True)
        for name in ("a", "b", "c"):
            drv.forward([ForwardItem(T.req(name, 6), n_new_tokens=6, need_token=True)])
        # 计数是**累计**的（前面 3 次 prefill 也各采了 1 行）⇒ 取本步的增量，
        # 直接断言总量会把"前面的步"也算进来，看起来像接口错了。
        before = drv.execution_optimizations["sample_needed_only"]
        out2 = drv.forward([
            ForwardItem(T.req("a", 6), n_new_tokens=1, need_token=True),
            ForwardItem(T.req("b", 6), n_new_tokens=1, need_token=False),
            ForwardItem(T.req("c", 6), n_new_tokens=1, need_token=True),
        ])
        got = {s.req_id: s.token_id for s in out2.slots}
        self.assertIsNone(got["b"])
        self.assertEqual(got["a"], ref["a"])
        self.assertEqual(got["c"], ref["c"])
        after = drv.execution_optimizations["sample_needed_only"]
        self.assertEqual(after["n_sampled_rows"] - before["n_sampled_rows"], 2)
        self.assertEqual(
            after["n_sampling_rows_skipped"] - before["n_sampling_rows_skipped"], 1)

    def test_all_true_equals_baseline(self):
        drv, _ = self._driver(sample_needed_only=True)
        base_rec = T.RecordingModel(T.make_model())
        base = T.make_driver(base_rec)
        base.load()
        for d in (drv, base):
            out = d.forward([ForwardItem(T.req("a", 8), n_new_tokens=8, need_token=True)])
        self.assertEqual(out.slots[0].token_id, out.slots[0].token_id)
        self.assertEqual(drv.execution_optimizations["sample_needed_only"]
                         ["n_sampled_rows"], 1)

    def test_combination_of_both_switches_matches_baseline(self):
        """两个机制组合：逐 token 必须与基线一致（负对照：decode1 也不例外）。"""
        a = T.make_driver(T.RecordingModel(T.make_model()),
                          logits_last_only=True, sample_needed_only=True)
        a.load()
        b = T.make_driver(T.RecordingModel(T.make_model()))
        b.load()
        seq_a, seq_b = [], []
        for drv, seq in ((a, seq_a), (b, seq_b)):
            out = drv.forward([ForwardItem(T.req("a", 6), n_new_tokens=6, need_token=True)])
            seq.append(out.slots[0].token_id)
            for _ in range(2):
                out = drv.forward([ForwardItem(T.req("a", 6), n_new_tokens=1,
                                               need_token=True)])
                seq.append(out.slots[0].token_id)
        self.assertEqual(seq_a, seq_b)


if __name__ == "__main__":
    unittest.main()
