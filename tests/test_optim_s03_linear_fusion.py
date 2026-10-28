"""S03 / P3：QKV 3→1 与 gate/up 2→1 线性投影融合。

要保的性质：

1. **默认关闭**：不装融合、`state_dict` 与行为完全不变。
2. **权重只拼一次、且一个字节都不改**：`state_dict` 前后**逐键相同**；
   融合权重是 `cat` 出来的独立张量。
3. **数值等价**：开启任一/任一组合后，逐 token 与逐层 KV 必须与基线一致。
4. **复用守卫真的成立**：QKV 融合靠"q 先算、k/v 复用"，但复用必须同时校验
   `(data_ptr, shape)` **与剩余分量计数**；任一条不成立就**回退**，绝不返回陈旧值。
   ⚠️ 这条必须有**反例**：拿一个不同的输入张量去调 k/v，结果必须等于原 `F.linear`。
5. **切分代价被如实记账**：QKV 的窄片必须 `contiguous()`（HF 会对它做 `.view()`），
   这是融合的**固有代价**；gate/up 不需要（SiLU 与逐元素乘法对非连续张量可用）
   ⇒ 两者的 `bytes_copied` 应分别为 >0 与 0。

⚠️ 性能（M=1/2/4/8/128/512 的算子耗时与净收益）**不属于本地可验范围** ⇒ `REMOTE_PENDING`。
本文件只回答"融合是否正确、代价长什么样"。
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
import linear_fusion as LF  # noqa: E402
from real_backend import ForwardItem  # noqa: E402

if T.torch is not None:
    import torch  # noqa: E402
    import torch.nn.functional as F  # noqa: E402


def _gen(drv, name: str, prompt_len: int, steps: int = 3):
    seq = []
    out = drv.forward([ForwardItem(T.req(name, prompt_len), n_new_tokens=prompt_len,
                                   need_token=True)])
    seq.append(out.slots[0].token_id)
    for _ in range(steps):
        out = drv.forward([ForwardItem(T.req(name, prompt_len), n_new_tokens=1,
                                       need_token=True)])
        seq.append(out.slots[0].token_id)
    return seq


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestDisabledByDefault(unittest.TestCase):
    def test_no_fusion_installed_and_state_dict_unchanged(self):
        model = T.make_model()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        drv = T.make_driver(model)
        drv.load()
        self.assertIsNone(drv._fusion)
        eo = drv.execution_optimizations["linear_fusion"]
        self.assertIsNone(eo)
        after = model.state_dict()
        self.assertEqual(sorted(before), sorted(after))
        for k in before:
            self.assertTrue(torch.equal(before[k], after[k]))
        self.assertEqual(drv.execution_optimizations["logits_last_only"]["requested"], False)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestManifestFacts(unittest.TestCase):
    def test_qkv_manifest_reports_real_shapes_and_dtype(self):
        model = T.make_model(layers=2)
        m = LF.install_fusions(model, fuse_qkv=True)
        try:
            self.assertEqual(m.n_layers, 2)
            self.assertEqual(m.installed, 2)
            qkv = m.describe()["qkv"]
            # 微型模型：4 个 Q 头 × head_dim 8 = 32；2 个 KV 头 × 8 = 16
            self.assertEqual(qkv["split"], [32, 16, 16])
            self.assertEqual(qkv["weight_shape"], [64, 32])
            self.assertEqual(qkv["dtype"], "torch.float32")
            # ★ bias 的**有无**必须与真实模块一致：Qwen2 系（含 Qwen2.5）默认
            # `attention_bias=True` ⇒ q/k/v 都带 bias，融合必须把 bias 一起按同一
            # 顺序拼上，否则数值会偏（而形状仍然合法）。
            layer0 = LF.iter_layers(model)[0]
            self.assertEqual(qkv["bias"], layer0.self_attn.q_proj.bias is not None)
            self.assertTrue(qkv["need_contiguous_on_split"])
        finally:
            LF.uninstall_fusions(m)

    def test_weights_untouched_by_install(self):
        model = T.make_model(layers=1)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        m = LF.install_fusions(model, fuse_qkv=True, fuse_gate_up=True)
        try:
            after = model.state_dict()
            self.assertEqual(sorted(before), sorted(after), "state_dict 键不得变化")
            for k in before:
                self.assertTrue(torch.equal(before[k], after[k]), f"{k} 被改动了")
        finally:
            LF.uninstall_fusions(m)

    def test_uninstall_restores_class_forward(self):
        model = T.make_model(layers=1)
        layer = LF.iter_layers(model)[0]
        q_proj = layer.self_attn.q_proj
        orig_func = type(q_proj).forward
        m = LF.install_fusions(model, fuse_qkv=True)
        # 安装后 `forward` 变成**实例属性**（遮蔽类上的实现）
        self.assertIn("forward", vars(q_proj))
        self.assertIsNot(q_proj.forward, type(q_proj).forward)
        n = LF.uninstall_fusions(m)
        self.assertGreater(n, 0)
        # 卸载后实例属性被删掉 ⇒ 重新走类上的实现
        # （⚠️ 不能用 `assertIs` 比绑定方法：每次取属性都会新建一个绑定对象）
        self.assertNotIn("forward", vars(q_proj))
        self.assertIs(q_proj.forward.__func__, orig_func)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestNumericalEquivalence(unittest.TestCase):
    """全模型逐 token 等价性 —— 这是"形状对、数值错"的总闸门。"""

    def _run(self, **flags):
        drv = T.make_driver(T.RecordingModel(T.make_model()), **flags)
        drv.load()
        return drv, _gen(drv, "a", 7)

    def test_qkv_only(self):
        base, ref = self._run()
        fused, got = self._run(fuse_qkv=True)
        self.assertEqual(got, ref)
        self.assertGreater(fused._fusion.describe()["qkv"]["fused_calls"], 0)

    def test_gate_up_only(self):
        base, ref = self._run()
        fused, got = self._run(fuse_gate_up=True)
        self.assertEqual(got, ref)
        self.assertGreater(fused._fusion.describe()["gate_up"]["fused_calls"], 0)

    def test_both(self):
        base, ref = self._run()
        fused, got = self._run(fuse_qkv=True, fuse_gate_up=True)
        self.assertEqual(got, ref)
        d = fused._fusion.describe()
        self.assertGreater(d["qkv"]["fused_calls"], 0)
        self.assertGreater(d["gate_up"]["fused_calls"], 0)

    def test_kv_states_identical(self):
        base = T.make_driver(T.make_model())
        base.load()
        fused = T.make_driver(T.make_model(), fuse_qkv=True, fuse_gate_up=True)
        fused.load()
        base.forward([ForwardItem(T.req("a", 6), n_new_tokens=6, need_token=True)])
        fused.forward([ForwardItem(T.req("a", 6), n_new_tokens=6, need_token=True)])
        self.assertEqual(base._slots["a"].past_len, fused._slots["a"].past_len)
        for (ka, va), (kb, vb) in zip(base._slots["a"].past, fused._slots["a"].past):
            self.assertTrue(torch.allclose(ka, kb, atol=1e-6))
            self.assertTrue(torch.allclose(va, vb, atol=1e-6))

    def test_with_last_position_switch_too(self):
        base = T.make_driver(T.make_model())
        base.load()
        ref = _gen(base, "a", 6)
        fused = T.make_driver(T.make_model(), fuse_qkv=True, fuse_gate_up=True,
                              logits_last_only=True, sample_needed_only=True)
        fused.load()
        self.assertEqual(_gen(fused, "a", 6), ref)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestReuseGuard(unittest.TestCase):
    """★ 反例：复用必须被 `(data_ptr, shape)` + 剩余计数守住。"""

    def _layer(self):
        model = T.make_model(layers=1)
        m = LF.install_fusions(model, fuse_qkv=True, fuse_gate_up=True)
        layer = LF.iter_layers(model)[0]
        return model, m, layer

    def test_k_proj_with_a_different_input_falls_back(self):
        """q 用的是 x1，k 收到 x2 ⇒ **不得**复用 q 的陈旧结果。"""
        torch.manual_seed(3)
        _, m, layer = self._layer()
        try:
            attn = layer.self_attn
            x1 = torch.randn(1, 4, 32)
            x2 = torch.randn(1, 4, 32)
            attn.q_proj(x1)                      # 建立桩
            got = attn.k_proj(x2)                # 不同输入 ⇒ 必须回退
            ref = F.linear(x2, attn.k_proj.weight, attn.k_proj.bias)
            self.assertTrue(torch.allclose(got, ref, atol=1e-6),
                            "不同输入却复用了陈旧结果 ⇒ 静默算错")
            self.assertGreater(m.groups[0].n_fallback_calls, 0)
        finally:
            LF.uninstall_fusions(m)

    def test_v_proj_called_first_falls_back(self):
        """v 先于 q 被调用 ⇒ 没有桩可用 ⇒ 回退到原 `F.linear`（结果仍正确）。"""
        torch.manual_seed(4)
        _, m, layer = self._layer()
        try:
            attn = layer.self_attn
            x = torch.randn(1, 3, 32)
            got = attn.v_proj(x)
            ref = F.linear(x, attn.v_proj.weight, attn.v_proj.bias)
            self.assertTrue(torch.allclose(got, ref, atol=1e-6))
            qkv = m.describe()["qkv"]
            self.assertEqual(qkv["fused_calls"], 0)
        finally:
            LF.uninstall_fusions(m)

    def test_q_then_k_then_v_reuses_twice(self):
        torch.manual_seed(5)
        _, m, layer = self._layer()
        try:
            attn = layer.self_attn
            x = torch.randn(1, 5, 32)
            q = attn.q_proj(x)
            k = attn.k_proj(x)
            v = attn.v_proj(x)
            self.assertTrue(torch.allclose(q, F.linear(x, attn.q_proj.weight), atol=1e-6))
            self.assertTrue(torch.allclose(k, F.linear(x, attn.k_proj.weight), atol=1e-6))
            self.assertTrue(torch.allclose(v, F.linear(x, attn.v_proj.weight), atol=1e-6))
            qkv = m.describe()["qkv"]
            self.assertEqual(qkv["fused_calls"], 1)
            self.assertEqual(qkv["reuse_hits"], 2)
            self.assertEqual(qkv["fallback_calls"], 0)
        finally:
            LF.uninstall_fusions(m)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestShapeSelection(unittest.TestCase):
    """形状选择：只在集合内走融合，其余回退；结果必须都正确。"""

    def test_only_listed_shapes_are_fused(self):
        drv = T.make_driver(T.RecordingModel(T.make_model()),
                            fuse_qkv=True, linear_fuse_shapes=[5])
        drv.load()
        # prefill 5 token ⇒ M=5 ⇒ 融合
        drv.forward([ForwardItem(T.req("a", 5), n_new_tokens=5, need_token=True)])
        d1 = drv._fusion.describe()["qkv"]
        self.assertGreater(d1["fused_calls"], 0)
        self.assertEqual(d1["skipped_by_shape"], 0)
        # decode 1 token ⇒ M=1 不在集合内 ⇒ 回退
        drv.forward([ForwardItem(T.req("a", 5), n_new_tokens=1, need_token=True)])
        d2 = drv._fusion.describe()["qkv"]
        self.assertGreater(d2["skipped_by_shape"], 0)

    def test_selection_does_not_change_tokens(self):
        base = T.make_driver(T.make_model())
        base.load()
        ref = _gen(base, "a", 5, steps=2)
        sel = T.make_driver(T.make_model(), fuse_qkv=True, fuse_gate_up=True,
                            linear_fuse_shapes=[5])
        sel.load()
        self.assertEqual(_gen(sel, "a", 5, steps=2), ref)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestSplitCostAccounting(unittest.TestCase):
    """★ 切分代价必须记账，且两种融合的代价形状**不同**。"""

    def test_qkv_pays_contiguous_copies_but_gate_up_does_not(self):
        drv = T.make_driver(T.RecordingModel(T.make_model()),
                            fuse_qkv=True, fuse_gate_up=True)
        drv.load()
        drv.forward([ForwardItem(T.req("a", 6), n_new_tokens=6, need_token=True)])
        d = drv._fusion.describe()
        self.assertGreater(d["qkv"]["split_copies"], 0, "QKV 窄片必须 contiguous")
        self.assertGreater(d["qkv"]["bytes_copied"], 0)
        # gate/up 不需要 contiguous（下游是 SiLU 与逐元素乘法）
        self.assertFalse(d["gate_up"]["need_contiguous_on_split"])
        self.assertEqual(d["gate_up"]["split_copies"], 0)
        self.assertEqual(d["gate_up"]["bytes_copied"], 0)

    def test_qkv_slices_are_contiguous(self):
        torch.manual_seed(6)
        model = T.make_model(layers=1)
        m = LF.install_fusions(model, fuse_qkv=True)
        try:
            attn = LF.iter_layers(model)[0].self_attn
            x = torch.randn(1, 4, 32)
            for name in ("q_proj", "k_proj", "v_proj"):
                out = getattr(attn, name)(x)
                self.assertTrue(out.is_contiguous(), f"{name} 的输出必须连续（下游要 .view()）")
        finally:
            LF.uninstall_fusions(m)


if __name__ == "__main__":
    unittest.main()
