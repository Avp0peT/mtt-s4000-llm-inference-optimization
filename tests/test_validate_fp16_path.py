"""FP16 验收脚本的**判据本身**是否可信（本机可跑，不需要设备）。

为什么必须有这个文件
--------------------
这个脚本的产物是 PHASE 1 第②步"数值路径已验证"的唯一依据，而它原先有两条
**永远不会失败**的路径：

1. ``passed = not failures``，而"token 分岔但分岔步取证失败""分岔步 gap 不小"
   这两种情况只写 ``notes`` —— 于是"我们不知道分岔可不可解释"拿到了 passed=true。
2. ``V4`` 在分块 prefill 之后立刻 ``retire``，从不继续 decode。产物里因此出现
   "参考生成 16 个 token、分块驱动只生成 1 个 token"而判据仍然 passed：
   它验证的是**部分**检查，却被读成"完整路径验收通过"。

两条都属于"守卫不会失败"，而**不会失败的守卫等于没有守卫**：它不报错、
不留痕，还会被当成一次成功的验收。因此这里测的不是"脚本能跑通"，
而是"**这几条判据真的会失败**"——每条判据都配一个正对照：
把应当被拒绝的输入喂进去，断言它确实被拒绝。

本文件全部在 CPU 上用微型随机模型跑（``transformers`` + ``torch`` 缺失时自动跳过）。
"测出结论"与"测不出结论"必须分开：跳过会显式说明原因。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import validate_fp16_path as V  # noqa: E402

TORCH_MISSING_REASON = ""

try:  # pragma: no cover - 环境探测
    import torch
    from torch import nn
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from ascend_opt import Request
    from real_backend import ForwardItem, SyntheticPromptSource, TransformerDriver
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_MISSING_REASON = f"{type(exc).__name__}: {exc}"

#: **凡是类体、``setUpClass`` 或任一测试/辅助方法里会碰到 ``torch`` 的类，都必须加这个装饰器。**
#:
#: 不加的后果不是"测试失败"，而是**缺依赖时整类报 ERROR**——ERROR 与 FAIL 在报告里同形，
#: 但它其实完全没执行到断言；把"环境缺依赖"读成"代码坏了"，或者反过来把它当背景噪音
#: 长期不追，两种都会掩盖真实缺陷（AGENTS 附录 A5：区分"测不出"与"测出来是坏的"）。
#:
#: ⚠️ 判据是"**这里会不会调用 torch**"，**不是**"类名像不像数值测试"。
#: 2026-09-24 实测的形状：12 个类里只有 1 个加过守卫，另 2 个在**静态辅助方法**里
#: 裸调 ``torch.tensor`` ⇒ 在无 torch 的解释器下 8 项报 ERROR。
_requires_torch = unittest.skipIf(
    torch is None, f"环境缺少 torch/transformers：{TORCH_MISSING_REASON}"
)


# --------------------------------------------------------------------------- #
# 纯判据：不需要模型
# --------------------------------------------------------------------------- #


class TestAcceptanceRule(unittest.TestCase):
    """总通过条件：``failures`` 与 ``needs_review`` 都必须为空。"""

    def test_passes_only_when_both_lists_empty(self):
        self.assertTrue(V.overall_passed([], []))
        self.assertFalse(V.overall_passed(["某个判据不通过"], []))
        self.assertFalse(V.overall_passed([], ["某个必须人看的现象"]))

    def test_reverse_control_the_old_formula_would_have_passed(self):
        """**正对照**：证明"只判 failures"的旧写法在这组输入上会放行。

        这条断言的价值在于：如果有人把 ``overall_passed`` 改回
        ``not failures``，它会失败。没有这一条，测试只能证明新写法"能用"，
        不能证明它挡住了原来漏掉的那一类。
        """
        needs_review = ["V3 行 v3_0: 分岔步取证失败（…）——需复核"]
        old_formula = not []  # 旧实现：passed = not failures
        self.assertTrue(old_formula, "旧写法在这组输入上确实会放行（这正是要修的）")
        self.assertFalse(
            V.overall_passed([], needs_review),
            "新写法必须拦住 needs_review",
        )

    def test_every_blocking_verdict_is_a_known_verdict(self):
        """阻断集合只能由已知裁决组成，防止拼错的字符串悄悄变成"不阻断"。"""
        known = {
            V.DIVERGENCE_NEAR_TIE,
            V.DIVERGENCE_FORENSICS_FAILED,
            V.DIVERGENCE_NOT_NEAR_TIE,
        }
        self.assertTrue(V.BLOCKING_VERDICTS <= known)
        self.assertNotIn(V.DIVERGENCE_NEAR_TIE, V.BLOCKING_VERDICTS)


class TestDivergenceClassification(unittest.TestCase):
    """分岔取证的三态分流；关键在于**两个"说不清"的分支都要阻断**。"""

    def test_near_tie_does_not_block(self):
        verdict, msg = V.classify_token_divergence(
            "V3 行 v3_0",
            {"index": 3, "top1": 5, "top2": 6, "top1_top2_gap": 0.0},
        )
        self.assertEqual(verdict, V.DIVERGENCE_NEAR_TIE)
        self.assertFalse(V.blocks_acceptance(verdict))
        self.assertIn("不是实现错误", msg)
        self.assertIn("0.0000", msg)

    def test_forensics_failure_blocks(self):
        verdict, msg = V.classify_token_divergence(
            "V3 行 v3_1", {"error": "RuntimeError: boom", "index": 2}
        )
        self.assertEqual(verdict, V.DIVERGENCE_FORENSICS_FAILED)
        self.assertTrue(V.blocks_acceptance(verdict))
        self.assertIn("取证失败", msg)
        self.assertIn("需复核", msg)

    def test_missing_forensics_result_blocks(self):
        """``det=None``（取证函数返回空）也必须阻断，不能当成"没有问题"。"""
        for det in (None, {}):
            with self.subTest(det=det):
                verdict, msg = V.classify_token_divergence("V3 行 v3_2", det)
                self.assertEqual(verdict, V.DIVERGENCE_FORENSICS_FAILED)
                self.assertTrue(V.blocks_acceptance(verdict))
                self.assertIn("取证结果为空", msg)

    def test_result_without_gap_field_blocks(self):
        """有取证结果但缺 gap：同样"说不清"，同样阻断。

        这是最容易漏的一支——``det`` 非空且不含 ``error``，
        只按既有两个分支判断会掉进 ``else``（"gap 不够小"），
        把一次字段缺失说成一次确定的实现问题。
        """
        verdict, msg = V.classify_token_divergence("V3 行 v3_3", {"index": 1})
        self.assertEqual(verdict, V.DIVERGENCE_FORENSICS_FAILED)
        self.assertTrue(V.blocks_acceptance(verdict))
        self.assertIn("没有 gap 字段", msg)

    def test_large_gap_blocks(self):
        verdict, msg = V.classify_token_divergence(
            "V4b 多步 decode",
            {"index": 2, "top1": 5, "top2": 6, "top1_top2_gap": 3.7},
        )
        self.assertEqual(verdict, V.DIVERGENCE_NOT_NEAR_TIE)
        self.assertTrue(V.blocks_acceptance(verdict))
        self.assertIn("需复核", msg)

    def test_threshold_boundary_is_exclusive(self):
        """``gap == NEAR_TIE_GAP`` 不算近平局。

        边界定成"小于"还是"小于等于"是个真实分歧点：定成含等号，
        判据就在等号处悄悄放宽一档。这里把方向钉死。
        """
        at_boundary = V.classify_token_divergence(
            "x", {"index": 0, "top1": 1, "top2": 2, "top1_top2_gap": V.NEAR_TIE_GAP}
        )
        self.assertEqual(at_boundary[0], V.DIVERGENCE_NOT_NEAR_TIE)
        just_below = V.classify_token_divergence(
            "x",
            {"index": 0, "top1": 1, "top2": 2,
             "top1_top2_gap": V.NEAR_TIE_GAP - 1e-9},
        )
        self.assertEqual(just_below[0], V.DIVERGENCE_NEAR_TIE)


@_requires_torch
class TestStepwiseComparison(unittest.TestCase):
    """逐步分布比对：不得静默截断，不得把"缺一步"算成"一致"。"""

    @staticmethod
    def _t(values):
        return torch.tensor(values, dtype=torch.float32)

    def test_identical_steps_have_zero_diff(self):
        steps = [self._t([1.0, 2.0]), self._t([3.0, 4.0])]
        res = V.compare_step_logits(steps, steps)
        self.assertEqual(res["n_steps"], 2)
        self.assertEqual(res["max_abs_diff"], 0.0)

    def test_length_mismatch_is_an_error_not_a_pass(self):
        """**正对照**：驱动少跑 15 步，绝不能显示成"逐位一致"。

        旧缺陷的长相就是"参考 16 步、驱动 1 步"。用 ``zip`` 比会静默截断，
        于是短的那串看起来完全一致。
        """
        driver = [self._t([1.0, 2.0])]
        reference = [self._t([1.0, 2.0])] * 16
        res = V.compare_step_logits(driver, reference)
        self.assertIn("error", res)
        self.assertIn("步数不一致", res["error"])
        self.assertIn("16", res["error"])

    def test_missing_step_distribution_is_an_error(self):
        driver = [self._t([1.0, 2.0]), None]
        reference = [self._t([1.0, 2.0]), self._t([1.0, 2.0])]
        res = V.compare_step_logits(driver, reference)
        self.assertIn("error", res)
        self.assertIn("第 1 步分布缺失", res["error"])

    def test_worst_step_is_reported(self):
        driver = [self._t([0.0, 0.0]), self._t([0.0, 3.0]), self._t([0.0, 0.0])]
        reference = [self._t([0.0, 0.0])] * 3
        res = V.compare_step_logits(driver, reference)
        self.assertEqual(res["n_steps"], 3)
        self.assertEqual(res["worst_step"], 1)
        self.assertAlmostEqual(res["max_abs_diff"], 3.0, places=6)


class TestFirstTokenDivergence(unittest.TestCase):
    """首个分岔步；"说不清"与"没问题"必须是两个值。"""

    def test_identical_sequences_report_no_divergence_and_length_match(self):
        res = V.first_token_divergence([1, 2, 3], [1, 2, 3])
        self.assertTrue(res["length_match"])
        self.assertIsNone(res["first_divergence"])

    def test_divergence_index_is_reported(self):
        res = V.first_token_divergence([1, 9, 3], [1, 2, 3])
        self.assertTrue(res["length_match"])
        self.assertEqual(res["first_divergence"], 1)

    def test_length_mismatch_is_distinguishable_from_agreement(self):
        """**正对照**：长度不等时 ``first_divergence`` 也是 None，
        但 ``length_match=False`` 把它和"完全一致"分开。

        第一版直接在长度不等时返回 ``None``，于是
        "驱动只跑了 1 步、参考跑了 16 步"显示为"没有分岔"。
        """
        mismatch = V.first_token_divergence([1], [1, 2, 3])
        agreement = V.first_token_divergence([1, 2, 3], [1, 2, 3])
        self.assertIsNone(mismatch["first_divergence"])
        self.assertIsNone(agreement["first_divergence"])
        self.assertIsNot(
            mismatch["length_match"], agreement["length_match"],
            "两者必须靠 length_match 区分，否则'说不清'会被读成'没问题'",
        )
        self.assertFalse(mismatch["length_match"])
        self.assertEqual(mismatch["n_driver"], 1)
        self.assertEqual(mismatch["n_ref"], 3)


# --------------------------------------------------------------------------- #
# 机制：teacher forcing 是否真的把参考 token 喂了进去
# --------------------------------------------------------------------------- #


@_requires_torch
class TestTeacherForcingMechanism(unittest.TestCase):
    """``decode_with_teacher_forcing`` 的机制验证（CPU + 微型随机模型）。

    要证的不是"函数没崩"，而是**它注入的 token 真的成了下一步的输入**。
    这需要一个**会失败的正对照**：把 teacher 序列故意改错，
    如果实现没真的注入，结果就不会有任何变化——那样这个函数等于没做事。
    """

    VOCAB = 64
    PROMPT_LEN = 12
    GEN_LEN = 4

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cfg = Qwen2Config(
            vocab_size=cls.VOCAB,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
        )
        cls.base = Qwen2ForCausalLM(cfg)
        cls.base.eval()
        cls.src = SyntheticPromptSource(vocab_size=cls.VOCAB)
        cls.req = Request(req_id="v4", prompt_len=cls.PROMPT_LEN, gen_len=cls.GEN_LEN)
        cls.ids = cls.src.prompt_ids(cls.req)
        cls.ref_tokens, cls.ref_logits = V.reference_greedy_trace(
            cls.base, cls.ids, cls.GEN_LEN, "cpu"
        )

    def _driver(self):
        """产出 (驱动, 抓取器, prefill 那一步的分布)。

        抓取器包在模型外层，因此驱动每一次 ``forward`` 的 logits 都会被记下，
        与生产代码里的用法完全一致（生产路径上也是这么取分布的）。
        """

        class _CpuDriver(TransformerDriver):
            hardware_backed = False

            def _device_available(self, torch_mod: Any) -> bool:
                return True

        grab = V._LogitsGrabber(self.base)
        drv = _CpuDriver(
            model_name="local-tiny",
            device="cpu",
            precision="fp32",
            max_seq_len=512,
            model=grab,
            prompt_source=self.src,
        )
        drv.load()
        drv.forward(
            [ForwardItem(req=self.req, n_new_tokens=len(self.ids), need_token=True)]
        )
        prefill_lg = grab.last[0, -1, :].clone()
        return drv, grab, prefill_lg

    def test_teacher_tokens_reproduce_reference_step_logits(self):
        """注入参考 token 后，逐步分布必须与参考实现一致。"""
        drv, grab, prefill_lg = self._driver()
        own, steps = V.decode_with_teacher_forcing(
            drv, grab, self.req, self.ref_tokens, self.GEN_LEN
        )
        self.assertEqual(len(steps), self.GEN_LEN - 1)
        self.assertEqual(len(own), self.GEN_LEN - 1)

        driver_steps = [prefill_lg] + steps
        res = V.compare_step_logits(driver_steps, self.ref_logits)
        self.assertNotIn("error", res, res.get("error"))
        self.assertLess(
            res["max_abs_diff"], 1e-4,
            f"逐步分布与参考不一致：max|Δ|={res['max_abs_diff']:.3e}"
            f"（第 {res['worst_step']} 步）",
        )

    def test_corrupting_the_teacher_sequence_changes_the_result(self):
        """**正对照**：把 teacher token 改掉，分布必须随之改变。

        若实现只是"用驱动自己的采样结果继续"（即 teacher forcing 没生效），
        改 teacher 序列不会有任何影响——这条断言就会失败。
        它证的是"注入真的决定了下一步输入"，而不是"函数没抛异常"。
        """
        drv, grab, prefill_lg = self._driver()
        _, steps = V.decode_with_teacher_forcing(
            drv, grab, self.req, self.ref_tokens, self.GEN_LEN
        )
        corrupted = [(t + 1) % self.VOCAB for t in self.ref_tokens]
        self.assertNotEqual(corrupted[0], self.ref_tokens[0], "前提：确实改掉了")

        drv2, grab2, _ = self._driver()
        _, steps2 = V.decode_with_teacher_forcing(
            drv2, grab2, self.req, corrupted, self.GEN_LEN
        )
        differs = (steps[0] - steps2[0]).abs().max().item()
        self.assertGreater(
            differs, 1e-3,
            "改掉 teacher token 后第一步分布竟然没变 —— 注入没有生效",
        )

    def test_released_slot_raises_instead_of_silently_stopping(self):
        """**正对照**：prefill 完就释放（旧实现的做法）必须显形报错。

        旧 V4 正是这么做的：``retire`` 之后不再 decode，"参考 16 个 token、
        驱动 1 个 token"也能 passed。这里断言那条路会抛错，
        而不是少跑几步然后照样返回。
        """
        drv, grab, _ = self._driver()
        drv.retire([self.req.req_id])
        with self.assertRaises(RuntimeError) as ctx:
            V.decode_with_teacher_forcing(
                drv, grab, self.req, self.ref_tokens, self.GEN_LEN
            )
        self.assertIn("提前释放", str(ctx.exception))

    def test_reference_shorter_than_gen_len_is_rejected(self):
        """参考序列不够长时必须**先报错**，不能在第 N 步索引越界。"""
        drv, grab, _ = self._driver()
        with self.assertRaises(ValueError) as ctx:
            V.decode_with_teacher_forcing(
                drv, grab, self.req, self.ref_tokens[:1], self.GEN_LEN
            )
        self.assertIn("无法构造相同历史", str(ctx.exception))

    def test_non_positive_gen_len_is_rejected(self):
        drv, grab, _ = self._driver()
        with self.assertRaises(ValueError):
            V.decode_with_teacher_forcing(drv, grab, self.req, self.ref_tokens, 0)

    def test_from_prefill_can_continue_without_teacher_forcing(self):
        """不注入 teacher（用驱动自喂）也能跑：只跑 1 步时必须与 prefill 自洽。

        这条覆盖"gen_len=1 ⇒ 不做 decode"的退化情形，保证这个参数
        不会因为多步逻辑而被改坏。
        """
        drv, grab, prefill_lg = self._driver()
        own, steps = V.decode_with_teacher_forcing(
            drv, grab, self.req, self.ref_tokens, 1
        )
        self.assertEqual(own, [])
        self.assertEqual(steps, [])
        self.assertIsNotNone(prefill_lg)


# --------------------------------------------------------------------------- #
# 2026-09-24 补的两处漏检
# --------------------------------------------------------------------------- #


class TestAllTokenDivergences(unittest.TestCase):
    """分岔必须**逐个**验收，不能只看首个。

    旧判据只取首个分岔步：首步若是近平局（可解释），后面某步 gap 很大
    （与"分布等价"直接矛盾）就被**完全掩盖**。而 teacher forcing 下每个位置
    的输入都来自参考 ⇒ 每一步都是彼此独立的验收点。
    """

    def test_reports_every_step_not_only_the_first(self):
        self.assertEqual(V.all_token_divergences([1, 2, 3, 4, 5], [1, 9, 3, 8, 5]), [1, 3])

    def test_first_divergence_does_not_hide_later_ones(self):
        """这就是旧漏洞的形状：第 0 步与第 2 步都分岔，清单里必须两步都在。"""
        self.assertEqual(V.all_token_divergences([7, 1, 1, 2], [3, 1, 1, 2]), [0])
        self.assertEqual(V.all_token_divergences([7, 1, 9, 2], [3, 1, 1, 2]), [0, 2])

    def test_empty_when_identical(self):
        self.assertEqual(V.all_token_divergences([1, 2, 3], [1, 2, 3]), [])

    def test_only_common_prefix_is_compared(self):
        # 长度不等本身由调用方按失败处置；这里只保证不会越界
        self.assertEqual(V.all_token_divergences([1, 9], [1, 2, 3, 4]), [1])


class TestDivergenceDetailAtIndex(unittest.TestCase):
    """``_first_divergence_detail(index=...)`` 必须能在**指定的**步取证，
    并**拒绝**与事实不符的清单——否则"逐个验收"会退化成"随便取一步的 gap"。"""

    VOCAB = 64

    @classmethod
    def setUpClass(cls):
        if torch is None:  # pragma: no cover
            raise unittest.SkipTest(TORCH_MISSING_REASON)
        torch.manual_seed(0)
        cfg = Qwen2Config(
            vocab_size=cls.VOCAB, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512,
        )
        cls.base = Qwen2ForCausalLM(cfg)
        cls.base.eval()

    def test_identical_sequences_are_reported_as_inconsistent(self):
        det = V._first_divergence_detail(self.base, [1, 2, 3], [1, 2, 3], [1, 2, 3], "cpu")
        self.assertIn("error", det)

    def test_index_that_is_not_a_divergence_is_rejected(self):
        """传一个**并非**分岔的步：必须显形为错误，不能返回一个看似正常的 gap。"""
        det = V._first_divergence_detail(
            self.base, [1, 2, 3], [1, 9, 3], [1, 2, 3], "cpu", index=2
        )
        self.assertIn("error", det)

    def test_divergent_index_gives_gap(self):
        det = V._first_divergence_detail(
            self.base, [1, 2, 3], [1, 9, 3], [1, 2, 3], "cpu", index=1
        )
        self.assertNotIn("error", det)
        self.assertIn("top1_top2_gap", det)
        self.assertEqual(det["index"], 1)

    def test_default_still_returns_the_first_divergence(self):
        det = V._first_divergence_detail(
            self.base, [1, 2, 3], [1, 9, 8], [1, 2, 3], "cpu"
        )
        self.assertEqual(det["index"], 1)


@_requires_torch
class TestStepRowCapture(unittest.TestCase):
    """``_StepRowCapture`` 的两个判据：多步不误判歧义；"合批"要能被证伪。"""

    def _cap(self):
        class _Out:
            def __init__(self, lg):
                self.logits = lg

        def inner(*_a, **k):
            n = int(k["input_ids"].shape[0])
            return _Out(torch.zeros(n, 1, 8))

        return V._StepRowCapture(inner)

    def test_same_token_repeated_across_steps_is_not_ambiguous(self):
        """「同一个 token 在不同步再次出现」在多步 decode 里很常见，不算歧义。

        这正是不能复用 ``_PerRowCapture`` 的原因（它跨调用累计 key 并判歧义）。
        """
        cap = self._cap()
        cap(input_ids=torch.tensor([[5]]))
        cap(input_ids=torch.tensor([[5]]))
        self.assertEqual(cap.call_containing([(1, 5)]), 0)
        self.assertEqual(cap.call_containing([(1, 5)], start=1), 1)

    def test_two_single_row_forwards_are_not_a_batch(self):
        """两次单行前向**不能**被当成"合批"——否则补位检查又会退化成空跑。"""
        cap = self._cap()
        cap(input_ids=torch.tensor([[5]]))
        cap(input_ids=torch.tensor([[6]]))
        self.assertIsNone(cap.call_containing([(1, 5), (1, 6)]))

    def test_two_rows_in_one_forward_is_a_batch(self):
        cap = self._cap()
        cap(input_ids=torch.tensor([[7], [8]]))
        self.assertEqual(cap.call_containing([(1, 7), (1, 8)]), 0)

    def test_duplicate_tokens_in_one_forward_are_not_judged(self):
        """同一 token 两行 ⇒ 无法按内容归属 ⇒ 弃权（不放行），不是"通过"。"""
        cap = self._cap()
        cap(input_ids=torch.tensor([[9], [9]]))
        self.assertIsNone(cap.call_containing([(1, 9)]))


class TestMixedHistoryBatchedDecode(unittest.TestCase):
    """**不同历史长度的合批 decode**——V3 真正覆盖补位的那一段。

    要证两件事，缺一不可：
      ① 合批真的发生（多行在同一次 forward 里，且各行 ``past_len`` 参差）；
      ② 判据真的会失败（把参考分布改坏，差异必须被判出来）。
    只会"通过"的检查等于没有检查，所以 ② 是必须的正对照。
    """

    VOCAB = 64
    LENGTHS = [12, 18, 24]
    N_STEPS = 2

    @classmethod
    def setUpClass(cls):
        if torch is None:  # pragma: no cover
            raise unittest.SkipTest(TORCH_MISSING_REASON)
        torch.manual_seed(0)
        cfg = Qwen2Config(
            vocab_size=cls.VOCAB, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512,
        )
        cls.base = Qwen2ForCausalLM(cfg)
        cls.base.eval()
        cls.src = SyntheticPromptSource(vocab_size=cls.VOCAB)
        cls.reqs = [
            Request(req_id=f"m{i}", prompt_len=n, gen_len=4)
            for i, n in enumerate(cls.LENGTHS)
        ]
        cls.same = [
            Request(req_id=f"s{i}", prompt_len=16, gen_len=4) for i in range(3)
        ]

    def _mk(self, reqs, extra, ref_mutator=None):
        class _CpuDriver(TransformerDriver):
            hardware_backed = False

            def _device_available(self, torch_mod: Any) -> bool:
                return True

        cap = V._StepRowCapture(self.base)
        drv = _CpuDriver(
            model_name="local-tiny", device="cpu", precision="fp32",
            max_seq_len=512, model=cap, prompt_source=self.src,
        )
        drv.load()
        drv.forward(
            [ForwardItem(req=r, n_new_tokens=r.prompt_len, need_token=True) for r in reqs]
        )
        # 强制 token **由我们挑**：逐行归属按喂进去的 token 做，
        # 必须保证各行在同一位置互不相同（此处每行偏移 7，vocab=64 够用）。
        forced = {
            r.req_id: [3 + 7 * i + j for j in range(int(extra[i]) + self.N_STEPS + 1)]
            for i, r in enumerate(reqs)
        }
        ref = {}
        for i, r in enumerate(reqs):
            lgs, _ = V.reference_forced_trace(
                self.base, self.src.prompt_ids(r), forced[r.req_id], "cpu"
            )
            if ref_mutator is not None:
                lgs = ref_mutator(lgs, i)
            ref[r.req_id] = lgs
        return drv, cap, forced, ref

    def test_batch_is_real_and_histories_differ(self):
        extra = [0, 1, 2]
        drv, cap, forced, ref = self._mk(self.reqs, extra)
        res = V.batched_mixed_history_decode(
            drv, cap, self.reqs, forced, extra, self.N_STEPS, ref
        )
        self.assertEqual(res["errors"], [])
        # ① 合批：每一步都是"多行在同一次 forward 里"
        self.assertEqual(res["rows_per_call"], [len(self.reqs)] * self.N_STEPS)
        # ② 补位前提：各行的 KV 长度互不相同
        kv = [h["kv_len_at_first_batch_step"] for h in res["histories"].values()]
        self.assertEqual(len(set(kv)), len(kv))
        # ③ 逐行分布等价
        mx = max(
            row["max_abs_diff"]
            for rows in res["rows"].values()
            for row in rows
            if "max_abs_diff" in row
        )
        self.assertLessEqual(mx, V.FP16_LOGIT_TOL)

    def test_equal_history_lengths_are_rejected(self):
        """各行历史一样长 ⇒ 合批时不会补位 ⇒ 本检查不成立，必须显形。

        用**等长 prompt** 的请求构造：否则 prompt 长度不同本身就带来了差异，
        这个前提就永远为真，判据也就永远不触发。
        """
        extra = [0, 0, 0]
        drv, cap, forced, ref = self._mk(self.same, extra)
        res = V.batched_mixed_history_decode(
            drv, cap, self.same, forced, extra, self.N_STEPS, ref
        )
        self.assertTrue(res["errors"])
        self.assertIn("不会发生补位", " ".join(res["errors"]))

    def test_wrong_reference_is_detected(self):
        """正对照：把参考分布改坏，差异必须被判出来（判据不是恒真）。"""
        extra = [0, 1, 2]

        def _break(lgs, _i):
            return [lg * -1.0 for lg in lgs]

        drv, cap, forced, ref = self._mk(self.reqs, extra, ref_mutator=_break)
        res = V.batched_mixed_history_decode(
            drv, cap, self.reqs, forced, extra, self.N_STEPS, ref
        )
        mx = max(
            row["max_abs_diff"]
            for rows in res["rows"].values()
            for row in rows
            if "max_abs_diff" in row
        )
        self.assertGreater(mx, V.FP16_LOGIT_TOL)


class TestGroupedPrefillDoesNotClaimPadding(unittest.TestCase):
    """锁住一个事实：**各行 ``n_new`` 不同时不会合批，也不会补位**。

    旧 V3 的"补位检查"就是这么写的（每行传 ``n_new_tokens=r.prompt_len``），
    产物里却报 ``padding_preserves_rows: true``。把"每次前向只有 1 行"这个事实
    记进产物，缺口才不会再次被藏住。
    """

    VOCAB = 64

    @classmethod
    def setUpClass(cls):
        if torch is None:  # pragma: no cover
            raise unittest.SkipTest(TORCH_MISSING_REASON)
        torch.manual_seed(0)
        cfg = Qwen2Config(
            vocab_size=cls.VOCAB, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512,
        )
        cls.base = Qwen2ForCausalLM(cfg)
        cls.base.eval()
        cls.src = SyntheticPromptSource(vocab_size=cls.VOCAB)

    def test_mixed_n_new_is_split_into_single_row_forwards(self):
        class _CpuDriver(TransformerDriver):
            hardware_backed = False

            def _device_available(self, torch_mod: Any) -> bool:
                return True

        cap = V._PerRowCapture(self.base)
        drv = _CpuDriver(
            model_name="local-tiny", device="cpu", precision="fp32",
            max_seq_len=512, model=cap, prompt_source=self.src,
        )
        drv.load()
        reqs = [
            Request(req_id=f"g{i}", prompt_len=n, gen_len=2)
            for i, n in enumerate([16, 24, 33, 48])
        ]
        drv.forward(
            [ForwardItem(req=r, n_new_tokens=r.prompt_len, need_token=False) for r in reqs]
        )
        # 4 个不同的 n_new ⇒ 4 次前向、每次 1 行 ⇒ pad 恒为 0
        self.assertEqual(cap.call_rows, [1, 1, 1, 1])
        self.assertEqual(len(set(cap.call_n_new)), 4)
        self.assertEqual(max(cap.call_rows), 1)


class TestJudgedDivergenceSteps(unittest.TestCase):
    """能不能"逐个验收"取决于**比较口径**，不取决于序列本身。

    踩过（真机首跑，2026-09-24 7B）：把 V4b 的口径（teacher forcing ⇒
    每步的历史都等于参考 ⇒ 每步都是独立验收点）套到 V3 上——而 V3 的两条序列
    是**贪心自喂**的，第 4 步一分岔，第 6~15 步的"分岔"就只是它的下游
    （gap 0.24~6.51），逐个验收凭空造出 10 条"需复核"。
    """

    def test_shared_history_returns_every_step(self):
        self.assertEqual(
            V.judged_divergence_steps([1, 2, 3, 4], [1, 9, 3, 8], shared_history=True),
            [1, 3],
        )

    def test_greedy_returns_only_the_first(self):
        self.assertEqual(
            V.judged_divergence_steps([1, 2, 3, 4], [1, 9, 3, 8], shared_history=False),
            [1],
        )

    def test_greedy_with_no_divergence_is_empty(self):
        self.assertEqual(
            V.judged_divergence_steps([1, 2], [1, 2], shared_history=False), []
        )

    def test_greedy_is_a_prefix_of_shared_history(self):
        """两个口径**只在首个分岔步上**一致 —— 贪心是共享历史的真子集。"""
        a, b = [7, 1, 9, 2, 5], [3, 1, 1, 2, 6]
        first = V.judged_divergence_steps(a, b, shared_history=False)
        every = V.judged_divergence_steps(a, b, shared_history=True)
        self.assertTrue(every)
        self.assertEqual(first, every[:1])

    def test_length_mismatch_yields_empty_not_a_fake_step(self):
        """长度不等时**没有**"第几步分岔"可言 ⇒ 空表，由调用方判失败。"""
        self.assertEqual(
            V.judged_divergence_steps([1, 2], [1, 2, 3, 4], shared_history=True), []
        )


class TestResourceLifecycleVerdict(unittest.TestCase):
    """KV 槽位判据的三态：该失败时会失败、该通过时会通过、计数器不可信时显形。

    这条判据第一版在 ``retire`` **之前**读 ``live_slots()`` ⇒ 读到的永远是 1
    ⇒ 判据"收尾应为 0"**恒不成立**。真机首跑（2026-09-24，0.5B/7B 各一次）
    5 个 V4 切法全部报"仍有 1 个常驻 KV 槽"。恒假判据与恒真判据一样不携带信息。
    """

    def test_healthy_records_pass(self):
        f, r = V.resource_lifecycle_verdict(
            driver_live_at_end={"drv2": 0, "drv3": 0},
            v4_live_after_retire={"v4_[16, 16, 8]": 0},
            v4_live_before_retire={"v4_[16, 16, 8]": 1},
        )
        self.assertEqual(f, [])
        self.assertEqual(r, [])

    def test_driver_leak_is_a_failure(self):
        f, r = V.resource_lifecycle_verdict(
            driver_live_at_end={"drv3": 1},
            v4_live_after_retire={"v4_whole": 0},
            v4_live_before_retire={"v4_whole": 1},
        )
        self.assertTrue(any("drv3" in x for x in f), f)
        self.assertEqual(r, [])

    def test_v4_leak_is_a_failure(self):
        f, _ = V.resource_lifecycle_verdict(
            driver_live_at_end={},
            v4_live_after_retire={"v4_[20, 20]": 2},
            v4_live_before_retire={"v4_[20, 20]": 3},
        )
        self.assertTrue(any("v4_[20, 20]" in x for x in f), f)

    def test_dead_counter_is_flagged_not_silently_passed(self):
        """**反向对照**：``retire`` 之前就该是 1；恒 0 说明计数器坏了，
        那么"释放后为 0"是**什么都没测**得来的 ⇒ 必须进 needs_review。"""
        f, r = V.resource_lifecycle_verdict(
            driver_live_at_end={"drv2": 0},
            v4_live_after_retire={"v4_whole": 0},
            v4_live_before_retire={"v4_whole": 0},
        )
        self.assertEqual(f, [])
        self.assertTrue(any("计数器" in x for x in r), r)

    def test_none_means_unmeasurable_not_zero(self):
        """没有计数能力 ⇒ 不判，**不能**当成 0（"没查" ≠ "没问题"）。"""
        f, r = V.resource_lifecycle_verdict(
            driver_live_at_end={"drv2": None},
            v4_live_after_retire={},
            v4_live_before_retire={},
        )
        self.assertEqual(f, [])
        self.assertEqual(r, [])


if __name__ == "__main__":
    unittest.main()
