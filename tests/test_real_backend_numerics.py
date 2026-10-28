"""真机驱动的**数值等价性**测试（CPU + 微型随机模型）。

为什么必须有这个文件
--------------------
真机驱动的风险不在"会不会崩"，而在**形状对、数值错**。补位、position_ids、
KV 写回三处都是典型的"不报错但算错"：

- 左补位之后若忘掉 position_ids 的前缀和修正，位置编号会整体偏移，
  注意力仍然算得出结果，只是结果不对；
- 写回 KV 时把切片写成 ``[:keep]``（而不是 ``[pad : pad + keep]``），
  就会把**别人的历史**当成自己的——形状完全正确。

这两处都不会有任何异常，只会让 TTFT/TPOT 测出一批"正常"的数。
因此这里用**确定性微型模型**做等价性比对：

    合法路径 A：某条序列单独跑
    合法路径 B：同一条序列与历史长度不同的序列合批跑

两条路径在同一时刻的 logits 必须相等（同一份权重、同一份输入、
同一套注意力语义时，批内补位不应改变任何一条序列的计算结果）。
只要补位/位置/写回有一处错，这个等式立刻不成立。

模型是随机初始化的（不需要联网、不需要下载），因为这里验证的是
**计算语义是否等价**，与权重是不是训练过完全无关。
"""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt import Request  # noqa: E402
from real_backend import (  # noqa: E402
    ForwardItem,
    SyntheticPromptSource,
    TransformerDriver,
)

TORCH_MISSING_REASON = ""

try:  # pragma: no cover - 环境探测
    import torch
    from torch import nn
    from transformers import Qwen2Config, Qwen2ForCausalLM
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_MISSING_REASON = f"{type(exc).__name__}: {exc}"


class _RecordingModel(nn.Module if torch is not None else object):
    """把内层模型的每步 logits 记下来，便于逐数值比对。

    只在测试侧存在，生产代码不需要为"可测"而在驱动里留调试口——
    调试口一旦存在，就有人会在正式运行里打开它。
    """

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self.inner = inner
        self.logits: list[Any] = []
        self.calls = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = self.inner(*args, **kwargs)
        self.logits.append(out.logits.detach().clone())
        self.calls += 1
        return out


class _CpuDriver(TransformerDriver):
    """把真机驱动跑在 CPU 上，**仅用于数值验证**。

    ``hardware_backed=False`` 是刻意的：它没有在硬件上执行，因此绝不能被
    当成性能证据（``RealBackend`` 会拒绝它）。这正是"本地 CPU 结果不得表述为
    硬件实测"这条纪律在代码里的落点——生产驱动在设备不可用时**报错**，
    不会自己退回 CPU；这个子类是测试专用的显式开口。
    """

    hardware_backed = False

    def _device_available(self, torch: Any) -> bool:
        return True


def _make_model(seed: int = 0, vocab: int = 64, layers: int = 2) -> Any:
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


@unittest.skipIf(torch is None, f"环境缺少 torch/transformers：{TORCH_MISSING_REASON}")
class TestLogitsDigest(unittest.TestCase):
    """★ `capture_logits_digest`（S04 重入验证用的逐行 logits 摘要）。

    为什么单独测：它是 2026-10-03 新加的**生产代码**，而它存在的唯一理由是
    "给真机验证提供比 token 更早暴露偏差的对照量"。如果它在真机上静默不产出
    （开关没接线、字段没回传），那么真机那份产物里的
    「logits 逐点相同」就会在**两个空 dict** 上恒真 —— 一次假通过。
    所以这里要同时验：**开了有货**、**关了没有**、**相同输入可复现**、
    **不同输入会变**（最后一条防"摘要恒为常数"）。
    """

    VOCAB = 64

    def _driver(self, *, capture: bool):
        tick = itertools.count()
        model = _make_model(vocab=self.VOCAB)
        drv = _CpuDriver(
            model_name="local-tiny",
            device="cpu",
            precision="fp32",
            max_seq_len=512,
            model=_RecordingModel(model),
            prompt_source=SyntheticPromptSource(vocab_size=self.VOCAB),
            clock=lambda: next(tick) * 1e-6,
            capture_logits_digest=capture,
        )
        drv.load()
        return drv

    @staticmethod
    def _one_step(drv, req_id: str, prompt_len: int):
        req = Request(req_id=req_id, prompt_len=prompt_len, gen_len=4)
        item = ForwardItem(req, n_new_tokens=prompt_len, need_token=True)
        return drv.forward([item])

    @staticmethod
    def _digest_of(out, req_id: str):
        """从 `StepOutcome.slots` 里取该槽位的 logits 摘要（**公开接口**）。"""
        for slot in out.slots:
            if slot.req_id == req_id:
                return slot.logits_digest
        raise AssertionError(f"返回里没有槽位 {req_id}")

    def test_disabled_by_default_produces_no_digests(self):
        """**默认关**：不给开关时既有的 `_GroupOutcome` 行为不变（空 dict）。"""
        drv = self._driver(capture=False)
        out = self._one_step(drv, "a", 9)
        self.assertIsNone(self._digest_of(out, "a"),
                          "默认必须不采集（既有行为不变）")

    def test_enabled_produces_a_digest_per_row(self):
        drv = self._driver(capture=True)
        out = self._one_step(drv, "a", 9)
        d = self._digest_of(out, "a")
        self.assertIsNotNone(d, "开了就必须有货，否则真机判据会空转")
        self.assertEqual(sorted(d), ["argmax", "max", "sha256", "sum"])
        self.assertIsInstance(d["argmax"], int)
        self.assertEqual(len(d["sha256"]), 64, "sha256 是逐位判据")
        self.assertEqual(
            drv.execution_optimizations["capture_logits_digest"]["n_logits_digests"], 1)

    def test_same_input_is_reproducible(self):
        """同输入两次跑 ⇒ 摘要逐位相同（否则"两臂相同"这条判据无从谈起）。"""
        d1 = self._digest_of(self._one_step(self._driver(capture=True), "a", 9), "a")
        d2 = self._digest_of(self._one_step(self._driver(capture=True), "a", 9), "a")
        self.assertEqual(d1, d2)

    def test_different_prompt_len_changes_the_digest(self):
        """★ 否定对照：换长度必须改摘要 —— 防"摘要恒为常数"这种假证据。"""
        d1 = self._digest_of(self._one_step(self._driver(capture=True), "a", 9), "a")
        d2 = self._digest_of(self._one_step(self._driver(capture=True), "a", 13), "a")
        self.assertNotEqual(d1["sha256"], d2["sha256"],
                            "不同输入若给出同一摘要，说明摘要在算无关的东西")


@unittest.skipIf(torch is None, f"环境缺少 torch/transformers：{TORCH_MISSING_REASON}")
class TestNumericalEquivalence(unittest.TestCase):
    VOCAB = 64
    PROMPTS = {"a": 9, "b": 5, "c": 13}

    def setUp(self):
        self.base = _make_model(vocab=self.VOCAB)

    def _driver(self) -> tuple[Any, Any]:
        """造一个带记录器的 CPU 驱动，并与基准模型共享权重。

        时钟用**递增计数器**而不是固定 0：``pack_ms`` 是"两次读钟之差"，
        固定时钟下它恒为 0，于是"这一步发生了补位搬运"这种断言无法成立
        （断言 ``pack_ms > 0`` 会永远失败，而失败原因看起来像"补位没做"）。
        递增 1µs 让差值有量纲，且与真实时间无关。
        """
        tick = itertools.count()
        recorder = _RecordingModel(self.base)
        drv = _CpuDriver(
            model_name="local-tiny",
            device="cpu",
            precision="fp32",
            max_seq_len=512,
            model=recorder,
            prompt_source=SyntheticPromptSource(vocab_size=self.VOCAB),
            clock=lambda: next(tick) * 1e-6,
        )
        drv.load()
        return drv, recorder

    @staticmethod
    def _req(name: str, prompt_len: int) -> Request:
        return Request(req_id=name, prompt_len=prompt_len, gen_len=4)

    def test_batched_logits_equal_solo_logits(self):
        """同一步里不同历史长度的序列合批后，logits 必须与单独跑一致。

        这是补位 / position_ids / KV 写回三处缺陷的总闸门。
        """
        reqs = {k: self._req(k, n) for k, n in self.PROMPTS.items()}

        # ---- 基准：A 单独 prefill，再单独 decode 两步 ----
        solo, solo_rec = self._driver()
        solo.forward(
            [ForwardItem(reqs["a"], n_new_tokens=self.PROMPTS["a"], need_token=True)]
        )
        solo_logits = [solo_rec.logits[-1][-1, -1, :].clone()]
        for _ in range(2):
            solo.forward([ForwardItem(reqs["a"], n_new_tokens=1, need_token=True)])
            solo_logits.append(solo_rec.logits[-1][-1, -1, :].clone())

        # ---- 对照：A/B/C 各自 prefill（长度不同），然后**一起** decode ----
        batched, batched_rec = self._driver()
        for key in ("a", "b", "c"):
            batched.forward(
                [
                    ForwardItem(
                        reqs[key],
                        n_new_tokens=self.PROMPTS[key],
                        need_token=True,
                    )
                ]
            )
        row_of = {"a": 0, "b": 1, "c": 2}
        for step in range(2):
            out = batched.forward(
                [
                    ForwardItem(reqs[key], n_new_tokens=1, need_token=True)
                    for key in ("a", "b", "c")
                ]
            )
            # 三步 decode 的历史长度不同（9/5/13），因此这一步必然发生补位。
            self.assertEqual(out.timing.n_forwards, 1)
            self.assertGreater(out.timing.pack_ms, 0.0)
            got = batched_rec.logits[-1][row_of["a"], -1, :]
            ref = solo_logits[step + 1]
            self.assertTrue(
                torch.allclose(got, ref, atol=1e-5, rtol=1e-4),
                f"第 {step + 1} 步 decode 的 logits 不一致："
                f"max_abs_diff={(got - ref).abs().max().item():.3e}",
            )

    def test_each_row_matches_its_own_solo_run(self):
        """批内**每一行**都要对，不能只看第一行。

        写回切片错位时，通常只有部分行拿到别人的历史；只验第一行会漏掉。
        """
        reqs = {k: self._req(k, n) for k, n in self.PROMPTS.items()}
        solo_logits: dict[str, Any] = {}
        for key, n in self.PROMPTS.items():
            drv, rec = self._driver()
            drv.forward([ForwardItem(reqs[key], n_new_tokens=n, need_token=True)])
            drv.forward([ForwardItem(reqs[key], n_new_tokens=1, need_token=True)])
            solo_logits[key] = rec.logits[-1][-1, -1, :].clone()

        batched, rec = self._driver()
        for key, n in self.PROMPTS.items():
            batched.forward([ForwardItem(reqs[key], n_new_tokens=n, need_token=True)])
        order = list(self.PROMPTS)
        batched.forward(
            [ForwardItem(reqs[k], n_new_tokens=1, need_token=True) for k in order]
        )
        for i, key in enumerate(order):
            with self.subTest(req=key):
                got = rec.logits[-1][i, -1, :]
                self.assertTrue(
                    torch.allclose(got, solo_logits[key], atol=1e-5, rtol=1e-4),
                    f"{key} 行与单独运行不一致："
                    f"max_abs_diff={(got - solo_logits[key]).abs().max().item():.3e}",
                )

    def test_chunked_prefill_equals_full_prefill(self):
        """分块预填充必须与一次算完等价。

        分块把同一段 prompt 拆成多次前向并依赖 KV 累积；若 KV 累积或
        position_ids 有偏差，结果会在这里分叉。
        """
        n = 12
        req = self._req("a", n)

        whole, whole_rec = self._driver()
        whole.forward([ForwardItem(req, n_new_tokens=n, need_token=True)])

        chunked, chunked_rec = self._driver()
        for k, is_last in ((5, False), (4, False), (3, True)):
            out = chunked.forward(
                [ForwardItem(req, n_new_tokens=k, need_token=is_last)]
            )
            if is_last:
                # 最后一个 chunk 把 prompt 算完了 → **必须产出首 token**。
                # 断言它为空是测试自己写错了：完整 prefill 结束后模型本就会给出
                # 下一个 token，正式路径（_Slot.take_next 进入 decode 分支）
                # 也依赖 last_token 非空。
                self.assertIsNotNone(out.slots[0].token_id)
                self.assertIsNotNone(out.slots[0].ready_s)
            else:
                # 中途前向：prompt 还没算完，不产出 token。
                self.assertIsNone(out.slots[0].token_id)
                # 但**前向本身的结束时刻是有值的**：它表达"这一步走到哪了"，
                # 与"有没有产出 token"是两件事，不该挤进同一个字段。
                self.assertIsNotNone(out.slots[0].ready_s)

        got = chunked_rec.logits[-1][-1, -1, :]
        ref = whole_rec.logits[-1][-1, -1, :]
        self.assertTrue(
            torch.allclose(got, ref, atol=1e-5, rtol=1e-4),
            f"分块与整段不一致：max_abs_diff={(got - ref).abs().max().item():.3e}",
        )
        # 而且只有最后一次前向产出 token。
        self.assertEqual(chunked._slots["a"].past_len, n)

    def test_past_len_advances_by_k_not_by_past_max(self):
        """写回时每个槽位的 past_len 只能按**自己**喂入的 k 前进。

        若按批内最长的 past 前进，短序列会把自己的历史长度记成别人的，
        下一步的 mask 与 position 全部错位——而且形状仍然合法。
        """
        reqs = {k: self._req(k, n) for k, n in self.PROMPTS.items()}
        drv, _ = self._driver()
        for key, n in self.PROMPTS.items():
            drv.forward([ForwardItem(reqs[key], n_new_tokens=n, need_token=True)])
        for key, n in self.PROMPTS.items():
            self.assertEqual(drv._slots[key].past_len, n)

        drv.forward(
            [ForwardItem(reqs[k], n_new_tokens=1, need_token=True) for k in reqs]
        )
        for key, n in self.PROMPTS.items():
            with self.subTest(req=key):
                self.assertEqual(drv._slots[key].past_len, n + 1)

    def test_fresh_slot_joins_mixed_past_batch(self):
        """新序列（无历史）与有历史的序列同批，不得索引空列表。

        这是补位函数里最容易漏的一支：新序列的 ``past`` 是空列表，
        直接按层索引会 IndexError——但那要等真机上出现这种批才炸。
        """
        reqs = {k: self._req(k, n) for k, n in self.PROMPTS.items()}
        drv, _ = self._driver()
        drv.forward(
            [ForwardItem(reqs["a"], n_new_tokens=self.PROMPTS["a"], need_token=True)]
        )
        # 此时 c 是新序列，与 a 同批 decode（k 都是 1，历史长度不同）
        out = drv.forward(
            [
                ForwardItem(reqs["a"], n_new_tokens=1, need_token=True),
                ForwardItem(reqs["c"], n_new_tokens=1, need_token=True),
            ]
        )
        self.assertEqual(out.timing.n_forwards, 1)
        self.assertIsNotNone(out.slots[0].token_id)

    def test_shape_groups_split_heterogeneous_k(self):
        """喂入长度不同的批应被拆成多次前向，并被如实计数。

        这是 HF 规则张量 KV 的真实代价：不拆就得做无效计算，拆了就多一次
        launch。两者都行，但必须**显式计数**——否则"每步开销"里混着
        launch 次数而读者不知道。
        """
        reqs = {k: self._req(k, n) for k, n in self.PROMPTS.items()}
        drv, _ = self._driver()
        out = drv.forward(
            [
                ForwardItem(reqs["a"], n_new_tokens=9, need_token=True),
                ForwardItem(reqs["b"], n_new_tokens=5, need_token=True),
                ForwardItem(reqs["c"], n_new_tokens=9, need_token=True),
            ]
        )
        self.assertEqual(out.timing.n_forwards, 2)  # 9 与 5 两组
        self.assertEqual(out.timing.n_tokens, 9 + 5 + 9)

    def test_empty_batch_is_rejected(self):
        drv, _ = self._driver()
        with self.assertRaises(ValueError):
            drv.forward([])

    def test_zero_token_item_is_rejected(self):
        drv, _ = self._driver()
        with self.assertRaises(ValueError):
            drv.forward([ForwardItem(self._req("a", 4), n_new_tokens=0, need_token=True)])

    def test_past_is_kv_pairs_not_a_packed_tensor(self):
        """``past`` 必须是 ``list[(key, value)]``——每层的 K 与 V 分开。

        这条是"首次前向就失败"那个缺陷的**结构性**守卫。把 ``(key, value)``
        当成一个张量切片时，报错发生在 transformers 内部深处
        （``'list' object has no attribute 'get_seq_length'`` 或形状不匹配），
        看不出真正的原因；而修好之后再退回旧写法，同样不会有直观提示。
        因此这里直接断言存储形态。
        """
        req = self._req("a", 7)
        drv, _ = self._driver()
        drv.forward([ForwardItem(req, n_new_tokens=7, need_token=True)])
        slot = drv._slots["a"]
        self.assertEqual(len(slot.past), drv._n_layers, "每层一项")
        for layer_idx, item in enumerate(slot.past):
            with self.subTest(layer=layer_idx):
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 2, "每层必须是 (key, value) 二元组")
                key, value = item
                self.assertEqual(key.shape, value.shape)
                self.assertEqual(
                    key.shape[2], slot.past_len, "序列维长度必须等于真实历史长度"
                )

    def test_padded_batch_mask_matches_padded_kv_width(self):
        """补位后的批里，mask 宽度必须等于补位后的 KV 宽度。

        这是本轮最隐蔽的一处：``input_ids`` 只有 ``k`` 个 token，KV 却补到了
        ``past_max + k``。mask 若写成与 ``input_ids`` 同宽（``[1]*k``），
        HF 会让全零的补位槽也参与注意力——短行的结果静默算错，形状全程合法。
        这里比对"设备实际收到的 mask 宽度"与"设备实际收到的 KV 宽度"，
        让这条不变量显式成立，而不是只靠一条端到端的数值断言间接覆盖。
        """
        reqs = {k: self._req(k, n) for k, n in self.PROMPTS.items()}
        drv, rec = self._driver()
        for key, n in self.PROMPTS.items():
            drv.forward([ForwardItem(reqs[key], n_new_tokens=n, need_token=True)])

        captured: dict[str, Any] = {}
        original = drv.model.forward

        def spy(*args: Any, **kwargs: Any):
            captured.update(kwargs)
            return original(*args, **kwargs)

        drv.model.forward = spy
        drv.forward(
            [ForwardItem(reqs[k], n_new_tokens=1, need_token=True) for k in reqs]
        )

        mask = captured["attention_mask"]
        past = captured["past_key_values"]
        # ``get_seq_length()`` 是缓存里**当前**的 key 长度（补位后的历史 +
        # 本次写入的 token）。mask 的 key 维必须与它相等——这正是"补位槽
        # 必须出现在 mask 里"这条不变量的可检验形式。
        kv_width = past.get_seq_length()
        self.assertEqual(
            mask.shape[1],
            kv_width,
            "mask 的 key 维必须覆盖补位后的整段 KV；写窄了会让补位槽参与注意力",
        )
        # 逐行检查：左补位区为 0，真实历史 + 本次输入为 1。
        for i, key in enumerate(self.PROMPTS):
            real = self.PROMPTS[key]
            with self.subTest(req=key):
                row = mask[i]
                self.assertEqual(int(row.sum().item()), real + 1, "可见 token 数")
                pad = kv_width - real - 1
                self.assertTrue(bool((row[:pad] == 0).all()), "补位区必须屏蔽")
                self.assertTrue(bool((row[pad:] == 1).all()), "真实区必须可见")

    def test_each_row_position_ids_built_from_its_own_past_len(self):
        """position_ids 必须逐行用自己的历史长度，而不是批内最长的那个。

        用 ``past_max + 列号`` 会让短序列的位置整体前移，于是它以为自己的历史
        更长——形状合法、数值错。这里直接检查设备收到的 position_ids 数值。
        """
        reqs = {k: self._req(k, n) for k, n in self.PROMPTS.items()}
        drv, _ = self._driver()
        for key, n in self.PROMPTS.items():
            drv.forward([ForwardItem(reqs[key], n_new_tokens=n, need_token=True)])

        captured: dict[str, Any] = {}
        original = drv.model.forward

        def spy(*args: Any, **kwargs: Any):
            captured.update(kwargs)
            return original(*args, **kwargs)

        drv.model.forward = spy
        drv.forward(
            [ForwardItem(reqs[k], n_new_tokens=1, need_token=True) for k in reqs]
        )
        pos = captured["position_ids"]
        for i, key in enumerate(self.PROMPTS):
            with self.subTest(req=key):
                # 每个序列只喂 1 个新 token，它的位置 = 自己的历史长度。
                self.assertEqual(int(pos[i, -1].item()), self.PROMPTS[key])


@unittest.skipIf(torch is None, f"环境缺少 torch/transformers：{TORCH_MISSING_REASON}")
class TestPromptSourceContract(unittest.TestCase):
    """合成 prompt 的确定性契约（共享前缀是缓存实验的基数）。"""

    def test_same_request_is_reproducible(self):
        src = SyntheticPromptSource(vocab_size=128)
        req = Request(req_id="r7", prompt_len=20, gen_len=2)
        self.assertEqual(src.prompt_ids(req), src.prompt_ids(req))

    def test_shared_prefix_is_identical_within_group(self):
        src = SyntheticPromptSource(vocab_size=128)
        a = Request(
            req_id="r1", prompt_len=32, gen_len=2,
            shared_prefix_len=16, prefix_group="g000",
        )
        b = Request(
            req_id="r2", prompt_len=32, gen_len=2,
            shared_prefix_len=16, prefix_group="g000",
        )
        c = Request(
            req_id="r3", prompt_len=32, gen_len=2,
            shared_prefix_len=16, prefix_group="g001",
        )
        pa, pb, pc = src.prompt_ids(a), src.prompt_ids(b), src.prompt_ids(c)
        self.assertEqual(pa[:16], pb[:16], "同组共享段必须逐位相同")
        self.assertNotEqual(pa[:16], pc[:16], "不同组不应共享")
        self.assertNotEqual(pa[16:], pb[16:], "组内非共享段不应相同")

    def test_ids_never_hit_special_tokens(self):
        src = SyntheticPromptSource(vocab_size=32, excluded_ids=[1, 2, 3, 30, 31])
        ids = src.prompt_ids(Request(req_id="r0", prompt_len=200, gen_len=2))
        self.assertEqual(len(ids), 200)
        self.assertFalse(set(ids) & {1, 2, 3, 30, 31})

    def test_ids_stay_in_vocab_range(self):
        src = SyntheticPromptSource(vocab_size=16)
        ids = src.prompt_ids(Request(req_id="r0", prompt_len=100, gen_len=2))
        self.assertTrue(all(0 < i < 16 for i in ids))

    def test_excluding_whole_vocab_is_rejected(self):
        with self.assertRaises(ValueError):
            SyntheticPromptSource(vocab_size=4, excluded_ids=[1, 2, 3])


if __name__ == "__main__":
    unittest.main()
