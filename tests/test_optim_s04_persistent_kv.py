"""S04 / P2b-c：常驻 KV 缓冲快路径（纯 decode + 历史等长）。

要保的性质：

1. **默认关闭**：不装缓冲、行为与基线**逐字节相同**。
2. **数值等价**：开启后逐 token 与逐层 KV 必须与基线一致 —— 这是本会话的
   核心正确性断言（搬运转述再快，算错就是零分）。
3. ★ **回退后不失效**：走若干步快路径 → 插入一个混长/新序列步（触发回退）→
   后续结果必须**仍然正确**。这条最容易漏：快路径若只写缓冲而不维护
   `s.past`，回退时 `_pack_past` 会拿到**陈旧一截**的 KV ⇒ 数值悄悄错、
   形状全合法。必须有正例（走快路径）与反例（回退）**接续**的用例。
4. **判据与其原因可数**：快路径步数与回退步数**同时**记录，且回退原因分桶；
   配了但全程回退（容量不足）必须能被看出来。
5. **内存代价如实**：预分配字节数被记出来，且 `retire` **扣得回去**（无泄漏）。

⚠️ 性能（真机 pack_ms 下降多少）**不属于本地可验范围** ⇒ `REMOTE_PENDING`。
本文件只回答"快路径是否正确、判据是否成立、记账是否如实"。
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


def _run(drv, batch_spec, steps=6):
    """按 `(name, prompt_len)` 列表推进 `steps` 步 decode，收集每步每请求的 token。

    `batch_spec` 是 `list[(name, prompt_len, gen_len)]`；第一步一次性吃掉 prompt
    （n_new_tokens=prompt_len），之后每步 decode 一个 token。
    """
    tokens: dict[str, list] = {}
    items = [
        ForwardItem(T.req(n, pl, g), n_new_tokens=pl, need_token=True)
        for n, pl, g in batch_spec
    ]
    out = drv.forward(items)
    for sl in out.slots:
        tokens.setdefault(sl.req_id, []).append(sl.token_id)
    for _ in range(steps):
        items = [
            ForwardItem(T.req(n, pl, g), n_new_tokens=1, need_token=True)
            for n, pl, g in batch_spec
        ]
        out = drv.forward(items)
        for sl in out.slots:
            tokens.setdefault(sl.req_id, []).append(sl.token_id)
    return tokens


def _kv_snapshot(drv, names):
    """取这些槽位**逐层** KV 的克隆（用于跨路径逐层比对）。"""
    snap = {}
    for n in names:
        s = drv._slots[n]
        snap[n] = [
            (k.detach().clone(), v.detach().clone()) for k, v in s.past
        ]
    return snap


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestDisabledByDefault(unittest.TestCase):
    def test_no_buffer_allocated_and_counts_zero(self):
        model = T.make_model()
        drv = T.make_driver(model)
        drv.load()
        eo = drv.execution_optimizations["persistent_kv_slots"]
        self.assertFalse(eo["requested"])
        self.assertFalse(eo["effective"])
        self.assertEqual(eo["n_fastpath_steps"], 0)
        self.assertEqual(eo["n_fallback_steps"], 0)
        self.assertEqual(eo["bytes_allocated"], 0)
        self.assertEqual(drv._kv_buf, {})

    def test_stays_disabled_over_a_run(self):
        model = T.make_model()
        drv = T.make_driver(model)
        drv.load()
        _run(drv, [("a", 8, 4), ("b", 8, 4)], steps=5)
        eo = drv.execution_optimizations["persistent_kv_slots"]
        self.assertEqual(eo["n_fastpath_steps"], 0)
        self.assertEqual(drv._kv_buf, {})


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestNumericalEquivalence(unittest.TestCase):
    def test_tokens_and_kv_match_baseline_when_fastpath_active(self):
        batch = [("a", 9, 6), ("b", 9, 6), ("c", 9, 6)]

        m0 = T.make_model(seed=7)
        d0 = T.make_driver(m0)
        d0.load()
        base_tokens = _run(d0, batch, steps=6)

        m1 = T.make_model(seed=7)
        d1 = T.make_driver(m1, persistent_kv_slots=True, persistent_kv_capacity=64)
        d1.load()
        opt_tokens = _run(d1, batch, steps=6)

        # ★ 快路径必须真的走过（否则这条"等价"什么都没验到）。
        eo = d1.execution_optimizations["persistent_kv_slots"]
        self.assertGreater(eo["n_fastpath_steps"], 0, "快路径一步都没走，等价性无意义")
        self.assertEqual(base_tokens, opt_tokens, "快路径的 token 与基线不一致")

        # 逐层 KV 也必须一致（token 相同但 KV 错位是可能的：错位后下一步才现形）。
        for n in ("a", "b", "c"):
            kb = d0._slots[n].past
            ko = d1._slots[n].past
            self.assertEqual(len(kb), len(ko))
            for layer in range(len(kb)):
                self.assertTrue(
                    torch.equal(kb[layer][0], ko[layer][0]),
                    f"{n} 第 {layer} 层 K 不一致",
                )
                self.assertTrue(
                    torch.equal(kb[layer][1], ko[layer][1]),
                    f"{n} 第 {layer} 层 V 不一致",
                )
            self.assertEqual(d0._slots[n].past_len, d1._slots[n].past_len)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestFallbackKeepsResultsCorrect(unittest.TestCase):
    """★ 本组是 S04 最关键的正确性守卫：快路径与通用路径**接续**跑。"""

    def test_fastpath_then_unequal_len_fallback_then_continue(self):
        """走几步快路径 → 补入一个**历史不等长**的新请求 → 再继续。

        新请求的 prompt 更长，其 `past_len` 与既有槽位不同 ⇒ 批变成混长
        ⇒ 判据返回 `unequal_kv_len` ⇒ 本步走通用路径。此时既有槽位的
        `s.past` 必须已经是**最新的**（快路径写过的那一份），否则通用路径
        拿到的历史短一截，后续 token 悄悄错开。
        """
        m0 = T.make_model(seed=11)
        d0 = T.make_driver(m0)
        d0.load()
        # 基线：两次预热后补入长请求，再继续。
        base = _fallback_scenario(d0)

        m1 = T.make_model(seed=11)
        d1 = T.make_driver(m1, persistent_kv_slots=True, persistent_kv_capacity=128)
        d1.load()
        opt = _fallback_scenario(d1)

        eo = d1.execution_optimizations["persistent_kv_slots"]
        self.assertGreater(eo["n_fastpath_steps"], 0, "快路径没走，接续用例无意义")
        self.assertGreater(eo["n_fallback_steps"], 0, "回退没发生，接续用例无意义")
        self.assertIn("unequal_kv_len", eo["fallback_reasons"])
        self.assertEqual(base, opt, "快路径接续通用路径后的结果与基线不一致")

    def test_new_sequence_forces_fallback_and_stays_correct(self):
        """全新序列在**同一 group 内**必须回退，且不污染既有槽位的结果。

        ⚠️ 场景设计要点（本用例最初写错过）：`new_sequence` 只在
        "新序列与既有序列落在**同一形状组**里"时才会被观察到，而新请求的第一步
        通常是"整段 prompt"（`k = prompt_len > 1`），那种时刻**先**撞上
        `not_pure_decode`。因此这里用 `prompt_len = 1` 的请求 —— 它既是新序列、
        又满足 `k == 1`，是唯一能把 `new_sequence` 单独观察到的常规形状。
        （判据顺序已把 `new_sequence` 提到前面，见 `_pkv_eligible` 的注释。）
        """
        batch = [("a", 9, 6), ("b", 9, 6)]

        m0 = T.make_model(seed=13)
        d0 = T.make_driver(m0)
        d0.load()
        base = _newcomer_scenario(d0)

        m1 = T.make_model(seed=13)
        d1 = T.make_driver(m1, persistent_kv_slots=True, persistent_kv_capacity=128)
        d1.load()
        opt = _newcomer_scenario(d1)

        eo = d1.execution_optimizations["persistent_kv_slots"]
        self.assertIn("new_sequence", eo["fallback_reasons"])
        self.assertEqual(base, opt)

    def test_not_pure_decode_forces_fallback(self):
        """**已有历史**的序列本步喂多个 token（chunked prefill 续传）⇒ `not_pure_decode`。

        ⚠️ 不能拿"新序列首步喂整段 prompt"来测这一条：那种形状会**先**命中
        `new_sequence`（判据顺序已把结构性阻塞提前），见 `_pkv_eligible`。
        要让 `not_pure_decode` 单独出现，必须让**已有 KV 的序列**喂 `k > 1`。
        """
        model = T.make_model(seed=17)
        drv = T.make_driver(model, persistent_kv_slots=True, persistent_kv_capacity=128)
        drv.load()
        # 第一步把 prompt 喂完（这步归因为 new_sequence）。
        drv.forward(
            [
                ForwardItem(T.req("a", 8, 8), n_new_tokens=8, need_token=True),
                ForwardItem(T.req("b", 8, 8), n_new_tokens=8, need_token=True),
            ]
        )
        # 第二步：prompt 更长（16），只喂前 8 个 ⇒ 仍是 prefill 续传（k=8 > 1）。
        # ⚠️ 不能用 decode 阶段喂 k=2 来伪造这一步：`take_next` 会拦下
        # "decode 阶段一次喂 >1 个 token"（那本来就不是合法形状）。
        drv2 = T.make_driver(T.make_model(seed=17), persistent_kv_slots=True,
                             persistent_kv_capacity=128)
        drv2.load()
        drv2.forward(
            [
                ForwardItem(T.req("a", 16, 8), n_new_tokens=8, need_token=False),
                ForwardItem(T.req("b", 16, 8), n_new_tokens=8, need_token=False),
            ]
        )
        drv2.forward(
            [
                ForwardItem(T.req("a", 16, 8), n_new_tokens=8, need_token=True),
                ForwardItem(T.req("b", 16, 8), n_new_tokens=8, need_token=True),
            ]
        )
        eo = drv2.execution_optimizations["persistent_kv_slots"]
        self.assertEqual(eo["n_fastpath_steps"], 0)
        self.assertIn("not_pure_decode", eo["fallback_reasons"])

    def test_fastpath_then_capacity_exceeded_falls_back_correctly(self):
        """容量刚好只够几步 ⇒ 越界后回退，且结果仍正确。"""
        batch = [("a", 9, 8), ("b", 9, 8)]

        m0 = T.make_model(seed=19)
        d0 = T.make_driver(m0)
        d0.load()
        base = _run(d0, batch, steps=8)

        m1 = T.make_model(seed=19)
        # 容量 = prompt(9) + 2 ⇒ 只能容纳 2 步 decode 后的长度。
        d1 = T.make_driver(m1, persistent_kv_slots=True, persistent_kv_capacity=11)
        d1.load()
        opt = _run(d1, batch, steps=8)

        eo = d1.execution_optimizations["persistent_kv_slots"]
        self.assertGreater(eo["n_fastpath_steps"], 0)
        self.assertIn("capacity_exceeded", eo["fallback_reasons"])
        self.assertEqual(base, opt)


def _fallback_scenario(drv):
    """两步同类 decode → 补入更长历史的新请求（混长）→ 继续两步。"""
    out = {}
    for n, pl in [("a", 9), ("b", 9)]:
        drv.forward([ForwardItem(T.req(n, pl, 8), n_new_tokens=pl, need_token=True)])
        out[n] = []
    # 各 decode 两步（历史等长 ⇒ 若开了快路径则走快路径）。
    for _ in range(2):
        o = drv.forward(
            [
                ForwardItem(T.req("a", 9, 8), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("b", 9, 8), n_new_tokens=1, need_token=True),
            ]
        )
        for sl in o.slots:
            out[sl.req_id].append(sl.token_id)
    # 补入历史更长的 c：先把它的 prompt 算出一部分，使 past_len 与 a/b 不同。
    drv.forward([ForwardItem(T.req("c", 17, 8), n_new_tokens=17, need_token=True)])
    out["c"] = []
    # 此刻 a/b 的 past_len 与 c 不同 ⇒ 下一次同批必为混长 ⇒ 回退。
    for _ in range(2):
        o = drv.forward(
            [
                ForwardItem(T.req("a", 9, 8), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("b", 9, 8), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("c", 17, 8), n_new_tokens=1, need_token=True),
            ]
        )
        for sl in o.slots:
            out[sl.req_id].append(sl.token_id)
    return out


def _newcomer_scenario(drv):
    """先让两个请求各自推进，再让**全新请求**与它们同批（触发 `new_sequence`）。

    ⚠️ 新请求用 `prompt_len = 1`：这样它落在同一形状组内（`k == 1`），
    且是"新序列"，才能把 `new_sequence` 这个原因单独观察到。
    """
    out = {}
    for n, pl in [("a", 9), ("b", 9)]:
        drv.forward([ForwardItem(T.req(n, pl, 6), n_new_tokens=pl, need_token=True)])
        out[n] = []
    for _ in range(2):
        o = drv.forward(
            [
                ForwardItem(T.req("a", 9, 6), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("b", 9, 6), n_new_tokens=1, need_token=True),
            ]
        )
        for sl in o.slots:
            out[sl.req_id].append(sl.token_id)
    # c 是全新序列（past 为空），prompt_len=1 ⇒ k==1 ⇒ 与 a/b 同组。
    o = drv.forward(
        [
            ForwardItem(T.req("a", 9, 6), n_new_tokens=1, need_token=True),
            ForwardItem(T.req("b", 9, 6), n_new_tokens=1, need_token=True),
            ForwardItem(T.req("c", 1, 6), n_new_tokens=1, need_token=True),
        ]
    )
    for sl in o.slots:
        out.setdefault(sl.req_id, []).append(sl.token_id)
    for _ in range(2):
        o = drv.forward(
            [
                ForwardItem(T.req("a", 9, 6), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("b", 9, 6), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("c", 1, 6), n_new_tokens=1, need_token=True),
            ]
        )
        for sl in o.slots:
            out[sl.req_id].append(sl.token_id)
    return out


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestMemoryAccounting(unittest.TestCase):
    def test_bytes_recorded_and_reclaimed_on_retire(self):
        model = T.make_model()
        drv = T.make_driver(model, persistent_kv_slots=True, persistent_kv_capacity=32)
        drv.load()
        _run(drv, [("a", 8, 4), ("b", 8, 4)], steps=4)
        eo = drv.execution_optimizations["persistent_kv_slots"]
        self.assertGreater(eo["bytes_allocated"], 0)
        # 露出的细节：容量、头数、head_dim 都应可从产物读出。
        self.assertEqual(eo["capacity"], 32)
        self.assertIsNotNone(eo["n_kv_heads"])
        self.assertIsNotNone(eo["head_dim"])
        self.assertEqual(len(drv._kv_buf), 2)

        drv.retire(["a"])
        self.assertEqual(len(drv._kv_buf), 1)
        eo2 = drv.execution_optimizations["persistent_kv_slots"]
        self.assertEqual(
            eo2["bytes_allocated"], eo["bytes_allocated"] - drv._pkv_bytes_per_slot
        )

        drv.retire(["b"])
        eo3 = drv.execution_optimizations["persistent_kv_slots"]
        self.assertEqual(eo3["bytes_allocated"], 0)
        self.assertEqual(drv._kv_buf, {})

    def test_zero_capacity_is_rejected(self):
        model = T.make_model()
        drv = T.make_driver(model, persistent_kv_slots=True, persistent_kv_capacity=0)
        with self.assertRaises(ValueError):
            drv.load()

    def test_capacity_counted_in_load_info(self):
        """容量必须在装载期定下（否则第一次前向会额外分配一大块）。"""
        model = T.make_model()
        drv = T.make_driver(model, persistent_kv_slots=True, persistent_kv_capacity=48)
        drv.load()
        self.assertEqual(drv._pkv_capacity, 48)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestEligibilityReasonNegativeControl(unittest.TestCase):
    """★ 否定测试：判据必须**能**返回"不可走"，并给出正确原因。

    只测"能走"的判据等于没测——一个恒返回 True 的判据也能通过那类用例。
    """

    def test_reason_is_specific_for_each_blocker(self):
        model = T.make_model()
        drv = T.make_driver(model, persistent_kv_slots=True, persistent_kv_capacity=128)
        drv.load()

        # 纯 decode + 等长 ⇒ 可走（对照组）
        drv.forward(
            [
                ForwardItem(T.req("a", 8, 6), n_new_tokens=8, need_token=True),
                ForwardItem(T.req("b", 8, 6), n_new_tokens=8, need_token=True),
            ]
        )
        pairs = [
            (drv._slot(T.req("a", 8, 6)), ForwardItem(T.req("a", 8, 6), 1, True)),
            (drv._slot(T.req("b", 8, 6)), ForwardItem(T.req("b", 8, 6), 1, True)),
        ]
        ok, why = drv._pkv_eligible(pairs, 1)
        self.assertTrue(ok, f"对照组应可走，实际被拒：{why}")
        self.assertEqual(why, "")

        # 非纯 decode
        ok2, why2 = drv._pkv_eligible(pairs, 4)
        self.assertFalse(ok2)
        self.assertEqual(why2, "not_pure_decode")

        # 容量不足：把容量设得很小后重建
        model2 = T.make_model()
        drv2 = T.make_driver(model2, persistent_kv_slots=True, persistent_kv_capacity=8)
        drv2.load()
        drv2.forward(
            [
                ForwardItem(T.req("a", 8, 6), n_new_tokens=8, need_token=True),
                ForwardItem(T.req("b", 8, 6), n_new_tokens=8, need_token=True),
            ]
        )
        # 上面这一步已把 past_len 推过容量；此刻判据应报 capacity_exceeded。
        p2 = [
            (drv2._slot(T.req("a", 8, 6)), ForwardItem(T.req("a", 8, 6), 1, True)),
            (drv2._slot(T.req("b", 8, 6)), ForwardItem(T.req("b", 8, 6), 1, True)),
        ]
        ok3, why3 = drv2._pkv_eligible(p2, 1)
        self.assertFalse(ok3)
        self.assertEqual(why3, "capacity_exceeded")


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestManifestFacts(unittest.TestCase):
    def test_report_shape_is_stable(self):
        model = T.make_model()
        drv = T.make_driver(model, persistent_kv_slots=True, persistent_kv_capacity=32)
        drv.load()
        _run(drv, [("a", 8, 4), ("b", 8, 4)], steps=3)
        eo = drv.describe()["execution_optimizations"]["persistent_kv_slots"]
        for key in (
            "requested",
            "effective",
            "n_fastpath_steps",
            "n_fallback_steps",
            "fallback_reasons",
            "capacity",
            "bytes_allocated",
            # ★ 新增：`bytes_allocated` 是**当前驻留**，run 收尾（全部
            #   `retire`）后恒为 0 ⇒ 单看它无法区分「从未分配」与
            #   「分配后全部释放」。判"到底分配过没有"必须看这两个。
            "bytes_peak",
            "n_allocations",
        ):
            self.assertIn(key, eo)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestAllocationVisibleAfterRetire(unittest.TestCase):
    """★ 终态 `bytes_allocated == 0` **不能**被读成"没分配过"。

    实测（EXECOPT4 真机）：跑了 193 步快路径，报告里 `bytes_allocated=0`
    —— 因为所有请求都已 `retire`、缓冲已释放。峰值与次数才是证据。
    """

    def test_peak_and_count_survive_retire(self):
        model = T.make_model()
        drv = T.make_driver(model, persistent_kv_slots=True, persistent_kv_capacity=64)
        drv.load()
        _run(drv, [("a", 8, 4), ("b", 8, 4)], steps=3)
        names = ["a", "b"]
        eo = drv.execution_optimizations["persistent_kv_slots"]
        self.assertGreater(eo["n_allocations"], 0, "内存记账缺失 ⇒ 无法证明分配发生过")

        drv.retire(names)
        eo2 = drv.execution_optimizations["persistent_kv_slots"]
        self.assertEqual(eo2["bytes_allocated"], 0, "全部释放后当前驻留应为 0")
        # ★ 关键断言：**峰值与次数不因释放而归零**。
        self.assertGreater(eo2["bytes_peak"], 0)
        self.assertEqual(eo2["n_allocations"], eo["n_allocations"])

    def test_disabled_reports_zero_peak_and_zero_allocations(self):
        """反向对照：没开机制时峰值/次数都必须是 0（否则上面的用例恒真）。"""
        model = T.make_model()
        drv = T.make_driver(model)
        drv.load()
        _run(drv, [("a", 8, 4)], steps=3)
        eo = drv.execution_optimizations["persistent_kv_slots"]
        self.assertEqual(eo["bytes_peak"], 0)
        self.assertEqual(eo["n_allocations"], 0)


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestMeasuredWindowScope(unittest.TestCase):
    """★ 预热与正式段共用累加器 ⇒ `*_total` 含预热，跨臂不可直接比。

    实测（EXECOPT4 真机）：两臂的 `warmup.iterations` 是 2 vs 1，
    `n_forwards` 差 117（= 恰好一轮预热的 1 prefill + 116 decode）
    ⇒ 直接比 `pack_ms_total` 会把"预热跑了几轮"当成机制效应
    （看到 pack +31%，而消除预热口径后只有 +0.2%）。
    """

    def test_baseline_is_none_before_warmup_and_reports_totals(self):
        model = T.make_model()
        drv = T.make_driver(model)
        drv.load()
        st = drv.stats()
        self.assertIsNone(drv._warmup_baseline)
        mw = st["measured_window"]
        self.assertEqual(mw["basis"], "no_warmup")
        self.assertEqual(mw["n_forwards_measured"], st["n_forwards"])

    def test_warmup_is_excluded_from_measured_keys(self):
        model = T.make_model()
        drv = T.make_driver(model)
        drv.load()
        drv.warmup(0.0, batch=2, prompt_len=6, gen_len=3)
        n_after_warmup = drv.stats()["n_forwards"]
        self.assertGreater(n_after_warmup, 0, "预热应当真的跑过前向")
        _run(drv, [("a", 8, 4), ("b", 8, 4)], steps=3)
        st = drv.stats()
        mw = st["measured_window"]
        self.assertEqual(mw["basis"], "total_minus_warmup")
        # 正式段有前向 ⇒ 测量段计数必须为正……
        self.assertGreater(mw["n_forwards_measured"], 0)
        # ……且**严格小于**总量（预热那部分被剔掉了）。
        self.assertLess(mw["n_forwards_measured"], st["n_forwards"])
        self.assertLess(mw["pack_ms_measured"], st["pack_ms_total"])


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestReattachAfterFallback(unittest.TestCase):
    """★★ 缺了这个用例，S04 就在真机上**带着数值错误**跑完了整批。

    既有 `_fallback_scenario` 补入混长请求后**从不移除它** ⇒ 批一直是混长
    ⇒ 快路径**再也没有被重新进入** ⇒ "快→回退"之后的**重入**路径一次都没被覆盖。
    而重入正是过期缓冲被读到的时刻：

      快路径（缓冲已同步到 L0）→ 回退（只推进 `s.past`，**不碰缓冲**）
      → 新请求退出 ⇒ 历史重新等长 → 重入快路径 ⇒ 读缓冲
      ⇒ 缓冲里 [L0, L1) 段是**旧的**（对被释放请求而言还是错的别人的数据）
      ⇒ 形状合法、不报错、token 悄悄错开。

    真机证据：EXECOPT4 的 `s04_pkv` 跑了 193 个快路径组，其中约 116 来自预热，
    正式窗口里主要就是**单请求尾段**——正是"成员退出后重入"的形态。
    """

    @staticmethod
    def _scenario(drv):
        """等长快路径 → 混长回退 → 新请求退出 → **重入快路径** → 继续。

        返回 `(逐请求 token 序列, 逐层 KV 快照)`。
        """
        out: dict[str, list] = {}
        for n in ("a", "b"):
            drv.forward([ForwardItem(T.req(n, 9, 12), n_new_tokens=9, need_token=True)])
            out[n] = []
        # ① 快路径：a/b 历史等长。
        for _ in range(2):
            o = drv.forward([
                ForwardItem(T.req("a", 9, 12), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("b", 9, 12), n_new_tokens=1, need_token=True),
            ])
            for sl in o.slots:
                out[sl.req_id].append(sl.token_id)
        # ② 混长回退：c 的历史更长 ⇒ `unequal_kv_len`。
        drv.forward([ForwardItem(T.req("c", 17, 12), n_new_tokens=17, need_token=True)])
        out["c"] = []
        for _ in range(2):
            o = drv.forward([
                ForwardItem(T.req("a", 9, 12), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("b", 9, 12), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("c", 17, 12), n_new_tokens=1, need_token=True),
            ])
            for sl in o.slots:
                out[sl.req_id].append(sl.token_id)
        # ③ ★ c 退出 ⇒ a/b 重新等长 ⇒ **重新进入快路径**（过期缓冲在此被读到）。
        drv.retire(["c"])
        out.pop("c")
        for _ in range(3):
            o = drv.forward([
                ForwardItem(T.req("a", 9, 12), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("b", 9, 12), n_new_tokens=1, need_token=True),
            ])
            for sl in o.slots:
                out[sl.req_id].append(sl.token_id)
        return out, _kv_snapshot(drv, ["a", "b"])

    def test_reattach_matches_baseline(self):
        m0 = T.make_model(seed=23)
        d0 = T.make_driver(m0)
        d0.load()
        base_tokens, base_kv = self._scenario(d0)

        m1 = T.make_model(seed=23)
        d1 = T.make_driver(m1, persistent_kv_slots=True, persistent_kv_capacity=128)
        d1.load()
        opt_tokens, opt_kv = self._scenario(d1)

        # —— 先证"这条路径真的被走到了"（否则下面的相等什么也没验到）——
        eo = d1.execution_optimizations["persistent_kv_slots"]
        self.assertGreater(eo["n_fastpath_steps"], 0, "快路径没走")
        self.assertGreater(eo["n_fallback_steps"], 0, "回退没发生")
        self.assertIn("unequal_kv_len", eo["fallback_reasons"])
        self.assertGreater(eo["n_resyncs"], 0, "重入重同步没发生 ⇒ 本用例没覆盖重入")

        # —— token 级对照 ——
        self.assertEqual(base_tokens, opt_tokens, "重入快路径后 token 与基线不一致")

        # —— 逐层 KV 对照（比 token 更早暴露，token 可能前几步恰好还相同）——
        self.assertEqual(sorted(base_kv), sorted(opt_kv))
        for name in base_kv:
            self.assertEqual(len(base_kv[name]), len(opt_kv[name]), f"{name} 层数不同")
            for li, ((bk, bv), (ok, ov)) in enumerate(zip(base_kv[name], opt_kv[name])):
                self.assertEqual(bk.shape, ok.shape, f"{name} 第 {li} 层 K 形状不同")
                self.assertTrue(
                    bool((bk == ok).all()),
                    f"{name} 第 {li} 层 K 在重入快路径后与基线不同"
                    f"（最大差 {float((bk - ok).abs().max()):.3e}）",
                )
                self.assertTrue(bool((bv == ov).all()), f"{name} 第 {li} 层 V 不同")

    def test_resync_counter_is_zero_in_pure_fastpath(self):
        """**反向对照**：全程等长（不回退）时不得出现重同步。

        没有这条，`n_resyncs > 0` 可能只是恒真（例如被算成"分配次数"）。
        """
        m = T.make_model(seed=29)
        drv = T.make_driver(m, persistent_kv_slots=True, persistent_kv_capacity=128)
        drv.load()
        for n in ("a", "b"):
            drv.forward([ForwardItem(T.req(n, 9, 8), n_new_tokens=9, need_token=True)])
        for _ in range(3):
            drv.forward([
                ForwardItem(T.req("a", 9, 8), n_new_tokens=1, need_token=True),
                ForwardItem(T.req("b", 9, 8), n_new_tokens=1, need_token=True),
            ])
        eo = drv.execution_optimizations["persistent_kv_slots"]
        self.assertGreater(eo["n_fastpath_steps"], 0, "快路径没走，反向对照无意义")
        # 预填充那两步各自是新序列 ⇒ 会有 `new_sequence` 回退；关键是**没有**
        # 混长回退（那才是让缓冲变脏的原因）。
        self.assertNotIn("unequal_kv_len", eo["fallback_reasons"])
        self.assertEqual(eo["n_resyncs"], 0, "没回退就不该有重同步")


@unittest.skipIf(T.torch is None, f"环境缺少 torch/transformers：{T.TORCH_MISSING_REASON}")
class TestTokensOutCountsFastpath(unittest.TestCase):
    """★ `tokens_out` 必须在**快路径**上也计。

    原实现只在通用分支 `self._tokens_out += 1` ⇒ 快路径产出的 token 全被漏记。
    实测（CPU 微型模型）实际 17 个 token、只记 9 个。下游"每 token 偏早量"
    之类派生量会整体偏大，且**不报错**。
    """

    def test_tokens_out_equals_produced_tokens(self):
        batch = [("a", 8, 6), ("b", 8, 6), ("c", 8, 6)]
        steps = 5

        m0 = T.make_model(seed=31)
        d0 = T.make_driver(m0)
        d0.load()
        base = _run(d0, batch, steps=steps)

        m1 = T.make_model(seed=31)
        d1 = T.make_driver(m1, persistent_kv_slots=True, persistent_kv_capacity=128)
        d1.load()
        opt = _run(d1, batch, steps=steps)

        eo = d1.execution_optimizations["persistent_kv_slots"]
        self.assertGreater(eo["n_fastpath_steps"], 0, "快路径没走，本用例没覆盖到")

        # 每请求：1 次 prefill + steps 次 decode，各产 1 个 token。
        expected = len(batch) * (1 + steps)
        self.assertEqual(sum(len(v) for v in opt.values()), expected)
        self.assertEqual(d1.stats()["tokens_out"], expected,
                         "快路径产出的 token 未被计入 tokens_out")
        # 基线同样口径（防止"改了个恒等式"）。
        self.assertEqual(d0.stats()["tokens_out"], expected)
        self.assertEqual(base, opt)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
