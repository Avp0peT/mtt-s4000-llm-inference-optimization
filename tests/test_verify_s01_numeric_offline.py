"""数值验收脚本的**离线集成测试**。

★ 两条路径都要测，它们证明的是**不同的事**：

1. **假驱动**（不需要 torch）——证明**判据会红**。
   判据的唯一价值就是"坏输入时它会红"；真机跑一次只能证明"好输入时它绿"。

2. **真实 CPU 微型驱动**（需要 torch/transformers，本机有）——证明
   **判据挂在生产代码路径上**。`verify_s01_numeric_acceptance.py` 里那 6 项断言
   走的是**真 `TransformerDriver.forward`**（含 `logits_last_only` /
   `sample_needed_only` 两个分支），不是假驱动自说自话。
   ⇒ 只测假驱动会漏掉一种最坏情况：**判据本身对、但接错了线**
     （e.g. 断言永远看到假驱动的输出，生产分支其实没走到）。

⚠️ 真实 CPU 路径 `hardware_backed=False` ⇒ **性能无效**、不得当标定证据；
   本文件只用它验**接口与语义**。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load():
    p = _ROOT / "scripts" / "verify_s01_numeric_acceptance.py"
    spec = importlib.util.spec_from_file_location("_vna", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_vna"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeCfg:
    def __init__(self, raw):
        self.raw = raw


class _FakeSlot:
    def __init__(self, req_id, token_id, logits_digest=None):
        self.req_id = req_id
        self.token_id = token_id
        self.logits_digest = logits_digest


class _FakeOut:
    def __init__(self, slots):
        self.slots = slots


class _FakeDriver:
    """一个**刻意可配坏**的假驱动（忠实于新固定计划的形状）。

    `mode` 决定它会犯哪种错，用于验证判据能抓到它：
    - "good": 正确行为（严格按 `SCHEDULE` 的行数与 need_token 产出）
    - "leak_false": 对 `need_token=False` 的行也产出 token
    - "drop_row": 少返回一行（应被抓：slot 数量 + 请求集合）
    - "dup_row": 同一行返回两次（应被抓：唯一性）
    - "reorder": 打乱返回顺序（应被抓：顺序）
    - "wrong_len": 在最末一步不给 s0 token（应被抓：长度）
    - "no_skip": 跳过计数恒 0（应被抓）
    """

    def __init__(self, cfg, mode="good"):
        self.mode = mode
        self.rt = (cfg.raw.get("runtime") or {})
        self.s01_on = bool(self.rt.get("sample_needed_only"))
        self._skipped = 0
        self._sampled = 0
        self._seen: dict[str, int] = {}

    def load(self):
        return {}

    def forward(self, items):
        slots = []
        for it in items:
            rid = it.req.req_id
            if self.mode == "drop_row" and rid == "s2":
                continue
            tok = None
            if it.need_token:
                tok = 100 + self._seen.get(rid, 0)
                self._sampled += 1
            else:
                if self.mode == "leak_false":
                    tok = 999                      # 本该 None 却给了 token
                else:
                    self._skipped += 1
            self._seen[rid] = self._seen.get(rid, 0) + 1
            slots.append(_FakeSlot(rid, tok))
        if self.mode == "dup_row" and slots:
            slots.append(_FakeSlot(slots[0].req_id, 7))
        if self.mode == "reorder" and len(slots) > 1:
            slots = list(reversed(slots))
        if self.mode == "wrong_len":
            # 把本步 s0 的 token 抹掉 —— 只在它**本来要出**的时候
            for s in slots:
                if s.req_id == "s0" and s.token_id is not None:
                    s.token_id = None
                    break
        return _FakeOut(slots)

    def describe(self):
        return {
            "kind": "fake", "model_revision": "rev", "model_revision_resolved": "rev",
            "device": "cpu", "precision": "fp16", "sampling": "greedy",
            "execution_optimizations": {
                "sample_needed_only": {
                    "requested": self.s01_on, "effective": self.s01_on,
                    "n_sampled_rows": self._sampled,
                    "n_sampling_rows_skipped": (0 if self.mode == "no_skip"
                                                else self._skipped),
                },
            },
        }


class TestFixedScheduleDerivation(unittest.TestCase):
    """★ 计划本身必须**自洽**：期望值从 `SCHEDULE` 派生，且与用户给的数字一致。

    用户原文点名："每请求 8 个 token、总 24、False 行 2 个、真实混合组 1 个"。
    这四条**不是**从计划里"顺便读出来"的，而是**用户给定的约束**
    ⇒ 写成断言，一旦有人改了计划而没同步这些数字，立刻红。
    """

    def setUp(self):
        self.A = _load()

    def test_matches_user_stated_numbers(self):
        E = self.A.EXPECT
        self.assertEqual(E["tokens_per_req"], {"s0": 8, "s1": 8, "s2": 8})
        self.assertEqual(E["total"], 24)
        self.assertEqual(E["n_false_rows"], 2)
        self.assertEqual(E["n_mixed_groups"], 1)

    def test_the_single_mixed_group_is_the_decode_prompt_step(self):
        """唯一的混合组必须是"decode1 / prompt1 / decode1"那一步。"""
        E = self.A.EXPECT
        self.assertEqual(E["mixed_groups"], ["g3.mixed_decode_prompt"])

    def test_no_token_false_rows_are_the_two_s1_chunks(self):
        E = self.A.EXPECT
        self.assertEqual(sorted(r for _s, r in E["false_rows"]), ["s1", "s1"])

    def test_s1_prompt_is_split_into_three_feeds(self):
        """s1 的 72 个 prompt token 分三次喂完（70 + 1 + 1），两次 False 一次 True。"""
        A = self.A
        self.assertEqual(A.S1_FIRST_CHUNK + 1 + A.S1_TOPUP, A.PLEN["s1"])
        n_false_s1 = sum(1 for _s, it in A.SCHEDULE for r, _n, w in it
                         if r == "s1" and not w)
        self.assertEqual(n_false_s1, 2)

    def test_every_step_feeds_at_least_one_row(self):
        for step, plan in self.A.SCHEDULE:
            self.assertTrue(plan, f"{step} 是空组 ⇒ 驱动会拿到空批")


class TestNumericChecksOffline(unittest.TestCase):
    """判据层（假驱动）：坏输入必须红。"""

    def setUp(self):
        self.A = _load()
        self.A.load_config = lambda p: _FakeCfg({"runtime": {
            "logits_last_only": "s01_both" in str(p),
            "sample_needed_only": "s01_both" in str(p),
            "model_revision": "a09a35454c", "precision": "fp16"}})

    def _run(self, mode):
        self.A._build_driver_for = lambda cfg_path, driver: (
            _FakeDriver(self.A.load_config(cfg_path), mode=mode),
            self.A.load_config(cfg_path))
        return self.A._run_arm(Path("s01_both.yaml"), driver="fake")

    # ---- 好路径 ----
    def test_good_path_has_no_violations(self):
        arm = self._run("good")
        self.assertEqual(arm["violations"], [], arm["violations"])
        probs: list[str] = []
        self.A._check_expected_set(arm, label="x", probs=probs)
        self.A._check_lengths(arm, label="x", probs=probs)
        self.A._check_no_token_places(arm, label="x", probs=probs)
        self.A._check_mixed_group(arm, label="x", probs=probs)
        self.A._check_skip_count(arm, label="x", probs=probs)
        self.assertEqual(probs, [], probs)

    def test_good_path_per_step_shape_is_all_clean(self):
        arm = self._run("good")
        for st in arm["per_step"]:
            self.assertEqual(st["problems"], [], f"{st['step']}: {st['problems']}")
            self.assertEqual(st["n_slots"], st["n_items"], st["step"])
            self.assertEqual(st["returned_ids"], st["expected_ids"], st["step"])

    # ---- ① 请求集合 ----
    def test_dropped_row_is_caught(self):
        arm = self._run("drop_row")
        self.assertTrue(any("slot 数量" in v for v in arm["violations"]),
                        arm["violations"])
        probs: list[str] = []
        self.A._check_expected_set(arm, label="x", probs=probs)
        self.assertTrue(any("请求集合不符" in p for p in probs), probs)

    # ---- ② 长度/总数对账 ----
    def test_wrong_length_is_caught(self):
        arm = self._run("wrong_len")
        probs: list[str] = []
        self.A._check_lengths(arm, label="x", probs=probs)
        self.assertTrue(any("预期" in p for p in probs), probs)

    # ---- ③④ need_token=False 位置被污染 ----
    def test_leaked_token_on_false_row_is_caught(self):
        arm = self._run("leak_false")
        self.assertTrue(any("need_token=False 却产出" in v
                            for v in arm["violations"]), arm["violations"])
        probs: list[str] = []
        self.A._check_no_token_places(arm, label="x", probs=probs)
        self.assertTrue(any("no-token 位置" in p for p in probs), probs)

    def test_missing_false_coverage_is_caught(self):
        """场景根本没做 need_token=False ⇒ 必须报"覆盖不足"。"""
        arm = self._run("good")
        arm["no_token_positions"] = []
        probs: list[str] = []
        self.A._check_no_token_places(arm, label="x", probs=probs)
        self.assertTrue(any("覆盖" in p for p in probs), probs)

    # ---- 唯一性 / 顺序（本轮新增）----
    def test_duplicate_row_is_caught(self):
        arm = self._run("dup_row")
        self.assertTrue(any("重复" in v for v in arm["violations"]),
                        arm["violations"])

    def test_reordered_rows_are_caught(self):
        arm = self._run("reorder")
        self.assertTrue(any("顺序" in v for v in arm["violations"]),
                        arm["violations"])

    # ---- 混合组覆盖 ----
    def test_missing_mixed_group_is_caught(self):
        arm = self._run("good")
        # 人为把所有 per_step 标成非混合 ⇒ 判据必须红
        for st in arm["per_step"]:
            st["expect_no_token"] = []
        probs: list[str] = []
        self.A._check_mixed_group(arm, label="x", probs=probs)
        self.assertTrue(any("混合组" in p for p in probs), probs)

    # ---- ⑤ 跳过计数 ----
    def test_zero_skip_count_is_caught(self):
        arm = self._run("no_skip")
        probs: list[str] = []
        self.A._check_skip_count(arm, label="x", probs=probs)
        self.assertTrue(any("> 0" in p for p in probs), probs)

    def test_missing_skip_count_is_caught(self):
        arm = self._run("good")
        arm["execution_optimizations"] = {}
        probs: list[str] = []
        self.A._check_skip_count(arm, label="x", probs=probs)
        self.assertTrue(any("未报告" in p for p in probs), probs)

    # ---- ⑧ 摘要与身份 ----
    def test_token_digest_is_deterministic(self):
        a = self._run("good")
        b = self._run("good")
        self.assertEqual(a["token_digest"], b["token_digest"])
        self.assertIn("git_commit", self.A._git_identity())

    def test_logits_unverified_is_not_pass_wording(self):
        src = (_ROOT / "scripts" / "verify_s01_numeric_acceptance.py").read_text(
            encoding="utf-8")
        self.assertIn("未验证", src)
        self.assertIn('logits_status = "PASS" if same_lg else "FAIL"', src)


# --------------------------------------------------------------------------- #
# 真实 CPU 微型驱动（生产代码路径）
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - 环境探测
    import _tiny_support as _T

    _TORCH_OK = _T.torch is not None
    _TORCH_REASON = _T.TORCH_MISSING_REASON
except Exception as _exc:  # pragma: no cover
    _T = None  # type: ignore[assignment]
    _TORCH_OK = False
    _TORCH_REASON = f"{type(_exc).__name__}: {_exc}"


@unittest.skipIf(not _TORCH_OK, f"环境缺少 torch/transformers：{_TORCH_REASON}")
class TestRealCpuMicroDriver(unittest.TestCase):
    """★ 用**真实 CPU 微型驱动**跑完整固定计划 —— 判据必须挂在生产代码路径上。

    ⚠️ 这不是"多跑一遍假驱动"：`_build_cpu_driver` 造的是真
    `TransformerDriver` 子类，`forward` 走的是生产分支。
    若判据接错线（永远看到假输出），本类会**照样绿** ⇒ 所以本类**必须**存在。
    """

    CFG_A = _ROOT / "configs" / "mthreads" / "exec_opt" / "s01_reference.yaml"
    CFG_B = _ROOT / "configs" / "mthreads" / "exec_opt" / "s01_both.yaml"

    def setUp(self):
        self.A = _load()
        # ★ 两臂配置**在仓库内**、必须存在 ⇒ 缺失即**失败**（不转 skip）。
        #   理由见 `_tiny_support.require_local_file`：skip 会让"没跑"看起来像"通过"。
        _T.require_local_file(self.CFG_A, what="S01 对照臂配置")
        _T.require_local_file(self.CFG_B, what="S01 处理臂配置")

    def test_full_plan_passes_on_real_cpu_driver(self):
        a = self.A._run_arm(self.CFG_A, driver="cpu")
        b = self.A._run_arm(self.CFG_B, driver="cpu")

        # 生产路径上确实走到了两个开关的分支
        self.assertEqual(a["hardware_backed"], False)
        self.assertEqual(self.A._optimization_problems([a, b]), [])
        self.assertEqual(
            [arm["execution_optimizations"]["sample_needed_only"]["n_sampling_rows_skipped"]
             for arm in (a, b)], [0, 2])
        self.assertTrue(b["execution_optimizations"]["sample_needed_only"]["effective"],
                        "处理臂的 sample_needed_only 未生效 ⇒ 判据验的不是这条路")
        self.assertGreater(
            b["execution_optimizations"]["sample_needed_only"]["n_sampling_rows_skipped"],
            0, "跳过计数必须 > 0（证明 need_token=False 的行真的被跳过）")

        # 逐步结构全过
        for arm, label in ((a, "对照"), (b, "处理")):
            self.assertEqual(arm["violations"], [], f"{label}: {arm['violations']}")
            self.assertEqual(len(arm["per_step"]), self.A.EXPECT["n_steps"])

        # 六项判据全过
        probs: list[str] = []
        for arm, label in ((a, "对照"), (b, "处理")):
            self.A._check_expected_set(arm, label=label, probs=probs)
            self.A._check_lengths(arm, label=label, probs=probs)
            self.A._check_no_token_places(arm, label=label, probs=probs)
            self.A._check_mixed_group(arm, label=label, probs=probs)
        self.A._check_skip_count(b, label="处理", probs=probs)
        self.assertEqual(probs, [], probs)

        # 两臂逐位相同（这是 S01 的核心判据）
        self.assertEqual(a["token_digest"], b["token_digest"])
        self.assertEqual(a["tokens"], b["tokens"])

    def test_control_arm_does_not_claim_the_skip_path(self):
        """负对照：对照臂**没有**开开关 ⇒ 跳过计数必须为 0。

        若这里也 > 0，说明"跳过"不是由开关驱动的 ⇒ 判据在测一个常量。
        """
        _T.require_local_file(self.CFG_A, what="S01 对照臂配置")   # 缺 ⇒ 失败，不 skip
        a = self.A._run_arm(self.CFG_A, driver="cpu")
        sn = a["execution_optimizations"]["sample_needed_only"]
        self.assertFalse(sn["effective"])
        self.assertEqual(sn["n_sampling_rows_skipped"], 0)

    def test_token_counts_are_exactly_the_planned_8(self):
        """在真实生产路径上，每请求恰好 8 个 token（计划的算术后果）。"""
        _T.require_local_file(self.CFG_B, what="S01 处理臂配置")   # 缺 ⇒ 失败，不 skip
        b = self.A._run_arm(self.CFG_B, driver="cpu")
        self.assertEqual({k: len(v) for k, v in b["tokens"].items()},
                         {"s0": 8, "s1": 8, "s2": 8})


if __name__ == "__main__":
    unittest.main(verbosity=2)
