"""pilot 流程的**结构**判据（`NEXT_BATCHES_PREREG.md` §10）。

★ 为什么用测试锁文档（2026-10-03，任务 D）
------------------------------------------
用户点名四项修订，其中三项的失效形态都是"**文档改了、但下次又被写回去**"：
文档没有编译期错误，一个"看起来更严格"的旧规则很容易在下次编辑时复活。
⇒ 把**结构性**要求写成测试：标题必须分节、"停"这个字不能与达标率 0/1 同现、
预算只能有一处定义。

⚠️ 这里**不**检查措辞是否好读 —— 只检查会导致误判的**结构**。
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PREREG = _ROOT / "NEXT_BATCHES_PREREG.md"


class TestPilotSectionsAreSeparated(unittest.TestCase):
    """★ 要求①：拆开"开跑前检查"与"pilot 结束后的扩大条件"。"""

    @classmethod
    def setUpClass(cls):
        cls.text = PREREG.read_text(encoding="utf-8")

    def test_doc_exists(self):
        self.assertTrue(PREREG.is_file())

    def test_three_distinct_subsections(self):
        for h in ("### 10.1", "### 10.2", "### 10.3", "### 10.4"):
            self.assertIn(h, self.text, f"缺章节 {h}")

    def test_precheck_and_expand_are_named(self):
        self.assertIn("开跑前检查", self.text)
        self.assertIn("扩大条件", self.text)

    def test_pilot_purpose_is_stated_and_bounded(self):
        """★ 目的必须写明"不用于宣布性能优势"。"""
        self.assertIn("pilot 不用于宣布性能优势", self.text,
                      "必须明写 pilot 不得用于宣布性能优势")

    def test_purpose_says_it_cannot_update_conclusion_table(self):
        self.assertIn("不得进 `EXPERIMENTS.md` 结论表", self.text)


class TestRateZeroOrOneStopRuleIsDeleted(unittest.TestCase):
    """★★ 要求②：删除"仅因静态对照达标率 0 或 1 就停"的规则。"""

    @classmethod
    def setUpClass(cls):
        cls.text = PREREG.read_text(encoding="utf-8")

    def test_old_rule_is_recorded_as_abolished(self):
        """旧规则要被**登记为废止**（撤回必须传播到引用处），不是悄悄删掉。"""
        self.assertIn("已废止的旧规则", self.text,
                      "旧规则必须显式登记为废止（撤回要留痕）")
        self.assertIn("不再生效", self.text)

    def test_no_live_rule_stopping_on_rate_in_zero_or_one(self):
        """★ 不许存在**仍在生效**的"达标率 ∈ {0,1} ⇒ 停"。

        做法：找出所有含 "∈ {0, 1}" 或 "∈ {0,1}" 的**行**，
        逐行确认它要么在"已废止"的引用块里（`>` 引用或"原规则"上下文），
        要么明确标了废止。任何"裸"的这类行都判失败。
        """
        bad: list[str] = []
        for i, ln in enumerate(self.text.splitlines(), 1):
            if "∈ {0, 1}" in ln or "∈ {0,1}" in ln:
                # 合法出现：引用块（以 > 开头）或显式标"原规则/已删/废止/修订说明"
                if ln.lstrip().startswith(">"):
                    continue
                if any(k in ln for k in ("原规则", "已删", "废止", "旧规则", "修订", "删掉")):
                    continue
                bad.append(f"L{i}: {ln.strip()[:90]}")
        self.assertEqual(bad, [], f"仍有生效的『达标率 0/1 就停』规则：{bad}")

    def test_no_live_stop_on_satisfaction_rate_alone(self):
        """★ 更一般地：不许出现"`slo_satisfaction_rate` ⇒ 停"这种单独判停。

        ⚠️ 与上一条同样要**豁免"已删的旧规则"的引用行**：
        登记废止时必须能**原文引述**旧规则，否则"撤回了什么"无从核对。
        """
        bad: list[str] = []
        for i, ln in enumerate(self.text.splitlines(), 1):
            if "slo_satisfaction_rate" not in ln:
                continue
            if not ("⇒ **停**" in ln or "⇒ 停" in ln):
                continue
            if ln.lstrip().startswith(">"):
                continue
            if any(k in ln for k in ("原规则", "已删", "废止", "旧规则")):
                continue
            bad.append(f"L{i}: {ln.strip()[:100]}")
        self.assertEqual(bad, [], f"仍有生效的『达标率单独判停』规则：{bad}")

    def test_open_interval_rule_is_also_abolished(self):
        """★★ 终裁（用户 2026-10-03 第二次点名）：**(0,1) 这条也要删**。

        ⚠️ 第一次修订删了"`∈ {0,1}` ⇒ 停"，却换上"必须落在**开区间 (0,1)**"——
           **同族错误**：仍用一个数值区间去决定"要不要扩大"。
        ⇒ 现在要求：① (0,1) 只能出现在**废止记录**里（`>` 引用行）；
          ② 现行判据必须是 **E1 机制被行使 + E4 身份**；
          ③ 达标率**只作解释**，明确写"不单独构成停/续"。
        """
        bad: list[str] = []
        for i, ln in enumerate(self.text.splitlines(), 1):
            if "开区间 (0,1)" not in ln:
                continue
            if ln.lstrip().startswith(">"):      # 废止记录里的原文引述 ⇒ 允许
                continue
            if any(k in ln for k in ("已废止", "旧规则", "废止", "原规则", "修订", "删掉")):
                continue
            bad.append(f"L{i}: {ln.strip()[:90]}")
        self.assertEqual(bad, [], f"仍有生效的『必须落在 (0,1)』规则：{bad}")

    def test_replacement_judges_by_mechanism_and_identity(self):
        """★ 替代判据 = **机制被行使（E1）+ 身份一致（E4）**，不是数值区间。"""
        sec = self.text[self.text.find("### 10.3"): self.text.find("### 10.4")]
        self.assertIn("E1", sec)
        self.assertIn("E4", sec)
        self.assertIn("机制被行使", sec)
        self.assertIn("身份", sec)

    def test_rate_is_only_interpretation_not_a_switch(self):
        """★ 达标率三个取值都**只作解释**，写清"不是停/续的判据"。"""
        sec = self.text[self.text.find("### 10.3"): self.text.find("### 10.4")]
        self.assertIn("只用于解释", sec)
        self.assertIn("不单独构成", sec)
        # 分母仍必须报（防"分母是 0 / 字段缺失"被读成"真没达标"）
        self.assertIn("n_slo_satisfied", sec)
        self.assertIn("n_requests", sec)
        self.assertIn("分母", sec)

    def test_zero_rate_requires_cause_attribution(self):
        """★ 达标率 = 0 时必须**先分辨成因**，不得直接判停。"""
        idx = self.text.find("先分辨成因")
        self.assertGreater(idx, 0, "0.0 的处置必须写『先分辨成因』")
        # 该行附近必须出现"不是『停』"（或等价否定）
        window = self.text[idx: idx + 200]
        self.assertIn("不是", window)


class TestBudgetConflictIsReconciled(unittest.TestCase):
    """★★ 要求③：统一 30 分钟与 45 分钟的预算矛盾。"""

    @classmethod
    def setUpClass(cls):
        cls.text = PREREG.read_text(encoding="utf-8")

    def test_single_authoritative_definition(self):
        self.assertIn("10.4", self.text)
        self.assertIn("已统一", self.text)
        self.assertIn("只有一处定义", self.text)

    def test_45_is_cap_30_is_expected(self):
        sec = self.text[self.text.find("### 10.4"):]
        self.assertIn("45 min", sec)
        self.assertIn("30 min", sec)
        self.assertIn("上限", sec)
        self.assertIn("预期", sec)

    def test_budget_flag_semantics_match_decide_stop(self):
        """★★ 要求⑤（用户 2026-10-03 点名）：预算说明必须与**实际停止逻辑**一致。

        `--budget-minutes` 的真实语义是 `benchmark/exec_opt_plan.py:decide_stop`：
        **只在整轮边界上**决定要不要开下一轮（预算不够跑完当前整轮 ⇒ 不开始那一轮）；
        **不是**"到点 kill"，也**不打断**正在跑的 run。
        ⇒ 文档必须同时写出这三条，而不是笼统说"干净退出守卫"。
        """
        sec = self.text[self.text.find("### 10.4"):]
        self.assertIn("decide_stop", sec, "必须点名真实判定函数，否则是'描述'而非'对齐'")
        self.assertIn("不是", sec)
        self.assertIn("硬超时", sec)
        # 关键：必须写出"只在整轮之间"与"不打断"
        self.assertIn("整轮", sec)
        self.assertIn("不打断", sec)
        # 且必须写出"不允许不均衡重复"这条真实动机
        self.assertIn("均衡", sec)

    def test_no_second_contradicting_budget_number(self):
        """★ §11 不得再写第二套 pilot 预算数（那正是原矛盾的来源）。"""
        sec11 = self.text[self.text.find("## 11."): self.text.find("## 12.")]
        self.assertIn("45", sec11, "§11 应引用统一的 45 min 上限")
        # 不许在 §11 出现"≤ 30 min"这种另一种表述
        self.assertNotIn("≤ 30 min", sec11,
                         "§11 又出现了一套独立的 pilot 预算 ⇒ 第二处真相")


class TestPrecheckGateExistence(unittest.TestCase):
    """要求④配套：开跑前检查必须是**表**，且含远端可达与指纹两条硬门。"""

    @classmethod
    def setUpClass(cls):
        cls.text = PREREG.read_text(encoding="utf-8")
        cls.sec = cls.text[cls.text.find("### 10.2"): cls.text.find("### 10.3")]

    def test_precheck_has_a_table(self):
        self.assertIn("| # | 检查 |", self.sec)

    def test_precheck_covers_remote_and_fingerprint(self):
        self.assertIn("远端可达", self.sec)
        self.assertIn("指纹", self.sec)

    def test_precheck_covers_config_explicitness(self):
        """A1–A5 必须覆盖本批四类显式项（否则 D 与前序任务脱节）。"""
        for k in ("S01", "内部预算", "max_num_seqs", "load_config"):
            self.assertIn(k, self.sec, f"开跑前检查未覆盖 {k}")


class TestSshRootCauseStaysUnconfirmed(unittest.TestCase):
    """★★ SSH 不可达的**根因必须保持待确认**（用户点名）。

    失效形态：症状（"TCP 可连、无 banner"）在历史上**两次**确实由端口变更引起
    ⇒ 很容易被顺手写成"端口已经变化"。那是一次**未验证的因果跳跃**：
    同一症状还可能来自"服务未起 / 端口被占用 / 上游策略变更"。
    ⇒ 把"不许写成结论"钉住。
    """

    def _texts(self) -> dict[str, str]:
        return {
            "NEXT_BATCHES_PREREG.md": PREREG.read_text(encoding="utf-8"),
            "STATE.json": (_ROOT / "handoff" / "optimization_20260929"
                           / "STATE.json").read_text(encoding="utf-8"),
        }

    def test_root_cause_is_marked_unconfirmed(self):
        for name, t in self._texts().items():
            self.assertIn("待确认", t, f"{name}: 未标注根因待确认")

    def test_does_not_assert_port_has_changed(self):
        """★ 不许把"端口已经变化"写成**结论**。"""
        for name, t in self._texts().items():
            for bad in ("端口已经变化", "端口已变 ⇒", "已确认端口变更"):
                # 允许出现在"不写成…"这类否定/引用语境里
                for i, ln in enumerate(t.splitlines(), 1):
                    if bad not in ln:
                        continue
                    ok = any(k in ln for k in ("不写成", "不是结论", "是假设",
                                               "不得据此", "保持待确认"))
                    self.assertTrue(ok, f"{name} L{i}: 把端口变更写成了结论：{ln.strip()[:110]}")

    def test_diagnostic_signature_is_labelled_a_hypothesis(self):
        t = self._texts()["NEXT_BATCHES_PREREG.md"]
        self.assertIn("假设", t)
        self.assertIn("控制台", t, "必须指向控制台核对，而不是猜端口")


if __name__ == "__main__":
    unittest.main(verbosity=2)
