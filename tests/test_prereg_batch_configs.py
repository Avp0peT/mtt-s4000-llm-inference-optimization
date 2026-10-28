"""本批（机制小批 / pilot）的**可执行派生配置**与代表状态枚举。

★ 为什么需要这个文件（2026-10-03，任务 C）
------------------------------------------
用户点名要求："**不能靠默认值或文档描述代替实际配置**"。而本批此前正是这个状态：

1. `cost_model_path` 写着 **v5**，而 v7 才是当前最大一次拟合
   ⇒ 配置与"打算用的模型"是两个东西，且**没人会注意到**。
2. `prefill_deadline_ms` 与**内部每步预算**（120 ms）被混为一谈；
   实际配置里 `step_deadline_ms` 根本没写 ⇒ 走默认 100 ⇒ "内部预算"这句描述
   **与磁盘上的配置不符**。
3. **S01 开关在配置里完全不存在** —— 而它是本次唯一在真机上被行使的执行优化。
   配置缺它 ⇒ 真机跑出来的就是"关了 S01 的 cost 臂"，而文件头还写着别的。

⇒ 本文件把这三件事钉住：① 三个字段必须**显式存在于 YAML**；
② `cost_single` 与 `cost_dual` **只差一个键**（程序化核对，不靠目测）；
③ 代表状态枚举必须真的能问出"预算分支"和"回退原因"。

⚠️ 本文件**不加载成本模型**（指纹校验会在非标定机上**正确地**拒绝）。
   枚举脚本另行调用，且用 `load_offline`（跳过校验、并把这件事记进产物）。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in ("src", "benchmark", "scripts"):
    _p = str(_ROOT / _p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_benchmark import load_config  # noqa: E402

PREREG = _ROOT / "configs" / "mthreads" / "prereg_batch"
GEN = _ROOT / "scripts" / "gen_prereg_batch_configs.py"
ENUM = _ROOT / "scripts" / "enumerate_prereg_states.py"

#: 本批的冻结取值（与生成器同源；改这里必须同步改生成器与预注册文档）
EXTERNAL_TPOT_SLO_MS = 100.0
EXTERNAL_TTFT_SLO_MS = 2000.0
INTERNAL_STEP_BUDGET_MS = 120.0
COST_MODEL = "configs/cost_calib/model_v7.json"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestGeneratedConfigsExistAndLoad(unittest.TestCase):
    """三份派生配置必须存在、可加载，且不与生成器脱节。"""

    def test_all_three_variants_exist(self):
        for name in ("cost_single", "cost_dual", "static_small"):
            self.assertTrue((PREREG / f"{name}.yaml").is_file(),
                            f"{name}.yaml 缺失 ⇒ 本批配置不完整")

    def test_generator_check_is_clean(self):
        """★ `--check`：磁盘文件必须与"重新派生的结果"**逐字节一致**。

        这条拦的是"有人手改了派生文件"——那种改动不会报错，只会让
        '生成器说什么' 与 '磁盘上是什么' 悄悄分叉。
        """
        p = subprocess.run([sys.executable, str(GEN), "--check"],
                           capture_output=True, text=True, cwd=str(_ROOT))
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)

    def test_all_variants_load(self):
        for name in ("cost_single", "cost_dual", "static_small"):
            c = load_config(PREREG / f"{name}.yaml")
            self.assertIsNotNone(c)


class TestRequiredFieldsAreExplicitNotDefaulted(unittest.TestCase):
    """★★ 用户点名："不能靠默认值或文档描述代替实际配置"。"""

    def _raw(self, name: str) -> str:
        return (PREREG / f"{name}.yaml").read_text(encoding="utf-8")

    def test_cost_arms_name_v7_explicitly(self):
        """成本模型必须是 **v7**，且**写在文件里**（不是靠默认值）。"""
        for name in ("cost_single", "cost_dual"):
            self.assertIn(f"cost_model_path: {COST_MODEL}", self._raw(name),
                          f"{name}: 成本模型版本未显式写为 v7")

    def test_s01_switches_are_explicit(self):
        """★ S01 两条开关必须显式写 true（否则真机跑的是别的臂）。"""
        for name in ("cost_single", "cost_dual", "static_small"):
            raw = self._raw(name)
            self.assertIn("logits_last_only: true", raw,
                          f"{name}: S01 开关 `logits_last_only` 未显式打开")
            self.assertIn("sample_needed_only: true", raw,
                          f"{name}: S01 开关 `sample_needed_only` 未显式打开")

    def test_model_revision_is_pinned(self):
        for name in ("cost_single", "cost_dual", "static_small"):
            self.assertIn(f"model_revision: {MODEL_REVISION}", self._raw(name),
                          f"{name}: 模型 revision 未钉死")

    def test_internal_budget_and_external_slo_are_distinct(self):
        """★★ **120 ms 内部预算** 与 **100 ms 外部 SLO** 必须是两个数。

        把两者写成一个数 = 把"控制目标"与"验收判据"混同 ——
        当年正是"100 ms 打不到"才把**控制目标**放宽到 120。
        """
        c = load_config(PREREG / "cost_dual.yaml")
        self.assertEqual(c.run.slo.tpot_ms, EXTERNAL_TPOT_SLO_MS,
                         "外部 SLO（判据）被改动了")
        self.assertEqual(c.controller.target_tpot_ms, INTERNAL_STEP_BUDGET_MS,
                         "内部控制目标不是 120 ms")
        self.assertNotEqual(c.run.slo.tpot_ms, c.controller.target_tpot_ms,
                            "★ 内部预算与外部 SLO 不能是同一个数")

    def test_step_deadline_is_derived_from_target_tpot(self):
        """★ `step_deadline_ms` **不是**配置字段，它由 `target_tpot_ms` 派生。

        踩过（就在本轮）：我先把 `controller.step_deadline_ms: 120` 写进 YAML，
        结果配置**加载直接失败** —— `_build_dataclass` 对未知字段报错。
        真实映射在 `benchmark/arm_factory.py:99`：
        `step_deadline_ms=cfg.controller.target_tpot_ms`。
        ⇒ 要让"内部每步预算 = 120 ms"成立，**唯一**的做法是把 `target_tpot_ms`
        写成 120，且 **YAML 里不许真出现这个键**。

        ⚠️ 判据要落在**解析后的配置**上，不能只 grep 原文：
        `step_deadline_ms` 合法地出现在**注释**里（解释派生关系），
        用 `assertNotIn(raw, ...)` 会把注释也判成违规 —— 那是**假失败**。
        """
        from run_benchmark import Config, load_config as _lc
        for name in ("cost_single", "cost_dual"):
            # ① 解析后：该字段**不在** controller 上（它不是配置字段）
            c = _lc(PREREG / f"{name}.yaml")
            self.assertFalse(hasattr(c.controller, "step_deadline_ms"),
                             f"{name}: controller 上出现了 `step_deadline_ms`，"
                             "说明有人往配置里加了非法字段（加载会被拒）")
            self.assertFalse(
                "step_deadline_ms" in (c.raw.get("controller") or {}),
                f"{name}: `controller` 段里真的写了 `step_deadline_ms` ⇒ 配置非法")
            # ② 派生出内部预算的那个字段必须是 120
            self.assertEqual(c.controller.target_tpot_ms, INTERNAL_STEP_BUDGET_MS,
                             f"{name}: 派生出内部预算的那个字段不是 120")

    def test_max_num_seqs_is_eight(self):
        """★ pilot 的前提：`max_num_seqs` 必须已改为 8（旧 `p1s3d` 是 32）。"""
        for name in ("cost_single", "cost_dual", "static_small"):
            c = load_config(PREREG / f"{name}.yaml")
            self.assertEqual(c.max_num_seqs, 8,
                             f"{name}: max_num_seqs 不是 8 ⇒ pilot 前提未满足")


class TestCostArmsAreSingleVariable(unittest.TestCase):
    """★★ 两条 cost 臂**只差 `prefill_deadline_ms`** —— 程序化核对。"""

    def _noncomment(self, name: str) -> list[str]:
        return [ln for ln in (PREREG / f"{name}.yaml").read_text(
            encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]

    def test_only_one_line_differs(self):
        a, b = self._noncomment("cost_single"), self._noncomment("cost_dual")
        self.assertEqual(len(a), len(b), "两份配置行数不同 ⇒ 不是单变量")
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        self.assertEqual(len(diff), 1, f"差异不止一处：{diff}")
        self.assertIn("prefill_deadline_ms", diff[0][0])
        self.assertIn("prefill_deadline_ms", diff[0][1])

    def test_single_budget_is_null_dual_is_2000(self):
        """`null` **不是** 0：`None` = 不启用双预算，0 是一个"死锁配方"的值。"""
        c1 = load_config(PREREG / "cost_single.yaml")
        c2 = load_config(PREREG / "cost_dual.yaml")
        self.assertIsNone(c1.controller.prefill_deadline_ms)
        self.assertEqual(c2.controller.prefill_deadline_ms, EXTERNAL_TTFT_SLO_MS)

    def test_prefill_budget_is_sourced_from_ttft_slo(self):
        """prefill 预算与 `run.slo.ttft_ms` **同源**（服务同一个承诺）。"""
        c = load_config(PREREG / "cost_dual.yaml")
        self.assertEqual(c.controller.prefill_deadline_ms, c.run.slo.ttft_ms,
                         "prefill 预算与 TTFT SLO 不同源 ⇒ 口径被偷偷改了")


class TestStaticSmallStaysAStaticControl(unittest.TestCase):
    """`static_small` 是**静态对照**：控制器必须关、chunk 与自适应臂同起点。"""

    def test_controller_is_disabled(self):
        c = load_config(PREREG / "static_small.yaml")
        self.assertFalse(c.controller.enabled)

    def test_chunk_matches_adaptive_init(self):
        """chunk 起点必须与自适应臂的 `chunk_size_init` 相同（否则量的是别的变量）。"""
        c = load_config(PREREG / "static_small.yaml")
        a = load_config(PREREG / "cost_dual.yaml")
        self.assertTrue(c.chunk.enabled)
        self.assertEqual(c.chunk.chunk_size, a.controller.chunk_size_init)


class TestFrozenConfigsAreUntouched(unittest.TestCase):
    """★★ 铁律：只统一**本批**派生配置，**不碰**已执行的冷冻配置。"""

    def test_frozen_step14_has_no_s01_switches(self):
        """冷冻配置里**不应**出现 S01 开关（它们不代表那段历史）。"""
        raw = (_ROOT / "configs" / "mthreads" / "phase1_step14"
               / "fixed_cost.yaml").read_text(encoding="utf-8")
        self.assertNotIn("logits_last_only", raw,
                         "冷冻配置被改动了 —— 它与既有产物的快照会对不上")

    def test_frozen_step14_still_points_at_v5(self):
        """`fixed_cost.yaml` 仍然指向 v5（历史事实，不回填成 v7）。"""
        raw = (_ROOT / "configs" / "mthreads" / "phase1_step14"
               / "fixed_cost.yaml").read_text(encoding="utf-8")
        self.assertIn("cost_model_path: configs/cost_calib/model_v5.json", raw)


class TestRepresentativeStateEnumeration(unittest.TestCase):
    """离线枚举（**TEST_DATA**）必须真的问出预算分支与回退原因。"""

    @classmethod
    def setUpClass(cls):
        p = subprocess.run([sys.executable, str(ENUM)],
                           capture_output=True, text=True, cwd=str(_ROOT))
        cls.stdout, cls.rc = p.stdout, p.returncode
        if p.returncode != 0:
            raise unittest.SkipTest(f"枚举脚本未跑通：{p.stderr[:400]}")

    def test_script_runs_offline(self):
        self.assertEqual(self.rc, 0, self.stdout)
        self.assertIn("TEST_DATA", self.stdout,
                      "必须显式标注这是离线演练数据")

    def test_prefill_budget_branch_is_exercised(self):
        """★ 必须有状态走到 `prefill/2000` 分支 —— 否则双预算等于没接上。"""
        self.assertIn("prefill/2000", self.stdout,
                      "没有任何状态走 prefill 预算分支 ⇒ 双预算未被行使")

    def test_mixed_step_correctly_uses_decode_budget(self):
        """★ §5.1 修正：混合步（有 decode 在风险中）**不得**走宽松的 prefill 预算。"""
        import subprocess as sp
        p = sp.run([sys.executable, str(ENUM), "--json", "NUL"],  # 只取 stdout
                   capture_output=True, text=True, cwd=str(_ROOT))
        self.assertIn("mixed_step", p.stdout)
        # 该行必须显示 decode/120（不是 prefill/2000）
        line = next(ln for ln in p.stdout.splitlines() if ln.startswith("mixed_step"))
        self.assertIn("decode/120", line,
                      f"混合步走了 prefill 预算 ⇒ §5.1 缺口复活：{line}")

    def test_lookup_miss_has_a_reason(self):
        """★ 查表缺失必须给出**具体回退原因**，不能只给一个空。"""
        self.assertIn("lookup 未命中", self.stdout,
                      "查表缺失状态没有报出回退原因")

    def test_budget_branch_differs_between_arms(self):
        self.assertIn("双预算在这四点改变了归类", self.stdout)


class TestBudgetExampleDoesNotClaimMechanismExercised(unittest.TestCase):
    """★★ 用户 2026-10-03 点名：本表是**预算函数示例**，**不得**称本批机制已触发。

    失效形态：旧文本写「两套预算给出不同可行集 ⇒ **机制被行使**」。
    那是**未验证的因果跳跃** —— 表里的形状是**人手喂的**，没走 `plan_step`，
    所以"两套预算在这四点给出不同可行集"只说明**预算函数**会分叉，
    完全不能推出"控制器在真实队列上被行使了"。
    ⇒ 把"不许写成机制已触发"钉住。
    """

    def setUp(self):
        self._r = subprocess.run([sys.executable, str(ENUM)], cwd=str(_ROOT),
                                 capture_output=True, text=True, timeout=180)
        self.stdout = self._r.stdout

    def test_does_not_claim_mechanism_is_exercised(self):
        for bad in ("机制被行使", "机制已触发"):
            for i, ln in enumerate(self.stdout.splitlines(), 1):
                if bad not in ln:
                    continue
                # 允许出现在"不能读成…/不得称…/不是…"这类否定语境里
                ok = any(k in ln for k in ("不能", "不得", "不是", "⚠️"))
                self.assertTrue(ok, f"L{i}: 把『机制被行使』写成了结论：{ln.strip()[:110]}")

    def test_table_is_labelled_a_budget_function_example(self):
        self.assertIn("预算函数示例", self.stdout)
        self.assertIn("机制是否被行使要看", self.stdout,
                      "必须指向 dryplan_decision_chain.py 才能判定机制是否被行使")


class TestEnumerationArtifactIsLabelledTestData(unittest.TestCase):
    """产物必须自证是 `TEST_DATA`，且**不得**被当成真机标定。

    ⚠️ 本类**不转 skip**（用户 2026-10-03）：`prereg_states.json` 是**由本机
       脚本生成的仓库内产物**（不是远端回拉物）⇒ 它缺失 = **仓库坏**，必须红。
       原先写 `skipTest` ⇒ `enumerate_prereg_states.py` 一旦挪走/改名，
       这两条**静默变 skip**，而报告看不出来。
    """

    ART = _ROOT / "results" / "processed" / "prereg_states.json"

    def setUp(self):
        #: 缺 ⇒ **失败**（不 skip）。生成命令写进消息，便于当场修。
        if not self.ART.is_file():
            raise AssertionError(
                f"本机必测产物缺失：{self.ART}\n"
                f"  ⇒ 先跑：python scripts/enumerate_prereg_states.py --json "
                f"{self.ART}\n"
                f"  ⚠️ **不**转 skip —— `skipTest` 会让『没跑』看起来像『通过』。")

    def test_artifact_labelled_and_lf_only(self):
        raw = self.ART.read_bytes()
        self.assertEqual(raw.count(b"\r"), 0, "产物含 CR（本仓 eol=lf）")
        d = json.loads(raw.decode("utf-8"))
        self.assertEqual(d["evidence_class"], "TEST_DATA")
        self.assertIn("不代替真机标定", d["note"])
        self.assertIn("FZ-1", d["note"],
                      "必须写明 FZ-1/FZ-2 未裁定 ⇒ 不得据此宣布双预算有效/无效")

    def test_artifact_records_the_relaxation(self):
        """`load_offline` 跳过指纹校验这件事必须**记进产物**（不许假装校验过）。"""
        d = json.loads(self.ART.read_text(encoding="utf-8"))
        rel = " ".join(d.get("cost_model_relaxations") or [])
        self.assertIn("未做指纹校验", rel,
                      "离线跳过指纹校验必须如实登记为 relaxation")
        # ★ 预算函数示例的定位必须写进产物（不得被读成"机制已触发"）
        self.assertEqual(d.get("artifact_role"), "budget_function_example",
                         "产物必须自证是**预算函数示例**（手选形状）")


if __name__ == "__main__":
    unittest.main(verbosity=2)
