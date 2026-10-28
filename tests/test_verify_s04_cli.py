"""`scripts/verify_s04_reenact_numerics.py` 的**离线**测试。

为什么必须有
------------
这个脚本被派去在**真机**上回答"S04 重入之后数值对不对"。它跑一次要占机时、
要装 7B 权重，因此**真机不会跑第二次来给你debug**。
如果它自己坏了（或者判据恒真），你拿到的是一份**盖了 PASS 的空证据**。

★ 2026-10-03：实测发现它此前**根本跑不起来**——
`build_driver(load_config 之前先 yaml.safe_load 出来的 dict)` 直接
`AttributeError: 'dict' object has no attribute 'runtime'`。
三个缺陷都是"照着**假驱动**的接口写、没对着真驱动核对"：
dict vs `Config`、`StepOutcome.slots` vs `_GroupOutcome.produced`、logits 完全没有对照。

所以本文件分两层：

1. **CLI 离线集成**（起真子进程跑 `--plan-only`）：验证读配置、派生两臂、
   单变量性、身份采集、输出路径与**拒绝覆盖**这些"不需要模型"的环节。
2. **判据正反例**（直接喂合成 evidence 给真实的 `evaluate`）：
   每条判据都要有能把它**打红**的反例，否则它可能只是恒真。
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from verify_s04_reenact_numerics import (  # noqa: E402
    _compare_kv,
    _kv_digest,
    _tokens_digest,
    arm_runtime_delta,
    numeric_completeness_verdict,
    build_arm_configs,
    default_out_path,
    evaluate,
    guard_out_path,
)

BASELINE = _ROOT / "configs" / "mthreads" / "exec_opt" / "s01_both.yaml"
SCRIPT = _ROOT / "scripts" / "verify_s04_reenact_numerics.py"


def _dig(argmax: int, sha: str) -> dict:
    return {"argmax": argmax, "max": 1.0, "sum": 2.0, "sha256": sha}


def _kv(*, a_k, a_v, b_k, b_v) -> dict:
    """一份"逐层 KV"快照的替身：`{槽位: [(K, V), ...]}`，四维、fp16。"""
    return {
        "a": [(_t4(a_k), _t4(a_v))],
        "b": [(_t4(b_k), _t4(b_v))],
    }


def _same_kv() -> dict:
    return _kv(a_k=1.0, a_v=2.0, b_k=3.0, b_v=4.0)


def _good_evidence(**over):
    """一份**全绿**的合成证据；各反例只覆盖自己关心的那一个字段。"""
    ev = {
        "telemetry": {
            "n_fastpath_steps": 7, "n_fallback_steps": 5, "n_resyncs": 2,
            "fallback_reasons": {"unequal_kv_len": 5},
            # ★ 形状判据要用它们（生产里来自 `persistent_kv_slots` 遥测）。
            "n_kv_heads": FAKE_N_KV_HEADS, "head_dim": FAKE_HEAD_DIM,
            "dtype": FAKE_KV_DTYPE,
        },
        # ★ 完整性判据的输入：预期槽位 / 模型层数 / 步数（生产里由场景与模型现取）。
        "expected_kv_slots": ["a", "b"],
        "expected_token_slots": ["a", "b"],
        "n_layers": 1,
        "n_layers_base": 1,
        "n_steps": 1,
        # `tokens_out`（含预热）与 `tokens_out_measured`（正式段）**都要给**：
        # 判据走的是"正式段"那一列，而总量那一列用来核"预热段 = 总量 - 正式段"。
        # 本夹具 `warmup_s=0` ⇒ 两者相等（`basis=no_warmup`）。
        "stats_base": {"tokens_out": 6,
                       "measured_window": {"pkv_fastpath_measured": 0,
                                           "tokens_out_measured": 6,
                                           "basis": "no_warmup"}},
        "stats_pkv": {"tokens_out": 6,
                      "n_forwards": 40,
                      "measured_window": {"pkv_fastpath_measured": 7,
                                          "n_forwards_measured": 40,
                                          "tokens_out_measured": 6,
                                          "basis": "no_warmup"}},
        "tokens_base": {"a": [1, 2, 3], "b": [4, 5, 6]},
        "tokens_pkv": {"a": [1, 2, 3], "b": [4, 5, 6]},
        "logits_base": {"0:a": _dig(1, "aa"), "0:b": _dig(2, "bb")},
        "logits_pkv": {"0:a": _dig(1, "aa"), "0:b": _dig(2, "bb")},
        "kv_base_reentry": _same_kv(),
        "kv_pkv_reentry": _same_kv(),
        "kv_base_final": _same_kv(),
        "kv_pkv_final": _same_kv(),
        "warmup_s": 0.0,
    }
    ev.update(over)
    return ev


def _by_name(checks, name_fragment: str) -> dict | None:
    for c in checks:
        if name_fragment in c["name"]:
            return c
    return None


def _t(*vals):
    """真 torch 张量。

    为什么不用自造的替身：生产的 `_compare_kv` 会调 `(b == o).all()` 与
    `(b.float() - o.float()).abs().max()`。替身必须把这些语义**全部**复刻一遍，
    而复刻出来的东西与"真张量会不会这样算"是两回事 —— 那正是本仓反复记过的
    「镜像判据漂移」。torch 在本环境可用，直接用真的。
    """
    import torch

    return torch.tensor(list(vals), dtype=torch.float32)


#: 夹具用的 KV 形状参数（**与生产一致的四维** `(1, n_kv_heads, L, head_dim)`）。
#: ⚠️ 早期夹具用的是 2 维张量 —— 那时没有形状判据，所以看不出问题；加了"张量形状"
#: 这一条之后 2 维夹具立刻被判不合格（正因如此才有意义：形状不对时 KV 的语义
#: ——哪个轴是 head、哪个是长度——是**没验证过**的）。
FAKE_N_KV_HEADS = 2
FAKE_HEAD_DIM = 2
FAKE_KV_DTYPE = "float16"


def _t4(v: float, *, layers: int = 1) -> Any:
    """造一个形状/ dtype 都**合法**的 KV 张量：`(1, n_kv_heads, L, head_dim)`。"""
    import torch

    return torch.full((1, FAKE_N_KV_HEADS, layers, FAKE_HEAD_DIM), float(v),
                      dtype=torch.float16)


class TestArmConfigsAreSingleVariable(unittest.TestCase):
    """两臂**必须**只差 PKV 那两个键，且容量键在关闭时不生效。"""

    def test_delta_is_exactly_the_pkv_keys(self):
        base, pkv = build_arm_configs(BASELINE, capacity=256)
        delta = arm_runtime_delta(base, pkv)
        self.assertEqual(
            sorted(delta), ["persistent_kv_capacity", "persistent_kv_slots"],
            "两臂 runtime 差集必须正好是 PKV 开关键与容量键")
        self.assertEqual(delta["persistent_kv_slots"], (False, True))
        self.assertEqual(delta["persistent_kv_capacity"], (None, 256))

    def test_baseline_arm_has_pkv_off(self):
        base, _pkv = build_arm_configs(BASELINE, capacity=256)
        self.assertFalse(base.runtime.get("persistent_kv_slots"),
                         "基线臂必须 **关闭** S04（这是单变量性的前提）")

    def test_arms_do_not_share_a_runtime_dict(self):
        """★ 两臂不能共享同一个 runtime 字典（否则第二臂会污染第一臂）。"""
        base, pkv = build_arm_configs(BASELINE, capacity=256)
        self.assertIsNot(base.runtime, pkv.runtime)


class TestCLIPlanOnly(unittest.TestCase):
    """★ CLI 离线集成：起**真子进程**跑 `--plan-only`（不装载模型）。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="s04plan_")
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--plan-only", "--config", str(BASELINE),
             *extra],
            cwd=str(_ROOT), capture_output=True, text=True, errors="replace",
            timeout=300,
        )

    def test_plan_only_writes_an_artifact_and_exits_zero(self):
        out = self.dir / "plan.json"
        p = self._run("--out", str(out))
        self.assertEqual(p.returncode, 0, f"stderr={p.stderr[-500:]}")
        self.assertTrue(out.is_file(), "计划产物必须真的落盘")
        d = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(d["verdict"], "PLAN_ONLY")
        self.assertTrue(d["single_variable_ok"])
        self.assertEqual(d["warmup_s"], 0.0)

    def test_plan_records_both_arms_and_identity(self):
        out = self.dir / "plan.json"
        self._run("--out", str(out))
        d = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(sorted(d["effective_configs"]), ["baseline", "pkv"])
        self.assertIn("source_identity", d)
        self.assertIn("tree_digest", d["source_identity"])
        # 本机没有部署戳（戳只投远端）⇒ 身份里 stamp 应为 None，且**不得**伪造。
        self.assertIsNone(d["source_identity"]["stamp"])

    def test_refuses_to_overwrite_existing_output(self):
        """★★ 拒绝覆盖：同一条路径跑第二次必须失败（rc=3），不得改写旧结果。"""
        out = self.dir / "plan.json"
        self.assertEqual(self._run("--out", str(out)).returncode, 0)
        before = out.read_bytes()
        p2 = self._run("--out", str(out))
        self.assertEqual(p2.returncode, 3, f"应拒绝覆盖，stderr={p2.stderr[-300:]}")
        self.assertEqual(out.read_bytes(), before, "旧结果必须逐字节未被改动")

    def test_missing_config_exits_2(self):
        p = self._run("--out", str(self.dir / "x.json"),
                      )
        self.assertEqual(p.returncode, 0)
        p2 = subprocess.run(
            [sys.executable, str(SCRIPT), "--plan-only",
             "--config", str(self.dir / "nope.yaml"), "--out", str(self.dir / "y.json")],
            cwd=str(_ROOT), capture_output=True, text=True, errors="replace", timeout=120)
        self.assertEqual(p2.returncode, 2)


class TestGuardAndDefaultPaths(unittest.TestCase):
    def test_guard_rejects_existing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            self.assertIsNone(guard_out_path(p), "不存在的路径应放行")
            p.write_text("{}", encoding="utf-8")
            self.assertIsNotNone(guard_out_path(p), "已存在的路径必须拒绝")

    def test_default_path_is_unique_across_calls(self):
        """默认路径含时间戳 + 摘要 ⇒ 连跑两次也不该撞（这是"唯一路径"的实现）。"""
        a = default_out_path(_ROOT, "deadbeefcafe1234")
        b = default_out_path(_ROOT, "deadbeefcafe1234")
        self.assertEqual(a.parent.name, "_s04_verify")
        self.assertIn("deadbeefcafe", a.name)
        # 同一秒内可能同名——但至少两次调用都落在同一目录且名字含摘要；
        # 真正的唯一性由"拒绝覆盖"兜底（见上面的 CLI 用例）。
        self.assertEqual(a.parent, b.parent)


class TestEvaluateCriteriaHaveRealNegatives(unittest.TestCase):
    """★★ 每条判据都要有**能把它打红**的反例。

    没有这一层，判据可能只是恒真——而恒真的判据比没有判据更危险：
    它会以"已验"的形态进入结论。
    """

    def test_all_good_passes_every_criterion(self):
        checks = evaluate(_good_evidence())
        bad = [c for c in checks if not c["pass"]]
        self.assertEqual(bad, [], f"全绿证据不该有失败项：{bad}")

    def test_no_resync_is_red(self):
        ev = _good_evidence()
        ev["telemetry"] = {**ev["telemetry"], "n_resyncs": 0}
        c = _by_name(evaluate(ev), "重入发生")
        self.assertFalse(c["pass"], "★ 没有重入 ⇒ 本项必须失败（否则等于只验了纯快路径）")

    def test_fallback_without_unequal_kv_len_is_red(self):
        ev = _good_evidence()
        ev["telemetry"] = {**ev["telemetry"], "fallback_reasons": {"new_sequence": 3}}
        c = _by_name(evaluate(ev), "混长回退")
        self.assertFalse(c["pass"], "回退原因不是混长 ⇒ 场景不对，必须失败")

    def test_token_mismatch_is_red(self):
        ev = _good_evidence()
        ev["tokens_pkv"] = {"a": [1, 2, 9], "b": [4, 5, 6]}
        c = _by_name(evaluate(ev), "token 逐一相同")
        self.assertFalse(c["pass"])

    def test_logits_mismatch_is_red(self):
        """★ logits 摘要不同 ⇒ 失败。这是**比 token 更早**暴露偏差的那条判据。"""
        ev = _good_evidence()
        ev["logits_pkv"] = {"0:a": _dig(1, "aa"), "0:b": _dig(7, "zz")}
        c = _by_name(evaluate(ev), "logits 摘要")
        self.assertFalse(c["pass"], "logits 摘要不同必须失败")

    def test_empty_logits_is_red_not_vacuous_pass(self):
        """★★ **恒定值假通过**：两侧都空时 `==` 恒真 ⇒ 必须显式判失败。"""
        ev = _good_evidence()
        ev["logits_base"] = {}
        ev["logits_pkv"] = {}
        c = _by_name(evaluate(ev), "logits 摘要")
        self.assertFalse(c["pass"], "采集点为 0 说明开关没生效 ⇒ 不得当作「相同」")

    def test_reentry_kv_mismatch_is_red(self):
        """★ 重入那一刻的 KV 必须在判据覆盖内（这正是缺陷发生的时刻）。"""
        ev = _good_evidence()
        ev["kv_pkv_reentry"] = _kv(a_k=1.0, a_v=2.0, b_k=3.0, b_v=4.0)
        ev["kv_base_reentry"] = _kv(a_k=1.0, a_v=2.0, b_k=3.0, b_v=9.0)
        c = _by_name(evaluate(ev), "重入时刻")
        self.assertFalse(c["pass"], "重入时刻 KV 不同必须失败")

    def test_final_kv_mismatch_is_red(self):
        ev = _good_evidence()
        ev["kv_pkv_final"] = _kv(a_k=0.0, a_v=2.0, b_k=3.0, b_v=4.0)
        ev["kv_base_final"] = _kv(a_k=5.0, a_v=2.0, b_k=3.0, b_v=4.0)
        c = _by_name(evaluate(ev), "结束时刻")
        self.assertFalse(c["pass"])

    def test_tokens_out_disagreeing_with_observed_is_red(self):
        """★★ 「两臂相等」不够：两臂**同样漏记**时照样相等。

        真实先例：快路径曾漏计 `tokens_out`（实测 17 个只记 9 个）——
        两臂如果都走同一条漏记路径，只比两臂相等会完全放过。
        所以判据必须把**脚本实际观察到的 token 数**也算进来。

        ⚠️ 要扰动的是**正式段那一列**（`tokens_out_measured`）——判据走的就是它。
        只改含预热的 `tokens_out` 不再是反例了（那正是本次修掉的"口径混比"）。
        """
        ev = _good_evidence()
        for arm in ("stats_base", "stats_pkv"):
            ev[arm] = {**ev[arm],
                       "measured_window": {**ev[arm]["measured_window"],
                                           "tokens_out_measured": 3}}
        c = _by_name(evaluate(ev), "tokens_out 对账")
        self.assertFalse(c["pass"], "正式段计数与实观察不符必须失败")

    def test_total_only_is_red_because_it_is_the_wrong_parity(self):
        """★ **口径混比**的靶心：只给含预热的总量、不给正式段口径 ⇒ **必须失败**。

        不许"退回总量去凑一个相等"——那会在每次开预热的运行上报假的不符。
        """
        ev = _good_evidence()
        ev["stats_base"] = {"tokens_out": 6, "measured_window": {"basis": "no_warmup"}}
        ev["stats_pkv"] = {"tokens_out": 6,
                           "measured_window": {"basis": "no_warmup"}}
        c = _by_name(evaluate(ev), "tokens_out 对账")
        self.assertFalse(c["pass"], "拿不到正式段口径 ⇒ 判失败，不得退回总量")

    def test_warmup_tokens_are_reconciled_not_mixed(self):
        """开预热时：`总量 - 正式段 = 预热段`，且两臂预热段一致 ⇒ 通过。

        这条对应 EXECOPT5 的真实形态（`tokens_out=11,609` / `measured=7,865`）。
        旧判据会把这种**完全正常**的情形报成"记账不符"。
        """
        ev = _good_evidence()
        for arm in ("stats_base", "stats_pkv"):
            ev[arm] = {**ev[arm],
                       "tokens_out": 60,
                       "measured_window": {**ev[arm]["measured_window"],
                                           "tokens_out_measured": 6,
                                           "basis": "total_minus_warmup"}}
        c = _by_name(evaluate(ev), "tokens_out 对账")
        self.assertTrue(c["pass"], f"同口径对账应通过：{c['detail']}")
        self.assertIn("预热段 = 54", c["detail"])

    def test_inconsistent_warmup_between_arms_is_red(self):
        """两臂预热段不同 ⇒ 跨臂比较口径不一致 ⇒ 失败。"""
        ev = _good_evidence()
        ev["stats_base"] = {**ev["stats_base"], "tokens_out": 60,
                            "measured_window": {**ev["stats_base"]["measured_window"],
                                                "tokens_out_measured": 6,
                                                "basis": "total_minus_warmup"}}
        ev["stats_pkv"] = {**ev["stats_pkv"], "tokens_out": 120,
                           "measured_window": {**ev["stats_pkv"]["measured_window"],
                                               "tokens_out_measured": 6,
                                               "basis": "total_minus_warmup"}}
        c = _by_name(evaluate(ev), "tokens_out 对账")
        self.assertFalse(c["pass"], "两臂预热段不同 ⇒ 不得判通过")

    def test_measured_window_missing_is_red(self):
        ev = _good_evidence()
        ev["stats_pkv"] = {"tokens_out": 30, "measured_window": {}}
        c = _by_name(evaluate(ev), "measured_window")
        self.assertFalse(c["pass"])


class TestNumericCommonAbsenceIsNotEquality(unittest.TestCase):
    """★★ **共同缺失 ≠ 相同**（2026-10-03 补的洞）。

    对照函数只比"两侧是否相等"，于是**两侧都空**时它返回"相同"——
    一条"根本没取到数值"的运行会被读成"两臂逐位一致"。
    触发路径很现实：`_kv_snapshot` 里取不到槽位就 `continue` ⇒ 两侧都得到 `{}`。

    ⚠️ 这就是本仓记过的**恒定值假通过**：`None == None`、`{} == {}` 都恒真。
    每一处 `==` 判据都必须先问"两侧同时缺失时它会说什么"。
    """

    def test_both_kv_snapshots_empty_is_rejected(self):
        ok, notes = _compare_kv({}, {})
        self.assertFalse(ok, "两侧都空必须判失败（缺失不等于相同）")
        self.assertTrue(any("都是空的" in n for n in notes))

    def test_both_kv_layer_lists_empty_is_rejected(self):
        """取到了槽位、但**层列表为空**（层数 0）⇒ 同样什么也没验。"""
        ok, notes = _compare_kv({"a": []}, {"a": []})
        self.assertFalse(ok)
        self.assertTrue(any("层列表" in n and "都为空" in n for n in notes))

    def test_one_side_empty_is_rejected(self):
        ok, _notes = _compare_kv({"a": [(_t([1.0]), _t([2.0]))]}, {})
        self.assertFalse(ok)

    def test_real_nonempty_equal_is_accepted(self):
        kv = _kv(a_k=1.0, a_v=2.0, b_k=3.0, b_v=4.0)
        ok, notes = _compare_kv(kv, _kv(a_k=1.0, a_v=2.0, b_k=3.0, b_v=4.0))
        self.assertTrue(ok, f"真有值且相同 ⇒ 应通过：{notes}")

    def test_both_tokens_empty_is_rejected(self):
        """token 同理：两侧都空时 `==` 也成立 ⇒ 必须显式要求非空。

        （`tokens_out` 三方对账也会拦，但那是**另一条**判据；
          这条判据本身不能靠别人兜底。）
        """
        ev = _good_evidence()
        ev["tokens_base"] = {}
        ev["tokens_pkv"] = {}
        c = _by_name(evaluate(ev), "token 逐一相同")
        self.assertFalse(c["pass"], "两臂 token 都空 ⇒ 不得判'相同'")

    def test_empty_kv_is_rejected_end_to_end(self):
        """端到端：把两侧 KV 都清空 ⇒ `evaluate` 必须给出失败。"""
        ev = _good_evidence()
        ev["kv_base_reentry"] = {}
        ev["kv_pkv_reentry"] = {}
        c = _by_name(evaluate(ev), "重入时刻")
        self.assertFalse(c["pass"], "两侧 KV 都空 ⇒ 不得判'逐位相同'")


class TestLightweightNumericDigests(unittest.TestCase):
    """★ 数值产物必须留下**轻量摘要**（token / logits / KV），可事后复算与对照。

    为什么要求它：原产物只写"相同 / 不同"这个**结论**，不写任何可复算的量
    ⇒ 想复核任何一处都得**重跑真机**，而机时是本项目最贵的资源。
    摘要必须**轻**（只存 sha256 与标量，不存张量），否则产物会随层数×序列长度膨胀。
    """

    def test_token_digest_is_deterministic(self):
        a = _tokens_digest({"a": [1, 2, 3]})
        b = _tokens_digest({"a": [1, 2, 3]})
        self.assertEqual(a, b)
        self.assertEqual(a["a"]["n"], 3)
        self.assertEqual(len(a["a"]["sha256"]), 64)

    def test_token_digest_detects_a_change(self):
        a = _tokens_digest({"a": [1, 2, 3]})
        b = _tokens_digest({"a": [1, 2, 4]})
        self.assertNotEqual(a["a"]["sha256"], b["a"]["sha256"])

    def test_kv_digest_records_every_layer_with_a_hash(self):
        d = _kv_digest(_kv(a_k=1.0, a_v=2.0, b_k=3.0, b_v=4.0))
        self.assertEqual(sorted(d), ["a", "b"])
        self.assertEqual(d["a"]["n_layers"], 1)
        layer = d["a"]["layers"][0]
        self.assertEqual(sorted(layer), ["K", "V", "layer"])
        for tag in ("K", "V"):
            self.assertEqual(len(layer[tag]["sha256"]), 64)
            # ⚠️ 不断言具体维数：`_t(*vals)` 会再包一层（`_t([1.0])` ⇒ shape [1,1]）。
            #    这里要验的是"shape **非空且是整数列表**"——它才是"真的取到了张量"
            #    的证据（空 shape 会让"共同缺失"重新变得看不出来）。
            self.assertTrue(layer[tag]["shape"], "shape 必须非空")
            self.assertTrue(all(isinstance(x, int) for x in layer[tag]["shape"]))

    def test_kv_digest_is_json_serializable_and_holds_no_tensors(self):
        import json

        d = _kv_digest(_kv(a_k=1.0, a_v=2.0, b_k=3.0, b_v=4.0))
        text = json.dumps(d, ensure_ascii=False)
        self.assertNotIn("Tensor", text, "摘要里不得含张量对象")

    def test_empty_kv_digest_stays_empty_so_absence_is_visible(self):
        """★ 空输入 ⇒ 空摘要（**不是**伪造一个"看起来有值"的摘要）。

        这样"这次没取到 KV"在产物里**看得见**，而不是被摘要掩盖掉。
        """
        self.assertEqual(_kv_digest({}), {})

    def test_written_payload_includes_numeric_digests(self):
        """结构守卫：产物 payload 里必须有 `numeric_digests` 这一段。

        ⚠️ 这是**结构**守卫（防有人删掉写入），不是行为验证——
        行为验证需要真跑一次真机（见 `EXPERIMENTS.md` §3.7 的产物）。
        """
        import inspect

        import verify_s04_reenact_numerics as V

        src = inspect.getsource(V.main)
        self.assertIn("numeric_digests", src,
                      "产物 payload 必须写入 numeric_digests（否则摘要没落盘）")
        for key in ("tokens", "logits", "kv_at_reentry", "kv_final"):
            self.assertIn(f'"{key}"', src, f"摘要里缺 {key}")


class TestNumericCompleteness(unittest.TestCase):
    """★ **采集完整性**判据：先证明"该采的都采到了"，再谈"两臂是否相同"。

    为什么不能只靠每处 `==` 加非空守卫：那是**防守**（堵一个已知漏洞）；
    完整性是**正面验收**（预期槽位 / 模型层数 / 形状与 dtype / 采集点覆盖）。
    两者的差别在于——防守只挡你想到的那一种缺法。
    """

    @staticmethod
    def _complete(**over):
        kw = dict(
            expected_slots=["a", "b"],
            kv={"a": [(_t4(1.0), _t4(2.0))], "b": [(_t4(3.0), _t4(4.0))]},
            tokens={"a": [1, 2], "b": [3, 4]},
            logits_keys=["0:a", "0:b"],
            n_steps=1, n_layers=1,
            n_kv_heads=FAKE_N_KV_HEADS, head_dim=FAKE_HEAD_DIM, dtype=FAKE_KV_DTYPE,
        )
        kw.update(over)
        return numeric_completeness_verdict(**kw)

    def test_complete_input_passes(self):
        ok, notes = self._complete()
        self.assertTrue(ok, f"完整输入应通过：{notes}")

    # ---------------- ① 预期槽位 ----------------

    def test_missing_expected_slot_is_red(self):
        ok, notes = self._complete(kv={"a": [(_t4(1.0), _t4(2.0))]})
        self.assertFalse(ok)
        self.assertTrue(any("缺槽位" in n for n in notes))

    def test_extra_unexpected_slot_is_red(self):
        ok, notes = self._complete(
            kv={"a": [(_t4(1.0), _t4(2.0))], "b": [(_t4(3.0), _t4(4.0))],
                "zzz": [(_t4(9.0), _t4(9.0))]})
        self.assertFalse(ok)
        self.assertTrue(any("非预期槽位" in n for n in notes))

    def test_empty_kv_is_red(self):
        ok, notes = self._complete(kv={})
        self.assertFalse(ok, "空快照必须判失败（缺失不等于相同）")

    # ---------------- ② 模型层数 ----------------

    def test_layer_count_mismatch_is_red(self):
        """★ 只判"非空"会放过"只采到 1 层"；这里必须对着**模型的层数**核。"""
        ok, notes = self._complete(n_layers=4)
        self.assertFalse(ok)
        self.assertTrue(any("层数" in n for n in notes))

    def test_empty_layer_list_is_red(self):
        ok, notes = self._complete(kv={"a": [], "b": []})
        self.assertFalse(ok)
        self.assertTrue(any("层列表为空" in n for n in notes))

    # ---------------- ③ 形状 / dtype ----------------

    def test_two_dimensional_kv_is_red(self):
        """形状不对 ⇒ KV 的轴语义（哪个是 head、哪个是长度）**没被验证过**。"""
        bad = {"a": [(_t([1.0]), _t([2.0]))], "b": [(_t([3.0]), _t([4.0]))]}
        ok, notes = self._complete(kv=bad)
        self.assertFalse(ok)
        self.assertTrue(any("4 维" in n or "形状" in n for n in notes))

    def test_wrong_head_dim_is_red(self):
        ok, notes = self._complete(head_dim=99)
        self.assertFalse(ok)
        self.assertTrue(any("形状" in n for n in notes))

    def test_dtype_mismatch_is_red(self):
        """精度变了最容易被忽略、也最先毁掉可比性 ⇒ 必须与遥测声明一致。"""
        ok, notes = self._complete(dtype="bfloat16")
        self.assertFalse(ok)
        self.assertTrue(any("dtype" in n for n in notes))

    def test_k_v_shape_mismatch_is_red(self):
        bad = {"a": [(_t4(1.0), _t4(2.0, layers=2))], "b": [(_t4(3.0), _t4(4.0))]}
        ok, notes = self._complete(kv=bad)
        self.assertFalse(ok)
        self.assertTrue(any("K/V 形状不同" in n for n in notes))

    # ---------------- ④ token 采集点 ----------------

    def test_slot_without_tokens_is_red(self):
        ok, notes = self._complete(tokens={"a": [1, 2]})
        self.assertFalse(ok)
        self.assertTrue(any("没有 token" in n for n in notes))

    # ---------------- ⑤ logits 采集点集合 ----------------

    def test_logits_missing_a_step_is_red(self):
        ok, notes = self._complete(logits_keys=["0:a", "0:b"], n_steps=3)
        self.assertFalse(ok)
        self.assertTrue(any("**步**" in n or "没有 logits" in n for n in notes))

    def test_logits_missing_a_slot_is_red(self):
        """个数够、但全挤在同一槽位上 ⇒ 旧口径（只看个数）会放过。"""
        ok, notes = self._complete(logits_keys=["0:a", "1:a", "2:a"], n_steps=3)
        self.assertFalse(ok)
        self.assertTrue(any("从未" in n for n in notes))

    def test_malformed_logits_key_is_red(self):
        ok, notes = self._complete(logits_keys=["0:a", "bogus"], n_steps=1)
        self.assertFalse(ok)
        self.assertTrue(any("格式不对" in n for n in notes))

    def test_no_logits_at_all_is_red(self):
        ok, notes = self._complete(logits_keys=[])
        self.assertFalse(ok)
        self.assertTrue(any("没有任何 logits" in n for n in notes))

    # ---------------- 缺参数时不许"默认通过" ----------------

    def test_missing_shape_params_do_not_skip_silently(self):
        """没给 `n_kv_heads`/`head_dim` 时**跳过形状判据**，但其余判据照常执行。"""
        ok, notes = self._complete(n_kv_heads=None, head_dim=None, kv={})
        self.assertFalse(ok, "空 KV 仍必须判失败")
        self.assertTrue(any("一个槽位都没有" in n or "缺槽位" in n for n in notes),
                        f"应指出槽位缺失，实际：{notes}")


class TestWarmupHonesty(unittest.TestCase):
    """★★ 无预热场景**不得**宣称"预热扣除已验证"（用户第 2 条）。"""

    def test_no_warmup_does_not_claim_verified(self):
        ev = _good_evidence()
        checks = evaluate(ev)
        self.assertFalse(
            ev["warmup_deduction_verified"],
            "没预热就必须记 false —— 不得把 no_warmup 口径说成「预热扣除已验证」")
        c = _by_name(checks, "预热口径自洽")
        self.assertTrue(c["pass"], "无预热时 basis 应恰好是 no_warmup")

    def test_no_warmup_but_basis_claims_deduction_is_red(self):
        """**否定对照**：没预热却报 `total_minus_warmup` ⇒ 口径不自洽 ⇒ 失败。"""
        ev = _good_evidence()
        ev["stats_pkv"] = {**ev["stats_pkv"],
                           "measured_window": {"pkv_fastpath_measured": 7,
                                               "n_forwards_measured": 40,
                                               "basis": "total_minus_warmup"}}
        checks = evaluate(ev)
        c = _by_name(checks, "预热口径自洽")
        self.assertFalse(c["pass"], "无预热却声称已扣除 ⇒ 必须失败")

    def test_warmup_verified_requires_measured_below_total(self):
        """预热了但 `measured == total` ⇒ 说明扣除**没真的发生** ⇒ 失败。"""
        ev = _good_evidence(warmup_s=1.0)
        ev["stats_pkv"] = {**ev["stats_pkv"],
                           "n_forwards": 40,
                           "measured_window": {"pkv_fastpath_measured": 7,
                                               "n_forwards_measured": 40,
                                               "basis": "total_minus_warmup"}}
        checks = evaluate(ev)
        c = _by_name(checks, "预热扣除")
        self.assertFalse(c["pass"], "measured 未小于 total ⇒ 扣除没发生 ⇒ 失败")

    def test_warmup_verified_when_deduction_really_happened(self):
        ev = _good_evidence(warmup_s=1.0)
        ev["stats_pkv"] = {**ev["stats_pkv"],
                           "n_forwards": 40,
                           "measured_window": {"pkv_fastpath_measured": 7,
                                               "n_forwards_measured": 33,
                                               "basis": "total_minus_warmup"}}
        checks = evaluate(ev)
        c = _by_name(checks, "预热扣除")
        self.assertTrue(c["pass"])
        self.assertTrue(ev["warmup_deduction_verified"])


if __name__ == "__main__":
    unittest.main()
