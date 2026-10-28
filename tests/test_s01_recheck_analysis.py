"""S01 复核分析器的**针对性反例**测试。

★ 为什么必须写反例
----------------
分析器的最坏失效不是"崩了"，而是**在输入不成立时照样给出 SUPPORTED**。
所以这里每一条都必须能**失败**——用 `assert` 造出"坏输入"，
确认分析器**拒绝**输出 SUPPORTED/REGRESSED。破坏性对照见文件末尾。

覆盖（对应用户点名的五类反例）：
    ① 臂反转（对照/处理搞反）
    ② 缺一轮
    ③ 重复轮次
    ④ 非 MEASURED
    ⑤ 吞吐方向
外加：配置相同（单变量前提不成立）、跨臂身份不一致、有限数值缺失。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "analyze_s01_recheck.py"

sys.path.insert(0, str(_ROOT / "tests"))
import _tiny_support as _T  # noqa: E402  （本机必测 / 外部依赖的分工）

#: 端到端用例依赖的**远端产物**（`results/raw/**`，需回拉）。
#: ⚠️ 它们是**外部依赖**：缺失时合法 skip，但必须用 `ExternalArtifactMissing`
#:    （带明确理由），**不得**用裸 `skipTest("...")` —— 那让"没跑"看起来像"通过"。
_END_TO_END_ARTIFACT = _ROOT / "results" / "raw" / "S01RECHECK-short_chat-s01_both"


def _skip_if_remote_artifact_absent(have: bool) -> None:
    """远端产物缺失 ⇒ 合法 skip（**外部依赖**）；理由是显式类型，不是一句安慰话。"""
    if not have:
        raise _T.ExternalArtifactMissing(
            f"远端产物未回拉：{_END_TO_END_ARTIFACT} 派生目录不在本机 ⇒ "
            "端到端链路无法核验（离线单测已覆盖判据逻辑）")



def _load_mod():
    spec = importlib.util.spec_from_file_location("_ana_s01", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ana_s01"] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load_mod()


def _round(rnd: int, *, ttft=190.0, thr=64.0, sat=2, n=64, label="MEASURED",
           tree="472697170d43", machine="autodl-container-05264bbf0d",
           rev="a09a35458c702b33eeacc393d103063234e8bc28", cfg="cfgA",
           name=None) -> dict:
    """造一份 `read_round` 形状的记录。"""
    return {
        "dir": name or f"T-w-{cfg}-r{rnd}", "round": rnd,
        "data_label": label, "tree_digest": tree, "git_commit": "deadbeef",
        "git_source": "stamp", "machine": machine, "config_digest": cfg,
        "model_revision": rev,
        "ttft_p50": ttft, "ttft_p95": 914.0, "ttft_p99": 1279.0,
        "tpot_p50": 117.0, "tpot_p90": 121.0,
        "throughput_tok_s": thr, "slo_goodput_tok_s": 1.9,
        "slo_satisfaction_rate": sat / n, "n_slo_satisfied": float(sat),
        "n_requests": float(n), "n_success": float(n), "n_failed": 0.0,
        "n_rejected": 0.0, "n_unfinished": 0.0,
        "wall_ms": 122000.0, "fwd_pack_ms": 121000.0,
        "fwd_ms": 99000.0, "pack_ms": 22000.0, "n_steps": 1080,
        "n_long_fwd": 1, "n_long_pack": 0,
    }


def _pair(metric: str, c_val: float, t_val: float) -> dict:
    row = {"round": 0}
    row[f"{metric}__c"] = c_val
    row[f"{metric}__t"] = t_val
    row[f"{metric}__ratio"] = (t_val / c_val) if c_val else None
    row[f"{metric}__delta"] = t_val - c_val
    return row


class TestPairingGate(unittest.TestCase):
    """★★ 配对硬闸门：只要不成立，就不许下结论。"""

    def test_exactly_five_pairs_pass(self):
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(5)]
        ok, probs = A.check_pairing(c, t, expect_rounds=5)
        self.assertTrue(ok, probs)

    def test_missing_one_round_is_rejected(self):
        """② 缺一轮：对照有 r0..r4，处理只有 r0..r3。"""
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(4)]
        ok, probs = A.check_pairing(c, t, expect_rounds=5)
        self.assertFalse(ok)
        self.assertTrue(any("轮次集合不同" in p for p in probs), probs)
        self.assertTrue(any("可比轮次数" in p for p in probs), probs)

    def test_duplicate_round_is_rejected(self):
        """③ 重复轮次：`-r1` 出现两次（用 dict 去重会**静默丢掉一份产物**）。"""
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(5)]
        t.append(_round(1, cfg="cfgB", name="T-w-cfgB-r1-dup"))
        ok, probs = A.check_pairing(c, t, expect_rounds=5)
        self.assertFalse(ok)
        self.assertTrue(any("重复" in p for p in probs), probs)

    def test_four_pairs_is_rejected(self):
        c = [_round(i, cfg="cfgA") for i in range(4)]
        t = [_round(i, cfg="cfgB") for i in range(4)]
        ok, probs = A.check_pairing(c, t, expect_rounds=5)
        self.assertFalse(ok)
        self.assertTrue(any("可比轮次数" in p for p in probs), probs)


class TestIdentityGate(unittest.TestCase):
    """身份 / 标签 / 配置差异 / 有限数值。"""

    def test_clean_pair_passes(self):
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(5)]
        ok, probs = A.check_identity(c, t)
        self.assertTrue(ok, probs)

    def test_non_measured_is_rejected(self):
        """④ 非 MEASURED：一行 `NOT_MEASURED` 就必须拦住结论。"""
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(5)]
        t[2]["data_label"] = "NOT_MEASURED"
        ok, probs = A.check_identity(c, t)
        self.assertFalse(ok)
        self.assertTrue(any("非 MEASURED" in p for p in probs), probs)

    def test_same_config_is_rejected(self):
        """两臂跑**同一份配置** ⇒ 单变量前提不成立。"""
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgA") for i in range(5)]
        ok, probs = A.check_identity(c, t)
        self.assertFalse(ok)
        self.assertTrue(any("单变量前提不成立" in p for p in probs), probs)

    def test_cross_arm_tree_mismatch_is_rejected(self):
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(5)]
        for r in t:
            r["tree_digest"] = "OTHERDIGEST00"
        ok, probs = A.check_identity(c, t)
        self.assertFalse(ok)
        self.assertTrue(any("跨臂 tree_digest 不一致" in p for p in probs), probs)

    def test_machine_mismatch_is_rejected(self):
        """环境身份错位（A7 那类）：两臂跑在不同容器上。"""
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(5)]
        for r in t:
            r["machine"] = "autodl-container-cf99469eb6"
        ok, probs = A.check_identity(c, t)
        self.assertFalse(ok)
        self.assertTrue(any("跨臂 machine 不一致" in p for p in probs), probs)

    def test_missing_digest_is_rejected(self):
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(5)]
        c[0]["config_digest"] = None
        ok, probs = A.check_identity(c, t)
        self.assertFalse(ok)


class TestDirection(unittest.TestCase):
    """⑤ 吞吐方向：吞吐**越大越好**；方向搞反会把改善报成退化。"""

    def test_throughput_higher_is_better(self):
        self.assertIn("throughput_tok_s", A.HIGHER_IS_BETTER)
        rows = [_pair("throughput_tok_s", 64.0, 64.5)]
        s = A.summarize_pair(rows, "throughput_tok_s")
        self.assertTrue(s["higher_is_better"])
        # 处理更高 ⇒ 应计为"更好"，而不是旧版那样的"更快 0/1"
        self.assertEqual(s["n_better"], 1)
        self.assertEqual(s["n_worse"], 0)

    def test_throughput_lower_is_worse(self):
        rows = [_pair("throughput_tok_s", 64.0, 63.0)]
        s = A.summarize_pair(rows, "throughput_tok_s")
        self.assertEqual(s["n_better"], 0)
        self.assertEqual(s["n_worse"], 1)

    def test_latency_lower_is_better(self):
        for m in ("ttft_p50", "tpot_p50", "wall_ms", "fwd_pack_ms"):
            self.assertNotIn(m, A.HIGHER_IS_BETTER)
            rows = [_pair(m, 190.0, 180.0)]
            s = A.summarize_pair(rows, m)
            self.assertEqual(s["n_better"], 1, m)

    def test_verdict_uses_direction_for_latency_only(self):
        """吞吐不该进主指标判定；主指标是时延且越小越好。"""
        self.assertEqual(A.PREREG["primary_metric"], "ttft.p50")
        rows = [_pair("ttft_p50", 190.0, 180.0)]
        s = {"ttft_p50": A.summarize_pair(rows, "ttft_p50")}
        v = A.verdict(s, gates_ok=True)
        self.assertEqual(v["state"], "SUPPORTED", v)


class TestRatioNotAbsoluteRate(unittest.TestCase):
    """★★ 比值 ≠ 绝对率（上一版把 `slo_satisfaction_rate` 的比值读成 1.0000 的根因）。"""

    def test_ratio_of_equal_rates_is_one_but_rate_is_not(self):
        c = _round(0, cfg="cfgA", sat=2, n=64)
        t = _round(0, cfg="cfgB", sat=2, n=64)
        self.assertAlmostEqual(c["slo_satisfaction_rate"], 2 / 64)
        rows = [_pair("slo_satisfaction_rate", c["slo_satisfaction_rate"],
                      t["slo_satisfaction_rate"])]
        s = A.summarize_pair(rows, "slo_satisfaction_rate")
        # 比值 1.0 只说明"两臂一样"，**不说**达标率是 100%
        self.assertAlmostEqual(s["ratio_median"], 1.0)
        self.assertAlmostEqual(s["control_abs_median"], 2 / 64)
        self.assertAlmostEqual(s["treatment_abs_median"], 2 / 64)
        self.assertEqual(c["n_requests"], 64)
        self.assertEqual(c["n_slo_satisfied"], 2)

    def test_rate_is_recomputed_from_integers(self):
        """达标率必须由 `n_slo_satisfied / n_requests` 重算（分母显式可查）。"""
        r = _round(0, sat=7, n=64)
        self.assertAlmostEqual(r["slo_satisfaction_rate"], 7 / 64)


class TestZeroCountsAreResults(unittest.TestCase):
    """计数类 0 vs 0 是**结果**，不是"没测到"。"""

    def test_all_zero_delta_flagged(self):
        rows = [_pair("n_failed", 0.0, 0.0) for _ in range(5)]
        s = A.summarize_pair(rows, "n_failed")
        self.assertTrue(s["all_zero_delta"])
        self.assertIsNone(s["ratio_median"])

    def test_nan_is_not_a_finite_value(self):
        rows = [_pair("ttft_p50", 190.0, 180.0)]
        rows[0]["ttft_p50__ratio"] = float("nan")
        # summarize_pair 只收非 None；NaN 由 paired_rows 的守卫负责剔除
        c = [_round(0, ttft=float("nan"))]
        t = [_round(0, cfg="cfgB", ttft=180.0)]
        paired = A.paired_rows(c, t, ["ttft_p50"])
        self.assertIsNone(paired[0]["ttft_p50__ratio"])


class TestGatesBlockVerdict(unittest.TestCase):
    """★ 闸门不过 ⇒ 无论数字多漂亮，都不得 SUPPORTED。"""

    def test_gates_false_forces_unknown(self):
        rows = [_pair("ttft_p50", 190.0, 180.0) for _ in range(5)]
        s = {"ttft_p50": A.summarize_pair(rows, "ttft_p50")}
        v = A.verdict(s, gates_ok=False)
        self.assertEqual(v["state"], "UNKNOWN")
        self.assertIn("闸门", v["why"])

    def test_arm_reversal_changes_verdict_direction(self):
        """① 臂反转：同样两份产物，角色互换后结论方向相反。

        ⇒ 这正是"必须显式指定 --control"的理由：若默认取 sorted 第一个，
        两臂目录名一变就可能**静默反转**结论。
        """
        rows_fwd = [_pair("ttft_p50", 190.0, 180.0) for _ in range(5)]
        rows_rev = [_pair("ttft_p50", 180.0, 190.0) for _ in range(5)]
        vf = A.verdict({"ttft_p50": A.summarize_pair(rows_fwd, "ttft_p50")}, gates_ok=True)
        vr = A.verdict({"ttft_p50": A.summarize_pair(rows_rev, "ttft_p50")}, gates_ok=True)
        self.assertEqual(vf["state"], "SUPPORTED")
        self.assertEqual(vr["state"], "REGRESSED")

    def test_control_arg_is_mandatory(self):
        """`--control` 无默认值（`action` 未给 default）⇒ 不指定就报错退出。"""
        src = _SCRIPT.read_text(encoding="utf-8")
        # --control 的 help 明确写了"防臂反转"，且 argparse 里 default=None
        self.assertIn("--control", src)
        self.assertIn("防臂反转", src)


class TestJsonWriteIsSelfIdentifying(unittest.TestCase):
    """★★ **旧 JSON 不得冒充新结果** —— 这条来自一次真实误判。

    2026-10-03 实况：我跑了同配置的否定对照（`--control s01_both --treatment s01_both`），
    控制台正确打出 `UNKNOWN`，但我紧接着读 `results/processed/s01_recheck_analysis.json`
    却看到 `SUPPORTED` / `gates_ok=True` ⇒ **我一度判定"闸门没传播"**。
    真相：**不加 `--json` 时脚本根本不写文件**，磁盘上那份是**上一次**（好输入）留下的。

    ⇒ 危险方向**恰好相反**于直觉：不是"闸门失效"，而是"**旧产物被当成新结论**"。
    （与用户长期记录同类：判定"文件没生成/没更新"前，先怀疑它是旧的。）

    本类把两件事钉住：
      ① 不开 `--json` ⇒ **不写文件**，且**必须显式提示**磁盘上的是旧产物；
      ② 开了 `--json` ⇒ 产物带 `written_at` + `argv`，可自证"是哪次调用写的"。
    """

    TAG = "S01RECHECK"
    CTRL, TREAT = "s01_reference", "s01_both"

    def _run(self, *extra: str):
        import subprocess
        return subprocess.run(
            [sys.executable, str(_SCRIPT), "--tag", self.TAG, *extra],
            capture_output=True, text=True, cwd=str(_ROOT))

    def _has_artifacts(self) -> bool:
        d = _ROOT / "results" / "raw" / f"{self.TAG}-short_chat-{self.TREAT}-r1"
        return d.is_dir()

    def test_without_json_flag_no_file_is_written(self):
        """★ 不开 `--json` ⇒ 目标文件**不产生**（证明它没被写过）。"""
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "out.json"
            p = self._run("--control", self.CTRL, "--treatment", self.TREAT,
                          "--json", str(target))
            self.assertEqual(p.returncode, 0, p.stderr)
            # 控制组：传了 --json 就有文件
            self.assertTrue(target.exists())
            # 实验组：不传 --json 就没有文件
            target2 = Path(td) / "out2.json"
            p2 = self._run("--control", self.CTRL, "--treatment", self.TREAT)
            self.assertEqual(p2.returncode, 0, p2.stderr)
            self.assertFalse(target2.exists(), "未传 --json 却写了文件？")

    def test_without_json_flag_warns_about_stale_file(self):
        """★ 不写文件时**必须**在 stdout 提示"磁盘上的是旧产物"。"""
        _skip_if_remote_artifact_absent(self._has_artifacts())
        p = self._run("--control", self.CTRL, "--treatment", self.TREAT)
        self.assertIn("未加 --json", p.stdout,
                      "必须显式提示：本次未写文件、磁盘上的是旧产物")
        self.assertIn("与本次运行无关", p.stdout)

    def test_json_payload_carries_written_at_and_argv(self):
        """★ 产物自证身份：`written_at` + `argv` 缺一不可。"""
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "out.json"
            p = self._run("--control", self.CTRL, "--treatment", self.TREAT,
                          "--json", str(target))
            self.assertEqual(p.returncode, 0, p.stderr)
            d = json.loads(target.read_text(encoding="utf-8"))
            self.assertIn("written_at", d, "产物必须带写入时间戳")
            self.assertIn("argv", d, "产物必须带本次调用参数")
            self.assertIn("--control", d["argv"])
            self.assertIn(self.CTRL, d["argv"])

    def test_gate_failure_does_propagate_into_written_json(self):
        """★★ 否定对照：同配置两臂 **且传 `--json`** ⇒ 落盘结论必须是 `UNKNOWN`。

        这条正是我误判的那一步 —— 加上 `--json` 后，闸门失败**确实**传播到产物：
        `gates_ok=False`、`identity_ok=False`、`verdict.state="UNKNOWN"`。
        """
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "neg.json"
            # 同配置 ⇒ 身份闸门必须拦下
            p = self._run("--control", self.TREAT, "--treatment", self.TREAT,
                          "--json", str(target))
            self.assertEqual(p.returncode, 1, "闸门不过 ⇒ 退出码必须非 0")
            d = json.loads(target.read_text(encoding="utf-8"))
            self.assertFalse(d["gates_ok"])
            self.assertFalse(d["identity_ok"])
            self.assertEqual(d["verdict"]["state"], "UNKNOWN")
            self.assertTrue(d["identity_problems"], "必须写出具体问题，不能只给一个布尔")

    def test_written_json_is_lf_only(self):
        """产物必须 LF-only（本仓 `eol=lf` 约定，CRLF 是静默偏差）。"""
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "lf.json"
            self._run("--control", self.CTRL, "--treatment", self.TREAT,
                      "--json", str(target))
            self.assertEqual(target.read_bytes().count(b"\r"), 0, "JSON 含 CR")


class TestValidNumericGate(unittest.TestCase):
    """★★ **有效数值门**：主指标必须**恰好 5 对有限且合法**。

    用户点名的四类真实 `main` 入口反例：**四轮缺失 / Infinity / NaN / 零分母**。
    共同纪律：不是"打印个警告"，而是 —— 结论 = `UNKNOWN`、**退出码非 0**、
    且**必须写出具体原因**（只给一个布尔等于没给信息）。

    ⚠️ 为什么需要这道门（而不是让 NaN 自己"融入"统计）：JSON 允许 `Infinity`/`NaN`
    字面量；它们**不会**让 `statistics.median` 报错，却会让整批判定翻面：

        * `float("inf")` 参与比较时**恒为真** ⇒ 会被数成"更好/更快"；
        * `float("nan")` 恒为假 ⇒ 从所有方向统计里**静默消失**，行数看起来还够。
    ⇒ 必须在入口按**有限数**过滤，并要求"有效对数 == 期望对数"。
    """

    EXPECT = 5

    #: 主指标实例键（`ttft.p50` → `ttft_p50`），由模块自身导出，避免硬编码走样
    @property
    def M(self) -> str:
        return A.PREREG["primary_metric"].replace(".", "_")

    # ---- 门本身（纯函数层）----
    def _summary_with(self, pairs: list[tuple[float, float]]) -> dict:
        """用 **(对照值, 处理值)** 列表造一份 `summary`（其余指标留空）。"""
        m = self.M
        rows = []
        for c, t in pairs:
            row = {f"{m}__c": c, f"{m}__t": t}
            row[f"{m}__ratio"] = (t / c) if (c and A._finite(c) and A._finite(t)) else None
            row[f"{m}__delta"] = (t - c) if (A._finite(c) and A._finite(t)) else None
            rows.append(row)
        return {m: A.summarize_pair(rows, m)}

    def test_exactly_five_finite_pairs_pass(self):
        ok, prob = A.check_valid_numeric(
            self._summary_with([(190.0, 183.0)] * 5), expect_pairs=self.EXPECT)
        self.assertTrue(ok, prob)
        self.assertEqual(prob, [])

    def test_four_pairs_is_rejected(self):
        """★ 反例①：只有四轮 ⇒ 有效对数 4 ≠ 5。"""
        ok, prob = A.check_valid_numeric(
            self._summary_with([(190.0, 183.0)] * 4), expect_pairs=self.EXPECT)
        self.assertFalse(ok, "四轮竟然通过了有效数值门")
        self.assertTrue(any("有效配对" in p or "有效" in p for p in prob),
                        f"必须说明是哪种不足：{prob}")

    def test_infinity_is_rejected(self):
        """★ 反例②：`Infinity` 是合法 JSON、且恒为真 ⇒ 必须被拦。"""
        pairs = [(190.0, 183.0)] * 4 + [(190.0, float("inf"))]
        ok, prob = A.check_valid_numeric(
            self._summary_with(pairs), expect_pairs=self.EXPECT)
        self.assertFalse(ok, "Infinity 竟然通过了有效数值门")

    def test_nan_is_rejected(self):
        """★ 反例③：`NaN` 恒为假 ⇒ 会从所有方向统计里静默消失，必须被拦。"""
        pairs = [(190.0, 183.0)] * 4 + [(190.0, float("nan"))]
        ok, prob = A.check_valid_numeric(
            self._summary_with(pairs), expect_pairs=self.EXPECT)
        self.assertFalse(ok, "NaN 竟然通过了有效数值门")

    def test_zero_denominator_is_rejected(self):
        """★ 反例④：对照值为 0 ⇒ 比值无定义（`0/0` / `x/0`），必须被拦。"""
        pairs = [(190.0, 183.0)] * 4 + [(0.0, 0.0)]
        ok, prob = A.check_valid_numeric(
            self._summary_with(pairs), expect_pairs=self.EXPECT)
        self.assertFalse(ok, "零分母竟然通过了有效数值门")
        self.assertTrue(any("比值" in p for p in prob), f"必须点名比值非法：{prob}")

    def test_none_value_is_rejected(self):
        """缺值（`None`）与"值恰好是 0"必须分开：前者不可用，后者是真实结果。"""
        m = self.M
        rows = [{f"{m}__c": 190.0, f"{m}__t": 183.0,
                 f"{m}__ratio": 183.0 / 190.0, f"{m}__delta": -7.0}
                for _ in range(4)]
        rows.append({f"{m}__c": 190.0, f"{m}__t": None,
                     f"{m}__ratio": None, f"{m}__delta": None})
        ok, prob = A.check_valid_numeric({m: A.summarize_pair(rows, m)},
                                         expect_pairs=self.EXPECT)
        self.assertFalse(ok, "缺值的对竟然算进了有效配对")

    # ---- 结论层：门不过 ⇒ 一律 UNKNOWN ----
    def test_verdict_is_unknown_when_numeric_gate_fails(self):
        """★ 门不过时 `verdict()` 必须返回 UNKNOWN，且原因点名"有效数值门"。"""
        s = self._summary_with([(190.0, 150.0)] * 5)  # 幅度大到本该 SUPPORTED
        v = A.verdict(s, gates_ok=True, valid_ok=False,
                      valid_problems=["只有 4 对"])
        self.assertEqual(v["state"], "UNKNOWN", "有效数值门未过却给了结论")
        self.assertIn("有效数值门", v["why"])
        self.assertIn("只有 4 对", v["why"], "必须把具体原因带进 why")

    def test_verdict_still_supported_when_gate_passes(self):
        """控制：门通过时**不得**被这道门卡住（否则等于把好输入也毙了）。"""
        s = self._summary_with([(190.0, 150.0)] * 5)
        ok, _ = A.check_valid_numeric(s, expect_pairs=self.EXPECT)
        self.assertTrue(ok)
        v = A.verdict(s, gates_ok=True, valid_ok=ok)
        self.assertEqual(v["state"], "SUPPORTED")

    def test_base_and_numeric_gate_reasons_are_distinct(self):
        """两条原因必须**分开成句**：一个说去补产物，一个说去查数值。"""
        v1 = A.verdict({}, gates_ok=False, valid_ok=True)
        self.assertEqual(v1["state"], "UNKNOWN")
        self.assertIn("完整性", v1["why"])
        self.assertNotIn("有效数值门", v1["why"])

        v2 = A.verdict({}, gates_ok=True, valid_ok=False,
                       valid_problems=["x"])
        self.assertEqual(v2["state"], "UNKNOWN")
        self.assertIn("有效数值门", v2["why"])


class TestValidNumericGateAtMainEntry(unittest.TestCase):
    """★★ 四类反例走**真实 `main()` 入口**（不是只调纯函数）。

    做法：把既有 S01RECHECK 产物复制到临时根，再用**逐字节改写**的方式注入坏值，
    然后跑真 `main`。这样验的是"端到端"：读取 → 配对 → 汇总 → 门 → 退出码。
    ⚠️ 复制是必须的 —— `results/raw/**` 只读，绝不原地改。
    """

    TAG = "S01RECHECK"
    CTRL, TREAT = "s01_reference", "s01_both"
    SRC = {"s01_reference": "S01RECHECK-short_chat-s01_reference",
           "s01_both": "S01RECHECK-short_chat-s01_both"}
    #: ⚠️ 主指标的**真实字段路径**是 `meta.json` → `summary.ttft.p50`
    #: （`read_round` 就是这么读的）。**不是** `summary.json` —— 本目录里根本没有
    #: 那个文件；按记忆写路径会造出一个"改了个不存在的键"的假反例。
    METRIC_PATH = ("summary", "ttft", "p50")

    #: ⚠️ 轮次目录名的**真实形态**：不带后缀 = 第 0 轮，然后 `-r1` … `-r4`。
    #: **没有 `-r5`** —— 我第一版按 `r1..r5` 去探，于是 5 轮全被漏掉、
    #: 6 条端到端反例**静默跳过**（`skipTest` 让"没跑"看起来像"通过"）。
    #: 教训与仓库长期约定一致：**判定"产物不存在"之前，先把真实名字打出来看**。
    RUNS = ("", "-r1", "-r2", "-r3", "-r4")

    #: 要注入坏值的**目标轮目录后缀**（默认动第 0 轮；`-r4` 是与"缺一轮"配套用的）
    def _run_dirs(self, arm: str) -> list[Path]:
        base = _ROOT / "results" / "raw"
        return [base / f"{self.SRC[arm]}{suf}" for suf in self.RUNS]

    def _has_artifacts(self) -> bool:
        for arm in (self.CTRL, self.TREAT):
            for d in self._run_dirs(arm):
                if not (d / "meta.json").is_file():
                    return False
        return True

    def _inject(self, run_dir: Path, value) -> None:
        """把 `meta.json` 的主指标改成一个**坏值**（Inf/NaN/0）。"""
        import json
        mp = run_dir / "meta.json"
        d = json.loads(mp.read_text(encoding="utf-8"))
        node = d
        for k in self.METRIC_PATH[:-1]:
            node = node[k]
        node[self.METRIC_PATH[-1]] = value
        # `json.dumps` 默认把 inf/nan 写成 `Infinity`/`NaN`（合法 JSON 扩展）——
        # 这正是我们要复现的输入形态。
        mp.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")

    def _stage(self, td: str, *, round1_treat: Any = "__UNSET__",
               round1_ctrl: Any = "__UNSET__",
               drop_round: str | None = None) -> Path:
        """把 10 份产物复制到临时 root；可选注入坏值。

        ⚠️ 只改**处理臂 r1**（或按需改对照 r1），保证"其余 4 对仍然合法"，
        从而能把失败**精确归因**到"有效对数 ≠ 5"，而不是"全都没了"。
        """
        import shutil
        root = Path(td) / "raw"
        for arm in (self.CTRL, self.TREAT):
            for suf in self.RUNS:
                src = _ROOT / "results" / "raw" / f"{self.SRC[arm]}{suf}"
                dst = root / f"{self.SRC[arm]}{suf}"
                shutil.copytree(src, dst)
                if arm == self.TREAT and suf == "-r1" and round1_treat != "__UNSET__":
                    self._inject(dst, round1_treat)
                if arm == self.CTRL and suf == "-r1" and round1_ctrl != "__UNSET__":
                    self._inject(dst, round1_ctrl)
        if drop_round is not None:
            shutil.rmtree(root / f"{self.SRC[self.TREAT]}{drop_round}")
        return root

    def _run(self, root: Path, *extra: str):
        import subprocess
        return subprocess.run(
            [sys.executable, str(_SCRIPT), "--tag", self.TAG,
             "--root", str(root),
             "--control", self.CTRL, "--treatment", self.TREAT, *extra],
            capture_output=True, text=True, cwd=str(_ROOT))

    def _assert_unknown_nonzero(self, p, *, who: str):
        self.assertNotEqual(p.returncode, 0, f"{who}：门不过却返回了 0")
        self.assertIn("UNKNOWN", p.stdout, f"{who}：结论必须打成 UNKNOWN")
        self.assertNotIn("★ 结论 = **SUPPORTED**", p.stdout, f"{who}：竟然给了 SUPPORTED")

    def test_baseline_copy_still_passes(self):
        """控制：**不改任何值**的复制件必须仍然通过（否则后面的反例没有对照）。"""
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = self._stage(td)
            p = self._run(root)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("③b 有效数值门", p.stdout)
            self.assertIn("有效配对 5", p.stdout)

    def test_main_entry_rejects_infinity(self):
        """★ 反例 Infinity：真实 `main` 必须 UNKNOWN + 非 0 退出。"""
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = self._stage(td, round1_treat=float("inf"))
            p = self._run(root)
            self._assert_unknown_nonzero(p, who="Infinity")

    def test_main_entry_rejects_nan(self):
        """★ 反例 NaN：真实 `main` 必须 UNKNOWN + 非 0 退出。"""
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = self._stage(td, round1_treat=float("nan"))
            p = self._run(root)
            self._assert_unknown_nonzero(p, who="NaN")

    def test_main_entry_rejects_zero_denominator(self):
        """★ 反例零分母：处理 0、对照 0 ⇒ 比值无定义 ⇒ UNKNOWN + 非 0。

        ⚠️ 只用"处理=0"是**测不出**的（分母在对照侧）—— 必须**同时**把对照也置 0，
        等式 `0/0` 才成立。这正是"零分母"反例最容易写错的地方。
        """
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = self._stage(td, round1_treat=0.0, round1_ctrl=0.0)
            p = self._run(root)
            self._assert_unknown_nonzero(p, who="零分母")

    def test_main_entry_rejects_four_rounds(self):
        """★ 反例四轮缺失：删掉处理臂的 `-r4` ⇒ 有效对数 4 ⇒ UNKNOWN + 非 0。

        ⚠️ 这条同时会打中**配对闸门**（两侧轮次集合不同）—— 那是预期的：
        两层闸门都会拦。这里断言的是"最终结论仍为 UNKNOWN"，不区分是哪层拦的。
        """
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = self._stage(td, drop_round="-r4")
            p = self._run(root)
            self._assert_unknown_nonzero(p, who="四轮缺失")

    def test_gate_problems_are_written_to_json(self):
        """门不过时，`--json` 产物必须写出 `valid_numeric_ok=False` + 具体原因。"""
        _skip_if_remote_artifact_absent(self._has_artifacts())
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = self._stage(td, round1_treat=float("nan"))
            out = Path(td) / "neg.json"
            p = self._run(root, "--json", str(out))
            self.assertNotEqual(p.returncode, 0)
            d = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(d["valid_numeric_ok"])
            self.assertFalse(d["gates_ok"])
            self.assertTrue(d["valid_numeric_problems"], "必须写出具体原因")
            self.assertEqual(d["verdict"]["state"], "UNKNOWN")
            self.assertIn("valid_numeric_problems", d)


class TestLocalMustRunScriptIsNotSilentlySkipped(unittest.TestCase):
    """★★ **本机必测脚本**不得因缺输入而"转 skip"（用户 2026-10-03 点名）。

    ⚠️ 这条**不是**在测分析器，而是在测**报告的可信度**：
      端到端用例依赖远端回拉产物 ⇒ 允许 skip；但若 `results/raw/**` **已在本机**
      （即"可以跑"），那么这 6 条端到端反例**必须真的跑到**，
      不允许因为**命名/路径写错**这类本地原因被静默跳过。
      —— 本仓已踩过：`-r5` 写错 ⇒ 6 条反例**全跳过**，报告看起来"全绿"。

    判据：`results/raw/**` 存在 ⇒ 逐条核对**轮次目录真实存在**（不是"猜名字"）。
    """

    RUNS = ("", "-r1", "-r2", "-r3", "-r4")

    def test_when_artifacts_present_every_run_dir_really_exists(self):
        if not _END_TO_END_ARTIFACT.is_dir():
            raise _T.ExternalArtifactMissing(
                f"远端产物未回拉：{_END_TO_END_ARTIFACT} 不存在 ⇒ 本项无法核验")
        missing: list[str] = []
        for arm in ("s01_reference", "s01_both"):
            for suf in self.RUNS:
                d = _ROOT / "results" / "raw" / f"S01RECHECK-short_chat-{arm}{suf}"
                if not (d / "meta.json").is_file():
                    missing.append(str(d))
        self.assertEqual(
            missing, [],
            "产物在本机、但下列轮次目录/meta.json 不存在 ⇒ 端到端用例会**静默跳过**：\n  "
            + "\n  ".join(missing)
            + "\n（先打印真实目录名核对，不要按记忆写轮次后缀）")


class TestTtftMustBeFiniteAndPositive(unittest.TestCase):
    """★★ 要求 ⑤（用户 2026-10-03 点名）：TTFT **两侧**必须有限且 **> 0**。

    ⚠️ 为什么 `_finite` 不够：`0` 与负数都能通过 `_finite`，但它们在时延上
       **不可能发生**，且都会**往"更好"的方向**污染结论：
         · `ttft = 0` ⇒ `ratio → 0` ⇒ 读成"零延迟"（比任何真值都好）；
         · `ttft < 0` ⇒ 压低中位数 ⇒ 处理臂看起来更快。
       ⇒ 必须显式要求 `> 0`，并逐条报出轮次与数值。
    """

    def test_zero_ttft_is_rejected(self):
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(5)]
        t[1]["ttft_p50"] = 0.0
        summary = {m: A.summarize_pair(A.paired_rows(c, t, [m]), m)
                   for m in A.POSITIVE_LATENCY_METRICS}
        ok, probs = A.check_valid_numeric(summary, expect_pairs=5)
        self.assertFalse(ok, "ttft_p50 = 0 必须被拒（时延不可能为 0）")
        self.assertTrue(any("非正" in p or "非有限" in p for p in probs), probs)

    def test_negative_ttft_is_rejected(self):
        """★ 负值出现在**非主指标**（`ttft_p95`）上时也必须被拦。

        ⚠️ 这正好覆盖一个真实盲区：主指标闸门只看 `ttft_p50`，
        若只在主指标上做正值检查，`ttft_p95 = -12` 这类病态值会**穿过**。
        """
        c = [_round(i, cfg="cfgA") for i in range(5)]
        t = [_round(i, cfg="cfgB") for i in range(5)]
        c[3]["ttft_p95"] = -12.0
        # 主指标（ttft_p50）必须一并给出，否则 `check_valid_numeric` 会在
        # "缺 n_total_rows" 处提前返回（那是**另一类**失败，与本用例无关）。
        summary = {m: A.summarize_pair(A.paired_rows(c, t, [m]), m)
                   for m in A.POSITIVE_LATENCY_METRICS}
        ok, probs = A.check_valid_numeric(summary, expect_pairs=5)
        self.assertFalse(ok, "负 ttft 必须被拒（说明时间戳口径/符号写反）")
        self.assertTrue(any("非正" in p for p in probs), probs)

    def test_positive_ttft_passes(self):
        """正向对照：合法正值必须**通过**（否则判据在测一个恒假条件）。"""
        c = [_round(i, cfg="cfgA", ttft=190.0) for i in range(5)]
        t = [_round(i, cfg="cfgB", ttft=182.0) for i in range(5)]
        summary = {m: A.summarize_pair(A.paired_rows(c, t, [m]), m)
                   for m in A.POSITIVE_LATENCY_METRICS}
        ok, probs = A.check_valid_numeric(summary, expect_pairs=5)
        self.assertTrue(ok, probs)

    def test_zero_is_allowed_for_count_metrics(self):
        """★ 计数类**不**受本规则约束：`n_long_fwd = 0` 是合法结果，不是"不可用"。"""
        self.assertNotIn("n_long_fwd", A.POSITIVE_LATENCY_METRICS)
        self.assertNotIn("n_failed", A.POSITIVE_LATENCY_METRICS)
        for m in A.POSITIVE_LATENCY_METRICS:
            self.assertTrue(m.startswith("ttft"), m)


class TestNonFiniteBecomesJsonNull(unittest.TestCase):
    """★★ 要求 ④（用户 2026-10-03 点名）：非有限诊断值 → `null` + 错误说明，**严格** JSON。"""

    def test_sanitizer_nulls_nonfinite_and_records_reason(self):
        obj = {"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": float("-inf")},
               "ok": 2.5, "n": 3, "flag": True, "s": None}
        clean, notes = A.sanitize_json(obj)
        self.assertIsNone(clean["a"])
        self.assertIsNone(clean["b"][0])
        self.assertEqual(clean["b"][1], 1.0)
        self.assertIsNone(clean["c"]["d"])
        self.assertEqual(clean["ok"], 2.5)
        self.assertIs(clean["n"], 3)
        self.assertIs(clean["flag"], True)          # bool 不能被当成数字
        self.assertIsNone(clean["s"])
        self.assertEqual(len(notes), 3, notes)
        for n in notes:
            self.assertIn("不是有限数", n)

    def test_sanitized_object_is_strict_json_serializable(self):
        """★ 关键：清洗后 `allow_nan=False` 必须**能写出**（否则产物永远写不出来）。"""
        import json as _json
        obj = {"x": float("inf"), "y": [float("nan")], "z": 1.0}
        clean, _ = A.sanitize_json(obj)
        text = _json.dumps(clean, ensure_ascii=False, allow_nan=False)
        self.assertNotIn("Infinity", text)
        self.assertNotIn("NaN", text)
        self.assertEqual(_json.loads(text), {"x": None, "y": [None], "z": 1.0})

    def test_negative_control_raw_dump_would_be_illegal(self):
        """破坏性对照：**不清洗**时 `allow_nan=False` 必须**抛异常**。

        ⇒ 证明本修真正在做的事（否则 `sanitize_json` 只是摆设）。
        """
        import json as _json
        with self.assertRaises(ValueError):
            _json.dumps({"x": float("inf")}, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
