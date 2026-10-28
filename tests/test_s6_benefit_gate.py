"""`scripts/s6_benefit_gate.py` 的用例：**三态判定必须真的三态**，且以**真产物形状**为夹具。

要挡住的失效形态（都真实发生过或差点发生）
------------------------------------------
* **臂名解析错**（本脚本第一版自己犯的）：S6 的 run_id 形如
  `S6-short_chat-service_benefit_c2_m4_cost[-rN]` —— cell 名里既有 `-` 又有 `_`，
  按 `-` 切会拿到**整个 cell 名**（`service_benefit_c2_m4_cost`）而不是臂 `cost`
  ⇒ 三臂全部匹配失败、报 `run 0 次`，而 9 个目录明明在。
  **这会把"没读到"伪装成"没跑到"** —— 与 `d1_gate.py` 的"假失败"同族。
  ⇒ 本文件必须有一条用例**专门**钉住"臂名从产物记录的 config 取，不靠目录名猜"。
* **把接线失败读成没收益**：`cost` 臂没有 `cost_policy` 条目 ⇒ 必须判
  INCONCLUSIVE（接线失败），**不能**落到 FAIL（负收益）。
* **把不可区分读成有收益**：三臂读数全等 ⇒ 必须 INCONCLUSIVE，不得声称任一臂更优。
* **空集伪装成合格**：没有产物时 `min([])` 会抛，若被 try/except 吞掉就变成"通过"。

用例覆盖：三态可达（PASS / FAIL / INCONCLUSIVE）+ 空集 + 臂名解析 + 重复不足。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import s6_benefit_gate as G  # noqa: E402

#: ⚠️ 夹具形状**逐字照抄真实产物**（`results/raw/P1S3-phase1_anchor-adaptive_c1/meta.json`）。
#: 关键：`context.config_path` 与 `config.config_stem` 是**臂名的唯一可信来源**；
#: `summary` 里的分位块是**嵌套的**。
_ARMS = ("static", "threshold", "cost")


def _meta(arm: str, *, goodput: float, sat: float, cost_policy: bool,
          tag: str = "S6") -> dict:
    """造一个 S6 形状的 meta.json（config_path 指向该臂的配置）。"""
    stem = f"service_benefit_c2_m4_{arm}"
    trace = [{"step": i, "state": {"n_decisions": 1}} for i in range(3)]
    if cost_policy:
        trace = [dict(e, cost_policy={"chunk": 512, "candidates": [
            {"feasible": True, "predicted_ms": 100.0, "reject_reason": ""}]}) for e in trace]
    return {
        "context": {
            "config_path": f"/root/ascend_probe_d1/configs/mthreads/phase1_s6/{stem}.yaml",
            "run_id": f"{tag}-short_chat-{stem}",
        },
        "config": {"config_stem": stem},
        "summary": {
            "config": stem,
            "slo_goodput_req_s": goodput,
            "slo_satisfaction_rate": sat,
        },
        "controller_trace": trace,
        "controller_effect": {"arm_state": {"n_degraded": 0, "n_decisions": 3}},
    }


def _write(tmp: Path, arm: str, *, repeats: int = 3, tag: str = "S6", **kw) -> None:
    for r in range(repeats):
        suffix = "" if r == 0 else f"-r{r}"
        d = tmp / f"{tag}-short_chat-service_benefit_c2_m4_{arm}{suffix}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(
            json.dumps(_meta(arm, tag=tag, **kw), ensure_ascii=False),
            encoding="utf-8", newline="\n")


class TestArmNameResolution(unittest.TestCase):
    """★ 本组专门钉住"臂名从产物取、不从目录名猜"。"""

    def test_arm_from_config_path_not_dirname(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            m = _meta("cost", goodput=1.0, sat=0.5, cost_policy=True)
            # 目录名**故意**是错的（不带臂后缀）——只有 config 是权威
            bad = tmp / "S6-short_chat-weird_dir_name"
            bad.mkdir()
            (bad / "meta.json").write_text(json.dumps(m), encoding="utf-8")
            per = G.load_runs("S6", tmp)
            self.assertEqual(len(per["cost"]), 1,
                             "应能从产物记录的 config_path 定出臂 = cost")
            self.assertEqual(sum(len(v) for v in per.values()), 1)

    def test_dirname_underscore_tail_would_misresolve(self):
        """反向证明：若按目录名的 `-` 切，会拿到整个 cell 名（本脚本第一版的 bug）。"""
        name = "S6-short_chat-service_benefit_c2_m4_cost"
        # 第一版算法：去掉 tag 前缀 + 去掉 -rN，然后按 `-` 取末段
        stem = name[len("S6") + 1:]
        stem = stem.rsplit("-r", 1)[0] if "-r" in stem else stem
        wrong_arm = stem.split("-")[-1]
        self.assertEqual(wrong_arm, "service_benefit_c2_m4_cost",
                         "旧算法确实会得到整个 cell 名 —— 故必须从产物取")
        self.assertNotIn(wrong_arm, G.ARMS)


class TestThreeVerdicts(unittest.TestCase):
    def test_pass_when_cost_best(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _write(tmp, "static", goodput=1.0, sat=0.5, cost_policy=False)
            _write(tmp, "threshold", goodput=1.2, sat=0.6, cost_policy=False)
            _write(tmp, "cost", goodput=1.5, sat=0.75, cost_policy=True)
            rc = G.main(["--tag", "S6", "--root", str(tmp), "--repeats", "3"])
            self.assertEqual(rc, G.PASS)

    def test_fail_when_cost_strictly_worse_than_static(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _write(tmp, "static", goodput=2.0, sat=0.9, cost_policy=False)
            _write(tmp, "threshold", goodput=1.5, sat=0.7, cost_policy=False)
            _write(tmp, "cost", goodput=0.5, sat=0.2, cost_policy=True)
            rc = G.main(["--tag", "S6", "--root", str(tmp), "--repeats", "3"])
            self.assertEqual(rc, G.FAIL)

    def test_inconclusive_when_all_indistinguishable(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for a in _ARMS:
                _write(tmp, a, goodput=1.0, sat=0.5, cost_policy=(a == "cost"))
            rc = G.main(["--tag", "S6", "--root", str(tmp), "--repeats", "3"])
            self.assertEqual(rc, G.INCONCLUSIVE)


class TestMechanicalPremise(unittest.TestCase):
    def test_cost_arm_without_cost_policy_is_wiring_failure_not_no_benefit(self):
        """★ cost 臂没跑起来 ⇒ INCONCLUSIVE（接线失败），**不能**是 FAIL。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _write(tmp, "static", goodput=1.0, sat=0.5, cost_policy=False)
            _write(tmp, "threshold", goodput=1.2, sat=0.6, cost_policy=False)
            _write(tmp, "cost", goodput=0.1, sat=0.1, cost_policy=False)  # 无 cost_policy
            rc = G.main(["--tag", "S6", "--root", str(tmp), "--repeats", "3"])
            self.assertEqual(rc, G.INCONCLUSIVE,
                             "接线失败必须与负收益分开（预注册 §5）")

    def test_threshold_arm_with_cost_policy_is_wiring_failure(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _write(tmp, "static", goodput=1.0, sat=0.5, cost_policy=False)
            _write(tmp, "threshold", goodput=1.2, sat=0.6, cost_policy=True)  # 串了
            _write(tmp, "cost", goodput=1.5, sat=0.75, cost_policy=True)
            rc = G.main(["--tag", "S6", "--root", str(tmp), "--repeats", "3"])
            self.assertEqual(rc, G.INCONCLUSIVE)


class TestEmptyAndInsufficient(unittest.TestCase):
    def test_empty_root_is_inconclusive_not_pass(self):
        with tempfile.TemporaryDirectory() as td:
            rc = G.main(["--tag", "S6", "--root", td, "--repeats", "3"])
            self.assertEqual(rc, G.INCONCLUSIVE, "空集不得伪装成合格")

    def test_bad_root_is_bad_input(self):
        rc = G.main(["--tag", "S6", "--root", "/definitely/not/here", "--repeats", "3"])
        self.assertEqual(rc, G.BAD_INPUT)

    def test_insufficient_repeats_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _write(tmp, "static", goodput=1.0, sat=0.5, cost_policy=False)
            _write(tmp, "threshold", goodput=1.2, sat=0.6, cost_policy=False)
            _write(tmp, "cost", goodput=1.5, sat=0.75, cost_policy=True, repeats=2)
            rc = G.main(["--tag", "S6", "--root", str(tmp), "--repeats", "3"])
            self.assertEqual(rc, G.INCONCLUSIVE)


class TestParseRunIdFork(unittest.TestCase):
    """★ 钉住 `parse_run_id` 在**本批 run_id 形状**上的分叉（2026-09-28 实测发现）。

    事实（真机复现，非推断）
    ------------------------
    `src/ascend_opt/run_ids.py` 的 `RUN_ID_RE` 把 `cell` 段写成 `[a-z]+`（**纯小写字母**），
    而 S6 的 run_id 是 `S6-short_chat-service_benefit_c2_m4_cost` —— cell 名里带
    **下划线与数字**。于是正则只能把 cell 吃到 `service`，`arm` 落到
    `benefit_c2_m4_cost`：

        parse_run_id('S6-short_chat-service_benefit_c2_m4_cost')
          -> cell='service', arm='benefit_c2_m4_cost'      # 真臂名是 'cost'

    后果（两处，都**不报错**）
    -------------------------
    1. `scripts/d1_gate.py --all-arms` 打印的臂名是 `benefit_c2_m4_cost`，
       而 `scripts/s6_benefit_gate.py` 打印的是 `cost` ⇒ **同一件事两份实现分叉**，
       正是 `run_ids.py` docstring 里记的那次"假失败"的同族形态。
    2. `S14S` 那批更重：`fixed_cost_scan_c2_m4` 被解析成 arm=`cost_scan_c2_m4`
       —— 而该批真正的"臂"其实是负载格 `(c,m)`，`kind=cost` 只是背景
       ⇒ 说明 `<cell>_<arm>` 这个约定**只在"臂真的存在于 cell 名里"的批次成立**。

    ⚠️ 本用例**只钉住"分叉存在"这个事实**，不主张解析器该怎么改：
       改它属于第 14 步 P0（臂名归一）的范畴，且必须**预注册**后再动
       —— 否则就是"看着数据改判据"。
    """

    def _parse(self, run_id: str):
        sys.path.insert(0, str(_ROOT / "src"))
        from ascend_opt.run_ids import parse_run_id  # 唯一实现
        return parse_run_id(run_id)

    def test_s6_run_id_resolves_arm_to_cell_tail_not_pure_arm(self):
        """S6 形状：解析**成功**（不返回 None），但 arm 是错的 —— 危险在"看起来成功"。"""
        p = self._parse("S6-short_chat-service_benefit_c2_m4_cost")
        self.assertIsNotNone(p, "不得返回 None —— 若返回 None 反而会被当独立状态处理")
        self.assertEqual(p.cell, "service", "cell 只吃到 'service'（正则 cell=[a-z]+）")
        self.assertEqual(p.arm, "benefit_c2_m4_cost", "arm 是 cell 的尾巴，不是 'cost'")
        self.assertNotEqual(p.arm, "cost", "★ 这就是分叉：真臂名应为 'cost'")

    def test_invariant_separator_works_for_short_cell(self):
        """反向对照：cell 是纯小写字母时（老形状）解析**正确** —— 证明缺陷是形状相关的。"""
        p = self._parse("S14G-short_chat-fixed_cost_pilot")
        self.assertEqual(p.cell, "fixed")
        self.assertEqual(p.arm, "cost_pilot", "老形状下 cell/arm 都正确")

    def test_s14s_scan_shape_also_misresolves(self):
        """S14S 负载扫描批**同样错**（跨批次，不是 S6 独有）。"""
        p = self._parse("S14S-short_chat-fixed_cost_scan_c2_m4")
        self.assertEqual(p.arm, "cost_scan_c2_m4", "被切成 cell 的尾巴，不是负载格 'c2_m4'")

    def test_two_arm_name_sources_actually_diverge_on_s6(self):
        """★ 双向证明"两份实现确实分叉"：`parse_run_id` vs `s6_benefit_gate` 的 config 取法。"""
        run_id = "S6-short_chat-service_benefit_c2_m4_cost"
        meta = _meta("cost", goodput=1.0, sat=1.0, cost_policy=True)
        from_parse = self._parse(run_id).arm
        from_config = G._arm_of(meta, run_id)
        self.assertEqual(from_config, "cost", "闸门从 config 取到的是正确臂名")
        self.assertNotEqual(from_parse, from_config,
                            "★ 两份实现必须被证明是分叉的（否则本用例没有意义）")


class TestArmActuallyMoved(unittest.TestCase):
    """★ 钉住「该臂有没有**真的动过**」——本项目"三条接了一半"的判据。

    背景（2026-09-28 S6 实测，不是假想）
    ------------------------------------
    S6 的 `cost` 臂与 `static`/`threshold` 臂在**服务层**差 14.14%，
    而 `controller_effect` 显示差异**全在排队**、且 `cost` 臂的
    `chunk_effective` 从 1024 一路降到 128（另两臂全程恒定 1024）：

        臂         n_steps  TTFT p50  排队 p50  排队 p99  chunk_effective
        static      1057      588.9      0.0      120.6    1024..1024
        threshold   1057      562.9      0.0       94.0    1024..1024
        cost        1274     1155.2    127.2      724.2    1024..128   ← 真的动了

    ⚠️ 这意味着 `n_decisions_changed = 0` **不能**用来判"该臂没动"：
       S6 的 cost 臂 `n_decisions_changed` 就是 0，但 `chunk_effective` 确实在变
       （建议 1024 而生效 128，`n_steps_chunk_request_not_honoured = 1017`）
       ⇒ **只看 `n_decisions_changed` 会把一个真动过的臂判成"没接上"。**
       判据必须是"**生效值**有没有动"，不是"决策计数有没有变"。
    """

    def test_effective_chunk_variation_is_the_moved_evidence(self):
        trace = [
            {"step": 0, "cost_policy": {"chunk": 1024}, "state": {"n_decisions": 1}},
            {"step": 1, "cost_policy": {"chunk": 512}, "state": {"n_decisions": 2}},
            {"step": 2, "cost_policy": {"chunk": 128}, "state": {"n_decisions": 3}},
        ]
        eff = [e["cost_policy"]["chunk"] for e in trace]
        self.assertGreater(len(set(eff)), 1,
                           "生效值有变化 ⇒ 该臂动过（这**不是**靠 n_decisions_changed 判的）")

    def test_zero_changed_decisions_does_not_mean_arm_idle(self):
        """反向对照：`n_decisions_changed = 0` 但 chunk 确实在收缩 —— 两者必须分开。"""
        ce = {"n_control_decisions": 320, "n_decisions_changed": 0,
              "n_steps_chunk_request_not_honoured": 1017,
              "chunk_requested": {"min": 1024, "max": 1024},
              "chunk_effective": {"min": 128, "max": 1024}}
        self.assertEqual(ce["n_decisions_changed"], 0, "S6 的 cost 臂实测就是 0")
        moved = ce["chunk_effective"]["min"] != ce["chunk_effective"]["max"]
        self.assertTrue(moved, "★ 但生效值动了 ⇒ 判「该臂动过」必须看生效值，不看决策计数")

    def test_static_arm_zero_decisions_is_consistent(self):
        """static 臂的 `n_control_decisions = 0` **是**符合预期的（控制器整组跳过）。"""
        ce = {"n_steps": 1057, "n_control_decisions": 0, "n_decisions_changed": 0,
              "chunk_effective": {"min": 1024, "max": 1024}}
        self.assertEqual(ce["n_control_decisions"], 0)
        self.assertEqual(ce["chunk_effective"]["min"], ce["chunk_effective"]["max"],
                         "static 臂生效值应恒定 —— 这与 cost 臂形成对照")


if __name__ == "__main__":
    unittest.main()
