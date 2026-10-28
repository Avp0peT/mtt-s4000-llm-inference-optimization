"""`scripts/d1_gate.py` 的用例：**三态**判定必须真的三态。

要挡住的失效形态（都真实发生过）
--------------------------------
* **假失败**：臂名解析与另一处不一致 ⇒ 在自己的字典里查不到数据 ⇒ 把"数不出来"
  报成"没发生"（2026-09-27：条件1「小 chunk 一步都没被执行」、条件2「arm_state 数不出来」
  两条都是假的，真值分别是 **168 步**、`n_degraded=34`）。
* **假阴性（本脚本第一版自己犯的）**：把 `execution_environment` / `hardware_platform` /
  `label_source` 当成 `meta.json` 的**顶层键**去读 —— 而真产物里前两个在
  **`reproducibility.*` 与 `config.runtime.*`**、`label_source` **根本不在 meta 里**。
  ⇒ 会对**每一个合法 `MEASURED` 产物**判 `INCONCLUSIVE`。
  **本文件因此改成"照真产物的形状"造夹具**（形状抄自 `results/raw/S14P-*/meta.json`）。
* **假通过**：`grep -c GATE_PASS` 打 append-only 日志 ⇒ 历史一次通过会把之后每次运行都误触发。
* **把缺失当 0**：没有 telemetry 就写 `small_steps=0`，于是"没测"长成"测到没发生"。

用例覆盖（对应用户点名的五类）：真实目录名 / 空集 / 非 MEASURED / 缺字段 / 历史日志干扰，
外加双向（有效证据下 PASS 与 FAIL 都必须可达）。
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import d1_gate as G  # noqa: E402

#: ⚠️ 这个形状**逐字照抄真实产物**（`S14P-short_chat-fixed_cost_min512/meta.json`，
#: 2026-09-27 用 `remote_run.py exec` 打印过）——**不是想象出来的**。
#: 尤其是：`execution_environment` / `hardware_platform` 在 `reproducibility` 与
#: `config.runtime` 两处，`label_source` 不在 meta 里。
_COMMIT = "d20727d5024249de6e18848452453d5175f7c41c"


def make_run(root: Path, run_id: str, *, small_steps: int = 0, pf_steps: int = 5,
             n_degraded: int = 10, n_decisions: int = 20, n_fallback: int = 1,
             data_label: str | None = "MEASURED", git_commit: str | None = _COMMIT,
             git_dirty: bool | None = False,
             execution_environment: str | None = "remote_mthreads",
             hardware_platform: str | None = "mthreads",
             config_runtime: bool = True, arm_state: bool = True,
             telemetry: bool = True, run_id_in_meta: str | None = None,
             config_stem: str | None = "__derive__"):
    """造一个**结构真实**的 run 目录（meta.json + telemetry.jsonl）。

    ⚠️ `arm_state` 的键**逐字照抄真实产物**（`S14G-..._cost_pilot/meta.json`）：
    它含 `n_decisions`/`n_fallback`/`n_degraded`/`n_no_candidates`/
    `n_memory_conservative`/`n_prefill_starved`/`n_idle`/`last_fallback_reason`。
    少造 `n_fallback` 会让 v2 的机制3 判"数不出来" —— 那是**夹具的缺口**，
    不是判据的错（判据要求"回退也被如实计数"是对的）。

    ⚠️ `config_stem`（2026-09-28 补）：**臂名的唯一权威来源**。缺了它，
    `d1_gate` 就只能退回名字解析 —— 而名字解析在本项目**已被证明会给出错臂名**
    （见 `tests/test_d1_gate_arm_name.py`）。默认 `"__derive__"` 表示
    **按真实产物的形状**从 run_id 的尾部推出来（= `config_stem` 就是 run_id 的尾段），
    这样夹具与真产物同形；要造"名字与 stem 分叉"的病例就**显式传** `config_stem`。
    """
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    if config_stem == "__derive__":
        # 真产物的形状：run_id = `<tag>-<workload>-<config_stem>[-rN]`
        config_stem = re.sub(r"-r\d+$", "", run_id).split("-", 2)[-1]
    meta: dict = {
        "summary": {"run_id": run_id_in_meta if run_id_in_meta is not None else run_id},
        "reproducibility": {"git_commit": git_commit, "git_source": "stamp",
                            "git_dirty": git_dirty,
                            "execution_environment": execution_environment,
                            "hardware_platform": hardware_platform},
        "config": {"runtime": ({"execution_environment": execution_environment,
                                "hardware_platform": hardware_platform}
                               if config_runtime else {})},
        "controller_effect": ({"arm_state": {"n_degraded": n_degraded,
                                             "n_decisions": n_decisions,
                                             "n_fallback": n_fallback,
                                             "n_no_candidates": 0,
                                             "n_memory_conservative": 0,
                                             "n_prefill_starved": 1,
                                             "n_idle": 0,
                                             "last_fallback_reason": "模型范围外（不可外推）"}}
                              if arm_state else {}),
    }
    if config_stem is not None:
        meta["config"]["config_stem"] = config_stem
    if data_label is not None:
        meta["data_label"] = data_label
    with open(d / "meta.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    if telemetry:
        lines = [json.dumps({"step": i, "note": "prefill_tok=100 decode=1"})
                 for i in range(pf_steps)]
        lines += [json.dumps({"step": pf_steps + i, "note": "prefill_tok=128 decode=0"})
                  for i in range(small_steps)]
        with open(d / "telemetry.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    return d


def make_batch(root: Path, *, arm_small: int = 3, repeats: int = 3, **kw):
    for i in range(repeats):
        sfx = "" if i == 0 else f"-r{i}"
        make_run(root, f"S14G-short_chat-fixed_cost_pilot{sfx}", small_steps=arm_small, **kw)
        make_run(root, f"S14G-short_chat-fixed_cost_min512{sfx}", small_steps=0, **kw)
    return root


class _Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def write_diag(self, feasible_rate: float | None = 0.10,
                   arms=("cost_pilot", "cost_min512"),
                   coverage: float | None = None,
                   n_feasible: int | None = None) -> Path:
        """造一个诊断产物（D0 可行率 + 决策级覆盖率）。`None` ⇒ 该键缺失。

        ⚠️ `coverage=None` 必须**真的不写这个键**（不是写 0）—— v2 的准入条件
        要能区分"覆盖率=0"（模型没被用上）与"覆盖率缺失"（没测），两者都判
        `INCONCLUSIVE` 但**理由不同**。

        ⚠️ `d0` 的键**逐字照抄真实产物**（`diag_S14G.json`）：含
        `n_decisions`/`n_candidates`/`n_feasible`/`feasible_rate`/`top_reject_reasons`。
        v2 的机制2 读 `n_feasible` ⇒ 夹具少了它会把"夹具缺口"报成"机制失败"。
        `n_feasible=None` 时**由 `feasible_rate` 派生**一个非零值（默认行为），
        显式传 `0` 才表示"一个可行候选都没有"。
        """
        p = self.root / "diag.json"
        body: dict = {"tag": "S14G", "arms": {}}
        for a in arms:
            nf = n_feasible
            if nf is None and feasible_rate:
                nf = max(1, int(round(feasible_rate * 100)))
            d0 = {"n_decisions": 100, "n_candidates": 1000,
                  "n_feasible": nf, "feasible_rate": feasible_rate,
                  "top_reject_reasons": []}
            body["arms"][a] = {"d0": d0}
            if coverage is not None:
                body["arms"][a]["coverage"] = {"n_decisions": 100, "n_covered": 70,
                                               "coverage": coverage}
        p.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8", newline="\n")
        return p

    def judge(self, **kw):
        # ⚠️ 只在调用方**没给** diag_path 时才造默认的：早先写成
        # `diag_path=self.write_diag()` 会在每次 judge() 时覆盖用例自己写的那份，
        # 于是"臂不在诊断里"这条用例测的是被覆盖后的夹具（自己骗自己）。
        if "diag_path" not in kw:
            kw["diag_path"] = self.write_diag()
        args = dict(raw_root=self.root, tag="S14G", arm="cost_pilot", ref="cost_min512",
                    min_repeats=3, criteria="v1")
        args.update(kw)
        return G.judge(**args)


class TestThreeStatesAreReal(_Base):
    def test_valid_evidence_can_pass(self):
        """★ 双向之一：三条判据都满足 ⇒ 必须能给 PASS（且是 MEASURED 型产物）。"""
        make_batch(self.root, arm_small=3, n_degraded=5, n_decisions=20)
        for i in range(3):
            sfx = "" if i == 0 else f"-r{i}"
            make_run(self.root, f"S14G-short_chat-fixed_cost_min512{sfx}",
                     small_steps=0, n_degraded=18, n_decisions=20)
        rep = self.judge()
        self.assertEqual(rep["state"], "PASS", rep.get("reasons"))

    def test_valid_evidence_can_fail(self):
        """★ 双向之二：有效证据 + 判据不满足 ⇒ FAIL（而不是 INCONCLUSIVE）。"""
        make_batch(self.root, arm_small=0, n_degraded=30, n_decisions=20)
        rep = self.judge()
        self.assertEqual(rep["state"], "FAIL")
        self.assertTrue(rep["criteria_evaluated"])
        self.assertTrue(any("条件1" in r for r in rep["reasons"]))

    def test_exit_codes_are_distinct(self):
        self.assertEqual({G.PASS, G.FAIL, G.INCONCLUSIVE, G.BAD_INPUT}, {0, 1, 2, 3})


class TestFingerprintLocationMatchesRealArtifacts(_Base):
    """★ 单独钉住"读哪":第一版把三个身世字段当顶层键读 ⇒ 对合法产物全判不可判。"""

    def test_measured_artifact_with_real_shape_is_accepted(self):
        make_batch(self.root, arm_small=3, n_degraded=5, n_decisions=20)
        for i in range(3):
            sfx = "" if i == 0 else f"-r{i}"
            make_run(self.root, f"S14G-short_chat-fixed_cost_min512{sfx}",
                     small_steps=0, n_degraded=18, n_decisions=20)
        rep = self.judge()
        self.assertTrue(rep["criteria_evaluated"],
                        f"合法 MEASURED 产物不得被判不可判：{rep['reasons']}")
        r0 = rep["runs"][0]
        self.assertEqual(r0["execution_environment"], "remote_mthreads")
        self.assertEqual(r0["hardware_platform"], "mthreads")
        self.assertEqual(r0["label_source"], "explicit")

    def test_config_runtime_is_a_fallback_when_reproducibility_lacks_it(self):
        """`reproducibility` 里缺、`config.runtime` 里有 ⇒ 仍应判为有指纹。"""
        make_batch(self.root, arm_small=3)
        # 把 reproducibility 里的两项换成缺省，只留 config.runtime
        for p in self.root.glob("S14G-*/meta.json"):
            m = json.loads(p.read_text(encoding="utf-8"))
            m["reproducibility"].pop("execution_environment", None)
            m["reproducibility"].pop("hardware_platform", None)
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(m, fh, ensure_ascii=False)
        rep = self.judge()
        self.assertFalse(any("缺指纹" in r for r in rep["reasons"]), rep["reasons"])

    def test_missing_data_label_means_no_label_source(self):
        """`label_source` 由"产物自带 data_label"推出 —— 没有 data_label 就没有它。"""
        make_batch(self.root, arm_small=3, data_label=None)
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        joined = "\n".join(rep["reasons"])
        self.assertIn("label_source", joined)
        self.assertIn("data_label", joined)

    def test_dirty_tree_is_inconclusive(self):
        make_batch(self.root, arm_small=3, git_dirty=True)
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertTrue(any("git_dirty" in r for r in rep["reasons"]))


class TestInconclusiveCases(_Base):
    """缺数据 / 缺指纹 / 解析异常 ⇒ 一律 INCONCLUSIVE，**不得**判 PASS/FAIL。"""

    def test_empty_set_is_inconclusive_not_pass(self):
        """空集最容易被读成"没有坏消息 ⇒ 过" —— 必须挡住。"""
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertFalse(rep["criteria_evaluated"])
        self.assertIn("没有任何 run 目录", rep["reasons"][0])

    def test_non_measured_is_inconclusive(self):
        make_batch(self.root, arm_small=3, data_label="NOT_MEASURED")
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertTrue(any("只接受 MEASURED" in r for r in rep["reasons"]))

    def test_missing_fingerprint_fields_is_inconclusive(self):
        make_batch(self.root, arm_small=3, execution_environment=None,
                   hardware_platform=None)
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        joined = "\n".join(rep["reasons"])
        self.assertIn("execution_environment", joined)
        self.assertIn("hardware_platform", joined)

    def test_missing_git_commit_is_inconclusive(self):
        make_batch(self.root, arm_small=3, git_commit="NOT_MEASURED")
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertTrue(any("git_commit" in r for r in rep["reasons"]))

    def test_missing_arm_state_is_inconclusive(self):
        """`arm_state` 缺失（真产物里它就在 `controller_effect` 下）。"""
        make_batch(self.root, arm_small=3, arm_state=False)
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertTrue(any("arm_state" in r for r in rep["reasons"]))

    def test_missing_telemetry_is_inconclusive_not_zero(self):
        """★ "没测"不得长成"测到 0 步"。"""
        make_batch(self.root, arm_small=3, telemetry=False)
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertTrue(any("telemetry" in r for r in rep["reasons"]))

    def test_unparsable_run_id_is_inconclusive(self):
        """run_id 解析失败 ⇒ 不可判（**不得**回退成 controller.kind 去凑一个臂名）。"""
        make_batch(self.root, arm_small=3)
        # ⚠️ 目录名必须仍落在 `S14G-*` 通配里（否则它根本不会被读进来，
        #    用例就会"因为没加载"而假通过）。
        make_run(self.root, "S14G-short_chat-fixed_weirdname", small_steps=3,
                 run_id_in_meta="nonsense")
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertTrue(any("解析" in r for r in rep["reasons"]), rep["reasons"])

    def test_too_few_repeats_is_inconclusive(self):
        make_batch(self.root, arm_small=3, repeats=2)
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertTrue(any("重复" in r for r in rep["reasons"]))

    def test_missing_arm_is_inconclusive_with_arm_list(self):
        """★ 直接覆盖那次假失败的形态：产物里的臂名与期望臂不一致。"""
        make_batch(self.root, arm_small=3)
        rep = self.judge(arm="fixed_cost_pilot")   # 旧代码算出来的那个错名
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        joined = "\n".join(rep["reasons"])
        self.assertIn("fixed_cost_pilot", joined)
        self.assertIn("cost_pilot", joined)        # 报错时必须把"现有的臂"列出来

    def test_arm_missing_from_diag_is_inconclusive(self):
        make_batch(self.root, arm_small=3)
        diag = self.root / "diag_partial.json"
        diag.write_text(json.dumps({"tag": "S14G", "arms": {"cost_min512": {}}}),
                        encoding="utf-8", newline="\n")
        rep = self.judge(diag_path=diag)
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertTrue(any("解析口径不一致" in r for r in rep["reasons"]), rep["reasons"])

    def test_inconclusive_still_reports_metrics(self):
        """不可判 ≠ 不报数：metrics 要留给人修产物用，但必须标为「不是判定」。"""
        make_batch(self.root, arm_small=7, data_label="NOT_MEASURED")
        rep = self.judge()
        self.assertEqual(rep["state"], "INCONCLUSIVE")
        self.assertEqual(rep["metrics"]["cost_pilot"]["small_steps"], 21)  # 7 × 3 轮
        self.assertFalse(rep["criteria_evaluated"])
        self.assertIn("不是判定", rep["note"])
        self.assertIn("criteria_if_valid_v1", rep)


class TestRealS14GShape(_Base):
    """用**真实 S14G 的目录名与字段形状**走一遍（形状照抄产物）。"""

    def test_real_run_names_group_into_two_arms(self):
        for i in range(3):
            sfx = "" if i == 0 else f"-r{i}"
            make_run(self.root, f"S14G-short_chat-fixed_cost_pilot{sfx}", small_steps=56,
                     n_degraded=39, n_decisions=42)
            make_run(self.root, f"S14G-short_chat-fixed_cost_min512{sfx}", small_steps=0,
                     n_degraded=34, n_decisions=37)
        rep = self.judge()
        self.assertEqual(rep["arms_present"], ["cost_min512", "cost_pilot"])
        m = rep["metrics"]["cost_pilot"]
        self.assertEqual((m["n_runs"], m["small_steps"], m["n_degraded"], m["n_decisions"]),
                         (3, 168, 117, 126))
        self.assertAlmostEqual(m["degraded_rate"], 117 / 126)

    def test_all_arms_report_uses_the_same_parser(self):
        """`all_arms_report` 与 `judge` 必须用**同一份**臂名派生（这条是本用例的真意图）。

        ⚠️ 旧版本这里期望 `['cost_min128', 'cost_min256']`（= 旧 `parse_run_id`
        剥掉 `fixed_` 之后的写法）。**2026-09-28 判定该期望是错的**，三条理由：
        ① 本文件的上一条 `test_real_run_names_group_into_two_arms` 对**同一个形状**
           （`fixed_cost_min512`）期望的是 **`cost_min512`**，两条自相矛盾；
        ② `S14H`/`S14K` **从未跑过**（任务书：批次未启动）⇒ 这条只为演示而手造的期望；
        ③ 决定性：`S14G` 真产物的 `config.config_stem` = **`fixed_cost_min512`**
           ⇒ **`cost_min512` 这个配置名在产物里根本不存在**，`--arm cost_min512`
           能取到数据纯属"旧解析器错得恰好一致"。
        ⇒ 现在以**产物的 `config_stem`** 为准 = `fixed_cost_*`。
        """
        for i in range(2):
            sfx = "" if i == 0 else f"-r{i}"
            make_run(self.root, f"S14H-short_chat-fixed_cost_min256{sfx}", small_steps=5)
            make_run(self.root, f"S14H-short_chat-fixed_cost_min128{sfx}", small_steps=9)
        rep = G.all_arms_report(self.root, "S14H")
        self.assertEqual(sorted(rep["arms"]),
                         ["fixed_cost_min128", "fixed_cost_min256"],
                         "臂名以产物的 config_stem 为准（名字解析只做兜底）")
        self.assertEqual(rep["arms"]["fixed_cost_min128"]["small_steps"], 18)
        # ★ 同一份派生的证据：judge 侧取同一个名字也应当命中同一批 run
        runs = G.load_runs(self.root, "S14H")
        self.assertEqual(len(G._per_arm(runs)["fixed_cost_min128"]), 2)

    def test_all_arms_report_on_empty_tag_is_empty_not_error(self):
        rep = G.all_arms_report(self.root, "NOPE")
        self.assertEqual(rep["n_run_dirs"], 0)
        self.assertEqual(rep["arms"], {})


class TestPreflightGuards(unittest.TestCase):
    """★ 开跑前检查必须有"缺指纹就拒跑"，否则整批会被静默标成 `NOT_MEASURED`。

    2026-09-27 实测：新根没有 `results/raw/_env/` ⇒ 6 个 S14G 产物全部 `NOT_MEASURED`，
    而 run 正常结束、数字正常，只有一行警告埋在长日志里。
    ⇒ 让它在**开跑前**就拒绝，而不是跑完再发现。
    """

    def _code(self, name: str) -> str:
        p = _ROOT / "scripts" / name
        return "\n".join(ln for ln in p.read_text(encoding="utf-8").splitlines()
                         if not ln.lstrip().startswith("#"))

    def test_gate1_refuses_without_fingerprint(self):
        code = self._code("run_s14_d1.sh")
        self.assertIn("results/raw/_env/env.json", code)
        self.assertIn("S14D1_ABORTED_NO_ENV", code)

    def test_gate2_refuses_without_fingerprint(self):
        code = self._code("run_s14_d1_min.sh")
        self.assertIn("results/raw/_env/env.json", code)
        self.assertIn("S14D2_ABORTED_NO_ENV", code)

    def test_both_queues_call_the_shared_gate(self):
        for name in ("run_s14_d1.sh", "run_s14_d1_min.sh"):
            with self.subTest(script=name):
                self.assertIn("scripts/d1_gate.py", self._code(name))


class TestHistoryLogCannotTrigger(_Base):
    """★ 触发只能来自**本次运行的结构化结果**。"""

    SHELL = _ROOT / "scripts" / "run_s14_d1.sh"

    def _code_lines(self, path: Path) -> str:
        return "\n".join(ln for ln in path.read_text(encoding="utf-8").splitlines()
                         if not ln.lstrip().startswith("#"))

    def test_shell_does_not_grep_the_append_only_log(self):
        code = self._code_lines(self.SHELL)
        self.assertNotIn('grep -c "S14D1_GATE_PASS"', code,
                         "禁止再用 grep 累计日志触发下一阶段（历史 PASS 会误触发）")
        self.assertNotIn("GATE_PASS", code.replace("d1_gate", ""))

    def test_gate2_shell_also_has_no_log_grep(self):
        code = self._code_lines(_ROOT / "scripts" / "run_s14_d1_min.sh")
        self.assertNotIn("GATE_PASS", code.replace("d1_gate", ""))
        self.assertIn("scripts/d1_gate.py", code)

    def test_shell_branches_on_exit_code(self):
        src = self.SHELL.read_text(encoding="utf-8")
        self.assertIn("scripts/d1_gate.py", src)
        self.assertIn("GATE_RC=$?", src)
        self.assertIn('case "$GATE_RC" in', src)

    def test_gate_never_reads_a_log_file(self):
        """判据本体不得依赖任何日志（那会把"环境状态"变成输入）。"""
        code = self._code_lines(_ROOT / "scripts" / "d1_gate.py")
        self.assertNotIn(".log", code)

    def test_verdict_is_a_pure_function_of_inputs(self):
        """同样输入跑两次必须同结论；且**另一批的存在**不影响本批。"""
        make_batch(self.root, arm_small=0, n_degraded=30, n_decisions=20)
        first = self.judge()["state"]
        make_batch(self.root, arm_small=3, n_degraded=5, n_decisions=20)   # 另一批全好
        second = self.judge()["state"]
        self.assertEqual(first, "FAIL")
        self.assertEqual(second, "FAIL", "另一批的存在不得改变本批判定")


class TestCriteriaV2MechanismTier(_Base):
    """v2 = **机制诊断档**（2026-09-27 冻结）。钉住三件事：

    1. **不判服务收益**（过载区里 D3 恒 0，判它等于判"负载不可达"）；
    2. **覆盖率作准入**（@50%）——这是 v1 没有的；
    3. **D0 低不判失败**（那说明落在过载区，不是机制坏）。
    """

    def _good_batch(self):
        make_batch(self.root, arm_small=3, n_degraded=5, n_decisions=20)
        for i in range(3):
            sfx = "" if i == 0 else f"-r{i}"
            make_run(self.root, f"S14G-short_chat-fixed_cost_min512{sfx}",
                     small_steps=0, n_degraded=18, n_decisions=20)
        return self.root

    def test_v2_passes_mechanism_even_in_overload(self):
        """★ 关键：过载区（D0 极低）+ 机制在工作 ⇒ v2 **PASS**（v1 会 FAIL）。

        这就是"过载批次用于机制诊断"这条口径的可执行形式。
        """
        self._good_batch()
        rep = self.judge(criteria="v2", diag_path=self.write_diag(feasible_rate=0.0476,
                                                                  coverage=0.976))
        self.assertEqual(rep["state"], "PASS", rep.get("reasons"))
        self.assertEqual(rep["criteria_v2"]["tier"], "mechanism_diagnosis")
        self.assertIn("D3 服务收益", rep["criteria_v2"]["does_not_judge"])

    def test_v1_fails_on_same_batch(self):
        """同一批：v1 的跨模型阈值 + 只比次数 ⇒ FAIL。保留旧判据是刻意的。"""
        self._good_batch()
        rep = self.judge(criteria="v1", diag_path=self.write_diag(feasible_rate=0.0476,
                                                                  coverage=0.976))
        self.assertEqual(rep["state"], "FAIL")
        self.assertTrue(any("条件3" in f for f in rep["criteria_v1"]["fails"]))

    def test_v2_blocks_on_low_coverage(self):
        """准入：覆盖率 < 50% ⇒ INCONCLUSIVE（模型没被用上不算验了机制）。"""
        self._good_batch()
        rep = self.judge(criteria="v2", diag_path=self.write_diag(feasible_rate=0.30,
                                                                  coverage=0.10))
        self.assertEqual(rep["state"], "FAIL")
        self.assertTrue(any("覆盖率" in f and "准入" in f
                            for f in rep["criteria_v2"]["fails"]))

    def test_v2_coverage_exactly_at_threshold_passes(self):
        """边界：恰好 50% ⇒ 过（判据是 `<`，不是 `<=`）。"""
        self._good_batch()
        rep = self.judge(criteria="v2", diag_path=self.write_diag(feasible_rate=0.10,
                                                                  coverage=0.50))
        self.assertTrue(any("准入过" in n for n in rep["criteria_v2"]["notes"]))

    def test_v2_missing_coverage_is_not_a_pass(self):
        """★ 缺覆盖率 ⇒ 不得当通过。缺失与 0 都必须挡住（理由不同）。"""
        self._good_batch()
        rep = self.judge(criteria="v2", diag_path=self.write_diag(feasible_rate=0.10,
                                                                  coverage=None))
        self.assertEqual(rep["state"], "FAIL")
        self.assertTrue(any("数不出来" in f for f in rep["criteria_v2"]["fails"]))

    def test_v2_requires_executed_action(self):
        """机制1：遥测**在**、但一步 prefill 都没执行 ⇒ FAIL。

        ⚠️ 夹具要区分两种"没有 prefill"：遥测缺失（`telemetry_ok=False` ⇒ 有效性前置
        先挡，判 `INCONCLUSIVE`）与"测到了、确实是 0 步"（⇒ 机制1 `FAIL`）。
        本用例要的是后者，所以必须让遥测**有内容**（只含 decode 步）。
        """
        for i in range(3):
            sfx = "" if i == 0 else f"-r{i}"
            for arm in ("cost_pilot", "cost_min512"):
                d = make_run(self.root, f"S14G-short_chat-fixed_{arm}{sfx}",
                             small_steps=0, pf_steps=0, n_degraded=5, n_decisions=20)
                # 遥测文件非空、但每步都是 decode（prefill_tok=0）
                with open(d / "telemetry.jsonl", "w", encoding="utf-8", newline="\n") as fh:
                    fh.write("\n".join(
                        json.dumps({"step": i, "note": "prefill_tok=0 decode=1"})
                        for i in range(5)) + "\n")
        rep = self.judge(criteria="v2", diag_path=self.write_diag(feasible_rate=0.5,
                                                                  coverage=0.9))
        self.assertEqual(rep["state"], "FAIL", rep.get("reasons"))
        self.assertTrue(any("机制1" in f for f in rep["criteria_v2"]["fails"]))

    def test_v2_requires_feasible_candidate(self):
        """机制2：D0 里没有任何可行候选被采纳 ⇒ 判据把候选全拒了 ⇒ FAIL。"""
        self._good_batch()
        rep = self.judge(criteria="v2",
                         diag_path=self.write_diag(feasible_rate=0.0, coverage=0.9,
                                                   n_feasible=0))
        self.assertEqual(rep["state"], "FAIL")
        self.assertTrue(any("机制2" in f for f in rep["criteria_v2"]["fails"]))

    def test_v2_reports_denominators_for_degraded_and_fallback(self):
        """机制3：降级与回退**都连报分母**（v1 只比次数的口径已废弃）。"""
        self._good_batch()
        rep = self.judge(criteria="v2", diag_path=self.write_diag(feasible_rate=0.1,
                                                                  coverage=0.9))
        joined = " ".join(rep["criteria_v2"]["notes"])
        self.assertIn("连报分母", joined)

    def test_v2_unknown_version_is_inconclusive(self):
        """未知判据版本 ⇒ INCONCLUSIVE（不得默默当 v1）。"""
        self._good_batch()
        rep = self.judge(criteria="v9")
        self.assertEqual(rep["state"], "INCONCLUSIVE")

    def test_v2_default_in_cli(self):
        """CLI 默认判据必须是 v2（改默认值要显式改这里，防止回退成 v1 而无人察觉）。"""
        import argparse
        src = (_ROOT / "scripts" / "d1_gate.py").read_text(encoding="utf-8")
        self.assertIn('default="v2"', src)
        self.assertIn('choices=("v1", "v2")', src)


if __name__ == "__main__":
    unittest.main()
