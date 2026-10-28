"""``inspect_step3_runs.py`` 里"控制器动作有没有落到 prefill 上"的守卫测试。

要钉住的缺陷与"无法失败的守卫"是同一族，但换了张面孔：

    最初的分类只有四态：``applied`` / ``not_applied`` / ``no_prefill_after`` /
    ``value_unchanged``。跑第一次真机数据，它输出"c2 有 **5 次**改了却没生效"。
    听起来像个机制缺陷——但它是**假的**：那 5 次改的值（512 / 1024 / 1536）在
    下一个 prefill（step 129）发生之前就已被后续决策改回 2048，也就是说
    **那个窗口里一次 prefill 都没有**。值本身一直是生效的，只是没有对象可施加。

    于是"负载形状的读数"被当成了"控制器缺陷"。两者对下一步的指示完全相反：
    前者说明下一轮要把负载改成持续补入，后者说明控制器有 bug。所以必须分态。

本文件要证明的事：

1. ``parse_note`` **按键名**取值。真机 ``note`` 是给人看的字符串，形如
   ``prefill_tok=1024 decode=0 forwards=1 fwd_ms=247.50 pack_ms=0.38``；
   取不到就返回 ``None``／不计入，**不填 0**（0 是"确实没做 prefill"这一具体断言）。
2. 五态分类各自可复现，且 ``superseded`` 与 ``not_applied`` 的**分界点是
   "下一个 prefill 之前有没有再改一次"**——同一组输入，只增删这一条中间决策，
   分类就必须翻转。这是本文件的核心阳性对照。
3. ``not_applied_to_prefill`` **目前没有真机样本**（见 ``TestRealArtifacts``），
   因此它由合成输入覆盖——**不得**把它写成"已观测到的缺陷"。
4. ``chunk_before`` 取"紧邻上一条 trace 的取值"，不是"上一条 changed 决策的取值"。
   写成后者时第一个动作的 before 恒为 ``None``（真机上真的打印出过
   ``chunk None→512``），而 ``None`` 会让"取值有没有变"恒判为"变了"，
   于是 ``value_unchanged`` 这一态被静默清空。

关于变异测试的一个反直觉结论
------------------------------
本文件写完时按项目惯例做了变异测试（往生产脚本里注入缺陷，断言测试必须失败）。
10 个变异里 **2 个没被抓住，但它们不是测试太弱——是等价变异体**：

* M4：把 ``chunks[i - 1]`` 换成 ``chunks[idx[pos - 1]]``（"紧邻上一条 trace"改成
  "上一条 changed 决策"）。二者在真机上**恒等**：36 份产物里 pos>0 的 117 个样本
  全部相等，因为冷却期的 trace 写入的就是上一次决策后的值。
* M4c：把 ``if i > 0`` 换成 ``if pos > 0``。首个改动发生之前不可能有改动，
  因此那时 ``chunk_size`` 必然等于 ``init_chunk_size``，两式同样恒等（21 个样本实证）。

``rc == 0`` 于是有两种成因：**守卫太弱**，或者**变异体根本不改变行为**。要分清，
就得换一个能改变行为的变异：M4b（去掉初值回退）8 个失败、M4d（before 恒取首个值）
13 个失败——证明这一族守卫是被观测地钉住的。防的是"看到 rc=0 就去加断言"，
那只会加上一堆同样抓不住东西的断言。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_module():
    path = _ROOT / "scripts" / "inspect_step3_runs.py"
    if not path.exists():
        raise unittest.SkipTest(f"缺少脚本 {path}")
    spec = importlib.util.spec_from_file_location("_inspect_step3_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


# --------------------------------------------------------------------------- #
# 逐字抄自真机产物的 note（P1S3B，2026-09-23）
# --------------------------------------------------------------------------- #
#: c8 step 0：一次纯 prefill（1024 tok，无 decode）
NOTE_REAL_PREFILL_ONLY = "prefill_tok=1024 decode=0 forwards=1 fwd_ms=247.50 pack_ms=0.38"
#: c8 step 2：prefill 与 decode 混在同一步
NOTE_REAL_MIXED = "prefill_tok=1024 decode=1 forwards=2 fwd_ms=304.45 pack_ms=2.34"
#: c8 step 126：**纯 decode 步**。第一个数字是 8（decode），prefill_tok 是 0 ——
#: "取第一个数字"会读出 8，从而谎报"这一步做了 prefill"。
NOTE_REAL_DECODE_ONLY = "prefill_tok=0 decode=8 forwards=1 fwd_ms=115.09 pack_ms=27.19"


def _row(step: int, note: str, chunk: int | None = 1024) -> dict:
    return {"step": step, "note": note, "chunk_size": chunk}


def _trace_entry(step: int, chunk: int, changed: bool, reason: str = "",
                 init_chunk: int = 1024) -> dict:
    return {
        "step": step,
        "changed": changed,
        "reason": reason,
        "state": {"chunk_size": chunk, "init_chunk_size": init_chunk},
    }


def _pf(step: int, tok: float, chunk: int) -> dict:
    return {"step": step, "prefill_tok": tok, "chunk_size": chunk}


class TestParseNoteAnchorsOnKeys(unittest.TestCase):
    """note 是文本不是结构化字段，所以解析必须锚在**键名**上。"""

    def test_real_prefill_only_note(self):
        d = M.parse_note(NOTE_REAL_PREFILL_ONLY)
        self.assertEqual(d["prefill_tok"], 1024.0)
        self.assertEqual(d["decode"], 0.0)
        self.assertEqual(d["fwd_ms"], 247.50)
        self.assertEqual(d["pack_ms"], 0.38)

    def test_real_mixed_note_keeps_fields_apart(self):
        d = M.parse_note(NOTE_REAL_MIXED)
        self.assertEqual(d["prefill_tok"], 1024.0)
        self.assertEqual(d["decode"], 1.0)
        self.assertEqual(d["forwards"], 2.0)

    def test_decode_only_note_is_not_read_as_prefill(self):
        """阳性对照：按位置取数会把这一步读成"做了 8 个 token 的 prefill"。"""
        d = M.parse_note(NOTE_REAL_DECODE_ONLY)
        self.assertEqual(d["prefill_tok"], 0.0)
        self.assertEqual(d["decode"], 8.0)
        # 这个诱饵是真的会咬人的：第一个数字 8 ≠ prefill_tok 0
        first_number = float(M._NOTE_KV.search(NOTE_REAL_DECODE_ONLY).group(2))
        self.assertEqual(first_number, 0.0)  # 本样例里两者恰好一致……

    def test_decoy_would_be_misread_by_taking_the_last_number(self):
        """……所以再用"取最后一个数字"作诱饵：那是 pack_ms=27.19，会说成 27 个 token。"""
        nums = [float(m.group(2)) for m in M._NOTE_KV.finditer(NOTE_REAL_DECODE_ONLY)]
        self.assertEqual(nums[-1], 27.19)
        self.assertNotEqual(M.parse_note(NOTE_REAL_DECODE_ONLY)["prefill_tok"], nums[-1])

    def test_absent_or_empty_note_yields_empty_mapping(self):
        for n in (None, "", "   "):
            self.assertEqual(M.parse_note(n), {})


class TestPrefillTokensIsNoneNotZero(unittest.TestCase):
    """不可测的值保持 None；0 是真实测量值。两者在这一问上含义相反。"""

    def test_zero_is_preserved_as_zero(self):
        self.assertEqual(M.prefill_tokens(_row(0, NOTE_REAL_DECODE_ONLY)), 0.0)

    def test_missing_field_is_none_not_zero(self):
        self.assertIsNone(M.prefill_tokens(_row(0, "decode=8 forwards=1 fwd_ms=115.09")))
        self.assertIsNone(M.prefill_tokens(_row(0, "")))
        self.assertIsNone(M.prefill_tokens({"step": 0}))

    def test_count_unparsed_makes_the_gap_visible(self):
        rows = [
            _row(0, NOTE_REAL_PREFILL_ONLY),
            _row(1, "decode=8 forwards=1"),          # 解析不到 prefill_tok
            _row(2, NOTE_REAL_MIXED),
        ]
        self.assertEqual(M.count_unparsed(rows), 1)
        # 不可解析的行既不进事件列表……
        self.assertEqual(len(M.prefill_events(rows)), 2)
        # ……也不被当成 0 混进去（否则事件数会与上面相等却含义不同）
        self.assertNotIn(1, [p["step"] for p in M.prefill_events(rows)])

    def test_zero_prefill_rows_are_excluded_from_events(self):
        rows = [_row(0, NOTE_REAL_PREFILL_ONLY), _row(1, NOTE_REAL_DECODE_ONLY)]
        pf = M.prefill_events(rows)
        self.assertEqual([p["step"] for p in pf], [0])
        self.assertEqual(M.count_unparsed(rows), 0)


class TestActionClassification(unittest.TestCase):
    """五态各自可复现。分类的**分界点**见 TestSupersededVsNotApplied。"""

    def test_applied_real_c8_shape(self):
        """c8 真机形状：step 8 收缩到 512，下一个 prefill 在 step 9 用 512。"""
        trace = [_trace_entry(0, 1024, changed=False),
                 _trace_entry(8, 512, changed=True, reason="TPOT P90 超目标")]
        prefill = [_pf(0, 1024, 1024), _pf(1, 1024, 1024), _pf(9, 512, 512)]
        eff = M.action_effects(trace, prefill)
        self.assertEqual(len(eff), 1)
        self.assertEqual(eff[0]["kind"], M.ACTION_APPLIED)
        self.assertEqual(eff[0]["chunk_before"], 1024)
        self.assertEqual(eff[0]["chunk_after"], 512)
        self.assertEqual(eff[0]["next_prefill_step"], 9)

    def test_value_unchanged_real_c8_shape(self):
        """c8 真机形状：chunk 已触下界，再改仍是 512；动的是 max_batched_tokens。"""
        trace = [_trace_entry(8, 512, changed=True),
                 _trace_entry(32, 512, changed=True, reason="TPOT P90 仍超目标")]
        eff = M.action_effects(trace, [_pf(9, 512, 512)])
        self.assertEqual([e["kind"] for e in eff],
                         [M.ACTION_APPLIED, M.ACTION_VALUE_UNCHANGED])

    def test_no_prefill_after_real_c2_shape(self):
        """c2 真机形状：step 392/416 的改动之后直到 run 结束都没有 prefill。"""
        trace = [_trace_entry(392, 1536, changed=True),
                 _trace_entry(416, 2048, changed=True)]
        eff = M.action_effects(trace, [_pf(129, 2048, 2048)])
        self.assertEqual([e["kind"] for e in eff],
                         [M.ACTION_NO_PREFILL_AFTER, M.ACTION_NO_PREFILL_AFTER])
        self.assertIsNone(eff[0]["next_prefill_step"])

    def test_superseded_real_c2_shape(self):
        """c2 真机形状：step 8→512、32→1024、56→1536、80→2048，prefill 直到 129 才来。

        链条里**必须**保留 56/80 这两次中间决策。少了它们，step 32 那个 1024 就一直
        保持到 step 129 而 prefill 用了 2048——那是真正的 ``not_applied``（合成样本
        见过，真机没有）。所以这组断言同时钉住了两件事：中间决策一多，前几次动作
        就是"被覆盖"，最后一次才是"落到 prefill 上"。
        """
        trace = [_trace_entry(0, 1024, changed=False),
                 _trace_entry(8, 512, changed=True, reason="TPOT P90=291.6ms 超目标"),
                 _trace_entry(32, 1024, changed=True, reason="TTFT P90 超目标"),
                 _trace_entry(56, 1536, changed=True, reason="TTFT P90 超目标"),
                 _trace_entry(80, 2048, changed=True, reason="TTFT P90 超目标")]
        prefill = [_pf(0, 1024, 1024), _pf(3, 1024, 1024), _pf(129, 2048, 2048)]
        eff = M.action_effects(trace, prefill)
        self.assertEqual([e["kind"] for e in eff],
                         [M.ACTION_SUPERSEDED] * 3 + [M.ACTION_APPLIED])
        self.assertEqual(eff[3]["chunk_before"], 1536)
        self.assertEqual(eff[3]["chunk_after"], 2048)
        self.assertEqual(eff[3]["next_prefill_step"], 129)

    def test_chain_without_middle_decisions_yields_not_applied(self):
        """把 56/80 抽掉，同一组输入里 step 32 那次就变成 ``not_applied``。

        这是"分界点是**下一个 prefill 之前有没有再改一次**"的正面证明：
        改的只是链条长度，分类必须翻转。
        """
        trace = [_trace_entry(0, 1024, changed=False),
                 _trace_entry(8, 512, changed=True),
                 _trace_entry(32, 1024, changed=True)]
        prefill = [_pf(0, 1024, 1024), _pf(3, 1024, 1024), _pf(129, 2048, 2048)]
        eff = M.action_effects(trace, prefill)
        self.assertEqual([e["kind"] for e in eff],
                         [M.ACTION_SUPERSEDED, M.ACTION_NOT_APPLIED])


class TestSupersededVsNotApplied(unittest.TestCase):
    """核心阳性对照：两态的唯一区别是"下一个 prefill 之前有没有再改一次"。

    这组测试的存在理由：``not_applied_to_prefill`` 是**唯一**指向控制器机制缺陷
    的那一态。若它与 ``superseded`` 被合并，那么"窗口里没有 prefill"就会冒充
    "改了却没生效"，结论从"下一轮改负载形状"变成"控制器有 bug"。
    """

    BASE_TRACE = [_trace_entry(8, 512, changed=True, reason="TPOT 超目标")]
    BASE_PREFILL = [_pf(129, 2048, 2048)]

    def test_without_intervening_decision_it_is_not_applied(self):
        eff = M.action_effects(self.BASE_TRACE, self.BASE_PREFILL)
        self.assertEqual(len(eff), 1)
        self.assertEqual(eff[0]["kind"], M.ACTION_NOT_APPLIED)
        self.assertIsNone(eff[0]["next_decision_step"])

    def test_with_intervening_decision_it_flips_to_superseded(self):
        trace = self.BASE_TRACE + [_trace_entry(32, 2048, changed=True, reason="TTFT 超目标")]
        eff = M.action_effects(trace, self.BASE_PREFILL)
        self.assertEqual(eff[0]["kind"], M.ACTION_SUPERSEDED)
        self.assertEqual(eff[0]["next_decision_step"], 32)

    def test_intervening_decision_after_the_prefill_does_not_flip_it(self):
        """中间那次决策必须**落在下一个 prefill 之前**才算覆盖。

        step 200 > prefill step 129 ⇒ 值在 129 那一刻仍是 512，prefill 却用了 2048，
        这才是真正的"改了却没生效"。少判这一步，"覆盖"会吞掉真缺陷。
        """
        trace = self.BASE_TRACE + [_trace_entry(200, 2048, changed=True)]
        eff = M.action_effects(trace, self.BASE_PREFILL)
        self.assertEqual(eff[0]["kind"], M.ACTION_NOT_APPLIED)

    def test_tally_reports_all_five_states_even_at_zero(self):
        """缺的态记 0 而不是消失：否则上游会以为'没出现过的态不存在'。"""
        tally = M.effect_tally(M.action_effects(self.BASE_TRACE, self.BASE_PREFILL))
        self.assertEqual(set(tally), set(M.ACTION_KIND_NOTE))
        self.assertEqual(len(tally), 5)
        self.assertEqual(tally[M.ACTION_NOT_APPLIED], 1)
        self.assertEqual(tally[M.ACTION_APPLIED], 0)


class TestChunkBeforeUsesImmediatelyPrecedingTrace(unittest.TestCase):
    """before 取"紧邻上一条 trace 的取值"，不是"上一条 changed 决策的取值"。"""

    def test_before_is_not_none_for_the_first_action(self):
        """真机上打印过 ``chunk None→512``：before 为 None 时"有没有变"恒判为"变了"。"""
        trace = [_trace_entry(0, 1024, changed=False),
                 _trace_entry(4, 1024, changed=False),
                 _trace_entry(8, 512, changed=True)]
        eff = M.action_effects(trace, [_pf(9, 512, 512)])
        self.assertEqual(eff[0]["chunk_before"], 1024)
        self.assertIsNotNone(eff[0]["chunk_before"])

    def test_cooling_period_entries_do_not_freeze_the_before_value(self):
        """冷却期会重复写入同样的 state；若只在上一条 changed 处取值，
        before 会停在更早的值上，于是把"没变"误报成"变了"。"""
        trace = [
            _trace_entry(0, 1024, changed=False),
            _trace_entry(8, 512, changed=True),
            _trace_entry(9, 512, changed=False),      # 冷却期
            _trace_entry(10, 512, changed=False),     # 冷却期
            _trace_entry(32, 512, changed=True),
        ]
        eff = M.action_effects(trace, [_pf(9, 512, 512)])
        self.assertEqual(eff[1]["chunk_before"], 512)
        self.assertEqual(eff[1]["kind"], M.ACTION_VALUE_UNCHANGED)

    def test_first_entry_changed_falls_back_to_init_value(self):
        trace = [_trace_entry(0, 512, changed=True, init_chunk=1024)]
        eff = M.action_effects(trace, [_pf(1, 512, 512)])
        self.assertEqual(eff[0]["chunk_before"], 1024)

    def test_non_numeric_step_does_not_raise(self):
        """step 若被写成字符串，不应抛 TypeError，也不应给出假比较结果。"""
        trace = [{"step": "8", "changed": True, "state": {"chunk_size": 512}}]
        eff = M.action_effects(trace, [_pf(9, 512, 512)])
        self.assertEqual(len(eff), 1)
        self.assertIsNone(eff[0]["next_prefill_step"])


class TestConcurrencyOf(unittest.TestCase):
    """臂名里本身就含 ``_c``（``static_chunk``），从左切会切错。"""

    def test_adaptive_with_repeat_suffix(self):
        self.assertEqual(M.concurrency_of("P1S3B-phase1_anchor-adaptive_c8-r2"), 8)

    def test_static_chunk_is_not_cut_inside_the_arm_name(self):
        self.assertEqual(M.concurrency_of("P1S3B-phase1_anchor-static_chunk_c4"), 4)
        # 阳性对照：从左切出来的第二个字段根本不是数字
        naive = "P1S3B-phase1_anchor-static_chunk_c4".split("_c")[1]
        self.assertEqual(naive, "hunk")
        self.assertIsNone(M.concurrency_of("P1S3B-phase1_anchor-static"))
        self.assertIsNone(M.concurrency_of("no-concurrency-here"))


class TestExpectedPrefillTokens(unittest.TestCase):
    def _write(self, tmp: Path, rows: list[dict]) -> Path:
        d = tmp / "run"
        d.mkdir(parents=True, exist_ok=True)
        with (d / "requests.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return d

    def test_sums_prompt_len(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._write(Path(td), [{"prompt_len": 2048}] * 8)
            self.assertEqual(M.expected_prefill_tokens(d), 16384.0)

    def test_missing_file_is_none_not_zero(self):
        """``None`` = 读不到；``0.0`` = "这个 run 没有 prefill 工作"。不可混。"""
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(M.expected_prefill_tokens(Path(td) / "nope"))

    def test_present_but_empty_is_none_not_zero(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._write(Path(td), [])
            self.assertIsNone(M.expected_prefill_tokens(d))

    def test_rows_without_prompt_len_are_not_counted_as_zero(self):
        with tempfile.TemporaryDirectory() as td:
            d = self._write(Path(td), [{"req_id": "r0"}, {"prompt_len": 2048}])
            self.assertEqual(M.expected_prefill_tokens(d), 2048.0)


class TestRealArtifacts(unittest.TestCase):
    """在真机上产出的 36 份 P1S3B 上复核分类结果。

    这些断言是**读数**，不是规格：它们钉住"这批产物在该分类下得到什么"，
    一旦解析口径或分类改了就应当一起复核，而不是让测试跟着改。
    """

    RAW = _ROOT / "results" / "raw"

    def _run(self, name: str) -> tuple[list[dict], dict]:
        d = self.RAW / name
        if not d.exists():
            raise unittest.SkipTest(f"缺少产物 {d}")
        rows, bad = M.load_telemetry(d)
        self.assertEqual(bad, 0, "telemetry 里有 JSON 坏行")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return rows, meta

    def _effective(self, name: str) -> tuple[list[dict], dict[str, int]]:
        rows, meta = self._run(name)
        pf = M.prefill_events(rows)
        eff = M.action_effects(meta.get("controller_trace") or [], pf)
        return pf, M.effect_tally(eff)

    def test_every_note_row_is_parseable(self):
        """全部 36 份产物里没有一行解析不出 prefill_tok。

        这一条同时是"别把不可解析行静默算作 0"的回归：若引擎改了日志格式，
        这里会立刻红，而不是悄悄让 prefill 事件数变成 0。
        """
        d = self.RAW
        if not d.exists():
            raise unittest.SkipTest("缺少 results/raw")
        total_unparsed = 0
        total_rows = 0
        for run in sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith("P1S3")):
            rows, _ = M.load_telemetry(run)
            total_rows += len(rows)
            total_unparsed += M.count_unparsed(rows)
        self.assertGreater(total_rows, 0, "被验的是空集——先确认输入非空")
        self.assertEqual(total_unparsed, 0)

    def test_c1_has_zero_actions(self):
        pf, tally = self._effective("P1S3B-phase1_anchor-adaptive_c1")
        self.assertEqual(sum(tally.values()), 0)
        self.assertEqual({p["chunk_size"] for p in pf}, {1024})

    def test_c2_tally(self):
        _, tally = self._effective("P1S3B-phase1_anchor-adaptive_c2")
        self.assertEqual(tally[M.ACTION_APPLIED], 3)
        self.assertEqual(tally[M.ACTION_SUPERSEDED], 5)
        self.assertEqual(tally[M.ACTION_NO_PREFILL_AFTER], 2)
        self.assertEqual(tally[M.ACTION_NOT_APPLIED], 0)

    def test_c4_tally(self):
        _, tally = self._effective("P1S3B-phase1_anchor-adaptive_c4")
        self.assertEqual(tally[M.ACTION_APPLIED], 1)
        self.assertEqual(tally[M.ACTION_SUPERSEDED], 3)
        self.assertEqual(tally[M.ACTION_NO_PREFILL_AFTER], 2)
        self.assertEqual(tally[M.ACTION_NOT_APPLIED], 0)

    def test_c8_tally_value_unchanged_and_applied(self):
        _, tally = self._effective("P1S3B-phase1_anchor-adaptive_c8")
        self.assertEqual(tally[M.ACTION_APPLIED], 1)
        self.assertEqual(tally[M.ACTION_VALUE_UNCHANGED], 5)

    def test_not_applied_never_occurs_in_this_batch(self):
        """**这一态目前没有真机样本。**

        因此它只由 ``TestSupersededVsNotApplied`` 的合成输入覆盖。结论里
        **不得**把它写成"已观测到的控制器缺陷"——它在这批产物里一次都没出现。
        """
        for name in ("P1S3B-phase1_anchor-adaptive_c1", "P1S3B-phase1_anchor-adaptive_c2",
                     "P1S3B-phase1_anchor-adaptive_c4", "P1S3B-phase1_anchor-adaptive_c8"):
            d = self.RAW / name
            if not d.exists():
                raise unittest.SkipTest(f"缺少产物 {d}")
            _, tally = self._effective(name)
            self.assertEqual(tally[M.ACTION_NOT_APPLIED], 0, name)

    def test_prefill_work_is_conserved_across_arms(self):
        """三臂四档的**实际 prefill 总 token 完全相同**，且等于请求侧求和。

        这是审计的自校验：它把"控制器把 prefill 重新分配了"与"控制器弄丢/重复了
        prefill 工作"分开。总和对不上就说明解析器丢行或重复计数了。
        """
        d = self.RAW
        if not d.exists():
            raise unittest.SkipTest("缺少 results/raw")
        checked = 0
        for run in sorted(p for p in d.iterdir() if p.is_dir() and p.name.startswith("P1S3B-")):
            if "-r1" in run.name or "-r2" in run.name:
                continue
            rows, _ = M.load_telemetry(run)
            got = sum(p["prefill_tok"] for p in M.prefill_events(rows))
            need = M.expected_prefill_tokens(run)
            self.assertIsNotNone(need, run.name)
            self.assertEqual(got, need, f"{run.name}: 实际 {got} vs 请求侧 {need}")
            self.assertEqual(got, 16384.0, run.name)
            checked += 1
        self.assertEqual(checked, 12, "三臂 × 四档 = 12 个基准 run")

    def test_prefill_event_counts_differ_across_arms(self):
        """**动作确实改变了 prefill 的节奏**——事件数就是它的直接读数。

        c8：控制器把 chunk 从 1024 收缩到 512 之后，同一批 token 被切成更多步
        （23 个事件 vs static_chunk 的 16 个）。若两边事件数相同，说明动作没有
        到达 prefill，那"TTFT 差异可归因于控制器"这句话就不成立。
        """
        def count(name: str) -> int:
            d = self.RAW / name
            if not d.exists():
                raise unittest.SkipTest(f"缺少产物 {d}")
            rows, _ = M.load_telemetry(d)
            return len(M.prefill_events(rows))

        self.assertEqual(count("P1S3B-phase1_anchor-static_chunk_c8"), 16)
        self.assertEqual(count("P1S3B-phase1_anchor-adaptive_c8"), 23)
        self.assertEqual(count("P1S3B-phase1_anchor-static_chunk_c1"), 16)
        self.assertEqual(count("P1S3B-phase1_anchor-adaptive_c1"), 16)


if __name__ == "__main__":
    unittest.main()
