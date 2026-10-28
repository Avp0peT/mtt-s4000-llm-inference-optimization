"""`controller.candidate_chunks`：**评估点**与**可动范围**是两件事。

为什么需要这个键
----------------
证据生成侧原先只取 `{chunk_size_min, chunk_size_max}` **两个点**
（`benchmark/arm_factory.py`）⇒ **改下限不会自动新增中间点**：
把 `chunk_size_min` 改成 128 只会得到 `{128, 2048}`，**不会**产生 192/256。
⇒ 要做"小 chunk 能不能被选中并执行"的验证，必须**显式**给出评估点。

分工（写死在用例里，避免以后混淆）
----------------------------------
* `chunk_size_min/max` = **可动范围**；运行期上限的唯一权威仍是 `ChunkLimits`（由它派生）。
* `candidate_chunks` = 在范围内**挑哪几个点去评估**。
⇒ 想让 128 真的可动，必须**同时**把 `chunk_size_min` 降到 128；
否则候选会被夹回下限 —— 而**静默夹取**会让"我改了候选集"这件事**没有发生**，
两次对照的差异被悄悄抹掉（"改了个不生效的参数"）。
所以越界必须**报错**，用例专门钉这一条。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ascend_opt.controller import ControllerConfig  # noqa: E402
from run_benchmark import load_config  # noqa: E402

_BASE = _ROOT / "configs" / "mthreads" / "phase1_step14b" / "fixed_cost.yaml"


def _cfg(tmp: Path, *, extra: str = "", model: str = "model_v4.json") -> Path:
    src = _BASE.read_text(encoding="utf-8")
    src = src.replace("cost_model_path: configs/cost_calib/model_v6.json",
                      f"cost_model_path: configs/cost_calib/{model}")
    src = src.replace("cost_model_fingerprint_source: observed",
                      "cost_model_fingerprint_source: declared")
    if extra:
        anchor = "  signal_criterion: decode_p90\n"
        assert anchor in src, "锚点变了 ⇒ 用例会静默失效"
        src = src.replace(anchor, anchor + extra)
    p = tmp / "cfg.yaml"
    p.write_text(src, encoding="utf-8", newline="\n")
    return p


def _candidates(path: Path) -> tuple[int, ...]:
    from arm_factory import build_arm
    return tuple(build_arm(load_config(path)).config.candidate_chunks)


class TestDefaultKeepsOldBehaviour(unittest.TestCase):
    def test_none_yields_only_min_and_max(self):
        with tempfile.TemporaryDirectory() as td:
            got = _candidates(_cfg(Path(td)))
        self.assertEqual(got, (512, 2048),
                         "默认必须是旧行为（只取 min/max 两个点）—— 否则既有批次不可比")

    def test_min_change_alone_still_yields_only_two_points(self):
        """★ 钉住那个被忽略的事实：只改下限**不会**新增中间点。"""
        with tempfile.TemporaryDirectory() as td:
            got = _candidates(_cfg(Path(td), extra="  chunk_size_min: 128\n"))
        self.assertEqual(got, (128, 2048), "只改 min 却多出了点 ⇒ 说明实现变了，用例要更新")


class TestExplicitList(unittest.TestCase):
    def test_explicit_list_is_used_verbatim(self):
        with tempfile.TemporaryDirectory() as td:
            got = _candidates(_cfg(Path(td),
                                   extra="  chunk_size_min: 128\n"
                                         "  candidate_chunks: [128, 256, 512, 2048]\n"))
        self.assertEqual(got, (128, 256, 512, 2048))

    def test_dedup_and_sort(self):
        with tempfile.TemporaryDirectory() as td:
            got = _candidates(_cfg(Path(td),
                                   extra="  chunk_size_min: 128\n"
                                         "  candidate_chunks: [512, 128, 512, 256]\n"))
        self.assertEqual(got, (128, 256, 512), "应去重并排序（否则候选顺序会变成隐变量）")


class TestOutOfRangeIsLoud(unittest.TestCase):
    def test_candidate_below_min_is_rejected(self):
        """★ 越界必须**报错**，不许静默夹取（夹取 = "改了候选集"没发生）。"""
        with tempfile.TemporaryDirectory() as td:
            path = _cfg(Path(td), extra="  candidate_chunks: [128, 512, 2048]\n")
            with self.assertRaises(ValueError) as cm:
                _candidates(path)
        msg = str(cm.exception)
        self.assertIn("越界", msg)
        self.assertIn("chunk_size_min", msg, "报错必须指明修法（同时降下限），否则下一个人只会重试")

    def test_candidate_above_max_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = _cfg(Path(td), extra="  candidate_chunks: [512, 4096]\n")
            with self.assertRaises(ValueError):
                _candidates(path)

    def test_in_range_list_after_lowering_min_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            got = _candidates(_cfg(Path(td),
                                   extra="  chunk_size_min: 128\n"
                                         "  candidate_chunks: [128, 512, 2048]\n"))
        self.assertEqual(got, (128, 512, 2048))


class TestRecordedInArtifact(unittest.TestCase):
    def test_describe_records_none_by_default(self):
        self.assertIsNone(ControllerConfig().describe()["candidate_chunks"])

    def test_describe_records_the_list(self):
        d = ControllerConfig(candidate_chunks=[128, 256, 512, 2048]).describe()
        self.assertEqual(d["candidate_chunks"], [128, 256, 512, 2048],
                         "候选列表是 D1 两次对照的**唯一变量** ⇒ 漏记则变量不可核")


if __name__ == "__main__":
    unittest.main()
