"""第 14 步分析脚本的**一致性闸门**必须有回归用例锁住（双向）。

为什么要有这一条
----------------
实测踩到：`analyze_step14.py` 的"实际生效的判据 vs 冻结的 `decode_p90`"闸门原先
**不豁免** `controller.enabled == false` 的臂，而 `tests/test_step14_configs.py`（46–48 行）
**豁免**了它 ⇒ **两个守卫口径不一致** ⇒ 任何含 `static` 臂的批次必出**假警**
（"⚠️ 与第 09 步 §⑥ 冻结的 decode_p90 不一致"）。

⚠️ 这类假警的害处不是"多一条噪声"：它与**真回归**（自适应配置漏声明 `signal_criterion`）
共用同一句告警文本 ⇒ 告警一旦常亮就不再被读 ⇒ 真回归会**静默通过**。

`28fcdc1` 修了实现，但**当时没有用例** ⇒ 下次重构会无声退回。

两个方向都必须测
----------------
* **方向 A（不报假警）**：`static` 臂（`enabled=false`、产物里 `criterion=None`）+ 自适应臂
  （`decode_p90`）⇒ **不得**出现告警；
* **方向 B（真警必须报）**：自适应臂的 `criterion` 缺失（`None`）⇒ **必须**出现告警。

只测方向 A 是不够的：把闸门整条删掉也能通过方向 A ⇒ 那条断言其实没在测东西。
方向 B 才是"证明它真的在测"的那一半（这是加守卫的硬要求）。
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "scripts"), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analyze_step14  # noqa: E402

#: 告警的唯一可识别片段（改闸门文本时这里要一起改，否则用例会失去意义）
_ALARM = "与第 09 步"


def _write_run(root: Path, *, dirname: str, run_id: str, enabled: bool,
               criterion: str | None, commit: str = "deadbeef",
               dirty: bool = False, shadow: bool = False) -> None:
    """造一个**最小可被 load_runs 读入**的产物目录（只写闸门需要的那几个字段）。"""
    d = root / "results" / "raw" / dirname
    d.mkdir(parents=True, exist_ok=True)
    ctrl: dict[str, object] = {"kind": "threshold", "enabled": enabled, "shadow": shadow}
    if criterion is not None:
        # ⚠️ 有意不写 None：产物里"没有这个字段"与"字段是 None"对闸门是同一件事，
        # 但前者才是真机上的实际形状（`describe()` 当年还没记这个键）。
        ctrl["signal_criterion"] = criterion
    meta = {
        "data_label": "MEASURED",
        "reproducibility": {"git_commit": commit, "git_dirty": dirty},
        "config": {"controller": ctrl},
        "summary": {"run_id": run_id},
        "controller_effect": {},
        "controller_trace": [],
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                 encoding="utf-8", newline="\n")


def _run_gate(tmp: Path, tag: str) -> str:
    """把模块的 _ROOT 指到临时树（用例**不得**读到真产物），跑 main，回 stdout。"""
    old = analyze_step14._ROOT
    analyze_step14._ROOT = tmp
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = analyze_step14.main(["--tag", tag])
    finally:
        analyze_step14._ROOT = old
    assert rc == 0, f"main 返回 {rc}（应 0；用例造的是完整的最小产物）"
    return buf.getvalue()


class TestCriterionGateExemptsDisabledControllers(unittest.TestCase):
    def test_direction_A_no_false_alarm_with_static_arm(self):
        """static 臂在场时**不得**告警（这正是 `28fcdc1` 修掉的那个假警）。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _write_run(tmp, dirname="T9-wl-fixed_static", run_id="T9-wl-fixed_static",
                       enabled=False, criterion=None)
            _write_run(tmp, dirname="T9-wl-fixed_threshold", run_id="T9-wl-fixed_threshold",
                       enabled=True, criterion="decode_p90")
            out = _run_gate(tmp, "T9")
        self.assertNotIn(_ALARM, out,
                         "含 static 臂的批次出了假警 ⇒ 豁免失效（两个守卫口径再次不一致）")
        # 顺带确认走的**确实是**豁免路径，而不是"两个 run 都没被读进来"
        self.assertIn("已豁免控制器关闭的臂 1 个", out,
                      "豁免计数不是 1 ⇒ 用例的前置条件没建立（产物字段名或 run_id 格式变了？）")

    def test_direction_B_real_alarm_still_fires(self):
        """自适应臂**没声明**判据时必须告警 —— 这是方向 A 的反向对照。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _write_run(tmp, dirname="T8-wl-fixed_static", run_id="T8-wl-fixed_static",
                       enabled=False, criterion=None)
            _write_run(tmp, dirname="T8-wl-fixed_cost", run_id="T8-wl-fixed_cost",
                       enabled=True, criterion=None)  # ← 真回归：漏声明，加载成默认 tpot_p90
            out = _run_gate(tmp, "T8")
        self.assertIn(_ALARM, out,
                      "自适应臂漏声明判据却没告警 ⇒ 闸门没在测东西（真回归会静默通过）")
        self.assertIn("未声明", out, "告警里没有把'未声明'与'声明了别的值'分开")

    def test_direction_B_prime_other_declared_value_also_fires(self):
        """声明成**别的值**（默认 `tpot_p90`）同样必须告警 —— 与"未声明"分开测。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            _write_run(tmp, dirname="T7-wl-fixed_threshold", run_id="T7-wl-fixed_threshold",
                       enabled=True, criterion="tpot_p90")
            out = _run_gate(tmp, "T7")
        self.assertIn(_ALARM, out, "声明了非冻结判据却没告警")


class TestGateDoesNotAlarmOnCleanBatches(unittest.TestCase):
    def test_all_enabled_declare_frozen_criterion(self):
        """全部自适应臂都声明 `decode_p90` ⇒ 干净批次，不得有任何告警。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for arm in ("threshold", "cost", "slo_weave"):
                _write_run(tmp, dirname=f"T6-wl-fixed_{arm}", run_id=f"T6-wl-fixed_{arm}",
                           enabled=True, criterion="decode_p90")
            out = _run_gate(tmp, "T6")
        self.assertNotIn(_ALARM, out, "干净批次被误报")


if __name__ == "__main__":
    unittest.main()
