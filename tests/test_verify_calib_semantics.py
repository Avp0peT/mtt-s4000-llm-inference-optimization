"""`verify_calib_semantics.py` 的回归用例。

这个脚本是"口径改了、真的生效了吗"的**验收器**，所以它自己更需要被验收 ——
否则它会像它要替代的那段 heredoc 一样，只打印、不判。
用例的重点是**四个方向**都要对：
① 真改对了 ⇒ 通过；② 对照组漂了 ⇒ 处理组**必须**降级成未判；
③ 没改（比值≈1）⇒ **必须**判失败；④ 语义无据/前提不成立 ⇒ **必须**判失败。
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

import verify_calib_semantics as V  # noqa: E402

_BS = (2, 4, 8)
_PFS = (0, 512, 1024)
_KVS = (512, 1024)


def _cost(b: int, pf: int, kv: int, *, mixed_semantics: bool) -> float:
    """人造成本：`pf=0` 时两种口径同值；`pf>0` 时 uniform 是 `b*pf`、real_mixed 是 `pf+(b-1)`。"""
    n_new = (1 if pf == 0 else (pf + (b - 1) if mixed_semantics else b * pf))
    return 3.0 + 0.42 * n_new + 0.006 * kv


def _write_model(path: Path, *, mixed_semantics: bool, drift_pf0: float = 1.0) -> None:
    tbl: dict[str, dict[str, float]] = {}
    for b in _BS:
        for pf in _PFS:
            for kv in _KVS:
                v = _cost(b, pf, kv, mixed_semantics=mixed_semantics)
                if pf == 0:
                    v *= drift_pf0        # 对照组被人为扰动（模拟机器漂移）
                tbl[f"b{b}_pf{pf}_kv{kv}"] = {"median": v, "max": v, "n": 1}
    path.write_text(json.dumps({"lookup_table": tbl, "default_mode": "lookup"},
                               ensure_ascii=False), encoding="utf-8", newline="\n")


def _write_artifact(path: Path, *, real_mixed_on_mixed: bool, has_field: bool = True) -> None:
    pts = []
    for b in _BS:
        for pf in _PFS:
            for kv in _KVS:
                shape: dict[str, object] = {"batch": b, "prefill_tok": pf, "kv_len": kv,
                                            "group": "mixed_serving" if pf else "pure_decode"}
                if has_field:
                    shape["n_new_mode"] = ("real_mixed" if (pf and b > 1 and real_mixed_on_mixed)
                                           else "uniform")
                pts.append({"shape": shape, "key": f"b{b}_pf{pf}_kv{kv}"})
    path.write_text(json.dumps({"points": pts}, ensure_ascii=False),
                    encoding="utf-8", newline="\n")


def _run(tmp: Path, *, new_mixed_semantics: bool = True, drift_pf0: float = 1.0,
         new_artifact_real_mixed: bool = True, old_artifact_real_mixed: bool = False,
         old_has_field: bool = True, new_has_field: bool = True,
         with_artifacts: bool = True) -> tuple[int, str, dict]:
    old_m, new_m = tmp / "old.json", tmp / "new.json"
    old_a, new_a = tmp / "old_a.json", tmp / "new_a.json"
    _write_model(old_m, mixed_semantics=False, drift_pf0=drift_pf0)
    # ⚠️ 漂移**只**加在旧模型上（模拟"基线那次是在不同机器状态下测的"）。
    # 若两边都乘，对照组比值恒为 1 ⇒ C1 永远通过 ⇒ 守门形同虚设（用例会静默失去意义）。
    _write_model(new_m, mixed_semantics=new_mixed_semantics, drift_pf0=1.0)
    _write_artifact(old_a, real_mixed_on_mixed=old_artifact_real_mixed, has_field=old_has_field)
    _write_artifact(new_a, real_mixed_on_mixed=new_artifact_real_mixed, has_field=new_has_field)
    jp = tmp / "out.json"
    argv = ["--old-model", str(old_m), "--new-model", str(new_m), "--json", str(jp)]
    if with_artifacts:
        argv += ["--old-artifacts", str(old_a), "--new-artifacts", str(new_a)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = V.main(argv)
    return rc, buf.getvalue(), json.loads(jp.read_text(encoding="utf-8"))


class TestRealFix(unittest.TestCase):
    def test_correct_fix_passes_all_three(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out, v = _run(Path(td))
        self.assertEqual(rc, 0, out)
        self.assertEqual(v["verdict"], "pass")
        for c in ("C1_control_ratio", "C2_mixed_ratio", "C3_semantics_self_evidence"):
            self.assertTrue(v["checks"][c]["ok"], f"{c} 应为通过：{out}")
        # 处理组比值应当≈1/batch（这里 cross-check 一下量级，避免"过了但没意义"）
        self.assertLess(v["checks"]["C2_mixed_ratio"]["median"], 0.35, out)
        self.assertIn("✓ 口径生效", out)


class TestControlIsTheGuard(unittest.TestCase):
    def test_drift_breaks_control_and_suppresses_treatment(self):
        """★ C1 是守门人：对照组漂了 ⇒ 处理组**必须**降级为未判，不得单独解释。"""
        with tempfile.TemporaryDirectory() as td:
            rc, out, v = _run(Path(td), drift_pf0=1.20)
        self.assertEqual(rc, 1, out)
        self.assertFalse(v["checks"]["C1_control_ratio"]["ok"], out)
        self.assertIsNone(v["checks"]["C2_mixed_ratio"]["ok"],
                          "C1 未过却仍对 C2 下了结论 ⇒ 守门失效")
        self.assertEqual(v["checks"]["C2_mixed_ratio"]["reason"], "C1 未通过")
        self.assertIn("不得**单独解释", out)


class TestFalsifiability(unittest.TestCase):
    def test_no_change_at_all_fails(self):
        """没改（比值≈1）⇒ 必须判失败 —— 这是"改错了/没生效"要找的那个信号。"""
        with tempfile.TemporaryDirectory() as td:
            rc, out, v = _run(Path(td), new_mixed_semantics=False,
                              new_artifact_real_mixed=False)
        self.assertEqual(rc, 1, out)
        self.assertFalse(v["checks"]["C2_mixed_ratio"]["ok"], out)
        self.assertIn("没有生效", out)

    def test_missing_self_evidence_fails(self):
        """模型文件声称改了，但产物里找不到 `real_mixed` ⇒ 语义无据 ⇒ 失败。"""
        with tempfile.TemporaryDirectory() as td:
            rc, out, v = _run(Path(td), new_artifact_real_mixed=False)
        self.assertEqual(rc, 1, out)
        self.assertFalse(v["checks"]["C3_semantics_self_evidence"]["ok"], out)
        self.assertIn("无据", out)

    def test_old_artifacts_already_real_mixed_invalidates_premise(self):
        """旧产物**也**标了 `real_mixed` ⇒ "v5 口径错"这个前提不成立，必须报出来。"""
        with tempfile.TemporaryDirectory() as td:
            rc, out, v = _run(Path(td), old_artifact_real_mixed=True)
        self.assertEqual(rc, 1, out)
        self.assertIn("前提不成立", out)

    def test_old_artifacts_without_field_are_reported_as_old_caliber(self):
        """旧产物没有该字段（v4 及更早）⇒ 报成"旧口径"，**不**与"记了 uniform"混为一谈。"""
        with tempfile.TemporaryDirectory() as td:
            rc, out, v = _run(Path(td), old_has_field=False)
        self.assertEqual(rc, 0, out)
        self.assertIn("无此字段=旧口径", out)


class TestTriState(unittest.TestCase):
    def test_insufficient_common_keys_is_undetermined_not_pass(self):
        """同键太少 ⇒ 退出码 3（未判），**不得**混成 0（通过）。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "old.json").write_text(json.dumps(
                {"lookup_table": {"b2_pf0_kv512": {"max": 10.0}}}), encoding="utf-8")
            (tmp / "new.json").write_text(json.dumps(
                {"lookup_table": {"b2_pf0_kv512": {"max": 10.5}}}), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = V.main(["--old-model", str(tmp / "old.json"),
                             "--new-model", str(tmp / "new.json")])
            out = buf.getvalue()
        self.assertEqual(rc, 3, out)
        self.assertIn("undetermined", out)
        self.assertIn("不把'测不出'当'通过'", out)


class TestInputErrors(unittest.TestCase):
    def test_model_without_lookup_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            (tmp / "m.json").write_text(json.dumps({"coefficients": {}}), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = V.main(["--old-model", str(tmp / "m.json"),
                             "--new-model", str(tmp / "m.json")])
        self.assertEqual(rc, 2, "无查表的模型应判输入错（2），不是「通过」")

    def test_key_without_nf_not_strictly_required(self):
        """键解析：`_kb()` 不含 `nf`，故正则**不得**要求它（否则全部键被丢弃 ⇒ 静默恒未判）。"""
        self.assertIsNotNone(V._parse_key("b8_pf512_kv1024"))
        self.assertEqual(V._parse_key("b8_pf512_kv1024")["b"], 8)
        self.assertIsNone(V._parse_key("b8_pf512_kv1024_nf1"))


if __name__ == "__main__":
    unittest.main()
