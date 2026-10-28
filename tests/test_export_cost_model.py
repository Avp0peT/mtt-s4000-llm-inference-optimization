"""`export_cost_model.py` 的**导出期选择**必须有回归用例（否则只能靠事后手改 JSON）。

为什么要有这一条
----------------
`src/ascend_opt/cost_model.py` 解析模式**只认** `payload["default_mode"]`，配置里
**没有**这个键 ⇒ "这批实验到底跑 segmented 还是 lookup" **只**由模型文件的这一个字段决定。

实测踩到的静默通道：链里用 `heredoc` **事后改**这个字段，改失败（键名写错、文件没写进去、
`python3 - <<PY` 没执行）**不报错** —— 模型静默按 `segmented` 跑；更糟的是覆盖率闸门
自己显式传 `--replay-mode lookup`，**照样通过** ⇒ 闸门与实跑口径脱节，
而产物（`MEASURED`、戳齐、账目误差 0）**看不出来**。

⇒ 把选择变成**导出时的入参**（默认值 = 旧行为，故旧路径逐字节不变），并加一条
**真能触发**的语义守卫。

关于那条守卫为什么这么写
------------------------
键 `_kb()` 是 `b{batch}_pf{prefill}_kv{kv}_nf{nf}`，**不含 `n_new_mode`**
（这正是 v5/v6 "同形状同键、可证伪对比"的前提）⇒ 单看键**分不出**
"每条序列各做 `prefill_tok` 个"与"1 条做 `prefill_tok` 个、其余各 1 个"。
所以必须回**产物**里查 `shape.n_new_mode`，否则模型文件里那个字符串就是语义的唯一证据
—— 而 v5 正是这么错的，且无任何机制能发现。
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

import export_cost_model  # noqa: E402


def _records() -> list[dict[str, object]]:
    """两个 batch 各 4 点、秩 3（否则 `_ols` 退化，用例会在无关的地方失败）。"""
    out: list[dict[str, object]] = []
    pts = [(0, 512), (512, 512), (0, 1024), (512, 1024)]
    for b in (2, 8):
        for pf, kv in pts:
            out.append({"batch": b, "prefill_tok": pf, "kv_len": kv,
                        "actual_ms": 3.0 + 0.42 * pf + 0.006 * kv + 0.5 * b,
                        "session": "s1"})
    return out


def _write_eval(path: Path) -> None:
    path.write_text(json.dumps({
        "protocol_version": "cost-calib-v6",
        "records": _records(),
        "contexts": {"① 形状留出": {"分段模型(按batch)": {"underest_p95": 0.0612}}},
        "verdict": {"model_holds": False, "reasons": ["低估 p95 6.12% > 5%"]},
    }, ensure_ascii=False), encoding="utf-8", newline="\n")


def _write_artifact(path: Path, *, n_new_mode: str | None) -> None:
    def shape(b: int, pf: int, kv: int) -> dict[str, object]:
        s: dict[str, object] = {"group": "mixed_serving" if pf else "pure_decode",
                               "batch": b, "prefill_tok": pf, "kv_len": kv, "n_forwards": 1}
        if n_new_mode is not None:
            s["n_new_mode"] = n_new_mode
        return s
    path.write_text(json.dumps({
        "protocol_version": "cost-calib-v6",
        "grid_sha256": "a" * 64,
        "fingerprint": {"expected": {"device": "MTT S4000", "precision": "fp16"}},
        "points": [{"shape": shape(b, pf, kv), "key": f"b{b}_pf{pf}_kv{kv}_nf1"}
                   for b in (2, 8) for pf, kv in ((0, 512), (512, 512), (0, 1024), (512, 1024))],
    }, ensure_ascii=False), encoding="utf-8", newline="\n")


def _export(tmp: Path, extra: list[str], *, n_new_mode: str | None = "uniform") -> tuple[int, str, str]:
    ev, art, out = tmp / "eval.json", tmp / "art.json", tmp / "model.json"
    _write_eval(ev)
    _write_artifact(art, n_new_mode=n_new_mode)
    so, se = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
        rc = export_cost_model.main(["--eval", str(ev), "--artifacts", str(art),
                                     "--out", str(out), *extra])
    return rc, so.getvalue(), se.getvalue()


class TestExportCostModelDefaults(unittest.TestCase):
    def test_default_keeps_old_behaviour(self):
        """不传任何新参数 ⇒ 旧行为（`segmented`），且**不写入**新键（旧产物逐字节不变）。"""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rc, _, err = _export(tmp, [])
            self.assertEqual(rc, 0, err)
            m = json.loads((tmp / "model.json").read_text(encoding="utf-8"))
        self.assertEqual(m["default_mode"], "segmented")
        self.assertNotIn("lookup_bucket", m, "没传桶宽却写了这个键 ⇒ 旧产物字节会变")
        self.assertNotIn("mixed_shape_mode", m, "没传口径却写了这个键 ⇒ 旧产物字节会变")
        self.assertTrue(m["lookup_table"], "查表应总有内容（有记录就非空）")
        self.assertEqual(m["safety_margin"], 0.0612)
        self.assertFalse(m["validation"]["model_holds"], "拟合未过判据这件事必须如实记下")


class TestExportCostModelFlags(unittest.TestCase):
    def test_lookup_mode_and_buckets_land_in_file(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            rc, out, err = _export(tmp, ["--default-mode", "lookup",
                                         "--lookup-bucket-kv-step", "256",
                                         "--mixed-shape-mode", "real_mixed"],
                                   n_new_mode="real_mixed")
            self.assertEqual(rc, 0, err)
            m = json.loads((tmp / "model.json").read_text(encoding="utf-8"))
        self.assertEqual(m["default_mode"], "lookup")
        self.assertEqual(m["lookup_bucket"], {"pf_step": 0, "kv_step": 256})
        self.assertEqual(m["mixed_shape_mode"], "real_mixed")
        # 打印里必须能看见"生效的模式" —— 否则日志没法复核这批跑的到底是哪种
        self.assertIn("default_mode=lookup", out)

    def test_negative_bucket_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _export(Path(td), ["--lookup-bucket-kv-step", "-256"])
        self.assertNotEqual(rc, 0, "负桶宽应被拒")
        self.assertIn("桶宽", err)


class TestMixedShapeModeSelfEvidence(unittest.TestCase):
    def test_claiming_real_mixed_without_evidence_is_rejected(self):
        """★ 核心：声称 `real_mixed` 而产物里记的是 `uniform` ⇒ **必须拒绝导出**。

        这一条就是"v5 错了却没人发现"的反面对照：若只把字符串写进模型文件、
        不去产物里核对，同样的错会被再犯一次，而且**不会报错**。
        """
        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _export(Path(td), ["--mixed-shape-mode", "real_mixed"],
                                 n_new_mode="uniform")
        self.assertNotEqual(rc, 0, "语义声明与产物不符却导出了 ⇒ 守卫失效")
        self.assertIn("无据", err)
        self.assertIn("uniform", err, "报错里要给出产物实际记的是什么")

    def test_claiming_real_mixed_on_old_artifacts_is_rejected(self):
        """旧产物（无 `n_new_mode` 字段）**不得**被用来声称 `real_mixed`。"""
        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _export(Path(td), ["--mixed-shape-mode", "real_mixed"],
                                 n_new_mode=None)
        self.assertNotEqual(rc, 0, "旧口径产物被用来声称 real_mixed ⇒ 守卫失效")
        self.assertIn("旧口径", err)

    def test_no_claim_no_check(self):
        """不声称口径时**不做**这个检查（否则会挡住历史产物的正常重导出）。"""
        with tempfile.TemporaryDirectory() as td:
            rc, _, err = _export(Path(td), [], n_new_mode=None)
        self.assertEqual(rc, 0, err)


if __name__ == "__main__":
    unittest.main()
