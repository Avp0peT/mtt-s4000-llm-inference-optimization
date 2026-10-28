"""检查器（``inspect_step3_runs.py``）离散度判据的守卫测试。

这里要钉住的是一个**真实漏判**，而不是假想的边界：原版判据只查 TTFT、且只查
稳健 CV，于是漏掉了矩阵里最该看的那条——

    static_c8 · stall p99 = 479.49 / 458.41 / 458.49
    稳健CV = 0.0002（判据说"极稳"）   普通CV = 0.0213（首点离群 4.6%）

原因是稳健 CV 里 MAD 取的是"偏离中位数"的**中位数**，而 n=3 时**一个**离群点
不改变这个中位数，于是离群被抹平。只报稳健 CV 会把"有一个离群点"说成"很稳"，
方向恰好相反。

配套的反向对照：把同一个序列里的离群点抹平，``masked`` 必须**不再**出现。
没有这条对照，判据可能只是在无脑报警——而无脑报警的判据和没有判据一样没用。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    if not path.exists():
        raise unittest.SkipTest(f"缺少脚本 {path}")
    spec = importlib.util.spec_from_file_location(f"_insp_{name}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mk_run(
    root: Path,
    name: str,
    *,
    ttft: float = 600.0,
    tpot: float = 60.0,
    stall_p99: float = 60.0,
    label: str = "MEASURED",
) -> None:
    """造一个最小可被检查器识别的 run 目录（顶层 meta.json 即可被发现）。"""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_label": label,
        "summary": {
            "ttft": {"p50": ttft},
            "tpot": {"p50": tpot},
            "stall": {"p50": stall_p99 * 0.99, "p99": stall_p99},
        },
        "context": {"started_at": "2026-09-23T00:00:00"},
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _run_inspector(mod, raw_dir: Path, *extra: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = mod.main([str(raw_dir), *extra])
    return rc, buf.getvalue()


class TestPlainCV(unittest.TestCase):
    """普通 CV 只作对照量，但它必须算对——它是"离群被抹平"这一判据的分子。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    def test_none_when_fewer_than_two(self):
        self.assertIsNone(self.mod._plain_cv([1.0]))
        self.assertIsNone(self.mod._plain_cv([]))

    def test_zero_when_identical(self):
        cv = self.mod._plain_cv([100.0, 100.0, 100.0])
        self.assertIsNotNone(cv)
        self.assertAlmostEqual(cv, 0.0, places=12)

    def test_none_when_mean_is_zero(self):
        self.assertIsNone(self.mod._plain_cv([-1.0, 1.0]))

    def test_outlier_makes_plain_cv_much_larger_than_robust(self):
        vals = [479.49, 458.41, 458.49]
        cp = self.mod._plain_cv(vals)
        cr = self.mod._robust_cv(vals)
        self.assertIsNotNone(cp)
        self.assertIsNotNone(cr)
        # 实测量级：普通 CV 比稳健 CV 大两个数量级
        self.assertGreater(cp / cr, 10.0)


class TestMaskedOutlierGuard(unittest.TestCase):
    """判据的**正向对照**：真出现被抹平的离群点时，必须报警。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    def test_fires_on_real_static_c8_pattern(self):
        # 逐字抄自真机产物（static_c8 的三次 stall p99）
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, s in enumerate((479.49, 458.41, 458.49)):
                suffix = "" if i == 0 else f"-r{i}"
                _mk_run(root, f"T-x-static_c8{suffix}", stall_p99=s)
            rc, out = _run_inspector(self.mod, root, "--tag", "T-x")
        self.assertEqual(rc, 0)
        self.assertIn("单点离群被 MAD 抹平", out)
        self.assertIn("stall p99", out)
        # 报警文案必须把原始值带上，否则读的人无法自己复核
        self.assertIn("479.49", out)
        self.assertIn("458.41", out)

    def test_does_not_fire_when_the_outlier_is_removed(self):
        """**反向对照**：抹掉离群点，判据必须安静。

        若这一条不过，说明上一条的报警与"有没有离群点"无关——那判据就是噪声。
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, s in enumerate((458.41, 458.49, 458.45)):
                suffix = "" if i == 0 else f"-r{i}"
                _mk_run(root, f"T-y-static_c8{suffix}", stall_p99=s)
            rc, out = _run_inspector(self.mod, root, "--tag", "T-y")
        self.assertEqual(rc, 0)
        self.assertNotIn("单点离群被 MAD 抹平", out)

    def test_a_large_but_consistent_stall_is_not_flagged(self):
        """绝对值大 ≠ 可疑。判据看的是**离散度**，不是量级。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, s in enumerate((5000.0, 5001.0, 4999.0)):
                suffix = "" if i == 0 else f"-r{i}"
                _mk_run(root, f"T-z-static_c8{suffix}", stall_p99=s)
            rc, out = _run_inspector(self.mod, root, "--tag", "T-z")
        self.assertEqual(rc, 0)
        self.assertNotIn("单点离群被 MAD 抹平", out)


class TestGlobalVersusPerCell(unittest.TestCase):
    """离散度极小必须是**整体现象**才成立；逐格罗列会退化成"全部可疑"。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    def test_all_tiny_is_reported_as_global(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for cell, base in (("c1", 600.0), ("c2", 800.0), ("c4", 960.0)):
                for i in range(3):
                    suffix = "" if i == 0 else f"-r{i}"
                    _mk_run(
                        root,
                        f"T-g-a_{cell}{suffix}",
                        ttft=base + i * 0.01,
                        tpot=60.0 + i * 0.01,
                        stall_p99=60.0 + i * 0.01,
                    )
            rc, out = _run_inspector(self.mod, root, "--tag", "T-g")
        self.assertEqual(rc, 0)
        self.assertIn("整体现象", out)
        # 计数行要显形，读的人才知道"整体"到什么程度
        self.assertIn("/9", out)  # 3 格 × 3 指标
        # 噪声底必须逐指标给，而不是给一个笼统的数
        for metric in ("TTFT p50", "TPOT p50", "stall p99"):
            self.assertIn(metric, out)

    def test_single_cell_with_large_dispersion_is_visible_in_raw_values(self):
        """只有一格散得很大时，原始值表里必须能直接看出来（不靠 CV 判据）。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, t in enumerate((600.0, 601.0, 599.0)):
                suffix = "" if i == 0 else f"-r{i}"
                _mk_run(root, f"T-h-a_c1{suffix}", ttft=t)
            rc, out = _run_inspector(self.mod, root, "--tag", "T-h")
        self.assertEqual(rc, 0)
        self.assertIn("599.00", out)
        self.assertIn("601.00", out)


class TestDiscoveryRules(unittest.TestCase):
    """发现规则本身也是判据的一部分：漏发现 = 悄悄少算一格。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    def test_underscore_dirs_are_invisible(self):
        # 这里**不能**带 --tag：带了之后 _pilot 也会被名字过滤掉，
        # 于是即使"下划线目录不可见"这条规则被删掉，断言照样通过——
        # 那验的就不是这条规则了。
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_run(root, "T-i-a_c1")
            _mk_run(root, "_pilot")  # 顶层以 _ 开头 → 必须被跳过
            rc, out = _run_inspector(self.mod, root)
        self.assertEqual(rc, 0)
        self.assertIn("T-i-a_c1", out)
        self.assertNotIn("_pilot", out)

    def test_dir_without_meta_json_is_invisible(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_run(root, "T-j-a_c1")
            (root / "T-j-orphan").mkdir()
            rc, out = _run_inspector(self.mod, root, "--tag", "T-j")
        self.assertEqual(rc, 0)
        self.assertNotIn("T-j-orphan", out)

    def test_no_runs_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            rc, _ = _run_inspector(self.mod, Path(td), "--tag", "T-none")
        self.assertEqual(rc, 2)


class TestLabelDisclosure(unittest.TestCase):
    """非 MEASURED 的 run 在场时必须显形——否则聚合会把它们当结论。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    def test_non_measured_label_is_warned(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_run(root, "T-k-a_c1", label="MEASURED")
            _mk_run(root, "T-k-b_c1-r1", label="TEST_DATA")
            rc, out = _run_inspector(self.mod, root, "--tag", "T-k")
        self.assertEqual(rc, 0)
        self.assertIn("TEST_DATA", out)
        self.assertIn("按标签过滤", out)


def _mk_run_with_snapshot(root: Path, name: str, cfg: dict, *, label: str = "MEASURED") -> None:
    """造一个带 ``config.snapshot.yaml`` 的 run（``--verify-config`` 的输入）。"""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_label": label,
        "summary": {},
        "context": {"started_at": "2026-09-23T00:00:00"},
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _mk_cfg_dir(path: Path, cell: str, cfg: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{cell}.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _run_inspector_both(mod, raw_dir: Path, *extra: str) -> tuple[int, str, str]:
    """同 :func:`_run_inspector`，但**同时**收 stderr——拒绝类的判据只印在 stderr 上。"""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = mod.main([str(raw_dir), *extra])
    return rc, out.getvalue(), err.getvalue()


#: 一份最小的"当前版"配置：两层嵌套 + 一个非整数，足够覆盖展平与取值比较。
_REAL_CFG = {
    "run": {"repeats": 3, "slo": {"tpot_ms": 100}},
    "controller": {"target_tpot_ms": 120.0, "hysteresis": 0.08},
}


class TestCellName(unittest.TestCase):
    """格名是 run → 配置文件的**唯一**桥梁；取错就会比到另一份文件上，且不报错。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    def test_strips_repeat_suffix(self):
        self.assertEqual(self.mod.cell_of("P1S3D-phase1_anchor-adaptive_c4-r2"), "adaptive_c4")

    def test_keeps_underscored_arm(self):
        self.assertEqual(
            self.mod.cell_of("P1S3D-phase1_anchor-static_chunk_c8"), "static_chunk_c8"
        )

    def test_first_repeat_has_no_suffix(self):
        self.assertEqual(self.mod.cell_of("P1S3D-phase1_anchor-static_c1"), "static_c1")

    def test_none_when_no_separator(self):
        # 取不出必须返回 None，**不许**猜一个默认格名：猜错会静默比到别的配置文件，
        # 而两条分支（"比中了另一份"与"格式变了"）在输出上长得一样。
        self.assertIsNone(self.mod.cell_of("adaptive_c4"))


class TestConfigMismatch(unittest.TestCase):
    """比对是**单向**的：配置=意图，快照=这次 run 实际加载到的。"""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    def test_identical_is_clean(self):
        probs, extra = self.mod.config_mismatches(_REAL_CFG, _REAL_CFG)
        self.assertEqual(probs, [])
        self.assertEqual(extra, 0)

    def test_differing_value_reports_dotted_path_and_both_values(self):
        snap = {
            "run": {"repeats": 3, "slo": {"tpot_ms": 100}},
            "controller": {"target_tpot_ms": 110.0, "hysteresis": 0.08},
        }
        probs, _ = self.mod.config_mismatches(_REAL_CFG, snap)
        self.assertEqual(len(probs), 1)
        # 点分路径 + 两个取值都要在：只说"不一致"读的人无法复核
        self.assertIn("controller.target_tpot_ms", probs[0])
        self.assertIn("120.0", probs[0])
        self.assertIn("110.0", probs[0])

    def test_key_absent_from_snapshot_is_a_problem(self):
        """配置声明了、快照里没有 ⇒ 这条设置**根本没进**这次 run。"""
        snap = {
            "run": {"repeats": 3, "slo": {"tpot_ms": 100}},
            "controller": {"target_tpot_ms": 120.0},
        }
        probs, _ = self.mod.config_mismatches(_REAL_CFG, snap)
        self.assertEqual(len(probs), 1)
        self.assertIn("controller.hysteresis", probs[0])
        self.assertIn("没有这个键", probs[0])

    def test_extra_snapshot_keys_are_only_counted(self):
        """快照多出的是**默认值展开**，不是不一致——否则"字段多"会被误读成错。"""
        snap = {
            "run": {"repeats": 3, "slo": {"tpot_ms": 100}},
            "controller": {"target_tpot_ms": 120.0, "hysteresis": 0.08},
            "device_model": {"step_overhead_ms": 2.0},
        }
        probs, extra = self.mod.config_mismatches(_REAL_CFG, snap)
        self.assertEqual(probs, [])
        self.assertEqual(extra, 1)

    def test_flattening_reaches_nested_leaf(self):
        """要能一路展平到叶子。只比顶层 key 的话，``run.slo.tpot_ms`` 这类改动会漏掉。"""
        snap = {
            "run": {"repeats": 3, "slo": {"tpot_ms": 50}},
            "controller": {"target_tpot_ms": 120.0, "hysteresis": 0.08},
        }
        probs, _ = self.mod.config_mismatches(_REAL_CFG, snap)
        self.assertEqual(len(probs), 1)
        self.assertIn("run.slo.tpot_ms", probs[0])


class TestVerifyConfigGuard(unittest.TestCase):
    """部署戳的 ``digest_scope`` **不含 configs/**，所以"代码对"推不出"配置对"。

    已实测过的事故：远端留着旧模板（``slo.tpot_ms`` 还是 50）且没有新配置目录，
    批次照样跑得动、照样盖 ``MEASURED``——从产物表面看不出任何异常。
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    def test_matching_config_dir_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfgd = root / "cfgs"
            for i in range(3):
                suffix = "" if i == 0 else f"-r{i}"
                _mk_run_with_snapshot(
                    root, f"T-v-phase1_anchor-adaptive_c4{suffix}", _REAL_CFG
                )
            _mk_cfg_dir(cfgd, "adaptive_c4", _REAL_CFG)
            rc, out = _run_inspector(
                self.mod, root, "--tag", "T-v", "--verify-config", str(cfgd)
            )
        self.assertEqual(rc, 0)
        self.assertIn("逐字段一致", out)

    def test_stale_config_dir_returns_nonzero(self):
        """**关键反向对照**：配置不是这一版时必须返回非零。

        这一条不过，说明"输入新鲜度"只是个打印，挡不住上面那次事故——
        即"失败不了"的检查等于没有检查。
        """
        stale = {
            "run": {"repeats": 3, "slo": {"tpot_ms": 50}},
            "controller": {"target_tpot_ms": 110.0, "hysteresis": 0.1},
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfgd = root / "cfgs"
            _mk_run_with_snapshot(root, "T-w-phase1_anchor-adaptive_c4", _REAL_CFG)
            _mk_cfg_dir(cfgd, "adaptive_c4", stale)
            rc, out = _run_inspector(
                self.mod, root, "--tag", "T-w", "--verify-config", str(cfgd)
            )
        self.assertEqual(rc, 1)
        self.assertIn("controller.target_tpot_ms", out)
        self.assertIn("不能当作", out)

    def test_missing_config_for_a_run_is_a_problem(self):
        """run 在、对应配置文件不在：这是"这个格没被生成"，不能当通过。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfgd = root / "cfgs"
            _mk_run_with_snapshot(root, "T-x2-phase1_anchor-adaptive_c8", _REAL_CFG)
            _mk_cfg_dir(cfgd, "adaptive_c4", _REAL_CFG)  # 故意只给 c4
            rc, out = _run_inspector(
                self.mod, root, "--tag", "T-x2", "--verify-config", str(cfgd)
            )
        self.assertEqual(rc, 1)
        self.assertIn("配置目录里没有", out)

    def test_missing_snapshot_is_a_problem(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfgd = root / "cfgs"
            _mk_run(root, "T-x3-phase1_anchor-adaptive_c4")  # 只写 meta.json
            _mk_cfg_dir(cfgd, "adaptive_c4", _REAL_CFG)
            rc, out = _run_inspector(
                self.mod, root, "--tag", "T-x3", "--verify-config", str(cfgd)
            )
        self.assertEqual(rc, 1)
        self.assertIn("快照缺失", out)

    def test_cross_batch_tag_is_refused(self):
        """``--tag`` 是**子串**匹配，会跨批次命中；一个配置目录不可能同时是两批的输入。

        这种输入下比出来的"不一致"全是假的，所以宁可直接拒绝，也不要印一堆噪音。
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfgd = root / "cfgs"
            _mk_run_with_snapshot(root, "B1-x-adaptive_c4", _REAL_CFG)
            _mk_run_with_snapshot(root, "B2-x-adaptive_c4", _REAL_CFG)
            _mk_cfg_dir(cfgd, "adaptive_c4", _REAL_CFG)
            rc, out, err = _run_inspector_both(
                self.mod, root, "--tag", "x-adaptive_c4", "--verify-config", str(cfgd)
            )
        self.assertEqual(rc, 2)
        self.assertIn("2 个批次", err)
        # 拒绝时**不得**先印一半结论出去
        self.assertNotIn("逐字段一致", out)

    def test_missing_config_dir_returns_two(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_run_with_snapshot(root, "T-x4-phase1_anchor-adaptive_c4", _REAL_CFG)
            rc, _out, err = _run_inspector_both(
                self.mod, root, "--tag", "T-x4",
                "--verify-config", str(root / "nope"),
            )
        self.assertEqual(rc, 2)
        self.assertIn("配置目录不存在", err)

    def test_unused_config_is_only_noted(self):
        """配置目录里多出的格只作提示：它是"这一格没跑"，不是"跑错了"。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfgd = root / "cfgs"
            _mk_run_with_snapshot(root, "T-x5-phase1_anchor-adaptive_c4", _REAL_CFG)
            _mk_cfg_dir(cfgd, "adaptive_c4", _REAL_CFG)
            _mk_cfg_dir(cfgd, "static_c8", _REAL_CFG)
            rc, out = _run_inspector(
                self.mod, root, "--tag", "T-x5", "--verify-config", str(cfgd)
            )
        self.assertEqual(rc, 0)
        self.assertIn("static_c8", out)
        self.assertIn("仅提示", out)


class TestMultipleViews(unittest.TestCase):
    """多个视图标志必须**都**输出。

    早先每个视图各自 ``return 0``，于是 ``--controller --prefill-effect`` 会**静默**
    只出第一个——调用方以为两件事都查过了。这正是本项目最防的一类失效。
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    def _mk_adaptive_run(self, root: Path, name: str) -> None:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "data_label": "MEASURED",
            "summary": {},
            # 只要 controller_effect 在，--controller 就有话可说
            "controller_effect": {
                "n_control_decisions": 5, "n_decisions_changed": 0,
                "n_decisions_recover": 0, "n_cycles": 4,
                "n_decisions_over_memory": 0,
            },
        }
        (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_two_views_are_both_printed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._mk_adaptive_run(root, "T-m-phase1_anchor-adaptive_c1")
            rc, out = _run_inspector(
                self.mod, root, "--tag", "T-m", "--controller", "--prefill-effect"
            )
        self.assertEqual(rc, 0)
        self.assertIn("控制器实际动作", out)      # 第一个视图
        self.assertIn("控制器动作是否", out)      # 第二个视图（早先会被吞掉）
        # 叠加时要先声明顺序，免得读的人以为只有一份输出
        self.assertIn("指定了 2 个视图", out)

    def test_single_view_does_not_print_the_multi_view_note(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._mk_adaptive_run(root, "T-n-phase1_anchor-adaptive_c1")
            rc, out = _run_inspector(self.mod, root, "--tag", "T-n", "--controller")
        self.assertEqual(rc, 0)
        self.assertIn("控制器实际动作", out)
        self.assertNotIn("指定了", out)
        # 单视图时不得落到"逐 run 明细"这条默认路径上
        self.assertNotIn("每格的离散度", out)


def _mk_cycles_run(
    root: Path,
    name: str,
    *,
    n_steps: int = 263,
    arrivals: list[float] | None = None,
    concurrency: int = 4,
    controller_enabled: bool = True,
    trace: list[dict] | None = None,
) -> None:
    """造一个 ``--cycles`` 能读的 run：meta + telemetry.jsonl + requests.jsonl。"""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_label": "MEASURED",
        "summary": {},
        "config": {
            "controller": {"enabled": controller_enabled},
            "workload_overrides": {"concurrency": concurrency},
        },
        "controller_trace": trace
        if trace is not None
        else [
            {"step": 8, "changed": True, "bottleneck": "tpot"},
            {"step": 16, "changed": False, "bottleneck": "ttft"},
            {"step": 32, "changed": True, "bottleneck": "ttft"},
        ],
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # 一行 = 一步；t_ms 每步 +100，便于把到达时刻映射回 step
    (d / "telemetry.jsonl").write_text(
        "\n".join(json.dumps({"step": i, "t_ms": i * 100.0}) for i in range(n_steps)) + "\n",
        encoding="utf-8",
    )
    arr = arrivals if arrivals is not None else [0.3] * 4 + [14461.0] * 4
    (d / "requests.jsonl").write_text(
        "\n".join(json.dumps({"req_id": f"r{i:05d}", "arrival_ms": a}) for i, a in enumerate(arr))
        + "\n",
        encoding="utf-8",
    )


class TestCyclesView(unittest.TestCase):
    """``--cycles`` 这一视图的存在理由：**决策步号是 run 内的局部量**。

    本仓库踩过：把 c2 的 ``264/288/392/416`` 当成了 c4 的决策链写进结论，
    而 c4 首轮只有 263 步。只有把"总步数 / 批次数 / 批起点 / 决策步号"放在**同一行**，
    这种接错才会在对读时暴露出来。下面的用例就守住这一点。
    """

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_script("inspect_step3_runs")

    # ---- 纯函数：切批 ----

    def test_batches_split_on_large_gap(self):
        # 批内 ~600 ms、批间 ~11000 ms ⇒ 应切出 3 批
        self.assertEqual(
            self.mod.arrival_batches([0.0, 0.2, 11371.0, 11975.0, 22721.0]),
            [0.0, 11371.0, 22721.0],
        )

    def test_all_at_once_is_one_batch(self):
        # 全部同刻（间隔 < 500 ms 下限）⇒ **1 批**，不得被切成一堆单请求批
        self.assertEqual(self.mod.arrival_batches([0.1, 0.2, 0.3, 0.4]), [0.1])

    def test_no_arrivals_is_empty_not_zero(self):
        # 读不到 ≠ "0 批"：返回空列表，让调用方印 "-"
        self.assertEqual(self.mod.arrival_batches([]), [])

    def test_step_at_ms_picks_nearest(self):
        rows = [{"step": i, "t_ms": i * 100.0} for i in range(10)]
        self.assertEqual(self.mod.step_at_ms(rows, 461.0), 5)   # 500 更近
        self.assertIsNone(self.mod.step_at_ms([], 1.0))          # 映射不出 ⇒ None

    def test_step_at_ms_ignores_string_step(self):
        """坏字段（step 写成字符串）不能比大小，但也不该让整段崩掉。"""
        rows = [{"step": "5", "t_ms": 0.0}, {"step": 7, "t_ms": 100.0}]
        self.assertEqual(self.mod.step_at_ms(rows, 100.0), 7)

    # ---- 视图 ----

    def test_total_steps_is_line_count(self):
        """总步数 = telemetry 行数。**这是防盗引的那个数**：263 步的格不可能有 step 264。"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_cycles_run(root, "T-c1-phase1_anchor-adaptive_c4", n_steps=263)
            rc, out = _run_inspector(self.mod, root, "--tag", "T-c1", "--cycles")
        self.assertEqual(rc, 0)
        self.assertIn("263", out)
        self.assertIn("8:收缩 32:放大", out)

    def test_batch_count_is_requests_over_concurrency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_cycles_run(root, "T-c2-phase1_anchor-adaptive_c4", n_steps=263, concurrency=4)
            rep = self.mod.cycle_report(
                "T-c2-phase1_anchor-adaptive_c4",
                json.loads((root / "T-c2-phase1_anchor-adaptive_c4" / "meta.json").read_text("utf-8")),
                root / "T-c2-phase1_anchor-adaptive_c4",
            )
        self.assertEqual(rep["n_requests"], 8)
        self.assertEqual(len(rep["batches"]), 2)          # 8 / 4
        self.assertEqual(rep["batches"], [0, 145])        # 14461 ms → step 145
        self.assertEqual(rep["total_steps"], 263)

    def test_disabled_controller_is_not_zero_actions(self):
        """控制器**未启用** ≠ "零动作"。

        两者含义相反：前者是"这一次没让它参与"，后者是"它参与了但无事可做"。
        印成同一句话会把"配置错误"读成"负载无事可做"。
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_cycles_run(
                root, "T-c3-phase1_anchor-adaptive_c4",
                controller_enabled=False, trace=[],
            )
            rc, out = _run_inspector(self.mod, root, "--tag", "T-c3", "--cycles")
        self.assertEqual(rc, 0)
        self.assertIn("控制器未启用", out)
        self.assertNotIn("（零动作）", out)

    def test_enabled_controller_with_no_decisions_says_zero_actions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_cycles_run(
                root, "T-c4-phase1_anchor-adaptive_c1",
                controller_enabled=True, trace=[{"step": 8, "changed": False, "bottleneck": "none"}],
            )
            rc, out = _run_inspector(self.mod, root, "--tag", "T-c4", "--cycles")
        self.assertEqual(rc, 0)
        self.assertIn("（零动作）", out)
        self.assertNotIn("控制器未启用", out)

    def test_cycles_stacks_with_controller_view(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _mk_cycles_run(root, "T-c5-phase1_anchor-adaptive_c4")
            rc, out = _run_inspector(self.mod, root, "--tag", "T-c5", "--cycles", "--controller")
        self.assertEqual(rc, 0)
        self.assertIn("补入批次", out)          # 第二个视图
        self.assertIn("指定了 2 个视图", out)


if __name__ == "__main__":
    unittest.main()
