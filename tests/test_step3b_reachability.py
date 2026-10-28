"""修正版控制器阈值的**开环可达性预检**（``check_step3b_reachability.py``）的守卫测试。

这道门要防的是第③步第一次执行的那个失效：``adaptive`` 臂在 36/36 次运行里
``chunk_req`` 恒为 512/512、``n_recovered=0`` —— 控制器一次都没动过。
产物齐全、无报错、标签全绿，但那一臂没测到它声称要测的东西。
其中一半成因（``chunk_size_init == chunk_size_min``）是静态可查的；
另一半（``target_tpot_ms`` 低于实测信号下限）**只能靠实录信号判定**。本文件守的是后者。

三个必须钉住的性质
------------------
1. **复放要忠实**：喂进去的是控制器实际读到的 P90；缺信号的决策点要**跳过**而不是填 0
   （0 会被读成"测出来是 0"，是一条假信号，会让复放结论偏乐观）。
2. **轨迹要能看见第一步**：``chunk_series`` 必须以初值为第 0 项。只记 observe
   **之后**的值时，"第一步就收缩"在相邻比较里不可见 —— 实测症状是 c8 明明收敛到
   512，转移计数却报"收缩 0 次、min=max=512"，自相矛盾且看起来像"没动"。
3. **门要能失败**：每个方向的可达性判断都必须配一个"故意设成不可达"的反向对照。
   一个不会失败的门等于没有门。
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

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CFG_DIR_B = _ROOT / "configs" / "mthreads" / "phase1_step3b"


def _load_script(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    if not path.exists():
        raise unittest.SkipTest(f"缺少脚本 {path}")
    modname = f"_reach_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # 必须先注册进 sys.modules：脚本用了 ``from __future__ import annotations``，
    # 于是 @dataclass 的字段注解是**字符串**，dataclasses 解析时要拿
    # ``sys.modules[cls.__module__]`` —— 不注册就会以
    # ``AttributeError: 'NoneType' object has no attribute '__dict__'`` 整块失败。
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _controller_cfg(**overrides):
    """取修正版配置里的 ``ControllerConfig``，可选覆盖若干字段。"""
    from dataclasses import replace

    from run_benchmark import load_config

    cfg = load_config(_CFG_DIR_B / "adaptive_c1.yaml")
    return replace(cfg.controller, **overrides) if overrides else cfg.controller


def _write_run(root: Path, cell: str, signals: list[tuple[float, float]]) -> Path:
    """造一个只有 ``controller_trace`` 的 run 目录（够本脚本用）。"""
    d = root / f"P1S3-phase1_anchor-{cell}"
    d.mkdir(parents=True, exist_ok=True)
    trace = [
        {
            "step": (i + 1) * 8,
            "bottleneck": "none",
            "signals": {"ttft_p90_ms": t, "tpot_p90_ms": p},
        }
        for i, (t, p) in enumerate(signals)
    ]
    (d / "meta.json").write_text(
        json.dumps({"controller_trace": trace}), encoding="utf-8"
    )
    return d


class TestSignalLoading(unittest.TestCase):
    def test_reads_in_order_and_skips_missing_signals(self):
        mod = _load_script("check_step3b_reachability")
        with tempfile.TemporaryDirectory() as td:
            meta = Path(td) / "meta.json"
            meta.write_text(json.dumps({"controller_trace": [
                {"step": 0, "bottleneck": "none", "signals": {}},        # 无有效观测样本
                {"step": 8, "bottleneck": "tpot", "signals": {
                    "ttft_p90_ms": 631.7, "tpot_p90_ms": 61.7}},
                {"step": 16, "bottleneck": "tpot", "signals": {
                    "ttft_p90_ms": 632.0, "tpot_p90_ms": 62.0}},
            ]}), encoding="utf-8")
            got = mod.load_signal_sequence(meta)
            self.assertEqual(got, [(631.7, 61.7), (632.0, 62.0)],
                             "缺信号的决策点必须被跳过，且顺序不得改变")

    def test_missing_signal_is_skipped_not_zero_filled(self):
        """填 0 会被读成"TPOT 测出来是 0" ⇒ 控制器认为毫无压力 ⇒ 复放结论偏乐观。"""
        mod = _load_script("check_step3b_reachability")
        with tempfile.TemporaryDirectory() as td:
            meta = Path(td) / "meta.json"
            meta.write_text(json.dumps({"controller_trace": [
                {"step": 8, "bottleneck": "none", "signals": {
                    "ttft_p90_ms": None, "tpot_p90_ms": 61.7}},
            ]}), encoding="utf-8")
            self.assertEqual(mod.load_signal_sequence(meta), [])


class TestReplayTrace(unittest.TestCase):
    def test_chunk_series_starts_at_init(self):
        """轨迹第 0 项必须是初值，否则"第一步就动"这件事消失。"""
        mod = _load_script("check_step3b_reachability")
        ctl_cfg = _controller_cfg()
        rep = mod.replay_signals(ctl_cfg, [(631.7, 61.7)])
        self.assertEqual(rep["chunk_series"][0], ctl_cfg.chunk_size_init)
        self.assertGreater(len(rep["chunk_series"]), 1,
                           "至少要有一个 observe 之后的点")

    def test_first_cycle_shrink_is_visible(self):
        """反向对照：不把初值记进轨迹时，第一步收缩会算成"收缩 0 次"。"""
        mod = _load_script("check_step3b_reachability")
        ctl_cfg = _controller_cfg()  # init=1024, min=512, TPOT 上限 121
        rep = mod.replay_signals(ctl_cfg, [(1000.0, 300.0)])  # TPOT 远超目标
        self.assertEqual(rep["chunk_series"], [1024, 512])
        self.assertEqual(rep["n_shrink_steps"], 1,
                         "第一步就收缩，转移计数必须看得见")
        self.assertEqual(rep["n_grow_steps"], 0)

    def test_grow_only_when_tpot_has_headroom(self):
        """TTFT 超目标但 TPOT 已无余量时**不**放大 —— 这是取舍，不是免费收益。"""
        mod = _load_script("check_step3b_reachability")
        ctl_cfg = _controller_cfg()
        limit_tpot = ctl_cfg.target_tpot_ms * (1 + ctl_cfg.hysteresis)
        # TTFT 远超目标，但 TPOT 刚好压在上限上 ⇒ tpot 分支优先 ⇒ 收缩而非放大
        rep = mod.replay_signals(ctl_cfg, [(9999.0, limit_tpot + 1.0)])
        self.assertEqual(rep["n_grow_steps"], 0)
        self.assertEqual(rep["n_shrink_steps"], 1)


class TestRecoveryProbe(unittest.TestCase):
    def test_probe_shrinks_then_returns_to_init(self):
        mod = _load_script("check_step3b_reachability")
        probe = mod.recovery_probe(_controller_cfg())
        self.assertTrue(probe["shrank"],
                        "探针的压力段没能逼出收缩 —— 探针本身失效，后面的结论都不可信")
        self.assertTrue(probe["recovered_to_init"],
                        f"压力消退后没回到初值（停在 {probe['after_relax_chunk']}）"
                        f" ⇒ 恢复路径可疑（棘轮）")
        self.assertGreaterEqual(probe["n_recovered"], 1)

    def test_probe_fails_when_target_is_unreachable(self):
        """反向对照：目标设成不可达时，压力段逼不出收缩，探针必须显形为失效。"""
        mod = _load_script("check_step3b_reachability")
        probe = mod.recovery_probe(_controller_cfg(target_tpot_ms=1e9))
        self.assertFalse(probe["shrank"])


class TestGateEndToEnd(unittest.TestCase):
    """整道门的反向控制：可达 ⇒ 0；不可达 ⇒ 非零，且理由要能对上。"""

    def _run_main(self, cfg_overrides: dict, signals_by_cell: dict):
        mod = _load_script("check_step3b_reachability")
        import yaml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            base = yaml.safe_load(
                (_CFG_DIR_B / "adaptive_c1.yaml").read_text(encoding="utf-8")
            )
            base["controller"].update(cfg_overrides)
            (cfg_dir / "adaptive_c1.yaml").write_text(
                yaml.safe_dump(base, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            runs = root / "runs"
            runs.mkdir()
            for cell, sig in signals_by_cell.items():
                _write_run(runs, cell, sig)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = mod.main([
                    "--configs", str(cfg_dir),
                    "--runs", str(runs),
                    "--cell", "adaptive_c1",
                ])
            return rc, out.getvalue(), err.getvalue()

    def test_reachable_placement_passes(self):
        """低并发信号 ⇒ 放大；高并发信号 ⇒ 收缩 ⇒ 门通过。"""
        rc, out, err = self._run_main(
            {},
            {
                "adaptive_c1": [(631.7, 61.7)] * 12,
                "adaptive_c8": [(6958.5, 141.0)] * 12,
            },
        )
        self.assertEqual(rc, 0, f"可达的阈值布置被判失败：{err}\n{out}")
        self.assertIn("可达性预检通过", out)

    def test_unreachable_placement_fails(self):
        """目标高到任何实测信号都够不着 ⇒ 控制器一动不动 ⇒ 门必须拦下。"""
        rc, out, err = self._run_main(
            {"target_tpot_ms": 1e6, "target_ttft_ms": 1e6},
            {
                "adaptive_c1": [(631.7, 61.7)] * 12,
                "adaptive_c8": [(6958.5, 141.0)] * 12,
            },
        )
        self.assertNotEqual(rc, 0, "不可达的阈值布置竟然通过了预检")
        self.assertIn("ttft 分支不可达", err)
        self.assertIn("tpot 分支不可达", err)

    def test_no_source_runs_fails_loudly(self):
        """没有实录信号时必须失败 —— 空输入会伪装成合格结果。"""
        mod = _load_script("check_step3b_reachability")
        import yaml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            base = yaml.safe_load(
                (_CFG_DIR_B / "adaptive_c1.yaml").read_text(encoding="utf-8")
            )
            (cfg_dir / "adaptive_c1.yaml").write_text(
                yaml.safe_dump(base, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            runs = root / "runs"
            runs.mkdir()
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = mod.main([
                    "--configs", str(cfg_dir), "--runs", str(runs),
                    "--cell", "adaptive_c1",
                ])
            self.assertNotEqual(rc, 0)
            self.assertIn("找不到", err.getvalue())


if __name__ == "__main__":
    unittest.main()
