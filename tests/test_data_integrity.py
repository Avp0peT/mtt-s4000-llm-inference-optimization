"""数据完整性约束（评审 item 3）。

这里测的三件事有一个共同点：**缺了它们都不会报错**。

  A. run_id 唯一     ``results/raw`` 是证据目录。同名产物被覆盖后，事后无法区分
                     这一行数字是改动前还是改动后的结果，而两条互相矛盾的结论
                     会共用同一个 run_id。所以写入层必须主动拒绝。
  B. 导出前置校验     原始明细 / 环境指纹 / 有效重复 / 实验登记，四项缺一不可。
                     ``summary.csv`` 缺了它们照样生成、图照样画出来——
                     因此必须有一道独立的校验把它们变成阻断性错误。
  C. （缺来源标签不得升格，见 ``test_collect_metrics``。）

断言针对的是"会不会被挡住"，不是"报错文案好不好看"。
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import preflight as pf  # noqa: E402
import run_benchmark as rb  # noqa: E402
from ascend_opt.grouping import group_key as canonical_group_key  # noqa: E402

BASELINE = _ROOT / "configs" / "baseline" / "default.yaml"


def _empty_result() -> "rb.SimResult":
    return rb.SimResult(
        metrics=[], samples=[], wall_ms=0.0, steps=0, dropped=0, controller_trace=[]
    )


def _stage_run_dir(root: Path, name: str, label: str) -> Path:
    """造一个「已存在」的 run 目录，``meta.json`` 里带上指定标签。

    刻意直接手写 meta.json 而不是真跑一次：这里要构造的是**下游才关心**的状态
    （"目录里躺着一个 MEASURED 记录"），真跑一次反而要依赖标签推导链路。
    """
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(
        json.dumps({"data_label": label, "summary": {"data_label": label}}),
        encoding="utf-8",
    )
    (d / "requests.jsonl").write_text("", encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# A. run_id 唯一
# --------------------------------------------------------------------------- #

class TestRunIdIsNotOverwritten(unittest.TestCase):
    def test_write_run_outputs_refuses_when_artifacts_exist(self):
        """写入层自身必须拒绝——它是产物写入的唯一出口。

        只在调用方拦是不够的：只要有另一条路径直接调 ``write_run_outputs``，
        覆盖就照样会发生。
        """
        cfg = rb.load_config(BASELINE)
        result = rb.SimResult(
            metrics=[], samples=[], wall_ms=0.0, steps=0, dropped=0, controller_trace=[]
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "fixed-id"
            kwargs = dict(
                run_id="fixed-id", repeat_index=0, dry_run=True,
                workload_fingerprint={}, summary={},
            )
            rb.write_run_outputs(run_dir, cfg, result, **kwargs)
            self.assertTrue((run_dir / "meta.json").exists())

            with self.assertRaises(rb.RunDirConflict) as ctx:
                rb.write_run_outputs(run_dir, cfg, result, **kwargs)
            # 报错必须指名冲突的是哪个 run、已经存在哪些产物
            self.assertIn("fixed-id", str(ctx.exception))
            self.assertIn("meta.json", str(ctx.exception))

    def test_empty_leftover_dir_does_not_block_the_run_id(self):
        """判据是「产物」而不是「目录存在」。

        run 目录会被 ``mkdir(parents=True)`` 提前建出来，一次中断留下的空目录
        若也算冲突，这个 run_id 就被永久堵死了。
        """
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "fixed-id"
            run_dir.mkdir(parents=True)
            self.assertEqual(rb.occupying_artifacts(run_dir), [])

            cfg = rb.load_config(BASELINE)
            cfg.run.repeats = 1
            produced = rb.run_once(
                cfg, dry_run=True, out_dir=Path(tmp), run_id="fixed-id", verbose=False
            )
            self.assertTrue((produced / "meta.json").exists())

    def test_run_once_fails_before_doing_the_work(self):
        """已占用的 run_id 要在**计算之前**就报错。

        跑完一次基准才发现写不进去，是纯粹的浪费；而且剩下的重复会变成
        半批产物，那种"不完整的 1 次重复"最容易被下游当成完整数据用掉。
        """
        cfg = rb.load_config(BASELINE)
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rb.run_once(cfg, dry_run=True, out_dir=out, run_id="dup", verbose=False)
            before = (out / "dup" / "meta.json").read_text(encoding="utf-8")
            with self.assertRaises(rb.RunDirConflict):
                rb.run_once(cfg, dry_run=True, out_dir=out, run_id="dup", verbose=False)
            # 内容必须原封不动：拒绝的意思是"什么都没发生"
            self.assertEqual(
                (out / "dup" / "meta.json").read_text(encoding="utf-8"), before
            )

    def test_overwrite_flag_permits_rewrite(self):
        cfg = rb.load_config(BASELINE)
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rb.run_once(cfg, dry_run=True, out_dir=out, run_id="dup", verbose=False)
            rb.run_once(
                cfg, dry_run=True, out_dir=out, run_id="dup", verbose=False,
                overwrite=True,
            )
            self.assertTrue((out / "dup" / "meta.json").exists())

    def test_cli_refuses_with_exit_code_4_and_force_bypasses_it(self):
        """CLI 层的契约：重复 run_id → 退出码 4。

        这里的产物是 temp 目录里的 dry-run（标签 ``TEST_DATA``），属于
        ``--force`` 允许的那一类，因此它应当放行——这同时是下面那批"拒绝"
        测试的正向对照：闸门不是"一律拒绝"。
        """
        with tempfile.TemporaryDirectory() as tmp:
            args = [
                "--config", str(BASELINE), "--dry-run", "--out", tmp,
                "--run-id", "fixed-id", "--repeats", "1", "--quiet",
            ]
            self.assertEqual(rb.main(args), 0)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(rb.main(args), 4, "重复 run_id 必须以非零退出码拒绝")
            self.assertIn("fixed-id", err.getvalue())
            self.assertEqual(
                rb.main(args + ["--force"]), 0,
                "可丢弃的 dry-run 产物（TEST_DATA）应当被 --force 放行",
            )


class TestForceIsNotABlanketOverride(unittest.TestCase):
    """``--force`` 只放行**可丢弃的调试产物**（评审 Part 1 #1）。

    三条件缺一不可：目标不在 ``results/raw`` 下、本次是 dry-run、已有产物标签是
    ``TEST_DATA``。之所以这么严，是因为**覆盖不会报错**——两条互相矛盾的结论
    共用一个 run_id 之后，事后无法分辨哪一行是改动前、哪一行是改动后。
    因此"已有实验换新 run_id"必须是默认且唯一的正路。

    这批测试同时也是"证明测试会失败"的靶子：把 ``overwrite_refusal`` 换回
    ``if not overwrite: <只查存在>``，下面每一条都会失败。
    """

    def test_force_refuses_to_overwrite_measured(self):
        """已有 MEASURED 记录：即便在临时目录、即便加了 --force，也拒绝。"""
        cfg = rb.load_config(BASELINE)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _stage_run_dir(tmp, "m", "MEASURED")
            before = (run_dir / "meta.json").read_text(encoding="utf-8")
            with self.assertRaises(rb.RunDirConflict) as ctx:
                rb.write_run_outputs(
                    run_dir, cfg, _empty_result(),
                    run_id="m", repeat_index=0, dry_run=True,
                    workload_fingerprint={}, summary={}, overwrite=True,
                )
            # 拒绝理由必须显形，而不是笼统一句"被占用"
            self.assertIn("MEASURED", str(ctx.exception))
            # 拒绝 = 什么都没发生
            self.assertEqual(
                (run_dir / "meta.json").read_text(encoding="utf-8"), before
            )

    def test_force_refuses_when_this_run_is_not_dry_run(self):
        """真机 run 会产出实测数据，不能靠 --force 盖掉已有产物。"""
        cfg = rb.load_config(BASELINE)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _stage_run_dir(tmp, "d", "TEST_DATA")
            with self.assertRaises(rb.RunDirConflict) as ctx:
                rb.write_run_outputs(
                    run_dir, cfg, _empty_result(),
                    run_id="d", repeat_index=0, dry_run=False,
                    workload_fingerprint={}, summary={}, overwrite=True,
                )
            self.assertIn("dry-run", str(ctx.exception))

    def test_force_refuses_under_evidence_dir_even_for_test_data(self):
        """证据目录之下，--force 一律无效——**哪怕**产物只是 TEST_DATA。

        把 ``_RAW_ROOT`` 指到临时目录，是为了在**不碰真实 results/raw** 的前提下
        走一遍真实代码路径（直接调 overwrite_refusal 会绕过写入口）。
        """
        cfg = rb.load_config(BASELINE)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            run_dir = _stage_run_dir(raw, "ev", "TEST_DATA")
            with unittest.mock.patch.object(rb, "_RAW_ROOT", raw):
                with self.assertRaises(rb.RunDirConflict) as ctx:
                    rb.write_run_outputs(
                        run_dir, cfg, _empty_result(),
                        run_id="ev", repeat_index=0, dry_run=True,
                        workload_fingerprint={}, summary={}, overwrite=True,
                    )
            self.assertIn("证据目录", str(ctx.exception))

    def test_unknown_existing_label_is_refused_not_assumed_overwritable(self):
        """取不到标签 = **未知**，按不可覆盖处理，而不是"没标签=可覆盖"。"""
        cfg = rb.load_config(BASELINE)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "u"
            run_dir.mkdir(parents=True)
            # 只有 telemetry.jsonl，没有 meta.json → 标签无从得知
            (run_dir / "telemetry.jsonl").write_text("", encoding="utf-8")
            self.assertIsNone(
                rb.load_existing_label(run_dir),
                "meta.json 缺失时标签应为 None（未知），不是某个默认值",
            )
            with self.assertRaises(rb.RunDirConflict) as ctx:
                rb.write_run_outputs(
                    run_dir, cfg, _empty_result(),
                    run_id="u", repeat_index=0, dry_run=True,
                    workload_fingerprint={}, summary={}, overwrite=True,
                )
            self.assertIn("取不到", str(ctx.exception))

    def test_corrupt_meta_does_not_unlock_overwrite(self):
        """meta.json 损坏 → 标签取不到 → 拒绝覆盖（宁可保守）。"""
        cfg = rb.load_config(BASELINE)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "c"
            run_dir.mkdir(parents=True)
            (run_dir / "meta.json").write_text("{ not json", encoding="utf-8")
            self.assertIsNone(rb.load_existing_label(run_dir))
            with self.assertRaises(rb.RunDirConflict):
                rb.write_run_outputs(
                    run_dir, cfg, _empty_result(),
                    run_id="c", repeat_index=0, dry_run=True,
                    workload_fingerprint={}, summary={}, overwrite=True,
                )

    def test_force_still_allows_discardable_dry_run(self):
        """正向对照：temp 目录里的 dry-run 产物该被放行，闸门不是一律拒绝。"""
        cfg = rb.load_config(BASELINE)
        cfg.run.repeats = 1
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            rb.run_once(cfg, dry_run=True, out_dir=out, run_id="dup", verbose=False)
            rb.run_once(
                cfg, dry_run=True, out_dir=out, run_id="dup", verbose=False,
                overwrite=True,
            )
            self.assertTrue((out / "dup" / "meta.json").exists())

    def test_cli_force_on_measured_returns_4(self):
        """CLI 层：目标已被 MEASURED 记录占用，--force 也要返回 4。"""
        with tempfile.TemporaryDirectory() as tmp:
            _stage_run_dir(tmp, "fixed-id", "MEASURED")
            args = [
                "--config", str(BASELINE), "--dry-run", "--out", tmp,
                "--run-id", "fixed-id", "--repeats", "1", "--quiet", "--force",
            ]
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(
                    rb.main(args), 4,
                    "已有 MEASURED 产物，--force 也不得覆盖",
                )
            self.assertIn("MEASURED", err.getvalue())


# --------------------------------------------------------------------------- #
# B. 导出前置校验
# --------------------------------------------------------------------------- #

def stage(
    root: Path,
    *,
    bases: tuple[str, ...] = ("20260101-000000-w-cfgA",),
    repeats: int = 3,
    label: str = "MEASURED",
    env: bool = True,
    env_payload: object | None = None,
    detail: bool = True,
    detail_body: str | None = None,
    snapshot: bool = True,
    meta_label: str | None = None,
    meta_run_id: str | None = None,
    register: bool = True,
    register_as: str = "table",
    experiments: bool = True,
    claim_sufficient: bool | None = None,
) -> dict[str, Path]:
    """造一套最小的 raw + processed + EXPERIMENTS.md，用于驱动校验。

    每个开关对应一项检查，默认全齐 —— 这样"改坏一样东西"就能单独验证那一项。

    产物必须**内容合格**（而不仅是存在）：校验已经升级为逐 run 解析
    ``meta.json`` / ``requests.jsonl`` / ``config.snapshot.yaml`` 与指纹内容，
    因此 fixture 也得写成真实结构，否则"完整的一套"根本过不了。
    """
    raw = root / "raw"
    processed = root / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    if env:
        (raw / "_env").mkdir(exist_ok=True)
        payload = {"available": True} if env_payload is None else env_payload
        if isinstance(payload, str):
            (raw / "_env" / "env.json").write_text(payload, encoding="utf-8")
        else:
            (raw / "_env" / "env.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

    rows: list[dict[str, str]] = []
    groups: list[dict[str, str]] = []
    for base in bases:
        members = [base] + [f"{base}-r{i}" for i in range(1, repeats)]
        for rid in members:
            run_dir = raw / rid
            run_dir.mkdir(parents=True, exist_ok=True)
            # meta.json：键齐 + run_id / data_label 自洽（可被开关改坏）
            meta = {
                "data_label": meta_label if meta_label is not None else label,
                "summary": {"data_label": label},
                "reproducibility": {"data_label": label, "git_commit": "deadbeef"},
                "config": {"config_path": "configs/baseline/default.yaml"},
                "context": {"run_id": meta_run_id if meta_run_id is not None else rid},
            }
            (run_dir / "meta.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            if detail:
                (run_dir / "requests.jsonl").write_text(
                    detail_body if detail_body is not None else '{"req_id": "r0"}\n',
                    encoding="utf-8",
                )
            if snapshot:
                (run_dir / "config.snapshot.yaml").write_text(
                    "run:\n  repeats: 3\nworkload:\n  name: short_chat\n",
                    encoding="utf-8",
                )
            rows.append({"run_id": rid, "data_label": label, "workload": "w", "config": "cfgA"})
        sufficient = repeats >= pf.MIN_REPEATS_FOR_CONCLUSION
        if claim_sufficient is not None:
            sufficient = claim_sufficient
        # 夹具必须写成**真实**的聚合产物形状：分组键含平台与后端（这里两列都缺，
        # 故落到 unknown），成员清单显式落盘。导出校验不再按 run_id 前缀自己
        # 推导成员，缺 member_run_ids 会被报成"派生文件是旧版"。
        groups.append({
            "group": canonical_group_key(
                {"run_id": base, "workload": "w", "config": "cfgA"}
            ),
            "run_id_base": base,
            "member_run_ids": ";".join(sorted(members)),
            "n_repeats": str(repeats),
            "sufficient_repeats": str(sufficient),
            "data_label": label,
        })

    with (processed / "summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["run_id", "data_label", "workload", "config"])
        writer.writeheader()
        writer.writerows(rows)
    with (processed / "repeats.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "group", "run_id_base", "member_run_ids", "n_repeats",
                "sufficient_repeats", "data_label",
            ],
        )
        writer.writeheader()
        writer.writerows(groups)

    exp = root / "EXPERIMENTS.md"
    if experiments:
        body = "# EXPERIMENTS\n\n"
        if register:
            if register_as == "table":
                body += "\n".join(f"| EXP-x-001 | {b} | short_chat | cfgA |" for b in bases) + "\n"
            elif register_as == "prose":
                # 只在正文里提一嘴：不算登记
                body += "\n".join(f"- 我们跑过 {b} 这个 run。" for b in bases) + "\n"
        exp.write_text(body, encoding="utf-8")

    return {"raw": raw, "processed": processed, "experiments": exp}


class TestExportPreflight(unittest.TestCase):
    def _check(self, staged: dict[str, Path], **kw) -> list[pf.Problem]:
        return pf.check_export(
            raw_dir=staged["raw"],
            processed_dir=staged["processed"],
            experiments_md=staged["experiments"],
            **kw,
        )

    def _codes(self, problems: list[pf.Problem]) -> set[str]:
        return {p.code for p in problems}

    def test_complete_setup_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._check(stage(Path(tmp))), [])

    def test_missing_request_detail_blocks(self):
        """没有逐请求明细，汇总里的每个分位数都成了不可追溯的断言。"""
        with tempfile.TemporaryDirectory() as tmp:
            problems = self._check(stage(Path(tmp), detail=False))
            self.assertIn("raw_detail_missing", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_missing_raw_run_dir_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp))
            # 删掉整个 run 目录（模拟 raw 被清理过）
            target = staged["raw"] / "20260101-000000-w-cfgA-r1"
            for f in target.iterdir():
                f.unlink()
            target.rmdir()
            problems = self._check(staged)
            self.assertIn("raw_detail_missing", self._codes(problems))

    def test_missing_env_fingerprint_blocks(self):
        """指纹是 .gitignore 的，克隆到新机器后它一定不在——这条检查就是为那个时刻准备的。"""
        with tempfile.TemporaryDirectory() as tmp:
            problems = self._check(stage(Path(tmp), env=False))
            self.assertIn("env_fingerprint_missing", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_insufficient_repeats_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = self._check(stage(Path(tmp), repeats=2))
            self.assertIn("insufficient_repeats", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_unregistered_run_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = self._check(stage(Path(tmp), register=False))
            self.assertIn("experiment_not_registered", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_missing_experiments_table_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = self._check(stage(Path(tmp), experiments=False))
            self.assertIn("experiments_table_missing", self._codes(problems))

    def test_non_measured_rows_do_not_block_traceability(self):
        """非 MEASURED 行（诊断包场景）不该把可追溯性要求升级成阻断。

        但缺陷仍然要报——降级成提示，不是消失。
        """
        with tempfile.TemporaryDirectory() as tmp:
            problems = self._check(
                stage(Path(tmp), label="TEST_DATA", env=False, repeats=2, register=False)
            )
            self.assertTrue(problems, "缺陷必须被报出来，不能被静默吞掉")
            self.assertTrue(all(p.level == pf.WARN for p in problems))

    def test_lenient_still_blocks_on_missing_detail(self):
        """诊断包也要能对上明细：缺原始明细在**任何**模式下都是阻断。"""
        with tempfile.TemporaryDirectory() as tmp:
            problems = self._check(stage(Path(tmp), detail=False), lenient=True)
            self.assertIn("raw_detail_missing", self._codes(problems))
            self.assertTrue(any(p.level == pf.BLOCK for p in problems))

    def test_lenient_downgrades_the_other_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            problems = self._check(
                stage(Path(tmp), env=False, repeats=2, register=False), lenient=True
            )
            self.assertTrue(problems)
            self.assertTrue(all(p.level == pf.WARN for p in problems))

    def test_empty_summary_is_left_to_the_exporter(self):
        """汇总表为空时不在这里报错：导出脚本的措辞更准确（它知道是标签过滤导致的）。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp))
            (staged["processed"] / "summary.csv").write_text("", encoding="utf-8")
            self.assertEqual(self._check(staged), [])

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), env=False)
            argv = [
                "--raw", str(staged["raw"]),
                "--processed", str(staged["processed"]),
                "--experiments", str(staged["experiments"]),
            ]
            self.assertEqual(pf.main(argv), 4)
            staged2 = stage(Path(tmp) / "ok")
            argv2 = [
                "--raw", str(staged2["raw"]),
                "--processed", str(staged2["processed"]),
                "--experiments", str(staged2["experiments"]),
            ]
            self.assertEqual(pf.main(argv2), 0)

    def test_multiple_runs_report_each_offender(self):
        """报错必须逐条指名，而不是"有 1 项问题"——否则修的人得自己去找是哪一条。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), bases=("20260101-000000-w-cfgA", "20260101-000001-w-cfgB"),
                           register=False)
            problems = self._check(staged)
            details = " ".join(p.detail for p in problems)
            self.assertIn("20260101-000000-w-cfgA", details)
            self.assertIn("20260101-000001-w-cfgB", details)


class TestPreflightChecksContentNotJustPresence(unittest.TestCase):
    """只查"文件在不在"会放过一批**内容非法**的产物（评审 Part 1 #2）。

    这些缺陷的共同点与整个模块一样：不会报错。一个合法外壳、内容却是空的或
    自相矛盾的文件，会让"可追溯"变成一句没有兑现的话。所以这里逐条固定住
    "内容必须解释得通"。
    """

    def _check(self, staged: dict[str, Path], **kw) -> list[pf.Problem]:
        return pf.check_export(
            raw_dir=staged["raw"],
            processed_dir=staged["processed"],
            experiments_md=staged["experiments"],
            **kw,
        )

    def _codes(self, problems: list[pf.Problem]) -> set[str]:
        return {p.code for p in problems}

    # ---- 逐请求明细：存在 ≠ 可解析 ----

    def test_truncated_detail_line_blocks(self):
        """截断的写入：字节数非零，内容却是半截——大小检查会放行。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), detail_body='{"req_id": "r0"}\n{"req_id": "r1"')
            problems = self._check(staged)
            self.assertIn("detail_unreadable", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_whitespace_only_detail_blocks(self):
        """只有换行的文件"非空"，却一条记录都读不出。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), detail_body="\n\n   \n")
            # 注意：这里文件 size > 0，所以旧的"存在且非空"检查会放行
            detail = next(staged["raw"].glob("*/requests.jsonl"))
            self.assertGreater(detail.stat().st_size, 0)
            problems = self._check(staged)
            self.assertIn("detail_unreadable", self._codes(problems))

    # ---- 环境指纹：合法 JSON 但内容不成立 ----

    def test_malformed_env_json_blocks(self):
        """非法 JSON 的指纹，旧检查只看存在与否，会照样通过。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), env_payload="{ this is not json")
            problems = self._check(staged)
            self.assertIn("env_unreadable", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_env_without_availability_declaration_blocks(self):
        """合法 JSON、但没有声明可用性 → 无法为 MEASURED 行背书。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), env_payload={"hostname": "somewhere"})
            problems = self._check(staged)
            self.assertIn("env_does_not_declare_availability", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_env_declaring_unavailable_blocks_measured(self):
        """指纹说"没有 NPU"，却存在 MEASURED 行——直接矛盾。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), env_payload={"available": False})
            problems = self._check(staged)
            self.assertIn("env_declares_unavailable", self._codes(problems))

    def test_env_declaring_unavailable_is_fine_for_test_data(self):
        """诊断包里指纹声明不可用是**正常**的：那正是它为什么是 TEST_DATA。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), env_payload={"available": False}, label="TEST_DATA")
            problems = self._check(staged)
            self.assertNotIn("env_declares_unavailable", self._codes(problems))

    def test_old_format_env_error_says_recollect_not_recheck(self):
        """旧格式（无 available）与"探测没结论"的修法不同，报错必须说清是哪一种。

        缺字段只能靠重采修，改导出检查或手填 available 都是在伪造证据。
        所以报错里必须出现采集脚本的名字。
        """
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(
                Path(tmp), env_payload={"hostname": "somewhere", "schema_version": 1}
            )
            problems = self._check(staged)
            detail = next(
                p.detail for p in problems
                if p.code == "env_does_not_declare_availability"
            )
            self.assertIn("inspect_env.sh", detail)
            self.assertIn("schema_version=1", detail)

    def test_env_null_availability_is_reported_as_inconclusive_not_absent(self):
        """``available=null`` 是"探测跑过、没得出结论"，不是"漏写字段"。

        两者的修法不同：前者修探测工具后重采，后者重跑采集脚本。把 null 报成
        "未声明"会让人以为只要把字段填上就行——那等于把"没查出来"改写成结论。
        报错必须点明 null 的含义，并禁止改成 false。
        """
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), env_payload={"available": None, "platform": "ascend"})
            problems = self._check(staged)
            self.assertIn("env_declares_unavailable", self._codes(problems))
            detail = next(
                p.detail for p in problems if p.code == "env_declares_unavailable"
            )
            self.assertIn("没得出结论", detail)
            self.assertIn("不要把它改成 false", detail)

    def test_env_availability_must_be_a_real_boolean(self):
        """``available`` 只接受 JSON ``true``。

        用真值测试判定时，``"yes"`` / ``1`` 这类值会静默通过——于是一份
        **没有任何布尔可用性字段**的指纹给 MEASURED 行背了书。这是最隐蔽的一种：
        文件里"看起来写了可用性"，实际什么都没证明。
        """
        for fake in ("yes", 1, "true"):
            with self.subTest(value=fake), tempfile.TemporaryDirectory() as tmp:
                staged = stage(Path(tmp), env_payload={"available": fake})
                problems = self._check(staged)
                self.assertIn("env_declares_unavailable", self._codes(problems))

    def test_env_error_names_the_declared_platform(self):
        """报错点名平台：同一句"没有可用硬件"，昇腾与 MTT 要去的主机不同。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), env_payload={"available": False, "platform": "mthreads"})
            problems = self._check(staged)
            detail = next(
                p.detail for p in problems if p.code == "env_declares_unavailable"
            )
            self.assertIn("platform=mthreads", detail)

    def test_env_error_does_not_invent_a_platform(self):
        """指纹没声明平台时不得凭空点名——挑错平台就是把读的人引到错误的机器。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), env_payload={"available": False})
            problems = self._check(staged)
            detail = next(
                p.detail for p in problems if p.code == "env_declares_unavailable"
            )
            for name in ("ascend", "mthreads", "昇腾", "摩尔线程"):
                self.assertNotIn(name, detail)

    def test_new_format_env_passes_the_gate(self):
        """新采集脚本写出的三段式指纹（平台 + 可用性 + 探测明细）必须能过闸门。

        这条是 #1 的回归护栏：曾出现过"运行时认为可用、导出时被拒"——
        采集脚本不写 available，遥测在内存里补，导出闸门只认文件里显式写的。
        """
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), env_payload={
                "schema_version": 2,
                "platform": "ascend",
                "available": True,
                "available_reason": "npu-smi -> True",
                "probes": {"ascend": {"result": True}, "mthreads": {"result": False}},
            })
            problems = self._check(staged)
            self.assertEqual(
                [], [c for c in self._codes(problems) if c.startswith("env_")],
                f"新格式指纹不该被拒：{problems}",
            )

    # ---- meta.json：身份与标签自洽 ----

    def test_meta_missing_required_key_blocks(self):
        """缺 summary/reproducibility 之类的必需键 → 这个 run 复核不了。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp))
            run_dir = next(p for p in staged["raw"].iterdir() if p.is_dir() and p.name != "_env")
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
            meta.pop("reproducibility", None)
            (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            problems = self._check(staged)
            self.assertIn("meta_incomplete", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_meta_run_id_mismatch_blocks(self):
        """目录被改名/复制后，追溯链断了——目录名与 meta 记的 run_id 不一致。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), meta_run_id="some-other-run")
            problems = self._check(staged)
            self.assertIn("meta_run_id_mismatch", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_meta_label_mismatch_blocks_for_measured(self):
        """汇总行说 MEASURED、meta 自报 TEST_DATA → 同一行数字有两套身份。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), label="MEASURED", meta_label="TEST_DATA")
            problems = self._check(staged)
            self.assertIn("meta_label_mismatch", self._codes(problems))
            mismatch = [p for p in problems if p.code == "meta_label_mismatch"]
            self.assertTrue(all(p.level == pf.BLOCK for p in mismatch))

    def test_meta_label_mismatch_downgrades_for_non_measured(self):
        """非 MEASURED 行上的标签不一致仍是缺陷，但降为提示。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), label="TEST_DATA", meta_label="NOT_MEASURED")
            problems = self._check(staged)
            mismatch = [p for p in problems if p.code == "meta_label_mismatch"]
            self.assertTrue(mismatch, "标签不一致必须被报出来")
            self.assertTrue(all(p.level == pf.WARN for p in mismatch))

    def test_unparseable_meta_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp))
            run_dir = next(p for p in staged["raw"].iterdir() if p.is_dir() and p.name != "_env")
            (run_dir / "meta.json").write_text("{ broken", encoding="utf-8")
            problems = self._check(staged)
            self.assertIn("meta_unreadable", self._codes(problems))

    # ---- 配置快照 ----

    def test_missing_snapshot_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), snapshot=False)
            problems = self._check(staged)
            self.assertIn("config_snapshot_missing", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_unparseable_snapshot_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp))
            run_dir = next(p for p in staged["raw"].iterdir() if p.is_dir() and p.name != "_env")
            # YAML 里 ": " 后跟非法结构的行
            (run_dir / "config.snapshot.yaml").write_text("a: [1, 2\n", encoding="utf-8")
            problems = self._check(staged)
            self.assertIn("config_snapshot_unreadable", self._codes(problems))

    # ---- 重复次数：重新推导，不信自报旗标 ----

    def test_sufficient_flag_is_not_trusted(self):
        """repeats.csv 自报 sufficient=true，但汇总行只有 2 个成员。

        这正是"派生文件过期"能造成的后果：旗标说够了，实际不够。
        判据必须从汇总行重新数出来。
        """
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), repeats=2, claim_sufficient=True)
            problems = self._check(staged)
            self.assertIn("insufficient_repeats", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_repeats_member_unknown_is_reported(self):
        """成员清单里列了汇总表里不存在的 run —— 派生文件滞后于原始明细。

        现象：repeats.csv 说这一组有 3 个成员，汇总表里只找得到 2 个。
        旧实现在这里报 ``repeats_count_mismatch``（自报 3、数到 2）；现在报
        ``repeats_member_unknown``（**点名**是哪几个陌生成员），并同时触发
        ``insufficient_repeats``。两种口径指同一件事，但后者能直接告诉修的人
        少了谁——而"少了谁"才是能动手修的信息。
        """
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp))  # 3 次重复
            # 抹掉一个成员目录 + 汇总行，模拟"派生文件没跟着更新"
            (staged["processed"] / "summary.csv").write_text(
                "run_id,data_label,workload,config\n"
                "20260101-000000-w-cfgA,MEASURED,w,cfgA\n"
                "20260101-000000-w-cfgA-r1,MEASURED,w,cfgA\n",
                encoding="utf-8",
            )
            problems = self._check(staged)
            codes = self._codes(problems)
            self.assertIn("repeats_member_unknown", codes)
            self.assertIn("insufficient_repeats", codes)
            unknown = next(p for p in problems if p.code == "repeats_member_unknown")
            self.assertIn("20260101-000000-w-cfgA-r2", unknown.detail)

    def test_repeats_count_mismatch_is_reported(self):
        """自报 n_repeats 与**同一行**的成员清单对不上 —— 派生文件自身不自洽。

        与 ``repeats_member_unknown`` 是两种不同的缺陷，必须分开报：
        那个是"清单里的 run 在汇总表里不存在"（与汇总表比），
        这个是"清单长度与自报次数矛盾"（与自身比）。合成一条会让人
        去改错的地方。
        """
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp))  # 清单 3 个成员、自报 3 次
            repeats_path = staged["processed"] / "repeats.csv"
            text = repeats_path.read_text(encoding="utf-8").replace(",3,", ",5,")
            repeats_path.write_text(text, encoding="utf-8")
            problems = self._check(staged)
            self.assertIn("repeats_count_mismatch", self._codes(problems))
            self.assertNotIn("repeats_member_unknown", self._codes(problems))

    # ---- 登记必须是表格行 ----

    def test_registration_in_prose_only_does_not_count(self):
        """只在正文里提到 run_id 不算登记——它给不出重复次数与离散度。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), register_as="prose")
            problems = self._check(staged)
            self.assertIn("experiment_not_registered", self._codes(problems))
            self.assertTrue(all(p.level == pf.BLOCK for p in problems))

    def test_registration_as_table_row_counts(self):
        """正向对照：登记成表格行的应当通过。"""
        with tempfile.TemporaryDirectory() as tmp:
            staged = stage(Path(tmp), register_as="table")
            self.assertEqual(self._check(staged), [])


if __name__ == "__main__":
    unittest.main()
